#define NOMINMAX
#include "renderer.h"
#include "microprofile.h"
#include <dxgi1_6.h>

#include <d3dcompiler.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <queue>
#include <sstream>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

struct CBPerFrame {
    float viewProj[16];
    float camPos[3];
    float mode;
    float lightDir[3];
    float ambient;
    float pointNormal[3];
    float _pad;
    float invViewProj[16];
    float screenW, screenH;
    float _pad2[2];
    float camRight[3];
    float _pad3;
    float camUp[3];
    float _pad4;
    float camForward[3];
    float tanHalfFovY;
    float fogColor[3];
    float fogDensity;            // 0 = depth fog off
    float heightFogDensity;      // 0 = height fog off
    float heightFogFalloff;
    float heightFogStart;
    float _padHF;
    float sceneOrigin[3];
    float nearZ;        // reverse-Z infinite-far: viewZ = nearZ / ndcZ
    float sceneSpan[3];
    float _pad6;
    float prevViewProj[16];
    float jitter[2];
    float _pad7[2];
    float sunViewProj[16];        // unused (kept for shader CB layout compatibility)
    float shadowBias;
    float shadowMapSize;
    float shadowEnable;           // always 0 now (shadows removed)
    float sunIntensity;
    float exposure;
    float roughness;
    float colorizeClusters;       // 0/1 tint on/off
    float gridSize;               // for shadow wrap (1 = no wrap)
};

struct CBPerChunk {
    float    chunkBase[3];
    float    _pad;          // lodHalfExtent (points)
    uint32_t voxelBase;     // PolyVID: chunk's first voxel index in pointSb_
    uint32_t chunkLodIdx;   // 0..3 splat LOD index
    uint32_t chunkTint;     // colorize-clusters debug
    uint32_t _pad2[2];
};

struct CBPerCS {
    float    viewProj[16];
    float    chunkBase[3];
    uint32_t voxelOffset;
    uint32_t voxelCount;
    uint32_t w;
    uint32_t h;
    float    lodHalfExtent;
};

static std::string ReadTextFile(const char* path)
{
    std::ifstream f(path);
    if (!f.good()) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool Renderer::Init(HWND hwnd, int adapterIdx)
{
    if (!CreateDeviceAndSwap(hwnd, adapterIdx)) return false;

    RECT rc; GetClientRect(hwnd, &rc);
    width_  = (uint32_t)(rc.right  - rc.left);
    height_ = (uint32_t)(rc.bottom - rc.top);

    if (!CreateRenderTargets()) return false;
    if (!CreateShaders())       return false;
    if (!CreatePipelineState()) return false;
    return true;
}

void Renderer::Shutdown()
{
    subs_.clear();
}

std::vector<std::string> Renderer::EnumerateAdapters()
{
    std::vector<std::string> out;
    ComPtr<IDXGIFactory6> f6;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(f6.GetAddressOf()));
    if (SUCCEEDED(hr)) {
        // Factory6 path: enumerate in performance-preferred order so idx 0 is
        // typically the discrete GPU on a laptop.
        for (UINT i = 0; ; ++i) {
            ComPtr<IDXGIAdapter1> a;
            if (f6->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                               IID_PPV_ARGS(a.GetAddressOf())) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 d{};
            a->GetDesc1(&d);
            if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            char buf[256];
            size_t cv = 0;
            wcstombs_s(&cv, buf, sizeof(buf), d.Description, _TRUNCATE);
            out.emplace_back(buf);
        }
        return out;
    }
    ComPtr<IDXGIFactory1> f1;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(f1.GetAddressOf())))) return out;
    for (UINT i = 0; ; ++i) {
        ComPtr<IDXGIAdapter1> a;
        if (f1->EnumAdapters1(i, a.GetAddressOf()) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 d{};
        a->GetDesc1(&d);
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        char buf[256];
        size_t cv = 0;
        wcstombs_s(&cv, buf, sizeof(buf), d.Description, _TRUNCATE);
        out.emplace_back(buf);
    }
    return out;
}

bool Renderer::CreateDeviceAndSwap(HWND hwnd, int adapterIdx)
{
    UINT flags = 0;
#ifdef _DEBUG
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    D3D_FEATURE_LEVEL flvl;
    D3D_FEATURE_LEVEL want[] = { D3D_FEATURE_LEVEL_11_0 };

    // Resolve adapter: explicit index uses DXGI enumeration matching
    // EnumerateAdapters() order (high-perf first on Factory6). idx < 0 = system default.
    ComPtr<IDXGIAdapter1> chosen;
    if (adapterIdx >= 0) {
        ComPtr<IDXGIFactory6> f6;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(f6.GetAddressOf())))) {
            int seen = 0;
            for (UINT i = 0; ; ++i) {
                ComPtr<IDXGIAdapter1> a;
                if (f6->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                   IID_PPV_ARGS(a.GetAddressOf())) == DXGI_ERROR_NOT_FOUND) break;
                DXGI_ADAPTER_DESC1 d{};
                a->GetDesc1(&d);
                if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
                if (seen == adapterIdx) { chosen = a; break; }
                ++seen;
            }
        }
        if (!chosen) {
            ComPtr<IDXGIFactory1> f1;
            if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(f1.GetAddressOf())))) {
                int seen = 0;
                for (UINT i = 0; ; ++i) {
                    ComPtr<IDXGIAdapter1> a;
                    if (f1->EnumAdapters1(i, a.GetAddressOf()) == DXGI_ERROR_NOT_FOUND) break;
                    DXGI_ADAPTER_DESC1 d{};
                    a->GetDesc1(&d);
                    if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
                    if (seen == adapterIdx) { chosen = a; break; }
                    ++seen;
                }
            }
        }
    }

    // Note: when chosen is non-null, driver type MUST be UNKNOWN.
    D3D_DRIVER_TYPE drvType = chosen ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE;
    IDXGIAdapter* adp = chosen.Get();

    HRESULT hr = D3D11CreateDevice(
        adp, drvType, nullptr, flags,
        want, _countof(want), D3D11_SDK_VERSION,
        device_.GetAddressOf(), &flvl, ctx_.GetAddressOf());
#ifdef _DEBUG
    if (FAILED(hr)) {
        flags &= ~D3D11_CREATE_DEVICE_DEBUG;
        hr = D3D11CreateDevice(
            adp, drvType, nullptr, flags,
            want, _countof(want), D3D11_SDK_VERSION,
            device_.GetAddressOf(), &flvl, ctx_.GetAddressOf());
    }
#endif
    if (FAILED(hr)) return false;

    ComPtr<IDXGIDevice> dxgiDev;
    device_.As(&dxgiDev);
    ComPtr<IDXGIAdapter> adapter;
    dxgiDev->GetAdapter(adapter.GetAddressOf());
    ComPtr<IDXGIFactory2> factory;
    adapter->GetParent(IID_PPV_ARGS(factory.GetAddressOf()));

    DXGI_SWAP_CHAIN_DESC1 sd = {};
    sd.Width = 0; sd.Height = 0;
    sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 2;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sd.Scaling = DXGI_SCALING_STRETCH;

    hr = factory->CreateSwapChainForHwnd(
        device_.Get(), hwnd, &sd, nullptr, nullptr, swap_.GetAddressOf());
    return SUCCEEDED(hr);
}

bool Renderer::CreateRenderTargets()
{
    rtv_.Reset();
    dsv_.Reset();
    depthTex_.Reset();
    csColorTex_.Reset();
    csDepthTex_.Reset();
    csColorUav_.Reset();
    csDepthUav_.Reset();
    csColorSrv_.Reset();
    splatColorTex_.Reset();
    splatColorRtv_.Reset();
    splatColorSrv_.Reset();
    splatDepthTex_.Reset();
    splatDsv_.Reset();
    splatFinalTex_.Reset();
    splatFinalUav_.Reset();
    splatFinalDepthTex_.Reset();
    splatFinalDepthUav_.Reset();
    splatFinalDepthSrv_.Reset();
    splatFinal2Tex_.Reset();
    splatFinal2Uav_.Reset();
    splatFinal2Srv_.Reset();
    splatFinal2DepthTex_.Reset();
    splatFinal2DepthUav_.Reset();
    splatFinal2DepthSrv_.Reset();
    splatMaskTex_.Reset();
    splatMaskRtv_.Reset();
    splatMaskSrv_.Reset();

    ComPtr<ID3D11Texture2D> backBuf;
    HRESULT hr = swap_->GetBuffer(0, IID_PPV_ARGS(backBuf.GetAddressOf()));
    if (FAILED(hr)) return false;
    hr = device_->CreateRenderTargetView(backBuf.Get(), nullptr, rtv_.GetAddressOf());
    if (FAILED(hr)) return false;

    D3D11_TEXTURE2D_DESC td = {};
    td.Width = width_;
    td.Height = height_;
    td.MipLevels = 1;
    td.ArraySize = 1;
    // R32_TYPELESS so TAA can sample depth as SRV.
    td.Format = DXGI_FORMAT_R32_TYPELESS;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
    hr = device_->CreateTexture2D(&td, nullptr, depthTex_.GetAddressOf());
    if (FAILED(hr)) return false;
    D3D11_DEPTH_STENCIL_VIEW_DESC dsvd = {};
    dsvd.Format = DXGI_FORMAT_D32_FLOAT;
    dsvd.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    hr = device_->CreateDepthStencilView(depthTex_.Get(), &dsvd, dsv_.GetAddressOf());
    if (FAILED(hr)) return false;
    depthSrv_.Reset();
    {
        D3D11_SHADER_RESOURCE_VIEW_DESC dsv = {};
        dsv.Format = DXGI_FORMAT_R32_FLOAT;
        dsv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        dsv.Texture2D.MipLevels = 1;
        device_->CreateShaderResourceView(depthTex_.Get(), &dsv, depthSrv_.GetAddressOf());
    }

    // TAA RT (scene) + ping-pong history (R8G8B8A8 with RT + SR binding).
    taaSceneTex_.Reset();  taaSceneRtv_.Reset();  taaSceneSrv_.Reset();
    for (int i = 0; i < 2; ++i) {
        taaHistTex_[i].Reset(); taaHistRtv_[i].Reset(); taaHistSrv_[i].Reset();
        taaHistValid_[i] = false;
    }
    {
        D3D11_TEXTURE2D_DESC ct = {};
        ct.Width = width_; ct.Height = height_;
        ct.MipLevels = 1; ct.ArraySize = 1;
        ct.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        ct.SampleDesc.Count = 1;
        ct.Usage = D3D11_USAGE_DEFAULT;
        ct.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        device_->CreateTexture2D(&ct, nullptr, taaSceneTex_.GetAddressOf());
        device_->CreateRenderTargetView(taaSceneTex_.Get(), nullptr, taaSceneRtv_.GetAddressOf());
        device_->CreateShaderResourceView(taaSceneTex_.Get(), nullptr, taaSceneSrv_.GetAddressOf());

        D3D11_TEXTURE2D_DESC ht = ct;
        ht.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        for (int i = 0; i < 2; ++i) {
            device_->CreateTexture2D(&ht, nullptr, taaHistTex_[i].GetAddressOf());
            device_->CreateRenderTargetView(taaHistTex_[i].Get(), nullptr, taaHistRtv_[i].GetAddressOf());
            device_->CreateShaderResourceView(taaHistTex_[i].Get(), nullptr, taaHistSrv_[i].GetAddressOf());
        }
    }

    // PointCS color/depth UAV textures.
    D3D11_TEXTURE2D_DESC ud = {};
    ud.Width = width_;
    ud.Height = height_;
    ud.MipLevels = 1;
    ud.ArraySize = 1;
    ud.Format = DXGI_FORMAT_R32_UINT;
    ud.SampleDesc.Count = 1;
    ud.Usage = D3D11_USAGE_DEFAULT;
    ud.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(device_->CreateTexture2D(&ud, nullptr, csColorTex_.GetAddressOf()))) return false;
    if (FAILED(device_->CreateUnorderedAccessView(csColorTex_.Get(), nullptr, csColorUav_.GetAddressOf()))) return false;
    if (FAILED(device_->CreateShaderResourceView(csColorTex_.Get(), nullptr, csColorSrv_.GetAddressOf()))) return false;

    ud.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    if (FAILED(device_->CreateTexture2D(&ud, nullptr, csDepthTex_.GetAddressOf()))) return false;
    if (FAILED(device_->CreateUnorderedAccessView(csDepthTex_.Get(), nullptr, csDepthUav_.GetAddressOf()))) return false;

    // Splat color RT + final UAV target.
    D3D11_TEXTURE2D_DESC sd2 = {};
    sd2.Width = width_;
    sd2.Height = height_;
    sd2.MipLevels = 1;
    sd2.ArraySize = 1;
    sd2.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd2.SampleDesc.Count = 1;
    sd2.Usage = D3D11_USAGE_DEFAULT;
    sd2.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(device_->CreateTexture2D(&sd2, nullptr, splatColorTex_.GetAddressOf()))) return false;
    if (FAILED(device_->CreateRenderTargetView(splatColorTex_.Get(), nullptr, splatColorRtv_.GetAddressOf()))) return false;
    if (FAILED(device_->CreateShaderResourceView(splatColorTex_.Get(), nullptr, splatColorSrv_.GetAddressOf()))) return false;

    sd2.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(device_->CreateTexture2D(&sd2, nullptr, splatFinalTex_.GetAddressOf()))) return false;
    if (FAILED(device_->CreateUnorderedAccessView(splatFinalTex_.Get(), nullptr, splatFinalUav_.GetAddressOf()))) return false;
    if (FAILED(device_->CreateShaderResourceView(splatFinalTex_.Get(), nullptr, splatFinalSrv_.GetAddressOf()))) return false;

    // Per-pixel visMask emitted by point PS (MRT slot 1), consumed by CS.
    D3D11_TEXTURE2D_DESC smd = sd2;
    smd.Format = DXGI_FORMAT_R8_UINT;
    smd.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(device_->CreateTexture2D(&smd, nullptr, splatMaskTex_.GetAddressOf()))) return false;
    if (FAILED(device_->CreateRenderTargetView(splatMaskTex_.Get(), nullptr, splatMaskRtv_.GetAddressOf()))) return false;
    if (FAILED(device_->CreateShaderResourceView(splatMaskTex_.Get(), nullptr, splatMaskSrv_.GetAddressOf()))) return false;

    // Dilated depth target.
    D3D11_TEXTURE2D_DESC sdf = sd2;
    sdf.Format = DXGI_FORMAT_R32_FLOAT;
    if (FAILED(device_->CreateTexture2D(&sdf, nullptr, splatFinalDepthTex_.GetAddressOf()))) return false;
    if (FAILED(device_->CreateUnorderedAccessView(splatFinalDepthTex_.Get(), nullptr, splatFinalDepthUav_.GetAddressOf()))) return false;
    if (FAILED(device_->CreateShaderResourceView(splatFinalDepthTex_.Get(), nullptr, splatFinalDepthSrv_.GetAddressOf()))) return false;

    // Second-pass dilate output (ping-pong target for csmain_splat_fill).
    if (FAILED(device_->CreateTexture2D(&sd2, nullptr, splatFinal2Tex_.GetAddressOf()))) return false;
    if (FAILED(device_->CreateUnorderedAccessView(splatFinal2Tex_.Get(), nullptr, splatFinal2Uav_.GetAddressOf()))) return false;
    if (FAILED(device_->CreateShaderResourceView(splatFinal2Tex_.Get(), nullptr, splatFinal2Srv_.GetAddressOf()))) return false;
    if (FAILED(device_->CreateTexture2D(&sdf, nullptr, splatFinal2DepthTex_.GetAddressOf()))) return false;
    if (FAILED(device_->CreateUnorderedAccessView(splatFinal2DepthTex_.Get(), nullptr, splatFinal2DepthUav_.GetAddressOf()))) return false;
    if (FAILED(device_->CreateShaderResourceView(splatFinal2DepthTex_.Get(), nullptr, splatFinal2DepthSrv_.GetAddressOf()))) return false;

    // Dedicated depth for splat pass.
    D3D11_TEXTURE2D_DESC sdd = {};
    sdd.Width = width_;
    sdd.Height = height_;
    sdd.MipLevels = 1;
    sdd.ArraySize = 1;
    sdd.Format = DXGI_FORMAT_R32_TYPELESS;
    sdd.SampleDesc.Count = 1;
    sdd.Usage = D3D11_USAGE_DEFAULT;
    sdd.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(device_->CreateTexture2D(&sdd, nullptr, splatDepthTex_.GetAddressOf()))) return false;
    D3D11_DEPTH_STENCIL_VIEW_DESC sddv = {};
    sddv.Format = DXGI_FORMAT_D32_FLOAT;
    sddv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    if (FAILED(device_->CreateDepthStencilView(splatDepthTex_.Get(), &sddv, splatDsv_.GetAddressOf()))) return false;
    D3D11_SHADER_RESOURCE_VIEW_DESC sdsv = {};
    sdsv.Format = DXGI_FORMAT_R32_FLOAT;
    sdsv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    sdsv.Texture2D.MipLevels = 1;
    if (FAILED(device_->CreateShaderResourceView(splatDepthTex_.Get(), &sdsv, splatDepthSrv_.GetAddressOf()))) return false;

    // GPU-memory accounting for the splat pass RTs/UAVs.
    auto bytesOf = [&](DXGI_FORMAT f) -> uint64_t {
        switch (f) {
            case DXGI_FORMAT_R8G8B8A8_UNORM: return 4ull;
            case DXGI_FORMAT_R32_FLOAT:
            case DXGI_FORMAT_R32_UINT:
            case DXGI_FORMAT_R32_TYPELESS:   return 4ull;
            case DXGI_FORMAT_R8_UINT:        return 1ull;
            default: return 4ull;
        }
    };
    uint64_t pix = (uint64_t)width_ * (uint64_t)height_;
    splatRtBytes_ = pix * (
          bytesOf(DXGI_FORMAT_R8G8B8A8_UNORM)   // splatColorTex_
        + bytesOf(DXGI_FORMAT_R8G8B8A8_UNORM)   // splatFinalTex_
        + bytesOf(DXGI_FORMAT_R8_UINT)          // splatMaskTex_
        + bytesOf(DXGI_FORMAT_R32_FLOAT)        // splatFinalDepthTex_
        + bytesOf(DXGI_FORMAT_R32_TYPELESS));   // splatDepthTex_
    return true;
}

bool Renderer::CreateShaders()
{
    std::string src = ReadTextFile("shaders/voxel.hlsl");
    if (src.empty()) {
        MessageBoxA(nullptr, "shaders/voxel.hlsl not found", "Renderer", MB_ICONERROR);
        return false;
    }
    UINT cflags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3;

    auto compile = [&](const char* entry, const char* target, ComPtr<ID3DBlob>& blob) -> bool {
        ComPtr<ID3DBlob> errs;
        HRESULT hr = D3DCompile(src.data(), src.size(), "voxel.hlsl", nullptr, nullptr,
                                entry, target, cflags, 0, blob.GetAddressOf(), errs.GetAddressOf());
        if (FAILED(hr)) {
            std::string msg = "Shader compile error [";
            msg += entry; msg += "]\n";
            if (errs) msg += std::string((const char*)errs->GetBufferPointer(), errs->GetBufferSize());
            OutputDebugStringA(msg.c_str());
            OutputDebugStringA("\n");
            MessageBoxA(nullptr, msg.c_str(), entry, MB_ICONERROR);
            return false;
        }
        return true;
    };

    ComPtr<ID3DBlob> vsbp, psbp, vsbSh;
    if (!compile("vsmain_points", "vs_5_0", vsbp)) return false;
    if (!compile("psmain_points", "ps_5_0", psbp)) return false;
    if (!compile("vsmain_shadow", "vs_5_0", vsbSh)) return false;
    ComPtr<ID3DBlob> psbps;
    if (!compile("psmain_points_simple", "ps_5_0", psbps)) return false;
    // HexSprite disabled — see comment block in shaders/voxel.hlsl.
    // ComPtr<ID3DBlob> vsbh, psbh;
    // if (!compile("vsmain_hex", "vs_5_0", vsbh)) return false;
    // if (!compile("psmain_hex", "ps_5_0", psbh)) return false;
    ComPtr<ID3DBlob> vsbv, psbv;
    if (!compile("vsmain_polyvid", "vs_5_0", vsbv)) return false;
    ComPtr<ID3DBlob> vsbAx;
    if (!compile("vsmain_polyaxis", "vs_5_0", vsbAx)) return false;
    ComPtr<ID3DBlob> vsbAxI;
    if (!compile("vsmain_polyaxis_instanced", "vs_5_0", vsbAxI)) return false;
    if (!compile("psmain_polyvid", "ps_5_0", psbv)) return false;
    ComPtr<ID3DBlob> vsbB, psbB, vsbBT;
    if (!compile("vsmain_billboard", "vs_5_0", vsbB)) return false;
    if (!compile("psmain_billboard", "ps_5_0", psbB)) return false;
    if (!compile("vsmain_billboard_tri", "vs_5_0", vsbBT)) return false;
    ComPtr<ID3DBlob> csbSp, psbSa, vsbTa, psbTa, psbPo, csbSpFill;
    if (!compile("csmain_splat",        "cs_5_0", csbSp)) return false;
    if (!compile("csmain_splat_fill",   "cs_5_0", csbSpFill)) return false;
    ComPtr<ID3DBlob> csbShBlur;
    if (!compile("csmain_shadow_blur",  "cs_5_0", csbShBlur)) return false;
    if (!compile("psmain_splat_albedo", "ps_5_0", psbSa)) return false;
    ComPtr<ID3DBlob> psbSc;
    if (!compile("psmain_splat_composite", "ps_5_0", psbSc)) return false;
    ComPtr<ID3DBlob> psbSr;
    if (!compile("psmain_splat_reconstruct", "ps_5_0", psbSr)) return false;
    if (!compile("vsmain_taa",          "vs_5_0", vsbTa)) return false;
    if (!compile("psmain_taa",          "ps_5_0", psbTa)) return false;
    if (!compile("psmain_post",         "ps_5_0", psbPo)) return false;
    ComPtr<ID3DBlob> csb, vsbl, psbl;
    if (!compile("csmain_points_cs", "cs_5_0", csb)) return false;
    if (!compile("vsmain_blit",      "vs_5_0", vsbl)) return false;
    if (!compile("psmain_blit",      "ps_5_0", psbl)) return false;

    HRESULT hr;
    hr = device_->CreateVertexShader(vsbSh->GetBufferPointer(), vsbSh->GetBufferSize(), nullptr, vsShadow_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreateVertexShader(vsbp->GetBufferPointer(), vsbp->GetBufferSize(), nullptr, vsPoints_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbp->GetBufferPointer(), psbp->GetBufferSize(), nullptr, psPoints_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbps->GetBufferPointer(), psbps->GetBufferSize(), nullptr, psPointsSimple_.GetAddressOf());
    if (FAILED(hr)) return false;
    // HexSprite disabled.
    // hr = device_->CreateVertexShader(vsbh->GetBufferPointer(), vsbh->GetBufferSize(), nullptr, vsHex_.GetAddressOf());
    // if (FAILED(hr)) return false;
    // hr = device_->CreatePixelShader(psbh->GetBufferPointer(), psbh->GetBufferSize(), nullptr, psHex_.GetAddressOf());
    // if (FAILED(hr)) return false;
    hr = device_->CreateVertexShader(vsbv->GetBufferPointer(), vsbv->GetBufferSize(), nullptr, vsPolyVid_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreateVertexShader(vsbAx->GetBufferPointer(), vsbAx->GetBufferSize(), nullptr, vsPolyAxis_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreateVertexShader(vsbAxI->GetBufferPointer(), vsbAxI->GetBufferSize(), nullptr, vsPolyAxisInstanced_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbv->GetBufferPointer(), psbv->GetBufferSize(), nullptr, psPolyVid_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreateVertexShader(vsbB->GetBufferPointer(), vsbB->GetBufferSize(), nullptr, vsBillboard_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbB->GetBufferPointer(), psbB->GetBufferSize(), nullptr, psBillboard_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreateVertexShader(vsbBT->GetBufferPointer(), vsbBT->GetBufferSize(), nullptr, vsBillboardTri_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreateComputeShader(csbSp->GetBufferPointer(), csbSp->GetBufferSize(), nullptr, csSplat_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreateComputeShader(csbSpFill->GetBufferPointer(), csbSpFill->GetBufferSize(), nullptr, csSplatFill_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreateComputeShader(csbShBlur->GetBufferPointer(), csbShBlur->GetBufferSize(), nullptr, csShadowBlur_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbSa->GetBufferPointer(), psbSa->GetBufferSize(), nullptr, psSplatAlbedo_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbSc->GetBufferPointer(), psbSc->GetBufferSize(), nullptr, psSplatComposite_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbSr->GetBufferPointer(), psbSr->GetBufferSize(), nullptr, psSplatRecon_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreateVertexShader(vsbTa->GetBufferPointer(), vsbTa->GetBufferSize(), nullptr, vsTaa_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbTa->GetBufferPointer(), psbTa->GetBufferSize(), nullptr, psTaa_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbPo->GetBufferPointer(), psbPo->GetBufferSize(), nullptr, psPost_.GetAddressOf());
    if (FAILED(hr)) return false;
    {
        D3D11_SAMPLER_DESC sm = {};
        sm.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sm.AddressU = sm.AddressV = sm.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sm.MinLOD = 0; sm.MaxLOD = D3D11_FLOAT32_MAX;
        device_->CreateSamplerState(&sm, linearClampSampler_.GetAddressOf());
    }
    hr = device_->CreateComputeShader(csb->GetBufferPointer(), csb->GetBufferSize(), nullptr, csPoints_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreateVertexShader(vsbl->GetBufferPointer(), vsbl->GetBufferSize(), nullptr, vsBlit_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbl->GetBufferPointer(), psbl->GetBufferSize(), nullptr, psBlit_.GetAddressOf());
    if (FAILED(hr)) return false;

    // Unified per-vertex layout: 16-bit-per-axis scene-relative position +
    // RGBA8 color (uint4, alpha byte = visMask for point verts). Stride 12 B.
    D3D11_INPUT_ELEMENT_DESC il[] = {
        { "POSITION", 0, DXGI_FORMAT_R16G16B16A16_UINT, 0, 0,                            D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR",    0, DXGI_FORMAT_R8G8B8A8_UINT,     0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    hr = device_->CreateInputLayout(il, _countof(il),
                                    vsbp->GetBufferPointer(), vsbp->GetBufferSize(),
                                    inputLayout_.GetAddressOf());
    if (FAILED(hr)) return false;

    // HexSprite disabled.
    // D3D11_INPUT_ELEMENT_DESC ilHex[] = {
    //     { "POSITION", 0, DXGI_FORMAT_R16G16B16A16_UINT, 0, 0,                            D3D11_INPUT_PER_INSTANCE_DATA, 1 },
    //     { "COLOR",    0, DXGI_FORMAT_R8G8B8A8_UINT,     0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
    // };
    // hr = device_->CreateInputLayout(ilHex, _countof(ilHex),
    //                                 vsbh->GetBufferPointer(), vsbh->GetBufferSize(),
    //                                 inputLayoutHex_.GetAddressOf());
    // if (FAILED(hr)) return false;

    // PolyVID input layout: per-instance step on pointVb_.
    D3D11_INPUT_ELEMENT_DESC ilV[] = {
        { "POSITION", 0, DXGI_FORMAT_R8G8B8A8_UINT,  0, 0, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
        { "COLOR",    0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
    };
    hr = device_->CreateInputLayout(ilV, _countof(ilV),
                                    vsbv->GetBufferPointer(), vsbv->GetBufferSize(),
                                    inputLayoutPolyVid_.GetAddressOf());
    if (FAILED(hr)) return false;

    return true;
}

bool Renderer::CreatePipelineState()
{
    D3D11_BUFFER_DESC bd = {};
    bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    bd.ByteWidth = (sizeof(CBPerFrame) + 15) & ~15;
    if (FAILED(device_->CreateBuffer(&bd, nullptr, cbPerFrame_.GetAddressOf()))) return false;
    bd.ByteWidth = (sizeof(CBPerChunk) + 15) & ~15;
    if (FAILED(device_->CreateBuffer(&bd, nullptr, cbPerChunk_.GetAddressOf()))) return false;
    bd.ByteWidth = (sizeof(CBPerCS) + 15) & ~15;
    if (FAILED(device_->CreateBuffer(&bd, nullptr, cbCS_.GetAddressOf()))) return false;

    D3D11_RASTERIZER_DESC rd = {};
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_BACK;
    rd.FrontCounterClockwise = FALSE;
    rd.DepthClipEnable = TRUE;
    if (FAILED(device_->CreateRasterizerState(&rd, rsSolid_.GetAddressOf()))) return false;

    D3D11_RASTERIZER_DESC rdNo = rd;
    rdNo.CullMode = D3D11_CULL_NONE;
    if (FAILED(device_->CreateRasterizerState(&rdNo, rsNoCull_.GetAddressOf()))) return false;

    // Shadow rasterizer: depth bias + slope-scaled bias to kill acne. Two
    // variants — toggle between cull-back (default) and cull-front from the UI.
    // Point splats are 1-pixel primitives so the cull mode has no effect on
    // them; the toggle exists for future polygon shadow casters.
    D3D11_RASTERIZER_DESC rdShB = rd;
    rdShB.CullMode = D3D11_CULL_BACK;
    rdShB.DepthBias = 4;
    rdShB.SlopeScaledDepthBias = 2.0f;
    rdShB.DepthClipEnable = TRUE;
    if (FAILED(device_->CreateRasterizerState(&rdShB, rsShadowBack_.GetAddressOf()))) return false;
    D3D11_RASTERIZER_DESC rdShF = rdShB;
    rdShF.CullMode = D3D11_CULL_FRONT;
    if (FAILED(device_->CreateRasterizerState(&rdShF, rsShadowFront_.GetAddressOf()))) return false;

    // Comparison sampler for hardware PCF.
    D3D11_SAMPLER_DESC ssd = {};
    ssd.Filter = D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
    ssd.AddressU = ssd.AddressV = ssd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    ssd.ComparisonFunc = D3D11_COMPARISON_GREATER_EQUAL;   // reverse-Z shadow
    ssd.MinLOD = 0; ssd.MaxLOD = D3D11_FLOAT32_MAX;
    if (FAILED(device_->CreateSamplerState(&ssd, shadowSamp_.GetAddressOf()))) return false;

    D3D11_DEPTH_STENCIL_DESC dd = {};
    dd.DepthEnable = TRUE;
    dd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    dd.DepthFunc = D3D11_COMPARISON_GREATER;
    if (FAILED(device_->CreateDepthStencilState(&dd, dsTest_.GetAddressOf()))) return false;

    D3D11_DEPTH_STENCIL_DESC ddNo = {};
    ddNo.DepthEnable = FALSE;
    ddNo.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
    ddNo.DepthFunc = D3D11_COMPARISON_ALWAYS;
    if (FAILED(device_->CreateDepthStencilState(&ddNo, dsAlways_.GetAddressOf()))) return false;

    // Alpha-over: src=ALPHA, dst=INV_ALPHA. Alpha=0 -> destination preserved.
    D3D11_BLEND_DESC bsa = {};
    bsa.RenderTarget[0].BlendEnable = TRUE;
    bsa.RenderTarget[0].SrcBlend  = D3D11_BLEND_SRC_ALPHA;
    bsa.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    bsa.RenderTarget[0].BlendOp   = D3D11_BLEND_OP_ADD;
    bsa.RenderTarget[0].SrcBlendAlpha  = D3D11_BLEND_ONE;
    bsa.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    bsa.RenderTarget[0].BlendOpAlpha   = D3D11_BLEND_OP_ADD;
    bsa.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    if (FAILED(device_->CreateBlendState(&bsa, bsAlphaOver_.GetAddressOf()))) return false;

    // Splat depth states.
    {
        D3D11_DEPTH_STENCIL_DESC dde = {};
        dde.DepthEnable = TRUE;
        dde.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
        dde.DepthFunc = D3D11_COMPARISON_GREATER;
        dde.StencilEnable = FALSE;
        if (FAILED(device_->CreateDepthStencilState(&dde, dsSplatPoint_.GetAddressOf()))) return false;
    }
    {
        D3D11_DEPTH_STENCIL_DESC dde = {};
        dde.DepthEnable = TRUE;
        dde.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
        dde.DepthFunc = D3D11_COMPARISON_ALWAYS;
        dde.StencilEnable = FALSE;
        if (FAILED(device_->CreateDepthStencilState(&dde, dsAlwaysWrite_.GetAddressOf()))) return false;
    }
    return true;
}

void Renderer::Resize(uint32_t w, uint32_t h)
{
    if (!swap_ || (w == width_ && h == height_) || w == 0 || h == 0) return;
    ctx_->OMSetRenderTargets(0, nullptr, nullptr);
    rtv_.Reset();
    dsv_.Reset();
    depthTex_.Reset();
    swap_->ResizeBuffers(0, w, h, DXGI_FORMAT_UNKNOWN, 0);
    width_ = w; height_ = h;
    CreateRenderTargets();
}

void Renderer::UploadScene(const Scene& scene)
{
    subs_.clear();
    pointVb_.Reset();
    pointSb_.Reset();
    pointSrv_.Reset();
    pointAo6Srv_.Reset();
    pointAo6Sb_.Reset();
    polyVidIb_.Reset();
    billboardIb_.Reset();
    pointBytes_ = 0;

    if (scene.pointVertices.empty()) return;

    // Single structured-SRV buffer. All surviving point techs (Splat, Points,
    // PolyVID, PolyAxis, PolyAxisInst, Billboard, BillboardTri, PointCS) read
    // voxel data via SV_VertexID + gVoxelBase from this SRV — no VB binding.
    // (HexSprite previously needed a separate VB; it's disabled now.)
    const uint64_t pvBytes64 = (uint64_t)scene.pointVertices.size() * sizeof(Vertex);
    if (pvBytes64 > 0xFFFFFFFFull) {
        std::fprintf(stderr,
            "[upload] FATAL: pointVertices = %llu bytes > 4 GB. D3D11 buffer max "
            "is UINT bytes. Regenerate the .vox with a coarser --scale (try 4-6 m).\n",
            (unsigned long long)pvBytes64);
        return;
    }

    D3D11_BUFFER_DESC sbd = {};
    sbd.Usage = D3D11_USAGE_IMMUTABLE;
    sbd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    sbd.ByteWidth = (UINT)pvBytes64;
    sbd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    sbd.StructureByteStride = sizeof(Vertex);
    D3D11_SUBRESOURCE_DATA psd = {};
    psd.pSysMem = scene.pointVertices.data();
    if (FAILED(device_->CreateBuffer(&sbd, &psd, pointSb_.GetAddressOf()))) {
        std::fprintf(stderr, "[upload] FATAL: CreateBuffer pointSb_ (%llu bytes) failed.\n",
                     (unsigned long long)pvBytes64);
        pointSb_.Reset();
        return;
    }
    D3D11_SHADER_RESOURCE_VIEW_DESC srvd = {};
    srvd.Format = DXGI_FORMAT_UNKNOWN;
    srvd.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    srvd.Buffer.NumElements = (UINT)scene.pointVertices.size();
    device_->CreateShaderResourceView(pointSb_.Get(), &srvd, pointSrv_.GetAddressOf());

    // Parallel per-voxel face-AO buffer for PolyAxis (24 bits packed).
    if (!scene.pointAo6.empty()) {
        D3D11_BUFFER_DESC abd = {};
        abd.Usage = D3D11_USAGE_IMMUTABLE;
        abd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        abd.ByteWidth = (UINT)(scene.pointAo6.size() * sizeof(uint32_t));
        abd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        abd.StructureByteStride = sizeof(uint32_t);
        D3D11_SUBRESOURCE_DATA asd = {};
        asd.pSysMem = scene.pointAo6.data();
        ComPtr<ID3D11Buffer> aoSb;
        if (SUCCEEDED(device_->CreateBuffer(&abd, &asd, aoSb.GetAddressOf()))) {
            D3D11_SHADER_RESOURCE_VIEW_DESC asd2 = {};
            asd2.Format = DXGI_FORMAT_UNKNOWN;
            asd2.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
            asd2.Buffer.NumElements = (UINT)scene.pointAo6.size();
            device_->CreateShaderResourceView(aoSb.Get(), &asd2, pointAo6Srv_.GetAddressOf());
            pointAo6Sb_ = aoSb;
        }
    }

    subs_.clear();
    subs_.reserve(scene.subs.size());
    // Force shadow map rebuild next frame (new geometry).
    shadowMapDirty_ = true;
    for (const auto& s : scene.subs) {
        GpuSubMesh gs;
        gs.pointFirst = s.pointFirst;
        gs.pointCount = s.pointCount;
        gs.pointFirstL1 = s.pointFirstL1;
        gs.pointCountL1 = s.pointCountL1;
        gs.pointFirstL2 = s.pointFirstL2;
        gs.pointCountL2 = s.pointCountL2;
        gs.pointFirstL3 = s.pointFirstL3;
        gs.pointCountL3 = s.pointCountL3;
        gs.aabbMin[0] = s.aabbMin[0]; gs.aabbMin[1] = s.aabbMin[1]; gs.aabbMin[2] = s.aabbMin[2];
        gs.aabbMax[0] = s.aabbMax[0]; gs.aabbMax[1] = s.aabbMax[1]; gs.aabbMax[2] = s.aabbMax[2];
        gs.chunkBase[0] = s.chunkBase[0]; gs.chunkBase[1] = s.chunkBase[1]; gs.chunkBase[2] = s.chunkBase[2];
        subs_.push_back(gs);
    }
    sceneOrigin_[0] = (float)scene.origin[0];
    sceneOrigin_[1] = (float)scene.origin[1];
    sceneOrigin_[2] = (float)scene.origin[2];
    sceneSpan_[0] = scene.aabbMax[0] - scene.aabbMin[0];
    sceneSpan_[1] = scene.aabbMax[1] - scene.aabbMin[1];
    sceneSpan_[2] = scene.aabbMax[2] - scene.aabbMin[2];

    // PolyVID + Billboard shared IBs sized to the largest chunk.
    {
        uint32_t maxVox = 0;
        for (const auto& s : scene.subs) {
            uint32_t c = std::max(s.pointCount, std::max(s.pointCountL1, s.pointCountL2));
            maxVox = std::max(maxVox, c);
        }
        if (maxVox > 0) {
            std::vector<uint32_t> ib((size_t)maxVox * 36);
            uint32_t* dst = ib.data();
            static const uint8_t kQuadIdx[6] = { 0, 1, 2, 0, 2, 3 };
            for (uint32_t v = 0; v < maxVox; ++v) {
                uint32_t vBase = v * 24;
                for (uint32_t f = 0; f < 6; ++f) {
                    uint32_t fBase = vBase + f * 4;
                    for (int k = 0; k < 6; ++k) *dst++ = fBase + kQuadIdx[k];
                }
            }
            D3D11_BUFFER_DESC ibd = {};
            ibd.Usage = D3D11_USAGE_IMMUTABLE;
            ibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
            ibd.ByteWidth = (UINT)(ib.size() * sizeof(uint32_t));
            D3D11_SUBRESOURCE_DATA isd = { ib.data(), 0, 0 };
            device_->CreateBuffer(&ibd, &isd, polyVidIb_.GetAddressOf());

            // Billboard IB: 6 indices per voxel, pattern v*4 + {0,1,2,0,2,3}.
            std::vector<uint32_t> bib((size_t)maxVox * 6);
            uint32_t* bdst = bib.data();
            for (uint32_t v = 0; v < maxVox; ++v) {
                uint32_t base = v * 4;
                for (int k = 0; k < 6; ++k) *bdst++ = base + kQuadIdx[k];
            }
            D3D11_BUFFER_DESC ibd2 = {};
            ibd2.Usage = D3D11_USAGE_IMMUTABLE;
            ibd2.BindFlags = D3D11_BIND_INDEX_BUFFER;
            ibd2.ByteWidth = (UINT)(bib.size() * sizeof(uint32_t));
            D3D11_SUBRESOURCE_DATA isd2 = { bib.data(), 0, 0 };
            device_->CreateBuffer(&ibd2, &isd2, billboardIb_.GetAddressOf());
        }
    }
    pointBytes_ = scene.pointVertices.size() * sizeof(Vertex)
                + scene.pointAo6.size() * sizeof(uint32_t);
    pointCountL0_ = 0;
    pointCountL1_ = 0;
    pointCountL2_ = 0;
    pointCountL3_ = 0;
    for (const auto& s : scene.subs) {
        pointCountL0_ += s.pointCount;
        pointCountL1_ += s.pointCountL1;
        pointCountL2_ += s.pointCountL2;
        pointCountL3_ += s.pointCountL3;
    }
    colorHistogram_ = scene.colorHistogram;
    compRawBytes_         = scene.compRawBytes;
    compPosBytes_         = scene.compPosBytes;
    compMaskBytes_        = scene.compMaskBytes;
    compAoBytes_          = scene.compAoBytes;
    compPaletteBytes_     = scene.compPaletteBytes;
    compColorPalIdxBytes_ = scene.compColorPalIdxBytes;
    compColorHuffBytes_   = scene.compColorHuffBytes;
    compPosBitsPerAxis_   = scene.compPosBitsPerAxis;
    compChunkDim_         = scene.compChunkDim;
    compSubclusterPosBytes_ = scene.compSubclusterPosBytes;
    compSubclusterDim_      = scene.compSubclusterDim;
    compLz4PosBytes_        = scene.compLz4PosBytes;
    compLz4MaskBytes_       = scene.compLz4MaskBytes;
    compLz4AoBytes_         = scene.compLz4AoBytes;
    compLz4ColorPalBytes_   = scene.compLz4ColorPalBytes;
    compLz4TotalBytes_      = scene.compLz4TotalBytes;

    // Color entropy / Huffman avg bits per symbol.
    {
        colorEntropyBits_ = 0.0;
        colorHuffmanBits_ = 0.0;
        colorPaletteBits_ = 0;
        uint64_t total = 0;
        for (const auto& p : colorHistogram_) total += p.second;
        if (total > 0 && !colorHistogram_.empty()) {
            const double invT = 1.0 / (double)total;
            for (const auto& p : colorHistogram_) {
                const double pr = (double)p.second * invT;
                if (pr > 0.0) colorEntropyBits_ -= pr * std::log2(pr);
            }
            const size_t n = colorHistogram_.size();
            colorPaletteBits_ = (n <= 1) ? 1u
                : (uint32_t)std::ceil(std::log2((double)n));

            if (n == 1) {
                colorHuffmanBits_ = 1.0;
            } else {
                std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<uint64_t>> pq;
                for (const auto& p : colorHistogram_) pq.push(p.second);
                uint64_t totalCodeLen = 0;
                while (pq.size() > 1) {
                    uint64_t a = pq.top(); pq.pop();
                    uint64_t b = pq.top(); pq.pop();
                    uint64_t s = a + b;
                    totalCodeLen += s;
                    pq.push(s);
                }
                colorHuffmanBits_ = (double)totalCodeLen / (double)total;
            }
        }
    }
}

void Renderer::TryHotReloadShaders()
{
    HANDLE h = CreateFileA("shaders/voxel.hlsl", GENERIC_READ, FILE_SHARE_READ,
                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return;
    FILETIME ft = {};
    GetFileTime(h, nullptr, nullptr, &ft);
    CloseHandle(h);
    uint64_t mtime = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    if (mtime == shaderMtime_) return;
    if (shaderMtime_ == 0) {
        shaderMtime_ = mtime;
        return;
    }
    OutputDebugStringA("[HotReload] voxel.hlsl changed - recompiling\n");
    if (CreateShaders()) {
        OutputDebugStringA("[HotReload] success\n");
    } else {
        OutputDebugStringA("[HotReload] FAILED - check error dialog\n");
    }
    shaderMtime_ = mtime;
}

void Renderer::BeginFrame(float clear[4])
{
    TryHotReloadShaders();
    lastClear_[0] = clear[0];
    lastClear_[1] = clear[1];
    lastClear_[2] = clear[2];
    lastClear_[3] = clear[3];
    ID3D11RenderTargetView* rtvs[] = { rtv_.Get() };
    ctx_->OMSetRenderTargets(1, rtvs, dsv_.Get());
    ctx_->ClearRenderTargetView(rtv_.Get(), clear);
    ctx_->ClearDepthStencilView(dsv_.Get(), D3D11_CLEAR_DEPTH, 0.0f, 0);

    D3D11_VIEWPORT vp = {};
    vp.Width  = (float)width_;
    vp.Height = (float)height_;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    ctx_->RSSetViewports(1, &vp);
}

void Renderer::DrawScene(const Camera& cam, const DrawSceneParams& args)
{
    MICROPROFILE_SCOPEI("CPU", "DrawScene", 0xff80c0ff);
    MICROPROFILE_SCOPEGPUI("DrawScene", 0xff80c0ff);
    ShadingMode    mode             = args.mode;
    int            gridSize         = args.gridSize;
    RenderTech     techClose        = args.tech;
    RenderTech     techFar          = args.techFar;
    bool           closeEnabled     = args.closeEnabled;
    bool           farEnabled       = args.farEnabled;
    PointLighting  pointLight       = args.pointLight;
    PointLod       pointLod         = args.pointLod;
    float          pointLodScale    = args.pointLodScale;
    bool           splatFilter      = args.splatFilter;
    const float*   fogColor         = args.fogColor;
    float          fogDensity       = args.fogDensity;
    float          heightFogDensity = args.heightFogDensity;
    float          heightFogFalloff = args.heightFogFalloff;
    float          heightFogStart   = args.heightFogStart;
    int            splatRadius      = args.splatRadius;
    bool           taa              = args.taa;
    float          sunIntensity     = args.sunIntensity;
    float          exposure         = args.exposure;
    float          roughness        = args.roughness;

    // Halton(2,3) sub-pixel jitter for this frame.
    auto halton = [](uint32_t i, uint32_t b) {
        float f = 1.0f, r = 0.0f;
        while (i > 0) { f /= (float)b; r += f * (float)(i % b); i /= b; }
        return r;
    };
    float jitterNdcX = 0.0f, jitterNdcY = 0.0f;
    if (taa) {
        uint32_t k = (taaFrame_ % 16u) + 1u;
        jitterNdcX = (halton(k, 2) - 0.5f) * 2.0f / (float)width_;
        jitterNdcY = (halton(k, 3) - 0.5f) * 2.0f / (float)height_;
    }
    pointLodScale = std::max(pointLodScale, 0.01f);
    gridSize = std::clamp(gridSize, 1, 10);
    hlslpp::float4x4 v  = cam.view();
    hlslpp::float4x4 p  = cam.proj((float)width_ / (float)(height_ ? height_ : 1));
    hlslpp::float4x4 vpUnjittered = hlslpp::mul(v, p);
    if (taa) {
        float pStore[16];
        hlslpp::store(pStore, p);
        pStore[2*4 + 0] = jitterNdcX;
        pStore[2*4 + 1] = jitterNdcY;
        p = hlslpp::float4x4(
            pStore[0],  pStore[1],  pStore[2],  pStore[3],
            pStore[4],  pStore[5],  pStore[6],  pStore[7],
            pStore[8],  pStore[9],  pStore[10], pStore[11],
            pStore[12], pStore[13], pStore[14], pStore[15]);
    }
    hlslpp::float4x4 vp = hlslpp::mul(v, p);

    D3D11_MAPPED_SUBRESOURCE m;
    ctx_->Map(cbPerFrame_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m);
    const bool postEnabled = (taaSceneRtv_ && psPost_);
    postWroteBackbuf_ = false;
    if (postEnabled) {
        float alpha0[4] = { lastClear_[0], lastClear_[1], lastClear_[2], 0.0f };
        ctx_->ClearRenderTargetView(taaSceneRtv_.Get(), alpha0);
        ID3D11RenderTargetView* rtvs[] = { taaSceneRtv_.Get() };
        ctx_->OMSetRenderTargets(1, rtvs, dsv_.Get());
    }

    {
        CBPerFrame cb;
        hlslpp::store(cb.viewProj, vp);
        hlslpp::store(cb.camPos, cam.position);
        cb.mode = (float)(int)mode;
        cb.lightDir[0] = args.sunDir[0]; cb.lightDir[1] = args.sunDir[1]; cb.lightDir[2] = args.sunDir[2];
        cb.ambient = 0.7f;
        hlslpp::float3 fwd = cam.forward();
        hlslpp::float3 pn  = -fwd;
        hlslpp::store(cb.pointNormal, pn);
        cb._pad = 0.0f;
        hlslpp::float4x4 invVp = hlslpp::inverse(vp);
        hlslpp::store(cb.invViewProj, invVp);
        cb.screenW = (float)width_;
        cb.screenH = (float)height_;
        cb._pad2[0] = (float)splatRadius;
        cb._pad2[1] = 0;
        hlslpp::float3 right = cam.right();
        hlslpp::float3 upVec = hlslpp::cross(fwd, right);
        hlslpp::store(cb.camRight,   right);
        hlslpp::store(cb.camUp,      upVec);
        hlslpp::store(cb.camForward, fwd);
        cb._pad3 = cb._pad4 = 0;
        cb.tanHalfFovY = tanf(cam.fovDeg * 3.14159265358979f / 180.0f * 0.5f);
        cb.fogColor[0] = fogColor[0]; cb.fogColor[1] = fogColor[1]; cb.fogColor[2] = fogColor[2];
        cb.fogDensity  = fogDensity;
        cb.heightFogDensity = heightFogDensity;
        cb.heightFogFalloff = heightFogFalloff;
        cb.heightFogStart   = heightFogStart;
        cb._padHF = 0;
        cb.sceneOrigin[0] = sceneOrigin_[0]; cb.sceneOrigin[1] = sceneOrigin_[1]; cb.sceneOrigin[2] = sceneOrigin_[2];
        cb.nearZ = cam.nearZ;
        cb.sceneSpan[0] = sceneSpan_[0]; cb.sceneSpan[1] = sceneSpan_[1]; cb.sceneSpan[2] = sceneSpan_[2];
        cb._pad6 = 0;
        for (int i = 0; i < 16; ++i) cb.prevViewProj[i] = taaPrevVP_[i];
        cb.jitter[0] = jitterNdcX; cb.jitter[1] = jitterNdcY;
        cb._pad7[0] = cb._pad7[1] = 0;
        // Sun shadow setup (cascade 0 only). Build ortho fit to scene AABB.
        hlslpp::float4x4 sunVP{};
        float sunVPstore[16] = {};
        bool sunShadowsOn = args.sunShadows && pointSb_ && !subs_.empty();
        if (sunShadowsOn) {
            const uint32_t reqSize = (uint32_t)std::max(64, args.shadowMapSize);
            if (shadowSize_ != reqSize) {
                shadowTex_.Reset(); shadowDsv_.Reset(); shadowSrv_.Reset();
                shadowFilledTex_.Reset(); shadowFilledUav_.Reset(); shadowFilledSrv_.Reset();
                D3D11_TEXTURE2D_DESC td = {};
                td.Width = reqSize; td.Height = reqSize;
                td.MipLevels = 1; td.ArraySize = 1;
                td.Format = DXGI_FORMAT_R32_TYPELESS;
                td.SampleDesc.Count = 1;
                td.Usage = D3D11_USAGE_DEFAULT;
                td.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
                if (FAILED(device_->CreateTexture2D(&td, nullptr, shadowTex_.GetAddressOf()))) sunShadowsOn = false;
                if (sunShadowsOn) {
                    D3D11_DEPTH_STENCIL_VIEW_DESC dvd = {};
                    dvd.Format = DXGI_FORMAT_D32_FLOAT;
                    dvd.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
                    device_->CreateDepthStencilView(shadowTex_.Get(), &dvd, shadowDsv_.GetAddressOf());
                    D3D11_SHADER_RESOURCE_VIEW_DESC svd = {};
                    svd.Format = DXGI_FORMAT_R32_FLOAT;
                    svd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
                    svd.Texture2D.MipLevels = 1;
                    device_->CreateShaderResourceView(shadowTex_.Get(), &svd, shadowSrv_.GetAddressOf());
                    // Filled copy (CS hole-fill target). R32_FLOAT, UAV+SRV.
                    D3D11_TEXTURE2D_DESC tdf = {};
                    tdf.Width = reqSize; tdf.Height = reqSize;
                    tdf.MipLevels = 1; tdf.ArraySize = 1;
                    tdf.Format = DXGI_FORMAT_R32_FLOAT;
                    tdf.SampleDesc.Count = 1;
                    tdf.Usage = D3D11_USAGE_DEFAULT;
                    tdf.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
                    device_->CreateTexture2D(&tdf, nullptr, shadowFilledTex_.GetAddressOf());
                    device_->CreateUnorderedAccessView(shadowFilledTex_.Get(), nullptr, shadowFilledUav_.GetAddressOf());
                    device_->CreateShaderResourceView(shadowFilledTex_.Get(), nullptr, shadowFilledSrv_.GetAddressOf());
                    shadowSize_ = reqSize;
                }
            }
        }
        if (sunShadowsOn) {
            hlslpp::float3 sceneMin  (sceneOrigin_[0], sceneOrigin_[1], sceneOrigin_[2]);
            hlslpp::float3 sceneSpan3(sceneSpan_[0],   sceneSpan_[1],   sceneSpan_[2]);
            hlslpp::float3 center = sceneMin + sceneSpan3 * 0.5f;
            float radius = 0.5f * (float)hlslpp::length(sceneSpan3);
            if (radius < 1.0f) radius = 1.0f;
            hlslpp::float3 sd = hlslpp::normalize(
                hlslpp::float3(args.sunDir[0], args.sunDir[1], args.sunDir[2]));
            hlslpp::float3 eye = center + sd * (radius * 2.0f);
            float sy = (float)sd.y;
            hlslpp::float3 upRef = (std::fabs(sy) > 0.99f)
                ? hlslpp::float3(0.0f, 0.0f, 1.0f)
                : hlslpp::float3(0.0f, 1.0f, 0.0f);
            hlslpp::float4x4 sunView = hlslpp::float4x4::look_at(eye, center, upRef);
            const float wO = radius * 2.5f, hO = radius * 2.5f;
            const float zn = 0.0f, zf = radius * 4.0f;
            // LH reverse-Z ortho, row-major (vector * matrix).
            hlslpp::float4x4 sunProj(
                2.0f / wO, 0,          0,                0,
                0,          2.0f / hO, 0,                0,
                0,          0,          -1.0f / (zf - zn), 0,
                0,          0,          zf / (zf - zn),    1);
            sunVP = hlslpp::mul(sunView, sunProj);
            hlslpp::store(sunVPstore, sunVP);
        }
        for (int i = 0; i < 16; ++i) cb.sunViewProj[i] = sunShadowsOn ? sunVPstore[i] : 0.0f;
        for (int i = 0; i < 16; ++i) shadowVP_[i] = sunVPstore[i];
        cb.shadowBias    = args.shadowBias;
        cb.shadowMapSize = (float)shadowSize_;
        cb.shadowEnable  = sunShadowsOn ? 1.0f : 0.0f;
        shadowBias_   = args.shadowBias;
        shadowEnable_ = cb.shadowEnable;
        cb.sunIntensity  = sunIntensity;
        cb.exposure      = exposure;
        cb.roughness     = roughness;
        // LodViz mode uses ClusterTint downstream; field repurposed as a flag.
        cb.colorizeClusters = (args.mode == ShadingMode::LodViz) ? 1.0f : 0.0f;
        cb.gridSize = (float)std::max(1, gridSize);
        memcpy(m.pData, &cb, sizeof(cb));
    }
    ctx_->Unmap(cbPerFrame_.Get(), 0);

    ctx_->IASetInputLayout(inputLayout_.Get());
    ID3D11Buffer* cbs[] = { cbPerFrame_.Get(), cbPerChunk_.Get() };
    ctx_->VSSetConstantBuffers(0, 2, cbs);
    ctx_->PSSetConstantBuffers(0, 2, cbs);
    ctx_->RSSetState(rsSolid_.Get());
    ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);

    if (!pointSb_) return;

    // ---- Sun shadow caster pass (cascade 0) ----
    // Skip if nothing relevant changed since the last bake.
    if (shadowEnable_ > 0.5f && shadowDsv_ && pointSb_) {
        const float epsDir = 1e-4f;
        bool dirChanged =
              std::fabs(args.sunDir[0] - lastSunDir_[0]) > epsDir
           || std::fabs(args.sunDir[1] - lastSunDir_[1]) > epsDir
           || std::fabs(args.sunDir[2] - lastSunDir_[2]) > epsDir;
        bool sizeChanged    = (shadowSize_ != lastShadowSize_);
        bool cullChanged    = (args.shadowCullFront != lastShadowCullFront_);
        bool lodChanged     = (args.shadowLod != lastShadowLod_);
        bool blurChanged    = (args.shadowBlur != lastShadowBlur_);
        if (!args.shadowForceRebuild && !shadowMapDirty_ && !dirChanged && !sizeChanged && !cullChanged && !lodChanged && !blurChanged) {
            // Reuse cached shadow tex from previous frame. Still bind SRV +
            // sampler for downstream lit passes. Pick blurred copy if active.
            ID3D11RenderTargetView* mainRtv = postEnabled ? taaSceneRtv_.Get() : rtv_.Get();
            ID3D11RenderTargetView* mainRtvs[] = { mainRtv };
            ctx_->OMSetRenderTargets(1, mainRtvs, dsv_.Get());
            ID3D11ShaderResourceView* shSrv[] = {
                (lastShadowBlur_ && shadowFilledSrv_) ? shadowFilledSrv_.Get() : shadowSrv_.Get()
            };
            ctx_->PSSetShaderResources(6, 1, shSrv);
            ID3D11SamplerState* shSm[] = { shadowSamp_.Get() };
            ctx_->PSSetSamplers(1, 1, shSm);
            goto shadowDone;
        }
        MICROPROFILE_SCOPEGPUI("Shadow", 0xff909090);
        ID3D11RenderTargetView* nullRtv[] = { nullptr };
        ctx_->OMSetRenderTargets(1, nullRtv, shadowDsv_.Get());
        ctx_->ClearDepthStencilView(shadowDsv_.Get(), D3D11_CLEAR_DEPTH, 0.0f, 0);
        D3D11_VIEWPORT svp = { 0, 0, (float)shadowSize_, (float)shadowSize_, 0.0f, 1.0f };
        ctx_->RSSetViewports(1, &svp);
        ctx_->RSSetState(args.shadowCullFront ? rsShadowFront_.Get() : rsShadowBack_.Get());
        ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
        ctx_->IASetInputLayout(nullptr);
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
        ID3D11Buffer* nullVbs[] = { nullptr };
        UINT zs = 0;
        ctx_->IASetVertexBuffers(0, 1, nullVbs, &zs, &zs);
        ctx_->VSSetShader(vsShadow_.Get(), nullptr, 0);
        ctx_->PSSetShader(nullptr, nullptr, 0);
        ID3D11ShaderResourceView* vsSrvs[] = { nullptr, pointSrv_.Get() };
        ctx_->VSSetShaderResources(0, 2, vsSrvs);

        // Auto-LOD: pick the coarsest LOD whose voxel size is still smaller
        // than a shadow-map pixel. Eliminates per-pixel ROP contention from
        // 100s of L0 voxels collapsing onto the same shadow texel.
        int castLod = args.shadowLod;
        if (castLod < 0) {
            const float orthoWorldW = std::max(sceneSpan_[0], sceneSpan_[2]) * 1.25f;
            const float pxWorld = orthoWorldW / std::max(1.0f, (float)shadowSize_);
            if      (pxWorld >= 8.0f) castLod = 3;
            else if (pxWorld >= 4.0f) castLod = 2;
            else if (pxWorld >= 2.0f) castLod = 1;
            else                      castLod = 0;
        }
        if (castLod < 0) castLod = 0; if (castLod > 3) castLod = 3;
        const float halfExtArr[4] = { 0.5f, 1.0f, 2.0f, 4.0f };
        float halfExtCast = halfExtArr[castLod];

        // Render each chunk at the picked LOD; fall back to a finer LOD if the
        // chunk has zero voxels at that level.
        for (const auto& gs : subs_) {
            uint32_t pf = 0, pc = 0;
            int useLod = castLod;
            while (useLod >= 0) {
                if      (useLod == 0) { pf = gs.pointFirst;   pc = gs.pointCount;   }
                else if (useLod == 1) { pf = gs.pointFirstL1; pc = gs.pointCountL1; }
                else if (useLod == 2) { pf = gs.pointFirstL2; pc = gs.pointCountL2; }
                else                  { pf = gs.pointFirstL3; pc = gs.pointCountL3; }
                if (pc > 0) break;
                --useLod;
            }
            if (pc == 0) continue;
            float halfExtUse = halfExtArr[useLod < 0 ? 0 : useLod];
            D3D11_MAPPED_SUBRESOURCE mm;
            ctx_->Map(cbPerChunk_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mm);
            CBPerChunk cc = {};
            cc.chunkBase[0] = sceneOrigin_[0];
            cc.chunkBase[1] = sceneOrigin_[1];
            cc.chunkBase[2] = sceneOrigin_[2];
            cc._pad        = halfExtUse;
            cc.voxelBase   = pf;
            cc.chunkLodIdx = (uint32_t)(useLod < 0 ? 0 : useLod);
            cc.chunkTint   = 0;
            memcpy(mm.pData, &cc, sizeof(cc));
            ctx_->Unmap(cbPerChunk_.Get(), 0);
            ctx_->Draw(pc, 0);
        }
        (void)halfExtCast;
        // Detach DSV (the shadow tex needs to be SRV-bindable).
        ID3D11RenderTargetView* nullRtv2[] = { nullptr };
        ctx_->OMSetRenderTargets(1, nullRtv2, nullptr);

        // ---- Optional blur fill pass (CS): replace empty texels with neighbour avg ----
        if (args.shadowBlur && csShadowBlur_ && shadowFilledUav_) {
            MICROPROFILE_SCOPEGPUI("ShadowBlur", 0xff80a0a0);
            ctx_->CSSetShader(csShadowBlur_.Get(), nullptr, 0);
            ID3D11Buffer* csCbs[] = { cbPerFrame_.Get() };
            ctx_->CSSetConstantBuffers(0, 1, csCbs);
            ID3D11ShaderResourceView* srcSrv[] = { shadowSrv_.Get() };
            ctx_->CSSetShaderResources(7, 1, srcSrv);            // gShadowSrcTex
            ID3D11UnorderedAccessView* dstUav[] = { shadowFilledUav_.Get() };
            UINT initc[] = { 0 };
            ctx_->CSSetUnorderedAccessViews(3, 1, dstUav, initc);
            UINT gx = (shadowSize_ + 7) / 8;
            UINT gy = (shadowSize_ + 7) / 8;
            ctx_->Dispatch(gx, gy, 1);
            ID3D11ShaderResourceView* nullSrv[] = { nullptr };
            ctx_->CSSetShaderResources(7, 1, nullSrv);
            ID3D11UnorderedAccessView* nullUav[] = { nullptr };
            ctx_->CSSetUnorderedAccessViews(3, 1, nullUav, initc);
            ctx_->CSSetShader(nullptr, nullptr, 0);
        }

        // Restore main RT + DSV. Without this, any tech that doesn't re-bind
        // RTV itself (PolyAxis, Points, etc.) would draw into a null target.
        ID3D11RenderTargetView* mainRtv = postEnabled ? taaSceneRtv_.Get() : rtv_.Get();
        ID3D11RenderTargetView* mainRtvs[] = { mainRtv };
        ctx_->OMSetRenderTargets(1, mainRtvs, dsv_.Get());
        D3D11_VIEWPORT mvp = { 0, 0, (float)width_, (float)height_, 0.0f, 1.0f };
        ctx_->RSSetViewports(1, &mvp);
        ctx_->RSSetState(rsSolid_.Get());
        // Bind shadow SRV at t6 + comparison sampler at s1 for all lit passes.
        // Choose filled copy if blur active.
        ID3D11ShaderResourceView* shSrv[] = {
            (args.shadowBlur && shadowFilledSrv_) ? shadowFilledSrv_.Get() : shadowSrv_.Get()
        };
        ctx_->PSSetShaderResources(6, 1, shSrv);
        ID3D11SamplerState* shSm[] = { shadowSamp_.Get() };
        ctx_->PSSetSamplers(1, 1, shSm);
        // Update cache so subsequent frames can skip the caster pass.
        lastSunDir_[0] = args.sunDir[0];
        lastSunDir_[1] = args.sunDir[1];
        lastSunDir_[2] = args.sunDir[2];
        lastShadowSize_ = shadowSize_;
        lastShadowCullFront_ = args.shadowCullFront;
        lastShadowLod_       = args.shadowLod;
        lastShadowBlur_      = args.shadowBlur;
        shadowMapDirty_ = false;
    }
    shadowDone:;

    // Extract 6 frustum planes from row-major viewProj.
    float M[16];
    hlslpp::store(M, vp);
    float planes[6][4];
    planes[0][0] = M[0] + M[3];  planes[0][1] = M[4] + M[7];
    planes[0][2] = M[8] + M[11]; planes[0][3] = M[12] + M[15];
    planes[1][0] = M[3] - M[0];  planes[1][1] = M[7] - M[4];
    planes[1][2] = M[11] - M[8]; planes[1][3] = M[15] - M[12];
    planes[2][0] = M[1] + M[3];  planes[2][1] = M[5] + M[7];
    planes[2][2] = M[9] + M[11]; planes[2][3] = M[13] + M[15];
    planes[3][0] = M[3] - M[1];  planes[3][1] = M[7] - M[5];
    planes[3][2] = M[11] - M[9]; planes[3][3] = M[15] - M[13];
    planes[4][0] = M[2];         planes[4][1] = M[6];
    planes[4][2] = M[10];        planes[4][3] = M[14];
    planes[5][0] = M[3] - M[2];  planes[5][1] = M[7] - M[6];
    planes[5][2] = M[11] - M[10];planes[5][3] = M[15] - M[14];

    struct Job {
        const GpuSubMesh* gs;
        float ox, oz;
        PointLod lod;
    };
    // One job vector per RenderTech ordinal. Each chunk gets routed to either
    // techClose or techFar based on its projected pixel-per-voxel size.
    static constexpr int kTechCount = 13;
    static thread_local std::vector<Job> jobsByTech[kTechCount];
    for (int i = 0; i < kTechCount; ++i) jobsByTech[i].clear();

    // Near/far switch threshold (pixels-per-voxel). Heuristic: when a chunk's
    // voxels project to fewer than splatRadius pixels, splat-class techs work
    // well — use that as a generic boundary. Treat single-tech mode as "all
    // chunks go to the enabled tech".
    float switchPpv = (float)std::max(1, splatRadius);

    float fovRad = cam.fovDeg * 3.14159265358979f / 180.0f;
    float focalPx = (float)height_ / (2.0f * tanf(fovRad * 0.5f));
    hlslpp::float3 camPosHl = cam.position;
    float camP[3]; hlslpp::store(camP, camPosHl);

    lastDrawn_ = 0;
    lastDrawnTris_ = 0;
    lastPointCount_ = 0;
    const float spanX = sceneSpan_[0];
    const float spanZ = sceneSpan_[2];

    auto chunkDist = [&](const float aMin[3], const float aMax[3]) -> float {
        float dx = camP[0] < aMin[0] ? aMin[0] - camP[0]
                 : camP[0] > aMax[0] ? camP[0] - aMax[0] : 0.0f;
        float dy = camP[1] < aMin[1] ? aMin[1] - camP[1]
                 : camP[1] > aMax[1] ? camP[1] - aMax[1] : 0.0f;
        float dz = camP[2] < aMin[2] ? aMin[2] - camP[2]
                 : camP[2] > aMax[2] ? camP[2] - aMax[2] : 0.0f;
        float d = sqrtf(dx*dx + dy*dy + dz*dz);
        return d < 1e-3f ? 1e-3f : d;
    };
    auto cullAabb = [&](const float aMin[3], const float aMax[3]) -> bool {
        for (int pi = 0; pi < 6; ++pi) {
            const float a = planes[pi][0], b = planes[pi][1], c = planes[pi][2], d = planes[pi][3];
            float px = a >= 0 ? aMax[0] : aMin[0];
            float py = b >= 0 ? aMax[1] : aMin[1];
            float pz = c >= 0 ? aMax[2] : aMin[2];
            if (a * px + b * py + c * pz + d < 0.0f) return true;
        }
        return false;
    };
    auto aabbFullyInside = [&](const float aMin[3], const float aMax[3]) -> bool {
        for (int pi = 0; pi < 6; ++pi) {
            const float a = planes[pi][0], b = planes[pi][1], c = planes[pi][2], d = planes[pi][3];
            float nx = a >= 0 ? aMin[0] : aMax[0];
            float ny = b >= 0 ? aMin[1] : aMax[1];
            float nz = c >= 0 ? aMin[2] : aMax[2];
            if (a * nx + b * ny + c * nz + d < 0.0f) return false;
        }
        return true;
    };
    {
    MICROPROFILE_SCOPEI("CPU", "BuildJobs", 0xffffa0a0);
    for (int gz = 0; gz < gridSize; ++gz) {
        for (int gx = 0; gx < gridSize; ++gx) {
            float ox = gx * spanX;
            float oz = gz * spanZ;

            float wholeMin[3] = {
                sceneOrigin_[0] + ox,
                sceneOrigin_[1],
                sceneOrigin_[2] + oz,
            };
            float wholeMax[3] = {
                wholeMin[0] + sceneSpan_[0],
                wholeMin[1] + sceneSpan_[1],
                wholeMin[2] + sceneSpan_[2],
            };
            if (cullAabb(wholeMin, wholeMax)) continue;
            const bool wholeIn = aabbFullyInside(wholeMin, wholeMax);

            for (const auto& gs : subs_) {
                float aMin[3] = { gs.aabbMin[0] + ox, gs.aabbMin[1], gs.aabbMin[2] + oz };
                float aMax[3] = { gs.aabbMax[0] + ox, gs.aabbMax[1], gs.aabbMax[2] + oz };
                if (!wholeIn && cullAabb(aMin, aMax)) continue;

                PointLod effLod = pointLod;
                float ppvHere = focalPx / chunkDist(aMin, aMax);
                if (pointLod == PointLod::Auto) {
                    float t0 = 1.333f / pointLodScale;
                    float t1 = 0.333f / pointLodScale;
                    float t2 = 0.083f / pointLodScale;
                    if      (ppvHere >= t0) effLod = PointLod::L0;
                    else if (ppvHere >= t1) effLod = PointLod::L1;
                    else if (ppvHere >= t2) effLod = PointLod::L2;
                    else                    effLod = PointLod::L3;
                }
                // Per-chunk near/far routing. Unchecked side culls those chunks
                // entirely — they do NOT spill over to the other tech.
                const bool isClose = (ppvHere >= switchPpv);
                if (isClose && !closeEnabled) continue;
                if (!isClose && !farEnabled)  continue;
                RenderTech jt = isClose ? techClose : techFar;
                jobsByTech[(int)jt].push_back({ &gs, ox, oz, effLod });
            }
        }
    }
    } // BuildJobs

    UINT stride = sizeof(Vertex);
    UINT offset = 0;

    uint32_t curTint = 0;
    auto lodToIdx = [](PointLod l) -> uint32_t {
        if (l == PointLod::L3) return 3u;
        if (l == PointLod::L2) return 2u;
        if (l == PointLod::L1) return 1u;
        return 0u;
    };
    auto issueChunkCbEx = [&](const Job& j, float padVal, uint32_t voxelBase) {
        D3D11_MAPPED_SUBRESOURCE mm;
        ctx_->Map(cbPerChunk_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mm);
        CBPerChunk cc = {};
        cc.chunkBase[0] = sceneOrigin_[0] + j.ox;
        cc.chunkBase[1] = sceneOrigin_[1];
        cc.chunkBase[2] = sceneOrigin_[2] + j.oz;
        cc._pad = padVal;
        cc.voxelBase = voxelBase;
        cc.chunkLodIdx = lodToIdx(j.lod);
        cc.chunkTint   = curTint;
        memcpy(mm.pData, &cc, sizeof(cc));
        ctx_->Unmap(cbPerChunk_.Get(), 0);
    };
    auto issueChunkCb = [&](const Job& j) { issueChunkCbEx(j, 0.0f, 0u); };

    auto lodToHalfExtent = [](PointLod l) -> float {
        if (l == PointLod::L3) return 4.0f;
        if (l == PointLod::L2) return 2.0f;
        if (l == PointLod::L1) return 1.0f;
        return 0.5f;
    };
    auto pointRangeForJob = [](const Job& j, uint32_t& first, uint32_t& count) {
        if      (j.lod == PointLod::L1) { first = j.gs->pointFirstL1; count = j.gs->pointCountL1; }
        else if (j.lod == PointLod::L2) { first = j.gs->pointFirstL2; count = j.gs->pointCountL2; }
        else if (j.lod == PointLod::L3) { first = j.gs->pointFirstL3; count = j.gs->pointCountL3; }
        else                            { first = j.gs->pointFirst;  count = j.gs->pointCount;  }
    };

    // -------------- Splat tech --------------
    {
    auto& jobs = jobsByTech[(int)RenderTech::Splat];
    if (!jobs.empty() && pointSb_ && splatColorRtv_ && splatFinalUav_)
    {
        MICROPROFILE_SCOPEGPUI("Splat", 0xffffd060);
        MICROPROFILE_SCOPEI("CPU", "Splat", 0xffffd060);
        float clr[4] = { lastClear_[0], lastClear_[1], lastClear_[2], 0.0f };
        ctx_->ClearRenderTargetView(splatColorRtv_.Get(), clr);
        const float zeroClear[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        ctx_->ClearRenderTargetView(splatMaskRtv_.Get(), zeroClear);
        ctx_->ClearDepthStencilView(splatDsv_.Get(), D3D11_CLEAR_DEPTH, 0.0f, 0);

        // Point pass writes color (slot 0) + visMask (slot 1).
        ID3D11RenderTargetView* splatRtvs[] = { splatColorRtv_.Get(), splatMaskRtv_.Get() };
        ctx_->OMSetRenderTargets(2, splatRtvs, splatDsv_.Get());
        ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);

        {
            MICROPROFILE_SCOPEGPUI("Splat/Points", 0xffffc040);
            // No VB — vsmain_points reads vertex data from pointSrv_ (t1) via SV_VertexID.
            ctx_->IASetInputLayout(nullptr);
            ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
            ctx_->VSSetShader(vsPoints_.Get(), nullptr, 0);
            ctx_->PSSetShader(psSplatAlbedo_.Get(), nullptr, 0);
            ID3D11ShaderResourceView* vsSrvs[] = { nullptr, pointSrv_.Get(), pointAo6Srv_.Get() };
            ctx_->VSSetShaderResources(0, 3, vsSrvs);
            ID3D11Buffer* nullVbs[] = { nullptr };
            UINT vbStride = 0, vbOff = 0;
            ctx_->IASetVertexBuffers(0, 1, nullVbs, &vbStride, &vbOff);
            size_t i = 0;
            while (i < jobs.size()) {
                float ox = jobs[i].ox, oz = jobs[i].oz;
                PointLod lod = jobs[i].lod;
                curTint = 2u + lodToIdx(lod);
                const Job firstInGroup = jobs[i];
                while (i < jobs.size()
                       && jobs[i].ox == ox && jobs[i].oz == oz
                       && jobs[i].lod == lod) {
                    uint32_t spanFirst, count0;
                    pointRangeForJob(jobs[i], spanFirst, count0);
                    if (count0 == 0) { ++i; continue; }
                    uint32_t spanEnd = spanFirst + count0;
                    ++i;
                    while (i < jobs.size()
                           && jobs[i].ox == ox && jobs[i].oz == oz
                           && jobs[i].lod == lod) {
                        uint32_t f, c;
                        pointRangeForJob(jobs[i], f, c);
                        if (f != spanEnd) break;
                        spanEnd += c;
                        ++i;
                    }
                    uint32_t spanCount = spanEnd - spanFirst;
                    issueChunkCbEx(firstInGroup, lodToHalfExtent(lod), spanFirst);
                    // SRV path: SV_VertexID = 0..N-1; VS adds gVoxelBase = spanFirst.
                    ctx_->Draw(spanCount, 0);
                    ++lastDrawn_;
                    lastDrawnTris_ += spanCount;
                    lastPointCount_ += spanCount;
                }
            }
        }

        // CS dilate -> splatFinalTex_.
        if (splatFilter) {
            MICROPROFILE_SCOPEGPUI("Splat/CS", 0xffffa030);
            ID3D11RenderTargetView* nullRtvs[] = { nullptr };
            ctx_->OMSetRenderTargets(1, nullRtvs, nullptr);
            ctx_->CSSetShader(csSplat_.Get(), nullptr, 0);
            ID3D11Buffer* csCbs[] = { cbPerFrame_.Get() };
            ctx_->CSSetConstantBuffers(0, 1, csCbs);
            // t0..t3: splat dilation source. t6: sun shadow map (so the CS-side
            // ApplyShadowLighting -> SampleShadow can sample).
            ID3D11ShaderResourceView* csSrvs[] = { nullptr, splatDepthSrv_.Get(), splatColorSrv_.Get(), splatMaskSrv_.Get() };
            ctx_->CSSetShaderResources(0, 4, csSrvs);
            {
                ID3D11ShaderResourceView* effectiveSrv =
                    (args.shadowBlur && shadowFilledSrv_) ? shadowFilledSrv_.Get() : shadowSrv_.Get();
                if (effectiveSrv) {
                    ID3D11ShaderResourceView* shSrvCS[] = { effectiveSrv };
                    ctx_->CSSetShaderResources(6, 1, shSrvCS);
                }
            }
            if (shadowSamp_) {
                ID3D11SamplerState* shSmCS[] = { shadowSamp_.Get() };
                ctx_->CSSetSamplers(1, 1, shSmCS);
            }
            ID3D11UnorderedAccessView* csUavs[] = { nullptr, splatFinalUav_.Get(), splatFinalDepthUav_.Get() };
            UINT initc[] = { 0, 0, 0 };
            ctx_->CSSetUnorderedAccessViews(0, 3, csUavs, initc);
            UINT gx = (width_ + 7) / 8;
            UINT gy = (height_ + 7) / 8;
            ctx_->Dispatch(gx, gy, 1);
            ID3D11ShaderResourceView* nullCsSrvs[] = { nullptr, nullptr, nullptr, nullptr };
            ctx_->CSSetShaderResources(0, 4, nullCsSrvs);
            ID3D11ShaderResourceView* nullShCS[] = { nullptr };
            ctx_->CSSetShaderResources(6, 1, nullShCS);
            ID3D11UnorderedAccessView* nullCsUavs[] = { nullptr, nullptr, nullptr };
            ctx_->CSSetUnorderedAccessViews(0, 3, nullCsUavs, initc);

            // ---- Pass 2: hole-fill on pass-1 output ----
            // Reads pass-1 color/depth, fills alpha==0 pixels from nearest
            // filled neighbour. Output ping-pongs into splatFinal2*.
            if (args.splatDilate2Pass && csSplatFill_ && splatFinal2Uav_) {
                MICROPROFILE_SCOPEGPUI("Splat/CS2", 0xffff8020);
                ctx_->CSSetShader(csSplatFill_.Get(), nullptr, 0);
                // Inputs (pass-1 output) at t1 (depth) + t2 (color).
                ID3D11ShaderResourceView* csSrvs2[] = { nullptr, splatFinalDepthSrv_.Get(), splatFinalSrv_.Get() };
                ctx_->CSSetShaderResources(0, 3, csSrvs2);
                ID3D11UnorderedAccessView* csUavs2[] = { nullptr, splatFinal2Uav_.Get(), splatFinal2DepthUav_.Get() };
                UINT initc2[] = { 0, 0, 0 };
                ctx_->CSSetUnorderedAccessViews(0, 3, csUavs2, initc2);
                ctx_->Dispatch(gx, gy, 1);
                ID3D11ShaderResourceView* nullCsSrvs2[] = { nullptr, nullptr, nullptr };
                ctx_->CSSetShaderResources(0, 3, nullCsSrvs2);
                ctx_->CSSetUnorderedAccessViews(0, 3, nullCsUavs, initc2);
            }
            ctx_->CSSetShader(nullptr, nullptr, 0);
        }

        // Composite splat onto main scene RT.
        ID3D11RenderTargetView* compositeRtv = postEnabled ? taaSceneRtv_.Get() : rtv_.Get();
        ctx_->OMSetRenderTargets(1, &compositeRtv, dsv_.Get());
        ctx_->OMSetDepthStencilState(dsAlwaysWrite_.Get(), 0);
        ctx_->RSSetState(rsNoCull_.Get());
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx_->IASetInputLayout(nullptr);
        ID3D11Buffer* nVb[] = { nullptr }; UINT zz = 0;
        ctx_->IASetVertexBuffers(0, 1, nVb, &zz, &zz);
        ctx_->VSSetShader(vsBlit_.Get(), nullptr, 0);
        ctx_->PSSetShader(psSplatComposite_.Get(), nullptr, 0);
        // If 2-pass is enabled, sample from pass-2 ping-pong target.
        const bool use2 = splatFilter && args.splatDilate2Pass && splatFinal2Srv_;
        ID3D11ShaderResourceView* compSrv[] = {
            !splatFilter ? splatColorSrv_.Get()
                          : (use2 ? splatFinal2Srv_.Get() : splatFinalSrv_.Get())
        };
        ctx_->PSSetShaderResources(8, 1, compSrv);
        ID3D11ShaderResourceView* compDepthSrv[] = {
            !splatFilter ? splatDepthSrv_.Get()
                          : (use2 ? splatFinal2DepthSrv_.Get() : splatFinalDepthSrv_.Get())
        };
        ctx_->PSSetShaderResources(7, 1, compDepthSrv);
        ctx_->OMSetBlendState(bsAlphaOver_.Get(), nullptr, 0xFFFFFFFFu);
        ctx_->Draw(3, 0);
        ctx_->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFFu);
        ID3D11ShaderResourceView* nullCSrv[] = { nullptr };
        ctx_->PSSetShaderResources(8, 1, nullCSrv);
        ctx_->PSSetShaderResources(7, 1, nullCSrv);
        ctx_->RSSetState(rsSolid_.Get());

        ID3D11RenderTargetView* active = postEnabled ? taaSceneRtv_.Get() : rtv_.Get();
        ID3D11RenderTargetView* rrtvs[] = { active };
        ctx_->OMSetRenderTargets(1, rrtvs, dsv_.Get());
        ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
    }
    } // splat scope

    // -------------- Points tech --------------
    {
    auto& jobs = jobsByTech[(int)RenderTech::Points];
    if (!jobs.empty() && pointSb_) {
        MICROPROFILE_SCOPEGPUI("Points", 0xff80ff80);
        MICROPROFILE_SCOPEI("CPU", "Points", 0xff80ff80);
        curTint = 2;
        // No VB — VS pulls voxel data from pointSrv_ at t1 via SV_VertexID.
        ctx_->IASetInputLayout(nullptr);
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
        ctx_->VSSetShader(vsPoints_.Get(), nullptr, 0);
        ctx_->PSSetShader(pointLight == PointLighting::Simple
                              ? psPointsSimple_.Get()
                              : psPoints_.Get(), nullptr, 0);
        ID3D11ShaderResourceView* vsSrvs[] = { nullptr, pointSrv_.Get(), pointAo6Srv_.Get() };
        ctx_->VSSetShaderResources(0, 3, vsSrvs);
        ID3D11Buffer* nullVbs[] = { nullptr };
        UINT vbS = 0, vbO = 0;
        ctx_->IASetVertexBuffers(0, 1, nullVbs, &vbS, &vbO);

        size_t i = 0;
        while (i < jobs.size()) {
            float ox = jobs[i].ox, oz = jobs[i].oz;
            PointLod lod = jobs[i].lod;
            curTint = 2u + lodToIdx(lod);
            const Job firstInGroup = jobs[i];
            while (i < jobs.size()
                   && jobs[i].ox == ox && jobs[i].oz == oz
                   && jobs[i].lod == lod) {
                uint32_t spanFirst, count0;
                pointRangeForJob(jobs[i], spanFirst, count0);
                if (count0 == 0) { ++i; continue; }
                uint32_t spanEnd = spanFirst + count0;
                ++i;
                while (i < jobs.size()
                       && jobs[i].ox == ox && jobs[i].oz == oz
                       && jobs[i].lod == lod) {
                    uint32_t f, c;
                    pointRangeForJob(jobs[i], f, c);
                    if (f != spanEnd) break;
                    spanEnd += c;
                    ++i;
                }
                uint32_t spanCount = spanEnd - spanFirst;
                issueChunkCbEx(firstInGroup, lodToHalfExtent(lod), spanFirst);
                ctx_->Draw(spanCount, 0);
                ++lastDrawn_;
                lastDrawnTris_ += spanCount;
                lastPointCount_ += spanCount;
            }
        }
    }

    } // points scope

    // -------------- PointCS tech --------------
    {
    auto& jobs = jobsByTech[(int)RenderTech::PointCS];
    if (!jobs.empty() && pointSb_ && pointSrv_ && csColorUav_) {
        MICROPROFILE_SCOPEGPUI("PointCS", 0xff80c0a0);
        MICROPROFILE_SCOPEI("CPU", "PointCS", 0xff80c0a0);
        curTint = 2;
        uint32_t clearC[4] = { 0xFF291F1Au, 0, 0, 0 };
        ctx_->ClearUnorderedAccessViewUint(csColorUav_.Get(), clearC);

        ID3D11RenderTargetView* nullRtv[] = { nullptr };
        ctx_->OMSetRenderTargets(1, nullRtv, nullptr);

        ctx_->CSSetShader(csPoints_.Get(), nullptr, 0);
        ID3D11UnorderedAccessView* uavs[] = { csColorUav_.Get() };
        UINT initCounts[] = { 0 };
        ctx_->CSSetUnorderedAccessViews(0, 1, uavs, initCounts);
        ID3D11ShaderResourceView* srvs[] = { pointSrv_.Get() };
        ctx_->CSSetShaderResources(0, 1, srvs);
        ID3D11Buffer* csCbs[] = { cbCS_.Get() };
        ctx_->CSSetConstantBuffers(2, 1, csCbs);

        for (const auto& j : jobs) {
            uint32_t pFirst, pCount;
            pointRangeForJob(j, pFirst, pCount);
            if (pCount == 0) continue;
            D3D11_MAPPED_SUBRESOURCE mm;
            ctx_->Map(cbCS_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mm);
            CBPerCS cb;
            hlslpp::store(cb.viewProj, vp);
            cb.chunkBase[0] = sceneOrigin_[0] + j.ox;
            cb.chunkBase[1] = sceneOrigin_[1];
            cb.chunkBase[2] = sceneOrigin_[2] + j.oz;
            cb.voxelOffset = pFirst;
            cb.voxelCount  = pCount;
            cb.w = width_;
            cb.h = height_;
            cb.lodHalfExtent = lodToHalfExtent(j.lod);
            memcpy(mm.pData, &cb, sizeof(cb));
            ctx_->Unmap(cbCS_.Get(), 0);

            UINT groups = (pCount + 63) / 64;
            ctx_->Dispatch(groups, 1, 1);
            ++lastDrawn_;
            lastDrawnTris_ += pCount;
            lastPointCount_ += pCount;
        }

        ID3D11UnorderedAccessView* noUav[] = { nullptr };
        ctx_->CSSetUnorderedAccessViews(0, 1, noUav, initCounts);
        ID3D11ShaderResourceView* noSrv[] = { nullptr };
        ctx_->CSSetShaderResources(0, 1, noSrv);
        ctx_->CSSetShader(nullptr, nullptr, 0);

        // Blit UAV color back to whichever RT we render the rest of the scene to.
        ID3D11RenderTargetView* active = postEnabled ? taaSceneRtv_.Get() : rtv_.Get();
        ID3D11RenderTargetView* rtvs[] = { active };
        ctx_->OMSetRenderTargets(1, rtvs, nullptr);
        ctx_->OMSetDepthStencilState(dsAlways_.Get(), 0);
        ctx_->IASetInputLayout(nullptr);
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx_->VSSetShader(vsBlit_.Get(), nullptr, 0);
        ctx_->PSSetShader(psBlit_.Get(), nullptr, 0);
        ID3D11Buffer* nullVb[] = { nullptr };
        UINT zero = 0;
        ctx_->IASetVertexBuffers(0, 1, nullVb, &zero, &zero);
        ID3D11ShaderResourceView* blitSrv[] = { csColorSrv_.Get() };
        ctx_->PSSetShaderResources(0, 1, blitSrv);
        ctx_->Draw(3, 0);

        ID3D11ShaderResourceView* nullSrv[] = { nullptr };
        ctx_->PSSetShaderResources(0, 1, nullSrv);
        ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
        ctx_->OMSetRenderTargets(1, rtvs, dsv_.Get());
        ctx_->IASetInputLayout(inputLayout_.Get());
    }

    } // pointcs scope

    // -------------- PolyVID tech --------------
    {
    auto& jobs = jobsByTech[(int)RenderTech::PolyVID];
    if (!jobs.empty() && pointSrv_ && polyVidIb_) {
        MICROPROFILE_SCOPEGPUI("PolyVID", 0xff70b0ff);
        MICROPROFILE_SCOPEI("CPU", "PolyVID", 0xff70b0ff);
        curTint = 1;
        ctx_->IASetInputLayout(nullptr);
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx_->VSSetShader(vsPolyVid_.Get(), nullptr, 0);
        ctx_->PSSetShader(psPolyVid_.Get(), nullptr, 0);
        ID3D11Buffer* nullVbs[] = { nullptr };
        UINT z = 0;
        ctx_->IASetVertexBuffers(0, 1, nullVbs, &z, &z);
        ctx_->IASetIndexBuffer(polyVidIb_.Get(), DXGI_FORMAT_R32_UINT, 0);
        ID3D11ShaderResourceView* srvs[] = { nullptr, pointSrv_.Get() };
        ctx_->VSSetShaderResources(0, 2, srvs);
        for (const auto& j : jobs) {
            issueChunkCbEx(j, 0.0f, j.gs->pointFirst);
            // 36 indices per voxel = 12 tris (6 faces x 2).
            ctx_->DrawIndexed(j.gs->pointCount * 36, 0, 0);
            ++lastDrawn_;
            lastDrawnTris_ += (uint64_t)j.gs->pointCount * 12;
            lastPointCount_ += j.gs->pointCount;
        }
        ID3D11ShaderResourceView* nullSrvs[] = { nullptr, nullptr };
        ctx_->VSSetShaderResources(0, 2, nullSrvs);
        ctx_->IASetInputLayout(inputLayout_.Get());
    }

    } // polyvid scope

    // -------------- PolyAxis tech --------------
    {
    auto& jobs = jobsByTech[(int)RenderTech::PolyAxis];
    if (!jobs.empty() && pointSrv_) {
        MICROPROFILE_SCOPEGPUI("PolyAxis", 0xff60a0e0);
        MICROPROFILE_SCOPEI("CPU", "PolyAxis", 0xff60a0e0);
        curTint = 1;
        if (pointAo6Srv_) {
            ID3D11ShaderResourceView* aoSrvs[] = { pointAo6Srv_.Get() };
            ctx_->VSSetShaderResources(2, 1, aoSrvs);
        }
        ctx_->IASetInputLayout(nullptr);
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx_->VSSetShader(vsPolyAxis_.Get(), nullptr, 0);
        ctx_->PSSetShader(psPolyVid_.Get(), nullptr, 0);
        ID3D11Buffer* nullVbs[] = { nullptr };
        UINT z = 0;
        ctx_->IASetVertexBuffers(0, 1, nullVbs, &z, &z);
        ID3D11ShaderResourceView* srvs[] = { nullptr, pointSrv_.Get() };
        ctx_->VSSetShaderResources(0, 2, srvs);
        for (const auto& j : jobs) {
            issueChunkCbEx(j, 0.0f, j.gs->pointFirst);
            ctx_->Draw(j.gs->pointCount * 18u, 0);
            ++lastDrawn_;
            lastDrawnTris_ += (uint64_t)j.gs->pointCount * 6;
            lastPointCount_ += j.gs->pointCount;
        }
        ID3D11ShaderResourceView* nullSrvs[] = { nullptr, nullptr, nullptr };
        ctx_->VSSetShaderResources(0, 3, nullSrvs);
        ctx_->IASetInputLayout(inputLayout_.Get());
    }

    } // polyaxis scope

    // -------------- PolyAxisInstanced tech --------------
    {
    auto& jobs = jobsByTech[(int)RenderTech::PolyAxisInstanced];
    if (!jobs.empty() && pointSrv_) {
        MICROPROFILE_SCOPEGPUI("PolyAxisInst", 0xff5090d0);
        MICROPROFILE_SCOPEI("CPU", "PolyAxisInst", 0xff5090d0);
        curTint = 1;
        ctx_->IASetInputLayout(nullptr);
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx_->VSSetShader(vsPolyAxisInstanced_.Get(), nullptr, 0);
        ctx_->PSSetShader(psPolyVid_.Get(), nullptr, 0);
        ID3D11Buffer* nullVbs[] = { nullptr };
        UINT z = 0;
        ctx_->IASetVertexBuffers(0, 1, nullVbs, &z, &z);
        ID3D11ShaderResourceView* srvs[] = { nullptr, pointSrv_.Get() };
        ctx_->VSSetShaderResources(0, 2, srvs);
        if (pointAo6Srv_) {
            ID3D11ShaderResourceView* aoSrvs[] = { pointAo6Srv_.Get() };
            ctx_->VSSetShaderResources(2, 1, aoSrvs);
        }
        for (const auto& j : jobs) {
            issueChunkCbEx(j, 0.0f, j.gs->pointFirst);
            ctx_->DrawInstanced(18, j.gs->pointCount, 0, 0);
            ++lastDrawn_;
            lastDrawnTris_ += (uint64_t)j.gs->pointCount * 6;
            lastPointCount_ += j.gs->pointCount;
        }
        ID3D11ShaderResourceView* nullSrvs[] = { nullptr, nullptr, nullptr };
        ctx_->VSSetShaderResources(0, 3, nullSrvs);
        ctx_->IASetInputLayout(inputLayout_.Get());
    }

    } // polyaxisinstanced scope

    // -------------- Billboard tech --------------
    {
    auto& jobs = jobsByTech[(int)RenderTech::Billboard];
    if (!jobs.empty() && pointSrv_ && billboardIb_) {
        MICROPROFILE_SCOPEGPUI("Billboard", 0xffa0c060);
        MICROPROFILE_SCOPEI("CPU", "Billboard", 0xffa0c060);
        curTint = 2;
        ctx_->IASetInputLayout(nullptr);
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx_->VSSetShader(vsBillboard_.Get(), nullptr, 0);
        ctx_->PSSetShader(psBillboard_.Get(), nullptr, 0);
        ctx_->RSSetState(rsNoCull_.Get());
        ID3D11Buffer* nullVbs[] = { nullptr };
        UINT z = 0;
        ctx_->IASetVertexBuffers(0, 1, nullVbs, &z, &z);
        ctx_->IASetIndexBuffer(billboardIb_.Get(), DXGI_FORMAT_R32_UINT, 0);
        ID3D11ShaderResourceView* srvs[] = { nullptr, pointSrv_.Get() };
        ctx_->VSSetShaderResources(0, 2, srvs);
        for (const auto& j : jobs) {
            issueChunkCbEx(j, 0.0f, j.gs->pointFirst);
            ctx_->DrawIndexed(j.gs->pointCount * 6, 0, 0);
            ++lastDrawn_;
            lastDrawnTris_ += (uint64_t)j.gs->pointCount * 2;
            lastPointCount_ += j.gs->pointCount;
        }
        ID3D11ShaderResourceView* nullSrvs[] = { nullptr, nullptr };
        ctx_->VSSetShaderResources(0, 2, nullSrvs);
        ctx_->RSSetState(rsSolid_.Get());
        ctx_->IASetInputLayout(inputLayout_.Get());
    }

    } // billboard scope

    // -------------- BillboardTri tech --------------
    {
    auto& jobs = jobsByTech[(int)RenderTech::BillboardTri];
    if (!jobs.empty() && pointSrv_) {
        MICROPROFILE_SCOPEGPUI("BillboardTri", 0xff90b050);
        MICROPROFILE_SCOPEI("CPU", "BillboardTri", 0xff90b050);
        curTint = 2;
        ctx_->IASetInputLayout(nullptr);
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx_->VSSetShader(vsBillboardTri_.Get(), nullptr, 0);
        ctx_->PSSetShader(psBillboard_.Get(), nullptr, 0);
        ctx_->RSSetState(rsNoCull_.Get());
        ID3D11Buffer* nullVbs[] = { nullptr };
        UINT z = 0;
        ctx_->IASetVertexBuffers(0, 1, nullVbs, &z, &z);
        ID3D11ShaderResourceView* srvs[] = { nullptr, pointSrv_.Get() };
        ctx_->VSSetShaderResources(0, 2, srvs);
        for (const auto& j : jobs) {
            issueChunkCbEx(j, 0.0f, j.gs->pointFirst);
            ctx_->Draw(j.gs->pointCount * 3, 0);
            ++lastDrawn_;
            lastDrawnTris_ += j.gs->pointCount;
            lastPointCount_ += j.gs->pointCount;
        }
        ID3D11ShaderResourceView* nullSrvs[] = { nullptr, nullptr };
        ctx_->VSSetShaderResources(0, 2, nullSrvs);
        ctx_->RSSetState(rsSolid_.Get());
        ctx_->IASetInputLayout(inputLayout_.Get());
    }

    } // billboardtri scope

    // -------------- HexSprite tech (DISABLED) --------------
    // Disabled so we can run all point techs through a single structured-SRV
    // buffer (HexSprite needed a per-instance VB, forcing a duplicate copy).
    // To revive: uncomment + re-enable the enum entry + restore vsHex_/psHex_.
    #if 0
    {
    auto& jobs = jobsByTech[(int)RenderTech::HexSprite];
    if (!jobs.empty() && pointVb_) {
        MICROPROFILE_SCOPEGPUI("HexSprite", 0xff80b070);
        curTint = 2;
        ctx_->IASetInputLayout(inputLayoutHex_.Get());
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx_->VSSetShader(vsHex_.Get(), nullptr, 0);
        ctx_->PSSetShader(psHex_.Get(), nullptr, 0);
        ctx_->RSSetState(rsNoCull_.Get());
        ID3D11Buffer* vbs[] = { pointVb_.Get() };
        ctx_->IASetVertexBuffers(0, 1, vbs, &stride, &offset);
        for (const auto& j : jobs) {
            issueChunkCb(j);
            ctx_->DrawInstanced(18, j.gs->pointCount, 0, j.gs->pointFirst);
            ++lastDrawn_;
            lastDrawnTris_ += j.gs->pointCount * 6;
            lastPointCount_ += j.gs->pointCount;
        }
        ctx_->RSSetState(rsSolid_.Get());
        ctx_->IASetInputLayout(inputLayout_.Get());
    }
    } // hexsprite scope
    #endif

    // ------------------- TAA composite (optional) -> post (always) -------------------
    ID3D11ShaderResourceView* postInput = nullptr;
    if (postEnabled) {
        ctx_->IASetInputLayout(nullptr);
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ID3D11Buffer* nullVb[] = { nullptr }; UINT zero = 0;
        ctx_->IASetVertexBuffers(0, 1, nullVb, &zero, &zero);
        ID3D11SamplerState* samps[] = { linearClampSampler_.Get() };
        ctx_->PSSetSamplers(0, 1, samps);
        ctx_->OMSetDepthStencilState(dsAlways_.Get(), 0);
        ctx_->RSSetState(rsNoCull_.Get());

        if (taa && vsTaa_ && psTaa_) {
            MICROPROFILE_SCOPEGPUI("TAA", 0xff40ffd0);
            uint32_t curr = taaHistIdx_;
            uint32_t prev = curr ^ 1u;
            ID3D11RenderTargetView* hRtv = taaHistRtv_[curr].Get();
            ctx_->OMSetRenderTargets(1, &hRtv, nullptr);
            ctx_->VSSetShader(vsTaa_.Get(), nullptr, 0);
            ctx_->PSSetShader(psTaa_.Get(), nullptr, 0);
            ID3D11ShaderResourceView* taaSrvs[] = {
                nullptr, nullptr, nullptr, nullptr,
                taaSceneSrv_.Get(),
                taaHistValid_[prev] ? taaHistSrv_[prev].Get() : taaSceneSrv_.Get(),
                depthSrv_.Get()
            };
            ctx_->PSSetShaderResources(0, 7, taaSrvs);
            ctx_->Draw(3, 0);
            ID3D11ShaderResourceView* nullsForPost[] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
            ctx_->PSSetShaderResources(0, 7, nullsForPost);
            postInput = taaHistSrv_[curr].Get();
            taaHistValid_[curr] = true;
            taaHistIdx_ = prev;
            ++taaFrame_;
        } else {
            postInput = taaSceneSrv_.Get();
        }

        // Post pass: sky + sharpen + tonemap.
        {
            MICROPROFILE_SCOPEGPUI("FinalPost", 0xffff8040);
            ID3D11RenderTargetView* bRtv = rtv_.Get();
            ctx_->OMSetRenderTargets(1, &bRtv, nullptr);
            ctx_->VSSetShader(vsTaa_.Get(), nullptr, 0);
            ctx_->PSSetShader(psPost_.Get(), nullptr, 0);
            ID3D11ShaderResourceView* postSrvs[] = { depthSrv_.Get(), postInput };
            ctx_->PSSetShaderResources(6, 2, postSrvs);
            ctx_->Draw(3, 0);
        }
        ctx_->RSSetState(rsSolid_.Get());
        postWroteBackbuf_ = true;

        ID3D11ShaderResourceView* nulls[] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
        ctx_->PSSetShaderResources(0, 8, nulls);

        ID3D11RenderTargetView* backRtv = rtv_.Get();
        ctx_->OMSetRenderTargets(1, &backRtv, nullptr);
    }

    // Remember this frame's un-jittered view-proj for next frame's reprojection.
    hlslpp::store(taaPrevVP_, vpUnjittered);
}

void Renderer::EndFrame(bool vsync)
{
    swap_->Present(vsync ? 1 : 0, 0);
}
