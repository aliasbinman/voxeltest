#define NOMINMAX
#include "renderer.h"
#include "microprofile.h"

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

bool Renderer::Init(HWND hwnd)
{
    if (!CreateDeviceAndSwap(hwnd)) return false;

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

bool Renderer::CreateDeviceAndSwap(HWND hwnd)
{
    UINT flags = 0;
#ifdef _DEBUG
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    D3D_FEATURE_LEVEL flvl;
    D3D_FEATURE_LEVEL want[] = { D3D_FEATURE_LEVEL_11_0 };

    HRESULT hr = D3D11CreateDevice(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
        want, _countof(want), D3D11_SDK_VERSION,
        device_.GetAddressOf(), &flvl, ctx_.GetAddressOf());
#ifdef _DEBUG
    if (FAILED(hr)) {
        flags &= ~D3D11_CREATE_DEVICE_DEBUG;
        hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
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

    ComPtr<ID3DBlob> vsbp, psbp, vsbh, psbh;
    if (!compile("vsmain_points", "vs_5_0", vsbp)) return false;
    if (!compile("psmain_points", "ps_5_0", psbp)) return false;
    ComPtr<ID3DBlob> psbps;
    if (!compile("psmain_points_simple", "ps_5_0", psbps)) return false;
    if (!compile("vsmain_hex", "vs_5_0", vsbh)) return false;
    if (!compile("psmain_hex", "ps_5_0", psbh)) return false;
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
    ComPtr<ID3DBlob> csbSp, psbSa, vsbTa, psbTa, psbPo;
    if (!compile("csmain_splat",        "cs_5_0", csbSp)) return false;
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
    hr = device_->CreateVertexShader(vsbp->GetBufferPointer(), vsbp->GetBufferSize(), nullptr, vsPoints_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbp->GetBufferPointer(), psbp->GetBufferSize(), nullptr, psPoints_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbps->GetBufferPointer(), psbps->GetBufferSize(), nullptr, psPointsSimple_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreateVertexShader(vsbh->GetBufferPointer(), vsbh->GetBufferSize(), nullptr, vsHex_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbh->GetBufferPointer(), psbh->GetBufferSize(), nullptr, psHex_.GetAddressOf());
    if (FAILED(hr)) return false;
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

    // HexSprite: same vertex format, per-instance stepping.
    D3D11_INPUT_ELEMENT_DESC ilHex[] = {
        { "POSITION", 0, DXGI_FORMAT_R16G16B16A16_UINT, 0, 0,                            D3D11_INPUT_PER_INSTANCE_DATA, 1 },
        { "COLOR",    0, DXGI_FORMAT_R8G8B8A8_UINT,     0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
    };
    hr = device_->CreateInputLayout(ilHex, _countof(ilHex),
                                    vsbh->GetBufferPointer(), vsbh->GetBufferSize(),
                                    inputLayoutHex_.GetAddressOf());
    if (FAILED(hr)) return false;

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

    // Vertex buffer (immutable, plain).
    D3D11_BUFFER_DESC pbd = {};
    pbd.Usage = D3D11_USAGE_IMMUTABLE;
    pbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    pbd.ByteWidth = (UINT)(scene.pointVertices.size() * sizeof(Vertex));
    D3D11_SUBRESOURCE_DATA psd = {};
    psd.pSysMem = scene.pointVertices.data();
    device_->CreateBuffer(&pbd, &psd, pointVb_.GetAddressOf());

    // Parallel StructuredBuffer for CS / VS access (same data).
    D3D11_BUFFER_DESC sbd = {};
    sbd.Usage = D3D11_USAGE_IMMUTABLE;
    sbd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    sbd.ByteWidth = pbd.ByteWidth;
    sbd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    sbd.StructureByteStride = sizeof(Vertex);
    ComPtr<ID3D11Buffer> sb;
    if (SUCCEEDED(device_->CreateBuffer(&sbd, &psd, sb.GetAddressOf()))) {
        D3D11_SHADER_RESOURCE_VIEW_DESC srvd = {};
        srvd.Format = DXGI_FORMAT_UNKNOWN;
        srvd.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        srvd.Buffer.NumElements = (UINT)scene.pointVertices.size();
        device_->CreateShaderResourceView(sb.Get(), &srvd, pointSrv_.GetAddressOf());
        pointSb_ = sb;
    }

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

    subs_.reserve(scene.subs.size());
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
        for (int i = 0; i < 16; ++i) cb.sunViewProj[i] = 0.0f;
        cb.shadowBias    = 0.0f;
        cb.shadowMapSize = 1.0f;
        cb.shadowEnable  = 0.0f;
        cb.sunIntensity  = sunIntensity;
        cb.exposure      = exposure;
        cb.roughness     = roughness;
        cb.colorizeClusters = args.colorizeClusters ? 1.0f : 0.0f;
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

    if (!pointVb_) return;

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
    if (!jobs.empty() && pointVb_ && splatColorRtv_ && splatFinalUav_)
    {
        MICROPROFILE_SCOPEGPUI("Splat", 0xffffd060);
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
            ctx_->IASetInputLayout(inputLayout_.Get());
            ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
            ctx_->VSSetShader(vsPoints_.Get(), nullptr, 0);
            ctx_->PSSetShader(psSplatAlbedo_.Get(), nullptr, 0);
            if (pointAo6Srv_) {
                ID3D11ShaderResourceView* aoSrvs[] = { pointAo6Srv_.Get() };
                ctx_->VSSetShaderResources(2, 1, aoSrvs);
            }
            ID3D11Buffer* vbs[] = { pointVb_.Get() };
            UINT vbStride = sizeof(Vertex), vbOff = 0;
            ctx_->IASetVertexBuffers(0, 1, vbs, &vbStride, &vbOff);
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
                    ctx_->Draw(spanCount, spanFirst);
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
            ID3D11ShaderResourceView* csSrvs[] = { nullptr, splatDepthSrv_.Get(), splatColorSrv_.Get(), splatMaskSrv_.Get() };
            ctx_->CSSetShaderResources(0, 4, csSrvs);
            ID3D11UnorderedAccessView* csUavs[] = { nullptr, splatFinalUav_.Get(), splatFinalDepthUav_.Get() };
            UINT initc[] = { 0, 0, 0 };
            ctx_->CSSetUnorderedAccessViews(0, 3, csUavs, initc);
            UINT gx = (width_ + 7) / 8;
            UINT gy = (height_ + 7) / 8;
            ctx_->Dispatch(gx, gy, 1);
            ID3D11ShaderResourceView* nullCsSrvs[] = { nullptr, nullptr, nullptr, nullptr };
            ctx_->CSSetShaderResources(0, 4, nullCsSrvs);
            ID3D11UnorderedAccessView* nullCsUavs[] = { nullptr, nullptr, nullptr };
            ctx_->CSSetUnorderedAccessViews(0, 3, nullCsUavs, initc);
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
        ID3D11ShaderResourceView* compSrv[] = { splatFilter ? splatFinalSrv_.Get() : splatColorSrv_.Get() };
        ctx_->PSSetShaderResources(8, 1, compSrv);
        ID3D11ShaderResourceView* compDepthSrv[] = {
            splatFilter ? splatFinalDepthSrv_.Get() : splatDepthSrv_.Get()
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
    if (!jobs.empty() && pointVb_) {
        MICROPROFILE_SCOPEGPUI("Points", 0xff80ff80);
        curTint = 2;
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
        ctx_->VSSetShader(vsPoints_.Get(), nullptr, 0);
        if (pointAo6Srv_) {
            ID3D11ShaderResourceView* aoSrvs[] = { pointAo6Srv_.Get() };
            ctx_->VSSetShaderResources(2, 1, aoSrvs);
        }
        ctx_->PSSetShader(pointLight == PointLighting::Simple
                              ? psPointsSimple_.Get()
                              : psPoints_.Get(), nullptr, 0);
        ID3D11Buffer* vbs[] = { pointVb_.Get() };
        ctx_->IASetVertexBuffers(0, 1, vbs, &stride, &offset);

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
                ctx_->Draw(spanCount, spanFirst);
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
    if (!jobs.empty() && pointVb_ && pointSrv_ && csColorUav_) {
        MICROPROFILE_SCOPEGPUI("PointCS", 0xff80c0a0);
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

    // -------------- HexSprite tech --------------
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
