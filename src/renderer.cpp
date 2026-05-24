#define NOMINMAX
#include "renderer.h"
#include "microprofile.h"

#include <d3dcompiler.h>
#include <algorithm>
#include <fstream>
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
    float sunViewProj[16];        // sun ortho VP for shadow map
    float shadowBias;
    float shadowMapSize;
    float shadowEnable;           // 0 = bypass shadow sample
    float sunIntensity;           // linear (2^EV)
    float exposure;               // linear (2^EV)
    float roughness;              // reserved
    float colorizeClusters;       // 0/1 tint on/off
    float _padExp[1];
};

struct CBPerChunk {
    float    chunkBase[3];
    float    _pad;          // lodHalfExtent (points) / chunkSize (bounds)
    uint32_t voxelBase;     // PolyVID: chunk's first voxel index in pointSb_
    uint32_t chunkLodIdx;   // 0..3 splat LOD index
    uint32_t chunkTint;     // colorize-clusters debug: 1 green close, 2/3/4 red/orange/yellow L0/L1/L2
    uint32_t _pad2[2];      // bounds path repurposes as asfloat(_pad2.xy) = sizeY/sizeZ
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
    vb_.Reset();
    ib_.Reset();
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
    msaaColorTex_.Reset();
    msaaRtv_.Reset();
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

    // Validate MSAA support; fall back to 1x on unsupported counts.
    DXGI_SAMPLE_DESC sd = { 1, 0 };
    if (msaaSamples_ > 1) {
        UINT q = 0;
        device_->CheckMultisampleQualityLevels(DXGI_FORMAT_R8G8B8A8_UNORM, msaaSamples_, &q);
        if (q > 0) {
            sd.Count   = msaaSamples_;
            sd.Quality = q - 1;
            msaaQuality_ = q - 1;
        } else {
            msaaSamples_ = 1;
            msaaQuality_ = 0;
        }
    } else {
        msaaQuality_ = 0;
    }

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
    td.SampleDesc = sd;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_DEPTH_STENCIL | (msaaSamples_ > 1 ? 0 : D3D11_BIND_SHADER_RESOURCE);
    hr = device_->CreateTexture2D(&td, nullptr, depthTex_.GetAddressOf());
    if (FAILED(hr)) return false;
    D3D11_DEPTH_STENCIL_VIEW_DESC dsvd = {};
    dsvd.Format = DXGI_FORMAT_D32_FLOAT;
    dsvd.ViewDimension = (msaaSamples_ > 1) ? D3D11_DSV_DIMENSION_TEXTURE2DMS
                                            : D3D11_DSV_DIMENSION_TEXTURE2D;
    hr = device_->CreateDepthStencilView(depthTex_.Get(), &dsvd, dsv_.GetAddressOf());
    if (FAILED(hr)) return false;
    depthSrv_.Reset();
    if (msaaSamples_ == 1) {
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
        // Scene RT matches backbuffer format (we CopyResource between them).
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

        // History needs higher precision so repeated lerp doesn't quantize down.
        D3D11_TEXTURE2D_DESC ht = ct;
        ht.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        for (int i = 0; i < 2; ++i) {
            device_->CreateTexture2D(&ht, nullptr, taaHistTex_[i].GetAddressOf());
            device_->CreateRenderTargetView(taaHistTex_[i].Get(), nullptr, taaHistRtv_[i].GetAddressOf());
            device_->CreateShaderResourceView(taaHistTex_[i].Get(), nullptr, taaHistSrv_[i].GetAddressOf());
        }
    }

    // Optional MSAA color target. Backbuffer stays 1x; resolve at EndFrame.
    if (msaaSamples_ > 1) {
        D3D11_TEXTURE2D_DESC mt = {};
        mt.Width = width_;
        mt.Height = height_;
        mt.MipLevels = 1;
        mt.ArraySize = 1;
        mt.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        mt.SampleDesc = sd;
        mt.Usage = D3D11_USAGE_DEFAULT;
        mt.BindFlags = D3D11_BIND_RENDER_TARGET;
        if (FAILED(device_->CreateTexture2D(&mt, nullptr, msaaColorTex_.GetAddressOf()))) return false;
        D3D11_RENDER_TARGET_VIEW_DESC rtvd = {};
        rtvd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        rtvd.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMS;
        if (FAILED(device_->CreateRenderTargetView(msaaColorTex_.Get(), &rtvd, msaaRtv_.GetAddressOf()))) return false;
    }

    // PointCS color/depth UAV textures (R32_UINT for atomic ops + RGBA8 packing).
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

    // Splat color RT (R8G8B8A8_UNORM) + final UAV target.
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

    // Dedicated non-MSAA depth for splat pass. R32_TYPELESS so the CS can
    // read the same texture as a SRV alongside the DSV binding.
    D3D11_TEXTURE2D_DESC sdd = {};
    sdd.Width = width_;
    sdd.Height = height_;
    sdd.MipLevels = 1;
    sdd.ArraySize = 1;
    // Splat depth is now D24S8 so the reconstruction PS can sample stencil to
    // distinguish poly pixels (stencil=1) from splat / bg pixels (stencil=0).
    sdd.Format = DXGI_FORMAT_R24G8_TYPELESS;
    sdd.SampleDesc.Count = 1;
    sdd.Usage = D3D11_USAGE_DEFAULT;
    sdd.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(device_->CreateTexture2D(&sdd, nullptr, splatDepthTex_.GetAddressOf()))) return false;
    D3D11_DEPTH_STENCIL_VIEW_DESC sddv = {};
    sddv.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    sddv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    if (FAILED(device_->CreateDepthStencilView(splatDepthTex_.Get(), &sddv, splatDsv_.GetAddressOf()))) return false;
    D3D11_SHADER_RESOURCE_VIEW_DESC sdsv = {};
    sdsv.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    sdsv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    sdsv.Texture2D.MipLevels = 1;
    if (FAILED(device_->CreateShaderResourceView(splatDepthTex_.Get(), &sdsv, splatDepthSrv_.GetAddressOf()))) return false;
    D3D11_SHADER_RESOURCE_VIEW_DESC ssv = {};
    ssv.Format = DXGI_FORMAT_X24_TYPELESS_G8_UINT;
    ssv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    ssv.Texture2D.MipLevels = 1;
    if (FAILED(device_->CreateShaderResourceView(splatDepthTex_.Get(), &ssv, splatStencilSrv_.GetAddressOf()))) return false;
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
            // Mirror to the VS debugger Output window so the full message
            // stays visible (the modal MessageBox truncates long error lists).
            OutputDebugStringA(msg.c_str());
            OutputDebugStringA("\n");
            MessageBoxA(nullptr, msg.c_str(), entry, MB_ICONERROR);
            return false;
        }
        return true;
    };

    ComPtr<ID3DBlob> vsb, psb, vsbp, psbp, vsbh, psbh, vsbb, psbb;
    if (!compile("vsmain", "vs_5_0", vsb)) return false;
    if (!compile("psmain", "ps_5_0", psb)) return false;
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
    ComPtr<ID3DBlob> vsbM, psbM;
    if (!compile("vsmain_merged", "vs_5_0", vsbM)) return false;
    if (!compile("psmain_merged", "ps_5_0", psbM)) return false;
    ComPtr<ID3DBlob> vsbA, psbA;
    if (!compile("vsmain_atlas", "vs_5_0", vsbA)) return false;
    if (!compile("psmain_atlas", "ps_5_0", psbA)) return false;
    ComPtr<ID3DBlob> csbSp, psbSa, vsbTa, psbTa, psbPo;
    if (!compile("csmain_splat",        "cs_5_0", csbSp)) return false;
    if (!compile("psmain_splat_albedo", "ps_5_0", psbSa)) return false;
    ComPtr<ID3DBlob> psbSc;
    if (!compile("psmain_splat_composite", "ps_5_0", psbSc)) return false;
    ComPtr<ID3DBlob> psbSr;
    if (!compile("psmain_splat_reconstruct", "ps_5_0", psbSr)) return false;
    ComPtr<ID3DBlob> psbSp;
    if (!compile("psmain_splat_albedo_poly", "ps_5_0", psbSp)) return false;
    ComPtr<ID3DBlob> psbPa0;
    if (!compile("psmain_polyvid_alpha0", "ps_5_0", psbPa0)) return false;
    if (!compile("vsmain_taa",          "vs_5_0", vsbTa)) return false;
    if (!compile("psmain_taa",          "ps_5_0", psbTa)) return false;
    if (!compile("psmain_post",         "ps_5_0", psbPo)) return false;
    if (!compile("vsmain_bounds", "vs_5_0", vsbb)) return false;
    if (!compile("psmain_bounds", "ps_5_0", psbb)) return false;
    ComPtr<ID3DBlob> vsbd;
    if (!compile("vsmain_depth", "vs_5_0", vsbd)) return false;
    ComPtr<ID3DBlob> vsbSh;
    if (!compile("vsmain_shadow", "vs_5_0", vsbSh)) return false;
    ComPtr<ID3DBlob> csb, vsbl, psbl;
    if (!compile("csmain_points_cs", "cs_5_0", csb)) return false;
    if (!compile("vsmain_blit",      "vs_5_0", vsbl)) return false;
    if (!compile("psmain_blit",      "ps_5_0", psbl)) return false;

    HRESULT hr = device_->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, vs_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, ps_.GetAddressOf());
    if (FAILED(hr)) return false;
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
    hr = device_->CreateVertexShader(vsbM->GetBufferPointer(), vsbM->GetBufferSize(), nullptr, vsMerged_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbM->GetBufferPointer(), psbM->GetBufferSize(), nullptr, psMerged_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreateVertexShader(vsbA->GetBufferPointer(), vsbA->GetBufferSize(), nullptr, vsAtlas_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbA->GetBufferPointer(), psbA->GetBufferSize(), nullptr, psAtlas_.GetAddressOf());
    if (FAILED(hr)) return false;
    {
        D3D11_INPUT_ELEMENT_DESC ilA[] = {
            { "POSITION", 0, DXGI_FORMAT_R16G16B16A16_UINT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "UV",       0, DXGI_FORMAT_R16G16_UINT,       0, 8, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        };
        hr = device_->CreateInputLayout(ilA, _countof(ilA),
                                        vsbA->GetBufferPointer(), vsbA->GetBufferSize(),
                                        atlasInputLayout_.GetAddressOf());
        if (FAILED(hr)) return false;
    }
    hr = device_->CreateComputeShader(csbSp->GetBufferPointer(), csbSp->GetBufferSize(), nullptr, csSplat_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbSa->GetBufferPointer(), psbSa->GetBufferSize(), nullptr, psSplatAlbedo_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbSc->GetBufferPointer(), psbSc->GetBufferSize(), nullptr, psSplatComposite_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbSr->GetBufferPointer(), psbSr->GetBufferSize(), nullptr, psSplatRecon_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbSp->GetBufferPointer(), psbSp->GetBufferSize(), nullptr, psSplatAlbedoPoly_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbPa0->GetBufferPointer(), psbPa0->GetBufferSize(), nullptr, psPolyVidAlpha0_.GetAddressOf());
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
    D3D11_INPUT_ELEMENT_DESC ilM[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,                            D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR",    0, DXGI_FORMAT_R8G8B8A8_UNORM,  0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    hr = device_->CreateInputLayout(ilM, _countof(ilM),
                                    vsbM->GetBufferPointer(), vsbM->GetBufferSize(),
                                    mergedInputLayout_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreateVertexShader(vsbb->GetBufferPointer(), vsbb->GetBufferSize(), nullptr, vsBounds_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbb->GetBufferPointer(), psbb->GetBufferSize(), nullptr, psBounds_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreateVertexShader(vsbd->GetBufferPointer(), vsbd->GetBufferSize(), nullptr, vsDepth_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreateVertexShader(vsbSh->GetBufferPointer(), vsbSh->GetBufferSize(), nullptr, vsShadow_.GetAddressOf());
    if (FAILED(hr)) return false;
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
                                    vsb->GetBufferPointer(), vsb->GetBufferSize(),
                                    inputLayout_.GetAddressOf());
    if (FAILED(hr)) return false;
    inputLayoutPoly_ = inputLayout_;   // same layout; alias kept for clarity

    // HexSprite: same vertex format, per-instance stepping.
    D3D11_INPUT_ELEMENT_DESC ilHex[] = {
        { "POSITION", 0, DXGI_FORMAT_R16G16B16A16_UINT, 0, 0,                            D3D11_INPUT_PER_INSTANCE_DATA, 1 },
        { "COLOR",    0, DXGI_FORMAT_R8G8B8A8_UINT,     0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
    };
    hr = device_->CreateInputLayout(ilHex, _countof(ilHex),
                                    vsbh->GetBufferPointer(), vsbh->GetBufferSize(),
                                    inputLayoutHex_.GetAddressOf());
    if (FAILED(hr)) return false;

    D3D11_INPUT_ELEMENT_DESC ilB[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    hr = device_->CreateInputLayout(ilB, _countof(ilB),
                                    vsbb->GetBufferPointer(), vsbb->GetBufferSize(),
                                    inputLayoutBounds_.GetAddressOf());
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

    // Unit cube wireframe: 8 corner positions + 24 line-list indices (12 edges).
    float cubeVerts[8][3] = {
        {0,0,0},{1,0,0},{0,1,0},{1,1,0},
        {0,0,1},{1,0,1},{0,1,1},{1,1,1},
    };
    uint32_t cubeLines[24] = {
        0,1, 0,2, 0,4, 1,3, 1,5, 2,3,
        2,6, 3,7, 4,5, 4,6, 5,7, 6,7,
    };
    D3D11_BUFFER_DESC bdb = {};
    bdb.Usage = D3D11_USAGE_IMMUTABLE;
    bdb.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bdb.ByteWidth = sizeof(cubeVerts);
    D3D11_SUBRESOURCE_DATA sdb = { cubeVerts, 0, 0 };
    if (FAILED(device_->CreateBuffer(&bdb, &sdb, boundsVb_.GetAddressOf()))) return false;
    bdb.BindFlags = D3D11_BIND_INDEX_BUFFER;
    bdb.ByteWidth = sizeof(cubeLines);
    sdb.pSysMem = cubeLines;
    if (FAILED(device_->CreateBuffer(&bdb, &sdb, boundsIb_.GetAddressOf()))) return false;

    // PolyVID IB is built per-scene in UploadScene (depends on max chunk size).
    return true;
}

static const uint8_t kCubeTriCornerPattern[36] = {
    1, 3, 7,  1, 7, 5,    // +X
    4, 6, 2,  4, 2, 0,    // -X
    2, 6, 7,  2, 7, 3,    // +Y
    0, 1, 5,  0, 5, 4,    // -Y
    5, 7, 6,  5, 6, 4,    // +Z
    0, 2, 3,  0, 3, 1,    // -Z
};

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

    D3D11_RASTERIZER_DESC rdW = rd;
    rdW.FillMode = D3D11_FILL_WIREFRAME;
    rdW.CullMode = D3D11_CULL_NONE;     // see thin lines from both sides
    if (FAILED(device_->CreateRasterizerState(&rdW, rsWire_.GetAddressOf()))) return false;

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

    D3D11_DEPTH_STENCIL_DESC ddEq = {};
    ddEq.DepthEnable = TRUE;
    ddEq.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;   // prepass already wrote depth
    ddEq.DepthFunc = D3D11_COMPARISON_GREATER_EQUAL;
    if (FAILED(device_->CreateDepthStencilState(&ddEq, dsEqual_.GetAddressOf()))) return false;

    D3D11_BLEND_DESC bsd = {};
    bsd.RenderTarget[0].BlendEnable = FALSE;
    bsd.RenderTarget[0].RenderTargetWriteMask = 0;       // no color writes
    if (FAILED(device_->CreateBlendState(&bsd, bsNoColor_.GetAddressOf()))) return false;

    // Sun shadow map: D32_FLOAT, reverse-Z ortho, back-face culling, depth bias.
    {
        D3D11_TEXTURE2D_DESC td = {};
        td.Width  = shadowSize_;
        td.Height = shadowSize_;
        td.MipLevels = 1; td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R32_TYPELESS;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(device_->CreateTexture2D(&td, nullptr, shadowTex_.GetAddressOf()))) return false;

        D3D11_DEPTH_STENCIL_VIEW_DESC dvd = {};
        dvd.Format = DXGI_FORMAT_D32_FLOAT;
        dvd.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
        if (FAILED(device_->CreateDepthStencilView(shadowTex_.Get(), &dvd, shadowDsv_.GetAddressOf()))) return false;

        D3D11_SHADER_RESOURCE_VIEW_DESC svd = {};
        svd.Format = DXGI_FORMAT_R32_FLOAT;
        svd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        svd.Texture2D.MipLevels = 1;
        if (FAILED(device_->CreateShaderResourceView(shadowTex_.Get(), &svd, shadowSrv_.GetAddressOf()))) return false;

        D3D11_SAMPLER_DESC sd = {};
        sd.Filter = D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
        sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_BORDER;
        sd.BorderColor[0] = sd.BorderColor[1] = sd.BorderColor[2] = sd.BorderColor[3] = 1.0f;
        // Reverse-Z ortho: lit when sample >= ref → use GREATER_EQUAL.
        sd.ComparisonFunc = D3D11_COMPARISON_GREATER_EQUAL;
        if (FAILED(device_->CreateSamplerState(&sd, shadowSamp_.GetAddressOf()))) return false;

        D3D11_RASTERIZER_DESC rsd = {};
        rsd.FillMode = D3D11_FILL_SOLID;
        rsd.CullMode = D3D11_CULL_BACK;
        rsd.DepthClipEnable = TRUE;
        rsd.DepthBias = 0;
        rsd.DepthBiasClamp = 0.0f;
        rsd.SlopeScaledDepthBias = 0.0f;
        if (FAILED(device_->CreateRasterizerState(&rsd, rsShadow_.GetAddressOf()))) return false;
    }

    // Splat stencil-based pipeline. dsSplatPoly: poly draws set stencil=1.
    // dsSplatPoint: splat-marker draws don't touch stencil.
    {
        D3D11_DEPTH_STENCIL_DESC dd = {};
        dd.DepthEnable = TRUE;
        dd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
        dd.DepthFunc = D3D11_COMPARISON_GREATER;
        dd.StencilEnable = TRUE;
        dd.StencilReadMask = 0xFF;
        dd.StencilWriteMask = 0xFF;
        dd.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
        dd.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
        dd.FrontFace.StencilPassOp = D3D11_STENCIL_OP_REPLACE;
        dd.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
        dd.BackFace = dd.FrontFace;
        if (FAILED(device_->CreateDepthStencilState(&dd, dsSplatPoly_.GetAddressOf()))) return false;
    }
    {
        D3D11_DEPTH_STENCIL_DESC dd = {};
        dd.DepthEnable = TRUE;
        dd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
        dd.DepthFunc = D3D11_COMPARISON_GREATER;
        dd.StencilEnable = FALSE;
        if (FAILED(device_->CreateDepthStencilState(&dd, dsSplatPoint_.GetAddressOf()))) return false;
    }
    {
        D3D11_DEPTH_STENCIL_DESC dd = {};
        dd.DepthEnable = TRUE;
        dd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
        dd.DepthFunc = D3D11_COMPARISON_ALWAYS;
        dd.StencilEnable = FALSE;
        if (FAILED(device_->CreateDepthStencilState(&dd, dsAlwaysWrite_.GetAddressOf()))) return false;
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
    vb_.Reset();
    ib_.Reset();
    totalTriangles_ = 0;
    totalVertices_  = 0;
    vbBytes_ = 0;
    ibBytes_ = 0;

    if (scene.vertices.empty()) return;

    D3D11_BUFFER_DESC bd = {};
    bd.Usage = D3D11_USAGE_IMMUTABLE;
    D3D11_SUBRESOURCE_DATA sd = {};

    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bd.ByteWidth = (UINT)(scene.vertices.size() * sizeof(VoxelPolyVertex));
    sd.pSysMem = scene.vertices.data();
    if (FAILED(device_->CreateBuffer(&bd, &sd, vb_.GetAddressOf()))) return;

    std::vector<uint32_t> sharedIb;
    const uint32_t* ibPtr = nullptr;
    size_t ibCount = 0;

    if (scene.indices.empty()) {
        // Shared cube IB: pattern (4f, 4f+1, 4f+2, 4f, 4f+2, 4f+3) per face.
        // Size for the largest chunk so any DrawIndexed fits.
        uint32_t maxIndexCount = 0;
        for (const auto& s : scene.subs)
            maxIndexCount = std::max(maxIndexCount, s.indexCount);
        uint32_t maxFaces = maxIndexCount / 6;
        sharedIb.resize((size_t)maxFaces * 6);
        for (uint32_t f = 0; f < maxFaces; ++f) {
            uint32_t base = f * 4;
            uint32_t* o = sharedIb.data() + f * 6;
            o[0] = base + 0; o[1] = base + 1; o[2] = base + 2;
            o[3] = base + 0; o[4] = base + 2; o[5] = base + 3;
        }
        ibPtr = sharedIb.data();
        ibCount = sharedIb.size();
    } else {
        ibPtr = scene.indices.data();
        ibCount = scene.indices.size();
    }

    bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    bd.ByteWidth = (UINT)(ibCount * sizeof(uint32_t));
    sd.pSysMem = ibPtr;
    if (FAILED(device_->CreateBuffer(&bd, &sd, ib_.GetAddressOf()))) return;

    pointSrv_.Reset();
    if (!scene.pointVertices.empty()) {
        // Vertex buffer (immutable, plain).
        D3D11_BUFFER_DESC pbd = {};
        pbd.Usage = D3D11_USAGE_IMMUTABLE;
        pbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        pbd.ByteWidth = (UINT)(scene.pointVertices.size() * sizeof(Vertex));
        D3D11_SUBRESOURCE_DATA psd = {};
        psd.pSysMem = scene.pointVertices.data();
        device_->CreateBuffer(&pbd, &psd, pointVb_.GetAddressOf());

        // Parallel StructuredBuffer for CS access (same data).
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
    }

    subs_.reserve(scene.subs.size());
    for (const auto& s : scene.subs) {
        GpuSubMesh gs;
        gs.firstIndex = s.firstIndex;
        gs.indexCount = s.indexCount;
        gs.baseVertex = s.baseVertex;
        gs.pointFirst = s.pointFirst;
        gs.pointCount = s.pointCount;
        gs.pointFirstL1 = s.pointFirstL1;
        gs.pointCountL1 = s.pointCountL1;
        gs.pointFirstL2 = s.pointFirstL2;
        gs.pointCountL2 = s.pointCountL2;
        gs.aabbMin[0] = s.aabbMin[0]; gs.aabbMin[1] = s.aabbMin[1]; gs.aabbMin[2] = s.aabbMin[2];
        gs.aabbMax[0] = s.aabbMax[0]; gs.aabbMax[1] = s.aabbMax[1]; gs.aabbMax[2] = s.aabbMax[2];
        gs.chunkBase[0] = s.chunkBase[0]; gs.chunkBase[1] = s.chunkBase[1]; gs.chunkBase[2] = s.chunkBase[2];
        subs_.push_back(gs);
    }
    totalVertices_  = scene.vertices.size();
    totalTriangles_ = 0;
    for (const auto& s : scene.subs) totalTriangles_ += s.indexCount / 3;
    sceneOrigin_[0] = (float)scene.origin[0];
    sceneOrigin_[1] = (float)scene.origin[1];
    sceneOrigin_[2] = (float)scene.origin[2];
    sceneSpan_[0] = scene.aabbMax[0] - scene.aabbMin[0];
    sceneSpan_[1] = scene.aabbMax[1] - scene.aabbMin[1];
    sceneSpan_[2] = scene.aabbMax[2] - scene.aabbMin[2];

    // PolyVID shared IB: per voxel 24 unique vertex slots (6 faces * 4 corners),
    // 36 indices = 6 faces * (3+3). IB[v*36+f*6+k] = v*24 + f*4 + cornerWithinFace[k]
    // where cornerWithinFace pattern is (0,1,2,0,2,3).
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
            polyVidIb_.Reset();
            device_->CreateBuffer(&ibd, &isd, polyVidIb_.GetAddressOf());
        }

        // Billboard IB: 6 indices per voxel, pattern v*4 + {0,1,2,0,2,3}.
        if (maxVox > 0) {
            std::vector<uint32_t> bib((size_t)maxVox * 6);
            uint32_t* dst = bib.data();
            static const uint8_t kQuadIdx[6] = { 0, 1, 2, 0, 2, 3 };
            for (uint32_t v = 0; v < maxVox; ++v) {
                uint32_t base = v * 4;
                for (int k = 0; k < 6; ++k) *dst++ = base + kQuadIdx[k];
            }
            D3D11_BUFFER_DESC ibd2 = {};
            ibd2.Usage = D3D11_USAGE_IMMUTABLE;
            ibd2.BindFlags = D3D11_BIND_INDEX_BUFFER;
            ibd2.ByteWidth = (UINT)(bib.size() * sizeof(uint32_t));
            D3D11_SUBRESOURCE_DATA isd2 = { bib.data(), 0, 0 };
            billboardIb_.Reset();
            device_->CreateBuffer(&ibd2, &isd2, billboardIb_.GetAddressOf());
        }
    }
    vbBytes_ = scene.vertices.size() * sizeof(VoxelPolyVertex);
    ibBytes_ = ibCount * sizeof(uint32_t);
}

void Renderer::UploadMergedMesh(const MergedMesh& mesh)
{
    mergedVb_.Reset();
    mergedIb_.Reset();
    mergedIndexCount_ = 0;
    if (mesh.vertices.empty() || mesh.indices.empty()) return;

    D3D11_BUFFER_DESC bd = {};
    bd.Usage = D3D11_USAGE_IMMUTABLE;
    D3D11_SUBRESOURCE_DATA sd = {};

    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bd.ByteWidth = (UINT)(mesh.vertices.size() * sizeof(MergedVertex));
    sd.pSysMem = mesh.vertices.data();
    if (FAILED(device_->CreateBuffer(&bd, &sd, mergedVb_.GetAddressOf()))) return;

    bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    bd.ByteWidth = (UINT)(mesh.indices.size() * sizeof(uint32_t));
    sd.pSysMem = mesh.indices.data();
    if (FAILED(device_->CreateBuffer(&bd, &sd, mergedIb_.GetAddressOf()))) return;

    mergedIndexCount_ = (uint32_t)mesh.indices.size();
}

void Renderer::UploadAtlasMesh(const AtlasMesh& mesh)
{
    atlasVb_.Reset();
    atlasIb_.Reset();
    atlasTex_.Reset();
    atlasSrv_.Reset();
    atlasIndexCount_ = 0;
    if (mesh.vertices.empty() || mesh.indices.empty()
        || mesh.atlasPixels.empty() || mesh.atlasW == 0 || mesh.atlasH == 0) return;

    D3D11_BUFFER_DESC bd = {};
    bd.Usage = D3D11_USAGE_IMMUTABLE;
    D3D11_SUBRESOURCE_DATA sd = {};

    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bd.ByteWidth = (UINT)(mesh.vertices.size() * sizeof(AtlasVertex));
    sd.pSysMem = mesh.vertices.data();
    if (FAILED(device_->CreateBuffer(&bd, &sd, atlasVb_.GetAddressOf()))) return;

    bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    bd.ByteWidth = (UINT)(mesh.indices.size() * sizeof(uint32_t));
    sd.pSysMem = mesh.indices.data();
    if (FAILED(device_->CreateBuffer(&bd, &sd, atlasIb_.GetAddressOf()))) return;

    D3D11_TEXTURE2D_DESC td = {};
    td.Width = mesh.atlasW;
    td.Height = mesh.atlasH;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_IMMUTABLE;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA tsd = {};
    tsd.pSysMem = mesh.atlasPixels.data();
    tsd.SysMemPitch = mesh.atlasW * 4;
    if (FAILED(device_->CreateTexture2D(&td, &tsd, atlasTex_.GetAddressOf()))) return;
    if (FAILED(device_->CreateShaderResourceView(atlasTex_.Get(), nullptr, atlasSrv_.GetAddressOf()))) return;

    atlasIndexCount_ = (uint32_t)mesh.indices.size();
    atlasOrigin_[0] = (float)mesh.origin[0];
    atlasOrigin_[1] = (float)mesh.origin[1];
    atlasOrigin_[2] = (float)mesh.origin[2];

    atlasSubs_.clear();
    atlasSubByChunk_.clear();
    atlasSubs_.reserve(mesh.chunks.size());
    for (const auto& c : mesh.chunks) {
        GpuAtlasSub s;
        s.cx = c.cx; s.cy = c.cy; s.cz = c.cz;
        for (int d = 0; d < 3; ++d) {
            s.aabbMin[d] = (float)c.aabbMin[d];
            s.aabbMax[d] = (float)c.aabbMax[d];
        }
        s.firstIndex = c.firstIndex;
        s.indexCount = c.indexCount;
        uint64_t key = ((uint64_t)s.cx) | ((uint64_t)s.cy << 20) | ((uint64_t)s.cz << 40);
        atlasSubByChunk_[key] = (uint32_t)atlasSubs_.size();
        atlasSubs_.push_back(s);
    }

    // Use AABB span for grid replication (same convention as MergedMesh).
    sceneSpan_[0] = mesh.aabbMax[0] - mesh.aabbMin[0] + 1.0f;
    sceneSpan_[1] = mesh.aabbMax[1] - mesh.aabbMin[1] + 1.0f;
    sceneSpan_[2] = mesh.aabbMax[2] - mesh.aabbMin[2] + 1.0f;
}

void Renderer::BeginFrame(float clear[4])
{
    lastClear_[0] = clear[0];
    lastClear_[1] = clear[1];
    lastClear_[2] = clear[2];
    lastClear_[3] = clear[3];
    ID3D11RenderTargetView* active = (msaaSamples_ > 1) ? msaaRtv_.Get() : rtv_.Get();
    ID3D11RenderTargetView* rtvs[] = { active };
    ctx_->OMSetRenderTargets(1, rtvs, dsv_.Get());
    ctx_->ClearRenderTargetView(active, clear);
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
    // Unpack args into locals — many spots in this function clamp/mutate
    // values, and dragging const refs through ~1500 lines is verbose. Locals
    // are cheaper than refactoring every reference.
    ShadingMode    mode             = args.mode;
    int            gridSize         = args.gridSize;
    RenderTech     techClose        = args.techClose;
    RenderTech     techFar          = args.techFar;
    DataSet        dataset          = args.dataset;
    bool           showChunkBounds  = args.showChunkBounds;
    bool           zPrepass         = args.zPrepass;
    PointLighting  pointLight       = args.pointLight;
    PointLod       pointLod         = args.pointLod;
    float          pointLodScale    = args.pointLodScale;
    bool           splatFilter      = args.splatFilter;
    const float*   fogColor         = args.fogColor;
    float          fogDensity       = args.fogDensity;
    float          heightFogDensity = args.heightFogDensity;
    float          heightFogFalloff = args.heightFogFalloff;
    float          heightFogStart   = args.heightFogStart;
    float          hybridThreshold  = args.hybridThreshold;
    bool           wireframe        = args.wireframe;
    int            splatRadius      = args.splatRadius;
    bool           taa              = args.taa;
    float          sunIntensity     = args.sunIntensity;
    float          exposure         = args.exposure;
    float          roughness        = args.roughness;

    hybridThreshold = std::max(hybridThreshold, 0.01f);

    // techClose is treated as `tech` for the existing single-tech code paths;
    // techFar (if not None) picks a different tech for low-ppv chunks via the
    // per-chunk dispatch below. Splat-RT pipeline triggers if either side
    // uses Splat / SplatHybrid.
    RenderTech tech = techClose;
    const bool anySplat = (techClose == RenderTech::Splat) ||
                          (techClose == RenderTech::SplatHybrid) ||
                          (techFar   == RenderTech::Splat) ||
                          (techFar   == RenderTech::SplatHybrid);
    const bool anyPoly  = (techClose == RenderTech::PolygonBased) ||
                          (techClose == RenderTech::Hybrid) ||
                          (techFar   == RenderTech::PolygonBased) ||
                          (techFar   == RenderTech::Hybrid);
    (void)anyPoly;

    // TAA disabled when MSAA on (composite reads non-MSAA depth/color).
    if (msaaSamples_ > 1) taa = false;

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
    // Un-jittered VP is what gets stored as "prev" for next frame's TAA reproject:
    // history is the merged output at pixel grid (jitter already undone in PS),
    // so prev reproject must use unjittered.
    hlslpp::float4x4 vpUnjittered = hlslpp::mul(v, p);
    if (taa) {
        // Inject sub-pixel jitter into proj. clip = view*proj, clip.w = view.z;
        // adding (jitter * view.z) to clip.xy after divide gives NDC offset = jitter.
        // Row index 2 (view.z multiplier) of row-major proj.
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
    // Post (sky + sharpen + tonemap) reads scene RT after main draws. TAA
    // composite sits between when enabled. Sky pixels marked by alpha=0.
    const bool postEnabled = (taaSceneRtv_ && psPost_);
    const bool useMsaa = (msaaSamples_ > 1);
    postWroteBackbuf_ = false;
    if (postEnabled) {
        float alpha0[4] = { lastClear_[0], lastClear_[1], lastClear_[2], 0.0f };
        ID3D11RenderTargetView* sceneRtv = useMsaa ? msaaRtv_.Get() : taaSceneRtv_.Get();
        ctx_->ClearRenderTargetView(sceneRtv, alpha0);
        ID3D11RenderTargetView* rtvs[] = { sceneRtv };
        ctx_->OMSetRenderTargets(1, rtvs, dsv_.Get());
    }

    {
        CBPerFrame cb;
        hlslpp::store(cb.viewProj, vp);
        hlslpp::store(cb.camPos, cam.position);
        // Wireframe forces FlatColor shading so lines aren't black (the lit PS
        // computes normals via ddx/ddy which degenerates on wireframe edges).
        cb.mode = (float)(wireframe ? (int)ShadingMode::FlatColor : (int)mode);
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
        cb._pad2[0] = (float)splatRadius;       // gSplatRadius (CS dilation R)
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
        // Sun shadow: row-major LH ortho around scene AABB, reverse-Z. Camera
        // at sceneCenter + sunDir * dist; looks toward sceneCenter.
        {
        float sd[3] = { args.sunDir[0], args.sunDir[1], args.sunDir[2] };
        {
            float sl = sqrtf(sd[0]*sd[0] + sd[1]*sd[1] + sd[2]*sd[2]);
            if (sl < 1e-6f) sl = 1.0f;
            sd[0] /= sl; sd[1] /= sl; sd[2] /= sl;
        }
        float center[3] = { sceneOrigin_[0] + sceneSpan_[0] * 0.5f,
                            sceneOrigin_[1] + sceneSpan_[1] * 0.5f,
                            sceneOrigin_[2] + sceneSpan_[2] * 0.5f };
        float diag = 0.5f * sqrtf(sceneSpan_[0]*sceneSpan_[0]
                                 + sceneSpan_[1]*sceneSpan_[1]
                                 + sceneSpan_[2]*sceneSpan_[2]);
        if (diag < 1.0f) diag = 1.0f;
        float sdist = diag * 2.0f;
        float eye[3] = { center[0] + sd[0] * sdist,
                         center[1] + sd[1] * sdist,
                         center[2] + sd[2] * sdist };
        float fwd[3] = { center[0] - eye[0], center[1] - eye[1], center[2] - eye[2] };
        {
            float fl = sqrtf(fwd[0]*fwd[0] + fwd[1]*fwd[1] + fwd[2]*fwd[2]);
            if (fl < 1e-6f) fl = 1.0f;
            fwd[0] /= fl; fwd[1] /= fl; fwd[2] /= fl;
        }
        float upRef[3] = { 0.0f, 1.0f, 0.0f };
        if (fabsf(fwd[1]) > 0.95f) { upRef[0] = 1.0f; upRef[1] = 0.0f; }
        // right = normalize(cross(up, fwd)), up = cross(fwd, right) — LH
        float right[3] = { upRef[1]*fwd[2] - upRef[2]*fwd[1],
                           upRef[2]*fwd[0] - upRef[0]*fwd[2],
                           upRef[0]*fwd[1] - upRef[1]*fwd[0] };
        {
            float rl = sqrtf(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
            if (rl < 1e-6f) rl = 1.0f;
            right[0] /= rl; right[1] /= rl; right[2] /= rl;
        }
        float up[3] = { fwd[1]*right[2] - fwd[2]*right[1],
                        fwd[2]*right[0] - fwd[0]*right[2],
                        fwd[0]*right[1] - fwd[1]*right[0] };
        // View matrix (row-major LH): rows = right/up/fwd; last col = -dot(axis, eye)
        float view[16] = {
            right[0], up[0], fwd[0], 0.0f,
            right[1], up[1], fwd[1], 0.0f,
            right[2], up[2], fwd[2], 0.0f,
            -(right[0]*eye[0] + right[1]*eye[1] + right[2]*eye[2]),
            -(up[0]   *eye[0] + up[1]   *eye[1] + up[2]   *eye[2]),
            -(fwd[0]  *eye[0] + fwd[1]  *eye[1] + fwd[2]  *eye[2]),
            1.0f };
        // Reverse-Z ortho LH: z=near->1, z=far->0. depth' = (far - z)/(far - near)
        float oNear = 0.0f;
        float oFar  = 2.0f * sdist;
        float ow = diag * 1.2f;       // half extent in X/Y
        float oh = diag * 1.2f;
        float proj[16] = {
            1.0f / ow, 0.0f,      0.0f,                       0.0f,
            0.0f,      1.0f / oh, 0.0f,                       0.0f,
            0.0f,      0.0f,     -1.0f / (oFar - oNear),      0.0f,
            0.0f,      0.0f,      oFar  / (oFar - oNear),     1.0f
        };
        // svp = view * proj (row-major LH composition).
        float svp[16];
        for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += view[r*4 + k] * proj[k*4 + c];
            svp[r*4 + c] = s;
        }
        for (int i = 0; i < 16; ++i) cb.sunViewProj[i] = svp[i];
        }
        cb.shadowBias    = 0.0005f;
        cb.shadowMapSize = (float)shadowSize_;
        cb.shadowEnable  = (args.sunShadows && shadowDsv_) ? 1.0f : 0.0f;
        cb.sunIntensity  = sunIntensity;
        cb.exposure      = exposure;
        cb.roughness     = roughness;
        cb.colorizeClusters = args.colorizeClusters ? 1.0f : 0.0f;
        cb._padExp[0] = 0.0f;
        memcpy(m.pData, &cb, sizeof(cb));
    }
    ctx_->Unmap(cbPerFrame_.Get(), 0);

    ctx_->IASetInputLayout(inputLayout_.Get());
    ID3D11Buffer* cbs[] = { cbPerFrame_.Get(), cbPerChunk_.Get() };
    ctx_->VSSetConstantBuffers(0, 2, cbs);
    ctx_->PSSetConstantBuffers(0, 2, cbs);
    ctx_->RSSetState(wireframe ? rsWire_.Get() : rsSolid_.Get());
    ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);

    if (!vb_ || !ib_) return;

    // Sun shadow pass. Renders all polygonal chunks once (no grid replication;
    // the scene AABB drives the ortho frustum). Binds vsShadow + null PS into
    // the shadow DSV.
    if (args.sunShadows && shadowDsv_ && vsShadow_ && dataset == DataSet::Full)
    {
        MICROPROFILE_SCOPEGPUI("ShadowMap", 0xff8080ff);
        D3D11_VIEWPORT svp = { 0.0f, 0.0f, (float)shadowSize_, (float)shadowSize_, 0.0f, 1.0f };
        ctx_->RSSetViewports(1, &svp);
        ctx_->ClearDepthStencilView(shadowDsv_.Get(), D3D11_CLEAR_DEPTH, 0.0f, 0);
        ID3D11RenderTargetView* nullRtv[] = { nullptr };
        ctx_->OMSetRenderTargets(1, nullRtv, shadowDsv_.Get());
        ctx_->OMSetBlendState(bsNoColor_.Get(), nullptr, 0xFFFFFFFFu);
        ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
        ctx_->RSSetState(rsShadow_.Get());
        ctx_->IASetInputLayout(inputLayoutPoly_.Get());
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx_->VSSetShader(vsShadow_.Get(), nullptr, 0);
        ctx_->PSSetShader(nullptr, nullptr, 0);
        UINT polyStride = sizeof(VoxelPolyVertex);
        UINT polyOff = 0;
        ID3D11Buffer* vbs[] = { vb_.Get() };
        ctx_->IASetVertexBuffers(0, 1, vbs, &polyStride, &polyOff);
        ctx_->IASetIndexBuffer(ib_.Get(), DXGI_FORMAT_R32_UINT, 0);

        // One chunkBase per group (chunks within a sub-list share scene origin).
        auto setBase = [&](float ox, float oz) {
            D3D11_MAPPED_SUBRESOURCE mm;
            ctx_->Map(cbPerChunk_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mm);
            CBPerChunk cc = {};
            cc.chunkBase[0] = sceneOrigin_[0] + ox;
            cc.chunkBase[1] = sceneOrigin_[1];
            cc.chunkBase[2] = sceneOrigin_[2] + oz;
            memcpy(mm.pData, &cc, sizeof(cc));
            ctx_->Unmap(cbPerChunk_.Get(), 0);
        };
        setBase(0.0f, 0.0f);
        for (const auto& gs : subs_) {
            ctx_->DrawIndexed(gs.indexCount, gs.firstIndex, 0);
        }

        // Restore main pass state.
        D3D11_VIEWPORT mvp = { 0.0f, 0.0f, (float)width_, (float)height_, 0.0f, 1.0f };
        ctx_->RSSetViewports(1, &mvp);
        ctx_->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFFu);
        ctx_->RSSetState(wireframe ? rsWire_.Get() : rsSolid_.Get());

        ID3D11RenderTargetView* sceneRtv =
            (msaaSamples_ > 1) ? msaaRtv_.Get()
                               : (taaSceneRtv_ && psPost_) ? taaSceneRtv_.Get() : rtv_.Get();
        ID3D11RenderTargetView* rtvs[] = { sceneRtv };
        ctx_->OMSetRenderTargets(1, rtvs, dsv_.Get());

        // Bind shadow SRV+sampler for lit PS + splat CS.
        ID3D11ShaderResourceView* shadowSrvs[] = { shadowSrv_.Get() };
        ctx_->PSSetShaderResources(10, 1, shadowSrvs);
        ID3D11SamplerState*       shadowSmps[] = { shadowSamp_.Get() };
        ctx_->PSSetSamplers(2, 1, shadowSmps);
        ctx_->CSSetShaderResources(10, 1, shadowSrvs);
        ctx_->CSSetSamplers(2, 1, shadowSmps);
    }

    // Extract 6 frustum planes from row-major viewProj.
    // Plane equation: a*x + b*y + c*z + d >= 0 means inside.
    float M[16];
    hlslpp::store(M, vp);
    float planes[6][4];
    // Left: clip.x + clip.w >= 0
    planes[0][0] = M[0] + M[3];  planes[0][1] = M[4] + M[7];
    planes[0][2] = M[8] + M[11]; planes[0][3] = M[12] + M[15];
    // Right: clip.w - clip.x >= 0
    planes[1][0] = M[3] - M[0];  planes[1][1] = M[7] - M[4];
    planes[1][2] = M[11] - M[8]; planes[1][3] = M[15] - M[12];
    // Bottom: clip.y + clip.w >= 0
    planes[2][0] = M[1] + M[3];  planes[2][1] = M[5] + M[7];
    planes[2][2] = M[9] + M[11]; planes[2][3] = M[13] + M[15];
    // Top: clip.w - clip.y >= 0
    planes[3][0] = M[3] - M[1];  planes[3][1] = M[7] - M[5];
    planes[3][2] = M[11] - M[9]; planes[3][3] = M[15] - M[13];
    // Near: clip.z >= 0 (DX zclip::zero)
    planes[4][0] = M[2];         planes[4][1] = M[6];
    planes[4][2] = M[10];        planes[4][3] = M[14];
    // Far: clip.w - clip.z >= 0
    planes[5][0] = M[3] - M[2];  planes[5][1] = M[7] - M[6];
    planes[5][2] = M[11] - M[10];planes[5][3] = M[15] - M[14];

    // Collect visible chunk instances + decide path per chunk (Hybrid).
    struct Job {
        const GpuSubMesh* gs;
        float ox, oz;
        PointLod lod;     // effective LOD (resolved if Auto)
    };
    static thread_local std::vector<Job> polyJobs;
    static thread_local std::vector<Job> pointJobs;
    static thread_local std::vector<Job> hexJobs;
    static thread_local std::vector<Job> csJobs;
    static thread_local std::vector<Job> vidJobs;
    static thread_local std::vector<Job> axisJobs;
    static thread_local std::vector<Job> axisInstJobs;
    static thread_local std::vector<Job> splatJobs;     // chunks routed to Splat tech
    static thread_local std::vector<Job> billJobs;
    static thread_local std::vector<Job> billTriJobs;
    polyJobs.clear();
    pointJobs.clear();
    hexJobs.clear();
    csJobs.clear();
    vidJobs.clear();
    axisJobs.clear();
    axisInstJobs.clear();
    splatJobs.clear();
    billJobs.clear();
    billTriJobs.clear();

    // For hybrid: voxel projects to pixelsPerVoxel = focalPx / distance.
    // Use 1.0 px as cutoff.
    float fovRad = cam.fovDeg * 3.14159265358979f / 180.0f;
    float focalPx = (float)height_ / (2.0f * tanf(fovRad * 0.5f));
    hlslpp::float3 camPosHl = cam.position;
    float camP[3]; hlslpp::store(camP, camPosHl);

    lastDrawn_ = 0;
    lastDrawnTris_ = 0;
    lastPolyTris_ = 0;
    lastPolyVerts_ = 0;
    lastPointCount_ = 0;
    const float spanX = sceneSpan_[0];
    const float spanZ = sceneSpan_[2];

    // Shared AABB-vs-camera distance: nearest point of [aMin..aMax] to camP.
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
    // Shared 6-plane frustum cull of AABB.
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
    // True iff every corner of the AABB is on the inside of every frustum
    // plane (lets the caller skip per-cluster cull when the whole mesh fits).
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

            // Whole-mesh AABB tests first: skip the cell entirely if outside;
            // skip per-cluster cull if the entire mesh fits inside the frustum.
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

                // Resolve LOD (Auto -> per-chunk by projected voxel size).
                // Splat-aware LOD: CS dilate of radius R fills R-pixel gaps.
                // A cluster of size S voxels has screen spacing S*ppv pixels;
                // splats stitch seamlessly when S*ppv <= R. Pick the LARGEST
                // cluster that still fits, since coarser = fewer points.
                //   L0 (S=1)  needs ppv <= R          ( ppv > R   -> close poly)
                //   L1 (S=2)  needs ppv <= R/2
                //   L2 (S=4)  needs ppv <= R/4        (smaller and gaps remain)
                // Thresholds: choose largest LOD whose spacing fits R.
                //   ppv >= R/2  -> L0   (R/2 < ppv <= R range)
                //   ppv >= R/4  -> L1
                //   else        -> L2
                // pointLodScale biases (>1 = stay coarser longer).
                const bool splatFar = (techFar == RenderTech::Splat
                                    || techClose == RenderTech::Splat);
                PointLod effLod = pointLod;
                float ppvHere = focalPx / chunkDist(aMin, aMax);
                if (pointLod == PointLod::Auto) {
                    float t0, t1;
                    if (splatFar) {
                        float R = (float)std::max(1, splatRadius);
                        t0 = (R * 0.5f) / pointLodScale;
                        t1 = (R * 0.25f) / pointLodScale;
                    } else {
                        t0 = 1.333f / pointLodScale;
                        t1 = 0.333f / pointLodScale;
                    }
                    if      (ppvHere >= t0) effLod = PointLod::L0;
                    else if (ppvHere >= t1) effLod = PointLod::L1;
                    else                    effLod = PointLod::L2;
                }
                Job j{ &gs, ox, oz, effLod };
                // If a Far tech is set, pick Close vs Far per chunk by ppv.
                // Splat-far switch derived from splatRadius/res/fov: when a
                // single voxel covers fewer than ~1 pixel (after dilate), the
                // splat tech kicks in for that chunk.
                if (techFar != RenderTech::None) {
                    float ppv = ppvHere;
                    // Switch to far tech when projected voxel coverage drops
                    // below the threshold. For splat-far, the CS dilate of
                    // radius R fills R-pixel gaps, so splat is appropriate when
                    // a voxel covers fewer than R pixels (ppv < R).
                    float switchPpv = hybridThreshold;   // default = 1.0
                    if (splatFar) {
                        float R = (float)std::max(1, splatRadius);
                        switchPpv = R;
                    }
                    RenderTech jt = (ppv >= switchPpv) ? techClose : techFar;
                    if      (jt == RenderTech::Points)        pointJobs.push_back(j);
                    else if (jt == RenderTech::PolygonBased)  polyJobs.push_back(j);
                    else if (jt == RenderTech::HexSprite)     hexJobs.push_back(j);
                    else if (jt == RenderTech::PointCS)       csJobs.push_back(j);
                    else if (jt == RenderTech::PolyVID)       vidJobs.push_back(j);
                    else if (jt == RenderTech::PolyAxis)      axisJobs.push_back(j);
                    else if (jt == RenderTech::PolyAxisInstanced) axisInstJobs.push_back(j);
                    else if (jt == RenderTech::Billboard)     billJobs.push_back(j);
                    else if (jt == RenderTech::BillboardTri)  billTriJobs.push_back(j);
                    else if (jt == RenderTech::Splat)         splatJobs.push_back(j);
                    continue;
                }
                if (tech == RenderTech::Points) {
                    pointJobs.push_back(j);
                } else if (tech == RenderTech::PolygonBased) {
                    polyJobs.push_back(j);
                } else if (tech == RenderTech::HexSprite) {
                    hexJobs.push_back(j);
                } else if (tech == RenderTech::PointCS) {
                    csJobs.push_back(j);
                } else if (tech == RenderTech::PolyVID) {
                    vidJobs.push_back(j);
                } else if (tech == RenderTech::PolyAxis) {
                    axisJobs.push_back(j);
                } else if (tech == RenderTech::PolyAxisInstanced) {
                    axisInstJobs.push_back(j);
                } else if (tech == RenderTech::Billboard) {
                    billJobs.push_back(j);
                } else if (tech == RenderTech::BillboardTri) {
                    billTriJobs.push_back(j);
                } else if (tech == RenderTech::Splat) {
                    splatJobs.push_back(j);
                } else if (tech == RenderTech::SplatHybrid) {
                    // Polygons close, splats far. Both consumed by the Splat block.
                    float ppv = focalPx / chunkDist(aMin, aMax);
                    if (ppv >= hybridThreshold) polyJobs.push_back(j);
                    else                        splatJobs.push_back(j);
                } else if (tech == RenderTech::Hybrid2) {
                    // Polys close (dataset-driven), BillboardTri far.
                    float ppv = focalPx / chunkDist(aMin, aMax);
                    if (ppv >= hybridThreshold) polyJobs.push_back(j);
                    else                        billTriJobs.push_back(j);
                } else {
                    // Hybrid: polys close (dataset-driven), points far.
                    float ppv = focalPx / chunkDist(aMin, aMax);
                    if (ppv < hybridThreshold) pointJobs.push_back(j);
                    else                       polyJobs.push_back(j);
                }
            }
        }
    }

    UINT stride = sizeof(Vertex);
    UINT offset = 0;

    // Colorize-clusters debug: 1 green (close poly tech), 2/3/4 red/orange/yellow
    // for L0/L1/L2 (points or splat). Lambdas below write `curTint` into the
    // per-chunk CB; each draw block sets curTint before its loop.
    uint32_t curTint = 0;
    auto lodToIdx = [](PointLod l) -> uint32_t {
        if (l == PointLod::L2) return 2u;
        if (l == PointLod::L1) return 1u;
        return 0u;
    };
    auto issueChunkCbEx = [&](const Job& j, float padVal, uint32_t voxelBase) {
        // All vertex positions are scene-relative now; chunkBase = scene origin
        // + grid offset (same for every chunk in a grid cell).
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

    // Per-job LOD resolution.
    auto lodToHalfExtent = [](PointLod l) -> float {
        if (l == PointLod::L2) return 2.0f;
        if (l == PointLod::L1) return 1.0f;
        return 0.5f;
    };
    auto pointRangeForJob = [](const Job& j, uint32_t& first, uint32_t& count) {
        if      (j.lod == PointLod::L1) { first = j.gs->pointFirstL1; count = j.gs->pointCountL1; }
        else if (j.lod == PointLod::L2) { first = j.gs->pointFirstL2; count = j.gs->pointCountL2; }
        else                            { first = j.gs->pointFirst;  count = j.gs->pointCount;  }
    };

    const bool doPrepass = zPrepass &&
        (tech == RenderTech::PolygonBased || tech == RenderTech::Hybrid) &&
        !polyJobs.empty();

    if (doPrepass) {
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx_->VSSetShader(vsDepth_.Get(), nullptr, 0);
        ctx_->PSSetShader(nullptr, nullptr, 0);
        ctx_->OMSetBlendState(bsNoColor_.Get(), nullptr, 0xFFFFFFFFu);
        ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
        ID3D11Buffer* vbs[] = { vb_.Get() };
        ctx_->IASetVertexBuffers(0, 1, vbs, &stride, &offset);
        ctx_->IASetIndexBuffer(ib_.Get(), DXGI_FORMAT_R32_UINT, 0);
        for (const auto& j : polyJobs) {
            issueChunkCb(j);
            ctx_->DrawIndexed(j.gs->indexCount, j.gs->firstIndex, j.gs->baseVertex);
        }
        ctx_->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFFu);
    }

    // Unified poly draw: route polyJobs through the dataset's polygon source.
    // Used directly by PolygonBased / Hybrid / Hybrid2, and re-invoked inside
    // the Splat block for SplatHybrid (into the splat RT).
    // Set gChunkBase to scene-origin + grid offset (shared by all chunks in a
    // grid cell since their vertices are scene-relative).
    auto setSceneChunkBase = [&](float ox, float oz) {
        D3D11_MAPPED_SUBRESOURCE mm;
        ctx_->Map(cbPerChunk_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mm);
        CBPerChunk cc = {};
        cc.chunkBase[0] = sceneOrigin_[0] + ox;
        cc.chunkBase[1] = sceneOrigin_[1];
        cc.chunkBase[2] = sceneOrigin_[2] + oz;
        cc.chunkTint    = curTint;
        memcpy(mm.pData, &cc, sizeof(cc));
        ctx_->Unmap(cbPerChunk_.Get(), 0);
    };

    auto drawPolyJobs = [&](const std::vector<Job>& jobs) {
        if (jobs.empty()) return;
        MICROPROFILE_SCOPEGPUI("Poly", 0xff60d0ff);
        curTint = 1;            // green: close poly tech
        if (dataset == DataSet::Full) {
            UINT polyStride = sizeof(VoxelPolyVertex);
            UINT polyOff = 0;
            ctx_->IASetInputLayout(inputLayoutPoly_.Get());
            ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx_->VSSetShader(vs_.Get(), nullptr, 0);
            ctx_->PSSetShader(ps_.Get(), nullptr, 0);
            ctx_->OMSetDepthStencilState(doPrepass ? dsEqual_.Get() : dsTest_.Get(), 0);
            ID3D11Buffer* vbs[] = { vb_.Get() };
            ctx_->IASetVertexBuffers(0, 1, vbs, &polyStride, &polyOff);
            ctx_->IASetIndexBuffer(ib_.Get(), DXGI_FORMAT_R32_UINT, 0);

            // Jobs come in subs_ order, grouped per grid cell. Walk groups; in
            // each group, merge contiguous-firstIndex jobs into single
            // DrawIndexed (cluster span batching).
            size_t i = 0;
            while (i < jobs.size()) {
                float ox = jobs[i].ox, oz = jobs[i].oz;
                setSceneChunkBase(ox, oz);
                while (i < jobs.size() && jobs[i].ox == ox && jobs[i].oz == oz) {
                    uint32_t spanFirst = jobs[i].gs->firstIndex;
                    uint32_t spanEnd   = spanFirst + jobs[i].gs->indexCount;
                    ++i;
                    while (i < jobs.size() && jobs[i].ox == ox && jobs[i].oz == oz
                           && jobs[i].gs->firstIndex == spanEnd) {
                        spanEnd += jobs[i].gs->indexCount;
                        ++i;
                    }
                    uint32_t spanCount = spanEnd - spanFirst;
                    ctx_->DrawIndexed(spanCount, spanFirst, 0);
                    ++lastDrawn_;
                    lastDrawnTris_ += spanCount / 3;
                    lastPolyTris_  += spanCount / 3;
                    lastPolyVerts_ += spanCount; // upper-bound on referenced verts
                }
            }
            ctx_->IASetInputLayout(inputLayout_.Get());
            if (doPrepass) ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
        } else if (dataset == DataSet::Merged && mergedVb_ && mergedIb_) {
            // MSH1 is a single mesh; no per-chunk path. Draw once per grid cell.
            ctx_->IASetInputLayout(mergedInputLayout_.Get());
            ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx_->VSSetShader(vsMerged_.Get(), nullptr, 0);
            ctx_->PSSetShader(psMerged_.Get(), nullptr, 0);
            ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
            UINT stridM = sizeof(MergedVertex);
            UINT offsM = 0;
            ID3D11Buffer* vbsM[] = { mergedVb_.Get() };
            ctx_->IASetVertexBuffers(0, 1, vbsM, &stridM, &offsM);
            ctx_->IASetIndexBuffer(mergedIb_.Get(), DXGI_FORMAT_R32_UINT, 0);
            for (int gz = 0; gz < gridSize; ++gz) {
                for (int gx = 0; gx < gridSize; ++gx) {
                    D3D11_MAPPED_SUBRESOURCE mm;
                    ctx_->Map(cbPerChunk_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mm);
                    CBPerChunk cc = {};
                    cc.chunkBase[0] = gx * sceneSpan_[0];
                    cc.chunkBase[1] = 0.0f;
                    cc.chunkBase[2] = gz * sceneSpan_[2];
                    memcpy(mm.pData, &cc, sizeof(cc));
                    ctx_->Unmap(cbPerChunk_.Get(), 0);
                    ctx_->DrawIndexed(mergedIndexCount_, 0, 0);
                    ++lastDrawn_;
                    lastDrawnTris_ += mergedIndexCount_ / 3;
                }
            }
            ctx_->IASetInputLayout(inputLayout_.Get());
        } else if (dataset == DataSet::Reduced && atlasVb_ && atlasIb_ && atlasSrv_) {
            // Per-chunk atlas mesh. Look up atlas sub for each polyJob, then
            // batch contiguous (in atlas IB) visible subs into single DrawIndexed.
            ctx_->IASetInputLayout(atlasInputLayout_.Get());
            ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx_->VSSetShader(vsAtlas_.Get(), nullptr, 0);
            ctx_->PSSetShader(psAtlas_.Get(), nullptr, 0);
            ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
            UINT stridA = sizeof(AtlasVertex);
            UINT offsA = 0;
            ID3D11Buffer* vbsA[] = { atlasVb_.Get() };
            ctx_->IASetVertexBuffers(0, 1, vbsA, &stridA, &offsA);
            ctx_->IASetIndexBuffer(atlasIb_.Get(), DXGI_FORMAT_R32_UINT, 0);
            ID3D11ShaderResourceView* srvs[] = { nullptr, nullptr, nullptr, atlasSrv_.Get() };
            ctx_->PSSetShaderResources(0, 4, srvs);

            auto subForJob = [&](const Job& j) -> const GpuAtlasSub* {
                int cx = (int)((j.gs->chunkBase[0] - atlasOrigin_[0]) / (float)chunkDim_);
                int cy = (int)((j.gs->chunkBase[1] - atlasOrigin_[1]) / (float)chunkDim_);
                int cz = (int)((j.gs->chunkBase[2] - atlasOrigin_[2]) / (float)chunkDim_);
                uint64_t key = ((uint64_t)(uint16_t)cx)
                             | ((uint64_t)(uint16_t)cy << 20)
                             | ((uint64_t)(uint16_t)cz << 40);
                auto it = atlasSubByChunk_.find(key);
                if (it == atlasSubByChunk_.end()) return nullptr;
                const GpuAtlasSub& s = atlasSubs_[it->second];
                return s.indexCount ? &s : nullptr;
            };

            size_t i = 0;
            while (i < jobs.size()) {
                float ox = jobs[i].ox, oz = jobs[i].oz;
                D3D11_MAPPED_SUBRESOURCE mm;
                ctx_->Map(cbPerChunk_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mm);
                CBPerChunk cc = {};
                cc.chunkBase[0] = atlasOrigin_[0] + ox;
                cc.chunkBase[1] = atlasOrigin_[1];
                cc.chunkBase[2] = atlasOrigin_[2] + oz;
                memcpy(mm.pData, &cc, sizeof(cc));
                ctx_->Unmap(cbPerChunk_.Get(), 0);

                while (i < jobs.size() && jobs[i].ox == ox && jobs[i].oz == oz) {
                    const GpuAtlasSub* s0 = subForJob(jobs[i]);
                    if (!s0) { ++i; continue; }
                    uint32_t spanFirst = s0->firstIndex;
                    uint32_t spanEnd   = spanFirst + s0->indexCount;
                    ++i;
                    while (i < jobs.size() && jobs[i].ox == ox && jobs[i].oz == oz) {
                        const GpuAtlasSub* sn = subForJob(jobs[i]);
                        if (!sn || sn->firstIndex != spanEnd) break;
                        spanEnd += sn->indexCount;
                        ++i;
                    }
                    uint32_t spanCount = spanEnd - spanFirst;
                    ctx_->DrawIndexed(spanCount, spanFirst, 0);
                    ++lastDrawn_;
                    lastDrawnTris_ += spanCount / 3;
                }
            }
            ID3D11ShaderResourceView* nullSrvs[] = { nullptr, nullptr, nullptr, nullptr };
            ctx_->PSSetShaderResources(0, 4, nullSrvs);
            ctx_->IASetInputLayout(inputLayout_.Get());
        }
    };

    // Compositor mode: close tech draws to main RT first; splat output is
    // composited on top via alpha-mask PS (discards bg pixels), so close
    // content (any tech — Poly, PolyAxis, PolyVID, etc.) is preserved.
    const bool splatFarCompositor = (techFar == RenderTech::Splat
                                  && techClose != RenderTech::Splat
                                  && techClose != RenderTech::SplatHybrid);
    if (tech != RenderTech::SplatHybrid) {
        drawPolyJobs(polyJobs);
    }
    // Point pass: batch contiguous chunks within same (grid cell, LOD) into
    // single Draw. Possible now that point positions are scene-relative and
    // L0/L1/L2 are stored in contiguous blocks (so adjacent same-LOD chunks
    // have adjacent first-vertex offsets).
    if (!pointJobs.empty() && pointVb_ && tech != RenderTech::Splat && tech != RenderTech::SplatHybrid) {
        MICROPROFILE_SCOPEGPUI("Points", 0xff80ff80);
        curTint = 2;            // points are tinted per-LOD; overridden in span loop below
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
        ctx_->VSSetShader(vsPoints_.Get(), nullptr, 0);
        ctx_->PSSetShader(pointLight == PointLighting::Simple
                              ? psPointsSimple_.Get()
                              : psPoints_.Get(), nullptr, 0);
        ID3D11Buffer* vbs[] = { pointVb_.Get() };
        ctx_->IASetVertexBuffers(0, 1, vbs, &stride, &offset);

        size_t i = 0;
        while (i < pointJobs.size()) {
            float ox = pointJobs[i].ox, oz = pointJobs[i].oz;
            PointLod lod = pointJobs[i].lod;
            curTint = 2u + lodToIdx(lod);
            issueChunkCbEx(pointJobs[i], lodToHalfExtent(lod), 0u);
            while (i < pointJobs.size()
                   && pointJobs[i].ox == ox && pointJobs[i].oz == oz
                   && pointJobs[i].lod == lod) {
                uint32_t spanFirst, count0;
                pointRangeForJob(pointJobs[i], spanFirst, count0);
                if (count0 == 0) { ++i; continue; }
                uint32_t spanEnd = spanFirst + count0;
                ++i;
                while (i < pointJobs.size()
                       && pointJobs[i].ox == ox && pointJobs[i].oz == oz
                       && pointJobs[i].lod == lod) {
                    uint32_t f, c;
                    pointRangeForJob(pointJobs[i], f, c);
                    if (f != spanEnd) break;
                    spanEnd += c;
                    ++i;
                }
                uint32_t spanCount = spanEnd - spanFirst;
                ctx_->Draw(spanCount, spanFirst);
                ++lastDrawn_;
                lastDrawnTris_ += spanCount;
                lastPointCount_ += spanCount;
            }
        }
    }
    if (!csJobs.empty() && pointVb_ && pointSrv_ && csColorUav_) {
        MICROPROFILE_SCOPEGPUI("PointCS", 0xff80c0a0);
        curTint = 2;            // tinted per LOD via CB writes inside dispatch loop
        // Clear color UAV (packed RGBA little-endian).
        // 0xFF291F1A = R=0x1A G=0x1F B=0x29 A=0xFF — dark blue-gray to match default bg.
        uint32_t clearC[4] = { 0xFF291F1Au, 0, 0, 0 };
        ctx_->ClearUnorderedAccessViewUint(csColorUav_.Get(), clearC);

        // Unbind RTs so we can bind UAVs.
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

        for (const auto& j : csJobs) {
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

        // Unbind UAVs and SRVs from CS.
        ID3D11UnorderedAccessView* noUav[] = { nullptr };
        ctx_->CSSetUnorderedAccessViews(0, 1, noUav, initCounts);
        ID3D11ShaderResourceView* noSrv[] = { nullptr };
        ctx_->CSSetShaderResources(0, 1, noSrv);
        ctx_->CSSetShader(nullptr, nullptr, 0);

        // Blit UAV color back to whichever RT we render the rest of the scene to.
        ID3D11RenderTargetView* active = (msaaSamples_ > 1) ? msaaRtv_.Get() : rtv_.Get();
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

        // Restore.
        ID3D11ShaderResourceView* nullSrv[] = { nullptr };
        ctx_->PSSetShaderResources(0, 1, nullSrv);
        ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
        ctx_->OMSetRenderTargets(1, rtvs, dsv_.Get());
        ctx_->IASetInputLayout(inputLayout_.Get());
    }

    if (!vidJobs.empty() && pointSrv_ && ib_) {
        MICROPROFILE_SCOPEGPUI("PolyVID", 0xff70b0ff);
        curTint = 1;            // PolyVID is a poly-class close tech
        // Reuse polygon IB (visMask-culled chunk-local indices = 8*voxLocal + corner).
        ctx_->IASetInputLayout(nullptr);
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx_->VSSetShader(vsPolyVid_.Get(), nullptr, 0);
        ctx_->PSSetShader(psPolyVid_.Get(), nullptr, 0);
        ID3D11Buffer* nullVbs[] = { nullptr };
        UINT z = 0;
        ctx_->IASetVertexBuffers(0, 1, nullVbs, &z, &z);
        ctx_->IASetIndexBuffer(ib_.Get(), DXGI_FORMAT_R32_UINT, 0);
        ID3D11ShaderResourceView* srvs[] = { nullptr, pointSrv_.Get() };
        ctx_->VSSetShaderResources(0, 2, srvs);
        for (const auto& j : vidJobs) {
            issueChunkCbEx(j, 0.0f, j.gs->pointFirst);
            ctx_->DrawIndexed(j.gs->indexCount, j.gs->firstIndex, 0);
            ++lastDrawn_;
            lastDrawnTris_ += j.gs->indexCount / 3;
            lastPolyTris_  += j.gs->indexCount / 3;
            lastPolyVerts_ += j.gs->indexCount;
        }
        ID3D11ShaderResourceView* nullSrvs[] = { nullptr, nullptr };
        ctx_->VSSetShaderResources(0, 2, nullSrvs);
        ctx_->IASetInputLayout(inputLayout_.Get());
    }

    if (!axisJobs.empty() && pointSrv_) {
        MICROPROFILE_SCOPEGPUI("PolyAxis", 0xff60a0e0);
        curTint = 1;
        // Bind per-voxel face-AO SB at t2 (PolyAxis VS samples by faceAxis+sgn).
        if (pointAo6Srv_) {
            ID3D11ShaderResourceView* aoSrvs[] = { pointAo6Srv_.Get() };
            ctx_->VSSetShaderResources(2, 1, aoSrvs);
        }
        // Pure-math instanced cube: no VB / no IB. DrawInstanced(18, count).
        ctx_->IASetInputLayout(nullptr);
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx_->VSSetShader(vsPolyAxis_.Get(), nullptr, 0);
        ctx_->PSSetShader(psPolyVid_.Get(), nullptr, 0);
        ID3D11Buffer* nullVbs[] = { nullptr };
        UINT z = 0;
        ctx_->IASetVertexBuffers(0, 1, nullVbs, &z, &z);
        ID3D11ShaderResourceView* srvs[] = { nullptr, pointSrv_.Get() };
        ctx_->VSSetShaderResources(0, 2, srvs);
        for (const auto& j : axisJobs) {
            issueChunkCbEx(j, 0.0f, j.gs->pointFirst);
            // 18 verts per voxel, one non-instanced draw per chunk.
            ctx_->Draw(j.gs->pointCount * 18u, 0);
            ++lastDrawn_;
            lastDrawnTris_ += (uint64_t)j.gs->pointCount * 6;
            lastPolyTris_  += (uint64_t)j.gs->pointCount * 6;
            lastPolyVerts_ += (uint64_t)j.gs->pointCount * 18u;
        }
        ID3D11ShaderResourceView* nullSrvs[] = { nullptr, nullptr, nullptr };
        ctx_->VSSetShaderResources(0, 3, nullSrvs);
        ctx_->IASetInputLayout(inputLayout_.Get());
    }

    if (!axisInstJobs.empty() && pointSrv_) {
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
        for (const auto& j : axisInstJobs) {
            issueChunkCbEx(j, 0.0f, j.gs->pointFirst);
            // 18 verts × pointCount instances. Each instance = 1 voxel.
            ctx_->DrawInstanced(18, j.gs->pointCount, 0, 0);
            ++lastDrawn_;
            lastDrawnTris_ += (uint64_t)j.gs->pointCount * 6;
            lastPolyTris_  += (uint64_t)j.gs->pointCount * 6;
            lastPolyVerts_ += (uint64_t)j.gs->pointCount * 18u;
        }
        ID3D11ShaderResourceView* nullSrvs[] = { nullptr, nullptr, nullptr };
        ctx_->VSSetShaderResources(0, 3, nullSrvs);
        ctx_->IASetInputLayout(inputLayout_.Get());
    }

    if (!billJobs.empty() && pointSrv_ && billboardIb_) {
        MICROPROFILE_SCOPEGPUI("Billboard", 0xffa0c060);
        curTint = 2;
        ctx_->IASetInputLayout(nullptr);
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx_->VSSetShader(vsBillboard_.Get(), nullptr, 0);
        ctx_->PSSetShader(psBillboard_.Get(), nullptr, 0);
        ctx_->RSSetState(rsNoCull_.Get());        // billboard quad
        ID3D11Buffer* nullVbs[] = { nullptr };
        UINT z = 0;
        ctx_->IASetVertexBuffers(0, 1, nullVbs, &z, &z);
        ctx_->IASetIndexBuffer(billboardIb_.Get(), DXGI_FORMAT_R32_UINT, 0);
        ID3D11ShaderResourceView* srvs[] = { nullptr, pointSrv_.Get() };
        ctx_->VSSetShaderResources(0, 2, srvs);
        for (const auto& j : billJobs) {
            issueChunkCbEx(j, 0.0f, j.gs->pointFirst);
            ctx_->DrawIndexed(j.gs->pointCount * 6, 0, 0);
            ++lastDrawn_;
            lastDrawnTris_ += (uint64_t)j.gs->pointCount * 2;
            lastPointCount_ += j.gs->pointCount;
        }
        ID3D11ShaderResourceView* nullSrvs[] = { nullptr, nullptr };
        ctx_->VSSetShaderResources(0, 2, nullSrvs);
        ctx_->RSSetState(wireframe ? rsWire_.Get() : rsSolid_.Get());
        ctx_->IASetInputLayout(inputLayout_.Get());
    }

    if (!billTriJobs.empty() && pointSrv_) {
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
        for (const auto& j : billTriJobs) {
            issueChunkCbEx(j, 0.0f, j.gs->pointFirst);
            ctx_->Draw(j.gs->pointCount * 3, 0);
            ++lastDrawn_;
            lastDrawnTris_ += j.gs->pointCount;
            lastPointCount_ += j.gs->pointCount;
        }
        ID3D11ShaderResourceView* nullSrvs[] = { nullptr, nullptr };
        ctx_->VSSetShaderResources(0, 2, nullSrvs);
        ctx_->RSSetState(wireframe ? rsWire_.Get() : rsSolid_.Get());
        ctx_->IASetInputLayout(inputLayout_.Get());
    }

    if (!hexJobs.empty() && pointVb_) {
        MICROPROFILE_SCOPEGPUI("HexSprite", 0xff80b070);
        curTint = 2;
        ctx_->IASetInputLayout(inputLayoutHex_.Get());
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx_->VSSetShader(vsHex_.Get(), nullptr, 0);
        ctx_->PSSetShader(psHex_.Get(), nullptr, 0);
        ctx_->RSSetState(rsNoCull_.Get());
        ID3D11Buffer* vbs[] = { pointVb_.Get() };
        ctx_->IASetVertexBuffers(0, 1, vbs, &stride, &offset);
        for (const auto& j : hexJobs) {
            issueChunkCb(j);
            // 18 verts per voxel = 6 triangles fanning closest corner.
            ctx_->DrawInstanced(18, j.gs->pointCount, 0, j.gs->pointFirst);
            ++lastDrawn_;
            lastDrawnTris_ += j.gs->pointCount * 6;
            lastPointCount_ += j.gs->pointCount;
        }
        // restore default culling for any subsequent draws this frame
        ctx_->RSSetState(wireframe ? rsWire_.Get() : rsSolid_.Get());
        ctx_->IASetInputLayout(inputLayout_.Get());
    }

    if ((anySplat) && (!splatJobs.empty() || !polyJobs.empty()) && pointVb_
        && splatColorRtv_ && psSplatRecon_) {
        MICROPROFILE_SCOPEGPUI("Splat", 0xffffd060);
        // New pipeline:
        //   1. Clear splatColorRtv_ (alpha=0) + splatDsv_ (depth=0, stencil=0).
        //   2. Sort close polys front-to-back; draw with dsSplatPoly_ (depth+stencil=1, alpha=0).
        //   3. Draw splat points with dsSplatPoint_ (depth test, no stencil change, alpha=lodIdx+1).
        //   4. Fullscreen PS reads splatColorSrv_+depth+stencil. stencil==1: passthrough poly.
        //      stencil==0: neighbor-search splat markers, ray-AABB, light.

        float clr[4] = { lastClear_[0], lastClear_[1], lastClear_[2], 0.0f };
        ctx_->ClearRenderTargetView(splatColorRtv_.Get(), clr);
        ctx_->ClearDepthStencilView(splatDsv_.Get(),
                                    D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 0.0f, 0);
        ID3D11RenderTargetView* splatRtvs[] = { splatColorRtv_.Get() };
        ctx_->OMSetRenderTargets(1, splatRtvs, splatDsv_.Get());

        // Step 2: close polys (for SplatHybrid or any close tech in splat-far compositor).
        if (tech == RenderTech::SplatHybrid || splatFarCompositor) {
            MICROPROFILE_SCOPEGPUI("Splat/Poly", 0xffffe080);
            std::sort(polyJobs.begin(), polyJobs.end(), [&](const Job& a, const Job& b) {
                float am[3] = { a.gs->aabbMin[0] + a.ox, a.gs->aabbMin[1], a.gs->aabbMin[2] + a.oz };
                float aM[3] = { a.gs->aabbMax[0] + a.ox, a.gs->aabbMax[1], a.gs->aabbMax[2] + a.oz };
                float bm[3] = { b.gs->aabbMin[0] + b.ox, b.gs->aabbMin[1], b.gs->aabbMin[2] + b.oz };
                float bM[3] = { b.gs->aabbMax[0] + b.ox, b.gs->aabbMax[1], b.gs->aabbMax[2] + b.oz };
                return chunkDist(am, aM) < chunkDist(bm, bM);
            });
            ctx_->OMSetDepthStencilState(dsSplatPoly_.Get(), 1);
            // Render polys with the splat-poly PS (writes alpha=0 + lit color).
            // drawPolyJobs uses dataset polygon source; psSplatAlbedoPoly_ is the
            // PS that outputs alpha=0 marker.
            // Override PS for this pass:
            ID3D11Buffer* cbs[] = { cbPerFrame_.Get(), cbPerChunk_.Get() };
            ctx_->PSSetConstantBuffers(0, 2, cbs);
            ctx_->VSSetConstantBuffers(0, 2, cbs);
            curTint = 1;
            UINT polyStride = sizeof(VoxelPolyVertex);
            UINT polyOff = 0;
            ctx_->IASetInputLayout(inputLayoutPoly_.Get());
            ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx_->VSSetShader(vs_.Get(), nullptr, 0);
            ctx_->PSSetShader(psSplatAlbedoPoly_.Get(), nullptr, 0);
            ID3D11Buffer* vbs[] = { vb_.Get() };
            ctx_->IASetVertexBuffers(0, 1, vbs, &polyStride, &polyOff);
            ctx_->IASetIndexBuffer(ib_.Get(), DXGI_FORMAT_R32_UINT, 0);
            for (const auto& j : polyJobs) {
                D3D11_MAPPED_SUBRESOURCE mm;
                ctx_->Map(cbPerChunk_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mm);
                CBPerChunk cc = {};
                cc.chunkBase[0] = sceneOrigin_[0] + j.ox;
                cc.chunkBase[1] = sceneOrigin_[1];
                cc.chunkBase[2] = sceneOrigin_[2] + j.oz;
                cc.chunkTint = curTint;
                memcpy(mm.pData, &cc, sizeof(cc));
                ctx_->Unmap(cbPerChunk_.Get(), 0);
                ctx_->DrawIndexed(j.gs->indexCount, j.gs->firstIndex, j.gs->baseVertex);
                ++lastDrawn_;
                lastDrawnTris_ += j.gs->indexCount / 3;
                lastPolyTris_  += j.gs->indexCount / 3;
                lastPolyVerts_ += j.gs->indexCount;
            }
        }

        // PolyVID / PolyAxis / PolyAxisInstanced when used as close-side tech
        // alongside splat-far: draw into splat RT with alpha=0 + stencil=1.
        if (splatFarCompositor && pointSrv_
            && (!vidJobs.empty() || !axisJobs.empty() || !axisInstJobs.empty())) {
            MICROPROFILE_SCOPEGPUI("Splat/PolyAxis", 0xffffd080);
            ctx_->OMSetDepthStencilState(dsSplatPoly_.Get(), 1);
            curTint = 1;
            ID3D11Buffer* cbs[] = { cbPerFrame_.Get(), cbPerChunk_.Get() };
            ctx_->PSSetConstantBuffers(0, 2, cbs);
            ctx_->VSSetConstantBuffers(0, 2, cbs);
            ctx_->IASetInputLayout(nullptr);
            ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx_->PSSetShader(psPolyVidAlpha0_.Get(), nullptr, 0);
            ID3D11Buffer* nullVbs[] = { nullptr };
            UINT z = 0;
            ctx_->IASetVertexBuffers(0, 1, nullVbs, &z, &z);
            ID3D11ShaderResourceView* srvs[] = { nullptr, pointSrv_.Get() };
            ctx_->VSSetShaderResources(0, 2, srvs);
            if (pointAo6Srv_) {
                ID3D11ShaderResourceView* aoSrvs[] = { pointAo6Srv_.Get() };
                ctx_->VSSetShaderResources(2, 1, aoSrvs);
            }
            auto issueCb = [&](const Job& j) {
                D3D11_MAPPED_SUBRESOURCE mm;
                ctx_->Map(cbPerChunk_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mm);
                CBPerChunk cc = {};
                cc.chunkBase[0] = sceneOrigin_[0] + j.ox;
                cc.chunkBase[1] = sceneOrigin_[1];
                cc.chunkBase[2] = sceneOrigin_[2] + j.oz;
                cc.voxelBase = j.gs->pointFirst;
                cc.chunkTint = curTint;
                memcpy(mm.pData, &cc, sizeof(cc));
                ctx_->Unmap(cbPerChunk_.Get(), 0);
            };
            if (!vidJobs.empty() && ib_) {
                ctx_->VSSetShader(vsPolyVid_.Get(), nullptr, 0);
                ctx_->IASetIndexBuffer(ib_.Get(), DXGI_FORMAT_R32_UINT, 0);
                for (const auto& j : vidJobs) {
                    issueCb(j);
                    ctx_->DrawIndexed(j.gs->indexCount, j.gs->firstIndex, 0);
                    ++lastDrawn_;
                    lastDrawnTris_ += j.gs->indexCount / 3;
                    lastPolyTris_  += j.gs->indexCount / 3;
                    lastPolyVerts_ += j.gs->indexCount;
                }
            }
            if (!axisJobs.empty()) {
                ctx_->VSSetShader(vsPolyAxis_.Get(), nullptr, 0);
                for (const auto& j : axisJobs) {
                    issueCb(j);
                    ctx_->Draw(j.gs->pointCount * 18u, 0);
                    ++lastDrawn_;
                    lastDrawnTris_ += (uint64_t)j.gs->pointCount * 6;
                    lastPolyTris_  += (uint64_t)j.gs->pointCount * 6;
                    lastPolyVerts_ += (uint64_t)j.gs->pointCount * 18u;
                }
            }
            if (!axisInstJobs.empty()) {
                ctx_->VSSetShader(vsPolyAxisInstanced_.Get(), nullptr, 0);
                for (const auto& j : axisInstJobs) {
                    issueCb(j);
                    ctx_->DrawInstanced(18, j.gs->pointCount, 0, 0);
                    ++lastDrawn_;
                    lastDrawnTris_ += (uint64_t)j.gs->pointCount * 6;
                    lastPolyTris_  += (uint64_t)j.gs->pointCount * 6;
                    lastPolyVerts_ += (uint64_t)j.gs->pointCount * 18u;
                }
            }
            ID3D11ShaderResourceView* nullSrvs[] = { nullptr, nullptr, nullptr };
            ctx_->VSSetShaderResources(0, 3, nullSrvs);
            ctx_->IASetInputLayout(inputLayout_.Get());
            // Clear job lists so the normal main-RT blocks below don't re-draw.
            vidJobs.clear();
            axisJobs.clear();
            axisInstJobs.clear();
        }

        // Step 3: splat points (alpha = encoded lodIdx+1).
        {
            MICROPROFILE_SCOPEGPUI("Splat/Points", 0xffffc040);
            ctx_->OMSetDepthStencilState(dsSplatPoint_.Get(), 0);
            ctx_->IASetInputLayout(inputLayout_.Get());
            ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
            ctx_->VSSetShader(vsPoints_.Get(), nullptr, 0);
            ctx_->PSSetShader(psSplatAlbedo_.Get(), nullptr, 0);
            ID3D11Buffer* vbs[] = { pointVb_.Get() };
            UINT vbStride = sizeof(Vertex), vbOff = 0;
            ctx_->IASetVertexBuffers(0, 1, vbs, &vbStride, &vbOff);

            size_t i = 0;
            while (i < splatJobs.size()) {
                float ox = splatJobs[i].ox, oz = splatJobs[i].oz;
                PointLod lod = splatJobs[i].lod;
                curTint = 2u + lodToIdx(lod);
                issueChunkCbEx(splatJobs[i], lodToHalfExtent(lod), 0u);
                while (i < splatJobs.size()
                       && splatJobs[i].ox == ox && splatJobs[i].oz == oz
                       && splatJobs[i].lod == lod) {
                    uint32_t spanFirst, count0;
                    pointRangeForJob(splatJobs[i], spanFirst, count0);
                    if (count0 == 0) { ++i; continue; }
                    uint32_t spanEnd = spanFirst + count0;
                    ++i;
                    while (i < splatJobs.size()
                           && splatJobs[i].ox == ox && splatJobs[i].oz == oz
                           && splatJobs[i].lod == lod) {
                        uint32_t f, c;
                        pointRangeForJob(splatJobs[i], f, c);
                        if (f != spanEnd) break;
                        spanEnd += c;
                        ++i;
                    }
                    uint32_t spanCount = spanEnd - spanFirst;
                    ctx_->Draw(spanCount, spanFirst);
                    ++lastDrawn_;
                    lastDrawnTris_ += spanCount;
                    lastPointCount_ += spanCount;
                }
            }
        }

        // Step 4: fullscreen reconstruction PS -> output RT.
        // Bind the main DSV here so the PS's SV_Depth output populates depthTex_
        // (TAA reprojection + post sky need scene depth everywhere).
        {
            MICROPROFILE_SCOPEGPUI("Splat/ReconstructPS", 0xffffa030);
            ID3D11RenderTargetView* dstRtv = postEnabled ? taaSceneRtv_.Get()
                                             : ((msaaSamples_ > 1) ? msaaRtv_.Get() : rtv_.Get());
            ctx_->ClearDepthStencilView(dsv_.Get(), D3D11_CLEAR_DEPTH, 0.0f, 0);
            ctx_->OMSetRenderTargets(1, &dstRtv, dsv_.Get());
            ctx_->OMSetDepthStencilState(dsAlwaysWrite_.Get(), 0);
            ctx_->RSSetState(rsNoCull_.Get());
            ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx_->IASetInputLayout(nullptr);
            ID3D11Buffer* nVb[] = { nullptr }; UINT zz = 0;
            ctx_->IASetVertexBuffers(0, 1, nVb, &zz, &zz);
            ctx_->VSSetShader(vsBlit_.Get(), nullptr, 0);
            ctx_->PSSetShader(psSplatRecon_.Get(), nullptr, 0);
            // Slots: t1 = splatDepth, t2 = splatColor, t9 = splatStencil.
            ID3D11ShaderResourceView* srvs[] = { nullptr, splatDepthSrv_.Get(), splatColorSrv_.Get() };
            ctx_->PSSetShaderResources(0, 3, srvs);
            ID3D11ShaderResourceView* stencilSrvs[] = { splatStencilSrv_.Get() };
            ctx_->PSSetShaderResources(9, 1, stencilSrvs);
            ctx_->Draw(3, 0);
            ID3D11ShaderResourceView* nulls3[] = { nullptr, nullptr, nullptr };
            ctx_->PSSetShaderResources(0, 3, nulls3);
            ID3D11ShaderResourceView* nullsS[] = { nullptr };
            ctx_->PSSetShaderResources(9, 1, nullsS);
            ctx_->RSSetState(wireframe ? rsWire_.Get() : rsSolid_.Get());
        }

        // Restore RT/DSV for any subsequent passes (bounds, etc.).
        ID3D11RenderTargetView* active = postEnabled ? taaSceneRtv_.Get()
                                             : ((msaaSamples_ > 1) ? msaaRtv_.Get() : rtv_.Get());
        ID3D11RenderTargetView* rrtvs[] = { active };
        ctx_->OMSetRenderTargets(1, rrtvs, dsv_.Get());
        ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
    }

    if (showChunkBounds && boundsVb_ && boundsIb_) {
        ctx_->IASetInputLayout(inputLayoutBounds_.Get());
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
        ctx_->VSSetShader(vsBounds_.Get(), nullptr, 0);
        ctx_->PSSetShader(psBounds_.Get(), nullptr, 0);
        ctx_->OMSetDepthStencilState(dsAlways_.Get(), 0);
        UINT bstride = sizeof(float) * 3;
        UINT boff = 0;
        ID3D11Buffer* bvbs[] = { boundsVb_.Get() };
        ctx_->IASetVertexBuffers(0, 1, bvbs, &bstride, &boff);
        ctx_->IASetIndexBuffer(boundsIb_.Get(), DXGI_FORMAT_R32_UINT, 0);

        // Tight per-chunk AABB: chunkBase = world aabbMin, size = aabb extents.
        auto drawJob = [&](const Job& j) {
            float sizeX = j.gs->aabbMax[0] - j.gs->aabbMin[0];
            float sizeY = j.gs->aabbMax[1] - j.gs->aabbMin[1];
            float sizeZ = j.gs->aabbMax[2] - j.gs->aabbMin[2];
            D3D11_MAPPED_SUBRESOURCE mm;
            ctx_->Map(cbPerChunk_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mm);
            CBPerChunk cc = {};
            cc.chunkBase[0] = j.gs->aabbMin[0] + j.ox;
            cc.chunkBase[1] = j.gs->aabbMin[1];
            cc.chunkBase[2] = j.gs->aabbMin[2] + j.oz;
            cc._pad = sizeX;
            cc.voxelBase = 0;
            cc.chunkLodIdx = 0;
            memcpy(&cc._pad2[0], &sizeY, sizeof(float));
            memcpy(&cc._pad2[1], &sizeZ, sizeof(float));
            memcpy(mm.pData, &cc, sizeof(cc));
            ctx_->Unmap(cbPerChunk_.Get(), 0);
            ctx_->DrawIndexed(24, 0, 0);
        };
        for (const auto& j : polyJobs)  drawJob(j);
        for (const auto& j : pointJobs) drawJob(j);
        for (const auto& j : hexJobs)   drawJob(j);

        ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
        ctx_->IASetInputLayout(inputLayout_.Get());
    }

    // ------------------- TAA composite (optional) -> post (always) -------------------
    ID3D11ShaderResourceView* postInput = nullptr;
    if (postEnabled) {
        // MSAA: resolve scene MSAA color into non-MSAA taaSceneTex_ for post.
        // Splat/SplatHybrid already wrote taaSceneTex_ via CopyResource.
        const bool splatTech = (tech == RenderTech::Splat || tech == RenderTech::SplatHybrid);
        if (useMsaa && !splatTech && msaaColorTex_ && taaSceneTex_) {
            ctx_->ResolveSubresource(taaSceneTex_.Get(), 0, msaaColorTex_.Get(), 0,
                                     DXGI_FORMAT_R8G8B8A8_UNORM);
        }
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

        // Post pass: sky + sharpen + tonemap. Always runs.
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

        // ImGui writes to bound RTV -> already backbuffer.
        // the backbuffer so the UI lands on screen.
        ID3D11RenderTargetView* backRtv = rtv_.Get();
        ctx_->OMSetRenderTargets(1, &backRtv, nullptr);
    }

    // Remember this frame's un-jittered view-proj for next frame's reprojection.
    // History is at pixel grid (jitter undone in PS), so prev VP must be unjittered.
    hlslpp::store(taaPrevVP_, vpUnjittered);
}

void Renderer::EndFrame(bool vsync)
{
    if (msaaSamples_ > 1 && msaaColorTex_ && !postWroteBackbuf_) {
        ComPtr<ID3D11Texture2D> backBuf;
        swap_->GetBuffer(0, IID_PPV_ARGS(backBuf.GetAddressOf()));
        if (backBuf) {
            ctx_->ResolveSubresource(backBuf.Get(), 0, msaaColorTex_.Get(), 0,
                                     DXGI_FORMAT_R8G8B8A8_UNORM);
        }
    }
    swap_->Present(vsync ? 1 : 0, 0);
}

void Renderer::SetMsaa(uint32_t samples)
{
    if (samples != 1 && samples != 2 && samples != 4 && samples != 8) samples = 1;
    if (samples == msaaSamples_) return;
    msaaSamples_ = samples;
    ctx_->OMSetRenderTargets(0, nullptr, nullptr);
    CreateRenderTargets();
}
