#define NOMINMAX
#include "renderer.h"

#include <d3dcompiler.h>
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
    float fogDensity;   // 0 = fog off
};

struct CBPerChunk {
    float    chunkBase[3];
    float    _pad;          // lodHalfExtent (points) / chunkSize (bounds)
    uint32_t voxelBase;     // PolyVID: chunk's first voxel index in pointSb_
    uint32_t _pad2[3];
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

static std::string ReadTextFile(const char* path) {
    std::ifstream f(path);
    if (!f.good()) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool Renderer::Init(HWND hwnd) {
    if (!CreateDeviceAndSwap(hwnd)) return false;

    RECT rc; GetClientRect(hwnd, &rc);
    width_  = (uint32_t)(rc.right  - rc.left);
    height_ = (uint32_t)(rc.bottom - rc.top);

    if (!CreateRenderTargets()) return false;
    if (!CreateShaders())       return false;
    if (!CreatePipelineState()) return false;
    return true;
}

void Renderer::Shutdown() {
    subs_.clear();
    vb_.Reset();
    ib_.Reset();
}

bool Renderer::CreateDeviceAndSwap(HWND hwnd) {
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

bool Renderer::CreateRenderTargets() {
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
    td.Format = DXGI_FORMAT_D32_FLOAT;
    td.SampleDesc = sd;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    hr = device_->CreateTexture2D(&td, nullptr, depthTex_.GetAddressOf());
    if (FAILED(hr)) return false;
    D3D11_DEPTH_STENCIL_VIEW_DESC dsvd = {};
    dsvd.Format = DXGI_FORMAT_D32_FLOAT;
    dsvd.ViewDimension = (msaaSamples_ > 1) ? D3D11_DSV_DIMENSION_TEXTURE2DMS
                                            : D3D11_DSV_DIMENSION_TEXTURE2D;
    hr = device_->CreateDepthStencilView(depthTex_.Get(), &dsvd, dsv_.GetAddressOf());
    if (FAILED(hr)) return false;

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

    // Dedicated non-MSAA depth for splat pass.
    D3D11_TEXTURE2D_DESC sdd = {};
    sdd.Width = width_;
    sdd.Height = height_;
    sdd.MipLevels = 1;
    sdd.ArraySize = 1;
    sdd.Format = DXGI_FORMAT_D32_FLOAT;
    sdd.SampleDesc.Count = 1;
    sdd.Usage = D3D11_USAGE_DEFAULT;
    sdd.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    if (FAILED(device_->CreateTexture2D(&sdd, nullptr, splatDepthTex_.GetAddressOf()))) return false;
    D3D11_DEPTH_STENCIL_VIEW_DESC sddv = {};
    sddv.Format = DXGI_FORMAT_D32_FLOAT;
    sddv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    if (FAILED(device_->CreateDepthStencilView(splatDepthTex_.Get(), &sddv, splatDsv_.GetAddressOf()))) return false;
    return true;
}

bool Renderer::CreateShaders() {
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
            std::string msg = "Shader compile error\n";
            if (errs) msg += std::string((const char*)errs->GetBufferPointer(), errs->GetBufferSize());
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
    if (!compile("psmain_polyvid", "ps_5_0", psbv)) return false;
    ComPtr<ID3DBlob> vsbB, psbB, vsbBT;
    if (!compile("vsmain_billboard", "vs_5_0", vsbB)) return false;
    if (!compile("psmain_billboard", "ps_5_0", psbB)) return false;
    if (!compile("vsmain_billboard_tri", "vs_5_0", vsbBT)) return false;
    ComPtr<ID3DBlob> vsbM, psbM;
    if (!compile("vsmain_merged", "vs_5_0", vsbM)) return false;
    if (!compile("psmain_merged", "ps_5_0", psbM)) return false;
    ComPtr<ID3DBlob> vsbSp, psbSp, csbSp;
    if (!compile("vsmain_splat",  "vs_5_0", vsbSp)) return false;
    if (!compile("psmain_splat",  "ps_5_0", psbSp)) return false;
    if (!compile("csmain_splat",  "cs_5_0", csbSp)) return false;
    if (!compile("vsmain_bounds", "vs_5_0", vsbb)) return false;
    if (!compile("psmain_bounds", "ps_5_0", psbb)) return false;
    ComPtr<ID3DBlob> vsbd;
    if (!compile("vsmain_depth", "vs_5_0", vsbd)) return false;
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
    hr = device_->CreateVertexShader(vsbSp->GetBufferPointer(), vsbSp->GetBufferSize(), nullptr, vsSplat_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbSp->GetBufferPointer(), psbSp->GetBufferSize(), nullptr, psSplat_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreateComputeShader(csbSp->GetBufferPointer(), csbSp->GetBufferSize(), nullptr, csSplat_.GetAddressOf());
    if (FAILED(hr)) return false;
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
    hr = device_->CreateComputeShader(csb->GetBufferPointer(), csb->GetBufferSize(), nullptr, csPoints_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreateVertexShader(vsbl->GetBufferPointer(), vsbl->GetBufferSize(), nullptr, vsBlit_.GetAddressOf());
    if (FAILED(hr)) return false;
    hr = device_->CreatePixelShader(psbl->GetBufferPointer(), psbl->GetBufferSize(), nullptr, psBlit_.GetAddressOf());
    if (FAILED(hr)) return false;

    D3D11_INPUT_ELEMENT_DESC il[] = {
        // x,y,z,faceIdx packed as 4 unsigned bytes
        { "POSITION", 0, DXGI_FORMAT_R8G8B8A8_UINT,  0, 0,                            D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR",    0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    hr = device_->CreateInputLayout(il, _countof(il),
                                    vsb->GetBufferPointer(), vsb->GetBufferSize(),
                                    inputLayout_.GetAddressOf());
    if (FAILED(hr)) return false;

    // HexSprite: same buffer, per-instance stepping.
    D3D11_INPUT_ELEMENT_DESC ilHex[] = {
        { "POSITION", 0, DXGI_FORMAT_R8G8B8A8_UINT,  0, 0, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
        { "COLOR",    0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
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

bool Renderer::CreatePipelineState() {
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

    D3D11_DEPTH_STENCIL_DESC ddEq = {};
    ddEq.DepthEnable = TRUE;
    ddEq.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;   // prepass already wrote depth
    ddEq.DepthFunc = D3D11_COMPARISON_GREATER_EQUAL;
    if (FAILED(device_->CreateDepthStencilState(&ddEq, dsEqual_.GetAddressOf()))) return false;

    D3D11_BLEND_DESC bsd = {};
    bsd.RenderTarget[0].BlendEnable = FALSE;
    bsd.RenderTarget[0].RenderTargetWriteMask = 0;       // no color writes
    if (FAILED(device_->CreateBlendState(&bsd, bsNoColor_.GetAddressOf()))) return false;
    return true;
}

void Renderer::Resize(uint32_t w, uint32_t h) {
    if (!swap_ || (w == width_ && h == height_) || w == 0 || h == 0) return;
    ctx_->OMSetRenderTargets(0, nullptr, nullptr);
    rtv_.Reset();
    dsv_.Reset();
    depthTex_.Reset();
    swap_->ResizeBuffers(0, w, h, DXGI_FORMAT_UNKNOWN, 0);
    width_ = w; height_ = h;
    CreateRenderTargets();
}

void Renderer::UploadScene(const Scene& scene) {
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
    bd.ByteWidth = (UINT)(scene.vertices.size() * sizeof(Vertex));
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
            if (s.indexCount > maxIndexCount) maxIndexCount = s.indexCount;
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
    sceneSpan_[0] = scene.aabbMax[0] - scene.aabbMin[0];
    sceneSpan_[1] = scene.aabbMax[1] - scene.aabbMin[1];
    sceneSpan_[2] = scene.aabbMax[2] - scene.aabbMin[2];

    // PolyVID shared IB: per voxel 24 unique vertex slots (6 faces * 4 corners),
    // 36 indices = 6 faces * (3+3). IB[v*36+f*6+k] = v*24 + f*4 + cornerWithinFace[k]
    // where cornerWithinFace pattern is (0,1,2,0,2,3).
    {
        uint32_t maxVox = 0;
        for (const auto& s : scene.subs) {
            uint32_t c = s.pointCount;
            if (s.pointCountL1 > c) c = s.pointCountL1;
            if (s.pointCountL2 > c) c = s.pointCountL2;
            if (c > maxVox) maxVox = c;
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
    vbBytes_ = scene.vertices.size() * sizeof(Vertex);
    ibBytes_ = ibCount * sizeof(uint32_t);
}

void Renderer::UploadMergedMesh(const MergedMesh& mesh) {
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

void Renderer::BeginFrame(float clear[4]) {
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

void Renderer::DrawScene(const Camera& cam, ShadingMode mode, int gridSize, RenderTech tech, bool showChunkBounds, bool zPrepass, PointLighting pointLight, PointLod pointLod, float pointLodScale, bool splatFilter, const float fogColor[3], float fogDensity, float hybridThreshold) {
    if (hybridThreshold < 0.01f) hybridThreshold = 0.01f;
    if (pointLodScale < 0.01f) pointLodScale = 0.01f;
    if (gridSize < 1) gridSize = 1;
    if (gridSize > 10) gridSize = 10;
    hlslpp::float4x4 v  = cam.view();
    hlslpp::float4x4 p  = cam.proj((float)width_ / (float)(height_ ? height_ : 1));
    hlslpp::float4x4 vp = hlslpp::mul(v, p);

    D3D11_MAPPED_SUBRESOURCE m;
    ctx_->Map(cbPerFrame_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m);
    {
        CBPerFrame cb;
        hlslpp::store(cb.viewProj, vp);
        hlslpp::store(cb.camPos, cam.position);
        cb.mode = (float)(int)mode;
        cb.lightDir[0] = 0.4f; cb.lightDir[1] = 0.8f; cb.lightDir[2] = 0.2f;
        cb.ambient = 0.7f;
        hlslpp::float3 fwd = cam.forward();
        hlslpp::float3 pn  = -fwd;
        hlslpp::store(cb.pointNormal, pn);
        cb._pad = 0.0f;
        hlslpp::float4x4 invVp = hlslpp::inverse(vp);
        hlslpp::store(cb.invViewProj, invVp);
        cb.screenW = (float)width_;
        cb.screenH = (float)height_;
        cb._pad2[0] = cb._pad2[1] = 0;
        hlslpp::float3 right = cam.right();
        hlslpp::float3 upVec = hlslpp::cross(fwd, right);
        hlslpp::store(cb.camRight,   right);
        hlslpp::store(cb.camUp,      upVec);
        hlslpp::store(cb.camForward, fwd);
        cb._pad3 = cb._pad4 = 0;
        cb.tanHalfFovY = tanf(cam.fovDeg * 3.14159265358979f / 180.0f * 0.5f);
        cb.fogColor[0] = fogColor[0]; cb.fogColor[1] = fogColor[1]; cb.fogColor[2] = fogColor[2];
        cb.fogDensity  = fogDensity;
        memcpy(m.pData, &cb, sizeof(cb));
    }
    ctx_->Unmap(cbPerFrame_.Get(), 0);

    ctx_->IASetInputLayout(inputLayout_.Get());
    ID3D11Buffer* cbs[] = { cbPerFrame_.Get(), cbPerChunk_.Get() };
    ctx_->VSSetConstantBuffers(0, 2, cbs);
    ctx_->PSSetConstantBuffers(0, 1, cbs);
    ctx_->RSSetState(rsSolid_.Get());
    ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);

    if (!vb_ || !ib_) return;

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
    static thread_local std::vector<Job> billJobs;
    static thread_local std::vector<Job> billTriJobs;
    polyJobs.clear();
    pointJobs.clear();
    hexJobs.clear();
    csJobs.clear();
    vidJobs.clear();
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
    const float spanX = sceneSpan_[0];
    const float spanZ = sceneSpan_[2];
    for (int gz = 0; gz < gridSize; ++gz) {
        for (int gx = 0; gx < gridSize; ++gx) {
            float ox = gx * spanX;
            float oz = gz * spanZ;
            for (const auto& gs : subs_) {
                float aMin[3] = { gs.aabbMin[0] + ox, gs.aabbMin[1], gs.aabbMin[2] + oz };
                float aMax[3] = { gs.aabbMax[0] + ox, gs.aabbMax[1], gs.aabbMax[2] + oz };
                bool outside = false;
                for (int pi = 0; pi < 6; ++pi) {
                    const float a = planes[pi][0], b = planes[pi][1], c = planes[pi][2], d = planes[pi][3];
                    float px = a >= 0 ? aMax[0] : aMin[0];
                    float py = b >= 0 ? aMax[1] : aMin[1];
                    float pz = c >= 0 ? aMax[2] : aMin[2];
                    if (a * px + b * py + c * pz + d < 0.0f) { outside = true; break; }
                }
                if (outside) continue;

                // Resolve LOD (Auto -> per-chunk by projected voxel size).
                PointLod effLod = pointLod;
                if (pointLod == PointLod::Auto) {
                    float dx = camP[0] < aMin[0] ? aMin[0] - camP[0]
                             : camP[0] > aMax[0] ? camP[0] - aMax[0] : 0.0f;
                    float dy = camP[1] < aMin[1] ? aMin[1] - camP[1]
                             : camP[1] > aMax[1] ? camP[1] - aMax[1] : 0.0f;
                    float dz = camP[2] < aMin[2] ? aMin[2] - camP[2]
                             : camP[2] > aMax[2] ? camP[2] - aMax[2] : 0.0f;
                    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                    if (dist < 1e-3f) dist = 1e-3f;
                    float ppv = focalPx / dist;
                    // Slider scales thresholds: higher scale pushes LODs farther
                    // (keeps L0 for longer).
                    float t0 = 1.333f / pointLodScale;
                    float t1 = 0.333f / pointLodScale;
                    if      (ppv >= t0) effLod = PointLod::L0;
                    else if (ppv >= t1) effLod = PointLod::L1;
                    else                effLod = PointLod::L2;
                }
                Job j{ &gs, ox, oz, effLod };
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
                } else if (tech == RenderTech::Billboard) {
                    billJobs.push_back(j);
                } else if (tech == RenderTech::BillboardTri) {
                    billTriJobs.push_back(j);
                } else if (tech == RenderTech::MergedMesh) {
                    // MergedMesh has its own draw pass; per-chunk jobs unused.
                } else if (tech == RenderTech::Splat) {
                    pointJobs.push_back(j);   // reuse pointJobs as the splat list
                } else if (tech == RenderTech::SplatHybrid) {
                    // Polygons close, splats far. Both consumed by the Splat block.
                    float dx = camP[0] < aMin[0] ? aMin[0] - camP[0]
                             : camP[0] > aMax[0] ? camP[0] - aMax[0] : 0.0f;
                    float dy = camP[1] < aMin[1] ? aMin[1] - camP[1]
                             : camP[1] > aMax[1] ? camP[1] - aMax[1] : 0.0f;
                    float dz = camP[2] < aMin[2] ? aMin[2] - camP[2]
                             : camP[2] > aMax[2] ? camP[2] - aMax[2] : 0.0f;
                    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                    if (dist < 1e-3f) dist = 1e-3f;
                    float ppv = focalPx / dist;
                    if (ppv >= hybridThreshold) polyJobs.push_back(j);
                    else                        pointJobs.push_back(j);
                } else if (tech == RenderTech::Hybrid2) {
                    float dx = camP[0] < aMin[0] ? aMin[0] - camP[0]
                             : camP[0] > aMax[0] ? camP[0] - aMax[0] : 0.0f;
                    float dy = camP[1] < aMin[1] ? aMin[1] - camP[1]
                             : camP[1] > aMax[1] ? camP[1] - aMax[1] : 0.0f;
                    float dz = camP[2] < aMin[2] ? aMin[2] - camP[2]
                             : camP[2] > aMax[2] ? camP[2] - aMax[2] : 0.0f;
                    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                    if (dist < 1e-3f) dist = 1e-3f;
                    float ppv = focalPx / dist;
                    if (ppv >= hybridThreshold) vidJobs.push_back(j);
                    else                        billTriJobs.push_back(j);
                } else {
                    // Hybrid: nearest point of chunk AABB to camera.
                    float dx = camP[0] < aMin[0] ? aMin[0] - camP[0]
                             : camP[0] > aMax[0] ? camP[0] - aMax[0] : 0.0f;
                    float dy = camP[1] < aMin[1] ? aMin[1] - camP[1]
                             : camP[1] > aMax[1] ? camP[1] - aMax[1] : 0.0f;
                    float dz = camP[2] < aMin[2] ? aMin[2] - camP[2]
                             : camP[2] > aMax[2] ? camP[2] - aMax[2] : 0.0f;
                    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                    if (dist < 1e-3f) dist = 1e-3f;
                    float pixelsPerVoxel = focalPx / dist;
                    if (pixelsPerVoxel < hybridThreshold) pointJobs.push_back(j);
                    else                                  vidJobs.push_back(j);   // Hybrid uses PolyVID
                }
            }
        }
    }

    UINT stride = sizeof(Vertex);
    UINT offset = 0;

    auto issueChunkCbEx = [&](const Job& j, float padVal, uint32_t voxelBase) {
        D3D11_MAPPED_SUBRESOURCE mm;
        ctx_->Map(cbPerChunk_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mm);
        CBPerChunk cc = {};
        cc.chunkBase[0] = j.gs->chunkBase[0] + j.ox;
        cc.chunkBase[1] = j.gs->chunkBase[1];
        cc.chunkBase[2] = j.gs->chunkBase[2] + j.oz;
        cc._pad = padVal;
        cc.voxelBase = voxelBase;
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
        else                            { first = j.gs->pointFirst;   count = j.gs->pointCount;   }
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

    if (!polyJobs.empty() && tech != RenderTech::SplatHybrid) {
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx_->VSSetShader(vs_.Get(), nullptr, 0);
        ctx_->PSSetShader(ps_.Get(), nullptr, 0);
        ctx_->OMSetDepthStencilState(doPrepass ? dsEqual_.Get() : dsTest_.Get(), 0);
        ID3D11Buffer* vbs[] = { vb_.Get() };
        ctx_->IASetVertexBuffers(0, 1, vbs, &stride, &offset);
        ctx_->IASetIndexBuffer(ib_.Get(), DXGI_FORMAT_R32_UINT, 0);
        for (const auto& j : polyJobs) {
            issueChunkCb(j);
            ctx_->DrawIndexed(j.gs->indexCount, j.gs->firstIndex, j.gs->baseVertex);
            ++lastDrawn_;
            lastDrawnTris_ += j.gs->indexCount / 3;
        }
        if (doPrepass) ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
    }
    if (!pointJobs.empty() && pointVb_ && tech != RenderTech::Splat && tech != RenderTech::SplatHybrid) {
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
        ctx_->VSSetShader(vsPoints_.Get(), nullptr, 0);
        ctx_->PSSetShader(pointLight == PointLighting::Simple
                              ? psPointsSimple_.Get()
                              : psPoints_.Get(), nullptr, 0);
        ID3D11Buffer* vbs[] = { pointVb_.Get() };
        ctx_->IASetVertexBuffers(0, 1, vbs, &stride, &offset);
        for (const auto& j : pointJobs) {
            uint32_t first, count;
            pointRangeForJob(j, first, count);
            if (count == 0) continue;
            issueChunkCbEx(j, lodToHalfExtent(j.lod), 0u);
            ctx_->Draw(count, first);
            ++lastDrawn_;
            lastDrawnTris_ += count;
        }
    }
    if (!csJobs.empty() && pointVb_ && pointSrv_ && csColorUav_) {
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
            cb.chunkBase[0] = j.gs->chunkBase[0] + j.ox;
            cb.chunkBase[1] = j.gs->chunkBase[1];
            cb.chunkBase[2] = j.gs->chunkBase[2] + j.oz;
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
        }
        ID3D11ShaderResourceView* nullSrvs[] = { nullptr, nullptr };
        ctx_->VSSetShaderResources(0, 2, nullSrvs);
        ctx_->IASetInputLayout(inputLayout_.Get());
    }

    if (!billJobs.empty() && pointSrv_ && billboardIb_) {
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
        }
        ID3D11ShaderResourceView* nullSrvs[] = { nullptr, nullptr };
        ctx_->VSSetShaderResources(0, 2, nullSrvs);
        ctx_->RSSetState(rsSolid_.Get());
        ctx_->IASetInputLayout(inputLayout_.Get());
    }

    if (!billTriJobs.empty() && pointSrv_) {
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
        }
        ID3D11ShaderResourceView* nullSrvs[] = { nullptr, nullptr };
        ctx_->VSSetShaderResources(0, 2, nullSrvs);
        ctx_->RSSetState(rsSolid_.Get());
        ctx_->IASetInputLayout(inputLayout_.Get());
    }

    if (!hexJobs.empty() && pointVb_) {
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
        }
        // restore default culling for any subsequent draws this frame
        ctx_->RSSetState(rsSolid_.Get());
        ctx_->IASetInputLayout(inputLayout_.Get());
    }

    if ((tech == RenderTech::Splat || tech == RenderTech::SplatHybrid)
        && (!pointJobs.empty() || !polyJobs.empty()) && pointVb_
        && splatColorRtv_ && splatFinalUav_) {
        // Pass 1: always render points to splatColorTex_ + splatDsv_. Identical
        // raster behavior whether or not CS filter runs. Clear RGB with the
        // current scene clear color so the backbuffer appearance matches when
        // CopyResource lands. Keep alpha = 0 so CS reconstruction can
        // distinguish background (a==0) from lit pixels (PS writes a=1).
        float clr[4] = { lastClear_[0], lastClear_[1], lastClear_[2], 0.0f };
        ctx_->ClearRenderTargetView(splatColorRtv_.Get(), clr);
        ctx_->ClearDepthStencilView(splatDsv_.Get(), D3D11_CLEAR_DEPTH, 0.0f, 0);
        ID3D11RenderTargetView* rtvs[] = { splatColorRtv_.Get() };
        ctx_->OMSetRenderTargets(1, rtvs, splatDsv_.Get());
        ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);

        // SplatHybrid: polygons up close, into the same splat RT (depth-tested
        // against splatDsv_ so points can fill gaps behind/beside them).
        if (tech == RenderTech::SplatHybrid && !polyJobs.empty()) {
            ctx_->IASetInputLayout(inputLayout_.Get());
            ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx_->VSSetShader(vs_.Get(), nullptr, 0);
            ctx_->PSSetShader(ps_.Get(), nullptr, 0);
            ID3D11Buffer* vbsP[] = { vb_.Get() };
            ctx_->IASetVertexBuffers(0, 1, vbsP, &stride, &offset);
            ctx_->IASetIndexBuffer(ib_.Get(), DXGI_FORMAT_R32_UINT, 0);
            for (const auto& j : polyJobs) {
                issueChunkCb(j);
                ctx_->DrawIndexed(j.gs->indexCount, j.gs->firstIndex, j.gs->baseVertex);
                ++lastDrawn_;
                lastDrawnTris_ += j.gs->indexCount / 3;
            }
        }

        ctx_->IASetInputLayout(inputLayout_.Get());
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
        // Use Points-mode shaders so Splat looks identical to Points tech.
        ctx_->VSSetShader(vsPoints_.Get(), nullptr, 0);
        ctx_->PSSetShader(pointLight == PointLighting::Simple
                              ? psPointsSimple_.Get()
                              : psPoints_.Get(), nullptr, 0);
        ID3D11Buffer* vbs[] = { pointVb_.Get() };
        UINT vbStride = sizeof(Vertex), vbOff = 0;
        ctx_->IASetVertexBuffers(0, 1, vbs, &vbStride, &vbOff);
        for (const auto& j : pointJobs) {
            uint32_t first, count;
            pointRangeForJob(j, first, count);
            if (count == 0) continue;
            issueChunkCbEx(j, lodToHalfExtent(j.lod), 0u);
            ctx_->Draw(count, first);
            ++lastDrawn_;
            lastDrawnTris_ += count;
        }

        // Pass 2: optional CS reconstruction; else copy raw RT to backbuffer.
        ComPtr<ID3D11Texture2D> backBuf;
        swap_->GetBuffer(0, IID_PPV_ARGS(backBuf.GetAddressOf()));
        if (splatFilter) {
            ID3D11RenderTargetView* nullRtvs[] = { nullptr };
            ctx_->OMSetRenderTargets(1, nullRtvs, nullptr);
            ctx_->CSSetShader(csSplat_.Get(), nullptr, 0);
            ID3D11Buffer* csCbs[] = { cbPerFrame_.Get() };
            ctx_->CSSetConstantBuffers(0, 1, csCbs);
            ID3D11ShaderResourceView* csSrvs[] = { nullptr, nullptr, splatColorSrv_.Get() };
            ctx_->CSSetShaderResources(0, 3, csSrvs);
            ID3D11UnorderedAccessView* csUavs[] = { nullptr, splatFinalUav_.Get() };
            UINT initc[] = { 0, 0 };
            ctx_->CSSetUnorderedAccessViews(0, 2, csUavs, initc);
            UINT gx = (width_ + 7) / 8;
            UINT gy = (height_ + 7) / 8;
            ctx_->Dispatch(gx, gy, 1);
            ID3D11ShaderResourceView* nullCsSrvs[] = { nullptr, nullptr, nullptr };
            ctx_->CSSetShaderResources(0, 3, nullCsSrvs);
            ID3D11UnorderedAccessView* nullCsUavs[] = { nullptr, nullptr };
            ctx_->CSSetUnorderedAccessViews(0, 2, nullCsUavs, initc);
            ctx_->CSSetShader(nullptr, nullptr, 0);
            if (backBuf) ctx_->CopyResource(backBuf.Get(), splatFinalTex_.Get());
        } else {
            if (backBuf) ctx_->CopyResource(backBuf.Get(), splatColorTex_.Get());
        }

        // Restore active RTV/DSV.
        ID3D11RenderTargetView* active = (msaaSamples_ > 1) ? msaaRtv_.Get() : rtv_.Get();
        ID3D11RenderTargetView* rrtvs[] = { active };
        ctx_->OMSetRenderTargets(1, rrtvs, dsv_.Get());
        ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
    }

    if (tech == RenderTech::MergedMesh && mergedVb_ && mergedIb_) {
        ctx_->IASetInputLayout(mergedInputLayout_.Get());
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx_->VSSetShader(vsMerged_.Get(), nullptr, 0);
        ctx_->PSSetShader(psMerged_.Get(), nullptr, 0);
        ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
        UINT stridM = sizeof(MergedVertex);
        UINT offsM = 0;
        ID3D11Buffer* vbs[] = { mergedVb_.Get() };
        ctx_->IASetVertexBuffers(0, 1, vbs, &stridM, &offsM);
        ctx_->IASetIndexBuffer(mergedIb_.Get(), DXGI_FORMAT_R32_UINT, 0);
        // VS adds gChunkBase to world-space vertex (already world). Use it as
        // grid offset to replicate the merged mesh across the grid.
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

        // Union of all visible jobs across passes covers what we actually drew.
        auto drawJob = [&](const Job& j) {
            D3D11_MAPPED_SUBRESOURCE mm;
            ctx_->Map(cbPerChunk_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mm);
            CBPerChunk cc;
            cc.chunkBase[0] = j.gs->chunkBase[0] + j.ox;
            cc.chunkBase[1] = j.gs->chunkBase[1];
            cc.chunkBase[2] = j.gs->chunkBase[2] + j.oz;
            cc._pad = (float)chunkDim_;
            cc.voxelBase = 0;
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
}

void Renderer::EndFrame(bool vsync) {
    if (msaaSamples_ > 1 && msaaColorTex_) {
        ComPtr<ID3D11Texture2D> backBuf;
        swap_->GetBuffer(0, IID_PPV_ARGS(backBuf.GetAddressOf()));
        if (backBuf) {
            ctx_->ResolveSubresource(backBuf.Get(), 0, msaaColorTex_.Get(), 0,
                                     DXGI_FORMAT_R8G8B8A8_UNORM);
        }
    }
    swap_->Present(vsync ? 1 : 0, 0);
}

void Renderer::SetMsaa(uint32_t samples) {
    if (samples != 1 && samples != 2 && samples != 4 && samples != 8) samples = 1;
    if (samples == msaaSamples_) return;
    msaaSamples_ = samples;
    ctx_->OMSetRenderTargets(0, nullptr, nullptr);
    CreateRenderTargets();
}
