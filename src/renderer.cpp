// DX12 Phase-1 (M1) renderer skeleton.
// Provides: device, swapchain, command queue/list, RTV/DSV heaps, fence,
// per-frame clear + present. ImGui font descriptor heap exposed.
// All voxel rendering / LW upload paths live in renderer.h as inline stubs
// returning safe defaults — they will be ported incrementally in M3+.
#define NOMINMAX
#include "renderer.h"
#include "microprofile.h"
#include <hlsl++.h>

#include <DescriptorHeap.h>
#include <GraphicsMemory.h>
#include <DirectXHelpers.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    // 6 frustum planes from a row-major viewProj. Sign: plane.xyz dot p + .w
    // >= 0 = inside.
    void ExtractFrustumPlanes(const float M[16], float planes[6][4])
    {
        planes[0][0] = M[0] + M[3];
        planes[0][1] = M[4] + M[7];
        planes[0][2] = M[8] + M[11];
        planes[0][3] = M[12] + M[15];
        planes[1][0] = M[3] - M[0];
        planes[1][1] = M[7] - M[4];
        planes[1][2] = M[11] - M[8];
        planes[1][3] = M[15] - M[12];
        planes[2][0] = M[1] + M[3];
        planes[2][1] = M[5] + M[7];
        planes[2][2] = M[9] + M[11];
        planes[2][3] = M[13] + M[15];
        planes[3][0] = M[3] - M[1];
        planes[3][1] = M[7] - M[5];
        planes[3][2] = M[11] - M[9];
        planes[3][3] = M[15] - M[13];
        planes[4][0] = M[2];
        planes[4][1] = M[6];
        planes[4][2] = M[10];
        planes[4][3] = M[14];
        planes[5][0] = M[3] - M[2];
        planes[5][1] = M[7] - M[6];
        planes[5][2] = M[11] - M[10];
        planes[5][3] = M[15] - M[14];
    }

    void ThrowIfFailed(HRESULT hr, const char* what)
    {
        if (FAILED(hr))
        {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "D3D12 op failed (0x%08X): %s", (unsigned)hr, what);
            throw std::runtime_error(buf);
        }
    }

    void NameObject(ID3D12Object* obj, const wchar_t* name)
    {
        if (obj && name) obj->SetName(name);
    }
}

Renderer::Renderer() = default;
Renderer::~Renderer() { Shutdown(); }

std::vector<std::string> Renderer::EnumerateAdapters()
{
    std::vector<std::string> names;
    ComPtr<IDXGIFactory6> f;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&f)))) return names;

    ComPtr<IDXGIAdapter1> a;
    for (UINT i = 0;
         f->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                      IID_PPV_ARGS(&a)) != DXGI_ERROR_NOT_FOUND;
         ++i)
    {
        DXGI_ADAPTER_DESC1 d{};
        a->GetDesc1(&d);
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { a.Reset(); continue; }
        char buf[256];
        size_t n = 0;
        for (; n < 127 && d.Description[n]; ++n) buf[n] = (char)d.Description[n];
        buf[n] = 0;
        names.emplace_back(buf);
        a.Reset();
    }
    return names;
}

bool Renderer::Init(HWND hwnd, int adapterIdx)
{
    hwnd_ = hwnd;

    RECT rc{};
    GetClientRect(hwnd, &rc);
    width_  = (uint32_t)std::max<LONG>(1, rc.right  - rc.left);
    height_ = (uint32_t)std::max<LONG>(1, rc.bottom - rc.top);

    try
    {
        if (!CreateDeviceAndSwap(hwnd, adapterIdx)) return false;
        if (!CreateRenderTargets()) return false;

        ThrowIfFailed(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)),
                      "CreateFence");
        fenceValues_[frameIndex_] = 1;
        fenceEvent_ = CreateEventEx(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
        if (!fenceEvent_) return false;

        ThrowIfFailed(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                 cmdAlloc_[frameIndex_].Get(),
                                                 nullptr,
                                                 IID_PPV_ARGS(&cmdList_)),
                      "CreateCommandList");
        ThrowIfFailed(cmdList_->Close(), "Close initial cmd list");

        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.NumDescriptors = kImGuiSrvHeapSize;
        hd.Type  = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ThrowIfFailed(device_->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&imguiSrvHeap_)),
                      "imgui SRV heap");
        NameObject(imguiSrvHeap_.Get(), L"imguiSrvHeap");
        imguiSrvDescSize_ = device_->GetDescriptorHandleIncrementSize(
                                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        imguiSrvNextSlot_ = 0;

        graphicsMemory_ = std::make_unique<DirectX::GraphicsMemory>(device_.Get());

        if (!shaderc_.Init())
        {
            OutputDebugStringA("ShaderCompiler::Init failed\n");
            return false;
        }
        if (!CreateM2Demo()) return false;

        // LW SRV heap — shader-visible, shared across all LODs.
        D3D12_DESCRIPTOR_HEAP_DESC ld{};
        ld.NumDescriptors = kLwSrvHeapSize;
        ld.Type  = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        ld.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ThrowIfFailed(device_->CreateDescriptorHeap(&ld, IID_PPV_ARGS(&lwSrvHeap_)),
                      "lw SRV heap");
        NameObject(lwSrvHeap_.Get(), L"lwSrvHeap");
        lwSrvDescSize_ = device_->GetDescriptorHandleIncrementSize(
                                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        lwSrvNextSlot_ = 0;

        if (!CreateM4()) return false;
        if (!CreateVisTextures(width_, height_)) return false;
    }
    catch (const std::exception& e)
    {
        OutputDebugStringA(e.what());
        OutputDebugStringA("\n");
        return false;
    }
    return true;
}

bool Renderer::CreateDeviceAndSwap(HWND hwnd, int adapterIdx)
{
    UINT dxgiFlags = 0;
#if defined(_DEBUG)
    {
        ComPtr<ID3D12Debug> dbg;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg))))
        {
            dbg->EnableDebugLayer();
            dxgiFlags |= DXGI_CREATE_FACTORY_DEBUG;
        }
    }
#endif

    ThrowIfFailed(CreateDXGIFactory2(dxgiFlags, IID_PPV_ARGS(&factory_)),
                  "CreateDXGIFactory2");

    ComPtr<IDXGIAdapter1> adapter;
    int wantIdx = adapterIdx;
    UINT seen = 0;
    for (UINT i = 0; ; ++i)
    {
        ComPtr<IDXGIAdapter1> a;
        if (factory_->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                 IID_PPV_ARGS(&a)) == DXGI_ERROR_NOT_FOUND)
            break;
        DXGI_ADAPTER_DESC1 d{};
        a->GetDesc1(&d);
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (wantIdx < 0 || (int)seen == wantIdx) { adapter = a; break; }
        ++seen;
    }
    if (!adapter) return false;

    ThrowIfFailed(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                    IID_PPV_ARGS(&device_)),
                  "D3D12CreateDevice");
    NameObject(device_.Get(), L"device");

#if defined(_DEBUG)
    {
        ComPtr<ID3D12InfoQueue> iq;
        if (SUCCEEDED(device_.As(&iq)))
        {
            iq->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
            iq->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
        }
    }
#endif

    {
        BOOL t = FALSE;
        ComPtr<IDXGIFactory5> f5;
        if (SUCCEEDED(factory_.As(&f5)))
            f5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &t, sizeof(t));
        tearingSupported_ = (t == TRUE);
    }

    {
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        qd.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        ThrowIfFailed(device_->CreateCommandQueue(&qd, IID_PPV_ARGS(&cmdQueue_)),
                      "CreateCommandQueue");
        NameObject(cmdQueue_.Get(), L"cmdQueue");
    }

    {
        DXGI_SWAP_CHAIN_DESC1 sd{};
        sd.Width  = width_;
        sd.Height = height_;
        sd.Format = BackBufferFormat();
        sd.SampleDesc.Count = 1;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.BufferCount = kFrameCount;
        sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        sd.Flags = tearingSupported_ ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

        ComPtr<IDXGISwapChain1> sc1;
        ThrowIfFailed(factory_->CreateSwapChainForHwnd(cmdQueue_.Get(), hwnd, &sd,
                                                      nullptr, nullptr, &sc1),
                      "CreateSwapChainForHwnd");
        ThrowIfFailed(factory_->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER), "MWA");
        ThrowIfFailed(sc1.As(&swap_), "QI IDXGISwapChain3");
        frameIndex_ = swap_->GetCurrentBackBufferIndex();
    }

    for (UINT i = 0; i < kFrameCount; ++i)
    {
        ThrowIfFailed(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                     IID_PPV_ARGS(&cmdAlloc_[i])),
                      "CreateCommandAllocator");
    }
    return true;
}

bool Renderer::CreateRenderTargets()
{
    {
        D3D12_DESCRIPTOR_HEAP_DESC d{};
        d.NumDescriptors = kFrameCount;
        d.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        ThrowIfFailed(device_->CreateDescriptorHeap(&d, IID_PPV_ARGS(&rtvHeap_)),
                      "RTV heap");
        rtvDescSize_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    }
    {
        D3D12_CPU_DESCRIPTOR_HANDLE h = rtvHeap_->GetCPUDescriptorHandleForHeapStart();
        for (UINT i = 0; i < kFrameCount; ++i)
        {
            ThrowIfFailed(swap_->GetBuffer(i, IID_PPV_ARGS(&backBuffers_[i])),
                          "GetBuffer");
            device_->CreateRenderTargetView(backBuffers_[i].Get(), nullptr, h);
            h.ptr += rtvDescSize_;
        }
    }

    {
        D3D12_DESCRIPTOR_HEAP_DESC d{};
        d.NumDescriptors = 1;
        d.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        ThrowIfFailed(device_->CreateDescriptorHeap(&d, IID_PPV_ARGS(&dsvHeap_)),
                      "DSV heap");
    }
    {
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = width_;
        rd.Height = height_;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_D32_FLOAT;
        rd.SampleDesc.Count = 1;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

        D3D12_CLEAR_VALUE cv{};
        cv.Format = DXGI_FORMAT_D32_FLOAT;
        cv.DepthStencil.Depth = 1.0f;

        ThrowIfFailed(device_->CreateCommittedResource(
                          &hp, D3D12_HEAP_FLAG_NONE, &rd,
                          D3D12_RESOURCE_STATE_DEPTH_WRITE, &cv,
                          IID_PPV_ARGS(&depthTex_)),
                      "depth tex");

        D3D12_DEPTH_STENCIL_VIEW_DESC dvd{};
        dvd.Format = DXGI_FORMAT_D32_FLOAT;
        dvd.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        device_->CreateDepthStencilView(depthTex_.Get(), &dvd,
                                        dsvHeap_->GetCPUDescriptorHandleForHeapStart());
    }
    return true;
}

bool Renderer::CreateM2Demo()
{
    // Empty root signature — shader uses only SV_VertexID + immediate consts.
    {
        D3D12_ROOT_SIGNATURE_DESC rsd{};
        rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        ComPtr<ID3DBlob> sig, err;
        HRESULT hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1,
                                                 &sig, &err);
        if (FAILED(hr))
        {
            if (err) OutputDebugStringA((const char*)err->GetBufferPointer());
            return false;
        }
        ThrowIfFailed(device_->CreateRootSignature(0, sig->GetBufferPointer(),
                                                   sig->GetBufferSize(),
                                                   IID_PPV_ARGS(&m2RootSig_)),
                      "m2 rootsig");
        NameObject(m2RootSig_.Get(), L"m2RootSig");
    }

    // Compile vs + ps from shaders/m2_demo.hlsl.
    ComPtr<IDxcBlob> vs, ps;
    std::string err;
    if (!shaderc_.Compile(L"shaders/m2_demo.hlsl", L"vsmain", L"vs_6_0", {}, vs, &err))
    {
        OutputDebugStringA(("M2 vs compile failed: " + err + "\n").c_str());
        return false;
    }
    if (!shaderc_.Compile(L"shaders/m2_demo.hlsl", L"psmain", L"ps_6_0", {}, ps, &err))
    {
        OutputDebugStringA(("M2 ps compile failed: " + err + "\n").c_str());
        return false;
    }

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = m2RootSig_.Get();
    pd.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
    pd.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.SampleMask = UINT_MAX;
    pd.SampleDesc.Count = 1;
    pd.NumRenderTargets = 1;
    pd.RTVFormats[0] = BackBufferFormat();
    pd.DSVFormat = DXGI_FORMAT_D32_FLOAT;

    // Rasterizer — default solid, back-cull, but we want no cull for a simple
    // demo so winding doesn't matter.
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.RasterizerState.FrontCounterClockwise = FALSE;
    pd.RasterizerState.DepthClipEnable = TRUE;

    // Blend — opaque, no blend.
    for (auto& rt : pd.BlendState.RenderTarget)
    {
        rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    }

    // Depth — write+test off (drawing before depth-using passes anyway).
    pd.DepthStencilState.DepthEnable = FALSE;
    pd.DepthStencilState.StencilEnable = FALSE;

    ThrowIfFailed(device_->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&m2Pso_)),
                  "m2 PSO");
    NameObject(m2Pso_.Get(), L"m2Pso");
    return true;
}

void Renderer::ImGuiSrvAlloc(D3D12_CPU_DESCRIPTOR_HANDLE* outCpu,
                             D3D12_GPU_DESCRIPTOR_HANDLE* outGpu)
{
    assert(imguiSrvHeap_ && imguiSrvNextSlot_ < kImGuiSrvHeapSize);
    UINT slot = imguiSrvNextSlot_++;
    D3D12_CPU_DESCRIPTOR_HANDLE c = imguiSrvHeap_->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE g = imguiSrvHeap_->GetGPUDescriptorHandleForHeapStart();
    c.ptr += SIZE_T(slot) * imguiSrvDescSize_;
    g.ptr += UINT64(slot) * imguiSrvDescSize_;
    if (outCpu) *outCpu = c;
    if (outGpu) *outGpu = g;
}

void Renderer::ImGuiSrvFree(D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE)
{
    // Bump allocator — no per-slot free. Heap reclaimed on Shutdown().
}

void Renderer::Resize(uint32_t w, uint32_t h)
{
    if (!device_ || !swap_) return;
    if (w == width_ && h == height_) return;
    if (w == 0 || h == 0) return;

    WaitForGpu();

    for (UINT i = 0; i < kFrameCount; ++i) backBuffers_[i].Reset();
    depthTex_.Reset();

    UINT flags = tearingSupported_ ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
    ThrowIfFailed(swap_->ResizeBuffers(kFrameCount, w, h, BackBufferFormat(), flags),
                  "ResizeBuffers");
    width_ = w;
    height_ = h;
    frameIndex_ = swap_->GetCurrentBackBufferIndex();
    CreateRenderTargets();
    if (m4TexHeap_) CreateVisTextures(w, h);
}

void Renderer::BeginFrame(float clear[4], bool skipClear, bool /*skipDsvClear*/)
{
    if (!device_) return;

    HRESULT removed = device_->GetDeviceRemovedReason();
    if (FAILED(removed))
    {
        static bool printed = false;
        if (!printed)
        {
            std::printf("[gpu] DEVICE REMOVED reason=0x%08X\n", (unsigned)removed);
            std::fflush(stdout);
            printed = true;
        }
        return;
    }

#if defined(_DEBUG)
    // Drain D3D12 info queue → stdout so warnings / errors aren't silent.
    {
        ComPtr<ID3D12InfoQueue> iq;
        if (SUCCEEDED(device_.As(&iq)))
        {
            UINT64 n = iq->GetNumStoredMessages();
            for (UINT64 i = 0; i < n; ++i)
            {
                SIZE_T sz = 0;
                iq->GetMessage(i, nullptr, &sz);
                std::vector<char> buf(sz);
                auto* m = reinterpret_cast<D3D12_MESSAGE*>(buf.data());
                if (SUCCEEDED(iq->GetMessage(i, m, &sz)))
                {
                    std::printf("[d3d12] sev=%d id=%d %.*s\n",
                                (int)m->Severity, (int)m->ID,
                                (int)m->DescriptionByteLength, m->pDescription);
                }
            }
            if (n) std::fflush(stdout);
            iq->ClearStoredMessages();
        }
    }
#endif

    auto& alloc = cmdAlloc_[frameIndex_];
    ThrowIfFailed(alloc->Reset(), "alloc reset");
    ThrowIfFailed(cmdList_->Reset(alloc.Get(), nullptr), "list reset");

    // Register the cmd list as the GPU context for MicroProfile so that any
    // MICROPROFILE_SCOPEGPUI() calls during this frame record into it.
    MicroProfileGpuSetContext(cmdList_.Get());

    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = backBuffers_[frameIndex_].Get();
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    b.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList_->ResourceBarrier(1, &b);

    D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtvHeap_->GetCPUDescriptorHandleForHeapStart();
    rtv.ptr += SIZE_T(frameIndex_) * rtvDescSize_;
    D3D12_CPU_DESCRIPTOR_HANDLE dsv = dsvHeap_->GetCPUDescriptorHandleForHeapStart();
    cmdList_->OMSetRenderTargets(1, &rtv, FALSE, &dsv);

    {
        MICROPROFILE_SCOPEGPUI("BeginFrame/Clear", 0xff60a0c0);
        if (!skipClear)
            cmdList_->ClearRenderTargetView(rtv, clear, 0, nullptr);
        cmdList_->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
    }

    D3D12_VIEWPORT vp{ 0, 0, (float)width_, (float)height_, 0.0f, 1.0f };
    D3D12_RECT     sc{ 0, 0, (LONG)width_, (LONG)height_ };
    cmdList_->RSSetViewports(1, &vp);
    cmdList_->RSSetScissorRects(1, &sc);

    ID3D12DescriptorHeap* heaps[] = { imguiSrvHeap_.Get() };
    cmdList_->SetDescriptorHeaps(1, heaps);

    // M2 demo only when LW world isn't loaded — once we have voxels, draw those.
    if (m2Pso_ && !lwHasWorld_)
    {
        cmdList_->SetGraphicsRootSignature(m2RootSig_.Get());
        cmdList_->SetPipelineState(m2Pso_.Get());
        cmdList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmdList_->DrawInstanced(3, 1, 0, 0);
    }
}

void Renderer::EndFrame(bool vsync)
{
    if (!device_) return;

    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = backBuffers_[frameIndex_].Get();
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    b.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList_->ResourceBarrier(1, &b);

    ThrowIfFailed(cmdList_->Close(), "cmdList close");
    ID3D12CommandList* lists[] = { cmdList_.Get() };
    cmdQueue_->ExecuteCommandLists(1, lists);

    UINT syncInterval = vsync ? 1 : 0;
    UINT presentFlags = (!vsync && tearingSupported_) ? DXGI_PRESENT_ALLOW_TEARING : 0;
    swap_->Present(syncInterval, presentFlags);

    if (graphicsMemory_) graphicsMemory_->Commit(cmdQueue_.Get());

    MoveToNextFrame();
}

void Renderer::WaitForGpu()
{
    if (!cmdQueue_ || !fence_) return;
    const UINT64 v = fenceValues_[frameIndex_];
    ThrowIfFailed(cmdQueue_->Signal(fence_.Get(), v), "Signal");
    if (fence_->GetCompletedValue() < v)
    {
        ThrowIfFailed(fence_->SetEventOnCompletion(v, fenceEvent_), "SetEventOnCompletion");
        WaitForSingleObjectEx(fenceEvent_, INFINITE, FALSE);
    }
    fenceValues_[frameIndex_] = v + 1;
}

void Renderer::MoveToNextFrame()
{
    const UINT64 sig = fenceValues_[frameIndex_];
    ThrowIfFailed(cmdQueue_->Signal(fence_.Get(), sig), "Signal");
    frameIndex_ = swap_->GetCurrentBackBufferIndex();
    if (fence_->GetCompletedValue() < fenceValues_[frameIndex_])
    {
        ThrowIfFailed(fence_->SetEventOnCompletion(fenceValues_[frameIndex_], fenceEvent_),
                      "SetEventOnCompletion");
        WaitForSingleObjectEx(fenceEvent_, INFINITE, FALSE);
    }
    fenceValues_[frameIndex_] = sig + 1;
}

void Renderer::Shutdown()
{
    if (cmdQueue_ && fence_) WaitForGpu();
    if (fenceEvent_) { CloseHandle(fenceEvent_); fenceEvent_ = nullptr; }
    ClearLwWorld();
    lwSrvHeap_.Reset();
    graphicsMemory_.reset();
    cmdList_.Reset();
    for (auto& a : cmdAlloc_) a.Reset();
    for (auto& b : backBuffers_) b.Reset();
    rtvHeap_.Reset();
    dsvHeap_.Reset();
    depthTex_.Reset();
    imguiSrvHeap_.Reset();
    m2Pso_.Reset();
    m2RootSig_.Reset();
    cmdQueue_.Reset();
    swap_.Reset();
    fence_.Reset();
    device_.Reset();
    factory_.Reset();
}

// ===== M3 — LW upload =======================================================
namespace
{
    // Synchronous batched buffer uploader. One fence wait per Flush().
    // Records CopyBufferRegion + transition to NON_PIXEL_SHADER_RESOURCE for
    // each Upload() call into a private cmd list. Releases upload-heap scratch
    // resources after Flush() (fence-waited).
    class BufferUploader
    {
    public:
        bool Init(ID3D12Device* dev, ID3D12CommandQueue* q)
        {
            device_ = dev; queue_ = q;
            if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                   IID_PPV_ARGS(&alloc_)))) return false;
            if (FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                              alloc_.Get(), nullptr,
                                              IID_PPV_ARGS(&cmd_)))) return false;
            // CreateCommandList opens the list in recording state. Close it so
            // Begin()'s alloc->Reset() doesn't error out (alloc reset is illegal
            // while any associated list is still recording).
            if (FAILED(cmd_->Close())) return false;
            if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)))) return false;
            evt_ = CreateEventEx(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
            return evt_ != nullptr;
        }

        ~BufferUploader()
        {
            if (evt_) CloseHandle(evt_);
        }

        bool Begin()
        {
            recording_ = true;
            scratch_.clear();
            if (FAILED(alloc_->Reset())) return false;
            if (FAILED(cmd_->Reset(alloc_.Get(), nullptr))) return false;
            return true;
        }

        bool Upload(const void* data, size_t bytes,
                    Microsoft::WRL::ComPtr<ID3D12Resource>& outDefault)
        {
            if (!recording_ || bytes == 0) return false;

            // Default-heap committed resource (final home).
            D3D12_HEAP_PROPERTIES hpDef{}; hpDef.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC rd{};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            rd.Width = bytes;
            rd.Height = 1;
            rd.DepthOrArraySize = 1;
            rd.MipLevels = 1;
            rd.SampleDesc.Count = 1;
            rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            if (FAILED(device_->CreateCommittedResource(
                          &hpDef, D3D12_HEAP_FLAG_NONE, &rd,
                          D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                          IID_PPV_ARGS(&outDefault)))) return false;

            // Upload-heap committed resource (scratch).
            D3D12_HEAP_PROPERTIES hpUp{}; hpUp.Type = D3D12_HEAP_TYPE_UPLOAD;
            Microsoft::WRL::ComPtr<ID3D12Resource> up;
            if (FAILED(device_->CreateCommittedResource(
                          &hpUp, D3D12_HEAP_FLAG_NONE, &rd,
                          D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                          IID_PPV_ARGS(&up)))) return false;

            // Map + memcpy.
            void* mapped = nullptr;
            D3D12_RANGE noRead{0, 0};
            if (FAILED(up->Map(0, &noRead, &mapped))) return false;
            std::memcpy(mapped, data, bytes);
            up->Unmap(0, nullptr);

            // Record copy + transition.
            cmd_->CopyBufferRegion(outDefault.Get(), 0, up.Get(), 0, bytes);
            D3D12_RESOURCE_BARRIER b{};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource = outDefault.Get();
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            b.Transition.StateAfter  = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                                     | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            cmd_->ResourceBarrier(1, &b);

            scratch_.push_back(std::move(up));
            return true;
        }

        bool Flush()
        {
            if (!recording_) return true;
            recording_ = false;
            if (FAILED(cmd_->Close())) return false;
            ID3D12CommandList* lists[] = { cmd_.Get() };
            queue_->ExecuteCommandLists(1, lists);
            const UINT64 v = ++fenceVal_;
            if (FAILED(queue_->Signal(fence_.Get(), v))) return false;
            if (fence_->GetCompletedValue() < v)
            {
                if (FAILED(fence_->SetEventOnCompletion(v, evt_))) return false;
                WaitForSingleObjectEx(evt_, INFINITE, FALSE);
            }
            scratch_.clear(); // upload heap resources safe to release now
            return true;
        }

    private:
        ID3D12Device* device_ = nullptr;
        ID3D12CommandQueue* queue_ = nullptr;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> alloc_;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> cmd_;
        Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
        UINT64 fenceVal_ = 0;
        HANDLE evt_ = nullptr;
        bool recording_ = false;
        std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> scratch_;
    };
}

void Renderer::ClearLwWorld()
{
    // Same as UploadLwLod: GPU may still hold references to these resources.
    bool anyLive = false;
    for (int L = 0; L < lw::kLodCount; ++L)
        if (lwGpu_[L].chunkInfoSb) { anyLive = true; break; }
    if (anyLive) WaitForGpu();
    for (int L = 0; L < lw::kLodCount; ++L)
        lwGpu_[L] = LwGpu{};
    // Bump-allocator reset: M3 doesn't free per-LOD slots, so a full ClearLwWorld
    // reclaims all of them at once. Sufficient for one-world-at-a-time workflow.
    lwSrvNextSlot_ = 0;
    lwHasWorld_ = false;
}

// Allocates a single SRV slot in lwSrvHeap_; returns slot index and writes the
// SRV. Caller provides the structured-buffer view desc.
static uint32_t CreateStructuredBufferSrv(ID3D12Device* dev,
                                          ID3D12DescriptorHeap* heap,
                                          UINT descSize,
                                          UINT& nextSlot,
                                          ID3D12Resource* buf,
                                          UINT numElements,
                                          UINT stride)
{
    if (!buf || numElements == 0) return UINT32_MAX;
    D3D12_SHADER_RESOURCE_VIEW_DESC d{};
    d.Format = DXGI_FORMAT_UNKNOWN;
    d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d.Buffer.NumElements = numElements;
    d.Buffer.StructureByteStride = stride;
    const uint32_t slot = nextSlot++;
    D3D12_CPU_DESCRIPTOR_HANDLE h = heap->GetCPUDescriptorHandleForHeapStart();
    h.ptr += SIZE_T(slot) * descSize;
    dev->CreateShaderResourceView(buf, &d, h);
    return slot;
}

bool Renderer::UploadLwLod(const lw::World& w, int L)
{
    if (!device_) return false;
    if (L < 0 || L >= lw::kLodCount) return false;
    const lw::LODWorld& src = w.lods[L];
    LwGpu& g = lwGpu_[L];
    // Old GPU buffers for this LOD may still be referenced by frames in flight.
    // Releasing the ComPtrs here would free memory the GPU is reading -> TDR.
    // Flush before reset. Slow during stream but correct; deferred-delete via
    // fence-tagged retention list is M4f+ optimization.
    if (g.chunkInfoSb) WaitForGpu();
    g = LwGpu{};
    const uint32_t slotCount = (uint32_t)src.chunks.size();
    if (slotCount == 0) return true;
    if (slotCount > lw::kMaxResidentChunksPerLod)
    {
        std::fprintf(stderr, "[lw] LOD %d has %u chunks > max %u\n",
                     L, slotCount, lw::kMaxResidentChunksPerLod);
        return false;
    }

    BufferUploader up;
    if (!up.Init(device_.Get(), cmdQueue_.Get())) return false;
    if (!up.Begin()) return false;

    // ---- ChunkInfo: one entry per slot ----
    std::vector<lw::GpuChunkInfo> infos(slotCount);
    for (uint32_t i = 0; i < slotCount; ++i)
    {
        const lw::RuntimeChunk& rc = src.chunks[i];
        lw::GpuChunkInfo& gi = infos[i];
        gi.worldOriginX = (float)rc.worldOriginX;
        gi.worldOriginY = (float)rc.worldOriginY;
        gi.worldOriginZ = (float)rc.worldOriginZ;
        gi.lodScale = (float)src.lodScale;
        gi._unused0 = 0;
        gi.paletteBase = i * lw::kPaletteSize;
        gi._pad[0] = gi._pad[1] = 0;
    }
    if (!up.Upload(infos.data(), infos.size() * sizeof(lw::GpuChunkInfo),
                   g.chunkInfoSb)) return false;

    // ---- Palette atlas: slotCount × kPaletteSize uint32 ----
    std::vector<uint32_t> atlas((size_t)slotCount * lw::kPaletteSize, 0u);
    for (uint32_t i = 0; i < slotCount; ++i)
    {
        const lw::RuntimeChunk& rc = src.chunks[i];
        const uint32_t n = std::min(rc.paletteCount, (uint32_t)lw::kPaletteSize);
        std::memcpy(atlas.data() + (size_t)i * lw::kPaletteSize,
                    rc.palette, n * sizeof(uint32_t));
    }
    if (!up.Upload(atlas.data(), atlas.size() * sizeof(uint32_t),
                   g.paletteSb)) return false;

    // ---- BlockPos / BlockCol / BlockVis ----
    const uint64_t posBytes = (uint64_t)src.blockPosPool.size() * sizeof(lw::BlockPos);
    const uint64_t colBytes = (uint64_t)src.blockColPool.size() * sizeof(lw::BlockCol);
    const uint64_t visBytes = (uint64_t)src.blockVisPool.size() * sizeof(lw::BlockVis);
    if (posBytes > 0 && !up.Upload(src.blockPosPool.data(), posBytes, g.blockPosSb))
        return false;
    if (colBytes > 0 && !up.Upload(src.blockColPool.data(), colBytes, g.blockColSb))
        return false;
    if (visBytes > 0 && !up.Upload(src.blockVisPool.data(), visBytes, g.blockVisSb))
        return false;

    if (!up.Flush()) return false;

    // Build SRVs on lwSrvHeap_.
    auto* dev = device_.Get();
    auto* heap = lwSrvHeap_.Get();
    g.chunkInfoSrv = CreateStructuredBufferSrv(dev, heap, lwSrvDescSize_, lwSrvNextSlot_,
        g.chunkInfoSb.Get(), (UINT)infos.size(), (UINT)sizeof(lw::GpuChunkInfo));
    g.paletteSrv = CreateStructuredBufferSrv(dev, heap, lwSrvDescSize_, lwSrvNextSlot_,
        g.paletteSb.Get(), (UINT)atlas.size(), (UINT)sizeof(uint32_t));
    if (posBytes)
        g.blockPosSrv = CreateStructuredBufferSrv(dev, heap, lwSrvDescSize_, lwSrvNextSlot_,
            g.blockPosSb.Get(), (UINT)src.blockPosPool.size(), (UINT)sizeof(lw::BlockPos));
    if (colBytes)
        g.blockColSrv = CreateStructuredBufferSrv(dev, heap, lwSrvDescSize_, lwSrvNextSlot_,
            g.blockColSb.Get(), (UINT)src.blockColPool.size(), (UINT)sizeof(lw::BlockCol));
    if (visBytes)
        g.blockVisSrv = CreateStructuredBufferSrv(dev, heap, lwSrvDescSize_, lwSrvNextSlot_,
            g.blockVisSb.Get(), (UINT)src.blockVisPool.size(), (UINT)sizeof(lw::BlockVis));

    g.slotCount = slotCount;
    g.blockCount = (uint32_t)src.blockPosPool.size();
    g.bytes = infos.size() * sizeof(lw::GpuChunkInfo)
            + atlas.size() * sizeof(uint32_t)
            + posBytes + colBytes + visBytes;
    std::printf("[lw] LOD %d uploaded: %u slots, %u blocks, %.2f MB GPU\n",
                L, g.slotCount, g.blockCount, g.bytes / (1024.0 * 1024.0));
    lwHasWorld_ = true;
    return true;
}

bool Renderer::UploadLwWorld(const lw::World& w)
{
    if (!device_) return false;
    ClearLwWorld();
    lwWorld_ = w;
    for (int L = 0; L < lw::kLodCount; ++L)
    {
        if (!UploadLwLod(w, L)) return false;
        lwWorld_.lods[L].blockPosPool.clear();
        lwWorld_.lods[L].blockPosPool.shrink_to_fit();
        lwWorld_.lods[L].blockColPool.clear();
        lwWorld_.lods[L].blockColPool.shrink_to_fit();
        lwWorld_.lods[L].blockVisPool.clear();
        lwWorld_.lods[L].blockVisPool.shrink_to_fit();
    }
    return true;
}

bool Renderer::UploadLwLodOnly(const lw::World& w, int L)
{
    if (L < 0 || L >= lw::kLodCount) return false;
    lwWorld_.lods[L] = w.lods[L];
    return UploadLwLod(w, L);
}

void Renderer::PrepLwWorld(const lw::World& w)
{
    ClearLwWorld();
    for (int i = 0; i < 3; ++i)
    {
        lwWorld_.worldAabbMin[i] = w.worldAabbMin[i];
        lwWorld_.worldAabbMax[i] = w.worldAabbMax[i];
    }
    lwHasWorld_ = true;
}

// ===== Shader hot reload ====================================================
static uint64_t LatestShaderMtime()
{
    const wchar_t* paths[] = {
        L"shaders/m4_lw.hlsl",
        L"shaders/m4_taa_post.hlsl",
    };
    uint64_t mx = 0;
    for (auto* p : paths)
    {
        HANDLE f = CreateFileW(p, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (f == INVALID_HANDLE_VALUE) continue;
        FILETIME ft{};
        if (GetFileTime(f, nullptr, nullptr, &ft))
        {
            uint64_t t = ((uint64_t)ft.dwHighDateTime << 32) | (uint64_t)ft.dwLowDateTime;
            if (t > mx) mx = t;
        }
        CloseHandle(f);
    }
    return mx;
}

void Renderer::PollShaderHotReload()
{
    uint64_t now = LatestShaderMtime();
    if (lastShaderMtime_ == 0) { lastShaderMtime_ = now; return; }
    if (now > lastShaderMtime_)
    {
        lastShaderMtime_ = now;
        if (RecompileShaders())
        {
            lastReloadStatus_ = ReloadStatus::Success;
            ++reloadCount_;
            std::printf("[hot] shader reload OK (count=%u)\n", (unsigned)reloadCount_);
        }
        else
        {
            lastReloadStatus_ = ReloadStatus::Failed;
            std::printf("[hot] shader reload FAILED\n");
        }
        std::fflush(stdout);
    }
}

bool Renderer::ForceReloadShaders()
{
    lastShaderMtime_ = 0; // force next Poll to detect "changed"
    return RecompileShaders();
}

bool Renderer::RecompileShaders()
{
    // Recompile each entry. Build new PSOs into temp ComPtrs. Only swap in if
    // EVERY compile succeeded — partial reload is worse than none.
    ComPtr<IDxcBlob> cs1, cs2, csDilate, vsR, psR, vsTaa, psTaa, psPost, psGrMark, psGrBlur;
    std::string err;
    auto comp = [&](const wchar_t* path, const wchar_t* entry, const wchar_t* profile,
                    ComPtr<IDxcBlob>& out, const char* tag) -> bool
    {
        if (!shaderc_.Compile(path, entry, profile, {}, out, &err)) {
            std::printf("[hot] %s compile fail:\n%s\n", tag, err.c_str());
            std::fflush(stdout);
            return false;
        }
        return true;
    };
    if (!comp(L"shaders/m4_lw.hlsl",        L"csmain_pass1_depth", L"cs_6_0", cs1, "cs1")) return false;
    if (!comp(L"shaders/m4_lw.hlsl",        L"csmain_pass2_color", L"cs_6_0", cs2, "cs2")) return false;
    if (!comp(L"shaders/m4_lw.hlsl",        L"csmain_dilate",      L"cs_6_0", csDilate, "csDilate")) return false;
    if (!comp(L"shaders/m4_lw.hlsl",        L"vsmain_resolve",     L"vs_6_0", vsR,   "vsR")) return false;
    if (!comp(L"shaders/m4_lw.hlsl",        L"psmain_resolve",     L"ps_6_0", psR,   "psR")) return false;
    if (!comp(L"shaders/m4_taa_post.hlsl",  L"vsmain_taa",         L"vs_6_0", vsTaa, "vsTaa")) return false;
    if (!comp(L"shaders/m4_taa_post.hlsl",  L"psmain_taa",         L"ps_6_0", psTaa, "psTaa")) return false;
    if (!comp(L"shaders/m4_taa_post.hlsl",  L"psmain_post",        L"ps_6_0", psPost,"psPost")) return false;
    if (!comp(L"shaders/m4_taa_post.hlsl",  L"psmain_godray_mark", L"ps_6_0", psGrMark, "psGrMark")) return false;
    if (!comp(L"shaders/m4_taa_post.hlsl",  L"psmain_godray_blur", L"ps_6_0", psGrBlur, "psGrBlur")) return false;

    // Wait for GPU idle before swapping PSOs — old PSOs may still be in flight.
    WaitForGpu();

    auto buildCs = [&](ID3D12RootSignature* rs, IDxcBlob* cs,
                       ComPtr<ID3D12PipelineState>& outNew)
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = rs;
        pd.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
        return SUCCEEDED(device_->CreateComputePipelineState(&pd, IID_PPV_ARGS(&outNew)));
    };
    auto buildGfx = [&](IDxcBlob* vs, IDxcBlob* ps, DXGI_FORMAT rt, bool useDsv,
                        ID3D12RootSignature* rs,
                        ComPtr<ID3D12PipelineState>& outNew)
    {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = rs;
        pd.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
        pd.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };
        pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pd.SampleMask = UINT_MAX;
        pd.SampleDesc.Count = 1;
        pd.NumRenderTargets = 1;
        pd.RTVFormats[0] = rt;
        pd.DSVFormat = useDsv ? DXGI_FORMAT_D32_FLOAT : DXGI_FORMAT_UNKNOWN;
        pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pd.RasterizerState.DepthClipEnable = TRUE;
        for (auto& b : pd.BlendState.RenderTarget) b.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pd.DepthStencilState.DepthEnable = FALSE;
        pd.DepthStencilState.StencilEnable = FALSE;
        return SUCCEEDED(device_->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&outNew)));
    };

    ComPtr<ID3D12PipelineState> p1, p2, pd, prs, ptaa, ppost, pgm, pgb;
    if (!buildCs(m4Pass1RootSig_.Get(),  cs1.Get(),      p1))   return false;
    if (!buildCs(m4Pass2RootSig_.Get(),  cs2.Get(),      p2))   return false;
    if (!buildCs(m4DilateRootSig_.Get(), csDilate.Get(), pd))   return false;
    if (!buildGfx(vsR.Get(),   psR.Get(),  DXGI_FORMAT_R16G16B16A16_FLOAT, false, m4ResolveRootSig_.Get(), prs))   return false;
    if (!buildGfx(vsTaa.Get(), psTaa.Get(),DXGI_FORMAT_R16G16B16A16_FLOAT, false, m4TaaRootSig_.Get(),     ptaa))  return false;
    if (!buildGfx(vsTaa.Get(), psPost.Get(),BackBufferFormat(),         true,  m4PostRootSig_.Get(),    ppost)) return false;
    if (!buildGfx(vsTaa.Get(), psGrMark.Get(),DXGI_FORMAT_R16_FLOAT,    false, m4GodrayMarkRootSig_.Get(), pgm)) return false;
    if (!buildGfx(vsTaa.Get(), psGrBlur.Get(),DXGI_FORMAT_R16_FLOAT,    false, m4GodrayBlurRootSig_.Get(), pgb)) return false;

    m4Pass1Pso_       = p1;
    m4Pass2Pso_       = p2;
    m4DilatePso_      = pd;
    m4ResolvePso_     = prs;
    m4TaaPso_         = ptaa;
    m4PostPso_        = ppost;
    m4GodrayMarkPso_  = pgm;
    m4GodrayBlurPso_  = pgb;
    return true;
}

// ===== M4 — PointCS_Block compute rasterizer ================================

bool Renderer::CreateM4()
{
    D3D12_DESCRIPTOR_HEAP_DESC td{};
    td.NumDescriptors = 14;
    td.Type  = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    td.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(device_->CreateDescriptorHeap(&td, IID_PPV_ARGS(&m4TexHeap_)),
                  "m4 tex heap");
    NameObject(m4TexHeap_.Get(), L"m4TexHeap");

    D3D12_DESCRIPTOR_HEAP_DESC tdc = td;
    tdc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    ThrowIfFailed(device_->CreateDescriptorHeap(&tdc, IID_PPV_ARGS(&m4TexClearHeap_)),
                  "m4 tex clear heap");
    m4TexDescSize_ = device_->GetDescriptorHandleIncrementSize(
                                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // ---- Root signatures ----
    auto buildRootSig = [&](const D3D12_ROOT_SIGNATURE_DESC& rsd,
                            ComPtr<ID3D12RootSignature>& out, const wchar_t* name)
    {
        ComPtr<ID3DBlob> sig, err;
        HRESULT hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1,
                                                 &sig, &err);
        if (FAILED(hr))
        {
            if (err) OutputDebugStringA((const char*)err->GetBufferPointer());
            return false;
        }
        if (FAILED(device_->CreateRootSignature(0, sig->GetBufferPointer(),
                                                sig->GetBufferSize(),
                                                IID_PPV_ARGS(&out)))) return false;
        NameObject(out.Get(), name);
        return true;
    };

    // Pass1 root sig: 2 CBVs, 3 root SRVs, 1 UAV descriptor table.
    {
        D3D12_DESCRIPTOR_RANGE uavRange{};
        uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        uavRange.NumDescriptors = 1;
        uavRange.BaseShaderRegister = 0;
        uavRange.OffsetInDescriptorsFromTableStart = 0;

        D3D12_ROOT_PARAMETER p[6]{};
        p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        p[0].Descriptor = { 0, 0 };  // b0
        p[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        p[1].Descriptor = { 1, 0 };  // b1
        p[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        p[2].Descriptor = { 0, 0 };  // t0 chunkInfo
        p[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        p[3].Descriptor = { 2, 0 };  // t2 blockPos
        p[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        p[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        p[4].Descriptor = { 4, 0 };  // t4 workItems
        p[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        p[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[5].DescriptorTable = { 1, &uavRange };
        p[5].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_ROOT_SIGNATURE_DESC rsd{};
        rsd.NumParameters = 6;
        rsd.pParameters = p;
        if (!buildRootSig(rsd, m4Pass1RootSig_, L"m4Pass1RootSig")) return false;
    }

    // Pass2 root sig: 2 CBVs, 5 root SRVs, 2 descriptor tables.
    {
        D3D12_DESCRIPTOR_RANGE srvRange{};
        srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        srvRange.NumDescriptors = 1;
        srvRange.BaseShaderRegister = 5;  // t5
        srvRange.OffsetInDescriptorsFromTableStart = 0;
        D3D12_DESCRIPTOR_RANGE uavRange{};
        uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        uavRange.NumDescriptors = 1;
        uavRange.BaseShaderRegister = 0;  // u0
        uavRange.OffsetInDescriptorsFromTableStart = 0;

        D3D12_ROOT_PARAMETER p[10]{};
        p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[0].Descriptor = {0,0};
        p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[1].Descriptor = {1,0};
        p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; p[2].Descriptor = {0,0}; // chunkInfo
        p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; p[3].Descriptor = {1,0}; // palette
        p[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; p[4].Descriptor = {2,0}; // blockPos
        p[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; p[5].Descriptor = {3,0}; // blockCol
        p[6].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; p[6].Descriptor = {4,0}; // workItems
        p[7].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[7].DescriptorTable = { 1, &srvRange };
        p[8].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[8].DescriptorTable = { 1, &uavRange };
        p[9].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; p[9].Descriptor = {6,0}; // blockVis
        for (auto& x : p) x.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_ROOT_SIGNATURE_DESC rsd{};
        rsd.NumParameters = 10;
        rsd.pParameters = p;
        if (!buildRootSig(rsd, m4Pass2RootSig_, L"m4Pass2RootSig")) return false;
    }

    // Static linear-clamp sampler at s0 for both resolve & blit.
    D3D12_STATIC_SAMPLER_DESC ssLinear{};
    ssLinear.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    ssLinear.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    ssLinear.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    ssLinear.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    ssLinear.ShaderRegister = 0;
    ssLinear.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    // Resolve root sig: CBV b0 + table(SRV t0..t1, dilated depth+color) +
    // table(SRV t2 taa hist).
    {
        D3D12_DESCRIPTOR_RANGE visRange{};
        visRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        visRange.NumDescriptors = 2;
        visRange.BaseShaderRegister = 0;
        D3D12_DESCRIPTOR_RANGE taaRange{};
        taaRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        taaRange.NumDescriptors = 1;
        taaRange.BaseShaderRegister = 2;
        D3D12_ROOT_PARAMETER p[3]{};
        p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        p[0].Descriptor = { 0, 0 };
        p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[1].DescriptorTable = { 1, &visRange };
        p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[2].DescriptorTable = { 1, &taaRange };
        for (auto& x : p) x.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

        D3D12_ROOT_SIGNATURE_DESC rsd{};
        rsd.NumParameters = 3;
        rsd.pParameters = p;
        rsd.NumStaticSamplers = 1;
        rsd.pStaticSamplers = &ssLinear;
        rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        if (!buildRootSig(rsd, m4ResolveRootSig_, L"m4ResolveRootSig")) return false;
    }

    // Dilate root sig: CBV b0 + CBV b1 + table(SRV t0..t1) + table(UAV u0..u1).
    {
        D3D12_DESCRIPTOR_RANGE srvRange{};
        srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        srvRange.NumDescriptors = 2;
        srvRange.BaseShaderRegister = 0;
        D3D12_DESCRIPTOR_RANGE uavRange{};
        uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        uavRange.NumDescriptors = 2;
        uavRange.BaseShaderRegister = 0;
        D3D12_ROOT_PARAMETER p[4]{};
        p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[0].Descriptor = {0, 0};
        p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[1].Descriptor = {1, 0};
        p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[2].DescriptorTable = {1, &srvRange};
        p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[3].DescriptorTable = {1, &uavRange};
        for (auto& x : p) x.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_ROOT_SIGNATURE_DESC rsd{};
        rsd.NumParameters = 4;
        rsd.pParameters = p;
        if (!buildRootSig(rsd, m4DilateRootSig_, L"m4DilateRootSig")) return false;
    }

    // TAA root sig — CSTiles psmain_taa expects t4 (taaScene), t5 (taaHist),
    // t6 (taaDepth), s0 sampler, b0 CB.
    {
        D3D12_DESCRIPTOR_RANGE sceneRange{};
        sceneRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        sceneRange.NumDescriptors = 1;
        sceneRange.BaseShaderRegister = 4;
        D3D12_DESCRIPTOR_RANGE histRange{};
        histRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        histRange.NumDescriptors = 1;
        histRange.BaseShaderRegister = 5;
        D3D12_DESCRIPTOR_RANGE depthRange{};
        depthRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        depthRange.NumDescriptors = 1;
        depthRange.BaseShaderRegister = 6;
        D3D12_ROOT_PARAMETER p[4]{};
        p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[0].Descriptor = {0, 0};
        p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[1].DescriptorTable = {1, &sceneRange};
        p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[2].DescriptorTable = {1, &histRange};
        p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[3].DescriptorTable = {1, &depthRange};
        for (auto& x : p) x.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_ROOT_SIGNATURE_DESC rsd{};
        rsd.NumParameters = 4;
        rsd.pParameters = p;
        rsd.NumStaticSamplers = 1;
        rsd.pStaticSamplers = &ssLinear;
        rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        if (!buildRootSig(rsd, m4TaaRootSig_, L"m4TaaRootSig")) return false;
    }

    // Post root sig — psmain_post needs b0 + b3 (cbGodray) + t6 (postDepth) +
    // t7 (postIn) + t9 (godrayTex) + t10 (godrayHistTex) + s0.
    {
        D3D12_DESCRIPTOR_RANGE depthRange{};
        depthRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        depthRange.NumDescriptors = 1;
        depthRange.BaseShaderRegister = 6;
        D3D12_DESCRIPTOR_RANGE inRange{};
        inRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        inRange.NumDescriptors = 1;
        inRange.BaseShaderRegister = 7;
        D3D12_DESCRIPTOR_RANGE gr9{};
        gr9.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        gr9.NumDescriptors = 1;
        gr9.BaseShaderRegister = 9;
        D3D12_DESCRIPTOR_RANGE gr10{};
        gr10.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        gr10.NumDescriptors = 1;
        gr10.BaseShaderRegister = 10;
        D3D12_ROOT_PARAMETER p[6]{};
        p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[0].Descriptor = {0, 0};
        p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[1].Descriptor = {3, 0};
        p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[2].DescriptorTable = {1, &depthRange};
        p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[3].DescriptorTable = {1, &inRange};
        p[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[4].DescriptorTable = {1, &gr9};
        p[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[5].DescriptorTable = {1, &gr10};
        for (auto& x : p) x.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_ROOT_SIGNATURE_DESC rsd{};
        rsd.NumParameters = 6;
        rsd.pParameters = p;
        rsd.NumStaticSamplers = 1;
        rsd.pStaticSamplers = &ssLinear;
        rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        if (!buildRootSig(rsd, m4PostRootSig_, L"m4PostRootSig")) return false;
    }

    // ---- Compute PSOs ----
    auto buildComputePso = [&](ID3D12RootSignature* rs, IDxcBlob* cs,
                               ComPtr<ID3D12PipelineState>& out, const wchar_t* name)
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = rs;
        pd.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
        if (FAILED(device_->CreateComputePipelineState(&pd, IID_PPV_ARGS(&out)))) return false;
        NameObject(out.Get(), name);
        return true;
    };

    ComPtr<IDxcBlob> cs1, cs2, csDilate, vsR, psR, vsTaa, psTaa, psPost, psGrMark, psGrBlur;
    std::string err;
    if (!shaderc_.Compile(L"shaders/m4_lw.hlsl", L"csmain_pass1_depth", L"cs_6_0", {}, cs1, &err))
    { OutputDebugStringA(("[m4] cs1 compile: " + err + "\n").c_str()); return false; }
    if (!shaderc_.Compile(L"shaders/m4_lw.hlsl", L"csmain_pass2_color", L"cs_6_0", {}, cs2, &err))
    { OutputDebugStringA(("[m4] cs2 compile: " + err + "\n").c_str()); return false; }
    if (!shaderc_.Compile(L"shaders/m4_lw.hlsl", L"csmain_dilate",       L"cs_6_0", {}, csDilate, &err))
    { OutputDebugStringA(("[m4] csDilate compile: " + err + "\n").c_str()); return false; }
    if (!shaderc_.Compile(L"shaders/m4_lw.hlsl", L"vsmain_resolve",    L"vs_6_0", {}, vsR, &err))
    { OutputDebugStringA(("[m4] vsR compile: " + err + "\n").c_str()); return false; }
    if (!shaderc_.Compile(L"shaders/m4_lw.hlsl", L"psmain_resolve",    L"ps_6_0", {}, psR, &err))
    { OutputDebugStringA(("[m4] psR compile: " + err + "\n").c_str()); return false; }
    if (!shaderc_.Compile(L"shaders/m4_taa_post.hlsl", L"vsmain_taa", L"vs_6_0", {}, vsTaa, &err))
    { OutputDebugStringA(("[m4] vsTaa compile: " + err + "\n").c_str()); return false; }
    if (!shaderc_.Compile(L"shaders/m4_taa_post.hlsl", L"psmain_taa", L"ps_6_0", {}, psTaa, &err))
    { OutputDebugStringA(("[m4] psTaa compile: " + err + "\n").c_str()); return false; }
    if (!shaderc_.Compile(L"shaders/m4_taa_post.hlsl", L"psmain_post", L"ps_6_0", {}, psPost, &err))
    { OutputDebugStringA(("[m4] psPost compile: " + err + "\n").c_str()); return false; }
    if (!shaderc_.Compile(L"shaders/m4_taa_post.hlsl", L"psmain_godray_mark", L"ps_6_0", {}, psGrMark, &err))
    { OutputDebugStringA(("[m4] psGrMark compile: " + err + "\n").c_str()); return false; }
    if (!shaderc_.Compile(L"shaders/m4_taa_post.hlsl", L"psmain_godray_blur", L"ps_6_0", {}, psGrBlur, &err))
    { OutputDebugStringA(("[m4] psGrBlur compile: " + err + "\n").c_str()); return false; }

    if (!buildComputePso(m4Pass1RootSig_.Get(), cs1.Get(), m4Pass1Pso_, L"m4Pass1Pso")) return false;
    if (!buildComputePso(m4Pass2RootSig_.Get(), cs2.Get(), m4Pass2Pso_, L"m4Pass2Pso")) return false;
    if (!buildComputePso(m4DilateRootSig_.Get(), csDilate.Get(), m4DilatePso_, L"m4DilatePso")) return false;

    // Fullscreen-triangle PSO builder. Takes vs + ps + RT format + rootsig.
    auto buildFsTri = [&](IDxcBlob* vs, IDxcBlob* ps, DXGI_FORMAT rt, bool useDsv,
                          ID3D12RootSignature* rs,
                          ComPtr<ID3D12PipelineState>& out, const wchar_t* name) -> bool
    {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = rs;
        pd.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
        pd.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };
        pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pd.SampleMask = UINT_MAX;
        pd.SampleDesc.Count = 1;
        pd.NumRenderTargets = 1;
        pd.RTVFormats[0] = rt;
        pd.DSVFormat = useDsv ? DXGI_FORMAT_D32_FLOAT : DXGI_FORMAT_UNKNOWN;
        pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pd.RasterizerState.DepthClipEnable = TRUE;
        for (auto& blend : pd.BlendState.RenderTarget)
            blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pd.DepthStencilState.DepthEnable = FALSE;
        pd.DepthStencilState.StencilEnable = FALSE;
        if (FAILED(device_->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&out)))) return false;
        NameObject(out.Get(), name);
        return true;
    };
    if (!buildFsTri(vsR.Get(), psR.Get(),    DXGI_FORMAT_R16G16B16A16_FLOAT, false,
                    m4ResolveRootSig_.Get(), m4ResolvePso_, L"m4ResolvePso")) return false;
    if (!buildFsTri(vsTaa.Get(), psTaa.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT, false,
                    m4TaaRootSig_.Get(), m4TaaPso_, L"m4TaaPso")) return false;
    if (!buildFsTri(vsTaa.Get(), psPost.Get(), BackBufferFormat(),         true,
                    m4PostRootSig_.Get(), m4PostPso_, L"m4PostPso")) return false;

    // Godray Mark root sig: CBV b0 (frame for gScreenSize), CBV b3 (godray),
    // table SRV t6 (gPostDepth = visDepth2), static linear sampler s0.
    {
        D3D12_DESCRIPTOR_RANGE depthRange{};
        depthRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        depthRange.NumDescriptors = 1;
        depthRange.BaseShaderRegister = 6;
        D3D12_ROOT_PARAMETER p[3]{};
        p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[0].Descriptor = {0,0};
        p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[1].Descriptor = {3,0};
        p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[2].DescriptorTable = {1, &depthRange};
        for (auto& x : p) x.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_ROOT_SIGNATURE_DESC rsd{};
        rsd.NumParameters = 3;
        rsd.pParameters = p;
        rsd.NumStaticSamplers = 1;
        rsd.pStaticSamplers = &ssLinear;
        rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        if (!buildRootSig(rsd, m4GodrayMarkRootSig_, L"m4GodrayMarkRootSig")) return false;
    }

    // Godray Blur root sig: CBV b3 (godray), table SRV t9 (mark), table SRV t10 (prev blur),
    // static linear sampler s0 (mapped to gTaaSamp in shader).
    {
        D3D12_DESCRIPTOR_RANGE markRange{};
        markRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        markRange.NumDescriptors = 1;
        markRange.BaseShaderRegister = 9;
        D3D12_DESCRIPTOR_RANGE histRange{};
        histRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        histRange.NumDescriptors = 1;
        histRange.BaseShaderRegister = 10;
        D3D12_ROOT_PARAMETER p[3]{};
        p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[0].Descriptor = {3,0};
        p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[1].DescriptorTable = {1, &markRange};
        p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[2].DescriptorTable = {1, &histRange};
        for (auto& x : p) x.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_ROOT_SIGNATURE_DESC rsd{};
        rsd.NumParameters = 3;
        rsd.pParameters = p;
        rsd.NumStaticSamplers = 1;
        rsd.pStaticSamplers = &ssLinear;
        rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        if (!buildRootSig(rsd, m4GodrayBlurRootSig_, L"m4GodrayBlurRootSig")) return false;
    }

    if (!buildFsTri(vsTaa.Get(), psGrMark.Get(), DXGI_FORMAT_R16_FLOAT, false,
                    m4GodrayMarkRootSig_.Get(), m4GodrayMarkPso_, L"m4GodrayMarkPso")) return false;
    if (!buildFsTri(vsTaa.Get(), psGrBlur.Get(), DXGI_FORMAT_R16_FLOAT, false,
                    m4GodrayBlurRootSig_.Get(), m4GodrayBlurPso_, L"m4GodrayBlurPso")) return false;

    // RTV heap: 0,1=taaHist 2=taaScene 3=godrayMark 4,5=godrayBlur[1,2].
    D3D12_DESCRIPTOR_HEAP_DESC rh{};
    rh.NumDescriptors = 6;
    rh.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    ThrowIfFailed(device_->CreateDescriptorHeap(&rh, IID_PPV_ARGS(&taaRtvHeap_)),
                  "taa RTV heap");
    taaRtvDescSize_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    return true;
}

bool Renderer::CreateVisTextures(uint32_t w, uint32_t h)
{
    if (w == 0 || h == 0) return false;
    WaitForGpu();
    visDepthTex_.Reset();
    visColorTex_.Reset();
    visColor2Tex_.Reset();
    visDepth2Tex_.Reset();
    taaHistTex_[0].Reset();
    taaHistTex_[1].Reset();
    taaSceneTex_.Reset();
    godrayTex_[0].Reset();
    godrayTex_[1].Reset();
    godrayTex_[2].Reset();

    auto createUintTex = [&](ComPtr<ID3D12Resource>& out, const wchar_t* name)
    {
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = w; rd.Height = h;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_R32_UINT;
        rd.SampleDesc.Count = 1;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (FAILED(device_->CreateCommittedResource(
                       &hp, D3D12_HEAP_FLAG_NONE, &rd,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                       IID_PPV_ARGS(&out)))) return false;
        NameObject(out.Get(), name);
        return true;
    };
    auto createRtTex = [&](ComPtr<ID3D12Resource>& out, const wchar_t* name)
    {
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = w; rd.Height = h;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        rd.SampleDesc.Count = 1;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_CLEAR_VALUE cv{};
        cv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        if (FAILED(device_->CreateCommittedResource(
                       &hp, D3D12_HEAP_FLAG_NONE, &rd,
                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &cv,
                       IID_PPV_ARGS(&out)))) return false;
        NameObject(out.Get(), name);
        return true;
    };
    if (!createUintTex(visDepthTex_,  L"visDepthTex"))  return false;
    if (!createUintTex(visColorTex_,  L"visColorTex"))  return false;
    if (!createUintTex(visColor2Tex_, L"visColor2Tex")) return false;
    // visDepth2 is R32_FLOAT (CSTiles depth = gNearZ/viewZ).
    {
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = w; rd.Height = h;
        rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_R32_FLOAT;
        rd.SampleDesc.Count = 1;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (FAILED(device_->CreateCommittedResource(
                       &hp, D3D12_HEAP_FLAG_NONE, &rd,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                       IID_PPV_ARGS(&visDepth2Tex_)))) return false;
        NameObject(visDepth2Tex_.Get(), L"visDepth2Tex");
    }
    // taaScene = R11G11B10F RT for resolve output (consumed by psmain_taa).
    {
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = w; rd.Height = h;
        rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        rd.SampleDesc.Count = 1;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_CLEAR_VALUE cv{}; cv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        if (FAILED(device_->CreateCommittedResource(
                       &hp, D3D12_HEAP_FLAG_NONE, &rd,
                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &cv,
                       IID_PPV_ARGS(&taaSceneTex_)))) return false;
        NameObject(taaSceneTex_.Get(), L"taaSceneTex");
    }
    // Godray textures: 64x64 R16F. [0]=mark, [1,2]=blur ping-pong.
    for (int g = 0; g < 3; ++g)
    {
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = 64; rd.Height = 64;
        rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_R16_FLOAT;
        rd.SampleDesc.Count = 1;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_CLEAR_VALUE cv{}; cv.Format = DXGI_FORMAT_R16_FLOAT;
        if (FAILED(device_->CreateCommittedResource(
                       &hp, D3D12_HEAP_FLAG_NONE, &rd,
                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &cv,
                       IID_PPV_ARGS(&godrayTex_[g])))) return false;
        wchar_t nm[32]; std::swprintf(nm, 32, L"godrayTex[%d]", g);
        NameObject(godrayTex_[g].Get(), nm);
    }
    if (!createRtTex (taaHistTex_[0], L"taaHist0")) return false;
    if (!createRtTex (taaHistTex_[1], L"taaHist1")) return false;

    auto cpu = [&](ID3D12DescriptorHeap* h, UINT slot) {
        D3D12_CPU_DESCRIPTOR_HANDLE c = h->GetCPUDescriptorHandleForHeapStart();
        c.ptr += SIZE_T(slot) * m4TexDescSize_;
        return c;
    };

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavD{};
    uavD.Format = DXGI_FORMAT_R32_UINT;
    uavD.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    D3D12_SHADER_RESOURCE_VIEW_DESC srvUint{};
    srvUint.Format = DXGI_FORMAT_R32_UINT;
    srvUint.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvUint.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvUint.Texture2D.MipLevels = 1;
    D3D12_SHADER_RESOURCE_VIEW_DESC srvRgba{};
    srvRgba.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    srvRgba.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvRgba.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvRgba.Texture2D.MipLevels = 1;

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavF{};
    uavF.Format = DXGI_FORMAT_R32_FLOAT;
    uavF.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    D3D12_SHADER_RESOURCE_VIEW_DESC srvFloat{};
    srvFloat.Format = DXGI_FORMAT_R32_FLOAT;
    srvFloat.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvFloat.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvFloat.Texture2D.MipLevels = 1;

    for (auto* heap : { m4TexHeap_.Get(), m4TexClearHeap_.Get() })
    {
        device_->CreateUnorderedAccessView(visDepthTex_.Get(),   nullptr, &uavD,    cpu(heap, 0));
        device_->CreateUnorderedAccessView(visColorTex_.Get(),   nullptr, &uavD,    cpu(heap, 1));
        device_->CreateShaderResourceView (visDepthTex_.Get(),   &srvUint,         cpu(heap, 2));
        device_->CreateShaderResourceView (visColorTex_.Get(),   &srvUint,         cpu(heap, 3));
        device_->CreateShaderResourceView (taaHistTex_[0].Get(), &srvRgba,         cpu(heap, 4));
        device_->CreateShaderResourceView (taaHistTex_[1].Get(), &srvRgba,         cpu(heap, 5));
        device_->CreateUnorderedAccessView(visColor2Tex_.Get(),  nullptr, &uavD,    cpu(heap, 6));
        device_->CreateUnorderedAccessView(visDepth2Tex_.Get(),  nullptr, &uavF,    cpu(heap, 7));
        device_->CreateShaderResourceView (visDepth2Tex_.Get(),  &srvFloat,        cpu(heap, 8));
        device_->CreateShaderResourceView (visColor2Tex_.Get(),  &srvUint,         cpu(heap, 9));
        device_->CreateShaderResourceView (taaSceneTex_.Get(),   &srvRgba,         cpu(heap, 10));
        D3D12_SHADER_RESOURCE_VIEW_DESC srvR16{};
        srvR16.Format = DXGI_FORMAT_R16_FLOAT;
        srvR16.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvR16.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvR16.Texture2D.MipLevels = 1;
        device_->CreateShaderResourceView (godrayTex_[0].Get(),  &srvR16,          cpu(heap, 11));
        device_->CreateShaderResourceView (godrayTex_[1].Get(),  &srvR16,          cpu(heap, 12));
        device_->CreateShaderResourceView (godrayTex_[2].Get(),  &srvR16,          cpu(heap, 13));
    }

    // RTVs into taaRtvHeap_: 0,1=taaHist 2=taaScene 3,4,5=godray[0..2].
    {
        D3D12_RENDER_TARGET_VIEW_DESC rtvD{};
        rtvD.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        rtvD.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        D3D12_CPU_DESCRIPTOR_HANDLE h = taaRtvHeap_->GetCPUDescriptorHandleForHeapStart();
        device_->CreateRenderTargetView(taaHistTex_[0].Get(), &rtvD, h); h.ptr += taaRtvDescSize_;
        device_->CreateRenderTargetView(taaHistTex_[1].Get(), &rtvD, h); h.ptr += taaRtvDescSize_;
        device_->CreateRenderTargetView(taaSceneTex_.Get(),   &rtvD, h); h.ptr += taaRtvDescSize_;
        D3D12_RENDER_TARGET_VIEW_DESC rtvGr{};
        rtvGr.Format = DXGI_FORMAT_R16_FLOAT;
        rtvGr.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        device_->CreateRenderTargetView(godrayTex_[0].Get(), &rtvGr, h); h.ptr += taaRtvDescSize_;
        device_->CreateRenderTargetView(godrayTex_[1].Get(), &rtvGr, h); h.ptr += taaRtvDescSize_;
        device_->CreateRenderTargetView(godrayTex_[2].Get(), &rtvGr, h);
    }

    // Imgui debug previews — reserve 2 slots in imguiSrvHeap_ for godray
    // mark + blur. Write R16F SRVs there so ImGui::Image can sample them.
    if (imguiGodrayMarkGpu_.ptr == 0)
    {
        ImGuiSrvAlloc(&imguiGodrayMarkCpu_, &imguiGodrayMarkGpu_);
        ImGuiSrvAlloc(&imguiGodrayBlurCpu_, &imguiGodrayBlurGpu_);
    }
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvR16{};
        srvR16.Format = DXGI_FORMAT_R16_FLOAT;
        srvR16.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvR16.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvR16.Texture2D.MipLevels = 1;
        device_->CreateShaderResourceView(godrayTex_[0].Get(), &srvR16, imguiGodrayMarkCpu_);
        device_->CreateShaderResourceView(godrayTex_[1].Get(), &srvR16, imguiGodrayBlurCpu_);
    }

    taaHistValid_[0] = false;
    taaHistValid_[1] = false;
    taaFrame_ = 0;
    visTexW_ = w;
    visTexH_ = h;
    return true;
}

void Renderer::DrawLwScene(const Camera& cam, const DrawSceneParams& args)
{
    if (!lwHasWorld_ || !m4Pass1Pso_ || !visDepthTex_) return;
    MICROPROFILE_SCOPEI("CPU", "DrawLwScene", 0xff60c060);
    MICROPROFILE_SCOPEGPUI("DrawLwScene", 0xff60c060);

    PollShaderHotReload();

    // ---- Frustum + LOD heuristic ----
    hlslpp::float4x4 view = cam.view();
    hlslpp::float4x4 proj = cam.proj((float)width_ / (float)std::max(1u, height_));
    hlslpp::float4x4 vp   = hlslpp::mul(view, proj);
    float M[16];
    hlslpp::store(M, vp);
    float planes[6][4];
    ExtractFrustumPlanes(M, planes);

    auto cullAabb = [&](float mnx, float mny, float mnz, float mxx, float mxy, float mxz) -> bool
    {
        // Skip far plane (5): reverse-Z infinite-far is ill-defined.
        for (int pi = 0; pi < 5; ++pi)
        {
            float a = planes[pi][0], b = planes[pi][1], c = planes[pi][2], d = planes[pi][3];
            float px = a >= 0 ? mxx : mnx;
            float py = b >= 0 ? mxy : mny;
            float pz = c >= 0 ? mxz : mnz;
            if (a * px + b * py + c * pz + d < 0.0f) return true;
        }
        return false;
    };

    float camP[3];
    hlslpp::store(camP, cam.position);
    const float fovRad   = cam.fovDeg * 3.14159265358979f / 180.0f;
    const float focalPx  = (float)height_ / (2.0f * tanf(fovRad * 0.5f));
    const float lodScaleUi = std::max(args.pointLodScale, 0.01f);
    const float thresh   = 1.0f / lodScaleUi;

    auto desiredLodForDist = [&](float dist) -> int
    {
        if (dist < 1.0f) return 0;
        for (int L = 0; L < lw::kLodCount; ++L)
        {
            float ppv = focalPx * (float)(1u << L) / dist;
            if (ppv >= thresh) return L;
        }
        return lw::kLodCount - 1;
    };
    auto nearAabbDist = [&](float mnx, float mny, float mnz, float mxx, float mxy, float mxz) -> float
    {
        float dx = (camP[0] < mnx) ? (mnx - camP[0]) : (camP[0] > mxx) ? (camP[0] - mxx) : 0.0f;
        float dy = (camP[1] < mny) ? (mny - camP[1]) : (camP[1] > mxy) ? (camP[1] - mxy) : 0.0f;
        float dz = (camP[2] < mnz) ? (mnz - camP[2]) : (camP[2] > mxz) ? (camP[2] - mxz) : 0.0f;
        return sqrtf(dx*dx + dy*dy + dz*dz);
    };

    // ---- Per-LOD draw lists. Walker = port of CSTiles per-cluster recursion.
    struct DrawItem { uint32_t slot, blockFirst, blockCount; };
    std::vector<DrawItem> drawList[lw::kLodCount];

    // Recursive lambda via Y-combinator pattern — avoids std::function's
    // type-erased indirect call (was ~0.4ms CPU per frame on busy LOD0).
    auto visitCluster = [&](auto& self, int L, uint32_t chunkSlot, int clSlot) -> void
    {
        const lw::LODWorld& lwL = lwWorld_.lods[L];
        if (chunkSlot >= lwL.chunks.size()) return;
        const lw::RuntimeChunk& rc = lwL.chunks[chunkSlot];
        const lw::DiskCluster&  cl = rc.clusters[clSlot];
        if (cl.numPoints == 0) return;

        int cz_g = clSlot / (lw::kClustersX * lw::kClustersY);
        int cy_g = (clSlot / lw::kClustersX) % lw::kClustersY;
        int cx_g = clSlot % lw::kClustersX;
        const float lodScaleF = (float)lwL.lodScale;
        uint8_t bnds[6];
        lw::UnpackClusterBounds(cl.bounds, bnds);
        float mnx = (float)rc.worldOriginX + ((float)(cx_g * lw::kClusterVoxX + bnds[0])) * lodScaleF;
        float mny = (float)rc.worldOriginY + ((float)(cy_g * lw::kClusterVoxY + bnds[1])) * lodScaleF;
        float mnz = (float)rc.worldOriginZ + ((float)(cz_g * lw::kClusterVoxZ + bnds[2])) * lodScaleF;
        float mxx = (float)rc.worldOriginX + ((float)(cx_g * lw::kClusterVoxX + bnds[3] + 1)) * lodScaleF;
        float mxy = (float)rc.worldOriginY + ((float)(cy_g * lw::kClusterVoxY + bnds[4] + 1)) * lodScaleF;
        float mxz = (float)rc.worldOriginZ + ((float)(cz_g * lw::kClusterVoxZ + bnds[5] + 1)) * lodScaleF;
        if (cullAabb(mnx, mny, mnz, mxx, mxy, mxz)) return;

        float distNearC = nearAabbDist(mnx, mny, mnz, mxx, mxy, mxz);
        int   desNearC  = desiredLodForDist(distNearC);

        int oct = (cx_g >> 2) | ((cy_g & 1) << 1) | ((cz_g >> 2) << 2);
        uint32_t childChunkId = rc.childId[oct];
        bool childLoaded = (L > 0)
                        && (childChunkId != lw::kNoChild)
                        && (childChunkId < lwWorld_.lods[L-1].chunks.size())
                        && (lwWorld_.lods[L-1].chunks[childChunkId].blockCount > 0);

        if (desNearC >= L || !childLoaded || L == 0)
        {
            drawList[L].push_back({chunkSlot,
                                   rc.clusterBlockFirst[clSlot],
                                   rc.clusterBlockCount[clSlot]});
            return;
        }

        int lcx = cx_g & 3;
        int lcz = cz_g & 3;
        for (int dx = 0; dx < 2; ++dx)
            for (int dy = 0; dy < 2; ++dy)
                for (int dz = 0; dz < 2; ++dz)
                {
                    int childCx = 2*lcx + dx;
                    int childCy = dy;
                    int childCz = 2*lcz + dz;
                    int childSlot = childCz * (lw::kClustersX * lw::kClustersY)
                                  + childCy * lw::kClustersX + childCx;
                    self(self, L-1, childChunkId, childSlot);
                }
    };

    const int topL = lw::kLodCount - 1;
    for (uint32_t i = 0; i < (uint32_t)lwWorld_.lods[topL].chunks.size(); ++i)
    {
        const lw::LODWorld& lwT = lwWorld_.lods[topL];
        float mnX = lwT.cull.minX[i], mnY = lwT.cull.minY[i], mnZ = lwT.cull.minZ[i];
        float mxX = lwT.cull.maxX[i], mxY = lwT.cull.maxY[i], mxZ = lwT.cull.maxZ[i];
        if (cullAabb(mnX, mnY, mnZ, mxX, mxY, mxZ)) continue;
        const lw::RuntimeChunk& rc = lwT.chunks[i];
        if (rc.blockCount == 0) continue;
        for (int c = 0; c < lw::kClustersPerChunk; ++c)
        {
            if (rc.clusters[c].numPoints == 0) continue;
            visitCluster(visitCluster, topL, i, c);
        }
    }

    bool anyDraw = false;
    for (int L = 0; L < lw::kLodCount; ++L)
        if (!drawList[L].empty()) { anyDraw = true; break; }
    if (!anyDraw) return;

    // ---- Per-LOD sort + coalesce contiguous items in the same chunk. ----
    struct WI { uint32_t slot, baseGlobal, count, first; };
    std::vector<WI> perLodWl[lw::kLodCount];
    uint32_t perLodTotal[lw::kLodCount] = {};
    for (int L = 0; L < lw::kLodCount; ++L)
    {
        auto& dl = drawList[L];
        if (dl.empty()) continue;
        std::sort(dl.begin(), dl.end(),
                  [](const DrawItem& a, const DrawItem& b) {
                      if (a.slot != b.slot) return a.slot < b.slot;
                      return a.blockFirst < b.blockFirst;
                  });
        size_t w = 0;
        for (size_t r = 0; r < dl.size(); ++r) {
            if (w > 0 && dl[w-1].slot == dl[r].slot &&
                dl[w-1].blockFirst + dl[w-1].blockCount == dl[r].blockFirst)
                dl[w-1].blockCount += dl[r].blockCount;
            else
                dl[w++] = dl[r];
        }
        dl.resize(w);

        const lw::LODWorld& lwL = lwWorld_.lods[L];
        auto& wl = perLodWl[L];
        wl.reserve(dl.size());
        uint32_t cumul = 0;
        for (const DrawItem& it : dl)
        {
            if (it.blockCount == 0) continue;
            const lw::RuntimeChunk& rc = lwL.chunks[it.slot];
            wl.push_back({ rc.slotIdx, rc.blockBase + it.blockFirst, it.blockCount, cumul });
            cumul += it.blockCount;
        }
        perLodTotal[L] = cumul;
    }

    // ---- CB layouts (must match m4_lw.hlsl) ----
    // CSTiles cbPerFrame layout, 512 bytes, byte-for-byte.
    struct CBFrame {
        float viewProj[16];          //   0
        float camPos[3];             //  64
        float mode;                  //  76
        float lightDir[3];           //  80
        float ambient;               //  92
        float pointNormal[3];        //  96
        float _pad0;                 // 108
        float invViewProj[16];       // 112
        float screenSize[2];         // 176
        float _pad1[2];              // 184
        float camRight[3];           // 192
        float _pad3;                 // 204
        float camUp[3];              // 208
        float _pad4;                 // 220
        float camForward[3];         // 224
        float tanHalfFovY;           // 236
        float fogColor[3];           // 240
        float fogDensity;            // 252
        float heightFogDensity;      // 256
        float heightFogFalloff;      // 260
        float heightFogStart;        // 264
        float _padHF;                // 268
        float sceneOrigin[3];        // 272
        float nearZ;                 // 284
        float sceneSpan[3];          // 288
        float _pad6;                 // 300
        float prevViewProj[16];      // 304
        float jitter[2];             // 368
        float _pad7[2];              // 376
        float sunViewProj[16];       // 384
        float shadowBias;            // 448
        float shadowMapSize;         // 452
        float shadowEnable;          // 456
        float sunIntensity;          // 460
        float exposure;              // 464
        float roughness;             // 468
        float colorizeClusters;      // 472
        float gridSize;              // 476
        float invScreenSize[2];      // 480
        float aspect;                // 488
        float invAspect;             // 492
        float aspectTanFov;          // 496
        float _padPC[3];             // 500..512
    };
    static_assert(sizeof(CBFrame) == 512, "CBFrame must match CSTiles cbPerFrame layout");
    struct CBLwCs  { uint32_t vwSize[2]; uint32_t pointCount; uint32_t numItems;
                     uint32_t lodIdx; int32_t splatRadius; uint32_t _pad[2]; };

    // Halton(2/3) jitter, applied in NDC (so worldPos * jitteredVP shifts
    // sub-pixel each frame). Reset history when camera moved noticeably.
    auto halton = [](uint32_t i, uint32_t base) {
        float f = 1.0f, r = 0.0f;
        while (i > 0) { f /= (float)base; r += f * (float)(i % base); i /= base; }
        return r;
    };
    if (!args.taa) taaFrame_ = 0;

    CBFrame cbf{};
    hlslpp::store(cbf.viewProj, vp);
    for (int i = 0; i < 16; ++i) cbf.prevViewProj[i] = taaPrevVP_[i];
    cbf.mode = (float)(int)args.mode;
    cbf.camPos[0] = camP[0];
    cbf.camPos[1] = camP[1];
    cbf.camPos[2] = camP[2];
    cbf.ambient = 0.35f;
    {
        float sx = args.sunDir[0], sy = args.sunDir[1], sz = args.sunDir[2];
        float il = 1.0f / std::max(1e-4f, sqrtf(sx*sx + sy*sy + sz*sz));
        cbf.lightDir[0] = sx * il;
        cbf.lightDir[1] = sy * il;
        cbf.lightDir[2] = sz * il;
    }
    cbf.fogColor[0] = args.fogColor[0];
    cbf.fogColor[1] = args.fogColor[1];
    cbf.fogColor[2] = args.fogColor[2];
    cbf.fogDensity = args.fogDensity;
    cbf.heightFogDensity = args.heightFogDensity;
    cbf.heightFogFalloff = args.heightFogFalloff;
    cbf.heightFogStart = args.heightFogStart;
    cbf.nearZ = std::max(cam.nearZ, 1e-3f);
    cbf.sunIntensity = std::max(args.sunIntensity, 0.0f);
    cbf.exposure = std::max(args.exposure, 0.001f);
    cbf.roughness = args.roughness;
    cbf.gridSize = (float)std::max(1, args.gridSize);
    cbf.shadowEnable = 0.0f;
    cbf.shadowBias = args.shadowBias;
    cbf.shadowMapSize = 0.0f;
    if (args.taa)
    {
        uint32_t k = (taaFrame_ % 16) + 1;
        cbf.jitter[0] = (halton(k, 2) - 0.5f) * 2.0f / (float)width_;
        cbf.jitter[1] = (halton(k, 3) - 0.5f) * 2.0f / (float)height_;
    }
    {
        hlslpp::float3 fwd = cam.forward();
        hlslpp::float3 right = cam.right();
        hlslpp::float3 up = hlslpp::cross(fwd, right);
        hlslpp::store(cbf.camRight, right);
        hlslpp::store(cbf.camUp, up);
        hlslpp::store(cbf.camForward, fwd);
        cbf.tanHalfFovY = tanf(cam.fovDeg * 3.14159265358979f / 180.0f * 0.5f);
        cbf.aspect = (float)width_ / (float)std::max(1u, height_);
        cbf.invAspect = 1.0f / cbf.aspect;
        cbf.aspectTanFov = cbf.aspect * cbf.tanHalfFovY;
        cbf.screenSize[0] = (float)width_;
        cbf.screenSize[1] = (float)height_;
        cbf.invScreenSize[0] = 1.0f / (float)std::max(1u, width_);
        cbf.invScreenSize[1] = 1.0f / (float)std::max(1u, height_);
    }
    auto cbfAlloc = graphicsMemory_->AllocateConstant(cbf);

    // Per-LOD CB + worklist allocations.
    DirectX::GraphicsResource cbcsAlloc[lw::kLodCount];
    DirectX::GraphicsResource wlAlloc[lw::kLodCount];
    for (int L = 0; L < lw::kLodCount; ++L)
    {
        if (perLodWl[L].empty()) continue;
        CBLwCs cbcs{};
        cbcs.vwSize[0] = visTexW_;
        cbcs.vwSize[1] = visTexH_;
        cbcs.pointCount = perLodTotal[L];
        cbcs.numItems   = (uint32_t)perLodWl[L].size();
        cbcs.lodIdx     = (uint32_t)L;
        // Per-LOD splat: coarser LOD voxels render at fewer screen pixels
        // (well, more), but our LOD selection already trades screen-pixels-per-
        // voxel for LOD index. So one global radius is fine for M4c.
        cbcs.splatRadius = std::max(0, args.splatRadius);
        cbcsAlloc[L] = graphicsMemory_->AllocateConstant(cbcs);
        wlAlloc[L]   = graphicsMemory_->Allocate(perLodWl[L].size() * sizeof(WI));
        std::memcpy(wlAlloc[L].Memory(), perLodWl[L].data(), perLodWl[L].size() * sizeof(WI));
    }

    // ---- Bind heap + descriptor handle helpers ----
    ID3D12DescriptorHeap* heaps[] = { m4TexHeap_.Get() };
    cmdList_->SetDescriptorHeaps(1, heaps);
    auto gpu = [&](UINT slot) {
        D3D12_GPU_DESCRIPTOR_HANDLE h = m4TexHeap_->GetGPUDescriptorHandleForHeapStart();
        h.ptr += UINT64(slot) * m4TexDescSize_;
        return h;
    };
    auto cpuClr = [&](UINT slot) {
        D3D12_CPU_DESCRIPTOR_HANDLE h = m4TexClearHeap_->GetCPUDescriptorHandleForHeapStart();
        h.ptr += SIZE_T(slot) * m4TexDescSize_;
        return h;
    };

    // ---- Clear vis textures ----
    {
        MICROPROFILE_SCOPEGPUI("LW/ClearVis", 0xff404060);
        UINT clearMax[4]  = { 0xFFFFFFFFu, 0, 0, 0 };
        UINT clearZero[4] = { 0, 0, 0, 0 };
        cmdList_->ClearUnorderedAccessViewUint(gpu(0), cpuClr(0), visDepthTex_.Get(), clearMax,  0, nullptr);
        cmdList_->ClearUnorderedAccessViewUint(gpu(1), cpuClr(1), visColorTex_.Get(), clearZero, 0, nullptr);
    }

    auto uavBarrierDepth = [&](){
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        b.UAV.pResource = visDepthTex_.Get();
        cmdList_->ResourceBarrier(1, &b);
    };
    auto uavBarrierColor = [&](){
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        b.UAV.pResource = visColorTex_.Get();
        cmdList_->ResourceBarrier(1, &b);
    };

    // ---- Pass1 across LODs ----
    {
        MICROPROFILE_SCOPEGPUI("LW/Pass1Depth", 0xffe0a0ff);
        cmdList_->SetComputeRootSignature(m4Pass1RootSig_.Get());
        cmdList_->SetPipelineState(m4Pass1Pso_.Get());
        cmdList_->SetComputeRootConstantBufferView(0, cbfAlloc.GpuAddress());
        cmdList_->SetComputeRootDescriptorTable(5, gpu(0)); // u0 = depthUav
        bool firstPass1 = true;
        for (int L = 0; L < lw::kLodCount; ++L)
        {
            if (perLodWl[L].empty()) continue;
            if (!firstPass1) uavBarrierDepth();
            firstPass1 = false;
            const LwGpu& g = lwGpu_[L];
            cmdList_->SetComputeRootConstantBufferView(1, cbcsAlloc[L].GpuAddress());
            cmdList_->SetComputeRootShaderResourceView(2, g.chunkInfoSb->GetGPUVirtualAddress());
            cmdList_->SetComputeRootShaderResourceView(3, g.blockPosSb->GetGPUVirtualAddress());
            cmdList_->SetComputeRootShaderResourceView(4, wlAlloc[L].GpuAddress());
            UINT groups = (perLodTotal[L] + 63) / 64;
            cmdList_->Dispatch(groups, 1, 1);
        }
    }

    // Transition depth UAV → SRV for pass2 input.
    {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = visDepthTex_.Get();
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        b.Transition.StateAfter  = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmdList_->ResourceBarrier(1, &b);
    }

    // ---- Pass2 across LODs ----
    {
        MICROPROFILE_SCOPEGPUI("LW/Pass2Color", 0xffe080ff);
        cmdList_->SetComputeRootSignature(m4Pass2RootSig_.Get());
        cmdList_->SetPipelineState(m4Pass2Pso_.Get());
        cmdList_->SetComputeRootConstantBufferView(0, cbfAlloc.GpuAddress());
        cmdList_->SetComputeRootDescriptorTable(7, gpu(2)); // t5 = depthSrv
        cmdList_->SetComputeRootDescriptorTable(8, gpu(1)); // u0 = colorUav
        bool firstPass2 = true;
        for (int L = 0; L < lw::kLodCount; ++L)
        {
            if (perLodWl[L].empty()) continue;
            if (!firstPass2) uavBarrierColor();
            firstPass2 = false;
            const LwGpu& g = lwGpu_[L];
            cmdList_->SetComputeRootConstantBufferView(1, cbcsAlloc[L].GpuAddress());
            cmdList_->SetComputeRootShaderResourceView(2, g.chunkInfoSb->GetGPUVirtualAddress());
            cmdList_->SetComputeRootShaderResourceView(3, g.paletteSb->GetGPUVirtualAddress());
            cmdList_->SetComputeRootShaderResourceView(4, g.blockPosSb->GetGPUVirtualAddress());
            cmdList_->SetComputeRootShaderResourceView(5, g.blockColSb->GetGPUVirtualAddress());
            cmdList_->SetComputeRootShaderResourceView(6, wlAlloc[L].GpuAddress());
            if (g.blockVisSb)
                cmdList_->SetComputeRootShaderResourceView(9, g.blockVisSb->GetGPUVirtualAddress());
            else
                cmdList_->SetComputeRootShaderResourceView(9, g.blockPosSb->GetGPUVirtualAddress()); // fallback (mask read as garbage; will pick face)
            UINT groups = (perLodTotal[L] + 63) / 64;
            cmdList_->Dispatch(groups, 1, 1);
        }
    }

    // Transition visColor UAV → NON_PIXEL_SRV for dilate input. visDepth stays
    // NON_PIXEL_SRV (dilate also reads it).
    {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = visColorTex_.Get();
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        b.Transition.StateAfter  = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmdList_->ResourceBarrier(1, &b);
    }

    // ---- Dilate compute pass: visColor + visDepth → visColor2 ----
    {
        MICROPROFILE_SCOPEGPUI("LW/Dilate", 0xffd0a0c0);
        cmdList_->SetComputeRootSignature(m4DilateRootSig_.Get());
        cmdList_->SetPipelineState(m4DilatePso_.Get());
        D3D12_GPU_VIRTUAL_ADDRESS csCb = 0;
        for (int L = 0; L < lw::kLodCount; ++L)
            if (!perLodWl[L].empty()) { csCb = cbcsAlloc[L].GpuAddress(); break; }
        cmdList_->SetComputeRootConstantBufferView(0, cbfAlloc.GpuAddress()); // b0 cam basis
        cmdList_->SetComputeRootConstantBufferView(1, csCb);                  // b1 vwSize/radius
        cmdList_->SetComputeRootDescriptorTable(2, gpu(2));                   // t0..t1
        cmdList_->SetComputeRootDescriptorTable(3, gpu(6));                   // u0=visColor2, u1=visDepth2
        UINT gx = (visTexW_ + 7) / 8;
        UINT gy = (visTexH_ + 7) / 8;
        cmdList_->Dispatch(gx, gy, 1);
    }

    // Post-dilate: visDepth back to UAV (next frame's pass1 needs it); dilated
    // outputs UAV → PIXEL_SRV for resolve.
    {
        D3D12_RESOURCE_BARRIER b[3]{};
        b[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b[0].Transition.pResource = visDepthTex_.Get();
        b[0].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        b[0].Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        b[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b[1].Transition.pResource = visColor2Tex_.Get();
        b[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        b[1].Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        b[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b[2].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b[2].Transition.pResource = visDepth2Tex_.Get();
        b[2].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        b[2].Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        b[2].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmdList_->ResourceBarrier(3, b);
    }

    // ---- Resolve PS — fullscreen blit to backbuffer (already bound by BeginFrame) ----
    const uint32_t currIdx = taaCurrIdx_;
    const uint32_t prevIdx = currIdx ^ 1;

    // cbGodray — CSTiles layout. Filled from camera + sun direction.
    struct CBGodray {
        float sunScreenNdc[2];
        float sunOnScreen;
        float godrayEmaAlpha;
        float godrayHalfScreenUV[2];
        float sunFacing;
        float _padG1;
        float sunScreenUV[2];
        float godrayStridePx;
        float _padG2;
        float godrayTint[3];
        float godrayStrength;
    };
    CBGodray cbg{};
    {
        hlslpp::float3 fwdH = cam.forward();
        float fwd[3]; hlslpp::store(fwd, fwdH);
        float sxN = cbf.lightDir[0], syN = cbf.lightDir[1], szN = cbf.lightDir[2];
        cbg.sunFacing = fwd[0]*sxN + fwd[1]*syN + fwd[2]*szN;
        // Project sun direction (treated as direction at infinity) to NDC by
        // building world = camPos + sunDir * far, applying viewProj.
        float sunW[3] = { camP[0] + sxN * 10000.0f,
                          camP[1] + syN * 10000.0f,
                          camP[2] + szN * 10000.0f };
        const float* vp = cbf.viewProj;
        float clip[4];
        for (int j = 0; j < 4; ++j)
            clip[j] = sunW[0]*vp[0*4+j] + sunW[1]*vp[1*4+j] + sunW[2]*vp[2*4+j] + vp[3*4+j];
        bool inFront = (clip[3] > 0.0f) && (cbg.sunFacing > 0.0f);
        float ndcX = 0, ndcY = 0;
        if (inFront) {
            ndcX = clip[0] / clip[3];
            ndcY = clip[1] / clip[3];
        }
        cbg.sunScreenNdc[0] = ndcX;
        cbg.sunScreenNdc[1] = ndcY;
        bool onScreen = inFront && (ndcX >= -1 && ndcX <= 1 && ndcY >= -1 && ndcY <= 1);
        cbg.sunOnScreen = onScreen ? 1.0f : 0.0f;
        cbg.sunScreenUV[0] = ndcX * 0.5f + 0.5f;
        cbg.sunScreenUV[1] = 0.5f - ndcY * 0.5f;
        // Half-extent of godray local space in screen UV. Sized to cover
        // args.godrayAngleDeg of arc around sun, matching CSTiles formula.
        const float kDeg2Rad = 3.14159265358979f / 180.0f;
        float halfPx = (float)height_ * 0.5f
                     * tanf(args.godrayAngleDeg * 0.5f * kDeg2Rad)
                     / tanf(cam.fovDeg * 0.5f * kDeg2Rad);
        cbg.godrayHalfScreenUV[0] = halfPx / (float)std::max(1u, width_);
        cbg.godrayHalfScreenUV[1] = halfPx / (float)std::max(1u, height_);
        cbg.godrayEmaAlpha = std::clamp(args.godrayEmaAlpha, 0.02f, 1.0f);
        cbg.godrayStridePx = 0.0f; // separable path unused
        cbg.godrayTint[0]  = args.godrayTint[0];
        cbg.godrayTint[1]  = args.godrayTint[1];
        cbg.godrayTint[2]  = args.godrayTint[2];
        cbg.godrayStrength = std::max(args.godrayStrength, 0.0f);
    }
    auto cbgAlloc = graphicsMemory_->AllocateConstant(cbg);

    // ---- Godray Mark + Blur (only when strength > 0) ----
    const uint32_t grCurr = godrayCurrIdx_;
    const uint32_t grPrev = grCurr ^ 1;
    if (args.godrayStrength > 0.0f)
    {
        MICROPROFILE_SCOPEGPUI("LW/GodrayMark+Blur", 0xffd0a060);

        D3D12_VIEWPORT vp64{ 0, 0, 64.0f, 64.0f, 0.0f, 1.0f };
        D3D12_RECT     sc64{ 0, 0, 64, 64 };
        cmdList_->RSSetViewports(1, &vp64);
        cmdList_->RSSetScissorRects(1, &sc64);

        // Mark: write godrayTex[0]. Reads visDepth2 (gPostDepth at t6).
        {
            D3D12_RESOURCE_BARRIER b{};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource = godrayTex_[0].Get();
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            b.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
            b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            cmdList_->ResourceBarrier(1, &b);
            D3D12_CPU_DESCRIPTOR_HANDLE rtv = taaRtvHeap_->GetCPUDescriptorHandleForHeapStart();
            rtv.ptr += SIZE_T(3) * taaRtvDescSize_;
            cmdList_->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
            cmdList_->SetGraphicsRootSignature(m4GodrayMarkRootSig_.Get());
            cmdList_->SetPipelineState(m4GodrayMarkPso_.Get());
            cmdList_->SetGraphicsRootConstantBufferView(0, cbfAlloc.GpuAddress());
            cmdList_->SetGraphicsRootConstantBufferView(1, cbgAlloc.GpuAddress());
            cmdList_->SetGraphicsRootDescriptorTable(2, gpu(8)); // t6 = visDepth2 SRV
            cmdList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmdList_->DrawInstanced(3, 1, 0, 0);
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            b.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            cmdList_->ResourceBarrier(1, &b);
        }

        // Blur: write godrayTex[1+grCurr] from mark (godrayTex[0]) + prev blur.
        {
            D3D12_RESOURCE_BARRIER b{};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource = godrayTex_[1 + grCurr].Get();
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            b.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
            b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            cmdList_->ResourceBarrier(1, &b);
            D3D12_CPU_DESCRIPTOR_HANDLE rtv = taaRtvHeap_->GetCPUDescriptorHandleForHeapStart();
            rtv.ptr += SIZE_T(4 + grCurr) * taaRtvDescSize_;
            cmdList_->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
            cmdList_->SetGraphicsRootSignature(m4GodrayBlurRootSig_.Get());
            cmdList_->SetPipelineState(m4GodrayBlurPso_.Get());
            cmdList_->SetGraphicsRootConstantBufferView(0, cbgAlloc.GpuAddress());
            cmdList_->SetGraphicsRootDescriptorTable(1, gpu(11));           // t9 = godrayTex[0] mark
            cmdList_->SetGraphicsRootDescriptorTable(2, gpu(12 + grPrev));  // t10 = prev blur
            cmdList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmdList_->DrawInstanced(3, 1, 0, 0);
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            b.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            cmdList_->ResourceBarrier(1, &b);
        }

        // Restore full-res viewport for resolve/taa/post.
        D3D12_VIEWPORT vpFull{ 0, 0, (float)width_, (float)height_, 0.0f, 1.0f };
        D3D12_RECT     scFull{ 0, 0, (LONG)width_, (LONG)height_ };
        cmdList_->RSSetViewports(1, &vpFull);
        cmdList_->RSSetScissorRects(1, &scFull);
    }

    // ---- Resolve → taaScene RTV ----
    {
        MICROPROFILE_SCOPEGPUI("LW/Resolve", 0xffc080ff);
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = taaSceneTex_.Get();
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        b.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmdList_->ResourceBarrier(1, &b);
        D3D12_CPU_DESCRIPTOR_HANDLE rtv = taaRtvHeap_->GetCPUDescriptorHandleForHeapStart();
        rtv.ptr += SIZE_T(2) * taaRtvDescSize_; // slot 2 = taaScene
        cmdList_->OMSetRenderTargets(1, &rtv, FALSE, nullptr);

        cmdList_->SetGraphicsRootSignature(m4ResolveRootSig_.Get());
        cmdList_->SetPipelineState(m4ResolvePso_.Get());
        cmdList_->SetGraphicsRootConstantBufferView(0, cbfAlloc.GpuAddress());
        cmdList_->SetGraphicsRootDescriptorTable(1, gpu(8));           // t0=visDepth2, t1=visColor2
        cmdList_->SetGraphicsRootDescriptorTable(2, gpu(4 + prevIdx)); // t2=taaHist[prev]  (unused by resolve now)
        cmdList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmdList_->DrawInstanced(3, 1, 0, 0);

        b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        b.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        cmdList_->ResourceBarrier(1, &b);
    }

    // ---- TAA: taaScene + taaHist[prev] + taaDepth → taaHist[curr] ----
    {
        MICROPROFILE_SCOPEGPUI("LW/TAA", 0xff80a0ff);
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = taaHistTex_[currIdx].Get();
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        b.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmdList_->ResourceBarrier(1, &b);
        D3D12_CPU_DESCRIPTOR_HANDLE rtv = taaRtvHeap_->GetCPUDescriptorHandleForHeapStart();
        rtv.ptr += SIZE_T(currIdx) * taaRtvDescSize_;
        cmdList_->OMSetRenderTargets(1, &rtv, FALSE, nullptr);

        cmdList_->SetGraphicsRootSignature(m4TaaRootSig_.Get());
        cmdList_->SetPipelineState(m4TaaPso_.Get());
        cmdList_->SetGraphicsRootConstantBufferView(0, cbfAlloc.GpuAddress());
        cmdList_->SetGraphicsRootDescriptorTable(1, gpu(10));          // t4 = taaScene
        cmdList_->SetGraphicsRootDescriptorTable(2, gpu(4 + prevIdx)); // t5 = taaHist[prev]
        cmdList_->SetGraphicsRootDescriptorTable(3, gpu(8));           // t6 = taaDepth (= visDepth2)
        cmdList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmdList_->DrawInstanced(3, 1, 0, 0);

        b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        b.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        cmdList_->ResourceBarrier(1, &b);
    }

    // ---- Post: taaHist[curr] → backbuffer (sky + tonemap + godray placeholder) ----
    {
        MICROPROFILE_SCOPEGPUI("LW/Post", 0xff80c0ff);
        D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtvHeap_->GetCPUDescriptorHandleForHeapStart();
        rtv.ptr += SIZE_T(frameIndex_) * rtvDescSize_;
        D3D12_CPU_DESCRIPTOR_HANDLE dsv = dsvHeap_->GetCPUDescriptorHandleForHeapStart();
        cmdList_->OMSetRenderTargets(1, &rtv, FALSE, &dsv);

        cmdList_->SetGraphicsRootSignature(m4PostRootSig_.Get());
        cmdList_->SetPipelineState(m4PostPso_.Get());
        cmdList_->SetGraphicsRootConstantBufferView(0, cbfAlloc.GpuAddress());
        cmdList_->SetGraphicsRootConstantBufferView(1, cbgAlloc.GpuAddress()); // cbGodray zeroed
        cmdList_->SetGraphicsRootDescriptorTable(2, gpu(8));           // t6 = postDepth = taaDepth
        cmdList_->SetGraphicsRootDescriptorTable(3, gpu(4 + currIdx)); // t7 = postIn = taaHist[curr]
        cmdList_->SetGraphicsRootDescriptorTable(4, gpu(12 + grCurr)); // t9 = current frame's blurred godray
        cmdList_->SetGraphicsRootDescriptorTable(5, gpu(11));          // t10 = mark (unused at post; needs valid bind)
        cmdList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmdList_->DrawInstanced(3, 1, 0, 0);
    }

    taaHistValid_[currIdx] = true;
    taaCurrIdx_ = prevIdx;
    ++taaFrame_;
    godrayCurrIdx_ ^= 1;
    // Stash this frame's VP for next frame's reprojection.
    hlslpp::store(taaPrevVP_, vp);

    // Restore vis textures to UAV state for next frame.
    //  visDepth: already UAV (transitioned right after dilate).
    //  visColor: NON_PIXEL_SRV (last set as dilate input) → UAV.
    //  visColor2 / visDepth2: PIXEL_SRV (resolve read) → UAV.
    {
        D3D12_RESOURCE_BARRIER b[3]{};
        b[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b[0].Transition.pResource = visColorTex_.Get();
        b[0].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        b[0].Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        b[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b[1].Transition.pResource = visColor2Tex_.Get();
        b[1].Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        b[1].Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        b[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b[2].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b[2].Transition.pResource = visDepth2Tex_.Get();
        b[2].Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        b[2].Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        b[2].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmdList_->ResourceBarrier(3, b);
    }

    // Restore imgui heap so the upcoming imgui render binds against it.
    ID3D12DescriptorHeap* imguiHeap[] = { imguiSrvHeap_.Get() };
    cmdList_->SetDescriptorHeaps(1, imguiHeap);
}

Renderer::StreamLodInfo Renderer::GetStreamLodInfo(int L) const
{
    StreamLodInfo info{};
    if (L < 0 || L >= lw::kLodCount) return info;
    const LwGpu& g = lwGpu_[L];
    auto fill = [](StreamPoolInfo& p, const char* name,
                   ID3D12Resource* buf, uint32_t stride)
    {
        p.name = name;
        p.stride = stride;
        if (buf)
        {
            D3D12_RESOURCE_DESC rd = buf->GetDesc();
            p.bytes = rd.Width;
            p.numElements = stride ? (uint32_t)(rd.Width / stride) : 0;
        }
    };
    fill(info.pools[0], "chunkInfoSb", g.chunkInfoSb.Get(), (uint32_t)sizeof(lw::GpuChunkInfo));
    fill(info.pools[1], "paletteSb",   g.paletteSb.Get(),   (uint32_t)sizeof(uint32_t));
    fill(info.pools[2], "blockPosSb",  g.blockPosSb.Get(),  (uint32_t)sizeof(lw::BlockPos));
    fill(info.pools[3], "blockColSb",  g.blockColSb.Get(),  (uint32_t)sizeof(lw::BlockCol));
    fill(info.pools[4], "blockVisSb",  g.blockVisSb.Get(),  (uint32_t)sizeof(lw::BlockVis));
    return info;
}
