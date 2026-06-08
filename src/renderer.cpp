// DX12 Phase-1 (M1) renderer skeleton.
// Provides: device, swapchain, command queue/list, RTV/DSV heaps, fence,
// per-frame clear + present. ImGui font descriptor heap exposed.
// All voxel rendering / LW upload paths live in renderer.h as inline stubs
// returning safe defaults — they will be ported incrementally in M3+.
#define NOMINMAX
#include "renderer.h"

#include <DescriptorHeap.h>
#include <GraphicsMemory.h>
#include <DirectXHelpers.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
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
}

void Renderer::BeginFrame(float clear[4], bool skipClear, bool /*skipDsvClear*/)
{
    if (!device_) return;

    auto& alloc = cmdAlloc_[frameIndex_];
    ThrowIfFailed(alloc->Reset(), "alloc reset");
    ThrowIfFailed(cmdList_->Reset(alloc.Get(), nullptr), "list reset");

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

    if (!skipClear)
        cmdList_->ClearRenderTargetView(rtv, clear, 0, nullptr);
    cmdList_->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

    D3D12_VIEWPORT vp{ 0, 0, (float)width_, (float)height_, 0.0f, 1.0f };
    D3D12_RECT     sc{ 0, 0, (LONG)width_, (LONG)height_ };
    cmdList_->RSSetViewports(1, &vp);
    cmdList_->RSSetScissorRects(1, &sc);

    ID3D12DescriptorHeap* heaps[] = { imguiSrvHeap_.Get() };
    cmdList_->SetDescriptorHeaps(1, heaps);

    // M2 demo — IA-less triangle to validate DXC + rootsig + PSO path.
    if (m2Pso_)
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
