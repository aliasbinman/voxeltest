// DX12 Phase-1 (M1) renderer skeleton.
// Provides: device, swapchain, command queue/list, RTV/DSV heaps, fence,
// per-frame clear + present. ImGui font descriptor heap exposed.
// All voxel rendering / LW upload paths live in renderer.h as inline stubs
// returning safe defaults — they will be ported incrementally in M3+.
#ifndef NOMINMAX
  #define NOMINMAX
#endif
#include "renderer.h"
#if !defined(VOXELTEST_XBOX)
#include "microprofile.h"
#else
#define MICROPROFILE_SCOPEI(group, name, color) do{}while(0)
#define MICROPROFILE_SCOPEGPUI(name, color)     do{}while(0)
#endif
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
#if defined(VOXELTEST_XBOX)
    names.emplace_back("Xbox Series X|S (Scarlett)");
    return names;
#else
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
#endif
}

bool Renderer::Init(HWND hwnd, int adapterIdx)
{
    hwnd_ = hwnd;

#if defined(VOXELTEST_XBOX)
    (void)hwnd;
    width_  = 2560;
    height_ = 1440;
#else
    RECT rc{};
    GetClientRect(hwnd, &rc);
    width_  = (uint32_t)std::max<LONG>(1, rc.right  - rc.left);
    height_ = (uint32_t)std::max<LONG>(1, rc.bottom - rc.top);
#endif

    try
    {
        if (!CreateDeviceAndSwap(hwnd, adapterIdx)) return false;
        if (!CreateRenderTargets()) return false;

        ThrowIfFailed(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)),
                      "CreateFence");
        ThrowIfFailed(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&lwUploadFence_)),
                      "CreateFence(lwUpload)");
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

#if defined(VOXELTEST_XBOX)
bool Renderer::CreateDeviceAndSwap(HWND, int)
{
    // ---- Xbox Scarlett device + swap chain (PresentX flow) ----
    D3D12XBOX_CREATE_DEVICE_PARAMETERS params = {};
    params.Version = D3D12_SDK_VERSION;
#if defined(_DEBUG)
    // Debug: validation + PIX hookable.
    params.ProcessDebugFlags = D3D12_PROCESS_DEBUG_FLAG_DEBUG_LAYER_ENABLED
                             | D3D12XBOX_PROCESS_DEBUG_FLAG_INSTRUMENTED;
#elif defined(PROFILE)
    // Profile: PIX-capturable retail runtime, no validation.
    params.ProcessDebugFlags = D3D12XBOX_PROCESS_DEBUG_FLAG_INSTRUMENTED;
#else
    // Release: pure retail runtime, no PIX, no validation.
    params.ProcessDebugFlags = (D3D12XBOX_PROCESS_DEBUG_FLAGS)0;
#endif
    params.GraphicsCommandQueueRingSizeBytes = D3D12XBOX_DEFAULT_SIZE_BYTES;
    params.GraphicsScratchMemorySizeBytes    = D3D12XBOX_DEFAULT_SIZE_BYTES;
    params.ComputeScratchMemorySizeBytes     = D3D12XBOX_DEFAULT_SIZE_BYTES;
    if (FAILED(D3D12XboxCreateDevice(nullptr, &params, IID_PPV_ARGS(&device_))))
        return false;
    NameObject(device_.Get(), L"device");

    {
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(device_->CreateCommandQueue(&qd, IID_PPV_ARGS(&cmdQueue_)))) return false;
        NameObject(cmdQueue_.Get(), L"cmdQueue");
    }

    // Backbuffers as committed default-heap render-target textures. No DXGI swap.
    for (UINT i = 0; i < kFrameCount; ++i)
    {
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = width_; rd.Height = height_;
        rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.Format = BackBufferFormat();
        rd.SampleDesc.Count = 1;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_CLEAR_VALUE cv{}; cv.Format = BackBufferFormat();
        if (FAILED(device_->CreateCommittedResource(
                       &hp, D3D12_HEAP_FLAG_ALLOW_DISPLAY, &rd,
                       D3D12_RESOURCE_STATE_PRESENT, &cv,
                       IID_PPV_ARGS(&backBuffers_[i])))) return false;
    }

    // Frame interval registration. Path = device → DXGI device → adapter →
    // output → SetFrameIntervalX. WaitFrameEventX(ORIGIN) sources the per-frame
    // token in BeginFrame.
    {
        ComPtr<IDXGIDevice1> dxgiDevice;
        if (FAILED(device_.As(&dxgiDevice))) return false;
        ComPtr<IDXGIAdapter> dxgiAdapter;
        if (FAILED(dxgiDevice->GetAdapter(&dxgiAdapter))) return false;
        ComPtr<IDXGIOutput> dxgiOutput;
        if (FAILED(dxgiAdapter->EnumOutputs(0, &dxgiOutput))) return false;
        if (FAILED(device_->SetFrameIntervalX(dxgiOutput.Get(),
                                              D3D12XBOX_FRAME_INTERVAL_60_HZ,
                                              kFrameCount - 1u,
                                              D3D12XBOX_FRAME_INTERVAL_FLAG_NONE)))
            return false;
        if (FAILED(device_->ScheduleFrameEventX(D3D12XBOX_FRAME_EVENT_ORIGIN,
                                                 0, nullptr,
                                                 D3D12XBOX_SCHEDULE_FRAME_EVENT_FLAG_NONE)))
            return false;
    }

    for (UINT i = 0; i < kFrameCount; ++i)
    {
        if (FAILED(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                   IID_PPV_ARGS(&cmdAlloc_[i])))) return false;
    }
    frameIndex_ = 0;
    tearingSupported_ = false;
    return true;
}
#else
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
            // NOTE: GPU-based validation left OFF — the removal is DEVICE_HUNG
            // (TDR timeout), not a fault, and GBV's 10-100x slowdown would itself
            // trip TDR and confound the timing. Re-enable to hunt OOB/bad-state.
        }
        // DRED: capture auto-breadcrumbs + page-fault VA so a device removal
        // tells us exactly which op / address faulted.
        ComPtr<ID3D12DeviceRemovedExtendedDataSettings> dred;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dred))))
        {
            dred->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            dred->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            std::printf("[d3d12] DRED breadcrumbs + page-fault tracking ENABLED\n");
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
            // ERROR break OFF for now: let DEVICE_HUNG flow to BeginFrame's DRED
            // dump (which op/allocation faulted) instead of breaking immediately.
            iq->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, FALSE);
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
#endif // !VOXELTEST_XBOX

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
#if !defined(VOXELTEST_XBOX)
            ThrowIfFailed(swap_->GetBuffer(i, IID_PPV_ARGS(&backBuffers_[i])),
                          "GetBuffer");
#endif
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
        cv.DepthStencil.Depth = 0.0f; // reverse-Z: far plane = 0, depth func GREATER

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
#if defined(VOXELTEST_XBOX)
    // Xbox = fixed 2560x1440 backbuffer; no dynamic resize.
    (void)w; (void)h;
    return;
#else
    if (!device_ || !swap_) return;
    if (w == width_ && h == height_) return;
    if (w == 0 || h == 0) return;

    WaitForGpu();
    const UINT64 curFence = fenceValues_[frameIndex_];

    for (UINT i = 0; i < kFrameCount; ++i) backBuffers_[i].Reset();
    depthTex_.Reset();

    UINT flags = tearingSupported_ ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
    ThrowIfFailed(swap_->ResizeBuffers(kFrameCount, w, h, BackBufferFormat(), flags),
                  "ResizeBuffers");
    width_ = w;
    height_ = h;
    frameIndex_ = swap_->GetCurrentBackBufferIndex();
    // frameIndex_ jumps after ResizeBuffers; stale per-frame fence values would
    // signal the fence backwards and deadlock the next frame's wait.
    for (UINT i = 0; i < kFrameCount; ++i) fenceValues_[i] = curFence;
    CreateRenderTargets();
    if (m4TexHeap_) CreateVisTextures(w, h);
#endif
}

void Renderer::BeginFrame(float clear[4], bool skipClear, bool /*skipDsvClear*/)
{
    if (!device_) return;

    CollectLwRetired(); // free streamed buffers whose copy fence has signalled

    HRESULT removed = device_->GetDeviceRemovedReason();
    if (FAILED(removed))
    {
        static bool printed = false;
        if (!printed)
        {
            std::printf("[gpu] DEVICE REMOVED reason=0x%08X\n", (unsigned)removed);
#if defined(_DEBUG) && !defined(VOXELTEST_XBOX)
            // DRED: which op/address faulted.
            ComPtr<ID3D12DeviceRemovedExtendedData> dred;
            if (SUCCEEDED(device_.As(&dred)))
            {
                D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT bc{};
                if (SUCCEEDED(dred->GetAutoBreadcrumbsOutput(&bc)))
                {
                    std::printf("[dred] --- auto breadcrumbs (last GPU ops) ---\n");
                    const D3D12_AUTO_BREADCRUMB_NODE* n = bc.pHeadAutoBreadcrumbNode;
                    int guard = 0;
                    while (n && guard++ < 32)
                    {
                        UINT done = n->pLastBreadcrumbValue ? *n->pLastBreadcrumbValue : 0;
                        std::printf("[dred] cmdlist '%ls' op %u/%u\n",
                                    n->pCommandListDebugNameW ? n->pCommandListDebugNameW : L"?",
                                    done, n->BreadcrumbCount);
                        n = n->pNext;
                    }
                }
                D3D12_DRED_PAGE_FAULT_OUTPUT pf{};
                if (SUCCEEDED(dred->GetPageFaultAllocationOutput(&pf)))
                {
                    std::printf("[dred] PAGE FAULT VA = 0x%llx\n",
                                (unsigned long long)pf.PageFaultVA);
                    for (const D3D12_DRED_ALLOCATION_NODE* a = pf.pHeadExistingAllocationNode; a; a = a->pNext)
                        std::printf("[dred]   existing alloc: '%ls' type %d\n",
                                    a->ObjectNameW ? a->ObjectNameW : L"?", (int)a->AllocationType);
                    for (const D3D12_DRED_ALLOCATION_NODE* a = pf.pHeadRecentFreedAllocationNode; a; a = a->pNext)
                        std::printf("[dred]   RECENTLY FREED: '%ls' type %d\n",
                                    a->ObjectNameW ? a->ObjectNameW : L"?", (int)a->AllocationType);
                }
            }
#endif
            std::fflush(stdout);
            printed = true;
        }
        return;
    }

#if defined(_DEBUG) && !defined(VOXELTEST_XBOX)
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

#if defined(VOXELTEST_XBOX)
    // Block on frame origin so the frame token is valid for PresentX later.
    frameToken_ = D3D12XBOX_FRAME_PIPELINE_TOKEN_NULL;
    device_->WaitFrameEventX(D3D12XBOX_FRAME_EVENT_ORIGIN, INFINITE,
                             nullptr, D3D12XBOX_WAIT_FRAME_EVENT_FLAG_NONE,
                             &frameToken_);
#endif

    auto& alloc = cmdAlloc_[frameIndex_];
    ThrowIfFailed(alloc->Reset(), "alloc reset");
    ThrowIfFailed(cmdList_->Reset(alloc.Get(), nullptr), "list reset");

#if MICROPROFILE_ENABLED
    MicroProfileGpuSetContext(cmdList_.Get());
#endif

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
        cmdList_->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.0f, 0, 0, nullptr); // reverse-Z far
    }

    D3D12_VIEWPORT vp{ 0, 0, (float)width_, (float)height_, 0.0f, 1.0f };
    D3D12_RECT     sc{ 0, 0, (LONG)width_, (LONG)height_ };
    cmdList_->RSSetViewports(1, &vp);
    cmdList_->RSSetScissorRects(1, &sc);

    if (imguiSrvHeap_) {
        ID3D12DescriptorHeap* heaps[] = { imguiSrvHeap_.Get() };
        cmdList_->SetDescriptorHeaps(1, heaps);
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

#if defined(VOXELTEST_XBOX)
    (void)vsync;
    D3D12XBOX_PRESENT_PLANE_PARAMETERS plane{};
    plane.Token = frameToken_;
    plane.ResourceCount = 1;
    plane.ppResources = backBuffers_[frameIndex_].GetAddressOf();
    cmdQueue_->PresentX(1, &plane, nullptr);
#else
    UINT syncInterval = vsync ? 1 : 0;
    UINT presentFlags = (!vsync && tearingSupported_) ? DXGI_PRESENT_ALLOW_TEARING : 0;
    swap_->Present(syncInterval, presentFlags);
#endif

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
#if defined(VOXELTEST_XBOX)
    frameIndex_ = (frameIndex_ + 1) % kFrameCount;
#else
    frameIndex_ = swap_->GetCurrentBackBufferIndex();
#endif
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
    lwRetire_.clear(); // GPU idle — release any parked streamed resources
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
    cmdQueue_.Reset();
#if !defined(VOXELTEST_XBOX)
    swap_.Reset();
#endif
    fence_.Reset();
    device_.Reset();
#if !defined(VOXELTEST_XBOX)
    factory_.Reset();
#endif
}


void Renderer::ClearLwWorld()
{
    // Same as UploadLwLod: GPU may still hold references to these resources.
    bool anyLive = false;
    for (int L = 0; L < lw::kLodCount; ++L)
        if (lwGpu_[L].chunkInfoSb) { anyLive = true; break; }
    if (anyLive) WaitForGpu();
    lwRetire_.clear(); // GPU idle after WaitForGpu — parked resources safe to free
    for (int L = 0; L < lw::kLodCount; ++L)
    {
        lwGpu_[L] = LwGpu{};
        pendingUpload_[L] = PendingLwUpload{}; // drop any uncommitted prepared upload
    }
    // Bump-allocator reset: M3 doesn't free per-LOD slots, so a full ClearLwWorld
    // reclaims all of them at once. Sufficient for one-world-at-a-time workflow.
    lwSrvNextSlot_ = 0;
    lwHasWorld_ = false;
}

// Free streamed-buffer retirements whose copy fence has completed. Cheap poll;
// called each frame and before each new upload.
void Renderer::CollectLwRetired()
{
    for (size_t i = 0; i < lwRetire_.size();)
    {
        LwRetireBatch& b = lwRetire_[i];
        if (!b.fence || b.fence->GetCompletedValue() >= b.value)
        {
            lwRetire_[i] = std::move(lwRetire_.back());
            lwRetire_.pop_back();
        }
        else
        {
            ++i;
        }
    }
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

// LOADER-THREAD phase: build staging, create resources, memcpy into upload
// heaps, and record the copy+barrier command list. Touches neither cmdQueue_
// nor lwGpu_, so it runs concurrently with main-thread rendering. Result parked
// in pendingUpload_[L]; main thread later calls CommitLwLod(L).
bool Renderer::PrepareLwLod(const lw::World& w, int L)
{
    if (!device_) return false;
    if (L < 0 || L >= lw::kLodCount) return false;
    const lw::LODWorld& src = w.lods[L];
    PendingLwUpload& pu = pendingUpload_[L];
    pu = PendingLwUpload{};

    // CPU-side metadata the draw walk needs (chunks + cull). Copied here on the
    // loader thread (not the big block pools — those live on the GPU now).
    pu.meta.lodLevel = src.lodLevel;
    pu.meta.lodScale = src.lodScale;
    pu.meta.chunks   = src.chunks;
    pu.meta.cull     = src.cull;

    const uint32_t slotCount = (uint32_t)src.chunks.size();
    if (slotCount == 0) { pu.valid = true; pu.empty = true; return true; }
    if (slotCount > lw::kMaxResidentChunksPerLod)
    {
        std::fprintf(stderr, "[lw] LOD %d has %u chunks > max %u\n",
                     L, slotCount, lw::kMaxResidentChunksPerLod);
        return false;
    }

    if (FAILED(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                               IID_PPV_ARGS(&pu.alloc)))) return false;
    if (FAILED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                          pu.alloc.Get(), nullptr,
                                          IID_PPV_ARGS(&pu.cmd)))) return false;
    // CreateCommandList opens it recording.

    // Create default+upload committed resources, map+memcpy, record copy+barrier.
    auto record = [&](const void* data, size_t bytes,
                      Microsoft::WRL::ComPtr<ID3D12Resource>& outDefault) -> bool
    {
        if (bytes == 0) return true;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = bytes; rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES hpDef{}; hpDef.Type = D3D12_HEAP_TYPE_DEFAULT;
        if (FAILED(device_->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &rd,
                      D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                      IID_PPV_ARGS(&outDefault)))) return false;
        D3D12_HEAP_PROPERTIES hpUp{}; hpUp.Type = D3D12_HEAP_TYPE_UPLOAD;
        Microsoft::WRL::ComPtr<ID3D12Resource> up;
        if (FAILED(device_->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &rd,
                      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                      IID_PPV_ARGS(&up)))) return false;
        void* mapped = nullptr; D3D12_RANGE noRead{0, 0};
        if (FAILED(up->Map(0, &noRead, &mapped))) return false;
        std::memcpy(mapped, data, bytes);
        up->Unmap(0, nullptr);
        pu.cmd->CopyBufferRegion(outDefault.Get(), 0, up.Get(), 0, bytes);
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = outDefault.Get();
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        b.Transition.StateAfter  = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                                 | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        pu.cmd->ResourceBarrier(1, &b);
        pu.scratch.push_back(up);
        return true;
    };

    // ---- ChunkInfo ----
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
    if (!record(infos.data(), infos.size() * sizeof(lw::GpuChunkInfo), pu.gpu.chunkInfoSb)) return false;

    // ---- Palette atlas ----
    std::vector<uint32_t> atlas((size_t)slotCount * lw::kPaletteSize, 0u);
    for (uint32_t i = 0; i < slotCount; ++i)
    {
        const lw::RuntimeChunk& rc = src.chunks[i];
        const uint32_t n = std::min(rc.paletteCount, (uint32_t)lw::kPaletteSize);
        std::memcpy(atlas.data() + (size_t)i * lw::kPaletteSize,
                    rc.palette, n * sizeof(uint32_t));
    }
    if (!record(atlas.data(), atlas.size() * sizeof(uint32_t), pu.gpu.paletteSb)) return false;

    // ---- BlockPos / BlockCol / BlockVis (raw memcpy) ----
    const uint64_t posBytes = (uint64_t)src.blockPosPool.size() * sizeof(lw::BlockPos);
    const uint64_t colBytes = (uint64_t)src.blockColPool.size() * sizeof(lw::BlockCol);
    const uint64_t visBytes = (uint64_t)src.blockVisPool.size() * sizeof(lw::BlockVis);
    if (!record(src.blockPosPool.data(), posBytes, pu.gpu.blockPosSb)) return false;
    if (!record(src.blockColPool.data(), colBytes, pu.gpu.blockColSb)) return false;
    if (!record(src.blockVisPool.data(), visBytes, pu.gpu.blockVisSb)) return false;

    // ---- BlockAo: 3 B/voxel -> 1 uint/voxel (low 24 = 6 faces * 4-bit). Tight
    // pack, no per-voxel average loop (top byte unused; dilate indexes per face).
    std::vector<uint32_t> aoFlat;
    const uint64_t aoBytes = (uint64_t)src.blockAoPool.size() * 8u * sizeof(uint32_t);
    if (!src.blockAoPool.empty())
    {
        aoFlat.resize(src.blockAoPool.size() * 8u);
        for (size_t b = 0; b < src.blockAoPool.size(); ++b)
            for (int i = 0; i < 8; ++i)
            {
                const uint8_t* ao = src.blockAoPool[b].ao[i];
                aoFlat[b * 8 + i] = (uint32_t)ao[0] | ((uint32_t)ao[1] << 8) | ((uint32_t)ao[2] << 16);
            }
        if (!record(aoFlat.data(), aoBytes, pu.gpu.blockAoSb)) return false;
    }

    if (FAILED(pu.cmd->Close())) return false;
    pu.gpu.slotCount = slotCount;
    pu.gpu.blockCount = (uint32_t)src.blockPosPool.size();
    pu.gpu.bytes = infos.size() * sizeof(lw::GpuChunkInfo)
                 + atlas.size() * sizeof(uint32_t)
                 + posBytes + colBytes + visBytes + aoBytes;
    pu.valid = true;
    return true;
}

// MAIN-THREAD phase: execute the pre-recorded copy on cmdQueue_, install the new
// buffers, and park the old ones + upload scratch on the fenced retire list.
// Cheap — a few API calls, no memcpy, no resource creation, no GPU stall.
bool Renderer::CommitLwLod(int L)
{
    if (L < 0 || L >= lw::kLodCount) return false;
    PendingLwUpload& pu = pendingUpload_[L];
    if (!pu.valid) return false;
    CollectLwRetired();

    // Park this LOD's old buffers (still read by in-flight frames) + the upload
    // scratch/cmd objects; freed once lwUploadFence_ reaches `v` — by then every
    // prior frame (in-order on cmdQueue_) plus this copy has completed.
    LwRetireBatch batch;
    auto park = [&](Microsoft::WRL::ComPtr<ID3D12Resource>& r)
    { if (r) batch.objs.push_back(r); };
    LwGpu& g = lwGpu_[L];
    park(g.chunkInfoSb); park(g.paletteSb); park(g.blockPosSb);
    park(g.blockColSb);  park(g.blockVisSb); park(g.blockAoSb);

    if (!pu.empty)
    {
        ID3D12CommandList* lists[] = { pu.cmd.Get() };
        cmdQueue_->ExecuteCommandLists(1, lists);
        for (auto& s : pu.scratch) batch.objs.push_back(s);
        if (pu.alloc) batch.objs.push_back(pu.alloc);
        if (pu.cmd)   batch.objs.push_back(pu.cmd);
    }
    const UINT64 v = ++lwUploadFenceVal_;
    cmdQueue_->Signal(lwUploadFence_.Get(), v);
    batch.fence = lwUploadFence_;
    batch.value = v;
    lwRetire_.push_back(std::move(batch));

    g = std::move(pu.gpu);                  // install new GPU buffers
    lwWorld_.lods[L] = std::move(pu.meta);  // install CPU metadata for the draw walk
    if (!pu.empty) lwHasWorld_ = true;
    pu = PendingLwUpload{};
    return true;
}

// Synchronous single-thread upload (non-streaming UploadLwWorld path).
bool Renderer::UploadLwLod(const lw::World& w, int L)
{
    return PrepareLwLod(w, L) && CommitLwLod(L);
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
        lwWorld_.lods[L].blockAoPool.clear();
        lwWorld_.lods[L].blockAoPool.shrink_to_fit();
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
        L"shaders/m4_octet.hlsl",
        L"shaders/m4_octetgeo.hlsl",
        L"shaders/m4_common.hlsli",  // shared includes — touch → recompile all
        L"shaders/m4_frame.hlsli",
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

#if !defined(VOXELTEST_XBOX)
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
    ComPtr<IDxcBlob> csDilateSw;
    if (!comp(L"shaders/m4_lw.hlsl",        L"csmain_dilate_swizzle", L"cs_6_0", csDilateSw, "csDilateSw")) return false;
    if (!comp(L"shaders/m4_lw.hlsl",        L"vsmain_resolve",     L"vs_6_0", vsR,   "vsR")) return false;
    if (!comp(L"shaders/m4_lw.hlsl",        L"psmain_resolve",     L"ps_6_0", psR,   "psR")) return false;
    if (!comp(L"shaders/m4_taa_post.hlsl",  L"vsmain_taa",         L"vs_6_0", vsTaa, "vsTaa")) return false;
    if (!comp(L"shaders/m4_taa_post.hlsl",  L"psmain_taa",         L"ps_6_0", psTaa, "psTaa")) return false;
    if (!comp(L"shaders/m4_taa_post.hlsl",  L"psmain_post",        L"ps_6_0", psPost,"psPost")) return false;
    if (!comp(L"shaders/m4_taa_post.hlsl",  L"psmain_godray_mark", L"ps_6_0", psGrMark, "psGrMark")) return false;
    if (!comp(L"shaders/m4_taa_post.hlsl",  L"psmain_godray_blur", L"ps_6_0", psGrBlur, "psGrBlur")) return false;
    ComPtr<IDxcBlob> vsOct, psOct, vsGeo, psGeo;
    if (!comp(L"shaders/m4_octet.hlsl",     L"vsmain_octet",       L"vs_6_0", vsOct, "vsOct")) return false;
    if (!comp(L"shaders/m4_octet.hlsl",     L"psmain_octet",       L"ps_6_0", psOct, "psOct")) return false;
    if (!comp(L"shaders/m4_octetgeo.hlsl",  L"vsmain_octetgeo",    L"vs_6_0", vsGeo, "vsGeo")) return false;
    if (!comp(L"shaders/m4_octetgeo.hlsl",  L"psmain_octetgeo",    L"ps_6_0", psGeo, "psGeo")) return false;

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

    ComPtr<ID3D12PipelineState> p1, p2, pd, pdsw, prs, ptaa, ppost, pgm, pgb, pgeo;
    if (!buildCs(m4Pass1RootSig_.Get(),  cs1.Get(),      p1))   return false;
    if (!buildCs(m4Pass2RootSig_.Get(),  cs2.Get(),      p2))   return false;
    if (!buildCs(m4DilateRootSig_.Get(), csDilate.Get(), pd))   return false;
    if (!buildCs(m4DilateRootSig_.Get(), csDilateSw.Get(), pdsw)) return false;
    if (!buildGfx(vsR.Get(),   psR.Get(),  DXGI_FORMAT_R11G11B10_FLOAT, false, m4ResolveRootSig_.Get(), prs))   return false;
    if (!buildGfx(vsTaa.Get(), psTaa.Get(),DXGI_FORMAT_R11G11B10_FLOAT, false, m4TaaRootSig_.Get(),     ptaa))  return false;
    if (!buildGfx(vsTaa.Get(), psPost.Get(),BackBufferFormat(),         true,  m4PostRootSig_.Get(),    ppost)) return false;
    if (!buildGfx(vsTaa.Get(), psGrMark.Get(),DXGI_FORMAT_R8_UNORM,    false, m4GodrayMarkRootSig_.Get(), pgm)) return false;
    if (!buildGfx(vsTaa.Get(), psGrBlur.Get(),DXGI_FORMAT_R8_UNORM,    false, m4GodrayBlurRootSig_.Get(), pgb)) return false;

    // Octet PSO: depth ON (reverse-Z GREATER+write), 2 RTVs (R32_UINT + R32_FLOAT).
    ComPtr<ID3D12PipelineState> poct;
    {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC od{};
        od.pRootSignature = m4OctetRootSig_.Get();
        od.VS = { vsOct->GetBufferPointer(), vsOct->GetBufferSize() };
        od.PS = { psOct->GetBufferPointer(), psOct->GetBufferSize() };
        od.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        od.SampleMask = UINT_MAX;
        od.SampleDesc.Count = 1;
        od.NumRenderTargets = 2;
        od.RTVFormats[0] = DXGI_FORMAT_R32_UINT;
        od.RTVFormats[1] = DXGI_FORMAT_R32_FLOAT;
        od.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        od.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        od.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        od.RasterizerState.DepthClipEnable = TRUE;
        for (auto& bl : od.BlendState.RenderTarget) bl.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        od.DepthStencilState.DepthEnable = TRUE;
        od.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        od.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_GREATER;
        od.DepthStencilState.StencilEnable = FALSE;
        if (FAILED(device_->CreateGraphicsPipelineState(&od, IID_PPV_ARGS(&poct)))) return false;
        // OctetGeo: same desc, swap shaders.
        od.VS = { vsGeo->GetBufferPointer(), vsGeo->GetBufferSize() };
        od.PS = { psGeo->GetBufferPointer(), psGeo->GetBufferSize() };
        if (FAILED(device_->CreateGraphicsPipelineState(&od, IID_PPV_ARGS(&pgeo)))) return false;
    }

    m4Pass1Pso_       = p1;
    m4Pass2Pso_       = p2;
    m4DilatePso_      = pd;
    m4DilateSwizzlePso_ = pdsw;
    m4ResolvePso_     = prs;
    m4TaaPso_         = ptaa;
    m4PostPso_        = ppost;
    m4GodrayMarkPso_  = pgm;
    m4GodrayBlurPso_  = pgb;
    m4OctetPso_       = poct;
    m4OctetGeoPso_    = pgeo;
    return true;
}

#endif // !VOXELTEST_XBOX — end of hot-reload block; CreateM4/Vis are PC+Xbox

// ===== M4 — PointCS_Block compute rasterizer ================================

bool Renderer::CreateM4()
{
    D3D12_DESCRIPTOR_HEAP_DESC td{};
    td.NumDescriptors = 16; // 0..13 + visAoUav(14) + visAoSrv(15)
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
        D3D12_DESCRIPTOR_RANGE aoUavRange{};
        aoUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        aoUavRange.NumDescriptors = 1;
        aoUavRange.BaseShaderRegister = 1;  // u1 = visAo
        aoUavRange.OffsetInDescriptorsFromTableStart = 0;

        D3D12_ROOT_PARAMETER p[12]{};
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
        p[9].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; p[9].Descriptor = {6,0};  // blockVis
        p[10].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; p[10].Descriptor = {7,0}; // blockAo
        p[11].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[11].DescriptorTable = { 1, &aoUavRange }; // u1 = visAo splat texture
        for (auto& x : p) x.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_ROOT_SIGNATURE_DESC rsd{};
        rsd.NumParameters = 12;
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
        D3D12_DESCRIPTOR_RANGE aoSrvRange{};
        aoSrvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        aoSrvRange.NumDescriptors = 1;
        aoSrvRange.BaseShaderRegister = 2;  // t2 = visAo
        D3D12_ROOT_PARAMETER p[6]{};
        p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[0].Descriptor = {0, 0};
        p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[1].Descriptor = {1, 0};
        p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[2].DescriptorTable = {1, &srvRange};
        p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[3].DescriptorTable = {1, &uavRange};
        p[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[4].DescriptorTable = {1, &aoSrvRange}; // t2 = visAo splat texture
        // b2 = per-dispatch base group column (swizzle dilate; 0 otherwise).
        p[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        p[5].Constants = {2, 0, 1}; // register b2, space 0, 1 value
        for (auto& x : p) x.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_ROOT_SIGNATURE_DESC rsd{};
        rsd.NumParameters = 6;
        rsd.pParameters = p;
        if (!buildRootSig(rsd, m4DilateRootSig_, L"m4DilateRootSig")) return false;
    }

    // OctetBillboards root sig: CBV b0 (frame) + CBV b1 (work counts) + root SRVs
    // t0 chunkInfo, t1 palette, t2 blockPos, t3 blockCol, t4 workItems,
    // t6 blockVis, t7 blockAo. Same root-SRV-by-GVA pattern as the compute path.
    {
        D3D12_ROOT_PARAMETER p[9]{};
        p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[0].Descriptor = {0,0}; // b0
        p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[1].Descriptor = {1,0}; // b1
        p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; p[2].Descriptor = {0,0}; // chunkInfo
        p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; p[3].Descriptor = {1,0}; // palette
        p[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; p[4].Descriptor = {2,0}; // blockPos
        p[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; p[5].Descriptor = {3,0}; // blockCol
        p[6].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; p[6].Descriptor = {4,0}; // workItems
        p[7].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; p[7].Descriptor = {6,0}; // blockVis
        p[8].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; p[8].Descriptor = {7,0}; // blockAo
        for (auto& x : p) x.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_ROOT_SIGNATURE_DESC rsd{};
        rsd.NumParameters = 9;
        rsd.pParameters = p;
        if (!buildRootSig(rsd, m4OctetRootSig_, L"m4OctetRootSig")) return false;
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

    // Post root sig — psmain_post needs b0 + b3 (cbGodray) +
    // t7 (postIn) + t9 (godrayTex) + t10 (godrayHistTex) + s0.
    {
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
        D3D12_ROOT_PARAMETER p[5]{};
        p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[0].Descriptor = {0, 0};
        p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; p[1].Descriptor = {3, 0};
        p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[2].DescriptorTable = {1, &inRange};
        p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[3].DescriptorTable = {1, &gr9};
        p[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[4].DescriptorTable = {1, &gr10};
        for (auto& x : p) x.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_ROOT_SIGNATURE_DESC rsd{};
        rsd.NumParameters = 5;
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
    auto loadShader = [&](const wchar_t* hlsl, const wchar_t* entry,
                          const wchar_t* profile, ComPtr<IDxcBlob>& out,
                          const char* tag) -> bool
    {
#if defined(VOXELTEST_XBOX)
        (void)profile;
        std::wstring cso = std::wstring(L"shaders/") + (hlsl + 8) // strip "shaders/"
                         + L"_" + entry + L".cso";
        if (!shaderc_.LoadCso(cso, out, &err))
        { OutputDebugStringA(("[m4] " + std::string(tag) + " load: " + err + "\n").c_str()); return false; }
#else
        if (!shaderc_.Compile(hlsl, entry, profile, {}, out, &err))
        { OutputDebugStringA(("[m4] " + std::string(tag) + " compile: " + err + "\n").c_str()); return false; }
#endif
        return true;
    };
    if (!loadShader(L"shaders/m4_lw.hlsl",       L"csmain_pass1_depth", L"cs_6_0", cs1,      "cs1")) return false;
    if (!loadShader(L"shaders/m4_lw.hlsl",       L"csmain_pass2_color", L"cs_6_0", cs2,      "cs2")) return false;
    if (!loadShader(L"shaders/m4_lw.hlsl",       L"csmain_dilate",      L"cs_6_0", csDilate, "csDilate")) return false;
    ComPtr<IDxcBlob> csDilateSw;
    if (!loadShader(L"shaders/m4_lw.hlsl",       L"csmain_dilate_swizzle", L"cs_6_0", csDilateSw, "csDilateSw")) return false;
    if (!loadShader(L"shaders/m4_lw.hlsl",       L"vsmain_resolve",     L"vs_6_0", vsR,      "vsR")) return false;
    if (!loadShader(L"shaders/m4_lw.hlsl",       L"psmain_resolve",     L"ps_6_0", psR,      "psR")) return false;
    if (!loadShader(L"shaders/m4_taa_post.hlsl", L"vsmain_taa",         L"vs_6_0", vsTaa,    "vsTaa")) return false;
    if (!loadShader(L"shaders/m4_taa_post.hlsl", L"psmain_taa",         L"ps_6_0", psTaa,    "psTaa")) return false;
    if (!loadShader(L"shaders/m4_taa_post.hlsl", L"psmain_post",        L"ps_6_0", psPost,   "psPost")) return false;
    if (!loadShader(L"shaders/m4_taa_post.hlsl", L"psmain_godray_mark", L"ps_6_0", psGrMark, "psGrMark")) return false;
    if (!loadShader(L"shaders/m4_taa_post.hlsl", L"psmain_godray_blur", L"ps_6_0", psGrBlur, "psGrBlur")) return false;

    if (!buildComputePso(m4Pass1RootSig_.Get(), cs1.Get(), m4Pass1Pso_, L"m4Pass1Pso")) return false;
    if (!buildComputePso(m4Pass2RootSig_.Get(), cs2.Get(), m4Pass2Pso_, L"m4Pass2Pso")) return false;
    if (!buildComputePso(m4DilateRootSig_.Get(), csDilate.Get(), m4DilatePso_, L"m4DilatePso")) return false;
    if (!buildComputePso(m4DilateRootSig_.Get(), csDilateSw.Get(), m4DilateSwizzlePso_, L"m4DilateSwizzlePso")) return false;

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
    if (!buildFsTri(vsR.Get(), psR.Get(),    DXGI_FORMAT_R11G11B10_FLOAT, false,
                    m4ResolveRootSig_.Get(), m4ResolvePso_, L"m4ResolvePso")) return false;
    if (!buildFsTri(vsTaa.Get(), psTaa.Get(), DXGI_FORMAT_R11G11B10_FLOAT, false,
                    m4TaaRootSig_.Get(), m4TaaPso_, L"m4TaaPso")) return false;
    if (!buildFsTri(vsTaa.Get(), psPost.Get(), BackBufferFormat(),         true,
                    m4PostRootSig_.Get(), m4PostPso_, L"m4PostPso")) return false;

    // ---- OctetBillboards graphics PSO (depth ON, reverse-Z GREATER + write) ----
    // Shaders precompiled to .cso for Xbox (see <ShaderEntry> in voxeltest.vcxproj).
    {
        ComPtr<IDxcBlob> vsOct, psOct;
        if (!loadShader(L"shaders/m4_octet.hlsl", L"vsmain_octet", L"vs_6_0", vsOct, "vsOct")) return false;
        if (!loadShader(L"shaders/m4_octet.hlsl", L"psmain_octet", L"ps_6_0", psOct, "psOct")) return false;
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = m4OctetRootSig_.Get();
        pd.VS = { vsOct->GetBufferPointer(), vsOct->GetBufferSize() };
        pd.PS = { psOct->GetBufferPointer(), psOct->GetBufferSize() };
        pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pd.SampleMask = UINT_MAX;
        pd.SampleDesc.Count = 1;
        pd.NumRenderTargets = 2;
        pd.RTVFormats[0] = DXGI_FORMAT_R32_UINT;  // visColor2 (packed color)
        pd.RTVFormats[1] = DXGI_FORMAT_R32_FLOAT; // visDepth2 (gNearZ/viewZ)
        pd.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pd.RasterizerState.DepthClipEnable = TRUE;
        for (auto& blend : pd.BlendState.RenderTarget)
            blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pd.DepthStencilState.DepthEnable = TRUE;
        pd.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        pd.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_GREATER; // reverse-Z
        pd.DepthStencilState.StencilEnable = FALSE;
        if (FAILED(device_->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&m4OctetPso_)))) return false;
        NameObject(m4OctetPso_.Get(), L"m4OctetPso");

        // OctetGeo PSO: same root sig / RTV / depth, just different shaders +
        // triangle-list topology (real geometry, no per-pixel ray).
        ComPtr<IDxcBlob> vsGeo, psGeo;
        if (!loadShader(L"shaders/m4_octetgeo.hlsl", L"vsmain_octetgeo", L"vs_6_0", vsGeo, "vsGeo")) return false;
        if (!loadShader(L"shaders/m4_octetgeo.hlsl", L"psmain_octetgeo", L"ps_6_0", psGeo, "psGeo")) return false;
        pd.VS = { vsGeo->GetBufferPointer(), vsGeo->GetBufferSize() };
        pd.PS = { psGeo->GetBufferPointer(), psGeo->GetBufferSize() };
        if (FAILED(device_->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&m4OctetGeoPso_)))) return false;
        NameObject(m4OctetGeoPso_.Get(), L"m4OctetGeoPso");
    }

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

    if (!buildFsTri(vsTaa.Get(), psGrMark.Get(), DXGI_FORMAT_R8_UNORM, false,
                    m4GodrayMarkRootSig_.Get(), m4GodrayMarkPso_, L"m4GodrayMarkPso")) return false;
    if (!buildFsTri(vsTaa.Get(), psGrBlur.Get(), DXGI_FORMAT_R8_UNORM, false,
                    m4GodrayBlurRootSig_.Get(), m4GodrayBlurPso_, L"m4GodrayBlurPso")) return false;

    // RTV heap: 0,1=taaHist 2=taaScene 3=godrayMark 4,5=godrayBlur[1,2]
    //           6=visColor2 7=visDepth2 (octet composite RTVs).
    D3D12_DESCRIPTOR_HEAP_DESC rh{};
    rh.NumDescriptors = 8;
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
    visAoTex_.Reset();
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
        // RENDER_TARGET too: visColor2 is RTV-written by the OctetBillboards pass
        // (composited over the dilate output). Harmless capability on the others.
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                 | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
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
        rd.Format = DXGI_FORMAT_R11G11B10_FLOAT;
        rd.SampleDesc.Count = 1;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_CLEAR_VALUE cv{};
        cv.Format = DXGI_FORMAT_R11G11B10_FLOAT;
        if (FAILED(device_->CreateCommittedResource(
                       &hp, D3D12_HEAP_FLAG_NONE, &rd,
                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &cv,
                       IID_PPV_ARGS(&out)))) return false;
        NameObject(out.Get(), name);
        return true;
    };
    if (!createUintTex(visDepthTex_,  L"visDepthTex"))  return false;
    if (!createUintTex(visColorTex_,  L"visColorTex"))  return false;
    if (!createUintTex(visAoTex_,     L"visAoTex"))     return false;
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
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                 | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET; // octet RTV write
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
        rd.Format = DXGI_FORMAT_R11G11B10_FLOAT;
        rd.SampleDesc.Count = 1;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_CLEAR_VALUE cv{}; cv.Format = DXGI_FORMAT_R11G11B10_FLOAT;
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
        rd.Format = DXGI_FORMAT_R8_UNORM;
        rd.SampleDesc.Count = 1;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_CLEAR_VALUE cv{}; cv.Format = DXGI_FORMAT_R8_UNORM;
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
    srvRgba.Format = DXGI_FORMAT_R11G11B10_FLOAT;
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
        srvR16.Format = DXGI_FORMAT_R8_UNORM;
        srvR16.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvR16.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvR16.Texture2D.MipLevels = 1;
        device_->CreateShaderResourceView (godrayTex_[0].Get(),  &srvR16,          cpu(heap, 11));
        device_->CreateShaderResourceView (godrayTex_[1].Get(),  &srvR16,          cpu(heap, 12));
        device_->CreateShaderResourceView (godrayTex_[2].Get(),  &srvR16,          cpu(heap, 13));
        // Per-face AO splat texture: UAV (pass2 write) + SRV (dilate read).
        device_->CreateUnorderedAccessView(visAoTex_.Get(),      nullptr, &uavD,    cpu(heap, 14));
        device_->CreateShaderResourceView (visAoTex_.Get(),      &srvUint,         cpu(heap, 15));
    }

    // RTVs into taaRtvHeap_: 0,1=taaHist 2=taaScene 3,4,5=godray[0..2].
    {
        D3D12_RENDER_TARGET_VIEW_DESC rtvD{};
        rtvD.Format = DXGI_FORMAT_R11G11B10_FLOAT;
        rtvD.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        D3D12_CPU_DESCRIPTOR_HANDLE h = taaRtvHeap_->GetCPUDescriptorHandleForHeapStart();
        device_->CreateRenderTargetView(taaHistTex_[0].Get(), &rtvD, h); h.ptr += taaRtvDescSize_;
        device_->CreateRenderTargetView(taaHistTex_[1].Get(), &rtvD, h); h.ptr += taaRtvDescSize_;
        device_->CreateRenderTargetView(taaSceneTex_.Get(),   &rtvD, h); h.ptr += taaRtvDescSize_;
        D3D12_RENDER_TARGET_VIEW_DESC rtvGr{};
        rtvGr.Format = DXGI_FORMAT_R8_UNORM;
        rtvGr.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        device_->CreateRenderTargetView(godrayTex_[0].Get(), &rtvGr, h); h.ptr += taaRtvDescSize_;
        device_->CreateRenderTargetView(godrayTex_[1].Get(), &rtvGr, h); h.ptr += taaRtvDescSize_;
        device_->CreateRenderTargetView(godrayTex_[2].Get(), &rtvGr, h); h.ptr += taaRtvDescSize_;
        // Octet composite RTVs: 6=visColor2 (R32_UINT), 7=visDepth2 (R32_FLOAT).
        D3D12_RENDER_TARGET_VIEW_DESC rtvU{};
        rtvU.Format = DXGI_FORMAT_R32_UINT;
        rtvU.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        device_->CreateRenderTargetView(visColor2Tex_.Get(), &rtvU, h); h.ptr += taaRtvDescSize_;
        D3D12_RENDER_TARGET_VIEW_DESC rtvDf{};
        rtvDf.Format = DXGI_FORMAT_R32_FLOAT;
        rtvDf.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        device_->CreateRenderTargetView(visDepth2Tex_.Get(), &rtvDf, h);
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
        srvR16.Format = DXGI_FORMAT_R8_UNORM;
        srvR16.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvR16.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvR16.Texture2D.MipLevels = 1;
        device_->CreateShaderResourceView(godrayTex_[0].Get(), &srvR16, imguiGodrayMarkCpu_);
        device_->CreateShaderResourceView(godrayTex_[1].Get(), &srvR16, imguiGodrayBlurCpu_);
    }

    taaHistValid_[0] = false;
    taaHistValid_[1] = false;
    godrayHistCleared_ = false;
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

#if !defined(VOXELTEST_XBOX)
    PollShaderHotReload();
#endif

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
    // nearDist used only by the octet path (front-to-back sort for early-Z).
    struct DrawItem { uint32_t slot, blockFirst, blockCount; float nearDist; };
    std::vector<DrawItem> drawList[lw::kLodCount];
    // OctetBillboards close ring: only the nearest LOD0 clusters billboard, so
    // overdraw + instance count stay bounded (the splat covers the rest of LOD0).
    // Three LOD0 techs chosen per cluster by the closest voxel's SCREEN SIZE
    // (pixels-per-voxel): >= geoMinPx → OctetGeo, else >= bbMinPx → OctetBillboards,
    // else → SplatCS. Each is independently toggleable (viz which pixels it draws).
    // OctetGeo only wins where its threshold is the higher one — if geoMinPx <
    // bbMinPx the billboard band always covers it, so OctetGeo never runs.
    std::vector<DrawItem> octetDrawList[2]; // [0]=billboard [1]=geo
    // Billboard kicks in below the splat's effective resolution (dilate radius * 2);
    // geoMinPx (UI) is where OctetGeo takes over from the billboard.
    const float bbMinPx  = (float)args.splatRadius * 2.0f;
    const float geoMinPx = args.geoMinPx;

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
            // pixels-per-voxel of the closest voxel = focalPx * voxelSize / dist.
            // Classify by size only (enables gate at draw time so disabled techs
            // leave visible holes). Geo wins only when geoMinPx is the higher band.
            const float voxelPx = focalPx * lodScaleF / std::max(distNearC, 1e-3f);
            if (args.splatOnlyDebug)
            {
                // Debug: force every cluster (incl. close LOD0) through the splat +
                // dilate path so the dilate passes can be inspected in isolation.
                drawList[L].push_back({chunkSlot,
                                       rc.clusterBlockFirst[clSlot],
                                       rc.clusterBlockCount[clSlot], 0.0f});
            }
            else if (L == 0 && geoMinPx >= bbMinPx && voxelPx >= geoMinPx)
            {
                octetDrawList[1].push_back({chunkSlot,
                                            rc.clusterBlockFirst[clSlot],
                                            rc.clusterBlockCount[clSlot], distNearC}); // geo
            }
            else if (L == 0 && voxelPx >= bbMinPx)
            {
                octetDrawList[0].push_back({chunkSlot,
                                            rc.clusterBlockFirst[clSlot],
                                            rc.clusterBlockCount[clSlot], distNearC}); // billboard
            }
            else
            {
                drawList[L].push_back({chunkSlot,
                                       rc.clusterBlockFirst[clSlot],
                                       rc.clusterBlockCount[clSlot], 0.0f}); // splat
            }
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

    // Per-tech enable toggles — disabling a tech leaves its clusters undrawn so you
    // can see exactly which pixels each tech owns.
    if (!args.enableSplat)
        for (int L = 0; L < lw::kLodCount; ++L) drawList[L].clear();
    if (!args.enableBillboard || !m4OctetPso_)    octetDrawList[0].clear();
    if (!args.enableGeo       || !m4OctetGeoPso_) octetDrawList[1].clear();

    bool anyDraw = !octetDrawList[0].empty() || !octetDrawList[1].empty();
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
        // Same runaway guard as the octet path — a corrupt chunk's blockCount
        // would balloon the splat dispatch and TDR. 50M blocks ≈ absurd.
        const uint32_t kLodMaxBlocks = 50u * 1000u * 1000u;
        if (perLodTotal[L] > kLodMaxBlocks)
        {
            std::printf("[splat] ASSERT LOD%d total=%u exceeds cap %u (items=%zu)\n",
                        L, perLodTotal[L], kLodMaxBlocks, wl.size());
            std::fflush(stdout);
            assert(perLodTotal[L] <= kLodMaxBlocks && "splat block count runaway");
            wl.clear(); perLodTotal[L] = 0;
        }
    }

    // Octet worklists (one per tech). Sorted FRONT-TO-BACK by cluster near-distance
    // so the closest clusters rasterise first and write their conservative-near
    // depth — HW early-Z then rejects octets occluded behind them before their PS
    // runs. (No slot-coalesce: depth order scatters slots, and the per-cluster WI
    // count is tiny relative to the instance/overdraw win.)
    std::vector<WI> octetWl[2];
    uint32_t octetTotal[2] = {};
    for (int t = 0; t < 2; ++t)
    {
        auto& dl = octetDrawList[t];
        if (dl.empty()) continue;
        std::sort(dl.begin(), dl.end(),
                  [](const DrawItem& a, const DrawItem& b) { return a.nearDist < b.nearDist; });
        const lw::LODWorld& lwL = lwWorld_.lods[0];
        octetWl[t].reserve(dl.size());
        for (const DrawItem& it : dl) {
            if (it.blockCount == 0) continue;
            const lw::RuntimeChunk& rc = lwL.chunks[it.slot];
            octetWl[t].push_back({ rc.slotIdx, rc.blockBase + it.blockFirst, it.blockCount, octetTotal[t] });
            octetTotal[t] += it.blockCount;
        }
        // Safety cap: never hand DrawInstanced a runaway instance count (corrupt
        // worklist would spin the GPU into a TDR). 10M octets ≈ absurd.
        const uint32_t kOctetMaxInstances = 10u * 1000u * 1000u;
        if (octetTotal[t] > kOctetMaxInstances)
        {
            std::printf("[octet] ASSERT total=%u exceeds cap %u (tech %d) — skipping\n",
                        octetTotal[t], kOctetMaxInstances, t);
            std::fflush(stdout);
            assert(octetTotal[t] <= kOctetMaxInstances && "octet instance count runaway");
            octetTotal[t] = 0;
        }
    }

    // ---- CB layouts (must match m4_lw.hlsl) ----
    // CSTiles cbPerFrame layout extended with Burnout reproject coeffs, 576 bytes.
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
        // Burnout Paradise reproject (Mvel = Mh1_to_h0 - I), paper rows x/y/w
        // in UV-space HScreen. Used by psmain_taa to skip world-recon + divide.
        float reprojMx[4];           // 512  (mxx, mxy, mxz, mxw)
        float reprojMy[4];           // 528  (myx, myy, myz, myw)
        float reprojMw[4];           // 544  (mwx, mwy, mwz, mww)
        float _padReproj[4];         // 560..576
    };
    static_assert(sizeof(CBFrame) == 576, "CBFrame size mismatch");
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
    cbf.ambient = std::max(0.0f, args.ambient);
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
    cbf._padPC[0] = std::max(0.0f, args.aoStrength); // baked AO darkening strength
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

    // Burnout Paradise reproject matrix for TAA. Mvel = Mh1_to_h0 - I, in
    // HScreen-UV space (row-vec convention to match m4_lw.hlsl row_major).
    //   V_uv:  clip -> hscreen_uv:  (x,y,z,w) -> (0.5x+0.5w, -0.5y+0.5w, z, w)
    //   C = inv(vp_curr) * vp_prev  (row-vec: clip_curr * C = clip_prev)
    //   Mh1_to_h0 = invV_uv * C * V_uv
    {
        const hlslpp::float4x4 V_uv(
            0.5f,  0.0f, 0.0f, 0.0f,
            0.0f, -0.5f, 0.0f, 0.0f,
            0.0f,  0.0f, 1.0f, 0.0f,
            0.5f,  0.5f, 0.0f, 1.0f);
        const hlslpp::float4x4 invV_uv(
            2.0f,  0.0f, 0.0f, 0.0f,
            0.0f, -2.0f, 0.0f, 0.0f,
            0.0f,  0.0f, 1.0f, 0.0f,
           -1.0f,  1.0f, 0.0f, 1.0f);
        hlslpp::float4x4 vpPrev(
            taaPrevVP_[ 0], taaPrevVP_[ 1], taaPrevVP_[ 2], taaPrevVP_[ 3],
            taaPrevVP_[ 4], taaPrevVP_[ 5], taaPrevVP_[ 6], taaPrevVP_[ 7],
            taaPrevVP_[ 8], taaPrevVP_[ 9], taaPrevVP_[10], taaPrevVP_[11],
            taaPrevVP_[12], taaPrevVP_[13], taaPrevVP_[14], taaPrevVP_[15]);
        hlslpp::float4x4 invVpCurr = hlslpp::inverse(vp);
        hlslpp::float4x4 C        = hlslpp::mul(invVpCurr, vpPrev);
        hlslpp::float4x4 H        = hlslpp::mul(hlslpp::mul(invV_uv, C), V_uv);
        float M[16];
        hlslpp::store(M, H);
        // Subtract identity → Mvel.
        M[ 0] -= 1.0f; M[ 5] -= 1.0f; M[10] -= 1.0f; M[15] -= 1.0f;
        // Paper rows in column-vec form map to columns of row-vec storage.
        // Mvel_rv[r][c] = M[r*4 + c]. Paper Mx row = column 0, My = col 1, Mw = col 3.
        cbf.reprojMx[0] = M[ 0]; cbf.reprojMx[1] = M[ 4]; cbf.reprojMx[2] = M[ 8]; cbf.reprojMx[3] = M[12];
        cbf.reprojMy[0] = M[ 1]; cbf.reprojMy[1] = M[ 5]; cbf.reprojMy[2] = M[ 9]; cbf.reprojMy[3] = M[13];
        cbf.reprojMw[0] = M[ 3]; cbf.reprojMw[1] = M[ 7]; cbf.reprojMw[2] = M[11]; cbf.reprojMw[3] = M[15];
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
        cbcs._pad[1]     = (uint32_t)std::max(0, args.tileCS); // gTileCS (dilate swizzle)
        cbcsAlloc[L] = graphicsMemory_->AllocateConstant(cbcs);
        wlAlloc[L]   = graphicsMemory_->Allocate(perLodWl[L].size() * sizeof(WI));
        std::memcpy(wlAlloc[L].Memory(), perLodWl[L].data(), perLodWl[L].size() * sizeof(WI));
    }

    // Octet worklist + CB allocations, per tech (composite on top of the splat
    // output visColor2/visDepth2 after the dilate — see the sub-pass below).
    DirectX::GraphicsResource octetCbAlloc[2], octetWlAlloc[2];
    for (int t = 0; t < 2; ++t)
    {
        if (octetTotal[t] == 0 || octetWl[t].empty()) continue;
        CBLwCs cbcs{};
        cbcs.vwSize[0] = visTexW_;
        cbcs.vwSize[1] = visTexH_;
        cbcs.pointCount = octetTotal[t];
        cbcs.numItems   = (uint32_t)octetWl[t].size();
        cbcs.lodIdx     = 0;
        cbcs.splatRadius = 0;
        cbcs._pad[0]     = lwGpu_[0].blockAoSb ? (lwGpu_[0].blockCount * 8u) : 0u;  // gAoCount: clamp root-SRV read
        octetCbAlloc[t] = graphicsMemory_->AllocateConstant(cbcs);
        octetWlAlloc[t] = graphicsMemory_->Allocate(octetWl[t].size() * sizeof(WI));
        std::memcpy(octetWlAlloc[t].Memory(), octetWl[t].data(), octetWl[t].size() * sizeof(WI));
    }
    const bool octetAny = (octetTotal[0] > 0 && !octetWl[0].empty())
                       || (octetTotal[1] > 0 && !octetWl[1].empty());

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
            const LwGpu& g = lwGpu_[L];
            if (!g.chunkInfoSb || !g.blockPosSb) continue;   // LOD upload still pending
            if (!firstPass1) uavBarrierDepth();
            firstPass1 = false;
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
        cmdList_->SetComputeRootDescriptorTable(7, gpu(2));  // t5 = depthSrv
        cmdList_->SetComputeRootDescriptorTable(8, gpu(1));  // u0 = colorUav
        cmdList_->SetComputeRootDescriptorTable(11, gpu(14)); // u1 = visAoUav
        bool firstPass2 = true;
        for (int L = 0; L < lw::kLodCount; ++L)
        {
            if (perLodWl[L].empty()) continue;
            const LwGpu& g = lwGpu_[L];
            if (!g.chunkInfoSb || !g.paletteSb || !g.blockPosSb || !g.blockColSb) continue;
            if (!firstPass2) uavBarrierColor();
            firstPass2 = false;
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
            if (g.blockAoSb)
                cmdList_->SetComputeRootShaderResourceView(10, g.blockAoSb->GetGPUVirtualAddress());
            else
                cmdList_->SetComputeRootShaderResourceView(10, g.blockPosSb->GetGPUVirtualAddress()); // fallback (AO viz reads garbage)
            UINT groups = (perLodTotal[L] + 63) / 64;
            cmdList_->Dispatch(groups, 1, 1);
        }
    }

    // Transition visColor + visAo UAV → NON_PIXEL_SRV for dilate input. visDepth
    // stays NON_PIXEL_SRV (dilate also reads it).
    {
        D3D12_RESOURCE_BARRIER b[2]{};
        for (int k = 0; k < 2; ++k)
        {
            b[k].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b[k].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            b[k].Transition.StateAfter  = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            b[k].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        }
        b[0].Transition.pResource = visColorTex_.Get();
        b[1].Transition.pResource = visAoTex_.Get();
        cmdList_->ResourceBarrier(2, b);
    }

    // ---- Dilate compute pass: visColor + visDepth → visColor2 ----
    {
        MICROPROFILE_SCOPEGPUI("LW/Dilate", 0xffd0a0c0);
        cmdList_->SetComputeRootSignature(m4DilateRootSig_.Get());
        // tileCS != 0 → thread-group-ID swizzle variant for L2 locality.
        cmdList_->SetPipelineState((args.tileCS != 0 && m4DilateSwizzlePso_)
                                   ? m4DilateSwizzlePso_.Get() : m4DilatePso_.Get());
        D3D12_GPU_VIRTUAL_ADDRESS csCb = 0;
        for (int L = 0; L < lw::kLodCount; ++L)
            if (!perLodWl[L].empty()) { csCb = cbcsAlloc[L].GpuAddress(); break; }
        cmdList_->SetComputeRootConstantBufferView(0, cbfAlloc.GpuAddress()); // b0 cam basis
        cmdList_->SetComputeRootConstantBufferView(1, csCb);                  // b1 vwSize/radius
        cmdList_->SetComputeRootDescriptorTable(2, gpu(2));                   // t0..t1
        cmdList_->SetComputeRootDescriptorTable(3, gpu(6));                   // u0=visColor2, u1=visDepth2
        cmdList_->SetComputeRootDescriptorTable(4, gpu(15));                  // t2 = visAo
        UINT gx = (visTexW_ + 7) / 8;
        UINT gy = (visTexH_ + 7) / 8;
        if (args.tileCS != 0 && m4DilateSwizzlePso_)
        {
            // Tile width in the Z dim, two EXACT dispatches (no over-range groups):
            //   main slabs cover [0, full*tileW); remainder covers the leftover cols.
            UINT tileW = (UINT)args.tileCS;
            UINT full  = gx / tileW;          // whole slabs
            UINT rem   = gx % tileW;          // leftover columns
            if (full > 0)
            {
                cmdList_->SetComputeRoot32BitConstant(5, 0u, 0);          // base column = 0
                cmdList_->Dispatch(tileW, gy, full);
            }
            if (rem > 0)
            {
                cmdList_->SetComputeRoot32BitConstant(5, full * tileW, 0); // base = full*tileW
                cmdList_->Dispatch(rem, gy, 1);
            }
        }
        else
        {
            cmdList_->Dispatch(gx, gy, 1);
        }
    }

    // ---- OctetBillboards composite (LOD0 close ring) -----------------------
    // Renders the near octets over the dilate's visColor2/visDepth2 (MRT) using
    // an HW depth buffer for octet-vs-octet occlusion. PS discards on miss so the
    // splat result shows through; close octets (nearest ring) overwrite where hit.
    bool octetRan = false;
    if (octetAny)
    {
        const LwGpu& g = lwGpu_[0];
        if (g.chunkInfoSb && g.paletteSb && g.blockPosSb && g.blockColSb)
        {
            octetRan = true;
            // visColor2 + visDepth2: UAV → RENDER_TARGET.
            D3D12_RESOURCE_BARRIER tb[2]{};
            for (int k = 0; k < 2; ++k) {
                tb[k].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                tb[k].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                tb[k].Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
                tb[k].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            }
            tb[0].Transition.pResource = visColor2Tex_.Get();
            tb[1].Transition.pResource = visDepth2Tex_.Get();
            cmdList_->ResourceBarrier(2, tb);

            auto rtvStart = taaRtvHeap_->GetCPUDescriptorHandleForHeapStart();
            D3D12_CPU_DESCRIPTOR_HANDLE rtvs[2];
            rtvs[0] = rtvStart; rtvs[0].ptr += SIZE_T(6) * taaRtvDescSize_; // visColor2
            rtvs[1] = rtvStart; rtvs[1].ptr += SIZE_T(7) * taaRtvDescSize_; // visDepth2
            D3D12_CPU_DESCRIPTOR_HANDLE dsv = dsvHeap_->GetCPUDescriptorHandleForHeapStart();
            cmdList_->OMSetRenderTargets(2, rtvs, FALSE, &dsv);
            cmdList_->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.0f, 0, 0, nullptr); // reverse-Z far
            D3D12_VIEWPORT vpr{ 0, 0, (float)visTexW_, (float)visTexH_, 0.0f, 1.0f };
            D3D12_RECT     scr{ 0, 0, (LONG)visTexW_, (LONG)visTexH_ };
            cmdList_->RSSetViewports(1, &vpr);
            cmdList_->RSSetScissorRects(1, &scr);
            cmdList_->SetGraphicsRootSignature(m4OctetRootSig_.Get());
            // Shared SRV binds (same root sig for both octet PSOs).
            cmdList_->SetGraphicsRootConstantBufferView(0, cbfAlloc.GpuAddress());
            cmdList_->SetGraphicsRootShaderResourceView(2, g.chunkInfoSb->GetGPUVirtualAddress());
            cmdList_->SetGraphicsRootShaderResourceView(3, g.paletteSb->GetGPUVirtualAddress());
            cmdList_->SetGraphicsRootShaderResourceView(4, g.blockPosSb->GetGPUVirtualAddress());
            cmdList_->SetGraphicsRootShaderResourceView(5, g.blockColSb->GetGPUVirtualAddress());
            cmdList_->SetGraphicsRootShaderResourceView(7, g.blockVisSb ? g.blockVisSb->GetGPUVirtualAddress()
                                                                        : g.blockPosSb->GetGPUVirtualAddress());
            cmdList_->SetGraphicsRootShaderResourceView(8, g.blockAoSb ? g.blockAoSb->GetGPUVirtualAddress()
                                                                       : g.blockPosSb->GetGPUVirtualAddress());
            // Draw both octet techs into the same MRT, each with its own GPU marker.
            // Both front-to-back for early-Z. t=0 billboards (4-vert strip), t=1 geo
            // (144 verts/instance = 8 voxels * 3 faces * 2 tris * 3 verts).
            if (octetTotal[0] > 0 && !octetWl[0].empty())
            {
                MICROPROFILE_SCOPEGPUI("LW/OctetBillboards", 0xff40c0ff);
                cmdList_->SetPipelineState(m4OctetPso_.Get());
                cmdList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
                cmdList_->SetGraphicsRootConstantBufferView(1, octetCbAlloc[0].GpuAddress());
                cmdList_->SetGraphicsRootShaderResourceView(6, octetWlAlloc[0].GpuAddress());
                cmdList_->DrawInstanced(4u, octetTotal[0], 0, 0);
            }
            if (octetTotal[1] > 0 && !octetWl[1].empty())
            {
                MICROPROFILE_SCOPEGPUI("LW/OctetGeo", 0xff40ffc0);
                cmdList_->SetPipelineState(m4OctetGeoPso_.Get());
                cmdList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                cmdList_->SetGraphicsRootConstantBufferView(1, octetCbAlloc[1].GpuAddress());
                cmdList_->SetGraphicsRootShaderResourceView(6, octetWlAlloc[1].GpuAddress());
                cmdList_->DrawInstanced(144u, octetTotal[1], 0, 0);
            }
            // Left in RENDER_TARGET; the barrier below takes them → PIXEL_SRV.
        }
    }

    // Post-dilate: visDepth back to UAV (next frame's pass1 needs it); dilated
    // outputs (UAV, or RENDER_TARGET if octet composited) → PIXEL_SRV for resolve.
    {
        const D3D12_RESOURCE_STATES c2Before = octetRan
            ? D3D12_RESOURCE_STATE_RENDER_TARGET : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_BARRIER b[4]{};
        b[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b[0].Transition.pResource = visDepthTex_.Get();
        b[0].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        b[0].Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        b[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b[1].Transition.pResource = visColor2Tex_.Get();
        b[1].Transition.StateBefore = c2Before;
        b[1].Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        b[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b[2].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b[2].Transition.pResource = visDepth2Tex_.Get();
        b[2].Transition.StateBefore = c2Before;
        b[2].Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        b[2].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        // visAo NON_PIXEL_SRV → UAV for next frame's pass2 write.
        b[3].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b[3].Transition.pResource = visAoTex_.Get();
        b[3].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        b[3].Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        b[3].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmdList_->ResourceBarrier(4, b);
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

    // ---- One-shot history clear: blur ping-pong + taaHist start uninitialized
    // (PSR state with garbage), so without this the first frame leaks junk into
    // the EMA chain that never converges out.
    if (!godrayHistCleared_)
    {
        D3D12_RESOURCE_BARRIER tb[5]{};
        for (int i = 0; i < 3; ++i) {
            tb[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            tb[i].Transition.pResource   = godrayTex_[i].Get();
            tb[i].Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            tb[i].Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
            tb[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        }
        for (int i = 0; i < 2; ++i) {
            tb[3 + i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            tb[3 + i].Transition.pResource   = taaHistTex_[i].Get();
            tb[3 + i].Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            tb[3 + i].Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
            tb[3 + i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        }
        cmdList_->ResourceBarrier(5, tb);

        D3D12_CPU_DESCRIPTOR_HANDLE rtv0 = taaRtvHeap_->GetCPUDescriptorHandleForHeapStart();
        const float zero[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        // RTV layout: 0,1=taaHist 2=taaScene 3,4,5=godray[0..2].
        D3D12_CPU_DESCRIPTOR_HANDLE rtvH0 = rtv0;                                   // taaHist[0]
        D3D12_CPU_DESCRIPTOR_HANDLE rtvH1 = rtv0; rtvH1.ptr += taaRtvDescSize_;     // taaHist[1]
        D3D12_CPU_DESCRIPTOR_HANDLE rtvG0 = rtv0; rtvG0.ptr += SIZE_T(3) * taaRtvDescSize_;
        D3D12_CPU_DESCRIPTOR_HANDLE rtvG1 = rtv0; rtvG1.ptr += SIZE_T(4) * taaRtvDescSize_;
        D3D12_CPU_DESCRIPTOR_HANDLE rtvG2 = rtv0; rtvG2.ptr += SIZE_T(5) * taaRtvDescSize_;
        cmdList_->ClearRenderTargetView(rtvH0, zero, 0, nullptr);
        cmdList_->ClearRenderTargetView(rtvH1, zero, 0, nullptr);
        cmdList_->ClearRenderTargetView(rtvG0, zero, 0, nullptr);
        cmdList_->ClearRenderTargetView(rtvG1, zero, 0, nullptr);
        cmdList_->ClearRenderTargetView(rtvG2, zero, 0, nullptr);

        for (int i = 0; i < 5; ++i)
            std::swap(tb[i].Transition.StateBefore, tb[i].Transition.StateAfter);
        cmdList_->ResourceBarrier(5, tb);

        godrayHistCleared_ = true;
    }

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
        cmdList_->SetGraphicsRootDescriptorTable(2, gpu(4 + currIdx)); // t7 = postIn = taaHist[curr]
        cmdList_->SetGraphicsRootDescriptorTable(3, gpu(12 + grCurr)); // t9 = current frame's blurred godray
        cmdList_->SetGraphicsRootDescriptorTable(4, gpu(11));          // t10 = mark (unused at post; needs valid bind)
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
