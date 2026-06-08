#pragma once
#include "camera.h"
#include "lodworld.h"
#include "shader.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <vector>
#include <string>
#include <cstdint>
#include <memory>
#include <wrl/client.h>

template <typename T>
using ComPtr = Microsoft::WRL::ComPtr<T>;

#include <GraphicsMemory.h>

enum class ShadingMode : int
{
    Lit = 0,
    FlatColor = 1,
    Normals = 2,
    Ao = 3,
    LodViz = 4,
};

// Surviving render techs (LW path). PointCS_Block + OctetBillboards wired up.
// PolyAxis / Splat reserved for future octet-driven revivals.
enum class RenderTech : int
{
    PointCS_Block    = 0,
    OctetBillboards  = 1,
    PolyAxis         = 2,
    Splat            = 3,
};

enum class PointLighting : int { Simple = 0, Complex = 1 };
enum class PointLod : int { L0 = 0, L1 = 1, L2 = 2, L3 = 3, Auto = 4 };

struct DrawSceneParams
{
    ShadingMode mode = ShadingMode::Lit;
    int gridSize = 1;
    RenderTech tech    = RenderTech::PointCS_Block;
    RenderTech techFar = RenderTech::PointCS_Block;
    bool closeEnabled = true;
    bool farEnabled   = true;
    PointLighting pointLight = PointLighting::Complex;
    PointLod pointLod = PointLod::Auto;
    float pointLodScale = 1.0f;
    bool splatFilter = true;
    float fogColor[3] = {0.55f, 0.60f, 0.70f};
    float fogDensity = 0.0f;
    float heightFogDensity = 0.0f;
    float heightFogFalloff = 0.05f;
    float heightFogStart = 0.0f;
    int splatRadius = 3;
    bool taa = true;
    float sunDir[3] = {0.4f, 0.8f, 0.2f};
    float sunIntensity = 1.0f;
    float exposure = 1.0f;
    float roughness = 0.6f;
    bool sunShadows = false;
    int shadowCascades = 1;
    int shadowMapSize = 2048;
    float shadowBias = 0.001f;
    bool shadowCullFront = false;
    bool shadowForceRebuild = false;
    int shadowLod = -1;
    // (extra TAA / godray fields preserved as in DX11 build — leave defaults)
    float godrayStrength = 0.0f;
    float godrayAngleDeg = 4.0f;
    float godrayEmaAlpha = 1.0f;
    float godrayTint[3] = {1.0f, 0.9f, 0.7f};
    bool godrayAniso = false;
    bool godraySeparable = false;
    int godraySeparableStride = 6;
    bool splatDilate2Pass = false;
    bool shadowBlur = false;
    bool lwShowBounds = false;
    bool cheapAO = false;
    float aoStrength = 0.5f;
    float aoFadeUnits = 64.0f;
    int   aoPushTexels = 0;
};

// Phase-1 (DX12 port) skeleton.
// M1 milestone: device + swapchain + clear + imgui + present.
// All voxel-rendering methods are no-ops returning safe defaults; will be
// re-implemented incrementally per the plan in memory:xbox_dx12_port_plan.md.
class Renderer
{
public:
    static constexpr UINT kFrameCount = 3;

    Renderer();
    ~Renderer();

    bool Init(HWND hwnd, int adapterIdx = -1);
    void Shutdown();

    static std::vector<std::string> EnumerateAdapters();

    // ---- LW upload path (M3 — currently stubs) ----
    bool UploadLwWorld(const lw::World&) { return false; }
    void PrepLwWorld(const lw::World&) {}
    bool UploadLwLodOnly(const lw::World&, int) { return false; }
    bool FinalizeLwUploads() { return false; }
    void ClearLwWorld() {}
    bool HasLwWorld() const { return false; }
    void DrawLwScene(const Camera&, const DrawSceneParams&) {}

    struct StreamPoolInfo
    {
        const char* name = "";
        uint64_t bytes = 0;
        uint32_t numElements = 0;
        uint32_t stride = 0;
    };
    struct StreamLodInfo { StreamPoolInfo pools[5] = {}; };
    StreamLodInfo GetStreamLodInfo(int) const { return {}; }

    uint32_t LwSlotCount(int) const { return 0; }
    uint32_t LwPointCount(int) const { return 0; }
    uint64_t LwBytes(int) const { return 0; }
    uint64_t LwTotalBytes() const { return 0; }

    // ---- Frame ----
    void Resize(uint32_t w, uint32_t h);
    void BeginFrame(float clear[4], bool skipClear = false, bool skipDsvClear = false);
    void EndFrame(bool vsync);

    // ---- Accessors (DX12) ----
    ID3D12Device*               Device() const     { return device_.Get(); }
    ID3D12GraphicsCommandList*  CommandList() const{ return cmdList_.Get(); }
    ID3D12CommandQueue*         CommandQueue() const{ return cmdQueue_.Get(); }
    DXGI_FORMAT                 BackBufferFormat() const { return DXGI_FORMAT_R8G8B8A8_UNORM; }
    UINT                        FrameIndex() const { return frameIndex_; }

    // Shader-visible CBV/SRV/UAV heap shared with imgui. Reserve a small range
    // for imgui-managed textures (font + multi-viewport icons, future debug
    // previews). Bump-allocated for M1; will switch to a proper free-list when
    // we add LW SRVs in M3.
    static constexpr UINT kImGuiSrvHeapSize = 64;
    ID3D12DescriptorHeap*       SrvHeapForImGui() const { return imguiSrvHeap_.Get(); }
    void                        ImGuiSrvAlloc(D3D12_CPU_DESCRIPTOR_HANDLE* outCpu,
                                              D3D12_GPU_DESCRIPTOR_HANDLE* outGpu);
    void                        ImGuiSrvFree(D3D12_CPU_DESCRIPTOR_HANDLE cpu,
                                             D3D12_GPU_DESCRIPTOR_HANDLE gpu);

    uint32_t Width() const  { return width_; }
    uint32_t Height() const { return height_; }
    uint64_t SplatRtBytes() const { return 0; }

    // SRV getters return null GPU handles in M1 (stub); ImGui::Image gated on .ptr != 0.
    D3D12_GPU_DESCRIPTOR_HANDLE ShadowSrv() const       { return {}; }
    D3D12_GPU_DESCRIPTOR_HANDLE ShadowFilledSrv() const { return {}; }
    D3D12_GPU_DESCRIPTOR_HANDLE GodraySrv() const       { return {}; }
    D3D12_GPU_DESCRIPTOR_HANDLE GodrayMarkSrv() const   { return {}; }
    uint32_t ShadowMapSize() const { return 0; }

    // ---- Stats (all stubs in M1) ----
    uint32_t LastDrawnCount()       const { return 0; }
    uint64_t LastPointCount()       const { return 0; }
    uint64_t LastTriCount()         const { return 0; }
    uint64_t LastPolyVoxelCount()   const { return 0; }
    uint64_t LastSplatVoxelCount()  const { return 0; }
    uint32_t LastSplatDrawCalls()   const { return 0; }
    uint32_t LastPolyDrawCalls()    const { return 0; }
    uint32_t LastFastDrawCalls()    const { return 0; }
    uint64_t LastBlockTotal()       const { return 0; }
    uint32_t LastBlockDispatches()  const { return 0; }
    uint64_t LastBlockTotalAt(int)  const { return 0; }

    enum class ReloadStatus { None, Success, Failed };
    ReloadStatus lastReloadStatus_ = ReloadStatus::None;
    uint32_t reloadCount_ = 0;
    bool ForceReloadShaders() { return false; }

private:
    bool CreateDeviceAndSwap(HWND hwnd, int adapterIdx);
    bool CreateRenderTargets();
    bool CreateM2Demo();
    void WaitForGpu();
    void MoveToNextFrame();

    ComPtr<ID3D12Device>             device_;
    ComPtr<IDXGIFactory6>            factory_;
    ComPtr<IDXGISwapChain3>          swap_;
    ComPtr<ID3D12CommandQueue>       cmdQueue_;
    ComPtr<ID3D12CommandAllocator>   cmdAlloc_[kFrameCount];
    ComPtr<ID3D12GraphicsCommandList> cmdList_;

    ComPtr<ID3D12Resource>           backBuffers_[kFrameCount];
    ComPtr<ID3D12DescriptorHeap>     rtvHeap_;
    UINT                             rtvDescSize_ = 0;

    ComPtr<ID3D12Resource>           depthTex_;
    ComPtr<ID3D12DescriptorHeap>     dsvHeap_;

    // Single CBV/SRV/UAV heap reserved for imgui font (slot 0).
    // Renderer-internal SRVs will live elsewhere later (per-LOD heap).
    ComPtr<ID3D12DescriptorHeap>     imguiSrvHeap_;
    UINT                             imguiSrvDescSize_ = 0;
    UINT                             imguiSrvNextSlot_ = 0;

    // Fence + per-frame sync.
    ComPtr<ID3D12Fence>              fence_;
    UINT64                           fenceValues_[kFrameCount] = {};
    HANDLE                           fenceEvent_ = nullptr;

    std::unique_ptr<DirectX::GraphicsMemory> graphicsMemory_;

    // M2 — validation demo (DXC compile + rootsig + PSO + IA-less triangle).
    ShaderCompiler                   shaderc_;
    ComPtr<ID3D12RootSignature>      m2RootSig_;
    ComPtr<ID3D12PipelineState>      m2Pso_;

    HWND     hwnd_ = nullptr;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    UINT     frameIndex_ = 0;
    bool     tearingSupported_ = false;
};
