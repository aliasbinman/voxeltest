#pragma once

// D3D headers come FIRST — on Xbox `d3d12_xs.h` errors if stock dxgicommon.h
// is pulled before it. Everything else (camera.h via hlsl++, lodworld.h, etc.)
// goes after.
#if defined(_GAMING_XBOX_SCARLETT) || defined(_GAMING_XBOX_XBOXONE)
  #define VOXELTEST_XBOX 1
  #include <wrl/client.h>
  #include <gxdk.h>
  #include <d3d12_xs.h>
  // Xbox D3D types don't derive from standard IUnknown — WRL's IID_PPV_ARGS
  // hits a static_assert. Redirect to IID_GRAPHICS_PPV_ARGS which expects T**.
  // Existing call sites pass `&ptr` (ComPtr<T>*). `(&ptr)->ReleaseAndGetAddressOf()`
  // == `ptr.ReleaseAndGetAddressOf()` → T**.
  #ifdef IID_PPV_ARGS
    #undef IID_PPV_ARGS
  #endif
  #define IID_PPV_ARGS(x) IID_GRAPHICS_PPV_ARGS((x).ReleaseAndGetAddressOf())
#else
  #include <d3d12.h>
  #include <dxgi1_6.h>
#endif

#include "camera.h"
#include "lodworld.h"
#include "shader.h"
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
    OctetGeo         = 4,   // real tris: 3 front faces per occupied child voxel
};

enum class PointLighting : int { Simple = 0, Complex = 1 };
enum class PointLod : int { L0 = 0, L1 = 1, L2 = 2, L3 = 3, Auto = 4 };

struct DrawSceneParams
{
    ShadingMode mode = ShadingMode::Lit;
    int gridSize = 1;
    RenderTech tech    = RenderTech::PointCS_Block; // (legacy; LOD0 tech now size-driven)
    RenderTech techFar = RenderTech::PointCS_Block;
    // Per-tech enables (independent so each tech's pixels can be isolated). The
    // billboard kicks in at splatRadius*2 px (splat's effective resolution); geoMinPx
    // is where OctetGeo takes over. geoMinPx < splatRadius*2 ⇒ Geo never used.
    bool  enableSplat     = true;
    bool  enableBillboard = true;
    bool  enableGeo       = true;
    float geoMinPx        = 8.0f;
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
    float ambient = 0.35f; // ambient term boost (gAmbient)
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
    float aoPushTexels = 1.0f;
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

    // ---- LW upload path ----
    bool UploadLwWorld(const lw::World& w);
    void PrepLwWorld(const lw::World& w);
    bool UploadLwLodOnly(const lw::World& w, int L);
    // Streaming, two-phase to keep the main thread hitch-free:
    //   PrepareLwLod — heavy (staging build + memcpy + resource create + record
    //                  copy list). Safe to call from the LOADER thread; never
    //                  touches cmdQueue_ or lwGpu_.
    //   CommitLwLod  — cheap (ExecuteCommandLists + install). MAIN thread only.
    bool PrepareLwLod(const lw::World& w, int L);
    bool CommitLwLod(int L);
    bool FinalizeLwUploads() { return true; } // combined-LOD path deferred to M4+
    void ClearLwWorld();
    bool HasLwWorld() const { return lwHasWorld_; }
    void DrawLwScene(const Camera& cam, const DrawSceneParams& args);

    struct StreamPoolInfo
    {
        const char* name = "";
        uint64_t bytes = 0;
        uint32_t numElements = 0;
        uint32_t stride = 0;
    };
    struct StreamLodInfo { StreamPoolInfo pools[5] = {}; };
    StreamLodInfo GetStreamLodInfo(int L) const;

    uint32_t LwSlotCount(int L) const
    { return (L >= 0 && L < lw::kLodCount) ? lwGpu_[L].slotCount : 0; }
    uint32_t LwPointCount(int) const { return 0; }
    uint64_t LwBytes(int L) const
    { return (L >= 0 && L < lw::kLodCount) ? lwGpu_[L].bytes : 0; }
    uint64_t LwTotalBytes() const
    {
        uint64_t t = 0;
        for (int L = 0; L < lw::kLodCount; ++L) t += lwGpu_[L].bytes;
        return t;
    }

    // ---- Frame ----
    void Resize(uint32_t w, uint32_t h);
    void BeginFrame(float clear[4], bool skipClear = false, bool skipDsvClear = false);
    void EndFrame(bool vsync);

    // ---- Accessors (DX12) ----
    ID3D12Device*               Device() const     { return device_.Get(); }
    ID3D12GraphicsCommandList*  CommandList() const{ return cmdList_.Get(); }
    ID3D12CommandQueue*         CommandQueue() const{ return cmdQueue_.Get(); }
    DXGI_FORMAT                 BackBufferFormat() const {
#if defined(VOXELTEST_XBOX)
        return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
#else
        return DXGI_FORMAT_R8G8B8A8_UNORM;
#endif
    }
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
    D3D12_GPU_DESCRIPTOR_HANDLE GodraySrv() const       { return imguiGodrayBlurGpu_; }
    D3D12_GPU_DESCRIPTOR_HANDLE GodrayMarkSrv() const   { return imguiGodrayMarkGpu_; }
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
    bool ForceReloadShaders();
private:
    void PollShaderHotReload();
    bool RecompileShaders();
    uint64_t lastShaderMtime_ = 0;
public:

private:
    bool CreateDeviceAndSwap(HWND hwnd, int adapterIdx);
    bool CreateRenderTargets();
    bool CreateM4();
    bool CreateVisTextures(uint32_t w, uint32_t h);
    bool UploadLwLod(const lw::World& w, int L);
    void WaitForGpu();
    void MoveToNextFrame();

    struct LwGpu
    {
        ComPtr<ID3D12Resource> chunkInfoSb;
        ComPtr<ID3D12Resource> paletteSb;
        ComPtr<ID3D12Resource> blockPosSb;
        ComPtr<ID3D12Resource> blockColSb;
        ComPtr<ID3D12Resource> blockVisSb;
        ComPtr<ID3D12Resource> blockAoSb; // 1 uint/voxel (24-bit AO), 8/block
        // Shader-visible SRV descriptor indices into lwSrvHeap_ (UINT32_MAX = none).
        uint32_t chunkInfoSrv = UINT32_MAX;
        uint32_t paletteSrv   = UINT32_MAX;
        uint32_t blockPosSrv  = UINT32_MAX;
        uint32_t blockColSrv  = UINT32_MAX;
        uint32_t blockVisSrv  = UINT32_MAX;
        uint32_t blockAoSrv   = UINT32_MAX;
        uint32_t slotCount    = 0;
        uint32_t blockCount   = 0;
        uint64_t bytes        = 0;
    };

    // On Xbox d3d12_xs.h decorates ID3D12Device with WaitFrameEventX etc.
    // Same interface name, different methods. Sample uses ID3D12Device.
    ComPtr<ID3D12Device>             device_;
#if defined(VOXELTEST_XBOX)
    D3D12XBOX_FRAME_PIPELINE_TOKEN   frameToken_ = D3D12XBOX_FRAME_PIPELINE_TOKEN_NULL;
#else
    ComPtr<IDXGIFactory6>            factory_;
    ComPtr<IDXGISwapChain3>          swap_;
#endif
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
    D3D12_CPU_DESCRIPTOR_HANDLE      imguiGodrayMarkCpu_ = {};
    D3D12_GPU_DESCRIPTOR_HANDLE      imguiGodrayMarkGpu_ = {};
    D3D12_CPU_DESCRIPTOR_HANDLE      imguiGodrayBlurCpu_ = {};
    D3D12_GPU_DESCRIPTOR_HANDLE      imguiGodrayBlurGpu_ = {};

    // Fence + per-frame sync.
    ComPtr<ID3D12Fence>              fence_;
    UINT64                           fenceValues_[kFrameCount] = {};
    HANDLE                           fenceEvent_ = nullptr;

    // Fence-tagged deferred release for streamed LW buffers + upload scratch.
    // A per-LOD stream upload no longer stalls the GPU: old buffers and the
    // upload command objects are parked here and freed once their copy fence
    // signals (checked each frame in CollectLwRetired).
    struct LwRetireBatch
    {
        ComPtr<ID3D12Fence>          fence;
        UINT64                       value = 0;
        // ID3D12Object (not IUnknown): Xbox D3D12 types don't derive from
        // IUnknown but do share ID3D12Object as a common base.
        std::vector<ComPtr<ID3D12Object>> objs;
    };
    std::vector<LwRetireBatch>       lwRetire_;
    void                             CollectLwRetired();

    // Prepared-but-not-yet-committed per-LOD upload (built on the loader thread).
    struct PendingLwUpload
    {
        LwGpu                                 gpu;     // new default resources
        lw::LODWorld                          meta;    // CPU chunk + cull metadata
                                                       // (pools empty) installed into
                                                       // lwWorld_ for the draw walk
        ComPtr<ID3D12CommandAllocator>        alloc;
        ComPtr<ID3D12GraphicsCommandList>     cmd;     // pre-recorded copy+barriers
        std::vector<ComPtr<ID3D12Resource>>   scratch; // upload heaps (retire after copy)
        bool                                  valid = false;
        bool                                  empty = false; // LOD had 0 chunks
    };
    PendingLwUpload                  pendingUpload_[lw::kLodCount];
    // Fence signalled on cmdQueue_ after each CommitLwLod for retiring old data.
    ComPtr<ID3D12Fence>              lwUploadFence_;
    UINT64                           lwUploadFenceVal_ = 0;

    std::unique_ptr<DirectX::GraphicsMemory> graphicsMemory_;

    ShaderCompiler                   shaderc_;

    // M3 — LW upload. Per-LOD GPU buffers + a shared shader-visible SRV heap
    // (sized for all LODs' structured buffers, with headroom for M4 combined +
    // worklist + visDepth/Color UAVs).
    static constexpr UINT             kLwSrvHeapSize = 512;
    ComPtr<ID3D12DescriptorHeap>      lwSrvHeap_;
    UINT                              lwSrvDescSize_ = 0;
    UINT                              lwSrvNextSlot_ = 0;
    LwGpu                             lwGpu_[lw::kLodCount];
    bool                              lwHasWorld_ = false;
    lw::World                         lwWorld_;

    // M4 — PointCS_Block compute rasterizer (LOD0-only minimum viable).
    // Vis textures (R32_UINT): atomic-min depth + ARGB color. Slots in
    // m4TexHeap_:  0=depthUav  1=colorUav  2=depthSrv  3=colorSrv.
    ComPtr<ID3D12Resource>            visDepthTex_;
    ComPtr<ID3D12Resource>            visColorTex_;
    ComPtr<ID3D12Resource>            visAoTex_;         // per-face AO at splat winner (R32_UINT)
    ComPtr<ID3D12Resource>            visColor2Tex_;     // dilated color (R32_UINT)
    ComPtr<ID3D12Resource>            visDepth2Tex_;     // dilated depth (R32_FLOAT, gNearZ/viewZ form)
    ComPtr<ID3D12Resource>            taaSceneTex_;      // resolve output (R11G11B10F)
    ComPtr<ID3D12Resource>            godrayTex_[3];     // [0]=mark, [1,2]=blur ping-pong (64x64 R16F)
    uint32_t                          godrayCurrIdx_ = 0;
    bool                              godrayHistCleared_ = false;
    // m4TexHeap slot layout (14 entries):
    //   0=visDepthUav   1=visColorUav   2=visDepthSrv   3=visColorSrv
    //   4=taaHistSrv[0] 5=taaHistSrv[1]
    //   6=visColor2Uav  7=visDepth2Uav
    //   8=visDepth2Srv  9=visColor2Srv
    //   10=taaSceneSrv  11=godrayMark0Srv  12=godrayBlur1Srv  13=godrayBlur2Srv
    ComPtr<ID3D12DescriptorHeap>      m4TexHeap_;
    ComPtr<ID3D12DescriptorHeap>      m4TexClearHeap_;
    UINT                              m4TexDescSize_ = 0;
    ComPtr<ID3D12RootSignature>       m4Pass1RootSig_;
    ComPtr<ID3D12RootSignature>       m4Pass2RootSig_;
    ComPtr<ID3D12RootSignature>       m4DilateRootSig_;
    ComPtr<ID3D12RootSignature>       m4ResolveRootSig_;
    ComPtr<ID3D12RootSignature>       m4TaaRootSig_;
    ComPtr<ID3D12RootSignature>       m4PostRootSig_;
    ComPtr<ID3D12RootSignature>       m4GodrayMarkRootSig_;
    ComPtr<ID3D12RootSignature>       m4GodrayBlurRootSig_;
    ComPtr<ID3D12RootSignature>       m4OctetRootSig_;   // OctetBillboards graphics
    ComPtr<ID3D12PipelineState>       m4OctetPso_;
    ComPtr<ID3D12PipelineState>       m4OctetGeoPso_;    // OctetGeo (real tris), same rootsig
    ComPtr<ID3D12PipelineState>       m4Pass1Pso_;
    ComPtr<ID3D12PipelineState>       m4Pass2Pso_;
    ComPtr<ID3D12PipelineState>       m4DilatePso_;
    ComPtr<ID3D12PipelineState>       m4ResolvePso_;
    ComPtr<ID3D12PipelineState>       m4TaaPso_;
    ComPtr<ID3D12PipelineState>       m4PostPso_;
    ComPtr<ID3D12PipelineState>       m4GodrayMarkPso_;
    ComPtr<ID3D12PipelineState>       m4GodrayBlurPso_;

    // TAA — ping-pong history RGBA8 RTs. Resolve writes to taaHist[curr]
    // sampling taaHist[prev]; blit then samples taaHist[curr] to backbuffer.
    ComPtr<ID3D12Resource>            taaHistTex_[2];
    ComPtr<ID3D12DescriptorHeap>      taaRtvHeap_; // 0,1=taaHist[0,1] 2=taaScene 3=godray[0] 4=godray[1] 5=godray[2]
    UINT                              taaRtvDescSize_ = 0;
    uint32_t                          taaCurrIdx_ = 0;
    uint32_t                          taaFrame_   = 0; // resets to 0 on world change / resize
    bool                              taaHistValid_[2] = {false, false};
    float                             taaPrevVP_[16] = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    uint32_t                          visTexW_ = 0;
    uint32_t                          visTexH_ = 0;

    HWND     hwnd_ = nullptr;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    UINT     frameIndex_ = 0;
    bool     tearingSupported_ = false;
};
