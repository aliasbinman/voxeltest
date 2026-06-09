#pragma once
#include "camera.h"
#include "lodworld.h"

#include <d3d11.h>
#include <dxgi1_2.h>
#include <vector>
#include <string>
#include <cstdint>
#include <wrl/client.h>

template <typename T>
using ComPtr = Microsoft::WRL::ComPtr<T>;

enum class ShadingMode : int
{
    Lit = 0,
    FlatColor = 1,
    Normals = 2,
    Ao = 3,
    LodViz = 4,
};

// Surviving render techs (LW path). Splat is the default; PolyAxis renders
// real cube faces post-dilate. Others kept for reference / port targets.
// Render technique selector. PointCS_Block + OctetBillboards are wired up.
// PolyAxis / Splat reserved for future octet-driven revivals.
enum class RenderTech : int
{
    PointCS_Block    = 0, // CS worklist Pass1 depth + Pass2 colour
    OctetBillboards  = 1, // billboard quad per occupied octet, PS ray-AABB
    PolyAxis         = 2, // reserved
    Splat            = 3, // reserved
};

enum class PointLighting : int
{
    Simple = 0,
    Complex = 1
};

enum class PointLod : int
{
    L0 = 0,
    L1 = 1,
    L2 = 2,
    L3 = 3,
    Auto = 4
};

struct DrawSceneParams
{
    ShadingMode mode = ShadingMode::Lit;
    int gridSize = 1;
    RenderTech tech    = RenderTech::PointCS_Block; // close-ring technique
    RenderTech techFar = RenderTech::PointCS_Block; // far-ring technique
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
    bool splatDilate2Pass = false;
    bool shadowBlur = false;
    bool lwShowBounds = false;
    bool cheapAO = false;        // top-down depth-based AO (multiplied in pass2)
    float aoFadeUnits = 16.0f;   // world units of soft falloff below top
    float aoPushTexels = 1.0f;   // lateral neighbour push distance (texels)
    float aoStrength = 1.0f;     // 0..1 multiplier on final AO darkening
    bool skipBackbufferClear = false; // post pass writes every pixel
    float godrayStrength = 0.55f;
    float godrayAngleDeg = 10.0f; // angular extent of texture; real sun = 0.5° (tiny), 10° = nice halo
    float godrayEmaAlpha = 0.15f; // 1 = no smoothing, lower = more temporal damping
    float godrayTint[3] = {1.00f, 0.85f, 0.45f};
    bool  godrayAniso = false;       // false = 24-tap line blur, true = SampleGrad anisotropic
    bool  godraySeparable = false;   // 2-pass sparse (5 strided taps + fill)
    int   godraySeparableStride = 6; // stride (pixels) between sparse taps
};

class Renderer
{
public:
    bool Init(HWND hwnd, int adapterIdx = -1);
    void Shutdown();

    static std::vector<std::string> EnumerateAdapters();

    bool UploadLwWorld(const lw::World& w);
    // Streaming entry points:
    //   PrepLwWorld    — stash world metadata (chunk AABBs / childId / cull arrays).
    //   UploadLwLodOnly — upload one LOD's GPU buffers (chunkInfo / palette /
    //                     blockPos / blockCol).
    // Use these when loading LODs incrementally; render starts once any LOD is up.
    void PrepLwWorld(const lw::World& w);
    bool UploadLwLodOnly(const lw::World& w, int L);

    // ---- GPU resource inspection (debug window) ----
    struct StreamPoolInfo
    {
        const char* name = "";
        uint64_t bytes      = 0; // total bytes the SB occupies
        uint32_t numElements = 0;
        uint32_t stride      = 0;
    };
    struct StreamLodInfo
    {
        StreamPoolInfo pools[5] = {}; // chunkInfo, palette, blockPos, blockCol, blockVis
    };
    StreamLodInfo GetStreamLodInfo(int L) const;
    // Call once after a batch of UploadLwLodOnly calls (a stream cycle's worth
    // of LOD updates). Rebuilds the combined-LOD GPU buffers if invalidated.
    bool FinalizeLwUploads();
    void ClearLwWorld();
    bool HasLwWorld() const
    {
        return lwHasWorld_;
    }
    void DrawLwScene(const Camera& cam, const DrawSceneParams& args);

    uint32_t LwSlotCount(int L) const
    {
        return (L >= 0 && L < lw::kLodCount) ? lwGpu_[L].slotCount : 0;
    }
    uint32_t LwPointCount(int /*L*/) const
    {
        return 0;
    }
    uint64_t LwBytes(int L) const
    {
        return (L >= 0 && L < lw::kLodCount) ? lwGpu_[L].bytes : 0;
    }
    uint64_t LwTotalBytes() const
    {
        uint64_t t = 0;
        for (int L = 0; L < lw::kLodCount; ++L)
            t += lwGpu_[L].bytes;
        return t;
    }

    void Resize(uint32_t w, uint32_t h);
    // skipClear:    swapchain RTV clear (psmain_post writes every pixel → safe).
    // skipDsvClear: main depth clear (only needed when a depth-reader runs:
    //               polyaxis, HW point CS_Block direct, lwShowBounds wireframe).
    void BeginFrame(float clear[4], bool skipClear = false, bool skipDsvClear = false);
    void EndFrame(bool vsync);

    ID3D11Device* Device() const
    {
        return device_.Get();
    }
    ID3D11DeviceContext* Context() const
    {
        return ctx_.Get();
    }

    uint32_t Width() const
    {
        return width_;
    }
    uint32_t Height() const
    {
        return height_;
    }
    uint64_t SplatRtBytes() const
    {
        return splatRtBytes_;
    }
    ID3D11ShaderResourceView* ShadowSrv() const
    {
        return shadowSrv_.Get();
    }
    ID3D11ShaderResourceView* ShadowFilledSrv() const
    {
        return shadowFilledSrv_.Get();
    }
    ID3D11ShaderResourceView* GodraySrv() const
    {
        return godraySrv_[godrayCurrIdx_].Get();
    }
    ID3D11ShaderResourceView* GodrayMarkSrv() const
    {
        return godraySrv_[0].Get();
    }
    uint32_t ShadowMapSize() const
    {
        return shadowSize_;
    }
    uint32_t LastDrawnCount() const
    {
        return lastDrawn_;
    }
    uint64_t LastPointCount() const
    {
        return lastPointCount_;
    }
    uint64_t LastTriCount() const
    {
        return lastTriCount_;
    }
    uint64_t LastPolyVoxelCount() const
    {
        return lastPolyVoxelCount_;
    }
    uint64_t LastSplatVoxelCount() const
    {
        return lastSplatVoxelCount_;
    }
    uint32_t LastSplatDrawCalls() const
    {
        return lastSplatDrawCalls_;
    }
    uint32_t LastPolyDrawCalls() const
    {
        return lastPolyDrawCalls_;
    }
    uint32_t LastFastDrawCalls() const
    {
        return lastFastDrawCalls_;
    }
    // PointCS_Block stats: total blocks dispatched across all LODs this frame.
    // Each block = up to 8 voxels (octet) → atomic upper bound = blocks * 8.
    // Pass1 + Pass2 share the same input set; this counts blocks once.
    uint64_t LastBlockTotal() const { return lastBlockTotal_; }
    uint32_t LastBlockDispatches() const { return lastBlockDispatches_; }
    uint64_t LastBlockTotalAt(int lod) const
    {
        if (lod < 0 || lod >= 5) return 0;
        return lastBlockTotalPerLod_[lod];
    }

private:
    bool CreateDeviceAndSwap(HWND hwnd, int adapterIdx);
    bool CreateRenderTargets();
    bool CreateShaders();
    bool CreatePipelineState();
    void TryHotReloadShaders();
    bool UploadLwLod(const lw::World& w, int L);
    void FillCbPerFrame(const Camera& cam,
                        const DrawSceneParams& args,
                        const hlslpp::float4x4& vp,
                        float jitterNdcX, float jitterNdcY,
                        const float sceneOrigin[3],
                        const float sceneSpan[3],
                        float colorizeClusters,
                        int gridSize,
                        const float* sunVPstore16OrNull,
                        bool sunShadowsOn,
                        uint32_t shadowMapSize);

    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> ctx_;
    ComPtr<IDXGISwapChain1> swap_;
    ComPtr<ID3D11RenderTargetView> rtv_;
    ComPtr<ID3D11DepthStencilView> dsv_;
    ComPtr<ID3D11Texture2D> depthTex_;

    // ---- Splat pipeline (shared LW splat + dilate + composite) ----
    ComPtr<ID3D11Texture2D> splatColorTex_;
    ComPtr<ID3D11RenderTargetView> splatColorRtv_;
    ComPtr<ID3D11ShaderResourceView> splatColorSrv_;
    ComPtr<ID3D11UnorderedAccessView> splatColorUav_; // PointCS_Block → splat path
    ComPtr<ID3D11Texture2D> splatMaskTex_; // R8_UINT 6-bit visMask per pixel
    ComPtr<ID3D11ShaderResourceView> splatMaskSrv_;
    ComPtr<ID3D11UnorderedAccessView> splatMaskUav_;
    ComPtr<ID3D11Texture2D> splatDepthTex_;
    ComPtr<ID3D11DepthStencilView> splatDsv_;
    ComPtr<ID3D11ShaderResourceView> splatDepthSrv_;
    ComPtr<ID3D11UnorderedAccessView> splatDepthUavF_; // R32F UAV of splatBlockDepthTex_
    ComPtr<ID3D11Texture2D> splatBlockDepthTex_;       // R32F, UAV+SRV for block-CS path
    ComPtr<ID3D11ShaderResourceView> splatBlockDepthSrv_;
    ComPtr<ID3D11DepthStencilState> dsAlwaysWrite_;
    ComPtr<ID3D11Texture2D> splatFinalTex_;
    ComPtr<ID3D11UnorderedAccessView> splatFinalUav_;
    ComPtr<ID3D11ShaderResourceView> splatFinalSrv_;
    ComPtr<ID3D11Texture2D> splatFinalDepthTex_;
    ComPtr<ID3D11UnorderedAccessView> splatFinalDepthUav_;
    ComPtr<ID3D11ShaderResourceView> splatFinalDepthSrv_;
    ComPtr<ID3D11Texture2D> splatFinal2Tex_;
    ComPtr<ID3D11UnorderedAccessView> splatFinal2Uav_;
    ComPtr<ID3D11ShaderResourceView> splatFinal2Srv_;
    ComPtr<ID3D11Texture2D> splatFinal2DepthTex_;
    ComPtr<ID3D11UnorderedAccessView> splatFinal2DepthUav_;
    ComPtr<ID3D11ShaderResourceView> splatFinal2DepthSrv_;
    ComPtr<ID3D11ComputeShader> csSplat_;
    ComPtr<ID3D11ComputeShader> csSplatFill_;
    ComPtr<ID3D11PixelShader> psSplatComposite_;

    ComPtr<ID3D11RasterizerState> rsSolid_;
    ComPtr<ID3D11RasterizerState> rsNoCull_;
    ComPtr<ID3D11DepthStencilState> dsTest_;
    ComPtr<ID3D11DepthStencilState> dsAlways_;
    ComPtr<ID3D11BlendState> bsAlphaOver_;

    // ---- Sun shadow ----
    ComPtr<ID3D11Texture2D> shadowTex_;
    ComPtr<ID3D11DepthStencilView> shadowDsv_;
    ComPtr<ID3D11ShaderResourceView> shadowSrv_;
    ComPtr<ID3D11SamplerState> shadowSamp_;
    ComPtr<ID3D11RasterizerState> rsShadowBack_;
    ComPtr<ID3D11RasterizerState> rsShadowFront_;
    ComPtr<ID3D11Texture2D> shadowFilledTex_;
    ComPtr<ID3D11UnorderedAccessView> shadowFilledUav_;
    ComPtr<ID3D11ShaderResourceView> shadowFilledSrv_;
    ComPtr<ID3D11ComputeShader> csShadowBlur_;
    bool lastShadowBlur_ = false;
    uint32_t shadowSize_ = 0;
    float shadowVP_[16] = {};
    float shadowBias_ = 0.0f;
    float shadowEnable_ = 0.0f;
    float lastSunDir_[3] = {0, 0, 0};
    uint32_t lastShadowSize_ = 0;
    bool lastShadowCullFront_ = false;
    int lastShadowLod_ = -2;
    bool shadowMapDirty_ = true;

    // ---- TAA + post ----
    ComPtr<ID3D11Texture2D> taaSceneTex_;
    ComPtr<ID3D11RenderTargetView> taaSceneRtv_;
    ComPtr<ID3D11ShaderResourceView> taaSceneSrv_;
    ComPtr<ID3D11Texture2D> taaHistTex_[2];
    ComPtr<ID3D11RenderTargetView> taaHistRtv_[2];
    ComPtr<ID3D11ShaderResourceView> taaHistSrv_[2];
    ComPtr<ID3D11ShaderResourceView> depthSrv_;
    ComPtr<ID3D11VertexShader> vsTaa_;
    ComPtr<ID3D11PixelShader> psTaa_;
    ComPtr<ID3D11PixelShader> psPost_;
    ComPtr<ID3D11VertexShader> vsBlit_;
    ComPtr<ID3D11SamplerState> linearClampSampler_;

    // ---- God rays (64x64 R8) ----
    // [0]   = mark output (transient each frame)
    // [1,2] = ping-pong blur+EMA output (sampled by post; previous frame's
    //         blend used by next frame's blur via EMA history input)
    // [0] = mark, [1,2] = ping-pong blur+EMA, [3] = separable pass-1 intermediate.
    ComPtr<ID3D11Texture2D> godrayTex_[4];
    ComPtr<ID3D11RenderTargetView> godrayRtv_[4];
    ComPtr<ID3D11ShaderResourceView> godraySrv_[4];
    ComPtr<ID3D11PixelShader> psGodrayMark_;
    ComPtr<ID3D11PixelShader> psGodrayBlur_;
    ComPtr<ID3D11PixelShader> psGodrayBlurAniso_;   // SampleGrad anisotropic variant
    ComPtr<ID3D11PixelShader> psGodrayBlurSparse_;  // separable pass 1: 5 strided taps
    ComPtr<ID3D11PixelShader> psGodrayBlurFill_;    // separable pass 2: fills gaps
    ComPtr<ID3D11SamplerState> anisoSamp_;          // wrap, max-aniso 16
    ComPtr<ID3D11Buffer> cbGodray_;
    uint32_t godrayCurrIdx_ = 1; // most recent blur write target
    uint32_t taaHistIdx_ = 0;
    uint32_t taaFrame_ = 0;
    bool taaHistValid_[2] = {false, false};
    float taaPrevVP_[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float lastClear_[4] = {0, 0, 0, 1};

    // ---- LW (lodworld) GPU resources ----
    struct LwGpu
    {
        ComPtr<ID3D11Buffer> chunkInfoSb;
        ComPtr<ID3D11ShaderResourceView> chunkInfoSrv;
        ComPtr<ID3D11Buffer> paletteSb;
        ComPtr<ID3D11ShaderResourceView> paletteSrv;
        ComPtr<ID3D11Buffer> blockPosSb; // BlockPos pool (PointCS_Block, 4B/block: pos+occ)
        ComPtr<ID3D11ShaderResourceView> blockPosSrv;
        ComPtr<ID3D11Buffer> blockColSb; // BlockCol pool (8B/block: palIdx[8])
        ComPtr<ID3D11ShaderResourceView> blockColSrv;
        ComPtr<ID3D11Buffer> blockVisSb; // BlockVis pool (8B/block: 6-bit visMask[8])
        ComPtr<ID3D11ShaderResourceView> blockVisSrv;
        uint32_t slotCount = 0;
        uint32_t blockCount = 0;
        uint64_t bytes = 0;
    };
    LwGpu lwGpu_[lw::kLodCount];

    // Combined LwGpu: all per-LOD pools concatenated into single SRVs so the
    // worklist pass1/pass2 can run as ONE Dispatch across all LODs (eliminating
    // per-LOD UAV barriers). Rebuilt after every UploadLwLodOnly.
    struct LwGpuCombined
    {
        ComPtr<ID3D11Buffer> chunkInfoSb;
        ComPtr<ID3D11ShaderResourceView> chunkInfoSrv;
        ComPtr<ID3D11Buffer> paletteSb;
        ComPtr<ID3D11ShaderResourceView> paletteSrv;
        ComPtr<ID3D11Buffer> blockPosSb;
        ComPtr<ID3D11ShaderResourceView> blockPosSrv;
        ComPtr<ID3D11Buffer> blockColSb;
        ComPtr<ID3D11ShaderResourceView> blockColSrv;
        uint32_t lodSlotBase[lw::kLodCount]   = {};
        uint32_t lodBlockBase[lw::kLodCount]  = {};
        uint32_t lodPaletteBase[lw::kLodCount]= {};
        bool     valid = false;
    };
    LwGpuCombined lwGpuC_;
    bool RebuildCombinedLwGpu();
    bool lwHasWorld_ = false;
    lw::World lwWorld_;

    ComPtr<ID3D11VertexShader> vsLwPoints_;
    ComPtr<ID3D11PixelShader> psLwSplatAlbedo_;
    ComPtr<ID3D11PixelShader> psLwDebug_;
    ComPtr<ID3D11PixelShader> psLwLodViz_;
    ComPtr<ID3D11VertexShader> vsLwPolyAxis_;
    ComPtr<ID3D11PixelShader> psLwPolyAxisLit_;
    ComPtr<ID3D11VertexShader> vsLwBounds_;
    ComPtr<ID3D11PixelShader> psLwBounds_;
    ComPtr<ID3D11Buffer> cbLwFrame_;
    ComPtr<ID3D11Buffer> cbLwLod_;
    ComPtr<ID3D11Buffer> cbLwBounds_;
    bool tearingSupported_ = false;

    // ---- Compute rasterizer (PointCS_Block two-pass worklist) ----
    // Pass1 writes R32 depth (atomic). Pass2 reads depth, writes R16 colour.
    ComPtr<ID3D11Texture2D> visDepthTex_;
    ComPtr<ID3D11UnorderedAccessView> visDepthUav_;
    ComPtr<ID3D11ShaderResourceView> visDepthSrv_;
    ComPtr<ID3D11Texture2D> visColorTex_;
    ComPtr<ID3D11UnorderedAccessView> visColorUav_;
    ComPtr<ID3D11ShaderResourceView> visColorSrv_;
    ComPtr<ID3D11Buffer> cbLwCS_;
    ComPtr<ID3D11ComputeShader> csLwBlockDepthWorklist_; // pass1
    ComPtr<ID3D11ComputeShader> csLwBlockColorWorklist_; // pass2 → R16 RGB565
    ComPtr<ID3D11ComputeShader> csLwBlockSplatWorklist_; // pass2 → splat-format targets
    ComPtr<ID3D11VertexShader>  vsLwOctetBillboard_;     // OctetBillboards: 6 verts per occupied block
    ComPtr<ID3D11PixelShader>   psLwOctetBillboard_;     // PS: ray-vs-8 child AABB, shade inline
    // ---- Cheap top-down AO ----
    // R32_UINT 2048x2048 mapped over the scene X-Z AABB. Each occupied voxel
    // does InterlockedMax(worldY) at its (worldX, worldZ) UV → texture stores
    // the maximum-Y opaque voxel per column. AO at shading = function of
    // (maxY - voxelY): top of column → fully lit, deep below → darker.
    ComPtr<ID3D11Texture2D> aoTopDownTex_;
    ComPtr<ID3D11UnorderedAccessView> aoTopDownUav_;
    ComPtr<ID3D11ShaderResourceView> aoTopDownSrv_;
    ComPtr<ID3D11Texture2D> aoOcclTex_;       // R8_UNORM HBAO sweep result
    ComPtr<ID3D11UnorderedAccessView> aoOcclUav_;
    ComPtr<ID3D11ShaderResourceView> aoOcclSrv_;
    ComPtr<ID3D11ComputeShader> csAoTopDownBuild_;
    ComPtr<ID3D11ComputeShader> csAoHbaoFilter_;
    ComPtr<ID3D11Buffer> cbLwAo_; // b4
    bool aoDirty_ = true; // rebuild on next frame
    uint32_t aoTexSize_ = 0; // 0 = not yet created
    static constexpr uint32_t kAoTexSizeMax = 2048;
    bool EnsureAoTextures(uint32_t size);
    ComPtr<ID3D11VertexShader>  vsLwBlockPoint_;      // A/B: HW point primitive rasterizer
    ComPtr<ID3D11PixelShader>   psLwBlockPoint_;
    ComPtr<ID3D11PixelShader>   psLwBlockPointSplat_; // writes splat MRT for csSplat
    // Per-LOD worklist buffer (uint4 per item: slot, blockBaseGlobal, count, firstThread).
    ComPtr<ID3D11Buffer> worklistSb_;
    ComPtr<ID3D11ShaderResourceView> worklistSrv_;
    uint32_t worklistCapacity_ = 0;
    ComPtr<ID3D11VertexShader> vsLwResolve_;
    ComPtr<ID3D11PixelShader> psLwResolveTwoPass_;

    ComPtr<ID3D11Buffer> cbPerFrame_;

    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint64_t splatRtBytes_ = 0;
    uint32_t lastDrawn_ = 0;
    uint64_t lastPointCount_ = 0;
    uint64_t lastTriCount_ = 0;
    uint64_t lastPolyVoxelCount_ = 0;
    uint64_t lastSplatVoxelCount_ = 0;
    uint32_t lastSplatDrawCalls_ = 0;
    uint32_t lastPolyDrawCalls_ = 0;
    uint32_t lastFastDrawCalls_ = 0;
    uint64_t lastBlockTotal_ = 0;
    uint32_t lastBlockDispatches_ = 0;
    uint64_t lastBlockTotalPerLod_[5] = {};
    uint64_t shaderMtime_ = 0;
    bool shaderReloading_ = false;

public:
    enum class ReloadStatus
    {
        None,
        Success,
        Failed
    };
    ReloadStatus lastReloadStatus_ = ReloadStatus::None;
    uint32_t reloadCount_ = 0;
    bool ForceReloadShaders();
};
