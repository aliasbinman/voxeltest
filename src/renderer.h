#pragma once
#include "camera.h"
#include "lodworld.h"

#include <d3d11.h>
#include <dxgi1_2.h>
#include <vector>
#include <string>
#include <cstdint>
#include <wrl/client.h>

template<typename T> using ComPtr = Microsoft::WRL::ComPtr<T>;

enum class ShadingMode : int {
    Lit       = 0,
    FlatColor = 1,
    Normals   = 2,
    Ao        = 3,
    LodViz    = 4,
};

// Surviving render techs (LW path). Splat is the default; PolyAxis renders
// real cube faces post-dilate. Others kept for reference / port targets.
enum class RenderTech : int {
    Splat             = 9,
    PolyAxis          = 11,
    PolyAxisInstanced = 12,
    HexSprite         = 3,
};

enum class PointLighting : int { Simple = 0, Complex = 1 };

enum class PointLod : int { L0 = 0, L1 = 1, L2 = 2, L3 = 3, Auto = 4 };

struct DrawSceneParams {
    ShadingMode    mode             = ShadingMode::Lit;
    int            gridSize         = 1;
    RenderTech     tech             = RenderTech::Splat;
    RenderTech     techFar          = RenderTech::Splat;
    bool           closeEnabled     = true;
    bool           farEnabled       = true;
    PointLighting  pointLight       = PointLighting::Complex;
    PointLod       pointLod         = PointLod::Auto;
    float          pointLodScale    = 1.0f;
    bool           splatFilter      = true;
    float          fogColor[3]      = { 0.55f, 0.60f, 0.70f };
    float          fogDensity       = 0.0f;
    float          heightFogDensity = 0.0f;
    float          heightFogFalloff = 0.05f;
    float          heightFogStart   = 0.0f;
    int            splatRadius      = 3;
    bool           taa              = true;
    float          sunDir[3]        = { 0.4f, 0.8f, 0.2f };
    float          sunIntensity     = 1.0f;
    float          exposure         = 1.0f;
    float          roughness        = 0.6f;
    bool           sunShadows       = false;
    int            shadowCascades   = 1;
    int            shadowMapSize    = 2048;
    float          shadowBias       = 0.001f;
    bool           shadowCullFront  = false;
    bool           shadowForceRebuild = false;
    int            shadowLod        = -1;
    bool           splatDilate2Pass = false;
    bool           shadowBlur       = false;
    bool           lwShowBounds     = false;
    bool           lwPolyAxis       = false;
    float          godrayStrength   = 0.55f;
    float          godrayAngleDeg   = 10.0f;  // angular extent of texture; real sun = 0.5° (tiny), 10° = nice halo
    float          godrayEmaAlpha   = 0.15f;  // 1 = no smoothing, lower = more temporal damping
    float          godrayTint[3]    = { 1.00f, 0.85f, 0.45f };
};

class Renderer {
public:
    bool Init(HWND hwnd, int adapterIdx = -1);
    void Shutdown();

    static std::vector<std::string> EnumerateAdapters();

    bool UploadLwWorld(const lw::World& w);
    // Streaming entry points:
    //   PrepLwWorld    — stash world metadata (chunk AABBs / childId / cull arrays).
    //   UploadLwLodOnly — upload one LOD's GPU buffers (points/chunkInfo/palette),
    //                     drop its CPU pointPool, refresh identity IB.
    // Use these when loading LODs incrementally; render starts once any LOD is up.
    void PrepLwWorld(const lw::World& w);
    bool UploadLwLodOnly(const lw::World& w, int L);
    void ClearLwWorld();
    bool HasLwWorld() const { return lwHasWorld_; }
    void DrawLwScene(const Camera& cam, const DrawSceneParams& args);

    uint32_t LwSlotCount (int L) const { return (L >= 0 && L < lw::kLodCount) ? lwGpu_[L].slotCount  : 0; }
    uint32_t LwPointCount(int L) const { return (L >= 0 && L < lw::kLodCount) ? lwGpu_[L].pointCount : 0; }
    uint64_t LwBytes     (int L) const { return (L >= 0 && L < lw::kLodCount) ? lwGpu_[L].bytes      : 0; }
    uint64_t LwTotalBytes() const {
        uint64_t t = 0;
        for (int L = 0; L < lw::kLodCount; ++L) t += lwGpu_[L].bytes;
        return t;
    }

    void Resize(uint32_t w, uint32_t h);
    void BeginFrame(float clear[4]);
    void EndFrame(bool vsync);

    ID3D11Device*        Device()  const { return device_.Get(); }
    ID3D11DeviceContext* Context() const { return ctx_.Get(); }

    uint32_t Width()  const { return width_; }
    uint32_t Height() const { return height_; }
    uint64_t SplatRtBytes() const { return splatRtBytes_; }
    ID3D11ShaderResourceView* ShadowSrv()        const { return shadowSrv_.Get(); }
    ID3D11ShaderResourceView* ShadowFilledSrv()  const { return shadowFilledSrv_.Get(); }
    ID3D11ShaderResourceView* GodraySrv()        const { return godraySrv_[godrayCurrIdx_].Get(); }
    ID3D11ShaderResourceView* GodrayMarkSrv()    const { return godraySrv_[0].Get(); }
    uint32_t                  ShadowMapSize()    const { return shadowSize_; }
    uint32_t LastDrawnCount() const { return lastDrawn_; }
    uint64_t LastPointCount() const { return lastPointCount_; }
    uint64_t LastTriCount()   const { return lastTriCount_; }
    uint64_t LastPolyVoxelCount() const { return lastPolyVoxelCount_; }
    uint64_t LastSplatVoxelCount() const { return lastSplatVoxelCount_; }
    uint32_t LastSplatDrawCalls() const { return lastSplatDrawCalls_; }
    uint32_t LastPolyDrawCalls()  const { return lastPolyDrawCalls_;  }
    uint32_t LastFastDrawCalls()  const { return lastFastDrawCalls_;  }

private:
    bool CreateDeviceAndSwap(HWND hwnd, int adapterIdx);
    bool CreateRenderTargets();
    bool CreateShaders();
    bool CreatePipelineState();
    void TryHotReloadShaders();
    bool UploadLwLod(const lw::World& w, int L);
    bool RebuildLwIdentityIb();
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

    ComPtr<ID3D11Device>           device_;
    ComPtr<ID3D11DeviceContext>    ctx_;
    ComPtr<IDXGISwapChain1>        swap_;
    ComPtr<ID3D11RenderTargetView> rtv_;
    ComPtr<ID3D11DepthStencilView> dsv_;
    ComPtr<ID3D11Texture2D>        depthTex_;

    // ---- Splat pipeline (shared LW splat + dilate + composite) ----
    ComPtr<ID3D11Texture2D>            splatColorTex_;
    ComPtr<ID3D11RenderTargetView>     splatColorRtv_;
    ComPtr<ID3D11ShaderResourceView>   splatColorSrv_;
    ComPtr<ID3D11Texture2D>            splatDepthTex_;
    ComPtr<ID3D11DepthStencilView>     splatDsv_;
    ComPtr<ID3D11ShaderResourceView>   splatDepthSrv_;
    ComPtr<ID3D11DepthStencilState>    dsAlwaysWrite_;
    ComPtr<ID3D11Texture2D>            splatFinalTex_;
    ComPtr<ID3D11UnorderedAccessView>  splatFinalUav_;
    ComPtr<ID3D11ShaderResourceView>   splatFinalSrv_;
    ComPtr<ID3D11Texture2D>            splatFinalDepthTex_;
    ComPtr<ID3D11UnorderedAccessView>  splatFinalDepthUav_;
    ComPtr<ID3D11ShaderResourceView>   splatFinalDepthSrv_;
    ComPtr<ID3D11Texture2D>            splatFinal2Tex_;
    ComPtr<ID3D11UnorderedAccessView>  splatFinal2Uav_;
    ComPtr<ID3D11ShaderResourceView>   splatFinal2Srv_;
    ComPtr<ID3D11Texture2D>            splatFinal2DepthTex_;
    ComPtr<ID3D11UnorderedAccessView>  splatFinal2DepthUav_;
    ComPtr<ID3D11ShaderResourceView>   splatFinal2DepthSrv_;
    ComPtr<ID3D11ComputeShader>        csSplat_;
    ComPtr<ID3D11ComputeShader>        csSplatFill_;
    ComPtr<ID3D11Texture2D>            splatMaskTex_;
    ComPtr<ID3D11RenderTargetView>     splatMaskRtv_;
    ComPtr<ID3D11ShaderResourceView>   splatMaskSrv_;
    ComPtr<ID3D11PixelShader>          psSplatComposite_;

    ComPtr<ID3D11RasterizerState>      rsSolid_;
    ComPtr<ID3D11RasterizerState>      rsNoCull_;
    ComPtr<ID3D11DepthStencilState>    dsTest_;
    ComPtr<ID3D11DepthStencilState>    dsAlways_;
    ComPtr<ID3D11BlendState>           bsAlphaOver_;

    // ---- Sun shadow ----
    ComPtr<ID3D11Texture2D>            shadowTex_;
    ComPtr<ID3D11DepthStencilView>     shadowDsv_;
    ComPtr<ID3D11ShaderResourceView>   shadowSrv_;
    ComPtr<ID3D11SamplerState>         shadowSamp_;
    ComPtr<ID3D11RasterizerState>      rsShadowBack_;
    ComPtr<ID3D11RasterizerState>      rsShadowFront_;
    ComPtr<ID3D11Texture2D>            shadowFilledTex_;
    ComPtr<ID3D11UnorderedAccessView>  shadowFilledUav_;
    ComPtr<ID3D11ShaderResourceView>   shadowFilledSrv_;
    ComPtr<ID3D11ComputeShader>        csShadowBlur_;
    bool                               lastShadowBlur_ = false;
    uint32_t                           shadowSize_ = 0;
    float                              shadowVP_[16]  = {};
    float                              shadowBias_    = 0.0f;
    float                              shadowEnable_  = 0.0f;
    float                              lastSunDir_[3] = { 0, 0, 0 };
    uint32_t                           lastShadowSize_ = 0;
    bool                               lastShadowCullFront_ = false;
    int                                lastShadowLod_       = -2;
    bool                               shadowMapDirty_ = true;

    // ---- TAA + post ----
    ComPtr<ID3D11Texture2D>          taaSceneTex_;
    ComPtr<ID3D11RenderTargetView>   taaSceneRtv_;
    ComPtr<ID3D11ShaderResourceView> taaSceneSrv_;
    ComPtr<ID3D11Texture2D>          taaHistTex_[2];
    ComPtr<ID3D11RenderTargetView>   taaHistRtv_[2];
    ComPtr<ID3D11ShaderResourceView> taaHistSrv_[2];
    ComPtr<ID3D11ShaderResourceView> depthSrv_;
    ComPtr<ID3D11VertexShader>       vsTaa_;
    ComPtr<ID3D11PixelShader>        psTaa_;
    ComPtr<ID3D11PixelShader>        psPost_;
    ComPtr<ID3D11VertexShader>       vsBlit_;
    ComPtr<ID3D11SamplerState>       linearClampSampler_;

    // ---- God rays (64x64 R8) ----
    // [0]   = mark output (transient each frame)
    // [1,2] = ping-pong blur+EMA output (sampled by post; previous frame's
    //         blend used by next frame's blur via EMA history input)
    ComPtr<ID3D11Texture2D>          godrayTex_[3];
    ComPtr<ID3D11RenderTargetView>   godrayRtv_[3];
    ComPtr<ID3D11ShaderResourceView> godraySrv_[3];
    ComPtr<ID3D11PixelShader>        psGodrayMark_;
    ComPtr<ID3D11PixelShader>        psGodrayBlur_;
    ComPtr<ID3D11Buffer>             cbGodray_;
    uint32_t                         godrayCurrIdx_ = 1;   // most recent blur write target
    uint32_t taaHistIdx_ = 0;
    uint32_t taaFrame_   = 0;
    bool     taaHistValid_[2] = { false, false };
    float    taaPrevVP_[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
    float    lastClear_[4] = { 0, 0, 0, 1 };

    // ---- LW (lodworld) GPU resources ----
    struct LwGpu {
        ComPtr<ID3D11Buffer>             pointSb;
        ComPtr<ID3D11ShaderResourceView> pointSrv;
        ComPtr<ID3D11Buffer>             chunkInfoSb;
        ComPtr<ID3D11ShaderResourceView> chunkInfoSrv;
        ComPtr<ID3D11Buffer>             paletteSb;
        ComPtr<ID3D11ShaderResourceView> paletteSrv;
        uint32_t slotCount = 0;
        uint32_t pointCount = 0;
        uint64_t bytes = 0;
    };
    LwGpu lwGpu_[lw::kLodCount];
    bool      lwHasWorld_ = false;
    lw::World lwWorld_;

    ComPtr<ID3D11VertexShader> vsLwPoints_;
    ComPtr<ID3D11PixelShader>  psLwSplatAlbedo_;
    ComPtr<ID3D11PixelShader>  psLwDebug_;
    ComPtr<ID3D11PixelShader>  psLwLodViz_;
    ComPtr<ID3D11VertexShader> vsLwPolyAxis_;
    ComPtr<ID3D11PixelShader>  psLwPolyAxisLit_;
    ComPtr<ID3D11VertexShader> vsLwBounds_;
    ComPtr<ID3D11PixelShader>  psLwBounds_;
    ComPtr<ID3D11Buffer>       cbLwFrame_;
    ComPtr<ID3D11Buffer>       cbLwLod_;
    ComPtr<ID3D11Buffer>       cbLwBounds_;
    bool                       tearingSupported_ = false;
    ComPtr<ID3D11Buffer>       lwIdentityIb_;
    uint32_t                   lwIdentityIbCount_ = 0;

    ComPtr<ID3D11Buffer> cbPerFrame_;

    uint32_t width_  = 0;
    uint32_t height_ = 0;
    uint64_t splatRtBytes_ = 0;
    uint32_t lastDrawn_ = 0;
    uint64_t lastPointCount_ = 0;
    uint64_t lastTriCount_   = 0;
    uint64_t lastPolyVoxelCount_  = 0;
    uint64_t lastSplatVoxelCount_ = 0;
    uint32_t lastSplatDrawCalls_  = 0;
    uint32_t lastPolyDrawCalls_   = 0;
    uint32_t lastFastDrawCalls_   = 0;
    uint64_t shaderMtime_ = 0;
    bool     shaderReloading_ = false;
public:
    enum class ReloadStatus { None, Success, Failed };
    ReloadStatus lastReloadStatus_ = ReloadStatus::None;
    uint32_t     reloadCount_      = 0;
    bool         ForceReloadShaders();
};
