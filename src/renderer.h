#pragma once
#include "mesh.h"
#include "mesh_loader.h"
#include "camera.h"

#include <d3d11.h>
#include <dxgi1_2.h>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <wrl/client.h>

template<typename T> using ComPtr = Microsoft::WRL::ComPtr<T>;

struct GpuSubMesh {
    uint32_t firstIndex;
    uint32_t indexCount;
    int32_t  baseVertex;
    uint32_t pointFirst;
    uint32_t pointCount;
    uint32_t pointFirstL1, pointCountL1;
    uint32_t pointFirstL2, pointCountL2;
    uint32_t pointFirstL3, pointCountL3;
    float    aabbMin[3];
    float    aabbMax[3];
    float    chunkBase[3];
};

enum class ShadingMode : int {
    Lit = 0,
    FlatColor = 1,
    Normals = 2,
    Ao = 3,
};

// Render path. Independent of which data is bound (that's DataSet).
enum class RenderTech : int {
    None         = -1,  // sentinel for "no Far tech => use Close everywhere"
    PolygonBased = 0,
    Points       = 1,
    Hybrid       = 2,   // points when voxel projects to <1 px, polygons otherwise
    HexSprite    = 3,   // 7-vertex hexagon per voxel (closest corner + silhouette)
    PointCS      = 4,   // compute shader software-rasterized points
    PolyVID      = 5,   // 1 voxel = 1 instance, VS expands cube via SV_VertexID
    Billboard    = 6,   // screen-space quad per voxel; PS ray-cube intersects
    BillboardTri = 7,   // billboard but a single bounding triangle per voxel
    Hybrid2      = 8,   // PolyVID up close, BillboardTri at distance
    Splat        = 9,   // points -> color RT (alpha=size); CS sphere reconstruct
    SplatHybrid  = 10,  // polygons up close, splats far (renders into splat RT)
    PolyAxis          = 11,  // pure-math instanced cube; VS picks the 3 camera-facing axis faces
    PolyAxisInstanced = 12,  // same math as PolyAxis but via DrawInstanced(18, count)
};

// Which baked dataset to draw from. Determines the polygon source (for
// PolygonBased / Hybrid / SplatHybrid close path) AND the point source (for
// Points / Splat / PointCS / Hybrid points branch).
//   Full    -> per-chunk voxel polys + voxel-center L0 points (rungholt.vox)
//   Merged  -> single MSH1 mesh + voxel-center L0 points       (rungholt.vox)
//   Reduced -> per-chunk atlas mesh + face-aligned L0 points    (rungholt_culled.vox)
enum class DataSet : int {
    Full    = 0,
    Merged  = 1,
    Reduced = 2,
};

enum class PointLighting : int {
    Simple  = 0,        // cam-to-voxel normal, triplanar ambient + ndotl
    Complex = 1,        // visMask 3-face weighted avg (proper but heavier)
};

enum class PointLod : int {
    L0   = 0,           // 1 point per voxel
    L1   = 1,           // 1 point per 2x2x2 (avg color)
    L2   = 2,           // 1 point per 4x4x4 (avg color)
    L3   = 3,           // 1 point per 8x8x8 (avg color)
    Auto = 4,           // pick per-chunk by distance
};

struct DrawSceneParams {
    ShadingMode    mode             = ShadingMode::Lit;
    int            gridSize         = 1;
    RenderTech     techClose        = RenderTech::PolygonBased;
    RenderTech     techFar          = RenderTech::None;
    DataSet        dataset          = DataSet::Reduced;
    bool           showChunkBounds  = false;
    bool           zPrepass         = false;
    PointLighting  pointLight       = PointLighting::Complex;
    PointLod       pointLod         = PointLod::Auto;
    float          pointLodScale    = 1.0f;
    bool           splatFilter      = true;
    float          fogColor[3]      = { 0.55f, 0.60f, 0.70f };
    float          fogDensity       = 0.0f;
    float          heightFogDensity = 0.0f;
    float          heightFogFalloff = 0.05f;
    float          heightFogStart   = 0.0f;
    float          hybridThreshold  = 1.0f;
    bool           wireframe        = false;
    int            splatRadius      = 3;
    bool           taa              = true;
    float          sunDir[3]        = { 0.4f, 0.8f, 0.2f };
    float          sunIntensity     = 1.0f;          // linear (2^EV)
    float          exposure         = 1.0f;          // linear (2^EV)
    float          roughness        = 0.6f;
    bool           sunShadows       = false;
    bool           colorizeClusters = false;
    bool           closeEnabled     = true;
    bool           farEnabled       = true;
};

class Renderer {
public:
    bool Init(HWND hwnd);
    void Shutdown();
    void Resize(uint32_t w, uint32_t h);
    void SetMsaa(uint32_t samples);
    uint32_t Msaa() const { return msaaSamples_; }
    void UploadScene(const Scene& scene);
    void UploadMergedMesh(const MergedMesh& mesh);
    void UploadAtlasMesh(const AtlasMesh& mesh);
    void BeginFrame(float clear[4]);
    void DrawScene(const Camera& cam, const DrawSceneParams& args);
    void EndFrame(bool vsync);

    ID3D11Device*        Device()  const { return device_.Get(); }
    ID3D11DeviceContext* Context() const { return ctx_.Get(); }

    uint32_t Width()  const { return width_; }
    uint32_t Height() const { return height_; }
    uint64_t TotalTriangles() const { return totalTriangles_; }
    uint64_t TotalVertices()  const { return totalVertices_; }
    uint64_t VbBytes()        const { return vbBytes_; }
    uint64_t IbBytes()        const { return ibBytes_; }
    uint64_t PointBytes()     const { return pointBytes_; }
    uint64_t PointCountL0()   const { return pointCountL0_; }
    uint64_t PointCountL1()   const { return pointCountL1_; }
    uint64_t PointCountL2()   const { return pointCountL2_; }
    uint64_t PointCountL3()   const { return pointCountL3_; }
    uint64_t MergedVbBytes()  const { return mergedVbBytes_; }
    uint64_t MergedIbBytes()  const { return mergedIbBytes_; }
    uint64_t AtlasVbBytes()   const { return atlasVbBytes_; }
    uint64_t AtlasIbBytes()   const { return atlasIbBytes_; }
    uint64_t AtlasTexBytes()  const { return atlasTexBytes_; }
    const std::vector<std::pair<uint32_t, uint64_t>>& ColorHistogram() const { return colorHistogram_; }
    double   ColorEntropyBits() const { return colorEntropyBits_; }
    double   ColorHuffmanBits() const { return colorHuffmanBits_; }
    uint32_t ColorPaletteBits() const { return colorPaletteBits_; }
    uint64_t CompRawBytes()        const { return compRawBytes_; }
    uint64_t CompPosBytes()        const { return compPosBytes_; }
    uint64_t CompMaskBytes()       const { return compMaskBytes_; }
    uint64_t CompAoBytes()         const { return compAoBytes_; }
    uint64_t CompPaletteBytes()    const { return compPaletteBytes_; }
    uint64_t CompColorPalIdxBytes()const { return compColorPalIdxBytes_; }
    uint64_t CompColorHuffBytes()  const { return compColorHuffBytes_; }
    uint32_t CompPosBitsPerAxis()  const { return compPosBitsPerAxis_; }
    uint32_t CompChunkDim()        const { return compChunkDim_; }
    uint64_t CompSubclusterPosBytes() const { return compSubclusterPosBytes_; }
    uint32_t CompSubclusterDim()      const { return compSubclusterDim_; }
    uint64_t CompLz4PosBytes()        const { return compLz4PosBytes_; }
    uint64_t CompLz4MaskBytes()       const { return compLz4MaskBytes_; }
    uint64_t CompLz4AoBytes()         const { return compLz4AoBytes_; }
    uint64_t CompLz4ColorPalBytes()   const { return compLz4ColorPalBytes_; }
    uint64_t CompLz4TotalBytes()      const { return compLz4TotalBytes_; }
    size_t   DrawCount()      const { return subs_.size(); }
    uint32_t LastDrawnCount() const { return lastDrawn_; }
    uint64_t LastDrawnTris()  const { return lastDrawnTris_; }
    uint64_t LastPolyTris()   const { return lastPolyTris_; }
    uint64_t LastPolyVerts()  const { return lastPolyVerts_; }
    uint64_t LastPointCount() const { return lastPointCount_; }

private:
    bool CreateDeviceAndSwap(HWND hwnd);
    bool CreateRenderTargets();
    bool CreateShaders();
    bool CreatePipelineState();

    ComPtr<ID3D11Device>         device_;
    ComPtr<ID3D11DeviceContext>  ctx_;
    ComPtr<IDXGISwapChain1>      swap_;
    ComPtr<ID3D11RenderTargetView> rtv_;
    ComPtr<ID3D11DepthStencilView> dsv_;
    ComPtr<ID3D11Texture2D>      depthTex_;
    ComPtr<ID3D11Texture2D>      msaaColorTex_;
    ComPtr<ID3D11RenderTargetView> msaaRtv_;
    uint32_t                     msaaSamples_ = 1;
    uint32_t                     msaaQuality_ = 0;

    ComPtr<ID3D11VertexShader>   vs_;
    ComPtr<ID3D11PixelShader>    ps_;
    ComPtr<ID3D11VertexShader>   vsPoints_;
    ComPtr<ID3D11PixelShader>    psPoints_;          // Complex
    ComPtr<ID3D11PixelShader>    psPointsSimple_;    // Simple
    ComPtr<ID3D11VertexShader>   vsHex_;
    ComPtr<ID3D11PixelShader>    psHex_;
    ComPtr<ID3D11VertexShader>   vsPolyVid_;
    ComPtr<ID3D11VertexShader>   vsPolyAxis_;
    ComPtr<ID3D11VertexShader>   vsPolyAxisInstanced_;
    ComPtr<ID3D11PixelShader>    psPolyVid_;
    ComPtr<ID3D11InputLayout>    inputLayoutPolyVid_;
    ComPtr<ID3D11Buffer>         polyVidIb_;
    ComPtr<ID3D11VertexShader>   vsBillboard_;
    ComPtr<ID3D11PixelShader>    psBillboard_;
    ComPtr<ID3D11Buffer>         billboardIb_;
    ComPtr<ID3D11VertexShader>   vsBillboardTri_;

    // Merged mesh path (greedy-merged single VB/IB).
    ComPtr<ID3D11Buffer>         mergedVb_;
    ComPtr<ID3D11Buffer>         mergedIb_;
    ComPtr<ID3D11InputLayout>    mergedInputLayout_;
    ComPtr<ID3D11VertexShader>   vsMerged_;
    ComPtr<ID3D11PixelShader>    psMerged_;
    uint32_t                     mergedIndexCount_ = 0;

    // Atlas mesh path (binary-greedy quads + UV texture atlas).
    ComPtr<ID3D11Buffer>             atlasVb_;
    ComPtr<ID3D11Buffer>             atlasIb_;
    ComPtr<ID3D11InputLayout>        atlasInputLayout_;
    ComPtr<ID3D11VertexShader>       vsAtlas_;
    ComPtr<ID3D11PixelShader>        psAtlas_;
    ComPtr<ID3D11Texture2D>          atlasTex_;
    ComPtr<ID3D11ShaderResourceView> atlasSrv_;
    uint32_t                         atlasIndexCount_ = 0;
    float                            atlasOrigin_[3] = { 0, 0, 0 };

    struct GpuAtlasSub {
        uint16_t cx, cy, cz;
        float    aabbMin[3];     // world voxel coords (no grid shift)
        float    aabbMax[3];
        uint32_t firstIndex;
        uint32_t indexCount;
    };
    std::vector<GpuAtlasSub>         atlasSubs_;
    std::unordered_map<uint64_t,uint32_t> atlasSubByChunk_;

    // Splat path.
    ComPtr<ID3D11Texture2D>            splatColorTex_;
    ComPtr<ID3D11RenderTargetView>     splatColorRtv_;
    ComPtr<ID3D11ShaderResourceView>   splatColorSrv_;
    ComPtr<ID3D11Texture2D>            splatDepthTex_;
    ComPtr<ID3D11DepthStencilView>     splatDsv_;
    ComPtr<ID3D11ShaderResourceView>   splatDepthSrv_;
    ComPtr<ID3D11ShaderResourceView>   splatStencilSrv_;
    ComPtr<ID3D11DepthStencilState>    dsSplatPoly_;       // poly draws: depth GT, write 1
    ComPtr<ID3D11DepthStencilState>    dsSplatPoint_;      // splat draws: depth GT, keep stencil
    ComPtr<ID3D11DepthStencilState>    dsAlwaysWrite_;     // depth ALWAYS pass, write ON (SV_Depth lands)
    ComPtr<ID3D11PixelShader>          psSplatRecon_;
    ComPtr<ID3D11PixelShader>          psSplatAlbedoPoly_;
    ComPtr<ID3D11PixelShader>          psPolyVidAlpha0_;
    ComPtr<ID3D11Texture2D>            splatFinalTex_;
    ComPtr<ID3D11UnorderedAccessView>  splatFinalUav_;
    ComPtr<ID3D11ShaderResourceView>   splatFinalSrv_;
    ComPtr<ID3D11PixelShader>          psSplatComposite_;
    ComPtr<ID3D11ComputeShader>        csSplat_;
    ComPtr<ID3D11PixelShader>          psSplatAlbedo_;
    // Sun shadow map.
    ComPtr<ID3D11Texture2D>            shadowTex_;
    ComPtr<ID3D11DepthStencilView>     shadowDsv_;
    ComPtr<ID3D11ShaderResourceView>   shadowSrv_;
    ComPtr<ID3D11SamplerState>         shadowSamp_;
    ComPtr<ID3D11VertexShader>         vsShadow_;
    ComPtr<ID3D11RasterizerState>      rsShadow_;
    uint32_t                           shadowSize_ = 2048;
    float                              shadowLastSunDir_[3] = { 0.0f, 0.0f, 0.0f };
    bool                               shadowValid_ = false;

    ComPtr<ID3D11InputLayout>    inputLayout_;
    ComPtr<ID3D11InputLayout>    inputLayoutHex_;
    ComPtr<ID3D11InputLayout>    inputLayoutBounds_;
    ComPtr<ID3D11RasterizerState> rsNoCull_;
    ComPtr<ID3D11VertexShader>   vsBounds_;
    ComPtr<ID3D11PixelShader>    psBounds_;
    ComPtr<ID3D11Buffer>         boundsVb_;
    ComPtr<ID3D11Buffer>         boundsIb_;

    // PointCS compute path
    ComPtr<ID3D11ComputeShader>  csPoints_;
    ComPtr<ID3D11VertexShader>   vsBlit_;
    ComPtr<ID3D11PixelShader>    psBlit_;
    ComPtr<ID3D11Texture2D>      csColorTex_;
    ComPtr<ID3D11Texture2D>      csDepthTex_;
    ComPtr<ID3D11UnorderedAccessView> csColorUav_;
    ComPtr<ID3D11UnorderedAccessView> csDepthUav_;
    ComPtr<ID3D11ShaderResourceView>  csColorSrv_;
    ComPtr<ID3D11ShaderResourceView>  pointSrv_;
    ComPtr<ID3D11Buffer>              pointAo6Sb_;
    ComPtr<ID3D11ShaderResourceView>  pointAo6Srv_;
    ComPtr<ID3D11Buffer>         cbCS_;
    uint32_t chunkDim_ = 64;
    ComPtr<ID3D11Buffer>         cbPerFrame_;
    ComPtr<ID3D11Buffer>         cbPerChunk_;
    ComPtr<ID3D11RasterizerState> rsSolid_;
    ComPtr<ID3D11RasterizerState> rsWire_;
    ComPtr<ID3D11DepthStencilState> dsTest_;
    ComPtr<ID3D11DepthStencilState> dsAlways_;
    ComPtr<ID3D11DepthStencilState> dsEqual_;
    ComPtr<ID3D11BlendState>        bsNoColor_;
    ComPtr<ID3D11VertexShader>      vsDepth_;

    ComPtr<ID3D11Buffer>         vb_;
    ComPtr<ID3D11Buffer>         ib_;
    ComPtr<ID3D11Buffer>         pointVb_;
    ComPtr<ID3D11Buffer>         pointSb_;
    std::vector<GpuSubMesh>      subs_;
    uint32_t width_ = 0, height_ = 0;
    uint64_t totalTriangles_ = 0;
    uint64_t totalVertices_  = 0;
    uint64_t vbBytes_ = 0;
    uint64_t ibBytes_ = 0;
    uint64_t pointBytes_     = 0;
    uint64_t pointCountL0_   = 0;
    uint64_t pointCountL1_   = 0;
    uint64_t pointCountL2_   = 0;
    uint64_t pointCountL3_   = 0;
    std::vector<std::pair<uint32_t, uint64_t>> colorHistogram_;
    double   colorEntropyBits_ = 0.0;
    double   colorHuffmanBits_ = 0.0;
    uint32_t colorPaletteBits_ = 0;
    uint64_t compRawBytes_         = 0;
    uint64_t compPosBytes_         = 0;
    uint64_t compMaskBytes_        = 0;
    uint64_t compAoBytes_          = 0;
    uint64_t compPaletteBytes_     = 0;
    uint64_t compColorPalIdxBytes_ = 0;
    uint64_t compColorHuffBytes_   = 0;
    uint32_t compPosBitsPerAxis_   = 0;
    uint32_t compChunkDim_         = 0;
    uint64_t compSubclusterPosBytes_ = 0;
    uint32_t compSubclusterDim_      = 0;
    uint64_t compLz4PosBytes_        = 0;
    uint64_t compLz4MaskBytes_       = 0;
    uint64_t compLz4AoBytes_         = 0;
    uint64_t compLz4ColorPalBytes_   = 0;
    uint64_t compLz4TotalBytes_      = 0;
    uint64_t mergedVbBytes_  = 0;
    uint64_t mergedIbBytes_  = 0;
    uint64_t atlasVbBytes_   = 0;
    uint64_t atlasIbBytes_   = 0;
    uint64_t atlasTexBytes_  = 0;
    uint32_t lastDrawn_ = 0;
    uint64_t lastDrawnTris_ = 0;
    uint64_t lastPolyTris_ = 0;
    uint64_t lastPolyVerts_ = 0;
    uint64_t lastPointCount_ = 0;
    float    sceneSpan_[3] = { 0, 0, 0 };
    float    sceneOrigin_[3] = { 0, 0, 0 };
    uint64_t shaderMtime_ = 0;
    void TryHotReloadShaders();
    ComPtr<ID3D11InputLayout> inputLayoutPoly_;

    // TAA: scene render target, two history targets (ping-pong), depth SRV
    // (shares depthTex_), composite shaders.
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
    ComPtr<ID3D11SamplerState>       linearClampSampler_;
    uint32_t taaHistIdx_ = 0;
    uint32_t taaFrame_ = 0;
    bool     taaHistValid_[2] = { false, false };
    bool     postWroteBackbuf_ = false;   // set in DrawScene if post wrote backbuf -> skip MSAA resolve
    float    taaPrevVP_[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
    float    lastClear_[4] = { 0, 0, 0, 1 };
};
