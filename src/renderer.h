#pragma once
#include "mesh.h"
#include "mesh_loader.h"
#include "camera.h"

#include <d3d11.h>
#include <dxgi1_2.h>
#include <vector>
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
    float    aabbMin[3];
    float    aabbMax[3];
    float    chunkBase[3];
};

enum class ShadingMode : int {
    Lit = 0,
    FlatColor = 1,
    Normals = 2,
};

enum class RenderTech : int {
    PolygonBased = 0,
    Points       = 1,
    Hybrid       = 2,   // points when voxel projects to <1 px, polygons otherwise
    HexSprite    = 3,   // 7-vertex hexagon per voxel (closest corner + silhouette)
    PointCS      = 4,   // compute shader software-rasterized points
    PolyVID      = 5,   // 1 voxel = 1 instance, VS expands cube via SV_VertexID
    Billboard    = 6,   // screen-space quad per voxel; PS ray-cube intersects
    BillboardTri = 7,   // billboard but a single bounding triangle per voxel
    Hybrid2      = 8,   // PolyVID up close, BillboardTri at distance
    MergedMesh   = 9,   // greedy-meshed single VB/IB
    Splat        = 10,  // points -> color RT (alpha=size); CS sphere reconstruct
};

enum class PointLighting : int {
    Simple  = 0,        // cam-to-voxel normal, triplanar ambient + ndotl
    Complex = 1,        // visMask 3-face weighted avg (proper but heavier)
};

enum class PointLod : int {
    L0   = 0,           // 1 point per voxel
    L1   = 1,           // 1 point per 2x2x2 (avg color)
    L2   = 2,           // 1 point per 4x4x4 (avg color)
    Auto = 3,           // pick per-chunk by distance
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
    void BeginFrame(float clear[4]);
    void DrawScene(const Camera& cam, ShadingMode mode, int gridSize, RenderTech tech, bool showChunkBounds, bool zPrepass, PointLighting pointLight, PointLod pointLod, float pointLodScale, bool splatFilter);
    void EndFrame(bool vsync);

    ID3D11Device*        Device()  const { return device_.Get(); }
    ID3D11DeviceContext* Context() const { return ctx_.Get(); }

    uint32_t Width()  const { return width_; }
    uint32_t Height() const { return height_; }
    uint64_t TotalTriangles() const { return totalTriangles_; }
    uint64_t TotalVertices()  const { return totalVertices_; }
    uint64_t VbBytes()        const { return vbBytes_; }
    uint64_t IbBytes()        const { return ibBytes_; }
    size_t   DrawCount()      const { return subs_.size(); }
    uint32_t LastDrawnCount() const { return lastDrawn_; }
    uint64_t LastDrawnTris()  const { return lastDrawnTris_; }

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

    // Splat path.
    ComPtr<ID3D11Texture2D>            splatColorTex_;
    ComPtr<ID3D11RenderTargetView>     splatColorRtv_;
    ComPtr<ID3D11ShaderResourceView>   splatColorSrv_;
    ComPtr<ID3D11Texture2D>            splatDepthTex_;
    ComPtr<ID3D11DepthStencilView>     splatDsv_;
    ComPtr<ID3D11Texture2D>            splatFinalTex_;
    ComPtr<ID3D11UnorderedAccessView>  splatFinalUav_;
    ComPtr<ID3D11VertexShader>         vsSplat_;
    ComPtr<ID3D11PixelShader>          psSplat_;
    ComPtr<ID3D11ComputeShader>        csSplat_;
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
    ComPtr<ID3D11Buffer>         cbCS_;
    uint32_t chunkDim_ = 64;
    ComPtr<ID3D11Buffer>         cbPerFrame_;
    ComPtr<ID3D11Buffer>         cbPerChunk_;
    ComPtr<ID3D11RasterizerState> rsSolid_;
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
    uint32_t lastDrawn_ = 0;
    uint64_t lastDrawnTris_ = 0;
    float    sceneSpan_[3] = { 0, 0, 0 };
};
