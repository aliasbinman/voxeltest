#define NOMINMAX
#include "renderer.h"
#include "microprofile.h"
#include <dxgi1_6.h>
#include <functional>

// Extract 6 frustum planes from row-major viewProj (16-float storage).
// Order: 0=L, 1=R, 2=B, 3=T, 4=N, 5=F.
static inline void ExtractFrustumPlanes(const float M[16], float planes[6][4])
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

#include <d3dcompiler.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <queue>
#include <sstream>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

struct CBPerFrame
{
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
    float fogDensity;       // 0 = depth fog off
    float heightFogDensity; // 0 = height fog off
    float heightFogFalloff;
    float heightFogStart;
    float _padHF;
    float sceneOrigin[3];
    float nearZ; // reverse-Z infinite-far: viewZ = nearZ / ndcZ
    float sceneSpan[3];
    float _pad6;
    float prevViewProj[16];
    float jitter[2];
    float _pad7[2];
    float sunViewProj[16]; // unused (kept for shader CB layout compatibility)
    float shadowBias;
    float shadowMapSize;
    float shadowEnable; // always 0 now (shadows removed)
    float sunIntensity;
    float exposure;
    float roughness;
    float colorizeClusters; // 0/1 tint on/off
    float gridSize;         // for shadow wrap (1 = no wrap)
    // CPU-precomputed derived values (mirror of cbPerFrame tail in shaders).
    float invScreenSize[2]; // 1/W, 1/H
    float aspect;           // W/H
    float invAspect;        // H/W
    float aspectTanFov;     // aspect * tanHalfFovY
    float _padPC[3];
};

struct CBPerChunk
{
    float chunkBase[3];
    float _pad;           // lodHalfExtent (points)
    uint32_t voxelBase;   // PolyVID: chunk's first voxel index in pointSb_
    uint32_t chunkLodIdx; // 0..3 splat LOD index
    uint32_t chunkTint;   // colorize-clusters debug
    uint32_t _pad2[2];
};

struct CBPerCS
{
    float viewProj[16];
    float chunkBase[3];
    uint32_t voxelOffset;
    uint32_t voxelCount;
    uint32_t w;
    uint32_t h;
    float lodHalfExtent;
};

static std::string ReadTextFile(const char* path)
{
    std::ifstream f(path);
    if (!f.good())
        return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool Renderer::Init(HWND hwnd, int adapterIdx)
{
    if (!CreateDeviceAndSwap(hwnd, adapterIdx))
        return false;

    RECT rc;
    GetClientRect(hwnd, &rc);
    width_ = (uint32_t)(rc.right - rc.left);
    height_ = (uint32_t)(rc.bottom - rc.top);

    if (!CreateRenderTargets())
        return false;
    if (!CreateShaders())
        return false;
    if (!CreatePipelineState())
        return false;
    return true;
}

void Renderer::Shutdown()
{
    ClearLwWorld();
}

void Renderer::ClearLwWorld()
{
    for (int L = 0; L < lw::kLodCount; ++L)
    {
        lwGpu_[L] = LwGpu{};
    }
    lwHasWorld_ = false;
}

bool Renderer::UploadLwLod(const lw::World& w, int L)
{
    if (!device_)
        return false;
    if (L < 0 || L >= lw::kLodCount)
        return false;
    const lw::LODWorld& src = w.lods[L];
    LwGpu& g = lwGpu_[L];
    g = LwGpu{};
    const uint32_t slotCount = (uint32_t)src.chunks.size();
    if (slotCount == 0)
        return true;
    if (slotCount > lw::kMaxResidentChunksPerLod)
    {
        std::fprintf(stderr, "[lw] LOD %d has %u chunks > max %u\n",
                     L, slotCount, lw::kMaxResidentChunksPerLod);
        return false;
    }

    // ---- Point pool ----
    const uint64_t pointBytes = (uint64_t)src.pointPool.size() * sizeof(lw::DiskPoint);
    if (pointBytes > 0xFFFFFFFFull)
    {
        std::fprintf(stderr, "[lw] LOD %d point pool %llu bytes > 4 GB.\n",
                     L, (unsigned long long)pointBytes);
        return false;
    }
    if (pointBytes > 0)
    {
        D3D11_BUFFER_DESC pd = {};
        pd.Usage = D3D11_USAGE_IMMUTABLE;
        pd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        pd.ByteWidth = (UINT)pointBytes;
        pd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        pd.StructureByteStride = sizeof(lw::DiskPoint);
        D3D11_SUBRESOURCE_DATA psd = {};
        psd.pSysMem = src.pointPool.data();
        if (FAILED(device_->CreateBuffer(&pd, &psd, g.pointSb.GetAddressOf())))
        {
            std::fprintf(stderr, "[lw] LOD %d CreateBuffer points failed (%llu bytes)\n",
                         L, (unsigned long long)pointBytes);
            return false;
        }
        D3D11_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.Format = DXGI_FORMAT_UNKNOWN;
        sv.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        sv.Buffer.NumElements = (UINT)src.pointPool.size();
        device_->CreateShaderResourceView(g.pointSb.Get(), &sv, g.pointSrv.GetAddressOf());
    }

    // ---- ChunkInfo SRV (one entry per slot, indexed by slotIdx) ----
    std::vector<lw::GpuChunkInfo> infos(slotCount);
    for (uint32_t i = 0; i < slotCount; ++i)
    {
        const lw::RuntimeChunk& rc = src.chunks[i];
        lw::GpuChunkInfo& gi = infos[i];
        gi.worldOriginX = (float)rc.worldOriginX;
        gi.worldOriginY = (float)rc.worldOriginY;
        gi.worldOriginZ = (float)rc.worldOriginZ;
        gi.lodScale = (float)src.lodScale;
        gi.poolBase = rc.poolBase;
        gi.paletteBase = i * lw::kPaletteSize;
        gi._pad[0] = gi._pad[1] = 0;
    }
    {
        D3D11_BUFFER_DESC bd = {};
        bd.Usage = D3D11_USAGE_IMMUTABLE;
        bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        bd.ByteWidth = (UINT)(infos.size() * sizeof(lw::GpuChunkInfo));
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        bd.StructureByteStride = sizeof(lw::GpuChunkInfo);
        D3D11_SUBRESOURCE_DATA sd = {infos.data(), 0, 0};
        if (FAILED(device_->CreateBuffer(&bd, &sd, g.chunkInfoSb.GetAddressOf())))
            return false;
        D3D11_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.Format = DXGI_FORMAT_UNKNOWN;
        sv.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        sv.Buffer.NumElements = (UINT)infos.size();
        device_->CreateShaderResourceView(g.chunkInfoSb.Get(), &sv, g.chunkInfoSrv.GetAddressOf());
    }

    // ---- Palette atlas (slotCount * 256 uint32, one slot strip per chunk) ----
    std::vector<uint32_t> atlas((size_t)slotCount * lw::kPaletteSize, 0u);
    for (uint32_t i = 0; i < slotCount; ++i)
    {
        const lw::RuntimeChunk& rc = src.chunks[i];
        const uint32_t n = std::min(rc.paletteCount, (uint32_t)lw::kPaletteSize);
        std::memcpy(atlas.data() + (size_t)i * lw::kPaletteSize,
                    rc.palette, n * sizeof(uint32_t));
    }
    {
        D3D11_BUFFER_DESC bd = {};
        bd.Usage = D3D11_USAGE_IMMUTABLE;
        bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        bd.ByteWidth = (UINT)(atlas.size() * sizeof(uint32_t));
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        bd.StructureByteStride = sizeof(uint32_t);
        D3D11_SUBRESOURCE_DATA sd = {atlas.data(), 0, 0};
        if (FAILED(device_->CreateBuffer(&bd, &sd, g.paletteSb.GetAddressOf())))
            return false;
        D3D11_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.Format = DXGI_FORMAT_UNKNOWN;
        sv.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        sv.Buffer.NumElements = (UINT)atlas.size();
        device_->CreateShaderResourceView(g.paletteSb.Get(), &sv, g.paletteSrv.GetAddressOf());
    }

    // ---- Block pools (PointCS_Block) — split SoA: pos (4B) + col (8B) ----
    uint64_t posBytes = (uint64_t)src.blockPosPool.size() * sizeof(lw::BlockPos);
    uint64_t colBytes = (uint64_t)src.blockColPool.size() * sizeof(lw::BlockCol);
    if (posBytes > 0)
    {
        D3D11_BUFFER_DESC bd = {};
        bd.Usage = D3D11_USAGE_IMMUTABLE;
        bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        bd.ByteWidth = (UINT)posBytes;
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        bd.StructureByteStride = sizeof(lw::BlockPos);
        D3D11_SUBRESOURCE_DATA sd = {src.blockPosPool.data(), 0, 0};
        if (FAILED(device_->CreateBuffer(&bd, &sd, g.blockPosSb.GetAddressOf())))
            return false;
        D3D11_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.Format = DXGI_FORMAT_UNKNOWN;
        sv.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        sv.Buffer.NumElements = (UINT)src.blockPosPool.size();
        device_->CreateShaderResourceView(g.blockPosSb.Get(), &sv, g.blockPosSrv.GetAddressOf());
    }
    if (colBytes > 0)
    {
        D3D11_BUFFER_DESC bd = {};
        bd.Usage = D3D11_USAGE_IMMUTABLE;
        bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        bd.ByteWidth = (UINT)colBytes;
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        bd.StructureByteStride = sizeof(lw::BlockCol);
        D3D11_SUBRESOURCE_DATA sd = {src.blockColPool.data(), 0, 0};
        if (FAILED(device_->CreateBuffer(&bd, &sd, g.blockColSb.GetAddressOf())))
            return false;
        D3D11_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.Format = DXGI_FORMAT_UNKNOWN;
        sv.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        sv.Buffer.NumElements = (UINT)src.blockColPool.size();
        device_->CreateShaderResourceView(g.blockColSb.Get(), &sv, g.blockColSrv.GetAddressOf());
    }

    // ---- PointCS A/B: per-voxel expanded point list ----
    uint64_t pointABytes = (uint64_t)src.blockPointPool.size() * sizeof(uint32_t);
    if (pointABytes > 0)
    {
        D3D11_BUFFER_DESC bd = {};
        bd.Usage = D3D11_USAGE_IMMUTABLE;
        bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        bd.ByteWidth = (UINT)pointABytes;
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        bd.StructureByteStride = sizeof(uint32_t);
        D3D11_SUBRESOURCE_DATA sd = {src.blockPointPool.data(), 0, 0};
        if (FAILED(device_->CreateBuffer(&bd, &sd, g.blockPointSb.GetAddressOf())))
            return false;
        D3D11_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.Format = DXGI_FORMAT_UNKNOWN;
        sv.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        sv.Buffer.NumElements = (UINT)src.blockPointPool.size();
        device_->CreateShaderResourceView(g.blockPointSb.Get(), &sv, g.blockPointSrv.GetAddressOf());
    }

    g.slotCount = slotCount;
    g.pointCount = (uint32_t)src.pointPool.size();
    g.blockCount = (uint32_t)src.blockPosPool.size();
    g.blockPointCount = (uint32_t)src.blockPointPool.size();
    g.bytes = pointBytes + infos.size() * sizeof(lw::GpuChunkInfo) + atlas.size() * sizeof(uint32_t) + posBytes + colBytes + pointABytes;
    std::printf("[lw] LOD %d uploaded: %u slots, %u points, %u blocks, %u blockPoints, %.2f MB GPU\n",
                L, g.slotCount, g.pointCount, g.blockCount, g.blockPointCount, g.bytes / (1024.0 * 1024.0));
    return true;
}

bool Renderer::UploadLwWorld(const lw::World& w)
{
    if (!device_)
        return false;
    ClearLwWorld();
    // Stash world metadata (chunk AABBs, childId, cull arrays). Per-LOD point
    // data uploaded below; CPU pointPool dropped afterward to free memory.
    lwWorld_ = w;
    for (int L = 0; L < lw::kLodCount; ++L)
    {
        if (!UploadLwLod(w, L))
            return false;
        lwWorld_.lods[L].pointPool.clear();
        lwWorld_.lods[L].pointPool.shrink_to_fit();
        lwWorld_.lods[L].blockPosPool.clear();
        lwWorld_.lods[L].blockPosPool.shrink_to_fit();
        lwWorld_.lods[L].blockColPool.clear();
        lwWorld_.lods[L].blockColPool.shrink_to_fit();
    }
    return RebuildLwIdentityIb();
}

bool Renderer::UploadLwLodOnly(const lw::World& w, int L)
{
    // Streaming entry: copies this LOD's metadata into lwWorld_, uploads GPU
    // buffers, refreshes identity IB. Caller must have already called
    // PrepLwWorld with the file's AABB (other LODs may still be loading).
    if (L < 0 || L >= lw::kLodCount)
        return false;
    lwWorld_.lods[L] = w.lods[L]; // copy this LOD only (safe to read)
    if (!UploadLwLod(w, L))
        return false;
    lwWorld_.lods[L].pointPool.clear();
    lwWorld_.lods[L].pointPool.shrink_to_fit();
    return RebuildLwIdentityIb();
}

bool Renderer::FinalizeLwUploads()
{
    return true; // combined-LOD path reverted; no-op kept for caller compat.
}


bool Renderer::RebuildCombinedLwGpu()
{
    if (!device_)
        return false;

    // Compute per-LOD bases + totals across all LODs from CPU-side lwWorld_.
    uint32_t totalSlots = 0, totalBlocks = 0, totalPoints = 0;
    uint32_t totalPaletteEntries = 0;
    for (int L = 0; L < lw::kLodCount; ++L)
    {
        lwGpuC_.lodSlotBase[L]    = totalSlots;
        lwGpuC_.lodBlockBase[L]   = totalBlocks;
        lwGpuC_.lodPointBase[L]   = totalPoints;
        lwGpuC_.lodPaletteBase[L] = totalPaletteEntries;
        const auto& lw = lwWorld_.lods[L];
        totalSlots          += (uint32_t)lw.chunks.size();
        totalBlocks         += (uint32_t)lw.blockPosPool.size();
        totalPoints         += (uint32_t)lw.blockPointPool.size();
        totalPaletteEntries += (uint32_t)lw.chunks.size() * lw::kPaletteSize;
    }
    lwGpuC_ = LwGpuCombined{}; // reset all SRVs
    if (totalSlots == 0)
        return true;

    // Re-stash bases (LwGpuCombined{} reset cleared them).
    {
        uint32_t s = 0, b = 0, p = 0, pal = 0;
        for (int L = 0; L < lw::kLodCount; ++L)
        {
            lwGpuC_.lodSlotBase[L]    = s;
            lwGpuC_.lodBlockBase[L]   = b;
            lwGpuC_.lodPointBase[L]   = p;
            lwGpuC_.lodPaletteBase[L] = pal;
            const auto& lw = lwWorld_.lods[L];
            s   += (uint32_t)lw.chunks.size();
            b   += (uint32_t)lw.blockPosPool.size();
            p   += (uint32_t)lw.blockPointPool.size();
            pal += (uint32_t)lw.chunks.size() * lw::kPaletteSize;
        }
    }

    // ---- ChunkInfo (concat all LODs, fix paletteBase to global index) ----
    {
        std::vector<lw::GpuChunkInfo> infos(totalSlots);
        for (int L = 0; L < lw::kLodCount; ++L)
        {
            const auto& lw = lwWorld_.lods[L];
            const uint32_t base = lwGpuC_.lodSlotBase[L];
            for (uint32_t i = 0; i < (uint32_t)lw.chunks.size(); ++i)
            {
                const lw::RuntimeChunk& rc = lw.chunks[i];
                lw::GpuChunkInfo& gi = infos[base + i];
                gi.worldOriginX = (float)rc.worldOriginX;
                gi.worldOriginY = (float)rc.worldOriginY;
                gi.worldOriginZ = (float)rc.worldOriginZ;
                gi.lodScale = (float)lw.lodScale;
                gi.poolBase = rc.poolBase;
                gi.paletteBase = (base + i) * lw::kPaletteSize;
                gi._pad[0] = gi._pad[1] = 0;
            }
        }
        D3D11_BUFFER_DESC bd = {};
        bd.Usage = D3D11_USAGE_IMMUTABLE;
        bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        bd.ByteWidth = (UINT)(infos.size() * sizeof(lw::GpuChunkInfo));
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        bd.StructureByteStride = sizeof(lw::GpuChunkInfo);
        D3D11_SUBRESOURCE_DATA sd = {infos.data(), 0, 0};
        if (FAILED(device_->CreateBuffer(&bd, &sd, lwGpuC_.chunkInfoSb.GetAddressOf())))
            return false;
        D3D11_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.Format = DXGI_FORMAT_UNKNOWN;
        sv.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        sv.Buffer.NumElements = (UINT)infos.size();
        device_->CreateShaderResourceView(lwGpuC_.chunkInfoSb.Get(), &sv, lwGpuC_.chunkInfoSrv.GetAddressOf());
    }

    // ---- Palette atlas (concat) ----
    {
        std::vector<uint32_t> atlas((size_t)totalPaletteEntries, 0u);
        for (int L = 0; L < lw::kLodCount; ++L)
        {
            const auto& lw = lwWorld_.lods[L];
            const uint32_t base = lwGpuC_.lodSlotBase[L];
            for (uint32_t i = 0; i < (uint32_t)lw.chunks.size(); ++i)
            {
                const lw::RuntimeChunk& rc = lw.chunks[i];
                const uint32_t n = std::min(rc.paletteCount, (uint32_t)lw::kPaletteSize);
                std::memcpy(atlas.data() + (size_t)(base + i) * lw::kPaletteSize,
                            rc.palette, n * sizeof(uint32_t));
            }
        }
        D3D11_BUFFER_DESC bd = {};
        bd.Usage = D3D11_USAGE_IMMUTABLE;
        bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        bd.ByteWidth = (UINT)(atlas.size() * sizeof(uint32_t));
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        bd.StructureByteStride = sizeof(uint32_t);
        D3D11_SUBRESOURCE_DATA sd = {atlas.data(), 0, 0};
        if (FAILED(device_->CreateBuffer(&bd, &sd, lwGpuC_.paletteSb.GetAddressOf())))
            return false;
        D3D11_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.Format = DXGI_FORMAT_UNKNOWN;
        sv.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        sv.Buffer.NumElements = (UINT)atlas.size();
        device_->CreateShaderResourceView(lwGpuC_.paletteSb.Get(), &sv, lwGpuC_.paletteSrv.GetAddressOf());
    }

    // ---- BlockPos (concat) ----
    if (totalBlocks > 0)
    {
        std::vector<lw::BlockPos> pool(totalBlocks);
        for (int L = 0; L < lw::kLodCount; ++L)
        {
            const auto& lw = lwWorld_.lods[L];
            if (lw.blockPosPool.empty()) continue;
            std::memcpy(pool.data() + lwGpuC_.lodBlockBase[L],
                        lw.blockPosPool.data(),
                        lw.blockPosPool.size() * sizeof(lw::BlockPos));
        }
        D3D11_BUFFER_DESC bd = {};
        bd.Usage = D3D11_USAGE_IMMUTABLE;
        bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        bd.ByteWidth = (UINT)(pool.size() * sizeof(lw::BlockPos));
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        bd.StructureByteStride = sizeof(lw::BlockPos);
        D3D11_SUBRESOURCE_DATA sd = {pool.data(), 0, 0};
        if (FAILED(device_->CreateBuffer(&bd, &sd, lwGpuC_.blockPosSb.GetAddressOf())))
            return false;
        D3D11_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.Format = DXGI_FORMAT_UNKNOWN;
        sv.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        sv.Buffer.NumElements = (UINT)pool.size();
        device_->CreateShaderResourceView(lwGpuC_.blockPosSb.Get(), &sv, lwGpuC_.blockPosSrv.GetAddressOf());

        // BlockCol (concat, parallel to BlockPos).
        std::vector<lw::BlockCol> colPool(totalBlocks);
        for (int L = 0; L < lw::kLodCount; ++L)
        {
            const auto& lw = lwWorld_.lods[L];
            if (lw.blockColPool.empty()) continue;
            std::memcpy(colPool.data() + lwGpuC_.lodBlockBase[L],
                        lw.blockColPool.data(),
                        lw.blockColPool.size() * sizeof(lw::BlockCol));
        }
        D3D11_BUFFER_DESC bdc = bd;
        bdc.ByteWidth = (UINT)(colPool.size() * sizeof(lw::BlockCol));
        bdc.StructureByteStride = sizeof(lw::BlockCol);
        D3D11_SUBRESOURCE_DATA sdc = {colPool.data(), 0, 0};
        if (FAILED(device_->CreateBuffer(&bdc, &sdc, lwGpuC_.blockColSb.GetAddressOf())))
            return false;
        D3D11_SHADER_RESOURCE_VIEW_DESC svc = {};
        svc.Format = DXGI_FORMAT_UNKNOWN;
        svc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        svc.Buffer.NumElements = (UINT)colPool.size();
        device_->CreateShaderResourceView(lwGpuC_.blockColSb.Get(), &svc, lwGpuC_.blockColSrv.GetAddressOf());
    }

    // ---- BlockPoint (concat) ----
    if (totalPoints > 0)
    {
        std::vector<uint32_t> pool(totalPoints);
        for (int L = 0; L < lw::kLodCount; ++L)
        {
            const auto& lw = lwWorld_.lods[L];
            if (lw.blockPointPool.empty()) continue;
            std::memcpy(pool.data() + lwGpuC_.lodPointBase[L],
                        lw.blockPointPool.data(),
                        lw.blockPointPool.size() * sizeof(uint32_t));
        }
        D3D11_BUFFER_DESC bd = {};
        bd.Usage = D3D11_USAGE_IMMUTABLE;
        bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        bd.ByteWidth = (UINT)(pool.size() * sizeof(uint32_t));
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        bd.StructureByteStride = sizeof(uint32_t);
        D3D11_SUBRESOURCE_DATA sd = {pool.data(), 0, 0};
        if (FAILED(device_->CreateBuffer(&bd, &sd, lwGpuC_.blockPointSb.GetAddressOf())))
            return false;
        D3D11_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.Format = DXGI_FORMAT_UNKNOWN;
        sv.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        sv.Buffer.NumElements = (UINT)pool.size();
        device_->CreateShaderResourceView(lwGpuC_.blockPointSb.Get(), &sv, lwGpuC_.blockPointSrv.GetAddressOf());
    }

    lwGpuC_.valid = true;
    return true;
}

void Renderer::PrepLwWorld(const lw::World& w)
{
    ClearLwWorld();
    // Copy just AABB; per-LOD data filled by subsequent UploadLwLodOnly calls.
    for (int i = 0; i < 3; ++i)
    {
        lwWorld_.worldAabbMin[i] = w.worldAabbMin[i];
        lwWorld_.worldAabbMax[i] = w.worldAabbMax[i];
    }
    lwHasWorld_ = true;
}

bool Renderer::RebuildLwIdentityIb()
{
    // ---- Identity IB ----
    // Size = max chunk poolCount across all LODs (each draw indexes 0..N-1).
    uint32_t maxChunkPoints = 0;
    for (int L = 0; L < lw::kLodCount; ++L)
    {
        for (const auto& rc : lwWorld_.lods[L].chunks)
        {
            if (rc.poolCount > maxChunkPoints)
                maxChunkPoints = rc.poolCount;
        }
    }
    if (maxChunkPoints > 0 && maxChunkPoints != lwIdentityIbCount_)
    {
        std::vector<uint32_t> ib(maxChunkPoints);
        for (uint32_t i = 0; i < maxChunkPoints; ++i)
            ib[i] = i;
        D3D11_BUFFER_DESC bd = {};
        bd.Usage = D3D11_USAGE_IMMUTABLE;
        bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
        bd.ByteWidth = (UINT)(ib.size() * sizeof(uint32_t));
        D3D11_SUBRESOURCE_DATA sd = {ib.data(), 0, 0};
        lwIdentityIb_.Reset();
        if (FAILED(device_->CreateBuffer(&bd, &sd, lwIdentityIb_.GetAddressOf())))
        {
            std::fprintf(stderr, "[lw] identity IB CreateBuffer failed (%u uints)\n", maxChunkPoints);
            return false;
        }
        lwIdentityIbCount_ = maxChunkPoints;
        std::printf("[lw] identity IB: %u uint32 (%.2f MB)\n",
                    maxChunkPoints, maxChunkPoints * 4.0 / (1024.0 * 1024.0));
    }

    // New world means any cached shadow map is stale.
    shadowMapDirty_ = true;

    lwHasWorld_ = true;
    return true;
}

std::vector<std::string> Renderer::EnumerateAdapters()
{
    std::vector<std::string> out;
    ComPtr<IDXGIFactory6> f6;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(f6.GetAddressOf()));
    if (SUCCEEDED(hr))
    {
        // Factory6 path: enumerate in performance-preferred order so idx 0 is
        // typically the discrete GPU on a laptop.
        for (UINT i = 0;; ++i)
        {
            ComPtr<IDXGIAdapter1> a;
            if (f6->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                               IID_PPV_ARGS(a.GetAddressOf())) == DXGI_ERROR_NOT_FOUND)
                break;
            DXGI_ADAPTER_DESC1 d{};
            a->GetDesc1(&d);
            if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                continue;
            char buf[256];
            size_t cv = 0;
            wcstombs_s(&cv, buf, sizeof(buf), d.Description, _TRUNCATE);
            out.emplace_back(buf);
        }
        return out;
    }
    ComPtr<IDXGIFactory1> f1;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(f1.GetAddressOf()))))
        return out;
    for (UINT i = 0;; ++i)
    {
        ComPtr<IDXGIAdapter1> a;
        if (f1->EnumAdapters1(i, a.GetAddressOf()) == DXGI_ERROR_NOT_FOUND)
            break;
        DXGI_ADAPTER_DESC1 d{};
        a->GetDesc1(&d);
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            continue;
        char buf[256];
        size_t cv = 0;
        wcstombs_s(&cv, buf, sizeof(buf), d.Description, _TRUNCATE);
        out.emplace_back(buf);
    }
    return out;
}

bool Renderer::CreateDeviceAndSwap(HWND hwnd, int adapterIdx)
{
    UINT flags = 0;
#ifdef _DEBUG
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    D3D_FEATURE_LEVEL flvl;
    D3D_FEATURE_LEVEL want[] = {D3D_FEATURE_LEVEL_11_0};

    // Resolve adapter: explicit index uses DXGI enumeration matching
    // EnumerateAdapters() order (high-perf first on Factory6). idx < 0 = system default.
    ComPtr<IDXGIAdapter1> chosen;
    if (adapterIdx >= 0)
    {
        ComPtr<IDXGIFactory6> f6;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(f6.GetAddressOf()))))
        {
            int seen = 0;
            for (UINT i = 0;; ++i)
            {
                ComPtr<IDXGIAdapter1> a;
                if (f6->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                   IID_PPV_ARGS(a.GetAddressOf())) == DXGI_ERROR_NOT_FOUND)
                    break;
                DXGI_ADAPTER_DESC1 d{};
                a->GetDesc1(&d);
                if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                    continue;
                if (seen == adapterIdx)
                {
                    chosen = a;
                    break;
                }
                ++seen;
            }
        }
        if (!chosen)
        {
            ComPtr<IDXGIFactory1> f1;
            if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(f1.GetAddressOf()))))
            {
                int seen = 0;
                for (UINT i = 0;; ++i)
                {
                    ComPtr<IDXGIAdapter1> a;
                    if (f1->EnumAdapters1(i, a.GetAddressOf()) == DXGI_ERROR_NOT_FOUND)
                        break;
                    DXGI_ADAPTER_DESC1 d{};
                    a->GetDesc1(&d);
                    if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                        continue;
                    if (seen == adapterIdx)
                    {
                        chosen = a;
                        break;
                    }
                    ++seen;
                }
            }
        }
    }

    // Note: when chosen is non-null, driver type MUST be UNKNOWN.
    D3D_DRIVER_TYPE drvType = chosen ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE;
    IDXGIAdapter* adp = chosen.Get();

    HRESULT hr = D3D11CreateDevice(
        adp, drvType, nullptr, flags,
        want, _countof(want), D3D11_SDK_VERSION,
        device_.GetAddressOf(), &flvl, ctx_.GetAddressOf());
#ifdef _DEBUG
    if (FAILED(hr))
    {
        flags &= ~D3D11_CREATE_DEVICE_DEBUG;
        hr = D3D11CreateDevice(
            adp, drvType, nullptr, flags,
            want, _countof(want), D3D11_SDK_VERSION,
            device_.GetAddressOf(), &flvl, ctx_.GetAddressOf());
    }
#endif
    if (FAILED(hr))
        return false;

    ComPtr<IDXGIDevice> dxgiDev;
    device_.As(&dxgiDev);
    ComPtr<IDXGIAdapter> adapter;
    dxgiDev->GetAdapter(adapter.GetAddressOf());
    ComPtr<IDXGIFactory2> factory;
    adapter->GetParent(IID_PPV_ARGS(factory.GetAddressOf()));

    // Detect tearing support so windowed Present can uncap above refresh.
    tearingSupported_ = false;
    {
        ComPtr<IDXGIFactory5> f5;
        if (SUCCEEDED(factory.As(&f5)))
        {
            BOOL allow = FALSE;
            if (SUCCEEDED(f5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allow, sizeof(allow))))
            {
                tearingSupported_ = (allow == TRUE);
            }
        }
    }

    DXGI_SWAP_CHAIN_DESC1 sd = {};
    sd.Width = 0;
    sd.Height = 0;
    sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 2;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sd.Scaling = DXGI_SCALING_STRETCH;
    if (tearingSupported_)
        sd.Flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;

    hr = factory->CreateSwapChainForHwnd(
        device_.Get(), hwnd, &sd, nullptr, nullptr, swap_.GetAddressOf());
    if (FAILED(hr))
        return false;
    // Disable DXGI's Alt-Enter auto fullscreen (needed for tearing).
    factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
    return true;
}

bool Renderer::CreateRenderTargets()
{
    rtv_.Reset();
    dsv_.Reset();
    depthTex_.Reset();
    splatColorTex_.Reset();
    splatColorRtv_.Reset();
    splatColorUav_.Reset();
    splatMaskUav_.Reset();
    splatDepthUavF_.Reset();
    splatBlockDepthTex_.Reset();
    splatBlockDepthSrv_.Reset();
    splatColorSrv_.Reset();
    splatDepthTex_.Reset();
    splatDsv_.Reset();
    splatFinalTex_.Reset();
    splatFinalUav_.Reset();
    splatFinalDepthTex_.Reset();
    splatFinalDepthUav_.Reset();
    splatFinalDepthSrv_.Reset();
    splatFinal2Tex_.Reset();
    splatFinal2Uav_.Reset();
    splatFinal2Srv_.Reset();
    splatFinal2DepthTex_.Reset();
    splatFinal2DepthUav_.Reset();
    splatFinal2DepthSrv_.Reset();
    splatMaskTex_.Reset();
    splatMaskRtv_.Reset();
    splatMaskSrv_.Reset();

    ComPtr<ID3D11Texture2D> backBuf;
    HRESULT hr = swap_->GetBuffer(0, IID_PPV_ARGS(backBuf.GetAddressOf()));
    if (FAILED(hr))
        return false;
    hr = device_->CreateRenderTargetView(backBuf.Get(), nullptr, rtv_.GetAddressOf());
    if (FAILED(hr))
        return false;

    D3D11_TEXTURE2D_DESC td = {};
    td.Width = width_;
    td.Height = height_;
    td.MipLevels = 1;
    td.ArraySize = 1;
    // R32_TYPELESS so TAA can sample depth as SRV.
    td.Format = DXGI_FORMAT_R32_TYPELESS;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
    hr = device_->CreateTexture2D(&td, nullptr, depthTex_.GetAddressOf());
    if (FAILED(hr))
        return false;
    D3D11_DEPTH_STENCIL_VIEW_DESC dsvd = {};
    dsvd.Format = DXGI_FORMAT_D32_FLOAT;
    dsvd.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    hr = device_->CreateDepthStencilView(depthTex_.Get(), &dsvd, dsv_.GetAddressOf());
    if (FAILED(hr))
        return false;
    depthSrv_.Reset();
    {
        D3D11_SHADER_RESOURCE_VIEW_DESC dsv = {};
        dsv.Format = DXGI_FORMAT_R32_FLOAT;
        dsv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        dsv.Texture2D.MipLevels = 1;
        device_->CreateShaderResourceView(depthTex_.Get(), &dsv, depthSrv_.GetAddressOf());
    }

    // TAA RT (scene) + ping-pong history (R8G8B8A8 with RT + SR binding).
    taaSceneTex_.Reset();
    taaSceneRtv_.Reset();
    taaSceneSrv_.Reset();
    for (int i = 0; i < 2; ++i)
    {
        taaHistTex_[i].Reset();
        taaHistRtv_[i].Reset();
        taaHistSrv_[i].Reset();
        taaHistValid_[i] = false;
    }
    {
        D3D11_TEXTURE2D_DESC ct = {};
        ct.Width = width_;
        ct.Height = height_;
        ct.MipLevels = 1;
        ct.ArraySize = 1;
        ct.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        ct.SampleDesc.Count = 1;
        ct.Usage = D3D11_USAGE_DEFAULT;
        ct.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        device_->CreateTexture2D(&ct, nullptr, taaSceneTex_.GetAddressOf());
        device_->CreateRenderTargetView(taaSceneTex_.Get(), nullptr, taaSceneRtv_.GetAddressOf());
        device_->CreateShaderResourceView(taaSceneTex_.Get(), nullptr, taaSceneSrv_.GetAddressOf());

        D3D11_TEXTURE2D_DESC ht = ct;
        ht.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        for (int i = 0; i < 2; ++i)
        {
            device_->CreateTexture2D(&ht, nullptr, taaHistTex_[i].GetAddressOf());
            device_->CreateRenderTargetView(taaHistTex_[i].Get(), nullptr, taaHistRtv_[i].GetAddressOf());
            device_->CreateShaderResourceView(taaHistTex_[i].Get(), nullptr, taaHistSrv_[i].GetAddressOf());
        }
    }

    // Godray (64x64 R8 — fixed size). [0] = mark, [1,2] = blur+EMA ping-pong.
    for (int i = 0; i < 3; ++i)
    {
        godrayTex_[i].Reset();
        godrayRtv_[i].Reset();
        godraySrv_[i].Reset();
        D3D11_TEXTURE2D_DESC gt = {};
        gt.Width = 64;
        gt.Height = 64;
        gt.MipLevels = 1;
        gt.ArraySize = 1;
        gt.Format = DXGI_FORMAT_R8_UNORM;
        gt.SampleDesc.Count = 1;
        gt.Usage = D3D11_USAGE_DEFAULT;
        gt.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        device_->CreateTexture2D(&gt, nullptr, godrayTex_[i].GetAddressOf());
        device_->CreateRenderTargetView(godrayTex_[i].Get(), nullptr, godrayRtv_[i].GetAddressOf());
        device_->CreateShaderResourceView(godrayTex_[i].Get(), nullptr, godraySrv_[i].GetAddressOf());
    }

    // Two-pass split vis textures: depth R32_UINT + colour R16_UINT.
    {
        D3D11_TEXTURE2D_DESC vd = {};
        vd.Width = width_;
        vd.Height = height_;
        vd.MipLevels = 1;
        vd.ArraySize = 1;
        vd.Format = DXGI_FORMAT_R32_UINT;
        vd.SampleDesc.Count = 1;
        vd.Usage = D3D11_USAGE_DEFAULT;
        vd.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        visDepthTex_.Reset(); visDepthUav_.Reset(); visDepthSrv_.Reset();
        visColorTex_.Reset(); visColorUav_.Reset(); visColorSrv_.Reset();
        device_->CreateTexture2D(&vd, nullptr, visDepthTex_.GetAddressOf());
        device_->CreateUnorderedAccessView(visDepthTex_.Get(), nullptr, visDepthUav_.GetAddressOf());
        device_->CreateShaderResourceView(visDepthTex_.Get(), nullptr, visDepthSrv_.GetAddressOf());
        D3D11_TEXTURE2D_DESC vcd = vd;
        vcd.Format = DXGI_FORMAT_R16_UINT;
        device_->CreateTexture2D(&vcd, nullptr, visColorTex_.GetAddressOf());
        device_->CreateUnorderedAccessView(visColorTex_.Get(), nullptr, visColorUav_.GetAddressOf());
        device_->CreateShaderResourceView(visColorTex_.Get(), nullptr, visColorSrv_.GetAddressOf());
    }

    // Splat color RT + final UAV target.
    D3D11_TEXTURE2D_DESC sd2 = {};
    sd2.Width = width_;
    sd2.Height = height_;
    sd2.MipLevels = 1;
    sd2.ArraySize = 1;
    sd2.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd2.SampleDesc.Count = 1;
    sd2.Usage = D3D11_USAGE_DEFAULT;
    sd2.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    if (FAILED(device_->CreateTexture2D(&sd2, nullptr, splatColorTex_.GetAddressOf())))
        return false;
    if (FAILED(device_->CreateRenderTargetView(splatColorTex_.Get(), nullptr, splatColorRtv_.GetAddressOf())))
        return false;
    if (FAILED(device_->CreateShaderResourceView(splatColorTex_.Get(), nullptr, splatColorSrv_.GetAddressOf())))
        return false;
    if (FAILED(device_->CreateUnorderedAccessView(splatColorTex_.Get(), nullptr, splatColorUav_.GetAddressOf())))
        return false;

    sd2.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(device_->CreateTexture2D(&sd2, nullptr, splatFinalTex_.GetAddressOf())))
        return false;
    if (FAILED(device_->CreateUnorderedAccessView(splatFinalTex_.Get(), nullptr, splatFinalUav_.GetAddressOf())))
        return false;
    if (FAILED(device_->CreateShaderResourceView(splatFinalTex_.Get(), nullptr, splatFinalSrv_.GetAddressOf())))
        return false;

    // Per-pixel splat info emitted by point PS (MRT slot 1), consumed by CS.
    // Layout: visMask(6) | face0_ao(4) | face1_ao(4) | ... | face5_ao(4) = 30 bits.
    // R32_UINT — CS picks the relevant face during dilate and decodes its AO.
    D3D11_TEXTURE2D_DESC smd = sd2;
    smd.Format = DXGI_FORMAT_R32_UINT;
    smd.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    if (FAILED(device_->CreateTexture2D(&smd, nullptr, splatMaskTex_.GetAddressOf())))
        return false;
    if (FAILED(device_->CreateRenderTargetView(splatMaskTex_.Get(), nullptr, splatMaskRtv_.GetAddressOf())))
        return false;
    if (FAILED(device_->CreateShaderResourceView(splatMaskTex_.Get(), nullptr, splatMaskSrv_.GetAddressOf())))
        return false;
    if (FAILED(device_->CreateUnorderedAccessView(splatMaskTex_.Get(), nullptr, splatMaskUav_.GetAddressOf())))
        return false;

    // Dilated depth target.
    D3D11_TEXTURE2D_DESC sdf = sd2;
    sdf.Format = DXGI_FORMAT_R32_FLOAT;
    if (FAILED(device_->CreateTexture2D(&sdf, nullptr, splatFinalDepthTex_.GetAddressOf())))
        return false;
    if (FAILED(device_->CreateUnorderedAccessView(splatFinalDepthTex_.Get(), nullptr, splatFinalDepthUav_.GetAddressOf())))
        return false;
    if (FAILED(device_->CreateShaderResourceView(splatFinalDepthTex_.Get(), nullptr, splatFinalDepthSrv_.GetAddressOf())))
        return false;

    // Second-pass dilate output (ping-pong target for csmain_splat_fill).
    if (FAILED(device_->CreateTexture2D(&sd2, nullptr, splatFinal2Tex_.GetAddressOf())))
        return false;
    if (FAILED(device_->CreateUnorderedAccessView(splatFinal2Tex_.Get(), nullptr, splatFinal2Uav_.GetAddressOf())))
        return false;
    if (FAILED(device_->CreateShaderResourceView(splatFinal2Tex_.Get(), nullptr, splatFinal2Srv_.GetAddressOf())))
        return false;
    if (FAILED(device_->CreateTexture2D(&sdf, nullptr, splatFinal2DepthTex_.GetAddressOf())))
        return false;
    if (FAILED(device_->CreateUnorderedAccessView(splatFinal2DepthTex_.Get(), nullptr, splatFinal2DepthUav_.GetAddressOf())))
        return false;
    if (FAILED(device_->CreateShaderResourceView(splatFinal2DepthTex_.Get(), nullptr, splatFinal2DepthSrv_.GetAddressOf())))
        return false;

    // Dedicated depth for splat pass.
    D3D11_TEXTURE2D_DESC sdd = {};
    sdd.Width = width_;
    sdd.Height = height_;
    sdd.MipLevels = 1;
    sdd.ArraySize = 1;
    sdd.Format = DXGI_FORMAT_R32_TYPELESS;
    sdd.SampleDesc.Count = 1;
    sdd.Usage = D3D11_USAGE_DEFAULT;
    sdd.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(device_->CreateTexture2D(&sdd, nullptr, splatDepthTex_.GetAddressOf())))
        return false;
    D3D11_DEPTH_STENCIL_VIEW_DESC sddv = {};
    sddv.Format = DXGI_FORMAT_D32_FLOAT;
    sddv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    if (FAILED(device_->CreateDepthStencilView(splatDepthTex_.Get(), &sddv, splatDsv_.GetAddressOf())))
        return false;
    D3D11_SHADER_RESOURCE_VIEW_DESC sdsv = {};
    sdsv.Format = DXGI_FORMAT_R32_FLOAT;
    sdsv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    sdsv.Texture2D.MipLevels = 1;
    if (FAILED(device_->CreateShaderResourceView(splatDepthTex_.Get(), &sdsv, splatDepthSrv_.GetAddressOf())))
        return false;
    // Separate R32F texture for the CS splat-route depth output (DSV+UAV on the
    // same resource isn't allowed in D3D11). splatCS still reads splatDepthSrv_
    // — when block-route is active, we'll bind splatBlockDepthSrv_ at the same
    // slot instead via SRV swap before the dispatch.
    {
        D3D11_TEXTURE2D_DESC bdd = sdd;
        bdd.Format = DXGI_FORMAT_R32_FLOAT;
        bdd.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(device_->CreateTexture2D(&bdd, nullptr, splatBlockDepthTex_.GetAddressOf())))
            return false;
        if (FAILED(device_->CreateUnorderedAccessView(splatBlockDepthTex_.Get(), nullptr, splatDepthUavF_.GetAddressOf())))
            return false;
        if (FAILED(device_->CreateShaderResourceView(splatBlockDepthTex_.Get(), nullptr, splatBlockDepthSrv_.GetAddressOf())))
            return false;
    }

    // GPU-memory accounting for the splat pass RTs/UAVs.
    auto bytesOf = [&](DXGI_FORMAT f) -> uint64_t
    {
        switch (f)
        {
        case DXGI_FORMAT_R8G8B8A8_UNORM:
            return 4ull;
        case DXGI_FORMAT_R32_FLOAT:
        case DXGI_FORMAT_R32_UINT:
        case DXGI_FORMAT_R32_TYPELESS:
            return 4ull;
        case DXGI_FORMAT_R8_UINT:
            return 1ull;
        default:
            return 4ull;
        }
    };
    uint64_t pix = (uint64_t)width_ * (uint64_t)height_;
    splatRtBytes_ = pix * (bytesOf(DXGI_FORMAT_R8G8B8A8_UNORM)   // splatColorTex_
                           + bytesOf(DXGI_FORMAT_R8G8B8A8_UNORM) // splatFinalTex_
                           + bytesOf(DXGI_FORMAT_R32_UINT)       // splatMaskTex_
                           + bytesOf(DXGI_FORMAT_R32_FLOAT)      // splatFinalDepthTex_
                           + bytesOf(DXGI_FORMAT_R32_TYPELESS)); // splatDepthTex_
    return true;
}

bool Renderer::CreateShaders()
{
    UINT cflags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3;
    HRESULT hr = S_OK;

    // -------------- Shared shaders in voxel.hlsl (post-process / dilate / TAA) --------------
    // Shaders #include "shading.hlsli" for shared lighting code. Use
    // D3D_COMPILE_STANDARD_FILE_INCLUDE handler with the .hlsl's directory
    // as the base so the include resolves relative to shaders/.
    std::string src = ReadTextFile("shaders/voxel.hlsl");
    if (src.empty())
    {
        std::fprintf(stderr, "shaders/voxel.hlsl not found\n");
        if (!shaderReloading_)
            MessageBoxA(nullptr, "shaders/voxel.hlsl not found", "Renderer", MB_ICONERROR);
        return false;
    }
    auto compile = [&](const char* entry, const char* target, ComPtr<ID3DBlob>& blob) -> bool
    {
        ComPtr<ID3DBlob> errs;
        HRESULT chr = D3DCompile(src.data(), src.size(), "shaders/voxel.hlsl", nullptr,
                                 D3D_COMPILE_STANDARD_FILE_INCLUDE,
                                 entry, target, cflags, 0, blob.GetAddressOf(), errs.GetAddressOf());
        if (FAILED(chr))
        {
            std::string msg = "Shader compile error [";
            msg += entry;
            msg += "]\n";
            if (errs)
                msg += std::string((const char*)errs->GetBufferPointer(), errs->GetBufferSize());
            std::fprintf(stderr, "%s\n", msg.c_str());
            OutputDebugStringA(msg.c_str());
            OutputDebugStringA("\n");
            if (!shaderReloading_)
                MessageBoxA(nullptr, msg.c_str(), entry, MB_ICONERROR);
            return false;
        }
        return true;
    };

    ComPtr<ID3DBlob> csbSp, csbSpFill, csbShBlur;
    if (!compile("csmain_splat", "cs_5_0", csbSp))
        return false;
    if (!compile("csmain_splat_fill", "cs_5_0", csbSpFill))
        return false;
    if (!compile("csmain_shadow_blur", "cs_5_0", csbShBlur))
        return false;

    // Fullscreen VS/PS passes live in postfx.hlsl (separate translation unit).
    std::string fxSrc = ReadTextFile("shaders/postfx.hlsl");
    if (fxSrc.empty())
    {
        std::fprintf(stderr, "shaders/postfx.hlsl not found\n");
        if (!shaderReloading_)
            MessageBoxA(nullptr, "shaders/postfx.hlsl not found", "Renderer", MB_ICONERROR);
        return false;
    }
    auto compileFx = [&](const char* entry, const char* target, ComPtr<ID3DBlob>& blob) -> bool
    {
        ComPtr<ID3DBlob> errs;
        HRESULT chr = D3DCompile(fxSrc.data(), fxSrc.size(), "shaders/postfx.hlsl", nullptr,
                                 D3D_COMPILE_STANDARD_FILE_INCLUDE,
                                 entry, target, cflags, 0, blob.GetAddressOf(), errs.GetAddressOf());
        if (FAILED(chr))
        {
            std::string msg = "Shader compile error [";
            msg += entry;
            msg += "]\n";
            if (errs)
                msg += std::string((const char*)errs->GetBufferPointer(), errs->GetBufferSize());
            std::fprintf(stderr, "%s\n", msg.c_str());
            OutputDebugStringA(msg.c_str());
            OutputDebugStringA("\n");
            if (!shaderReloading_)
                MessageBoxA(nullptr, msg.c_str(), entry, MB_ICONERROR);
            return false;
        }
        return true;
    };
    ComPtr<ID3DBlob> psbSc, vsbTa, psbTa, psbPo, vsbl, psbGrMark, psbGrBlur;
    if (!compileFx("psmain_splat_composite", "ps_5_0", psbSc))
        return false;
    if (!compileFx("vsmain_taa", "vs_5_0", vsbTa))
        return false;
    if (!compileFx("psmain_taa", "ps_5_0", psbTa))
        return false;
    if (!compileFx("psmain_post", "ps_5_0", psbPo))
        return false;
    if (!compileFx("vsmain_blit", "vs_5_0", vsbl))
        return false;
    if (!compileFx("psmain_godray_mark", "ps_5_0", psbGrMark))
        return false;
    if (!compileFx("psmain_godray_blur", "ps_5_0", psbGrBlur))
        return false;
    ComPtr<ID3DBlob> psbGrBlurAniso;
    if (!compileFx("psmain_godray_blur_aniso", "ps_5_0", psbGrBlurAniso))
        return false;

    hr = device_->CreateComputeShader(csbSp->GetBufferPointer(), csbSp->GetBufferSize(), nullptr, csSplat_.GetAddressOf());
    if (FAILED(hr))
        return false;
    hr = device_->CreateComputeShader(csbSpFill->GetBufferPointer(), csbSpFill->GetBufferSize(), nullptr, csSplatFill_.GetAddressOf());
    if (FAILED(hr))
        return false;
    hr = device_->CreateComputeShader(csbShBlur->GetBufferPointer(), csbShBlur->GetBufferSize(), nullptr, csShadowBlur_.GetAddressOf());
    if (FAILED(hr))
        return false;
    hr = device_->CreatePixelShader(psbSc->GetBufferPointer(), psbSc->GetBufferSize(), nullptr, psSplatComposite_.GetAddressOf());
    if (FAILED(hr))
        return false;
    hr = device_->CreateVertexShader(vsbTa->GetBufferPointer(), vsbTa->GetBufferSize(), nullptr, vsTaa_.GetAddressOf());
    if (FAILED(hr))
        return false;
    hr = device_->CreatePixelShader(psbTa->GetBufferPointer(), psbTa->GetBufferSize(), nullptr, psTaa_.GetAddressOf());
    if (FAILED(hr))
        return false;
    hr = device_->CreatePixelShader(psbPo->GetBufferPointer(), psbPo->GetBufferSize(), nullptr, psPost_.GetAddressOf());
    if (FAILED(hr))
        return false;
    hr = device_->CreateVertexShader(vsbl->GetBufferPointer(), vsbl->GetBufferSize(), nullptr, vsBlit_.GetAddressOf());
    if (FAILED(hr))
        return false;
    hr = device_->CreatePixelShader(psbGrMark->GetBufferPointer(), psbGrMark->GetBufferSize(), nullptr, psGodrayMark_.GetAddressOf());
    if (FAILED(hr))
        return false;
    hr = device_->CreatePixelShader(psbGrBlur->GetBufferPointer(), psbGrBlur->GetBufferSize(), nullptr, psGodrayBlur_.GetAddressOf());
    if (FAILED(hr))
        return false;
    hr = device_->CreatePixelShader(psbGrBlurAniso->GetBufferPointer(), psbGrBlurAniso->GetBufferSize(), nullptr, psGodrayBlurAniso_.GetAddressOf());
    if (FAILED(hr))
        return false;
    {
        D3D11_SAMPLER_DESC sd = {};
        sd.Filter = D3D11_FILTER_ANISOTROPIC;
        sd.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        sd.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sd.MaxAnisotropy = 16;
        sd.MaxLOD = D3D11_FLOAT32_MAX;
        sd.ComparisonFunc = D3D11_COMPARISON_NEVER;
        device_->CreateSamplerState(&sd, anisoSamp_.GetAddressOf());
    }
    {
        D3D11_SAMPLER_DESC sm = {};
        sm.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sm.AddressU = sm.AddressV = sm.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sm.MinLOD = 0;
        sm.MaxLOD = D3D11_FLOAT32_MAX;
        device_->CreateSamplerState(&sm, linearClampSampler_.GetAddressOf());
    }

    // -------------- LW shaders (lodworld.hlsl) --------------
    {
        std::string lwSrc = ReadTextFile("shaders/lodworld.hlsl");
        if (lwSrc.empty())
        {
            std::fprintf(stderr, "shaders/lodworld.hlsl not found\n");
            if (!shaderReloading_)
                MessageBoxA(nullptr, "shaders/lodworld.hlsl not found", "Renderer", MB_ICONERROR);
            return false;
        }
        auto compileLw = [&](const char* entry, const char* target, ComPtr<ID3DBlob>& blob) -> bool
        {
            ComPtr<ID3DBlob> errs;
            HRESULT chr = D3DCompile(lwSrc.data(), lwSrc.size(), "shaders/lodworld.hlsl", nullptr,
                                     D3D_COMPILE_STANDARD_FILE_INCLUDE,
                                     entry, target, cflags, 0, blob.GetAddressOf(), errs.GetAddressOf());
            if (FAILED(chr))
            {
                std::string msg = "LW shader compile [";
                msg += entry;
                msg += "]\n";
                if (errs)
                    msg += std::string((const char*)errs->GetBufferPointer(), errs->GetBufferSize());
                std::fprintf(stderr, "%s\n", msg.c_str());
                OutputDebugStringA(msg.c_str());
                OutputDebugStringA("\n");
                if (!shaderReloading_)
                    MessageBoxA(nullptr, msg.c_str(), entry, MB_ICONERROR);
                return false;
            }
            return true;
        };
        ComPtr<ID3DBlob> bvs, bps, bpsd, bpslv, bvsB, bpsB, bvsPa, bpsPaLit;
        if (!compileLw("vsmain_lw_points", "vs_5_0", bvs))
            return false;
        if (!compileLw("psmain_lw_splat_albedo", "ps_5_0", bps))
            return false;
        if (!compileLw("psmain_lw_debug", "ps_5_0", bpsd))
            return false;
        if (!compileLw("psmain_lw_lodviz", "ps_5_0", bpslv))
            return false;
        if (!compileLw("vsmain_lw_bounds", "vs_5_0", bvsB))
            return false;
        if (!compileLw("psmain_lw_bounds", "ps_5_0", bpsB))
            return false;
        if (!compileLw("vsmain_lw_polyaxis", "vs_5_0", bvsPa))
            return false;
        if (!compileLw("psmain_lw_polyaxis_lit", "ps_5_0", bpsPaLit))
            return false;
        ComPtr<ID3DBlob> bvsR, bcsBlkWl, bcsBlkCWl, bpsR2;
        if (!compileLw("csmain_lw_block_depth_worklist", "cs_5_0", bcsBlkWl))
            return false;
        if (!compileLw("csmain_lw_block_color_worklist", "cs_5_0", bcsBlkCWl))
            return false;
        ComPtr<ID3DBlob> bcsBlkSpWl;
        if (!compileLw("csmain_lw_block_splat_worklist", "cs_5_0", bcsBlkSpWl))
            return false;
        ComPtr<ID3DBlob> bvsBlkPt, bpsBlkPt;
        if (!compileLw("vsmain_lw_blockpoint", "vs_5_0", bvsBlkPt))
            return false;
        if (!compileLw("psmain_lw_blockpoint", "ps_5_0", bpsBlkPt))
            return false;
        if (!compileLw("vsmain_lw_resolve", "vs_5_0", bvsR))
            return false;
        if (!compileLw("psmain_lw_resolve_twopass", "ps_5_0", bpsR2))
            return false;
        hr = device_->CreateVertexShader(bvs->GetBufferPointer(), bvs->GetBufferSize(), nullptr, vsLwPoints_.GetAddressOf());
        if (FAILED(hr))
            return false;
        hr = device_->CreatePixelShader(bps->GetBufferPointer(), bps->GetBufferSize(), nullptr, psLwSplatAlbedo_.GetAddressOf());
        if (FAILED(hr))
            return false;
        hr = device_->CreatePixelShader(bpsd->GetBufferPointer(), bpsd->GetBufferSize(), nullptr, psLwDebug_.GetAddressOf());
        if (FAILED(hr))
            return false;
        hr = device_->CreatePixelShader(bpslv->GetBufferPointer(), bpslv->GetBufferSize(), nullptr, psLwLodViz_.GetAddressOf());
        if (FAILED(hr))
            return false;
        hr = device_->CreateVertexShader(bvsB->GetBufferPointer(), bvsB->GetBufferSize(), nullptr, vsLwBounds_.GetAddressOf());
        if (FAILED(hr))
            return false;
        hr = device_->CreatePixelShader(bpsB->GetBufferPointer(), bpsB->GetBufferSize(), nullptr, psLwBounds_.GetAddressOf());
        if (FAILED(hr))
            return false;
        hr = device_->CreateVertexShader(bvsPa->GetBufferPointer(), bvsPa->GetBufferSize(), nullptr, vsLwPolyAxis_.GetAddressOf());
        if (FAILED(hr))
            return false;
        hr = device_->CreatePixelShader(bpsPaLit->GetBufferPointer(), bpsPaLit->GetBufferSize(), nullptr, psLwPolyAxisLit_.GetAddressOf());
        if (FAILED(hr))
            return false;
        hr = device_->CreateComputeShader(bcsBlkWl->GetBufferPointer(), bcsBlkWl->GetBufferSize(), nullptr, csLwBlockDepthWorklist_.GetAddressOf());
        if (FAILED(hr))
            return false;
        hr = device_->CreateComputeShader(bcsBlkCWl->GetBufferPointer(), bcsBlkCWl->GetBufferSize(), nullptr, csLwBlockColorWorklist_.GetAddressOf());
        if (FAILED(hr))
            return false;
        hr = device_->CreateComputeShader(bcsBlkSpWl->GetBufferPointer(), bcsBlkSpWl->GetBufferSize(), nullptr, csLwBlockSplatWorklist_.GetAddressOf());
        if (FAILED(hr))
            return false;
        hr = device_->CreateVertexShader(bvsBlkPt->GetBufferPointer(), bvsBlkPt->GetBufferSize(), nullptr, vsLwBlockPoint_.GetAddressOf());
        if (FAILED(hr))
            return false;
        hr = device_->CreatePixelShader(bpsBlkPt->GetBufferPointer(), bpsBlkPt->GetBufferSize(), nullptr, psLwBlockPoint_.GetAddressOf());
        if (FAILED(hr))
            return false;
        hr = device_->CreateVertexShader(bvsR->GetBufferPointer(), bvsR->GetBufferSize(), nullptr, vsLwResolve_.GetAddressOf());
        if (FAILED(hr))
            return false;
        hr = device_->CreatePixelShader(bpsR2->GetBufferPointer(), bpsR2->GetBufferSize(), nullptr, psLwResolveTwoPass_.GetAddressOf());
        if (FAILED(hr))
            return false;

        // CBs.
        D3D11_BUFFER_DESC bd = {};
        bd.Usage = D3D11_USAGE_DYNAMIC;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        // CBLwFrame: 64(vp) + 16(cam+near) + 16(light+sun) + 16(amb+expo)
        //          + 16(fog) + 16(heightFog+shadowEn) + 16(shadowBias+colorize+ambient)
        //          + 64(sunVP) + 16(mode+pad) = 240 B
        bd.ByteWidth = 240;
        if (FAILED(device_->CreateBuffer(&bd, nullptr, cbLwFrame_.GetAddressOf())))
            return false;
        // CBLwLod: uint + float + uint2 = 16 bytes.
        bd.ByteWidth = 16;
        if (FAILED(device_->CreateBuffer(&bd, nullptr, cbLwLod_.GetAddressOf())))
            return false;
        // CBLwBounds: 2 * (float3 + float pad) = 32 bytes.
        bd.ByteWidth = 32;
        if (FAILED(device_->CreateBuffer(&bd, nullptr, cbLwBounds_.GetAddressOf())))
            return false;
        // CBGodray: 64 bytes (sunNdc+flags / halfScreenUV+facing+pad / sunUV+pad / tint+strength)
        bd.ByteWidth = 64;
        if (FAILED(device_->CreateBuffer(&bd, nullptr, cbGodray_.GetAddressOf())))
            return false;
        // CBLwCS: 32 bytes (vwSize+count+pad / fadeStart+fadeEnd+pad)
        bd.ByteWidth = 32;
        if (FAILED(device_->CreateBuffer(&bd, nullptr, cbLwCS_.GetAddressOf())))
            return false;
    }

    return true;
}

bool Renderer::CreatePipelineState()
{
    D3D11_BUFFER_DESC bd = {};
    bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    bd.ByteWidth = (sizeof(CBPerFrame) + 15) & ~15;
    if (FAILED(device_->CreateBuffer(&bd, nullptr, cbPerFrame_.GetAddressOf())))
        return false;

    D3D11_RASTERIZER_DESC rd = {};
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_BACK;
    rd.FrontCounterClockwise = FALSE;
    rd.DepthClipEnable = TRUE;
    if (FAILED(device_->CreateRasterizerState(&rd, rsSolid_.GetAddressOf())))
        return false;

    D3D11_RASTERIZER_DESC rdNo = rd;
    rdNo.CullMode = D3D11_CULL_NONE;
    if (FAILED(device_->CreateRasterizerState(&rdNo, rsNoCull_.GetAddressOf())))
        return false;

    // Shadow rasterizer: depth bias + slope-scaled bias to kill acne. Two
    // variants — toggle between cull-back (default) and cull-front from the UI.
    // Point splats are 1-pixel primitives so the cull mode has no effect on
    // them; the toggle exists for future polygon shadow casters.
    D3D11_RASTERIZER_DESC rdShB = rd;
    rdShB.CullMode = D3D11_CULL_BACK;
    rdShB.DepthBias = 4;
    rdShB.SlopeScaledDepthBias = 2.0f;
    rdShB.DepthClipEnable = TRUE;
    if (FAILED(device_->CreateRasterizerState(&rdShB, rsShadowBack_.GetAddressOf())))
        return false;
    D3D11_RASTERIZER_DESC rdShF = rdShB;
    rdShF.CullMode = D3D11_CULL_FRONT;
    if (FAILED(device_->CreateRasterizerState(&rdShF, rsShadowFront_.GetAddressOf())))
        return false;

    // Comparison sampler for hardware PCF.
    D3D11_SAMPLER_DESC ssd = {};
    ssd.Filter = D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
    ssd.AddressU = ssd.AddressV = ssd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    ssd.ComparisonFunc = D3D11_COMPARISON_GREATER_EQUAL; // reverse-Z shadow
    ssd.MinLOD = 0;
    ssd.MaxLOD = D3D11_FLOAT32_MAX;
    if (FAILED(device_->CreateSamplerState(&ssd, shadowSamp_.GetAddressOf())))
        return false;

    D3D11_DEPTH_STENCIL_DESC dd = {};
    dd.DepthEnable = TRUE;
    dd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    dd.DepthFunc = D3D11_COMPARISON_GREATER;
    if (FAILED(device_->CreateDepthStencilState(&dd, dsTest_.GetAddressOf())))
        return false;

    D3D11_DEPTH_STENCIL_DESC ddNo = {};
    ddNo.DepthEnable = FALSE;
    ddNo.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
    ddNo.DepthFunc = D3D11_COMPARISON_ALWAYS;
    if (FAILED(device_->CreateDepthStencilState(&ddNo, dsAlways_.GetAddressOf())))
        return false;

    // Alpha-over: src=ALPHA, dst=INV_ALPHA. Alpha=0 -> destination preserved.
    D3D11_BLEND_DESC bsa = {};
    bsa.RenderTarget[0].BlendEnable = TRUE;
    bsa.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    bsa.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    bsa.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    bsa.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    bsa.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    bsa.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    bsa.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    if (FAILED(device_->CreateBlendState(&bsa, bsAlphaOver_.GetAddressOf())))
        return false;

    // Always-write depth (composite sets SV_Depth unconditionally).
    {
        D3D11_DEPTH_STENCIL_DESC dde = {};
        dde.DepthEnable = TRUE;
        dde.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
        dde.DepthFunc = D3D11_COMPARISON_ALWAYS;
        dde.StencilEnable = FALSE;
        if (FAILED(device_->CreateDepthStencilState(&dde, dsAlwaysWrite_.GetAddressOf())))
            return false;
    }
    return true;
}

void Renderer::Resize(uint32_t w, uint32_t h)
{
    if (!swap_ || (w == width_ && h == height_) || w == 0 || h == 0)
        return;
    ctx_->OMSetRenderTargets(0, nullptr, nullptr);
    rtv_.Reset();
    dsv_.Reset();
    depthTex_.Reset();
    swap_->ResizeBuffers(0, w, h, DXGI_FORMAT_UNKNOWN,
                         tearingSupported_ ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0);
    width_ = w;
    height_ = h;
    CreateRenderTargets();
}

void Renderer::TryHotReloadShaders()
{
    auto mtimeOf = [](const char* path) -> uint64_t
    {
        HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE)
            return 0;
        FILETIME ft = {};
        GetFileTime(h, nullptr, nullptr, &ft);
        CloseHandle(h);
        return ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    };
    uint64_t m1 = mtimeOf("shaders/voxel.hlsl");
    uint64_t m2 = mtimeOf("shaders/lodworld.hlsl");
    uint64_t m3 = mtimeOf("shaders/shading.hlsli");
    uint64_t m4 = mtimeOf("shaders/postfx.hlsl");
    uint64_t mtime = m1;
    if (m2 > mtime)
        mtime = m2;
    if (m3 > mtime)
        mtime = m3;
    if (m4 > mtime)
        mtime = m4;
    if (mtime == 0)
        return;
    if (mtime == shaderMtime_)
        return;
    if (shaderMtime_ == 0)
    {
        shaderMtime_ = mtime;
        return;
    }
    std::fprintf(stderr, "[HotReload] shaders changed - recompiling\n");
    OutputDebugStringA("[HotReload] shaders changed - recompiling\n");
    shaderReloading_ = true;
    bool ok = CreateShaders();
    shaderReloading_ = false;
    if (ok)
    {
        std::fprintf(stderr, "[HotReload] success\n");
        OutputDebugStringA("[HotReload] success\n");
    }
    else
    {
        std::fprintf(stderr, "[HotReload] FAILED - kept previous shaders\n");
        OutputDebugStringA("[HotReload] FAILED - kept previous shaders\n");
    }
    shaderMtime_ = mtime;
}

void Renderer::BeginFrame(float clear[4], bool skipClear)
{
    TryHotReloadShaders();
    lastClear_[0] = clear[0];
    lastClear_[1] = clear[1];
    lastClear_[2] = clear[2];
    lastClear_[3] = clear[3];
    ID3D11RenderTargetView* rtvs[] = {rtv_.Get()};
    ctx_->OMSetRenderTargets(1, rtvs, dsv_.Get());
    if (!skipClear)
    {
        ctx_->ClearRenderTargetView(rtv_.Get(), clear);
    }
    ctx_->ClearDepthStencilView(dsv_.Get(), D3D11_CLEAR_DEPTH, 0.0f, 0);

    D3D11_VIEWPORT vp = {};
    vp.Width = (float)width_;
    vp.Height = (float)height_;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    ctx_->RSSetViewports(1, &vp);
}

void Renderer::FillCbPerFrame(const Camera& cam,
                              const DrawSceneParams& args,
                              const hlslpp::float4x4& vp,
                              float jitterNdcX, float jitterNdcY,
                              const float sceneOrigin[3],
                              const float sceneSpan[3],
                              float colorizeClusters,
                              int gridSize,
                              const float* sunVPstore16OrNull,
                              bool sunShadowsOn,
                              uint32_t shadowMapSize)
{
    D3D11_MAPPED_SUBRESOURCE m;
    ctx_->Map(cbPerFrame_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m);
    CBPerFrame cb{};
    hlslpp::store(cb.viewProj, vp);
    hlslpp::store(cb.camPos, cam.position);
    cb.mode = (float)(int)args.mode;
    cb.lightDir[0] = args.sunDir[0];
    cb.lightDir[1] = args.sunDir[1];
    cb.lightDir[2] = args.sunDir[2];
    cb.ambient = 0.7f;
    hlslpp::float3 fwd = cam.forward();
    hlslpp::float3 pn = -fwd;
    hlslpp::store(cb.pointNormal, pn);
    cb._pad = 0.0f;
    hlslpp::float4x4 invVp = hlslpp::inverse(vp);
    hlslpp::store(cb.invViewProj, invVp);
    cb.screenW = (float)width_;
    cb.screenH = (float)height_;
    cb._pad2[0] = (float)args.splatRadius;
    cb._pad2[1] = 0;
    hlslpp::float3 right = cam.right();
    hlslpp::float3 upVec = hlslpp::cross(fwd, right);
    hlslpp::store(cb.camRight, right);
    hlslpp::store(cb.camUp, upVec);
    hlslpp::store(cb.camForward, fwd);
    cb._pad3 = cb._pad4 = 0;
    cb.tanHalfFovY = tanf(cam.fovDeg * 3.14159265358979f / 180.0f * 0.5f);
    cb.fogColor[0] = args.fogColor[0];
    cb.fogColor[1] = args.fogColor[1];
    cb.fogColor[2] = args.fogColor[2];
    cb.fogDensity = args.fogDensity;
    cb.heightFogDensity = args.heightFogDensity;
    cb.heightFogFalloff = args.heightFogFalloff;
    cb.heightFogStart = args.heightFogStart;
    cb._padHF = 0;
    cb.sceneOrigin[0] = sceneOrigin[0];
    cb.sceneOrigin[1] = sceneOrigin[1];
    cb.sceneOrigin[2] = sceneOrigin[2];
    cb.nearZ = cam.nearZ;
    cb.sceneSpan[0] = sceneSpan[0];
    cb.sceneSpan[1] = sceneSpan[1];
    cb.sceneSpan[2] = sceneSpan[2];
    cb._pad6 = 0;
    for (int i = 0; i < 16; ++i)
        cb.prevViewProj[i] = taaPrevVP_[i];
    cb.jitter[0] = jitterNdcX;
    cb.jitter[1] = jitterNdcY;
    cb._pad7[0] = cb._pad7[1] = 0;
    for (int i = 0; i < 16; ++i)
        cb.sunViewProj[i] = (sunVPstore16OrNull && sunShadowsOn) ? sunVPstore16OrNull[i] : 0.0f;
    cb.shadowBias = args.shadowBias;
    cb.shadowMapSize = (float)shadowMapSize;
    cb.shadowEnable = sunShadowsOn ? 1.0f : 0.0f;
    cb.sunIntensity = args.sunIntensity;
    cb.exposure = args.exposure;
    cb.roughness = args.roughness;
    cb.colorizeClusters = colorizeClusters;
    cb.gridSize = (float)std::max(1, gridSize);
    {
        float W = (float)std::max(1u, width_);
        float H = (float)std::max(1u, height_);
        cb.invScreenSize[0] = 1.0f / W;
        cb.invScreenSize[1] = 1.0f / H;
        cb.aspect = W / H;
        cb.invAspect = H / W;
        cb.aspectTanFov = cb.aspect * cb.tanHalfFovY;
    }
    cb._padPC[0] = cb._padPC[1] = cb._padPC[2] = 0.0f;
    memcpy(m.pData, &cb, sizeof(cb));
    ctx_->Unmap(cbPerFrame_.Get(), 0);
}

// Minimal Phase-1 LW renderer. LOD0 only. Per-chunk frustum cull. Renders
// LW points directly to the main RT via `psmain_lw_debug` (flat-shaded). No
// splat dilation / shading yet — that wires in once data flow is validated.
void Renderer::DrawLwScene(const Camera& cam, const DrawSceneParams& args)
{
    MICROPROFILE_SCOPEI("CPU", "DrawLwScene", 0xff60c060);
    MICROPROFILE_SCOPEGPUI("DrawLwScene", 0xff60c060);
    if (!lwHasWorld_)
        return;

    // ---- View / proj / viewProj (with optional TAA jitter) ----
    const bool postEnabled = (taaSceneRtv_ && psPost_);
    auto halton = [](uint32_t i, uint32_t b)
    {
        float f = 1.0f, r = 0.0f;
        while (i > 0)
        {
            f /= (float)b;
            r += f * (float)(i % b);
            i /= b;
        }
        return r;
    };
    float jitterNdcX = 0.0f, jitterNdcY = 0.0f;
    if (args.taa)
    {
        uint32_t k = (taaFrame_ % 16u) + 1u;
        jitterNdcX = (halton(k, 2) - 0.5f) * 2.0f / (float)width_;
        jitterNdcY = (halton(k, 3) - 0.5f) * 2.0f / (float)height_;
    }
    hlslpp::float4x4 v = cam.view();
    float aspect = (float)width_ / (float)std::max(1u, height_);
    hlslpp::float4x4 p = cam.proj(aspect);
    hlslpp::float4x4 vpUnjittered = hlslpp::mul(v, p);
    if (args.taa)
    {
        float pStore[16];
        hlslpp::store(pStore, p);
        pStore[2 * 4 + 0] = jitterNdcX;
        pStore[2 * 4 + 1] = jitterNdcY;
        p = hlslpp::float4x4(
            pStore[0], pStore[1], pStore[2], pStore[3],
            pStore[4], pStore[5], pStore[6], pStore[7],
            pStore[8], pStore[9], pStore[10], pStore[11],
            pStore[12], pStore[13], pStore[14], pStore[15]);
    }
    hlslpp::float4x4 vp = hlslpp::mul(v, p);

    // Helper: write per-chunk LOD CB (slot + drawBase + halfExt + lodIdx).
    auto setLodCbForLod = [&](int L, uint32_t slot, uint32_t drawBase)
    {
        D3D11_MAPPED_SUBRESOURCE mm;
        ctx_->Map(cbLwLod_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mm);
        struct
        {
            uint32_t lodIdx;
            float halfExt;
            uint32_t slot;
            uint32_t drawBase;
        } cbl;
        cbl.lodIdx = (uint32_t)L;
        cbl.halfExt = 0.5f * (float)(1u << L);
        cbl.slot = slot;
        cbl.drawBase = drawBase;
        memcpy(mm.pData, &cbl, sizeof(cbl));
        ctx_->Unmap(cbLwLod_.Get(), 0);
    };
    // Helper: write a given viewproj + shading state into cbLwFrame_.
    // Lambda captures of sunVPstore + shadowEnable happen at call sites which
    // pass them in (init values are zero -> shadow stub returns 1.0 -> off).
    auto mapCbLwFrame = [&](const float vp16[16], const float sunVP16[16], float shadowEnable)
    {
        D3D11_MAPPED_SUBRESOURCE mm;
        ctx_->Map(cbLwFrame_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mm);
        struct CbLw
        {
            float vp[16];
            float camPos[3];
            float nearZ;
            float lightDir[3];
            float sunIntensity;
            float ambientColor[3];
            float exposure;
            float fogColor[3];
            float fogDensity;
            float heightFogDensity, heightFogFalloff, heightFogStart, shadowEnable;
            float shadowBias, shadowMapSize, colorizeClusters, ambient;
            float sunViewProj[16];
            uint32_t mode;
            uint32_t _padFrame[3];
        } cb;
        for (int i = 0; i < 16; ++i)
            cb.vp[i] = vp16[i];
        float cp[3];
        hlslpp::store(cp, cam.position);
        cb.camPos[0] = cp[0];
        cb.camPos[1] = cp[1];
        cb.camPos[2] = cp[2];
        cb.nearZ = cam.nearZ;
        cb.lightDir[0] = args.sunDir[0];
        cb.lightDir[1] = args.sunDir[1];
        cb.lightDir[2] = args.sunDir[2];
        cb.sunIntensity = args.sunIntensity;
        cb.ambientColor[0] = 0.35f;
        cb.ambientColor[1] = 0.40f;
        cb.ambientColor[2] = 0.50f;
        cb.exposure = args.exposure;
        cb.fogColor[0] = args.fogColor[0];
        cb.fogColor[1] = args.fogColor[1];
        cb.fogColor[2] = args.fogColor[2];
        cb.fogDensity = args.fogDensity;
        cb.heightFogDensity = args.heightFogDensity;
        cb.heightFogFalloff = args.heightFogFalloff;
        cb.heightFogStart = args.heightFogStart;
        cb.shadowEnable = shadowEnable;
        cb.shadowBias = args.shadowBias;
        cb.shadowMapSize = (float)((args.shadowMapSize > 0) ? args.shadowMapSize : 2048);
        cb.colorizeClusters = 0.0f;
        cb.ambient = 0.7f; // sun multiplier (matches voxel.hlsl gAmbient default)
        for (int i = 0; i < 16; ++i)
            cb.sunViewProj[i] = sunVP16[i];
        cb.mode = (uint32_t)args.mode;
        cb._padFrame[0] = cb._padFrame[1] = cb._padFrame[2] = 0;
        memcpy(mm.pData, &cb, sizeof(cb));
        ctx_->Unmap(cbLwFrame_.Get(), 0);
    };
    float vpStoreMain[16];
    hlslpp::store(vpStoreMain, vp);

    // ---- Build sun VP + ensure shadow textures (if shadows enabled) ----
    float sunVPstore[16] = {};
    bool sunShadowsOn = args.sunShadows && lwHasWorld_;
    if (sunShadowsOn)
    {
        const uint32_t reqSize = (uint32_t)std::max(64, args.shadowMapSize);
        if (shadowSize_ != reqSize)
        {
            shadowTex_.Reset();
            shadowDsv_.Reset();
            shadowSrv_.Reset();
            shadowFilledTex_.Reset();
            shadowFilledUav_.Reset();
            shadowFilledSrv_.Reset();
            D3D11_TEXTURE2D_DESC td = {};
            td.Width = reqSize;
            td.Height = reqSize;
            td.MipLevels = 1;
            td.ArraySize = 1;
            td.Format = DXGI_FORMAT_R32_TYPELESS;
            td.SampleDesc.Count = 1;
            td.Usage = D3D11_USAGE_DEFAULT;
            td.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
            if (FAILED(device_->CreateTexture2D(&td, nullptr, shadowTex_.GetAddressOf())))
                sunShadowsOn = false;
            if (sunShadowsOn)
            {
                D3D11_DEPTH_STENCIL_VIEW_DESC dvd = {};
                dvd.Format = DXGI_FORMAT_D32_FLOAT;
                dvd.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
                device_->CreateDepthStencilView(shadowTex_.Get(), &dvd, shadowDsv_.GetAddressOf());
                D3D11_SHADER_RESOURCE_VIEW_DESC svd = {};
                svd.Format = DXGI_FORMAT_R32_FLOAT;
                svd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
                svd.Texture2D.MipLevels = 1;
                device_->CreateShaderResourceView(shadowTex_.Get(), &svd, shadowSrv_.GetAddressOf());
                D3D11_TEXTURE2D_DESC tdf = {};
                tdf.Width = reqSize;
                tdf.Height = reqSize;
                tdf.MipLevels = 1;
                tdf.ArraySize = 1;
                tdf.Format = DXGI_FORMAT_R32_FLOAT;
                tdf.SampleDesc.Count = 1;
                tdf.Usage = D3D11_USAGE_DEFAULT;
                tdf.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
                device_->CreateTexture2D(&tdf, nullptr, shadowFilledTex_.GetAddressOf());
                device_->CreateUnorderedAccessView(shadowFilledTex_.Get(), nullptr, shadowFilledUav_.GetAddressOf());
                device_->CreateShaderResourceView(shadowFilledTex_.Get(), nullptr, shadowFilledSrv_.GetAddressOf());
                shadowSize_ = reqSize;
            }
        }
    }
    if (sunShadowsOn)
    {
        hlslpp::float3 sceneMin((float)lwWorld_.worldAabbMin[0],
                                (float)lwWorld_.worldAabbMin[1],
                                (float)lwWorld_.worldAabbMin[2]);
        hlslpp::float3 sceneSpan3((float)(lwWorld_.worldAabbMax[0] - lwWorld_.worldAabbMin[0]),
                                  (float)(lwWorld_.worldAabbMax[1] - lwWorld_.worldAabbMin[1]),
                                  (float)(lwWorld_.worldAabbMax[2] - lwWorld_.worldAabbMin[2]));
        hlslpp::float3 center = sceneMin + sceneSpan3 * 0.5f;
        float radius = 0.5f * (float)hlslpp::length(sceneSpan3);
        if (radius < 1.0f)
            radius = 1.0f;
        hlslpp::float3 sd = hlslpp::normalize(
            hlslpp::float3(args.sunDir[0], args.sunDir[1], args.sunDir[2]));
        hlslpp::float3 eye = center + sd * (radius * 2.0f);
        float sy = (float)sd.y;
        hlslpp::float3 upRef = (std::fabs(sy) > 0.99f)
                                   ? hlslpp::float3(0.0f, 0.0f, 1.0f)
                                   : hlslpp::float3(0.0f, 1.0f, 0.0f);
        hlslpp::float4x4 sunView = hlslpp::float4x4::look_at(eye, center, upRef);
        const float wO = radius * 2.5f, hO = radius * 2.5f;
        const float zn = 0.0f, zf = radius * 4.0f;
        hlslpp::float4x4 sunProj(
            2.0f / wO, 0, 0, 0,
            0, 2.0f / hO, 0, 0,
            0, 0, -1.0f / (zf - zn), 0,
            0, 0, zf / (zf - zn), 1);
        hlslpp::float4x4 sunVP = hlslpp::mul(sunView, sunProj);
        hlslpp::store(sunVPstore, sunVP);
    }
    for (int i = 0; i < 16; ++i)
        shadowVP_[i] = sunVPstore[i];
    shadowBias_ = args.shadowBias;
    shadowEnable_ = sunShadowsOn ? 1.0f : 0.0f;

    // ---- Populate cbPerFrame_ (used by splat dilate CS + composite) ----
    {
        MICROPROFILE_SCOPEGPUI("LW/CB", 0xff60c0c0);
        const float sceneOrigin[3] = {
            (float)lwWorld_.worldAabbMin[0],
            (float)lwWorld_.worldAabbMin[1],
            (float)lwWorld_.worldAabbMin[2],
        };
        const float sceneSpan[3] = {
            (float)(lwWorld_.worldAabbMax[0] - lwWorld_.worldAabbMin[0]),
            (float)(lwWorld_.worldAabbMax[1] - lwWorld_.worldAabbMin[1]),
            (float)(lwWorld_.worldAabbMax[2] - lwWorld_.worldAabbMin[2]),
        };
        const float colorizeClustersFlag = (args.mode == ShadingMode::LodViz) ? 1.0f : 0.0f;
        FillCbPerFrame(cam, args, vp, jitterNdcX, jitterNdcY,
                       sceneOrigin, sceneSpan,
                       colorizeClustersFlag, 1,
                       sunShadowsOn ? sunVPstore : nullptr, sunShadowsOn,
                       shadowSize_);
    }

    // ---- Shadow caster pass: render LW points to shadowDsv_, depth-only ----
    bool shadowSkipped = false;
    if (sunShadowsOn && shadowDsv_)
    {
        const float epsDir = 1e-4f;
        bool dirChanged =
            std::fabs(args.sunDir[0] - lastSunDir_[0]) > epsDir || std::fabs(args.sunDir[1] - lastSunDir_[1]) > epsDir || std::fabs(args.sunDir[2] - lastSunDir_[2]) > epsDir;
        bool sizeChanged = (shadowSize_ != lastShadowSize_);
        bool cullChanged = (args.shadowCullFront != lastShadowCullFront_);
        bool lodChanged = (args.shadowLod != lastShadowLod_);
        bool blurChanged = (args.shadowBlur != lastShadowBlur_);
        if (!args.shadowForceRebuild && !shadowMapDirty_ && !dirChanged && !sizeChanged && !cullChanged && !lodChanged && !blurChanged)
        {
            shadowSkipped = true; // reuse cached shadow map this frame
        }
    }
    if (sunShadowsOn && shadowDsv_ && !shadowSkipped)
    {
        MICROPROFILE_SCOPEGPUI("LW/Shadow", 0xff909090);
        mapCbLwFrame(sunVPstore, sunVPstore, sunShadowsOn ? 1.0f : 0.0f); // VS projects via sunVP

        ID3D11RenderTargetView* nullRtv[] = {nullptr};
        ctx_->OMSetRenderTargets(1, nullRtv, shadowDsv_.Get());
        ctx_->ClearDepthStencilView(shadowDsv_.Get(), D3D11_CLEAR_DEPTH, 0.0f, 0);
        D3D11_VIEWPORT svp = {0, 0, (float)shadowSize_, (float)shadowSize_, 0.0f, 1.0f};
        ctx_->RSSetViewports(1, &svp);
        ctx_->RSSetState(args.shadowCullFront ? rsShadowFront_.Get() : rsShadowBack_.Get());
        ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
        ctx_->IASetInputLayout(nullptr);
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
        ID3D11Buffer* nullVbs[] = {nullptr};
        UINT zs = 0;
        ctx_->IASetVertexBuffers(0, 1, nullVbs, &zs, &zs);
        ctx_->VSSetShader(vsLwPoints_.Get(), nullptr, 0);
        ctx_->PSSetShader(nullptr, nullptr, 0);
        ID3D11Buffer* castCbs[] = {cbLwFrame_.Get(), cbLwLod_.Get()};
        ctx_->VSSetConstantBuffers(0, 2, castCbs);

        // Auto-LOD: coarser when a LOD-voxel projects to >= shadow texel.
        int castLod = args.shadowLod;
        if (castLod < 0)
        {
            float orthoWorldW = std::max((float)(lwWorld_.worldAabbMax[0] - lwWorld_.worldAabbMin[0]),
                                         (float)(lwWorld_.worldAabbMax[2] - lwWorld_.worldAabbMin[2])) *
                                1.25f;
            float pxWorld = orthoWorldW / std::max(1.0f, (float)shadowSize_);
            if (pxWorld >= 16.0f)
                castLod = 4;
            else if (pxWorld >= 8.0f)
                castLod = 3;
            else if (pxWorld >= 4.0f)
                castLod = 2;
            else if (pxWorld >= 2.0f)
                castLod = 1;
            else
                castLod = 0;
        }
        if (castLod < 0)
            castLod = 0;
        if (castLod >= lw::kLodCount)
            castLod = lw::kLodCount - 1;

        const LwGpu& gC = lwGpu_[castLod];
        const lw::LODWorld& lwC = lwWorld_.lods[castLod];
        if (gC.slotCount > 0 && gC.pointSrv)
        {
            ID3D11ShaderResourceView* vsSrvs[] = {
                gC.pointSrv.Get(), gC.chunkInfoSrv.Get(), gC.paletteSrv.Get()};
            ctx_->VSSetShaderResources(0, 3, vsSrvs);
            for (uint32_t i = 0; i < gC.slotCount; ++i)
            {
                const lw::RuntimeChunk& rc = lwC.chunks[i];
                if (rc.poolCount == 0)
                    continue;
                setLodCbForLod(castLod, rc.slotIdx, 0);
                ctx_->Draw(rc.poolCount, 0);
            }
        }

        // Detach DSV so shadowSrv_ can be sampled downstream.
        ID3D11RenderTargetView* nullRtv2[] = {nullptr};
        ctx_->OMSetRenderTargets(1, nullRtv2, nullptr);

        // Optional shadow blur fill CS: empty texels = avg of non-empty neighbours.
        if (args.shadowBlur && csShadowBlur_ && shadowFilledUav_)
        {
            MICROPROFILE_SCOPEGPUI("LW/ShadowBlur", 0xff80a0a0);
            ctx_->CSSetShader(csShadowBlur_.Get(), nullptr, 0);
            ID3D11Buffer* csCbs[] = {cbPerFrame_.Get()};
            ctx_->CSSetConstantBuffers(0, 1, csCbs);
            ID3D11ShaderResourceView* srcSrv[] = {shadowSrv_.Get()};
            ctx_->CSSetShaderResources(7, 1, srcSrv);
            ID3D11UnorderedAccessView* dstUav[] = {shadowFilledUav_.Get()};
            UINT initc[] = {0};
            ctx_->CSSetUnorderedAccessViews(3, 1, dstUav, initc);
            UINT bgx = (shadowSize_ + 7) / 8;
            UINT bgy = (shadowSize_ + 7) / 8;
            ctx_->Dispatch(bgx, bgy, 1);
            ID3D11ShaderResourceView* nullSrv[] = {nullptr};
            ctx_->CSSetShaderResources(7, 1, nullSrv);
            ID3D11UnorderedAccessView* nullUav[] = {nullptr};
            ctx_->CSSetUnorderedAccessViews(3, 1, nullUav, initc);
            ctx_->CSSetShader(nullptr, nullptr, 0);
        }

        // Mark cache valid until inputs change.
        lastSunDir_[0] = args.sunDir[0];
        lastSunDir_[1] = args.sunDir[1];
        lastSunDir_[2] = args.sunDir[2];
        lastShadowSize_ = shadowSize_;
        lastShadowCullFront_ = args.shadowCullFront;
        lastShadowLod_ = args.shadowLod;
        lastShadowBlur_ = args.shadowBlur;
        shadowMapDirty_ = false;

        // Restore main vp into cbLwFrame_ for subsequent passes.
        mapCbLwFrame(vpStoreMain, sunVPstore, sunShadowsOn ? 1.0f : 0.0f);
    }
    else
    {
        // No shadow caster (off, or cached) — still need main vp in cbLwFrame_.
        mapCbLwFrame(vpStoreMain, sunVPstore, sunShadowsOn ? 1.0f : 0.0f);
    }

    // ---- Splat RT setup (color + mask) + splat DSV ----
    D3D11_VIEWPORT vp_d3d = {0, 0, (float)width_, (float)height_, 0.0f, 1.0f};
    ctx_->RSSetViewports(1, &vp_d3d);
    ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
    ctx_->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFFu);
    ctx_->RSSetState(rsSolid_.Get());

    // Splat clears only needed if Splat tech actually in use this frame.
    const bool needSplat = (args.tech == RenderTech::Splat) || (args.techFar == RenderTech::Splat);
    if (needSplat)
    {
        float clr[4] = {lastClear_[0], lastClear_[1], lastClear_[2], 0.0f};
        ctx_->ClearRenderTargetView(splatColorRtv_.Get(), clr);
        const float zeroClr[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        ctx_->ClearRenderTargetView(splatMaskRtv_.Get(), zeroClr);
        ctx_->ClearDepthStencilView(splatDsv_.Get(), D3D11_CLEAR_DEPTH, 0.0f, 0);
    }
    ID3D11RenderTargetView* mrt[] = {splatColorRtv_.Get(), splatMaskRtv_.Get()};
    ctx_->OMSetRenderTargets(2, mrt, splatDsv_.Get());

    // ---- Shader setup (per-LOD sub-passes pick splat vs polyaxis VS) ----
    ctx_->IASetInputLayout(nullptr);
    ID3D11Buffer* nullVbs[] = {nullptr};
    UINT vbStride = 0, vbOff = 0;
    ctx_->IASetVertexBuffers(0, 1, nullVbs, &vbStride, &vbOff);
    ctx_->PSSetShader(psLwSplatAlbedo_.Get(), nullptr, 0);
    ID3D11Buffer* vsCbs[] = {cbLwFrame_.Get(), cbLwLod_.Get()};
    ctx_->VSSetConstantBuffers(0, 2, vsCbs);
    ctx_->PSSetConstantBuffers(0, 2, vsCbs);

    // ---- Frustum planes ----
    float M[16];
    hlslpp::store(M, vp);
    float planes[6][4];
    ExtractFrustumPlanes(M, planes);
    auto cullAabb = [&](float mnx, float mny, float mnz, float mxx, float mxy, float mxz) -> bool
    {
        // Skip the far plane (index 5) — reverse-Z infinite-far makes it
        // ill-defined and rejects valid distant chunks. Side + near planes only.
        for (int pi = 0; pi < 5; ++pi)
        {
            float a = planes[pi][0], b = planes[pi][1], c = planes[pi][2], d = planes[pi][3];
            float px = a >= 0 ? mxx : mnx;
            float py = b >= 0 ? mxy : mny;
            float pz = c >= 0 ? mxz : mnz;
            if (a * px + b * py + c * pz + d < 0.0f)
                return true;
        }
        return false;
    };

    // ---- Top-down LOD traversal (with per-cluster straddle path) ----
    MICROPROFILE_SCOPEI("CPU", "LW/Traverse", 0xff70d070);
    float camP[3];
    hlslpp::store(camP, cam.position);
    const float fovRad = cam.fovDeg * 3.14159265358979f / 180.0f;
    const float focalPx = (float)height_ / (2.0f * tanf(fovRad * 0.5f));
    const float lodScaleUi = std::max(args.pointLodScale, 0.01f);
    const float thresh = 1.0f / lodScaleUi;

    auto desiredLodForDist = [&](float dist) -> int
    {
        if (dist < 1.0f)
            return 0;
        for (int L = 0; L < lw::kLodCount; ++L)
        {
            float ppv = focalPx * (float)(1u << L) / dist;
            if (ppv >= thresh)
                return L;
        }
        return lw::kLodCount - 1;
    };
    // Closest point of AABB to camera (0 if camera inside).
    auto nearAabbDist = [&](float mnx, float mny, float mnz, float mxx, float mxy, float mxz) -> float
    {
        float dx = (camP[0] < mnx) ? (mnx - camP[0]) : (camP[0] > mxx) ? (camP[0] - mxx)
                                                                       : 0.0f;
        float dy = (camP[1] < mny) ? (mny - camP[1]) : (camP[1] > mxy) ? (camP[1] - mxy)
                                                                       : 0.0f;
        float dz = (camP[2] < mnz) ? (mnz - camP[2]) : (camP[2] > mxz) ? (camP[2] - mxz)
                                                                       : 0.0f;
        return sqrtf(dx * dx + dy * dy + dz * dz);
    };
    // Farthest point of AABB from camera (always positive).
    auto farAabbDist = [&](float mnx, float mny, float mnz, float mxx, float mxy, float mxz) -> float
    {
        float ax = std::max(std::fabs(mnx - camP[0]), std::fabs(mxx - camP[0]));
        float ay = std::max(std::fabs(mny - camP[1]), std::fabs(mxy - camP[1]));
        float az = std::max(std::fabs(mnz - camP[2]), std::fabs(mxz - camP[2]));
        return sqrtf(ax * ax + ay * ay + az * az);
    };

    struct DrawItem
    {
        uint32_t slot;
        uint32_t drawBase;  // splat/poly: cluster.pointFirst (vertex offset)
        uint32_t drawCount; // splat/poly: cluster.numPoints  (vertex count)
        uint32_t blockFirst = 0;       // PointCS_Block: offset relative to rc.blockBase
        uint32_t blockCount = 0;       // PointCS_Block: blocks to dispatch this item
        uint32_t blockPointFirst = 0;  // HW point draw: offset relative to rc.blockPointBase
        uint32_t blockPointCount = 0;  // HW point draw: voxels to draw
    };
    std::vector<DrawItem> drawListSplat[lw::kLodCount];
    std::vector<DrawItem> drawListPoly[lw::kLodCount];
    std::vector<DrawItem> drawListBlock[lw::kLodCount]; // PointCS_Block (2x2x2 blocks)

    // Per-chunk tech pick: distance-based close/far ring. LOD-independent so
    // recursion can't flip a chunk's classification mid-traversal.
    // closeRadius = where a unit-voxel splat covers `splatRadius` screen pixels.
    const float switchPpv = (float)std::max(1, args.splatRadius);
    const float closeRadius = focalPx / switchPpv;
    auto pickListForChunk = [&](int L, float chunkCenterX, float chunkCenterY, float chunkCenterZ) -> std::vector<DrawItem>*
    {
        float dx = camP[0] - chunkCenterX, dy = camP[1] - chunkCenterY, dz = camP[2] - chunkCenterZ;
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        bool isClose = (dist <= closeRadius);
        if (isClose && !args.closeEnabled)
            return nullptr;
        if (!isClose && !args.farEnabled)
            return nullptr;
        RenderTech tech = isClose ? args.tech : args.techFar;
        if (tech == RenderTech::PointCS_Block)
            return &drawListBlock[L];
        bool usePoly = (tech == RenderTech::PolyAxis) || args.lwPolyAxis;
        return usePoly ? &drawListPoly[L] : &drawListSplat[L];
    };

    // Per-cluster recursive walker. Each cluster: cull, compute closest LOD;
    // if finer than its parent chunk's LOD AND child chunk exists, descend
    // into the 8 child clusters covering this cluster's region.
    std::function<void(int, uint32_t, int)> visitCluster =
        [&](int L, uint32_t chunkSlot, int clSlot)
    {
        const lw::LODWorld& lwL = lwWorld_.lods[L];
        const lw::RuntimeChunk& rc = lwL.chunks[chunkSlot];
        const lw::DiskCluster& cl = rc.clusters[clSlot];
        if (cl.numPoints == 0)
            return;

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
        if (cullAabb(mnx, mny, mnz, mxx, mxy, mxz))
            return;

        float distNearC = nearAabbDist(mnx, mny, mnz, mxx, mxy, mxz);
        int desNearC = desiredLodForDist(distNearC);
        (void)farAabbDist;

        // 8 child chunks per parent, picked by parent cluster's octant.
        int oct = (cx_g >> 2) | ((cy_g & 1) << 1) | ((cz_g >> 2) << 2);
        uint32_t childChunkId = rc.childId[oct];

        // Treat "child chunk not resident" (streaming/shell) as no-child so
        // we render this coarser LOD instead of dropping the chunk entirely.
        // Use blockCount (set only after the block stream walk completes) rather
        // than poolCount (which uses a dummy=1 during chunk init). Avoids
        // recursing into a chunk that's been partially populated by the loader
        // thread but has no actual block data yet.
        bool childLoaded = (L > 0) && (childChunkId != lw::kNoChild) && (childChunkId < lwWorld_.lods[L - 1].chunks.size()) && (lwWorld_.lods[L - 1].chunks[childChunkId].blockCount > 0);

        // Recurse only when the cluster's nearest point wants finer than L.
        // Straddle alone (near @ L, far @ L+1) shouldn't recurse — we can't go
        // coarser than L here anyway, so push at L for the whole cluster.
        if (desNearC >= L || !childLoaded || L == 0)
        {
            std::vector<DrawItem>* dl = pickListForChunk(L,
                                                         0.5f * (mnx + mxx), 0.5f * (mny + mxy), 0.5f * (mnz + mxz));
            if (dl)
                dl->push_back({chunkSlot, cl.pointFirst, cl.numPoints,
                               rc.clusterBlockFirst[clSlot],
                               rc.clusterBlockCount[clSlot],
                               rc.clusterPointFirst[clSlot],
                               rc.clusterPointCount[clSlot]});
            return;
        }

        // Recurse: this parent cluster covers 2x2x2 = 8 child clusters.
        // Octant-local indices of parent: (lcx in 0..3, lcy = 0 since
        // kClustersY = 2 -> 1 Y slice per octant, lcz in 0..3). Each parent
        // cluster maps to child cluster grid (2*lcx + dx, dy, 2*lcz + dz).
        int lcx = cx_g & 3;
        int lcz = cz_g & 3;
        for (int dx = 0; dx < 2; ++dx)
            for (int dy = 0; dy < 2; ++dy)
                for (int dz = 0; dz < 2; ++dz)
                {
                    int childCx = 2 * lcx + dx;
                    int childCy = dy;
                    int childCz = 2 * lcz + dz;
                    int childSlot = childCz * (lw::kClustersX * lw::kClustersY) + childCy * lw::kClustersX + childCx;
                    visitCluster(L - 1, childChunkId, childSlot);
                }
    };

    std::function<void(int, uint32_t)> visit = [&](int L, uint32_t slot)
    {
        const lw::LODWorld& lwL = lwWorld_.lods[L];
        if (slot >= lwL.chunks.size())
            return;
        float mnX = lwL.cull.minX[slot], mnY = lwL.cull.minY[slot], mnZ = lwL.cull.minZ[slot];
        float mxX = lwL.cull.maxX[slot], mxY = lwL.cull.maxY[slot], mxZ = lwL.cull.maxZ[slot];
        if (cullAabb(mnX, mnY, mnZ, mxX, mxY, mxZ))
            return;
        const lw::RuntimeChunk& rc = lwL.chunks[slot];
        if (rc.poolCount == 0)
            return;

        // Pick draw list (splat or polyaxis) based on chunk-center ppv vs near/far techs.
        // null = this chunk's tech side disabled; push-sites skip, recursion still tries
        // finer children (they may classify differently and have an enabled tech).
        std::vector<DrawItem>* dl = pickListForChunk(
            L,
            0.5f * (mnX + mxX),
            0.5f * (mnY + mxY),
            0.5f * (mnZ + mxZ));

        float distNear = nearAabbDist(mnX, mnY, mnZ, mxX, mxY, mxZ);
        float distFar = farAabbDist(mnX, mnY, mnZ, mxX, mxY, mxZ);
        int desNear = desiredLodForDist(distNear); // finest LOD wanted anywhere in chunk
        int desFar = desiredLodForDist(distFar);   // coarsest LOD wanted anywhere in chunk

        const bool isBlock = (dl == &drawListBlock[L]);
        (void)isBlock; (void)desFar;
        // Always use per-cluster path so each cluster picks its own LOD/cull.
        // visitCluster handles cull, finer-LOD recursion, and terminal push.

        // ---- Per-cluster path: each cluster makes its own cull + LOD choice. ----
        // Each cluster checked independently. If its closest LOD is finer
        // than the current chunk's LOD AND a child chunk exists, recurse
        // into the 2x2x2 = 8 child clusters that cover this parent cluster.
        // Otherwise draw the parent cluster at this LOD.
        for (int slot_c = 0; slot_c < lw::kClustersPerChunk; ++slot_c)
        {
            const lw::DiskCluster& cl = rc.clusters[slot_c];
            if (cl.numPoints == 0)
                continue;
            visitCluster(L, slot, slot_c);
        }
    };

    // Roots = top LOD chunks. Each is its own subtree.
    const int topL = lw::kLodCount - 1;
    for (uint32_t i = 0; i < (uint32_t)lwWorld_.lods[topL].chunks.size(); ++i)
    {
        visit(topL, i);
    }

    // ---- Issue draws, per-LOD batched (SRVs rebind on LOD change) ----
    MICROPROFILE_SCOPEGPUI("LW/Points", 0xffc0a040);
    uint32_t drawCount = 0;
    uint64_t pointCountTotal = 0;
    uint64_t splatVoxels = 0;

    // Compute rasterizer: clear vis buffer if block tech in use.
    bool anyBlock = false;
    for (int Li = 0; Li < lw::kLodCount; ++Li)
    {
        if (!drawListBlock[Li].empty())
        {
            anyBlock = true;
            break;
        }
    }
    bool anyCS = anyBlock;
    if (anyCS && visDepthUav_ && visColorUav_)
    {
        MICROPROFILE_SCOPEGPUI("LW/PointCS/Clear", 0xff404060);
        ID3D11RenderTargetView* nullRtvsA[] = {nullptr, nullptr};
        ctx_->OMSetRenderTargets(2, nullRtvsA, nullptr);
        uint32_t clearVis[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
        ctx_->ClearUnorderedAccessViewUint(visDepthUav_.Get(), clearVis);
        ctx_->ClearUnorderedAccessViewUint(visColorUav_.Get(), clearVis);
        // Restore splat RTs for the splat draws inside the loop.
        ID3D11RenderTargetView* mrtA[] = {splatColorRtv_.Get(), splatMaskRtv_.Get()};
        ctx_->OMSetRenderTargets(2, mrtA, splatDsv_.Get());
    }
    uint64_t polyVoxels = 0;
    uint64_t triCount = 0;
    uint32_t splatDraws = 0;
    uint32_t polyDraws = 0;
    uint32_t fastDraws = 0; // consecutive draws with same chunk slot
    uint32_t prevSlot = 0xFFFFFFFFu;
    static const uint32_t kLodColors[5] = {
        0xffff6060, 0xffffa030, 0xff60c060, 0xff6098c0, 0xffc060ff};
    // Two-pass LODs that completed pass1 and need pass2 after the per-LOD loop.
    std::vector<int> twoPassLods;
    // PointCS_Block stats — accumulated across LODs.
    uint64_t blockTotalAll = 0;
    uint32_t blockDispatchesAll = 0;
    uint64_t blockTotalPerLod[5] = {};
    for (int L = topL; L >= 0; --L)
    {
        if (drawListSplat[L].empty() && drawListPoly[L].empty() && drawListBlock[L].empty())
            continue;
        const LwGpu& g = lwGpu_[L];
        if (g.slotCount == 0)
            continue;
        // pointSrv may be null in block-only mode; only required by splat/poly paths.
        if (!g.pointSrv && !g.blockPosSrv)
            continue;
        const lw::LODWorld& lwL = lwWorld_.lods[L];
        MICROPROFILE_SCOPEGPUI("LW/Points/LOD", kLodColors[L < 5 ? L : 4]);
        ID3D11ShaderResourceView* vsSrvs[] = {
            g.pointSrv.Get(),
            g.chunkInfoSrv.Get(),
            g.paletteSrv.Get(),
        };
        ctx_->VSSetShaderResources(0, 3, vsSrvs);

        // Splat sub-pass only — PolyAxis runs AFTER dilate + composite so its
        // solid cube triangles don't get treated as splats by csmain_splat.
        // Coalesce contiguous (same chunk, adjacent drawBase+drawCount) items
        // into single draws to avoid per-cluster Map/Unmap on cbLwLod_ + Draw.
        if (!drawListSplat[L].empty())
        {
            ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
            ctx_->VSSetShader(vsLwPoints_.Get(), nullptr, 0);
            std::sort(drawListSplat[L].begin(), drawListSplat[L].end(),
                      [](const DrawItem& a, const DrawItem& b)
                      {
                          if (a.slot != b.slot)
                              return a.slot < b.slot;
                          return a.drawBase < b.drawBase;
                      });
            uint32_t curSlot = 0xFFFFFFFFu, curBase = 0, curCount = 0;
            auto flushSplat = [&]()
            {
                if (curCount == 0)
                    return;
                const lw::RuntimeChunk& rc = lwL.chunks[curSlot];
                setLodCbForLod(L, rc.slotIdx, curBase);
                ctx_->Draw(curCount, 0);
                ++drawCount;
                ++splatDraws;
                if (curSlot == prevSlot)
                    ++fastDraws;
                prevSlot = curSlot;
                splatVoxels += curCount;
                curCount = 0;
            };
            for (const DrawItem& it : drawListSplat[L])
            {
                if (it.slot == curSlot && curBase + curCount == it.drawBase)
                {
                    curCount += it.drawCount;
                }
                else
                {
                    flushSplat();
                    curSlot = it.slot;
                    curBase = it.drawBase;
                    curCount = it.drawCount;
                }
            }
            flushSplat();
        }

        // Helper to write the compute CB for this LOD's dispatches.
        auto writeCbLwCS = [&](uint32_t curCount, float fadeStart = 0.0f, float fadeEnd = 0.0f)
        {
            D3D11_MAPPED_SUBRESOURCE mm;
            ctx_->Map(cbLwCS_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mm);
            struct
            {
                uint32_t w, h, cnt, _pad0;
                float fadeStart, fadeEnd, _pad1, _pad2;
            } cbcs;
            cbcs.w = width_;
            cbcs.h = height_;
            cbcs.cnt = curCount;
            cbcs._pad0 = 0;
            cbcs.fadeStart = fadeStart;
            cbcs.fadeEnd = fadeEnd;
            cbcs._pad1 = cbcs._pad2 = 0.0f;
            memcpy(mm.pData, &cbcs, sizeof(cbcs));
            ctx_->Unmap(cbLwCS_.Get(), 0);
        };

        // PointCS_Block — two-pass single-dispatch worklist pass1.
        if (!drawListBlock[L].empty() && csLwBlockDepthWorklist_)
        {
            // Per-LOD fade: switch-out distance = focalPx * (1<<L) / thresh.
            float switchOut = focalPx * (float)(1u << L) / thresh;
            float fadeStart = 0.75f * switchOut;
            float fadeEnd = switchOut;

            // Bind shared state (SRVs + CBs) once.
            ID3D11RenderTargetView* nullRtvsB[] = {nullptr, nullptr};
            ctx_->OMSetRenderTargets(2, nullRtvsB, nullptr);
            ID3D11ShaderResourceView* nullVsB[3] = {nullptr, nullptr, nullptr};
            ctx_->VSSetShaderResources(0, 3, nullVsB);

            ID3D11ShaderResourceView* csSrvBlk[] = {
                g.pointSrv.Get(), g.chunkInfoSrv.Get(), g.paletteSrv.Get(),
                g.blockPosSrv.Get(), g.blockColSrv.Get()};
            ctx_->CSSetShaderResources(0, 5, csSrvBlk);
            ID3D11Buffer* csCbsB[] = {cbLwFrame_.Get(), cbLwLod_.Get(), nullptr, cbLwCS_.Get()};
            ctx_->CSSetConstantBuffers(0, 4, csCbsB);
            UINT initB[] = {0};

            // Sort by (slot, blockFirst) then coalesce adjacent items with
            // contiguous block ranges in the same chunk. Loader appends cluster
            // blocks in stream order, so consecutive cluster slots usually map
            // to consecutive pool ranges → coalesce collapses most per-cluster
            // items back to per-chunk-region dispatches.
            std::sort(drawListBlock[L].begin(), drawListBlock[L].end(),
                      [](const DrawItem& a, const DrawItem& b)
                      {
                          if (a.slot != b.slot) return a.slot < b.slot;
                          return a.blockFirst < b.blockFirst;
                      });
            {
                auto& dl_ = drawListBlock[L];
                size_t w = 0;
                for (size_t r = 0; r < dl_.size(); ++r)
                {
                    if (w > 0 && dl_[w - 1].slot == dl_[r].slot &&
                        dl_[w - 1].blockFirst + dl_[w - 1].blockCount == dl_[r].blockFirst)
                    {
                        dl_[w - 1].blockCount += dl_[r].blockCount;
                        dl_[w - 1].blockPointCount += dl_[r].blockPointCount;
                    }
                    else
                    {
                        dl_[w++] = dl_[r];
                    }
                }
                dl_.resize(w);
            }

            uint32_t blockDispatches = 0;
            uint64_t blockTotal = 0;
            uint32_t blockSkipEmpty = 0;

            // Single-dispatch worklist pass1 (per-LOD block CS path).
            if (!args.csUsePointList)
            {
                MICROPROFILE_SCOPEGPUI("LW/PointCS_Block/Pass1DepthWorklist", 0xffe0a0ff);
                struct WI { uint32_t slot, baseGlobal, count, first; };
                static thread_local std::vector<WI> wl;
                wl.clear();
                wl.reserve(drawListBlock[L].size());
                uint32_t cumul = 0;
                for (const DrawItem& it : drawListBlock[L])
                {
                    if (it.blockCount == 0)
                    {
                        ++blockSkipEmpty;
                        continue;
                    }
                    const lw::RuntimeChunk& rc = lwL.chunks[it.slot];
                    wl.push_back({rc.slotIdx, rc.blockBase + it.blockFirst, it.blockCount, cumul});
                    cumul += it.blockCount;
                    blockTotal += it.blockCount;
                }
                if (!wl.empty())
                {
                    uint32_t needCap = (uint32_t)wl.size();
                    if (!worklistSb_ || worklistCapacity_ < needCap)
                    {
                        worklistSb_.Reset();
                        worklistSrv_.Reset();
                        uint32_t cap = 256;
                        while (cap < needCap) cap *= 2;
                        D3D11_BUFFER_DESC bd = {};
                        bd.ByteWidth = cap * 16;
                        bd.Usage = D3D11_USAGE_DYNAMIC;
                        bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
                        bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
                        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
                        bd.StructureByteStride = 16;
                        device_->CreateBuffer(&bd, nullptr, worklistSb_.GetAddressOf());
                        D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
                        sd.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
                        sd.Format = DXGI_FORMAT_UNKNOWN;
                        sd.Buffer.FirstElement = 0;
                        sd.Buffer.NumElements = cap;
                        device_->CreateShaderResourceView(worklistSb_.Get(), &sd, worklistSrv_.GetAddressOf());
                        worklistCapacity_ = cap;
                    }
                    D3D11_MAPPED_SUBRESOURCE mw;
                    ctx_->Map(worklistSb_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mw);
                    memcpy(mw.pData, wl.data(), wl.size() * 16);
                    ctx_->Unmap(worklistSb_.Get(), 0);

                    D3D11_MAPPED_SUBRESOURCE mm;
                    ctx_->Map(cbLwCS_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mm);
                    struct
                    {
                        uint32_t w, h, totalThreads, numItems;
                        float fadeStart, fadeEnd;
                        uint32_t lodIdx, _pad;
                    } cbcs;
                    cbcs.w = width_;
                    cbcs.h = height_;
                    cbcs.totalThreads = cumul;
                    cbcs.numItems = (uint32_t)wl.size();
                    cbcs.fadeStart = 0.0f;
                    cbcs.fadeEnd = 0.0f;
                    cbcs.lodIdx = (uint32_t)L;
                    cbcs._pad = 0;
                    memcpy(mm.pData, &cbcs, sizeof(cbcs));
                    ctx_->Unmap(cbLwCS_.Get(), 0);

                    ID3D11ShaderResourceView* wlSrv[] = {worklistSrv_.Get()};
                    ctx_->CSSetShaderResources(6, 1, wlSrv);
                    ctx_->CSSetShader(csLwBlockDepthWorklist_.Get(), nullptr, 0);
                    ID3D11UnorderedAccessView* uavP1[] = {visDepthUav_.Get()};
                    ctx_->CSSetUnorderedAccessViews(0, 1, uavP1, initB);
                    uint32_t groups = (cumul + 63) / 64;
                    ctx_->Dispatch(groups, 1, 1);
                    ++drawCount;
                    ++blockDispatches;
                    ID3D11UnorderedAccessView* nullUav[] = {nullptr};
                    ctx_->CSSetUnorderedAccessViews(0, 1, nullUav, initB);
                    ID3D11ShaderResourceView* nullWl[] = {nullptr};
                    ctx_->CSSetShaderResources(6, 1, nullWl);
                }
                twoPassLods.push_back(L);
            }

            ID3D11ShaderResourceView* nullCsSrvB[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
            ctx_->CSSetShaderResources(0, 5, nullCsSrvB);
            ID3D11RenderTargetView* mrtRestoreB[] = {splatColorRtv_.Get(), splatMaskRtv_.Get()};
            ctx_->OMSetRenderTargets(2, mrtRestoreB, splatDsv_.Get());
            ctx_->VSSetShaderResources(0, 3, vsSrvs);

            blockTotalAll += blockTotal;
            blockDispatchesAll += blockDispatches;
            if (L >= 0 && L < 5) blockTotalPerLod[L] += blockTotal;
        }
    }
    (void)pointCountTotal; // stats consolidated below after polyaxis pass

    // ---- PointCS_Block Pass2 worklist: all LODs after all pass1s done ----
    // splat-route: when args.splatFilter, write splat-format targets so csSplat_
    // can dilate + light. Otherwise write R16 RGB565 into visColorUav for the
    // existing resolve PS path.
    const bool blockSplatRoute = !args.csUsePointList && args.splatFilter
                                 && csLwBlockSplatWorklist_ && splatColorUav_
                                 && splatMaskUav_ && splatDepthUavF_;
    if (!args.csUsePointList && !twoPassLods.empty() && csLwBlockColorWorklist_ && visColorUav_ && visDepthSrv_)
    {
        MICROPROFILE_SCOPEGPUI("LW/PointCS_Block/Pass2Color", 0xffe080ff);
        ID3D11RenderTargetView* nullRtvsP2[] = {nullptr, nullptr};
        ctx_->OMSetRenderTargets(2, nullRtvsP2, nullptr);
        ID3D11ShaderResourceView* nullVsP2[3] = {nullptr, nullptr, nullptr};
        ctx_->VSSetShaderResources(0, 3, nullVsP2);

        UINT initP2[] = {0, 0, 0, 0};
        if (blockSplatRoute)
        {
            // Clear splat targets; pass2 only writes where it wins depth, so
            // unwritten pixels need sentinel state for splatCS.
            float clearC[4] = {0, 0, 0, 0};
            uint32_t clearM[4] = {0, 0, 0, 0};
            float clearD[4] = {0, 0, 0, 0}; // reverse-Z far = 0
            ctx_->ClearUnorderedAccessViewFloat(splatColorUav_.Get(), clearC);
            ctx_->ClearUnorderedAccessViewUint(splatMaskUav_.Get(), clearM);
            ctx_->ClearUnorderedAccessViewFloat(splatDepthUavF_.Get(), clearD);
            ctx_->CSSetShader(csLwBlockSplatWorklist_.Get(), nullptr, 0);
            ID3D11ShaderResourceView* dSrv[] = {visDepthSrv_.Get()};
            ctx_->CSSetShaderResources(5, 1, dSrv);
            // u0 unused by splat shader; bind splat UAVs at u1..u3.
            ID3D11UnorderedAccessView* uavP2[] = {
                nullptr,
                splatColorUav_.Get(),
                splatMaskUav_.Get(),
                splatDepthUavF_.Get()};
            ctx_->CSSetUnorderedAccessViews(0, 4, uavP2, initP2);
        }
        else
        {
            ctx_->CSSetShader(csLwBlockColorWorklist_.Get(), nullptr, 0);
            ID3D11ShaderResourceView* dSrv[] = {visDepthSrv_.Get()};
            ctx_->CSSetShaderResources(5, 1, dSrv);
            ID3D11UnorderedAccessView* uavP2[] = {visColorUav_.Get()};
            ctx_->CSSetUnorderedAccessViews(0, 1, uavP2, initP2);
        }
        ID3D11Buffer* csCbsP2[] = {cbLwFrame_.Get(), cbLwLod_.Get(), nullptr, cbLwCS_.Get()};
        ctx_->CSSetConstantBuffers(0, 4, csCbsP2);

        for (int Lp : twoPassLods)
        {
            const LwGpu& gp = lwGpu_[Lp];
            const lw::LODWorld& lwLp = lwWorld_.lods[Lp];
            ID3D11ShaderResourceView* csSrvP2[] = {
                gp.pointSrv.Get(), gp.chunkInfoSrv.Get(), gp.paletteSrv.Get(),
                gp.blockPosSrv.Get(), gp.blockColSrv.Get()};
            ctx_->CSSetShaderResources(0, 5, csSrvP2);

            struct WI { uint32_t slot, baseGlobal, count, first; };
            static thread_local std::vector<WI> wlP2;
            wlP2.clear();
            wlP2.reserve(drawListBlock[Lp].size());
            uint32_t cumul = 0;
            for (const DrawItem& it : drawListBlock[Lp])
            {
                if (it.blockCount == 0) continue;
                const lw::RuntimeChunk& rc = lwLp.chunks[it.slot];
                wlP2.push_back({rc.slotIdx, rc.blockBase + it.blockFirst, it.blockCount, cumul});
                cumul += it.blockCount;
            }
            if (wlP2.empty())
                continue;
            uint32_t needCap = (uint32_t)wlP2.size();
            if (!worklistSb_ || worklistCapacity_ < needCap)
            {
                worklistSb_.Reset();
                worklistSrv_.Reset();
                uint32_t cap = 256;
                while (cap < needCap) cap *= 2;
                D3D11_BUFFER_DESC bd = {};
                bd.ByteWidth = cap * 16;
                bd.Usage = D3D11_USAGE_DYNAMIC;
                bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
                bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
                bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
                bd.StructureByteStride = 16;
                device_->CreateBuffer(&bd, nullptr, worklistSb_.GetAddressOf());
                D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
                sd.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
                sd.Format = DXGI_FORMAT_UNKNOWN;
                sd.Buffer.FirstElement = 0;
                sd.Buffer.NumElements = cap;
                device_->CreateShaderResourceView(worklistSb_.Get(), &sd, worklistSrv_.GetAddressOf());
                worklistCapacity_ = cap;
            }
            D3D11_MAPPED_SUBRESOURCE mw;
            ctx_->Map(worklistSb_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mw);
            memcpy(mw.pData, wlP2.data(), wlP2.size() * 16);
            ctx_->Unmap(worklistSb_.Get(), 0);

            D3D11_MAPPED_SUBRESOURCE mm;
            ctx_->Map(cbLwCS_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mm);
            struct
            {
                uint32_t w, h, totalThreads, numItems;
                float fadeStart, fadeEnd;
                uint32_t lodIdx, _pad;
            } cbcs;
            cbcs.w = width_;
            cbcs.h = height_;
            cbcs.totalThreads = cumul;
            cbcs.numItems = (uint32_t)wlP2.size();
            cbcs.fadeStart = 0.0f;
            cbcs.fadeEnd = 0.0f;
            cbcs.lodIdx = (uint32_t)Lp;
            cbcs._pad = 0;
            memcpy(mm.pData, &cbcs, sizeof(cbcs));
            ctx_->Unmap(cbLwCS_.Get(), 0);

            ID3D11ShaderResourceView* wlSrv[] = {worklistSrv_.Get()};
            ctx_->CSSetShaderResources(6, 1, wlSrv);
            uint32_t groups = (cumul + 63) / 64;
            ctx_->Dispatch(groups, 1, 1);
            ++drawCount;
            ID3D11ShaderResourceView* nullWl[] = {nullptr};
            ctx_->CSSetShaderResources(6, 1, nullWl);
        }

        UINT slots = blockSplatRoute ? 4u : 1u;
        ID3D11UnorderedAccessView* nullUavP2[] = {nullptr, nullptr, nullptr, nullptr};
        ctx_->CSSetUnorderedAccessViews(0, slots, nullUavP2, initP2);
        ID3D11ShaderResourceView* nullCsSrvP2[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
        ctx_->CSSetShaderResources(0, 5, nullCsSrvP2);
        ID3D11ShaderResourceView* nullDSrv[] = {nullptr};
        ctx_->CSSetShaderResources(5, 1, nullDSrv);
        ID3D11RenderTargetView* mrtRestoreP2[] = {splatColorRtv_.Get(), splatMaskRtv_.Get()};
        ctx_->OMSetRenderTargets(2, mrtRestoreP2, splatDsv_.Get());
    }

    // ---- Splat dilate CS (csSplat_) — fills holes, lighting, shadow lookup ----
    bool anySplat = false;
    for (int Ls = 0; Ls < lw::kLodCount; ++Ls)
        if (!drawListSplat[Ls].empty())
        {
            anySplat = true;
            break;
        }
    // PointCS_Block splat-route also writes to splat targets → run splat dilate.
    bool anyBlockSplat = blockSplatRoute && !twoPassLods.empty();
    if (anyBlockSplat) anySplat = true;
    if (anySplat && args.splatFilter && csSplat_)
    {
        MICROPROFILE_SCOPEGPUI("LW/SplatCS", 0xffffa030);
        ID3D11RenderTargetView* nullRtvs[] = {nullptr, nullptr};
        ctx_->OMSetRenderTargets(2, nullRtvs, nullptr);
        // Unbind VS SRVs (we'll rebind for CS).
        ID3D11ShaderResourceView* nullVs[3] = {nullptr, nullptr, nullptr};
        ctx_->VSSetShaderResources(0, 3, nullVs);

        ctx_->CSSetShader(csSplat_.Get(), nullptr, 0);
        ID3D11Buffer* csCbs[] = {cbPerFrame_.Get()};
        ctx_->CSSetConstantBuffers(0, 1, csCbs);
        // Block-route uses the R32F UAV-friendly depth texture; raster path uses
        // the DSV texture's R32F SRV view. splatColor + splatMask are shared.
        ID3D11ShaderResourceView* depthSrvForCS =
            anyBlockSplat ? splatBlockDepthSrv_.Get() : splatDepthSrv_.Get();
        ID3D11ShaderResourceView* csSrvs[] = {
            nullptr,
            depthSrvForCS,
            splatColorSrv_.Get(),
            splatMaskSrv_.Get()};
        ctx_->CSSetShaderResources(0, 4, csSrvs);
        // Shadow map at t6 + sampler at s1 (ApplyShadowLighting reads these).
        if (sunShadowsOn && shadowSrv_)
        {
            ID3D11ShaderResourceView* effSh =
                (args.shadowBlur && shadowFilledSrv_) ? shadowFilledSrv_.Get() : shadowSrv_.Get();
            ID3D11ShaderResourceView* shSrvCS[] = {effSh};
            ctx_->CSSetShaderResources(6, 1, shSrvCS);
            if (shadowSamp_)
            {
                ID3D11SamplerState* shSmCS[] = {shadowSamp_.Get()};
                ctx_->CSSetSamplers(1, 1, shSmCS);
            }
        }
        ID3D11UnorderedAccessView* csUavs[] = {
            nullptr,
            splatFinalUav_.Get(),
            splatFinalDepthUav_.Get()};
        UINT initc[] = {0, 0, 0};
        ctx_->CSSetUnorderedAccessViews(0, 3, csUavs, initc);
        UINT gx = (width_ + 7) / 8;
        UINT gy = (height_ + 7) / 8;
        ctx_->Dispatch(gx, gy, 1);
        ID3D11ShaderResourceView* nullCsSrvs[] = {nullptr, nullptr, nullptr, nullptr};
        ctx_->CSSetShaderResources(0, 4, nullCsSrvs);
        ID3D11ShaderResourceView* nullShCS[] = {nullptr};
        ctx_->CSSetShaderResources(6, 1, nullShCS);
        ID3D11UnorderedAccessView* nullCsUavs[] = {nullptr, nullptr, nullptr};
        ctx_->CSSetUnorderedAccessViews(0, 3, nullCsUavs, initc);
        ctx_->CSSetShader(nullptr, nullptr, 0);
    }

    // ---- Scene RT clear + (optional) splat composite ----
    {
        MICROPROFILE_SCOPEGPUI("LW/Composite", 0xffff8040);
        ID3D11RenderTargetView* sceneRtv = postEnabled ? taaSceneRtv_.Get() : rtv_.Get();
        // Scene RT alpha = sky mask (post pass draws sky/godrays where alpha<0.5).
        float sceneClear[4] = {lastClear_[0], lastClear_[1], lastClear_[2], postEnabled ? 0.0f : lastClear_[3]};
        ctx_->ClearRenderTargetView(sceneRtv, sceneClear);
        // dsv already cleared in BeginFrame; no other path writes it before here.
        ID3D11RenderTargetView* compositeRtv[] = {sceneRtv};
        ctx_->OMSetRenderTargets(1, compositeRtv, dsv_.Get());
        if (anySplat)
        {
            ctx_->OMSetDepthStencilState(dsAlwaysWrite_.Get(), 0);
            ctx_->RSSetState(rsNoCull_.Get());
            ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx_->IASetInputLayout(nullptr);
            ID3D11Buffer* nVb[] = {nullptr};
            UINT zz = 0;
            ctx_->IASetVertexBuffers(0, 1, nVb, &zz, &zz);
            ctx_->VSSetShader(vsBlit_.Get(), nullptr, 0);
            ctx_->PSSetShader(psSplatComposite_.Get(), nullptr, 0);
            ID3D11ShaderResourceView* compSrv[] = {
                args.splatFilter ? splatFinalSrv_.Get() : splatColorSrv_.Get()};
            ctx_->PSSetShaderResources(8, 1, compSrv);
            ID3D11ShaderResourceView* compDepthSrv[] = {
                args.splatFilter ? splatFinalDepthSrv_.Get() : splatDepthSrv_.Get()};
            ctx_->PSSetShaderResources(7, 1, compDepthSrv);
            ctx_->OMSetBlendState(bsAlphaOver_.Get(), nullptr, 0xFFFFFFFFu);
            ctx_->Draw(3, 0);
            ctx_->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFFu);
            ID3D11ShaderResourceView* nullCSrv[] = {nullptr};
            ctx_->PSSetShaderResources(8, 1, nullCSrv);
            ctx_->PSSetShaderResources(7, 1, nullCSrv);
        }
        ctx_->RSSetState(rsSolid_.Get());
        ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
    }

    // ---- Compute rasterizer resolve (two-pass): depth+colour → scene RT ----
    // Skip when block-splat route is on — composite path handles output.
    if (!args.csUsePointList && !blockSplatRoute && anyCS && psLwResolveTwoPass_ && vsLwResolve_ && visDepthSrv_ && visColorSrv_)
    {
        MICROPROFILE_SCOPEGPUI("LW/PointCS/Resolve", 0xff8040c0);
        ID3D11RenderTargetView* sceneRtv = postEnabled ? taaSceneRtv_.Get() : rtv_.Get();
        ctx_->OMSetRenderTargets(1, &sceneRtv, nullptr);
        ctx_->OMSetDepthStencilState(dsAlways_.Get(), 0);
        ctx_->RSSetState(rsNoCull_.Get());
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx_->IASetInputLayout(nullptr);
        ID3D11Buffer* nvb[] = {nullptr};
        UINT zz = 0;
        ctx_->IASetVertexBuffers(0, 1, nvb, &zz, &zz);
        ctx_->VSSetShader(vsLwResolve_.Get(), nullptr, 0);
        ctx_->PSSetShader(psLwResolveTwoPass_.Get(), nullptr, 0);
        ID3D11ShaderResourceView* rSrv[] = {visDepthSrv_.Get(), visColorSrv_.Get()};
        ctx_->PSSetShaderResources(3, 2, rSrv);
        ctx_->Draw(3, 0);
        ID3D11ShaderResourceView* nullR[] = {nullptr, nullptr};
        ctx_->PSSetShaderResources(3, 2, nullR);
        ctx_->RSSetState(rsSolid_.Get());
        ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
    }

    // ---- PointCS A/B: hardware-rasterized POINTLIST via VS+PS ----
    if (args.csUsePointList && vsLwBlockPoint_ && psLwBlockPoint_)
    {
        MICROPROFILE_SCOPEGPUI("LW/BlockPoints/Draw", 0xff80ffe0);
        ID3D11RenderTargetView* sceneRtv = postEnabled ? taaSceneRtv_.Get() : rtv_.Get();
        ID3D11RenderTargetView* mrt[] = {sceneRtv};
        ctx_->OMSetRenderTargets(1, mrt, dsv_.Get());
        ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
        ctx_->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFFu);
        ctx_->RSSetState(rsNoCull_.Get());
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
        ctx_->IASetInputLayout(nullptr);
        ID3D11Buffer* nullVb[] = {nullptr};
        UINT zS = 0, zO = 0;
        ctx_->IASetVertexBuffers(0, 1, nullVb, &zS, &zO);
        ctx_->VSSetShader(vsLwBlockPoint_.Get(), nullptr, 0);
        ctx_->PSSetShader(psLwBlockPoint_.Get(), nullptr, 0);
        ID3D11Buffer* ptCbs[] = {cbLwFrame_.Get(), cbLwLod_.Get()};
        ctx_->VSSetConstantBuffers(0, 2, ptCbs);
        ctx_->PSSetConstantBuffers(0, 2, ptCbs);

        for (int L = lw::kLodCount - 1; L >= 0; --L)
        {
            if (drawListBlock[L].empty())
                continue;
            const LwGpu& g = lwGpu_[L];
            if (!g.blockPointSrv)
                continue;
            const lw::LODWorld& lwL = lwWorld_.lods[L];
            // VS reads gLwChunkInfos(t1), gLwPalette(t2), gLwBlockPoints(t7).
            // PS reads gLwChunkInfos+gLwPalette too via gLwSlot.
            ID3D11ShaderResourceView* vsSrvs[] = {
                nullptr, g.chunkInfoSrv.Get(), g.paletteSrv.Get(),
            };
            ctx_->VSSetShaderResources(0, 3, vsSrvs);
            ctx_->PSSetShaderResources(0, 3, vsSrvs);
            ID3D11ShaderResourceView* ptSrv[] = {g.blockPointSrv.Get()};
            ctx_->VSSetShaderResources(7, 1, ptSrv);
            for (const DrawItem& it : drawListBlock[L])
            {
                if (it.blockPointCount == 0)
                    continue;
                const lw::RuntimeChunk& rc = lwL.chunks[it.slot];
                setLodCbForLod(L, rc.slotIdx, rc.blockPointBase + it.blockPointFirst);
                ctx_->Draw(it.blockPointCount, 0);
                ++drawCount;
            }
            ID3D11ShaderResourceView* nullPt[] = {nullptr};
            ctx_->VSSetShaderResources(7, 1, nullPt);
        }
    }

    // ---- PolyAxis pass: cube faces direct to scene RT, after composite ----
    {
        bool anyPoly = false;
        for (int L = 0; L < lw::kLodCount; ++L)
        {
            if (!drawListPoly[L].empty())
            {
                anyPoly = true;
                break;
            }
        }
        if (anyPoly)
        {
            MICROPROFILE_SCOPEGPUI("LW/PolyAxis", 0xff80b0e0);
            ID3D11RenderTargetView* sceneRtv = postEnabled ? taaSceneRtv_.Get() : rtv_.Get();
            ID3D11RenderTargetView* mrt[] = {sceneRtv};
            ctx_->OMSetRenderTargets(1, mrt, dsv_.Get());
            ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
            ctx_->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFFu);
            ctx_->RSSetState(rsNoCull_.Get());
            ctx_->IASetInputLayout(nullptr);
            ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ID3D11Buffer* nullVb[] = {nullptr};
            UINT vbS = 0, vbO = 0;
            ctx_->IASetVertexBuffers(0, 1, nullVb, &vbS, &vbO);
            ctx_->VSSetShader(vsLwPolyAxis_.Get(), nullptr, 0);
            ctx_->PSSetShader(psLwPolyAxisLit_.Get(), nullptr, 0);
            ID3D11Buffer* paCbs[] = {cbLwFrame_.Get(), cbLwLod_.Get()};
            ctx_->VSSetConstantBuffers(0, 2, paCbs);
            ctx_->PSSetConstantBuffers(0, 2, paCbs);
            // Shadow map for SampleShadow (t6) + comparison sampler (s1).
            // Match splat CS: only use blurred shadow if blur actually ran
            // (otherwise the filled tex contains zeros = blacks everything).
            ID3D11ShaderResourceView* psShadowSrv =
                (args.shadowBlur && shadowFilledSrv_) ? shadowFilledSrv_.Get() : shadowSrv_.Get();
            ctx_->PSSetShaderResources(6, 1, &psShadowSrv);
            ID3D11SamplerState* psSamps[] = {shadowSamp_.Get()};
            ctx_->PSSetSamplers(1, 1, psSamps);
            for (int L = lw::kLodCount - 1; L >= 0; --L)
            {
                if (drawListPoly[L].empty())
                    continue;
                const LwGpu& g = lwGpu_[L];
                if (g.slotCount == 0 || !g.pointSrv)
                    continue;
                const lw::LODWorld& lwL = lwWorld_.lods[L];
                ID3D11ShaderResourceView* vsSrvs[] = {
                    g.pointSrv.Get(),
                    g.chunkInfoSrv.Get(),
                    g.paletteSrv.Get(),
                };
                ctx_->VSSetShaderResources(0, 3, vsSrvs);
                std::sort(drawListPoly[L].begin(), drawListPoly[L].end(),
                          [](const DrawItem& a, const DrawItem& b)
                          {
                              if (a.slot != b.slot)
                                  return a.slot < b.slot;
                              return a.drawBase < b.drawBase;
                          });
                uint32_t curSlot = 0xFFFFFFFFu, curBase = 0, curCount = 0;
                auto flushPoly = [&]()
                {
                    if (curCount == 0)
                        return;
                    const lw::RuntimeChunk& rc = lwL.chunks[curSlot];
                    setLodCbForLod(L, rc.slotIdx, curBase);
                    ctx_->Draw(curCount * 36u, 0);
                    ++drawCount;
                    ++polyDraws;
                    if (curSlot == prevSlot)
                        ++fastDraws;
                    prevSlot = curSlot;
                    polyVoxels += curCount;
                    triCount += (uint64_t)curCount * 12ull;
                    curCount = 0;
                };
                for (const DrawItem& it : drawListPoly[L])
                {
                    if (it.slot == curSlot && curBase + curCount == it.drawBase)
                    {
                        curCount += it.drawCount;
                    }
                    else
                    {
                        flushPoly();
                        curSlot = it.slot;
                        curBase = it.drawBase;
                        curCount = it.drawCount;
                    }
                }
                flushPoly();
            }
        }
    }

    // Unbind VS SRVs.
    {
        ID3D11ShaderResourceView* nullSrvs[3] = {nullptr, nullptr, nullptr};
        ctx_->VSSetShaderResources(0, 3, nullSrvs);
    }

    // Finalize per-frame stats now that both splat + polyaxis passes ran.
    lastDrawn_ = drawCount;
    lastPointCount_ = splatVoxels + polyVoxels;
    lastPolyVoxelCount_ = polyVoxels;
    lastSplatVoxelCount_ = splatVoxels;
    lastTriCount_ = triCount;
    lastSplatDrawCalls_ = splatDraws;
    lastPolyDrawCalls_ = polyDraws;
    lastFastDrawCalls_ = fastDraws;
    lastBlockTotal_ = blockTotalAll;
    lastBlockDispatches_ = blockDispatchesAll;
    for (int L = 0; L < 5; ++L) lastBlockTotalPerLod_[L] = blockTotalPerLod[L];

    // ---- TAA composite + post (optional) ----
    ID3D11ShaderResourceView* postInput = nullptr;
    if (postEnabled)
    {
        ctx_->IASetInputLayout(nullptr);
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ID3D11Buffer* nullVb[] = {nullptr};
        UINT zero = 0;
        ctx_->IASetVertexBuffers(0, 1, nullVb, &zero, &zero);
        ID3D11SamplerState* samps[] = {linearClampSampler_.Get()};
        ctx_->PSSetSamplers(0, 1, samps);
        ctx_->OMSetDepthStencilState(dsAlways_.Get(), 0);
        ctx_->RSSetState(rsNoCull_.Get());

        // TAA + post shaders use cbuffer cbPerFrame at b0 (gNearZ, gJitter,
        // gCamPos/Right/Up/Forward, gPrevViewProj, gFog*, gLightDir, etc).
        // Earlier passes left cbLwFrame_ bound here -> wrong layout reads.
        ID3D11Buffer* psPostCbs[] = {cbPerFrame_.Get()};
        ctx_->VSSetConstantBuffers(0, 1, psPostCbs);
        ctx_->PSSetConstantBuffers(0, 1, psPostCbs);

        if (args.taa && vsTaa_ && psTaa_)
        {
            MICROPROFILE_SCOPEGPUI("LW/TAA", 0xff40ffd0);
            uint32_t curr = taaHistIdx_;
            uint32_t prev = curr ^ 1u;
            ID3D11RenderTargetView* hRtv = taaHistRtv_[curr].Get();
            ctx_->OMSetRenderTargets(1, &hRtv, nullptr);
            ctx_->VSSetShader(vsTaa_.Get(), nullptr, 0);
            ctx_->PSSetShader(psTaa_.Get(), nullptr, 0);
            ID3D11ShaderResourceView* taaSrvs[] = {
                nullptr, nullptr, nullptr, nullptr,
                taaSceneSrv_.Get(),
                taaHistValid_[prev] ? taaHistSrv_[prev].Get() : taaSceneSrv_.Get(),
                depthSrv_.Get()};
            ctx_->PSSetShaderResources(0, 7, taaSrvs);
            ctx_->Draw(3, 0);
            ID3D11ShaderResourceView* nullsForPost[] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
            ctx_->PSSetShaderResources(0, 7, nullsForPost);
            postInput = taaHistSrv_[curr].Get();
            taaHistValid_[curr] = true;
            taaHistIdx_ = prev;
            ++taaFrame_;
        }
        else
        {
            postInput = taaSceneSrv_.Get();
        }

        // ---- Godrays: mark + radial blur into 64x64 ----
        {
            MICROPROFILE_SCOPEGPUI("LW/Godray", 0xfffff080);
            // Project sun into NDC via a far point along light direction.
            float cp[3];
            hlslpp::store(cp, cam.position);
            float dx = args.sunDir[0], dy = args.sunDir[1], dz = args.sunDir[2];
            float farK = 1.0e6f;
            float sxw = cp[0] + dx * farK;
            float syw = cp[1] + dy * farK;
            float szw = cp[2] + dz * farK;
            float vpS[16];
            hlslpp::store(vpS, vpUnjittered);
            auto col = [&](int j)
            { return sxw * vpS[0 + j] + syw * vpS[4 + j] + szw * vpS[8 + j] + 1.0f * vpS[12 + j]; };
            float cxv = col(0), cyv = col(1), cwv = col(3);
            float sunNdcX = 0.0f, sunNdcY = 0.0f;
            float onScreen = 0.0f;
            if (cwv > 1e-3f)
            {
                sunNdcX = cxv / cwv;
                sunNdcY = cyv / cwv;
                if (fabsf(sunNdcX) <= 1.0f && fabsf(sunNdcY) <= 1.0f)
                    onScreen = 1.0f;
            }
            struct CBG
            {
                float sunNdc[2];
                float onScreen;
                float emaAlpha;
                float halfScreenUV[2];
                float facing;
                float _padG1;
                float sunUV[2];
                float _padG2[2];
                float tint[3];
                float strength;
            } cbg;
            cbg.sunNdc[0] = sunNdcX;
            cbg.sunNdc[1] = sunNdcY;
            cbg.onScreen = onScreen;
            cbg.emaAlpha = args.godrayEmaAlpha;
            // Texture covers exactly args.godrayAngleDeg of arc around the
            // sun, square in screen pixels. Perspective math:
            //   halfPx = (screenH / 2) * tan(angle/2) / tan(fov/2)
            const float kDeg2Rad = 3.14159265358979f / 180.0f;
            float halfPx = (float)height_ * 0.5f * tanf(args.godrayAngleDeg * 0.5f * kDeg2Rad) / tanf(cam.fovDeg * 0.5f * kDeg2Rad);
            cbg.halfScreenUV[0] = halfPx / (float)width_;
            cbg.halfScreenUV[1] = halfPx / (float)height_;
            // gSunFacing = dot(viewForward, sunDir). Positive when sun is in
            // front of camera, negative behind. Post uses this to fade godrays
            // out when looking away from sun (180° opposite).
            float fwd[3];
            {
                hlslpp::float3 f = cam.forward();
                hlslpp::store(fwd, f);
            }
            cbg.facing = fwd[0] * args.sunDir[0] + fwd[1] * args.sunDir[1] + fwd[2] * args.sunDir[2];
            cbg._padG1 = 0.0f;
            cbg.sunUV[0] = sunNdcX * 0.5f + 0.5f;
            cbg.sunUV[1] = sunNdcY * -0.5f + 0.5f;
            cbg._padG2[0] = cbg._padG2[1] = 0.0f;
            cbg.tint[0] = args.godrayTint[0];
            cbg.tint[1] = args.godrayTint[1];
            cbg.tint[2] = args.godrayTint[2];
            cbg.strength = args.godrayStrength;
            D3D11_MAPPED_SUBRESOURCE mmg;
            ctx_->Map(cbGodray_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mmg);
            memcpy(mmg.pData, &cbg, sizeof(cbg));
            ctx_->Unmap(cbGodray_.Get(), 0);

            D3D11_VIEWPORT vp64 = {};
            vp64.Width = 64.0f;
            vp64.Height = 64.0f;
            vp64.MaxDepth = 1.0f;
            ctx_->RSSetViewports(1, &vp64);
            ctx_->RSSetState(rsNoCull_.Get());
            ctx_->OMSetDepthStencilState(dsAlways_.Get(), 0);
            ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx_->IASetInputLayout(nullptr);
            ID3D11Buffer* nVbG[] = {nullptr};
            UINT vSG = 0, vOG = 0;
            ctx_->IASetVertexBuffers(0, 1, nVbG, &vSG, &vOG);
            ctx_->VSSetShader(vsTaa_.Get(), nullptr, 0);
            ID3D11Buffer* grCbs[4] = {cbPerFrame_.Get(), nullptr, nullptr, cbGodray_.Get()};
            ctx_->PSSetConstantBuffers(0, 4, grCbs);
            ctx_->PSSetSamplers(0, 1, linearClampSampler_.GetAddressOf());

            // Mark
            ctx_->PSSetShader(psGodrayMark_.Get(), nullptr, 0);
            ID3D11RenderTargetView* mark0[] = {godrayRtv_[0].Get()};
            ctx_->OMSetRenderTargets(1, mark0, nullptr);
            ID3D11ShaderResourceView* dsrv[] = {depthSrv_.Get()};
            ctx_->PSSetShaderResources(6, 1, dsrv);
            ctx_->Draw(3, 0);
            ID3D11ShaderResourceView* nullDsrv[] = {nullptr};
            ctx_->PSSetShaderResources(6, 1, nullDsrv);

            // Blur+EMA: read mark + previous frame's blend, write to other slot.
            uint32_t prevIdx = godrayCurrIdx_; // last frame's output
            uint32_t writeIdx = (prevIdx == 1u) ? 2u : 1u;
            ctx_->PSSetShader((args.godrayAniso && psGodrayBlurAniso_) ? psGodrayBlurAniso_.Get() : psGodrayBlur_.Get(), nullptr, 0);
            if (args.godrayAniso && anisoSamp_)
            {
                ID3D11SamplerState* anS[] = {anisoSamp_.Get()};
                ctx_->PSSetSamplers(2, 1, anS);
            }
            ID3D11RenderTargetView* blurRtv[] = {godrayRtv_[writeIdx].Get()};
            ctx_->OMSetRenderTargets(1, blurRtv, nullptr);
            ID3D11ShaderResourceView* grSrv[] = {godraySrv_[0].Get()};        // mark @ t9
            ID3D11ShaderResourceView* grHist[] = {godraySrv_[prevIdx].Get()}; // history @ t10
            ctx_->PSSetShaderResources(9, 1, grSrv);
            ctx_->PSSetShaderResources(10, 1, grHist);
            ctx_->Draw(3, 0);
            ID3D11ShaderResourceView* nullGrSrv[] = {nullptr};
            ctx_->PSSetShaderResources(9, 1, nullGrSrv);
            ctx_->PSSetShaderResources(10, 1, nullGrSrv);
            godrayCurrIdx_ = writeIdx;

            // Restore main viewport.
            D3D11_VIEWPORT vpFull = {};
            vpFull.Width = (float)width_;
            vpFull.Height = (float)height_;
            vpFull.MaxDepth = 1.0f;
            ctx_->RSSetViewports(1, &vpFull);
        }

        // Final post: sky/sharpen/tonemap + godrays to backbuffer.
        {
            MICROPROFILE_SCOPEGPUI("LW/FinalPost", 0xffff8040);
            ID3D11RenderTargetView* bRtv = rtv_.Get();
            ctx_->OMSetRenderTargets(1, &bRtv, nullptr);
            ctx_->VSSetShader(vsTaa_.Get(), nullptr, 0);
            ctx_->PSSetShader(psPost_.Get(), nullptr, 0);
            ID3D11ShaderResourceView* postSrvs[] = {depthSrv_.Get(), postInput};
            ctx_->PSSetShaderResources(6, 2, postSrvs);
            ID3D11ShaderResourceView* grPostSrv[] = {godraySrv_[godrayCurrIdx_].Get()};
            ctx_->PSSetShaderResources(9, 1, grPostSrv);
            ID3D11Buffer* postCbs[4] = {cbPerFrame_.Get(), nullptr, nullptr, cbGodray_.Get()};
            ctx_->PSSetConstantBuffers(0, 4, postCbs);
            ctx_->PSSetSamplers(0, 1, linearClampSampler_.GetAddressOf());
            ctx_->Draw(3, 0);
            ID3D11ShaderResourceView* nullGr[] = {nullptr};
            ctx_->PSSetShaderResources(9, 1, nullGr);
        }
        ctx_->RSSetState(rsSolid_.Get());

        ID3D11ShaderResourceView* nulls[] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        ctx_->PSSetShaderResources(0, 8, nulls);

        ID3D11RenderTargetView* backRtv = rtv_.Get();
        ctx_->OMSetRenderTargets(1, &backRtv, nullptr);
    }

    // ---- Per-chunk AABB wireframe (after TAA so it stays crisp) ----
    if (args.lwShowBounds)
    {
        MICROPROFILE_SCOPEGPUI("LW/Bounds", 0xff80c0ff);
        ID3D11RenderTargetView* bRtv[] = {rtv_.Get()};
        ctx_->OMSetRenderTargets(1, bRtv, dsv_.Get());
        ctx_->OMSetDepthStencilState(dsTest_.Get(), 0);
        ctx_->RSSetState(rsSolid_.Get());
        ctx_->IASetInputLayout(nullptr);
        ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
        ID3D11Buffer* nullVbs2[] = {nullptr};
        UINT vbS = 0, vbO = 0;
        ctx_->IASetVertexBuffers(0, 1, nullVbs2, &vbS, &vbO);
        ctx_->VSSetShader(vsLwBounds_.Get(), nullptr, 0);
        ctx_->PSSetShader(psLwBounds_.Get(), nullptr, 0);
        ID3D11Buffer* boundsCbs[] = {cbLwFrame_.Get(), cbLwLod_.Get(), cbLwBounds_.Get()};
        ctx_->VSSetConstantBuffers(0, 3, boundsCbs);
        ctx_->PSSetConstantBuffers(0, 3, boundsCbs);
        static const float kLodColors[lw::kLodCount][3] = {
            {1.0f, 0.3f, 0.3f},
            {1.0f, 0.8f, 0.2f},
            {0.4f, 1.0f, 0.4f},
            {0.3f, 0.7f, 1.0f},
            {0.9f, 0.4f, 1.0f},
        };
        for (int Lb = 0; Lb < lw::kLodCount; ++Lb)
        {
            const LwGpu& gb = lwGpu_[Lb];
            if (gb.slotCount == 0 || !gb.chunkInfoSrv)
                continue;
            ID3D11ShaderResourceView* srvB[] = {nullptr, gb.chunkInfoSrv.Get(), nullptr};
            ctx_->VSSetShaderResources(0, 3, srvB);
            D3D11_MAPPED_SUBRESOURCE mm;
            ctx_->Map(cbLwBounds_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mm);
            struct
            {
                float col[3];
                float p0;
                float dim[3];
                float p1;
            } cbb;
            cbb.col[0] = kLodColors[Lb][0];
            cbb.col[1] = kLodColors[Lb][1];
            cbb.col[2] = kLodColors[Lb][2];
            cbb.p0 = 0.0f;
            cbb.dim[0] = (float)lw::kChunkVoxX;
            cbb.dim[1] = (float)lw::kChunkVoxY;
            cbb.dim[2] = (float)lw::kChunkVoxZ;
            cbb.p1 = 0.0f;
            memcpy(mm.pData, &cbb, sizeof(cbb));
            ctx_->Unmap(cbLwBounds_.Get(), 0);
            ctx_->DrawInstanced(24, gb.slotCount, 0, 0);
        }
        ID3D11ShaderResourceView* nullSrvs[3] = {nullptr, nullptr, nullptr};
        ctx_->VSSetShaderResources(0, 3, nullSrvs);
    }

    // Stash this frame's un-jittered VP for next-frame reprojection.
    hlslpp::store(taaPrevVP_, vpUnjittered);
}

void Renderer::EndFrame(bool vsync)
{
    UINT presentFlags = (!vsync && tearingSupported_) ? DXGI_PRESENT_ALLOW_TEARING : 0;
    swap_->Present(vsync ? 1 : 0, presentFlags);
}
