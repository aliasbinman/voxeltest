// voxtile.cpp — replicate a .vox model in XZ. At the outer skirts (xMin/xMax
// and zMin/zMax edges of source) extend perimeter voxel columns DOWN to
// y=0 so adjacent tiles meet without vertical gaps. Re-chunks output and
// writes VXL3 .vox.
//
// Usage:
//   voxtile.exe <in.vox> <out.vox> <NX> <NZ>
// Defaults: assets/kingslanding.vox  assets/kingslanding25.vox  5  5

#define _CRT_SECURE_NO_WARNINGS
#include "asset_version.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

#pragma pack(push, 1)
struct DiskVoxel
{
    uint8_t x, y, z;
    uint8_t visMask;
    uint16_t paletteIdx;
    uint8_t aoPacked[3];
};
#pragma pack(pop)
static_assert(sizeof(DiskVoxel) == 9, "");

struct ChunkMeta
{
    uint16_t cx, cy, cz, _pad;
    uint32_t voxelCount;
    uint32_t voxelOffset;
};
static_assert(sizeof(ChunkMeta) == 16, "");

struct WV
{
    int32_t x, y, z;
    uint8_t visMask;
    uint16_t paletteIdx;
    uint8_t aoPacked[3];
};

struct CK
{
    int cx, cy, cz;
    bool operator==(const CK& o) const
    {
        return cx == o.cx && cy == o.cy && cz == o.cz;
    }
};
struct CKHash
{
    size_t operator()(const CK& k) const noexcept
    {
        uint64_t h = (uint64_t)(uint32_t)k.cx;
        h = h * 1315423911u + (uint64_t)(uint32_t)k.cy;
        h = h * 2654435761u + (uint64_t)(uint32_t)k.cz;
        return (size_t)h;
    }
};

int main(int argc, char** argv)
{
    const char* inPath = (argc > 1) ? argv[1] : "assets/kingslanding.vox";
    const char* outPath = (argc > 2) ? argv[2] : "assets/kingslanding25.vox";
    int NX = (argc > 3) ? std::atoi(argv[3]) : 5;
    int NZ = (argc > 4) ? std::atoi(argv[4]) : 5;
    if (NX < 1 || NZ < 1)
    {
        fprintf(stderr, "bad NX/NZ\n");
        return 1;
    }

    // ---- Read source .vox raw ----
    FILE* f = fopen(inPath, "rb");
    if (!f)
    {
        fprintf(stderr, "cannot open %s\n", inPath);
        return 1;
    }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "VXL3", 4) != 0)
    {
        fclose(f);
        fprintf(stderr, "bad magic\n");
        return 1;
    }
    uint32_t version = 0;
    fread(&version, sizeof(uint32_t), 1, f);
    if (version != kAssetVersion)
    {
        fclose(f);
        fprintf(stderr, "version mismatch: %u vs %u\n", version, kAssetVersion);
        return 1;
    }
    uint32_t chunkDim = 0, chunkCount = 0, totalVoxels = 0;
    int32_t srcOrigin[3] = {0, 0, 0};
    float srcSun[3] = {0, 0, 0};
    fread(&chunkDim, sizeof(uint32_t), 1, f);
    fread(&chunkCount, sizeof(uint32_t), 1, f);
    fread(&totalVoxels, sizeof(uint32_t), 1, f);
    fread(srcOrigin, sizeof(int32_t), 3, f);
    fread(srcSun, sizeof(float), 3, f);
    uint32_t paletteCount = 0;
    fread(&paletteCount, sizeof(uint32_t), 1, f);
    std::vector<uint32_t> palette(paletteCount);
    if (paletteCount)
        fread(palette.data(), sizeof(uint32_t), paletteCount, f);
    std::vector<ChunkMeta> srcMetas(chunkCount);
    if (chunkCount)
        fread(srcMetas.data(), sizeof(ChunkMeta), chunkCount, f);
    std::vector<DiskVoxel> srcVox(totalVoxels);
    if (totalVoxels)
        fread(srcVox.data(), sizeof(DiskVoxel), totalVoxels, f);
    fclose(f);
    printf("[in] %s chunkDim=%u chunkCount=%u voxels=%u palette=%u\n",
           inPath, chunkDim, chunkCount, totalVoxels, paletteCount);

    const int32_t D = (int32_t)chunkDim;

    // ---- Flatten source voxels into world (scene-relative) coords ----
    std::vector<WV> src;
    src.reserve(totalVoxels);
    int32_t mn[3] = {INT32_MAX, INT32_MAX, INT32_MAX};
    int32_t mx[3] = {-INT32_MAX, -INT32_MAX, -INT32_MAX};
    for (uint32_t ci = 0; ci < chunkCount; ++ci)
    {
        const ChunkMeta& m = srcMetas[ci];
        int32_t bx = (int32_t)m.cx * D;
        int32_t by = (int32_t)m.cy * D;
        int32_t bz = (int32_t)m.cz * D;
        const DiskVoxel* vs = srcVox.data() + m.voxelOffset;
        for (uint32_t i = 0; i < m.voxelCount; ++i)
        {
            const DiskVoxel& dv = vs[i];
            WV w;
            w.x = bx + (int32_t)dv.x;
            w.y = by + (int32_t)dv.y;
            w.z = bz + (int32_t)dv.z;
            w.visMask = dv.visMask;
            w.paletteIdx = dv.paletteIdx;
            w.aoPacked[0] = dv.aoPacked[0];
            w.aoPacked[1] = dv.aoPacked[1];
            w.aoPacked[2] = dv.aoPacked[2];
            src.push_back(w);
            if (w.x < mn[0])
                mn[0] = w.x;
            if (w.x > mx[0])
                mx[0] = w.x;
            if (w.y < mn[1])
                mn[1] = w.y;
            if (w.y > mx[1])
                mx[1] = w.y;
            if (w.z < mn[2])
                mn[2] = w.z;
            if (w.z > mx[2])
                mx[2] = w.z;
        }
    }
    const int32_t xSpan = mx[0] - mn[0] + 1;
    const int32_t ySpan = mx[1] - mn[1] + 1;
    const int32_t zSpan = mx[2] - mn[2] + 1;
    printf("[in] AABB x[%d..%d] y[%d..%d] z[%d..%d]  span=(%d,%d,%d)\n",
           mn[0], mx[0], mn[1], mx[1], mn[2], mx[2], xSpan, ySpan, zSpan);

    // Re-base source to local 0..span-1 so tiling math is clean.
    for (WV& w : src)
    {
        w.x -= mn[0];
        w.y -= mn[1];
        w.z -= mn[2];
    }

    // ---- Skirt fill: for each (x, z) column on perimeter rings, find lowest
    // y in the source and replicate that voxel DOWN to y=0. Output the new
    // fill voxels into a separate vector; original `src` stays unchanged.
    struct ColKey
    {
        int32_t x, z;
        bool operator==(const ColKey& o) const
        {
            return x == o.x && z == o.z;
        }
    };
    struct ColHash
    {
        size_t operator()(const ColKey& k) const noexcept
        {
            return (size_t)((uint64_t)(uint32_t)k.x * 0x9E3779B97F4A7C15ull + (uint64_t)(uint32_t)k.z);
        }
    };
    struct ColLow
    {
        int32_t yLo;
        uint16_t pal;
        uint8_t mask;
        uint8_t ao[3];
    };
    std::unordered_map<ColKey, ColLow, ColHash> perimLow;
    perimLow.reserve(2 * (xSpan + zSpan));
    const int32_t xLo = 0, xHi = xSpan - 1;
    const int32_t zLo = 0, zHi = zSpan - 1;

    for (const WV& w : src)
    {
        bool onPerim = (w.x == xLo) || (w.x == xHi) || (w.z == zLo) || (w.z == zHi);
        if (!onPerim)
            continue;
        ColKey k{w.x, w.z};
        auto it = perimLow.find(k);
        if (it == perimLow.end() || w.y < it->second.yLo)
        {
            ColLow c;
            c.yLo = w.y;
            c.pal = w.paletteIdx;
            c.mask = w.visMask;
            c.ao[0] = w.aoPacked[0];
            c.ao[1] = w.aoPacked[1];
            c.ao[2] = w.aoPacked[2];
            perimLow[k] = c;
        }
    }

    std::vector<WV> skirt;
    {
        uint64_t fillCount = 0;
        for (auto& kv : perimLow)
            fillCount += (uint64_t)kv.second.yLo;
        skirt.reserve((size_t)fillCount);
    }
    for (auto& kv : perimLow)
    {
        const ColKey& k = kv.first;
        const ColLow& c = kv.second;
        for (int32_t y = 0; y < c.yLo; ++y)
        {
            WV w;
            w.x = k.x;
            w.y = y;
            w.z = k.z;
            w.visMask = c.mask; // copy of source column-low voxel
            w.paletteIdx = c.pal;
            w.aoPacked[0] = c.ao[0];
            w.aoPacked[1] = c.ao[1];
            w.aoPacked[2] = c.ao[2];
            skirt.push_back(w);
        }
    }
    printf("[skirt] perimeter columns=%zu, fill voxels=%zu\n",
           perimLow.size(), skirt.size());

    // ---- Tile NX x NZ. Spacing = exact span -> tiles butt without overlap.
    const uint64_t totalOut = (uint64_t)(NX * NZ) * (uint64_t)(src.size() + skirt.size());
    printf("[tile] NX=%d NZ=%d span=(%d,%d) total voxels=%llu\n",
           NX, NZ, xSpan, zSpan, (unsigned long long)totalOut);

    // ---- Bucket into output chunks of D^3. Stream over tiles to keep peak
    // memory close to a single tile's worth.
    std::unordered_map<CK, std::vector<DiskVoxel>, CKHash> chunks;
    chunks.reserve((size_t)((totalOut / (uint64_t)(D * D * D)) + 1024));

    auto emit = [&](const WV& w, int32_t ox, int32_t oz)
    {
        int32_t X = w.x + ox;
        int32_t Y = w.y;
        int32_t Z = w.z + oz;
        CK k{X / D, Y / D, Z / D};
        DiskVoxel dv;
        dv.x = (uint8_t)(X % D);
        dv.y = (uint8_t)(Y % D);
        dv.z = (uint8_t)(Z % D);
        dv.visMask = w.visMask & 0x3Fu;
        dv.paletteIdx = w.paletteIdx;
        dv.aoPacked[0] = w.aoPacked[0];
        dv.aoPacked[1] = w.aoPacked[1];
        dv.aoPacked[2] = w.aoPacked[2];
        chunks[k].push_back(dv);
    };

    for (int gz = 0; gz < NZ; ++gz)
    {
        for (int gx = 0; gx < NX; ++gx)
        {
            int32_t ox = gx * xSpan;
            int32_t oz = gz * zSpan;
            for (const WV& w : src)
                emit(w, ox, oz);
            for (const WV& w : skirt)
                emit(w, ox, oz);
        }
        printf("[tile] row gz=%d done\n", gz);
    }
    printf("[chunk] %zu chunks\n", chunks.size());

    // ---- Flatten to disk arrays.
    std::vector<ChunkMeta> outMetas;
    std::vector<DiskVoxel> outVox;
    outMetas.reserve(chunks.size());
    {
        uint64_t totalV = 0;
        for (auto& kv : chunks)
            totalV += kv.second.size();
        outVox.reserve((size_t)totalV);
    }
    for (auto& kv : chunks)
    {
        ChunkMeta m;
        m.cx = (uint16_t)kv.first.cx;
        m.cy = (uint16_t)kv.first.cy;
        m.cz = (uint16_t)kv.first.cz;
        m._pad = 0;
        m.voxelCount = (uint32_t)kv.second.size();
        m.voxelOffset = (uint32_t)outVox.size();
        outMetas.push_back(m);
        outVox.insert(outVox.end(), kv.second.begin(), kv.second.end());
    }
    chunks.clear();

    // ---- Write VXL3.
    fs::create_directories(fs::path(outPath).parent_path());
    FILE* o = fopen(outPath, "wb");
    if (!o)
    {
        fprintf(stderr, "cannot open %s for write\n", outPath);
        return 1;
    }
    const char outMagic[4] = {'V', 'X', 'L', '3'};
    fwrite(outMagic, 1, 4, o);
    uint32_t ver = kAssetVersion;
    fwrite(&ver, sizeof(uint32_t), 1, o);
    uint32_t cdim = (uint32_t)D;
    uint32_t cct = (uint32_t)outMetas.size();
    uint32_t vct = (uint32_t)outVox.size();
    fwrite(&cdim, sizeof(uint32_t), 1, o);
    fwrite(&cct, sizeof(uint32_t), 1, o);
    fwrite(&vct, sizeof(uint32_t), 1, o);
    int32_t outOrigin[3] = {srcOrigin[0] + mn[0], srcOrigin[1] + mn[1], srcOrigin[2] + mn[2]};
    fwrite(outOrigin, sizeof(int32_t), 3, o);
    fwrite(srcSun, sizeof(float), 3, o);
    fwrite(&paletteCount, sizeof(uint32_t), 1, o);
    if (paletteCount)
        fwrite(palette.data(), sizeof(uint32_t), paletteCount, o);
    fwrite(outMetas.data(), sizeof(ChunkMeta), outMetas.size(), o);
    fwrite(outVox.data(), sizeof(DiskVoxel), outVox.size(), o);
    long bytes = ftell(o);
    fclose(o);
    printf("[out] %s: %ld bytes (%.2f MB), %u voxels, %u chunks\n",
           outPath, bytes, bytes / (1024.0 * 1024.0), vct, cct);

    // Delete stale baked sidecar.
    std::string vxb(outPath);
    size_t dot = vxb.find_last_of('.');
    if (dot != std::string::npos)
        vxb.resize(dot);
    vxb += ".vxb";
    std::error_code ec;
    if (fs::remove(vxb, ec))
        printf("Removed stale baked sidecar: %s\n", vxb.c_str());

    return 0;
}
