// vox2lw.cpp — convert a .vox (VXL3) into a .lw (LODWorld). Bakes all
// kLodCount LODs by downsampling, buckets per LOD into chunks + clusters,
// per-chunk palette-quantized to kPaletteSize. Writes raw (uncompressed)
// blobs per chunk; later phase will wrap with LZ4.
//
// Usage:
//   vox2lw.exe <in.vox> <out.lw>
// Defaults: assets/kingslanding.vox  assets/kingslanding.lw

#define _CRT_SECURE_NO_WARNINGS
#include "lodworld.h"
#include "asset_version.h"
#include "lz4.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <atomic>
#include <unordered_set>
#include <algorithm>
#include <filesystem>
#include <functional>

namespace fs = std::filesystem;

// ---------- source .vox raw types ----------
#pragma pack(push, 1)
struct VxlDiskVoxel
{
    uint8_t x, y, z, visMask;
    uint16_t paletteIdx;
    uint8_t aoPacked[3];
};
#pragma pack(pop)
static_assert(sizeof(VxlDiskVoxel) == 9, "");

struct VxlChunkMeta
{
    uint16_t cx, cy, cz, _pad;
    uint32_t voxelCount, voxelOffset;
};
static_assert(sizeof(VxlChunkMeta) == 16, "");

static inline uint8_t SrcAo(const VxlDiskVoxel& v, int fi)
{
    uint8_t n = (v.aoPacked[(fi * 4) >> 3] >> ((fi * 4) & 7)) & 0xFu;
    return (uint8_t)((n << 4) | n); // 0..255 in 16 steps
}

// ---------- helpers ----------
static inline int32_t FloorDiv(int32_t a, int32_t b)
{
    int32_t q = a / b;
    if ((a % b != 0) && ((a ^ b) < 0))
        --q;
    return q;
}
static inline int32_t FloorMod(int32_t a, int32_t b)
{
    int32_t m = a % b;
    if ((m != 0) && ((m ^ b) < 0))
        m += b;
    return m;
}

struct CellKey
{
    int32_t x, y, z;
    bool operator==(const CellKey& o) const
    {
        return x == o.x && y == o.y && z == o.z;
    }
};
struct CellHash
{
    size_t operator()(const CellKey& k) const noexcept
    {
        uint64_t h = (uint64_t)(uint32_t)k.x * 0x9E3779B97F4A7C15ull;
        h ^= (uint64_t)(uint32_t)k.y * 0xBF58476D1CE4E5B9ull;
        h ^= (uint64_t)(uint32_t)k.z * 0x94D049BB133111EBull;
        return (size_t)(h ^ (h >> 32));
    }
};

struct LodVox
{
    int32_t x, y, z;   // in LOD-cell coords (one cell per LOD-voxel)
    uint32_t color;    // 0x00BBGGRR averaged
    uint8_t visMask;   // OR of contributing src voxels
    uint8_t aoFace[6]; // max per face from contributors
};

// Aggregator during cell binning.
struct CellAcc
{
    uint32_t r = 0, g = 0, b = 0;
    uint32_t count = 0;
    uint8_t mask = 0;
    uint8_t aoMax[6] = {0, 0, 0, 0, 0, 0};
};

// Baker-local per-voxel record. Was DiskPoint when point pools were
// uploaded directly to GPU; now used solely as scratch while encoding
// the octet stream.
#pragma pack(push, 1)
struct DiskPoint
{
    uint8_t posX;        // 0..kChunkVoxX-1
    uint8_t posY;        // 0..kChunkVoxY-1
    uint8_t posZ;        // 0..kChunkVoxZ-1
    uint8_t palIdx;      // index into chunk's palette[]
    uint8_t visMask;     // low 6 bits = face visibility
    uint8_t aoPacked[3]; // 6 faces * 4 bits AO
};
#pragma pack(pop)
static_assert(sizeof(DiskPoint) == 8, "");

struct BakedChunk
{
    lw::DiskChunkHeader hdr;
    std::vector<uint32_t> palette;
    lw::DiskCluster clusters[lw::kClustersPerChunk];
    std::vector<DiskPoint> points;
};

// Per-cluster compression encoded data. Chosen ordering = smaller of
// {Y-major, Morton}.
struct ClusterEnc
{
    uint8_t orderMode = lw::kOrderYMajor;
    std::vector<uint8_t> bitGrid;
    std::vector<uint8_t> colors;
    std::vector<uint8_t> ao;      // legacy per-face AO (3 bytes/voxel) — unused in cellAO path
    std::vector<uint8_t> visMask; // legacy per-voxel visMask — unused in cellAO path
    std::vector<uint8_t> cellAo;  // 4 bits per AO-cell in chosen order; bytes packed (2 cells per byte)
    uint32_t cellAoCount = 0;
};

// Stripped: PointCS_Block format doesn't use AO or visMask. Defaults now off.
static bool g_storeAo = false;
static bool g_bakeAo = false;
static bool g_storeVisMask = false;
static bool g_storeCellAo = false;
static bool g_storeBlocks = true; // V2 octet stream for PointCS_Block (only output)
static uint64_t g_aoHisto[16] = {};
static uint64_t g_aoCount = 0;

// Tiny RAII helper for chunk-wide bit-grid (lookup over chunk in voxel coords).
struct ChunkBits
{
    static constexpr int W = lw::kChunkVoxX;
    static constexpr int H = lw::kChunkVoxY;
    static constexpr int D = lw::kChunkVoxZ;
    std::vector<uint8_t> b;
    ChunkBits() :
        b((size_t)((W * H * D + 7) / 8), 0)
    {
    }
    void set(int x, int y, int z)
    {
        size_t idx = (size_t)((y * D + z) * W + x);
        b[idx >> 3] |= (uint8_t)(1u << (idx & 7));
    }
    bool get(int x, int y, int z) const
    {
        if (x < 0 || y < 0 || z < 0 || x >= W || y >= H || z >= D)
            return false;
        size_t idx = (size_t)((y * D + z) * W + x);
        return (b[idx >> 3] >> (idx & 7)) & 1u;
    }
};

// `chunkBits` and `clusterOrigin{X,Y,Z}` are only used when g_storeCellAo is
// true; pass nullptr/zeros otherwise.
static ClusterEnc EncodeClusterTryBoth(const DiskPoint* pts, uint32_t numPts,
                                       uint64_t& outYBytes, uint64_t& outMBytes,
                                       const ChunkBits* chunkBits = nullptr,
                                       int clusterOriginX = 0,
                                       int clusterOriginY = 0,
                                       int clusterOriginZ = 0)
{
    auto tryOrder = [&](lw::LwOrderMode mode) -> ClusterEnc
    {
        uint8_t grid[lw::kClusterCellCount];
        std::memset(grid, 0, sizeof(grid));
        uint8_t palAt[lw::kClusterCellCount];
        uint8_t aoAt[lw::kClusterCellCount][3];
        uint8_t vmAt[lw::kClusterCellCount];
        for (uint32_t i = 0; i < numPts; ++i)
        {
            const DiskPoint& p = pts[i];
            uint32_t lx = (uint32_t)(p.posX % lw::kClusterVoxX);
            uint32_t ly = (uint32_t)(p.posY % lw::kClusterVoxY);
            uint32_t lz = (uint32_t)(p.posZ % lw::kClusterVoxZ);
            uint32_t idx = lw::LwCellIndex(mode, lx, ly, lz);
            grid[idx] = 1;
            palAt[idx] = p.palIdx;
            aoAt[idx][0] = p.aoPacked[0];
            aoAt[idx][1] = p.aoPacked[1];
            aoAt[idx][2] = p.aoPacked[2];
            vmAt[idx] = p.visMask & 0x3F;
        }
        ClusterEnc e;
        e.orderMode = (uint8_t)mode;
        lw::RleEncodeBitGrid(grid, e.bitGrid);
        e.colors.reserve(numPts);
        if (g_storeVisMask)
            e.visMask.reserve((size_t)numPts);
        // Variable-length AO: per voxel, pack popcount(visMask) nibbles
        // (4 bits per visible face) back-to-back. Hidden-face AO is dropped
        // entirely. Loader reads back using its own recomputed visMask.
        uint8_t aoBitBuf = 0;
        int aoBitCnt = 0;
        auto pushNib = [&](uint8_t nib)
        {
            if (aoBitCnt == 0)
            {
                aoBitBuf = (uint8_t)(nib & 0xF);
                aoBitCnt = 4;
            }
            else
            {
                aoBitBuf |= (uint8_t)((nib & 0xF) << 4);
                e.ao.push_back(aoBitBuf);
                aoBitCnt = 0;
            }
        };
        for (uint32_t i = 0; i < lw::kClusterCellCount; ++i)
        {
            if (!grid[i])
                continue;
            e.colors.push_back(palAt[i]);
            if (g_storeAo)
            {
                uint8_t vm = vmAt[i];
                uint32_t ap = (uint32_t)aoAt[i][0] | ((uint32_t)aoAt[i][1] << 8) | ((uint32_t)aoAt[i][2] << 16);
                for (int fi = 0; fi < 6; ++fi)
                {
                    if (!((vm >> fi) & 1u))
                        continue;
                    pushNib((uint8_t)((ap >> (fi * 4)) & 0xF));
                }
            }
            if (g_storeVisMask)
                e.visMask.push_back(vmAt[i]);
        }
        if (g_storeAo && aoBitCnt > 0)
            e.ao.push_back(aoBitBuf); // flush tail nibble

        // ---- Cell-AO bitstream ----
        // Walk cluster cells in chosen order. For each EMPTY cell adjacent to
        // a solid voxel (in CHUNK-wide grid), bake 4-bit AO from 3x3x3
        // occluder count and pack 2 cells per byte. Decoder walks same order.
        if (g_storeCellAo && chunkBits)
        {
            uint32_t aoCount = 0;
            uint8_t pending = 0;
            bool half = false;
            for (uint32_t i = 0; i < lw::kClusterCellCount; ++i)
            {
                if (grid[i])
                    continue; // solid cell — no AO
                uint32_t lx, ly, lz;
                lw::LwCellCoord(mode, i, lx, ly, lz);
                int wx = clusterOriginX + (int)lx;
                int wy = clusterOriginY + (int)ly;
                int wz = clusterOriginZ + (int)lz;
                // Adjacency check (6-neighbour) in chunk-wide grid.
                if (!(chunkBits->get(wx + 1, wy, wz) || chunkBits->get(wx - 1, wy, wz) ||
                      chunkBits->get(wx, wy + 1, wz) || chunkBits->get(wx, wy - 1, wz) ||
                      chunkBits->get(wx, wy, wz + 1) || chunkBits->get(wx, wy, wz - 1)))
                    continue;
                // Bake AO: count solid face-adjacent neighbours (6 max).
                // A flat surface cell has 1 face neighbour (the voxel below)
                // and full hemisphere sky → should be max bright. Edges +
                // corners around the cell at distance >1 don't block the open
                // sky/light above this cell, so we ignore them here.
                // N=1 → ao 15 (bright); N>=4 → ao 0 (dark pocket).
                int faceHits = 0;
                if (chunkBits->get(wx + 1, wy, wz))
                    ++faceHits;
                if (chunkBits->get(wx - 1, wy, wz))
                    ++faceHits;
                if (chunkBits->get(wx, wy + 1, wz))
                    ++faceHits;
                if (chunkBits->get(wx, wy - 1, wz))
                    ++faceHits;
                if (chunkBits->get(wx, wy, wz + 1))
                    ++faceHits;
                if (chunkBits->get(wx, wy, wz - 1))
                    ++faceHits;
                int ao4 = 15 - (faceHits - 1) * 5; // 1→15, 2→10, 3→5, 4+→0
                if (ao4 < 0)
                    ao4 = 0;
                if (ao4 > 15)
                    ao4 = 15;
                g_aoHisto[ao4]++;
                g_aoCount++;
                if (!half)
                {
                    pending = (uint8_t)ao4;
                    half = true;
                }
                else
                {
                    e.cellAo.push_back(pending | (uint8_t)(ao4 << 4));
                    half = false;
                }
                ++aoCount;
            }
            if (half)
                e.cellAo.push_back(pending);
            e.cellAoCount = aoCount;
        }
        return e;
    };
    ClusterEnc y = tryOrder(lw::kOrderYMajor);
    ClusterEnc m = tryOrder(lw::kOrderMorton);
    uint64_t ySize = y.bitGrid.size() + y.colors.size() + y.ao.size() + y.visMask.size() + y.cellAo.size();
    uint64_t mSize = m.bitGrid.size() + m.colors.size() + m.ao.size() + m.visMask.size() + m.cellAo.size();
    outYBytes += ySize;
    outMBytes += mSize;
    if (ySize <= mSize)
        return y;
    return m;
}

// ---------- main ----------
int main(int argc, char** argv)
{
    const char* inPath = (argc > 1) ? argv[1] : "assets/kingslanding.vox";
    const char* outPath = (argc > 2) ? argv[2] : "assets/kingslanding.lw";
    for (int a = 1; a < argc; ++a)
    {
        if (strcmp(argv[a], "--ao") == 0)
            g_storeAo = true;
        else if (strcmp(argv[a], "--noao") == 0)
            g_bakeAo = false;
        else if (strcmp(argv[a], "--vm") == 0)
            g_storeVisMask = true;
        else if (strcmp(argv[a], "--cellao") == 0)
            g_storeCellAo = true;
        else if (strcmp(argv[a], "--full") == 0)
        {
            g_storeAo = true;
            g_storeVisMask = true;
        }
    }
    printf("[cfg] AO=%s (bake=%s)  visMask=%s  cellAO=%s\n",
           g_storeAo ? "ON" : "OFF", g_bakeAo ? "ON" : "OFF",
           g_storeVisMask ? "ON" : "OFF", g_storeCellAo ? "ON" : "OFF");
    fflush(stdout);

    // ---- read source: detect magic VXL3 vs MagicaVoxel "VOX " ----
    FILE* f = fopen(inPath, "rb");
    if (!f)
    {
        fprintf(stderr, "cannot open %s\n", inPath);
        return 1;
    }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4)
    {
        fclose(f);
        fprintf(stderr, "short read\n");
        return 1;
    }

    struct SrcVox
    {
        int32_t x, y, z;
        uint32_t color;
        uint8_t mask;
        uint8_t aoFace[6];
    };
    std::vector<SrcVox> src;
    int32_t worldMn[3] = {INT32_MAX, INT32_MAX, INT32_MAX};
    int32_t worldMx[3] = {-INT32_MAX, -INT32_MAX, -INT32_MAX};
    int32_t srcOrigin[3] = {0, 0, 0};
    float srcSun[3] = {0.4f, 0.8f, 0.2f};

    if (memcmp(magic, "VXL3", 4) == 0)
    {
        // ---- VXL3 (custom) ----
        uint32_t version = 0;
        fread(&version, sizeof(uint32_t), 1, f);
        if (version != kAssetVersion)
        {
            fclose(f);
            fprintf(stderr, "VXL3 version mismatch: %u vs %u\n", version, kAssetVersion);
            return 1;
        }
        uint32_t chunkDim = 0, srcChunkCount = 0, totalVoxels = 0;
        fread(&chunkDim, sizeof(uint32_t), 1, f);
        fread(&srcChunkCount, sizeof(uint32_t), 1, f);
        fread(&totalVoxels, sizeof(uint32_t), 1, f);
        fread(srcOrigin, sizeof(int32_t), 3, f);
        fread(srcSun, sizeof(float), 3, f);
        uint32_t srcPalCount = 0;
        fread(&srcPalCount, sizeof(uint32_t), 1, f);
        std::vector<uint32_t> srcPalette(srcPalCount);
        if (srcPalCount)
            fread(srcPalette.data(), sizeof(uint32_t), srcPalCount, f);
        std::vector<VxlChunkMeta> srcMetas(srcChunkCount);
        if (srcChunkCount)
            fread(srcMetas.data(), sizeof(VxlChunkMeta), srcChunkCount, f);
        std::vector<VxlDiskVoxel> srcVox(totalVoxels);
        if (totalVoxels)
            fread(srcVox.data(), sizeof(VxlDiskVoxel), totalVoxels, f);
        fclose(f);
        printf("[in] VXL3 %s chunkDim=%u srcChunks=%u voxels=%u palette=%u\n",
               inPath, chunkDim, srcChunkCount, totalVoxels, srcPalCount);
        const int32_t D = (int32_t)chunkDim;
        src.reserve(totalVoxels);
        for (uint32_t ci = 0; ci < srcChunkCount; ++ci)
        {
            const VxlChunkMeta& m = srcMetas[ci];
            int32_t bx = (int32_t)m.cx * D, by = (int32_t)m.cy * D, bz = (int32_t)m.cz * D;
            const VxlDiskVoxel* vs = srcVox.data() + m.voxelOffset;
            for (uint32_t i = 0; i < m.voxelCount; ++i)
            {
                const VxlDiskVoxel& dv = vs[i];
                if (dv.visMask == 0)
                    continue;
                SrcVox s;
                s.x = bx + (int32_t)dv.x;
                s.y = by + (int32_t)dv.y;
                s.z = bz + (int32_t)dv.z;
                uint32_t col = (dv.paletteIdx < srcPalette.size()) ? srcPalette[dv.paletteIdx] : 0xFF000000u;
                s.color = col & 0x00FFFFFFu;
                s.mask = 0x3F;
                for (int fi = 0; fi < 6; ++fi)
                    s.aoFace[fi] = 0xFF;
                src.push_back(s);
                if (s.x < worldMn[0])
                    worldMn[0] = s.x;
                if (s.x > worldMx[0])
                    worldMx[0] = s.x;
                if (s.y < worldMn[1])
                    worldMn[1] = s.y;
                if (s.y > worldMx[1])
                    worldMx[1] = s.y;
                if (s.z < worldMn[2])
                    worldMn[2] = s.z;
                if (s.z > worldMx[2])
                    worldMx[2] = s.z;
            }
        }
    }
    else if (memcmp(magic, "VOX ", 4) == 0)
    {
        // ---- MagicaVoxel native ----
        uint32_t version = 0;
        fread(&version, sizeof(uint32_t), 1, f);
        printf("[in] MagicaVoxel %s (v%u)\n", inPath, version);
        struct SubSize
        {
            int32_t sx, sy, sz;
        };
        std::vector<SubSize> sizes;
        struct MvVox
        {
            uint8_t x, y, z, palIdx;
        };
        std::vector<std::vector<MvVox>> subVoxels;
        std::vector<uint32_t> mvPalette(256, 0xFFFFFFFFu);
        bool sawRgba = false;

        // Scene-graph node table: nTRN/nGRP/nSHP indexed by nodeId.
        struct NodeT
        {
            enum Type
            {
                Trn,
                Grp,
                Shp
            };
            Type type;
            int32_t child = -1;             // nTRN
            std::vector<int32_t> children;  // nGRP
            std::vector<int32_t> models;    // nSHP
            int32_t tx = 0, ty = 0, tz = 0; // nTRN translation (frame 0 _t)
        };
        std::unordered_map<int32_t, NodeT> nodes;

        auto readU32 = [&](uint32_t& v) -> bool
        { return fread(&v, 4, 1, f) == 1; };
        auto readI32 = [&](int32_t& v) -> bool
        { return fread(&v, 4, 1, f) == 1; };
        auto readStr = [&]() -> std::string
        {
            int32_t len = 0;
            if (fread(&len, 4, 1, f) != 1 || len < 0)
                return std::string();
            std::string s(len, '\0');
            if (len > 0)
                fread(s.data(), 1, len, f);
            return s;
        };
        auto readDict = [&]() -> std::unordered_map<std::string, std::string>
        {
            std::unordered_map<std::string, std::string> m;
            int32_t numPairs = 0;
            fread(&numPairs, 4, 1, f);
            for (int i = 0; i < numPairs; ++i)
            {
                std::string k = readStr();
                std::string v = readStr();
                m[k] = v;
            }
            return m;
        };

        std::function<bool(long)> readChunks = [&](long endOff) -> bool
        {
            while (ftell(f) < endOff)
            {
                char id[4];
                uint32_t cBytes = 0, kBytes = 0;
                if (fread(id, 1, 4, f) != 4)
                    return false;
                if (!readU32(cBytes))
                    return false;
                if (!readU32(kBytes))
                    return false;
                long after = ftell(f) + (long)cBytes + (long)kBytes;
                if (memcmp(id, "SIZE", 4) == 0)
                {
                    uint32_t sx, sy, sz;
                    readU32(sx);
                    readU32(sy);
                    readU32(sz);
                    sizes.push_back({(int32_t)sx, (int32_t)sy, (int32_t)sz});
                }
                else if (memcmp(id, "XYZI", 4) == 0)
                {
                    uint32_t n = 0;
                    readU32(n);
                    std::vector<MvVox> vs(n);
                    if (n)
                        fread(vs.data(), sizeof(MvVox), n, f);
                    subVoxels.push_back(std::move(vs));
                }
                else if (memcmp(id, "RGBA", 4) == 0)
                {
                    for (int i = 0; i < 256; ++i)
                    {
                        uint8_t r, g, b, a;
                        fread(&r, 1, 1, f);
                        fread(&g, 1, 1, f);
                        fread(&b, 1, 1, f);
                        fread(&a, 1, 1, f);
                        mvPalette[i] = (uint32_t)r | ((uint32_t)g << 8) | ((uint32_t)b << 16) | ((uint32_t)a << 24);
                    }
                    sawRgba = true;
                }
                else if (memcmp(id, "nTRN", 4) == 0)
                {
                    int32_t nodeId = 0;
                    readI32(nodeId);
                    readDict(); // node attribs
                    int32_t childId = 0;
                    readI32(childId);
                    int32_t reserved = 0;
                    readI32(reserved);
                    int32_t layerId = 0;
                    readI32(layerId);
                    int32_t numFrames = 0;
                    readI32(numFrames);
                    int32_t tx = 0, ty = 0, tz = 0;
                    for (int i = 0; i < numFrames; ++i)
                    {
                        auto frame = readDict();
                        if (i == 0)
                        {
                            auto it = frame.find("_t");
                            if (it != frame.end())
                            {
                                sscanf(it->second.c_str(), "%d %d %d", &tx, &ty, &tz);
                            }
                        }
                    }
                    NodeT n;
                    n.type = NodeT::Trn;
                    n.child = childId;
                    n.tx = tx;
                    n.ty = ty;
                    n.tz = tz;
                    nodes[nodeId] = n;
                }
                else if (memcmp(id, "nGRP", 4) == 0)
                {
                    int32_t nodeId = 0;
                    readI32(nodeId);
                    readDict();
                    int32_t numChildren = 0;
                    readI32(numChildren);
                    NodeT n;
                    n.type = NodeT::Grp;
                    n.children.resize(numChildren);
                    for (int i = 0; i < numChildren; ++i)
                        readI32(n.children[i]);
                    nodes[nodeId] = n;
                }
                else if (memcmp(id, "nSHP", 4) == 0)
                {
                    int32_t nodeId = 0;
                    readI32(nodeId);
                    readDict();
                    int32_t numModels = 0;
                    readI32(numModels);
                    NodeT n;
                    n.type = NodeT::Shp;
                    for (int i = 0; i < numModels; ++i)
                    {
                        int32_t modelId = 0;
                        readI32(modelId);
                        readDict();
                        n.models.push_back(modelId);
                    }
                    nodes[nodeId] = n;
                }
                else
                {
                    fseek(f, cBytes, SEEK_CUR);
                }
                if (kBytes > 0)
                {
                    long childEnd = ftell(f) + (long)kBytes;
                    if (!readChunks(childEnd))
                        return false;
                }
                fseek(f, after, SEEK_SET);
            }
            return true;
        };

        char mainId[4];
        uint32_t mainC = 0, mainK = 0;
        fread(mainId, 1, 4, f);
        readU32(mainC);
        readU32(mainK);
        fseek(f, mainC, SEEK_CUR);
        long mainEnd = ftell(f) + (long)mainK;
        readChunks(mainEnd);
        fclose(f);

        // Build list of model instances with accumulated translation. Use
        // scene-graph if present, else place each sub-model at origin.
        struct Inst
        {
            int32_t modelId;
            int32_t tx, ty, tz;
        };
        std::vector<Inst> insts;
        if (!nodes.empty())
        {
            std::function<void(int32_t, int32_t, int32_t, int32_t)> visit =
                [&](int32_t id, int32_t tx, int32_t ty, int32_t tz)
            {
                auto it = nodes.find(id);
                if (it == nodes.end())
                    return;
                const NodeT& n = it->second;
                if (n.type == NodeT::Trn)
                {
                    visit(n.child, tx + n.tx, ty + n.ty, tz + n.tz);
                }
                else if (n.type == NodeT::Grp)
                {
                    for (int32_t c : n.children)
                        visit(c, tx, ty, tz);
                }
                else if (n.type == NodeT::Shp)
                {
                    for (int32_t m : n.models)
                        insts.push_back({m, tx, ty, tz});
                }
            };
            visit(0, 0, 0, 0);
        }
        else
        {
            for (int i = 0; i < (int)subVoxels.size(); ++i)
                insts.push_back({i, 0, 0, 0});
        }
        printf("[in] MV: %zu sub-models, %zu instances, palette %s\n",
               subVoxels.size(), insts.size(), sawRgba ? "from RGBA chunk" : "default (white)");

        // Emit voxels. MV translation = model CENTER in MV world. World pos of
        // a voxel = (mv - size/2) + translation. Then axis swap Y↔Z for engine.
        size_t totalVox = 0;
        for (auto& inst : insts)
        {
            if (inst.modelId >= 0 && inst.modelId < (int)subVoxels.size())
                totalVox += subVoxels[inst.modelId].size();
        }
        src.reserve(totalVox);
        for (const auto& inst : insts)
        {
            if (inst.modelId < 0 || inst.modelId >= (int)subVoxels.size())
                continue;
            const auto& vs = subVoxels[inst.modelId];
            const auto& sz = sizes[inst.modelId];
            int hx = sz.sx / 2, hy = sz.sy / 2, hz = sz.sz / 2;
            for (const MvVox& mv : vs)
            {
                int mvX = (int)mv.x - hx + inst.tx;
                int mvY = (int)mv.y - hy + inst.ty;
                int mvZ = (int)mv.z - hz + inst.tz;
                SrcVox s;
                s.x = mvX;
                s.y = mvZ; // axis swap Y↔Z (MV Z-up → engine Y-up)
                s.z = mvY;
                uint32_t pi = mv.palIdx;
                uint32_t col = (pi < 256) ? mvPalette[pi] : 0xFFFFFFFFu;
                s.color = col & 0x00FFFFFFu;
                s.mask = 0x3F;
                for (int fi = 0; fi < 6; ++fi)
                    s.aoFace[fi] = 0xFF;
                src.push_back(s);
                if (s.x < worldMn[0])
                    worldMn[0] = s.x;
                if (s.x > worldMx[0])
                    worldMx[0] = s.x;
                if (s.y < worldMn[1])
                    worldMn[1] = s.y;
                if (s.y > worldMx[1])
                    worldMx[1] = s.y;
                if (s.z < worldMn[2])
                    worldMn[2] = s.z;
                if (s.z > worldMx[2])
                    worldMx[2] = s.z;
            }
        }
    }
    else
    {
        fclose(f);
        fprintf(stderr, "bad magic (need VXL3 or VOX )\n");
        return 1;
    }

    printf("[in] src voxels: %zu  AABB: x[%d..%d] y[%d..%d] z[%d..%d]\n",
           src.size(), worldMn[0], worldMx[0], worldMn[1], worldMx[1], worldMn[2], worldMx[2]);
    fflush(stdout);

    // ========================================================================
    // Per-source-voxel per-face AO bake via hemisphere raycast.
    // Build a dense bit-grid of source solidity, then for each exposed face
    // cast kAoSamples cosine-weighted rays into the outward hemisphere and
    // DDA-step up to kAoMaxSteps source cells. ao = 1 - hits/samples, scaled
    // to 0..255 and stored in SrcVox.aoFace[f]. LOD bake later averages these
    // per-source values across the LOD voxel's face area.
    // ========================================================================
    const int spanX0 = worldMx[0] - worldMn[0] + 1;
    const int spanY0 = worldMx[1] - worldMn[1] + 1;
    const int spanZ0 = worldMx[2] - worldMn[2] + 1;
    const size_t bmBytes = ((size_t)spanX0 * spanY0 * spanZ0 + 7) / 8;
    printf("[bake] building filledBM bit-grid: span %dx%dx%d = %.2f MB\n",
           spanX0, spanY0, spanZ0, bmBytes / (1024.0 * 1024.0));
    fflush(stdout);
    std::vector<uint8_t> filledBM(bmBytes, 0);
    auto bIdx = [&](int x, int y, int z) -> size_t
    {
        return (size_t)((z - worldMn[2]) * spanY0 + (y - worldMn[1])) * (size_t)spanX0 + (size_t)(x - worldMn[0]);
    };
    for (const SrcVox& s : src)
    {
        size_t i = bIdx(s.x, s.y, s.z);
        filledBM[i >> 3] |= (uint8_t)(1u << (i & 7));
    }
    auto isFilled = [&](int x, int y, int z) -> bool
    {
        if (x < worldMn[0] || x > worldMx[0] ||
            y < worldMn[1] || y > worldMx[1] ||
            z < worldMn[2] || z > worldMx[2])
            return false;
        size_t i = bIdx(x, y, z);
        return (filledBM[i >> 3] >> (i & 7)) & 1u;
    };
    // Map source voxel coord -> index in `src` so LOD bake can read the
    // pre-baked per-face AO of any specific source voxel. Skip entirely when
    // --noao: LOD bake just returns constant bright AO without lookup.
    std::unordered_map<CellKey, uint32_t, CellHash> srcIdx;
    if (g_bakeAo)
    {
        printf("[bake] building srcIdx hash: %zu entries (~%.2f MB est)\n",
               src.size(), (src.size() * 32.0) / (1024.0 * 1024.0));
        fflush(stdout);
        srcIdx.reserve(src.size());
        for (size_t i = 0; i < src.size(); ++i)
        {
            srcIdx[{src[i].x, src[i].y, src[i].z}] = (uint32_t)i;
            if ((i & 0xFFFFF) == 0 && i > 0)
            {
                printf("[bake] srcIdx: %zu / %zu (%.1f%%)\n",
                       i, src.size(), 100.0 * (double)i / (double)src.size());
                fflush(stdout);
            }
        }
    }
    else
    {
        printf("[bake] srcIdx skipped (--noao).\n");
        fflush(stdout);
    }

    if (!g_bakeAo)
    {
        printf("[bake] AO bake skipped (--noao). All faces default to bright (255).\n");
        for (SrcVox& s : src)
            for (int fi = 0; fi < 6; ++fi)
                s.aoFace[fi] = 255;
        fflush(stdout);
    }
    else
    {
        // Face delta: 0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z (matches engine convention).
        static const int kFaceDelta[6][3] = {
            {1, 0, 0},
            {-1, 0, 0},
            {0, 1, 0},
            {0, -1, 0},
            {0, 0, 1},
            {0, 0, -1},
        };
        const int kAoSamples = 48;
        const int kAoMaxSteps = 24;
        const float kTwoPi = 6.2831853071795864f;
        auto Hammersley = [](int i, int N, float& u, float& v)
        {
            u = (float)i / (float)N;
            uint32_t b = (uint32_t)i;
            b = ((b & 0x55555555u) << 1) | ((b & 0xAAAAAAAAu) >> 1);
            b = ((b & 0x33333333u) << 2) | ((b & 0xCCCCCCCCu) >> 2);
            b = ((b & 0x0F0F0F0Fu) << 4) | ((b & 0xF0F0F0F0u) >> 4);
            b = ((b & 0x00FF00FFu) << 8) | ((b & 0xFF00FF00u) >> 8);
            b = (b << 16) | (b >> 16);
            v = (float)b * 2.3283064365386963e-10f;
        };
        printf("[AO] hemisphere bake start: %zu src voxels, %d samples, %d max steps\n",
               src.size(), kAoSamples, kAoMaxSteps);
        fflush(stdout);
        const int64_t N = (int64_t)src.size();
        uint64_t totalRays = 0, totalHits = 0;
        std::atomic<int64_t> progress{0};
        const int64_t reportStride = (N > 100) ? (N / 100) : 1;
#pragma omp parallel for schedule(dynamic, 256) reduction(+ : totalRays) reduction(+ : totalHits)
        for (int64_t li = 0; li < N; ++li)
        {
            SrcVox& vd = src[(size_t)li];
            int vx = vd.x, vy = vd.y, vz = vd.z;
            // Cranley-Patterson rotation per voxel kills banding on flats.
            uint32_t h = (uint32_t)vx * 0x9E3779B1u ^ (uint32_t)vy * 0x85EBCA77u ^ (uint32_t)vz * 0xC2B2AE3Du;
            h ^= h >> 16;
            h *= 0x7FEB352Du;
            h ^= h >> 15;
            h *= 0x846CA68Bu;
            h ^= h >> 16;
            float jitU = (float)(h & 0xFFFFu) * (1.0f / 65536.0f);
            float jitV = (float)((h >> 16) & 0xFFFFu) * (1.0f / 65536.0f);
            for (int fi = 0; fi < 6; ++fi)
                vd.aoFace[fi] = 0;
            for (int fi = 0; fi < 6; ++fi)
            {
                int nxF = vx + kFaceDelta[fi][0];
                int nyF = vy + kFaceDelta[fi][1];
                int nzF = vz + kFaceDelta[fi][2];
                if (isFilled(nxF, nyF, nzF))
                    continue; // interior face
                float Nx = (float)kFaceDelta[fi][0];
                float Ny = (float)kFaceDelta[fi][1];
                float Nz = (float)kFaceDelta[fi][2];
                float Tx, Ty, Tz, Bx, By, Bz;
                if (fi < 2)
                {
                    Tx = 0;
                    Ty = 1;
                    Tz = 0;
                    Bx = 0;
                    By = 0;
                    Bz = 1;
                }
                else if (fi < 4)
                {
                    Tx = 1;
                    Ty = 0;
                    Tz = 0;
                    Bx = 0;
                    By = 0;
                    Bz = 1;
                }
                else
                {
                    Tx = 1;
                    Ty = 0;
                    Tz = 0;
                    Bx = 0;
                    By = 1;
                    Bz = 0;
                }
                int hits = 0;
                for (int s = 0; s < kAoSamples; ++s)
                {
                    float u, v;
                    Hammersley(s, kAoSamples, u, v);
                    u += jitU;
                    u -= floorf(u);
                    v += jitV;
                    v -= floorf(v);
                    float r = sqrtf(u);
                    float theta = kTwoPi * v;
                    float a = r * cosf(theta);
                    float b = r * sinf(theta);
                    float c = sqrtf(fmaxf(0.0f, 1.0f - u));
                    float dx = a * Tx + b * Bx + c * Nx;
                    float dy = a * Ty + b * By + c * Ny;
                    float dz = a * Tz + b * Bz + c * Nz;
                    float ox = (float)vx + 0.5f + 0.501f * Nx;
                    float oy = (float)vy + 0.5f + 0.501f * Ny;
                    float oz = (float)vz + 0.5f + 0.501f * Nz;
                    int ix = (int)floorf(ox), iy = (int)floorf(oy), iz = (int)floorf(oz);
                    float fx = ox - (float)ix, fy = oy - (float)iy, fz = oz - (float)iz;
                    int sx = dx > 0.0f ? 1 : (dx < 0.0f ? -1 : 0);
                    int sy = dy > 0.0f ? 1 : (dy < 0.0f ? -1 : 0);
                    int sz = dz > 0.0f ? 1 : (dz < 0.0f ? -1 : 0);
                    float invDx = (fabsf(dx) > 1e-6f) ? 1.0f / fabsf(dx) : 1e30f;
                    float invDy = (fabsf(dy) > 1e-6f) ? 1.0f / fabsf(dy) : 1e30f;
                    float invDz = (fabsf(dz) > 1e-6f) ? 1.0f / fabsf(dz) : 1e30f;
                    float tmx = sx > 0 ? (1.0f - fx) * invDx : sx < 0 ? fx * invDx
                                                                      : 1e30f;
                    float tmy = sy > 0 ? (1.0f - fy) * invDy : sy < 0 ? fy * invDy
                                                                      : 1e30f;
                    float tmz = sz > 0 ? (1.0f - fz) * invDz : sz < 0 ? fz * invDz
                                                                      : 1e30f;
                    bool hit = false;
                    for (int step = 0; step < kAoMaxSteps; ++step)
                    {
                        if (tmx < tmy && tmx < tmz)
                        {
                            ix += sx;
                            tmx += invDx;
                        }
                        else if (tmy < tmz)
                        {
                            iy += sy;
                            tmy += invDy;
                        }
                        else
                        {
                            iz += sz;
                            tmz += invDz;
                        }
                        if (isFilled(ix, iy, iz))
                        {
                            hit = true;
                            break;
                        }
                    }
                    if (hit)
                        ++hits;
                    ++totalRays;
                }
                totalHits += hits;
                float faceAo = 1.0f - (float)hits / (float)kAoSamples;
                int aoByte = (int)(faceAo * 255.0f + 0.5f);
                if (aoByte < 1)
                    aoByte = 1;
                if (aoByte > 255)
                    aoByte = 255;
                vd.aoFace[fi] = (uint8_t)aoByte;
            }
            int64_t done = progress.fetch_add(1, std::memory_order_relaxed) + 1;
            if (done % reportStride == 0)
            {
                int pct = (int)((done * 100) / N);
#pragma omp critical
                {
                    printf("[AO] %d%% (%lld / %lld voxels)\n", pct, (long long)done, (long long)N);
                    fflush(stdout);
                }
            }
        }
        printf("[AO] bake done: %llu rays, %.1f%% hit\n",
               (unsigned long long)totalRays,
               totalRays ? 100.0 * (double)totalHits / (double)totalRays : 0.0);
        fflush(stdout);
    } // end if/else (g_bakeAo)

    // ---- build all LODs ----
    std::vector<std::vector<BakedChunk>> lods(lw::kLodCount);

    for (int L = 0; L < lw::kLodCount; ++L)
    {
        const int32_t step = 1 << L;
        printf("[LOD %d] starting (step=%d)\n", L, step);
        fflush(stdout);

        // Aggregate src voxels into LOD-step cells.
        // LOD0 (step=1): every source voxel = its own cell. Hashmap insert per
        // voxel is the bottleneck for large scenes (2B+ entries). Skip the
        // hashmap entirely for LOD0 — no aggregation needed.
        std::unordered_map<CellKey, CellAcc, CellHash> cells;
        if (L > 0)
        {
            cells.reserve(src.size() / (size_t)(step * step * step) + 1024);
            for (const SrcVox& v : src)
            {
                CellKey k{FloorDiv(v.x, step), FloorDiv(v.y, step), FloorDiv(v.z, step)};
                CellAcc& a = cells[k];
                a.r += (v.color >> 0) & 0xFF;
                a.g += (v.color >> 8) & 0xFF;
                a.b += (v.color >> 16) & 0xFF;
                a.mask |= v.mask;
                for (int fi = 0; fi < 6; ++fi)
                {
                    if (v.aoFace[fi] > a.aoMax[fi])
                        a.aoMax[fi] = v.aoFace[fi];
                }
                ++a.count;
            }
        }
        // Flatten + per-chunk bucket.
        struct ChunkKey
        {
            int32_t gx, gy, gz;
            bool operator==(const ChunkKey& o) const
            {
                return gx == o.gx && gy == o.gy && gz == o.gz;
            }
        };
        struct ChunkHash
        {
            size_t operator()(const ChunkKey& k) const noexcept
            {
                uint64_t h = (uint64_t)(uint32_t)k.gx * 0x9E3779B97F4A7C15ull;
                h ^= (uint64_t)(uint32_t)k.gy * 0xBF58476D1CE4E5B9ull;
                h ^= (uint64_t)(uint32_t)k.gz * 0x94D049BB133111EBull;
                return (size_t)(h ^ (h >> 32));
            }
        };
        std::unordered_map<ChunkKey, std::vector<LodVox>, ChunkHash> chunkVox;
        if (L == 0)
        {
            // Direct path: each src voxel becomes one LodVox. Skip cells map.
            for (const SrcVox& s : src)
            {
                LodVox v;
                v.x = s.x;
                v.y = s.y;
                v.z = s.z;
                v.color = s.color & 0x00FFFFFFu;
                v.visMask = s.mask;
                for (int fi = 0; fi < 6; ++fi)
                    v.aoFace[fi] = s.aoFace[fi];
                ChunkKey ck{FloorDiv(v.x, lw::kChunkVoxX),
                            FloorDiv(v.y, lw::kChunkVoxY),
                            FloorDiv(v.z, lw::kChunkVoxZ)};
                chunkVox[ck].push_back(v);
            }
        }
        else
        {
            for (auto& kv : cells)
            {
                const CellKey& k = kv.first;
                const CellAcc& a = kv.second;
                LodVox v;
                v.x = k.x;
                v.y = k.y;
                v.z = k.z;
                uint32_t r = a.r / a.count, g = a.g / a.count, b = a.b / a.count;
                v.color = r | (g << 8) | (b << 16);
                v.visMask = a.mask;
                for (int fi = 0; fi < 6; ++fi)
                    v.aoFace[fi] = a.aoMax[fi];
                ChunkKey ck{FloorDiv(v.x, lw::kChunkVoxX),
                            FloorDiv(v.y, lw::kChunkVoxY),
                            FloorDiv(v.z, lw::kChunkVoxZ)};
                chunkVox[ck].push_back(v);
            }
        }
        // Keep `cells` alive — used below as a global solidity oracle so
        // per-face AO can sample voxels in adjacent chunks (avoids seams).
        printf("[LOD %d] chunks=%zu (step=%d)\n", L, chunkVox.size(), step);

        // Measurement totals for this LOD.
        uint64_t lodFaceCount = 0;
        uint64_t lodVoxelCount = 0;
        uint64_t lodInteriorVoxels = 0;
        uint64_t lodBoundaryFaces = 0;
        // Sub-palette stats: histogram of bits-per-idx needed per cluster.
        uint64_t lodSubPalBins[5] = {0, 0, 0, 0, 0}; // 1,2,4,8-bit clusters + n>=129
        uint64_t lodColorBits[4] = {0, 0, 0, 0};     // total bits if {fixed8, perCluster1,2,4,8 best}
        uint64_t lodSubPalHdrBytes = 0;              // sum of sub-palette index tables (1 byte per unique color used)
        // Per-cluster sub-palette stats for AO (24-bit aoPacked words).
        uint64_t lodAoSubBins[5] = {0, 0, 0, 0, 0}; // 1/2/4/8-bit clusters + giveup
        uint64_t lodAoBitsAdapt = 0;                // total bits via sub-pal
        uint64_t lodAoHdrBytes = 0;                 // 3 B per unique aoPacked in sub-pal
        // Per-FACE sub-palette: each face's nibble has its own per-cluster palette.
        uint64_t lodAoFaceBits = 0;    // sum of per-voxel per-face index bits
        uint64_t lodAoFaceHdrBits = 0; // sum of per-cluster palette bits (4 bits per entry × N)
        // RLE color packing: per-cluster, walk cells in chosen axis order, emit
        // alternating empty/solid runs with packed length+color. Replaces both
        // the bit-grid stream (positions recovered from walk) AND the color
        // stream (color included inline). Try 3 axis orders, pick smallest.
        uint64_t lodRleColorBytes = 0;
        uint64_t lodRleOrderHist[3] = {0, 0, 0};

        uint64_t lodCellAoBytes = 0; // raw bytes if cell-AO stored (4b per AO cell)
        uint64_t lodCellAoCells = 0; // count of cells needing AO

        // Bake each chunk.
        lods[L].reserve(chunkVox.size());
        size_t bakedChunks = 0;
        const size_t reportEvery = std::max((size_t)1, chunkVox.size() / 20);
        for (auto& kv : chunkVox)
        {
            if ((bakedChunks % reportEvery) == 0)
            {
                printf("[LOD %d] baking chunk %zu / %zu\n", L, bakedChunks, chunkVox.size());
                fflush(stdout);
            }
            ++bakedChunks;
            const ChunkKey& ck = kv.first;
            const std::vector<LodVox>& voxList = kv.second;
            if (voxList.empty())
                continue;

            BakedChunk bc;
            // Quantize palette (first kPaletteSize unique kept; rest mapped to nearest).
            std::unordered_map<uint32_t, uint16_t> palMap;
            palMap.reserve(256);
            for (const LodVox& v : voxList)
            {
                auto it = palMap.find(v.color);
                if (it != palMap.end())
                    continue;
                if (bc.palette.size() < lw::kPaletteSize)
                {
                    palMap[v.color] = (uint16_t)bc.palette.size();
                    bc.palette.push_back(v.color);
                }
                else
                {
                    int r1 = v.color & 0xFF, g1 = (v.color >> 8) & 0xFF, b1 = (v.color >> 16) & 0xFF;
                    int bestD = INT32_MAX;
                    uint16_t best = 0;
                    for (size_t i = 0; i < bc.palette.size(); ++i)
                    {
                        int r2 = bc.palette[i] & 0xFF, g2 = (bc.palette[i] >> 8) & 0xFF, b2 = (bc.palette[i] >> 16) & 0xFF;
                        int dR = r1 - r2, dG = g1 - g2, dB = b1 - b2;
                        int d = dR * dR + dG * dG + dB * dB;
                        if (d < bestD)
                        {
                            bestD = d;
                            best = (uint16_t)i;
                        }
                    }
                    palMap[v.color] = best;
                }
            }

            // Per-face AO: average the pre-baked hemisphere AO of every
            // source voxel on the LOD voxel's outward face surface. Returns
            // 4-bit nibble (0..15) for packing. Source AO is 0..255; we
            // average in 8-bit then shift to nibble. Empty sample cells
            // (no source voxel there) contribute the brightest value so
            // partially-empty LOD voxels don't get artificially darkened.
            auto pollSrcAo = [&](int sx, int sy, int sz, int f, int& sum, int& cnt)
            {
                auto it = srcIdx.find({sx, sy, sz});
                if (it == srcIdx.end())
                    return; // no source voxel here
                uint8_t a = src[it->second].aoFace[f];
                if (a == 0)
                    return; // interior face (not baked)
                sum += (int)a;
                ++cnt;
            };
            // Per-LOD contrast bias. Averaging across a LOD voxel's face
            // washes out dark crevices. Apply gentle power curve to 0..1 AO.
            // Capped at L=2 so LOD3/4 don't go pitch black on large voxels.
            //   L=0 exp 1.0 (no change), L=1 1.10, L=2/3/4 1.20.
            int lodForExp = (L > 2) ? 2 : L;
            const float lodExp = 1.0f + 0.1f * (float)lodForExp;
            // Per-face AO for the LOD voxel = average of aoFace[f] across ALL
            // step^3 source voxels that compose this LOD voxel. pollSrcAo
            // skips empty cells and hidden-face source voxels (aoFace=0), so
            // only lit face contributions average in.
            auto faceAo4 = [&](int vx, int vy, int vz, int f) -> uint8_t
            {
                if (!g_bakeAo)
                    return 15; // --noao: skip src lookup
                int sx0 = vx * step, sy0 = vy * step, sz0 = vz * step;
                int sum = 0, cnt = 0;
                for (int dx = 0; dx < step; ++dx)
                    for (int dy = 0; dy < step; ++dy)
                        for (int dz = 0; dz < step; ++dz)
                            pollSrcAo(sx0 + dx, sy0 + dy, sz0 + dz, f, sum, cnt);
                if (cnt == 0)
                    return 15;
                float a01 = (float)(sum / cnt) * (1.0f / 255.0f);
                if (lodExp != 1.0f)
                    a01 = powf(a01, lodExp);
                int q = (int)(a01 * 255.0f + 0.5f);
                if (q < 0)
                    q = 0;
                if (q > 255)
                    q = 255;
                return (uint8_t)(q >> 4);
            };

            // Chunk-local solid bit-grid (LOD-resolution). Used to compute
            // per-voxel visMask the SAME way lw_loader does (cross-chunk
            // neighbours treated as empty) so the variable-length AO stream
            // we emit matches what the loader expects to read back.
            const int CWv = lw::kChunkVoxX, CHv = lw::kChunkVoxY, CDv = lw::kChunkVoxZ;
            std::vector<uint8_t> chBits((size_t)((CWv * CHv * CDv + 7) / 8), 0);
            auto chSet = [&](int x, int y, int z)
            {
                size_t idx = (size_t)((y * CDv + z) * CWv + x);
                chBits[idx >> 3] |= (uint8_t)(1u << (idx & 7));
            };
            auto chGet = [&](int x, int y, int z) -> bool
            {
                if (x < 0 || y < 0 || z < 0 || x >= CWv || y >= CHv || z >= CDv)
                    return false;
                size_t idx = (size_t)((y * CDv + z) * CWv + x);
                return (chBits[idx >> 3] >> (idx & 7)) & 1u;
            };
            for (const LodVox& v : voxList)
            {
                chSet(v.x - ck.gx * lw::kChunkVoxX,
                      v.y - ck.gy * lw::kChunkVoxY,
                      v.z - ck.gz * lw::kChunkVoxZ);
            }
            auto visMaskFor = [&](int lx, int ly, int lz) -> uint8_t
            {
                uint8_t m = 0;
                if (!chGet(lx + 1, ly, lz))
                    m |= 0x01; // +X
                if (!chGet(lx - 1, ly, lz))
                    m |= 0x02; // -X
                if (!chGet(lx, ly + 1, lz))
                    m |= 0x04; // +Y
                if (!chGet(lx, ly - 1, lz))
                    m |= 0x08; // -Y
                if (!chGet(lx, ly, lz + 1))
                    m |= 0x10; // +Z
                if (!chGet(lx, ly, lz - 1))
                    m |= 0x20; // -Z
                return m;
            };

            // Bucket voxels into clusters (dense slots; empty slots untouched).
            std::vector<DiskPoint> bucket[lw::kClustersPerChunk];
            uint8_t cMn[lw::kClustersPerChunk][3];
            uint8_t cMx[lw::kClustersPerChunk][3];
            for (int s = 0; s < lw::kClustersPerChunk; ++s)
            {
                cMn[s][0] = cMn[s][1] = cMn[s][2] = 31;
                cMx[s][0] = cMx[s][1] = cMx[s][2] = 0;
            }
            for (const LodVox& v : voxList)
            {
                int32_t lx = v.x - ck.gx * lw::kChunkVoxX;
                int32_t ly = v.y - ck.gy * lw::kChunkVoxY;
                int32_t lz = v.z - ck.gz * lw::kChunkVoxZ;
                int cx = lx / lw::kClusterVoxX;
                int cy = ly / lw::kClusterVoxY;
                int cz = lz / lw::kClusterVoxZ;
                int slx = lx - cx * lw::kClusterVoxX;
                int sly = ly - cy * lw::kClusterVoxY;
                int slz = lz - cz * lw::kClusterVoxZ;
                int slot = lw::ClusterIdx(cx, cy, cz);
                DiskPoint p;
                p.posX = (uint8_t)lx;
                p.posY = (uint8_t)ly;
                p.posZ = (uint8_t)lz;
                p.palIdx = (uint8_t)palMap[v.color];
                // visMask + aoPacked unused by PointCS_Block; leave zero.
                p.visMask = 0;
                p.aoPacked[0] = 0;
                p.aoPacked[1] = 0;
                p.aoPacked[2] = 0;
                bucket[slot].push_back(p);
                // Measure: faces actually visible; boundary face = on cluster face.
                ++lodVoxelCount;
                uint8_t vm = p.visMask & 0x3F;
                int faces = 0;
                for (int fi = 0; fi < 6; ++fi)
                    if ((vm >> fi) & 1u)
                        ++faces;
                lodFaceCount += (uint64_t)faces;
                // Boundary faces of this voxel (faces on cluster boundary AND visible).
                int bf = 0;
                if (slx == 0 && (vm & 0x02))
                    ++bf; // -X
                if (slx == lw::kClusterVoxX - 1 && (vm & 0x01))
                    ++bf; // +X
                if (sly == 0 && (vm & 0x08))
                    ++bf; // -Y
                if (sly == lw::kClusterVoxY - 1 && (vm & 0x04))
                    ++bf; // +Y
                if (slz == 0 && (vm & 0x20))
                    ++bf; // -Z
                if (slz == lw::kClusterVoxZ - 1 && (vm & 0x10))
                    ++bf; // +Z
                lodBoundaryFaces += (uint64_t)bf;
                if (bf == 0)
                    ++lodInteriorVoxels;
                if (slx < cMn[slot][0])
                    cMn[slot][0] = (uint8_t)slx;
                if (sly < cMn[slot][1])
                    cMn[slot][1] = (uint8_t)sly;
                if (slz < cMn[slot][2])
                    cMn[slot][2] = (uint8_t)slz;
                if (slx > cMx[slot][0])
                    cMx[slot][0] = (uint8_t)slx;
                if (sly > cMx[slot][1])
                    cMx[slot][1] = (uint8_t)sly;
                if (slz > cMx[slot][2])
                    cMx[slot][2] = (uint8_t)slz;
            }

            // Cell-AO measurement (stats only) removed — not used for output.

            // ----- Sub-palette stats per cluster: color + AO -----
            for (int s = 0; s < lw::kClustersPerChunk; ++s)
            {
                if (bucket[s].empty())
                    continue;
                uint8_t seen[256] = {};
                int uniques = 0;
                for (const auto& p : bucket[s])
                {
                    if (!seen[p.palIdx])
                    {
                        seen[p.palIdx] = 1;
                        ++uniques;
                    }
                }
                int bpi;
                if (uniques <= 2)
                {
                    bpi = 1;
                    lodSubPalBins[0]++;
                }
                else if (uniques <= 4)
                {
                    bpi = 2;
                    lodSubPalBins[1]++;
                }
                else if (uniques <= 16)
                {
                    bpi = 4;
                    lodSubPalBins[2]++;
                }
                else
                {
                    bpi = 8;
                    lodSubPalBins[3]++;
                }
                uint64_t n = (uint64_t)bucket[s].size();
                lodColorBits[0] += n * 8;
                lodColorBits[1] += n * (uint64_t)bpi;
                lodSubPalHdrBytes += (uint64_t)uniques;

                // AO sub-palette over the 24-bit aoPacked word per voxel.
                std::unordered_map<uint32_t, uint8_t> aoSeen;
                aoSeen.reserve(32);
                for (const auto& p : bucket[s])
                {
                    uint32_t w = (uint32_t)p.aoPacked[0] | ((uint32_t)p.aoPacked[1] << 8) | ((uint32_t)p.aoPacked[2] << 16);
                    aoSeen.emplace(w, (uint8_t)aoSeen.size());
                }
                int aoUniques = (int)aoSeen.size();
                int aoBpi;
                if (aoUniques <= 2)
                {
                    aoBpi = 1;
                    lodAoSubBins[0]++;
                }
                else if (aoUniques <= 4)
                {
                    aoBpi = 2;
                    lodAoSubBins[1]++;
                }
                else if (aoUniques <= 16)
                {
                    aoBpi = 4;
                    lodAoSubBins[2]++;
                }
                else if (aoUniques <= 256)
                {
                    aoBpi = 8;
                    lodAoSubBins[3]++;
                }
                else
                {
                    aoBpi = 24;
                    lodAoSubBins[4]++;
                }
                lodAoBitsAdapt += n * (uint64_t)aoBpi;
                if (aoUniques <= 256)
                    lodAoHdrBytes += (uint64_t)aoUniques * 3;

                // ---- New RLE color packing measurement ----
                // For this cluster, try 3 axis-fastest orders and pick min.
                auto lenBytes = [](uint32_t c) -> size_t
                {
                    if (c <= 0x3F)
                        return 1;
                    size_t nb = 1;
                    c >>= 6;
                    while (c > 0x7F)
                    {
                        ++nb;
                        c >>= 7;
                    }
                    return nb + 1;
                };
                auto rleSize = [&](int order) -> size_t
                {
                    static thread_local uint8_t solBuf[lw::kClusterCellCount];
                    static thread_local uint8_t palBuf[lw::kClusterCellCount];
                    std::memset(solBuf, 0, sizeof(solBuf));
                    for (const auto& p : bucket[s])
                    {
                        uint32_t lx = p.posX % lw::kClusterVoxX;
                        uint32_t ly = p.posY % lw::kClusterVoxY;
                        uint32_t lz = p.posZ % lw::kClusterVoxZ;
                        uint32_t idx;
                        if (order == 0)
                            idx = (lz * lw::kClusterVoxY + ly) * lw::kClusterVoxX + lx; // X-fast
                        else if (order == 1)
                            idx = (lz * lw::kClusterVoxX + lx) * lw::kClusterVoxY + ly; // Y-fast
                        else
                            idx = (ly * lw::kClusterVoxX + lx) * lw::kClusterVoxZ + lz; // Z-fast
                        solBuf[idx] = 1;
                        palBuf[idx] = p.palIdx;
                    }
                    // Build runs.
                    size_t bytes = 0;
                    uint32_t pos = 0;
                    bool firstSolid = (lw::kClusterCellCount > 0 && solBuf[0]);
                    if (firstSolid)
                        bytes += 1; // empty len = 0 (1 byte)
                    while (pos < lw::kClusterCellCount)
                    {
                        // empty run
                        uint32_t es = pos;
                        while (pos < lw::kClusterCellCount && !solBuf[pos])
                            ++pos;
                        uint32_t emp = pos - es;
                        if (!(firstSolid && es == 0))
                            bytes += lenBytes(emp);
                        if (pos >= lw::kClusterCellCount)
                            break;
                        // solid run (same color)
                        uint8_t col = palBuf[pos];
                        uint32_t ss = pos;
                        while (pos < lw::kClusterCellCount && solBuf[pos] && palBuf[pos] == col)
                            ++pos;
                        uint32_t sol2 = pos - ss;
                        if (sol2 == 1)
                            bytes += 1; // just color
                        else
                            bytes += lenBytes(sol2) + 1; // length + color
                    }
                    return bytes;
                };
                size_t bestBytes = rleSize(0);
                int bestOrder = 0;
                for (int o = 1; o < 3; ++o)
                {
                    size_t b = rleSize(o);
                    if (b < bestBytes)
                    {
                        bestBytes = b;
                        bestOrder = o;
                    }
                }
                lodRleColorBytes += bestBytes;
                lodRleOrderHist[bestOrder]++;

                // Per-face sub-palette: each face direction independently.
                for (int fi = 0; fi < 6; ++fi)
                {
                    uint16_t mask = 0;
                    for (const auto& p : bucket[s])
                    {
                        uint8_t nib = (p.aoPacked[(fi * 4) >> 3] >> ((fi * 4) & 7)) & 0xFu;
                        mask |= (uint16_t)(1u << nib);
                    }
                    int u = __popcnt16(mask);
                    int bpi2;
                    if (u <= 1)
                        bpi2 = 0;
                    else if (u <= 2)
                        bpi2 = 1;
                    else if (u <= 4)
                        bpi2 = 2;
                    else if (u <= 8)
                        bpi2 = 3;
                    else
                        bpi2 = 4;
                    lodAoFaceBits += n * (uint64_t)bpi2;
                    lodAoFaceHdrBits += (uint64_t)u * 4; // 4 bits per palette entry
                }
            }

            // Concatenate cluster buckets into chunk's flat point array.
            // Update cluster headers (pointFirst, numPoints, bounds).
            uint32_t off = 0;
            uint8_t aMn[3] = {255, 255, 255};
            uint8_t aMx[3] = {0, 0, 0};
            for (int s = 0; s < lw::kClustersPerChunk; ++s)
            {
                bc.clusters[s].numPoints = (uint16_t)bucket[s].size();
                bc.clusters[s]._pad = 0;
                bc.clusters[s].pointFirst = off;
                if (bucket[s].empty())
                {
                    bc.clusters[s].bounds = 0;
                    continue;
                }
                bc.clusters[s].bounds = lw::PackClusterBounds(
                    cMn[s][0], cMn[s][1], cMn[s][2],
                    cMx[s][0], cMx[s][1], cMx[s][2]);
                for (const auto& p : bucket[s])
                {
                    bc.points.push_back(p);
                    if (p.posX < aMn[0])
                        aMn[0] = p.posX;
                    if (p.posY < aMn[1])
                        aMn[1] = p.posY;
                    if (p.posZ < aMn[2])
                        aMn[2] = p.posZ;
                    if (p.posX > aMx[0])
                        aMx[0] = p.posX;
                    if (p.posY > aMx[1])
                        aMx[1] = p.posY;
                    if (p.posZ > aMx[2])
                        aMx[2] = p.posZ;
                }
                off += (uint32_t)bucket[s].size();
            }

            // Header.
            bc.hdr.gridX = ck.gx;
            bc.hdr.gridY = ck.gy;
            bc.hdr.gridZ = ck.gz;
            // worldOrigin in LOD0 voxel units = grid * chunkVox * step.
            bc.hdr.worldOriginX = ck.gx * lw::kChunkVoxX * step;
            bc.hdr.worldOriginY = ck.gy * lw::kChunkVoxY * step;
            bc.hdr.worldOriginZ = ck.gz * lw::kChunkVoxZ * step;
            bc.hdr.lodLevel = (uint32_t)L;
            bc.hdr.paletteCount = (uint32_t)bc.palette.size();
            bc.hdr._unusedTotalPoints = (uint32_t)bc.points.size();
            for (int i = 0; i < 8; ++i)
                bc.hdr.childId[i] = lw::kNoChild;
            bc.hdr.aabbMin[0] = aMn[0];
            bc.hdr.aabbMin[1] = aMn[1];
            bc.hdr.aabbMin[2] = aMn[2];
            bc.hdr.aabbMax[0] = aMx[0];
            bc.hdr.aabbMax[1] = aMx[1];
            bc.hdr.aabbMax[2] = aMx[2];
            bc.hdr._pad = 0;
            lods[L].push_back(std::move(bc));
        }
        printf("[LOD %d] baked chunks=%zu points=%zu\n",
               L, lods[L].size(),
               [&]
               { size_t t=0; for (auto& c:lods[L]) t+=c.points.size(); return t; }());
        if (lodVoxelCount > 0)
        {
            const uint64_t aoFixed24 = (lodVoxelCount * 24 + 7) / 8; // bytes if 24 bits/vox
            const uint64_t aoVisBits = lodFaceCount * 4;             // 4 bits per visible face
            const uint64_t aoVisBytes = (aoVisBits + 7) / 8;
            double avgFaces = (double)lodFaceCount / (double)lodVoxelCount;
            double pctInterior = 100.0 * (double)lodInteriorVoxels / (double)lodVoxelCount;
            printf("[LOD %d]   AO: vox=%llu  faces=%llu  avg=%.2f f/vox  fixed24b=%.2f MB  visOnly=%.2f MB\n",
                   L, (unsigned long long)lodVoxelCount, (unsigned long long)lodFaceCount,
                   avgFaces, aoFixed24 / (1024.0 * 1024.0), aoVisBytes / (1024.0 * 1024.0));
            printf("[LOD %d]   boundary: voxels-with-bnd-face=%llu (%.1f%% interior-only)  bnd-faces=%llu\n",
                   L, (unsigned long long)(lodVoxelCount - lodInteriorVoxels),
                   pctInterior, (unsigned long long)lodBoundaryFaces);
            double mbFixed8 = (lodColorBits[0] / 8.0) / (1024.0 * 1024.0);
            double mbAdapt = ((lodColorBits[1] + 7) / 8 + lodSubPalHdrBytes) / (1024.0 * 1024.0);
            printf("[LOD %d]   color: fixed8b=%.2f MB  subPal(1/2/4/8 bit clusters: %llu/%llu/%llu/%llu)=%.2f MB (incl %llu B subpal hdrs)\n",
                   L, mbFixed8,
                   (unsigned long long)lodSubPalBins[0],
                   (unsigned long long)lodSubPalBins[1],
                   (unsigned long long)lodSubPalBins[2],
                   (unsigned long long)lodSubPalBins[3],
                   mbAdapt,
                   (unsigned long long)lodSubPalHdrBytes);
            printf("[LOD %d]   cellAO: %llu cells  4b/cell raw=%.2f MB\n",
                   L,
                   (unsigned long long)lodCellAoCells,
                   lodCellAoBytes / (1024.0 * 1024.0));
            // Sub-palette AO measurement (no format change yet).
            double aoMbFixed = (lodVoxelCount * 3.0) / (1024.0 * 1024.0);
            double aoMbAdapt = ((lodAoBitsAdapt + 7) / 8 + lodAoHdrBytes) / (1024.0 * 1024.0);
            double avgBitsAo = (double)lodAoBitsAdapt / (double)lodVoxelCount;
            double rleMb = lodRleColorBytes / (1024.0 * 1024.0);
            printf("[LOD %d]   RLE color packing: %.2f MB  (replaces bit-grid + colors)  orders(X/Y/Z fast): %llu/%llu/%llu\n",
                   L, rleMb,
                   (unsigned long long)lodRleOrderHist[0],
                   (unsigned long long)lodRleOrderHist[1],
                   (unsigned long long)lodRleOrderHist[2]);
            double aoFaceMb = ((lodAoFaceBits + lodAoFaceHdrBits + 7) / 8) / (1024.0 * 1024.0);
            double avgBitsFace = (double)lodAoFaceBits / (double)lodVoxelCount;
            printf("[LOD %d]   AO perFace subPal: %.2f MB  (%.2f bits/vox + %llu hdr bits)\n",
                   L, aoFaceMb, avgBitsFace, (unsigned long long)lodAoFaceHdrBits);
            printf("[LOD %d]   AO subPal(1/2/4/8/24-bit clusters: %llu/%llu/%llu/%llu/%llu)  fixed24b=%.2f MB  adapt=%.2f MB  (%.2f bits/vox + %llu B hdrs)\n",
                   L,
                   (unsigned long long)lodAoSubBins[0],
                   (unsigned long long)lodAoSubBins[1],
                   (unsigned long long)lodAoSubBins[2],
                   (unsigned long long)lodAoSubBins[3],
                   (unsigned long long)lodAoSubBins[4],
                   aoMbFixed, aoMbAdapt, avgBitsAo,
                   (unsigned long long)lodAoHdrBytes);
        }
    }
    src.clear();
    src.shrink_to_fit();

    // ---- fix up childId: link each L>0 chunk to its 8 LOD(L-1) children ----
    for (int L = 1; L < lw::kLodCount; ++L)
    {
        // Map (gridXYZ) -> idx for finer LOD.
        struct GK
        {
            int32_t x, y, z;
            bool operator==(const GK& o) const
            {
                return x == o.x && y == o.y && z == o.z;
            }
        };
        struct GKH
        {
            size_t operator()(const GK& k) const noexcept
            {
                uint64_t h = (uint64_t)(uint32_t)k.x * 0x9E3779B97F4A7C15ull;
                h ^= (uint64_t)(uint32_t)k.y * 0xBF58476D1CE4E5B9ull;
                h ^= (uint64_t)(uint32_t)k.z * 0x94D049BB133111EBull;
                return (size_t)(h ^ (h >> 32));
            }
        };
        std::unordered_map<GK, uint16_t, GKH> finer;
        finer.reserve(lods[L - 1].size());
        for (size_t i = 0; i < lods[L - 1].size(); ++i)
        {
            const auto& h = lods[L - 1][i].hdr;
            uint16_t idx = (uint16_t)i;
            if (i > 0xFFFEu)
            {
                fprintf(stderr, "warn: LOD %d has >65535 chunks; childId truncated\n", L - 1);
            }
            finer[{h.gridX, h.gridY, h.gridZ}] = idx;
        }
        for (auto& bc : lods[L])
        {
            for (int q = 0; q < 8; ++q)
            {
                int cx = (q >> 0) & 1, cy = (q >> 1) & 1, cz = (q >> 2) & 1;
                GK ck{bc.hdr.gridX * 2 + cx, bc.hdr.gridY * 2 + cy, bc.hdr.gridZ * 2 + cz};
                auto it = finer.find(ck);
                bc.hdr.childId[q] = (it == finer.end()) ? lw::kNoChild : it->second;
            }
        }
    }

    // ---- write .lw file ----
    fs::create_directories(fs::path(outPath).parent_path());
    FILE* o = fopen(outPath, "wb");
    if (!o)
    {
        fprintf(stderr, "cannot open %s for write\n", outPath);
        return 1;
    }

    // FileHeader.
    lw::FileHeader fh{};
    fh.magic = lw::kFileMagic;
    fh.version = lw::kFileVersion;
    fh.chunkVoxX = lw::kChunkVoxX;
    fh.chunkVoxY = lw::kChunkVoxY;
    fh.chunkVoxZ = lw::kChunkVoxZ;
    fh.clusterVoxX = lw::kClusterVoxX;
    fh.clusterVoxY = lw::kClusterVoxY;
    fh.clusterVoxZ = lw::kClusterVoxZ;
    fh.lodCount = lw::kLodCount;
    for (int i = 0; i < 3; ++i)
    {
        fh.worldAabbMin[i] = worldMn[i];
        fh.worldAabbMax[i] = worldMx[i];
    }
    fwrite(&fh, sizeof(fh), 1, o);

    // LODHeader[kLodCount] — chunkTableOffset patched in pass 2.
    long lodHdrPos = ftell(o);
    std::vector<lw::LODHeader> lodHdrs(lw::kLodCount);
    for (int L = 0; L < lw::kLodCount; ++L)
    {
        lodHdrs[L].lodLevel = (uint32_t)L;
        lodHdrs[L].chunkCount = (uint32_t)lods[L].size();
        lodHdrs[L].chunkTableOffset = 0;
    }
    fwrite(lodHdrs.data(), sizeof(lw::LODHeader), lw::kLodCount, o);

    // Per LOD: reserve space for ChunkEntry table, then write each blob, then
    // backfill ChunkEntry with blob offsets.
    for (int L = 0; L < lw::kLodCount; ++L)
    {
        lodHdrs[L].chunkTableOffset = (uint64_t)ftell(o);
        std::vector<lw::ChunkEntry> entries(lods[L].size());
        long entryStart = ftell(o);
        // reserve
        fwrite(entries.data(), sizeof(lw::ChunkEntry), entries.size(), o);

        uint64_t lodYBytes = 0, lodMBytes = 0;
        uint64_t lodChoseY = 0, lodChoseM = 0;
        for (size_t i = 0; i < lods[L].size(); ++i)
        {
            const BakedChunk& bc = lods[L][i];

            // Build chunk-wide bit-grid (used by cell-AO bake).
            ChunkBits chunkBits;
            if (g_storeCellAo)
            {
                for (const auto& p : bc.points)
                    chunkBits.set(p.posX, p.posY, p.posZ);
            }

            // Build per-cluster encoded blobs (bit-grid + colors + cellAO).
            std::vector<ClusterEnc> ces(lw::kClustersPerChunk);
            uint8_t clusterMask[16] = {};
            uint64_t yT = 0, mT = 0;
            // Cluster bit-grid / colour / AO encoding dropped — V2 octet stream
            // (below) is the only consumer of per-voxel data now.
            (void)yT;
            (void)mT;
            (void)lodChoseY;
            (void)lodChoseM;

            // Serialize compressed chunk blob:
            //   DiskChunkHeader
            //   palette
            //   clusterMask[16]
            //   per non-empty cluster: orderMode + LEB128 numPoints
            //                          + LEB128 bgSize + bgBytes + colorsBytes
            std::vector<uint8_t> blob;
            blob.reserve(sizeof(bc.hdr) + bc.palette.size() * 4 + 16 + 1024);
            auto push = [&](const void* p, size_t n)
            {
                blob.insert(blob.end(), (const uint8_t*)p, (const uint8_t*)p + n);
            };
            push(&bc.hdr, sizeof(bc.hdr));
            if (!bc.palette.empty())
                push(bc.palette.data(), bc.palette.size() * sizeof(uint32_t));
            // Cluster bit-grid + colors + AO/visMask/cellAO streams dropped
            // — PointCS_Block (V2 octets below) is the only consumer now.

            // ---- V2 compact octet stream for PointCS_Block ----
            //   per cluster: clusterID + octet stream (see lodworld.h for layout).
            if (g_storeBlocks)
            {
                // Group points by (clusterIdx, octetIdx) → voxelMask + palIdx[8].
                struct OctetEnc
                {
                    uint8_t mask = 0;
                    uint8_t pal[8] = {};
                };
                // Per-cluster: map octetIdx -> OctetEnc.
                std::vector<std::map<uint32_t, OctetEnc>> perCluster(lw::kClustersPerChunk);
                for (const auto& p : bc.points)
                {
                    // Cluster coords in chunk.
                    uint32_t cx = (uint32_t)p.posX / lw::kClusterVoxX;
                    uint32_t cy = (uint32_t)p.posY / lw::kClusterVoxY;
                    uint32_t cz = (uint32_t)p.posZ / lw::kClusterVoxZ;
                    uint32_t clusterIdx = (cz * lw::kClustersY + cy) * lw::kClustersX + cx;
                    // Voxel coords within cluster.
                    uint32_t vxInCluster = (uint32_t)p.posX - cx * lw::kClusterVoxX; // 0..31
                    uint32_t vyInCluster = (uint32_t)p.posY - cy * lw::kClusterVoxY;
                    uint32_t vzInCluster = (uint32_t)p.posZ - cz * lw::kClusterVoxZ;
                    // Octet coords in cluster (2 voxels per octet axis → 16 octets per axis).
                    uint32_t ox = vxInCluster >> 1;
                    uint32_t oy = vyInCluster >> 1;
                    uint32_t oz = vzInCluster >> 1;
                    uint32_t octetIdx = (oy * 16u + oz) * 16u + ox; // Y-major
                    // Voxel-in-octet bit (i = lx | (ly<<1) | (lz<<2)).
                    uint32_t lx = vxInCluster & 1u;
                    uint32_t ly = vyInCluster & 1u;
                    uint32_t lz = vzInCluster & 1u;
                    uint32_t vi = lx | (ly << 1) | (lz << 2);
                    OctetEnc& oe = perCluster[clusterIdx][octetIdx];
                    oe.mask |= (uint8_t)(1u << vi);
                    oe.pal[vi] = p.palIdx;
                }
                // Collect non-empty clusters and emit.
                std::vector<uint32_t> nonEmpty;
                for (int ci = 0; ci < lw::kClustersPerChunk; ++ci)
                {
                    if (!perCluster[ci].empty())
                        nonEmpty.push_back((uint32_t)ci);
                }
                // Classify an octet:
                //   uniform >= 0 → all set bits have the same palIdx (= uniform)
                //   uniform == -1 → varied (no single colour)
                auto classify = [&](const OctetEnc& oe) -> int
                {
                    int u = -1;
                    bool first = true;
                    for (int vi = 0; vi < 8; ++vi)
                    {
                        if (!(oe.mask & (1u << vi)))
                            continue;
                        if (first)
                        {
                            u = oe.pal[vi];
                            first = false;
                        }
                        else if (oe.pal[vi] != u)
                        {
                            return -1;
                        }
                    }
                    return u;
                };
                for (size_t ki = 0; ki < nonEmpty.size(); ++ki)
                {
                    uint32_t ci = nonEmpty[ki];
                    bool isLastCluster = (ki + 1 == nonEmpty.size());
                    uint8_t clusterID = (uint8_t)(ci & 0x7Fu);
                    if (isLastCluster)
                        clusterID |= 0x80u;
                    push(&clusterID, 1);

                    auto& octetsMap = perCluster[ci];
                    // Flatten std::map (sorted) into a vector for indexed access.
                    std::vector<std::pair<uint32_t, OctetEnc>> octs;
                    octs.reserve(octetsMap.size());
                    for (auto& kv : octetsMap)
                        octs.push_back({kv.first, kv.second});

                    // Pre-compute run-length / explicit flags per octet.
                    std::vector<uint8_t> runBits(octs.size(), 0);
                    std::vector<uint8_t> explicitFlag(octs.size(), 0);
                    size_t ii = 0;
                    while (ii < octs.size())
                    {
                        if (ii + 1 == octs.size())
                        {
                            explicitFlag[ii] = 1;
                            runBits[ii] = 15; // terminator
                            break;
                        }
                        uint32_t run = 0;
                        size_t jj = ii;
                        while (run < 14u)
                        {
                            size_t pk = jj + 1;
                            if (pk >= octs.size())
                                break;
                            if (octs[pk].first != octs[jj].first + 1u)
                                break;
                            if (pk + 1 == octs.size())
                                break; // keep final for terminator
                            ++run;
                            jj = pk;
                        }
                        explicitFlag[ii] = 1;
                        runBits[ii] = (uint8_t)run;
                        ii = jj + 1;
                    }

                    // Walk octets, interleaving modeByte every 4 octets.
                    int prevColor = -1; // resets per cluster
                    for (size_t oi = 0; oi < octs.size(); ++oi)
                    {
                        if ((oi & 3u) == 0u)
                        {
                            // Pre-compute modeByte for next up-to-4 octets
                            // (simulate prevColor updates without emitting payload).
                            int simPrev = prevColor;
                            uint8_t modeByte = 0;
                            size_t end4 = std::min(oi + 4u, octs.size());
                            for (size_t k = oi; k < end4; ++k)
                            {
                                int u = classify(octs[k].second);
                                uint8_t mode;
                                if (u >= 0)
                                {
                                    if (u == simPrev)
                                        mode = lw::kOctetModeUniformReuse;
                                    else
                                    {
                                        mode = lw::kOctetModeUniformNew;
                                        simPrev = u;
                                    }
                                }
                                else
                                {
                                    mode = lw::kOctetModeVaried;
                                }
                                modeByte |= (uint8_t)(mode << ((k - oi) * 2u));
                            }
                            push(&modeByte, 1);
                        }
                        // Octet header (only on explicit / run-start).
                        const auto& oe = octs[oi].second;
                        if (explicitFlag[oi])
                        {
                            uint16_t octetID = (uint16_t)((octs[oi].first & 0x0FFFu)
                                                          | ((uint32_t)runBits[oi] << 12));
                            push(&octetID, 2);
                        }
                        // Mask is always emitted.
                        push(&oe.mask, 1);
                        // Mode-specific payload.
                        int u = classify(oe);
                        if (u >= 0)
                        {
                            if (u == prevColor)
                            {
                                // mode 0: nothing
                            }
                            else
                            {
                                uint8_t col = (uint8_t)u;
                                push(&col, 1);
                                prevColor = u;
                            }
                        }
                        else
                        {
                            // mode 2: per-voxel palette bytes in bit order.
                            for (int vi = 0; vi < 8; ++vi)
                            {
                                if (oe.mask & (1u << vi))
                                    push(&oe.pal[vi], 1);
                            }
                        }
                    }
                }
            }

            // LZ4 on top (header + cluster mask + RLE+colors residual redundancy).
            std::vector<uint8_t> cblob;
            uint32_t writeBytes;
            uint32_t flags = 0;
            if (g_storeBlocks)
                flags |= lw::kFlagBlocks;
            const int rawSize = (int)blob.size();
            const int cap = LZ4_compressBound(rawSize);
            cblob.resize((size_t)cap);
            int cSize = LZ4_compress_default((const char*)blob.data(),
                                             (char*)cblob.data(),
                                             rawSize, cap);
            if (cSize > 0 && cSize < rawSize)
            {
                writeBytes = (uint32_t)cSize;
                flags |= lw::kFlagLz4;
            }
            else
            {
                writeBytes = (uint32_t)rawSize;
            }
            uint64_t blobOff = (uint64_t)ftell(o);
            if (flags & lw::kFlagLz4)
                fwrite(cblob.data(), 1, writeBytes, o);
            else
                fwrite(blob.data(), 1, writeBytes, o);

            entries[i].gridX = bc.hdr.gridX;
            entries[i].gridY = bc.hdr.gridY;
            entries[i].gridZ = bc.hdr.gridZ;
            entries[i].blobOffset = blobOff;
            entries[i].blobBytes = writeBytes;
            entries[i].blobBytesRaw = (uint32_t)rawSize;
            entries[i].flags = flags;
        }

        // Patch ChunkEntry table.
        long end = ftell(o);
        fseek(o, entryStart, SEEK_SET);
        fwrite(entries.data(), sizeof(lw::ChunkEntry), entries.size(), o);
        fseek(o, end, SEEK_SET);

        if (!lods[L].empty())
        {
            printf("[LOD %d] order pick: Y=%llu M=%llu  total Y-only=%llu B, M-only=%llu B\n",
                   L,
                   (unsigned long long)lodChoseY, (unsigned long long)lodChoseM,
                   (unsigned long long)lodYBytes, (unsigned long long)lodMBytes);
        }

        // ----- LOD0 detailed breakdown: chunk overhead vs cluster overhead vs payload -----
        if (L == 0)
        {
            uint64_t chunkHdrBytes = 0;     // DiskChunkHeader * chunks
            uint64_t chunkPalBytes = 0;     // palette * chunks
            uint64_t chunkMaskBytes = 0;    // 16 B cluster-mask * chunks
            uint64_t clusterFixedBytes = 0; // 1B orderMode + LEB128(numPoints+bgSize) per non-empty cluster
            uint64_t payloadBitGrid = 0;
            uint64_t payloadColors = 0;
            uint64_t payloadAo = 0;
            uint64_t payloadVisMask = 0;
            uint64_t fileRawBytes = 0;  // sum of pre-LZ4 blob sizes
            uint64_t fileCompBytes = 0; // sum of on-disk blob sizes
            uint64_t gpuPointBytes = 0; // points in pool * 8

            for (size_t i = 0; i < lods[L].size(); ++i)
            {
                const BakedChunk& bc = lods[L][i];
                chunkHdrBytes += sizeof(lw::DiskChunkHeader);
                chunkPalBytes += bc.palette.size() * sizeof(uint32_t);
                chunkMaskBytes += 16;
                gpuPointBytes += bc.points.size() * sizeof(DiskPoint);

                fileRawBytes += entries[i].blobBytesRaw;
                fileCompBytes += entries[i].blobBytes;

                // Re-encode to count cluster-level breakdown (cheap; same logic).
                uint64_t dummyY = 0, dummyM = 0;
                for (int s = 0; s < lw::kClustersPerChunk; ++s)
                {
                    const lw::DiskCluster& cl = bc.clusters[s];
                    if (cl.numPoints == 0)
                        continue;
                    ClusterEnc ce = EncodeClusterTryBoth(&bc.points[cl.pointFirst], cl.numPoints, dummyY, dummyM);
                    // orderMode (1) + LEB128(numPoints) + LEB128(bgSize)
                    auto leb = [](uint32_t v) -> uint32_t
                    {
                        uint32_t n = 1;
                        while (v >= 0x80u)
                        {
                            ++n;
                            v >>= 7;
                        }
                        return n;
                    };
                    clusterFixedBytes += 1 + leb((uint32_t)cl.numPoints) + leb((uint32_t)ce.bitGrid.size());
                    payloadBitGrid += ce.bitGrid.size();
                    payloadColors += ce.colors.size();
                    payloadAo += ce.ao.size();
                    payloadVisMask += ce.visMask.size();
                }
            }
            const double MB = 1024.0 * 1024.0;
            printf("\n========== LOD0 memory breakdown ==========\n");
            printf("  Chunks: %zu\n", lods[L].size());
            printf("  ---- Per-chunk fixed ----\n");
            printf("    DiskChunkHeader (60 B/ea)         : %8.2f KB\n", chunkHdrBytes / 1024.0);
            printf("    Palette (paletteCount*4)          : %8.2f KB\n", chunkPalBytes / 1024.0);
            printf("    Cluster mask (16 B/ea)            : %8.2f KB\n", chunkMaskBytes / 1024.0);
            uint64_t chunkFixedTot = chunkHdrBytes + chunkPalBytes + chunkMaskBytes;
            printf("    SUBTOTAL chunk fixed              : %8.2f KB\n", chunkFixedTot / 1024.0);
            printf("  ---- Per-cluster fixed ----\n");
            printf("    orderMode + LEB128 counts         : %8.2f KB\n", clusterFixedBytes / 1024.0);
            printf("  ---- Per-cluster payload ----\n");
            printf("    Bit-grid (RLE+LEB128)             : %8.2f MB\n", payloadBitGrid / MB);
            printf("    Colors (8 bit/voxel)              : %8.2f MB\n", payloadColors / MB);
            if (payloadAo > 0)
                printf("    AO (24 bits/voxel)                : %8.2f MB\n", payloadAo / MB);
            if (payloadVisMask > 0)
                printf("    visMask (8 bits/voxel)            : %8.2f MB\n", payloadVisMask / MB);
            uint64_t payloadTot = payloadBitGrid + payloadColors + payloadAo + payloadVisMask;
            printf("    SUBTOTAL payload                  : %8.2f MB\n", payloadTot / MB);
            uint64_t totalRaw = chunkFixedTot + clusterFixedBytes + payloadTot;
            printf("  ---- Disk totals ----\n");
            printf("    Raw blob sum (pre-LZ4)            : %8.2f MB\n", totalRaw / MB);
            printf("    File raw sum (verify)             : %8.2f MB\n", fileRawBytes / MB);
            printf("    File on-disk (LZ4)                : %8.2f MB\n", fileCompBytes / MB);
            printf("    LZ4 ratio                         : %.2f%%\n", 100.0 * fileCompBytes / (double)fileRawBytes);
            printf("  ---- GPU side ----\n");
            printf("    Point pool (8 B/voxel)            : %8.2f MB  (%llu pts)\n",
                   gpuPointBytes / MB, (unsigned long long)(gpuPointBytes / 8));
            printf("===========================================\n\n");
        }
    }

    // Patch LODHeaders with offsets.
    long fileEnd = ftell(o);
    fseek(o, lodHdrPos, SEEK_SET);
    fwrite(lodHdrs.data(), sizeof(lw::LODHeader), lw::kLodCount, o);
    fseek(o, fileEnd, SEEK_SET);

    long bytes = ftell(o);
    fclose(o);
    printf("[out] %s: %ld bytes (%.2f MB)\n",
           outPath, bytes, bytes / (1024.0 * 1024.0));
    if (g_aoCount > 0)
    {
        printf("[AO bake] %llu values: ", (unsigned long long)g_aoCount);
        for (int i = 0; i < 16; ++i)
        {
            printf("%d=%.1f%% ", i, 100.0 * g_aoHisto[i] / (double)g_aoCount);
        }
        printf("\n");
    }
    return 0;
}
