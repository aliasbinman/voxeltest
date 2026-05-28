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
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

// ---------- source .vox raw types ----------
#pragma pack(push, 1)
struct VxlDiskVoxel { uint8_t x, y, z, visMask; uint16_t paletteIdx; uint8_t aoPacked[3]; };
#pragma pack(pop)
static_assert(sizeof(VxlDiskVoxel) == 9, "");

struct VxlChunkMeta {
    uint16_t cx, cy, cz, _pad;
    uint32_t voxelCount, voxelOffset;
};
static_assert(sizeof(VxlChunkMeta) == 16, "");

static inline uint8_t SrcAo(const VxlDiskVoxel& v, int fi)
{
    uint8_t n = (v.aoPacked[(fi * 4) >> 3] >> ((fi * 4) & 7)) & 0xFu;
    return (uint8_t)((n << 4) | n);   // 0..255 in 16 steps
}

// ---------- helpers ----------
static inline int32_t FloorDiv(int32_t a, int32_t b)
{
    int32_t q = a / b;
    if ((a % b != 0) && ((a ^ b) < 0)) --q;
    return q;
}
static inline int32_t FloorMod(int32_t a, int32_t b)
{
    int32_t m = a % b;
    if ((m != 0) && ((m ^ b) < 0)) m += b;
    return m;
}

struct CellKey {
    int32_t x, y, z;
    bool operator==(const CellKey& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct CellHash {
    size_t operator()(const CellKey& k) const noexcept {
        uint64_t h = (uint64_t)(uint32_t)k.x * 0x9E3779B97F4A7C15ull;
        h ^= (uint64_t)(uint32_t)k.y * 0xBF58476D1CE4E5B9ull;
        h ^= (uint64_t)(uint32_t)k.z * 0x94D049BB133111EBull;
        return (size_t)(h ^ (h >> 32));
    }
};

struct LodVox {
    int32_t x, y, z;         // in LOD-cell coords (one cell per LOD-voxel)
    uint32_t color;          // 0x00BBGGRR averaged
    uint8_t  visMask;        // OR of contributing src voxels
    uint8_t  aoFace[6];      // max per face from contributors
};

// Aggregator during cell binning.
struct CellAcc {
    uint32_t r = 0, g = 0, b = 0;
    uint32_t count = 0;
    uint8_t  mask = 0;
    uint8_t  aoMax[6] = { 0, 0, 0, 0, 0, 0 };
};

struct BakedChunk {
    lw::DiskChunkHeader  hdr;
    std::vector<uint32_t> palette;
    lw::DiskCluster      clusters[lw::kClustersPerChunk];
    std::vector<lw::DiskPoint> points;
};

// Per-cluster compression encoded data. Chosen ordering = smaller of
// {Y-major, Morton}.
struct ClusterEnc {
    uint8_t orderMode = lw::kOrderYMajor;
    std::vector<uint8_t> bitGrid;
    std::vector<uint8_t> colors;
};

static ClusterEnc EncodeClusterTryBoth(const lw::DiskPoint* pts, uint32_t numPts,
                                       uint64_t& outYBytes, uint64_t& outMBytes)
{
    auto tryOrder = [&](lw::LwOrderMode mode) -> ClusterEnc {
        uint8_t grid[lw::kClusterCellCount];
        std::memset(grid, 0, sizeof(grid));
        uint8_t palAt[lw::kClusterCellCount];
        for (uint32_t i = 0; i < numPts; ++i) {
            const lw::DiskPoint& p = pts[i];
            uint32_t lx = (uint32_t)(p.posX % lw::kClusterVoxX);
            uint32_t ly = (uint32_t)(p.posY % lw::kClusterVoxY);
            uint32_t lz = (uint32_t)(p.posZ % lw::kClusterVoxZ);
            uint32_t idx = lw::LwCellIndex(mode, lx, ly, lz);
            grid[idx] = 1;
            palAt[idx] = p.palIdx;
        }
        ClusterEnc e;
        e.orderMode = (uint8_t)mode;
        lw::RleEncodeBitGrid(grid, e.bitGrid);
        e.colors.reserve(numPts);
        for (uint32_t i = 0; i < lw::kClusterCellCount; ++i) {
            if (grid[i]) e.colors.push_back(palAt[i]);
        }
        return e;
    };
    ClusterEnc y = tryOrder(lw::kOrderYMajor);
    ClusterEnc m = tryOrder(lw::kOrderMorton);
    outYBytes += y.bitGrid.size() + y.colors.size();
    outMBytes += m.bitGrid.size() + m.colors.size();
    if (y.bitGrid.size() + y.colors.size() <= m.bitGrid.size() + m.colors.size())
        return y;
    return m;
}

// ---------- main ----------
int main(int argc, char** argv)
{
    const char* inPath  = (argc > 1) ? argv[1] : "assets/kingslanding.vox";
    const char* outPath = (argc > 2) ? argv[2] : "assets/kingslanding.lw";

    // ---- read source .vox ----
    FILE* f = fopen(inPath, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", inPath); return 1; }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "VXL3", 4) != 0) {
        fclose(f); fprintf(stderr, "bad magic\n"); return 1;
    }
    uint32_t version = 0;
    fread(&version, sizeof(uint32_t), 1, f);
    if (version != kAssetVersion) {
        fclose(f); fprintf(stderr, "version mismatch: %u vs %u\n", version, kAssetVersion); return 1;
    }
    uint32_t chunkDim = 0, srcChunkCount = 0, totalVoxels = 0;
    int32_t  srcOrigin[3] = { 0, 0, 0 };
    float    srcSun[3]    = { 0, 0, 0 };
    fread(&chunkDim,      sizeof(uint32_t), 1, f);
    fread(&srcChunkCount, sizeof(uint32_t), 1, f);
    fread(&totalVoxels,   sizeof(uint32_t), 1, f);
    fread(srcOrigin,      sizeof(int32_t),  3, f);
    fread(srcSun,         sizeof(float),    3, f);
    uint32_t srcPalCount = 0;
    fread(&srcPalCount, sizeof(uint32_t), 1, f);
    std::vector<uint32_t> srcPalette(srcPalCount);
    if (srcPalCount) fread(srcPalette.data(), sizeof(uint32_t), srcPalCount, f);
    std::vector<VxlChunkMeta> srcMetas(srcChunkCount);
    if (srcChunkCount) fread(srcMetas.data(), sizeof(VxlChunkMeta), srcChunkCount, f);
    std::vector<VxlDiskVoxel> srcVox(totalVoxels);
    if (totalVoxels) fread(srcVox.data(), sizeof(VxlDiskVoxel), totalVoxels, f);
    fclose(f);
    printf("[in] %s chunkDim=%u srcChunks=%u voxels=%u palette=%u\n",
           inPath, chunkDim, srcChunkCount, totalVoxels, srcPalCount);

    const int32_t D = (int32_t)chunkDim;

    // ---- flatten src voxels into world-space LOD0 cells ----
    struct SrcVox { int32_t x, y, z; uint32_t color; uint8_t mask; uint8_t aoFace[6]; };
    std::vector<SrcVox> src;
    src.reserve(totalVoxels);
    int32_t worldMn[3] = {  INT32_MAX,  INT32_MAX,  INT32_MAX };
    int32_t worldMx[3] = { -INT32_MAX, -INT32_MAX, -INT32_MAX };
    for (uint32_t ci = 0; ci < srcChunkCount; ++ci) {
        const VxlChunkMeta& m = srcMetas[ci];
        int32_t bx = (int32_t)m.cx * D, by = (int32_t)m.cy * D, bz = (int32_t)m.cz * D;
        const VxlDiskVoxel* vs = srcVox.data() + m.voxelOffset;
        for (uint32_t i = 0; i < m.voxelCount; ++i) {
            const VxlDiskVoxel& dv = vs[i];
            if (dv.visMask == 0) continue;
            SrcVox s;
            s.x = bx + (int32_t)dv.x;
            s.y = by + (int32_t)dv.y;
            s.z = bz + (int32_t)dv.z;
            uint32_t col = (dv.paletteIdx < srcPalette.size()) ? srcPalette[dv.paletteIdx] : 0xFF000000u;
            s.color = col & 0x00FFFFFFu;
            s.mask  = dv.visMask & 0x3F;
            for (int fi = 0; fi < 6; ++fi) s.aoFace[fi] = SrcAo(dv, fi);
            src.push_back(s);
            if (s.x < worldMn[0]) worldMn[0] = s.x; if (s.x > worldMx[0]) worldMx[0] = s.x;
            if (s.y < worldMn[1]) worldMn[1] = s.y; if (s.y > worldMx[1]) worldMx[1] = s.y;
            if (s.z < worldMn[2]) worldMn[2] = s.z; if (s.z > worldMx[2]) worldMx[2] = s.z;
        }
    }
    srcVox.clear(); srcVox.shrink_to_fit();
    srcMetas.clear(); srcMetas.shrink_to_fit();
    printf("[in] src voxels: %zu  AABB: x[%d..%d] y[%d..%d] z[%d..%d]\n",
           src.size(), worldMn[0], worldMx[0], worldMn[1], worldMx[1], worldMn[2], worldMx[2]);

    // ---- build all LODs ----
    std::vector<std::vector<BakedChunk>> lods(lw::kLodCount);

    for (int L = 0; L < lw::kLodCount; ++L) {
        const int32_t step = 1 << L;

        // Aggregate src voxels into LOD-step cells.
        std::unordered_map<CellKey, CellAcc, CellHash> cells;
        cells.reserve(src.size() / (size_t)(step * step * step) + 1024);
        for (const SrcVox& v : src) {
            CellKey k{ FloorDiv(v.x, step), FloorDiv(v.y, step), FloorDiv(v.z, step) };
            CellAcc& a = cells[k];
            a.r += (v.color >>  0) & 0xFF;
            a.g += (v.color >>  8) & 0xFF;
            a.b += (v.color >> 16) & 0xFF;
            a.mask |= v.mask;
            for (int fi = 0; fi < 6; ++fi) {
                if (v.aoFace[fi] > a.aoMax[fi]) a.aoMax[fi] = v.aoFace[fi];
            }
            ++a.count;
        }
        // Flatten + per-chunk bucket.
        struct ChunkKey {
            int32_t gx, gy, gz;
            bool operator==(const ChunkKey& o) const { return gx == o.gx && gy == o.gy && gz == o.gz; }
        };
        struct ChunkHash {
            size_t operator()(const ChunkKey& k) const noexcept {
                uint64_t h = (uint64_t)(uint32_t)k.gx * 0x9E3779B97F4A7C15ull;
                h ^= (uint64_t)(uint32_t)k.gy * 0xBF58476D1CE4E5B9ull;
                h ^= (uint64_t)(uint32_t)k.gz * 0x94D049BB133111EBull;
                return (size_t)(h ^ (h >> 32));
            }
        };
        std::unordered_map<ChunkKey, std::vector<LodVox>, ChunkHash> chunkVox;
        for (auto& kv : cells) {
            const CellKey& k = kv.first;
            const CellAcc& a = kv.second;
            LodVox v;
            v.x = k.x; v.y = k.y; v.z = k.z;
            uint32_t r = a.r / a.count, g = a.g / a.count, b = a.b / a.count;
            v.color = r | (g << 8) | (b << 16);
            v.visMask = a.mask;
            for (int fi = 0; fi < 6; ++fi) v.aoFace[fi] = a.aoMax[fi];
            ChunkKey ck{ FloorDiv(v.x, lw::kChunkVoxX),
                         FloorDiv(v.y, lw::kChunkVoxY),
                         FloorDiv(v.z, lw::kChunkVoxZ) };
            chunkVox[ck].push_back(v);
        }
        cells.clear();
        printf("[LOD %d] chunks=%zu (step=%d)\n", L, chunkVox.size(), step);

        // Bake each chunk.
        lods[L].reserve(chunkVox.size());
        for (auto& kv : chunkVox) {
            const ChunkKey& ck = kv.first;
            const std::vector<LodVox>& voxList = kv.second;
            if (voxList.empty()) continue;

            BakedChunk bc;
            // Quantize palette (first kPaletteSize unique kept; rest mapped to nearest).
            std::unordered_map<uint32_t, uint16_t> palMap;
            palMap.reserve(256);
            for (const LodVox& v : voxList) {
                auto it = palMap.find(v.color);
                if (it != palMap.end()) continue;
                if (bc.palette.size() < lw::kPaletteSize) {
                    palMap[v.color] = (uint16_t)bc.palette.size();
                    bc.palette.push_back(v.color);
                } else {
                    int r1 = v.color & 0xFF, g1 = (v.color >> 8) & 0xFF, b1 = (v.color >> 16) & 0xFF;
                    int bestD = INT32_MAX; uint16_t best = 0;
                    for (size_t i = 0; i < bc.palette.size(); ++i) {
                        int r2 = bc.palette[i] & 0xFF, g2 = (bc.palette[i] >> 8) & 0xFF, b2 = (bc.palette[i] >> 16) & 0xFF;
                        int dR = r1 - r2, dG = g1 - g2, dB = b1 - b2;
                        int d = dR * dR + dG * dG + dB * dB;
                        if (d < bestD) { bestD = d; best = (uint16_t)i; }
                    }
                    palMap[v.color] = best;
                }
            }

            // Bucket voxels into clusters (dense slots; empty slots untouched).
            std::vector<lw::DiskPoint> bucket[lw::kClustersPerChunk];
            uint8_t cMn[lw::kClustersPerChunk][3];
            uint8_t cMx[lw::kClustersPerChunk][3];
            for (int s = 0; s < lw::kClustersPerChunk; ++s) {
                cMn[s][0] = cMn[s][1] = cMn[s][2] = 31;
                cMx[s][0] = cMx[s][1] = cMx[s][2] = 0;
            }
            for (const LodVox& v : voxList) {
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
                lw::DiskPoint p;
                p.posX = (uint8_t)lx;
                p.posY = (uint8_t)ly;
                p.posZ = (uint8_t)lz;
                p.palIdx = (uint8_t)palMap[v.color];
                p.visMask = v.visMask & 0x3F;
                uint32_t aoPacked = 0;
                for (int fi = 0; fi < 6; ++fi) {
                    aoPacked |= (uint32_t)((v.aoFace[fi] >> 4) & 0xF) << (fi * 4);
                }
                p.aoPacked[0] = (uint8_t)(aoPacked & 0xFF);
                p.aoPacked[1] = (uint8_t)((aoPacked >> 8) & 0xFF);
                p.aoPacked[2] = (uint8_t)((aoPacked >> 16) & 0xFF);
                bucket[slot].push_back(p);
                if (slx < cMn[slot][0]) cMn[slot][0] = (uint8_t)slx;
                if (sly < cMn[slot][1]) cMn[slot][1] = (uint8_t)sly;
                if (slz < cMn[slot][2]) cMn[slot][2] = (uint8_t)slz;
                if (slx > cMx[slot][0]) cMx[slot][0] = (uint8_t)slx;
                if (sly > cMx[slot][1]) cMx[slot][1] = (uint8_t)sly;
                if (slz > cMx[slot][2]) cMx[slot][2] = (uint8_t)slz;
            }

            // Concatenate cluster buckets into chunk's flat point array.
            // Update cluster headers (pointFirst, numPoints, bounds).
            uint32_t off = 0;
            uint8_t aMn[3] = { 255, 255, 255 };
            uint8_t aMx[3] = { 0, 0, 0 };
            for (int s = 0; s < lw::kClustersPerChunk; ++s) {
                bc.clusters[s].numPoints = (uint16_t)bucket[s].size();
                bc.clusters[s]._pad = 0;
                bc.clusters[s].pointFirst = off;
                if (bucket[s].empty()) {
                    bc.clusters[s].bounds = 0;
                    continue;
                }
                bc.clusters[s].bounds = lw::PackClusterBounds(
                    cMn[s][0], cMn[s][1], cMn[s][2],
                    cMx[s][0], cMx[s][1], cMx[s][2]);
                for (const auto& p : bucket[s]) {
                    bc.points.push_back(p);
                    if (p.posX < aMn[0]) aMn[0] = p.posX;
                    if (p.posY < aMn[1]) aMn[1] = p.posY;
                    if (p.posZ < aMn[2]) aMn[2] = p.posZ;
                    if (p.posX > aMx[0]) aMx[0] = p.posX;
                    if (p.posY > aMx[1]) aMx[1] = p.posY;
                    if (p.posZ > aMx[2]) aMx[2] = p.posZ;
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
            bc.hdr.totalPoints = (uint32_t)bc.points.size();
            for (int i = 0; i < 8; ++i) bc.hdr.childId[i] = lw::kNoChild;
            bc.hdr.aabbMin[0] = aMn[0]; bc.hdr.aabbMin[1] = aMn[1]; bc.hdr.aabbMin[2] = aMn[2];
            bc.hdr.aabbMax[0] = aMx[0]; bc.hdr.aabbMax[1] = aMx[1]; bc.hdr.aabbMax[2] = aMx[2];
            bc.hdr._pad = 0;
            lods[L].push_back(std::move(bc));
        }
        printf("[LOD %d] baked chunks=%zu points=%zu\n",
               L, lods[L].size(),
               [&]{ size_t t=0; for (auto& c:lods[L]) t+=c.points.size(); return t; }());
    }
    src.clear(); src.shrink_to_fit();

    // ---- fix up childId: link each L>0 chunk to its 8 LOD(L-1) children ----
    for (int L = 1; L < lw::kLodCount; ++L) {
        // Map (gridXYZ) -> idx for finer LOD.
        struct GK { int32_t x, y, z;
                    bool operator==(const GK& o) const { return x==o.x && y==o.y && z==o.z; } };
        struct GKH { size_t operator()(const GK& k) const noexcept {
            uint64_t h = (uint64_t)(uint32_t)k.x * 0x9E3779B97F4A7C15ull;
            h ^= (uint64_t)(uint32_t)k.y * 0xBF58476D1CE4E5B9ull;
            h ^= (uint64_t)(uint32_t)k.z * 0x94D049BB133111EBull;
            return (size_t)(h ^ (h >> 32));
        }};
        std::unordered_map<GK, uint16_t, GKH> finer;
        finer.reserve(lods[L-1].size());
        for (size_t i = 0; i < lods[L-1].size(); ++i) {
            const auto& h = lods[L-1][i].hdr;
            uint16_t idx = (uint16_t)i;
            if (i > 0xFFFEu) { fprintf(stderr, "warn: LOD %d has >65535 chunks; childId truncated\n", L-1); }
            finer[{ h.gridX, h.gridY, h.gridZ }] = idx;
        }
        for (auto& bc : lods[L]) {
            for (int q = 0; q < 8; ++q) {
                int cx = (q >> 0) & 1, cy = (q >> 1) & 1, cz = (q >> 2) & 1;
                GK ck{ bc.hdr.gridX * 2 + cx, bc.hdr.gridY * 2 + cy, bc.hdr.gridZ * 2 + cz };
                auto it = finer.find(ck);
                bc.hdr.childId[q] = (it == finer.end()) ? lw::kNoChild : it->second;
            }
        }
    }

    // ---- write .lw file ----
    fs::create_directories(fs::path(outPath).parent_path());
    FILE* o = fopen(outPath, "wb");
    if (!o) { fprintf(stderr, "cannot open %s for write\n", outPath); return 1; }

    // FileHeader.
    lw::FileHeader fh{};
    fh.magic = lw::kFileMagic;
    fh.version = lw::kFileVersion;
    fh.chunkVoxX = lw::kChunkVoxX; fh.chunkVoxY = lw::kChunkVoxY; fh.chunkVoxZ = lw::kChunkVoxZ;
    fh.clusterVoxX = lw::kClusterVoxX; fh.clusterVoxY = lw::kClusterVoxY; fh.clusterVoxZ = lw::kClusterVoxZ;
    fh.lodCount = lw::kLodCount;
    for (int i = 0; i < 3; ++i) { fh.worldAabbMin[i] = worldMn[i]; fh.worldAabbMax[i] = worldMx[i]; }
    fwrite(&fh, sizeof(fh), 1, o);

    // LODHeader[kLodCount] — chunkTableOffset patched in pass 2.
    long lodHdrPos = ftell(o);
    std::vector<lw::LODHeader> lodHdrs(lw::kLodCount);
    for (int L = 0; L < lw::kLodCount; ++L) {
        lodHdrs[L].lodLevel = (uint32_t)L;
        lodHdrs[L].chunkCount = (uint32_t)lods[L].size();
        lodHdrs[L].chunkTableOffset = 0;
    }
    fwrite(lodHdrs.data(), sizeof(lw::LODHeader), lw::kLodCount, o);

    // Per LOD: reserve space for ChunkEntry table, then write each blob, then
    // backfill ChunkEntry with blob offsets.
    for (int L = 0; L < lw::kLodCount; ++L) {
        lodHdrs[L].chunkTableOffset = (uint64_t)ftell(o);
        std::vector<lw::ChunkEntry> entries(lods[L].size());
        long entryStart = ftell(o);
        // reserve
        fwrite(entries.data(), sizeof(lw::ChunkEntry), entries.size(), o);

        uint64_t lodYBytes = 0, lodMBytes = 0;
        uint64_t lodChoseY = 0, lodChoseM = 0;
        for (size_t i = 0; i < lods[L].size(); ++i) {
            const BakedChunk& bc = lods[L][i];

            // Build per-cluster encoded blobs (bit-grid + colors).
            std::vector<ClusterEnc> ces(lw::kClustersPerChunk);
            uint8_t clusterMask[16] = {};
            uint64_t yT = 0, mT = 0;
            for (int s = 0; s < lw::kClustersPerChunk; ++s) {
                const lw::DiskCluster& cl = bc.clusters[s];
                if (cl.numPoints == 0) continue;
                clusterMask[s >> 3] |= (uint8_t)(1u << (s & 7));
                ces[s] = EncodeClusterTryBoth(&bc.points[cl.pointFirst], cl.numPoints, yT, mT);
                if (ces[s].orderMode == lw::kOrderYMajor) ++lodChoseY; else ++lodChoseM;
            }
            lodYBytes += yT;
            lodMBytes += mT;

            // Serialize compressed chunk blob:
            //   DiskChunkHeader
            //   palette
            //   clusterMask[16]
            //   per non-empty cluster: orderMode + LEB128 numPoints
            //                          + LEB128 bgSize + bgBytes + colorsBytes
            std::vector<uint8_t> blob;
            blob.reserve(sizeof(bc.hdr) + bc.palette.size() * 4 + 16 + 1024);
            auto push = [&](const void* p, size_t n) {
                blob.insert(blob.end(), (const uint8_t*)p, (const uint8_t*)p + n);
            };
            push(&bc.hdr, sizeof(bc.hdr));
            if (!bc.palette.empty()) push(bc.palette.data(), bc.palette.size() * sizeof(uint32_t));
            push(clusterMask, sizeof(clusterMask));
            for (int s = 0; s < lw::kClustersPerChunk; ++s) {
                const lw::DiskCluster& cl = bc.clusters[s];
                if (cl.numPoints == 0) continue;
                const ClusterEnc& ce = ces[s];
                blob.push_back(ce.orderMode);
                lw::Leb128PutU32(blob, (uint32_t)cl.numPoints);
                lw::Leb128PutU32(blob, (uint32_t)ce.bitGrid.size());
                if (!ce.bitGrid.empty()) push(ce.bitGrid.data(), ce.bitGrid.size());
                if (!ce.colors.empty()) push(ce.colors.data(), ce.colors.size());
            }

            // LZ4 on top (header + cluster mask + RLE+colors residual redundancy).
            std::vector<uint8_t> cblob;
            uint32_t writeBytes;
            uint32_t flags = lw::kFlagBitGrid;
            const int rawSize = (int)blob.size();
            const int cap = LZ4_compressBound(rawSize);
            cblob.resize((size_t)cap);
            int cSize = LZ4_compress_default((const char*)blob.data(),
                                             (char*)cblob.data(),
                                             rawSize, cap);
            if (cSize > 0 && cSize < rawSize) {
                writeBytes = (uint32_t)cSize;
                flags |= lw::kFlagLz4;
            } else {
                writeBytes = (uint32_t)rawSize;
            }
            uint64_t blobOff = (uint64_t)ftell(o);
            if (flags & lw::kFlagLz4) fwrite(cblob.data(), 1, writeBytes, o);
            else                       fwrite(blob.data(),  1, writeBytes, o);

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

        if (!lods[L].empty()) {
            printf("[LOD %d] order pick: Y=%llu M=%llu  total Y-only=%llu B, M-only=%llu B\n",
                   L,
                   (unsigned long long)lodChoseY, (unsigned long long)lodChoseM,
                   (unsigned long long)lodYBytes, (unsigned long long)lodMBytes);
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
    return 0;
}
