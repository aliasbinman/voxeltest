// box_stats.cpp — bucket .vox L0 voxels into power-of-2 boxes (XZ x XZ x Y,
// all in [64..1024], Y <= XZ). Reports min/max/avg points per non-empty
// box, per LOD (L0 raw, L1=floor/2, L2=floor/4, L3=floor/8 unique cells).
// Reads .vox raw so it works on huge files without going through vox_loader.
//
// Usage:
//   box_stats.exe [in.vox] [out.md]
// Defaults: assets/kingslanding.vox  docs/box_stats_kingslanding.md

#define _CRT_SECURE_NO_WARNINGS
#include "asset_version.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

#pragma pack(push, 1)
struct DiskVoxel { uint8_t x, y, z; uint8_t visMask; uint16_t paletteIdx; uint8_t aoPacked[3]; };
#pragma pack(pop)
static_assert(sizeof(DiskVoxel) == 9, "");

struct ChunkMeta {
    uint16_t cx, cy, cz, _pad;
    uint32_t voxelCount;
    uint32_t voxelOffset;
};
static_assert(sizeof(ChunkMeta) == 16, "");

struct V { int32_t x, y, z; };

struct BoxKey { int32_t bx, by, bz; bool operator==(const BoxKey& o) const { return bx == o.bx && by == o.by && bz == o.bz; } };
struct BoxKeyHash { size_t operator()(const BoxKey& k) const noexcept {
    uint64_t h = (uint64_t)(uint32_t)k.bx * 0x9E3779B97F4A7C15ull;
    h ^= (uint64_t)(uint32_t)k.by * 0xBF58476D1CE4E5B9ull;
    h ^= (uint64_t)(uint32_t)k.bz * 0x94D049BB133111EBull;
    return (size_t)(h ^ (h >> 32));
}};

struct CellKey { int32_t x, y, z; bool operator==(const CellKey& o) const { return x == o.x && y == o.y && z == o.z; } };
struct CellKeyHash { size_t operator()(const CellKey& k) const noexcept {
    uint64_t h = (uint64_t)(uint32_t)k.x * 0x9E3779B97F4A7C15ull;
    h ^= (uint64_t)(uint32_t)k.y * 0xBF58476D1CE4E5B9ull;
    h ^= (uint64_t)(uint32_t)k.z * 0x94D049BB133111EBull;
    return (size_t)(h ^ (h >> 32));
}};

static inline int32_t FloorDiv(int32_t a, int32_t b) {
    int32_t q = a / b;
    if ((a % b != 0) && ((a ^ b) < 0)) --q;
    return q;
}

struct Row {
    int32_t  xz, y;
    uint64_t nonEmpty;
    uint32_t minN, maxN;
    double   avgN;
    uint64_t coverage;
};

static std::vector<Row> StatsForLod(const std::vector<V>& positions,
                                    const int32_t mn[3], const int32_t mx[3])
{
    std::vector<Row> rows;
    const int sizes[] = { 64, 128, 256, 512, 1024 };
    for (int xz : sizes) {
        for (int y : sizes) {
            if (y > xz) continue;
            std::unordered_map<BoxKey, uint32_t, BoxKeyHash> bins;
            bins.reserve(positions.size() / 64 + 1);
            for (const V& v : positions) {
                BoxKey k{ FloorDiv(v.x, xz), FloorDiv(v.y, y), FloorDiv(v.z, xz) };
                bins[k]++;
            }
            uint32_t bmin = UINT32_MAX, bmax = 0;
            uint64_t bsum = 0;
            for (auto& kv : bins) {
                bmin = std::min(bmin, kv.second);
                bmax = std::max(bmax, kv.second);
                bsum += kv.second;
            }
            double avg = bins.empty() ? 0.0 : (double)bsum / (double)bins.size();
            if (bins.empty()) bmin = 0;
            int32_t bxLo = FloorDiv(mn[0], xz), bxHi = FloorDiv(mx[0], xz);
            int32_t byLo = FloorDiv(mn[1], y ), byHi = FloorDiv(mx[1], y );
            int32_t bzLo = FloorDiv(mn[2], xz), bzHi = FloorDiv(mx[2], xz);
            uint64_t coverage =
                (uint64_t)(bxHi - bxLo + 1) *
                (uint64_t)(byHi - byLo + 1) *
                (uint64_t)(bzHi - bzLo + 1);
            rows.push_back({ xz, y, (uint64_t)bins.size(), bmin, bmax, avg, coverage });
        }
    }
    return rows;
}

static std::vector<V> DownsampleUnique(const std::vector<V>& src, int step)
{
    std::unordered_set<CellKey, CellKeyHash> seen;
    seen.reserve(src.size() / (size_t)(step * step * step) + 1024);
    std::vector<V> out;
    for (const V& v : src) {
        CellKey k{ FloorDiv(v.x, step), FloorDiv(v.y, step), FloorDiv(v.z, step) };
        if (seen.insert(k).second) out.push_back({ k.x, k.y, k.z });
    }
    return out;
}

int main(int argc, char** argv)
{
    const char* inPath  = (argc > 1) ? argv[1] : "assets/kingslanding.vox";
    const char* outPath = (argc > 2) ? argv[2] : "docs/box_stats_kingslanding.md";

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
    uint32_t chunkDim = 0, chunkCount = 0, totalVoxels = 0;
    int32_t  srcOrigin[3] = { 0, 0, 0 };
    float    srcSun[3]    = { 0, 0, 0 };
    fread(&chunkDim,    sizeof(uint32_t), 1, f);
    fread(&chunkCount,  sizeof(uint32_t), 1, f);
    fread(&totalVoxels, sizeof(uint32_t), 1, f);
    fread(srcOrigin,    sizeof(int32_t),  3, f);
    fread(srcSun,       sizeof(float),    3, f);
    uint32_t paletteCount = 0;
    fread(&paletteCount, sizeof(uint32_t), 1, f);
    if (paletteCount) fseek(f, (long)(paletteCount * sizeof(uint32_t)), SEEK_CUR);
    std::vector<ChunkMeta> srcMetas(chunkCount);
    if (chunkCount) fread(srcMetas.data(), sizeof(ChunkMeta), chunkCount, f);
    std::vector<DiskVoxel> srcVox(totalVoxels);
    if (totalVoxels) fread(srcVox.data(), sizeof(DiskVoxel), totalVoxels, f);
    fclose(f);
    printf("[in] %s chunkDim=%u chunkCount=%u voxels=%u\n",
           inPath, chunkDim, chunkCount, totalVoxels);

    const int32_t D = (int32_t)chunkDim;

    // L0 = raw voxels (only counting those with visMask != 0; matches engine
    // which culls fully-hidden voxels in vox_loader pass 1).
    std::vector<V> l0;
    l0.reserve(totalVoxels);
    int32_t mn[3] = {  INT32_MAX,  INT32_MAX,  INT32_MAX };
    int32_t mx[3] = { -INT32_MAX, -INT32_MAX, -INT32_MAX };
    for (uint32_t ci = 0; ci < chunkCount; ++ci) {
        const ChunkMeta& m = srcMetas[ci];
        int32_t bx = (int32_t)m.cx * D, by = (int32_t)m.cy * D, bz = (int32_t)m.cz * D;
        const DiskVoxel* vs = srcVox.data() + m.voxelOffset;
        for (uint32_t i = 0; i < m.voxelCount; ++i) {
            const DiskVoxel& dv = vs[i];
            if (dv.visMask == 0) continue;
            int32_t x = bx + (int32_t)dv.x, y = by + (int32_t)dv.y, z = bz + (int32_t)dv.z;
            l0.push_back({ x, y, z });
            if (x < mn[0]) mn[0] = x; if (x > mx[0]) mx[0] = x;
            if (y < mn[1]) mn[1] = y; if (y > mx[1]) mx[1] = y;
            if (z < mn[2]) mn[2] = z; if (z > mx[2]) mx[2] = z;
        }
    }
    srcVox.clear(); srcVox.shrink_to_fit();
    srcMetas.clear(); srcMetas.shrink_to_fit();

    printf("L0 voxels: %llu\n", (unsigned long long)l0.size());
    printf("AABB: x[%d..%d] y[%d..%d] z[%d..%d]  span=(%d,%d,%d)\n",
           mn[0], mx[0], mn[1], mx[1], mn[2], mx[2],
           mx[0] - mn[0] + 1, mx[1] - mn[1] + 1, mx[2] - mn[2] + 1);

    printf("Computing LOD unique-cell sets...\n");
    std::vector<V> l1 = DownsampleUnique(l0, 2);
    printf("  L1 cells: %llu\n", (unsigned long long)l1.size());
    std::vector<V> l2 = DownsampleUnique(l0, 4);
    printf("  L2 cells: %llu\n", (unsigned long long)l2.size());
    std::vector<V> l3 = DownsampleUnique(l0, 8);
    printf("  L3 cells: %llu\n", (unsigned long long)l3.size());

    struct LodOut { const char* name; std::vector<V>* pts; int step; };
    LodOut lods[] = {
        { "L0", &l0, 1 },
        { "L1", &l1, 2 },
        { "L2", &l2, 4 },
        { "L3", &l3, 8 },
    };

    fs::create_directories(fs::path(outPath).parent_path());
    FILE* o = fopen(outPath, "w");
    if (!o) { fprintf(stderr, "cannot open %s\n", outPath); return 1; }

    fprintf(o, "# Box-bucketing stats: %s\n\n", fs::path(inPath).filename().string().c_str());
    fprintf(o, "Source: `%s`  \n", inPath);
    fprintf(o, "L0 voxels: **%llu**, L1: %llu, L2: %llu, L3: %llu  \n",
            (unsigned long long)l0.size(), (unsigned long long)l1.size(),
            (unsigned long long)l2.size(), (unsigned long long)l3.size());
    fprintf(o, "Scene AABB (voxels): x[%d..%d] y[%d..%d] z[%d..%d]  \n",
            mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]);
    fprintf(o, "Scene span: %d x %d x %d voxels\n\n",
            mx[0] - mn[0] + 1, mx[1] - mn[1] + 1, mx[2] - mn[2] + 1);

    fprintf(o, "## Method\n\n");
    fprintf(o, "L0 = raw visible voxels. L1/L2/L3 = unique cells at `floor(pos/{2,4,8})` ");
    fprintf(o, "(matches the renderer's downsample factor; counts are equal to engine LOD vert counts).\n\n");
    fprintf(o, "Each point binned into box `(bx,by,bz) = (floor(x/XZ), floor(y/Y), floor(z/XZ))`. ");
    fprintf(o, "Box dimensions: `XZ x Y x XZ`, `XZ, Y in {64,128,256,512,1024}`, `Y <= XZ`. ");
    fprintf(o, "Box origins are multiples of box size per axis.\n\n");
    fprintf(o, "- **non-empty**: boxes with >= 1 point\n");
    fprintf(o, "- **min/max/avg**: points per non-empty box\n");
    fprintf(o, "- **coverage**: boxes intersecting scene AABB; fill = non-empty / coverage\n\n");

    for (LodOut& lo : lods) {
        // AABB at this LOD = downsample of L0 AABB.
        int32_t lmn[3] = { FloorDiv(mn[0], lo.step), FloorDiv(mn[1], lo.step), FloorDiv(mn[2], lo.step) };
        int32_t lmx[3] = { FloorDiv(mx[0], lo.step), FloorDiv(mx[1], lo.step), FloorDiv(mx[2], lo.step) };

        printf("Stats for %s (step=%d, %llu points)...\n",
               lo.name, lo.step, (unsigned long long)lo.pts->size());
        std::vector<Row> rows = StatsForLod(*lo.pts, lmn, lmx);

        fprintf(o, "## %s (%llu points, downsample step=%d)\n\n",
                lo.name, (unsigned long long)lo.pts->size(), lo.step);
        fprintf(o, "Coordinates here are in `%s` cells (each cell = %d source voxels per axis).\n\n",
                lo.name, lo.step);
        fprintf(o, "| Box (XxYxZ in %s cells) | Non-empty | Min | Max | Avg | Coverage | Fill |\n", lo.name);
        fprintf(o, "|---|---:|---:|---:|---:|---:|---:|\n");
        for (const Row& r : rows) {
            double fill = r.coverage ? (double)r.nonEmpty / (double)r.coverage : 0.0;
            fprintf(o, "| %d x %d x %d | %llu | %u | %u | %.1f | %llu | %.1f%% |\n",
                    r.xz, r.y, r.xz,
                    (unsigned long long)r.nonEmpty,
                    r.minN, r.maxN, r.avgN,
                    (unsigned long long)r.coverage, fill * 100.0);
        }
        fprintf(o, "\n");
    }

    fprintf(o, "## Notes\n\n");
    fprintf(o, "- L0 count excludes voxels with visMask=0 (engine-culled).\n");
    fprintf(o, "- L1/L2/L3 box sizes are in their own cell units. To compare in source-voxel units, multiply by the LOD step.\n");
    fprintf(o, "- Avg is over non-empty boxes; total points = Non-empty × Avg.\n");

    fclose(o);
    printf("\nWrote %s\n", outPath);
    return 0;
}
