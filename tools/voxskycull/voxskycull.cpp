// voxskycull.cpp — sky-flood cull. Reads a .vox (VXL3) scene and removes
// solid voxels that are not 6-connectivity reachable through air from
// outside the scene's chunked AABB. Caves, sealed interiors, and other
// non-sky-exposed solids drop. Surface, walls, overhangs, and any voxel
// reachable through an open-air path (e.g. door under an overhang) survive.
//
// Output is a fresh .vox with new chunk metas, recomputed visMask, and
// unchanged palette / origin / sunDir.
//
// Usage:
//   voxskycull.exe <in.vox> <out.vox>

#define _CRT_SECURE_NO_WARNINGS
#include "asset_version.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <deque>
#include <algorithm>

#pragma pack(push, 1)
struct DiskVoxel
{
    uint8_t  x, y, z, visMask;
    uint16_t paletteIdx;
    uint8_t  aoPacked[3];
};
struct ChunkMeta
{
    uint16_t cx, cy, cz, _pad;
    uint32_t voxelCount;
    uint32_t voxelOffset;
};
#pragma pack(pop)
static_assert(sizeof(DiskVoxel) == 9, "");
static_assert(sizeof(ChunkMeta) == 16, "");

static const int kNbr[6][3] = {
    { +1, 0, 0 }, { -1, 0, 0 },
    { 0, +1, 0 }, { 0, -1, 0 },
    { 0, 0, +1 }, { 0, 0, -1 },
};

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: voxskycull <in.vox> <out.vox>\n");
        return 1;
    }
    const char* inPath = argv[1];
    const char* outPath = argv[2];

    FILE* fi = fopen(inPath, "rb");
    if (!fi) { fprintf(stderr, "open %s failed\n", inPath); return 1; }

    char magic[4] = {};
    fread(magic, 1, 4, fi);
    if (memcmp(magic, "VXL3", 4) != 0) {
        fprintf(stderr, "bad magic in %s\n", inPath); fclose(fi); return 1;
    }
    uint32_t version = 0;
    fread(&version, 4, 1, fi);
    uint32_t chunkDim = 0, chunkCount = 0, totalVoxels = 0;
    fread(&chunkDim, 4, 1, fi);
    fread(&chunkCount, 4, 1, fi);
    fread(&totalVoxels, 4, 1, fi);
    int32_t origin[3]; fread(origin, 4, 3, fi);
    float   sunDir[3]; fread(sunDir, 4, 3, fi);
    uint32_t paletteCount = 0; fread(&paletteCount, 4, 1, fi);
    std::vector<uint32_t> palette(paletteCount);
    if (paletteCount) fread(palette.data(), 4, paletteCount, fi);

    std::vector<ChunkMeta> metas(chunkCount);
    if (chunkCount) fread(metas.data(), sizeof(ChunkMeta), chunkCount, fi);
    std::vector<DiskVoxel> voxels(totalVoxels);
    if (totalVoxels) fread(voxels.data(), sizeof(DiskVoxel), totalVoxels, fi);
    fclose(fi);

    printf("[voxskycull] in=%s ver=%u D=%u chunks=%u voxels=%u pal=%u\n",
           inPath, version, chunkDim, chunkCount, totalVoxels, paletteCount);

    if (version != kAssetVersion) {
        fprintf(stderr, "version mismatch: file=%u tool=%u\n", version, kAssetVersion);
        return 1;
    }
    const int D = (int)chunkDim;
    const int N = D * D * D;
    const int u64PerChunk = (N + 63) / 64;

    struct Grid
    {
        std::vector<uint64_t> solid;
        std::vector<uint64_t> exposed; // air cells reachable from outside
        std::vector<uint64_t> kept;    // solid AND surviving cull
    };

    auto CKey = [](int cx, int cy, int cz) -> uint64_t {
        return ((uint64_t)(uint16_t)cx)
             | ((uint64_t)(uint16_t)cy << 16)
             | ((uint64_t)(uint16_t)cz << 32);
    };
    auto Idx = [&](int x, int y, int z) { return (y * D + z) * D + x; };
    auto Get = [](const std::vector<uint64_t>& b, int i) {
        return (b[i >> 6] >> (i & 63)) & 1ull;
    };
    auto Set = [](std::vector<uint64_t>& b, int i) {
        b[i >> 6] |= (uint64_t)1 << (i & 63);
    };

    std::unordered_map<uint64_t, Grid> chunks;
    chunks.reserve(chunkCount * 2);

    // Build solidity bitsets.
    for (uint32_t c = 0; c < chunkCount; ++c) {
        ChunkMeta& m = metas[c];
        Grid& g = chunks[CKey((int16_t)m.cx, (int16_t)m.cy, (int16_t)m.cz)];
        if (g.solid.empty()) {
            g.solid.assign(u64PerChunk, 0);
            g.exposed.assign(u64PerChunk, 0);
            g.kept.assign(u64PerChunk, 0);
        }
        uint32_t off = m.voxelOffset, cnt = m.voxelCount;
        for (uint32_t i = 0; i < cnt; ++i) {
            DiskVoxel& v = voxels[off + i];
            Set(g.solid, Idx(v.x, v.y, v.z));
        }
    }
    printf("[voxskycull] grid: %zu chunks, %.1f MB solid+exposed+kept\n",
           chunks.size(),
           (double)chunks.size() * u64PerChunk * 8 * 3 / (1024.0 * 1024.0));

    // Helper: get pointer to grid given chunk coord, or null.
    auto FindGrid = [&](int cx, int cy, int cz) -> Grid* {
        auto it = chunks.find(CKey(cx, cy, cz));
        return it == chunks.end() ? nullptr : &it->second;
    };

    // ----- Chunk-grid air BFS: classify absent chunks as air vs solid. -----
    // Compute chunk-grid AABB.
    int cxMin = INT32_MAX, cxMax = INT32_MIN;
    int cyMin = INT32_MAX, cyMax = INT32_MIN;
    int czMin = INT32_MAX, czMax = INT32_MIN;
    for (auto& kv : chunks) {
        int cx = (int16_t)(kv.first & 0xFFFF);
        int cy = (int16_t)((kv.first >> 16) & 0xFFFF);
        int cz = (int16_t)((kv.first >> 32) & 0xFFFF);
        cxMin = std::min(cxMin, cx); cxMax = std::max(cxMax, cx);
        cyMin = std::min(cyMin, cy); cyMax = std::max(cyMax, cy);
        czMin = std::min(czMin, cz); czMax = std::max(czMax, cz);
    }
    // BFS over absent chunks inside extended box [cxMin-1..cxMax+1] etc.
    // Absent chunks reachable from outer shell = "open air".
    std::unordered_map<uint64_t, uint8_t> chunkAir;  // present = true-air
    auto AddIfAbsent = [&](int cx, int cy, int cz, std::deque<std::tuple<int,int,int>>& q) {
        if (chunks.find(CKey(cx, cy, cz)) != chunks.end()) return;
        uint64_t k = CKey(cx, cy, cz);
        if (chunkAir.emplace(k, 1).second) q.push_back({ cx, cy, cz });
    };
    {
        std::deque<std::tuple<int,int,int>> q;
        // Seed: outer shell (one layer outside AABB).
        for (int cy = cyMin - 1; cy <= cyMax + 1; ++cy)
        for (int cz = czMin - 1; cz <= czMax + 1; ++cz) {
            AddIfAbsent(cxMin - 1, cy, cz, q);
            AddIfAbsent(cxMax + 1, cy, cz, q);
        }
        for (int cx = cxMin - 1; cx <= cxMax + 1; ++cx)
        for (int cz = czMin - 1; cz <= czMax + 1; ++cz) {
            AddIfAbsent(cx, cyMin - 1, cz, q);
            AddIfAbsent(cx, cyMax + 1, cz, q);
        }
        for (int cx = cxMin - 1; cx <= cxMax + 1; ++cx)
        for (int cy = cyMin - 1; cy <= cyMax + 1; ++cy) {
            AddIfAbsent(cx, cy, czMin - 1, q);
            AddIfAbsent(cx, cy, czMax + 1, q);
        }
        while (!q.empty()) {
            auto [cx, cy, cz] = q.front(); q.pop_front();
            for (int d = 0; d < 6; ++d) {
                int nx = cx + kNbr[d][0];
                int ny = cy + kNbr[d][1];
                int nz = cz + kNbr[d][2];
                if (nx < cxMin - 1 || nx > cxMax + 1) continue;
                if (ny < cyMin - 1 || ny > cyMax + 1) continue;
                if (nz < czMin - 1 || nz > czMax + 1) continue;
                AddIfAbsent(nx, ny, nz, q);
            }
        }
        printf("[voxskycull] chunk-grid AABB cx[%d..%d] cy[%d..%d] cz[%d..%d]  air-absent=%zu\n",
               cxMin, cxMax, cyMin, cyMax, czMin, czMax, chunkAir.size());
    }

    auto AbsentIsAir = [&](int cx, int cy, int cz) -> bool {
        return chunkAir.find(CKey(cx, cy, cz)) != chunkAir.end();
    };

    // Normalize neighbor (cx,cy,cz, lx,ly,lz) crossing chunk boundaries.
    auto Norm = [&](int& cx, int& cy, int& cz, int& lx, int& ly, int& lz) {
        if (lx < 0)      { cx -= 1; lx += D; }
        else if (lx >= D){ cx += 1; lx -= D; }
        if (ly < 0)      { cy -= 1; ly += D; }
        else if (ly >= D){ cy += 1; ly -= D; }
        if (lz < 0)      { cz -= 1; lz += D; }
        else if (lz >= D){ cz += 1; lz -= D; }
    };

    // Seed BFS queue from chunk faces whose neighbor chunk is missing.
    struct QE { int cx, cy, cz, lx, ly, lz; };
    std::deque<QE> q;
    size_t seedCount = 0;
    for (auto& kv : chunks) {
        int cx = (int16_t)(kv.first & 0xFFFF);
        int cy = (int16_t)((kv.first >> 16) & 0xFFFF);
        int cz = (int16_t)((kv.first >> 32) & 0xFFFF);
        Grid& g = kv.second;
        for (int d = 0; d < 6; ++d) {
            int ncx = cx + kNbr[d][0];
            int ncy = cy + kNbr[d][1];
            int ncz = cz + kNbr[d][2];
            if (FindGrid(ncx, ncy, ncz)) continue;     // neighbor chunk exists
            if (!AbsentIsAir(ncx, ncy, ncz)) continue; // absent-solid: no seed
            // Iterate face cells.
            int axis = d >> 1;             // 0=X, 1=Y, 2=Z
            bool plus = (d & 1) == 0;
            int fixed = plus ? D - 1 : 0;
            for (int a = 0; a < D; ++a)
            for (int b = 0; b < D; ++b) {
                int lx, ly, lz;
                if (axis == 0)      { lx = fixed; ly = a; lz = b; }
                else if (axis == 1) { lx = a; ly = fixed; lz = b; }
                else                { lx = a; ly = b; lz = fixed; }
                int i = Idx(lx, ly, lz);
                if (Get(g.solid, i)) continue;
                if (Get(g.exposed, i)) continue;
                Set(g.exposed, i);
                q.push_back({ cx, cy, cz, lx, ly, lz });
                ++seedCount;
            }
        }
    }
    printf("[voxskycull] seeded %zu boundary-air cells, BFS...\n", seedCount);

    // BFS propagate exposed-air.
    size_t bfsVisited = 0;
    while (!q.empty()) {
        QE c = q.front(); q.pop_front();
        ++bfsVisited;
        for (int d = 0; d < 6; ++d) {
            int ncx = c.cx, ncy = c.cy, ncz = c.cz;
            int nx = c.lx + kNbr[d][0];
            int ny = c.ly + kNbr[d][1];
            int nz = c.lz + kNbr[d][2];
            Norm(ncx, ncy, ncz, nx, ny, nz);
            Grid* ng = FindGrid(ncx, ncy, ncz);
            if (!ng) continue;                     // absent → already classified; flood does not enter
            int ni = Idx(nx, ny, nz);
            if (Get(ng->solid, ni)) continue;
            if (Get(ng->exposed, ni)) continue;
            Set(ng->exposed, ni);
            q.push_back({ ncx, ncy, ncz, nx, ny, nz });
        }
        if ((bfsVisited & 0xFFFFFF) == 0) {
            printf("[voxskycull] BFS visited=%zu queue=%zu\n", bfsVisited, q.size());
        }
    }
    printf("[voxskycull] BFS done, visited=%zu\n", bfsVisited);

    // Determine kept-solid: any 6-neighbor cell is exposed-air OR outside-chunk.
    size_t keptCount = 0;
    for (auto& kv : chunks) {
        int cx = (int16_t)(kv.first & 0xFFFF);
        int cy = (int16_t)((kv.first >> 16) & 0xFFFF);
        int cz = (int16_t)((kv.first >> 32) & 0xFFFF);
        Grid& g = kv.second;
        for (int ly = 0; ly < D; ++ly)
        for (int lz = 0; lz < D; ++lz)
        for (int lx = 0; lx < D; ++lx) {
            int i = Idx(lx, ly, lz);
            if (!Get(g.solid, i)) continue;
            bool exposed = false;
            for (int d = 0; d < 6; ++d) {
                int ncx = cx, ncy = cy, ncz = cz;
                int nx = lx + kNbr[d][0];
                int ny = ly + kNbr[d][1];
                int nz = lz + kNbr[d][2];
                Norm(ncx, ncy, ncz, nx, ny, nz);
                Grid* ng = FindGrid(ncx, ncy, ncz);
                if (!ng) {
                    if (AbsentIsAir(ncx, ncy, ncz)) { exposed = true; break; }
                    continue;                          // absent-solid: neighbor opaque
                }
                int ni = Idx(nx, ny, nz);
                if (Get(ng->solid, ni)) continue;     // neighbor solid
                if (Get(ng->exposed, ni)) { exposed = true; break; }
            }
            if (exposed) { Set(g.kept, i); ++keptCount; }
        }
    }
    printf("[voxskycull] kept=%zu of %u  (%.1f%% cull)\n",
           keptCount, totalVoxels,
           100.0 - 100.0 * (double)keptCount / std::max<uint32_t>(totalVoxels, 1));

    // Re-emit: per chunk, walk original voxels, retain only kept ones, recompute visMask.
    std::vector<ChunkMeta> outMetas;
    std::vector<DiskVoxel> outVoxels;
    outMetas.reserve(chunkCount);
    outVoxels.reserve(keptCount);

    for (uint32_t c = 0; c < chunkCount; ++c) {
        ChunkMeta& m = metas[c];
        Grid* g = FindGrid((int16_t)m.cx, (int16_t)m.cy, (int16_t)m.cz);
        if (!g) continue;
        int cx = (int16_t)m.cx, cy = (int16_t)m.cy, cz = (int16_t)m.cz;
        uint32_t off = m.voxelOffset, cnt = m.voxelCount;

        ChunkMeta nm = m;
        nm.voxelOffset = (uint32_t)outVoxels.size();
        nm.voxelCount = 0;
        for (uint32_t i = 0; i < cnt; ++i) {
            DiskVoxel v = voxels[off + i];
            int idx = Idx(v.x, v.y, v.z);
            if (!Get(g->kept, idx)) continue;

            // Recompute visMask: face open iff neighbor cell is not (solid AND kept).
            uint8_t vm = 0;
            const int faceBit[6] = { 0, 1, 2, 3, 4, 5 };  // +X -X +Y -Y +Z -Z
            for (int d = 0; d < 6; ++d) {
                int ncx = cx, ncy = cy, ncz = cz;
                int nx = v.x + kNbr[d][0];
                int ny = v.y + kNbr[d][1];
                int nz = v.z + kNbr[d][2];
                Norm(ncx, ncy, ncz, nx, ny, nz);
                Grid* ng = FindGrid(ncx, ncy, ncz);
                if (!ng) {
                    if (AbsentIsAir(ncx, ncy, ncz)) vm |= 1u << faceBit[d];
                    continue;                           // absent-solid: face hidden
                }
                int ni = Idx(nx, ny, nz);
                if (Get(ng->kept, ni)) continue;        // neighbor solid+kept → face hidden
                vm |= 1u << faceBit[d];
            }
            vm &= ~(1u << 3);   // -Y always 0 per format spec
            v.visMask = vm;
            outVoxels.push_back(v);
            ++nm.voxelCount;
        }
        if (nm.voxelCount > 0) outMetas.push_back(nm);
    }

    // Write output.
    FILE* fo = fopen(outPath, "wb");
    if (!fo) { fprintf(stderr, "open %s for write failed\n", outPath); return 1; }
    fwrite("VXL3", 1, 4, fo);
    fwrite(&version, 4, 1, fo);
    fwrite(&chunkDim, 4, 1, fo);
    uint32_t ncc = (uint32_t)outMetas.size();
    fwrite(&ncc, 4, 1, fo);
    uint32_t ntv = (uint32_t)outVoxels.size();
    fwrite(&ntv, 4, 1, fo);
    fwrite(origin, 4, 3, fo);
    fwrite(sunDir, 4, 3, fo);
    fwrite(&paletteCount, 4, 1, fo);
    if (paletteCount) fwrite(palette.data(), 4, paletteCount, fo);
    if (!outMetas.empty()) fwrite(outMetas.data(), sizeof(ChunkMeta), outMetas.size(), fo);
    if (!outVoxels.empty()) fwrite(outVoxels.data(), sizeof(DiskVoxel), outVoxels.size(), fo);
    fclose(fo);

    printf("[voxskycull] wrote %s  chunks=%u  voxels=%u\n", outPath, ncc, ntv);
    return 0;
}
