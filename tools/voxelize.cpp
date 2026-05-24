#define _CRT_SECURE_NO_WARNINGS
#include "obj_loader.h"
#include "obj_scene.h"
#include "../src/asset_version.h"

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <climits>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <atomic>
#include <unordered_map>
#include <intrin.h>

// File format (VXL2)
//   char     magic[4] = "VXL2"
//   uint32_t chunkDim
//   uint32_t chunkCount
//   uint32_t totalVoxelCount
//   int32_t  origin[3]
//   ChunkMeta[chunkCount]
//   DiskVoxel[totalVoxelCount]
//
// ChunkMeta (16 bytes):
//   uint16_t cx, cy, cz, _pad
//   uint32_t voxelCount
//   uint32_t voxelOffset       // index into voxel array
//
// DiskVoxel (7 bytes, #pragma packed):
//   uint8_t  x, y, z
//   uint32_t color             // RGBA8 little-endian

static constexpr int CHUNK_DIM = 64;
static_assert(CHUNK_DIM > 0 && CHUNK_DIM <= 256, "");

struct VKey {
    int32_t x, y, z;
    bool operator==(const VKey& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct VKeyHash {
    size_t operator()(const VKey& k) const noexcept {
        uint64_t h = (uint64_t)(uint32_t)k.x * 0x9E3779B185EBCA87ull;
        h ^= (uint64_t)(uint32_t)k.y * 0xC2B2AE3D27D4EB4Full;
        h ^= (uint64_t)(uint32_t)k.z * 0x165667B19E3779F9ull;
        h ^= h >> 32;
        return (size_t)h;
    }
};

#pragma pack(push, 1)
struct DiskVoxel { uint8_t x, y, z; uint8_t visMask; uint32_t color; uint8_t ao[6]; };
#pragma pack(pop)
static_assert(sizeof(DiskVoxel) == 14, "");

// In-memory voxel record. Color RGB + per-face AO (one byte per cardinal
// face, indexed by kFaceDelta order: 0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z).
struct VoxData { uint32_t color; uint8_t ao[6]; };

// Face order must match vox_loader.cpp:
// 0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z
static const int kFaceDelta[6][3] = {
    { 1, 0, 0 }, { -1, 0, 0 },
    { 0, 1, 0 }, { 0, -1, 0 },
    { 0, 0, 1 }, { 0, 0, -1 },
};

struct ChunkMeta {
    uint16_t cx, cy, cz, _pad;
    uint32_t voxelCount;
    uint32_t voxelOffset;
};
static_assert(sizeof(ChunkMeta) == 16, "");

static inline uint32_t PackRGBA(float r, float g, float b) {
    auto q = [](float v) -> uint32_t {
        v = std::clamp(v, 0.0f, 1.0f);
        return (uint32_t)(v * 255.0f + 0.5f);
    };
    return q(r) | (q(g) << 8) | (q(b) << 16) | (255u << 24);
}

int main(int argc, char** argv)
{
    const char* in  = argc > 1 ? argv[1] : "assets/rungholt/rungholt.obj";
    const char* out = argc > 2 ? argv[2] : "assets/rungholt.vox";

    printf("Loading %s\n", in);
    ObjScene scene;
    std::string err;
    if (!LoadObjScene(in, scene, err)) {
        fprintf(stderr, "Load failed: %s\n", err.c_str());
        return 1;
    }
    printf("OBJ tris=%llu  subs=%zu\n",
           (unsigned long long)scene.totalTriangles, scene.subs.size());

    std::unordered_map<VKey, VoxData, VKeyHash> voxels;
    voxels.reserve(4u << 20);

    for (const auto& s : scene.subs) {
        size_t tris = s.indexCount / 3;
        for (size_t t = 0; t < tris; ++t) {
            uint32_t i0 = scene.indices[s.firstIndex + t * 3 + 0] + s.baseVertex;
            uint32_t i1 = scene.indices[s.firstIndex + t * 3 + 1] + s.baseVertex;
            uint32_t i2 = scene.indices[s.firstIndex + t * 3 + 2] + s.baseVertex;
            const ObjVertex& a = scene.vertices[i0];
            const ObjVertex& b = scene.vertices[i1];
            const ObjVertex& c = scene.vertices[i2];
            float cx = (a.px + b.px + c.px) * (1.0f / 3.0f);
            float cy = (a.py + b.py + c.py) * (1.0f / 3.0f);
            float cz = (a.pz + b.pz + c.pz) * (1.0f / 3.0f);
            float vx = cx - 0.5f * a.nx;
            float vy = cy - 0.5f * a.ny;
            float vz = cz - 0.5f * a.nz;
            VKey k{ (int32_t)floorf(vx), (int32_t)floorf(vy), (int32_t)floorf(vz) };
            VoxData vd{ a.color, { 0, 0, 0, 0, 0, 0 } };
            voxels.emplace(k, vd);
        }
    }
    printf("Unique voxels: %zu\n", voxels.size());
    if (voxels.empty()) { fprintf(stderr, "No voxels produced\n"); return 1; }

    int32_t minX = INT32_MAX, minY = INT32_MAX, minZ = INT32_MAX;
    int32_t maxX = INT32_MIN, maxY = INT32_MIN, maxZ = INT32_MIN;
    for (auto& kv : voxels) {
        minX = std::min(minX, kv.first.x);
        minY = std::min(minY, kv.first.y);
        minZ = std::min(minZ, kv.first.z);
        maxX = std::max(maxX, kv.first.x);
        maxY = std::max(maxY, kv.first.y);
        maxZ = std::max(maxZ, kv.first.z);
    }
    printf("Voxel AABB: (%d,%d,%d) .. (%d,%d,%d)\n", minX, minY, minZ, maxX, maxY, maxZ);

    // Sun direction is written to the file for runtime lighting (no longer
    // pre-baked into per-face shadow; runtime does sun shading).
    float sunDirX = 0.4f, sunDirY = 0.8f, sunDirZ = 0.2f;
    {
        float len = sqrtf(sunDirX*sunDirX + sunDirY*sunDirY + sunDirZ*sunDirZ);
        sunDirX /= len; sunDirY /= len; sunDirZ /= len;
    }

    // ---------------------------------------------------------------------
    // AO bake (per voxel, hemisphere raycast).
    // For each exposed face, cast N cosine-weighted rays into the outer
    // hemisphere; count hits within kAoMaxSteps cells. Voxel AO = average
    // (1 - hitFraction) across exposed faces. Stored in color alpha (0..255,
    // 0 = fully occluded, 255 = fully open).
    // ---------------------------------------------------------------------
    int32_t spanX0 = maxX - minX + 1;
    int32_t spanY0 = maxY - minY + 1;
    int32_t spanZ0 = maxZ - minZ + 1;
    std::vector<uint8_t> filledBitmap(((size_t)spanX0 * spanY0 * spanZ0 + 7) / 8, 0);
    auto bIdx0 = [&](int x, int y, int z) -> size_t {
        return (size_t)((z - minZ) * spanY0 + (y - minY)) * (size_t)spanX0 + (size_t)(x - minX);
    };
    for (auto& kv : voxels) {
        size_t i = bIdx0(kv.first.x, kv.first.y, kv.first.z);
        filledBitmap[i >> 3] |= (uint8_t)(1u << (i & 7));
    }
    auto isFilledBM = [&](int x, int y, int z) -> bool {
        if (x < minX || x > maxX || y < minY || y > maxY || z < minZ || z > maxZ) return false;
        size_t i = bIdx0(x, y, z);
        return (filledBitmap[i >> 3] >> (i & 7)) & 1u;
    };
    {
        const int kAoSamples = 48;
        const int kAoMaxSteps = 14;
        const float kTwoPi = 6.2831853071795864f;

        // Hammersley (van der Corput) for sample i in [0,N).
        auto Hammersley = [&](int i, float& u, float& v) {
            u = (float)i / (float)kAoSamples;
            uint32_t b = (uint32_t)i;
            b = ((b & 0x55555555u) << 1) | ((b & 0xAAAAAAAAu) >> 1);
            b = ((b & 0x33333333u) << 2) | ((b & 0xCCCCCCCCu) >> 2);
            b = ((b & 0x0F0F0F0Fu) << 4) | ((b & 0xF0F0F0F0u) >> 4);
            b = ((b & 0x00FF00FFu) << 8) | ((b & 0xFF00FF00u) >> 8);
            b = (b << 16) | (b >> 16);
            v = (float)b * 2.3283064365386963e-10f;
        };

        // Snapshot map entries into a vector so OpenMP can parallelize by index.
        // Pointers into the unordered_map's values stay valid while we no
        // longer insert/erase from `voxels`.
        std::vector<std::pair<VKey, VoxData*>> voxList;
        voxList.reserve(voxels.size());
        for (auto& kv : voxels) voxList.emplace_back(kv.first, &kv.second);

        uint64_t totalRays = 0, totalHits = 0;
        double aoSumAll = 0.0; uint64_t aoVoxels = 0; uint64_t aoFaces = 0;
        const int64_t voxTot = (int64_t)voxList.size();
        printf("AO bake start: %lld voxels, samples=%d, maxSteps=%d\n",
               (long long)voxTot, kAoSamples, kAoMaxSteps);
        fflush(stdout);

        std::atomic<int64_t> progress{0};
        const int64_t reportStride = voxTot / 10;
        #pragma omp parallel for schedule(dynamic, 256) \
                    reduction(+:totalRays) reduction(+:totalHits) \
                    reduction(+:aoSumAll)  reduction(+:aoVoxels) reduction(+:aoFaces)
        for (int64_t li = 0; li < voxTot; ++li) {
            const VKey  k  = voxList[(size_t)li].first;
            VoxData*    vd = voxList[(size_t)li].second;
            int vx = k.x, vy = k.y, vz = k.z;
            // Per-voxel Cranley-Patterson rotation: shifts the Hammersley
            // sequence by a hash of voxel coords so adjacent voxels don't
            // share identical sample directions (kills banding on flat
            // surfaces sitting next to occluders).
            uint32_t h = (uint32_t)vx * 0x9E3779B1u
                       ^ (uint32_t)vy * 0x85EBCA77u
                       ^ (uint32_t)vz * 0xC2B2AE3Du;
            h ^= h >> 16; h *= 0x7FEB352Du;
            h ^= h >> 15; h *= 0x846CA68Bu;
            h ^= h >> 16;
            float jitU = (float)(h & 0xFFFFu)         * (1.0f / 65536.0f);
            float jitV = (float)((h >> 16) & 0xFFFFu) * (1.0f / 65536.0f);
            // Per-face AO; interior face stays 0 (won't be rendered).
            for (int fi = 0; fi < 6; ++fi) { vd->ao[fi] = 0; }
            bool anyExposed = false;
            for (int fi = 0; fi < 6; ++fi) {
                int nxF = vx + kFaceDelta[fi][0];
                int nyF = vy + kFaceDelta[fi][1];
                int nzF = vz + kFaceDelta[fi][2];
                if (isFilledBM(nxF, nyF, nzF)) continue;     // interior face
                anyExposed = true;

                float Nx = (float)kFaceDelta[fi][0];
                float Ny = (float)kFaceDelta[fi][1];
                float Nz = (float)kFaceDelta[fi][2];
                // Axis-aligned tangent frame.
                float Tx, Ty, Tz, Bx, By, Bz;
                if (fi < 2)      { Tx = 0; Ty = 1; Tz = 0; Bx = 0; By = 0; Bz = 1; }
                else if (fi < 4) { Tx = 1; Ty = 0; Tz = 0; Bx = 0; By = 0; Bz = 1; }
                else             { Tx = 1; Ty = 0; Tz = 0; Bx = 0; By = 1; Bz = 0; }

                int hits = 0;
                for (int s = 0; s < kAoSamples; ++s) {
                    float u, v;
                    Hammersley(s, u, v);
                    u += jitU; u -= floorf(u);
                    v += jitV; v -= floorf(v);
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
                    float tmx = sx > 0 ? (1.0f - fx) * invDx : sx < 0 ? fx * invDx : 1e30f;
                    float tmy = sy > 0 ? (1.0f - fy) * invDy : sy < 0 ? fy * invDy : 1e30f;
                    float tmz = sz > 0 ? (1.0f - fz) * invDz : sz < 0 ? fz * invDz : 1e30f;
                    bool hit = false;
                    for (int step = 0; step < kAoMaxSteps; ++step) {
                        if (tmx < tmy && tmx < tmz)      { ix += sx; tmx += invDx; }
                        else if (tmy < tmz)              { iy += sy; tmy += invDy; }
                        else                             { iz += sz; tmz += invDz; }
                        if (isFilledBM(ix, iy, iz)) { hit = true; break; }
                    }
                    if (hit) ++hits;
                    ++totalRays;
                }
                totalHits += hits;
                float faceAo = 1.0f - (float)hits / (float)kAoSamples;
                uint32_t aoByte = (uint32_t)(faceAo * 255.0f + 0.5f);
                aoByte = std::clamp(aoByte, 1u, 255u);   // reserve 0 as "no data"
                vd->ao[fi] = (uint8_t)aoByte;
                aoSumAll += faceAo; ++aoFaces;
            }
            if (anyExposed) ++aoVoxels;

            // Lock-free progress report.
            if (reportStride > 0) {
                int64_t done = progress.fetch_add(1, std::memory_order_relaxed) + 1;
                if (done % reportStride == 0) {
                    int pct = (int)((done * 100) / voxTot);
                    printf("  AO bake %d%%\n", pct);
                    fflush(stdout);
                }
            }
        }
        printf("AO bake: %llu rays, %.1f%% hit, avg face AO %.3f (samples=%d, maxSteps=%d)\n",
               (unsigned long long)totalRays,
               totalRays ? 100.0 * (double)totalHits / (double)totalRays : 0.0,
               aoFaces ? aoSumAll / (double)aoFaces : 0.0,
               kAoSamples, kAoMaxSteps);

        // Per-face smoothing, parallelized. aoArr[i][fi] mirrors voxList[i]'s
        // current AO; threads read aoArr concurrently and write to a fresh
        // `next` vector, no races. keyIdx maps VKey -> voxList index for
        // tangent-neighbor lookups.
        std::vector<std::array<uint8_t,6>> aoArr(voxList.size());
        std::unordered_map<VKey, uint32_t, VKeyHash> keyIdx;
        keyIdx.reserve(voxList.size());
        for (size_t i = 0; i < voxList.size(); ++i) {
            keyIdx[voxList[i].first] = (uint32_t)i;
            for (int fi = 0; fi < 6; ++fi) aoArr[i][fi] = voxList[i].second->ao[fi];
        }
        const int kTan[6][4][3] = {
            {{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}},
            {{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}},
            {{1,0,0},{-1,0,0},{0,0,1},{0,0,-1}},
            {{1,0,0},{-1,0,0},{0,0,1},{0,0,-1}},
            {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0}},
            {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0}},
        };
        const int kSmoothIters = 2;
        std::vector<std::array<uint8_t,6>> next(voxList.size());
        for (int iter = 0; iter < kSmoothIters; ++iter) {
            printf("  AO smooth iter %d/%d\n", iter + 1, kSmoothIters);
            fflush(stdout);
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < (int64_t)voxList.size(); ++i) {
                const VKey k = voxList[(size_t)i].first;
                std::array<uint8_t,6> nv{};
                const std::array<uint8_t,6>& self = aoArr[(size_t)i];
                for (int fi = 0; fi < 6; ++fi) {
                    if (self[fi] == 0) { nv[fi] = 0; continue; }
                    uint32_t sum = self[fi];
                    int cnt = 1;
                    for (int t = 0; t < 4; ++t) {
                        VKey nk{ k.x + kTan[fi][t][0],
                                 k.y + kTan[fi][t][1],
                                 k.z + kTan[fi][t][2] };
                        auto it = keyIdx.find(nk);
                        if (it == keyIdx.end()) continue;
                        uint8_t na = aoArr[it->second][fi];
                        if (na == 0) continue;
                        sum += na;
                        ++cnt;
                    }
                    uint8_t a = (uint8_t)(sum / (uint32_t)cnt);
                    a = std::max(a, (uint8_t)1);
                    nv[fi] = a;
                }
                next[(size_t)i] = nv;
            }
            aoArr.swap(next);
        }
        for (size_t i = 0; i < voxList.size(); ++i) {
            for (int fi = 0; fi < 6; ++fi) voxList[i].second->ao[fi] = aoArr[i][fi];
        }
        printf("AO smoothing: %d iters of 4-tangent-neighbor average (per face)\n", kSmoothIters);
    }

    struct CKey { uint16_t x, y, z; };
    struct CKeyHash {
        size_t operator()(const CKey& k) const noexcept {
            return ((size_t)k.x * 73856093u) ^ ((size_t)k.y * 19349663u) ^ ((size_t)k.z * 83492791u);
        }
    };
    struct CKeyEq { bool operator()(const CKey& a, const CKey& b) const { return a.x == b.x && a.y == b.y && a.z == b.z; } };

    // Pre-pass: for each (x, z) column, find the highest y. Used to keep
    // only the top voxel on the four extreme XZ walls (min/max X, min/max Z).
    // Bottom layer (y == minY) is dropped entirely.
    struct ColKey { int32_t x, z; };
    struct ColHash { size_t operator()(const ColKey& k) const noexcept {
        return ((size_t)(uint32_t)k.x * 73856093u) ^ ((size_t)(uint32_t)k.z * 83492791u);
    }};
    struct ColEq { bool operator()(const ColKey& a, const ColKey& b) const { return a.x == b.x && a.z == b.z; } };
    std::unordered_map<ColKey, int32_t, ColHash, ColEq> colTopY;
    colTopY.reserve(voxels.size());
    for (auto& kv : voxels) {
        ColKey c{ kv.first.x, kv.first.z };
        auto it = colTopY.find(c);
        if (it == colTopY.end()) colTopY.emplace(c, kv.first.y);
        else if (kv.first.y > it->second) it->second = kv.first.y;
    }
    uint64_t droppedBottom = 0, droppedEdge = 0;

    std::unordered_map<CKey, std::vector<DiskVoxel>, CKeyHash, CKeyEq> chunks;
    uint64_t visibleFaces = 0;
    for (auto& kv : voxels) {
        // Drop entire bottom layer.
        if (kv.first.y == minY) { ++droppedBottom; continue; }
        // Extreme XZ edges: keep only the topmost voxel in that column.
        const bool onEdgeXZ = (kv.first.x == minX || kv.first.x == maxX
                            || kv.first.z == minZ || kv.first.z == maxZ);
        if (onEdgeXZ) {
            ColKey c{ kv.first.x, kv.first.z };
            auto it = colTopY.find(c);
            if (it != colTopY.end() && kv.first.y != it->second) {
                ++droppedEdge;
                continue;
            }
        }
        int32_t rx = kv.first.x - minX;
        int32_t ry = kv.first.y - minY;
        int32_t rz = kv.first.z - minZ;
        CKey ck{ (uint16_t)(rx / CHUNK_DIM), (uint16_t)(ry / CHUNK_DIM), (uint16_t)(rz / CHUNK_DIM) };

        uint8_t mask = 0;
        for (int fi = 0; fi < 6; ++fi) {
            VKey n{ kv.first.x + kFaceDelta[fi][0],
                    kv.first.y + kFaceDelta[fi][1],
                    kv.first.z + kFaceDelta[fi][2] };
            if (voxels.find(n) == voxels.end()) mask |= (uint8_t)(1u << fi);
        }
        // Drop the -Y (downward) face: never seen from above. Voxels whose
        // ONLY visible face was -Y get culled entirely.
        mask &= (uint8_t)~0x08u;
        if (mask == 0) continue;
        visibleFaces += __popcnt(mask);

        DiskVoxel v;
        v.x = (uint8_t)(rx % CHUNK_DIM);
        v.y = (uint8_t)(ry % CHUNK_DIM);
        v.z = (uint8_t)(rz % CHUNK_DIM);
        v.visMask = mask;
        v.color = kv.second.color;
        for (int fi = 0; fi < 6; ++fi) v.ao[fi] = kv.second.ao[fi];
        chunks[ck].push_back(v);
    }
    printf("Visible faces (after neighbor cull): %llu (avg %.2f/voxel)\n",
           (unsigned long long)visibleFaces, (double)visibleFaces / (double)voxels.size());
    printf("Chunks: %zu (dim=%d)\n", chunks.size(), CHUNK_DIM);

    // Build flat metadata + voxel array
    std::vector<ChunkMeta> metas;
    metas.reserve(chunks.size());
    std::vector<DiskVoxel> flatVox;
    {
        size_t total = 0;
        for (auto& kv : chunks) total += kv.second.size();
        flatVox.reserve(total);
    }
    for (auto& kv : chunks) {
        ChunkMeta m;
        m.cx = kv.first.x; m.cy = kv.first.y; m.cz = kv.first.z; m._pad = 0;
        m.voxelCount  = (uint32_t)kv.second.size();
        m.voxelOffset = (uint32_t)flatVox.size();
        metas.push_back(m);
        for (auto& v : kv.second) flatVox.push_back(v);
    }
    uint32_t totalVox = (uint32_t)flatVox.size();

    FILE* f = fopen(out, "wb");
    if (!f) { fprintf(stderr, "open %s failed\n", out); return 1; }

    const char magic[4] = { 'V','X','L','3' };
    fwrite(magic, 1, 4, f);
    uint32_t version = kAssetVersion;
    fwrite(&version, sizeof(uint32_t), 1, f);
    uint32_t cdim = CHUNK_DIM;
    uint32_t ccount = (uint32_t)metas.size();
    fwrite(&cdim,     sizeof(uint32_t), 1, f);
    fwrite(&ccount,   sizeof(uint32_t), 1, f);
    fwrite(&totalVox, sizeof(uint32_t), 1, f);
    int32_t origin[3] = { minX, minY, minZ };
    fwrite(origin, sizeof(int32_t), 3, f);
    float sunDir[3] = { sunDirX, sunDirY, sunDirZ };
    fwrite(sunDir, sizeof(float), 3, f);

    fwrite(metas.data(),   sizeof(ChunkMeta), metas.size(),   f);
    fwrite(flatVox.data(), sizeof(DiskVoxel), flatVox.size(), f);

    long bytes = ftell(f);
    fclose(f);

    printf("Wrote %s: %ld bytes (%.2f MB), %u voxels, %.2f B/voxel total\n",
           out, bytes, bytes / (1024.0 * 1024.0), totalVox,
           (double)bytes / (double)totalVox);

    // ---------------------------------------------------------------------
    // Build a dense occupancy bitmap (for fast O(1) neighbour queries).
    // ---------------------------------------------------------------------
    int32_t spanX = maxX - minX + 1;
    int32_t spanY = maxY - minY + 1;
    int32_t spanZ = maxZ - minZ + 1;
    auto bitIdx = [&](int x, int y, int z) -> size_t {
        return (size_t)((z - minZ) * spanY + (y - minY)) * (size_t)spanX + (size_t)(x - minX);
    };
    std::vector<uint8_t> filled(((size_t)spanX * spanY * spanZ + 7) / 8, 0);
    auto isFilled = [&](int x, int y, int z) -> bool {
        if (x < minX || x > maxX || y < minY || y > maxY || z < minZ || z > maxZ) return false;
        size_t i = bitIdx(x, y, z);
        return (filled[i >> 3] >> (i & 7)) & 1u;
    };
    for (auto& kv : voxels) {
        size_t i = bitIdx(kv.first.x, kv.first.y, kv.first.z);
        filled[i >> 3] |= (uint8_t)(1u << (i & 7));
    }
    std::vector<uint32_t> colorGrid((size_t)spanX * spanY * spanZ, 0);
    for (auto& kv : voxels) {
        colorGrid[bitIdx(kv.first.x, kv.first.y, kv.first.z)] = kv.second.color;
    }

    // ---------------------------------------------------------------------
    // Flood-fill interior cull. Mark every empty cell reachable from outside
    // the AABB. Faces whose adjacent cell isn't reachable are interior.
    // ---------------------------------------------------------------------
    // Pad bounds by 1 so we have guaranteed-exterior seeds on all sides.
    int32_t pMinX = minX - 1, pMaxX = maxX + 1;
    int32_t pMinY = minY - 1, pMaxY = maxY + 1;
    int32_t pMinZ = minZ - 1, pMaxZ = maxZ + 1;
    int32_t pSpanX = pMaxX - pMinX + 1;
    int32_t pSpanY = pMaxY - pMinY + 1;
    int32_t pSpanZ = pMaxZ - pMinZ + 1;
    auto pIdx = [&](int x, int y, int z) -> size_t {
        return (size_t)((z - pMinZ) * pSpanY + (y - pMinY)) * (size_t)pSpanX + (size_t)(x - pMinX);
    };
    auto inPad = [&](int x, int y, int z) {
        return x >= pMinX && x <= pMaxX && y >= pMinY && y <= pMaxY && z >= pMinZ && z <= pMaxZ;
    };
    std::vector<uint8_t> reachable(((size_t)pSpanX * pSpanY * pSpanZ + 7) / 8, 0);
    auto setReach = [&](size_t i) { reachable[i >> 3] |= (uint8_t)(1u << (i & 7)); };
    auto getReach = [&](size_t i) -> bool { return (reachable[i >> 3] >> (i & 7)) & 1u; };

    {
        std::vector<int> stack;
        stack.reserve(1u << 20);
        // Seed: every cell on the padded boundary (must be empty since voxels
        // never sit on the pad border).
        auto seed = [&](int x, int y, int z) {
            size_t i = pIdx(x, y, z);
            if (getReach(i)) return;
            setReach(i);
            stack.push_back(x); stack.push_back(y); stack.push_back(z);
        };
        for (int z = pMinZ; z <= pMaxZ; ++z) {
            for (int y = pMinY; y <= pMaxY; ++y) {
                seed(pMinX, y, z); seed(pMaxX, y, z);
            }
            for (int x = pMinX; x <= pMaxX; ++x) {
                seed(x, pMinY, z); seed(x, pMaxY, z);
            }
        }
        for (int y = pMinY; y <= pMaxY; ++y) {
            for (int x = pMinX; x <= pMaxX; ++x) {
                seed(x, y, pMinZ); seed(x, y, pMaxZ);
            }
        }
        // BFS (using stack) over 6-connected empty cells.
        static const int kDX[6] = {  1, -1,  0,  0,  0,  0 };
        static const int kDY[6] = {  0,  0,  1, -1,  0,  0 };
        static const int kDZ[6] = {  0,  0,  0,  0,  1, -1 };
        while (!stack.empty()) {
            int z = stack.back(); stack.pop_back();
            int y = stack.back(); stack.pop_back();
            int x = stack.back(); stack.pop_back();
            for (int d = 0; d < 6; ++d) {
                int nx = x + kDX[d], ny = y + kDY[d], nz = z + kDZ[d];
                if (!inPad(nx, ny, nz)) continue;
                if (isFilled(nx, ny, nz)) continue;        // blocked by voxel
                size_t ni = pIdx(nx, ny, nz);
                if (getReach(ni)) continue;
                setReach(ni);
                stack.push_back(nx); stack.push_back(ny); stack.push_back(nz);
            }
        }
    }
    auto faceExposed = [&](int x, int y, int z, int fi) -> bool {
        int nx = x + kFaceDelta[fi][0];
        int ny = y + kFaceDelta[fi][1];
        int nz = z + kFaceDelta[fi][2];
        if (!inPad(nx, ny, nz)) return true;
        return getReach(pIdx(nx, ny, nz));
    };

    // Build culled voxels: recompute mask from occupancy + cone-cull.
    std::vector<ChunkMeta> cMetas;
    std::vector<DiskVoxel> cFlatVox;
    cMetas.reserve(chunks.size());
    {
        std::unordered_map<CKey, std::vector<DiskVoxel>, CKeyHash, CKeyEq> cChunks;
        uint64_t culledFaces = 0;
        uint32_t droppedVoxels = 0;
        for (auto& kv : voxels) {
            // Bottom-layer + edge-walls cull (matches uncculled pass).
            if (kv.first.y == minY) { ++droppedVoxels; continue; }
            const bool onEdgeXZ2 = (kv.first.x == minX || kv.first.x == maxX
                                 || kv.first.z == minZ || kv.first.z == maxZ);
            if (onEdgeXZ2) {
                ColKey c{ kv.first.x, kv.first.z };
                auto it = colTopY.find(c);
                if (it != colTopY.end() && kv.first.y != it->second) { ++droppedVoxels; continue; }
            }
            uint8_t mask = 0;
            for (int fi = 0; fi < 6; ++fi) {
                if (faceExposed(kv.first.x, kv.first.y, kv.first.z, fi)) {
                    mask |= (uint8_t)(1u << fi);
                }
            }
            // Drop -Y (downward) face — never seen from above.
            mask &= (uint8_t)~0x08u;
            if (mask == 0) { ++droppedVoxels; continue; }
            culledFaces += __popcnt(mask);
            int32_t rx = kv.first.x - minX;
            int32_t ry = kv.first.y - minY;
            int32_t rz = kv.first.z - minZ;
            CKey ck{ (uint16_t)(rx / CHUNK_DIM), (uint16_t)(ry / CHUNK_DIM), (uint16_t)(rz / CHUNK_DIM) };
            DiskVoxel dv;
            dv.x = (uint8_t)(rx % CHUNK_DIM);
            dv.y = (uint8_t)(ry % CHUNK_DIM);
            dv.z = (uint8_t)(rz % CHUNK_DIM);
            dv.visMask = mask;
            dv.color = kv.second.color;
            for (int fi = 0; fi < 6; ++fi) dv.ao[fi] = kv.second.ao[fi];
            cChunks[ck].push_back(dv);
        }
        for (auto& kv : cChunks) {
            ChunkMeta m;
            m.cx = kv.first.x; m.cy = kv.first.y; m.cz = kv.first.z; m._pad = 0;
            m.voxelCount  = (uint32_t)kv.second.size();
            m.voxelOffset = (uint32_t)cFlatVox.size();
            cMetas.push_back(m);
            for (auto& dv : kv.second) cFlatVox.push_back(dv);
        }
        printf("Cone-cull: dropped %u voxels, remaining faces %llu (was %llu)\n",
               droppedVoxels, (unsigned long long)culledFaces, (unsigned long long)visibleFaces);
    }

    // Output: assets/rungholt_culled.vox
    {
        std::string cOut = std::string(out);
        size_t dot = cOut.find_last_of('.');
        if (dot != std::string::npos) cOut.insert(dot, "_culled");
        else cOut += "_culled.vox";
        FILE* cf = fopen(cOut.c_str(), "wb");
        if (!cf) { fprintf(stderr, "open %s failed\n", cOut.c_str()); return 1; }
        fwrite(magic, 1, 4, cf);
        fwrite(&version, sizeof(uint32_t), 1, cf);
        uint32_t cccount = (uint32_t)cMetas.size();
        uint32_t cTotal  = (uint32_t)cFlatVox.size();
        fwrite(&cdim,    sizeof(uint32_t), 1, cf);
        fwrite(&cccount, sizeof(uint32_t), 1, cf);
        fwrite(&cTotal,  sizeof(uint32_t), 1, cf);
        fwrite(origin,   sizeof(int32_t),  3, cf);
        fwrite(sunDir,   sizeof(float),    3, cf);
        fwrite(cMetas.data(),   sizeof(ChunkMeta), cMetas.size(),   cf);
        fwrite(cFlatVox.data(), sizeof(DiskVoxel), cFlatVox.size(), cf);
        long cb = ftell(cf);
        fclose(cf);
        printf("Wrote %s: %.2f MB, %u voxels\n",
               cOut.c_str(), cb / (1024.0 * 1024.0), cTotal);
    }

    // ---------------------------------------------------------------------
    // Greedy mesh: combine coplanar same-color faces into bigger quads.
    // Output: assets/rungholt_merged.msh
    //   char magic[4] = "MSH1"
    //   uint32 vertCount, indexCount
    //   Vertex[vertCount]: float3 pos + uint32 color
    //   uint32 indices[indexCount]
    // ---------------------------------------------------------------------
    struct MVert { float px, py, pz; uint32_t color; };
    std::vector<MVert> mVerts;
    std::vector<uint32_t> mIndices;

    // For each axis (X/Y/Z), each sign (+/-), sweep slices.
    // axis 0..2, sign 0=+ 1=-. faceIdx (matches kFaceDelta order) = axis*2 + sign.
    for (int axis = 0; axis < 3; ++axis) {
        int uAxis = (axis + 1) % 3;
        int vAxis = (axis + 2) % 3;
        int aMin = (axis == 0) ? minX : (axis == 1) ? minY : minZ;
        int aMax = (axis == 0) ? maxX : (axis == 1) ? maxY : maxZ;
        int uMin = (uAxis == 0) ? minX : (uAxis == 1) ? minY : minZ;
        int uMax = (uAxis == 0) ? maxX : (uAxis == 1) ? maxY : maxZ;
        int vMin = (vAxis == 0) ? minX : (vAxis == 1) ? minY : minZ;
        int vMax = (vAxis == 0) ? maxX : (vAxis == 1) ? maxY : maxZ;
        int spanUu = uMax - uMin + 1;
        int spanVv = vMax - vMin + 1;
        std::vector<uint32_t> mask2D((size_t)spanUu * spanVv);

        for (int sign = 0; sign < 2; ++sign) {
            int fi = axis * 2 + sign;
            float aFaceOffset = (sign == 0) ? 1.0f : 0.0f;
            for (int sliceA = aMin; sliceA <= aMax; ++sliceA) {
                std::fill(mask2D.begin(), mask2D.end(), 0u);
                // Populate this slice's mask from the dense grid.
                for (int vv = 0; vv < spanVv; ++vv) {
                    for (int uu = 0; uu < spanUu; ++uu) {
                        int xyz[3];
                        xyz[axis]  = sliceA;
                        xyz[uAxis] = uu + uMin;
                        xyz[vAxis] = vv + vMin;
                        if (!isFilled(xyz[0], xyz[1], xyz[2])) continue;
                        // Check neighbour along face normal.
                        int nx = xyz[0] + kFaceDelta[fi][0];
                        int ny = xyz[1] + kFaceDelta[fi][1];
                        int nz = xyz[2] + kFaceDelta[fi][2];
                        if (isFilled(nx, ny, nz)) continue;
                        // Strip alpha (per-voxel baked AO) so greedy merge keys
                        // on RGB only — otherwise voxels with identical color
                        // but different AO split into many 1x1 quads. MSH1
                        // doesn't carry AO (no per-vertex storage).
                        uint32_t color = colorGrid[bitIdx(xyz[0], xyz[1], xyz[2])];
                        mask2D[(size_t)uu + (size_t)vv * spanUu] = (color & 0x00FFFFFFu) | 0xFF000000u;
                    }
                }

                // Greedy rectangles.
                for (int vv = 0; vv < spanVv; ++vv) {
                    for (int uu = 0; uu < spanUu; ) {
                        uint32_t c = mask2D[(size_t)uu + (size_t)vv * spanUu];
                        if (c == 0) { ++uu; continue; }
                        // Expand u.
                        int w = 1;
                        while (uu + w < spanUu && mask2D[(size_t)(uu+w) + (size_t)vv * spanUu] == c) ++w;
                        // Expand v.
                        int h = 1;
                        while (vv + h < spanVv) {
                            bool ok = true;
                            for (int j = 0; j < w; ++j) {
                                if (mask2D[(size_t)(uu+j) + (size_t)(vv+h) * spanUu] != c) { ok = false; break; }
                            }
                            if (!ok) break;
                            ++h;
                        }
                        // Emit quad.
                        float p0[3], p1[3], p2[3], p3[3];
                        p0[axis] = p1[axis] = p2[axis] = p3[axis] = (float)sliceA + aFaceOffset;
                        p0[uAxis] = (float)(uMin + uu);     p0[vAxis] = (float)(vMin + vv);
                        p1[uAxis] = (float)(uMin + uu + w); p1[vAxis] = (float)(vMin + vv);
                        p2[uAxis] = (float)(uMin + uu + w); p2[vAxis] = (float)(vMin + vv + h);
                        p3[uAxis] = (float)(uMin + uu);     p3[vAxis] = (float)(vMin + vv + h);
                        uint32_t base = (uint32_t)mVerts.size();
                        mVerts.push_back({ p0[0], p0[1], p0[2], c });
                        mVerts.push_back({ p1[0], p1[1], p1[2], c });
                        mVerts.push_back({ p2[0], p2[1], p2[2], c });
                        mVerts.push_back({ p3[0], p3[1], p3[2], c });
                        // Winding: + sign => cross +A (outward = +A); - sign => -A.
                        if (sign == 0) {
                            mIndices.push_back(base + 0); mIndices.push_back(base + 1); mIndices.push_back(base + 2);
                            mIndices.push_back(base + 0); mIndices.push_back(base + 2); mIndices.push_back(base + 3);
                        } else {
                            mIndices.push_back(base + 0); mIndices.push_back(base + 3); mIndices.push_back(base + 2);
                            mIndices.push_back(base + 0); mIndices.push_back(base + 2); mIndices.push_back(base + 1);
                        }
                        // Clear used cells.
                        for (int dv = 0; dv < h; ++dv) {
                            for (int du = 0; du < w; ++du) {
                                mask2D[(size_t)(uu + du) + (size_t)(vv + dv) * spanUu] = 0;
                            }
                        }
                        uu += w;
                    }
                }
            }
        }
    }

    {
        std::string mOut = std::string(out);
        size_t dot = mOut.find_last_of('.');
        if (dot != std::string::npos) mOut = mOut.substr(0, dot) + "_merged.msh";
        else mOut += "_merged.msh";
        FILE* mf = fopen(mOut.c_str(), "wb");
        if (!mf) { fprintf(stderr, "open %s failed\n", mOut.c_str()); return 1; }
        const char mmagic[4] = { 'M','S','H','1' };
        fwrite(mmagic, 1, 4, mf);
        fwrite(&version, sizeof(uint32_t), 1, mf);
        uint32_t vc = (uint32_t)mVerts.size();
        uint32_t ic = (uint32_t)mIndices.size();
        fwrite(&vc, sizeof(uint32_t), 1, mf);
        fwrite(&ic, sizeof(uint32_t), 1, mf);
        fwrite(mVerts.data(),   sizeof(MVert),    mVerts.size(),   mf);
        fwrite(mIndices.data(), sizeof(uint32_t), mIndices.size(), mf);
        long mb = ftell(mf);
        fclose(mf);
        printf("Wrote %s: %.2f MB, %u verts, %u tris\n",
               mOut.c_str(), mb / (1024.0 * 1024.0), vc, ic / 3);
    }

    // ---------------------------------------------------------------------
    // Atlas mesh: per-chunk binary greedy-merge coplanar visible faces, store
    // per-voxel colors in a global UV-indexed texture atlas. Each chunk emits
    // an indexed submesh so the renderer can frustum-cull.
    // Output: assets/rungholt_atlas.msh (MSH2)
    //   magic[4]="MSH2"
    //   uint32 vertCount, indexCount, atlasW, atlasH, vertexBytes(=12)
    //   int32  origin[3]
    //   uint32 chunkCount
    //   ChunkSub[chunkCount]:
    //     uint16 cx,cy,cz,_pad
    //     int32  aabbMin[3]   (world voxel coords)
    //     int32  aabbMax[3]
    //     uint32 firstIndex, indexCount
    //   AtlasVertex[vertCount] (12 B): uint16 px,py,pz; uint8 face; uint8 _pad; uint16 u,v
    //   uint32 indices[indexCount]
    //   uint32 atlas[atlasW*atlasH]  (RGBA8)
    // ---------------------------------------------------------------------
    #pragma pack(push, 1)
    struct AVert {
        uint16_t px, py, pz;
        uint8_t  face;
        uint8_t  _pad;
        uint16_t u, v;
    };
    struct AChunkSub {
        uint16_t cx, cy, cz, _pad;
        int32_t  aabbMin[3];
        int32_t  aabbMax[3];
        uint32_t firstIndex, indexCount;
    };
    #pragma pack(pop)
    static_assert(sizeof(AVert) == 12, "AVert size");
    static_assert(sizeof(AChunkSub) == 40, "AChunkSub size");

    struct AtlasRect {
        int axis, sign, sliceA;
        int u0, v0, w, h;          // u0/v0 in world voxel coords
        int atlasX, atlasY;
        int chunkIdx;
        std::vector<uint32_t> pixels;
    };

    int32_t spanXm = maxX - minX + 1;
    int32_t spanYm = maxY - minY + 1;
    int32_t spanZm = maxZ - minZ + 1;
    if (spanXm > 65535 || spanYm > 65535 || spanZm > 65535) {
        fprintf(stderr, "Atlas mesh: span exceeds uint16 (%d,%d,%d)\n", spanXm, spanYm, spanZm);
        return 1;
    }

    const int CD = CHUNK_DIM;
    int nCX = (spanXm + CD - 1) / CD;
    int nCY = (spanYm + CD - 1) / CD;
    int nCZ = (spanZm + CD - 1) / CD;

    std::vector<AtlasRect> rects;
    std::vector<AChunkSub> aSubs;
    aSubs.reserve(64);

    for (int cz = 0; cz < nCZ; ++cz)
    for (int cy = 0; cy < nCY; ++cy)
    for (int cx = 0; cx < nCX; ++cx) {
        int chunkMinX = minX + cx * CD;
        int chunkMinY = minY + cy * CD;
        int chunkMinZ = minZ + cz * CD;
        int chunkMaxX = std::min(chunkMinX + CD - 1, maxX);
        int chunkMaxY = std::min(chunkMinY + CD - 1, maxY);
        int chunkMaxZ = std::min(chunkMinZ + CD - 1, maxZ);

        size_t firstRectIdx = rects.size();
        int    abMin[3] = { INT32_MAX, INT32_MAX, INT32_MAX };
        int    abMax[3] = { INT32_MIN, INT32_MIN, INT32_MIN };

        for (int axis = 0; axis < 3; ++axis) {
            int uAxis = (axis + 1) % 3;
            int vAxis = (axis + 2) % 3;
            int aLow  = (axis == 0) ? chunkMinX : (axis == 1) ? chunkMinY : chunkMinZ;
            int aHigh = (axis == 0) ? chunkMaxX : (axis == 1) ? chunkMaxY : chunkMaxZ;
            int uLow  = (uAxis == 0) ? chunkMinX : (uAxis == 1) ? chunkMinY : chunkMinZ;
            int uHigh = (uAxis == 0) ? chunkMaxX : (uAxis == 1) ? chunkMaxY : chunkMaxZ;
            int vLow  = (vAxis == 0) ? chunkMinX : (vAxis == 1) ? chunkMinY : chunkMinZ;
            int vHigh = (vAxis == 0) ? chunkMaxX : (vAxis == 1) ? chunkMaxY : chunkMaxZ;
            int spanU = uHigh - uLow + 1;
            int spanV = vHigh - vLow + 1;
            std::vector<uint8_t>  mask((size_t)spanU * spanV);
            std::vector<uint32_t> col((size_t)spanU * spanV);

            for (int sign = 0; sign < 2; ++sign) {
                int fi = axis * 2 + sign;
                for (int sliceA = aLow; sliceA <= aHigh; ++sliceA) {
                    // Scene viewed from outside its AABB: bottom -Y faces in
                    // the lowest 2 voxel layers are never visible.
                    if (fi == 3 && sliceA <= minY + 1) continue;
                    std::fill(mask.begin(), mask.end(), (uint8_t)0);
                    std::fill(col.begin(), col.end(), 0u);
                    for (int vv = 0; vv < spanV; ++vv) {
                        for (int uu = 0; uu < spanU; ++uu) {
                            int xyz[3];
                            xyz[axis]  = sliceA;
                            xyz[uAxis] = uu + uLow;
                            xyz[vAxis] = vv + vLow;
                            if (!isFilled(xyz[0], xyz[1], xyz[2])) continue;
                            if (!faceExposed(xyz[0], xyz[1], xyz[2], fi)) continue;
                            size_t idx = (size_t)uu + (size_t)vv * spanU;
                            mask[idx] = 1;
                            // Atlas pixel: RGB = voxel color, alpha = this
                            // face's baked AO (looked up per voxel; greedy
                            // merge keys on visibility mask, not color, so
                            // per-pixel AO doesn't hurt merge ratio).
                            uint32_t rgb = colorGrid[bitIdx(xyz[0], xyz[1], xyz[2])] & 0x00FFFFFFu;
                            VKey vk{ xyz[0], xyz[1], xyz[2] };
                            auto it = voxels.find(vk);
                            uint32_t aoByte = (it != voxels.end()) ? (uint32_t)it->second.ao[fi] : 0u;
                            col[idx] = rgb | (aoByte << 24);
                        }
                    }

                    for (int vv = 0; vv < spanV; ++vv) {
                        for (int uu = 0; uu < spanU; ) {
                            if (!mask[(size_t)uu + (size_t)vv * spanU]) { ++uu; continue; }
                            int w = 1;
                            while (uu + w < spanU && mask[(size_t)(uu+w) + (size_t)vv * spanU]) ++w;
                            int h = 1;
                            while (vv + h < spanV) {
                                bool ok = true;
                                for (int j = 0; j < w; ++j) {
                                    if (!mask[(size_t)(uu+j) + (size_t)(vv+h) * spanU]) { ok = false; break; }
                                }
                                if (!ok) break;
                                ++h;
                            }
                            AtlasRect r;
                            r.axis = axis; r.sign = sign; r.sliceA = sliceA;
                            r.u0 = uLow + uu;
                            r.v0 = vLow + vv;
                            r.w = w; r.h = h;
                            r.atlasX = r.atlasY = 0;
                            r.chunkIdx = (int)aSubs.size();
                            r.pixels.resize((size_t)w * h);
                            for (int dv = 0; dv < h; ++dv)
                                for (int du = 0; du < w; ++du)
                                    r.pixels[(size_t)du + (size_t)dv * w]
                                        = col[(size_t)(uu+du) + (size_t)(vv+dv) * spanU];

                            int aFaceOffset = (r.sign == 0) ? 1 : 0;
                            int corners[4][3];
                            corners[0][axis] = corners[1][axis] = corners[2][axis] = corners[3][axis]
                                = r.sliceA + aFaceOffset;
                            corners[0][uAxis] = r.u0;        corners[0][vAxis] = r.v0;
                            corners[1][uAxis] = r.u0 + r.w;  corners[1][vAxis] = r.v0;
                            corners[2][uAxis] = r.u0 + r.w;  corners[2][vAxis] = r.v0 + r.h;
                            corners[3][uAxis] = r.u0;        corners[3][vAxis] = r.v0 + r.h;
                            for (int k = 0; k < 4; ++k) {
                                for (int d = 0; d < 3; ++d) {
                                    if (corners[k][d] < abMin[d]) abMin[d] = corners[k][d];
                                    if (corners[k][d] > abMax[d]) abMax[d] = corners[k][d];
                                }
                            }

                            rects.push_back(std::move(r));
                            for (int dv = 0; dv < h; ++dv)
                                for (int du = 0; du < w; ++du)
                                    mask[(size_t)(uu+du) + (size_t)(vv+dv) * spanU] = 0;
                            uu += w;
                        }
                    }
                }
            }
        }

        if (rects.size() == firstRectIdx) continue;

        AChunkSub sub;
        sub.cx = (uint16_t)cx; sub.cy = (uint16_t)cy; sub.cz = (uint16_t)cz; sub._pad = 0;
        for (int d = 0; d < 3; ++d) { sub.aabbMin[d] = abMin[d]; sub.aabbMax[d] = abMax[d]; }
        sub.firstIndex = 0;
        sub.indexCount = (uint32_t)((rects.size() - firstRectIdx) * 6);
        aSubs.push_back(sub);
    }
    printf("Atlas rects: %zu  chunks: %zu\n", rects.size(), aSubs.size());

    // -----------------------------------------------------------------
    // Stats: what fraction is unmerged 1x1 rects (could go to a point/atlas
    // path) vs genuinely-merged larger quads (worth drawing as triangles)?
    // -----------------------------------------------------------------
    {
        size_t   singletons = 0;
        size_t   merged = 0;
        uint64_t areaSingletons = 0;
        uint64_t areaMerged = 0;
        for (auto& r : rects) {
            uint64_t a = (uint64_t)r.w * (uint64_t)r.h;
            if (r.w == 1 && r.h == 1) {
                ++singletons;
                areaSingletons += a;
            } else {
                ++merged;
                areaMerged += a;
            }
        }
        uint64_t totalArea = areaSingletons + areaMerged;
        printf("[stats] rects: 1x1=%zu (%.1f%% of rects, %.1f%% of face area)\n",
               singletons,
               100.0 * (double)singletons / (double)rects.size(),
               totalArea ? 100.0 * (double)areaSingletons / (double)totalArea : 0.0);
        printf("[stats] rects: merged=%zu (%.1f%% of rects, %.1f%% of face area, avg %.2fx%.2f)\n",
               merged,
               100.0 * (double)merged / (double)rects.size(),
               totalArea ? 100.0 * (double)areaMerged / (double)totalArea : 0.0,
               merged ? (double)areaMerged / (double)merged : 0.0,
               1.0);
    }


    // -----------------------------------------------------------------
    // Experiment: global (cross-chunk) greedy mesh for comparison only.
    // Same binary visibility mask but greedy spans the full scene. Counts
    // rects + tris; does not emit verts/atlas/file.
    // -----------------------------------------------------------------
    {
        size_t globalRects = 0;
        for (int axis = 0; axis < 3; ++axis) {
            int uAxis = (axis + 1) % 3;
            int vAxis = (axis + 2) % 3;
            int aMin = (axis == 0) ? minX : (axis == 1) ? minY : minZ;
            int aMax = (axis == 0) ? maxX : (axis == 1) ? maxY : maxZ;
            int uMinL = (uAxis == 0) ? minX : (uAxis == 1) ? minY : minZ;
            int uMaxL = (uAxis == 0) ? maxX : (uAxis == 1) ? maxY : maxZ;
            int vMinL = (vAxis == 0) ? minX : (vAxis == 1) ? minY : minZ;
            int vMaxL = (vAxis == 0) ? maxX : (vAxis == 1) ? maxY : maxZ;
            int spanU = uMaxL - uMinL + 1;
            int spanV = vMaxL - vMinL + 1;
            std::vector<uint8_t> mask((size_t)spanU * spanV);
            for (int sign = 0; sign < 2; ++sign) {
                int fi = axis * 2 + sign;
                for (int sliceA = aMin; sliceA <= aMax; ++sliceA) {
                    if (fi == 3 && sliceA <= minY + 1) continue;
                    std::fill(mask.begin(), mask.end(), (uint8_t)0);
                    for (int vv = 0; vv < spanV; ++vv) {
                        for (int uu = 0; uu < spanU; ++uu) {
                            int xyz[3];
                            xyz[axis]  = sliceA;
                            xyz[uAxis] = uu + uMinL;
                            xyz[vAxis] = vv + vMinL;
                            if (!isFilled(xyz[0], xyz[1], xyz[2])) continue;
                            if (!faceExposed(xyz[0], xyz[1], xyz[2], fi)) continue;
                            mask[(size_t)uu + (size_t)vv * spanU] = 1;
                        }
                    }
                    for (int vv = 0; vv < spanV; ++vv) {
                        for (int uu = 0; uu < spanU; ) {
                            if (!mask[(size_t)uu + (size_t)vv * spanU]) { ++uu; continue; }
                            int w = 1;
                            while (uu + w < spanU && mask[(size_t)(uu+w) + (size_t)vv * spanU]) ++w;
                            int h = 1;
                            while (vv + h < spanV) {
                                bool ok = true;
                                for (int j = 0; j < w; ++j) {
                                    if (!mask[(size_t)(uu+j) + (size_t)(vv+h) * spanU]) { ok = false; break; }
                                }
                                if (!ok) break;
                                ++h;
                            }
                            ++globalRects;
                            for (int dv = 0; dv < h; ++dv)
                                for (int du = 0; du < w; ++du)
                                    mask[(size_t)(uu+du) + (size_t)(vv+dv) * spanU] = 0;
                            uu += w;
                        }
                    }
                }
            }
        }
        double perChunkTris = (double)rects.size() * 2.0;
        double globalTris   = (double)globalRects * 2.0;
        printf("[experiment] global-greedy atlas: %zu rects, %.0f tris (%.1f%% of per-chunk %.0f)\n",
               globalRects, globalTris, 100.0 * globalTris / perChunkTris, perChunkTris);
    }


    // Pick atlas width so total area packs into ~square. Pad slack ~33%.
    uint64_t totalArea = 0;
    for (auto& r : rects) totalArea += (uint64_t)r.w * r.h;
    uint32_t atlasW = 64;
    while ((uint64_t)atlasW * atlasW < totalArea * 4 / 3) atlasW *= 2;
    atlasW = std::min(atlasW, 16384u);

    // Shelf pack, tallest first.
    std::vector<size_t> order(rects.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (rects[a].h != rects[b].h) return rects[a].h > rects[b].h;
        return rects[a].w > rects[b].w;
    });

    uint32_t shelfY = 0, shelfX = 0, shelfH = 0;
    for (size_t k : order) {
        AtlasRect& r = rects[k];
        if ((uint32_t)r.w > atlasW) atlasW = (uint32_t)r.w;
        if (shelfX + (uint32_t)r.w > atlasW) {
            shelfY += shelfH;
            shelfX = 0;
            shelfH = 0;
        }
        r.atlasX = (int)shelfX;
        r.atlasY = (int)shelfY;
        shelfX += (uint32_t)r.w;
        if ((uint32_t)r.h > shelfH) shelfH = (uint32_t)r.h;
    }
    uint32_t atlasH = shelfY + shelfH;
    atlasH = (atlasH + 3u) & ~3u;
    atlasH = std::max(atlasH, 4u);

    double atlasMB = (double)atlasW * atlasH * 4.0 / (1024.0 * 1024.0);
    printf("Atlas: %u x %u (%.2f MB, fill %.1f%%)\n",
           atlasW, atlasH, atlasMB,
           100.0 * (double)totalArea / ((double)atlasW * atlasH));

    // Blit each rect's pixels into the atlas buffer.
    std::vector<uint32_t> atlas((size_t)atlasW * atlasH, 0xFFFF00FFu);
    for (auto& r : rects) {
        for (int dv = 0; dv < r.h; ++dv) {
            uint32_t* dst = &atlas[(size_t)(r.atlasY + dv) * atlasW + r.atlasX];
            const uint32_t* src = &r.pixels[(size_t)dv * r.w];
            memcpy(dst, src, sizeof(uint32_t) * (size_t)r.w);
        }
    }

    // Emit vertices/indices per chunk so each AChunkSub points at a contiguous
    // index range. Pos stored as uint16 scene-local (origin subtracted).
    std::vector<AVert> aVerts;
    std::vector<uint32_t> aIdx;
    aVerts.reserve(rects.size() * 4);
    aIdx.reserve(rects.size() * 6);

    // Bin rects by chunk index for emission order.
    std::vector<std::vector<size_t>> rectsByChunk(aSubs.size());
    for (size_t i = 0; i < rects.size(); ++i) {
        rectsByChunk[rects[i].chunkIdx].push_back(i);
    }

    for (size_t ci = 0; ci < aSubs.size(); ++ci) {
        aSubs[ci].firstIndex = (uint32_t)aIdx.size();
        for (size_t ri : rectsByChunk[ci]) {
            AtlasRect& r = rects[ri];
            int axis = r.axis;
            int uAxis = (axis + 1) % 3;
            int vAxis = (axis + 2) % 3;
            int aFaceOffset = (r.sign == 0) ? 1 : 0;

            int p0[3], p1[3], p2[3], p3[3];
            p0[axis] = p1[axis] = p2[axis] = p3[axis] = r.sliceA + aFaceOffset;
            p0[uAxis] = r.u0;          p0[vAxis] = r.v0;
            p1[uAxis] = r.u0 + r.w;    p1[vAxis] = r.v0;
            p2[uAxis] = r.u0 + r.w;    p2[vAxis] = r.v0 + r.h;
            p3[uAxis] = r.u0;          p3[vAxis] = r.v0 + r.h;

            int uv0[2] = { r.atlasX,         r.atlasY         };
            int uv1[2] = { r.atlasX + r.w,   r.atlasY         };
            int uv2[2] = { r.atlasX + r.w,   r.atlasY + r.h   };
            int uv3[2] = { r.atlasX,         r.atlasY + r.h   };

            uint8_t face = (uint8_t)(r.axis * 2 + r.sign);
            auto store = [&](const int p[3], const int uv[2]) {
                AVert v;
                v.px = (uint16_t)(p[0] - minX);
                v.py = (uint16_t)(p[1] - minY);
                v.pz = (uint16_t)(p[2] - minZ);
                v.face = face;
                v._pad = 0;
                v.u = (uint16_t)uv[0];
                v.v = (uint16_t)uv[1];
                aVerts.push_back(v);
            };
            uint32_t base = (uint32_t)aVerts.size();
            store(p0, uv0);
            store(p1, uv1);
            store(p2, uv2);
            store(p3, uv3);

            if (r.sign == 0) {
                aIdx.push_back(base + 0); aIdx.push_back(base + 1); aIdx.push_back(base + 2);
                aIdx.push_back(base + 0); aIdx.push_back(base + 2); aIdx.push_back(base + 3);
            } else {
                aIdx.push_back(base + 0); aIdx.push_back(base + 3); aIdx.push_back(base + 2);
                aIdx.push_back(base + 0); aIdx.push_back(base + 2); aIdx.push_back(base + 1);
            }
        }
        aSubs[ci].indexCount = (uint32_t)aIdx.size() - aSubs[ci].firstIndex;
    }
    printf("Atlas mesh: %zu verts, %zu tris  (vertex=%zu B, index=%zu B, atlas=%.2f MB)\n",
           aVerts.size(), aIdx.size() / 3,
           sizeof(AVert) * aVerts.size(),
           sizeof(uint32_t) * aIdx.size(),
           atlasMB);

    {
        std::string aOut = std::string(out);
        size_t dot = aOut.find_last_of('.');
        if (dot != std::string::npos) aOut = aOut.substr(0, dot) + "_atlas.msh";
        else aOut += "_atlas.msh";
        FILE* af = fopen(aOut.c_str(), "wb");
        if (!af) { fprintf(stderr, "open %s failed\n", aOut.c_str()); return 1; }
        const char amagic[4] = { 'M','S','H','2' };
        fwrite(amagic, 1, 4, af);
        fwrite(&version, sizeof(uint32_t), 1, af);
        uint32_t vc = (uint32_t)aVerts.size();
        uint32_t ic = (uint32_t)aIdx.size();
        uint32_t vbytes = (uint32_t)sizeof(AVert);
        int32_t  aorigin[3] = { minX, minY, minZ };
        uint32_t chunkCount = (uint32_t)aSubs.size();
        fwrite(&vc,         sizeof(uint32_t), 1, af);
        fwrite(&ic,         sizeof(uint32_t), 1, af);
        fwrite(&atlasW,     sizeof(uint32_t), 1, af);
        fwrite(&atlasH,     sizeof(uint32_t), 1, af);
        fwrite(&vbytes,     sizeof(uint32_t), 1, af);
        fwrite(aorigin,     sizeof(int32_t),  3, af);
        fwrite(&chunkCount, sizeof(uint32_t), 1, af);
        fwrite(aSubs.data(), sizeof(AChunkSub), aSubs.size(), af);
        fwrite(aVerts.data(), sizeof(AVert),    aVerts.size(), af);
        fwrite(aIdx.data(),   sizeof(uint32_t), aIdx.size(),   af);
        fwrite(atlas.data(),  sizeof(uint32_t), atlas.size(),  af);
        long ab = ftell(af);
        fclose(af);
        printf("Wrote %s: %.2f MB\n", aOut.c_str(), ab / (1024.0 * 1024.0));
    }

    return 0;
}
