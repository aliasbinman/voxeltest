#define _CRT_SECURE_NO_WARNINGS
#include "obj_loader.h"
#include "obj_scene.h"

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <climits>
#include <string>
#include <vector>
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
struct DiskVoxel { uint8_t x, y, z; uint8_t visMask; uint32_t color; };
#pragma pack(pop)
static_assert(sizeof(DiskVoxel) == 8, "");

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
        if (v < 0.0f) v = 0.0f; if (v > 1.0f) v = 1.0f;
        return (uint32_t)(v * 255.0f + 0.5f);
    };
    return q(r) | (q(g) << 8) | (q(b) << 16) | (255u << 24);
}

int main(int argc, char** argv) {
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

    std::unordered_map<VKey, uint32_t, VKeyHash> voxels;
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
            voxels.emplace(k, a.color);
        }
    }
    printf("Unique voxels: %zu\n", voxels.size());
    if (voxels.empty()) { fprintf(stderr, "No voxels produced\n"); return 1; }

    int32_t minX = INT32_MAX, minY = INT32_MAX, minZ = INT32_MAX;
    int32_t maxX = INT32_MIN, maxY = INT32_MIN, maxZ = INT32_MIN;
    for (auto& kv : voxels) {
        if (kv.first.x < minX) minX = kv.first.x;
        if (kv.first.y < minY) minY = kv.first.y;
        if (kv.first.z < minZ) minZ = kv.first.z;
        if (kv.first.x > maxX) maxX = kv.first.x;
        if (kv.first.y > maxY) maxY = kv.first.y;
        if (kv.first.z > maxZ) maxZ = kv.first.z;
    }
    printf("Voxel AABB: (%d,%d,%d) .. (%d,%d,%d)\n", minX, minY, minZ, maxX, maxY, maxZ);

    struct CKey { uint16_t x, y, z; };
    struct CKeyHash {
        size_t operator()(const CKey& k) const noexcept {
            return ((size_t)k.x * 73856093u) ^ ((size_t)k.y * 19349663u) ^ ((size_t)k.z * 83492791u);
        }
    };
    struct CKeyEq { bool operator()(const CKey& a, const CKey& b) const { return a.x == b.x && a.y == b.y && a.z == b.z; } };

    std::unordered_map<CKey, std::vector<DiskVoxel>, CKeyHash, CKeyEq> chunks;
    uint64_t visibleFaces = 0;
    for (auto& kv : voxels) {
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
        visibleFaces += __popcnt(mask);

        DiskVoxel v;
        v.x = (uint8_t)(rx % CHUNK_DIM);
        v.y = (uint8_t)(ry % CHUNK_DIM);
        v.z = (uint8_t)(rz % CHUNK_DIM);
        v.visMask = mask;
        v.color = kv.second;
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
    uint32_t cdim = CHUNK_DIM;
    uint32_t ccount = (uint32_t)metas.size();
    fwrite(&cdim,     sizeof(uint32_t), 1, f);
    fwrite(&ccount,   sizeof(uint32_t), 1, f);
    fwrite(&totalVox, sizeof(uint32_t), 1, f);
    int32_t origin[3] = { minX, minY, minZ };
    fwrite(origin, sizeof(int32_t), 3, f);

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
        colorGrid[bitIdx(kv.first.x, kv.first.y, kv.first.z)] = kv.second;
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
            uint8_t mask = 0;
            for (int fi = 0; fi < 6; ++fi) {
                if (faceExposed(kv.first.x, kv.first.y, kv.first.z, fi)) {
                    mask |= (uint8_t)(1u << fi);
                }
            }
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
            dv.color = kv.second;
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
        uint32_t cccount = (uint32_t)cMetas.size();
        uint32_t cTotal  = (uint32_t)cFlatVox.size();
        fwrite(&cdim,    sizeof(uint32_t), 1, cf);
        fwrite(&cccount, sizeof(uint32_t), 1, cf);
        fwrite(&cTotal,  sizeof(uint32_t), 1, cf);
        fwrite(origin,   sizeof(int32_t),  3, cf);
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
                        uint32_t color = colorGrid[bitIdx(xyz[0], xyz[1], xyz[2])];
                        // Avoid zero colour collision: encode "present" via low bit
                        // of alpha. Our packed color always has A=255 so col!=0.
                        mask2D[(size_t)uu + (size_t)vv * spanUu] = color;
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

    return 0;
}
