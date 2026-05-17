#define _CRT_SECURE_NO_WARNINGS
#include "vox_loader.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <unordered_map>

namespace {

#pragma pack(push, 1)
struct DiskVoxel { uint8_t x, y, z; uint8_t visMask; uint32_t color; };
#pragma pack(pop)
static_assert(sizeof(DiskVoxel) == 8, "");

struct ChunkMeta {
    uint16_t cx, cy, cz, _pad;
    uint32_t voxelCount;
    uint32_t voxelOffset;
};
static_assert(sizeof(ChunkMeta) == 16, "");

// 8 cube corners. Index bits: bit0=x, bit1=y, bit2=z.
static const uint8_t kCornerOffset[8][3] = {
    {0,0,0}, {1,0,0}, {0,1,0}, {1,1,0},
    {0,0,1}, {1,0,1}, {0,1,1}, {1,1,1},
};

// Per face: 4 corner indices into the 8-corner cube, in winding order so
// that triangles (c0,c1,c2)(c0,c2,c3) have outward cross product.
// Order: 0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z (matches visMask bits)
static const uint8_t kFaceCornerIdx[6][4] = {
    { 1, 3, 7, 5 },   // +X
    { 4, 6, 2, 0 },   // -X
    { 2, 6, 7, 3 },   // +Y
    { 0, 1, 5, 4 },   // -Y
    { 5, 7, 6, 4 },   // +Z
    { 0, 2, 3, 1 },   // -Z
};

} // namespace

bool LoadVoxScene(const char* path, Scene& out, std::string& err) {
    FILE* f = fopen(path, "rb");
    if (!f) { err = "open failed"; return false; }

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "VXL3", 4) != 0) {
        fclose(f); err = "bad magic (need VXL3)"; return false;
    }
    uint32_t chunkDim = 0, chunkCount = 0, totalVoxels = 0;
    int32_t origin[3] = { 0, 0, 0 };
    fread(&chunkDim,    sizeof(uint32_t), 1, f);
    fread(&chunkCount,  sizeof(uint32_t), 1, f);
    fread(&totalVoxels, sizeof(uint32_t), 1, f);
    fread(origin,       sizeof(int32_t),  3, f);

    std::vector<ChunkMeta> metas(chunkCount);
    if (chunkCount) fread(metas.data(), sizeof(ChunkMeta), chunkCount, f);

    std::vector<DiskVoxel> voxels(totalVoxels);
    if (totalVoxels) fread(voxels.data(), sizeof(DiskVoxel), totalVoxels, f);
    fclose(f);

    const int D = (int)chunkDim;
    if (D <= 0 || D > 256) { err = "bad chunkDim"; return false; }

    out.vertices.clear();
    out.indices.clear();
    out.pointVertices.clear();
    out.subs.clear();
    out.aabbMin[0] = out.aabbMin[1] = out.aabbMin[2] =  1e30f;
    out.aabbMax[0] = out.aabbMax[1] = out.aabbMax[2] = -1e30f;
    out.totalVertices = 0;
    out.totalTriangles = 0;
    out.subs.reserve(chunkCount);
    out.vertices.reserve((size_t)totalVoxels * 8);
    out.indices.reserve((size_t)totalVoxels * 12);
    out.pointVertices.reserve(totalVoxels);

    for (uint32_t ci = 0; ci < chunkCount; ++ci) {
        const ChunkMeta& m = metas[ci];
        if (m.voxelCount == 0) continue;

        float baseX = (float)(origin[0] + (int32_t)m.cx * D);
        float baseY = (float)(origin[1] + (int32_t)m.cy * D);
        float baseZ = (float)(origin[2] + (int32_t)m.cz * D);

        SubMesh sm;
        sm.firstIndex = (uint32_t)out.indices.size();
        sm.baseVertex = (int32_t)out.vertices.size();
        sm.pointFirst = (uint32_t)out.pointVertices.size();
        sm.pointCount = 0;   // counted as we emit (excludes mask==0 voxels)
        sm.chunkBase[0] = baseX;
        sm.chunkBase[1] = baseY;
        sm.chunkBase[2] = baseZ;
        sm.aabbMin[0] = sm.aabbMin[1] = sm.aabbMin[2] =  1e30f;
        sm.aabbMax[0] = sm.aabbMax[1] = sm.aabbMax[2] = -1e30f;

        const DiskVoxel* chunkVox = voxels.data() + m.voxelOffset;
        uint32_t localVertCursor = 0;  // verts written so far in this chunk

        for (uint32_t i = 0; i < m.voxelCount; ++i) {
            const DiskVoxel& dv = chunkVox[i];
            uint8_t mask = dv.visMask;
            if (mask == 0) continue;     // invisible cube; skip points + faces

            // Point vertex (one per visible voxel). faceIdx field carries
            // the 6-bit visMask (used for Points-tech lighting + PolyVID align).
            {
                Vertex pv;
                pv.x = dv.x; pv.y = dv.y; pv.z = dv.z;
                pv.faceIdx = dv.visMask;
                pv.color = dv.color;
                out.pointVertices.push_back(pv);
            }
            ++sm.pointCount;

            // Emit all 8 cube corners for this voxel (shared across visible faces).
            uint32_t vBase = localVertCursor;
            for (int c = 0; c < 8; ++c) {
                Vertex v;
                v.x = (uint8_t)(dv.x + kCornerOffset[c][0]);
                v.y = (uint8_t)(dv.y + kCornerOffset[c][1]);
                v.z = (uint8_t)(dv.z + kCornerOffset[c][2]);
                v.faceIdx = 0;            // unused; PS computes normal via ddx/ddy
                v.color = dv.color;
                out.vertices.push_back(v);

                float wx = baseX + v.x;
                float wy = baseY + v.y;
                float wz = baseZ + v.z;
                if (wx < sm.aabbMin[0]) sm.aabbMin[0] = wx;
                if (wy < sm.aabbMin[1]) sm.aabbMin[1] = wy;
                if (wz < sm.aabbMin[2]) sm.aabbMin[2] = wz;
                if (wx > sm.aabbMax[0]) sm.aabbMax[0] = wx;
                if (wy > sm.aabbMax[1]) sm.aabbMax[1] = wy;
                if (wz > sm.aabbMax[2]) sm.aabbMax[2] = wz;
            }
            localVertCursor += 8;

            // Emit indices for each visible face (6 indices per face).
            for (int fi = 0; fi < 6; ++fi) {
                if (!((mask >> fi) & 1u)) continue;
                uint32_t c0 = vBase + kFaceCornerIdx[fi][0];
                uint32_t c1 = vBase + kFaceCornerIdx[fi][1];
                uint32_t c2 = vBase + kFaceCornerIdx[fi][2];
                uint32_t c3 = vBase + kFaceCornerIdx[fi][3];
                out.indices.push_back(c0);
                out.indices.push_back(c1);
                out.indices.push_back(c2);
                out.indices.push_back(c0);
                out.indices.push_back(c2);
                out.indices.push_back(c3);
            }
        }
        sm.indexCount = (uint32_t)out.indices.size() - sm.firstIndex;

        // Build LOD 1 (2x2x2 group) and LOD 2 (4x4x4 group) point clouds.
        // Average color, OR visMask across child voxels.
        struct Accum { uint32_t r = 0, g = 0, b = 0; uint16_t count = 0; uint8_t mask = 0; };
        auto buildLod = [&](int step, uint32_t& outFirst, uint32_t& outCount) {
            std::unordered_map<uint32_t, Accum> bins;
            bins.reserve(m.voxelCount);
            for (uint32_t i = 0; i < m.voxelCount; ++i) {
                const DiskVoxel& dv = chunkVox[i];
                uint32_t sx = dv.x / step;
                uint32_t sy = dv.y / step;
                uint32_t sz = dv.z / step;
                uint32_t key = sx | (sy << 8) | (sz << 16);
                Accum& a = bins[key];
                a.r += (dv.color >>  0) & 0xFF;
                a.g += (dv.color >>  8) & 0xFF;
                a.b += (dv.color >> 16) & 0xFF;
                a.count++;
                a.mask |= dv.visMask;
            }
            outFirst = (uint32_t)out.pointVertices.size();
            outCount = (uint32_t)bins.size();
            for (auto& kv : bins) {
                uint32_t key = kv.first;
                const Accum& a = kv.second;
                Vertex pv;
                pv.x = (uint8_t)(((key >>  0) & 0xFF) * step);
                pv.y = (uint8_t)(((key >>  8) & 0xFF) * step);
                pv.z = (uint8_t)(((key >> 16) & 0xFF) * step);
                pv.faceIdx = a.mask;
                uint32_t r = a.r / a.count;
                uint32_t g = a.g / a.count;
                uint32_t b = a.b / a.count;
                pv.color = r | (g << 8) | (b << 16) | (255u << 24);
                out.pointVertices.push_back(pv);
            }
        };
        buildLod(2, sm.pointFirstL1, sm.pointCountL1);
        buildLod(4, sm.pointFirstL2, sm.pointCountL2);

        if (sm.indexCount == 0 && sm.pointCount == 0) continue;
        if (sm.aabbMax[0] < sm.aabbMin[0]) {
            // no triangle verts emitted; fall back to point-derived AABB
            sm.aabbMin[0] = baseX; sm.aabbMin[1] = baseY; sm.aabbMin[2] = baseZ;
            sm.aabbMax[0] = baseX + D; sm.aabbMax[1] = baseY + D; sm.aabbMax[2] = baseZ + D;
        }
        if (sm.aabbMin[0] < out.aabbMin[0]) out.aabbMin[0] = sm.aabbMin[0];
        if (sm.aabbMin[1] < out.aabbMin[1]) out.aabbMin[1] = sm.aabbMin[1];
        if (sm.aabbMin[2] < out.aabbMin[2]) out.aabbMin[2] = sm.aabbMin[2];
        if (sm.aabbMax[0] > out.aabbMax[0]) out.aabbMax[0] = sm.aabbMax[0];
        if (sm.aabbMax[1] > out.aabbMax[1]) out.aabbMax[1] = sm.aabbMax[1];
        if (sm.aabbMax[2] > out.aabbMax[2]) out.aabbMax[2] = sm.aabbMax[2];
        out.subs.push_back(sm);
    }

    out.totalVertices  = out.vertices.size();
    out.totalTriangles = out.indices.size() / 3;
    return true;
}
