#define _CRT_SECURE_NO_WARNINGS
#include "vox_loader.h"
#include "asset_version.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>
#include <unordered_map>

namespace {

#pragma pack(push, 1)
struct DiskVoxel { uint8_t x, y, z; uint8_t visMask; uint32_t color; uint8_t ao[6]; };
#pragma pack(pop)
static_assert(sizeof(DiskVoxel) == 14, "");

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

bool LoadVoxScene(const char* path, Scene& out, std::string& err)
{
    FILE* f = fopen(path, "rb");
    if (!f) { err = "open failed"; return false; }

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "VXL3", 4) != 0) {
        fclose(f); err = "bad magic (need VXL3)"; return false;
    }
    uint32_t version = 0;
    if (fread(&version, sizeof(uint32_t), 1, f) != 1 || version != kAssetVersion) {
        fclose(f); err = "asset version mismatch"; return false;
    }
    uint32_t chunkDim = 0, chunkCount = 0, totalVoxels = 0;
    int32_t origin[3] = { 0, 0, 0 };
    float sunDir[3] = { 0.4f, 0.8f, 0.2f };   // default if missing
    fread(&chunkDim,    sizeof(uint32_t), 1, f);
    fread(&chunkCount,  sizeof(uint32_t), 1, f);
    fread(&totalVoxels, sizeof(uint32_t), 1, f);
    fread(origin,       sizeof(int32_t),  3, f);
    fread(sunDir,       sizeof(float),    3, f);
    out.sunDir[0] = sunDir[0];
    out.sunDir[1] = sunDir[1];
    out.sunDir[2] = sunDir[2];

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
    out.pointAo6.reserve(totalVoxels);
    out.origin[0] = origin[0];
    out.origin[1] = origin[1];
    out.origin[2] = origin[2];

    // -----------------------------------------------------------------
    // 3 passes so all of L0 lives contiguous in pointVertices, then all
    // L1, then all L2. That lets us batch contiguous chunks per LOD.
    // -----------------------------------------------------------------
    std::vector<SubMesh> temp(chunkCount);
    std::vector<uint8_t> chunkValid(chunkCount, 0);

    // Pass 1: poly mesh + L0 points (one per visible voxel).
    for (uint32_t ci = 0; ci < chunkCount; ++ci) {
        const ChunkMeta& m = metas[ci];
        if (m.voxelCount == 0) continue;

        float baseX = (float)(origin[0] + (int32_t)m.cx * D);
        float baseY = (float)(origin[1] + (int32_t)m.cy * D);
        float baseZ = (float)(origin[2] + (int32_t)m.cz * D);

        SubMesh sm;
        sm.firstIndex = (uint32_t)out.indices.size();
        sm.baseVertex = 0;
        sm.pointFirst = (uint32_t)out.pointVertices.size();
        sm.pointCount = 0;
        sm.chunkBase[0] = baseX;
        sm.chunkBase[1] = baseY;
        sm.chunkBase[2] = baseZ;
        sm.aabbMin[0] = sm.aabbMin[1] = sm.aabbMin[2] =  1e30f;
        sm.aabbMax[0] = sm.aabbMax[1] = sm.aabbMax[2] = -1e30f;

        const DiskVoxel* chunkVox = voxels.data() + m.voxelOffset;
        const int sceneBaseX = (int32_t)m.cx * D;
        const int sceneBaseY = (int32_t)m.cy * D;
        const int sceneBaseZ = (int32_t)m.cz * D;

        for (uint32_t i = 0; i < m.voxelCount; ++i) {
            const DiskVoxel& dv = chunkVox[i];
            uint8_t mask = dv.visMask;
            if (mask == 0) continue;

            int srx = sceneBaseX + dv.x;
            int sry = sceneBaseY + dv.y;
            int srz = sceneBaseZ + dv.z;

            // Point vertex AO: brightest visible face (max). Matches the poly
            // path's per-face AO for top-facing surfaces so splats and polys
            // line up visually in SplatHybrid mode.
            // Max-of-visible-faces AO. Matches the L1/L2/L3 cluster aggregation
            // (which sums voxAoMax then averages over the bin), so AO debug
            // view stays consistent across LODs.
            uint8_t pointAo = 0;
            for (int fi = 0; fi < 6; ++fi) {
                if (!((mask >> fi) & 1u)) continue;
                pointAo = std::max(pointAo, dv.ao[fi]);
            }
            out.pointVertices.push_back(MakeVoxVertex(srx, sry, srz, dv.color, dv.visMask, pointAo));
            // Pack per-face AO: 4 bits per face × 6 faces in low 24 bits. Face
            // order matches visMask (0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z).
            uint32_t ao6 = 0;
            for (int fi = 0; fi < 6; ++fi) {
                uint32_t a4 = (uint32_t)(dv.ao[fi] >> 4) & 0xFu;
                ao6 |= (a4 << (fi * 4));
            }
            out.pointAo6.push_back(ao6);
            ++sm.pointCount;

            // Poly: emit 4 verts per visible face (each carries that face's
            // AO in vertex aux). Drops the 8-corner sharing — same vert count
            // overall since each voxel averages ~2 visible faces × 4 = 8.
            for (int fi = 0; fi < 6; ++fi) {
                if (!((mask >> fi) & 1u)) continue;
                uint8_t faceAo = dv.ao[fi];
                uint32_t vBase = (uint32_t)out.vertices.size();
                for (int k = 0; k < 4; ++k) {
                    int c = kFaceCornerIdx[fi][k];
                    int x = srx + kCornerOffset[c][0];
                    int y = sry + kCornerOffset[c][1];
                    int z = srz + kCornerOffset[c][2];
                    out.vertices.push_back(MakeVoxVertex(x, y, z, dv.color, 0xFF, faceAo));

                    float wx = baseX + (float)kCornerOffset[c][0] + (float)dv.x;
                    float wy = baseY + (float)kCornerOffset[c][1] + (float)dv.y;
                    float wz = baseZ + (float)kCornerOffset[c][2] + (float)dv.z;
                    if (wx < sm.aabbMin[0]) sm.aabbMin[0] = wx;
                    if (wy < sm.aabbMin[1]) sm.aabbMin[1] = wy;
                    if (wz < sm.aabbMin[2]) sm.aabbMin[2] = wz;
                    if (wx > sm.aabbMax[0]) sm.aabbMax[0] = wx;
                    if (wy > sm.aabbMax[1]) sm.aabbMax[1] = wy;
                    if (wz > sm.aabbMax[2]) sm.aabbMax[2] = wz;
                }
                out.indices.push_back(vBase + 0);
                out.indices.push_back(vBase + 1);
                out.indices.push_back(vBase + 2);
                out.indices.push_back(vBase + 0);
                out.indices.push_back(vBase + 2);
                out.indices.push_back(vBase + 3);
            }
        }
        sm.indexCount = (uint32_t)out.indices.size() - sm.firstIndex;
        if (sm.indexCount == 0 && sm.pointCount == 0) continue;
        if (sm.aabbMax[0] < sm.aabbMin[0]) {
            sm.aabbMin[0] = baseX; sm.aabbMin[1] = baseY; sm.aabbMin[2] = baseZ;
            sm.aabbMax[0] = baseX + D; sm.aabbMax[1] = baseY + D; sm.aabbMax[2] = baseZ + D;
        }
        temp[ci] = sm;
        chunkValid[ci] = 1;
    }

    // Pass 2 + 3: per-LOD aggregates appended contiguously.
    struct Accum { uint32_t r = 0, g = 0, b = 0, ao = 0; uint16_t count = 0; uint8_t mask = 0; };
    auto lodPass = [&](int step, uint32_t SubMesh::*offsetField, uint32_t SubMesh::*countField) {
        for (uint32_t ci = 0; ci < chunkCount; ++ci) {
            if (!chunkValid[ci]) continue;
            const ChunkMeta& m = metas[ci];
            const DiskVoxel* chunkVox = voxels.data() + m.voxelOffset;
            const int sceneBaseX = (int32_t)m.cx * D;
            const int sceneBaseY = (int32_t)m.cy * D;
            const int sceneBaseZ = (int32_t)m.cz * D;
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
                // Per-voxel AO = brightest visible face (max), then averaged
                // across the LOD bin. Keeps points/splats bright on open-top
                // surfaces like the poly path.
                uint32_t voxAoMax = 0;
                for (int fi = 0; fi < 6; ++fi) {
                    if (!((dv.visMask >> fi) & 1u)) continue;
                    voxAoMax = std::max(voxAoMax, (uint32_t)dv.ao[fi]);
                }
                a.ao += voxAoMax;
                a.count++;
                a.mask |= dv.visMask;
            }
            temp[ci].*offsetField = (uint32_t)out.pointVertices.size();
            temp[ci].*countField  = (uint32_t)bins.size();
            for (auto& kv : bins) {
                uint32_t key = kv.first;
                const Accum& a = kv.second;
                int srx = sceneBaseX + (int)(((key >>  0) & 0xFF) * step);
                int sry = sceneBaseY + (int)(((key >>  8) & 0xFF) * step);
                int srz = sceneBaseZ + (int)(((key >> 16) & 0xFF) * step);
                uint32_t r = a.r / a.count;
                uint32_t g = a.g / a.count;
                uint32_t b = a.b / a.count;
                uint8_t  ao = (uint8_t)(a.ao / a.count);
                out.pointVertices.push_back(
                    MakeVoxVertex(srx, sry, srz, r | (g << 8) | (b << 16), a.mask, ao));
                // L1/L2 use the same averaged AO on all faces so the SB stays
                // aligned with pointVertices. PolyAxis only consumes L0 in
                // practice; this keeps indexing safe.
                uint32_t a4 = (uint32_t)(ao >> 4) & 0xFu;
                uint32_t ao6 = 0;
                for (int fi = 0; fi < 6; ++fi) ao6 |= (a4 << (fi * 4));
                out.pointAo6.push_back(ao6);
            }
        }
    };
    lodPass(2, &SubMesh::pointFirstL1, &SubMesh::pointCountL1);
    lodPass(4, &SubMesh::pointFirstL2, &SubMesh::pointCountL2);
    lodPass(8, &SubMesh::pointFirstL3, &SubMesh::pointCountL3);

    // Commit valid subs in chunk order (matches subs_ order in renderer).
    for (uint32_t ci = 0; ci < chunkCount; ++ci) {
        if (!chunkValid[ci]) continue;
        const SubMesh& sm = temp[ci];
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
