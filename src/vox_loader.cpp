#define _CRT_SECURE_NO_WARNINGS
#include "vox_loader.h"
#include "asset_version.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>
#include <unordered_map>

#include "lz4.h"

namespace {

#pragma pack(push, 1)
// AO stored as 4-bit-per-face (6 faces = 24 bits = 3 bytes packed).
// Face index = visMask bit order: 0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z.
// Color stored as 16-bit palette index; palette table sits at file start.
struct DiskVoxel { uint8_t x, y, z; uint8_t visMask; uint16_t paletteIdx; uint8_t aoPacked[3]; };
#pragma pack(pop)
static_assert(sizeof(DiskVoxel) == 9, "");

inline uint8_t DvAo(const DiskVoxel& v, int fi)
{
    uint8_t n = (v.aoPacked[(fi * 4) >> 3] >> ((fi * 4) & 7)) & 0xFu;
    return (uint8_t)((n << 4) | n); // replicate nibble -> 0..255 in 16 steps
}

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

    uint32_t paletteCount = 0;
    if (fread(&paletteCount, sizeof(uint32_t), 1, f) != 1) {
        fclose(f); err = "missing palette header"; return false;
    }
    std::vector<uint32_t> palette(paletteCount);
    if (paletteCount) fread(palette.data(), sizeof(uint32_t), paletteCount, f);

    std::vector<ChunkMeta> metas(chunkCount);
    if (chunkCount) fread(metas.data(), sizeof(ChunkMeta), chunkCount, f);

    std::vector<DiskVoxel> voxels(totalVoxels);
    if (totalVoxels) fread(voxels.data(), sizeof(DiskVoxel), totalVoxels, f);
    fclose(f);

    auto DvColor = [&](const DiskVoxel& v) -> uint32_t {
        return v.paletteIdx < palette.size() ? palette[v.paletteIdx] : 0xFF000000u;
    };

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
                pointAo = std::max(pointAo, DvAo(dv, fi));
            }
            const uint32_t dvCol = DvColor(dv);
            out.pointVertices.push_back(MakeVoxVertex(srx, sry, srz, dvCol, dv.visMask, pointAo));
            // Pack per-face AO: 4 bits per face × 6 faces in low 24 bits. Face
            // order matches visMask (0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z).
            uint32_t ao6 = 0;
            for (int fi = 0; fi < 6; ++fi) {
                uint32_t a4 = (uint32_t)(DvAo(dv, fi) >> 4) & 0xFu;
                ao6 |= (a4 << (fi * 4));
            }
            out.pointAo6.push_back(ao6);
            ++sm.pointCount;

            // Poly: emit 4 verts per visible face (each carries that face's
            // AO in vertex aux). Drops the 8-corner sharing — same vert count
            // overall since each voxel averages ~2 visible faces × 4 = 8.
            for (int fi = 0; fi < 6; ++fi) {
                if (!((mask >> fi) & 1u)) continue;
                uint8_t faceAo = DvAo(dv, fi);
                uint32_t vBase = (uint32_t)out.vertices.size();
                for (int k = 0; k < 4; ++k) {
                    int c = kFaceCornerIdx[fi][k];
                    int x = srx + kCornerOffset[c][0];
                    int y = sry + kCornerOffset[c][1];
                    int z = srz + kCornerOffset[c][2];
                    out.vertices.push_back(MakeVoxVertex(x, y, z, dvCol, 0xFF, faceAo));

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
                const uint32_t dvCol = DvColor(dv);
                a.r += (dvCol >>  0) & 0xFF;
                a.g += (dvCol >>  8) & 0xFF;
                a.b += (dvCol >> 16) & 0xFF;
                // Per-voxel AO = brightest visible face (max), then averaged
                // across the LOD bin. Keeps points/splats bright on open-top
                // surfaces like the poly path.
                uint32_t voxAoMax = 0;
                for (int fi = 0; fi < 6; ++fi) {
                    if (!((dv.visMask >> fi) & 1u)) continue;
                    voxAoMax = std::max(voxAoMax, (uint32_t)DvAo(dv, fi));
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

    {
        uint64_t l0 = 0, l1 = 0, l2 = 0, l3 = 0;
        for (const auto& s : out.subs) {
            l0 += s.pointCount;
            l1 += s.pointCountL1;
            l2 += s.pointCountL2;
            l3 += s.pointCountL3;
        }
        const uint64_t total = l0 + l1 + l2 + l3;
        const uint64_t bytesPerPoint = sizeof(Vertex) + sizeof(uint32_t); // pointVertices + pointAo6
        auto mb = [](uint64_t b) { return (double)b / (1024.0 * 1024.0); };
        std::printf("[pointcloud] L0: %10llu pts  %8.2f MB\n", (unsigned long long)l0, mb(l0 * bytesPerPoint));
        std::printf("[pointcloud] L1: %10llu pts  %8.2f MB\n", (unsigned long long)l1, mb(l1 * bytesPerPoint));
        std::printf("[pointcloud] L2: %10llu pts  %8.2f MB\n", (unsigned long long)l2, mb(l2 * bytesPerPoint));
        std::printf("[pointcloud] L3: %10llu pts  %8.2f MB\n", (unsigned long long)l3, mb(l3 * bytesPerPoint));
        std::printf("[pointcloud] TOTAL: %llu pts  %.2f MB (vtx %zu + ao %zu actual = %.2f MB)\n",
                    (unsigned long long)total, mb(total * bytesPerPoint),
                    out.pointVertices.size(), out.pointAo6.size(),
                    mb(out.pointVertices.size() * sizeof(Vertex) + out.pointAo6.size() * sizeof(uint32_t)));
    }

    {
        std::unordered_map<uint32_t, uint64_t> hist;
        hist.reserve(4096);
        for (const auto& s : out.subs) {
            const uint32_t end = s.pointFirst + s.pointCount; // L0 only: 1 per voxel
            for (uint32_t i = s.pointFirst; i < end; ++i) {
                const uint32_t rgb = out.pointVertices[i].color & 0x00FFFFFFu;
                ++hist[rgb];
            }
        }
        out.colorHistogram.assign(hist.begin(), hist.end());
        std::sort(out.colorHistogram.begin(), out.colorHistogram.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        std::printf("[colorhist] %zu unique colors (L0 voxels)\n", out.colorHistogram.size());
        for (const auto& p : out.colorHistogram) {
            const uint32_t c = p.first;
            std::printf("  #%02X%02X%02X  %10llu\n",
                        c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF,
                        (unsigned long long)p.second);
        }
    }

    // -----------------------------------------------------------------
    // Disk-compression simulation (L0 only).
    // Streams (interleaved at load time, separated on disk):
    //   pos:      ceil(log2(chunkDim)) bits/axis * 3 = posBits per voxel
    //   visMask:  6 bits
    //   AO6:      24 bits (4 bits * 6 faces)
    //   color:    8-bit palette idx OR Huffman over palette idx
    // Plus per-chunk header (cx,cy,cz,count = 8 B) and palette table (n*3 B).
    // -----------------------------------------------------------------
    {
        // True L0 count (one entry per visible voxel). pointVertices includes
        // L1/L2/L3 aggregates too — exclude those: LODs are re-derivable from
        // L0 on load so they don't need to live on disk.
        uint64_t L0 = 0;
        for (const auto& s : out.subs) L0 += s.pointCount;
        const uint32_t posBitsPerAxis = (D <= 1) ? 1u
            : (uint32_t)std::ceil(std::log2((double)D));
        const uint32_t posBits  = posBitsPerAxis * 3u;
        const uint32_t maskBits = 6u;
        const uint32_t aoBits   = 24u;
        const uint32_t palIdxBits = 8u; // assumes <=256 unique; otherwise raise

        out.compChunkDim       = (uint32_t)D;
        out.compPosBitsPerAxis = posBitsPerAxis;
        out.compRawBytes       = L0 * (sizeof(Vertex) + sizeof(uint32_t));
        out.compPaletteBytes   = (uint64_t)out.colorHistogram.size() * 3ull;
        out.compPosBytes       = (L0 * posBits + 7) / 8;
        out.compMaskBytes      = (L0 * maskBits + 7) / 8;
        out.compAoBytes        = (L0 * aoBits + 7) / 8;
        out.compColorPalIdxBytes = (L0 * palIdxBits + 7) / 8;

        // Exact Huffman over color symbols (weights = histogram counts).
        double huffBitsPerSym = 0.0;
        const size_t n = out.colorHistogram.size();
        if (n == 1) {
            huffBitsPerSym = 1.0;
        } else if (n >= 2) {
            std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<uint64_t>> pq;
            for (const auto& p : out.colorHistogram) pq.push(p.second);
            uint64_t totalCodeLen = 0;
            while (pq.size() > 1) {
                uint64_t a = pq.top(); pq.pop();
                uint64_t b = pq.top(); pq.pop();
                uint64_t s = a + b;
                totalCodeLen += s;
                pq.push(s);
            }
            huffBitsPerSym = (double)totalCodeLen / (double)L0;
        }
        // Total color bits = huffBitsPerSym * L0. Plus code-table cost
        // (canonical Huffman: ~n bytes for code lengths). Negligible vs payload.
        const uint64_t huffPayloadBits = (uint64_t)std::ceil(huffBitsPerSym * (double)L0);
        out.compColorHuffBytes = (huffPayloadBits + 7) / 8 + n; // +n for code-len table

        auto mb = [](uint64_t b) { return (double)b / (1024.0 * 1024.0); };
        const uint64_t fixedStreams = out.compPosBytes + out.compMaskBytes + out.compAoBytes;
        const uint64_t totalPal  = fixedStreams + out.compColorPalIdxBytes + out.compPaletteBytes;
        const uint64_t totalHuff = fixedStreams + out.compColorHuffBytes   + out.compPaletteBytes;

        std::printf("[compress] L0=%llu voxels, chunkDim=%d, posBits/axis=%u\n",
                    (unsigned long long)L0, D, posBitsPerAxis);
        std::printf("[compress]   raw (in-mem):      %8.2f MB  (%u B/voxel)\n",
                    mb(out.compRawBytes), (unsigned)(sizeof(Vertex) + sizeof(uint32_t)));
        std::printf("[compress]   pos stream:       %8.2f MB  (%u bits/voxel)\n", mb(out.compPosBytes), posBits);
        std::printf("[compress]   visMask stream:   %8.2f MB  (%u bits/voxel)\n", mb(out.compMaskBytes), maskBits);
        std::printf("[compress]   AO stream:        %8.2f MB  (%u bits/voxel)\n", mb(out.compAoBytes), aoBits);
        std::printf("[compress]   color (pal8):     %8.2f MB  (8 bits/voxel) + palette %llu B\n",
                    mb(out.compColorPalIdxBytes), (unsigned long long)out.compPaletteBytes);
        std::printf("[compress]   color (huffman):  %8.2f MB  (%.3f bits/voxel avg) + palette %llu B\n",
                    mb(out.compColorHuffBytes), huffBitsPerSym, (unsigned long long)out.compPaletteBytes);
        std::printf("[compress]   TOTAL pal8:       %8.2f MB  (ratio %.2fx)\n",
                    mb(totalPal), out.compRawBytes ? (double)out.compRawBytes / (double)totalPal : 0.0);
        std::printf("[compress]   TOTAL huffman:    %8.2f MB  (ratio %.2fx)\n",
                    mb(totalHuff), out.compRawBytes ? (double)out.compRawBytes / (double)totalHuff : 0.0);
    }

    // -----------------------------------------------------------------
    // Sub-cluster position scheme + LZ4 over per-chunk binary blobs.
    // Build palette map from colorHistogram (palette index = rank).
    // -----------------------------------------------------------------
    {
        std::unordered_map<uint32_t, uint32_t> palMap;
        palMap.reserve(out.colorHistogram.size() * 2);
        for (uint32_t i = 0; i < out.colorHistogram.size(); ++i)
            palMap[out.colorHistogram[i].first] = i;

        const uint32_t posBitsPerAxis = (D <= 1) ? 1u
            : (uint32_t)std::ceil(std::log2((double)D));

        // Sub-cluster S: pick largest power-of-two <= D/4 (at least 2).
        uint32_t S = 4;
        while (S * 2 <= (uint32_t)D / 2) S *= 2;
        if (S < 2) S = 2;
        const uint32_t cellsPerAxis = (uint32_t)D / S;          // assumes D % S == 0
        const uint32_t cellIdxBits  = (cellsPerAxis <= 1) ? 1u
            : (uint32_t)std::ceil(std::log2((double)cellsPerAxis)) * 3u;
        const uint32_t intraBits    = (S <= 1) ? 3u
            : (uint32_t)std::ceil(std::log2((double)S)) * 3u;
        // Per-cell header: cellIdx + uint16 count.
        const uint32_t cellHdrBits  = cellIdxBits + 16u;

        out.compSubclusterDim = S;

        // LZ4 stream-per-chunk: accumulate compressed sizes.
        uint64_t lz4Pos = 0, lz4Mask = 0, lz4Ao = 0, lz4Pal = 0;
        uint64_t subPosBits = 0;

        // Reusable scratch buffers.
        std::vector<uint8_t> posBlob, maskBlob, aoBlob, palBlob, compScratch;

        auto packStream = [](std::vector<uint8_t>& dst, uint32_t bitsPerSym,
                             const std::vector<uint32_t>& syms) {
            const uint64_t totalBits = (uint64_t)bitsPerSym * syms.size();
            dst.assign((totalBits + 7) / 8, 0);
            uint64_t bitPos = 0;
            for (uint32_t v : syms) {
                for (uint32_t b = 0; b < bitsPerSym; ++b) {
                    if ((v >> b) & 1u) dst[bitPos >> 3] |= (uint8_t)(1u << (bitPos & 7));
                    ++bitPos;
                }
            }
        };

        auto lz4Compress = [&](const std::vector<uint8_t>& src) -> int {
            if (src.empty()) return 0;
            const int bound = LZ4_compressBound((int)src.size());
            if (bound <= 0) return 0;
            if ((int)compScratch.size() < bound) compScratch.resize((size_t)bound);
            return LZ4_compress_default((const char*)src.data(), (char*)compScratch.data(),
                                        (int)src.size(), bound);
        };

        for (uint32_t ci = 0; ci < chunkCount; ++ci) {
            const ChunkMeta& m = metas[ci];
            if (m.voxelCount == 0) continue;
            const DiskVoxel* cv = voxels.data() + m.voxelOffset;

            // Collect symbol streams (visible voxels only — match L0 pipeline).
            std::vector<uint32_t> posSyms, maskSyms, aoSyms, palSyms;
            posSyms.reserve(m.voxelCount);
            maskSyms.reserve(m.voxelCount);
            aoSyms.reserve(m.voxelCount);
            palSyms.reserve(m.voxelCount);

            // Sub-cluster bin count per cell (for header cost).
            std::unordered_map<uint32_t, uint32_t> cellCount;
            cellCount.reserve(64);

            for (uint32_t i = 0; i < m.voxelCount; ++i) {
                const DiskVoxel& dv = cv[i];
                if (dv.visMask == 0) continue;
                uint32_t pos = (uint32_t)dv.x
                             | ((uint32_t)dv.y << 8)
                             | ((uint32_t)dv.z << 16);
                posSyms.push_back(pos);
                maskSyms.push_back(dv.visMask & 0x3Fu);
                uint32_t ao6 = 0;
                for (int fi = 0; fi < 6; ++fi)
                    ao6 |= (uint32_t)((DvAo(dv, fi) >> 4) & 0xFu) << (fi * 4);
                aoSyms.push_back(ao6);
                auto it = palMap.find(DvColor(dv) & 0x00FFFFFFu);
                palSyms.push_back(it != palMap.end() ? it->second : 0u);

                uint32_t cx = dv.x / S, cy = dv.y / S, cz = dv.z / S;
                uint32_t cellId = cx + cy * cellsPerAxis + cz * cellsPerAxis * cellsPerAxis;
                ++cellCount[cellId];
            }
            if (posSyms.empty()) continue;

            // Sub-cluster bit cost: occupied cell headers + per-voxel intra bits.
            subPosBits += (uint64_t)cellHdrBits * cellCount.size()
                       +  (uint64_t)intraBits   * posSyms.size();

            // Build packed blobs for LZ4. Pos stream = raw 15..18 bits/voxel
            // (whatever posBits resolved to); palette idx = 1 byte each.
            packStream(posBlob,  posBitsPerAxis * 3u, posSyms);
            packStream(maskBlob, 6u,                  maskSyms);
            packStream(aoBlob,   24u,                 aoSyms);
            palBlob.assign(palSyms.size(), 0);
            for (size_t k = 0; k < palSyms.size(); ++k) palBlob[k] = (uint8_t)palSyms[k];

            lz4Pos  += (uint64_t)lz4Compress(posBlob);
            lz4Mask += (uint64_t)lz4Compress(maskBlob);
            lz4Ao   += (uint64_t)lz4Compress(aoBlob);
            lz4Pal  += (uint64_t)lz4Compress(palBlob);
        }

        out.compSubclusterPosBytes = (subPosBits + 7) / 8;
        out.compLz4PosBytes        = lz4Pos;
        out.compLz4MaskBytes       = lz4Mask;
        out.compLz4AoBytes         = lz4Ao;
        out.compLz4ColorPalBytes   = lz4Pal;
        out.compLz4TotalBytes      = lz4Pos + lz4Mask + lz4Ao + lz4Pal + out.compPaletteBytes;

        auto mb = [](uint64_t b) { return (double)b / (1024.0 * 1024.0); };
        std::printf("[compress] subcluster S=%u (cellIdx %u bits + intra %u bits): pos = %.2f MB\n",
                    S, cellIdxBits, intraBits, mb(out.compSubclusterPosBytes));
        std::printf("[compress] LZ4 per-chunk:\n");
        std::printf("[compress]   pos     %.2f MB  (raw %.2f MB)\n", mb(lz4Pos),  mb(out.compPosBytes));
        std::printf("[compress]   mask    %.2f MB  (raw %.2f MB)\n", mb(lz4Mask), mb(out.compMaskBytes));
        std::printf("[compress]   ao      %.2f MB  (raw %.2f MB)\n", mb(lz4Ao),   mb(out.compAoBytes));
        std::printf("[compress]   colorPI %.2f MB  (raw %.2f MB)\n", mb(lz4Pal),  mb(out.compColorPalIdxBytes));
        std::printf("[compress]   TOTAL   %.2f MB + palette %llu B  (ratio vs raw %.2fx)\n",
                    mb(out.compLz4TotalBytes), (unsigned long long)out.compPaletteBytes,
                    out.compRawBytes ? (double)out.compRawBytes / (double)out.compLz4TotalBytes : 0.0);
    }

    return true;
}
