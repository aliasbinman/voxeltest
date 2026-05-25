#pragma once
#include <cstdint>
#include <utility>
#include <vector>

// Unified 12-byte vertex used by pointVb_. Scene-relative position with
// 16-bit-per-axis range (0..65535) so cluster-span batching works and very
// large scenes don't wrap. pos in px/py/pz; color RGB+visMask in alpha.
struct Vertex {
    uint16_t px, py, pz, aux;       // aux low 8 bits = baked AO; low 6 bits also shadow mask
    uint32_t color;                  // RGB8 + visMask byte
};
static_assert(sizeof(Vertex) == 12, "");

inline Vertex MakeVoxVertex(int sx, int sy, int sz, uint32_t rgba, uint8_t mask, uint8_t ao)
{
    Vertex v;
    v.px = (uint16_t)sx;
    v.py = (uint16_t)sy;
    v.pz = (uint16_t)sz;
    v.aux = (uint16_t)ao;           // low 8 bits = baked AO (0=occluded, 255=open)
    v.color = (rgba & 0x00FFFFFFu) | ((uint32_t)mask << 24);
    return v;
}

struct SubMesh {
    // 4 LODs of point vertices: L0 = 1 per voxel (faceIdx = visMask),
    // L1 = 2x2x2 avg, L2 = 4x4x4 avg, L3 = 8x8x8 avg.
    uint32_t pointFirst   = 0;
    uint32_t pointCount   = 0;
    uint32_t pointFirstL1 = 0;
    uint32_t pointCountL1 = 0;
    uint32_t pointFirstL2 = 0;
    uint32_t pointCountL2 = 0;
    uint32_t pointFirstL3 = 0;
    uint32_t pointCountL3 = 0;
    float    aabbMin[3]   = { 0, 0, 0 };
    float    aabbMax[3]   = { 0, 0, 0 };
    float    chunkBase[3] = { 0, 0, 0 };  // world-space offset added to local xyz
};

struct Scene {
    std::vector<Vertex>          pointVertices; // L0 + LOD1/2/3 (chunk-local)
    std::vector<uint32_t>        pointAo6;      // packed face AO (4 bits * 6 faces in low 24); aligned with pointVertices
    std::vector<SubMesh>         subs;
    int32_t  origin[3] = { 0, 0, 0 };         // scene origin; gChunkBase base value
    float    sunDir[3] = { 0.4f, 0.8f, 0.2f }; // baked sun direction (runtime lighting)
    float    aabbMin[3] = {  1e30f,  1e30f,  1e30f };
    float    aabbMax[3] = { -1e30f, -1e30f, -1e30f };
    // Per-color voxel counts (L0). RGB packed 0x00BBGGRR; sorted desc by count.
    std::vector<std::pair<uint32_t, uint64_t>> colorHistogram;

    // Disk-compression simulation results (L0 only). All bytes.
    uint64_t compRawBytes        = 0;  // pointVertices+pointAo6 today
    uint64_t compPaletteBytes    = 0;  // palette table cost (n_unique * 3)
    uint64_t compPosBytes        = 0;  // bit-packed positions (chunk-local)
    uint64_t compMaskBytes       = 0;  // bit-packed 6-bit visMask
    uint64_t compAoBytes         = 0;  // bit-packed 24-bit AO (6 faces * 4 bits)
    uint64_t compColorPalIdxBytes = 0; // 8-bit palette indices
    uint64_t compColorHuffBytes  = 0;  // Huffman-coded color indices
    uint32_t compPosBitsPerAxis  = 0;  // ceil(log2(chunkDim))
    uint32_t compChunkDim        = 0;
    uint64_t compSubclusterPosBytes = 0; // pos via sub-cluster headers + intra-cell bits
    uint32_t compSubclusterDim      = 0; // S (sub-cell edge in voxels)
    uint64_t compLz4PosBytes       = 0; // LZ4 over pos stream (per-chunk concat)
    uint64_t compLz4MaskBytes      = 0;
    uint64_t compLz4AoBytes        = 0;
    uint64_t compLz4ColorPalBytes  = 0;
    uint64_t compLz4TotalBytes     = 0; // sum + palette table
};
