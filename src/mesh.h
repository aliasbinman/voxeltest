#pragma once
#include <cstdint>
#include <vector>

// Unified 12-byte vertex (vb_ and pointVb_). Scene-relative position with
// 16-bit-per-axis range (0..65535) so cluster-span batching works and very
// large scenes don't wrap. pos in px/py/pz; color RGB+visMask in alpha.
struct Vertex {
    uint16_t px, py, pz, aux;       // aux unused (alignment)
    uint32_t color;                  // RGB8 + visMask byte
};
static_assert(sizeof(Vertex) == 12, "");

using VoxelPolyVertex = Vertex;

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
    uint32_t firstIndex   = 0;
    uint32_t indexCount   = 0;
    int32_t  baseVertex   = 0;
    // 3 LODs of point vertices: L0 = 1 per voxel (faceIdx = visMask), L1 = 2x2x2 avg, L2 = 4x4x4 avg.
    uint32_t pointFirst   = 0;
    uint32_t pointCount   = 0;
    uint32_t pointFirstL1 = 0;
    uint32_t pointCountL1 = 0;
    uint32_t pointFirstL2 = 0;
    uint32_t pointCountL2 = 0;
    float    aabbMin[3]   = { 0, 0, 0 };
    float    aabbMax[3]   = { 0, 0, 0 };
    float    chunkBase[3] = { 0, 0, 0 };  // world-space offset added to local xyz
};

struct Scene {
    std::vector<VoxelPolyVertex> vertices;    // poly mesh (vb_)
    std::vector<uint32_t>        indices;     // ABSOLUTE indices into vertices
    std::vector<Vertex>          pointVertices; // 1 per voxel + LOD1 + LOD2 (chunk-local)
    std::vector<uint32_t>        pointAo6;      // packed face AO (4 bits * 6 faces in low 24); aligned with pointVertices
    std::vector<SubMesh>         subs;
    int32_t  origin[3] = { 0, 0, 0 };         // scene origin; gChunkBase base value
    float    sunDir[3] = { 0.4f, 0.8f, 0.2f }; // baked sun direction (runtime lighting)
    float    aabbMin[3] = {  1e30f,  1e30f,  1e30f };
    float    aabbMax[3] = { -1e30f, -1e30f, -1e30f };
    uint64_t totalTriangles = 0;
    uint64_t totalVertices  = 0;
};
