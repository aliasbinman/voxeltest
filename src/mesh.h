#pragma once
#include <cstdint>
#include <vector>

// Compact GPU vertex (8 bytes).
// xyz: local 0..chunkDim inclusive (corner pos).
// faceIdx: 0..5 (lookup into normal table in shader).
// color: RGBA8.
struct Vertex {
    uint8_t  x, y, z;
    uint8_t  faceIdx;
    uint32_t color;
};
static_assert(sizeof(Vertex) == 8, "");

struct SubMesh {
    uint32_t firstIndex   = 0;
    uint32_t indexCount   = 0;
    int32_t  baseVertex   = 0;
    // 3 LODs of point vertices: L0 = 1 per voxel, L1 = 1 per 2x2x2, L2 = 1 per 4x4x4.
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
    std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;  // empty -> renderer builds shared cube IB
    std::vector<Vertex>   pointVertices;  // 1 per voxel (faceIdx unused)
    std::vector<SubMesh>  subs;
    float aabbMin[3] = {  1e30f,  1e30f,  1e30f };
    float aabbMax[3] = { -1e30f, -1e30f, -1e30f };
    uint64_t totalTriangles = 0;
    uint64_t totalVertices  = 0;
};
