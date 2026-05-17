#pragma once
#include <cstdint>
#include <vector>

struct ObjVertex {
    float    px, py, pz;
    float    nx, ny, nz;
    uint32_t color;
};

struct ObjSubMesh {
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    int32_t  baseVertex = 0;
};

struct ObjScene {
    std::vector<ObjVertex> vertices;
    std::vector<uint32_t>  indices;
    std::vector<ObjSubMesh> subs;
    float aabbMin[3] = {  1e30f,  1e30f,  1e30f };
    float aabbMax[3] = { -1e30f, -1e30f, -1e30f };
    uint64_t totalTriangles = 0;
    uint64_t totalVertices  = 0;
};
