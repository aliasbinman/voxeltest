#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct MergedVertex {
    float    px, py, pz;
    uint32_t color;
};

struct MergedMesh {
    std::vector<MergedVertex> vertices;
    std::vector<uint32_t>     indices;
    float aabbMin[3] = {  1e30f,  1e30f,  1e30f };
    float aabbMax[3] = { -1e30f, -1e30f, -1e30f };
};

bool LoadMergedMesh(const char* path, MergedMesh& out, std::string& err);
