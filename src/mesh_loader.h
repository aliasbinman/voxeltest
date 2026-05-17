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

#pragma pack(push, 1)
struct AtlasVertex {
    uint16_t px, py, pz;
    uint8_t  face;
    uint8_t  _pad;
    uint16_t u, v;
};
struct AtlasChunkSub {
    uint16_t cx, cy, cz, _pad;
    int32_t  aabbMin[3];     // world voxel coords
    int32_t  aabbMax[3];
    uint32_t firstIndex;
    uint32_t indexCount;
};
#pragma pack(pop)
static_assert(sizeof(AtlasVertex) == 12, "AtlasVertex size");
static_assert(sizeof(AtlasChunkSub) == 40, "AtlasChunkSub size");

struct AtlasMesh {
    std::vector<AtlasVertex>   vertices;
    std::vector<uint32_t>      indices;
    std::vector<AtlasChunkSub> chunks;
    std::vector<uint32_t>      atlasPixels;     // RGBA8, atlasW * atlasH
    uint32_t atlasW = 0;
    uint32_t atlasH = 0;
    int32_t  origin[3] = { 0, 0, 0 };           // world-space offset to add to uint16 pos
    float    aabbMin[3] = {  1e30f,  1e30f,  1e30f };  // world-space scene AABB
    float    aabbMax[3] = { -1e30f, -1e30f, -1e30f };
};

bool LoadAtlasMesh(const char* path, AtlasMesh& out, std::string& err);
