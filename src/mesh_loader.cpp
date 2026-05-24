#define _CRT_SECURE_NO_WARNINGS
#include "mesh_loader.h"
#include "asset_version.h"

#include <cstdio>
#include <cstring>

bool LoadMergedMesh(const char* path, MergedMesh& out, std::string& err)
{
    FILE* f = fopen(path, "rb");
    if (!f) { err = "open failed"; return false; }

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "MSH1", 4) != 0) {
        fclose(f); err = "bad magic (need MSH1)"; return false;
    }
    uint32_t version = 0;
    if (fread(&version, sizeof(uint32_t), 1, f) != 1 || version != kAssetVersion) {
        fclose(f); err = "asset version mismatch"; return false;
    }
    uint32_t vc = 0, ic = 0;
    fread(&vc, sizeof(uint32_t), 1, f);
    fread(&ic, sizeof(uint32_t), 1, f);

    out.vertices.resize(vc);
    if (vc) fread(out.vertices.data(), sizeof(MergedVertex), vc, f);
    out.indices.resize(ic);
    if (ic) fread(out.indices.data(), sizeof(uint32_t), ic, f);
    fclose(f);

    out.aabbMin[0] = out.aabbMin[1] = out.aabbMin[2] =  1e30f;
    out.aabbMax[0] = out.aabbMax[1] = out.aabbMax[2] = -1e30f;
    for (auto& v : out.vertices) {
        if (v.px < out.aabbMin[0]) out.aabbMin[0] = v.px;
        if (v.py < out.aabbMin[1]) out.aabbMin[1] = v.py;
        if (v.pz < out.aabbMin[2]) out.aabbMin[2] = v.pz;
        if (v.px > out.aabbMax[0]) out.aabbMax[0] = v.px;
        if (v.py > out.aabbMax[1]) out.aabbMax[1] = v.py;
        if (v.pz > out.aabbMax[2]) out.aabbMax[2] = v.pz;
    }
    return true;
}

bool LoadAtlasMesh(const char* path, AtlasMesh& out, std::string& err)
{
    FILE* f = fopen(path, "rb");
    if (!f) { err = "open failed"; return false; }

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "MSH2", 4) != 0) {
        fclose(f); err = "bad magic (need MSH2)"; return false;
    }
    uint32_t version = 0;
    if (fread(&version, sizeof(uint32_t), 1, f) != 1 || version != kAssetVersion) {
        fclose(f); err = "asset version mismatch"; return false;
    }
    uint32_t vc = 0, ic = 0, aw = 0, ah = 0, vbytes = 0;
    int32_t  origin[3] = { 0, 0, 0 };
    uint32_t chunkCount = 0;
    fread(&vc, sizeof(uint32_t), 1, f);
    fread(&ic, sizeof(uint32_t), 1, f);
    fread(&aw, sizeof(uint32_t), 1, f);
    fread(&ah, sizeof(uint32_t), 1, f);
    fread(&vbytes, sizeof(uint32_t), 1, f);
    fread(origin, sizeof(int32_t), 3, f);
    fread(&chunkCount, sizeof(uint32_t), 1, f);
    if (vbytes != sizeof(AtlasVertex)) {
        fclose(f); err = "atlas vertex size mismatch"; return false;
    }

    out.chunks.resize(chunkCount);
    if (chunkCount) fread(out.chunks.data(), sizeof(AtlasChunkSub), chunkCount, f);
    out.vertices.resize(vc);
    if (vc) fread(out.vertices.data(), sizeof(AtlasVertex), vc, f);
    out.indices.resize(ic);
    if (ic) fread(out.indices.data(), sizeof(uint32_t), ic, f);
    out.atlasW = aw;
    out.atlasH = ah;
    out.atlasPixels.resize((size_t)aw * ah);
    if (!out.atlasPixels.empty()) {
        fread(out.atlasPixels.data(), sizeof(uint32_t), out.atlasPixels.size(), f);
    }
    fclose(f);

    out.origin[0] = origin[0];
    out.origin[1] = origin[1];
    out.origin[2] = origin[2];

    out.aabbMin[0] = out.aabbMin[1] = out.aabbMin[2] =  1e30f;
    out.aabbMax[0] = out.aabbMax[1] = out.aabbMax[2] = -1e30f;
    for (auto& c : out.chunks) {
        for (int d = 0; d < 3; ++d) {
            float lo = (float)c.aabbMin[d];
            float hi = (float)c.aabbMax[d];
            if (lo < out.aabbMin[d]) out.aabbMin[d] = lo;
            if (hi > out.aabbMax[d]) out.aabbMax[d] = hi;
        }
    }
    return true;
}
