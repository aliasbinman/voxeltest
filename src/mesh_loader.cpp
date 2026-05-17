#define _CRT_SECURE_NO_WARNINGS
#include "mesh_loader.h"

#include <cstdio>
#include <cstring>

bool LoadMergedMesh(const char* path, MergedMesh& out, std::string& err) {
    FILE* f = fopen(path, "rb");
    if (!f) { err = "open failed"; return false; }

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "MSH1", 4) != 0) {
        fclose(f); err = "bad magic (need MSH1)"; return false;
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
