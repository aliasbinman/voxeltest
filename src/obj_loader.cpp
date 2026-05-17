#define _CRT_SECURE_NO_WARNINGS
#include "obj_loader.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <filesystem>

namespace {

struct TempGroup {
    std::string name;
    float kd[3] = { 0.8f, 0.8f, 0.8f };
    std::vector<ObjVertex> vertices;
    std::vector<uint32_t>  indices;
};

struct Mtl { float kd[3] = { 0.8f, 0.8f, 0.8f }; };

static std::string DirOf(const std::string& path) {
    std::filesystem::path p(path);
    return p.parent_path().string();
}

static bool ReadWhole(const char* path, std::vector<char>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long long sz = _ftelli64(f);
    fseek(f, 0, SEEK_SET);
    out.resize((size_t)sz + 1);
    size_t got = fread(out.data(), 1, (size_t)sz, f);
    fclose(f);
    out[(size_t)sz] = 0;
    return got == (size_t)sz;
}

static inline const char* SkipSpaces(const char* p) {
    while (*p == ' ' || *p == '\t') ++p;
    return p;
}
static inline const char* EndOfLine(const char* p) {
    while (*p && *p != '\n' && *p != '\r') ++p;
    return p;
}
static inline const char* NextLine(const char* p) {
    p = EndOfLine(p);
    while (*p == '\r' || *p == '\n') ++p;
    return p;
}

static inline const char* ParseFloat(const char* p, float& out) {
    p = SkipSpaces(p);
    bool neg = false;
    if (*p == '-') { neg = true; ++p; }
    else if (*p == '+') ++p;
    double val = 0.0;
    while (*p >= '0' && *p <= '9') { val = val * 10.0 + (*p - '0'); ++p; }
    if (*p == '.') {
        ++p;
        double frac = 0.0, scale = 1.0;
        while (*p >= '0' && *p <= '9') { frac = frac * 10.0 + (*p - '0'); scale *= 10.0; ++p; }
        val += frac / scale;
    }
    if (*p == 'e' || *p == 'E') {
        ++p;
        bool eneg = false;
        if (*p == '-') { eneg = true; ++p; }
        else if (*p == '+') ++p;
        int e = 0;
        while (*p >= '0' && *p <= '9') { e = e * 10 + (*p - '0'); ++p; }
        double s = 1.0;
        for (int i = 0; i < e; ++i) s *= 10.0;
        if (eneg) val /= s; else val *= s;
    }
    out = neg ? -float(val) : float(val);
    return p;
}

static inline const char* ParseInt(const char* p, int& out) {
    bool neg = false;
    if (*p == '-') { neg = true; ++p; }
    else if (*p == '+') ++p;
    int v = 0;
    while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); ++p; }
    out = neg ? -v : v;
    return p;
}

static bool LoadMtl(const char* path, std::unordered_map<std::string, Mtl>& mtls) {
    std::vector<char> buf;
    if (!ReadWhole(path, buf)) return false;
    const char* p = buf.data();
    Mtl* cur = nullptr;
    std::string curName;
    while (*p) {
        p = SkipSpaces(p);
        if (p[0] == 'n' && p[1] == 'e' && p[2] == 'w' && p[3] == 'm' &&
            p[4] == 't' && p[5] == 'l' && p[6] == ' ') {
            p += 7;
            p = SkipSpaces(p);
            const char* s = p;
            const char* e = EndOfLine(p);
            while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r')) --e;
            curName.assign(s, e - s);
            cur = &mtls[curName];
            p = NextLine(p);
            continue;
        }
        if (cur && p[0] == 'K' && p[1] == 'd' && (p[2] == ' ' || p[2] == '\t')) {
            p += 3;
            p = ParseFloat(p, cur->kd[0]);
            p = ParseFloat(p, cur->kd[1]);
            p = ParseFloat(p, cur->kd[2]);
            p = NextLine(p);
            continue;
        }
        p = NextLine(p);
    }
    return true;
}

} // namespace

bool LoadObjScene(const char* objPath, ObjScene& out, std::string& err) {
    std::vector<char> buf;
    if (!ReadWhole(objPath, buf)) { err = "Failed to open OBJ"; return false; }

    std::vector<float> positions; positions.reserve(3 * 1024 * 1024);
    std::vector<float> normals;   normals.reserve(3 * 1024 * 1024);

    std::unordered_map<std::string, Mtl> mtls;
    std::string baseDir = DirOf(objPath);

    out.vertices.clear();
    out.indices.clear();
    out.subs.clear();
    out.aabbMin[0] = out.aabbMin[1] = out.aabbMin[2] =  1e30f;
    out.aabbMax[0] = out.aabbMax[1] = out.aabbMax[2] = -1e30f;

    std::vector<TempGroup> tempGroups;
    int curGroupIdx = -1;
    std::unordered_map<std::string, int> groupByName;
    std::unordered_map<uint64_t, uint32_t> dedupe;

    auto ensureGroup = [&](const std::string& name) {
        auto it = groupByName.find(name);
        if (it != groupByName.end()) { curGroupIdx = it->second; return; }
        TempGroup g;
        g.name = name;
        auto mit = mtls.find(name);
        if (mit != mtls.end()) {
            g.kd[0] = mit->second.kd[0];
            g.kd[1] = mit->second.kd[1];
            g.kd[2] = mit->second.kd[2];
        }
        tempGroups.push_back(std::move(g));
        curGroupIdx = (int)tempGroups.size() - 1;
        groupByName[name] = curGroupIdx;
        dedupe.clear();
    };

    const char* p = buf.data();
    int faceVerts[64];
    int faceNorms[64];

    while (*p) {
        // tags
        if (p[0] == 'v' && p[1] == ' ') {
            p += 2;
            float x, y, z;
            p = ParseFloat(p, x);
            p = ParseFloat(p, y);
            p = ParseFloat(p, z);
            positions.push_back(x);
            positions.push_back(y);
            positions.push_back(z);
            if (x < out.aabbMin[0]) out.aabbMin[0] = x;
            if (y < out.aabbMin[1]) out.aabbMin[1] = y;
            if (z < out.aabbMin[2]) out.aabbMin[2] = z;
            if (x > out.aabbMax[0]) out.aabbMax[0] = x;
            if (y > out.aabbMax[1]) out.aabbMax[1] = y;
            if (z > out.aabbMax[2]) out.aabbMax[2] = z;
            p = NextLine(p);
            continue;
        }
        if (p[0] == 'v' && p[1] == 'n' && p[2] == ' ') {
            p += 3;
            float x, y, z;
            p = ParseFloat(p, x);
            p = ParseFloat(p, y);
            p = ParseFloat(p, z);
            normals.push_back(x);
            normals.push_back(y);
            normals.push_back(z);
            p = NextLine(p);
            continue;
        }
        if (p[0] == 'v' && p[1] == 't' && p[2] == ' ') {
            // ignore uvs
            p = NextLine(p);
            continue;
        }
        if (p[0] == 'f' && p[1] == ' ') {
            if (curGroupIdx < 0) ensureGroup("__default");
            p += 2;
            int nVerts = 0;
            while (*p && *p != '\n' && *p != '\r') {
                p = SkipSpaces(p);
                if (*p == '\n' || *p == '\r' || *p == 0) break;
                int vi = 0, ti = 0, ni = 0;
                p = ParseInt(p, vi);
                if (*p == '/') {
                    ++p;
                    if (*p != '/') p = ParseInt(p, ti);
                    if (*p == '/') { ++p; p = ParseInt(p, ni); }
                }
                if (nVerts < 64) {
                    faceVerts[nVerts] = vi;
                    faceNorms[nVerts] = ni;
                    ++nVerts;
                }
            }

            int posCount = (int)(positions.size() / 3);
            int nrmCount = (int)(normals.size() / 3);

            int resolvedV[64];
            int resolvedN[64];
            for (int i = 0; i < nVerts; ++i) {
                int vi = faceVerts[i];
                int ni = faceNorms[i];
                resolvedV[i] = vi > 0 ? vi - 1 : posCount + vi;
                resolvedN[i] = ni > 0 ? ni - 1 : (ni < 0 ? nrmCount + ni : -1);
            }

            TempGroup& g = tempGroups[curGroupIdx];
            auto pack8 = [](float v) -> uint32_t {
                if (v < 0.0f) v = 0.0f; if (v > 1.0f) v = 1.0f;
                return (uint32_t)(v * 255.0f + 0.5f);
            };
            uint32_t gcolor = pack8(g.kd[0]) | (pack8(g.kd[1]) << 8) | (pack8(g.kd[2]) << 16) | (255u << 24);
            auto emit = [&](int idx) -> uint32_t {
                uint64_t key = (uint64_t)(uint32_t)resolvedV[idx] << 32
                             | (uint64_t)(uint32_t)resolvedN[idx];
                auto it = dedupe.find(key);
                if (it != dedupe.end()) return it->second;
                ObjVertex v;
                int vi = resolvedV[idx];
                v.px = positions[vi * 3 + 0];
                v.py = positions[vi * 3 + 1];
                v.pz = positions[vi * 3 + 2];
                int ni = resolvedN[idx];
                if (ni >= 0) {
                    v.nx = normals[ni * 3 + 0];
                    v.ny = normals[ni * 3 + 1];
                    v.nz = normals[ni * 3 + 2];
                } else {
                    v.nx = 0; v.ny = 1; v.nz = 0;
                }
                v.color = gcolor;
                uint32_t newIdx = (uint32_t)g.vertices.size();
                g.vertices.push_back(v);
                dedupe[key] = newIdx;
                return newIdx;
            };

            // fan triangulation
            for (int i = 1; i + 1 < nVerts; ++i) {
                uint32_t a = emit(0);
                uint32_t b = emit(i);
                uint32_t c = emit(i + 1);
                g.indices.push_back(a);
                g.indices.push_back(b);
                g.indices.push_back(c);
            }
            p = NextLine(p);
            continue;
        }
        if (p[0] == 'u' && strncmp(p, "usemtl ", 7) == 0) {
            p += 7;
            p = SkipSpaces(p);
            const char* s = p;
            const char* e = EndOfLine(p);
            while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r')) --e;
            std::string name(s, e - s);
            ensureGroup(name);
            p = NextLine(p);
            continue;
        }
        if (p[0] == 'm' && strncmp(p, "mtllib ", 7) == 0) {
            p += 7;
            p = SkipSpaces(p);
            const char* s = p;
            const char* e = EndOfLine(p);
            while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r')) --e;
            std::string fname(s, e - s);
            std::string full = baseDir + "/" + fname;
            LoadMtl(full.c_str(), mtls);
            p = NextLine(p);
            continue;
        }
        p = NextLine(p);
    }

    // flatten into Scene: one big VB, one big IB, one SubMesh per non-empty group
    size_t totalV = 0, totalI = 0;
    for (auto& g : tempGroups) { totalV += g.vertices.size(); totalI += g.indices.size(); }
    out.vertices.reserve(totalV);
    out.indices.reserve(totalI);
    out.subs.reserve(tempGroups.size());

    for (auto& g : tempGroups) {
        if (g.indices.empty()) continue;
        ObjSubMesh sm;
        sm.firstIndex = (uint32_t)out.indices.size();
        sm.indexCount = (uint32_t)g.indices.size();
        sm.baseVertex = (int32_t)out.vertices.size();
        for (auto& v : g.vertices) {
            if (v.px < out.aabbMin[0]) out.aabbMin[0] = v.px;
            if (v.py < out.aabbMin[1]) out.aabbMin[1] = v.py;
            if (v.pz < out.aabbMin[2]) out.aabbMin[2] = v.pz;
            if (v.px > out.aabbMax[0]) out.aabbMax[0] = v.px;
            if (v.py > out.aabbMax[1]) out.aabbMax[1] = v.py;
            if (v.pz > out.aabbMax[2]) out.aabbMax[2] = v.pz;
        }
        out.vertices.insert(out.vertices.end(), g.vertices.begin(), g.vertices.end());
        out.indices.insert(out.indices.end(), g.indices.begin(), g.indices.end());
        out.totalVertices  += g.vertices.size();
        out.totalTriangles += g.indices.size() / 3;
        out.subs.push_back(sm);
    }
    return true;
}
