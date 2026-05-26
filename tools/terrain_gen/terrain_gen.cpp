// terrain_gen.cpp — Grand Canyon (or any lat/lon bbox) heightmap -> .vox
//
// Data source: AWS Terrain Tiles (Mapzen Terrarium PNG, public S3, no auth).
//   https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png
//   height_m = (R*256 + G + B/256) - 32768
//
// Output: .vox in the same format src/vox_loader.cpp reads (VXL3).
// On first engine load the .vox is converted to .vxb sidecar for fast reload.
//
// Usage (defaults to Grand Canyon, zoom 12, ~10km square):
//   terrain_gen.exe [--zoom Z] [--bbox lat0,lon0,lat1,lon1] [--out path.vox]
//                   [--scale meters_per_voxel] [--chunk N]

#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "asset_version.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <filesystem>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace fs = std::filesystem;

// ---------------- .vox on-disk types (must mirror src/vox_loader.cpp) ----------------
#pragma pack(push, 1)
struct DiskVoxel {
    uint8_t  x, y, z;
    uint8_t  visMask;
    uint16_t paletteIdx;
    uint8_t  aoPacked[3];
};
#pragma pack(pop)
static_assert(sizeof(DiskVoxel) == 9, "");

struct ChunkMeta {
    uint16_t cx, cy, cz, _pad;
    uint32_t voxelCount;
    uint32_t voxelOffset;
};
static_assert(sizeof(ChunkMeta) == 16, "");

// ---------------- Slippy-map math ----------------
static int LonToTileX(double lon, int z) {
    return (int)std::floor((lon + 180.0) / 360.0 * (1 << z));
}
static int LatToTileY(double lat, int z) {
    double rad = lat * 3.14159265358979 / 180.0;
    return (int)std::floor((1.0 - std::asinh(std::tan(rad)) / 3.14159265358979) / 2.0 * (1 << z));
}
// Meters per pixel at given lat + zoom (Web Mercator, 256px tiles).
static double MetersPerPixel(double lat, int z) {
    const double earthC = 40075016.686; // m
    return earthC * std::cos(lat * 3.14159265358979 / 180.0) / (256.0 * (1 << z));
}

// ---------------- Download a tile to cacheDir/z_x_y.png via curl ----------------
static bool FetchTile(int z, int x, int y, const std::string& cacheDir, std::string& outPath) {
    outPath = cacheDir + "/" + std::to_string(z) + "_" + std::to_string(x) + "_" + std::to_string(y) + ".png";
    if (fs::exists(outPath) && fs::file_size(outPath) > 0) return true;

    char url[512];
    std::snprintf(url, sizeof(url),
        "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/%d/%d/%d.png", z, x, y);

    std::string cmd = "curl -sSL -o \"" + outPath + "\" \"" + url + "\"";
    std::printf("  fetch %s\n", url);
    int rc = std::system(cmd.c_str());
    if (rc != 0 || !fs::exists(outPath) || fs::file_size(outPath) == 0) {
        std::fprintf(stderr, "  fetch FAILED (rc=%d): %s\n", rc, url);
        return false;
    }
    return true;
}

// ---------------- Terrarium decode: PNG RGB -> height in meters ----------------
static bool DecodeTile(const std::string& pngPath, std::vector<float>& heights) {
    int w, h, n;
    unsigned char* px = stbi_load(pngPath.c_str(), &w, &h, &n, 4);
    if (!px) { std::fprintf(stderr, "stbi_load failed: %s\n", pngPath.c_str()); return false; }
    if (w != 256 || h != 256) {
        std::fprintf(stderr, "unexpected tile size %dx%d (need 256x256): %s\n", w, h, pngPath.c_str());
        stbi_image_free(px); return false;
    }
    heights.assign(256 * 256, 0.0f);
    for (int i = 0; i < 256 * 256; ++i) {
        float R = px[i*4 + 0];
        float G = px[i*4 + 1];
        float B = px[i*4 + 2];
        heights[i] = (R * 256.0f + G + B / 256.0f) - 32768.0f;
    }
    stbi_image_free(px);
    return true;
}

// Elevation -> color (gradient). RGBA8 little-endian; alpha = 0xFF dead byte.
static uint32_t HeightToColor(float h_m) {
    struct Stop { float m; uint8_t r, g, b; };
    static const Stop stops[] = {
        {    0.0f,  50,  90, 140 },   // water/low blue
        {  300.0f, 160, 140,  90 },   // sand
        {  900.0f, 100, 150,  70 },   // grass
        { 1800.0f, 130, 100,  70 },   // rock brown
        { 2500.0f, 160, 140, 130 },   // light rock
        { 3500.0f, 230, 230, 235 },   // snow
    };
    const int N = (int)(sizeof(stops) / sizeof(stops[0]));
    if (h_m <= stops[0].m)     return 0xFF000000u | stops[0].r | (stops[0].g << 8) | (stops[0].b << 16);
    if (h_m >= stops[N-1].m)   return 0xFF000000u | stops[N-1].r | (stops[N-1].g << 8) | (stops[N-1].b << 16);
    for (int i = 1; i < N; ++i) {
        if (h_m < stops[i].m) {
            float t = (h_m - stops[i-1].m) / (stops[i].m - stops[i-1].m);
            uint8_t r = (uint8_t)(stops[i-1].r + (stops[i].r - stops[i-1].r) * t);
            uint8_t g = (uint8_t)(stops[i-1].g + (stops[i].g - stops[i-1].g) * t);
            uint8_t b = (uint8_t)(stops[i-1].b + (stops[i].b - stops[i-1].b) * t);
            return 0xFF000000u | r | (g << 8) | (b << 16);
        }
    }
    return 0xFFFFFFFFu;
}

int main(int argc, char** argv)
{
    // Grand Canyon default bbox (south rim through inner canyon to north rim,
    // includes Colorado River). ~7km × 4km at zoom 12.
    // Wider section: Bright Angel area, both rims, ~22 x 22 km.
    double lat0 = 36.000;  // south
    double lon0 = -112.300; // west
    double lat1 = 36.200;  // north
    double lon1 = -112.050; // east
    int    zoom = 14;       // ~8 m / voxel at this latitude
    double metersPerVoxel = 0.0;   // 0 = derive from zoom + latitude (1 voxel = 1 DEM sample)
    int    chunkDim = 64;
    std::string outPath = "assets/grand_canyon.vox";
    std::string cacheDir = "assets/tilecache";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--zoom") == 0 && i+1 < argc) zoom = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--bbox") == 0 && i+1 < argc) {
            std::sscanf(argv[++i], "%lf,%lf,%lf,%lf", &lat0, &lon0, &lat1, &lon1);
        }
        else if (std::strcmp(argv[i], "--out") == 0 && i+1 < argc) outPath = argv[++i];
        else if (std::strcmp(argv[i], "--scale") == 0 && i+1 < argc) metersPerVoxel = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--chunk") == 0 && i+1 < argc) chunkDim = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--cache") == 0 && i+1 < argc) cacheDir = argv[++i];
        else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 1;
        }
    }
    fs::create_directories(cacheDir);
    fs::create_directories(fs::path(outPath).parent_path());

    // Tile range covering bbox.
    int tx0 = LonToTileX(lon0, zoom);
    int tx1 = LonToTileX(lon1, zoom);
    int ty0 = LatToTileY(lat1, zoom);   // note: y grows southward
    int ty1 = LatToTileY(lat0, zoom);
    if (tx1 < tx0) std::swap(tx0, tx1);
    if (ty1 < ty0) std::swap(ty0, ty1);
    int tilesX = tx1 - tx0 + 1;
    int tilesY = ty1 - ty0 + 1;

    double midLat = 0.5 * (lat0 + lat1);
    double mPerPx = MetersPerPixel(midLat, zoom);
    if (metersPerVoxel <= 0.0) metersPerVoxel = mPerPx; // 1 voxel = 1 DEM pixel by default

    int W = tilesX * 256;
    int H = tilesY * 256;
    std::printf("bbox lat[%.4f..%.4f] lon[%.4f..%.4f] zoom=%d\n",
                lat0, lat1, lon0, lon1, zoom);
    std::printf("Tiles %dx%d (%d total). Heightmap %dx%d. m/pixel=%.2f, m/voxel=%.2f\n",
                tilesX, tilesY, tilesX*tilesY, W, H, mPerPx, metersPerVoxel);

    // Fetch + decode -> single big heightmap (meters).
    std::vector<float> heightMap((size_t)W * H, 0.0f);
    for (int ty = 0; ty < tilesY; ++ty) {
        for (int tx = 0; tx < tilesX; ++tx) {
            std::string path;
            if (!FetchTile(zoom, tx0 + tx, ty0 + ty, cacheDir, path)) return 1;
            std::vector<float> tile;
            if (!DecodeTile(path, tile)) return 1;
            for (int py = 0; py < 256; ++py) {
                int dstY = ty * 256 + py;
                for (int px = 0; px < 256; ++px) {
                    int dstX = tx * 256 + px;
                    heightMap[(size_t)dstY * W + dstX] = tile[py * 256 + px];
                }
            }
        }
    }

    // Clamp NoData / bathymetry sentinels. AWS Terrarium encodes ocean depth
    // and missing tiles with very negative values — those balloon the voxel
    // count with useless underwater columns. Pin the floor to sea level.
    for (float& h : heightMap) if (h < 0.0f) h = 0.0f;

    // Mirror rows: DEM row 0 = northernmost tile, but engine expects +Z = north.
    // Flip in-place so voxel.z = H-1 corresponds to geographic north.
    {
        std::vector<float> tmp((size_t)W);
        for (int z = 0; z < H / 2; ++z) {
            int zm = H - 1 - z;
            std::memcpy(tmp.data(), &heightMap[(size_t)z * W], W * sizeof(float));
            std::memcpy(&heightMap[(size_t)z * W], &heightMap[(size_t)zm * W], W * sizeof(float));
            std::memcpy(&heightMap[(size_t)zm * W], tmp.data(), W * sizeof(float));
        }
    }

    // Find min/max elevation.
    float hMin =  1e30f, hMax = -1e30f;
    for (float h : heightMap) { if (h < hMin) hMin = h; if (h > hMax) hMax = h; }
    std::printf("Elevation range: %.1fm .. %.1fm (relief %.1fm)\n", hMin, hMax, hMax - hMin);

    // Quantize to integer voxel heights. World layout:
    //   X axis = east (DEM column)
    //   Y axis = up (height)
    //   Z axis = south (DEM row)
    auto Hv = [&](int x, int z) -> int {
        float h = heightMap[(size_t)z * W + x];
        return (int)std::round((h - hMin) / (float)metersPerVoxel);
    };

    // Surface-voxel emission: top voxel per column + cliff faces for any
    // step difference > 1 against the 4 lateral neighbours.
    struct Voxel {
        int32_t x, y, z;
        uint8_t mask;       // visMask: +X -X +Y -Y +Z -Z
        uint32_t color;
    };
    std::vector<Voxel> voxels;
    voxels.reserve((size_t)W * H * 2);

    auto emit = [&](int x, int y, int z, uint8_t mask, uint32_t color) {
        voxels.push_back({ x, y, z, mask, color });
    };

    static const int kDX[4] = { +1, -1,  0,  0 };
    static const int kDZ[4] = {  0,  0, +1, -1 };
    static const uint8_t kFace[4] = { 0x01, 0x02, 0x10, 0x20 }; // +X -X +Z -Z

    for (int z = 0; z < H; ++z) {
        for (int x = 0; x < W; ++x) {
            int y = Hv(x, z);
            uint32_t col = HeightToColor(heightMap[(size_t)z * W + x]);
            uint8_t mask = 0x04; // +Y always exposed (top of column)
            for (int k = 0; k < 4; ++k) {
                int nx = x + kDX[k], nz = z + kDZ[k];
                int yn = (nx < 0 || nx >= W || nz < 0 || nz >= H) ? -1 : Hv(nx, nz);
                if (yn < y) mask |= kFace[k];
            }
            emit(x, y, z, mask, col);
            // Cliff voxels below the top: exposed only on the side(s) where
            // a neighbour is even lower than that row.
            for (int k = 0; k < 4; ++k) {
                int nx = x + kDX[k], nz = z + kDZ[k];
                int yn = (nx < 0 || nx >= W || nz < 0 || nz >= H) ? -1 : Hv(nx, nz);
                if (yn >= y - 1) continue;
                // Cliff rows (y-1 down to yn+1) — each exposes face kFace[k].
                for (int yc = y - 1; yc > yn; --yc) {
                    // Existing voxel at same (x,yc,z) might have been added by
                    // another direction's cliff loop — merge mask. Use a map.
                    // For simplicity (and because at this density collisions
                    // are common only inside cliffs), just emit; dedupe pass below.
                    emit(x, yc, z, kFace[k], col);
                }
            }
        }
    }

    // Dedupe: combine voxels with same (x,y,z), OR their masks, keep first colour.
    // unordered_map keyed on packed (x,y,z) int64.
    std::printf("Raw emitted voxels: %zu — deduping...\n", voxels.size());
    {
        std::unordered_map<uint64_t, Voxel> uniq;
        uniq.reserve(voxels.size() * 2);
        for (const Voxel& v : voxels) {
            // 21 bits per axis fits any reasonable terrain.
            uint64_t key = ((uint64_t)(uint32_t)v.x & 0x1FFFFF)
                         | (((uint64_t)(uint32_t)v.y & 0x1FFFFF) << 21)
                         | (((uint64_t)(uint32_t)v.z & 0x1FFFFF) << 42);
            auto it = uniq.find(key);
            if (it == uniq.end()) uniq[key] = v;
            else                  it->second.mask |= v.mask;
        }
        voxels.clear();
        voxels.reserve(uniq.size());
        for (auto& kv : uniq) voxels.push_back(kv.second);
    }
    std::printf("Unique surface voxels: %zu\n", voxels.size());

    // Build palette from unique colors (sorted desc by occurrence).
    std::unordered_map<uint32_t, uint64_t> palCnt;
    palCnt.reserve(4096);
    for (const Voxel& v : voxels) ++palCnt[v.color];
    std::vector<std::pair<uint32_t, uint64_t>> palSorted(palCnt.begin(), palCnt.end());
    std::sort(palSorted.begin(), palSorted.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    if (palSorted.size() > 65535) {
        std::fprintf(stderr, "palette > 65535 — quantize colours\n");
        return 1;
    }
    std::vector<uint32_t> palette;
    std::unordered_map<uint32_t, uint16_t> palMap;
    palette.reserve(palSorted.size());
    for (auto& p : palSorted) {
        palMap[p.first] = (uint16_t)palette.size();
        palette.push_back(p.first);
    }
    std::printf("Palette: %zu unique colours\n", palette.size());

    // Group voxels into chunks of (chunkDim x chunkDim x chunkDim).
    struct ChunkKey { int cx, cy, cz; };
    struct CKHash { size_t operator()(const std::pair<int, std::pair<int,int>>& k) const {
        size_t h = std::hash<int>{}(k.first);
        h = h * 1315423911u + std::hash<int>{}(k.second.first);
        h = h * 1315423911u + std::hash<int>{}(k.second.second);
        return h;
    }};
    using CK = std::pair<int, std::pair<int,int>>;
    std::unordered_map<CK, std::vector<DiskVoxel>, CKHash> chunks;

    // Recenter so all coords are non-negative. Origin field carries the original
    // world-space offset (no special meaning to engine other than display).
    int32_t minX = 0, minY = 0, minZ = 0;
    {
        int32_t mxX = 0, mxY = 0, mxZ = 0;
        bool first = true;
        for (const Voxel& v : voxels) {
            if (first) { minX = mxX = v.x; minY = mxY = v.y; minZ = mxZ = v.z; first = false; continue; }
            if (v.x < minX) minX = v.x; if (v.x > mxX) mxX = v.x;
            if (v.y < minY) minY = v.y; if (v.y > mxY) mxY = v.y;
            if (v.z < minZ) minZ = v.z; if (v.z > mxZ) mxZ = v.z;
        }
        std::printf("Voxel bounds: [%d..%d, %d..%d, %d..%d]\n", minX, mxX, minY, mxY, minZ, mxZ);
    }

    // ---- AO bake (per face, 4-bit). Sample a 3x3 grid one step in front of
    // each visible face; "occupied" = sample y <= terrain height at that (x,z).
    // Hits → occlusion. Cheap (O(1) lookups via heightmap), parallelizable.
    std::printf("Baking AO (per face)...\n");
    auto occupied = [&](int x, int y, int z) -> bool {
        if (x < 0 || x >= W || z < 0 || z >= H) return false;
        return y <= Hv(x, z) && y >= 0;
    };
    // Face -> (normal, u tangent, v tangent). Face order: +X -X +Y -Y +Z -Z.
    static const int kN[6][3] = {
        { +1, 0, 0 }, { -1, 0, 0 },
        { 0, +1, 0 }, { 0, -1, 0 },
        { 0, 0, +1 }, { 0, 0, -1 },
    };
    static const int kU[6][3] = {
        { 0, 1, 0 }, { 0, 1, 0 },
        { 1, 0, 0 }, { 1, 0, 0 },
        { 1, 0, 0 }, { 1, 0, 0 },
    };
    static const int kV[6][3] = {
        { 0, 0, 1 }, { 0, 0, 1 },
        { 0, 0, 1 }, { 0, 0, 1 },
        { 0, 1, 0 }, { 0, 1, 0 },
    };
    auto faceAo4 = [&](int x, int y, int z, int fi) -> uint8_t {
        int hits = 0;
        for (int du = -1; du <= 1; ++du) {
            for (int dv = -1; dv <= 1; ++dv) {
                int sx = x + kN[fi][0] + du * kU[fi][0] + dv * kV[fi][0];
                int sy = y + kN[fi][1] + du * kU[fi][1] + dv * kV[fi][1];
                int sz = z + kN[fi][2] + du * kU[fi][2] + dv * kV[fi][2];
                if (occupied(sx, sy, sz)) ++hits;
            }
        }
        // 9 samples; AO = 1 - hits/9, quantized to 4 bits (0..15).
        int ao4 = (int)std::round((1.0f - hits / 9.0f) * 15.0f);
        if (ao4 < 0) ao4 = 0; if (ao4 > 15) ao4 = 15;
        return (uint8_t)ao4;
    };

    for (const Voxel& v : voxels) {
        int rx = v.x - minX, ry = v.y - minY, rz = v.z - minZ;
        CK ck = { rx / chunkDim, { ry / chunkDim, rz / chunkDim } };
        DiskVoxel dv;
        dv.x = (uint8_t)(rx % chunkDim);
        dv.y = (uint8_t)(ry % chunkDim);
        dv.z = (uint8_t)(rz % chunkDim);
        dv.visMask    = v.mask & 0x3Fu;
        dv.paletteIdx = palMap[v.color];
        dv.aoPacked[0] = dv.aoPacked[1] = dv.aoPacked[2] = 0;
        for (int fi = 0; fi < 6; ++fi) {
            // Faces not in visMask get AO=15 (max) — never read, but cheap default.
            uint8_t a4 = ((v.mask >> fi) & 1u) ? faceAo4(v.x, v.y, v.z, fi) : 15;
            int byte  = (fi * 4) >> 3;
            int shift = (fi * 4) & 7;
            dv.aoPacked[byte] |= (uint8_t)(a4 << shift);
        }
        chunks[ck].push_back(dv);
    }
    std::printf("Chunks: %zu (chunkDim=%d)\n", chunks.size(), chunkDim);

    // Flatten chunk + voxel arrays for .vox write.
    std::vector<ChunkMeta> metas;
    std::vector<DiskVoxel> flatVox;
    metas.reserve(chunks.size());
    size_t totalV = 0;
    for (auto& kv : chunks) totalV += kv.second.size();
    flatVox.reserve(totalV);
    for (auto& kv : chunks) {
        ChunkMeta m;
        m.cx = (uint16_t)kv.first.first;
        m.cy = (uint16_t)kv.first.second.first;
        m.cz = (uint16_t)kv.first.second.second;
        m._pad = 0;
        m.voxelCount  = (uint32_t)kv.second.size();
        m.voxelOffset = (uint32_t)flatVox.size();
        metas.push_back(m);
        for (auto& v : kv.second) flatVox.push_back(v);
    }

    FILE* f = std::fopen(outPath.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "open %s failed\n", outPath.c_str()); return 1; }
    const char magic[4] = { 'V','X','L','3' };
    std::fwrite(magic, 1, 4, f);
    uint32_t version = kAssetVersion;
    std::fwrite(&version, sizeof(uint32_t), 1, f);
    uint32_t cdim = (uint32_t)chunkDim;
    uint32_t ccount = (uint32_t)metas.size();
    uint32_t totalVox = (uint32_t)flatVox.size();
    std::fwrite(&cdim,     sizeof(uint32_t), 1, f);
    std::fwrite(&ccount,   sizeof(uint32_t), 1, f);
    std::fwrite(&totalVox, sizeof(uint32_t), 1, f);
    int32_t origin[3] = { minX, minY, minZ };
    std::fwrite(origin, sizeof(int32_t), 3, f);
    float sunDir[3] = { 0.4f, 0.8f, 0.2f };
    std::fwrite(sunDir, sizeof(float), 3, f);
    uint32_t palCount = (uint32_t)palette.size();
    std::fwrite(&palCount, sizeof(uint32_t), 1, f);
    if (palCount) std::fwrite(palette.data(), sizeof(uint32_t), palCount, f);
    std::fwrite(metas.data(),   sizeof(ChunkMeta), metas.size(),   f);
    std::fwrite(flatVox.data(), sizeof(DiskVoxel), flatVox.size(), f);
    long bytes = std::ftell(f);
    std::fclose(f);
    std::printf("Wrote %s: %ld bytes (%.2f MB), %u voxels, %.2f B/voxel\n",
                outPath.c_str(), bytes, bytes / (1024.0 * 1024.0), totalVox,
                (double)bytes / (double)totalVox);

    // Invalidate any stale .vxb sidecar so the engine re-bakes from this .vox.
    std::string vxb = outPath;
    size_t dot = vxb.find_last_of('.');
    if (dot != std::string::npos) vxb.resize(dot);
    vxb += ".vxb";
    std::error_code ec;
    if (fs::remove(vxb, ec)) std::printf("Removed stale baked sidecar: %s\n", vxb.c_str());
    return 0;
}
