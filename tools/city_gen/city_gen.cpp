// city_gen.cpp — OpenStreetMap (Overpass) building footprints -> .vox
//
// Pulls building polygons + heights via Overpass API, rasterizes each footprint
// into voxel columns, extrudes between min_height and height, computes visMask,
// writes engine-compatible .vox.
//
// Usage:
//   city_gen.exe [--bbox lat0,lon0,lat1,lon1] [--out path.vox]
//                [--scale meters_per_voxel] [--chunk N] [--cache file]
//
// Default = Manhattan core. Overpass query is one-shot; result cached on disk.

#define _CRT_SECURE_NO_WARNINGS
#include "nlohmann/json.hpp"
#include "asset_version.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <filesystem>
#include <fstream>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace fs   = std::filesystem;
using json     = nlohmann::json;

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

// ---------------- Overpass fetch ----------------
static bool FetchOverpass(double lat0, double lon0, double lat1, double lon1,
                          const std::string& cachePath)
{
    if (fs::exists(cachePath) && fs::file_size(cachePath) > 1024) {
        std::printf("Using cached Overpass response: %s\n", cachePath.c_str());
        return true;
    }
    // Note: bbox order in Overpass is (south, west, north, east).
    char query[2048];
    std::snprintf(query, sizeof(query),
        "[out:json][timeout:600];"
        "("
          "way[\"building\"](%.6f,%.6f,%.6f,%.6f);"
          "way[\"highway\"](%.6f,%.6f,%.6f,%.6f);"
          "way[\"railway\"](%.6f,%.6f,%.6f,%.6f);"
          "way[\"waterway\"](%.6f,%.6f,%.6f,%.6f);"
          "way[\"natural\"=\"water\"](%.6f,%.6f,%.6f,%.6f);"
          "way[\"landuse\"~\"^(grass|forest|meadow|park|recreation_ground|cemetery|farmland)$\"](%.6f,%.6f,%.6f,%.6f);"
          "way[\"leisure\"~\"^(park|garden|pitch|playground|golf_course)$\"](%.6f,%.6f,%.6f,%.6f);"
          "relation[\"natural\"=\"water\"](%.6f,%.6f,%.6f,%.6f);"
          "relation[\"waterway\"=\"riverbank\"](%.6f,%.6f,%.6f,%.6f);"
        ");"
        "out body;>;out skel qt;",
        lat0, lon0, lat1, lon1,
        lat0, lon0, lat1, lon1,
        lat0, lon0, lat1, lon1,
        lat0, lon0, lat1, lon1,
        lat0, lon0, lat1, lon1,
        lat0, lon0, lat1, lon1,
        lat0, lon0, lat1, lon1,
        lat0, lon0, lat1, lon1,
        lat0, lon0, lat1, lon1);

    // POST to public Overpass endpoint. Use curl with --data-urlencode.
    // Quote escaping on Windows: build a temp file with the body, then -d @file.
    std::string bodyPath = cachePath + ".query";
    {
        std::ofstream bf(bodyPath, std::ios::trunc);
        bf << query;
    }
    // Use --data-urlencode "data@file" so curl URL-encodes the body and sends
    // it as application/x-www-form-urlencoded — Overpass requires this form.
    // Public Overpass requires a real User-Agent or returns 406.
    std::string cmd = "curl -sSL "
                      "-A \"voxeltest-city_gen/1.0\" "
                      "--data-urlencode \"data@" + bodyPath + "\" "
                      "-o \"" + cachePath + "\" "
                      "https://overpass-api.de/api/interpreter";
    std::printf("Querying Overpass (%s)...\n", cachePath.c_str());
    int rc = std::system(cmd.c_str());
    std::error_code ec;
    fs::remove(bodyPath, ec);
    if (rc != 0 || !fs::exists(cachePath) || fs::file_size(cachePath) < 1024) {
        std::fprintf(stderr, "Overpass fetch FAILED (rc=%d). Try again, the public "
                             "instance is rate-limited. Or use a mirror.\n", rc);
        return false;
    }
    std::printf("Overpass response: %llu bytes\n",
                (unsigned long long)fs::file_size(cachePath));
    return true;
}

// ---------------- Height parsing helpers ----------------
static double ParseLength(const std::string& s)
{
    // "12", "12.5", "12 m", "12m", "40'2\"" (rare). Crude — first number wins.
    if (s.empty()) return 0.0;
    char* end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str()) return 0.0;
    // If 'ft' or "'", convert.
    while (end && *end == ' ') ++end;
    if (end && (std::strncmp(end, "ft", 2) == 0 || *end == '\'')) v *= 0.3048;
    return v;
}

static uint32_t ColorFromTags(const json& tags, uint64_t fallbackSeed, double heightM)
{
    // Hash building id → hue, modulated by height (taller = lighter). RGBA8.
    uint32_t h = (uint32_t)(fallbackSeed ^ (fallbackSeed >> 32)) * 2654435761u;
    float hue = (float)(h & 0xFFFu) / 4095.0f;
    float sat = 0.18f;
    float val = 0.55f + std::min(0.35f, (float)heightM / 300.0f);
    // tag-driven overrides
    if (tags.contains("building")) {
        std::string b = tags.at("building").get<std::string>();
        if (b == "industrial" || b == "warehouse") { sat = 0.10f; val *= 0.85f; }
        else if (b == "commercial" || b == "office") { sat = 0.12f; }
        else if (b == "residential" || b == "apartments" || b == "house") { sat = 0.20f; }
        else if (b == "church" || b == "cathedral") { sat = 0.30f; val = 0.7f; }
    }
    // HSV → RGB
    float r, g, bl;
    int   hi  = (int)std::floor(hue * 6.0f);
    float f   = hue * 6.0f - hi;
    float p   = val * (1.0f - sat);
    float q   = val * (1.0f - sat * f);
    float t   = val * (1.0f - sat * (1.0f - f));
    switch (hi % 6) {
        case 0: r = val; g = t;   bl = p;  break;
        case 1: r = q;   g = val; bl = p;  break;
        case 2: r = p;   g = val; bl = t;  break;
        case 3: r = p;   g = q;   bl = val;break;
        case 4: r = t;   g = p;   bl = val;break;
        default:r = val; g = p;   bl = q;  break;
    }
    auto q8 = [](float v) -> uint8_t { v = std::min(1.0f, std::max(0.0f, v)); return (uint8_t)(v * 255.0f + 0.5f); };
    return 0xFF000000u | q8(r) | (q8(g) << 8) | (q8(bl) << 16);
}

// ---------------- Line rasterizer with width ----------------
// Stamps a disk of radius r at sub-pixel-spaced points along each segment.
static void RasterizePolyline(const std::vector<std::pair<double,double>>& pts,
                              double widthVox,
                              std::vector<std::pair<int,int>>& outCells)
{
    if (pts.size() < 2) return;
    const double r = std::max(0.5, widthVox * 0.5);
    const int    R = (int)std::ceil(r);
    const double r2 = r * r;
    for (size_t i = 0; i + 1 < pts.size(); ++i) {
        double ax = pts[i].first,     az = pts[i].second;
        double bx = pts[i + 1].first, bz = pts[i + 1].second;
        double dx = bx - ax, dz = bz - az;
        double len = std::sqrt(dx * dx + dz * dz);
        if (len < 1e-6) continue;
        const double step = 0.5;       // sub-pixel along segment
        int   nSteps = (int)std::ceil(len / step) + 1;
        for (int s = 0; s <= nSteps; ++s) {
            double t = std::min(1.0, (double)s * step / len);
            double cx = ax + dx * t;
            double cz = az + dz * t;
            int x0 = (int)std::floor(cx - r), x1 = (int)std::ceil(cx + r);
            int z0 = (int)std::floor(cz - r), z1 = (int)std::ceil(cz + r);
            for (int z = z0; z <= z1; ++z) {
                for (int x = x0; x <= x1; ++x) {
                    double ddx = (x + 0.5) - cx;
                    double ddz = (z + 0.5) - cz;
                    if (ddx * ddx + ddz * ddz <= r2) outCells.emplace_back(x, z);
                }
            }
        }
        (void)R;
    }
}

// ---------------- Polygon scan-line rasterizer ----------------
// Emits integer (x,z) cells inside the closed polygon. Standard even-odd fill.
static void RasterizePolygon(const std::vector<std::pair<double,double>>& poly,
                             std::vector<std::pair<int,int>>& outCells)
{
    if (poly.size() < 3) return;
    // Bbox.
    double zMin = poly[0].second, zMax = poly[0].second;
    for (const auto& p : poly) { zMin = std::min(zMin, p.second); zMax = std::max(zMax, p.second); }
    int zLo = (int)std::floor(zMin);
    int zHi = (int)std::ceil (zMax);
    std::vector<double> xs;
    xs.reserve(poly.size());
    for (int z = zLo; z <= zHi; ++z) {
        xs.clear();
        double zs = z + 0.5;
        for (size_t i = 0; i < poly.size(); ++i) {
            const auto& a = poly[i];
            const auto& b = poly[(i + 1) % poly.size()];
            if (a.second == b.second) continue;                       // horizontal
            if ((a.second <= zs && b.second > zs) ||
                (b.second <= zs && a.second > zs)) {
                double t = (zs - a.second) / (b.second - a.second);
                xs.push_back(a.first + t * (b.first - a.first));
            }
        }
        std::sort(xs.begin(), xs.end());
        for (size_t i = 0; i + 1 < xs.size(); i += 2) {
            int x0 = (int)std::ceil (xs[i]     - 0.5);
            int x1 = (int)std::floor(xs[i + 1] - 0.5);
            for (int x = x0; x <= x1; ++x) outCells.emplace_back(x, z);
        }
    }
}

int main(int argc, char** argv)
{
    // Manhattan full island (Battery to Inwood, river to river).
    double lat0 = 40.700;
    double lon0 = -74.030;
    double lat1 = 40.880;
    double lon1 = -73.910;
    double metersPerVoxel = 2.0;
    int    chunkDim = 64;
    std::string outPath  = "assets/manhattan.vox";
    std::string cachePath = "assets/manhattan_overpass.json";
    double defaultLevelM = 3.0;          // assume 3 m per storey
    double defaultMissingH = 6.0;        // height assumed if no tag
    bool   wantGround = false;           // --ground: fill bbox with ground slab

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bbox") == 0 && i + 1 < argc) {
            std::sscanf(argv[++i], "%lf,%lf,%lf,%lf", &lat0, &lon0, &lat1, &lon1);
        } else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            outPath = argv[++i];
        } else if (std::strcmp(argv[i], "--scale") == 0 && i + 1 < argc) {
            metersPerVoxel = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) {
            chunkDim = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--cache") == 0 && i + 1 < argc) {
            cachePath = argv[++i];
        } else if (std::strcmp(argv[i], "--ground") == 0) {
            wantGround = true;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]); return 1;
        }
    }
    fs::create_directories(fs::path(outPath).parent_path());
    fs::create_directories(fs::path(cachePath).parent_path());

    std::printf("bbox lat[%.4f..%.4f] lon[%.4f..%.4f] m/voxel=%.1f\n",
                lat0, lat1, lon0, lon1, metersPerVoxel);

    if (!FetchOverpass(lat0, lon0, lat1, lon1, cachePath)) return 1;

    // Parse JSON.
    json doc;
    {
        std::ifstream f(cachePath);
        if (!f) { std::fprintf(stderr, "open %s failed\n", cachePath.c_str()); return 1; }
        f >> doc;
    }
    if (!doc.contains("elements")) { std::fprintf(stderr, "no elements field\n"); return 1; }

    // Collect nodes (lat/lon by id) + classified ways.
    enum class Kind : uint8_t { Building, Water, Park, Road, Railway, Skip };
    auto classify = [](const json& tags, Kind& outKind, double& outWidthM, uint32_t& outColor) {
        if (tags.contains("building")) {
            outKind = Kind::Building; outWidthM = 0; outColor = 0;
            return true;
        }
        if (tags.contains("natural") && tags["natural"].get<std::string>() == "water") {
            outKind = Kind::Water; outWidthM = 0; outColor = 0xFFB07640u; // ARGB-le -> blue-ish
            return true;
        }
        if (tags.contains("landuse")) {
            std::string lu = tags["landuse"].get<std::string>();
            if (lu == "grass" || lu == "meadow" || lu == "park" || lu == "recreation_ground"
                || lu == "cemetery" || lu == "farmland" || lu == "forest") {
                outKind = Kind::Park; outWidthM = 0;
                outColor = (lu == "forest") ? 0xFF2C5A30u : 0xFF4C8F44u;
                return true;
            }
        }
        if (tags.contains("leisure")) {
            std::string l = tags["leisure"].get<std::string>();
            if (l == "park" || l == "garden" || l == "pitch" || l == "playground" || l == "golf_course") {
                outKind = Kind::Park; outWidthM = 0; outColor = 0xFF4C8F44u;
                return true;
            }
        }
        if (tags.contains("waterway")) {
            std::string ww = tags["waterway"].get<std::string>();
            double w = 4.0;
            if      (ww == "river")  w = 80.0;   // Thames-sized
            else if (ww == "canal")  w = 18.0;
            else if (ww == "stream") w = 5.0;
            else if (ww == "drain" || ww == "ditch") w = 2.0;
            outKind = Kind::Water; outWidthM = w; outColor = 0xFFB07640u;
            return true;
        }
        if (tags.contains("railway")) {
            outKind = Kind::Railway; outWidthM = 4; outColor = 0xFF505860u;
            return true;
        }
        if (tags.contains("highway")) {
            std::string h = tags["highway"].get<std::string>();
            double w = 5.0; uint32_t col = 0xFF606060u;  // mid-grey asphalt
            if (h == "motorway" || h == "motorway_link" || h == "trunk" || h == "trunk_link") { w = 14; col = 0xFF505050u; }
            else if (h == "primary"   || h == "primary_link")   { w = 11; col = 0xFF585858u; }
            else if (h == "secondary" || h == "secondary_link") { w = 9;  col = 0xFF606060u; }
            else if (h == "tertiary"  || h == "tertiary_link")  { w = 7;  col = 0xFF686868u; }
            else if (h == "residential" || h == "unclassified" || h == "living_street") { w = 6; col = 0xFF707070u; }
            else if (h == "service") { w = 4; col = 0xFF787878u; }
            else if (h == "pedestrian" || h == "footway" || h == "path" || h == "track") { w = 3; col = 0xFF8A7A60u; }
            else if (h == "cycleway") { w = 3; col = 0xFF6A5848u; }
            else if (h == "steps") { return false; }
            else { w = 5; }
            outKind = Kind::Road; outWidthM = w; outColor = col;
            return true;
        }
        return false;
    };

    std::unordered_map<uint64_t, std::pair<double,double>> nodes;
    struct Way {
        uint64_t id;
        std::vector<uint64_t> nodeIds;
        Kind     kind;
        double   widthM;       // for line features
        double   heightM;      // building extrusion top
        double   minHeightM;
        uint32_t color;
        bool     closed;
    };
    std::vector<Way> ways;
    nodes.reserve(doc["elements"].size());
    ways.reserve(doc["elements"].size() / 2);

    for (const auto& e : doc["elements"]) {
        std::string type = e.value("type", std::string{});
        if (type == "node") {
            nodes[e["id"].get<uint64_t>()] = { e["lat"].get<double>(), e["lon"].get<double>() };
        } else if (type == "way") {
            json tags = e.value("tags", json::object());
            Kind k; double widthM; uint32_t col;
            if (!classify(tags, k, widthM, col)) continue;

            Way w;
            w.id = e["id"].get<uint64_t>();
            for (const auto& n : e["nodes"]) w.nodeIds.push_back(n.get<uint64_t>());
            if (w.nodeIds.size() < 2) continue;
            w.closed = (w.nodeIds.front() == w.nodeIds.back());
            w.kind   = k;
            w.widthM = widthM;
            w.heightM = 0; w.minHeightM = 0;
            w.color = col;

            if (k == Kind::Building) {
                double h = 0.0, mh = 0.0;
                if (tags.contains("height"))     h  = ParseLength(tags["height"].get<std::string>());
                if (tags.contains("min_height")) mh = ParseLength(tags["min_height"].get<std::string>());
                if (h <= 0.0 && tags.contains("building:levels")) {
                    h = std::atof(tags["building:levels"].get<std::string>().c_str()) * defaultLevelM;
                }
                if (h <= 0.0) h = defaultMissingH;
                w.heightM = h; w.minHeightM = mh;
                w.color   = ColorFromTags(tags, w.id, h);
            }
            ways.push_back(std::move(w));
        }
    }
    // ---- Process relations (natural=water multipolygons, e.g. Thames). ----
    // Build map of all ways encountered so far AND any raw member ways present
    // in the dump (they may not have natural=water tag — the relation does).
    std::unordered_map<uint64_t, std::vector<uint64_t>> wayNodes;
    wayNodes.reserve(doc["elements"].size());
    for (const auto& e : doc["elements"]) {
        if (e.value("type", std::string{}) != "way") continue;
        std::vector<uint64_t> nl;
        for (const auto& n : e["nodes"]) nl.push_back(n.get<uint64_t>());
        wayNodes[e["id"].get<uint64_t>()] = std::move(nl);
    }

    size_t nRelWaterPolys = 0;
    for (const auto& e : doc["elements"]) {
        if (e.value("type", std::string{}) != "relation") continue;
        json tags = e.value("tags", json::object());
        bool isWater = (tags.contains("natural") && tags["natural"].get<std::string>() == "water")
                    || (tags.contains("waterway") && tags["waterway"].get<std::string>() == "riverbank");
        if (!isWater) continue;
        if (!e.contains("members")) continue;

        // Collect outer member way ids.
        std::vector<uint64_t> outerWayIds;
        for (const auto& m : e["members"]) {
            std::string mt = m.value("type", std::string{});
            std::string mr = m.value("role", std::string{});
            if (mt != "way") continue;
            if (mr != "outer" && mr != "") continue;   // empty role often means outer
            outerWayIds.push_back(m["ref"].get<uint64_t>());
        }
        if (outerWayIds.empty()) continue;

        // Chain ways into closed rings. Each iteration picks an unused way,
        // walks until it closes back on itself (matching first node).
        std::vector<bool> used(outerWayIds.size(), false);
        for (size_t seed = 0; seed < outerWayIds.size(); ++seed) {
            if (used[seed]) continue;
            auto it0 = wayNodes.find(outerWayIds[seed]);
            if (it0 == wayNodes.end() || it0->second.size() < 2) { used[seed] = true; continue; }
            used[seed] = true;
            std::vector<uint64_t> ring = it0->second;

            bool grew = true;
            while (grew && ring.front() != ring.back()) {
                grew = false;
                for (size_t k = 0; k < outerWayIds.size(); ++k) {
                    if (used[k]) continue;
                    auto itK = wayNodes.find(outerWayIds[k]);
                    if (itK == wayNodes.end() || itK->second.size() < 2) { used[k] = true; continue; }
                    const auto& w2 = itK->second;
                    if (w2.front() == ring.back()) {
                        ring.insert(ring.end(), w2.begin() + 1, w2.end());
                        used[k] = true; grew = true; break;
                    }
                    if (w2.back() == ring.back()) {
                        ring.insert(ring.end(), w2.rbegin() + 1, w2.rend());
                        used[k] = true; grew = true; break;
                    }
                    if (w2.back() == ring.front()) {
                        ring.insert(ring.begin(), w2.begin(), w2.end() - 1);
                        used[k] = true; grew = true; break;
                    }
                    if (w2.front() == ring.front()) {
                        ring.insert(ring.begin(), w2.rbegin(), w2.rend() - 1);
                        used[k] = true; grew = true; break;
                    }
                }
            }
            if (ring.size() < 4 || ring.front() != ring.back()) continue;

            Way rw;
            rw.id        = e["id"].get<uint64_t>() ^ (seed * 0x9E3779B1ull);
            rw.nodeIds   = std::move(ring);
            rw.kind      = Kind::Water;
            rw.widthM    = 0;
            rw.heightM   = 0;
            rw.minHeightM= 0;
            rw.color     = 0xFFB07640u;
            rw.closed    = true;
            ways.push_back(std::move(rw));
            ++nRelWaterPolys;
        }
    }

    size_t nBldg=0, nRoad=0, nPark=0, nWat=0, nRail=0;
    for (const Way& w : ways) {
        switch (w.kind) {
            case Kind::Building: ++nBldg; break;
            case Kind::Road:     ++nRoad; break;
            case Kind::Park:     ++nPark; break;
            case Kind::Water:    ++nWat;  break;
            case Kind::Railway:  ++nRail; break;
            default: break;
        }
    }
    std::printf("Parsed: %zu nodes  ways: %zu bldg, %zu road, %zu park, %zu water (+%zu from relations), %zu rail\n",
                nodes.size(), nBldg, nRoad, nPark, nWat, nRelWaterPolys, nRail);

    // Equirectangular projection centred on bbox midpoint.
    const double midLat = 0.5 * (lat0 + lat1);
    const double mPerDegLat = 111320.0;
    const double mPerDegLon = 111320.0 * std::cos(midLat * 3.14159265358979 / 180.0);
    auto proj = [&](double lat, double lon, double& ox, double& oz) {
        ox = (lon - lon0) * mPerDegLon / metersPerVoxel;
        oz = (lat - lat0) * mPerDegLat / metersPerVoxel;   // +Z = north
    };

    // Rasterize each polygon → voxel columns.
    std::unordered_map<uint64_t, uint32_t> occColor;
    occColor.reserve(8 * 1024 * 1024);
    auto key3 = [](int x, int y, int z) -> uint64_t {
        return ((uint64_t)(uint32_t)x & 0x1FFFFF)
             | (((uint64_t)(uint32_t)y & 0x1FFFFF) << 21)
             | (((uint64_t)(uint32_t)z & 0x1FFFFF) << 42);
    };

    // Process in painter's order so later layers overwrite earlier:
    //   1. Water  (y=0 slab)
    //   2. Park   (y=0 slab)
    //   3. Road   (y=0 slab; widened polyline)
    //   4. Railway(y=0 slab; widened polyline)
    //   5. Building extrusion (y=[0..h])
    // Open ways for water (waterway tag) and roads/rail emit as lines with width;
    // closed ways for water/park emit as filled polygons.
    auto processWay = [&](const Way& w) {
        std::vector<std::pair<double,double>> pts;
        pts.reserve(w.nodeIds.size());
        for (uint64_t nid : w.nodeIds) {
            auto it = nodes.find(nid);
            if (it == nodes.end()) { pts.clear(); break; }
            double ox, oz; proj(it->second.first, it->second.second, ox, oz);
            pts.emplace_back(ox, oz);
        }
        if (pts.size() < 2) return;
        std::vector<std::pair<int,int>> cells;
        bool isLine = (w.kind == Kind::Road) || (w.kind == Kind::Railway)
                   || (w.kind == Kind::Water && !w.closed);
        if (isLine) {
            RasterizePolyline(pts, w.widthM / metersPerVoxel, cells);
        } else if (w.closed && pts.size() >= 3) {
            RasterizePolygon(pts, cells);
        } else {
            return;
        }
        if (cells.empty()) return;

        if (w.kind == Kind::Building) {
            int yLo = (int)std::round(w.minHeightM / metersPerVoxel);
            int yHi = (int)std::round(w.heightM    / metersPerVoxel);
            if (yHi <= yLo) yHi = yLo + 1;
            for (const auto& c : cells)
                for (int y = yLo; y < yHi; ++y)
                    occColor[key3(c.first, y, c.second)] = w.color;
        } else {
            // Ground-layer slab — 1 voxel thick at y=0.
            for (const auto& c : cells)
                occColor[key3(c.first, 0, c.second)] = w.color;
        }
    };

    auto pass = [&](Kind k) {
        size_t n = 0;
        for (const Way& w : ways) if (w.kind == k) { processWay(w); ++n; }
        return n;
    };
    std::printf("Rasterizing layers...\n");

    // Compute projected bbox in voxel coords (used by ground emit later).
    int groundX0 = 0, groundX1 = 0, groundZ0 = 0, groundZ1 = 0;
    {
        double x0d, z0d, x1d, z1d;
        proj(lat0, lon0, x0d, z1d);
        proj(lat1, lon1, x1d, z0d);
        groundX0 = (int)std::floor(std::min(x0d, x1d));
        groundX1 = (int)std::ceil (std::max(x0d, x1d));
        groundZ0 = (int)std::floor(std::min(z0d, z1d));
        groundZ1 = (int)std::ceil (std::max(z0d, z1d));
    }

    pass(Kind::Water);
    pass(Kind::Park);
    pass(Kind::Road);
    pass(Kind::Railway);
    pass(Kind::Building);
    std::printf("Total occupied voxels: %zu\n", occColor.size());
    if (occColor.empty()) {
        std::fprintf(stderr, "no voxels emitted — bbox may have no buildings\n"); return 1;
    }

    // visMask: 6-neighbour exposure test against the occupancy hash.
    // Faces: 0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z.
    static const int kDX[6] = { +1, -1, 0, 0, 0, 0 };
    static const int kDY[6] = {  0,  0, +1, -1, 0, 0 };
    static const int kDZ[6] = {  0,  0, 0, 0, +1, -1 };
    static const uint8_t kBit[6] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20 };

    struct Voxel { int32_t x, y, z; uint8_t mask; uint32_t color; };
    std::vector<Voxel> voxels;
    voxels.reserve(occColor.size());
    int32_t minX = INT32_MAX, minY = INT32_MAX, minZ = INT32_MAX;
    int32_t mxX  = INT32_MIN, mxY  = INT32_MIN, mxZ  = INT32_MIN;
    for (const auto& kv : occColor) {
        int x = (int)( kv.first        & 0x1FFFFF); if (x & 0x100000) x -= 0x200000;
        int y = (int)((kv.first >> 21) & 0x1FFFFF); if (y & 0x100000) y -= 0x200000;
        int z = (int)((kv.first >> 42) & 0x1FFFFF); if (z & 0x100000) z -= 0x200000;
        uint8_t mask = 0;
        for (int f = 0; f < 6; ++f) {
            if (!occColor.count(key3(x + kDX[f], y + kDY[f], z + kDZ[f]))) mask |= kBit[f];
        }
        if (mask == 0) continue;        // fully internal
        // Drop downward face (never seen from above, halves bottom voxel count).
        mask &= (uint8_t)~0x08u;
        if (mask == 0) continue;
        voxels.push_back({ x, y, z, mask, kv.second });
        minX = std::min(minX, x); mxX = std::max(mxX, x);
        minY = std::min(minY, y); mxY = std::max(mxY, y);
        minZ = std::min(minZ, z); mxZ = std::max(mxZ, z);
    }
    std::printf("Exposed surface voxels: %zu  bounds[%d..%d, %d..%d, %d..%d]\n",
                voxels.size(), minX, mxX, minY, mxY, minZ, mxZ);

    // If ground fill enabled, expand bounds to bbox so ground voxels in empty
    // areas don't get clipped by feature-only bounds.
    if (wantGround) {
        minX = std::min(minX, groundX0); mxX = std::max(mxX, groundX1 - 1);
        minZ = std::min(minZ, groundZ0); mxZ = std::max(mxZ, groundZ1 - 1);
        minY = std::min(minY, 0);        mxY = std::max(mxY, 0);
        std::printf("Bounds after ground bbox union: [%d..%d, %d..%d, %d..%d]\n",
                    minX, mxX, minY, mxY, minZ, mxZ);
    }

    // Palette.
    std::unordered_map<uint32_t, uint64_t> palCnt;
    for (const Voxel& v : voxels) ++palCnt[v.color];
    std::vector<std::pair<uint32_t,uint64_t>> palSorted(palCnt.begin(), palCnt.end());
    std::sort(palSorted.begin(), palSorted.end(),
              [](const auto& a, const auto& b){ return a.second > b.second; });
    if (palSorted.size() > 65535) {
        std::fprintf(stderr, "palette > 65535 (%zu) — too many unique colours, reduce diversity\n",
                     palSorted.size());
        return 1;
    }
    std::vector<uint32_t> palette;
    std::unordered_map<uint32_t, uint16_t> palMap;
    palette.reserve(palSorted.size());
    for (auto& p : palSorted) { palMap[p.first] = (uint16_t)palette.size(); palette.push_back(p.first); }
    std::printf("Palette: %zu unique colours\n", palette.size());

    // Chunk + write.
    using CK = std::pair<int, std::pair<int,int>>;
    struct CKHash { size_t operator()(const CK& k) const {
        size_t h = std::hash<int>{}(k.first);
        h = h * 1315423911u + std::hash<int>{}(k.second.first);
        h = h * 1315423911u + std::hash<int>{}(k.second.second);
        return h;
    }};
    // ---- AO bake (per face, 4-bit). 3x3 sample grid one step in front of each
    // visible face; occupancy via the existing occColor hash. Hits = occlusion.
    std::printf("Baking AO (per face)...\n");
    static const int kN[6][3] = {
        { +1, 0, 0 }, { -1, 0, 0 },
        { 0, +1, 0 }, { 0, -1, 0 },
        { 0, 0, +1 }, { 0, 0, -1 },
    };
    static const int kUaxis[6][3] = {
        { 0, 1, 0 }, { 0, 1, 0 },
        { 1, 0, 0 }, { 1, 0, 0 },
        { 1, 0, 0 }, { 1, 0, 0 },
    };
    static const int kVaxis[6][3] = {
        { 0, 0, 1 }, { 0, 0, 1 },
        { 0, 0, 1 }, { 0, 0, 1 },
        { 0, 1, 0 }, { 0, 1, 0 },
    };
    auto occHas = [&](int x, int y, int z) -> bool {
        return occColor.count(key3(x, y, z)) > 0;
    };
    auto faceAo4 = [&](int x, int y, int z, int fi) -> uint8_t {
        int hits = 0;
        for (int du = -1; du <= 1; ++du) {
            for (int dv = -1; dv <= 1; ++dv) {
                int sx = x + kN[fi][0] + du * kUaxis[fi][0] + dv * kVaxis[fi][0];
                int sy = y + kN[fi][1] + du * kUaxis[fi][1] + dv * kVaxis[fi][1];
                int sz = z + kN[fi][2] + du * kUaxis[fi][2] + dv * kVaxis[fi][2];
                if (occHas(sx, sy, sz)) ++hits;
            }
        }
        int ao4 = (int)std::round((1.0f - hits / 9.0f) * 15.0f);
        if (ao4 < 0) ao4 = 0; if (ao4 > 15) ao4 = 15;
        return (uint8_t)ao4;
    };

    std::unordered_map<CK, std::vector<DiskVoxel>, CKHash> chunks;
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
            uint8_t a4 = ((v.mask >> fi) & 1u) ? faceAo4(v.x, v.y, v.z, fi) : 15;
            int byte  = (fi * 4) >> 3;
            int shift = (fi * 4) & 7;
            dv.aoPacked[byte] |= (uint8_t)(a4 << shift);
        }
        chunks[ck].push_back(dv);
    }

    // Ground emit: dense 2D pass over bbox. Skip cells where a feature already
    // occupies y=0. Each emitted ground voxel has visMask = +Y only (top face).
    // Memory: 1 byte per cell of the bbox grid → ~414 MB for M25 at 3 m/voxel.
    if (wantGround) {
        const int W2 = groundX1 - groundX0;
        const int H2 = groundZ1 - groundZ0;
        if (W2 > 0 && H2 > 0) {
            std::printf("Ground emit: %d x %d cells...\n", W2, H2);
            // Need ground palette entry.
            const uint32_t groundCol = 0xFF606A60u;
            uint16_t groundIdx;
            auto pit = palMap.find(groundCol);
            if (pit == palMap.end()) {
                groundIdx = (uint16_t)palette.size();
                palette.push_back(groundCol);
                palMap[groundCol] = groundIdx;
            } else {
                groundIdx = pit->second;
            }

            size_t before = 0;
            for (auto& kv : chunks) before += kv.second.size();
            size_t emitted = 0;
            for (int z = groundZ0; z < groundZ1; ++z) {
                for (int x = groundX0; x < groundX1; ++x) {
                    // Skip if any feature voxel already sits at y=0 here.
                    if (occColor.count(key3(x, 0, z))) continue;
                    int rx = x - minX, rz = z - minZ;
                    if (rx < 0 || rz < 0) continue;
                    CK ck = { rx / chunkDim, { 0 / chunkDim, rz / chunkDim } };
                    DiskVoxel dv;
                    dv.x = (uint8_t)(rx % chunkDim);
                    dv.y = (uint8_t)(0  % chunkDim);
                    dv.z = (uint8_t)(rz % chunkDim);
                    dv.visMask    = 0x04;             // +Y top face only
                    dv.paletteIdx = groundIdx;
                    dv.aoPacked[0] = dv.aoPacked[1] = dv.aoPacked[2] = 0xFF;
                    chunks[ck].push_back(dv);
                    ++emitted;
                }
            }
            std::printf("Ground emit: +%zu voxels (was %zu, total %zu)\n",
                        emitted, before, before + emitted);
        }
    }
    std::printf("Chunks: %zu (chunkDim=%d)\n", chunks.size(), chunkDim);

    std::vector<ChunkMeta> metas;
    std::vector<DiskVoxel> flat;
    size_t total = 0;
    for (auto& kv : chunks) total += kv.second.size();
    flat.reserve(total);
    metas.reserve(chunks.size());
    for (auto& kv : chunks) {
        ChunkMeta m;
        m.cx = (uint16_t)kv.first.first;
        m.cy = (uint16_t)kv.first.second.first;
        m.cz = (uint16_t)kv.first.second.second;
        m._pad = 0;
        m.voxelCount  = (uint32_t)kv.second.size();
        m.voxelOffset = (uint32_t)flat.size();
        metas.push_back(m);
        for (auto& v : kv.second) flat.push_back(v);
    }

    FILE* f = std::fopen(outPath.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "open %s failed\n", outPath.c_str()); return 1; }
    const char magic[4] = { 'V','X','L','3' };
    std::fwrite(magic, 1, 4, f);
    uint32_t version = kAssetVersion;
    std::fwrite(&version, sizeof(uint32_t), 1, f);
    uint32_t cdim = (uint32_t)chunkDim;
    uint32_t ccount = (uint32_t)metas.size();
    uint32_t totalVox = (uint32_t)flat.size();
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
    std::fwrite(metas.data(), sizeof(ChunkMeta), metas.size(), f);
    std::fwrite(flat.data(),  sizeof(DiskVoxel), flat.size(), f);
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
