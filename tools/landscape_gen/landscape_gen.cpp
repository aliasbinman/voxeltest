// landscape_gen.cpp — procedural kidney-shaped island with mixed biomes.
//
// Pipeline:
//   1. Island mask (2-blob SDF + coastline noise) — kidney/bean shape.
//   2. Biome region map (mountain vs jungle) via low-freq noise.
//   3. Base heightmap: per-region elevation noise tapered by mask.
//   4. Hydraulic erosion (droplet sim).
//   5. Flow accumulation -> rivers; depression detection -> lakes.
//   6. Per-cell biome+color (ocean/river/lake/sand/grass/jungle/rock/snow).
//   7. Voxelize: surface column + cliff faces + ocean slab at y=0.
//   8. Multi-distance per-face AO bake (longer reach on +Y).
//   9. Quantize colors, chunk, write .vox + invalidate .vxb sidecar.

#define _CRT_SECURE_NO_WARNINGS
#define STB_PERLIN_IMPLEMENTATION
#include "stb_perlin.h"

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
#include <random>
#include <queue>

namespace fs = std::filesystem;

// ---------------- .vox on-disk types (mirror src/vox_loader.cpp) ----------------
#pragma pack(push, 1)
struct DiskVoxel { uint8_t x, y, z, visMask; uint16_t paletteIdx; uint8_t aoPacked[3]; };
#pragma pack(pop)
static_assert(sizeof(DiskVoxel) == 9, "");

struct ChunkMeta {
    uint16_t cx, cy, cz, _pad;
    uint32_t voxelCount;
    uint32_t voxelOffset;
};
static_assert(sizeof(ChunkMeta) == 16, "");

// ---------------- Noise + math helpers ----------------
static inline float Fbm(float x, float z, int oct, float lac = 2.0f, float gain = 0.5f) {
    return stb_perlin_fbm_noise3(x, 0.0f, z, lac, gain, oct);
}
static inline float Ridged(float x, float z, int oct) {
    float total = 0.0f, amp = 1.0f, freq = 1.0f, norm = 0.0f;
    for (int i = 0; i < oct; ++i) {
        float n = stb_perlin_noise3(x * freq, 0.0f, z * freq, 0, 0, 0);
        n = 1.0f - std::fabs(n);
        n = n * n;
        total += n * amp;
        norm  += amp;
        amp   *= 0.5f;
        freq  *= 2.0f;
    }
    return total / norm;
}
static inline float SmoothStep(float v, float a, float b) {
    float t = (v - a) / (b - a);
    if (t < 0.0f) t = 0.0f; if (t > 1.0f) t = 1.0f;
    return t * t * (3.0f - 2.0f * t);
}
static inline float Lerpf(float a, float b, float t) { return a + (b - a) * t; }
static inline float Clamp(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }

static inline uint32_t RGB(uint8_t r, uint8_t g, uint8_t b) {
    return 0xFF000000u | r | ((uint32_t)g << 8) | ((uint32_t)b << 16);
}
static inline uint32_t LerpRGB(uint32_t a, uint32_t b, float t) {
    t = Clamp(t, 0.0f, 1.0f);
    int ar=a&0xFF, ag=(a>>8)&0xFF, ab=(a>>16)&0xFF;
    int br=b&0xFF, bg=(b>>8)&0xFF, bb=(b>>16)&0xFF;
    return RGB((uint8_t)(ar + (br - ar) * t),
               (uint8_t)(ag + (bg - ag) * t),
               (uint8_t)(ab + (bb - ab) * t));
}
static inline uint32_t Quantize4(uint32_t c) {
    auto q = [](int v) -> int { v &= 0xFF; return (v >> 2) << 2; };
    return RGB((uint8_t)q(c & 0xFF), (uint8_t)q((c >> 8) & 0xFF), (uint8_t)q((c >> 16) & 0xFF));
}

// ---------------- Heightmap field helpers ----------------
struct Field {
    int W, H;
    std::vector<float> data;
    Field(int w, int h) : W(w), H(h), data((size_t)w * h, 0.0f) {}
    inline float  at(int x, int z) const { return data[(size_t)z * W + x]; }
    inline float& at(int x, int z)       { return data[(size_t)z * W + x]; }
    inline float  bilin(float x, float z) const {
        if (x < 0) x = 0; if (x > W - 1.001f) x = W - 1.001f;
        if (z < 0) z = 0; if (z > H - 1.001f) z = H - 1.001f;
        int ix = (int)x, iz = (int)z;
        float fx = x - ix, fz = z - iz;
        float h00 = at(ix,     iz);
        float h10 = at(ix + 1, iz);
        float h01 = at(ix,     iz + 1);
        float h11 = at(ix + 1, iz + 1);
        return Lerpf(Lerpf(h00, h10, fx), Lerpf(h01, h11, fx), fz);
    }
};

// ---------------- Analytical erosion filter (Runevision style) ----------------
// Per-cell, stateless, no simulation. Stacks oriented stripe noise across
// octaves; each octave's stripes are perpendicular to the (modulated) gradient,
// so they line up into dendritic gullies/ridges. Fast and embarrassingly
// parallel; runs entirely from local data.
//
// Caveats from the source: gullies are interpolated sines, not traced flow,
// so they can break partway down a slope. Run the depression-fill + river
// tracer afterwards if you need continuous drainage.
static inline float Hash01(int x, int z) {
    uint32_t h = (uint32_t)(x * 73856093) ^ (uint32_t)(z * 19349663);
    h = (h ^ (h >> 16)) * 2654435761u;
    h ^= h >> 16;
    return (float)(h & 0xFFFFFF) / (float)0x1000000;
}
static inline float Saturate(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }
static inline float EaseOut(float t)  { float s = 1.0f - Saturate(t); return 1.0f - s * s; }
static inline float PowInv(float t, float p) { return 1.0f - std::pow(1.0f - Saturate(t), p); }

static void ErosionFilter(Field& H, int octaves,
                          float baseCellSize,    // largest cell side in voxels
                          float gullyWeight,     // 0..1, ridge sharpness
                          float erosionAmp,      // scales gully height contribution
                          float detail)          // mask compounding exponent
{
    const int W = H.W, Z = H.H;
    // Sample h range for fadeTarget normalisation.
    float hMin = 1e30f, hMax = -1e30f;
    for (float v : H.data) { if (v < hMin) hMin = v; if (v > hMax) hMax = v; }
    const float hRange = std::max(1.0f, hMax - hMin);

    Field out(W, Z);

    auto computeGrad = [&](int x, int z, float& gx, float& gz) {
        int xm = std::max(0, x - 1), xp = std::min(W - 1, x + 1);
        int zm = std::max(0, z - 1), zp = std::min(Z - 1, z + 1);
        gx = (H.at(xp, z) - H.at(xm, z)) * 0.5f;
        gz = (H.at(x, zp) - H.at(x, zm)) * 0.5f;
    };

    const float twoPi = 6.28318530718f;

    #pragma omp parallel for schedule(dynamic, 32)
    for (int z = 0; z < Z; ++z) {
        for (int x = 0; x < W; ++x) {
            float h = H.at(x, z);
            float gx, gz; computeGrad(x, z, gx, gz);
            float fadeTarget = ((h - hMin) / hRange) * 2.0f - 1.0f;
            float combiMask  = 1.0f;
            float gradX = gx, gradZ = gz;
            float heightOut = h;
            float cellSize = baseCellSize;

            for (int n = 0; n < octaves; ++n) {
                // Anti-gradient direction (downhill).
                float gl = std::sqrt(gradX * gradX + gradZ * gradZ);
                if (gl < 1e-5f) gl = 1e-5f;
                float ngx = -gradX / gl, ngz = -gradZ / gl;
                // Perpendicular axis (along stripe).
                float perpX = -ngz, perpZ = ngx;

                // Worley cell for this point.
                float cx = x / cellSize, cz = z / cellSize;
                int icx = (int)std::floor(cx), icz = (int)std::floor(cz);

                // Accumulate stripe (cos, sin) from 3×3 neighbour cells.
                float accCos = 0.0f, accSin = 0.0f, accW = 0.0f;
                for (int dzi = -1; dzi <= 1; ++dzi) {
                    for (int dxi = -1; dxi <= 1; ++dxi) {
                        int nx = icx + dxi, nz = icz + dzi;
                        float px = (nx + Hash01(nx, nz)) * cellSize;
                        float pz = (nz + Hash01(nx + 977, nz + 31)) * cellSize;
                        float vx = (float)x - px, vz = (float)z - pz;
                        // Along-gradient distance -> stripe phase.
                        float d  = vx * ngx + vz * ngz;
                        // Perpendicular distance -> blend weight (further off the
                        // pivot's centreline contributes less).
                        float perpD = vx * perpX + vz * perpZ;
                        float w = std::exp(-(perpD * perpD) / (cellSize * cellSize));
                        float phase = d * twoPi / cellSize;
                        accCos += std::cos(phase) * w;
                        accSin += std::sin(phase) * w;
                        accW   += w;
                    }
                }
                if (accW > 1e-6f) { accCos /= accW; accSin /= accW; }

                // Magnitude clamp: prevent spikes when stripes constructively pile up.
                float m = std::sqrt(accCos * accCos + accSin * accSin);
                if (m > 0.5f) { float k = 0.5f / m; accCos *= k; accSin *= k; }

                // Straight-gully trick: use sign(sin) so branch angles don't curl.
                float slopeSign = accSin >= 0.0f ? 1.0f : -1.0f;

                // Slope magnitude estimate (driving the mask).
                float slopeMag = std::fabs(accSin);
                float newMask  = EaseOut(slopeMag * 2.0f);
                combiMask = PowInv(combiMask, detail) * newMask;

                // Mix toward fade target on flats, gully on ridges/creases.
                float gullyFaded = fadeTarget * (1.0f - combiMask) + accCos * combiMask;
                fadeTarget = gullyFaded;

                // Add to height. Erosion strength scales with cell size so big
                // ridges shape the silhouette more than fine ridges.
                heightOut += gullyFaded * erosionAmp * (cellSize / baseCellSize);

                // Bend gradient for next octave so finer gullies trace coarse ones.
                gradX += slopeSign * (-gradZ) * gullyWeight;
                gradZ += slopeSign * ( gradX) * gullyWeight;

                cellSize *= 0.5f;
            }
            out.at(x, z) = heightOut;
        }
    }
    H = std::move(out);
}

// ---------------- Erosion (droplet hydraulic) — kept for reference ----------------
static void ErodeHeightmap(Field& H, int numDroplets, std::mt19937& rng) {
    const int   maxLifetime    = 30;
    const float inertia        = 0.05f;
    const float capacityFactor = 4.0f;
    const float minSedimentCap = 0.01f;
    const float erodeSpeed     = 0.3f;
    const float depositSpeed   = 0.3f;
    const float evaporateSpeed = 0.01f;
    const float gravity        = 4.0f;
    const float startSpeed     = 1.0f;
    const float startWater     = 1.0f;

    std::uniform_real_distribution<float> ux(0.5f, (float)H.W - 1.5f);
    std::uniform_real_distribution<float> uz(0.5f, (float)H.H - 1.5f);

    for (int i = 0; i < numDroplets; ++i) {
        float px = ux(rng), pz = uz(rng);
        float vx = 0, vz = 0;
        float speed = startSpeed, water = startWater, sediment = 0.0f;
        for (int step = 0; step < maxLifetime; ++step) {
            int ix = (int)px, iz = (int)pz;
            float fx = px - ix, fz = pz - iz;
            if (ix < 0 || ix >= H.W - 1 || iz < 0 || iz >= H.H - 1) break;
            float h00 = H.at(ix, iz);
            float h10 = H.at(ix + 1, iz);
            float h01 = H.at(ix, iz + 1);
            float h11 = H.at(ix + 1, iz + 1);
            float gx = (h10 - h00) * (1 - fz) + (h11 - h01) * fz;
            float gz = (h01 - h00) * (1 - fx) + (h11 - h10) * fx;
            // Update direction (smooth blend).
            vx = vx * inertia - gx * (1 - inertia);
            vz = vz * inertia - gz * (1 - inertia);
            float vl = std::sqrt(vx * vx + vz * vz);
            if (vl < 1e-6f) break;
            vx /= vl; vz /= vl;
            float npx = px + vx, npz = pz + vz;
            float oldH = h00 * (1-fx) * (1-fz) + h10 * fx * (1-fz)
                       + h01 * (1-fx) * fz     + h11 * fx * fz;
            float newH;
            if (npx < 0 || npx >= H.W - 1 || npz < 0 || npz >= H.H - 1) break;
            else {
                int nix = (int)npx, niz = (int)npz;
                float nfx = npx - nix, nfz = npz - niz;
                float n00 = H.at(nix, niz);
                float n10 = H.at(nix + 1, niz);
                float n01 = H.at(nix, niz + 1);
                float n11 = H.at(nix + 1, niz + 1);
                newH = n00 * (1-nfx) * (1-nfz) + n10 * nfx * (1-nfz)
                     + n01 * (1-nfx) * nfz     + n11 * nfx * nfz;
            }
            float dh = newH - oldH;
            float cap = std::max(-dh * speed * water * capacityFactor, minSedimentCap);
            if (sediment > cap || dh > 0.0f) {
                float deposit = (dh > 0.0f) ? std::min(dh, sediment)
                                            : (sediment - cap) * depositSpeed;
                sediment -= deposit;
                H.at(ix,     iz)     += deposit * (1 - fx) * (1 - fz);
                H.at(ix + 1, iz)     += deposit * fx * (1 - fz);
                H.at(ix,     iz + 1) += deposit * (1 - fx) * fz;
                H.at(ix + 1, iz + 1) += deposit * fx * fz;
            } else {
                float erode = std::min((cap - sediment) * erodeSpeed, -dh);
                H.at(ix,     iz)     -= erode * (1 - fx) * (1 - fz);
                H.at(ix + 1, iz)     -= erode * fx * (1 - fz);
                H.at(ix,     iz + 1) -= erode * (1 - fx) * fz;
                H.at(ix + 1, iz + 1) -= erode * fx * fz;
                sediment += erode;
            }
            speed = std::sqrt(std::max(0.0f, speed * speed + dh * gravity));
            water *= (1 - evaporateSpeed);
            px = npx; pz = npz;
        }
    }
}

// ---------------- Flow accumulation (rivers) ----------------
// Per-cell sum of upstream cells. Sort cells by elevation desc, push 1 unit
// downhill to steepest 8-neighbour.
static void ComputeFlow(const Field& Hf, std::vector<float>& flowOut) {
    const int W = Hf.W, H = Hf.H;
    flowOut.assign((size_t)W * H, 1.0f);
    // Index list sorted by descending height (peaks first → drain downward).
    std::vector<int> idx((size_t)W * H);
    for (int i = 0; i < (int)idx.size(); ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return Hf.data[a] > Hf.data[b];
    });
    static const int dx[8] = { +1, +1, 0, -1, -1, -1, 0, +1 };
    static const int dz[8] = {  0, +1, +1, +1, 0, -1, -1, -1 };
    for (int i : idx) {
        int x = i % W, z = i / W;
        float h = Hf.data[i];
        int best = -1;
        float bestDH = 0.0f;
        for (int k = 0; k < 8; ++k) {
            int nx = x + dx[k], nz = z + dz[k];
            if (nx < 0 || nx >= W || nz < 0 || nz >= H) continue;
            float dh = h - Hf.at(nx, nz);
            float d  = (k & 1) ? 1.41421356f : 1.0f;
            float slope = dh / d;
            if (slope > bestDH) { bestDH = slope; best = nx + nz * W; }
        }
        if (best >= 0) flowOut[best] += flowOut[i];
    }
}

// ---------------- Main ----------------
int main(int argc, char** argv)
{
    int    seed   = 1337;
    float  worldM = 8192.0f;
    float  metersPerVoxel = 1.5f;
    int    chunkDim = 64;
    int    numDroplets = 200000;     // legacy, droplet erosion disabled by default
    int    filterOctaves = 6;
    float  filterCellSize = 120.0f;  // largest stripe cell, in voxels
    float  filterGullyWeight = 0.5f;
    float  filterErosionAmp  = 8.0f;
    float  filterDetail      = 1.6f;
    int    aoReach = 8;
    std::string outPath = "assets/landscape.vox";

    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--seed")     && i + 1 < argc) seed = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--size")     && i + 1 < argc) worldM = (float)std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--scale")    && i + 1 < argc) metersPerVoxel = (float)std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--chunk")    && i + 1 < argc) chunkDim = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--droplets") && i + 1 < argc) numDroplets = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--foct")     && i + 1 < argc) filterOctaves = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--fcell")    && i + 1 < argc) filterCellSize = (float)std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--fgully")   && i + 1 < argc) filterGullyWeight = (float)std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--famp")     && i + 1 < argc) filterErosionAmp = (float)std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--fdetail")  && i + 1 < argc) filterDetail = (float)std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--aoreach")  && i + 1 < argc) aoReach = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--out")      && i + 1 < argc) outPath = argv[++i];
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); return 1; }
    }
    fs::create_directories(fs::path(outPath).parent_path());

    const int W = (int)std::round(worldM / metersPerVoxel);
    const int H = W;
    const float seaLevelM = 4.0f;
    const int   seaLevelV = (int)std::round(seaLevelM / metersPerVoxel);

    std::printf("Landscape: %.0fm x %.0fm @ %.2f m/voxel -> %d x %d cells, seed=%d\n",
                worldM, worldM, metersPerVoxel, W, H, seed);

    // ---- 1. Island mask (kidney) ----
    // Two overlapping discs + low-freq noise distortion of the implicit edge.
    Field Mask(W, H);
    const float cxA = W * 0.42f, czA = H * 0.50f, rA = W * 0.30f;
    const float cxB = W * 0.62f, czB = H * 0.45f, rB = W * 0.24f;
    std::printf("Building island mask...\n");
    for (int z = 0; z < H; ++z) {
        for (int x = 0; x < W; ++x) {
            float fx = (float)x, fz = (float)z;
            float dA = std::sqrt((fx - cxA) * (fx - cxA) + (fz - czA) * (fz - czA)) / rA;
            float dB = std::sqrt((fx - cxB) * (fx - cxB) + (fz - czB) * (fz - czB)) / rB;
            // Carve a slight concavity between the two centres so the kidney bend is visible.
            float midX = 0.5f * (cxA + cxB), midZ = 0.5f * (czA + czB);
            float dM   = std::sqrt((fx - midX) * (fx - midX)
                                 + (fz - (midZ - rA * 0.45f)) * (fz - (midZ - rA * 0.45f))) / (rA * 0.55f);
            float dMin = std::min(dA, dB);
            // Subtract dM (closer = deeper concavity) from the mask.
            float carve = std::max(0.0f, 1.0f - dM) * 0.45f;
            float dEff  = dMin + carve;
            // Coastline noise so the shore isn't a perfect curve.
            float n = Fbm(fx * 0.004f, fz * 0.004f, 5) * 0.16f
                    + Fbm(fx * 0.020f, fz * 0.020f, 3) * 0.05f;
            dEff += n;
            Mask.at(x, z) = SmoothStep(1.0f - dEff, -0.04f, 0.12f);
        }
    }

    // ---- 2. Biome region map (mountain vs jungle) ----
    Field Region(W, H);   // 0 = mountain, 1 = jungle (smooth blend in middle)
    for (int z = 0; z < H; ++z) {
        for (int x = 0; x < W; ++x) {
            float fx = (float)x, fz = (float)z;
            // Big-scale noise so each region covers ~half the island.
            float r = Fbm(fx * 0.0005f, fz * 0.0005f, 3) * 0.5f + 0.5f;
            Region.at(x, z) = SmoothStep(r, 0.42f, 0.58f);
        }
    }

    // ---- 3. Base heightmap (more rugged) ----
    Field Hm(W, H);
    std::printf("Building heightmap...\n");
    for (int z = 0; z < H; ++z) {
        for (int x = 0; x < W; ++x) {
            float mask = Mask.at(x, z);
            if (mask < 0.001f) { Hm.at(x, z) = 0.0f; continue; }
            float region = Region.at(x, z);
            float fx = (float)x, fz = (float)z;
            // Mountain: big ridged peaks (up to ~620m) with sub-ridge detail.
            // Domain warp adds non-uniform direction so ranges don't line up.
            float warpX = Fbm(fx * 0.0015f, fz * 0.0015f, 3) * 120.0f;
            float warpZ = Fbm((fx + 1000.0f) * 0.0015f, (fz - 1000.0f) * 0.0015f, 3) * 120.0f;
            float mt1   = Ridged((fx + warpX) * 0.0009f, (fz + warpZ) * 0.0009f, 6);
            float mt2   = Ridged(fx * 0.0035f, fz * 0.0035f, 4);  // sub-ridges
            float mtH   = 70.0f + 540.0f * mt1 + 110.0f * mt2 * mt1;
            // Jungle: bumpy hills with ridged sub-detail for ravines.
            float jBase = Fbm(fx * 0.0030f, fz * 0.0030f, 5) * 0.5f + 0.5f;
            float jRid  = Ridged(fx * 0.008f, fz * 0.008f, 3);
            float jH    = 25.0f + 95.0f * jBase + 35.0f * jRid * jBase
                        + 8.0f * (Fbm(fx * 0.025f, fz * 0.025f, 3) * 0.5f + 0.5f);
            float landH = Lerpf(mtH, jH, region);
            // Coastal taper.
            float coast = SmoothStep(mask, 0.05f, 0.35f);
            landH = Lerpf(seaLevelM, landH, coast);
            Hm.at(x, z) = landH;
        }
    }

    // ---- 4. Erosion (Runevision analytical filter) ----
    std::printf("Erosion filter: %d octaves, cellSize=%.1f, gullyW=%.2f, amp=%.2f, detail=%.2f\n",
                filterOctaves, filterCellSize, filterGullyWeight, filterErosionAmp, filterDetail);
    ErosionFilter(Hm, filterOctaves, filterCellSize,
                  filterGullyWeight, filterErosionAmp, filterDetail);
    (void)numDroplets;

    // ---- 5. Depression fill (lakes) ----
    // Priority flood from the boundary: each cell's water level = max of its
    // own height and the lowest incoming water level. Where filled > orig the
    // basin is a lake; its surface sits at the spillover height.
    std::printf("Filling depressions...\n");
    Field Filled(W, H);
    {
        for (size_t i = 0; i < Filled.data.size(); ++i) Filled.data[i] = 1e30f;
        using QE = std::pair<float, int>;
        std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
        for (int z = 0; z < H; ++z) {
            for (int x = 0; x < W; ++x) {
                if (x == 0 || x == W - 1 || z == 0 || z == H - 1) {
                    float h = Hm.at(x, z);
                    Filled.at(x, z) = h;
                    pq.push({ h, z * W + x });
                }
            }
        }
        static const int d4x[4] = { -1, +1, 0, 0 };
        static const int d4z[4] = {  0, 0, -1, +1 };
        while (!pq.empty()) {
            auto top = pq.top(); pq.pop();
            float w = top.first; int idx = top.second;
            int x = idx % W, z = idx / W;
            if (Filled.data[idx] != w) continue;
            for (int k = 0; k < 4; ++k) {
                int nx = x + d4x[k], nz = z + d4z[k];
                if (nx < 0 || nx >= W || nz < 0 || nz >= H) continue;
                float nh   = Hm.at(nx, nz);
                float newW = std::max(w, nh);
                if (newW < Filled.at(nx, nz)) {
                    Filled.at(nx, nz) = newW;
                    pq.push({ newW, nz * W + nx });
                }
            }
        }
    }

    std::vector<uint8_t> isLake((size_t)W * H, 0);
    int lakeCells = 0;
    for (int z = 0; z < H; ++z) {
        for (int x = 0; x < W; ++x) {
            float orig = Hm.at(x, z);
            float fl   = Filled.at(x, z);
            // Only count above sea level (otherwise it's just ocean).
            if (orig > seaLevelM && fl > orig + 0.6f) {
                isLake[(size_t)z * W + x] = 1;
                ++lakeCells;
            }
        }
    }
    std::printf("Lake cells: %d\n", lakeCells);

    // ---- Flow accumulation on the *filled* heightmap so rivers can cross flat lake floors ----
    std::vector<float> flow;
    std::printf("Computing flow accumulation...\n");
    ComputeFlow(Filled, flow);

    // ---- River tracing + dilation ----
    // 1. Mark a cell as a river source if its accumulated flow exceeds threshold.
    // 2. Walk steepest descent from each source until reaching sea / existing river.
    // 3. Dilate marked cells by 1 to widen the channel to ~3 voxels.
    std::vector<uint8_t> isRiver((size_t)W * H, 0);
    {
        const float sourceFlow = std::max(60.0f, (float)(W * H) * 0.0005f);
        static const int d8x[8] = { +1, +1, 0, -1, -1, -1, 0, +1 };
        static const int d8z[8] = {  0, +1, +1, +1, 0, -1, -1, -1 };
        for (int z = 1; z < H - 1; ++z) {
            for (int x = 1; x < W - 1; ++x) {
                if (flow[(size_t)z * W + x] < sourceFlow) continue;
                int px = x, pz = z;
                for (int step = 0; step < W + H; ++step) {
                    size_t i = (size_t)pz * W + px;
                    if (isRiver[i]) break;
                    if (Mask.at(px, pz) < 0.001f) break;            // hit ocean
                    if (Hm.at(px, pz) <= seaLevelM + 0.2f) break;
                    isRiver[i] = 1;
                    float h = Filled.at(px, pz);
                    int   bx = px, bz = pz;
                    float bestSlope = 0.0f;
                    for (int k = 0; k < 8; ++k) {
                        int nx = px + d8x[k], nz = pz + d8z[k];
                        if (nx < 0 || nx >= W || nz < 0 || nz >= H) continue;
                        float dh = h - Filled.at(nx, nz);
                        float d  = (k & 1) ? 1.41421356f : 1.0f;
                        float slope = dh / d;
                        if (slope > bestSlope) { bestSlope = slope; bx = nx; bz = nz; }
                    }
                    if (bx == px && bz == pz) break;
                    px = bx; pz = bz;
                }
            }
        }
        // Dilate (1 cell) so river is ~3 voxels wide.
        std::vector<uint8_t> tmp = isRiver;
        for (int z = 1; z < H - 1; ++z) {
            for (int x = 1; x < W - 1; ++x) {
                if (tmp[(size_t)z * W + x]) continue;
                if (tmp[(size_t)(z - 1) * W + x] | tmp[(size_t)(z + 1) * W + x]
                  | tmp[(size_t)z * W + (x - 1)] | tmp[(size_t)z * W + (x + 1)]) {
                    isRiver[(size_t)z * W + x] = 1;
                }
            }
        }
        size_t riverCount = 0;
        for (uint8_t v : isRiver) if (v) ++riverCount;
        std::printf("River cells: %zu (source threshold %.0f)\n", riverCount, sourceFlow);
    }

    // ---- Apply lake water surface: set heightmap to fill level there. ----
    for (int z = 0; z < H; ++z) {
        for (int x = 0; x < W; ++x) {
            if (isLake[(size_t)z * W + x]) Hm.at(x, z) = Filled.at(x, z);
        }
    }

    // ---- Carve rivers (~1.5m incision) so channels are visible from above. ----
    for (int z = 0; z < H; ++z) {
        for (int x = 0; x < W; ++x) {
            if (!isRiver[(size_t)z * W + x]) continue;
            Hm.at(x, z) = std::max(seaLevelM, Hm.at(x, z) - 1.5f);
        }
    }

    // ---- 6. Per-cell biome + colour ----
    Field Slope(W, H);
    for (int z = 0; z < H; ++z) {
        for (int x = 0; x < W; ++x) {
            int xm = std::max(0, x - 1), xp = std::min(W - 1, x + 1);
            int zm = std::max(0, z - 1), zp = std::min(H - 1, z + 1);
            float dx = (Hm.at(xp, z) - Hm.at(xm, z)) * 0.5f;
            float dz = (Hm.at(x, zp) - Hm.at(x, zm)) * 0.5f;
            Slope.at(x, z) = std::sqrt(dx * dx + dz * dz);
        }
    }

    enum class CellKind : uint8_t { Ocean, Beach, Grass, Jungle, Rock, Snow, River, Lake };
    std::vector<uint8_t>  Kind((size_t)W * H, 0);
    std::vector<uint32_t> Color((size_t)W * H, 0);
    std::printf("Classifying biome + color...\n");
    for (int z = 0; z < H; ++z) {
        for (int x = 0; x < W; ++x) {
            float mask = Mask.at(x, z);
            float h    = Hm.at(x, z);
            float region = Region.at(x, z);
            float slope  = Slope.at(x, z);
            float f      = flow[(size_t)z * W + x];

            uint32_t col;
            CellKind k;
            (void)f;
            // Ocean: outside mask OR below sea level (eroded down).
            if (mask < 0.001f || h <= seaLevelM - 0.5f) {
                k = CellKind::Ocean;
                col = RGB(28, 60, 100);
                // Slight depth darkening based on how far inside ocean.
                float depthN = SmoothStep(1.0f - mask, 0.0f, 1.0f);
                col = LerpRGB(col, RGB(10, 30, 60), depthN);
            }
            else if (isLake[(size_t)z * W + x]) {
                k = CellKind::Lake; col = RGB(60, 110, 150);
            }
            else if (isRiver[(size_t)z * W + x]) {
                k = CellKind::River; col = RGB(70, 130, 175);
            }
            else if (h - seaLevelM < 4.0f) {
                k = CellKind::Beach; col = RGB(218, 200, 150);
            }
            else {
                bool mountainous = region < 0.5f;
                float snowLine = 280.0f;
                bool snow = (h > snowLine) ||
                            (mountainous && h > 200.0f && slope < 0.6f);
                if (snow) {
                    k = CellKind::Snow;
                    col = LerpRGB(RGB(220, 225, 230), RGB(250, 250, 252),
                                  SmoothStep(h, 200.0f, 380.0f));
                }
                else if (mountainous && (h > 140.0f || slope > 0.9f)) {
                    k = CellKind::Rock;
                    uint32_t darkRock  = RGB(80, 78, 76);
                    uint32_t lightRock = RGB(140, 138, 132);
                    col = LerpRGB(darkRock, lightRock, SmoothStep(h, 120.0f, 280.0f));
                    // Slope-driven darker rock on steep faces.
                    col = LerpRGB(col, darkRock, SmoothStep(slope, 0.8f, 2.0f) * 0.6f);
                }
                else if (region > 0.55f) {
                    k = CellKind::Jungle;
                    uint32_t dark = RGB(20, 70, 30);
                    uint32_t mid  = RGB(40, 110, 45);
                    uint32_t hi   = RGB(60, 140, 55);
                    float t1 = SmoothStep(h, 20.0f, 80.0f);
                    col = LerpRGB(dark, mid, t1);
                    col = LerpRGB(col, hi, SmoothStep(h, 80.0f, 140.0f));
                    // Small noisy mottling.
                    float n = Fbm((float)x * 0.03f, (float)z * 0.03f, 2);
                    col = LerpRGB(col, dark, SmoothStep(n, 0.2f, 0.5f) * 0.3f);
                }
                else {
                    k = CellKind::Grass;
                    uint32_t darkG = RGB(60, 105, 55);
                    uint32_t midG  = RGB(110, 155, 80);
                    uint32_t hiG   = RGB(140, 180, 100);
                    col = LerpRGB(darkG, midG, SmoothStep(h, 10.0f, 90.0f));
                    col = LerpRGB(col, hiG, SmoothStep(h, 90.0f, 160.0f));
                }
            }
            Kind[(size_t)z * W + x]  = (uint8_t)k;
            Color[(size_t)z * W + x] = Quantize4(col);
        }
    }

    // Beach widening: any land cell within 2 of an ocean and h < 6m → beach.
    {
        std::vector<uint8_t> isOcean((size_t)W * H, 0);
        for (size_t i = 0; i < Kind.size(); ++i)
            if ((CellKind)Kind[i] == CellKind::Ocean) isOcean[i] = 1;
        for (int z = 0; z < H; ++z) {
            for (int x = 0; x < W; ++x) {
                size_t i = (size_t)z * W + x;
                if (isOcean[i]) continue;
                float h = Hm.at(x, z);
                if (h - seaLevelM > 6.0f) continue;
                bool nearOcean = false;
                for (int dz = -2; dz <= 2 && !nearOcean; ++dz) {
                    for (int dx = -2; dx <= 2 && !nearOcean; ++dx) {
                        int nx = x + dx, nz = z + dz;
                        if (nx < 0 || nx >= W || nz < 0 || nz >= H) continue;
                        if (isOcean[(size_t)nz * W + nx]) nearOcean = true;
                    }
                }
                if (nearOcean) {
                    Kind[i]  = (uint8_t)CellKind::Beach;
                    Color[i] = Quantize4(RGB(218, 200, 150));
                }
            }
        }
    }

    // ---- 7. Voxelize ----
    auto Hv = [&](int x, int z) -> int {
        // Voxel height for the cell — ocean is exactly at seaLevelV.
        CellKind k = (CellKind)Kind[(size_t)z * W + x];
        if (k == CellKind::Ocean) return seaLevelV;
        float h = Hm.at(x, z);
        if (h < seaLevelM) h = seaLevelM;
        return (int)std::round(h / metersPerVoxel);
    };
    auto Cv = [&](int x, int z) -> uint32_t {
        return Color[(size_t)z * W + x];
    };

    struct Voxel { int32_t x, y, z; uint8_t mask; uint32_t color; };
    std::vector<Voxel> voxels;
    voxels.reserve((size_t)W * H * 2);
    static const int kDX[4] = { +1, -1, 0, 0 };
    static const int kDZ[4] = {  0, 0, +1, -1 };
    static const uint8_t kFace[4] = { 0x01, 0x02, 0x10, 0x20 };

    int hMinAll = INT32_MAX, hMaxAll = INT32_MIN;
    std::printf("Emitting voxels...\n");
    for (int z = 0; z < H; ++z) {
        for (int x = 0; x < W; ++x) {
            int y = Hv(x, z);
            uint32_t col = Cv(x, z);
            uint8_t mask = 0x04;
            for (int k = 0; k < 4; ++k) {
                int nx = x + kDX[k], nz = z + kDZ[k];
                int yn = (nx < 0 || nx >= W || nz < 0 || nz >= H) ? -1 : Hv(nx, nz);
                if (yn < y) mask |= kFace[k];
            }
            voxels.push_back({ x, y, z, mask, col });
            if (y < hMinAll) hMinAll = y;
            if (y > hMaxAll) hMaxAll = y;
            // Cliff voxels.
            for (int k = 0; k < 4; ++k) {
                int nx = x + kDX[k], nz = z + kDZ[k];
                int yn = (nx < 0 || nx >= W || nz < 0 || nz >= H) ? -1 : Hv(nx, nz);
                if (yn >= y - 1) continue;
                for (int yc = y - 1; yc > yn; --yc) {
                    voxels.push_back({ x, yc, z, kFace[k], col });
                }
            }
        }
        if ((z & 0x1FF) == 0) std::printf("  emit row %d / %d  (%zu voxels)\n", z, H, voxels.size());
    }
    std::printf("Raw: %zu voxels (Y range %d..%d)\n", voxels.size(), hMinAll, hMaxAll);

    // Dedupe + OR masks.
    {
        std::unordered_map<uint64_t, Voxel> uniq;
        uniq.reserve(voxels.size() * 2);
        for (const Voxel& v : voxels) {
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

    // ---- Palette ----
    std::unordered_map<uint32_t, uint64_t> palCnt;
    palCnt.reserve(8192);
    for (const Voxel& v : voxels) ++palCnt[v.color];
    std::vector<std::pair<uint32_t, uint64_t>> palSorted(palCnt.begin(), palCnt.end());
    std::sort(palSorted.begin(), palSorted.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    if (palSorted.size() > 65535) { std::fprintf(stderr, "palette too big\n"); return 1; }
    std::vector<uint32_t> palette;
    std::unordered_map<uint32_t, uint16_t> palMap;
    for (auto& p : palSorted) { palMap[p.first] = (uint16_t)palette.size(); palette.push_back(p.first); }
    std::printf("Palette: %zu unique colors\n", palette.size());

    // ---- Bounds ----
    int32_t minX = W, minY = INT32_MAX, minZ = H;
    int32_t mxX = 0, mxY = INT32_MIN, mxZ = 0;
    for (const Voxel& v : voxels) {
        minX = std::min(minX, v.x); mxX = std::max(mxX, v.x);
        minY = std::min(minY, v.y); mxY = std::max(mxY, v.y);
        minZ = std::min(minZ, v.z); mxZ = std::max(mxZ, v.z);
    }
    std::printf("Voxel bounds: [%d..%d, %d..%d, %d..%d]\n",
                minX, mxX, minY, mxY, minZ, mxZ);

    // ---- 8. AO with extended reach (heightmap occupancy) ----
    std::printf("Baking AO (reach=%d)...\n", aoReach);
    auto occupied = [&](int x, int y, int z) -> bool {
        if (x < 0 || x >= W || z < 0 || z >= H) return false;
        return y <= Hv(x, z) && y >= 0;
    };
    // Top-face AO: hemisphere over (+Y). Sample 8 compass directions at
    // distances 1, 2, 4, 8, aoReach. For each, count fraction occluded.
    auto topAo4 = [&](int x, int y, int z) -> uint8_t {
        static const int dx8[8] = { +1, +1, 0, -1, -1, -1, 0, +1 };
        static const int dz8[8] = {  0, +1, +1, +1, 0, -1, -1, -1 };
        const int rings[] = { 1, 2, 4, 8 };
        const int nRings = (int)(sizeof(rings) / sizeof(rings[0]));
        float occWeighted = 0.0f, totalWeighted = 0.0f;
        for (int ri = 0; ri < nRings; ++ri) {
            int r = rings[ri];
            if (r > aoReach) break;
            float w = 1.0f / (float)(ri + 1);   // closer rings weigh more
            for (int k = 0; k < 8; ++k) {
                int sx = x + dx8[k] * r;
                int sz = z + dz8[k] * r;
                if (sx < 0 || sx >= W || sz < 0 || sz >= H) { totalWeighted += w; continue; }
                int sh = Hv(sx, sz);
                // Apparent-elevation angle: occluder if neighbour is higher.
                if (sh > y) {
                    float dh = (float)(sh - y);
                    float dist = (float)r;
                    float tan = dh / dist;
                    // Soft occlusion (saturating).
                    float occ = SmoothStep(tan, 0.05f, 1.0f);
                    occWeighted += w * occ;
                }
                totalWeighted += w;
            }
        }
        float ao = (totalWeighted > 0.0f) ? (1.0f - occWeighted / totalWeighted) : 1.0f;
        int v = (int)std::round(ao * 15.0f);
        if (v < 0) v = 0; if (v > 15) v = 15;
        return (uint8_t)v;
    };
    // Side face AO: 3x3 sample one step ahead + 3 samples 4 steps ahead.
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
    auto sideAo4 = [&](int x, int y, int z, int fi) -> uint8_t {
        int hits = 0, total = 0;
        // Two distance shells: 1 and 4.
        const int dShells[2] = { 1, 4 };
        for (int s : dShells) {
            for (int du = -1; du <= 1; ++du) {
                for (int dv = -1; dv <= 1; ++dv) {
                    int sx = x + s * kN[fi][0] + du * kU[fi][0] + dv * kV[fi][0];
                    int sy = y + s * kN[fi][1] + du * kU[fi][1] + dv * kV[fi][1];
                    int sz = z + s * kN[fi][2] + du * kU[fi][2] + dv * kV[fi][2];
                    if (occupied(sx, sy, sz)) ++hits;
                    ++total;
                }
            }
        }
        int v = (int)std::round((1.0f - (float)hits / total) * 15.0f);
        if (v < 0) v = 0; if (v > 15) v = 15;
        return (uint8_t)v;
    };

    // ---- Chunk + write ----
    using CK = std::pair<int, std::pair<int,int>>;
    struct CKHash { size_t operator()(const CK& k) const {
        size_t h = std::hash<int>{}(k.first);
        h = h * 1315423911u + std::hash<int>{}(k.second.first);
        h = h * 1315423911u + std::hash<int>{}(k.second.second);
        return h;
    }};
    std::unordered_map<CK, std::vector<DiskVoxel>, CKHash> chunks;
    for (const Voxel& v : voxels) {
        int rx = v.x - minX, ry = v.y - minY, rz = v.z - minZ;
        CK ck = { rx / chunkDim, { ry / chunkDim, rz / chunkDim } };
        DiskVoxel dv;
        dv.x = (uint8_t)(rx % chunkDim);
        dv.y = (uint8_t)(ry % chunkDim);
        dv.z = (uint8_t)(rz % chunkDim);
        dv.visMask = v.mask & 0x3Fu;
        dv.paletteIdx = palMap[v.color];
        dv.aoPacked[0] = dv.aoPacked[1] = dv.aoPacked[2] = 0;
        for (int fi = 0; fi < 6; ++fi) {
            if (((v.mask >> fi) & 1u) == 0u) continue;
            uint8_t a4;
            if (fi == 2)      a4 = topAo4(v.x, v.y, v.z);          // +Y
            else if (fi == 3) a4 = 15;                              // -Y never seen
            else              a4 = sideAo4(v.x, v.y, v.z, fi);
            int byte  = (fi * 4) >> 3;
            int shift = (fi * 4) & 7;
            dv.aoPacked[byte] |= (uint8_t)(a4 << shift);
        }
        chunks[ck].push_back(dv);
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
    std::fwrite(flat.data(),  sizeof(DiskVoxel), flat.size(),  f);
    long bytes = std::ftell(f);
    std::fclose(f);
    std::printf("Wrote %s: %ld bytes (%.2f MB), %u voxels\n",
                outPath.c_str(), bytes, bytes / (1024.0 * 1024.0), totalVox);

    std::string vxb = outPath;
    size_t dot = vxb.find_last_of('.');
    if (dot != std::string::npos) vxb.resize(dot);
    vxb += ".vxb";
    std::error_code ec;
    if (fs::remove(vxb, ec)) std::printf("Removed stale baked sidecar: %s\n", vxb.c_str());
    return 0;
}
