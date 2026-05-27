// vox2vxw.cpp — convert a .vox (existing engine format) into a .vxw
// (clustered streamable world). Uses the existing vox_loader to parse the
// source file, then re-buckets voxels into 32³ clusters and emits the
// .vxw layout described in src/vxw.h.
//
// Usage:
//   vox2vxw.exe <in.vox> <out.vxw>
//
// Defaults the input to assets/kingslanding.vox if no args given.

#define _CRT_SECURE_NO_WARNINGS
#include "vxw.h"
#include "vox_loader.h"
#include "asset_version.h"
#include "lz4.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

// --------------------------------------------------------------------------

// Pack `bits` bits of `value` starting at bit `outBitPos` into `dst[]`.
// Caller manages outBitPos advancement.
static inline void BitPackPut(uint8_t* dst, uint64_t& outBitPos, uint32_t value, uint32_t bits)
{
    for (uint32_t i = 0; i < bits; ++i) {
        if ((value >> i) & 1u) {
            dst[outBitPos >> 3] |= (uint8_t)(1u << (outBitPos & 7));
        }
        ++outBitPos;
    }
}

// Re-derive the voxel coord (within cluster, 0..N-1) at given LOD step from the
// voxel's full chunk-local coord. For aggregating L1..L3 from L0.
struct LodAccum { uint32_t r=0, g=0, b=0, ao=0; uint8_t mask=0; uint16_t count=0; };

// Bake one cluster's per-LOD blob from the gathered L0 voxels. Writes one raw
// (uncompressed) blob into `outRaw` and returns the number of voxels at that LOD.
struct VoxelEntry {
    uint8_t lx, ly, lz;        // local 0..31 inside cluster
    uint8_t mask;
    uint16_t paletteIdx;
    uint8_t aoPacked[3];       // already 24-bit packed (4 bits per face)
    uint32_t color;            // RGBA, used for averaging at coarser LODs
};

static uint32_t BuildBlobL0(const std::vector<VoxelEntry>& voxels,
                            bool palette16bit,
                            std::vector<uint8_t>& outRaw)
{
    const uint32_t n = (uint32_t)voxels.size();
    if (n == 0) return 0;
    outRaw.clear();
    const uint32_t posBytes = vxw::PosBytes(n, 0);
    const uint32_t mskBytes = vxw::MaskBytes(n);
    const uint32_t aoBytes  = vxw::AoBytes(n);
    const uint32_t palBytes = vxw::PalIdxBytes(n, palette16bit);
    outRaw.assign(posBytes + mskBytes + aoBytes + palBytes, 0);

    uint8_t* posDst  = outRaw.data();
    uint8_t* mskDst  = posDst + posBytes;
    uint8_t* aoDst   = mskDst + mskBytes;
    uint8_t* palDst  = aoDst  + aoBytes;

    uint64_t posBit = 0, mskBit = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const VoxelEntry& v = voxels[i];
        BitPackPut(posDst, posBit, v.lx, 5);
        BitPackPut(posDst, posBit, v.ly, 5);
        BitPackPut(posDst, posBit, v.lz, 5);
        BitPackPut(mskDst, mskBit, v.mask & 0x3Fu, 6);
        aoDst[i*3 + 0] = v.aoPacked[0];
        aoDst[i*3 + 1] = v.aoPacked[1];
        aoDst[i*3 + 2] = v.aoPacked[2];
        if (palette16bit) {
            palDst[i*2 + 0] = (uint8_t)(v.paletteIdx & 0xFF);
            palDst[i*2 + 1] = (uint8_t)((v.paletteIdx >> 8) & 0xFF);
        } else {
            palDst[i] = (uint8_t)v.paletteIdx;
        }
    }
    return n;
}

// Aggregate L0 voxels into an LOD step (step = 2/4/8) and emit the blob.
static uint32_t BuildBlobLodN(const std::vector<VoxelEntry>& l0,
                              const std::vector<uint32_t>& palette,
                              uint32_t lod,
                              uint32_t step,
                              std::unordered_map<uint32_t, uint16_t>& palMap,
                              bool palette16bit,
                              std::vector<uint8_t>& outRaw)
{
    if (l0.empty()) return 0;
    // Bin voxels by (lx/step, ly/step, lz/step). Average color+ao, OR mask.
    std::unordered_map<uint32_t, LodAccum> bins;
    bins.reserve(l0.size());
    for (const VoxelEntry& v : l0) {
        uint32_t sx = v.lx / step;
        uint32_t sy = v.ly / step;
        uint32_t sz = v.lz / step;
        uint32_t key = sx | (sy << 4) | (sz << 8);   // 4 bits/axis fits step>=2
        LodAccum& a = bins[key];
        a.r += (v.color >>  0) & 0xFF;
        a.g += (v.color >>  8) & 0xFF;
        a.b += (v.color >> 16) & 0xFF;
        // Use the max AO across faces for the aggregated voxel
        uint32_t maxAo = 0;
        for (int f = 0; f < 6; ++f) {
            uint32_t n4 = (v.aoPacked[(f*4)>>3] >> ((f*4)&7)) & 0xFu;
            if (n4 > maxAo) maxAo = n4;
        }
        a.ao += maxAo;
        a.count++;
        a.mask |= v.mask;
    }

    const uint32_t n = (uint32_t)bins.size();
    const uint32_t posBytes = vxw::PosBytes(n, lod);
    const uint32_t mskBytes = vxw::MaskBytes(n);
    const uint32_t aoBytes  = vxw::AoBytes(n);
    const uint32_t palBytes = vxw::PalIdxBytes(n, palette16bit);
    outRaw.assign(posBytes + mskBytes + aoBytes + palBytes, 0);

    uint8_t* posDst = outRaw.data();
    uint8_t* mskDst = posDst + posBytes;
    uint8_t* aoDst  = mskDst + mskBytes;
    uint8_t* palDst = aoDst  + aoBytes;

    uint64_t posBit = 0, mskBit = 0;
    const uint32_t bpa = vxw::kBitsPerAxis[lod];
    uint32_t idx = 0;
    for (auto& kv : bins) {
        uint32_t key = kv.first;
        const LodAccum& a = kv.second;
        uint32_t sx = key & 0xF;
        uint32_t sy = (key >> 4) & 0xF;
        uint32_t sz = (key >> 8) & 0xF;
        BitPackPut(posDst, posBit, sx, bpa);
        BitPackPut(posDst, posBit, sy, bpa);
        BitPackPut(posDst, posBit, sz, bpa);
        BitPackPut(mskDst, mskBit, a.mask & 0x3Fu, 6);
        // Aggregated AO: same value on all 6 faces (we lost per-face data).
        uint32_t avgAo4 = (uint32_t)((a.ao / a.count) & 0xFu);
        uint32_t ao6 = 0;
        for (int f = 0; f < 6; ++f) ao6 |= avgAo4 << (f * 4);
        aoDst[idx*3 + 0] = (uint8_t)(ao6 & 0xFF);
        aoDst[idx*3 + 1] = (uint8_t)((ao6 >> 8) & 0xFF);
        aoDst[idx*3 + 2] = (uint8_t)((ao6 >> 16) & 0xFF);
        // Averaged color -> nearest palette entry.
        uint32_t avgR = a.r / a.count;
        uint32_t avgG = a.g / a.count;
        uint32_t avgB = a.b / a.count;
        uint32_t avgRGB = (avgR & 0xFF) | ((avgG & 0xFF) << 8) | ((avgB & 0xFF) << 16);
        uint16_t palIdx = 0;
        auto it = palMap.find(avgRGB);
        if (it == palMap.end()) {
            // Nearest-palette quantize (linear scan). Acceptable: palettes <= few hundred.
            uint32_t bestD = 0xFFFFFFFFu;
            for (uint32_t p = 0; p < palette.size(); ++p) {
                uint32_t c = palette[p];
                int dr = (int)((c >>  0) & 0xFF) - (int)avgR;
                int dg = (int)((c >>  8) & 0xFF) - (int)avgG;
                int db = (int)((c >> 16) & 0xFF) - (int)avgB;
                uint32_t d = (uint32_t)(dr*dr + dg*dg + db*db);
                if (d < bestD) { bestD = d; palIdx = (uint16_t)p; }
            }
            palMap[avgRGB] = palIdx;
        } else {
            palIdx = it->second;
        }
        if (palette16bit) {
            palDst[idx*2 + 0] = (uint8_t)(palIdx & 0xFF);
            palDst[idx*2 + 1] = (uint8_t)((palIdx >> 8) & 0xFF);
        } else {
            palDst[idx] = (uint8_t)palIdx;
        }
        ++idx;
    }
    return n;
}

// --------------------------------------------------------------------------

int main(int argc, char** argv)
{
    std::string inPath  = "assets/kingslanding.vox";
    std::string outPath = "assets/kingslanding.vxw";
    if (argc >= 2) inPath  = argv[1];
    if (argc >= 3) outPath = argv[2];

    std::printf("Reading %s...\n", inPath.c_str());

    // Reuse the existing loader to ingest the .vox into a Scene. We discard
    // its LOD aggregates (we'll re-bake) but keep L0 pointVertices+pointAo6.
    Scene scene;
    std::string err;
    if (!LoadVoxScene(inPath.c_str(), scene, err)) {
        std::fprintf(stderr, "vox load failed: %s\n", err.c_str());
        return 1;
    }
    std::printf("Loaded: %zu point entries, %zu subs, palette histogram %zu unique\n",
                scene.pointVertices.size(), scene.subs.size(), scene.colorHistogram.size());

    // Walk only the L0 entries of pointVertices (per subs[].pointFirst/pointCount)
    // and bucket by cluster grid using world voxel coords.
    // World coord = chunkBase + (px - chunkBase) ... actually pointVertices[i].px
    // already stores scene-relative coords (uint16 across whole scene). We need
    // ABSOLUTE world voxel coords; that's exactly what px/py/pz are.

    // Build palette from scene.colorHistogram (most-common first). We re-index
    // so the .vxw palette only contains colours actually used at L0.
    std::vector<uint32_t> palette;
    std::unordered_map<uint32_t, uint16_t> palMap;
    palette.reserve(scene.colorHistogram.size());
    for (const auto& e : scene.colorHistogram) {
        uint32_t rgb = e.first | 0xFF000000u;
        palMap[rgb & 0x00FFFFFFu] = (uint16_t)palette.size();
        palette.push_back(rgb);
    }
    bool palette16bit = palette.size() > 256;
    std::printf("Palette: %zu unique, %d-bit indices\n",
                palette.size(), palette16bit ? 16 : 8);

    // Bucket L0 voxels into clusters.
    // Cluster grid coord = floor(voxWorld / 32). Cluster local coord = voxWorld & 31.
    // (For negative coords floor is bit-shift -> arithmetic shift right by 5.)
    std::unordered_map<uint64_t, std::vector<VoxelEntry>> clusterBins;
    clusterBins.reserve(scene.subs.size() * 8);

    auto packCK = [](int32_t cx, int32_t cy, int32_t cz) -> uint64_t {
        // 21 bits/axis signed -> fits comfortably for any realistic scene.
        return  ((uint64_t)(uint32_t)cx & 0x1FFFFFu)
             | (((uint64_t)(uint32_t)cy & 0x1FFFFFu) << 21)
             | (((uint64_t)(uint32_t)cz & 0x1FFFFFu) << 42);
    };

    size_t l0Visited = 0;
    for (const SubMesh& s : scene.subs) {
        const uint32_t end = s.pointFirst + s.pointCount;
        for (uint32_t i = s.pointFirst; i < end; ++i) {
            const Vertex& v = scene.pointVertices[i];
            int32_t wx = (int32_t)v.px;
            int32_t wy = (int32_t)v.py;
            int32_t wz = (int32_t)v.pz;
            int32_t cx = (wx >= 0) ? (wx / 32) : ((wx - 31) / 32);
            int32_t cy = (wy >= 0) ? (wy / 32) : ((wy - 31) / 32);
            int32_t cz = (wz >= 0) ? (wz / 32) : ((wz - 31) / 32);
            uint8_t lx = (uint8_t)(wx - cx * 32);
            uint8_t ly = (uint8_t)(wy - cy * 32);
            uint8_t lz = (uint8_t)(wz - cz * 32);

            VoxelEntry ve;
            ve.lx = lx; ve.ly = ly; ve.lz = lz;
            ve.mask = (uint8_t)((v.color >> 24) & 0x3Fu);
            uint32_t rgb = v.color & 0x00FFFFFFu;
            auto it = palMap.find(rgb);
            ve.paletteIdx = (it != palMap.end()) ? it->second : 0;
            // ao6 is stored parallel in scene.pointAo6 (24-bit packed in low bits).
            uint32_t ao6 = scene.pointAo6[i];
            ve.aoPacked[0] = (uint8_t)(ao6 & 0xFF);
            ve.aoPacked[1] = (uint8_t)((ao6 >> 8) & 0xFF);
            ve.aoPacked[2] = (uint8_t)((ao6 >> 16) & 0xFF);
            ve.color = v.color & 0x00FFFFFFu;
            clusterBins[packCK(cx, cy, cz)].push_back(ve);
            ++l0Visited;
        }
    }
    std::printf("L0 voxels bucketed: %zu into %zu clusters\n", l0Visited, clusterBins.size());

    // Per cluster: enforce cap, build per-LOD blobs, store in a sortable struct.
    struct ClusterBake {
        int32_t cx, cy, cz;
        std::vector<uint8_t> blobs[vxw::kLodCount];      // raw bytes (pre-LZ4)
        bool                 lz4 [vxw::kLodCount] = { false, false, false, false };
        std::vector<uint8_t> lz4Buf[vxw::kLodCount];     // compressed bytes if used
        uint16_t             voxelCount[vxw::kLodCount] = {};
    };
    std::vector<ClusterBake> bakes;
    bakes.reserve(clusterBins.size());

    size_t cappedClusters = 0;
    for (auto& kv : clusterBins) {
        std::vector<VoxelEntry>& vs = kv.second;
        if (vs.size() > vxw::kMaxVoxAtLod[0]) {
            ++cappedClusters;
            vs.resize(vxw::kMaxVoxAtLod[0]);     // drop overflow (warn aggregate below)
        }
        // Unpack cluster grid coord from key.
        uint64_t k = kv.first;
        auto unpackS21 = [](uint64_t v, int shift) -> int32_t {
            uint32_t u = (uint32_t)((v >> shift) & 0x1FFFFFu);
            return (u & 0x100000u) ? (int32_t)(u | 0xFFE00000u) : (int32_t)u;
        };
        ClusterBake cb;
        cb.cx = unpackS21(k, 0);
        cb.cy = unpackS21(k, 21);
        cb.cz = unpackS21(k, 42);
        cb.voxelCount[0] = (uint16_t)BuildBlobL0(vs, palette16bit, cb.blobs[0]);
        cb.voxelCount[1] = (uint16_t)BuildBlobLodN(vs, palette, 1, 2, palMap, palette16bit, cb.blobs[1]);
        cb.voxelCount[2] = (uint16_t)BuildBlobLodN(vs, palette, 2, 4, palMap, palette16bit, cb.blobs[2]);
        cb.voxelCount[3] = (uint16_t)BuildBlobLodN(vs, palette, 3, 8, palMap, palette16bit, cb.blobs[3]);
        // Compress LZ4 selectively: L0/L1 typically, L2/L3 raw.
        for (uint32_t lod = 0; lod < vxw::kLodCount; ++lod) {
            if (cb.voxelCount[lod] == 0) continue;
            const auto& raw = cb.blobs[lod];
            if (lod <= 1 && raw.size() >= 256) {
                int bound = LZ4_compressBound((int)raw.size());
                cb.lz4Buf[lod].resize((size_t)bound);
                int cs = LZ4_compress_default(
                    (const char*)raw.data(), (char*)cb.lz4Buf[lod].data(),
                    (int)raw.size(), bound);
                if (cs > 0 && (size_t)cs < raw.size()) {
                    cb.lz4Buf[lod].resize((size_t)cs);
                    cb.lz4[lod] = true;
                } else {
                    cb.lz4Buf[lod].clear();
                }
            }
        }
        bakes.push_back(std::move(cb));
    }
    if (cappedClusters) {
        std::printf("WARN: %zu clusters had >%u L0 voxels and were truncated.\n",
                    cappedClusters, vxw::kMaxVoxAtLod[0]);
    }

    // Group clusters by region.
    struct RegionBake {
        int32_t rx, ry, rz;
        std::vector<uint32_t> clusterIdx;     // indices into `bakes`, ordered by local slot
        uint8_t clusterMask[64] = {};
    };
    std::unordered_map<uint64_t, RegionBake> regionBins;
    for (uint32_t i = 0; i < bakes.size(); ++i) {
        const ClusterBake& cb = bakes[i];
        int32_t rx, ry, rz; vxw::RegionOf(cb.cx, cb.cy, cb.cz, rx, ry, rz);
        uint64_t rk = packCK(rx, ry, rz);
        RegionBake& rb = regionBins[rk];
        rb.rx = rx; rb.ry = ry; rb.rz = rz;
        uint32_t lx = (uint32_t)(cb.cx - rx * (int32_t)vxw::kClustersPerRegionAx);
        uint32_t ly = (uint32_t)(cb.cy - ry * (int32_t)vxw::kClustersPerRegionAx);
        uint32_t lz = (uint32_t)(cb.cz - rz * (int32_t)vxw::kClustersPerRegionAx);
        uint32_t slot = vxw::LocalClusterSlot(lx, ly, lz);
        rb.clusterMask[slot >> 3] |= (uint8_t)(1u << (slot & 7));
        rb.clusterIdx.push_back(i);
    }
    // Sort each region's clusters by local slot order so the on-disk entries
    // match the order produced by iterating clusterMask bits ascending.
    for (auto& kv : regionBins) {
        RegionBake& rb = kv.second;
        std::sort(rb.clusterIdx.begin(), rb.clusterIdx.end(),
                  [&](uint32_t a, uint32_t b) {
                      const ClusterBake& A = bakes[a];
                      const ClusterBake& B = bakes[b];
                      uint32_t la = vxw::LocalClusterSlot(
                          (uint32_t)(A.cx - rb.rx * 8),
                          (uint32_t)(A.cy - rb.ry * 8),
                          (uint32_t)(A.cz - rb.rz * 8));
                      uint32_t lb = vxw::LocalClusterSlot(
                          (uint32_t)(B.cx - rb.rx * 8),
                          (uint32_t)(B.cy - rb.ry * 8),
                          (uint32_t)(B.cz - rb.rz * 8));
                      return la < lb;
                  });
    }
    std::printf("Regions: %zu\n", regionBins.size());

    // Compute each region's payload bytes (ClusterEntry array + concatenated blobs).
    std::vector<RegionBake*> regionsOrdered;
    regionsOrdered.reserve(regionBins.size());
    for (auto& kv : regionBins) regionsOrdered.push_back(&kv.second);
    std::sort(regionsOrdered.begin(), regionsOrdered.end(),
              [](const RegionBake* a, const RegionBake* b) {
                  if (a->rz != b->rz) return a->rz < b->rz;
                  if (a->ry != b->ry) return a->ry < b->ry;
                  return a->rx < b->rx;
              });

    // Pre-compute payload offsets per region.
    std::vector<uint64_t> regionPayloadOffsets(regionsOrdered.size(), 0);
    std::vector<uint32_t> regionPayloadSizes  (regionsOrdered.size(), 0);
    for (size_t ri = 0; ri < regionsOrdered.size(); ++ri) {
        const RegionBake& rb = *regionsOrdered[ri];
        uint64_t bytes = (uint64_t)rb.clusterIdx.size() * sizeof(vxw::ClusterEntry);
        for (uint32_t ci : rb.clusterIdx) {
            const ClusterBake& cb = bakes[ci];
            for (uint32_t lod = 0; lod < vxw::kLodCount; ++lod) {
                if (cb.voxelCount[lod] == 0) continue;
                bytes += cb.lz4[lod] ? cb.lz4Buf[lod].size() : cb.blobs[lod].size();
            }
        }
        regionPayloadSizes[ri] = (uint32_t)bytes;
    }

    // ---- Write file -------------------------------------------------------
    FILE* f = std::fopen(outPath.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "open %s failed\n", outPath.c_str()); return 1; }

    // Header.
    vxw::Header h{};
    std::memcpy(h.magic, vxw::kMagic, 4);
    h.version = kAssetVersion;
    h.metersPerVoxelQ16 = 0x10000u;      // 1.0 m per voxel — Kings Landing format default
    h.originVoxel[0] = scene.origin[0];
    h.originVoxel[1] = scene.origin[1];
    h.originVoxel[2] = scene.origin[2];
    h.clusterVoxAxis = vxw::kClusterVoxAxis;
    h.paletteCount   = (uint32_t)palette.size();
    std::fwrite(&h, sizeof(h), 1, f);
    if (h.paletteCount) std::fwrite(palette.data(), sizeof(uint32_t), h.paletteCount, f);

    // payloadStart placeholder; we'll patch after writing region table.
    uint64_t payloadStart = 0;
    std::fwrite(&payloadStart, sizeof(uint64_t), 1, f);
    uint32_t regionCount = (uint32_t)regionsOrdered.size();
    std::fwrite(&regionCount, sizeof(uint32_t), 1, f);
    long regionTableOffset = std::ftell(f);
    std::vector<vxw::RegionEntry> regionEntries(regionCount);
    std::fwrite(regionEntries.data(), sizeof(vxw::RegionEntry), regionCount, f);

    payloadStart = (uint64_t)std::ftell(f);
    // Write each region payload.
    for (size_t ri = 0; ri < regionsOrdered.size(); ++ri) {
        const RegionBake& rb = *regionsOrdered[ri];
        regionPayloadOffsets[ri] = (uint64_t)std::ftell(f);
        // ClusterEntry array.
        std::vector<vxw::ClusterEntry> entries(rb.clusterIdx.size());
        uint32_t cursor = (uint32_t)(rb.clusterIdx.size() * sizeof(vxw::ClusterEntry));
        for (size_t k = 0; k < rb.clusterIdx.size(); ++k) {
            const ClusterBake& cb = bakes[rb.clusterIdx[k]];
            vxw::ClusterEntry& ce = entries[k];
            for (uint32_t lod = 0; lod < vxw::kLodCount; ++lod) {
                ce.voxelCount[lod] = cb.voxelCount[lod];
                if (cb.voxelCount[lod] == 0) {
                    ce.blobOffset[lod] = 0;
                    ce.blobSize[lod]   = 0;
                    continue;
                }
                const std::vector<uint8_t>& body = cb.lz4[lod] ? cb.lz4Buf[lod] : cb.blobs[lod];
                ce.blobOffset[lod] = cursor;
                uint32_t bs = (uint32_t)body.size();
                if (cb.lz4[lod]) bs |= vxw::kBlobLz4Flag;
                ce.blobSize[lod] = bs;
                cursor += (uint32_t)body.size();
            }
        }
        std::fwrite(entries.data(), sizeof(vxw::ClusterEntry), entries.size(), f);
        // Cluster blobs.
        for (size_t k = 0; k < rb.clusterIdx.size(); ++k) {
            const ClusterBake& cb = bakes[rb.clusterIdx[k]];
            for (uint32_t lod = 0; lod < vxw::kLodCount; ++lod) {
                if (cb.voxelCount[lod] == 0) continue;
                const std::vector<uint8_t>& body = cb.lz4[lod] ? cb.lz4Buf[lod] : cb.blobs[lod];
                std::fwrite(body.data(), 1, body.size(), f);
            }
        }
    }

    // Now patch payloadStart + region table.
    std::fseek(f, (long)(sizeof(vxw::Header) + h.paletteCount * sizeof(uint32_t)), SEEK_SET);
    std::fwrite(&payloadStart, sizeof(uint64_t), 1, f);
    std::fseek(f, regionTableOffset, SEEK_SET);
    for (size_t ri = 0; ri < regionsOrdered.size(); ++ri) {
        const RegionBake& rb = *regionsOrdered[ri];
        vxw::RegionEntry re{};
        re.rx = rb.rx; re.ry = rb.ry; re.rz = rb.rz;
        re.payloadOffset = regionPayloadOffsets[ri];
        re.payloadSize   = regionPayloadSizes[ri];
        re.clusterCount  = (uint16_t)rb.clusterIdx.size();
        std::memcpy(re.clusterMask, rb.clusterMask, sizeof(rb.clusterMask));
        std::fwrite(&re, sizeof(re), 1, f);
    }

    long endPos = std::fseek(f, 0, SEEK_END), totalBytes;
    (void)endPos;
    totalBytes = std::ftell(f);
    std::fclose(f);
    std::printf("Wrote %s: %ld bytes (%.2f MB), %zu clusters, %u regions\n",
                outPath.c_str(), totalBytes, totalBytes / (1024.0 * 1024.0),
                bakes.size(), regionCount);
    return 0;
}
