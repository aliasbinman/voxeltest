#pragma once
//
// .vxw — clustered, streamable voxel world format.
// See dataformat-spec discussion. 2-tier hierarchy:
//   REGION = 8 x 8 x 8 clusters    (= 256 voxels per axis at default)
//   CLUSTER = 32 x 32 x 32 voxels  (per-cluster pre-baked L0..L3 blobs)
//
// Per-cluster per-LOD blob layouts use bit-packed positions, visMask, AO,
// palette index. Blobs may be LZ4-compressed (top bit of blobSize flags it).
//

#include <cstdint>

namespace vxw {

// ---- constants ------------------------------------------------------------

constexpr uint32_t kClusterVoxAxis      = 32;
constexpr uint32_t kClustersPerRegionAx = 8;
constexpr uint32_t kRegionVoxAxis       = kClusterVoxAxis * kClustersPerRegionAx;   // 256
constexpr uint32_t kClustersPerRegion   = kClustersPerRegionAx
                                         * kClustersPerRegionAx
                                         * kClustersPerRegionAx;                    // 512
constexpr uint32_t kLodCount            = 4;

// Bits per axis at each LOD (max coord = (1<<bits) - 1).
constexpr uint32_t kBitsPerAxis[kLodCount] = { 5, 4, 3, 2 };
constexpr uint32_t kVoxAxisAtLod[kLodCount] = { 32, 16, 8, 4 };

// Hard caps per cluster per LOD. Bake errors if exceeded; pool blocks sized
// to match.
constexpr uint32_t kMaxVoxAtLod[kLodCount] = { 8192, 1024, 256, 64 };

// Per-voxel storage sizes (bits).
constexpr uint32_t kVisMaskBits = 6;       // 6 cube faces
constexpr uint32_t kAoFaceBits  = 4;
constexpr uint32_t kAoBits      = kAoFaceBits * 6;   // 24

// Compression flag in the high bit of blobSize.
constexpr uint32_t kBlobLz4Flag = 0x80000000u;
constexpr uint32_t kBlobSizeMask = 0x7FFFFFFFu;

// File magic / version.
constexpr char     kMagic[4] = { 'V','X','W','1' };

// ---- on-disk structs ------------------------------------------------------

#pragma pack(push, 1)

// Header sits at file offset 0; followed by palette[paletteCount],
// then payloadStart (uint64), then regionCount + RegionEntry[regionCount].
struct Header {
    char     magic[4];                 // "VXW1"
    uint32_t version;                  // kAssetVersion
    uint32_t metersPerVoxelQ16;        // 1.0 m -> 0x10000 (Q16.16)
    int32_t  originVoxel[3];
    uint32_t clusterVoxAxis;           // = kClusterVoxAxis
    uint32_t paletteCount;             // followed by uint32 palette[paletteCount]
    // After the palette: uint64 payloadStart, uint32 regionCount, RegionEntry regions[regionCount]
};

struct RegionEntry {
    int32_t  rx, ry, rz;               // region grid coord
    uint32_t _pad0;
    uint64_t payloadOffset;            // byte offset into file
    uint32_t payloadSize;              // total bytes for this region payload
    uint16_t clusterCount;             // popcount(clusterMask)
    uint16_t flags;                    // reserved
    uint8_t  clusterMask[64];          // 512-bit; bit n = local cluster slot n present
};
static_assert(sizeof(RegionEntry) == 96, "RegionEntry must be 96 B");

// Region payload layout:
//   ClusterEntry clusters[clusterCount]   (in mask-iteration order)
//   [blob bytes follow, addressed by ClusterEntry.blobOffset]
struct ClusterEntry {
    uint16_t voxelCount[kLodCount];    // 8 (0 = LOD absent)
    uint32_t blobOffset[kLodCount];    // 16 (rel to region payload start; 0 if absent)
    uint32_t blobSize[kLodCount];      // 16 (top bit = LZ4 flag; lower 31 bits = bytes)
    uint8_t  _pad[8];                  // 8
};
static_assert(sizeof(ClusterEntry) == 48, "ClusterEntry must be 48 B");

#pragma pack(pop)

// ---- per-cluster local addressing ----------------------------------------

// Pack (lx, ly, lz) cluster slot inside region (each 0..7) into 0..511.
inline uint32_t LocalClusterSlot(uint32_t lx, uint32_t ly, uint32_t lz)
{
    return lx | (ly << 3) | (lz << 6);
}
inline void UnpackLocalClusterSlot(uint32_t s, uint32_t& lx, uint32_t& ly, uint32_t& lz)
{
    lx = s & 7u; ly = (s >> 3) & 7u; lz = (s >> 6) & 7u;
}

// Region grid coord from cluster grid coord.
inline void RegionOf(int32_t cx, int32_t cy, int32_t cz,
                     int32_t& rx, int32_t& ry, int32_t& rz)
{
    rx = cx >> 3; ry = cy >> 3; rz = cz >> 3;
}

// Bit-packed blob field byte sizes given voxel count + LOD.
inline uint32_t PosBytes(uint32_t voxCount, uint32_t lod)
{
    const uint32_t bits = kBitsPerAxis[lod] * 3u;
    return (voxCount * bits + 7u) / 8u;
}
inline uint32_t MaskBytes(uint32_t voxCount)
{
    return (voxCount * kVisMaskBits + 7u) / 8u;
}
inline uint32_t AoBytes(uint32_t voxCount)
{
    return voxCount * 3u;              // 24 bits / 8 = 3 bytes per voxel
}
inline uint32_t PalIdxBytes(uint32_t voxCount, bool palette16bit)
{
    return voxCount * (palette16bit ? 2u : 1u);
}
inline uint32_t RawBlobBytes(uint32_t voxCount, uint32_t lod, bool palette16bit)
{
    return PosBytes(voxCount, lod) + MaskBytes(voxCount) + AoBytes(voxCount)
         + PalIdxBytes(voxCount, palette16bit);
}

// Half-extent (world units) for a voxel at given LOD.
inline float HalfExtAtLod(uint32_t lod)
{
    static const float kHE[kLodCount] = { 0.5f, 1.0f, 2.0f, 4.0f };
    return kHE[lod];
}

} // namespace vxw
