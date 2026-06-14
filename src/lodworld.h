#pragma once
// LODWorld — clipmap-style hierarchical voxel world format.
//
// Each LOD level is its own "world" with its own grid of chunks centred on
// the camera. Voxels in LOD N are 2^N source voxels per axis (= 8^N volume).
// Chunks and clusters share the same voxel-axis dimensions across all LODs;
// world-space coverage scales with the LOD voxel size.
//
// Phase 1 scope: static load (no streaming), single .lw file per world,
// uncompressed blobs. Streaming + per-chunk LZ4 added in later phases.

#include <cstdint>
#include <cstring>
#include <vector>
#include <string>

namespace lw
{

// =====================================================================
// Compile-time config. Tweak here; on-disk header sanity-checks against
// these so changing them invalidates old .lw files.
// =====================================================================

// Voxels per chunk axis. 256x256x64 = wide+shallow city ground tile. Y up.
inline constexpr int kChunkVoxX = 256;
inline constexpr int kChunkVoxY = 64;
inline constexpr int kChunkVoxZ = 256;

// Voxels per cluster axis. Cluster = unit of cull + draw fragment.
inline constexpr int kClusterVoxX = 32;
inline constexpr int kClusterVoxY = 32;
inline constexpr int kClusterVoxZ = 32;

// Derived cluster grid within a chunk. 8 * 2 * 8 = 128 dense slots.
inline constexpr int kClustersX = kChunkVoxX / kClusterVoxX;                   // 8
inline constexpr int kClustersY = kChunkVoxY / kClusterVoxY;                   // 2
inline constexpr int kClustersZ = kChunkVoxZ / kClusterVoxZ;                   // 8
inline constexpr int kClustersPerChunk = kClustersX * kClustersY * kClustersZ; // 128

// Linear index of a cluster within the dense per-chunk array.
inline constexpr int ClusterIdx(int cx, int cy, int cz)
{
    return (cz * kClustersY + cy) * kClustersX + cx;
}

// LOD levels. LOD0 = source detail; LOD4 = 16x linear (4096x cells).
inline constexpr int kLodCount = 5;

// Default streaming grid per LOD: chunks per axis around the camera.
// Camera always in centre chunk. Y typically thin.
inline constexpr int kDefaultGridXZ = 5;
inline constexpr int kDefaultGridY = 1;

// Per-chunk palette. vox2lw quantizes excess colours to nearest entry.
inline constexpr int kPaletteSize = 256;

// Sentinel: childId == kNoChild means "no chunk exists in this slot of
// the finer LOD" (the finer LOD has authored nothing here, e.g. all-empty
// region, or the chunk simply isn't streamed in yet — runtime sets
// streamed-out slots to kNoChild too).
inline constexpr uint16_t kNoChild = 0xFFFFu;

// =====================================================================
// Runtime BLOCK split into two parallel SoA arrays for the
// PointCS_Block path. Same index used into both pools.
//
//   BlockPos (4 B) — pos + occupancy. Pass 1 (depth-only) reads this.
//     blockX/Y/Z   = chunk-local block coord (0..127 for 256-voxel chunk)
//     occupancy    = 8-bit mask, bit i = voxel i present
//                    voxel-i local offset = (i & 1, (i>>1)&1, (i>>2)&1)
//
//   BlockCol (8 B) — palette per voxel. Pass 2 / single-pass reads this.
//     palIdx[i]    = chunk palette index for voxel i (valid where occ bit set)
//
// parentRgb565 dropped (was used for LOD pop-hide; will be recomputed in
// shader when needed).
// =====================================================================
#pragma pack(push, 1)
struct BlockPos
{
    uint8_t blockX;
    uint8_t blockY;
    uint8_t blockZ;
    uint8_t occupancy;
};
struct BlockCol
{
    uint8_t palIdx[8];
};
// Per-voxel 6-bit visibility mask (bits 0..5 = +X,-X,+Y,-Y,+Z,-Z). One byte
// per voxel slot, mirrors BlockCol layout. Populated from kFlagVisMask payload
// in octet stream (1 byte per occupied voxel).
struct BlockVis
{
    uint8_t visMask[8];
};
// Per-voxel baked AO. 6 faces * 4-bit = 24 bits = 3 B/voxel, mirrors BlockCol
// layout (8 voxel slots). Packing matches DiskVoxel::aoPacked: face fi's nibble
// lives at bit fi*4 across the 3 bytes. Populated from kFlagAo octet payload.
struct BlockAo
{
    uint8_t ao[8][3];
};
#pragma pack(pop)
static_assert(sizeof(BlockPos) == 4, "");
static_assert(sizeof(BlockCol) == 8, "");
static_assert(sizeof(BlockVis) == 8, "");
static_assert(sizeof(BlockAo) == 24, "");

// kFlag for chunk blob: compact per-chunk OCTET stream at end of blob.
// Format (V3, file version 2):
//   stream of clusters until clusterID byte has bit 7 set:
//     uint8 clusterID                ;  bits 0..6 = cluster idx, bit 7 = last in chunk
//     // Per cluster:
//     //   modeByte at every 4th octet boundary (groups 4 modes, 2 bits each)
//     //   octet group sequence ends when an octet has run-bits == 15
//     For each octet (counting from 0 within cluster):
//       if (octetCount % 4 == 0): uint8 modeByte
//                                  ; 4 modes packed (low bits = mode for next octet)
//                                  ;   mode 00 = uniform reuse (no colour payload; uses prev uniform colour)
//                                  ;   mode 01 = uniform new   (1B palette idx follows; sets prev colour)
//                                  ;   mode 10 = varied        (popcount(mask) palette bytes follow)
//                                  ;   mode 11 = reserved
//       if (not in implicit-run): uint16 octetID
//                                  ;  bits 0..11 = octet idx (Y-major in cluster)
//                                  ;  bits 12..15 = run-bits:
//                                  ;    0      = next octet uses fresh octetID
//                                  ;    1..14  = N implicit octets follow (no octetID; idx = prev+1)
//                                  ;    15     = this is final octet of cluster
//       uint8  voxelMask           ;  bit i = voxel i present
//       payload depending on mode (see above)
//
//   prevUniformColour is per-cluster state, reset to "undefined" at clusterID.
//   modeByte position resets at clusterID too.
//
// Cluster idx encoding: idx = (cz * kClustersY + cy) * kClustersX + cx
// Octet idx in cluster (Y-major): idx = (oy * 16 + oz) * 16 + ox
inline constexpr uint32_t kFlagBlocks = 1u << 5;

// Octet payload modes (2-bit values inside modeByte).
inline constexpr uint8_t kOctetModeUniformReuse = 0u;
inline constexpr uint8_t kOctetModeUniformNew = 1u;
inline constexpr uint8_t kOctetModeVaried = 2u;

// =====================================================================
// Per-cluster runtime info (12 bytes). Despite the "Disk" prefix it's
// rebuilt by the loader from the OCTET stream — not read off disk.
// bounds packs start/end XYZ in cluster-local voxel coords (5 bits each,
// 30 bits total, top 2 spare). Used for tighter VFC than the nominal
// cluster AABB. numPoints != 0 = cluster present (1 dummy, real count is
// clusterBlockCount[]).
// =====================================================================
#pragma pack(push, 1)
struct DiskCluster
{
    uint32_t bounds;
    uint16_t numPoints;
    uint16_t _pad;
    uint32_t pointFirst; // unused (kept for layout)
};
#pragma pack(pop)
static_assert(sizeof(DiskCluster) == 12, "");

inline uint32_t PackClusterBounds(uint32_t sx, uint32_t sy, uint32_t sz,
                                  uint32_t ex, uint32_t ey, uint32_t ez)
{
    return (sx & 0x1Fu) | ((sy & 0x1Fu) << 5) | ((sz & 0x1Fu) << 10) | ((ex & 0x1Fu) << 15) | ((ey & 0x1Fu) << 20) | ((ez & 0x1Fu) << 25);
}
inline void UnpackClusterBounds(uint32_t b, uint8_t out[6])
{
    out[0] = (b >> 0) & 0x1Fu;
    out[1] = (b >> 5) & 0x1Fu;
    out[2] = (b >> 10) & 0x1Fu;
    out[3] = (b >> 15) & 0x1Fu;
    out[4] = (b >> 20) & 0x1Fu;
    out[5] = (b >> 25) & 0x1Fu;
}

// =====================================================================
// Disk-resident chunk header (60 bytes). Followed in the blob by:
//   uint32_t palette[paletteCount];   // 0x00BBGGRR
//   <octet stream>                    // see kFlagBlocks doc
//
// gridX/Y/Z are this chunk's coords in its LOD's chunk grid. worldOrigin
// is the corner of the chunk in LOD0 voxel units. aabb is tight, in
// chunk-local LOD-voxel coords. childId[8] indexes LOD(N-1).chunks for
// recursion. 0xFFFF = no child authored.
// =====================================================================
#pragma pack(push, 1)
struct DiskChunkHeader
{
    int32_t gridX, gridY, gridZ;
    int32_t worldOriginX, worldOriginY, worldOriginZ; // LOD0 voxel units
    uint32_t lodLevel;                                // 0..kLodCount-1
    uint32_t paletteCount;                            // <= kPaletteSize
    uint32_t _unusedTotalPoints;                      // kept for wire layout
    uint16_t childId[8];
    uint8_t aabbMin[3]; // chunk-local LOD-voxel coords
    uint8_t aabbMax[3];
    uint16_t _pad;
};
#pragma pack(pop)
static_assert(sizeof(DiskChunkHeader) == 60, "");

// =====================================================================
// File-level layout.
//
//   FileHeader
//   LODHeader[kLodCount]
//   for each lod:
//     ChunkEntry[lod.chunkCount]           // sorted by gridZ,gridY,gridX
//     <chunk blob 0>                       // DiskChunkHeader + palette + octet stream
//     <chunk blob 1>
//     ...
//
// ChunkEntry holds compressed + raw byte sizes so the streamer sizes
// buffers without seeking into the blob. Optional LZ4 wrap (kFlagLz4).
// =====================================================================

inline constexpr uint32_t kFileMagic = 0x31574F4Cu; // "LOW1" little-endian
inline constexpr uint32_t kFileVersion = 3u;

inline constexpr uint32_t kFlagLz4 = 1u << 0;
// kFlagVisMask: octet stream payload appends 1 byte (6-bit visMask) per
// occupied voxel after the mode-specific palette bytes.
inline constexpr uint32_t kFlagVisMask = 1u << 3;
// kFlagAo: octet stream payload appends 3 bytes (6 faces * 4-bit AO) per
// occupied voxel after the visMask byte. Packing matches DiskVoxel::aoPacked.
inline constexpr uint32_t kFlagAo = 1u << 4;
                                                  // (empty cell adjacent to solid voxel in chunk grid)

// =====================================================================
// Per-cluster ordering modes. Selected per-cluster at bake time; encoder
// tries each and keeps the smallest. Decoder reads stored byte to know
// how to walk the bit-grid.
//   Y-major (slab):  idx = (y * 32 + z) * 32 + x   — wins on flat tiles
//   Morton:          standard 3D Z-order            — wins on cubic blobs
// =====================================================================
enum LwOrderMode : uint8_t
{
    kOrderYMajor = 0,
    kOrderMorton = 1,
};

// LEB128 unsigned encode. Appends 1..5 bytes to `out` for uint32 value.
inline void Leb128PutU32(std::vector<uint8_t>& out, uint32_t v)
{
    while (v >= 0x80u)
    {
        out.push_back((uint8_t)((v & 0x7Fu) | 0x80u));
        v >>= 7;
    }
    out.push_back((uint8_t)v);
}
// LEB128 unsigned decode. Advances `p`. Caller ensures bounds.
inline uint32_t Leb128GetU32(const uint8_t*& p)
{
    uint32_t v = 0;
    uint32_t shift = 0;
    while (true)
    {
        uint8_t b = *p++;
        v |= (uint32_t)(b & 0x7Fu) << shift;
        if ((b & 0x80u) == 0)
            break;
        shift += 7;
    }
    return v;
}

// Pre-baked Morton interleave for 5-bit axis. Returns 15-bit morton code.
inline uint32_t MortonEncode5(uint32_t x, uint32_t y, uint32_t z)
{
    auto spread = [](uint32_t v) -> uint32_t
    {
        v &= 0x1F;
        v = (v | (v << 8)) & 0x0300F00F;
        v = (v | (v << 4)) & 0x030C30C3;
        v = (v | (v << 2)) & 0x09249249;
        return v;
    };
    return spread(x) | (spread(y) << 1) | (spread(z) << 2);
}
// Inverse: extract (x,y,z) from 15-bit morton code.
inline void MortonDecode5(uint32_t m, uint32_t& x, uint32_t& y, uint32_t& z)
{
    auto compact = [](uint32_t v) -> uint32_t
    {
        v &= 0x09249249;
        v = (v | (v >> 2)) & 0x030C30C3;
        v = (v | (v >> 4)) & 0x0300F00F;
        v = (v | (v >> 8)) & 0x1F;
        return v;
    };
    x = compact(m);
    y = compact(m >> 1);
    z = compact(m >> 2);
}

// Voxels-per-cluster (each cluster is 32^3). Used for bit-grid size.
inline constexpr uint32_t kClusterCellCount = (uint32_t)kClusterVoxX * (uint32_t)kClusterVoxY * (uint32_t)kClusterVoxZ;
static_assert(kClusterCellCount == 32768, "bit-grid sizing assumes 32^3 cluster");

// Map cluster-local (x,y,z) → linear cell index for a given order mode.
inline uint32_t LwCellIndex(LwOrderMode mode, uint32_t x, uint32_t y, uint32_t z)
{
    if (mode == kOrderMorton)
        return MortonEncode5(x, y, z);
    return (y * kClusterVoxZ + z) * kClusterVoxX + x;
}
// Inverse.
inline void LwCellCoord(LwOrderMode mode, uint32_t idx,
                        uint32_t& x, uint32_t& y, uint32_t& z)
{
    if (mode == kOrderMorton)
    {
        MortonDecode5(idx, x, y, z);
        return;
    }
    x = idx % kClusterVoxX;
    z = (idx / kClusterVoxX) % kClusterVoxZ;
    y = idx / (kClusterVoxX * kClusterVoxZ);
}

// RLE bit-grid encode: alternating zero-run / one-run counts (LEB128),
// starting with a zero-run (count may be 0). Total = kClusterCellCount.
inline void RleEncodeBitGrid(const uint8_t* bits, std::vector<uint8_t>& out)
{
    uint32_t i = 0;
    uint32_t expect = 0;
    while (i < kClusterCellCount)
    {
        uint32_t run = 0;
        while (i < kClusterCellCount && bits[i] == expect)
        {
            ++i;
            ++run;
        }
        Leb128PutU32(out, run);
        expect ^= 1;
    }
}
// RLE bit-grid decode. Writes kClusterCellCount bytes (0 or 1) into `bits`.
inline void RleDecodeBitGrid(const uint8_t* enc, uint32_t encBytes, uint8_t* bits)
{
    memset(bits, 0, kClusterCellCount);
    const uint8_t* p = enc;
    const uint8_t* end = enc + encBytes;
    uint32_t i = 0;
    uint32_t curBit = 0;
    while (p < end && i < kClusterCellCount)
    {
        uint32_t run = Leb128GetU32(p);
        if (curBit)
        {
            for (uint32_t k = 0; k < run && i < kClusterCellCount; ++k)
                bits[i++] = 1;
        }
        else
        {
            i += run;
        }
        curBit ^= 1;
    }
}

#pragma pack(push, 1)
struct FileHeader
{
    uint32_t magic;
    uint32_t version;
    uint32_t chunkVoxX, chunkVoxY, chunkVoxZ;
    uint32_t clusterVoxX, clusterVoxY, clusterVoxZ;
    uint32_t lodCount;
    int32_t worldAabbMin[3]; // LOD0 voxel units
    int32_t worldAabbMax[3];
    uint32_t _reserved[4];
};
#pragma pack(pop)
static_assert(sizeof(FileHeader) == 76, "");

#pragma pack(push, 1)
struct LODHeader
{
    uint32_t lodLevel;
    uint32_t chunkCount;
    uint64_t chunkTableOffset; // bytes from file start to ChunkEntry[0]
};
#pragma pack(pop)
static_assert(sizeof(LODHeader) == 16, "");

#pragma pack(push, 1)
struct ChunkEntry
{
    int32_t gridX, gridY, gridZ;
    uint64_t blobOffset;   // bytes from file start to DiskChunkHeader
    uint32_t blobBytes;    // size on disk (== raw size if uncompressed)
    uint32_t blobBytesRaw; // uncompressed size
    uint32_t flags;        // kFlag*
};
#pragma pack(pop)
static_assert(sizeof(ChunkEntry) == 32, "");

// =====================================================================
// Runtime representation. Loader inflates the disk blob into these.
// Dense cluster array kept inline for cache-locality during recursion.
// =====================================================================

struct RuntimeChunk
{
    int32_t gridX, gridY, gridZ;
    int32_t worldOriginX, worldOriginY, worldOriginZ;
    uint8_t lodLevel;
    uint8_t aabbMin[3];
    uint8_t aabbMax[3];
    uint16_t childId[8];                     // LOD(N-1) chunk index
    DiskCluster clusters[kClustersPerChunk]; // dense; numPoints=0 = empty

    uint32_t slotIdx;   // index into LODWorld's ChunkInfo SRV; top byte
                        //  of startVertex during Draw.

    // PointCS_Block: per-chunk block range. Same index into blockPosPool
    // and blockColPool of this LOD.
    uint32_t blockBase;
    uint32_t blockCount;
    // Per-cluster block sub-ranges (offsets relative to blockBase). Filled by
    // loader during V3 stream walk. clusterBlockCount[ci]==0 = no blocks for
    // that cluster. Used by per-cluster cull + LOD on the block dispatch path.
    uint32_t clusterBlockFirst[kClustersPerChunk];
    uint32_t clusterBlockCount[kClustersPerChunk];

    // Per-chunk palette (uploaded once to per-LOD palette atlas).
    uint32_t paletteCount;
    uint32_t palette[kPaletteSize];
};

// SoA cull arrays per LOD world. Parallel to LODWorld::chunks[].
// Refreshed when chunks load/unload. Frustum cull walks these with SIMD
// and writes `culled` bitmask which the recursion then reads.
struct CullArrays
{
    std::vector<float> minX, minY, minZ;
    std::vector<float> maxX, maxY, maxZ;
    std::vector<uint8_t> culled; // 0 = visible, 1 = culled, parallel to chunks
};

struct LODWorld
{
    uint8_t lodLevel;
    uint32_t lodScale; // 1 << lodLevel — voxel size in LOD0 units
    std::vector<RuntimeChunk> chunks;
    CullArrays cull;
    // Parallel block pools for PointCS_Block. Empty if .lw lacks block stream.
    // Same index into all three. blockVisPool is empty unless kFlagVisMask was
    // set on the chunk blob.
    std::vector<BlockPos> blockPosPool;
    std::vector<BlockCol> blockColPool;
    std::vector<BlockVis> blockVisPool;
    // Per-voxel baked AO (3 B/voxel). Empty unless kFlagAo was set on the chunk
    // blob. Same index into pool as blockPosPool/blockColPool.
    std::vector<BlockAo> blockAoPool;
};

struct World
{
    LODWorld lods[kLodCount];
    int32_t worldAabbMin[3];
    int32_t worldAabbMax[3];
};

// =====================================================================
// GPU-side per-chunk info. Bound as StructuredBuffer<ChunkInfo>.
// Indexed by slotIdx (top byte of SV_VertexID). VS extracts and reads.
// =====================================================================
#pragma pack(push, 1)
struct GpuChunkInfo
{
    float worldOriginX, worldOriginY, worldOriginZ;
    float lodScale;       // multiplier from LOD-voxel to world units
    uint32_t _unused0;    // was poolBase (point pool dropped with CS-only path)
    uint32_t paletteBase; // slotIdx * kPaletteSize, offset into palette atlas
    uint32_t _pad[2];
};
#pragma pack(pop)
static_assert(sizeof(GpuChunkInfo) == 32, "");

// =====================================================================
// startVertex packing for the cluster-fused draw scheme.
//   [slot : 8 bits | vtxIdx : 24 bits]
// (point-VS path retired; helpers retained for future revivals.)
// =====================================================================
inline constexpr int kStartVertexSlotBits = 8;
inline constexpr int kStartVertexVtxBits = 24;
inline constexpr uint32_t kStartVertexVtxMask = (1u << kStartVertexVtxBits) - 1u;

inline uint32_t MakeStartVertex(uint32_t slotIdx, uint32_t vtxIdxInChunk)
{
    return (slotIdx << kStartVertexVtxBits) | (vtxIdxInChunk & kStartVertexVtxMask);
}

// Max resident chunks per LOD. Slot now passed via per-draw CB (gLwSlot),
// so no hard encoding cap — limit chosen to fit GPU SRV + sanity check.
inline constexpr uint32_t kMaxResidentChunksPerLod = 65536;

// =====================================================================
// Loader / writer prototypes. Implementations in src/lw_loader.cpp and
// tools/vox2lw/vox2lw.cpp respectively. Phase 1: synchronous load of
// whole file. Streaming in Phase 2 will use ChunkEntry index for
// per-chunk seeks.
// =====================================================================
bool LoadWorld(const char* path, World& out, std::string& err);
// Streaming load: LODs in reverse order (coarsest first), per-LOD callback.
// Optional StreamCfg restricts which chunks load per LOD by distance to the
// given world-space camera point. LOD with radius <= 0 = load everything.
struct StreamCfg
{
    float camX, camY, camZ;  // world-space camera position (LOD0 voxel units)
    float radius[kLodCount]; // per-LOD max chunk-center distance, 0 = unlimited
    // Frustum planes (a,b,c,d) for in-frustum priority. Chunks inside the
    // frustum load before chunks only inside the radius shell. Both still load
    // within the same LOD — main can render whichever arrives first. Set
    // hasFrustum=false to disable; loader then only uses radius shell.
    bool hasFrustum;
    float frustumPlanes[6][4];
};
using LodReadyFn = void (*)(void* user, int L);
bool LoadWorldStreaming(const char* path, World& out, std::string& err,
                        LodReadyFn onLodReady, void* user,
                        const StreamCfg* cfg = nullptr);
bool SaveWorld(const char* path, const World& w, std::string& err);

} // namespace lw
