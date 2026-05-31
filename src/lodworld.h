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
// Disk-resident point (8 bytes).
// Position is CHUNK-local (NOT cluster-local). Y range only uses low 6
// bits; the wasted 2 bits keep the struct 8-byte aligned and avoid bit
// packing in the hot path.
// =====================================================================
#pragma pack(push, 1)
struct DiskPoint
{
    uint8_t posX;        // 0..kChunkVoxX-1
    uint8_t posY;        // 0..kChunkVoxY-1
    uint8_t posZ;        // 0..kChunkVoxZ-1
    uint8_t palIdx;      // index into chunk's palette[]
    uint8_t visMask;     // low 6 bits = face visibility (+X,-X,+Y,-Y,+Z,-Z)
    uint8_t aoPacked[3]; // 6 faces * 4 bits AO = 24 bits
};
#pragma pack(pop)
static_assert(sizeof(DiskPoint) == 8, "");

// =====================================================================
// Disk-resident BLOCK (16 bytes). 2x2x2 voxels packed together.
//   blockX/Y/Z   = chunk-local block coord (0..127 for 256-voxel chunk)
//   occupancy    = 8-bit mask, bit i = voxel i present
//                  voxel-i local offset = (i & 1, (i>>1)&1, (i>>2)&1)
//   palIdx[i]    = chunk palette index for voxel i (valid where occupancy bit set)
//   parentRgb565 = direct RGB565 of the coarser-LOD voxel covering this block's
//                  centre. Used to LOD pop-hide via fade in the compute rasterizer.
//                  Stored as colour (not palIdx) so the shader doesn't need
//                  cross-LOD palette indirection.
// Used by PointCS_Block path. Coexists with DiskPoint format.
// =====================================================================
#pragma pack(push, 1)
struct DiskBlock
{
    uint8_t blockX;
    uint8_t blockY;
    uint8_t blockZ;
    uint8_t occupancy;
    uint8_t palIdx[8];
    uint16_t parentRgb565;
    uint8_t _pad[2];
};
#pragma pack(pop)
static_assert(sizeof(DiskBlock) == 16, "");

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
// Disk-resident cluster header (12 bytes).
// bounds packs start/end XYZ in cluster-local voxel coords (5 bits each,
// 30 bits total, top 2 spare). Used for tighter VFC than the nominal
// cluster AABB.
// numPoints == 0 means empty slot in the dense cluster array.
// pointFirst is offset (in DiskPoint units) into the CHUNK's point array.
// The renderer translates to pool offset by adding the chunk's poolBase.
// =====================================================================
#pragma pack(push, 1)
struct DiskCluster
{
    uint32_t bounds;
    uint16_t numPoints;
    uint16_t _pad;
    uint32_t pointFirst;
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
// Disk-resident chunk header (56 bytes).
// Followed in the blob by:
//   uint32_t   palette[paletteCount];          // 0x00BBGGRR
//   DiskCluster clusters[kClustersPerChunk];   // dense; numPoints=0 = empty
//   DiskPoint   points[totalPoints];
//
// gridX/Y/Z are this chunk's coords in its LOD's chunk grid. worldOrigin
// is the corner of the chunk in LOD0 voxel units (always, regardless of
// the chunk's own LOD). aabb is tight, in chunk-local LOD-voxel coords.
//
// childId[8] indexes into LOD(N-1).chunks at file-load time and is used
// for top-down recursion. Bit 0=X, bit 1=Y, bit 2=Z of the child quadrant.
// 0xFFFF = no child authored.
// =====================================================================
#pragma pack(push, 1)
struct DiskChunkHeader
{
    int32_t gridX, gridY, gridZ;
    int32_t worldOriginX, worldOriginY, worldOriginZ; // LOD0 voxel units
    uint32_t lodLevel;                                // 0..kLodCount-1
    uint32_t paletteCount;                            // <= kPaletteSize
    uint32_t totalPoints;
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
//     <chunk blob 0>                       // header+palette+clusters+points
//     <chunk blob 1>
//     ...
//
// ChunkEntry contains both compressed and raw byte sizes so the streamer
// can decide buffer sizes without seeking into the blob. Phase 1 emits
// raw blobs (flags = 0); Phase 2 wraps in LZ4 (flags |= kFlagLz4).
// =====================================================================

inline constexpr uint32_t kFileMagic = 0x31574F4Cu; // "LOW1" little-endian
inline constexpr uint32_t kFileVersion = 2u;

inline constexpr uint32_t kFlagLz4 = 1u << 0;
inline constexpr uint32_t kFlagBitGrid = 1u << 1; // per-cluster bit-grid + color stream
inline constexpr uint32_t kFlagAo = 1u << 2;      // per-cluster appends 3 bytes AO per occupied cell
inline constexpr uint32_t kFlagVisMask = 1u << 3; // per-cluster appends 1 byte visMask per occupied cell
inline constexpr uint32_t kFlagCellAo = 1u << 4;  // per-cluster appends 4-bit AO per "AO cell"
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

    // GPU residency (set by renderer when chunk goes resident).
    uint32_t poolBase;  // first DiskPoint index in the LOD's point pool
    uint32_t poolCount; // total points uploaded (sum of cluster numPoints)
    uint32_t slotIdx;   // index into LODWorld's ChunkInfo SRV; top byte
                        //  of startVertex during Draw.

    // PointCS_Block: per-chunk DiskBlock range in this LOD's blockPool.
    uint32_t blockBase;
    uint32_t blockCount;

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
    // Phase 1: CPU-side point pool, one per LOD. RuntimeChunk.poolBase indexes
    // into here. Renderer uploads to a GPU buffer at the same offsets, so
    // poolBase serves both CPU+GPU. Loader fills; renderer can drop after
    // upload if it wants to.
    std::vector<DiskPoint> pointPool;
    // Parallel block pool for PointCS_Block. Empty if .lw lacks block stream.
    std::vector<DiskBlock> blockPool;
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
    uint32_t poolBase;    // first DiskPoint index in pool
    uint32_t paletteBase; // slotIdx * kPaletteSize, offset into palette atlas
    uint32_t _pad[2];
};
#pragma pack(pop)
static_assert(sizeof(GpuChunkInfo) == 32, "");

// =====================================================================
// startVertex packing for the cluster-fused draw scheme.
//   [slot : 8 bits | vtxIdx : 24 bits]
// VS:
//   uint sv      = SV_VertexID;
//   uint slot    = sv >> 24;
//   uint vtxIdx  = sv & 0xFFFFFFu;
//   ChunkInfo ci = gChunkInfos[slot];
//   DiskPoint p  = gPointPool[ci.poolBase + vtxIdx];
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
