// lw_loader.cpp — read .lw (LODWorld) into runtime World.
//
// Phase 1: synchronous, whole-file load. Concatenates each LOD's chunk
// points into a single CPU pool (LODWorld::pointPool); RuntimeChunk.poolBase
// is the offset into that pool. Renderer uploads the pool to GPU and uses
// the same offset for vertex addressing.

#define _CRT_SECURE_NO_WARNINGS
#include "lodworld.h"
#include "lz4.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace lw {

static bool ReadAt(FILE* f, uint64_t off, void* dst, size_t n)
{
#if defined(_WIN32)
    if (_fseeki64(f, (long long)off, SEEK_SET) != 0) return false;
#else
    if (fseeko(f, (off_t)off, SEEK_SET) != 0) return false;
#endif
    return fread(dst, 1, n, f) == n;
}

bool LoadWorld(const char* path, World& out, std::string& err)
{
    FILE* f = fopen(path, "rb");
    if (!f) { err = "open failed"; return false; }

    // ---- File header ----
    FileHeader fh{};
    if (fread(&fh, sizeof(fh), 1, f) != 1) { fclose(f); err = "header read"; return false; }
    if (fh.magic != kFileMagic)     { fclose(f); err = "bad magic"; return false; }
    if (fh.version != kFileVersion) { fclose(f); err = "version mismatch"; return false; }
    if ((int)fh.chunkVoxX != kChunkVoxX || (int)fh.chunkVoxY != kChunkVoxY || (int)fh.chunkVoxZ != kChunkVoxZ
     || (int)fh.clusterVoxX != kClusterVoxX || (int)fh.clusterVoxY != kClusterVoxY || (int)fh.clusterVoxZ != kClusterVoxZ
     || (int)fh.lodCount != kLodCount) {
        fclose(f);
        err = "dim/lod mismatch with build constants";
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        out.worldAabbMin[i] = fh.worldAabbMin[i];
        out.worldAabbMax[i] = fh.worldAabbMax[i];
    }

    // ---- LOD headers ----
    LODHeader lh[kLodCount]{};
    if (fread(lh, sizeof(LODHeader), kLodCount, f) != kLodCount) {
        fclose(f); err = "lod header read"; return false;
    }

    // ---- Per-LOD: read ChunkEntry table, then load each chunk blob ----
    for (int L = 0; L < kLodCount; ++L) {
        LODWorld& lw = out.lods[L];
        lw.lodLevel = (uint8_t)L;
        lw.lodScale = 1u << L;
        lw.chunks.clear();
        lw.pointPool.clear();
        const uint32_t cc = lh[L].chunkCount;
        if (cc == 0) continue;

        std::vector<ChunkEntry> entries(cc);
        if (!ReadAt(f, lh[L].chunkTableOffset, entries.data(), cc * sizeof(ChunkEntry))) {
            fclose(f); err = "chunk table read"; return false;
        }

        lw.chunks.resize(cc);
        std::vector<uint8_t> diskBlob;
        std::vector<uint8_t> rawBlob;

        for (uint32_t i = 0; i < cc; ++i) {
            const ChunkEntry& ce = entries[i];
            diskBlob.resize(ce.blobBytes);
            if (!ReadAt(f, ce.blobOffset, diskBlob.data(), ce.blobBytes)) {
                fclose(f); err = "chunk blob read"; return false;
            }
            const uint8_t* blobPtr = diskBlob.data();
            size_t blobSize = diskBlob.size();
            if (ce.flags & kFlagLz4) {
                rawBlob.resize(ce.blobBytesRaw);
                int dec = LZ4_decompress_safe((const char*)diskBlob.data(),
                                              (char*)rawBlob.data(),
                                              (int)ce.blobBytes,
                                              (int)ce.blobBytesRaw);
                if (dec != (int)ce.blobBytesRaw) {
                    fclose(f); err = "lz4 decompress failed"; return false;
                }
                blobPtr = rawBlob.data();
                blobSize = rawBlob.size();
            }
            // Parse common header.
            const uint8_t* p = blobPtr;
            const uint8_t* end = blobPtr + blobSize;
            if ((size_t)(end - p) < sizeof(DiskChunkHeader)) {
                fclose(f); err = "blob too small for header"; return false;
            }
            DiskChunkHeader dch;
            memcpy(&dch, p, sizeof(dch)); p += sizeof(dch);

            const size_t palBytes = dch.paletteCount * sizeof(uint32_t);
            if ((size_t)(end - p) < palBytes) {
                fclose(f); err = "blob too small for palette"; return false;
            }
            RuntimeChunk& rc = lw.chunks[i];
            rc.gridX = dch.gridX; rc.gridY = dch.gridY; rc.gridZ = dch.gridZ;
            rc.worldOriginX = dch.worldOriginX;
            rc.worldOriginY = dch.worldOriginY;
            rc.worldOriginZ = dch.worldOriginZ;
            rc.lodLevel = (uint8_t)dch.lodLevel;
            rc.aabbMin[0] = dch.aabbMin[0]; rc.aabbMin[1] = dch.aabbMin[1]; rc.aabbMin[2] = dch.aabbMin[2];
            rc.aabbMax[0] = dch.aabbMax[0]; rc.aabbMax[1] = dch.aabbMax[1]; rc.aabbMax[2] = dch.aabbMax[2];
            for (int k = 0; k < 8; ++k) rc.childId[k] = dch.childId[k];
            rc.paletteCount = dch.paletteCount;
            memset(rc.palette, 0, sizeof(rc.palette));
            if (palBytes) memcpy(rc.palette, p, palBytes);
            p += palBytes;

            rc.poolBase = (uint32_t)lw.pointPool.size();
            rc.poolCount = dch.totalPoints;
            rc.slotIdx = i;

            if (ce.flags & kFlagBitGrid) {
                // Compressed: cluster mask + per-cluster (orderMode, leb128 nP,
                // leb128 bgSize, bgBytes, colorBytes). Reconstruct
                // rc.clusters[] + DiskPoint pool entries.
                if ((size_t)(end - p) < 16) { fclose(f); err = "blob short for clusterMask"; return false; }
                uint8_t clusterMask[16];
                memcpy(clusterMask, p, 16); p += 16;

                memset(rc.clusters, 0, sizeof(rc.clusters));
                lw.pointPool.resize(rc.poolBase + dch.totalPoints);
                uint32_t writePos = rc.poolBase;
                uint8_t bits[kClusterCellCount];

                for (int s = 0; s < kClustersPerChunk; ++s) {
                    if (!(clusterMask[s >> 3] & (1u << (s & 7)))) continue;
                    if (p >= end) { fclose(f); err = "blob short in cluster stream"; return false; }
                    uint8_t orderMode = *p++;
                    uint32_t nP = Leb128GetU32(p);
                    uint32_t bgSize = Leb128GetU32(p);
                    if (bgSize > (uint32_t)(end - p)) { fclose(f); err = "bgSize > remaining"; return false; }
                    RleDecodeBitGrid(p, bgSize, bits);
                    p += bgSize;
                    if (nP > (uint32_t)(end - p)) { fclose(f); err = "colors > remaining"; return false; }
                    const uint8_t* colors = p;
                    p += nP;

                    int cz_g = s / (kClustersX * kClustersY);
                    int cy_g = (s / kClustersX) % kClustersY;
                    int cx_g = s % kClustersX;
                    uint8_t clMn[3] = { 31, 31, 31 };
                    uint8_t clMx[3] = { 0, 0, 0 };

                    rc.clusters[s].pointFirst = writePos - rc.poolBase;
                    rc.clusters[s].numPoints  = (uint16_t)nP;
                    rc.clusters[s]._pad = 0;

                    uint32_t emitted = 0;
                    for (uint32_t cidx = 0; cidx < kClusterCellCount; ++cidx) {
                        if (!bits[cidx]) continue;
                        uint32_t lx, ly, lz;
                        LwCellCoord((LwOrderMode)orderMode, cidx, lx, ly, lz);
                        DiskPoint dp;
                        dp.posX = (uint8_t)(cx_g * kClusterVoxX + lx);
                        dp.posY = (uint8_t)(cy_g * kClusterVoxY + ly);
                        dp.posZ = (uint8_t)(cz_g * kClusterVoxZ + lz);
                        dp.palIdx = colors[emitted];
                        dp.visMask = 0x3F;          // AO/visMask not stored in compressed v0
                        dp.aoPacked[0] = 0xFF;
                        dp.aoPacked[1] = 0xFF;
                        dp.aoPacked[2] = 0xFF;
                        lw.pointPool[writePos + emitted] = dp;
                        if (lx < clMn[0]) clMn[0] = (uint8_t)lx;
                        if (ly < clMn[1]) clMn[1] = (uint8_t)ly;
                        if (lz < clMn[2]) clMn[2] = (uint8_t)lz;
                        if (lx > clMx[0]) clMx[0] = (uint8_t)lx;
                        if (ly > clMx[1]) clMx[1] = (uint8_t)ly;
                        if (lz > clMx[2]) clMx[2] = (uint8_t)lz;
                        ++emitted;
                    }
                    rc.clusters[s].bounds = PackClusterBounds(
                        clMn[0], clMn[1], clMn[2], clMx[0], clMx[1], clMx[2]);
                    writePos += emitted;
                }
            } else {
                // Legacy raw format (no compression).
                const size_t clusterBytes = sizeof(DiskCluster) * kClustersPerChunk;
                const size_t pointBytes = (size_t)dch.totalPoints * sizeof(DiskPoint);
                if ((size_t)(end - p) < clusterBytes + pointBytes) {
                    fclose(f); err = "blob too small for raw body"; return false;
                }
                memcpy(rc.clusters, p, clusterBytes); p += clusterBytes;
                if (pointBytes) {
                    lw.pointPool.resize(lw.pointPool.size() + dch.totalPoints);
                    memcpy(lw.pointPool.data() + rc.poolBase, p, pointBytes);
                    p += pointBytes;
                }
            }
        }

        // ---- Build SoA cull arrays (world float AABBs) ----
        lw.cull.minX.resize(cc); lw.cull.minY.resize(cc); lw.cull.minZ.resize(cc);
        lw.cull.maxX.resize(cc); lw.cull.maxY.resize(cc); lw.cull.maxZ.resize(cc);
        lw.cull.culled.assign(cc, 0);
        for (uint32_t i = 0; i < cc; ++i) {
            const RuntimeChunk& rc = lw.chunks[i];
            // AABB in LOD0 world voxel units: worldOrigin + (aabbLocal * lodScale).
            // aabbLocal is in chunk-local LOD-voxel coords (0..chunkVoxAxis-1).
            const float s = (float)lw.lodScale;
            lw.cull.minX[i] = (float)rc.worldOriginX + (float)rc.aabbMin[0] * s;
            lw.cull.minY[i] = (float)rc.worldOriginY + (float)rc.aabbMin[1] * s;
            lw.cull.minZ[i] = (float)rc.worldOriginZ + (float)rc.aabbMin[2] * s;
            // +1 because aabbMax is inclusive (the voxel cell occupies [max, max+1]).
            lw.cull.maxX[i] = (float)rc.worldOriginX + ((float)rc.aabbMax[0] + 1.0f) * s;
            lw.cull.maxY[i] = (float)rc.worldOriginY + ((float)rc.aabbMax[1] + 1.0f) * s;
            lw.cull.maxZ[i] = (float)rc.worldOriginZ + ((float)rc.aabbMax[2] + 1.0f) * s;
        }
    }

    fclose(f);
    return true;
}

bool SaveWorld(const char* /*path*/, const World& /*w*/, std::string& err)
{
    // Phase 1: writer lives in tools/vox2lw. Engine-side save not needed yet.
    err = "SaveWorld not implemented";
    return false;
}

} // namespace lw
