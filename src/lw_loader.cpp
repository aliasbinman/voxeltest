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
            // Re-bind `blob` alias used by parser below.
            std::vector<uint8_t> blob(blobPtr, blobPtr + blobSize);
            // Parse: DiskChunkHeader, palette[paletteCount], DiskCluster[kClustersPerChunk], DiskPoint[totalPoints]
            const uint8_t* p = blob.data();
            const uint8_t* end = p + blob.size();
            if ((size_t)(end - p) < sizeof(DiskChunkHeader)) {
                fclose(f); err = "blob too small for header"; return false;
            }
            DiskChunkHeader dch;
            memcpy(&dch, p, sizeof(dch)); p += sizeof(dch);

            const size_t palBytes = dch.paletteCount * sizeof(uint32_t);
            const size_t clusterBytes = sizeof(DiskCluster) * kClustersPerChunk;
            const size_t pointBytes = (size_t)dch.totalPoints * sizeof(DiskPoint);
            if ((size_t)(end - p) < palBytes + clusterBytes + pointBytes) {
                fclose(f); err = "blob too small for body"; return false;
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

            memcpy(rc.clusters, p, clusterBytes); p += clusterBytes;

            // Pool: append chunk's points; record base.
            rc.poolBase = (uint32_t)lw.pointPool.size();
            rc.poolCount = dch.totalPoints;
            rc.slotIdx = i;  // For Phase 1 (no streaming), slot == chunk index.

            if (pointBytes) {
                lw.pointPool.resize(lw.pointPool.size() + dch.totalPoints);
                memcpy(lw.pointPool.data() + rc.poolBase, p, pointBytes);
                p += pointBytes;
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
