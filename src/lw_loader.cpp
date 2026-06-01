// lw_loader.cpp — read .lw (LODWorld) into runtime World.
//
// Phase 1: synchronous, whole-file load. Concatenates each LOD's chunk
// points into a single CPU pool (LODWorld::pointPool); RuntimeChunk.poolBase
// is the offset into that pool. Renderer uploads the pool to GPU and uses
// the same offset for vertex addressing.

#define _CRT_SECURE_NO_WARNINGS
#include "lodworld.h"
#include "lz4.h"
#include "microprofile.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>

namespace lw
{

static bool ReadAt(FILE* f, uint64_t off, void* dst, size_t n)
{
#if defined(_WIN32)
    if (_fseeki64(f, (long long)off, SEEK_SET) != 0)
        return false;
#else
    if (fseeko(f, (off_t)off, SEEK_SET) != 0)
        return false;
#endif
    return fread(dst, 1, n, f) == n;
}

bool LoadWorld(const char* path, World& out, std::string& err)
{
    return LoadWorldStreaming(path, out, err, nullptr, nullptr, nullptr);
}

bool LoadWorldStreaming(const char* path, World& out, std::string& err,
                        LodReadyFn onLodReady, void* user, const StreamCfg* cfg)
{
    FILE* f = fopen(path, "rb");
    if (!f)
    {
        err = "open failed";
        return false;
    }

    // ---- File header ----
    FileHeader fh{};
    if (fread(&fh, sizeof(fh), 1, f) != 1)
    {
        fclose(f);
        err = "header read";
        return false;
    }
    if (fh.magic != kFileMagic)
    {
        fclose(f);
        err = "bad magic";
        return false;
    }
    if (fh.version != kFileVersion)
    {
        fclose(f);
        err = "version mismatch";
        return false;
    }
    if ((int)fh.chunkVoxX != kChunkVoxX || (int)fh.chunkVoxY != kChunkVoxY || (int)fh.chunkVoxZ != kChunkVoxZ || (int)fh.clusterVoxX != kClusterVoxX || (int)fh.clusterVoxY != kClusterVoxY || (int)fh.clusterVoxZ != kClusterVoxZ || (int)fh.lodCount != kLodCount)
    {
        fclose(f);
        err = "dim/lod mismatch with build constants";
        return false;
    }
    for (int i = 0; i < 3; ++i)
    {
        out.worldAabbMin[i] = fh.worldAabbMin[i];
        out.worldAabbMax[i] = fh.worldAabbMax[i];
    }

    // ---- LOD headers ----
    LODHeader lh[kLodCount]{};
    if (fread(lh, sizeof(LODHeader), kLodCount, f) != kLodCount)
    {
        fclose(f);
        err = "lod header read";
        return false;
    }

    // ---- Per-LOD load order ----
    // 1) Coarsest (kLodCount-1) first  -> instant whole-world coarse view.
    // 2) Finest (LOD0) next             -> high detail near camera (tightest shell, fewest chunks).
    // 3) Mid LODs in ascending order    -> progressive refinement of the middle distance band.
    // Total chunks ordered so the user sees coarse + sharpest detail quickly.
    using clk = std::chrono::steady_clock;
    int lodOrder[kLodCount];
    lodOrder[0] = kLodCount - 1;
    for (int i = 1; i < kLodCount; ++i)
        lodOrder[i] = i - 1;
    for (int oi = 0; oi < kLodCount; ++oi)
    {
        int L = lodOrder[oi];
        MICROPROFILE_SCOPEI("Loader", "LOD", 0xff80a0ff);
        auto tLodStart = clk::now();
        double tReadMs = 0, tLz4Ms = 0, tHeaderMs = 0;
        double tRankMs = 0, tCullMs = 0, tBlocksMs = 0;
        uint64_t totalBlocks = 0;
        uint64_t bytesRead = 0, bytesLz4Raw = 0;
        LODWorld& lw = out.lods[L];
        lw.lodLevel = (uint8_t)L;
        lw.lodScale = 1u << L;
        lw.chunks.clear();
        lw.pointPool.clear();
        lw.blockPosPool.clear();
        lw.blockColPool.clear();
        lw.blockPointPool.clear();
        const uint32_t cc = lh[L].chunkCount;
        if (cc == 0)
            continue;

        std::vector<ChunkEntry> entries(cc);
        if (!ReadAt(f, lh[L].chunkTableOffset, entries.data(), cc * sizeof(ChunkEntry)))
        {
            fclose(f);
            err = "chunk table read";
            return false;
        }

        lw.chunks.resize(cc);
        std::vector<uint8_t> diskBlob;
        std::vector<uint8_t> rawBlob;

        // Per-LOD shell + frustum priority. Pre-classify chunks into:
        //   priority 0 = in view frustum (load first)
        //   priority 1 = in radius shell but out of frustum (load after)
        //   skipped    = neither
        // Within each priority bucket, sort by distance (nearest first).
        const float shellRadius = (cfg && cfg->radius[L] > 0.0f) ? cfg->radius[L] : 0.0f;
        const float chunkW = (float)kChunkVoxX * (float)lw.lodScale;
        const float chunkH = (float)kChunkVoxY * (float)lw.lodScale;
        const float chunkD = (float)kChunkVoxZ * (float)lw.lodScale;
        struct ChunkRank
        {
            uint32_t idx;
            float dist2;
            int prio;
        };
        std::vector<ChunkRank> ranks;
        ranks.reserve(cc);
        auto aabbInFrustum = [&](float mnx, float mny, float mnz,
                                 float mxx, float mxy, float mxz) -> bool
        {
            for (int pi = 0; pi < 5; ++pi)
            { // sides + near, skip far
                float a = cfg->frustumPlanes[pi][0], b = cfg->frustumPlanes[pi][1];
                float c = cfg->frustumPlanes[pi][2], d = cfg->frustumPlanes[pi][3];
                float px = a >= 0 ? mxx : mnx;
                float py = b >= 0 ? mxy : mny;
                float pz = c >= 0 ? mxz : mnz;
                if (a * px + b * py + c * pz + d < 0.0f)
                    return false;
            }
            return true;
        };
        auto tRank0 = clk::now();
        for (uint32_t i = 0; i < cc; ++i)
        {
            const ChunkEntry& ce = entries[i];
            float mnx = (float)ce.gridX * chunkW;
            float mny = (float)ce.gridY * chunkH;
            float mnz = (float)ce.gridZ * chunkD;
            float mxx = mnx + chunkW, mxy = mny + chunkH, mxz = mnz + chunkD;
            float cx = mnx + chunkW * 0.5f;
            float cy = mny + chunkH * 0.5f;
            float cz = mnz + chunkD * 0.5f;
            float dx = cx - (cfg ? cfg->camX : 0.0f);
            float dy = cy - (cfg ? cfg->camY : 0.0f);
            float dz = cz - (cfg ? cfg->camZ : 0.0f);
            float d2 = dx * dx + dy * dy + dz * dz;
            // Shell decides INCLUSION; frustum decides PRIORITY within shell.
            // Without this gate, in-frustum chunks loaded the entire view
            // direction regardless of distance (LOD0 ended up loading every
            // chunk the camera could see = several GB).
            bool inShell = (shellRadius <= 0.0f) || (d2 <= shellRadius * shellRadius);
            if (!inShell)
                continue;
            bool inFrust = cfg && cfg->hasFrustum && aabbInFrustum(mnx, mny, mnz, mxx, mxy, mxz);
            ranks.push_back({i, d2, inFrust ? 0 : 1});
        }
        std::sort(ranks.begin(), ranks.end(), [](const ChunkRank& a, const ChunkRank& b)
                  {
            if (a.prio != b.prio) return a.prio < b.prio;
            return a.dist2 < b.dist2; });
        tRankMs = std::chrono::duration<double, std::milli>(clk::now() - tRank0).count();

        for (size_t r = 0; r < ranks.size(); ++r)
        {
            MICROPROFILE_SCOPEI("Loader", "Chunk", 0xffffd060);
            uint32_t i = ranks[r].idx;
            const ChunkEntry& ce = entries[i];
            {
                MICROPROFILE_SCOPEI("Loader", "ReadBlob", 0xffd08040);
                auto t0 = clk::now();
                diskBlob.resize(ce.blobBytes);
                if (!ReadAt(f, ce.blobOffset, diskBlob.data(), ce.blobBytes))
                {
                    fclose(f);
                    err = "chunk blob read";
                    return false;
                }
                tReadMs += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
                bytesRead += ce.blobBytes;
            }
            const uint8_t* blobPtr = diskBlob.data();
            size_t blobSize = diskBlob.size();
            if (ce.flags & kFlagLz4)
            {
                MICROPROFILE_SCOPEI("Loader", "LZ4", 0xff60d060);
                auto t0 = clk::now();
                rawBlob.resize(ce.blobBytesRaw);
                int dec = LZ4_decompress_safe((const char*)diskBlob.data(),
                                              (char*)rawBlob.data(),
                                              (int)ce.blobBytes,
                                              (int)ce.blobBytesRaw);
                if (dec != (int)ce.blobBytesRaw)
                {
                    fclose(f);
                    err = "lz4 decompress failed";
                    return false;
                }
                blobPtr = rawBlob.data();
                blobSize = rawBlob.size();
                tLz4Ms += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
                bytesLz4Raw += ce.blobBytesRaw;
            }
            // Parse common header.
            auto tHdr0 = clk::now();
            const uint8_t* p = blobPtr;
            const uint8_t* end = blobPtr + blobSize;
            if ((size_t)(end - p) < sizeof(DiskChunkHeader))
            {
                fclose(f);
                err = "blob too small for header";
                return false;
            }
            DiskChunkHeader dch;
            memcpy(&dch, p, sizeof(dch));
            p += sizeof(dch);

            const size_t palBytes = dch.paletteCount * sizeof(uint32_t);
            if ((size_t)(end - p) < palBytes)
            {
                fclose(f);
                err = "blob too small for palette";
                return false;
            }
            RuntimeChunk& rc = lw.chunks[i];
            rc.gridX = dch.gridX;
            rc.gridY = dch.gridY;
            rc.gridZ = dch.gridZ;
            rc.worldOriginX = dch.worldOriginX;
            rc.worldOriginY = dch.worldOriginY;
            rc.worldOriginZ = dch.worldOriginZ;
            rc.lodLevel = (uint8_t)dch.lodLevel;
            rc.aabbMin[0] = dch.aabbMin[0];
            rc.aabbMin[1] = dch.aabbMin[1];
            rc.aabbMin[2] = dch.aabbMin[2];
            rc.aabbMax[0] = dch.aabbMax[0];
            rc.aabbMax[1] = dch.aabbMax[1];
            rc.aabbMax[2] = dch.aabbMax[2];
            for (int k = 0; k < 8; ++k)
                rc.childId[k] = dch.childId[k];
            rc.paletteCount = dch.paletteCount;
            memset(rc.palette, 0, sizeof(rc.palette));
            if (palBytes)
                memcpy(rc.palette, p, palBytes);
            p += palBytes;

            rc.poolBase = 0;
            rc.poolCount = 1; // dummy non-zero so visit()/childChunks check passes
            rc.slotIdx = i;
            memset(rc.clusters, 0, sizeof(rc.clusters));
            // Per-cluster numPoints + bounds filled below as the V3 stream walk
            // visits each cluster. Clusters absent from the stream stay 0 so
            // visitCluster() early-returns instead of doing 128 frustum tests
            // per chunk.
            tHeaderMs += std::chrono::duration<double, std::milli>(clk::now() - tHdr0).count();

            // ---- V3 compact octet stream (PointCS_Block) — only payload now ----
            rc.blockBase = (uint32_t)lw.blockPosPool.size();
            rc.blockCount = 0;
            memset(rc.clusterBlockFirst, 0, sizeof(rc.clusterBlockFirst));
            memset(rc.clusterBlockCount, 0, sizeof(rc.clusterBlockCount));
            rc.blockPointBase = (uint32_t)lw.blockPointPool.size();
            rc.blockPointCount = 0;
            memset(rc.clusterPointFirst, 0, sizeof(rc.clusterPointFirst));
            memset(rc.clusterPointCount, 0, sizeof(rc.clusterPointCount));
            auto tBlk0 = clk::now();
            if (ce.flags & kFlagBlocks)
            {
                MICROPROFILE_SCOPEI("Loader", "BlocksV3", 0xff80c0c0);
                {
                    bool clusterDone = false;
                    while (!clusterDone)
                    {
                        // EOF checks disabled for perf
                        uint8_t cid = *p++;
                        bool lastCluster = (cid & 0x80u) != 0;
                        uint32_t ci = (uint32_t)(cid & 0x7Fu);
                        uint32_t clusterBlockFirst = (uint32_t)lw.blockPosPool.size() - rc.blockBase;
                        uint32_t clusterPointFirst = (uint32_t)lw.blockPointPool.size() - rc.blockPointBase;
                        uint32_t clusterPointEmitted = 0;
                        uint32_t cx = ci % kClustersX;
                        uint32_t cy = (ci / kClustersX) % kClustersY;
                        uint32_t cz = ci / (kClustersX * kClustersY);
                        uint32_t clusterOriginVoxX = cx * kClusterVoxX;
                        uint32_t clusterOriginVoxY = cy * kClusterVoxY;
                        uint32_t clusterOriginVoxZ = cz * kClusterVoxZ;

                        // Per-cluster state.
                        bool clusterEnd = false;
                        uint32_t implicitRemaining = 0;
                        uint32_t curOctetIdx = 0;
                        uint32_t octetCount = 0;
                        uint8_t modeByte = 0;
                        int prevColor = -1;
                        // Tight per-cluster voxel-AABB (cluster-local 0..31 per axis).
                        uint32_t cMinX = 31, cMinY = 31, cMinZ = 31;
                        uint32_t cMaxX = 0, cMaxY = 0, cMaxZ = 0;
                        while (!clusterEnd)
                        {
                            if ((octetCount & 3u) == 0u)
                                modeByte = *p++;
                            uint8_t mode = (uint8_t)((modeByte >> ((octetCount & 3u) * 2u)) & 0x3u);

                            if (implicitRemaining == 0)
                            {
                                uint16_t oid;
                                memcpy(&oid, p, 2);
                                p += 2;
                                curOctetIdx = (uint32_t)(oid & 0x0FFFu);
                                uint32_t runBits = (uint32_t)((oid >> 12) & 0x0Fu);
                                if (runBits == 15u)
                                {
                                    clusterEnd = true;
                                    implicitRemaining = 0;
                                }
                                else
                                {
                                    implicitRemaining = runBits;
                                }
                            }
                            else
                            {
                                curOctetIdx += 1u;
                                --implicitRemaining;
                            }

                            uint8_t mask = *p++;

                            uint8_t palFull[8] = {};
                            if (mode == kOctetModeUniformReuse)
                            {
                                uint8_t col = (uint8_t)prevColor;
                                for (int vi = 0; vi < 8; ++vi)
                                {
                                    if (mask & (1u << vi))
                                        palFull[vi] = col;
                                }
                            }
                            else if (mode == kOctetModeUniformNew)
                            {
                                uint8_t col = *p++;
                                prevColor = (int)col;
                                for (int vi = 0; vi < 8; ++vi)
                                {
                                    if (mask & (1u << vi))
                                        palFull[vi] = col;
                                }
                            }
                            else // kOctetModeVaried (mode 3 reserved, treat same)
                            {
                                uint32_t pop = 0;
                                for (uint8_t m = mask; m; m &= m - 1)
                                    ++pop;
                                const uint8_t* src = p;
                                p += pop;
                                uint32_t k = 0;
                                for (int vi = 0; vi < 8; ++vi)
                                {
                                    if (mask & (1u << vi))
                                        palFull[vi] = src[k++];
                                }
                            }

                            uint32_t ox = curOctetIdx & 0xFu;
                            uint32_t oz = (curOctetIdx >> 4) & 0xFu;
                            uint32_t oy = (curOctetIdx >> 8) & 0xFu;
                            uint32_t blockX = (clusterOriginVoxX >> 1) + ox;
                            uint32_t blockY = (clusterOriginVoxY >> 1) + oy;
                            uint32_t blockZ = (clusterOriginVoxZ >> 1) + oz;
                            BlockPos bp{};
                            bp.blockX = (uint8_t)blockX;
                            bp.blockY = (uint8_t)blockY;
                            bp.blockZ = (uint8_t)blockZ;
                            bp.occupancy = mask;
                            lw.blockPosPool.push_back(bp);
                            // Tight AABB in cluster-local LOD-voxel coords.
                            // Block covers voxels [block*2, block*2+1]; subtract cluster origin in voxels.
                            uint32_t lvxMin = (blockX * 2u) - clusterOriginVoxX;
                            uint32_t lvxMax = lvxMin + 1u;
                            uint32_t lvyMin = (blockY * 2u) - clusterOriginVoxY;
                            uint32_t lvyMax = lvyMin + 1u;
                            uint32_t lvzMin = (blockZ * 2u) - clusterOriginVoxZ;
                            uint32_t lvzMax = lvzMin + 1u;
                            if (lvxMin < cMinX) cMinX = lvxMin;
                            if (lvyMin < cMinY) cMinY = lvyMin;
                            if (lvzMin < cMinZ) cMinZ = lvzMin;
                            if (lvxMax > cMaxX) cMaxX = lvxMax;
                            if (lvyMax > cMaxY) cMaxY = lvyMax;
                            if (lvzMax > cMaxZ) cMaxZ = lvzMax;
                            BlockCol bc{};
                            for (int vi = 0; vi < 8; ++vi)
                                bc.palIdx[vi] = palFull[vi];
                            lw.blockColPool.push_back(bc);
                            // PointCS A/B: expand each occupied voxel to a packed uint.
                            for (int vi = 0; vi < 8; ++vi)
                            {
                                if (!(mask & (1u << vi))) continue;
                                uint32_t lvx = (vi >> 0) & 1u;
                                uint32_t lvy = (vi >> 1) & 1u;
                                uint32_t lvz = (vi >> 2) & 1u;
                                uint32_t vx = blockX * 2u + lvx;
                                uint32_t vy = blockY * 2u + lvy;
                                uint32_t vz = blockZ * 2u + lvz;
                                uint32_t pck = (vx & 0xFFu)
                                              | ((vy & 0xFFu) << 8)
                                              | ((vz & 0xFFu) << 16)
                                              | ((uint32_t)palFull[vi] << 24);
                                lw.blockPointPool.push_back(pck);
                                ++clusterPointEmitted;
                                ++rc.blockPointCount;
                            }
                            ++rc.blockCount;
                            ++octetCount;
                            if (clusterEnd && implicitRemaining == 0)
                                break;
                        }
                        // Record per-cluster block range (relative to chunk's blockBase).
                        if (ci < (uint32_t)kClustersPerChunk)
                        {
                            rc.clusterBlockFirst[ci] = clusterBlockFirst;
                            rc.clusterBlockCount[ci] = octetCount;
                            rc.clusterPointFirst[ci] = clusterPointFirst;
                            rc.clusterPointCount[ci] = clusterPointEmitted;
                            if (octetCount > 0)
                            {
                                rc.clusters[ci].numPoints = 1; // visitCluster gate
                                rc.clusters[ci].pointFirst = 0;
                                rc.clusters[ci].bounds = PackClusterBounds(
                                    cMinX, cMinY, cMinZ, cMaxX, cMaxY, cMaxZ);
                            }
                        }
                        if (lastCluster)
                            clusterDone = true;
                    }
                }
                tBlocksMs += std::chrono::duration<double, std::milli>(clk::now() - tBlk0).count();
                totalBlocks += rc.blockCount;
                // Reflect block count into poolCount so visit() / childLoaded
                // gating treats this chunk as resident only when it has blocks.
                rc.poolCount = rc.blockCount;
            }
        } // end of for(size_t r ...) chunk loop

        // ---- Build SoA cull arrays (world float AABBs) ----
        auto tCull0 = clk::now();
        lw.cull.minX.resize(cc);
        lw.cull.minY.resize(cc);
        lw.cull.minZ.resize(cc);
        lw.cull.maxX.resize(cc);
        lw.cull.maxY.resize(cc);
        lw.cull.maxZ.resize(cc);
        lw.cull.culled.assign(cc, 0);
        for (uint32_t i = 0; i < cc; ++i)
        {
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
        tCullMs = std::chrono::duration<double, std::milli>(clk::now() - tCull0).count();

        // Per-LOD timing summary.
        double tTotalMs = std::chrono::duration<double, std::milli>(clk::now() - tLodStart).count();
        double tAccountedMs = tReadMs + tLz4Ms + tHeaderMs + tBlocksMs + tRankMs + tCullMs;
        double tOtherMs = tTotalMs - tAccountedMs;
        std::printf("[Loader] LOD %d  total=%.1fms  read=%.1f  lz4=%.1f  hdr=%.1f  blocks=%.1f (%llu)  rank=%.1f  cull=%.1f  other=%.1f  ranked=%zu/%u  bytes=%.1f/%.1fMB\n",
                    L, tTotalMs, tReadMs, tLz4Ms, tHeaderMs,
                    tBlocksMs, (unsigned long long)totalBlocks,
                    tRankMs, tCullMs, tOtherMs,
                    ranks.size(), cc,
                    bytesRead / (1024.0 * 1024.0), bytesLz4Raw / (1024.0 * 1024.0));
        std::fflush(stdout);

        // Notify caller this LOD is ready for upload.
        if (onLodReady)
            onLodReady(user, L);
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
