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
    return LoadWorldStreaming(path, out, err, nullptr, nullptr, nullptr);
}

bool LoadWorldStreaming(const char* path, World& out, std::string& err,
                        LodReadyFn onLodReady, void* user, const StreamCfg* cfg)
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

    // ---- Per-LOD load order ----
    // 1) Coarsest (kLodCount-1) first  -> instant whole-world coarse view.
    // 2) Finest (LOD0) next             -> high detail near camera (tightest shell, fewest chunks).
    // 3) Mid LODs in ascending order    -> progressive refinement of the middle distance band.
    // Total chunks ordered so the user sees coarse + sharpest detail quickly.
    using clk = std::chrono::steady_clock;
    int lodOrder[kLodCount];
    lodOrder[0] = kLodCount - 1;
    for (int i = 1; i < kLodCount; ++i) lodOrder[i] = i - 1;
    for (int oi = 0; oi < kLodCount; ++oi) {
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
        lw.blockPool.clear();
        const uint32_t cc = lh[L].chunkCount;
        if (cc == 0) continue;

        std::vector<ChunkEntry> entries(cc);
        if (!ReadAt(f, lh[L].chunkTableOffset, entries.data(), cc * sizeof(ChunkEntry))) {
            fclose(f); err = "chunk table read"; return false;
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
        struct ChunkRank { uint32_t idx; float dist2; int prio; };
        std::vector<ChunkRank> ranks;
        ranks.reserve(cc);
        auto aabbInFrustum = [&](float mnx, float mny, float mnz,
                                  float mxx, float mxy, float mxz) -> bool {
            for (int pi = 0; pi < 5; ++pi) {   // sides + near, skip far
                float a = cfg->frustumPlanes[pi][0], b = cfg->frustumPlanes[pi][1];
                float c = cfg->frustumPlanes[pi][2], d = cfg->frustumPlanes[pi][3];
                float px = a >= 0 ? mxx : mnx;
                float py = b >= 0 ? mxy : mny;
                float pz = c >= 0 ? mxz : mnz;
                if (a*px + b*py + c*pz + d < 0.0f) return false;
            }
            return true;
        };
        auto tRank0 = clk::now();
        for (uint32_t i = 0; i < cc; ++i) {
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
            float d2 = dx*dx + dy*dy + dz*dz;
            // Shell decides INCLUSION; frustum decides PRIORITY within shell.
            // Without this gate, in-frustum chunks loaded the entire view
            // direction regardless of distance (LOD0 ended up loading every
            // chunk the camera could see = several GB).
            bool inShell = (shellRadius <= 0.0f) || (d2 <= shellRadius * shellRadius);
            if (!inShell) continue;
            bool inFrust = cfg && cfg->hasFrustum && aabbInFrustum(mnx,mny,mnz,mxx,mxy,mxz);
            ranks.push_back({ i, d2, inFrust ? 0 : 1 });
        }
        std::sort(ranks.begin(), ranks.end(), [](const ChunkRank& a, const ChunkRank& b) {
            if (a.prio != b.prio) return a.prio < b.prio;
            return a.dist2 < b.dist2;
        });
        tRankMs = std::chrono::duration<double, std::milli>(clk::now() - tRank0).count();

        for (size_t r = 0; r < ranks.size(); ++r) {
            MICROPROFILE_SCOPEI("Loader", "Chunk", 0xffffd060);
            uint32_t i = ranks[r].idx;
            const ChunkEntry& ce = entries[i];
            {
                MICROPROFILE_SCOPEI("Loader", "ReadBlob", 0xffd08040);
                auto t0 = clk::now();
                diskBlob.resize(ce.blobBytes);
                if (!ReadAt(f, ce.blobOffset, diskBlob.data(), ce.blobBytes)) {
                    fclose(f); err = "chunk blob read"; return false;
                }
                tReadMs += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
                bytesRead += ce.blobBytes;
            }
            const uint8_t* blobPtr = diskBlob.data();
            size_t blobSize = diskBlob.size();
            if (ce.flags & kFlagLz4) {
                MICROPROFILE_SCOPEI("Loader", "LZ4", 0xff60d060);
                auto t0 = clk::now();
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
                tLz4Ms += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
                bytesLz4Raw += ce.blobBytesRaw;
            }
            // Parse common header.
            auto tHdr0 = clk::now();
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

            rc.poolBase  = 0;
            rc.poolCount = 1;     // dummy non-zero so visit()/childChunks check passes
            rc.slotIdx   = i;
            memset(rc.clusters, 0, sizeof(rc.clusters));
            // HACK: visit() recursion uses cluster.numPoints>0 to descend.
            // Now that pass3 is gone there are no real per-cluster counts.
            // Mark all clusters non-empty with dummy data so visit() walks
            // every chunk and routes to drawListBlock via pickListForChunk.
            // Block dispatch dedups by slot and dispatches chunk's full
            // blockCount once — bogus per-cluster counts never reach a draw.
            for (int s = 0; s < kClustersPerChunk; ++s) {
                rc.clusters[s].numPoints = 1;
                rc.clusters[s].pointFirst = 0;
                rc.clusters[s].bounds = PackClusterBounds(0, 0, 0,
                    kClusterVoxX - 1, kClusterVoxY - 1, kClusterVoxZ - 1);
            }
            tHeaderMs += std::chrono::duration<double, std::milli>(clk::now() - tHdr0).count();

            // ---- V2 compact octet stream (PointCS_Block) — only payload now ----
            rc.blockBase  = (uint32_t)lw.blockPool.size();
            rc.blockCount = 0;
            auto tBlk0 = clk::now();
            if (ce.flags & kFlagBlocks) {
                MICROPROFILE_SCOPEI("Loader", "BlocksV2", 0xff80c0c0);
                {
                    bool clusterDone = false;
                    while (!clusterDone) {
                        if (p >= end) { fclose(f); err = "block clusterID eof"; return false; }
                        uint8_t cid = *p++;
                        bool lastCluster = (cid & 0x80u) != 0;
                        uint32_t ci = (uint32_t)(cid & 0x7Fu);
                        uint32_t cx = ci % kClustersX;
                        uint32_t cy = (ci / kClustersX) % kClustersY;
                        uint32_t cz = ci / (kClustersX * kClustersY);
                        uint32_t clusterOriginVoxX = cx * kClusterVoxX;
                        uint32_t clusterOriginVoxY = cy * kClusterVoxY;
                        uint32_t clusterOriginVoxZ = cz * kClusterVoxZ;

                        // Walk octets in this cluster.
                        bool clusterEnd = false;
                        uint32_t implicitRemaining = 0;
                        uint32_t curOctetIdx = 0;
                        while (!clusterEnd) {
                            if (implicitRemaining == 0) {
                                if ((size_t)(end - p) < 2) { fclose(f); err = "octetID eof"; return false; }
                                uint16_t oid; 
                                memcpy(&oid, p, 2); 
                                p += 2;
                                curOctetIdx = (uint32_t)(oid & 0x0FFFu);
                                uint32_t runBits = (uint32_t)((oid >> 12) & 0x0Fu);
                                if (runBits == 15u) {
                                    clusterEnd = true;       // terminator after emitting this octet
                                    implicitRemaining = 0;
                                } else {
                                    implicitRemaining = runBits;
                                }
                            } else {
                                curOctetIdx += 1u;
                                --implicitRemaining;
                            }
                            // Emit one block.
                            if ((size_t)(end - p) < 1) { fclose(f); err = "voxelMask eof"; return false; }
                            uint8_t mask = *p++;
                            uint32_t pop = 0;
                            for (uint8_t m = mask; m; m &= m - 1) 
                                ++pop;

                            if ((size_t)(end - p) < pop) 
                            { 
                                fclose(f); 
                                err = "voxelID stream eof"; 
                                return false; 
                            }
                            uint8_t palBytes[8] = {};
                            const uint8_t* src = p;
                            p += pop;
                            // Expand to 8-slot palIdx[] using mask.
                            uint8_t palFull[8] = {};
                            uint32_t k = 0;
                            for (int vi = 0; vi < 8; ++vi) {
                                if (mask & (1u << vi)) palFull[vi] = src[k++];
                            }
                            // Decode octet idx (Y-major) to (ox, oy, oz) in cluster.
                            uint32_t ox = curOctetIdx & 0xFu;
                            uint32_t oz = (curOctetIdx >> 4) & 0xFu;
                            uint32_t oy = (curOctetIdx >> 8) & 0xFu;
                            // Block coord in CHUNK = (clusterOriginVox + octet*2) / 2 = clusterOriginVox/2 + octet.
                            uint32_t blockX = (clusterOriginVoxX >> 1) + ox;
                            uint32_t blockY = (clusterOriginVoxY >> 1) + oy;
                            uint32_t blockZ = (clusterOriginVoxZ >> 1) + oz;
                            DiskBlock b{};
                            b.blockX = (uint8_t)blockX;
                            b.blockY = (uint8_t)blockY;
                            b.blockZ = (uint8_t)blockZ;
                            b.occupancy = mask;
                            for (int vi = 0; vi < 8; ++vi) 
                                b.palIdx[vi] = palFull[vi];
                            // parentRgb565 = avg of voxel palette colors.
                            uint32_t sumR = 0, sumG = 0, sumB = 0, cnt = 0;
                            for (int vi = 0; vi < 8; ++vi) {
                                if (!(mask & (1u << vi))) continue;
                                uint32_t cPal = rc.palette[palFull[vi]];
                                sumR += (cPal >>  0) & 0xFFu;
                                sumG += (cPal >>  8) & 0xFFu;
                                sumB += (cPal >> 16) & 0xFFu;
                                ++cnt;
                            }
                            uint32_t rr = cnt ? (sumR / cnt) : 0u;
                            uint32_t gg = cnt ? (sumG / cnt) : 0u;
                            uint32_t bb = cnt ? (sumB / cnt) : 0u;
                            b.parentRgb565 = (uint16_t)(((rr >> 3) << 11) | ((gg >> 2) << 5) | (bb >> 3));
                            lw.blockPool.push_back(b);
                            ++rc.blockCount;
                            if (clusterEnd && implicitRemaining == 0) break;
                        }
                        if (lastCluster) clusterDone = true;
                    }
                }
            tBlocksMs += std::chrono::duration<double, std::milli>(clk::now() - tBlk0).count();
            totalBlocks += rc.blockCount;
            // Reflect block count into poolCount so visit() / childLoaded
            // gating treats this chunk as resident only when it has blocks.
            rc.poolCount = rc.blockCount;
        }
        }   // end of for(size_t r ...) chunk loop

        // ---- Build SoA cull arrays (world float AABBs) ----
        auto tCull0 = clk::now();
        lw.cull.minX.resize(cc); 
        lw.cull.minY.resize(cc); 
        lw.cull.minZ.resize(cc);
        lw.cull.maxX.resize(cc); 
        lw.cull.maxY.resize(cc); 
        lw.cull.maxZ.resize(cc);
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
                    bytesRead / (1024.0*1024.0), bytesLz4Raw / (1024.0*1024.0));
        std::fflush(stdout);

        // Notify caller this LOD is ready for upload.
        if (onLodReady) onLodReady(user, L);
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
