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
        double tReadMs = 0, tLz4Ms = 0, tDecodeMs = 0;
        uint64_t bytesRead = 0, bytesLz4Raw = 0;
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
                if ((size_t)(end - p) < 16) { fclose(f); err = "blob short for clusterMask"; return false; }
                uint8_t clusterMask[16];
                memcpy(clusterMask, p, 16); p += 16;

                memset(rc.clusters, 0, sizeof(rc.clusters));
                lw.pointPool.resize(rc.poolBase + dch.totalPoints);

                // Pass 1: parse all sub-blobs, decode bit-grids, fill chunk-wide bit-grid.
                struct ClusterDec {
                    uint8_t  orderMode;
                    uint32_t nP;
                    uint8_t  bits[kClusterCellCount];
                    const uint8_t* colors;
                    const uint8_t* ao;        // 3 bytes per emitted point (per-face packed)
                    const uint8_t* cellAo;
                    uint32_t cellAoCount;
                    int oX, oY, oZ;
                };
                std::vector<ClusterDec> cds(kClustersPerChunk);
                const int CW = kChunkVoxX, CH = kChunkVoxY, CD = kChunkVoxZ;
                std::vector<uint8_t> chunkBits((size_t)((CW * CH * CD + 7) / 8), 0);
                auto cbSet = [&](int x, int y, int z) {
                    size_t idx = (size_t)((y * CD + z) * CW + x);
                    chunkBits[idx >> 3] |= (uint8_t)(1u << (idx & 7));
                };
                auto cbGet = [&](int x, int y, int z) -> bool {
                    if (x < 0 || y < 0 || z < 0 || x >= CW || y >= CH || z >= CD) return false;
                    size_t idx = (size_t)((y * CD + z) * CW + x);
                    return (chunkBits[idx >> 3] >> (idx & 7)) & 1u;
                };

                for (int s = 0; s < kClustersPerChunk; ++s) {
                    if (!(clusterMask[s >> 3] & (1u << (s & 7)))) continue;
                    if (p >= end) { fclose(f); err = "blob short cluster stream"; return false; }
                    ClusterDec& cd = cds[s];
                    cd.orderMode = *p++;
                    cd.nP = Leb128GetU32(p);
                    uint32_t bgSize = Leb128GetU32(p);
                    if (bgSize > (uint32_t)(end - p)) { fclose(f); err = "bgSize remaining"; return false; }
                    RleDecodeBitGrid(p, bgSize, cd.bits);
                    p += bgSize;
                    if (cd.nP > (uint32_t)(end - p)) { fclose(f); err = "colors remaining"; return false; }
                    cd.colors = p; p += cd.nP;
                    if (ce.flags & kFlagAo) {
                        // Variable-length: leb128 byte count, then packed
                        // nibbles per visMask-set face per voxel.
                        uint32_t aoBytes = Leb128GetU32(p);
                        if (aoBytes > (uint32_t)(end - p)) { fclose(f); err = "ao remaining"; return false; }
                        cd.ao = p;
                        p += aoBytes;
                    } else {
                        cd.ao = nullptr;
                    }
                    if (ce.flags & kFlagVisMask) {
                        if (cd.nP > (uint32_t)(end - p)) { fclose(f); err = "vm remaining"; return false; }
                        p += cd.nP;
                    }
                    if (ce.flags & kFlagCellAo) {
                        cd.cellAoCount = Leb128GetU32(p);
                        uint32_t cellAoBytes = (cd.cellAoCount + 1) / 2;
                        if (cellAoBytes > (uint32_t)(end - p)) { fclose(f); err = "cellao remaining"; return false; }
                        cd.cellAo = p;
                        p += cellAoBytes;
                    } else {
                        cd.cellAo = nullptr;
                        cd.cellAoCount = 0;
                    }
                    cd.oZ = (s / (kClustersX * kClustersY)) * kClusterVoxZ;
                    cd.oY = ((s / kClustersX) % kClustersY) * kClusterVoxY;
                    cd.oX = (s % kClustersX) * kClusterVoxX;
                    // Fill chunk-wide bit-grid.
                    for (uint32_t cidx = 0; cidx < kClusterCellCount; ++cidx) {
                        if (!cd.bits[cidx]) continue;
                        uint32_t lx, ly, lz;
                        LwCellCoord((LwOrderMode)cd.orderMode, cidx, lx, ly, lz);
                        cbSet(cd.oX + lx, cd.oY + ly, cd.oZ + lz);
                    }
                }

                // Pass 2: build chunk-wide cell-AO grid from per-cluster cellAo streams.
                std::vector<uint8_t> chunkCellAo;
                if (ce.flags & kFlagCellAo) {
                    chunkCellAo.assign((size_t)((CW * CH * CD + 1) / 2), 0);
                    auto caSet = [&](int x, int y, int z, uint8_t ao4) {
                        size_t idx = (size_t)((y * CD + z) * CW + x);
                        uint8_t& byte = chunkCellAo[idx >> 1];
                        if (idx & 1) byte = (byte & 0x0F) | (uint8_t)(ao4 << 4);
                        else         byte = (byte & 0xF0) | (uint8_t)(ao4 & 0x0F);
                    };
                    for (int s = 0; s < kClustersPerChunk; ++s) {
                        if (!(clusterMask[s >> 3] & (1u << (s & 7)))) continue;
                        const ClusterDec& cd = cds[s];
                        if (!cd.cellAo) continue;
                        uint32_t consumed = 0;
                        for (uint32_t cidx = 0; cidx < kClusterCellCount; ++cidx) {
                            if (cd.bits[cidx]) continue;
                            uint32_t lx, ly, lz;
                            LwCellCoord((LwOrderMode)cd.orderMode, cidx, lx, ly, lz);
                            int wx = cd.oX + lx, wy = cd.oY + ly, wz = cd.oZ + lz;
                            if (!(cbGet(wx+1,wy,wz) || cbGet(wx-1,wy,wz) ||
                                  cbGet(wx,wy+1,wz) || cbGet(wx,wy-1,wz) ||
                                  cbGet(wx,wy,wz+1) || cbGet(wx,wy,wz-1))) continue;
                            if (consumed >= cd.cellAoCount) break;
                            uint8_t b = cd.cellAo[consumed >> 1];
                            uint8_t ao4 = (consumed & 1) ? (b >> 4) : (b & 0x0F);
                            caSet(wx, wy, wz, ao4);
                            ++consumed;
                        }
                    }
                }
                bool haveCellAo = (ce.flags & kFlagCellAo) != 0;
                auto caGet = [&](int x, int y, int z) -> uint8_t {
                    if (!haveCellAo) return 0xF;
                    if (x < 0 || y < 0 || z < 0 || x >= CW || y >= CH || z >= CD) return 0xF;
                    size_t idx = (size_t)((y * CD + z) * CW + x);
                    uint8_t byte = chunkCellAo[idx >> 1];
                    return (idx & 1) ? (byte >> 4) : (byte & 0x0F);
                };

                MICROPROFILE_SCOPEI("Loader", "Pass3", 0xffe080e0);
                auto tD0 = clk::now();
                // Pass 3: emit DiskPoints. visMask recomputed from chunk
                // bit-grid; per-face AO unpacked from variable-length nibble
                // stream (one nibble per set visMask bit, packed back-to-back).
                uint32_t writePos = rc.poolBase;
                for (int s = 0; s < kClustersPerChunk; ++s) {
                    if (!(clusterMask[s >> 3] & (1u << (s & 7)))) continue;
                    const ClusterDec& cd = cds[s];
                    uint8_t clMn[3] = { 31, 31, 31 };
                    uint8_t clMx[3] = { 0, 0, 0 };
                    rc.clusters[s].pointFirst = writePos - rc.poolBase;
                    rc.clusters[s].numPoints  = (uint16_t)cd.nP;
                    rc.clusters[s]._pad = 0;
                    uint32_t emitted = 0;
                    uint32_t aoBitPos = 0;       // per-cluster nibble cursor
                    for (uint32_t cidx = 0; cidx < kClusterCellCount; ++cidx) {
                        if (!cd.bits[cidx]) continue;
                        uint32_t lx, ly, lz;
                        LwCellCoord((LwOrderMode)cd.orderMode, cidx, lx, ly, lz);
                        int wx = cd.oX + lx, wy = cd.oY + ly, wz = cd.oZ + lz;
                        DiskPoint dp;
                        dp.posX = (uint8_t)wx;
                        dp.posY = (uint8_t)wy;
                        dp.posZ = (uint8_t)wz;
                        dp.palIdx = cd.colors[emitted];
                        uint8_t mask = 0;
                        if (!cbGet(wx+1,wy,wz)) mask |= 0x01;
                        if (!cbGet(wx-1,wy,wz)) mask |= 0x02;
                        if (!cbGet(wx,wy+1,wz)) mask |= 0x04;
                        if (!cbGet(wx,wy-1,wz)) mask |= 0x08;
                        if (!cbGet(wx,wy,wz+1)) mask |= 0x10;
                        if (!cbGet(wx,wy,wz-1)) mask |= 0x20;
                        dp.visMask = mask;
                        // Unpack variable-length AO nibbles per visMask bit.
                        // Hidden faces get nibble 0 (never sampled at runtime).
                        uint32_t ap = 0;
                        if (cd.ao) {
                            for (int fi = 0; fi < 6; ++fi) {
                                if (!((mask >> fi) & 1u)) continue;
                                uint8_t byte = cd.ao[aoBitPos >> 1];
                                uint8_t nib  = (aoBitPos & 1u) ? (byte >> 4) : (byte & 0x0F);
                                ap |= (uint32_t)nib << (fi * 4);
                                ++aoBitPos;
                            }
                        } else {
                            ap = 0xFFFFFFu;   // all 15 = bright fallback
                        }
                        dp.aoPacked[0] = (uint8_t)(ap & 0xFF);
                        dp.aoPacked[1] = (uint8_t)((ap >> 8) & 0xFF);
                        dp.aoPacked[2] = (uint8_t)((ap >> 16) & 0xFF);
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
                tDecodeMs += std::chrono::duration<double, std::milli>(clk::now() - tD0).count();
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

        // Per-LOD timing summary.
        double tTotalMs = std::chrono::duration<double, std::milli>(clk::now() - tLodStart).count();
        double tOtherMs = tTotalMs - tReadMs - tLz4Ms - tDecodeMs;
        std::printf("[Loader] LOD %d  total=%.1fms  read=%.1f (%.1f MB)  lz4=%.1f (%.1f MB raw)  decode=%.1f  other=%.1f  ranked=%zu/%u\n",
                    L, tTotalMs, tReadMs, bytesRead / (1024.0*1024.0),
                    tLz4Ms, bytesLz4Raw / (1024.0*1024.0),
                    tDecodeMs, tOtherMs, ranks.size(), cc);
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
