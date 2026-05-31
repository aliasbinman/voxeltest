// lwstats — dump per-section byte breakdown of a .lw file.

#define _CRT_SECURE_NO_WARNINGS
#include "lodworld.h"
#include "lz4.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printf("usage: %s file.lw [file.lw ...]\n", argv[0]);
        return 1;
    }

    for (int ai = 1; ai < argc; ++ai)
    {
        const char* path = argv[ai];
        FILE* f = fopen(path, "rb");
        if (!f)
        {
            printf("could not open %s\n", path);
            continue;
        }
        _fseeki64(f, 0, SEEK_END);
        uint64_t fileSize = (uint64_t)_ftelli64(f);
        _fseeki64(f, 0, SEEK_SET);

        lw::FileHeader fh{};
        fread(&fh, sizeof(fh), 1, f);

        lw::LODHeader lh[lw::kLodCount]{};
        fread(lh, sizeof(lw::LODHeader), lw::kLodCount, f);

        const uint64_t fileHdrBytes = sizeof(fh);
        const uint64_t lodHdrBytes = sizeof(lh);

        printf("\n========================================================\n");
        printf("File: %s  (%.2f MB)\n", path, fileSize / 1048576.0);
        printf("========================================================\n");
        printf("  FileHeader:   %llu B\n", (unsigned long long)fileHdrBytes);
        printf("  LODHeader[5]: %llu B\n", (unsigned long long)lodHdrBytes);

        uint64_t sumEntries = 0;
        uint64_t sumBlobsDisk = 0, sumBlobsRaw = 0;
        uint64_t sumHdr = 0, sumPal = 0, sumOctets = 0;
        uint64_t totalChunks = 0;
        uint64_t totalPalEntries = 0;
        uint64_t popHist[9] = {};   // popHist[i] = octets with i occupied voxels (0..8)
        uint64_t slabHist[6] = {};  // slab(uniform color): ±X, ±Y, ±Z
        uint64_t slab4NotUniform = 0; // slab-shaped 4-vox but not uniform color
        uint64_t full8Uniform = 0;    // popcount=8 with single color
        uint64_t full8Mixed   = 0;
        // Bit-stream proposal: per octet 8b mask + 2b mode + payload (8b color | 0b | popcount*8b)
        uint64_t modeUniformNew = 0;   // mode 1: new uniform color
        uint64_t modeUniformReuse = 0; // mode 2: same as previous
        uint64_t modeVaried = 0;       // mode 3: per-voxel
        uint64_t bitStreamBits = 0;    // sum of bits per octet under proposed scheme
        uint64_t octExplicit = 0;       // octets with octetID prefix
        uint64_t octImplicit = 0;       // octets that follow a run (no octetID)

        std::vector<uint8_t> diskBlob, rawBlob;
        for (int L = 0; L < lw::kLodCount; ++L)
        {
            const uint32_t cc = lh[L].chunkCount;
            if (cc == 0)
            {
                printf("\n  LOD %d: (empty)\n", L);
                continue;
            }
            uint64_t entryBytes = (uint64_t)cc * sizeof(lw::ChunkEntry);
            std::vector<lw::ChunkEntry> entries(cc);
            _fseeki64(f, (long long)lh[L].chunkTableOffset, SEEK_SET);
            fread(entries.data(), sizeof(lw::ChunkEntry), cc, f);

            uint64_t blobDisk = 0, blobRaw = 0;
            uint64_t hdrBytes = 0, palBytes = 0, octBytes = 0;
            uint64_t palEntries = 0;
            for (uint32_t i = 0; i < cc; ++i)
            {
                const lw::ChunkEntry& ce = entries[i];
                blobDisk += ce.blobBytes;
                blobRaw += ce.blobBytesRaw;
                diskBlob.resize(ce.blobBytes);
                _fseeki64(f, (long long)ce.blobOffset, SEEK_SET);
                fread(diskBlob.data(), 1, ce.blobBytes, f);

                const uint8_t* p = diskBlob.data();
                size_t plen = diskBlob.size();
                if (ce.flags & lw::kFlagLz4)
                {
                    rawBlob.resize(ce.blobBytesRaw);
                    int dec = LZ4_decompress_safe((const char*)diskBlob.data(),
                                                  (char*)rawBlob.data(),
                                                  (int)ce.blobBytes,
                                                  (int)ce.blobBytesRaw);
                    if (dec != (int)ce.blobBytesRaw)
                    {
                        printf("  WARN: LZ4 fail at LOD %d chunk %u\n", L, i);
                        continue;
                    }
                    p = rawBlob.data();
                    plen = rawBlob.size();
                }
                lw::DiskChunkHeader dch{};
                memcpy(&dch, p, sizeof(dch));
                uint64_t hdrSz = sizeof(dch);
                uint64_t palSz = (uint64_t)dch.paletteCount * 4u;
                uint64_t octSz = plen - hdrSz - palSz;
                hdrBytes += hdrSz;
                palBytes += palSz;
                octBytes += octSz;
                palEntries += dch.paletteCount;

                // Walk V3 octet stream to histogram occupancy popcount.
                const uint8_t* op = p + hdrSz + palSz;
                const uint8_t* oend = p + plen;
                bool chunkDone = false;
                while (!chunkDone && op < oend)
                {
                    uint8_t cid = *op++;
                    bool lastCluster = (cid & 0x80u) != 0;
                    bool clusterEnd = false;
                    uint32_t implicitRemaining = 0;
                    uint32_t octetCount = 0;
                    uint8_t modeByte = 0;
                    int prevColor = -1;
                    while (!clusterEnd && op < oend)
                    {
                        // ModeByte every 4 octets.
                        if ((octetCount & 3u) == 0u)
                        {
                            if (op >= oend) break;
                            modeByte = *op++;
                        }
                        uint8_t mode = (uint8_t)((modeByte >> ((octetCount & 3u) * 2u)) & 0x3u);

                        if (implicitRemaining == 0)
                        {
                            if (oend - op < 2) break;
                            uint16_t oid;
                            memcpy(&oid, op, 2);
                            op += 2;
                            uint32_t runBits = (uint32_t)((oid >> 12) & 0x0Fu);
                            if (runBits == 15u) clusterEnd = true;
                            else                implicitRemaining = runBits;
                            ++octExplicit;
                        }
                        else
                        {
                            --implicitRemaining;
                            ++octImplicit;
                        }
                        if (op >= oend) break;
                        uint8_t mask = *op++;
                        uint32_t pop = 0;
                        for (uint8_t m = mask; m; m &= m - 1) ++pop;
                        if (pop <= 8) ++popHist[pop];

                        // Resolve effective per-voxel colours from mode.
                        uint8_t cols[8] = {};
                        bool uniform = false;
                        if (mode == 0)
                        {
                            uniform = true;
                            uint8_t c = (uint8_t)(prevColor < 0 ? 0 : prevColor);
                            for (int vi = 0; vi < 8; ++vi)
                                if (mask & (1u << vi)) cols[vi] = c;
                            ++modeUniformReuse;
                        }
                        else if (mode == 1)
                        {
                            if (op >= oend) break;
                            uint8_t c = *op++;
                            prevColor = (int)c;
                            uniform = true;
                            for (int vi = 0; vi < 8; ++vi)
                                if (mask & (1u << vi)) cols[vi] = c;
                            ++modeUniformNew;
                        }
                        else
                        {
                            if ((uint64_t)(oend - op) < pop) break;
                            uint32_t k = 0;
                            for (int vi = 0; vi < 8; ++vi)
                                if (mask & (1u << vi)) cols[vi] = op[k++];
                            op += pop;
                            ++modeVaried;
                        }

                        // Compact ordered palette bytes (in bit order) for slab analysis.
                        uint8_t idsBegin[8];
                        {
                            uint32_t k = 0;
                            for (int vi = 0; vi < 8; ++vi)
                                if (mask & (1u << vi)) idsBegin[k++] = cols[vi];
                        }
                        if (pop == 4)
                        {
                            static const uint8_t kSlabMasks[6] = {
                                0x55, 0xAA, 0x33, 0xCC, 0x0F, 0xF0
                            };
                            int slabIdx = -1;
                            for (int s = 0; s < 6; ++s)
                            {
                                if (mask == kSlabMasks[s]) { slabIdx = s; break; }
                            }
                            if (slabIdx >= 0)
                            {
                                if (uniform) ++slabHist[slabIdx];
                                else         ++slab4NotUniform;
                            }
                        }
                        else if (pop == 8)
                        {
                            if (uniform) ++full8Uniform;
                            else         ++full8Mixed;
                        }

                        // Bit-stream proposal estimate (unchanged).
                        if (pop > 0)
                        {
                            uint64_t bitsThisOctet = 8 + 2;
                            if (mode == 0)         { /* reuse: no payload */ }
                            else if (mode == 1)    { bitsThisOctet += 8; }
                            else                   { bitsThisOctet += 8u * pop; }
                            bitStreamBits += bitsThisOctet;
                        }

                        ++octetCount;
                        if (clusterEnd && implicitRemaining == 0) break;
                    }
                    if (lastCluster) chunkDone = true;
                }
            }

            double rawMB = blobRaw / 1048576.0;
            double diskMB = blobDisk / 1048576.0;
            double ratio = blobDisk ? (double)blobRaw / blobDisk : 0.0;

            printf("\n  LOD %d: %u chunks\n", L, cc);
            printf("    ChunkEntry table: %.2f KB\n", entryBytes / 1024.0);
            printf("    Blobs on-disk:    %.2f MB  (LZ4 in)\n", diskMB);
            printf("    Blobs raw:        %.2f MB  (LZ4 out, %.2fx ratio)\n", rawMB, ratio);
            printf("    Inside raw blobs:\n");
            printf("      DiskChunkHeader: %.2f KB  (%.1f%% of raw)  [60B/chunk]\n",
                   hdrBytes / 1024.0, blobRaw ? 100.0 * hdrBytes / blobRaw : 0.0);
            printf("      Palette:         %.2f MB  (%.1f%%)  [avg %.1f entries/chunk]\n",
                   palBytes / 1048576.0,
                   blobRaw ? 100.0 * palBytes / blobRaw : 0.0,
                   cc ? (double)palEntries / cc : 0.0);
            printf("      V3 octets:       %.2f MB  (%.1f%%)  [avg %.1f KB/chunk]\n",
                   octBytes / 1048576.0,
                   blobRaw ? 100.0 * octBytes / blobRaw : 0.0,
                   cc ? (octBytes / 1024.0) / cc : 0.0);

            sumEntries += entryBytes;
            sumBlobsDisk += blobDisk;
            sumBlobsRaw += blobRaw;
            sumHdr += hdrBytes;
            sumPal += palBytes;
            sumOctets += octBytes;
            totalChunks += cc;
            totalPalEntries += palEntries;
        }

        uint64_t sumHeaders = fileHdrBytes + lodHdrBytes + sumEntries;
        double pctHeaders = 100.0 * sumHeaders / fileSize;
        double pctBlobsDisk = 100.0 * sumBlobsDisk / fileSize;
        printf("\n  ===== TOTALS =====\n");
        printf("  Chunks: %llu  Palette entries: %llu  (avg %.1f/chunk)\n",
               (unsigned long long)totalChunks,
               (unsigned long long)totalPalEntries,
               totalChunks ? (double)totalPalEntries / totalChunks : 0.0);
        printf("  Headers (file+LOD+entries): %.2f KB  (%.2f%% of file)\n",
               sumHeaders / 1024.0, pctHeaders);
        printf("  Blobs on-disk total:        %.2f MB  (%.2f%% of file)\n",
               sumBlobsDisk / 1048576.0, pctBlobsDisk);
        printf("  Blobs raw inflated:         %.2f MB  (LZ4 ratio %.2fx)\n",
               sumBlobsRaw / 1048576.0,
               sumBlobsDisk ? (double)sumBlobsRaw / sumBlobsDisk : 0.0);
        printf("  Raw breakdown:\n");
        printf("    DiskChunkHeader: %.2f KB  (%.1f%% of raw)\n",
               sumHdr / 1024.0, sumBlobsRaw ? 100.0 * sumHdr / sumBlobsRaw : 0.0);
        printf("    Palette:         %.2f MB  (%.1f%% of raw)\n",
               sumPal / 1048576.0, sumBlobsRaw ? 100.0 * sumPal / sumBlobsRaw : 0.0);
        printf("    V3 octets:       %.2f MB  (%.1f%% of raw)\n",
               sumOctets / 1048576.0, sumBlobsRaw ? 100.0 * sumOctets / sumBlobsRaw : 0.0);
        uint64_t popTotal = 0;
        for (int i = 1; i <= 8; ++i) popTotal += popHist[i];
        printf("  Octet popcount histogram (occupied voxels per octet):\n");
        printf("    voxels  count        %% of octets\n");
        for (int i = 1; i <= 8; ++i)
        {
            double pct = popTotal ? 100.0 * popHist[i] / popTotal : 0.0;
            int bars = (int)(pct * 0.5);
            if (bars > 60) bars = 60;
            std::string bar(bars, '#');
            printf("    %d       %10llu   %5.1f%%  %s\n",
                   i, (unsigned long long)popHist[i], pct, bar.c_str());
        }
        printf("    total   %10llu\n", (unsigned long long)popTotal);

        uint64_t slabUniformTotal = 0;
        for (int s = 0; s < 6; ++s) slabUniformTotal += slabHist[s];
        uint64_t slabAny = slabUniformTotal + slab4NotUniform;
        uint64_t pop4 = popHist[4];
        printf("  4-vox octet analysis (slab = 4 voxels coplanar on one face):\n");
        printf("    4-vox total:                 %10llu\n", (unsigned long long)pop4);
        printf("    slab-shaped (any colour):    %10llu  (%.1f%% of 4-vox)\n",
               (unsigned long long)slabAny, pop4 ? 100.0 * slabAny / pop4 : 0.0);
        printf("    slab + uniform colour:       %10llu  (%.1f%% of 4-vox)  (%.1f%% of ALL octets)\n",
               (unsigned long long)slabUniformTotal,
               pop4 ? 100.0 * slabUniformTotal / pop4 : 0.0,
               popTotal ? 100.0 * slabUniformTotal / popTotal : 0.0);
        static const char* kSlabNames[6] = { "-X", "+X", "-Y", "+Y", "-Z", "+Z" };
        for (int s = 0; s < 6; ++s)
        {
            printf("      slab %s uniform:           %10llu  (%.1f%% of slabs)\n",
                   kSlabNames[s], (unsigned long long)slabHist[s],
                   slabUniformTotal ? 100.0 * slabHist[s] / slabUniformTotal : 0.0);
        }
        printf("  8-vox octets:\n");
        printf("    uniform colour:              %10llu  (%.1f%% of 8-vox)\n",
               (unsigned long long)full8Uniform,
               popHist[8] ? 100.0 * full8Uniform / popHist[8] : 0.0);
        printf("    mixed:                       %10llu\n", (unsigned long long)full8Mixed);

        uint64_t modeTotal = modeUniformNew + modeUniformReuse + modeVaried;
        printf("  Bit-stream proposal (8b mask + 2b mode + payload):\n");
        printf("    mode 1 (uniform, new):       %10llu  (%.1f%%)\n",
               (unsigned long long)modeUniformNew,
               modeTotal ? 100.0 * modeUniformNew / modeTotal : 0.0);
        printf("    mode 2 (uniform, reuse):     %10llu  (%.1f%%)\n",
               (unsigned long long)modeUniformReuse,
               modeTotal ? 100.0 * modeUniformReuse / modeTotal : 0.0);
        printf("    mode 3 (per-voxel varied):   %10llu  (%.1f%%)\n",
               (unsigned long long)modeVaried,
               modeTotal ? 100.0 * modeVaried / modeTotal : 0.0);
        double bsBytes = bitStreamBits / 8.0;
        double bsMB = bsBytes / 1048576.0;
        double bsPerOctet = modeTotal ? (double)bitStreamBits / modeTotal : 0.0;
        printf("    total bits:       %14llu  (%.2f MB)\n",
               (unsigned long long)bitStreamBits, bsMB);
        printf("    bits per octet:   %.2f  (= %.2f B/octet)\n",
               bsPerOctet, bsPerOctet / 8.0);
        printf("    vs current %.2f B/octet raw  →  %.0f%% of current raw\n",
               sumOctets / (double)modeTotal,
               modeTotal ? 100.0 * bsBytes / (double)sumOctets : 0.0);
        printf("\n");
        fclose(f);
    }
    return 0;
}
