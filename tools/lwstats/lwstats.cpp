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

                // Walk V2 octet stream to histogram occupancy popcount.
                const uint8_t* op = p + hdrSz + palSz;
                const uint8_t* oend = p + plen;
                bool chunkDone = false;
                while (!chunkDone && op < oend)
                {
                    uint8_t cid = *op++;
                    bool lastCluster = (cid & 0x80u) != 0;
                    bool clusterEnd = false;
                    uint32_t implicitRemaining = 0;
                    while (!clusterEnd && op < oend)
                    {
                        if (implicitRemaining == 0)
                        {
                            if (oend - op < 2) break;
                            uint16_t oid;
                            memcpy(&oid, op, 2);
                            op += 2;
                            uint32_t runBits = (uint32_t)((oid >> 12) & 0x0Fu);
                            if (runBits == 15u) clusterEnd = true;
                            else                implicitRemaining = runBits;
                        }
                        else
                        {
                            --implicitRemaining;
                        }
                        if (op >= oend) break;
                        uint8_t mask = *op++;
                        uint32_t pop = 0;
                        for (uint8_t m = mask; m; m &= m - 1) ++pop;
                        if (pop <= 8) ++popHist[pop];
                        op += pop;   // skip voxelID bytes
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
            printf("      V2 octets:       %.2f MB  (%.1f%%)  [avg %.1f KB/chunk]\n",
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
        printf("    V2 octets:       %.2f MB  (%.1f%% of raw)\n",
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
        printf("\n");
        fclose(f);
    }
    return 0;
}
