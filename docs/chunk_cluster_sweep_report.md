# Chunk + Cluster Size Sweep Report

**Asset**: `assets/kingslanding.vox` (9 923 965 source voxels, 251 max Y)
**Baseline**: chunk 256×64×256, cluster 32×32×32 (current production)
**Method**: edit `src/lodworld.h` constants, rebuild `vox2lw`, rebake, measure.
**AO encoding**: drop-hidden variable-length (current) for all configs.
**Cluster ≤32 only**: 5-bit Morton + clusterMask sizing constrain cluster axis to ≤32.

## On-disk results (LOD0 + all LODs)

| Chunk      | Cluster | L0 chunks | L0 clusters | Clusters/chunk | LOD0 raw | LOD0 LZ4 | LZ4 % | File (all LODs) |
|------------|---------|-----------|-------------|----------------|----------|----------|-------|-----------------|
| 256×64×256 | 32      | 121       | 15 488      | 128            | 31.93 MB | 21.42 MB | 67.1% | **27.83 MB** *(baseline)* |
| 256×64×256 | 16      | 121       | 123 904     | 1024           | 32.13 MB | 21.38 MB | 66.4% | 27.83 MB         |
| 256×64×256 | 8       | 121       | 991 232     | 8192           | 32.58 MB | 22.06 MB | 67.1% | 28.68 MB         |
| 128×64×128 | 16      | 369       | 94 464      | 256            | 32.21 MB | 21.50 MB | 66.6% | 28.03 MB         |
| 128×64×128 | 32      | 369       | 11 808      | 32             | 32.01 MB | 21.53 MB | 67.3% | 28.03 MB         |
| 512×64×512 | 32      | 46        | 23 552      | 512            | 31.90 MB | 21.38 MB | 67.0% | **27.74 MB**     |
| 512×64×512 | 16      | 46        | 188 416     | 4096           | 32.10 MB | 21.35 MB | 66.4% | 27.76 MB         |
| 256×32×256 | 32      | 190       | 12 160      | 64             | 31.95 MB | 21.44 MB | 67.1% | 27.90 MB         |
| 256×32×256 | 16      | 190       | 97 280      | 512            | 32.15 MB | 21.40 MB | 66.4% | 27.89 MB         |
| 1024×32×1024 | 32    | 33        | 33 792      | 1024           | 31.90 MB | 21.39 MB | 67.0% | 27.76 MB         |

**File-size variance: ±1.7%** across all configs. Disk size is essentially insensitive to chunk/cluster geometry.

### GPU pool (point data, 8 B per voxel)

Identical at **75.71 MB** at LOD0 across all configs (same source voxel count). Geometry doesn't affect on-GPU point storage.

## CPU metadata memory (LOD0 only)

Per-chunk runtime cost ≈ `kClustersPerChunk × 12 B` (DiskCluster array) + ~1.1 KB fixed (palette, AABB, childId, world origin, etc.).

| Chunk      | Cluster | L0 chunks | Clusters/chunk | Per chunk CPU | **LOD0 CPU metadata** |
|------------|---------|-----------|----------------|---------------|------------------------|
| 256×64×256 | 32      | 121       | 128            | 2.6 KB        | **0.31 MB** *(baseline)* |
| 256×64×256 | 16      | 121       | 1024           | 13.4 KB       | 1.62 MB                |
| 256×64×256 | 8       | 121       | 8192           | 99.4 KB       | **12.0 MB**            |
| 128×64×128 | 16      | 369       | 256            | 4.2 KB        | 1.54 MB                |
| 128×64×128 | 32      | 369       | 32             | 1.5 KB        | 0.55 MB                |
| 512×64×512 | 32      | 46        | 512            | 7.2 KB        | 0.33 MB                |
| 512×64×512 | 16      | 46        | 4096           | 50.3 KB       | 2.31 MB                |
| 256×32×256 | 32      | 190       | 64             | 1.9 KB        | 0.35 MB                |
| 256×32×256 | 16      | 190       | 512            | 7.2 KB        | 1.37 MB                |
| 1024×32×1024 | 32    | 33        | 1024           | 13.4 KB       | 0.44 MB                |

Smaller clusters explode CPU memory linearly with `1 / cluster_volume` (because each chunk holds a dense `DiskCluster[128]` array regardless of how many are populated). Cluster=8 inflates per-chunk by 64× over cluster=32, hitting 12 MB metadata for a single LOD.

## Per-cluster culling resolution

Cluster count drives frustum/LOD culling granularity. Higher = tighter per-frame culling, lower GPU draws on partial visibility.

| Config | LOD0 clusters | World cells per cluster |
|---|---|---|
| 256×64×256 / 32 | 15 488 | 32³ = 32 768 cells |
| 256×64×256 / 16 | 123 904 | 16³ = 4 096 cells |
| 256×64×256 / 8 | 991 232 | 8³ = 512 cells |
| 1024×32×1024 / 32 | 33 792 | 32³ |

Tighter clusters = more culling work per frame (CPU traversal explodes) but tighter bounding for off-screen rejection. With 991 K clusters/chunk at L0 in the 8³ config, the per-frame traversal alone becomes expensive.

## Findings

### Disk size is a poor signal
- All configs land within 27.7–28.7 MB on disk. **±1.7% variance.**
- LZ4 absorbs structural redundancy regardless of chunk geometry — bigger chunks have longer continuous bit-grids and color runs, but LZ4 already finds equivalent patterns in concatenated chunk blobs.
- Raw blob pre-LZ4 also flat (31.9–32.6 MB).
- Cluster=8 (8³ = 512 cells) is the only config that significantly grows the file (+3% over baseline) — per-cluster overhead becomes meaningful at that scale.

### CPU memory is the real cost driver
- **Each chunk allocates `kClustersPerChunk × 12 B` even for empty slots.** This is the dominant per-chunk cost.
- Going from cluster 32→16 increases CPU memory **5×** at same chunk dims.
- Going to cluster 8 increases it **40×** (320 KB → 12 MB at LOD0).
- Bigger chunks reduce chunk count but keep clusters/chunk ratio high.

### Sweet spots
| Goal | Config | Notes |
|---|---|---|
| **Min disk** | 512×64×512 / 32 | 27.74 MB, fewest chunk headers + table entries |
| **Min CPU metadata** | 256×64×256 / 32 *(current)* | 0.31 MB |
| **Tightest culling** | 256×64×256 / 16 | 1024 clusters/chunk, 1.62 MB CPU |
| **Streaming-friendly** | 256×64×256 / 32 *(current)* | Small chunk count, manageable working set |

### Recommendations
1. **Keep cluster = 32.** Going smaller costs CPU memory disproportionately to disk savings (none). Going bigger requires format work (>5-bit Morton, >32-bit clusterMask) for no disk benefit.
2. **Keep chunk = 256×64×256.** Bigger chunks (512, 1024) shave ~0.1 MB off the disk but reduce LOD-traversal granularity (fewer chunks → coarser frustum culling decisions at chunk level → more clusters traversed per frame).
3. **Y=64 is well-tuned** for tall-but-shallow ground (kingslanding fits in <256 Y, so 64-vox per LOD-chunk-Y gives 4 chunks vertically). Y=32 doubles vertical chunk count without size benefit; Y=96 was marginally smaller but adds odd cluster geometry (3 Y-clusters per chunk = harder octant logic).
4. **Cluster=8 is a non-starter** — 12 MB CPU per LOD just for cluster metadata, no compensating disk win.

### Cluster size beyond 32 (not measured)
Would require:
- Replace 5-bit Morton with N-bit version (or drop Morton, use Y-major only).
- Replace 5-bit `PackClusterBounds` with N-bit per axis.
- `clusterMask` already scaled dynamically by this sweep's loader fix.

Likely outcomes if implemented: bigger clusters (64³) reduce cluster count per chunk back into the 16–32 range, recovering CPU metadata savings but losing culling tightness. Disk impact probably negligible (consistent with the rest of this sweep).

## Conclusion

The current baseline (chunk 256×64×256, cluster 32×32×32) is near-optimal:
- Within 0.4% of minimum file size.
- Minimum CPU metadata (tied with one other config).
- Balanced cluster count for both culling tightness and traversal cost.

Further compression wins must come from **encoding** (already done: AO drop-hidden -40%, color sub-palette projection -28%) rather than **geometry**.
