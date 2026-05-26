# Data Formats

Two on-disk formats, both little-endian, packed (`#pragma pack(push, 1)` where noted). Asset version checked against `kAssetVersion` in `src/asset_version.h`; mismatch → file ignored / regenerated.

---

## `.vox` — raw voxel input

Produced previously by `voxelize.exe` (now removed). Still loaded by `LoadVoxScene` when no baked sidecar exists.

### File layout (sequential)

```
char     magic[4]          = "VXL3"
uint32_t version           = kAssetVersion
uint32_t chunkDim          // voxel cube edge per chunk (e.g. 64)
uint32_t chunkCount
uint32_t totalVoxels       // sum of all chunks' voxelCount
int32_t  origin[3]         // scene world-space origin (voxel coords)
float    sunDir[3]         // baked sun direction
uint32_t paletteCount
uint32_t palette[paletteCount]              // RGBA8 little-endian (alpha = 0xFF, dead byte)
ChunkMeta metas[chunkCount]                 // 16 B each
DiskVoxel voxels[totalVoxels]               // 9 B each
```

### `ChunkMeta` — 16 B

```c++
struct ChunkMeta {
    uint16_t cx, cy, cz, _pad;  // chunk grid coord
    uint32_t voxelCount;        // voxels in this chunk
    uint32_t voxelOffset;       // index into voxels[] where this chunk starts
};
```

### `DiskVoxel` — 9 B

```c++
struct DiskVoxel {
    uint8_t  x, y, z;        // chunk-local position (0..chunkDim-1)
    uint8_t  visMask;        // 6 bits, one per face: +X -X +Y -Y +Z -Z. -Y always 0.
    uint16_t paletteIdx;     // index into palette[]
    uint8_t  aoPacked[3];    // AO per face, 4 bits/face, 6 faces = 24 bits
};
```

`aoPacked` layout: face `fi`'s nibble lives at bit `fi*4` across the 3 bytes. Decode replicates the nibble to a byte (16 levels: 0, 17, 34, ..., 255).

```c++
uint8_t DvAo(DiskVoxel v, int fi) {
    uint8_t n = (v.aoPacked[(fi*4) >> 3] >> ((fi*4) & 7)) & 0xF;
    return (n << 4) | n;
}
```

### Load-time expansion

`LoadVoxScene` (slow path) does the following per chunk:

1. **L0 pass** — for each visible voxel: push one `Vertex` (12 B) to `Scene::pointVertices` and one `uint32_t` AO6 to `Scene::pointAo6`. Resolves `paletteIdx → RGB` via `palette[]`. Per-chunk `SubMesh.pointFirst/pointCount` tracks the range.
2. **L1 / L2 / L3 passes** — for each chunk, bin voxels at step S=2/4/8 in an `unordered_map`. Average color, AO, OR visMask per bin. Push one aggregated `Vertex` + `uint32_t` AO6 per bin. Per-chunk `SubMesh.pointFirstLN/pointCountLN` tracks the range.

All four LOD blocks live contiguously in `pointVertices`: `[L0 chunk0 .. chunkN][L1 chunk0 .. chunkN][L2 ..][L3 ..]`.

After the slow load, `.vxb` sidecar is written so next launch skips this work entirely.

---

## `.vxb` — baked sidecar

Lives next to `.vox` with the same basename (e.g. `kingslanding.vox` → `kingslanding.vxb`). Generated automatically on first slow load. Loaded directly into final `Scene` memory layout — no post-processing.

### Header (88 B)

```c++
struct VxbHeader {
    char     magic[4];      // "VXB1"
    uint32_t version;       // kAssetVersion
    int32_t  origin[3];
    float    sunDir[3];
    float    aabbMin[3];
    float    aabbMax[3];
    uint32_t pointTotal;    // pointVertices.size() (L0+L1+L2+L3 concat)
    uint32_t subCount;      // subs.size() (= visible chunks)
    uint32_t histCount;     // colorHistogram.size()
    uint32_t _pad;
};
```

### File layout

```
VxbHeader header
SubMesh   subs[subCount]                    // 56 B each
Vertex    verts[pointTotal]                 // 12 B each
uint32_t  ao6[pointTotal]                   // 4 B each
{ uint32_t rgb; uint64_t count; }  hist[histCount]   // 12 B each
```

### `SubMesh` — 56 B

```c++
struct SubMesh {
    // Ranges into pointVertices / pointAo6 (parallel arrays).
    uint32_t pointFirst,   pointCount;    // L0 (1 per voxel)
    uint32_t pointFirstL1, pointCountL1;  // L1 (2x2x2 avg)
    uint32_t pointFirstL2, pointCountL2;  // L2 (4x4x4 avg)
    uint32_t pointFirstL3, pointCountL3;  // L3 (8x8x8 avg)
    float    aabbMin[3];   // world-space tight bounds (visible voxels only)
    float    aabbMax[3];
    float    chunkBase[3]; // world-space offset of this chunk (origin + cx*D etc.)
};
```

### `Vertex` — 12 B (final GPU format)

```c++
struct Vertex {
    uint16_t px, py, pz;     // scene-relative voxel coord (16 bpc; supports very large scenes)
    uint16_t aux;            // low 8 bits = baked AO (per-voxel max-of-visible-faces)
    uint32_t color;          // low 24 bits = RGB8; high 8 bits = visMask
};
```

### `ao6` — 4 B per point

`uint32_t`, low 24 bits = 6 faces × 4-bit AO (same packing as `DiskVoxel::aoPacked`). Top byte unused. Parallel to `pointVertices` (same index).

### Color histogram entry — 12 B

```
uint32_t rgb       // 0x00BBGGRR
uint64_t count     // voxel occurrences (L0 only)
```

Sorted descending by `count`.

### Load fast path

`LoadBakedScene` (`src/vox_loader.cpp`):

1. Read `VxbHeader`; reject on magic/version mismatch.
2. Resize `Scene::subs` to `subCount`, `fread` directly.
3. Resize `Scene::pointVertices` and `Scene::pointAo6` to `pointTotal`, `fread` each block.
4. Read `histCount` `(rgb,count)` pairs into `Scene::colorHistogram`.

No allocations beyond the final vectors. No per-voxel work. No GPU-side change vs slow path — `UploadScene` consumes the same `Scene`.

---

## GPU-side mapping

`Renderer::UploadScene` creates:

| Resource          | Source                | Bind                          |
|-------------------|-----------------------|-------------------------------|
| `pointVb_`        | `pointVertices`       | VB (12 B stride) + SRV `t1`   |
| `pointAo6Sb_`     | `pointAo6`            | StructuredBuffer SRV `t2`     |
| `subs_`           | `subs` (→ `GpuSubMesh`) | CPU-side cull / batching    |

Per-frame the renderer issues one `Draw` per contiguous chunk span × LOD, using `SubMesh.pointFirstLN` + `pointCountLN` to pick the right slice of `pointVb_`. AO6 is sampled in VS via `SV_VertexID + gVoxelBase`.

---

## Versioning

`kAssetVersion` (in `src/asset_version.h`) gates BOTH formats. Bump it whenever:
- `DiskVoxel` layout changes
- `Vertex` / `SubMesh` layout changes
- `VxbHeader` layout changes
- LOD aggregation rules change (downstream visual mismatch)

On version mismatch the loader rejects the file. `.vox` mismatch is fatal (no source to regen from since `voxelize` is gone). `.vxb` mismatch falls back to `.vox` + auto-rebake.

---

## On-demand compression analysis

Not stored on disk. `RunCompressionAnalysis(voxPath, scene, err)` re-reads `.vox` raw bytes and computes:

- Per-voxel bit-packed pos / visMask / AO / palette-index streams
- Exact Huffman code length over color palette indices
- Sub-cluster pos packing (auto-picks S ∈ {4, 8, 16, ...} ≤ chunkDim/4)
- LZ4 per chunk per stream

Results live in `Scene::comp*` fields, displayed by the Stats window. Triggered by the "Run analysis" button — never auto-run.
