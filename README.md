# voxeltest

A Win32 + DirectX 11 playground for experimenting with techniques to render
large voxel models. The reference dataset is the Mineways extract of Morgan
McGuire's *Rungholt* (~1.6M voxels) but anything Mineways exports as an OBJ
will work.

The point of the project is to compare rendering strategies side-by-side at
runtime via an ImGui panel.

## Techniques implemented

| Technique      | Idea                                                                 |
|----------------|----------------------------------------------------------------------|
| `PolygonBased` | Classic per-face triangles, vis-mask culled at bake time.            |
| `Points`       | One `POINT_LIST` vertex per voxel; lit via faked normal.             |
| `Hybrid`       | Polygons up close, points at distance (per-chunk pixels-per-voxel).  |
| `HexSprite`    | Per voxel: closest corner + 6 silhouette corners as a triangle fan.  |
| `PointCS`      | Compute shader software-rasterises points into a UAV.                |
| `PolyVID`      | One voxel = 1 instance, cube expanded via `SV_VertexID` (no VB).     |
| `Billboard`    | One quad per voxel, PS ray-vs-cube intersect + shade.                |
| `BillboardTri` | Same as Billboard but a single bounding triangle (3 verts/voxel).    |
| `Hybrid2`      | PolyVID up close, BillboardTri at distance.                          |
| `MergedMesh`   | Offline greedy-meshed single VB + IB drawn in one call.              |
| `Splat`        | Points → R8G8B8A8 RT (alpha = LOD), then CS expanding-ring filter.   |

Other UI knobs: shading mode (lit / flat / normals), point lighting (simple /
complex visMask-weighted), LOD selection (L0/L1/L2/Auto + distance scale),
grid replication (1–10×, replicates scene for stress testing), MSAA (off/2x/
4x/8x), Z prepass, chunk-bounds overlay, voxel-data source (original / cone-
& flood-culled).

## Dependencies (vendored)

* [`ocornut/imgui`](https://github.com/ocornut/imgui) — `docking` branch, in
  `third_party/imgui/`.
* [`redorav/hlslpp`](https://github.com/redorav/hlslpp) — in
  `third_party/hlslpp/`.

Both shipped as plain files in this repo; no submodule init required.

## Getting set up

1. **Get the source dataset.** Download Morgan McGuire's rungholt archive
   ([Computer Graphics Archive](https://casual-effects.com/data/)),
   unzip into `assets/rungholt/` so the structure looks like:

   ```
   assets/rungholt/rungholt.obj
   assets/rungholt/rungholt.mtl
   assets/rungholt/rungholt-RGB.png
   ...
   ```

2. **Build.** Open `voxeltest.sln` in Visual Studio 2022, set configuration
   to `Release | x64`, build. Two exes drop into `build/Release/`:
   * `voxelize.exe` — offline tool.
   * `voxeltest.exe` — the viewer.

3. **Bake the voxel data once.** From a shell at the repo root:

   ```
   build\Release\voxelize.exe
   ```

   This reads `assets/rungholt/rungholt.obj` and writes:
   * `assets/rungholt.vox` — chunked voxel data (VXL3).
   * `assets/rungholt_culled.vox` — same, with flood-fill interior cull.
   * `assets/rungholt_merged.msh` — greedy-meshed single VB+IB.

4. **Run the viewer.**

   ```
   build\Release\voxeltest.exe
   ```

   Or just hit F5 in Visual Studio.

## Controls

| Input                           | Action                                  |
|--------------------------------|------------------------------------------|
| `WASD`                          | Move                                    |
| `Q` / `E` or `Ctrl` / `Space`   | Down / Up                               |
| RMB + drag                      | Look                                    |
| Mouse wheel                     | Adjust move speed (log scale)           |
| `Esc`                           | Quit                                    |

Everything else is in the ImGui panel.

## File format notes (offline)

* **`.vox` (VXL3)**: header + chunk metadata (16 B each) + flat voxel blob.
  Per voxel: `uint8 x,y,z + uint8 visMask + uint32 color` (8 B). Chunk dim
  = 64. See `tools/voxelize.cpp` and `src/vox_loader.cpp`.
* **`.msh` (MSH1)**: header + flat vertex array + uint32 index buffer.
  Vertex: `float3 pos + uint32 color` (16 B). See `src/mesh_loader.cpp`.

## Layout

```
src/                  app + renderer + loaders
tools/voxelize.cpp    OBJ → .vox / .msh baker
shaders/voxel.hlsl    every technique (VS/PS/CS entry points)
third_party/          vendored libs (imgui, hlslpp)
assets/rungholt/      (you provide — Mineways extract)
```

## License

Code in this repository is provided as-is for experimentation. Vendored
libraries retain their own licenses (see their respective folders).
The rungholt dataset is © Morgan McGuire / Mineways and is NOT included.
