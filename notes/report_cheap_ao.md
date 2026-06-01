# Cheap AO — top-down depth map + HBAO-style filter

## Idea

- Render world from above, ortho. Each texel = 1 voxel in X-Z. Pixel value = max-Y of any voxel.
- Apply HBAO-style horizon filter on that depth map → second channel = AO contribution per texel.
- Output: 2-channel world-space map. ch0 = max-Y, ch1 = HBAO filter result.
- At shading time: sample the two values, AO = f(d0, d1).

## How it fits the renderer

- Already rendering top-down for sun shadows (`gShadowMapSize`, `gSunViewProj`). Could share the pass.
- A separate ortho pass dedicated to AO would render only opaque-voxel heights — single channel R16/R32 depth.
- Second pass: compute shader, runs HBAO filter, writes R8 AO.
- Bound at PS (resolve PS / pass2 colour CS) as a sampled SRV. Sample via worldX/worldZ → UV. Compare worldY to map value to decide AO factor.

## Pros

- 1-pixel-per-voxel resolution = perfect alignment at LOD0. Sample point-perfect.
- World-space → temporally stable. No view-dependent flicker like SSAO.
- Cheap shading-time cost: 1 SRV fetch + cheap math.
- Resolution scales with world size, not screen. Constant per-frame regardless of camera distance.

## Cons / unknowns

- **Memory**: kingslanding25 world ~512×?×512 voxels. R16 depth + R8 AO = 3B per X-Z column. 512² × 3B ≈ 0.75 MB. Acceptable. Larger worlds (4096²) = 48 MB — still OK.
- **Overhangs**: top-down map captures only the highest Y per (X,Z). Cave/tunnel surfaces below the surface get no AO contribution from above. Needs either:
  - Multiple Y "layers" (sliced top-down maps), or
  - Accept top-down-only AO (matches Minecraft-style overworld).
- **Update cost when streaming**: when chunks load/unload, the affected region of the map must be rebuilt. Streaming bookkeeping needed.
- **LOD interaction**: map is at world resolution (LOD0 voxel size). When rendering at LOD4 voxels (16× larger), 16² LOD0 texels collapse to one rendered voxel — could downsample/min-pool the AO channel for LOD-consistent shading.

## Alternative: 3D texture

- Sample a precomputed 3D volume of AO values at world position.
- Pro: handles overhangs / caves. Equal sampling regardless of view.
- Con: memory is N³ instead of N² — for 512³ at R8 = 128 MB. For 256³ = 16 MB. Tight but feasible. For larger worlds memory blows up fast.
- Con: precompute is heavy. Streaming-friendly volume updates need 3D chunk loader.
- Best for moderate-size worlds; less ideal for streamable infinite worlds.

## Recommendation

Start with top-down (cheaper, ships sooner). If overhangs matter visually for the kingslanding-style world, layered top-down (2-3 slices) covers most cases at ~3× the memory. 3D texture only if scenes need true volumetric AO and memory budget allows.

## Open questions

- Do we share the sun-shadow ortho viewport, or render a tighter map around camera that follows the player?
- Filter radius in HBAO step — fixed world-units or scaled by LOD?
- Where in shading pipeline does AO multiply? Pass2 colour CS or final post?
