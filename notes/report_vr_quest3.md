# VR on Quest 3

## Hardware reality

- Quest 3 = Snapdragon XR2 Gen 2, Adreno 740 GPU, ~2k×2k per eye target, 90/120 Hz.
- Standalone (no PC) means **mobile** GPU power budget. Roughly 1/10 of a desktop 4070-class GPU at sustained loads.
- Vulkan-only (no D3D11, no D3D12).
- Multiview rendering for stereo (single draw, both eyes in array texture) — required for sensible perf.

## What works in your favour

- The atomic-min CS rasterizer is **mobile-friendly**: no fixed-function rasterizer pipeline pressure, no triangle setup, no MSAA cost.
- Block CS / Point CS use one atomic per voxel, modest ALU. Adreno can do these.
- LOD streaming already cuts work for far geometry — essential on mobile.

## What hurts

- **Per-eye render at 2k×2k = 4× pixels vs your current 1080p window**. Voxel count visible per frame doubles (left eye + right eye) — multiview helps share VS but compute is duplicated.
- **120 Hz target = 8.3 ms frame budget**. You're currently at 0.4 ms for pass1 alone on a 5070 desktop. On Adreno 740 expect roughly 5–10× slowdown for the same workload. Per-pass headroom is thin.
- **Tile-based deferred GPU**: Adreno is TBDR-like. UAV atomics work but defeat tiling benefits. Performance characterisation needs hands-on testing.
- **Memory bandwidth**: GDDR5X on Quest 3 = ~70 GB/s vs your 5070's ~672 GB/s. Atomic-heavy workloads suffer.
- **Heat / sustained perf**: laptop GPU has a fan. Quest 3 throttles after 10–15 minutes of intense load. Plan for it.

## Port plan (assumes Vulkan port already done)

1. **OpenXR**: use Khronos's standard. Replaces window + swapchain on flat Vulkan.
2. **Multiview**: enable `VK_KHR_multiview`. Wrap your viewProj matrix into a `gViewProj[2]` array, shader uses `gl_ViewIndex`. Doubles vertex/fragment work without doubling CPU draw calls.
3. **Foveated rendering**: Quest 3 supports `VK_EXT_fragment_density_map`. Render periphery at lower res — for your rasterizer that means smaller pass1 UAV at outer regions. Real perf win.
4. **Aggressive LOD**: shrink LOD distance thresholds. Per-eye res is high but most pixels are far. Tune for 1 pixel per voxel at L=1 (not L=0) by default.
5. **Asset budget**: Quest 3 has 8 GB RAM but apps capped to ~4 GB. Stream less, evict more.
6. **Input**: hand-tracking + controller via OpenXR action sets. Replace mouse-look entirely.
7. **Locomotion**: gameplay/UX problem — teleport, smooth, snap-turn options.

## Estimate

- **Best case** (Vulkan port done first): 2–3 weeks for OpenXR + multiview + a usable Quest build. Perf tuning beyond that.
- **From current D3D11 state**: Vulkan port (4–6 weeks) + above (2–3 weeks) = ~8 weeks before you put on the headset.

## Open questions

- Is the goal "walk through the kingslanding scene in VR" (room-scale exploration) or "build/edit voxels in VR"? Second is much harder.
- Single-player only, or multi-user shared world?
- Target visual quality: smooth (less detail) or detailed (lower frame rate)? Quest 3 users are conditioned to expect smooth.

## Recommendation

Vulkan port unlocks both macOS + Android + Quest. Don't VR-port a Windows-only D3D11 codebase — go Vulkan first, OpenXR second.

Quest 3 is a serious GPU but not desktop-class. Your existing aggressive LOD + streaming is the right foundation. Expect to spend significant time on per-pass GPU optimisation (your worklist work was good prep) before VR is comfortable.
