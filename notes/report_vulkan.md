# Vulkan port — macOS + Android target

## Why

- D3D11 locked to Windows. No macOS, no Android, no Linux, no consoles.
- macOS: Vulkan via MoltenVK (Vulkan-on-Metal). Works but adds a translation layer.
- Android: Vulkan native since API 24+.
- Bonus: Vulkan gives explicit control over barriers (the actual root cause of your atomic-dispatch perf issue today). The hazard-tracking pipeline drain you fought is invisible in Vulkan.

## What it touches in this codebase

| Subsystem | Effort |
|---|---|
| Renderer abstraction (currently raw ID3D11* everywhere) | **Major** — rip out every `ID3D11*` ComPtr, introduce a thin VkDevice/queue/cmd-buffer wrapper |
| Shader compile (HLSL → DXC → DXBC → DX) | Medium — DXC can target SPIR-V, your HLSL keeps mostly working. Some entry-point fixups (`register(tN)` → explicit set/binding) |
| Resource bindings (CB at b0..b3, SRV at t0..t7) | Major — Vulkan descriptor sets are unrecognisable next to D3D11 slots; needs descriptor set layout design |
| Swapchain / present | Major — VkSwapchain + queues + acquire/release semaphores |
| ImGui | Easy — `imgui_impl_vulkan` exists and is good |
| MicroProfile | Easy — has Vulkan backend |
| Window / input | Easy — already minimal Win32; on macOS use GLFW or SDL3 |
| Compute dispatch (worklist path) | Easy — Vulkan compute is straightforward |
| Asset format | None — `.lw` files stay byte-identical |

## Roadmap (rough)

1. **Abstract the renderer**: extract a `GpuDevice`/`GpuCommands` interface. Even before Vulkan, this lets you stub-test without D3D11.
2. **Get triangle on screen via Vulkan**: bring up swapchain + clear + one full-screen triangle.
3. **Port one compute shader** (pass1 worklist depth). Validate atomics work, perf matches D3D11.
4. **Port pass2 + resolve PS**. Two-pass pipeline running.
5. **Port splat/poly tech**. Renderable scene.
6. **Port shadow + post + godrays + TAA**. Feature-complete.
7. **macOS via MoltenVK** — should just work after that.
8. **Android port** — touch input replaces mouse/kb, screen-sized concerns.

Expect 4–6 weeks of focused work to reach feature parity with current Windows D3D11 build. macOS adds a week of MoltenVK fixups (shader compatibility, validation layer gotchas).

## Risks / unknowns

- **MoltenVK gaps**: no native R32 UAV atomics on older Mac GPUs (M1+ should be fine via Metal 3 atomics). Verify before committing.
- **Android GPU diversity**: Mali / Adreno / PowerVR all support compute + atomics, but driver bugs are real. Plan for fallbacks.
- **Quest 3 (Snapdragon XR2 Gen 2)**: Adreno 740, Vulkan-only. Same Android constraints + multiview rendering for stereo (see VR report).
- **Wave intrinsics**: SM6 wave ops in HLSL map to Vulkan subgroup ops via SPIR-V. Usable but syntax differs.

## Recommendation

Worth it if cross-platform is a real goal. Don't half-port — D3D11 + Vulkan double-renderer is a maintenance trap. Cut over fully once Vulkan reaches parity.

If only Quest 3 matters, going straight to Vulkan/Android without macOS is simpler and skips MoltenVK. macOS gives you a desktop testbed that's nicer than building APKs every iteration though.
