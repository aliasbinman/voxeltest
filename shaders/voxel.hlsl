// voxel.hlsl — compute passes (shadow blur + splat dilate). All fullscreen
// VS/PS passes (TAA, post, godrays, splat composite) live in postfx.hlsl.
// LW point/cube draws live in lodworld.hlsl.

cbuffer cbPerFrame : register(b0)
{
    row_major float4x4 gViewProj;
    float3   gCamPos;
    float    gMode;
    float3   gLightDir;
    float    gAmbient;
    float3   gPointNormal;
    float    _pad0;
    row_major float4x4 gInvViewProj;
    float2   gScreenSize;
    float2   _pad1;          // .x = splat dilate radius
    float3   gCamRight;
    float    _pad3;
    float3   gCamUp;
    float    _pad4;
    float3   gCamForward;
    float    gTanHalfFovY;
    float3   gFogColor;
    float    gFogDensity;
    float    gHeightFogDensity;
    float    gHeightFogFalloff;
    float    gHeightFogStart;
    float    _padHF;
    float3   gSceneOrigin;
    float    gNearZ;
    float3   gSceneSpan;
    float    _pad6;
    row_major float4x4 gPrevViewProj;
    float2   gJitter;
    float2   _pad7;
    row_major float4x4 gSunViewProj;
    float    gShadowBias;
    float    gShadowMapSize;
    float    gShadowEnable;
    float    gSunIntensity;
    float    gExposure;
    float    gRoughness;
    float    gColorizeClusters;
    float    gGridSize;
    // ---- CPU-precomputed derived values to keep shaders free of redundant math ----
    float2   gInvScreenSize;   // 1/W, 1/H
    float    gAspect;          // W/H
    float    gInvAspect;       // H/W
    float    gAspectTanFov;    // gAspect * gTanHalfFovY
    float3   _padPC;
};

// Shared lighting (ApplyFog, SampleShadow, SampleAmbientCubeTriplanar,
// ClusterTint, Tonemap, ShadeWithLighting). Declares gShadowTex(t6) +
// gShadowSamp(s1). Included AFTER cbPerFrame so helpers see the uniforms.
#include "shading.hlsli"

// ---------------- Shadow blur (fill empty shadow texels) ----------------
// Sparse point-splat casters leave many shadow texels at depth 0. This CS
// replaces empty texels with the avg of non-empty 8-neighbours.
Texture2D<float>     gShadowSrcTex : register(t7);
RWTexture2D<float>   gShadowDstUav : register(u3);
[numthreads(8, 8, 1)]
void csmain_shadow_blur(uint3 dt : SV_DispatchThreadID)
{
    int W = (int)gShadowMapSize;
    int H = (int)gShadowMapSize;
    int2 pix = (int2)dt.xy;
    if (pix.x >= W || pix.y >= H) return;

    float self = gShadowSrcTex.Load(int3(pix, 0));
    if (self > 0.0001) {
        gShadowDstUav[pix] = self;
        return;
    }
    float sum = 0.0;
    int   cnt = 0;
    [unroll] for (int dy = -1; dy <= 1; ++dy) {
        [unroll] for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int2 sp = pix + int2(dx, dy);
            if (sp.x < 0 || sp.x >= W || sp.y < 0 || sp.y >= H) continue;
            float v = gShadowSrcTex.Load(int3(sp, 0));
            if (v > 0.0001) { sum += v; ++cnt; }
        }
    }
    gShadowDstUav[pix] = (cnt > 0) ? (sum / (float)cnt) : 0.0;
}

// ---------------- Splat pipeline ----------------
// Splat RT alpha encoding (written by lodworld.hlsl psmain_lw_splat_albedo):
//   bit 7    = marker (1 = splat pixel)
//   bits 6-5 = lodIdx (0..3)
//   bits 4-1 = AO 4-bit (fallback avg; per-face AO lives in mask R32)
//   bit 0    = cluster checker parity
// Splat mask RT (R32_UINT) layout:
//   bits 0-5  = visMask (6 face visibility)
//   bits 6-29 = 6 face AOs × 4 bits (face f at bit 6+f*4)
Texture2D<float>    gSplatDepth    : register(t1);
Texture2D<float4>   gSplatColorSrv : register(t2);
Texture2D<uint>     gSplatMaskSrv  : register(t3);
RWTexture2D<float4> gSplatFinalUav      : register(u1);
RWTexture2D<float>  gSplatFinalDepthUav : register(u2);

// World-space ray dir for pixel center (camera basis; avoids invVP).
float3 PixelWorldDir(int2 pix, int W, int H)
{
    float2 uv = (float2(pix) + 0.5) * gInvScreenSize;
    float ndcX = uv.x * 2.0 - 1.0;
    float ndcY = 1.0 - uv.y * 2.0;
    float3 viewDir = float3(ndcX * gAspectTanFov,
                            ndcY * gTanHalfFovY,
                            1.0);
    return normalize(gCamRight   * viewDir.x +
                     gCamUp      * viewDir.y +
                     gCamForward * viewDir.z);
}

// Reconstruct neighbour pixel's world pos from its depth + jittered NDC.
float3 ReconstructNeighborWorld(int2 sp, float zN, int W, int H)
{
    float viewZ = gNearZ / zN;
    float2 uv = (float2(sp) + 0.5) * gInvScreenSize;
    float ndcX = uv.x * 2.0 - 1.0 - gJitter.x;
    float ndcY = 1.0 - uv.y * 2.0 - gJitter.y;
    float viewX = ndcX * gAspectTanFov * viewZ;
    float viewY = ndcY * gTanHalfFovY  * viewZ;
    return gCamPos + gCamRight * viewX + gCamUp * viewY + gCamForward * viewZ;
}

// Pass 1: per-pixel splat dilation. Search nbour kernel for marker pixels,
// ray-test against each marker's voxel AABB, pick nearest hit. Light using
// per-face AO from the picked face (decoded from mask R32 channel).
[numthreads(8, 8, 1)]
void csmain_splat(uint3 dt : SV_DispatchThreadID)
{
    int2 pix = (int2)dt.xy;
    int W = (int)gScreenSize.x;
    int H = (int)gScreenSize.y;
    if (pix.x >= W || pix.y >= H) return;

    float3 ro = gCamPos;
    float3 rd = PixelWorldDir(pix, W, H);
    float3 invRd = 1.0 / rd;

    int R = (int)max(1.0, _pad1.x);
    float  bestT      = 1e30;
    float3 bestAlbedo = float3(0, 0, 0);
    float3 bestN      = float3(0, 0, 1);
    uint   bestMask   = 0x3Fu;
    bool   anyHit     = false;

    float bestAo = 1.0;
    uint  bestLodIdx = 0u;
    uint  bestParity = 0u;
    // Depth-only fallback: nearest-Z valid neighbour so missed-AABB pixels
    // still get continuous depth for TAA / sky / DOF.
    bool  fbHave = false;
    float fbZ    = -1.0;
    [loop] for (int dy = -R; dy <= R; ++dy) {
        [loop] for (int dx = -R; dx <= R; ++dx) {
            int2 sp = pix + int2(dx, dy);
            if (sp.x < 0 || sp.x >= W || sp.y < 0 || sp.y >= H) continue;
            float4 s = gSplatColorSrv.Load(int3(sp, 0));
            uint a8 = (uint)(s.a * 255.0 + 0.5);
            if (a8 == 0u) continue;
            if ((a8 & 0x80u) == 0u) continue;
            uint lodIdx = (a8 >> 4) & 7u;       // 3 bits, supports LOD0..7
            uint ao4    = a8 & 0xFu;
            float aoN   = (float)ao4 / 15.0;
            // parity now lives in mask channel bit 30 (decoded below after mask Load).
            uint parityN = 0u;
            // halfExt = 0.5 * (1 << lodIdx) — LOD voxel half-extent in world units.
            float halfExt = 0.5 * (float)(1u << lodIdx);
            float zN = gSplatDepth.Load(int3(sp, 0));
            if (zN <= 0.0) continue;
            if (zN > fbZ) { fbZ = zN; fbHave = true; }
            uint maskFull = gSplatMaskSrv.Load(int3(sp, 0));
            uint visMaskN = maskFull & 0x3Fu;
            parityN = (maskFull >> 30u) & 1u;   // cluster checker parity
            float3 wp = ReconstructNeighborWorld(sp, zN, W, H);
            // Snap to LOD-aligned voxel grid: collapses adjacent splats from
            // the same cluster onto the same AABB (kills cube-edge seams on
            // flat surfaces).
            float S = 2.0 * halfExt;
            float3 vmin = floor(wp / S) * S;
            float3 vmax = vmin + S;
            float3 t0v = (vmin - ro) * invRd;
            float3 t1v = (vmax - ro) * invRd;
            float3 tmn = min(t0v, t1v);
            float3 tmx = max(t0v, t1v);
            float tNear = max(max(tmn.x, tmn.y), tmn.z);
            float tFar  = min(min(tmx.x, tmx.y), tmx.z);
            if (tFar < 0.0 || tNear > tFar) continue;
            float tHit = max(tNear, 0.0);
            if (tHit < bestT) {
                float3 hit = ro + rd * tHit;
                float3 center = (vmin + vmax) * 0.5;
                float3 d = hit - center;
                // Pick face: dominant axis of (hit-center) restricted to faces
                // present in visMask. Decode that face's AO from packed mask.
                float bestProj = -1.0;
                float3 n = float3(0, 1, 0);
                float pickedFaceAo = aoN;
                [unroll] for (uint fi = 0u; fi < 6u; ++fi) {
                    if (((visMaskN >> fi) & 1u) == 0u) continue;
                    float3 fn = float3(0, 0, 0);
                    if      (fi == 0u) fn = float3( 1, 0, 0);
                    else if (fi == 1u) fn = float3(-1, 0, 0);
                    else if (fi == 2u) fn = float3( 0, 1, 0);
                    else if (fi == 3u) fn = float3( 0,-1, 0);
                    else if (fi == 4u) fn = float3( 0, 0, 1);
                    else               fn = float3( 0, 0,-1);
                    float p = dot(d, fn);
                    if (p > bestProj) {
                        bestProj = p; n = fn;
                        uint nib = (maskFull >> (6u + fi * 4u)) & 0xFu;
                        pickedFaceAo = (float)nib / 15.0;
                    }
                }
                bestT      = tHit;
                bestAlbedo = s.rgb;
                bestN      = n;
                bestMask   = visMaskN;
                bestAo     = pickedFaceAo;
                bestLodIdx = lodIdx;
                bestParity = parityN;
                anyHit     = true;
            }
        }
    }

    if (anyHit) {
        float3 hit = ro + rd * bestT;
        // Cheap top-down AO: push lookup along surface normal so vertical
        // faces sample the open neighbour column instead of their own roof.
        float aoTop = AoSampleWithNormal(hit, bestN);
        bestAlbedo *= aoTop;
        float3 outRgb;
        if ((int)gMode == 3) {
            // AO viz: top-down AO directly.
            outRgb = aoTop.xxx;
        } else if ((int)gMode == 4) {
            // AO + LOD viz: top-down AO tinted by per-LOD colour + cluster checker.
            float check = (bestParity == 0u) ? 0.55 : 1.00;
            outRgb = aoTop * ClusterTint(2u + bestLodIdx) * check;
        } else {
            outRgb = ShadeWithLighting(bestAlbedo, bestN, hit, bestAo,
                                       bestLodIdx, bestParity, (int)gMode);
        }
        gSplatFinalUav[pix] = float4(outRgb, 1.0);
        // Reproject winning hit -> clip depth (more accurate than the
        // un-dilated source depth, which is 0 at filled-in pixels).
        float4 clipHit = mul(float4(hit, 1.0), gViewProj);
        gSplatFinalDepthUav[pix] = saturate(clipHit.z / max(clipHit.w, 1e-6));
    } else {
        // No AABB hit. Match colour: leave depth at 0 (sky) so depth + colour
        // dilation regions stay in sync. (Previously this wrote a neighbour's
        // depth as a "continuous Z" fallback, but that depth-extends past the
        // colour-filled region.)
        gSplatFinalUav[pix]      = float4(0, 0, 0, 0);
        gSplatFinalDepthUav[pix] = 0.0;
    }
}

// Pass 2: fill holes left by pass 1. Scan small kernel for nearest-Z filled
// neighbour. Already-filled pixels pass through.
[numthreads(8, 8, 1)]
void csmain_splat_fill(uint3 dt : SV_DispatchThreadID)
{
    int2 pix = (int2)dt.xy;
    int W = (int)gScreenSize.x;
    int H = (int)gScreenSize.y;
    if (pix.x >= W || pix.y >= H) return;

    float4 c0 = gSplatColorSrv.Load(int3(pix, 0));
    float  z0 = gSplatDepth.Load(int3(pix, 0));
    if (c0.a > 0.5) {
        gSplatFinalUav[pix]      = c0;
        gSplatFinalDepthUav[pix] = z0;
        return;
    }

    int R = max(1, (int)_pad1.x);
    float bestZ = -1.0;
    float4 bestC = float4(0, 0, 0, 0);
    [loop] for (int dy = -R; dy <= R; ++dy) {
        [loop] for (int dx = -R; dx <= R; ++dx) {
            int2 sp = pix + int2(dx, dy);
            if (sp.x < 0 || sp.x >= W || sp.y < 0 || sp.y >= H) continue;
            float4 sc = gSplatColorSrv.Load(int3(sp, 0));
            if (sc.a <= 0.5) continue;
            float sz = gSplatDepth.Load(int3(sp, 0));
            if (sz > bestZ) { bestZ = sz; bestC = sc; }
        }
    }
    if (bestZ <= 0.0) {
        gSplatFinalUav[pix]      = float4(0, 0, 0, 0);
        gSplatFinalDepthUav[pix] = 0.0;
    } else {
        gSplatFinalUav[pix]      = bestC;
        gSplatFinalDepthUav[pix] = bestZ;
    }
}

