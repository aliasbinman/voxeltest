// voxel.hlsl — shared post-process / dilate / TAA / sky shaders.
// LW-specific point/cube VS+PS live in lodworld.hlsl. Legacy Scene/Vertex
// paths (Points, PolyVID, PolyAxis, Billboard, HexSprite, PointCS) were
// archived; see code_Archive.txt for restoration via git.

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
    float2 ndc;
    ndc.x = ((float)pix.x + 0.5) / (float)W * 2.0 - 1.0;
    ndc.y = 1.0 - ((float)pix.y + 0.5) / (float)H * 2.0;
    float aspect = (float)W / (float)H;
    float3 viewDir = float3(ndc.x * aspect * gTanHalfFovY,
                            ndc.y           * gTanHalfFovY,
                            1.0);
    return normalize(gCamRight   * viewDir.x +
                     gCamUp      * viewDir.y +
                     gCamForward * viewDir.z);
}

// Reconstruct neighbour pixel's world pos from its depth + jittered NDC.
float3 ReconstructNeighborWorld(int2 sp, float zN, int W, int H)
{
    float viewZ = gNearZ / zN;
    float aspect = (float)W / (float)H;
    float ndcX = ((float)sp.x + 0.5) / (float)W * 2.0 - 1.0 - gJitter.x;
    float ndcY = 1.0 - ((float)sp.y + 0.5) / (float)H * 2.0 - gJitter.y;
    float viewX = ndcX * aspect * gTanHalfFovY * viewZ;
    float viewY = ndcY *          gTanHalfFovY * viewZ;
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
        float3 outRgb = ShadeWithLighting(bestAlbedo, bestN, hit, bestAo,
                                          bestLodIdx, bestParity, (int)gMode);
        gSplatFinalUav[pix] = float4(outRgb, 1.0);
        // Reproject winning hit -> clip depth (more accurate than the
        // un-dilated source depth, which is 0 at filled-in pixels).
        float4 clipHit = mul(float4(hit, 1.0), gViewProj);
        gSplatFinalDepthUav[pix] = saturate(clipHit.z / max(clipHit.w, 1e-6));
    } else {
        // No AABB hit. Keep depth from nearest neighbour (if any) so the post
        // pass has continuous Z; signal bg via alpha=0.
        gSplatFinalUav[pix]      = float4(0, 0, 0, 0);
        gSplatFinalDepthUav[pix] = fbHave ? fbZ : 0.0;
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

// ---------------- Fullscreen triangle for blit-style passes ----------------
struct VBlitOut { float4 pos : SV_Position; };
VBlitOut vsmain_blit(uint vid : SV_VertexID)
{
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VBlitOut o;
    o.pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return o;
}

// ---------------- Composite splat output over scene RT ----------------
Texture2D<float4> gSplatComposite      : register(t8);
Texture2D<float>  gSplatCompositeDepth : register(t7);
struct CompositeOut { float4 color : SV_Target; float depth : SV_Depth; };
CompositeOut psmain_splat_composite(VBlitOut i)
{
    CompositeOut o;
    int2 pix = int2(i.pos.xy);
    float4 c = gSplatComposite.Load(int3(pix, 0));
    float  z = gSplatCompositeDepth.Load(int3(pix, 0));
    o.color = c;
    if (z <= 0.0) { o.depth = 0.0; discard; }
    o.depth = z;
    return o;
}

// ---------------- TAA ----------------
Texture2D<float4> gTaaScene  : register(t4);
Texture2D<float4> gTaaHist   : register(t5);
Texture2D<float>  gTaaDepth  : register(t6);
SamplerState      gTaaSamp   : register(s0);

struct VTaaOut { float4 pos : SV_Position; float2 uv : UV; };

VTaaOut vsmain_taa(uint vid : SV_VertexID)
{
    float2 p = float2((vid == 1u) ? 3.0 : -1.0,
                      (vid == 2u) ? 3.0 : -1.0);
    VTaaOut o;
    o.pos = float4(p, 0.0, 1.0);
    o.uv  = float2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return o;
}

float4 psmain_taa(VTaaOut i) : SV_Target
{
    int2 pix = (int2)i.pos.xy;
    int W = (int)gScreenSize.x;
    int H = (int)gScreenSize.y;
    // Resample current frame at +jitter UV: undoes projection jitter so the
    // output pixel represents the world point at the un-jittered pixel centre.
    float2 currUv = (float2(pix) + 0.5) / float2((float)W, (float)H);
    currUv += float2(gJitter.x * 0.5, -gJitter.y * 0.5);
    float4 curSample = gTaaScene.SampleLevel(gTaaSamp, currUv, 0);
    float3 curC = curSample.rgb;
    float outAlpha = curSample.a;
    int2 jPix = clamp(int2(currUv * float2((float)W, (float)H)),
                      int2(0, 0), int2(W - 1, H - 1));
    float  d  = gTaaDepth.Load(int3(jPix, 0));
    if (d <= 0.0) {
        return float4(curC, outAlpha);
    }

    float viewZ = gNearZ / d;
    float aspect = (float)W / (float)H;
    float ndcX = currUv.x * 2.0 - 1.0;
    float ndcY = 1.0 - currUv.y * 2.0;
    float viewX = ndcX * aspect * gTanHalfFovY * viewZ;
    float viewY = ndcY *          gTanHalfFovY * viewZ;
    float3 world = gCamPos + gCamRight * viewX + gCamUp * viewY + gCamForward * viewZ;

    float4 prevClip = mul(float4(world, 1.0), gPrevViewProj);
    if (prevClip.w <= 0.0) return float4(curC, outAlpha);
    float3 prevNdc = prevClip.xyz / prevClip.w;
    if (any(abs(prevNdc.xy) > 1.0)) return float4(curC, outAlpha);
    float2 prevUv = float2(prevNdc.x * 0.5 + 0.5, 0.5 - prevNdc.y * 0.5);

    // Catmull-Rom 5-tap (Karis) — bicubic via 5 bilinear fetches.
    float2 texSize = float2((float)W, (float)H);
    float2 sp = prevUv * texSize;
    float2 tp1 = floor(sp - 0.5) + 0.5;
    float2 f = sp - tp1;
    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);
    float2 w12 = w1 + w2;
    float2 off12 = w2 / max(w12, 1e-5);
    float2 tp0  = (tp1 - 1.0) / texSize;
    float2 tp3  = (tp1 + 2.0) / texSize;
    float2 tp12 = (tp1 + off12) / texSize;

    float k0 = w12.x * w0.y;
    float k1 = w0.x  * w12.y;
    float k2 = w12.x * w12.y;
    float k3 = w3.x  * w12.y;
    float k4 = w12.x * w3.y;
    float kSum = max(k0 + k1 + k2 + k3 + k4, 1e-5);
    float3 prevC = float3(0, 0, 0);
    prevC += gTaaHist.SampleLevel(gTaaSamp, float2(tp12.x, tp0.y ), 0).rgb * k0;
    prevC += gTaaHist.SampleLevel(gTaaSamp, float2(tp0.x,  tp12.y), 0).rgb * k1;
    prevC += gTaaHist.SampleLevel(gTaaSamp, float2(tp12.x, tp12.y), 0).rgb * k2;
    prevC += gTaaHist.SampleLevel(gTaaSamp, float2(tp3.x,  tp12.y), 0).rgb * k3;
    prevC += gTaaHist.SampleLevel(gTaaSamp, float2(tp12.x, tp3.y ), 0).rgb * k4;
    prevC /= kSum;

    // 3x3 neighbourhood color clamp: kills ghosting on disocclusion / edges.
    float2 px = 1.0 / float2((float)W, (float)H);
    float3 nMin = curC;
    float3 nMax = curC;
    float3 nMean = curC;
    float3 nMean2 = curC * curC;
    [unroll] for (int dy = -1; dy <= 1; ++dy)
    [unroll] for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) continue;
        float3 s = gTaaScene.SampleLevel(gTaaSamp, currUv + float2(dx, dy) * px, 0).rgb;
        nMin = min(nMin, s);
        nMax = max(nMax, s);
        nMean  += s;
        nMean2 += s * s;
    }
    nMean  /= 9.0;
    nMean2 /= 9.0;
    float3 sigma = sqrt(max(nMean2 - nMean * nMean, 0.0));
    const float gamma = 1.25;
    float3 vMin = max(nMin, nMean - gamma * sigma);
    float3 vMax = min(nMax, nMean + gamma * sigma);
    float3 clipped = clamp(prevC, vMin, vMax);

    float dist = length(clipped - prevC) / max(length(nMax - nMin), 1e-4);
    float alpha = lerp(0.1, 0.5, saturate(dist));
    float3 outC = lerp(clipped, curC, alpha);
    return float4(outC, outAlpha);
}

// ---------------- Post pass: sky for empty pixels + unsharp ----------------
Texture2D<float>  gPostDepth : register(t6);
Texture2D<float4> gPostIn    : register(t7);

// ---------------- God rays ----------------
// 64x64 occlusion texture centered on sun in NDC. Mark pass writes 1.0 if
// the corresponding screen pixel is sky (reverse-Z depth == 0), 0 if blocked
// or off-screen. Radial blur smears toward center. Post pass samples by
// computing offset from sun screen pos and adds yellow tint.
cbuffer cbGodray : register(b3)
{
    float2 gSunScreenNdc;        // -1..1 NDC sun position (xy)
    float  gSunOnScreen;         // 0 = sun behind cam or NDC outside; 1 = on screen
    float  gGodrayEmaAlpha;      // 0..1: how much of the new frame to mix in (1 = no smoothing)
    float2 gGodrayHalfScreenUV;  // half-extent in screen-UV space (square in pixels)
    float2 _padG1;
    float3 gGodrayTint;
    float  gGodrayStrength;      // 0 = off
};
Texture2D<float> gGodrayTex     : register(t9);   // mark (in blur) / blur+EMA (in post)
Texture2D<float> gGodrayHistTex : register(t10);  // previous frame's blur+EMA (blur input only)

float4 psmain_godray_mark(VTaaOut i) : SV_Target
{
    if (gSunOnScreen < 0.5) return 0.0;
    int2 tex = (int2)i.pos.xy;
    float2 local = (float2(tex) + 0.5) / 64.0 * 2.0 - 1.0;  // -1..1 in tex space
    // Circle mask.
    if (dot(local, local) > 1.0) return 0.0;
    // Map to screen UV: square is gGodrayHalfScreenUV (x,y) around sun.
    float2 sunScreenUV = gSunScreenNdc * float2(0.5, -0.5) + 0.5;
    float2 uv = sunScreenUV + local * gGodrayHalfScreenUV;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 0.0;
    int2 px = int2(uv * gScreenSize);
    px = clamp(px, int2(0,0), int2((int)gScreenSize.x - 1, (int)gScreenSize.y - 1));
    float d = gPostDepth.Load(int3(px, 0));   // reverse-Z: 0 = far / sky
    return (d <= 0.0001) ? 1.0 : 0.0;
}

float4 psmain_godray_blur(VTaaOut i) : SV_Target
{
    int2 tex = (int2)i.pos.xy;
    float2 uv = (float2(tex) + 0.5) / 64.0;
    float2 toCenter = float2(0.5, 0.5) - uv;
    const int N = 24;
    float sum = 0.0;
    [unroll] for (int s = 0; s < N; ++s) {
        float t = (float)s / (float)N;
        sum += gGodrayTex.SampleLevel(gTaaSamp, uv + toCenter * t, 0);
    }
    float curr = sum / (float)N;
    float prev = gGodrayHistTex.SampleLevel(gTaaSamp, uv, 0);
    return lerp(prev, curr, gGodrayEmaAlpha);
}

float3 PostPixelWorldDir(int2 pix, int W, int H)
{
    float ndcX = ((float)pix.x + 0.5) / (float)W * 2.0 - 1.0;
    float ndcY = 1.0 - ((float)pix.y + 0.5) / (float)H * 2.0;
    float aspect = (float)W / (float)H;
    float3 v = float3(ndcX * aspect * gTanHalfFovY, ndcY * gTanHalfFovY, 1.0);
    return normalize(gCamRight * v.x + gCamUp * v.y + gCamForward * v.z);
}

float3 SkyColor(float3 rd)
{
    // Gradient only — sun disc + glow are added by the godray post pass.
    float  t = saturate(rd.y * 0.5 + 0.5);
    float3 horizon = float3(0.70, 0.75, 0.85);
    float3 zenith  = float3(0.20, 0.40, 0.80);
    float3 ground  = gFogColor;
    float3 above   = lerp(horizon, zenith, smoothstep(0.5, 1.0, t));
    return lerp(ground, above, smoothstep(0.48, 0.52, t));
}

float4 psmain_post(VTaaOut i) : SV_Target
{
    int2 pix = (int2)i.pos.xy;
    int W = (int)gScreenSize.x;
    int H = (int)gScreenSize.y;
    // Sky: scene RT cleared with alpha=0; scene PS writes alpha=1.
    float4 inC = gPostIn.Load(int3(pix, 0));
    float3 c;
    if (inC.a < 0.5) {
        float3 rd = PostPixelWorldDir(pix, W, H);
        c = SkyColor(rd);
    } else {
        c = inC.rgb;
        int2 pL = int2(max(pix.x - 1, 0),     pix.y);
        int2 pR = int2(min(pix.x + 1, W - 1), pix.y);
        int2 pU = int2(pix.x, max(pix.y - 1, 0));
        int2 pD = int2(pix.x, min(pix.y + 1, H - 1));
        float3 avg = (gPostIn.Load(int3(pL, 0)).rgb + gPostIn.Load(int3(pR, 0)).rgb +
                      gPostIn.Load(int3(pU, 0)).rgb + gPostIn.Load(int3(pD, 0)).rgb) * 0.25;
        c = c + 0.5 * (c - avg);   // unsharp mask
    }
    // Add godrays. Every pixel samples the blurred occlusion texture at its
    // sun-relative offset (in screen-UV space), clamped to the unit circle
    // so there's no hard rectangle edge. Fades with screen distance from sun.
    if (gSunOnScreen > 0.5 && gGodrayStrength > 0.0) {
        float2 sunScreenUV = gSunScreenNdc * float2(0.5, -0.5) + 0.5;
        float2 screenUV    = (float2(pix) + 0.5) / float2(W, H);
        float2 off         = (screenUV - sunScreenUV) / gGodrayHalfScreenUV;   // -1..1 = inside square
        float  rOff        = length(off);
        if (rOff > 1.0) off *= (1.0 / rOff);
        float2 godrayUV    = 0.5 + off * 0.5;
        float  gr          = saturate(gGodrayTex.SampleLevel(gTaaSamp, godrayUV, 0));
        float  aspect      = gScreenSize.x / gScreenSize.y;
        float2 dScreen     = (screenUV - sunScreenUV) * float2(aspect, 1.0);
        float  dist        = length(dScreen);
        float  fade        = saturate(1.0 - dist * 1.5);
        c += gGodrayTint * gr * gGodrayStrength * fade;
    }
    return float4(c, 1.0);
}
