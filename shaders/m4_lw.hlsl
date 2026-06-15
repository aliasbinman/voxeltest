// Lighting mode toggle (hot-reloadable). Save file to swap modes.
//   1 = per-pixel hit-face: ray-AABB tells you the entry face; light by that
//       face's normal alone. Sharp per-pixel face shading.
//   0 = triplanar visMask-weighted: for each axis, light its visible face
//       (sign(vdir.axis)) gated by visMask, weighted by vdir.axis² (projected
//       cube area on screen). Sum of 3 faces. Smooth cube lighting.
#define USE_HIT_FACE 0

// M4a — minimum-viable PointCS_Block compute rasterizer for the LW path.
// Stripped port of the relevant pieces from lodworld.hlsl: no AO, no shadows,
// no fog, no LOD fade, no clusters tint, no LodViz mode. Just pass1 atomic
// linear-depth + pass2 color + resolve PS.
//
// Bindings:
//   b0  CBLwFrame  — gViewProj, gMode
//   b1  CBLwCS    — gVwSize, gLwPointCount, gLwNumWorkItems, gLwLodIdx
//   t0  ChunkInfo (struct)
//   t1  Palette   (uint)
//   t2  BlockPos  (uint)
//   t3  BlockCol  (uint2)
//   t4  WorkItems (uint4)
//   t5  DepthSrv  (Texture2D<uint>) — pass2 only
//   u0  VisUav    (RWTexture2D<uint>) — pass1 = depth, pass2 = ARGB color
//
// Dilate (post-pass2 hole-fill):
//   b0  CBLwFrame
//   b1  CBLwCS
//   t0  DepthSrv  (Texture2D<uint>)
//   t1  ColorIn   (Texture2D<uint>, raw visColor from pass2)
//   u0  ColorOut  (RWTexture2D<uint>, dilated visColor2)
//
// Resolve / Blit:
//   t0  DepthSrv (Texture2D<uint>)
//   t1  ColorSrv (Texture2D<uint>, post-dilate visColor2)

// Per-frame CB, LW structured buffers (t0-t7), LookupItem, ambient cube + fog —
// all shared with the octet techs.
#include "m4_common.hlsli"

RWTexture2D<uint>             gLwVisUav     : register(u0);
// Per-face AO (low 24 bits = 6 faces * 4-bit) written at the splat winner pixel.
// Dilate reads it and indexes by the true hit face for Lit shading.
RWTexture2D<uint>             gLwAoUav      : register(u1);

// Linear-depth encoding — uniform precision over [0, 100km] view-Z.
static const float kLinDepthFar   = 100000.0;
static const float kLinDepthMaxF  = 4294967294.0; // 0xFFFFFFFE
static const float kLinDepthScale = kLinDepthMaxF / kLinDepthFar;
uint EncodeLinDepth(float viewZ)
{
    return (uint)clamp(viewZ * kLinDepthScale, 0.0, kLinDepthMaxF);
}

// ============================================================
// Pass 1 — write linear depth via InterlockedMin.
// ============================================================
[numthreads(64, 1, 1)]
void csmain_pass1_depth(uint3 dt : SV_DispatchThreadID)
{
    uint gid = dt.x;
    if (gid >= gLwPointCount) 
        return;
    uint4 item = LookupItem(gid);
    uint slot        = item.x;
    uint blockBase   = item.y;
    uint count       = item.z;
    uint firstThread = item.w;
    if (gid - firstThread >= count) return;

    LwChunkInfo ci = gLwChunkInfos[slot];
    uint pack0 = gLwBlockPos[blockBase + (gid - firstThread)];

    uint bx  = (pack0 >>  0) & 0xFFu;
    uint by  = (pack0 >>  8) & 0xFFu;
    uint bz  = (pack0 >> 16) & 0xFFu;
    uint occ = (pack0 >> 24) & 0xFFu;

    [unroll]
    for (uint i = 0; i < 8; ++i)
    {
        if (((occ >> i) & 1u) == 0u) 
            continue;
        uint lx = (i >> 0) & 1u;
        uint ly = (i >> 1) & 1u;
        uint lz = (i >> 2) & 1u;
        float3 local = float3(
            (float)(bx * 2u + lx),
            (float)(by * 2u + ly),
            (float)(bz * 2u + lz)) + 0.5;
        float3 world = ci.worldOrigin + local * ci.lodScale;

        float4 clip = mul(float4(world, 1.0), gViewProj);
        if (clip.z <= 0.0) continue;
        float3 ndc = clip.xyz / clip.w;
        ndc.xy += gJitter;
        if (ndc.x < -1.0 || ndc.x > 1.0 ||
            ndc.y < -1.0 || ndc.y > 1.0 ||
            ndc.z <  0.0 || ndc.z > 1.0) continue;
        int2 pix;
        pix.x = (int)((ndc.x * 0.5 + 0.5) * (float)gVwSize.x);
        pix.y = (int)((-ndc.y * 0.5 + 0.5) * (float)gVwSize.y);
        if (pix.x < 0 || pix.x >= (int)gVwSize.x ||
            pix.y < 0 || pix.y >= (int)gVwSize.y) continue;

        uint linD = EncodeLinDepth(clip.w);
        // Single-pixel atomic-min. Coverage filling is handled by a dilate
        // compute pass (M4c — TBD port from CSTiles csSplat_), not in-shader.
        InterlockedMin(gLwVisUav[pix], linD);
    }
}

// ============================================================
// Pass 2 — write ARGB color where this voxel is the depth winner.
// ============================================================
[numthreads(64, 1, 1)]
void csmain_pass2_color(uint3 dt : SV_DispatchThreadID)
{
    uint gid = dt.x;
    if (gid >= gLwPointCount) 
        return;
    uint4 item = LookupItem(gid);
    uint slot        = item.x;
    uint blockBase   = item.y;
    uint count       = item.z;
    uint firstThread = item.w;
    if (gid - firstThread >= count) return;

    LwChunkInfo ci  = gLwChunkInfos[slot];
    uint   pack0    = gLwBlockPos[blockBase + (gid - firstThread)];
    uint2  cols     = gLwBlockCol[blockBase + (gid - firstThread)];
    uint2  visPack  = gLwBlockVis[blockBase + (gid - firstThread)];

    uint bx  = (pack0 >>  0) & 0xFFu;
    uint by  = (pack0 >>  8) & 0xFFu;
    uint bz  = (pack0 >> 16) & 0xFFu;
    uint occ = (pack0 >> 24) & 0xFFu;

    [unroll]
    for (uint i = 0; i < 8; ++i)
    {
        if (((occ >> i) & 1u) == 0u) continue;
        uint palIdx = (i < 4u) ? ((cols.x >> (i * 8u)) & 0xFFu)
                                : ((cols.y >> ((i - 4u) * 8u)) & 0xFFu);
        uint lx = (i >> 0) & 1u;
        uint ly = (i >> 1) & 1u;
        uint lz = (i >> 2) & 1u;
        float3 local = float3(
            (float)(bx * 2u + lx),
            (float)(by * 2u + ly),
            (float)(bz * 2u + lz)) + 0.5;
        float3 world = ci.worldOrigin + local * ci.lodScale;

        float4 clip = mul(float4(world, 1.0), gViewProj);
        if (clip.z <= 0.0) 
            continue;
        float3 ndc = clip.xyz / clip.w;
        ndc.xy += gJitter;
        if (ndc.x < -1.0 || ndc.x > 1.0 ||
            ndc.y < -1.0 || ndc.y > 1.0 ||
            ndc.z <  0.0 || ndc.z > 1.0) continue;
        int2 pix;
        pix.x = (int)((ndc.x * 0.5 + 0.5) * (float)gVwSize.x);
        pix.y = (int)((-ndc.y * 0.5 + 0.5) * (float)gVwSize.y);
        if (pix.x < 0 || pix.x >= (int)gVwSize.x ||
            pix.y < 0 || pix.y >= (int)gVwSize.y) continue;

        uint myLin = EncodeLinDepth(clip.w);
        uint colPck = gLwPalette[ci.paletteBase + palIdx];
        uint r8 = (colPck >>  0) & 0xFFu;
        uint g8 = (colPck >>  8) & 0xFFu;
        uint b8 = (colPck >> 16) & 0xFFu;

        // Per-voxel 6-bit visMask. Pass2 emits raw albedo + mask. Lighting
        // happens in dilate where ray-AABB hit-face is known per output pixel.
        uint mask = (i < 4u) ? ((visPack.x >> (i * 8u)) & 0x3Fu)
                              : ((visPack.y >> ((i - 4u) * 8u)) & 0x3Fu);
        if (mask == 0u) mask = 0x3Fu;

        // Depth-win gate FIRST: only the winning voxel pays for the AO fetch +
        // average loop and pack below; discarded voxels skip all of it.
        const uint kLinDepthSlop = 64u;
        if (myLin > gLwDepthSrv.Load(int3(pix, 0)) + kLinDepthSlop) continue;

        // Per-face AO (low 24 bits) for this winning voxel. Written to the AO
        // splat texture so the dilate can index it by the real hit face for Lit
        // shading — same per-face data the AO-viz uses, now in Lit too.
        uint aoFace24 = gLwBlockAo[(blockBase + (gid - firstThread)) * 8u + i] & 0x00FFFFFFu;
        gLwAoUav[pix] = aoFace24;

        // AO viz (gMode == 3): stash the full 24-bit packed per-face AO into the
        // RGB bits. Dilate indexes it by the true ray-AABB hit face (bestFace),
        // so AO is view-independent — no per-face guessing here.
        if ((int)gMode == 3)
        {
            r8 =  aoFace24        & 0xFFu;
            g8 = (aoFace24 >>  8) & 0xFFu;
            b8 = (aoFace24 >> 16) & 0xFFu;
        }

        // Pack: bits 0-7=R, 8-15=G, 16-23=B, 24-29=visMask, 30-31=marker(0b11).
        // Marker keeps argb != 0 so resolve's sky test ('color==0') still works
        // even for pitch-black albedo + visMask 0.
        uint argb32 = (3u << 30) | (mask << 24) | (b8 << 16) | (g8 << 8) | r8;

        // Pass2 writes color at the SINGLE projected pixel only (depth-win gate
        // already applied above). Pass1's NxN depth splat handles coverage; if
        // we splatted color too, multiple voxels with the same winning depth
        // would last-writer-wins per pixel → horizontal banding from dispatch
        // order. Single-pixel write keeps each voxel honest.
        gLwVisUav[pix] = argb32;
    }
}

// ============================================================
// Dilate — for each output pixel, scan NxN neighborhood. Pick the pixel with
// smallest depth that has a non-zero color. Output that color. Center has
// color → keep center. CSTiles csSplat does fancier visMask-aware fill; this
// is the minimum-viable nearest-neighbor variant.
// ============================================================
Texture2D<uint>   gDilateDepthSrv : register(t0);
Texture2D<uint>   gDilateColorIn  : register(t1);
Texture2D<uint>   gDilateAoIn     : register(t2); // per-face AO at splat winner
RWTexture2D<uint>  gDilateColorOut : register(u0);
RWTexture2D<float> gDilateDepthOut : register(u1); // gNearZ/viewZ form (CSTiles)

// Helpers used by dilate (and resolve below).
float3 PixelWorldDir(float2 pixCenter, float2 invScreen)
{
    float2 uv = pixCenter * invScreen;
    float ndcX = uv.x * 2.0 - 1.0;
    float ndcY = 1.0 - uv.y * 2.0;
    float3 v = float3(ndcX * gAspect * gTanHalfFovY, ndcY * gTanHalfFovY, 1.0);
    return normalize(gCamRight * v.x + gCamUp * v.y + gCamForward * v.z);
}

float3 SkyDome(float3 rd)
{
    float  t = saturate(rd.y * 0.5 + 0.5);
    float3 horizon = float3(0.70, 0.75, 0.85);
    float3 zenith  = float3(0.20, 0.40, 0.80);
    float3 ground  = gFogColor;
    float3 above   = lerp(horizon, zenith, smoothstep(0.5, 1.0, t));
    float3 col     = lerp(ground, above, smoothstep(0.48, 0.52, t));
    float NL   = dot(rd, gLightDir);
    float disc = smoothstep(0.9995, 0.99975, NL);
    float glow = pow(saturate(NL), 64.0) * 0.4;
    col += float3(1.5, 1.35, 1.05) * (disc * 6.0 + glow) * gSunIntensity;
    return col;
}

// Port of CSTiles csmain_splat — for each output pixel, scan NxN neighbors.
// Each neighbor with color = represents a voxel center reconstructable from
// its linear depth + screen coords. Ray-AABB against snapped LOD-grid voxel.
// Pick nearest hit; output that color. Eliminates the "constant-color square"
// artifact of nearest-neighbor dilate by giving each output pixel its own
// geometrically-correct color.
// Shared dilate body — `pix` is the output pixel (raw dispatch or swizzled).
void DilateAt(int2 pix)
{
    int W = (int)gVwSize.x;
    int H = (int)gVwSize.y;
    if (pix.x >= W || pix.y >= H) return;

    float3 ro = gCamPos;
    float2 invScreen = float2(1.0 / (float)W, 1.0 / (float)H);
    int    R = max(1, gSplatRadius);

    // Source select: if THIS pixel is itself a splat hit, use it directly — no
    // search (the old ray-AABB reconstruction could miss a pixel's own splat).
    // Otherwise expand square rings outward; the FIRST ring containing any hit
    // wins, and within that ring the nearest-depth hit is taken.
    uint  pck   = 0;
    int2  srcSp = pix;
    float viewZ = 0.0;
    bool  found = false;
    {
        uint c0 = gDilateColorIn.Load(int3(pix, 0));
        uint d0 = gDilateDepthSrv.Load(int3(pix, 0));
        if (c0 != 0u && d0 != 0xFFFFFFFFu) {
            pck = c0; srcSp = pix; viewZ = (float)d0 / kLinDepthScale; found = true;
        } else {
            float bestD = 1e30;
            [loop] for (int r = 1; r <= R && !found; ++r) {
                bool ringHit = false;
                [loop] for (int dy = -r; dy <= r; ++dy)
                [loop] for (int dx = -r; dx <= r; ++dx) {
                    if (max(abs(dx), abs(dy)) != r) continue;   // ring perimeter only
                    int2 sp = pix + int2(dx, dy);
                    if (sp.x < 0 || sp.x >= W || sp.y < 0 || sp.y >= H) continue;
                    uint c = gDilateColorIn.Load(int3(sp, 0));
                    if (c == 0u) continue;
                    uint d = gDilateDepthSrv.Load(int3(sp, 0));
                    if (d == 0xFFFFFFFFu) continue;
                    float vz = (float)d / kLinDepthScale;
                    if (vz < bestD) { bestD = vz; pck = c; srcSp = sp; }
                    ringHit = true;
                }
                if (ringHit) { found = true; viewZ = bestD; }
            }
        }
    }
    if (!found) {
        gDilateColorOut[pix] = 0;
        gDilateDepthOut[pix] = 0.0;
        return;
    }

    // Reconstruct the chosen voxel's centre from (srcSp, viewZ) for triplanar
    // shading. LOD (voxel size S) derived from depth, same as pass2's selection.
    int2  aoSp = srcSp;
    float2 uvN  = (float2(srcSp) + 0.5) * invScreen;
    float ndcX = uvN.x * 2.0 - 1.0;
    float ndcY = 1.0 - uvN.y * 2.0;
    float vx = ndcX * gAspect * gTanHalfFovY * viewZ;
    float vy = ndcY * gTanHalfFovY * viewZ;
    float3 worldN = ro + gCamRight * vx + gCamUp * vy + gCamForward * viewZ;
    float focalPx = gScreenSize.y * 0.5 / gTanHalfFovY;
    int   lodIdx = (int)clamp(floor(log2(max(viewZ / focalPx, 1.0))), 0.0, 4.0);
    float S = (float)(1u << lodIdx);
    float3 centerW = floor(worldN / S) * S + S * 0.5;

    gDilateDepthOut[pix] = gNearZ / max(viewZ, 1e-4);

    // Decode raw albedo + visMask; triplanar normal from camera→centre direction,
    // visMask-gated and weighted by projected face area (vdir.axis²).
    uint mask = (pck >> 24) & 0x3Fu;
    float3 albedo = float3(
        (float)( pck        & 0xFFu) / 255.0,
        (float)((pck >>  8) & 0xFFu) / 255.0,
        (float)((pck >> 16) & 0xFFu) / 255.0);

    float3 vdir = normalize(gCamPos - centerW);
    uint f0 = (vdir.x >= 0.0) ? 0u : 1u;
    uint f1 = (vdir.y >= 0.0) ? 2u : 3u;
    uint f2 = (vdir.z >= 0.0) ? 4u : 5u;
    float w0 = ((mask >> f0) & 1u) ? vdir.x * vdir.x : 0.0;
    float w1 = ((mask >> f1) & 1u) ? vdir.y * vdir.y : 0.0;
    float w2 = ((mask >> f2) & 1u) ? vdir.z * vdir.z : 0.0;
    float wsum = max(1e-4, w0 + w1 + w2);
    w0 /= wsum; w1 /= wsum; w2 /= wsum;
    float3 N0 = float3(sign(vdir.x), 0, 0);
    float3 N1 = float3(0, sign(vdir.y), 0);
    float3 N2 = float3(0, 0, sign(vdir.z));
    float3 Nshade = normalize(w0 * N0 + w1 * N1 + w2 * N2);
    float  NdotL  = w0 * max(0.0, dot(N0, gLightDir))
                  + w1 * max(0.0, dot(N1, gLightDir))
                  + w2 * max(0.0, dot(N2, gLightDir));
    uint   domFace = (w0 >= w1 && w0 >= w2) ? f0 : (w1 >= w2 ? f1 : f2);

    float3 lit;
    if ((int)gMode == 2)
    {
        const float3 kFaceColor[6] = {
            float3(1.0, 0.2, 0.2), float3(1.0, 0.2, 1.0),
            float3(0.2, 1.0, 0.2), float3(1.0, 0.5, 0.1),
            float3(0.2, 0.4, 1.0), float3(0.2, 1.0, 1.0),
        };
        lit = kFaceColor[domFace];           // dominant triplanar face
    }
    else if ((int)gMode == 3)
    {
        // AO viz: average the visible faces' baked AO (low 24 bits of pck) — stable
        // under camera move (no per-pixel face now).
        uint aoW = pck & 0x00FFFFFFu;
        float s = 0.0; uint n = 0u;
        [unroll]
        for (uint f = 0u; f < 6u; ++f) {
            if (((mask >> f) & 1u) == 0u) continue;
            s += (float)((aoW >> (f * 4u)) & 0xFu) * (1.0 / 15.0); ++n;
        }
        float ao = (n > 0u) ? (s / (float)n) : 1.0;
        lit = float3(ao, ao, ao);
    }
    else
    {
        // Per-face baked AO from the source pixel, averaged over visible faces.
        uint  aoPck = gDilateAoIn.Load(int3(aoSp, 0));
        float s = 0.0; uint n = 0u;
        [unroll]
        for (uint f = 0u; f < 6u; ++f) {
            if (((mask >> f) & 1u) == 0u) continue;
            s += (float)((aoPck >> (f * 4u)) & 0xFu) * (1.0 / 15.0); ++n;
        }
        float ao = (n > 0u) ? (s / (float)n) : 1.0;
        float3 sunCol = float3(1.0, 0.95, 0.85) * (gSunIntensity * 3.0);
        float3 amb    = AmbientCube(Nshade) * gAmbient;
        lit = albedo * (amb + NdotL * sunCol);
        lit *= lerp(1.0, ao, gAoStrength);   // baked AO doubles as contact shadow
        lit = ApplyFog(lit, centerW);
    }

    uint rR = (uint)clamp(lit.r * 255.0, 0.0, 255.0);
    uint rG = (uint)clamp(lit.g * 255.0, 0.0, 255.0);
    uint rB = (uint)clamp(lit.b * 255.0, 0.0, 255.0);
    gDilateColorOut[pix] = 0xFF000000u | (rB << 16) | (rG << 8) | rR;
}

// Linear dispatch: pixel = SV_DispatchThreadID.
[numthreads(8, 8, 1)]
void csmain_dilate(uint3 dt : SV_DispatchThreadID)
{
    DilateAt(int2(dt.xy));
}

// Per-dispatch base group column (root constant b2). The CPU issues two EXACT
// dispatches so no group is ever out of the X range:
//   main:      Dispatch(gTileCS, gridY, gridX / gTileCS),  gColumnBase = 0
//   remainder: Dispatch(gridX % gTileCS, gridY, 1),        gColumnBase = full*tileW
// Groups scheduled together cover a gTileCS-wide vertical strip (L2-friendly).
cbuffer CBDilateBase : register(b2) { uint gColumnBase; }

[numthreads(8, 8, 1)]
void csmain_dilate_swizzle(uint3 GTid : SV_GroupThreadID, uint3 GId : SV_GroupID)
{
    uint column = gColumnBase + GId.x + GId.z * gTileCS;  // real group column
    uint2 pix   = uint2(column * 8u, GId.y * 8u) + GTid.xy;
    DilateAt(int2(pix));
}

// ============================================================
// Resolve — fullscreen triangle reads depth + color, outputs RGB.
// ============================================================
struct VOut { float4 pos : SV_Position; };
VOut vsmain_resolve(uint vid : SV_VertexID)
{
    float2 p = float2((vid == 1u) ? 3.0 : -1.0,
                      (vid == 2u) ? 3.0 : -1.0);
    VOut o;
    o.pos = float4(p, 0.0, 1.0);
    return o;
}

Texture2D<uint>   gResolveDepthSrv : register(t0);
Texture2D<uint>   gResolveColorSrv : register(t1);
Texture2D<float4> gTaaHistSrv      : register(t2);
SamplerState      gLinearClamp     : register(s0);
// PixelWorldDir + SkyDome moved up before csmain_dilate.

float4 psmain_resolve(VOut i) : SV_Target
{
    int2 pix = int2(i.pos.xy);
    // Dilate writes color into pixels that have no original depth → can't gate
    // on depth-sentinel for sky test. Gate on color: 0 = no voxel reached this
    // pixel (truly sky). Non-zero = voxel color (own or dilated from neighbor).
    uint argb = gResolveColorSrv.Load(int3(pix, 0));
    float3 cur;
    if (argb == 0u)
    {
        uint w, h; gResolveColorSrv.GetDimensions(w, h);
        float3 rd = PixelWorldDir(i.pos.xy, float2(1.0 / (float)w, 1.0 / (float)h));
        cur = SkyDome(rd);
    }
    else
    {
        float3 col;
        col.r = (float)( argb        & 0xFFu) / 255.0;
        col.g = (float)((argb >>  8) & 0xFFu) / 255.0;
        col.b = (float)((argb >> 16) & 0xFFu) / 255.0;
        cur = col;
    }

    // Resolve writes scene-only (linear HDR). TAA + post run in m4_taa_post.hlsl.
    // alpha 1 = surface hit (real depth), alpha 0 = sky — used by psmain_taa to
    // decide whether to reproject vs. take current sample directly.
    uint argbCheck = gResolveColorSrv.Load(int3(pix, 0));
    float alpha = (argbCheck == 0u) ? 0.0 : 1.0;
    return float4(cur, alpha);
}

// Final post — sample taaHist (linear HDR after TAA), apply exposure + ACES
// tonemap, write to backbuffer. Godrays / sky gradient slot in here later.
float4 psmain_blit(VOut i) : SV_Target
{
    uint w, h;
    gTaaHistSrv.GetDimensions(w, h);
    float2 uv = i.pos.xy / float2((float)w, (float)h);
    float3 c = gTaaHistSrv.SampleLevel(gLinearClamp, uv, 0).rgb;

    // ACES filmic tonemap (Narkowicz fit).
    c = c * gExposure;
    const float a = 2.51, ta = 0.03, tc = 2.43, td = 0.59, te = 0.14;
    c = saturate((c * (a * c + ta)) / (c * (tc * c + td) + te));
    c.r = 1.0;
    
    return float4(c, 1.0);
}
