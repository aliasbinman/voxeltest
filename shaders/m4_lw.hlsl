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

// CSTiles cbPerFrame byte layout — identical to postfx.hlsl. Same offsets so
// shaders ported from CSTiles compile + read correct fields.
cbuffer cbPerFrame : register(b0)
{
    row_major float4x4 gViewProj;     //   0
    float3   gCamPos;                  //  64
    float    gMode;                    //  76
    float3   gLightDir;                //  80
    float    gAmbient;                 //  92
    float3   gPointNormal;             //  96
    float    _pad0;                    // 108
    row_major float4x4 gInvViewProj;   // 112
    float2   gScreenSize;              // 176
    float2   _pad1;                    // 184
    float3   gCamRight;                // 192
    float    _pad3;                    // 204
    float3   gCamUp;                   // 208
    float    _pad4;                    // 220
    float3   gCamForward;              // 224
    float    gTanHalfFovY;             // 236
    float3   gFogColor;                // 240
    float    gFogDensity;              // 252
    float    gHeightFogDensity;        // 256
    float    gHeightFogFalloff;        // 260
    float    gHeightFogStart;          // 264
    float    _padHF;                   // 268
    float3   gSceneOrigin;             // 272
    float    gNearZ;                   // 284
    float3   gSceneSpan;               // 288
    float    _pad6;                    // 300
    row_major float4x4 gPrevViewProj;  // 304
    float2   gJitter;                  // 368
    float2   _pad7;                    // 376
    row_major float4x4 gSunViewProj;   // 384
    float    gShadowBias;              // 448
    float    gShadowMapSize;           // 452
    float    gShadowEnable;            // 456
    float    gSunIntensity;            // 460
    float    gExposure;                // 464
    float    gRoughness;               // 468
    float    gColorizeClusters;        // 472
    float    gGridSize;                // 476
    float2   gInvScreenSize;           // 480
    float    gAspect;                  // 488
    float    gInvAspect;               // 492
    float    gAspectTanFov;            // 496
    float3   _padPC;                   // 500..512
    // Burnout Paradise reproject matrix rows (HScreen-UV). Mvel = Mh1_to_h0 - I.
    float4   gReprojMx;                // 512  (mxx, mxy, mxz, mxw)
    float4   gReprojMy;                // 528  (myx, myy, myz, myw)
    float4   gReprojMw;                // 544  (mwx, mwy, mwz, mww)
    float4   _padReproj;               // 560..576
};

cbuffer CBLwCS : register(b1)
{
    uint2 gVwSize;
    uint  gLwPointCount;
    uint  gLwNumWorkItems;
    uint  gLwLodIdx;
    int   gSplatRadius;     // 0 = single pixel; N writes (2N+1)x(2N+1).
    uint2 _padCs;
};

struct LwChunkInfo
{
    float3 worldOrigin;
    float  lodScale;
    uint   poolBase;
    uint   paletteBase;
    uint2  _padCI;
};

StructuredBuffer<LwChunkInfo> gLwChunkInfos : register(t0);
StructuredBuffer<uint>        gLwPalette    : register(t1);
StructuredBuffer<uint>        gLwBlockPos   : register(t2);
StructuredBuffer<uint2>       gLwBlockCol   : register(t3);
StructuredBuffer<uint4>       gLwWorkItems  : register(t4);
Texture2D<uint>               gLwDepthSrv   : register(t5);
StructuredBuffer<uint2>       gLwBlockVis   : register(t6);

RWTexture2D<uint>             gLwVisUav     : register(u0);

// Linear-depth encoding — uniform precision over [0, 100km] view-Z.
static const float kLinDepthFar   = 100000.0;
static const float kLinDepthMaxF  = 4294967294.0; // 0xFFFFFFFE
static const float kLinDepthScale = kLinDepthMaxF / kLinDepthFar;
uint EncodeLinDepth(float viewZ)
{
    return (uint)clamp(viewZ * kLinDepthScale, 0.0, kLinDepthMaxF);
}

// gid -> work item via binary search over the cumulative firstThread (.w).
uint4 LookupItem(uint gid)
{
    uint lo = 0u;
    uint hi = gLwNumWorkItems;
    while (lo + 1u < hi)
    {
        uint mid = (lo + hi) >> 1u;
        if (gLwWorkItems[mid].w <= gid) lo = mid;
        else                            hi = mid;
    }
    return gLwWorkItems[lo];
}

// ============================================================
// Pass 1 — write linear depth via InterlockedMin.
// ============================================================
[numthreads(64, 1, 1)]
void csmain_pass1_depth(uint3 dt : SV_DispatchThreadID)
{
    uint gid = dt.x;
    if (gid >= gLwPointCount) return;
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
        if (((occ >> i) & 1u) == 0u) continue;
        uint lx = (i >> 0) & 1u;
        uint ly = (i >> 1) & 1u;
        uint lz = (i >> 2) & 1u;
        float3 local = float3(
            (float)(bx * 2u + lx),
            (float)(by * 2u + ly),
            (float)(bz * 2u + lz)) + 0.5;
        float3 world = ci.worldOrigin + local * ci.lodScale;

        float4 clip = mul(float4(world, 1.0), gViewProj);
        if (clip.w <= 0.0) continue;
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
    if (gid >= gLwPointCount) return;
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
        if (clip.w <= 0.0) continue;
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

        // Pack: bits 0-7=R, 8-15=G, 16-23=B, 24-29=visMask, 30-31=marker(0b11).
        // Marker keeps argb != 0 so resolve's sky test ('color==0') still works
        // even for pitch-black albedo + visMask 0.
        uint argb32 = (3u << 30) | (mask << 24) | (b8 << 16) | (g8 << 8) | r8;

        // Pass2 writes color at the SINGLE projected pixel only. Pass1's NxN
        // depth splat handles coverage; if we splatted color too, multiple
        // voxels with the same winning depth would last-writer-wins per pixel
        // → horizontal banding from dispatch order. Single-pixel write keeps
        // each voxel honest, accepting holes that Pass1's depth-only splat
        // leaves at distance.
        const uint kLinDepthSlop = 64u;
        uint winDepth = gLwDepthSrv.Load(int3(pix, 0));
        if (myLin > winDepth + kLinDepthSlop) continue;
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

// ApplyFog — verbatim port from CSTiles shading.hlsli.
float3 ApplyFog(float3 color, float3 wpos)
{
    float3 d = wpos - gCamPos;
    float dist = length(d);
    float optical = gFogDensity * dist;
    float3 rd = (dist > 1e-4) ? d / dist : float3(0,0,1);
    if (gHeightFogDensity > 0.0 && dist > 1e-4) {
        float b  = gHeightFogFalloff;
        float c  = gHeightFogDensity;
        float ey = exp(-(gCamPos.y - gHeightFogStart) * b);
        float t;
        if (abs(rd.y) > 1e-4)
            t = c * ey * (1.0 - exp(-dist * rd.y * b)) / rd.y;
        else
            t = c * ey * dist;
        optical += max(t, 0.0);
    }
    if (optical <= 0.0)
        return color;
    float3 sunDir  = normalize(gLightDir);
    float  sunAmt  = pow(saturate(dot(rd, sunDir)), 8.0);
    float3 sunTint = float3(1.10, 0.85, 0.55);
    float3 fogCol  = lerp(gFogColor, sunTint, sunAmt);
    return lerp(fogCol, color, exp(-optical));
}

// Ambient cube — warm/cool per axis face. Sampled triplanar from the surface
// normal so each cube face picks up a different indirect tint instead of one
// flat scalar.
static const float3 kAmbientCube[6] = {
    float3(0.85, 0.65, 0.45),  // +X warm
    float3(0.40, 0.50, 0.65),  // -X cool
    float3(0.85, 1.00, 1.20),  // +Y sky
    float3(0.18, 0.14, 0.10),  // -Y ground
    float3(0.65, 0.65, 0.55),  // +Z
    float3(0.40, 0.45, 0.55),  // -Z
};
float3 AmbientCube(float3 n)
{
    float3 an = abs(n);
    float t = max(an.x + an.y + an.z, 1e-5);
    float3 amb = an.x * (n.x > 0.0 ? kAmbientCube[0] : kAmbientCube[1])
               + an.y * (n.y > 0.0 ? kAmbientCube[2] : kAmbientCube[3])
               + an.z * (n.z > 0.0 ? kAmbientCube[4] : kAmbientCube[5]);
    return amb / t;
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
[numthreads(8, 8, 1)]
void csmain_dilate(uint3 dt : SV_DispatchThreadID)
{
    int2 pix = int2(dt.xy);
    int W = (int)gVwSize.x;
    int H = (int)gVwSize.y;
    if (pix.x >= W || pix.y >= H) return;

    float3 ro = gCamPos;
    float2 invScreen = float2(1.0 / (float)W, 1.0 / (float)H);
    float3 rd = PixelWorldDir(float2(pix) + 0.5, invScreen);
    float3 invRd = 1.0 / rd;

    int  R     = max(1, gSplatRadius);

    // Two candidates:
    //   bestHit  = AABB hit on an exposed face (sharp per-pixel face shading)
    //   fallback = nearest-depth valid neighbor center (triplanar splat shading
    //              for pixels where no exposed-face AABB hit reaches us — small
    //              distant voxels where the surface sub-sample misses)
    float bestT       = 1e30;
    uint  bestPck     = 0;
    uint  bestFace    = 0;
    bool  haveHit     = false;
    float fbBestD     = 1e30;
    uint  fbPck       = 0;
    float3 fbCenter   = float3(0, 0, 0);
    bool  haveFb      = false;

    [loop] for (int dy = -R; dy <= R; ++dy) {
    [loop] for (int dx = -R; dx <= R; ++dx) {
        int2 sp = pix + int2(dx, dy);
        if (sp.x < 0 || sp.x >= W || sp.y < 0 || sp.y >= H) continue;
        uint c = gDilateColorIn.Load(int3(sp, 0));
        if (c == 0) continue;
        uint d = gDilateDepthSrv.Load(int3(sp, 0));
        if (d == 0xFFFFFFFFu) continue;

        float viewZ = (float)d / kLinDepthScale;
        float2 uvN  = (float2(sp) + 0.5) * invScreen;
        float ndcX = uvN.x * 2.0 - 1.0;
        float ndcY = 1.0 - uvN.y * 2.0;
        float vx = ndcX * gAspect * gTanHalfFovY * viewZ;
        float vy = ndcY * gTanHalfFovY * viewZ;
        float3 worldN = ro + gCamRight * vx + gCamUp * vy + gCamForward * viewZ;

        // Derive LOD from depth so AABB matches actual voxel size. CPU LOD
        // selection uses focalPx*(2^L)/dist >= 1/lodScale; reverse:
        //   2^L ≈ viewZ * thresh / focalPx
        // For thresh=1 (default lodScale=1): L ≈ log2(viewZ / focalPx).
        float focalPx = gScreenSize.y * 0.5 / gTanHalfFovY;
        float Lf = log2(max(viewZ / focalPx, 1.0));
        int lodIdx = (int)clamp(floor(Lf), 0.0, 4.0);
        float S = (float)(1u << lodIdx);

        // Track nearest-depth fallback for triplanar splat lighting.
        if (viewZ < fbBestD) {
            fbBestD  = viewZ;
            fbPck    = c;
            float3 vmin0 = floor(worldN / S) * S;
            fbCenter = vmin0 + S * 0.5;
            haveFb = true;
        }

        float3 vmin = floor(worldN / S) * S;
        float3 vmax = vmin + S;
        float3 t0v = (vmin - ro) * invRd;
        float3 t1v = (vmax - ro) * invRd;
        float3 tmn = min(t0v, t1v);
        float3 tmx = max(t0v, t1v);
        float tNear = max(max(tmn.x, tmn.y), tmn.z);
        float tFar  = min(min(tmx.x, tmx.y), tmx.z);
        if (tFar < 0.0 || tNear > tFar) continue;
        float tHit = max(tNear, 0.0);

        uint face;
        if (tNear == tmn.x)      face = (rd.x > 0.0) ? 1u : 0u;
        else if (tNear == tmn.y) face = (rd.y > 0.0) ? 3u : 2u;
        else                     face = (rd.z > 0.0) ? 5u : 4u;

        // visMask face rejection disabled — until pass2 encodes lodIdx, the
        // shader-derived LOD can put the AABB on a sub-cell of the real voxel,
        // making ray-entry face unreliable vs the voxel's true visMask.
        if (tHit < bestT) {
            bestT    = tHit;
            bestPck  = c;
            bestFace = face;
            haveHit  = true;
        }
    }}

    if (!haveHit) {
        // No ray-AABB hit at this pixel → sky. Write 0 color AND 0 depth so
        // TAA / post both treat this pixel as sky (no reprojection from fake
        // depth that would bleed neighbor color into history).
        gDilateColorOut[pix] = 0;
        gDilateDepthOut[pix] = 0.0;
        return;
    }

    // Depth in CSTiles form: d = gNearZ / viewZ. psmain_taa reads as
    // Texture2D<float> and recovers viewZ = gNearZ / d.
    float viewZOut;
    if (haveHit) {
        float3 hitPos = ro + rd * bestT;
        viewZOut = max(0.001, dot(hitPos - gCamPos, gCamForward));
    } else {
        viewZOut = fbBestD;
    }
    gDilateDepthOut[pix] = gNearZ / max(viewZOut, 1e-4);

    // Decode raw albedo + visMask.
    uint mask = (bestPck >> 24) & 0x3Fu;
    float3 albedo = float3(
        (float)( bestPck        & 0xFFu) / 255.0,
        (float)((bestPck >>  8) & 0xFFu) / 255.0,
        (float)((bestPck >> 16) & 0xFFu) / 255.0);

    const float3 kFaceN[6] = {
        float3( 1,0,0), float3(-1,0,0),
        float3(0, 1,0), float3(0,-1,0),
        float3(0,0, 1), float3(0,0,-1),
    };

    float  NdotL;
    float3 Nshade;
#if USE_HIT_FACE
    // --- Mode 1: per-pixel hit-face (sharp) ---
    if (haveHit)
    {
        Nshade = kFaceN[bestFace];
        NdotL  = max(0.0, dot(Nshade, gLightDir));
    }
    else
    {
        // No hit → triplanar fallback against fb neighbor's center.
        float3 vdir = normalize(gCamPos - fbCenter);
        uint fb0 = (vdir.x >= 0.0) ? 0u : 1u;
        uint fb1 = (vdir.y >= 0.0) ? 2u : 3u;
        uint fb2 = (vdir.z >= 0.0) ? 4u : 5u;
        float w0 = ((mask >> fb0) & 1u) ? vdir.x * vdir.x : 0.0;
        float w1 = ((mask >> fb1) & 1u) ? vdir.y * vdir.y : 0.0;
        float w2 = ((mask >> fb2) & 1u) ? vdir.z * vdir.z : 0.0;
        float wsum = max(1e-4, w0 + w1 + w2);
        w0 /= wsum; w1 /= wsum; w2 /= wsum;
        float3 N0 = float3(sign(vdir.x), 0, 0);
        float3 N1 = float3(0, sign(vdir.y), 0);
        float3 N2 = float3(0, 0, sign(vdir.z));
        Nshade = normalize(w0 * N0 + w1 * N1 + w2 * N2);
        NdotL  = w0 * max(0.0, dot(N0, gLightDir))
               + w1 * max(0.0, dot(N1, gLightDir))
               + w2 * max(0.0, dot(N2, gLightDir));
    }
#else
    // --- Mode 0: triplanar visMask-weighted (smooth, always used) ---
    // Pick voxel center: AABB-hit position if we got one, else fallback
    // neighbor center. Same triplanar formula either way.
    float3 centerW = haveHit ? floor((ro + rd * bestT) / 1.0) + 0.5
                              : fbCenter;
    float3 vdir = normalize(gCamPos - centerW);
    uint fb0 = (vdir.x >= 0.0) ? 0u : 1u;
    uint fb1 = (vdir.y >= 0.0) ? 2u : 3u;
    uint fb2 = (vdir.z >= 0.0) ? 4u : 5u;
    float w0 = ((mask >> fb0) & 1u) ? vdir.x * vdir.x : 0.0;
    float w1 = ((mask >> fb1) & 1u) ? vdir.y * vdir.y : 0.0;
    float w2 = ((mask >> fb2) & 1u) ? vdir.z * vdir.z : 0.0;
    float wsum = max(1e-4, w0 + w1 + w2);
    w0 /= wsum; w1 /= wsum; w2 /= wsum;
    float3 N0 = float3(sign(vdir.x), 0, 0);
    float3 N1 = float3(0, sign(vdir.y), 0);
    float3 N2 = float3(0, 0, sign(vdir.z));
    Nshade = normalize(w0 * N0 + w1 * N1 + w2 * N2);
    NdotL  = w0 * max(0.0, dot(N0, gLightDir))
           + w1 * max(0.0, dot(N1, gLightDir))
           + w2 * max(0.0, dot(N2, gLightDir));
#endif

    float3 lit;
    if ((int)gMode == 2)
    {
        const float3 kFaceColor[6] = {
            float3(1.0, 0.2, 0.2),
            float3(1.0, 0.2, 1.0),
            float3(0.2, 1.0, 0.2),
            float3(1.0, 0.5, 0.1),
            float3(0.2, 0.4, 1.0),
            float3(0.2, 1.0, 1.0),
        };
        // Hit path: per-pixel face color. Fallback path: gray (triplanar).
        lit = haveHit ? kFaceColor[bestFace] : float3(0.5, 0.5, 0.5);
    }
    else
    {
        float3 sunCol = float3(1.0, 0.95, 0.85) * (gSunIntensity * 3.0);
        float3 amb    = AmbientCube(Nshade) * gAmbient;
        float3 light  = amb + NdotL * sunCol;
        lit = albedo * light;
        // Fog applied in world space — distance + height-based attenuation
        // matching CSTiles ApplyFog. Hit pixels use ray-AABB hit position;
        // fallback pixels use fbCenter (close enough — neighbor's voxel).
        float3 wposShade = haveHit ? (ro + rd * bestT) : fbCenter;
        lit = ApplyFog(lit, wposShade);
    }

    uint rR = (uint)clamp(lit.r * 255.0, 0.0, 255.0);
    uint rG = (uint)clamp(lit.g * 255.0, 0.0, 255.0);
    uint rB = (uint)clamp(lit.b * 255.0, 0.0, 255.0);
    gDilateColorOut[pix] = 0xFF000000u | (rB << 16) | (rG << 8) | rR;
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
    return float4(c, 1.0);
}
