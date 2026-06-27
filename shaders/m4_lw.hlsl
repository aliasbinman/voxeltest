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
        // Single-pixel atomic-min. The dilate fills coverage (ray-AABB) so the
        // splat stays one pixel per voxel — keeps silhouettes definable/tight.
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

        // Depth-win gate: only the winner shades + writes its single pixel.
        const uint kLinDepthSlop = 64u;
        if (myLin > gLwDepthSrv.Load(int3(pix, 0)) + kLinDepthSlop) continue;

        // Emit RAW shading inputs — the dilate lights per output pixel using the
        // actual ray-AABB hit FACE (so each face shades separately instead of one
        // averaged colour). visColor = albedo + visMask; visAo = per-face AO + LOD.
        uint colPck = gLwPalette[ci.paletteBase + palIdx];
        uint mask = (i < 4u) ? ((visPack.x >> (i * 8u)) & 0x3Fu)
                              : ((visPack.y >> ((i - 4u) * 8u)) & 0x3Fu);
        if (mask == 0u) mask = 0x3Fu;

        // Cull a splat with no camera-facing visible face (back face only): the
        // dilate would otherwise light a face pointing away from the camera.
        float3 vdir = gCamPos - world;
        bool anyFront = false;
        [unroll] for (uint f = 0u; f < 6u; ++f)
            if (((mask >> f) & 1u) && dot(kFaceN[f], vdir) > 0.0) anyFront = true;
        if (!anyFront) continue;

        uint aoFace24 = gLwBlockAo[(blockBase + (gid - firstThread)) * 8u + i] & 0x00FFFFFFu;
        uint L = firstbithigh((uint)ci.lodScale);   // lodScale is a power of two

        // visColor packs everything the dilate needs for COVERAGE + shading in one
        // reliable texture: albedo 7-7-6 (bits 0-19), visMask (20-25), LOD (26-29),
        // marker (30-31, keeps argb != 0). LOD lives here (not visAo) because the
        // footprint size depends on it and must always be correct. visAo carries
        // only the cosmetic per-face AO value.
        uint r7 = (colPck        & 0xFFu) >> 1;       // 8->7
        uint g7 = ((colPck >> 8) & 0xFFu) >> 1;       // 8->7
        uint b6 = ((colPck >> 16) & 0xFFu) >> 2;      // 8->6
        uint alb = r7 | (g7 << 7) | (b6 << 14);       // 20-bit albedo
        gLwVisUav[pix] = (3u << 30) | (L << 26) | (mask << 20) | alb;
        gLwAoUav[pix]  = aoFace24;                     // per-face AO (6 * 4-bit)
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
// Footprint coverage (no bleed) + PER-FACE lighting. The splat stores raw albedo
// + visMask (visColor) and per-face AO + LOD (visAo). Coverage is decided by the
// projected footprint, then the WINNING voxel is reconstructed and ray-AABB'd by
// THIS pixel's ray to find the hit face, so each face shades separately (matches
// the octet PS via ShadeFace) instead of one averaged colour per voxel.

// Projected half-size in px of a voxel: 0.5 * S * focalPx / viewZ * (1/cosθ).
// The 1/cosθ edge stretch (unnormalised view-ray length) makes the footprint
// match real splat spacing at the screen sides; +1px closes rounding seams.
float DilateHalfPx(int2 sp, float S, float vzN, float focalPx, float2 invScreen)
{
    float2 uvN = (float2(sp) + 0.5) * invScreen;
    float3 vd  = float3((uvN.x * 2.0 - 1.0) * gAspect * gTanHalfFovY,
                        (1.0 - uvN.y * 2.0) * gTanHalfFovY, 1.0);
    return 0.5 * S * focalPx * length(vd) / max(vzN, 1e-4) + 1.0;
}

// Shade the winning splat for output `pix`: reconstruct its voxel, ray-AABB for
// the hit face, light that face. c = albedo+visMask, aoPck = per-face AO + LOD,
// d = view depth, winSp = winner splat pixel. Returns packed lit colour + viewZ.
uint ShadeDilateWinner(int2 winSp, uint c, uint aoPck, uint d,
                       float3 ro, float3 rd, float3 invRd, float2 invScreen,
                       out float outViewZ)
{
    float  vzN    = (float)d / kLinDepthScale;
    uint   mask   = (c >> 20) & 0x3Fu;
    float3 albedo = float3((c & 0x7Fu) / 127.0, ((c >> 7) & 0x7Fu) / 127.0,
                           ((c >> 14) & 0x3Fu) / 63.0);          // 7-7-6
    float  S      = (float)(1u << ((c >> 26) & 0xFu));           // LOD from colour

    // Voxel from winner splat pixel + its depth, snapped to the LOD grid. (Used
    // only for the FACE; coverage already came from the footprint, so a sub-pixel
    // snap error can't bleed — at worst it picks a slightly different face.)
    float2 uvN = (float2(winSp) + 0.5) * invScreen;
    float3 center = ro + gCamRight * ((uvN.x * 2.0 - 1.0) * gAspect * gTanHalfFovY * vzN)
                       + gCamUp    * ((1.0 - uvN.y * 2.0) * gTanHalfFovY * vzN)
                       + gCamForward * vzN;
    float3 vmin = floor(center / S) * S;

    float tHit; uint face; float3 hitW;
    // tHit > epsilon: a real hit in FRONT of the camera. RayAabb clamps tHit to 0
    // when the camera is inside the box — which happens for near voxels the octet
    // near-plane cull pushed to the splat path; without this guard every such
    // pixel "hits" at t=0 and the screen looks like the inside of one big cube.
    if (RayAabb(ro, rd, invRd, vmin, vmin + S, mask, tHit, face) && tHit > 1e-3)
    {
        hitW = ro + rd * tHit;
        outViewZ = max(0.001, dot(hitW - gCamPos, gCamForward));
    }
    else
    {
        // Ray missed (or camera inside) — fall back to the dominant camera-facing
        // visible face so coverage pixels still shade sensibly.
        float best = -1e9; face = 0u;
        float3 vc = gCamPos - center;
        [unroll] for (uint f = 0u; f < 6u; ++f)
        {
            if (((mask >> f) & 1u) == 0u) continue;
            float dd = dot(kFaceN[f], vc);
            if (dd > best) { best = dd; face = f; }
        }
        hitW = center;
        outViewZ = vzN;
    }

    float  ao  = (float)((aoPck >> (face * 4u)) & 0xFu) * (1.0 / 15.0);
    float3 lit = ShadeFace(albedo, face, ao, hitW);
    uint R8 = (uint)clamp(lit.r * 255.0, 0.0, 255.0);
    uint G8 = (uint)clamp(lit.g * 255.0, 0.0, 255.0);
    uint B8 = (uint)clamp(lit.b * 255.0, 0.0, 255.0);
    return 0xFF000000u | (B8 << 16) | (G8 << 8) | R8;
}

// Scalar fallback (R > pxPadMax): no LDS, reads textures directly.
void DilateAtRef(int2 pix)
{
    int W = (int)gVwSize.x;
    int H = (int)gVwSize.y;
    if (pix.x >= W || pix.y >= H) return;

    float3 ro = gCamPos;
    float3 rd = PixelWorldDir(float2(pix) + 0.5, gInvVwSize);
    float3 invRd = 1.0 / rd;
    int    R = max(gSplatRadius, 1);

    // Primary: nearest ray-cube hit (watertight). Fallback: footprint coverage,
    // nearest depth (fills sub-pixel-size voxels in the distance + boundary misses
    // up close where depth quantisation snaps the cube to the wrong cell).
    int2 winSp = int2(0, 0); uint winC = 0, winAo = 0; uint bestD = 0xFFFFFFFFu;
    float bestT = 1e30;            // nearest ray-cube hit distance
    bool found = false;            // slab hit found
    int2 fbSp = int2(0, 0); uint fbC = 0, fbAo = 0, fbD = 0xFFFFFFFFu;
    bool fbFound = false;          // footprint fallback found
    [loop] for (int dy = -R; dy <= R; ++dy)
    {
        [loop] for (int dx = -R; dx <= R; ++dx)
        {
            int2 sp = pix + int2(dx, dy);
            if (sp.x < 0 || sp.x >= W || sp.y < 0 || sp.y >= H) 
                continue;
            
            // Just get the depth and determine t of when this ray intersects box.
            
            
            uint c = gDilateColorIn.Load(int3(sp, 0));
            if (c == 0u) continue;
            
            uint d = gDilateDepthSrv.Load(int3(sp, 0));
            if (d == 0xFFFFFFFFu) continue;
            
            uint ao = gDilateAoIn.Load(int3(sp, 0));
            
            
            

            float S    = (float)(1u << ((c >> 26) & 0xFu));   // LOD from colour bits
            float vzN  = (float)d / kLinDepthScale;

            // --- Slab-test path: reconstruct this neighbour's voxel cube and
            //     intersect THIS pixel's ray with it. Nearest tHit wins → exact
            //     watertight cubes (no footprint gaps). LOD baked in S (lod0=1).
            float2 uvN = (float2(sp) + 0.5) * gInvVwSize;
            float3 center = ro
                + gCamRight   * ((uvN.x * 2.0 - 1.0) * gAspect * gTanHalfFovY * vzN)
                + gCamUp      * ((1.0 - uvN.y * 2.0) * gTanHalfFovY * vzN)
                + gCamForward * vzN;
            float3 vmin = floor(center / S) * S;
            uint   mask = (c >> 20) & 0x3Fu;
            float  tHit; uint face;
            if (RayAabb(ro, rd, invRd, vmin, vmin + S, mask, tHit, face)
                && tHit > 1e-3 && tHit < bestT)
            {
                bestT = tHit; bestD = d; winSp = sp; winC = c; winAo = ao; found = true;
            }

            // --- Footprint coverage fallback: sub-pixel-size voxels never get a
            //     real ray-cube hit, and boundary pixels can miss; keep the nearest
            //     covered splat so those pixels still draw (shaded center+face).
            float half = DilateHalfPx(sp, S, vzN, gFocalPx, gInvVwSize);
            if (max(abs(dx), abs(dy)) <= half && d < fbD)
            {
                fbD = d; fbSp = sp; fbC = c; fbAo = ao; fbFound = true;
            }
        }
 
    }
   
    if (!found && fbFound)        // no slab hit → use footprint coverage winner
    {
        winSp = fbSp; winC = fbC; winAo = fbAo; bestD = fbD; found = true;
    }
    if (found)
    {
        float viewZ;
        gDilateColorOut[pix] = ShadeDilateWinner(winSp, winC, winAo, bestD,
                                                 ro, rd, invRd, gInvVwSize, viewZ);
        gDilateDepthOut[pix] = gNearZ / max(viewZ, 1e-4);
        return;
    }
    gDilateColorOut[pix] = 0;                              // true sky
    gDilateDepthOut[pix] = 0.0;
}

static const int pxPadMax = 3;                          // max supported radius
static const int groupSz   = 8;
static const int PadSz     = groupSz + pxPadMax + pxPadMax;   // 14 (fixed stride)

// Window cache: the dilate scans an (8+2R)x(8+2R) neighbourhood per output pixel.
// The whole 8x8 group only touches one such screen region, so cache its splat
// inputs in LDS once (3 loads/thread) instead of re-loading per scan cell. Cached
// per cell: depth, colour (albedo+visMask), AO (per-face AO + LOD) and the
// precomputed footprint half-size. Arrays sized for Rmax=3 (14x14=196); smaller R
// uses a centred sub-region at the fixed stride. Empty/off-screen cells store the
// d==0xFFFFFFFF sentinel so the scan's skip covers edges with no bounds test.
groupshared uint   gsDepth[PadSz * PadSz];
groupshared uint   gsColor[PadSz * PadSz];   // albedo (0-23) + visMask (24-29)
groupshared uint   gsAo   [PadSz * PadSz];   // per-face AO (0-23) + LOD (24-27)
groupshared float  gsHalf [PadSz * PadSz];   // projected half-size in px (footprint)

void DilateAt(int2 pix, int2 gtid)
{
    int R = max(gSplatRadius, 1);
   // if (R > pxPadMax)
        return DilateAtRef(pix);                // beyond cache size: scalar path

    int W = (int) gVwSize.x;
    int H = (int) gVwSize.y;

    // Fill the (8+2R)x(8+2R) LDS window (8x8 group + R px pad each side, stored at
    // fixed stride PadSz). Footprint is centred on the splat pixel (where the voxel
    // really projected), sized to the true voxel extent, so it can't reach sky.
    int  ext = groupSz + R + R;                 // window width this dispatch
    int2 winOrigin = (pix - gtid) - int2(R, R);
    int  tid = gtid.y * groupSz + gtid.x;       // 0..63
    [loop] for (int k = tid; k < ext * ext; k += groupSz * groupSz)
    {
        int  lx = k % ext, ly = k / ext;
        int2 sp = winOrigin + int2(lx, ly);
        int  slot = ly * PadSz + lx;            // fixed-stride LDS slot
        uint d = 0xFFFFFFFFu;
        uint c = 0u, ao = 0u;
        if (sp.x >= 0 && sp.x < W && sp.y >= 0 && sp.y < H)
        {
            d  = gDilateDepthSrv.Load(int3(sp, 0));
            c  = gDilateColorIn.Load(int3(sp, 0));
            ao = gDilateAoIn.Load(int3(sp, 0));
        }
        if (c == 0u) d = 0xFFFFFFFFu;           // no splat here → ignore in scan
        gsDepth[slot] = d;
        gsColor[slot] = c;
        gsAo[slot]    = ao;
        if (d != 0xFFFFFFFFu)
        {
            float S   = (float) (1u << ((c >> 26) & 0xFu));    // LOD from colour bits
            float vzN = (float) d / kLinDepthScale;
            gsHalf[slot] = DilateHalfPx(sp, S, vzN, gFocalPx, gInvVwSize);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Bounds return AFTER the fill+barrier so no thread skips the sync.
    if (pix.x >= W || pix.y >= H)
        return;

    // This pixel is coloured iff some neighbour voxel's projected footprint covers
    // it. No proximity fallback beyond the footprint, so a sky pixel no voxel
    // projects onto stays sky. Nearest depth wins (occlusion).
    int   winLidx = -1;
    int2  winSp   = int2(0, 0);
    uint  bestD   = 0xFFFFFFFFu;
  /*
    [loop]
    for (int dy = -R; dy <= R; ++dy)
    {
        [loop]
        for (int dx = -R; dx <= R; ++dx)
        {
            int  lidx = (gtid.y + R + dy) * PadSz + (gtid.x + R + dx);
            uint d = gsDepth[lidx];
            if (d == 0xFFFFFFFFu)
                continue;
            if (max(abs(dx), abs(dy)) <= gsHalf[lidx] && d < bestD)
            {
                bestD   = d;
                winLidx = lidx;
                winSp   = pix + int2(dx, dy);
            }
        }
    }
*/
    if (winLidx >= 0)                                      // covered by a voxel
    {
        float3 ro = gCamPos;
        float3 rd = PixelWorldDir(float2(pix) + 0.5, gInvVwSize);
        float  viewZ;
        gDilateColorOut[pix] = ShadeDilateWinner(winSp, gsColor[winLidx],
                                                 gsAo[winLidx], bestD,
                                                 ro, rd, 1.0 / rd, gInvVwSize, viewZ);
        gDilateDepthOut[pix] = gNearZ / max(viewZ, 1e-4);
        return;
    }
    gDilateColorOut[pix] = 0; // true sky
    gDilateDepthOut[pix] = 0.0;
}

// Linear dispatch: pixel = SV_DispatchThreadID.
[numthreads(8, 8, 1)]
void csmain_dilate(uint3 dt : SV_DispatchThreadID, uint3 gt : SV_GroupThreadID)
{
    DilateAt(int2(dt.xy), int2(gt.xy));
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
    DilateAt(int2(pix), int2(GTid.xy));
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
