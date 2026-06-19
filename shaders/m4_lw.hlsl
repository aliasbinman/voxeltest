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

        // Light the hit HERE (triplanar from camera→voxel-centre, visMask-gated,
        // baked AO, fog). The dilate then just spreads the finished colour/depth.
        uint   colPck = gLwPalette[ci.paletteBase + palIdx];
        float3 albedo = float3((colPck & 0xFFu), ((colPck >> 8) & 0xFFu),
                               ((colPck >> 16) & 0xFFu)) / 255.0;
        uint mask = (i < 4u) ? ((visPack.x >> (i * 8u)) & 0x3Fu)
                              : ((visPack.y >> ((i - 4u) * 8u)) & 0x3Fu);
        if (mask == 0u) mask = 0x3Fu;
        uint aoFace24 = gLwBlockAo[(blockBase + (gid - firstThread)) * 8u + i] & 0x00FFFFFFu;

        float3 centerW = world;
        float3 vdir = normalize(gCamPos - centerW);
        uint f0 = (vdir.x >= 0.0) ? 0u : 1u;
        uint f1 = (vdir.y >= 0.0) ? 2u : 3u;
        uint f2 = (vdir.z >= 0.0) ? 4u : 5u;
        float w0 = ((mask >> f0) & 1u) ? vdir.x * vdir.x : 0.0;
        float w1 = ((mask >> f1) & 1u) ? vdir.y * vdir.y : 0.0;
        float w2 = ((mask >> f2) & 1u) ? vdir.z * vdir.z : 0.0;
        // No camera-facing visible face → back-facing splat (e.g. underside of a
        // flat +Y-only surface seen grazing). Its normal would be normalize(0)=NaN
        // → black. Skip it; the dilate fills the pixel from valid neighbours.
        float ws = w0 + w1 + w2;
        if (ws <= 0.0) continue;
        w0 /= ws; w1 /= ws; w2 /= ws;
        float3 N0 = float3(sign(vdir.x), 0, 0);
        float3 N1 = float3(0, sign(vdir.y), 0);
        float3 N2 = float3(0, 0, sign(vdir.z));
        float3 Nshade = normalize(w0 * N0 + w1 * N1 + w2 * N2);
        float  NdotL  = w0 * max(0.0, dot(N0, gLightDir))
                      + w1 * max(0.0, dot(N1, gLightDir))
                      + w2 * max(0.0, dot(N2, gLightDir));
        uint   domFace = (w0 >= w1 && w0 >= w2) ? f0 : (w1 >= w2 ? f1 : f2);

        float aos = 0.0; uint aon = 0u;
        [unroll] for (uint f = 0u; f < 6u; ++f) {
            if (((mask >> f) & 1u) == 0u) continue;
            aos += (float)((aoFace24 >> (f * 4u)) & 0xFu) * (1.0 / 15.0); ++aon;
        }
        float ao = (aon > 0u) ? (aos / (float)aon) : 1.0;

        float3 lit;
        if ((int)gMode == 3) lit = float3(ao, ao, ao);
        else if ((int)gMode == 2) {
            const float3 kFaceColor[6] = {
                float3(1.0,0.2,0.2), float3(1.0,0.2,1.0), float3(0.2,1.0,0.2),
                float3(1.0,0.5,0.1), float3(0.2,0.4,1.0), float3(0.2,1.0,1.0) };
            lit = kFaceColor[domFace];
        } else {
            float3 sunCol = float3(1.0, 0.95, 0.85) * (gSunIntensity * 3.0);
            float3 amb    = AmbientCube(Nshade) * gAmbient;
            lit = albedo * (amb + NdotL * sunCol);
            lit *= lerp(1.0, ao, gAoStrength);
            lit = ApplyFog(lit, centerW);
        }
        uint R8 = (uint)clamp(lit.r * 255.0, 0.0, 255.0);
        uint G8 = (uint)clamp(lit.g * 255.0, 0.0, 255.0);
        uint B8 = (uint)clamp(lit.b * 255.0, 0.0, 255.0);
        // Alpha byte carries this voxel's LOD exponent L (lodScale == 1<<L) so the
        // dilate uses the TRUE voxel size instead of guessing it from depth — a
        // depth-guessed size can overshoot and bleed the cube's silhouette into
        // sky. Marker bit 7 keeps argb != 0 so resolve's sky test still works.
        uint L = firstbithigh((uint)ci.lodScale);   // lodScale is a power of two
        uint aTag = 0x80u | (L & 0x7Fu);
        gLwVisUav[pix] = (aTag << 24) | (B8 << 16) | (G8 << 8) | R8; // lit + LOD tag
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
void DilateAtRef(int2 pix)
{
    int W = (int)gVwSize.x;
    int H = (int)gVwSize.y;
    if (pix.x >= W || pix.y >= H) return;

    // A pixel is coloured iff some neighbour voxel's projected screen footprint
    // covers it: |dx|,|dy| within that voxel's projected half-size in px
    // (0.5 * S * focalPx / viewZ, S = true voxel world size from the LOD tag).
    // The footprint is centred on the splat pixel (where the voxel really
    // projected), so it can't extend past the voxel's true extent into sky.
    // Among covering voxels the nearest depth wins (occlusion).
    float  focalPx = gScreenSize.y * 0.5 / gTanHalfFovY;
    int    R = max(gSplatRadius, 1);

    uint winCol = 0; uint bestD = 0xFFFFFFFFu; bool found = false;
    [loop] for (int dy = -R; dy <= R; ++dy)
    [loop] for (int dx = -R; dx <= R; ++dx)
    {
        int2 sp = pix + int2(dx, dy);
        if (sp.x < 0 || sp.x >= W || sp.y < 0 || sp.y >= H) continue;
        uint c = gDilateColorIn.Load(int3(sp, 0));
        if (c == 0u) continue;
        uint d = gDilateDepthSrv.Load(int3(sp, 0));
        if (d == 0xFFFFFFFFu) continue;

        float vzN  = (float)d / kLinDepthScale;          // forward distance
        float S    = (float)(1u << ((c >> 24) & 0x7Fu)); // true voxel world size
        // +1px so adjacent footprints overlap (they'd only just touch at exactly
        // projPx spacing → integer-rounding seams, worst at diagonal corners).
        float half = 0.5 * S * focalPx / max(vzN, 1e-4) + 1.0;
        if (max(abs(dx), abs(dy)) <= half && d < bestD)
        {
            bestD = d; winCol = c; found = true;
        }
    }

    if (found)                                            // covered by a voxel
    {
        gDilateColorOut[pix] = winCol;
        gDilateDepthOut[pix] = gNearZ / max((float)bestD / kLinDepthScale, 1e-4);
        return;
    }
    gDilateColorOut[pix] = 0;                              // true sky
    gDilateDepthOut[pix] = 0.0;
}

static const int pxPadMax = 3;                          // max supported radius
static const int groupSz   = 8;
static const int PadSz     = groupSz + pxPadMax + pxPadMax;   // 14 (fixed stride)

// Window cache: the dilate scans an (8+2R)x(8+2R) neighbourhood per output
// pixel. The whole 8x8 group only touches one such screen region, so cache it
// once in LDS. The reconstructed voxel AABB (vmin + size S) for a cell depends
// ONLY on that cell's screen pos + depth — not on the output pixel — so it's
// identical for every one of the 64 threads that scan it. We therefore do the
// expensive reconstruction (log2/floor/divide) ONCE per window cell in the fill
// and cache the AABB; the inner loop then only runs the cheap RayAabb. Colour is
// NOT cached: the scan is pure geometry, so we resolve the single winning cell
// and do ONE texture Load for its (already-lit) colour at the end. Splats are
// co-located (pass1 depth + pass2 colour hit the same pixel), so valid depth
// implies a colour — except pass2's grazing back-face skip, guarded by the c!=0
// fallback. The arrays are sized for Rmax=3 (14x14=196); smaller R uses a
// centred sub-region at the same fixed stride. Off-screen / empty cells store
// the d==0xFFFFFFFF sentinel so the scan's skip covers edges with no bounds test.
groupshared uint   gsDepth[PadSz * PadSz];
groupshared uint   gsColor[PadSz * PadSz];   // packed lit colour (+ LOD tag in a)
groupshared float  gsHalf [PadSz * PadSz];   // projected half-size in px (footprint)

void DilateAt(int2 pix, int2 gtid)
{
    int R = max(gSplatRadius, 1);
    if (R > pxPadMax)
        return DilateAtRef(pix);                // beyond cache size: scalar path

    int W = (int) gVwSize.x;
    int H = (int) gVwSize.y;
    float focalPx = gScreenSize.y * 0.5 / gTanHalfFovY;

    // Fill the (8+2R)x(8+2R) LDS window (8x8 group + R px pad each side, stored at
    // fixed stride PadSz). For each splat cell, cache its colour, view depth and
    // its voxel's PROJECTED HALF-SIZE in pixels (0.5 * S * focalPx / viewZ). We do
    // NOT reconstruct a world cube: the splat is a single integer pixel, so the
    // voxel's sub-pixel world position is lost, and snapping a reconstructed point
    // to the LOD grid can land the cube in the wrong (sky-side) cell at distance —
    // that misplacement is what bled coarse voxels into the sky. The footprint is
    // centred on the splat pixel itself (where the voxel really projected), so it
    // can never extend past the voxel's true screen extent.
    int  ext = groupSz + R + R;                 // window width this dispatch
    int2 winOrigin = (pix - gtid) - int2(R, R);
    int  tid = gtid.y * groupSz + gtid.x;       // 0..63
    [loop] for (int k = tid; k < ext * ext; k += groupSz * groupSz)
    {
        int  lx = k % ext, ly = k / ext;
        int2 sp = winOrigin + int2(lx, ly);
        int  slot = ly * PadSz + lx;            // fixed-stride LDS slot
        uint d = 0xFFFFFFFFu;
        uint c = 0u;
        if (sp.x >= 0 && sp.x < W && sp.y >= 0 && sp.y < H)
        {
            d = gDilateDepthSrv.Load(int3(sp, 0));
            c = gDilateColorIn.Load(int3(sp, 0));
        }
        // A cell participates only if it has BOTH depth and colour. (Pass2 can
        // write depth but skip colour on a grazing back-face → c==0; drop those
        // by forcing the depth sentinel so the scan ignores them.)
        if (c == 0u) d = 0xFFFFFFFFu;
        gsDepth[slot] = d;
        gsColor[slot] = c;
        if (d != 0xFFFFFFFFu)
        {
            float vzN = (float) d / kLinDepthScale;       // forward distance
            uint  L   = (c >> 24) & 0x7Fu;                // LOD tag (lodScale = 1<<L)
            float S   = (float) (1u << L);                // true voxel world size
            // +1px so adjacent footprints overlap (they'd only just touch at exactly
            // projPx spacing → integer-rounding seams, worst at diagonal corners).
            gsHalf[slot] = 0.5 * S * focalPx / max(vzN, 1e-4) + 1.0;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Bounds return AFTER the fill+barrier so no thread skips the sync.
    if (pix.x >= W || pix.y >= H)
        return;

    // This pixel is coloured iff some neighbour voxel's projected footprint covers
    // it (|dx|,|dy| within that voxel's projected half-size). No proximity/nearest
    // fallback beyond the footprint, so a sky pixel no voxel projects onto stays
    // sky → tight silhouettes. Among covering voxels the nearest depth wins.
    int   winLidx = -1;
    uint  bestD   = 0xFFFFFFFFu;
    [loop]
    for (int dy = -R; dy <= R; ++dy)
    [loop]
        for (int dx = -R; dx <= R; ++dx)
        {
            // Local index = (group thread + pad + delta); off-screen cells carry
            // the d sentinel so this skip also covers screen edges.
            int  lidx = (gtid.y + R + dy) * PadSz + (gtid.x + R + dx);
            uint d = gsDepth[lidx];
            if (d == 0xFFFFFFFFu)
                continue;

            // Is this pixel within the neighbour voxel's projected footprint?
            float half = gsHalf[lidx];
            if (max(abs(dx), abs(dy)) <= half && d < bestD)
            {
                bestD   = d;
                winLidx = lidx;
            }
        }

    if (winLidx >= 0)                                      // covered by a voxel
    {
        gDilateColorOut[pix] = gsColor[winLidx];
        gDilateDepthOut[pix] = gNearZ / max((float) bestD / kLinDepthScale, 1e-4);
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
