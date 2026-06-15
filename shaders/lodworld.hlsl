// lodworld.hlsl — minimal point splat shaders for the .lw render path.
// Stays decoupled from voxel.hlsl so the legacy path keeps working. Output
// RTs match the existing splat pipeline (color RGBA8 with LOD+AO packed in
// alpha, mask R8_UINT with visMask), so the existing CS dilate / composite
// pass can be reused unchanged.

// ---- CBs ----
cbuffer CBLwFrame : register(b0)
{
    row_major float4x4 gViewProj;
    float3   gCamPos;
    float    gNearZ;
    float3   gLightDir;       // sun direction (world-space, normalized)
    float    gSunIntensity;
    float3   gAmbientColor;   // unused since shared lighting uses ambient cube
    float    gExposure;
    float3   gFogColor;
    float    gFogDensity;
    float    gHeightFogDensity;
    float    gHeightFogFalloff;
    float    gHeightFogStart;
    float    gShadowEnable;
    float    gShadowBias;
    float    gShadowMapSize;
    float    gColorizeClusters;
    float    gAmbient;        // sun multiplier in shared ApplyShadowLighting
    row_major float4x4 gSunViewProj;
    row_major float4x4 gInvViewProj; // for OctetBillboard PS pixel-ray unproject
    uint     gMode;           // ShadingMode: 0=Lit 1=FlatColor 2=Normals 3=Ao 4=LodViz
    uint3    _padFrame;
};
cbuffer CBLwLod : register(b1)
{
    uint  gLodIdx;            // 0..4 — encoded into splat alpha
    float gHalfExt;           // half-extent of voxel in world units (kLodScale * 0.5)
    uint  gLwSlot;            // chunk slot index for this draw
    uint  gLwDrawBase;        // offset added to SV_VertexID when fetching from pool
                              //  (= cluster.pointFirst when drawing a sub-span)
};

// Shared lighting (same as voxel.hlsl). Reads CBLwFrame uniforms.
#include "shading.hlsli"

// ---- SRVs (per-LOD; rebound when switching LODs) ----
struct LwPoint { uint pack0; uint pack1; };           // 8 bytes
struct LwChunkInfo {
    float3 worldOrigin;
    float  lodScale;
    uint   poolBase;
    uint   paletteBase;
    uint2  _padCI;
};

StructuredBuffer<LwPoint>     gLwPoints     : register(t0);
StructuredBuffer<LwChunkInfo> gLwChunkInfos : register(t1);
StructuredBuffer<uint>        gLwPalette    : register(t2);

// ---- Splat alpha encoding ----
//   bit 7    = marker
//   bits 6:4 = lodIdx (3 bits = 0..7 supports kLodCount=5)
//   bits 3:0 = AO 4-bit
//   (parity dropped to free bit for 3-bit LOD)
float EncodeSplatAlpha(float ao01, uint lodIdx, uint parity)
{
    uint ao4 = (uint)(saturate(ao01) * 15.0 + 0.5);
    uint dummy = parity;   // suppress unused-param warning; encoding drops parity
    uint a8  = 0x80u | ((lodIdx & 7u) << 4) | (ao4 & 0xFu);
    return (float)a8 / 255.0;
}

// ---- VS ----
struct VSOut {
    float4 svpos : SV_Position;
    float4 colAO   : COLOR0;
    nointerpolation uint mask  : COLOR2;
    nointerpolation uint parity : COLOR3;   // cluster checker: (cx+cy+cz) & 1
    float3 wpos  : TEXCOORD0;
};

VSOut vsmain_lw_points(uint vid : SV_VertexID)
{
    uint slot   = gLwSlot;
    uint vtxIdx = vid + gLwDrawBase;

    LwChunkInfo ci = gLwChunkInfos[slot];
    LwPoint     p  = gLwPoints[ci.poolBase + vtxIdx];

    uint px = (p.pack0 >>  0) & 0xFFu;
    uint py = (p.pack0 >>  8) & 0xFFu;
    uint pz = (p.pack0 >> 16) & 0xFFu;
    uint palIdx = (p.pack0 >> 24) & 0xFFu;

    uint mask    = (p.pack1 >> 0) & 0x3Fu;
    uint aoPck   = (p.pack1 >> 8);                  // 24 bits AO (6*4)

    // Voxel center in world units: chunk origin + (local + 0.5) * lodScale.
    float3 local = float3((float)px, (float)py, (float)pz) + 0.5;
    float3 world = ci.worldOrigin + local * ci.lodScale;

    // Cluster index within chunk: each cluster spans 32 LOD-voxels per axis.
    uint cx = px >> 5;   // px / 32
    uint cy = py >> 5;
    uint cz = pz >> 5;

    VSOut o;
    o.svpos  = mul(float4(world, 1.0), gViewProj);
    o.wpos   = world;
    // Pack visMask + all 6 face AOs + parity into mask channel (R32_UINT).
    // bits 0-5  : visMask
    // bits 6-29 : 6 face AOs × 4 bits (face 0 at bit 6, face 5 at bit 26)
    // bit 30    : cluster checker parity (free bit; alpha now uses 3-bit LOD)
    o.parity = (cx + cy + cz) & 1u;
    o.mask   = (mask & 0x3Fu) | (aoPck << 6u) | (o.parity << 30u);

    uint colPck = gLwPalette[ci.paletteBase + palIdx];
    o.colAO.r = ((float)  (colPck & 0xFFu)) / 255.0;
    o.colAO.g = ((float) ((colPck >> 8) & 0xFFu)) / 255.0;
    o.colAO.b = ((float) ((colPck >> 16) & 0xFFu)) / 255.0;
    

    // Splat AO = average of visible face AOs. Camera-direction face pick
    // would flicker when camera angle crosses an octant boundary; averaging
    // gives one stable value per voxel.
    float sumAo = 0.0;
    int   cntAo = 0;
    [unroll] for (uint f = 0u; f < 6u; ++f) {
        if (((mask >> f) & 1u) == 0u) continue;
        sumAo += (float)((aoPck >> (f * 4u)) & 0xFu);
        ++cntAo;
    }
    o.colAO.a = (cntAo > 0) ? (sumAo / (15.0 * (float) cntAo)) : 1.0;
    return o;
}

// ---- PolyAxis: 6 quads per voxel (36 verts). Per-face AO direct. ----
//   vid layout: vid/36 = voxel index in chunk, (vid%36)/6 = face, vid%6 = vert
//   Hidden faces emit NaN positions (discarded by raster).
static const float3 kFaceVerts[6][6] = {
    // +X face (normal +X): vertices on x=1 plane
    { float3(1,0,0), float3(1,0,1), float3(1,1,0), float3(1,1,0), float3(1,0,1), float3(1,1,1) },
    // -X face (x=0)
    { float3(0,0,0), float3(0,1,0), float3(0,0,1), float3(0,0,1), float3(0,1,0), float3(0,1,1) },
    // +Y face (y=1)
    { float3(0,1,0), float3(1,1,0), float3(0,1,1), float3(0,1,1), float3(1,1,0), float3(1,1,1) },
    // -Y face (y=0)
    { float3(0,0,0), float3(0,0,1), float3(1,0,0), float3(1,0,0), float3(0,0,1), float3(1,0,1) },
    // +Z face (z=1)
    { float3(0,0,1), float3(0,1,1), float3(1,0,1), float3(1,0,1), float3(0,1,1), float3(1,1,1) },
    // -Z face (z=0)
    { float3(0,0,0), float3(1,0,0), float3(0,1,0), float3(0,1,0), float3(1,0,0), float3(1,1,0) },
};

VSOut vsmain_lw_polyaxis(uint vid : SV_VertexID)
{
    uint voxelIdx = vid / 36u;
    uint face     = (vid % 36u) / 6u;
    uint vertIn   = vid % 6u;

    uint slot = gLwSlot;
    LwChunkInfo ci = gLwChunkInfos[slot];
    LwPoint     p  = gLwPoints[ci.poolBase + voxelIdx + gLwDrawBase];

    uint px = (p.pack0 >>  0) & 0xFFu;
    uint py = (p.pack0 >>  8) & 0xFFu;
    uint pz = (p.pack0 >> 16) & 0xFFu;
    uint palIdx = (p.pack0 >> 24) & 0xFFu;
    uint mask   = (p.pack1 >>  0) & 0x3Fu;
    uint aoPck = (p.pack1 >> 8);

    VSOut o;
    if (((mask >> face) & 1u) == 0u) {
        // Hidden face: emit clip pos with w=0 → guaranteed clip-discarded.
        o.svpos = float4(0, 0, 0, 0);
        o.colAO.rgb = float3(0,0,0); 
        o.colAO.a = 0; 
        o.mask = 0; 
        o.parity = 0; 
        o.wpos = float3(0,0,0);
        return o;
    }

    float3 voxOrigin = float3((float)px, (float)py, (float)pz);
    float3 corner    = voxOrigin + kFaceVerts[face][vertIn];
    float3 world     = ci.worldOrigin + corner * ci.lodScale;
    o.svpos = mul(float4(world, 1.0), gViewProj);
    
    
    o.wpos  = world;
    o.mask  = mask;
    uint cx = px >> 5;
    uint cy = py >> 5;
    uint cz = pz >> 5;
    o.parity = (cx + cy + cz) & 1u;

    uint colPck = gLwPalette[ci.paletteBase + palIdx];
    o.colAO.rgb = float3((float)( colPck         & 0xFFu),
                   (float)((colPck >>  8u) & 0xFFu),
                   (float)((colPck >> 16u) & 0xFFu)) / 255.0;

    uint nib = (aoPck >> (face * 4)) & 0xF;
    o.colAO.a = (float) nib / 15.0;
    return o;
}

// ---- PS: splat albedo (mask output dropped) ----
float4 psmain_lw_splat_albedo(VSOut i) : SV_Target
{
    return float4(i.colAO.rgb, EncodeSplatAlpha(i.colAO.a, gLodIdx, i.parity));
}

// ---- PS: opaque debug (no splat encoding — for "just see voxels" sanity) ----
float4 psmain_lw_debug(VSOut i) : SV_Target
{
    return float4(i.colAO.rgb, 1.0);
}

// ---- PS: PolyAxis (post-dilate, direct to scene RT) ----
// Face normal from ddx/ddy of wpos. All lighting via ShadeWithLighting
// (shaders/shading.hlsli) so PolyAxis + splat CS produce identical look:
// ambient cube * AO, sun N·L with shadow PCF, depth+height fog, ACES tonemap.
float4 psmain_lw_polyaxis_lit(VSOut i) : SV_Target
{
    //return float4(1, 0, 1, 1);
    
    float3 N = normalize(cross(ddx(i.wpos), ddy(i.wpos)));
    float3 toCam = normalize(gCamPos - i.wpos);
    if (dot(N, toCam) < 0.0) N = -N;
    float3 outRgb = ShadeWithLighting(i.colAO.rgb, N, i.wpos, i.colAO.a,
                                      gLodIdx, i.parity, (int)gMode);
    return float4(outRgb, 1.0);
}

// ---- PS: AO + per-LOD tint + per-cluster checker ----
float4 psmain_lw_lodviz(VSOut i) : SV_Target
{
    // Distinct hue per LOD level.
    static const float3 kLodTints[5] = {
        float3(1.00, 0.40, 0.40),   // L0 red
        float3(1.00, 0.80, 0.30),   // L1 orange
        float3(0.40, 1.00, 0.40),   // L2 green
        float3(0.40, 0.70, 1.00),   // L3 blue
        float3(0.90, 0.40, 1.00),   // L4 magenta
    };
    float3 tint = kLodTints[min(gLodIdx, 4u)];
    // Checker: parity 0 = dark, parity 1 = light.
    float check = (i.parity == 0u) ? 0.55 : 1.00;
    float ao = lerp(0.45, 1.0, i.colAO.a);     // keep some floor so dark voxels visible
    return float4(tint * check * ao, 1.0);
}

// ============================================================
// Chunk AABB wireframe — one DrawInstanced call per LOD.
// 24 verts (12 edges × 2 corners), N instances (= slot count).
// SV_InstanceID picks chunk; LOD passed via CBLwBounds.
// ============================================================
cbuffer CBLwBounds : register(b2)
{
    float3 gBoundsColor;
    float  _padBoundsA;
    float3 gChunkDim;        // (kChunkVoxX, kChunkVoxY, kChunkVoxZ) in LOD-voxel units
    float  _padBoundsB;
};

// Box-corner index in bit 0=X, bit 1=Y, bit 2=Z. 12 edges = 24 vert indices.
static const uint kBoundsEdges[24] = {
    0,1, 1,3, 3,2, 2,0,
    4,5, 5,7, 7,6, 6,4,
    0,4, 1,5, 2,6, 3,7,
};

struct VSBoundsOut {
    float4 svpos : SV_Position;
    float3 col   : COLOR0;
};

VSBoundsOut vsmain_lw_bounds(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    LwChunkInfo ci = gLwChunkInfos[iid];
    float3 mn = ci.worldOrigin;
    float3 mx = ci.worldOrigin + gChunkDim * ci.lodScale;
    uint corner = kBoundsEdges[vid];
    float3 w;
    w.x = ((corner >> 0) & 1u) ? mx.x : mn.x;
    w.y = ((corner >> 1) & 1u) ? mx.y : mn.y;
    w.z = ((corner >> 2) & 1u) ? mx.z : mn.z;

    VSBoundsOut o;
    o.svpos = mul(float4(w, 1.0), gViewProj);
    o.col = gBoundsColor;
    return o;
}

float4 psmain_lw_bounds(VSBoundsOut i) : SV_Target
{
    return float4(i.col, 1.0);
}

// ============================================================
// Compute rasterizer (Schütz-style atomic min).
// Packed payload: (invDepth16 << 16) | rgb565. Lowest packed = closest.
// 0xFFFFFFFFu = sky/no-point (clear value).
// Per-chunk dispatch: 1 thread per voxel.
// ============================================================
cbuffer CBLwCS : register(b3)
{
    uint2 gVwSize;             // viewport pixels
    uint  gLwPointCount;       // worklist: total threads (sum of item counts)
    uint  gLwNumWorkItems;     // worklist: number of items in gLwWorkItems
    float gLodFadeStart;       // world distance — fade to parent starts
    float gLodFadeEnd;         // world distance — fully parent (LOD about to be replaced)
    uint  gLwLodIdx;           // LOD index of the current dispatch (used by LodViz)
    uint  _padCs;
};

RWTexture2D<uint>         gLwVisUav      : register(u0);
// PointCS_Block → splat path UAVs (used by csmain_lw_block_splat_worklist).
// Bound at u1..u3 so they don't alias the u0 used by other pass1/pass2 shaders.
RWTexture2D<float4> gLwSplatColorUav : register(u1); // RGBA8 albedo + alpha = EncodeSplatAlpha
RWTexture2D<uint>   gLwSplatMaskUav  : register(u2); // R8_UINT 6-bit visMask per pixel
RWTexture2D<float>  gLwSplatDepthUav : register(u3); // reverse-Z (0..1)
// Split SoA blocks:
//   gLwBlockPos at t3: 4B/block — pack0 = bx|by|bz|occ (low..high bytes)
//   gLwBlockCol at t4: 8B/block — pack1 = palIdx[0..3], pack2 = palIdx[4..7]
StructuredBuffer<uint>  gLwBlockPos : register(t3);
StructuredBuffer<uint2> gLwBlockCol : register(t4);

// Two-pass: pass1 writes 32-bit depth via UAV; pass2 reads it as SRV at t5.
Texture2D<uint> gLwDepthSrv : register(t5);

// Worklist for single-dispatch pass1. Sorted ascending by .w (firstThread).
//   x = slot, y = blockBaseGlobal, z = count, w = firstThread (cumulative).
StructuredBuffer<uint4> gLwWorkItems : register(t6);

// Per-block visMask pool — 8 bytes per block (one 6-bit mask per voxel slot).
StructuredBuffer<uint2> gLwBlockVis : register(t7);

// ---- Cheap top-down AO map ----
// R32_UINT atomic-max of voxel Y per (X, Z) texel. UV maps the scene X-Z AABB
// 1:1 to the texture extents.
// AO build UAVs (CB + SRVs come from shading.hlsli).
RWTexture2D<uint>        gAoTopDownUav : register(u1);
RWTexture2D<unorm float> gAoOcclUav    : register(u2);

uint AoEncodeY(float worldY)
{
    return (uint)max(0.0, (worldY - gAoYMin) * gAoYScale);
}

// Normal-less wrapper for paths without per-voxel normal (block color RGB565,
// HW point non-splat PS). Just samples in-place.
float AoSample(float3 world)
{
    return AoSampleAt(world, float2(0, 0));
}

// ---- Linear-depth encoding for the atomic-min visibility buffer ----
// Linear view-Z mapped to uint: 0 at near plane, kLinDepthMax at 100km.
// Uniform precision (~23.3 µm per uint) → constant tolerance in pass2 works
// at any depth. Smaller uint = closer → atomic-min picks nearest directly.
// Sky / unwritten = 0xFFFFFFFFu sentinel (set by clear).
static const float kLinDepthFar  = 100000.0;
static const float kLinDepthMaxF = 4294967294.0; // 0xFFFFFFFE
static const float kLinDepthScale = kLinDepthMaxF / kLinDepthFar;
uint EncodeLinDepth(float viewZ)
{
    return (uint)clamp(viewZ * kLinDepthScale, 0.0, kLinDepthMaxF);
}

// Resolve PS bindings (two-pass): t3 = depth (R32_UINT), t4 = color (R32_UINT ARGB).
Texture2D<uint> gLwTpDepthSrv : register(t3);
Texture2D<uint> gLwTpColorSrv : register(t4);

struct VLwResolveOut { float4 pos : SV_Position; };
VLwResolveOut vsmain_lw_resolve(uint vid : SV_VertexID)
{
    float2 p = float2((vid == 1u) ? 3.0 : -1.0,
                      (vid == 2u) ? 3.0 : -1.0);
    VLwResolveOut o;
    o.pos = float4(p, 0.0, 1.0);
    return o;
}

float4 psmain_lw_resolve_twopass(VLwResolveOut i) : SV_Target
{
    int2 pix = int2(i.pos.xy);
    uint depth = gLwTpDepthSrv.Load(int3(pix, 0));
    if (depth == 0xFFFFFFFFu) discard;
    uint argb = gLwTpColorSrv.Load(int3(pix, 0));
    float r = (float)( argb        & 0xFFu) / 255.0;
    float g = (float)((argb >>  8) & 0xFFu) / 255.0;
    float b = (float)((argb >> 16) & 0xFFu) / 255.0;
    return float4(r, g, b, 1.0);
}


// ============================================================
// Single-dispatch worklist pass1. Caller submits ONE Dispatch covering all
// LOD items at once. gid -> item via binary search over firstThread. Avoids
// per-item UAV barriers and CB Map/Unmap stalls.
// ============================================================
[numthreads(64, 1, 1)]
void csmain_lw_block_depth_worklist(uint3 dt : SV_DispatchThreadID)
{
    uint gid = dt.x;
    if (gid >= gLwPointCount) return;

  //  gLwVisUav[int2(gid & 1023, (gid >> 10) & 1023)] = 0;
  //  return;
    
    
    // Binary search: find largest i where gLwWorkItems[i].w <= gid.
    uint lo = 0u;
    uint hi = gLwNumWorkItems; // exclusive
    while (lo + 1u < hi) {
        uint mid = (lo + hi) >> 1u;
        if (gLwWorkItems[mid].w <= gid) lo = mid;
        else                            hi = mid;
    }
    uint4 item = gLwWorkItems[lo];
    uint slot         = item.x;
    uint blockBase    = item.y;
    uint count        = item.z;
    uint firstThread  = item.w;
    if (gid - firstThread >= count) return; // padding tail

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
        
    //[unroll] for (uint j = 0; j < 8; ++j) {
    //    uint i = firstbitlow(occ);
        occ = occ & ~(1u << i); // clear lowest set bit for next iteration
        
        uint lx = (i >> 0) & 1u;
        uint ly = (i >> 1) & 1u;
        uint lz = (i >> 2) & 1u;
        float3 local = float3((float)(bx * 2u + lx), (float)(by * 2u + ly), (float)(bz * 2u + lz)) + 0.5;
        float3 world = ci.worldOrigin + local * ci.lodScale;

        float4 clip = mul(float4(world, 1.0), gViewProj);
        if (clip.w <= 0.0) continue;
        float3 ndc = clip.xyz / clip.w;
        if (ndc.x < -1.0 || ndc.x > 1.0 ||
            ndc.y < -1.0 || ndc.y > 1.0 ||
            ndc.z <  0.0 || ndc.z > 1.0) continue;
        int2 pix;
        pix.x = (int)((ndc.x * 0.5 + 0.5) * (float)gVwSize.x);
        pix.y = (int)((-ndc.y * 0.5 + 0.5) * (float)gVwSize.y);
        if (pix.x < 0 || pix.x >= (int)gVwSize.x ||
            pix.y < 0 || pix.y >= (int)gVwSize.y) continue;

        uint linD = EncodeLinDepth(clip.w);
        InterlockedMin(gLwVisUav[pix], linD);
    }
}

// Single-dispatch worklist pass2 colour. Same gid→item lookup as depth_worklist.
[numthreads(64, 1, 1)]
void csmain_lw_block_color_worklist(uint3 dt : SV_DispatchThreadID)
{
    uint gid = dt.x;
    if (gid >= gLwPointCount) return;

 //   gLwVisUav[int2(gid & 1023, (gid >> 10) & 1023)] = 0x00ff0000;
 //   return;
    
    uint lo = 0u, hi = gLwNumWorkItems;
    while (lo + 1u < hi) {
        uint mid = (lo + hi) >> 1u;
        if (gLwWorkItems[mid].w <= gid) lo = mid;
        else                            hi = mid;
    }
    uint4 item = gLwWorkItems[lo];
    uint slot        = item.x;
    uint blockBase   = item.y;
    uint count       = item.z;
    uint firstThread = item.w;
    if (gid - firstThread >= count) return;

    LwChunkInfo ci = gLwChunkInfos[slot];
    uint  pack0 = gLwBlockPos[blockBase + (gid - firstThread)];
    uint2 cols  = gLwBlockCol[blockBase + (gid - firstThread)];

    uint bx  = (pack0 >>  0) & 0xFFu;
    uint by  = (pack0 >>  8) & 0xFFu;
    uint bz  = (pack0 >> 16) & 0xFFu;
    uint occ = (pack0 >> 24) & 0xFFu;

    [unroll] for (uint i = 0; i < 8; ++i) {
        if (((occ >> i) & 1u) == 0u) continue;
        
        
        
        uint palIdx = (i < 4u) ? ((cols.x >> (i * 8u)) & 0xFFu)
                                : ((cols.y >> ((i - 4u) * 8u)) & 0xFFu);
        uint lx = (i >> 0) & 1u;
        uint ly = (i >> 1) & 1u;
        uint lz = (i >> 2) & 1u;
        float3 local = float3((float)(bx * 2u + lx), (float)(by * 2u + ly), (float)(bz * 2u + lz)) + 0.5;
        float3 world = ci.worldOrigin + local * ci.lodScale;

        float4 clip = mul(float4(world, 1.0), gViewProj);
        if (clip.w <= 0.0) continue;
        float3 ndc = clip.xyz / clip.w;
        if (ndc.x < -1.0 || ndc.x > 1.0 ||
            ndc.y < -1.0 || ndc.y > 1.0 ||
            ndc.z <  0.0 || ndc.z > 1.0) continue;
        int2 pix;
        pix.x = (int)((ndc.x * 0.5 + 0.5) * (float)gVwSize.x);
        pix.y = (int)((-ndc.y * 0.5 + 0.5) * (float)gVwSize.y);
        if (pix.x < 0 || pix.x >= (int)gVwSize.x ||
            pix.y < 0 || pix.y >= (int)gVwSize.y) continue;

        uint myLin    = EncodeLinDepth(clip.w);
        uint winDepth = gLwDepthSrv.Load(int3(pix, 0));
        const uint kLinDepthSlop = 64u; // ~1.5 mm tolerance for FP non-det
        if (myLin > winDepth + kLinDepthSlop) continue;

        uint argb32;
        if (gMode == 4u) {
            static const float3 kLodTints[5] = {
                float3(1.00, 0.40, 0.40),
                float3(1.00, 0.80, 0.30),
                float3(0.40, 1.00, 0.40),
                float3(0.40, 0.70, 1.00),
                float3(0.90, 0.40, 1.00),
            };
            float3 tint = kLodTints[min(gLwLodIdx, 4u)];
            uint cx = bx >> 4u;
            uint cy = by >> 4u;
            uint cz = bz >> 4u;
            uint parity = (cx + cy + cz) & 1u;
            float check = (parity == 0u) ? 0.55 : 1.0;
            float3 col = saturate(tint * check);
            uint rR = (uint)(col.r * 255.0);
            uint rG = (uint)(col.g * 255.0);
            uint rB = (uint)(col.b * 255.0);
            argb32 = 0xFF000000u | (rB << 16) | (rG << 8) | rR;
        } else {
            uint colPck = gLwPalette[ci.paletteBase + palIdx];
            float ao = AoSample(world);
            uint rR = (uint)((float)((colPck >>  0) & 0xFFu) * ao);
            uint rG = (uint)((float)((colPck >>  8) & 0xFFu) * ao);
            uint rB = (uint)((float)((colPck >> 16) & 0xFFu) * ao);
            argb32 = 0xFF000000u | (rB << 16) | (rG << 8) | rR;
        }
        gLwVisUav[pix] = argb32;
    }
}

// ============================================================
// PointCS_Block pass2 → splat-format output (for csSplat_ dilation/lighting).
// Same gid→item lookup. Writes RGBA8 colour, R32 mask with synthetic visMask
// (0x3F = all faces visible) + per-face AO=15 + cluster parity, and reverse-Z
// linear depth — formats that the existing splatCS dilate shader consumes.
// ============================================================
[numthreads(64, 1, 1)]
void csmain_lw_block_splat_worklist(uint3 dt : SV_DispatchThreadID)
{
    uint gid = dt.x;
    if (gid >= gLwPointCount) return;

    uint lo = 0u, hi = gLwNumWorkItems;
    while (lo + 1u < hi) {
        uint mid = (lo + hi) >> 1u;
        if (gLwWorkItems[mid].w <= gid) lo = mid;
        else                            hi = mid;
    }
    uint4 item = gLwWorkItems[lo];
    uint slot        = item.x;
    uint blockBase   = item.y;
    uint count       = item.z;
    uint firstThread = item.w;
    if (gid - firstThread >= count) return;

    LwChunkInfo ci = gLwChunkInfos[slot];
    uint  pack0 = gLwBlockPos[blockBase + (gid - firstThread)];
    uint2 cols  = gLwBlockCol[blockBase + (gid - firstThread)];
    uint2 vms   = gLwBlockVis[blockBase + (gid - firstThread)];

    uint bx  = (pack0 >>  0) & 0xFFu;
    uint by  = (pack0 >>  8) & 0xFFu;
    uint bz  = (pack0 >> 16) & 0xFFu;
    uint occ = (pack0 >> 24) & 0xFFu;

    [unroll] for (uint i = 0; i < 8; ++i) {
        if (((occ >> i) & 1u) == 0u) continue;
        uint palIdx = (i < 4u) ? ((cols.x >> (i * 8u)) & 0xFFu)
                                : ((cols.y >> ((i - 4u) * 8u)) & 0xFFu);
        uint visMask = (i < 4u) ? ((vms.x >> (i * 8u)) & 0x3Fu)
                                  : ((vms.y >> ((i - 4u) * 8u)) & 0x3Fu);
        uint lx = (i >> 0) & 1u;
        uint ly = (i >> 1) & 1u;
        uint lz = (i >> 2) & 1u;
        uint vx = bx * 2u + lx;
        uint vy = by * 2u + ly;
        uint vz = bz * 2u + lz;
        float3 local = float3((float)vx, (float)vy, (float)vz) + 0.5;
        float3 world = ci.worldOrigin + local * ci.lodScale;

        float4 clip = mul(float4(world, 1.0), gViewProj);
        if (clip.w <= 0.0) continue;
        float3 ndc = clip.xyz / clip.w;
        if (ndc.x < -1.0 || ndc.x > 1.0 ||
            ndc.y < -1.0 || ndc.y > 1.0 ||
            ndc.z <  0.0 || ndc.z > 1.0) continue;
        int2 pix;
        pix.x = (int)((ndc.x * 0.5 + 0.5) * (float)gVwSize.x);
        pix.y = (int)((-ndc.y * 0.5 + 0.5) * (float)gVwSize.y);
        if (pix.x < 0 || pix.x >= (int)gVwSize.x ||
            pix.y < 0 || pix.y >= (int)gVwSize.y) continue;

        // Linear-depth winner gate. Slop tuned for FP non-determinism between
        // pass1 + pass2 shaders (~few uint units), well under the minimum
        // gap between any two real voxels at the same pixel.
        uint myLin    = EncodeLinDepth(clip.w);
        uint winDepth = gLwDepthSrv.Load(int3(pix, 0));
        const uint kLinDepthSlop = 64u; // ~1.5 mm
        if (myLin > winDepth + kLinDepthSlop) continue;

        // Albedo from palette.
        uint colPck = gLwPalette[ci.paletteBase + palIdx];
        float r = (float)( colPck         & 0xFFu) / 255.0;
        float g = (float)((colPck >>  8u) & 0xFFu) / 255.0;
        float b = (float)((colPck >> 16u) & 0xFFu) / 255.0;

        // AO applied later in csSplat dilate (it has reconstructed normals
        // to push the AO lookup into the open neighbour column).

        // Cluster parity (32 voxels per cluster = vx>>5 etc).
        uint cx = vx >> 5u;
        uint cy = vy >> 5u;
        uint cz = vz >> 5u;
        uint parity = (cx + cy + cz) & 1u;

        // Splat alpha encoding (mirror of EncodeSplatAlpha in lodworld.hlsl VS).
        // bit 7 marker, bits 6:4 lodIdx (3 bits), bits 3:0 AO (4 bits, 15=full).
        uint a8 = 0x80u | ((gLwLodIdx & 7u) << 4) | 0xFu;
        float alpha = (float)a8 / 255.0;

        gLwSplatColorUav[pix] = float4(r, g, b, alpha);
        gLwSplatDepthUav[pix] = saturate(ndc.z); // reverse-Z (near=1, far=0)
        gLwSplatMaskUav[pix]  = visMask;
    }
}



// ============================================================
// AO top-down build. One thread per block (worklist dispatch). For each
// occupied voxel, project worldXZ → texel and InterlockedMax(worldY → uint).
// ============================================================
[numthreads(64, 1, 1)]
void csmain_ao_build_topdown(uint3 dt : SV_DispatchThreadID)
{
    uint gid = dt.x;
    if (gid >= gLwPointCount) return;

    uint lo = 0u, hi = gLwNumWorkItems;
    while (lo + 1u < hi) {
        uint mid = (lo + hi) >> 1u;
        if (gLwWorkItems[mid].w <= gid) lo = mid;
        else                            hi = mid;
    }
    uint4 item = gLwWorkItems[lo];
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

    [unroll] for (uint i = 0; i < 8; ++i) {
        if (((occ >> i) & 1u) == 0u) continue;
        uint lx = (i >> 0) & 1u;
        uint ly = (i >> 1) & 1u;
        uint lz = (i >> 2) & 1u;
        float3 local = float3((float)(bx * 2u + lx),
                              (float)(by * 2u + ly),
                              (float)(bz * 2u + lz)) + 0.5;
        float3 world = ci.worldOrigin + local * ci.lodScale;
        float2 uv = (world.xz - gAoOriginXZ) * gAoInvSizeXZ;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) continue;
        int2 px = int2(uv * gAoTexSizeF);
        InterlockedMax(gAoTopDownUav[px], AoEncodeY(world.y));
    }
}

// ============================================================
// HBAO sweep on the top-down depth map. For each texel, walk 8 directions
// out to log-spaced radii, record max horizon angle, integrate visible
// hemisphere. Output 0..1 occlusion (1 = fully open).
// ============================================================
[numthreads(8, 8, 1)]
void csmain_ao_hbao_filter(uint3 dt : SV_DispatchThreadID)
{
    uint texSize = (uint)gAoTexSizeF;
    if (dt.x >= texSize || dt.y >= texSize) return;
    int2 baseT = (int2)dt.xy;
    float myY = AoDecodeY(gAoTopDownSrv.Load(int3(baseT, 0)));

    const int kDirs = 8;
    const int kSteps = 8;
    // Log-spaced texel radii: covers ~1..128 texels (~world units near building scale).
    const int kRadii[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };

    float aoSum = 0.0;
    [unroll] for (int d = 0; d < kDirs; ++d) {
        float ang = (float)d * (6.2831853 / (float)kDirs);
        float2 dir = float2(cos(ang), sin(ang));
        float maxSin = 0.0;
        [unroll] for (int s = 0; s < kSteps; ++s) {
            float r = (float)kRadii[s];
            int2 t = baseT + int2(dir * r);
            t = clamp(t, int2(0,0), int2((int)texSize - 1, (int)texSize - 1));
            float h = AoDecodeY(gAoTopDownSrv.Load(int3(t, 0)));
            float dh   = h - myY;
            float dist = r * gAoWorldPerTexel;
            float sH   = dh / max(sqrt(dh * dh + dist * dist), 1e-4);
            maxSin = max(maxSin, sH);
        }
        // Visible hemisphere fraction along this direction.
        aoSum += saturate(1.0 - max(0.0, maxSin));
    }
    float ao = aoSum / (float)kDirs;
    gAoOcclUav[baseT] = ao;
}

// ============================================================
// OctetBillboards — per-octet billboard quad. VS projects 8 corners of the
// octet's world AABB to NDC, computes screen-space bbox, emits 6 verts as
// triangle list. PS does ray-vs-AABB on the 8 child voxels (gated by
// occupancy), picks nearest hit, shades via ShadeWithLighting6Face.
//
// Inputs: same worklist as splat path (gLwWorkItems at t6). VS uses
// SV_VertexID / 6 → block index in worklist, then unpacks BlockPos/Col/Vis.
// Output: scene RT (linear HDR) + SV_Depth via SV_Depth from ray hit.
// ============================================================
struct VSOutOctet
{
    float4 svpos     : SV_Position;
    nointerpolation float3 amin   : AMIN;   // octet world AABB min
    nointerpolation float3 amax   : AMAX;   // world AABB max
    nointerpolation uint   occ    : OCC;
    nointerpolation uint2  cols   : COLS;
    nointerpolation uint2  vms    : VMS;
    nointerpolation float  lodScale   : LODSC;
    nointerpolation uint   lodIdxV    : LODIDX;
    nointerpolation uint   paletteBase: PALBASE;
    nointerpolation uint3  baseVx     : BVX;
};

VSOutOctet vsmain_octet_billboard(uint vid : SV_VertexID)
{
    // 6 verts per quad (triangle list, two tris).
    uint blockIdx  = vid / 6u;
    uint cornerIdx = vid % 6u;

        // === DEBUG: bypass ALL VS math/buffer reads — emit a tiny fixed quad far in
    // the frustum. Rules out /0 / NaN in projection producing a degenerate or
    // infinite primitive (a known GPU-hang cause). occ=0 → PS discards.
    {
        float2 dbgMin = float2(0.40, 0.40);
        float2 dbgMax = float2(0.45, 0.45);
        float2 c = float2((vid & 1u) ? dbgMax.x : dbgMin.x,
                          (vid & 2u) ? dbgMax.y : dbgMin.y);
        o.pos = float4(c, 0.01, 1.0); // far depth (reverse-Z small)
        o.octMin = 0.0;
        o.vsize = 1.0;
        o.occ = 0u;
        o.blockIdx = 0u;
        o.palBase = 0u;
        return o;
    }
    
    
    // Mapping cornerIdx → quad corner index (0..3): 0,1,2, 0,2,3.
    // 0=(mnx,mny) 1=(mxx,mny) 2=(mxx,mxy) 3=(mnx,mxy).
    static const uint kCornerMap[6] = {0u, 1u, 2u, 0u, 2u, 3u};
    uint corner = kCornerMap[cornerIdx];

    // Locate the item this block lives in via binary search over firstThread.
    uint lo = 0u, hi = gLwNumWorkItems;
    while (lo + 1u < hi) {
        uint mid = (lo + hi) >> 1u;
        if (gLwWorkItems[mid].w <= blockIdx) lo = mid;
        else                                  hi = mid;
    }
    uint4 item = gLwWorkItems[lo];
    uint slot        = item.x;
    uint blockBase   = item.y;
    uint count       = item.z;
    uint firstThread = item.w;
    bool pad = (blockIdx - firstThread >= count);

    LwChunkInfo ci = gLwChunkInfos[slot];
    uint  pack0 = gLwBlockPos[blockBase + (blockIdx - firstThread)];
    uint2 cols  = gLwBlockCol[blockBase + (blockIdx - firstThread)];
    uint2 vms   = gLwBlockVis[blockBase + (blockIdx - firstThread)];

    uint bx  = (pack0 >>  0) & 0xFFu;
    uint by  = (pack0 >>  8) & 0xFFu;
    uint bz  = (pack0 >> 16) & 0xFFu;
    uint occ = (pack0 >> 24) & 0xFFu;

    // Octet AABB in world space: covers voxels [(bx*2,by*2,bz*2)..(bx*2+2,...)] * lodScale.
    float3 amin = ci.worldOrigin + float3(bx*2u, by*2u, bz*2u) * ci.lodScale;
    float3 amax = amin + (2.0 * ci.lodScale);

    // Project 8 corners to NDC, take min/max of valid (clip.w>0) projections.
    // Track max ndc.z (= nearest in reverse-Z) for the quad's depth so the
    // depth test pre-rejects occluded octets. PS refines per-pixel via SV_Depth.
    float mnx =  1e9, mny =  1e9;
    float mxx = -1e9, mxy = -1e9;
    float maxNdcZ = 0.0; // reverse-Z far baseline
    bool anyInFront = false;
    [unroll] for (uint ci2 = 0; ci2 < 8u; ++ci2) {
        float3 w = float3(
            (ci2 & 1u) ? amax.x : amin.x,
            (ci2 & 2u) ? amax.y : amin.y,
            (ci2 & 4u) ? amax.z : amin.z);
        float4 c = mul(float4(w, 1.0), gViewProj);
        if (c.w <= 0.001) continue;
        anyInFront = true;
        float3 ndc = c.xyz / c.w;
        mnx = min(mnx, ndc.x); mxx = max(mxx, ndc.x);
        mny = min(mny, ndc.y); mxy = max(mxy, ndc.y);
        maxNdcZ = max(maxNdcZ, ndc.z);
    }

    // Clip to [-1,1] (cheaper than guard band).
    mnx = max(mnx, -1.0); mxx = min(mxx, 1.0);
    mny = max(mny, -1.0); mxy = min(mxy, 1.0);

    float2 qpos;
    [branch] if (pad || !anyInFront || mnx >= mxx || mny >= mxy) {
        // Degenerate quad → all 4 corners collapse off-screen.
        qpos = float2(-2, -2);
    } else {
        qpos = float2(
            (corner == 0u || corner == 3u) ? mnx : mxx,
            (corner == 0u || corner == 1u) ? mny : mxy);
    }

    VSOutOctet o;
    // Z = nearest corner (max in reverse-Z) so depth test admits this quad
    // wherever it could potentially hit. PS SV_Depth writes refined per-pixel.
    o.svpos       = float4(qpos, saturate(maxNdcZ), 1.0);
    o.amin        = amin;
    o.amax        = amax;
    o.occ         = occ;
    o.cols        = cols;
    o.vms         = vms;
    o.lodScale    = ci.lodScale;
    o.lodIdxV     = gLwLodIdx;
    o.paletteBase = ci.paletteBase;
    o.baseVx      = uint3(bx*2u, by*2u, bz*2u);
    return o;
}

struct PSOutOctet
{
    float4 color : SV_Target;
    float  depth : SV_Depth;
};

PSOutOctet psmain_octet_billboard(VSOutOctet i)
{
    PSOutOctet o;

    // Build ray from camera through this pixel via gInvViewProj.
    float2 sp  = i.svpos.xy;
    float  invW = 1.0 / (float)gVwSize.x;
    float  invH = 1.0 / (float)gVwSize.y;
    float  u   = (sp.x + 0.5) * invW;
    float  v   = (sp.y + 0.5) * invH;
    float2 ndcP = float2(u * 2.0 - 1.0, 1.0 - v * 2.0);
    // Reverse-Z: ndc.z=1 is near plane (finite world point). ndc.z=0 would
    // unproject to the infinite-far plane and collapse all pixel rays.
    float4 nearH = mul(float4(ndcP, 1.0, 1.0), gInvViewProj);
    float3 nearW = nearH.xyz / nearH.w;
    float3 rd    = normalize(nearW - gCamPos);
    float3 invRd = 1.0 / rd;

    float lodScale = i.lodScale;
    float bestT = 1e9;
    int   bestVi = -1;
    float3 bestN = float3(0, 0, 1);

    [unroll] for (uint vi = 0; vi < 8u; ++vi) {
        if (((i.occ >> vi) & 1u) == 0u) continue;
        uint lx = (vi >> 0) & 1u;
        uint ly = (vi >> 1) & 1u;
        uint lz = (vi >> 2) & 1u;
        float3 vmin = i.amin + float3((float)lx, (float)ly, (float)lz) * lodScale;
        float3 vmax = vmin + lodScale;
        float3 t0   = (vmin - gCamPos) * invRd;
        float3 t1   = (vmax - gCamPos) * invRd;
        float3 tmn  = min(t0, t1);
        float3 tmx  = max(t0, t1);
        float tNear = max(max(tmn.x, tmn.y), tmn.z);
        float tFar  = min(min(tmx.x, tmx.y), tmx.z);
        if (tFar < 0.0 || tNear > tFar) continue;
        float tHit = max(tNear, 0.0);
        if (tHit < bestT) {
            bestT = tHit;
            bestVi = (int)vi;
            // Face normal: axis with the largest tmn = entry plane.
            if (tmn.x >= tmn.y && tmn.x >= tmn.z) bestN = float3(rd.x < 0 ? 1 : -1, 0, 0);
            else if (tmn.y >= tmn.z)              bestN = float3(0, rd.y < 0 ? 1 : -1, 0);
            else                                  bestN = float3(0, 0, rd.z < 0 ? 1 : -1);
        }
    }

    if (bestVi < 0) {
        // Pixel inside billboard rect but not over any voxel → discard.
        discard;
    }

    float3 hit = gCamPos + rd * bestT;
    float4 chit = mul(float4(hit, 1.0), gViewProj);
    // DIAGNOSTIC: rasterized VS depth (nearest corner per octet, per-quad
    // constant). Confirms sort path; per-pixel refinement next.
    o.depth = i.svpos.z;

    // Extract palette + visMask for the winning voxel.
    uint palIdx = (bestVi < 4) ? ((i.cols.x >> (bestVi * 8u)) & 0xFFu)
                                : ((i.cols.y >> ((bestVi - 4u) * 8u)) & 0xFFu);
    uint visMask = (bestVi < 4) ? ((i.vms.x >> (bestVi * 8u)) & 0x3Fu)
                                  : ((i.vms.y >> ((bestVi - 4u) * 8u)) & 0x3Fu);

    uint colPck = gLwPalette[i.paletteBase + palIdx];
    float3 albedo = float3(
        (float)( colPck         & 0xFFu),
        (float)((colPck >>  8u) & 0xFFu),
        (float)((colPck >> 16u) & 0xFFu)) / 255.0;

    uint vx = i.baseVx.x + (uint)((bestVi >> 0) & 1);
    uint vy = i.baseVx.y + (uint)((bestVi >> 1) & 1);
    uint vz = i.baseVx.z + (uint)((bestVi >> 2) & 1);
    uint cx = vx >> 5u, cy = vy >> 5u, cz = vz >> 5u;
    uint parity = (cx + cy + cz) & 1u;

    float ao = 1.0;
    float3 rgb = ShadeWithLighting6Face(albedo, hit, ao, visMask, i.lodIdxV, parity, (int)gMode);

    o.color = float4(rgb, 1.0);
    return o;
}
