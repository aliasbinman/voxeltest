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

// ---- PS: splat albedo ----
struct SplatOut {
    float4 col  : SV_Target0;
    uint   mask : SV_Target1;
};

SplatOut psmain_lw_splat_albedo(VSOut i)
{
    SplatOut o;
    o.col  = float4(i.colAO.rgb, EncodeSplatAlpha(i.colAO.a, gLodIdx, i.parity));
    o.mask = i.mask;     // full 30-bit packed: visMask(6) + 6 face AOs(24)
    return o;
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
