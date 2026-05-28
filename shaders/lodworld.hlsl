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
    float    gNearZ;          // for any future reconstruction maths
};
cbuffer CBLwLod : register(b1)
{
    uint  gLodIdx;            // 0..4 — encoded into splat alpha
    float gHalfExt;           // half-extent of voxel in world units (kLodScale * 0.5)
    uint  gLwSlot;            // chunk slot index for this draw
    uint  gLwDrawBase;        // offset added to SV_VertexID when fetching from pool
                              //  (= cluster.pointFirst when drawing a sub-span)
};

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

// ---- Splat alpha encoding (mirrors SplatEncodeAlpha in voxel.hlsl) ----
//   bit 7      = marker
//   bits 6:5   = lodIdx
//   bits 4:1   = AO 4-bit
//   bit 0      = cluster checker parity (0 = dark, 1 = light) — used by lodViz
float EncodeSplatAlpha(float ao01, uint lodIdx, uint parity)
{
    uint ao4 = (uint)(saturate(ao01) * 15.0 + 0.5);
    uint a8  = 0x80u | ((lodIdx & 3u) << 5) | ((ao4 & 0xFu) << 1) | (parity & 1u);
    return (float)a8 / 255.0;
}

// ---- VS ----
struct VSOut {
    float4 svpos : SV_Position;
    float3 col   : COLOR0;
    float  ao    : COLOR1;
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
    o.mask   = mask;
    o.parity = (cx + cy + cz) & 1u;

    uint colPck = gLwPalette[ci.paletteBase + palIdx];
    o.col = float3((float)(colPck         & 0xFFu),
                   (float)((colPck >>  8) & 0xFFu),
                   (float)((colPck >> 16) & 0xFFu)) / 255.0;

    // Pick AO face whose outward normal best aligns with toCam, restricted to
    // visible faces (matches the legacy vsmain_points heuristic).
    float3 toCam = normalize(gCamPos - world);
    float3 faceN[6] = {
        float3( 1, 0, 0), float3(-1, 0, 0),
        float3( 0, 1, 0), float3( 0,-1, 0),
        float3( 0, 0, 1), float3( 0, 0,-1),
    };
    float bestDot = -2.0;
    uint  bestF = 0u;
    [unroll] for (uint f = 0u; f < 6u; ++f) {
        if (((mask >> f) & 1u) == 0u) continue;
        float d = dot(faceN[f], toCam);
        if (d > bestDot) { bestDot = d; bestF = f; }
    }
    uint nib = (aoPck >> (bestF * 4u)) & 0xFu;
    o.ao = (float)nib / 15.0;
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
    o.col  = float4(i.col, EncodeSplatAlpha(i.ao, gLodIdx, i.parity));
    o.mask = i.mask & 0x3Fu;
    return o;
}

// ---- PS: opaque debug (no splat encoding — for "just see voxels" sanity) ----
float4 psmain_lw_debug(VSOut i) : SV_Target
{
    return float4(i.col, 1.0);
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
    float ao = lerp(0.45, 1.0, i.ao);     // keep some floor so dark voxels visible
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
