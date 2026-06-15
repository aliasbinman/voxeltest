// ============================================================================
// OctetBillboards — per-octet billboard rasterization.
//
//   VS: one instanced quad per occupied octet (block = 2x2x2 voxels). Projects
//       the octet's bounding sphere to a screen quad. HW rasterizes covered pixels.
//   PS: builds a ray through the pixel, ray-vs-AABB against the (up to) 8 child
//       voxels of the octet, nearest hit wins. Shades per-face (visMask + AO),
//       writes packed colour + viewZ depth (drop-in for the splat/dilate output).
//
// Output matches the dilate so resolve -> TAA -> post are unchanged:
//   SV_Target0 (R32_UINT)  : 0xFF000000 | (B<<16)|(G<<8)|R   (0 = sky)
//   SV_Target1 (R32_FLOAT) : gNearZ / viewZ                  (CSTiles form)
// Depth from the VS nearest-corner reverse-Z so early-Z rejects occluded octets.
// ============================================================================

#include "m4_common.hlsli"

struct VOut
{
    float4 pos     : SV_Position;
    nointerpolation float3 octMin   : TEXCOORD0; // octet world min corner
    nointerpolation float  vsize    : TEXCOORD1; // voxel edge (lodScale)
    nointerpolation uint   occ      : TEXCOORD2; // 8-bit occupancy
    nointerpolation uint   blockIdx : TEXCOORD3; // global block index
    nointerpolation uint   palBase  : TEXCOORD4; // palette atlas base
};

// One instanced quad (triangle strip, 4 verts) per octet.
VOut vsmain_octet(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    VOut o;

    uint4 item       = LookupItem(iid);
    uint  slot       = item.x;
    uint  blockBase  = item.y;
    uint  count      = item.z;
    uint  first      = item.w;

    // Guard padding instances (instance count rounded up): emit off-screen.
    if (iid - first >= count)
    {
        o.pos = float4(2.0, 2.0, 0.0, 1.0);
        o.octMin = 0; o.vsize = 0; o.occ = 0; o.blockIdx = 0; o.palBase = 0;
        return o;
    }

    uint bi    = blockBase + (iid - first);
    uint pack0 = gLwBlockPos[bi];
    uint bx = (pack0 >>  0) & 0xFFu;
    uint by = (pack0 >>  8) & 0xFFu;
    uint bz = (pack0 >> 16) & 0xFFu;
    uint occ = (pack0 >> 24) & 0xFFu;

    LwChunkInfo ci = gLwChunkInfos[slot];
    float vs   = ci.lodScale;
    float3 omin = ci.worldOrigin + float3(bx * 2u, by * 2u, bz * 2u) * vs;

    // ---- Robust billboard: the quad MUST fully enclose the octet's screen
    // silhouette, else its own edge pixels (ray misses the AABB) discard to black —
    // an axis-aligned quad can't tightly bound a rotated box, so use the bounding
    // sphere (radius = box half-diagonal sqrt(3)*vs), which always contains it.
    // Size at the sphere's NEAREST depth (cc.w - R) so perspective never clips it.
    float3 center = omin + vs;        // octet centre (spans 2*vs → half = vs)
    float  R      = vs * 1.7320508;   // bounding-sphere radius = box half-diagonal

    float4 cc = mul(float4(center, 1.0), gViewProj);
    // cc.w == view-space Z (reverse-Z proj). Cull if the octet crosses/behind the
    // near plane — the projection is unstable there; the splat covers it.
    if (cc.w - R <= gNearZ)
    {
        o.pos = float4(2.0, 2.0, 0.0, 1.0);
        o.octMin = omin; o.vsize = vs; o.occ = 0u; o.blockIdx = 0u; o.palBase = 0u;
        return o;
    }

    float2 cn    = cc.xy / cc.w;                          // centre in NDC (centre depth)
    float  wNear = cc.w - R;                              // sphere nearest depth
    float  ndcRy = R / (wNear * gTanHalfFovY);            // sphere radius (NDC y)
    float  ndcRx = ndcRy / gAspect;                       // sphere radius (NDC x)

    float2 ext = float2(ndcRx, ndcRy) + gInvScreenSize; // + half-pixel pad
    float2 corner = float2((vid & 1u) ? cn.x + ext.x : cn.x - ext.x,
                           (vid & 2u) ? cn.y + ext.y : cn.y - ext.y);
    // Conservative nearest reverse-Z (sphere nearest depth) for HW early-Z.
    float maxRz = cc.z / wNear;

    o.pos      = float4(corner, maxRz, 1.0);
    o.octMin   = omin;
    o.vsize    = vs;
    o.occ      = occ;
    o.blockIdx = bi;
    o.palBase  = ci.paletteBase;
    return o;
}

struct POut
{
    uint  color  : SV_Target0; // packed 0xFF000000|(B<<16)|(G<<8)|R (drop-in for dilate)
    float vdepth : SV_Target1; // gNearZ / viewZ (CSTiles form, for TAA)
};

POut psmain_octet(VOut i)
{
    POut o;
    o.vdepth = 0.0;
    o.color = 0;

    // Ray through this pixel (camera basis; matches dilate PixelWorldDir).
    float2 uv = i.pos.xy * gInvScreenSize;
    float ndcX = uv.x * 2.0 - 1.0;
    float ndcY = 1.0 - uv.y * 2.0;
    float3 ro = gCamPos;
    float3 rd = normalize(gCamRight * (ndcX * gAspect * gTanHalfFovY)
                        + gCamUp    * (ndcY * gTanHalfFovY)
                        + gCamForward);
    float3 invRd = 1.0 / rd;

    float vs   = i.vsize;
    uint  occ  = i.occ;

    float bestT = 1e30;
    int   bestC = -1;
    uint  bestFace = 0;
    [unroll]
    for (uint c = 0u; c < 8u; ++c)
    {
        if (((occ >> c) & 1u) == 0u)
            continue;

        float3 cmin = i.octMin + vs * float3(c & 1u, (c >> 1) & 1u, (c >> 2) & 1u);
        float3 cmax = cmin + vs;
        float3 t0 = (cmin - ro) * invRd;
        float3 t1 = (cmax - ro) * invRd;
        float3 tsm = min(t0, t1);
        float3 tbg = max(t0, t1);
        float tN = max(max(tsm.x, tsm.y), tsm.z);
        float tF = min(min(tbg.x, tbg.y), tbg.z);
        if (tF < max(tN, 0.0))
            continue;        // miss
        float tHit = max(tN, 0.0);
        if (tHit >= bestT)
            continue;
        bestT = tHit;
        bestC = (int)c;

        // Entry face = axis that produced tN.
        if      (tN == tsm.x) bestFace = (rd.x > 0.0) ? 1u : 0u;
        else if (tN == tsm.y) bestFace = (rd.y > 0.0) ? 3u : 2u;
        else                  bestFace = (rd.z > 0.0) ? 5u : 4u;
    }
    if (bestC < 0) { discard; }

    float3 hit = ro + rd * bestT;
    float viewZ = max(dot(hit - ro, gCamForward), 1e-3);
    o.vdepth = gNearZ / viewZ;

    // Palette colour for the winning child.
    uint2 cols = gLwBlockCol[i.blockIdx];
    uint palIdx = (bestC < 4) ? ((cols.x >> (bestC * 8u)) & 0xFFu)
                              : ((cols.y >> ((bestC - 4) * 8u)) & 0xFFu);
    uint colPck = gLwPalette[i.palBase + palIdx];
    float3 albedo = float3((colPck & 0xFFu),
                           ((colPck >> 8) & 0xFFu),
                           ((colPck >> 16) & 0xFFu)) / 255.0;

    // Per-face baked AO (indexed by the actual hit face). gLwBlockAo is a ROOT SRV
    // (no HW bounds check) so a bad index page-faults the GPU → clamp against the
    // real element count passed in gAoCount (0 = no AO buffer bound).
    float ao = 1.0;
    uint aoIdx = i.blockIdx * 8u + (uint)bestC;
    if (aoIdx < gAoCount)
    {
        uint aoW = gLwBlockAo[aoIdx];
        ao = (float)((aoW >> (bestFace * 4u)) & 0xFu) * (1.0 / 15.0);
    }

    // Shade: ambient cube (AO-attenuated) + sun. Mode 3 = AO viz.
    float3 N = kFaceN[bestFace];
    float3 lit;
    if ((int)gMode == 3)
    {
        lit = float3(ao, ao, ao);
    }
    else
    {
        float3 sunCol = float3(1.0, 0.95, 0.85) * (gSunIntensity * 3.0);
        float  NdotL  = max(0.0, dot(N, gLightDir));
        float3 amb    = AmbientCube(N) * gAmbient;
        lit = albedo * (amb + NdotL * sunCol);
        lit *= lerp(1.0, ao, gAoStrength);
        lit = ApplyFog(lit, hit);
    }

    // Pack linear-lit to the same 8-bit ABGR the dilate writes; resolve→TAA→post
    // consume it (sky/fog/tonemap there). Drop-in with the splat output.
    uint R = (uint)clamp(lit.r * 255.0, 0.0, 255.0);
    uint G = (uint)clamp(lit.g * 255.0, 0.0, 255.0);
    uint B = (uint)clamp(lit.b * 255.0, 0.0, 255.0);
    o.color = 0xFF000000u | (B << 16) | (G << 8) | R;
    return o;
}
