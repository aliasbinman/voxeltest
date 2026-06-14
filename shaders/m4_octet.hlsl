// ============================================================================
// OctetBillboards — per-octet billboard rasterization.
//
//   VS: one instanced quad per occupied octet (block = 2x2x2 voxels). Projects
//       the octet's 8 world-space AABB corners, takes the screen-space min/max
//       as the quad bounds. HW rasterizes every covered pixel.
//   PS: builds a ray through the pixel, ray-vs-AABB against the (up to) 8 child
//       voxels of the octet, nearest hit wins. Shades per-face (visMask + AO),
//       writes packed colour + viewZ depth (drop-in for the splat/dilate output)
//       and SV_Depth so HW depth resolves occlusion between overlapping octets.
//
// Output matches the dilate so resolve -> TAA -> post are unchanged:
//   SV_Target0 (R32_UINT)  : 0xFF000000 | (B<<16)|(G<<8)|R   (0 = sky)
//   SV_Target1 (R32_FLOAT) : gNearZ / viewZ                  (CSTiles form)
//   SV_Depth               : clip.z/clip.w (reverse-Z, GREATER, clear 0)
// ============================================================================

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
    float    gAoStrength;              // 500
    float2   _padPC;                   // 504..512
    float4   gReprojMx;                // 512
    float4   gReprojMy;                // 528
    float4   gReprojMw;                // 544
    float4   _padReproj;               // 560..576
};

cbuffer CBLwCS : register(b1)
{
    uint2 gVwSize;
    uint  gLwPointCount;   // unused here
    uint  gLwNumWorkItems;
    uint  gLwLodIdx;
    int   gSplatRadius;    // unused here
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
StructuredBuffer<uint2>       gLwBlockVis   : register(t6);
StructuredBuffer<uint>        gLwBlockAo    : register(t7);

// gid -> work item via binary search over cumulative firstThread (.w).
uint4 LookupItem(uint gid)
{
    uint lo = 0u, hi = gLwNumWorkItems;
    while (lo + 1u < hi)
    {
        uint mid = (lo + hi) >> 1u;
        if (gLwWorkItems[mid].w <= gid) lo = mid; else hi = mid;
    }
    return gLwWorkItems[lo];
}

static const float3 kFaceN[6] = {
    float3( 1, 0, 0), float3(-1, 0, 0),
    float3( 0, 1, 0), float3( 0,-1, 0),
    float3( 0, 0, 1), float3( 0, 0,-1),
};

static const float3 kAmbientCube[6] = {
    float3(0.85, 0.65, 0.45), float3(0.40, 0.50, 0.65),
    float3(0.85, 1.00, 1.20), float3(0.18, 0.14, 0.10),
    float3(0.65, 0.65, 0.55), float3(0.40, 0.45, 0.55),
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
    float  esz  = vs * 2.0; // octet spans 2 voxels per axis

    // Project the 8 corners; screen-space bbox in NDC over the corners in front
    // of the camera.
    float2 mn =  1e9;
    float2 mx = -1e9;
    uint nFront = 0u;
    [unroll]
    for (uint k = 0u; k < 8u; ++k)
    {
        float3 c = omin + esz * float3((k) & 1u, (k >> 1) & 1u, (k >> 2) & 1u);
        float4 cp = mul(float4(c, 1.0), gViewProj);
        if (cp.w <= 1e-4) continue;
        ++nFront;
        float2 n = cp.xy / cp.w;
        mn = min(mn, n);
        mx = max(mx, n);
    }
    // Octet entirely behind the camera → cull (emit a degenerate off-screen
    // vertex). Critical: do NOT fall back to a full-screen quad, or every behind-
    // camera octet becomes full-screen overdraw → GPU hang.
    if (nFront == 0u)
    {
        o.pos = float4(2.0, 2.0, 0.0, 1.0);
        o.octMin = omin; o.vsize = vs; o.occ = 0; o.blockIdx = 0; o.palBase = 0;
        return o;
    }
    // Straddling the near plane (some corners behind) → front-corner bbox is
    // incomplete; clamp to the screen so coverage stays bounded.
    if (nFront < 8u)
    {
        mn = max(mn, float2(-1, -1));
        mx = min(mx, float2(1, 1));
    }
    // Half-pixel pad so HW coverage doesn't drop edge pixels.
    mn -= gInvScreenSize;
    mx += gInvScreenSize;

    float2 corner = float2((vid & 1u) ? mx.x : mn.x,
                           (vid & 2u) ? mx.y : mn.y);
    o.pos     = float4(corner, 0.0, 1.0); // real depth comes from PS (SV_Depth)
    o.octMin  = omin;
    o.vsize   = vs;
    o.occ     = occ;
    o.blockIdx = bi;
    o.palBase  = ci.paletteBase;
    return o;
}

struct POut
{
    float4 color : SV_Target0; // tonemapped RGB straight to backbuffer (v1)
    float  depth : SV_Depth;   // clip.z/clip.w (reverse-Z)
};

POut psmain_octet(VOut i)
{
    POut o;

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
        if (((occ >> c) & 1u) == 0u) continue;
        float3 cmin = i.octMin + vs * float3(c & 1u, (c >> 1) & 1u, (c >> 2) & 1u);
        float3 cmax = cmin + vs;
        float3 t0 = (cmin - ro) * invRd;
        float3 t1 = (cmax - ro) * invRd;
        float3 tsm = min(t0, t1);
        float3 tbg = max(t0, t1);
        float tN = max(max(tsm.x, tsm.y), tsm.z);
        float tF = min(min(tbg.x, tbg.y), tbg.z);
        if (tF < max(tN, 0.0)) continue;        // miss
        float tHit = max(tN, 0.0);
        if (tHit >= bestT) continue;
        bestT = tHit;
        bestC = (int)c;
        // Entry face = axis that produced tN.
        if      (tN == tsm.x) bestFace = (rd.x > 0.0) ? 1u : 0u;
        else if (tN == tsm.y) bestFace = (rd.y > 0.0) ? 3u : 2u;
        else                  bestFace = (rd.z > 0.0) ? 5u : 4u;
    }
    if (bestC < 0) { discard; }

    float3 hit = ro + rd * bestT;
    float4 hc  = mul(float4(hit, 1.0), gViewProj);
    o.depth = hc.z / hc.w;

    // Palette colour for the winning child.
    uint2 cols = gLwBlockCol[i.blockIdx];
    uint palIdx = (bestC < 4) ? ((cols.x >> (bestC * 8u)) & 0xFFu)
                              : ((cols.y >> ((bestC - 4) * 8u)) & 0xFFu);
    uint colPck = gLwPalette[i.palBase + palIdx];
    float3 albedo = float3((colPck & 0xFFu),
                           ((colPck >> 8) & 0xFFu),
                           ((colPck >> 16) & 0xFFu)) / 255.0;

    // Per-face baked AO (indexed by the actual hit face).
    uint aoW = gLwBlockAo[i.blockIdx * 8u + (uint)bestC];
    float ao = (float)((aoW >> (bestFace * 4u)) & 0xFu) * (1.0 / 15.0);

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
    }

    // ACES filmic tonemap (Narkowicz) so linear-lit writes look right on the
    // backbuffer without the resolve/post chain (v1 direct path).
    float3 c = lit * gExposure;
    const float a = 2.51, ta = 0.03, tc = 2.43, td = 0.59, te = 0.14;
    c = saturate((c * (a * c + ta)) / (c * (tc * c + td) + te));
    o.color = float4(c, 1.0);
    return o;
}
