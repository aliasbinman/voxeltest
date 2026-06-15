// ============================================================================
// OctetGeo — real triangle geometry per octet (no per-pixel ray).
//
//   VS: vertex-pulling, 144 verts per octet instance = 8 child voxels * 3 camera-
//       facing faces * 2 tris * 3 verts. Each voxel emits only its 3 front faces
//       (the face per axis whose normal points at the camera). Empty children and
//       internal faces (not in visMask) collapse to a degenerate offscreen tri.
//   PS: flat shade — palette colour * (ambient cube + sun), per-face baked AO, fog.
//
// Same root signature + MRT output as OctetBillboards (drop-in for resolve/TAA):
//   SV_Target0 (R32_UINT)  : 0xFF000000 | (B<<16)|(G<<8)|R
//   SV_Target1 (R32_FLOAT) : gNearZ / viewZ
//   Depth from real geometry (reverse-Z, GREATER, depth write) → HW occlusion.
//
// Best for very close geo where each triangle is large on screen: the rasteriser
// + flat PS beat the billboard's per-pixel 8-AABB ray.
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
    uint  gLwPointCount;
    uint  gLwNumWorkItems;
    uint  gLwLodIdx;
    int   gSplatRadius;
    uint  gAoCount;        // gLwBlockAo element count (blockCount*8); 0 = no AO buffer
    uint  _padCs;
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

float3 ApplyFog(float3 color, float3 wpos)
{
    float3 d = wpos - gCamPos;
    float dist = length(d);
    float optical = gFogDensity * dist;
    float3 rd = (dist > 1e-4) ? d / dist : float3(0, 0, 1);
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

struct GOut
{
    float4 pos     : SV_Position;
    // Full shading done in the VS: flat terms (palette/AO/ambient/sun) are constant
    // across a face, fog varies per vertex — so the final colour is INTERPOLATED.
    // Linear fog across a face is fine for the big near faces OctetGeo targets.
    float3 color   : TEXCOORD0;
};

// 2-tri quad as 6 verts → corner ids {0:(0,0) 1:(1,0) 2:(1,1) 3:(0,1)}.
static const uint kQuadCorner[6] = { 0u, 1u, 2u, 0u, 2u, 3u };

GOut DEGEN(GOut o) { o.pos = float4(2.0, 2.0, 2.0, 1.0); o.color = 0; return o; }

GOut vsmain_octetgeo(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    GOut o;
    o.color = 0;

    uint4 item      = LookupItem(iid);
    uint  slot      = item.x;
    uint  blockBase = item.y;
    uint  count     = item.z;
    uint  first     = item.w;
    if (iid - first >= count) return DEGEN(o);

    uint bi    = blockBase + (iid - first);
    uint pack0 = gLwBlockPos[bi];
    uint bx = (pack0 >>  0) & 0xFFu;
    uint by = (pack0 >>  8) & 0xFFu;
    uint bz = (pack0 >> 16) & 0xFFu;
    uint occ = (pack0 >> 24) & 0xFFu;

    // 144 verts/instance: voxel = vid/18, then 3 faces * 6 verts.
    uint voxel = vid / 18u;
    uint r     = vid % 18u;
    uint axis  = r / 6u;     // 0=x,1=y,2=z
    uint v6    = r % 6u;
    if (((occ >> voxel) & 1u) == 0u) return DEGEN(o);

    LwChunkInfo ci = gLwChunkInfos[slot];
    float vs = ci.lodScale;
    float3 octBase = ci.worldOrigin + float3(bx * 2u, by * 2u, bz * 2u) * vs;
    uint3  c3 = uint3(voxel & 1u, (voxel >> 1) & 1u, (voxel >> 2) & 1u);
    float3 vmin = octBase + float3(c3) * vs;
    float3 vcen = vmin + 0.5 * vs;

    // Front face on this axis = the side whose outward normal faces the camera.
    bool positive = gCamPos[axis] > vcen[axis];
    uint faceIdx  = axis * 2u + (positive ? 0u : 1u); // matches kFaceN order
    float faceCoord = positive ? (vmin[axis] + vs) : vmin[axis];

    // visMask cull — skip faces hidden by a neighbour (internal to the surface).
    uint2 visPack = gLwBlockVis[bi];
    uint  vmask   = (voxel < 4u) ? ((visPack.x >> (voxel * 8u)) & 0x3Fu)
                                 : ((visPack.y >> ((voxel - 4u) * 8u)) & 0x3Fu);
    if ((vmask & (1u << faceIdx)) == 0u) return DEGEN(o);

    // In-plane axes; build the quad corner for this vertex.
    uint b = (axis + 1u) % 3u;
    uint c = (axis + 2u) % 3u;
    uint corner = kQuadCorner[v6];
    float bb = (corner == 1u || corner == 2u) ? 1.0 : 0.0;
    float cc = (corner == 2u || corner == 3u) ? 1.0 : 0.0;
    float3 wp = vmin;
    wp[axis] = faceCoord;
    wp[b] += bb * vs;
    wp[c] += cc * vs;

    // ---- Constant-per-face shading (was in the PS; flat-interpolated now) ----
    uint2 cols = gLwBlockCol[bi];
    uint  palIdx = (voxel < 4u) ? ((cols.x >> (voxel * 8u)) & 0xFFu)
                                : ((cols.y >> ((voxel - 4u) * 8u)) & 0xFFu);
    uint  colPck = gLwPalette[ci.paletteBase + palIdx];
    float3 albedo = float3((colPck & 0xFFu),
                           ((colPck >> 8) & 0xFFu),
                           ((colPck >> 16) & 0xFFu)) / 255.0;

    float ao = 1.0;
    uint  aoIdx = bi * 8u + voxel;
    if (aoIdx < gAoCount)
    {
        uint aoW = gLwBlockAo[aoIdx];
        ao = (float)((aoW >> (faceIdx * 4u)) & 0xFu) * (1.0 / 15.0);
    }

    float3 N = kFaceN[faceIdx];
    float3 lit;
    if ((int)gMode == 3)
    {
        lit = float3(ao, ao, ao);     // AO viz — no fog
    }
    else
    {
        float3 sunCol = float3(1.0, 0.95, 0.85) * (gSunIntensity * 3.0);
        float  NdotL  = max(0.0, dot(N, gLightDir));
        float3 amb    = AmbientCube(N) * gAmbient;
        lit = albedo * (amb + NdotL * sunCol);
        lit *= lerp(1.0, ao, gAoStrength);
        lit = ApplyFog(lit, wp);       // per-vertex fog, interpolated across the face
    }

    // TAA jitter: gViewProj is the un-jittered base (splat adds gJitter in NDC), so
    // shift clip.xy by gJitter*clip.w → after the raster divide it lands as ndc+jitter.
    float4 clip = mul(float4(wp, 1.0), gViewProj);
    clip.xy += gJitter * clip.w;
    o.pos   = clip;
    o.color = lit;
    return o;
}

struct POut
{
    uint  color  : SV_Target0;
    float vdepth : SV_Target1;
};

POut psmain_octetgeo(GOut i)
{
    POut o;

    // clip.w == view Z for this projection, so the rasteriser already interpolated
    // it into SV_Position.w — no need to recompute viewZ from world.
    o.vdepth = gNearZ / i.pos.w;
    
    // Fully shaded (incl. fog) + interpolated in the VS — PS only packs the colour.
    float3 lit = i.color;
    uint R = (uint)(saturate(lit.r) * 255.0 + 0.5);
    uint G = (uint)(saturate(lit.g) * 255.0 + 0.5);
    uint B = (uint)(saturate(lit.b) * 255.0 + 0.5);
    o.color = 0xFF000000u | (B << 16) | (G << 8) | R;
    return o;
}
