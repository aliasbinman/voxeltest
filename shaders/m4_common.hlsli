// ============================================================================
// m4_common.hlsli — shared definitions for the LW render techs (m4_lw splat,
// m4_octet billboards, m4_octetgeo). One source of truth for the per-frame
// constant buffer, the LW structured buffers, and the common shading helpers
// (ambient cube, fog, palette/worklist lookup) so the techs can't drift.
//
// Shaders that don't use a given SRV/field simply don't reference it — DXC strips
// unused resources, so the including tech's root signature need not bind them.
// ============================================================================
#ifndef M4_COMMON_HLSLI
#define M4_COMMON_HLSLI

#include "m4_frame.hlsli"   // cbPerFrame (b0)

cbuffer CBLwCS : register(b1)
{
    uint2 gVwSize;
    uint  gLwPointCount;
    uint  gLwNumWorkItems;
    uint  gLwLodIdx;
    int   gSplatRadius;    // 0 = single pixel; N writes (2N+1)x(2N+1).
    uint  gAoCount;        // gLwBlockAo element count (blockCount*8); 0 = no AO buffer
    uint  gTileCS;         // thread-group swizzle tile width (0 = off / linear)
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
// Baked per-voxel AO, 1 uint/voxel (low 24 bits = 6 faces * 4-bit), 8/block.
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

// Face normals, indexed 0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z (matches visMask/AO bits).
static const float3 kFaceN[6] = {
    float3( 1, 0, 0), float3(-1, 0, 0),
    float3( 0, 1, 0), float3( 0,-1, 0),
    float3( 0, 0, 1), float3( 0, 0,-1),
};

// Ray-vs-AABB slab test, visMask-aware. Hit when the exit t is non-negative and
// >= the entry t AND the entry face is enabled in faceMask (6 bits, kFaceN order
// 0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z) — a face occluded by a neighbour voxel (bit 0)
// is not a real surface, so the ray passes through it. Pass 0x3Fu for "all faces".
// Outputs the clamped entry distance (tHit >= 0) and the entry face. invRd = 1/rd.
bool RayAabb(float3 ro, float3 rd, float3 invRd, float3 vmin, float3 vmax,
             uint faceMask, out float tHit, out uint face)
{
    float3 t0 = (vmin - ro) * invRd;
    float3 t1 = (vmax - ro) * invRd;
    float3 tsm = min(t0, t1);
    float3 tbg = max(t0, t1);
    float tN = max(max(tsm.x, tsm.y), tsm.z);
    float tF = min(min(tbg.x, tbg.y), tbg.z);
    tHit = max(tN, 0.0);
    if (tF < 0.0 || tN > tF) { face = 0u; return false; }
    if      (tN == tsm.x) face = (rd.x > 0.0) ? 1u : 0u;
    else if (tN == tsm.y) face = (rd.y > 0.0) ? 3u : 2u;
    else                  face = (rd.z > 0.0) ? 5u : 4u;
    return ((faceMask >> face) & 1u) != 0u;   // occluded entry face → not a hit
}

// Ambient cube — warm/cool per axis face, triplanar-blended from the normal so
// each face picks up a different indirect tint instead of one flat scalar.
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

// Distance + height fog with a cheap sun-tinted forward-scatter term.
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

#endif // M4_COMMON_HLSLI
