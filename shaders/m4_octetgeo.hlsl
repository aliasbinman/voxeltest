// ============================================================================
// OctetGeo — real triangle geometry per octet (no per-pixel ray).
//
//   VS: vertex-pulling, 144 verts per octet instance = 8 child voxels * 3 camera-
//       facing faces * 2 tris * 3 verts. Each voxel emits only its 3 front faces;
//       empty children and internal faces (visMask) collapse to a degenerate tri.
//       Full shading (palette/AO/ambient/sun + fog) is done here and interpolated.
//   PS: pack the interpolated colour; depth from the rasterised geometry (pos.w).
//
// Same root signature + MRT output as OctetBillboards (drop-in for resolve/TAA).
// Best for very close geo where each triangle is large on screen.
// ============================================================================

#include "m4_common.hlsli"

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
