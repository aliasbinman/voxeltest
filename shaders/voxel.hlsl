cbuffer cbPerFrame : register(b0)
{
    row_major float4x4 gViewProj;
    float3   gCamPos;
    float    gMode;
    float3   gLightDir;
    float    gAmbient;
    float3   gPointNormal;   // points technique: faked normal (= -camForward)
    float    _pad0;
    row_major float4x4 gInvViewProj;
    float2   gScreenSize;
    float2   _pad1;
    float3   gCamRight;
    float    _pad3;
    float3   gCamUp;
    float    _pad4;
    float3   gCamForward;
    float    gTanHalfFovY;
    float3   gFogColor;
    float    gFogDensity;       // 0 = depth fog disabled
    float    gHeightFogDensity; // 0 = height fog disabled
    float    gHeightFogFalloff; // exponential falloff per world unit of height
    float    gHeightFogStart;   // world Y of fog "ground plane" (max density)
    float    _padHF;
    float3   gSceneOrigin;
    float    gNearZ;            // reverse-Z infinite-far: viewZ = gNearZ / depth
    float3   gSceneSpan;
    float    _pad6;
    row_major float4x4 gPrevViewProj;     // last frame's VP (for TAA reprojection)
    float2   gJitter;                     // sub-pixel NDC offset applied to current proj
    float2   _pad7;
    row_major float4x4 gSunViewProj;      // sun ortho VP for shadow map
    float    gShadowBias;
    float    gShadowMapSize;
    float    gShadowEnable;
    float    gSunIntensity;
    float    gExposure;
    float    gRoughness;
    float    gColorizeClusters;   // 0/1 toggle
    float    gGridSize;           // scene grid size (>1 = replicated, wrap shadow)
};

// Combined depth + height fog. Height fog uses Inigo Quilez closed-form:
//   density(h) = c * exp(-h * b)
//   ∫ along ray of length D with rd.y != 0 = c * exp(-camY*b) * (1 - exp(-D*rd.y*b)) / rd.y
// Add depth-fog (constant density) optical depth, then transmittance = exp(-total).
float3 ApplyFog(float3 color, float3 wpos)
{
    float3 d = wpos - gCamPos;
    float dist = length(d);
    float optical = gFogDensity * dist;
    float3 rd = (dist > 1e-4) ? d / dist : float3(0,0,1);
    if (gHeightFogDensity > 0.0 && dist > 1e-4) {
        float b  = gHeightFogFalloff;
        float c  = gHeightFogDensity;
        float ey = exp(-(gCamPos.y - gHeightFogStart) * b);
        float t;
        if (abs(rd.y) > 1e-4) {
            t = c * ey * (1.0 - exp(-dist * rd.y * b)) / rd.y;
        } else {
            t = c * ey * dist;
        }
        optical += max(t, 0.0);
    }
    if (optical <= 0.0) return color;
    // Directional / sun-tinted fog (Quilez "Better fog" bonus): warmer toward sun.
    float3 sunDir  = normalize(gLightDir);
    float  sunAmt  = pow(saturate(dot(rd, sunDir)), 8.0);
    float3 sunTint = float3(1.10, 0.85, 0.55);
    float3 fogCol  = lerp(gFogColor, sunTint, sunAmt);
    return lerp(fogCol, color, exp(-optical));
}

cbuffer cbPerChunk : register(b1)
{
    float3 gChunkBase;
    float  _pad;
    uint   gVoxelBase;       // PolyVID: chunk's first voxel index in pointSb_
    uint   gChunkLodIdx;     // 0..3 splat LOD index (used by splat encoding)
    uint   gChunkTint;       // colorize-clusters tint: 1 green, 2 red, 3 orange, 4 yellow, 0 off
    uint2  _pad2;
};

// Cardinal face normals indexed by faceIdx (0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z).
static const float3 kFaceNormals[6] = {
    float3( 1, 0, 0), float3(-1, 0, 0),
    float3( 0, 1, 0), float3( 0,-1, 0),
    float3( 0, 0, 1), float3( 0, 0,-1),
};

// Ambient cube indexed by faceIdx (0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z).
static const float3 kAmbientCube[6] = {
    float3(0.85, 0.65, 0.45),  // +X warm
    float3(0.40, 0.50, 0.65),  // -X cool
    float3(0.85, 1.00, 1.20),  // +Y sky
    float3(0.18, 0.14, 0.10),  // -Y ground
    float3(0.65, 0.65, 0.55),  // +Z
    float3(0.40, 0.45, 0.55),  // -Z
};

// Unified 12-byte vertex used by both vb_ and pointVb_.
//   pck = uint4: .xyz = 16-bit scene-relative voxel coord (0..65535), .w unused
//   col = uint4: R,G,B,A bytes. alpha byte = visMask for point verts.
struct VSIn  {
    uint4 pck : POSITION;
    uint4 col : COLOR;
};
typedef VSIn VSPolyIn;
struct VSOut {
    float4 svpos : SV_Position;
    float3 wpos  : WPOS;
    float4 col   : COL;
    nointerpolation uint faceIdx : FIDX;
    nointerpolation uint shadowMask : SHADOW;
    nointerpolation float ao : AO;
};

float3 UnpackVoxPos(uint4 p)
{
    return float3(p.x, p.y, p.z);
}

// Per-face shadow mask: low 6 bits of aux (pck.w for VSIn / (pck.y>>16) for
// VoxelP). Bit i: 1 = face i directly lit, 0 = shadowed. Face order matches
// kFaceNormals (0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z).
uint ShadowMaskFromAux(uint aux) { return aux & 0x3Fu; }

// Per-voxel baked AO. Stored in the LOW 8 bits of aux (the same uint16 word
// also carries the 6-bit shadow mask in its low 6 bits — these are separate
// bit ranges of the 16-bit aux; see vox_loader.cpp). 0 = fully occluded,
// 1 = fully open.
float AoFromAux(uint aux) { return (float)(aux & 0xFFu) * (1.0 / 255.0); }

// Pick cardinal face index from a (possibly noisy) normal — dominant axis wins.
uint FaceFromNormal(float3 n)
{
    float3 a = abs(n);
    if (a.x >= a.y && a.x >= a.z) return n.x > 0.0 ? 0u : 1u;
    if (a.y >= a.z)               return n.y > 0.0 ? 2u : 3u;
    return n.z > 0.0 ? 4u : 5u;
}

float ShadowBit(uint mask, uint fi)         { return (float)((mask >> fi) & 1u); }
float ShadowBitN(uint mask, float3 n)       { return ShadowBit(mask, FaceFromNormal(n)); }

// Sun shadow map (cascade 0). Single ortho-projected depth tex sampled with
// a SamplerComparisonState for hardware PCF. gShadowEnable < 0.5 -> stub 1.0.
Texture2D<float>          gShadowTex  : register(t6);
SamplerComparisonState    gShadowSamp : register(s1);
float SampleShadow(float3 wpos) {
    if (gShadowEnable < 0.5) return 1.0;
    float4 sp = mul(float4(wpos, 1.0), gSunViewProj);
    float3 ndc = sp.xyz / sp.w;
    // Outside frustum -> lit.
    if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 ||
        ndc.z <  0.0 || ndc.z > 1.0) return 1.0;
    float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    // Reverse-Z: closer to sun = larger z. Sampler is GREATER_EQUAL:
    // lit when (ref >= stored). Add bias (not subtract) so a receiver right
    // on a caster surface stays just on the lit side.
    float ref = ndc.z + gShadowBias;
    // 3x3 PCF.
    float texel = 1.0 / max(gShadowMapSize, 1.0);
    float sum = 0.0;
    [unroll] for (int dy = -1; dy <= 1; ++dy)
    [unroll] for (int dx = -1; dx <= 1; ++dx) {
        sum += gShadowTex.SampleCmpLevelZero(gShadowSamp, uv + float2(dx, dy) * texel, ref);
    }
    return sum / 9.0;
}

// Combined lighting (no shadow map). shadow argument retained for call-site compat.
float3 ApplyShadowLighting(float3 amb, float3 sunDir, float3 n, float shadow)
{
    float ndotl = saturate(dot(n, sunDir));
    float3 sun  = float3(1.10, 1.00, 0.85) * ndotl * shadow * gSunIntensity;
    return amb + gAmbient * sun;
}

// LodViz tint: 1=Poly green, 2=PointL0 blue, 3=PointL1 purple,
// 4=PointL2 yellow, 5=PointL3 orange.
float3 ClusterTint(uint t)
{
    if (t == 1u) return float3(0.3, 1.0, 0.3);  // poly = green
    if (t == 2u) return float3(0.3, 0.5, 1.0);  // L0 = blue
    if (t == 3u) return float3(0.8, 0.3, 1.0);  // L1 = purple
    if (t == 4u) return float3(1.0, 1.0, 0.3);  // L2 = yellow
    if (t == 5u) return float3(1.0, 0.55, 0.1); // L3 = orange
    return float3(1.0, 1.0, 1.0);
}
float3 ApplyClusterTint(float3 c)
{
    return (gColorizeClusters > 0.5) ? c * ClusterTint(gChunkTint) : c;
}

// ACES filmic tonemap with exposure pre-multiply.
float3 Tonemap(float3 x)
{
    x = x * gExposure;
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// Per-chunk tint pre-tonemap. Lit PS paths read gChunkTint from per-chunk CB.
float3 TonemapTinted(float3 x) { return Tonemap(ApplyClusterTint(x)); }

// Per-voxel packed face AO (4 bits per face × 6 faces, low 24 bits). Indexed
// by global vertex ID for point paths; by voxelIdx for PolyVID/PolyAxis.
StructuredBuffer<uint> gPolyAo6 : register(t2);

// Voxel record: 12 B, matches CPU `Vertex` struct in src/mesh.h.
// pck.x = (x | y<<16), pck.y = (z | aux<<16). col = RGB low24 | visMask<<24.
struct VoxelP { uint2 pck; uint col; };
float3 UnpackVoxPosCS(uint2 p)
{
    return float3((float)(p.x & 0xFFFFu),
                  (float)((p.x >> 16u) & 0xFFFFu),
                  (float)(p.y & 0xFFFFu));
}
StructuredBuffer<VoxelP> gPolyVoxels : register(t1);

// ---------------- Shadow map blur fill ----------------
// Sparse point-splat casters leave many shadow texels at depth 0 (far). This
// CS runs after the caster pass: each empty texel (depth ≈ 0) is replaced by
// the average of its 8 non-empty neighbours. Already-written texels passthrough.
// Reads gShadowSrcTex (raw shadow), writes gShadowDstUav (filled shadow).
Texture2D<float>     gShadowSrcTex : register(t7);
RWTexture2D<float>   gShadowDstUav : register(u3);
[numthreads(8, 8, 1)]
void csmain_shadow_blur(uint3 dt : SV_DispatchThreadID)
{
    int W = (int)gShadowMapSize;
    int H = (int)gShadowMapSize;
    int2 pix = (int2)dt.xy;
    if (pix.x >= W || pix.y >= H) return;

    float self = gShadowSrcTex.Load(int3(pix, 0));
    if (self > 0.0001) {
        gShadowDstUav[pix] = self;
        return;
    }
    float sum = 0.0;
    int   cnt = 0;
    [unroll] for (int dy = -1; dy <= 1; ++dy) {
        [unroll] for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int2 sp = pix + int2(dx, dy);
            if (sp.x < 0 || sp.x >= W || sp.y < 0 || sp.y >= H) continue;
            float v = gShadowSrcTex.Load(int3(sp, 0));
            if (v > 0.0001) { sum += v; ++cnt; }
        }
    }
    gShadowDstUav[pix] = (cnt > 0) ? (sum / (float)cnt) : 0.0;
}

// ---------------- Shadow caster (depth-only) ----------------
// Reads from gPolyVoxels via SV_VertexID + gVoxelBase, transforms with
// gSunViewProj. No PS bound -> just writes depth into the shadow map.
struct VSShadowOut { float4 svpos : SV_Position; };
VSShadowOut vsmain_shadow(uint vid : SV_VertexID)
{
    VSShadowOut o;
    VoxelP v = gPolyVoxels[vid + gVoxelBase];
    float3 local = UnpackVoxPosCS(v.pck) + _pad;
    float3 world = gChunkBase + local;
    o.svpos = mul(float4(world, 1.0), gSunViewProj);
    return o;
}

// ---------------- Points technique ----------------
// faceIdx field carries visMask (6 bits, one per cube face direction).
struct VSPointOut {
    float4 svpos : SV_Position;
    float3 col   : COL;
    float3 wpos  : WPOS;
    nointerpolation uint mask : MASK;
    nointerpolation uint shadowMask : SHADOW;
    nointerpolation float ao : AO;
};

// Reads voxel data from the structured buffer (gPolyVoxels at t1) via
// SV_VertexID + gVoxelBase — no VB input. Replaces the older VSIn-based path
// so we no longer need a duplicate D3D11 vertex buffer alongside the SRV.
VSPointOut vsmain_points(uint vid : SV_VertexID)
{
    VSPointOut o;
    VoxelP v = gPolyVoxels[vid + gVoxelBase];
    // _pad carries half-extent of the LOD voxel (0.5/1.0/2.0); scene-rel pos + pad = voxel center.
    float3 local = UnpackVoxPosCS(v.pck) + _pad;
    float3 world = gChunkBase + local;
    o.svpos = mul(float4(world, 1.0), gViewProj);
    o.col   = float3((v.col & 0xFFu),
                     (v.col >> 8u) & 0xFFu,
                     (v.col >> 16u) & 0xFFu) / 255.0;
    o.wpos  = world;
    o.mask  = (v.col >> 24u) & 0x3Fu;     // visMask packed in color alpha byte
    o.shadowMask = 0xFFu;
    // Pick the visible face whose outward normal is most aligned with the
    // direction from voxel center toward the camera. Restricting to faces in
    // visMask is critical: dominant-axis-of-toCam picks the wrong face for
    // distant voxels (e.g. top voxels at far-away positions can have toCam.x
    // > toCam.y and end up showing +X AO instead of +Y).
    uint ao6 = gPolyAo6[vid + gVoxelBase];
    float3 toCam = normalize(gCamPos - world);
    float3 faceN[6] = {
        float3( 1, 0, 0), float3(-1, 0, 0),
        float3( 0, 1, 0), float3( 0,-1, 0),
        float3( 0, 0, 1), float3( 0, 0,-1),
    };
    float bestDot = -2.0;
    uint  bestFace = 0u;
    [unroll] for (uint f = 0u; f < 6u; ++f) {
        if (((o.mask >> f) & 1u) == 0u) continue;
        float d = dot(faceN[f], toCam);
        if (d > bestDot) { bestDot = d; bestFace = f; }
    }
    uint nib = (ao6 >> (bestFace * 4u)) & 0xFu;
    o.ao = (float)nib / 15.0;
    return o;
}

// Triplanar-style ambient cube: linear blend of 3 active faces, weighted
// by |n.x|, |n.y|, |n.z|, normalized so the result sits inside the cube hull.
float3 SampleAmbientCubeTriplanar(float3 n)
{
    float3 an = abs(n);
    float  total = an.x + an.y + an.z;
    total = max(total, 1e-5);
    float3 amb = an.x * (n.x > 0.0 ? kAmbientCube[0] : kAmbientCube[1])
               + an.y * (n.y > 0.0 ? kAmbientCube[2] : kAmbientCube[3])
               + an.z * (n.z > 0.0 ? kAmbientCube[4] : kAmbientCube[5]);
    return amb / total;
}

// Simple: cam-to-voxel pseudo-normal, triplanar ambient + ndotl.
float4 psmain_points_simple(VSPointOut i) : SV_Target
{
    int mode = (int)gMode;
    float fogDist = length(i.wpos - gCamPos);
    if (mode == 1) return float4(ApplyFog(i.col, i.wpos), 1.0);
    if (mode == 3) return float4(ApplyFog(i.ao.xxx, i.wpos), 1.0);
    if (mode == 4) return float4(i.ao * ClusterTint(gChunkTint), 1.0);
    float3 n = normalize(gCamPos - i.wpos);
    if (mode == 2) return float4(ApplyFog(n * 0.5 + 0.5, i.wpos), 1.0);
    float3 amb  = SampleAmbientCubeTriplanar(n) * i.ao;
    float3 light = ApplyShadowLighting(amb, normalize(gLightDir), n, ShadowBitN(i.shadowMask, n) * SampleShadow(i.wpos));
    return float4(TonemapTinted(ApplyFog(i.col * light, i.wpos)), 1.0);
}

// Complex: per-axis up-to-3 cardinal faces, weighted by projected area.
float4 psmain_points(VSPointOut i) : SV_Target
{
    int mode = (int)gMode;
    if (mode == 1) return float4(ApplyFog(i.col, i.wpos), 1.0);
    if (mode == 3) return float4(ApplyFog(i.ao.xxx, i.wpos), 1.0);
    if (mode == 4) return float4(i.ao * ClusterTint(gChunkTint), 1.0);

    // Choose up to 3 candidate cardinal faces: the ones the camera could see.
    // Per axis: positive face if D.axis >= 0, else negative face.
    // Weight each contribution by projected area (foreshortening) =
    // |dot(faceNormal, normalize(camera->voxel))|.
    float3 D  = gCamPos - i.wpos;
    float3 Dn = normalize(D);
    float3 L  = normalize(gLightDir);

    float3 accumCol = float3(0, 0, 0);
    float3 accumN   = float3(0, 0, 0);
    float  accumW   = 0.0;

    [unroll]
    for (int axis = 0; axis < 3; ++axis) {
        float dval = (axis == 0) ? D.x : (axis == 1) ? D.y : D.z;
        bool posDir = dval >= 0.0;
        uint bitIdx = (uint)axis * 2u + (posDir ? 0u : 1u);
        if (((i.mask >> bitIdx) & 1u) == 0u) continue;

        float3 n = float3(0, 0, 0);
        if      (axis == 0) n.x = posDir ? 1.0 : -1.0;
        else if (axis == 1) n.y = posDir ? 1.0 : -1.0;
        else                n.z = posDir ? 1.0 : -1.0;

        // Projected-area weight (n always points toward camera here).
        float w = saturate(dot(n, Dn));

        float3 amb  = kAmbientCube[bitIdx] * i.ao;
        float3 light = ApplyShadowLighting(amb, L, n, ShadowBit(i.shadowMask, bitIdx) * SampleShadow(i.wpos));
        accumCol += i.col * light * w;
        accumN   += n * w;
        accumW   += w;
    }

    float fogDist = length(i.wpos - gCamPos);
    if (accumW <= 1e-5) {
        if (mode == 2) return float4(ApplyFog(float3(0.5, 0.5, 0.5), i.wpos), 1.0);
        return float4(ApplyFog(i.col, i.wpos), 1.0);
    }

    if (mode == 2) {
        return float4(ApplyFog(normalize(accumN) * 0.5 + 0.5, i.wpos), 1.0);
    }
    return float4(TonemapTinted(ApplyFog(accumCol / accumW, i.wpos)), 1.0);
}

// ---------------- PointCS compute path ----------------
// CB layout matches host CBPerCS in renderer.cpp.
cbuffer cbCS : register(b2)
{
    row_major float4x4 gCsViewProj;
    float3   gCsChunkBase;
    uint     gCsVoxelOffset;
    uint     gCsVoxelCount;
    uint     gCsW;
    uint     gCsH;
    float    gCsLodHalfExtent;   // 0.5/1.0/2.0 (bit-cast from float in host)
};

// VoxelP + UnpackVoxPosCS hoisted above vsmain_points; only the CS-only
// buffer alias stays here.
StructuredBuffer<VoxelP> gVoxels  : register(t0);
RWTexture2D<uint>        gColorUav : register(u0);

[numthreads(64, 1, 1)]
void csmain_points_cs(uint3 dt : SV_DispatchThreadID)
{
    uint idx = dt.x;
    if (idx >= gCsVoxelCount) return;
    VoxelP v = gVoxels[gCsVoxelOffset + idx];
    float3 local = UnpackVoxPosCS(v.pck);
    float3 world = gCsChunkBase + local + gCsLodHalfExtent;
    float4 clip = mul(float4(world, 1.0), gCsViewProj);
    if (clip.w <= 0.0) return;
    float3 ndc = clip.xyz / clip.w;
    if (ndc.x < -1.0 || ndc.x > 1.0 ||
        ndc.y < -1.0 || ndc.y > 1.0 ||
        ndc.z <  0.0 || ndc.z > 1.0) return;

    int2 pix;
    pix.x = (int)((ndc.x * 0.5 + 0.5) * (float)gCsW);
    pix.y = (int)((-ndc.y * 0.5 + 0.5) * (float)gCsH);
    if (pix.x < 0 || pix.x >= (int)gCsW || pix.y < 0 || pix.y >= (int)gCsH) return;

    // No depth test: last-writer-wins. Order = dispatch order.
    gColorUav[pix] = v.col;
}

// ---------------- Blit pass: R32_UINT (packed RGBA) -> backbuffer ----------------
Texture2D<uint> gBlitSrc : register(t0);

struct VBlitOut { float4 pos : SV_Position; };

VBlitOut vsmain_blit(uint vid : SV_VertexID)
{
    // Full-screen triangle from 3 verts.
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VBlitOut o;
    o.pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return o;
}

float4 psmain_blit(VBlitOut i) : SV_Target
{
    int2 pix = int2(i.pos.xy);
    uint pack = gBlitSrc.Load(int3(pix, 0));
    float4 c;
    c.r = ((pack >>  0u) & 0xFFu) / 255.0;
    c.g = ((pack >>  8u) & 0xFFu) / 255.0;
    c.b = ((pack >> 16u) & 0xFFu) / 255.0;
    c.a = 1.0;
    return c;
}

// ---------------- PolyVID technique ----------------
// Shared IB has values = i*8 + corner (0..7) per voxel i. Per-chunk Draw uses
// baseVertex = chunkVoxelBase*8 so SV_VertexID = (chunkVoxelBase + i)*8 + corner.
// VS pulls voxel data from gPolyVoxels (declared earlier with the Points VS).

struct VSPolyVidOut {
    float4 svpos : SV_Position;
    float3 nrm   : NRM;
    float3 col   : COL;
    float3 wpos  : WPOS;
    nointerpolation uint shadowMask : SHADOW;
    nointerpolation float ao : AO;
};

VSPolyVidOut vsmain_polyvid(uint vid : SV_VertexID)
{
    VSPolyVidOut o;
    uint voxelIdx  = (vid / 8u) + gVoxelBase;
    uint cornerIdx = vid & 7u;
    VoxelP v = gPolyVoxels[voxelIdx];
    // Scene-relative packed: 10|10|10|2.
    float3 voxLocal  = UnpackVoxPosCS(v.pck);
    float3 voxCenter = gChunkBase + voxLocal + 0.5;
    float3 cornerOff = float3((cornerIdx & 1u) ? 0.5 : -0.5,
                              (cornerIdx & 2u) ? 0.5 : -0.5,
                              (cornerIdx & 4u) ? 0.5 : -0.5);
    float3 world = voxCenter + cornerOff;
    o.svpos = mul(float4(world, 1.0), gViewProj);
    o.nrm   = normalize(cornerOff);
    o.col   = float3((v.col >>  0u) & 0xFFu,
                     (v.col >>  8u) & 0xFFu,
                     (v.col >> 16u) & 0xFFu) / 255.0;
    o.wpos  = world;
    o.shadowMask = ShadowMaskFromAux(v.pck.y >> 16u);
    return o;
}

// ---------------- PolyAxis technique ----------------
// Single point in => up to 3 camera-facing axis faces (6 tris / 18 verts).
// No VB / no IB. DrawInstanced(18, voxelCount). VS picks +/- sign per axis
// from `sign(camPos - voxelCenter)`, kills the face when visMask says it's
// occluded, and flips one in-plane axis so winding stays outward.
VSPolyVidOut vsmain_polyaxis(uint vid : SV_VertexID)
{
    VSPolyVidOut o;
    // 18 verts per voxel, single non-instanced Draw(18*count). vid decomposes
    // into: voxel index, axis-face (0..2), and which of the 6 verts in that
    // face. No VB, no IB.
    uint voxelIdx = (vid / 18u) + gVoxelBase;
    uint inVox    = vid - (vid / 18u) * 18u;   // 0..17
    VoxelP v = gPolyVoxels[voxelIdx];
    float3 voxLocal  = UnpackVoxPosCS(v.pck);
    float3 voxCenter = gChunkBase + voxLocal + 0.5;

    uint faceAxis = inVox / 6u;             // 0=X 1=Y 2=Z
    uint inFace   = inVox - faceAxis * 6u;  // 0..5 within the face
    // Two tris as 6 verts: (0,1,2)(0,2,3) over quad corners 0..3.
    uint qcLut[6] = { 0u, 1u, 2u, 0u, 2u, 3u };
    uint qc = qcLut[inFace];
    float2 uv = float2((qc == 1u || qc == 2u) ? 1.0 : 0.0,
                       (qc == 2u || qc == 3u) ? 1.0 : 0.0);

    // Pick face sign from camera position along the face axis.
    float3 toCam = gCamPos - voxCenter;
    float axDist = (faceAxis == 0u) ? toCam.x
                 : (faceAxis == 1u) ? toCam.y : toCam.z;
    float sgn = (axDist >= 0.0) ? 1.0 : -1.0;

    // visMask bit order: 0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z. Stored in point col.a.
    uint visMask = (v.col >> 24u) & 0x3Fu;
    uint visBit  = faceAxis * 2u + ((sgn >= 0.0) ? 0u : 1u);
    bool visible = ((visMask >> visBit) & 1u) != 0u;

    // Build the corner offset: face axis = sgn*0.5; the two in-plane axes
    // get (uv - 0.5). Flip U sign on negative faces so winding stays outward.
    uint uAx = (faceAxis + 1u) % 3u;
    uint vAx = (faceAxis + 2u) % 3u;
    float uOff = uv.x - 0.5;
    float vOff = uv.y - 0.5;
    if (sgn < 0.0) uOff = -uOff;

    float3 offset = float3(0.0, 0.0, 0.0);
    if (faceAxis == 0u) offset.x = sgn * 0.5;
    if (faceAxis == 1u) offset.y = sgn * 0.5;
    if (faceAxis == 2u) offset.z = sgn * 0.5;
    if (uAx == 0u) offset.x += uOff;
    else if (uAx == 1u) offset.y += uOff;
    else                offset.z += uOff;
    if (vAx == 0u) offset.x += vOff;
    else if (vAx == 1u) offset.y += vOff;
    else                offset.z += vOff;

    float3 world = voxCenter + offset;

    if (!visible) {
        // Push vertex off-screen so the triangle is clipped without rasterization.
        o.svpos = float4(2.0, 2.0, 2.0, 1.0);
        o.nrm   = float3(0, 1, 0);
        o.col   = float3(0, 0, 0);
        o.wpos  = world;
        o.shadowMask = 0;
        o.ao = 0;
        return o;
    }

    o.svpos = mul(float4(world, 1.0), gViewProj);
    float3 n = float3(0, 0, 0);
    if (faceAxis == 0u) n.x = sgn;
    else if (faceAxis == 1u) n.y = sgn;
    else                     n.z = sgn;
    o.nrm   = n;
    o.col   = float3((v.col >>  0u) & 0xFFu,
                     (v.col >>  8u) & 0xFFu,
                     (v.col >> 16u) & 0xFFu) / 255.0;
    o.wpos  = world;
    o.shadowMask = 0xFFu;
    // Per-face AO from gPolyAo6: 4-bit nibble per face indexed by visBit
    // (visBit = faceAxis*2 + (sgn<0)). 15 = fully lit, 0 = fully shadowed.
    uint ao6   = gPolyAo6[voxelIdx];
    uint nib   = (ao6 >> (visBit * 4u)) & 0xFu;
    o.ao       = (float)nib / 15.0;
    return o;
}

// Instanced variant of PolyAxis: DrawInstanced(18, count). SV_InstanceID picks
// voxel, SV_VertexID is 0..17 within the cube. Identical math.
VSPolyVidOut vsmain_polyaxis_instanced(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    VSPolyVidOut o;
    uint voxelIdx = iid + gVoxelBase;
    uint inVox    = vid;                          // 0..17
    VoxelP v = gPolyVoxels[voxelIdx];
    float3 voxLocal  = UnpackVoxPosCS(v.pck);
    float3 voxCenter = gChunkBase + voxLocal + 0.5;

    uint faceAxis = inVox / 6u;
    uint inFace   = inVox - faceAxis * 6u;
    uint qcLut[6] = { 0u, 1u, 2u, 0u, 2u, 3u };
    uint qc = qcLut[inFace];
    float2 uv = float2((qc == 1u || qc == 2u) ? 1.0 : 0.0,
                       (qc == 2u || qc == 3u) ? 1.0 : 0.0);

    float3 toCam = gCamPos - voxCenter;
    float axDist = (faceAxis == 0u) ? toCam.x
                 : (faceAxis == 1u) ? toCam.y : toCam.z;
    float sgn = (axDist >= 0.0) ? 1.0 : -1.0;

    uint visMask = (v.col >> 24u) & 0x3Fu;
    uint visBit  = faceAxis * 2u + ((sgn >= 0.0) ? 0u : 1u);
    bool visible = ((visMask >> visBit) & 1u) != 0u;

    uint uAx = (faceAxis + 1u) % 3u;
    uint vAx = (faceAxis + 2u) % 3u;
    float uOff = uv.x - 0.5;
    float vOff = uv.y - 0.5;
    if (sgn < 0.0) uOff = -uOff;

    float3 offset = float3(0.0, 0.0, 0.0);
    if (faceAxis == 0u) offset.x = sgn * 0.5;
    if (faceAxis == 1u) offset.y = sgn * 0.5;
    if (faceAxis == 2u) offset.z = sgn * 0.5;
    if (uAx == 0u) offset.x += uOff;
    else if (uAx == 1u) offset.y += uOff;
    else                offset.z += uOff;
    if (vAx == 0u) offset.x += vOff;
    else if (vAx == 1u) offset.y += vOff;
    else                offset.z += vOff;

    float3 world = voxCenter + offset;
    if (!visible) {
        o.svpos = float4(2.0, 2.0, 2.0, 1.0);
        o.nrm = float3(0, 1, 0); o.col = float3(0, 0, 0); o.wpos = world;
        o.shadowMask = 0; o.ao = 0;
        return o;
    }
    o.svpos = mul(float4(world, 1.0), gViewProj);
    float3 n = float3(0, 0, 0);
    if (faceAxis == 0u) n.x = sgn;
    else if (faceAxis == 1u) n.y = sgn;
    else                     n.z = sgn;
    o.nrm = n;
    o.col = float3((v.col >>  0u) & 0xFFu,
                   (v.col >>  8u) & 0xFFu,
                   (v.col >> 16u) & 0xFFu) / 255.0;
    o.wpos = world;
    o.shadowMask = 0xFFu;
    uint ao6 = gPolyAo6[voxelIdx];
    uint nib = (ao6 >> (visBit * 4u)) & 0xFu;
    o.ao = (float)nib / 15.0;
    return o;
}

float4 psmain_polyvid(VSPolyVidOut i) : SV_Target
{
    float3 n = i.nrm * i.nrm;
    n = n * n;
    n = n * n;
    n = normalize(i.nrm * n);
    int mode = (int)gMode;
    float fogDist = length(i.wpos - gCamPos);
    if (mode == 2) return float4(ApplyFog(n * 0.5 + 0.5, i.wpos), 1.0);
    if (mode == 1) return float4(ApplyFog(i.col, i.wpos), 1.0);
    if (mode == 3) return float4(ApplyFog(i.ao.xxx, i.wpos), 1.0);
    if (mode == 4) return float4(i.ao * ClusterTint(gChunkTint), 1.0);
    float3 amb  = SampleAmbientCubeTriplanar(n) * i.ao;
    float3 light = ApplyShadowLighting(amb, normalize(gLightDir), n, ShadowBitN(i.shadowMask, n) * SampleShadow(i.wpos));
    return float4(TonemapTinted(ApplyFog(i.col * light, i.wpos)), 1.0);
}

// Same lighting as psmain_polyvid but alpha=0 — used when PolyAxis / PolyVID
// draws into the splat RT (stencil=1 marks poly pixels, alpha=0 tells the
// reconstruction PS to skip these during the neighbor splat search).
float4 psmain_polyvid_alpha0(VSPolyVidOut i) : SV_Target
{
    float3 n = i.nrm * i.nrm;
    n = n * n;
    n = n * n;
    n = normalize(i.nrm * n);
    float3 amb  = SampleAmbientCubeTriplanar(n) * i.ao;
    float3 light = ApplyShadowLighting(amb, normalize(gLightDir), n, ShadowBitN(i.shadowMask, n) * SampleShadow(i.wpos));
    return float4(TonemapTinted(ApplyFog(i.col * light, i.wpos)), 0.0);
}

// ---------------- TAA composite ----------------
// Inputs:
//   t4 = current frame scene color (post-jitter draw)
//   t5 = previous frame history (reprojected this frame)
//   t6 = current frame depth (R32_FLOAT view of D32 depth)
// Uses gPrevViewProj + gJitter (sub-pixel NDC offset of current proj).
Texture2D<float4> gTaaScene  : register(t4);
Texture2D<float4> gTaaHist   : register(t5);
Texture2D<float>  gTaaDepth  : register(t6);
SamplerState      gTaaSamp   : register(s0);

struct VTaaOut { float4 pos : SV_Position; float2 uv : UV; };

VTaaOut vsmain_taa(uint vid : SV_VertexID)
{
    // Fullscreen triangle: (-1,-1), (3,-1), (-1,3) covers screen.
    float2 p = float2((vid == 1u) ? 3.0 : -1.0,
                      (vid == 2u) ? 3.0 : -1.0);
    VTaaOut o;
    o.pos = float4(p, 0.0, 1.0);
    o.uv  = float2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return o;
}

float4 psmain_taa(VTaaOut i) : SV_Target
{
    int2 pix = (int2)i.pos.xy;
    int W = (int)gScreenSize.x;
    int H = (int)gScreenSize.y;
    // Resample current frame at +jitter UV offset: undoes the projection jitter
    // so the output pixel represents the world point at the (un-jittered) pixel
    // center -> consecutive frames produce the same value on static scenes.
    float2 currUv = (float2(pix) + 0.5) / float2((float)W, (float)H);
    currUv += float2(gJitter.x * 0.5, -gJitter.y * 0.5);
    float4 curSample = gTaaScene.SampleLevel(gTaaSamp, currUv, 0);
    float3 curC = curSample.rgb;
    // Carry the scene-RT alpha through TAA: post pass uses alpha < 0.5 to
    // detect sky pixels (alpha 0 = empty bg, 1 = scene draw). Without this
    // every TAA output had alpha = 1 -> sky shader killed.
    float outAlpha = curSample.a;
    // Read depth at the SAME pixel as color (the jittered sample), not at
    // integer pix. Otherwise depth + color describe two different world
    // points and reprojection wobbles at depth discontinuities (= ghosting).
    int2 jPix = clamp(int2(currUv * float2((float)W, (float)H)),
                      int2(0, 0), int2(W - 1, H - 1));
    float  d  = gTaaDepth.Load(int3(jPix, 0));
    if (d <= 0.0) {
        return float4(curC, outAlpha);
    }

    // World pos must reconstruct from the SAME UV that depth+color were read
    // at (currUv = pix center + jitter). Using un-jittered pix NDC here was
    // the ghosting source: the world point reprojected into prev frame missed
    // the actual surface by ~half a pixel each frame.
    float viewZ = gNearZ / d;
    float aspect = (float)W / (float)H;
    float ndcX = currUv.x * 2.0 - 1.0;
    float ndcY = 1.0 - currUv.y * 2.0;
    float viewX = ndcX * aspect * gTanHalfFovY * viewZ;
    float viewY = ndcY *          gTanHalfFovY * viewZ;
    float3 world = gCamPos + gCamRight * viewX + gCamUp * viewY + gCamForward * viewZ;

    // Reproject to previous frame.
    float4 prevClip = mul(float4(world, 1.0), gPrevViewProj);
    if (prevClip.w <= 0.0) return float4(curC, outAlpha);
    float3 prevNdc = prevClip.xyz / prevClip.w;
    if (any(abs(prevNdc.xy) > 1.0)) return float4(curC, outAlpha);
    float2 prevUv = float2(prevNdc.x * 0.5 + 0.5, 0.5 - prevNdc.y * 0.5);

    // Catmull-Rom 5-tap (Karis) - bicubic resample via 5 bilinear fetches.
    // Sharper than plain bilinear; standard TAA history reconstruction.
    float2 texSize = float2((float)W, (float)H);
    float2 sp = prevUv * texSize;
    float2 tp1 = floor(sp - 0.5) + 0.5;
    float2 f = sp - tp1;
    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);
    float2 w12 = w1 + w2;
    float2 off12 = w2 / max(w12, 1e-5);
    float2 tp0  = (tp1 - 1.0) / texSize;
    float2 tp3  = (tp1 + 2.0) / texSize;
    float2 tp12 = (tp1 + off12) / texSize;

    float k0 = w12.x * w0.y;
    float k1 = w0.x  * w12.y;
    float k2 = w12.x * w12.y;
    float k3 = w3.x  * w12.y;
    float k4 = w12.x * w3.y;
    float kSum = max(k0 + k1 + k2 + k3 + k4, 1e-5);
    float3 prevC = float3(0, 0, 0);
    prevC += gTaaHist.SampleLevel(gTaaSamp, float2(tp12.x, tp0.y ), 0).rgb * k0;
    prevC += gTaaHist.SampleLevel(gTaaSamp, float2(tp0.x,  tp12.y), 0).rgb * k1;
    prevC += gTaaHist.SampleLevel(gTaaSamp, float2(tp12.x, tp12.y), 0).rgb * k2;
    prevC += gTaaHist.SampleLevel(gTaaSamp, float2(tp3.x,  tp12.y), 0).rgb * k3;
    prevC += gTaaHist.SampleLevel(gTaaSamp, float2(tp12.x, tp3.y ), 0).rgb * k4;
    prevC /= kSum;       // re-normalize (5-tap drops 4 corners)

    // Neighborhood color clamp: history outside the 3x3 current-frame AABB is
    // a disocclusion / moving edge -> ghost. Compute min/max around currUv,
    // expand slightly via variance for stability, then clip prevC to that box.
    float2 px = 1.0 / float2((float)W, (float)H);
    float3 nMin = curC;
    float3 nMax = curC;
    float3 nMean = curC;
    float3 nMean2 = curC * curC;
    [unroll] for (int dy = -1; dy <= 1; ++dy)
    [unroll] for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) continue;
        float3 s = gTaaScene.SampleLevel(gTaaSamp, currUv + float2(dx, dy) * px, 0).rgb;
        nMin = min(nMin, s);
        nMax = max(nMax, s);
        nMean  += s;
        nMean2 += s * s;
    }
    nMean  /= 9.0;
    nMean2 /= 9.0;
    float3 sigma = sqrt(max(nMean2 - nMean * nMean, 0.0));
    // Blend pure min/max with mean +/- gamma*sigma; gamma controls strictness.
    const float gamma = 1.25;
    float3 vMin = max(nMin, nMean - gamma * sigma);
    float3 vMax = min(nMax, nMean + gamma * sigma);
    float3 clipped = clamp(prevC, vMin, vMax);

    // Adaptive blend: more current when clip moved the history a lot (disocclusion).
    float dist = length(clipped - prevC) / max(length(nMax - nMin), 1e-4);
    float alpha = lerp(0.1, 0.5, saturate(dist));
    float3 outC = lerp(clipped, curC, alpha);
    return float4(outC, outAlpha);
}

// Post-process: sky for empty pixels + sharpen + tonemap. Reads TAA history
// (R16) + scene depth (R32_FLOAT view of D32) -> backbuf.
Texture2D<float>  gPostDepth : register(t6);
Texture2D<float4> gPostIn    : register(t7);

// Local helper: world-ray dir for a pixel (mirrors splat-CS version which is
// defined later in the file).
float3 PostPixelWorldDir(int2 pix, int W, int H)
{
    float ndcX = ((float)pix.x + 0.5) / (float)W * 2.0 - 1.0;
    float ndcY = 1.0 - ((float)pix.y + 0.5) / (float)H * 2.0;
    float aspect = (float)W / (float)H;
    float3 v = float3(ndcX * aspect * gTanHalfFovY, ndcY * gTanHalfFovY, 1.0);
    return normalize(gCamRight * v.x + gCamUp * v.y + gCamForward * v.z);
}

float3 SkyColor(float3 rd)
{
    float3 sunDir = normalize(gLightDir);
    float  t = saturate(rd.y * 0.5 + 0.5);
    float3 horizon = float3(0.70, 0.75, 0.85);
    float3 zenith  = float3(0.20, 0.40, 0.80);
    float3 ground  = gFogColor;        // lower hemisphere matches fog
    float3 above = lerp(horizon, zenith, smoothstep(0.5, 1.0, t));
    float3 sky   = lerp(ground, above, smoothstep(0.48, 0.52, t));
    float sunDot = max(0.0, dot(rd, sunDir));
    float disc   = smoothstep(0.9990, 0.9996, sunDot);
    float glow   = pow(sunDot, 6.0) * 0.6;
    sky += float3(1.10, 0.95, 0.75) * (disc + glow);
    return sky;
}

float3 ACES(float3 x)
{
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

float3 PhotoTonemap(float3 x)
{
    return 1.0 - exp(-x * 0.5 );
}

float4 psmain_post(VTaaOut i) : SV_Target
{
    int2 pix = (int2)i.pos.xy;
    int W = (int)gScreenSize.x;
    int H = (int)gScreenSize.y;

    // Sky detection: scene RT is cleared with alpha=0; scene PS writes alpha=1.
    float4 inC = gPostIn.Load(int3(pix, 0));
    float3 c;
    if (inC.a < 0.5) {
        float3 rd = PostPixelWorldDir(pix, W, H);
        c = SkyColor(rd);
    } else {
        c = inC.rgb;
        int2 pL = int2(max(pix.x - 1, 0),     pix.y);
        int2 pR = int2(min(pix.x + 1, W - 1), pix.y);
        int2 pU = int2(pix.x, max(pix.y - 1, 0));
        int2 pD = int2(pix.x, min(pix.y + 1, H - 1));
        float3 avg = (gPostIn.Load(int3(pL, 0)).rgb + gPostIn.Load(int3(pR, 0)).rgb +
                      gPostIn.Load(int3(pU, 0)).rgb + gPostIn.Load(int3(pD, 0)).rgb) * 0.25;
        c = c + 0.5 * (c - avg);   // unsharp mask
    }
    return float4(c, 1.0);
}

// ---------------- Splat technique ----------------
// Splat RT alpha encoding:
//   bit  7      = marker (always 1 for a splat pixel; 0 = bg)
//   bits 6:5    = lodIdx (0..3 = L0..L3)
//   bits 4:1    = AO quantized (16 levels)
//   bit  0      = reserved
float SplatEncodeAlpha(float ao01, uint lodIdx)
{
    uint ao4 = (uint)(saturate(ao01) * 15.0 + 0.5);
    uint a8  = 0x80u | ((lodIdx & 3u) << 5) | ((ao4 & 0xFu) << 1);
    return (float)a8 / 255.0;
}
struct SplatPointOut {
    float4 color : SV_Target0;
    uint   mask  : SV_Target1;
};
SplatPointOut psmain_splat_albedo(VSPointOut i)
{
    SplatPointOut o;
    o.color = float4(i.col, SplatEncodeAlpha(i.ao, gChunkLodIdx));
    o.mask  = i.mask & 0x3Fu;
    return o;
}
// CS reconstruction: for each output pixel, search neighbors for the nearest
// point whose splat sphere covers it; use its color (background if none).
Texture2D<float>    gSplatDepth    : register(t1);
Texture2D<float4>   gSplatColorSrv : register(t2);
Texture2D<uint>     gSplatMaskSrv  : register(t3);
RWTexture2D<float4> gSplatFinalUav      : register(u1);
RWTexture2D<float>  gSplatFinalDepthUav : register(u2);

// Reconstruct world ray for pixel center using the camera basis (avoids the
// numerically-fragile inverse view-proj matrix).
float3 PixelWorldDir(int2 pix, int W, int H)
{
    float2 ndc;
    ndc.x = ((float)pix.x + 0.5) / (float)W * 2.0 - 1.0;
    ndc.y = 1.0 - ((float)pix.y + 0.5) / (float)H * 2.0;
    float aspect = (float)W / (float)H;
    float3 viewDir = float3(ndc.x * aspect * gTanHalfFovY,
                            ndc.y           * gTanHalfFovY,
                            1.0);
    return normalize(gCamRight   * viewDir.x +
                     gCamUp      * viewDir.y +
                     gCamForward * viewDir.z);
}

// Fast world-space reconstruction from depth (reverse-Z infinite-far).
//   viewZ = nearZ / depth                 (one recip)
//   viewX = ndc.x * aspect * tanFov * viewZ
//   viewY = ndc.y *          tanFov * viewZ
//   world = camPos + camRight*viewX + camUp*viewY + camForward*viewZ
float3 ReconstructNeighborWorld(int2 sp, float zN, int W, int H)
{
    float viewZ = gNearZ / zN;
    float aspect = (float)W / (float)H;
    // Splat rasterization uses jittered proj: pixel_ndc = clip_ndc + jitter.
    // So the world point that landed at pixel sp came from clip_ndc = pixel_ndc - jitter.
    // gJitter.y sign matches the proj inject (Y not flipped at inject site).
    float ndcX = ((float)sp.x + 0.5) / (float)W * 2.0 - 1.0 - gJitter.x;
    float ndcY = 1.0 - ((float)sp.y + 0.5) / (float)H * 2.0 - gJitter.y;
    float viewX = ndcX * aspect * gTanHalfFovY * viewZ;
    float viewY = ndcY *          gTanHalfFovY * viewZ;
    return gCamPos + gCamRight * viewX + gCamUp * viewY + gCamForward * viewZ;
}

// Fullscreen reconstruction PS (replaces csmain_splat). Reads splatColorSrv
// (t2), splatDepthSrv (t1), splatStencilSrv (t9). Stencil==1 -> poly pixel,
// output passthrough. Stencil==0 -> search neighbor splat markers in alpha.
Texture2D<uint2> gSplatStencil : register(t9);

struct ReconOut { float4 color : SV_Target; float depth : SV_Depth; };
ReconOut psmain_splat_reconstruct(VBlitOut vIn)
{
    ReconOut o;
    int2 pix = int2(vIn.pos.xy);
    int W = (int)gScreenSize.x;
    int H = (int)gScreenSize.y;

    // Propagate splat-pipeline depth into the main depth buffer so TAA's
    // reprojection + post-pass sky have the right Z everywhere.
    o.depth = gSplatDepth.Load(int3(pix, 0));

    // Poly pixel: pass through.
    uint stencil = gSplatStencil.Load(int3(pix, 0)).g;
    if (stencil == 1u) {
        o.color = float4(gSplatColorSrv.Load(int3(pix, 0)).rgb, 1.0);
        return o;
    }

    float3 ro = gCamPos;
    float3 rd = PixelWorldDir(pix, W, H);
    float3 invRd = 1.0 / rd;

    int R = (int)max(1.0, _pad1.x);
    float  bestT = 1e30;
    float3 bestAlbedo = float3(0, 0, 0);
    float3 bestN = float3(0, 0, 1);
    float  bestAo = 1.0;
    uint   bestLodIdx = 0u;
    bool   anyHit = false;

    [loop] for (int dy = -R; dy <= R; ++dy) {
        [loop] for (int dx = -R; dx <= R; ++dx) {
            int2 sp = pix + int2(dx, dy);
            if (sp.x < 0 || sp.x >= W || sp.y < 0 || sp.y >= H) continue;
            float4 s = gSplatColorSrv.Load(int3(sp, 0));
            uint a8 = (uint)(s.a * 255.0 + 0.5);
            if (a8 == 0u) continue;            // poly (stencil=1) or pure bg
            if ((a8 & 0x80u) == 0u) continue;
            uint lodIdx = (a8 >> 5) & 3u;
            uint ao4 = (a8 >> 1) & 0xFu;
            float aoN = (float)ao4 / 15.0;
            float halfExt = (lodIdx == 0u) ? 0.5
                          : (lodIdx == 1u) ? 1.0
                          : (lodIdx == 2u) ? 2.0
                          :                  4.0;
            float zN = gSplatDepth.Load(int3(sp, 0));
            if (zN <= 0.0) continue;
            float3 wp = ReconstructNeighborWorld(sp, zN, W, H);
            float S = 2.0 * halfExt;
            float3 vmin = floor(wp / S) * S;
            float3 vmax = vmin + S;
            float3 t0v = (vmin - ro) * invRd;
            float3 t1v = (vmax - ro) * invRd;
            float3 tmn = min(t0v, t1v);
            float3 tmx = max(t0v, t1v);
            float tNear = max(max(tmn.x, tmn.y), tmn.z);
            float tFar  = min(min(tmx.x, tmx.y), tmx.z);
            if (tFar < 0.0 || tNear > tFar) continue;
            float tHit = max(tNear, 0.0);
            if (tHit < bestT) {
                float3 hit = ro + rd * tHit;
                float3 center = (vmin + vmax) * 0.5;
                float3 d = hit - center;
                float3 absD = abs(d);
                float maxC = max(max(absD.x, absD.y), absD.z);
                float3 n = float3(0, 0, 0);
                if      (absD.x >= maxC - 1e-3) n.x = d.x >= 0 ? 1.0 : -1.0;
                else if (absD.y >= maxC - 1e-3) n.y = d.y >= 0 ? 1.0 : -1.0;
                else                            n.z = d.z >= 0 ? 1.0 : -1.0;
                bestT = tHit;
                bestAlbedo = s.rgb;
                bestN = n;
                bestAo = aoN;
                bestLodIdx = lodIdx;
                anyHit = true;
            }
        }
    }

    if (anyHit) {
        float3 hit = ro + rd * bestT;
        float3 amb = SampleAmbientCubeTriplanar(bestN) * bestAo;
        float3 light = ApplyShadowLighting(amb, normalize(gLightDir), bestN, SampleShadow(hit));
        float3 lit = bestAlbedo * light;
        float3 splatTint = (gColorizeClusters > 0.5) ? ClusterTint(2u + bestLodIdx) : float3(1.0, 1.0, 1.0);
        o.color = float4(Tonemap(ApplyFog(lit * splatTint, hit)), 1.0);
        // Use reprojected hit depth (more accurate than the neighbor pixel's
        // splat depth, which is one of the source markers).
        float4 clipHit = mul(float4(hit, 1.0), gViewProj);
        o.depth = saturate(clipHit.z / max(clipHit.w, 1e-6));
        return o;
    }
    // Pure background: signal alpha=0 so the post pass / sky takes over.
    o.color = float4(0.0, 0.0, 0.0, 0.0);
    return o;
}

[numthreads(8, 8, 1)]
void csmain_splat(uint3 dt : SV_DispatchThreadID)
{
    int2 pix = (int2)dt.xy;
    int W = (int)gScreenSize.x;
    int H = (int)gScreenSize.y;
    if (pix.x >= W || pix.y >= H) return;

    // Camera ray for the current pixel.
    float3 ro = gCamPos;
    float3 rd = PixelWorldDir(pix, W, H);
    float3 invRd = 1.0 / rd;

    int R = (int)max(1.0, _pad1.x);
    float  bestT      = 1e30;
    float3 bestAlbedo = float3(0, 0, 0);
    float3 bestN      = float3(0, 0, 1);
    uint   bestMask   = 0x3Fu;
    bool   anyHit     = false;

    float bestAo = 1.0;
    uint  bestLodIdx = 0u;
    uint  bestParity = 0u;
    // Depth-only fallback: nearest-Z valid neighbor. Doesn't pollute color
    // (no color is taken from this), only fills depth at pixels whose ray
    // missed every AABB. Keeps depth buffer contiguous for TAA / sky / DOF.
    bool  fbHave = false;
    float fbZ    = -1.0;        // reverse-Z: larger = nearer
    [loop] for (int dy = -R; dy <= R; ++dy) {
        [loop] for (int dx = -R; dx <= R; ++dx) {
            int2 sp = pix + int2(dx, dy);
            if (sp.x < 0 || sp.x >= W || sp.y < 0 || sp.y >= H) continue;
            float4 s = gSplatColorSrv.Load(int3(sp, 0));
            uint a8 = (uint)(s.a * 255.0 + 0.5);
            if (a8 == 0u) continue;                // poly pixel or untouched bg
            if ((a8 & 0x80u) == 0u) continue;
            uint lodIdx = (a8 >> 5) & 3u;
            uint ao4    = (a8 >> 1) & 0xFu;
            uint parityN = a8 & 1u;
            float aoN   = (float)ao4 / 15.0;
            float halfExt = (lodIdx == 0u) ? 0.5
                          : (lodIdx == 1u) ? 1.0
                          : (lodIdx == 2u) ? 2.0
                          :                  4.0;
            float zN = gSplatDepth.Load(int3(sp, 0));
            if (zN <= 0.0) continue;
            if (zN > fbZ) { fbZ = zN; fbHave = true; }
            uint visMaskN = gSplatMaskSrv.Load(int3(sp, 0)) & 0x3Fu;
            float3 wp = ReconstructNeighborWorld(sp, zN, W, H);
            // Snap to LOD-aligned voxel grid so adjacent splats from the same
            // cluster collapse to the same AABB. Without this, neighbor pixels
            // produce offset overlapping cubes -> visible cube-edge seams on
            // flat surfaces.
            float S = 2.0 * halfExt;
            float3 vmin = floor(wp / S) * S;
            float3 vmax = vmin + S;
            float3 t0v = (vmin - ro) * invRd;
            float3 t1v = (vmax - ro) * invRd;
            float3 tmn = min(t0v, t1v);
            float3 tmx = max(t0v, t1v);
            float tNear = max(max(tmn.x, tmn.y), tmn.z);
            float tFar  = min(min(tmx.x, tmx.y), tmx.z);
            if (tFar < 0.0 || tNear > tFar) continue;
            float tHit = max(tNear, 0.0);
            if (tHit < bestT) {
                float3 hit = ro + rd * tHit;
                float3 center = (vmin + vmax) * 0.5;
                float3 d = hit - center;
                // Pick face: dominant axis of (hit - center), but restricted to
                // faces actually present in the voxel's visMask. Without the
                // mask, edge/corner hits flip between axes pixel-to-pixel where
                // |d.x| ~= |d.y| ~= |d.z| -> normal noise in lighting.
                float3 absD = abs(d);
                float bestProj = -1.0;
                float3 n = float3(0, 1, 0);
                [unroll] for (uint fi = 0u; fi < 6u; ++fi) {
                    if (((visMaskN >> fi) & 1u) == 0u) continue;
                    float3 fn = float3(0, 0, 0);
                    if      (fi == 0u) fn = float3( 1, 0, 0);
                    else if (fi == 1u) fn = float3(-1, 0, 0);
                    else if (fi == 2u) fn = float3( 0, 1, 0);
                    else if (fi == 3u) fn = float3( 0,-1, 0);
                    else if (fi == 4u) fn = float3( 0, 0, 1);
                    else               fn = float3( 0, 0,-1);
                    // Project (hit-center) onto face normal: larger = closer to that face.
                    float p = dot(d, fn);
                    if (p > bestProj) { bestProj = p; n = fn; }
                }
                bestT      = tHit;
                bestAlbedo = s.rgb;
                bestN      = n;
                bestMask   = visMaskN;
                bestAo     = aoN;
                bestLodIdx = lodIdx;
                bestParity = parityN;
                anyHit     = true;
            }
        }
    }

    if (anyHit) {
        float3 hit = ro + rd * bestT;
        int mode = (int)gMode;
        float3 outRgb;
        if (mode == 2) {
            outRgb = ApplyFog(bestN * 0.5 + 0.5, hit);
        } else if (mode == 1) {
            outRgb = ApplyFog(bestAlbedo, hit);
        } else if (mode == 3) {
            outRgb = ApplyFog(bestAo.xxx, hit);
        } else if (mode == 4) {
            // AO * per-LOD tint, with per-cluster checker (parity bit from alpha).
            float check = (bestParity == 0u) ? 0.55 : 1.00;
            outRgb = bestAo * ClusterTint(2u + bestLodIdx) * check;
        } else {
            float3 amb = SampleAmbientCubeTriplanar(bestN) * bestAo;
            float3 light = ApplyShadowLighting(amb, normalize(gLightDir), bestN, SampleShadow(hit));
            float3 lit = bestAlbedo * light;
            float3 splatTint = (gColorizeClusters > 0.5) ? ClusterTint(2u + bestLodIdx) : float3(1.0, 1.0, 1.0);
            outRgb = Tonemap(ApplyFog(lit * splatTint, hit));
        }
        gSplatFinalUav[pix] = float4(outRgb, 1.0);
        // Reproject winning hit -> clip depth so composite PS emits SV_Depth
        // consistent with the dilated color (not the un-dilated source depth,
        // which is 0 at filled-in pixels).
        float4 clipHit = mul(float4(hit, 1.0), gViewProj);
        gSplatFinalDepthUav[pix] = saturate(clipHit.z / max(clipHit.w, 1e-6));
    } else {
        // No AABB hit. Center pixel may still own a splat marker — light flat
        // up so far/empty regions stay coherent with the rest of the scene.
        float4 c0 = gSplatColorSrv.Load(int3(pix, 0));
        uint a8 = (uint)(c0.a * 255.0 + 0.5);
        if ((a8 & 0x80u) != 0u) {
            float3 n = float3(0, 1, 0);
            float3 amb = SampleAmbientCubeTriplanar(n);
            float3 light = ApplyShadowLighting(amb, normalize(gLightDir), n, 1.0);
            float3 wp = ReconstructNeighborWorld(pix, max(gSplatDepth.Load(int3(pix, 0)), 1e-6), W, H);
            gSplatFinalUav[pix] = float4(Tonemap(ApplyFog(c0.rgb * light, wp)), 1.0);
            gSplatFinalDepthUav[pix] = gSplatDepth.Load(int3(pix, 0));
        } else {
            // Pure background pixel for color (alpha = 0). Depth: write the
            // nearest-Z valid neighbor seen during dilation if any, so the
            // composite PS can emit SV_Depth even where no color was filled.
            gSplatFinalUav[pix] = float4(c0.rgb, 0.0);
            gSplatFinalDepthUav[pix] = fbHave ? fbZ : 0.0;
        }
    }
}

// ---------------- Second-pass dilate (hole fill) ----------------
// Reads pass-1 output (color in gSplatColorSrv, depth in gSplatDepth) and
// fills any remaining holes (alpha == 0) by sampling the nearest filled
// neighbour inside a small kernel. Already-filled pixels pass through.
// Writes to gSplatFinalUav / gSplatFinalDepthUav (pass-2 ping-pong target).
[numthreads(8, 8, 1)]
void csmain_splat_fill(uint3 dt : SV_DispatchThreadID)
{
    int2 pix = (int2)dt.xy;
    int W = (int)gScreenSize.x;
    int H = (int)gScreenSize.y;
    if (pix.x >= W || pix.y >= H) return;

    float4 c0 = gSplatColorSrv.Load(int3(pix, 0));
    float  z0 = gSplatDepth.Load(int3(pix, 0));
    if (c0.a > 0.5) {
        // Already filled by pass 1 — pass through unchanged.
        gSplatFinalUav[pix]      = c0;
        gSplatFinalDepthUav[pix] = z0;
        return;
    }

    // Hole. Scan small kernel for nearest filled neighbour by reverse-Z.
    int R = max(1, (int)_pad1.x);
    float bestZ = -1.0;
    float4 bestC = float4(0, 0, 0, 0);
    [loop] for (int dy = -R; dy <= R; ++dy) {
        [loop] for (int dx = -R; dx <= R; ++dx) {
            int2 sp = pix + int2(dx, dy);
            if (sp.x < 0 || sp.x >= W || sp.y < 0 || sp.y >= H) continue;
            float4 sc = gSplatColorSrv.Load(int3(sp, 0));
            if (sc.a <= 0.5) continue;
            float sz = gSplatDepth.Load(int3(sp, 0));
            if (sz > bestZ) { bestZ = sz; bestC = sc; }
        }
    }
    if (bestZ <= 0.0) {
        // Still nothing — leave as bg (alpha 0, depth 0). Composite skips.
        gSplatFinalUav[pix]      = float4(0, 0, 0, 0);
        gSplatFinalDepthUav[pix] = 0.0;
    } else {
        gSplatFinalUav[pix]      = bestC;
        gSplatFinalDepthUav[pix] = bestZ;
    }
}

// Composite splat output over the main RT. Discards background pixels.
// Also writes SV_Depth so TAA reprojection + post sky have valid scene depth
// at splat pixels (without this, splat-covered pixels read main depth = 0
// (far) every frame, jitter sampling adjacent splats -> heavy wobble).
Texture2D<float4> gSplatComposite : register(t8);
Texture2D<float>  gSplatCompositeDepth : register(t7);
struct CompositeOut { float4 color : SV_Target; float depth : SV_Depth; };
CompositeOut psmain_splat_composite(VBlitOut i)
{
    CompositeOut o;
    int2 pix = int2(i.pos.xy);
    float4 c = gSplatComposite.Load(int3(pix, 0));
    float  z = gSplatCompositeDepth.Load(int3(pix, 0));
    // Color: alpha=0 -> blend state preserves main RT (no discard, so SV_Depth
    // still fires). Alpha=1 -> src wins. Depth: written unconditionally if z>0
    // (CS wrote either a real hit or a nearest-neighbor fallback). Zero-depth
    // pixels are true bg / outside any dilation reach -> let main DSV stand.
    o.color = c;
    if (z <= 0.0) { o.depth = 0.0; discard; } // genuine bg: keep main RT + DSV
    o.depth = z;
    return o;
}

// ---------------- Billboard technique ----------------
// 1 voxel = 1 quad (4 verts) in screen space covering voxel's NDC bounding rect.
// PS performs ray-vs-AABB intersection to find the hit face + true depth.
struct VSBillOut {
    float4 svpos : SV_Position;
    nointerpolation float3 boxMin : BMIN;
    nointerpolation float3 boxMax : BMAX;
    nointerpolation float3 col    : COL;
    nointerpolation uint shadowMask : SHADOW;
    nointerpolation float ao : AO;
};

// Shared code for both billboard variants: emits one vertex of an axis-aligned
// figure on a face-camera plane at the front of the voxel's bounding sphere.
// `s` is the 2D offset in (voxRight, voxUp) space, in units of r.
VSBillOut BillboardVertex(uint voxelIdx, float2 s)
{
    VoxelP v = gPolyVoxels[voxelIdx];
    float3 voxLocal = UnpackVoxPosCS(v.pck);
    float3 boxMin = gChunkBase + voxLocal;
    float3 boxMax = boxMin + 1.0;
    float3 center = boxMin + 0.5;

    const float r = 0.8660254;     // sqrt(3)/2, unit-cube bounding-sphere radius
    float3 toCam = normalize(gCamPos - center);
    float3 voxRight = normalize(cross(float3(0.0, 1.0, 0.0), toCam));
    float3 voxUp    = cross(toCam, voxRight);
    float3 quadCenter = center + toCam * r;
    float3 world = quadCenter + (voxRight * s.x + voxUp * s.y) * r;

    VSBillOut o;
    o.svpos  = mul(float4(world, 1.0), gViewProj);
    o.boxMin = boxMin;
    o.boxMax = boxMax;
    o.col    = float3((v.col >>  0u) & 0xFFu,
                      (v.col >>  8u) & 0xFFu,
                      (v.col >> 16u) & 0xFFu) / 255.0;
    o.shadowMask = 0xFFu;
    o.ao         = AoFromAux(v.pck.y >> 16u);
    return o;
}

VSBillOut vsmain_billboard(uint vid : SV_VertexID)
{
    uint voxelLocal = vid / 4u;
    uint quadCorner = vid & 3u;
    float2 quadLut[4] = {
        float2(-1.0, -1.0), float2( 1.0, -1.0),
        float2( 1.0,  1.0), float2(-1.0,  1.0)
    };
    return BillboardVertex(voxelLocal + gVoxelBase, quadLut[quadCorner]);
}

VSBillOut vsmain_billboard_tri(uint vid : SV_VertexID)
{
    uint voxelLocal = vid / 3u;
    uint triCorner  = vid - voxelLocal * 3u;
    // Equilateral triangle circumscribing the unit circle of inscribed
    // radius 1. Vertices at distance 2 from origin (R = 2 * r_in).
    float2 triLut[3] = {
        float2( 0.0,        2.0),
        float2(-1.7320508, -1.0),
        float2( 1.7320508, -1.0),
    };
    return BillboardVertex(voxelLocal + gVoxelBase, triLut[triCorner]);
}

// SV_SampleIndex input forces per-sample shading so MSAA edge coverage works
// correctly when we also output SV_Depth.
float4 psmain_billboard(VSBillOut i,
                        uint sampleIdx : SV_SampleIndex,
                        out float depthOut : SV_Depth) : SV_Target
{
    depthOut = 0.0;   // miss -> far in reverse-Z (will be discarded anyway)

    // Reconstruct world ray directly from camera basis (no matrix inverse).
    float2 pix = i.svpos.xy;
    float2 ndc;
    ndc.x = (pix.x / gScreenSize.x) * 2.0 - 1.0;
    ndc.y = 1.0 - (pix.y / gScreenSize.y) * 2.0;
    float  aspect = gScreenSize.x / gScreenSize.y;
    float3 viewDir = float3(ndc.x * aspect * gTanHalfFovY,
                            ndc.y           * gTanHalfFovY,
                            1.0);
    float3 ro = gCamPos;
    float3 rd = normalize(gCamRight   * viewDir.x +
                          gCamUp      * viewDir.y +
                          gCamForward * viewDir.z);

    // Ray-vs-AABB slabs.
    float3 invRd = 1.0 / rd;
    float3 t0 = (i.boxMin - ro) * invRd;
    float3 t1 = (i.boxMax - ro) * invRd;
    float3 tmin3 = min(t0, t1);
    float3 tmax3 = max(t0, t1);
    float tEnter = max(max(tmin3.x, tmin3.y), tmin3.z);
    float tExit  = min(min(tmax3.x, tmax3.y), tmax3.z);

    if (tExit < max(tEnter, 0.0)) { discard; return float4(0,0,0,0); }
    float t = max(tEnter, 0.0);

    float3 hit = ro + t * rd;

    // Pick entry-face normal from which axis dominated tmin3.
    float3 n;
    if (tmin3.x >= tmin3.y && tmin3.x >= tmin3.z)      n = float3(rd.x < 0.0 ? 1.0 : -1.0, 0, 0);
    else if (tmin3.y >= tmin3.z)                       n = float3(0, rd.y < 0.0 ? 1.0 : -1.0, 0);
    else                                               n = float3(0, 0, rd.z < 0.0 ? 1.0 : -1.0);

    // True per-pixel depth from ray hit.
    float4 clipHit = mul(float4(hit, 1.0), gViewProj);
    depthOut = saturate(clipHit.z / max(clipHit.w, 1e-6));

    int mode = (int)gMode;
    if (mode == 2) return float4(ApplyFog(n * 0.5 + 0.5, hit), 1.0);
    if (mode == 1) return float4(ApplyFog(i.col, hit), 1.0);
    if (mode == 3) return float4(ApplyFog(i.ao.xxx, hit), 1.0);
    if (mode == 4) return float4(i.ao * ClusterTint(gChunkTint), 1.0);

    float3 amb  = SampleAmbientCubeTriplanar(n) * i.ao;
    float3 light = ApplyShadowLighting(amb, normalize(gLightDir), n, ShadowBitN(i.shadowMask, n) * SampleShadow(hit));
    return float4(TonemapTinted(ApplyFog(i.col * light, hit)), 1.0);
}

// ---------------- HexSprite technique (DISABLED) ----------------
// Kept commented out for reference. Removed from the active tech set because
// it was the only path that needed a duplicate vertex buffer alongside the
// structured SRV — dropping it lets us run all point-techs through one buffer.
// To revive: uncomment this block, restore the HexSprite enum entry + draw
// block in renderer.cpp, and re-add the inputLayoutHex_ path.
#if 0
// Per voxel: 18 vertices = 6 triangles fanning from closest-corner-to-camera.
// Voxel data delivered as per-instance attributes; SV_VertexID 0..17 picks
// the triangle (0..5) and the role (center, v1, v2).
struct VSHexIn {
    uint4  inst_pck : POSITION;   // scene-relative pos (16-bit per axis)
    uint4  inst_col : COLOR;
    uint   vid : SV_VertexID;
};
struct VSHexOut {
    float4 svpos : SV_Position;
    float3 wpos  : WPOS;
    float3 col   : COL;
    nointerpolation uint shadowMask : SHADOW;
    nointerpolation float ao : AO;
};

VSHexOut vsmain_hex(VSHexIn i)
{
    uint tri    = i.vid / 3;          // 0..5
    uint role   = i.vid % 3;          // 0=center, 1=outer[tri], 2=outer[(tri+1)%6]
    uint cornerID;
    if      (role == 0) cornerID = 0;
    else if (role == 1) cornerID = 1 + tri;
    else                cornerID = 1 + ((tri + 1) % 6);

    float3 voxLocal  = UnpackVoxPos(i.inst_pck);
    float3 voxCenter = gChunkBase + voxLocal + 0.5;
    float3 D = gCamPos - voxCenter;
    float3 sgn;
    sgn.x = D.x >= 0.0 ? 1.0 : -1.0;
    sgn.y = D.y >= 0.0 ? 1.0 : -1.0;
    sgn.z = D.z >= 0.0 ? 1.0 : -1.0;

    // Closest corner offset within voxel (0 or 1 per axis).
    float3 closest = (sgn + 1.0) * 0.5;

    // Outer-corner offsets (silhouette of cube from closest corner).
    // 1: -X edge ; 2: -X -Y ; 3: -Y ; 4: -Y -Z ; 5: -Z ; 6: -Z -X
    float3 ofs;
    if      (cornerID == 0) ofs = float3(0, 0, 0);
    else if (cornerID == 1) ofs = float3(-sgn.x, 0,      0);
    else if (cornerID == 2) ofs = float3(-sgn.x, -sgn.y, 0);
    else if (cornerID == 3) ofs = float3(0,      -sgn.y, 0);
    else if (cornerID == 4) ofs = float3(0,      -sgn.y, -sgn.z);
    else if (cornerID == 5) ofs = float3(0,      0,      -sgn.z);
    else                    ofs = float3(-sgn.x, 0,      -sgn.z);

    float3 world = gChunkBase + voxLocal + closest + ofs;

    VSHexOut o;
    o.svpos = mul(float4(world, 1.0), gViewProj);
    o.wpos  = world;
    o.col   = float3(i.inst_col.rgb) / 255.0;
    o.shadowMask = 0xFFu;
    o.ao         = AoFromAux(i.inst_pck.w);
    return o;
}

float4 psmain_hex(VSHexOut i) : SV_Target
{
    float3 dpx = ddx(i.wpos);
    float3 dpy = ddy(i.wpos);
    float3 n   = normalize(cross(dpx, dpy));

    int mode = (int)gMode;
    float fogDist = length(i.wpos - gCamPos);
    if (mode == 2) return float4(ApplyFog(n * 0.5 + 0.5, i.wpos), 1.0);
    if (mode == 1) return float4(ApplyFog(i.col, i.wpos), 1.0);
    if (mode == 3) return float4(ApplyFog(i.ao.xxx, i.wpos), 1.0);
    if (mode == 4) return float4(i.ao * ClusterTint(gChunkTint), 1.0);

    float3 amb  = SampleAmbientCubeTriplanar(n) * i.ao;
    float3 light = ApplyShadowLighting(amb, normalize(gLightDir), n, ShadowBitN(i.shadowMask, n) * SampleShadow(i.wpos));
    return float4(TonemapTinted(ApplyFog(i.col * light, i.wpos)), 1.0);
}
#endif // HexSprite disabled
