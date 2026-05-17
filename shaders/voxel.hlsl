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
    float    gFogDensity;       // 0 = fog disabled
};

float3 ApplyFog(float3 color, float dist)
{
    if (gFogDensity <= 0.0) return color;
    float t = exp(-dist * gFogDensity);     // in-scatter (object) weight
    return lerp(gFogColor, color, t);       // (1-t) of fog mixed in
}

cbuffer cbPerChunk : register(b1)
{
    float3 gChunkBase;
    float  _pad;
    uint   gVoxelBase;       // PolyVID: chunk's first voxel index in pointSb_
    uint3  _pad2;
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

struct VSIn  {
    uint4  pck : POSITION;   // .xyz = local 0..D, .w = faceIdx
    float4 col : COLOR;
};
struct VSOut {
    float4 svpos : SV_Position;
    float3 wpos  : WPOS;
    float4 col   : COL;
    nointerpolation uint faceIdx : FIDX;
};

// Depth-only VS (Z prepass). No PS bound.
struct VSDepthOut { float4 svpos : SV_Position; };
// `precise` forces bit-exact identical math in both VS variants so the
// prepass and main pass produce the same SV_Position -> no z-fighting.
VSDepthOut vsmain_depth(VSIn i)
{
    VSDepthOut o;
    precise float3 local = float3(i.pck.x, i.pck.y, i.pck.z);
    precise float3 world = gChunkBase + local;
    precise float4 clip  = mul(float4(world, 1.0), gViewProj);
    o.svpos = clip;
    return o;
}

VSOut vsmain(VSIn i)
{
    VSOut o;
    precise float3 local = float3(i.pck.x, i.pck.y, i.pck.z);
    precise float3 world = gChunkBase + local;
    precise float4 clip  = mul(float4(world, 1.0), gViewProj);
    o.wpos    = world;
    o.svpos   = clip;
    o.col     = i.col;
    o.faceIdx = i.pck.w & 7;
    return o;
}

// ---------------- Points technique ----------------
// faceIdx field carries visMask (6 bits, one per cube face direction).
struct VSPointOut {
    float4 svpos : SV_Position;
    float3 col   : COL;
    float3 wpos  : WPOS;
    nointerpolation uint mask : MASK;
};

VSPointOut vsmain_points(VSIn i)
{
    VSPointOut o;
    // _pad in cbPerChunk carries half-extent of the LOD voxel (0.5/1.0/2.0).
    float3 local = float3(i.pck.x, i.pck.y, i.pck.z) + _pad;
    float3 world = gChunkBase + local;
    o.svpos = mul(float4(world, 1.0), gViewProj);
    o.col   = i.col.rgb;
    o.wpos  = world;
    o.mask  = i.pck.w & 0x3Fu;
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
    if (mode == 1) return float4(ApplyFog(i.col, fogDist), 1.0);
    float3 n = normalize(gCamPos - i.wpos);
    if (mode == 2) return float4(ApplyFog(n * 0.5 + 0.5, fogDist), 1.0);
    float ndotl = saturate(dot(n, normalize(gLightDir)));
    float3 amb  = SampleAmbientCubeTriplanar(n);
    float3 sun  = float3(1.10, 1.00, 0.85) * ndotl;
    float3 light = amb + gAmbient * sun;
    return float4(ApplyFog(i.col * light, fogDist), 1.0);
}

// Complex: per-axis up-to-3 cardinal faces, weighted by projected area.
float4 psmain_points(VSPointOut i) : SV_Target
{
    int mode = (int)gMode;
    if (mode == 1) return float4(ApplyFog(i.col, length(i.wpos - gCamPos)), 1.0);

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

        float ndotl = saturate(dot(n, L));
        float3 amb  = kAmbientCube[bitIdx];
        float3 sun  = float3(1.10, 1.00, 0.85) * ndotl;
        float3 light = amb + gAmbient * sun;
        accumCol += i.col * light * w;
        accumN   += n * w;
        accumW   += w;
    }

    float fogDist = length(i.wpos - gCamPos);
    if (accumW <= 1e-5) {
        if (mode == 2) return float4(ApplyFog(float3(0.5, 0.5, 0.5), fogDist), 1.0);
        return float4(ApplyFog(i.col, fogDist), 1.0);
    }

    if (mode == 2) {
        return float4(ApplyFog(normalize(accumN) * 0.5 + 0.5, fogDist), 1.0);
    }
    return float4(ApplyFog(accumCol / accumW, fogDist), 1.0);
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

// Packed voxel from existing point VB (8 bytes per entry):
// pck = x | (y<<8) | (z<<16) | (faceIdx<<24)
// col = RGBA8 packed
struct VoxelP { uint pck; uint col; };

StructuredBuffer<VoxelP> gVoxels  : register(t0);
RWTexture2D<uint>        gColorUav : register(u0);

[numthreads(64, 1, 1)]
void csmain_points_cs(uint3 dt : SV_DispatchThreadID)
{
    uint idx = dt.x;
    if (idx >= gCsVoxelCount) return;
    VoxelP v = gVoxels[gCsVoxelOffset + idx];
    float3 local = float3((v.pck >> 0u) & 0xFFu,
                          (v.pck >> 8u) & 0xFFu,
                          (v.pck >> 16u) & 0xFFu);
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

// ---------------- Chunk bounds debug (line list) ----------------
// gChunkBase = chunk origin (world), _pad0 = chunk box size (e.g. 64).
struct VSBoundsIn  { float3 pos : POSITION; };
struct VSBoundsOut { float4 svpos : SV_Position; };

VSBoundsOut vsmain_bounds(VSBoundsIn i)
{
    VSBoundsOut o;
    float3 world = gChunkBase + i.pos * _pad;   // _pad in cbPerChunk = chunkSize
    o.svpos = mul(float4(world, 1.0), gViewProj);
    return o;
}

float4 psmain_bounds(VSBoundsOut i) : SV_Target
{
    return float4(1.0, 1.0, 0.0, 1.0);   // yellow
}

// ---------------- PolyVID technique ----------------
// Shared IB has values = i*8 + corner (0..7) per voxel i. Per-chunk Draw uses
// baseVertex = chunkVoxelBase*8 so SV_VertexID = (chunkVoxelBase + i)*8 + corner.
// VS pulls voxel data from a StructuredBuffer; no VB needed.
StructuredBuffer<VoxelP> gPolyVoxels : register(t1);

struct VSPolyVidOut {
    float4 svpos : SV_Position;
    float3 nrm   : NRM;
    float3 col   : COL;
    float3 wpos  : WPOS;
};

VSPolyVidOut vsmain_polyvid(uint vid : SV_VertexID)
{
    VSPolyVidOut o;
    uint voxelIdx  = (vid / 8u) + gVoxelBase;
    uint cornerIdx = vid & 7u;
    VoxelP v = gPolyVoxels[voxelIdx];
    float3 voxLocal  = float3((v.pck >> 0u) & 0xFFu,
                              (v.pck >> 8u) & 0xFFu,
                              (v.pck >> 16u) & 0xFFu);
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
    if (mode == 2) return float4(ApplyFog(n * 0.5 + 0.5, fogDist), 1.0);
    if (mode == 1) return float4(ApplyFog(i.col, fogDist), 1.0);
    float ndotl = saturate(dot(n, normalize(gLightDir)));
    float3 amb  = SampleAmbientCubeTriplanar(n);
    float3 sun  = float3(1.10, 1.00, 0.85) * ndotl;
    float3 light = amb + gAmbient * sun;
    return float4(ApplyFog(i.col * light, fogDist), 1.0);
}

// ---------------- Splat technique ----------------
// Points rendered into an RGBA8 RT: rgb = base color, a = LOD tag (for CS pass).
// SPLAT_FANCY_SHADING: 0 = raw base color (cheap, needed for CS filter alpha tag),
//                     1 = full point shading (triplanar ambient + sun + fog).
#define SPLAT_FANCY_SHADING 1

#if SPLAT_FANCY_SHADING

VSPointOut vsmain_splat(VSIn i)
{
    return vsmain_points(i);
}

float4 psmain_splat(VSPointOut i) : SV_Target
{
    return psmain_points_simple(i);
}

#else

struct VSSplatOut {
    float4 svpos : SV_Position;
    float4 col   : COL;      // rgb = base color, a = LOD index (0/1/2) as raw byte
};

VSSplatOut vsmain_splat(VSIn i)
{
    VSSplatOut o;
    float3 local = float3(i.pck.x, i.pck.y, i.pck.z) + _pad;
    float3 world = gChunkBase + local;
    o.svpos = mul(float4(world, 1.0), gViewProj);
    // _pad = lodHalfExtent (0.5/1.0/2.0) -> lod index 0/1/2.
    // Store lod+1 in alpha byte so 0 means "empty pixel".
    uint lod = (_pad < 0.75) ? 0u : (_pad < 1.5) ? 1u : 2u;
    o.col = float4(i.col.rgb, (float)(lod + 1u) / 255.0);
    return o;
}

float4 psmain_splat(VSSplatOut i) : SV_Target
{
    return i.col;
}

#endif

// CS reconstruction: for each output pixel, search neighbors for the nearest
// point whose splat sphere covers it; use its color (background if none).
Texture2D<float4>   gSplatColorSrv : register(t2);
RWTexture2D<float4> gSplatFinalUav : register(u1);

[numthreads(8, 8, 1)]
void csmain_splat(uint3 dt : SV_DispatchThreadID)
{
    int2 pix = (int2)dt.xy;
    int W = (int)gScreenSize.x;
    int H = (int)gScreenSize.y;
    if (pix.x >= W || pix.y >= H) return;

    // Expanding rings: return as soon as we find any hits.
    const float kAlphaEps = 0.5 / 255.0;

    // Ring 0: center pixel.
    float4 c0 = gSplatColorSrv.Load(int3(pix, 0));
    if (c0.a >= kAlphaEps) {
        gSplatFinalUav[pix] = float4(c0.rgb, 1.0);
        return;
    }

    // Ring 1: 3x3 minus center.
    {
        float3 accum = 0;
        uint count = 0;
        [unroll] for (int dy = -1; dy <= 1; ++dy)
        [unroll] for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int2 sp = pix + int2(dx, dy);
            if (sp.x < 0 || sp.x >= W || sp.y < 0 || sp.y >= H) continue;
            float4 s = gSplatColorSrv.Load(int3(sp, 0));
            if (s.a < kAlphaEps) continue;
            accum += s.rgb;
            ++count;
        }
        if (count > 0u) {
            gSplatFinalUav[pix] = float4(accum / (float)count, 1.0);
            return;
        }
    }

    // Ring 2: 5x5 outer (exclude 3x3 interior).
    {
        float3 accum = 0;
        uint count = 0;
        [unroll] for (int dy = -2; dy <= 2; ++dy)
        [unroll] for (int dx = -2; dx <= 2; ++dx) {
            if (abs(dx) < 2 && abs(dy) < 2) continue;
            int2 sp = pix + int2(dx, dy);
            if (sp.x < 0 || sp.x >= W || sp.y < 0 || sp.y >= H) continue;
            float4 s = gSplatColorSrv.Load(int3(sp, 0));
            if (s.a < kAlphaEps) continue;
            accum += s.rgb;
            ++count;
        }
        if (count > 0u) {
            gSplatFinalUav[pix] = float4(accum / (float)count, 1.0);
            return;
        }
    }

    // No hit in any ring -> fall back to the cleared background color (the
    // host clears splatColorTex_ RGB to the current scene clear color).
    gSplatFinalUav[pix] = float4(c0.rgb, 1.0);
}

// ---------------- MergedMesh technique ----------------
// Greedy-meshed triangles, single big VB/IB. Float3 pos + rgba8 color.
// Normal computed per-pixel via ddx/ddy of world position.
struct VSMergedIn {
    float3 pos : POSITION;
    float4 col : COLOR;
};
struct VSMergedOut {
    float4 svpos : SV_Position;
    float3 wpos  : WPOS;
    float3 col   : COL;
};
VSMergedOut vsmain_merged(VSMergedIn i)
{
    VSMergedOut o;
    float3 world = i.pos + gChunkBase;   // gChunkBase used as grid offset
    o.wpos  = world;
    o.svpos = mul(float4(world, 1.0), gViewProj);
    o.col   = i.col.rgb;
    return o;
}
float4 psmain_merged(VSMergedOut i) : SV_Target
{
    float3 dpx = ddx(i.wpos);
    float3 dpy = ddy(i.wpos);
    float3 n = normalize(cross(dpx, dpy));
    n = sign(n) * pow(abs(n), 7.0);
    n = normalize(n);
    int mode = (int)gMode;
    float fogDist = length(i.wpos - gCamPos);
    if (mode == 2) return float4(ApplyFog(n * 0.5 + 0.5, fogDist), 1.0);
    if (mode == 1) return float4(ApplyFog(i.col, fogDist), 1.0);
    float ndotl = saturate(dot(n, normalize(gLightDir)));
    float3 amb  = SampleAmbientCubeTriplanar(n);
    float3 sun  = float3(1.10, 1.00, 0.85) * ndotl;
    float3 light = amb + gAmbient * sun;
    return float4(ApplyFog(i.col * light, fogDist), 1.0);
}

// ---------------- Billboard technique ----------------
// 1 voxel = 1 quad (4 verts) in screen space covering voxel's NDC bounding rect.
// PS performs ray-vs-AABB intersection to find the hit face + true depth.
struct VSBillOut {
    float4 svpos : SV_Position;
    nointerpolation float3 boxMin : BMIN;
    nointerpolation float3 boxMax : BMAX;
    nointerpolation float3 col    : COL;
};

// Shared code for both billboard variants: emits one vertex of an axis-aligned
// figure on a face-camera plane at the front of the voxel's bounding sphere.
// `s` is the 2D offset in (voxRight, voxUp) space, in units of r.
VSBillOut BillboardVertex(uint voxelIdx, float2 s)
{
    VoxelP v = gPolyVoxels[voxelIdx];
    float3 voxLocal = float3((v.pck >> 0u) & 0xFFu,
                             (v.pck >> 8u) & 0xFFu,
                             (v.pck >> 16u) & 0xFFu);
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
    float fogDist = length(hit - gCamPos);
    if (mode == 2) return float4(ApplyFog(n * 0.5 + 0.5, fogDist), 1.0);
    if (mode == 1) return float4(ApplyFog(i.col, fogDist), 1.0);

    float ndotl = saturate(dot(n, normalize(gLightDir)));
    float3 amb  = SampleAmbientCubeTriplanar(n);
    float3 sun  = float3(1.10, 1.00, 0.85) * ndotl;
    float3 light = amb + gAmbient * sun;
    return float4(ApplyFog(i.col * light, fogDist), 1.0);
}

// ---------------- HexSprite technique ----------------
// Per voxel: 18 vertices = 6 triangles fanning from closest-corner-to-camera.
// Voxel data delivered as per-instance attributes; SV_VertexID 0..17 picks
// the triangle (0..5) and the role (center, v1, v2).
struct VSHexIn {
    uint4  inst_pck : POSITION;
    float4 inst_col : COLOR;
    uint   vid : SV_VertexID;
};
struct VSHexOut {
    float4 svpos : SV_Position;
    float3 wpos  : WPOS;
    float3 col   : COL;
};

VSHexOut vsmain_hex(VSHexIn i)
{
    uint tri    = i.vid / 3;          // 0..5
    uint role   = i.vid % 3;          // 0=center, 1=outer[tri], 2=outer[(tri+1)%6]
    uint cornerID;
    if      (role == 0) cornerID = 0;
    else if (role == 1) cornerID = 1 + tri;
    else                cornerID = 1 + ((tri + 1) % 6);

    float3 voxLocal  = float3(i.inst_pck.xyz);
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
    o.col   = i.inst_col.rgb;
    return o;
}

float4 psmain_hex(VSHexOut i) : SV_Target
{
    float3 dpx = ddx(i.wpos);
    float3 dpy = ddy(i.wpos);
    float3 n   = normalize(cross(dpx, dpy));

    int mode = (int)gMode;
    float fogDist = length(i.wpos - gCamPos);
    if (mode == 2) return float4(ApplyFog(n * 0.5 + 0.5, fogDist), 1.0);
    if (mode == 1) return float4(ApplyFog(i.col, fogDist), 1.0);

    float ndotl = saturate(dot(n, normalize(gLightDir)));
    float3 amb  = SampleAmbientCubeTriplanar(n);
    float3 sun  = float3(1.10, 1.00, 0.85) * ndotl;
    float3 light = amb + gAmbient * sun;
    return float4(ApplyFog(i.col * light, fogDist), 1.0);
}

// ---------------- Polygon technique ----------------
float4 psmain(VSOut i) : SV_Target
{
    // Per-pixel face normal from world-pos derivatives.
    // DX screen-y is top-to-bottom; cross(ddx, ddy) yields the outward
    // normal for front-facing triangles.
    float3 dpx = ddx(i.wpos);
    float3 dpy = ddy(i.wpos);
    float3 n   = normalize(cross(dpx, dpy));
    // ddx/ddy is noisy for near-camera small triangles. Sharpen toward the
    // dominant cardinal axis: raise to high odd power (sign preserved), renorm.
    n = sign(n) * pow(abs(n), 7.0);
    n = normalize(n);

    int mode = (int)gMode;
    float fogDist = length(i.wpos - gCamPos);
    if (mode == 2) return float4(ApplyFog(n * 0.5 + 0.5, fogDist), 1.0);
    if (mode == 1) return float4(ApplyFog(i.col.rgb, fogDist), 1.0);

    float ndotl = saturate(dot(n, normalize(gLightDir)));
    float3 amb  = SampleAmbientCubeTriplanar(n);  // axis-aligned n => single bucket
    float3 sun  = float3(1.10, 1.00, 0.85) * ndotl;
    float3 light = amb + gAmbient * sun;
    return float4(ApplyFog(i.col.rgb * light, fogDist), 1.0);
}
