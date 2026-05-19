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
    uint3  _pad2;
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
};

float3 UnpackVoxPos(uint4 p)
{
    return float3(p.x, p.y, p.z);
}

// Per-face shadow mask: low 6 bits of aux (pck.w for VSIn / (pck.y>>16) for
// VoxelP). Bit i: 1 = face i directly lit, 0 = shadowed. Face order matches
// kFaceNormals (0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z).
uint ShadowMaskFromAux(uint aux) { return aux & 0x3Fu; }

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

// Combined lighting using baked sun-shadow term. shadow=0 kills sun but keeps
// full ambient so shadowed faces stay readable; shadow=1 = fully lit.
float3 ApplyShadowLighting(float3 amb, float3 sunDir, float3 n, float shadow)
{
    float ndotl = saturate(dot(n, sunDir));
    float3 sun  = float3(1.10, 1.00, 0.85) * ndotl * shadow;
    return amb + gAmbient * sun;
}

// Depth-only VS (Z prepass). No PS bound.
struct VSDepthOut { float4 svpos : SV_Position; };
// `precise` forces bit-exact identical math in both VS variants so the
// prepass and main pass produce the same SV_Position -> no z-fighting.
VSDepthOut vsmain_depth(VSPolyIn i)
{
    VSDepthOut o;
    precise float3 local = UnpackVoxPos(i.pck);
    precise float3 world = gChunkBase + local;
    precise float4 clip  = mul(float4(world, 1.0), gViewProj);
    o.svpos = clip;
    return o;
}

VSOut vsmain(VSPolyIn i)
{
    VSOut o;
    precise float3 local = UnpackVoxPos(i.pck);
    precise float3 world = gChunkBase + local;
    precise float4 clip  = mul(float4(world, 1.0), gViewProj);
    o.wpos    = world;
    o.svpos   = clip;
    o.col     = float4(float3(i.col.rgb) / 255.0, 1.0);
    o.faceIdx = 0;
    o.shadowMask = ShadowMaskFromAux(i.pck.w);
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
};

VSPointOut vsmain_points(VSIn i)
{
    VSPointOut o;
    // _pad carries half-extent of the LOD voxel (0.5/1.0/2.0); scene-rel pos + pad = voxel center.
    float3 local = UnpackVoxPos(i.pck) + _pad;
    float3 world = gChunkBase + local;
    o.svpos = mul(float4(world, 1.0), gViewProj);
    o.col   = float3(i.col.rgb) / 255.0;
    o.wpos  = world;
    o.mask  = i.col.a & 0x3Fu;       // visMask packed in color alpha
    o.shadowMask = ShadowMaskFromAux(i.pck.w);
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
    float3 n = normalize(gCamPos - i.wpos);
    if (mode == 2) return float4(ApplyFog(n * 0.5 + 0.5, i.wpos), 1.0);
    float3 amb  = SampleAmbientCubeTriplanar(n);
    float3 light = ApplyShadowLighting(amb, normalize(gLightDir), n, ShadowBitN(i.shadowMask, n));
    return float4(ApplyFog(i.col * light, i.wpos), 1.0);
}

// Complex: per-axis up-to-3 cardinal faces, weighted by projected area.
float4 psmain_points(VSPointOut i) : SV_Target
{
    int mode = (int)gMode;
    if (mode == 1) return float4(ApplyFog(i.col, i.wpos), 1.0);

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

        float3 amb  = kAmbientCube[bitIdx];
        float3 light = ApplyShadowLighting(amb, L, n, ShadowBit(i.shadowMask, bitIdx));
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
    return float4(ApplyFog(accumCol / accumW, i.wpos), 1.0);
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
// 12-byte structured-buffer entry. pck.x = (x | y<<16), pck.y = (z | aux<<16).
struct VoxelP { uint2 pck; uint col; };
float3 UnpackVoxPosCS(uint2 p)
{
    return float3((float)(p.x & 0xFFFFu),
                  (float)((p.x >> 16u) & 0xFFFFu),
                  (float)(p.y & 0xFFFFu));
}

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

// ---------------- Chunk bounds debug (line list) ----------------
// gChunkBase = chunk origin (world), _pad0 = chunk box size (e.g. 64).
struct VSBoundsIn  { float3 pos : POSITION; };
struct VSBoundsOut { float4 svpos : SV_Position; };

VSBoundsOut vsmain_bounds(VSBoundsIn i)
{
    VSBoundsOut o;
    // Tight AABB: chunkBase = world aabbMin; size = (_pad, asfloat(_pad2.xy)).
    float3 size = float3(_pad, asfloat(_pad2.x), asfloat(_pad2.y));
    float3 world = gChunkBase + i.pos * size;
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
    nointerpolation uint shadowMask : SHADOW;
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
    float3 amb  = SampleAmbientCubeTriplanar(n);
    float3 light = ApplyShadowLighting(amb, normalize(gLightDir), n, ShadowBitN(i.shadowMask, n));
    return float4(ApplyFog(i.col * light, i.wpos), 1.0);
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
    float3 curC = gTaaScene.SampleLevel(gTaaSamp, currUv, 0).rgb;
    float  d    = gTaaDepth.Load(int3(pix, 0));
    if (d <= 0.0) {
        return float4(curC, 1.0);
    }

    // Reconstruct world position from current depth + camera basis (undo jitter).
    float viewZ = gNearZ / d;
    float aspect = (float)W / (float)H;
    float ndcX = ((float)pix.x + 0.5) / (float)W * 2.0 - 1.0 - gJitter.x;
    float ndcY = 1.0 - ((float)pix.y + 0.5) / (float)H * 2.0 - gJitter.y;
    float viewX = ndcX * aspect * gTanHalfFovY * viewZ;
    float viewY = ndcY *          gTanHalfFovY * viewZ;
    float3 world = gCamPos + gCamRight * viewX + gCamUp * viewY + gCamForward * viewZ;

    // Reproject to previous frame.
    float4 prevClip = mul(float4(world, 1.0), gPrevViewProj);
    if (prevClip.w <= 0.0) return float4(curC, 1.0);
    float3 prevNdc = prevClip.xyz / prevClip.w;
    if (any(abs(prevNdc.xy) > 1.0)) return float4(curC, 1.0);
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

    float3 outC = lerp(prevC, curC, 0.05);
    return float4(outC, 1.0);
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
// splatColorTex_ stores ALBEDO in RGB and per-voxel sun-shadow in alpha.
// Filled-marker check uses gSplatDepth (cleared to 0; reverse-Z so >0 = hit).
// Encode 6-bit shadow mask into alpha byte (UNORM round-trip via *255 in CS).
float4 psmain_splat_albedo(VSPointOut i) : SV_Target
{
    return float4(i.col, (float)i.shadowMask / 255.0);
}
float4 psmain_splat_albedo_poly(VSOut i) : SV_Target
{
    return float4(i.col.rgb, (float)i.shadowMask / 255.0);
}

// CS reconstruction: for each output pixel, search neighbors for the nearest
// point whose splat sphere covers it; use its color (background if none).
Texture2D<float>    gSplatDepth    : register(t1);
Texture2D<float4>   gSplatColorSrv : register(t2);
RWTexture2D<float4> gSplatFinalUav : register(u1);

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
    float ndcX = ((float)sp.x + 0.5) / (float)W * 2.0 - 1.0;
    float ndcY = 1.0 - ((float)sp.y + 0.5) / (float)H * 2.0;
    float viewX = ndcX * aspect * gTanHalfFovY * viewZ;
    float viewY = ndcY *          gTanHalfFovY * viewZ;
    return gCamPos + gCamRight * viewX + gCamUp * viewY + gCamForward * viewZ;
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

    [loop] for (int dy = -R; dy <= R; ++dy) {
        [loop] for (int dx = -R; dx <= R; ++dx) {
            int2 sp = pix + int2(dx, dy);
            if (sp.x < 0 || sp.x >= W || sp.y < 0 || sp.y >= H) continue;
            float zN = gSplatDepth.Load(int3(sp, 0));
            if (zN <= 0.0) continue;       // depth>0 marks filled (alpha = shadow)
            float4 s = gSplatColorSrv.Load(int3(sp, 0));
            float3 wp = ReconstructNeighborWorld(sp, zN, W, H);
            float3 vmin = floor(wp);
            float3 vmax = vmin + 1.0;
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
                bestT      = tHit;
                bestAlbedo = s.rgb;
                bestN      = n;
                bestMask   = (uint)(s.a * 255.0 + 0.5);
                anyHit     = true;
            }
        }
    }

    if (anyHit) {
        float3 hit = ro + rd * bestT;
        float3 amb = SampleAmbientCubeTriplanar(bestN);
        float3 light = ApplyShadowLighting(amb, normalize(gLightDir), bestN, ShadowBitN(bestMask, bestN));
        float3 lit = bestAlbedo * light;
        gSplatFinalUav[pix] = float4(ApplyFog(lit, hit), 1.0);
    } else {
        float4 c0 = gSplatColorSrv.Load(int3(pix, 0));
        gSplatFinalUav[pix] = float4(c0.rgb, 1.0);
    }
}

// ---------------- AtlasMesh technique ----------------
// Compact 12 B vertex: uint16 px,py,pz; uint8 face; uint8 _pad; uint16 u,v.
// Sampled with Load (no sampler/no atlas-size needed) — UV is already in
// texel coords; linear interp + truncation gives exact texel per pixel.
Texture2D<float4> gAtlasTex : register(t3);

struct VSAtlasIn {
    uint4 pos : POSITION;   // .xyz = scene-local grid coords (uint16), .w = (pad<<8)|face
    uint2 uv  : UV;         // atlas texel coords (uint16)
};
struct VSAtlasOut {
    float4 svpos : SV_Position;
    float3 wpos  : WPOS;
    float2 uv    : UV;
    nointerpolation uint face : FIDX;
};

VSAtlasOut vsmain_atlas(VSAtlasIn i)
{
    VSAtlasOut o;
    float3 local = float3(i.pos.xyz);
    float3 world = gChunkBase + local;
    o.svpos = mul(float4(world, 1.0), gViewProj);
    o.wpos  = world;
    o.uv    = float2(i.uv);
    o.face  = i.pos.w & 0xFFu;
    return o;
}

float4 psmain_atlas(VSAtlasOut i) : SV_Target
{
    int2 tx = int2(i.uv);                            // truncate -> exact texel
    float3 col = gAtlasTex.Load(int3(tx, 0)).rgb;
    float3 n = kFaceNormals[i.face & 7];

    int mode = (int)gMode;
    float fogDist = length(i.wpos - gCamPos);
    if (mode == 2) return float4(ApplyFog(n * 0.5 + 0.5, i.wpos), 1.0);
    if (mode == 1) return float4(ApplyFog(col, i.wpos), 1.0);

    float ndotl = saturate(dot(n, normalize(gLightDir)));
    float3 amb  = kAmbientCube[i.face & 7];
    float3 sun  = float3(1.10, 1.00, 0.85) * 1.5  * ndotl;
    float3 light = amb + gAmbient * sun;
    return float4(ApplyFog(col * light, i.wpos), 1.0);
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
    if (mode == 2) return float4(ApplyFog(n * 0.5 + 0.5, i.wpos), 1.0);
    if (mode == 1) return float4(ApplyFog(i.col, i.wpos), 1.0);
    float ndotl = saturate(dot(n, normalize(gLightDir)));
    float3 amb  = SampleAmbientCubeTriplanar(n);
    float3 sun  = float3(1.10, 1.00, 0.85) * ndotl;
    float3 light = amb + gAmbient * sun;
    return float4(ApplyFog(i.col * light, i.wpos), 1.0);
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
    o.shadowMask = ShadowMaskFromAux(v.pck.y >> 16u);
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

    float3 amb  = SampleAmbientCubeTriplanar(n);
    float3 light = ApplyShadowLighting(amb, normalize(gLightDir), n, ShadowBitN(i.shadowMask, n));
    return float4(ApplyFog(i.col * light, hit), 1.0);
}

// ---------------- HexSprite technique ----------------
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
    o.shadowMask = ShadowMaskFromAux(i.inst_pck.w);
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

    float3 amb  = SampleAmbientCubeTriplanar(n);
    float3 light = ApplyShadowLighting(amb, normalize(gLightDir), n, ShadowBitN(i.shadowMask, n));
    return float4(ApplyFog(i.col * light, i.wpos), 1.0);
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
    if (mode == 2) return float4(ApplyFog(n * 0.5 + 0.5, i.wpos), 1.0);
    if (mode == 1) return float4(ApplyFog(i.col.rgb, i.wpos), 1.0);

    float3 amb  = SampleAmbientCubeTriplanar(n);  // axis-aligned n => single bucket
    float3 light = ApplyShadowLighting(amb, normalize(gLightDir), n, ShadowBitN(i.shadowMask, n));
    return float4(ApplyFog(i.col.rgb * light, i.wpos), 1.0);
}
