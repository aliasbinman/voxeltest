// shading.hlsli — shared lighting code used by splat CS (voxel.hlsl) and
// PolyAxis PS (lodworld.hlsl). Prepended to both files at compile time.
//
// Required uniforms (each shader's CB must declare these names):
//   gCamPos, gLightDir, gSunIntensity, gAmbient, gExposure, gColorizeClusters
//   gFogColor, gFogDensity, gHeightFogDensity, gHeightFogFalloff, gHeightFogStart
//   gShadowEnable, gShadowBias, gShadowMapSize, gSunViewProj
//
// Required bindings:
//   t6 = gShadowTex (declared below)
//   s1 = gShadowSamp (declared below)

Texture2D<float>          gShadowTex  : register(t6);
SamplerComparisonState    gShadowSamp : register(s1);

// ---- Depth + height fog (Inigo Quilez analytical), sun-tinted toward sun. ----
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

// ---- Ambient cube (face order: +X -X +Y -Y +Z -Z), triplanar blend. ----
static const float3 kAmbientCube[6] = {
    float3(0.85, 0.65, 0.45),  // +X warm
    float3(0.40, 0.50, 0.65),  // -X cool
    float3(0.85, 1.00, 1.20),  // +Y sky
    float3(0.18, 0.14, 0.10),  // -Y ground
    float3(0.65, 0.65, 0.55),  // +Z
    float3(0.40, 0.45, 0.55),  // -Z
};
float3 SampleAmbientCubeTriplanar(float3 n)
{
    float3 an = abs(n);
    float  total = max(an.x + an.y + an.z, 1e-5);
    float3 amb = an.x * (n.x > 0.0 ? kAmbientCube[0] : kAmbientCube[1])
               + an.y * (n.y > 0.0 ? kAmbientCube[2] : kAmbientCube[3])
               + an.z * (n.z > 0.0 ? kAmbientCube[4] : kAmbientCube[5]);
    return amb / total;
}

// ---- Sun shadow PCF (3x3). gShadowEnable<0.5 -> stub 1.0. ----
float SampleShadow(float3 wpos)
{
    if (gShadowEnable < 0.5) return 1.0;
    float4 sp = mul(float4(wpos, 1.0), gSunViewProj);
    float3 ndc = sp.xyz / sp.w;
    if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 ||
        ndc.z <  0.0 || ndc.z > 1.0) return 1.0;
    float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    float ref = ndc.z + gShadowBias;
    float texel = 1.0 / max(gShadowMapSize, 1.0);
    float sum = 0.0;
    [unroll] for (int dy = -1; dy <= 1; ++dy)
    [unroll] for (int dx = -1; dx <= 1; ++dx) {
        sum += gShadowTex.SampleCmpLevelZero(gShadowSamp, uv + float2(dx, dy) * texel, ref);
    }
    return sum / 9.0;
}

// ---- LodViz tints (2..6 = LOD0..4, anything else white) ----
float3 ClusterTint(uint t)
{
    if (t == 1u) return float3(0.3, 1.0, 0.3);
    if (t == 2u) return float3(0.3, 0.5, 1.0);   // L0 blue
    if (t == 3u) return float3(0.8, 0.3, 1.0);   // L1 purple
    if (t == 4u) return float3(1.0, 1.0, 0.3);   // L2 yellow
    if (t == 5u) return float3(1.0, 0.55, 0.1);  // L3 orange
    if (t == 6u) return float3(1.0, 0.3, 0.3);   // L4 red
    return float3(1.0, 1.0, 1.0);
}

// ---- ACES filmic tonemap with exposure pre-multiply. ----
float3 Tonemap(float3 x)
{
    x = x * gExposure;
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// ============================================================
// Top-level lighting: ambient cube * AO + sun N·L * shadow, fog, tonemap.
// Handles all ShadingMode branches (0=Lit, 1=FlatColor, 2=Normals, 3=AO,
// 4=LodViz). Both splat CS and PolyAxis PS call this — identical look.
//   albedo  : base color
//   N       : surface normal (world-space, outward)
//   wpos    : surface position (world-space)
//   ao      : 0..1 ambient occlusion (per-face)
//   lodIdx  : 0..3 for LodViz tint
//   parity  : cluster checker bit for LodViz
//   mode    : ShadingMode (cast from gMode)
// ============================================================
float3 ShadeWithLighting(float3 albedo, float3 N, float3 wpos, float ao,
                         uint lodIdx, uint parity, int mode)
{
    if (mode == 1) return ApplyFog(albedo, wpos);
    if (mode == 2) return ApplyFog(N * 0.5 + 0.5, wpos);
    if (mode == 3) return ao.xxx;     // raw AO, no fog
    if (mode == 4) {
        float check = (parity == 0u) ? 0.55 : 1.00;
        return ao * ClusterTint(2u + lodIdx) * check;
    }
    float3 amb   = SampleAmbientCubeTriplanar(N) * ao;
    float  shad  = SampleShadow(wpos);
    float  ndotl = saturate(dot(N, normalize(gLightDir)));
    float3 sun   = float3(1.10, 1.00, 0.85) * ndotl * shad * gSunIntensity;
    float3 light = amb + gAmbient * sun;
    float3 lit   = albedo * light;
    float3 tint  = (gColorizeClusters > 0.5) ? ClusterTint(2u + lodIdx) : float3(1, 1, 1);
    return ApplyFog(lit * tint, wpos); // tonemap deferred to final post
}

// Per-face accumulation gated by 6-bit visMask. Hidden faces (bit clear)
// drop out so buried voxels don't over-light. Shares one shadow tap + fog.
float3 ShadeWithLighting6Face(float3 albedo, float3 wpos, float ao,
                              uint visMask,
                              uint lodIdx, uint parity, int mode)
{
    if (mode == 1) return ApplyFog(albedo, wpos);
    if (mode == 2) return ApplyFog(float3(0.5, 0.5, 0.5), wpos);
    if (mode == 3) return ao.xxx;
    if (mode == 4) {
        float check = (parity == 0u) ? 0.55 : 1.00;
        return ao * ClusterTint(2u + lodIdx) * check;
    }
    static const float3 kFaceN[6] = {
        float3( 1, 0, 0), float3(-1, 0, 0),
        float3( 0, 1, 0), float3( 0,-1, 0),
        float3( 0, 0, 1), float3( 0, 0,-1),
    };
    float3 L = normalize(gLightDir);
    float  shad = SampleShadow(wpos);
    float3 ambSum = 0;
    float  ndotlSum = 0;
    uint visible = 0;
    [unroll] for (uint f = 0u; f < 6u; ++f) {
        if (((visMask >> f) & 1u) == 0u) continue;
        ambSum   += SampleAmbientCubeTriplanar(kFaceN[f]);
        ndotlSum += saturate(dot(kFaceN[f], L));
        ++visible;
    }
    float invN = (visible > 0u) ? (1.0 / (float)visible) : 0.0;
    float3 amb = ambSum * invN * ao;
    float ndotl = ndotlSum * invN;
    float3 sun = float3(1.10, 1.00, 0.85) * ndotl * shad * gSunIntensity;
    float3 light = amb + gAmbient * sun;
    float3 lit = albedo * light;
    float3 tint = (gColorizeClusters > 0.5) ? ClusterTint(2u + lodIdx) : float3(1, 1, 1);
    return ApplyFog(lit * tint, wpos);
}

// 3-face area-weighted variant. Shares shadow tap + fog.
// Per-face only ambient cube fetch + ndotl. Weights must sum to 1.
float3 ShadeWithLighting3Face(float3 albedo,
                              float3 N0, float3 N1, float3 N2,
                              float w0, float w1, float w2,
                              float3 wpos, float ao,
                              uint lodIdx, uint parity, int mode)
{
    if (mode == 1) return ApplyFog(albedo, wpos);
    if (mode == 2) {
        float3 Nb = w0 * N0 + w1 * N1 + w2 * N2;
        return ApplyFog(Nb * 0.5 + 0.5, wpos);
    }
    if (mode == 3) return ao.xxx;
    if (mode == 4) {
        float check = (parity == 0u) ? 0.55 : 1.00;
        return ao * ClusterTint(2u + lodIdx) * check;
    }
    float3 L = normalize(gLightDir);
    float3 amb = (w0 * SampleAmbientCubeTriplanar(N0)
                + w1 * SampleAmbientCubeTriplanar(N1)
                + w2 * SampleAmbientCubeTriplanar(N2)) * ao;
    float ndotl = w0 * saturate(dot(N0, L))
                + w1 * saturate(dot(N1, L))
                + w2 * saturate(dot(N2, L));
    float  shad  = SampleShadow(wpos);
    float3 sun   = float3(1.10, 1.00, 0.85) * ndotl * shad * gSunIntensity;
    float3 light = amb + gAmbient * sun;
    float3 lit   = albedo * light;
    float3 tint  = (gColorizeClusters > 0.5) ? ClusterTint(2u + lodIdx) : float3(1, 1, 1);
    return ApplyFog(lit * tint, wpos); // tonemap deferred to final post
}

// ============================================================
// Top-down cheap AO (depth map built in lodworld.hlsl). Shared by csSplat
// dilate which has reconstructed normals — push lookup along normal so
// vertical faces read their open neighbour column.
// ============================================================
cbuffer CBLwAo : register(b4)
{
    float2 gAoOriginXZ;
    float2 gAoInvSizeXZ;
    float  gAoTexSizeF;
    float  gAoYMin;
    float  gAoYScale;
    float  gAoStrength;
    float  gAoFadeUnits;
    float  gAoPushTexels;
    float  gAoWorldPerTexel;
    float  _padAo;
};
Texture2D<uint>  gAoTopDownSrv : register(t8);
Texture2D<float> gAoOcclSrv    : register(t9);

float AoDecodeY(uint encY) { return (float)encY / gAoYScale + gAoYMin; }

// Raw lookup — no depth check, no neighbour blend. Just sample HBAO at
// world-XZ projection.
float AoSampleAt(float3 world, float2 lateralTexelOfs)
{
    if (gAoStrength <= 0.0) return 1.0;
    float2 uv = (world.xz - gAoOriginXZ) * gAoInvSizeXZ;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 1.0;
    int2 baseT = int2(uv * gAoTexSizeF);
    int2 t = clamp(baseT + (int2)lateralTexelOfs, int2(0,0),
                   int2((int)gAoTexSizeF - 1, (int)gAoTexSizeF - 1));
    float occ = gAoOcclSrv.Load(int3(t, 0));
    return lerp(1.0, occ, gAoStrength);
}

float AoSampleWithNormal(float3 world, float3 N)
{
    float2 nXZ = N.xz;
    float  l2  = dot(nXZ, nXZ);
    float2 ofs = float2(0, 0);
    if (l2 > 1e-4) {
        nXZ *= rsqrt(l2);
        ofs  = nXZ * gAoPushTexels;
    }
    return AoSampleAt(world, ofs);
}
