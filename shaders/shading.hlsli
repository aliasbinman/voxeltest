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

// ---- LodViz tints (2..5 = LOD0..3, anything else white) ----
float3 ClusterTint(uint t)
{
    if (t == 1u) return float3(0.3, 1.0, 0.3);
    if (t == 2u) return float3(0.3, 0.5, 1.0);
    if (t == 3u) return float3(0.8, 0.3, 1.0);
    if (t == 4u) return float3(1.0, 1.0, 0.3);
    if (t == 5u) return float3(1.0, 0.55, 0.1);
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
    return Tonemap(ApplyFog(lit * tint, wpos));
}
