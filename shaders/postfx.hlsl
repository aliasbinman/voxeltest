// postfx.hlsl — fullscreen-triangle VS/PS passes (splat composite, TAA,
// godrays, final post). CS passes (splat dilate, shadow blur) live in
// voxel.hlsl. Both files share the cbPerFrame layout below.

cbuffer cbPerFrame : register(b0)
{
    row_major float4x4 gViewProj;
    float3   gCamPos;
    float    gMode;
    float3   gLightDir;
    float    gAmbient;
    float3   gPointNormal;
    float    _pad0;
    row_major float4x4 gInvViewProj;
    float2   gScreenSize;
    float2   _pad1;          // .x = splat dilate radius
    float3   gCamRight;
    float    _pad3;
    float3   gCamUp;
    float    _pad4;
    float3   gCamForward;
    float    gTanHalfFovY;
    float3   gFogColor;
    float    gFogDensity;
    float    gHeightFogDensity;
    float    gHeightFogFalloff;
    float    gHeightFogStart;
    float    _padHF;
    float3   gSceneOrigin;
    float    gNearZ;
    float3   gSceneSpan;
    float    _pad6;
    row_major float4x4 gPrevViewProj;
    float2   gJitter;
    float2   _pad7;
    row_major float4x4 gSunViewProj;
    float    gShadowBias;
    float    gShadowMapSize;
    float    gShadowEnable;
    float    gSunIntensity;
    float    gExposure;
    float    gRoughness;
    float    gColorizeClusters;
    float    gGridSize;
    float2   gInvScreenSize;   // 1/W, 1/H
    float    gAspect;          // W/H
    float    gInvAspect;       // H/W
    float    gAspectTanFov;    // gAspect * gTanHalfFovY
    float3   _padPC;
};

// ---------------- Fullscreen triangle for blit-style passes ----------------
struct VBlitOut { float4 pos : SV_Position; };
VBlitOut vsmain_blit(uint vid : SV_VertexID)
{
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VBlitOut o;
    o.pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return o;
}

// ---------------- Composite splat output over scene RT ----------------
Texture2D<float4> gSplatComposite      : register(t8);
Texture2D<float>  gSplatCompositeDepth : register(t7);
struct CompositeOut { float4 color : SV_Target; float depth : SV_Depth; };
CompositeOut psmain_splat_composite(VBlitOut i)
{
    CompositeOut o;
    int2 pix = int2(i.pos.xy);
    float4 c = gSplatComposite.Load(int3(pix, 0));
    float  z = gSplatCompositeDepth.Load(int3(pix, 0));
    o.color = c;
    if (z <= 0.0) { o.depth = 0.0; discard; }
    o.depth = z;
    return o;
}

// ---------------- TAA ----------------
Texture2D<float4> gTaaScene  : register(t4);
Texture2D<float4> gTaaHist   : register(t5);
Texture2D<float>  gTaaDepth  : register(t6);
SamplerState      gTaaSamp   : register(s0);

struct VTaaOut { float4 pos : SV_Position; float2 uv : UV; };

VTaaOut vsmain_taa(uint vid : SV_VertexID)
{
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
    // Pixel-centre UV (no jitter). Use this for depth Load + world reconstruction
    // so reprojection into history is jitter-free → static scenes tap the same
    // history texel every frame. Jitter only applies to the current-frame
    // scene sample.
    float2 currUvCenter = (float2(pix) + 0.5) * gInvScreenSize;
    float2 currUv = currUvCenter + float2(gJitter.x * 0.5, -gJitter.y * 0.5);
    float4 curSample = gTaaScene.SampleLevel(gTaaSamp, currUv, 0);
    float3 curC = curSample.rgb;
    float outAlpha = curSample.a;
    float  d  = gTaaDepth.Load(int3(pix, 0));
    if (d <= 0.0) {
        return float4(curC, outAlpha);
    }

    float viewZ = gNearZ / d;
    float ndcX = currUvCenter.x * 2.0 - 1.0;
    float ndcY = 1.0 - currUvCenter.y * 2.0;
    float viewX = ndcX * gAspectTanFov * viewZ;
    float viewY = ndcY * gTanHalfFovY  * viewZ;
    float3 world = gCamPos + gCamRight * viewX + gCamUp * viewY + gCamForward * viewZ;

    float4 prevClip = mul(float4(world, 1.0), gPrevViewProj);
    if (prevClip.w <= 0.0) return float4(curC, outAlpha);
    float3 prevNdc = prevClip.xyz / prevClip.w;
    if (any(abs(prevNdc.xy) > 1.0)) return float4(curC, outAlpha);
    float2 prevUv = float2(prevNdc.x * 0.5 + 0.5, 0.5 - prevNdc.y * 0.5);

    float2 texSize = gScreenSize;
    float2 sp = prevUv * texSize;
    float2 tp1 = floor(sp - 0.5) + 0.5;
    float2 f = sp - tp1;
    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);
    float2 w12 = w1 + w2;
    float2 off12 = w2 / max(w12, 1e-5);
    // /texSize → *gInvScreenSize: 3 divides become 3 muls.
    float2 tp0  = (tp1 - 1.0)   * gInvScreenSize;
    float2 tp3  = (tp1 + 2.0)   * gInvScreenSize;
    float2 tp12 = (tp1 + off12) * gInvScreenSize;

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
    prevC /= kSum;

    float2 px = gInvScreenSize;
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
    const float gamma = 1.25;
    float3 vMin = max(nMin, nMean - gamma * sigma);
    float3 vMax = min(nMax, nMean + gamma * sigma);
    float3 clipped = clamp(prevC, vMin, vMax);

    float dist = length(clipped - prevC) / max(length(nMax - nMin), 1e-4);
    float alpha = lerp(0.1, 0.5, saturate(dist));
    float3 outC = lerp(clipped, curC, alpha);
    return float4(outC, outAlpha);
}

// ---------------- Post pass: sky for empty pixels + unsharp + godrays ----------------
Texture2D<float>  gPostDepth : register(t6);
Texture2D<float4> gPostIn    : register(t7);

// ---------------- God rays ----------------
cbuffer cbGodray : register(b3)
{
    float2 gSunScreenNdc;        // -1..1 NDC sun position (xy)
    float  gSunOnScreen;         // 0 = sun behind cam or NDC outside; 1 = on screen
    float  gGodrayEmaAlpha;      // 0..1: how much of the new frame to mix in (1 = no smoothing)
    float2 gGodrayHalfScreenUV;  // half-extent in screen-UV space (square in pixels)
    float  gSunFacing;           // dot(viewFwd, sunDir): >0 in front, <0 behind
    float  _padG1;
    float2 gSunScreenUV;         // CPU-precomputed: gSunScreenNdc * (0.5, -0.5) + 0.5
    float  gGodrayStridePx;      // separable: pixels between sparse taps
    float  _padG2;
    float3 gGodrayTint;
    float  gGodrayStrength;      // 0 = off
};
Texture2D<float> gGodrayTex     : register(t9);   // mark (in blur) / blur+EMA (in post)
Texture2D<float> gGodrayHistTex : register(t10);  // previous-frame blur+EMA (blur input only)

SamplerState gAnisoSamp : register(s2); // anisotropic, max 16x; used by the SampleGrad variant

float4 psmain_godray_mark(VTaaOut i) : SV_Target
{
    // Sun off-screen → texel is "open sky" (no occluder) so godrays bleed in
    // from the off-screen sun direction.
    if (gSunOnScreen < 0.5) return 1.0;
    int2 tex = (int2)i.pos.xy;
    float2 local = (float2(tex) + 0.5) / 64.0 * 2.0 - 1.0;
    if (dot(local, local) > 0.97) return 0.0;
    float2 uv = gSunScreenUV + local * gGodrayHalfScreenUV;
    // Off-screen → treat as unoccluded sky so godrays continue past the edge.
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 1.0;
    int2 px = int2(uv * gScreenSize);
    px = clamp(px, int2(0,0), int2((int)gScreenSize.x - 1, (int)gScreenSize.y - 1));
    float d = gPostDepth.Load(int3(px, 0));
    return (d <= 0.0001) ? 1.0 : 0.0;
}

float4 psmain_godray_blur(VTaaOut i) : SV_Target
{
    int2 tex = (int2)i.pos.xy;
    float2 uv = (float2(tex) + 0.5) / 64.0;
    float2 toCenter = float2(0.5, 0.5) - uv;
    const int N = 24;
    float sum = 0.0;
    [unroll] for (int s = 0; s < N; ++s) {
        float t = (float)s / (float)N;
        sum += gGodrayTex.SampleLevel(gTaaSamp, uv + toCenter * t, 0);
    }
    float curr = sum / (float)N;
    float prev = gGodrayHistTex.SampleLevel(gTaaSamp, uv, 0);
    return lerp(prev, curr, gGodrayEmaAlpha);
}

// Anisotropic variant: one SampleGrad with long footprint along the radial
// direction (toward texture center / sun) and a narrow footprint perpendicular.
float4 psmain_godray_blur_aniso(VTaaOut i) : SV_Target
{
    int2 tex = (int2)i.pos.xy;
    float2 uv = (float2(tex) + 0.5) / 64.0;
    float2 toCenter = float2(0.5, 0.5) - uv;
    float  d        = length(toCenter);
    
    float weight = d;
    float2 radial = toCenter / max(d, 0.1);
    float2 perp     = float2(-radial.y, radial.x) ;

    const float streakHalf = 0.135;          // in UV (64-texel space)
    float2 sampleUV = uv + radial * (streakHalf * 0.5);
    float2 ddxUV    = radial * streakHalf;  // long axis
    float2 ddyUV    = perp   * (1.0 / 64.0);// 1-texel across

    float curr = gGodrayTex.SampleGrad(gAnisoSamp, sampleUV, ddxUV, ddyUV);// * weight;
    float prev = gGodrayHistTex.SampleLevel(gTaaSamp, uv, 0);
    return lerp(prev, curr, gGodrayEmaAlpha);
}

// ---- Separable godray blur ----
// Pass 1 (sparse): 5 taps along radial, each gGodrayStridePx pixels apart.
// Reads mark texture (t9). No history blend. Output → intermediate.
float4 psmain_godray_blur_sparse(VTaaOut i) : SV_Target
{
    int2 tex = (int2)i.pos.xy;
    float2 uv = (float2(tex) + 0.5) / 64.0;
    float2 toCenter = float2(0.5, 0.5) - uv;
    // Step in UV per pixel of texture (texture is 64x64).
    float2 stepUV = toCenter / max(length(toCenter) * 64.0, 1.0) * gGodrayStridePx;
    float sum = 0.0;
    [unroll] for (int s = 0; s < 5; ++s) {
        sum += gGodrayTex.SampleLevel(gTaaSamp, uv + stepUV * (float)s, 0);
    }
    return sum * (1.0 / 5.0);
}

// Pass 2 (fill): 5 taps at fractional offsets within the pass-1 stride to fill
// in the gaps between sparse samples. Reads pass-1 intermediate (t9).
// Blends with previous-frame EMA history (t10).
float4 psmain_godray_blur_fill(VTaaOut i) : SV_Target
{
    int2 tex = (int2)i.pos.xy;
    float2 uv = (float2(tex) + 0.5) / 64.0;
    float2 toCenter = float2(0.5, 0.5) - uv;
    float2 stepUV = toCenter / max(length(toCenter) * 64.0, 1.0) * gGodrayStridePx;
    // Offsets in [-0.5..0.5] of one stride → interpolates between pass-1 samples.
    float sum = 0.0;
    const float offs[5] = { -0.4, -0.2, 0.0, 0.2, 0.4 };
    [unroll] for (int s = 0; s < 5; ++s) {
        sum += gGodrayTex.SampleLevel(gTaaSamp, uv + stepUV * offs[s], 0);
    }
    float curr = sum * (1.0 / 5.0);
    float prev = gGodrayHistTex.SampleLevel(gTaaSamp, uv, 0);
    return lerp(prev, curr, gGodrayEmaAlpha);
}

float3 PostPixelWorldDir(int2 pix, int W, int H)
{
    float2 uv = (float2(pix) + 0.5) * gInvScreenSize;
    float ndcX = uv.x * 2.0 - 1.0;
    float ndcY = 1.0 - uv.y * 2.0;
    float3 v = float3(ndcX * gAspectTanFov, ndcY * gTanHalfFovY, 1.0);
    return normalize(gCamRight * v.x + gCamUp * v.y + gCamForward * v.z);
}

float3 SkyColor(float3 rd)
{
    // Gradient only — sun disc + glow are added by the godray post pass.
    float  t = saturate(rd.y * 0.5 + 0.5);
    float3 horizon = float3(0.70, 0.75, 0.85);
    float3 zenith  = float3(0.20, 0.40, 0.80);
    float3 ground  = gFogColor;
    float3 above   = lerp(horizon, zenith, smoothstep(0.5, 1.0, t));
    return lerp(ground, above, smoothstep(0.48, 0.52, t));
}

float4 psmain_post(VTaaOut i) : SV_Target
{
    int2 pix = (int2)i.pos.xy;
    int W = (int)gScreenSize.x;
    int H = (int)gScreenSize.y;
    float4 inC = gPostIn.Load(int3(pix, 0));
    float3 c;
    if (inC.a < 0.5) {
        float3 rd = PostPixelWorldDir(pix, W, H);
        c = SkyColor(rd);
    } else {
        c = inC.rgb;
        // Unsharp mask: avg of 4 cardinal neighbours. Gears Of War 4 trick
        // (Xfest 2017, slide 55-56): 2 of the 4 neighbours are in the same 2x2
        // quad as this pixel, so derive them from ddx_fine/ddy_fine (free —
        // hardware computes derivatives across the quad with no extra texture
        // reads). Only the OTHER two neighbours need real loads.
        // QuadVector: (-1,-1)..(+1,+1) — direction away from the quad.
        float2 qv = float2(float(pix.x & 1) * 2.0 - 1.0,
                           float(pix.y & 1) * 2.0 - 1.0);
        // In-quad neighbours via derivatives. ddx_fine(A) = A_right - A_left
        // within the 2x2 lane group, so subtracting (ddx * qv.x) reflects to
        // the OTHER in-quad lane along that axis.
        float3 cH = c - ddx_fine(c) * qv.x;   // in-quad horizontal neighbour
        float3 cV = c - ddy_fine(c) * qv.y;   // in-quad vertical neighbour
        // Out-of-quad neighbours via real samples — one H, one V.
        int2 oH = int2(clamp(pix.x + (int)qv.x, 0, W - 1), pix.y);
        int2 oV = int2(pix.x, clamp(pix.y + (int)qv.y, 0, H - 1));
        float3 cOH = gPostIn.Load(int3(oH, 0)).rgb;
        float3 cOV = gPostIn.Load(int3(oV, 0)).rgb;
        float3 avg = (cH + cV + cOH + cOV) * 0.25;
        c = c + 0.5 * (c - avg);
    }
    // Smooth fade of godrays as the view turns away from the sun. Full strength
    // when sun is in front (dot >= ~0.2), zero behind. Prevents rays leaking
    // when looking 180° opposite the sun.
    float facingFade = saturate(gSunFacing * 5.0);
    if (gGodrayStrength > 0.0 && facingFade > 0.0) {
        float2 screenUV    = (float2(pix) + 0.5) * gInvScreenSize;
        float2 off         = (screenUV - gSunScreenUV) / gGodrayHalfScreenUV;
        float  rOff        = length(off);
       
        if (rOff > 1.0) 
            off *= (1.0 / rOff);
        
        float2 godrayUV    = 0.5 + off * 0.5;
        float  gr          = saturate(gGodrayTex.SampleLevel(gTaaSamp, godrayUV, 0));
        float2 dScreen     = (screenUV - gSunScreenUV) * float2(gAspect, 1.0);
        float  dist        = length(dScreen);
        float  fade        = saturate(1.0 - dist * 1.5);
        c += gGodrayTint * gr * gGodrayStrength * fade * facingFade;
    }

    
    return float4(c, 1.0);
}
