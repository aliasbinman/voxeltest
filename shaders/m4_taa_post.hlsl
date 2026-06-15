// Reprojection method toggle (hot-reloadable — save file to swap).
//   1 = Burnout Paradise 2-mad reproject matrix (gReprojMx/My/Mw, CPU-baked)
//   0 = reference: reconstruct world pos from depth, project with gPrevViewProj
#define USE_BURNOUT_REPROJECT 1

// m4_taa_post.hlsl — TAA resolve + post (godray/sky/tonemap). cbPerFrame (b0) is
// the shared layout from m4_frame.hlsli so all passes read matching offsets.
// Godray + shadow CB / SRVs bound to zeroed dummies — corresponding shader
// branches early-out on gGodrayStrength<=0 / gShadowEnable<0.5.

#include "m4_frame.hlsli"   // cbPerFrame (b0)

// ---- TAA bindings (match CSTiles postfx.hlsl) ----
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

// CSTiles psmain_taa, verbatim (debug `outC.r=1.0` early-returns stripped).
float4 psmain_taa(VTaaOut i) : SV_Target
{
    int2 pix = (int2)i.pos.xy;
    int W = (int)gScreenSize.x;
    int H = (int)gScreenSize.y;
    float2 currUvCenter = (float2(pix) + 0.5) * gInvScreenSize;
    float2 currUv = currUvCenter + float2(gJitter.x * 0.5, -gJitter.y * 0.5);
    float4 curSample = gTaaScene.SampleLevel(gTaaSamp, currUv, 0);
    float3 curC = curSample.rgb;
      
    float outAlpha = curSample.a;
    float  d  = gTaaDepth.Load(int3(pix, 0));
    if (d <= 0.0) {
        return float4(curC, outAlpha);
    }

#if USE_BURNOUT_REPROJECT
    // Burnout Paradise reproject — 2-mad form. Mvel rows baked CPU-side.
    // F.x = mxx*u + mxy*v + mxw, K.x = mxz; same for y, w. G = K*d + F.
    // G = homogeneous delta (prevH - currH) in HScreen-UV. Paper velocity is
    // the divide-free first-order form: V = G.xy - S*G.z (≈ exact
    // (S+G.xy)/(1+G.z) - S for small per-frame deltas).
    // Our d (gTaaDepth) = nearZ/viewZ = NDC z (reverse-Z infinite-far proj).
    float Fx = gReprojMx.x * currUvCenter.x + (gReprojMx.y * currUvCenter.y + gReprojMx.w);
    float Fy = gReprojMy.x * currUvCenter.x + (gReprojMy.y * currUvCenter.y + gReprojMy.w);
    float Fw = gReprojMw.x * currUvCenter.x + (gReprojMw.y * currUvCenter.y + gReprojMw.w);
    float3 G = float3(gReprojMx.z, gReprojMy.z, gReprojMw.z) * d + float3(Fx, Fy, Fw);
    float2 vel = G.xy - currUvCenter * G.z;
    float2 prevUv = currUvCenter + vel;
    if (any(prevUv < 0.0) || any(prevUv > 1.0)) return float4(curC, outAlpha);
#else
    // Reference reproject — reconstruct world position from linear depth,
    // project with previous frame's view-proj.
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
#endif

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

// ---- Post bindings (match CSTiles postfx.hlsl) ----
Texture2D<float>  gPostDepth : register(t6);
Texture2D<float4> gPostIn    : register(t7);

// cbGodray layout — CSTiles verbatim.
cbuffer cbGodray : register(b3)
{
    float2 gSunScreenNdc;
    float  gSunOnScreen;
    float  gGodrayEmaAlpha;
    float2 gGodrayHalfScreenUV;
    float  gSunFacing;
    float  _padG1;
    float2 gSunScreenUV;
    float  gGodrayStridePx;
    float  _padG2;
    float3 gGodrayTint;
    float  gGodrayStrength;
};
Texture2D<float> gGodrayTex     : register(t9);
Texture2D<float> gGodrayHistTex : register(t10);

// CSTiles psmain_godray_mark — runs at 64x64, samples gPostDepth at the screen
// coord under each godray-local-space sample to mark sky (1) vs occluder (0).
float4 psmain_godray_mark(VTaaOut i) : SV_Target
{
    if (gSunOnScreen < 0.5) return 1.0;
    int2 tex = (int2)i.pos.xy;
    float2 local = (float2(tex) + 0.5) / 64.0 * 2.0 - 1.0;
    if (dot(local, local) > 0.97) return 0.0;
    float2 uv = gSunScreenUV + local * gGodrayHalfScreenUV;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 1.0;
    int2 px = int2(uv * gScreenSize);
    px = clamp(px, int2(0,0), int2((int)gScreenSize.x - 1, (int)gScreenSize.y - 1));
    float d = gPostDepth.Load(int3(px, 0));
    return (d <= 0.0001) ? 1.0 : 0.0;
}

// CSTiles psmain_godray_blur — 24-tap radial blur toward (0.5, 0.5) (= sun
// position in godray local space). Reads mark tex (t9), EMA-blends with prev
// frame's blur output (t10).
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
    float  t = saturate(rd.y * 0.5 + 0.5);
    float3 horizon = float3(0.70, 0.75, 0.85);
    float3 zenith  = float3(0.20, 0.40, 0.80);
    float3 ground  = gFogColor;
    float3 above   = lerp(horizon, zenith, smoothstep(0.5, 1.0, t));
    return lerp(ground, above, smoothstep(0.48, 0.52, t));
}

// CSTiles psmain_post, verbatim.
float4 psmain_post(VTaaOut i) : SV_Target
{
    int2 pix = (int2)i.pos.xy;
    int W = (int)gScreenSize.x;
    int H = (int)gScreenSize.y;
    float4 inC = gPostIn.Load(int3(pix, 0));
    // Sky already baked into taaHist by resolve+TAA. Unsharp runs everywhere
    // (no-op on smooth sky gradient).
    float3 c = inC.rgb;
    {
        float3 cpH = gPostIn.Load(int3(pix.x + 1, pix.y, 0)).rgb;
        float3 cnH = gPostIn.Load(int3(pix.x - 1, pix.y, 0)).rgb;
        float3 cpV = gPostIn.Load(int3(pix.x, pix.y + 1, 0)).rgb;
        float3 cnV = gPostIn.Load(int3(pix.x, pix.y - 1, 0)).rgb;
        
        float3 avg = (cpH + cpV + cnH + cnV) * 0.25;
        c = c + 0.5 * (c - avg);
    }
    float facingFade = saturate(gSunFacing * 5.0);
    
    if (gGodrayStrength > 0.0 && facingFade > 0.0) {
        float2 screenUV    = (float2(pix) + 0.5) * gInvScreenSize;
        float2 off         = (screenUV - gSunScreenUV) / gGodrayHalfScreenUV;
        float  rOff        = length(off);
        if (rOff > 1.0) off *= (1.0 / rOff);
        float2 godrayUV    = 0.5 + off * 0.5;
        float  gr          = saturate(gGodrayTex.SampleLevel(gTaaSamp, godrayUV, 0));
        float2 dScreen     = (screenUV - gSunScreenUV) * float2(gAspect, 1.0);
        float  dist        = length(dScreen);
        float  fade        = saturate(1.0 - dist * 1.5);
        c += gGodrayTint * gr * gGodrayStrength * fade * facingFade;
    }

    
    c = c * gExposure;
 //   c = 1.0 - exp2(-c);
    //const float a = 2.51, ta = 0.03, tc = 2.43, td = 0.59, te = 0.14;
    //c = saturate((c * (a * c + ta)) / (c * (tc * c + td) + te));
    return float4(c, 1.0);
}
