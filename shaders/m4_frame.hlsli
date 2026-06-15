// ============================================================================
// m4_frame.hlsli — the per-frame constant buffer (b0), shared by every m4 shader.
// One byte layout so the LW techs, the dilate, and the TAA/post pass all read
// matching offsets from the single CBFrame the renderer uploads.
// ============================================================================
#ifndef M4_FRAME_HLSLI
#define M4_FRAME_HLSLI

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
    float    gAoStrength;              // 500  baked AO darkening (0..1)
    float2   _padPC;                   // 504..512
    // Burnout Paradise reproject matrix rows (HScreen-UV). Mvel = Mh1_to_h0 - I.
    float4   gReprojMx;                // 512  (mxx, mxy, mxz, mxw)
    float4   gReprojMy;                // 528  (myx, myy, myz, myw)
    float4   gReprojMw;                // 544  (mwx, mwy, mwz, mww)
    float4   _padReproj;               // 560..576
};

#endif // M4_FRAME_HLSLI
