// M4a — minimum-viable PointCS_Block compute rasterizer for the LW path.
// Stripped port of the relevant pieces from lodworld.hlsl: no AO, no shadows,
// no fog, no LOD fade, no clusters tint, no LodViz mode. Just pass1 atomic
// linear-depth + pass2 color + resolve PS.
//
// Bindings:
//   b0  CBLwFrame  — gViewProj, gMode
//   b1  CBLwCS    — gVwSize, gLwPointCount, gLwNumWorkItems, gLwLodIdx
//   t0  ChunkInfo (struct)
//   t1  Palette   (uint)
//   t2  BlockPos  (uint)
//   t3  BlockCol  (uint2)
//   t4  WorkItems (uint4)
//   t5  DepthSrv  (Texture2D<uint>) — pass2 only
//   u0  VisUav    (RWTexture2D<uint>) — pass1 = depth, pass2 = ARGB color
//
// Resolve:
//   t0  DepthSrv (Texture2D<uint>)
//   t1  ColorSrv (Texture2D<uint>)

cbuffer CBLwFrame : register(b0)
{
    row_major float4x4 gViewProj;
    uint  gMode;       // ShadingMode (unused here; gMode==4 picks LOD tints in stock shader)
    uint3 _padFrame;
};

cbuffer CBLwCS : register(b1)
{
    uint2 gVwSize;
    uint  gLwPointCount;
    uint  gLwNumWorkItems;
    uint  gLwLodIdx;
    uint3 _padCs;
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

RWTexture2D<uint>             gLwVisUav     : register(u0);

// Linear-depth encoding — uniform precision over [0, 100km] view-Z.
static const float kLinDepthFar   = 100000.0;
static const float kLinDepthMaxF  = 4294967294.0; // 0xFFFFFFFE
static const float kLinDepthScale = kLinDepthMaxF / kLinDepthFar;
uint EncodeLinDepth(float viewZ)
{
    return (uint)clamp(viewZ * kLinDepthScale, 0.0, kLinDepthMaxF);
}

// gid -> work item via binary search over the cumulative firstThread (.w).
uint4 LookupItem(uint gid)
{
    uint lo = 0u;
    uint hi = gLwNumWorkItems;
    while (lo + 1u < hi)
    {
        uint mid = (lo + hi) >> 1u;
        if (gLwWorkItems[mid].w <= gid) lo = mid;
        else                            hi = mid;
    }
    return gLwWorkItems[lo];
}

// ============================================================
// Pass 1 — write linear depth via InterlockedMin.
// ============================================================
[numthreads(64, 1, 1)]
void csmain_pass1_depth(uint3 dt : SV_DispatchThreadID)
{
    uint gid = dt.x;
    if (gid >= gLwPointCount) return;
    uint4 item = LookupItem(gid);
    uint slot        = item.x;
    uint blockBase   = item.y;
    uint count       = item.z;
    uint firstThread = item.w;
    if (gid - firstThread >= count) return;

    LwChunkInfo ci = gLwChunkInfos[slot];
    uint pack0 = gLwBlockPos[blockBase + (gid - firstThread)];

    uint bx  = (pack0 >>  0) & 0xFFu;
    uint by  = (pack0 >>  8) & 0xFFu;
    uint bz  = (pack0 >> 16) & 0xFFu;
    uint occ = (pack0 >> 24) & 0xFFu;

    [unroll]
    for (uint i = 0; i < 8; ++i)
    {
        if (((occ >> i) & 1u) == 0u) continue;
        uint lx = (i >> 0) & 1u;
        uint ly = (i >> 1) & 1u;
        uint lz = (i >> 2) & 1u;
        float3 local = float3(
            (float)(bx * 2u + lx),
            (float)(by * 2u + ly),
            (float)(bz * 2u + lz)) + 0.5;
        float3 world = ci.worldOrigin + local * ci.lodScale;

        float4 clip = mul(float4(world, 1.0), gViewProj);
        if (clip.w <= 0.0) continue;
        float3 ndc = clip.xyz / clip.w;
        if (ndc.x < -1.0 || ndc.x > 1.0 ||
            ndc.y < -1.0 || ndc.y > 1.0 ||
            ndc.z <  0.0 || ndc.z > 1.0) continue;
        int2 pix;
        pix.x = (int)((ndc.x * 0.5 + 0.5) * (float)gVwSize.x);
        pix.y = (int)((-ndc.y * 0.5 + 0.5) * (float)gVwSize.y);
        if (pix.x < 0 || pix.x >= (int)gVwSize.x ||
            pix.y < 0 || pix.y >= (int)gVwSize.y) continue;

        uint linD = EncodeLinDepth(clip.w);
        InterlockedMin(gLwVisUav[pix], linD);
    }
}

// ============================================================
// Pass 2 — write ARGB color where this voxel is the depth winner.
// ============================================================
[numthreads(64, 1, 1)]
void csmain_pass2_color(uint3 dt : SV_DispatchThreadID)
{
    uint gid = dt.x;
    if (gid >= gLwPointCount) return;
    uint4 item = LookupItem(gid);
    uint slot        = item.x;
    uint blockBase   = item.y;
    uint count       = item.z;
    uint firstThread = item.w;
    if (gid - firstThread >= count) return;

    LwChunkInfo ci  = gLwChunkInfos[slot];
    uint   pack0    = gLwBlockPos[blockBase + (gid - firstThread)];
    uint2  cols     = gLwBlockCol[blockBase + (gid - firstThread)];

    uint bx  = (pack0 >>  0) & 0xFFu;
    uint by  = (pack0 >>  8) & 0xFFu;
    uint bz  = (pack0 >> 16) & 0xFFu;
    uint occ = (pack0 >> 24) & 0xFFu;

    [unroll]
    for (uint i = 0; i < 8; ++i)
    {
        if (((occ >> i) & 1u) == 0u) continue;
        uint palIdx = (i < 4u) ? ((cols.x >> (i * 8u)) & 0xFFu)
                                : ((cols.y >> ((i - 4u) * 8u)) & 0xFFu);
        uint lx = (i >> 0) & 1u;
        uint ly = (i >> 1) & 1u;
        uint lz = (i >> 2) & 1u;
        float3 local = float3(
            (float)(bx * 2u + lx),
            (float)(by * 2u + ly),
            (float)(bz * 2u + lz)) + 0.5;
        float3 world = ci.worldOrigin + local * ci.lodScale;

        float4 clip = mul(float4(world, 1.0), gViewProj);
        if (clip.w <= 0.0) continue;
        float3 ndc = clip.xyz / clip.w;
        if (ndc.x < -1.0 || ndc.x > 1.0 ||
            ndc.y < -1.0 || ndc.y > 1.0 ||
            ndc.z <  0.0 || ndc.z > 1.0) continue;
        int2 pix;
        pix.x = (int)((ndc.x * 0.5 + 0.5) * (float)gVwSize.x);
        pix.y = (int)((-ndc.y * 0.5 + 0.5) * (float)gVwSize.y);
        if (pix.x < 0 || pix.x >= (int)gVwSize.x ||
            pix.y < 0 || pix.y >= (int)gVwSize.y) continue;

        uint myLin    = EncodeLinDepth(clip.w);
        uint winDepth = gLwDepthSrv.Load(int3(pix, 0));
        const uint kLinDepthSlop = 64u;
        if (myLin > winDepth + kLinDepthSlop) continue;

        uint argb32;
        // Palette is stored ABGR (low byte = R). Reconstruct ARGB for storage.
        uint colPck = gLwPalette[ci.paletteBase + palIdx];
        uint rR = (colPck >>  0) & 0xFFu;
        uint rG = (colPck >>  8) & 0xFFu;
        uint rB = (colPck >> 16) & 0xFFu;
        argb32 = 0xFF000000u | (rB << 16) | (rG << 8) | rR;
        gLwVisUav[pix] = argb32;
    }
}

// ============================================================
// Resolve — fullscreen triangle reads depth + color, outputs RGB.
// ============================================================
struct VOut { float4 pos : SV_Position; };
VOut vsmain_resolve(uint vid : SV_VertexID)
{
    float2 p = float2((vid == 1u) ? 3.0 : -1.0,
                      (vid == 2u) ? 3.0 : -1.0);
    VOut o;
    o.pos = float4(p, 0.0, 1.0);
    return o;
}

Texture2D<uint> gResolveDepthSrv : register(t0);
Texture2D<uint> gResolveColorSrv : register(t1);

float4 psmain_resolve(VOut i) : SV_Target
{
    int2 pix = int2(i.pos.xy);
    uint depth = gResolveDepthSrv.Load(int3(pix, 0));
    if (depth == 0xFFFFFFFFu) discard;
    uint argb = gResolveColorSrv.Load(int3(pix, 0));
    float r = (float)( argb        & 0xFFu) / 255.0;
    float g = (float)((argb >>  8) & 0xFFu) / 255.0;
    float b = (float)((argb >> 16) & 0xFFu) / 255.0;
    return float4(r, g, b, 1.0);
}
