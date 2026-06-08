// M2 validation shader: hello-triangle via SV_VertexID, no IA, no resources.
// Confirms DXC compile + root signature + PSO + DrawInstanced path is alive.

struct VSOut
{
    float4 pos : SV_Position;
    float3 col : COLOR0;
};

VSOut vsmain(uint vid : SV_VertexID)
{
    // Big NDC triangle covering the lower-left third of the screen.
    float2 verts[3] =
    {
        float2(-0.8f, -0.8f),
        float2( 0.0f,  0.8f),
        float2( 0.8f, -0.8f),
    };
    float3 cols[3] =
    {
        float3(1.0f, 0.0f, 1.0f), // magenta
        float3(0.2f, 1.0f, 0.2f), // green
        float3(0.2f, 0.5f, 1.0f), // blue
    };
    VSOut o;
    o.pos = float4(verts[vid], 0.0f, 1.0f);
    o.col = cols[vid];
    return o;
}

float4 psmain(VSOut i) : SV_Target
{
    return float4(i.col, 1.0f);
}
