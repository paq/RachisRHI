struct VertexOutput
{
    float4 position : SV_Position;
    nointerpolation uint viewId : TEXCOORD0;
};

VertexOutput vsMain(uint vertexId : SV_VertexID, uint viewId : SV_ViewID)
{
    const float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(-1.0, 3.0),
        float2(3.0, -1.0),
    };
    VertexOutput result;
    result.position = float4(positions[vertexId], 0.0, 1.0);
    result.viewId = viewId;
    return result;
}

float4 psMain(VertexOutput input) : SV_Target0
{
    if (input.viewId == 1)
    {
        return float4(1.0, 0.0, 0.0, 1.0);
    }
    if (input.viewId == 2)
    {
        return float4(0.0, 1.0, 0.0, 1.0);
    }
    return float4(0.0, 0.0, 1.0, 1.0);
}
