struct VertexOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID)
{
    static const float2 positions[3] = {
        float2(0.0, 0.72), float2(-0.72, -0.72), float2(0.72, -0.72)
    };
    VertexOutput output;
    const float xOffset = instanceId == 0 ? -0.45 : 0.45;
    output.position = float4(positions[vertexId].x * 0.5 + xOffset, positions[vertexId].y, 0.0, 1.0);
    output.color = instanceId == 0 ? float3(1, 0, 0) : float3(0, 0, 1);
    return output;
}

[shader("pixel")]
float4 pixelMain(VertexOutput input) : SV_Target0
{
    return float4(input.color, 1.0);
}
