struct VertexOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexId : SV_VertexID)
{
    const float2 positions[3] = {float2(0.0, 0.75), float2(0.75, -0.75), float2(-0.75, -0.75)};
    const float3 colors[3] = {float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0), float3(0.0, 0.0, 1.0)};

    VertexOutput output;
    output.position = float4(positions[vertexId], 0.0, 1.0);
    output.color = colors[vertexId];
    return output;
}

[shader("pixel")]
float4 pixelMain(VertexOutput input) : SV_Target0
{
    return float4(input.color, 1.0);
}
