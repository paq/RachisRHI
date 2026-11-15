struct VertexOutput
{
    float4 position : SV_Position;
    nointerpolation float4 color : COLOR0;
    uint layer : SV_RenderTargetArrayIndex;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID)
{
    const float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(-1.0, 3.0),
        float2(3.0, -1.0),
    };

    VertexOutput output;
    output.position = float4(positions[vertexId], 0.0, 1.0);
    output.color = instanceId == 0 ? float4(1.0, 0.0, 0.0, 1.0) : float4(0.0, 1.0, 0.0, 1.0);
    output.layer = instanceId;
    return output;
}

[shader("pixel")]
float4 pixelMain(VertexOutput input) : SV_Target0
{
    return input.color;
}
