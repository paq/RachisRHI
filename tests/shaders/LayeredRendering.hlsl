struct VertexOutput
{
    float4 position : SV_Position;
    nointerpolation float4 color : COLOR0;
    uint layer : SV_RenderTargetArrayIndex;
};

VertexOutput vertexMain(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID)
{
    const float2 positions[3] = {
        float2(-1.0F, -1.0F),
        float2(-1.0F, 3.0F),
        float2(3.0F, -1.0F),
    };

    VertexOutput output;
    output.position = float4(positions[vertexId], 0.0F, 1.0F);
    output.color = instanceId == 0 ? float4(1.0F, 0.0F, 0.0F, 1.0F) : float4(0.0F, 1.0F, 0.0F, 1.0F);
    output.layer = instanceId;
    return output;
}

float4 pixelMain(VertexOutput input) : SV_Target0
{
    return input.color;
}
