struct PushConstants
{
    float4 tint;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConstants;

struct VertexOutput
{
    float4 position : SV_Position;
};

VertexOutput vertexMain(uint vertexId : SV_VertexID)
{
    float2 points[3] = {float2(-1.0, -1.0), float2(-1.0, 3.0), float2(3.0, -1.0)};
    VertexOutput output;
    output.position = float4(points[vertexId], 0.0, 1.0);
    return output;
}

float4 pixelMain() : SV_Target0
{
    return pushConstants.tint;
}
