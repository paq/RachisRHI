struct DistancePushConstants
{
    float clipDistance;
    float cullDistance;
};

#ifdef __spirv__
[[vk::push_constant]] DistancePushConstants distances;
#else
ConstantBuffer<DistancePushConstants> distances : register(b0, space4);
#endif

struct VertexOutput
{
    float4 position : SV_Position;
    float clipDistance : SV_ClipDistance;
    float cullDistance : SV_CullDistance;
};

VertexOutput vertexMain(uint vertexId : SV_VertexID)
{
    const float2 positions[3] = {
        float2(-1.0F, -1.0F),
        float2(-1.0F, 3.0F),
        float2(3.0F, -1.0F),
    };

    VertexOutput output;
    output.position = float4(positions[vertexId], 0.0F, 1.0F);
    output.clipDistance = distances.clipDistance;
    output.cullDistance = distances.cullDistance;
    return output;
}

float4 pixelMain() : SV_Target0
{
    return float4(1.0F, 0.0F, 0.0F, 1.0F);
}
