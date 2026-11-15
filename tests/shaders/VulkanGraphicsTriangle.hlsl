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

VertexOutput directVertex(uint vertexId : SV_VertexID)
{
    const float2 positions[3] = {
        float2(0.0F, 0.75F),
        float2(0.75F, -0.75F),
        float2(-0.75F, -0.75F),
    };

    VertexOutput output;
    output.position = float4(positions[vertexId], 0.0F, 1.0F);
    return output;
}

VertexOutput depthBiasVertex(uint vertexId : SV_VertexID)
{
    VertexOutput output = directVertex(vertexId);
    output.position.z = 0.5F;
    return output;
}

VertexOutput inputVertex(float2 position : TEXCOORD0)
{
    VertexOutput output;
    output.position = float4(position, 0.0F, 1.0F);
    return output;
}

float4 pixelMain() : SV_Target0
{
    return pushConstants.tint;
}
