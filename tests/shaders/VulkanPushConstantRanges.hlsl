struct VertexOutput
{
    float4 position : SV_Position;
};

#if defined(RACHIS_PUSH_VERTEX_LOW) || defined(RACHIS_PUSH_VERTEX_HIGH)

struct PushConstants
{
#if defined(RACHIS_PUSH_VERTEX_HIGH)
    [[vk::offset(16)]]
#endif
    float4 offset;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConstants;

VertexOutput vertexMain(uint vertexId : SV_VertexID)
{
    const float2 positions[3] = {
        float2(0.0F, 0.75F),
        float2(0.75F, -0.75F),
        float2(-0.75F, -0.75F),
    };

    VertexOutput output;
    output.position = float4(positions[vertexId] + pushConstants.offset.xy, 0.0F, 1.0F);
    return output;
}

#else

struct PushConstants
{
    float4 base;
#if defined(RACHIS_PUSH_PIXEL_SPLIT)
    [[vk::offset(32)]]
#endif
    float4 tint;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConstants;

float4 pixelMain() : SV_Target0
{
    return pushConstants.base + pushConstants.tint;
}

#endif
