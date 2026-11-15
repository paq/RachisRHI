#if defined(RACHIS_PARITY_PIXEL)
[[vk::push_constant]]
cbuffer PushConstants : register(b0, space4)
{
    float4 tint;
};
#endif

#if defined(RACHIS_PARITY_TEXTURED)
Texture2D<float4> sourceTexture : register(t0, space1);
SamplerState nearestSampler : register(s1, space3);
#endif

struct VertexOutput
{
    float4 position : SV_Position;
};

struct TexturedOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

#if defined(RACHIS_PARITY_VERTEX)
[shader("vertex")]
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

[shader("vertex")]
VertexOutput depthBiasVertex(uint vertexId : SV_VertexID)
{
    VertexOutput output = directVertex(vertexId);
    output.position.z = 0.5F;
    return output;
}
#endif

#if defined(RACHIS_PARITY_PIXEL)
[shader("pixel")]
float4 pixelMain() : SV_Target0
{
    return tint;
}
#endif

#if defined(RACHIS_PARITY_TEXTURED_VERTEX)
[shader("vertex")]
TexturedOutput texturedVertex(uint vertexId : SV_VertexID)
{
    const float2 positions[3] = {
        float2(0.0F, 0.75F),
        float2(0.75F, -0.75F),
        float2(-0.75F, -0.75F),
    };

    TexturedOutput output;
    output.position = float4(positions[vertexId], 0.0F, 1.0F);
    output.uv = float2(0.25F, 0.25F);
    return output;
}
#endif

#if defined(RACHIS_PARITY_TEXTURED)
[shader("pixel")]
float4 texturedPixel(TexturedOutput input) : SV_Target0
{
    return sourceTexture.Sample(nearestSampler, input.uv);
}
#endif
