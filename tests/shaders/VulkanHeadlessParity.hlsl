[[vk::binding(0, 0)]] Texture2D<float4> sourceTexture : register(t0);
[[vk::binding(1, 0)]] SamplerState nearestSampler : register(s0);

struct TexturedOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

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

float4 texturedPixel(TexturedOutput input) : SV_Target0
{
    return sourceTexture.Sample(nearestSampler, input.uv);
}
