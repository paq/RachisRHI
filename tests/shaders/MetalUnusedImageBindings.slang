[[vk::push_constant]]
cbuffer PushConstants : register(b0, space4)
{
    float4 positions[6];
};

Texture2D<float4> imageTexture : register(t0, space0);
SamplerState imageSampler : register(s1, space0);

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexId : SV_VertexID)
{
    VertexOutput output;
    output.position = positions[vertexId % 6];
    output.uv = output.position.xy * 0.5F + 0.5F;
    return output;
}

[shader("fragment")]
float4 pixelMain(VertexOutput input) : SV_Target0
{
    return imageTexture.Sample(imageSampler, input.uv);
}
