[[vk::push_constant]]
cbuffer VertexConstants : register(b0, space4)
{
    float4 vertexPosition;
};

Texture2D<float4> imageTexture : register(t0, space0);
SamplerState imageSampler : register(s1, space0);

struct VertexOutput
{
    float4 position : SV_Position;
};

[shader("vertex")]
VertexOutput vertexMain()
{
    VertexOutput output;
    output.position = vertexPosition;
    return output;
}

[shader("fragment")]
float4 pixelMain(VertexOutput input) : SV_Target0
{
    return imageTexture.Sample(imageSampler, input.position.xy);
}
