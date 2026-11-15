Texture2D<float4> sourceTexture : register(t0, space0);
SamplerState sourceSampler : register(s1, space0);

struct SampleOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

SampleOutput vertexMain(uint vertexId : SV_VertexID)
{
    const float2 positions[3] = {
        float2(-1.0F, -1.0F),
        float2(3.0F, -1.0F),
        float2(-1.0F, 3.0F),
    };

    SampleOutput output;
    output.position = float4(positions[vertexId], 0.0F, 1.0F);
    output.uv = float2(positions[vertexId].x + 1.0F, 1.0F - positions[vertexId].y) * 0.5F;
    return output;
}

float4 pixelMain(SampleOutput input) : SV_Target0
{
    return sourceTexture.Sample(sourceSampler, input.uv);
}
