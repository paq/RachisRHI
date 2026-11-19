RWStructuredBuffer<float> results : register(u0, space0);
Texture2D<float4> image : register(t1, space0);
SamplerState firstSamplers[] : register(s0, space1);
SamplerState secondSamplers[] : register(s0, space2);

[shader("compute")]
[numthreads(1, 1, 1)]
void samplerArraysMain()
{
    const float2 center = float2(0.5F, 0.5F);
    results[0] = image.SampleLevel(firstSamplers[NonUniformResourceIndex(0)], center, 0.0F).x +
                 image.SampleLevel(secondSamplers[NonUniformResourceIndex(0)], center, 0.0F).x;
}
