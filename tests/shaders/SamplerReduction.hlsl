Texture2D<float> sourceTexture : register(t0);
SamplerState minimumSampler : register(s1);
SamplerState maximumSampler : register(s2);
RWStructuredBuffer<float> output : register(u3);

[numthreads(1, 1, 1)]
void main()
{
    const float2 center = float2(0.5, 0.5);
    output[0] = sourceTexture.SampleLevel(minimumSampler, center, 0.0);
    output[1] = sourceTexture.SampleLevel(maximumSampler, center, 0.0);
}
