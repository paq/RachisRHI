Texture2D<float> sourceTexture : register(t0);
SamplerState pointSampler : register(s1);
RWStructuredBuffer<float> output : register(u2);

[numthreads(1, 1, 1)]
void main()
{
    output[0] = sourceTexture.SampleLevel(pointSampler, float2(0.25, 0.25), 0.0);
    output[1] = sourceTexture.SampleLevel(pointSampler, float2(0.75, 0.25), 0.0);
    output[2] = sourceTexture.SampleLevel(pointSampler, float2(0.25, 0.75), 0.0);
    output[3] = sourceTexture.SampleLevel(pointSampler, float2(0.75, 0.75), 0.0);
}
