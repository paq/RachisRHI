Texture2D<float4> sampledTexture : register(t0);
FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedbackMap : register(u1);
SamplerState sampledTextureSampler : register(s2);

[numthreads(1, 1, 1)]
void main()
{
    feedbackMap.WriteSamplerFeedbackLevel(sampledTexture, sampledTextureSampler, float2(0.125F, 0.125F), 0.0F);
}
