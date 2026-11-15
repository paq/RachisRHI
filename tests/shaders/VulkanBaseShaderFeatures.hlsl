Texture2D<float4> featureTexture : register(t0, space0);
SamplerState featureSampler : register(s1, space0);
RWStructuredBuffer<uint> vertexCounter : register(u2, space0);
StructuredBuffer<float4> vertexPositions : register(t3, space0);

float4 fullscreenVertex(uint vertexId : SV_VertexID) : SV_Position
{
    const float2 positions[3] = {
        float2(-1.0F, -1.0F),
        float2(3.0F, -1.0F),
        float2(-1.0F, 3.0F),
    };
    return float4(positions[vertexId % 3], 0.0F, 1.0F);
}

float4 storeVertex(uint vertexId : SV_VertexID) : SV_Position
{
    InterlockedAdd(vertexCounter[0], 1);
    const float2 positions[3] = {
        float2(0.0F, 0.75F),
        float2(0.75F, -0.75F),
        float2(-0.75F, -0.75F),
    };
    return float4(positions[vertexId % 3], 0.0F, 1.0F);
}

float4 pulledVertex(uint vertexId : SV_VertexID) : SV_Position
{
    return vertexPositions[vertexId];
}

float4 solidPixel() : SV_Target0
{
    return float4(1.0F, 0.0F, 0.0F, 1.0F);
}

float4 sampleIndexPixel(float4 position : SV_Position, uint sampleIndex : SV_SampleIndex) : SV_Target0
{
    return float4(float(sampleIndex) / 4.0F, 0.0F, 0.0F, 1.0F);
}

float4 minLodPixel(float4 position : SV_Position) : SV_Target0
{
    return featureTexture.Sample(featureSampler, position.xy / 64.0F, int2(0, 0), 1.0F);
}

float4 gatherOffsetPixel(float4 position : SV_Position) : SV_Target0
{
    const int2 offset = int2(position.xy) & 3;
    return featureTexture.Gather(featureSampler, position.xy / 64.0F, offset);
}
