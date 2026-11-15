Texture2D<float4> generatedChain : register(t0, space0);
RWStructuredBuffer<float4> readValue : register(u1, space0);

static const uint readMipLevel = 2;

float4 fullscreenVertex(uint vertexId : SV_VertexID) : SV_Position
{
    const float2 positions[3] = {
        float2(-1.0F, -1.0F),
        float2(-1.0F, 3.0F),
        float2(3.0F, -1.0F),
    };
    return float4(positions[vertexId], 0.0F, 1.0F);
}

float4 pixelMain() : SV_Target0
{
    return generatedChain.Load(int3(0, 0, readMipLevel));
}

[numthreads(1, 1, 1)]
void computeMain()
{
    readValue[0] = generatedChain.Load(int3(0, 0, readMipLevel));
}
