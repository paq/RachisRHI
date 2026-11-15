RWStructuredBuffer<half2> halfValues : register(u0, space0);
RWStructuredBuffer<uint> doubleValues : register(u1, space0);

struct HalfStageVaryings
{
    float4 position : SV_Position;
    half4 color : COLOR0;
};

struct FloatStageVaryings
{
    float4 position : SV_Position;
    float4 color : COLOR0;
};

static const float2 fullscreenPositions[3] = {
    float2(-1.0F, -1.0F),
    float2(3.0F, -1.0F),
    float2(-1.0F, 3.0F),
};

[numthreads(1, 1, 1)]
void halfStorageCompute(uint3 id : SV_DispatchThreadID)
{
    halfValues[id.x] = halfValues[id.x] * (half)2.0 + (half)1.0;
}

[numthreads(1, 1, 1)]
void doubleCompute(uint3 id : SV_DispatchThreadID)
{
    const double value = (double)doubleValues[id.x];
    doubleValues[id.x] = (uint)(value * 2.0 + 1.0);
}

HalfStageVaryings halfStageVertex(uint vertexId : SV_VertexID)
{
    HalfStageVaryings output;
    output.position = float4(fullscreenPositions[vertexId % 3], 0.0F, 1.0F);
    output.color = half4((half)0.25, (half)0.5, (half)0.75, (half)1.0);
    return output;
}

float4 halfStagePixel(HalfStageVaryings input) : SV_Target0
{
    return float4(input.color);
}

FloatStageVaryings floatStageVertex(uint vertexId : SV_VertexID)
{
    FloatStageVaryings output;
    output.position = float4(fullscreenPositions[vertexId % 3], 0.0F, 1.0F);
    output.color = float4(0.25F, 0.5F, 0.75F, 1.0F);
    return output;
}

float4 floatStagePixel(FloatStageVaryings input) : SV_Target0
{
    return input.color;
}
