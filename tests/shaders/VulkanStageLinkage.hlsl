static const float2 trianglePositions[3] = {
    float2(0.0F, 0.75F),
    float2(0.75F, -0.75F),
    float2(-0.75F, -0.75F),
};

struct Float4Output
{
    float4 position : SV_Position;
    [[vk::location(0)]] float4 value : TEXCOORD0;
};

struct Float2Output
{
    float4 position : SV_Position;
    [[vk::location(0)]] float2 value : TEXCOORD0;
};

struct Uint4Output
{
    float4 position : SV_Position;
    [[vk::location(0)]] nointerpolation uint4 value : TEXCOORD0;
};

struct LayoutOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float2 values[2] : TEXCOORD0;
    [[vk::location(2)]] float2x3 matrix : TEXCOORD2;
    [[vk::location(4)]] nointerpolation double3 wide : TEXCOORD4;
};

Float4Output float4Vertex(uint vertexId : SV_VertexID)
{
    Float4Output output;
    output.position = float4(trianglePositions[vertexId], 0.0F, 1.0F);
    output.value = float4(1.0F, 0.0F, 0.0F, 1.0F);
    return output;
}

Float2Output float2Vertex(uint vertexId : SV_VertexID)
{
    Float2Output output;
    output.position = float4(trianglePositions[vertexId], 0.0F, 1.0F);
    output.value = float2(1.0F, 0.0F);
    return output;
}

Uint4Output uint4Vertex(uint vertexId : SV_VertexID)
{
    Uint4Output output;
    output.position = float4(trianglePositions[vertexId], 0.0F, 1.0F);
    output.value = uint4(1, 0, 0, 1);
    return output;
}

LayoutOutput layoutVertex(uint vertexId : SV_VertexID)
{
    LayoutOutput output;
    output.position = float4(trianglePositions[vertexId], 0.0F, 1.0F);
    output.values[0] = float2(0.0F, 1.0F);
    output.values[1] = float2(2.0F, 3.0F);
    output.matrix = float2x3(0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F);
    output.wide = double3(vertexId, 0.0, 1.0);
    return output;
}

float4 float4Pixel([[vk::location(0)]] float4 value : TEXCOORD0) : SV_Target0
{
    return value;
}

float4 float2Pixel([[vk::location(0)]] float2 value : TEXCOORD0) : SV_Target0
{
    return float4(value, 0.0F, 1.0F);
}

float4 location1Pixel([[vk::location(1)]] float4 value : TEXCOORD1) : SV_Target0
{
    return value;
}
