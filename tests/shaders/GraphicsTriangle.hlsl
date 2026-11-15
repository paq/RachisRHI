cbuffer PushConstants : register(b0, space4)
{
    float4 tint;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float4 capture : TEXCOORD0;
};

VertexOutput directVertex(uint vertexId : SV_VertexID)
{
    const float2 positions[3] = {
        float2(0.0F, 0.75F),
        float2(0.75F, -0.75F),
        float2(-0.75F, -0.75F),
    };

    VertexOutput output;
    output.position = float4(positions[vertexId], 0.0F, 1.0F);
    output.capture = output.position;
    return output;
}

struct ShadingRateVertexOutput
{
    float4 position : SV_Position;
    uint shadingRate : SV_ShadingRate;
};

ShadingRateVertexOutput shadingRateVertex(uint vertexId : SV_VertexID)
{
    ShadingRateVertexOutput output;
    output.position = directVertex(vertexId).position;

    output.shadingRate = 5;
    return output;
}

VertexOutput unclippedVertex(uint vertexId : SV_VertexID)
{
    VertexOutput output = directVertex(vertexId);
    output.position.z = 2.0F;
    return output;
}

VertexOutput depthBiasVertex(uint vertexId : SV_VertexID)
{
    VertexOutput output = directVertex(vertexId);
    output.position.z = 0.5F;
    return output;
}

VertexOutput inputVertex(float2 position : TEXCOORD0)
{
    VertexOutput output;
    output.position = float4(position, 0.0F, 1.0F);
    output.capture = output.position;
    return output;
}

float4 pixelMain() : SV_Target0
{
    return tint;
}
