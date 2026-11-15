struct VertexOutput
{
    float4 position : SV_Position;
};

VertexOutput primitiveIdVertex(uint vertexId : SV_VertexID)
{
    const float2 positions[6] = {
        float2(-0.9F, 0.7F),
        float2(-0.1F, 0.7F),
        float2(-0.5F, -0.7F),
        float2(0.1F, 0.7F),
        float2(0.9F, 0.7F),
        float2(0.5F, -0.7F),
    };

    VertexOutput output;
    output.position = float4(positions[vertexId], 0.0F, 1.0F);
    return output;
}

float4 primitiveIdPixel(uint primitiveId : SV_PrimitiveID) : SV_Target0
{
    if (primitiveId == 0)
    {
        return float4(1.0F, 0.0F, 0.0F, 1.0F);
    }
    if (primitiveId == 1)
    {
        return float4(0.0F, 1.0F, 0.0F, 1.0F);
    }
    return float4(0.0F, 0.0F, 1.0F, 1.0F);
}

VertexOutput barycentricVertex(uint vertexId : SV_VertexID)
{
    const float2 positions[3] = {
        float2(0.0F, 0.9F),
        float2(0.9F, -0.9F),
        float2(-0.9F, -0.9F),
    };

    VertexOutput output;
    output.position = float4(positions[vertexId], 0.0F, 1.0F);
    return output;
}

float4 barycentricPixel(float3 barycentrics : SV_Barycentrics) : SV_Target0
{
    if (barycentrics.x > barycentrics.y && barycentrics.x > barycentrics.z)
    {
        return float4(1.0F, 0.0F, 0.0F, 1.0F);
    }
    if (barycentrics.y > barycentrics.z)
    {
        return float4(0.0F, 1.0F, 0.0F, 1.0F);
    }
    return float4(0.0F, 0.0F, 1.0F, 1.0F);
}
