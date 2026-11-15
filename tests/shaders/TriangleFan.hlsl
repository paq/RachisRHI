float4 vertexMain(uint vertexId : SV_VertexID) : SV_Position
{
    const float2 positions[4] = {
        float2(-0.75F, 0.75F),
        float2(0.75F, 0.75F),
        float2(0.75F, -0.75F),
        float2(-0.75F, -0.75F),
    };
    return float4(positions[vertexId], 0.0F, 1.0F);
}

float4 pixelMain(uint primitiveId : SV_PrimitiveID) : SV_Target0
{
    return primitiveId == 0 ? float4(1.0F, 0.0F, 0.0F, 1.0F) : float4(0.0F, 1.0F, 0.0F, 1.0F);
}
