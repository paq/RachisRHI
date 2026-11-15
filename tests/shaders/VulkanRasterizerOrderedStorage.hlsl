float4 vertexMain(uint vertexId : SV_VertexID) : SV_Position
{
    const float2 positions[3] = {
        float2(-1.0F, -1.0F),
        float2(3.0F, -1.0F),
        float2(-1.0F, 3.0F),
    };
    return float4(positions[vertexId], 0.0F, 1.0F);
}
