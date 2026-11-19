StructuredBuffer<uint> scales[] : register(t0, space0);

struct ArrayVertexOutput
{
    float4 position : SV_Position;
};

[shader("vertex")]
ArrayVertexOutput arrayVertex(uint vertexId : SV_VertexID)
{
    const float2 corner = float2(float((vertexId << 1U) & 2U), float(vertexId & 2U));
    ArrayVertexOutput output;
    output.position = float4((corner * 2.0F - 1.0F) * float(scales[NonUniformResourceIndex(0)][0]), 0.0F, 1.0F);
    return output;
}

[shader("pixel")]
float4 arrayPixel() : SV_Target0
{
    return float4(1.0F, 1.0F, 1.0F, 1.0F);
}
