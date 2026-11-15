float4 pointPosition(uint vertexId)
{
    return float4(-1.0F + (4.0F * float(vertexId) + 3.0F) / 8.0F, 0.125F, 0.0F, 1.0F);
}

float4 vertexMain(uint vertexId : SV_VertexID) : SV_Position
{
    return pointPosition(vertexId);
}

#ifdef __spirv__
struct SizedPointOutput
{
    float4 position : SV_Position;
    [[vk::builtin("PointSize")]] float pointSize : PSIZE;
};

SizedPointOutput sizedVertexMain(uint vertexId : SV_VertexID)
{
    SizedPointOutput output;
    output.position = pointPosition(vertexId);
    output.pointSize = 1.0F;
    return output;
}
#endif

float4 pixelMain() : SV_Target0
{
    return float4(1.0F, 1.0F, 1.0F, 1.0F);
}
