struct DrawIdOutput
{
    float4 position : SV_Position;
    nointerpolation float4 ids : TEXCOORD0;
};

[shader("vertex")]
DrawIdOutput drawIdVertex(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID)
{
    const uint triangleId = vertexId / 3u;
    const uint corner = vertexId % 3u;
    const float column = float((triangleId & 7u) + 8u * (instanceId & 7u));
    const float center = -1.0 + (2.0 * column + 1.0) / 64.0;
    const float halfPixel = 1.0 / 64.0;
    float2 position = float2(center, 3.0);
    if (corner == 0u)
    {
        position = float2(center - 0.8 * halfPixel, -3.0);
    }
    else if (corner == 1u)
    {
        position = float2(center + 0.8 * halfPixel, -3.0);
    }
    DrawIdOutput output;
    output.position = float4(position, 0.0, 1.0);
    output.ids = float4(float(triangleId) / 255.0, float(instanceId) / 255.0, 0.0, 1.0);
    return output;
}

[shader("pixel")]
float4 drawIdPixel(DrawIdOutput input) : SV_Target0
{
    return input.ids;
}
