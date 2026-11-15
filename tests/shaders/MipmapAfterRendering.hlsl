struct VertexOutput
{
    float4 position : SV_Position;
};

#ifdef RACHIS_METAL
[shader("vertex")]
#endif
VertexOutput fullscreenVertex(uint vertexId : SV_VertexID)
{
    const float2 corner = float2(float((vertexId << 1) & 2U), float(vertexId & 2U));
    VertexOutput output;
    output.position = float4(corner * 2.0F - 1.0F, 0.0F, 1.0F);
    return output;
}

#ifdef RACHIS_METAL
[shader("pixel")]
#endif
float4 slowPixel(VertexOutput input) : SV_Target0
{
    float value = frac(input.position.x * 0.0013F + input.position.y * 0.0007F);
    [loop]
    for (int index = 0; index < 4000; ++index)
    {
        value = frac(value * 1.0001F + 0.00001F * sin(value * 12.0F));
    }
    return float4(1.0F, 0.5F, 0.25F + saturate(value * 1e-30F), 1.0F);
}

#ifdef RACHIS_METAL
[shader("pixel")]
#endif
float4 blendPixel() : SV_Target0
{
    return float4(0.0F, 0.25F, 0.5F, 0.0F);
}
