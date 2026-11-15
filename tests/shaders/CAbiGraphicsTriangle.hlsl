#ifdef RACHIS_METAL
[[vk::push_constant]]
#endif
cbuffer PushConstants : register(b0, space4)
{
    float4 tint;
};

struct VertexOutput
{
    float4 position : SV_Position;
};

#ifdef RACHIS_METAL
[shader("vertex")]
#endif
VertexOutput vertexMain(uint vertexId : SV_VertexID)
{
    float2 points[3] = {float2(-1.0, -1.0), float2(-1.0, 3.0), float2(3.0, -1.0)};
    VertexOutput output;
    output.position = float4(points[vertexId], 0.0, 1.0);
    return output;
}

#ifdef RACHIS_METAL
[shader("pixel")]
#endif
float4 pixelMain() : SV_Target0
{
    return tint;
}
