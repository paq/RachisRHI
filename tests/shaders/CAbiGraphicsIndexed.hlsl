struct VertexInput
{
    float2 position : TEXCOORD0;
    float4 color : TEXCOORD1;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float4 color : COLOR0;
};

#ifdef RACHIS_METAL
[shader("vertex")]
#endif
VertexOutput vertexMain(VertexInput input)
{
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

#ifdef RACHIS_METAL
[shader("pixel")]
#endif
float4 pixelMain(VertexOutput input) : SV_Target0
{
    return input.color;
}
