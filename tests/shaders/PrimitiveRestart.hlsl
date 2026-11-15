struct VertexOutput
{
    float4 position : SV_Position;
    float4 color : TEXCOORD0;
};

VertexOutput vertexMain(float2 position : TEXCOORD0, float4 color : TEXCOORD1)
{
    VertexOutput output;
    output.position = float4(position, 0.0, 1.0);
    output.color = color;
    return output;
}

float4 pixelMain(VertexOutput input) : SV_Target0
{
    return input.color;
}
