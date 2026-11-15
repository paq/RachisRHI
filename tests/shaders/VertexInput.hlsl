struct VertexOutput
{
    float4 position : SV_Position;
    float4 color : TEXCOORD0;
};

VertexOutput vertexMain(float2 position : TEXCOORD0, float2 offset : TEXCOORD1, float4 color : TEXCOORD2)
{
    VertexOutput output;
    output.position = float4(position + offset, 0.0, 1.0);
    output.color = color;
    return output;
}

VertexOutput matrixVertexMain(float2 position : TEXCOORD0, float4x4 world : TEXCOORD1)
{
    VertexOutput output;
    output.position = float4(position, 0.0, 1.0);
    output.color = mul(world, float4(1.0, 1.0, 1.0, 1.0));
    return output;
}

VertexOutput arrayVertexMain(float2 position : TEXCOORD0, float4 weights[2] : TEXCOORD1)
{
    VertexOutput output;
    output.position = float4(position, 0.0, 1.0);
    output.color = weights[0] + weights[1];
    return output;
}

float4 pixelMain(VertexOutput input) : SV_Target0
{
    return input.color;
}
