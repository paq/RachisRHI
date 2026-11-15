struct VertexOutput
{
    float4 position : SV_Position;
    float4 color : COLOR0;
    nointerpolation uint2 ids : TEXCOORD1;
    float2 uv : TEXCOORD2;
};

VertexOutput vertexMain(uint vertexId : SV_VertexID)
{
    VertexOutput output;
    output.position = float4(float(vertexId & 1U), float(vertexId >> 1U), 0.0, 1.0);
    output.color = float4(1.0, 0.5, 0.25, 1.0);
    output.ids = uint2(vertexId, 7U);
    output.uv = float2(0.0, 1.0);
    return output;
}

float4 pixelMatching(VertexOutput input) : SV_Target0
{
    return input.color + float4(input.uv, float2(input.ids));
}

struct FewerInput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

float4 pixelFewer(FewerInput input) : SV_Target0
{
    return float4(input.color, 1.0);
}

struct MissingInput
{
    float4 position : SV_Position;
    float4 color : COLOR0;
    nointerpolation uint2 ids : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float extra : TEXCOORD3;
};

float4 pixelMissing(MissingInput input) : SV_Target0
{
    return input.color * input.extra;
}

struct TypeMismatchInput
{
    float4 position : SV_Position;
    float4 color : COLOR0;
    float2 ids : TEXCOORD1;
};

float4 pixelTypeMismatch(TypeMismatchInput input) : SV_Target0
{
    return input.color + float4(input.ids, 0.0, 0.0);
}

struct MoreComponentsInput
{
    float4 position : SV_Position;
    float4 color : COLOR0;
    nointerpolation uint2 ids : TEXCOORD1;
    float4 uv : TEXCOORD2;
};

float4 pixelMoreComponents(MoreComponentsInput input) : SV_Target0
{
    return input.color + input.uv;
}

struct NoPositionInput
{
    float4 color : COLOR0;
};

float4 pixelNoPosition(NoPositionInput input) : SV_Target0
{
    return input.color;
}
