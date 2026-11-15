struct VertexOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
    nointerpolation uint2 ids : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexId : SV_VertexID)
{
    const float2 positions[3] = {float2(0.0, 0.75), float2(0.75, -0.75), float2(-0.75, -0.75)};
    VertexOutput output;
    output.position = float4(positions[vertexId % 3], 0.0, 1.0);
    output.color = float3(1.0, 0.5, 0.25);
    output.ids = uint2(vertexId, 7);
    return output;
}

[shader("pixel")]
float4 matchingPixel(VertexOutput input) : SV_Target0
{
    return float4(input.color, float(input.ids.y));
}

struct FewerComponentsInput
{
    float4 position : SV_Position;
    float2 color : COLOR0;
};

[shader("pixel")]
float4 fewerComponentsPixel(FewerComponentsInput input) : SV_Target0
{
    return float4(input.color, 0.0, 1.0);
}

struct MoreComponentsInput
{
    float4 position : SV_Position;
    float4 color : COLOR0;
};

[shader("pixel")]
float4 moreComponentsPixel(MoreComponentsInput input) : SV_Target0
{
    return input.color;
}

struct TypeMismatchInput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
    nointerpolation int2 ids : TEXCOORD0;
};

[shader("pixel")]
float4 typeMismatchPixel(TypeMismatchInput input) : SV_Target0
{
    return float4(input.color, float(input.ids.y));
}

struct MissingLocationInput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
    nointerpolation uint2 ids : TEXCOORD0;
    float extra : TEXCOORD1;
};

[shader("pixel")]
float4 missingLocationPixel(MissingLocationInput input) : SV_Target0
{
    return float4(input.color, input.extra + float(input.ids.y));
}

struct RenamedInput
{
    float4 position : SV_Position;
    float3 tint : TINT0;
};

[shader("pixel")]
float4 renamedPixel(RenamedInput input) : SV_Target0
{
    return float4(input.tint, 1.0);
}
