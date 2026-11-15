cbuffer GeometryResources : register(b0, space0)
{
    float4 geometryColor;
};

#ifdef __spirv__
struct GeometryPushConstants
{
    float4 geometryControl;
};
[[vk::push_constant]] GeometryPushConstants geometryPush;
#define geometryControl geometryPush.geometryControl
#else
cbuffer PushConstants : register(b0, space4)
{
    float4 geometryControl;
};
#endif

struct VertexOutput
{
    float4 position : SV_Position;
};

struct GeometryOutput
{
    float4 position : SV_Position;
    float4 color : COLOR0;
};

struct ViewportGeometryOutput
{
    float4 position : SV_Position;
    float4 color : COLOR0;
    uint viewportIndex : SV_ViewportArrayIndex;
};

VertexOutput pointVertex(uint vertexId : SV_VertexID)
{
    VertexOutput output;
    output.position = float4(float(vertexId), 0.0F, 0.0F, 1.0F);
    return output;
}

void emitTriangle(float scale, float2 offset, float4 color, inout TriangleStream<GeometryOutput> stream)
{
    const float2 positions[3] = {
        float2(0.0F, 0.75F),
        float2(0.75F, -0.75F),
        float2(-0.75F, -0.75F),
    };
    for (uint index = 0; index < 3; ++index)
    {
        GeometryOutput output;
        output.position = float4(positions[index] * scale + offset, 0.0F, 1.0F);
        output.color = color;
        stream.Append(output);
    }
}

[maxvertexcount(3)]
void geometryMain(point VertexOutput input[1], inout TriangleStream<GeometryOutput> stream)
{
    emitTriangle(1.0F, float2(0.0F, 0.0F), float4(1.0F, 1.0F, 1.0F, 1.0F), stream);
}

[maxvertexcount(3)]
void geometryResourceMain(point VertexOutput input[1], inout TriangleStream<GeometryOutput> stream)
{
    emitTriangle(geometryControl.x, geometryControl.yz, geometryColor, stream);
}

void emitViewportTriangle(uint viewportIndex, float4 color, inout TriangleStream<ViewportGeometryOutput> stream)
{
    const float2 positions[3] = {
        float2(0.0F, 0.75F),
        float2(0.75F, -0.75F),
        float2(-0.75F, -0.75F),
    };
    for (uint index = 0; index < 3; ++index)
    {
        ViewportGeometryOutput output;
        output.position = float4(positions[index], 0.0F, 1.0F);
        output.color = color;
        output.viewportIndex = viewportIndex;
        stream.Append(output);
    }
}

[maxvertexcount(6)]
void viewportArrayMain(point VertexOutput input[1], inout TriangleStream<ViewportGeometryOutput> stream)
{
    emitViewportTriangle(0, float4(1.0F, 0.0F, 0.0F, 1.0F), stream);
    stream.RestartStrip();
    emitViewportTriangle(1, float4(0.0F, 1.0F, 0.0F, 1.0F), stream);
}

float4 geometryPixelMain(GeometryOutput input) : SV_Target0
{
    return input.color;
}
