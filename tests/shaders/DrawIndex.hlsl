#include "Rachis/DrawIndex.hlsli"

struct DrawIndexVertex
{
    float4 position : SV_Position;
    nointerpolation uint drawIndex : TEXCOORD0;
};

static const float4 recordColors[5] = {
    float4(1.0F, 0.0F, 0.0F, 1.0F),
    float4(0.0F, 1.0F, 0.0F, 1.0F),
    float4(0.0F, 0.0F, 1.0F, 1.0F),
    float4(1.0F, 1.0F, 0.0F, 1.0F),
    float4(1.0F, 0.0F, 1.0F, 1.0F),
};

DrawIndexVertex drawIndexVertex(float2 position : TEXCOORD0)
{
    DrawIndexVertex output;
    output.position = float4(position, 0.0F, 1.0F);
    output.drawIndex = rachisDrawIndex();
    return output;
}

DrawIndexVertex plainVertex(float2 position : TEXCOORD0)
{
    DrawIndexVertex output;
    output.position = float4(position, 0.0F, 1.0F);
    output.drawIndex = 4;
    return output;
}

[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void drawIndexMesh(uint3 groupId : SV_GroupID, out vertices DrawIndexVertex vertices[4], out indices uint3 triangles[2])
{
    const uint drawIndex = rachisDrawIndex();
    const uint column = drawIndex + 4 * groupId.x;
    const float left = -1.0F + 0.4F * float(column) + 0.05F;
    const float right = left + 0.3F;
    SetMeshOutputCounts(4, 2);
    vertices[0].position = float4(left, -0.8F, 0.0F, 1.0F);
    vertices[1].position = float4(left, 0.8F, 0.0F, 1.0F);
    vertices[2].position = float4(right, -0.8F, 0.0F, 1.0F);
    vertices[3].position = float4(right, 0.8F, 0.0F, 1.0F);
    for (uint index = 0; index < 4; ++index)
    {
        vertices[index].drawIndex = drawIndex;
    }
    triangles[0] = uint3(0, 1, 2);
    triangles[1] = uint3(2, 1, 3);
}

float4 recordPixel(DrawIndexVertex input) : SV_Target0
{
    return recordColors[min(input.drawIndex, 4U)];
}
