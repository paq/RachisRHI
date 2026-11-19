#include "Rachis/DrawIndex.hlsli"

struct CellVertex
{
    float4 position : SV_Position;
    nointerpolation uint drawIndex : TEXCOORD0;
};

struct CellPayload
{
    uint column;
    uint row;
    uint colorIndex;
};

static const float4 recordColors[5] = {
    float4(1.0F, 0.0F, 0.0F, 1.0F),
    float4(0.0F, 1.0F, 0.0F, 1.0F),
    float4(0.0F, 0.0F, 1.0F, 1.0F),
    float4(1.0F, 1.0F, 0.0F, 1.0F),
    float4(1.0F, 0.0F, 1.0F, 1.0F),
};

CellVertex cellVertex(uint column, uint row, uint corner, uint drawIndex)
{
    const float left = -1.0F + 0.5F * float(column);
    const float top = 1.0F - 0.5F * float(row);
    CellVertex vertex;
    vertex.position = float4((corner & 1U) != 0 ? left + 0.5F : left, (corner & 2U) != 0 ? top - 0.5F : top, 0.0F, 1.0F);
    vertex.drawIndex = drawIndex;
    return vertex;
}

CellVertex indirectVertex(float2 cell : TEXCOORD0)
{
    const uint drawIndex = rachisDrawIndex();
    CellVertex output;
    output.position = float4(-1.0F + 0.5F * (float(drawIndex) + cell.x), cell.y, 0.0F, 1.0F);
    output.drawIndex = drawIndex;
    return output;
}

[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void meshMain(uint3 groupId : SV_GroupID, out vertices CellVertex vertices[4], out indices uint3 triangles[2])
{
    const uint drawIndex = rachisDrawIndex();
    SetMeshOutputCounts(4, 2);
    for (uint corner = 0; corner < 4; ++corner)
    {
        vertices[corner] = cellVertex(drawIndex, groupId.x, corner, drawIndex);
    }
    triangles[0] = uint3(0, 1, 2);
    triangles[1] = uint3(2, 1, 3);
}

groupshared CellPayload taskPayload;

[numthreads(1, 1, 1)]
void taskMain()
{
    taskPayload.column = 3;
    taskPayload.row = 3;
    taskPayload.colorIndex = 4;
    DispatchMesh(1, 1, 1, taskPayload);
}

[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void meshFromTask(in payload CellPayload payload, out vertices CellVertex vertices[4], out indices uint3 triangles[2])
{
    SetMeshOutputCounts(4, 2);
    for (uint corner = 0; corner < 4; ++corner)
    {
        vertices[corner] = cellVertex(payload.column, payload.row, corner, payload.colorIndex);
    }
    triangles[0] = uint3(0, 1, 2);
    triangles[1] = uint3(2, 1, 3);
}

float4 recordPixel(CellVertex input) : SV_Target0
{
    return recordColors[min(input.drawIndex, 4U)];
}
