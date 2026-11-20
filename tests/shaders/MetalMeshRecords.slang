#include "Rachis/DrawIndex.hlsli"

struct RecordPayload
{
    uint drawIndex;
};

struct RecordVertex
{
    float4 position : SV_Position;
    float4 color : COLOR0;
};

static const float2 recordTriangle[3] = {
    float2(0.0F, 0.75F),
    float2(0.75F, -0.75F),
    float2(-0.75F, -0.75F),
};

[shader("amplification")]
[numthreads(1, 1, 1)]
void recordTask()
{
    RecordPayload payload;
    payload.drawIndex = rachisDrawIndex();
    DispatchMesh(1, 1, 1, payload);
}

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void recordMesh(out vertices RecordVertex vertices[3], out indices uint3 triangles[1])
{
    const float offset = -0.5F + 0.5F * float(min(rachisDrawIndex(), 2U));
    SetMeshOutputCounts(3, 1);
    for (uint index = 0; index < 3; ++index)
    {
        RecordVertex vertex;
        vertex.position = float4(recordTriangle[index] * 0.3F + float2(offset, 0.0F), 0.0F, 1.0F);
        vertex.color = float4(0.0F, 0.25F, 0.0F, 1.0F);
        vertices[index] = vertex;
    }
    triangles[0] = uint3(0, 1, 2);
}

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void recordMeshFromTask(in payload RecordPayload payload,
                        out vertices RecordVertex vertices[3],
                        out indices uint3 triangles[1])
{
    const float offset = -0.5F + 0.5F * float(min(payload.drawIndex, 2U));
    const bool sameIndex = payload.drawIndex == rachisDrawIndex();
    SetMeshOutputCounts(3, 1);
    for (uint index = 0; index < 3; ++index)
    {
        RecordVertex vertex;
        vertex.position = float4(recordTriangle[index] * 0.3F + float2(offset, 0.0F), 0.0F, 1.0F);
        vertex.color = sameIndex ? float4(0.0F, 0.25F, 0.0F, 1.0F) : float4(1.0F, 0.0F, 0.0F, 1.0F);
        vertices[index] = vertex;
    }
    triangles[0] = uint3(0, 1, 2);
}

[shader("pixel")]
float4 recordColor(RecordVertex input) : SV_Target
{
    return input.color;
}
