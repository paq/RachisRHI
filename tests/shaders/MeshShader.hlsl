cbuffer TaskResources : register(b0, space0)
{
    float4 taskColor;
};

cbuffer PushConstants : register(b0, space4)
{
    float4 meshControl;
};

struct MeshPayload
{
    float4 color;
};

struct MeshVertex
{
    float4 position : SV_Position;
    float4 color : COLOR0;
};

[numthreads(1, 1, 1)]
void taskMain()
{
    MeshPayload payload;
    payload.color = taskColor;
    DispatchMesh(1, 1, 1, payload);
}

[numthreads(1, 1, 1)]
void taskStatic()
{
    MeshPayload payload;
    payload.color = float4(0.0F, 1.0F, 0.0F, 1.0F);
    DispatchMesh(1, 1, 1, payload);
}

void writeTriangle(float4 color, out vertices MeshVertex vertices[3], out indices uint3 triangles[1])
{
    const float2 positions[3] = {
        float2(0.0F, 0.75F),
        float2(0.75F, -0.75F),
        float2(-0.75F, -0.75F),
    };
    SetMeshOutputCounts(3, 1);
    for (uint index = 0; index < 3; ++index)
    {
        vertices[index].position = float4(positions[index] * meshControl.x + meshControl.yz, 0.0F, 1.0F);
        vertices[index].color = color;
    }
    triangles[0] = uint3(0, 1, 2);
}

[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void meshMain(in payload MeshPayload payload,
              out vertices MeshVertex vertices[3],
              out indices uint3 triangles[1])
{
    writeTriangle(payload.color, vertices, triangles);
}

[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void meshStatic(in payload MeshPayload payload,
                out vertices MeshVertex vertices[3],
                out indices uint3 triangles[1])
{
    const float2 positions[3] = {
        float2(0.0F, 0.75F),
        float2(0.75F, -0.75F),
        float2(-0.75F, -0.75F),
    };
    SetMeshOutputCounts(3, 1);
    for (uint index = 0; index < 3; ++index)
    {
        vertices[index].position = float4(positions[index], 0.0F, 1.0F);
        vertices[index].color = payload.color;
    }
    triangles[0] = uint3(0, 1, 2);
}

[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void meshNoTask(out vertices MeshVertex vertices[3], out indices uint3 triangles[1])
{
    const float2 positions[3] = {
        float2(0.0F, 0.75F),
        float2(0.75F, -0.75F),
        float2(-0.75F, -0.75F),
    };
    SetMeshOutputCounts(3, 1);
    for (uint index = 0; index < 3; ++index)
    {
        vertices[index].position = float4(positions[index], 0.0F, 1.0F);
        vertices[index].color = float4(0.0F, 1.0F, 0.0F, 1.0F);
    }
    triangles[0] = uint3(0, 1, 2);
}

float4 meshPixel(MeshVertex input) : SV_Target0
{
    return input.color;
}
