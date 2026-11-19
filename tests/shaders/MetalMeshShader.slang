struct TaskResources
{
    float4 taskColor;
};

struct MeshControl
{
    float4 meshControl;
};

ConstantBuffer<TaskResources> taskResources : register(b0, space0);

[[vk::push_constant]] ConstantBuffer<MeshControl> pushConstants : register(b0, space4);

struct MeshPayload
{
    float4 color;
};

struct MeshVertex
{
    float4 position : SV_Position;
    float4 color : COLOR0;
};

static const float2 trianglePositions[3] = {
    float2(0.0F, 0.75F),
    float2(0.75F, -0.75F),
    float2(-0.75F, -0.75F),
};

[shader("amplification")]
[numthreads(1, 1, 1)]
void taskMain()
{
    MeshPayload payload;
    payload.color = taskResources.taskColor;
    DispatchMesh(1, 1, 1, payload);
}

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void meshMain(in payload MeshPayload payload, out vertices MeshVertex vertices[3], out indices uint3 triangles[1])
{
    SetMeshOutputCounts(3, 1);
    for (uint index = 0; index < 3; ++index)
    {
        MeshVertex vertex;
        vertex.position =
            float4(trianglePositions[index] * pushConstants.meshControl.x + pushConstants.meshControl.yz, 0.0F, 1.0F);
        vertex.color = payload.color;
        vertices[index] = vertex;
    }
    triangles[0] = uint3(0, 1, 2);
}

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void meshNoTask(out vertices MeshVertex vertices[3], out indices uint3 triangles[1])
{
    SetMeshOutputCounts(3, 1);
    for (uint index = 0; index < 3; ++index)
    {
        MeshVertex vertex;
        vertex.position =
            float4(trianglePositions[index] * pushConstants.meshControl.x + pushConstants.meshControl.yz, 0.0F, 1.0F);
        vertex.color = float4(0.0F, pushConstants.meshControl.w, 0.0F, 1.0F);
        vertices[index] = vertex;
    }
    triangles[0] = uint3(0, 1, 2);
}

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void meshFixed(out vertices MeshVertex vertices[3], out indices uint3 triangles[1])
{
    SetMeshOutputCounts(3, 1);
    for (uint index = 0; index < 3; ++index)
    {
        MeshVertex vertex;
        vertex.position = float4(trianglePositions[index] * 0.3F + float2(-0.5F, 0.0F), 0.0F, 1.0F);
        vertex.color = float4(0.0F, 0.25F, 0.0F, 1.0F);
        vertices[index] = vertex;
    }
    triangles[0] = uint3(0, 1, 2);
}

struct MissingInput
{
    float4 position : SV_Position;
    float4 color : COLOR0;
    float4 extra : TEXCOORD5;
};

[shader("pixel")]
float4 meshPixel(MeshVertex input) : SV_Target
{
    return input.color;
}

[shader("pixel")]
float4 meshPixelMissing(MissingInput input) : SV_Target
{
    return input.color + input.extra;
}
