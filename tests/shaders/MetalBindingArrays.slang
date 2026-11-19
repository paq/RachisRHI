RWStructuredBuffer<uint> results : register(u0, space0);
Texture2D<float4> textures[] : register(t0, space1);
StructuredBuffer<uint> buffers[] : register(t0, space2);

struct ArrayCounts
{
    uint textureCount;
    uint bufferCount;
};

[[vk::push_constant]] ConstantBuffer<ArrayCounts> counts : register(b0, space4);

[shader("compute")]
[numthreads(64, 1, 1)]
void bindingArraysMain(uint3 threadId : SV_DispatchThreadID)
{
    const uint index = threadId.x;
    uint value = 0;
    if (index < counts.textureCount)
    {
        value += uint(textures[NonUniformResourceIndex(index)].Load(int3(0, 0, 0)).x);
    }
    if (index < counts.bufferCount)
    {
        value += 1000 * buffers[NonUniformResourceIndex(index)][0];
    }
    results[index] = value;
}
