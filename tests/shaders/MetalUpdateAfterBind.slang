RWStructuredBuffer<uint> results : register(u0, space0);
StructuredBuffer<uint> values[] : register(t1, space0);

struct ReadIndex
{
    uint element;
    uint slot;
};

[[vk::push_constant]] ConstantBuffer<ReadIndex> readIndex : register(b0, space4);

[shader("compute")]
[numthreads(1, 1, 1)]
void updateAfterBindMain()
{
    results[readIndex.slot] = values[NonUniformResourceIndex(readIndex.element)][0];
}
