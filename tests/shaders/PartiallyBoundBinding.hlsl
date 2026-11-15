RWStructuredBuffer<uint> outputs[4] : register(u0, space0);

[numthreads(2, 1, 1)]
void main(uint3 threadId : SV_GroupThreadID)
{
    const uint descriptorIndex = threadId.x * 2 + 1;
    outputs[NonUniformResourceIndex(descriptorIndex)][0] = 100 + descriptorIndex;
}
