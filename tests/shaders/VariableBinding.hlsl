RWStructuredBuffer<uint> outputs[] : register(u0, space0);

[numthreads(2, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    const uint descriptorIndex = threadId.x * 2 + 1;
    outputs[NonUniformResourceIndex(descriptorIndex)][0] = 200 + descriptorIndex;
}
