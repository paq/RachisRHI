StructuredBuffer<uint> structuredInput : register(t0, space0);
RWStructuredBuffer<uint> outputValues : register(u1, space0);
ByteAddressBuffer byteAddressInput : register(t2, space0);

[shader("compute")]
[numthreads(64, 1, 1)]
void readOnlyMain(uint3 threadId : SV_DispatchThreadID)
{
    const uint index = threadId.x;
    outputValues[index] = structuredInput[index] + byteAddressInput.Load(index * 4u) * 5u;
}
