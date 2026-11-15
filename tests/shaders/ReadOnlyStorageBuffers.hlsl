StructuredBuffer<uint> structuredInput : register(t0, space0);
RWStructuredBuffer<uint> outputValues : register(u1, space0);
ByteAddressBuffer byteAddressInput : register(t2, space0);
StructuredBuffer<uint> structuredInputs[4] : register(t3, space0);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    const uint index = threadId.x;
    const uint arrayed = structuredInputs[0][index] + structuredInputs[1][index] * 2u +
                         structuredInputs[2][index] * 3u + structuredInputs[3][index] * 4u;
    outputValues[index] = structuredInput[index] + byteAddressInput.Load(index * 4u) * 5u + arrayed * 7u;
}
