RWStructuredBuffer<uint> outputBuffer : register(u0, space0);

groupshared uint64_t accumulator;

[numthreads(64, 1, 1)]
void main(uint threadIndex : SV_GroupIndex)
{
    if (threadIndex == 0)
    {
        accumulator = uint64_t(0);
    }
    GroupMemoryBarrierWithGroupSync();

    uint64_t original;
    InterlockedAdd(accumulator, uint64_t(threadIndex + 1), original);
    GroupMemoryBarrierWithGroupSync();

    if (threadIndex == 0)
    {
        outputBuffer[0] = uint(accumulator);
        outputBuffer[1] = uint(accumulator >> 32);
    }
}
