RWTexture2D<uint64_t> outputTexture : register(u0, space0);

[numthreads(64, 1, 1)]
void main(uint threadIndex : SV_GroupIndex)
{
    if (threadIndex == 0)
    {
        outputTexture[uint2(0, 0)] = uint64_t(0);
    }
    DeviceMemoryBarrierWithGroupSync();

    uint64_t original;
    InterlockedAdd(outputTexture[uint2(0, 0)], uint64_t(threadIndex + 1), original);
}
