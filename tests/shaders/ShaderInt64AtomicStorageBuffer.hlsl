RWStructuredBuffer<uint64_t> outputBuffer : register(u0, space0);

[numthreads(1, 1, 1)]
void main()
{
    uint64_t original;
    InterlockedAdd(outputBuffer[0], uint64_t(1), original);
}
