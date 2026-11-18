RWStructuredBuffer<uint64_t> extremes : register(u0, space0);

[numthreads(64, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    const uint lane = threadId.x;
    const uint high = ((lane >> 1) + 1) * 2654435761u;
    const uint low = lane * 2246822519u;
    const uint64_t key = (uint64_t(high) << 32) | uint64_t(low);
    InterlockedMax(extremes[0], key);
    InterlockedMin(extremes[1], key);
}
