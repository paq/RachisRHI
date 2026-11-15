RWStructuredBuffer<uint4> outputBuffer : register(u0, space0);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint laneCount = WaveGetLaneCount();
    const uint laneIndex = WaveGetLaneIndex();
    const uint activeCount = WaveActiveSum(1u);
    const uint activeIndexSum = WaveActiveSum(laneIndex);
    outputBuffer[dispatchThreadId.x] = uint4(laneCount, laneIndex, activeCount, activeIndexSum);
}
