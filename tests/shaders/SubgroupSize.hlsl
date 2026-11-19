RWStructuredBuffer<uint2> laneCounts : register(u0, space0);

#if defined(RACHIS_SUBGROUP_SIZE)
#if defined(RACHIS_D3D12)
[WaveSize(RACHIS_SUBGROUP_SIZE)]
#endif
[numthreads(RACHIS_SUBGROUP_SIZE, 1, 1)]
#else
[numthreads(128, 1, 1)]
#endif
void main(uint3 threadId : SV_DispatchThreadID)
{
    laneCounts[threadId.x] = uint2(WaveGetLaneCount(), WaveActiveSum(1U));
}
