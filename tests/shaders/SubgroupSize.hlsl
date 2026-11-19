RWStructuredBuffer<uint2> laneCounts : register(u0, space0);

#if defined(RACHIS_D3D12) && defined(RACHIS_SUBGROUP_SIZE)
[WaveSize(RACHIS_SUBGROUP_SIZE)]
#endif
[numthreads(128, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    laneCounts[threadId.x] = uint2(WaveGetLaneCount(), WaveActiveCountBits(true));
}
