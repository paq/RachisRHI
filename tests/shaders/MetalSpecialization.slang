RWStructuredBuffer<uint> outputValues : register(u0, space0);

#ifndef RACHIS_SPECIALIZED_VALUE
#define RACHIS_SPECIALIZED_VALUE 0
#endif

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    outputValues[dispatchThreadId.x] = RACHIS_SPECIALIZED_VALUE;
}
