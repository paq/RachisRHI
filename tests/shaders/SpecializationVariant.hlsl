RWStructuredBuffer<uint> output : register(u0);

#ifndef RACHIS_SPECIALIZED_VALUE
#define RACHIS_SPECIALIZED_VALUE 0
#endif

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    output[dispatchThreadId.x] = RACHIS_SPECIALIZED_VALUE;
}
