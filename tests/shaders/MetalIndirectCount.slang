RWStructuredBuffer<uint> counter : register(u0, space0);

[shader("compute")]
[numthreads(1, 1, 1)]
void countMain(uint3 threadId : SV_DispatchThreadID)
{
    InterlockedAdd(counter[0], 1);
}
