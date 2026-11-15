RWStructuredBuffer<uint> values : register(u0, space0);

[numthreads(64, 1, 1)]
void fillMain(uint3 threadId : SV_DispatchThreadID)
{
    values[threadId.x] = threadId.x + FILL_OFFSET;
}
