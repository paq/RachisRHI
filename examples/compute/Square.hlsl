RWStructuredBuffer<uint> values : register(u0, space0);

[shader("compute")]
[numthreads(64, 1, 1)]
void squareMain(uint3 threadId : SV_DispatchThreadID)
{
    const uint value = values[threadId.x];
    values[threadId.x] = value * value;
}
