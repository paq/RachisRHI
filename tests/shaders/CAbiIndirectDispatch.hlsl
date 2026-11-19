RWStructuredBuffer<uint> groupCounter : register(u0, space0);

[numthreads(1, 1, 1)]
void countGroups()
{
    InterlockedAdd(groupCounter[0], 1U);
}
