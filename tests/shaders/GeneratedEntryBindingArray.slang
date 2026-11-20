RWStructuredBuffer<uint> outputs : register(u0, space1);

[shader("compute")]
[numthreads(1, 1, 1)]
void entryArrayMain(uint3 threadId : SV_DispatchThreadID, uniform StructuredBuffer<uint> values[])
{
#if defined(RACHIS_METAL)
    outputs[threadId.x] = values[threadId.x][0];
#else
    outputs[threadId.x] = values[NonUniformResourceIndex(threadId.x)][0];
#endif
}
