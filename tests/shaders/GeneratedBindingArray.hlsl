StructuredBuffer<uint> values[] : register(t0, space1);
RWStructuredBuffer<uint> results : register(u0, space0);

[numthreads(64, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
#if defined(RACHIS_METAL)
    results[threadId.x] = values[threadId.x][0];
#else
    results[threadId.x] = values[NonUniformResourceIndex(threadId.x)][0];
#endif
}
