RWStructuredBuffer<uint> outputBuffer : register(u0, space0);

#ifdef RACHIS_METAL
[shader("compute")]
#endif
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    outputBuffer[dispatchThreadId.x] = dispatchThreadId.x * 3u + 7u;
}
