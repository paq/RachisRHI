#ifdef RACHIS_METAL
RWStructuredBuffer<uint> outputBuffer : register(u1, space0);
#else
RWStructuredBuffer<uint> outputBuffer : register(u0, space0);
#endif

[[vk::push_constant]]
cbuffer PushConstants : register(b0, space4)
{
    uint multiplier;
    uint addend;
    uint pad0;
    uint pad1;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    outputBuffer[dispatchThreadId.x] = dispatchThreadId.x * multiplier + addend;
}
