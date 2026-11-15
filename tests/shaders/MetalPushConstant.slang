#ifndef RACHIS_OUTPUT_SPACE
#define RACHIS_OUTPUT_SPACE space0
#endif

#ifdef RACHIS_METAL
RWStructuredBuffer<uint> outputValues : register(u1, RACHIS_OUTPUT_SPACE);
#else
RWStructuredBuffer<uint> outputValues : register(u0, space0);
#endif

[[vk::push_constant]]
cbuffer PushConstants : register(b0, space4)
{
    uint marker;
    uint pad0;
    uint pad1;
    uint pad2;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    outputValues[dispatchThreadId.x] = marker;
}
