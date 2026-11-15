RWStructuredBuffer<uint> outputBuffer : register(u0, space0);

cbuffer PushConstants : register(b0, space4)
{
    uint multiplier;
    uint addend;
};

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    outputBuffer[dispatchThreadId.x] = dispatchThreadId.x * multiplier + addend;
}

[numthreads(64, 1, 1)]
void countDispatch(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    InterlockedAdd(outputBuffer[dispatchThreadId.x], addend);
}
