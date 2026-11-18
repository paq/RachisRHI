RWStructuredBuffer<uint> outputBuffer : register(u0, space0);

struct PushConstants
{
    uint multiplier;
    uint addend;
};

[[vk::push_constant]] PushConstants pushConstants;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    outputBuffer[dispatchThreadId.x] = dispatchThreadId.x * pushConstants.multiplier + pushConstants.addend;
}

[numthreads(64, 1, 1)]
void scale(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    outputBuffer[dispatchThreadId.x] *= pushConstants.multiplier;
}

[numthreads(64, 1, 1)]
void countDispatch(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    InterlockedAdd(outputBuffer[dispatchThreadId.x], pushConstants.addend);
}
