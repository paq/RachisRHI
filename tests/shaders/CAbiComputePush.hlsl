#ifdef RACHIS_METAL
RWStructuredBuffer<uint> outputBuffer : register(u1, space0);
#else
RWStructuredBuffer<uint> outputBuffer : register(u0, space0);
#endif

struct PushConstants
{
    uint multiplier;
    uint addend;
    uint pad0;
    uint pad1;
};

#if defined(RACHIS_METAL) || defined(__spirv__)
[[vk::push_constant]]
#endif
ConstantBuffer<PushConstants> pushConstants : register(b0, space4);

#ifdef RACHIS_METAL
[shader("compute")]
#endif
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    outputBuffer[dispatchThreadId.x] = dispatchThreadId.x * pushConstants.multiplier + pushConstants.addend;
}
