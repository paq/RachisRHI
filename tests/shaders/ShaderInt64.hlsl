RWStructuredBuffer<uint> outputBuffer : register(u0, space0);

struct PushConstants
{
    uint lowInput;
    uint highInput;
    int signedOffset;
    uint unused;
};

#ifdef __spirv__
[[vk::push_constant]] PushConstants pushConstants;
#else
ConstantBuffer<PushConstants> pushConstants : register(b0, space4);
#endif

[numthreads(1, 1, 1)]
void main()
{
    uint64_t input = (uint64_t(pushConstants.highInput) << 32) | uint64_t(pushConstants.lowInput);
    uint64_t unsignedResult = input * uint64_t(3) + uint64_t(7);

    int64_t signedResult = int64_t(input) - (int64_t(1) << 39) + int64_t(pushConstants.signedOffset);
    uint64_t signedBits = uint64_t(signedResult);

    outputBuffer[0] = uint(unsignedResult);
    outputBuffer[1] = uint(unsignedResult >> 32);
    outputBuffer[2] = uint(signedBits);
    outputBuffer[3] = uint(signedBits >> 32);
}
