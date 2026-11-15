RWStructuredBuffer<uint> outputBuffer : register(u0, space0);

struct PushConstants
{
    uint floatInputBits;
    int integerInput;
};

#ifdef __spirv__
[[vk::push_constant]] PushConstants pushConstants;
#else
ConstantBuffer<PushConstants> pushConstants : register(b0, space4);
#endif

[numthreads(1, 1, 1)]
void main()
{
    float16_t floatInput = (float16_t)asfloat(pushConstants.floatInputBits);
    float16_t floatResult = floatInput * (float16_t)1.5 + (float16_t)2.0;

    int16_t signedInput = (int16_t)pushConstants.integerInput;
    int16_t signedResult = signedInput * (int16_t)3 - (int16_t)5;

    outputBuffer[0] = (uint)floatResult;
    outputBuffer[1] = (uint)(int)signedResult;
}
