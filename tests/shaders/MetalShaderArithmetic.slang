#ifdef RACHIS_METAL
RWStructuredBuffer<uint> outputValues : register(u1, space0);
#else
RWStructuredBuffer<uint> outputValues : register(u0, space0);
#endif

[[vk::push_constant]]
cbuffer ArithmeticPush : register(b0, space4)
{
    uint floatInputBits;
    int integerInput;
    uint pad0;
    uint pad1;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void arithmetic16Main()
{
    float16_t floatInput = (float16_t)asfloat(floatInputBits);
    float16_t floatResult = floatInput * (float16_t)1.5 + (float16_t)2.0;

    int16_t signedInput = (int16_t)integerInput;
    int16_t signedResult = signedInput * (int16_t)3 - (int16_t)5;

    outputValues[0] = (uint)floatResult;
    outputValues[1] = (uint)(int)signedResult;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void int64Main()
{
    int64_t value = int64_t(1) << 40;
    value += int64_t(7);
    outputValues[0] = (uint)value;
    outputValues[1] = (uint)(value >> 32);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void subgroupMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint laneCount = WaveGetLaneCount();
    const uint laneIndex = WaveGetLaneIndex();
    const uint activeCount = WaveActiveSum(1u);
    const uint activeIndexSum = WaveActiveSum(laneIndex);
    const uint base = dispatchThreadId.x * 4u;
    outputValues[base + 0u] = laneCount;
    outputValues[base + 1u] = laneIndex;
    outputValues[base + 2u] = activeCount;
    outputValues[base + 3u] = activeIndexSum;
}
