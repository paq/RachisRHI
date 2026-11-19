#include "fill values/ReflectBinding.hlsli"

RWStructuredBuffer<uint> reflectedValues : register(REFLECT_REGISTER, space0);

[shader("compute")]
[numthreads(64, 1, 1)]
void reflectMain(uint3 threadId : SV_DispatchThreadID)
{
    reflectedValues[threadId.x] = threadId.x;
}
