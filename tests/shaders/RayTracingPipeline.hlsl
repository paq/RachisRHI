RWStructuredBuffer<uint> outputBuffer : register(u0, space0);

#ifndef __spirv__
cbuffer ShaderRecordConstants : register(b1, space4)
{
    uint recordValue;
};

[shader("raygeneration")]
void rayGeneration()
{
    outputBuffer[0] = recordValue;
}

[shader("raygeneration")]
void rayGenerationCounter()
{
    InterlockedAdd(outputBuffer[0], 1);
}
#endif

#ifdef __spirv__
[[vk::constant_id(7)]]
const uint rayGenerationValue = 0x52545852;
#else
static const uint rayGenerationValue = 0x52545852;
#endif

[shader("raygeneration")]
void rayGenerationWithoutRecord()
{
    outputBuffer[0] = rayGenerationValue;
}

struct RayPayload
{
    uint value;
};

[shader("miss")]
void missShader(inout RayPayload payload)
{
    payload.value = 0;
}

#ifndef __spirv__
[shader("closesthit")]
void closestHitShader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attributes)
{
    payload.value = 1;
}

RaytracingAccelerationStructure scene : register(t1, space0);

[shader("raygeneration")]
void rayGenerationTracePerHitGroup()
{
    RayDesc ray;
    ray.Origin = float3(0.0, 0.0, -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 10.0;
    RayPayload payload;
    payload.value = 0;
    const uint rayIndex = DispatchRaysIndex().x;
    TraceRay(scene, RAY_FLAG_NONE, 0xFF, rayIndex, 0, 0, ray, payload);
    outputBuffer[rayIndex] = payload.value;
}

[shader("closesthit")]
void closestHitRecordValue(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attributes)
{
    payload.value = recordValue;
}

cbuffer AlphaTestRecord : register(b2, space4)
{
    uint alphaTestValue;
    uint alphaTestThreshold;
};

[shader("anyhit")]
void anyHitRecordValue(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attributes)
{
    if (recordValue == 0)
    {
        IgnoreHit();
    }
}

[shader("anyhit")]
void anyHitAlphaTest(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attributes)
{
    if (alphaTestValue < alphaTestThreshold)
    {
        IgnoreHit();
    }
}
#endif
