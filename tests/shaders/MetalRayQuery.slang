RaytracingAccelerationStructure scene : register(t0, space0);
RWStructuredBuffer<uint> outputBuffer : register(u1, space0);

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain()
{
    RayDesc ray;
    ray.Origin = float3(0.0, 0.0, -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 10.0;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);
    while (query.Proceed())
    {
    }
    outputBuffer[0] = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 1 : 0;
}
