RaytracingAccelerationStructure scene : register(t0, space0);
RWStructuredBuffer<uint4> results : register(u1, space0);

[numthreads(3, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    RayDesc ray;
    ray.Origin = float3(-2.0 + 2.0 * float(id.x), 0.0, -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 10.0;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);
    while (query.Proceed())
    {
    }
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        results[id.x] = uint4(1, query.CommittedInstanceID(), asuint(query.CommittedRayT()), query.CommittedInstanceIndex());
    }
    else
    {
        results[id.x] = uint4(0, 0, 0, 0);
    }
}
