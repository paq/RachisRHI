#ifdef RACHIS_METAL
RWStructuredBuffer<uint> values : register(u1, space0);
RWTexture2D<float4> image : register(u0, space0);
#else
RWStructuredBuffer<uint> values : register(u0, space0);
RWTexture2D<float4> image : register(u1, space0);
#endif

[shader("compute")]
[numthreads(8, 8, 1)]
void scatterMain(uint3 tid : SV_DispatchThreadID)
{
    const uint index = tid.y * 8 + tid.x;
    values[index] = index * 3u + 1u;
    image[int2(tid.xy)] = float4(float(tid.x) / 7.0, float(tid.y) / 7.0, 0.25, 1.0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void reduceMain(uint3 tid : SV_DispatchThreadID)
{
    uint sum = 0;
    [unroll]
    for (uint i = 0; i < 64; ++i)
    {
        sum += values[i];
    }
    values[64] = sum;
}
