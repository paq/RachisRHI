Texture2D<float4> bc2Linear : register(t0, space0);
Texture2D<float4> bc2Srgb : register(t1, space0);
Texture2D<float4> bc4Unorm : register(t2, space0);
Texture2D<float4> bc4Snorm : register(t3, space0);
Texture2D<float4> bc5Snorm : register(t4, space0);
Texture2D<float4> bc6Unsigned : register(t5, space0);
Texture2D<float4> bc6Signed : register(t6, space0);
RWStructuredBuffer<float4> output : register(u7, space0);

[numthreads(1, 1, 1)]
void main()
{
    const int3 origin = int3(0, 0, 0);
    output[0] = bc2Linear.Load(origin);
    output[1] = bc2Srgb.Load(origin);
    output[2] = bc4Unorm.Load(origin);
    output[3] = bc4Snorm.Load(origin);
    output[4] = bc5Snorm.Load(origin);
    output[5] = bc6Unsigned.Load(origin);
    output[6] = bc6Signed.Load(origin);
}
