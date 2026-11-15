Texture2D<float4> inputTexture : register(t0, space0);
RWStructuredBuffer<float4> output : register(u1, space0);

[numthreads(1, 1, 1)]
void main()
{
    output[0] = inputTexture.Load(int3(0, 0, 0));
}
