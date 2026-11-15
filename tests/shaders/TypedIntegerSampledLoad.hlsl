Texture2D<uint> inputTexture : register(t0, space0);
RWStructuredBuffer<uint> output : register(u1, space0);

[numthreads(4, 4, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    output[id.y * 4 + id.x] = inputTexture.Load(int3(id.xy, 0));
}
