RWStructuredBuffer<float> results : register(u0, space0);
Texture2D<float4> firstTextures[] : register(t0, space1);
Texture2D<float4> secondTextures[] : register(t0, space2);
Texture2D<float4> thirdTextures[] : register(t0, space3);

[shader("compute")]
[numthreads(1, 1, 1)]
void textureArraysMain()
{
    results[0] = firstTextures[NonUniformResourceIndex(0)].Load(int3(0, 0, 0)).x +
                 secondTextures[NonUniformResourceIndex(0)].Load(int3(0, 0, 0)).x +
                 thirdTextures[NonUniformResourceIndex(0)].Load(int3(0, 0, 0)).x;
}
