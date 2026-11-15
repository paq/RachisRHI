RWTexture2D<uint4> unsignedOutput : register(u0);
RWTexture2D<int4> signedOutput : register(u1);

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    unsignedOutput[id.xy] = uint4(17, 34, 51, 255);
    signedOutput[id.xy] = int4(-1, -2, 3, 4);
}
