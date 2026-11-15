RWTexture2D<unorm float4> outputImage : register(u0, space0);

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    outputImage[dispatchThreadId.xy] = float4(0.25F, 0.5F, 0.75F, 1.0F);
}
