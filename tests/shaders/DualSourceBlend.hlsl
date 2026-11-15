struct PixelOutput
{
#ifdef __spirv__
    [[vk::location(0), vk::index(0)]] float4 source0 : SV_Target0;
    [[vk::location(0), vk::index(1)]] float4 source1 : SV_Target1;
#else
    float4 source0 : SV_Target0;
    float4 source1 : SV_Target1;
#endif
};

PixelOutput main()
{
    PixelOutput output;
    output.source0 = float4(1.0F, 0.0F, 0.5F, 1.0F);
    output.source1 = float4(0.25F, 0.5F, 0.75F, 0.5F);
    return output;
}
