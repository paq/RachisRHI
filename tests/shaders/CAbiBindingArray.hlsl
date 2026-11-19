StructuredBuffer<uint> colors[] : register(t0, space0);

struct CellVertex
{
    float4 position : SV_Position;
    nointerpolation uint drawIndex : TEXCOORD0;
};

float4 arrayPixel(CellVertex input) : SV_Target0
{
    const uint element = input.drawIndex * 2U + 1U;
#if defined(RACHIS_METAL)
    const uint packed = colors[element][0];
#else
    const uint packed = colors[NonUniformResourceIndex(element)][0];
#endif
    return float4(float(packed & 0xFFU), float((packed >> 8U) & 0xFFU), float((packed >> 16U) & 0xFFU),
                  float(packed >> 24U)) /
           255.0F;
}
