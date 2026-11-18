#include "Rachis/DrawIndex.hlsli"

struct DrawIndexVertex
{
    float4 position : SV_Position;
    nointerpolation uint drawIndex : TEXCOORD0;
};

DrawIndexVertex drawIndexMain(float2 position : TEXCOORD0)
{
    DrawIndexVertex output;
    output.position = float4(position, 0.0F, 1.0F);
    output.drawIndex = rachisDrawIndex();
    return output;
}
