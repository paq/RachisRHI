globallycoherent RasterizerOrderedByteAddressBuffer orderedValues : register(u0, space0);

float4 main() : SV_Target0
{
    const uint prior = orderedValues.Load(0);
    orderedValues.Store(0, prior + 1);
    return float4(0.0F, 0.0F, 0.0F, 1.0F);
}
