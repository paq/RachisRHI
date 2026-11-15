Texture2DMSArray<float> sourceDepth : register(t0);

cbuffer ResolveConstants : register(b0)
{
    uint sampleCount;
    uint useMaximum;
};

struct VertexOutput
{
    float4 position : SV_Position;
};

#ifdef RACHIS_METAL
[shader("vertex")]
#endif
VertexOutput vertexMain(uint vertexId : SV_VertexID)
{
    const float2 corner = float2((vertexId << 1U) & 2U, vertexId & 2U);
    VertexOutput output;
    output.position = float4(corner * float2(2.0F, -2.0F) + float2(-1.0F, 1.0F), 0.0F, 1.0F);
    return output;
}

#ifdef RACHIS_METAL
float metalMsaaDepthSample(Texture2DMSArray<float> tex, uint2 coord, uint layer, uint sample)
{
    __intrinsic_asm "$0.read($1, $2, $3).x";
}
#endif

#ifdef RACHIS_METAL
[shader("fragment")]
#endif
float pixelMain(VertexOutput input) : SV_Depth
{
#ifdef RACHIS_METAL
    const uint2 pixel = uint2(uint(input.position.x), uint(input.position.y));
    float resolved = metalMsaaDepthSample(sourceDepth, pixel, 0, 0);
    for (uint sample = 1; sample < sampleCount; ++sample)
    {
        const float value = metalMsaaDepthSample(sourceDepth, pixel, 0, sample);
        resolved = useMaximum != 0 ? max(resolved, value) : min(resolved, value);
    }
    return resolved;
#else
    const int3 location = int3(int2(input.position.xy), 0);
    float resolved = sourceDepth.Load(location, 0);
    for (uint sample = 1; sample < sampleCount; ++sample)
    {
        const float value = sourceDepth.Load(location, sample);
        resolved = useMaximum != 0 ? max(resolved, value) : min(resolved, value);
    }
    return resolved;
#endif
}
