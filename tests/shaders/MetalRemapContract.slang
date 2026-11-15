struct CompactUniforms24
{
    uint w0;
    uint w1;
    uint w2;
    uint w3;
    uint t0;
    uint t1;
};

struct CompactUniforms16
{
    uint4 words;
};

#if defined(RACHIS_REMAP_SINGLE_COMPUTE)
[[vk::push_constant]]
cbuffer SingleComputePush : register(b0, space4)
{
    uint4 pushWords;
};

ConstantBuffer<CompactUniforms24> set0Ubo : register(b0, space0);
Texture2D<float4> set0Tex : register(t1, space0);
SamplerState set0Sampler : register(s2, space0);
RWStructuredBuffer<uint> singleOut : register(u3, space0);

[shader("compute")]
[numthreads(1, 1, 1)]
void pushAndBindingZeroCompute()
{
    const float4 sampled = set0Tex.SampleLevel(set0Sampler, float2(0.75F, 0.25F), 0.0F);
    singleOut[0] = pushWords.x ^ set0Ubo.w0;
    singleOut[1] = pushWords.y + set0Ubo.w1;
    singleOut[2] = pushWords.z + set0Ubo.t0;
    singleOut[3] = pushWords.w + set0Ubo.t1;
    singleOut[4] = uint(round(sampled.x * 255.0F));
    singleOut[5] = uint(round(sampled.y * 255.0F));
    singleOut[6] = uint(round(sampled.z * 255.0F));
    singleOut[7] = uint(round(sampled.w * 255.0F)) + set0Ubo.w2 + set0Ubo.w3;
}
#endif

#if defined(RACHIS_REMAP_MULTI_COMPUTE)
struct MultiPushData
{
    uint4 words;
};

[[vk::push_constant]]
ConstantBuffer<MultiPushData> multiPush;

[[vk::binding(0, 0)]] ConstantBuffer<CompactUniforms24> multiSet0Ubo;
[[vk::binding(1, 0)]] Texture2D<float4> multiSet0Tex;
[[vk::binding(2, 0)]] SamplerState multiSet0Sampler;
[[vk::binding(3, 0)]] StructuredBuffer<uint> multiSet0In;

[[vk::binding(0, 1)]] ConstantBuffer<CompactUniforms24> multiSet1Ubo;
[[vk::binding(1, 1)]] Texture2D<float4> multiSet1Tex;
[[vk::binding(2, 1)]] SamplerState multiSet1Sampler;
[[vk::binding(3, 1)]] RWStructuredBuffer<uint> multiSet1Out;

[shader("compute")]
[numthreads(1, 1, 1)]
void multiSetCompute()
{
    const float4 tex0 = multiSet0Tex.SampleLevel(multiSet0Sampler, float2(1.25F, 0.25F), 0.0F);
    const float4 tex1 = multiSet1Tex.SampleLevel(multiSet1Sampler, float2(1.25F, 0.75F), 0.0F);
    multiSet1Out[0] = multiPush.words.x + multiSet0Ubo.w0;
    multiSet1Out[1] = multiPush.words.y + multiSet1Ubo.w0;
    multiSet1Out[2] = multiSet0Ubo.t0 + multiSet0In[0];
    multiSet1Out[3] = multiSet1Ubo.t1 + multiPush.words.z;
    multiSet1Out[4] = uint(round(tex0.x * 255.0F)) | (uint(round(tex0.y * 255.0F)) << 8);
    multiSet1Out[5] = uint(round(tex1.z * 255.0F)) | (uint(round(tex1.w * 255.0F)) << 8);
    multiSet1Out[6] = multiSet0Ubo.t1 ^ multiSet1Ubo.t0;
    multiSet1Out[7] = multiPush.words.w + multiSet0Ubo.w3 + multiSet1Ubo.w3;
}
#endif

struct RemapVertexInput
{
    float2 position : TEXCOORD0;
    uint vertexTag : TEXCOORD1;
};

struct RemapVaryings
{
    float4 position : SV_Position;
    nointerpolation uint4 payload : TEXCOORD0;
};

#if defined(RACHIS_REMAP_SINGLE_GRAPHICS_VS)
[[vk::push_constant]]
cbuffer SingleGraphicsPush : register(b0, space4)
{
    uint4 singleGraphicsPush;
};

ConstantBuffer<CompactUniforms16> singleVsSet0Ubo : register(b0, space0);

[shader("vertex")]
RemapVaryings pushAndBindingZeroVertex(RemapVertexInput input)
{
    RemapVaryings output;
    output.position = float4(input.position, 0.0F, 1.0F);
    output.payload = uint4(
        input.vertexTag + singleGraphicsPush.x,
        singleVsSet0Ubo.words.x + singleVsSet0Ubo.words.w,
        singleVsSet0Ubo.words.y,
        singleGraphicsPush.y);
    return output;
}
#endif

#if defined(RACHIS_REMAP_SINGLE_GRAPHICS_PS)
[[vk::push_constant]]
cbuffer SingleGraphicsPush : register(b0, space4)
{
    uint4 singleGraphicsPush;
};

ConstantBuffer<CompactUniforms16> singlePsSet0Ubo : register(b0, space0);
Texture2D<float4> singlePsSet0Tex : register(t1, space0);
SamplerState singlePsSet0Sampler : register(s2, space0);

[shader("pixel")]
uint4 pushAndBindingZeroPixel(RemapVaryings input) : SV_Target0
{
    const float4 sampled = singlePsSet0Tex.SampleLevel(singlePsSet0Sampler, float2(0.25F, 0.75F), 0.0F);
    const uint packedTex = uint(round(sampled.x * 255.0F)) | (uint(round(sampled.y * 255.0F)) << 8);
    return uint4(
        input.payload.x + singlePsSet0Ubo.words.x,
        input.payload.y + singlePsSet0Ubo.words.z,
        input.payload.z + singleGraphicsPush.z,
        packedTex + singlePsSet0Ubo.words.w + singleGraphicsPush.w);
}
#endif

#if defined(RACHIS_REMAP_MULTI_GRAPHICS_VS)
[[vk::push_constant]]
cbuffer MultiGraphicsPush : register(b0, space4)
{
    uint4 multiGraphicsPush;
};

ConstantBuffer<CompactUniforms24> multiVsSet0Ubo : register(b0, space0);
ConstantBuffer<CompactUniforms24> multiVsSet1Ubo : register(b0, space1);

[shader("vertex")]
RemapVaryings multiSetVertex(RemapVertexInput input)
{
    RemapVaryings output;
    output.position = float4(input.position, 0.0F, 1.0F);
    output.payload = uint4(
        input.vertexTag + multiGraphicsPush.x,
        multiVsSet0Ubo.w0 + multiVsSet0Ubo.t0,
        multiVsSet1Ubo.w0 + multiVsSet1Ubo.t1,
        multiGraphicsPush.y);
    return output;
}
#endif

#if defined(RACHIS_REMAP_MULTI_GRAPHICS_PS)
[[vk::push_constant]]
cbuffer MultiGraphicsPush : register(b0, space4)
{
    uint4 multiGraphicsPush;
};

ConstantBuffer<CompactUniforms24> multiPsSet0Ubo : register(b0, space0);
Texture2D<float4> multiPsSet0Tex : register(t1, space0);
SamplerState multiPsSet0Sampler : register(s2, space0);

ConstantBuffer<CompactUniforms24> multiPsSet1Ubo : register(b0, space1);
Texture2D<float4> multiPsSet1Tex : register(t1, space1);
SamplerState multiPsSet1Sampler : register(s2, space1);

[shader("pixel")]
uint4 multiSetPixel(RemapVaryings input) : SV_Target0
{
    const float4 tex0 = multiPsSet0Tex.SampleLevel(multiPsSet0Sampler, float2(1.25F, 0.25F), 0.0F);
    const float4 tex1 = multiPsSet1Tex.SampleLevel(multiPsSet1Sampler, float2(1.25F, 0.75F), 0.0F);
    const uint packed0 = uint(round(tex0.x * 255.0F)) | (uint(round(tex0.y * 255.0F)) << 8);
    const uint packed1 = uint(round(tex1.z * 255.0F)) | (uint(round(tex1.w * 255.0F)) << 8);
    return uint4(
        input.payload.x + multiPsSet0Ubo.w0 + multiPsSet1Ubo.w0,
        input.payload.y + multiPsSet0Ubo.t1,
        input.payload.z + multiPsSet1Ubo.t0 + multiGraphicsPush.z,
        ((packed0 << 16) | packed1) ^ multiGraphicsPush.w);
}
#endif
