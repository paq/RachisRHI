#ifndef RACHIS_MIPMAP_DIMENSION
#define RACHIS_MIPMAP_DIMENSION 2
#endif

#ifdef RACHIS_METAL
#define RACHIS_MIPMAP_BINDING(index, space) [[vk::binding(index, space)]]
#else
#define RACHIS_MIPMAP_BINDING(index, space)
#endif

#if defined(__spirv__) || defined(RACHIS_METAL)
#define RACHIS_MIPMAP_UNKNOWN_IMAGE_FORMAT [[vk::image_format("unknown")]]
#else
#define RACHIS_MIPMAP_UNKNOWN_IMAGE_FORMAT
#endif

#if RACHIS_MIPMAP_DIMENSION == 1
RACHIS_MIPMAP_BINDING(0, 1) Texture1DArray<float4> sourceTexture : register(t0);
RACHIS_MIPMAP_BINDING(1, 1) RACHIS_MIPMAP_UNKNOWN_IMAGE_FORMAT RWTexture1DArray<float4> destinationTexture : register(u0);
#elif RACHIS_MIPMAP_DIMENSION == 2
RACHIS_MIPMAP_BINDING(0, 1) Texture2DArray<float4> sourceTexture : register(t0);
RACHIS_MIPMAP_BINDING(1, 1) RACHIS_MIPMAP_UNKNOWN_IMAGE_FORMAT RWTexture2DArray<float4> destinationTexture : register(u0);
#elif RACHIS_MIPMAP_DIMENSION == 3
RACHIS_MIPMAP_BINDING(0, 1) Texture3D<float4> sourceTexture : register(t0);
RACHIS_MIPMAP_BINDING(1, 1) RACHIS_MIPMAP_UNKNOWN_IMAGE_FORMAT RWTexture3D<float4> destinationTexture : register(u0);
#else
#error Unsupported RHI mipmap dimension
#endif

RACHIS_MIPMAP_BINDING(0, 2) SamplerState linearClampSampler : register(s0);

cbuffer MipmapGenerationConstants : register(b0)
{
    uint encodeSrgb;
};

float3 linearToSrgb(float3 value)
{
    const float3 linearValue = saturate(value);
    const float3 low = linearValue * 12.92f;
    const float3 high = 1.055f * pow(linearValue, 1.0f / 2.4f) - 0.055f;
    return select(linearValue <= 0.0031308f, low, high);
}

float4 encodeDestination(float4 value)
{
    return encodeSrgb == 0 ? value : float4(linearToSrgb(value.rgb), value.a);
}

#if RACHIS_MIPMAP_DIMENSION == 1
#ifdef RACHIS_METAL
[shader("compute")]
#endif
[numthreads(64, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint width;
    uint layers;
    destinationTexture.GetDimensions(width, layers);
    if (threadId.x >= width || threadId.z >= layers)
    {
        return;
    }

    const float coordinate = (float(threadId.x) + 0.5f) / float(width);
    destinationTexture[uint2(threadId.x, threadId.z)] =
        encodeDestination(sourceTexture.SampleLevel(linearClampSampler, float2(coordinate, threadId.z), 0.0f));
}
#elif RACHIS_MIPMAP_DIMENSION == 2
#ifdef RACHIS_METAL
[shader("compute")]
#endif
[numthreads(8, 8, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint width;
    uint height;
    uint layers;
    destinationTexture.GetDimensions(width, height, layers);
    if (threadId.x >= width || threadId.y >= height || threadId.z >= layers)
    {
        return;
    }

    const float2 coordinate = (float2(threadId.xy) + 0.5f) / float2(width, height);
    destinationTexture[threadId] =
        encodeDestination(sourceTexture.SampleLevel(linearClampSampler, float3(coordinate, threadId.z), 0.0f));
}
#else
#ifdef RACHIS_METAL
[shader("compute")]
#endif
[numthreads(4, 4, 4)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint width;
    uint height;
    uint depth;
    destinationTexture.GetDimensions(width, height, depth);
    if (threadId.x >= width || threadId.y >= height || threadId.z >= depth)
    {
        return;
    }

    const float3 coordinate = (float3(threadId) + 0.5f) / float3(width, height, depth);
    destinationTexture[threadId] = encodeDestination(sourceTexture.SampleLevel(linearClampSampler, coordinate, 0.0f));
}
#endif
