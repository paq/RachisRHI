struct SourceVertex
{
    float4 position;
    float4 color;
    float4 normal;
};

struct RenderVertex
{
    float4 position;
    float4 color;
};

#ifdef RACHIS_METAL
StructuredBuffer<SourceVertex> sourceVertices : register(t1, space0);
RWStructuredBuffer<RenderVertex> renderVertices : register(u2, space0);
#else
StructuredBuffer<SourceVertex> sourceVertices : register(t0, space0);
RWStructuredBuffer<RenderVertex> renderVertices : register(u1, space0);
#endif

[[vk::push_constant]]
cbuffer PushConstants : register(b0, space4)
{
    float animationTimeSeconds;
    float aspectRatio;
    float2 padding;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint vertexId = dispatchThreadId.x;
    if (vertexId >= 36)
    {
        return;
    }

    const SourceVertex source = sourceVertices[vertexId];
    const float yaw = animationTimeSeconds * 0.75F;
    const float pitch = animationTimeSeconds * 0.47F + 0.35F;
    float sinYaw;
    float cosYaw;
    float sinPitch;
    float cosPitch;
    sincos(yaw, sinYaw, cosYaw);
    sincos(pitch, sinPitch, cosPitch);

    const float3 modelPosition = source.position.xyz * 0.78F;
    const float3 yawPosition = float3(
        cosYaw * modelPosition.x + sinYaw * modelPosition.z,
        modelPosition.y,
        -sinYaw * modelPosition.x + cosYaw * modelPosition.z
    );
    const float3 rotatedPosition = float3(
        yawPosition.x,
        cosPitch * yawPosition.y - sinPitch * yawPosition.z,
        sinPitch * yawPosition.y + cosPitch * yawPosition.z
    );

    const float3 modelNormal = source.normal.xyz;
    const float3 yawNormal = float3(
        cosYaw * modelNormal.x + sinYaw * modelNormal.z,
        modelNormal.y,
        -sinYaw * modelNormal.x + cosYaw * modelNormal.z
    );
    const float3 rotatedNormal = float3(
        yawNormal.x,
        cosPitch * yawNormal.y - sinPitch * yawNormal.z,
        sinPitch * yawNormal.y + cosPitch * yawNormal.z
    );
    const float3 lightDirection = normalize(float3(-0.35F, 0.65F, -0.68F));
    const float lighting = 0.38F + 0.62F * saturate(dot(rotatedNormal, lightDirection));

    const float viewZ = rotatedPosition.z + 3.4F;
    const float projectionScale = 1.920982F;
    const float nearPlane = 0.1F;
    const float farPlane = 100.0F;
    RenderVertex output;
    output.position = float4(
        rotatedPosition.x * projectionScale / aspectRatio,
        rotatedPosition.y * projectionScale,
        viewZ * farPlane / (farPlane - nearPlane)
            - nearPlane * farPlane / (farPlane - nearPlane),
        viewZ
    );
    output.color = float4(source.color.xyz * lighting, 1.0F);
    renderVertices[vertexId] = output;
}

struct VertexOutput
{
    float4 position : SV_Position;
    float4 color : COLOR0;
};

struct VertexInput
{
    float4 position : TEXCOORD0;
    float4 color : TEXCOORD1;
};

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    VertexOutput output;
    output.position = input.position;
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 pixelMain(VertexOutput input) : SV_Target0
{
    return input.color;
}
