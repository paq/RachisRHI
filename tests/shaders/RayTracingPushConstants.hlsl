RWStructuredBuffer<uint> outputBuffer : register(u0, space0);

struct RayGenerationConstants
{
    uint4 values;
};

[[vk::push_constant]]
RayGenerationConstants rayGenerationConstants;

[shader("raygeneration")]
void rayGenerationWithPushConstants()
{
    outputBuffer[0] = rayGenerationConstants.values.x;
}
