cbuffer ConstantMatrices : register(b1, space0)
{
    float4x4 constantSquare;
    float3x4 constantWide;
    float4 constantTail;
};

struct PushMatrices
{
    float3x4 wide;
    float4x4 square;
};

#if defined(RACHIS_VULKAN) || defined(RACHIS_METAL)
[[vk::push_constant]]
#endif
ConstantBuffer<PushMatrices> pushMatrices : register(b0, space4);

struct StructuredMatrices
{
    float4x4 square;
    float3x4 wide;
};

StructuredBuffer<StructuredMatrices> structuredMatrices : register(t2, space0);
RWStructuredBuffer<float> outputValues : register(u3, space0);

void writeSquare(uint first, float4x4 value)
{
    [unroll] for (uint row = 0; row < 4; ++row)
    {
        [unroll] for (uint column = 0; column < 4; ++column)
        {
            outputValues[first + (row * 4) + column] = value[row][column];
        }
    }
}

void writeWide(uint first, float3x4 value)
{
    [unroll] for (uint row = 0; row < 3; ++row)
    {
        [unroll] for (uint column = 0; column < 4; ++column)
        {
            outputValues[first + (row * 4) + column] = value[row][column];
        }
    }
}

[numthreads(1, 1, 1)]
void matrixLayoutMain()
{
    writeSquare(0, constantSquare);
    writeWide(16, constantWide);
    [unroll] for (uint component = 0; component < 4; ++component)
    {
        outputValues[28 + component] = constantTail[component];
    }
    writeWide(32, pushMatrices.wide);
    writeSquare(44, pushMatrices.square);
    [unroll] for (uint element = 0; element < 2; ++element)
    {
        writeSquare(60 + (element * 28), structuredMatrices[element].square);
        writeWide(76 + (element * 28), structuredMatrices[element].wide);
    }
}
