struct Nested
{
    float a;
    float3 b;
};

struct LayoutConstants
{
    Nested nested;
    float3x2 matrix;
    float arrayValues[2];
    uint marker;
};

ConstantBuffer<LayoutConstants> constants : register(b0, space0);
RWStructuredBuffer<uint> outputValues : register(u1, space0);

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 id : SV_DispatchThreadID)
{
    outputValues[0] = asuint(constants.nested.a);
    outputValues[1] = asuint(constants.nested.b.x);
    outputValues[2] = asuint(constants.nested.b.y);
    outputValues[3] = asuint(constants.nested.b.z);
    outputValues[4] = asuint(constants.matrix[0][0]);
    outputValues[5] = asuint(constants.arrayValues[1]);
    outputValues[6] = constants.marker;
    outputValues[7] = 0xA5A5A5A5u;
}
