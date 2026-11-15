cbuffer Coefficients : register(b0, space0)
{
    uint multiplier;
    uint addend;
};

RWStructuredBuffer<uint> output : register(u1, space0);

cbuffer Bias : register(b0, space1)
{
    uint bias;
};

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    output[dispatchThreadId.x] = dispatchThreadId.x * multiplier + addend + bias;
}
