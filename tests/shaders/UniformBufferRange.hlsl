cbuffer Source : register(b0, space0)
{
    float4 value;
};

RWStructuredBuffer<float4> destination : register(u1, space0);

[numthreads(1, 1, 1)]
void main()
{
    destination[0] = value;
}
