struct Source
{
    float4 value;
};

ConstantBuffer<Source> sources[2] : register(b0, space0);

RWStructuredBuffer<float4> destination : register(u2, space0);

[numthreads(1, 1, 1)]
void main()
{
    destination[0] = sources[0].value;
    destination[1] = sources[1].value;
}
