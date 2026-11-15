[[vk::constant_id(7)]] const uint kMultiplier = 1;

RWStructuredBuffer<uint> values : register(u0, space0);

[numthreads(1, 1, 1)]
void main()
{
    values[0] = kMultiplier;
}
