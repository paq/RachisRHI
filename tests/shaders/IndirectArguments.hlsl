RWStructuredBuffer<uint> arguments : register(u0, space0);

[numthreads(1, 1, 1)]
void main()
{
    arguments[0] = 3;
    arguments[1] = 1;
    arguments[2] = 3;
    arguments[3] = 0;
}
