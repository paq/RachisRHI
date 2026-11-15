[[vk::binding(0, 0)]] RWStructuredBuffer<uint> arguments : register(u0);

[numthreads(1, 1, 1)]
void writeDraw()
{
    arguments[0] = 3;
    arguments[1] = 1;
    arguments[2] = 0;
    arguments[3] = 0;
}

[numthreads(1, 1, 1)]
void writeIndexedDraw()
{
    arguments[0] = 3;
    arguments[1] = 1;
    arguments[2] = 0;
    arguments[3] = 0;
    arguments[4] = 0;
}

[numthreads(1, 1, 1)]
void writeDispatch()
{
    arguments[0] = 1;
    arguments[1] = 1;
    arguments[2] = 1;
}
