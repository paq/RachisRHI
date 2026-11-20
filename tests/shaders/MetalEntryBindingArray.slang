RWStructuredBuffer<uint> outputs : register(u0, space1);

[shader("compute")]
[numthreads(1, 1, 1)]
void entryArrayMain(uniform StructuredBuffer<uint> values[])
{
    outputs[0] = values[NonUniformResourceIndex(1)][0];
}
