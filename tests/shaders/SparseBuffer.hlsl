ByteAddressBuffer sparseInput : register(t0, space0);
RWStructuredBuffer<uint> outputBuffer : register(u1, space0);

[numthreads(1, 1, 1)]
void main()
{
    outputBuffer[0] = sparseInput.Load(0);
    outputBuffer[1] = sparseInput.Load(65536);
}
