cbuffer SparseTextureParameters : register(b0, space0)
{
    uint2 mappedCoordinate;
    uint2 unmappedCoordinate;
};

Texture2D<float> sparseInput : register(t1, space0);
RWTexture2D<float> sparseOutput : register(u2, space0);
RWStructuredBuffer<uint> observedValues : register(u3, space0);

[numthreads(1, 1, 1)]
void main()
{
    uint mappedStatus;
    uint unmappedStatus;
    const float mappedValue = sparseInput.Load(int3(mappedCoordinate, 0), int2(0, 0), mappedStatus);
    const float unmappedValue = sparseInput.Load(int3(unmappedCoordinate, 0), int2(0, 0), unmappedStatus);
    observedValues[0] = asuint(mappedValue);
    observedValues[1] = asuint(unmappedValue);
    observedValues[2] = CheckAccessFullyMapped(mappedStatus) ? 1 : 0;
    observedValues[3] = CheckAccessFullyMapped(unmappedStatus) ? 1 : 0;
    sparseOutput[mappedCoordinate] = mappedValue + 1.0;
    sparseOutput[unmappedCoordinate] = -12345.0;
}
