RWByteAddressBuffer output : register(u0);

[numthreads(1, 1, 1)]
void main()
{
    uint originalValue;
    output.InterlockedAdd(0, 1, originalValue);
}
