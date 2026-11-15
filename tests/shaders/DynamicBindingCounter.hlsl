RWByteAddressBuffer counters : register(u0, space0);

[numthreads(1, 1, 1)]
void main()
{
    counters.InterlockedAdd(0, 1);
}
