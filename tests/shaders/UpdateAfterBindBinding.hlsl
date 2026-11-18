struct Selection
{
    uint element;
    uint slot;
};

#ifdef __spirv__
[[vk::push_constant]]
#endif
ConstantBuffer<Selection> selection : register(b0, space4);

RWStructuredBuffer<uint> results : register(u0, space0);
StructuredBuffer<uint> inputs[100000] : register(t1, space0);

[numthreads(1, 1, 1)]
void main()
{
    results[selection.slot] = inputs[NonUniformResourceIndex(selection.element)][0];
}
