struct Params
{
    uint count;
    float3 bias;
    float scale;
};

struct Element
{
    float3 position;
    uint id;
};

struct Push
{
    uint offset;
    uint stride;
};

StructuredBuffer<Element> elements : register(t0, space0);
RWStructuredBuffer<uint> output : register(u1, space0);
ConstantBuffer<Params> params : register(b0, space1);
[[vk::push_constant]] ConstantBuffer<Push> push : register(b0, space4);

interface IElementMap
{
    uint map(Element element);
}

struct ScaledSum : IElementMap
{
    float3 bias;
    float scale;

    uint map(Element element)
    {
        const float3 shifted = (element.position + bias) * scale;
        return uint(shifted.x + shifted.y + shifted.z) + element.id;
    }
}

uint mapElement<T : IElementMap>(T mapping, Element element)
{
    return mapping.map(element);
}

[shader("compute")]
[numthreads(4, 1, 1)]
void slangMain(uint3 threadId : SV_DispatchThreadID)
{
    if (threadId.x >= params.count)
    {
        return;
    }
    ScaledSum mapping = { params.bias, params.scale };
    output[push.offset + threadId.x * push.stride] = mapElement(mapping, elements[threadId.x]);
}
