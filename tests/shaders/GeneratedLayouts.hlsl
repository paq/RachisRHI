struct Inner
{
    uint2 ids;
    float3 v;
};

cbuffer Constants : register(b0, space0)
{
    float a;
    float3 b;
    float2 c;
    float d;
    float3 e;
    float weights[3];
    float after;
    float2x3 m;
    Inner inner;
};

struct Element
{
    float3 position;
    float radius;
    uint2 ids;
};

StructuredBuffer<Element> elements : register(t1, space0);
RWStructuredBuffer<uint> output : register(u2, space0);

struct Push
{
    uint first;
    float3 scale;
};

#if defined(RACHIS_METAL) || defined(__spirv__)
[[vk::push_constant]]
#endif
ConstantBuffer<Push> push : register(b0, space4);

[shader("compute")]
[numthreads(1, 1, 1)]
void layoutMain()
{
    uint index = 0;
    output[index++] = asuint(a);
    output[index++] = asuint(b.x);
    output[index++] = asuint(b.y);
    output[index++] = asuint(b.z);
    output[index++] = asuint(c.x);
    output[index++] = asuint(c.y);
    output[index++] = asuint(d);
    output[index++] = asuint(e.x);
    output[index++] = asuint(e.y);
    output[index++] = asuint(e.z);
    for (uint weight = 0; weight < 3; ++weight)
    {
        output[index++] = asuint(weights[weight]);
    }
    output[index++] = asuint(after);
    for (uint column = 0; column < 3; ++column)
    {
        for (uint row = 0; row < 2; ++row)
        {
            output[index++] = asuint(m[row][column]);
        }
    }
    output[index++] = inner.ids.x;
    output[index++] = inner.ids.y;
    output[index++] = asuint(inner.v.x);
    output[index++] = asuint(inner.v.y);
    output[index++] = asuint(inner.v.z);
    for (uint element = 0; element < 2; ++element)
    {
        output[index++] = asuint(elements[element].position.x);
        output[index++] = asuint(elements[element].position.y);
        output[index++] = asuint(elements[element].position.z);
        output[index++] = asuint(elements[element].radius);
        output[index++] = elements[element].ids.x;
        output[index++] = elements[element].ids.y;
    }
    output[index++] = push.first;
    output[index++] = asuint(push.scale.x);
    output[index++] = asuint(push.scale.y);
    output[index++] = asuint(push.scale.z);
}
