struct InstanceRecord
{
    float4 transform[3];
    uint instanceIdAndMask;
    uint hitGroupOffsetAndFlags;
    uint2 bottomLevel;
};

cbuffer InstanceInputs : register(b0, space0)
{
    uint4 inputs[3];
};

RWStructuredBuffer<InstanceRecord> records : register(u1, space0);

[numthreads(3, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint4 input = inputs[id.x];
    InstanceRecord record;
    record.transform[0] = float4(1.0, 0.0, 0.0, asfloat(input.z));
    record.transform[1] = float4(0.0, 1.0, 0.0, 0.0);
    record.transform[2] = float4(0.0, 0.0, 1.0, 0.0);
    record.instanceIdAndMask = (input.w & 0x00FFFFFFu) | (0xFFu << 24);
    record.hitGroupOffsetAndFlags = 1u << 24;
    record.bottomLevel = input.xy;
    records[id.x] = record;
}
