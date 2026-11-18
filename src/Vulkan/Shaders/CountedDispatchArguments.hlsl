struct CountedDispatchParameters
{
    uint64_t arguments;
    uint64_t count;
    uint64_t records;
    uint stride;
    uint firstRecord;
    uint maxCount;
    uint padding;
};

[[vk::push_constant]] CountedDispatchParameters parameters;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint record = parameters.firstRecord + dispatchThreadId.x;
    if (dispatchThreadId.x >= parameters.maxCount - parameters.firstRecord)
    {
        return;
    }
    const uint count = vk::RawBufferLoad<uint>(parameters.count, 4);
    uint3 groups = uint3(0, 0, 0);
    if (record < count)
    {
        groups = vk::RawBufferLoad<uint3>(parameters.arguments + uint64_t(record) * uint64_t(parameters.stride), 4);
    }
    vk::RawBufferStore<uint3>(parameters.records + uint64_t(record) * 12, groups, 4);
}
