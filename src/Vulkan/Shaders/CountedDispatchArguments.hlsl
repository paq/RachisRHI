[[vk::constant_id(0)]] const uint parametersLow = 0;
[[vk::constant_id(1)]] const uint parametersHigh = 0;

[numthreads(64, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    const uint64_t parameters = (uint64_t(parametersHigh) << 32) | uint64_t(parametersLow);
    const uint4 sources = vk::RawBufferLoad<uint4>(parameters, 4);
    const uint4 destination = vk::RawBufferLoad<uint4>(parameters + 16, 4);
    const uint groupsPerRow = vk::RawBufferLoad<uint>(parameters + 32, 4);
    const uint64_t arguments = uint64_t(sources.x) | (uint64_t(sources.y) << 32);
    const uint64_t countAddress = uint64_t(sources.z) | (uint64_t(sources.w) << 32);
    const uint64_t records = uint64_t(destination.x) | (uint64_t(destination.y) << 32);
    const uint stride = destination.z;
    const uint maxCount = destination.w;

    const uint64_t record = (uint64_t(groupId.y) * groupsPerRow + groupId.x) * 64 + groupThreadId.x;
    if (record >= maxCount)
    {
        return;
    }
    const uint count = vk::RawBufferLoad<uint>(countAddress, 4);
    uint3 groups = uint3(0, 0, 0);
    if (record < count)
    {
        groups = vk::RawBufferLoad<uint3>(arguments + record * uint64_t(stride), 4);
    }
    vk::RawBufferStore<uint3>(records + record * 12, groups, 4);
}
