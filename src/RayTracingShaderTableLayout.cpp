#include "Rachis/Device.h"
#include "Rachis/Pipeline.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rachis
{
    namespace
    {

        [[nodiscard]] constexpr std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment) noexcept
        {
            return ((value + alignment - 1U) / alignment) * alignment;
        }

        [[nodiscard]] constexpr bool isHitGroup(RayTracingShaderGroupType type) noexcept
        {
            return type == RayTracingShaderGroupType::trianglesHitGroup ||
                   type == RayTracingShaderGroupType::proceduralHitGroup;
        }

        struct TableSpec final
        {
            std::span<const std::uint32_t> groups;
            bool hitGroups = false;
            std::string_view name;
            RayTracingShaderTableRegion* region = nullptr;
        };

        [[nodiscard]] Result<std::uint64_t> layoutTable(const RayTracingShaderTableLayoutDesc& desc,
                                                        const RayTracingCapabilities& limits,
                                                        const TableSpec& table,
                                                        std::uint64_t offset,
                                                        std::vector<RayTracingShaderTableRecord>& records)
        {
            if (table.groups.empty())
            {
                return offset;
            }
            const auto firstRecord = records.size();
            std::uint64_t stride = 0;
            for (const auto groupIndex : table.groups)
            {
                if (groupIndex >= desc.groups.size())
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "The " + std::string(table.name) + " table names group " + std::to_string(groupIndex) +
                            ", but the pipeline has " + std::to_string(desc.groups.size()) + " groups.",
                    });
                }
                const auto& group = desc.groups[groupIndex];
                if (isHitGroup(group.type) != table.hitGroups)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Group " + std::to_string(groupIndex) + " in the " + std::string(table.name) +
                            " table must be a " + (table.hitGroups ? "hit group." : "general group."),
                    });
                }
                const auto constantsSize =
                    group.shaderRecordConstants.has_value() ? group.shaderRecordConstants->size : 0U;
                if (group.shaderRecordConstants.has_value() &&
                    (constantsSize == 0 || constantsSize % sizeof(std::uint32_t) != 0))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Group " + std::to_string(groupIndex) +
                            " declares shader-record constants that are empty or not DWORD aligned.",
                    });
                }
                const auto recordSize =
                    alignUp(static_cast<std::uint64_t>(limits.shaderGroupHandleSize) + constantsSize,
                            limits.shaderGroupHandleAlignment);
                if (recordSize > limits.maxShaderGroupStride)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Group " + std::to_string(groupIndex) + " needs a " + std::to_string(recordSize) +
                            "-byte shader record, more than the device's maximum stride of " +
                            std::to_string(limits.maxShaderGroupStride) + " bytes.",
                    });
                }
                stride = (std::max)(stride, recordSize);
                records.push_back(RayTracingShaderTableRecord{
                    .group = groupIndex,
                    .recordConstantsSize = constantsSize,
                    .recordSize = static_cast<std::uint32_t>(recordSize),
                });
            }
            for (std::size_t index = firstRecord; index < records.size(); ++index)
            {
                auto& record = records[index];
                record.handleOffset = offset + (index - firstRecord) * stride;
                record.recordConstantsOffset = record.handleOffset + limits.shaderGroupHandleSize;
            }
            const auto size = stride * table.groups.size();
            *table.region = RayTracingShaderTableRegion{.offset = offset, .size = size, .stride = stride};
            return offset + size;
        }

    } // namespace

    Result<RayTracingShaderTableLayout> layoutRayTracingShaderTable(const RayTracingShaderTableLayoutDesc& desc,
                                                                    const RayTracingCapabilities& limits)
    {
        if (limits.shaderGroupHandleSize == 0)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device reports no shader-group handles, so it has no ray-tracing pipelines.",
            });
        }
        if (limits.shaderGroupHandleAlignment == 0 || limits.shaderGroupBaseAlignment == 0 ||
            limits.maxShaderGroupStride < limits.shaderGroupHandleSize)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The ray-tracing limits need non-zero handle and base alignments and a maximum stride that holds "
                "a handle.",
            });
        }

        RayTracingShaderTableLayout layout{.alignment = limits.shaderGroupBaseAlignment};
        const std::array rayGenerationGroups{desc.rayGenerationGroup};
        const std::array tables{
            TableSpec{.groups = rayGenerationGroups, .name = "ray-generation", .region = &layout.rayGeneration},
            TableSpec{.groups = desc.missGroups, .name = "miss", .region = &layout.miss},
            TableSpec{.groups = desc.hitGroups, .hitGroups = true, .name = "hit-group", .region = &layout.hitGroup},
            TableSpec{.groups = desc.callableGroups, .name = "callable", .region = &layout.callable},
        };
        layout.records.reserve(1U + desc.missGroups.size() + desc.hitGroups.size() + desc.callableGroups.size());
        std::uint64_t offset = 0;
        for (const auto& table : tables)
        {
            auto end =
                layoutTable(desc, limits, table, alignUp(offset, limits.shaderGroupBaseAlignment), layout.records);
            if (!end)
            {
                return std::unexpected(std::move(end.error()));
            }
            if (table.region->size != 0)
            {
                layout.size = *end;
                offset = *end;
            }
        }
        return layout;
    }

} // namespace rachis
