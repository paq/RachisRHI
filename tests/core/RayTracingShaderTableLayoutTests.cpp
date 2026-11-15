#include "Rachis/Device.h"
#include "Rachis/Pipeline.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace
{

    using rachis::RayTracingCapabilities;
    using rachis::RayTracingShaderGroupDesc;
    using rachis::RayTracingShaderGroupType;
    using rachis::RayTracingShaderTableLayout;
    using rachis::RayTracingShaderTableLayoutDesc;
    using rachis::RayTracingShaderTableRecord;
    using rachis::RayTracingShaderTableRegion;
    using rachis::ShaderRecordConstantRange;

    constexpr RayTracingCapabilities desktopLimits{
        .shaderGroupHandleSize = 32,
        .shaderGroupHandleAlignment = 32,
        .shaderGroupBaseAlignment = 64,
        .maxShaderGroupStride = 4096,
    };

    constexpr RayTracingCapabilities narrowAlignmentLimits{
        .shaderGroupHandleSize = 32,
        .shaderGroupHandleAlignment = 16,
        .shaderGroupBaseAlignment = 32,
        .maxShaderGroupStride = 16384,
    };

    [[nodiscard]] RayTracingShaderGroupDesc general(std::uint32_t constantBytes = 0)
    {
        RayTracingShaderGroupDesc group{.type = RayTracingShaderGroupType::general};
        if (constantBytes != 0)
        {
            group.shaderRecordConstants = ShaderRecordConstantRange{.size = constantBytes};
        }
        return group;
    }

    [[nodiscard]] RayTracingShaderGroupDesc hitGroup(RayTracingShaderGroupType type, std::uint32_t constantBytes = 0)
    {
        auto group = general(constantBytes);
        group.type = type;
        return group;
    }

    [[nodiscard]] bool sameRegion(const RayTracingShaderTableRegion& actual,
                                  std::uint64_t offset,
                                  std::uint64_t size,
                                  std::uint64_t stride) noexcept
    {
        return !actual.buffer.isValid() && actual.offset == offset && actual.size == size && actual.stride == stride;
    }

    [[nodiscard]] bool sameRecord(const RayTracingShaderTableRecord& actual,
                                  std::uint32_t group,
                                  std::uint64_t handleOffset,
                                  std::uint32_t constantsSize,
                                  std::uint32_t recordSize) noexcept
    {
        return actual.group == group && actual.handleOffset == handleOffset &&
               actual.recordConstantsOffset == handleOffset + 32 && actual.recordConstantsSize == constantsSize &&
               actual.recordSize == recordSize;
    }

    void requireDispatchableRegions(const RayTracingShaderTableLayout& layout, const RayTracingCapabilities& limits)
    {
        CHECK(layout.alignment == limits.shaderGroupBaseAlignment);
        const std::array regions{&layout.rayGeneration, &layout.miss, &layout.hitGroup, &layout.callable};
        for (std::size_t index = 0; index < regions.size(); ++index)
        {
            const auto& region = *regions[index];
            if (region.size == 0)
            {
                CHECK(region.offset == 0);
                CHECK(region.stride == 0);
                continue;
            }
            CHECK(region.offset % limits.shaderGroupBaseAlignment == 0);
            CHECK(region.stride >= limits.shaderGroupHandleSize);
            CHECK(region.stride <= limits.maxShaderGroupStride);
            CHECK(region.stride % limits.shaderGroupHandleAlignment == 0);
            CHECK(region.offset + region.size <= layout.size);
            if (index == 0)
            {
                CHECK(region.size == region.stride);
            }
            else
            {
                CHECK(region.size % region.stride == 0);
            }
        }
        for (const auto& record : layout.records)
        {
            CHECK(record.recordConstantsOffset + record.recordConstantsSize <= record.handleOffset + record.recordSize);
            CHECK(record.handleOffset + record.recordSize <= layout.size);
        }
    }

} // namespace

TEST_CASE("layoutRayTracingShaderTable places each table at the base alignment with its largest record as stride",
          "[size:small][rhi][core][ray-tracing-pipeline][shader-table]")
{
    const std::array groups{
        general(),
        general(),
        general(8),
        hitGroup(RayTracingShaderGroupType::trianglesHitGroup),
        hitGroup(RayTracingShaderGroupType::proceduralHitGroup, 16),
    };
    const std::array missGroups{1U, 2U};
    const std::array hitGroups{3U, 3U, 4U};
    const auto layout = rachis::layoutRayTracingShaderTable(
        {
            .groups = groups,
            .rayGenerationGroup = 0,
            .missGroups = missGroups,
            .hitGroups = hitGroups,
        },
        desktopLimits);
    REQUIRE(layout.has_value());

    CHECK(sameRegion(layout->rayGeneration, 0, 32, 32));
    CHECK(sameRegion(layout->miss, 64, 128, 64));
    CHECK(sameRegion(layout->hitGroup, 192, 192, 64));
    CHECK(sameRegion(layout->callable, 0, 0, 0));
    CHECK(layout->size == 384);

    REQUIRE(layout->records.size() == 6);
    CHECK(sameRecord(layout->records[0], 0, 0, 0, 32));
    CHECK(sameRecord(layout->records[1], 1, 64, 0, 32));
    CHECK(sameRecord(layout->records[2], 2, 128, 8, 64));
    CHECK(sameRecord(layout->records[3], 3, 192, 0, 32));
    CHECK(sameRecord(layout->records[4], 3, 256, 0, 32));
    CHECK(sameRecord(layout->records[5], 4, 320, 16, 64));
    requireDispatchableRegions(*layout, desktopLimits);
}

TEST_CASE("layoutRayTracingShaderTable rounds records to a handle alignment below the handle size",
          "[size:small][rhi][core][ray-tracing-pipeline][shader-table]")
{
    const std::array groups{
        general(8),
        general(),
        hitGroup(RayTracingShaderGroupType::trianglesHitGroup, 4),
        general(24),
    };
    const std::array missGroups{1U};
    const std::array hitGroups{2U};
    const std::array callableGroups{3U, 1U};
    const auto layout = rachis::layoutRayTracingShaderTable(
        {
            .groups = groups,
            .rayGenerationGroup = 0,
            .missGroups = missGroups,
            .hitGroups = hitGroups,
            .callableGroups = callableGroups,
        },
        narrowAlignmentLimits);
    REQUIRE(layout.has_value());

    CHECK(sameRegion(layout->rayGeneration, 0, 48, 48));
    CHECK(sameRegion(layout->miss, 64, 32, 32));
    CHECK(sameRegion(layout->hitGroup, 96, 48, 48));
    CHECK(sameRegion(layout->callable, 160, 128, 64));
    CHECK(layout->size == 288);
    REQUIRE(layout->records.size() == 5);
    CHECK(sameRecord(layout->records[3], 3, 160, 24, 64));
    CHECK(sameRecord(layout->records[4], 1, 224, 0, 32));
    requireDispatchableRegions(*layout, narrowAlignmentLimits);
}

TEST_CASE("layoutRayTracingShaderTable leaves empty tables zero and setBuffer names only the used ones",
          "[size:small][rhi][core][ray-tracing-pipeline][shader-table]")
{
    const std::array groups{general(), general()};
    const std::array callableGroups{1U};
    auto layout = rachis::layoutRayTracingShaderTable(
        {.groups = groups, .rayGenerationGroup = 0, .callableGroups = callableGroups}, desktopLimits);
    REQUIRE(layout.has_value());
    CHECK(sameRegion(layout->rayGeneration, 0, 32, 32));
    CHECK(sameRegion(layout->miss, 0, 0, 0));
    CHECK(sameRegion(layout->hitGroup, 0, 0, 0));
    CHECK(sameRegion(layout->callable, 64, 32, 32));
    CHECK(layout->size == 96);
    requireDispatchableRegions(*layout, desktopLimits);

    const rachis::BufferHandle table{.identity = 7};
    layout->setBuffer(table);
    CHECK(layout->rayGeneration.buffer == table);
    CHECK(layout->callable.buffer == table);
    CHECK_FALSE(layout->miss.buffer.isValid());
    CHECK_FALSE(layout->hitGroup.buffer.isValid());
}

TEST_CASE("layoutRayTracingShaderTable rejects groups that cannot form the table",
          "[size:small][rhi][core][ray-tracing-pipeline][shader-table]")
{
    const std::array groups{
        general(),
        hitGroup(RayTracingShaderGroupType::trianglesHitGroup),
        general(6),
        general(4096),
    };
    const auto rejects =
        [&](const RayTracingShaderTableLayoutDesc& desc, const RayTracingCapabilities& limits, rachis::ErrorCode code) {
            const auto layout = rachis::layoutRayTracingShaderTable(desc, limits);
            REQUIRE_FALSE(layout.has_value());
            CHECK(layout.error().code == code);
        };

    rejects({.groups = groups}, RayTracingCapabilities{}, rachis::ErrorCode::unsupported);
    rejects({.groups = groups},
            RayTracingCapabilities{.shaderGroupHandleSize = 32, .shaderGroupBaseAlignment = 64},
            rachis::ErrorCode::invalidArgument);
    rejects({}, desktopLimits, rachis::ErrorCode::invalidArgument);
    rejects({.groups = groups, .rayGenerationGroup = 4}, desktopLimits, rachis::ErrorCode::invalidArgument);
    rejects({.groups = groups, .rayGenerationGroup = 1}, desktopLimits, rachis::ErrorCode::invalidArgument);

    const std::array generalAsHit{0U};
    rejects({.groups = groups, .hitGroups = generalAsHit}, desktopLimits, rachis::ErrorCode::invalidArgument);
    const std::array hitAsMiss{1U};
    rejects({.groups = groups, .missGroups = hitAsMiss}, desktopLimits, rachis::ErrorCode::invalidArgument);
    const std::array unalignedConstants{2U};
    rejects({.groups = groups, .missGroups = unalignedConstants}, desktopLimits, rachis::ErrorCode::invalidArgument);
    const std::array oversizedRecord{3U};
    rejects({.groups = groups, .callableGroups = oversizedRecord}, desktopLimits, rachis::ErrorCode::invalidArgument);
}
