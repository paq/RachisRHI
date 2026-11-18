#pragma once

#include "Rachis/Resource.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>
#include <variant>

namespace rachis
{

    enum class AccelerationStructureType : std::uint8_t
    {
        bottomLevel,
        topLevel,
    };

    enum class AccelerationStructureBuildFlag : std::uint8_t
    {
        none = 0,
        preferFastTrace = 1U << 0U,
        preferFastBuild = 1U << 1U,
        minimizeMemory = 1U << 2U,
        allowUpdate = 1U << 3U,
        allowCompaction = 1U << 4U,
    };

    [[nodiscard]] constexpr AccelerationStructureBuildFlag operator|(AccelerationStructureBuildFlag left,
                                                                     AccelerationStructureBuildFlag right) noexcept
    {
        return static_cast<AccelerationStructureBuildFlag>(
            static_cast<std::underlying_type_t<AccelerationStructureBuildFlag>>(left) |
            static_cast<std::underlying_type_t<AccelerationStructureBuildFlag>>(right));
    }

    [[nodiscard]] constexpr AccelerationStructureBuildFlag operator&(AccelerationStructureBuildFlag left,
                                                                     AccelerationStructureBuildFlag right) noexcept
    {
        return static_cast<AccelerationStructureBuildFlag>(
            static_cast<std::underlying_type_t<AccelerationStructureBuildFlag>>(left) &
            static_cast<std::underlying_type_t<AccelerationStructureBuildFlag>>(right));
    }

    [[nodiscard]] constexpr bool hasAny(AccelerationStructureBuildFlag value,
                                        AccelerationStructureBuildFlag flags) noexcept
    {
        return (value & flags) != AccelerationStructureBuildFlag::none;
    }

    enum class AccelerationStructureGeometryFlag : std::uint8_t
    {
        none = 0,
        opaque = 1U << 0U,
        noDuplicateAnyHitInvocation = 1U << 1U,
    };

    [[nodiscard]] constexpr AccelerationStructureGeometryFlag operator|(
        AccelerationStructureGeometryFlag left, AccelerationStructureGeometryFlag right) noexcept
    {
        return static_cast<AccelerationStructureGeometryFlag>(
            static_cast<std::underlying_type_t<AccelerationStructureGeometryFlag>>(left) |
            static_cast<std::underlying_type_t<AccelerationStructureGeometryFlag>>(right));
    }

    [[nodiscard]] constexpr AccelerationStructureGeometryFlag operator&(
        AccelerationStructureGeometryFlag left, AccelerationStructureGeometryFlag right) noexcept
    {
        return static_cast<AccelerationStructureGeometryFlag>(
            static_cast<std::underlying_type_t<AccelerationStructureGeometryFlag>>(left) &
            static_cast<std::underlying_type_t<AccelerationStructureGeometryFlag>>(right));
    }

    [[nodiscard]] constexpr bool hasAny(AccelerationStructureGeometryFlag value,
                                        AccelerationStructureGeometryFlag flags) noexcept
    {
        return (value & flags) != AccelerationStructureGeometryFlag::none;
    }

    enum class AccelerationStructureVertexFormat : std::uint8_t
    {
        float32x2,
        float32x3,
        float16x2,
        float16x4,
        snorm16x2,
        snorm16x4,
    };

    struct AccelerationStructureTriangleGeometry final
    {
        BufferHandle vertexBuffer = {};
        std::uint64_t vertexOffset = 0;
        std::uint32_t vertexCount = 0;
        std::uint32_t vertexStride = 0;
        AccelerationStructureVertexFormat vertexFormat = AccelerationStructureVertexFormat::float32x3;

        BufferHandle indexBuffer = {};
        std::uint64_t indexOffset = 0;
        std::uint32_t indexCount = 0;
        IndexFormat indexFormat = IndexFormat::uint32;

        BufferHandle transformBuffer = {};
        std::uint64_t transformOffset = 0;
    };

    struct AccelerationStructureAabbGeometry final
    {
        BufferHandle buffer = {};
        std::uint64_t offset = 0;
        std::uint32_t count = 0;

        std::uint32_t stride = 6U * sizeof(float);
    };

    using AccelerationStructureGeometryData =
        std::variant<AccelerationStructureTriangleGeometry, AccelerationStructureAabbGeometry>;

    struct AccelerationStructureGeometry final
    {
        AccelerationStructureGeometryData data = {};
        AccelerationStructureGeometryFlag flags = AccelerationStructureGeometryFlag::none;
    };

    struct AccelerationStructureBuildSizes final
    {
        std::uint64_t accelerationStructureSize = 0;
        std::uint64_t buildScratchSize = 0;
        std::uint64_t updateScratchSize = 0;
    };

    struct BottomLevelAccelerationStructureBuildInput final
    {
        std::span<const AccelerationStructureGeometry> geometries = {};
        AccelerationStructureBuildFlag flags = AccelerationStructureBuildFlag::none;
    };

    struct TopLevelAccelerationStructureBuildInput final
    {
        std::uint32_t instanceCount = 0;
        AccelerationStructureBuildFlag flags = AccelerationStructureBuildFlag::none;
    };

    struct AccelerationStructureDesc final
    {
        AccelerationStructureType type = AccelerationStructureType::bottomLevel;
        BufferHandle storage = {};
        std::uint64_t offset = 0;
        std::uint64_t size = wholeResource;
        std::string_view label = {};
    };

    enum class AccelerationStructureInstanceFlag : std::uint8_t
    {
        none = 0,
        triangleCullDisable = 1U << 0U,
        triangleFrontCounterClockwise = 1U << 1U,
        forceOpaque = 1U << 2U,
        forceNonOpaque = 1U << 3U,
    };

    [[nodiscard]] constexpr AccelerationStructureInstanceFlag operator|(
        AccelerationStructureInstanceFlag left, AccelerationStructureInstanceFlag right) noexcept
    {
        return static_cast<AccelerationStructureInstanceFlag>(
            static_cast<std::underlying_type_t<AccelerationStructureInstanceFlag>>(left) |
            static_cast<std::underlying_type_t<AccelerationStructureInstanceFlag>>(right));
    }

    [[nodiscard]] constexpr AccelerationStructureInstanceFlag operator&(
        AccelerationStructureInstanceFlag left, AccelerationStructureInstanceFlag right) noexcept
    {
        return static_cast<AccelerationStructureInstanceFlag>(
            static_cast<std::underlying_type_t<AccelerationStructureInstanceFlag>>(left) &
            static_cast<std::underlying_type_t<AccelerationStructureInstanceFlag>>(right));
    }

    [[nodiscard]] constexpr bool hasAny(AccelerationStructureInstanceFlag value,
                                        AccelerationStructureInstanceFlag flags) noexcept
    {
        return (value & flags) != AccelerationStructureInstanceFlag::none;
    }

    struct AccelerationStructureInstance final
    {
        std::array<float, 12> transform{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
        AccelerationStructureHandle bottomLevel = {};
        std::uint32_t instanceId = 0;
        std::uint32_t instanceMask = 0xFF;
        std::uint32_t hitGroupOffset = 0;
        AccelerationStructureInstanceFlag flags = AccelerationStructureInstanceFlag::none;
    };

    enum class AccelerationStructureInstanceFormat : std::uint8_t
    {
        packed64,
    };

    struct AccelerationStructureInstanceLayout final
    {
        AccelerationStructureInstanceFormat format = AccelerationStructureInstanceFormat::packed64;
        std::uint32_t size = 0;
        std::uint32_t alignment = 0;
    };

    struct alignas(16) AccelerationStructureInstanceRecord final
    {
        std::array<float, 12> transform{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
        std::uint32_t instanceIdAndMask = 0xFF00'0000U;
        std::uint32_t hitGroupOffsetAndFlags = 0;
        std::uint64_t bottomLevel = 0;
    };

    static_assert(sizeof(AccelerationStructureInstanceRecord) == 64);
    static_assert(offsetof(AccelerationStructureInstanceRecord, instanceIdAndMask) == 48);
    static_assert(offsetof(AccelerationStructureInstanceRecord, hitGroupOffsetAndFlags) == 52);
    static_assert(offsetof(AccelerationStructureInstanceRecord, bottomLevel) == 56);

    struct BottomLevelAccelerationStructureBuildDesc final
    {
        AccelerationStructureHandle destination = {};
        BottomLevelAccelerationStructureBuildInput input = {};
        BufferHandle scratch = {};
        std::uint64_t scratchOffset = 0;
    };

    struct BottomLevelAccelerationStructureUpdateDesc final
    {
        AccelerationStructureHandle source = {};
        AccelerationStructureHandle destination = {};
        BottomLevelAccelerationStructureBuildInput input = {};
        BufferHandle scratch = {};
        std::uint64_t scratchOffset = 0;
    };

    struct BottomLevelAccelerationStructureInPlaceUpdateDesc final
    {
        AccelerationStructureHandle target = {};
        BottomLevelAccelerationStructureBuildInput input = {};
        BufferHandle scratch = {};
        std::uint64_t scratchOffset = 0;
    };

    struct TopLevelAccelerationStructureBuildDesc final
    {
        AccelerationStructureHandle destination = {};
        std::span<const AccelerationStructureInstance> instances = {};
        AccelerationStructureBuildFlag flags = AccelerationStructureBuildFlag::none;
        BufferHandle scratch = {};
        std::uint64_t scratchOffset = 0;
    };

    struct TopLevelAccelerationStructureUpdateDesc final
    {
        AccelerationStructureHandle source = {};
        AccelerationStructureHandle destination = {};
        std::span<const AccelerationStructureInstance> instances = {};
        AccelerationStructureBuildFlag flags = AccelerationStructureBuildFlag::none;
        BufferHandle scratch = {};
        std::uint64_t scratchOffset = 0;
    };

    struct TopLevelAccelerationStructureInPlaceUpdateDesc final
    {
        AccelerationStructureHandle target = {};
        std::span<const AccelerationStructureInstance> instances = {};
        AccelerationStructureBuildFlag flags = AccelerationStructureBuildFlag::none;
        BufferHandle scratch = {};
        std::uint64_t scratchOffset = 0;
    };

    struct TopLevelAccelerationStructureBufferBuildDesc final
    {
        AccelerationStructureHandle destination = {};
        BufferHandle instanceBuffer = {};
        std::uint64_t instanceOffset = 0;
        std::uint32_t instanceCount = 0;
        AccelerationStructureBuildFlag flags = AccelerationStructureBuildFlag::none;
        BufferHandle scratch = {};
        std::uint64_t scratchOffset = 0;
    };

    struct TopLevelAccelerationStructureBufferUpdateDesc final
    {
        AccelerationStructureHandle source = {};
        AccelerationStructureHandle destination = {};
        BufferHandle instanceBuffer = {};
        std::uint64_t instanceOffset = 0;
        std::uint32_t instanceCount = 0;
        AccelerationStructureBuildFlag flags = AccelerationStructureBuildFlag::none;
        BufferHandle scratch = {};
        std::uint64_t scratchOffset = 0;
    };

    struct TopLevelAccelerationStructureBufferInPlaceUpdateDesc final
    {
        AccelerationStructureHandle target = {};
        BufferHandle instanceBuffer = {};
        std::uint64_t instanceOffset = 0;
        std::uint32_t instanceCount = 0;
        AccelerationStructureBuildFlag flags = AccelerationStructureBuildFlag::none;
        BufferHandle scratch = {};
        std::uint64_t scratchOffset = 0;
    };

    struct AccelerationStructureCompactionDesc final
    {
        AccelerationStructureHandle source = {};
        AccelerationStructureHandle destination = {};
        std::uint64_t compactedSize = 0;
    };

    struct AccelerationStructureSerializationDesc final
    {
        AccelerationStructureHandle source = {};
        BufferHandle destination = {};
        std::uint64_t destinationOffset = 0;
        std::uint64_t serializedSize = 0;
    };

    enum class AccelerationStructureArchiveCompatibility : std::uint8_t
    {
        compatible,
        unsupportedType,
        unrecognized,
        incompatibleVersion,
        incompatibleType,
    };

    struct AccelerationStructureArchiveInfo final
    {
        AccelerationStructureArchiveCompatibility compatibility =
            AccelerationStructureArchiveCompatibility::unrecognized;
        std::uint64_t serializedSize = 0;
        std::uint64_t deserializedSize = 0;
        std::uint64_t bottomLevelPointerCount = 0;
    };

    struct AccelerationStructureDeserializationDesc final
    {
        AccelerationStructureHandle destination = {};
        std::span<const std::byte> archive = {};
        std::span<const AccelerationStructureHandle> bottomLevels = {};
    };

} // namespace rachis
