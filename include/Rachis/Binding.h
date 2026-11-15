#pragma once

#include "Rachis/Resource.h"
#include "Rachis/Sampler.h"

#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>
#include <variant>

namespace rachis
{

    enum class ShaderStage : std::uint16_t
    {
        none = 0,
        vertex = 1U << 0U,
        pixel = 1U << 1U,
        compute = 1U << 2U,
        geometry = 1U << 3U,
        hull = 1U << 4U,
        domain = 1U << 5U,
        task = 1U << 6U,
        mesh = 1U << 7U,
        rayGeneration = 1U << 8U,
        anyHit = 1U << 9U,
        closestHit = 1U << 10U,
        miss = 1U << 11U,
        intersection = 1U << 12U,
        callable = 1U << 13U,
        allGraphics = (1U << 0U) | (1U << 1U) | (1U << 3U) | (1U << 4U) | (1U << 5U) | (1U << 6U) | (1U << 7U),
        allRayTracing = (1U << 8U) | (1U << 9U) | (1U << 10U) | (1U << 11U) | (1U << 12U) | (1U << 13U),
        all = (1U << 0U) | (1U << 1U) | (1U << 2U) | (1U << 3U) | (1U << 4U) | (1U << 5U) | (1U << 6U) | (1U << 7U) |
              (1U << 8U) | (1U << 9U) | (1U << 10U) | (1U << 11U) | (1U << 12U) | (1U << 13U),
    };

    [[nodiscard]] constexpr ShaderStage operator|(ShaderStage left, ShaderStage right) noexcept
    {
        return static_cast<ShaderStage>(static_cast<std::underlying_type_t<ShaderStage>>(left) |
                                        static_cast<std::underlying_type_t<ShaderStage>>(right));
    }

    [[nodiscard]] constexpr ShaderStage operator&(ShaderStage left, ShaderStage right) noexcept
    {
        return static_cast<ShaderStage>(static_cast<std::underlying_type_t<ShaderStage>>(left) &
                                        static_cast<std::underlying_type_t<ShaderStage>>(right));
    }

    [[nodiscard]] constexpr bool hasAny(ShaderStage value, ShaderStage flags) noexcept
    {
        return (value & flags) != ShaderStage::none;
    }

    enum class BindingType : std::uint8_t
    {
        uniformBuffer,
        readOnlyStorageBuffer,
        storageBuffer,
        sampledTexture,
        storageTexture,
        sampler,
        dynamicUniformBuffer,
        dynamicReadOnlyStorageBuffer,
        dynamicStorageBuffer,
        accelerationStructure,

        minMipFeedback,

        mipRegionUsedFeedback,
    };

    inline constexpr std::size_t bindingTypeCount = static_cast<std::size_t>(BindingType::mipRegionUsedFeedback) + 1U;

    enum class BindingFlag : std::uint8_t
    {
        none = 0,
        partiallyBound = 1U << 0U,
        variableDescriptorCount = 1U << 1U,
    };

    [[nodiscard]] constexpr BindingFlag operator|(BindingFlag left, BindingFlag right) noexcept
    {
        return static_cast<BindingFlag>(static_cast<std::underlying_type_t<BindingFlag>>(left) |
                                        static_cast<std::underlying_type_t<BindingFlag>>(right));
    }

    [[nodiscard]] constexpr BindingFlag operator&(BindingFlag left, BindingFlag right) noexcept
    {
        return static_cast<BindingFlag>(static_cast<std::underlying_type_t<BindingFlag>>(left) &
                                        static_cast<std::underlying_type_t<BindingFlag>>(right));
    }

    [[nodiscard]] constexpr bool hasAny(BindingFlag value, BindingFlag flags) noexcept
    {
        return (value & flags) != BindingFlag::none;
    }

    inline constexpr std::uint32_t maxDynamicBufferOffsetsPerBindingSet = 8;
    inline constexpr std::uint32_t unboundedBindingCount = std::numeric_limits<std::uint32_t>::max();

    struct BindingLayoutEntry final
    {
        std::uint32_t binding = 0;
        BindingType type = BindingType::uniformBuffer;
        ShaderStage visibility = ShaderStage::all;

        std::uint32_t count = 1;

        BindingFlag flags = BindingFlag::none;
    };

    struct BindingLayoutDesc final
    {
        std::span<const BindingLayoutEntry> entries = {};
        std::string_view label = {};
    };

    struct BindingPoolSize final
    {
        BindingType type = BindingType::uniformBuffer;
        std::uint32_t descriptorCount = 0;
    };

    struct BindingPoolDesc final
    {
        std::uint32_t maxBindingSets = 0;

        std::span<const BindingPoolSize> sizes = {};
        std::string_view label = {};
    };

    struct BufferBinding final
    {
        BufferHandle buffer = {};
        std::uint64_t offset = 0;
        std::uint64_t size = wholeResource;

        std::uint32_t structureByteStride = 0;
    };

    using BindingResource = std::variant<std::monostate,
                                         BufferBinding,
                                         TextureViewHandle,
                                         SamplerHandle,
                                         AccelerationStructureHandle,
                                         SamplerFeedbackMapHandle>;

    struct BindingWrite final
    {
        std::uint32_t binding = 0;
        std::uint32_t arrayElement = 0;
        BindingResource resource = {};
    };

    struct BindingSetDesc final
    {
        BindingLayoutHandle layout = {};

        std::span<const BindingWrite> writes = {};

        std::uint32_t variableDescriptorCount = 0;
        std::string_view label = {};

        BindingPoolHandle pool = {};
    };

    struct BindingUpdateStatistics final
    {
        std::uint64_t resourceDescriptorsPrepared = 0;
        std::uint64_t samplerDescriptorsPrepared = 0;
        std::uint64_t resourceTablePublishCount = 0;
        std::uint64_t samplerTablePublishCount = 0;
        std::uint64_t resourceDescriptorsPublished = 0;
        std::uint64_t samplerDescriptorsPublished = 0;
    };

    static_assert(sizeof(BindingUpdateStatistics) == 6 * sizeof(std::uint64_t));

} // namespace rachis
