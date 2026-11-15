#pragma once

#include "Rachis/Query.h"
#include "Rachis/Resource.h"

#include <cstdint>
#include <optional>
#include <span>

namespace rachis
{

    inline constexpr std::uint32_t maxViewportCount = 16;

    enum class LoadOperation : std::uint8_t
    {
        load,
        clear,
        discard,
    };

    enum class StoreOperation : std::uint8_t
    {
        store,
        discard,
    };

    struct ClearColor final
    {
        float red = 0.0F;
        float green = 0.0F;
        float blue = 0.0F;
        float alpha = 0.0F;
    };

    struct BlendConstants final
    {
        float red = 0.0F;
        float green = 0.0F;
        float blue = 0.0F;
        float alpha = 0.0F;
    };

    struct DepthBias final
    {
        float constantFactor = 0.0F;
        float clamp = 0.0F;
        float slopeScale = 0.0F;
    };

    struct DepthBounds final
    {
        float minimum = 0.0F;
        float maximum = 1.0F;
    };

    enum class ShadingRate : std::uint8_t
    {
        oneByOne,
        oneByTwo,
        twoByOne,
        twoByTwo,
        twoByFour,
        fourByTwo,
        fourByFour,
    };

    [[nodiscard]] constexpr std::optional<std::uint8_t> shadingRateImageTexel(ShadingRate rate) noexcept
    {
        switch (rate)
        {
        case ShadingRate::oneByOne:
            return std::uint8_t{0};
        case ShadingRate::oneByTwo:
            return std::uint8_t{1};
        case ShadingRate::twoByOne:
            return std::uint8_t{4};
        case ShadingRate::twoByTwo:
            return std::uint8_t{5};
        case ShadingRate::twoByFour:
            return std::uint8_t{6};
        case ShadingRate::fourByTwo:
            return std::uint8_t{9};
        case ShadingRate::fourByFour:
            return std::uint8_t{10};
        default:
            return std::nullopt;
        }
    }

    enum class ShadingRateCombiner : std::uint8_t
    {
        passthrough,
        override,
        minimum,
        maximum,
        sum,
    };

    struct ShadingRateCombiners final
    {
        ShadingRateCombiner primitive = ShadingRateCombiner::passthrough;

        ShadingRateCombiner image = ShadingRateCombiner::passthrough;
    };

    struct ColorAttachment final
    {
        TextureViewHandle view = {};
        LoadOperation loadOperation = LoadOperation::load;
        StoreOperation storeOperation = StoreOperation::store;
        ClearColor clearValue = {};
    };

    struct DepthStencilAttachment final
    {
        TextureViewHandle view = {};
        LoadOperation depthLoadOperation = LoadOperation::load;
        StoreOperation depthStoreOperation = StoreOperation::store;
        float depthClearValue = 1.0F;
        LoadOperation stencilLoadOperation = LoadOperation::load;
        StoreOperation stencilStoreOperation = StoreOperation::store;
        std::uint32_t stencilClearValue = 0;
    };

    struct RenderingDesc final
    {
        std::span<const ColorAttachment> colorAttachments = {};
        std::optional<DepthStencilAttachment> depthStencilAttachment = {};
        std::optional<PassTimestampWrites> timestampWrites = {};

        std::uint32_t viewMask = 0;
    };

    struct Viewport final
    {
        float x = 0.0F;
        float y = 0.0F;
        float width = 0.0F;
        float height = 0.0F;
        float minDepth = 0.0F;
        float maxDepth = 1.0F;
    };

    struct ScissorRect final
    {
        std::int32_t x = 0;
        std::int32_t y = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
    };

    enum class DepthStencilClearAspect : std::uint8_t
    {
        none = 0,
        depth = 1U << 0U,
        stencil = 1U << 1U,
    };

    [[nodiscard]] constexpr DepthStencilClearAspect operator|(DepthStencilClearAspect left,
                                                              DepthStencilClearAspect right) noexcept
    {
        return static_cast<DepthStencilClearAspect>(static_cast<std::uint8_t>(left) | static_cast<std::uint8_t>(right));
    }

    [[nodiscard]] constexpr bool hasAny(DepthStencilClearAspect value, DepthStencilClearAspect flags) noexcept
    {
        return (static_cast<std::uint8_t>(value) & static_cast<std::uint8_t>(flags)) != 0;
    }

    inline constexpr std::uint64_t indirectArgumentAlignment = sizeof(std::uint32_t);

    struct DispatchIndirectArguments final
    {
        std::uint32_t groupCountX = 0;
        std::uint32_t groupCountY = 0;
        std::uint32_t groupCountZ = 0;
    };

    struct DrawIndirectArguments final
    {
        std::uint32_t vertexCount = 0;
        std::uint32_t instanceCount = 0;
        std::uint32_t firstVertex = 0;
        std::uint32_t firstInstance = 0;
    };

    struct DrawIndexedIndirectArguments final
    {
        std::uint32_t indexCount = 0;
        std::uint32_t instanceCount = 0;
        std::uint32_t firstIndex = 0;
        std::int32_t baseVertex = 0;
        std::uint32_t firstInstance = 0;
    };

    static_assert(sizeof(DispatchIndirectArguments) == 12);
    static_assert(sizeof(DrawIndirectArguments) == 16);
    static_assert(sizeof(DrawIndexedIndirectArguments) == 20);

    struct IndirectDispatchDesc final
    {
        BufferHandle argumentBuffer = {};
        std::uint64_t argumentOffset = 0;
        std::uint32_t maxDispatchCount = 1;
        std::uint32_t stride = 0;
        BufferHandle countBuffer = {};
        std::uint64_t countOffset = 0;
    };

    struct IndirectDrawDesc final
    {
        BufferHandle argumentBuffer = {};
        std::uint64_t argumentOffset = 0;
        std::uint32_t maxDrawCount = 1;
        std::uint32_t stride = 0;
        BufferHandle countBuffer = {};
        std::uint64_t countOffset = 0;
    };

    struct IndirectMeshDispatchDesc final
    {
        BufferHandle argumentBuffer = {};
        std::uint64_t argumentOffset = 0;
        std::uint32_t maxDispatchCount = 1;
        std::uint32_t stride = 0;
        BufferHandle countBuffer = {};
        std::uint64_t countOffset = 0;
    };

} // namespace rachis
