#pragma once

#include "Rachis/Error.h"
#include "Rachis/Pipeline.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

namespace rachis::detail
{

    [[nodiscard]] constexpr std::uint32_t vertexFormatSize(VertexFormat format) noexcept
    {
        switch (format)
        {
        case VertexFormat::uint8x2:
        case VertexFormat::sint8x2:
        case VertexFormat::unorm8x2:
        case VertexFormat::snorm8x2:
            return 2;
        case VertexFormat::float32:
        case VertexFormat::uint32:
        case VertexFormat::sint32:
        case VertexFormat::uint8x4:
        case VertexFormat::sint8x4:
        case VertexFormat::unorm8x4:
        case VertexFormat::snorm8x4:
        case VertexFormat::uint16x2:
        case VertexFormat::sint16x2:
        case VertexFormat::unorm16x2:
        case VertexFormat::snorm16x2:
        case VertexFormat::float16x2:
            return 4;
        case VertexFormat::float32x2:
        case VertexFormat::uint32x2:
        case VertexFormat::sint32x2:
        case VertexFormat::uint16x4:
        case VertexFormat::sint16x4:
        case VertexFormat::unorm16x4:
        case VertexFormat::snorm16x4:
        case VertexFormat::float16x4:
            return 8;
        case VertexFormat::float32x3:
        case VertexFormat::uint32x3:
        case VertexFormat::sint32x3:
            return 12;
        case VertexFormat::float32x4:
        case VertexFormat::uint32x4:
        case VertexFormat::sint32x4:
            return 16;
        }
        return 0;
    }

    [[nodiscard]] constexpr std::uint32_t vertexFormatAlignment(VertexFormat format) noexcept
    {
        return std::clamp(vertexFormatSize(format), 1U, 4U);
    }

    [[nodiscard]] constexpr std::uint32_t vertexAttributesAlignment(
        std::span<const VertexAttribute> attributes) noexcept
    {
        std::uint32_t alignment = 1;
        for (const auto& attribute : attributes)
        {
            alignment = (std::max)(alignment, vertexFormatAlignment(attribute.format));
        }
        return alignment;
    }

    [[nodiscard]] inline Status validateVertexInputAlignment(std::span<const VertexBufferLayout> vertexBuffers)
    {
        for (const auto& buffer : vertexBuffers)
        {
            const bool misalignedAttribute =
                std::ranges::any_of(buffer.attributes, [](const VertexAttribute& attribute) {
                    return attribute.offset % vertexFormatAlignment(attribute.format) != 0;
                });
            if (misalignedAttribute || buffer.stride % vertexAttributesAlignment(buffer.attributes) != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Vertex attribute offsets and buffer strides must be multiples of each attribute's format "
                    "alignment, the smaller of 4 and the format size.",
                });
            }
        }
        return {};
    }

    [[nodiscard]] inline Status validateVertexBufferOffsetAlignment(std::span<const std::uint32_t> slotAlignments,
                                                                    std::uint32_t firstSlot,
                                                                    std::span<const VertexBufferBinding> bindings)
    {
        for (std::size_t index = 0; index < bindings.size(); ++index)
        {
            const auto slot = static_cast<std::size_t>(firstSlot) + index;
            if (slot < slotAlignments.size() && bindings[index].offset % slotAlignments[slot] != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Vertex-buffer offsets must be multiples of the bound pipeline's alignment for their slot.",
                });
            }
        }
        return {};
    }

} // namespace rachis::detail
