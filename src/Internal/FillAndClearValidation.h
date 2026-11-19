#pragma once

#include "Rachis/Error.h"
#include "Rachis/Resource.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace rachis::detail
{

    inline constexpr std::uint64_t bufferFillAlignment = 4;

    [[nodiscard]] inline Status validateBufferFillRequest(std::uint64_t offset, std::uint64_t size)
    {
        const bool alignedOffset = offset % bufferFillAlignment == 0;
        const bool alignedSize = size == wholeResource || (size != 0 && size % bufferFillAlignment == 0);
        const bool overflows = size != wholeResource && offset > std::numeric_limits<std::uint64_t>::max() - size;
        if (!alignedOffset || !alignedSize || overflows)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "fillBuffer requires a 4-byte-aligned offset and a non-zero, 4-byte-aligned size that does not "
                "overflow 64-bit offsets.",
            });
        }
        return {};
    }

    [[nodiscard]] inline Result<std::uint64_t> resolveBufferFillRange(const BufferDesc& desc,
                                                                      std::uint64_t offset,
                                                                      std::uint64_t size)
    {
        if (auto valid = validateBufferFillRequest(offset, size); !valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        if (!hasAny(desc.usage, BufferUsage::storage) || !hasAny(desc.usage, BufferUsage::copyDestination))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "fillBuffer requires a buffer with storage and copy-destination usage.",
            });
        }
        if (offset >= desc.size)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The fillBuffer offset lies outside the buffer.",
            });
        }
        const std::uint64_t resolved = size == wholeResource ? desc.size - offset : size;
        if (resolved > desc.size - offset || resolved % bufferFillAlignment != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The fillBuffer range must lie inside the buffer and be a multiple of 4 bytes.",
            });
        }
        return resolved;
    }

    struct StorageTextureClearFormat final
    {
        StorageTextureClearType type = StorageTextureClearType::float32;
        std::uint32_t channelCount = 0;
        std::uint32_t channelBits = 0;
    };

    [[nodiscard]] constexpr std::optional<StorageTextureClearFormat> storageTextureClearFormat(Format format) noexcept
    {
        using enum StorageTextureClearType;
        switch (format)
        {
        case Format::r8Unorm:
            return StorageTextureClearFormat{float32, 1, 8};
        case Format::rg8Unorm:
            return StorageTextureClearFormat{float32, 2, 8};
        case Format::rgba8Unorm:
        case Format::rgba8UnormSrgb:
        case Format::bgra8Unorm:
        case Format::bgra8UnormSrgb:
            return StorageTextureClearFormat{float32, 4, 8};
        case Format::r16Float:
            return StorageTextureClearFormat{float32, 1, 16};
        case Format::rg16Float:
            return StorageTextureClearFormat{float32, 2, 16};
        case Format::rgba16Float:
            return StorageTextureClearFormat{float32, 4, 16};
        case Format::r32Float:
            return StorageTextureClearFormat{float32, 1, 32};
        case Format::rg32Float:
            return StorageTextureClearFormat{float32, 2, 32};
        case Format::rgba32Float:
            return StorageTextureClearFormat{float32, 4, 32};
        case Format::r8Uint:
            return StorageTextureClearFormat{uint32, 1, 8};
        case Format::rg8Uint:
            return StorageTextureClearFormat{uint32, 2, 8};
        case Format::rgba8Uint:
            return StorageTextureClearFormat{uint32, 4, 8};
        case Format::r16Uint:
            return StorageTextureClearFormat{uint32, 1, 16};
        case Format::rg16Uint:
            return StorageTextureClearFormat{uint32, 2, 16};
        case Format::rgba16Uint:
            return StorageTextureClearFormat{uint32, 4, 16};
        case Format::r32Uint:
            return StorageTextureClearFormat{uint32, 1, 32};
        case Format::rg32Uint:
            return StorageTextureClearFormat{uint32, 2, 32};
        case Format::rgba32Uint:
            return StorageTextureClearFormat{uint32, 4, 32};
        case Format::r8Sint:
            return StorageTextureClearFormat{sint32, 1, 8};
        case Format::rg8Sint:
            return StorageTextureClearFormat{sint32, 2, 8};
        case Format::rgba8Sint:
            return StorageTextureClearFormat{sint32, 4, 8};
        case Format::r16Sint:
            return StorageTextureClearFormat{sint32, 1, 16};
        case Format::rg16Sint:
            return StorageTextureClearFormat{sint32, 2, 16};
        case Format::rgba16Sint:
            return StorageTextureClearFormat{sint32, 4, 16};
        case Format::r32Sint:
            return StorageTextureClearFormat{sint32, 1, 32};
        case Format::rg32Sint:
            return StorageTextureClearFormat{sint32, 2, 32};
        case Format::rgba32Sint:
            return StorageTextureClearFormat{sint32, 4, 32};
        default:
            return std::nullopt;
        }
    }

    [[nodiscard]] inline Status validateStorageTextureClearType(const StorageTextureClearValue& value)
    {
        switch (value.type)
        {
        case StorageTextureClearType::float32:
        case StorageTextureClearType::uint32:
        case StorageTextureClearType::sint32:
            return {};
        }
        return std::unexpected(Error{
            ErrorCode::invalidArgument,
            0,
            "The storage-texture clear value has an unknown type.",
        });
    }

    [[nodiscard]] inline bool storageTextureClearValueFinite(const StorageTextureClearFormat& format,
                                                             const StorageTextureClearValue& value) noexcept
    {
        for (std::uint32_t channel = 0; channel < format.channelCount; ++channel)
        {
            if (format.type == StorageTextureClearType::float32 && !std::isfinite(value.float32[channel]))
            {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] constexpr bool storageTextureClearValueFits(const StorageTextureClearFormat& format,
                                                              const StorageTextureClearValue& value) noexcept
    {
        if (format.channelBits >= 32)
        {
            return true;
        }
        for (std::uint32_t channel = 0; channel < format.channelCount; ++channel)
        {
            if (format.type == StorageTextureClearType::uint32 &&
                value.uint32[channel] > (std::uint32_t{1} << format.channelBits) - 1U)
            {
                return false;
            }
            if (format.type == StorageTextureClearType::sint32)
            {
                const std::int32_t maximum = (std::int32_t{1} << (format.channelBits - 1U)) - 1;
                if (value.sint32[channel] > maximum || value.sint32[channel] < -maximum - 1)
                {
                    return false;
                }
            }
        }
        return true;
    }

    [[nodiscard]] inline Result<StorageTextureClearFormat> validateStorageTextureClear(
        const TextureDesc& texture,
        TextureViewUsage viewUsage,
        Format viewFormat,
        const StorageTextureClearValue& value)
    {
        if (auto valid = validateStorageTextureClearType(value); !valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        if (viewUsage != TextureViewUsage::storage)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "clearStorageTexture requires a storage texture view.",
            });
        }
        if (!hasAny(texture.usage, TextureUsage::storage) || !hasAny(texture.usage, TextureUsage::copyDestination) ||
            texture.sampleCount != 1)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "clearStorageTexture requires a single-sampled texture with storage and copy-destination usage.",
            });
        }
        const Format resolvedFormat = viewFormat == Format::unknown ? texture.format : viewFormat;
        const auto format = storageTextureClearFormat(resolvedFormat);
        if (resolvedFormat != texture.format || !format.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "clearStorageTexture requires a view in the texture's own format, and that format must be an "
                "uncompressed color format with 8-, 16-, or 32-bit channels.",
            });
        }
        if (format->type != value.type)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The storage-texture clear value type does not match the view format: use float32 for float and "
                "normalized formats, uint32 for unsigned-integer formats, and sint32 for signed-integer formats.",
            });
        }
        if (!storageTextureClearValueFinite(*format, value))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A float32 storage-texture clear value is not finite in a channel of the view format.",
            });
        }
        if (!storageTextureClearValueFits(*format, value))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "An integer storage-texture clear value does not fit in the channels of the view format.",
            });
        }
        return *format;
    }

} // namespace rachis::detail
