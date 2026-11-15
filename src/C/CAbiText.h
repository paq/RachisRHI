#pragma once

#include <cstddef>
#include <cstring>
#include <span>
#include <string_view>

namespace rachis::cabi::detail
{

    inline constexpr std::size_t maxUtf8ContinuationBytes = 3;

    [[nodiscard]] constexpr bool isUtf8Continuation(char value) noexcept
    {
        return (static_cast<unsigned char>(value) & 0xC0U) == 0x80U;
    }

    [[nodiscard]] constexpr std::size_t utf8PrefixLength(std::string_view text, std::size_t limit) noexcept
    {
        if (text.size() <= limit)
        {
            return text.size();
        }
        auto length = limit;
        while (length != 0 && limit - length < maxUtf8ContinuationBytes && isUtf8Continuation(text[length]))
        {
            --length;
        }
        return isUtf8Continuation(text[length]) ? limit : length;
    }

    inline void copyText(std::span<char> destination, std::string_view text) noexcept
    {
        if (destination.empty())
        {
            return;
        }
        const auto length = utf8PrefixLength(text, destination.size() - 1);
        if (length != 0)
        {
            std::memcpy(destination.data(), text.data(), length);
        }
        destination[length] = '\0';
    }

} // namespace rachis::cabi::detail
