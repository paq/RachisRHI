#pragma once

#include "D3D12/D3D12RuntimePolicies.h"
#include "Rachis/Error.h"

#include <Windows.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace rachis::d3d12
{

    [[nodiscard]] inline Error nativeError(ErrorCode code, HRESULT result, std::string_view operation)
    {
        return formatD3D12HresultError(code, static_cast<std::int32_t>(result), operation);
    }

    [[nodiscard]] inline std::wstring wideLabel(std::string_view label)
    {
        if (label.empty() || label.size() > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        {
            return {};
        }
        const int length = MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, label.data(), static_cast<int>(label.size()), nullptr, 0);
        if (length <= 0)
        {
            return {};
        }
        std::wstring result(static_cast<std::size_t>(length), L'\0');
        MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, label.data(), static_cast<int>(label.size()), result.data(), length);
        return result;
    }

    [[nodiscard]] constexpr std::uint32_t mipDimension(std::uint32_t size, std::uint32_t mipLevel) noexcept
    {
        return (std::max)(1U, size >> mipLevel);
    }

} // namespace rachis::d3d12
