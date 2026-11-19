#pragma once

#include <cstdint>

namespace rachis::metal
{

    inline constexpr std::uint32_t kMetalClampThreadgroupSize = 64;

    [[nodiscard]] constexpr std::uint32_t metalClampThreadgroups(std::uint32_t records) noexcept
    {
        return records / kMetalClampThreadgroupSize + (records % kMetalClampThreadgroupSize != 0 ? 1U : 0U);
    }

} // namespace rachis::metal
