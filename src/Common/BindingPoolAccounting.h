#pragma once

#include "Rachis/Binding.h"
#include "Rachis/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace rachis::detail
{

    using BindingDescriptorCounts = std::array<std::uint32_t, bindingTypeCount>;

    enum class BindingPoolVersionUpdate : std::uint8_t
    {
        additionalVersion,
        reuseCurrentVersion,
    };

    [[nodiscard]] constexpr BindingPoolVersionUpdate bindingPoolVersionUpdateFor(std::uint64_t activeCommandRetentions,
                                                                                 bool currentVersionArmed) noexcept
    {
        return activeCommandRetentions == 0 && currentVersionArmed ? BindingPoolVersionUpdate::reuseCurrentVersion
                                                                   : BindingPoolVersionUpdate::additionalVersion;
    }

    [[nodiscard]] inline Status checkBindingPoolVersionCapacity(const BindingDescriptorCounts& capacity,
                                                                const BindingDescriptorCounts& live,
                                                                BindingPoolVersionUpdate update,
                                                                const BindingDescriptorCounts& currentCounts,
                                                                const BindingDescriptorCounts& nextCounts)
    {
        for (std::size_t index = 0; index < nextCounts.size(); ++index)
        {
            const auto released =
                update == BindingPoolVersionUpdate::reuseCurrentVersion ? currentCounts[index] : std::uint32_t{0};
            if (nextCounts[index] > capacity[index] - (live[index] - released))
            {
                return std::unexpected(Error{
                    ErrorCode::resourceExhausted,
                    0,
                    "The binding pool has insufficient per-type capacity for another immutable descriptor version.",
                });
            }
        }
        return {};
    }

    inline void applyBindingPoolVersionUpdate(BindingDescriptorCounts& live,
                                              BindingPoolVersionUpdate update,
                                              const BindingDescriptorCounts& currentCounts,
                                              const BindingDescriptorCounts& nextCounts) noexcept
    {
        for (std::size_t index = 0; index < nextCounts.size(); ++index)
        {
            const auto released =
                update == BindingPoolVersionUpdate::reuseCurrentVersion ? currentCounts[index] : std::uint32_t{0};
            live[index] = live[index] - released + nextCounts[index];
        }
    }

    [[nodiscard]] inline Result<BindingPoolVersionUpdate> commitBindingPoolVersionCapacity(
        const BindingDescriptorCounts& capacity,
        BindingDescriptorCounts& live,
        std::uint64_t activeCommandRetentions,
        bool currentVersionArmed,
        const BindingDescriptorCounts& currentCounts,
        const BindingDescriptorCounts& nextCounts)
    {
        const auto update = bindingPoolVersionUpdateFor(activeCommandRetentions, currentVersionArmed);
        if (auto fits = checkBindingPoolVersionCapacity(capacity, live, update, currentCounts, nextCounts); !fits)
        {
            return std::unexpected(std::move(fits.error()));
        }
        applyBindingPoolVersionUpdate(live, update, currentCounts, nextCounts);
        return update;
    }

} // namespace rachis::detail
