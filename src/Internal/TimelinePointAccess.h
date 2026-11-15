#pragma once

#include "Rachis/Command.h"

namespace rachis::detail
{

    struct TimelinePointAccess final
    {
        [[nodiscard]] static constexpr TimelinePoint make(QueueType queueType,
                                                          std::uint64_t value,
                                                          std::uint64_t deviceToken) noexcept
        {
            return {queueType, value, deviceToken};
        }

        [[nodiscard]] static constexpr std::uint64_t deviceToken(TimelinePoint point) noexcept
        {
            return point.deviceToken_;
        }
    };

} // namespace rachis::detail
