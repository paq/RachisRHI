#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <optional>

namespace rachis::detail
{

    class MonotonicIdentitySource final
    {
    public:
        MonotonicIdentitySource() noexcept = default;

        explicit MonotonicIdentitySource(std::uint64_t nextIdentity) noexcept : nextIdentity_(nextIdentity)
        {
        }

        MonotonicIdentitySource(const MonotonicIdentitySource&) = delete;
        MonotonicIdentitySource& operator=(const MonotonicIdentitySource&) = delete;
        MonotonicIdentitySource(MonotonicIdentitySource&&) = delete;
        MonotonicIdentitySource& operator=(MonotonicIdentitySource&&) = delete;
        ~MonotonicIdentitySource() = default;

        [[nodiscard]] std::optional<std::uint64_t> acquire() noexcept
        {
            constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
            auto candidate = nextIdentity_.load(std::memory_order_relaxed);
            while (candidate != 0)
            {
                const auto successor = candidate == maximum ? 0 : candidate + 1;
                if (nextIdentity_.compare_exchange_weak(
                        candidate, successor, std::memory_order_relaxed, std::memory_order_relaxed))
                {
                    return candidate;
                }
            }
            return std::nullopt;
        }

    private:
        std::atomic_uint64_t nextIdentity_{1};
    };

} // namespace rachis::detail
