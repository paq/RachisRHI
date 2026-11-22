#pragma once

#include "Rachis/Error.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <utility>

namespace rachis::detail
{
    class AcquireDeadline final
    {
    public:
        using Clock = std::chrono::steady_clock;

        AcquireDeadline(std::optional<std::chrono::nanoseconds> timeout, std::chrono::nanoseconds untimedWait) noexcept
            : timed_(timeout.has_value()), untimedWait_(untimedWait)
        {
            if (!timeout.has_value() || *timeout == std::chrono::nanoseconds::max())
            {
                return;
            }
            const auto start = Clock::now();
            const auto budget = std::max(*timeout, std::chrono::nanoseconds::zero());
            const auto headroom =
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::time_point::max() - start);
            deadline_ = budget >= headroom ? Clock::time_point::max()
                                           : start + std::chrono::duration_cast<Clock::duration>(budget);
        }

        [[nodiscard]] bool timed() const noexcept
        {
            return timed_;
        }

        [[nodiscard]] bool unbounded() const noexcept
        {
            return timed_ && !deadline_.has_value();
        }

        [[nodiscard]] std::chrono::nanoseconds remaining() const noexcept
        {
            if (!timed_)
            {
                return untimedWait_;
            }
            if (!deadline_.has_value())
            {
                return std::chrono::nanoseconds::max();
            }
            const auto now = Clock::now();
            if (now >= *deadline_)
            {
                return std::chrono::nanoseconds::zero();
            }
            return std::chrono::duration_cast<std::chrono::nanoseconds>(*deadline_ - now);
        }

        [[nodiscard]] std::uint64_t remainingNs() const noexcept
        {
            const auto left = remaining();
            return left == std::chrono::nanoseconds::max() ? std::numeric_limits<std::uint64_t>::max()
                                                           : static_cast<std::uint64_t>(left.count());
        }

        [[nodiscard]] std::uint64_t remainingMs() const noexcept
        {
            const auto left = remaining();
            return left == std::chrono::nanoseconds::max()
                       ? std::numeric_limits<std::uint64_t>::max()
                       : static_cast<std::uint64_t>(
                             std::chrono::duration_cast<std::chrono::milliseconds>(left).count());
        }

        [[nodiscard]] bool expired() const noexcept
        {
            return timed_ && deadline_.has_value() && Clock::now() >= *deadline_;
        }

        // Locks mutex, giving up at the deadline of a bounded timed acquire so that waiting for a lock never
        // overruns it. Untimed and unbounded acquires block as before. Check owns_lock() on the result.
        template<typename Mutex>
        [[nodiscard]] std::unique_lock<Mutex> lock(Mutex& mutex) const
        {
            if (!timed_ || !deadline_.has_value())
            {
                return std::unique_lock<Mutex>(mutex);
            }
            return std::unique_lock<Mutex>(mutex, *deadline_);
        }

        template<typename Mutex>
        [[nodiscard]] std::shared_lock<Mutex> lockShared(Mutex& mutex) const
        {
            if (!timed_ || !deadline_.has_value())
            {
                return std::shared_lock<Mutex>(mutex);
            }
            return std::shared_lock<Mutex>(mutex, *deadline_);
        }

        // Relocks a lock released during a wait, under the same deadline. Returns false when it timed out.
        template<typename Mutex>
        [[nodiscard]] bool relock(std::unique_lock<Mutex>& held) const
        {
            if (!timed_ || !deadline_.has_value())
            {
                held.lock();
                return true;
            }
            return held.try_lock_until(*deadline_);
        }

        [[nodiscard]] Error lockTimeout() const
        {
            return expiry("Another call held the swapchain past the acquire timeout; no image was acquired.");
        }

        [[nodiscard]] Error expiry(std::string message, std::int64_t nativeCode = 0) const
        {
            return Error{timed_ ? ErrorCode::timeout : ErrorCode::notReady, nativeCode, std::move(message)};
        }

        [[nodiscard]] Error expiry(Error waitError) const
        {
            if (timed_ && waitError.code == ErrorCode::notReady)
            {
                waitError.code = ErrorCode::timeout;
            }
            return waitError;
        }

    private:
        bool timed_ = false;
        std::chrono::nanoseconds untimedWait_{};
        std::optional<Clock::time_point> deadline_;
    };
} // namespace rachis::detail
