#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <set>

namespace rachis::metal
{

    class MetalSubmissionFeedback final
    {
    public:
        void begin(std::uint64_t value)
        {
            const std::scoped_lock lock(mutex_);
            pending_.insert(value);
        }

        void complete(std::uint64_t value)
        {
            {
                const std::scoped_lock lock(mutex_);
                pending_.erase(value);
            }
            changed_.notify_all();
        }

        [[nodiscard]] std::uint64_t completedValue(std::uint64_t signaled) const
        {
            const std::scoped_lock lock(mutex_);
            return pending_.empty() ? signaled : std::min(signaled, *pending_.begin() - 1);
        }

        void wait(std::uint64_t value, std::chrono::milliseconds timeout) const
        {
            std::unique_lock lock(mutex_);
            changed_.wait_for(lock, timeout, [&] { return pending_.empty() || *pending_.begin() > value; });
        }

    private:
        mutable std::mutex mutex_;
        mutable std::condition_variable changed_;
        std::set<std::uint64_t> pending_;
    };

} // namespace rachis::metal
