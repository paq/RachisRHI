#pragma once

#include "Rachis/Presentation.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>

namespace rachis::detail
{

    // The outcome of each present of one swapchain generation, by present sequence (1, 2, 3, ...).
    // A backend calls submitted() as a present takes its sequence and settle() or settleBefore() as it
    // learns the outcome. Thread-safe: Metal settles from its presented handlers.
    class PresentTimingHistory final
    {
    public:
        static constexpr std::size_t capacity = 64;
        static_assert(capacity >= presentTimingMinHistory);

        void reset() noexcept
        {
            const std::scoped_lock lock(mutex_);
            entries_ = {};
            submitted_ = 0;
        }

        void submitted(std::uint64_t sequence) noexcept
        {
            const std::scoped_lock lock(mutex_);
            if (sequence <= submitted_)
            {
                return;
            }
            for (auto next = std::max(submitted_ + 1, sequence > capacity ? sequence - capacity + 1 : 1);
                 next <= sequence;
                 ++next)
            {
                entries_[slot(next)] = Entry{.sequence = next};
            }
            submitted_ = sequence;
        }

        void settle(std::uint64_t sequence, std::uint64_t presentTime, PresentTimingFlags flags) noexcept
        {
            const std::scoped_lock lock(mutex_);
            if (auto* entry = find(sequence); entry != nullptr && !entry->settled)
            {
                entry->presentTime = hasAny(flags, PresentTimingFlags::displayed) ? presentTime : 0;
                entry->flags = flags;
                entry->settled = true;
            }
        }

        // Settles every pending present older than sequence with an unknown outcome.
        void settleBefore(std::uint64_t sequence) noexcept
        {
            const std::scoped_lock lock(mutex_);
            for (auto next = firstAvailable(); next != 0 && next < sequence && next <= submitted_; ++next)
            {
                auto& entry = entries_[slot(next)];
                if (!entry.settled)
                {
                    entry.settled = true;
                    entry.flags = PresentTimingFlags::none;
                    entry.presentTime = 0;
                }
            }
        }

        struct Snapshot final
        {
            std::uint64_t submittedSequence = 0;
            std::uint64_t settledSequence = 0;
            std::uint64_t firstAvailableSequence = 0;
            std::uint32_t recordCount = 0;
        };

        [[nodiscard]] Snapshot read(std::uint64_t firstSequence, std::span<PresentTiming> records) const noexcept
        {
            const std::scoped_lock lock(mutex_);
            Snapshot snapshot{.submittedSequence = submitted_, .firstAvailableSequence = firstAvailable()};
            if (snapshot.firstAvailableSequence == 0)
            {
                return snapshot;
            }
            auto settled = snapshot.firstAvailableSequence - 1;
            while (settled < submitted_ && entries_[slot(settled + 1)].settled)
            {
                ++settled;
            }
            snapshot.settledSequence = settled;
            for (auto next = std::max(firstSequence, snapshot.firstAvailableSequence);
                 next <= settled && snapshot.recordCount < records.size();
                 ++next)
            {
                const auto& entry = entries_[slot(next)];
                records[snapshot.recordCount++] = PresentTiming{
                    .presentSequence = next,
                    .presentTime = entry.presentTime,
                    .flags = entry.flags,
                };
            }
            return snapshot;
        }

    private:
        struct Entry final
        {
            std::uint64_t sequence = 0;
            std::uint64_t presentTime = 0;
            PresentTimingFlags flags = PresentTimingFlags::none;
            bool settled = false;
        };

        [[nodiscard]] static std::size_t slot(std::uint64_t sequence) noexcept
        {
            return static_cast<std::size_t>(sequence % capacity);
        }

        [[nodiscard]] std::uint64_t firstAvailable() const noexcept
        {
            if (submitted_ == 0)
            {
                return 0;
            }
            return submitted_ > capacity ? submitted_ - capacity + 1 : 1;
        }

        [[nodiscard]] Entry* find(std::uint64_t sequence) noexcept
        {
            const auto first = firstAvailable();
            if (first == 0 || sequence < first || sequence > submitted_)
            {
                return nullptr;
            }
            auto& entry = entries_[slot(sequence)];
            return entry.sequence == sequence ? &entry : nullptr;
        }

        mutable std::mutex mutex_;
        std::array<Entry, capacity> entries_{};
        std::uint64_t submitted_ = 0;
    };

} // namespace rachis::detail
