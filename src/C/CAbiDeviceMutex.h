#pragma once

#include "Common/TimedMutex.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace rachis::cabi::detail
{

    class CAbiDeviceMutex final
    {
    public:
        CAbiDeviceMutex() = default;
        ~CAbiDeviceMutex() = default;

        CAbiDeviceMutex(const CAbiDeviceMutex&) = delete;
        CAbiDeviceMutex& operator=(const CAbiDeviceMutex&) = delete;
        CAbiDeviceMutex(CAbiDeviceMutex&&) = delete;
        CAbiDeviceMutex& operator=(CAbiDeviceMutex&&) = delete;

        void lock()
        {
            if (!try_lock())
            {
                static_cast<void>(lockContended(false, nullptr));
            }
        }

        template<typename Clock, typename Duration>
        [[nodiscard]] bool try_lock_until(const std::chrono::time_point<Clock, Duration>& deadline)
        {
            if (try_lock())
            {
                return true;
            }
            const auto until = rachis::detail::toSteadyTimePoint(deadline);
            return lockContended(false, &until);
        }

        [[nodiscard]] bool try_lock() noexcept
        {
            std::uint64_t expected = 0;
            return state_.compare_exchange_strong(
                expected, exclusiveHeld, std::memory_order_acquire, std::memory_order_relaxed);
        }

        void unlock() noexcept
        {
            std::uint64_t expected = exclusiveHeld;
            if (!state_.compare_exchange_strong(expected, 0, std::memory_order_release, std::memory_order_relaxed))
            {
                handOff();
            }
        }

        void lock_shared()
        {
            if (!try_lock_shared())
            {
                static_cast<void>(lockContended(true, nullptr));
            }
        }

        template<typename Clock, typename Duration>
        [[nodiscard]] bool try_lock_shared_until(const std::chrono::time_point<Clock, Duration>& deadline)
        {
            if (try_lock_shared())
            {
                return true;
            }
            const auto until = rachis::detail::toSteadyTimePoint(deadline);
            return lockContended(true, &until);
        }

        [[nodiscard]] bool try_lock_shared() noexcept
        {
            std::uint64_t state = 0;
            while ((state & (exclusiveHeld | waitersQueued)) == 0)
            {
                if (state_.compare_exchange_weak(
                        state, state + 1, std::memory_order_acquire, std::memory_order_relaxed))
                {
                    return true;
                }
            }
            return false;
        }

        void unlock_shared() noexcept
        {
            const auto previous = state_.fetch_sub(1, std::memory_order_release);
            if ((previous & waitersQueued) != 0 && (previous & sharedCountMask) == 1)
            {
                handOff();
            }
        }

    private:
        struct Waiter final
        {
            std::condition_variable admittedChanged;
            Waiter* next = nullptr;
            bool shared = false;
            bool admitted = false;
        };

        static constexpr std::uint64_t exclusiveHeld = std::uint64_t{1} << 63;
        static constexpr std::uint64_t waitersQueued = std::uint64_t{1} << 62;
        static constexpr std::uint64_t sharedCountMask = waitersQueued - 1;

        [[nodiscard]] bool lockContended(bool shared, const std::chrono::steady_clock::time_point* deadline)
        {
            Waiter waiter;
            waiter.shared = shared;
            std::unique_lock lock(queueMutex_);
            auto state = state_.load(std::memory_order_relaxed);
            for (;;)
            {
                const bool available = shared ? (state & (exclusiveHeld | waitersQueued)) == 0 : state == 0;
                if (available)
                {
                    if (state_.compare_exchange_weak(state,
                                                     shared ? state + 1 : exclusiveHeld,
                                                     std::memory_order_acquire,
                                                     std::memory_order_relaxed))
                    {
                        return true;
                    }
                }
                else if ((state & waitersQueued) != 0 ||
                         state_.compare_exchange_weak(
                             state, state | waitersQueued, std::memory_order_relaxed, std::memory_order_relaxed))
                {
                    break;
                }
            }

            if (tail_ == nullptr)
            {
                head_ = &waiter;
            }
            else
            {
                tail_->next = &waiter;
            }
            tail_ = &waiter;
            if (deadline == nullptr)
            {
                waiter.admittedChanged.wait(lock, [&] { return waiter.admitted; });
                return true;
            }
            if (rachis::detail::strictWaitUntil(
                    lock, waiter.admittedChanged, *deadline, [&] { return waiter.admitted; }))
            {
                return true;
            }
            withdraw(waiter);
            return false;
        }

        void withdraw(Waiter& waiter) noexcept
        {
            Waiter* previous = nullptr;
            for (Waiter* current = head_; current != &waiter; current = current->next)
            {
                previous = current;
            }
            (previous == nullptr ? head_ : previous->next) = waiter.next;
            if (tail_ == &waiter)
            {
                tail_ = previous;
            }
            if (head_ != nullptr)
            {
                return;
            }
            auto state = state_.load(std::memory_order_relaxed);
            while ((state & (exclusiveHeld | sharedCountMask)) != 0 &&
                   !state_.compare_exchange_weak(
                       state, state & ~waitersQueued, std::memory_order_relaxed, std::memory_order_relaxed))
            {
            }
        }

        void handOff() noexcept
        {
            const std::scoped_lock lock(queueMutex_);
            std::uint64_t next = 0;
            if (head_ != nullptr && !head_->shared)
            {
                admitHead();
                next = exclusiveHeld;
            }
            else
            {
                while (head_ != nullptr && head_->shared)
                {
                    admitHead();
                    ++next;
                }
            }
            if (head_ != nullptr)
            {
                next |= waitersQueued;
            }
            state_.exchange(next, std::memory_order_acq_rel);
        }

        void admitHead() noexcept
        {
            Waiter& waiter = *head_;
            head_ = waiter.next;
            if (head_ == nullptr)
            {
                tail_ = nullptr;
            }
            waiter.admitted = true;
            waiter.admittedChanged.notify_one();
        }

        std::atomic<std::uint64_t> state_{0};
        std::mutex queueMutex_;
        Waiter* head_ = nullptr;
        Waiter* tail_ = nullptr;
    };

} // namespace rachis::cabi::detail
