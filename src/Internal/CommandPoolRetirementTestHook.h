#pragma once

#ifdef RACHIS_TESTING

#include <atomic>
#include <chrono>
#include <thread>
#include <utility>

namespace rachis::detail::test
{

    using CommandPoolRetirementObserver = void (*)() noexcept;

    [[nodiscard]] inline std::atomic<CommandPoolRetirementObserver>& commandPoolRetirementObserver() noexcept
    {
        static std::atomic<CommandPoolRetirementObserver> s_value = nullptr;
        return s_value;
    }

    inline void observeCommandPoolRetirement() noexcept
    {
        const auto observer = commandPoolRetirementObserver().load(std::memory_order_acquire);
        if (observer != nullptr)
        {
            observer();
        }
    }

    class CommandPoolRetirementParking final
    {
    public:
        CommandPoolRetirementParking() noexcept
        {
            state().store(idle, std::memory_order_release);
            commandPoolRetirementObserver().store(&park, std::memory_order_release);
        }

        ~CommandPoolRetirementParking()
        {
            commandPoolRetirementObserver().store(nullptr, std::memory_order_release);
            release();
        }

        CommandPoolRetirementParking(const CommandPoolRetirementParking&) = delete;
        CommandPoolRetirementParking& operator=(const CommandPoolRetirementParking&) = delete;
        CommandPoolRetirementParking(CommandPoolRetirementParking&&) = delete;
        CommandPoolRetirementParking& operator=(CommandPoolRetirementParking&&) = delete;

        static void armCurrentThread() noexcept
        {
            armedThread() = true;
        }

        [[nodiscard]] static bool waitUntilParked(std::chrono::milliseconds timeout) noexcept
        {
            const auto deadline = std::chrono::steady_clock::now() + timeout;
            while (state().load(std::memory_order_acquire) != parked)
            {
                if (std::chrono::steady_clock::now() >= deadline)
                {
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return true;
        }

        static void release() noexcept
        {
            state().store(released, std::memory_order_release);
            state().notify_all();
        }

    private:
        static constexpr int idle = 0;
        static constexpr int parked = 1;
        static constexpr int released = 2;

        [[nodiscard]] static std::atomic_int& state() noexcept
        {
            static std::atomic_int s_value = idle;
            return s_value;
        }

        [[nodiscard]] static bool& armedThread() noexcept
        {
            thread_local bool s_armed = false;
            return s_armed;
        }

        static void park() noexcept
        {
            if (!std::exchange(armedThread(), false))
            {
                return;
            }
            auto expected = idle;
            if (!state().compare_exchange_strong(expected, parked, std::memory_order_acq_rel))
            {
                return;
            }
            state().notify_all();
            state().wait(parked, std::memory_order_acquire);
        }
    };

} // namespace rachis::detail::test

#endif
