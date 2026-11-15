#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>

namespace rachis::test
{

    void setNextPresentResult(std::int32_t result) noexcept;
    [[nodiscard]] std::recursive_mutex& presentInjectionMutex() noexcept;

    class ScopedPresentInjection final
    {
    public:
        explicit ScopedPresentInjection(std::int32_t result) noexcept;
        ~ScopedPresentInjection() noexcept;

        ScopedPresentInjection(const ScopedPresentInjection&) = delete;
        ScopedPresentInjection& operator=(const ScopedPresentInjection&) = delete;
        ScopedPresentInjection(ScopedPresentInjection&&) = delete;
        ScopedPresentInjection& operator=(ScopedPresentInjection&&) = delete;

    private:
        std::unique_lock<std::recursive_mutex> lock_;
    };

    template<typename T>
    class ScopedAtomicTestHook final
    {
    public:
        ScopedAtomicTestHook(std::atomic<T>& target, T injectedValue, T resetValue = T{}) noexcept
            : target_(target), resetValue_(resetValue), lock_(hookMutex())
        {
            target_.store(injectedValue, std::memory_order_release);
        }

        ~ScopedAtomicTestHook() noexcept
        {
            target_.store(resetValue_, std::memory_order_release);
        }

        ScopedAtomicTestHook(const ScopedAtomicTestHook&) = delete;
        ScopedAtomicTestHook& operator=(const ScopedAtomicTestHook&) = delete;
        ScopedAtomicTestHook(ScopedAtomicTestHook&&) = delete;
        ScopedAtomicTestHook& operator=(ScopedAtomicTestHook&&) = delete;

    private:
        [[nodiscard]] static std::recursive_mutex& hookMutex() noexcept
        {
            static std::recursive_mutex s_mutex;
            return s_mutex;
        }

        std::atomic<T>& target_;
        T resetValue_;
        std::unique_lock<std::recursive_mutex> lock_;
    };

} // namespace rachis::test
