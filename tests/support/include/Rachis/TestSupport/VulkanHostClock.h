#pragma once

#include <volk.h>

#ifdef _WIN32
#include <Windows.h>
#else
#include <ctime>
#endif

#include <cstdint>

namespace rachis::test
{

    struct HostClockReading final
    {
        std::uint64_t timestamp = 0;
        std::uint64_t frequency = 0;
    };

    [[nodiscard]] inline HostClockReading readVulkanHostClock() noexcept
    {
#ifdef _WIN32
        LARGE_INTEGER frequency{};
        LARGE_INTEGER counter{};
        if (QueryPerformanceFrequency(&frequency) == 0 || frequency.QuadPart <= 0 ||
            QueryPerformanceCounter(&counter) == 0)
        {
            return {};
        }
        return HostClockReading{
            .timestamp = static_cast<std::uint64_t>(counter.QuadPart),
            .frequency = static_cast<std::uint64_t>(frequency.QuadPart),
        };
#else
#ifdef __APPLE__
        constexpr clockid_t hostClock = CLOCK_MONOTONIC_RAW;
#else
        constexpr clockid_t hostClock = CLOCK_MONOTONIC;
#endif
        timespec ts{};
        if (clock_gettime(hostClock, &ts) != 0 || ts.tv_sec < 0 || ts.tv_nsec < 0)
        {
            return {};
        }
        return HostClockReading{
            .timestamp =
                static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL + static_cast<std::uint64_t>(ts.tv_nsec),
            .frequency = 1'000'000'000ULL,
        };
#endif
    }

    [[nodiscard]] constexpr VkTimeDomainEXT vulkanHostTimeDomain() noexcept
    {
#ifdef _WIN32
        return VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT;
#elif defined(__APPLE__)
        return VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT;
#else
        return VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT;
#endif
    }

} // namespace rachis::test
