#pragma once

#include "Rachis/Diagnostics.h"
#include "Rachis/Error.h"

#include <atomic>
#include <cstdint>

namespace rachis::vulkan::test
{

    [[nodiscard]] inline std::atomic_int32_t& nextPresentResult() noexcept
    {
        static std::atomic_int32_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_int32_t& nextAcquireResult() noexcept
    {
        static std::atomic_int32_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_int32_t& nextCreateSwapchainResult() noexcept
    {
        static std::atomic_int32_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_int32_t& nextPresentFenceWaitResult() noexcept
    {
        static std::atomic_int32_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_int32_t& nextGraphicsPresentationSyncResult() noexcept
    {
        static std::atomic_int32_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_int32_t& nextEndCommandBufferResult() noexcept
    {
        static std::atomic_int32_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_size_t& liveShadingRateViews() noexcept
    {
        static std::atomic_size_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_size_t& liveNativeKeepAlives() noexcept
    {
        static std::atomic_size_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_bool& hideMaintenance5() noexcept
    {
        static std::atomic_bool s_value = false;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_bool& forceAllCommandsBarrierSources() noexcept
    {
        static std::atomic_bool s_value = false;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_bool& forceIsolatedDynamicBufferShaders() noexcept
    {
        static std::atomic_bool s_value = false;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_bool& forceScissorClipping() noexcept
    {
        static std::atomic_bool s_value = false;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_uint64_t& spirvReflectionTypeVisits() noexcept
    {
        static std::atomic_uint64_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] Status injectCreateResult(std::int32_t vkResult);

    [[nodiscard]] inline std::atomic_int32_t& nextSemaphoreCounterValueResult() noexcept
    {
        static std::atomic_int32_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_bool& forceSubmissionRetirementFences() noexcept
    {
        static std::atomic_bool s_value = false;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_int32_t& retirementFenceStatusOverride() noexcept
    {
        static std::atomic_int32_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_uint64_t& initialQueueTimelineValue() noexcept
    {
        static std::atomic_uint64_t s_value = 0;
        return s_value;
    }

} // namespace rachis::vulkan::test

#ifdef RACHIS_TESTING
namespace rachis::vulkan
{

    [[nodiscard]] Status injectVulkanDeviceLoss(DeviceFaultStatus status);

} // namespace rachis::vulkan
#endif
