#include "Rachis/TestSupport/PresentInjection.h"

#if RACHIS_ENABLE_D3D12
#include "D3D12/D3D12TestHooks.h"
#endif
#if RACHIS_ENABLE_VULKAN
#include "Vulkan/VulkanTestHooks.h"
#endif
#if RACHIS_ENABLE_METAL
#include "Metal/MetalTestHooks.h"
#endif

namespace rachis::test
{

    std::recursive_mutex& presentInjectionMutex() noexcept
    {
        static std::recursive_mutex s_mutex;
        return s_mutex;
    }

    void setNextPresentResult(std::int32_t result) noexcept
    {
#if RACHIS_ENABLE_D3D12
        d3d12::test::nextPresentResult().store(result, std::memory_order_relaxed);
#endif
#if RACHIS_ENABLE_VULKAN
        vulkan::test::nextPresentResult().store(result, std::memory_order_relaxed);
#endif
#if RACHIS_ENABLE_METAL
        metal::test::nextPresentResult().store(result, std::memory_order_relaxed);
#endif
    }

    ScopedPresentInjection::ScopedPresentInjection(std::int32_t result) noexcept : lock_(presentInjectionMutex())
    {
        setNextPresentResult(result);
    }

    ScopedPresentInjection::~ScopedPresentInjection() noexcept
    {
        setNextPresentResult(0);
    }

} // namespace rachis::test
