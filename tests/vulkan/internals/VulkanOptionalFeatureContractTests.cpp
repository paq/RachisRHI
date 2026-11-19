#include "Internal/VulkanLinuxProfile.h"
#include "Internal/VulkanMacosProfile.h"
#include "Internal/VulkanWindowsProfile.h"

#include <catch2/catch_test_macros.hpp>

#ifdef _WINDOWS_
#error Public Rachis headers and pure profile headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers and pure profile headers must not include Vulkan headers.
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if RACHIS_ENABLE_VULKAN
#include "Internal/VulkanAcceleration.h"
#include "Internal/VulkanBindings.h"
#include "Internal/VulkanCommands.h"
#include "Internal/VulkanDevice.h"
#include "Internal/VulkanDiagnostics.h"
#include "Internal/VulkanGraphics.h"
#include "Internal/VulkanPipelines.h"
#include "Internal/VulkanQueues.h"
#include "Internal/VulkanResources.h"
#include "Internal/VulkanRuntime.h"
#include "Internal/VulkanShaders.h"
#include "Internal/VulkanSwapchain.h"
#include "Vulkan/VulkanTestHooks.h"
#endif

namespace
{

    [[nodiscard]] rachis::VulkanPhysicalDeviceReport baselineProfileDevice()
    {
        return rachis::VulkanPhysicalDeviceReport{
            .apiMajor = 1,
            .apiMinor = 3,
            .deviceName = "Capability Contract Adapter",
            .driverId = "TEST_DRIVER",
            .dynamicRendering = true,
            .synchronization2 = true,
            .timelineSemaphore = true,
            .independentBlend = true,
            .samplerAnisotropy = true,
            .textureCompressionBC = true,
            .shaderSampledImageArrayDynamicIndexing = true,
            .timestampComputeAndGraphics = true,
            .hostQueryReset = true,
            .khrSwapchain = true,
            .khrPortabilitySubset = true,
            .formatRgba8Unorm = true,
            .formatRgba8UnormSrgb = true,
            .formatBgra8Unorm = true,
            .formatBgra8UnormSrgb = true,
            .formatD32Float = true,
            .formatD32FloatS8Uint = true,
            .formatRgba16Float = true,
            .formatR32Uint = true,
            .maxImageDimension2D = 16384,
            .maxBoundDescriptorSets = 8,
            .maxPushConstantsSize = 128,
            .maxColorAttachments = 8,
            .maxComputeWorkgroupInvocations = 1024,
        };
    }

} // namespace

TEST_CASE("Vulkan platform profiles gate presentWait on both VK_KHR_present_id and VK_KHR_present_wait",
          "[size:small][rhi][vulkan][profile][present-wait]")
{
    const rachis::VulkanInstanceReport winInstance{
        .apiMajor = 1,
        .apiMinor = 3,
        .khrSurface = true,
        .khrWin32Surface = true,
    };
    const rachis::VulkanInstanceReport linuxInstance{
        .apiMajor = 1,
        .apiMinor = 3,
        .khrSurface = true,
        .khrWaylandSurface = true,
    };
    const rachis::VulkanInstanceReport macosInstance{
        .apiMajor = 1,
        .apiMinor = 3,
        .khrPortabilityEnumeration = true,
    };

    {
        const auto device = baselineProfileDevice();
        const auto win = rachis::evaluateWindowsVulkanProfile(winInstance, device);
        const auto lin = rachis::evaluateLinuxVulkanProfile(linuxInstance, device);
        const auto mac = rachis::evaluateMacosVulkanProfile(macosInstance, device);
        REQUIRE(win.accepted);
        REQUIRE(lin.accepted);
        REQUIRE(mac.accepted);
        CHECK(win.presentCompletion == rachis::VulkanPresentCompletionStrategy::imageLifetimeAssociation);
        CHECK(lin.presentCompletion == rachis::VulkanPresentCompletionStrategy::imageLifetimeAssociation);
        CHECK(mac.presentCompletion == rachis::VulkanPresentCompletionStrategy::imageLifetimeAssociation);
    }

    {
        auto missingPresentWait = baselineProfileDevice();
        missingPresentWait.khrPresentId = true;
        missingPresentWait.khrPresentWait = false;
        CHECK(rachis::evaluateWindowsVulkanProfile(winInstance, missingPresentWait).presentCompletion ==
              rachis::VulkanPresentCompletionStrategy::imageLifetimeAssociation);
        CHECK(rachis::evaluateLinuxVulkanProfile(linuxInstance, missingPresentWait).presentCompletion ==
              rachis::VulkanPresentCompletionStrategy::imageLifetimeAssociation);
        CHECK(rachis::evaluateMacosVulkanProfile(macosInstance, missingPresentWait).presentCompletion ==
              rachis::VulkanPresentCompletionStrategy::imageLifetimeAssociation);

        auto missingPresentId = baselineProfileDevice();
        missingPresentId.khrPresentId = false;
        missingPresentId.khrPresentWait = true;
        CHECK(rachis::evaluateWindowsVulkanProfile(winInstance, missingPresentId).presentCompletion ==
              rachis::VulkanPresentCompletionStrategy::imageLifetimeAssociation);
        CHECK(rachis::evaluateLinuxVulkanProfile(linuxInstance, missingPresentId).presentCompletion ==
              rachis::VulkanPresentCompletionStrategy::imageLifetimeAssociation);
        CHECK(rachis::evaluateMacosVulkanProfile(macosInstance, missingPresentId).presentCompletion ==
              rachis::VulkanPresentCompletionStrategy::imageLifetimeAssociation);
    }

    {
        auto fullPresentWait = baselineProfileDevice();
        fullPresentWait.khrPresentId = true;
        fullPresentWait.khrPresentWait = true;

        const auto win = rachis::evaluateWindowsVulkanProfile(winInstance, fullPresentWait);
        const auto lin = rachis::evaluateLinuxVulkanProfile(linuxInstance, fullPresentWait);
        const auto mac = rachis::evaluateMacosVulkanProfile(macosInstance, fullPresentWait);
        REQUIRE(win.accepted);
        REQUIRE(lin.accepted);
        REQUIRE(mac.accepted);
        CHECK(win.presentCompletion == rachis::VulkanPresentCompletionStrategy::presentWait);
        CHECK(lin.presentCompletion == rachis::VulkanPresentCompletionStrategy::presentWait);
        CHECK(mac.presentCompletion == rachis::VulkanPresentCompletionStrategy::presentWait);
    }

    {
        auto bothStrategies = baselineProfileDevice();
        bothStrategies.khrPresentId = true;
        bothStrategies.khrPresentWait = true;
        bothStrategies.khrSwapchainMaintenance1 = true;

        const rachis::VulkanInstanceReport winMaintenanceInstance{
            .apiMajor = 1,
            .apiMinor = 3,
            .khrSurface = true,
            .khrWin32Surface = true,
            .khrGetSurfaceCapabilities2 = true,
            .extSurfaceMaintenance1 = true,
        };
        const auto win = rachis::evaluateWindowsVulkanProfile(winMaintenanceInstance, bothStrategies);
        REQUIRE(win.accepted);
        CHECK(win.presentCompletion == rachis::VulkanPresentCompletionStrategy::swapchainMaintenancePresentFence);
    }
}

#if RACHIS_ENABLE_VULKAN
namespace
{

    template<typename Handle>
    Handle fakeVk(std::uintptr_t value)
    {
        return reinterpret_cast<Handle>(value);
    }

    enum class PresentLifecycleOp : std::uint8_t
    {
        present,
        waitForPresent,
        presentQueueIdle,
        deviceIdle,
        destroySemaphore,
        destroySwapchain,
    };

    struct PresentMockBatch final
    {
        VkQueue queue = VK_NULL_HANDLE;
        std::vector<VkSemaphore> binaryWaits;
        std::vector<std::pair<VkSemaphore, std::uint64_t>> timelineSignals;
    };

    struct PresentMockState final
    {
        std::uintptr_t nextHandle = 0x2000;
        std::uint64_t timelineValue = 0;
        std::uint32_t nextAcquiredIndex = 0;
        std::uint32_t outOfRangeAcquires = 0;
        std::vector<std::uint64_t> presentedIds;
        std::vector<std::uint64_t> waitedPresentIds;
        std::vector<std::uint64_t> waitedTimeouts;
        std::vector<PresentLifecycleOp> ops;
        VkResult waitForPresentResult = VK_SUCCESS;
        bool deferCompletion = false;
        std::vector<PresentMockBatch> batches;
        std::map<VkSemaphore, std::uint64_t> completedTimelines;
        std::map<VkQueue, std::size_t> idleBatches;
        int acquiresWithPendingWait = 0;
        int acquireCalls = 0;
        bool stallTimelines = false;
    };

    thread_local PresentMockState* gPresentMock = nullptr;

    void presentMockCompleteTimeline(VkSemaphore semaphore, std::uint64_t value)
    {
        auto& completed = gPresentMock->completedTimelines[semaphore];
        completed = std::max(completed, value);
    }

    void presentMockIdleQueue(VkQueue queue)
    {
        gPresentMock->idleBatches[queue] = gPresentMock->batches.size();
        for (const auto& batch : gPresentMock->batches)
        {
            if (batch.queue != queue)
            {
                continue;
            }
            for (const auto& [semaphore, value] : batch.timelineSignals)
            {
                presentMockCompleteTimeline(semaphore, value);
            }
        }
    }

    [[nodiscard]] bool presentMockBatchComplete(std::size_t index)
    {
        const auto& batch = gPresentMock->batches[index];
        const auto idle = gPresentMock->idleBatches.find(batch.queue);
        if (idle != gPresentMock->idleBatches.end() && index < idle->second)
        {
            return true;
        }
        for (std::size_t later = index; later < gPresentMock->batches.size(); ++later)
        {
            const auto& candidate = gPresentMock->batches[later];
            if (candidate.queue != batch.queue)
            {
                continue;
            }
            for (const auto& [semaphore, value] : candidate.timelineSignals)
            {
                const auto completed = gPresentMock->completedTimelines.find(semaphore);
                if (completed != gPresentMock->completedTimelines.end() && completed->second >= value)
                {
                    return true;
                }
            }
        }
        return false;
    }

    [[nodiscard]] bool presentMockHasPendingWait(VkSemaphore semaphore)
    {
        for (std::size_t index = 0; index < gPresentMock->batches.size(); ++index)
        {
            if (std::ranges::find(gPresentMock->batches[index].binaryWaits, semaphore) !=
                    gPresentMock->batches[index].binaryWaits.end() &&
                !presentMockBatchComplete(index))
            {
                return true;
            }
        }
        return false;
    }

    VKAPI_ATTR VkResult VKAPI_CALL presentMockCreateSemaphore(VkDevice,
                                                              const VkSemaphoreCreateInfo*,
                                                              const VkAllocationCallbacks*,
                                                              VkSemaphore* out)
    {
        *out = fakeVk<VkSemaphore>(gPresentMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL presentMockDestroySemaphore(VkDevice, VkSemaphore, const VkAllocationCallbacks*)
    {
        gPresentMock->ops.push_back(PresentLifecycleOp::destroySemaphore);
    }

    VKAPI_ATTR VkResult VKAPI_CALL presentMockGetCounter(VkDevice, VkSemaphore semaphore, std::uint64_t* value)
    {
        *value =
            gPresentMock->deferCompletion ? gPresentMock->completedTimelines[semaphore] : gPresentMock->timelineValue;
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL presentMockWaitSemaphores(VkDevice, const VkSemaphoreWaitInfo* info, std::uint64_t)
    {
        if (gPresentMock->stallTimelines)
        {
            return VK_TIMEOUT;
        }
        for (std::uint32_t index = 0; index < info->semaphoreCount; ++index)
        {
            presentMockCompleteTimeline(info->pSemaphores[index], info->pValues[index]);
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL presentMockCreateCommandPool(VkDevice,
                                                                const VkCommandPoolCreateInfo*,
                                                                const VkAllocationCallbacks*,
                                                                VkCommandPool* out)
    {
        *out = fakeVk<VkCommandPool>(gPresentMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL presentMockDestroyCommandPool(VkDevice, VkCommandPool, const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL presentMockAllocateCommandBuffers(VkDevice,
                                                                     const VkCommandBufferAllocateInfo* info,
                                                                     VkCommandBuffer* out)
    {
        for (std::uint32_t i = 0; i < info->commandBufferCount; ++i)
        {
            out[i] = fakeVk<VkCommandBuffer>(gPresentMock->nextHandle++);
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL presentMockFreeCommandBuffers(VkDevice,
                                                             VkCommandPool,
                                                             std::uint32_t,
                                                             const VkCommandBuffer*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL presentMockBeginCommandBuffer(VkCommandBuffer, const VkCommandBufferBeginInfo*)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL presentMockEndCommandBuffer(VkCommandBuffer)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL presentMockResetCommandBuffer(VkCommandBuffer, VkCommandBufferResetFlags)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL presentMockResetCommandPool(VkDevice, VkCommandPool, VkCommandPoolResetFlags)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL presentMockQueueSubmit2(VkQueue queue,
                                                           std::uint32_t count,
                                                           const VkSubmitInfo2* submits,
                                                           VkFence)
    {
        for (std::uint32_t s = 0; s < count; ++s)
        {
            PresentMockBatch batch{.queue = queue, .binaryWaits = {}, .timelineSignals = {}};
            for (std::uint32_t wait = 0; wait < submits[s].waitSemaphoreInfoCount; ++wait)
            {
                if (submits[s].pWaitSemaphoreInfos[wait].value == 0)
                {
                    batch.binaryWaits.push_back(submits[s].pWaitSemaphoreInfos[wait].semaphore);
                }
            }
            for (std::uint32_t sig = 0; sig < submits[s].signalSemaphoreInfoCount; ++sig)
            {
                const auto& signal = submits[s].pSignalSemaphoreInfos[sig];
                gPresentMock->timelineValue = std::max(gPresentMock->timelineValue, signal.value);
                if (signal.value != 0)
                {
                    batch.timelineSignals.emplace_back(signal.semaphore, signal.value);
                }
            }
            gPresentMock->batches.push_back(std::move(batch));
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL presentMockAcquireNextImage(
        VkDevice, VkSwapchainKHR, std::uint64_t, VkSemaphore semaphore, VkFence, std::uint32_t* imageIndex)
    {
        ++gPresentMock->acquireCalls;
        if (gPresentMock->deferCompletion && presentMockHasPendingWait(semaphore))
        {
            ++gPresentMock->acquiresWithPendingWait;
        }
        if (gPresentMock->outOfRangeAcquires != 0)
        {
            --gPresentMock->outOfRangeAcquires;
            *imageIndex = 7;
            return VK_SUCCESS;
        }
        *imageIndex = gPresentMock->nextAcquiredIndex;
        gPresentMock->nextAcquiredIndex = (gPresentMock->nextAcquiredIndex + 1U) % 2U;
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL presentMockQueuePresent(VkQueue, const VkPresentInfoKHR* presentInfo)
    {
        gPresentMock->ops.push_back(PresentLifecycleOp::present);
        const auto* cursor = static_cast<const VkBaseInStructure*>(presentInfo->pNext);
        while (cursor != nullptr)
        {
            if (cursor->sType == VK_STRUCTURE_TYPE_PRESENT_ID_KHR)
            {
                const auto* idInfo = reinterpret_cast<const VkPresentIdKHR*>(cursor);
                if (idInfo->swapchainCount == 1 && idInfo->pPresentIds != nullptr)
                {
                    gPresentMock->presentedIds.push_back(idInfo->pPresentIds[0]);
                }
            }
            cursor = cursor->pNext;
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL presentMockWaitForPresent(VkDevice,
                                                             VkSwapchainKHR,
                                                             std::uint64_t presentId,
                                                             std::uint64_t timeout)
    {
        gPresentMock->ops.push_back(PresentLifecycleOp::waitForPresent);
        gPresentMock->waitedPresentIds.push_back(presentId);
        gPresentMock->waitedTimeouts.push_back(timeout);
        return gPresentMock->waitForPresentResult;
    }

    VKAPI_ATTR VkResult VKAPI_CALL presentMockQueueWaitIdle(VkQueue queue)
    {
        if (queue == fakeVk<VkQueue>(0x302))
        {
            gPresentMock->ops.push_back(PresentLifecycleOp::presentQueueIdle);
        }
        presentMockIdleQueue(queue);
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL presentMockDeviceWaitIdle(VkDevice)
    {
        gPresentMock->ops.push_back(PresentLifecycleOp::deviceIdle);
        presentMockIdleQueue(fakeVk<VkQueue>(0x301));
        presentMockIdleQueue(fakeVk<VkQueue>(0x302));
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL presentMockDestroySwapchain(VkDevice, VkSwapchainKHR, const VkAllocationCallbacks*)
    {
        gPresentMock->ops.push_back(PresentLifecycleOp::destroySwapchain);
    }

    VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL presentMockGetDeviceProcAddr(VkDevice, const char*)
    {
        return nullptr;
    }

} // namespace

namespace rachis::vulkan
{

    struct VulkanCapabilitiesSwapchainTestAccess final
    {
        VulkanInstanceTable instanceTable{};
        VolkDeviceTable table{};
        VulkanResourceAllocator resources;
        VulkanCommandAllocator commands;
        VulkanQueueManager queues;
        VulkanSwapchainManager manager;
        SwapchainHandle handle{};
        CommandPoolHandle pool{};

        explicit VulkanCapabilitiesSwapchainTestAccess(VulkanPresentCompletionStrategy strategy)
        {
            instanceTable.vkGetDeviceProcAddr = &presentMockGetDeviceProcAddr;
            table.vkCreateSemaphore = &presentMockCreateSemaphore;
            table.vkDestroySemaphore = &presentMockDestroySemaphore;
            table.vkGetSemaphoreCounterValue = &presentMockGetCounter;
            table.vkWaitSemaphores = &presentMockWaitSemaphores;
            table.vkCreateCommandPool = &presentMockCreateCommandPool;
            table.vkDestroyCommandPool = &presentMockDestroyCommandPool;
            table.vkAllocateCommandBuffers = &presentMockAllocateCommandBuffers;
            table.vkFreeCommandBuffers = &presentMockFreeCommandBuffers;
            table.vkBeginCommandBuffer = &presentMockBeginCommandBuffer;
            table.vkEndCommandBuffer = &presentMockEndCommandBuffer;
            table.vkResetCommandBuffer = &presentMockResetCommandBuffer;
            table.vkResetCommandPool = &presentMockResetCommandPool;
            table.vkQueueSubmit2 = &presentMockQueueSubmit2;
            table.vkAcquireNextImageKHR = &presentMockAcquireNextImage;
            table.vkQueuePresentKHR = &presentMockQueuePresent;
            table.vkWaitForPresentKHR = &presentMockWaitForPresent;
            table.vkQueueWaitIdle = &presentMockQueueWaitIdle;
            table.vkDeviceWaitIdle = &presentMockDeviceWaitIdle;
            table.vkDestroySwapchainKHR = &presentMockDestroySwapchain;

            const auto device = fakeVk<VkDevice>(0x100);
            const VulkanQueueHandles nativeQueues{
                .graphics = fakeVk<VkQueue>(0x301),
                .compute = fakeVk<VkQueue>(0x301),
                .copy = fakeVk<VkQueue>(0x301),
                .present = fakeVk<VkQueue>(0x302),
                .families = {.graphics = 0, .compute = 0, .transfer = 0},
                .presentCompletion = strategy,
            };
            constexpr std::uint64_t kToken = 0x55AA;
            resources.initializeDispatchForTesting(
                device, &table, &instanceTable, fakeVk<VkPhysicalDevice>(0x99), nativeQueues.families);
            commands.initialize(device, &table, nativeQueues, kToken, &resources, &instanceTable);
            REQUIRE(queues.initialize(device, &table, nativeQueues, kToken, false, &instanceTable).has_value());
            const auto createdPool =
                commands.createCommandPool({.queueType = QueueType::graphics, .maxCommandBuffers = 16});
            REQUIRE(createdPool.has_value());
            pool = *createdPool;

            manager.initialize(fakeVk<VkInstance>(0x98),
                               &instanceTable,
                               fakeVk<VkPhysicalDevice>(0x99),
                               device,
                               &table,
                               &resources,
                               &queues,
                               0,
                               nativeQueues.graphics,
                               strategy,
                               false);

            auto record = std::make_shared<VulkanSwapchainManager::SwapchainRecord>();
            record->surface = std::make_shared<VulkanSwapchainManager::SurfaceRecord>();
            record->surface->surface = fakeVk<VkSurfaceKHR>(0x600);
            record->surface->kind = SurfaceKind::caMetalLayer;
            record->swapchain = fakeVk<VkSwapchainKHR>(0x400);
            record->desc = {
                .width = 64,
                .height = 64,
                .imageCount = 2,
                .format = Format::rgba8Unorm,
                .usage = TextureUsage::renderTarget,
            };
            record->images.resize(2);
            record->images[0].image = fakeVk<VkImage>(0x501);
            record->images[1].image = fakeVk<VkImage>(0x502);
            REQUIRE(manager.createPresentationSync(*record).has_value());
            const auto published = manager.swapchains_.emplace(std::move(record));
            REQUIRE(published.has_value());
            handle = *published;
        }

        ~VulkanCapabilitiesSwapchainTestAccess()
        {
            manager.shutdown();
            commands.shutdown();
            queues.shutdown();
            resources.shutdown();
        }

        VulkanCapabilitiesSwapchainTestAccess(const VulkanCapabilitiesSwapchainTestAccess&) = delete;
        VulkanCapabilitiesSwapchainTestAccess& operator=(const VulkanCapabilitiesSwapchainTestAccess&) = delete;
        VulkanCapabilitiesSwapchainTestAccess(VulkanCapabilitiesSwapchainTestAccess&&) = delete;
        VulkanCapabilitiesSwapchainTestAccess& operator=(VulkanCapabilitiesSwapchainTestAccess&&) = delete;

        [[nodiscard]] TimelinePoint submitEmptyGraphicsWork()
        {
            auto cb = commands.acquireCommandBuffer(pool);
            REQUIRE(cb.has_value());
            std::array batch{std::move(*cb)};
            const auto point = queues.submit(batch, {});
            REQUIRE(point.has_value());
            return *point;
        }

        [[nodiscard]] TimelinePoint submitEmptyComputeWork()
        {
            if (!computePool.isValid())
            {
                const auto created =
                    commands.createCommandPool({.queueType = QueueType::compute, .maxCommandBuffers = 4});
                REQUIRE(created.has_value());
                computePool = *created;
            }
            auto cb = commands.acquireCommandBuffer(computePool);
            REQUIRE(cb.has_value());
            std::array batch{std::move(*cb)};
            const auto point = queues.submit(batch, {});
            REQUIRE(point.has_value());
            return *point;
        }

        [[nodiscard]] VulkanPresentationSyncSnapshot snapshot() const
        {
            const auto taken = manager.presentationSyncSnapshot(handle);
            REQUIRE(taken.has_value());
            return *taken;
        }

        [[nodiscard]] std::uint32_t inFlightPresents() const
        {
            const auto timing = manager.queryFrameTiming(handle);
            REQUIRE(timing.has_value());
            return timing->inFlightPresents;
        }

        void presentFrame()
        {
            REQUIRE(manager.acquireNextImage(handle).has_value());
            const auto point = submitEmptyGraphicsWork();
            bool injectedLoss = false;
            const auto presented = manager.present(handle, point, injectedLoss);
            REQUIRE(presented.has_value());
            CHECK(*presented == PresentStatus::presented);
        }

        CommandPoolHandle computePool{};
    };

} // namespace rachis::vulkan

TEST_CASE(
    "VulkanSwapchainManager chains VkPresentIdKHR, retires via vkWaitForPresentKHR, and idles fallback semaphores",
    "[size:small][rhi][vulkan][present-wait]")
{
    SECTION("presentWait strategy chains monotonic VkPresentIdKHR and calls vkWaitForPresentKHR before semaphore "
            "destruction")
    {
        PresentMockState mock;
        gPresentMock = &mock;
        {
            rachis::vulkan::VulkanCapabilitiesSwapchainTestAccess fixture(
                rachis::VulkanPresentCompletionStrategy::presentWait);

            REQUIRE(fixture.manager.acquireNextImage(fixture.handle).has_value());
            const auto point1 = fixture.submitEmptyGraphicsWork();
            bool injectedLoss = false;
            const auto p1 = fixture.manager.present(fixture.handle, point1, injectedLoss);
            REQUIRE(p1.has_value());
            CHECK(*p1 == rachis::PresentStatus::presented);

            REQUIRE(fixture.manager.acquireNextImage(fixture.handle).has_value());
            const auto point2 = fixture.submitEmptyGraphicsWork();
            const auto p2 = fixture.manager.present(fixture.handle, point2, injectedLoss);
            REQUIRE(p2.has_value());
            CHECK(*p2 == rachis::PresentStatus::presented);

            REQUIRE(mock.presentedIds == std::vector<std::uint64_t>{1, 2});
            mock.ops.clear();

            REQUIRE(fixture.manager.destroySwapchain(fixture.handle).has_value());
        }
        gPresentMock = nullptr;

        REQUIRE(mock.waitedPresentIds == std::vector<std::uint64_t>{2});
        REQUIRE(mock.waitedTimeouts == std::vector<std::uint64_t>{UINT64_MAX});

        const auto waitIt = std::ranges::find(mock.ops, PresentLifecycleOp::waitForPresent);
        const auto destroySemIt = std::ranges::find(mock.ops, PresentLifecycleOp::destroySemaphore);
        REQUIRE(waitIt != mock.ops.end());
        REQUIRE(destroySemIt != mock.ops.end());
        CHECK(waitIt < destroySemIt);
    }

    SECTION("fallback strategy without maintenance1 or present_wait idles present queue before destroying semaphores")
    {
        PresentMockState mock;
        gPresentMock = &mock;
        {
            rachis::vulkan::VulkanCapabilitiesSwapchainTestAccess fixture(
                rachis::VulkanPresentCompletionStrategy::none);

            mock.ops.clear();
            REQUIRE(fixture.manager.acquireNextImage(fixture.handle).has_value());
            const auto point = fixture.submitEmptyGraphicsWork();
            bool injectedLoss = false;
            REQUIRE(fixture.manager.present(fixture.handle, point, injectedLoss).has_value());
            CHECK(mock.presentedIds.empty());

            mock.ops.clear();
            REQUIRE(fixture.manager.destroySwapchain(fixture.handle).has_value());
        }
        gPresentMock = nullptr;

        CHECK(mock.waitedPresentIds.empty());
        const auto queueIdleIt = std::ranges::find(mock.ops, PresentLifecycleOp::presentQueueIdle);
        const auto destroySemIt = std::ranges::find(mock.ops, PresentLifecycleOp::destroySemaphore);
        REQUIRE(queueIdleIt != mock.ops.end());
        REQUIRE(destroySemIt != mock.ops.end());
        CHECK(queueIdleIt < destroySemIt);
    }

    SECTION("presentWait strategy skips vkWaitForPresentKHR when no present was ever submitted")
    {
        PresentMockState mock;
        gPresentMock = &mock;
        {
            rachis::vulkan::VulkanCapabilitiesSwapchainTestAccess fixture(
                rachis::VulkanPresentCompletionStrategy::presentWait);
            mock.ops.clear();
            REQUIRE(fixture.manager.destroySwapchain(fixture.handle).has_value());
            CHECK(mock.waitedPresentIds.empty());
            CHECK(std::ranges::find(mock.ops, PresentLifecycleOp::waitForPresent) == mock.ops.end());
        }
        gPresentMock = nullptr;
    }
}

TEST_CASE("VulkanSwapchainManager waits for an image's previous present before reacquiring it under presentWait",
          "[size:small][rhi][vulkan][present-wait]")
{
    PresentMockState mock;
    gPresentMock = &mock;
    {
        rachis::vulkan::VulkanCapabilitiesSwapchainTestAccess fixture(
            rachis::VulkanPresentCompletionStrategy::presentWait);
        fixture.presentFrame();
        fixture.presentFrame();
        REQUIRE(mock.presentedIds == std::vector<std::uint64_t>{1, 2});
        CHECK(mock.waitedPresentIds.empty());
        CHECK(fixture.inFlightPresents() == 2);

        mock.waitForPresentResult = VK_TIMEOUT;
        const auto pending = fixture.manager.acquireNextImage(fixture.handle);
        REQUIRE_FALSE(pending.has_value());
        CHECK(pending.error().code == rachis::ErrorCode::notReady);
        CHECK(pending.error().nativeCode == static_cast<std::int64_t>(VK_TIMEOUT));
        REQUIRE(mock.waitedPresentIds == std::vector<std::uint64_t>{1});
        CHECK(mock.waitedTimeouts.front() != UINT64_MAX);
        CHECK_FALSE(fixture.snapshot().acquired);
        CHECK(fixture.inFlightPresents() == 2);

        mock.waitForPresentResult = VK_SUCCESS;
        const auto reacquired = fixture.manager.acquireNextImage(fixture.handle);
        REQUIRE(reacquired.has_value());
        CHECK(reacquired->imageIndex == 0);
        CHECK(mock.waitedPresentIds == std::vector<std::uint64_t>{1, 1});
        CHECK(fixture.inFlightPresents() == 1);
        const auto point = fixture.submitEmptyGraphicsWork();
        bool injectedLoss = false;
        REQUIRE(fixture.manager.present(fixture.handle, point, injectedLoss).has_value());

        fixture.presentFrame();
        CHECK(mock.waitedPresentIds == std::vector<std::uint64_t>{1, 1, 2});
        CHECK(mock.presentedIds == std::vector<std::uint64_t>{1, 2, 3, 4});
        CHECK(fixture.inFlightPresents() == 2);
        REQUIRE(fixture.manager.destroySwapchain(fixture.handle).has_value());
    }
    gPresentMock = nullptr;
}

TEST_CASE("VulkanSwapchainManager reuses an acquire semaphore only after its graphics wait retires",
          "[size:small][rhi][vulkan][present]")
{
    PresentMockState mock;
    mock.deferCompletion = true;
    gPresentMock = &mock;
    {
        rachis::vulkan::VulkanCapabilitiesSwapchainTestAccess fixture(
            rachis::VulkanPresentCompletionStrategy::imageLifetimeAssociation);
        const auto semaphores = fixture.snapshot().unusedAcquireSemaphores;

        SECTION("steady-state frames")
        {
            for (int frame = 0; frame < 8; ++frame)
            {
                INFO("frame " << frame);
                fixture.presentFrame();
            }
        }

        SECTION("an acquire that returns an image index outside the swapchain")
        {
            mock.outOfRangeAcquires = 1;
            const auto failed = fixture.manager.acquireNextImage(fixture.handle);
            REQUIRE_FALSE(failed.has_value());
            CHECK(failed.error().code == rachis::ErrorCode::nativeFailure);
            for (int frame = 0; frame < 8; ++frame)
            {
                INFO("frame " << frame);
                fixture.presentFrame();
            }
        }

        SECTION("a graphics queue that has not reached the previous acquire wait")
        {
            for (int frame = 0; frame < 3; ++frame)
            {
                INFO("frame " << frame);
                fixture.presentFrame();
            }
            mock.stallTimelines = true;
            const auto acquireCalls = mock.acquireCalls;
            const auto stalled = fixture.manager.acquireNextImage(fixture.handle);
            REQUIRE_FALSE(stalled.has_value());
            CHECK(stalled.error().code == rachis::ErrorCode::notReady);
            CHECK(mock.acquireCalls == acquireCalls);
            CHECK_FALSE(fixture.snapshot().acquired);
            mock.stallTimelines = false;
            for (int frame = 0; frame < 5; ++frame)
            {
                INFO("frame " << frame);
                fixture.presentFrame();
            }
        }

        CHECK(mock.acquiresWithPendingWait == 0);
        const auto snapshot = fixture.snapshot();
        CHECK(snapshot.unusedAcquireSemaphores + snapshot.assignedAcquireSemaphores == semaphores);
        REQUIRE(fixture.manager.destroySwapchain(fixture.handle).has_value());
    }
    gPresentMock = nullptr;
}

TEST_CASE("VulkanSwapchainManager presents only a graphics timeline point submitted after the acquire",
          "[size:small][rhi][vulkan][present]")
{
    PresentMockState mock;
    gPresentMock = &mock;
    {
        rachis::vulkan::VulkanCapabilitiesSwapchainTestAccess fixture(
            rachis::VulkanPresentCompletionStrategy::imageLifetimeAssociation);
        const auto beforeAcquire = fixture.submitEmptyGraphicsWork();
        REQUIRE(fixture.manager.acquireNextImage(fixture.handle).has_value());
        const auto onCompute = fixture.submitEmptyComputeWork();
        bool injectedLoss = false;

        const auto computePresent = fixture.manager.present(fixture.handle, onCompute, injectedLoss);
        REQUIRE_FALSE(computePresent.has_value());
        CHECK(computePresent.error().code == rachis::ErrorCode::invalidArgument);
        const auto stalePresent = fixture.manager.present(fixture.handle, beforeAcquire, injectedLoss);
        REQUIRE_FALSE(stalePresent.has_value());
        CHECK(stalePresent.error().code == rachis::ErrorCode::invalidArgument);
        CHECK(std::ranges::count(mock.ops, PresentLifecycleOp::present) == 0);
        CHECK(fixture.snapshot().acquired);

        const auto afterAcquire = fixture.submitEmptyGraphicsWork();
        const auto presented = fixture.manager.present(fixture.handle, afterAcquire, injectedLoss);
        REQUIRE(presented.has_value());
        CHECK(*presented == rachis::PresentStatus::presented);
        CHECK(std::ranges::count(mock.ops, PresentLifecycleOp::present) == 1);
        REQUIRE(fixture.manager.destroySwapchain(fixture.handle).has_value());
    }
    gPresentMock = nullptr;
}

namespace
{

    struct CapturedDeviceCreation final
    {
        std::vector<std::string> enabledExtensions;
        bool multiDrawIndirect = false;
        bool drawIndirectCount = false;
        bool maintenance4 = false;
        bool maintenance5 = false;
        bool conditionalRendering = false;
        bool shaderDrawParameters = false;
        bool presentIdFeature = false;
        bool presentWaitFeature = false;
        bool deviceFaultFeature = false;
    };

    thread_local CapturedDeviceCreation* gCapturedDeviceCreate = nullptr;
    thread_local bool gMockSupportMaintenance4 = true;
    thread_local bool gMockSupportShaderDrawParameters = true;

    VKAPI_ATTR void VKAPI_CALL mockDeviceCreateFeatures2(VkPhysicalDevice, VkPhysicalDeviceFeatures2* features2)
    {
        features2->features.independentBlend = VK_TRUE;
        features2->features.samplerAnisotropy = VK_TRUE;
        features2->features.textureCompressionBC = VK_TRUE;
        features2->features.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
        features2->features.multiDrawIndirect = VK_TRUE;
        auto* cursor = static_cast<VkBaseOutStructure*>(features2->pNext);
        while (cursor != nullptr)
        {
            if (cursor->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES)
            {
                auto* v11 = reinterpret_cast<VkPhysicalDeviceVulkan11Features*>(cursor);
                v11->shaderDrawParameters = gMockSupportShaderDrawParameters ? VK_TRUE : VK_FALSE;
            }
            else if (cursor->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
            {
                auto* v12 = reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(cursor);
                v12->hostQueryReset = VK_TRUE;
                v12->drawIndirectCount = VK_TRUE;
            }
            else if (cursor->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES)
            {
                auto* v13 = reinterpret_cast<VkPhysicalDeviceVulkan13Features*>(cursor);
                v13->maintenance4 = gMockSupportMaintenance4 ? VK_TRUE : VK_FALSE;
            }
            cursor = cursor->pNext;
        }
    }

    VKAPI_ATTR void VKAPI_CALL mockGetDeviceQueue(VkDevice, std::uint32_t, std::uint32_t, VkQueue* pQueue)
    {
        *pQueue = fakeVk<VkQueue>(0x777);
    }

    VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL mockCreateDeviceGetProcAddr(VkDevice, const char* name)
    {
        if (std::string_view(name) == "vkGetDeviceQueue")
        {
            return reinterpret_cast<PFN_vkVoidFunction>(&mockGetDeviceQueue);
        }
        return nullptr;
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockCaptureCreateDevice(VkPhysicalDevice,
                                                           const VkDeviceCreateInfo* createInfo,
                                                           const VkAllocationCallbacks*,
                                                           VkDevice* pDevice)
    {
        gCapturedDeviceCreate->enabledExtensions.clear();
        for (std::uint32_t i = 0; i < createInfo->enabledExtensionCount; ++i)
        {
            gCapturedDeviceCreate->enabledExtensions.emplace_back(createInfo->ppEnabledExtensionNames[i]);
        }
        const auto* cursor = static_cast<const VkBaseInStructure*>(createInfo->pNext);
        while (cursor != nullptr)
        {
            if (cursor->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
            {
                const auto* f2 = reinterpret_cast<const VkPhysicalDeviceFeatures2*>(cursor);
                gCapturedDeviceCreate->multiDrawIndirect = f2->features.multiDrawIndirect == VK_TRUE;
            }
            else if (cursor->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES)
            {
                const auto* v11 = reinterpret_cast<const VkPhysicalDeviceVulkan11Features*>(cursor);
                gCapturedDeviceCreate->shaderDrawParameters = v11->shaderDrawParameters == VK_TRUE;
            }
            else if (cursor->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
            {
                const auto* v12 = reinterpret_cast<const VkPhysicalDeviceVulkan12Features*>(cursor);
                gCapturedDeviceCreate->drawIndirectCount = v12->drawIndirectCount == VK_TRUE;
            }
            else if (cursor->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES)
            {
                const auto* v13 = reinterpret_cast<const VkPhysicalDeviceVulkan13Features*>(cursor);
                gCapturedDeviceCreate->maintenance4 = v13->maintenance4 == VK_TRUE;
            }
            else if (cursor->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR)
            {
                const auto* pid = reinterpret_cast<const VkPhysicalDevicePresentIdFeaturesKHR*>(cursor);
                gCapturedDeviceCreate->presentIdFeature = pid->presentId == VK_TRUE;
            }
            else if (cursor->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR)
            {
                const auto* pwait = reinterpret_cast<const VkPhysicalDevicePresentWaitFeaturesKHR*>(cursor);
                gCapturedDeviceCreate->presentWaitFeature = pwait->presentWait == VK_TRUE;
            }
            else if (cursor->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT)
            {
                const auto* fault = reinterpret_cast<const VkPhysicalDeviceFaultFeaturesEXT*>(cursor);
                gCapturedDeviceCreate->deviceFaultFeature = fault->deviceFault == VK_TRUE;
            }
            else if (cursor->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR)
            {
                const auto* m5 = reinterpret_cast<const VkPhysicalDeviceMaintenance5FeaturesKHR*>(cursor);
                gCapturedDeviceCreate->maintenance5 = m5->maintenance5 == VK_TRUE;
            }
            else if (cursor->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT)
            {
                const auto* conditional =
                    reinterpret_cast<const VkPhysicalDeviceConditionalRenderingFeaturesEXT*>(cursor);
                gCapturedDeviceCreate->conditionalRendering = conditional->conditionalRendering == VK_TRUE;
            }
            cursor = cursor->pNext;
        }
        *pDevice = fakeVk<VkDevice>(0x888);
        return VK_SUCCESS;
    }

} // namespace

TEST_CASE("createVulkanLogicalDevice enables present_id, present_wait, device_fault, multiDrawIndirect, and "
          "drawIndirectCount",
          "[size:small][rhi][vulkan][device-create]")
{
    CapturedDeviceCreation captured;
    gCapturedDeviceCreate = &captured;

    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetPhysicalDeviceFeatures2 = &mockDeviceCreateFeatures2;
    instanceTable.vkCreateDevice = &mockCaptureCreateDevice;
    instanceTable.vkGetDeviceProcAddr = &mockCreateDeviceGetProcAddr;

    rachis::VulkanInstanceReport instanceReport{
        .apiMajor = 1,
        .apiMinor = 3,
        .khrSurface = true,
        .khrWin32Surface = true,
        .khrWaylandSurface = true,
        .khrPortabilityEnumeration = true,
    };

    rachis::vulkan::VulkanPhysicalDeviceRecord record{};
    record.physical = fakeVk<VkPhysicalDevice>(0x42);
    record.report = baselineProfileDevice();
    record.report.khrPresentId = true;
    record.report.khrPresentWait = true;
    record.report.extDeviceFault = true;
    record.presentId = true;
    record.presentWait = true;
    record.queues = {.graphics = 0, .compute = 0, .transfer = 0};
    record.deviceFault = true;
    record.multiDrawIndirect = true;
    record.drawIndirectCount = true;

    rachis::vulkan::VulkanQueueHandles queues{};
    gMockSupportMaintenance4 = true;
    const auto created = rachis::vulkan::createVulkanLogicalDevice(instanceTable, record, instanceReport, queues, true);
    gCapturedDeviceCreate = nullptr;

    REQUIRE(created.has_value());
    CHECK(queues.presentCompletion == rachis::VulkanPresentCompletionStrategy::presentWait);
    CHECK(queues.maintenance4Enabled);
    CHECK(std::ranges::contains(captured.enabledExtensions, std::string_view{VK_KHR_PRESENT_ID_EXTENSION_NAME}));
    CHECK(std::ranges::contains(captured.enabledExtensions, std::string_view{VK_KHR_PRESENT_WAIT_EXTENSION_NAME}));
    CHECK(std::ranges::contains(captured.enabledExtensions, std::string_view{VK_EXT_DEVICE_FAULT_EXTENSION_NAME}));
    CHECK(captured.presentIdFeature);
    CHECK(captured.presentWaitFeature);
    CHECK(captured.deviceFaultFeature);
    CHECK(captured.multiDrawIndirect);
    CHECK(captured.drawIndirectCount);
    CHECK(captured.maintenance4);
    CHECK(queues.shaderFeaturesEnabled.shaderDrawParameters);
    CHECK(captured.shaderDrawParameters);

    CapturedDeviceCreation capturedWithoutDrawParameters;
    gCapturedDeviceCreate = &capturedWithoutDrawParameters;
    gMockSupportShaderDrawParameters = false;
    rachis::vulkan::VulkanQueueHandles queuesWithoutDrawParameters{};
    const auto createdWithoutDrawParameters = rachis::vulkan::createVulkanLogicalDevice(
        instanceTable, record, instanceReport, queuesWithoutDrawParameters, true);
    gMockSupportShaderDrawParameters = true;
    gCapturedDeviceCreate = nullptr;
    REQUIRE(createdWithoutDrawParameters.has_value());
    CHECK_FALSE(queuesWithoutDrawParameters.shaderFeaturesEnabled.shaderDrawParameters);
    CHECK_FALSE(capturedWithoutDrawParameters.shaderDrawParameters);
    CHECK(queuesWithoutDrawParameters.maintenance4Enabled);

    CapturedDeviceCreation capturedWithoutM4;
    gCapturedDeviceCreate = &capturedWithoutM4;
    gMockSupportMaintenance4 = false;
    rachis::vulkan::VulkanQueueHandles queuesWithoutM4{};
    const auto createdWithoutM4 =
        rachis::vulkan::createVulkanLogicalDevice(instanceTable, record, instanceReport, queuesWithoutM4, false);
    gMockSupportMaintenance4 = true;
    gCapturedDeviceCreate = nullptr;
    REQUIRE(createdWithoutM4.has_value());
    CHECK_FALSE(queuesWithoutM4.maintenance4Enabled);
    CHECK_FALSE(capturedWithoutM4.maintenance4);

    const auto publishedCaps = rachis::vulkan::publishVulkanCapabilities(instanceTable, record, false);
    CHECK(publishedCaps.multiDrawIndirect);
    CHECK(publishedCaps.drawIndirectCount);
}

TEST_CASE("createVulkanLogicalDevice enables VK_KHR_maintenance5 only when the device provides it",
          "[size:small][rhi][vulkan][device-create]")
{
    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetPhysicalDeviceFeatures2 = &mockDeviceCreateFeatures2;
    instanceTable.vkCreateDevice = &mockCaptureCreateDevice;
    instanceTable.vkGetDeviceProcAddr = &mockCreateDeviceGetProcAddr;

    const rachis::VulkanInstanceReport instanceReport{
        .apiMajor = 1,
        .apiMinor = 3,
        .khrSurface = true,
        .khrWin32Surface = true,
        .khrWaylandSurface = true,
        .khrPortabilityEnumeration = true,
    };

    rachis::vulkan::VulkanPhysicalDeviceRecord record{};
    record.physical = fakeVk<VkPhysicalDevice>(0x42);
    record.report = baselineProfileDevice();
    record.queues = {.graphics = 0, .compute = 0, .transfer = 0};
    gMockSupportMaintenance4 = true;

    for (const bool maintenance5 : {true, false})
    {
        CAPTURE(maintenance5);
        record.maintenance5 = maintenance5;
        CapturedDeviceCreation captured;
        gCapturedDeviceCreate = &captured;
        rachis::vulkan::VulkanQueueHandles queues{};
        const auto created =
            rachis::vulkan::createVulkanLogicalDevice(instanceTable, record, instanceReport, queues, false);
        gCapturedDeviceCreate = nullptr;
        REQUIRE(created.has_value());
        CHECK(queues.maintenance5Enabled == maintenance5);
        CHECK(captured.maintenance5 == maintenance5);
        CHECK(std::ranges::contains(captured.enabledExtensions,
                                    std::string_view{VK_KHR_MAINTENANCE_5_EXTENSION_NAME}) == maintenance5);
        CHECK(captured.maintenance4);
    }
}

TEST_CASE("createVulkanLogicalDevice enables VK_EXT_conditional_rendering only when the device provides it",
          "[size:small][rhi][vulkan][device-create][predicate]")
{
    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetPhysicalDeviceFeatures2 = &mockDeviceCreateFeatures2;
    instanceTable.vkCreateDevice = &mockCaptureCreateDevice;
    instanceTable.vkGetDeviceProcAddr = &mockCreateDeviceGetProcAddr;

    const rachis::VulkanInstanceReport instanceReport{
        .apiMajor = 1,
        .apiMinor = 3,
        .khrSurface = true,
        .khrWin32Surface = true,
        .khrWaylandSurface = true,
        .khrPortabilityEnumeration = true,
    };

    rachis::vulkan::VulkanPhysicalDeviceRecord record{};
    record.physical = fakeVk<VkPhysicalDevice>(0x42);
    record.report = baselineProfileDevice();
    record.queues = {.graphics = 0, .compute = 0, .transfer = 0};
    gMockSupportMaintenance4 = true;

    for (const bool conditionalRendering : {true, false})
    {
        CAPTURE(conditionalRendering);
        record.conditionalRendering = conditionalRendering;
        CapturedDeviceCreation captured;
        gCapturedDeviceCreate = &captured;
        rachis::vulkan::VulkanQueueHandles queues{};
        const auto created =
            rachis::vulkan::createVulkanLogicalDevice(instanceTable, record, instanceReport, queues, false);
        gCapturedDeviceCreate = nullptr;
        REQUIRE(created.has_value());
        CHECK(captured.conditionalRendering == conditionalRendering);
        CHECK(std::ranges::contains(captured.enabledExtensions,
                                    std::string_view{VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME}) ==
              conditionalRendering);

        const auto published = rachis::vulkan::publishVulkanCapabilities(instanceTable, record, false);
        CHECK(published.conditionalExecution == conditionalRendering);
        CHECK(published.multiDispatchIndirect);
    }

    for (const bool bufferDeviceAddress : {true, false})
    {
        for (const bool shaderInt64 : {true, false})
        {
            CAPTURE(bufferDeviceAddress, shaderInt64);
            record.bufferDeviceAddress = bufferDeviceAddress;
            record.shaderInt64 = shaderInt64;
            const auto published = rachis::vulkan::publishVulkanCapabilities(instanceTable, record, false);
            CHECK(published.dispatchIndirectCount == (bufferDeviceAddress && shaderInt64));
        }
    }
}

namespace
{

    struct CapturedMultiviewFeatures final
    {
        bool multiview = false;
        bool multiviewGeometryShader = false;
        bool multiviewTessellationShader = false;
        bool meshShader = false;
        bool multiviewMeshShader = false;
    };

    thread_local CapturedMultiviewFeatures* gCapturedMultiview = nullptr;
    thread_local bool gMockReportsMultiview = true;
    thread_local bool gMockReportsStageFeatures = true;

    VKAPI_ATTR void VKAPI_CALL mockMultiviewDeviceFeatures2(VkPhysicalDevice physical,
                                                            VkPhysicalDeviceFeatures2* features2)
    {
        mockDeviceCreateFeatures2(physical, features2);
        features2->features.geometryShader = gMockReportsStageFeatures ? VK_TRUE : VK_FALSE;
        features2->features.tessellationShader = gMockReportsStageFeatures ? VK_TRUE : VK_FALSE;
        auto* cursor = static_cast<VkBaseOutStructure*>(features2->pNext);
        while (cursor != nullptr)
        {
            if (cursor->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES)
            {
                auto* v11 = reinterpret_cast<VkPhysicalDeviceVulkan11Features*>(cursor);
                v11->multiview = gMockReportsMultiview ? VK_TRUE : VK_FALSE;
            }
            cursor = cursor->pNext;
        }
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockCaptureMultiviewCreateDevice(VkPhysicalDevice,
                                                                    const VkDeviceCreateInfo* createInfo,
                                                                    const VkAllocationCallbacks*,
                                                                    VkDevice* pDevice)
    {
        const auto* cursor = static_cast<const VkBaseInStructure*>(createInfo->pNext);
        while (cursor != nullptr)
        {
            if (cursor->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES)
            {
                const auto* v11 = reinterpret_cast<const VkPhysicalDeviceVulkan11Features*>(cursor);
                gCapturedMultiview->multiview = v11->multiview == VK_TRUE;
                gCapturedMultiview->multiviewGeometryShader = v11->multiviewGeometryShader == VK_TRUE;
                gCapturedMultiview->multiviewTessellationShader = v11->multiviewTessellationShader == VK_TRUE;
            }
            else if (cursor->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT)
            {
                const auto* mesh = reinterpret_cast<const VkPhysicalDeviceMeshShaderFeaturesEXT*>(cursor);
                gCapturedMultiview->meshShader = mesh->meshShader == VK_TRUE;
                gCapturedMultiview->multiviewMeshShader = mesh->multiviewMeshShader == VK_TRUE;
            }
            cursor = cursor->pNext;
        }
        *pDevice = fakeVk<VkDevice>(0x889);
        return VK_SUCCESS;
    }

} // namespace

TEST_CASE("createVulkanLogicalDevice enables multiview geometry, tessellation, and mesh features only with multiview "
          "and their stage",
          "[size:small][rhi][vulkan][device-create][multiview]")
{
    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetPhysicalDeviceFeatures2 = &mockMultiviewDeviceFeatures2;
    instanceTable.vkCreateDevice = &mockCaptureMultiviewCreateDevice;
    instanceTable.vkGetDeviceProcAddr = &mockCreateDeviceGetProcAddr;

    const rachis::VulkanInstanceReport instanceReport{
        .apiMajor = 1,
        .apiMinor = 3,
        .khrSurface = true,
        .khrWin32Surface = true,
        .khrWaylandSurface = true,
        .khrPortabilityEnumeration = true,
    };

    rachis::vulkan::VulkanPhysicalDeviceRecord record{};
    record.physical = fakeVk<VkPhysicalDevice>(0x43);
    record.report = baselineProfileDevice();
    record.queues = {.graphics = 0, .compute = 0, .transfer = 0};
    record.geometryShaders = true;
    record.tessellationShaders = true;
    record.multiview = true;
    record.multiviewGeometryShader = true;
    record.multiviewTessellationShader = true;
    record.meshShaders = true;
    record.multiviewMeshShader = true;

    CapturedMultiviewFeatures captured;
    gCapturedMultiview = &captured;
    rachis::vulkan::VulkanQueueHandles queues{};

    SECTION("every stage feature is enabled when the device has multiview and the stage")
    {
        REQUIRE(rachis::vulkan::createVulkanLogicalDevice(instanceTable, record, instanceReport, queues, false));
        CHECK(captured.multiview);
        CHECK(captured.multiviewGeometryShader);
        CHECK(captured.multiviewTessellationShader);
        CHECK(captured.meshShader);
        CHECK(captured.multiviewMeshShader);
        CHECK(queues.multiviewStagesEnabled.geometry);
        CHECK(queues.multiviewStagesEnabled.tessellation);
        CHECK(queues.multiviewStagesEnabled.mesh);
    }

    SECTION("no stage feature is enabled when multiview is not enabled")
    {
        gMockReportsMultiview = false;
        const auto created =
            rachis::vulkan::createVulkanLogicalDevice(instanceTable, record, instanceReport, queues, false);
        gMockReportsMultiview = true;
        REQUIRE(created);
        CHECK_FALSE(captured.multiview);
        CHECK_FALSE(captured.multiviewGeometryShader);
        CHECK_FALSE(captured.multiviewTessellationShader);
        CHECK(captured.meshShader);
        CHECK_FALSE(captured.multiviewMeshShader);
        CHECK_FALSE(queues.multiviewStagesEnabled.geometry);
        CHECK_FALSE(queues.multiviewStagesEnabled.tessellation);
        CHECK_FALSE(queues.multiviewStagesEnabled.mesh);
    }

    SECTION("no stage feature is enabled without its base stage feature")
    {
        gMockReportsStageFeatures = false;
        record.meshShaders = false;
        const auto created =
            rachis::vulkan::createVulkanLogicalDevice(instanceTable, record, instanceReport, queues, false);
        gMockReportsStageFeatures = true;
        REQUIRE(created);
        CHECK(captured.multiview);
        CHECK_FALSE(captured.multiviewGeometryShader);
        CHECK_FALSE(captured.multiviewTessellationShader);
        CHECK_FALSE(captured.meshShader);
        CHECK_FALSE(captured.multiviewMeshShader);
        CHECK_FALSE(queues.multiviewStagesEnabled.geometry);
        CHECK_FALSE(queues.multiviewStagesEnabled.tessellation);
        CHECK_FALSE(queues.multiviewStagesEnabled.mesh);
    }

    SECTION("a stage feature stays disabled when the device lacks it")
    {
        record.multiviewGeometryShader = false;
        record.multiviewTessellationShader = false;
        record.multiviewMeshShader = false;
        REQUIRE(rachis::vulkan::createVulkanLogicalDevice(instanceTable, record, instanceReport, queues, false));
        CHECK(captured.multiview);
        CHECK_FALSE(captured.multiviewGeometryShader);
        CHECK_FALSE(captured.multiviewTessellationShader);
        CHECK(captured.meshShader);
        CHECK_FALSE(captured.multiviewMeshShader);
        CHECK_FALSE(queues.multiviewStagesEnabled.geometry);
        CHECK_FALSE(queues.multiviewStagesEnabled.tessellation);
        CHECK_FALSE(queues.multiviewStagesEnabled.mesh);
    }

    gCapturedMultiview = nullptr;
}

namespace
{

    thread_local rachis::vulkan::VulkanShaderFeatureSupport* gCapturedShaderFeatures = nullptr;
    thread_local rachis::vulkan::VulkanShaderFeatureSupport gMockReportedShaderFeatures{};

    VKAPI_ATTR void VKAPI_CALL mockShaderFeatureDeviceFeatures2(VkPhysicalDevice physical,
                                                                VkPhysicalDeviceFeatures2* features2)
    {
        mockDeviceCreateFeatures2(physical, features2);
        const auto& reported = gMockReportedShaderFeatures;
        features2->features.sampleRateShading = reported.sampleRateShading ? VK_TRUE : VK_FALSE;
        features2->features.vertexPipelineStoresAndAtomics =
            reported.vertexPipelineStoresAndAtomics ? VK_TRUE : VK_FALSE;
        features2->features.shaderImageGatherExtended = reported.shaderImageGatherExtended ? VK_TRUE : VK_FALSE;
        features2->features.shaderResourceMinLod = reported.shaderResourceMinLod ? VK_TRUE : VK_FALSE;
        features2->features.shaderFloat64 = reported.shaderFloat64 ? VK_TRUE : VK_FALSE;
        auto* cursor = static_cast<VkBaseOutStructure*>(features2->pNext);
        while (cursor != nullptr)
        {
            if (cursor->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES)
            {
                auto* v11 = reinterpret_cast<VkPhysicalDeviceVulkan11Features*>(cursor);
                v11->storageBuffer16BitAccess = reported.storageBuffer16BitAccess ? VK_TRUE : VK_FALSE;
                v11->uniformAndStorageBuffer16BitAccess =
                    reported.uniformAndStorageBuffer16BitAccess ? VK_TRUE : VK_FALSE;
                v11->storagePushConstant16 = reported.storagePushConstant16 ? VK_TRUE : VK_FALSE;
                v11->storageInputOutput16 = reported.storageInputOutput16 ? VK_TRUE : VK_FALSE;
            }
            cursor = cursor->pNext;
        }
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockCaptureShaderFeatureCreateDevice(VkPhysicalDevice,
                                                                        const VkDeviceCreateInfo* createInfo,
                                                                        const VkAllocationCallbacks*,
                                                                        VkDevice* pDevice)
    {
        const auto* cursor = static_cast<const VkBaseInStructure*>(createInfo->pNext);
        while (cursor != nullptr)
        {
            if (cursor->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
            {
                const auto& enabled = reinterpret_cast<const VkPhysicalDeviceFeatures2*>(cursor)->features;
                gCapturedShaderFeatures->sampleRateShading = enabled.sampleRateShading == VK_TRUE;
                gCapturedShaderFeatures->vertexPipelineStoresAndAtomics =
                    enabled.vertexPipelineStoresAndAtomics == VK_TRUE;
                gCapturedShaderFeatures->shaderImageGatherExtended = enabled.shaderImageGatherExtended == VK_TRUE;
                gCapturedShaderFeatures->shaderResourceMinLod = enabled.shaderResourceMinLod == VK_TRUE;
                gCapturedShaderFeatures->shaderFloat64 = enabled.shaderFloat64 == VK_TRUE;
            }
            else if (cursor->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES)
            {
                const auto* v11 = reinterpret_cast<const VkPhysicalDeviceVulkan11Features*>(cursor);
                gCapturedShaderFeatures->storageBuffer16BitAccess = v11->storageBuffer16BitAccess == VK_TRUE;
                gCapturedShaderFeatures->uniformAndStorageBuffer16BitAccess =
                    v11->uniformAndStorageBuffer16BitAccess == VK_TRUE;
                gCapturedShaderFeatures->storagePushConstant16 = v11->storagePushConstant16 == VK_TRUE;
                gCapturedShaderFeatures->storageInputOutput16 = v11->storageInputOutput16 == VK_TRUE;
            }
            cursor = cursor->pNext;
        }
        *pDevice = fakeVk<VkDevice>(0x88A);
        return VK_SUCCESS;
    }

} // namespace

TEST_CASE("createVulkanLogicalDevice enables sampleRateShading, vertexPipelineStoresAndAtomics, "
          "shaderImageGatherExtended, and shaderResourceMinLod when the device has them",
          "[size:small][rhi][vulkan][device-create]")
{
    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetPhysicalDeviceFeatures2 = &mockShaderFeatureDeviceFeatures2;
    instanceTable.vkCreateDevice = &mockCaptureShaderFeatureCreateDevice;
    instanceTable.vkGetDeviceProcAddr = &mockCreateDeviceGetProcAddr;

    const rachis::VulkanInstanceReport instanceReport{
        .apiMajor = 1,
        .apiMinor = 3,
        .khrSurface = true,
        .khrWin32Surface = true,
        .khrWaylandSurface = true,
        .khrPortabilityEnumeration = true,
    };

    rachis::vulkan::VulkanPhysicalDeviceRecord record{};
    record.physical = fakeVk<VkPhysicalDevice>(0x44);
    record.report = baselineProfileDevice();
    record.queues = {.graphics = 0, .compute = 0, .transfer = 0};

    rachis::vulkan::VulkanShaderFeatureSupport captured{};
    gCapturedShaderFeatures = &captured;
    rachis::vulkan::VulkanQueueHandles queues{};

    SECTION("every feature the device reports is enabled and recorded")
    {
        gMockReportedShaderFeatures = {
            .sampleRateShading = true,
            .vertexPipelineStoresAndAtomics = true,
            .shaderImageGatherExtended = true,
            .shaderResourceMinLod = true,
        };
        REQUIRE(rachis::vulkan::createVulkanLogicalDevice(instanceTable, record, instanceReport, queues, false));
        CHECK(captured.sampleRateShading);
        CHECK(captured.vertexPipelineStoresAndAtomics);
        CHECK(captured.shaderImageGatherExtended);
        CHECK(captured.shaderResourceMinLod);
        CHECK(queues.shaderFeaturesEnabled.sampleRateShading);
        CHECK(queues.shaderFeaturesEnabled.vertexPipelineStoresAndAtomics);
        CHECK(queues.shaderFeaturesEnabled.shaderImageGatherExtended);
        CHECK(queues.shaderFeaturesEnabled.shaderResourceMinLod);
    }

    SECTION("a feature the device lacks stays disabled and unrecorded")
    {
        gMockReportedShaderFeatures = {
            .sampleRateShading = true,
            .vertexPipelineStoresAndAtomics = true,
            .shaderImageGatherExtended = true,
            .shaderResourceMinLod = false,
        };
        REQUIRE(rachis::vulkan::createVulkanLogicalDevice(instanceTable, record, instanceReport, queues, false));
        CHECK(captured.sampleRateShading);
        CHECK(captured.vertexPipelineStoresAndAtomics);
        CHECK(captured.shaderImageGatherExtended);
        CHECK_FALSE(captured.shaderResourceMinLod);
        CHECK(queues.shaderFeaturesEnabled.sampleRateShading);
        CHECK(queues.shaderFeaturesEnabled.vertexPipelineStoresAndAtomics);
        CHECK(queues.shaderFeaturesEnabled.shaderImageGatherExtended);
        CHECK_FALSE(queues.shaderFeaturesEnabled.shaderResourceMinLod);
    }

    SECTION("no feature is enabled when the device reports none")
    {
        gMockReportedShaderFeatures = {};
        REQUIRE(rachis::vulkan::createVulkanLogicalDevice(instanceTable, record, instanceReport, queues, false));
        CHECK_FALSE(captured.sampleRateShading);
        CHECK_FALSE(captured.vertexPipelineStoresAndAtomics);
        CHECK_FALSE(captured.shaderImageGatherExtended);
        CHECK_FALSE(captured.shaderResourceMinLod);
        CHECK_FALSE(queues.shaderFeaturesEnabled.sampleRateShading);
        CHECK_FALSE(queues.shaderFeaturesEnabled.vertexPipelineStoresAndAtomics);
        CHECK_FALSE(queues.shaderFeaturesEnabled.shaderImageGatherExtended);
        CHECK_FALSE(queues.shaderFeaturesEnabled.shaderResourceMinLod);
    }

    gMockReportedShaderFeatures = {};
    gCapturedShaderFeatures = nullptr;
}

TEST_CASE("createVulkanLogicalDevice enables shaderFloat64 and the 16-bit storage features when the device has them",
          "[size:small][rhi][vulkan][device-create]")
{
    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetPhysicalDeviceFeatures2 = &mockShaderFeatureDeviceFeatures2;
    instanceTable.vkCreateDevice = &mockCaptureShaderFeatureCreateDevice;
    instanceTable.vkGetDeviceProcAddr = &mockCreateDeviceGetProcAddr;

    const rachis::VulkanInstanceReport instanceReport{
        .apiMajor = 1,
        .apiMinor = 3,
        .khrSurface = true,
        .khrWin32Surface = true,
        .khrWaylandSurface = true,
        .khrPortabilityEnumeration = true,
    };

    rachis::vulkan::VulkanPhysicalDeviceRecord record{};
    record.physical = fakeVk<VkPhysicalDevice>(0x45);
    record.report = baselineProfileDevice();
    record.queues = {.graphics = 0, .compute = 0, .transfer = 0};

    rachis::vulkan::VulkanShaderFeatureSupport captured{};
    gCapturedShaderFeatures = &captured;
    rachis::vulkan::VulkanQueueHandles queues{};

    SECTION("every feature the device reports is enabled and recorded")
    {
        gMockReportedShaderFeatures = {
            .shaderFloat64 = true,
            .storageBuffer16BitAccess = true,
            .uniformAndStorageBuffer16BitAccess = true,
            .storagePushConstant16 = true,
            .storageInputOutput16 = true,
        };
        REQUIRE(rachis::vulkan::createVulkanLogicalDevice(instanceTable, record, instanceReport, queues, false));
        CHECK(captured.shaderFloat64);
        CHECK(captured.storageBuffer16BitAccess);
        CHECK(captured.uniformAndStorageBuffer16BitAccess);
        CHECK(captured.storagePushConstant16);
        CHECK(captured.storageInputOutput16);
        CHECK(queues.shaderFeaturesEnabled.shaderFloat64);
        CHECK(queues.shaderFeaturesEnabled.storageBuffer16BitAccess);
        CHECK(queues.shaderFeaturesEnabled.uniformAndStorageBuffer16BitAccess);
        CHECK(queues.shaderFeaturesEnabled.storagePushConstant16);
        CHECK(queues.shaderFeaturesEnabled.storageInputOutput16);
    }

    SECTION("16-bit stage inputs and outputs stay disabled when only the buffer features are reported")
    {
        gMockReportedShaderFeatures = {
            .shaderFloat64 = true,
            .storageBuffer16BitAccess = true,
            .uniformAndStorageBuffer16BitAccess = true,
            .storagePushConstant16 = true,
            .storageInputOutput16 = false,
        };
        REQUIRE(rachis::vulkan::createVulkanLogicalDevice(instanceTable, record, instanceReport, queues, false));
        CHECK(captured.storageBuffer16BitAccess);
        CHECK(captured.uniformAndStorageBuffer16BitAccess);
        CHECK(captured.storagePushConstant16);
        CHECK_FALSE(captured.storageInputOutput16);
        CHECK(queues.shaderFeaturesEnabled.storageBuffer16BitAccess);
        CHECK(queues.shaderFeaturesEnabled.uniformAndStorageBuffer16BitAccess);
        CHECK(queues.shaderFeaturesEnabled.storagePushConstant16);
        CHECK_FALSE(queues.shaderFeaturesEnabled.storageInputOutput16);
    }

    SECTION("no feature is enabled when the device reports none")
    {
        gMockReportedShaderFeatures = {};
        REQUIRE(rachis::vulkan::createVulkanLogicalDevice(instanceTable, record, instanceReport, queues, false));
        CHECK_FALSE(captured.shaderFloat64);
        CHECK_FALSE(captured.storageBuffer16BitAccess);
        CHECK_FALSE(captured.uniformAndStorageBuffer16BitAccess);
        CHECK_FALSE(captured.storagePushConstant16);
        CHECK_FALSE(captured.storageInputOutput16);
        CHECK_FALSE(queues.shaderFeaturesEnabled.shaderFloat64);
        CHECK_FALSE(queues.shaderFeaturesEnabled.storageBuffer16BitAccess);
        CHECK_FALSE(queues.shaderFeaturesEnabled.uniformAndStorageBuffer16BitAccess);
        CHECK_FALSE(queues.shaderFeaturesEnabled.storagePushConstant16);
        CHECK_FALSE(queues.shaderFeaturesEnabled.storageInputOutput16);
    }

    gMockReportedShaderFeatures = {};
    gCapturedShaderFeatures = nullptr;
}

namespace
{

    struct CapturedQueueRequest final
    {
        std::uint32_t family = 0;
        std::uint32_t count = 0;
    };

    struct CapturedQueueFetch final
    {
        std::uint32_t family = 0;
        std::uint32_t index = 0;
    };

    struct CapturedQueueCreation final
    {
        std::vector<CapturedQueueRequest> requests;
        std::vector<CapturedQueueFetch> fetches;
    };

    thread_local CapturedQueueCreation* gCapturedQueueCreate = nullptr;

    [[nodiscard]] VkQueueFamilyProperties queueFamily(VkQueueFlags flags,
                                                      std::uint32_t queueCount,
                                                      VkExtent3D granularity = {1, 1, 1}) noexcept
    {
        VkQueueFamilyProperties properties{};
        properties.queueFlags = flags;
        properties.queueCount = queueCount;
        properties.timestampValidBits = 64;
        properties.minImageTransferGranularity = granularity;
        return properties;
    }

    VKAPI_ATTR void VKAPI_CALL mockIndexedGetDeviceQueue(VkDevice,
                                                         std::uint32_t family,
                                                         std::uint32_t index,
                                                         VkQueue* pQueue)
    {
        gCapturedQueueCreate->fetches.push_back(CapturedQueueFetch{.family = family, .index = index});
        *pQueue = fakeVk<VkQueue>(0x1000U + (family * 0x10U) + index);
    }

    VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL mockIndexedQueueGetProcAddr(VkDevice, const char* name)
    {
        if (std::string_view(name) == "vkGetDeviceQueue")
        {
            return reinterpret_cast<PFN_vkVoidFunction>(&mockIndexedGetDeviceQueue);
        }
        return nullptr;
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockQueueCaptureCreateDevice(VkPhysicalDevice,
                                                                const VkDeviceCreateInfo* createInfo,
                                                                const VkAllocationCallbacks*,
                                                                VkDevice* pDevice)
    {
        for (std::uint32_t i = 0; i < createInfo->queueCreateInfoCount; ++i)
        {
            const auto& request = createInfo->pQueueCreateInfos[i];
            for (std::uint32_t queue = 0; queue < request.queueCount; ++queue)
            {
                CHECK(request.pQueuePriorities[queue] == 1.0F);
            }
            gCapturedQueueCreate->requests.push_back(
                CapturedQueueRequest{.family = request.queueFamilyIndex, .count = request.queueCount});
        }
        *pDevice = fakeVk<VkDevice>(0x889);
        return VK_SUCCESS;
    }

    struct ExpectedQueue final
    {
        std::uint32_t family = 0;
        std::uint32_t index = 0;
    };

    void checkSelectedQueues(const rachis::vulkan::VulkanQueueFamilies& selected,
                             ExpectedQueue graphics,
                             ExpectedQueue compute,
                             ExpectedQueue copy)
    {
        CHECK(selected.graphics == graphics.family);
        CHECK(selected.graphicsIndex == graphics.index);
        CHECK(selected.compute == compute.family);
        CHECK(selected.computeIndex == compute.index);
        CHECK(selected.transfer == copy.family);
        CHECK(selected.transferIndex == copy.index);
    }

    [[nodiscard]] CapturedQueueCreation createDeviceWithQueues(const rachis::vulkan::VulkanQueueFamilies& families,
                                                               rachis::vulkan::VulkanQueueHandles& queues)
    {
        CapturedQueueCreation captured;
        gCapturedQueueCreate = &captured;

        rachis::vulkan::VulkanInstanceTable instanceTable{};
        instanceTable.vkGetPhysicalDeviceFeatures2 = &mockDeviceCreateFeatures2;
        instanceTable.vkCreateDevice = &mockQueueCaptureCreateDevice;
        instanceTable.vkGetDeviceProcAddr = &mockIndexedQueueGetProcAddr;

        const rachis::VulkanInstanceReport instanceReport{
            .apiMajor = 1,
            .apiMinor = 3,
            .khrSurface = true,
            .khrWin32Surface = true,
            .khrWaylandSurface = true,
            .khrPortabilityEnumeration = true,
        };
        rachis::vulkan::VulkanPhysicalDeviceRecord record{};
        record.physical = fakeVk<VkPhysicalDevice>(0x43);
        record.report = baselineProfileDevice();
        record.queues = families;

        const auto created =
            rachis::vulkan::createVulkanLogicalDevice(instanceTable, record, instanceReport, queues, false);
        gCapturedQueueCreate = nullptr;
        REQUIRE(created.has_value());
        return captured;
    }

} // namespace

TEST_CASE("Vulkan queue selection gives each queue type its own native queue unless the device has too few",
          "[size:small][rhi][vulkan][device-create][queue]")
{
    constexpr VkQueueFlags graphics = VK_QUEUE_GRAPHICS_BIT;
    constexpr VkQueueFlags compute = VK_QUEUE_COMPUTE_BIT;
    constexpr VkQueueFlags transfer = VK_QUEUE_TRANSFER_BIT;
    constexpr VkQueueFlags sparse = VK_QUEUE_SPARSE_BINDING_BIT;
    constexpr VkQueueFlags general = graphics | compute | transfer;
    constexpr VkExtent3D coarseGranularity{16, 16, 8};

    SECTION("a second graphics-family queue is preferred over a dedicated compute family")
    {
        const std::array families{
            queueFamily(general | sparse, 16),
            queueFamily(transfer | sparse, 2),
            queueFamily(compute | transfer | sparse, 8),
            queueFamily(transfer | sparse | VK_QUEUE_VIDEO_DECODE_BIT_KHR, 1),
        };
        const auto selected = rachis::vulkan::selectVulkanQueueFamilies(families);
        checkSelectedQueues(selected, {.family = 0, .index = 0}, {.family = 0, .index = 1}, {.family = 1, .index = 0});

        rachis::vulkan::VulkanQueueHandles queues{};
        const auto captured = createDeviceWithQueues(selected, queues);
        REQUIRE(captured.requests.size() == 2);
        CHECK(captured.requests[0].family == 0);
        CHECK(captured.requests[0].count == 2);
        CHECK(captured.requests[1].family == 1);
        CHECK(captured.requests[1].count == 1);
        REQUIRE(captured.fetches.size() == 3);
        CHECK(captured.fetches[1].family == 0);
        CHECK(captured.fetches[1].index == 1);
        CHECK(queues.graphics != queues.compute);
        CHECK(queues.graphics != queues.copy);
        CHECK(queues.compute != queues.copy);
        CHECK(queues.present == queues.graphics);
    }

    SECTION("graphics takes a family that also computes, and compute never lands on a graphics-only family")
    {
        const std::array graphicsFirst{
            queueFamily(graphics | transfer, 2),
            queueFamily(general, 2),
        };
        const auto selected = rachis::vulkan::selectVulkanQueueFamilies(graphicsFirst);
        checkSelectedQueues(selected, {.family = 1, .index = 0}, {.family = 1, .index = 1}, {.family = 0, .index = 0});

        const std::array onlyGraphicsOnly{
            queueFamily(graphics | transfer, 2),
            queueFamily(compute | transfer, 1),
        };
        const auto separated = rachis::vulkan::selectVulkanQueueFamilies(onlyGraphicsOnly);
        CHECK(separated.graphics == 0);
        CHECK(separated.compute == 1);
        CHECK(separated.computeIndex == 0);
    }

    SECTION("a dedicated compute family serves compute when the graphics family has a single queue")
    {
        const std::array families{
            queueFamily(general | sparse, 1),
            queueFamily(compute | transfer | sparse, 4),
            queueFamily(transfer | sparse, 2),
        };
        const auto selected = rachis::vulkan::selectVulkanQueueFamilies(families);
        checkSelectedQueues(selected, {.family = 0, .index = 0}, {.family = 1, .index = 0}, {.family = 2, .index = 0});

        rachis::vulkan::VulkanQueueHandles queues{};
        const auto captured = createDeviceWithQueues(selected, queues);
        REQUIRE(captured.requests.size() == 3);
        for (std::uint32_t family = 0; family < 3; ++family)
        {
            CHECK(captured.requests[family].family == family);
            CHECK(captured.requests[family].count == 1);
        }
        CHECK(queues.graphics != queues.compute);
        CHECK(queues.compute != queues.copy);
    }

    SECTION("a compute-only family that also serves copies hands out a second queue")
    {
        const std::array families{
            queueFamily(general, 1),
            queueFamily(compute | transfer, 4),
        };
        const auto selected = rachis::vulkan::selectVulkanQueueFamilies(families);
        checkSelectedQueues(selected, {.family = 0, .index = 0}, {.family = 1, .index = 0}, {.family = 1, .index = 1});

        rachis::vulkan::VulkanQueueHandles queues{};
        const auto captured = createDeviceWithQueues(selected, queues);
        REQUIRE(captured.requests.size() == 2);
        CHECK(captured.requests[0].family == 0);
        CHECK(captured.requests[0].count == 1);
        CHECK(captured.requests[1].family == 1);
        CHECK(captured.requests[1].count == 2);
        CHECK(queues.compute != queues.copy);
    }

    SECTION("one family with several queues gives every queue type its own index")
    {
        const std::array families{queueFamily(general, 4)};
        const auto selected = rachis::vulkan::selectVulkanQueueFamilies(families);
        checkSelectedQueues(selected, {.family = 0, .index = 0}, {.family = 0, .index = 1}, {.family = 0, .index = 2});

        rachis::vulkan::VulkanQueueHandles queues{};
        const auto captured = createDeviceWithQueues(selected, queues);
        REQUIRE(captured.requests.size() == 1);
        CHECK(captured.requests[0].family == 0);
        CHECK(captured.requests[0].count == 3);
        REQUIRE(captured.fetches.size() == 3);
        for (std::uint32_t index = 0; index < 3; ++index)
        {
            CHECK(captured.fetches[index].family == 0);
            CHECK(captured.fetches[index].index == index);
        }
        CHECK(queues.graphics != queues.compute);
        CHECK(queues.graphics != queues.copy);
        CHECK(queues.compute != queues.copy);
    }

    SECTION("general families with one queue each give every queue type its own family")
    {
        const std::array families{
            queueFamily(general, 1),
            queueFamily(general, 1),
            queueFamily(general, 1),
            queueFamily(general, 1),
        };
        const auto selected = rachis::vulkan::selectVulkanQueueFamilies(families);
        checkSelectedQueues(selected, {.family = 0, .index = 0}, {.family = 1, .index = 0}, {.family = 2, .index = 0});
    }

    SECTION("two queues leave copy on the graphics queue")
    {
        const std::array families{queueFamily(general, 2)};
        const auto selected = rachis::vulkan::selectVulkanQueueFamilies(families);
        checkSelectedQueues(selected, {.family = 0, .index = 0}, {.family = 0, .index = 1}, {.family = 0, .index = 0});

        rachis::vulkan::VulkanQueueHandles queues{};
        const auto captured = createDeviceWithQueues(selected, queues);
        REQUIRE(captured.requests.size() == 1);
        CHECK(captured.requests[0].count == 2);
        CHECK(queues.graphics != queues.compute);
        CHECK(queues.copy == queues.graphics);
    }

    SECTION("copy takes a third graphics-family queue over a coarse-granularity transfer-only family")
    {
        const std::array families{
            queueFamily(general, 3),
            queueFamily(transfer, 1, coarseGranularity),
        };
        const auto selected = rachis::vulkan::selectVulkanQueueFamilies(families);
        checkSelectedQueues(selected, {.family = 0, .index = 0}, {.family = 0, .index = 1}, {.family = 0, .index = 2});

        rachis::vulkan::VulkanQueueHandles queues{};
        const auto captured = createDeviceWithQueues(selected, queues);
        REQUIRE(captured.requests.size() == 1);
        CHECK(captured.requests[0].family == 0);
        CHECK(captured.requests[0].count == 3);
        CHECK(queues.graphics != queues.compute);
        CHECK(queues.graphics != queues.copy);
        CHECK(queues.compute != queues.copy);
    }

    SECTION("copy shares the graphics queue when compute takes the second graphics-family queue and the transfer-only "
            "family has coarse granularity")
    {
        const std::array families{
            queueFamily(general, 2),
            queueFamily(transfer, 1, coarseGranularity),
        };
        const auto selected = rachis::vulkan::selectVulkanQueueFamilies(families);
        checkSelectedQueues(selected, {.family = 0, .index = 0}, {.family = 0, .index = 1}, {.family = 0, .index = 0});

        rachis::vulkan::VulkanQueueHandles queues{};
        const auto captured = createDeviceWithQueues(selected, queues);
        REQUIRE(captured.requests.size() == 1);
        CHECK(captured.requests[0].family == 0);
        CHECK(captured.requests[0].count == 2);
        CHECK(queues.graphics != queues.compute);
        CHECK(queues.copy == queues.graphics);
    }

    SECTION("copy takes a second compute-family queue over a coarse-granularity transfer-only family")
    {
        const std::array families{
            queueFamily(general, 1),
            queueFamily(compute | transfer, 2),
            queueFamily(transfer, 1, coarseGranularity),
        };
        const auto selected = rachis::vulkan::selectVulkanQueueFamilies(families);
        checkSelectedQueues(selected, {.family = 0, .index = 0}, {.family = 1, .index = 0}, {.family = 1, .index = 1});

        rachis::vulkan::VulkanQueueHandles queues{};
        const auto captured = createDeviceWithQueues(selected, queues);
        REQUIRE(captured.requests.size() == 2);
        CHECK(captured.requests[0].family == 0);
        CHECK(captured.requests[0].count == 1);
        CHECK(captured.requests[1].family == 1);
        CHECK(captured.requests[1].count == 2);
        CHECK(queues.graphics != queues.compute);
        CHECK(queues.compute != queues.copy);
    }

    SECTION("a single queue is shared by every queue type")
    {
        const std::array families{queueFamily(general | sparse, 1)};
        const auto selected = rachis::vulkan::selectVulkanQueueFamilies(families);
        checkSelectedQueues(selected, {.family = 0, .index = 0}, {.family = 0, .index = 0}, {.family = 0, .index = 0});

        rachis::vulkan::VulkanQueueHandles queues{};
        const auto captured = createDeviceWithQueues(selected, queues);
        REQUIRE(captured.requests.size() == 1);
        CHECK(captured.requests[0].count == 1);
        CHECK(queues.graphics == queues.compute);
        CHECK(queues.graphics == queues.copy);
    }

    SECTION("a device without a graphics family selects nothing")
    {
        const std::array families{queueFamily(compute | transfer, 4)};
        const auto selected = rachis::vulkan::selectVulkanQueueFamilies(families);
        CHECK(selected.graphics == VK_QUEUE_FAMILY_IGNORED);
    }
}

namespace
{

    struct DeviceFaultMockState final
    {
        std::uint32_t callCount = 0;
        std::uint32_t reportedAddressCount = 2;
        std::uint32_t reportedVendorCount = 1;
        VkDeviceSize reportedVendorBinarySize = 0;
        std::uint32_t secondCallAddressCap = 2;
        std::uint32_t secondCallVendorCap = 1;
        VkResult secondCallResult = VK_SUCCESS;
        std::string description = "GPU page fault in compute dispatch";
        std::vector<VkDeviceFaultAddressInfoEXT> addresses;
        std::vector<VkDeviceFaultVendorInfoEXT> vendors;
    };

    thread_local DeviceFaultMockState* gFaultMock = nullptr;

    VKAPI_ATTR VkResult VKAPI_CALL mockGetDeviceFaultInfoEXT(VkDevice,
                                                             VkDeviceFaultCountsEXT* pFaultCounts,
                                                             VkDeviceFaultInfoEXT* pFaultInfo)
    {
        ++gFaultMock->callCount;
        if (pFaultInfo == nullptr)
        {
            pFaultCounts->addressInfoCount = gFaultMock->reportedAddressCount;
            pFaultCounts->vendorInfoCount = gFaultMock->reportedVendorCount;
            pFaultCounts->vendorBinarySize = gFaultMock->reportedVendorBinarySize;
            return VK_SUCCESS;
        }

        CHECK(pFaultCounts->vendorBinarySize == 0);
        CHECK(pFaultInfo->pVendorBinaryData == nullptr);

        std::memset(pFaultInfo->description, 0, sizeof(pFaultInfo->description));
        std::memcpy(pFaultInfo->description,
                    gFaultMock->description.data(),
                    std::min(gFaultMock->description.size(), sizeof(pFaultInfo->description) - 1));

        const auto addrToCopy = std::min<std::uint32_t>({pFaultCounts->addressInfoCount,
                                                         gFaultMock->secondCallAddressCap,
                                                         static_cast<std::uint32_t>(gFaultMock->addresses.size())});
        for (std::uint32_t i = 0; i < addrToCopy; ++i)
        {
            pFaultInfo->pAddressInfos[i] = gFaultMock->addresses[i];
        }
        pFaultCounts->addressInfoCount = addrToCopy;

        const auto vendorToCopy = std::min<std::uint32_t>({pFaultCounts->vendorInfoCount,
                                                           gFaultMock->secondCallVendorCap,
                                                           static_cast<std::uint32_t>(gFaultMock->vendors.size())});
        for (std::uint32_t i = 0; i < vendorToCopy; ++i)
        {
            pFaultInfo->pVendorInfos[i] = gFaultMock->vendors[i];
        }
        pFaultCounts->vendorInfoCount = vendorToCopy;
        return gFaultMock->secondCallResult;
    }

    VKAPI_ATTR VkResult VKAPI_CALL mockLostCreateCommandPool(VkDevice,
                                                             const VkCommandPoolCreateInfo*,
                                                             const VkAllocationCallbacks*,
                                                             VkCommandPool*)
    {
        return VK_ERROR_DEVICE_LOST;
    }

} // namespace

TEST_CASE("VK_EXT_device_fault captures two-call fault info, aligns addresses, handles VK_INCOMPLETE, and wires into "
          "VulkanDevice loss",
          "[size:small][rhi][vulkan][device-fault]")
{
    SECTION(
        "two-call capture aligns addresses, maps vendor breadcrumbs, and ignores vendorBinarySize-only VK_INCOMPLETE")
    {
        DeviceFaultMockState mock;
        mock.reportedVendorBinarySize = 4096;
        mock.secondCallResult = VK_INCOMPLETE;
        mock.addresses = {
            VkDeviceFaultAddressInfoEXT{
                .addressType = VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_EXT,
                .reportedAddress = 0x1000'1234ULL,
                .addressPrecision = 0x100ULL,
            },
            VkDeviceFaultAddressInfoEXT{
                .addressType = VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT,
                .reportedAddress = 0xDEAD'BEEF'1357ULL,
                .addressPrecision = 0x1000ULL,
            },
        };
        VkDeviceFaultVendorInfoEXT vendor{};
        std::snprintf(vendor.description, sizeof(vendor.description), "%s", "MMU write fault");
        vendor.vendorFaultCode = 0x42U;
        vendor.vendorFaultData = 0x99AAU;
        mock.vendors = {vendor};

        VolkDeviceTable table{};
        table.vkGetDeviceFaultInfoEXT = &mockGetDeviceFaultInfoEXT;

        rachis::DeviceFaultReport report{};
        report.status = rachis::DeviceFaultStatus::hung;
        report.dredAvailable = false;
        std::string summary;

        gFaultMock = &mock;
        CHECK(rachis::vulkan::captureVulkanDeviceFaultInfo(fakeVk<VkDevice>(0x60), table, report, summary));
        gFaultMock = nullptr;

        CHECK(mock.callCount == 2);
        CHECK_FALSE(report.dredAvailable);

        CHECK_FALSE(report.allocationsTruncated);
        CHECK_FALSE(report.breadcrumbsTruncated);

        CHECK(report.pageFaultAddress == 0xDEAD'BEEF'1000ULL);
        REQUIRE(report.allocations.size() == 2);
        CHECK(report.allocations[0].typeName == "instructionPointerUnknown");
        CHECK(report.allocations[0].label.find("address=0x10001200") != std::string::npos);
        CHECK(report.allocations[0].label.find("precision=0x100") != std::string::npos);
        CHECK(report.allocations[1].typeName == "writeInvalid");
        CHECK(report.allocations[1].label.find("address=0xdeadbeef1000") != std::string::npos);
        CHECK(report.allocations[1].label.find("precision=0x1000") != std::string::npos);

        REQUIRE(report.breadcrumbs.size() == 1);
        CHECK(report.breadcrumbs[0].queueLabel == "VK_EXT_device_fault");
        CHECK(report.breadcrumbs[0].commandListLabel == "GPU page fault in compute dispatch");
        REQUIRE(report.breadcrumbs[0].operations.size() == 1);
        CHECK(report.breadcrumbs[0].operations[0].name == "MMU write fault");
        CHECK(report.breadcrumbs[0].operations[0].context.find("code=0x42") != std::string::npos);
        CHECK(summary.find("GPU page fault in compute dispatch") != std::string::npos);
        CHECK(summary.find("0xdeadbeef1000") != std::string::npos);
    }

    SECTION("genuine VK_INCOMPLETE array truncation sets allocationsTruncated and breadcrumbsTruncated")
    {
        DeviceFaultMockState mock;
        mock.reportedAddressCount = 3;
        mock.reportedVendorCount = 2;
        mock.secondCallAddressCap = 1;
        mock.secondCallVendorCap = 1;
        mock.secondCallResult = VK_INCOMPLETE;
        mock.addresses = {
            VkDeviceFaultAddressInfoEXT{
                .addressType = VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT,
                .reportedAddress = 0x2000'0008ULL,
                .addressPrecision = 16,
            },
            VkDeviceFaultAddressInfoEXT{},
        };
        VkDeviceFaultVendorInfoEXT vendor{};
        std::snprintf(vendor.description, sizeof(vendor.description), "%s", "Truncated vendor fault");
        mock.vendors = {vendor, vendor};

        VolkDeviceTable table{};
        table.vkGetDeviceFaultInfoEXT = &mockGetDeviceFaultInfoEXT;
        rachis::DeviceFaultReport report{};
        std::string summary;

        gFaultMock = &mock;
        CHECK(rachis::vulkan::captureVulkanDeviceFaultInfo(fakeVk<VkDevice>(0x61), table, report, summary));
        gFaultMock = nullptr;

        CHECK(report.allocationsTruncated);
        CHECK(report.breadcrumbsTruncated);
        CHECK(report.pageFaultAddress == 0x2000'0000ULL);
    }

    SECTION("VulkanDevice driver loss publishes VK_EXT_device_fault report, error summary, and diagnostic")
    {
        DeviceFaultMockState mock;
        mock.reportedAddressCount = 1;
        mock.reportedVendorCount = 0;
        mock.description = "Page fault reading unbound descriptor";
        mock.addresses = {
            VkDeviceFaultAddressInfoEXT{
                .addressType = VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT,
                .reportedAddress = 0xCAFE'0044ULL,
                .addressPrecision = 0x100ULL,
            },
        };

        VolkDeviceTable table{};
        table.vkGetDeviceFaultInfoEXT = &mockGetDeviceFaultInfoEXT;
        table.vkCreateCommandPool = &mockLostCreateCommandPool;
        table.vkAllocateCommandBuffers = &presentMockAllocateCommandBuffers;

        rachis::vulkan::VulkanPhysicalDeviceRecord record{};
        record.physical = fakeVk<VkPhysicalDevice>(0x70);
        record.deviceName = "Fault Test Adapter";
        record.deviceFault = true;
        record.queues = {.graphics = 0, .compute = 0, .transfer = 0};

        std::vector<std::string> diagnosticMessages;
        rachis::vulkan::VulkanInstanceInfo instanceInfo{};
        instanceInfo.sink = std::make_unique<rachis::vulkan::VulkanDiagnosticSink>();
        instanceInfo.sink->callback = [](const rachis::DiagnosticMessage& msg, void* userData) noexcept {
            if (msg.severity == rachis::DiagnosticSeverity::error)
            {
                static_cast<std::vector<std::string>*>(userData)->emplace_back(msg.description);
            }
        };
        instanceInfo.sink->userData = &diagnosticMessages;

        rachis::DeviceCapabilities caps{};
        const rachis::vulkan::VulkanQueueHandles queues{
            .graphics = fakeVk<VkQueue>(0x71),
            .compute = fakeVk<VkQueue>(0x71),
            .copy = fakeVk<VkQueue>(0x71),
            .families = record.queues,
            .deviceFaultEnabled = true,
        };

        gFaultMock = &mock;
        {
            rachis::vulkan::VulkanDevice dev(record, fakeVk<VkDevice>(0x72), table, queues, caps, 1);
            dev.adoptInstance(std::move(instanceInfo));

            const auto poolResult =
                dev.createCommandPool({.queueType = rachis::QueueType::graphics, .maxCommandBuffers = 4});
            REQUIRE_FALSE(poolResult.has_value());
            CHECK(poolResult.error().code == rachis::ErrorCode::deviceLost);
            CHECK(dev.operationalState() == rachis::DeviceOperationalState::lost);

            const auto lossErr = dev.deviceLossError();
            REQUIRE(lossErr.has_value());
            CHECK(lossErr->message.find("Page fault reading unbound descriptor") != std::string::npos);
            CHECK(lossErr->message.find("0xcafe0000") != std::string::npos);

            const auto faultReport = dev.captureDeviceFaultReport();
            REQUIRE(faultReport.has_value());
            CHECK_FALSE(faultReport->dredAvailable);
            CHECK(faultReport->pageFaultAddress == 0xCAFE'0000ULL);
            REQUIRE(faultReport->allocations.size() == 1);

            REQUIRE(diagnosticMessages.size() == 1);
            CHECK(diagnosticMessages[0].find("VK_EXT_device_fault") != std::string::npos);
        }
        gFaultMock = nullptr;
    }
}

namespace
{

    constexpr std::array<std::uint32_t, 453> kSlangMultiEntryComputeSpirv = {
        0x07230203U, 0x00010500U, 0x00280000U, 0x0000004dU, 0x00000000U, 0x00020011U, 0x00000001U, 0x000b000aU,
        0x5f565053U, 0x5f52484bU, 0x726f7473U, 0x5f656761U, 0x66667562U, 0x735f7265U, 0x61726f74U, 0x635f6567U,
        0x7373616cU, 0x00000000U, 0x0003000eU, 0x00000000U, 0x00000001U, 0x000a000fU, 0x00000005U, 0x00000002U,
        0x706d6f63U, 0x41657475U, 0x6168706cU, 0x00000000U, 0x00000012U, 0x0000001dU, 0x00000009U, 0x000a000fU,
        0x00000005U, 0x00000028U, 0x706d6f63U, 0x42657475U, 0x00617465U, 0x00000033U, 0x00000037U, 0x0000003bU,
        0x00000009U, 0x00060010U, 0x00000002U, 0x00000011U, 0x00000008U, 0x00000004U, 0x00000001U, 0x00060010U,
        0x00000028U, 0x00000011U, 0x00000010U, 0x00000002U, 0x00000001U, 0x00030003U, 0x0000000bU, 0x00000001U,
        0x00070005U, 0x0000000fU, 0x74535752U, 0x74637572U, 0x64657275U, 0x66667542U, 0x00007265U, 0x00060006U,
        0x0000000fU, 0x00000000U, 0x656d5f5fU, 0x7265626dU, 0x00000030U, 0x00040005U, 0x00000012U, 0x74754f67U,
        0x00000041U, 0x00060005U, 0x0000001aU, 0x68737550U, 0x61746144U, 0x6474735fU, 0x00303334U, 0x00050006U,
        0x0000001aU, 0x00000000U, 0x756c6176U, 0x00007365U, 0x00040005U, 0x0000001dU, 0x73755067U, 0x00000068U,
        0x00060005U, 0x00000002U, 0x706d6f63U, 0x41657475U, 0x6168706cU, 0x00000000U, 0x00070005U, 0x00000030U,
        0x74535752U, 0x74637572U, 0x64657275U, 0x66667542U, 0x00007265U, 0x00060006U, 0x00000030U, 0x00000000U,
        0x656d5f5fU, 0x7265626dU, 0x00000030U, 0x00040005U, 0x00000033U, 0x74754f67U, 0x00000042U, 0x00040005U,
        0x00000037U, 0x78655467U, 0x00000042U, 0x00040005U, 0x0000003bU, 0x6d615367U, 0x00004270U, 0x00050005U,
        0x0000003eU, 0x61735f5fU, 0x656c706dU, 0x00000064U, 0x00050005U, 0x00000028U, 0x706d6f63U, 0x42657475U,
        0x00617465U, 0x00040047U, 0x00000009U, 0x0000000bU, 0x0000001cU, 0x00040047U, 0x0000000dU, 0x00000006U,
        0x00000004U, 0x00040047U, 0x00000010U, 0x00000006U, 0x00000004U, 0x00030047U, 0x0000000fU, 0x00000002U,
        0x00050048U, 0x0000000fU, 0x00000000U, 0x00000023U, 0x00000000U, 0x00040047U, 0x00000012U, 0x00000021U,
        0x00000000U, 0x00040047U, 0x00000012U, 0x00000022U, 0x00000000U, 0x00030047U, 0x0000001aU, 0x00000002U,
        0x00050048U, 0x0000001aU, 0x00000000U, 0x00000023U, 0x00000000U, 0x00040047U, 0x0000002eU, 0x00000006U,
        0x00000010U, 0x00040047U, 0x00000031U, 0x00000006U, 0x00000010U, 0x00030047U, 0x00000030U, 0x00000002U,
        0x00050048U, 0x00000030U, 0x00000000U, 0x00000023U, 0x00000000U, 0x00040047U, 0x00000033U, 0x00000021U,
        0x00000003U, 0x00040047U, 0x00000033U, 0x00000022U, 0x00000000U, 0x00040047U, 0x00000037U, 0x00000021U,
        0x00000001U, 0x00040047U, 0x00000037U, 0x00000022U, 0x00000000U, 0x00040047U, 0x0000003bU, 0x00000021U,
        0x00000002U, 0x00040047U, 0x0000003bU, 0x00000022U, 0x00000000U, 0x00020013U, 0x00000001U, 0x00030021U,
        0x00000003U, 0x00000001U, 0x00040015U, 0x00000005U, 0x00000020U, 0x00000000U, 0x00040017U, 0x00000006U,
        0x00000005U, 0x00000003U, 0x00040020U, 0x00000008U, 0x00000001U, 0x00000006U, 0x00040015U, 0x0000000bU,
        0x00000020U, 0x00000001U, 0x0004002bU, 0x0000000bU, 0x0000000cU, 0x00000000U, 0x00040020U, 0x0000000dU,
        0x0000000cU, 0x00000005U, 0x0003001dU, 0x00000010U, 0x00000005U, 0x0003001eU, 0x0000000fU, 0x00000010U,
        0x00040020U, 0x00000011U, 0x0000000cU, 0x0000000fU, 0x00040017U, 0x0000001bU, 0x00000005U, 0x00000004U,
        0x0003001eU, 0x0000001aU, 0x0000001bU, 0x00040020U, 0x0000001cU, 0x00000009U, 0x0000001aU, 0x00040020U,
        0x0000001eU, 0x00000009U, 0x0000001bU, 0x0004002bU, 0x00000005U, 0x00000021U, 0x00000003U, 0x00040020U,
        0x00000022U, 0x00000009U, 0x00000005U, 0x00030016U, 0x0000002cU, 0x00000020U, 0x00040017U, 0x0000002dU,
        0x0000002cU, 0x00000004U, 0x00040020U, 0x0000002eU, 0x0000000cU, 0x0000002dU, 0x0003001dU, 0x00000031U,
        0x0000002dU, 0x0003001eU, 0x00000030U, 0x00000031U, 0x00040020U, 0x00000032U, 0x0000000cU, 0x00000030U,
        0x00090019U, 0x00000034U, 0x0000002cU, 0x00000001U, 0x00000002U, 0x00000000U, 0x00000000U, 0x00000001U,
        0x00000000U, 0x00040020U, 0x00000036U, 0x00000000U, 0x00000034U, 0x0002001aU, 0x00000038U, 0x00040020U,
        0x0000003aU, 0x00000000U, 0x00000038U, 0x0003001bU, 0x0000003cU, 0x00000034U, 0x00040017U, 0x0000003fU,
        0x0000002cU, 0x00000002U, 0x0004002bU, 0x0000002cU, 0x00000041U, 0x3f000000U, 0x0005002cU, 0x0000003fU,
        0x00000040U, 0x00000041U, 0x00000041U, 0x0004002bU, 0x0000002cU, 0x00000042U, 0x00000000U, 0x0004003bU,
        0x00000008U, 0x00000009U, 0x00000001U, 0x0004003bU, 0x00000011U, 0x00000012U, 0x0000000cU, 0x0004003bU,
        0x0000001cU, 0x0000001dU, 0x00000009U, 0x0004003bU, 0x00000032U, 0x00000033U, 0x0000000cU, 0x0004003bU,
        0x00000036U, 0x00000037U, 0x00000000U, 0x0004003bU, 0x0000003aU, 0x0000003bU, 0x00000000U, 0x00050036U,
        0x00000001U, 0x00000002U, 0x00000000U, 0x00000003U, 0x000200f8U, 0x00000004U, 0x0004003dU, 0x00000006U,
        0x00000007U, 0x00000009U, 0x00050051U, 0x00000005U, 0x0000000aU, 0x00000007U, 0x00000000U, 0x00060041U,
        0x0000000dU, 0x0000000eU, 0x00000012U, 0x0000000cU, 0x0000000aU, 0x0004003dU, 0x00000006U, 0x00000013U,
        0x00000009U, 0x00050051U, 0x00000005U, 0x00000014U, 0x00000013U, 0x00000000U, 0x00050041U, 0x0000001eU,
        0x00000049U, 0x0000001dU, 0x0000000cU, 0x000500c7U, 0x00000005U, 0x0000004aU, 0x00000014U, 0x00000021U,
        0x00050041U, 0x00000022U, 0x0000004bU, 0x00000049U, 0x0000004aU, 0x0004003dU, 0x00000005U, 0x0000004cU,
        0x0000004bU, 0x0003003eU, 0x0000000eU, 0x0000004cU, 0x000100fdU, 0x00010038U, 0x00050036U, 0x00000001U,
        0x00000028U, 0x00000000U, 0x00000003U, 0x000200f8U, 0x00000029U, 0x0004003dU, 0x00000006U, 0x0000002aU,
        0x00000009U, 0x00050051U, 0x00000005U, 0x0000002bU, 0x0000002aU, 0x00000000U, 0x00060041U, 0x0000002eU,
        0x0000002fU, 0x00000033U, 0x0000000cU, 0x0000002bU, 0x0004003dU, 0x00000034U, 0x00000035U, 0x00000037U,
        0x0004003dU, 0x00000038U, 0x00000039U, 0x0000003bU, 0x00050056U, 0x0000003cU, 0x0000003dU, 0x00000035U,
        0x00000039U, 0x00070058U, 0x0000002dU, 0x0000003eU, 0x0000003dU, 0x00000040U, 0x00000002U, 0x00000042U,
        0x0003003eU, 0x0000002fU, 0x0000003eU, 0x000100fdU, 0x00010038U,
    };

    struct SpirvBuilder final
    {
        std::vector<std::uint32_t> words{0x07230203U, 0x00010500U, 0U, 128U, 0U};

        void op(std::uint16_t opcode, std::initializer_list<std::uint32_t> operands)
        {
            const auto wordCount = static_cast<std::uint32_t>(1 + operands.size());
            words.push_back((wordCount << 16U) | opcode);
            words.insert(words.end(), operands.begin(), operands.end());
        }

        void extInstImport(std::uint32_t resultId, std::string_view name)
        {
            std::vector<std::uint32_t> nameWords((name.size() + 4U) / 4U, 0U);
            std::memcpy(nameWords.data(), name.data(), std::min(name.size(), nameWords.size() * sizeof(std::uint32_t)));
            const auto wordCount = static_cast<std::uint32_t>(2U + nameWords.size());
            words.push_back((wordCount << 16U) | 11U);
            words.push_back(resultId);
            words.insert(words.end(), nameWords.begin(), nameWords.end());
        }

        void entryPoint(std::uint32_t execModel,
                        std::uint32_t funcId,
                        std::string_view name,
                        std::initializer_list<std::uint32_t> interfaceIds)
        {
            std::vector<std::uint32_t> nameWords((name.size() + 4U) / 4U, 0U);
            std::memcpy(nameWords.data(), name.data(), std::min(name.size(), nameWords.size() * sizeof(std::uint32_t)));
            const auto wordCount = static_cast<std::uint32_t>(3U + nameWords.size() + interfaceIds.size());
            words.push_back((wordCount << 16U) | 15U);
            words.push_back(execModel);
            words.push_back(funcId);
            words.insert(words.end(), nameWords.begin(), nameWords.end());
            words.insert(words.end(), interfaceIds.begin(), interfaceIds.end());
        }

        [[nodiscard]] std::span<const std::byte> bytes() const noexcept
        {
            return std::as_bytes(std::span<const std::uint32_t>(words));
        }
    };

    struct AsyncPsoMockState final
    {
        std::mutex mutex;
        std::uintptr_t nextHandle = 0x9000;
        bool gatePipelineCreate = false;
        std::binary_semaphore enteredCreate{0};
        std::binary_semaphore releaseCreate{0};
        std::atomic<std::uint32_t> createdPipelines{0};
        std::atomic<std::uint32_t> destroyedPipelines{0};
    };

    thread_local AsyncPsoMockState* gPsoMock = nullptr;
    AsyncPsoMockState* gSharedPsoMock = nullptr;

    [[nodiscard]] AsyncPsoMockState* activePsoMock() noexcept
    {
        return gPsoMock != nullptr ? gPsoMock : gSharedPsoMock;
    }

    VKAPI_ATTR VkResult VKAPI_CALL psoMockCreateShaderModule(VkDevice,
                                                             const VkShaderModuleCreateInfo*,
                                                             const VkAllocationCallbacks*,
                                                             VkShaderModule* out)
    {
        auto* m = activePsoMock();
        const std::scoped_lock lock(m->mutex);
        *out = fakeVk<VkShaderModule>(m->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL psoMockDestroyShaderModule(VkDevice, VkShaderModule, const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL psoMockCreateDescriptorSetLayout(VkDevice,
                                                                    const VkDescriptorSetLayoutCreateInfo*,
                                                                    const VkAllocationCallbacks*,
                                                                    VkDescriptorSetLayout* out)
    {
        auto* m = activePsoMock();
        const std::scoped_lock lock(m->mutex);
        *out = fakeVk<VkDescriptorSetLayout>(m->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL psoMockDestroyDescriptorSetLayout(VkDevice,
                                                                 VkDescriptorSetLayout,
                                                                 const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL psoMockCreatePipelineLayout(VkDevice,
                                                               const VkPipelineLayoutCreateInfo*,
                                                               const VkAllocationCallbacks*,
                                                               VkPipelineLayout* out)
    {
        auto* m = activePsoMock();
        const std::scoped_lock lock(m->mutex);
        *out = fakeVk<VkPipelineLayout>(m->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL psoMockDestroyPipelineLayout(VkDevice, VkPipelineLayout, const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL psoMockCreateComputePipelines(VkDevice,
                                                                 VkPipelineCache,
                                                                 std::uint32_t count,
                                                                 const VkComputePipelineCreateInfo*,
                                                                 const VkAllocationCallbacks*,
                                                                 VkPipeline* out)
    {
        auto* m = activePsoMock();
        if (m->gatePipelineCreate)
        {
            m->enteredCreate.release();
            m->releaseCreate.acquire();
        }
        const std::scoped_lock lock(m->mutex);
        for (std::uint32_t i = 0; i < count; ++i)
        {
            out[i] = fakeVk<VkPipeline>(m->nextHandle++);
            m->createdPipelines.fetch_add(1, std::memory_order_relaxed);
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL psoMockCreateGraphicsPipelines(VkDevice,
                                                                  VkPipelineCache,
                                                                  std::uint32_t count,
                                                                  const VkGraphicsPipelineCreateInfo*,
                                                                  const VkAllocationCallbacks*,
                                                                  VkPipeline* out)
    {
        auto* m = activePsoMock();
        if (m->gatePipelineCreate)
        {
            m->enteredCreate.release();
            m->releaseCreate.acquire();
        }
        const std::scoped_lock lock(m->mutex);
        for (std::uint32_t i = 0; i < count; ++i)
        {
            out[i] = fakeVk<VkPipeline>(m->nextHandle++);
            m->createdPipelines.fetch_add(1, std::memory_order_relaxed);
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL psoMockDestroyPipeline(VkDevice, VkPipeline pipeline, const VkAllocationCallbacks*)
    {
        if (pipeline != VK_NULL_HANDLE)
        {
            activePsoMock()->destroyedPipelines.fetch_add(1, std::memory_order_relaxed);
        }
    }

    VKAPI_ATTR void VKAPI_CALL psoMockGetFormatProps(VkPhysicalDevice, VkFormat, VkFormatProperties* out)
    {
        out->linearTilingFeatures = 0;
        out->optimalTilingFeatures = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                                     VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT |
                                     VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
        out->bufferFeatures = 0;
    }

    VKAPI_ATTR VkResult VKAPI_CALL psoMockGetImageFormatProps(VkPhysicalDevice,
                                                              VkFormat,
                                                              VkImageType,
                                                              VkImageTiling,
                                                              VkImageUsageFlags,
                                                              VkImageCreateFlags,
                                                              VkImageFormatProperties* out)
    {
        *out = VkImageFormatProperties{
            .maxExtent = {4096, 4096, 256},
            .maxMipLevels = 12,
            .maxArrayLayers = 256,
            .sampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
            .maxResourceSize = 1ULL << 30,
        };
        return VK_SUCCESS;
    }

    [[nodiscard]] rachis::DeviceCapabilities baselineShaderTestCapabilities() noexcept
    {
        rachis::DeviceCapabilities caps{};
        caps.limits.maxTextureDimension2D = 16384;
        caps.limits.maxComputeWorkgroupInvocations = 1024;
        caps.limits.maxComputeWorkgroupSize = {1024, 1024, 64};
        caps.limits.maxComputeWorkgroupCount = {65535, 65535, 65535};
        caps.limits.maxBufferSize = 1ULL << 30;
        caps.limits.maxUniformBufferSize = 65536;
        caps.limits.maxColorAttachments = 8;
        caps.limits.maxVertexBuffers = 16;
        caps.limits.maxVertexAttributes = 16;
        caps.limits.maxBindingSets = 8;
        caps.limits.maxPushConstantBytes = 128;
        return caps;
    }

} // namespace

TEST_CASE("VulkanShaderManager scopes SPIR-V reflection to the requested (stage, entryPoint) and call graph",
          "[size:small][rhi][vulkan][spirv-reflection]")
{
    AsyncPsoMockState mock;
    gPsoMock = &mock;
    gSharedPsoMock = &mock;

    VolkDeviceTable table{};
    table.vkCreateShaderModule = &psoMockCreateShaderModule;
    table.vkDestroyShaderModule = &psoMockDestroyShaderModule;

    rachis::vulkan::VulkanShaderManager shaders;
    shaders.initialize(fakeVk<VkDevice>(0x90), &table, baselineShaderTestCapabilities(), nullptr, true);

    const auto hasWorkgroup =
        [](const std::optional<rachis::ShaderWorkgroupSize>& wg, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
            return wg.has_value() && wg->x == x && wg->y == y && wg->z == z;
        };
    const auto matchesBinding = [](const rachis::ShaderBindingRequirement& b,
                                   std::uint32_t set,
                                   std::uint32_t binding,
                                   rachis::ShaderResourceType type,
                                   std::uint32_t count = 1) {
        return b.set == set && b.binding == binding && b.type == type && b.count == count;
    };

    SECTION("real slangc multi-entry-point compute SPIR-V isolates bindings, push constants, and workgroup sizes")
    {
        const auto spvBytes = std::as_bytes(std::span(kSlangMultiEntryComputeSpirv));

        const std::array alphaBindings{
            rachis::ShaderBindingRequirement{
                .set = 0,
                .binding = 0,
                .type = rachis::ShaderResourceType::storageBuffer,
                .count = 1,
            },
        };
        const std::array alphaPush{rachis::ShaderPushConstantRequirement{.offset = 0, .size = 16}};
        const auto alphaMod = shaders.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = spvBytes,
            .entryPoint = "computeAlpha",
            .shaderInterface =
                {
                    .bindings = alphaBindings,
                    .pushConstants = alphaPush,
                    .workgroupSize = rachis::ShaderWorkgroupSize{8, 4, 1},
                },
        });
        REQUIRE(alphaMod.has_value());
        const auto alphaSnap = shaders.shaderModuleInterface(*alphaMod);
        REQUIRE(alphaSnap.has_value());
        CHECK(hasWorkgroup(alphaSnap->workgroupSize, 8, 4, 1));
        const auto alphaLayout = shaders.queryLayout(*alphaMod);
        REQUIRE(alphaLayout.has_value());
        CHECK(alphaLayout->pushConstantBytes == 16);
        REQUIRE(alphaSnap->bindings.size() == 1);
        CHECK(matchesBinding(alphaSnap->bindings[0], 0, 0, rachis::ShaderResourceType::storageBuffer));

        const std::array betaBindings{
            rachis::ShaderBindingRequirement{
                .set = 0,
                .binding = 1,
                .type = rachis::ShaderResourceType::sampledTexture,
                .count = 1,
            },
            rachis::ShaderBindingRequirement{
                .set = 0,
                .binding = 2,
                .type = rachis::ShaderResourceType::sampler,
                .count = 1,
            },
            rachis::ShaderBindingRequirement{
                .set = 0,
                .binding = 3,
                .type = rachis::ShaderResourceType::storageBuffer,
                .count = 1,
            },
        };
        const auto betaMod = shaders.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = spvBytes,
            .entryPoint = "computeBeta",
            .shaderInterface =
                {
                    .bindings = betaBindings,
                    .workgroupSize = rachis::ShaderWorkgroupSize{16, 2, 1},
                },
        });
        REQUIRE(betaMod.has_value());
        const auto betaSnap = shaders.shaderModuleInterface(*betaMod);
        REQUIRE(betaSnap.has_value());
        CHECK(hasWorkgroup(betaSnap->workgroupSize, 16, 2, 1));
        const auto betaLayout = shaders.queryLayout(*betaMod);
        REQUIRE(betaLayout.has_value());
        CHECK(betaLayout->pushConstantBytes == 0);
        REQUIRE(betaSnap->bindings.size() == 3);
        CHECK(matchesBinding(betaSnap->bindings[0], 0, 1, rachis::ShaderResourceType::sampledTexture));
        CHECK(matchesBinding(betaSnap->bindings[1], 0, 2, rachis::ShaderResourceType::sampler));
        CHECK(matchesBinding(betaSnap->bindings[2], 0, 3, rachis::ShaderResourceType::storageBuffer));

        const std::array unionBindings{alphaBindings[0], betaBindings[0], betaBindings[1], betaBindings[2]};
        const auto leakedUnion = shaders.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = spvBytes,
            .entryPoint = "computeAlpha",
            .shaderInterface =
                {
                    .bindings = unionBindings,
                    .pushConstants = alphaPush,
                    .workgroupSize = rachis::ShaderWorkgroupSize{8, 4, 1},
                },
        });
        REQUIRE_FALSE(leakedUnion.has_value());
        CHECK(leakedUnion.error().code == rachis::ErrorCode::invalidArgument);

        const auto wrongStage = shaders.createShaderModule({
            .stage = rachis::ShaderStage::vertex,
            .bytecode = spvBytes,
            .entryPoint = "computeAlpha",
        });
        REQUIRE_FALSE(wrongStage.has_value());
        CHECK(wrongStage.error().code == rachis::ErrorCode::invalidArgument);

        const auto missingEntry = shaders.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = spvBytes,
            .entryPoint = "doesNotExist",
            .shaderInterface =
                {
                    .bindings = alphaBindings,
                    .pushConstants = alphaPush,
                    .workgroupSize = rachis::ShaderWorkgroupSize{8, 4, 1},
                },
        });
        REQUIRE_FALSE(missingEntry.has_value());
        CHECK(missingEntry.error().code == rachis::ErrorCode::invalidArgument);
    }

    SECTION("OpFunctionCall transitive call graph, LocalSizeId, and RowMajor uniform blocks")
    {
        SpirvBuilder b;
        b.op(17, {1});
        b.op(14, {0, 1});
        b.entryPoint(5, 30, "entryCallChain", {});
        b.entryPoint(5, 40, "entryLocalSizeId", {});
        b.entryPoint(5, 50, "entryRowMajor", {});
        b.op(16, {30, 17, 4, 2, 2});
        b.op(331, {40, 38, 16, 17, 18});
        b.op(16, {50, 17, 1, 1, 1});

        b.op(71, {7, 2});
        b.op(72, {7, 0, 35, 0});
        b.op(71, {9, 34, 2});
        b.op(71, {9, 33, 5});

        b.op(71, {12, 34, 1});
        b.op(71, {12, 33, 7});

        b.op(71, {13, 2});
        b.op(72, {13, 0, 4});
        b.op(72, {13, 0, 35, 0});
        b.op(72, {13, 0, 7, 16});
        b.op(71, {15, 34, 0});
        b.op(71, {15, 33, 9});

        b.op(19, {1});
        b.op(33, {2, 1});
        b.op(21, {3, 32, 0});
        b.op(22, {4, 32});
        b.op(23, {5, 4, 4});
        b.op(24, {6, 5, 4});
        b.op(30, {7, 3});
        b.op(32, {8, 2, 7});
        b.op(59, {8, 9, 2});
        b.op(25, {10, 4, 1, 0, 0, 0, 2, 1});
        b.op(32, {11, 0, 10});
        b.op(59, {11, 12, 0});
        b.op(30, {13, 6});
        b.op(32, {14, 2, 13});
        b.op(59, {14, 15, 2});
        b.op(43, {3, 16, 32});
        b.op(43, {3, 17, 4});
        b.op(43, {3, 18, 1});
        b.op(43, {3, 19, 0});
        b.op(32, {20, 2, 3});
        b.op(32, {21, 2, 6});

        b.op(54, {1, 32, 0, 2});
        b.op(248, {60});
        b.op(65, {20, 61, 9, 19});
        b.op(61, {3, 62, 61});
        b.op(253, {});
        b.op(56, {});

        b.op(54, {1, 31, 0, 2});
        b.op(248, {63});
        b.op(57, {1, 64, 32});
        b.op(253, {});
        b.op(56, {});

        b.op(54, {1, 30, 0, 2});
        b.op(248, {65});
        b.op(57, {1, 66, 31});
        b.op(253, {});
        b.op(56, {});

        b.op(54, {1, 40, 0, 2});
        b.op(248, {67});
        b.op(61, {10, 68, 12});
        b.op(253, {});
        b.op(56, {});

        b.op(54, {1, 50, 0, 2});
        b.op(248, {69});
        b.op(65, {21, 70, 15, 19});
        b.op(253, {});
        b.op(56, {});

        const std::array chainBindings{
            rachis::ShaderBindingRequirement{
                .set = 2,
                .binding = 5,
                .type = rachis::ShaderResourceType::uniformBuffer,
                .count = 1,
            },
        };
        const auto chainMod = shaders.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = b.bytes(),
            .entryPoint = "entryCallChain",
            .shaderInterface =
                {
                    .bindings = chainBindings,
                    .workgroupSize = rachis::ShaderWorkgroupSize{4, 2, 2},
                },
        });
        REQUIRE(chainMod.has_value());
        const auto chainSnap = shaders.shaderModuleInterface(*chainMod);
        REQUIRE(chainSnap.has_value());
        CHECK(hasWorkgroup(chainSnap->workgroupSize, 4, 2, 2));
        REQUIRE(chainSnap->bindings.size() == 1);
        CHECK(matchesBinding(chainSnap->bindings[0], 2, 5, rachis::ShaderResourceType::uniformBuffer));

        const std::array sizeIdBindings{
            rachis::ShaderBindingRequirement{
                .set = 1,
                .binding = 7,
                .type = rachis::ShaderResourceType::storageTexture,
                .count = 1,
            },
        };
        const auto sizeIdMod = shaders.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = b.bytes(),
            .entryPoint = "entryLocalSizeId",
            .shaderInterface =
                {
                    .bindings = sizeIdBindings,
                    .workgroupSize = rachis::ShaderWorkgroupSize{32, 4, 1},
                },
        });
        REQUIRE(sizeIdMod.has_value());
        const auto sizeIdSnap = shaders.shaderModuleInterface(*sizeIdMod);
        REQUIRE(sizeIdSnap.has_value());
        CHECK(hasWorkgroup(sizeIdSnap->workgroupSize, 32, 4, 1));
        REQUIRE(sizeIdSnap->bindings.size() == 1);
        CHECK(matchesBinding(sizeIdSnap->bindings[0], 1, 7, rachis::ShaderResourceType::storageTexture));

        const std::array rowMajorBindings{
            rachis::ShaderBindingRequirement{
                .set = 0,
                .binding = 9,
                .type = rachis::ShaderResourceType::uniformBuffer,
                .count = 1,
            },
        };
        const auto rowMajorMod = shaders.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = b.bytes(),
            .entryPoint = "entryRowMajor",
            .shaderInterface =
                {
                    .bindings = rowMajorBindings,
                    .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1},
                },
        });
        if (!rowMajorMod)
        {
            UNSCOPED_INFO(rowMajorMod.error().message);
        }
        REQUIRE(rowMajorMod.has_value());
        const auto rowMajorSnap = shaders.shaderModuleInterface(*rowMajorMod);
        REQUIRE(rowMajorSnap.has_value());
        CHECK(hasWorkgroup(rowMajorSnap->workgroupSize, 1, 1, 1));
        REQUIRE(rowMajorSnap->bindings.size() == 1);
        CHECK(matchesBinding(rowMajorSnap->bindings[0], 0, 9, rachis::ShaderResourceType::uniformBuffer));

        rachis::vulkan::VulkanShaderManager shadersNoM4;
        shadersNoM4.initialize(fakeVk<VkDevice>(0x91), &table, baselineShaderTestCapabilities(), nullptr, false);
        const auto noM4Mod = shadersNoM4.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = b.bytes(),
            .entryPoint = "entryLocalSizeId",
            .shaderInterface =
                {
                    .bindings = sizeIdBindings,
                    .workgroupSize = rachis::ShaderWorkgroupSize{32, 4, 1},
                },
        });
        REQUIRE_FALSE(noM4Mod.has_value());
        CHECK(noM4Mod.error().code == rachis::ErrorCode::unsupported);
        shadersNoM4.shutdown();
    }

    SECTION("LocalSizeId resolves library specialization overrides and OpSpecConstantOp expressions")
    {
        SpirvBuilder b;
        b.op(17, {1});
        b.op(14, {0, 1});
        b.entryPoint(5, 20, "main", {});
        b.op(331, {20, 38, 12, 11, 14});
        b.op(71, {10, 1, 0});
        b.op(71, {11, 1, 1});
        b.op(19, {1});
        b.op(33, {2, 1});
        b.op(21, {3, 32, 0});
        b.op(50, {3, 10, 8});
        b.op(50, {3, 11, 2});
        b.op(43, {3, 13, 0});
        b.op(43, {3, 14, 1});
        b.op(52, {3, 12, 128, 10, 13});
        b.op(54, {1, 20, 0, 2});
        b.op(248, {21});
        b.op(253, {});
        b.op(56, {});

        const std::array override16x4{
            rachis::SpecializationConstant::makeUnsignedInt32(0, 16),
            rachis::SpecializationConstant::makeUnsignedInt32(1, 4),
        };
        const std::array zeroOverride{
            rachis::SpecializationConstant::makeUnsignedInt32(0, 0),
        };
        const std::array zeroEntries{
            rachis::ShaderLibraryEntryDesc{
                .stage = rachis::ShaderStage::compute,
                .bytecode = b.bytes(),
                .entryPoint = "main",
                .specializationConstants = zeroOverride,
            },
        };
        CHECK_FALSE(shaders.createShaderLibrary({.entries = zeroEntries}).has_value());

        const std::array libEntries{
            rachis::ShaderLibraryEntryDesc{
                .stage = rachis::ShaderStage::compute,
                .bytecode = b.bytes(),
                .entryPoint = "main",
                .specializationConstants = override16x4,
            },
        };
        const auto lib = shaders.createShaderLibrary({.entries = libEntries});
        REQUIRE(lib.has_value());

        const auto staleDefaultMod = shaders.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .entryPoint = "main",
            .library = *lib,
            .shaderInterface = {.workgroupSize = rachis::ShaderWorkgroupSize{8, 2, 1}},
            .specializationConstants = override16x4,
        });
        REQUIRE_FALSE(staleDefaultMod.has_value());
        CHECK(staleDefaultMod.error().code == rachis::ErrorCode::invalidArgument);

        const auto specializedMod = shaders.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .entryPoint = "main",
            .library = *lib,
            .shaderInterface = {.workgroupSize = rachis::ShaderWorkgroupSize{16, 4, 1}},
            .specializationConstants = override16x4,
        });
        REQUIRE(specializedMod.has_value());
        const auto specializedSnap = shaders.shaderModuleInterface(*specializedMod);
        REQUIRE(specializedSnap.has_value());
        CHECK(hasWorkgroup(specializedSnap->workgroupSize, 16, 4, 1));

        SpirvBuilder sb;
        sb.op(17, {1});
        sb.op(14, {0, 1});
        sb.entryPoint(5, 30, "main", {});
        sb.op(331, {30, 38, 12, 19, 22});
        sb.op(71, {10, 1, 0});
        sb.op(71, {11, 1, 1});
        sb.op(71, {13, 1, 2});
        sb.op(19, {1});
        sb.op(33, {2, 1});
        sb.op(21, {3, 32, 1});
        sb.op(21, {4, 32, 0});
        sb.op(23, {5, 3, 2});
        sb.op(50, {3, 10, std::bit_cast<std::uint32_t>(std::int32_t{-32})});
        sb.op(50, {3, 11, std::bit_cast<std::uint32_t>(std::int32_t{-2})});
        sb.op(50, {3, 13, std::bit_cast<std::uint32_t>(std::int32_t{-16})});
        sb.op(43, {4, 14, 2});
        sb.op(43, {3, 17, 1});
        sb.op(52, {3, 12, 135, 10, 11});
        sb.op(52, {3, 15, 195, 13, 14});
        sb.op(52, {3, 16, 126, 15});
        sb.op(51, {5, 18, 16, 17});
        sb.op(52, {3, 19, 81, 18, 0});
        sb.op(52, {3, 22, 81, 18, 1});
        sb.op(54, {1, 30, 0, 2});
        sb.op(248, {31});
        sb.op(253, {});
        sb.op(56, {});

        const auto signedDefaultMod = shaders.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = sb.bytes(),
            .entryPoint = "main",
            .shaderInterface = {.workgroupSize = rachis::ShaderWorkgroupSize{16, 4, 1}},
        });
        REQUIRE(signedDefaultMod.has_value());
        const auto signedSnap = shaders.shaderModuleInterface(*signedDefaultMod);
        REQUIRE(signedSnap.has_value());
        CHECK(hasWorkgroup(signedSnap->workgroupSize, 16, 4, 1));

        const std::array signedOverride{
            rachis::SpecializationConstant::makeSignedInt32(0, -64),
        };
        const std::array signedLibEntries{
            rachis::ShaderLibraryEntryDesc{
                .stage = rachis::ShaderStage::compute,
                .bytecode = sb.bytes(),
                .entryPoint = "main",
                .specializationConstants = signedOverride,
            },
        };
        const auto signedLib = shaders.createShaderLibrary({.entries = signedLibEntries});
        REQUIRE(signedLib.has_value());
        const auto signedOverrideMod = shaders.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .entryPoint = "main",
            .library = *signedLib,
            .shaderInterface = {.workgroupSize = rachis::ShaderWorkgroupSize{32, 4, 1}},
            .specializationConstants = signedOverride,
        });
        REQUIRE(signedOverrideMod.has_value());
    }

    SECTION("linked multi-stage graphics module with Vertex 'main' and Fragment 'main' disambiguates by stage")
    {
        SpirvBuilder b;
        b.op(17, {1});
        b.op(14, {0, 1});
        b.entryPoint(0, 20, "main", {10, 11, 12});
        b.entryPoint(4, 21, "main", {13, 14});
        b.op(16, {21, 7});

        b.op(71, {10, 30, 0});
        b.op(71, {11, 30, 2});
        b.op(71, {12, 11, 0});
        b.op(71, {13, 30, 2});
        b.op(71, {14, 30, 0});
        b.op(71, {7, 2});
        b.op(72, {7, 0, 35, 0});
        b.op(71, {9, 34, 0});
        b.op(71, {9, 33, 0});
        b.op(71, {17, 34, 1});
        b.op(71, {17, 33, 0});

        b.op(19, {1});
        b.op(33, {2, 1});
        b.op(22, {3, 32});
        b.op(23, {4, 3, 4});
        b.op(32, {5, 1, 4});
        b.op(32, {6, 3, 4});
        b.op(30, {7, 4});
        b.op(32, {8, 2, 7});
        b.op(59, {8, 9, 2});
        b.op(59, {5, 10, 1});
        b.op(59, {5, 11, 1});
        b.op(59, {6, 12, 3});
        b.op(59, {5, 13, 1});
        b.op(59, {6, 14, 3});
        b.op(25, {15, 3, 1, 0, 0, 0, 1, 0});
        b.op(32, {16, 0, 15});
        b.op(59, {16, 17, 0});

        b.op(54, {1, 20, 0, 2});
        b.op(248, {30});
        b.op(61, {7, 31, 9});
        b.op(253, {});
        b.op(56, {});

        b.op(54, {1, 21, 0, 2});
        b.op(248, {32});
        b.op(61, {15, 33, 17});
        b.op(253, {});
        b.op(56, {});

        const std::array vsBindings{
            rachis::ShaderBindingRequirement{
                .set = 0,
                .binding = 0,
                .type = rachis::ShaderResourceType::uniformBuffer,
                .count = 1,
            },
        };
        const std::array<std::uint32_t, 2> vsInputs{0, 2};
        const auto vsMod = shaders.createShaderModule({
            .stage = rachis::ShaderStage::vertex,
            .bytecode = b.bytes(),
            .entryPoint = "main",
            .shaderInterface =
                {
                    .bindings = vsBindings,
                    .vertexInputLocations = vsInputs,
                },
        });
        REQUIRE(vsMod.has_value());
        const auto vsSnap = shaders.shaderModuleInterface(*vsMod);
        REQUIRE(vsSnap.has_value());
        CHECK(vsSnap->vertexInputLocations == std::vector<std::uint32_t>{0, 2});
        REQUIRE(vsSnap->bindings.size() == 1);
        CHECK(matchesBinding(vsSnap->bindings[0], 0, 0, rachis::ShaderResourceType::uniformBuffer));

        const std::array fsBindings{
            rachis::ShaderBindingRequirement{
                .set = 1,
                .binding = 0,
                .type = rachis::ShaderResourceType::sampledTexture,
                .count = 1,
            },
        };
        const auto fsMod = shaders.createShaderModule({
            .stage = rachis::ShaderStage::pixel,
            .bytecode = b.bytes(),
            .entryPoint = "main",
            .shaderInterface =
                {
                    .bindings = fsBindings,
                },
        });
        REQUIRE(fsMod.has_value());
        const auto fsSnap = shaders.shaderModuleInterface(*fsMod);
        REQUIRE(fsSnap.has_value());
        CHECK(fsSnap->vertexInputLocations.empty());
        REQUIRE(fsSnap->bindings.size() == 1);
        CHECK(matchesBinding(fsSnap->bindings[0], 1, 0, rachis::ShaderResourceType::sampledTexture));
    }

    SECTION("literal GroupOperation, PackedVectorFormat, ImageSparseRead, and NonSemantic ExtInst operands do not "
            "collide with low descriptor IDs")
    {
        SpirvBuilder b;
        b.op(17, {1});
        b.op(17, {61});
        b.op(17, {63});
        b.extInstImport(9, "NonSemantic.Shader.DebugInfo.100");
        b.op(14, {0, 1});
        b.entryPoint(5, 50, "csMain", {});
        b.entryPoint(5, 51, "otherMain", {});
        b.op(16, {50, 17, 4, 1, 1});
        b.op(16, {51, 17, 1, 1, 1});

        b.op(71, {14, 2});
        b.op(72, {14, 0, 35, 0});
        b.op(71, {16, 2});
        b.op(72, {16, 0, 35, 0});
        b.op(71, {1, 34, 1});
        b.op(71, {1, 33, 1});
        b.op(71, {2, 34, 1});
        b.op(71, {2, 33, 2});
        b.op(71, {3, 34, 1});
        b.op(71, {3, 33, 3});
        b.op(71, {25, 34, 0});
        b.op(71, {25, 33, 0});

        b.op(19, {10});
        b.op(33, {11, 10});
        b.op(21, {12, 32, 0});
        b.op(23, {13, 12, 4});
        b.op(30, {14, 12});
        b.op(32, {15, 2, 14});
        b.op(30, {16, 12});
        b.op(32, {17, 12, 16});
        b.op(32, {18, 12, 12});
        b.op(43, {12, 20, 3});
        b.op(43, {12, 21, 0});
        b.op(43, {12, 22, 7});
        b.op(44, {13, 23, 22, 21, 21, 21});

        b.op(52, {13, 26, 79, 23, 23, 1, 2, 3, 0});
        b.op(52, {13, 27, 82, 22, 26, 2});
        b.op(52, {12, 28, 81, 27, 1});

        b.op(59, {15, 1, 2});
        b.op(59, {15, 2, 2});
        b.op(59, {15, 3, 2});
        b.op(59, {17, 25, 12});

        b.op(54, {10, 50, 0, 11});
        b.op(248, {60});

        b.op(12, {10, 61, 9, 1, 3});

        b.op(349, {12, 62, 20, 1, 28});

        b.op(342, {12, 63, 20, 2, 23});

        b.op(4454, {12, 64, 22, 22, 62, 2});

        b.op(65, {18, 65, 25, 21});
        b.op(240, {12, 66, 65, 20, 21, 64});
        b.op(253, {});
        b.op(56, {});

        b.op(54, {10, 51, 0, 11});
        b.op(248, {70});
        b.op(61, {14, 71, 1});
        b.op(61, {14, 72, 2});
        b.op(61, {14, 73, 3});
        b.op(253, {});
        b.op(56, {});

        auto subCaps = baselineShaderTestCapabilities();
        subCaps.subgroupOperations = true;
        rachis::vulkan::VulkanShaderManager subShaders;
        subShaders.initialize(fakeVk<VkDevice>(0x90), &table, subCaps, nullptr, true);

        const std::array csBindings{
            rachis::ShaderBindingRequirement{
                .set = 0,
                .binding = 0,
                .type = rachis::ShaderResourceType::storageBuffer,
                .count = 1,
            },
        };
        const auto csMod = subShaders.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = b.bytes(),
            .entryPoint = "csMain",
            .shaderInterface =
                {
                    .bindings = csBindings,
                    .workgroupSize = rachis::ShaderWorkgroupSize{4, 1, 1},
                },
        });
        REQUIRE(csMod.has_value());
        const auto csSnap = subShaders.shaderModuleInterface(*csMod);
        REQUIRE(csSnap.has_value());
        REQUIRE(csSnap->bindings.size() == 1);
        CHECK(matchesBinding(csSnap->bindings[0], 0, 0, rachis::ShaderResourceType::storageBuffer));
        subShaders.shutdown();
    }

    SECTION("duplicate (stage, entryPoint) OpEntryPoint declarations are rejected")
    {
        SpirvBuilder b;
        b.op(17, {1});
        b.op(14, {0, 1});
        b.entryPoint(5, 10, "dupMain", {});
        b.entryPoint(5, 11, "dupMain", {});
        b.op(16, {10, 17, 1, 1, 1});
        b.op(16, {11, 17, 1, 1, 1});
        b.op(19, {1});
        b.op(33, {2, 1});
        b.op(54, {1, 10, 0, 2});
        b.op(248, {20});
        b.op(253, {});
        b.op(56, {});
        b.op(54, {1, 11, 0, 2});
        b.op(248, {21});
        b.op(253, {});
        b.op(56, {});

        const auto dup = shaders.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = b.bytes(),
            .entryPoint = "dupMain",
            .shaderInterface =
                {
                    .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1},
                },
        });
        REQUIRE_FALSE(dup.has_value());
        CHECK(dup.error().code == rachis::ErrorCode::invalidArgument);
    }

    SECTION("self-referential buffer_reference storage structs visit each SPIR-V type at most once")
    {
        constexpr std::uint32_t fillerTypeCount = 64;
        constexpr std::uint64_t spirvTypeCount = fillerTypeCount + 9U;
        const auto bvhSceneModule = [](bool childArray) {
            SpirvBuilder b;
            b.op(17, {1});
            b.op(17, {5347});
            b.op(14, {5348, 1});
            b.entryPoint(5, 60, "main", {50});
            b.op(16, {60, 17, 1, 1, 1});
            b.op(71, {6, 2});
            b.op(72, {6, 0, 35, 0});
            b.op(72, {6, 1, 35, 8});
            b.op(71, {50, 34, 0});
            b.op(71, {50, 33, 0});
            b.op(71, {5, 2});
            b.op(72, {5, 0, 35, 0});
            if (childArray)
            {
                b.op(71, {9, 6, 8});
                b.op(72, {5, 1, 35, 16});
            }
            else
            {
                b.op(72, {5, 1, 35, 8});
                b.op(72, {5, 2, 35, 16});
            }
            b.op(39, {4, 5349});
            b.op(19, {1});
            b.op(33, {2, 1});
            b.op(21, {3, 32, 0});
            if (childArray)
            {
                b.op(43, {3, 10, 2});
                b.op(28, {9, 4, 10});
                b.op(30, {5, 9, 3});
            }
            else
            {
                b.op(30, {5, 4, 4, 3});
            }
            b.op(32, {4, 5349, 5});
            b.op(30, {6, 4, 3});
            b.op(32, {7, 12, 6});
            b.op(32, {8, 12, 4});
            for (std::uint32_t filler = 0; filler < fillerTypeCount; ++filler)
            {
                b.op(30, {64 + filler, 3});
            }
            b.op(43, {3, 11, 0});
            b.op(59, {7, 50, 12});
            b.op(54, {1, 60, 0, 2});
            b.op(248, {61});
            b.op(65, {8, 62, 50, 11});
            b.op(61, {4, 63, 62});
            b.op(253, {});
            b.op(56, {});
            return b;
        };
        const std::array sceneBindings{
            rachis::ShaderBindingRequirement{
                .set = 0,
                .binding = 0,
                .type = rachis::ShaderResourceType::storageBuffer,
                .count = 1,
            },
        };
        auto& typeVisits = rachis::vulkan::test::spirvReflectionTypeVisits();
        const auto checkScene = [&](bool childArray, std::uint32_t expectedStride) {
            const auto spirv = bvhSceneModule(childArray);
            const auto moduleVisitsBefore = typeVisits.load();
            const auto sceneMod = shaders.createShaderModule({
                .stage = rachis::ShaderStage::compute,
                .bytecode = spirv.bytes(),
                .entryPoint = "main",
                .shaderInterface =
                    {
                        .bindings = sceneBindings,
                        .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1},
                    },
            });
            REQUIRE(sceneMod.has_value());
            CHECK(typeVisits.load() - moduleVisitsBefore <= spirvTypeCount);
            const auto sceneLayout = shaders.queryLayout(*sceneMod);
            REQUIRE(sceneLayout.has_value());
            CHECK(sceneLayout->storageArrayStride == expectedStride);

            const std::array sceneEntries{
                rachis::ShaderLibraryEntryDesc{
                    .stage = rachis::ShaderStage::compute,
                    .bytecode = spirv.bytes(),
                    .entryPoint = "main",
                },
            };
            const auto libraryVisitsBefore = typeVisits.load();
            const auto sceneLib = shaders.createShaderLibrary({.entries = sceneEntries});
            REQUIRE(sceneLib.has_value());
            CHECK(typeVisits.load() - libraryVisitsBefore <= spirvTypeCount);
        };
        checkScene(false, 0);
        checkScene(true, 8);
    }

    SECTION("push-constant blocks of shared zero-size structs measure each SPIR-V type at most once")
    {
        constexpr std::uint32_t emptyStructLevels = 62;
        constexpr std::uint32_t firstLevelId = 64;
        constexpr std::uint64_t spirvTypeCount = emptyStructLevels + 7U;
        SpirvBuilder b;
        b.op(17, {1});
        b.op(14, {0, 1});
        b.entryPoint(5, 60, "main", {50});
        b.op(16, {60, 17, 1, 1, 1});
        b.op(71, {6, 2});
        b.op(72, {6, 0, 35, 0});
        b.op(72, {6, 1, 35, 0});
        for (std::uint32_t level = 1; level <= emptyStructLevels; ++level)
        {
            b.op(72, {firstLevelId + level, 0, 35, 0});
            b.op(72, {firstLevelId + level, 1, 35, 0});
        }
        b.op(19, {1});
        b.op(33, {2, 1});
        b.op(21, {3, 32, 0});
        b.op(30, {firstLevelId});
        for (std::uint32_t level = 1; level <= emptyStructLevels; ++level)
        {
            b.op(30, {firstLevelId + level, firstLevelId + level - 1, firstLevelId + level - 1});
        }
        b.op(30, {6, firstLevelId + emptyStructLevels, 3});
        b.op(32, {7, 9, 6});
        b.op(32, {8, 9, 3});
        b.op(43, {3, 11, 1});
        b.op(59, {7, 50, 9});
        b.op(54, {1, 60, 0, 2});
        b.op(248, {61});
        b.op(65, {8, 62, 50, 11});
        b.op(61, {3, 63, 62});
        b.op(253, {});
        b.op(56, {});

        auto& typeVisits = rachis::vulkan::test::spirvReflectionTypeVisits();
        const std::array push{rachis::ShaderPushConstantRequirement{.offset = 0, .size = 4}};
        const auto moduleVisitsBefore = typeVisits.load();
        const auto pushMod = shaders.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = b.bytes(),
            .entryPoint = "main",
            .shaderInterface =
                {
                    .pushConstants = push,
                    .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1},
                },
        });
        REQUIRE(pushMod.has_value());
        CHECK(typeVisits.load() - moduleVisitsBefore <= spirvTypeCount);
        const auto pushLayout = shaders.queryLayout(*pushMod);
        REQUIRE(pushLayout.has_value());
        CHECK(pushLayout->pushConstantBytes == 4);

        const std::array pushEntries{
            rachis::ShaderLibraryEntryDesc{
                .stage = rachis::ShaderStage::compute,
                .bytecode = b.bytes(),
                .entryPoint = "main",
            },
        };
        const auto libraryVisitsBefore = typeVisits.load();
        const auto pushLib = shaders.createShaderLibrary({.entries = pushEntries});
        REQUIRE(pushLib.has_value());
        CHECK(typeVisits.load() - libraryVisitsBefore <= spirvTypeCount);
    }

    SECTION("push-constant blocks measure RowMajor matrices by their rows and ColMajor matrices by their columns")
    {
        struct MatrixMember final
        {
            std::uint32_t offset = 0;
            std::uint32_t majorness = 0;
            bool array = false;
        };
        constexpr std::uint32_t rowMajor = 4;
        constexpr std::uint32_t colMajor = 5;
        const auto matrixPushModule =
            [](std::uint32_t columns, std::uint32_t rows, std::initializer_list<MatrixMember> members) {
                SpirvBuilder b;
                b.op(17, {1});
                b.op(14, {0, 1});
                b.entryPoint(5, 60, "main", {50});
                b.op(16, {60, 17, 1, 1, 1});
                b.op(71, {6, 2});
                b.op(71, {13, 6, 64});
                std::vector<std::uint32_t> structOperands{6};
                std::uint32_t member = 0;
                for (const auto& entry : members)
                {
                    b.op(72, {6, member, 35, entry.offset});
                    b.op(72, {6, member, 7, 16});
                    b.op(72, {6, member, entry.majorness});
                    structOperands.push_back(entry.array ? 13U : 5U);
                    ++member;
                }
                b.op(19, {1});
                b.op(33, {2, 1});
                b.op(22, {3, 32});
                b.op(23, {4, 3, rows});
                b.op(24, {5, 4, columns});
                b.op(21, {9, 32, 0});
                b.op(43, {9, 11, 0});
                b.op(43, {9, 12, 2});
                b.op(28, {13, 5, 12});
                b.words.push_back((static_cast<std::uint32_t>(1U + structOperands.size()) << 16U) | 30U);
                b.words.insert(b.words.end(), structOperands.begin(), structOperands.end());
                b.op(32, {7, 9, 6});
                b.op(32, {8, 9, structOperands[1]});
                b.op(59, {7, 50, 9});
                b.op(54, {1, 60, 0, 2});
                b.op(248, {61});
                b.op(65, {8, 62, 50, 11});
                b.op(61, {structOperands[1], 63, 62});
                b.op(253, {});
                b.op(56, {});
                return b;
            };
        const auto checkPushBytes = [&](const SpirvBuilder& spirv, std::uint32_t expected) {
            const auto create = [&](std::uint32_t declared) {
                const std::array push{rachis::ShaderPushConstantRequirement{.offset = 0, .size = declared}};
                return shaders.createShaderModule({
                    .stage = rachis::ShaderStage::compute,
                    .bytecode = spirv.bytes(),
                    .entryPoint = "main",
                    .shaderInterface =
                        {
                            .pushConstants = push,
                            .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1},
                        },
                });
            };
            const auto exact = create(expected);
            if (!exact)
            {
                UNSCOPED_INFO(exact.error().message);
            }
            REQUIRE(exact.has_value());
            const auto layout = shaders.queryLayout(*exact);
            REQUIRE(layout.has_value());
            CHECK(layout->pushConstantBytes == expected);
            REQUIRE(shaders.destroyShaderModule(*exact).has_value());
            const auto smaller = create(expected - 4);
            REQUIRE_FALSE(smaller.has_value());
            CHECK(smaller.error().code == rachis::ErrorCode::invalidArgument);
        };

        checkPushBytes(matrixPushModule(3, 4, {{.offset = 0, .majorness = rowMajor}}), 60);
        checkPushBytes(matrixPushModule(4, 3, {{.offset = 0, .majorness = rowMajor}}), 48);
        checkPushBytes(matrixPushModule(3, 4, {{.offset = 0, .majorness = colMajor}}), 48);
        checkPushBytes(matrixPushModule(4, 3, {{.offset = 0, .majorness = colMajor}}), 60);
        checkPushBytes(
            matrixPushModule(3, 4, {{.offset = 0, .majorness = rowMajor}, {.offset = 64, .majorness = colMajor}}), 112);
        checkPushBytes(
            matrixPushModule(3, 4, {{.offset = 0, .majorness = colMajor}, {.offset = 64, .majorness = rowMajor}}), 124);
        checkPushBytes(matrixPushModule(3, 4, {{.offset = 0, .majorness = rowMajor, .array = true}}), 124);
        checkPushBytes(matrixPushModule(3, 4, {{.offset = 0, .majorness = colMajor, .array = true}}), 112);
    }

    SECTION("type nesting past the SPIR-V structure nesting limit stops the reflection walks without exhausting "
            "the stack")
    {
        constexpr std::uint32_t firstLevelId = 64;
        const auto nestedBlockModule = [](std::uint32_t levels, bool pushConstant) {
            const std::uint32_t storageClass = pushConstant ? 9U : 12U;
            const std::uint32_t outerId = firstLevelId + levels;
            SpirvBuilder b;
            b.words[3] = outerId + 1U;
            b.op(17, {1});
            b.op(14, {0, 1});
            b.entryPoint(5, 60, "main", {50});
            b.op(16, {60, 17, 1, 1, 1});
            b.op(71, {outerId, 2});
            if (!pushConstant)
            {
                b.op(71, {50, 34, 0});
                b.op(71, {50, 33, 0});
                b.op(71, {4, 6, 4});
            }
            for (std::uint32_t level = 0; level <= levels; ++level)
            {
                b.op(72, {firstLevelId + level, 0, 35, 0});
            }
            b.op(19, {1});
            b.op(33, {2, 1});
            b.op(21, {3, 32, 0});
            if (pushConstant)
            {
                b.op(30, {firstLevelId, 3});
            }
            else
            {
                b.op(43, {3, 12, 4});
                b.op(28, {4, 3, 12});
                b.op(30, {firstLevelId, 4});
            }
            for (std::uint32_t level = 1; level <= levels; ++level)
            {
                b.op(30, {firstLevelId + level, firstLevelId + level - 1});
            }
            b.op(32, {5, storageClass, outerId});
            b.op(32, {7, storageClass, outerId - 1});
            b.op(43, {3, 11, 0});
            b.op(59, {5, 50, storageClass});
            b.op(54, {1, 60, 0, 2});
            b.op(248, {61});
            b.op(65, {7, 62, 50, 11});
            b.op(253, {});
            b.op(56, {});
            return b;
        };
        const std::array storageBindings{
            rachis::ShaderBindingRequirement{
                .set = 0,
                .binding = 0,
                .type = rachis::ShaderResourceType::storageBuffer,
                .count = 1,
            },
        };
        const std::array push{rachis::ShaderPushConstantRequirement{.offset = 0, .size = 4}};
        const auto createNested = [&](std::uint32_t levels, bool pushConstant) {
            const auto spirv = nestedBlockModule(levels, pushConstant);
            rachis::ShaderInterfaceDesc shaderInterface{.workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1}};
            if (pushConstant)
            {
                shaderInterface.pushConstants = push;
            }
            else
            {
                shaderInterface.bindings = storageBindings;
            }
            return shaders.createShaderModule({
                .stage = rachis::ShaderStage::compute,
                .bytecode = spirv.bytes(),
                .entryPoint = "main",
                .shaderInterface = shaderInterface,
            });
        };

        const auto shallowStorage = createNested(200, false);
        REQUIRE(shallowStorage.has_value());
        const auto shallowStorageLayout = shaders.queryLayout(*shallowStorage);
        REQUIRE(shallowStorageLayout.has_value());
        CHECK(shallowStorageLayout->storageArrayStride == 4);

        const auto shallowPush = createNested(200, true);
        REQUIRE(shallowPush.has_value());
        const auto shallowPushLayout = shaders.queryLayout(*shallowPush);
        REQUIRE(shallowPushLayout.has_value());
        CHECK(shallowPushLayout->pushConstantBytes == 4);

        constexpr std::uint32_t runawayLevels = 100000;
        const auto deepStorage = createNested(runawayLevels, false);
        REQUIRE(deepStorage.has_value());
        const auto deepStorageLayout = shaders.queryLayout(*deepStorage);
        REQUIRE(deepStorageLayout.has_value());
        CHECK(deepStorageLayout->storageArrayStride == 0);

        const auto deepPush = createNested(runawayLevels, true);
        REQUIRE_FALSE(deepPush.has_value());
        CHECK(deepPush.error().code == rachis::ErrorCode::invalidArgument);
    }

    shaders.shutdown();
    gSharedPsoMock = nullptr;
    gPsoMock = nullptr;
}

TEST_CASE("VulkanPipelineManager compiles compute and graphics PSOs asynchronously on background workers",
          "[size:small][rhi][vulkan][async-pso]")
{
    AsyncPsoMockState mock;
    gPsoMock = &mock;
    gSharedPsoMock = &mock;

    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetPhysicalDeviceFormatProperties = &psoMockGetFormatProps;
    instanceTable.vkGetPhysicalDeviceImageFormatProperties = &psoMockGetImageFormatProps;

    VolkDeviceTable table{};
    table.vkCreateShaderModule = &psoMockCreateShaderModule;
    table.vkDestroyShaderModule = &psoMockDestroyShaderModule;
    table.vkCreateDescriptorSetLayout = &psoMockCreateDescriptorSetLayout;
    table.vkDestroyDescriptorSetLayout = &psoMockDestroyDescriptorSetLayout;
    table.vkCreatePipelineLayout = &psoMockCreatePipelineLayout;
    table.vkDestroyPipelineLayout = &psoMockDestroyPipelineLayout;
    table.vkCreateComputePipelines = &psoMockCreateComputePipelines;
    table.vkCreateGraphicsPipelines = &psoMockCreateGraphicsPipelines;
    table.vkDestroyPipeline = &psoMockDestroyPipeline;

    const auto device = fakeVk<VkDevice>(0x95);
    const auto physical = fakeVk<VkPhysicalDevice>(0x96);
    const rachis::DeviceCapabilities caps = baselineShaderTestCapabilities();

    rachis::vulkan::VulkanShaderManager shaders;
    shaders.initialize(device, &table, caps, &instanceTable);
    rachis::vulkan::VulkanBindingManager bindings;
    bindings.initialize(device, &table, caps, &instanceTable);
    rachis::vulkan::VulkanPipelineManager pipelines;
    pipelines.initialize(device, &table, physical, caps, &shaders, &bindings, &instanceTable, true);

    SpirvBuilder csSpv;
    csSpv.op(17, {1});
    csSpv.op(14, {0, 1});
    csSpv.entryPoint(5, 10, "main", {});
    csSpv.op(16, {10, 17, 1, 1, 1});
    csSpv.op(19, {1});
    csSpv.op(33, {2, 1});
    csSpv.op(54, {1, 10, 0, 2});
    csSpv.op(248, {20});
    csSpv.op(253, {});
    csSpv.op(56, {});

    SpirvBuilder gfxSpv;
    gfxSpv.op(17, {1});
    gfxSpv.op(14, {0, 1});
    gfxSpv.entryPoint(0, 10, "vsMain", {});
    gfxSpv.entryPoint(4, 11, "psMain", {});
    gfxSpv.op(16, {11, 7});
    gfxSpv.op(19, {1});
    gfxSpv.op(33, {2, 1});
    gfxSpv.op(54, {1, 10, 0, 2});
    gfxSpv.op(248, {20});
    gfxSpv.op(253, {});
    gfxSpv.op(56, {});
    gfxSpv.op(54, {1, 11, 0, 2});
    gfxSpv.op(248, {21});
    gfxSpv.op(253, {});
    gfxSpv.op(56, {});

    SECTION("compute compilation runs off-thread, survives input handle destruction, and consumes compilation handle "
            "on wait")
    {
        const auto csMod = shaders.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = csSpv.bytes(),
            .entryPoint = "main",
            .shaderInterface =
                {
                    .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1},
                },
        });
        REQUIRE(csMod.has_value());
        const auto layout = bindings.createPipelineLayout({});
        REQUIRE(layout.has_value());

        mock.gatePipelineCreate = true;
        const auto comp = pipelines.beginComputePipelineCompilation({
            .layout = *layout,
            .computeShader = *csMod,
        });
        REQUIRE(comp.has_value());

        REQUIRE(mock.enteredCreate.try_acquire_for(std::chrono::seconds(5)));
        const auto pendingState = pipelines.pipelineCompilationState(*comp);
        REQUIRE(pendingState.has_value());
        CHECK(*pendingState == rachis::PipelineCompilationState::pending);

        REQUIRE(shaders.destroyShaderModule(*csMod).has_value());
        REQUIRE(bindings.destroyPipelineLayout(*layout).has_value());

        const auto staleBegin = pipelines.beginComputePipelineCompilation({
            .computeShader = *csMod,
        });
        REQUIRE_FALSE(staleBegin.has_value());
        CHECK(staleBegin.error().code == rachis::ErrorCode::invalidArgument);

        mock.releaseCreate.release();
        const auto built = pipelines.waitPipelineCompilation(*comp);
        REQUIRE(built.has_value());
        CHECK(built->isValid());

        const auto afterWait = pipelines.pipelineCompilationState(*comp);
        REQUIRE_FALSE(afterWait.has_value());
        CHECK(afterWait.error().code == rachis::ErrorCode::invalidArgument);
        REQUIRE(pipelines.destroyPipeline(*built).has_value());

        mock.gatePipelineCreate = false;
        const auto emptyBegin = pipelines.beginComputePipelineCompilation({});
        REQUIRE(emptyBegin.has_value());
        const auto emptyWait = pipelines.waitPipelineCompilation(*emptyBegin);
        REQUIRE_FALSE(emptyWait.has_value());
        CHECK(emptyWait.error().code == rachis::ErrorCode::invalidArgument);
    }

    SECTION("graphics compilation runs off-thread and cancellation destroys in-flight VkPipeline without leaking")
    {
        const auto vsMod = shaders.createShaderModule({
            .stage = rachis::ShaderStage::vertex,
            .bytecode = gfxSpv.bytes(),
            .entryPoint = "vsMain",
        });
        const auto psMod = shaders.createShaderModule({
            .stage = rachis::ShaderStage::pixel,
            .bytecode = gfxSpv.bytes(),
            .entryPoint = "psMain",
        });
        REQUIRE(vsMod.has_value());
        REQUIRE(psMod.has_value());
        const auto layout = bindings.createPipelineLayout({});
        REQUIRE(layout.has_value());
        const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};

        mock.gatePipelineCreate = true;
        const auto gfxComp = pipelines.beginGraphicsPipelineCompilation({
            .layout = *layout,
            .vertexShader = *vsMod,
            .pixelShader = *psMod,
            .colorTargets = colorTargets,
        });
        REQUIRE(gfxComp.has_value());
        REQUIRE(mock.enteredCreate.try_acquire_for(std::chrono::seconds(5)));

        const auto pendingState = pipelines.pipelineCompilationState(*gfxComp);
        REQUIRE(pendingState.has_value());
        CHECK(*pendingState == rachis::PipelineCompilationState::pending);

        REQUIRE(pipelines.destroyPipelineCompilation(*gfxComp).has_value());
        CHECK_FALSE(pipelines.pipelineCompilationState(*gfxComp).has_value());

        mock.releaseCreate.release();
        pipelines.shutdown();
        CHECK(mock.createdPipelines.load() == 1);
        CHECK(mock.destroyedPipelines.load() == 1);
    }

    pipelines.shutdown();
    bindings.shutdown();
    shaders.shutdown();
    gSharedPsoMock = nullptr;
    gPsoMock = nullptr;
}

TEST_CASE("VulkanPipelineManager rejects multiview geometry, tessellation, and mesh pipelines without their multiview "
          "features before vkCreateGraphicsPipelines",
          "[size:small][rhi][vulkan][multiview]")
{
    AsyncPsoMockState mock;
    gPsoMock = &mock;
    gSharedPsoMock = &mock;

    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetPhysicalDeviceFormatProperties = &psoMockGetFormatProps;
    instanceTable.vkGetPhysicalDeviceImageFormatProperties = &psoMockGetImageFormatProps;

    VolkDeviceTable table{};
    table.vkCreateShaderModule = &psoMockCreateShaderModule;
    table.vkDestroyShaderModule = &psoMockDestroyShaderModule;
    table.vkCreateDescriptorSetLayout = &psoMockCreateDescriptorSetLayout;
    table.vkDestroyDescriptorSetLayout = &psoMockDestroyDescriptorSetLayout;
    table.vkCreatePipelineLayout = &psoMockCreatePipelineLayout;
    table.vkDestroyPipelineLayout = &psoMockDestroyPipelineLayout;
    table.vkCreateGraphicsPipelines = &psoMockCreateGraphicsPipelines;
    table.vkDestroyPipeline = &psoMockDestroyPipeline;

    const auto device = fakeVk<VkDevice>(0x97);
    const auto physical = fakeVk<VkPhysicalDevice>(0x98);
    auto caps = baselineShaderTestCapabilities();
    caps.multiviewRendering = true;
    caps.maxMultiviewViewCount = 4;
    caps.geometryShaders = true;
    caps.tessellationShaders = true;
    caps.tessellation.maxPatchControlPoints = 32;
    caps.meshShaders = true;
    caps.meshShader.maxMeshWorkgroupInvocations = 128;

    rachis::vulkan::VulkanShaderManager shaders;
    shaders.initialize(device, &table, caps, &instanceTable);
    rachis::vulkan::VulkanBindingManager bindings;
    bindings.initialize(device, &table, caps, &instanceTable);

    SpirvBuilder spv;
    spv.op(17, {1});
    spv.op(14, {0, 1});
    spv.entryPoint(0, 10, "vsMain", {});
    spv.entryPoint(3, 11, "gsMain", {});
    spv.entryPoint(1, 12, "hsMain", {});
    spv.entryPoint(2, 13, "dsMain", {});
    spv.entryPoint(4, 14, "psMain", {});
    spv.entryPoint(5365, 15, "msMain", {});
    spv.op(16, {14, 7});
    spv.op(16, {15, 17, 1, 1, 1});
    spv.op(19, {1});
    spv.op(33, {2, 1});
    for (std::uint32_t function = 10; function <= 15; ++function)
    {
        spv.op(54, {1, function, 0, 2});
        spv.op(248, {function + 10});
        spv.op(253, {});
        spv.op(56, {});
    }

    const auto createModule = [&](rachis::ShaderStage stage, std::string_view entryPoint) {
        rachis::ShaderModuleDesc desc{
            .stage = stage,
            .bytecode = spv.bytes(),
            .entryPoint = entryPoint,
        };
        if (stage == rachis::ShaderStage::mesh)
        {
            desc.shaderInterface.workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1};
        }
        auto created = shaders.createShaderModule(desc);
        REQUIRE(created.has_value());
        return *created;
    };
    const auto vertex = createModule(rachis::ShaderStage::vertex, "vsMain");
    const auto geometry = createModule(rachis::ShaderStage::geometry, "gsMain");
    const auto hull = createModule(rachis::ShaderStage::hull, "hsMain");
    const auto domain = createModule(rachis::ShaderStage::domain, "dsMain");
    const auto pixel = createModule(rachis::ShaderStage::pixel, "psMain");
    const auto mesh = createModule(rachis::ShaderStage::mesh, "msMain");
    const auto layout = bindings.createPipelineLayout({});
    REQUIRE(layout.has_value());

    constexpr std::uint32_t twoViewMask = 0b11;
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    const rachis::GraphicsPipelineDesc geometryDesc{
        .layout = *layout,
        .vertexShader = vertex,
        .pixelShader = pixel,
        .geometryShader = geometry,
        .topology = rachis::PrimitiveTopology::pointList,
        .colorTargets = colorTargets,
        .viewMask = twoViewMask,
    };
    const rachis::GraphicsPipelineDesc tessellationDesc{
        .layout = *layout,
        .vertexShader = vertex,
        .pixelShader = pixel,
        .hullShader = hull,
        .domainShader = domain,
        .topology = rachis::PrimitiveTopology::patchList,
        .patchControlPoints = 3,
        .colorTargets = colorTargets,
        .viewMask = twoViewMask,
    };
    const rachis::GraphicsPipelineDesc meshDesc{
        .layout = *layout,
        .pixelShader = pixel,
        .meshShader = mesh,
        .colorTargets = colorTargets,
        .viewMask = twoViewMask,
    };
    const rachis::GraphicsPipelineDesc vertexDesc{
        .layout = *layout,
        .vertexShader = vertex,
        .pixelShader = pixel,
        .colorTargets = colorTargets,
        .viewMask = twoViewMask,
    };
    const std::array stageDescs{&geometryDesc, &tessellationDesc, &meshDesc};

    rachis::vulkan::VulkanPipelineManager pipelines;
    const auto createAndDestroy = [&](const rachis::GraphicsPipelineDesc& desc) {
        auto created = pipelines.createGraphicsPipeline(desc);
        REQUIRE(created.has_value());
        REQUIRE(pipelines.destroyPipeline(*created).has_value());
    };

    SECTION("without the stage features, viewMask pipelines fail with unsupported and never reach the driver")
    {
        pipelines.initialize(device, &table, physical, caps, &shaders, &bindings, &instanceTable, true);
        for (const auto* desc : stageDescs)
        {
            const auto rejected = pipelines.createGraphicsPipeline(*desc);
            REQUIRE_FALSE(rejected.has_value());
            CHECK(rejected.error().code == rachis::ErrorCode::unsupported);
        }
        CHECK(mock.createdPipelines.load() == 0);

        for (const auto* desc : stageDescs)
        {
            auto singleView = *desc;
            singleView.viewMask = 0;
            createAndDestroy(singleView);
        }
        createAndDestroy(vertexDesc);
        CHECK(mock.createdPipelines.load() == 4);
    }

    SECTION("with the stage features, viewMask pipelines reach the driver")
    {
        pipelines.initialize(device,
                             &table,
                             physical,
                             caps,
                             &shaders,
                             &bindings,
                             &instanceTable,
                             true,
                             nullptr,
                             false,
                             {.geometry = true, .tessellation = true, .mesh = true});
        for (const auto* desc : stageDescs)
        {
            createAndDestroy(*desc);
        }
        CHECK(mock.createdPipelines.load() == 3);
    }

    pipelines.shutdown();
    bindings.shutdown();
    shaders.shutdown();
    gSharedPsoMock = nullptr;
    gPsoMock = nullptr;
}

namespace
{

    struct RecordedDrawIndirectCall final
    {
        bool indexed = false;
        bool counted = false;
        bool mesh = false;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        VkBuffer countBuffer = VK_NULL_HANDLE;
        VkDeviceSize countOffset = 0;
        std::uint32_t maxDrawCount = 0;
        std::uint32_t stride = 0;
    };

    struct RecordedBindDescriptorsCall final
    {
        VkPipelineBindPoint bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        std::uint32_t firstSet = 0;
        std::uint32_t setCount = 0;
        std::vector<std::uint32_t> dynamicOffsets;
    };

    struct IndirectAndBindingMockState final
    {
        std::uintptr_t nextHandle = 0xC000;
        std::vector<std::unique_ptr<std::byte[]>> hostAllocations;
        std::vector<RecordedDrawIndirectCall> draws;
        std::vector<RecordedBindDescriptorsCall> binds;
        std::vector<VkDeviceSize> dispatchOffsets;
    };

    thread_local IndirectAndBindingMockState* gIndirectMock = nullptr;

    VKAPI_ATTR void VKAPI_CALL indMockGetPhysicalProps(VkPhysicalDevice, VkPhysicalDeviceProperties* props)
    {
        *props = VkPhysicalDeviceProperties{};
        props->apiVersion = VK_API_VERSION_1_3;
        props->limits.minUniformBufferOffsetAlignment = 64;
        props->limits.minStorageBufferOffsetAlignment = 32;
        props->limits.maxUniformBufferRange = 512;
        props->limits.maxStorageBufferRange = 2048;
        props->limits.maxDrawIndirectCount = 4;
        props->limits.nonCoherentAtomSize = 64;
    }

    VKAPI_ATTR void VKAPI_CALL indMockGetPhysicalProps2(VkPhysicalDevice, VkPhysicalDeviceProperties2*)
    {
    }

    VKAPI_ATTR void VKAPI_CALL indMockGetPhysicalMemProps(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* mem)
    {
        *mem = VkPhysicalDeviceMemoryProperties{};
        mem->memoryHeapCount = 1;
        mem->memoryHeaps[0] = {.size = 256ULL * 1024ULL * 1024ULL, .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
        mem->memoryTypeCount = 2;
        mem->memoryTypes[0] = {.propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .heapIndex = 0};
        mem->memoryTypes[1] = {
            .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            .heapIndex = 0,
        };
    }

    VKAPI_ATTR VkResult VKAPI_CALL indMockAllocateMemory(VkDevice,
                                                         const VkMemoryAllocateInfo* info,
                                                         const VkAllocationCallbacks*,
                                                         VkDeviceMemory* out)
    {
        auto backing =
            std::make_unique<std::byte[]>(static_cast<std::size_t>(std::max<VkDeviceSize>(info->allocationSize, 64)));
        gIndirectMock->hostAllocations.push_back(std::move(backing));
        *out = fakeVk<VkDeviceMemory>(gIndirectMock->hostAllocations.size());
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL indMockFreeMemory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL indMockMapMemory(
        VkDevice, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize, VkMemoryMapFlags, void** ppData)
    {
        const auto idx = reinterpret_cast<std::uintptr_t>(memory) - 1U;
        *ppData = gIndirectMock->hostAllocations[idx].get() + offset;
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL indMockUnmapMemory(VkDevice, VkDeviceMemory)
    {
    }
    VKAPI_ATTR VkResult VKAPI_CALL indMockFlushRanges(VkDevice, std::uint32_t, const VkMappedMemoryRange*)
    {
        return VK_SUCCESS;
    }
    VKAPI_ATTR VkResult VKAPI_CALL indMockInvalidateRanges(VkDevice, std::uint32_t, const VkMappedMemoryRange*)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL indMockCreateBuffer(VkDevice,
                                                       const VkBufferCreateInfo*,
                                                       const VkAllocationCallbacks*,
                                                       VkBuffer* out)
    {
        *out = fakeVk<VkBuffer>(gIndirectMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL indMockDestroyBuffer(VkDevice, VkBuffer, const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR void VKAPI_CALL indMockGetBufferMemReqs(VkDevice, VkBuffer, VkMemoryRequirements* reqs)
    {
        *reqs = VkMemoryRequirements{.size = 4096, .alignment = 64, .memoryTypeBits = 0x3U};
    }

    VKAPI_ATTR void VKAPI_CALL indMockGetBufferMemReqs2(VkDevice,
                                                        const VkBufferMemoryRequirementsInfo2*,
                                                        VkMemoryRequirements2* out)
    {
        out->memoryRequirements = VkMemoryRequirements{.size = 4096, .alignment = 64, .memoryTypeBits = 0x3U};
    }

    VKAPI_ATTR void VKAPI_CALL indMockGetDeviceBufferMemReqs(VkDevice,
                                                             const VkDeviceBufferMemoryRequirements*,
                                                             VkMemoryRequirements2* out)
    {
        out->memoryRequirements = VkMemoryRequirements{.size = 4096, .alignment = 64, .memoryTypeBits = 0x3U};
    }

    VKAPI_ATTR VkResult VKAPI_CALL indMockBindBufferMemory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL indMockBindBufferMemory2(VkDevice, std::uint32_t, const VkBindBufferMemoryInfo*)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL indMockCreateCommandPool(VkDevice,
                                                            const VkCommandPoolCreateInfo*,
                                                            const VkAllocationCallbacks*,
                                                            VkCommandPool* out)
    {
        *out = fakeVk<VkCommandPool>(gIndirectMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL indMockDestroyCommandPool(VkDevice, VkCommandPool, const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL indMockAllocCmdBuffers(VkDevice,
                                                          const VkCommandBufferAllocateInfo* info,
                                                          VkCommandBuffer* out)
    {
        for (std::uint32_t i = 0; i < info->commandBufferCount; ++i)
        {
            out[i] = fakeVk<VkCommandBuffer>(gIndirectMock->nextHandle++);
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL indMockFreeCmdBuffers(VkDevice, VkCommandPool, std::uint32_t, const VkCommandBuffer*)
    {
    }
    VKAPI_ATTR VkResult VKAPI_CALL indMockBeginCmdBuffer(VkCommandBuffer, const VkCommandBufferBeginInfo*)
    {
        return VK_SUCCESS;
    }
    VKAPI_ATTR VkResult VKAPI_CALL indMockEndCmdBuffer(VkCommandBuffer)
    {
        return VK_SUCCESS;
    }
    VKAPI_ATTR VkResult VKAPI_CALL indMockResetCmdBuffer(VkCommandBuffer, VkCommandBufferResetFlags)
    {
        return VK_SUCCESS;
    }
    VKAPI_ATTR VkResult VKAPI_CALL indMockResetCmdPool(VkDevice, VkCommandPool, VkCommandPoolResetFlags)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL indMockCreateDescSetLayout(VkDevice,
                                                              const VkDescriptorSetLayoutCreateInfo*,
                                                              const VkAllocationCallbacks*,
                                                              VkDescriptorSetLayout* out)
    {
        *out = fakeVk<VkDescriptorSetLayout>(gIndirectMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL indMockDestroyDescSetLayout(VkDevice,
                                                           VkDescriptorSetLayout,
                                                           const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL indMockCreateDescPool(VkDevice,
                                                         const VkDescriptorPoolCreateInfo*,
                                                         const VkAllocationCallbacks*,
                                                         VkDescriptorPool* out)
    {
        *out = fakeVk<VkDescriptorPool>(gIndirectMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL indMockDestroyDescPool(VkDevice, VkDescriptorPool, const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL indMockAllocDescSets(VkDevice,
                                                        const VkDescriptorSetAllocateInfo* info,
                                                        VkDescriptorSet* out)
    {
        for (std::uint32_t i = 0; i < info->descriptorSetCount; ++i)
        {
            out[i] = fakeVk<VkDescriptorSet>(gIndirectMock->nextHandle++);
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL indMockUpdateDescSets(
        VkDevice, std::uint32_t, const VkWriteDescriptorSet*, std::uint32_t, const VkCopyDescriptorSet*)
    {
    }

    VKAPI_ATTR void VKAPI_CALL indMockCmdBindDescSets(VkCommandBuffer,
                                                      VkPipelineBindPoint bindPoint,
                                                      VkPipelineLayout,
                                                      std::uint32_t firstSet,
                                                      std::uint32_t setCount,
                                                      const VkDescriptorSet*,
                                                      std::uint32_t dynamicOffsetCount,
                                                      const std::uint32_t* pDynamicOffsets)
    {
        gIndirectMock->binds.push_back({
            .bindPoint = bindPoint,
            .firstSet = firstSet,
            .setCount = setCount,
            .dynamicOffsets = std::vector<std::uint32_t>(pDynamicOffsets, pDynamicOffsets + dynamicOffsetCount),
        });
    }

    VKAPI_ATTR void VKAPI_CALL indMockCmdDrawIndirect(
        VkCommandBuffer, VkBuffer buffer, VkDeviceSize offset, std::uint32_t drawCount, std::uint32_t stride)
    {
        gIndirectMock->draws.push_back({
            .indexed = false,
            .counted = false,
            .buffer = buffer,
            .offset = offset,
            .maxDrawCount = drawCount,
            .stride = stride,
        });
    }

    VKAPI_ATTR void VKAPI_CALL indMockCmdDrawIndexedIndirect(
        VkCommandBuffer, VkBuffer buffer, VkDeviceSize offset, std::uint32_t drawCount, std::uint32_t stride)
    {
        gIndirectMock->draws.push_back({
            .indexed = true,
            .counted = false,
            .buffer = buffer,
            .offset = offset,
            .maxDrawCount = drawCount,
            .stride = stride,
        });
    }

    VKAPI_ATTR void VKAPI_CALL indMockCmdDrawIndirectCount(VkCommandBuffer,
                                                           VkBuffer buffer,
                                                           VkDeviceSize offset,
                                                           VkBuffer countBuffer,
                                                           VkDeviceSize countOffset,
                                                           std::uint32_t maxDrawCount,
                                                           std::uint32_t stride)
    {
        gIndirectMock->draws.push_back({
            .indexed = false,
            .counted = true,
            .buffer = buffer,
            .offset = offset,
            .countBuffer = countBuffer,
            .countOffset = countOffset,
            .maxDrawCount = maxDrawCount,
            .stride = stride,
        });
    }

    VKAPI_ATTR void VKAPI_CALL indMockCmdDrawIndexedIndirectCount(VkCommandBuffer,
                                                                  VkBuffer buffer,
                                                                  VkDeviceSize offset,
                                                                  VkBuffer countBuffer,
                                                                  VkDeviceSize countOffset,
                                                                  std::uint32_t maxDrawCount,
                                                                  std::uint32_t stride)
    {
        gIndirectMock->draws.push_back({
            .indexed = true,
            .counted = true,
            .buffer = buffer,
            .offset = offset,
            .countBuffer = countBuffer,
            .countOffset = countOffset,
            .maxDrawCount = maxDrawCount,
            .stride = stride,
        });
    }

    VKAPI_ATTR void VKAPI_CALL indMockCmdDrawMeshTasksIndirect(
        VkCommandBuffer, VkBuffer buffer, VkDeviceSize offset, std::uint32_t drawCount, std::uint32_t stride)
    {
        gIndirectMock->draws.push_back({
            .counted = false,
            .mesh = true,
            .buffer = buffer,
            .offset = offset,
            .maxDrawCount = drawCount,
            .stride = stride,
        });
    }

    VKAPI_ATTR void VKAPI_CALL indMockCmdDrawMeshTasksIndirectCount(VkCommandBuffer,
                                                                    VkBuffer buffer,
                                                                    VkDeviceSize offset,
                                                                    VkBuffer countBuffer,
                                                                    VkDeviceSize countOffset,
                                                                    std::uint32_t maxDrawCount,
                                                                    std::uint32_t stride)
    {
        gIndirectMock->draws.push_back({
            .counted = true,
            .mesh = true,
            .buffer = buffer,
            .offset = offset,
            .countBuffer = countBuffer,
            .countOffset = countOffset,
            .maxDrawCount = maxDrawCount,
            .stride = stride,
        });
    }

    VKAPI_ATTR void VKAPI_CALL indMockCmdDispatchIndirect(VkCommandBuffer, VkBuffer, VkDeviceSize offset)
    {
        gIndirectMock->dispatchOffsets.push_back(offset);
    }
    VKAPI_ATTR void VKAPI_CALL
    indMockCmdCopyBuffer(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*)
    {
    }

    VKAPI_ATTR void VKAPI_CALL indMockGetPhysicalMemProps2(VkPhysicalDevice dev, VkPhysicalDeviceMemoryProperties2* out)
    {
        indMockGetPhysicalMemProps(dev, &out->memoryProperties);
    }

    VKAPI_ATTR VkResult VKAPI_CALL indMockCreateImage(VkDevice,
                                                      const VkImageCreateInfo*,
                                                      const VkAllocationCallbacks*,
                                                      VkImage* out)
    {
        *out = fakeVk<VkImage>(gIndirectMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL indMockDestroyImage(VkDevice, VkImage, const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR void VKAPI_CALL indMockGetImageMemReqs(VkDevice, VkImage, VkMemoryRequirements* out)
    {
        *out = VkMemoryRequirements{.size = 4096, .alignment = 4096, .memoryTypeBits = 0x3};
    }

    VKAPI_ATTR void VKAPI_CALL indMockGetImageMemReqs2(VkDevice,
                                                       const VkImageMemoryRequirementsInfo2*,
                                                       VkMemoryRequirements2* out)
    {
        out->memoryRequirements = VkMemoryRequirements{.size = 4096, .alignment = 4096, .memoryTypeBits = 0x3};
    }

    VKAPI_ATTR void VKAPI_CALL indMockGetDeviceImageMemReqs(VkDevice,
                                                            const VkDeviceImageMemoryRequirements*,
                                                            VkMemoryRequirements2* out)
    {
        out->memoryRequirements = VkMemoryRequirements{.size = 4096, .alignment = 4096, .memoryTypeBits = 0x3};
    }

    VKAPI_ATTR VkResult VKAPI_CALL indMockBindImageMemory(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL indMockBindImageMemory2(VkDevice, std::uint32_t, const VkBindImageMemoryInfo*)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL indMockGetInstanceProcAddr(VkInstance, const char*)
    {
        return nullptr;
    }
    VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL indMockGetDeviceProcAddr(VkDevice, const char*)
    {
        return nullptr;
    }

} // namespace

TEST_CASE("Vulkan multi-draw indirect, counted indirect, and native buffer alignment/dynamic-offset validation",
          "[size:small][rhi][vulkan][indirect][bindings]")
{
    IndirectAndBindingMockState mock;
    gIndirectMock = &mock;

    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetInstanceProcAddr = &indMockGetInstanceProcAddr;
    instanceTable.vkGetDeviceProcAddr = &indMockGetDeviceProcAddr;
    instanceTable.vkGetPhysicalDeviceProperties = &indMockGetPhysicalProps;
    instanceTable.vkGetPhysicalDeviceProperties2 = &indMockGetPhysicalProps2;
    instanceTable.vkGetPhysicalDeviceMemoryProperties = &indMockGetPhysicalMemProps;
    instanceTable.vkGetPhysicalDeviceMemoryProperties2 = &indMockGetPhysicalMemProps2;

    VolkDeviceTable table{};
    table.vkAllocateMemory = &indMockAllocateMemory;
    table.vkFreeMemory = &indMockFreeMemory;
    table.vkMapMemory = &indMockMapMemory;
    table.vkUnmapMemory = &indMockUnmapMemory;
    table.vkFlushMappedMemoryRanges = &indMockFlushRanges;
    table.vkInvalidateMappedMemoryRanges = &indMockInvalidateRanges;
    table.vkCreateBuffer = &indMockCreateBuffer;
    table.vkDestroyBuffer = &indMockDestroyBuffer;
    table.vkGetBufferMemoryRequirements = &indMockGetBufferMemReqs;
    table.vkGetBufferMemoryRequirements2 = &indMockGetBufferMemReqs2;
    table.vkGetDeviceBufferMemoryRequirements = &indMockGetDeviceBufferMemReqs;
    table.vkBindBufferMemory = &indMockBindBufferMemory;
    table.vkBindBufferMemory2 = &indMockBindBufferMemory2;
    table.vkCreateImage = &indMockCreateImage;
    table.vkDestroyImage = &indMockDestroyImage;
    table.vkGetImageMemoryRequirements = &indMockGetImageMemReqs;
    table.vkGetImageMemoryRequirements2 = &indMockGetImageMemReqs2;
    table.vkGetDeviceImageMemoryRequirements = &indMockGetDeviceImageMemReqs;
    table.vkBindImageMemory = &indMockBindImageMemory;
    table.vkBindImageMemory2 = &indMockBindImageMemory2;
    table.vkCmdCopyBuffer = &indMockCmdCopyBuffer;
    table.vkCreateCommandPool = &indMockCreateCommandPool;
    table.vkDestroyCommandPool = &indMockDestroyCommandPool;
    table.vkAllocateCommandBuffers = &indMockAllocCmdBuffers;
    table.vkFreeCommandBuffers = &indMockFreeCmdBuffers;
    table.vkBeginCommandBuffer = &indMockBeginCmdBuffer;
    table.vkEndCommandBuffer = &indMockEndCmdBuffer;
    table.vkResetCommandBuffer = &indMockResetCmdBuffer;
    table.vkResetCommandPool = &indMockResetCmdPool;
    table.vkCreateDescriptorSetLayout = &indMockCreateDescSetLayout;
    table.vkDestroyDescriptorSetLayout = &indMockDestroyDescSetLayout;
    table.vkCreateDescriptorPool = &indMockCreateDescPool;
    table.vkDestroyDescriptorPool = &indMockDestroyDescPool;
    table.vkAllocateDescriptorSets = &indMockAllocDescSets;
    table.vkUpdateDescriptorSets = &indMockUpdateDescSets;
    table.vkCmdBindDescriptorSets = &indMockCmdBindDescSets;
    table.vkCmdDrawIndirect = &indMockCmdDrawIndirect;
    table.vkCmdDrawIndexedIndirect = &indMockCmdDrawIndexedIndirect;
    table.vkCmdDrawIndirectCount = &indMockCmdDrawIndirectCount;
    table.vkCmdDrawIndexedIndirectCount = &indMockCmdDrawIndexedIndirectCount;
    table.vkCmdDispatchIndirect = &indMockCmdDispatchIndirect;

    const auto instance = fakeVk<VkInstance>(0xD0);
    const auto physical = fakeVk<VkPhysicalDevice>(0xD1);
    const auto device = fakeVk<VkDevice>(0xD2);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    const rachis::vulkan::VulkanQueueHandles queueHandles{
        .graphics = fakeVk<VkQueue>(0xD3),
        .compute = fakeVk<VkQueue>(0xD3),
        .copy = fakeVk<VkQueue>(0xD3),
        .families = families,
    };

    VkPhysicalDeviceMemoryProperties memProps{};
    indMockGetPhysicalMemProps(physical, &memProps);

    rachis::vulkan::VulkanResourceAllocator resources;
    REQUIRE(
        resources
            .initialize(
                instance, &instanceTable, physical, device, &table, memProps, families, queueHandles.copy, 64, false)
            .has_value());

    const rachis::vulkan::VulkanBufferBindingLimits bufferLimits{
        .minUniformBufferOffsetAlignment = 64,
        .minStorageBufferOffsetAlignment = 32,
        .maxUniformBufferRange = 512,
        .maxStorageBufferRange = 2048,
    };

    rachis::DeviceCapabilities caps{};
    caps.multiDrawIndirect = true;
    caps.drawIndirectCount = true;

    rachis::vulkan::VulkanBindingManager bindings;
    bindings.initialize(device, &table, caps, &instanceTable, bufferLimits);

    rachis::vulkan::VulkanGraphicsEncoder graphics;
    graphics.initialize(device, &table, &caps, &resources, nullptr, &bindings, true, bufferLimits, 4);

    constexpr std::uint64_t kToken = 0xDDD0;
    rachis::vulkan::VulkanCommandAllocator commands;
    commands.initialize(device, &table, queueHandles, kToken, &resources, &instanceTable);
    const auto cmdPool = commands.createCommandPool({.queueType = rachis::QueueType::graphics, .maxCommandBuffers = 4});
    REQUIRE(cmdPool.has_value());
    auto cb = commands.acquireCommandBuffer(*cmdPool);
    REQUIRE(cb.has_value());
    auto& vkCb = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb);

    SECTION("drawIndirect and drawIndexedIndirect honor multiDrawIndirect, drawIndirectCount, and maxDrawIndirectCount")
    {
        const auto argBuf = resources.createBuffer({.size = 256, .usage = rachis::BufferUsage::indirect});
        const auto cntBuf = resources.createBuffer({.size = 64, .usage = rachis::BufferUsage::indirect});
        const auto idxBuf = resources.createBuffer({.size = 256, .usage = rachis::BufferUsage::index});
        REQUIRE(argBuf.has_value());
        REQUIRE(cntBuf.has_value());
        REQUIRE(idxBuf.has_value());

        auto gfxPipe = std::make_shared<rachis::vulkan::VulkanNativePipeline>();
        gfxPipe->pipeline = fakeVk<VkPipeline>(0xE001);
        gfxPipe->layout = fakeVk<VkPipelineLayout>(0xE002);
        vkCb.graphicsState().renderingOwner = std::make_shared<int>(1);
        vkCb.graphicsState().pipelineOwner = gfxPipe;
        vkCb.graphicsState().viewportCount = 1;
        vkCb.graphicsState().scissorCount = 1;
        vkCb.graphicsState().indexBufferBound = true;

        REQUIRE(graphics
                    .drawIndirect(vkCb,
                                  {
                                      .argumentBuffer = *argBuf,
                                      .argumentOffset = 16,
                                      .maxDrawCount = 3,
                                      .stride = 20,
                                  })
                    .has_value());
        REQUIRE(mock.draws.size() == 1);
        CHECK_FALSE(mock.draws[0].indexed);
        CHECK_FALSE(mock.draws[0].counted);
        CHECK(mock.draws[0].maxDrawCount == 3);
        CHECK(mock.draws[0].stride == 20);

        REQUIRE(graphics
                    .drawIndexedIndirect(vkCb,
                                         {
                                             .argumentBuffer = *argBuf,
                                             .argumentOffset = 0,
                                             .maxDrawCount = 3,
                                             .stride = 24,
                                             .countBuffer = *cntBuf,
                                             .countOffset = 4,
                                         })
                    .has_value());
        REQUIRE(mock.draws.size() == 2);
        CHECK(mock.draws[1].indexed);
        CHECK(mock.draws[1].counted);
        CHECK(mock.draws[1].countOffset == 4);
        CHECK(mock.draws[1].maxDrawCount == 3);
        CHECK(mock.draws[1].stride == 24);

        const auto overMax = graphics.drawIndirect(vkCb,
                                                   {
                                                       .argumentBuffer = *argBuf,
                                                       .maxDrawCount = 5,
                                                       .stride = 16,
                                                   });
        REQUIRE_FALSE(overMax.has_value());
        CHECK(overMax.error().code == rachis::ErrorCode::unsupported);

        caps.multiDrawIndirect = false;
        caps.drawIndirectCount = false;
        const auto multiDisabled = graphics.drawIndirect(vkCb, {.argumentBuffer = *argBuf, .maxDrawCount = 2});
        REQUIRE_FALSE(multiDisabled.has_value());
        CHECK(multiDisabled.error().code == rachis::ErrorCode::unsupported);

        const auto countDisabled =
            graphics.drawIndexedIndirect(vkCb, {.argumentBuffer = *argBuf, .maxDrawCount = 1, .countBuffer = *cntBuf});
        REQUIRE_FALSE(countDisabled.has_value());
        CHECK(countDisabled.error().code == rachis::ErrorCode::unsupported);

        caps.multiDrawIndirect = true;
        caps.drawIndirectCount = true;
        auto csPipe = std::make_shared<rachis::vulkan::VulkanNativePipeline>();
        csPipe->pipeline = fakeVk<VkPipeline>(0xE003);
        csPipe->layout = fakeVk<VkPipelineLayout>(0xE004);
        csPipe->compute = true;
        vkCb.graphicsState().renderingOwner.reset();
        vkCb.computeState().pipelineOwner = csPipe;
        CHECK(graphics.dispatchIndirect(vkCb, {.argumentBuffer = *argBuf, .maxDispatchCount = 2}).error().code ==
              rachis::ErrorCode::unsupported);
        CHECK(mock.dispatchOffsets.empty());

        caps.multiDispatchIndirect = true;
        REQUIRE(graphics
                    .dispatchIndirect(vkCb,
                                      {
                                          .argumentBuffer = *argBuf,
                                          .argumentOffset = 16,
                                          .maxDispatchCount = 3,
                                          .stride = 20,
                                      })
                    .has_value());
        CHECK(mock.dispatchOffsets == std::vector<VkDeviceSize>{16, 36, 56});
        REQUIRE(graphics.dispatchIndirect(vkCb, {.argumentBuffer = *argBuf, .maxDispatchCount = 0}).has_value());
        CHECK(mock.dispatchOffsets.size() == 3);
        CHECK(graphics.dispatchIndirect(vkCb, {.argumentBuffer = *argBuf, .countBuffer = *cntBuf}).error().code ==
              rachis::ErrorCode::unsupported);
        CHECK(mock.dispatchOffsets.size() == 3);
    }

    SECTION("dispatchMeshIndirect honors multiDrawIndirect, drawIndirectCount, maxDrawIndirectCount, and its "
            "native entry points")
    {
        const auto argBuf = resources.createBuffer({.size = 256, .usage = rachis::BufferUsage::indirect});
        const auto cntBuf = resources.createBuffer({.size = 64, .usage = rachis::BufferUsage::indirect});
        REQUIRE(argBuf.has_value());
        REQUIRE(cntBuf.has_value());

        caps.meshShaders = true;
        table.vkCmdDrawMeshTasksIndirectEXT = &indMockCmdDrawMeshTasksIndirect;
        table.vkCmdDrawMeshTasksIndirectCountEXT = &indMockCmdDrawMeshTasksIndirectCount;
        auto meshPipe = std::make_shared<rachis::vulkan::VulkanNativePipeline>();
        meshPipe->pipeline = fakeVk<VkPipeline>(0xE021);
        meshPipe->layout = fakeVk<VkPipelineLayout>(0xE022);
        meshPipe->mesh = true;
        vkCb.graphicsState().renderingOwner = std::make_shared<int>(1);
        vkCb.graphicsState().pipelineOwner = meshPipe;
        vkCb.graphicsState().viewportCount = 1;
        vkCb.graphicsState().scissorCount = 1;

        REQUIRE(graphics.dispatchMeshIndirect(vkCb, {.argumentBuffer = *argBuf, .maxDispatchCount = 4, .stride = 16})
                    .has_value());
        REQUIRE(mock.draws.size() == 1);
        CHECK(mock.draws[0].mesh);
        CHECK_FALSE(mock.draws[0].counted);
        CHECK(mock.draws[0].maxDrawCount == 4);
        CHECK(mock.draws[0].stride == 16);

        REQUIRE(graphics
                    .dispatchMeshIndirect(vkCb,
                                          {
                                              .argumentBuffer = *argBuf,
                                              .maxDispatchCount = 4,
                                              .countBuffer = *cntBuf,
                                              .countOffset = 4,
                                          })
                    .has_value());
        REQUIRE(mock.draws.size() == 2);
        CHECK(mock.draws[1].mesh);
        CHECK(mock.draws[1].counted);
        CHECK(mock.draws[1].countOffset == 4);
        CHECK(mock.draws[1].maxDrawCount == 4);
        CHECK(mock.draws[1].stride == sizeof(rachis::DispatchIndirectArguments));

        const auto overMax = graphics.dispatchMeshIndirect(vkCb, {.argumentBuffer = *argBuf, .maxDispatchCount = 5});
        REQUIRE_FALSE(overMax.has_value());
        CHECK(overMax.error().code == rachis::ErrorCode::unsupported);
        const auto countedOverMax = graphics.dispatchMeshIndirect(
            vkCb, {.argumentBuffer = *argBuf, .maxDispatchCount = 5, .countBuffer = *cntBuf});
        REQUIRE_FALSE(countedOverMax.has_value());
        CHECK(countedOverMax.error().code == rachis::ErrorCode::unsupported);
        CHECK(mock.draws.size() == 2);

        caps.multiDrawIndirect = false;
        caps.drawIndirectCount = false;
        const auto multiDisabled =
            graphics.dispatchMeshIndirect(vkCb, {.argumentBuffer = *argBuf, .maxDispatchCount = 2});
        REQUIRE_FALSE(multiDisabled.has_value());
        CHECK(multiDisabled.error().code == rachis::ErrorCode::unsupported);
        const auto countDisabled = graphics.dispatchMeshIndirect(
            vkCb, {.argumentBuffer = *argBuf, .maxDispatchCount = 1, .countBuffer = *cntBuf});
        REQUIRE_FALSE(countDisabled.has_value());
        CHECK(countDisabled.error().code == rachis::ErrorCode::unsupported);
        CHECK(mock.draws.size() == 2);
        REQUIRE(graphics.dispatchMeshIndirect(vkCb, {.argumentBuffer = *argBuf, .maxDispatchCount = 1}).has_value());
        REQUIRE(mock.draws.size() == 3);
        CHECK(mock.draws[2].maxDrawCount == 1);

        caps.multiDrawIndirect = true;
        caps.drawIndirectCount = true;
        table.vkCmdDrawMeshTasksIndirectEXT = nullptr;
        table.vkCmdDrawMeshTasksIndirectCountEXT = nullptr;
        const auto missingDirect = graphics.dispatchMeshIndirect(vkCb, {.argumentBuffer = *argBuf});
        REQUIRE_FALSE(missingDirect.has_value());
        CHECK(missingDirect.error().code == rachis::ErrorCode::unsupported);
        const auto missingCounted =
            graphics.dispatchMeshIndirect(vkCb, {.argumentBuffer = *argBuf, .countBuffer = *cntBuf});
        REQUIRE_FALSE(missingCounted.has_value());
        CHECK(missingCounted.error().code == rachis::ErrorCode::unsupported);
        CHECK(mock.draws.size() == 3);
    }

    SECTION("native UBO/SSBO alignments and atomic dynamic-offset validation in bindGraphicsBindingSets and "
            "bindComputeBindingSets")
    {
        const auto uboBuf = resources.createBuffer({.size = 512, .usage = rachis::BufferUsage::constant});
        const auto bigUboBuf = resources.createBuffer({.size = 2048, .usage = rachis::BufferUsage::constant});
        const auto ssboBuf = resources.createBuffer({.size = 512, .usage = rachis::BufferUsage::storage});
        REQUIRE(uboBuf.has_value());
        REQUIRE(bigUboBuf.has_value());
        REQUIRE(ssboBuf.has_value());

        const std::array entries{
            rachis::BindingLayoutEntry{
                .binding = 0,
                .type = rachis::BindingType::dynamicUniformBuffer,
                .visibility = rachis::ShaderStage::vertex | rachis::ShaderStage::compute,
            },
            rachis::BindingLayoutEntry{
                .binding = 1,
                .type = rachis::BindingType::dynamicStorageBuffer,
                .visibility = rachis::ShaderStage::vertex | rachis::ShaderStage::compute,
            },
        };
        const auto setLayout = bindings.createBindingLayout({.entries = entries});
        REQUIRE(setLayout.has_value());

        const std::array poolSizes{
            rachis::BindingPoolSize{.type = rachis::BindingType::dynamicUniformBuffer, .descriptorCount = 8},
            rachis::BindingPoolSize{.type = rachis::BindingType::dynamicStorageBuffer, .descriptorCount = 8},
        };
        const auto pool = bindings.createBindingPool({
            .maxBindingSets = 4,
            .sizes = poolSizes,
        });
        REQUIRE(pool.has_value());

        {
            const std::array badUboWrites{
                rachis::BindingWrite{
                    .binding = 0,
                    .resource = rachis::BufferBinding{.buffer = *uboBuf, .offset = 32, .size = 16},
                },
                rachis::BindingWrite{
                    .binding = 1,
                    .resource = rachis::BufferBinding{.buffer = *ssboBuf, .offset = 32, .size = 48},
                },
            };
            const auto badSet =
                bindings.createBindingSet(resources, {.layout = *setLayout, .writes = badUboWrites, .pool = *pool});
            REQUIRE_FALSE(badSet.has_value());
            CHECK(badSet.error().code == rachis::ErrorCode::invalidArgument);
        }

        {
            const std::array badSsboWrites{
                rachis::BindingWrite{
                    .binding = 0,
                    .resource = rachis::BufferBinding{.buffer = *uboBuf, .offset = 64, .size = 16},
                },
                rachis::BindingWrite{
                    .binding = 1,
                    .resource = rachis::BufferBinding{.buffer = *ssboBuf, .offset = 16, .size = 48},
                },
            };
            CHECK_FALSE(
                bindings.createBindingSet(resources, {.layout = *setLayout, .writes = badSsboWrites, .pool = *pool})
                    .has_value());

            const std::array overMaxUboWrites{
                rachis::BindingWrite{
                    .binding = 0,
                    .resource = rachis::BufferBinding{.buffer = *bigUboBuf, .offset = 0, .size = 1024},
                },
                rachis::BindingWrite{
                    .binding = 1,
                    .resource = rachis::BufferBinding{.buffer = *ssboBuf, .offset = 32, .size = 48},
                },
            };
            CHECK_FALSE(
                bindings.createBindingSet(resources, {.layout = *setLayout, .writes = overMaxUboWrites, .pool = *pool})
                    .has_value());
        }

        const std::array validWrites{
            rachis::BindingWrite{
                .binding = 0,
                .resource = rachis::BufferBinding{.buffer = *uboBuf, .offset = 64, .size = 16},
            },
            rachis::BindingWrite{
                .binding = 1,
                .resource = rachis::BufferBinding{.buffer = *ssboBuf, .offset = 32, .size = 64},
            },
        };
        const auto set = bindings.createBindingSet(resources,
                                                   {
                                                       .layout = *setLayout,
                                                       .writes = validWrites,
                                                       .pool = *pool,
                                                   });
        REQUIRE(set.has_value());

        const auto nativeLayout = bindings.nativeBindingLayout(*setLayout);
        REQUIRE(nativeLayout.has_value());
        auto pipe = std::make_shared<rachis::vulkan::VulkanNativePipeline>();
        pipe->pipeline = fakeVk<VkPipeline>(0xE010);
        pipe->layout = fakeVk<VkPipelineLayout>(0xE011);
        pipe->compute = true;
        pipe->setLayouts = {nativeLayout->setLayout};
        pipe->setDynamicCounts = {nativeLayout->dynamicBufferCount};
        pipe->setRequired = {1};
        vkCb.graphicsState().renderingOwner = std::make_shared<int>(1);
        vkCb.graphicsState().pipelineOwner = pipe;
        vkCb.graphicsState().boundBindingSets = {false};
        vkCb.computeState().pipelineOwner = pipe;
        vkCb.computeState().boundBindingSets = {false};

        const std::array sets{*set};

        const auto retainedBefore = vkCb.retained().size();
        const std::array misalignedDyn{32U, 0U};
        const auto badAlign = graphics.bindGraphicsBindingSets(vkCb, 0, sets, misalignedDyn);
        REQUIRE_FALSE(badAlign.has_value());
        CHECK(badAlign.error().code == rachis::ErrorCode::invalidArgument);
        CHECK(mock.binds.empty());
        CHECK(vkCb.graphicsState().boundBindingSets == rachis::detail::BindingSetMask{false});
        CHECK(vkCb.retained().size() == retainedBefore);

        const std::array oobDyn{448U, 0U};
        const auto badOob = graphics.bindComputeBindingSets(vkCb, 0, sets, oobDyn);
        REQUIRE_FALSE(badOob.has_value());
        CHECK(badOob.error().code == rachis::ErrorCode::invalidArgument);
        CHECK(mock.binds.empty());
        CHECK(vkCb.computeState().boundBindingSets == rachis::detail::BindingSetMask{false});
        CHECK(vkCb.retained().size() == retainedBefore);

        const std::array goodDyn{128U, 64U};
        REQUIRE(graphics.bindGraphicsBindingSets(vkCb, 0, sets, goodDyn).has_value());
        REQUIRE(graphics.bindComputeBindingSets(vkCb, 0, sets, goodDyn).has_value());
        REQUIRE(mock.binds.size() == 2);
        CHECK(mock.binds[0].bindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS);
        CHECK(mock.binds[0].dynamicOffsets == std::vector<std::uint32_t>{128U, 64U});
        CHECK(mock.binds[1].bindPoint == VK_PIPELINE_BIND_POINT_COMPUTE);
        CHECK(mock.binds[1].dynamicOffsets == std::vector<std::uint32_t>{128U, 64U});

        const std::array updatedWrites{
            rachis::BindingWrite{
                .binding = 0,
                .resource = rachis::BufferBinding{.buffer = *uboBuf, .offset = 0, .size = 256},
            },
        };
        REQUIRE(bindings.updateBindingSet(resources, *set, updatedWrites).has_value());
        const std::array updatedOobDyn{320U, 0U};
        CHECK_FALSE(graphics.bindComputeBindingSets(vkCb, 0, sets, updatedOobDyn).has_value());
        const std::array updatedGoodDyn{256U, 0U};
        REQUIRE(graphics.bindComputeBindingSets(vkCb, 0, sets, updatedGoodDyn).has_value());
    }

    cb->reset();
    commands.shutdown();
    bindings.shutdown();
    resources.shutdown();
    gIndirectMock = nullptr;
}

namespace
{

    constexpr VkDeviceSize shaderTableMockBufferAlignment = 16;
    constexpr std::uint32_t shaderTableMockBaseAlignment = 64;

    struct ShaderTableMockBuffer final
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
    };

    struct ShaderTableMockState final
    {
        std::vector<ShaderTableMockBuffer> buffers;
        std::vector<VkDeviceAddress> rayGenerationAddresses;
        VkDeviceAddress addressBias = 0;
    };

    thread_local ShaderTableMockState* gShaderTableMock = nullptr;

    [[nodiscard]] ShaderTableMockBuffer* findShaderTableMockBuffer(VkBuffer buffer) noexcept
    {
        const auto found = std::ranges::find(gShaderTableMock->buffers, buffer, &ShaderTableMockBuffer::buffer);
        return found == gShaderTableMock->buffers.end() ? nullptr : &*found;
    }

    [[nodiscard]] VkMemoryRequirements shaderTableMockRequirements(VkDeviceSize size) noexcept
    {
        const auto alignedSize = (size + shaderTableMockBufferAlignment - 1) / shaderTableMockBufferAlignment *
                                 shaderTableMockBufferAlignment;
        return VkMemoryRequirements{
            .size = alignedSize,
            .alignment = shaderTableMockBufferAlignment,
            .memoryTypeBits = 0x3U,
        };
    }

    VKAPI_ATTR VkResult VKAPI_CALL sbtMockCreateBuffer(VkDevice,
                                                       const VkBufferCreateInfo* info,
                                                       const VkAllocationCallbacks*,
                                                       VkBuffer* out)
    {
        *out = fakeVk<VkBuffer>(gIndirectMock->nextHandle++);
        gShaderTableMock->buffers.push_back({.buffer = *out, .size = info->size});
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL sbtMockGetBufferMemReqs(VkDevice, VkBuffer buffer, VkMemoryRequirements* reqs)
    {
        const auto* record = findShaderTableMockBuffer(buffer);
        *reqs = shaderTableMockRequirements(record == nullptr ? 0 : record->size);
    }

    VKAPI_ATTR void VKAPI_CALL sbtMockGetBufferMemReqs2(VkDevice,
                                                        const VkBufferMemoryRequirementsInfo2* info,
                                                        VkMemoryRequirements2* out)
    {
        sbtMockGetBufferMemReqs(VK_NULL_HANDLE, info->buffer, &out->memoryRequirements);
    }

    VKAPI_ATTR void VKAPI_CALL sbtMockGetDeviceBufferMemReqs(VkDevice,
                                                             const VkDeviceBufferMemoryRequirements* info,
                                                             VkMemoryRequirements2* out)
    {
        out->memoryRequirements = shaderTableMockRequirements(info->pCreateInfo->size);
    }

    VKAPI_ATTR VkResult VKAPI_CALL sbtMockBindBufferMemory(VkDevice,
                                                           VkBuffer buffer,
                                                           VkDeviceMemory memory,
                                                           VkDeviceSize offset)
    {
        auto* record = findShaderTableMockBuffer(buffer);
        if (record == nullptr)
        {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        record->memory = memory;
        record->offset = offset;
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL sbtMockBindBufferMemory2(VkDevice,
                                                            std::uint32_t count,
                                                            const VkBindBufferMemoryInfo* infos)
    {
        for (std::uint32_t index = 0; index < count; ++index)
        {
            if (const auto bound = sbtMockBindBufferMemory(
                    VK_NULL_HANDLE, infos[index].buffer, infos[index].memory, infos[index].memoryOffset);
                bound != VK_SUCCESS)
            {
                return bound;
            }
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkDeviceAddress VKAPI_CALL sbtMockGetBufferDeviceAddress(VkDevice, const VkBufferDeviceAddressInfo* info)
    {
        const auto* record = findShaderTableMockBuffer(info->buffer);
        if (record == nullptr || record->memory == VK_NULL_HANDLE)
        {
            return 0;
        }
        const auto memoryBase = static_cast<VkDeviceAddress>(reinterpret_cast<std::uintptr_t>(record->memory)) << 32U;
        return memoryBase + record->offset + gShaderTableMock->addressBias;
    }

    VKAPI_ATTR void VKAPI_CALL sbtMockCmdTraceRays(VkCommandBuffer,
                                                   const VkStridedDeviceAddressRegionKHR* rayGeneration,
                                                   const VkStridedDeviceAddressRegionKHR*,
                                                   const VkStridedDeviceAddressRegionKHR*,
                                                   const VkStridedDeviceAddressRegionKHR*,
                                                   std::uint32_t,
                                                   std::uint32_t,
                                                   std::uint32_t)
    {
        gShaderTableMock->rayGenerationAddresses.push_back(rayGeneration->deviceAddress);
    }

} // namespace

TEST_CASE("Vulkan shader-binding tables start at shaderGroupBaseAlignment device addresses",
          "[size:small][rhi][vulkan][ray-tracing-pipeline][bindings]")
{
    IndirectAndBindingMockState indirectMock;
    gIndirectMock = &indirectMock;
    ShaderTableMockState mock;
    gShaderTableMock = &mock;

    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetInstanceProcAddr = &indMockGetInstanceProcAddr;
    instanceTable.vkGetDeviceProcAddr = &indMockGetDeviceProcAddr;
    instanceTable.vkGetPhysicalDeviceProperties = &indMockGetPhysicalProps;
    instanceTable.vkGetPhysicalDeviceProperties2 = &indMockGetPhysicalProps2;
    instanceTable.vkGetPhysicalDeviceMemoryProperties = &indMockGetPhysicalMemProps;
    instanceTable.vkGetPhysicalDeviceMemoryProperties2 = &indMockGetPhysicalMemProps2;

    VolkDeviceTable table{};
    table.vkAllocateMemory = &indMockAllocateMemory;
    table.vkFreeMemory = &indMockFreeMemory;
    table.vkMapMemory = &indMockMapMemory;
    table.vkUnmapMemory = &indMockUnmapMemory;
    table.vkFlushMappedMemoryRanges = &indMockFlushRanges;
    table.vkInvalidateMappedMemoryRanges = &indMockInvalidateRanges;
    table.vkCreateBuffer = &sbtMockCreateBuffer;
    table.vkDestroyBuffer = &indMockDestroyBuffer;
    table.vkGetBufferMemoryRequirements = &sbtMockGetBufferMemReqs;
    table.vkGetBufferMemoryRequirements2 = &sbtMockGetBufferMemReqs2;
    table.vkGetDeviceBufferMemoryRequirements = &sbtMockGetDeviceBufferMemReqs;
    table.vkBindBufferMemory = &sbtMockBindBufferMemory;
    table.vkBindBufferMemory2 = &sbtMockBindBufferMemory2;
    table.vkGetBufferDeviceAddress = &sbtMockGetBufferDeviceAddress;
    table.vkCreateImage = &indMockCreateImage;
    table.vkDestroyImage = &indMockDestroyImage;
    table.vkGetImageMemoryRequirements = &indMockGetImageMemReqs;
    table.vkGetImageMemoryRequirements2 = &indMockGetImageMemReqs2;
    table.vkGetDeviceImageMemoryRequirements = &indMockGetDeviceImageMemReqs;
    table.vkBindImageMemory = &indMockBindImageMemory;
    table.vkBindImageMemory2 = &indMockBindImageMemory2;
    table.vkCmdCopyBuffer = &indMockCmdCopyBuffer;
    table.vkCreateCommandPool = &indMockCreateCommandPool;
    table.vkDestroyCommandPool = &indMockDestroyCommandPool;
    table.vkAllocateCommandBuffers = &indMockAllocCmdBuffers;
    table.vkFreeCommandBuffers = &indMockFreeCmdBuffers;
    table.vkBeginCommandBuffer = &indMockBeginCmdBuffer;
    table.vkEndCommandBuffer = &indMockEndCmdBuffer;
    table.vkResetCommandBuffer = &indMockResetCmdBuffer;
    table.vkResetCommandPool = &indMockResetCmdPool;
    table.vkCmdTraceRaysKHR = &sbtMockCmdTraceRays;

    const auto instance = fakeVk<VkInstance>(0xD8);
    const auto physical = fakeVk<VkPhysicalDevice>(0xD9);
    const auto device = fakeVk<VkDevice>(0xDA);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    const rachis::vulkan::VulkanQueueHandles queueHandles{
        .graphics = fakeVk<VkQueue>(0xDB),
        .compute = fakeVk<VkQueue>(0xDB),
        .copy = fakeVk<VkQueue>(0xDB),
        .families = families,
    };

    VkPhysicalDeviceMemoryProperties memProps{};
    indMockGetPhysicalMemProps(physical, &memProps);

    rachis::DeviceCapabilities caps{};
    caps.rayTracing = true;
    caps.rayTracingLimits = {
        .shaderGroupHandleSize = 32,
        .shaderGroupHandleAlignment = 32,
        .shaderGroupBaseAlignment = shaderTableMockBaseAlignment,
        .maxShaderGroupStride = 4096,
        .maxRecursionDepth = 1,
        .maxAttributeSize = 32,
        .maxDispatchInvocations = 1U << 30U,
    };

    rachis::vulkan::VulkanResourceAllocator resources;
    REQUIRE(resources
                .initialize(instance,
                            &instanceTable,
                            physical,
                            device,
                            &table,
                            memProps,
                            families,
                            queueHandles.copy,
                            64,
                            false,
                            false,
                            false,
                            false,
                            false,
                            false,
                            true,
                            true,
                            true,
                            shaderTableMockBaseAlignment)
                .has_value());

    rachis::vulkan::VulkanGraphicsEncoder graphics;
    graphics.initialize(device, &table, &caps, &resources, nullptr, nullptr);

    rachis::vulkan::VulkanCommandAllocator commands;
    commands.initialize(device, &table, queueHandles, 0xDDE0, &resources, &instanceTable);
    const auto cmdPool = commands.createCommandPool({.queueType = rachis::QueueType::compute, .maxCommandBuffers = 2});
    REQUIRE(cmdPool.has_value());
    auto cb = commands.acquireCommandBuffer(*cmdPool);
    REQUIRE(cb.has_value());
    auto& vkCb = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb);
    vkCb.rayTracingState().pipelineOwner = std::make_shared<int>(1);

    const auto padding = resources.createBuffer({.size = 16, .usage = rachis::BufferUsage::indirect});
    const auto shaderTable = resources.createBuffer({.size = 64, .usage = rachis::BufferUsage::shaderBindingTable});
    REQUIRE(padding.has_value());
    REQUIRE(shaderTable.has_value());
    const auto paddingAddress = resources.resolveBindingBuffer(*padding);
    const auto tableAddress = resources.resolveBindingBuffer(*shaderTable);
    REQUIRE(paddingAddress.has_value());
    REQUIRE(tableAddress.has_value());
    CAPTURE(paddingAddress->deviceAddress, tableAddress->deviceAddress);
    REQUIRE(tableAddress->deviceAddress != 0);
    CHECK(tableAddress->deviceAddress % shaderTableMockBaseAlignment == 0);

    const rachis::RayTracingDispatchDesc aligned{
        .rayGeneration = {.buffer = *shaderTable, .size = 32, .stride = 32},
    };
    REQUIRE(graphics.dispatchRays(vkCb, aligned).has_value());
    REQUIRE(mock.rayGenerationAddresses.size() == 1);
    CHECK(mock.rayGenerationAddresses[0] == tableAddress->deviceAddress);

    mock.addressBias = shaderTableMockBufferAlignment;
    const auto shiftedTable = resources.createBuffer({.size = 64, .usage = rachis::BufferUsage::shaderBindingTable});
    REQUIRE(shiftedTable.has_value());
    const auto shiftedAddress = resources.resolveBindingBuffer(*shiftedTable);
    REQUIRE(shiftedAddress.has_value());
    REQUIRE(shiftedAddress->deviceAddress % shaderTableMockBaseAlignment != 0);
    const auto shifted = graphics.dispatchRays(vkCb,
                                               {
                                                   .rayGeneration = {.buffer = *shiftedTable, .size = 32, .stride = 32},
                                               });
    REQUIRE_FALSE(shifted.has_value());
    CHECK(shifted.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(mock.rayGenerationAddresses.size() == 1);

    cb->reset();
    commands.shutdown();
    resources.shutdown();
    gShaderTableMock = nullptr;
    gIndirectMock = nullptr;
}

namespace
{

    constexpr std::uint32_t scratchMockAlignment = 128;
    constexpr VkDeviceSize scratchMockBuildScratchSize = 384;

    struct ScratchMockState final
    {
        std::uintptr_t nextStructure = 0xA500;
        std::vector<VkDeviceAddress> scratchAddresses;
    };

    thread_local ScratchMockState* gScratchMock = nullptr;

    VKAPI_ATTR VkResult VKAPI_CALL scratchMockCreateAccelerationStructure(VkDevice,
                                                                          const VkAccelerationStructureCreateInfoKHR*,
                                                                          const VkAllocationCallbacks*,
                                                                          VkAccelerationStructureKHR* out)
    {
        *out = fakeVk<VkAccelerationStructureKHR>(gScratchMock->nextStructure++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL scratchMockDestroyAccelerationStructure(VkDevice,
                                                                       VkAccelerationStructureKHR,
                                                                       const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR void VKAPI_CALL scratchMockGetBuildSizes(VkDevice,
                                                        VkAccelerationStructureBuildTypeKHR,
                                                        const VkAccelerationStructureBuildGeometryInfoKHR*,
                                                        const std::uint32_t*,
                                                        VkAccelerationStructureBuildSizesInfoKHR* sizes)
    {
        sizes->accelerationStructureSize = 256;
        sizes->buildScratchSize = scratchMockBuildScratchSize;
        sizes->updateScratchSize = 0;
    }

    VKAPI_ATTR VkDeviceAddress VKAPI_CALL
    scratchMockGetAccelerationStructureAddress(VkDevice, const VkAccelerationStructureDeviceAddressInfoKHR* info)
    {
        return static_cast<VkDeviceAddress>(reinterpret_cast<std::uintptr_t>(info->accelerationStructure)) << 8U;
    }

    VKAPI_ATTR void VKAPI_CALL scratchMockCmdBuild(VkCommandBuffer,
                                                   std::uint32_t count,
                                                   const VkAccelerationStructureBuildGeometryInfoKHR* infos,
                                                   const VkAccelerationStructureBuildRangeInfoKHR* const*)
    {
        for (std::uint32_t index = 0; index < count; ++index)
        {
            gScratchMock->scratchAddresses.push_back(infos[index].scratchData.deviceAddress);
        }
    }

    VKAPI_ATTR void VKAPI_CALL scratchMockCmdPipelineBarrier2(VkCommandBuffer, const VkDependencyInfo*)
    {
    }

} // namespace

TEST_CASE("Vulkan acceleration-structure scratch buffers start at the scratch-offset alignment",
          "[size:small][rhi][vulkan][ray-query]")
{
    IndirectAndBindingMockState indirectMock;
    gIndirectMock = &indirectMock;
    ShaderTableMockState bufferMock;
    gShaderTableMock = &bufferMock;
    ScratchMockState mock;
    gScratchMock = &mock;

    rachis::vulkan::VulkanInstanceTable instanceTable{};
    instanceTable.vkGetInstanceProcAddr = &indMockGetInstanceProcAddr;
    instanceTable.vkGetDeviceProcAddr = &indMockGetDeviceProcAddr;
    instanceTable.vkGetPhysicalDeviceProperties = &indMockGetPhysicalProps;
    instanceTable.vkGetPhysicalDeviceProperties2 = &indMockGetPhysicalProps2;
    instanceTable.vkGetPhysicalDeviceMemoryProperties = &indMockGetPhysicalMemProps;
    instanceTable.vkGetPhysicalDeviceMemoryProperties2 = &indMockGetPhysicalMemProps2;

    VolkDeviceTable table{};
    table.vkAllocateMemory = &indMockAllocateMemory;
    table.vkFreeMemory = &indMockFreeMemory;
    table.vkMapMemory = &indMockMapMemory;
    table.vkUnmapMemory = &indMockUnmapMemory;
    table.vkFlushMappedMemoryRanges = &indMockFlushRanges;
    table.vkInvalidateMappedMemoryRanges = &indMockInvalidateRanges;
    table.vkCreateBuffer = &sbtMockCreateBuffer;
    table.vkDestroyBuffer = &indMockDestroyBuffer;
    table.vkGetBufferMemoryRequirements = &sbtMockGetBufferMemReqs;
    table.vkGetBufferMemoryRequirements2 = &sbtMockGetBufferMemReqs2;
    table.vkGetDeviceBufferMemoryRequirements = &sbtMockGetDeviceBufferMemReqs;
    table.vkBindBufferMemory = &sbtMockBindBufferMemory;
    table.vkBindBufferMemory2 = &sbtMockBindBufferMemory2;
    table.vkGetBufferDeviceAddress = &sbtMockGetBufferDeviceAddress;
    table.vkCreateImage = &indMockCreateImage;
    table.vkDestroyImage = &indMockDestroyImage;
    table.vkGetImageMemoryRequirements = &indMockGetImageMemReqs;
    table.vkGetImageMemoryRequirements2 = &indMockGetImageMemReqs2;
    table.vkGetDeviceImageMemoryRequirements = &indMockGetDeviceImageMemReqs;
    table.vkBindImageMemory = &indMockBindImageMemory;
    table.vkBindImageMemory2 = &indMockBindImageMemory2;
    table.vkCmdCopyBuffer = &indMockCmdCopyBuffer;
    table.vkCreateCommandPool = &indMockCreateCommandPool;
    table.vkDestroyCommandPool = &indMockDestroyCommandPool;
    table.vkAllocateCommandBuffers = &indMockAllocCmdBuffers;
    table.vkFreeCommandBuffers = &indMockFreeCmdBuffers;
    table.vkBeginCommandBuffer = &indMockBeginCmdBuffer;
    table.vkEndCommandBuffer = &indMockEndCmdBuffer;
    table.vkResetCommandBuffer = &indMockResetCmdBuffer;
    table.vkResetCommandPool = &indMockResetCmdPool;
    table.vkCreateAccelerationStructureKHR = &scratchMockCreateAccelerationStructure;
    table.vkDestroyAccelerationStructureKHR = &scratchMockDestroyAccelerationStructure;
    table.vkGetAccelerationStructureBuildSizesKHR = &scratchMockGetBuildSizes;
    table.vkGetAccelerationStructureDeviceAddressKHR = &scratchMockGetAccelerationStructureAddress;
    table.vkCmdBuildAccelerationStructuresKHR = &scratchMockCmdBuild;
    table.vkCmdPipelineBarrier2 = &scratchMockCmdPipelineBarrier2;

    const auto instance = fakeVk<VkInstance>(0xE8);
    const auto physical = fakeVk<VkPhysicalDevice>(0xE9);
    const auto device = fakeVk<VkDevice>(0xEA);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    const rachis::vulkan::VulkanQueueHandles queueHandles{
        .graphics = fakeVk<VkQueue>(0xEB),
        .compute = fakeVk<VkQueue>(0xEB),
        .copy = fakeVk<VkQueue>(0xEB),
        .families = families,
    };

    VkPhysicalDeviceMemoryProperties memProps{};
    indMockGetPhysicalMemProps(physical, &memProps);

    rachis::DeviceCapabilities caps{};
    caps.rayQuery = true;

    rachis::vulkan::VulkanResourceAllocator resources;
    REQUIRE(resources
                .initialize(instance,
                            &instanceTable,
                            physical,
                            device,
                            &table,
                            memProps,
                            families,
                            queueHandles.copy,
                            64,
                            false,
                            false,
                            false,
                            false,
                            true,
                            false,
                            true,
                            true,
                            false,
                            0,
                            scratchMockAlignment)
                .has_value());

    rachis::vulkan::VulkanAccelerationManager acceleration;
    acceleration.initialize(device, &table, &resources, &caps, scratchMockAlignment, &instanceTable);

    rachis::vulkan::VulkanCommandAllocator commands;
    commands.initialize(device, &table, queueHandles, 0xEDE0, &resources, &instanceTable);
    const auto cmdPool = commands.createCommandPool({.queueType = rachis::QueueType::compute, .maxCommandBuffers = 2});
    REQUIRE(cmdPool.has_value());
    auto cb = commands.acquireCommandBuffer(*cmdPool);
    REQUIRE(cb.has_value());
    auto& vkCb = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb);

    const auto bounds = resources.createBuffer({
        .size = 6U * sizeof(float),
        .usage = rachis::BufferUsage::accelerationStructureBuildInput,
    });
    REQUIRE(bounds.has_value());
    const std::array geometry{
        rachis::AccelerationStructureGeometry{
            .data = rachis::AccelerationStructureAabbGeometry{.buffer = *bounds, .count = 1},
        },
    };
    const rachis::BottomLevelAccelerationStructureBuildInput input{.geometries = geometry};
    const auto sizes = acceleration.queryBottomLevelBuildSizes(input);
    REQUIRE(sizes.has_value());
    REQUIRE(sizes->buildScratchSize == scratchMockBuildScratchSize);

    std::vector<rachis::AccelerationStructureHandle> structures;
    const auto createDestination = [&] {
        const auto storage = resources.createBuffer({
            .size = sizes->accelerationStructureSize,
            .usage = rachis::BufferUsage::accelerationStructureStorage,
        });
        REQUIRE(storage.has_value());
        const auto destination = acceleration.createAccelerationStructure({
            .type = rachis::AccelerationStructureType::bottomLevel,
            .storage = *storage,
        });
        REQUIRE(destination.has_value());
        structures.push_back(*destination);
        return *destination;
    };
    const auto createScratch = [&](rachis::ResidencyPriority priority) {
        const auto padding = resources.createBuffer({
            .size = 16,
            .usage = rachis::BufferUsage::storage,
            .residencyPriority = priority,
        });
        REQUIRE(padding.has_value());
        const auto scratch = resources.createBuffer({
            .size = sizes->buildScratchSize,
            .usage = rachis::BufferUsage::accelerationStructureScratch,
            .residencyPriority = priority,
        });
        REQUIRE(scratch.has_value());
        return *scratch;
    };

    for (const auto priority : {rachis::ResidencyPriority::automatic, rachis::ResidencyPriority::high})
    {
        CAPTURE(static_cast<int>(priority));
        const auto destination = createDestination();
        const auto scratch = createScratch(priority);
        const auto scratchAddress = resources.resolveBindingBuffer(scratch);
        REQUIRE(scratchAddress.has_value());
        CAPTURE(scratchAddress->deviceAddress);
        REQUIRE(scratchAddress->deviceAddress != 0);
        CHECK(scratchAddress->deviceAddress % scratchMockAlignment == 0);
        const auto recorded = mock.scratchAddresses.size();
        REQUIRE(acceleration
                    .buildBottomLevel(vkCb,
                                      {
                                          .destination = destination,
                                          .input = input,
                                          .scratch = scratch,
                                      })
                    .has_value());
        REQUIRE(mock.scratchAddresses.size() == recorded + 1);
        CHECK(mock.scratchAddresses.back() == scratchAddress->deviceAddress);
        CHECK(mock.scratchAddresses.back() % scratchMockAlignment == 0);
    }

    bufferMock.addressBias = shaderTableMockBufferAlignment;
    const auto shiftedScratch = createScratch(rachis::ResidencyPriority::automatic);
    const auto shiftedAddress = resources.resolveBindingBuffer(shiftedScratch);
    REQUIRE(shiftedAddress.has_value());
    REQUIRE(shiftedAddress->deviceAddress % scratchMockAlignment != 0);
    const auto recorded = mock.scratchAddresses.size();
    const auto shifted = acceleration.buildBottomLevel(vkCb,
                                                       {
                                                           .destination = createDestination(),
                                                           .input = input,
                                                           .scratch = shiftedScratch,
                                                       });
    REQUIRE_FALSE(shifted.has_value());
    CHECK(shifted.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(mock.scratchAddresses.size() == recorded);

    cb->reset();
    commands.shutdown();
    for (const auto structure : structures)
    {
        REQUIRE(acceleration.destroyAccelerationStructure(structure).has_value());
    }
    acceleration.shutdown();
    resources.shutdown();
    gScratchMock = nullptr;
    gShaderTableMock = nullptr;
    gIndirectMock = nullptr;
}
#endif
