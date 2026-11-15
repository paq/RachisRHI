#if RACHIS_ENABLE_VULKAN
#include "Internal/VulkanSwapchain.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstdint>
#include <memory>
#include <utility>

namespace
{

    template<typename Handle>
    Handle fakeHandle(std::uintptr_t value)
    {
        return reinterpret_cast<Handle>(value);
    }

    struct DispatchState final
    {
        VkSemaphore pending = VK_NULL_HANDLE;
        VkResult submitFailure = VK_ERROR_OUT_OF_HOST_MEMORY;
        VkResult idleFailure = VK_ERROR_OUT_OF_DEVICE_MEMORY;
        VkResult fenceFailure = VK_ERROR_OUT_OF_HOST_MEMORY;
        int submitFailures = 1;
        int idleFailures = 0;
        int submitAttempts = 0;
        int successfulWaits = 0;
        int idleAttempts = 0;
        int nativeAcquires = 0;
        int pendingDestroyed = 0;
        int acquireDestroyed = 0;
        int fenceFailures = 1;
        int fenceWaits = 0;
        int pendingFencesDestroyed = 0;
        int imageViewsDestroyed = 0;
        int imageViewsDestroyedBeforeSwapchain = 0;
        bool fencePending = false;
        bool waitQueued = false;
        bool deviceLost = false;
        std::uintptr_t nextSemaphore = 100;
        std::uintptr_t nextImageView = 200;
    };

    DispatchState dispatch;

    VkResult VKAPI_PTR createSemaphore(VkDevice,
                                       const VkSemaphoreCreateInfo*,
                                       const VkAllocationCallbacks*,
                                       VkSemaphore* out)
    {
        *out = fakeHandle<VkSemaphore>(++dispatch.nextSemaphore);
        return VK_SUCCESS;
    }

    void VKAPI_PTR destroySemaphore(VkDevice, VkSemaphore semaphore, const VkAllocationCallbacks*)
    {
        if (semaphore == fakeHandle<VkSemaphore>(11))
        {
            ++dispatch.acquireDestroyed;
        }
        if (semaphore == dispatch.pending && !dispatch.deviceLost)
        {
            ++dispatch.pendingDestroyed;
        }
    }

    void VKAPI_PTR destroySwapchain(VkDevice, VkSwapchainKHR, const VkAllocationCallbacks*)
    {
        dispatch.imageViewsDestroyedBeforeSwapchain = dispatch.imageViewsDestroyed;
    }

    VkResult VKAPI_PTR swapchainImages(VkDevice, VkSwapchainKHR, std::uint32_t* count, VkImage* images)
    {
        if (images != nullptr)
        {
            images[0] = fakeHandle<VkImage>(30);
        }
        *count = 1;
        return VK_SUCCESS;
    }

    VkResult VKAPI_PTR createImageView(VkDevice,
                                       const VkImageViewCreateInfo*,
                                       const VkAllocationCallbacks*,
                                       VkImageView* out)
    {
        *out = fakeHandle<VkImageView>(++dispatch.nextImageView);
        return VK_SUCCESS;
    }

    void VKAPI_PTR destroyImageView(VkDevice, VkImageView, const VkAllocationCallbacks*)
    {
        ++dispatch.imageViewsDestroyed;
    }

    void VKAPI_PTR formatProperties(VkPhysicalDevice, VkFormat, VkFormatProperties* properties)
    {
        *properties = VkFormatProperties{
            .linearTilingFeatures = 0,
            .optimalTilingFeatures = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                                     VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT,
            .bufferFeatures = 0,
        };
    }

    VkResult VKAPI_PTR imageFormatProperties(VkPhysicalDevice,
                                             VkFormat,
                                             VkImageType,
                                             VkImageTiling,
                                             VkImageUsageFlags,
                                             VkImageCreateFlags,
                                             VkImageFormatProperties* properties)
    {
        *properties = VkImageFormatProperties{
            .maxExtent = {4096, 4096, 1},
            .maxMipLevels = 1,
            .maxArrayLayers = 1,
            .sampleCounts = VK_SAMPLE_COUNT_1_BIT,
            .maxResourceSize = 1ULL << 30U,
        };
        return VK_SUCCESS;
    }

    void VKAPI_PTR destroyFence(VkDevice, VkFence, const VkAllocationCallbacks*)
    {
        if (dispatch.fencePending && !dispatch.deviceLost)
        {
            ++dispatch.pendingFencesDestroyed;
        }
    }

    VkResult VKAPI_PTR waitFence(VkDevice, std::uint32_t, const VkFence*, VkBool32, std::uint64_t)
    {
        ++dispatch.fenceWaits;
        if (dispatch.fenceFailures-- > 0)
        {
            return dispatch.fenceFailure;
        }
        dispatch.fencePending = false;
        return VK_SUCCESS;
    }

    VkResult VKAPI_PTR resetFence(VkDevice, std::uint32_t, const VkFence*)
    {
        return VK_SUCCESS;
    }

    PFN_vkVoidFunction VKAPI_PTR getDeviceProcAddr(VkDevice, const char*)
    {
        return nullptr;
    }

    VkResult VKAPI_PTR
    acquire(VkDevice, VkSwapchainKHR, std::uint64_t, VkSemaphore semaphore, VkFence, std::uint32_t* index)
    {
        dispatch.pending = semaphore;
        ++dispatch.nativeAcquires;
        *index = 0;
        return VK_SUCCESS;
    }

    VkResult VKAPI_PTR submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo2* submits, VkFence)
    {
        ++dispatch.submitAttempts;
        if (dispatch.submitFailures-- > 0)
        {
            dispatch.deviceLost = dispatch.submitFailure == VK_ERROR_DEVICE_LOST;
            return dispatch.submitFailure;
        }
        if (queue != fakeHandle<VkQueue>(2) || count != 1 || submits->waitSemaphoreInfoCount != 1 ||
            submits->pWaitSemaphoreInfos->semaphore != dispatch.pending || dispatch.waitQueued)
        {
            return VK_ERROR_UNKNOWN;
        }
        dispatch.waitQueued = true;
        ++dispatch.successfulWaits;
        return VK_SUCCESS;
    }

    VkResult VKAPI_PTR queueIdle(VkQueue)
    {
        ++dispatch.idleAttempts;
        if (dispatch.idleFailures-- > 0)
        {
            dispatch.deviceLost = dispatch.idleFailure == VK_ERROR_DEVICE_LOST;
            return dispatch.idleFailure;
        }
        if (dispatch.waitQueued)
        {
            dispatch.pending = VK_NULL_HANDLE;
        }
        return VK_SUCCESS;
    }

    VkResult VKAPI_PTR deviceIdle(VkDevice)
    {
        return VK_SUCCESS;
    }

    VkResult VKAPI_PTR counterValue(VkDevice, VkSemaphore, std::uint64_t* value)
    {
        *value = 0;
        return VK_SUCCESS;
    }

} // namespace

namespace rachis::vulkan
{

    struct VulkanSwapchainTestAccess final
    {
        VulkanInstanceTable instanceTable{};
        VolkDeviceTable table{};
        VulkanQueueManager queues;
        VulkanResourceAllocator resources;
        VulkanSwapchainManager manager;
        SwapchainHandle handle{};

        VulkanSwapchainTestAccess()
        {
            dispatch = {};
            instanceTable.vkGetDeviceProcAddr = &getDeviceProcAddr;
            instanceTable.vkGetPhysicalDeviceFormatProperties = &formatProperties;
            instanceTable.vkGetPhysicalDeviceImageFormatProperties = &imageFormatProperties;
            table.vkCreateImageView = &createImageView;
            table.vkDestroyImageView = &destroyImageView;
            table.vkCreateSemaphore = &createSemaphore;
            table.vkDestroySemaphore = &destroySemaphore;
            table.vkQueueSubmit2 = &submit;
            table.vkQueueWaitIdle = &queueIdle;
            table.vkDeviceWaitIdle = &deviceIdle;
            table.vkGetSemaphoreCounterValue = &counterValue;
            table.vkDestroyFence = &destroyFence;
            table.vkWaitForFences = &waitFence;
            table.vkResetFences = &resetFence;
            auto* const graphics = fakeHandle<VkQueue>(2);
            const VulkanQueueHandles nativeQueues{
                .graphics = graphics,
                .compute = graphics,
                .copy = graphics,
                .present = fakeHandle<VkQueue>(3),
            };
            REQUIRE(queues.initialize(fakeHandle<VkDevice>(1), &table, nativeQueues, 1, false, &instanceTable));
            manager.device_ = fakeHandle<VkDevice>(1);
            manager.table_ = &table;
            manager.queues_ = &queues;
            manager.vkAcquireNextImageKHR_ = &acquire;
            manager.vkDestroySwapchainKHR_ = &destroySwapchain;
            auto record = std::make_shared<VulkanSwapchainManager::SwapchainRecord>();
            record->swapchain = fakeHandle<VkSwapchainKHR>(4);
            record->images.resize(1);
            record->unusedAcquire = {fakeHandle<VkSemaphore>(10), fakeHandle<VkSemaphore>(11)};
            auto published = manager.swapchains_.emplace(std::move(record));
            REQUIRE(published);
            handle = *published;
        }

        ~VulkanSwapchainTestAccess()
        {
            manager.flushDeferredAcquireWaits();
            manager.shutdown();
            queues.shutdown();
        }

        VulkanSwapchainTestAccess(const VulkanSwapchainTestAccess&) = delete;
        VulkanSwapchainTestAccess& operator=(const VulkanSwapchainTestAccess&) = delete;
        VulkanSwapchainTestAccess(VulkanSwapchainTestAccess&&) = delete;
        VulkanSwapchainTestAccess& operator=(VulkanSwapchainTestAccess&&) = delete;

        void failAcquire()
        {
            auto result = manager.acquireNextImage(handle);
            REQUIRE_FALSE(result);
            CHECK(result.error().code == ErrorCode::outOfMemory);
            const auto snapshot = manager.presentationSyncSnapshot(handle);
            REQUIRE(snapshot);
            CHECK_FALSE(snapshot->acquired);
            CHECK(dispatch.nativeAcquires == 1);
            CHECK(dispatch.successfulWaits == 0);
        }

        void parkAtPresentFence()
        {
            auto& record = **manager.swapchains_.find(handle);
            record.images[0].presentFence = fakeHandle<VkFence>(12);
            record.images[0].presentFencePending = true;
            dispatch.fencePending = true;
            failAcquire();
        }

        void tearDownDevice()
        {
            manager.flushDeferredAcquireWaits();
            static_cast<void>(queues.waitIdle());
            queues.shutdown();
            manager.shutdown();
        }

        [[nodiscard]] TextureHandle adoptSwapchainImages()
        {
            resources.initializeDispatchForTesting(
                fakeHandle<VkDevice>(1), &table, &instanceTable, fakeHandle<VkPhysicalDevice>(5));
            manager.resources_ = &resources;
            manager.vkGetSwapchainImagesKHR_ = &swapchainImages;
            auto& record = **manager.swapchains_.find(handle);
            record.desc.width = 64;
            record.desc.height = 64;
            REQUIRE(manager.createImages(record));
            REQUIRE(record.images.size() == 1);
            return record.images[0].texture;
        }
    };

} // namespace rachis::vulkan

TEST_CASE("Vulkan teardown drains an acquire through repeated submit OOM",
          "[size:small][rhi][vulkan][acquire-teardown]")
{
    rachis::vulkan::VulkanSwapchainTestAccess fixture;
    dispatch.submitFailure = GENERATE(VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY);
    fixture.failAcquire();
    dispatch.submitFailures = 3;
    fixture.tearDownDevice();
    CHECK(dispatch.submitAttempts == 5);
    CHECK(dispatch.successfulWaits == 1);
    CHECK(dispatch.pending == VK_NULL_HANDLE);
    CHECK(dispatch.pendingDestroyed == 0);
    CHECK(dispatch.acquireDestroyed == 1);
}

TEST_CASE("Vulkan teardown retries failed idle waits without resubmitting an acquire",
          "[size:small][rhi][vulkan][acquire-teardown]")
{
    rachis::vulkan::VulkanSwapchainTestAccess fixture;
    fixture.failAcquire();
    dispatch.idleFailure = GENERATE(VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY);
    dispatch.idleFailures = 3;
    fixture.tearDownDevice();
    CHECK(dispatch.idleAttempts == 4);
    CHECK(dispatch.successfulWaits == 1);
    CHECK(dispatch.pendingDestroyed == 0);
    CHECK(dispatch.acquireDestroyed == 1);
}

TEST_CASE("Vulkan teardown does not drain an acquire twice", "[size:small][rhi][vulkan][acquire-teardown]")
{
    rachis::vulkan::VulkanSwapchainTestAccess fixture;
    fixture.failAcquire();
    fixture.manager.flushDeferredAcquireWaits();
    fixture.manager.flushDeferredAcquireWaits();
    fixture.tearDownDevice();
    CHECK(dispatch.submitAttempts == 2);
    CHECK(dispatch.successfulWaits == 1);
    CHECK(dispatch.pendingDestroyed == 0);
}

TEST_CASE("Vulkan teardown skips the acquire drain once the queue manager has shut down",
          "[size:small][rhi][vulkan][acquire-teardown]")
{
    rachis::vulkan::VulkanSwapchainTestAccess fixture;
    fixture.failAcquire();
    fixture.queues.shutdown();
    fixture.manager.flushDeferredAcquireWaits();
    CHECK(dispatch.submitAttempts == 1);
    CHECK(dispatch.successfulWaits == 0);
    CHECK(dispatch.idleAttempts == 0);

    rachis::vulkan::VulkanQueueManager uninitialized;
    uninitialized.drainGraphicsAcquire(fakeHandle<VkSemaphore>(10));
    CHECK(dispatch.submitAttempts == 1);
    CHECK(dispatch.idleAttempts == 0);
}

TEST_CASE("Vulkan teardown stops draining when the driver reports device loss",
          "[size:small][rhi][vulkan][acquire-teardown]")
{
    rachis::vulkan::VulkanSwapchainTestAccess fixture;
    fixture.failAcquire();
    const bool loseOnSubmit = GENERATE(true, false);
    if (loseOnSubmit)
    {
        dispatch.submitFailure = VK_ERROR_DEVICE_LOST;
        dispatch.submitFailures = 1;
    }
    else
    {
        dispatch.idleFailure = VK_ERROR_DEVICE_LOST;
        dispatch.idleFailures = 1;
    }
    fixture.tearDownDevice();
    CHECK(dispatch.deviceLost);
    CHECK(dispatch.submitAttempts == 2);
    CHECK(dispatch.successfulWaits == (loseOnSubmit ? 0 : 1));
    CHECK(dispatch.pendingDestroyed == 0);
    CHECK(dispatch.acquireDestroyed == 1);
}

TEST_CASE("Vulkan teardown completes a parked acquire and its repeatedly failing present fence",
          "[size:small][rhi][vulkan][acquire-teardown]")
{
    rachis::vulkan::VulkanSwapchainTestAccess fixture;
    fixture.parkAtPresentFence();
    dispatch.fenceFailures = 3;
    fixture.tearDownDevice();
    CHECK(dispatch.successfulWaits == 1);
    CHECK(dispatch.fenceWaits == 5);
    CHECK(dispatch.pendingDestroyed == 0);
    CHECK(dispatch.pendingFencesDestroyed == 0);
    CHECK(dispatch.acquireDestroyed == 1);
}

TEST_CASE("Vulkan teardown gives up an acquire drain whose submit or idle wait keeps failing",
          "[size:small][rhi][vulkan][acquire-teardown]")
{
    constexpr int persistentFailures = 100'000;
    rachis::vulkan::VulkanSwapchainTestAccess fixture;
    fixture.failAcquire();
    const auto failure = GENERATE(VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_ERROR_UNKNOWN);
    const bool failSubmit = GENERATE(true, false);
    if (failSubmit)
    {
        dispatch.submitFailure = failure;
        dispatch.submitFailures = persistentFailures;
    }
    else
    {
        dispatch.idleFailure = failure;
        dispatch.idleFailures = persistentFailures;
    }
    fixture.tearDownDevice();
    const int attempts = failSubmit ? dispatch.submitAttempts - 1 : dispatch.idleAttempts;
    CHECK(attempts < persistentFailures);
    if (failure == VK_ERROR_UNKNOWN)
    {
        CHECK(attempts == 1);
    }
    else
    {
        CHECK(attempts > 4);
    }
    CHECK(dispatch.successfulWaits == (failSubmit ? 0 : 1));
    CHECK(dispatch.acquireDestroyed == 1);
}

TEST_CASE("Vulkan teardown gives up a parked present fence that keeps failing",
          "[size:small][rhi][vulkan][acquire-teardown]")
{
    constexpr int persistentFailures = 100'000;
    rachis::vulkan::VulkanSwapchainTestAccess fixture;
    fixture.parkAtPresentFence();
    const auto failure = GENERATE(VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_ERROR_UNKNOWN);
    dispatch.fenceFailure = failure;
    dispatch.fenceFailures = persistentFailures;
    fixture.tearDownDevice();
    const int attempts = dispatch.fenceWaits - 1;
    CHECK(attempts < persistentFailures);
    if (failure == VK_ERROR_UNKNOWN)
    {
        CHECK(attempts == 1);
    }
    else
    {
        CHECK(attempts > 4);
    }
    CHECK(dispatch.successfulWaits == 1);
    CHECK(dispatch.acquireDestroyed == 1);
}

TEST_CASE("Vulkan teardown retires swapchain images that still have application views",
          "[size:small][rhi][vulkan][acquire-teardown]")
{
    rachis::vulkan::VulkanSwapchainTestAccess fixture;
    const auto texture = fixture.adoptSwapchainImages();
    auto view = fixture.resources.createTextureView({
        .texture = texture,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(view);

    auto resized = fixture.manager.resizeSwapchain(fixture.handle, 32, 32);
    REQUIRE_FALSE(resized);
    CHECK(resized.error().code == rachis::ErrorCode::invalidState);
    auto destroyed = fixture.manager.destroySwapchain(fixture.handle);
    REQUIRE_FALSE(destroyed);
    CHECK(destroyed.error().code == rachis::ErrorCode::invalidState);
    CHECK(fixture.resources.resolveTexture(texture));
    CHECK(fixture.resources.resolveBindingTextureView(*view));

    fixture.tearDownDevice();
    CHECK(dispatch.imageViewsDestroyedBeforeSwapchain == 2);
    auto staleTexture = fixture.resources.resolveTexture(texture);
    REQUIRE_FALSE(staleTexture);
    CHECK(staleTexture.error().code == rachis::ErrorCode::invalidArgument);
    auto staleBinding = fixture.resources.resolveBindingTextureView(*view);
    REQUIRE_FALSE(staleBinding);
    CHECK(staleBinding.error().code == rachis::ErrorCode::invalidArgument);
    auto staleAttachment = fixture.resources.resolveRenderAttachment(*view);
    REQUIRE_FALSE(staleAttachment);
    CHECK(staleAttachment.error().code == rachis::ErrorCode::invalidArgument);
    auto recreated = fixture.resources.createTextureView({
        .texture = texture,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE_FALSE(recreated);
    CHECK(recreated.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(fixture.resources.destroyTextureView(*view));
    CHECK_FALSE(fixture.resources.destroyTextureView(*view));
    CHECK(dispatch.imageViewsDestroyed == 2);
}
#endif
