#pragma once

#include "Common/AcquireDeadline.h"
#include "Common/HandleTable.h"
#include "Common/TimedMutex.h"
#include "Internal/VulkanQueues.h"
#include "Internal/VulkanResources.h"
#include "Internal/VulkanWindowsProfile.h"
#include "Rachis/Presentation.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>
#include <volk.h>

namespace rachis::vulkan
{

#ifdef RACHIS_TESTING
    struct VulkanPresentationSyncSnapshot final
    {
        std::uint32_t imageCount = 0;
        std::uint32_t unusedAcquireSemaphores = 0;
        std::uint32_t assignedAcquireSemaphores = 0;
        std::uint32_t presentSemaphores = 0;
        std::uint32_t presentFences = 0;
        bool acquired = false;
        VulkanPresentCompletionStrategy strategy = VulkanPresentCompletionStrategy::none;
    };
#endif

    class VulkanSwapchainManager final
    {
    public:
        void initialize(VkInstance instance,
                        const VulkanInstanceTable* instanceTable,
                        VkPhysicalDevice physical,
                        VkDevice device,
                        const VolkDeviceTable* table,
                        VulkanResourceAllocator* resources,
                        VulkanQueueManager* queues,
                        std::uint32_t graphicsFamily,
                        VkQueue graphicsQueue,
                        VulkanPresentCompletionStrategy presentCompletion,
                        bool khrSwapchainMaintenance1,
                        bool presentWaitEnabled = false) noexcept;
        void shutdown() noexcept;

        void flushDeferredAcquireWaits() noexcept;

        [[nodiscard]] Result<SurfaceHandle> createSurface(const SurfaceDesc& desc);
        [[nodiscard]] Result<SurfaceCapabilities> querySurfaceCapabilities(SurfaceHandle surface) const;
        [[nodiscard]] Status destroySurface(SurfaceHandle surface);
        [[nodiscard]] Result<SwapchainHandle> createSwapchain(const SwapchainDesc& desc);
        [[nodiscard]] Status destroySwapchain(SwapchainHandle swapchain,
                                              bool deviceLost = false,
                                              bool retirePresentation = true);
        [[nodiscard]] Status resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width, std::uint32_t height);
        [[nodiscard]] Result<AcquiredSwapchainImage> acquireNextImage(
            SwapchainHandle swapchain, std::optional<std::chrono::nanoseconds> timeout = std::nullopt);
        [[nodiscard]] bool hasAcquiredImage(SwapchainHandle swapchain) const noexcept;
        [[nodiscard]] Status cancelSwapchainImage(SwapchainHandle swapchain);
        [[nodiscard]] Result<PresentStatus> present(SwapchainHandle swapchain,
                                                    TimelinePoint renderComplete,
                                                    bool& injectedDeviceLoss);
        [[nodiscard]] Result<SwapchainPresentationInfo> queryPresentation(SwapchainHandle swapchain) const;
        [[nodiscard]] Result<SwapchainFrameTiming> queryFrameTiming(SwapchainHandle swapchain) const;
#ifdef RACHIS_TESTING
        [[nodiscard]] Result<VulkanPresentationSyncSnapshot> presentationSyncSnapshot(SwapchainHandle swapchain) const;
#endif

    private:
#ifdef RACHIS_TESTING
        friend struct VulkanSwapchainTestAccess;
        friend struct VulkanCapabilitiesSwapchainTestAccess;
        friend struct VulkanPerformanceSwapchainTestAccess;
#endif
        struct SurfaceRecord final
        {
            mutable std::shared_mutex surfaceMutex;
            VkSurfaceKHR surface = VK_NULL_HANDLE;
            SurfaceKind kind = SurfaceKind::win32;
            void* window = nullptr;
            void* display = nullptr;
            std::string label;
            bool swapchainClaimed = false;
        };

        struct SwapchainImage final
        {
            VkImage image = VK_NULL_HANDLE;
            TextureHandle texture{};
            TextureViewHandle view{};
            std::shared_ptr<VulkanSwapchainImageLease> lease;
            VkSemaphore acquire = VK_NULL_HANDLE;
            TimelinePoint acquireBridge{};
            VkSemaphore present = VK_NULL_HANDLE;
            VkFence presentFence = VK_NULL_HANDLE;
            std::uint64_t presentId = 0;
            bool presentFencePending = false;
            bool contentsDefined = false;
        };

        struct RetiringAcquireSemaphore final
        {
            VkSemaphore semaphore = VK_NULL_HANDLE;
            TimelinePoint bridge{};
        };

        struct SwapchainRecord final
        {
            mutable rachis::detail::TimedMutex recordMutex;
            std::shared_ptr<SurfaceRecord> surface;
            SurfaceHandle surfaceHandle{};
            VkSwapchainKHR swapchain = VK_NULL_HANDLE;
            SwapchainDesc desc{};
            std::string label;
            std::vector<SwapchainImage> images;
            std::vector<VkSemaphore> unusedAcquire;
            std::vector<RetiringAcquireSemaphore> retiringAcquire;
            std::uint64_t generation = 1;
            std::uint32_t acquiredIndex = 0;
            bool acquired = false;
            bool cancelledAcquire = false;
            bool acquireWaitSubmitted = false;
            bool contentsDefined = false;
            bool requiresRefresh = false;
            bool suspended = false;
            bool destroyed = false;
            std::uint32_t presentsCompleted = 0;
            std::uint32_t inFlightPresents = 0;
            std::uint64_t lastAcquireNs = 0;
            std::uint64_t lastPresentSubmitNs = 0;
            std::uint64_t lastPresentIntervalNs = 0;
            std::uint64_t nextPresentId = 1;
            std::uint64_t lastPresentId = 0;
            bool hasUndefinedSurfaceExtent = false;
        };

        struct SubmittedPresent final
        {
            VulkanQueueManager::PresentResult outcome{};
            bool usesPresentFence = false;
            std::uint64_t presentId = 0;
        };

        [[nodiscard]] std::shared_ptr<SwapchainRecord> findSwapchainRecord(SwapchainHandle swapchain) const noexcept;
        [[nodiscard]] std::optional<std::shared_ptr<SwapchainRecord>> findSwapchainRecord(
            SwapchainHandle swapchain, const rachis::detail::AcquireDeadline& deadline) const;
        [[nodiscard]] Result<SurfaceCapabilities> capabilitiesFor(VkSurfaceKHR surface,
                                                                  VkSurfaceCapabilitiesKHR* nativeOut = nullptr) const;
        [[nodiscard]] Status createImages(SwapchainRecord& record);
        [[nodiscard]] AcquiredSwapchainImage describeAcquiredImage(const SwapchainRecord& record) const;
        void releaseImages(SwapchainRecord& record) noexcept;
        [[nodiscard]] Status rejectApplicationImageViews(const SwapchainRecord& record) const;
        [[nodiscard]] Status createPresentationSync(SwapchainRecord& record);
        void destroyPresentationSync(SwapchainRecord& record) noexcept;
        [[nodiscard]] Result<VkSemaphore> createBinarySemaphore();
        [[nodiscard]] Result<VkFence> createFence();
        void destroySemaphore(VkSemaphore semaphore) noexcept;
        void destroyFence(VkFence fence) noexcept;
        [[nodiscard]] Status retireBinarySemaphore(VkSemaphore semaphore);
        [[nodiscard]] Status waitPresentFence(SwapchainImage& image, std::uint64_t timeoutNs = UINT64_MAX);
        [[nodiscard]] Status waitImagePresentId(const SwapchainRecord& record,
                                                SwapchainImage& image,
                                                std::uint64_t timeoutNs);
        [[nodiscard]] Status retirePendingPresent(SwapchainRecord& record,
                                                  SwapchainImage& image,
                                                  std::uint64_t presentWaitTimeoutNs,
                                                  std::uint64_t presentFenceTimeoutNs = UINT64_MAX);
        [[nodiscard]] Status bridgeAcquire(SwapchainImage& image);
        [[nodiscard]] Result<VkSemaphore> takeAcquireSemaphore(SwapchainRecord& record,
                                                               const rachis::detail::AcquireDeadline& deadline);
        void releaseAcquireSemaphore(SwapchainRecord& record, VkSemaphore semaphore, TimelinePoint bridge) noexcept;
        void parkFailedAcquire(SwapchainRecord& record) noexcept;
        [[nodiscard]] Status submitDeferredAcquireWait(SwapchainRecord& record,
                                                       std::uint64_t presentWaitTimeoutNs,
                                                       std::uint64_t presentFenceTimeoutNs = UINT64_MAX);
        [[nodiscard]] Status waitPresentationRetirement(SwapchainRecord& record);
        [[nodiscard]] Status destroyNativeSwapchain(SwapchainRecord& record) noexcept;
        void waitPresentationIdle() noexcept;
        [[nodiscard]] bool surfaceIsHidden(const SwapchainRecord& record) const noexcept;
        [[nodiscard]] bool hiddenSurfaceHasZeroExtent(const SwapchainRecord& record) const noexcept;
        [[nodiscard]] bool surfaceExtentIsZero(const SwapchainRecord& record) const noexcept;
        [[nodiscard]] Status releaseAcquiredImage(SwapchainRecord& record);
        void recordPresentCompletion(SwapchainRecord& record) noexcept;
        [[nodiscard]] Result<SurfaceHandle> publishSurface(VkSurfaceKHR surface, const SurfaceDesc& desc);
        [[nodiscard]] Result<std::shared_ptr<SurfaceRecord>> claimSurface(SurfaceHandle surface);
        [[nodiscard]] Result<VkSwapchainKHR> createNativeSwapchain(const SurfaceRecord& surface,
                                                                   const VkSwapchainCreateInfoKHR& info);
        [[nodiscard]] Result<std::shared_ptr<SwapchainRecord>> assembleSwapchainRecord(
            const SwapchainDesc& desc,
            std::shared_ptr<SurfaceRecord> surface,
            VkSwapchainKHR swapchain,
            const VkSurfaceCapabilitiesKHR& nativeCaps);
        [[nodiscard]] Result<SwapchainHandle> publishSwapchain(const std::shared_ptr<SwapchainRecord>& record);
        [[nodiscard]] Status waitNativeGraphicsQueueIdle();
        [[nodiscard]] Status waitGraphicsAndPresentQueuesIdle();
        [[nodiscard]] Status waitLastPresentId(SwapchainRecord& record);
        void waitLastPresentDisplayed(SwapchainRecord& record, std::uint64_t timeoutNs) noexcept;
        [[nodiscard]] Result<VkSurfaceCapabilitiesKHR> nativeSurfaceCapabilities(const SwapchainRecord& record) const;
        [[nodiscard]] Status replaceNativeSwapchain(SwapchainRecord& record,
                                                    const VkSurfaceCapabilitiesKHR& nativeCaps,
                                                    VkExtent2D extent);
        [[nodiscard]] Status ensureAcquirable(SwapchainRecord& record);
        [[nodiscard]] Result<AcquiredSwapchainImage> reacquireCancelledImage(
            SwapchainRecord& record, const rachis::detail::AcquireDeadline& deadline);
        [[nodiscard]] AcquiredSwapchainImage publishAcquiredImage(SwapchainRecord& record);
        [[nodiscard]] Result<std::uint32_t> acquireImageWithPooledSemaphore(
            SwapchainRecord& record, const rachis::detail::AcquireDeadline& deadline);
        [[nodiscard]] Status settlePreviousPresent(SwapchainRecord& record,
                                                   SwapchainImage& image,
                                                   const rachis::detail::AcquireDeadline& deadline);
        [[nodiscard]] Status validatePresentable(const SwapchainRecord& record, TimelinePoint renderComplete) const;
        [[nodiscard]] bool retainImageIfWindowHidden(SwapchainRecord& record, TimelinePoint renderComplete);
        [[nodiscard]] Result<bool> releaseImageIfSurfaceHidden(SwapchainRecord& record, TimelinePoint renderComplete);
        [[nodiscard]] Result<SubmittedPresent> submitPresent(SwapchainRecord& record, TimelinePoint renderComplete);
        [[nodiscard]] Status settleSubmittedPresent(SwapchainRecord& record, const SubmittedPresent& submitted);
        [[nodiscard]] PresentStatus markImagePresented(SwapchainRecord& record, bool usesPresentFence);
        [[nodiscard]] Result<PresentStatus> presentAcquiredImage(SwapchainRecord& record,
                                                                 TimelinePoint renderComplete,
                                                                 bool& injectedDeviceLoss);

        VkInstance instance_ = VK_NULL_HANDLE;
        const VulkanInstanceTable* instanceTable_ = nullptr;
        VkPhysicalDevice physical_ = VK_NULL_HANDLE;
        VkDevice device_ = VK_NULL_HANDLE;
        const VolkDeviceTable* table_ = nullptr;
        VulkanResourceAllocator* resources_ = nullptr;
        VulkanQueueManager* queues_ = nullptr;
        std::uint32_t graphicsFamily_ = VK_QUEUE_FAMILY_IGNORED;
        VkQueue graphicsQueue_ = VK_NULL_HANDLE;
        VulkanPresentCompletionStrategy presentCompletion_ = VulkanPresentCompletionStrategy::imageLifetimeAssociation;
        bool khrSwapchainMaintenance1_ = false;
        bool presentWaitEnabled_ = false;
        PFN_vkCreateSwapchainKHR vkCreateSwapchainKHR_ = nullptr;
        PFN_vkDestroySwapchainKHR vkDestroySwapchainKHR_ = nullptr;
        PFN_vkGetSwapchainImagesKHR vkGetSwapchainImagesKHR_ = nullptr;
        PFN_vkAcquireNextImageKHR vkAcquireNextImageKHR_ = nullptr;
        PFN_vkWaitForPresentKHR vkWaitForPresentKHR_ = nullptr;
        PFN_vkVoidFunction vkReleaseSwapchainImages_ = nullptr;
        mutable rachis::detail::TimedMutex mutex_;
        detail::HandleTable<std::shared_ptr<SurfaceRecord>, SurfaceHandle> surfaces_;
        detail::HandleTable<std::shared_ptr<SwapchainRecord>, SwapchainHandle> swapchains_;
    };

} // namespace rachis::vulkan
