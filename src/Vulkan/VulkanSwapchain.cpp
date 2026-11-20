#include "Internal/VulkanSwapchain.h"

#include "Internal/TimelinePointAccess.h"
#include "Internal/VulkanDiagnostics.h"
#include "Internal/VulkanRuntime.h"
#ifdef RACHIS_TESTING
#include "Vulkan/VulkanTestHooks.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#ifdef _WIN32
#include <windows.h>
#endif

namespace rachis::vulkan
{
    namespace
    {

        [[nodiscard]] VkCompositeAlphaFlagBitsKHR chooseCompositeAlpha(VkCompositeAlphaFlagsKHR supported) noexcept
        {
            for (const auto alpha : {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                                     VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                                     VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
                                     VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR})
            {
                if ((supported & alpha) != 0)
                {
                    return alpha;
                }
            }
            return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        }

        [[nodiscard]] Error invalidSurface() noexcept
        {
            return Error{ErrorCode::invalidArgument, 0, "The surface handle is invalid or stale."};
        }

        [[nodiscard]] Error invalidSwapchain() noexcept
        {
            return Error{ErrorCode::invalidArgument, 0, "The swapchain handle is invalid or stale."};
        }

        template<typename SurfaceRecordT>
        class PendingSurfaceClaim final
        {
        public:
            PendingSurfaceClaim(std::shared_ptr<SurfaceRecordT> surface, std::mutex& mutex) noexcept
                : surface_(std::move(surface)), mutex_(&mutex)
            {
            }

            ~PendingSurfaceClaim()
            {
                if (surface_ != nullptr)
                {
                    const std::scoped_lock lock(*mutex_);
                    surface_->swapchainClaimed = false;
                }
            }

            PendingSurfaceClaim(const PendingSurfaceClaim&) = delete;
            PendingSurfaceClaim& operator=(const PendingSurfaceClaim&) = delete;
            PendingSurfaceClaim(PendingSurfaceClaim&&) = delete;
            PendingSurfaceClaim& operator=(PendingSurfaceClaim&&) = delete;

            void publish() noexcept
            {
                surface_.reset();
            }

        private:
            std::shared_ptr<SurfaceRecordT> surface_;
            std::mutex* mutex_ = nullptr;
        };

        template<typename SurfaceRecordT>
        [[nodiscard]] bool sharesNativeWindow(const SurfaceRecordT& left, const SurfaceRecordT& right) noexcept
        {
            return left.kind == right.kind && left.window == right.window &&
                   (left.kind == SurfaceKind::win32 || left.display == right.display);
        }

#ifndef _WIN32
        struct WaylandSurfaceCreateInfo final
        {
            VkStructureType sType = static_cast<VkStructureType>(1000006000);
            const void* pNext = nullptr;
            VkFlags flags = 0;
            void* display = nullptr;
            void* surface = nullptr;
        };

        struct XcbSurfaceCreateInfo final
        {
            VkStructureType sType = static_cast<VkStructureType>(1000005000);
            const void* pNext = nullptr;
            VkFlags flags = 0;
            void* connection = nullptr;
            std::uint32_t window = 0;
        };

        using PFN_CreateWaylandSurfaceKHR = VkResult(VKAPI_PTR*)(VkInstance,
                                                                 const WaylandSurfaceCreateInfo*,
                                                                 const VkAllocationCallbacks*,
                                                                 VkSurfaceKHR*);
        using PFN_CreateXcbSurfaceKHR = VkResult(VKAPI_PTR*)(VkInstance,
                                                             const XcbSurfaceCreateInfo*,
                                                             const VkAllocationCallbacks*,
                                                             VkSurfaceKHR*);
#endif

        [[nodiscard]] Error unsupportedWindowSystem(std::string_view message) noexcept
        {
            return Error{ErrorCode::unsupported, 0, std::string(message)};
        }

        [[nodiscard]] VkColorSpaceKHR nativeColorSpace(SurfaceColorSpace space) noexcept
        {
            switch (space)
            {
            case SurfaceColorSpace::extendedSrgbLinear:
                return VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT;
            case SurfaceColorSpace::hdr10St2084:
                return VK_COLOR_SPACE_HDR10_ST2084_EXT;
            case SurfaceColorSpace::srgbNonlinear:
            default:
                return VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
            }
        }

        [[nodiscard]] std::optional<SurfaceColorSpace> portableColorSpace(VkColorSpaceKHR space) noexcept
        {
            switch (space)
            {
            case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:
                return SurfaceColorSpace::srgbNonlinear;
            case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT:
                return SurfaceColorSpace::extendedSrgbLinear;
            case VK_COLOR_SPACE_HDR10_ST2084_EXT:
                return SurfaceColorSpace::hdr10St2084;
            default:
                return std::nullopt;
            }
        }

        [[nodiscard]] std::optional<Format> portableFormat(VkFormat format) noexcept
        {
            switch (format)
            {
            case VK_FORMAT_B8G8R8A8_UNORM:
                return Format::bgra8Unorm;
            case VK_FORMAT_B8G8R8A8_SRGB:
                return Format::bgra8UnormSrgb;
            case VK_FORMAT_R8G8B8A8_UNORM:
                return Format::rgba8Unorm;
            case VK_FORMAT_R8G8B8A8_SRGB:
                return Format::rgba8UnormSrgb;
            default:
                return std::nullopt;
            }
        }

        [[nodiscard]] TextureUsage portableUsage(VkImageUsageFlags usage) noexcept
        {
            TextureUsage result = TextureUsage::none;
            if ((usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0)
            {
                result = result | TextureUsage::renderTarget;
            }
            if ((usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0)
            {
                result = result | TextureUsage::copySource;
            }
            if ((usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0)
            {
                result = result | TextureUsage::copyDestination;
            }
            if ((usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0)
            {
                result = result | TextureUsage::sampled;
            }
            return result;
        }

        [[nodiscard]] VkImageUsageFlags nativeUsage(TextureUsage usage) noexcept
        {
            VkImageUsageFlags flags = 0;
            if (hasAny(usage, TextureUsage::renderTarget))
            {
                flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            }
            if (hasAny(usage, TextureUsage::copySource))
            {
                flags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            }
            if (hasAny(usage, TextureUsage::copyDestination))
            {
                flags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            }
            if (hasAny(usage, TextureUsage::sampled))
            {
                flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
            }
            return flags;
        }

        [[nodiscard]] VkPresentModeKHR nativePresentMode(PresentMode mode) noexcept
        {
            return mode == PresentMode::immediate ? VK_PRESENT_MODE_IMMEDIATE_KHR : VK_PRESENT_MODE_FIFO_KHR;
        }

        [[nodiscard]] bool usageIsSubset(TextureUsage requested, TextureUsage available) noexcept
        {
            return (static_cast<std::uint32_t>(requested) & ~static_cast<std::uint32_t>(available)) == 0;
        }

        [[nodiscard]] PFN_vkVoidFunction loadInstanceFunction(VkInstance instance,
                                                              const VulkanInstanceTable* instanceTable,
                                                              const char* name)
        {
            if (instance == VK_NULL_HANDLE || instanceTable == nullptr ||
                instanceTable->vkGetInstanceProcAddr == nullptr)
            {
                return nullptr;
            }
            return instanceTable->vkGetInstanceProcAddr(instance, name);
        }

#ifdef _WIN32
        [[nodiscard]] Result<VkSurfaceKHR> createWin32Surface(VkInstance instance,
                                                              const VulkanInstanceTable* instanceTable,
                                                              const SurfaceDesc& desc)
        {
            auto* const window = static_cast<HWND>(desc.windowHandle);
            if (window == nullptr || IsWindow(window) == FALSE)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A Win32 surface requires a live HWND.",
                });
            }
            const auto createSurfaceFunction = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
                loadInstanceFunction(instance, instanceTable, "vkCreateWin32SurfaceKHR"));
            if (createSurfaceFunction == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::backendUnavailable,
                    0,
                    "The Vulkan instance does not expose Win32 surfaces.",
                });
            }

            const VkWin32SurfaceCreateInfoKHR info{
                .sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
                .pNext = {},
                .flags = {},
                .hinstance = GetModuleHandleW(nullptr),
                .hwnd = window,
            };
            VkSurfaceKHR surface = VK_NULL_HANDLE;
            const auto created = createSurfaceFunction(instance, &info, nullptr, &surface);
            if (created != VK_SUCCESS)
            {
                return std::unexpected(
                    makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateWin32SurfaceKHR"));
            }
            return surface;
        }
#else
        [[nodiscard]] Result<VkSurfaceKHR> createWaylandSurface(VkInstance instance,
                                                                const VulkanInstanceTable* instanceTable,
                                                                const SurfaceDesc& desc)
        {
            if (desc.windowHandle == nullptr || desc.displayHandle == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A Wayland surface requires a host-owned wl_display and wl_surface.",
                });
            }
            const auto createSurfaceFunction = reinterpret_cast<PFN_CreateWaylandSurfaceKHR>(
                loadInstanceFunction(instance, instanceTable, "vkCreateWaylandSurfaceKHR"));
            if (createSurfaceFunction == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::backendUnavailable,
                    0,
                    "The Vulkan instance does not expose Wayland surfaces.",
                });
            }
            const WaylandSurfaceCreateInfo info{
                .display = desc.displayHandle,
                .surface = desc.windowHandle,
            };
            VkSurfaceKHR surface = VK_NULL_HANDLE;
            const auto created = createSurfaceFunction(instance, &info, nullptr, &surface);
            if (created != VK_SUCCESS)
            {
                return std::unexpected(
                    makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateWaylandSurfaceKHR"));
            }
            return surface;
        }

        [[nodiscard]] Result<VkSurfaceKHR> createXcbSurface(VkInstance instance,
                                                            const VulkanInstanceTable* instanceTable,
                                                            const SurfaceDesc& desc)
        {
            if (desc.windowHandle == nullptr || desc.displayHandle == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "An XCB surface requires a host-owned xcb_connection and xcb_window.",
                });
            }
            const auto createSurfaceFunction = reinterpret_cast<PFN_CreateXcbSurfaceKHR>(
                loadInstanceFunction(instance, instanceTable, "vkCreateXcbSurfaceKHR"));
            if (createSurfaceFunction == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::backendUnavailable,
                    0,
                    "The Vulkan instance does not expose XCB surfaces.",
                });
            }
            const XcbSurfaceCreateInfo info{
                .connection = desc.displayHandle,
                .window = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(desc.windowHandle)),
            };
            VkSurfaceKHR surface = VK_NULL_HANDLE;
            const auto created = createSurfaceFunction(instance, &info, nullptr, &surface);
            if (created != VK_SUCCESS)
            {
                return std::unexpected(
                    makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateXcbSurfaceKHR"));
            }
            return surface;
        }
#endif

        [[nodiscard]] Result<VkSurfaceKHR> createNativeSurface(VkInstance instance,
                                                               const VulkanInstanceTable* instanceTable,
                                                               const SurfaceDesc& desc)
        {
            switch (desc.kind)
            {
            case SurfaceKind::win32:
#ifdef _WIN32
                return createWin32Surface(instance, instanceTable, desc);
#else
                return std::unexpected(unsupportedWindowSystem(
                    "Win32 surfaces are Windows-only. Selection did not fall back to Wayland or XCB."));
#endif
            case SurfaceKind::wayland:
#ifdef _WIN32
                return std::unexpected(
                    unsupportedWindowSystem("Wayland surfaces are Linux-only. Selection did not fall back to Win32."));
#else
                return createWaylandSurface(instance, instanceTable, desc);
#endif
            case SurfaceKind::xcb:
#ifdef _WIN32
                return std::unexpected(
                    unsupportedWindowSystem("XCB surfaces are Linux-only. Selection did not fall back to Win32."));
#else
                return createXcbSurface(instance, instanceTable, desc);
#endif
            case SurfaceKind::caMetalLayer:
                return std::unexpected(unsupportedWindowSystem(
                    "CAMetalLayer surfaces are Metal-only. Selection did not fall back to Vulkan WSI."));
            }

            return std::unexpected(unsupportedWindowSystem(
                "Unsupported window system. Declared Linux paths are Wayland and XCB; Xlib is not implemented."));
        }

        [[nodiscard]] Status validateSwapchainDesc(const SwapchainDesc& desc, const SurfaceCapabilities& capabilities)
        {
            const bool imageCountOutOfRange =
                desc.imageCount < capabilities.minImageCount || desc.imageCount > capabilities.maxImageCount;
            if (desc.width == 0 || desc.height == 0 || imageCountOutOfRange ||
                !hasAny(desc.usage, TextureUsage::renderTarget))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A swapchain requires a supported non-empty extent and image count plus render-target usage.",
                });
            }
            const bool formatSupported =
                std::any_of(capabilities.formats.begin(), capabilities.formats.end(), [&](const SurfaceFormat& format) {
                    return format.format == desc.format && format.colorSpace == desc.colorSpace;
                });
            if (!formatSupported || !supportsPresentMode(capabilities.presentModes, desc.presentMode) ||
                !usageIsSubset(desc.usage, capabilities.usages) || vulkanFormat(desc.format) == VK_FORMAT_UNDEFINED)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The swapchain format, color space, present mode, or usage is not supported by this surface.",
                });
            }
            return {};
        }

        [[nodiscard]] VkSwapchainCreateInfoKHR swapchainCreateInfo(const SwapchainDesc& desc,
                                                                   VkSurfaceKHR surface,
                                                                   const VkSurfaceCapabilitiesKHR& nativeCaps) noexcept
        {
            return VkSwapchainCreateInfoKHR{
                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                .pNext = {},
                .flags = {},
                .surface = surface,
                .minImageCount = desc.imageCount,
                .imageFormat = vulkanFormat(desc.format),
                .imageColorSpace = nativeColorSpace(desc.colorSpace),
                .imageExtent = {.width = desc.width, .height = desc.height},
                .imageArrayLayers = 1,
                .imageUsage = nativeUsage(desc.usage),
                .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = {},
                .pQueueFamilyIndices = {},
                .preTransform = nativeCaps.currentTransform,
                .compositeAlpha = chooseCompositeAlpha(nativeCaps.supportedCompositeAlpha),
                .presentMode = nativePresentMode(desc.presentMode),
                .clipped = VK_TRUE,
                .oldSwapchain = {},
            };
        }

        [[nodiscard]] bool hasUndefinedExtent(const VkSurfaceCapabilitiesKHR& nativeCaps) noexcept
        {
            return nativeCaps.currentExtent.width == 0xffffffffU && nativeCaps.currentExtent.height == 0xffffffffU;
        }

        [[nodiscard]] Result<VkExtent2D> resolveResizeExtent(const VkSurfaceCapabilitiesKHR& nativeCaps,
                                                             std::uint32_t width,
                                                             std::uint32_t height)
        {
            const bool currentExtentFixed =
                nativeCaps.currentExtent.width != 0xffffffffU && nativeCaps.currentExtent.height != 0xffffffffU;
            if (currentExtentFixed)
            {
                return nativeCaps.currentExtent;
            }
            const bool requestedOutOfRange =
                width < nativeCaps.minImageExtent.width || height < nativeCaps.minImageExtent.height ||
                width > nativeCaps.maxImageExtent.width || height > nativeCaps.maxImageExtent.height;
            if (requestedOutOfRange)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Swapchain resize dimensions are outside the surface capability range.",
                });
            }
            return VkExtent2D{.width = width, .height = height};
        }

        [[nodiscard]] Result<VkExtent2D> resolveCreateExtent(const VkSurfaceCapabilitiesKHR& nativeCaps,
                                                             std::uint32_t width,
                                                             std::uint32_t height)
        {
            if (nativeCaps.currentExtent.width == 0 || nativeCaps.currentExtent.height == 0)
            {
                return VkExtent2D{};
            }
            auto extent = resolveResizeExtent(nativeCaps, width, height);
            if (!extent)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The swapchain extent is outside the surface capability range.",
                });
            }
            return *extent;
        }

        constexpr std::uint64_t acquireTimeoutNs = 50'000'000;

        [[nodiscard]] bool presentWaitSettled(VkResult result) noexcept
        {
            return result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR || result == VK_ERROR_OUT_OF_DATE_KHR ||
                   result == VK_ERROR_SURFACE_LOST_KHR;
        }

        [[nodiscard]] std::uint64_t steadyClockNowNs() noexcept
        {
            return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                  std::chrono::steady_clock::now().time_since_epoch())
                                                  .count());
        }

        class PresentCompletionChain final
        {
        public:
            PresentCompletionChain() noexcept = default;
            PresentCompletionChain(const PresentCompletionChain&) = delete;
            PresentCompletionChain& operator=(const PresentCompletionChain&) = delete;
            PresentCompletionChain(PresentCompletionChain&&) = delete;
            PresentCompletionChain& operator=(PresentCompletionChain&&) = delete;

            void attachPresentFence([[maybe_unused]] const VkFence* presentFence,
                                    [[maybe_unused]] bool preferKhrStruct) noexcept
            {
#ifdef VK_KHR_swapchain_maintenance1
                if (preferKhrStruct)
                {
                    khrFenceInfo_ = VkSwapchainPresentFenceInfoKHR{
                        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR,
                        .pNext = {},
                        .swapchainCount = 1,
                        .pFences = presentFence,
                    };
                    next_ = &khrFenceInfo_;
                    usesPresentFence_ = true;
                }
#endif
#ifdef VK_EXT_swapchain_maintenance1
                if (!usesPresentFence_)
                {
                    extFenceInfo_ = VkSwapchainPresentFenceInfoEXT{
                        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT,
                        .pNext = {},
                        .swapchainCount = 1,
                        .pFences = presentFence,
                    };
                    next_ = &extFenceInfo_;
                    usesPresentFence_ = true;
                }
#endif
            }

#ifdef VK_KHR_present_id
            void tagPresentId(std::uint64_t presentId) noexcept
            {
                presentId_ = presentId;
                presentIdInfo_ = VkPresentIdKHR{
                    .sType = VK_STRUCTURE_TYPE_PRESENT_ID_KHR,
                    .pNext = next_,
                    .swapchainCount = 1,
                    .pPresentIds = &presentId_,
                };
                next_ = &presentIdInfo_;
            }
#endif

            [[nodiscard]] const void* next() const noexcept
            {
                return next_;
            }
            [[nodiscard]] bool usesPresentFence() const noexcept
            {
                return usesPresentFence_;
            }
            [[nodiscard]] std::uint64_t presentId() const noexcept
            {
                return presentId_;
            }

        private:
            const void* next_ = nullptr;
            bool usesPresentFence_ = false;
            std::uint64_t presentId_ = 0;
#ifdef VK_KHR_swapchain_maintenance1
            VkSwapchainPresentFenceInfoKHR khrFenceInfo_{};
#endif
#ifdef VK_EXT_swapchain_maintenance1
            VkSwapchainPresentFenceInfoEXT extFenceInfo_{};
#endif
#ifdef VK_KHR_present_id
            VkPresentIdKHR presentIdInfo_{};
#endif
        };

        [[nodiscard]] Error presentFailure(VkResult result, bool injectedDeviceLoss)
        {
            if (result == VK_TIMEOUT)
            {
                return Error{ErrorCode::nativeFailure, result, "vkQueuePresentKHR timed out."};
            }
            return makeVulkanError(vulkanResultErrorCode(result),
                                   result,
                                   injectedDeviceLoss ? "vkQueuePresentKHR (injected)" : "vkQueuePresentKHR");
        }
    } // namespace

    void VulkanSwapchainManager::initialize(VkInstance instance,
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
                                            bool presentWaitEnabled) noexcept
    {
        instance_ = instance;
        instanceTable_ = instanceTable;
        physical_ = physical;
        device_ = device;
        table_ = table;
        resources_ = resources;
        queues_ = queues;
        graphicsFamily_ = graphicsFamily;
        graphicsQueue_ = graphicsQueue;
        presentCompletion_ = presentCompletion;
        khrSwapchainMaintenance1_ = khrSwapchainMaintenance1;
        presentWaitEnabled_ = presentWaitEnabled;
        const auto load = [instanceTable, device](const char* name) -> PFN_vkVoidFunction {
            if (device == VK_NULL_HANDLE || instanceTable == nullptr || instanceTable->vkGetDeviceProcAddr == nullptr)
            {
                return nullptr;
            }
            return instanceTable->vkGetDeviceProcAddr(device, name);
        };
        vkCreateSwapchainKHR_ = table != nullptr && table->vkCreateSwapchainKHR != nullptr
                                    ? table->vkCreateSwapchainKHR
                                    : reinterpret_cast<PFN_vkCreateSwapchainKHR>(load("vkCreateSwapchainKHR"));
        vkDestroySwapchainKHR_ = table != nullptr && table->vkDestroySwapchainKHR != nullptr
                                     ? table->vkDestroySwapchainKHR
                                     : reinterpret_cast<PFN_vkDestroySwapchainKHR>(load("vkDestroySwapchainKHR"));
        vkGetSwapchainImagesKHR_ = table != nullptr && table->vkGetSwapchainImagesKHR != nullptr
                                       ? table->vkGetSwapchainImagesKHR
                                       : reinterpret_cast<PFN_vkGetSwapchainImagesKHR>(load("vkGetSwapchainImagesKHR"));
        vkAcquireNextImageKHR_ = table != nullptr && table->vkAcquireNextImageKHR != nullptr
                                     ? table->vkAcquireNextImageKHR
                                     : reinterpret_cast<PFN_vkAcquireNextImageKHR>(load("vkAcquireNextImageKHR"));
        vkWaitForPresentKHR_ = table != nullptr && table->vkWaitForPresentKHR != nullptr
                                   ? table->vkWaitForPresentKHR
                                   : reinterpret_cast<PFN_vkWaitForPresentKHR>(load("vkWaitForPresentKHR"));
        vkReleaseSwapchainImages_ = load("vkReleaseSwapchainImagesKHR");
        if (vkReleaseSwapchainImages_ == nullptr)
        {
            vkReleaseSwapchainImages_ = table != nullptr && table->vkReleaseSwapchainImagesEXT != nullptr
                                            ? reinterpret_cast<PFN_vkVoidFunction>(table->vkReleaseSwapchainImagesEXT)
                                            : load("vkReleaseSwapchainImagesEXT");
        }
    }

    std::shared_ptr<VulkanSwapchainManager::SwapchainRecord> VulkanSwapchainManager::findSwapchainRecord(
        SwapchainHandle swapchain) const noexcept
    {
        const std::scoped_lock lock(mutex_);
        const auto* found = swapchains_.find(swapchain);
        return found != nullptr ? *found : nullptr;
    }

    void VulkanSwapchainManager::shutdown() noexcept
    {
        const std::scoped_lock lock(mutex_);
        waitPresentationIdle();
        swapchains_.forEach([this](SwapchainHandle, std::shared_ptr<SwapchainRecord>& record) {
            if (record != nullptr)
            {
                const std::scoped_lock recordLock(record->recordMutex);
                static_cast<void>(destroyNativeSwapchain(*record));
                record->destroyed = true;
                if (record->surface != nullptr)
                {
                    record->surface->swapchainClaimed = false;
                }
            }
        });
        swapchains_.clear();
        surfaces_.forEach([this](SurfaceHandle, std::shared_ptr<SurfaceRecord>& record) {
            if (record != nullptr && instance_ != VK_NULL_HANDLE && instanceTable_ != nullptr &&
                instanceTable_->vkDestroySurfaceKHR != nullptr && record->surface != VK_NULL_HANDLE)
            {
                instanceTable_->vkDestroySurfaceKHR(instance_, record->surface, nullptr);
                record->surface = VK_NULL_HANDLE;
            }
        });
        surfaces_.clear();
        instance_ = VK_NULL_HANDLE;
        instanceTable_ = nullptr;
        physical_ = VK_NULL_HANDLE;
        device_ = VK_NULL_HANDLE;
        table_ = nullptr;
        resources_ = nullptr;
        queues_ = nullptr;
        vkCreateSwapchainKHR_ = nullptr;
        vkDestroySwapchainKHR_ = nullptr;
        vkGetSwapchainImagesKHR_ = nullptr;
        vkAcquireNextImageKHR_ = nullptr;
        vkWaitForPresentKHR_ = nullptr;
        vkReleaseSwapchainImages_ = nullptr;
    }

    Result<SurfaceHandle> VulkanSwapchainManager::publishSurface(VkSurfaceKHR surface, const SurfaceDesc& desc)
    {
        if (instanceTable_ == nullptr || instanceTable_->vkGetPhysicalDeviceSurfaceSupportKHR == nullptr ||
            instanceTable_->vkDestroySurfaceKHR == nullptr)
        {
            if (instance_ != VK_NULL_HANDLE && instanceTable_ != nullptr &&
                instanceTable_->vkDestroySurfaceKHR != nullptr)
            {
                instanceTable_->vkDestroySurfaceKHR(instance_, surface, nullptr);
            }
            return std::unexpected(Error{
                ErrorCode::backendUnavailable,
                0,
                "The Vulkan instance does not expose VK_KHR_surface queries.",
            });
        }
        VkBool32 present = VK_FALSE;
        const auto presentResult =
            instanceTable_->vkGetPhysicalDeviceSurfaceSupportKHR(physical_, graphicsFamily_, surface, &present);
        if (presentResult != VK_SUCCESS || present != VK_TRUE)
        {
            instanceTable_->vkDestroySurfaceKHR(instance_, surface, nullptr);
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The graphics queue family cannot present to this surface.",
            });
        }

        auto record = std::make_shared<SurfaceRecord>();
        record->surface = surface;
        record->kind = desc.kind;
        record->window = desc.windowHandle;
        record->display = desc.displayHandle;
        record->label = std::string(desc.label);
        const std::scoped_lock lock(mutex_);
        return surfaces_.emplace(std::move(record));
    }

    Result<SurfaceHandle> VulkanSwapchainManager::createSurface(const SurfaceDesc& desc)
    {
        auto surface = createNativeSurface(instance_, instanceTable_, desc);
        if (!surface)
        {
            return std::unexpected(std::move(surface.error()));
        }
        return publishSurface(*surface, desc);
    }

    Result<SurfaceCapabilities> VulkanSwapchainManager::querySurfaceCapabilities(SurfaceHandle surface) const
    {
        std::shared_ptr<SurfaceRecord> record;
        {
            const std::scoped_lock lock(mutex_);
            const auto* found = surfaces_.find(surface);
            if (found == nullptr)
            {
                return std::unexpected(invalidSurface());
            }
            record = *found;
        }
        const std::shared_lock surfaceLock(record->surfaceMutex);
        if (record->surface == VK_NULL_HANDLE)
        {
            return std::unexpected(invalidSurface());
        }
        return capabilitiesFor(record->surface);
    }

    Status VulkanSwapchainManager::destroySurface(SurfaceHandle surface)
    {
        std::shared_ptr<SurfaceRecord> record;
        {
            const std::scoped_lock lock(mutex_);
            auto* found = surfaces_.find(surface);
            if (found == nullptr)
            {
                return std::unexpected(invalidSurface());
            }
            if ((*found)->swapchainClaimed)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A surface with a live swapchain cannot be destroyed.",
                });
            }
            record = *found;
            surfaces_.erase(surface);
        }
        const std::unique_lock surfaceLock(record->surfaceMutex);
        if (instance_ != VK_NULL_HANDLE && instanceTable_ != nullptr &&
            instanceTable_->vkDestroySurfaceKHR != nullptr && record->surface != VK_NULL_HANDLE)
        {
            instanceTable_->vkDestroySurfaceKHR(instance_, record->surface, nullptr);
            record->surface = VK_NULL_HANDLE;
        }
        return {};
    }

    Result<SurfaceCapabilities> VulkanSwapchainManager::capabilitiesFor(VkSurfaceKHR surface,
                                                                        VkSurfaceCapabilitiesKHR* nativeOut) const
    {
        if (instanceTable_ == nullptr || instanceTable_->vkGetPhysicalDeviceSurfaceCapabilitiesKHR == nullptr ||
            instanceTable_->vkGetPhysicalDeviceSurfaceFormatsKHR == nullptr ||
            instanceTable_->vkGetPhysicalDeviceSurfacePresentModesKHR == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::backendUnavailable,
                0,
                "The Vulkan instance does not expose VK_KHR_surface capability queries.",
            });
        }

        VkSurfaceCapabilitiesKHR native{};
        const auto capsResult = instanceTable_->vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_, surface, &native);
        if (capsResult != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(
                vulkanResultErrorCode(capsResult), capsResult, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"));
        }
        if (nativeOut != nullptr)
        {
            *nativeOut = native;
        }

        std::uint32_t formatCount = 0;
        const auto formatCountResult =
            instanceTable_->vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface, &formatCount, nullptr);
        if (formatCountResult != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(
                vulkanResultErrorCode(formatCountResult), formatCountResult, "vkGetPhysicalDeviceSurfaceFormatsKHR"));
        }
        std::vector<VkSurfaceFormatKHR> nativeFormats(formatCount);
        if (formatCount != 0)
        {
            const auto formatsResult = instanceTable_->vkGetPhysicalDeviceSurfaceFormatsKHR(
                physical_, surface, &formatCount, nativeFormats.data());
            if (formatsResult != VK_SUCCESS)
            {
                return std::unexpected(makeVulkanError(
                    vulkanResultErrorCode(formatsResult), formatsResult, "vkGetPhysicalDeviceSurfaceFormatsKHR"));
            }
        }

        std::uint32_t modeCount = 0;
        const auto modeCountResult =
            instanceTable_->vkGetPhysicalDeviceSurfacePresentModesKHR(physical_, surface, &modeCount, nullptr);
        if (modeCountResult != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(
                vulkanResultErrorCode(modeCountResult), modeCountResult, "vkGetPhysicalDeviceSurfacePresentModesKHR"));
        }
        std::vector<VkPresentModeKHR> nativeModes(modeCount);
        if (modeCount != 0)
        {
            const auto modesResult = instanceTable_->vkGetPhysicalDeviceSurfacePresentModesKHR(
                physical_, surface, &modeCount, nativeModes.data());
            if (modesResult != VK_SUCCESS)
            {
                return std::unexpected(makeVulkanError(
                    vulkanResultErrorCode(modesResult), modesResult, "vkGetPhysicalDeviceSurfacePresentModesKHR"));
            }
        }

        SurfaceCapabilities capabilities{};
        capabilities.minImageCount = native.minImageCount;
        capabilities.maxImageCount = native.maxImageCount == 0 ? 16 : native.maxImageCount;
        capabilities.minWidth = native.minImageExtent.width;
        capabilities.minHeight = native.minImageExtent.height;
        capabilities.maxWidth = native.maxImageExtent.width;
        capabilities.maxHeight = native.maxImageExtent.height;
        if (native.currentExtent.width != 0xffffffffU && native.currentExtent.height != 0xffffffffU)
        {
            capabilities.preferredWidth = native.currentExtent.width;
            capabilities.preferredHeight = native.currentExtent.height;
        }
        capabilities.usages = portableUsage(native.supportedUsageFlags);
        capabilities.presentModes = PresentModeMask::fifo;
        for (const auto mode : nativeModes)
        {
            if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
            {
                capabilities.presentModes |= PresentModeMask::immediate;
            }
        }
        for (const auto& format : nativeFormats)
        {
            const auto portable = portableFormat(format.format);
            const auto space = portableColorSpace(format.colorSpace);
            if (portable && space)
            {
                capabilities.formats.push_back(SurfaceFormat{.format = *portable, .colorSpace = *space});
            }
        }
        return capabilities;
    }

    Status VulkanSwapchainManager::createImages(SwapchainRecord& record)
    {
        if (resources_ == nullptr || vkGetSwapchainImagesKHR_ == nullptr)
        {
            return std::unexpected(Error{ErrorCode::backendUnavailable, 0, "Swapchain image query is unavailable."});
        }
        std::uint32_t count = 0;
        const auto countResult = vkGetSwapchainImagesKHR_(device_, record.swapchain, &count, nullptr);
        if (countResult != VK_SUCCESS)
        {
            return std::unexpected(
                makeVulkanError(vulkanResultErrorCode(countResult), countResult, "vkGetSwapchainImagesKHR"));
        }
        std::vector<VkImage> images(count);
        const auto imagesResult = vkGetSwapchainImagesKHR_(device_, record.swapchain, &count, images.data());
        if (imagesResult != VK_SUCCESS)
        {
            return std::unexpected(
                makeVulkanError(vulkanResultErrorCode(imagesResult), imagesResult, "vkGetSwapchainImagesKHR"));
        }
        record.images.resize(count);
        const TextureDesc textureDesc{
            .width = record.desc.width,
            .height = record.desc.height,
            .format = record.desc.format,
            .usage = record.desc.usage,
            .label = record.label,
        };
        for (std::uint32_t index = 0; index < count; ++index)
        {
            auto lease = std::make_shared<VulkanSwapchainImageLease>();
            auto texture = resources_->adoptSwapchainImage(images[index], textureDesc, lease);
            if (!texture)
            {
                releaseImages(record);
                return std::unexpected(std::move(texture.error()));
            }
            auto view = resources_->createTextureView({
                .texture = *texture,
                .usage = TextureViewUsage::renderTarget,
            });
            if (!view)
            {
                static_cast<void>(resources_->forgetSwapchainImage(*texture));
                releaseImages(record);
                return std::unexpected(std::move(view.error()));
            }
            record.images[index] = SwapchainImage{
                .image = images[index],
                .texture = *texture,
                .view = *view,
                .lease = std::move(lease),
            };
        }
        return {};
    }

    void VulkanSwapchainManager::releaseImages(SwapchainRecord& record) noexcept
    {
        if (resources_ == nullptr)
        {
            record.images.clear();
            return;
        }
        for (auto& image : record.images)
        {
            if (image.view.isValid())
            {
                static_cast<void>(resources_->destroyTextureView(image.view));
            }
            if (image.texture.isValid())
            {
                auto forgotten = resources_->forgetSwapchainImage(image.texture);
                if (!forgotten && forgotten.error().code == ErrorCode::invalidState)
                {
                    resources_->retireSwapchainImage(image.texture);
                }
            }
        }
        record.images.clear();
    }

    Status VulkanSwapchainManager::rejectApplicationImageViews(const SwapchainRecord& record) const
    {
        if (resources_ == nullptr)
        {
            return {};
        }
        for (const auto& image : record.images)
        {
            if (image.texture.isValid() && resources_->hasOtherTextureViews(image.texture, image.view))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Destroy views of swapchain images before resizing or destroying the swapchain.",
                });
            }
        }
        return {};
    }

    Result<VkSemaphore> VulkanSwapchainManager::createBinarySemaphore()
    {
        if (table_ == nullptr || table_->vkCreateSemaphore == nullptr)
        {
            return std::unexpected(Error{ErrorCode::backendUnavailable, 0, "vkCreateSemaphore is unavailable."});
        }
        const VkSemaphoreCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = {},
            .flags = {},
        };
        VkSemaphore semaphore = VK_NULL_HANDLE;
        const auto created = table_->vkCreateSemaphore(device_, &info, nullptr, &semaphore);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateSemaphore"));
        }
        return semaphore;
    }

    Result<VkFence> VulkanSwapchainManager::createFence()
    {
        if (table_ == nullptr || table_->vkCreateFence == nullptr)
        {
            return std::unexpected(Error{ErrorCode::backendUnavailable, 0, "vkCreateFence is unavailable."});
        }
        const VkFenceCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = {},
            .flags = {},
        };
        VkFence fence = VK_NULL_HANDLE;
        const auto created = table_->vkCreateFence(device_, &info, nullptr, &fence);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateFence"));
        }
        return fence;
    }

    void VulkanSwapchainManager::destroySemaphore(VkSemaphore semaphore) noexcept
    {
        if (table_ != nullptr && table_->vkDestroySemaphore != nullptr && semaphore != VK_NULL_HANDLE)
        {
            table_->vkDestroySemaphore(device_, semaphore, nullptr);
        }
    }

    void VulkanSwapchainManager::destroyFence(VkFence fence) noexcept
    {
        if (table_ != nullptr && table_->vkDestroyFence != nullptr && fence != VK_NULL_HANDLE)
        {
            table_->vkDestroyFence(device_, fence, nullptr);
        }
    }

    void VulkanSwapchainManager::destroyPresentationSync(SwapchainRecord& record) noexcept
    {
        for (auto* semaphore : record.unusedAcquire)
        {
            destroySemaphore(semaphore);
        }
        record.unusedAcquire.clear();
        for (const auto& retiring : record.retiringAcquire)
        {
            destroySemaphore(retiring.semaphore);
        }
        record.retiringAcquire.clear();
        for (auto& image : record.images)
        {
            destroySemaphore(image.acquire);
            destroySemaphore(image.present);
            destroyFence(image.presentFence);
            image.acquire = VK_NULL_HANDLE;
            image.acquireBridge = {};
            image.present = VK_NULL_HANDLE;
            image.presentFence = VK_NULL_HANDLE;
            image.presentId = 0;
            image.presentFencePending = false;
        }
    }

    Status VulkanSwapchainManager::createPresentationSync(SwapchainRecord& record)
    {
        destroyPresentationSync(record);
        const auto imageCount = static_cast<std::uint32_t>(record.images.size());
        if (imageCount == 0)
        {
            return std::unexpected(Error{ErrorCode::invalidState, 0, "A swapchain must expose at least one image."});
        }
        const bool usePresentFence =
            presentCompletion_ == VulkanPresentCompletionStrategy::swapchainMaintenancePresentFence;
        for (auto& image : record.images)
        {
            auto present = createBinarySemaphore();
            if (!present)
            {
                destroyPresentationSync(record);
                return std::unexpected(std::move(present.error()));
            }
            image.present = *present;
            if (usePresentFence)
            {
                auto fence = createFence();
                if (!fence)
                {
                    destroyPresentationSync(record);
                    return std::unexpected(std::move(fence.error()));
                }
                image.presentFence = *fence;
            }
        }
        try
        {
            record.unusedAcquire.reserve(static_cast<std::size_t>(imageCount) + 1);
            record.retiringAcquire.reserve(static_cast<std::size_t>(imageCount) + 1);
        }
        catch (const std::bad_alloc&)
        {
            destroyPresentationSync(record);
            return std::unexpected(Error{ErrorCode::outOfMemory, 0, "Failed to reserve the acquire semaphore pool."});
        }
        for (std::uint32_t index = 0; index < imageCount + 1; ++index)
        {
            auto acquire = createBinarySemaphore();
            if (!acquire)
            {
                destroyPresentationSync(record);
                return std::unexpected(std::move(acquire.error()));
            }
            record.unusedAcquire.push_back(*acquire);
        }
        return {};
    }

    Status VulkanSwapchainManager::retireBinarySemaphore(VkSemaphore semaphore)
    {
        if (semaphore == VK_NULL_HANDLE || queues_ == nullptr)
        {
            return {};
        }
        return queues_->submitPresentPresentationSync({.waitBinary = semaphore});
    }

    Status VulkanSwapchainManager::waitPresentFence(SwapchainImage& image)
    {
        if (image.presentFence == VK_NULL_HANDLE || !image.presentFencePending)
        {
            return {};
        }
        if (table_ == nullptr || table_->vkWaitForFences == nullptr || table_->vkResetFences == nullptr)
        {
            return std::unexpected(Error{ErrorCode::backendUnavailable, 0, "Present-fence waits are unavailable."});
        }
#ifdef RACHIS_TESTING
        const auto injected = test::nextPresentFenceWaitResult().exchange(0, std::memory_order_relaxed);
        if (injected != 0)
        {
            const auto result = static_cast<VkResult>(injected);
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(result), result, "vkWaitForFences"));
        }
#endif
        const auto waited = table_->vkWaitForFences(device_, 1, &image.presentFence, VK_TRUE, UINT64_MAX);
        if (waited != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(waited), waited, "vkWaitForFences"));
        }
        const auto reset = table_->vkResetFences(device_, 1, &image.presentFence);
        if (reset != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(reset), reset, "vkResetFences"));
        }
        image.presentFencePending = false;
        return {};
    }

    Status VulkanSwapchainManager::waitImagePresentId(const SwapchainRecord& record,
                                                      SwapchainImage& image,
                                                      std::uint64_t timeoutNs)
    {
        if (vkWaitForPresentKHR_ == nullptr || record.swapchain == VK_NULL_HANDLE)
        {
            image.presentId = 0;
            return {};
        }
        const auto waited = vkWaitForPresentKHR_(device_, record.swapchain, image.presentId, timeoutNs);
        if (waited == VK_TIMEOUT)
        {
            return std::unexpected(Error{
                ErrorCode::notReady,
                waited,
                "The previous present of the acquired swapchain image has not completed yet.",
            });
        }
        if (!presentWaitSettled(waited))
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(waited), waited, "vkWaitForPresentKHR"));
        }
        image.presentId = 0;
        return {};
    }

    Status VulkanSwapchainManager::retirePendingPresent(SwapchainRecord& record,
                                                        SwapchainImage& image,
                                                        std::uint64_t presentWaitTimeoutNs)
    {
        if (image.presentFencePending)
        {
            auto fenceReady = waitPresentFence(image);
            if (!fenceReady)
            {
                return fenceReady;
            }
        }
        else if (image.presentId != 0)
        {
            auto displayed = waitImagePresentId(record, image, presentWaitTimeoutNs);
            if (!displayed)
            {
                return displayed;
            }
        }
        else
        {
            return {};
        }
        if (record.inFlightPresents > 0)
        {
            --record.inFlightPresents;
        }
        return {};
    }

    Status VulkanSwapchainManager::bridgeAcquire(SwapchainImage& image)
    {
        auto bridged = queues_->submitGraphicsPresentationSync({.waitBinary = image.acquire});
        if (!bridged)
        {
            return std::unexpected(std::move(bridged.error()));
        }
        image.acquireBridge = *bridged;
        return {};
    }

    Result<VkSemaphore> VulkanSwapchainManager::takeAcquireSemaphore(SwapchainRecord& record)
    {
        if (!record.unusedAcquire.empty())
        {
            auto* const semaphore = record.unusedAcquire.back();
            record.unusedAcquire.pop_back();
            return semaphore;
        }
        const auto oldest = record.retiringAcquire.front();
        auto retired = queues_->wait(oldest.bridge, std::chrono::nanoseconds{acquireTimeoutNs});
        if (!retired)
        {
            return std::unexpected(std::move(retired.error()));
        }
        if (*retired != TimelinePointStatus::complete)
        {
            return std::unexpected(Error{
                ErrorCode::notReady,
                0,
                "No swapchain acquire semaphore has finished its previous wait yet.",
            });
        }
        record.retiringAcquire.erase(record.retiringAcquire.begin());
        return oldest.semaphore;
    }

    void VulkanSwapchainManager::releaseAcquireSemaphore(SwapchainRecord& record,
                                                         VkSemaphore semaphore,
                                                         TimelinePoint bridge) noexcept
    {
        if (bridge.isValid())
        {
            record.retiringAcquire.push_back({.semaphore = semaphore, .bridge = bridge});
            return;
        }
        record.unusedAcquire.push_back(semaphore);
    }

    void VulkanSwapchainManager::parkFailedAcquire(SwapchainRecord& record) noexcept
    {
        record.acquired = false;
        record.cancelledAcquire = true;
        record.acquireWaitSubmitted = false;
    }

    Status VulkanSwapchainManager::submitDeferredAcquireWait(SwapchainRecord& record,
                                                             std::uint64_t presentWaitTimeoutNs)
    {
        if (!record.cancelledAcquire || record.acquireWaitSubmitted)
        {
            return {};
        }
        if (queues_ == nullptr || record.acquiredIndex >= record.images.size())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A deferred swapchain acquire is missing its image.",
            });
        }
        auto& image = record.images[record.acquiredIndex];
        if (image.acquire == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A deferred swapchain acquire is missing its semaphore.",
            });
        }

        auto retired = retirePendingPresent(record, image, presentWaitTimeoutNs);
        if (!retired)
        {
            return retired;
        }
        auto bridged = bridgeAcquire(image);
        if (!bridged)
        {
            return bridged;
        }
        record.acquireWaitSubmitted = true;
        return {};
    }

    void VulkanSwapchainManager::flushDeferredAcquireWaits() noexcept
    {
        try
        {
            const std::scoped_lock lock(mutex_);
            swapchains_.forEach([this](SwapchainHandle, std::shared_ptr<SwapchainRecord>& record) {
                if (record != nullptr)
                {
                    const std::scoped_lock recordLock(record->recordMutex);
                    if (!record->destroyed && record->cancelledAcquire && !record->acquireWaitSubmitted)
                    {
                        auto& image = record->images[record->acquiredIndex];
                        queues_->drainGraphicsAcquire(image.acquire);
                        record->acquireWaitSubmitted = true;
                        if (image.presentFencePending)
                        {
                            static_cast<void>(retryTransientVulkanFailures([&] {
                                return table_->vkWaitForFences(device_, 1, &image.presentFence, VK_TRUE, UINT64_MAX);
                            }));
                            image.presentFencePending = false;
                        }
                    }
                }
            });
        }
        catch (...)
        {
            std::terminate();
        }
    }

    void VulkanSwapchainManager::waitPresentationIdle() noexcept
    {
        if (queues_ != nullptr)
        {
            static_cast<void>(queues_->waitIdle());
            return;
        }
        if (table_ != nullptr && table_->vkQueueWaitIdle != nullptr && graphicsQueue_ != VK_NULL_HANDLE)
        {
            static_cast<void>(table_->vkQueueWaitIdle(graphicsQueue_));
        }
    }

    Status VulkanSwapchainManager::waitNativeGraphicsQueueIdle()
    {
        if (table_ == nullptr || table_->vkQueueWaitIdle == nullptr || graphicsQueue_ == VK_NULL_HANDLE)
        {
            return {};
        }
        const auto idle = table_->vkQueueWaitIdle(graphicsQueue_);
        if (idle != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(idle), idle, "vkQueueWaitIdle"));
        }
        return {};
    }

    Status VulkanSwapchainManager::waitGraphicsAndPresentQueuesIdle()
    {
        if (queues_ == nullptr)
        {
            return waitNativeGraphicsQueueIdle();
        }
        auto graphicsIdle = queues_->waitIdle(QueueType::graphics);
        if (!graphicsIdle)
        {
            return graphicsIdle;
        }
        return queues_->waitPresentIdle();
    }

    Status VulkanSwapchainManager::waitLastPresentId(SwapchainRecord& record)
    {
        if (record.swapchain == VK_NULL_HANDLE || record.lastPresentId == 0)
        {
            return {};
        }
        const auto waited = vkWaitForPresentKHR_(device_, record.swapchain, record.lastPresentId, UINT64_MAX);
        if (!presentWaitSettled(waited))
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(waited), waited, "vkWaitForPresentKHR"));
        }
        record.lastPresentId = 0;
        return {};
    }

    void VulkanSwapchainManager::waitLastPresentDisplayed(SwapchainRecord& record, std::uint64_t timeoutNs) noexcept
    {
        if (record.swapchain != VK_NULL_HANDLE && record.lastPresentId != 0)
        {
            static_cast<void>(vkWaitForPresentKHR_(device_, record.swapchain, record.lastPresentId, timeoutNs));
        }
        record.lastPresentId = 0;
    }

    Status VulkanSwapchainManager::waitPresentationRetirement(SwapchainRecord& record)
    {
        if (presentCompletion_ == VulkanPresentCompletionStrategy::swapchainMaintenancePresentFence)
        {
            for (auto& image : record.images)
            {
                auto waited = waitPresentFence(image);
                if (!waited)
                {
                    return waited;
                }
            }
            if (presentWaitEnabled_ && vkWaitForPresentKHR_ != nullptr)
            {
                constexpr std::uint64_t displayedTimeoutNs = 250'000'000;
                waitLastPresentDisplayed(record, displayedTimeoutNs);
            }
            return waitGraphicsAndPresentQueuesIdle();
        }

        if (presentCompletion_ == VulkanPresentCompletionStrategy::presentWait && vkWaitForPresentKHR_ != nullptr)
        {
            auto waited = waitLastPresentId(record);
            if (!waited)
            {
                return waited;
            }
            return waitGraphicsAndPresentQueuesIdle();
        }

        if (queues_ != nullptr)
        {
            auto queuesIdle = waitGraphicsAndPresentQueuesIdle();
            if (!queuesIdle)
            {
                return queuesIdle;
            }
            return queues_->waitIdle();
        }
        if (table_ != nullptr && table_->vkDeviceWaitIdle != nullptr && device_ != VK_NULL_HANDLE)
        {
            const auto idle = table_->vkDeviceWaitIdle(device_);
            if (idle != VK_SUCCESS)
            {
                return std::unexpected(makeVulkanError(vulkanResultErrorCode(idle), idle, "vkDeviceWaitIdle"));
            }
            return {};
        }
        return waitNativeGraphicsQueueIdle();
    }

    Status VulkanSwapchainManager::destroyNativeSwapchain(SwapchainRecord& record) noexcept
    {
        destroyPresentationSync(record);
        releaseImages(record);
        if (vkDestroySwapchainKHR_ != nullptr && record.swapchain != VK_NULL_HANDLE)
        {
            vkDestroySwapchainKHR_(device_, record.swapchain, nullptr);
            record.swapchain = VK_NULL_HANDLE;
        }
        record.nextPresentId = 1;
        record.lastPresentId = 0;
        return {};
    }

    Result<std::shared_ptr<VulkanSwapchainManager::SurfaceRecord>> VulkanSwapchainManager::claimSurface(
        SurfaceHandle surface)
    {
        const std::scoped_lock lock(mutex_);
        const auto* found = surfaces_.find(surface);
        if (found == nullptr)
        {
            return std::unexpected(invalidSurface());
        }
        auto record = *found;
        if (record->swapchainClaimed)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A Vulkan surface can own only one live swapchain.",
            });
        }
        bool windowClaimed = false;
        surfaces_.forEach([&](SurfaceHandle, const std::shared_ptr<SurfaceRecord>& other) {
            if (other != nullptr && other != record && other->swapchainClaimed && sharesNativeWindow(*other, *record))
            {
                windowClaimed = true;
            }
        });
        if (windowClaimed)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The native window already has a live swapchain.",
            });
        }
        record->swapchainClaimed = true;
        return record;
    }

    Result<VkSwapchainKHR> VulkanSwapchainManager::createNativeSwapchain(const SurfaceRecord& surface,
                                                                         const VkSwapchainCreateInfoKHR& info)
    {
        if (vkCreateSwapchainKHR_ == nullptr)
        {
            return std::unexpected(Error{ErrorCode::backendUnavailable, 0, "vkCreateSwapchainKHR is unavailable."});
        }
        const std::unique_lock surfaceLock(surface.surfaceMutex);
        if (surface.surface == VK_NULL_HANDLE)
        {
            return std::unexpected(invalidSurface());
        }
        VkSwapchainKHR swapchain = VK_NULL_HANDLE;
        const auto created = vkCreateSwapchainKHR_(device_, &info, nullptr, &swapchain);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateSwapchainKHR"));
        }
        return swapchain;
    }

    Result<std::shared_ptr<VulkanSwapchainManager::SwapchainRecord>> VulkanSwapchainManager::assembleSwapchainRecord(
        const SwapchainDesc& desc,
        std::shared_ptr<SurfaceRecord> surface,
        VkSwapchainKHR swapchain,
        const VkSurfaceCapabilitiesKHR& nativeCaps)
    {
        auto record = std::make_shared<SwapchainRecord>();
        record->surface = std::move(surface);
        record->surfaceHandle = desc.surface;
        record->swapchain = swapchain;
        record->desc = desc;
        record->label = std::string(desc.label);
        record->desc.label = record->label;
        record->hasUndefinedSurfaceExtent = hasUndefinedExtent(nativeCaps);
        if (swapchain == VK_NULL_HANDLE)
        {
            record->suspended = true;
            return record;
        }
        auto images = createImages(*record);
        if (!images)
        {
            static_cast<void>(destroyNativeSwapchain(*record));
            return std::unexpected(std::move(images.error()));
        }
        auto sync = createPresentationSync(*record);
        if (!sync)
        {
            static_cast<void>(destroyNativeSwapchain(*record));
            return std::unexpected(std::move(sync.error()));
        }
        return record;
    }

    Result<SwapchainHandle> VulkanSwapchainManager::publishSwapchain(const std::shared_ptr<SwapchainRecord>& record)
    {
        const std::scoped_lock lock(mutex_);
        const auto* found = surfaces_.find(record->surfaceHandle);
        if (found == nullptr || *found != record->surface)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The swapchain surface handle became stale before creation.",
            });
        }
        return swapchains_.emplace(record);
    }

    Result<SwapchainHandle> VulkanSwapchainManager::createSwapchain(const SwapchainDesc& desc)
    {
        if (desc.presentMode != PresentMode::fifo && desc.presentMode != PresentMode::immediate)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A swapchain present mode must be a valid portable enumerator.",
            });
        }
        auto claimed = claimSurface(desc.surface);
        if (!claimed)
        {
            return std::unexpected(std::move(claimed.error()));
        }
        auto surface = std::move(*claimed);
        PendingSurfaceClaim pendingClaim(surface, mutex_);

        VkSurfaceCapabilitiesKHR nativeCaps{};
        auto capabilities = capabilitiesFor(surface->surface, &nativeCaps);
        if (!capabilities)
        {
            return std::unexpected(std::move(capabilities.error()));
        }
        auto supported = validateSwapchainDesc(desc, *capabilities);
        if (!supported)
        {
            return std::unexpected(std::move(supported.error()));
        }
        auto extent = resolveCreateExtent(nativeCaps, desc.width, desc.height);
        if (!extent)
        {
            return std::unexpected(std::move(extent.error()));
        }
        auto resolved = desc;
        resolved.width = extent->width;
        resolved.height = extent->height;

        VkSwapchainKHR native = VK_NULL_HANDLE;
        if (resolved.width != 0 && resolved.height != 0)
        {
            const auto info = swapchainCreateInfo(resolved, surface->surface, nativeCaps);
            auto swapchain = createNativeSwapchain(*surface, info);
            if (!swapchain)
            {
                return std::unexpected(std::move(swapchain.error()));
            }
            native = *swapchain;
        }
        auto assembled = assembleSwapchainRecord(resolved, std::move(surface), native, nativeCaps);
        if (!assembled)
        {
            return std::unexpected(std::move(assembled.error()));
        }
        const auto& record = *assembled;

        auto handle = publishSwapchain(record);
        if (!handle)
        {
            static_cast<void>(destroyNativeSwapchain(*record));
            return std::unexpected(std::move(handle.error()));
        }
        pendingClaim.publish();
        return *handle;
    }

    Status VulkanSwapchainManager::destroySwapchain(SwapchainHandle swapchain, bool deviceLost, bool retirePresentation)
    {
        const auto record = findSwapchainRecord(swapchain);
        if (record == nullptr)
        {
            return std::unexpected(invalidSwapchain());
        }
        std::unique_lock recordLock(record->recordMutex);
        if (record->destroyed)
        {
            return std::unexpected(invalidSwapchain());
        }
        if (!deviceLost && record->acquired)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "An acquired swapchain image must be presented or cancelled before destruction.",
            });
        }
        if (!deviceLost)
        {
            auto unviewed = rejectApplicationImageViews(*record);
            if (!unviewed)
            {
                return unviewed;
            }
        }
        if (retirePresentation)
        {
            auto retired = submitDeferredAcquireWait(*record, UINT64_MAX);
            if (retired)
            {
                retired = waitPresentationRetirement(*record);
            }
            if (!retired && !deviceLost)
            {
                return retired;
            }
        }
        static_cast<void>(destroyNativeSwapchain(*record));
        record->destroyed = true;
        recordLock.unlock();

        const std::scoped_lock lock(mutex_);
        if (record->surface != nullptr)
        {
            record->surface->swapchainClaimed = false;
        }
        swapchains_.erase(swapchain);
        return {};
    }

    Result<VkSurfaceCapabilitiesKHR> VulkanSwapchainManager::nativeSurfaceCapabilities(
        const SwapchainRecord& record) const
    {
        if (instanceTable_ == nullptr || instanceTable_->vkGetPhysicalDeviceSurfaceCapabilitiesKHR == nullptr)
        {
            return std::unexpected(
                Error{ErrorCode::backendUnavailable, 0, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR is unavailable."});
        }

        VkSurfaceCapabilitiesKHR nativeCaps{};
        const auto nativeCapsResult =
            instanceTable_->vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_, record.surface->surface, &nativeCaps);
        if (nativeCapsResult != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(nativeCapsResult),
                                                   nativeCapsResult,
                                                   "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"));
        }
        return nativeCaps;
    }

    Status VulkanSwapchainManager::replaceNativeSwapchain(SwapchainRecord& record,
                                                          const VkSurfaceCapabilitiesKHR& nativeCaps,
                                                          VkExtent2D extent)
    {
        record.requiresRefresh = true;
        const auto imageCount = std::clamp(record.desc.imageCount,
                                           nativeCaps.minImageCount,
                                           nativeCaps.maxImageCount == 0 ? UINT32_MAX : nativeCaps.maxImageCount);
        auto info = swapchainCreateInfo(record.desc, record.surface->surface, nativeCaps);
        info.minImageCount = imageCount;
        info.imageExtent = extent;
        info.oldSwapchain = record.swapchain;
        VkSwapchainKHR created = VK_NULL_HANDLE;
        auto result = VK_SUCCESS;
        {
            const std::unique_lock surfaceLock(record.surface->surfaceMutex);
            result = vkCreateSwapchainKHR_(device_, &info, nullptr, &created);
#ifdef RACHIS_TESTING
            const auto injected = test::nextCreateSwapchainResult().exchange(0, std::memory_order_relaxed);
            if (injected != 0 && result == VK_SUCCESS && vkDestroySwapchainKHR_ != nullptr)
            {
                vkDestroySwapchainKHR_(device_, created, nullptr);
                created = VK_NULL_HANDLE;
                result = static_cast<VkResult>(injected);
            }
#endif
        }
        if (result != VK_SUCCESS)
        {
            if (info.oldSwapchain != VK_NULL_HANDLE)
            {
                static_cast<void>(destroyNativeSwapchain(record));
                record.cancelledAcquire = false;
            }
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(result), result, "vkCreateSwapchainKHR"));
        }
        destroyPresentationSync(record);
        releaseImages(record);
        if (vkDestroySwapchainKHR_ != nullptr && record.swapchain != VK_NULL_HANDLE)
        {
            vkDestroySwapchainKHR_(device_, record.swapchain, nullptr);
        }
        record.swapchain = created;
        record.desc.width = extent.width;
        record.desc.height = extent.height;
        record.desc.imageCount = imageCount;
        ++record.generation;
        record.contentsDefined = false;
        record.cancelledAcquire = false;
        record.suspended = false;
        record.nextPresentId = 1;
        record.lastPresentId = 0;
        auto images = createImages(record);
        if (!images)
        {
            return images;
        }
        auto sync = createPresentationSync(record);
        record.requiresRefresh = !sync.has_value();
        return sync;
    }

    Status VulkanSwapchainManager::resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width, std::uint32_t height)
    {
        const auto recordOwner = findSwapchainRecord(swapchain);
        if (recordOwner == nullptr)
        {
            return std::unexpected(invalidSwapchain());
        }
        auto& record = *recordOwner;
        const std::scoped_lock recordLock(record.recordMutex);
        if (record.destroyed)
        {
            return std::unexpected(invalidSwapchain());
        }
        if (record.acquired)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "An acquired swapchain image must be presented or cancelled before resize.",
            });
        }
        auto unviewed = rejectApplicationImageViews(record);
        if (!unviewed)
        {
            return unviewed;
        }
        auto deferred = submitDeferredAcquireWait(record, UINT64_MAX);
        if (!deferred)
        {
            return deferred;
        }
        if (width == 0 || height == 0)
        {
            record.suspended = true;
            return {};
        }
        if (vkCreateSwapchainKHR_ == nullptr)
        {
            return std::unexpected(Error{ErrorCode::backendUnavailable, 0, "vkCreateSwapchainKHR is unavailable."});
        }
        auto retired = waitPresentationRetirement(record);
        if (!retired)
        {
            return retired;
        }

        auto nativeCaps = nativeSurfaceCapabilities(record);
        if (!nativeCaps)
        {
            return std::unexpected(std::move(nativeCaps.error()));
        }
        record.hasUndefinedSurfaceExtent = hasUndefinedExtent(*nativeCaps);
        const bool minimizedSurface = nativeCaps->currentExtent.width == 0 || nativeCaps->currentExtent.height == 0;
        if (minimizedSurface)
        {
            record.suspended = true;
            return {};
        }
        auto extent = resolveResizeExtent(*nativeCaps, width, height);
        if (!extent)
        {
            return std::unexpected(std::move(extent.error()));
        }
        return replaceNativeSwapchain(record, *nativeCaps, *extent);
    }

    Status VulkanSwapchainManager::ensureAcquirable(SwapchainRecord& record)
    {
        if (record.acquired)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A swapchain already has one acquired image.",
            });
        }
        if (record.requiresRefresh)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The swapchain must be resized before another acquire.",
            });
        }
        if (record.suspended)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A minimized or zero-extent swapchain cannot acquire an image.",
            });
        }
        if (record.surface != nullptr && surfaceExtentIsZero(record))
        {
            record.suspended = true;
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A minimized or zero-extent swapchain cannot acquire an image.",
            });
        }
        if (vkAcquireNextImageKHR_ == nullptr || queues_ == nullptr ||
            (record.unusedAcquire.empty() && record.retiringAcquire.empty()))
        {
            return std::unexpected(Error{ErrorCode::backendUnavailable, 0, "Swapchain acquire is unavailable."});
        }
        return {};
    }

    Result<AcquiredSwapchainImage> VulkanSwapchainManager::reacquireCancelledImage(SwapchainRecord& record)
    {
        auto deferred = submitDeferredAcquireWait(record, acquireTimeoutNs);
        if (!deferred)
        {
            return std::unexpected(std::move(deferred.error()));
        }
        record.cancelledAcquire = false;
        record.acquired = true;
        return publishAcquiredImage(record);
    }

    AcquiredSwapchainImage VulkanSwapchainManager::publishAcquiredImage(SwapchainRecord& record)
    {
        if (const auto& lease = record.images[record.acquiredIndex].lease; lease != nullptr)
        {
            lease->beginAcquisition();
        }
        return describeAcquiredImage(record);
    }

    Result<std::uint32_t> VulkanSwapchainManager::acquireImageWithPooledSemaphore(SwapchainRecord& record)
    {
        auto taken = takeAcquireSemaphore(record);
        if (!taken)
        {
            return std::unexpected(std::move(taken.error()));
        }
        auto* const pending = *taken;
        std::uint32_t index = 0;

#ifdef RACHIS_TESTING
        const auto injected = test::nextAcquireResult().exchange(0, std::memory_order_relaxed);
        const auto result =
            injected != 0
                ? static_cast<VkResult>(injected)
                : vkAcquireNextImageKHR_(device_, record.swapchain, acquireTimeoutNs, pending, VK_NULL_HANDLE, &index);
#else
        const auto result =
            vkAcquireNextImageKHR_(device_, record.swapchain, acquireTimeoutNs, pending, VK_NULL_HANDLE, &index);
#endif
        if (result == VK_TIMEOUT || result == VK_NOT_READY)
        {
            record.unusedAcquire.push_back(pending);
            return std::unexpected(Error{ErrorCode::notReady, result, "No swapchain image is available yet."});
        }
        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            record.unusedAcquire.push_back(pending);
            record.requiresRefresh = true;
            return std::unexpected(Error{ErrorCode::invalidState, 0, "The swapchain surface is out of date."});
        }
        if (result == VK_ERROR_DEVICE_LOST)
        {
            record.unusedAcquire.push_back(pending);
            return std::unexpected(makeVulkanError(ErrorCode::deviceLost, result, "vkAcquireNextImageKHR"));
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        {
            record.unusedAcquire.push_back(pending);
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(result), result, "vkAcquireNextImageKHR"));
        }
        if (index >= record.images.size())
        {
            auto bridged = queues_->submitGraphicsPresentationSync({.waitBinary = pending});
            releaseAcquireSemaphore(record, pending, bridged.value_or(TimelinePoint{}));
            return std::unexpected(
                Error{ErrorCode::nativeFailure, 0, "Acquire returned an image index outside the swapchain."});
        }

        auto& image = record.images[index];
        if (image.acquire != VK_NULL_HANDLE)
        {
            releaseAcquireSemaphore(record, image.acquire, image.acquireBridge);
        }
        image.acquire = pending;
        image.acquireBridge = {};
        return index;
    }

    Status VulkanSwapchainManager::settlePreviousPresent(SwapchainRecord& record, SwapchainImage& image)
    {
        if (image.presentFencePending || image.presentId != 0)
        {
            return retirePendingPresent(record, image, acquireTimeoutNs);
        }
        if (image.contentsDefined && record.inFlightPresents > 0)
        {
            --record.inFlightPresents;
        }
        return {};
    }

    Result<AcquiredSwapchainImage> VulkanSwapchainManager::acquireNextImage(SwapchainHandle swapchain)
    {
        const auto recordOwner = findSwapchainRecord(swapchain);
        if (recordOwner == nullptr)
        {
            return std::unexpected(invalidSwapchain());
        }
        auto& record = *recordOwner;
        const std::scoped_lock recordLock(record.recordMutex);
        if (record.destroyed)
        {
            return std::unexpected(invalidSwapchain());
        }
        auto acquirable = ensureAcquirable(record);
        if (!acquirable)
        {
            return std::unexpected(std::move(acquirable.error()));
        }
        if (record.cancelledAcquire)
        {
            return reacquireCancelledImage(record);
        }

        auto index = acquireImageWithPooledSemaphore(record);
        if (!index)
        {
            return std::unexpected(std::move(index.error()));
        }
        auto& image = record.images[*index];
        record.acquired = true;
        record.acquiredIndex = *index;
        record.acquireWaitSubmitted = false;
        auto previousPresentSettled = settlePreviousPresent(record, image);
        if (!previousPresentSettled)
        {
            parkFailedAcquire(record);
            return std::unexpected(std::move(previousPresentSettled.error()));
        }
        record.lastAcquireNs = steadyClockNowNs();

        auto bridged = bridgeAcquire(image);
        if (!bridged)
        {
            parkFailedAcquire(record);
            return std::unexpected(std::move(bridged.error()));
        }
        record.acquireWaitSubmitted = true;

        return publishAcquiredImage(record);
    }

    AcquiredSwapchainImage VulkanSwapchainManager::describeAcquiredImage(const SwapchainRecord& record) const
    {
        const auto& image = record.images[record.acquiredIndex];
        return AcquiredSwapchainImage{
            .texture = image.texture,
            .renderTargetView = image.view,
            .initialLayout = image.contentsDefined ? TextureLayout::present : TextureLayout::undefined,
            .initialStages = PipelineStage::none,
            .initialAccess = Access::none,
            .requiredFinalLayout = TextureLayout::present,
            .requiredFinalStages = PipelineStage::none,
            .requiredFinalAccess = Access::none,
            .contentsDefined = image.contentsDefined,
            .imageIndex = record.acquiredIndex,
            .width = record.desc.width,
            .height = record.desc.height,
            .format = record.desc.format,
            .colorSpace = record.desc.colorSpace,
            .generation = record.generation,
        };
    }

    bool VulkanSwapchainManager::hasAcquiredImage(SwapchainHandle swapchain) const noexcept
    {
        const auto record = findSwapchainRecord(swapchain);
        if (record == nullptr)
        {
            return false;
        }
        const std::scoped_lock recordLock(record->recordMutex);
        return !record->destroyed && record->acquired;
    }

    Status VulkanSwapchainManager::cancelSwapchainImage(SwapchainHandle swapchain)
    {
        const auto recordOwner = findSwapchainRecord(swapchain);
        if (recordOwner == nullptr)
        {
            return std::unexpected(invalidSwapchain());
        }
        auto& record = *recordOwner;
        const std::scoped_lock recordLock(record.recordMutex);
        if (record.destroyed)
        {
            return std::unexpected(invalidSwapchain());
        }
        if (!record.acquired)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Cancel requires one currently acquired swapchain image.",
            });
        }
        if (record.acquiredIndex < record.images.size())
        {
            if (const auto& lease = record.images[record.acquiredIndex].lease; lease != nullptr)
            {
                if (auto released = lease->cancel(); !released)
                {
                    return released;
                }
            }
        }
        record.acquired = false;
        record.cancelledAcquire = true;
        return {};
    }

    Status VulkanSwapchainManager::validatePresentable(const SwapchainRecord& record,
                                                       TimelinePoint renderComplete) const
    {
        if (!record.acquired || record.acquiredIndex >= record.images.size())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Presentation requires one currently acquired swapchain image.",
            });
        }
        if (record.requiresRefresh || record.suspended)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A suspended or out-of-date swapchain cannot present until it is resized.",
            });
        }
        if (queues_ == nullptr)
        {
            return std::unexpected(Error{ErrorCode::backendUnavailable, 0, "vkQueuePresentKHR is unavailable."});
        }
        auto submitted = queues_->validateSubmittedPoint(renderComplete);
        if (!submitted)
        {
            return submitted;
        }
        const auto& image = record.images[record.acquiredIndex];
        if (renderComplete.queueType() != QueueType::graphics || renderComplete.value() <= image.acquireBridge.value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Presentation requires a graphics-queue timeline point submitted after the image was acquired.",
            });
        }
        return {};
    }

    Result<bool> VulkanSwapchainManager::releaseImageIfSurfaceHidden(SwapchainRecord& record,
                                                                     TimelinePoint renderComplete)
    {
        const bool canReleaseImages =
            presentCompletion_ == VulkanPresentCompletionStrategy::swapchainMaintenancePresentFence &&
            vkReleaseSwapchainImages_ != nullptr;
        if (!canReleaseImages || !surfaceIsHidden(record))
        {
            return false;
        }
        auto waited = queues_->wait(renderComplete);
        if (!waited)
        {
            return std::unexpected(std::move(waited.error()));
        }
        auto released = releaseAcquiredImage(record);
        if (!released)
        {
            return false;
        }
        record.acquired = false;
        record.acquireWaitSubmitted = false;
        recordPresentCompletion(record);
        return true;
    }

    Result<VulkanSwapchainManager::SubmittedPresent> VulkanSwapchainManager::submitPresent(SwapchainRecord& record,
                                                                                           TimelinePoint renderComplete)
    {
        auto& image = record.images[record.acquiredIndex];
        auto fenceReady = waitPresentFence(image);
        if (!fenceReady)
        {
            return std::unexpected(std::move(fenceReady.error()));
        }

        auto bridged = queues_->submitPresentPresentationSync({
            .waitTimeline = renderComplete,
            .signalBinary = image.present,
        });
        if (!bridged)
        {
            return std::unexpected(std::move(bridged.error()));
        }

        PresentCompletionChain completion;
        if (presentCompletion_ == VulkanPresentCompletionStrategy::swapchainMaintenancePresentFence &&
            image.presentFence != VK_NULL_HANDLE)
        {
            completion.attachPresentFence(&image.presentFence, khrSwapchainMaintenance1_);
        }
#ifdef VK_KHR_present_id
        if (vkWaitForPresentKHR_ != nullptr && (presentCompletion_ == VulkanPresentCompletionStrategy::presentWait ||
                                                (presentWaitEnabled_ && completion.usesPresentFence())))
        {
            completion.tagPresentId(record.nextPresentId++);
        }
#endif

        const VkPresentInfoKHR info{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = completion.next(),
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &image.present,
            .swapchainCount = 1,
            .pSwapchains = &record.swapchain,
            .pImageIndices = &record.acquiredIndex,
            .pResults = {},
        };

        auto presented = queues_->presentOnPresentQueue(info);
        if (!presented)
        {
            static_cast<void>(retireBinarySemaphore(image.present));
            return std::unexpected(std::move(presented.error()));
        }
        return SubmittedPresent{
            .outcome = *presented,
            .usesPresentFence = completion.usesPresentFence(),
            .presentId = completion.presentId(),
        };
    }

    Status VulkanSwapchainManager::settleSubmittedPresent(SwapchainRecord& record, const SubmittedPresent& submitted)
    {
        auto& image = record.images[record.acquiredIndex];
        const auto result = submitted.outcome.result;
        if (submitted.outcome.waitsEnqueued)
        {
            image.presentFencePending = submitted.usesPresentFence;
            if (submitted.presentId != 0 && (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR))
            {
                record.lastPresentId = submitted.presentId;
                image.presentId = submitted.usesPresentFence ? 0 : submitted.presentId;
            }
            record.acquired = false;
            record.acquireWaitSubmitted = false;
            return {};
        }
        if (result != VK_ERROR_DEVICE_LOST || submitted.outcome.injectedDeviceLoss)
        {
            return retireBinarySemaphore(image.present);
        }
        return {};
    }

    PresentStatus VulkanSwapchainManager::markImagePresented(SwapchainRecord& record, bool usesPresentFence)
    {
        auto& image = record.images[record.acquiredIndex];
        if (usesPresentFence)
        {
            image.presentFencePending = true;
        }
        record.acquired = false;
        record.acquireWaitSubmitted = false;
        image.contentsDefined = true;
        record.contentsDefined = true;
        recordPresentCompletion(record);
        if (record.inFlightPresents < static_cast<std::uint32_t>(record.images.size()))
        {
            ++record.inFlightPresents;
        }
        if (surfaceIsHidden(record))
        {
            return PresentStatus::occluded;
        }
        return PresentStatus::presented;
    }

    Result<PresentStatus> VulkanSwapchainManager::present(SwapchainHandle swapchain,
                                                          TimelinePoint renderComplete,
                                                          bool& injectedDeviceLoss)
    {
        injectedDeviceLoss = false;
        const auto recordOwner = findSwapchainRecord(swapchain);
        if (recordOwner == nullptr)
        {
            return std::unexpected(invalidSwapchain());
        }
        auto& record = *recordOwner;
        const std::scoped_lock recordLock(record.recordMutex);
        if (record.destroyed)
        {
            return std::unexpected(invalidSwapchain());
        }
        auto presentable = validatePresentable(record, renderComplete);
        if (!presentable)
        {
            return std::unexpected(std::move(presentable.error()));
        }
        const auto lease = record.images[record.acquiredIndex].lease;
        if (lease != nullptr)
        {
            if (auto began = lease->beginPresentation(); !began)
            {
                return std::unexpected(std::move(began.error()));
            }
        }
        auto presented = presentAcquiredImage(record, renderComplete, injectedDeviceLoss);
        if (lease != nullptr && record.acquired)
        {
            lease->abortPresentation();
        }
        return presented;
    }

    Result<PresentStatus> VulkanSwapchainManager::presentAcquiredImage(SwapchainRecord& record,
                                                                       TimelinePoint renderComplete,
                                                                       bool& injectedDeviceLoss)
    {
        auto releasedWhileHidden = releaseImageIfSurfaceHidden(record, renderComplete);
        if (!releasedWhileHidden)
        {
            return std::unexpected(std::move(releasedWhileHidden.error()));
        }
        if (*releasedWhileHidden)
        {
            return PresentStatus::occluded;
        }

        auto submitted = submitPresent(record, renderComplete);
        if (!submitted)
        {
            return std::unexpected(std::move(submitted.error()));
        }
        injectedDeviceLoss = submitted->outcome.injectedDeviceLoss;
        auto settled = settleSubmittedPresent(record, *submitted);
        if (!settled)
        {
            return std::unexpected(std::move(settled.error()));
        }

        const auto result = submitted->outcome.result;
        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            record.requiresRefresh = true;
            return PresentStatus::surfaceOutOfDate;
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        {
            return std::unexpected(presentFailure(result, injectedDeviceLoss));
        }
        return markImagePresented(record, submitted->usesPresentFence);
    }

    bool VulkanSwapchainManager::surfaceIsHidden(const SwapchainRecord& record) const noexcept
    {
        if (record.surface == nullptr || record.surface->surface == VK_NULL_HANDLE)
        {
            return true;
        }
#ifdef _WIN32
        if (record.surface->kind == SurfaceKind::win32)
        {
            auto* const window = static_cast<HWND>(record.surface->window);
            return window == nullptr || IsWindow(window) == FALSE || IsWindowVisible(window) == FALSE ||
                   IsIconic(window) != FALSE;
        }
#endif
        return surfaceExtentIsZero(record);
    }

    bool VulkanSwapchainManager::surfaceExtentIsZero(const SwapchainRecord& record) const noexcept
    {
        if (record.surface == nullptr || record.surface->surface == VK_NULL_HANDLE)
        {
            return false;
        }
#ifdef _WIN32
        if (record.surface->kind == SurfaceKind::win32 && record.surface->window != nullptr)
        {
            auto* const window = static_cast<HWND>(record.surface->window);
            if (IsWindow(window) != FALSE)
            {
                if (IsIconic(window) != FALSE)
                {
                    return true;
                }
                RECT client{};
                if (GetClientRect(window, &client) != FALSE)
                {
                    return client.right <= client.left || client.bottom <= client.top;
                }
            }
        }
#endif
        if (record.surface->kind == SurfaceKind::wayland && record.hasUndefinedSurfaceExtent)
        {
            return false;
        }

        VkSurfaceCapabilitiesKHR native{};
        if (instanceTable_ == nullptr || instanceTable_->vkGetPhysicalDeviceSurfaceCapabilitiesKHR == nullptr ||
            instanceTable_->vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_, record.surface->surface, &native) !=
                VK_SUCCESS)
        {
            return false;
        }
        if (native.currentExtent.width == 0xffffffffU || native.currentExtent.height == 0xffffffffU)
        {
            return false;
        }
        return native.currentExtent.width == 0 || native.currentExtent.height == 0;
    }

    Status VulkanSwapchainManager::releaseAcquiredImage(SwapchainRecord& record)
    {
        if (vkReleaseSwapchainImages_ == nullptr || !record.acquired || record.acquiredIndex >= record.images.size())
        {
            return std::unexpected(Error{ErrorCode::backendUnavailable, 0, "Swapchain image release is unavailable."});
        }
#ifdef VK_KHR_swapchain_maintenance1
        VkReleaseSwapchainImagesInfoKHR info{
            .sType = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_KHR,
            .pNext = {},
            .swapchain = record.swapchain,
            .imageIndexCount = 1,
            .pImageIndices = &record.acquiredIndex,
        };
#elifdef VK_EXT_swapchain_maintenance1
        VkReleaseSwapchainImagesInfoEXT const info{
            .sType = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_EXT,
            .pNext = {},
            .swapchain = record.swapchain,
            .imageIndexCount = 1,
            .pImageIndices = &record.acquiredIndex,
        };
#else
        return std::unexpected(Error{ErrorCode::unsupported, 0, "Swapchain image release is unavailable."});
#endif
#if defined(VK_KHR_swapchain_maintenance1) || defined(VK_EXT_swapchain_maintenance1)
#ifdef VK_KHR_swapchain_maintenance1
        using ReleaseSwapchainImages = PFN_vkReleaseSwapchainImagesKHR;
#else
        using ReleaseSwapchainImages = PFN_vkReleaseSwapchainImagesEXT;
#endif
        const auto release = reinterpret_cast<ReleaseSwapchainImages>(vkReleaseSwapchainImages_);
        if (release == nullptr)
        {
            return std::unexpected(Error{ErrorCode::backendUnavailable, 0, "vkReleaseSwapchainImages is unavailable."});
        }
        const auto result = release(device_, &info);
        if (result != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(result), result, "vkReleaseSwapchainImages"));
        }
        return {};
#endif
    }

    void VulkanSwapchainManager::recordPresentCompletion(SwapchainRecord& record) noexcept
    {
        const auto nowNs = steadyClockNowNs();
        if (record.lastPresentSubmitNs != 0 && nowNs > record.lastPresentSubmitNs)
        {
            record.lastPresentIntervalNs = nowNs - record.lastPresentSubmitNs;
        }
        record.lastPresentSubmitNs = nowNs;
        ++record.presentsCompleted;
    }

    Result<SwapchainPresentationInfo> VulkanSwapchainManager::queryPresentation(SwapchainHandle swapchain) const
    {
        const auto recordOwner = findSwapchainRecord(swapchain);
        if (recordOwner == nullptr)
        {
            return std::unexpected(invalidSwapchain());
        }
        const auto& record = *recordOwner;
        const std::scoped_lock recordLock(record.recordMutex);
        if (record.destroyed)
        {
            return std::unexpected(invalidSwapchain());
        }
        return SwapchainPresentationInfo{
            .format = record.desc.format,
            .colorSpace = record.desc.colorSpace,
            .presentMode = record.desc.presentMode,
            .usage = record.desc.usage,
            .imageCount = static_cast<std::uint32_t>(record.images.size()),
            .framesInFlightBound = static_cast<std::uint32_t>(record.images.size()),
            .width = record.desc.width,
            .height = record.desc.height,
            .hdr = record.desc.colorSpace == SurfaceColorSpace::hdr10St2084 ||
                   record.desc.colorSpace == SurfaceColorSpace::extendedSrgbLinear,
        };
    }

    Result<SwapchainFrameTiming> VulkanSwapchainManager::queryFrameTiming(SwapchainHandle swapchain) const
    {
        const auto recordOwner = findSwapchainRecord(swapchain);
        if (recordOwner == nullptr)
        {
            return std::unexpected(invalidSwapchain());
        }
        const auto& record = *recordOwner;
        const std::scoped_lock recordLock(record.recordMutex);
        if (record.destroyed)
        {
            return std::unexpected(invalidSwapchain());
        }
        return SwapchainFrameTiming{
            .lastAcquireNs = record.lastAcquireNs,
            .lastPresentSubmitNs = record.lastPresentSubmitNs,
            .lastPresentIntervalNs = record.lastPresentIntervalNs,
            .presentsCompleted = record.presentsCompleted,
            .inFlightPresents = record.inFlightPresents,
            .hidden = surfaceIsHidden(record) || record.suspended,
        };
    }

#ifdef RACHIS_TESTING
    Result<VulkanPresentationSyncSnapshot> VulkanSwapchainManager::presentationSyncSnapshot(
        SwapchainHandle swapchain) const
    {
        const auto recordOwner = findSwapchainRecord(swapchain);
        if (recordOwner == nullptr)
        {
            return std::unexpected(invalidSwapchain());
        }
        const auto& record = *recordOwner;
        const std::scoped_lock recordLock(record.recordMutex);
        if (record.destroyed)
        {
            return std::unexpected(invalidSwapchain());
        }
        VulkanPresentationSyncSnapshot snapshot{
            .imageCount = static_cast<std::uint32_t>(record.images.size()),
            .unusedAcquireSemaphores =
                static_cast<std::uint32_t>(record.unusedAcquire.size() + record.retiringAcquire.size()),
            .acquired = record.acquired,
            .strategy = presentCompletion_,
        };
        for (const auto& image : record.images)
        {
            if (image.acquire != VK_NULL_HANDLE)
            {
                ++snapshot.assignedAcquireSemaphores;
            }
            if (image.present != VK_NULL_HANDLE)
            {
                ++snapshot.presentSemaphores;
            }
            if (image.presentFence != VK_NULL_HANDLE)
            {
                ++snapshot.presentFences;
            }
        }
        return snapshot;
    }
#endif

} // namespace rachis::vulkan
