#pragma once

#include "Internal/VulkanDispatch.h"
#include "Rachis/Diagnostics.h"
#include "Rachis/Error.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <volk.h>

namespace rachis::vulkan
{

    struct VulkanDispatchTraits final
    {
        using InstanceHandle = VkInstance;
        using PFN_vkGetInstanceProcAddr = ::PFN_vkGetInstanceProcAddr;
        using PFN_vkDestroyInstance = ::PFN_vkDestroyInstance;
        using PFN_vkGetDeviceProcAddr = ::PFN_vkGetDeviceProcAddr;
        using PFN_vkEnumeratePhysicalDevices = ::PFN_vkEnumeratePhysicalDevices;
        using PFN_vkGetPhysicalDeviceProperties = ::PFN_vkGetPhysicalDeviceProperties;
        using PFN_vkGetPhysicalDeviceProperties2 = ::PFN_vkGetPhysicalDeviceProperties2;
        using PFN_vkGetPhysicalDeviceFeatures = ::PFN_vkGetPhysicalDeviceFeatures;
        using PFN_vkGetPhysicalDeviceFeatures2 = ::PFN_vkGetPhysicalDeviceFeatures2;
        using PFN_vkGetPhysicalDeviceFormatProperties = ::PFN_vkGetPhysicalDeviceFormatProperties;
        using PFN_vkGetPhysicalDeviceFormatProperties2 = ::PFN_vkGetPhysicalDeviceFormatProperties2;
        using PFN_vkGetPhysicalDeviceImageFormatProperties = ::PFN_vkGetPhysicalDeviceImageFormatProperties;
        using PFN_vkGetPhysicalDeviceImageFormatProperties2 = ::PFN_vkGetPhysicalDeviceImageFormatProperties2;
        using PFN_vkGetPhysicalDeviceMemoryProperties = ::PFN_vkGetPhysicalDeviceMemoryProperties;
        using PFN_vkGetPhysicalDeviceMemoryProperties2 = ::PFN_vkGetPhysicalDeviceMemoryProperties2;
        using PFN_vkGetPhysicalDeviceMemoryProperties2KHR = ::PFN_vkGetPhysicalDeviceMemoryProperties2KHR;
        using PFN_vkGetPhysicalDeviceQueueFamilyProperties = ::PFN_vkGetPhysicalDeviceQueueFamilyProperties;
        using PFN_vkGetPhysicalDeviceQueueFamilyProperties2 = ::PFN_vkGetPhysicalDeviceQueueFamilyProperties2;
        using PFN_vkGetPhysicalDeviceExternalBufferProperties = ::PFN_vkGetPhysicalDeviceExternalBufferProperties;
        using PFN_vkGetPhysicalDeviceExternalFenceProperties = ::PFN_vkGetPhysicalDeviceExternalFenceProperties;
        using PFN_vkGetPhysicalDeviceExternalSemaphoreProperties = ::PFN_vkGetPhysicalDeviceExternalSemaphoreProperties;
        using PFN_vkEnumerateDeviceExtensionProperties = ::PFN_vkEnumerateDeviceExtensionProperties;
        using PFN_vkCreateDevice = ::PFN_vkCreateDevice;
        using PFN_vkCreateDebugUtilsMessengerEXT = ::PFN_vkCreateDebugUtilsMessengerEXT;
        using PFN_vkDestroyDebugUtilsMessengerEXT = ::PFN_vkDestroyDebugUtilsMessengerEXT;
        using PFN_vkSubmitDebugUtilsMessageEXT = ::PFN_vkSubmitDebugUtilsMessageEXT;
        using PFN_vkSetDebugUtilsObjectNameEXT = ::PFN_vkSetDebugUtilsObjectNameEXT;
        using PFN_vkSetDebugUtilsObjectTagEXT = ::PFN_vkSetDebugUtilsObjectTagEXT;
        using PFN_vkQueueBeginDebugUtilsLabelEXT = ::PFN_vkQueueBeginDebugUtilsLabelEXT;
        using PFN_vkQueueEndDebugUtilsLabelEXT = ::PFN_vkQueueEndDebugUtilsLabelEXT;
        using PFN_vkQueueInsertDebugUtilsLabelEXT = ::PFN_vkQueueInsertDebugUtilsLabelEXT;
        using PFN_vkCmdBeginDebugUtilsLabelEXT = ::PFN_vkCmdBeginDebugUtilsLabelEXT;
        using PFN_vkCmdEndDebugUtilsLabelEXT = ::PFN_vkCmdEndDebugUtilsLabelEXT;
        using PFN_vkCmdInsertDebugUtilsLabelEXT = ::PFN_vkCmdInsertDebugUtilsLabelEXT;
        using PFN_vkDestroySurfaceKHR = ::PFN_vkDestroySurfaceKHR;
        using PFN_vkGetPhysicalDeviceSurfaceSupportKHR = ::PFN_vkGetPhysicalDeviceSurfaceSupportKHR;
        using PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR = ::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR;
        using PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR = ::PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR;
        using PFN_vkGetPhysicalDeviceSurfaceFormatsKHR = ::PFN_vkGetPhysicalDeviceSurfaceFormatsKHR;
        using PFN_vkGetPhysicalDeviceSurfaceFormats2KHR = ::PFN_vkGetPhysicalDeviceSurfaceFormats2KHR;
        using PFN_vkGetPhysicalDeviceSurfacePresentModesKHR = ::PFN_vkGetPhysicalDeviceSurfacePresentModesKHR;
        using PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR = ::PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR;
        using PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT = ::PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT;
        using PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR = ::PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR;
    };

    using VulkanInstanceTable = VulkanInstanceTableT<VulkanDispatchTraits>;

    struct VulkanDiagnosticSink final
    {
        std::atomic<bool> alive{true};
        std::mutex mutex;
        DiagnosticCallback callback = nullptr;
        void* userData = nullptr;
        std::atomic<std::uint64_t> information{0};
        std::atomic<std::uint64_t> warnings{0};
        std::atomic<std::uint64_t> errors{0};
        std::atomic<std::uint64_t> corruptions{0};

        void publish(DiagnosticSeverity severity, std::uint32_t nativeId, std::string_view description) noexcept;
        [[nodiscard]] DiagnosticCounts counts() const noexcept;
    };

    [[nodiscard]] const char* vulkanResultName(VkResult result) noexcept;
    [[nodiscard]] Error makeVulkanError(ErrorCode code, VkResult result, std::string_view operation);
    [[nodiscard]] ErrorCode vulkanResultErrorCode(VkResult result) noexcept;

    VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugUtilsCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                            VkDebugUtilsMessageTypeFlagsEXT types,
                                                            const VkDebugUtilsMessengerCallbackDataEXT* data,
                                                            void* userData);

    void setVulkanObjectName(const VulkanInstanceTable* instanceTable,
                             VkDevice device,
                             VkObjectType type,
                             std::uint64_t handle,
                             const char* name) noexcept;

    [[nodiscard]] bool captureVulkanDeviceFaultInfo(VkDevice device,
                                                    const VolkDeviceTable& table,
                                                    DeviceFaultReport& report,
                                                    std::string& faultSummary) noexcept;

#ifdef RACHIS_TESTING
    void bindVulkanValidationTestDevice(VkDevice device, const VolkDeviceTable* table);
    void unbindVulkanValidationTestDevice(VkDevice device) noexcept;
    [[nodiscard]] Status emitIntentionalValidationFailure();
#endif

} // namespace rachis::vulkan
