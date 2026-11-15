#pragma once

#include <cstddef>
#include <type_traits>

namespace rachis::vulkan
{

    namespace dispatch_detail
    {

        template<typename Fn, typename Handle, typename GetProcAddr>
        [[nodiscard]] Fn resolveVulkanProc(Handle handle, GetProcAddr getProcAddr, const char* name) noexcept
        {
            if constexpr (std::is_same_v<std::remove_cvref_t<GetProcAddr>, std::nullptr_t>)
            {
                return nullptr;
            }
            else
            {
                if (handle == Handle{} || getProcAddr == nullptr)
                {
                    return nullptr;
                }

                return reinterpret_cast<Fn>(getProcAddr(handle, name));
            }
        }

        template<typename DeviceTable, typename DeviceHandle, typename GetDeviceProcAddr>
        void resolveWin32MemoryHandleProc(DeviceTable&, DeviceHandle, GetDeviceProcAddr) noexcept
        {
        }

        template<typename DeviceTable, typename DeviceHandle, typename GetDeviceProcAddr>
            requires requires(DeviceTable& table) { table.vkGetMemoryWin32HandleKHR; }
        void resolveWin32MemoryHandleProc(DeviceTable& table,
                                          DeviceHandle device,
                                          GetDeviceProcAddr getDeviceProcAddr) noexcept
        {
            table.vkGetMemoryWin32HandleKHR = resolveVulkanProc<decltype(table.vkGetMemoryWin32HandleKHR)>(
                device, getDeviceProcAddr, "vkGetMemoryWin32HandleKHR");
        }

        template<typename VmaFunctions, typename DeviceTable>
        void populateWin32MemoryHandleProc(VmaFunctions&, const DeviceTable&) noexcept
        {
        }

        template<typename VmaFunctions, typename DeviceTable>
            requires requires(VmaFunctions& functions, const DeviceTable& deviceTable) {
                functions.vkGetMemoryWin32HandleKHR = deviceTable.vkGetMemoryWin32HandleKHR;
            }
        void populateWin32MemoryHandleProc(VmaFunctions& functions, const DeviceTable& deviceTable) noexcept
        {
            functions.vkGetMemoryWin32HandleKHR = deviceTable.vkGetMemoryWin32HandleKHR;
        }

    } // namespace dispatch_detail

    template<typename Traits>
    struct VulkanInstanceTableT final
    {
        typename Traits::PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
        typename Traits::PFN_vkDestroyInstance vkDestroyInstance = nullptr;
        typename Traits::PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr = nullptr;
        typename Traits::PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceProperties2 = nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceFeatures vkGetPhysicalDeviceFeatures = nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFeatures2 = nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatProperties = nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceFormatProperties2 vkGetPhysicalDeviceFormatProperties2 = nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceImageFormatProperties vkGetPhysicalDeviceImageFormatProperties =
            nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceImageFormatProperties2 vkGetPhysicalDeviceImageFormatProperties2 =
            nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceMemoryProperties2 vkGetPhysicalDeviceMemoryProperties2 = nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceMemoryProperties2KHR vkGetPhysicalDeviceMemoryProperties2KHR = nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties =
            nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceQueueFamilyProperties2 vkGetPhysicalDeviceQueueFamilyProperties2 =
            nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceExternalBufferProperties vkGetPhysicalDeviceExternalBufferProperties =
            nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceExternalFenceProperties vkGetPhysicalDeviceExternalFenceProperties =
            nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceExternalSemaphoreProperties
            vkGetPhysicalDeviceExternalSemaphoreProperties = nullptr;
        typename Traits::PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties = nullptr;
        typename Traits::PFN_vkCreateDevice vkCreateDevice = nullptr;
        typename Traits::PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT = nullptr;
        typename Traits::PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT = nullptr;
        typename Traits::PFN_vkSubmitDebugUtilsMessageEXT vkSubmitDebugUtilsMessageEXT = nullptr;
        typename Traits::PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXT = nullptr;
        typename Traits::PFN_vkSetDebugUtilsObjectTagEXT vkSetDebugUtilsObjectTagEXT = nullptr;
        typename Traits::PFN_vkQueueBeginDebugUtilsLabelEXT vkQueueBeginDebugUtilsLabelEXT = nullptr;
        typename Traits::PFN_vkQueueEndDebugUtilsLabelEXT vkQueueEndDebugUtilsLabelEXT = nullptr;
        typename Traits::PFN_vkQueueInsertDebugUtilsLabelEXT vkQueueInsertDebugUtilsLabelEXT = nullptr;
        typename Traits::PFN_vkCmdBeginDebugUtilsLabelEXT vkCmdBeginDebugUtilsLabelEXT = nullptr;
        typename Traits::PFN_vkCmdEndDebugUtilsLabelEXT vkCmdEndDebugUtilsLabelEXT = nullptr;
        typename Traits::PFN_vkCmdInsertDebugUtilsLabelEXT vkCmdInsertDebugUtilsLabelEXT = nullptr;
        typename Traits::PFN_vkDestroySurfaceKHR vkDestroySurfaceKHR = nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceSurfaceSupportKHR vkGetPhysicalDeviceSurfaceSupportKHR = nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR vkGetPhysicalDeviceSurfaceCapabilitiesKHR =
            nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR vkGetPhysicalDeviceSurfaceCapabilities2KHR =
            nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceSurfaceFormatsKHR vkGetPhysicalDeviceSurfaceFormatsKHR = nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceSurfaceFormats2KHR vkGetPhysicalDeviceSurfaceFormats2KHR = nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceSurfacePresentModesKHR vkGetPhysicalDeviceSurfacePresentModesKHR =
            nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR vkGetPhysicalDeviceFragmentShadingRatesKHR =
            nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT
            vkGetPhysicalDeviceCalibrateableTimeDomainsEXT = nullptr;
        typename Traits::PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR
            vkGetPhysicalDeviceCalibrateableTimeDomainsKHR = nullptr;
    };

    template<typename Traits>
    void loadVulkanInstanceTable(VulkanInstanceTableT<Traits>& table,
                                 typename Traits::InstanceHandle instance,
                                 typename Traits::PFN_vkGetInstanceProcAddr getInstanceProcAddr) noexcept
    {
        table = {};
        if (instance == typename Traits::InstanceHandle{} || getInstanceProcAddr == nullptr)
        {
            return;
        }

        table.vkGetInstanceProcAddr = getInstanceProcAddr;
        table.vkDestroyInstance = dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkDestroyInstance>(
            instance, getInstanceProcAddr, "vkDestroyInstance");
        table.vkGetDeviceProcAddr = dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetDeviceProcAddr>(
            instance, getInstanceProcAddr, "vkGetDeviceProcAddr");
        table.vkEnumeratePhysicalDevices =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkEnumeratePhysicalDevices>(
                instance, getInstanceProcAddr, "vkEnumeratePhysicalDevices");
        table.vkGetPhysicalDeviceProperties =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceProperties>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceProperties");
        table.vkGetPhysicalDeviceProperties2 =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceProperties2>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceProperties2");
        table.vkGetPhysicalDeviceFeatures =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceFeatures>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceFeatures");
        table.vkGetPhysicalDeviceFeatures2 =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceFeatures2>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceFeatures2");
        table.vkGetPhysicalDeviceFormatProperties =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceFormatProperties>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceFormatProperties");
        table.vkGetPhysicalDeviceFormatProperties2 =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceFormatProperties2>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceFormatProperties2");
        table.vkGetPhysicalDeviceImageFormatProperties =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceImageFormatProperties>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceImageFormatProperties");
        table.vkGetPhysicalDeviceImageFormatProperties2 =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceImageFormatProperties2>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceImageFormatProperties2");
        table.vkGetPhysicalDeviceMemoryProperties =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceMemoryProperties>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceMemoryProperties");
        table.vkGetPhysicalDeviceMemoryProperties2 =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceMemoryProperties2>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceMemoryProperties2");
        table.vkGetPhysicalDeviceMemoryProperties2KHR =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceMemoryProperties2KHR>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceMemoryProperties2KHR");
        table.vkGetPhysicalDeviceQueueFamilyProperties =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceQueueFamilyProperties");
        table.vkGetPhysicalDeviceQueueFamilyProperties2 =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceQueueFamilyProperties2>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceQueueFamilyProperties2");
        table.vkGetPhysicalDeviceExternalBufferProperties =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceExternalBufferProperties>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceExternalBufferProperties");
        table.vkGetPhysicalDeviceExternalFenceProperties =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceExternalFenceProperties>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceExternalFenceProperties");
        table.vkGetPhysicalDeviceExternalSemaphoreProperties =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceExternalSemaphoreProperties>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceExternalSemaphoreProperties");
        table.vkEnumerateDeviceExtensionProperties =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkEnumerateDeviceExtensionProperties>(
                instance, getInstanceProcAddr, "vkEnumerateDeviceExtensionProperties");
        table.vkCreateDevice = dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkCreateDevice>(
            instance, getInstanceProcAddr, "vkCreateDevice");
        table.vkCreateDebugUtilsMessengerEXT =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkCreateDebugUtilsMessengerEXT>(
                instance, getInstanceProcAddr, "vkCreateDebugUtilsMessengerEXT");
        table.vkDestroyDebugUtilsMessengerEXT =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkDestroyDebugUtilsMessengerEXT>(
                instance, getInstanceProcAddr, "vkDestroyDebugUtilsMessengerEXT");
        table.vkSubmitDebugUtilsMessageEXT =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkSubmitDebugUtilsMessageEXT>(
                instance, getInstanceProcAddr, "vkSubmitDebugUtilsMessageEXT");
        table.vkSetDebugUtilsObjectNameEXT =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkSetDebugUtilsObjectNameEXT>(
                instance, getInstanceProcAddr, "vkSetDebugUtilsObjectNameEXT");
        table.vkSetDebugUtilsObjectTagEXT =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkSetDebugUtilsObjectTagEXT>(
                instance, getInstanceProcAddr, "vkSetDebugUtilsObjectTagEXT");
        table.vkQueueBeginDebugUtilsLabelEXT =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkQueueBeginDebugUtilsLabelEXT>(
                instance, getInstanceProcAddr, "vkQueueBeginDebugUtilsLabelEXT");
        table.vkQueueEndDebugUtilsLabelEXT =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkQueueEndDebugUtilsLabelEXT>(
                instance, getInstanceProcAddr, "vkQueueEndDebugUtilsLabelEXT");
        table.vkQueueInsertDebugUtilsLabelEXT =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkQueueInsertDebugUtilsLabelEXT>(
                instance, getInstanceProcAddr, "vkQueueInsertDebugUtilsLabelEXT");
        table.vkCmdBeginDebugUtilsLabelEXT =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkCmdBeginDebugUtilsLabelEXT>(
                instance, getInstanceProcAddr, "vkCmdBeginDebugUtilsLabelEXT");
        table.vkCmdEndDebugUtilsLabelEXT =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkCmdEndDebugUtilsLabelEXT>(
                instance, getInstanceProcAddr, "vkCmdEndDebugUtilsLabelEXT");
        table.vkCmdInsertDebugUtilsLabelEXT =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkCmdInsertDebugUtilsLabelEXT>(
                instance, getInstanceProcAddr, "vkCmdInsertDebugUtilsLabelEXT");
        table.vkDestroySurfaceKHR = dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkDestroySurfaceKHR>(
            instance, getInstanceProcAddr, "vkDestroySurfaceKHR");
        table.vkGetPhysicalDeviceSurfaceSupportKHR =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceSurfaceSupportKHR");
        table.vkGetPhysicalDeviceSurfaceCapabilitiesKHR =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
        table.vkGetPhysicalDeviceSurfaceCapabilities2KHR =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceSurfaceCapabilities2KHR");
        table.vkGetPhysicalDeviceSurfaceFormatsKHR =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceSurfaceFormatsKHR");
        table.vkGetPhysicalDeviceSurfaceFormats2KHR =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceSurfaceFormats2KHR>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceSurfaceFormats2KHR");
        table.vkGetPhysicalDeviceSurfacePresentModesKHR =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceSurfacePresentModesKHR>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceSurfacePresentModesKHR");
        table.vkGetPhysicalDeviceFragmentShadingRatesKHR =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceFragmentShadingRatesKHR");
        table.vkGetPhysicalDeviceCalibrateableTimeDomainsEXT =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT");
        table.vkGetPhysicalDeviceCalibrateableTimeDomainsKHR =
            dispatch_detail::resolveVulkanProc<typename Traits::PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR>(
                instance, getInstanceProcAddr, "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR");
    }

    template<typename DeviceTable, typename DeviceHandle, typename GetDeviceProcAddr>
    void loadVulkanBasicDeviceTable(DeviceTable& table,
                                    DeviceHandle device,
                                    GetDeviceProcAddr getDeviceProcAddr) noexcept
    {
        table = {};
        if (device == DeviceHandle{} || getDeviceProcAddr == nullptr)
        {
            return;
        }

        table.vkDestroyDevice = dispatch_detail::resolveVulkanProc<decltype(table.vkDestroyDevice)>(
            device, getDeviceProcAddr, "vkDestroyDevice");
        table.vkDeviceWaitIdle = dispatch_detail::resolveVulkanProc<decltype(table.vkDeviceWaitIdle)>(
            device, getDeviceProcAddr, "vkDeviceWaitIdle");
        table.vkGetDeviceQueue = dispatch_detail::resolveVulkanProc<decltype(table.vkGetDeviceQueue)>(
            device, getDeviceProcAddr, "vkGetDeviceQueue");
        table.vkQueueSubmit = dispatch_detail::resolveVulkanProc<decltype(table.vkQueueSubmit)>(
            device, getDeviceProcAddr, "vkQueueSubmit");
        table.vkQueueWaitIdle = dispatch_detail::resolveVulkanProc<decltype(table.vkQueueWaitIdle)>(
            device, getDeviceProcAddr, "vkQueueWaitIdle");
        table.vkAllocateMemory = dispatch_detail::resolveVulkanProc<decltype(table.vkAllocateMemory)>(
            device, getDeviceProcAddr, "vkAllocateMemory");
        table.vkFreeMemory =
            dispatch_detail::resolveVulkanProc<decltype(table.vkFreeMemory)>(device, getDeviceProcAddr, "vkFreeMemory");
        table.vkMapMemory =
            dispatch_detail::resolveVulkanProc<decltype(table.vkMapMemory)>(device, getDeviceProcAddr, "vkMapMemory");
        table.vkUnmapMemory = dispatch_detail::resolveVulkanProc<decltype(table.vkUnmapMemory)>(
            device, getDeviceProcAddr, "vkUnmapMemory");
        table.vkFlushMappedMemoryRanges = dispatch_detail::resolveVulkanProc<decltype(table.vkFlushMappedMemoryRanges)>(
            device, getDeviceProcAddr, "vkFlushMappedMemoryRanges");
        table.vkInvalidateMappedMemoryRanges =
            dispatch_detail::resolveVulkanProc<decltype(table.vkInvalidateMappedMemoryRanges)>(
                device, getDeviceProcAddr, "vkInvalidateMappedMemoryRanges");
        table.vkBindBufferMemory = dispatch_detail::resolveVulkanProc<decltype(table.vkBindBufferMemory)>(
            device, getDeviceProcAddr, "vkBindBufferMemory");
        table.vkBindImageMemory = dispatch_detail::resolveVulkanProc<decltype(table.vkBindImageMemory)>(
            device, getDeviceProcAddr, "vkBindImageMemory");
        table.vkGetBufferMemoryRequirements =
            dispatch_detail::resolveVulkanProc<decltype(table.vkGetBufferMemoryRequirements)>(
                device, getDeviceProcAddr, "vkGetBufferMemoryRequirements");
        table.vkGetImageMemoryRequirements =
            dispatch_detail::resolveVulkanProc<decltype(table.vkGetImageMemoryRequirements)>(
                device, getDeviceProcAddr, "vkGetImageMemoryRequirements");
        table.vkCreateBuffer = dispatch_detail::resolveVulkanProc<decltype(table.vkCreateBuffer)>(
            device, getDeviceProcAddr, "vkCreateBuffer");
        table.vkDestroyBuffer = dispatch_detail::resolveVulkanProc<decltype(table.vkDestroyBuffer)>(
            device, getDeviceProcAddr, "vkDestroyBuffer");
        table.vkCreateImage = dispatch_detail::resolveVulkanProc<decltype(table.vkCreateImage)>(
            device, getDeviceProcAddr, "vkCreateImage");
        table.vkDestroyImage = dispatch_detail::resolveVulkanProc<decltype(table.vkDestroyImage)>(
            device, getDeviceProcAddr, "vkDestroyImage");
        table.vkCreateCommandPool = dispatch_detail::resolveVulkanProc<decltype(table.vkCreateCommandPool)>(
            device, getDeviceProcAddr, "vkCreateCommandPool");
        table.vkDestroyCommandPool = dispatch_detail::resolveVulkanProc<decltype(table.vkDestroyCommandPool)>(
            device, getDeviceProcAddr, "vkDestroyCommandPool");
        table.vkAllocateCommandBuffers = dispatch_detail::resolveVulkanProc<decltype(table.vkAllocateCommandBuffers)>(
            device, getDeviceProcAddr, "vkAllocateCommandBuffers");
        table.vkBeginCommandBuffer = dispatch_detail::resolveVulkanProc<decltype(table.vkBeginCommandBuffer)>(
            device, getDeviceProcAddr, "vkBeginCommandBuffer");
        table.vkEndCommandBuffer = dispatch_detail::resolveVulkanProc<decltype(table.vkEndCommandBuffer)>(
            device, getDeviceProcAddr, "vkEndCommandBuffer");
        table.vkCmdCopyBuffer = dispatch_detail::resolveVulkanProc<decltype(table.vkCmdCopyBuffer)>(
            device, getDeviceProcAddr, "vkCmdCopyBuffer");
        table.vkBindBufferMemory2 = dispatch_detail::resolveVulkanProc<decltype(table.vkBindBufferMemory2)>(
            device, getDeviceProcAddr, "vkBindBufferMemory2");
        table.vkBindBufferMemory2KHR = dispatch_detail::resolveVulkanProc<decltype(table.vkBindBufferMemory2KHR)>(
            device, getDeviceProcAddr, "vkBindBufferMemory2KHR");
        table.vkBindImageMemory2 = dispatch_detail::resolveVulkanProc<decltype(table.vkBindImageMemory2)>(
            device, getDeviceProcAddr, "vkBindImageMemory2");
        table.vkBindImageMemory2KHR = dispatch_detail::resolveVulkanProc<decltype(table.vkBindImageMemory2KHR)>(
            device, getDeviceProcAddr, "vkBindImageMemory2KHR");
        table.vkGetBufferMemoryRequirements2 =
            dispatch_detail::resolveVulkanProc<decltype(table.vkGetBufferMemoryRequirements2)>(
                device, getDeviceProcAddr, "vkGetBufferMemoryRequirements2");
        table.vkGetBufferMemoryRequirements2KHR =
            dispatch_detail::resolveVulkanProc<decltype(table.vkGetBufferMemoryRequirements2KHR)>(
                device, getDeviceProcAddr, "vkGetBufferMemoryRequirements2KHR");
        table.vkGetImageMemoryRequirements2 =
            dispatch_detail::resolveVulkanProc<decltype(table.vkGetImageMemoryRequirements2)>(
                device, getDeviceProcAddr, "vkGetImageMemoryRequirements2");
        table.vkGetImageMemoryRequirements2KHR =
            dispatch_detail::resolveVulkanProc<decltype(table.vkGetImageMemoryRequirements2KHR)>(
                device, getDeviceProcAddr, "vkGetImageMemoryRequirements2KHR");
        table.vkGetDeviceBufferMemoryRequirements =
            dispatch_detail::resolveVulkanProc<decltype(table.vkGetDeviceBufferMemoryRequirements)>(
                device, getDeviceProcAddr, "vkGetDeviceBufferMemoryRequirements");
        table.vkGetDeviceBufferMemoryRequirementsKHR =
            dispatch_detail::resolveVulkanProc<decltype(table.vkGetDeviceBufferMemoryRequirementsKHR)>(
                device, getDeviceProcAddr, "vkGetDeviceBufferMemoryRequirementsKHR");
        table.vkGetDeviceImageMemoryRequirements =
            dispatch_detail::resolveVulkanProc<decltype(table.vkGetDeviceImageMemoryRequirements)>(
                device, getDeviceProcAddr, "vkGetDeviceImageMemoryRequirements");
        table.vkGetDeviceImageMemoryRequirementsKHR =
            dispatch_detail::resolveVulkanProc<decltype(table.vkGetDeviceImageMemoryRequirementsKHR)>(
                device, getDeviceProcAddr, "vkGetDeviceImageMemoryRequirementsKHR");
        dispatch_detail::resolveWin32MemoryHandleProc(table, device, getDeviceProcAddr);
    }

    template<typename VmaFunctions, typename InstanceTable, typename DeviceTable>
    void populateVulkanAllocatorFunctions(VmaFunctions& functions,
                                          const InstanceTable& instanceTable,
                                          const DeviceTable& deviceTable) noexcept
    {
        functions = {};
        functions.vkGetInstanceProcAddr = instanceTable.vkGetInstanceProcAddr;
        functions.vkGetDeviceProcAddr = instanceTable.vkGetDeviceProcAddr;
        functions.vkGetPhysicalDeviceProperties = instanceTable.vkGetPhysicalDeviceProperties;
        functions.vkGetPhysicalDeviceMemoryProperties = instanceTable.vkGetPhysicalDeviceMemoryProperties;
        functions.vkAllocateMemory = deviceTable.vkAllocateMemory;
        functions.vkFreeMemory = deviceTable.vkFreeMemory;
        functions.vkMapMemory = deviceTable.vkMapMemory;
        functions.vkUnmapMemory = deviceTable.vkUnmapMemory;
        functions.vkFlushMappedMemoryRanges = deviceTable.vkFlushMappedMemoryRanges;
        functions.vkInvalidateMappedMemoryRanges = deviceTable.vkInvalidateMappedMemoryRanges;
        functions.vkBindBufferMemory = deviceTable.vkBindBufferMemory;
        functions.vkBindImageMemory = deviceTable.vkBindImageMemory;
        functions.vkGetBufferMemoryRequirements = deviceTable.vkGetBufferMemoryRequirements;
        functions.vkGetImageMemoryRequirements = deviceTable.vkGetImageMemoryRequirements;
        functions.vkCreateBuffer = deviceTable.vkCreateBuffer;
        functions.vkDestroyBuffer = deviceTable.vkDestroyBuffer;
        functions.vkCreateImage = deviceTable.vkCreateImage;
        functions.vkDestroyImage = deviceTable.vkDestroyImage;
        functions.vkCmdCopyBuffer = deviceTable.vkCmdCopyBuffer;
        functions.vkGetBufferMemoryRequirements2KHR = deviceTable.vkGetBufferMemoryRequirements2 != nullptr
                                                          ? deviceTable.vkGetBufferMemoryRequirements2
                                                          : deviceTable.vkGetBufferMemoryRequirements2KHR;
        functions.vkGetImageMemoryRequirements2KHR = deviceTable.vkGetImageMemoryRequirements2 != nullptr
                                                         ? deviceTable.vkGetImageMemoryRequirements2
                                                         : deviceTable.vkGetImageMemoryRequirements2KHR;
        functions.vkBindBufferMemory2KHR = deviceTable.vkBindBufferMemory2 != nullptr
                                               ? deviceTable.vkBindBufferMemory2
                                               : deviceTable.vkBindBufferMemory2KHR;
        functions.vkBindImageMemory2KHR = deviceTable.vkBindImageMemory2 != nullptr ? deviceTable.vkBindImageMemory2
                                                                                    : deviceTable.vkBindImageMemory2KHR;
        functions.vkGetPhysicalDeviceMemoryProperties2KHR =
            instanceTable.vkGetPhysicalDeviceMemoryProperties2 != nullptr
                ? instanceTable.vkGetPhysicalDeviceMemoryProperties2
                : instanceTable.vkGetPhysicalDeviceMemoryProperties2KHR;
        functions.vkGetDeviceBufferMemoryRequirements = deviceTable.vkGetDeviceBufferMemoryRequirements != nullptr
                                                            ? deviceTable.vkGetDeviceBufferMemoryRequirements
                                                            : deviceTable.vkGetDeviceBufferMemoryRequirementsKHR;
        functions.vkGetDeviceImageMemoryRequirements = deviceTable.vkGetDeviceImageMemoryRequirements != nullptr
                                                           ? deviceTable.vkGetDeviceImageMemoryRequirements
                                                           : deviceTable.vkGetDeviceImageMemoryRequirementsKHR;
        dispatch_detail::populateWin32MemoryHandleProc(functions, deviceTable);
    }

} // namespace rachis::vulkan
