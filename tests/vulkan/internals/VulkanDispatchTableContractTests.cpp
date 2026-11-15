#include "Internal/VulkanDispatch.h"

#if RACHIS_ENABLE_VULKAN
#ifndef VMA_STATIC_VULKAN_FUNCTIONS
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#endif
#ifndef VMA_DYNAMIC_VULKAN_FUNCTIONS
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#endif

#include "Internal/VulkanDiagnostics.h"
#include "Internal/VulkanRuntime.h"

#include <vk_mem_alloc.h>
#include <volk.h>
#endif

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string_view>

namespace
{

    struct FakeInstanceObject final
    {
        int tag = 0;
    };

    struct FakeDeviceObject final
    {
        int tag = 0;
    };

    struct FakePhysicalDeviceObject final
    {
        int tag = 0;
    };

    struct FakeCommandBufferObject final
    {
        int tag = 0;
    };

    using FakeVoidFn = void (*)();

    struct FakeTraits final
    {
        using InstanceHandle = FakeInstanceObject*;
        using DeviceHandle = FakeDeviceObject*;
        using VoidFn = FakeVoidFn;

        using PFN_vkGetInstanceProcAddr = VoidFn (*)(InstanceHandle, const char*);
        using PFN_vkDestroyInstance = void (*)(InstanceHandle, const void*);
        using PFN_vkGetDeviceProcAddr = VoidFn (*)(DeviceHandle, const char*);
        using PFN_vkEnumeratePhysicalDevices = int (*)(InstanceHandle, std::uint32_t*, FakePhysicalDeviceObject**);
        using PFN_vkGetPhysicalDeviceProperties = void (*)(FakePhysicalDeviceObject*, std::uint32_t*);
        using PFN_vkGetPhysicalDeviceProperties2 = void (*)(FakePhysicalDeviceObject*, std::uint32_t*);
        using PFN_vkGetPhysicalDeviceFeatures = void (*)(FakePhysicalDeviceObject*, void*);
        using PFN_vkGetPhysicalDeviceFeatures2 = void (*)(FakePhysicalDeviceObject*, void*);
        using PFN_vkGetPhysicalDeviceFormatProperties = void (*)(FakePhysicalDeviceObject*,
                                                                 std::uint32_t,
                                                                 std::uint32_t*);
        using PFN_vkGetPhysicalDeviceFormatProperties2 = void (*)(FakePhysicalDeviceObject*, std::uint32_t, void*);
        using PFN_vkGetPhysicalDeviceImageFormatProperties = int (*)(FakePhysicalDeviceObject*, std::uint32_t, void*);
        using PFN_vkGetPhysicalDeviceImageFormatProperties2 = int (*)(FakePhysicalDeviceObject*, const void*, void*);
        using PFN_vkGetPhysicalDeviceMemoryProperties = void (*)(FakePhysicalDeviceObject*, std::uint32_t*);
        using PFN_vkGetPhysicalDeviceMemoryProperties2 = void (*)(FakePhysicalDeviceObject*, std::uint32_t*);
        using PFN_vkGetPhysicalDeviceMemoryProperties2KHR = void (*)(FakePhysicalDeviceObject*, std::uint32_t*);
        using PFN_vkGetPhysicalDeviceQueueFamilyProperties = void (*)(FakePhysicalDeviceObject*, std::uint32_t*, void*);
        using PFN_vkGetPhysicalDeviceQueueFamilyProperties2 = void (*)(FakePhysicalDeviceObject*,
                                                                       std::uint32_t*,
                                                                       void*);
        using PFN_vkGetPhysicalDeviceExternalBufferProperties = void (*)(FakePhysicalDeviceObject*, const void*, void*);
        using PFN_vkGetPhysicalDeviceExternalFenceProperties = void (*)(FakePhysicalDeviceObject*, const void*, void*);
        using PFN_vkGetPhysicalDeviceExternalSemaphoreProperties = void (*)(FakePhysicalDeviceObject*,
                                                                            const void*,
                                                                            void*);
        using PFN_vkEnumerateDeviceExtensionProperties = int (*)(FakePhysicalDeviceObject*,
                                                                 const char*,
                                                                 std::uint32_t*,
                                                                 void*);
        using PFN_vkCreateDevice = int (*)(FakePhysicalDeviceObject*, const void*, const void*, DeviceHandle*);
        using PFN_vkCreateDebugUtilsMessengerEXT = int (*)(InstanceHandle, const void*, const void*, std::uint64_t*);
        using PFN_vkDestroyDebugUtilsMessengerEXT = void (*)(InstanceHandle, std::uint64_t, const void*);
        using PFN_vkSubmitDebugUtilsMessageEXT = void (*)(InstanceHandle, std::uint32_t, std::uint32_t, const void*);
        using PFN_vkSetDebugUtilsObjectNameEXT = int (*)(DeviceHandle, const void*);
        using PFN_vkSetDebugUtilsObjectTagEXT = int (*)(DeviceHandle, const void*);
        using PFN_vkQueueBeginDebugUtilsLabelEXT = void (*)(void*, const void*);
        using PFN_vkQueueEndDebugUtilsLabelEXT = void (*)(void*);
        using PFN_vkQueueInsertDebugUtilsLabelEXT = void (*)(void*, const void*);
        using PFN_vkCmdBeginDebugUtilsLabelEXT = void (*)(FakeCommandBufferObject*, const void*);
        using PFN_vkCmdEndDebugUtilsLabelEXT = void (*)(FakeCommandBufferObject*);
        using PFN_vkCmdInsertDebugUtilsLabelEXT = void (*)(FakeCommandBufferObject*, const void*);
        using PFN_vkDestroySurfaceKHR = void (*)(InstanceHandle, std::uint64_t, const void*);
        using PFN_vkGetPhysicalDeviceSurfaceSupportKHR = int (*)(FakePhysicalDeviceObject*,
                                                                 std::uint32_t,
                                                                 std::uint64_t,
                                                                 std::uint32_t*);
        using PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR = int (*)(FakePhysicalDeviceObject*, std::uint64_t, void*);
        using PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR = int (*)(FakePhysicalDeviceObject*, const void*, void*);
        using PFN_vkGetPhysicalDeviceSurfaceFormatsKHR = int (*)(FakePhysicalDeviceObject*,
                                                                 std::uint64_t,
                                                                 std::uint32_t*,
                                                                 void*);
        using PFN_vkGetPhysicalDeviceSurfaceFormats2KHR = int (*)(FakePhysicalDeviceObject*,
                                                                  const void*,
                                                                  std::uint32_t*,
                                                                  void*);
        using PFN_vkGetPhysicalDeviceSurfacePresentModesKHR = int (*)(FakePhysicalDeviceObject*,
                                                                      std::uint64_t,
                                                                      std::uint32_t*,
                                                                      void*);
        using PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR = int (*)(FakePhysicalDeviceObject*,
                                                                       std::uint32_t*,
                                                                       void*);
        using PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT = int (*)(FakePhysicalDeviceObject*,
                                                                           std::uint32_t*,
                                                                           void*);
        using PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR = int (*)(FakePhysicalDeviceObject*,
                                                                           std::uint32_t*,
                                                                           void*);
    };

    using FakeDeviceCallFn = int (*)(FakeDeviceObject*, std::uint32_t*);
    using FakeWin32MemoryHandleFn = int (*)(FakeDeviceObject*, const void*, void**);

    struct FakeDeviceTable final
    {
        FakeDeviceCallFn vkDestroyDevice = nullptr;
        FakeDeviceCallFn vkDeviceWaitIdle = nullptr;
        FakeDeviceCallFn vkGetDeviceQueue = nullptr;
        FakeDeviceCallFn vkQueueSubmit = nullptr;
        FakeDeviceCallFn vkQueueWaitIdle = nullptr;
        FakeDeviceCallFn vkAllocateMemory = nullptr;
        FakeDeviceCallFn vkFreeMemory = nullptr;
        FakeDeviceCallFn vkMapMemory = nullptr;
        FakeDeviceCallFn vkUnmapMemory = nullptr;
        FakeDeviceCallFn vkFlushMappedMemoryRanges = nullptr;
        FakeDeviceCallFn vkInvalidateMappedMemoryRanges = nullptr;
        FakeDeviceCallFn vkBindBufferMemory = nullptr;
        FakeDeviceCallFn vkBindImageMemory = nullptr;
        FakeDeviceCallFn vkGetBufferMemoryRequirements = nullptr;
        FakeDeviceCallFn vkGetImageMemoryRequirements = nullptr;
        FakeDeviceCallFn vkCreateBuffer = nullptr;
        FakeDeviceCallFn vkDestroyBuffer = nullptr;
        FakeDeviceCallFn vkCreateImage = nullptr;
        FakeDeviceCallFn vkDestroyImage = nullptr;
        FakeDeviceCallFn vkCreateCommandPool = nullptr;
        FakeDeviceCallFn vkDestroyCommandPool = nullptr;
        FakeDeviceCallFn vkAllocateCommandBuffers = nullptr;
        FakeDeviceCallFn vkBeginCommandBuffer = nullptr;
        FakeDeviceCallFn vkEndCommandBuffer = nullptr;
        FakeDeviceCallFn vkCmdCopyBuffer = nullptr;
        FakeDeviceCallFn vkBindBufferMemory2 = nullptr;
        FakeDeviceCallFn vkBindBufferMemory2KHR = nullptr;
        FakeDeviceCallFn vkBindImageMemory2 = nullptr;
        FakeDeviceCallFn vkBindImageMemory2KHR = nullptr;
        FakeDeviceCallFn vkGetBufferMemoryRequirements2 = nullptr;
        FakeDeviceCallFn vkGetBufferMemoryRequirements2KHR = nullptr;
        FakeDeviceCallFn vkGetImageMemoryRequirements2 = nullptr;
        FakeDeviceCallFn vkGetImageMemoryRequirements2KHR = nullptr;
        FakeDeviceCallFn vkGetDeviceBufferMemoryRequirements = nullptr;
        FakeDeviceCallFn vkGetDeviceBufferMemoryRequirementsKHR = nullptr;
        FakeDeviceCallFn vkGetDeviceImageMemoryRequirements = nullptr;
        FakeDeviceCallFn vkGetDeviceImageMemoryRequirementsKHR = nullptr;
        FakeWin32MemoryHandleFn vkGetMemoryWin32HandleKHR = nullptr;
    };

    struct FakeVmaVulkanFunctions final
    {
        FakeTraits::PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
        FakeTraits::PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr = nullptr;
        FakeTraits::PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = nullptr;
        FakeTraits::PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr;
        FakeDeviceCallFn vkAllocateMemory = nullptr;
        FakeDeviceCallFn vkFreeMemory = nullptr;
        FakeDeviceCallFn vkMapMemory = nullptr;
        FakeDeviceCallFn vkUnmapMemory = nullptr;
        FakeDeviceCallFn vkFlushMappedMemoryRanges = nullptr;
        FakeDeviceCallFn vkInvalidateMappedMemoryRanges = nullptr;
        FakeDeviceCallFn vkBindBufferMemory = nullptr;
        FakeDeviceCallFn vkBindImageMemory = nullptr;
        FakeDeviceCallFn vkGetBufferMemoryRequirements = nullptr;
        FakeDeviceCallFn vkGetImageMemoryRequirements = nullptr;
        FakeDeviceCallFn vkCreateBuffer = nullptr;
        FakeDeviceCallFn vkDestroyBuffer = nullptr;
        FakeDeviceCallFn vkCreateImage = nullptr;
        FakeDeviceCallFn vkDestroyImage = nullptr;
        FakeDeviceCallFn vkCmdCopyBuffer = nullptr;
        FakeDeviceCallFn vkGetBufferMemoryRequirements2KHR = nullptr;
        FakeDeviceCallFn vkGetImageMemoryRequirements2KHR = nullptr;
        FakeDeviceCallFn vkBindBufferMemory2KHR = nullptr;
        FakeDeviceCallFn vkBindImageMemory2KHR = nullptr;
        FakeTraits::PFN_vkGetPhysicalDeviceMemoryProperties2KHR vkGetPhysicalDeviceMemoryProperties2KHR = nullptr;
        FakeDeviceCallFn vkGetDeviceBufferMemoryRequirements = nullptr;
        FakeDeviceCallFn vkGetDeviceImageMemoryRequirements = nullptr;
    };

    struct FakeSlotCounters final
    {
        std::uint32_t instanceDestroyed = 0;
        std::uint32_t messengerDestroyed = 0;
        std::uint32_t physicalProps = 0;
        std::uint32_t physicalMemoryProps2 = 0;
        std::uint32_t formatProps = 0;
        std::uint32_t objectNamed = 0;
        std::uint32_t cmdLabels = 0;
        std::uint32_t bufferCreated = 0;
        std::uint32_t memoryAllocated = 0;
        std::uint32_t bufferReqs2 = 0;
        std::uint32_t deviceBufferReqs = 0;
        std::uint32_t win32MemoryHandle = 0;
    };

    FakeSlotCounters gFakeCountersA{};
    FakeSlotCounters gFakeCountersB{};

    template<FakeSlotCounters* Counters, int ExpectedTag>
    struct FakeSlotCallbacks final
    {
        static void destroyInstance(FakeInstanceObject* instance, const void*)
        {
            REQUIRE(instance != nullptr);
            CHECK(instance->tag == ExpectedTag);
            ++Counters->instanceDestroyed;
        }

        static void destroyMessenger(FakeInstanceObject* instance, std::uint64_t, const void*)
        {
            REQUIRE(instance != nullptr);
            CHECK(instance->tag == ExpectedTag);
            ++Counters->messengerDestroyed;
        }

        static void getPhysicalProps(FakePhysicalDeviceObject* physical, std::uint32_t* outTag)
        {
            REQUIRE(physical != nullptr);
            CHECK(physical->tag == ExpectedTag);
            if (outTag != nullptr)
            {
                *outTag = static_cast<std::uint32_t>(ExpectedTag);
            }
            ++Counters->physicalProps;
        }

        static void getPhysicalMemoryProps2(FakePhysicalDeviceObject* physical, std::uint32_t* outTag)
        {
            REQUIRE(physical != nullptr);
            CHECK(physical->tag == ExpectedTag);
            if (outTag != nullptr)
            {
                *outTag = static_cast<std::uint32_t>(ExpectedTag);
            }
            ++Counters->physicalMemoryProps2;
        }

        static void getFormatProps(FakePhysicalDeviceObject* physical, std::uint32_t, std::uint32_t* outFeatures)
        {
            REQUIRE(physical != nullptr);
            CHECK(physical->tag == ExpectedTag);
            if (outFeatures != nullptr)
            {
                *outFeatures = static_cast<std::uint32_t>(ExpectedTag * 10);
            }
            ++Counters->formatProps;
        }

        static int setObjectName(FakeDeviceObject* device, const void*)
        {
            REQUIRE(device != nullptr);
            CHECK(device->tag == ExpectedTag);
            ++Counters->objectNamed;
            return 0;
        }

        static void cmdBeginLabel(FakeCommandBufferObject* commandBuffer, const void*)
        {
            REQUIRE(commandBuffer != nullptr);
            CHECK(commandBuffer->tag == ExpectedTag);
            ++Counters->cmdLabels;
        }

        static void cmdEndLabel(FakeCommandBufferObject* commandBuffer)
        {
            REQUIRE(commandBuffer != nullptr);
            CHECK(commandBuffer->tag == ExpectedTag);
            ++Counters->cmdLabels;
        }

        static void cmdInsertLabel(FakeCommandBufferObject* commandBuffer, const void*)
        {
            REQUIRE(commandBuffer != nullptr);
            CHECK(commandBuffer->tag == ExpectedTag);
            ++Counters->cmdLabels;
        }

        static int createBuffer(FakeDeviceObject* device, std::uint32_t* outTag)
        {
            REQUIRE(device != nullptr);
            CHECK(device->tag == ExpectedTag);
            if (outTag != nullptr)
            {
                *outTag = static_cast<std::uint32_t>(ExpectedTag);
            }
            ++Counters->bufferCreated;
            return 0;
        }

        static int allocateMemory(FakeDeviceObject* device, std::uint32_t* outTag)
        {
            REQUIRE(device != nullptr);
            CHECK(device->tag == ExpectedTag);
            if (outTag != nullptr)
            {
                *outTag = static_cast<std::uint32_t>(ExpectedTag);
            }
            ++Counters->memoryAllocated;
            return 0;
        }

        static int bufferMemoryReqs2(FakeDeviceObject* device, std::uint32_t* outTag)
        {
            REQUIRE(device != nullptr);
            CHECK(device->tag == ExpectedTag);
            if (outTag != nullptr)
            {
                *outTag = static_cast<std::uint32_t>(ExpectedTag);
            }
            ++Counters->bufferReqs2;
            return 0;
        }

        static int deviceBufferMemoryReqs(FakeDeviceObject* device, std::uint32_t* outTag)
        {
            REQUIRE(device != nullptr);
            CHECK(device->tag == ExpectedTag);
            if (outTag != nullptr)
            {
                *outTag = static_cast<std::uint32_t>(ExpectedTag);
            }
            ++Counters->deviceBufferReqs;
            return 0;
        }

        static int getMemoryWin32Handle(FakeDeviceObject* device, const void*, void** outHandle)
        {
            REQUIRE(device != nullptr);
            CHECK(device->tag == ExpectedTag);
            if (outHandle != nullptr)
            {
                *outHandle = reinterpret_cast<void*>(static_cast<std::uintptr_t>(ExpectedTag));
            }
            ++Counters->win32MemoryHandle;
            return 0;
        }

        static FakeVoidFn getDeviceProcAddr(FakeDeviceObject* device, const char* name)
        {
            if (device == nullptr || device->tag != ExpectedTag || name == nullptr)
            {
                return nullptr;
            }
            const std::string_view symbol(name);
            if (symbol == "vkCreateBuffer")
            {
                return reinterpret_cast<FakeVoidFn>(&createBuffer);
            }
            if (symbol == "vkAllocateMemory")
            {
                return reinterpret_cast<FakeVoidFn>(&allocateMemory);
            }
            if (symbol == "vkGetBufferMemoryRequirements2")
            {
                return reinterpret_cast<FakeVoidFn>(&bufferMemoryReqs2);
            }
            if (symbol == "vkGetDeviceBufferMemoryRequirements")
            {
                return reinterpret_cast<FakeVoidFn>(&deviceBufferMemoryReqs);
            }
            if (symbol == "vkGetMemoryWin32HandleKHR")
            {
                return reinterpret_cast<FakeVoidFn>(&getMemoryWin32Handle);
            }
            return nullptr;
        }

        static FakeVoidFn getInstanceProcAddr(FakeInstanceObject* instance, const char* name)
        {
            if (instance == nullptr || instance->tag != ExpectedTag || name == nullptr)
            {
                return nullptr;
            }
            const std::string_view symbol(name);
            if (symbol == "vkGetInstanceProcAddr")
            {
                return reinterpret_cast<FakeVoidFn>(&getInstanceProcAddr);
            }
            if (symbol == "vkGetDeviceProcAddr")
            {
                return reinterpret_cast<FakeVoidFn>(&getDeviceProcAddr);
            }
            if (symbol == "vkDestroyInstance")
            {
                return reinterpret_cast<FakeVoidFn>(&destroyInstance);
            }
            if (symbol == "vkDestroyDebugUtilsMessengerEXT")
            {
                return reinterpret_cast<FakeVoidFn>(&destroyMessenger);
            }
            if (symbol == "vkGetPhysicalDeviceProperties")
            {
                return reinterpret_cast<FakeVoidFn>(&getPhysicalProps);
            }
            if (symbol == "vkGetPhysicalDeviceMemoryProperties2")
            {
                return reinterpret_cast<FakeVoidFn>(&getPhysicalMemoryProps2);
            }
            if (symbol == "vkGetPhysicalDeviceFormatProperties")
            {
                return reinterpret_cast<FakeVoidFn>(&getFormatProps);
            }
            if (symbol == "vkSetDebugUtilsObjectNameEXT")
            {
                return reinterpret_cast<FakeVoidFn>(&setObjectName);
            }
            if (symbol == "vkCmdBeginDebugUtilsLabelEXT")
            {
                return reinterpret_cast<FakeVoidFn>(&cmdBeginLabel);
            }
            if (symbol == "vkCmdEndDebugUtilsLabelEXT")
            {
                return reinterpret_cast<FakeVoidFn>(&cmdEndLabel);
            }
            if (symbol == "vkCmdInsertDebugUtilsLabelEXT")
            {
                return reinterpret_cast<FakeVoidFn>(&cmdInsertLabel);
            }
            return nullptr;
        }
    };

    FakeVoidFn syntheticGetInstanceProcAddr(FakeInstanceObject* instance, const char* name)
    {
        if (instance == nullptr)
        {
            return nullptr;
        }
        if (instance->tag == 1)
        {
            return FakeSlotCallbacks<&gFakeCountersA, 1>::getInstanceProcAddr(instance, name);
        }
        if (instance->tag == 2)
        {
            return FakeSlotCallbacks<&gFakeCountersB, 2>::getInstanceProcAddr(instance, name);
        }
        return nullptr;
    }

#if RACHIS_ENABLE_VULKAN

    struct RealVulkanCounters final
    {
        std::uint32_t instanceDestroyed = 0;
        std::uint32_t physicalProps = 0;
        std::uint32_t formatProps = 0;
        std::uint32_t objectNamed = 0;
        std::uint32_t bufferCreated = 0;
        std::uint32_t queuePresent = 0;
    };

    RealVulkanCounters gRealCountersA{};
    RealVulkanCounters gRealCountersB{};

    template<typename Handle>
    [[nodiscard]] Handle fakeVkHandle(std::uintptr_t value) noexcept
    {
        return reinterpret_cast<Handle>(value);
    }

    template<RealVulkanCounters* Counters, std::uintptr_t Tag>
    struct RealVulkanSlotCallbacks final
    {
        static void VKAPI_PTR destroyInstance(VkInstance instance, const VkAllocationCallbacks*)
        {
            CHECK(instance == fakeVkHandle<VkInstance>(Tag));
            ++Counters->instanceDestroyed;
        }

        static void VKAPI_PTR getPhysicalDeviceProperties(VkPhysicalDevice physicalDevice,
                                                          VkPhysicalDeviceProperties* properties)
        {
            CHECK(physicalDevice == fakeVkHandle<VkPhysicalDevice>(Tag));
            if (properties != nullptr)
            {
                *properties = {};
                properties->vendorID = static_cast<std::uint32_t>(Tag);
            }
            ++Counters->physicalProps;
        }

        static void VKAPI_PTR getPhysicalDeviceFormatProperties(VkPhysicalDevice physicalDevice,
                                                                VkFormat,
                                                                VkFormatProperties* formatProperties)
        {
            CHECK(physicalDevice == fakeVkHandle<VkPhysicalDevice>(Tag));
            if (formatProperties != nullptr)
            {
                *formatProperties = {};
                formatProperties->optimalTilingFeatures = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
            }
            ++Counters->formatProps;
        }

        static VkResult VKAPI_PTR setDebugUtilsObjectName(VkDevice device, const VkDebugUtilsObjectNameInfoEXT* info)
        {
            CHECK(device == fakeVkHandle<VkDevice>(Tag));
            REQUIRE(info != nullptr);
            CHECK(info->sType == VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT);
            ++Counters->objectNamed;
            return VK_SUCCESS;
        }

        static VkResult VKAPI_PTR createBuffer(VkDevice device,
                                               const VkBufferCreateInfo*,
                                               const VkAllocationCallbacks*,
                                               VkBuffer* buffer)
        {
            CHECK(device == fakeVkHandle<VkDevice>(Tag));
            if (buffer != nullptr)
            {
                *buffer = fakeVkHandle<VkBuffer>(Tag * 100U);
            }
            ++Counters->bufferCreated;
            return VK_SUCCESS;
        }

        static VkResult VKAPI_PTR queuePresent(VkQueue, const VkPresentInfoKHR*)
        {
            ++Counters->queuePresent;
            return VK_SUCCESS;
        }

        static PFN_vkVoidFunction VKAPI_PTR getDeviceProcAddr(VkDevice device, const char* name)
        {
            if (device != fakeVkHandle<VkDevice>(Tag) || name == nullptr)
            {
                return nullptr;
            }
            const std::string_view symbol(name);
            if (symbol == "vkCreateBuffer")
            {
                return reinterpret_cast<PFN_vkVoidFunction>(&createBuffer);
            }
            if (symbol == "vkQueuePresentKHR")
            {
                return reinterpret_cast<PFN_vkVoidFunction>(&queuePresent);
            }
            return nullptr;
        }

        static PFN_vkVoidFunction VKAPI_PTR getInstanceProcAddr(VkInstance instance, const char* name)
        {
            if (instance != fakeVkHandle<VkInstance>(Tag) || name == nullptr)
            {
                return nullptr;
            }
            const std::string_view symbol(name);
            if (symbol == "vkGetInstanceProcAddr")
            {
                return reinterpret_cast<PFN_vkVoidFunction>(&getInstanceProcAddr);
            }
            if (symbol == "vkGetDeviceProcAddr")
            {
                return reinterpret_cast<PFN_vkVoidFunction>(&getDeviceProcAddr);
            }
            if (symbol == "vkDestroyInstance")
            {
                return reinterpret_cast<PFN_vkVoidFunction>(&destroyInstance);
            }
            if (symbol == "vkGetPhysicalDeviceProperties")
            {
                return reinterpret_cast<PFN_vkVoidFunction>(&getPhysicalDeviceProperties);
            }
            if (symbol == "vkGetPhysicalDeviceFormatProperties")
            {
                return reinterpret_cast<PFN_vkVoidFunction>(&getPhysicalDeviceFormatProperties);
            }
            if (symbol == "vkSetDebugUtilsObjectNameEXT")
            {
                return reinterpret_cast<PFN_vkVoidFunction>(&setDebugUtilsObjectName);
            }
            return nullptr;
        }
    };

    PFN_vkVoidFunction VKAPI_PTR syntheticVkGetInstanceProcAddr(VkInstance instance, const char* name)
    {
        if (instance == fakeVkHandle<VkInstance>(1))
        {
            return RealVulkanSlotCallbacks<&gRealCountersA, 1>::getInstanceProcAddr(instance, name);
        }
        if (instance == fakeVkHandle<VkInstance>(2))
        {
            return RealVulkanSlotCallbacks<&gRealCountersB, 2>::getInstanceProcAddr(instance, name);
        }
        return nullptr;
    }

#endif

} // namespace

TEST_CASE("allocator function population falls back to KHR aliases and clears tables on null handles",
          "[size:small][rhi][vulkan][dispatch]")
{
    rachis::vulkan::VulkanInstanceTableT<FakeTraits> instanceTable{};
    FakePhysicalDeviceObject physical{.tag = 1};
    FakeDeviceObject device{.tag = 1};
    gFakeCountersA = {};

    instanceTable.vkGetPhysicalDeviceMemoryProperties2KHR =
        &FakeSlotCallbacks<&gFakeCountersA, 1>::getPhysicalMemoryProps2;

    FakeDeviceTable deviceTable{};
    deviceTable.vkGetBufferMemoryRequirements2KHR = &FakeSlotCallbacks<&gFakeCountersA, 1>::bufferMemoryReqs2;
    deviceTable.vkGetDeviceBufferMemoryRequirementsKHR = &FakeSlotCallbacks<&gFakeCountersA, 1>::deviceBufferMemoryReqs;

    FakeVmaVulkanFunctions vma{};
    rachis::vulkan::populateVulkanAllocatorFunctions(vma, instanceTable, deviceTable);
    REQUIRE(vma.vkGetPhysicalDeviceMemoryProperties2KHR != nullptr);
    REQUIRE(vma.vkGetBufferMemoryRequirements2KHR != nullptr);
    REQUIRE(vma.vkGetDeviceBufferMemoryRequirements != nullptr);

    std::uint32_t outTag = 0;
    vma.vkGetPhysicalDeviceMemoryProperties2KHR(&physical, &outTag);
    CHECK(vma.vkGetBufferMemoryRequirements2KHR(&device, &outTag) == 0);
    CHECK(vma.vkGetDeviceBufferMemoryRequirements(&device, &outTag) == 0);
    CHECK(gFakeCountersA.physicalMemoryProps2 == 1);
    CHECK(gFakeCountersA.bufferReqs2 == 1);
    CHECK(gFakeCountersA.deviceBufferReqs == 1);

    rachis::vulkan::loadVulkanInstanceTable(
        instanceTable, static_cast<FakeInstanceObject*>(nullptr), &syntheticGetInstanceProcAddr);
    CHECK(instanceTable.vkGetPhysicalDeviceMemoryProperties2KHR == nullptr);

    rachis::vulkan::loadVulkanBasicDeviceTable(deviceTable, static_cast<FakeDeviceObject*>(nullptr), nullptr);
    CHECK(deviceTable.vkGetBufferMemoryRequirements2KHR == nullptr);
    CHECK(deviceTable.vkGetDeviceBufferMemoryRequirementsKHR == nullptr);
}

#if RACHIS_ENABLE_VULKAN
TEST_CASE("VulkanInstanceTable VolkDeviceTable and VmaVulkanFunctions isolate instances without volk globals",
          "[size:small][rhi][vulkan][dispatch]")
{
    gRealCountersA = {};
    gRealCountersB = {};

    const auto instanceA = fakeVkHandle<VkInstance>(1);
    const auto physicalA = fakeVkHandle<VkPhysicalDevice>(1);
    const auto deviceA = fakeVkHandle<VkDevice>(1);

    rachis::vulkan::VulkanInstanceTable instanceTableA{};
    rachis::vulkan::loadVulkanInstanceTable(instanceTableA, instanceA, &syntheticVkGetInstanceProcAddr);

    VolkDeviceTable deviceTableA{};
    rachis::vulkan::loadVulkanDeviceTable(deviceTableA, deviceA, instanceTableA.vkGetDeviceProcAddr);

    VmaVulkanFunctions vmaFunctionsA{};
    rachis::vulkan::populateVulkanAllocatorFunctions(vmaFunctionsA, instanceTableA, deviceTableA);

    {
        const auto instanceB = fakeVkHandle<VkInstance>(2);
        const auto physicalB = fakeVkHandle<VkPhysicalDevice>(2);
        const auto deviceB = fakeVkHandle<VkDevice>(2);

        rachis::vulkan::VulkanInstanceTable instanceTableB{};
        rachis::vulkan::loadVulkanInstanceTable(instanceTableB, instanceB, &syntheticVkGetInstanceProcAddr);

        VolkDeviceTable deviceTableB{};
        rachis::vulkan::loadVulkanDeviceTable(deviceTableB, deviceB, instanceTableB.vkGetDeviceProcAddr);

        VkPhysicalDeviceProperties propsB{};
        REQUIRE(instanceTableB.vkGetPhysicalDeviceProperties != nullptr);
        instanceTableB.vkGetPhysicalDeviceProperties(physicalB, &propsB);
        CHECK(propsB.vendorID == 2U);

        VkBuffer bufferB = VK_NULL_HANDLE;
        REQUIRE(deviceTableB.vkCreateBuffer != nullptr);
        CHECK(deviceTableB.vkCreateBuffer(deviceB, nullptr, nullptr, &bufferB) == VK_SUCCESS);
        CHECK(bufferB == fakeVkHandle<VkBuffer>(200));

        REQUIRE(instanceTableB.vkDestroyInstance != nullptr);
        instanceTableB.vkDestroyInstance(instanceB, nullptr);
    }

    CHECK(volkGetLoadedInstance() == VK_NULL_HANDLE);
    CHECK(volkGetLoadedDevice() == VK_NULL_HANDLE);
    CHECK(::vkDestroyInstance == nullptr);
    CHECK(::vkGetPhysicalDeviceFormatProperties == nullptr);
    CHECK(::vkCreateBuffer == nullptr);
    CHECK(::vkSetDebugUtilsObjectNameEXT == nullptr);

    VkFormatProperties formatPropsA{};
    REQUIRE(instanceTableA.vkGetPhysicalDeviceFormatProperties != nullptr);
    instanceTableA.vkGetPhysicalDeviceFormatProperties(physicalA, VK_FORMAT_R8G8B8A8_UNORM, &formatPropsA);
    CHECK((formatPropsA.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0);

    rachis::vulkan::setVulkanObjectName(&instanceTableA, deviceA, VK_OBJECT_TYPE_BUFFER, 42U, "Isolated A buffer");

    VkBuffer bufferA = VK_NULL_HANDLE;
    REQUIRE(deviceTableA.vkCreateBuffer != nullptr);
    CHECK(deviceTableA.vkCreateBuffer(deviceA, nullptr, nullptr, &bufferA) == VK_SUCCESS);
    CHECK(bufferA == fakeVkHandle<VkBuffer>(100));
    REQUIRE(deviceTableA.vkQueuePresentKHR != nullptr);
    CHECK(deviceTableA.vkQueuePresentKHR(VK_NULL_HANDLE, nullptr) == VK_SUCCESS);

    VkPhysicalDeviceProperties propsA{};
    REQUIRE(vmaFunctionsA.vkGetPhysicalDeviceProperties != nullptr);
    vmaFunctionsA.vkGetPhysicalDeviceProperties(physicalA, &propsA);
    CHECK(propsA.vendorID == 1U);
    REQUIRE(vmaFunctionsA.vkCreateBuffer != nullptr);
    CHECK(vmaFunctionsA.vkCreateBuffer(deviceA, nullptr, nullptr, &bufferA) == VK_SUCCESS);

    CHECK(gRealCountersA.formatProps == 1);
    CHECK(gRealCountersA.objectNamed == 1);
    CHECK(gRealCountersA.bufferCreated == 2);
    CHECK(gRealCountersA.queuePresent == 1);
    CHECK(gRealCountersA.physicalProps == 1);

    CHECK(gRealCountersB.physicalProps == 1);
    CHECK(gRealCountersB.bufferCreated == 1);
    CHECK(gRealCountersB.instanceDestroyed == 1);
    CHECK(gRealCountersB.formatProps == 0);
    CHECK(gRealCountersB.objectNamed == 0);
}
#endif
