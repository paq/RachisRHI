#include "Rachis/Command.h"
#include "Rachis/Device.h"
#include "Rachis/Pipeline.h"
#include "Rachis/Presentation.h"
#include "Rachis/Resource.h"

#include <catch2/catch_test_macros.hpp>

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if RACHIS_ENABLE_VULKAN
#include "Internal/VulkanBindings.h"
#include "Internal/VulkanCommands.h"
#include "Internal/VulkanGraphics.h"
#include "Internal/VulkanPipelines.h"
#include "Internal/VulkanQueries.h"
#include "Internal/VulkanQueues.h"
#include "Internal/VulkanResources.h"
#include "Internal/VulkanRuntime.h"
#include "Internal/VulkanShaders.h"
#include "Internal/VulkanSwapchain.h"
#endif

#if RACHIS_ENABLE_VULKAN

namespace
{

    template<typename HandleT>
    [[nodiscard]] HandleT fakeVk(std::uintptr_t raw) noexcept
    {
        return reinterpret_cast<HandleT>(raw);
    }

    struct SpirvBuilder final
    {
        std::vector<std::uint32_t> words{0x07230203U, 0x00010500U, 0U, 128U, 0U};

        void op(std::uint16_t opcode, std::initializer_list<std::uint32_t> operands)
        {
            const auto wordCount = static_cast<std::uint32_t>(1 + operands.size());
            words.push_back((wordCount << 16U) | opcode);
            words.insert(words.end(), operands.begin(), operands.end());
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

    [[nodiscard]] SpirvBuilder makeMinimalComputeSpirv()
    {
        SpirvBuilder spv;
        spv.op(17, {1});
        spv.op(14, {0, 1});
        spv.entryPoint(5, 10, "main", {});
        spv.op(16, {10, 17, 1, 1, 1});
        spv.op(19, {1});
        spv.op(33, {2, 1});
        spv.op(54, {1, 10, 0, 2});
        spv.op(248, {20});
        spv.op(253, {});
        spv.op(56, {});
        return spv;
    }

    [[nodiscard]] SpirvBuilder makeMinimalGraphicsSpirv()
    {
        SpirvBuilder spv;
        spv.op(17, {1});
        spv.op(14, {0, 1});
        spv.entryPoint(0, 10, "vsMain", {});
        spv.entryPoint(4, 11, "psMain", {});
        spv.op(16, {11, 7});
        spv.op(19, {1});
        spv.op(33, {2, 1});
        spv.op(54, {1, 10, 0, 2});
        spv.op(248, {20});
        spv.op(253, {});
        spv.op(56, {});
        spv.op(54, {1, 11, 0, 2});
        spv.op(248, {21});
        spv.op(253, {});
        spv.op(56, {});
        return spv;
    }

    [[nodiscard]] rachis::DeviceCapabilities baselinePerfCapabilities() noexcept
    {
        rachis::DeviceCapabilities caps{};
        caps.limits.maxTextureDimension1D = 16384;
        caps.limits.maxTextureDimension2D = 16384;
        caps.limits.maxTextureDimension3D = 2048;
        caps.limits.maxTextureArrayLayers = 2048;
        caps.limits.maxTextureMipLevels = 15;
        caps.limits.maxComputeWorkgroupInvocations = 1024;
        caps.limits.maxComputeWorkgroupSize = {1024, 1024, 64};
        caps.limits.maxComputeWorkgroupCount = {65535, 65535, 65535};
        caps.limits.maxBufferSize = 1ULL << 30;
        caps.limits.maxUniformBufferSize = 65536;
        caps.limits.maxColorAttachments = 8;
        caps.limits.maxVertexBuffers = 8;
        caps.limits.maxVertexAttributes = 16;
        caps.limits.maxBindingSets = 4;
        caps.limits.maxPushConstantBytes = 128;
        return caps;
    }

    enum class MemoryTopologyKind : std::uint8_t
    {
        coherentHostVisible,
        nonCoherentHostVisible,
        rebarDiscrete,
        smallBarDiscrete,
    };

    enum class PerfCommandKind : std::uint8_t
    {
        copyQueryResults,
        resetQueryPool,
        memoryBarrier,
        bufferBarrier,
    };

    struct PerfCommandEvent final
    {
        PerfCommandKind kind{};
        VkQueryPool queryPool = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        VkDeviceSize size = 0;
        VkDeviceSize stride = 0;
        VkQueryResultFlags resultFlags = 0;
        VkPipelineStageFlags2 srcStageMask = 0;
        VkAccessFlags2 srcAccessMask = 0;
        VkPipelineStageFlags2 dstStageMask = 0;
        VkAccessFlags2 dstAccessMask = 0;
    };

    struct PerfMockState final
    {
        std::mutex mutex;
        std::uintptr_t nextHandle = 0x10000;
        MemoryTopologyKind memoryTopology = MemoryTopologyKind::coherentHostVisible;
        float maxSamplerLodBias = 16.0F;

        std::vector<std::unique_ptr<std::byte[]>> hostAllocations;
        std::vector<std::uint32_t> allocatedMemoryTypes;
        std::vector<std::optional<float>> allocatedMemoryPriorities;
        std::vector<VkDeviceSize> allocatedMemorySizes;
        bool requireDedicatedAllocation = false;
        std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS> usedHeapBytes{};
        std::unordered_map<VkDeviceMemory, std::pair<std::uint32_t, VkDeviceSize>> memoryAllocations;
        std::unordered_map<VkBuffer, VkDeviceSize> bufferSizes;
        std::unordered_map<VkImage, VkImageCreateFlags> imageCreateFlags;
        VkDeviceSize aliasImageSize = 0;
        bool aliasImageRequiresDedicated = false;

        std::atomic<std::uint32_t> mapMemoryCalls{0};
        std::atomic<std::uint32_t> unmapMemoryCalls{0};
        std::atomic<std::uint32_t> flushRangeCalls{0};
        std::atomic<std::uint32_t> invalidateRangeCalls{0};
        std::atomic<VkDeviceSize> lastFlushSize{0};
        std::atomic<VkResult> flushResult{VK_SUCCESS};

        std::atomic<std::uint32_t> createBufferCalls{0};
        std::atomic<std::uint32_t> destroyBufferCalls{0};
        std::atomic<std::uint32_t> createImageCalls{0};
        std::atomic<std::uint32_t> destroyImageCalls{0};
        std::atomic<std::uint32_t> createImageViewCalls{0};
        std::atomic<std::uint32_t> destroyImageViewCalls{0};
        std::atomic<std::uint32_t> createSamplerCalls{0};
        std::atomic<std::uint32_t> destroySamplerCalls{0};
        std::vector<float> samplerMipLodBiases;

        std::atomic<std::uint32_t> legacyBufferMemReq2Calls{0};
        std::atomic<std::uint32_t> legacyImageMemReq2Calls{0};

        std::uint32_t imageFormatProps2Calls = 0;
        VkImageCreateFlags imageFormatQueryFlags = 0;
        VkImageUsageFlags imageFormatQueryUsage = 0;
        std::vector<VkFormat> imageFormatQueryViewFormats;
        VkResult imageFormatQueryResult = VK_SUCCESS;
        std::uint32_t imageFormatQueryMaxMipLevels = 15;

        std::atomic<std::uint32_t> createdPipelines{0};
        std::atomic<std::uint32_t> destroyedPipelines{0};
        std::atomic<std::uint32_t> createdDescPools{0};
        VkResult updateAfterBindDescPoolResult = VK_SUCCESS;
        std::atomic<std::uint32_t> destroyedDescPools{0};
        std::atomic<std::uint32_t> boundPipelines{0};
        std::atomic<std::uint32_t> boundDescriptorSetCalls{0};

        std::vector<VkSharingMode> bufferSharingModes;
        std::vector<std::uint32_t> bufferQueueFamilyCounts;
        std::vector<std::vector<std::uint32_t>> bufferQueueFamilies;
        std::vector<VkSharingMode> imageSharingModes;
        std::vector<std::uint32_t> imageQueueFamilyCounts;
        std::vector<std::vector<std::uint32_t>> imageQueueFamilies;

        std::vector<VkBufferMemoryBarrier2> capturedBufferBarriers;
        std::vector<VkImageMemoryBarrier2> capturedImageBarriers;
        std::vector<PerfCommandEvent> commandEvents;

        std::atomic<std::uint32_t> surfaceCapsQueries{0};
        std::atomic<std::uint32_t> queuePresentCalls{0};
        std::atomic<std::uint32_t> releaseSwapchainImagesCalls{0};
        VkExtent2D surfaceCurrentExtent{128, 128};
        VkResult nextPresentResult = VK_SUCCESS;
        VkResult nextAcquireNativeResult = VK_SUCCESS;
        VkResult nextCreateSwapchainResult = VK_SUCCESS;
        VkResult waitForPresentResult = VK_SUCCESS;
        std::atomic<VkResult> queueWaitIdleResult{VK_SUCCESS};
        std::vector<std::uint64_t> waitedPresentIds;
        std::vector<std::uint64_t> waitedPresentTimeouts;
        std::vector<VkSwapchainKHR> createdSwapchains;
        std::vector<VkSwapchainKHR> oldSwapchains;
        std::vector<VkSwapchainKHR> destroyedSwapchains;
        std::uint32_t nextAcquireImageIndex = 0;
        std::unordered_map<VkSemaphore, std::uint64_t> timelineValues;
    };

    PerfMockState* gPerfMock = nullptr;

#ifdef VK_KHR_swapchain_maintenance1
    using PerfReleaseSwapchainImagesInfo = VkReleaseSwapchainImagesInfoKHR;
#else
    using PerfReleaseSwapchainImagesInfo = VkReleaseSwapchainImagesInfoEXT;
#endif

    VKAPI_ATTR VkResult VKAPI_CALL perfReleaseSwapchainImages(VkDevice, const PerfReleaseSwapchainImagesInfo*)
    {
        gPerfMock->releaseSwapchainImagesCalls.fetch_add(1, std::memory_order_relaxed);
        return VK_SUCCESS;
    }

    VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL perfGetInstanceProcAddr(VkInstance, const char*)
    {
        return nullptr;
    }

    VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL perfGetDeviceProcAddr(VkDevice, const char* name)
    {
        if (name != nullptr && (std::string_view(name) == "vkReleaseSwapchainImagesKHR" ||
                                std::string_view(name) == "vkReleaseSwapchainImagesEXT"))
        {
            return reinterpret_cast<PFN_vkVoidFunction>(&perfReleaseSwapchainImages);
        }
        return nullptr;
    }

    VKAPI_ATTR void VKAPI_CALL perfGetPhysicalProps(VkPhysicalDevice, VkPhysicalDeviceProperties* props)
    {
        *props = VkPhysicalDeviceProperties{};
        props->apiVersion = VK_API_VERSION_1_3;
        props->limits.maxImageDimension1D = 16384;
        props->limits.maxImageDimension2D = 16384;
        props->limits.maxImageDimension3D = 2048;
        props->limits.maxImageArrayLayers = 2048;
        props->limits.maxUniformBufferRange = 65536;
        props->limits.maxStorageBufferRange = 1U << 28;
        props->limits.minUniformBufferOffsetAlignment = 64;
        props->limits.minStorageBufferOffsetAlignment = 32;
        props->limits.nonCoherentAtomSize = 64;
        props->limits.maxSamplerAnisotropy = 16.0F;
        props->limits.maxSamplerLodBias = gPerfMock->maxSamplerLodBias;
        props->limits.framebufferColorSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT;
        props->limits.framebufferDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT;
    }

    VKAPI_ATTR void VKAPI_CALL perfGetPhysicalProps2(VkPhysicalDevice dev, VkPhysicalDeviceProperties2* props2)
    {
        perfGetPhysicalProps(dev, &props2->properties);
    }

    VKAPI_ATTR void VKAPI_CALL perfGetPhysicalMemProps(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* mem)
    {
        *mem = VkPhysicalDeviceMemoryProperties{};
        switch (gPerfMock->memoryTopology)
        {
        case MemoryTopologyKind::coherentHostVisible:
            mem->memoryHeapCount = 2;
            mem->memoryHeaps[0] = {.size = 1024ULL * 1024ULL * 1024ULL, .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
            mem->memoryHeaps[1] = {.size = 1024ULL * 1024ULL * 1024ULL, .flags = 0};
            mem->memoryTypeCount = 3;
            mem->memoryTypes[0] = {.propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .heapIndex = 0};
            mem->memoryTypes[1] = {
                .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                .heapIndex = 1,
            };
            mem->memoryTypes[2] = {
                .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                 VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                .heapIndex = 1,
            };
            break;
        case MemoryTopologyKind::nonCoherentHostVisible:
            mem->memoryHeapCount = 2;
            mem->memoryHeaps[0] = {.size = 1024ULL * 1024ULL * 1024ULL, .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
            mem->memoryHeaps[1] = {.size = 1024ULL * 1024ULL * 1024ULL, .flags = 0};
            mem->memoryTypeCount = 3;
            mem->memoryTypes[0] = {.propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .heapIndex = 0};
            mem->memoryTypes[1] = {
                .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                .heapIndex = 1,
            };
            mem->memoryTypes[2] = {
                .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                .heapIndex = 1,
            };
            break;
        case MemoryTopologyKind::rebarDiscrete:
            mem->memoryHeapCount = 2;
            mem->memoryHeaps[0] = {.size = 8ULL * 1024ULL * 1024ULL * 1024ULL,
                                   .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
            mem->memoryHeaps[1] = {.size = 16ULL * 1024ULL * 1024ULL * 1024ULL, .flags = 0};
            mem->memoryTypeCount = 4;
            mem->memoryTypes[0] = {.propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .heapIndex = 0};

            mem->memoryTypes[1] = {
                .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                .heapIndex = 1,
            };

            mem->memoryTypes[2] = {
                .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                .heapIndex = 0,
            };

            mem->memoryTypes[3] = {
                .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                 VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                .heapIndex = 1,
            };
            break;
        case MemoryTopologyKind::smallBarDiscrete:
            mem->memoryHeapCount = 3;
            mem->memoryHeaps[0] = {.size = 8ULL * 1024ULL * 1024ULL * 1024ULL,
                                   .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
            mem->memoryHeaps[1] = {.size = 16ULL * 1024ULL * 1024ULL * 1024ULL, .flags = 0};
            mem->memoryHeaps[2] = {.size = 256ULL * 1024ULL * 1024ULL, .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
            mem->memoryTypeCount = 3;
            mem->memoryTypes[0] = {.propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .heapIndex = 0};

            mem->memoryTypes[1] = {
                .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                .heapIndex = 2,
            };

            mem->memoryTypes[2] = {
                .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                .heapIndex = 1,
            };
            break;
        }
    }

    VKAPI_ATTR void VKAPI_CALL perfGetPhysicalMemProps2(VkPhysicalDevice dev, VkPhysicalDeviceMemoryProperties2* mem2)
    {
        perfGetPhysicalMemProps(dev, &mem2->memoryProperties);
    }

    VKAPI_ATTR void VKAPI_CALL perfGetFormatProps(VkPhysicalDevice, VkFormat, VkFormatProperties* out)
    {
        out->linearTilingFeatures = 0;
        out->optimalTilingFeatures =
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
            VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT | VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT |
            VK_FORMAT_FEATURE_BLIT_DST_BIT;
        out->bufferFeatures = VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT | VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT |
                              VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfGetImageFormatProps(VkPhysicalDevice,
                                                           VkFormat,
                                                           VkImageType,
                                                           VkImageTiling,
                                                           VkImageUsageFlags,
                                                           VkImageCreateFlags,
                                                           VkImageFormatProperties* out)
    {
        *out = VkImageFormatProperties{
            .maxExtent = {16384, 16384, 2048},
            .maxMipLevels = 15,
            .maxArrayLayers = 2048,
            .sampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
            .maxResourceSize = 1ULL << 30,
        };
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfGetImageFormatProps2(VkPhysicalDevice physical,
                                                            const VkPhysicalDeviceImageFormatInfo2* info,
                                                            VkImageFormatProperties2* out)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        ++gPerfMock->imageFormatProps2Calls;
        gPerfMock->imageFormatQueryFlags = info->flags;
        gPerfMock->imageFormatQueryUsage = info->usage;
        gPerfMock->imageFormatQueryViewFormats.clear();
        for (const auto* next = static_cast<const VkBaseInStructure*>(info->pNext); next != nullptr; next = next->pNext)
        {
            if (next->sType == VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO)
            {
                const auto* list = reinterpret_cast<const VkImageFormatListCreateInfo*>(next);
                gPerfMock->imageFormatQueryViewFormats.assign(list->pViewFormats,
                                                              list->pViewFormats + list->viewFormatCount);
            }
        }
        if (gPerfMock->imageFormatQueryResult != VK_SUCCESS)
        {
            return gPerfMock->imageFormatQueryResult;
        }
        static_cast<void>(perfGetImageFormatProps(
            physical, info->format, info->type, info->tiling, info->usage, info->flags, &out->imageFormatProperties));
        out->imageFormatProperties.maxMipLevels = gPerfMock->imageFormatQueryMaxMipLevels;
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfAllocateMemory(VkDevice,
                                                      const VkMemoryAllocateInfo* info,
                                                      const VkAllocationCallbacks*,
                                                      VkDeviceMemory* out)
    {
        VkPhysicalDeviceMemoryProperties memProps{};
        perfGetPhysicalMemProps(VK_NULL_HANDLE, &memProps);
        if (info->memoryTypeIndex >= memProps.memoryTypeCount)
        {
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        const auto heapIndex = memProps.memoryTypes[info->memoryTypeIndex].heapIndex;
        if (heapIndex >= memProps.memoryHeapCount || info->allocationSize > memProps.memoryHeaps[heapIndex].size)
        {
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }

        const std::scoped_lock lock(gPerfMock->mutex);
        if (info->allocationSize > memProps.memoryHeaps[heapIndex].size - gPerfMock->usedHeapBytes[heapIndex])
        {
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        gPerfMock->usedHeapBytes[heapIndex] += info->allocationSize;
        const auto bytes = static_cast<std::size_t>(std::clamp<VkDeviceSize>(info->allocationSize, 256, 1ULL << 20));
        auto backing = std::make_unique<std::byte[]>(bytes);
        std::memset(backing.get(), 0, bytes);
        gPerfMock->hostAllocations.push_back(std::move(backing));
        gPerfMock->allocatedMemoryTypes.push_back(info->memoryTypeIndex);
        std::optional<float> priority;
        for (const auto* next = static_cast<const VkBaseInStructure*>(info->pNext); next != nullptr; next = next->pNext)
        {
            if (next->sType == VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT)
            {
                priority = reinterpret_cast<const VkMemoryPriorityAllocateInfoEXT*>(next)->priority;
            }
        }
        gPerfMock->allocatedMemoryPriorities.push_back(priority);
        gPerfMock->allocatedMemorySizes.push_back(info->allocationSize);
        const auto memory = fakeVk<VkDeviceMemory>(gPerfMock->hostAllocations.size());
        gPerfMock->memoryAllocations[memory] = {heapIndex, info->allocationSize};
        *out = memory;
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL perfFreeMemory(VkDevice, VkDeviceMemory memory, const VkAllocationCallbacks*)
    {
        if (memory == VK_NULL_HANDLE || gPerfMock == nullptr)
        {
            return;
        }
        const std::scoped_lock lock(gPerfMock->mutex);
        if (const auto it = gPerfMock->memoryAllocations.find(memory); it != gPerfMock->memoryAllocations.end())
        {
            const auto [heapIndex, allocationSize] = it->second;
            gPerfMock->usedHeapBytes[heapIndex] -= std::min(gPerfMock->usedHeapBytes[heapIndex], allocationSize);
            gPerfMock->memoryAllocations.erase(it);
        }
    }

    VKAPI_ATTR VkResult VKAPI_CALL
    perfMapMemory(VkDevice, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize, VkMemoryMapFlags, void** ppData)
    {
        gPerfMock->mapMemoryCalls.fetch_add(1, std::memory_order_relaxed);
        const std::scoped_lock lock(gPerfMock->mutex);
        const auto idx = reinterpret_cast<std::uintptr_t>(memory) - 1U;
        *ppData = gPerfMock->hostAllocations[idx].get() + offset;
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL perfUnmapMemory(VkDevice, VkDeviceMemory)
    {
        gPerfMock->unmapMemoryCalls.fetch_add(1, std::memory_order_relaxed);
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfFlushRanges(VkDevice, std::uint32_t count, const VkMappedMemoryRange* ranges)
    {
        gPerfMock->flushRangeCalls.fetch_add(count, std::memory_order_relaxed);
        if (count != 0)
        {
            gPerfMock->lastFlushSize.store(ranges[count - 1].size, std::memory_order_relaxed);
        }
        return gPerfMock->flushResult.load(std::memory_order_relaxed);
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfInvalidateRanges(VkDevice, std::uint32_t count, const VkMappedMemoryRange*)
    {
        gPerfMock->invalidateRangeCalls.fetch_add(count, std::memory_order_relaxed);
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfCreateBuffer(VkDevice,
                                                    const VkBufferCreateInfo* info,
                                                    const VkAllocationCallbacks*,
                                                    VkBuffer* out)
    {
        gPerfMock->createBufferCalls.fetch_add(1, std::memory_order_relaxed);
        const std::scoped_lock lock(gPerfMock->mutex);
        if (info != nullptr)
        {
            gPerfMock->bufferSharingModes.push_back(info->sharingMode);
            gPerfMock->bufferQueueFamilyCounts.push_back(info->queueFamilyIndexCount);
            std::vector<std::uint32_t> families;
            if (info->queueFamilyIndexCount > 0 && info->pQueueFamilyIndices != nullptr)
            {
                families.assign(info->pQueueFamilyIndices, info->pQueueFamilyIndices + info->queueFamilyIndexCount);
            }
            gPerfMock->bufferQueueFamilies.push_back(std::move(families));
        }
        const auto handle = fakeVk<VkBuffer>(gPerfMock->nextHandle++);
        gPerfMock->bufferSizes[handle] = info != nullptr ? info->size : 0;
        *out = handle;
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL perfDestroyBuffer(VkDevice, VkBuffer buffer, const VkAllocationCallbacks*)
    {
        if (buffer != VK_NULL_HANDLE)
        {
            gPerfMock->destroyBufferCalls.fetch_add(1, std::memory_order_relaxed);
            if (gPerfMock != nullptr)
            {
                const std::scoped_lock lock(gPerfMock->mutex);
                gPerfMock->bufferSizes.erase(buffer);
            }
        }
    }

    [[nodiscard]] VkDeviceSize alignedMockBufferBytes(VkDeviceSize requested) noexcept
    {
        const VkDeviceSize padded = requested > (std::numeric_limits<VkDeviceSize>::max() - 255ULL)
                                        ? requested
                                        : ((requested + 255ULL) & ~255ULL);
        return std::max<VkDeviceSize>(4096, padded);
    }

    VKAPI_ATTR void VKAPI_CALL perfGetBufferMemReqs(VkDevice, VkBuffer buffer, VkMemoryRequirements* out)
    {
        VkDeviceSize requested = 0;
        if (gPerfMock != nullptr)
        {
            const std::scoped_lock lock(gPerfMock->mutex);
            if (const auto it = gPerfMock->bufferSizes.find(buffer); it != gPerfMock->bufferSizes.end())
            {
                requested = it->second;
            }
        }
        *out =
            VkMemoryRequirements{.size = alignedMockBufferBytes(requested), .alignment = 256, .memoryTypeBits = 0xFU};
    }

    void reportMockDedicatedRequirements(VkMemoryRequirements2* out) noexcept
    {
        const auto required = gPerfMock->requireDedicatedAllocation ? VK_TRUE : VK_FALSE;
        for (auto* next = static_cast<VkBaseOutStructure*>(out->pNext); next != nullptr; next = next->pNext)
        {
            if (next->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS)
            {
                auto* dedicated = reinterpret_cast<VkMemoryDedicatedRequirements*>(next);
                dedicated->prefersDedicatedAllocation = required;
                dedicated->requiresDedicatedAllocation = required;
            }
        }
    }

    VKAPI_ATTR void VKAPI_CALL perfGetBufferMemReqs2(VkDevice,
                                                     const VkBufferMemoryRequirementsInfo2* info,
                                                     VkMemoryRequirements2* out)
    {
        gPerfMock->legacyBufferMemReq2Calls.fetch_add(1, std::memory_order_relaxed);
        reportMockDedicatedRequirements(out);
        VkDeviceSize requested = 0;
        if (gPerfMock != nullptr && info != nullptr)
        {
            const std::scoped_lock lock(gPerfMock->mutex);
            if (const auto it = gPerfMock->bufferSizes.find(info->buffer); it != gPerfMock->bufferSizes.end())
            {
                requested = it->second;
            }
        }
        out->memoryRequirements =
            VkMemoryRequirements{.size = alignedMockBufferBytes(requested), .alignment = 256, .memoryTypeBits = 0xFU};
    }

    VKAPI_ATTR void VKAPI_CALL perfGetDeviceBufferMemReqs(VkDevice,
                                                          const VkDeviceBufferMemoryRequirements* info,
                                                          VkMemoryRequirements2* out)
    {
        VkDeviceSize requested = 0;
        if (info != nullptr && info->pCreateInfo != nullptr)
        {
            requested = info->pCreateInfo->size;
            const std::scoped_lock lock(gPerfMock->mutex);
            gPerfMock->bufferSharingModes.push_back(info->pCreateInfo->sharingMode);
            gPerfMock->bufferQueueFamilyCounts.push_back(info->pCreateInfo->queueFamilyIndexCount);
            std::vector<std::uint32_t> families;
            if (info->pCreateInfo->queueFamilyIndexCount > 0 && info->pCreateInfo->pQueueFamilyIndices != nullptr)
            {
                families.assign(info->pCreateInfo->pQueueFamilyIndices,
                                info->pCreateInfo->pQueueFamilyIndices + info->pCreateInfo->queueFamilyIndexCount);
            }
            gPerfMock->bufferQueueFamilies.push_back(std::move(families));
        }
        out->memoryRequirements =
            VkMemoryRequirements{.size = alignedMockBufferBytes(requested), .alignment = 256, .memoryTypeBits = 0xFU};
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfBindBufferMemory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfBindBufferMemory2(VkDevice, std::uint32_t, const VkBindBufferMemoryInfo*)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfCreateImage(VkDevice,
                                                   const VkImageCreateInfo* info,
                                                   const VkAllocationCallbacks*,
                                                   VkImage* out)
    {
        gPerfMock->createImageCalls.fetch_add(1, std::memory_order_relaxed);
        const std::scoped_lock lock(gPerfMock->mutex);
        if (info != nullptr)
        {
            gPerfMock->imageSharingModes.push_back(info->sharingMode);
            gPerfMock->imageQueueFamilyCounts.push_back(info->queueFamilyIndexCount);
            std::vector<std::uint32_t> families;
            if (info->queueFamilyIndexCount > 0 && info->pQueueFamilyIndices != nullptr)
            {
                families.assign(info->pQueueFamilyIndices, info->pQueueFamilyIndices + info->queueFamilyIndexCount);
            }
            gPerfMock->imageQueueFamilies.push_back(std::move(families));
        }
        *out = fakeVk<VkImage>(gPerfMock->nextHandle++);
        gPerfMock->imageCreateFlags[*out] = info != nullptr ? info->flags : 0;
        return VK_SUCCESS;
    }

    [[nodiscard]] VkDeviceSize perfImageSize(VkImageCreateFlags flags) noexcept
    {
        return (flags & VK_IMAGE_CREATE_ALIAS_BIT) != 0 && gPerfMock->aliasImageSize != 0 ? gPerfMock->aliasImageSize
                                                                                          : 65536;
    }

    VKAPI_ATTR void VKAPI_CALL perfDestroyImage(VkDevice, VkImage image, const VkAllocationCallbacks*)
    {
        if (image != VK_NULL_HANDLE)
        {
            gPerfMock->destroyImageCalls.fetch_add(1, std::memory_order_relaxed);
        }
    }

    VKAPI_ATTR void VKAPI_CALL perfGetImageMemReqs(VkDevice, VkImage image, VkMemoryRequirements* out)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        const auto found = gPerfMock->imageCreateFlags.find(image);
        const auto flags = found != gPerfMock->imageCreateFlags.end() ? found->second : VkImageCreateFlags{0};
        *out = VkMemoryRequirements{.size = perfImageSize(flags), .alignment = 4096, .memoryTypeBits = 0x1U};
    }

    VKAPI_ATTR void VKAPI_CALL perfGetImageMemReqs2(VkDevice,
                                                    const VkImageMemoryRequirementsInfo2*,
                                                    VkMemoryRequirements2* out)
    {
        gPerfMock->legacyImageMemReq2Calls.fetch_add(1, std::memory_order_relaxed);
        reportMockDedicatedRequirements(out);
        out->memoryRequirements = VkMemoryRequirements{.size = 65536, .alignment = 4096, .memoryTypeBits = 0x1U};
    }

    VKAPI_ATTR void VKAPI_CALL perfGetDeviceImageMemReqs(VkDevice,
                                                         const VkDeviceImageMemoryRequirements* info,
                                                         VkMemoryRequirements2* out)
    {
        CHECK(info->planeAspect == VK_IMAGE_ASPECT_NONE);
        if (info != nullptr && info->pCreateInfo != nullptr)
        {
            const std::scoped_lock lock(gPerfMock->mutex);
            gPerfMock->imageSharingModes.push_back(info->pCreateInfo->sharingMode);
            gPerfMock->imageQueueFamilyCounts.push_back(info->pCreateInfo->queueFamilyIndexCount);
            std::vector<std::uint32_t> families;
            if (info->pCreateInfo->queueFamilyIndexCount > 0 && info->pCreateInfo->pQueueFamilyIndices != nullptr)
            {
                families.assign(info->pCreateInfo->pQueueFamilyIndices,
                                info->pCreateInfo->pQueueFamilyIndices + info->pCreateInfo->queueFamilyIndexCount);
            }
            gPerfMock->imageQueueFamilies.push_back(std::move(families));
        }
        const auto flags = info != nullptr && info->pCreateInfo != nullptr ? info->pCreateInfo->flags : 0;
        out->memoryRequirements = VkMemoryRequirements{
            .size = perfImageSize(flags),
            .alignment = 4096,
            .memoryTypeBits = 0x1U,
        };
        auto* const dedicated = static_cast<VkMemoryDedicatedRequirements*>(out->pNext);
        if (dedicated != nullptr && dedicated->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS)
        {
            const bool aliasDedicated =
                (flags & VK_IMAGE_CREATE_ALIAS_BIT) != 0 && gPerfMock->aliasImageRequiresDedicated;
            dedicated->requiresDedicatedAllocation = aliasDedicated ? VK_TRUE : VK_FALSE;
            dedicated->prefersDedicatedAllocation = aliasDedicated ? VK_TRUE : VK_FALSE;
        }
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfBindImageMemory(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfBindImageMemory2(VkDevice, std::uint32_t, const VkBindImageMemoryInfo*)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfCreateImageView(VkDevice,
                                                       const VkImageViewCreateInfo*,
                                                       const VkAllocationCallbacks*,
                                                       VkImageView* out)
    {
        gPerfMock->createImageViewCalls.fetch_add(1, std::memory_order_relaxed);
        const std::scoped_lock lock(gPerfMock->mutex);
        *out = fakeVk<VkImageView>(gPerfMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL perfDestroyImageView(VkDevice, VkImageView view, const VkAllocationCallbacks*)
    {
        if (view != VK_NULL_HANDLE)
        {
            gPerfMock->destroyImageViewCalls.fetch_add(1, std::memory_order_relaxed);
        }
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfCreateSampler(VkDevice,
                                                     const VkSamplerCreateInfo* info,
                                                     const VkAllocationCallbacks*,
                                                     VkSampler* out)
    {
        gPerfMock->createSamplerCalls.fetch_add(1, std::memory_order_relaxed);
        const std::scoped_lock lock(gPerfMock->mutex);
        gPerfMock->samplerMipLodBiases.push_back(info->mipLodBias);
        *out = fakeVk<VkSampler>(gPerfMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL perfDestroySampler(VkDevice, VkSampler sampler, const VkAllocationCallbacks*)
    {
        if (sampler != VK_NULL_HANDLE)
        {
            gPerfMock->destroySamplerCalls.fetch_add(1, std::memory_order_relaxed);
        }
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfCreateShaderModule(VkDevice,
                                                          const VkShaderModuleCreateInfo*,
                                                          const VkAllocationCallbacks*,
                                                          VkShaderModule* out)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        *out = fakeVk<VkShaderModule>(gPerfMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL perfDestroyShaderModule(VkDevice, VkShaderModule, const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfCreateDescSetLayout(VkDevice,
                                                           const VkDescriptorSetLayoutCreateInfo*,
                                                           const VkAllocationCallbacks*,
                                                           VkDescriptorSetLayout* out)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        *out = fakeVk<VkDescriptorSetLayout>(gPerfMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL perfDestroyDescSetLayout(VkDevice, VkDescriptorSetLayout, const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfCreatePipelineLayout(VkDevice,
                                                            const VkPipelineLayoutCreateInfo*,
                                                            const VkAllocationCallbacks*,
                                                            VkPipelineLayout* out)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        *out = fakeVk<VkPipelineLayout>(gPerfMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL perfDestroyPipelineLayout(VkDevice, VkPipelineLayout, const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfCreateDescPool(VkDevice,
                                                      const VkDescriptorPoolCreateInfo* info,
                                                      const VkAllocationCallbacks*,
                                                      VkDescriptorPool* out)
    {
        if ((info->flags & VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT) != 0 &&
            gPerfMock->updateAfterBindDescPoolResult != VK_SUCCESS)
        {
            return gPerfMock->updateAfterBindDescPoolResult;
        }
        gPerfMock->createdDescPools.fetch_add(1, std::memory_order_relaxed);
        const std::scoped_lock lock(gPerfMock->mutex);
        *out = fakeVk<VkDescriptorPool>(gPerfMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL perfDestroyDescPool(VkDevice, VkDescriptorPool pool, const VkAllocationCallbacks*)
    {
        if (pool != VK_NULL_HANDLE)
        {
            gPerfMock->destroyedDescPools.fetch_add(1, std::memory_order_relaxed);
        }
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfAllocDescSets(VkDevice,
                                                     const VkDescriptorSetAllocateInfo* info,
                                                     VkDescriptorSet* out)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        for (std::uint32_t i = 0; i < info->descriptorSetCount; ++i)
        {
            out[i] = fakeVk<VkDescriptorSet>(gPerfMock->nextHandle++);
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL
    perfUpdateDescSets(VkDevice, std::uint32_t, const VkWriteDescriptorSet*, std::uint32_t, const VkCopyDescriptorSet*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfCreateComputePipelines(VkDevice,
                                                              VkPipelineCache,
                                                              std::uint32_t count,
                                                              const VkComputePipelineCreateInfo*,
                                                              const VkAllocationCallbacks*,
                                                              VkPipeline* out)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        for (std::uint32_t i = 0; i < count; ++i)
        {
            out[i] = fakeVk<VkPipeline>(gPerfMock->nextHandle++);
            gPerfMock->createdPipelines.fetch_add(1, std::memory_order_relaxed);
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfCreateGraphicsPipelines(VkDevice,
                                                               VkPipelineCache,
                                                               std::uint32_t count,
                                                               const VkGraphicsPipelineCreateInfo*,
                                                               const VkAllocationCallbacks*,
                                                               VkPipeline* out)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        for (std::uint32_t i = 0; i < count; ++i)
        {
            out[i] = fakeVk<VkPipeline>(gPerfMock->nextHandle++);
            gPerfMock->createdPipelines.fetch_add(1, std::memory_order_relaxed);
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL perfDestroyPipeline(VkDevice, VkPipeline pipeline, const VkAllocationCallbacks*)
    {
        if (pipeline != VK_NULL_HANDLE)
        {
            gPerfMock->destroyedPipelines.fetch_add(1, std::memory_order_relaxed);
        }
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfCreateCommandPool(VkDevice,
                                                         const VkCommandPoolCreateInfo*,
                                                         const VkAllocationCallbacks*,
                                                         VkCommandPool* out)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        *out = fakeVk<VkCommandPool>(gPerfMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL perfDestroyCommandPool(VkDevice, VkCommandPool, const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfAllocCmdBuffers(VkDevice,
                                                       const VkCommandBufferAllocateInfo* info,
                                                       VkCommandBuffer* out)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        for (std::uint32_t i = 0; i < info->commandBufferCount; ++i)
        {
            out[i] = fakeVk<VkCommandBuffer>(gPerfMock->nextHandle++);
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL perfFreeCmdBuffers(VkDevice, VkCommandPool, std::uint32_t, const VkCommandBuffer*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfBeginCmdBuffer(VkCommandBuffer, const VkCommandBufferBeginInfo*)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfEndCmdBuffer(VkCommandBuffer)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfResetCmdBuffer(VkCommandBuffer, VkCommandBufferResetFlags)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfResetCmdPool(VkDevice, VkCommandPool, VkCommandPoolResetFlags)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL
    perfCmdCopyBuffer(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*)
    {
    }

    VKAPI_ATTR void VKAPI_CALL perfCmdBindPipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline)
    {
        gPerfMock->boundPipelines.fetch_add(1, std::memory_order_relaxed);
    }

    VKAPI_ATTR void VKAPI_CALL perfCmdBindDescSets(VkCommandBuffer,
                                                   VkPipelineBindPoint,
                                                   VkPipelineLayout,
                                                   std::uint32_t,
                                                   std::uint32_t,
                                                   const VkDescriptorSet*,
                                                   std::uint32_t,
                                                   const std::uint32_t*)
    {
        gPerfMock->boundDescriptorSetCalls.fetch_add(1, std::memory_order_relaxed);
    }

    VKAPI_ATTR void VKAPI_CALL perfCmdPipelineBarrier2(VkCommandBuffer, const VkDependencyInfo* dep)
    {
        if (dep == nullptr)
        {
            return;
        }
        const std::scoped_lock lock(gPerfMock->mutex);
        for (std::uint32_t i = 0; i < dep->memoryBarrierCount; ++i)
        {
            const auto& barrier = dep->pMemoryBarriers[i];
            gPerfMock->commandEvents.push_back({
                .kind = PerfCommandKind::memoryBarrier,
                .srcStageMask = barrier.srcStageMask,
                .srcAccessMask = barrier.srcAccessMask,
                .dstStageMask = barrier.dstStageMask,
                .dstAccessMask = barrier.dstAccessMask,
            });
        }
        for (std::uint32_t i = 0; i < dep->bufferMemoryBarrierCount; ++i)
        {
            const auto& barrier = dep->pBufferMemoryBarriers[i];
            gPerfMock->capturedBufferBarriers.push_back(barrier);
            gPerfMock->commandEvents.push_back({
                .kind = PerfCommandKind::bufferBarrier,
                .buffer = barrier.buffer,
                .offset = barrier.offset,
                .size = barrier.size,
                .srcStageMask = barrier.srcStageMask,
                .srcAccessMask = barrier.srcAccessMask,
                .dstStageMask = barrier.dstStageMask,
                .dstAccessMask = barrier.dstAccessMask,
            });
        }
        for (std::uint32_t i = 0; i < dep->imageMemoryBarrierCount; ++i)
        {
            gPerfMock->capturedImageBarriers.push_back(dep->pImageMemoryBarriers[i]);
        }
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfCreateQueryPool(VkDevice,
                                                       const VkQueryPoolCreateInfo*,
                                                       const VkAllocationCallbacks*,
                                                       VkQueryPool* out)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        *out = fakeVk<VkQueryPool>(gPerfMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL perfDestroyQueryPool(VkDevice, VkQueryPool, const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR void VKAPI_CALL perfResetQueryPool(VkDevice, VkQueryPool, std::uint32_t, std::uint32_t)
    {
    }

    VKAPI_ATTR void VKAPI_CALL perfCmdResetQueryPool(VkCommandBuffer, VkQueryPool pool, std::uint32_t, std::uint32_t)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        gPerfMock->commandEvents.push_back({.kind = PerfCommandKind::resetQueryPool, .queryPool = pool});
    }

    VKAPI_ATTR void VKAPI_CALL perfCmdBeginQuery(VkCommandBuffer, VkQueryPool, std::uint32_t, VkQueryControlFlags)
    {
    }

    VKAPI_ATTR void VKAPI_CALL perfCmdEndQuery(VkCommandBuffer, VkQueryPool, std::uint32_t)
    {
    }

    VKAPI_ATTR void VKAPI_CALL perfCmdWriteTimestamp2(VkCommandBuffer,
                                                      VkPipelineStageFlags2,
                                                      VkQueryPool,
                                                      std::uint32_t)
    {
    }

    VKAPI_ATTR void VKAPI_CALL perfCmdCopyQueryPoolResults(VkCommandBuffer,
                                                           VkQueryPool pool,
                                                           std::uint32_t,
                                                           std::uint32_t,
                                                           VkBuffer buffer,
                                                           VkDeviceSize offset,
                                                           VkDeviceSize stride,
                                                           VkQueryResultFlags flags)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        gPerfMock->commandEvents.push_back({
            .kind = PerfCommandKind::copyQueryResults,
            .queryPool = pool,
            .buffer = buffer,
            .offset = offset,
            .stride = stride,
            .resultFlags = flags,
        });
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfCreateSemaphore(VkDevice,
                                                       const VkSemaphoreCreateInfo* info,
                                                       const VkAllocationCallbacks*,
                                                       VkSemaphore* out)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        const auto sem = fakeVk<VkSemaphore>(gPerfMock->nextHandle++);
        std::uint64_t initial = 0;
        const auto* cursor = static_cast<const VkBaseInStructure*>(info != nullptr ? info->pNext : nullptr);
        while (cursor != nullptr)
        {
            if (cursor->sType == VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO)
            {
                initial = reinterpret_cast<const VkSemaphoreTypeCreateInfo*>(cursor)->initialValue;
            }
            cursor = cursor->pNext;
        }
        gPerfMock->timelineValues[sem] = initial;
        *out = sem;
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL perfDestroySemaphore(VkDevice, VkSemaphore sem, const VkAllocationCallbacks*)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        gPerfMock->timelineValues.erase(sem);
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfGetSemaphoreCounter(VkDevice, VkSemaphore sem, std::uint64_t* value)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        const auto it = gPerfMock->timelineValues.find(sem);
        *value = it != gPerfMock->timelineValues.end() ? it->second : 0;
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfWaitSemaphores(VkDevice, const VkSemaphoreWaitInfo* info, std::uint64_t)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        for (std::uint32_t i = 0; i < info->semaphoreCount; ++i)
        {
            auto& current = gPerfMock->timelineValues[info->pSemaphores[i]];
            current = std::max(current, info->pValues[i]);
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfSignalSemaphore(VkDevice, const VkSemaphoreSignalInfo* info)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        auto& current = gPerfMock->timelineValues[info->semaphore];
        current = std::max(current, info->value);
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfCreateFence(VkDevice,
                                                   const VkFenceCreateInfo*,
                                                   const VkAllocationCallbacks*,
                                                   VkFence* out)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        *out = fakeVk<VkFence>(gPerfMock->nextHandle++);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL perfDestroyFence(VkDevice, VkFence, const VkAllocationCallbacks*)
    {
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfWaitForFences(VkDevice, std::uint32_t, const VkFence*, VkBool32, std::uint64_t)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfResetFences(VkDevice, std::uint32_t, const VkFence*)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfQueueSubmit2(VkQueue,
                                                    std::uint32_t submitCount,
                                                    const VkSubmitInfo2* submits,
                                                    VkFence)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        for (std::uint32_t s = 0; s < submitCount; ++s)
        {
            for (std::uint32_t i = 0; i < submits[s].signalSemaphoreInfoCount; ++i)
            {
                const auto& sig = submits[s].pSignalSemaphoreInfos[i];
                auto& current = gPerfMock->timelineValues[sig.semaphore];
                current = std::max(current, sig.value);
            }
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfQueueWaitIdle(VkQueue)
    {
        return gPerfMock->queueWaitIdleResult.load(std::memory_order_relaxed);
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfDeviceWaitIdle(VkDevice)
    {
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfGetSurfaceCapabilities(VkPhysicalDevice,
                                                              VkSurfaceKHR,
                                                              VkSurfaceCapabilitiesKHR* out)
    {
        gPerfMock->surfaceCapsQueries.fetch_add(1, std::memory_order_relaxed);
        const std::scoped_lock lock(gPerfMock->mutex);
        *out = VkSurfaceCapabilitiesKHR{
            .minImageCount = 2,
            .maxImageCount = 4,
            .currentExtent = gPerfMock->surfaceCurrentExtent,
            .minImageExtent = {1, 1},
            .maxImageExtent = {4096, 4096},
            .maxImageArrayLayers = 1,
            .supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
            .currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
            .supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .supportedUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        };
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfGetSurfaceFormats(VkPhysicalDevice,
                                                         VkSurfaceKHR,
                                                         std::uint32_t* count,
                                                         VkSurfaceFormatKHR* formats)
    {
        if (formats == nullptr)
        {
            *count = 1;
            return VK_SUCCESS;
        }
        if (*count >= 1)
        {
            formats[0] = VkSurfaceFormatKHR{
                .format = VK_FORMAT_R8G8B8A8_UNORM,
                .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
            };
            *count = 1;
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfGetSurfacePresentModes(VkPhysicalDevice,
                                                              VkSurfaceKHR,
                                                              std::uint32_t* count,
                                                              VkPresentModeKHR* modes)
    {
        if (modes == nullptr)
        {
            *count = 1;
            return VK_SUCCESS;
        }
        if (*count >= 1)
        {
            modes[0] = VK_PRESENT_MODE_FIFO_KHR;
            *count = 1;
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfCreateSwapchain(VkDevice,
                                                       const VkSwapchainCreateInfoKHR* info,
                                                       const VkAllocationCallbacks*,
                                                       VkSwapchainKHR* out)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        gPerfMock->oldSwapchains.push_back(info->oldSwapchain);
        const auto result = gPerfMock->nextCreateSwapchainResult;
        gPerfMock->nextCreateSwapchainResult = VK_SUCCESS;
        if (result != VK_SUCCESS)
        {
            return result;
        }
        *out = fakeVk<VkSwapchainKHR>(gPerfMock->nextHandle++);
        gPerfMock->createdSwapchains.push_back(*out);
        return VK_SUCCESS;
    }

    VKAPI_ATTR void VKAPI_CALL perfDestroySwapchain(VkDevice, VkSwapchainKHR swapchain, const VkAllocationCallbacks*)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        gPerfMock->destroyedSwapchains.push_back(swapchain);
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfGetSwapchainImages(VkDevice,
                                                          VkSwapchainKHR,
                                                          std::uint32_t* count,
                                                          VkImage* images)
    {
        if (images == nullptr)
        {
            *count = 2;
            return VK_SUCCESS;
        }
        const std::scoped_lock lock(gPerfMock->mutex);
        for (std::uint32_t i = 0; i < *count; ++i)
        {
            images[i] = fakeVk<VkImage>(0x7000U + i);
        }
        return VK_SUCCESS;
    }

    VKAPI_ATTR VkResult VKAPI_CALL
    perfAcquireNextImage(VkDevice, VkSwapchainKHR, std::uint64_t, VkSemaphore, VkFence, std::uint32_t* index)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        const auto result = gPerfMock->nextAcquireNativeResult;
        gPerfMock->nextAcquireNativeResult = VK_SUCCESS;
        if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR)
        {
            *index = gPerfMock->nextAcquireImageIndex;
            gPerfMock->nextAcquireImageIndex = (gPerfMock->nextAcquireImageIndex + 1U) % 2U;
        }
        return result;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfQueuePresent(VkQueue, const VkPresentInfoKHR*)
    {
        gPerfMock->queuePresentCalls.fetch_add(1, std::memory_order_relaxed);
        const std::scoped_lock lock(gPerfMock->mutex);
        const auto result = gPerfMock->nextPresentResult;
        gPerfMock->nextPresentResult = VK_SUCCESS;
        return result;
    }

    VKAPI_ATTR VkResult VKAPI_CALL perfWaitForPresent(VkDevice,
                                                      VkSwapchainKHR,
                                                      std::uint64_t presentId,
                                                      std::uint64_t timeout)
    {
        const std::scoped_lock lock(gPerfMock->mutex);
        gPerfMock->waitedPresentIds.push_back(presentId);
        gPerfMock->waitedPresentTimeouts.push_back(timeout);
        return gPerfMock->waitForPresentResult;
    }

    void populatePerfTables(rachis::vulkan::VulkanInstanceTable& instanceTable, VolkDeviceTable& table)
    {
        instanceTable = rachis::vulkan::VulkanInstanceTable{};
        instanceTable.vkGetInstanceProcAddr = &perfGetInstanceProcAddr;
        instanceTable.vkGetDeviceProcAddr = &perfGetDeviceProcAddr;
        instanceTable.vkGetPhysicalDeviceProperties = &perfGetPhysicalProps;
        instanceTable.vkGetPhysicalDeviceProperties2 = &perfGetPhysicalProps2;
        instanceTable.vkGetPhysicalDeviceMemoryProperties = &perfGetPhysicalMemProps;
        instanceTable.vkGetPhysicalDeviceMemoryProperties2 = &perfGetPhysicalMemProps2;
        instanceTable.vkGetPhysicalDeviceFormatProperties = &perfGetFormatProps;
        instanceTable.vkGetPhysicalDeviceImageFormatProperties = &perfGetImageFormatProps;
        instanceTable.vkGetPhysicalDeviceSurfaceCapabilitiesKHR = &perfGetSurfaceCapabilities;
        instanceTable.vkGetPhysicalDeviceSurfaceFormatsKHR = &perfGetSurfaceFormats;
        instanceTable.vkGetPhysicalDeviceSurfacePresentModesKHR = &perfGetSurfacePresentModes;

        table = VolkDeviceTable{};
        table.vkAllocateMemory = &perfAllocateMemory;
        table.vkFreeMemory = &perfFreeMemory;
        table.vkMapMemory = &perfMapMemory;
        table.vkUnmapMemory = &perfUnmapMemory;
        table.vkFlushMappedMemoryRanges = &perfFlushRanges;
        table.vkInvalidateMappedMemoryRanges = &perfInvalidateRanges;
        table.vkCreateBuffer = &perfCreateBuffer;
        table.vkDestroyBuffer = &perfDestroyBuffer;
        table.vkGetBufferMemoryRequirements = &perfGetBufferMemReqs;
        table.vkGetBufferMemoryRequirements2 = &perfGetBufferMemReqs2;
        table.vkGetDeviceBufferMemoryRequirements = &perfGetDeviceBufferMemReqs;
        table.vkBindBufferMemory = &perfBindBufferMemory;
        table.vkBindBufferMemory2 = &perfBindBufferMemory2;
        table.vkCreateImage = &perfCreateImage;
        table.vkDestroyImage = &perfDestroyImage;
        table.vkGetImageMemoryRequirements = &perfGetImageMemReqs;
        table.vkGetImageMemoryRequirements2 = &perfGetImageMemReqs2;
        table.vkGetDeviceImageMemoryRequirements = &perfGetDeviceImageMemReqs;
        table.vkBindImageMemory = &perfBindImageMemory;
        table.vkBindImageMemory2 = &perfBindImageMemory2;
        table.vkCreateImageView = &perfCreateImageView;
        table.vkDestroyImageView = &perfDestroyImageView;
        table.vkCreateSampler = &perfCreateSampler;
        table.vkDestroySampler = &perfDestroySampler;
        table.vkCreateShaderModule = &perfCreateShaderModule;
        table.vkDestroyShaderModule = &perfDestroyShaderModule;
        table.vkCreateDescriptorSetLayout = &perfCreateDescSetLayout;
        table.vkDestroyDescriptorSetLayout = &perfDestroyDescSetLayout;
        table.vkCreatePipelineLayout = &perfCreatePipelineLayout;
        table.vkDestroyPipelineLayout = &perfDestroyPipelineLayout;
        table.vkCreateDescriptorPool = &perfCreateDescPool;
        table.vkDestroyDescriptorPool = &perfDestroyDescPool;
        table.vkAllocateDescriptorSets = &perfAllocDescSets;
        table.vkUpdateDescriptorSets = &perfUpdateDescSets;
        table.vkCreateComputePipelines = &perfCreateComputePipelines;
        table.vkCreateGraphicsPipelines = &perfCreateGraphicsPipelines;
        table.vkDestroyPipeline = &perfDestroyPipeline;
        table.vkCreateCommandPool = &perfCreateCommandPool;
        table.vkDestroyCommandPool = &perfDestroyCommandPool;
        table.vkAllocateCommandBuffers = &perfAllocCmdBuffers;
        table.vkFreeCommandBuffers = &perfFreeCmdBuffers;
        table.vkBeginCommandBuffer = &perfBeginCmdBuffer;
        table.vkEndCommandBuffer = &perfEndCmdBuffer;
        table.vkResetCommandBuffer = &perfResetCmdBuffer;
        table.vkResetCommandPool = &perfResetCmdPool;
        table.vkCmdCopyBuffer = &perfCmdCopyBuffer;
        table.vkCmdBindPipeline = &perfCmdBindPipeline;
        table.vkCmdBindDescriptorSets = &perfCmdBindDescSets;
        table.vkCmdPipelineBarrier2 = &perfCmdPipelineBarrier2;
        table.vkCreateQueryPool = &perfCreateQueryPool;
        table.vkDestroyQueryPool = &perfDestroyQueryPool;
        table.vkResetQueryPool = &perfResetQueryPool;
        table.vkCmdResetQueryPool = &perfCmdResetQueryPool;
        table.vkCmdBeginQuery = &perfCmdBeginQuery;
        table.vkCmdEndQuery = &perfCmdEndQuery;
        table.vkCmdWriteTimestamp2 = &perfCmdWriteTimestamp2;
        table.vkCmdCopyQueryPoolResults = &perfCmdCopyQueryPoolResults;
        table.vkCreateSemaphore = &perfCreateSemaphore;
        table.vkDestroySemaphore = &perfDestroySemaphore;
        table.vkGetSemaphoreCounterValue = &perfGetSemaphoreCounter;
        table.vkWaitSemaphores = &perfWaitSemaphores;
        table.vkSignalSemaphore = &perfSignalSemaphore;
        table.vkCreateFence = &perfCreateFence;
        table.vkDestroyFence = &perfDestroyFence;
        table.vkWaitForFences = &perfWaitForFences;
        table.vkResetFences = &perfResetFences;
        table.vkQueueSubmit2 = &perfQueueSubmit2;
        table.vkQueueWaitIdle = &perfQueueWaitIdle;
        table.vkDeviceWaitIdle = &perfDeviceWaitIdle;
        table.vkCreateSwapchainKHR = &perfCreateSwapchain;
        table.vkDestroySwapchainKHR = &perfDestroySwapchain;
        table.vkGetSwapchainImagesKHR = &perfGetSwapchainImages;
        table.vkAcquireNextImageKHR = &perfAcquireNextImage;
        table.vkQueuePresentKHR = &perfQueuePresent;
        table.vkWaitForPresentKHR = &perfWaitForPresent;
    }

    struct ScopedPerfMock final
    {
        PerfMockState state{};

        explicit ScopedPerfMock(MemoryTopologyKind topology = MemoryTopologyKind::coherentHostVisible)
        {
            state.memoryTopology = topology;
            gPerfMock = &state;
        }

        ~ScopedPerfMock()
        {
            gPerfMock = nullptr;
        }

        ScopedPerfMock(const ScopedPerfMock&) = delete;
        ScopedPerfMock& operator=(const ScopedPerfMock&) = delete;
        ScopedPerfMock(ScopedPerfMock&&) = delete;
        ScopedPerfMock& operator=(ScopedPerfMock&&) = delete;
    };

} // namespace

namespace rachis::vulkan
{

    struct VulkanPerformanceTestAccess final
    {
        [[nodiscard]] static std::uint32_t heapMemoryTypeIndex(const VulkanResourceAllocator& resources,
                                                               ResourceHeapHandle heap)
        {
            const auto* found = resources.heaps_.find(heap);
            REQUIRE(found != nullptr);
            REQUIRE(*found != nullptr);
            return (*found)->memoryTypeIndex;
        }

        [[nodiscard]] static std::uint32_t bufferMemoryTypeIndex(const VulkanResourceAllocator& resources,
                                                                 BufferHandle buffer)
        {
            const auto* found = resources.buffers_.find(buffer);
            REQUIRE(found != nullptr);
            return found->allocationInfo.memoryType;
        }

        [[nodiscard]] static Result<std::uint32_t> selectMemoryType(const VulkanResourceAllocator& resources,
                                                                    std::uint32_t typeBits,
                                                                    MemoryType memoryType,
                                                                    std::uint64_t requiredSize = 0)
        {
            return resources.findMemoryType(typeBits, memoryType, requiredSize);
        }

        struct StoredLabel final
        {
            std::string_view owned;
            std::string_view described;
        };

        [[nodiscard]] static StoredLabel bufferLabel(const VulkanResourceAllocator& resources, BufferHandle buffer)
        {
            const auto* found = resources.buffers_.find(buffer);
            REQUIRE(found != nullptr);
            return StoredLabel{.owned = found->label, .described = found->desc.label};
        }

        [[nodiscard]] static StoredLabel textureLabel(const VulkanResourceAllocator& resources, TextureHandle texture)
        {
            const auto* found = resources.textures_.find(texture);
            REQUIRE(found != nullptr);
            return StoredLabel{.owned = found->label, .described = found->desc.label};
        }

        [[nodiscard]] static StoredLabel textureViewLabel(const VulkanResourceAllocator& resources,
                                                          TextureViewHandle view)
        {
            const auto* found = resources.views_.find(view);
            REQUIRE(found != nullptr);
            return StoredLabel{.owned = found->label, .described = found->desc.label};
        }

        [[nodiscard]] static StoredLabel samplerLabel(const VulkanResourceAllocator& resources, SamplerHandle sampler)
        {
            const auto* found = resources.samplers_.find(sampler);
            REQUIRE(found != nullptr);
            return StoredLabel{.owned = found->label, .described = found->desc.label};
        }

        [[nodiscard]] static StoredLabel heapLabel(const VulkanResourceAllocator& resources, ResourceHeapHandle heap)
        {
            const auto* found = resources.heaps_.find(heap);
            REQUIRE(found != nullptr);
            REQUIRE(*found != nullptr);
            return StoredLabel{.owned = (*found)->label, .described = (*found)->desc.label};
        }
    };

    struct VulkanPerformanceSwapchainTestAccess final
    {
        VulkanInstanceTable instanceTable{};
        VolkDeviceTable table{};
        VulkanResourceAllocator resources{};
        VulkanCommandAllocator commands{};
        VulkanQueueManager queues{};
        VulkanSwapchainManager swapchains{};
        CommandPoolHandle commandPool{};
        SurfaceHandle surfaceHandle{};

        explicit VulkanPerformanceSwapchainTestAccess(
            SurfaceKind surfaceKind = SurfaceKind::wayland,
            VulkanPresentCompletionStrategy presentStrategy = VulkanPresentCompletionStrategy::imageLifetimeAssociation,
            bool khrSwapchainMaintenance1 = false,
            bool presentWaitEnabled = false)
        {
            populatePerfTables(instanceTable, table);
            const auto instance = fakeVk<VkInstance>(0x210);
            const auto physical = fakeVk<VkPhysicalDevice>(0x211);
            const auto device = fakeVk<VkDevice>(0x212);
            const VulkanQueueHandles queueHandles{
                .graphics = fakeVk<VkQueue>(0x213),
                .compute = fakeVk<VkQueue>(0x213),
                .copy = fakeVk<VkQueue>(0x213),
                .present = fakeVk<VkQueue>(0x214),
                .families = {.graphics = 0, .compute = 0, .transfer = 0},
                .presentCompletion = presentStrategy,
            };
            VkPhysicalDeviceMemoryProperties memProps{};
            perfGetPhysicalMemProps(physical, &memProps);
            REQUIRE(resources
                        .initialize(instance,
                                    &instanceTable,
                                    physical,
                                    device,
                                    &table,
                                    memProps,
                                    queueHandles.families,
                                    queueHandles.copy,
                                    64,
                                    false)
                        .has_value());
            constexpr std::uint64_t kToken = 0x77CC;
            commands.initialize(device, &table, queueHandles, kToken, &resources, &instanceTable);
            REQUIRE(queues.initialize(device, &table, queueHandles, kToken, false, &instanceTable).has_value());
            const auto pool = commands.createCommandPool({.queueType = QueueType::graphics, .maxCommandBuffers = 32});
            REQUIRE(pool.has_value());
            commandPool = *pool;

            swapchains.initialize(instance,
                                  &instanceTable,
                                  physical,
                                  device,
                                  &table,
                                  &resources,
                                  &queues,
                                  0,
                                  queueHandles.graphics,
                                  presentStrategy,
                                  khrSwapchainMaintenance1,
                                  presentWaitEnabled);

            auto surfaceRecord = std::make_shared<VulkanSwapchainManager::SurfaceRecord>();
            surfaceRecord->surface = fakeVk<VkSurfaceKHR>(0x220);
            surfaceRecord->kind = surfaceKind;
            const auto published = swapchains.surfaces_.emplace(std::move(surfaceRecord));
            REQUIRE(published.has_value());
            surfaceHandle = *published;
        }

        ~VulkanPerformanceSwapchainTestAccess()
        {
            swapchains.shutdown();
            commands.shutdown();
            queues.shutdown();
            resources.shutdown();
        }

        VulkanPerformanceSwapchainTestAccess(const VulkanPerformanceSwapchainTestAccess&) = delete;
        VulkanPerformanceSwapchainTestAccess& operator=(const VulkanPerformanceSwapchainTestAccess&) = delete;
        VulkanPerformanceSwapchainTestAccess(VulkanPerformanceSwapchainTestAccess&&) = delete;
        VulkanPerformanceSwapchainTestAccess& operator=(VulkanPerformanceSwapchainTestAccess&&) = delete;

        [[nodiscard]] TimelinePoint submitEmptyGraphicsWork()
        {
            queues.collectGarbage();
            auto cb = commands.acquireCommandBuffer(commandPool);
            REQUIRE(cb.has_value());
            std::array batch{std::move(*cb)};
            const auto point = queues.submit(batch, {});
            REQUIRE(point.has_value());
            return *point;
        }
    };

} // namespace rachis::vulkan

TEST_CASE("VulkanPipelineManager reuses immutable nativePipeline records in O(1) and retains lifetime per bind",
          "[size:small][rhi][vulkan]")
{
    ScopedPerfMock mock;
    rachis::vulkan::VulkanInstanceTable instanceTable{};
    VolkDeviceTable table{};
    populatePerfTables(instanceTable, table);

    const auto device = fakeVk<VkDevice>(0x301);
    const auto physical = fakeVk<VkPhysicalDevice>(0x302);
    const auto caps = baselinePerfCapabilities();

    rachis::vulkan::VulkanShaderManager shaders;
    shaders.initialize(device, &table, caps, &instanceTable);
    rachis::vulkan::VulkanBindingManager bindings;
    bindings.initialize(device, &table, caps, &instanceTable);
    rachis::vulkan::VulkanPipelineManager pipelines;
    pipelines.initialize(device, &table, physical, caps, &shaders, &bindings, &instanceTable, true);
    rachis::vulkan::VulkanGraphicsEncoder graphics;
    graphics.initialize(device, &table, &caps, nullptr, &pipelines, &bindings, true);

    const auto gfxSpv = makeMinimalGraphicsSpirv();
    const auto csSpv = makeMinimalComputeSpirv();
    const auto vs = shaders.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = gfxSpv.bytes(),
        .entryPoint = "vsMain",
    });
    const auto ps = shaders.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = gfxSpv.bytes(),
        .entryPoint = "psMain",
    });
    const auto cs = shaders.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = csSpv.bytes(),
        .entryPoint = "main",
        .shaderInterface = {.workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1}},
    });
    REQUIRE(vs.has_value());
    REQUIRE(ps.has_value());
    REQUIRE(cs.has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{
            .binding = 0,
            .type = rachis::BindingType::uniformBuffer,
            .visibility = rachis::ShaderStage::vertex | rachis::ShaderStage::pixel,
        },
    };
    const auto setLayout = bindings.createBindingLayout({.entries = bindingEntries});
    REQUIRE(setLayout.has_value());
    const std::array setLayouts{*setLayout};
    const auto pipeLayout = bindings.createPipelineLayout({
        .bindingLayouts = setLayouts,
    });
    const auto emptyLayout = bindings.createPipelineLayout({});
    REQUIRE(pipeLayout.has_value());
    REQUIRE(emptyLayout.has_value());

    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    const auto gfxPipe = pipelines.createGraphicsPipeline({
        .layout = *pipeLayout,
        .vertexShader = *vs,
        .pixelShader = *ps,
        .colorTargets = colorTargets,
    });
    const auto compPipe = pipelines.createComputePipeline({
        .layout = *emptyLayout,
        .computeShader = *cs,
    });
    REQUIRE(gfxPipe.has_value());
    REQUIRE(compPipe.has_value());

    const auto firstLookup = pipelines.nativePipeline(*gfxPipe);
    REQUIRE(firstLookup.has_value());
    REQUIRE(*firstLookup != nullptr);

    std::vector<std::thread> readers;
    std::atomic<std::uint32_t> mismatches{0};
    for (int t = 0; t < 4; ++t)
    {
        readers.emplace_back([&]() {
            for (int i = 0; i < 64; ++i)
            {
                const auto resolved = pipelines.nativePipeline(*gfxPipe);
                if (!resolved.has_value() || resolved->get() != firstLookup->get() ||
                    (*resolved)->colorFormats.data() != (*firstLookup)->colorFormats.data() ||
                    (*resolved)->setLayouts.data() != (*firstLookup)->setLayouts.data())
                {
                    mismatches.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& reader : readers)
    {
        reader.join();
    }
    CHECK(mismatches.load() == 0);

    const rachis::vulkan::VulkanQueueHandles queueHandles{
        .graphics = fakeVk<VkQueue>(0x408),
        .compute = fakeVk<VkQueue>(0x408),
        .copy = fakeVk<VkQueue>(0x408),
        .families = {.graphics = 0, .compute = 0, .transfer = 0},
    };
    rachis::vulkan::VulkanCommandAllocator commands;
    commands.initialize(device, &table, queueHandles, 0x801U, nullptr, &instanceTable);
    const auto cmdPool = commands.createCommandPool({.queueType = rachis::QueueType::graphics, .maxCommandBuffers = 2});
    REQUIRE(cmdPool.has_value());
    auto cb = commands.acquireCommandBuffer(*cmdPool);
    REQUIRE(cb.has_value());
    auto& cmdBuf = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb);
    auto scope = std::make_shared<rachis::vulkan::VulkanRenderingScope>();
    scope->colorFormats = {rachis::Format::rgba8Unorm};
    scope->sampleCount = 1;
    cmdBuf.graphicsState().renderingOwner = scope;

    REQUIRE(graphics.bindGraphicsPipeline(cmdBuf, *gfxPipe).has_value());
    CHECK(cmdBuf.retained().size() == 1U);
    CHECK(mock.state.boundPipelines.load() == 1U);

    const auto* rawPtr = firstLookup->get();
    const auto compLookup = pipelines.nativePipeline(*compPipe);
    REQUIRE(compLookup.has_value());
    REQUIRE(pipelines.destroyPipeline(*gfxPipe).has_value());
    REQUIRE(pipelines.destroyPipeline(*compPipe).has_value());

    CHECK(cmdBuf.graphicsState().pipelineOwner.get() == rawPtr);

    cb->reset();
    commands.shutdown();
    pipelines.shutdown();
    bindings.shutdown();
    shaders.shutdown();
}

TEST_CASE("VulkanPipelineManager defers VkPipeline destruction until CommandBufferData retention clears",
          "[size:small][rhi][vulkan]")
{
    ScopedPerfMock mock;
    rachis::vulkan::VulkanInstanceTable instanceTable{};
    VolkDeviceTable table{};
    populatePerfTables(instanceTable, table);

    const auto device = fakeVk<VkDevice>(0x305);
    const auto physical = fakeVk<VkPhysicalDevice>(0x306);
    const auto caps = baselinePerfCapabilities();

    rachis::vulkan::VulkanShaderManager shaders;
    shaders.initialize(device, &table, caps, &instanceTable);
    rachis::vulkan::VulkanBindingManager bindings;
    bindings.initialize(device, &table, caps, &instanceTable);
    rachis::vulkan::VulkanPipelineManager pipelines;
    pipelines.initialize(device, &table, physical, caps, &shaders, &bindings, &instanceTable, true);
    rachis::vulkan::VulkanGraphicsEncoder graphics;
    graphics.initialize(device, &table, &caps, nullptr, &pipelines, &bindings, true);

    const auto emptyLayout = bindings.createPipelineLayout({});
    REQUIRE(emptyLayout.has_value());
    const auto csSpv = makeMinimalComputeSpirv();
    const auto cs = shaders.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = csSpv.bytes(),
        .entryPoint = "main",
        .shaderInterface = {.workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1}},
    });
    REQUIRE(cs.has_value());
    const auto compPipe = pipelines.createComputePipeline({
        .layout = *emptyLayout,
        .computeShader = *cs,
    });
    REQUIRE(compPipe.has_value());
    CHECK(mock.state.createdPipelines.load() == 1U);

    const rachis::vulkan::VulkanQueueHandles queueHandles{
        .graphics = fakeVk<VkQueue>(0x409),
        .compute = fakeVk<VkQueue>(0x409),
        .copy = fakeVk<VkQueue>(0x409),
        .families = {.graphics = 0, .compute = 0, .transfer = 0},
    };
    rachis::vulkan::VulkanCommandAllocator commands;
    commands.initialize(device, &table, queueHandles, 0x802U, nullptr, &instanceTable);
    const auto cmdPool = commands.createCommandPool({.queueType = rachis::QueueType::compute, .maxCommandBuffers = 2});
    REQUIRE(cmdPool.has_value());
    auto cb = commands.acquireCommandBuffer(*cmdPool);
    REQUIRE(cb.has_value());
    auto& cmdBuf = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb);

    REQUIRE(graphics.bindComputePipeline(cmdBuf, *compPipe).has_value());
    CHECK(cmdBuf.retained().size() == 1U);

    REQUIRE(pipelines.destroyPipeline(*compPipe).has_value());
    CHECK(mock.state.destroyedPipelines.load() == 0U);

    cmdBuf.clearRecordingState();
    CHECK(mock.state.destroyedPipelines.load() == 1U);

    cb->reset();
    commands.shutdown();
    pipelines.shutdown();
    bindings.shutdown();
    shaders.shutdown();
}

TEST_CASE("VulkanBindingManager retains multi-descriptor BindingSetVersion in O(1) across updates and destruction",
          "[size:small][rhi][vulkan]")
{
    ScopedPerfMock mock;
    rachis::vulkan::VulkanInstanceTable instanceTable{};
    VolkDeviceTable table{};
    populatePerfTables(instanceTable, table);

    const auto instance = fakeVk<VkInstance>(0x310);
    const auto physical = fakeVk<VkPhysicalDevice>(0x311);
    const auto device = fakeVk<VkDevice>(0x312);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    VkPhysicalDeviceMemoryProperties memProps{};
    perfGetPhysicalMemProps(physical, &memProps);

    rachis::vulkan::VulkanResourceAllocator resources;
    REQUIRE(resources
                .initialize(instance,
                            &instanceTable,
                            physical,
                            device,
                            &table,
                            memProps,
                            families,
                            fakeVk<VkQueue>(0x313),
                            64,
                            false)
                .has_value());

    const auto caps = baselinePerfCapabilities();
    const rachis::vulkan::VulkanBufferBindingLimits bufferLimits{
        .minUniformBufferOffsetAlignment = 64,
        .minStorageBufferOffsetAlignment = 32,
        .maxUniformBufferRange = 65536,
        .maxStorageBufferRange = 1U << 20,
    };

    rachis::vulkan::VulkanShaderManager shaders;
    shaders.initialize(device, &table, caps, &instanceTable);
    rachis::vulkan::VulkanBindingManager bindings;
    bindings.initialize(device, &table, caps, &instanceTable, bufferLimits);
    rachis::vulkan::VulkanPipelineManager pipelines;
    pipelines.initialize(
        device, &table, physical, caps, &shaders, &bindings, &instanceTable, true, &resources.formatCache());
    rachis::vulkan::VulkanGraphicsEncoder graphics;
    graphics.initialize(device, &table, &caps, &resources, &pipelines, &bindings, true, bufferLimits);

    const auto uboDyn = resources.createBuffer({
        .size = 512,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
    });
    const auto ssbo = resources.createBuffer({
        .size = 1024,
        .usage = rachis::BufferUsage::storage,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    const auto uboStatic = resources.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
    });
    const auto sampledTex = resources.createTexture({
        .width = 16,
        .height = 16,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
    });
    const auto storageTex = resources.createTexture({
        .width = 16,
        .height = 16,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::storage,
    });
    REQUIRE(uboDyn.has_value());
    REQUIRE(ssbo.has_value());
    REQUIRE(uboStatic.has_value());
    REQUIRE(sampledTex.has_value());
    REQUIRE(storageTex.has_value());

    const auto sampledView = resources.createTextureView({
        .texture = *sampledTex,
        .usage = rachis::TextureViewUsage::sampled,
    });
    const auto storageView = resources.createTextureView({
        .texture = *storageTex,
        .usage = rachis::TextureViewUsage::storage,
    });
    const auto sampler = resources.createSampler({});
    REQUIRE(sampledView.has_value());
    REQUIRE(storageView.has_value());
    REQUIRE(sampler.has_value());

    const std::array entries{
        rachis::BindingLayoutEntry{
            .binding = 0,
            .type = rachis::BindingType::dynamicUniformBuffer,
            .visibility = rachis::ShaderStage::compute,
        },
        rachis::BindingLayoutEntry{
            .binding = 1,
            .type = rachis::BindingType::storageBuffer,
            .visibility = rachis::ShaderStage::compute,
        },
        rachis::BindingLayoutEntry{
            .binding = 2,
            .type = rachis::BindingType::sampledTexture,
            .visibility = rachis::ShaderStage::compute,
        },
        rachis::BindingLayoutEntry{
            .binding = 3,
            .type = rachis::BindingType::storageTexture,
            .visibility = rachis::ShaderStage::compute,
        },
        rachis::BindingLayoutEntry{
            .binding = 4,
            .type = rachis::BindingType::sampler,
            .visibility = rachis::ShaderStage::compute,
        },
        rachis::BindingLayoutEntry{
            .binding = 5,
            .type = rachis::BindingType::uniformBuffer,
            .visibility = rachis::ShaderStage::compute,
        },
    };
    const auto setLayout = bindings.createBindingLayout({.entries = entries});
    REQUIRE(setLayout.has_value());
    const std::array setLayouts{*setLayout};
    const auto pipeLayout = bindings.createPipelineLayout({.bindingLayouts = setLayouts});
    REQUIRE(pipeLayout.has_value());

    const auto csSpv = makeMinimalComputeSpirv();
    const auto cs = shaders.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = csSpv.bytes(),
        .entryPoint = "main",
        .shaderInterface = {.workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1}},
    });
    REQUIRE(cs.has_value());
    const auto compPipe = pipelines.createComputePipeline({
        .layout = *pipeLayout,
        .computeShader = *cs,
    });
    REQUIRE(compPipe.has_value());

    const std::array poolSizes{
        rachis::BindingPoolSize{.type = rachis::BindingType::dynamicUniformBuffer, .descriptorCount = 4},
        rachis::BindingPoolSize{.type = rachis::BindingType::uniformBuffer, .descriptorCount = 4},
        rachis::BindingPoolSize{.type = rachis::BindingType::storageBuffer, .descriptorCount = 4},
        rachis::BindingPoolSize{.type = rachis::BindingType::sampledTexture, .descriptorCount = 4},
        rachis::BindingPoolSize{.type = rachis::BindingType::storageTexture, .descriptorCount = 4},
        rachis::BindingPoolSize{.type = rachis::BindingType::sampler, .descriptorCount = 4},
    };
    const auto pool = bindings.createBindingPool({.maxBindingSets = 4, .sizes = poolSizes});
    REQUIRE(pool.has_value());

    const std::array initialWrites{
        rachis::BindingWrite{.binding = 0,
                             .resource = rachis::BufferBinding{.buffer = *uboDyn, .offset = 0, .size = 256}},
        rachis::BindingWrite{.binding = 1,
                             .resource = rachis::BufferBinding{.buffer = *ssbo, .offset = 0, .size = 1024}},
        rachis::BindingWrite{.binding = 2, .resource = *sampledView},
        rachis::BindingWrite{.binding = 3, .resource = *storageView},
        rachis::BindingWrite{.binding = 4, .resource = *sampler},
        rachis::BindingWrite{.binding = 5,
                             .resource = rachis::BufferBinding{.buffer = *uboStatic, .offset = 0, .size = 256}},
    };
    const auto set =
        bindings.createBindingSet(resources, {.layout = *setLayout, .writes = initialWrites, .pool = *pool});
    REQUIRE(set.has_value());

    VkDescriptorSet v1DescriptorSet = VK_NULL_HANDLE;
    {
        const auto firstSet = bindings.nativeBindingSet(*set);
        REQUIRE(firstSet.has_value());
        CHECK(firstSet->dynamicBufferCount == 1U);
        REQUIRE(firstSet->dynamicBuffers.size() == 1U);
        v1DescriptorSet = firstSet->descriptorSet;

        std::vector<std::thread> readers;
        std::atomic<std::uint32_t> setMismatches{0};
        for (int t = 0; t < 4; ++t)
        {
            readers.emplace_back([&]() {
                for (int i = 0; i < 64; ++i)
                {
                    const auto resolved = bindings.nativeBindingSet(*set);
                    if (!resolved.has_value() || resolved->owner.get() != firstSet->owner.get() ||
                        resolved->dynamicBuffers.data() != firstSet->dynamicBuffers.data())
                    {
                        setMismatches.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            });
        }
        for (auto& reader : readers)
        {
            reader.join();
        }
        CHECK(setMismatches.load() == 0);
    }

    const rachis::vulkan::VulkanQueueHandles queueHandles{
        .graphics = fakeVk<VkQueue>(0x501),
        .compute = fakeVk<VkQueue>(0x501),
        .copy = fakeVk<VkQueue>(0x501),
        .families = families,
    };
    rachis::vulkan::VulkanCommandAllocator commands;
    commands.initialize(device, &table, queueHandles, 0x803U, &resources, &instanceTable);
    const auto cmdPool = commands.createCommandPool({.queueType = rachis::QueueType::compute, .maxCommandBuffers = 2});
    REQUIRE(cmdPool.has_value());
    auto cb = commands.acquireCommandBuffer(*cmdPool);
    REQUIRE(cb.has_value());
    auto& cmdBuf = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb);

    REQUIRE(graphics.bindComputePipeline(cmdBuf, *compPipe).has_value());
    CHECK(cmdBuf.retained().size() == 1U);

    const std::array boundSets{*set};
    const std::array dynOffsets{64U};
    REQUIRE(graphics.bindComputeBindingSets(cmdBuf, 0, boundSets, dynOffsets).has_value());
    CHECK(cmdBuf.retained().size() == 2U);
    CHECK(mock.state.boundDescriptorSetCalls.load() == 1U);

    const auto replacementUbo = resources.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(replacementUbo.has_value());

    const std::array updateWrites{
        rachis::BindingWrite{.binding = 5,
                             .resource = rachis::BufferBinding{.buffer = *replacementUbo, .offset = 0, .size = 256}},
    };
    REQUIRE(bindings.updateBindingSet(resources, *set, updateWrites).has_value());

    {
        const auto v2Set = bindings.nativeBindingSet(*set);
        REQUIRE(v2Set.has_value());
        CHECK(v2Set->descriptorSet != v1DescriptorSet);
    }

    const auto destroyedBuffersBefore = mock.state.destroyBufferCalls.load();
    const auto destroyedViewsBefore = mock.state.destroyImageViewCalls.load();
    const auto destroyedSamplersBefore = mock.state.destroySamplerCalls.load();

    REQUIRE(resources.destroyBuffer(*uboStatic).has_value());
    REQUIRE(resources.destroyTextureView(*sampledView).has_value());
    REQUIRE(resources.destroySampler(*sampler).has_value());
    REQUIRE(bindings.destroyBindingSet(*set).has_value());
    REQUIRE(bindings.destroyBindingPool(*pool).has_value());

    CHECK(mock.state.destroyedDescPools.load() == 0U);
    CHECK(mock.state.destroyBufferCalls.load() == destroyedBuffersBefore);
    CHECK(mock.state.destroyImageViewCalls.load() == destroyedViewsBefore);
    CHECK(mock.state.destroySamplerCalls.load() == destroyedSamplersBefore);

    cmdBuf.clearRecordingState();
    CHECK(mock.state.destroyedDescPools.load() == 1U);
    CHECK(mock.state.destroyBufferCalls.load() == destroyedBuffersBefore + 1U);
    CHECK(mock.state.destroyImageViewCalls.load() == destroyedViewsBefore + 1U);
    CHECK(mock.state.destroySamplerCalls.load() == destroyedSamplersBefore + 1U);

    cb->reset();
    commands.shutdown();
    pipelines.shutdown();
    bindings.shutdown();
    shaders.shutdown();
    resources.shutdown();
}

TEST_CASE("VulkanBindingManager reuses an idle set's version capacity and needs copy-on-write headroom while bound",
          "[size:small][rhi][vulkan][binding]")
{
    ScopedPerfMock mock;
    rachis::vulkan::VulkanInstanceTable instanceTable{};
    VolkDeviceTable table{};
    populatePerfTables(instanceTable, table);

    const auto instance = fakeVk<VkInstance>(0x330);
    const auto physical = fakeVk<VkPhysicalDevice>(0x331);
    const auto device = fakeVk<VkDevice>(0x332);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    VkPhysicalDeviceMemoryProperties memProps{};
    perfGetPhysicalMemProps(physical, &memProps);

    rachis::vulkan::VulkanResourceAllocator resources;
    REQUIRE(resources
                .initialize(instance,
                            &instanceTable,
                            physical,
                            device,
                            &table,
                            memProps,
                            families,
                            fakeVk<VkQueue>(0x333),
                            64,
                            false)
                .has_value());

    const auto caps = baselinePerfCapabilities();
    rachis::vulkan::VulkanShaderManager shaders;
    shaders.initialize(device, &table, caps, &instanceTable);
    rachis::vulkan::VulkanBindingManager bindings;
    bindings.initialize(device, &table, caps, &instanceTable);
    rachis::vulkan::VulkanPipelineManager pipelines;
    pipelines.initialize(
        device, &table, physical, caps, &shaders, &bindings, &instanceTable, true, &resources.formatCache());
    rachis::vulkan::VulkanGraphicsEncoder graphics;
    graphics.initialize(device, &table, &caps, &resources, &pipelines, &bindings, true);

    const auto constantsA = resources.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
    });
    const auto constantsB = resources.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(constantsA.has_value());
    REQUIRE(constantsB.has_value());

    const std::array entries{
        rachis::BindingLayoutEntry{
            .binding = 0,
            .type = rachis::BindingType::uniformBuffer,
            .visibility = rachis::ShaderStage::compute,
        },
    };
    const auto setLayout = bindings.createBindingLayout({.entries = entries});
    REQUIRE(setLayout.has_value());
    const std::array setLayouts{*setLayout};
    const auto pipeLayout = bindings.createPipelineLayout({.bindingLayouts = setLayouts});
    REQUIRE(pipeLayout.has_value());

    const auto csSpv = makeMinimalComputeSpirv();
    const auto cs = shaders.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = csSpv.bytes(),
        .entryPoint = "main",
        .shaderInterface = {.workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1}},
    });
    REQUIRE(cs.has_value());
    const auto compPipe = pipelines.createComputePipeline({
        .layout = *pipeLayout,
        .computeShader = *cs,
    });
    REQUIRE(compPipe.has_value());

    const rachis::vulkan::VulkanQueueHandles queueHandles{
        .graphics = fakeVk<VkQueue>(0x334),
        .compute = fakeVk<VkQueue>(0x334),
        .copy = fakeVk<VkQueue>(0x334),
        .families = families,
    };
    rachis::vulkan::VulkanCommandAllocator commands;
    commands.initialize(device, &table, queueHandles, 0x805U, &resources, &instanceTable);
    const auto retainedPool =
        commands.createCommandPool({.queueType = rachis::QueueType::compute, .maxCommandBuffers = 1});
    const auto unretainedPool = commands.createCommandPool({
        .queueType = rachis::QueueType::compute,
        .maxCommandBuffers = 1,
        .flags = rachis::CommandPoolFlag::unretainedResources,
    });
    REQUIRE(retainedPool.has_value());
    REQUIRE(unretainedPool.has_value());

    const std::array writesA{
        rachis::BindingWrite{.binding = 0,
                             .resource = rachis::BufferBinding{.buffer = *constantsA, .offset = 0, .size = 256}},
    };
    const std::array writesB{
        rachis::BindingWrite{.binding = 0,
                             .resource = rachis::BufferBinding{.buffer = *constantsB, .offset = 0, .size = 256}},
    };

    const auto makePool = [&](std::uint32_t maxBindingSets, std::uint32_t uniformBuffers) {
        const std::array sizes{
            rachis::BindingPoolSize{.type = rachis::BindingType::uniformBuffer, .descriptorCount = uniformBuffers},
        };
        return bindings.createBindingPool({.maxBindingSets = maxBindingSets, .sizes = sizes});
    };
    const auto descriptorSetOf = [&](rachis::BindingSetHandle set) {
        const auto native = bindings.nativeBindingSet(set);
        REQUIRE(native.has_value());
        return native->descriptorSet;
    };

    SECTION("an exact-capacity pool accepts repeated idle updates without leaking or double-freeing capacity")
    {
        const auto pool = makePool(2, 1);
        REQUIRE(pool.has_value());
        const auto set = bindings.createBindingSet(resources, {.layout = *setLayout, .writes = writesA, .pool = *pool});
        REQUIRE(set.has_value());

        for (std::uint32_t iteration = 0; iteration < 4; ++iteration)
        {
            const auto before = descriptorSetOf(*set);
            const auto updated = bindings.updateBindingSet(resources, *set, iteration % 2 == 0 ? writesB : writesA);
            if (!updated)
            {
                INFO(updated.error().message);
            }
            REQUIRE(updated.has_value());
            CHECK(descriptorSetOf(*set) != before);
        }

        const auto second =
            bindings.createBindingSet(resources, {.layout = *setLayout, .writes = writesA, .pool = *pool});
        REQUIRE_FALSE(second.has_value());
        CHECK(second.error().code == rachis::ErrorCode::resourceExhausted);

        REQUIRE(bindings.destroyBindingSet(*set).has_value());
        const auto recycled =
            bindings.createBindingSet(resources, {.layout = *setLayout, .writes = writesA, .pool = *pool});
        REQUIRE(recycled.has_value());
        REQUIRE(bindings.destroyBindingSet(*recycled).has_value());
        REQUIRE(bindings.destroyBindingPool(*pool).has_value());
    }

    SECTION("a recording command buffer that binds the set holds its version until it is reset")
    {
        for (const bool unretained : {false, true})
        {
            for (const std::uint32_t versionCapacity : {1U, 2U})
            {
                CAPTURE(unretained, versionCapacity);
                const auto pool = makePool(1, versionCapacity);
                REQUIRE(pool.has_value());
                const auto set =
                    bindings.createBindingSet(resources, {.layout = *setLayout, .writes = writesA, .pool = *pool});
                REQUIRE(set.has_value());

                auto cb = commands.acquireCommandBuffer(unretained ? *unretainedPool : *retainedPool);
                REQUIRE(cb.has_value());
                auto& cmdBuf = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb);
                REQUIRE(graphics.bindComputePipeline(cmdBuf, *compPipe).has_value());

                const std::array boundSets{*set};
                const std::array unexpectedOffsets{0U};
                const auto badBind = graphics.bindComputeBindingSets(cmdBuf, 0, boundSets, unexpectedOffsets);
                REQUIRE_FALSE(badBind.has_value());
                CHECK(badBind.error().code == rachis::ErrorCode::invalidArgument);
                REQUIRE(bindings.updateBindingSet(resources, *set, writesB).has_value());

                const auto retainedBeforeBinds = cmdBuf.retained().size();
                REQUIRE(graphics.bindComputeBindingSets(cmdBuf, 0, boundSets, {}).has_value());
                REQUIRE(graphics.bindComputeBindingSets(cmdBuf, 0, boundSets, {}).has_value());
                CHECK(cmdBuf.retained().size() == retainedBeforeBinds + 1U);
                const auto uses = cmdBuf.reserveCommandUses(0);
                REQUIRE(uses.has_value());
                CHECK((*uses)->entries.size() == 1U);
                for (std::uint32_t version = 1; version < versionCapacity; ++version)
                {
                    REQUIRE(
                        bindings.updateBindingSet(resources, *set, version % 2 == 1 ? writesA : writesB).has_value());
                }
                const auto current = descriptorSetOf(*set);
                const auto rejected = bindings.updateBindingSet(resources, *set, writesA);
                REQUIRE_FALSE(rejected.has_value());
                CHECK(rejected.error().code == rachis::ErrorCode::resourceExhausted);
                CHECK(descriptorSetOf(*set) == current);

                cmdBuf.clearRecordingState();
                REQUIRE(bindings.updateBindingSet(resources, *set, writesA).has_value());
                CHECK(descriptorSetOf(*set) != current);

                cb->reset();
                REQUIRE(bindings.destroyBindingSet(*set).has_value());
                REQUIRE(bindings.destroyBindingPool(*pool).has_value());
            }
        }
    }

    SECTION("a two-version pool allows one copy-on-write update while bound and reuses capacity after reset")
    {
        const auto pool = makePool(1, 2);
        REQUIRE(pool.has_value());
        const auto set = bindings.createBindingSet(resources, {.layout = *setLayout, .writes = writesA, .pool = *pool});
        REQUIRE(set.has_value());

        auto cb = commands.acquireCommandBuffer(*retainedPool);
        REQUIRE(cb.has_value());
        auto& cmdBuf = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb);
        REQUIRE(graphics.bindComputePipeline(cmdBuf, *compPipe).has_value());
        const std::array boundSets{*set};
        REQUIRE(graphics.bindComputeBindingSets(cmdBuf, 0, boundSets, {}).has_value());

        REQUIRE(bindings.updateBindingSet(resources, *set, writesB).has_value());
        const auto exhausted = bindings.updateBindingSet(resources, *set, writesA);
        REQUIRE_FALSE(exhausted.has_value());
        CHECK(exhausted.error().code == rachis::ErrorCode::resourceExhausted);

        cmdBuf.clearRecordingState();
        REQUIRE(bindings.updateBindingSet(resources, *set, writesA).has_value());
        REQUIRE(bindings.updateBindingSet(resources, *set, writesB).has_value());

        cb->reset();
        REQUIRE(bindings.destroyBindingSet(*set).has_value());
        REQUIRE(bindings.destroyBindingPool(*pool).has_value());
    }

    SECTION("updating a set whose pool was destroyed returns invalidState")
    {
        const auto pool = makePool(1, 1);
        REQUIRE(pool.has_value());
        const auto set = bindings.createBindingSet(resources, {.layout = *setLayout, .writes = writesA, .pool = *pool});
        REQUIRE(set.has_value());
        REQUIRE(bindings.destroyBindingPool(*pool).has_value());
        const auto updated = bindings.updateBindingSet(resources, *set, writesB);
        REQUIRE_FALSE(updated.has_value());
        CHECK(updated.error().code == rachis::ErrorCode::invalidState);
        REQUIRE(bindings.destroyBindingSet(*set).has_value());
    }

    REQUIRE(pipelines.destroyPipeline(*compPipe).has_value());
    commands.shutdown();
    pipelines.shutdown();
    bindings.shutdown();
    shaders.shutdown();
    resources.shutdown();
}

TEST_CASE("VulkanBindingManager reports a fragmented update-after-bind pool budget as resource exhaustion",
          "[size:small][rhi][vulkan][binding]")
{
    ScopedPerfMock mock;
    rachis::vulkan::VulkanInstanceTable instanceTable{};
    VolkDeviceTable table{};
    populatePerfTables(instanceTable, table);

    const auto instance = fakeVk<VkInstance>(0x340);
    const auto physical = fakeVk<VkPhysicalDevice>(0x341);
    const auto device = fakeVk<VkDevice>(0x342);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    VkPhysicalDeviceMemoryProperties memProps{};
    perfGetPhysicalMemProps(physical, &memProps);
    rachis::vulkan::VulkanResourceAllocator resources;
    REQUIRE(resources
                .initialize(instance,
                            &instanceTable,
                            physical,
                            device,
                            &table,
                            memProps,
                            families,
                            fakeVk<VkQueue>(0x343),
                            64,
                            false)
                .has_value());

    auto caps = baselinePerfCapabilities();
    caps.partiallyBoundBindingArrays = true;
    caps.updateAfterBindBindingArrays = true;
    rachis::vulkan::VulkanBindingManager bindings;
    bindings.initialize(device, &table, caps, &instanceTable);
    const std::array entries{
        rachis::BindingLayoutEntry{
            .binding = 0,
            .type = rachis::BindingType::storageBuffer,
            .visibility = rachis::ShaderStage::compute,
            .count = 4,
            .flags = rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind,
        },
    };
    const auto layout = bindings.createBindingLayout({.entries = entries});
    REQUIRE(layout.has_value());
    const std::array sizes{rachis::BindingPoolSize{rachis::BindingType::storageBuffer, 4}};
    const auto pool = bindings.createBindingPool({.maxBindingSets = 1, .sizes = sizes});
    REQUIRE(pool.has_value());

    mock.state.updateAfterBindDescPoolResult = VK_ERROR_FRAGMENTATION;
    const auto rejected = bindings.createBindingSet(resources, {.layout = *layout, .pool = *pool});
    REQUIRE_FALSE(rejected.has_value());
    CHECK(rejected.error().code == rachis::ErrorCode::resourceExhausted);
    CHECK(rejected.error().nativeCode == static_cast<std::int64_t>(VK_ERROR_FRAGMENTATION));
    CHECK(rejected.error().message.find("VK_ERROR_FRAGMENTATION") != std::string::npos);

    mock.state.updateAfterBindDescPoolResult = VK_SUCCESS;
    const auto created = bindings.createBindingSet(resources, {.layout = *layout, .pool = *pool});
    REQUIRE(created.has_value());
    REQUIRE(bindings.destroyBindingSet(*created).has_value());
    REQUIRE(bindings.destroyBindingPool(*pool).has_value());
    REQUIRE(bindings.destroyBindingLayout(*layout).has_value());
    bindings.shutdown();
    resources.shutdown();
}

TEST_CASE("VulkanBindingManager keeps a bound texture view's VkImage alive after the view and texture are destroyed",
          "[size:small][rhi][vulkan]")
{
    ScopedPerfMock mock;
    rachis::vulkan::VulkanInstanceTable instanceTable{};
    VolkDeviceTable table{};
    populatePerfTables(instanceTable, table);

    const auto instance = fakeVk<VkInstance>(0x320);
    const auto physical = fakeVk<VkPhysicalDevice>(0x321);
    const auto device = fakeVk<VkDevice>(0x322);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    VkPhysicalDeviceMemoryProperties memProps{};
    perfGetPhysicalMemProps(physical, &memProps);

    rachis::vulkan::VulkanResourceAllocator resources;
    REQUIRE(resources
                .initialize(instance,
                            &instanceTable,
                            physical,
                            device,
                            &table,
                            memProps,
                            families,
                            fakeVk<VkQueue>(0x323),
                            64,
                            false)
                .has_value());

    const auto caps = baselinePerfCapabilities();
    const rachis::vulkan::VulkanBufferBindingLimits bufferLimits{
        .minUniformBufferOffsetAlignment = 64,
        .minStorageBufferOffsetAlignment = 32,
        .maxUniformBufferRange = 65536,
        .maxStorageBufferRange = 1U << 20,
    };

    rachis::vulkan::VulkanShaderManager shaders;
    shaders.initialize(device, &table, caps, &instanceTable);
    rachis::vulkan::VulkanBindingManager bindings;
    bindings.initialize(device, &table, caps, &instanceTable, bufferLimits);
    rachis::vulkan::VulkanPipelineManager pipelines;
    pipelines.initialize(
        device, &table, physical, caps, &shaders, &bindings, &instanceTable, true, &resources.formatCache());
    rachis::vulkan::VulkanGraphicsEncoder graphics;
    graphics.initialize(device, &table, &caps, &resources, &pipelines, &bindings, true, bufferLimits);

    const auto texture = resources.createTexture({
        .width = 16,
        .height = 16,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
    });
    REQUIRE(texture.has_value());
    const auto view = resources.createTextureView({
        .texture = *texture,
        .usage = rachis::TextureViewUsage::sampled,
    });
    REQUIRE(view.has_value());

    const std::array entries{
        rachis::BindingLayoutEntry{
            .binding = 0,
            .type = rachis::BindingType::sampledTexture,
            .visibility = rachis::ShaderStage::compute,
        },
    };
    const auto setLayout = bindings.createBindingLayout({.entries = entries});
    REQUIRE(setLayout.has_value());
    const std::array setLayouts{*setLayout};
    const auto pipeLayout = bindings.createPipelineLayout({.bindingLayouts = setLayouts});
    REQUIRE(pipeLayout.has_value());

    const auto csSpv = makeMinimalComputeSpirv();
    const auto cs = shaders.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = csSpv.bytes(),
        .entryPoint = "main",
        .shaderInterface = {.workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1}},
    });
    REQUIRE(cs.has_value());
    const auto compPipe = pipelines.createComputePipeline({
        .layout = *pipeLayout,
        .computeShader = *cs,
    });
    REQUIRE(compPipe.has_value());

    const std::array poolSizes{
        rachis::BindingPoolSize{.type = rachis::BindingType::sampledTexture, .descriptorCount = 1},
    };
    const auto pool = bindings.createBindingPool({.maxBindingSets = 1, .sizes = poolSizes});
    REQUIRE(pool.has_value());
    const std::array writes{
        rachis::BindingWrite{.binding = 0, .resource = *view},
    };
    const auto set = bindings.createBindingSet(resources, {.layout = *setLayout, .writes = writes, .pool = *pool});
    REQUIRE(set.has_value());

    const rachis::vulkan::VulkanQueueHandles queueHandles{
        .graphics = fakeVk<VkQueue>(0x502),
        .compute = fakeVk<VkQueue>(0x502),
        .copy = fakeVk<VkQueue>(0x502),
        .families = families,
    };
    rachis::vulkan::VulkanCommandAllocator commands;
    commands.initialize(device, &table, queueHandles, 0x803U, &resources, &instanceTable);
    const auto cmdPool = commands.createCommandPool({.queueType = rachis::QueueType::compute, .maxCommandBuffers = 1});
    REQUIRE(cmdPool.has_value());
    auto cb = commands.acquireCommandBuffer(*cmdPool);
    REQUIRE(cb.has_value());
    auto& cmdBuf = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb);

    REQUIRE(graphics.bindComputePipeline(cmdBuf, *compPipe).has_value());
    const std::array boundSets{*set};
    REQUIRE(graphics.bindComputeBindingSets(cmdBuf, 0, boundSets, {}).has_value());

    const auto destroyedImagesBefore = mock.state.destroyImageCalls.load();
    const auto destroyedViewsBefore = mock.state.destroyImageViewCalls.load();

    REQUIRE(bindings.destroyBindingSet(*set).has_value());
    REQUIRE(bindings.destroyBindingPool(*pool).has_value());
    REQUIRE(resources.destroyTextureView(*view).has_value());
    REQUIRE(resources.destroyTexture(*texture).has_value());

    CHECK(mock.state.destroyImageViewCalls.load() == destroyedViewsBefore);
    CHECK(mock.state.destroyImageCalls.load() == destroyedImagesBefore);

    cmdBuf.clearRecordingState();
    CHECK(mock.state.destroyImageViewCalls.load() == destroyedViewsBefore + 1U);
    CHECK(mock.state.destroyImageCalls.load() == destroyedImagesBefore + 1U);

    cb->reset();
    commands.shutdown();
    pipelines.shutdown();
    bindings.shutdown();
    shaders.shutdown();
    resources.shutdown();
}

TEST_CASE("VulkanResourceAllocator persistently maps VMA host buffers with zero runtime vkMapMemory/vkUnmapMemory",
          "[size:small][rhi][vulkan]")
{
    SECTION("coherent upload and readback buffers execute writeBuffer/readBuffer/mapBuffer/unmapBuffer with zero "
            "runtime map or flush calls")
    {
        ScopedPerfMock mock(MemoryTopologyKind::coherentHostVisible);
        rachis::vulkan::VulkanInstanceTable instanceTable{};
        VolkDeviceTable table{};
        populatePerfTables(instanceTable, table);

        const auto instance = fakeVk<VkInstance>(0x320);
        const auto physical = fakeVk<VkPhysicalDevice>(0x321);
        const auto device = fakeVk<VkDevice>(0x322);
        const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
        VkPhysicalDeviceMemoryProperties memProps{};
        perfGetPhysicalMemProps(physical, &memProps);

        rachis::vulkan::VulkanResourceAllocator resources;
        REQUIRE(resources
                    .initialize(instance,
                                &instanceTable,
                                physical,
                                device,
                                &table,
                                memProps,
                                families,
                                fakeVk<VkQueue>(0x323),
                                64,
                                false)
                    .has_value());

        const auto upload = resources.createBuffer({
            .size = 256,
            .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::constant,
            .memoryType = rachis::MemoryType::upload,
        });
        const auto readback = resources.createBuffer({
            .size = 256,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(upload.has_value());
        REQUIRE(readback.has_value());

        CHECK(mock.state.mapMemoryCalls.load() >= 1U);
        mock.state.mapMemoryCalls.store(0);
        mock.state.unmapMemoryCalls.store(0);
        mock.state.flushRangeCalls.store(0);
        mock.state.invalidateRangeCalls.store(0);

        std::array<std::byte, 64> pattern{};
        for (std::size_t i = 0; i < pattern.size(); ++i)
        {
            pattern[i] = static_cast<std::byte>(0xA0U + i);
        }

        for (int iter = 0; iter < 50; ++iter)
        {
            REQUIRE(resources.writeBuffer(*upload, 16, pattern).has_value());
            auto mappedUpload = resources.mapBuffer(*upload, 0, rachis::wholeResource);
            REQUIRE(mappedUpload.has_value());
            CHECK(std::memcmp(mappedUpload->data + 16, pattern.data(), pattern.size()) == 0);
            REQUIRE(resources.unmapBuffer(*upload, 0, rachis::wholeResource).has_value());

            auto mappedReadback = resources.mapBuffer(*readback, 0, rachis::wholeResource);
            REQUIRE(mappedReadback.has_value());
            std::memcpy(mappedReadback->data + 32, pattern.data(), pattern.size());
            REQUIRE(resources.unmapBuffer(*readback, 0, rachis::wholeResource).has_value());

            std::array<std::byte, 64> readOut{};
            REQUIRE(resources.readBuffer(*readback, 32, readOut).has_value());
            CHECK(readOut == pattern);
        }

        CHECK(mock.state.mapMemoryCalls.load() == 0U);
        CHECK(mock.state.unmapMemoryCalls.load() == 0U);
        CHECK(mock.state.flushRangeCalls.load() == 0U);
        CHECK(mock.state.invalidateRangeCalls.load() == 0U);

        CHECK_FALSE(resources.unmapBuffer(*upload, 0, rachis::wholeResource).has_value());
        auto activeMap = resources.mapBuffer(*upload, 0, 128);
        REQUIRE(activeMap.has_value());
        const auto doubleMap = resources.mapBuffer(*upload, 0, 128);
        REQUIRE_FALSE(doubleMap.has_value());
        CHECK(doubleMap.error().code == rachis::ErrorCode::invalidState);
        const auto writeWhileMapped = resources.writeBuffer(*upload, 0, pattern);
        REQUIRE_FALSE(writeWhileMapped.has_value());
        CHECK(writeWhileMapped.error().code == rachis::ErrorCode::invalidState);
        const auto destroyWhileMapped = resources.destroyBuffer(*upload);
        REQUIRE_FALSE(destroyWhileMapped.has_value());
        CHECK(destroyWhileMapped.error().code == rachis::ErrorCode::invalidState);
        REQUIRE(resources.unmapBuffer(*upload, 0, 128).has_value());

        auto activeReadMap = resources.mapBuffer(*readback, 0, 128);
        REQUIRE(activeReadMap.has_value());
        std::array<std::byte, 16> sink{};
        const auto readWhileMapped = resources.readBuffer(*readback, 0, sink);
        REQUIRE_FALSE(readWhileMapped.has_value());
        CHECK(readWhileMapped.error().code == rachis::ErrorCode::invalidState);
        REQUIRE(resources.unmapBuffer(*readback, 0, 128).has_value());

        CHECK(mock.state.mapMemoryCalls.load() == 0U);
        CHECK(mock.state.unmapMemoryCalls.load() == 0U);

        resources.shutdown();
    }

    SECTION("non-coherent host buffers still flush and invalidate while avoiding runtime vkMapMemory/vkUnmapMemory")
    {
        ScopedPerfMock mock(MemoryTopologyKind::nonCoherentHostVisible);
        rachis::vulkan::VulkanInstanceTable instanceTable{};
        VolkDeviceTable table{};
        populatePerfTables(instanceTable, table);

        const auto instance = fakeVk<VkInstance>(0x324);
        const auto physical = fakeVk<VkPhysicalDevice>(0x325);
        const auto device = fakeVk<VkDevice>(0x326);
        const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
        VkPhysicalDeviceMemoryProperties memProps{};
        perfGetPhysicalMemProps(physical, &memProps);

        rachis::vulkan::VulkanResourceAllocator resources;
        REQUIRE(resources
                    .initialize(instance,
                                &instanceTable,
                                physical,
                                device,
                                &table,
                                memProps,
                                families,
                                fakeVk<VkQueue>(0x327),
                                64,
                                false)
                    .has_value());

        const auto upload = resources.createBuffer({
            .size = 256,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
        });
        const auto readback = resources.createBuffer({
            .size = 256,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(upload.has_value());
        REQUIRE(readback.has_value());

        mock.state.mapMemoryCalls.store(0);
        mock.state.unmapMemoryCalls.store(0);
        mock.state.flushRangeCalls.store(0);
        mock.state.invalidateRangeCalls.store(0);

        const std::array<std::byte, 32> payload{std::byte{0x42}};
        REQUIRE(resources.writeBuffer(*upload, 0, payload).has_value());
        CHECK(mock.state.flushRangeCalls.load() == 1U);

        auto mapped = resources.mapBuffer(*upload, 0, 128);
        REQUIRE(mapped.has_value());
        REQUIRE(resources.unmapBuffer(*upload, 0, 128).has_value());
        CHECK(mock.state.flushRangeCalls.load() == 2U);

        std::array<std::byte, 32> out{};
        REQUIRE(resources.readBuffer(*readback, 0, out).has_value());
        CHECK(mock.state.invalidateRangeCalls.load() == 1U);

        auto partial = resources.mapBuffer(*upload, 64, 64);
        REQUIRE(partial.has_value());
        const auto outsideUnmap = resources.unmapBuffer(*upload, 0, 32);
        REQUIRE_FALSE(outsideUnmap.has_value());
        CHECK(outsideUnmap.error().code == rachis::ErrorCode::invalidArgument);
        mock.state.lastFlushSize.store(0);
        REQUIRE(resources.unmapBuffer(*upload, 0, rachis::wholeResource).has_value());
        CHECK(mock.state.flushRangeCalls.load() == 3U);
        CHECK(mock.state.lastFlushSize.load() == 64U);

        auto mappedAtLoss = resources.mapBuffer(*upload, 0, 128);
        REQUIRE(mappedAtLoss.has_value());
        mock.state.flushResult.store(VK_ERROR_DEVICE_LOST);
        const auto failedUnmap = resources.unmapBuffer(*upload, 0, 128);
        REQUIRE_FALSE(failedUnmap.has_value());
        CHECK(failedUnmap.error().code == rachis::ErrorCode::deviceLost);
        const auto stillMapped = resources.mapBuffer(*upload, 0, 128);
        REQUIRE_FALSE(stillMapped.has_value());
        CHECK(stillMapped.error().code == rachis::ErrorCode::invalidState);
        const auto flushesBeforeLostUnmap = mock.state.flushRangeCalls.load();
        REQUIRE(resources.unmapBuffer(*upload, 0, 128, false).has_value());
        CHECK(mock.state.flushRangeCalls.load() == flushesBeforeLostUnmap);
        mock.state.flushResult.store(VK_SUCCESS);
        auto remapped = resources.mapBuffer(*upload, 0, 128);
        REQUIRE(remapped.has_value());
        REQUIRE(resources.unmapBuffer(*upload, 0, 128).has_value());

        CHECK(mock.state.mapMemoryCalls.load() == 0U);
        CHECK(mock.state.unmapMemoryCalls.load() == 0U);

        resources.shutdown();
    }

    SECTION("MemoryType::upload selects ReBAR DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT when >256 MiB and preserves "
            "cached readback")
    {
        {
            ScopedPerfMock rebarMock(MemoryTopologyKind::rebarDiscrete);
            rachis::vulkan::VulkanInstanceTable instanceTable{};
            VolkDeviceTable table{};
            populatePerfTables(instanceTable, table);

            const auto instance = fakeVk<VkInstance>(0x328);
            const auto physical = fakeVk<VkPhysicalDevice>(0x329);
            const auto device = fakeVk<VkDevice>(0x32A);
            const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
            VkPhysicalDeviceMemoryProperties memProps{};
            perfGetPhysicalMemProps(physical, &memProps);

            rachis::vulkan::VulkanResourceAllocator resources;
            REQUIRE(resources
                        .initialize(instance,
                                    &instanceTable,
                                    physical,
                                    device,
                                    &table,
                                    memProps,
                                    families,
                                    fakeVk<VkQueue>(0x32B),
                                    64,
                                    false,
                                    true,
                                    false,
                                    false,
                                    true)
                        .has_value());

            const auto uploadHeap = resources.createResourceHeap({
                .size = 65536,
                .memoryType = rachis::MemoryType::upload,
                .resourceClass = rachis::ResourceHeapClass::buffer,
            });
            const auto readbackHeap = resources.createResourceHeap({
                .size = 65536,
                .memoryType = rachis::MemoryType::readback,
                .resourceClass = rachis::ResourceHeapClass::buffer,
            });
            REQUIRE(uploadHeap.has_value());
            REQUIRE(readbackHeap.has_value());

            CHECK(rachis::vulkan::VulkanPerformanceTestAccess::heapMemoryTypeIndex(resources, *uploadHeap) == 2U);

            CHECK(rachis::vulkan::VulkanPerformanceTestAccess::heapMemoryTypeIndex(resources, *readbackHeap) == 3U);

            const auto oversizedRebarUpload = resources.createResourceHeap({
                .size = 12ULL * 1024ULL * 1024ULL * 1024ULL,
                .memoryType = rachis::MemoryType::upload,
                .resourceClass = rachis::ResourceHeapClass::buffer,
            });
            REQUIRE(oversizedRebarUpload.has_value());
            CHECK(rachis::vulkan::VulkanPerformanceTestAccess::heapMemoryTypeIndex(resources, *oversizedRebarUpload) ==
                  1U);
            REQUIRE(resources.destroyResourceHeap(*oversizedRebarUpload).has_value());

            const auto smallPriorityUpload = resources.createBuffer({
                .size = 65536,
                .usage = rachis::BufferUsage::copySource,
                .memoryType = rachis::MemoryType::upload,
                .residencyPriority = rachis::ResidencyPriority::high,
            });
            REQUIRE(smallPriorityUpload.has_value());
            CHECK(rachis::vulkan::VulkanPerformanceTestAccess::bufferMemoryTypeIndex(resources, *smallPriorityUpload) ==
                  2U);

            const auto oversizedPriorityUpload = resources.createBuffer({
                .size = 12ULL * 1024ULL * 1024ULL * 1024ULL,
                .usage = rachis::BufferUsage::copySource,
                .memoryType = rachis::MemoryType::upload,
                .residencyPriority = rachis::ResidencyPriority::high,
            });
            REQUIRE(oversizedPriorityUpload.has_value());
            CHECK(rachis::vulkan::VulkanPerformanceTestAccess::bufferMemoryTypeIndex(resources,
                                                                                     *oversizedPriorityUpload) == 1U);
            REQUIRE(resources.destroyBuffer(*oversizedPriorityUpload).has_value());

            const auto rebarFiveGiBFirst = resources.createBuffer({
                .size = 5ULL * 1024ULL * 1024ULL * 1024ULL,
                .usage = rachis::BufferUsage::copySource,
                .memoryType = rachis::MemoryType::upload,
                .residencyPriority = rachis::ResidencyPriority::high,
            });
            REQUIRE(rebarFiveGiBFirst.has_value());
            CHECK(rachis::vulkan::VulkanPerformanceTestAccess::bufferMemoryTypeIndex(resources, *rebarFiveGiBFirst) ==
                  2U);

            const auto rebarFiveGiBSecond = resources.createBuffer({
                .size = 5ULL * 1024ULL * 1024ULL * 1024ULL,
                .usage = rachis::BufferUsage::copySource,
                .memoryType = rachis::MemoryType::upload,
                .residencyPriority = rachis::ResidencyPriority::high,
            });
            REQUIRE(rebarFiveGiBSecond.has_value());
            CHECK(rachis::vulkan::VulkanPerformanceTestAccess::bufferMemoryTypeIndex(resources, *rebarFiveGiBSecond) ==
                  1U);

            resources.shutdown();
        }
        {
            ScopedPerfMock smallBarMock(MemoryTopologyKind::smallBarDiscrete);
            rachis::vulkan::VulkanInstanceTable instanceTable{};
            VolkDeviceTable table{};
            populatePerfTables(instanceTable, table);

            const auto instance = fakeVk<VkInstance>(0x32C);
            const auto physical = fakeVk<VkPhysicalDevice>(0x32D);
            const auto device = fakeVk<VkDevice>(0x32E);
            const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
            VkPhysicalDeviceMemoryProperties memProps{};
            perfGetPhysicalMemProps(physical, &memProps);

            rachis::vulkan::VulkanResourceAllocator resources;
            REQUIRE(resources
                        .initialize(instance,
                                    &instanceTable,
                                    physical,
                                    device,
                                    &table,
                                    memProps,
                                    families,
                                    fakeVk<VkQueue>(0x32F),
                                    64,
                                    false,
                                    true,
                                    false,
                                    false,
                                    true)
                        .has_value());

            const auto smallUploadHeap = resources.createResourceHeap({
                .size = 65536,
                .memoryType = rachis::MemoryType::upload,
                .resourceClass = rachis::ResourceHeapClass::buffer,
            });
            REQUIRE(smallUploadHeap.has_value());

            CHECK(rachis::vulkan::VulkanPerformanceTestAccess::heapMemoryTypeIndex(resources, *smallUploadHeap) == 2U);

            const auto largeUploadHeap = resources.createResourceHeap({
                .size = 512ULL * 1024ULL * 1024ULL,
                .memoryType = rachis::MemoryType::upload,
                .resourceClass = rachis::ResourceHeapClass::buffer,
            });
            REQUIRE(largeUploadHeap.has_value());
            CHECK(rachis::vulkan::VulkanPerformanceTestAccess::heapMemoryTypeIndex(resources, *largeUploadHeap) == 2U);

            const auto largePriorityUpload = resources.createBuffer({
                .size = 512ULL * 1024ULL * 1024ULL,
                .usage = rachis::BufferUsage::copySource,
                .memoryType = rachis::MemoryType::upload,
                .residencyPriority = rachis::ResidencyPriority::high,
            });
            REQUIRE(largePriorityUpload.has_value());
            CHECK(rachis::vulkan::VulkanPerformanceTestAccess::bufferMemoryTypeIndex(resources, *largePriorityUpload) ==
                  2U);

            const auto barOnlySmall = rachis::vulkan::VulkanPerformanceTestAccess::selectMemoryType(
                resources, 1U << 1U, rachis::MemoryType::upload, 65536);
            REQUIRE(barOnlySmall.has_value());
            CHECK(*barOnlySmall == 1U);

            const auto barOnlyOversized = rachis::vulkan::VulkanPerformanceTestAccess::selectMemoryType(
                resources, 1U << 1U, rachis::MemoryType::upload, 512ULL * 1024ULL * 1024ULL);
            REQUIRE(!barOnlyOversized.has_value());
            CHECK(barOnlyOversized.error().code == rachis::ErrorCode::outOfMemory);

            resources.shutdown();
        }
    }
}

TEST_CASE("VulkanResourceAllocator gives automatic-priority dedicated allocations the default 0.5 memory priority",
          "[size:small][rhi][vulkan]")
{
    const auto initialize = [](rachis::vulkan::VulkanResourceAllocator& resources,
                               rachis::vulkan::VulkanInstanceTable& instanceTable,
                               VolkDeviceTable& table,
                               bool memoryPriority) {
        populatePerfTables(instanceTable, table);
        const auto physical = fakeVk<VkPhysicalDevice>(0x3A1);
        const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
        VkPhysicalDeviceMemoryProperties memProps{};
        perfGetPhysicalMemProps(physical, &memProps);
        return resources.initialize(fakeVk<VkInstance>(0x3A0),
                                    &instanceTable,
                                    physical,
                                    fakeVk<VkDevice>(0x3A2),
                                    &table,
                                    memProps,
                                    families,
                                    fakeVk<VkQueue>(0x3A3),
                                    64,
                                    false,
                                    false,
                                    false,
                                    false,
                                    memoryPriority);
    };
    const auto recordedPriorities = [](const PerfMockState& state) {
        std::vector<float> priorities;
        for (const auto& priority : state.allocatedMemoryPriorities)
        {
            REQUIRE(priority.has_value());
            priorities.push_back(*priority);
        }
        return priorities;
    };

    SECTION("dedicated automatic buffers and textures get 0.5 while explicit priorities keep their pool value")
    {
        ScopedPerfMock mock;
        mock.state.requireDedicatedAllocation = true;
        rachis::vulkan::VulkanInstanceTable instanceTable{};
        VolkDeviceTable table{};
        rachis::vulkan::VulkanResourceAllocator resources;
        REQUIRE(initialize(resources, instanceTable, table, true).has_value());

        const auto deviceBuffer = resources.createBuffer({
            .size = 4'096,
            .usage = rachis::BufferUsage::storage,
            .memoryType = rachis::MemoryType::deviceLocal,
        });
        REQUIRE(deviceBuffer.has_value());
        const auto uploadBuffer = resources.createBuffer({
            .size = 4'096,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
        });
        REQUIRE(uploadBuffer.has_value());
        const auto attachment = resources.createTexture({
            .width = 64,
            .height = 64,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::renderTarget,
        });
        REQUIRE(attachment.has_value());
        const auto lowBuffer = resources.createBuffer({
            .size = 4'096,
            .usage = rachis::BufferUsage::storage,
            .memoryType = rachis::MemoryType::deviceLocal,
            .residencyPriority = rachis::ResidencyPriority::low,
        });
        REQUIRE(lowBuffer.has_value());
        const auto highTexture = resources.createTexture({
            .width = 64,
            .height = 64,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::sampled,
            .residencyPriority = rachis::ResidencyPriority::high,
        });
        REQUIRE(highTexture.has_value());

        CHECK(recordedPriorities(mock.state) == std::vector<float>{0.5F, 0.5F, 0.5F, 0.25F, 1.0F});

        REQUIRE(resources.destroyTexture(*highTexture).has_value());
        REQUIRE(resources.destroyBuffer(*lowBuffer).has_value());
        REQUIRE(resources.destroyTexture(*attachment).has_value());
        REQUIRE(resources.destroyBuffer(*uploadBuffer).has_value());
        REQUIRE(resources.destroyBuffer(*deviceBuffer).has_value());
        resources.shutdown();
    }

    SECTION("an automatic buffer above half the default block size gets a dedicated 0.5 allocation")
    {
        ScopedPerfMock mock(MemoryTopologyKind::rebarDiscrete);
        rachis::vulkan::VulkanInstanceTable instanceTable{};
        VolkDeviceTable table{};
        rachis::vulkan::VulkanResourceAllocator resources;
        REQUIRE(initialize(resources, instanceTable, table, true).has_value());

        constexpr std::uint64_t largeBytes = 160ULL * 1024ULL * 1024ULL;
        const auto automaticBuffer = resources.createBuffer({
            .size = largeBytes,
            .usage = rachis::BufferUsage::storage,
            .memoryType = rachis::MemoryType::deviceLocal,
        });
        REQUIRE(automaticBuffer.has_value());
        REQUIRE(mock.state.allocatedMemorySizes.size() == 1U);
        CHECK(mock.state.allocatedMemorySizes.front() == largeBytes);

        const auto lowBuffer = resources.createBuffer({
            .size = largeBytes,
            .usage = rachis::BufferUsage::storage,
            .memoryType = rachis::MemoryType::deviceLocal,
            .residencyPriority = rachis::ResidencyPriority::low,
        });
        REQUIRE(lowBuffer.has_value());

        CHECK(recordedPriorities(mock.state) == std::vector<float>{0.5F, 0.25F});

        REQUIRE(resources.destroyBuffer(*lowBuffer).has_value());
        REQUIRE(resources.destroyBuffer(*automaticBuffer).has_value());
        resources.shutdown();
    }

    SECTION("devices without memory priority chain no priority to automatic allocations")
    {
        ScopedPerfMock mock;
        mock.state.requireDedicatedAllocation = true;
        rachis::vulkan::VulkanInstanceTable instanceTable{};
        VolkDeviceTable table{};
        rachis::vulkan::VulkanResourceAllocator resources;
        REQUIRE(initialize(resources, instanceTable, table, false).has_value());

        const auto buffer = resources.createBuffer({
            .size = 4'096,
            .usage = rachis::BufferUsage::storage,
            .memoryType = rachis::MemoryType::deviceLocal,
        });
        REQUIRE(buffer.has_value());
        const auto texture = resources.createTexture({
            .width = 64,
            .height = 64,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::renderTarget,
        });
        REQUIRE(texture.has_value());

        REQUIRE(mock.state.allocatedMemoryPriorities.size() == 2U);
        CHECK_FALSE(mock.state.allocatedMemoryPriorities[0].has_value());
        CHECK_FALSE(mock.state.allocatedMemoryPriorities[1].has_value());

        REQUIRE(resources.destroyTexture(*texture).has_value());
        REQUIRE(resources.destroyBuffer(*buffer).has_value());
        resources.shutdown();
    }
}

TEST_CASE("VulkanResourceAllocator binds short resource and heap record labels to their stored strings",
          "[size:small][rhi][vulkan]")
{
    ScopedPerfMock mock;
    rachis::vulkan::VulkanInstanceTable instanceTable{};
    VolkDeviceTable table{};
    populatePerfTables(instanceTable, table);

    const auto instance = fakeVk<VkInstance>(0x348);
    const auto physical = fakeVk<VkPhysicalDevice>(0x349);
    const auto device = fakeVk<VkDevice>(0x34A);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    VkPhysicalDeviceMemoryProperties memProps{};
    perfGetPhysicalMemProps(physical, &memProps);

    rachis::vulkan::VulkanResourceAllocator resources;
    REQUIRE(resources
                .initialize(instance,
                            &instanceTable,
                            physical,
                            device,
                            &table,
                            memProps,
                            families,
                            fakeVk<VkQueue>(0x34B),
                            64,
                            false)
                .has_value());

    using Access = rachis::vulkan::VulkanPerformanceTestAccess;
    const auto checkBound = [](const Access::StoredLabel& stored, std::string_view expected) {
        CHECK(stored.owned == expected);
        CHECK(static_cast<const void*>(stored.described.data()) == static_cast<const void*>(stored.owned.data()));
        CHECK(stored.described.size() == stored.owned.size());
    };

    const auto buffer = resources.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::storage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Buf",
    });
    REQUIRE(buffer.has_value());
    const auto bufferLabel = Access::bufferLabel(resources, *buffer);
    checkBound(bufferLabel, "Buf");
    const auto resolvedBuffer = resources.resolveBindingBuffer(*buffer);
    REQUIRE(resolvedBuffer.has_value());
    CHECK(static_cast<const void*>(resolvedBuffer->desc.label.data()) ==
          static_cast<const void*>(bufferLabel.owned.data()));
    CHECK(resolvedBuffer->desc.label.size() == bufferLabel.owned.size());

    const auto texture = resources.createTexture({
        .width = 16,
        .height = 16,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
        .label = "Tex",
    });
    REQUIRE(texture.has_value());
    const auto textureLabel = Access::textureLabel(resources, *texture);
    checkBound(textureLabel, "Tex");
    const auto resolvedTexture = resources.resolveTexture(*texture);
    REQUIRE(resolvedTexture.has_value());
    CHECK(static_cast<const void*>(resolvedTexture->desc.label.data()) ==
          static_cast<const void*>(textureLabel.owned.data()));
    CHECK(resolvedTexture->desc.label.size() == textureLabel.owned.size());

    const auto view = resources.createTextureView({
        .texture = *texture,
        .usage = rachis::TextureViewUsage::sampled,
        .label = "View",
    });
    REQUIRE(view.has_value());
    checkBound(Access::textureViewLabel(resources, *view), "View");

    const auto sampler = resources.createSampler({.label = "Smp"});
    REQUIRE(sampler.has_value());
    checkBound(Access::samplerLabel(resources, *sampler), "Smp");

    const auto adopted = resources.adoptSwapchainImage(fakeVk<VkImage>(0x34C),
                                                       {
                                                           .width = 16,
                                                           .height = 16,
                                                           .format = rachis::Format::bgra8Unorm,
                                                           .usage = rachis::TextureUsage::renderTarget,
                                                           .label = "Swap",
                                                       });
    REQUIRE(adopted.has_value());
    checkBound(Access::textureLabel(resources, *adopted), "Swap");

    const auto bufferHeap = resources.createResourceHeap({
        .size = 65536,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::buffer,
        .label = "BufHeap",
    });
    REQUIRE(bufferHeap.has_value());
    checkBound(Access::heapLabel(resources, *bufferHeap), "BufHeap");

    const auto placedBuffer = resources.createPlacedBuffer({.heap = *bufferHeap, .offset = 0},
                                                           {
                                                               .size = 256,
                                                               .usage = rachis::BufferUsage::storage,
                                                               .memoryType = rachis::MemoryType::deviceLocal,
                                                               .label = "PBuf",
                                                           });
    REQUIRE(placedBuffer.has_value());
    checkBound(Access::bufferLabel(resources, *placedBuffer), "PBuf");

    const auto textureHeap = resources.createResourceHeap({
        .size = 262144,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::nonAttachmentTexture,
        .label = "TexHeap",
    });
    REQUIRE(textureHeap.has_value());
    checkBound(Access::heapLabel(resources, *textureHeap), "TexHeap");

    const auto placedTexture = resources.createPlacedTexture({.heap = *textureHeap, .offset = 0},
                                                             {
                                                                 .width = 16,
                                                                 .height = 16,
                                                                 .format = rachis::Format::rgba8Unorm,
                                                                 .usage = rachis::TextureUsage::sampled,
                                                                 .label = "PTex",
                                                             });
    REQUIRE(placedTexture.has_value());
    checkBound(Access::textureLabel(resources, *placedTexture), "PTex");

    REQUIRE(resources.forgetSwapchainImage(*adopted).has_value());
    resources.shutdown();
}

TEST_CASE("VulkanResourceAllocator rejects a sampler lodBias beyond the device maxSamplerLodBias",
          "[size:small][rhi][vulkan]")
{
    ScopedPerfMock mock;
    mock.state.maxSamplerLodBias = 2.0F;
    rachis::vulkan::VulkanInstanceTable instanceTable{};
    VolkDeviceTable table{};
    populatePerfTables(instanceTable, table);

    const auto physical = fakeVk<VkPhysicalDevice>(0x3B1);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    VkPhysicalDeviceMemoryProperties memProps{};
    perfGetPhysicalMemProps(physical, &memProps);

    rachis::vulkan::VulkanResourceAllocator resources;
    REQUIRE(resources
                .initialize(fakeVk<VkInstance>(0x3B0),
                            &instanceTable,
                            physical,
                            fakeVk<VkDevice>(0x3B2),
                            &table,
                            memProps,
                            families,
                            fakeVk<VkQueue>(0x3B3),
                            64,
                            false)
                .has_value());

    for (const auto mipFilter : {rachis::MipmapFilterMode::none, rachis::MipmapFilterMode::linear})
    {
        for (const float bias : {2.5F, -2.5F, std::nextafter(2.0F, 3.0F), -std::nextafter(2.0F, 3.0F)})
        {
            CAPTURE(bias);
            const auto rejected = resources.createSampler({.mipFilter = mipFilter, .lodBias = bias});
            REQUIRE_FALSE(rejected.has_value());
            CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
        }
    }
    CHECK(mock.state.createSamplerCalls.load() == 0);
    CHECK(mock.state.samplerMipLodBiases.empty());

    for (const float bias : {2.0F, -2.0F, 0.0F})
    {
        CAPTURE(bias);
        const auto accepted = resources.createSampler({.mipFilter = rachis::MipmapFilterMode::linear, .lodBias = bias});
        REQUIRE(accepted.has_value());
        REQUIRE(resources.destroySampler(*accepted).has_value());
    }
    CHECK(mock.state.samplerMipLodBiases == std::vector<float>{2.0F, -2.0F, 0.0F});

    resources.shutdown();
}

TEST_CASE("Vulkan query resolves make results copied into readback buffers visible to the host",
          "[size:small][rhi][vulkan][queries]")
{
    ScopedPerfMock mock;
    rachis::vulkan::VulkanInstanceTable instanceTable{};
    VolkDeviceTable table{};
    populatePerfTables(instanceTable, table);

    const auto instance = fakeVk<VkInstance>(0x330);
    const auto physical = fakeVk<VkPhysicalDevice>(0x331);
    const auto device = fakeVk<VkDevice>(0x332);
    const rachis::vulkan::VulkanQueueHandles queueHandles{
        .graphics = fakeVk<VkQueue>(0x333),
        .compute = fakeVk<VkQueue>(0x333),
        .copy = fakeVk<VkQueue>(0x333),
        .families = {.graphics = 0, .compute = 0, .transfer = 0},
    };
    VkPhysicalDeviceMemoryProperties memProps{};
    perfGetPhysicalMemProps(physical, &memProps);

    rachis::vulkan::VulkanResourceAllocator resources;
    REQUIRE(resources
                .initialize(instance,
                            &instanceTable,
                            physical,
                            device,
                            &table,
                            memProps,
                            queueHandles.families,
                            queueHandles.copy,
                            64,
                            false)
                .has_value());

    rachis::DeviceCapabilities caps{};
    caps.occlusionQueries = true;
    caps.pipelineStatisticsQueries = true;
    caps.meshPipelineStatisticsQueries = true;
    rachis::vulkan::VulkanQueryManager queries;
    queries.initialize(device, &table, &caps, &resources, false, &instanceTable);
    rachis::vulkan::VulkanCommandAllocator commands;
    commands.initialize(device, &table, queueHandles, 0x803U, &resources, &instanceTable);
    const auto cmdPool = commands.createCommandPool({.queueType = rachis::QueueType::graphics, .maxCommandBuffers = 4});
    REQUIRE(cmdPool.has_value());

    constexpr std::uint32_t queryCount = 2;
    const auto timestampPool =
        queries.createTimestampQueryPool({.queueType = rachis::QueueType::graphics, .queryCount = queryCount});
    const auto occlusionPool = queries.createOcclusionQueryPool({.queryCount = queryCount});
    const auto statisticsPool = queries.createPipelineStatisticsQueryPool({.queryCount = queryCount});
    const auto meshStatisticsPool = queries.createPipelineStatisticsQueryPool({
        .mode = rachis::PipelineStatisticsQueryMode::meshShader,
        .queryCount = queryCount,
    });
    REQUIRE(timestampPool.has_value());
    REQUIRE(occlusionPool.has_value());
    REQUIRE(statisticsPool.has_value());
    REQUIRE(meshStatisticsPool.has_value());

    constexpr std::uint64_t destinationSize = 512;
    const auto readback = resources.createBuffer({
        .size = destinationSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    const auto deviceLocal = resources.createBuffer({
        .size = destinationSize,
        .usage = rachis::BufferUsage::copyDestination | rachis::BufferUsage::storage,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    REQUIRE(readback.has_value());
    REQUIRE(deviceLocal.has_value());
    const auto nativeBuffer = [&](rachis::BufferHandle buffer) {
        const auto resolved = resources.resolveBindingBuffer(buffer);
        REQUIRE(resolved.has_value());
        return resolved->buffer;
    };
    const auto readbackBuffer = nativeBuffer(*readback);
    const auto deviceLocalBuffer = nativeBuffer(*deviceLocal);

    enum class ResolveKind : std::uint8_t
    {
        timestamp,
        occlusion,
        pipelineStatistics,
        meshPipelineStatistics,
    };

    const auto recordResolve = [&](ResolveKind kind, rachis::BufferHandle destination) {
        auto cb = commands.acquireCommandBuffer(*cmdPool);
        REQUIRE(cb.has_value());
        auto& vkCb = static_cast<rachis::vulkan::VulkanCommandBuffer&>(**cb);
        for (std::uint32_t index = 0; index < queryCount; ++index)
        {
            switch (kind)
            {
            case ResolveKind::timestamp:
                REQUIRE(queries.writeTimestamp(vkCb, *timestampPool, index).has_value());
                break;
            case ResolveKind::occlusion:
                vkCb.graphicsState().renderingOwner = std::make_shared<rachis::vulkan::VulkanRenderingScope>();
                REQUIRE(queries.beginOcclusionQuery(vkCb, *occlusionPool, index).has_value());
                REQUIRE(queries.endOcclusionQuery(vkCb).has_value());
                vkCb.graphicsState().renderingOwner.reset();
                break;
            case ResolveKind::pipelineStatistics:
                REQUIRE(queries.beginPipelineStatisticsQuery(vkCb, *statisticsPool, index).has_value());
                REQUIRE(queries.endPipelineStatisticsQuery(vkCb).has_value());
                break;
            case ResolveKind::meshPipelineStatistics:
                REQUIRE(queries.beginPipelineStatisticsQuery(vkCb, *meshStatisticsPool, index).has_value());
                REQUIRE(queries.endPipelineStatisticsQuery(vkCb).has_value());
                break;
            }
        }
        {
            const std::scoped_lock lock(mock.state.mutex);
            mock.state.commandEvents.clear();
        }
        switch (kind)
        {
        case ResolveKind::timestamp:
            REQUIRE(queries
                        .resolveTimestamps(vkCb,
                                           {
                                               .queryPool = *timestampPool,
                                               .queryCount = queryCount,
                                               .destination = destination,
                                           })
                        .has_value());
            break;
        case ResolveKind::occlusion:
            REQUIRE(queries
                        .resolveOcclusionQueries(vkCb,
                                                 {
                                                     .queryPool = *occlusionPool,
                                                     .queryCount = queryCount,
                                                     .destination = destination,
                                                 })
                        .has_value());
            break;
        case ResolveKind::pipelineStatistics:
        case ResolveKind::meshPipelineStatistics:
            REQUIRE(queries
                        .resolvePipelineStatisticsQueries(vkCb,
                                                          {
                                                              .queryPool = kind == ResolveKind::meshPipelineStatistics
                                                                               ? *meshStatisticsPool
                                                                               : *statisticsPool,
                                                              .queryCount = queryCount,
                                                              .destination = destination,
                                                          })
                        .has_value());
            break;
        }
        std::vector<PerfCommandEvent> events;
        {
            const std::scoped_lock lock(mock.state.mutex);
            events = mock.state.commandEvents;
        }
        cb->reset();
        return events;
    };

    const auto isHostReadBarrier = [](const PerfCommandEvent& event) {
        return (event.kind == PerfCommandKind::memoryBarrier || event.kind == PerfCommandKind::bufferBarrier) &&
               ((event.dstStageMask & VK_PIPELINE_STAGE_2_HOST_BIT) != 0 ||
                (event.dstAccessMask & VK_ACCESS_2_HOST_READ_BIT) != 0);
    };

    for (const auto kind : {ResolveKind::timestamp,
                            ResolveKind::occlusion,
                            ResolveKind::pipelineStatistics,
                            ResolveKind::meshPipelineStatistics})
    {
        CAPTURE(static_cast<int>(kind));
        const std::size_t expectedCopies = kind == ResolveKind::meshPipelineStatistics ? 1 + queryCount : 1;

        const auto readbackEvents = recordResolve(kind, *readback);
        std::vector<std::size_t> copies;
        std::vector<std::size_t> resets;
        std::vector<std::size_t> hostBarriers;
        for (std::size_t index = 0; index < readbackEvents.size(); ++index)
        {
            const auto& event = readbackEvents[index];
            if (event.kind == PerfCommandKind::copyQueryResults)
            {
                CHECK(event.buffer == readbackBuffer);
                CHECK(event.resultFlags == (VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
                copies.push_back(index);
            }
            else if (event.kind == PerfCommandKind::resetQueryPool)
            {
                resets.push_back(index);
            }
            else if (isHostReadBarrier(event))
            {
                hostBarriers.push_back(index);
            }
        }
        REQUIRE(copies.size() == expectedCopies);
        REQUIRE_FALSE(resets.empty());
        CHECK(copies.back() < resets.front());
        REQUIRE(hostBarriers.size() == 1);
        const auto& hostBarrier = readbackEvents[hostBarriers.front()];
        CHECK(hostBarrier.kind == PerfCommandKind::bufferBarrier);
        CHECK(hostBarrier.buffer == readbackBuffer);
        CHECK(hostBarrier.offset == 0);
        CHECK(hostBarrier.size == destinationSize);
        CHECK(hostBarrier.srcStageMask == VK_PIPELINE_STAGE_2_COPY_BIT);
        CHECK(hostBarrier.srcAccessMask == VK_ACCESS_2_TRANSFER_WRITE_BIT);
        CHECK(hostBarrier.dstStageMask == VK_PIPELINE_STAGE_2_HOST_BIT);
        CHECK(hostBarrier.dstAccessMask == VK_ACCESS_2_HOST_READ_BIT);
        CHECK(hostBarriers.front() > copies.back());
        CHECK(hostBarriers.front() < resets.front());
        const auto resetOrder = std::ranges::find_if(readbackEvents, [](const PerfCommandEvent& event) {
            return event.kind == PerfCommandKind::memoryBarrier && event.srcStageMask == VK_PIPELINE_STAGE_2_COPY_BIT &&
                   event.srcAccessMask == VK_ACCESS_2_TRANSFER_WRITE_BIT &&
                   event.dstStageMask == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        });
        REQUIRE(resetOrder != readbackEvents.end());
        const auto resetOrderIndex = static_cast<std::size_t>(resetOrder - readbackEvents.begin());
        CHECK(resetOrderIndex > copies.back());
        CHECK(resetOrderIndex < resets.front());

        const auto deviceLocalEvents = recordResolve(kind, *deviceLocal);
        CHECK(std::ranges::count_if(deviceLocalEvents, [](const PerfCommandEvent& event) {
                  return event.kind == PerfCommandKind::copyQueryResults;
              }) == static_cast<std::ptrdiff_t>(expectedCopies));
        CHECK(std::ranges::none_of(deviceLocalEvents, isHostReadBarrier));
        CHECK(std::ranges::any_of(deviceLocalEvents, [&](const PerfCommandEvent& event) {
            return event.kind == PerfCommandKind::copyQueryResults && event.buffer == deviceLocalBuffer;
        }));
    }

    REQUIRE(resources.destroyBuffer(*readback).has_value());
    REQUIRE(resources.destroyBuffer(*deviceLocal).has_value());
    REQUIRE(queries.destroyQueryPool(*timestampPool).has_value());
    REQUIRE(queries.destroyQueryPool(*occlusionPool).has_value());
    REQUIRE(queries.destroyQueryPool(*statisticsPool).has_value());
    REQUIRE(queries.destroyQueryPool(*meshStatisticsPool).has_value());
    commands.shutdown();
    queries.shutdown();
    resources.shutdown();
}

TEST_CASE(
    "Vulkan usage-scoped sharing mode preserves DCC/HTILE on single-family usages and stays VUID-clean across queues",
    "[size:small][rhi][vulkan]")
{
    SECTION("3-family discrete topology ({0, 2, 4}): pure graphics is EXCLUSIVE, GFX+ACE excludes SDMA, "
            "transfer+compute is CONCURRENT")
    {
        ScopedPerfMock mock;
        rachis::vulkan::VulkanInstanceTable instanceTable{};
        VolkDeviceTable table{};
        populatePerfTables(instanceTable, table);

        const auto instance = fakeVk<VkInstance>(0x330);
        const auto physical = fakeVk<VkPhysicalDevice>(0x331);
        const auto device = fakeVk<VkDevice>(0x332);
        const rachis::vulkan::VulkanQueueFamilies threeFamilies{
            .graphics = 0,
            .compute = 2,
            .transfer = 4,
        };
        VkPhysicalDeviceMemoryProperties memProps{};
        perfGetPhysicalMemProps(physical, &memProps);

        rachis::vulkan::VulkanResourceAllocator resources;
        REQUIRE(resources
                    .initialize(instance,
                                &instanceTable,
                                physical,
                                device,
                                &table,
                                memProps,
                                threeFamilies,
                                fakeVk<VkQueue>(0x333),
                                64,
                                false,
                                false,
                                false,
                                false,
                                false,
                                true)
                    .has_value());

        {
            const std::scoped_lock lock(mock.state.mutex);
            mock.state.bufferSharingModes.clear();
            mock.state.bufferQueueFamilyCounts.clear();
            mock.state.bufferQueueFamilies.clear();
            mock.state.imageSharingModes.clear();
            mock.state.imageQueueFamilyCounts.clear();
            mock.state.imageQueueFamilies.clear();
        }

        const auto vertexBuf = resources.createBuffer({
            .size = 1024,
            .usage = rachis::BufferUsage::vertex | rachis::BufferUsage::index,
            .memoryType = rachis::MemoryType::deviceLocal,
        });

        const auto storageBuf = resources.createBuffer({
            .size = 1024,
            .usage = rachis::BufferUsage::storage,
            .memoryType = rachis::MemoryType::deviceLocal,
        });

        const auto xferBuf = resources.createBuffer({
            .size = 1024,
            .usage = rachis::BufferUsage::copyDestination | rachis::BufferUsage::storage,
            .memoryType = rachis::MemoryType::deviceLocal,
        });
        REQUIRE(vertexBuf.has_value());
        REQUIRE(storageBuf.has_value());
        REQUIRE(xferBuf.has_value());

        const auto rtTex = resources.createTexture({
            .width = 64,
            .height = 64,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::renderTarget,
        });

        const auto dsTex = resources.createTexture({
            .width = 64,
            .height = 64,
            .format = rachis::Format::d32Float,
            .usage = rachis::TextureUsage::depthStencil,
        });

        const auto vrsTex = resources.createTexture({
            .width = 16,
            .height = 16,
            .format = rachis::Format::r8Uint,
            .usage = rachis::TextureUsage::shadingRateSource,
        });

        const auto gbufferTex = resources.createTexture({
            .width = 64,
            .height = 64,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::sampled,
        });

        const auto xferTex = resources.createTexture({
            .width = 32,
            .height = 32,
            .mipLevels = 2,
            .format = rachis::Format::rgba8Unorm,
            .usage =
                rachis::TextureUsage::copyDestination | rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        });
        REQUIRE(rtTex.has_value());
        REQUIRE(dsTex.has_value());
        REQUIRE(vrsTex.has_value());
        REQUIRE(gbufferTex.has_value());
        REQUIRE(xferTex.has_value());

        {
            const std::scoped_lock lock(mock.state.mutex);
            REQUIRE(mock.state.bufferSharingModes.size() >= 3U);

            CHECK(mock.state.bufferSharingModes.front() == VK_SHARING_MODE_EXCLUSIVE);
            CHECK(mock.state.bufferQueueFamilyCounts.front() == 0U);
            CHECK(mock.state.bufferQueueFamilies.front().empty());

            bool sawTwoFamilyBuffer = false;
            bool sawThreeFamilyBuffer = false;
            for (std::size_t i = 0; i < mock.state.bufferSharingModes.size(); ++i)
            {
                if (mock.state.bufferQueueFamilyCounts[i] == 2U)
                {
                    sawTwoFamilyBuffer = true;
                    CHECK(mock.state.bufferSharingModes[i] == VK_SHARING_MODE_CONCURRENT);
                    CHECK(mock.state.bufferQueueFamilies[i] == std::vector<std::uint32_t>{0U, 2U});
                }
                else if (mock.state.bufferQueueFamilyCounts[i] == 3U)
                {
                    sawThreeFamilyBuffer = true;
                    CHECK(mock.state.bufferSharingModes[i] == VK_SHARING_MODE_CONCURRENT);
                    CHECK(mock.state.bufferQueueFamilies[i] == std::vector<std::uint32_t>{0U, 2U, 4U});
                }
            }
            CHECK(sawTwoFamilyBuffer);
            CHECK(sawThreeFamilyBuffer);

            std::size_t exclusiveImages = 0;
            bool sawGfxComputeImageWithoutSdma = false;
            bool sawThreeFamilyImage = false;
            for (std::size_t i = 0; i < mock.state.imageSharingModes.size(); ++i)
            {
                if (mock.state.imageSharingModes[i] == VK_SHARING_MODE_EXCLUSIVE)
                {
                    CHECK(mock.state.imageQueueFamilyCounts[i] == 0U);
                    CHECK(mock.state.imageQueueFamilies[i].empty());
                    ++exclusiveImages;
                }
                else if (mock.state.imageQueueFamilyCounts[i] == 2U)
                {
                    sawGfxComputeImageWithoutSdma = true;
                    CHECK(mock.state.imageQueueFamilies[i] == std::vector<std::uint32_t>{0U, 2U});
                }
                else if (mock.state.imageQueueFamilyCounts[i] == 3U)
                {
                    sawThreeFamilyImage = true;
                    CHECK(mock.state.imageQueueFamilies[i] == std::vector<std::uint32_t>{0U, 2U, 4U});
                }
            }
            CHECK(exclusiveImages >= 3U);
            CHECK(sawGfxComputeImageWithoutSdma);
            CHECK(sawThreeFamilyImage);
        }

        const auto copyCmd = fakeVk<VkCommandBuffer>(0x601);
        const auto abandonedCmd = fakeVk<VkCommandBuffer>(0x602);
        const auto gfxCmd = fakeVk<VkCommandBuffer>(0x603);
        const auto compCmd = fakeVk<VkCommandBuffer>(0x604);

        const auto badBufBarrier = resources.bufferBarrier(copyCmd,
                                                           rachis::QueueType::copy,
                                                           {
                                                               .buffer = *vertexBuf,
                                                               .stagesBefore = rachis::PipelineStage::all,
                                                               .stagesAfter = rachis::PipelineStage::all,
                                                               .accessBefore = rachis::Access::common,
                                                               .accessAfter = rachis::Access::common,
                                                           });
        REQUIRE_FALSE(badBufBarrier.has_value());
        CHECK(badBufBarrier.error().code == rachis::ErrorCode::invalidArgument);

        const auto badTexBarrier = resources.textureBarrier(copyCmd,
                                                            rachis::QueueType::copy,
                                                            {
                                                                .texture = *rtTex,
                                                                .stagesBefore = rachis::PipelineStage::all,
                                                                .stagesAfter = rachis::PipelineStage::all,
                                                                .accessBefore = rachis::Access::common,
                                                                .accessAfter = rachis::Access::common,
                                                                .layoutBefore = rachis::TextureLayout::undefined,
                                                                .layoutAfter = rachis::TextureLayout::common,
                                                            });
        REQUIRE_FALSE(badTexBarrier.has_value());
        CHECK(badTexBarrier.error().code == rachis::ErrorCode::invalidArgument);

        REQUIRE(resources
                    .bufferBarrier(copyCmd,
                                   rachis::QueueType::copy,
                                   {
                                       .buffer = *xferBuf,
                                       .stagesBefore = rachis::PipelineStage::none,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::none,
                                       .accessAfter = rachis::Access::copyDestination,
                                   })
                    .has_value());
        REQUIRE(resources
                    .textureBarrier(copyCmd,
                                    rachis::QueueType::copy,
                                    {
                                        .texture = *xferTex,
                                        .stagesBefore = rachis::PipelineStage::none,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                        .subresources = {.firstMipLevel = 0, .mipLevelCount = 1},
                                    })
                    .has_value());
        resources.commitCommandLayouts(copyCmd);

        REQUIRE(resources
                    .textureBarrier(abandonedCmd,
                                    rachis::QueueType::compute,
                                    {
                                        .texture = *xferTex,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::computeShader,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::unorderedAccess,
                                        .layoutBefore = rachis::TextureLayout::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::unorderedAccess,
                                        .subresources = {.firstMipLevel = 0, .mipLevelCount = 1},
                                    })
                    .has_value());
        resources.discardCommandLayouts(abandonedCmd);

        REQUIRE(resources
                    .bufferBarrier(gfxCmd,
                                   rachis::QueueType::graphics,
                                   {
                                       .buffer = *xferBuf,
                                       .stagesBefore = rachis::PipelineStage::copy,
                                       .stagesAfter = rachis::PipelineStage::computeShader,
                                       .accessBefore = rachis::Access::copyDestination,
                                       .accessAfter = rachis::Access::shaderResource,
                                   })
                    .has_value());
        REQUIRE(resources
                    .textureBarrier(gfxCmd,
                                    rachis::QueueType::graphics,
                                    {
                                        .texture = *xferTex,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::computeShader,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::shaderResource,
                                        .layoutBefore = rachis::TextureLayout::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::shaderResource,
                                        .subresources = {.firstMipLevel = 0, .mipLevelCount = 1},
                                    })
                    .has_value());
        resources.commitCommandLayouts(gfxCmd);

        REQUIRE(resources
                    .textureBarrier(compCmd,
                                    rachis::QueueType::compute,
                                    {
                                        .texture = *xferTex,
                                        .stagesBefore = rachis::PipelineStage::none,
                                        .stagesAfter = rachis::PipelineStage::computeShader,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::unorderedAccess,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::unorderedAccess,
                                        .subresources = {.firstMipLevel = 1, .mipLevelCount = 1},
                                    })
                    .has_value());
        resources.commitCommandLayouts(compCmd);

        for (const auto& b : mock.state.capturedBufferBarriers)
        {
            CHECK(b.srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
            CHECK(b.dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
        }
        for (const auto& img : mock.state.capturedImageBarriers)
        {
            CHECK(img.srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
            CHECK(img.dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
        }

        resources.shutdown();
    }

    SECTION("2-family ({0, 0, 4}) and 1-family ({0, 0, 0}) topologies deduplicate graphics==compute to EXCLUSIVE")
    {
        ScopedPerfMock mock;
        rachis::vulkan::VulkanInstanceTable instanceTable{};
        VolkDeviceTable table{};
        populatePerfTables(instanceTable, table);

        const auto instance = fakeVk<VkInstance>(0x334);
        const auto physical = fakeVk<VkPhysicalDevice>(0x335);
        const auto device = fakeVk<VkDevice>(0x336);
        VkPhysicalDeviceMemoryProperties memProps{};
        perfGetPhysicalMemProps(physical, &memProps);

        rachis::vulkan::VulkanResourceAllocator twoFamilyResources;
        REQUIRE(twoFamilyResources
                    .initialize(instance,
                                &instanceTable,
                                physical,
                                device,
                                &table,
                                memProps,
                                {.graphics = 0, .compute = 0, .transfer = 4},
                                fakeVk<VkQueue>(0x337),
                                64,
                                false)
                    .has_value());

        {
            const std::scoped_lock lock(mock.state.mutex);
            mock.state.bufferSharingModes.clear();
            mock.state.bufferQueueFamilyCounts.clear();
            mock.state.bufferQueueFamilies.clear();
            mock.state.imageSharingModes.clear();
            mock.state.imageQueueFamilyCounts.clear();
            mock.state.imageQueueFamilies.clear();
        }

        REQUIRE(twoFamilyResources
                    .createBuffer({
                        .size = 512,
                        .usage = rachis::BufferUsage::constant | rachis::BufferUsage::storage,
                        .memoryType = rachis::MemoryType::deviceLocal,
                    })
                    .has_value());
        REQUIRE(twoFamilyResources
                    .createTexture({
                        .width = 64,
                        .height = 64,
                        .format = rachis::Format::rgba8Unorm,
                        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::sampled |
                                 rachis::TextureUsage::storage,
                    })
                    .has_value());

        {
            const std::scoped_lock lock(mock.state.mutex);
            REQUIRE_FALSE(mock.state.bufferSharingModes.empty());
            REQUIRE_FALSE(mock.state.imageSharingModes.empty());
            for (const auto mode : mock.state.bufferSharingModes)
            {
                CHECK(mode == VK_SHARING_MODE_EXCLUSIVE);
            }
            for (const auto mode : mock.state.imageSharingModes)
            {
                CHECK(mode == VK_SHARING_MODE_EXCLUSIVE);
            }
        }
        twoFamilyResources.shutdown();

        rachis::vulkan::VulkanResourceAllocator singleFamilyResources;
        REQUIRE(singleFamilyResources
                    .initialize(instance,
                                &instanceTable,
                                physical,
                                device,
                                &table,
                                memProps,
                                {.graphics = 0, .compute = 0, .transfer = 0},
                                fakeVk<VkQueue>(0x338),
                                64,
                                false)
                    .has_value());
        {
            const std::scoped_lock lock(mock.state.mutex);
            mock.state.bufferSharingModes.clear();
            mock.state.imageSharingModes.clear();
        }

        REQUIRE(singleFamilyResources
                    .createBuffer({
                        .size = 512,
                        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination |
                                 rachis::BufferUsage::storage,
                        .memoryType = rachis::MemoryType::deviceLocal,
                    })
                    .has_value());
        REQUIRE(singleFamilyResources
                    .createTexture({
                        .width = 64,
                        .height = 64,
                        .format = rachis::Format::rgba8Unorm,
                        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                                 rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
                    })
                    .has_value());
        {
            const std::scoped_lock lock(mock.state.mutex);
            for (const auto mode : mock.state.bufferSharingModes)
            {
                CHECK(mode == VK_SHARING_MODE_EXCLUSIVE);
            }
            for (const auto mode : mock.state.imageSharingModes)
            {
                CHECK(mode == VK_SHARING_MODE_EXCLUSIVE);
            }
        }
        singleFamilyResources.shutdown();
    }
}

TEST_CASE("VulkanResourceAllocator checks textures with declared view formats against native image-format limits",
          "[size:small][rhi][vulkan]")
{
    ScopedPerfMock mock;
    rachis::vulkan::VulkanInstanceTable instanceTable{};
    VolkDeviceTable table{};
    populatePerfTables(instanceTable, table);
    instanceTable.vkGetPhysicalDeviceImageFormatProperties2 = &perfGetImageFormatProps2;

    const auto instance = fakeVk<VkInstance>(0x34C);
    const auto physical = fakeVk<VkPhysicalDevice>(0x34D);
    const auto device = fakeVk<VkDevice>(0x34E);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    VkPhysicalDeviceMemoryProperties memProps{};
    perfGetPhysicalMemProps(physical, &memProps);

    rachis::vulkan::VulkanResourceAllocator resources;
    REQUIRE(resources
                .initialize(instance,
                            &instanceTable,
                            physical,
                            device,
                            &table,
                            memProps,
                            families,
                            fakeVk<VkQueue>(0x34F),
                            64,
                            false)
                .has_value());

    const std::array linearView{rachis::Format::rgba8Unorm};
    const rachis::TextureDesc srgbStorage{
        .width = 16,
        .height = 16,
        .mipLevels = 2,
        .format = rachis::Format::rgba8UnormSrgb,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage | rachis::TextureUsage::copyDestination,
        .viewFormats = linearView,
        .label = "sRGB storage with linear view",
    };
    REQUIRE(resources.queryTextureMemoryRequirements(srgbStorage).has_value());
    CHECK(mock.state.imageFormatProps2Calls == 1U);
    CHECK((mock.state.imageFormatQueryFlags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) != 0);
    CHECK((mock.state.imageFormatQueryFlags & VK_IMAGE_CREATE_EXTENDED_USAGE_BIT) != 0);
    CHECK((mock.state.imageFormatQueryUsage & VK_IMAGE_USAGE_STORAGE_BIT) != 0);
    CHECK(mock.state.imageFormatQueryViewFormats ==
          std::vector<VkFormat>{VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R8G8B8A8_UNORM});

    auto withoutViews = srgbStorage;
    withoutViews.viewFormats = {};
    REQUIRE(resources.queryTextureMemoryRequirements(withoutViews).has_value());
    CHECK(mock.state.imageFormatProps2Calls == 1U);

    const auto imagesBefore = mock.state.createImageCalls.load();
    mock.state.imageFormatQueryMaxMipLevels = 1;
    const auto tooManyMips = resources.queryTextureMemoryRequirements(srgbStorage);
    REQUIRE_FALSE(tooManyMips.has_value());
    CHECK(tooManyMips.error().code == rachis::ErrorCode::unsupported);
    const auto tooManyMipsTexture = resources.createTexture(srgbStorage);
    REQUIRE_FALSE(tooManyMipsTexture.has_value());
    CHECK(tooManyMipsTexture.error().code == rachis::ErrorCode::unsupported);

    mock.state.imageFormatQueryMaxMipLevels = 15;
    mock.state.imageFormatQueryResult = VK_ERROR_FORMAT_NOT_SUPPORTED;
    const auto refused = resources.createTexture(srgbStorage);
    REQUIRE_FALSE(refused.has_value());
    CHECK(refused.error().code == rachis::ErrorCode::unsupported);
    CHECK(mock.state.createImageCalls.load() == imagesBefore);

    mock.state.imageFormatQueryResult = VK_SUCCESS;
    const auto accepted = resources.createTexture(srgbStorage);
    REQUIRE(accepted.has_value());
    REQUIRE(resources.destroyTexture(*accepted).has_value());

    resources.shutdown();
}

TEST_CASE(
    "Vulkan 1.3 vkGetDeviceBuffer/ImageMemoryRequirements eliminates dummy resource creation with legacy fallback",
    "[size:small][rhi][vulkan]")
{
    SECTION("Null Vulkan 1.3 device memory requirement pointers fall back cleanly to VkBuffer/VkImage queries")
    {
        ScopedPerfMock mock;
        rachis::vulkan::VulkanInstanceTable instanceTable{};
        VolkDeviceTable table{};
        populatePerfTables(instanceTable, table);
        table.vkGetDeviceBufferMemoryRequirements = nullptr;
        table.vkGetDeviceBufferMemoryRequirementsKHR = nullptr;
        table.vkGetDeviceImageMemoryRequirements = nullptr;
        table.vkGetDeviceImageMemoryRequirementsKHR = nullptr;

        const auto instance = fakeVk<VkInstance>(0x344);
        const auto physical = fakeVk<VkPhysicalDevice>(0x345);
        const auto device = fakeVk<VkDevice>(0x346);
        const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
        VkPhysicalDeviceMemoryProperties memProps{};
        perfGetPhysicalMemProps(physical, &memProps);

        rachis::vulkan::VulkanResourceAllocator resources;
        REQUIRE(resources
                    .initialize(instance,
                                &instanceTable,
                                physical,
                                device,
                                &table,
                                memProps,
                                families,
                                fakeVk<VkQueue>(0x347),
                                64,
                                false)
                    .has_value());

        mock.state.createBufferCalls.store(0);
        mock.state.destroyBufferCalls.store(0);
        mock.state.createImageCalls.store(0);
        mock.state.destroyImageCalls.store(0);

        REQUIRE(resources
                    .queryBufferMemoryRequirements({
                        .size = 1024,
                        .usage = rachis::BufferUsage::storage,
                    })
                    .has_value());
        REQUIRE(resources
                    .queryTextureMemoryRequirements({
                        .width = 32,
                        .height = 32,
                        .format = rachis::Format::rgba8Unorm,
                        .usage = rachis::TextureUsage::sampled,
                    })
                    .has_value());

        CHECK(mock.state.createBufferCalls.load() == 1U);
        CHECK(mock.state.destroyBufferCalls.load() == 1U);
        CHECK(mock.state.createImageCalls.load() == 1U);
        CHECK(mock.state.destroyImageCalls.load() == 1U);
        CHECK(mock.state.legacyBufferMemReq2Calls.load() == 1U);
        CHECK(mock.state.legacyImageMemReq2Calls.load() == 1U);

        resources.shutdown();
    }
}

TEST_CASE("Vulkan placed textures on aliasing heaps use the requirements of their aliasable image",
          "[size:small][rhi][vulkan]")
{
    ScopedPerfMock mock;
    rachis::vulkan::VulkanInstanceTable instanceTable{};
    VolkDeviceTable table{};
    populatePerfTables(instanceTable, table);

    const auto instance = fakeVk<VkInstance>(0x354);
    const auto physical = fakeVk<VkPhysicalDevice>(0x355);
    const auto device = fakeVk<VkDevice>(0x356);
    const rachis::vulkan::VulkanQueueFamilies families{.graphics = 0, .compute = 0, .transfer = 0};
    VkPhysicalDeviceMemoryProperties memProps{};
    perfGetPhysicalMemProps(physical, &memProps);

    rachis::vulkan::VulkanResourceAllocator resources;
    REQUIRE(resources
                .initialize(instance,
                            &instanceTable,
                            physical,
                            device,
                            &table,
                            memProps,
                            families,
                            fakeVk<VkQueue>(0x357),
                            64,
                            false)
                .has_value());

    constexpr std::uint64_t plainSize = 65536;
    constexpr std::uint64_t aliasSize = 3 * plainSize;
    mock.state.aliasImageSize = aliasSize;
    const rachis::TextureDesc desc{
        .width = 64,
        .height = 64,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
        .label = "Vulkan aliasable placed texture",
    };
    const auto portable = resources.queryTextureMemoryRequirements(desc);
    REQUIRE(portable.has_value());
    CHECK(portable->size == plainSize);

    const auto aliasingHeap = resources.createResourceHeap({
        .size = 8 * plainSize,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::nonAttachmentTexture,
        .flags = rachis::ResourceHeapFlag::allowAliasing,
    });
    const auto plainHeap = resources.createResourceHeap({
        .size = 8 * plainSize,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::nonAttachmentTexture,
    });
    REQUIRE(aliasingHeap.has_value());
    REQUIRE(plainHeap.has_value());

    const auto first = resources.createPlacedTexture({.heap = *aliasingHeap, .offset = 0}, desc);
    const auto second = resources.createPlacedTexture({.heap = *aliasingHeap, .offset = 2 * plainSize}, desc);
    auto plainDesc = desc;
    plainDesc.label = "Vulkan plain placed texture";
    const auto plain = resources.createPlacedTexture({.heap = *plainHeap, .offset = 0}, plainDesc);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    REQUIRE(plain.has_value());

    const auto snapshot = resources.captureResourceMemorySnapshot();
    REQUIRE(snapshot.has_value());
    std::size_t aliasEntries = 0;
    std::size_t plainEntries = 0;
    for (const auto& entry : snapshot->entries)
    {
        if (entry.label == desc.label)
        {
            ++aliasEntries;
            CHECK(entry.allocationBytes == aliasSize);
        }
        else if (entry.label == plainDesc.label)
        {
            ++plainEntries;
            CHECK(entry.allocationBytes == plainSize);
        }
    }
    CHECK(aliasEntries == 2);
    CHECK(plainEntries == 1);

    mock.state.aliasImageRequiresDedicated = true;
    const auto createImageCallsBefore = mock.state.createImageCalls.load();
    const auto dedicatedAlias = resources.createPlacedTexture({.heap = *aliasingHeap, .offset = 4 * plainSize}, desc);
    REQUIRE_FALSE(dedicatedAlias.has_value());
    CHECK(dedicatedAlias.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(mock.state.createImageCalls.load() == createImageCallsBefore);
    const auto plainAgain = resources.createPlacedTexture({.heap = *plainHeap, .offset = 2 * plainSize}, plainDesc);
    REQUIRE(plainAgain.has_value());

    REQUIRE(resources.destroyTexture(*first).has_value());
    REQUIRE(resources.destroyTexture(*second).has_value());
    REQUIRE(resources.destroyTexture(*plain).has_value());
    REQUIRE(resources.destroyTexture(*plainAgain).has_value());
    REQUIRE(resources.destroyResourceHeap(*aliasingHeap).has_value());
    REQUIRE(resources.destroyResourceHeap(*plainHeap).has_value());
    resources.shutdown();
}

TEST_CASE("VulkanSwapchainManager deduplicates WSI capability queries while preserving live XCB extent transitions",
          "[size:small][rhi][vulkan]")
{
    SECTION("Wayland undefined-extent swapchains avoid steady-state capability queries")
    {
        ScopedPerfMock mock;
        mock.state.surfaceCurrentExtent = {0xffffffffU, 0xffffffffU};
        rachis::vulkan::VulkanPerformanceSwapchainTestAccess fixture(rachis::SurfaceKind::wayland);

        mock.state.surfaceCapsQueries.store(0);
        const auto swapchain = fixture.swapchains.createSwapchain({
            .surface = fixture.surfaceHandle,
            .width = 128,
            .height = 128,
            .imageCount = 2,
            .format = rachis::Format::rgba8Unorm,
            .colorSpace = rachis::SurfaceColorSpace::srgbNonlinear,
            .usage = rachis::TextureUsage::renderTarget,
            .presentMode = rachis::PresentMode::fifo,
        });
        REQUIRE(swapchain.has_value());

        CHECK(mock.state.surfaceCapsQueries.load() == 1U);

        mock.state.surfaceCapsQueries.store(0);
        for (int frame = 0; frame < 10; ++frame)
        {
            const auto acquired = fixture.swapchains.acquireNextImage(*swapchain);
            REQUIRE(acquired.has_value());
            const auto point = fixture.submitEmptyGraphicsWork();
            bool injectedLoss = false;
            const auto presented = fixture.swapchains.present(*swapchain, point, injectedLoss);
            REQUIRE(presented.has_value());
            CHECK(*presented == rachis::PresentStatus::presented);
            const auto timing = fixture.swapchains.queryFrameTiming(*swapchain);
            REQUIRE(timing.has_value());
            CHECK_FALSE(timing->hidden);
        }
        CHECK(mock.state.surfaceCapsQueries.load() == 0U);

        REQUIRE(fixture.swapchains.resizeSwapchain(*swapchain, 256, 256).has_value());
        CHECK(mock.state.surfaceCapsQueries.load() == 1U);
        {
            const auto acquired = fixture.swapchains.acquireNextImage(*swapchain);
            REQUIRE(acquired.has_value());
            CHECK(acquired->width == 256U);
            const auto point = fixture.submitEmptyGraphicsWork();
            bool injectedLoss = false;
            const auto presented = fixture.swapchains.present(*swapchain, point, injectedLoss);
            REQUIRE(presented.has_value());
            CHECK(*presented == rachis::PresentStatus::presented);
        }
        CHECK(mock.state.surfaceCapsQueries.load() == 1U);
        REQUIRE(fixture.swapchains.destroySwapchain(*swapchain).has_value());
    }

    SECTION("XCB non-maintenance1 present short-circuits unreachable pre-present hidden check")
    {
        ScopedPerfMock mock;
        mock.state.surfaceCurrentExtent = {128, 128};
        rachis::vulkan::VulkanPerformanceSwapchainTestAccess fixture(
            rachis::SurfaceKind::xcb, rachis::VulkanPresentCompletionStrategy::imageLifetimeAssociation, false);

        mock.state.surfaceCapsQueries.store(0);
        const auto swapchain = fixture.swapchains.createSwapchain({
            .surface = fixture.surfaceHandle,
            .width = 128,
            .height = 128,
            .imageCount = 2,
            .format = rachis::Format::rgba8Unorm,
            .colorSpace = rachis::SurfaceColorSpace::srgbNonlinear,
            .usage = rachis::TextureUsage::renderTarget,
            .presentMode = rachis::PresentMode::fifo,
        });
        REQUIRE(swapchain.has_value());
        CHECK(mock.state.surfaceCapsQueries.load() == 1U);

        mock.state.surfaceCapsQueries.store(0);
        const auto acquired = fixture.swapchains.acquireNextImage(*swapchain);
        REQUIRE(acquired.has_value());
        CHECK(mock.state.surfaceCapsQueries.load() == 1U);

        mock.state.surfaceCapsQueries.store(0);
        const auto point = fixture.submitEmptyGraphicsWork();
        bool injectedLoss = false;
        const auto presented = fixture.swapchains.present(*swapchain, point, injectedLoss);
        REQUIRE(presented.has_value());
        CHECK(*presented == rachis::PresentStatus::presented);

        CHECK(mock.state.surfaceCapsQueries.load() == 1U);

        REQUIRE(fixture.swapchains.destroySwapchain(*swapchain).has_value());
    }

    SECTION("XCB maintenance1 swapchains detect live nonzero-to-zero and zero-to-nonzero transitions")
    {
        ScopedPerfMock mock;
        mock.state.surfaceCurrentExtent = {128, 128};
        rachis::vulkan::VulkanPerformanceSwapchainTestAccess fixture(
            rachis::SurfaceKind::xcb, rachis::VulkanPresentCompletionStrategy::swapchainMaintenancePresentFence, true);

        const auto swapchain = fixture.swapchains.createSwapchain({
            .surface = fixture.surfaceHandle,
            .width = 128,
            .height = 128,
            .imageCount = 2,
            .format = rachis::Format::rgba8Unorm,
            .colorSpace = rachis::SurfaceColorSpace::srgbNonlinear,
            .usage = rachis::TextureUsage::renderTarget,
            .presentMode = rachis::PresentMode::fifo,
        });
        REQUIRE(swapchain.has_value());

        const auto acquired = fixture.swapchains.acquireNextImage(*swapchain);
        REQUIRE(acquired.has_value());
        {
            const auto visibleTiming = fixture.swapchains.queryFrameTiming(*swapchain);
            REQUIRE(visibleTiming.has_value());
            CHECK_FALSE(visibleTiming->hidden);
        }

        {
            const std::scoped_lock lock(mock.state.mutex);
            mock.state.surfaceCurrentExtent = {0, 0};
        }
        {
            const auto minimizedTiming = fixture.swapchains.queryFrameTiming(*swapchain);
            REQUIRE(minimizedTiming.has_value());
            CHECK(minimizedTiming->hidden);
        }

        mock.state.queuePresentCalls.store(0);
        mock.state.releaseSwapchainImagesCalls.store(0);
        const auto point = fixture.submitEmptyGraphicsWork();
        bool injectedLoss = false;
        const auto occludedPresent = fixture.swapchains.present(*swapchain, point, injectedLoss);
        REQUIRE(occludedPresent.has_value());
        CHECK(*occludedPresent == rachis::PresentStatus::occluded);
        CHECK(mock.state.releaseSwapchainImagesCalls.load() == 1U);
        CHECK(mock.state.queuePresentCalls.load() == 0U);

        {
            const std::scoped_lock lock(mock.state.mutex);
            mock.state.surfaceCurrentExtent = {128, 128};
        }
        {
            const auto restoredTiming = fixture.swapchains.queryFrameTiming(*swapchain);
            REQUIRE(restoredTiming.has_value());
            CHECK_FALSE(restoredTiming->hidden);
        }
        {
            const auto restoredAcquire = fixture.swapchains.acquireNextImage(*swapchain);
            REQUIRE(restoredAcquire.has_value());
            const auto restoredPoint = fixture.submitEmptyGraphicsWork();
            const auto restoredPresent = fixture.swapchains.present(*swapchain, restoredPoint, injectedLoss);
            REQUIRE(restoredPresent.has_value());
            CHECK(*restoredPresent == rachis::PresentStatus::presented);
            CHECK(mock.state.queuePresentCalls.load() == 1U);
        }

        REQUIRE(fixture.swapchains.destroySwapchain(*swapchain).has_value());
    }
}

TEST_CASE("VulkanSwapchainManager destroys the swapchain that a failed resize retired", "[size:small][rhi][vulkan]")
{
    ScopedPerfMock mock;
    rachis::vulkan::VulkanPerformanceSwapchainTestAccess fixture;
    const auto swapchain = fixture.swapchains.createSwapchain({
        .surface = fixture.surfaceHandle,
        .width = 128,
        .height = 128,
        .imageCount = 2,
        .format = rachis::Format::rgba8Unorm,
        .colorSpace = rachis::SurfaceColorSpace::srgbNonlinear,
        .usage = rachis::TextureUsage::renderTarget,
        .presentMode = rachis::PresentMode::fifo,
    });
    REQUIRE(swapchain.has_value());
    {
        const auto acquired = fixture.swapchains.acquireNextImage(*swapchain);
        REQUIRE(acquired.has_value());
        const auto point = fixture.submitEmptyGraphicsWork();
        bool injectedLoss = false;
        const auto presented = fixture.swapchains.present(*swapchain, point, injectedLoss);
        REQUIRE(presented.has_value());
        CHECK(*presented == rachis::PresentStatus::presented);
    }

    VkSwapchainKHR retired = VK_NULL_HANDLE;
    {
        const std::scoped_lock lock(mock.state.mutex);
        REQUIRE(mock.state.createdSwapchains.size() == 1U);
        retired = mock.state.createdSwapchains.front();
        mock.state.nextCreateSwapchainResult = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    const auto failed = fixture.swapchains.resizeSwapchain(*swapchain, 128, 128);
    REQUIRE_FALSE(failed.has_value());
    CHECK(failed.error().code == rachis::ErrorCode::outOfMemory);
    CHECK(failed.error().nativeCode == static_cast<std::int64_t>(VK_ERROR_OUT_OF_DEVICE_MEMORY));
    {
        const std::scoped_lock lock(mock.state.mutex);
        REQUIRE(mock.state.oldSwapchains.size() == 2U);
        CHECK(mock.state.oldSwapchains.back() == retired);
        CHECK(std::ranges::count(mock.state.destroyedSwapchains, retired) == 1);
    }
    const auto refused = fixture.swapchains.acquireNextImage(*swapchain);
    REQUIRE_FALSE(refused.has_value());
    CHECK(refused.error().code == rachis::ErrorCode::invalidState);

    SECTION("A retried resize passes no retired oldSwapchain and presents again")
    {
        REQUIRE(fixture.swapchains.resizeSwapchain(*swapchain, 128, 128).has_value());
        {
            const std::scoped_lock lock(mock.state.mutex);
            REQUIRE(mock.state.oldSwapchains.size() == 3U);
            CHECK(mock.state.oldSwapchains.back() == VK_NULL_HANDLE);
        }
        const auto acquired = fixture.swapchains.acquireNextImage(*swapchain);
        REQUIRE(acquired.has_value());
        const auto point = fixture.submitEmptyGraphicsWork();
        bool injectedLoss = false;
        const auto presented = fixture.swapchains.present(*swapchain, point, injectedLoss);
        REQUIRE(presented.has_value());
        CHECK(*presented == rachis::PresentStatus::presented);
        REQUIRE(fixture.swapchains.destroySwapchain(*swapchain).has_value());
    }

    SECTION("destroySwapchain right after the failed resize succeeds")
    {
        REQUIRE(fixture.swapchains.destroySwapchain(*swapchain).has_value());
    }

    const std::scoped_lock lock(mock.state.mutex);
    CHECK(std::ranges::count(mock.state.destroyedSwapchains, retired) == 1);
}

TEST_CASE("VulkanSwapchainManager retires a fenced XCB swapchain when the bounded present wait reports a loss",
          "[size:small][rhi][vulkan][present-wait]")
{
    ScopedPerfMock mock;
    rachis::vulkan::VulkanPerformanceSwapchainTestAccess fixture(
        rachis::SurfaceKind::xcb,
        rachis::VulkanPresentCompletionStrategy::swapchainMaintenancePresentFence,
        true,
        true);
    const auto swapchain = fixture.swapchains.createSwapchain({
        .surface = fixture.surfaceHandle,
        .width = 128,
        .height = 128,
        .imageCount = 2,
        .format = rachis::Format::rgba8Unorm,
        .colorSpace = rachis::SurfaceColorSpace::srgbNonlinear,
        .usage = rachis::TextureUsage::renderTarget,
        .presentMode = rachis::PresentMode::fifo,
    });
    REQUIRE(swapchain.has_value());
    const auto presentFrame = [&fixture, &swapchain] {
        const auto acquired = fixture.swapchains.acquireNextImage(*swapchain);
        REQUIRE(acquired.has_value());
        const auto point = fixture.submitEmptyGraphicsWork();
        bool injectedLoss = false;
        const auto presented = fixture.swapchains.present(*swapchain, point, injectedLoss);
        REQUIRE(presented.has_value());
        CHECK(*presented == rachis::PresentStatus::presented);
    };
    for (int frame = 0; frame < 3; ++frame)
    {
        presentFrame();
    }
    VkSwapchainKHR original = VK_NULL_HANDLE;
    {
        const std::scoped_lock lock(mock.state.mutex);
        CHECK(mock.state.waitedPresentIds.empty());
        REQUIRE(mock.state.createdSwapchains.size() == 1U);
        original = mock.state.createdSwapchains.front();
        mock.state.waitForPresentResult = VK_ERROR_DEVICE_LOST;
    }

    SECTION("destroySwapchain succeeds while the queues stay healthy")
    {
        const auto destroyed = fixture.swapchains.destroySwapchain(*swapchain);
        REQUIRE(destroyed.has_value());
        const std::scoped_lock lock(mock.state.mutex);
        CHECK(mock.state.waitedPresentIds == std::vector<std::uint64_t>{3});
        CHECK(mock.state.waitedPresentTimeouts == std::vector<std::uint64_t>{250'000'000});
        CHECK(std::ranges::count(mock.state.destroyedSwapchains, original) == 1);
    }

    SECTION("resizeSwapchain succeeds and the new swapchain presents and retires again")
    {
        REQUIRE(fixture.swapchains.resizeSwapchain(*swapchain, 128, 128).has_value());
        {
            const std::scoped_lock lock(mock.state.mutex);
            CHECK(mock.state.waitedPresentIds == std::vector<std::uint64_t>{3});
            REQUIRE(mock.state.createdSwapchains.size() == 2U);
            CHECK(mock.state.oldSwapchains.back() == original);
        }
        presentFrame();
        REQUIRE(fixture.swapchains.destroySwapchain(*swapchain).has_value());
        const std::scoped_lock lock(mock.state.mutex);
        CHECK(mock.state.waitedPresentIds == std::vector<std::uint64_t>{3, 1});
    }

    SECTION("destroySwapchain still reports a loss that the queue idle wait sees")
    {
        mock.state.queueWaitIdleResult.store(VK_ERROR_DEVICE_LOST);
        const auto lost = fixture.swapchains.destroySwapchain(*swapchain);
        REQUIRE_FALSE(lost.has_value());
        CHECK(lost.error().code == rachis::ErrorCode::deviceLost);
        CHECK(lost.error().nativeCode == static_cast<std::int64_t>(VK_ERROR_DEVICE_LOST));
        CHECK(lost.error().message.find("vkQueueWaitIdle") != std::string::npos);
        mock.state.queueWaitIdleResult.store(VK_SUCCESS);
        {
            const std::scoped_lock lock(mock.state.mutex);
            CHECK(mock.state.destroyedSwapchains.empty());
        }
        REQUIRE(fixture.swapchains.destroySwapchain(*swapchain).has_value());
    }
}

#endif
