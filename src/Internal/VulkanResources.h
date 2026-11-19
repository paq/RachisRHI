#pragma once

#include "Common/HandleTable.h"
#include "Internal/VulkanRuntime.h"
#include "Internal/VulkanStageSupport.h"
#include "Internal/VulkanValidationHelpers.h"
#include "Rachis/Device.h"
#include "Rachis/Resource.h"
#include "Rachis/Sampler.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef VMA_STATIC_VULKAN_FUNCTIONS
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#endif
#ifndef VMA_DYNAMIC_VULKAN_FUNCTIONS
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#endif

#include <vk_mem_alloc.h>
#ifdef DeviceCapabilities
#undef DeviceCapabilities
#endif
#include <volk.h>

namespace rachis::vulkan
{

    struct PlacedResourceLease;

    struct VulkanExportedMemoryTotals final
    {
        std::array<std::atomic<std::uint64_t>, VK_MAX_MEMORY_HEAPS> bytes{};
        std::array<std::atomic<std::uint64_t>, VK_MAX_MEMORY_HEAPS> allocations{};

        void add(std::uint32_t heapIndex, std::uint64_t size) noexcept;
        void remove(std::uint32_t heapIndex, std::uint64_t size) noexcept;
    };

    struct VulkanNativeKeepAlive final
    {
        enum class Kind : std::uint8_t
        {
            buffer,
            image,
            view,
            sampler,
        };

        Kind kind = Kind::buffer;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        const VolkDeviceTable* table = nullptr;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        std::shared_ptr<PlacedResourceLease> placement;
        std::shared_ptr<VulkanExportedMemoryTotals> exportedTotals;
        std::uint32_t exportedHeapIndex = 0;
        std::uint64_t exportedBytes = 0;
        std::shared_ptr<VulkanNativeKeepAlive> parent;
        mutable std::mutex mutex;

        VulkanNativeKeepAlive() = default;
        ~VulkanNativeKeepAlive();
        VulkanNativeKeepAlive(const VulkanNativeKeepAlive&) = delete;
        VulkanNativeKeepAlive& operator=(const VulkanNativeKeepAlive&) = delete;
        VulkanNativeKeepAlive(VulkanNativeKeepAlive&&) = delete;
        VulkanNativeKeepAlive& operator=(VulkanNativeKeepAlive&&) = delete;

        void releaseNow() noexcept;
    };

    [[nodiscard]] std::shared_ptr<VulkanNativeKeepAlive> makeVulkanNativeKeepAlive(VulkanNativeKeepAlive::Kind kind,
                                                                                   VkDevice device,
                                                                                   const VolkDeviceTable* table);

    struct ResourceHeapRecord;

    struct PlacedResourceLease final
    {
        std::shared_ptr<ResourceHeapRecord> heap;
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
    };

    struct ResourceHeapRecord final
    {
        ResourceHeapDesc desc{};
        std::string label;
        VkDevice device = VK_NULL_HANDLE;
        const VolkDeviceTable* table = nullptr;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        std::uint32_t memoryTypeIndex = 0;
        std::uint32_t memoryHeapIndex = 0;
        std::uint64_t allocationId = 0;
        std::uint64_t allocationBytes = 0;
        bool alive = true;
        bool hostVisible = false;
        bool hostCoherent = false;
        std::byte* mappedBase = nullptr;
        mutable std::mutex mutex;
        std::vector<std::weak_ptr<PlacedResourceLease>> placements;

        ResourceHeapRecord() = default;
        ~ResourceHeapRecord();
        ResourceHeapRecord(const ResourceHeapRecord&) = delete;
        ResourceHeapRecord& operator=(const ResourceHeapRecord&) = delete;
        ResourceHeapRecord(ResourceHeapRecord&&) = delete;
        ResourceHeapRecord& operator=(ResourceHeapRecord&&) = delete;

        void releaseNow() noexcept;
    };

    enum class ResidencyPoolClass : std::uint8_t
    {
        deviceBuffer,
        uploadBuffer,
        readbackBuffer,
        nonAttachmentTexture,
        attachmentTexture,
        count,
    };

    inline constexpr auto externallyUnknownTextureLayout = static_cast<TextureLayout>(0xFFU);

    inline constexpr std::size_t explicitResidencyPriorityCount = 4;
    inline constexpr std::size_t residencyPoolClassCount = static_cast<std::size_t>(ResidencyPoolClass::count);

    struct VulkanSharingFamilies final
    {
        std::array<std::uint32_t, 3> indices{
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
        };
        std::uint32_t count = 0;

        constexpr void add(std::uint32_t family) noexcept
        {
            if (family == VK_QUEUE_FAMILY_IGNORED)
            {
                return;
            }
            for (std::uint32_t i = 0; i < count; ++i)
            {
                if (indices[i] == family)
                {
                    return;
                }
            }
            if (count < indices.size())
            {
                indices[count++] = family;
            }
        }

        [[nodiscard]] constexpr bool contains(std::uint32_t family) const noexcept
        {
            for (std::uint32_t i = 0; i < count; ++i)
            {
                if (indices[i] == family)
                {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] constexpr VkSharingMode sharingMode() const noexcept
        {
            return count > 1U ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
        }
    };

    class VulkanSwapchainImageLease final
    {
    public:
        void beginAcquisition() noexcept;
        [[nodiscard]] Status cancel();
        [[nodiscard]] Status beginPresentation();
        void abortPresentation() noexcept;
        [[nodiscard]] Status retainRecordedReference();
        void releaseRecordedReference() noexcept;
        void markSubmitted() noexcept;

    private:
        mutable std::mutex mutex_;
        std::size_t recordedReferenceCount_ = 0;
        bool acquired_ = false;
        bool submitted_ = false;
    };

    class VulkanResourceAllocator final
    {
    public:
        VulkanResourceAllocator() = default;
        ~VulkanResourceAllocator();

        VulkanResourceAllocator(const VulkanResourceAllocator&) = delete;
        VulkanResourceAllocator& operator=(const VulkanResourceAllocator&) = delete;
        VulkanResourceAllocator(VulkanResourceAllocator&&) = delete;
        VulkanResourceAllocator& operator=(VulkanResourceAllocator&&) = delete;

        [[nodiscard]] Status initialize(VkInstance instance,
                                        const VulkanInstanceTable* instanceTable,
                                        VkPhysicalDevice physical,
                                        VkDevice device,
                                        const VolkDeviceTable* table,
                                        const VkPhysicalDeviceMemoryProperties& memory,
                                        const VulkanQueueFamilies& families,
                                        VkQueue transferQueue,
                                        std::uint64_t nonCoherentAtomSize,
                                        bool samplerMinMaxReduction,
                                        bool textureViewComponentSwizzle = false,
                                        bool sharedHandles = false,
                                        bool memoryBudget = false,
                                        bool memoryPriority = false,
                                        bool variableRateShadingTier2 = false,
                                        bool bufferDeviceAddress = false,
                                        bool accelerationStructures = false,
                                        bool rayTracingPipeline = false,
                                        std::uint32_t shaderGroupBaseAlignment = 0,
                                        std::uint32_t accelerationStructureScratchAlignment = 0,
                                        ShaderStage enabledShaderStages = coreShaderStages);
        void waitOnAllCommandsBeforeBarriers(bool enabled) noexcept;
        [[nodiscard]] bool waitsOnAllCommandsBeforeBarriers() const noexcept;
        void enableConditionalRendering(bool enabled) noexcept;
        void shutdown() noexcept;

        [[nodiscard]] Result<std::vector<MemoryBudget>> queryMemoryBudgets() const;
        [[nodiscard]] Result<ResourceMemorySnapshot> captureResourceMemorySnapshot() const;
        [[nodiscard]] Result<ResourceMemoryRequirements> queryBufferMemoryRequirements(const BufferDesc& desc) const;
        [[nodiscard]] Result<ResourceHeapHandle> createResourceHeap(const ResourceHeapDesc& desc);
        [[nodiscard]] Status destroyResourceHeap(ResourceHeapHandle heap);
        [[nodiscard]] Result<BufferHandle> createBuffer(const BufferDesc& desc);
        [[nodiscard]] Result<BufferHandle> createSharedBuffer(const BufferDesc& desc);
        [[nodiscard]] Result<SharedHandle> exportSharedBuffer(BufferHandle buffer);
        [[nodiscard]] Result<BufferHandle> importSharedBuffer(SharedHandleView handle, const BufferDesc& desc);
        [[nodiscard]] Result<BufferHandle> createPlacedBuffer(const ResourcePlacement& placement,
                                                              const BufferDesc& desc);
        [[nodiscard]] Status destroyBuffer(BufferHandle buffer);
        [[nodiscard]] Result<MappedMemory> mapBuffer(BufferHandle buffer, std::uint64_t offset, std::uint64_t size);
        [[nodiscard]] Status unmapBuffer(BufferHandle buffer,
                                         std::uint64_t offset,
                                         std::uint64_t size,
                                         bool flushWrites = true);
        [[nodiscard]] Status flushMappedMemoryRanges(BufferHandle buffer, std::span<const MappedMemoryRange> ranges);
        [[nodiscard]] Status invalidateMappedMemoryRanges(BufferHandle buffer,
                                                          std::span<const MappedMemoryRange> ranges);
        [[nodiscard]] Status writeBuffer(BufferHandle buffer, std::uint64_t offset, std::span<const std::byte> data);
        [[nodiscard]] Status readBuffer(BufferHandle buffer, std::uint64_t offset, std::span<std::byte> destination);
        [[nodiscard]] Result<ResourceMemoryRequirements> queryTextureMemoryRequirements(const TextureDesc& desc) const;
        [[nodiscard]] Result<TextureHandle> createTexture(const TextureDesc& desc);
        [[nodiscard]] Result<TextureHandle> createSharedTexture(const TextureDesc& desc);
        [[nodiscard]] Result<SharedHandle> exportSharedTexture(TextureHandle texture);
        [[nodiscard]] Result<TextureHandle> importSharedTexture(SharedHandleView handle, const TextureDesc& desc);
        [[nodiscard]] Result<TextureHandle> createPlacedTexture(const ResourcePlacement& placement,
                                                                const TextureDesc& desc);
        [[nodiscard]] Result<TextureHandle> adoptSwapchainImage(VkImage image,
                                                                const TextureDesc& desc,
                                                                std::shared_ptr<VulkanSwapchainImageLease> lease = {});
        [[nodiscard]] Status destroyTexture(TextureHandle texture);
        [[nodiscard]] Status forgetSwapchainImage(TextureHandle texture);
        void retireSwapchainImage(TextureHandle texture) noexcept;
        [[nodiscard]] bool hasOtherTextureViews(TextureHandle texture, TextureViewHandle view) const noexcept;
        [[nodiscard]] Result<TextureViewHandle> createTextureView(const TextureViewDesc& desc);
        [[nodiscard]] Status destroyTextureView(TextureViewHandle view);
        [[nodiscard]] Result<SamplerHandle> createSampler(const SamplerDesc& desc);
        [[nodiscard]] Status destroySampler(SamplerHandle sampler);

        struct ResolvedBindingBuffer final
        {
            VkBuffer buffer = VK_NULL_HANDLE;
            BufferDesc desc{};
            std::shared_ptr<void> keepAlive;
            VkDeviceAddress deviceAddress = 0;
        };

        struct ResolvedBindingTextureView final
        {
            VkImageView view = VK_NULL_HANDLE;
            TextureViewUsage usage = TextureViewUsage::sampled;
            std::shared_ptr<void> keepAlive;
        };

        struct ResolvedBindingSampler final
        {
            VkSampler sampler = VK_NULL_HANDLE;
            std::shared_ptr<void> keepAlive;
        };

        struct ResolvedRenderAttachment final
        {
            VkImageView view = VK_NULL_HANDLE;
            Format format = Format::unknown;
            std::uint32_t width = 1;
            std::uint32_t height = 1;
            std::uint32_t sampleCount = 1;
            std::uint32_t layerCount = 1;
            TextureViewUsage usage = TextureViewUsage::sampled;
            std::shared_ptr<void> keepAlive;
            std::shared_ptr<void> textureKeepAlive;
            std::shared_ptr<VulkanSwapchainImageLease> swapchainLease;
        };

        struct ResolvedTexture final
        {
            VkImage image = VK_NULL_HANDLE;
            TextureDesc desc{};
            std::shared_ptr<void> keepAlive;
        };

        [[nodiscard]] Result<ResolvedBindingBuffer> resolveBindingBuffer(BufferHandle buffer) const;
        [[nodiscard]] Result<ResolvedTexture> resolveTexture(TextureHandle texture) const;
        [[nodiscard]] Result<ResolvedBindingTextureView> resolveBindingTextureView(TextureViewHandle view) const;
        [[nodiscard]] Result<ResolvedBindingSampler> resolveBindingSampler(SamplerHandle sampler) const;
        [[nodiscard]] Result<ResolvedRenderAttachment> resolveRenderAttachment(TextureViewHandle view) const;
        [[nodiscard]] Status retainSwapchainImages(VkCommandBuffer command,
                                                   std::span<const std::shared_ptr<VulkanSwapchainImageLease>> leases);
        [[nodiscard]] Status bufferBarrier(VkCommandBuffer command,
                                           QueueType queueType,
                                           const BufferBarrier& barrier,
                                           std::vector<std::shared_ptr<void>>* retained = nullptr);
        [[nodiscard]] Status textureBarrier(VkCommandBuffer command,
                                            QueueType queueType,
                                            const TextureBarrier& barrier,
                                            std::vector<std::shared_ptr<void>>* retained = nullptr);
        [[nodiscard]] Status resourceBarriers(VkCommandBuffer command,
                                              QueueType queueType,
                                              const ResourceBarrierBatch& barriers,
                                              std::vector<std::shared_ptr<void>>* retained = nullptr);
        [[nodiscard]] Status activateBufferAlias(VkCommandBuffer command,
                                                 QueueType queueType,
                                                 const BufferAliasActivation& activation,
                                                 std::vector<std::shared_ptr<void>>* retained = nullptr);
        [[nodiscard]] Status activateTextureAlias(VkCommandBuffer command,
                                                  QueueType queueType,
                                                  const TextureAliasActivation& activation,
                                                  std::vector<std::shared_ptr<void>>* retained = nullptr);
        [[nodiscard]] Status copyBufferRegions(VkCommandBuffer command,
                                               BufferHandle source,
                                               BufferHandle destination,
                                               std::span<const BufferCopyRegion> regions,
                                               std::vector<std::shared_ptr<void>>* retained = nullptr);
        [[nodiscard]] Result<TextureTransferLayout> queryTextureTransferLayout(
            TextureHandle texture, const BufferTextureCopyRegion& region) const;
        [[nodiscard]] Status copyBufferToTextureRegions(VkCommandBuffer command,
                                                        QueueType queueType,
                                                        BufferHandle source,
                                                        TextureHandle destination,
                                                        std::span<const BufferTextureCopyRegion> regions,
                                                        std::vector<std::shared_ptr<void>>* retained = nullptr);
        [[nodiscard]] Status copyTextureToBufferRegions(VkCommandBuffer command,
                                                        QueueType queueType,
                                                        TextureHandle source,
                                                        BufferHandle destination,
                                                        std::span<const BufferTextureCopyRegion> regions,
                                                        std::vector<std::shared_ptr<void>>* retained = nullptr);
        [[nodiscard]] Status copyTextureRegions(VkCommandBuffer command,
                                                TextureHandle source,
                                                TextureHandle destination,
                                                std::span<const TextureCopyRegion> regions,
                                                std::vector<std::shared_ptr<void>>* retained = nullptr);
        [[nodiscard]] Status fillBuffer(VkCommandBuffer command,
                                        BufferHandle buffer,
                                        std::uint64_t offset,
                                        std::uint64_t size,
                                        std::uint32_t value,
                                        std::vector<std::shared_ptr<void>>* retained = nullptr);
        [[nodiscard]] Status clearStorageTexture(VkCommandBuffer command,
                                                 TextureViewHandle view,
                                                 const StorageTextureClearValue& value,
                                                 std::vector<std::shared_ptr<void>>* retained = nullptr);

        struct NativeTexture final
        {
            TextureDesc desc{};
            std::vector<Format> viewFormats;
            VkImage image = VK_NULL_HANDLE;
            VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
            std::shared_ptr<void> keepAlive;
            std::shared_ptr<VulkanSwapchainImageLease> swapchainLease;
        };

        [[nodiscard]] Result<NativeTexture> nativeTexture(TextureHandle texture,
                                                          VkCommandBuffer command = VK_NULL_HANDLE,
                                                          TextureSubresource subresource = {}) const;

        struct PendingImageBarrierTransition final
        {
            VkImage image = VK_NULL_HANDLE;
            VkImageAspectFlags aspectMask = 0;
            std::uint32_t baseMipLevel = 0;
            std::uint32_t baseArrayLayer = 0;
            VkImageLayout oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkImageLayout newLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            std::shared_ptr<void> keepAlive;
        };

        struct CommandLayoutBridgePlan final
        {
            std::vector<PendingImageBarrierTransition> preTransitions;
            std::vector<PendingImageBarrierTransition> postTransitions;
        };

        void recordSubresourceTransferRequirement(VkCommandBuffer command,
                                                  TextureHandle texture,
                                                  TextureSubresource subresource,
                                                  VkImageLayout expectedLayout,
                                                  validation_detail::SubresourceTransferUse use);
        [[nodiscard]] Result<std::vector<CommandLayoutBridgePlan>> planBatchLayoutTransitions(
            std::span<const VkCommandBuffer> commands) const;
        void commitCommandLayouts(VkCommandBuffer command) noexcept;
        void discardCommandLayouts(VkCommandBuffer command) noexcept;
        [[nodiscard]] BackendAllocationSnapshot captureBackendAllocationSnapshot() const noexcept;
        [[nodiscard]] FormatProperties queryFormatProperties(Format format) const noexcept;
        [[nodiscard]] bool queueFamilySupportsGraphics(QueueType queueType) const noexcept;
        [[nodiscard]] VulkanFormatPropertiesCache& formatCache() noexcept
        {
            return formatCache_;
        }
#ifdef RACHIS_TESTING
        friend struct VulkanPerformanceTestAccess;
        void initializeDispatchForTesting(VkDevice device,
                                          const VolkDeviceTable* table,
                                          const VulkanInstanceTable* instanceTable = nullptr,
                                          VkPhysicalDevice physical = VK_NULL_HANDLE,
                                          const VulkanQueueFamilies& families = {},
                                          ShaderStage enabledShaderStages = coreShaderStages,
                                          bool accelerationStructures = false) noexcept;
        [[nodiscard]] Status copyBufferImmediate(BufferHandle source,
                                                 BufferHandle destination,
                                                 std::uint64_t sourceOffset,
                                                 std::uint64_t destinationOffset,
                                                 std::uint64_t size);
        [[nodiscard]] Status copyBufferToTextureImmediate(BufferHandle buffer,
                                                          TextureHandle texture,
                                                          std::uint32_t mipLevel,
                                                          std::uint32_t arrayLayer);
        [[nodiscard]] Status copyTextureToBufferImmediate(TextureHandle texture,
                                                          BufferHandle buffer,
                                                          std::uint32_t mipLevel,
                                                          std::uint32_t arrayLayer);
        [[nodiscard]] Status recordFillBuffer(VkCommandBuffer command,
                                              BufferHandle buffer,
                                              std::uint32_t value,
                                              std::vector<std::shared_ptr<void>>* retained = nullptr);
        void markTextureSharedForTesting(TextureHandle texture, bool layoutUnknown) noexcept;
#endif

    private:
        struct ResolvedTextureView final
        {
            Format format = Format::unknown;
            std::uint32_t mipCount = 0;
            std::uint32_t layerCount = 0;
        };

        struct BufferRecord final
        {
            VkBuffer buffer = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VmaAllocationInfo allocationInfo{};
            BufferDesc desc{};
            std::string label;
            std::uint64_t allocationId = 0;
            std::uint64_t allocationBytes = 0;
            std::uint32_t memoryHeapIndex = 0;
            bool hostVisible = false;
            bool hostCoherent = false;
            bool mapped = false;
            bool mappedViaVma = false;
            std::byte* persistentMappedData = nullptr;
            std::byte* mappedBase = nullptr;
            std::uint64_t mappedOffset = 0;
            std::uint64_t mappedSize = 0;
            VulkanSharingFamilies sharingFamilies{};
            bool alive = true;
            bool shared = false;
            bool dedicatedAllocation = false;
            bool imported = false;
            void* exportIdentity = nullptr;
            std::shared_ptr<PlacedResourceLease> placement;
            std::shared_ptr<VulkanNativeKeepAlive> keepAlive;
            VkDeviceAddress deviceAddress = 0;
        };

        struct TextureRecord final
        {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VmaAllocationInfo allocationInfo{};
            TextureDesc desc{};
            std::vector<Format> viewFormats;
            std::string label;
            std::uint64_t allocationId = 0;
            std::uint64_t allocationBytes = 0;
            std::uint32_t liveViews = 0;
            VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
            std::shared_ptr<std::vector<VkImageLayout>> submittedLayouts;
            std::shared_ptr<std::vector<TextureLayout>> submittedPublicLayouts;
            VulkanSharingFamilies sharingFamilies{};
            bool alive = true;
            bool swapchainBorrowed = false;
            bool shared = false;
            bool dedicatedAllocation = false;
            bool imported = false;
            void* exportIdentity = nullptr;
            std::shared_ptr<PlacedResourceLease> placement;
            std::shared_ptr<VulkanNativeKeepAlive> keepAlive;
            std::shared_ptr<VulkanSwapchainImageLease> swapchainLease;
        };

        struct TextureViewRecord final
        {
            VkImageView view = VK_NULL_HANDLE;
            TextureHandle texture{};
            TextureViewDesc desc{};
            ResolvedTextureView resolved{};
            std::string label;
            bool alive = true;
            std::shared_ptr<VulkanNativeKeepAlive> keepAlive;
            std::shared_ptr<VulkanNativeKeepAlive> textureKeepAlive;
        };

        struct SamplerRecord final
        {
            VkSampler sampler = VK_NULL_HANDLE;
            SamplerDesc desc{};
            std::string label;
            bool alive = true;
            std::shared_ptr<VulkanNativeKeepAlive> keepAlive;
        };

        [[nodiscard]] Status validateBufferDesc(const BufferDesc& desc) const;
        [[nodiscard]] Status validateResidencyRequest(ResidencyPriority priority, ResourceAllocationFlag flags) const;
        [[nodiscard]] Status validateAllocationSize(std::uint64_t bytes, MemoryType memoryType) const;
        [[nodiscard]] VulkanSharingFamilies bufferSharingFamilies(BufferUsage usage) const noexcept;
        [[nodiscard]] VulkanSharingFamilies textureSharingFamilies(TextureUsage usage) const noexcept;
        [[nodiscard]] Result<VkBufferCreateInfo> bufferCreateInfo(const BufferDesc& desc,
                                                                  VulkanSharingFamilies& sharing) const;
        [[nodiscard]] VmaAllocationCreateInfo allocationCreateInfo(MemoryType memoryType) const noexcept;
        [[nodiscard]] Result<VmaAllocationCreateInfo> bufferAllocationCreateInfo(const VkBufferCreateInfo& createInfo,
                                                                                 MemoryType memoryType,
                                                                                 ResidencyPriority priority,
                                                                                 std::uint32_t& outTypeBits,
                                                                                 std::uint64_t& outRequiredSize,
                                                                                 std::uint32_t& outSelectedMemoryType);
        [[nodiscard]] Result<VmaAllocationCreateInfo> imageAllocationCreateInfo(const VkImageCreateInfo& createInfo,
                                                                                MemoryType memoryType,
                                                                                ResidencyPriority priority,
                                                                                ResidencyPoolClass poolClass);
        [[nodiscard]] Result<VmaPool> residencyPool(ResidencyPoolClass poolClass,
                                                    std::uint32_t memoryTypeIndex,
                                                    ResidencyPriority priority);
        [[nodiscard]] std::uint32_t stripDeviceLocalMemoryTypes(std::uint32_t typeBits) const noexcept;
        [[nodiscard]] Status synchronizeMappedMemoryRanges(BufferRecord& record,
                                                           std::span<const MappedMemoryRange> ranges,
                                                           MemoryType requiredMemoryType,
                                                           bool flush);

        [[nodiscard]] Status synchronizeHostRange(const BufferRecord& record,
                                                  std::uint64_t offset,
                                                  std::uint64_t size,
                                                  bool flush);
        void recordFillBufferLocked(VkCommandBuffer command,
                                    const BufferRecord& record,
                                    std::uint64_t offset,
                                    std::uint64_t size,
                                    std::uint32_t value,
                                    std::vector<std::shared_ptr<void>>* retained);
        [[nodiscard]] BufferRecord* findBuffer(BufferHandle buffer) noexcept;
        [[nodiscard]] const BufferRecord* findBuffer(BufferHandle buffer) const noexcept;
        [[nodiscard]] Result<std::uint32_t> findMemoryType(std::uint32_t typeBits,
                                                           MemoryType memoryType,
                                                           std::uint64_t requiredSize = 0) const;
        [[nodiscard]] Result<std::shared_ptr<PlacedResourceLease>> reservePlacement(
            const std::shared_ptr<ResourceHeapRecord>& heap,
            ResourceHeapClass requiredClass,
            std::uint64_t offset,
            const ResourceMemoryRequirements& requirements,
            std::string_view resourceKind);
        [[nodiscard]] Result<std::uint32_t> queryResourceHeapMemoryTypeBits(const ResourceHeapDesc& desc) const;
        [[nodiscard]] Status allocateResourceHeapMemory(ResourceHeapRecord& record, std::uint32_t compatibleTypeBits);
        [[nodiscard]] Status mapHostVisibleResourceHeap(ResourceHeapRecord& record);
        [[nodiscard]] std::shared_ptr<ResourceHeapRecord> findResourceHeap(ResourceHeapHandle heap) const;
        [[nodiscard]] Result<ResourceMemoryRequirements> resolvePlacedBufferMemoryRequirements(
            const BufferDesc& desc) const;
        [[nodiscard]] VkDeviceAddress queryPlacedBufferDeviceAddress(VkBuffer buffer, BufferUsage usage) const;
        [[nodiscard]] Result<ResourceMemoryRequirements> queryTextureMemoryRequirements(
            const TextureDesc& desc, VkImageCreateFlags additionalFlags) const;
        [[nodiscard]] Result<ResourceMemoryRequirements> resolvePlacedTextureMemoryRequirements(
            const TextureDesc& desc, bool allowAliasing) const;
        [[nodiscard]] Result<VkImage> createPlacedImage(const TextureDesc& desc,
                                                        VulkanSharingFamilies& sharing,
                                                        bool allowAliasing);
        [[nodiscard]] Result<VkMemoryRequirements2> queryNativeBufferMemoryRequirements(
            const VkBufferCreateInfo& createInfo, VkMemoryDedicatedRequirements& dedicated) const;
        [[nodiscard]] Result<VkMemoryRequirements2> queryNativeImageMemoryRequirements(
            const VkImageCreateInfo& createInfo, VkMemoryDedicatedRequirements& dedicated) const;
        [[nodiscard]] Status validateTextureDesc(const TextureDesc& desc) const;
        [[nodiscard]] Result<VkImageCreateInfo> imageCreateInfo(const TextureDesc& desc,
                                                                std::vector<VkFormat>& viewFormats,
                                                                VulkanSharingFamilies& sharing) const;
        [[nodiscard]] Result<ResolvedTextureView> resolveTextureView(const TextureRecord& texture,
                                                                     const TextureViewDesc& desc) const;
        [[nodiscard]] Status validateSamplerDesc(const SamplerDesc& desc) const;
        [[nodiscard]] Status createNativeBuffer(const VkBufferCreateInfo& createInfo, BufferRecord& record);
        [[nodiscard]] Status validateSharedBufferCreation(const BufferDesc& desc) const;
        [[nodiscard]] Status validateSharedBufferImport(SharedHandleView handle, const BufferDesc& desc) const;
        [[nodiscard]] Status validateSharedTextureCreation(const TextureDesc& desc) const;
        [[nodiscard]] Status validateSharedTextureImport(SharedHandleView handle, const TextureDesc& desc) const;
        struct SharedMemoryImport;
        [[nodiscard]] Result<VmaAllocationInfo> backBufferWithExportableMemory(VkBuffer buffer,
                                                                               MemoryType memoryType,
                                                                               bool deviceAddress);
        [[nodiscard]] Result<VmaAllocationInfo> backBufferWithImportedMemory(VkBuffer buffer,
                                                                             MemoryType memoryType,
                                                                             const SharedMemoryImport& source,
                                                                             bool deviceAddress);
        [[nodiscard]] Result<VmaAllocationInfo> backImageWithExportableMemory(VkImage image);
        [[nodiscard]] Result<VmaAllocationInfo> backImageWithImportedMemory(VkImage image,
                                                                            const SharedMemoryImport& source);
        void discardAllRecordsLocked() noexcept;

        struct LayoutWrite final
        {
            std::shared_ptr<std::vector<VkImageLayout>> submitted;
            std::size_t index = 0;
            VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
            std::shared_ptr<std::vector<TextureLayout>> submittedPublic;
            TextureLayout publicLayout = TextureLayout::undefined;
        };
        struct LayoutCell final
        {
            const std::vector<VkImageLayout>* submitted = nullptr;
            std::size_t index = 0;

            [[nodiscard]] bool operator==(const LayoutCell&) const noexcept = default;
        };
        struct LayoutCellHash final
        {
            [[nodiscard]] std::size_t operator()(const LayoutCell& cell) const noexcept
            {
                const auto pointer = std::hash<const void*>{}(cell.submitted);
                return pointer ^ (std::hash<std::size_t>{}(cell.index) + 0x9e3779b97f4a7c15ULL + (pointer << 6U) +
                                  (pointer >> 2U));
            }
        };
        struct LayoutEntryRequirement final
        {
            std::shared_ptr<std::vector<TextureLayout>> submittedPublic;
            std::size_t index = 0;
            TextureLayout layout = TextureLayout::undefined;
            TextureSubresource subresource{};
        };
        struct LayoutRequirement final
        {
            std::shared_ptr<std::vector<VkImageLayout>> submitted;
            std::shared_ptr<void> keepAlive;
            VkImage image = VK_NULL_HANDLE;
            VkImageAspectFlags aspectMask = 0;
            std::uint32_t mipLevel = 0;
            std::uint32_t arrayLayer = 0;
            std::size_t index = 0;
            VkImageLayout expectedLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            validation_detail::SubresourceTransferUse use = validation_detail::SubresourceTransferUse::copySource;
            bool hasSubsequentBarrier = false;
            VkImageLayout firstBarrierOldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        };
        [[nodiscard]] VkImageLayout recordedLayout(VkCommandBuffer command,
                                                   const TextureRecord& record,
                                                   TextureSubresource subresource) const;
        void recordTransferRequirementLocked(VkCommandBuffer command,
                                             TextureRecord& record,
                                             TextureSubresource subresource,
                                             VkImageLayout expectedLayout,
                                             validation_detail::SubresourceTransferUse use);
        void noteBarrierOldLayoutLocked(VkCommandBuffer command,
                                        TextureRecord& record,
                                        const VkImageSubresourceRange& range,
                                        VkImageLayout oldLayout);
        void stageLayout(VkCommandBuffer command,
                         TextureRecord& record,
                         const VkImageSubresourceRange& range,
                         VkImageLayout layout,
                         TextureLayout publicLayout);
        void appendLayoutWrites(TextureRecord& record,
                                const VkImageSubresourceRange& range,
                                VkImageLayout layout,
                                TextureLayout publicLayout,
                                std::vector<LayoutWrite>& writes);
        static void ensureTrackedLayouts(TextureRecord& record);
        static void applyCommittedLayout(const LayoutWrite& write) noexcept;
        class SubmittedLayoutSimulation;
        struct ResolvedBufferBarriers;
        struct ResolvedTextureBarriers;
        [[nodiscard]] Result<ResolvedBufferBarriers> resolveBufferBarriersLocked(
            std::span<const BufferBarrier> barriers, QueueType queueType);
        [[nodiscard]] Result<ResolvedTextureBarriers> resolveTextureBarriersLocked(
            std::span<const TextureBarrier> barriers, QueueType queueType);
        [[nodiscard]] Status validateBarrierEntryLayoutsLocked(VkCommandBuffer command,
                                                               const ResolvedTextureBarriers& textures) const;
        void recordBarrierLayoutTransitionsLocked(VkCommandBuffer command, const ResolvedTextureBarriers& textures);

        struct CommandLayoutState final
        {
            std::vector<LayoutWrite> writes;
            std::unordered_map<LayoutCell, std::size_t, LayoutCellHash> writeIndex;
            std::vector<LayoutRequirement> transfers;
            std::unordered_map<LayoutCell, std::size_t, LayoutCellHash> transferIndex;
            std::vector<LayoutEntryRequirement> entries;
        };
        struct CommandLayoutMark final
        {
            std::size_t writes = 0;
            std::size_t transfers = 0;
            std::size_t entries = 0;
        };
        [[nodiscard]] const CommandLayoutState* commandLayoutState(VkCommandBuffer command) const noexcept;
        [[nodiscard]] CommandLayoutMark markCommandLayoutsLocked(VkCommandBuffer command) const noexcept;
        void rollBackCommandLayoutsLocked(VkCommandBuffer command, const CommandLayoutMark& mark) noexcept;
        [[nodiscard]] static std::optional<VkImageLayout> nonGeneralTransferLayout(const CommandLayoutState& state,
                                                                                   const LayoutCell& cell) noexcept;
        void recordStorageClearEntryTransitionsLocked(VkCommandBuffer command,
                                                      TextureRecord& texture,
                                                      const VkImageSubresourceRange& range,
                                                      const CommandLayoutMark& mark);
        [[nodiscard]] static std::optional<TextureLayout> commandPublicLayout(const CommandLayoutState* state,
                                                                              const LayoutCell& cell) noexcept;
        static void stageLayoutWrite(CommandLayoutState& state, const LayoutWrite& write);
        std::unordered_map<VkCommandBuffer, CommandLayoutState> commandLayouts_;
        [[nodiscard]] Status retainSwapchainImagesLocked(
            VkCommandBuffer command, std::span<const std::shared_ptr<VulkanSwapchainImageLease>> leases);
        void releaseSwapchainImagesLocked(VkCommandBuffer command) noexcept;
        std::unordered_map<VkCommandBuffer, std::vector<std::shared_ptr<VulkanSwapchainImageLease>>>
            recordedSwapchainImages_;

        const VulkanInstanceTable* instanceTable_ = nullptr;
        VkPhysicalDevice physical_ = VK_NULL_HANDLE;
        VkDevice device_ = VK_NULL_HANDLE;
        const VolkDeviceTable* table_ = nullptr;
        VmaAllocator allocator_ = VK_NULL_HANDLE;
        VkPhysicalDeviceMemoryProperties memory_{};
        VkCommandPool transferPool_ = VK_NULL_HANDLE;
        VkQueue transferQueue_ = VK_NULL_HANDLE;
        VulkanQueueFamilies families_{};
        std::array<bool, 3> queueGraphicsSupport_{true, true, true};
        std::uint64_t nonCoherentAtomSize_ = 1;
        std::uint64_t shaderGroupBaseAlignment_ = 0;
        std::uint64_t accelerationStructureScratchAlignment_ = 0;
        std::uint64_t maxMemoryAllocationSize_ = std::numeric_limits<std::uint64_t>::max();
        std::atomic<std::uint64_t> nextAllocationId_{1};
        mutable std::atomic<std::uint32_t> budgetRefreshIndex_{0};
        std::shared_ptr<VulkanExportedMemoryTotals> exportedMemory_ = std::make_shared<VulkanExportedMemoryTotals>();
        VkPhysicalDeviceLimits limits_{};
        mutable VulkanFormatPropertiesCache formatCache_{};
        bool samplerMinMaxReduction_ = false;
        bool textureViewComponentSwizzle_ = false;
        bool sharedHandles_ = false;
        bool memoryBudget_ = false;
        bool memoryPriority_ = false;
        bool variableRateShadingTier2_ = false;
        bool bufferDeviceAddress_ = false;
        bool accelerationStructures_ = false;
        bool rayTracingPipeline_ = false;
        PipelineStage barrierStages_ = corePipelineStages;
        bool allCommandsBarrierSources_ = false;
        bool conditionalRendering_ = false;
        std::mutex residencyPoolsMutex_;
        std::array<std::array<std::array<VmaPool, explicitResidencyPriorityCount>, VK_MAX_MEMORY_TYPES>,
                   residencyPoolClassCount>
            residencyPools_{};
        std::array<std::uint8_t, VK_UUID_SIZE> deviceUuid_{};
        void trackKeepAliveLocked(const std::shared_ptr<VulkanNativeKeepAlive>& keepAlive);

        template<typename Record, typename Handle>
        [[nodiscard]] static Result<Handle> emplaceRecordLocked(detail::HandleTable<Record, Handle>& records,
                                                                Record&& record)
        {
            auto handle = records.emplace(std::move(record));
            if (handle)
            {
                auto* const stored = records.find(*handle);
                stored->desc.label = stored->label;
            }
            return handle;
        }

        mutable std::mutex mutex_;
        detail::HandleTable<BufferRecord, BufferHandle> buffers_;
        detail::HandleTable<TextureRecord, TextureHandle> textures_;
        detail::HandleTable<TextureViewRecord, TextureViewHandle> views_;
        detail::HandleTable<SamplerRecord, SamplerHandle> samplers_;
        detail::HandleTable<std::shared_ptr<ResourceHeapRecord>, ResourceHeapHandle> heaps_;

        mutable std::vector<std::weak_ptr<ResourceHeapRecord>> heapRecords_;
        std::vector<std::weak_ptr<VulkanNativeKeepAlive>> nativeKeepAlives_;
        std::size_t nextKeepAlivePruneThreshold_ = 64;
    };

#ifdef RACHIS_TESTING
    void bindVulkanResourceAllocator(VulkanResourceAllocator* allocator) noexcept;
    void unbindVulkanResourceAllocator(VulkanResourceAllocator* allocator) noexcept;
    [[nodiscard]] Status copyVulkanBuffersImmediate(BufferHandle source,
                                                    BufferHandle destination,
                                                    std::uint64_t sourceOffset,
                                                    std::uint64_t destinationOffset,
                                                    std::uint64_t size);
    [[nodiscard]] Status copyVulkanBufferToTextureImmediate(BufferHandle buffer,
                                                            TextureHandle texture,
                                                            std::uint32_t mipLevel,
                                                            std::uint32_t arrayLayer);
    [[nodiscard]] Status copyVulkanTextureToBufferImmediate(TextureHandle texture,
                                                            BufferHandle buffer,
                                                            std::uint32_t mipLevel,
                                                            std::uint32_t arrayLayer);
    [[nodiscard]] Status recordVulkanFillBuffer(CommandBuffer& commandBuffer, BufferHandle buffer, std::uint32_t value);

    struct VulkanCapturedBarrier final
    {
        bool image = false;
        bool memory = false;
        VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        std::uint32_t recordingFamily = VK_QUEUE_FAMILY_IGNORED;
        std::uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        std::uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        VkImageLayout oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout newLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkPipelineStageFlags2 srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        VkPipelineStageFlags2 dstStageMask = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 srcAccessMask = VK_ACCESS_2_NONE;
        VkAccessFlags2 dstAccessMask = VK_ACCESS_2_NONE;
    };

    void captureVulkanResourceBarriers(std::vector<VulkanCapturedBarrier>* sink) noexcept;
    void captureAliasBarriers(const VkMemoryBarrier2& deactivate,
                              VkPipelineStageFlags2 activateSrcStages,
                              VkPipelineStageFlags2 activateDstStages,
                              VkAccessFlags2 activateDstAccess);

    enum class VulkanBadAllocPoint : std::uint8_t
    {
        nativeKeepAlive,
        sharedExportRegistry,
        sharedImportRegistry,
        indirectArgumentRetention,
        accelerationStructureRecord,
        commandUses,
    };

    void injectVulkanBadAlloc(VulkanBadAllocPoint point, bool armed) noexcept;
    void throwInjectedVulkanBadAlloc(VulkanBadAllocPoint point);
#endif

} // namespace rachis::vulkan
