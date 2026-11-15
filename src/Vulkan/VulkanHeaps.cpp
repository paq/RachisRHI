#include "Internal/VulkanDiagnostics.h"
#include "Internal/VulkanResources.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rachis::vulkan
{
    namespace
    {

        inline constexpr std::uint64_t defaultResourceHeapAlignment = 65'536;
        inline constexpr std::uint64_t multisampleResourceHeapAlignment = 4'194'304;

        [[nodiscard]] constexpr bool validMemoryType(MemoryType memoryType) noexcept
        {
            switch (memoryType)
            {
            case MemoryType::deviceLocal:
            case MemoryType::upload:
            case MemoryType::readback:
                return true;
            }
            return false;
        }

        [[nodiscard]] constexpr bool validResourceHeapClass(ResourceHeapClass resourceClass) noexcept
        {
            switch (resourceClass)
            {
            case ResourceHeapClass::buffer:
            case ResourceHeapClass::nonAttachmentTexture:
            case ResourceHeapClass::attachmentTexture:
                return true;
            }
            return false;
        }

        [[nodiscard]] constexpr bool validResourceHeapFlags(ResourceHeapFlag flags) noexcept
        {
            constexpr auto known = static_cast<std::uint32_t>(ResourceHeapFlag::allowAliasing) |
                                   static_cast<std::uint32_t>(ResourceHeapFlag::sparseBacking);
            return (static_cast<std::uint32_t>(flags) & ~known) == 0;
        }

        [[nodiscard]] constexpr ResourceHeapClass textureResourceHeapClass(TextureUsage usage) noexcept
        {
            return hasAny(usage, TextureUsage::renderTarget | TextureUsage::depthStencil)
                       ? ResourceHeapClass::attachmentTexture
                       : ResourceHeapClass::nonAttachmentTexture;
        }

        [[nodiscard]] constexpr TextureUsage dummyTextureUsage(ResourceHeapClass resourceClass) noexcept
        {
            return resourceClass == ResourceHeapClass::attachmentTexture
                       ? TextureUsage::renderTarget | TextureUsage::copySource | TextureUsage::copyDestination
                       : TextureUsage::sampled | TextureUsage::copySource | TextureUsage::copyDestination;
        }

        struct MemoryPropertyCriteria final
        {
            VkMemoryPropertyFlags required = 0;
            VkMemoryPropertyFlags preferred = 0;
            VkMemoryPropertyFlags avoided = 0;
        };

        [[nodiscard]] constexpr MemoryPropertyCriteria memoryPropertyCriteria(MemoryType memoryType) noexcept
        {
            switch (memoryType)
            {
            case MemoryType::deviceLocal:
                return {
                    .required = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    .preferred = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    .avoided = 0,
                };
            case MemoryType::upload:
                return {
                    .required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                    .preferred = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    .avoided = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                };
            case MemoryType::readback:
                return {
                    .required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                    .preferred = VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                    .avoided = 0,
                };
            }
            return {};
        }

        [[nodiscard]] bool memoryHeapFits(const VkPhysicalDeviceMemoryProperties& memory,
                                          const VkMemoryType& type,
                                          std::uint64_t requiredSize) noexcept
        {
            if (memory.memoryHeapCount == 0)
            {
                return true;
            }
            if (type.heapIndex >= memory.memoryHeapCount)
            {
                return false;
            }
            return requiredSize == 0 || memory.memoryHeaps[type.heapIndex].size >= requiredSize;
        }

        [[nodiscard]] bool isResizableBarMemoryType(const VkPhysicalDeviceMemoryProperties& memory,
                                                    const VkMemoryType& type) noexcept
        {
            constexpr std::uint64_t rebarHeapThresholdBytes = 256ULL * 1024ULL * 1024ULL;
            constexpr VkMemoryPropertyFlags rebarFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            return (type.propertyFlags & rebarFlags) == rebarFlags &&
                   (memory.memoryHeapCount <= 1 || memory.memoryHeaps[type.heapIndex].size > rebarHeapThresholdBytes);
        }

        [[nodiscard]] std::optional<std::uint32_t> findResizableBarMemoryType(
            const VkPhysicalDeviceMemoryProperties& memory, std::uint32_t typeBits, std::uint64_t requiredSize) noexcept
        {
            for (std::uint32_t index = 0; index < memory.memoryTypeCount; ++index)
            {
                if ((typeBits & (1U << index)) == 0)
                {
                    continue;
                }
                const auto& type = memory.memoryTypes[index];
                if (memoryHeapFits(memory, type, requiredSize) && isResizableBarMemoryType(memory, type))
                {
                    return index;
                }
            }
            return std::nullopt;
        }

        [[nodiscard]] constexpr bool improvesFallbackMemoryType(VkMemoryPropertyFlags fallbackFlags,
                                                                VkMemoryPropertyFlags candidateFlags,
                                                                const MemoryPropertyCriteria& criteria) noexcept
        {
            const bool fallbackAvoided = (fallbackFlags & criteria.avoided) != 0;
            const bool candidateAvoided = (candidateFlags & criteria.avoided) != 0;
            if (fallbackAvoided && !candidateAvoided)
            {
                return true;
            }
            return fallbackAvoided == candidateAvoided && (fallbackFlags & criteria.preferred) != criteria.preferred &&
                   (candidateFlags & criteria.preferred) == criteria.preferred;
        }

    } // namespace

    Result<std::uint32_t> VulkanResourceAllocator::findMemoryType(std::uint32_t typeBits,
                                                                  MemoryType memoryType,
                                                                  std::uint64_t requiredSize) const
    {
        if (memoryType == MemoryType::upload)
        {
            if (const auto resizableBar = findResizableBarMemoryType(memory_, typeBits, requiredSize))
            {
                return *resizableBar;
            }
        }

        const auto criteria = memoryPropertyCriteria(memoryType);
        bool rejectedForSize = false;
        std::int32_t fallback = -1;
        for (std::uint32_t index = 0; index < memory_.memoryTypeCount; ++index)
        {
            if ((typeBits & (1U << index)) == 0)
            {
                continue;
            }
            const auto& type = memory_.memoryTypes[index];
            const auto flags = type.propertyFlags;
            if ((flags & criteria.required) != criteria.required)
            {
                continue;
            }
            if (!memoryHeapFits(memory_, type, requiredSize))
            {
                rejectedForSize = true;
                continue;
            }
            if ((flags & criteria.preferred) == criteria.preferred && (flags & criteria.avoided) == 0)
            {
                return index;
            }
            if (fallback < 0)
            {
                fallback = static_cast<std::int32_t>(index);
                continue;
            }
            const auto fallbackFlags = memory_.memoryTypes[static_cast<std::uint32_t>(fallback)].propertyFlags;
            if (improvesFallbackMemoryType(fallbackFlags, flags, criteria))
            {
                fallback = static_cast<std::int32_t>(index);
            }
        }
        if (fallback < 0)
        {
            if (rejectedForSize)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "The allocation exceeds every compatible Vulkan memory heap.",
                });
            }
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The Vulkan device has no memory type compatible with this resource heap.",
            });
        }
        return static_cast<std::uint32_t>(fallback);
    }

    Result<std::shared_ptr<PlacedResourceLease>> VulkanResourceAllocator::reservePlacement(
        const std::shared_ptr<ResourceHeapRecord>& heap,
        ResourceHeapClass requiredClass,
        std::uint64_t offset,
        const ResourceMemoryRequirements& requirements,
        std::string_view resourceKind)
    {
        if (heap == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                std::string("A placed ") + std::string(resourceKind) + " requires a live resource heap.",
            });
        }
        if (heap->desc.resourceClass != requiredClass)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                std::string("A placed ") + std::string(resourceKind) + " requires a compatible resource-heap class.",
            });
        }
        if (hasAny(heap->desc.flags, ResourceHeapFlag::sparseBacking))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                std::string("A placed ") + std::string(resourceKind) + " cannot use a sparse-backing heap.",
            });
        }
        if (requirements.alignment > heap->desc.alignment || offset % requirements.alignment != 0 ||
            requirements.size > heap->desc.size || offset > heap->desc.size - requirements.size)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                std::string("A placed ") + std::string(resourceKind) +
                    " requires a heap with sufficient alignment and an aligned, in-bounds range.",
            });
        }

        auto lease = std::make_shared<PlacedResourceLease>();
        lease->heap = heap;
        lease->offset = offset;
        lease->size = requirements.size;

        const std::scoped_lock lock(heap->mutex);
        if (!heap->alive)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                std::string("A placed ") + std::string(resourceKind) + " requires a live resource heap.",
            });
        }
        auto iterator = heap->placements.begin();
        while (iterator != heap->placements.end())
        {
            const auto existing = iterator->lock();
            if (existing == nullptr)
            {
                iterator = heap->placements.erase(iterator);
                continue;
            }
            const bool overlaps =
                offset < existing->offset + existing->size && existing->offset < offset + requirements.size;
            if (overlaps && !hasAny(heap->desc.flags, ResourceHeapFlag::allowAliasing))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Placed-resource ranges cannot overlap while either resource remains physically retained.",
                });
            }
            ++iterator;
        }
        heap->placements.emplace_back(lease);
        return lease;
    }

    namespace
    {

        [[nodiscard]] constexpr bool validResourceHeapAlignment(ResourceHeapClass resourceClass,
                                                                std::uint64_t alignment) noexcept
        {
            return (alignment == defaultResourceHeapAlignment) ||
                   (resourceClass == ResourceHeapClass::attachmentTexture &&
                    alignment == multisampleResourceHeapAlignment) ||
                   (alignment >= defaultResourceHeapAlignment && std::has_single_bit(alignment));
        }

        [[nodiscard]] Status validateResourceHeapDesc(const ResourceHeapDesc& desc, std::uint64_t alignment)
        {
            const bool validSizeAndAlignment = desc.size != 0 &&
                                               validResourceHeapAlignment(desc.resourceClass, alignment) &&
                                               desc.size % alignment == 0;
            const bool validEnumerators = validMemoryType(desc.memoryType) &&
                                          validResourceHeapClass(desc.resourceClass) &&
                                          validResourceHeapFlags(desc.flags);
            const bool requiresDeviceLocalMemory =
                desc.resourceClass != ResourceHeapClass::buffer || hasAny(desc.flags, ResourceHeapFlag::allowAliasing);
            if (!validSizeAndAlignment || !validEnumerators ||
                (requiresDeviceLocalMemory && desc.memoryType != MemoryType::deviceLocal))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A resource heap requires compatible memory, placement, residency, and allocation properties.",
                });
            }
            if (hasAny(desc.flags, ResourceHeapFlag::sparseBacking))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Sparse-backing heaps are unavailable on this device.",
                });
            }
            return {};
        }

        [[nodiscard]] constexpr float nativeMemoryPriority(ResidencyPriority priority) noexcept
        {
            switch (priority)
            {
            case ResidencyPriority::minimum:
                return 0.0F;
            case ResidencyPriority::low:
                return 0.25F;
            case ResidencyPriority::high:
                return 1.0F;
            case ResidencyPriority::normal:
            case ResidencyPriority::automatic:
                break;
            }
            return 0.5F;
        }

        [[nodiscard]] bool isDeviceLocalMemoryType(const VkPhysicalDeviceMemoryProperties& memory,
                                                   std::uint32_t memoryTypeIndex) noexcept
        {
            return memoryTypeIndex < memory.memoryTypeCount &&
                   (memory.memoryTypes[memoryTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        }

    } // namespace

    Result<std::uint32_t> VulkanResourceAllocator::queryResourceHeapMemoryTypeBits(const ResourceHeapDesc& desc) const
    {
        if (desc.resourceClass == ResourceHeapClass::buffer)
        {
            const BufferDesc probe{
                .size = 256,
                .usage = BufferUsage::copySource | BufferUsage::copyDestination | BufferUsage::storage,
                .memoryType = desc.memoryType,
            };
            VulkanSharingFamilies sharing{};
            auto createInfo = bufferCreateInfo(probe, sharing);
            if (!createInfo)
            {
                return std::unexpected(std::move(createInfo.error()));
            }
            VkMemoryDedicatedRequirements dedicated{};
            auto requirements = queryNativeBufferMemoryRequirements(*createInfo, dedicated);
            if (!requirements)
            {
                return std::unexpected(std::move(requirements.error()));
            }
            return requirements->memoryRequirements.memoryTypeBits;
        }

        const TextureDesc probe{
            .width = 4,
            .height = 4,
            .format = Format::rgba8Unorm,
            .usage = dummyTextureUsage(desc.resourceClass),
        };
        std::vector<VkFormat> viewFormats;
        VulkanSharingFamilies sharing{};
        auto createInfo = imageCreateInfo(probe, viewFormats, sharing);
        if (!createInfo)
        {
            return std::unexpected(std::move(createInfo.error()));
        }
        VkMemoryDedicatedRequirements dedicated{};
        auto requirements = queryNativeImageMemoryRequirements(*createInfo, dedicated);
        if (!requirements)
        {
            return std::unexpected(std::move(requirements.error()));
        }
        return requirements->memoryRequirements.memoryTypeBits;
    }

    Status VulkanResourceAllocator::allocateResourceHeapMemory(ResourceHeapRecord& record,
                                                               std::uint32_t compatibleTypeBits)
    {
        const auto& desc = record.desc;
        VkMemoryPriorityAllocateInfoEXT priorityInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT,
            .pNext = {},
            .priority = nativeMemoryPriority(desc.residencyPriority),
        };
        VkMemoryAllocateInfo allocate{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = {},
            .allocationSize = desc.size,
            .memoryTypeIndex = record.memoryTypeIndex,
        };
        if (desc.residencyPriority != ResidencyPriority::automatic)
        {
            allocate.pNext = &priorityInfo;
        }

        VkMemoryAllocateFlagsInfo allocateFlags{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
            .pNext = allocate.pNext,
            .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
            .deviceMask = 0,
        };
        if (bufferDeviceAddress_ && desc.resourceClass == ResourceHeapClass::buffer)
        {
            allocate.pNext = &allocateFlags;
        }
        auto allocated = table_->vkAllocateMemory(device_, &allocate, nullptr, &record.memory);
        if (allocated == VK_ERROR_OUT_OF_DEVICE_MEMORY && desc.memoryType == MemoryType::upload &&
            isDeviceLocalMemoryType(memory_, record.memoryTypeIndex))
        {
            const auto fallbackBits = stripDeviceLocalMemoryTypes(compatibleTypeBits);
            if (auto fallbackType = findMemoryType(fallbackBits, desc.memoryType, desc.size); fallbackType.has_value())
            {
                record.memoryTypeIndex = *fallbackType;
                record.memoryHeapIndex = memory_.memoryTypes[*fallbackType].heapIndex;
                allocate.memoryTypeIndex = *fallbackType;
                allocated = table_->vkAllocateMemory(device_, &allocate, nullptr, &record.memory);
            }
        }
        if (allocated != VK_SUCCESS)
        {
            return std::unexpected(
                makeVulkanError(allocated == VK_ERROR_OUT_OF_DEVICE_MEMORY || allocated == VK_ERROR_OUT_OF_HOST_MEMORY
                                    ? ErrorCode::outOfMemory
                                    : vulkanResultErrorCode(allocated),
                                allocated,
                                "vkAllocateMemory"));
        }
        return {};
    }

    Status VulkanResourceAllocator::mapHostVisibleResourceHeap(ResourceHeapRecord& record)
    {
        void* mapped = nullptr;
        const auto mappedResult = table_->vkMapMemory(device_, record.memory, 0, record.desc.size, 0, &mapped);
        if (mappedResult != VK_SUCCESS)
        {
            table_->vkFreeMemory(device_, record.memory, nullptr);
            record.memory = VK_NULL_HANDLE;
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(mappedResult), mappedResult, "vkMapMemory"));
        }
        record.mappedBase = static_cast<std::byte*>(mapped);
        return {};
    }

    Result<ResourceHeapHandle> VulkanResourceAllocator::createResourceHeap(const ResourceHeapDesc& desc)
    {
        const auto alignment = desc.alignment == 0 ? defaultResourceHeapAlignment : desc.alignment;
        if (auto valid = validateResourceHeapDesc(desc, alignment); !valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        if (auto residency = validateResidencyRequest(desc.residencyPriority, desc.allocationFlags); !residency)
        {
            return std::unexpected(std::move(residency.error()));
        }
        if (auto sized = validateAllocationSize(desc.size, desc.memoryType); !sized)
        {
            return std::unexpected(std::move(sized.error()));
        }
        if (device_ == VK_NULL_HANDLE || table_ == nullptr || table_->vkAllocateMemory == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Resource heaps require a live Vulkan device.",
            });
        }

        auto typeBits = queryResourceHeapMemoryTypeBits(desc);
        if (!typeBits)
        {
            return std::unexpected(std::move(typeBits.error()));
        }
        auto memoryType = findMemoryType(*typeBits, desc.memoryType, desc.size);
        if (!memoryType)
        {
            return std::unexpected(std::move(memoryType.error()));
        }

        auto record = std::make_shared<ResourceHeapRecord>();
        record->desc = desc;
        record->desc.alignment = alignment;
        record->label = std::string(desc.label);
        record->desc.label = record->label;
        record->device = device_;
        record->table = table_;
        record->memoryTypeIndex = *memoryType;
        record->memoryHeapIndex = memory_.memoryTypes[*memoryType].heapIndex;
        record->allocationId = nextAllocationId_.fetch_add(1, std::memory_order_relaxed);
        record->allocationBytes = desc.size;

        if (auto allocated = allocateResourceHeapMemory(*record, *typeBits); !allocated)
        {
            return std::unexpected(std::move(allocated.error()));
        }
        const auto properties = memory_.memoryTypes[record->memoryTypeIndex].propertyFlags;
        record->hostVisible = (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
        record->hostCoherent = (properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        if (record->hostVisible)
        {
            if (auto mapped = mapHostVisibleResourceHeap(*record); !mapped)
            {
                return std::unexpected(std::move(mapped.error()));
            }
        }
        if (!record->label.empty())
        {
            setVulkanObjectName(instanceTable_,
                                device_,
                                VK_OBJECT_TYPE_DEVICE_MEMORY,
                                reinterpret_cast<std::uint64_t>(record->memory),
                                record->label.c_str());
        }

        const std::scoped_lock lock(mutex_);
        heapRecords_.push_back(record);
        return heaps_.emplace(std::move(record));
    }

    Status VulkanResourceAllocator::destroyResourceHeap(ResourceHeapHandle heap)
    {
        const std::scoped_lock lock(mutex_);
        auto* stored = heaps_.find(heap);
        if (stored == nullptr || *stored == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The resource-heap handle is invalid or stale.",
            });
        }

        const auto record = *stored;
        const std::scoped_lock recordLock(record->mutex);
        record->alive = false;
        static_cast<void>(heaps_.erase(heap));
        return {};
    }

    namespace
    {

        [[nodiscard]] bool nativeRequirementsFitHeap(const VkMemoryRequirements& native,
                                                     const ResourceHeapRecord& heap,
                                                     std::uint64_t offset) noexcept
        {
            return (native.memoryTypeBits & (1U << heap.memoryTypeIndex)) != 0 && offset % native.alignment == 0 &&
                   native.size <= heap.desc.size && offset <= heap.desc.size - native.size;
        }

        [[nodiscard]] Status validateBufferPlacement(const std::shared_ptr<ResourceHeapRecord>& heap,
                                                     MemoryType memoryType,
                                                     const VkMemoryRequirements& native,
                                                     std::uint64_t offset)
        {
            if (heap == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A placed buffer requires a live resource heap.",
                });
            }
            if (heap->desc.memoryType != memoryType)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A placed buffer's memory type must match its resource heap.",
                });
            }
            if (!nativeRequirementsFitHeap(native, *heap, offset))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A placed buffer requires a heap with sufficient alignment and an aligned, in-bounds range.",
                });
            }
            return {};
        }

    } // namespace

    std::shared_ptr<ResourceHeapRecord> VulkanResourceAllocator::findResourceHeap(ResourceHeapHandle heap) const
    {
        const std::scoped_lock lock(mutex_);
        const auto* stored = heaps_.find(heap);
        if (stored == nullptr)
        {
            return nullptr;
        }
        return *stored;
    }

    Result<ResourceMemoryRequirements> VulkanResourceAllocator::resolvePlacedBufferMemoryRequirements(
        const BufferDesc& desc) const
    {
        if (desc.residencyPriority != ResidencyPriority::automatic ||
            desc.allocationFlags != ResourceAllocationFlag::none)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Placed buffers inherit heap residency and cannot request resource-level residency policy.",
            });
        }
        auto valid = validateBufferDesc(desc);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        auto requirements = queryBufferMemoryRequirements(desc);
        if (!requirements)
        {
            return std::unexpected(std::move(requirements.error()));
        }
        if (requirements->dedicatedAllocation)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A dedicated buffer cannot be placed on a shared resource heap.",
            });
        }
        return requirements;
    }

    VkDeviceAddress VulkanResourceAllocator::queryPlacedBufferDeviceAddress(VkBuffer buffer, BufferUsage usage) const
    {
        const bool addressable =
            bufferDeviceAddress_ &&
            hasAny(usage,
                   BufferUsage::accelerationStructureStorage | BufferUsage::accelerationStructureScratch |
                       BufferUsage::accelerationStructureBuildInput | BufferUsage::shaderBindingTable |
                       BufferUsage::indirect) &&
            table_ != nullptr && table_->vkGetBufferDeviceAddress != nullptr;
        if (!addressable)
        {
            return 0;
        }
        const VkBufferDeviceAddressInfo addressInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .pNext = {},
            .buffer = buffer,
        };
        return table_->vkGetBufferDeviceAddress(device_, &addressInfo);
    }

    Result<BufferHandle> VulkanResourceAllocator::createPlacedBuffer(const ResourcePlacement& placement,
                                                                     const BufferDesc& desc)
    {
        auto requirements = resolvePlacedBufferMemoryRequirements(desc);
        if (!requirements)
        {
            return std::unexpected(std::move(requirements.error()));
        }
        VulkanSharingFamilies sharing{};
        auto createInfo = bufferCreateInfo(desc, sharing);
        if (!createInfo)
        {
            return std::unexpected(std::move(createInfo.error()));
        }

        auto keepAlive = makeVulkanNativeKeepAlive(VulkanNativeKeepAlive::Kind::buffer, device_, table_);
        VkBuffer buffer = VK_NULL_HANDLE;
        const auto created = table_->vkCreateBuffer(device_, &*createInfo, nullptr, &buffer);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateBuffer"));
        }
        keepAlive->buffer = buffer;

        VkMemoryRequirements native{};
        table_->vkGetBufferMemoryRequirements(device_, buffer, &native);

        const auto heap = findResourceHeap(placement.heap);
        if (auto placeable = validateBufferPlacement(heap, desc.memoryType, native, placement.offset); !placeable)
        {
            return std::unexpected(std::move(placeable.error()));
        }

        auto lease = reservePlacement(heap, ResourceHeapClass::buffer, placement.offset, *requirements, "buffer");
        if (!lease)
        {
            return std::unexpected(std::move(lease.error()));
        }
        keepAlive->placement = *lease;

        const auto bound = table_->vkBindBufferMemory(device_, buffer, heap->memory, placement.offset);
        if (bound != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(bound), bound, "vkBindBufferMemory"));
        }

        BufferRecord record{};
        record.buffer = buffer;
        record.desc = desc;
        record.sharingFamilies = sharing;
        record.label = std::string(desc.label);
        record.allocationInfo.deviceMemory = heap->memory;
        record.allocationInfo.offset = placement.offset;
        record.allocationInfo.size = native.size;
        record.allocationId = nextAllocationId_.fetch_add(1, std::memory_order_relaxed);
        record.allocationBytes = requirements->size;
        record.hostVisible = heap->hostVisible;
        record.hostCoherent = heap->hostCoherent;
        record.memoryHeapIndex = heap->memoryHeapIndex;
        record.placement = std::move(*lease);
        if (!record.label.empty())
        {
            setVulkanObjectName(instanceTable_,
                                device_,
                                VK_OBJECT_TYPE_BUFFER,
                                reinterpret_cast<std::uint64_t>(record.buffer),
                                record.label.c_str());
        }
        record.keepAlive = std::move(keepAlive);
        record.deviceAddress = queryPlacedBufferDeviceAddress(record.buffer, desc.usage);

        const std::scoped_lock lock(mutex_);
        trackKeepAliveLocked(record.keepAlive);
        return emplaceRecordLocked(buffers_, std::move(record));
    }

    namespace
    {

        [[nodiscard]] Status validatePlacedTextureHeap(const std::shared_ptr<ResourceHeapRecord>& heap)
        {
            if (heap == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A placed texture requires a live resource heap.",
                });
            }
            if (heap->desc.memoryType != MemoryType::deviceLocal)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Placed textures require device-local resource heaps.",
                });
            }
            return {};
        }

        [[nodiscard]] constexpr VkImageCreateFlags placedImageCreateFlags(bool allowAliasing) noexcept
        {
            return allowAliasing ? VkImageCreateFlags{VK_IMAGE_CREATE_ALIAS_BIT} : VkImageCreateFlags{0};
        }

    } // namespace

    Result<ResourceMemoryRequirements> VulkanResourceAllocator::resolvePlacedTextureMemoryRequirements(
        const TextureDesc& desc, bool allowAliasing) const
    {
        if (desc.residencyPriority != ResidencyPriority::automatic ||
            desc.allocationFlags != ResourceAllocationFlag::none)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Placed textures inherit heap residency and cannot request resource-level residency policy.",
            });
        }
        auto valid = validateTextureDesc(desc);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        auto requirements = queryTextureMemoryRequirements(desc, placedImageCreateFlags(allowAliasing));
        if (!requirements)
        {
            return std::unexpected(std::move(requirements.error()));
        }
        if (requirements->dedicatedAllocation)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A dedicated texture cannot be placed on a shared resource heap.",
            });
        }
        return requirements;
    }

    Result<VkImage> VulkanResourceAllocator::createPlacedImage(const TextureDesc& desc,
                                                               VulkanSharingFamilies& sharing,
                                                               bool allowAliasing)
    {
        std::vector<VkFormat> nativeViewFormats;
        auto createInfo = imageCreateInfo(desc, nativeViewFormats, sharing);
        if (!createInfo)
        {
            return std::unexpected(std::move(createInfo.error()));
        }
        createInfo->flags |= placedImageCreateFlags(allowAliasing);
        VkImageFormatListCreateInfo list{
            .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
            .pNext = {},
            .viewFormatCount = static_cast<std::uint32_t>(nativeViewFormats.size()),
            .pViewFormats = nativeViewFormats.data(),
        };
        if (!desc.viewFormats.empty())
        {
            createInfo->pNext = &list;
        }

        VkImage image = VK_NULL_HANDLE;
        const auto created = table_->vkCreateImage(device_, &*createInfo, nullptr, &image);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateImage"));
        }
        return image;
    }

    Result<TextureHandle> VulkanResourceAllocator::createPlacedTexture(const ResourcePlacement& placement,
                                                                       const TextureDesc& desc)
    {
        const auto heap = findResourceHeap(placement.heap);
        const bool allowAliasing = heap != nullptr && hasAny(heap->desc.flags, ResourceHeapFlag::allowAliasing);
        auto requirements = resolvePlacedTextureMemoryRequirements(desc, allowAliasing);
        if (!requirements)
        {
            return std::unexpected(std::move(requirements.error()));
        }
        if (auto placeable = validatePlacedTextureHeap(heap); !placeable)
        {
            return std::unexpected(std::move(placeable.error()));
        }

        TextureRecord record{};
        record.viewFormats.assign(desc.viewFormats.begin(), desc.viewFormats.end());
        record.desc = desc;
        record.label = std::string(desc.label);
        record.desc.viewFormats = record.viewFormats;
        record.keepAlive = makeVulkanNativeKeepAlive(VulkanNativeKeepAlive::Kind::image, device_, table_);
        auto image = createPlacedImage(record.desc, record.sharingFamilies, allowAliasing);
        if (!image)
        {
            return std::unexpected(std::move(image.error()));
        }
        record.image = *image;
        record.keepAlive->image = record.image;

        VkMemoryRequirements native{};
        table_->vkGetImageMemoryRequirements(device_, record.image, &native);
        if (!nativeRequirementsFitHeap(native, *heap, placement.offset))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A placed texture requires a heap with sufficient alignment and an aligned, in-bounds range.",
            });
        }

        auto lease =
            reservePlacement(heap, textureResourceHeapClass(desc.usage), placement.offset, *requirements, "texture");
        if (!lease)
        {
            return std::unexpected(std::move(lease.error()));
        }
        record.keepAlive->placement = *lease;

        const auto bound = table_->vkBindImageMemory(device_, record.image, heap->memory, placement.offset);
        if (bound != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(bound), bound, "vkBindImageMemory"));
        }

        record.allocationInfo.deviceMemory = heap->memory;
        record.allocationInfo.offset = placement.offset;
        record.allocationInfo.size = native.size;
        record.allocationId = nextAllocationId_.fetch_add(1, std::memory_order_relaxed);
        record.allocationBytes = requirements->size;
        record.placement = std::move(*lease);
        if (!record.label.empty())
        {
            setVulkanObjectName(instanceTable_,
                                device_,
                                VK_OBJECT_TYPE_IMAGE,
                                reinterpret_cast<std::uint64_t>(record.image),
                                record.label.c_str());
        }
        const auto submittedLayers =
            record.desc.dimension == TextureDimension::texture3D ? 1U : record.desc.depthOrArrayLayers;
        record.submittedLayouts = std::make_shared<std::vector<VkImageLayout>>(
            static_cast<std::size_t>(record.desc.mipLevels) * submittedLayers * 2U, record.layout);
        const std::scoped_lock lock(mutex_);
        trackKeepAliveLocked(record.keepAlive);
        return emplaceRecordLocked(textures_, std::move(record));
    }
} // namespace rachis::vulkan
