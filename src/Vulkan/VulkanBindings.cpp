#include "Internal/VulkanBindings.h"

#include "Common/BindingPoolAccounting.h"
#include "Internal/PortableValidation.h"
#include "Internal/VulkanAcceleration.h"
#include "Internal/VulkanDiagnostics.h"
#include "Internal/VulkanImageBarriers.h"
#include "Internal/VulkanStageSupport.h"
#include "Internal/VulkanValidationHelpers.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <ranges>
#include <utility>

namespace rachis::vulkan
{
    namespace
    {

        constexpr std::uint32_t resourceHeapCapacity = 1'000'000;
        constexpr std::uint32_t samplerHeapCapacity = 2048;

        [[nodiscard]] constexpr bool isDynamicBuffer(BindingType type) noexcept
        {
            return type == BindingType::dynamicUniformBuffer || type == BindingType::dynamicReadOnlyStorageBuffer ||
                   type == BindingType::dynamicStorageBuffer;
        }

        [[nodiscard]] constexpr bool isBufferType(BindingType type) noexcept
        {
            return type == BindingType::uniformBuffer || type == BindingType::readOnlyStorageBuffer ||
                   type == BindingType::storageBuffer || isDynamicBuffer(type);
        }

        [[nodiscard]] constexpr BindingType staticBufferType(BindingType type) noexcept
        {
            switch (type)
            {
            case BindingType::dynamicUniformBuffer:
                return BindingType::uniformBuffer;
            case BindingType::dynamicReadOnlyStorageBuffer:
                return BindingType::readOnlyStorageBuffer;
            case BindingType::dynamicStorageBuffer:
                return BindingType::storageBuffer;
            default:
                return type;
            }
        }

        [[nodiscard]] constexpr bool isSamplerFeedback(BindingType type) noexcept
        {
            return type == BindingType::minMipFeedback || type == BindingType::mipRegionUsedFeedback;
        }

        [[nodiscard]] constexpr std::size_t rangeClassIndex(BindingType type) noexcept
        {
            switch (staticBufferType(type))
            {
            case BindingType::uniformBuffer:
                return 0;
            case BindingType::readOnlyStorageBuffer:
            case BindingType::sampledTexture:
            case BindingType::accelerationStructure:
                return 1;
            case BindingType::storageBuffer:
            case BindingType::storageTexture:
            case BindingType::minMipFeedback:
            case BindingType::mipRegionUsedFeedback:
                return 2;
            default:
                return 3;
            }
        }

        [[nodiscard]] constexpr VkDescriptorType nativeDescriptorType(BindingType type) noexcept
        {
            switch (type)
            {
            case BindingType::uniformBuffer:
                return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            case BindingType::dynamicUniformBuffer:
                return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            case BindingType::readOnlyStorageBuffer:
            case BindingType::storageBuffer:
                return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            case BindingType::dynamicReadOnlyStorageBuffer:
            case BindingType::dynamicStorageBuffer:
                return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
            case BindingType::sampledTexture:
                return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            case BindingType::storageTexture:
                return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            case BindingType::accelerationStructure:
                return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            default:
                return VK_DESCRIPTOR_TYPE_SAMPLER;
            }
        }

        [[nodiscard]] constexpr bool isValidVisibility(ShaderStage visibility) noexcept
        {
            const auto bits = static_cast<std::uint16_t>(visibility);
            const auto allowed = static_cast<std::uint16_t>(ShaderStage::all);
            return bits != 0 && (bits & ~allowed) == 0;
        }

        [[nodiscard]] constexpr bool rangesOverlap(const PushConstantRange& left,
                                                   const PushConstantRange& right) noexcept
        {
            return left.offset < right.offset + right.size && right.offset < left.offset + left.size;
        }

        [[nodiscard]] Result<std::uint32_t> validatePushConstantRanges(std::span<const PushConstantRange> ranges,
                                                                       ShaderStage enabledStages)
        {
            std::uint32_t byteSize = 0;
            for (std::size_t index = 0; index < ranges.size(); ++index)
            {
                const auto& range = ranges[index];
                if (!isValidVisibility(range.visibility) || range.size == 0 ||
                    range.offset % sizeof(std::uint32_t) != 0 || range.size % sizeof(std::uint32_t) != 0 ||
                    range.offset > maxPushConstantBytes || range.size > maxPushConstantBytes - range.offset)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Push-constant ranges require valid visibility and a non-empty, 4-byte aligned range within "
                        "128 bytes.",
                    });
                }
                for (std::size_t previous = 0; previous < index; ++previous)
                {
                    if (rangesOverlap(range, ranges[previous]) && hasAny(range.visibility, ranges[previous].visibility))
                    {
                        return std::unexpected(Error{
                            ErrorCode::invalidArgument,
                            0,
                            "Overlapping push-constant ranges cannot be visible to the same shader stage.",
                        });
                    }
                }
                if ((range.visibility & enabledStages) == ShaderStage::none)
                {
                    return std::unexpected(Error{
                        ErrorCode::unsupported,
                        0,
                        "A push-constant range is visible only to shader stages this Vulkan device has not enabled.",
                    });
                }
                byteSize = (std::max)(byteSize, range.offset + range.size);
            }
            return byteSize;
        }

        [[nodiscard]] const BindingLayoutEntry* findEntry(const std::vector<BindingLayoutEntry>& entries,
                                                          std::uint32_t binding) noexcept
        {
            const auto found = std::ranges::find_if(
                entries, [&](const BindingLayoutEntry& entry) { return entry.binding == binding; });
            return found == entries.end() ? nullptr : &*found;
        }

        [[nodiscard]] constexpr std::uint32_t logicalDescriptorCount(const BindingLayoutEntry& entry,
                                                                     std::uint32_t variableDescriptorCount) noexcept
        {
            return hasAny(entry.flags, BindingFlag::variableDescriptorCount) ? variableDescriptorCount : entry.count;
        }

        [[nodiscard]] std::array<std::uint32_t, bindingTypeCount> descriptorCounts(
            const std::vector<BindingLayoutEntry>& entries, std::uint32_t variableDescriptorCount)
        {
            std::array<std::uint32_t, bindingTypeCount> counts{};
            for (const auto& entry : entries)
            {
                counts[static_cast<std::size_t>(entry.type)] += logicalDescriptorCount(entry, variableDescriptorCount);
            }
            return counts;
        }

        [[nodiscard]] Result<std::uint64_t> resolvedBindingSize(const BufferDesc& desc, const BufferBinding& binding)
        {
            if (binding.offset >= desc.size)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The buffer binding offset is out of bounds.",
                });
            }
            const auto size = binding.size == wholeResource ? desc.size - binding.offset : binding.size;
            if (size == 0 || size > desc.size - binding.offset)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The buffer binding range is empty or out of bounds.",
                });
            }
            return size;
        }

        [[nodiscard]] Status validateBufferBinding(const BufferDesc& desc,
                                                   BindingType type,
                                                   const BufferBinding& binding,
                                                   std::uint64_t size,
                                                   const VulkanBufferBindingLimits& limits)
        {
            type = staticBufferType(type);
            if (type == BindingType::uniformBuffer)
            {
                const auto alignment = std::max<std::uint64_t>(1, limits.minUniformBufferOffsetAlignment);
                if (!hasAny(desc.usage, BufferUsage::constant) || binding.offset % alignment != 0 ||
                    size > limits.maxUniformBufferRange)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Constant-buffer bindings require constant usage, an offset aligned to "
                        "minUniformBufferOffsetAlignment, and a range within maxUniformBufferRange.",
                    });
                }
                return {};
            }
            if (!hasAny(desc.usage, BufferUsage::storage))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Storage-buffer bindings require storage usage.",
                });
            }
            const auto alignment = std::max<std::uint64_t>(1, limits.minStorageBufferOffsetAlignment);
            const auto stride = binding.structureByteStride == 0 ? 4U : binding.structureByteStride;
            if ((binding.structureByteStride != 0 && binding.structureByteStride % 4 != 0) ||
                binding.offset % alignment != 0 || size > limits.maxStorageBufferRange ||
                binding.offset % stride != 0 || size % stride != 0 ||
                size / stride > std::numeric_limits<std::uint32_t>::max())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Storage-buffer binding ranges must satisfy Vulkan storage alignment, stride, and "
                    "maxStorageBufferRange limits.",
                });
            }
            return {};
        }

    } // namespace

    VulkanBindingManager::LayoutKeepAlive::~LayoutKeepAlive()
    {
        if (device != VK_NULL_HANDLE && table != nullptr && table->vkDestroyDescriptorSetLayout != nullptr &&
            setLayout != VK_NULL_HANDLE)
        {
            table->vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
        }
    }

    VulkanBindingManager::BindingPoolSetLease::~BindingPoolSetLease()
    {
        if (!armed || state == nullptr)
        {
            return;
        }
        const std::scoped_lock lock(state->mutex);
        if (state->liveBindingSets > 0)
        {
            --state->liveBindingSets;
        }
    }

    VulkanBindingManager::BindingPoolVersionLease::~BindingPoolVersionLease()
    {
        if (state == nullptr)
        {
            return;
        }
        const std::scoped_lock lock(state->mutex);
        if (state->device != VK_NULL_HANDLE && state->table != nullptr &&
            state->table->vkFreeDescriptorSets != nullptr && descriptorSet != VK_NULL_HANDLE &&
            nativePool != VK_NULL_HANDLE)
        {
            state->table->vkFreeDescriptorSets(state->device, nativePool, 1, &descriptorSet);
            descriptorSet = VK_NULL_HANDLE;
        }
        if (!armed)
        {
            return;
        }
        for (std::size_t index = 0; index < counts.size(); ++index)
        {
            if (state->liveDescriptors[index] >= counts[index])
            {
                state->liveDescriptors[index] -= counts[index];
            }
        }
    }

    VulkanBindingManager::BindingPoolState::~BindingPoolState()
    {
        if (device == VK_NULL_HANDLE || table == nullptr || table->vkDestroyDescriptorPool == nullptr)
        {
            return;
        }
        if (pool != VK_NULL_HANDLE)
        {
            table->vkDestroyDescriptorPool(device, pool, nullptr);
        }
        if (updateAfterBindPool != VK_NULL_HANDLE)
        {
            table->vkDestroyDescriptorPool(device, updateAfterBindPool, nullptr);
        }
    }

    VulkanBindingManager::PipelineLayoutKeepAlive::~PipelineLayoutKeepAlive()
    {
        if (device != VK_NULL_HANDLE && table != nullptr && table->vkDestroyPipelineLayout != nullptr &&
            pipelineLayout != VK_NULL_HANDLE)
        {
            table->vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        }
    }

    VulkanBindingManager::~VulkanBindingManager()
    {
        shutdown();
    }

    void VulkanBindingManager::initialize(VkDevice device,
                                          const VolkDeviceTable* table,
                                          const DeviceCapabilities& capabilities,
                                          const VulkanInstanceTable* instanceTable,
                                          VulkanBufferBindingLimits bufferLimits,
                                          VulkanDescriptorLimits descriptorLimits,
                                          VulkanDescriptorLimits updateAfterBindLimits) noexcept
    {
        device_ = device;
        table_ = table;
        instanceTable_ = instanceTable;
        partiallyBoundSupported_ = capabilities.partiallyBoundBindingArrays;
        variableDescriptorCountSupported_ = capabilities.variableDescriptorCountBindingArrays;
        updateAfterBindSupported_ = capabilities.updateAfterBindBindingArrays;
        maxVariableResourceDescriptors_ = capabilities.limits.maxVariableResourceDescriptors;
        maxVariableSamplerDescriptors_ = capabilities.limits.maxVariableSamplerDescriptors;
        descriptorLimits_ = descriptorLimits;
        updateAfterBindLimits_ = updateAfterBindLimits;
        accelerationStructures_ = capabilities.rayTracing || capabilities.rayQuery;
        enabledShaderStages_ = enabledShaderStages(capabilities);
        bufferLimits_ = VulkanBufferBindingLimits{
            .minUniformBufferOffsetAlignment =
                bufferLimits.minUniformBufferOffsetAlignment != 0 ? bufferLimits.minUniformBufferOffsetAlignment : 256,
            .minStorageBufferOffsetAlignment =
                bufferLimits.minStorageBufferOffsetAlignment != 0 ? bufferLimits.minStorageBufferOffsetAlignment : 16,
            .maxUniformBufferRange =
                bufferLimits.maxUniformBufferRange != 0 ? bufferLimits.maxUniformBufferRange : 65536,
            .maxStorageBufferRange =
                bufferLimits.maxStorageBufferRange != 0 ? bufferLimits.maxStorageBufferRange : (1ULL << 30),
        };
    }

    void VulkanBindingManager::setAccelerationManager(VulkanAccelerationManager* acceleration) noexcept
    {
        accelerationManager_ = acceleration;
    }

    void VulkanBindingManager::shutdown() noexcept
    {
        detail::HandleTable<std::shared_ptr<BindingLayoutRecord>, BindingLayoutHandle> layouts{};
        detail::HandleTable<std::shared_ptr<BindingPoolState>, BindingPoolHandle> pools{};
        detail::HandleTable<std::shared_ptr<BindingSetRecord>, BindingSetHandle> sets{};
        detail::HandleTable<std::shared_ptr<PipelineLayoutRecord>, PipelineLayoutHandle> pipelineLayouts{};
        {
            const std::scoped_lock lock(mutex_);
            std::swap(pipelineLayouts, pipelineLayouts_);
            std::swap(sets, sets_);
            std::swap(pools, pools_);
            std::swap(layouts, layouts_);
            device_ = VK_NULL_HANDLE;
            table_ = nullptr;
            instanceTable_ = nullptr;
            accelerationStructures_ = false;
            accelerationManager_ = nullptr;
        }
    }

    namespace
    {

        struct BindingLayoutSupport final
        {
            bool partiallyBound = false;
            bool variableDescriptorCount = false;
            bool updateAfterBind = false;
            bool accelerationStructures = false;
            ShaderStage shaderStages = coreShaderStages;
        };

        [[nodiscard]] Status validateLayoutEntrySupport(const BindingLayoutEntry& entry,
                                                        const BindingLayoutSupport& support)
        {
            if ((entry.visibility & support.shaderStages) == ShaderStage::none)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "A binding is visible only to shader stages this Vulkan device has not enabled.",
                });
            }
            if (hasAny(entry.flags, BindingFlag::variableDescriptorCount) && !support.variableDescriptorCount)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Variable-count binding arrays require descriptor indexing.",
                });
            }
            if (hasAny(entry.flags, BindingFlag::partiallyBound) && !support.partiallyBound)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Partially-bound binding arrays require descriptor indexing.",
                });
            }
            if (hasAny(entry.flags, BindingFlag::updateAfterBind) && !support.updateAfterBind)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Update-after-bind binding arrays require the descriptor-indexing update-after-bind features.",
                });
            }
            if (entry.type == BindingType::accelerationStructure && !support.accelerationStructures)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Acceleration-structure bindings require acceleration-structure support.",
                });
            }
            if (isSamplerFeedback(entry.type))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Sampler-feedback bindings require sampler-feedback support.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateLayoutEntry(const BindingLayoutEntry& entry, const BindingLayoutSupport& support)
        {
            const auto flags = static_cast<std::uint8_t>(entry.flags);
            const auto allowedFlags = static_cast<std::uint8_t>(BindingFlag::partiallyBound) |
                                      static_cast<std::uint8_t>(BindingFlag::variableDescriptorCount) |
                                      static_cast<std::uint8_t>(BindingFlag::updateAfterBind);
            const bool variable = hasAny(entry.flags, BindingFlag::variableDescriptorCount);
            const bool unbounded = entry.count == unboundedBindingCount;
            const bool registerRangeOverflows =
                !unbounded && entry.binding > std::numeric_limits<std::uint32_t>::max() - (entry.count - 1);
            const bool hasUnknownFlags = (flags & ~allowedFlags) != 0;
            if (entry.count == 0 || registerRangeOverflows || !isValidVisibility(entry.visibility) || hasUnknownFlags)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Binding layout entries require a non-empty register range, valid visibility, and valid flags.",
                });
            }
            if (variable != unbounded)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "An unbounded binding count and the variable-descriptor-count flag must be specified together.",
                });
            }
            auto supported = validateLayoutEntrySupport(entry, support);
            if (!supported)
            {
                return supported;
            }
            if (isDynamicBuffer(entry.type) && entry.flags != BindingFlag::none)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Dynamic buffer bindings cannot be partially bound or variable length.",
                });
            }
            if (hasAny(entry.flags, BindingFlag::updateAfterBind) && !detail::allowsUpdateAfterBind(entry.type))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Update-after-bind applies only to sampled-texture, storage-texture, and storage-buffer bindings.",
                });
            }
            return {};
        }

        [[nodiscard]] Result<bool> validateUpdateAfterBindLayout(const std::vector<BindingLayoutEntry>& entries)
        {
            const bool updateAfterBind = std::ranges::any_of(entries, [](const BindingLayoutEntry& entry) {
                return hasAny(entry.flags, BindingFlag::updateAfterBind);
            });
            const bool dynamicBuffers = std::ranges::any_of(
                entries, [](const BindingLayoutEntry& entry) { return isDynamicBuffer(entry.type); });
            if (updateAfterBind && dynamicBuffers)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A binding layout with an update-after-bind entry cannot contain dynamic buffer bindings.",
                });
            }
            return updateAfterBind;
        }

        [[nodiscard]] Result<std::optional<std::uint32_t>> findVariableBinding(
            const std::vector<BindingLayoutEntry>& sortedEntries)
        {
            std::optional<std::uint32_t> variableBinding;
            for (const auto& entry : sortedEntries)
            {
                if (!hasAny(entry.flags, BindingFlag::variableDescriptorCount))
                {
                    continue;
                }
                if (variableBinding.has_value() || entry.binding != sortedEntries.back().binding)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A binding layout may have one variable-count entry and it must use the largest binding "
                        "number.",
                    });
                }
                variableBinding = entry.binding;
            }
            return variableBinding;
        }

        struct FixedDescriptorCounts final
        {
            std::uint32_t resource = 0;
            std::uint32_t sampler = 0;
            std::uint32_t dynamicBuffer = 0;
        };

        [[nodiscard]] Status addFixedDescriptorCount(const BindingLayoutEntry& entry, FixedDescriptorCounts& counts)
        {
            if (entry.type == BindingType::sampler)
            {
                if (entry.count > samplerHeapCapacity - counts.sampler)
                {
                    return std::unexpected(Error{
                        ErrorCode::unsupported,
                        0,
                        "The binding layout exceeds the portable sampler descriptor heap limit.",
                    });
                }
                counts.sampler += entry.count;
                return {};
            }
            if (isDynamicBuffer(entry.type) &&
                entry.count > maxDynamicBufferOffsetsPerBindingSet - counts.dynamicBuffer)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "A binding layout cannot declare more than eight dynamic buffer offsets.",
                });
            }
            if (isDynamicBuffer(entry.type))
            {
                counts.dynamicBuffer += entry.count;
            }
            if (entry.count > resourceHeapCapacity - counts.resource)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The binding layout exceeds the RHI resource descriptor heap capacity.",
                });
            }
            counts.resource += entry.count;
            return {};
        }

        [[nodiscard]] Result<FixedDescriptorCounts> validateRegisterRangesAndCountDescriptors(
            const std::vector<BindingLayoutEntry>& sortedEntries)
        {
            std::array<std::uint64_t, 4> rangeEnds{};
            std::array<bool, 4> hasRange{};
            FixedDescriptorCounts counts{};
            for (const auto& entry : sortedEntries)
            {
                if (std::ranges::count(sortedEntries, entry.binding, &BindingLayoutEntry::binding) != 1)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Binding layout binding numbers must be unique.",
                    });
                }
                const auto classIndex = rangeClassIndex(entry.type);
                if (hasRange[classIndex] && entry.binding < rangeEnds[classIndex])
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Binding arrays cannot overlap another HLSL register range in the same descriptor class.",
                    });
                }
                rangeEnds[classIndex] = static_cast<std::uint64_t>(entry.binding) + entry.count;
                hasRange[classIndex] = true;
                if (hasAny(entry.flags, BindingFlag::variableDescriptorCount))
                {
                    continue;
                }
                auto counted = addFixedDescriptorCount(entry, counts);
                if (!counted)
                {
                    return std::unexpected(std::move(counted.error()));
                }
            }
            return counts;
        }

        [[nodiscard]] constexpr VkDescriptorBindingFlags nativeBindingFlags(BindingFlag flags) noexcept
        {
            VkDescriptorBindingFlags native = 0;
            if (hasAny(flags, BindingFlag::partiallyBound))
            {
                native |= VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
            }
            if (hasAny(flags, BindingFlag::variableDescriptorCount))
            {
                native |= VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
            }
            if (hasAny(flags, BindingFlag::updateAfterBind))
            {
                native |= VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                          VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
                          VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
            }
            return native;
        }

        enum class DescriptorLimitClass : std::uint8_t
        {
            sampler,
            uniformBuffer,
            storageBuffer,
            sampledImage,
            storageImage,
            accelerationStructure,
        };

        constexpr std::size_t descriptorLimitClassCount =
            static_cast<std::size_t>(DescriptorLimitClass::accelerationStructure) + 1U;
        constexpr std::size_t shaderStageBitCount = 14;

        [[nodiscard]] constexpr std::optional<DescriptorLimitClass> descriptorLimitClass(BindingType type) noexcept
        {
            switch (staticBufferType(type))
            {
            case BindingType::sampler:
                return DescriptorLimitClass::sampler;
            case BindingType::uniformBuffer:
                return DescriptorLimitClass::uniformBuffer;
            case BindingType::readOnlyStorageBuffer:
            case BindingType::storageBuffer:
                return DescriptorLimitClass::storageBuffer;
            case BindingType::sampledTexture:
                return DescriptorLimitClass::sampledImage;
            case BindingType::storageTexture:
                return DescriptorLimitClass::storageImage;
            case BindingType::accelerationStructure:
                return DescriptorLimitClass::accelerationStructure;
            default:
                return std::nullopt;
            }
        }

        [[nodiscard]] constexpr bool countsAgainstStageResources(DescriptorLimitClass limitClass) noexcept
        {
            return limitClass != DescriptorLimitClass::sampler &&
                   limitClass != DescriptorLimitClass::accelerationStructure;
        }

        [[nodiscard]] constexpr std::uint32_t perStageDescriptorLimit(const VulkanDescriptorLimits& limits,
                                                                      DescriptorLimitClass limitClass) noexcept
        {
            switch (limitClass)
            {
            case DescriptorLimitClass::sampler:
                return limits.maxPerStageDescriptorSamplers;
            case DescriptorLimitClass::uniformBuffer:
                return limits.maxPerStageDescriptorUniformBuffers;
            case DescriptorLimitClass::storageBuffer:
                return limits.maxPerStageDescriptorStorageBuffers;
            case DescriptorLimitClass::sampledImage:
                return limits.maxPerStageDescriptorSampledImages;
            case DescriptorLimitClass::storageImage:
                return limits.maxPerStageDescriptorStorageImages;
            case DescriptorLimitClass::accelerationStructure:
                return limits.maxPerStageDescriptorAccelerationStructures;
            }
            return 0;
        }

        [[nodiscard]] constexpr std::uint32_t pipelineDescriptorLimit(const VulkanDescriptorLimits& limits,
                                                                      DescriptorLimitClass limitClass) noexcept
        {
            switch (limitClass)
            {
            case DescriptorLimitClass::sampler:
                return limits.maxDescriptorSetSamplers;
            case DescriptorLimitClass::uniformBuffer:
                return limits.maxDescriptorSetUniformBuffers;
            case DescriptorLimitClass::storageBuffer:
                return limits.maxDescriptorSetStorageBuffers;
            case DescriptorLimitClass::sampledImage:
                return limits.maxDescriptorSetSampledImages;
            case DescriptorLimitClass::storageImage:
                return limits.maxDescriptorSetStorageImages;
            case DescriptorLimitClass::accelerationStructure:
                return limits.maxDescriptorSetAccelerationStructures;
            }
            return 0;
        }

        [[nodiscard]] constexpr bool visibleToStage(ShaderStage visibility, std::size_t stage) noexcept
        {
            return hasAny(visibility, static_cast<ShaderStage>(1U << stage));
        }

        struct DescriptorTotals final
        {
            std::array<std::array<std::uint64_t, descriptorLimitClassCount>, shaderStageBitCount> perStage{};
            std::array<std::uint64_t, shaderStageBitCount> perStageResources{};
            std::array<std::uint64_t, descriptorLimitClassCount> pipeline{};
            std::uint64_t uniformBuffersDynamic = 0;
            std::uint64_t storageBuffersDynamic = 0;
        };

        void addDescriptors(DescriptorTotals& totals,
                            const BindingLayoutEntry& entry,
                            std::uint32_t count,
                            ShaderStage enabledStages) noexcept
        {
            const auto visibility = entry.visibility & enabledStages;
            const auto limitClass = descriptorLimitClass(entry.type);
            if (!limitClass.has_value())
            {
                return;
            }
            const auto classIndex = static_cast<std::size_t>(*limitClass);
            totals.pipeline[classIndex] += count;
            if (entry.type == BindingType::dynamicUniformBuffer)
            {
                totals.uniformBuffersDynamic += count;
            }
            else if (isDynamicBuffer(entry.type))
            {
                totals.storageBuffersDynamic += count;
            }
            for (std::size_t stage = 0; stage < shaderStageBitCount; ++stage)
            {
                if (!visibleToStage(visibility, stage))
                {
                    continue;
                }
                totals.perStage[stage][classIndex] += count;
                if (countsAgainstStageResources(*limitClass))
                {
                    totals.perStageResources[stage] += count;
                }
            }
        }

        [[nodiscard]] bool withinDescriptorLimits(const DescriptorTotals& totals,
                                                  const VulkanDescriptorLimits& limits) noexcept
        {
            for (std::size_t classIndex = 0; classIndex < descriptorLimitClassCount; ++classIndex)
            {
                const auto limitClass = static_cast<DescriptorLimitClass>(classIndex);
                if (totals.pipeline[classIndex] > pipelineDescriptorLimit(limits, limitClass))
                {
                    return false;
                }
                const auto perStageLimit = perStageDescriptorLimit(limits, limitClass);
                if (std::ranges::any_of(totals.perStage,
                                        [&](const auto& stage) { return stage[classIndex] > perStageLimit; }))
                {
                    return false;
                }
            }
            return totals.uniformBuffersDynamic <= limits.maxDescriptorSetUniformBuffersDynamic &&
                   totals.storageBuffersDynamic <= limits.maxDescriptorSetStorageBuffersDynamic &&
                   std::ranges::all_of(totals.perStageResources,
                                       [&](std::uint64_t count) { return count <= limits.maxPerStageResources; });
        }

        [[nodiscard]] constexpr std::uint64_t remainingDescriptors(std::uint32_t limit, std::uint64_t used) noexcept
        {
            return used >= limit ? 0 : limit - used;
        }

        [[nodiscard]] std::uint32_t variableDescriptorCapacity(const std::vector<BindingLayoutEntry>& entries,
                                                               const BindingLayoutEntry& variable,
                                                               std::uint32_t classCapacity,
                                                               const VulkanDescriptorLimits& limits,
                                                               ShaderStage enabledStages) noexcept
        {
            const auto limitClass = descriptorLimitClass(variable.type);
            if (!limitClass.has_value())
            {
                return 0;
            }
            DescriptorTotals fixed{};
            for (const auto& entry : entries)
            {
                if (!hasAny(entry.flags, BindingFlag::variableDescriptorCount))
                {
                    addDescriptors(fixed, entry, entry.count, enabledStages);
                }
            }
            const auto classIndex = static_cast<std::size_t>(*limitClass);
            auto capacity = (std::min)(std::uint64_t{classCapacity},
                                       remainingDescriptors(
                                           variableArrayDescriptorLimit(pipelineDescriptorLimit(limits, *limitClass)),
                                           fixed.pipeline[classIndex]));
            for (std::size_t stage = 0; stage < shaderStageBitCount; ++stage)
            {
                if (!visibleToStage(variable.visibility & enabledStages, stage))
                {
                    continue;
                }
                capacity = (std::min)(capacity,
                                      remainingDescriptors(
                                          variableArrayDescriptorLimit(perStageDescriptorLimit(limits, *limitClass)),
                                          fixed.perStage[stage][classIndex]));
                if (countsAgainstStageResources(*limitClass))
                {
                    capacity =
                        (std::min)(capacity,
                                   remainingDescriptors(variableArrayDescriptorLimit(limits.maxPerStageResources),
                                                        fixed.perStageResources[stage]));
                }
            }
            return static_cast<std::uint32_t>(capacity);
        }

        struct NativeLayoutBindings final
        {
            std::vector<VkDescriptorSetLayoutBinding> bindings;
            std::vector<VkDescriptorBindingFlags> flags;
        };

        [[nodiscard]] Result<NativeLayoutBindings> buildNativeLayoutBindings(
            const std::vector<BindingLayoutEntry>& entries,
            std::uint32_t variableDescriptorCapacity,
            ShaderStage enabledStages)
        {
            NativeLayoutBindings native;
            native.bindings.reserve(entries.size());
            native.flags.reserve(entries.size());
            for (const auto& entry : entries)
            {
                const auto variable = hasAny(entry.flags, BindingFlag::variableDescriptorCount);
                const auto descriptorCount = variable ? variableDescriptorCapacity : entry.count;
                if (variable && descriptorCount == 0)
                {
                    return std::unexpected(Error{
                        ErrorCode::unsupported,
                        0,
                        "Variable-count binding arrays require descriptor capacity within the device's per-stage and "
                        "per-set descriptor limits.",
                    });
                }
                native.bindings.push_back(VkDescriptorSetLayoutBinding{
                    .binding = entry.binding,
                    .descriptorType = nativeDescriptorType(entry.type),
                    .descriptorCount = descriptorCount,
                    .stageFlags = nativeShaderStageFlags(entry.visibility & enabledStages),
                    .pImmutableSamplers = {},
                });
                native.flags.push_back(nativeBindingFlags(entry.flags));
            }
            return native;
        }

    } // namespace

    Result<std::shared_ptr<VulkanBindingManager::LayoutKeepAlive>> VulkanBindingManager::createNativeSetLayout(
        const BindingLayoutRecord& record) const
    {
        auto native =
            buildNativeLayoutBindings(record.entries, record.variableDescriptorCapacity, enabledShaderStages_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
            .pNext = {},
            .bindingCount = static_cast<std::uint32_t>(native->flags.size()),
            .pBindingFlags = native->flags.empty() ? nullptr : native->flags.data(),
        };
        const VkDescriptorSetLayoutCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = native->flags.empty() ? nullptr : &flagsInfo,
            .flags = record.updateAfterBind
                         ? VkDescriptorSetLayoutCreateFlags{VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT}
                         : VkDescriptorSetLayoutCreateFlags{},
            .bindingCount = static_cast<std::uint32_t>(native->bindings.size()),
            .pBindings = native->bindings.empty() ? nullptr : native->bindings.data(),
        };
        auto keepAlive = std::make_shared<LayoutKeepAlive>();
        keepAlive->device = device_;
        keepAlive->table = table_;
        const auto created = table_->vkCreateDescriptorSetLayout(device_, &createInfo, nullptr, &keepAlive->setLayout);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(
                makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateDescriptorSetLayout"));
        }
        if (!record.label.empty())
        {
            setVulkanObjectName(instanceTable_,
                                device_,
                                VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                reinterpret_cast<std::uint64_t>(keepAlive->setLayout),
                                record.label.c_str());
        }
        return keepAlive;
    }

    Result<BindingLayoutHandle> VulkanBindingManager::createBindingLayout(const BindingLayoutDesc& desc)
    {
        const BindingLayoutSupport support{
            .partiallyBound = partiallyBoundSupported_,
            .variableDescriptorCount = variableDescriptorCountSupported_,
            .updateAfterBind = updateAfterBindSupported_,
            .accelerationStructures = accelerationStructures_,
            .shaderStages = enabledShaderStages_,
        };
        auto record = std::make_shared<BindingLayoutRecord>();
        record->label = std::string(desc.label);
        record->entries.reserve(desc.entries.size());
        for (const auto& entry : desc.entries)
        {
            auto valid = validateLayoutEntry(entry, support);
            if (!valid)
            {
                return std::unexpected(std::move(valid.error()));
            }
            record->entries.push_back(entry);
        }

        std::ranges::sort(record->entries, {}, &BindingLayoutEntry::binding);
        auto variableBinding = findVariableBinding(record->entries);
        if (!variableBinding)
        {
            return std::unexpected(std::move(variableBinding.error()));
        }
        record->variableBinding = *variableBinding;
        auto updateAfterBind = validateUpdateAfterBindLayout(record->entries);
        if (!updateAfterBind)
        {
            return std::unexpected(std::move(updateAfterBind.error()));
        }
        record->updateAfterBind = *updateAfterBind;
        auto counts = validateRegisterRangesAndCountDescriptors(record->entries);
        if (!counts)
        {
            return std::unexpected(std::move(counts.error()));
        }
        record->resourceDescriptorCount = counts->resource;
        record->samplerDescriptorCount = counts->sampler;
        record->dynamicBufferCount = counts->dynamicBuffer;
        if (record->variableBinding.has_value())
        {
            const auto* variable = findEntry(record->entries, *record->variableBinding);
            const auto classCapacity = variable->type == BindingType::sampler ? maxVariableSamplerDescriptors_
                                                                              : maxVariableResourceDescriptors_;
            record->variableDescriptorCapacity =
                variableDescriptorCapacity(record->entries,
                                           *variable,
                                           classCapacity,
                                           record->updateAfterBind ? updateAfterBindLimits_ : descriptorLimits_,
                                           enabledShaderStages_);
        }

        if (table_ == nullptr || table_->vkCreateDescriptorSetLayout == nullptr || device_ == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The Vulkan binding manager is not initialized.",
            });
        }

        auto keepAlive = createNativeSetLayout(*record);
        if (!keepAlive)
        {
            return std::unexpected(std::move(keepAlive.error()));
        }
        record->keepAlive = std::move(*keepAlive);
        const std::scoped_lock lock(mutex_);
        return layouts_.emplace(std::move(record));
    }

    Status VulkanBindingManager::destroyBindingLayout(BindingLayoutHandle layout)
    {
        std::shared_ptr<BindingLayoutRecord> removed;
        {
            const std::scoped_lock lock(mutex_);
            auto* found = layouts_.find(layout);
            if (found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The binding layout handle is invalid or stale.",
                });
            }
            removed = *found;
            static_cast<void>(layouts_.erase(layout));
        }
        return {};
    }

    namespace
    {

        struct PoolSizePlan final
        {
            std::array<std::uint32_t, bindingTypeCount> descriptorCapacity{};
            std::vector<VkDescriptorPoolSize> nativeSizes;
            std::uint64_t nativeDescriptorCapacity = 0;
        };

        [[nodiscard]] Result<PoolSizePlan> planPoolSizes(std::span<const BindingPoolSize> sizes,
                                                         bool accelerationStructures)
        {
            PoolSizePlan plan;
            std::array<bool, bindingTypeCount> seen{};
            std::uint64_t resourceDescriptors = 0;
            std::uint64_t samplerDescriptors = 0;
            for (const auto& size : sizes)
            {
                const auto index = static_cast<std::size_t>(size.type);
                if (index >= bindingTypeCount || size.descriptorCount == 0 || seen[index])
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Binding-pool sizes require unique valid binding types and non-zero descriptor counts.",
                    });
                }
                seen[index] = true;
                plan.descriptorCapacity[index] = size.descriptorCount;
                if (size.type == BindingType::sampler)
                {
                    samplerDescriptors += size.descriptorCount;
                }
                else
                {
                    resourceDescriptors += size.descriptorCount;
                }
                if (isSamplerFeedback(size.type) ||
                    (size.type == BindingType::accelerationStructure && !accelerationStructures))
                {
                    continue;
                }
                plan.nativeDescriptorCapacity += size.descriptorCount;
                plan.nativeSizes.push_back(VkDescriptorPoolSize{
                    .type = nativeDescriptorType(size.type),
                    .descriptorCount = size.descriptorCount * 2,
                });
            }
            if (resourceDescriptors > resourceHeapCapacity || samplerDescriptors > samplerHeapCapacity)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A binding pool exceeds one shader-visible descriptor heap's portable capacity.",
                });
            }
            return plan;
        }

        [[nodiscard]] constexpr std::uint32_t nativeDescriptorSetCapacity(
            std::uint32_t setCount, std::uint64_t nativeDescriptorCapacity) noexcept
        {
            return static_cast<std::uint32_t>((std::min)(std::uint64_t{setCount} + nativeDescriptorCapacity,
                                                         std::uint64_t{std::numeric_limits<std::uint32_t>::max()}));
        }

    } // namespace

    Status VulkanBindingManager::createNativeDescriptorPool(BindingPoolState& state,
                                                            std::span<const VkDescriptorPoolSize> poolSizes,
                                                            std::uint32_t maxSets,
                                                            VkDescriptorPoolCreateFlags extraFlags,
                                                            VkDescriptorPool& target) const
    {
        const VkDescriptorPoolCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = {},
            .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | extraFlags,
            .maxSets = maxSets,
            .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.empty() ? nullptr : poolSizes.data(),
        };
        const auto created = table_->vkCreateDescriptorPool(device_, &createInfo, nullptr, &target);
        if (created != VK_SUCCESS)
        {
            target = VK_NULL_HANDLE;
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateDescriptorPool"));
        }
        if (!state.label.empty())
        {
            setVulkanObjectName(instanceTable_,
                                device_,
                                VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                reinterpret_cast<std::uint64_t>(target),
                                state.label.c_str());
        }
        return {};
    }

    Result<VkDescriptorPool> VulkanBindingManager::nativePoolLocked(BindingPoolState& pool, bool updateAfterBind) const
    {
        if (!updateAfterBind)
        {
            return pool.pool;
        }
        if (pool.updateAfterBindPool == VK_NULL_HANDLE)
        {
            auto created = createNativeDescriptorPool(pool,
                                                      pool.nativeSizes,
                                                      pool.nativeMaxSets,
                                                      VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
                                                      pool.updateAfterBindPool);
            if (!created)
            {
                return std::unexpected(std::move(created.error()));
            }
        }
        return pool.updateAfterBindPool;
    }

    Result<BindingPoolHandle> VulkanBindingManager::createBindingPool(const BindingPoolDesc& desc)
    {
        if (desc.maxBindingSets == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A binding pool requires a non-zero binding-set capacity.",
            });
        }
        if (desc.sizes.size() > bindingTypeCount)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A binding pool cannot contain more than one size for each binding type.",
            });
        }
        if (table_ == nullptr || table_->vkCreateDescriptorPool == nullptr || device_ == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The Vulkan binding manager is not initialized.",
            });
        }

        auto plan = planPoolSizes(desc.sizes, accelerationStructures_);
        if (!plan)
        {
            return std::unexpected(std::move(plan.error()));
        }

        auto state = std::make_shared<BindingPoolState>();
        state->device = device_;
        state->table = table_;
        state->maxBindingSets = desc.maxBindingSets;
        state->label = std::string(desc.label);
        state->descriptorCapacity = plan->descriptorCapacity;
        state->nativeMaxSets = nativeDescriptorSetCapacity(desc.maxBindingSets, plan->nativeDescriptorCapacity);
        auto created = createNativeDescriptorPool(*state, plan->nativeSizes, state->nativeMaxSets, 0, state->pool);
        if (!created)
        {
            return std::unexpected(std::move(created.error()));
        }
        state->nativeSizes = std::move(plan->nativeSizes);
        const std::scoped_lock lock(mutex_);
        return pools_.emplace(std::move(state));
    }

    Status VulkanBindingManager::destroyBindingPool(BindingPoolHandle pool)
    {
        std::shared_ptr<BindingPoolState> removed;
        {
            const std::scoped_lock lock(mutex_);
            auto* found = pools_.find(pool);
            if (found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The binding pool handle is invalid or stale.",
                });
            }
            removed = *found;
            {
                const std::scoped_lock poolLock(removed->mutex);
                removed->logicalAlive = false;
            }
            pools_.erase(pool);
        }
        return {};
    }

    namespace
    {

        [[nodiscard]] std::optional<std::size_t> findSlotIndex(const std::vector<BindingLayoutEntry>& entries,
                                                               std::uint32_t variableDescriptorCount,
                                                               const BindingWrite& write)
        {
            std::size_t index = 0;
            for (const auto& entry : entries)
            {
                const auto count = logicalDescriptorCount(entry, variableDescriptorCount);
                if (entry.binding == write.binding)
                {
                    if (write.arrayElement >= count)
                    {
                        return std::nullopt;
                    }
                    return index + write.arrayElement;
                }
                index += count;
            }
            return std::nullopt;
        }

        [[nodiscard]] Status clearPartiallyBoundSlot(const std::vector<BindingLayoutEntry>& entries,
                                                     std::uint32_t binding,
                                                     VulkanBindingManager::BindingSlot& slot)
        {
            const auto* entry = findEntry(entries, binding);
            if (entry == nullptr || !hasAny(entry->flags, BindingFlag::partiallyBound))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Only a partially-bound layout entry may be cleared.",
                });
            }
            slot.resource = std::monostate{};
            slot.swapchainView = false;
            slot.keepAlive.reset();
            slot.buffer = VK_NULL_HANDLE;
            slot.imageView = VK_NULL_HANDLE;
            slot.sampler = VK_NULL_HANDLE;
            slot.accelerationStructure = VK_NULL_HANDLE;
            slot.accelerationStructureUse = {};
            slot.range = 0;
            slot.bufferSize = 0;
            return {};
        }

        [[nodiscard]] Status bindBuffer(VulkanResourceAllocator& resources,
                                        const BindingWrite& write,
                                        const VulkanBufferBindingLimits& bufferLimits,
                                        VulkanBindingManager::BindingSlot& slot)
        {
            const auto* binding = std::get_if<BufferBinding>(&write.resource);
            if (binding == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The binding write resource type does not match its buffer layout entry.",
                });
            }
            auto source = resources.resolveBindingBuffer(binding->buffer);
            if (!source)
            {
                return std::unexpected(std::move(source.error()));
            }
            auto size = resolvedBindingSize(source->desc, *binding);
            if (!size)
            {
                return std::unexpected(std::move(size.error()));
            }
            auto valid = validateBufferBinding(source->desc, slot.type, *binding, *size, bufferLimits);
            if (!valid)
            {
                return valid;
            }
            auto retained = *binding;
            retained.size = *size;
            slot.resource = retained;
            slot.keepAlive = std::move(source->keepAlive);
            slot.buffer = source->buffer;
            slot.imageView = VK_NULL_HANDLE;
            slot.sampler = VK_NULL_HANDLE;
            slot.accelerationStructure = VK_NULL_HANDLE;
            slot.accelerationStructureUse = {};
            slot.range = *size;
            slot.bufferSize = source->desc.size;
            return {};
        }

        struct SwapchainViewBinding final
        {
            std::shared_ptr<void> view;
            std::shared_ptr<VulkanSwapchainImageLease> lease;
        };

        [[nodiscard]] Status bindTextureView(VulkanResourceAllocator& resources,
                                             const BindingWrite& write,
                                             VulkanBindingManager::BindingSlot& slot)
        {
            const auto* handle = std::get_if<TextureViewHandle>(&write.resource);
            if (handle == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The binding write resource type does not match its texture layout entry.",
                });
            }
            auto view = resources.resolveBindingTextureView(*handle);
            if (!view)
            {
                return std::unexpected(std::move(view.error()));
            }
            const auto expected =
                slot.type == BindingType::sampledTexture ? TextureViewUsage::sampled : TextureViewUsage::storage;
            if (view->usage != expected)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The texture view usage does not match the binding layout entry.",
                });
            }
            slot.resource = *handle;
            slot.swapchainView = view->swapchainLease != nullptr;
            slot.keepAlive = slot.swapchainView ? std::make_shared<SwapchainViewBinding>(SwapchainViewBinding{
                                                      .view = std::move(view->keepAlive),
                                                      .lease = std::move(view->swapchainLease),
                                                  })
                                                : std::move(view->keepAlive);
            slot.buffer = VK_NULL_HANDLE;
            slot.imageView = view->view;
            slot.sampler = VK_NULL_HANDLE;
            slot.accelerationStructure = VK_NULL_HANDLE;
            slot.accelerationStructureUse = {};
            slot.range = 0;
            slot.bufferSize = 0;
            return {};
        }

        [[nodiscard]] Status bindAccelerationStructure(VulkanAccelerationManager* acceleration,
                                                       const BindingWrite& write,
                                                       VulkanBindingManager::BindingSlot& slot)
        {
            const auto* handle = std::get_if<AccelerationStructureHandle>(&write.resource);
            if (handle == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The binding write resource type does not match its acceleration-structure layout entry.",
                });
            }
            if (acceleration == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Acceleration-structure bindings require an initialized acceleration-structure manager.",
                });
            }
            auto source = acceleration->resolve(*handle);
            if (!source)
            {
                return std::unexpected(std::move(source.error()));
            }
            if (source->type != AccelerationStructureType::topLevel)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Only a top-level acceleration structure can be bound for shader traversal.",
                });
            }
            slot.resource = *handle;
            slot.keepAlive = std::move(source->keepAlive);
            slot.buffer = VK_NULL_HANDLE;
            slot.imageView = VK_NULL_HANDLE;
            slot.sampler = VK_NULL_HANDLE;
            slot.accelerationStructure = source->accelerationStructure;
            slot.accelerationStructureUse = std::move(source->use);
            slot.range = 0;
            slot.bufferSize = 0;
            return {};
        }

        [[nodiscard]] Status bindSampler(VulkanResourceAllocator& resources,
                                         const BindingWrite& write,
                                         VulkanBindingManager::BindingSlot& slot)
        {
            const auto* handle = std::get_if<SamplerHandle>(&write.resource);
            if (handle == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The binding write resource type does not match its sampler layout entry.",
                });
            }
            auto sampler = resources.resolveBindingSampler(*handle);
            if (!sampler)
            {
                return std::unexpected(std::move(sampler.error()));
            }
            slot.resource = *handle;
            slot.keepAlive = std::move(sampler->keepAlive);
            slot.buffer = VK_NULL_HANDLE;
            slot.imageView = VK_NULL_HANDLE;
            slot.sampler = sampler->sampler;
            slot.accelerationStructure = VK_NULL_HANDLE;
            slot.accelerationStructureUse = {};
            slot.range = 0;
            slot.bufferSize = 0;
            return {};
        }

        struct BindingResourceSources final
        {
            VulkanResourceAllocator& resources;
            VulkanAccelerationManager* acceleration;
            const VulkanBufferBindingLimits& bufferLimits;
        };

        [[nodiscard]] Status bindResource(const BindingResourceSources& sources,
                                          const BindingWrite& write,
                                          VulkanBindingManager::BindingSlot& slot)
        {
            if (isBufferType(slot.type))
            {
                return bindBuffer(sources.resources, write, sources.bufferLimits, slot);
            }
            if (slot.type == BindingType::sampledTexture || slot.type == BindingType::storageTexture)
            {
                return bindTextureView(sources.resources, write, slot);
            }
            if (slot.type == BindingType::accelerationStructure)
            {
                return bindAccelerationStructure(sources.acceleration, write, slot);
            }
            return bindSampler(sources.resources, write, slot);
        }

        [[nodiscard]] Status rejectSwapchainViewUpdate(const std::vector<BindingLayoutEntry>& entries,
                                                       const BindingWrite& write,
                                                       const VulkanBindingManager::BindingSlot& slot)
        {
            const auto* entry = findEntry(entries, write.binding);
            if (slot.swapchainView && entry != nullptr && hasAny(entry->flags, BindingFlag::updateAfterBind))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "updateBindingSet cannot write a view of a swapchain image to an update-after-bind entry; "
                    "write it in createBindingSet instead.",
                });
            }
            return {};
        }

        enum class WriteSource : std::uint8_t
        {
            creation,
            update,
        };

        [[nodiscard]] Status applyWrites(const BindingResourceSources& sources,
                                         const std::vector<BindingLayoutEntry>& entries,
                                         std::uint32_t variableDescriptorCount,
                                         std::span<const BindingWrite> writes,
                                         WriteSource source,
                                         std::vector<VulkanBindingManager::BindingSlot>& slots)
        {
            std::vector<bool> written(slots.size(), false);
            for (const auto& write : writes)
            {
                const auto index = findSlotIndex(entries, variableDescriptorCount, write);
                if (!index.has_value())
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A binding write references an unknown binding or out-of-range array element.",
                    });
                }
                if (written[*index])
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A binding update cannot write the same descriptor element more than once.",
                    });
                }
                auto& slot = slots[*index];
                auto bound = std::holds_alternative<std::monostate>(write.resource)
                                 ? clearPartiallyBoundSlot(entries, write.binding, slot)
                                 : bindResource(sources, write, slot);
                if (!bound)
                {
                    return bound;
                }
                if (source == WriteSource::update)
                {
                    if (auto allowed = rejectSwapchainViewUpdate(entries, write, slot); !allowed)
                    {
                        return allowed;
                    }
                }
                written[*index] = true;
            }
            return {};
        }

        [[nodiscard]] Status validateRequiredSlotsBound(const std::vector<BindingLayoutEntry>& entries,
                                                        const std::vector<VulkanBindingManager::BindingSlot>& slots,
                                                        std::uint32_t variableDescriptorCount)
        {
            std::size_t currentSlot = 0;
            for (const auto& entry : entries)
            {
                const auto count = logicalDescriptorCount(entry, variableDescriptorCount);
                for (std::uint32_t element = 0; element < count; ++element, ++currentSlot)
                {
                    if (!hasAny(entry.flags, BindingFlag::partiallyBound) &&
                        std::holds_alternative<std::monostate>(slots[currentSlot].resource))
                    {
                        return std::unexpected(Error{
                            ErrorCode::invalidArgument,
                            0,
                            "Binding set creation requires every non-partially-bound descriptor element exactly once.",
                        });
                    }
                }
            }
            return {};
        }

        [[nodiscard]] std::vector<VulkanDynamicBufferBinding> collectDynamicBuffers(
            const std::vector<VulkanBindingManager::BindingSlot>& slots)
        {
            std::vector<VulkanDynamicBufferBinding> dynamicBuffers;
            for (const auto& slot : slots)
            {
                if (!isDynamicBuffer(slot.type))
                {
                    continue;
                }
                const auto* binding = std::get_if<BufferBinding>(&slot.resource);
                dynamicBuffers.push_back(VulkanDynamicBufferBinding{
                    .type = slot.type,
                    .offset = binding != nullptr ? binding->offset : 0,
                    .range = slot.range,
                    .bufferSize = slot.bufferSize,
                });
            }
            return dynamicBuffers;
        }

        [[nodiscard]] std::shared_ptr<const std::vector<VulkanAccelerationStructureUse>>
        collectAccelerationStructureUses(const std::vector<VulkanBindingManager::BindingSlot>& slots)
        {
            std::vector<VulkanAccelerationStructureUse> uses;
            for (const auto& slot : slots)
            {
                if (slot.type == BindingType::accelerationStructure && slot.accelerationStructureUse.version != nullptr)
                {
                    uses.push_back(slot.accelerationStructureUse);
                }
            }
            if (uses.empty())
            {
                return nullptr;
            }
            return std::make_shared<const std::vector<VulkanAccelerationStructureUse>>(std::move(uses));
        }

        [[nodiscard]] std::shared_ptr<const VulkanBindingManager::SwapchainViewSlots> collectSwapchainViews(
            const std::vector<VulkanBindingManager::BindingSlot>& slots)
        {
            VulkanBindingManager::SwapchainViewSlots views;
            for (std::size_t index = 0; index < slots.size(); ++index)
            {
                const auto& slot = slots[index];
                if (!slot.swapchainView || slot.keepAlive == nullptr)
                {
                    continue;
                }
                views.slots.push_back(index);
                views.leases.push_back(static_cast<const SwapchainViewBinding*>(slot.keepAlive.get())->lease);
            }
            if (views.slots.empty())
            {
                return nullptr;
            }
            return std::make_shared<const VulkanBindingManager::SwapchainViewSlots>(std::move(views));
        }

        [[nodiscard]] std::shared_ptr<const VulkanBindingManager::SwapchainViewSlots> swapchainViewsAfterInPlaceWrites(
            const std::shared_ptr<const VulkanBindingManager::SwapchainViewSlots>& current,
            std::span<const VulkanBindingManager::StagedSlot> staged)
        {
            if (current == nullptr || std::ranges::none_of(staged, [&](const VulkanBindingManager::StagedSlot& write) {
                    return std::ranges::binary_search(current->slots, write.index);
                }))
            {
                return current;
            }
            VulkanBindingManager::SwapchainViewSlots kept;
            for (std::size_t index = 0; index < current->slots.size(); ++index)
            {
                const auto slot = current->slots[index];
                if (std::ranges::binary_search(staged, slot, {}, &VulkanBindingManager::StagedSlot::index))
                {
                    continue;
                }
                kept.slots.push_back(slot);
                kept.leases.push_back(current->leases[index]);
            }
            if (kept.slots.empty())
            {
                return nullptr;
            }
            return std::make_shared<const VulkanBindingManager::SwapchainViewSlots>(std::move(kept));
        }

        [[nodiscard]] std::vector<VulkanBindingManager::BindingSlot> makeSlots(
            const std::vector<BindingLayoutEntry>& entries, std::uint32_t variableDescriptorCount)
        {
            std::vector<VulkanBindingManager::BindingSlot> slots;
            for (const auto& entry : entries)
            {
                const auto count = logicalDescriptorCount(entry, variableDescriptorCount);
                for (std::uint32_t index = 0; index < count; ++index)
                {
                    slots.push_back(VulkanBindingManager::BindingSlot{.type = entry.type});
                }
            }
            return slots;
        }

        class DescriptorWriteBatch final
        {
        public:
            explicit DescriptorWriteBatch(std::size_t capacity)
                : buffers_(capacity), images_(capacity), accelerationWrites_(capacity)
            {
                writes_.reserve(capacity);
            }

            void append(const BindingLayoutEntry& entry,
                        std::uint32_t element,
                        const VulkanBindingManager::BindingSlot& slot,
                        std::size_t storageIndex,
                        VkDescriptorSet set)
            {
                VkWriteDescriptorSet write{};
                write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write.dstSet = set;
                write.dstBinding = entry.binding;
                write.dstArrayElement = element;
                write.descriptorCount = 1;
                write.descriptorType = nativeDescriptorType(entry.type);

                if (isBufferType(entry.type))
                {
                    buffers_[storageIndex] = VkDescriptorBufferInfo{
                        .buffer = slot.buffer,
                        .offset = std::get<BufferBinding>(slot.resource).offset,
                        .range = slot.range,
                    };
                    write.pBufferInfo = &buffers_[storageIndex];
                }
                else if (entry.type == BindingType::sampledTexture || entry.type == BindingType::storageTexture)
                {
                    images_[storageIndex] = VkDescriptorImageInfo{
                        .sampler = {},
                        .imageView = slot.imageView,
                        .imageLayout = shaderImageDescriptorLayout,
                    };
                    write.pImageInfo = &images_[storageIndex];
                }
                else if (entry.type == BindingType::accelerationStructure)
                {
                    accelerationWrites_[storageIndex] = VkWriteDescriptorSetAccelerationStructureKHR{
                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                        .pNext = {},
                        .accelerationStructureCount = 1,
                        .pAccelerationStructures = &slot.accelerationStructure,
                    };
                    write.pNext = &accelerationWrites_[storageIndex];
                }
                else
                {
                    images_[storageIndex] = VkDescriptorImageInfo{
                        .sampler = slot.sampler,
                        .imageView = {},
                        .imageLayout = {},
                    };
                    write.pImageInfo = &images_[storageIndex];
                }
                writes_.push_back(write);
            }

            void submit(const VolkDeviceTable& table, VkDevice device) const
            {
                if (!writes_.empty())
                {
                    table.vkUpdateDescriptorSets(
                        device, static_cast<std::uint32_t>(writes_.size()), writes_.data(), 0, nullptr);
                }
            }

        private:
            std::vector<VkDescriptorBufferInfo> buffers_;
            std::vector<VkDescriptorImageInfo> images_;
            std::vector<VkWriteDescriptorSetAccelerationStructureKHR> accelerationWrites_;
            std::vector<VkWriteDescriptorSet> writes_;
        };

        [[nodiscard]] Status writeDescriptorSet(const VolkDeviceTable& table,
                                                VkDevice device,
                                                const std::vector<BindingLayoutEntry>& entries,
                                                const std::vector<VulkanBindingManager::BindingSlot>& slots,
                                                std::uint32_t variableDescriptorCount,
                                                VkDescriptorSet set)
        {
            DescriptorWriteBatch batch(slots.size());
            std::size_t slotIndex = 0;
            for (const auto& entry : entries)
            {
                const auto count = logicalDescriptorCount(entry, variableDescriptorCount);
                for (std::uint32_t element = 0; element < count; ++element, ++slotIndex)
                {
                    const auto& slot = slots[slotIndex];
                    if (std::holds_alternative<std::monostate>(slot.resource))
                    {
                        continue;
                    }
                    batch.append(entry, element, slot, slotIndex, set);
                }
            }
            batch.submit(table, device);
            return {};
        }

        void writeStagedDescriptors(const VolkDeviceTable& table,
                                    VkDevice device,
                                    const std::vector<BindingLayoutEntry>& entries,
                                    std::span<const VulkanBindingManager::StagedSlot> staged,
                                    std::span<const BindingWrite> writes,
                                    VkDescriptorSet set)
        {
            DescriptorWriteBatch batch(staged.size());
            for (std::size_t index = 0; index < staged.size(); ++index)
            {
                const auto& write = writes[staged[index].write];
                const auto* entry = findEntry(entries, write.binding);
                if (entry == nullptr || std::holds_alternative<std::monostate>(staged[index].slot.resource))
                {
                    continue;
                }
                batch.append(*entry, write.arrayElement, staged[index].slot, index, set);
            }
            batch.submit(table, device);
        }

        [[nodiscard]] bool writesOnlyUpdateAfterBindEntries(const std::vector<BindingLayoutEntry>& entries,
                                                            std::span<const BindingWrite> writes) noexcept
        {
            return std::ranges::all_of(writes, [&](const BindingWrite& write) {
                const auto* entry = findEntry(entries, write.binding);
                return entry != nullptr && hasAny(entry->flags, BindingFlag::updateAfterBind);
            });
        }

        [[nodiscard]] Status validateVariableDescriptorCount(bool layoutHasVariableBinding,
                                                             std::uint32_t variableDescriptorCount,
                                                             std::uint32_t variableDescriptorCapacity)
        {
            if (!layoutHasVariableBinding && variableDescriptorCount != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A binding set supplied a variable descriptor count for a fixed-size layout.",
                });
            }
            if (layoutHasVariableBinding && variableDescriptorCount > variableDescriptorCapacity)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The variable descriptor count exceeds the device limit.",
                });
            }
            return {};
        }

        struct DescriptorTableCounts final
        {
            std::uint32_t resource = 0;
            std::uint32_t sampler = 0;
        };

        [[nodiscard]] DescriptorTableCounts countLogicalDescriptorsByTable(
            const std::vector<BindingLayoutEntry>& entries, std::uint32_t variableDescriptorCount)
        {
            DescriptorTableCounts counts{};
            for (const auto& entry : entries)
            {
                const auto count = logicalDescriptorCount(entry, variableDescriptorCount);
                if (entry.type == BindingType::sampler)
                {
                    counts.sampler += count;
                }
                else
                {
                    counts.resource += count;
                }
            }
            return counts;
        }

        [[nodiscard]] DescriptorTableCounts countWrittenDescriptorsByTable(
            const std::vector<BindingLayoutEntry>& entries, std::span<const BindingWrite> writes)
        {
            DescriptorTableCounts counts{};
            for (const auto& write : writes)
            {
                const auto* entry = findEntry(entries, write.binding);
                if (entry != nullptr && entry->type == BindingType::sampler)
                {
                    ++counts.sampler;
                }
                else
                {
                    ++counts.resource;
                }
            }
            return counts;
        }

        void recordPublishedDescriptors(BindingUpdateStatistics& statistics,
                                        const DescriptorTableCounts& prepared,
                                        const DescriptorTableCounts& published) noexcept
        {
            statistics.resourceDescriptorsPrepared += prepared.resource;
            statistics.samplerDescriptorsPrepared += prepared.sampler;
            if (published.resource != 0)
            {
                ++statistics.resourceTablePublishCount;
                statistics.resourceDescriptorsPublished += published.resource;
            }
            if (published.sampler != 0)
            {
                ++statistics.samplerTablePublishCount;
                statistics.samplerDescriptorsPublished += published.sampler;
            }
        }

    } // namespace

    struct VulkanBindingManager::LayoutAndPool final
    {
        std::shared_ptr<BindingLayoutRecord> layout;
        std::shared_ptr<BindingPoolState> pool;
    };

    Result<VulkanBindingManager::LayoutAndPool> VulkanBindingManager::findLayoutAndPool(BindingLayoutHandle layout,
                                                                                        BindingPoolHandle pool) const
    {
        const std::shared_lock lock(mutex_);
        const auto* foundLayout = layouts_.find(layout);
        const auto* foundPool = pools_.find(pool);
        if (foundLayout == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The binding layout handle is invalid or stale.",
            });
        }
        if (foundPool == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The binding pool handle is invalid or stale.",
            });
        }
        return LayoutAndPool{
            .layout = *foundLayout,
            .pool = *foundPool,
        };
    }

    VulkanBindingManager::BindingSetVersion::~BindingSetVersion()
    {
        auto following = std::move(next);
        while (following != nullptr && following.use_count() == 1)
        {
            auto after = std::move(following->next);
            following = std::move(after);
        }
    }

    std::shared_ptr<VulkanBindingManager::BindingSetVersion> VulkanBindingManager::makeUnallocatedVersion(
        std::shared_ptr<BindingLayoutRecord> layout,
        std::shared_ptr<BindingPoolState> pool,
        std::shared_ptr<BindingPoolSetLease> poolSet,
        std::vector<BindingSlot> slots,
        std::uint32_t variableDescriptorCount)
    {
        auto versionLease = std::make_shared<BindingPoolVersionLease>();
        versionLease->state = pool;
        versionLease->counts = descriptorCounts(layout->entries, variableDescriptorCount);

        auto version = std::make_shared<BindingSetVersion>();
        version->layout = std::move(layout);
        version->pool = std::move(pool);
        version->poolSet = std::move(poolSet);
        version->versionLease = std::move(versionLease);
        version->slots = std::make_shared<std::vector<BindingSlot>>(std::move(slots));
        version->dynamicBuffers = collectDynamicBuffers(*version->slots);
        version->accelerationStructureUses = collectAccelerationStructureUses(*version->slots);
        version->swapchainViews = collectSwapchainViews(*version->slots);
        version->variableDescriptorCount = variableDescriptorCount;
        return version;
    }

    Status VulkanBindingManager::allocateNativeSetLocked(BindingSetVersion& version) const
    {
        auto nativePool = nativePoolLocked(*version.pool, version.layout->updateAfterBind);
        if (!nativePool)
        {
            return std::unexpected(std::move(nativePool.error()));
        }
        VkDescriptorSetVariableDescriptorCountAllocateInfo variableAllocate{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO,
            .pNext = {},
            .descriptorSetCount = 1,
            .pDescriptorCounts = &version.variableDescriptorCount,
        };
        const VkDescriptorSetAllocateInfo allocateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = version.layout->variableBinding.has_value() ? &variableAllocate : nullptr,
            .descriptorPool = *nativePool,
            .descriptorSetCount = 1,
            .pSetLayouts = &version.layout->keepAlive->setLayout,
        };
        auto& lease = *version.versionLease;
        const auto allocated = table_->vkAllocateDescriptorSets(device_, &allocateInfo, &lease.descriptorSet);
        if (allocated != VK_SUCCESS)
        {
            lease.descriptorSet = VK_NULL_HANDLE;
            return std::unexpected(
                makeVulkanError(vulkanResultErrorCode(allocated), allocated, "vkAllocateDescriptorSets"));
        }
        lease.nativePool = *nativePool;
        return {};
    }

    Status VulkanBindingManager::allocateVersionLocked(BindingSetVersion& version) const
    {
        auto& pool = *version.pool;
        const auto& counts = version.versionLease->counts;
        constexpr auto update = detail::BindingPoolVersionUpdate::additionalVersion;
        if (auto fits = detail::checkBindingPoolVersionCapacity(
                pool.descriptorCapacity, pool.liveDescriptors, update, detail::BindingDescriptorCounts{}, counts);
            !fits)
        {
            return fits;
        }
        if (auto allocated = allocateNativeSetLocked(version); !allocated)
        {
            return allocated;
        }
        detail::applyBindingPoolVersionUpdate(pool.liveDescriptors, update, detail::BindingDescriptorCounts{}, counts);
        version.versionLease->armed = true;
        return {};
    }

    Status VulkanBindingManager::allocateFirstVersion(BindingSetVersion& version) const
    {
        auto& pool = *version.pool;
        const std::scoped_lock poolLock(pool.mutex);
        if (!pool.logicalAlive || pool.pool == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The binding pool was destroyed before binding-set allocation completed.",
            });
        }
        if (pool.liveBindingSets == pool.maxBindingSets)
        {
            return std::unexpected(Error{
                ErrorCode::resourceExhausted,
                0,
                "The binding pool has no free physically live binding-set slots.",
            });
        }
        auto allocated = allocateVersionLocked(version);
        if (!allocated)
        {
            return allocated;
        }
        ++pool.liveBindingSets;
        version.poolSet->armed = true;
        return {};
    }

    Result<BindingSetHandle> VulkanBindingManager::publishFirstVersion(std::string_view label,
                                                                       std::shared_ptr<BindingLayoutRecord> layout,
                                                                       std::shared_ptr<BindingPoolSetLease> poolSet,
                                                                       std::shared_ptr<BindingSetVersion> version)
    {
        const auto prepared = countLogicalDescriptorsByTable(layout->entries, version->variableDescriptorCount);
        const std::scoped_lock lock(mutex_);
        recordPublishedDescriptors(statistics_, prepared, prepared);
        auto record = std::make_shared<BindingSetRecord>();
        record->layout = std::move(layout);
        record->poolSet = std::move(poolSet);
        record->version = std::move(version);
        record->label = std::string(label);
        return sets_.emplace(std::move(record));
    }

    Result<BindingSetHandle> VulkanBindingManager::createBindingSet(VulkanResourceAllocator& resources,
                                                                    const BindingSetDesc& desc)
    {
        auto found = findLayoutAndPool(desc.layout, desc.pool);
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        auto& [layout, pool] = *found;

        auto validCount = validateVariableDescriptorCount(
            layout->variableBinding.has_value(), desc.variableDescriptorCount, layout->variableDescriptorCapacity);
        if (!validCount)
        {
            return std::unexpected(std::move(validCount.error()));
        }

        const BindingResourceSources sources{
            .resources = resources,
            .acceleration = accelerationManager_,
            .bufferLimits = bufferLimits_,
        };
        auto slots = makeSlots(layout->entries, desc.variableDescriptorCount);
        auto applied = applyWrites(
            sources, layout->entries, desc.variableDescriptorCount, desc.writes, WriteSource::creation, slots);
        if (!applied)
        {
            return std::unexpected(std::move(applied.error()));
        }
        auto complete = validateRequiredSlotsBound(layout->entries, slots, desc.variableDescriptorCount);
        if (!complete)
        {
            return std::unexpected(std::move(complete.error()));
        }

        if (table_ == nullptr || table_->vkAllocateDescriptorSets == nullptr || layout->keepAlive == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The Vulkan binding manager is not initialized.",
            });
        }

        auto setLease = std::make_shared<BindingPoolSetLease>();
        setLease->state = pool;
        auto version = makeUnallocatedVersion(layout, pool, setLease, std::move(slots), desc.variableDescriptorCount);
        auto allocated = allocateFirstVersion(*version);
        if (!allocated)
        {
            return std::unexpected(std::move(allocated.error()));
        }
        auto written = writeDescriptorSet(*table_,
                                          device_,
                                          layout->entries,
                                          *version->slots,
                                          version->variableDescriptorCount,
                                          version->versionLease->descriptorSet);
        if (!written)
        {
            return std::unexpected(std::move(written.error()));
        }
        if (!std::string(desc.label).empty())
        {
            setVulkanObjectName(instanceTable_,
                                device_,
                                VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                reinterpret_cast<std::uint64_t>(version->versionLease->descriptorSet),
                                std::string(desc.label).c_str());
        }
        return publishFirstVersion(desc.label, std::move(layout), std::move(setLease), std::move(version));
    }

    Result<std::shared_ptr<VulkanBindingManager::BindingSetRecord>> VulkanBindingManager::findBindingSetRecord(
        BindingSetHandle set) const
    {
        const std::shared_lock lock(mutex_);
        const auto* found = sets_.find(set);
        if (found == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The binding set handle is invalid or stale.",
            });
        }
        return *found;
    }

    Status VulkanBindingManager::allocateReplacementVersion(const BindingSetRecord& current,
                                                            BindingSetVersion& version) const
    {
        auto& pool = *version.pool;
        const std::scoped_lock poolLock(pool.mutex);
        if (!pool.logicalAlive || pool.pool == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The binding pool was destroyed before descriptor-version allocation completed.",
            });
        }
        const auto& currentLease = current.version->versionLease;
        const bool currentArmed = currentLease != nullptr && currentLease->armed;
        const auto retentions = current.poolSet != nullptr
                                    ? current.poolSet->activeCommandRetentions.load(std::memory_order_acquire)
                                    : std::uint64_t{0};
        if (auto fits = detail::checkBindingPoolVersionCapacity(
                pool.descriptorCapacity,
                pool.liveDescriptors,
                detail::bindingPoolVersionUpdateFor(retentions, currentArmed),
                currentArmed ? currentLease->counts : detail::BindingDescriptorCounts{},
                version.versionLease->counts);
            !fits)
        {
            return fits;
        }
        return allocateNativeSetLocked(version);
    }

    Status VulkanBindingManager::publishReplacementVersion(BindingSetHandle set,
                                                           const std::shared_ptr<BindingSetRecord>& current,
                                                           std::shared_ptr<BindingSetVersion> version,
                                                           std::span<const BindingWrite> writes)
    {
        const auto prepared = countWrittenDescriptorsByTable(current->layout->entries, writes);
        const auto published =
            countLogicalDescriptorsByTable(current->layout->entries, version->variableDescriptorCount);

        auto updated = std::make_shared<BindingSetRecord>();
        updated->layout = current->layout;
        updated->poolSet = current->poolSet;
        updated->version = std::move(version);
        updated->label = current->label;

        std::shared_ptr<BindingSetRecord> retired;
        {
            const std::scoped_lock lock(mutex_);
            auto* stored = sets_.find(set);
            if (stored == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The binding set was destroyed while an update version was being prepared.",
                });
            }
            if (*stored != current)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A concurrent binding set update published a newer version.",
                });
            }
            {
                auto& pool = *updated->version->pool;
                const std::scoped_lock poolLock(pool.mutex);
                if (!pool.logicalAlive)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "The binding pool was destroyed before descriptor-version allocation completed.",
                    });
                }
                const auto& currentLease = current->version->versionLease;
                auto& nextLease = *updated->version->versionLease;
                const bool currentArmed = currentLease != nullptr && currentLease->armed;
                auto committed = detail::commitBindingPoolVersionCapacity(
                    pool.descriptorCapacity,
                    pool.liveDescriptors,
                    current->poolSet != nullptr
                        ? current->poolSet->activeCommandRetentions.load(std::memory_order_acquire)
                        : std::uint64_t{0},
                    currentArmed,
                    currentArmed ? currentLease->counts : detail::BindingDescriptorCounts{},
                    nextLease.counts);
                if (!committed)
                {
                    return std::unexpected(std::move(committed.error()));
                }
                if (*committed == detail::BindingPoolVersionUpdate::reuseCurrentVersion)
                {
                    currentLease->armed = false;
                }
                nextLease.armed = true;
            }
            recordPublishedDescriptors(statistics_, prepared, published);
            retired = std::exchange(*stored, std::move(updated));
        }
        return {};
    }

    std::vector<VulkanBindingManager::BindingSlot> VulkanBindingManager::snapshotSlots(
        const BindingSetRecord& record) const
    {
        const std::shared_lock lock(mutex_);
        return *record.version->slots;
    }

    Result<std::vector<VulkanBindingManager::StagedSlot>> VulkanBindingManager::stageSlots(
        VulkanResourceAllocator& resources, const BindingSetRecord& record, std::span<const BindingWrite> writes) const
    {
        const auto& entries = record.layout->entries;
        const auto variableDescriptorCount = record.version->variableDescriptorCount;
        std::vector<StagedSlot> staged;
        staged.reserve(writes.size());
        for (std::size_t index = 0; index < writes.size(); ++index)
        {
            const auto slotIndex = findSlotIndex(entries, variableDescriptorCount, writes[index]);
            if (!slotIndex.has_value())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A binding write references an unknown binding or out-of-range array element.",
                });
            }
            staged.push_back(StagedSlot{.index = *slotIndex, .write = index});
        }
        std::ranges::sort(staged, {}, &StagedSlot::index);
        if (std::ranges::adjacent_find(staged, {}, &StagedSlot::index) != staged.end())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A binding update cannot write the same descriptor element more than once.",
            });
        }
        {
            const std::shared_lock lock(mutex_);
            const auto& table = *record.version->slots;
            for (auto& slot : staged)
            {
                slot.slot = table[slot.index];
            }
        }
        const BindingResourceSources sources{
            .resources = resources,
            .acceleration = accelerationManager_,
            .bufferLimits = bufferLimits_,
        };
        for (auto& slot : staged)
        {
            const auto& write = writes[slot.write];
            auto bound = std::holds_alternative<std::monostate>(write.resource)
                             ? clearPartiallyBoundSlot(entries, write.binding, slot.slot)
                             : bindResource(sources, write, slot.slot);
            if (!bound)
            {
                return std::unexpected(std::move(bound.error()));
            }
            if (auto allowed = rejectSwapchainViewUpdate(entries, write, slot.slot); !allowed)
            {
                return std::unexpected(std::move(allowed.error()));
            }
        }
        return staged;
    }

    Status VulkanBindingManager::publishInPlaceUpdate(BindingSetHandle set,
                                                      const std::shared_ptr<BindingSetRecord>& current,
                                                      std::vector<StagedSlot> staged,
                                                      std::span<const BindingWrite> writes)
    {
        auto& currentVersion = *current->version;
        if (currentVersion.versionLease == nullptr || currentVersion.versionLease->descriptorSet == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The binding set has no native descriptor set to update in place.",
            });
        }
        auto version = std::make_shared<BindingSetVersion>();
        version->layout = currentVersion.layout;
        version->pool = currentVersion.pool;
        version->poolSet = currentVersion.poolSet;
        version->versionLease = currentVersion.versionLease;
        version->slots = currentVersion.slots;
        version->dynamicBuffers = currentVersion.dynamicBuffers;
        version->accelerationStructureUses = currentVersion.accelerationStructureUses;
        version->swapchainViews = swapchainViewsAfterInPlaceWrites(currentVersion.swapchainViews, staged);
        version->variableDescriptorCount = currentVersion.variableDescriptorCount;

        std::vector<BindingSlot> superseded;
        superseded.reserve(staged.size());

        auto updated = std::make_shared<BindingSetRecord>();
        updated->layout = current->layout;
        updated->poolSet = current->poolSet;
        updated->version = version;
        updated->label = current->label;

        const auto written = countWrittenDescriptorsByTable(current->layout->entries, writes);
        std::shared_ptr<BindingSetRecord> retired;
        {
            const std::scoped_lock lock(mutex_);
            auto* stored = sets_.find(set);
            if (stored == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The binding set was destroyed while an update was being prepared.",
                });
            }
            if (*stored != current)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A concurrent binding set update published a newer version.",
                });
            }
            {
                const std::scoped_lock poolLock(currentVersion.pool->mutex);
                if (!currentVersion.pool->logicalAlive)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "The binding pool was destroyed before the binding set update completed.",
                    });
                }
            }
            writeStagedDescriptors(
                *table_, device_, current->layout->entries, staged, writes, version->versionLease->descriptorSet);
            auto& table = *version->slots;
            for (auto& slot : staged)
            {
                superseded.push_back(std::move(table[slot.index]));
                table[slot.index] = std::move(slot.slot);
            }
            currentVersion.superseded = std::move(superseded);
            currentVersion.next = std::move(version);
            recordPublishedDescriptors(statistics_, written, written);
            retired = std::exchange(*stored, std::move(updated));
        }
        return {};
    }

    Status VulkanBindingManager::updateBindingSet(VulkanResourceAllocator& resources,
                                                  BindingSetHandle set,
                                                  std::span<const BindingWrite> writes)
    {
        if (writes.empty())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A binding set update requires at least one descriptor write.",
            });
        }
        auto found = findBindingSetRecord(set);
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        const auto& current = *found;
        if (table_ == nullptr || table_->vkAllocateDescriptorSets == nullptr || current->layout == nullptr ||
            current->layout->keepAlive == nullptr || current->version == nullptr || current->version->pool == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The Vulkan binding manager is not initialized.",
            });
        }
        if (current->layout->updateAfterBind && writesOnlyUpdateAfterBindEntries(current->layout->entries, writes))
        {
            auto staged = stageSlots(resources, *current, writes);
            if (!staged)
            {
                return std::unexpected(std::move(staged.error()));
            }
            return publishInPlaceUpdate(set, current, std::move(*staged), writes);
        }

        const BindingResourceSources sources{
            .resources = resources,
            .acceleration = accelerationManager_,
            .bufferLimits = bufferLimits_,
        };
        auto slots = snapshotSlots(*current);
        auto applied = applyWrites(sources,
                                   current->layout->entries,
                                   current->version->variableDescriptorCount,
                                   writes,
                                   WriteSource::update,
                                   slots);
        if (!applied)
        {
            return applied;
        }

        auto version = makeUnallocatedVersion(current->layout,
                                              current->version->pool,
                                              current->poolSet,
                                              std::move(slots),
                                              current->version->variableDescriptorCount);
        auto allocated = allocateReplacementVersion(*current, *version);
        if (!allocated)
        {
            return allocated;
        }
        auto written = writeDescriptorSet(*table_,
                                          device_,
                                          current->layout->entries,
                                          *version->slots,
                                          version->variableDescriptorCount,
                                          version->versionLease->descriptorSet);
        if (!written)
        {
            return written;
        }
        return publishReplacementVersion(set, current, std::move(version), writes);
    }

    Status VulkanBindingManager::destroyBindingSet(BindingSetHandle set)
    {
        std::shared_ptr<BindingSetRecord> removed;
        {
            const std::scoped_lock lock(mutex_);
            auto* found = sets_.find(set);
            if (found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The binding set handle is invalid or stale.",
                });
            }
            removed = *found;
            static_cast<void>(sets_.erase(set));
        }
        return {};
    }

    Result<std::vector<std::shared_ptr<VulkanBindingManager::BindingLayoutRecord>>> VulkanBindingManager::
        findPipelineSetLayouts(std::span<const BindingLayoutHandle> bindingLayouts) const
    {
        std::vector<std::shared_ptr<BindingLayoutRecord>> sets;
        sets.reserve(bindingLayouts.size());
        const std::shared_lock lock(mutex_);
        for (const auto handle : bindingLayouts)
        {
            const auto* found = layouts_.find(handle);
            if (found == nullptr || *found == nullptr || (*found)->keepAlive == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A pipeline layout references an invalid or stale binding layout.",
                });
            }
            sets.push_back(*found);
        }
        return sets;
    }

    namespace
    {

        [[nodiscard]] std::vector<VkPushConstantRange> nativePushConstantRanges(
            std::span<const PushConstantRange> ranges, ShaderStage enabledStages)
        {
            std::vector<VkPushConstantRange> nativeRanges;
            nativeRanges.reserve(ranges.size());
            for (const auto& range : ranges)
            {
                nativeRanges.push_back(VkPushConstantRange{
                    .stageFlags = nativeShaderStageFlags(range.visibility & enabledStages),
                    .offset = range.offset,
                    .size = range.size,
                });
            }
            return nativeRanges;
        }

    } // namespace

    Result<std::shared_ptr<VulkanBindingManager::PipelineLayoutKeepAlive>> VulkanBindingManager::
        createNativePipelineLayout(const PipelineLayoutRecord& record) const
    {
        auto keepAlive = std::make_shared<PipelineLayoutKeepAlive>();
        keepAlive->device = device_;
        keepAlive->table = table_;
        std::vector<VkDescriptorSetLayout> setLayouts;
        setLayouts.reserve(record.sets.size());
        for (const auto& set : record.sets)
        {
            keepAlive->setLayouts.push_back(set->keepAlive);
            setLayouts.push_back(set->keepAlive->setLayout);
        }
        const auto nativeRanges = nativePushConstantRanges(record.nativePushConstantRanges, enabledShaderStages_);
        const VkPipelineLayoutCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = {},
            .flags = {},
            .setLayoutCount = static_cast<std::uint32_t>(setLayouts.size()),
            .pSetLayouts = setLayouts.empty() ? nullptr : setLayouts.data(),
            .pushConstantRangeCount = static_cast<std::uint32_t>(nativeRanges.size()),
            .pPushConstantRanges = nativeRanges.empty() ? nullptr : nativeRanges.data(),
        };
        const auto created = table_->vkCreatePipelineLayout(device_, &createInfo, nullptr, &keepAlive->pipelineLayout);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(created), created, "vkCreatePipelineLayout"));
        }
        if (!record.label.empty())
        {
            setVulkanObjectName(instanceTable_,
                                device_,
                                VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                reinterpret_cast<std::uint64_t>(keepAlive->pipelineLayout),
                                record.label.c_str());
        }
        return keepAlive;
    }

    Result<PipelineLayoutHandle> VulkanBindingManager::createPipelineLayout(const PipelineLayoutDesc& desc)
    {
        if (desc.bindingLayouts.size() > maxBindingSets)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "A pipeline layout exceeds the portable four-set binding limit.",
            });
        }
        auto pushConstantSize = validatePushConstantRanges(desc.pushConstantRanges, enabledShaderStages_);
        if (!pushConstantSize)
        {
            return std::unexpected(std::move(pushConstantSize.error()));
        }
        if (table_ == nullptr || table_->vkCreatePipelineLayout == nullptr || device_ == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The Vulkan binding manager is not initialized.",
            });
        }

        auto sets = findPipelineSetLayouts(desc.bindingLayouts);
        if (!sets)
        {
            return std::unexpected(std::move(sets.error()));
        }
        DescriptorTotals totals{};
        DescriptorTotals withoutUpdateAfterBind{};
        bool anyUpdateAfterBind = false;
        for (const auto& set : *sets)
        {
            anyUpdateAfterBind = anyUpdateAfterBind || set->updateAfterBind;
            for (const auto& entry : set->entries)
            {
                const auto count = hasAny(entry.flags, BindingFlag::variableDescriptorCount)
                                       ? set->variableDescriptorCapacity
                                       : entry.count;
                addDescriptors(totals, entry, count, enabledShaderStages_);
                if (!set->updateAfterBind)
                {
                    addDescriptors(withoutUpdateAfterBind, entry, count, enabledShaderStages_);
                }
            }
        }
        const bool withinLimits = anyUpdateAfterBind
                                      ? withinDescriptorLimits(withoutUpdateAfterBind, descriptorLimits_) &&
                                            withinDescriptorLimits(totals, updateAfterBindLimits_)
                                      : withinDescriptorLimits(totals, descriptorLimits_);
        if (!withinLimits)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The pipeline layout exceeds the device's per-stage or per-pipeline-layout descriptor limits.",
            });
        }

        auto record = std::make_shared<PipelineLayoutRecord>();
        record->label = std::string(desc.label);
        record->pushConstantRanges.assign(desc.pushConstantRanges.begin(), desc.pushConstantRanges.end());
        std::vector<PushConstantRange> enabledPushConstantRanges(desc.pushConstantRanges.begin(),
                                                                 desc.pushConstantRanges.end());
        for (auto& range : enabledPushConstantRanges)
        {
            range.visibility = range.visibility & enabledShaderStages_;
        }
        record->nativePushConstantRanges =
            validation_detail::mergePushConstantRangesPerStage(enabledPushConstantRanges);
        record->pushConstantBytes = *pushConstantSize;
        record->drawIndex = desc.drawIndex;
        record->sets = std::move(*sets);
        auto keepAlive = createNativePipelineLayout(*record);
        if (!keepAlive)
        {
            return std::unexpected(std::move(keepAlive.error()));
        }
        record->keepAlive = std::move(*keepAlive);
        const std::scoped_lock lock(mutex_);
        return pipelineLayouts_.emplace(std::move(record));
    }

    Status VulkanBindingManager::destroyPipelineLayout(PipelineLayoutHandle pipelineLayout)
    {
        std::shared_ptr<PipelineLayoutRecord> removed;
        {
            const std::scoped_lock lock(mutex_);
            auto* found = pipelineLayouts_.find(pipelineLayout);
            if (found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The pipeline layout handle is invalid or stale.",
                });
            }
            removed = *found;
            static_cast<void>(pipelineLayouts_.erase(pipelineLayout));
        }
        return {};
    }

    Result<VulkanNativeBindingLayout> VulkanBindingManager::nativeBindingLayout(BindingLayoutHandle layout) const
    {
        const std::shared_lock lock(mutex_);
        const auto* found = layouts_.find(layout);
        if (found == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The binding layout handle is invalid or stale.",
            });
        }
        return VulkanNativeBindingLayout{
            .owner = *found,
            .setLayout = (*found)->keepAlive != nullptr ? (*found)->keepAlive->setLayout : VK_NULL_HANDLE,
            .entries = (*found)->entries,
            .dynamicBufferCount = (*found)->dynamicBufferCount,
        };
    }

    Result<VulkanNativeBindingSet> VulkanBindingManager::nativeBindingSet(BindingSetHandle set) const
    {
        const std::shared_lock lock(mutex_);
        const auto* found = sets_.find(set);
        if (found == nullptr || *found == nullptr || (*found)->version == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The binding set handle is invalid or stale.",
            });
        }
        return makeNativeBindingSet(**found);
    }

    Result<VulkanNativeBindingSet> VulkanBindingManager::retainBindingSetForCommand(BindingSetHandle set,
                                                                                    bool retainResources) const
    {
        const std::shared_lock lock(mutex_);
        const auto* found = sets_.find(set);
        if (found == nullptr || *found == nullptr || (*found)->version == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The binding set handle is invalid or stale.",
            });
        }
        const auto& record = **found;
        auto native = makeNativeBindingSet(record);
        auto counter =
            record.poolSet != nullptr
                ? std::shared_ptr<std::atomic<std::uint64_t>>(record.poolSet, &record.poolSet->activeCommandRetentions)
                : std::shared_ptr<std::atomic<std::uint64_t>>{};
        auto owner = retainResources ? std::shared_ptr<void>(record.version)
                                     : std::shared_ptr<void>(record.version->versionLease);
        native.commandUse = VulkanCommandUse(std::move(counter), std::move(owner));
        return native;
    }

    VulkanNativeBindingSet VulkanBindingManager::makeNativeBindingSet(const BindingSetRecord& record)
    {
        const auto& version = record.version;
        return VulkanNativeBindingSet{
            .owner = version,
            .descriptorSet = version->versionLease->descriptorSet,
            .setLayout = record.layout != nullptr && record.layout->keepAlive != nullptr
                             ? record.layout->keepAlive->setLayout
                             : VK_NULL_HANDLE,
            .dynamicBufferCount = record.layout != nullptr ? record.layout->dynamicBufferCount : 0,
            .dynamicBuffers = version->dynamicBuffers,
            .accelerationStructureUses =
                version->accelerationStructureUses != nullptr
                    ? std::span<const VulkanAccelerationStructureUse>(*version->accelerationStructureUses)
                    : std::span<const VulkanAccelerationStructureUse>{},
            .swapchainLeases =
                version->swapchainViews != nullptr
                    ? std::span<const std::shared_ptr<VulkanSwapchainImageLease>>(version->swapchainViews->leases)
                    : std::span<const std::shared_ptr<VulkanSwapchainImageLease>>{},
        };
    }

    Result<VulkanNativePipelineLayout> VulkanBindingManager::nativePipelineLayout(PipelineLayoutHandle layout) const
    {
        const std::shared_lock lock(mutex_);
        const auto* found = pipelineLayouts_.find(layout);
        if (found == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The pipeline layout handle is invalid or stale.",
            });
        }
        VulkanNativePipelineLayout native{
            .owner = *found,
            .pipelineLayout = (*found)->keepAlive != nullptr ? (*found)->keepAlive->pipelineLayout : VK_NULL_HANDLE,
            .pushConstantBytes = (*found)->pushConstantBytes,
            .pushConstantRanges = (*found)->pushConstantRanges,
            .nativePushConstantRanges = (*found)->nativePushConstantRanges,
        };
        for (const auto& set : (*found)->sets)
        {
            native.sets.push_back(VulkanNativeBindingLayout{
                .owner = set,
                .setLayout = set->keepAlive != nullptr ? set->keepAlive->setLayout : VK_NULL_HANDLE,
                .entries = set->entries,
                .dynamicBufferCount = set->dynamicBufferCount,
            });
        }
        native.drawIndex = (*found)->drawIndex;
        return native;
    }

    BindingUpdateStatistics VulkanBindingManager::statistics() const noexcept
    {
        const std::shared_lock lock(mutex_);
        return statistics_;
    }

} // namespace rachis::vulkan
