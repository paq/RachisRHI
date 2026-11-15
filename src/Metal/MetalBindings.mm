#include "Metal/MetalInternal.h"

#include "Common/BindingPoolAccounting.h"

namespace rachis::metal
{

    MetalBindingPoolSetLease::~MetalBindingPoolSetLease()
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

    MetalBindingPoolVersionLease::~MetalBindingPoolVersionLease()
    {
        if (!armed || state == nullptr)
        {
            return;
        }
        const std::scoped_lock lock(state->mutex);
        for (std::size_t index = 0; index < counts.size(); ++index)
        {
            if (counts[index] <= state->liveDescriptors[index])
            {
                state->liveDescriptors[index] -= counts[index];
            }
            else
            {
                state->liveDescriptors[index] = 0;
            }
        }
    }

    MetalBindingSetCommandRetention::~MetalBindingSetCommandRetention()
    {
        if (poolSet == nullptr || poolSet->state == nullptr)
        {
            return;
        }
        const std::scoped_lock lock(poolSet->state->mutex);
        if (poolSet->activeCommandRetentions > 0)
        {
            --poolSet->activeCommandRetentions;
        }
    }

    std::shared_ptr<MetalBindingSetCommandRetention> metalRetainBindingSetForCommand(
        const MetalBindingSetRecord& record)
    {
        auto retention = std::make_shared<MetalBindingSetCommandRetention>();
        if (record.version != nullptr)
        {
            retention->versionLease = record.version->versionLease;
        }
        if (record.poolSet != nullptr && record.poolSet->state != nullptr)
        {
            const std::scoped_lock lock(record.poolSet->state->mutex);
            ++record.poolSet->activeCommandRetentions;
            retention->poolSet = record.poolSet;
        }
        return retention;
    }

    Status retainMetalBindingSetRetentions(MetalCommandBuffer& command,
                                           std::span<std::shared_ptr<MetalBindingSetCommandRetention>> retentions)
    {
        try
        {
            for (auto& retention : retentions)
            {
                command.retained().push_back(std::move(retention));
            }
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::outOfMemory, "Failed to retain a Metal binding set for a command."));
        }
        return {};
    }

    namespace
    {

        constexpr std::uint64_t metalUniformAlignment = 16;
        constexpr std::uint64_t metalRawStorageAlignment = 16;
        constexpr std::uint64_t metalMaxUniformBindingBytes = 65536;
        constexpr std::uint32_t metalMaxStructuredStride = 2048;
        constexpr std::uint32_t metalResourceHeapCapacity = 1'000'000;
        constexpr std::uint32_t metalSamplerHeapCapacity = 2048;

        [[nodiscard]] const BindingLayoutEntry* metalFindEntry(const std::vector<BindingLayoutEntry>& entries,
                                                               std::uint32_t binding) noexcept
        {
            const auto found = std::ranges::find_if(
                entries, [&](const BindingLayoutEntry& entry) { return entry.binding == binding; });
            return found == entries.end() ? nullptr : &*found;
        }

        [[nodiscard]] std::array<std::uint32_t, bindingTypeCount> metalDescriptorCounts(
            const std::vector<BindingLayoutEntry>& entries, std::uint32_t variableDescriptorCount)
        {
            std::array<std::uint32_t, bindingTypeCount> counts{};
            for (const auto& entry : entries)
            {
                counts[static_cast<std::size_t>(entry.type)] +=
                    metalLogicalDescriptorCount(entry, variableDescriptorCount);
            }
            return counts;
        }

        [[nodiscard]] Result<std::uint64_t> metalResolvedBindingSize(const BufferDesc& desc,
                                                                     const BufferBinding& binding)
        {
            if (binding.offset >= desc.size)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "The buffer binding offset is out of bounds."));
            }
            const auto size = binding.size == wholeResource ? desc.size - binding.offset : binding.size;
            if (size == 0 || size > desc.size - binding.offset)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "The buffer binding range is empty or out of bounds."));
            }
            return size;
        }

        [[nodiscard]] Status metalValidateBufferBinding(const BufferDesc& desc,
                                                        BindingType type,
                                                        const BufferBinding& binding,
                                                        std::uint64_t size)
        {
            type = metalStaticBufferType(type);
            if (type == BindingType::uniformBuffer)
            {
                if (!hasAny(desc.usage, BufferUsage::constant) || binding.offset % metalUniformAlignment != 0 ||
                    size > metalMaxUniformBindingBytes)
                {
                    return std::unexpected(
                        metalSelectionError(ErrorCode::invalidArgument,
                                            "Constant-buffer bindings require constant usage, a 16-byte aligned "
                                            "offset, and a non-empty range no larger than 64 KiB."));
                }
                return {};
            }
            if (!hasAny(desc.usage, BufferUsage::storage))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "Storage-buffer bindings require storage usage."));
            }
            const auto stride = binding.structureByteStride == 0 ? 4U : binding.structureByteStride;
            if ((binding.structureByteStride != 0 &&
                 (binding.structureByteStride % 4 != 0 || binding.structureByteStride > metalMaxStructuredStride)) ||
                (binding.structureByteStride == 0 && binding.offset % metalRawStorageAlignment != 0) ||
                binding.offset % stride != 0 || size % stride != 0 ||
                size / stride > std::numeric_limits<std::uint32_t>::max())
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "Storage-buffer binding ranges must satisfy D3D12 "
                                                           "raw/structured alignment and element-count limits."));
            }
            return {};
        }

        [[nodiscard]] std::vector<MetalBindingSlot> metalMakeSlots(const std::vector<BindingLayoutEntry>& entries,
                                                                   std::uint32_t variableDescriptorCount)
        {
            std::vector<MetalBindingSlot> slots;
            for (const auto& entry : entries)
            {
                const auto count = metalLogicalDescriptorCount(entry, variableDescriptorCount);
                for (std::uint32_t index = 0; index < count; ++index)
                {
                    slots.push_back(MetalBindingSlot{
                        .type = entry.type,
                        .binding = entry.binding,
                        .arrayElement = index,
                        .metalIndex = entry.binding + index,
                    });
                }
            }
            return slots;
        }

        [[nodiscard]] std::optional<std::size_t> metalSlotIndex(const std::vector<BindingLayoutEntry>& entries,
                                                                std::uint32_t variableDescriptorCount,
                                                                std::uint32_t binding,
                                                                std::uint32_t arrayElement)
        {
            std::size_t index = 0;
            for (const auto& entry : entries)
            {
                const auto count = metalLogicalDescriptorCount(entry, variableDescriptorCount);
                if (entry.binding == binding)
                {
                    if (arrayElement >= count)
                    {
                        return std::nullopt;
                    }
                    return index + arrayElement;
                }
                index += count;
            }
            return std::nullopt;
        }

        [[nodiscard]] Status metalClearSlot(const std::vector<BindingLayoutEntry>& entries,
                                            const BindingWrite& write,
                                            MetalBindingSlot& slot)
        {
            const auto* entry = metalFindEntry(entries, write.binding);
            if (entry == nullptr || !hasAny(entry->flags, BindingFlag::partiallyBound))
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "Only a partially-bound layout entry may be cleared."));
            }
            slot.resource = std::monostate{};
            slot.keepAlive.reset();
            slot.buffer = nil;
            slot.texture = nil;
            slot.sampler = nil;
            slot.gpuAddress = 0;
            slot.offset = 0;
            slot.range = 0;
            return {};
        }

        [[nodiscard]] Status metalWriteBufferSlot(MetalDeviceNative& native,
                                                  const BindingWrite& write,
                                                  MetalBindingSlot& slot)
        {
            const auto* binding = std::get_if<BufferBinding>(&write.resource);
            if (binding == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "The binding write resource type does not match its buffer layout entry."));
            }
            auto* record = native.buffers.find(binding->buffer);
            if (record == nullptr || record->buffer == nil || record->owner == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "A buffer binding requires a live Metal buffer."));
            }
            auto size = metalResolvedBindingSize(record->desc, *binding);
            if (!size)
            {
                return std::unexpected(std::move(size.error()));
            }
            auto valid = metalValidateBufferBinding(record->desc, slot.type, *binding, *size);
            if (!valid)
            {
                return valid;
            }
            auto retained = *binding;
            retained.size = *size;
            slot.resource = retained;
            slot.keepAlive = record->owner;
            slot.buffer = record->buffer;
            slot.texture = nil;
            slot.sampler = nil;
            slot.gpuAddress = record->buffer.gpuAddress;
            slot.offset = binding->offset;
            slot.range = *size;
            return {};
        }

        [[nodiscard]] Status metalWriteTextureSlot(MetalDeviceNative& native,
                                                   const BindingWrite& write,
                                                   MetalBindingSlot& slot)
        {
            const auto* handle = std::get_if<TextureViewHandle>(&write.resource);
            if (handle == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "The binding write resource type does not match its texture layout entry."));
            }
            auto* view = native.views.find(*handle);
            if (view == nullptr || view->view == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "A texture binding requires a live Metal view."));
            }
            const auto expected =
                slot.type == BindingType::sampledTexture ? TextureViewUsage::sampled : TextureViewUsage::storage;
            if (view->desc.usage != expected)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "The texture view usage does not match the binding layout entry."));
            }
            slot.resource = *handle;
            slot.keepAlive = view->owner != nullptr ? std::static_pointer_cast<void>(view->owner) : view->parent;
            slot.buffer = nil;
            slot.texture = view->view;
            slot.sampler = nil;
            slot.gpuAddress = 0;
            slot.offset = 0;
            slot.range = 0;
            return {};
        }

        [[nodiscard]] Status metalWriteSamplerSlot(MetalDeviceNative& native,
                                                   const BindingWrite& write,
                                                   MetalBindingSlot& slot)
        {
            const auto* handle = std::get_if<SamplerHandle>(&write.resource);
            if (handle == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "The binding write resource type does not match its sampler layout entry."));
            }
            auto* sampler = native.samplers.find(*handle);
            if (sampler == nullptr || sampler->sampler == nil)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "A sampler binding requires a live Metal sampler."));
            }
            slot.resource = *handle;
            slot.keepAlive = sampler->owner;
            slot.buffer = nil;
            slot.texture = nil;
            slot.sampler = sampler->sampler;
            slot.gpuAddress = 0;
            slot.offset = 0;
            slot.range = 0;
            return {};
        }

        [[nodiscard]] Status metalWriteSlot(MetalDeviceNative& native,
                                            const std::vector<BindingLayoutEntry>& entries,
                                            const BindingWrite& write,
                                            MetalBindingSlot& slot)
        {
            if (std::holds_alternative<std::monostate>(write.resource))
            {
                return metalClearSlot(entries, write, slot);
            }
            if (metalIsBufferType(slot.type))
            {
                return metalWriteBufferSlot(native, write, slot);
            }
            if (slot.type == BindingType::sampledTexture || slot.type == BindingType::storageTexture)
            {
                return metalWriteTextureSlot(native, write, slot);
            }
            if (slot.type == BindingType::sampler)
            {
                return metalWriteSamplerSlot(native, write, slot);
            }
            return std::unexpected(
                detail::unsupportedBackendOperation("Metal", "acceleration-structure and sampler-feedback bindings"));
        }

        [[nodiscard]] Status metalRequireCompleteSlots(const std::vector<BindingLayoutEntry>& entries,
                                                       const std::vector<MetalBindingSlot>& slots,
                                                       std::uint32_t variableDescriptorCount)
        {
            std::size_t currentSlot = 0;
            for (const auto& entry : entries)
            {
                const auto count = metalLogicalDescriptorCount(entry, variableDescriptorCount);
                for (std::uint32_t element = 0; element < count; ++element, ++currentSlot)
                {
                    if (!hasAny(entry.flags, BindingFlag::partiallyBound) &&
                        std::holds_alternative<std::monostate>(slots[currentSlot].resource))
                    {
                        return std::unexpected(
                            metalSelectionError(ErrorCode::invalidArgument,
                                                "Binding set creation requires every non-partially-bound descriptor "
                                                "element exactly once."));
                    }
                }
            }
            return {};
        }

        [[nodiscard]] Status metalApplyWrites(MetalDeviceNative& native,
                                              const std::vector<BindingLayoutEntry>& entries,
                                              std::span<const BindingWrite> writes,
                                              std::vector<MetalBindingSlot>& slots,
                                              std::uint32_t variableDescriptorCount,
                                              bool requireComplete)
        {
            std::vector<bool> written(slots.size(), false);
            for (const auto& write : writes)
            {
                const auto index = metalSlotIndex(entries, variableDescriptorCount, write.binding, write.arrayElement);
                if (!index.has_value())
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument,
                        "A binding write references an unknown binding or out-of-range array element."));
                }
                if (written[*index])
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument,
                        "A binding update cannot write the same descriptor element more than once."));
                }
                auto applied = metalWriteSlot(native, entries, write, slots[*index]);
                if (!applied)
                {
                    return applied;
                }
                written[*index] = true;
            }

            if (requireComplete)
            {
                return metalRequireCompleteSlots(entries, slots, variableDescriptorCount);
            }
            return {};
        }

        [[nodiscard]] Status metalPublishArgumentTable(id<MTLDevice> device,
                                                       MetalBindingSetVersion& version,
                                                       std::uint32_t minimumBufferCount = 0)
        {
            std::uint32_t maxBuffer = minimumBufferCount;
            std::uint32_t maxTexture = 0;
            std::uint32_t maxSampler = 0;
            bool anyBuffer = minimumBufferCount != 0;
            bool anyTexture = false;
            bool anySampler = false;
            for (const auto& slot : version.slots)
            {
                if (slot.metalIndex >= metalArgumentLimit(slot.type))
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument, "The resource index exceeds Metal's argument-table capacity."));
                }
                const auto index = slot.metalIndex + 1;
                if (metalIsBufferType(slot.type))
                {
                    anyBuffer = true;
                    maxBuffer = (std::max)(maxBuffer, index);
                }
                else if (slot.type == BindingType::sampledTexture || slot.type == BindingType::storageTexture)
                {
                    anyTexture = true;
                    maxTexture = (std::max)(maxTexture, index);
                }
                else if (slot.type == BindingType::sampler)
                {
                    anySampler = true;
                    maxSampler = (std::max)(maxSampler, index);
                }
            }

            if (!anyBuffer && !anyTexture && !anySampler)
            {
                return {};
            }

            MTL4ArgumentTableDescriptor* const tableDesc = [MTL4ArgumentTableDescriptor new];
            tableDesc.initializeBindings = YES;
            tableDesc.maxBufferBindCount = anyBuffer ? maxBuffer : 0;
            tableDesc.maxTextureBindCount = anyTexture ? maxTexture : 0;
            tableDesc.maxSamplerStateBindCount = anySampler ? maxSampler : 0;
            NSError* tableError = nil;
            version.table = [device newArgumentTableWithDescriptor:tableDesc error:&tableError];
            if (version.table == nil)
            {
                return std::unexpected(
                    metalNsError(ErrorCode::nativeFailure, tableError, "newArgumentTableWithDescriptor"));
            }

            for (const auto& slot : version.slots)
            {
                if (std::holds_alternative<std::monostate>(slot.resource))
                {
                    continue;
                }
                if (metalIsBufferType(slot.type) && slot.buffer != nil)
                {
                    [version.table setAddress:(slot.gpuAddress + slot.offset) atIndex:slot.metalIndex];
                }
                else if ((slot.type == BindingType::sampledTexture || slot.type == BindingType::storageTexture) &&
                         slot.texture != nil)
                {
                    [version.table setTexture:slot.texture.gpuResourceID atIndex:slot.metalIndex];
                }
                else if (slot.type == BindingType::sampler && slot.sampler != nil)
                {
                    [version.table setSamplerState:slot.sampler.gpuResourceID atIndex:slot.metalIndex];
                }
            }
            return {};
        }

        [[nodiscard]] Status metalValidateLayoutEntry(const BindingLayoutEntry& entry)
        {
            const auto flags = static_cast<std::uint8_t>(entry.flags);
            const auto allowedFlags = static_cast<std::uint8_t>(BindingFlag::partiallyBound) |
                                      static_cast<std::uint8_t>(BindingFlag::variableDescriptorCount);
            const bool variable = hasAny(entry.flags, BindingFlag::variableDescriptorCount);
            const bool unbounded = entry.count == unboundedBindingCount;
            if (entry.count == 0 ||
                (!unbounded && entry.binding > std::numeric_limits<std::uint32_t>::max() - (entry.count - 1)) ||
                !metalIsValidVisibility(entry.visibility) || (flags & ~allowedFlags) != 0)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Binding layout entries require a non-empty register range, valid visibility, and valid flags."));
            }
            const auto limit = metalArgumentLimit(entry.type);
            if (entry.binding >= limit || (!variable && entry.count > limit - entry.binding))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "The binding layout exceeds Metal's argument-table capacity."));
            }
            if (variable != unbounded)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "An unbounded binding count and the variable-descriptor-count flag must be specified together."));
            }
            if (entry.type == BindingType::accelerationStructure || metalIsSamplerFeedback(entry.type))
            {
                return std::unexpected(detail::unsupportedBackendOperation(
                    "Metal", "acceleration-structure and sampler-feedback bindings"));
            }
            if (metalIsDynamicBuffer(entry.type) && entry.flags != BindingFlag::none)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "Dynamic buffer bindings cannot be partially bound or variable length."));
            }
            return {};
        }

        [[nodiscard]] Status metalAssignVariableBinding(MetalBindingLayoutRecord& record)
        {
            for (const auto& entry : record.entries)
            {
                if (!hasAny(entry.flags, BindingFlag::variableDescriptorCount))
                {
                    continue;
                }
                if (record.variableBinding.has_value() || entry.binding != record.entries.back().binding)
                {
                    return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                               "A binding layout may have one variable-count entry and "
                                                               "it must use the largest binding number."));
                }
                record.variableBinding = entry.binding;
            }
            return {};
        }

        [[nodiscard]] Status metalReserveHeapDescriptors(MetalBindingLayoutRecord& record,
                                                         const BindingLayoutEntry& entry)
        {
            if (entry.type == BindingType::sampler)
            {
                if (entry.count > metalSamplerHeapCapacity - record.samplerDescriptorCount)
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument,
                        "A binding layout exceeds one shader-visible sampler heap's portable capacity."));
                }
                record.samplerDescriptorCount += entry.count;
            }
            else
            {
                if (entry.count > metalResourceHeapCapacity - record.resourceDescriptorCount)
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument,
                        "A binding layout exceeds one shader-visible resource heap's portable capacity."));
                }
                record.resourceDescriptorCount += entry.count;
            }
            if (metalIsDynamicBuffer(entry.type))
            {
                if (entry.count > maxDynamicBufferOffsetsPerBindingSet - record.dynamicBufferCount)
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument, "A binding layout exceeds the portable dynamic-offset limit."));
                }
                record.dynamicBufferCount += entry.count;
            }
            return {};
        }

        [[nodiscard]] Status metalValidateLayoutRanges(MetalBindingLayoutRecord& record)
        {
            std::array<std::uint64_t, 4> rangeEnds{};
            std::array<bool, 4> hasRange{};
            for (const auto& entry : record.entries)
            {
                if (std::ranges::count(record.entries, entry.binding, &BindingLayoutEntry::binding) != 1)
                {
                    return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                               "Binding layout binding numbers must be unique."));
                }
                const auto classIndex = metalRangeClassIndex(entry.type);
                if (hasRange[classIndex] && entry.binding < rangeEnds[classIndex])
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument,
                        "Binding arrays cannot overlap another range in the same Metal argument table namespace."));
                }
                rangeEnds[classIndex] = static_cast<std::uint64_t>(entry.binding) + entry.count;
                hasRange[classIndex] = true;
                if (hasAny(entry.flags, BindingFlag::variableDescriptorCount))
                {
                    continue;
                }
                if (auto reserved = metalReserveHeapDescriptors(record, entry); !reserved)
                {
                    return reserved;
                }
            }
            return {};
        }

        [[nodiscard]] Result<std::shared_ptr<MetalBindingLayoutRecord>> metalBuildLayout(const BindingLayoutDesc& desc)
        {
            auto record = std::make_shared<MetalBindingLayoutRecord>();
            record->label = std::string(desc.label);
            record->entries.reserve(desc.entries.size());
            for (const auto& entry : desc.entries)
            {
                if (auto valid = metalValidateLayoutEntry(entry); !valid)
                {
                    return std::unexpected(std::move(valid.error()));
                }
                record->entries.push_back(entry);
            }

            std::ranges::sort(record->entries, {}, &BindingLayoutEntry::binding);
            if (auto assigned = metalAssignVariableBinding(*record); !assigned)
            {
                return std::unexpected(std::move(assigned.error()));
            }
            if (auto ranges = metalValidateLayoutRanges(*record); !ranges)
            {
                return std::unexpected(std::move(ranges.error()));
            }
            return record;
        }

        void metalPublishStatistics(MetalDeviceNative& native,
                                    std::uint32_t resourcePrepared,
                                    std::uint32_t samplerPrepared,
                                    std::uint32_t resourcePublished,
                                    std::uint32_t samplerPublished)
        {
            native.bindingStatistics.resourceDescriptorsPrepared += resourcePrepared;
            native.bindingStatistics.samplerDescriptorsPrepared += samplerPrepared;
            if (resourcePublished != 0)
            {
                ++native.bindingStatistics.resourceTablePublishCount;
                native.bindingStatistics.resourceDescriptorsPublished += resourcePublished;
            }
            if (samplerPublished != 0)
            {
                ++native.bindingStatistics.samplerTablePublishCount;
                native.bindingStatistics.samplerDescriptorsPublished += samplerPublished;
            }
        }

        [[nodiscard]] Result<std::shared_ptr<MetalBindingPoolSetLease>> metalAcquireSetLease(
            const std::shared_ptr<MetalBindingPoolState>& pool)
        {
            auto setLease = std::make_shared<MetalBindingPoolSetLease>();
            setLease->state = pool;
            const std::scoped_lock poolLock(pool->mutex);
            if (!pool->logicalAlive)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState,
                                        "The binding pool was destroyed before binding-set allocation completed."));
            }
            if (pool->liveBindingSets == pool->maxBindingSets)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::resourceExhausted, "The binding pool has no free physically live binding-set slots."));
            }
            ++pool->liveBindingSets;
            setLease->armed = true;
            return setLease;
        }

        [[nodiscard]] Result<std::shared_ptr<MetalBindingPoolVersionLease>> metalAcquireVersionLease(
            const std::shared_ptr<MetalBindingPoolState>& pool,
            const std::array<std::uint32_t, bindingTypeCount>& counts)
        {
            auto versionLease = std::make_shared<MetalBindingPoolVersionLease>();
            versionLease->state = pool;
            versionLease->counts = counts;
            const std::scoped_lock poolLock(pool->mutex);
            if (!pool->logicalAlive)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState,
                    "The binding pool was destroyed before descriptor-version allocation completed."));
            }
            for (std::size_t index = 0; index < counts.size(); ++index)
            {
                if (counts[index] > pool->descriptorCapacity[index] - pool->liveDescriptors[index])
                {
                    return std::unexpected(metalSelectionError(ErrorCode::resourceExhausted,
                                                               "The binding pool has insufficient per-type capacity "
                                                               "for another immutable descriptor version."));
                }
            }
            for (std::size_t index = 0; index < counts.size(); ++index)
            {
                pool->liveDescriptors[index] += counts[index];
            }
            versionLease->armed = true;
            return versionLease;
        }

        [[nodiscard]] Result<std::shared_ptr<MetalBindingPoolVersionLease>> metalPrepareReplacementVersionLease(
            const std::shared_ptr<MetalBindingPoolState>& pool,
            const std::shared_ptr<MetalBindingPoolSetLease>& poolSet,
            const std::shared_ptr<MetalBindingPoolVersionLease>& currentLease,
            const std::array<std::uint32_t, bindingTypeCount>& counts)
        {
            auto versionLease = std::make_shared<MetalBindingPoolVersionLease>();
            versionLease->state = pool;
            versionLease->counts = counts;
            const std::scoped_lock poolLock(pool->mutex);
            if (!pool->logicalAlive)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState,
                    "The binding pool was destroyed before descriptor-version allocation completed."));
            }
            const bool currentArmed = currentLease != nullptr && currentLease->armed;
            const auto retentions = poolSet != nullptr ? poolSet->activeCommandRetentions : std::uint64_t{0};
            if (auto fits = detail::checkBindingPoolVersionCapacity(
                    pool->descriptorCapacity,
                    pool->liveDescriptors,
                    detail::bindingPoolVersionUpdateFor(retentions, currentArmed),
                    currentArmed ? currentLease->counts : detail::BindingDescriptorCounts{},
                    counts);
                !fits)
            {
                return std::unexpected(std::move(fits.error()));
            }
            return versionLease;
        }

        [[nodiscard]] Status metalCommitReplacementVersionLease(
            const std::shared_ptr<MetalBindingPoolSetLease>& poolSet,
            const std::shared_ptr<MetalBindingPoolVersionLease>& currentLease,
            MetalBindingPoolVersionLease& nextLease)
        {
            auto& pool = *nextLease.state;
            const std::scoped_lock poolLock(pool.mutex);
            if (!pool.logicalAlive)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState,
                    "The binding pool was destroyed before descriptor-version allocation completed."));
            }
            const bool currentArmed = currentLease != nullptr && currentLease->armed;
            auto committed = detail::commitBindingPoolVersionCapacity(
                pool.descriptorCapacity,
                pool.liveDescriptors,
                poolSet != nullptr ? poolSet->activeCommandRetentions : std::uint64_t{0},
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
            return {};
        }

    } // namespace

    [[nodiscard]] Result<std::uint32_t> metalValidatePushConstantRanges(std::span<const PushConstantRange> ranges)
    {
        std::uint32_t byteSize = 0;
        for (std::size_t index = 0; index < ranges.size(); ++index)
        {
            const auto& range = ranges[index];
            if (!metalIsValidVisibility(range.visibility) || range.size == 0 ||
                range.offset % sizeof(std::uint32_t) != 0 || range.size % sizeof(std::uint32_t) != 0 ||
                range.offset > maxPushConstantBytes || range.size > maxPushConstantBytes - range.offset)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "Push-constant ranges require valid visibility and a "
                                                           "non-empty, 4-byte aligned range within 128 bytes."));
            }
            for (std::size_t previous = 0; previous < index; ++previous)
            {
                if (metalPushRangesOverlap(range, ranges[previous]) &&
                    hasAny(range.visibility, ranges[previous].visibility))
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument,
                        "Overlapping push-constant ranges cannot be visible to the same shader stage."));
                }
            }
            byteSize = (std::max)(byteSize, range.offset + range.size);
        }
        return byteSize;
    }

    Status metalApplyDynamicOffsets(std::vector<std::shared_ptr<MetalBindingSetVersion>>& versions,
                                    std::span<const std::uint32_t> offsets)
    {
        std::size_t cursor = 0;
        for (auto& version : versions)
        {
            if (version->layout->dynamicBufferCount == 0)
            {
                continue;
            }
            const std::size_t setStartCursor = cursor;
            bool anyNonZero = false;
            for (const auto& slot : version->slots)
            {
                if (!metalIsDynamicBuffer(slot.type))
                {
                    continue;
                }
                const auto extra = offsets[cursor++];
                const auto alignment = metalStaticBufferType(slot.type) == BindingType::uniformBuffer
                                           ? metalUniformAlignment
                                           : metalRawStorageAlignment;
                if (slot.buffer == nil || extra % alignment != 0)
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument, "Dynamic offsets require a live buffer and portable alignment."));
                }
                const auto length = static_cast<std::uint64_t>(slot.buffer.length);
                if (slot.range > length || slot.offset > length - slot.range ||
                    extra > length - (slot.offset + slot.range))
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument, "A dynamic offset is outside the captured Metal buffer range."));
                }
                anyNonZero = anyNonZero || (extra != 0);
            }
            if (!anyNonZero)
            {
                continue;
            }
            auto adjusted = std::make_shared<MetalBindingSetVersion>(*version);
            adjusted->table = nil;
            std::size_t applyCursor = setStartCursor;
            for (auto& slot : adjusted->slots)
            {
                if (!metalIsDynamicBuffer(slot.type))
                {
                    continue;
                }
                slot.offset += offsets[applyCursor++];
            }
            version = std::move(adjusted);
        }
        return {};
    }

    Result<BindingLayoutHandle> MetalDevice::createBindingLayout(const BindingLayoutDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        auto record = metalBuildLayout(desc);
        if (!record)
        {
            return std::unexpected(std::move(record.error()));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal createBindingLayout requires an active device."));
        }
        return native_->bindingLayouts.emplace(std::move(*record));
    }

    Status MetalDevice::destroyBindingLayout(BindingLayoutHandle layout)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || !native_->bindingLayouts.erase(layout))
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The binding layout handle is invalid or stale."));
        }
        return {};
    }

    Result<BindingPoolHandle> MetalDevice::createBindingPool(const BindingPoolDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (desc.maxBindingSets == 0)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "A binding pool requires a non-zero binding-set capacity."));
        }
        if (desc.sizes.size() > bindingTypeCount)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "A binding pool cannot contain more than one size for each binding type."));
        }
        auto state = std::make_shared<MetalBindingPoolState>();
        state->maxBindingSets = desc.maxBindingSets;
        state->label = std::string(desc.label);
        std::array<bool, bindingTypeCount> seen{};
        std::uint64_t resourceDescriptors = 0;
        std::uint64_t samplerDescriptors = 0;
        for (const auto& size : desc.sizes)
        {
            const auto index = static_cast<std::size_t>(size.type);
            if (index >= bindingTypeCount || size.descriptorCount == 0 || seen[index])
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Binding-pool sizes require unique valid binding types and non-zero descriptor counts."));
            }
            seen[index] = true;
            state->descriptorCapacity[index] = size.descriptorCount;
            if (size.type == BindingType::sampler)
            {
                samplerDescriptors += size.descriptorCount;
            }
            else
            {
                resourceDescriptors += size.descriptorCount;
            }
        }
        if (resourceDescriptors > metalResourceHeapCapacity || samplerDescriptors > metalSamplerHeapCapacity)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument,
                                    "A binding pool exceeds one shader-visible descriptor heap's portable capacity."));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal createBindingPool requires an active device."));
        }
        return native_->bindingPools.emplace(std::move(state));
    }

    Status MetalDevice::destroyBindingPool(BindingPoolHandle pool)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal destroyBindingPool requires an active device."));
        }
        auto* found = native_->bindingPools.find(pool);
        if (found == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The binding pool handle is invalid or stale."));
        }
        {
            const std::scoped_lock poolLock((*found)->mutex);
            (*found)->logicalAlive = false;
        }
        native_->bindingPools.erase(pool);
        return {};
    }

    namespace
    {

        struct MetalBindingSetSources final
        {
            std::shared_ptr<MetalBindingLayoutRecord> layout;
            std::shared_ptr<MetalBindingPoolState> pool;
        };

        struct MetalBindingSetLeases final
        {
            std::shared_ptr<MetalBindingPoolSetLease> set;
            std::shared_ptr<MetalBindingPoolVersionLease> version;
        };

        struct MetalDescriptorTally final
        {
            std::uint32_t resource = 0;
            std::uint32_t sampler = 0;
        };

        [[nodiscard]] Result<MetalBindingSetSources> metalFindBindingSetSources(MetalDeviceNative& native,
                                                                                const BindingSetDesc& desc)
        {
            auto* foundLayout = native.bindingLayouts.find(desc.layout);
            auto* foundPool = native.bindingPools.find(desc.pool);
            if (foundLayout == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "The binding layout handle is invalid or stale."));
            }
            if (foundPool == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "The binding pool handle is invalid or stale."));
            }
            return MetalBindingSetSources{
                .layout = *foundLayout,
                .pool = *foundPool,
            };
        }

        [[nodiscard]] Status metalValidateBindingSetRanges(const MetalBindingLayoutRecord& layout,
                                                           std::uint32_t variableDescriptorCount)
        {
            if (!layout.variableBinding.has_value() && variableDescriptorCount != 0)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "A binding set supplied a variable descriptor count for a fixed-size layout."));
            }
            for (const auto& entry : layout.entries)
            {
                const auto count = metalLogicalDescriptorCount(entry, variableDescriptorCount);
                const auto limit = metalArgumentLimit(entry.type);
                if (entry.binding >= limit || count > limit - entry.binding)
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument, "The binding range exceeds Metal's argument-table capacity."));
                }
            }
            return {};
        }

        [[nodiscard]] Result<MetalBindingSetLeases> metalAcquireBindingSetLeases(
            const std::shared_ptr<MetalBindingPoolState>& pool,
            const std::vector<BindingLayoutEntry>& entries,
            std::uint32_t variableDescriptorCount)
        {
            auto setLease = metalAcquireSetLease(pool);
            if (!setLease)
            {
                return std::unexpected(std::move(setLease.error()));
            }
            const auto counts = metalDescriptorCounts(entries, variableDescriptorCount);
            auto versionLease = metalAcquireVersionLease(pool, counts);
            if (!versionLease)
            {
                return std::unexpected(std::move(versionLease.error()));
            }
            return MetalBindingSetLeases{
                .set = std::move(*setLease),
                .version = std::move(*versionLease),
            };
        }

        [[nodiscard]] std::shared_ptr<MetalBindingSetVersion> metalMakeBindingSetVersion(
            const std::shared_ptr<MetalBindingLayoutRecord>& layout,
            const std::shared_ptr<MetalBindingPoolState>& pool,
            std::shared_ptr<MetalBindingPoolVersionLease>&& versionLease,
            std::vector<MetalBindingSlot>&& slots,
            std::uint32_t variableDescriptorCount)
        {
            auto version = std::make_shared<MetalBindingSetVersion>();
            version->layout = layout;
            version->pool = pool;
            version->versionLease = std::move(versionLease);
            version->slots = std::move(slots);
            version->variableDescriptorCount = variableDescriptorCount;
            return version;
        }

        [[nodiscard]] MetalDescriptorTally metalCountLayoutDescriptors(const std::vector<BindingLayoutEntry>& entries,
                                                                       std::uint32_t variableDescriptorCount)
        {
            MetalDescriptorTally tally;
            for (const auto& entry : entries)
            {
                const auto count = metalLogicalDescriptorCount(entry, variableDescriptorCount);
                if (entry.type == BindingType::sampler)
                {
                    tally.sampler += count;
                }
                else
                {
                    tally.resource += count;
                }
            }
            return tally;
        }

    } // namespace

    Result<BindingSetHandle> MetalDevice::createBindingSet(const BindingSetDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::shared_ptr<MetalBindingLayoutRecord> layout;
        std::shared_ptr<MetalBindingPoolState> pool;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal createBindingSet requires an active device."));
            }
            auto sources = metalFindBindingSetSources(*native_, desc);
            if (!sources)
            {
                return std::unexpected(std::move(sources.error()));
            }
            layout = std::move(sources->layout);
            pool = std::move(sources->pool);
        }

        if (auto ranges = metalValidateBindingSetRanges(*layout, desc.variableDescriptorCount); !ranges)
        {
            return std::unexpected(std::move(ranges.error()));
        }

        auto slots = metalMakeSlots(layout->entries, desc.variableDescriptorCount);
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal createBindingSet requires an active device."));
            }
            auto applied =
                metalApplyWrites(*native_, layout->entries, desc.writes, slots, desc.variableDescriptorCount, true);
            if (!applied)
            {
                return std::unexpected(std::move(applied.error()));
            }
        }

        auto leases = metalAcquireBindingSetLeases(pool, layout->entries, desc.variableDescriptorCount);
        if (!leases)
        {
            return std::unexpected(std::move(leases.error()));
        }
        auto version = metalMakeBindingSetVersion(
            layout, pool, std::move(leases->version), std::move(slots), desc.variableDescriptorCount);
        const auto prepared = metalCountLayoutDescriptors(layout->entries, desc.variableDescriptorCount);

        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal createBindingSet requires an active device."));
        }
        auto published = metalPublishArgumentTable(native_->device, *version);
        if (!published)
        {
            return std::unexpected(std::move(published.error()));
        }
        metalPublishStatistics(*native_, prepared.resource, prepared.sampler, prepared.resource, prepared.sampler);
        auto record = std::make_shared<MetalBindingSetRecord>();
        record->layout = std::move(layout);
        record->poolSet = std::move(leases->set);
        record->version = std::move(version);
        record->label = std::string(desc.label);
        return native_->bindingSets.emplace(std::move(record));
    }

    namespace
    {

        [[nodiscard]] MetalDescriptorTally metalCountWrittenDescriptors(const std::vector<BindingLayoutEntry>& entries,
                                                                        std::span<const BindingWrite> writes)
        {
            MetalDescriptorTally tally;
            for (const auto& write : writes)
            {
                const auto* entry = metalFindEntry(entries, write.binding);
                if (entry != nullptr && entry->type == BindingType::sampler)
                {
                    ++tally.sampler;
                }
                else
                {
                    ++tally.resource;
                }
            }
            return tally;
        }

        [[nodiscard]] Status metalRequireCurrentBindingSetVersion(
            MetalDeviceNative& native,
            BindingSetHandle set,
            const std::shared_ptr<MetalBindingSetRecord>& current,
            const std::shared_ptr<MetalBindingSetVersion>& previousVersion)
        {
            auto* stored = native.bindingSets.find(set);
            if (stored == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "The binding set was destroyed while an update version was being prepared."));
            }
            if (*stored != current || (*stored)->version != previousVersion)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState, "A concurrent binding set update published a newer version."));
            }
            return {};
        }

    } // namespace

    Status MetalDevice::updateBindingSet(BindingSetHandle set, std::span<const BindingWrite> writes)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (writes.empty())
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "A binding set update requires at least one descriptor write."));
        }
        std::shared_ptr<MetalBindingSetRecord> current;
        std::shared_ptr<MetalBindingSetVersion> previousVersion;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal updateBindingSet requires an active device."));
            }
            for (auto& timeline : native_->timelines)
            {
                collectMetalQueue(timeline);
            }
            auto* found = native_->bindingSets.find(set);
            if (found == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "The binding set handle is invalid or stale."));
            }
            current = *found;
            previousVersion = current->version;
        }

        auto slots = previousVersion->slots;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal updateBindingSet requires an active device."));
            }
            auto applied = metalApplyWrites(
                *native_, current->layout->entries, writes, slots, previousVersion->variableDescriptorCount, false);
            if (!applied)
            {
                return applied;
            }
        }

        const auto counts = metalDescriptorCounts(current->layout->entries, previousVersion->variableDescriptorCount);
        auto versionLease = metalPrepareReplacementVersionLease(
            previousVersion->pool, current->poolSet, previousVersion->versionLease, counts);
        if (!versionLease)
        {
            return std::unexpected(std::move(versionLease.error()));
        }
        auto version = metalMakeBindingSetVersion(current->layout,
                                                  previousVersion->pool,
                                                  std::move(*versionLease),
                                                  std::move(slots),
                                                  previousVersion->variableDescriptorCount);
        const auto prepared = metalCountWrittenDescriptors(current->layout->entries, writes);

        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal updateBindingSet requires an active device."));
        }
        if (auto unchanged = metalRequireCurrentBindingSetVersion(*native_, set, current, previousVersion); !unchanged)
        {
            return unchanged;
        }
        auto published = metalPublishArgumentTable(native_->device, *version);
        if (!published)
        {
            return published;
        }
        if (auto committed = metalCommitReplacementVersionLease(
                current->poolSet, previousVersion->versionLease, *version->versionLease);
            !committed)
        {
            return committed;
        }
        const auto publishedCounts =
            metalCountLayoutDescriptors(current->layout->entries, previousVersion->variableDescriptorCount);
        metalPublishStatistics(
            *native_, prepared.resource, prepared.sampler, publishedCounts.resource, publishedCounts.sampler);
        current->version = std::move(version);
        return {};
    }

    Status MetalDevice::destroyBindingSet(BindingSetHandle set)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || !native_->bindingSets.erase(set))
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The binding set handle is invalid or stale."));
        }
        return {};
    }

} // namespace rachis::metal
