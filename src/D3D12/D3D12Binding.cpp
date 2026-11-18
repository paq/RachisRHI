#include "D3D12/D3D12Binding.h"

#include "Common/HandleTable.h"
#include "D3D12/D3D12RuntimePolicies.h"
#include "Internal/PortableValidation.h"

#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace rachis::d3d12
{
    namespace
    {

        constexpr long exclusiveSamplerTableHolders = 2;
        constexpr std::size_t minimumSamplerTablePruneThreshold = 64;

        struct BindingLayoutRecordEntry final
        {
            BindingLayoutEntry entry;
            D3D12_DESCRIPTOR_RANGE_TYPE rangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
            std::uint32_t descriptorOffset = 0;
        };

        struct BindingLayoutRecord final
        {
            std::vector<BindingLayoutRecordEntry> entries;
            std::uint32_t resourceDescriptorCount = 0;
            std::uint32_t samplerDescriptorCount = 0;
            std::uint32_t dynamicBufferCount = 0;
            std::optional<std::uint32_t> variableBinding;
            std::string label;
        };

        struct BufferDescriptorSource final
        {
            BindingType type = BindingType::uniformBuffer;
            BufferBindingSource source;
            BufferBinding binding;
        };

        struct TextureDescriptorSource final
        {
            NativeTextureView source;
        };

        struct AccelerationStructureDescriptorSource final
        {
            AccelerationStructureBindingSource source;
        };

        struct SamplerFeedbackDescriptorSource final
        {
            SamplerFeedbackBindingSource source;
        };

        struct NullResourceDescriptorSource final
        {
            BindingType type = BindingType::uniformBuffer;
        };

        using ResourceDescriptorSource = std::variant<std::monostate,
                                                      NullResourceDescriptorSource,
                                                      BufferDescriptorSource,
                                                      TextureDescriptorSource,
                                                      AccelerationStructureDescriptorSource,
                                                      SamplerFeedbackDescriptorSource>;

        struct SamplerDescriptorSource final
        {
            NativeSampler source;
        };

        struct NullSamplerDescriptorSource final
        {
        };

        using SamplerDescriptorSourceVariant =
            std::variant<std::monostate, NullSamplerDescriptorSource, SamplerDescriptorSource>;

        using SamplerTableKey = std::vector<std::uintptr_t>;

        struct SharedSamplerTable final
        {
            std::shared_ptr<GpuDescriptorLease> descriptors;
            SamplerTableKey key;
            std::vector<std::shared_ptr<void>> samplerOwners;
        };

        struct BindingSetRecord final
        {
            std::shared_ptr<BindingLayoutRecord> layout;
            std::shared_ptr<BindingPoolSetLease> poolSet;
            std::shared_ptr<BindingPoolVersionLease> poolVersion;
            BindingDescriptorCounts descriptorCounts{};
            std::shared_ptr<GpuDescriptorLease> resources;
            std::shared_ptr<SharedSamplerTable> samplers;

            std::vector<std::shared_ptr<CpuDescriptorLease>> resourceStagingDescriptors;
            std::vector<std::shared_ptr<CpuDescriptorLease>> samplerStagingDescriptors;
            std::vector<ResourceDescriptorSource> resourceSources;
            std::vector<SamplerDescriptorSourceVariant> samplerSources;
            std::vector<std::shared_ptr<ResidencyRecord>> residencies;
            std::array<std::uint32_t, maxDynamicBufferOffsetsPerBindingSet> dynamicResourceIndices{};
            std::uint32_t dynamicResourceCount = 0;
            std::uint32_t variableDescriptorCount = 0;
            bool hasSpecialResourceCaptures = false;
            std::string label;
        };

        struct DescriptorPreparationMask final
        {
            std::vector<std::uint8_t> resources;
            std::vector<std::uint8_t> samplers;
        };

        struct BindingUpdateCounters final
        {
            std::atomic<std::uint64_t> resourceDescriptorsPrepared = 0;
            std::atomic<std::uint64_t> samplerDescriptorsPrepared = 0;
            std::atomic<std::uint64_t> resourceTablePublishCount = 0;
            std::atomic<std::uint64_t> samplerTablePublishCount = 0;
            std::atomic<std::uint64_t> resourceDescriptorsPublished = 0;
            std::atomic<std::uint64_t> samplerDescriptorsPublished = 0;

            [[nodiscard]] BindingUpdateStatistics snapshot() const noexcept
            {
                return {
                    .resourceDescriptorsPrepared = resourceDescriptorsPrepared.load(std::memory_order_relaxed),
                    .samplerDescriptorsPrepared = samplerDescriptorsPrepared.load(std::memory_order_relaxed),
                    .resourceTablePublishCount = resourceTablePublishCount.load(std::memory_order_relaxed),
                    .samplerTablePublishCount = samplerTablePublishCount.load(std::memory_order_relaxed),
                    .resourceDescriptorsPublished = resourceDescriptorsPublished.load(std::memory_order_relaxed),
                    .samplerDescriptorsPublished = samplerDescriptorsPublished.load(std::memory_order_relaxed),
                };
            }
        };

        [[nodiscard]] constexpr bool isDynamicBuffer(BindingType type) noexcept
        {
            return type == BindingType::dynamicUniformBuffer || type == BindingType::dynamicReadOnlyStorageBuffer ||
                   type == BindingType::dynamicStorageBuffer;
        }

        [[nodiscard]] constexpr bool isBuffer(BindingType type) noexcept
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

        [[nodiscard]] constexpr D3D12_DESCRIPTOR_RANGE_TYPE rangeType(BindingType type) noexcept
        {
            switch (type)
            {
            case BindingType::uniformBuffer:
            case BindingType::dynamicUniformBuffer:
                return D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
            case BindingType::readOnlyStorageBuffer:
            case BindingType::dynamicReadOnlyStorageBuffer:
            case BindingType::sampledTexture:
            case BindingType::accelerationStructure:
                return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            case BindingType::storageBuffer:
            case BindingType::dynamicStorageBuffer:
            case BindingType::storageTexture:
            case BindingType::minMipFeedback:
            case BindingType::mipRegionUsedFeedback:
                return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            default:
                return D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
            }
        }

        [[nodiscard]] constexpr bool isSamplerFeedback(BindingType type) noexcept
        {
            return type == BindingType::minMipFeedback || type == BindingType::mipRegionUsedFeedback;
        }

        [[nodiscard]] constexpr std::size_t rangeClassIndex(D3D12_DESCRIPTOR_RANGE_TYPE type) noexcept
        {
            switch (type)
            {
            case D3D12_DESCRIPTOR_RANGE_TYPE_CBV:
                return 0;
            case D3D12_DESCRIPTOR_RANGE_TYPE_SRV:
                return 1;
            case D3D12_DESCRIPTOR_RANGE_TYPE_UAV:
                return 2;
            default:
                return 3;
            }
        }

        [[nodiscard]] Status validateLayoutEntryShape(const BindingLayoutEntry& entry)
        {
            const auto visibility = static_cast<std::uint16_t>(entry.visibility);
            const auto allowedVisibility = static_cast<std::uint16_t>(ShaderStage::all);
            const auto flags = static_cast<std::uint8_t>(entry.flags);
            const auto allowedFlags = static_cast<std::uint8_t>(BindingFlag::partiallyBound) |
                                      static_cast<std::uint8_t>(BindingFlag::variableDescriptorCount) |
                                      static_cast<std::uint8_t>(BindingFlag::updateAfterBind);
            const bool variable = hasAny(entry.flags, BindingFlag::variableDescriptorCount);
            const bool unbounded = entry.count == unboundedBindingCount;
            const bool registerRangeOverflows =
                !unbounded && entry.binding > std::numeric_limits<std::uint32_t>::max() - (entry.count - 1);
            const bool validVisibility = visibility != 0 && (visibility & ~allowedVisibility) == 0;
            const bool validFlags = (flags & ~allowedFlags) == 0;
            if (entry.count == 0 || registerRangeOverflows || !validVisibility || !validFlags)
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
            return {};
        }

        [[nodiscard]] Status validateLayoutEntryFeatureSupport(const BindingLayoutEntry& entry,
                                                               bool variableDescriptorCountSupported,
                                                               bool rayTracingSupported,
                                                               bool samplerFeedbackSupported)
        {
            if (hasAny(entry.flags, BindingFlag::variableDescriptorCount) && !variableDescriptorCountSupported)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Variable-count binding arrays require D3D12 Resource Binding Tier 3.",
                });
            }
            if (entry.type == BindingType::accelerationStructure && !rayTracingSupported)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Acceleration-structure bindings require ray-tracing support.",
                });
            }
            if (isSamplerFeedback(entry.type) && !samplerFeedbackSupported)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Sampler-feedback bindings require sampler-feedback support.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateLayoutEntryTypeRestrictions(const BindingLayoutEntry& entry)
        {
            if (isSamplerFeedback(entry.type) && (entry.count != 1 || entry.flags != BindingFlag::none))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The initial sampler-feedback profile supports only one required map per binding.",
                });
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

        [[nodiscard]] Status validateUpdateAfterBindLayout(const BindingLayoutDesc& desc)
        {
            const bool updateAfterBind = std::ranges::any_of(desc.entries, [](const BindingLayoutEntry& entry) {
                return hasAny(entry.flags, BindingFlag::updateAfterBind);
            });
            const bool dynamicBuffers = std::ranges::any_of(
                desc.entries, [](const BindingLayoutEntry& entry) { return isDynamicBuffer(entry.type); });
            if (updateAfterBind && dynamicBuffers)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A binding layout with an update-after-bind entry cannot contain dynamic buffer bindings.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateLayoutEntry(const BindingLayoutEntry& entry,
                                                 bool variableDescriptorCountSupported,
                                                 bool rayTracingSupported,
                                                 bool samplerFeedbackSupported)
        {
            auto validShape = validateLayoutEntryShape(entry);
            if (!validShape)
            {
                return validShape;
            }
            auto supported = validateLayoutEntryFeatureSupport(
                entry, variableDescriptorCountSupported, rayTracingSupported, samplerFeedbackSupported);
            if (!supported)
            {
                return supported;
            }
            return validateLayoutEntryTypeRestrictions(entry);
        }

        [[nodiscard]] Result<std::optional<std::uint32_t>> findVariableBinding(
            const std::vector<BindingLayoutRecordEntry>& sortedEntries)
        {
            std::optional<std::uint32_t> variableBinding;
            for (const auto& entry : sortedEntries)
            {
                if (!hasAny(entry.entry.flags, BindingFlag::variableDescriptorCount))
                {
                    continue;
                }
                if (variableBinding.has_value() || entry.entry.binding != sortedEntries.back().entry.binding)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A binding layout may have one variable-count entry and it must use the largest binding "
                        "number.",
                    });
                }
                variableBinding = entry.entry.binding;
            }
            return variableBinding;
        }

        [[nodiscard]] Status validateUniqueBinding(const std::vector<BindingLayoutRecordEntry>& entries,
                                                   std::uint32_t binding)
        {
            const auto duplicate =
                std::ranges::count(entries, binding, [](const auto& candidate) { return candidate.entry.binding; });
            if (duplicate != 1)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Binding layout binding numbers must be unique.",
                });
            }
            return {};
        }

        struct ClaimedRegisterRanges final
        {
            std::array<std::uint64_t, 4> rangeEnds{};
            std::array<bool, 4> hasRange{};
        };

        [[nodiscard]] Status claimRegisterRange(const BindingLayoutRecordEntry& entry, ClaimedRegisterRanges& claimed)
        {
            const auto classIndex = rangeClassIndex(entry.rangeType);
            if (claimed.hasRange[classIndex] && entry.entry.binding < claimed.rangeEnds[classIndex])
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Binding arrays cannot overlap another HLSL register range in the same descriptor class.",
                });
            }
            claimed.rangeEnds[classIndex] = entry.entry.count == unboundedBindingCount
                                                ? std::numeric_limits<std::uint64_t>::max()
                                                : static_cast<std::uint64_t>(entry.entry.binding) + entry.entry.count;
            claimed.hasRange[classIndex] = true;
            return {};
        }

        [[nodiscard]] Status reserveSamplerDescriptors(BindingLayoutRecordEntry& entry, BindingLayoutRecord& record)
        {
            entry.descriptorOffset = record.samplerDescriptorCount;
            if (entry.entry.count == unboundedBindingCount)
            {
                if (record.samplerDescriptorCount >= bindingSamplerDescriptorBudget)
                {
                    return std::unexpected(Error{
                        ErrorCode::unsupported,
                        0,
                        "An unbounded sampler tail requires room for at least one physical descriptor.",
                    });
                }
                return {};
            }
            if (entry.entry.count > bindingSamplerDescriptorBudget - record.samplerDescriptorCount)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The binding layout exceeds the D3D12 shader-visible sampler heap limit.",
                });
            }
            record.samplerDescriptorCount += entry.entry.count;
            return {};
        }

        [[nodiscard]] Status reserveDynamicBufferOffsets(const BindingLayoutRecordEntry& entry,
                                                         BindingLayoutRecord& record)
        {
            if (!isDynamicBuffer(entry.entry.type))
            {
                return {};
            }
            if (entry.entry.count > maxDynamicBufferOffsetsPerBindingSet - record.dynamicBufferCount)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "A binding layout cannot declare more than eight dynamic buffer offsets.",
                });
            }
            record.dynamicBufferCount += entry.entry.count;
            return {};
        }

        [[nodiscard]] Status reserveResourceDescriptors(BindingLayoutRecordEntry& entry, BindingLayoutRecord& record)
        {
            auto dynamicOffsetsReserved = reserveDynamicBufferOffsets(entry, record);
            if (!dynamicOffsetsReserved)
            {
                return dynamicOffsetsReserved;
            }
            entry.descriptorOffset = record.resourceDescriptorCount;
            if (entry.entry.count == unboundedBindingCount)
            {
                if (record.resourceDescriptorCount >= bindingPoolResourceDescriptorBudget)
                {
                    return std::unexpected(Error{
                        ErrorCode::unsupported,
                        0,
                        "An unbounded resource tail requires room for at least one physical descriptor.",
                    });
                }
                return {};
            }
            if (entry.entry.count > bindingPoolResourceDescriptorBudget - record.resourceDescriptorCount)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The binding layout exceeds the RHI resource descriptor heap capacity.",
                });
            }
            record.resourceDescriptorCount += entry.entry.count;
            return {};
        }

        [[nodiscard]] Status claimRegistersAndReserveDescriptors(BindingLayoutRecord& record)
        {
            ClaimedRegisterRanges claimedRanges;
            for (auto& entry : record.entries)
            {
                auto unique = validateUniqueBinding(record.entries, entry.entry.binding);
                if (!unique)
                {
                    return unique;
                }
                auto claimed = claimRegisterRange(entry, claimedRanges);
                if (!claimed)
                {
                    return claimed;
                }
                auto reserved = entry.entry.type == BindingType::sampler ? reserveSamplerDescriptors(entry, record)
                                                                         : reserveResourceDescriptors(entry, record);
                if (!reserved)
                {
                    return reserved;
                }
            }
            return {};
        }

        [[nodiscard]] Result<std::shared_ptr<BindingLayoutRecord>> buildLayout(const BindingLayoutDesc& desc,
                                                                               bool variableDescriptorCountSupported,
                                                                               bool rayTracingSupported,
                                                                               bool samplerFeedbackSupported)
        {
            auto record = std::make_shared<BindingLayoutRecord>();
            record->label = desc.label;
            record->entries.reserve(desc.entries.size());
            for (const auto& entry : desc.entries)
            {
                auto valid = validateLayoutEntry(
                    entry, variableDescriptorCountSupported, rayTracingSupported, samplerFeedbackSupported);
                if (!valid)
                {
                    return std::unexpected(std::move(valid.error()));
                }
                record->entries.push_back(BindingLayoutRecordEntry{
                    .entry = entry,
                    .rangeType = rangeType(entry.type),
                });
            }
            auto validUpdateAfterBind = validateUpdateAfterBindLayout(desc);
            if (!validUpdateAfterBind)
            {
                return std::unexpected(std::move(validUpdateAfterBind.error()));
            }

            std::ranges::sort(record->entries, {}, [](const auto& entry) { return entry.entry.binding; });
            auto variableBinding = findVariableBinding(record->entries);
            if (!variableBinding)
            {
                return std::unexpected(std::move(variableBinding.error()));
            }
            record->variableBinding = *variableBinding;
            auto reserved = claimRegistersAndReserveDescriptors(*record);
            if (!reserved)
            {
                return std::unexpected(std::move(reserved.error()));
            }
            return record;
        }

        [[nodiscard]] const BindingLayoutRecordEntry* findEntry(const BindingLayoutRecord& layout,
                                                                std::uint32_t binding) noexcept
        {
            const auto found = std::ranges::lower_bound(
                layout.entries, binding, {}, [](const auto& entry) { return entry.entry.binding; });
            return found != layout.entries.end() && found->entry.binding == binding ? &*found : nullptr;
        }

        [[nodiscard]] Result<std::uint64_t> bindingSize(const BufferBindingSource& source, const BufferBinding& binding)
        {
            if (binding.offset >= source.desc.size)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The buffer binding offset is out of bounds.",
                });
            }
            const auto size = binding.size == wholeResource ? source.desc.size - binding.offset : binding.size;
            if (size == 0 || size > source.desc.size - binding.offset)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The buffer binding range is empty or out of bounds.",
                });
            }
            return size;
        }

        [[nodiscard]] Result<std::uint64_t> validateBufferBinding(const BufferBindingSource& source,
                                                                  BindingType type,
                                                                  const BufferBinding& binding)
        {
            auto size = bindingSize(source, binding);
            if (!size)
            {
                return std::unexpected(std::move(size.error()));
            }

            type = staticBufferType(type);
            if (type == BindingType::uniformBuffer)
            {
                constexpr std::uint64_t maxConstantBufferSize = D3D12_REQ_CONSTANT_BUFFER_ELEMENT_COUNT * 16ULL;
                if (!hasAny(source.desc.usage, BufferUsage::constant) ||
                    binding.offset % D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT != 0 ||
                    *size > maxConstantBufferSize)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Constant-buffer bindings require constant usage, a 256-byte aligned offset, and a range no "
                        "larger than 64 KiB.",
                    });
                }
                const auto viewSize = constantBufferViewSize(*size);
                if (viewSize > source.nativeWidth || binding.offset > source.nativeWidth - viewSize)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A constant-buffer view rounded up to 256 bytes must fit in the native buffer, so an imported "
                        "constant buffer needs a native width padded to a multiple of 256 bytes.",
                    });
                }
                return *size;
            }

            if (!hasAny(source.desc.usage, BufferUsage::storage))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Storage-buffer bindings require storage usage.",
                });
            }
            const auto stride = binding.structureByteStride == 0 ? 4U : binding.structureByteStride;
            if ((binding.structureByteStride != 0 &&
                 (binding.structureByteStride % 4 != 0 ||
                  binding.structureByteStride > D3D12_REQ_MULTI_ELEMENT_STRUCTURE_SIZE_IN_BYTES)) ||
                (binding.structureByteStride == 0 && binding.offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT != 0) ||
                binding.offset % stride != 0 || *size % stride != 0 ||
                *size / stride > std::numeric_limits<std::uint32_t>::max())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Storage-buffer binding ranges must satisfy D3D12 raw/structured alignment and element-count "
                    "limits.",
                });
            }
            return *size;
        }

        [[nodiscard]] Status writeBuffer(ID3D12Device& device,
                                         const BufferBindingSource& source,
                                         BindingType type,
                                         const BufferBinding& binding,
                                         D3D12_CPU_DESCRIPTOR_HANDLE destination)
        {
            auto size = validateBufferBinding(source, type, binding);
            if (!size)
            {
                return std::unexpected(std::move(size.error()));
            }

            type = staticBufferType(type);
            if (type == BindingType::uniformBuffer)
            {
                const D3D12_CONSTANT_BUFFER_VIEW_DESC native{
                    .BufferLocation = source.resource->GetGPUVirtualAddress() + binding.offset,
                    .SizeInBytes = static_cast<std::uint32_t>(constantBufferViewSize(*size)),
                };
                device.CreateConstantBufferView(&native, destination);
                return {};
            }

            const auto stride = binding.structureByteStride == 0 ? 4U : binding.structureByteStride;
            const auto firstElement = binding.offset / stride;
            const auto elementCount = static_cast<std::uint32_t>(*size / stride);
            const auto format = binding.structureByteStride == 0 ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_UNKNOWN;
            const auto flags =
                binding.structureByteStride == 0 ? D3D12_BUFFER_SRV_FLAG_RAW : D3D12_BUFFER_SRV_FLAG_NONE;

            if (type == BindingType::readOnlyStorageBuffer)
            {
                D3D12_SHADER_RESOURCE_VIEW_DESC native{};
                native.Format = format;
                native.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                native.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                native.Buffer = {
                    firstElement,
                    elementCount,
                    binding.structureByteStride,
                    flags,
                };
                device.CreateShaderResourceView(source.resource, &native, destination);
            }
            else
            {
                D3D12_UNORDERED_ACCESS_VIEW_DESC native{};
                native.Format = format;
                native.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                native.Buffer = {
                    firstElement,
                    elementCount,
                    binding.structureByteStride,
                    0,
                    binding.structureByteStride == 0 ? D3D12_BUFFER_UAV_FLAG_RAW : D3D12_BUFFER_UAV_FLAG_NONE,
                };
                device.CreateUnorderedAccessView(source.resource, nullptr, &native, destination);
            }
            return {};
        }

        void writeNullResource(ID3D12Device& device, BindingType type, D3D12_CPU_DESCRIPTOR_HANDLE destination) noexcept
        {
            type = staticBufferType(type);
            if (type == BindingType::uniformBuffer)
            {
                device.CreateConstantBufferView(nullptr, destination);
                return;
            }
            if (type == BindingType::readOnlyStorageBuffer)
            {
                D3D12_SHADER_RESOURCE_VIEW_DESC native{};
                native.Format = DXGI_FORMAT_R32_TYPELESS;
                native.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                native.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                native.Buffer.NumElements = 1;
                native.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
                device.CreateShaderResourceView(nullptr, &native, destination);
                return;
            }
            if (type == BindingType::storageBuffer)
            {
                D3D12_UNORDERED_ACCESS_VIEW_DESC native{};
                native.Format = DXGI_FORMAT_R32_TYPELESS;
                native.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                native.Buffer.NumElements = 1;
                native.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
                device.CreateUnorderedAccessView(nullptr, nullptr, &native, destination);
                return;
            }
            if (type == BindingType::sampledTexture)
            {
                D3D12_SHADER_RESOURCE_VIEW_DESC native{};
                native.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
                native.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                native.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                native.Texture2D.MipLevels = 1;
                device.CreateShaderResourceView(nullptr, &native, destination);
                return;
            }
            if (type == BindingType::accelerationStructure)
            {
                D3D12_SHADER_RESOURCE_VIEW_DESC native{};
                native.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
                native.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                native.RaytracingAccelerationStructure.Location = 0;
                device.CreateShaderResourceView(nullptr, &native, destination);
                return;
            }
            if (isSamplerFeedback(type))
            {
                Microsoft::WRL::ComPtr<ID3D12Device8> feedbackDevice;
                if (SUCCEEDED(device.QueryInterface(IID_PPV_ARGS(&feedbackDevice))))
                {
                    feedbackDevice->CreateSamplerFeedbackUnorderedAccessView(nullptr, nullptr, destination);
                }
                return;
            }

            D3D12_UNORDERED_ACCESS_VIEW_DESC native{};
            native.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            native.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            device.CreateUnorderedAccessView(nullptr, nullptr, &native, destination);
        }

        void writeNullSampler(ID3D12Device& device, D3D12_CPU_DESCRIPTOR_HANDLE destination) noexcept
        {
            const D3D12_SAMPLER_DESC native{
                .Filter = D3D12_FILTER_MIN_MAG_MIP_POINT,
                .AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                .AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                .AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                .MipLODBias = 0.0F,
                .MaxAnisotropy = 1,
                .ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER,
                .BorderColor = {},
                .MinLOD = 0.0F,
                .MaxLOD = D3D12_FLOAT32_MAX,
            };
            device.CreateSampler(&native, destination);
        }

        [[nodiscard]] std::uint32_t entryLogicalCount(const BindingLayoutRecordEntry& entry,
                                                      const BindingSetRecord& record) noexcept
        {
            return hasAny(entry.entry.flags, BindingFlag::variableDescriptorCount) ? record.variableDescriptorCount
                                                                                   : entry.entry.count;
        }

        [[nodiscard]] Result<BindingDescriptorCounts> bindingDescriptorCounts(const BindingLayoutRecord& layout,
                                                                              const BindingSetRecord& record)
        {
            BindingDescriptorCounts counts{};
            for (const auto& entry : layout.entries)
            {
                const auto index = static_cast<std::size_t>(entry.entry.type);
                const auto count = entryLogicalCount(entry, record);
                if (count > std::numeric_limits<std::uint32_t>::max() - counts[index])
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A binding layout overflows one binding-pool descriptor-type count.",
                    });
                }
                counts[index] += count;
            }
            return counts;
        }

        [[nodiscard]] Status validateVariableDescriptorCount(const BindingLayoutRecord& layout,
                                                             std::uint32_t variableDescriptorCount)
        {
            if (!layout.variableBinding.has_value())
            {
                if (variableDescriptorCount != 0)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A binding set supplied a variable descriptor count for a fixed-size layout.",
                    });
                }
                return {};
            }
            const auto* variable = findEntry(layout, *layout.variableBinding);
            const auto fixedCount = variable->entry.type == BindingType::sampler ? layout.samplerDescriptorCount
                                                                                 : layout.resourceDescriptorCount;
            const auto capacity = variable->entry.type == BindingType::sampler ? bindingSamplerDescriptorBudget
                                                                               : bindingPoolResourceDescriptorBudget;
            if (variableDescriptorCount > capacity - fixedCount)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The variable descriptor count exceeds the remaining shader-visible heap capacity.",
                });
            }
            return {};
        }

        void initializeDescriptorSources(const BindingLayoutRecord& layout, BindingSetRecord& record)
        {
            auto resourceCount = layout.resourceDescriptorCount;
            auto samplerCount = layout.samplerDescriptorCount;
            if (layout.variableBinding.has_value())
            {
                const auto* variable = findEntry(layout, *layout.variableBinding);
                const auto physicalCount = (std::max)(record.variableDescriptorCount, 1U);
                if (variable->entry.type == BindingType::sampler)
                {
                    samplerCount += physicalCount;
                }
                else
                {
                    resourceCount += physicalCount;
                }
            }
            record.resourceSources.resize(resourceCount);
            record.samplerSources.resize(samplerCount);
            for (const auto& entry : layout.entries)
            {
                const auto logicalCount = entryLogicalCount(entry, record);
                const bool backendOnlyPlaceholder =
                    logicalCount == 0 && hasAny(entry.entry.flags, BindingFlag::variableDescriptorCount);
                if (!hasAny(entry.entry.flags, BindingFlag::partiallyBound) && !backendOnlyPlaceholder)
                {
                    continue;
                }
                const auto physicalCount = backendOnlyPlaceholder ? 1U : logicalCount;
                for (std::uint32_t element = 0; element < physicalCount; ++element)
                {
                    const auto offset = entry.descriptorOffset + element;
                    if (entry.entry.type == BindingType::sampler)
                    {
                        record.samplerSources[offset] = NullSamplerDescriptorSource{};
                    }
                    else
                    {
                        record.resourceSources[offset] = NullResourceDescriptorSource{.type = entry.entry.type};
                    }
                }
            }
        }

        [[nodiscard]] bool matchesDescriptorTables(const DescriptorPreparationMask& preparation,
                                                   const BindingSetRecord& record) noexcept
        {
            return preparation.resources.size() == record.resourceSources.size() &&
                   preparation.samplers.size() == record.samplerSources.size();
        }

        [[nodiscard]] Status clearDescriptorSource(const BindingLayoutRecordEntry& entry,
                                                   std::uint32_t descriptorOffset,
                                                   BindingSetRecord& record)
        {
            if (!hasAny(entry.entry.flags, BindingFlag::partiallyBound))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Only a partially-bound layout entry may be cleared.",
                });
            }
            if (entry.entry.type == BindingType::sampler)
            {
                record.samplerSources[descriptorOffset] = NullSamplerDescriptorSource{};
            }
            else
            {
                record.resourceSources[descriptorOffset] = NullResourceDescriptorSource{.type = entry.entry.type};
            }
            return {};
        }

        [[nodiscard]] Result<BufferDescriptorSource> resolveBufferDescriptorSource(ResourceManager& resources,
                                                                                   BindingType type,
                                                                                   const BindingResource& resource)
        {
            const auto* binding = std::get_if<BufferBinding>(&resource);
            if (binding == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The binding write resource type does not match its buffer layout entry.",
                });
            }
            auto source = resources.bufferBindingSource(binding->buffer);
            if (!source)
            {
                return std::unexpected(std::move(source.error()));
            }
            auto size = bindingSize(*source, *binding);
            if (!size)
            {
                return std::unexpected(std::move(size.error()));
            }
            auto retainedBinding = *binding;
            retainedBinding.size = *size;
            return BufferDescriptorSource{
                .type = type,
                .source = std::move(*source),
                .binding = retainedBinding,
            };
        }

        [[nodiscard]] Result<TextureDescriptorSource> resolveTextureDescriptorSource(ViewManager& views,
                                                                                     BindingType type,
                                                                                     const BindingResource& resource)
        {
            const auto* handle = std::get_if<TextureViewHandle>(&resource);
            if (handle == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The binding write resource type does not match its texture layout entry.",
                });
            }
            auto view = views.nativeTextureView(*handle);
            if (!view)
            {
                return std::unexpected(std::move(view.error()));
            }
            const auto expectedUsage =
                type == BindingType::sampledTexture ? TextureViewUsage::sampled : TextureViewUsage::storage;
            if (view->usage != expectedUsage)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The texture view usage does not match the binding layout entry.",
                });
            }
            return TextureDescriptorSource{.source = std::move(*view)};
        }

        [[nodiscard]] Result<AccelerationStructureDescriptorSource> resolveAccelerationStructureDescriptorSource(
            ResourceManager& resources, const BindingResource& resource)
        {
            const auto* handle = std::get_if<AccelerationStructureHandle>(&resource);
            if (handle == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The binding write resource type does not match its acceleration-structure layout entry.",
                });
            }
            auto source = resources.accelerationStructureBindingSource(*handle);
            if (!source)
            {
                return std::unexpected(std::move(source.error()));
            }
            return AccelerationStructureDescriptorSource{.source = std::move(*source)};
        }

        [[nodiscard]] Result<SamplerFeedbackDescriptorSource> resolveSamplerFeedbackDescriptorSource(
            ResourceManager& resources, BindingType type, const BindingResource& resource)
        {
            const auto* handle = std::get_if<SamplerFeedbackMapHandle>(&resource);
            if (handle == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The binding write resource type does not match its sampler-feedback layout entry.",
                });
            }
            const auto expectedType = type == BindingType::minMipFeedback ? SamplerFeedbackMapType::minMip
                                                                          : SamplerFeedbackMapType::mipRegionUsed;
            auto source = resources.samplerFeedbackBindingSource(*handle, expectedType);
            if (!source)
            {
                return std::unexpected(std::move(source.error()));
            }
            return SamplerFeedbackDescriptorSource{.source = std::move(*source)};
        }

        [[nodiscard]] Result<SamplerDescriptorSource> resolveSamplerDescriptorSource(SamplerManager& samplers,
                                                                                     const BindingResource& resource)
        {
            const auto* handle = std::get_if<SamplerHandle>(&resource);
            if (handle == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The binding write resource type does not match its sampler layout entry.",
                });
            }
            auto sampler = samplers.nativeSampler(*handle);
            if (!sampler)
            {
                return std::unexpected(std::move(sampler.error()));
            }
            return SamplerDescriptorSource{.source = std::move(*sampler)};
        }

        struct BindingSourceManagers final
        {
            ResourceManager& resources;
            ViewManager& views;
            SamplerManager& samplers;
        };

        [[nodiscard]] Status storeDescriptorSource(const BindingSourceManagers& managers,
                                                   BindingType type,
                                                   const BindingResource& resource,
                                                   std::uint32_t descriptorOffset,
                                                   BindingSetRecord& record)
        {
            if (isBuffer(type))
            {
                auto source = resolveBufferDescriptorSource(managers.resources, type, resource);
                if (!source)
                {
                    return std::unexpected(std::move(source.error()));
                }
                record.resourceSources[descriptorOffset] = std::move(*source);
                return {};
            }
            if (type == BindingType::sampledTexture || type == BindingType::storageTexture)
            {
                auto source = resolveTextureDescriptorSource(managers.views, type, resource);
                if (!source)
                {
                    return std::unexpected(std::move(source.error()));
                }
                record.resourceSources[descriptorOffset] = std::move(*source);
                return {};
            }
            if (type == BindingType::accelerationStructure)
            {
                auto source = resolveAccelerationStructureDescriptorSource(managers.resources, resource);
                if (!source)
                {
                    return std::unexpected(std::move(source.error()));
                }
                record.resourceSources[descriptorOffset] = std::move(*source);
                return {};
            }
            if (isSamplerFeedback(type))
            {
                auto source = resolveSamplerFeedbackDescriptorSource(managers.resources, type, resource);
                if (!source)
                {
                    return std::unexpected(std::move(source.error()));
                }
                record.resourceSources[descriptorOffset] = std::move(*source);
                return {};
            }
            auto source = resolveSamplerDescriptorSource(managers.samplers, resource);
            if (!source)
            {
                return std::unexpected(std::move(source.error()));
            }
            record.samplerSources[descriptorOffset] = std::move(*source);
            return {};
        }

        [[nodiscard]] Status applyBindingWrites(const BindingSourceManagers& managers,
                                                const BindingLayoutRecord& layout,
                                                std::span<const BindingWrite> writes,
                                                BindingSetRecord& record,
                                                DescriptorPreparationMask* preparation = nullptr)
        {
            if (preparation != nullptr && !matchesDescriptorTables(*preparation, record))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A binding update preparation mask does not match its descriptor tables.",
                });
            }
            std::vector<bool> resourceWritten(record.resourceSources.size(), false);
            std::vector<bool> samplerWritten(record.samplerSources.size(), false);
            for (const auto& write : writes)
            {
                const auto* entry = findEntry(layout, write.binding);
                if (entry == nullptr || write.arrayElement >= entryLogicalCount(*entry, record))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A binding write references an unknown binding or out-of-range array element.",
                    });
                }

                const auto descriptorOffset = entry->descriptorOffset + write.arrayElement;
                auto& written = entry->entry.type == BindingType::sampler ? samplerWritten : resourceWritten;
                if (written[descriptorOffset])
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A binding update cannot write the same descriptor element more than once.",
                    });
                }

                auto stored =
                    std::holds_alternative<std::monostate>(write.resource)
                        ? clearDescriptorSource(*entry, descriptorOffset, record)
                        : storeDescriptorSource(managers, entry->entry.type, write.resource, descriptorOffset, record);
                if (!stored)
                {
                    return stored;
                }
                written[descriptorOffset] = true;
                if (preparation != nullptr)
                {
                    auto& prepared =
                        entry->entry.type == BindingType::sampler ? preparation->samplers : preparation->resources;
                    prepared[descriptorOffset] = 1;
                }
            }
            return {};
        }

        [[nodiscard]] Status validateEveryDescriptorWritten(const BindingSetRecord& record)
        {
            const auto hasSource = [](const auto& source) { return !std::holds_alternative<std::monostate>(source); };
            if (!std::ranges::all_of(record.resourceSources, hasSource) ||
                !std::ranges::all_of(record.samplerSources, hasSource))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Binding set creation requires every non-partially-bound descriptor element exactly once.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateSamplerFeedbackPairing(const BindingSetRecord& record)
        {
            for (const auto& source : record.resourceSources)
            {
                const auto* feedback = std::get_if<SamplerFeedbackDescriptorSource>(&source);
                if (feedback == nullptr)
                {
                    continue;
                }
                const bool hasPairedSampledTexture =
                    std::ranges::any_of(record.resourceSources, [&](const ResourceDescriptorSource& candidate) {
                        const auto* texture = std::get_if<TextureDescriptorSource>(&candidate);
                        return texture != nullptr && texture->source.usage == TextureViewUsage::sampled &&
                               texture->source.texture.resource == feedback->source.targetedResource;
                    });
                if (!hasPairedSampledTexture)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A sampler-feedback binding requires a sampled view of its immutable paired texture in the "
                        "same binding set.",
                    });
                }
            }
            return {};
        }

        void writeAccelerationStructureDescriptor(ID3D12Device& device,
                                                  const AccelerationStructureDescriptorSource& accelerationStructure,
                                                  D3D12_CPU_DESCRIPTOR_HANDLE destination)
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC native{};
            native.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
            native.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            native.RaytracingAccelerationStructure.Location = accelerationStructure.source.address;
            device.CreateShaderResourceView(nullptr, &native, destination);
        }

        [[nodiscard]] Status writeSamplerFeedbackDescriptor(ID3D12Device& device,
                                                            const SamplerFeedbackDescriptorSource& feedback,
                                                            D3D12_CPU_DESCRIPTOR_HANDLE destination)
        {
            Microsoft::WRL::ComPtr<ID3D12Device8> feedbackDevice;
            const HRESULT result = device.QueryInterface(IID_PPV_ARGS(&feedbackDevice));
            if (FAILED(result))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    static_cast<std::int64_t>(result),
                    "ID3D12Device8 became unavailable while materializing a sampler-feedback descriptor.",
                });
            }
            feedbackDevice->CreateSamplerFeedbackUnorderedAccessView(
                feedback.source.targetedResource, feedback.source.feedbackResource, destination);
            return {};
        }

        [[nodiscard]] Status writeStagingResourceDescriptor(ID3D12Device& device,
                                                            const ResourceDescriptorSource& source,
                                                            D3D12_CPU_DESCRIPTOR_HANDLE destination)
        {
            if (const auto* buffer = std::get_if<BufferDescriptorSource>(&source))
            {
                return writeBuffer(device, buffer->source, buffer->type, buffer->binding, destination);
            }
            if (const auto* accelerationStructure = std::get_if<AccelerationStructureDescriptorSource>(&source))
            {
                writeAccelerationStructureDescriptor(device, *accelerationStructure, destination);
                return {};
            }
            if (const auto* feedback = std::get_if<SamplerFeedbackDescriptorSource>(&source))
            {
                return writeSamplerFeedbackDescriptor(device, *feedback, destination);
            }
            if (const auto* null = std::get_if<NullResourceDescriptorSource>(&source))
            {
                writeNullResource(device, null->type, destination);
            }
            return {};
        }

        [[nodiscard]] Result<D3D12_CPU_DESCRIPTOR_HANDLE> resourceDescriptorHandle(
            ID3D12Device& device,
            CpuDescriptorPool& stagingHeap,
            const ResourceDescriptorSource& source,
            std::shared_ptr<CpuDescriptorLease>& staging,
            bool prepare)
        {
            if (const auto* texture = std::get_if<TextureDescriptorSource>(&source))
            {
                staging.reset();
                return texture->source.descriptor;
            }
            if (std::holds_alternative<std::monostate>(source))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A binding set version is missing resource descriptor source metadata.",
                });
            }
            if (prepare)
            {
                auto allocation = stagingHeap.allocate(device);
                if (!allocation)
                {
                    return std::unexpected(std::move(allocation.error()));
                }
                auto written = writeStagingResourceDescriptor(device, source, (*allocation)->handle());
                if (!written)
                {
                    return std::unexpected(std::move(written.error()));
                }
                staging = std::move(*allocation);
            }

            if (staging == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A binding set version is missing a reusable CPU resource descriptor.",
                });
            }
            return staging->handle();
        }

        [[nodiscard]] Result<D3D12_CPU_DESCRIPTOR_HANDLE> samplerDescriptorHandle(
            ID3D12Device& device,
            CpuDescriptorPool& stagingHeap,
            const SamplerDescriptorSourceVariant& source,
            std::shared_ptr<CpuDescriptorLease>& staging,
            bool prepare)
        {
            if (const auto* sampler = std::get_if<SamplerDescriptorSource>(&source))
            {
                staging.reset();
                return sampler->source.descriptor;
            }
            if (std::holds_alternative<NullSamplerDescriptorSource>(source))
            {
                if (prepare)
                {
                    auto allocation = stagingHeap.allocate(device);
                    if (!allocation)
                    {
                        return std::unexpected(std::move(allocation.error()));
                    }
                    writeNullSampler(device, (*allocation)->handle());
                    staging = std::move(*allocation);
                }
                if (staging != nullptr)
                {
                    return staging->handle();
                }
            }
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A binding set version is missing sampler descriptor source metadata.",
            });
        }

        void retainDescriptorResidencies(const ResourceDescriptorSource& source,
                                         std::vector<std::shared_ptr<ResidencyRecord>>& residencies)
        {
            if (const auto* buffer = std::get_if<BufferDescriptorSource>(&source))
            {
                residencies.push_back(buffer->source.residency);
            }
            else if (const auto* texture = std::get_if<TextureDescriptorSource>(&source))
            {
                residencies.push_back(texture->source.texture.residency);
            }
            else if (const auto* feedback = std::get_if<SamplerFeedbackDescriptorSource>(&source))
            {
                residencies.push_back(feedback->source.targetedResidency);
            }
        }

        [[nodiscard]] Status prepareStagingDescriptorTables(BindingSetRecord& record,
                                                            const DescriptorPreparationMask* preparation)
        {
            if (preparation == nullptr)
            {
                record.resourceStagingDescriptors.resize(record.resourceSources.size());
                record.samplerStagingDescriptors.resize(record.samplerSources.size());
                return {};
            }
            if (!matchesDescriptorTables(*preparation, record))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A binding set materialization mask does not match its descriptor tables.",
                });
            }
            if (record.resourceStagingDescriptors.size() != record.resourceSources.size() ||
                record.samplerStagingDescriptors.size() != record.samplerSources.size())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A binding set update lost its reusable CPU descriptor table.",
                });
            }
            return {};
        }

        [[nodiscard]] std::vector<std::shared_ptr<ResidencyRecord>> collectUniqueDescriptorResidencies(
            const std::vector<ResourceDescriptorSource>& sources)
        {
            std::vector<std::shared_ptr<ResidencyRecord>> residencies;
            residencies.reserve(sources.size());
            for (const auto& source : sources)
            {
                retainDescriptorResidencies(source, residencies);
            }
            deduplicateSharedPointers(residencies);
            return residencies;
        }

        struct StagedDescriptorHandles final
        {
            std::vector<D3D12_CPU_DESCRIPTOR_HANDLE> handles;
            std::vector<std::uint32_t> preparedIndices;
            std::uint64_t preparedCount = 0;
        };

        struct SamplerTableIdentity final
        {
            SamplerTableKey key;
            std::vector<std::shared_ptr<void>> owners;
        };

        struct StagedBindingSetDescriptors final
        {
            std::optional<StagedDescriptorHandles> resources;
            std::optional<StagedDescriptorHandles> samplers;
            SamplerTableIdentity samplerIdentity;
        };

        [[nodiscard]] bool reusesPublishedDescriptorTable(const std::vector<std::uint8_t>& prepared,
                                                          const GpuDescriptorLease* published,
                                                          std::size_t descriptorCount) noexcept
        {
            if (std::ranges::any_of(prepared, [](std::uint8_t prepare) { return prepare != 0; }))
            {
                return false;
            }
            return descriptorCount == 0 || (published != nullptr && published->count() == descriptorCount);
        }

        [[nodiscard]] const GpuDescriptorLease* publishedSamplerDescriptors(const BindingSetRecord& record) noexcept
        {
            return record.samplers != nullptr ? record.samplers->descriptors.get() : nullptr;
        }

        void recordStagedDescriptor(StagedDescriptorHandles& staged,
                                    D3D12_CPU_DESCRIPTOR_HANDLE handle,
                                    std::size_t index,
                                    bool prepare,
                                    const DescriptorPreparationMask* preparation)
        {
            staged.handles.push_back(handle);
            if (!prepare)
            {
                return;
            }
            ++staged.preparedCount;
            if (preparation != nullptr)
            {
                staged.preparedIndices.push_back(static_cast<std::uint32_t>(index));
            }
        }

        [[nodiscard]] Result<std::optional<StagedDescriptorHandles>> stageResourceDescriptorHandles(
            ID3D12Device& device,
            CpuDescriptorPool& stagingHeap,
            BindingSetRecord& record,
            const DescriptorPreparationMask* preparation)
        {
            if (preparation != nullptr && reusesPublishedDescriptorTable(preparation->resources,
                                                                         record.resources.get(),
                                                                         record.resourceSources.size()))
            {
                return std::optional<StagedDescriptorHandles>{};
            }
            StagedDescriptorHandles staged;
            staged.handles.reserve(record.resourceSources.size());
            for (std::size_t index = 0; index < record.resourceSources.size(); ++index)
            {
                const bool prepare = preparation == nullptr || preparation->resources[index] != 0;
                auto handle = resourceDescriptorHandle(device,
                                                       stagingHeap,
                                                       record.resourceSources[index],
                                                       record.resourceStagingDescriptors[index],
                                                       prepare);
                if (!handle)
                {
                    return std::unexpected(std::move(handle.error()));
                }
                recordStagedDescriptor(staged, *handle, index, prepare, preparation);
            }
            return staged;
        }

        [[nodiscard]] Result<std::optional<StagedDescriptorHandles>> stageSamplerDescriptorHandles(
            ID3D12Device& device,
            CpuDescriptorPool& stagingHeap,
            BindingSetRecord& record,
            const DescriptorPreparationMask* preparation)
        {
            if (preparation != nullptr && reusesPublishedDescriptorTable(preparation->samplers,
                                                                         publishedSamplerDescriptors(record),
                                                                         record.samplerSources.size()))
            {
                return std::optional<StagedDescriptorHandles>{};
            }
            StagedDescriptorHandles staged;
            staged.handles.reserve(record.samplerSources.size());
            for (std::size_t index = 0; index < record.samplerSources.size(); ++index)
            {
                const bool prepare = preparation == nullptr || preparation->samplers[index] != 0;
                auto handle = samplerDescriptorHandle(device,
                                                      stagingHeap,
                                                      record.samplerSources[index],
                                                      record.samplerStagingDescriptors[index],
                                                      prepare);
                if (!handle)
                {
                    return std::unexpected(std::move(handle.error()));
                }
                recordStagedDescriptor(staged, *handle, index, prepare, preparation);
            }
            return staged;
        }

        [[nodiscard]] SamplerTableIdentity samplerTableIdentity(const BindingSetRecord& record)
        {
            SamplerTableIdentity identity;
            identity.key.reserve(record.samplerSources.size());
            for (const auto& source : record.samplerSources)
            {
                const auto* sampler = std::get_if<SamplerDescriptorSource>(&source);
                if (sampler == nullptr)
                {
                    identity.key.push_back(0U);
                    continue;
                }
                identity.key.push_back(static_cast<std::uintptr_t>(sampler->source.descriptor.ptr));
                identity.owners.push_back(sampler->source.owner);
            }
            return identity;
        }

        void copyDescriptorTable(ID3D12Device& device,
                                 const GpuDescriptorLease& table,
                                 const StagedDescriptorHandles& staged,
                                 D3D12_DESCRIPTOR_HEAP_TYPE type)
        {
            const auto count = static_cast<UINT>(staged.handles.size());
            const auto destination = table.cpuHandle();
            device.CopyDescriptors(1, &destination, &count, count, staged.handles.data(), nullptr, type);
        }

        void copyPreparedDescriptors(ID3D12Device& device,
                                     const GpuDescriptorLease& table,
                                     const StagedDescriptorHandles& staged,
                                     D3D12_DESCRIPTOR_HEAP_TYPE type)
        {
            for (const auto index : staged.preparedIndices)
            {
                device.CopyDescriptorsSimple(1, table.cpuHandle(index), staged.handles[index], type);
            }
        }

        void countResourceTablePublish(BindingUpdateCounters& counters,
                                       const StagedDescriptorHandles& staged,
                                       std::uint64_t publishedCount)
        {
            counters.resourceDescriptorsPrepared.fetch_add(staged.preparedCount, std::memory_order_relaxed);
            counters.resourceTablePublishCount.fetch_add(1, std::memory_order_relaxed);
            counters.resourceDescriptorsPublished.fetch_add(publishedCount, std::memory_order_relaxed);
        }

        void countSamplerTablePublish(BindingUpdateCounters& counters,
                                      const StagedDescriptorHandles& staged,
                                      std::uint64_t publishedCount)
        {
            counters.samplerDescriptorsPrepared.fetch_add(staged.preparedCount, std::memory_order_relaxed);
            counters.samplerTablePublishCount.fetch_add(1, std::memory_order_relaxed);
            counters.samplerDescriptorsPublished.fetch_add(publishedCount, std::memory_order_relaxed);
        }

        class SamplerTableCache final
        {
        public:
            [[nodiscard]] Result<std::shared_ptr<SharedSamplerTable>> acquire(ID3D12Device& device,
                                                                              GpuDescriptorHeap& heap,
                                                                              SamplerTableIdentity identity,
                                                                              const StagedDescriptorHandles& staged,
                                                                              BindingUpdateCounters& counters)
            {
                const std::scoped_lock lock(mutex_);
                if (auto shared = findLocked(identity.key); shared != nullptr)
                {
                    counters.samplerDescriptorsPrepared.fetch_add(staged.preparedCount, std::memory_order_relaxed);
                    return shared;
                }
                return publishLocked(device, heap, std::move(identity), staged, counters);
            }

            [[nodiscard]] Result<std::shared_ptr<SharedSamplerTable>> update(
                ID3D12Device& device,
                GpuDescriptorHeap& heap,
                const std::shared_ptr<SharedSamplerTable>& current,
                bool idle,
                SamplerTableIdentity identity,
                const StagedDescriptorHandles& staged,
                BindingUpdateCounters& counters)
            {
                const std::scoped_lock lock(mutex_);
                if (auto shared = findLocked(identity.key); shared != nullptr)
                {
                    counters.samplerDescriptorsPrepared.fetch_add(staged.preparedCount, std::memory_order_relaxed);
                    return shared;
                }
                const bool rewritable = idle && current != nullptr &&
                                        current.use_count() == exclusiveSamplerTableHolders &&
                                        current->descriptors->count() == staged.handles.size();
                if (!rewritable)
                {
                    return publishLocked(device, heap, std::move(identity), staged, counters);
                }
                copyPreparedDescriptors(device, *current->descriptors, staged, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
                countSamplerTablePublish(counters, staged, staged.preparedIndices.size());
                tables_.erase(current->key);
                current->key = identity.key;
                current->samplerOwners = std::move(identity.owners);
                insertLocked(std::move(identity.key), current);
                return current;
            }

        private:
            [[nodiscard]] std::shared_ptr<SharedSamplerTable> findLocked(const SamplerTableKey& key) const
            {
                const auto found = tables_.find(key);
                return found != tables_.end() ? found->second.lock() : nullptr;
            }

            [[nodiscard]] Result<std::shared_ptr<SharedSamplerTable>> publishLocked(
                ID3D12Device& device,
                GpuDescriptorHeap& heap,
                SamplerTableIdentity identity,
                const StagedDescriptorHandles& staged,
                BindingUpdateCounters& counters)
            {
                auto descriptors = heap.allocate(static_cast<std::uint32_t>(staged.handles.size()));
                if (!descriptors)
                {
                    return std::unexpected(std::move(descriptors.error()));
                }
                copyDescriptorTable(device, **descriptors, staged, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
                countSamplerTablePublish(counters, staged, staged.handles.size());
                auto table = std::make_shared<SharedSamplerTable>(SharedSamplerTable{
                    .descriptors = std::move(*descriptors),
                    .key = identity.key,
                    .samplerOwners = std::move(identity.owners),
                });
                insertLocked(std::move(identity.key), table);
                return table;
            }

            void insertLocked(SamplerTableKey key, const std::shared_ptr<SharedSamplerTable>& table)
            {
                if (tables_.size() >= pruneThreshold_)
                {
                    std::erase_if(tables_, [](const auto& entry) { return entry.second.expired(); });
                    pruneThreshold_ = (std::max)(minimumSamplerTablePruneThreshold, tables_.size() * 2);
                }
                tables_.insert_or_assign(std::move(key), table);
            }

            std::mutex mutex_;
            std::map<SamplerTableKey, std::weak_ptr<SharedSamplerTable>> tables_;
            std::size_t pruneThreshold_ = minimumSamplerTablePruneThreshold;
        };

        class DescriptorBudget final
        {
        public:
            explicit DescriptorBudget(std::uint32_t capacity) noexcept : capacity_(capacity)
            {
            }

            [[nodiscard]] bool tryReserve(std::uint32_t count)
            {
                const std::scoped_lock lock(mutex_);
                if (count > capacity_ - reserved_)
                {
                    return false;
                }
                reserved_ += count;
                return true;
            }

            void release(std::uint32_t count)
            {
                const std::scoped_lock lock(mutex_);
                reserved_ -= count;
            }

        private:
            std::mutex mutex_;
            std::uint32_t capacity_ = 0;
            std::uint32_t reserved_ = 0;
        };

        class DescriptorBudgetReservation final
        {
        public:
            explicit DescriptorBudgetReservation(std::shared_ptr<DescriptorBudget> budget) noexcept
                : budget_(std::move(budget))
            {
            }

            ~DescriptorBudgetReservation()
            {
                if (count_ != 0)
                {
                    budget_->release(count_);
                }
            }

            DescriptorBudgetReservation(const DescriptorBudgetReservation&) = delete;
            DescriptorBudgetReservation& operator=(const DescriptorBudgetReservation&) = delete;
            DescriptorBudgetReservation(DescriptorBudgetReservation&&) = delete;
            DescriptorBudgetReservation& operator=(DescriptorBudgetReservation&&) = delete;

            [[nodiscard]] bool reserve(std::uint32_t count)
            {
                if (!budget_->tryReserve(count))
                {
                    return false;
                }
                count_ = count;
                return true;
            }

        private:
            std::shared_ptr<DescriptorBudget> budget_;
            std::uint32_t count_ = 0;
        };

        [[nodiscard]] bool isDynamicBufferSource(const ResourceDescriptorSource& source) noexcept
        {
            if (const auto* buffer = std::get_if<BufferDescriptorSource>(&source))
            {
                return isDynamicBuffer(buffer->type);
            }
            if (const auto* null = std::get_if<NullResourceDescriptorSource>(&source))
            {
                return isDynamicBuffer(null->type);
            }
            return false;
        }

        void indexDynamicBufferSources(BindingSetRecord& record)
        {
            record.dynamicResourceIndices.fill(0U);
            record.dynamicResourceCount = 0;
            for (std::uint32_t index = 0; index < record.resourceSources.size(); ++index)
            {
                if (isDynamicBufferSource(record.resourceSources[index]) &&
                    record.dynamicResourceCount < maxDynamicBufferOffsetsPerBindingSet)
                {
                    record.dynamicResourceIndices[record.dynamicResourceCount++] = index;
                }
            }
        }

        [[nodiscard]] bool needsSpecialResourceCapture(const ResourceDescriptorSource& source) noexcept
        {
            if (const auto* buffer = std::get_if<BufferDescriptorSource>(&source))
            {
                return buffer->source.sparse;
            }
            if (const auto* texture = std::get_if<TextureDescriptorSource>(&source))
            {
                return texture->source.texture.sparse || texture->source.texture.swapchainLease != nullptr;
            }
            return std::holds_alternative<AccelerationStructureDescriptorSource>(source);
        }

        [[nodiscard]] Result<StagedBindingSetDescriptors> stageBindingSetDescriptors(
            ID3D12Device& device,
            CpuDescriptorPool& resourceStagingHeap,
            CpuDescriptorPool& samplerStagingHeap,
            BindingSetRecord& record,
            const DescriptorPreparationMask* preparation = nullptr)
        {
            auto stagingTablesReady = prepareStagingDescriptorTables(record, preparation);
            if (!stagingTablesReady)
            {
                return std::unexpected(std::move(stagingTablesReady.error()));
            }

            auto residencies = collectUniqueDescriptorResidencies(record.resourceSources);
            auto resourceHandles = stageResourceDescriptorHandles(device, resourceStagingHeap, record, preparation);
            if (!resourceHandles)
            {
                return std::unexpected(std::move(resourceHandles.error()));
            }
            auto samplerHandles = stageSamplerDescriptorHandles(device, samplerStagingHeap, record, preparation);
            if (!samplerHandles)
            {
                return std::unexpected(std::move(samplerHandles.error()));
            }

            StagedBindingSetDescriptors staged{
                .resources = std::move(*resourceHandles),
                .samplers = std::move(*samplerHandles),
            };
            if (staged.samplers.has_value())
            {
                staged.samplerIdentity = samplerTableIdentity(record);
            }
            indexDynamicBufferSources(record);
            record.hasSpecialResourceCaptures =
                std::ranges::any_of(record.resourceSources, needsSpecialResourceCapture);
            record.residencies = std::move(residencies);
            return staged;
        }

        [[nodiscard]] Status validateBindingPoolAlive(const BindingSetRecord& record)
        {
            if (record.poolSet == nullptr)
            {
                return {};
            }
            const std::scoped_lock poolLock(record.poolSet->state->mutex);
            if (!record.poolSet->state->logicalAlive)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The binding pool was destroyed before descriptor-version allocation completed.",
                });
            }
            return {};
        }

        struct CommandRetainedBindingSet final
        {
            BindingPoolCommandRetention commandRetention;
            std::shared_ptr<BindingSetRecord> record;
        };

        struct SpecialResourceCaptures final
        {
            std::uint64_t recordingId = 0;
            SmallVector<std::shared_ptr<ResidencyRecord>, 8> extraResidencies;
            SmallVector<std::shared_ptr<SwapchainImageLeaseState>, 4> swapchainImageLeases;
            SmallVector<std::shared_ptr<void>, 4> sparseMappingOwners;
            SmallVector<TimelinePoint, 4> sparseMappingDependencies;
            SmallVector<std::shared_ptr<void>, 4> accelerationStructureOwners;
            SmallVector<AccelerationStructureExecutionLease, 4> accelerationStructureLeases;
        };

        struct DynamicBufferResolution final
        {
            std::span<const std::uint32_t> offsets;
            const RootSignatureSlotPlan* slotPlan = nullptr;
            std::array<NativeRootBufferBinding, maxDynamicBufferOffsetsPerBindingSet> rootBuffers{};
            std::uint32_t rootBufferCount = 0;
        };

        struct DynamicTableDescriptor final
        {
            const BufferDescriptorSource* buffer = nullptr;
            BindingType type = BindingType::dynamicUniformBuffer;
            BufferBinding binding{};
        };

        struct DynamicTableDescriptors final
        {
            std::array<DynamicTableDescriptor, maxDynamicBufferOffsetsPerBindingSet> descriptors{};
            std::uint32_t count = 0;
        };

        struct DynamicTableDescriptorKey final
        {
            std::uintptr_t resource = 0;
            std::uint64_t offset = 0;
            std::uint64_t size = 0;
            std::uint32_t structureByteStride = 0;
            BindingType type = BindingType::dynamicUniformBuffer;

            auto operator<=>(const DynamicTableDescriptorKey&) const = default;
        };

        struct DynamicTableKey final
        {
            std::array<DynamicTableDescriptorKey, maxDynamicBufferOffsetsPerBindingSet> descriptors{};
            std::uint32_t count = 0;

            auto operator<=>(const DynamicTableKey&) const = default;
        };

        [[nodiscard]] DynamicTableKey dynamicTableKey(const DynamicTableDescriptors& table) noexcept
        {
            DynamicTableKey key{.count = table.count};
            for (std::uint32_t index = 0; index < table.count; ++index)
            {
                const auto& descriptor = table.descriptors[index];
                key.descriptors[index] = DynamicTableDescriptorKey{
                    .resource = descriptor.buffer != nullptr
                                    ? reinterpret_cast<std::uintptr_t>(descriptor.buffer->source.resource)
                                    : 0U,
                    .offset = descriptor.binding.offset,
                    .size = descriptor.binding.size,
                    .structureByteStride = descriptor.binding.structureByteStride,
                    .type = descriptor.type,
                };
            }
            return key;
        }

        [[nodiscard]] Result<BufferBinding> applyDynamicOffset(const BufferBinding& base, std::uint32_t dynamicOffset)
        {
            if (dynamicOffset > std::numeric_limits<std::uint64_t>::max() - base.offset)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A dynamic buffer offset overflows the immutable base offset.",
                });
            }
            auto binding = base;
            binding.offset += dynamicOffset;
            return binding;
        }

        [[nodiscard]] Status promoteToRootBuffer(const BufferDescriptorSource& buffer,
                                                 const BufferBinding& binding,
                                                 std::size_t dynamicIndex,
                                                 DynamicBufferResolution& dynamicBuffers)
        {
            auto validatedSize = validateBufferBinding(buffer.source, buffer.type, binding);
            if (!validatedSize)
            {
                return std::unexpected(std::move(validatedSize.error()));
            }
            dynamicBuffers.rootBuffers[dynamicBuffers.rootBufferCount++] = NativeRootBufferBinding{
                .rootParameter = dynamicBuffers.slotPlan->dynamicRootParameters[dynamicIndex],
                .kind = dynamicBuffers.slotPlan->dynamicRootKinds[dynamicIndex],
                .gpuVirtualAddress = buffer.source.resource->GetGPUVirtualAddress() + binding.offset,
            };
            return {};
        }

        [[nodiscard]] Status placeDynamicTableDescriptor(const RootSignatureSlotPlan& slotPlan,
                                                         std::uint32_t dynamicIndex,
                                                         DynamicTableDescriptor descriptor,
                                                         DynamicTableDescriptors& table)
        {
            const auto tableOffset = slotPlan.dynamicTableOffsets[dynamicIndex];
            if (tableOffset >= slotPlan.dynamicTableDescriptorCount)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A dynamic buffer has no slot in its pipeline layout's dynamic descriptor table.",
                });
            }
            table.descriptors[tableOffset] = descriptor;
            return {};
        }

        [[nodiscard]] Status resolveDynamicBuffer(const BindingSetRecord& record,
                                                  std::uint32_t dynamicIndex,
                                                  DynamicBufferResolution& dynamicBuffers,
                                                  DynamicTableDescriptors& table)
        {
            const auto& slotPlan = *dynamicBuffers.slotPlan;
            const bool promoted = slotPlan.dynamicRootParameters[dynamicIndex] != unpromotedRootParameter;
            const auto& source = record.resourceSources[record.dynamicResourceIndices[dynamicIndex]];
            if (const auto* null = std::get_if<NullResourceDescriptorSource>(&source); null != nullptr && !promoted)
            {
                return placeDynamicTableDescriptor(
                    slotPlan, dynamicIndex, DynamicTableDescriptor{.type = null->type}, table);
            }
            const auto* buffer = std::get_if<BufferDescriptorSource>(&source);
            if (buffer == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A dynamic buffer binding is missing its buffer descriptor source.",
                });
            }
            auto binding = applyDynamicOffset(buffer->binding, dynamicBuffers.offsets[dynamicIndex]);
            if (!binding)
            {
                return std::unexpected(std::move(binding.error()));
            }
            if (promoted)
            {
                return promoteToRootBuffer(*buffer, *binding, dynamicIndex, dynamicBuffers);
            }
            auto validatedSize = validateBufferBinding(buffer->source, buffer->type, *binding);
            if (!validatedSize)
            {
                return std::unexpected(std::move(validatedSize.error()));
            }
            return placeDynamicTableDescriptor(
                slotPlan,
                dynamicIndex,
                DynamicTableDescriptor{.buffer = buffer, .type = buffer->type, .binding = *binding},
                table);
        }

        [[nodiscard]] Status resolveDynamicBuffers(const BindingSetRecord& record,
                                                   DynamicBufferResolution& dynamicBuffers,
                                                   DynamicTableDescriptors& table)
        {
            if (dynamicBuffers.offsets.empty())
            {
                return {};
            }
            if (dynamicBuffers.slotPlan == nullptr || record.dynamicResourceCount != dynamicBuffers.offsets.size() ||
                dynamicBuffers.slotPlan->dynamicBufferCount != dynamicBuffers.offsets.size())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A retained binding set has inconsistent dynamic descriptor source metadata.",
                });
            }
            table.count = dynamicBuffers.slotPlan->dynamicTableDescriptorCount;
            for (std::uint32_t dynamicIndex = 0; dynamicIndex < record.dynamicResourceCount; ++dynamicIndex)
            {
                auto resolved = resolveDynamicBuffer(record, dynamicIndex, dynamicBuffers, table);
                if (!resolved)
                {
                    return resolved;
                }
            }
            return {};
        }

        [[nodiscard]] Status writeDynamicTableDescriptors(ID3D12Device& device,
                                                          const DynamicTableDescriptors& table,
                                                          const GpuDescriptorLease& page,
                                                          std::uint32_t firstIndex)
        {
            for (std::uint32_t index = 0; index < table.count; ++index)
            {
                const auto& descriptor = table.descriptors[index];
                const auto destination = page.cpuHandle(firstIndex + index);
                if (descriptor.buffer == nullptr)
                {
                    writeNullResource(device, descriptor.type, destination);
                    continue;
                }
                auto written =
                    writeBuffer(device, descriptor.buffer->source, descriptor.type, descriptor.binding, destination);
                if (!written)
                {
                    return written;
                }
            }
            return {};
        }

        void retainSparseSnapshot(std::optional<SparseBindingSnapshot> captured, SpecialResourceCaptures& captures)
        {
            if (!captured.has_value())
            {
                return;
            }
            captures.sparseMappingOwners.push_back(std::move(captured->owner));
            if (captured->publication.isValid() &&
                std::ranges::find(captures.sparseMappingDependencies.span(), captured->publication) ==
                    captures.sparseMappingDependencies.span().end())
            {
                captures.sparseMappingDependencies.push_back(captured->publication);
            }
        }

        [[nodiscard]] Status captureSparseBufferMapping(ResourceManager& resources,
                                                        const BufferDescriptorSource& buffer,
                                                        SpecialResourceCaptures& captures)
        {
            if (!buffer.source.sparse)
            {
                return {};
            }
            auto captured = resources.captureSparseBufferBinding(buffer.source);
            if (!captured)
            {
                return std::unexpected(std::move(captured.error()));
            }
            retainSparseSnapshot(std::move(*captured), captures);
            return {};
        }

        void retainSwapchainImageLease(const TextureDescriptorSource& texture, SpecialResourceCaptures& captures)
        {
            const auto& lease = texture.source.texture.swapchainLease;
            if (lease == nullptr)
            {
                return;
            }
            if (std::ranges::find(captures.swapchainImageLeases.span(), lease) ==
                captures.swapchainImageLeases.span().end())
            {
                captures.swapchainImageLeases.push_back(lease);
            }
        }

        [[nodiscard]] Status captureSparseTextureMapping(ResourceManager& resources,
                                                         const TextureDescriptorSource& texture,
                                                         SpecialResourceCaptures& captures)
        {
            if (!texture.source.texture.sparse)
            {
                return {};
            }
            auto captured = resources.captureSparseTextureBinding(texture.source.texture);
            if (!captured)
            {
                return std::unexpected(std::move(captured.error()));
            }
            retainSparseSnapshot(std::move(*captured), captures);
            return {};
        }

        [[nodiscard]] Status captureAccelerationStructureBinding(
            ResourceManager& resources,
            const AccelerationStructureDescriptorSource& accelerationStructure,
            SpecialResourceCaptures& captures)
        {
            auto captured =
                resources.captureAccelerationStructureBinding(accelerationStructure.source, captures.recordingId);
            if (!captured)
            {
                return std::unexpected(std::move(captured.error()));
            }
            captures.accelerationStructureOwners.push_back(std::move(captured->owner));
            for (const auto& residency : captured->residencies)
            {
                if (residency != nullptr)
                {
                    captures.extraResidencies.push_back(residency);
                }
            }
            if (captured->lease.active())
            {
                captures.accelerationStructureLeases.push_back(std::move(captured->lease));
            }
            return {};
        }

        [[nodiscard]] Status captureSpecialResource(ResourceManager& resources,
                                                    const ResourceDescriptorSource& source,
                                                    SpecialResourceCaptures& captures)
        {
            if (const auto* buffer = std::get_if<BufferDescriptorSource>(&source))
            {
                return captureSparseBufferMapping(resources, *buffer, captures);
            }
            if (const auto* texture = std::get_if<TextureDescriptorSource>(&source))
            {
                retainSwapchainImageLease(*texture, captures);
                return captureSparseTextureMapping(resources, *texture, captures);
            }
            if (const auto* accelerationStructure = std::get_if<AccelerationStructureDescriptorSource>(&source))
            {
                return captureAccelerationStructureBinding(resources, *accelerationStructure, captures);
            }
            return {};
        }

        [[nodiscard]] Status captureAllSpecialResources(ResourceManager& resources,
                                                        const BindingSetRecord& record,
                                                        SpecialResourceCaptures& captures)
        {
            if (!record.hasSpecialResourceCaptures)
            {
                return {};
            }
            for (const auto& source : record.resourceSources)
            {
                auto captured = captureSpecialResource(resources, source, captures);
                if (!captured)
                {
                    return captured;
                }
            }
            return {};
        }

        [[nodiscard]] bool writesOnlyUpdateAfterBindEntries(const BindingLayoutRecord& layout,
                                                            std::span<const BindingWrite> writes) noexcept
        {
            return std::ranges::all_of(writes, [&](const BindingWrite& write) {
                const auto* entry = findEntry(layout, write.binding);
                return entry != nullptr && hasAny(entry->entry.flags, BindingFlag::updateAfterBind);
            });
        }

        [[nodiscard]] bool rewritesLiveResourceTable(const BindingSetRecord& current,
                                                     const StagedBindingSetDescriptors& staged) noexcept
        {
            return current.resources != nullptr && staged.resources.has_value() && !staged.samplers.has_value() &&
                   current.resources->count() == staged.resources->handles.size();
        }

    } // namespace

    struct DynamicDescriptorArena::Impl final
    {
        std::vector<std::shared_ptr<GpuDescriptorLease>> pages;
        std::uint32_t pageCursor = 0;
        std::map<DynamicTableKey, D3D12_GPU_DESCRIPTOR_HANDLE> tables;
    };

    DynamicDescriptorArena::DynamicDescriptorArena() : impl_(std::make_unique<Impl>())
    {
    }

    DynamicDescriptorArena::~DynamicDescriptorArena() = default;

    DynamicDescriptorArena::Impl& DynamicDescriptorArena::impl() noexcept
    {
        return *impl_;
    }

    struct BindingManager::Impl final
    {
        Impl()
            : resourceStagingHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
              samplerStagingHeap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER)
        {
        }

        ID3D12Device* device = nullptr;
        GpuDescriptorHeap resourceHeap;
        GpuDescriptorHeap samplerHeap;
        CpuDescriptorPool resourceStagingHeap;
        CpuDescriptorPool samplerStagingHeap;
        SamplerTableCache samplerTables;
        std::shared_ptr<DescriptorBudget> resourceDescriptorBudget =
            std::make_shared<DescriptorBudget>(bindingPoolResourceDescriptorBudget);
        BindingUpdateCounters counters;
        bool variableDescriptorCountSupported = false;
        bool rayTracingSupported = false;
        bool samplerFeedbackSupported = false;
        std::mutex mutex;
        detail::HandleTable<std::shared_ptr<BindingLayoutRecord>, BindingLayoutHandle> layouts;
        detail::HandleTable<std::shared_ptr<BindingPoolState>, BindingPoolHandle> pools;
        detail::HandleTable<std::shared_ptr<BindingSetRecord>, BindingSetHandle> sets;

        [[nodiscard]] Result<std::shared_ptr<BindingLayoutRecord>> layout(BindingLayoutHandle handle)
        {
            const std::scoped_lock lock(mutex);
            const auto* found = layouts.find(handle);
            if (found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The binding layout handle is invalid or stale.",
                });
            }
            return *found;
        }

        [[nodiscard]] Result<std::shared_ptr<BindingPoolState>> pool(BindingPoolHandle handle)
        {
            const std::scoped_lock lock(mutex);
            const auto* found = pools.find(handle);
            if (found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The binding pool handle is invalid or stale.",
                });
            }
            return *found;
        }

        [[nodiscard]] Result<std::shared_ptr<BindingSetRecord>> bindingSet(BindingSetHandle handle)
        {
            const std::scoped_lock lock(mutex);
            const auto* found = sets.find(handle);
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

        [[nodiscard]] Status publishBindingSetVersion(BindingSetHandle handle,
                                                      const std::shared_ptr<BindingSetRecord>& current,
                                                      std::shared_ptr<BindingSetRecord>&& candidate,
                                                      StagedBindingSetDescriptors& staged,
                                                      std::shared_ptr<BindingPoolVersionLease>&& nextPoolVersion)
        {
            std::shared_ptr<BindingSetRecord> retired;
            const std::scoped_lock lock(mutex);
            auto* stored = sets.find(handle);
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
            if (current->poolSet != nullptr)
            {
                auto committed =
                    commitBindingPoolVersionUpdate(*current->poolSet, current->poolVersion, *nextPoolVersion);
                if (!committed)
                {
                    return committed;
                }
            }
            const bool idle =
                current->poolSet != nullptr && current->poolVersion != nullptr && !current->poolVersion->armed;
            auto published = publishUpdatedTables(*current, *candidate, staged, idle);
            if (!published)
            {
                if (current->poolSet != nullptr)
                {
                    revertBindingPoolVersionUpdate(*current->poolSet, current->poolVersion, *nextPoolVersion);
                }
                return published;
            }
            if (current->poolSet != nullptr)
            {
                candidate->poolVersion = std::move(nextPoolVersion);
            }
            retired = std::exchange(*stored, std::move(candidate));
            return {};
        }

        [[nodiscard]] Status publishInPlaceUpdate(BindingSetHandle handle,
                                                  const std::shared_ptr<BindingSetRecord>& current,
                                                  std::shared_ptr<BindingSetRecord>&& candidate,
                                                  const StagedBindingSetDescriptors& staged)
        {
            std::shared_ptr<BindingSetRecord> retired;
            const std::scoped_lock lock(mutex);
            auto* stored = sets.find(handle);
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
            copyPreparedDescriptors(
                *device, *current->resources, *staged.resources, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            countResourceTablePublish(counters, *staged.resources, staged.resources->preparedIndices.size());
            candidate->poolVersion = current->poolVersion;
            retired = std::exchange(*stored, std::move(candidate));
            return {};
        }

        [[nodiscard]] Status publishUpdatedTables(const BindingSetRecord& current,
                                                  BindingSetRecord& candidate,
                                                  StagedBindingSetDescriptors& staged,
                                                  bool idle)
        {
            const bool resourcesInPlace = staged.resources.has_value() && idle && candidate.resources != nullptr &&
                                          candidate.resources == current.resources &&
                                          candidate.resources->count() == staged.resources->handles.size();
            std::shared_ptr<GpuDescriptorLease> copiedResources;
            if (staged.resources.has_value() && !resourcesInPlace && !staged.resources->handles.empty())
            {
                auto allocated = resourceHeap.allocate(static_cast<std::uint32_t>(staged.resources->handles.size()));
                if (!allocated)
                {
                    return std::unexpected(std::move(allocated.error()));
                }
                copiedResources = std::move(*allocated);
            }
            if (staged.samplers.has_value() && !staged.samplers->handles.empty())
            {
                auto samplerTable = samplerTables.update(*device,
                                                         samplerHeap,
                                                         current.samplers,
                                                         idle,
                                                         std::move(staged.samplerIdentity),
                                                         *staged.samplers,
                                                         counters);
                if (!samplerTable)
                {
                    return std::unexpected(std::move(samplerTable.error()));
                }
                candidate.samplers = std::move(*samplerTable);
            }
            else if (staged.samplers.has_value())
            {
                candidate.samplers.reset();
            }
            if (resourcesInPlace)
            {
                copyPreparedDescriptors(
                    *device, *candidate.resources, *staged.resources, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                countResourceTablePublish(counters, *staged.resources, staged.resources->preparedIndices.size());
            }
            else if (staged.resources.has_value())
            {
                if (copiedResources != nullptr)
                {
                    copyDescriptorTable(
                        *device, *copiedResources, *staged.resources, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                    countResourceTablePublish(counters, *staged.resources, staged.resources->handles.size());
                }
                candidate.resources = std::move(copiedResources);
            }
            return {};
        }

        [[nodiscard]] Status publishCreatedTables(BindingSetRecord& record, StagedBindingSetDescriptors& staged)
        {
            std::shared_ptr<GpuDescriptorLease> resourceTable;
            if (staged.resources.has_value() && !staged.resources->handles.empty())
            {
                auto allocated = resourceHeap.allocate(static_cast<std::uint32_t>(staged.resources->handles.size()));
                if (!allocated)
                {
                    return std::unexpected(std::move(allocated.error()));
                }
                resourceTable = std::move(*allocated);
            }
            std::shared_ptr<SharedSamplerTable> samplerTable;
            if (staged.samplers.has_value() && !staged.samplers->handles.empty())
            {
                auto acquired = samplerTables.acquire(
                    *device, samplerHeap, std::move(staged.samplerIdentity), *staged.samplers, counters);
                if (!acquired)
                {
                    return std::unexpected(std::move(acquired.error()));
                }
                samplerTable = std::move(*acquired);
            }
            if (resourceTable != nullptr)
            {
                copyDescriptorTable(*device, *resourceTable, *staged.resources, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                countResourceTablePublish(counters, *staged.resources, staged.resources->handles.size());
            }
            record.resources = std::move(resourceTable);
            record.samplers = std::move(samplerTable);
            return {};
        }

        [[nodiscard]] Result<CommandRetainedBindingSet> retainBindingSetForCommand(
            BindingSetHandle handle, std::size_t dynamicOffsetCount, const void* expectedBindingLayoutOwner)
        {
            CommandRetainedBindingSet retained;
            const std::scoped_lock lock(mutex);
            const auto* stored = sets.find(handle);
            if (stored == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The binding set handle is invalid or stale.",
                });
            }
            retained.record = *stored;
            if (expectedBindingLayoutOwner != nullptr && retained.record->layout.get() != expectedBindingLayoutOwner)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A binding set layout does not match its active pipeline layout batch slot.",
                });
            }
            if (dynamicOffsetCount != retained.record->layout->dynamicBufferCount)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Binding a set requires exactly one offset for each dynamic buffer in binding and array-element "
                    "order.",
                });
            }
            retained.commandRetention.arm(retained.record->poolSet, retained.record->poolVersion);
            return retained;
        }

        [[nodiscard]] Result<StagedBindingSetDescriptors> stageDescriptors(
            BindingSetRecord& record, const DescriptorPreparationMask* preparation = nullptr)
        {
            return stageBindingSetDescriptors(*device, resourceStagingHeap, samplerStagingHeap, record, preparation);
        }

        [[nodiscard]] Result<D3D12_GPU_DESCRIPTOR_HANDLE> publishDynamicTable(DynamicDescriptorArena& arena,
                                                                              const DynamicTableDescriptors& table)
        {
            auto& state = arena.impl();
            auto key = dynamicTableKey(table);
            if (const auto found = state.tables.find(key); found != state.tables.end())
            {
                return found->second;
            }
            if (state.pages.empty() || table.count > state.pages.back()->count() - state.pageCursor)
            {
                const auto previousPageSize = state.pages.empty() ? 0U : state.pages.back()->count();
                const auto pageSize = std::clamp(previousPageSize * 2U, table.count, maxDynamicTablePageDescriptors);
                auto page = resourceHeap.allocateBindTablePage(pageSize);
                if (!page && pageSize > table.count)
                {
                    page = resourceHeap.allocateBindTablePage(table.count);
                }
                if (!page)
                {
                    return std::unexpected(std::move(page.error()));
                }
                state.pages.push_back(std::move(*page));
                state.pageCursor = 0;
            }
            const auto& page = *state.pages.back();
            auto written = writeDynamicTableDescriptors(*device, table, page, state.pageCursor);
            if (!written)
            {
                return std::unexpected(std::move(written.error()));
            }
            const auto handle = page.gpuHandle(state.pageCursor);
            state.pageCursor += table.count;
            state.tables.emplace(std::move(key), handle);
            return handle;
        }
    };

    BindingManager::BindingManager() : impl_(std::make_unique<Impl>())
    {
    }

    BindingManager::~BindingManager() = default;

    Status BindingManager::initialize(ID3D12Device& device,
                                      bool variableDescriptorCountSupported,
                                      bool rayTracingSupported,
                                      bool samplerFeedbackSupported)
    {
        auto resources = impl_->resourceHeap.initialize(device,
                                                        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                                                        bindingPoolResourceDescriptorBudget,
                                                        utilityResourceDescriptorCapacity,
                                                        dynamicTableDescriptorCapacity);
        if (!resources)
        {
            return resources;
        }
        auto samplers =
            impl_->samplerHeap.initialize(device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, bindingSamplerDescriptorBudget);
        if (!samplers)
        {
            return samplers;
        }
        impl_->device = &device;
        impl_->variableDescriptorCountSupported = variableDescriptorCountSupported;
        impl_->rayTracingSupported = rayTracingSupported;
        impl_->samplerFeedbackSupported = samplerFeedbackSupported;
        return {};
    }

    Result<BindingLayoutHandle> BindingManager::createBindingLayout(const BindingLayoutDesc& desc)
    {
        auto record = buildLayout(
            desc, impl_->variableDescriptorCountSupported, impl_->rayTracingSupported, impl_->samplerFeedbackSupported);
        if (!record)
        {
            return std::unexpected(std::move(record.error()));
        }
        const std::scoped_lock lock(impl_->mutex);
        return impl_->layouts.emplace(std::move(*record));
    }

    Status BindingManager::destroyBindingLayout(BindingLayoutHandle layout)
    {
        const std::scoped_lock lock(impl_->mutex);
        if (!impl_->layouts.erase(layout))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The binding layout handle is invalid or stale.",
            });
        }
        return {};
    }

    Result<BindingPoolHandle> BindingManager::createBindingPool(const BindingPoolDesc& desc)
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

        auto state = std::make_shared<BindingPoolState>();
        state->maxBindingSets = desc.maxBindingSets;
        state->label = desc.label;
        std::array<bool, bindingTypeCount> seen{};
        std::uint64_t resourceDescriptors = 0;
        std::uint64_t samplerDescriptors = 0;
        for (const auto& size : desc.sizes)
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
        if (resourceDescriptors > bindingPoolResourceDescriptorBudget ||
            samplerDescriptors > bindingSamplerDescriptorBudget)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A binding pool exceeds one D3D12 shader-visible descriptor heap's portable capacity.",
            });
        }
        if (resourceDescriptors != 0)
        {
            auto reservation = std::make_shared<DescriptorBudgetReservation>(impl_->resourceDescriptorBudget);
            if (!reservation->reserve(static_cast<std::uint32_t>(resourceDescriptors)))
            {
                return std::unexpected(Error{
                    ErrorCode::resourceExhausted,
                    0,
                    "The live binding pools leave too little of the D3D12 shader-visible resource descriptor budget "
                    "for this pool.",
                });
            }
            state->descriptorBudgetReservation = std::move(reservation);
        }

        const std::scoped_lock lock(impl_->mutex);
        return impl_->pools.emplace(std::move(state));
    }

    Status BindingManager::destroyBindingPool(BindingPoolHandle pool)
    {
        const std::scoped_lock lock(impl_->mutex);
        auto* found = impl_->pools.find(pool);
        if (found == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The binding pool handle is invalid or stale.",
            });
        }
        {
            const std::scoped_lock poolLock((*found)->mutex);
            (*found)->logicalAlive = false;
        }
        impl_->pools.erase(pool);
        return {};
    }

    Result<BindingSetHandle> BindingManager::createBindingSet(ResourceManager& resources,
                                                              ViewManager& views,
                                                              SamplerManager& samplers,
                                                              const BindingSetDesc& desc)
    {
        auto layout = impl_->layout(desc.layout);
        if (!layout)
        {
            return std::unexpected(std::move(layout.error()));
        }

        auto poolState = impl_->pool(desc.pool);
        if (!poolState)
        {
            return std::unexpected(std::move(poolState.error()));
        }

        auto record = std::make_shared<BindingSetRecord>();
        record->layout = *layout;
        auto validVariableCount = validateVariableDescriptorCount(**layout, desc.variableDescriptorCount);
        if (!validVariableCount)
        {
            return std::unexpected(std::move(validVariableCount.error()));
        }
        record->variableDescriptorCount = desc.variableDescriptorCount;
        record->label = desc.label;
        initializeDescriptorSources(**layout, *record);
        const BindingSourceManagers sourceManagers{.resources = resources, .views = views, .samplers = samplers};
        auto applied = applyBindingWrites(sourceManagers, **layout, desc.writes, *record);
        if (!applied)
        {
            return std::unexpected(std::move(applied.error()));
        }
        auto everyDescriptorWritten = validateEveryDescriptorWritten(*record);
        if (!everyDescriptorWritten)
        {
            return std::unexpected(std::move(everyDescriptorWritten.error()));
        }
        auto validPairing = validateSamplerFeedbackPairing(*record);
        if (!validPairing)
        {
            return std::unexpected(std::move(validPairing.error()));
        }
        auto descriptorCounts = bindingDescriptorCounts(**layout, *record);
        if (!descriptorCounts)
        {
            return std::unexpected(std::move(descriptorCounts.error()));
        }
        record->descriptorCounts = *descriptorCounts;
        auto poolSet = acquireBindingPoolSet(*poolState);
        if (!poolSet)
        {
            return std::unexpected(std::move(poolSet.error()));
        }
        auto poolVersion = acquireBindingPoolVersion(*poolSet, record->descriptorCounts);
        if (!poolVersion)
        {
            return std::unexpected(std::move(poolVersion.error()));
        }
        record->poolSet = std::move(*poolSet);
        record->poolVersion = std::move(*poolVersion);
        auto staged = impl_->stageDescriptors(*record);
        if (!staged)
        {
            return std::unexpected(std::move(staged.error()));
        }
        auto published = impl_->publishCreatedTables(*record, *staged);
        if (!published)
        {
            return std::unexpected(std::move(published.error()));
        }

        const std::scoped_lock lock(impl_->mutex);
        return impl_->sets.emplace(std::move(record));
    }

    Status BindingManager::updateBindingSet(ResourceManager& resources,
                                            ViewManager& views,
                                            SamplerManager& samplers,
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

        auto stored = impl_->bindingSet(set);
        if (!stored)
        {
            return std::unexpected(std::move(stored.error()));
        }
        const auto& current = *stored;
        auto poolAlive = validateBindingPoolAlive(*current);
        if (!poolAlive)
        {
            return poolAlive;
        }

        auto candidate = std::make_shared<BindingSetRecord>(*current);
        candidate->poolVersion.reset();
        DescriptorPreparationMask preparation{
            .resources = std::vector<std::uint8_t>(candidate->resourceSources.size()),
            .samplers = std::vector<std::uint8_t>(candidate->samplerSources.size()),
        };
        const BindingSourceManagers sourceManagers{.resources = resources, .views = views, .samplers = samplers};
        auto applied = applyBindingWrites(sourceManagers, *current->layout, writes, *candidate, &preparation);
        if (!applied)
        {
            return applied;
        }
        auto validPairing = validateSamplerFeedbackPairing(*candidate);
        if (!validPairing)
        {
            return validPairing;
        }
        auto staged = impl_->stageDescriptors(*candidate, &preparation);
        if (!staged)
        {
            return std::unexpected(std::move(staged.error()));
        }
        if (writesOnlyUpdateAfterBindEntries(*current->layout, writes) && rewritesLiveResourceTable(*current, *staged))
        {
            return impl_->publishInPlaceUpdate(set, current, std::move(candidate), *staged);
        }

        std::shared_ptr<BindingPoolVersionLease> nextPoolVersion;
        if (current->poolSet != nullptr)
        {
            nextPoolVersion = std::make_shared<BindingPoolVersionLease>(
                current->poolSet->state, current->poolSet, candidate->descriptorCounts);
        }

        return impl_->publishBindingSetVersion(set, current, std::move(candidate), *staged, std::move(nextPoolVersion));
    }

    Status BindingManager::destroyBindingSet(BindingSetHandle set)
    {
        std::shared_ptr<BindingSetRecord> retired;
        {
            const std::scoped_lock lock(impl_->mutex);
            const auto* stored = impl_->sets.find(set);
            if (stored == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The binding set handle is invalid or stale.",
                });
            }
            retired = *stored;
            static_cast<void>(impl_->sets.erase(set));
        }
        return {};
    }

    Result<NativeBindingLayout> BindingManager::nativeBindingLayout(BindingLayoutHandle layout)
    {
        auto found = impl_->layout(layout);
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        NativeBindingLayout native{
            .owner = *found,
            .resourceDescriptorCount = (*found)->resourceDescriptorCount,
            .samplerDescriptorCount = (*found)->samplerDescriptorCount,
            .dynamicBufferCount = (*found)->dynamicBufferCount,
            .hasVariableDescriptorCount = (*found)->variableBinding.has_value(),
        };
        native.entries.reserve((*found)->entries.size());
        for (const auto& entry : (*found)->entries)
        {
            native.entries.push_back(NativeBindingLayoutEntry{
                .entry = entry.entry,
                .rangeType = entry.rangeType,
                .descriptorOffset = entry.descriptorOffset,
            });
        }
        return native;
    }

    Result<NativeBindingSet> BindingManager::nativeBindingSet(ResourceManager& resources,
                                                              BindingSetHandle set,
                                                              std::span<const std::uint32_t> dynamicOffsets,
                                                              const RootSignatureSlotPlan* slotPlan,
                                                              const void* expectedBindingLayoutOwner,
                                                              std::uint64_t recordingId,
                                                              DynamicDescriptorArena* dynamicArena)
    {
        try
        {
            auto retained = impl_->retainBindingSetForCommand(set, dynamicOffsets.size(), expectedBindingLayoutOwner);
            if (!retained)
            {
                return std::unexpected(std::move(retained.error()));
            }
            const auto& found = retained->record;

            DynamicBufferResolution dynamicBuffers{.offsets = dynamicOffsets, .slotPlan = slotPlan};
            DynamicTableDescriptors dynamicTable;
            auto resolved = resolveDynamicBuffers(*found, dynamicBuffers, dynamicTable);
            if (!resolved)
            {
                return std::unexpected(std::move(resolved.error()));
            }
            D3D12_GPU_DESCRIPTOR_HANDLE dynamicTableHandle{};
            if (dynamicTable.count != 0)
            {
                if (dynamicArena == nullptr)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "Binding dynamic buffers through a descriptor table requires a recording descriptor arena.",
                    });
                }
                auto published = impl_->publishDynamicTable(*dynamicArena, dynamicTable);
                if (!published)
                {
                    return std::unexpected(std::move(published.error()));
                }
                dynamicTableHandle = *published;
            }
            SpecialResourceCaptures captures{.recordingId = recordingId};
            auto captured = captureAllSpecialResources(resources, *found, captures);
            if (!captured)
            {
                return std::unexpected(std::move(captured.error()));
            }
            if (!captures.extraResidencies.empty())
            {
                deduplicateSharedPointers(captures.extraResidencies);
            }

            auto samplerDescriptors =
                found->samplers != nullptr
                    ? std::shared_ptr<GpuDescriptorLease>(found->samplers, found->samplers->descriptors.get())
                    : std::shared_ptr<GpuDescriptorLease>{};
            const auto samplerTable =
                samplerDescriptors != nullptr ? samplerDescriptors->gpuHandle() : D3D12_GPU_DESCRIPTOR_HANDLE{};
            const bool resourceTableIsPerBind = slotPlan != nullptr && slotPlan->resourceTableIsPerBind();
            const auto resourceTable = resourceTableIsPerBind        ? dynamicTableHandle
                                       : found->resources != nullptr ? found->resources->gpuHandle()
                                                                     : D3D12_GPU_DESCRIPTOR_HANDLE{};
            return NativeBindingSet{
                .owner = found,
                .bindingLayoutOwner = found->layout,
                .bindingPoolRetention = std::move(retained->commandRetention),
                .resourceDescriptorOwner = found->resources,
                .samplerDescriptorOwner = std::move(samplerDescriptors),
                .staticResidencies = found->residencies,
                .extraResidencies = std::move(captures.extraResidencies),
                .swapchainImageLeases = std::move(captures.swapchainImageLeases),
                .sparseMappingOwners = std::move(captures.sparseMappingOwners),
                .sparseMappingDependencies = std::move(captures.sparseMappingDependencies),
                .accelerationStructureOwners = std::move(captures.accelerationStructureOwners),
                .accelerationStructureLeases = std::move(captures.accelerationStructureLeases),
                .rootBuffers = dynamicBuffers.rootBuffers,
                .rootBufferCount = dynamicBuffers.rootBufferCount,
                .resources = resourceTable,
                .samplers = samplerTable,
                .dynamicResources = dynamicTableHandle,
                .resourceDescriptorCount = static_cast<std::uint32_t>(found->resourceSources.size()),
                .samplerDescriptorCount = static_cast<std::uint32_t>(found->samplerSources.size()),
                .dynamicDescriptorCount = dynamicTable.count,
            };
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate D3D12 binding-set preparation state.",
            });
        }
    }

    Result<std::shared_ptr<GpuDescriptorLease>> BindingManager::allocateResourceDescriptors(std::uint32_t count)
    {
        return impl_->resourceHeap.allocateTransient(count);
    }

    Result<ClearResourceDescriptorPair> BindingManager::allocateClearResourceDescriptors()
    {
        auto shaderVisible = impl_->resourceHeap.allocateTransient(1);
        if (!shaderVisible)
        {
            return std::unexpected(std::move(shaderVisible.error()));
        }
        auto cpuVisible = impl_->resourceStagingHeap.allocate(*impl_->device);
        if (!cpuVisible)
        {
            return std::unexpected(std::move(cpuVisible.error()));
        }
        return ClearResourceDescriptorPair{
            .shaderVisible = std::move(*shaderVisible),
            .cpuVisible = std::move(*cpuVisible),
        };
    }

    BindingUpdateStatistics BindingManager::statistics() const noexcept
    {
        return impl_->counters.snapshot();
    }

    DescriptorAllocationStatistics BindingManager::descriptorAllocationStatistics() const noexcept
    {
        auto result = impl_->resourceHeap.statistics();
        result += impl_->samplerHeap.statistics();
        result += impl_->resourceStagingHeap.statistics();
        result += impl_->samplerStagingHeap.statistics();
        return result;
    }

    NativeDescriptorHeaps BindingManager::nativeDescriptorHeaps() const noexcept
    {
        return NativeDescriptorHeaps{
            .resources = impl_->resourceHeap.nativeHeap(),
            .samplers = impl_->samplerHeap.nativeHeap(),
        };
    }

} // namespace rachis::d3d12
