#include "D3D12/D3D12ResourceInternal.h"
#include "Internal/TextureFormatRules.h"

#include "RachisResolveDepthPixel.h"
#include "RachisResolveDepthVertex.h"

namespace rachis::d3d12
{

    using namespace resource_detail;

    namespace resource_detail
    {

        [[nodiscard]] Result<std::uint64_t> reserveResourceMemoryIdentity(
            const std::shared_ptr<ResourceMemoryTrackingState>& state) noexcept
        {
            const auto identity = state->allocationIdentities.acquire();
            if (identity.has_value())
            {
                return *identity;
            }
            return std::unexpected(Error{
                ErrorCode::resourceExhausted,
                0,
                "The device resource-memory allocation identity space is exhausted.",
            });
        }

        [[nodiscard]] Result<std::shared_ptr<TrackedResourceMemory>> trackResourceMemory(
            const std::shared_ptr<ResourceMemoryTrackingState>& state,
            std::uint64_t identity,
            ResourceMemoryKind kind,
            MemoryType memoryType,
            std::uint64_t allocationBytes,
            bool dedicatedAllocation,
            bool shared,
            std::string_view label)
        {
            try
            {
                auto tracked = std::make_shared<TrackedResourceMemory>();
                tracked->entry = {
                    .allocationId = identity,
                    .kind = kind,
                    .memoryType = memoryType,
                    .allocationBytes = allocationBytes,
                    .logicalAlive = true,
                    .dedicatedAllocation = dedicatedAllocation,
                    .shared = shared,
                    .label = std::string(label),
                };

                const std::scoped_lock lock(state->mutex);
                const auto [entry, inserted] = state->entries.try_emplace(identity, tracked);
                if (!inserted)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "The device resource-memory identity source issued a duplicate allocation identity.",
                    });
                }
                static_cast<void>(entry);
                tracked->state = state;
                return tracked;
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Could not allocate resource-memory tracking state.",
                });
            }
        }

        [[nodiscard]] Result<std::shared_ptr<TrackedResourceMemory>> trackResourceMemory(
            const std::shared_ptr<ResourceMemoryTrackingState>& state,
            std::uint64_t identity,
            ResourceMemoryKind kind,
            MemoryType memoryType,
            D3D12MA::Allocation& allocation,
            bool shared,
            std::string_view label)
        {
            return trackResourceMemory(state,
                                       identity,
                                       kind,
                                       memoryType,
                                       allocation.GetSize(),
                                       allocation.GetHeap() == nullptr,
                                       shared,
                                       label);
        }

        void markResourceMemoryDestroyed(const std::shared_ptr<TrackedResourceMemory>& tracked)
        {
            if (tracked == nullptr)
            {
                return;
            }
            const auto state = tracked->state.lock();
            if (state == nullptr)
            {
                return;
            }
            const std::scoped_lock lock(state->mutex);
            tracked->entry.logicalAlive = false;
        }

        [[nodiscard]] Status captureSparseBufferMapping(const std::shared_ptr<BufferRecord>& record,
                                                        SparseMappingCommandLeases& leases,
                                                        SparseMappingDependencies& dependencies)
        {
            if (!record->sparse)
            {
                return {};
            }
            if (record->sparseMappings == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A sparse buffer has no immutable mapping version to retain.",
                });
            }
            if (record->sparseMappings->publicationUnavailable)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A sparse buffer mapping generation has no publishable completion point.",
                });
            }

            const auto alreadyCaptured = std::ranges::any_of(
                leases, [&](const auto& lease) { return lease->mapping() == record->sparseMappings; });
            if (!alreadyCaptured)
            {
                leases.push_back(std::make_shared<SparseMappingCommandLease>(record->sparseMappings));
            }

            const auto publication = record->sparseMappings->publication;
            if (publication.isValid() && std::ranges::find(dependencies, publication) == dependencies.end())
            {
                dependencies.push_back(publication);
            }
            return {};
        }

        [[nodiscard]] Status retainResourceCommandDependencies(
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            std::shared_ptr<void> applicationOwner,
            std::span<const std::shared_ptr<ResidencyRecord>> residencies,
            std::span<const std::shared_ptr<SparseMappingCommandLease>> sparseMappings,
            std::span<const TimelinePoint> sparseMappingDependencies,
            std::span<const std::shared_ptr<SwapchainImageLeaseState>> swapchainLeases,
            std::shared_ptr<void> backendOwner)
        {
            try
            {
                SmallVector<QueueManager::ResourceRetention, 8> resources;
                resources.reserve(sparseMappings.size() + (applicationOwner != nullptr ? 1U : 0U) +
                                  (backendOwner != nullptr ? 1U : 0U));
                if (applicationOwner != nullptr)
                {
                    resources.push_back(QueueManager::ResourceRetention{
                        .owner = std::move(applicationOwner),
                        .residencySet = residencies,
                        .internal = false,
                    });
                }
                if (backendOwner != nullptr)
                {
                    resources.push_back(QueueManager::ResourceRetention{
                        .owner = std::move(backendOwner),
                        .internal = true,
                    });
                }
                for (const auto& mapping : sparseMappings)
                {
                    resources.push_back(QueueManager::ResourceRetention{
                        .owner = mapping,
                        .internal = true,
                    });
                }
                return queues.retainCommandDependencies(
                    commandBuffer, resources.span(), swapchainLeases, sparseMappingDependencies);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to stage transactional resource-command dependencies.",
                });
            }
        }

        [[nodiscard]] Status captureSparseTextureMapping(const std::shared_ptr<TextureRecord>& record,
                                                         SparseMappingCommandLeases& leases,
                                                         SparseMappingDependencies& dependencies)
        {
            if (!record->sparse)
            {
                return {};
            }
            if (record->sparseMappings == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A sparse texture has no immutable mapping generation to retain.",
                });
            }
            if (record->sparseMappings->publicationUnavailable)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A sparse texture mapping generation has no publishable completion point.",
                });
            }
            const auto alreadyCaptured = std::ranges::any_of(
                leases, [&](const auto& lease) { return lease->mapping() == record->sparseMappings; });
            if (!alreadyCaptured)
            {
                leases.push_back(std::make_shared<SparseMappingCommandLease>(record->sparseMappings));
            }
            const auto publication = record->sparseMappings->publication;
            if (publication.isValid() && std::ranges::find(dependencies, publication) == dependencies.end())
            {
                dependencies.push_back(publication);
            }
            return {};
        }

        namespace
        {

            [[nodiscard]] Result<D3D12_FEATURE_DATA_FORMAT_SUPPORT> queryFormatSupport(ID3D12Device& device,
                                                                                       DXGI_FORMAT format)
            {
                D3D12_FEATURE_DATA_FORMAT_SUPPORT support{
                    .Format = format,
                    .Support1 = {},
                    .Support2 = {},
                };
                const HRESULT result =
                    device.CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support));
                if (FAILED(result))
                {
                    return std::unexpected(
                        nativeError(ErrorCode::nativeFailure, result, "ID3D12Device::CheckFeatureSupport(format)"));
                }
                return support;
            }

            [[nodiscard]] Error unsupportedFormatError(std::string_view purpose)
            {
                return Error{
                    ErrorCode::unsupported,
                    0,
                    std::string("The selected format does not support ") + std::string(purpose) + '.',
                };
            }

            [[nodiscard]] Status requireSampledFormatSupport(ID3D12Device& device,
                                                             DXGI_FORMAT format,
                                                             std::string_view purpose)
            {
                const auto support = queryFormatSupport(device, format);
                if (!support)
                {
                    return std::unexpected(support.error());
                }
                if (!format::supportsSampledUsage(support->Support1))
                {
                    return std::unexpected(unsupportedFormatError(purpose));
                }
                return {};
            }

        } // namespace

        [[nodiscard]] Status requireFormatSupport(ID3D12Device& device,
                                                  DXGI_FORMAT format,
                                                  D3D12_FORMAT_SUPPORT1 requiredSupport1,
                                                  D3D12_FORMAT_SUPPORT2 requiredSupport2,
                                                  std::string_view purpose)
        {
            const auto support = queryFormatSupport(device, format);
            if (!support)
            {
                return std::unexpected(support.error());
            }
            if ((support->Support1 & requiredSupport1) != requiredSupport1 ||
                (support->Support2 & requiredSupport2) != requiredSupport2)
            {
                return std::unexpected(unsupportedFormatError(purpose));
            }
            return {};
        }

        [[nodiscard]] DXGI_FORMAT storageValidationFormat(const TextureDesc& desc, const format::Info& info) noexcept
        {
            const auto linearFormat = linearStorageViewFormat(desc.format);
            if (linearFormat == Format::unknown ||
                std::ranges::find(desc.viewFormats, linearFormat) == desc.viewFormats.end())
            {
                return info.view;
            }
            const auto linearInfo = format::info(linearFormat);
            return linearInfo ? linearInfo->view : DXGI_FORMAT_UNKNOWN;
        }

        namespace
        {

            [[nodiscard]] Status validateTextureShapeAndUsageFlags(const TextureDesc& desc)
            {
                if (desc.width == 0 || desc.height == 0 || desc.depthOrArrayLayers == 0 || desc.mipLevels == 0 ||
                    !validTextureDimension(desc.dimension) || !validTextureUsage(desc.usage))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A texture requires a valid dimension, non-zero extents and mip count, plus valid usage flags.",
                    });
                }
                if (desc.sampleCount != 1 && desc.sampleCount != 2 && desc.sampleCount != 4 && desc.sampleCount != 8)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A texture sample count must be 1, 2, 4, or 8.",
                    });
                }
                return {};
            }

            [[nodiscard]] Status validateShadingRateImage(const TextureDesc& desc, bool variableRateShadingTier2)
            {
                if (!hasAny(desc.usage, TextureUsage::shadingRateSource))
                {
                    return {};
                }
                if (!variableRateShadingTier2)
                {
                    return std::unexpected(Error{
                        ErrorCode::unsupported,
                        0,
                        "Shading-rate images require Tier 2 variable-rate shading.",
                    });
                }
                if (desc.dimension != TextureDimension::texture2D || desc.depthOrArrayLayers != 1 ||
                    desc.mipLevels != 1 || desc.sampleCount != 1 || desc.format != Format::r8Uint ||
                    hasAny(desc.usage, TextureUsage::renderTarget | TextureUsage::depthStencil))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A shading-rate image must be a single-layer, single-mip, single-sample r8Uint 2D texture "
                        "without attachment usage.",
                    });
                }
                return {};
            }

            [[nodiscard]] Status validateMultisampleAndMipLayout(const TextureDesc& desc)
            {
                const bool colorAttachment = hasAny(desc.usage, TextureUsage::renderTarget);
                const bool depthAttachment = hasAny(desc.usage, TextureUsage::depthStencil);
                const bool singleMip2DAttachment = desc.dimension == TextureDimension::texture2D &&
                                                   desc.mipLevels == 1 && colorAttachment != depthAttachment;
                const bool shaderAccessible = hasAny(desc.usage, TextureUsage::sampled | TextureUsage::storage);
                if (desc.sampleCount != 1 && (!singleMip2DAttachment || shaderAccessible))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A multisampled texture must be a single-mip 2D attachment with optional copy usage.",
                    });
                }
                if (desc.mipLevels > fullMipCount(desc) || desc.mipLevels > D3D12_REQ_MIP_LEVELS)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "The texture mip count exceeds its dimensions or the D3D12 mip limit.",
                    });
                }
                return {};
            }

            [[nodiscard]] Status validateOptimizedClearValue(const TextureDesc& desc, const format::Info& info)
            {
                switch (desc.optimizedClearValue.kind)
                {
                case OptimizedClearValueKind::automatic:
                    break;
                case OptimizedClearValueKind::color:
                    if (!hasAny(desc.usage, TextureUsage::renderTarget) || !desc.viewFormats.empty() ||
                        !std::ranges::all_of(desc.optimizedClearValue.color,
                                             [](float value) { return std::isfinite(value); }))
                    {
                        return std::unexpected(Error{
                            ErrorCode::invalidArgument,
                            0,
                            "An explicit color optimized clear requires finite values and one fixed render-target "
                            "format.",
                        });
                    }
                    break;
                case OptimizedClearValueKind::depthStencil:
                    if (!hasAny(desc.usage, TextureUsage::depthStencil) ||
                        !std::isfinite(desc.optimizedClearValue.depth) || desc.optimizedClearValue.depth < 0.0F ||
                        desc.optimizedClearValue.depth > 1.0F ||
                        desc.optimizedClearValue.stencil > (info.stencil ? 0xFFU : 0U))
                    {
                        return std::unexpected(Error{
                            ErrorCode::invalidArgument,
                            0,
                            "An explicit depth-stencil optimized clear requires depth in [0, 1] and a supported "
                            "stencil value.",
                        });
                    }
                    break;
                default:
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "The optimized clear value kind is invalid.",
                    });
                }
                return {};
            }

            [[nodiscard]] Status validateFormatUsageRestrictions(const TextureDesc& desc, const format::Info& info)
            {
                if (isDepthFormat(info) && (hasAny(desc.usage, TextureUsage::renderTarget | TextureUsage::storage) ||
                                            !detail::depthTextureSupportsDimension(desc.dimension)))
                {
                    return std::unexpected(Error{
                        ErrorCode::unsupported,
                        0,
                        "Depth textures cannot be render-target, storage, or 3D resources.",
                    });
                }
                if (info.compressed &&
                    hasAny(desc.usage, TextureUsage::renderTarget | TextureUsage::storage | TextureUsage::depthStencil))
                {
                    return std::unexpected(Error{
                        ErrorCode::unsupported,
                        0,
                        "Block-compressed textures support copy and sampled usages only.",
                    });
                }
                return {};
            }

            [[nodiscard]] Status validateTextureDimensionLimits(const TextureDesc& desc)
            {
                switch (desc.dimension)
                {
                case TextureDimension::texture1D:
                    if (desc.height != 1 || desc.width > D3D12_REQ_TEXTURE1D_U_DIMENSION ||
                        desc.depthOrArrayLayers > D3D12_REQ_TEXTURE1D_ARRAY_AXIS_DIMENSION)
                    {
                        return std::unexpected(Error{
                            ErrorCode::invalidArgument,
                            0,
                            "A 1D texture requires height 1 and dimensions within D3D12 limits.",
                        });
                    }
                    break;
                case TextureDimension::texture2D:
                    if (desc.width > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
                        desc.height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
                        desc.depthOrArrayLayers > D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION)
                    {
                        return std::unexpected(Error{
                            ErrorCode::invalidArgument,
                            0,
                            "The 2D texture dimensions exceed D3D12 limits.",
                        });
                    }
                    break;
                case TextureDimension::texture3D:
                    if (desc.width > D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION ||
                        desc.height > D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION ||
                        desc.depthOrArrayLayers > D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION)
                    {
                        return std::unexpected(Error{
                            ErrorCode::invalidArgument,
                            0,
                            "The 3D texture dimensions exceed D3D12 limits.",
                        });
                    }
                    break;
                }
                return {};
            }

            [[nodiscard]] constexpr D3D12_FORMAT_SUPPORT1 dimensionFormatSupport(TextureDimension dimension) noexcept
            {
                switch (dimension)
                {
                case TextureDimension::texture1D:
                    return D3D12_FORMAT_SUPPORT1_TEXTURE1D;
                case TextureDimension::texture2D:
                    return D3D12_FORMAT_SUPPORT1_TEXTURE2D;
                case TextureDimension::texture3D:
                    return D3D12_FORMAT_SUPPORT1_TEXTURE3D;
                }
                return D3D12_FORMAT_SUPPORT1_NONE;
            }

            [[nodiscard]] bool repeatsEarlierViewFormat(std::span<const Format> viewFormats, std::size_t index) noexcept
            {
                const auto earlier = viewFormats.first(index);
                return std::ranges::find(earlier, viewFormats[index]) != earlier.end();
            }

            [[nodiscard]] Status validateAlternateViewFormat(ID3D12Device& device,
                                                             const TextureDesc& desc,
                                                             std::size_t index,
                                                             D3D12_FORMAT_SUPPORT1 dimensionSupport)
            {
                const auto viewFormat = desc.viewFormats[index];
                if (viewFormat == desc.format || repeatsEarlierViewFormat(desc.viewFormats, index) ||
                    !isTextureViewFormatCompatible(desc.format, viewFormat))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Alternate texture view formats must be unique, non-base members of one compatible family.",
                    });
                }
                const auto alternateInfo = format::info(viewFormat);
                if (!alternateInfo)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "An alternate texture view format is unknown or unsupported.",
                    });
                }
                return requireFormatSupport(device,
                                            alternateInfo->resource,
                                            dimensionSupport,
                                            D3D12_FORMAT_SUPPORT2_NONE,
                                            "the requested alternate view-format dimension");
            }

            [[nodiscard]] Status validateAlternateViewFormats(ID3D12Device& device,
                                                              const TextureDesc& desc,
                                                              D3D12_FORMAT_SUPPORT1 dimensionSupport)
            {
                if (desc.viewFormats.empty())
                {
                    return {};
                }
                if (format::mutableResource(desc.format) == DXGI_FORMAT_UNKNOWN)
                {
                    return std::unexpected(Error{
                        ErrorCode::unsupported,
                        0,
                        "The texture format has no portable alternate view-format family.",
                    });
                }
                for (std::size_t index = 0; index < desc.viewFormats.size(); ++index)
                {
                    auto valid = validateAlternateViewFormat(device, desc, index, dimensionSupport);
                    if (!valid)
                    {
                        return valid;
                    }
                }
                return {};
            }

            [[nodiscard]] Status requireUsageFormatSupport(ID3D12Device& device,
                                                           const TextureDesc& desc,
                                                           const format::Info& info,
                                                           D3D12_FORMAT_SUPPORT1 dimensionSupport)
            {
                auto supported = requireFormatSupport(device,
                                                      nativeResourceFormat(desc, info),
                                                      dimensionSupport,
                                                      D3D12_FORMAT_SUPPORT2_NONE,
                                                      "the requested texture dimension");
                if (!supported)
                {
                    return supported;
                }
                if (hasAny(desc.usage, TextureUsage::sampled))
                {
                    supported = requireSampledFormatSupport(device, info.sampled, "sampled textures");
                    if (!supported)
                    {
                        return supported;
                    }
                }
                if (hasAny(desc.usage, TextureUsage::renderTarget))
                {
                    supported = requireFormatSupport(device,
                                                     info.view,
                                                     D3D12_FORMAT_SUPPORT1_RENDER_TARGET,
                                                     D3D12_FORMAT_SUPPORT2_NONE,
                                                     "render targets");
                    if (!supported)
                    {
                        return supported;
                    }
                }
                if (hasAny(desc.usage, TextureUsage::depthStencil))
                {
                    supported = requireFormatSupport(device,
                                                     info.view,
                                                     D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL,
                                                     D3D12_FORMAT_SUPPORT2_NONE,
                                                     "depth-stencil attachments");
                    if (!supported)
                    {
                        return supported;
                    }
                }
                if (hasAny(desc.usage, TextureUsage::storage))
                {
                    supported =
                        requireFormatSupport(device,
                                             storageValidationFormat(desc, info),
                                             D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW,
                                             D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE,
                                             "storage textures through the base or a declared compatible linear view");
                    if (!supported)
                    {
                        return supported;
                    }
                }
                return {};
            }

            [[nodiscard]] Status requireMultisampleSupport(ID3D12Device& device,
                                                           const TextureDesc& desc,
                                                           const format::Info& info)
            {
                if (desc.sampleCount == 1 ||
                    supportsSampleCount(format::attachmentSampleCounts(device, info.view), desc.sampleCount))
                {
                    return {};
                }
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The selected format does not support the requested attachment sample count.",
                });
            }

        } // namespace

        [[nodiscard]] Status validateTextureDesc(ID3D12Device& device,
                                                 const TextureDesc& desc,
                                                 const format::Info& info,
                                                 bool variableRateShadingTier2)
        {
            auto valid = validateTextureShapeAndUsageFlags(desc);
            if (!valid)
            {
                return valid;
            }
            valid = validateShadingRateImage(desc, variableRateShadingTier2);
            if (!valid)
            {
                return valid;
            }
            valid = validateMultisampleAndMipLayout(desc);
            if (!valid)
            {
                return valid;
            }
            if (isDepthFormat(info) != hasAny(desc.usage, TextureUsage::depthStencil))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Depth formats and depth-stencil usage must be specified together.",
                });
            }
            valid = validateOptimizedClearValue(desc, info);
            if (!valid)
            {
                return valid;
            }
            valid = validateFormatUsageRestrictions(desc, info);
            if (!valid)
            {
                return valid;
            }
            valid = validateTextureDimensionLimits(desc);
            if (!valid)
            {
                return valid;
            }

            const auto dimensionSupport = dimensionFormatSupport(desc.dimension);
            valid = validateAlternateViewFormats(device, desc, dimensionSupport);
            if (!valid)
            {
                return valid;
            }
            valid = requireUsageFormatSupport(device, desc, info, dimensionSupport);
            if (!valid)
            {
                return valid;
            }
            return requireMultisampleSupport(device, desc, info);
        }

        [[nodiscard]] Status validateTextureAllocationDesc(ID3D12Device& device,
                                                           const TextureDesc& desc,
                                                           const format::Info& info,
                                                           bool variableRateShadingTier2,
                                                           bool residencyPrioritySupported,
                                                           bool manualResidencySupported,
                                                           bool shared)
        {
            auto valid = validateTextureDesc(device, desc, info, variableRateShadingTier2);
            if (!valid)
            {
                return valid;
            }
            if (!validResidencyPriority(desc.residencyPriority) || !validAllocationFlags(desc.allocationFlags))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The texture residency priority or allocation flags are invalid.",
                });
            }
            if (shared && (desc.residencyPriority != ResidencyPriority::automatic ||
                           desc.allocationFlags != ResourceAllocationFlag::none))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Shared textures require automatic residency and cannot request allocation flags.",
                });
            }
            const bool manualResidency = hasAny(desc.allocationFlags, ResourceAllocationFlag::manualResidency);
            if (manualResidency && !manualResidencySupported)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Manual resource residency is unavailable on this device.",
                });
            }
            if (!shared && desc.residencyPriority != ResidencyPriority::automatic && !residencyPrioritySupported)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Explicit resource residency priorities are not supported by this device.",
                });
            }
            return {};
        }

    } // namespace resource_detail

    namespace
    {

        static_assert(static_cast<std::int32_t>(E_OUTOFMEMORY) == kD3D12HresultOutOfMemory);

        static_assert(static_cast<std::int32_t>(DXGI_ERROR_DEVICE_REMOVED) == kDxgiErrorDeviceRemoved);

        static_assert(static_cast<std::int32_t>(DXGI_ERROR_DEVICE_HUNG) == kDxgiErrorDeviceHung);

        static_assert(static_cast<std::int32_t>(DXGI_ERROR_DEVICE_RESET) == kDxgiErrorDeviceReset);

        static_assert(static_cast<std::int32_t>(DXGI_ERROR_DRIVER_INTERNAL_ERROR) == kDxgiErrorDriverInternalError);

        [[nodiscard]] bool validHostRange(std::uint64_t resourceSize, std::uint64_t offset, std::uint64_t size) noexcept
        {
            return offset < resourceSize && size != 0 && size <= resourceSize - offset;
        }

        [[nodiscard]] D3D12_HEAP_TYPE nativeHeapType(MemoryType memoryType) noexcept
        {
            switch (memoryType)
            {
            case MemoryType::upload:
                return D3D12_HEAP_TYPE_UPLOAD;
            case MemoryType::readback:
                return D3D12_HEAP_TYPE_READBACK;
            default:
                return D3D12_HEAP_TYPE_DEFAULT;
            }
        }

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

        inline constexpr std::uint64_t defaultResourceHeapAlignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

        inline constexpr std::uint64_t multisampleResourceHeapAlignment =
            D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT;

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

        [[nodiscard]] constexpr D3D12_HEAP_FLAGS nativeResourceHeapFlags(ResourceHeapClass resourceClass) noexcept
        {
            switch (resourceClass)
            {
            case ResourceHeapClass::buffer:
                return D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
            case ResourceHeapClass::nonAttachmentTexture:
                return D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
            case ResourceHeapClass::attachmentTexture:
                return D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES;
            }
            return D3D12_HEAP_FLAG_NONE;
        }

        [[nodiscard]] constexpr ResourceHeapClass textureResourceHeapClass(TextureUsage usage) noexcept
        {
            return hasAny(usage, TextureUsage::renderTarget | TextureUsage::depthStencil)
                       ? ResourceHeapClass::attachmentTexture
                       : ResourceHeapClass::nonAttachmentTexture;
        }

        [[nodiscard]] constexpr bool validBufferUsage(BufferUsage usage) noexcept
        {
            constexpr auto allowed =
                static_cast<std::uint32_t>(BufferUsage::copySource) |
                static_cast<std::uint32_t>(BufferUsage::copyDestination) |
                static_cast<std::uint32_t>(BufferUsage::vertex) | static_cast<std::uint32_t>(BufferUsage::index) |
                static_cast<std::uint32_t>(BufferUsage::constant) | static_cast<std::uint32_t>(BufferUsage::storage) |
                static_cast<std::uint32_t>(BufferUsage::indirect) | static_cast<std::uint32_t>(BufferUsage::predicate) |
                static_cast<std::uint32_t>(BufferUsage::accelerationStructureStorage) |
                static_cast<std::uint32_t>(BufferUsage::accelerationStructureBuildInput) |
                static_cast<std::uint32_t>(BufferUsage::accelerationStructureScratch) |
                static_cast<std::uint32_t>(BufferUsage::shaderBindingTable) |
                static_cast<std::uint32_t>(BufferUsage::streamOutput);
            const auto raw = static_cast<std::uint32_t>(usage);
            return raw != 0 && (raw & ~allowed) == 0;
        }

        [[nodiscard]] constexpr std::optional<D3D12_RESIDENCY_PRIORITY> nativeResidencyPriority(
            ResidencyPriority priority) noexcept
        {
            switch (priority)
            {
            case ResidencyPriority::minimum:
                return D3D12_RESIDENCY_PRIORITY_MINIMUM;
            case ResidencyPriority::low:
                return D3D12_RESIDENCY_PRIORITY_LOW;
            case ResidencyPriority::normal:
                return D3D12_RESIDENCY_PRIORITY_NORMAL;
            case ResidencyPriority::high:
                return D3D12_RESIDENCY_PRIORITY_HIGH;
            case ResidencyPriority::automatic:
                return std::nullopt;
            }
            return std::nullopt;
        }

        [[nodiscard]] constexpr std::size_t residencyPriorityIndex(ResidencyPriority priority) noexcept
        {
            return static_cast<std::size_t>(priority) - 1U;
        }

        [[nodiscard]] constexpr ResidencyPoolClass bufferPoolClass(MemoryType memoryType) noexcept
        {
            switch (memoryType)
            {
            case MemoryType::upload:
                return ResidencyPoolClass::uploadBuffer;
            case MemoryType::readback:
                return ResidencyPoolClass::readbackBuffer;
            case MemoryType::deviceLocal:
                return ResidencyPoolClass::deviceBuffer;
            }
            return ResidencyPoolClass::deviceBuffer;
        }

        [[nodiscard]] constexpr ResidencyPoolClass texturePoolClass(TextureUsage usage) noexcept
        {
            return hasAny(usage, TextureUsage::renderTarget | TextureUsage::depthStencil)
                       ? ResidencyPoolClass::attachmentTexture
                       : ResidencyPoolClass::nonAttachmentTexture;
        }

        [[nodiscard]] constexpr D3D12_HEAP_TYPE residencyPoolHeapType(ResidencyPoolClass poolClass) noexcept
        {
            switch (poolClass)
            {
            case ResidencyPoolClass::uploadBuffer:
                return D3D12_HEAP_TYPE_UPLOAD;
            case ResidencyPoolClass::readbackBuffer:
                return D3D12_HEAP_TYPE_READBACK;
            case ResidencyPoolClass::deviceBuffer:
            case ResidencyPoolClass::nonAttachmentTexture:
            case ResidencyPoolClass::attachmentTexture:
            case ResidencyPoolClass::count:
                return D3D12_HEAP_TYPE_DEFAULT;
            }
            return D3D12_HEAP_TYPE_DEFAULT;
        }

        [[nodiscard]] constexpr D3D12_HEAP_FLAGS residencyPoolHeapFlags(ResidencyPoolClass poolClass) noexcept
        {
            switch (poolClass)
            {
            case ResidencyPoolClass::deviceBuffer:
            case ResidencyPoolClass::uploadBuffer:
            case ResidencyPoolClass::readbackBuffer:
                return D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
            case ResidencyPoolClass::nonAttachmentTexture:
                return D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
            case ResidencyPoolClass::attachmentTexture:
                return D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES;
            case ResidencyPoolClass::count:
                break;
            }
            return D3D12_HEAP_FLAG_NONE;
        }

        [[nodiscard]] D3D12_RESOURCE_STATES initialBufferState(MemoryType memoryType, BufferUsage usage) noexcept
        {
            switch (memoryType)
            {
            case MemoryType::upload:
                return D3D12_RESOURCE_STATE_GENERIC_READ;
            case MemoryType::readback:
                return D3D12_RESOURCE_STATE_COPY_DEST;
            default:
                if (hasAny(usage, BufferUsage::accelerationStructureStorage))
                {
                    return D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
                }
                return D3D12_RESOURCE_STATE_COMMON;
            }
        }

        [[nodiscard]] Status validateBufferDesc(const BufferDesc& desc,
                                                bool rayTracing,
                                                bool residencyPrioritySupported,
                                                bool manualResidencySupported)
        {
            if (desc.size == 0 || !validBufferUsage(desc.usage) || !validMemoryType(desc.memoryType) ||
                !validResidencyPriority(desc.residencyPriority) || !validAllocationFlags(desc.allocationFlags))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A buffer requires a non-zero size, valid memory and residency values, at least one usage flag, "
                    "and valid allocation flags.",
                });
            }
            const bool manualResidency = hasAny(desc.allocationFlags, ResourceAllocationFlag::manualResidency);
            const bool accelerationStructureStorage = hasAny(desc.usage, BufferUsage::accelerationStructureStorage);
            const bool accelerationStructureScratch = hasAny(desc.usage, BufferUsage::accelerationStructureScratch);
            const bool shaderBindingTable = hasAny(desc.usage, BufferUsage::shaderBindingTable);
            const bool streamOutput = hasAny(desc.usage, BufferUsage::streamOutput);
            if ((accelerationStructureStorage || accelerationStructureScratch || shaderBindingTable) && !rayTracing)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Acceleration-structure and shader-binding-table buffers require ray-tracing support.",
                });
            }
            if ((accelerationStructureStorage && desc.usage != BufferUsage::accelerationStructureStorage) ||
                (accelerationStructureScratch && desc.usage != BufferUsage::accelerationStructureScratch) ||
                ((accelerationStructureStorage || accelerationStructureScratch) &&
                 desc.memoryType != MemoryType::deviceLocal))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Acceleration-structure storage and scratch buffers are dedicated device-local resources.",
                });
            }
            if (streamOutput && desc.memoryType != MemoryType::deviceLocal)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Stream-output data and counter buffers must use device-local memory.",
                });
            }
            if (manualResidency && desc.memoryType != MemoryType::deviceLocal)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Manual residency is available only for device-local buffers.",
                });
            }
            if (manualResidency && !manualResidencySupported)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Manual resource residency is unavailable on this device.",
                });
            }
            if (desc.residencyPriority != ResidencyPriority::automatic && !residencyPrioritySupported)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Explicit resource residency priorities are not supported by this device.",
                });
            }
            if (desc.memoryType == MemoryType::upload &&
                hasAny(desc.usage, BufferUsage::copyDestination | BufferUsage::storage))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Upload buffers cannot be copy destinations or unordered-access storage.",
                });
            }
            if (desc.memoryType == MemoryType::readback && desc.usage != BufferUsage::copyDestination)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Readback buffers support copy-destination usage only.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateSharedBufferDesc(const BufferDesc& desc)
        {
            if (desc.memoryType != MemoryType::deviceLocal || desc.residencyPriority != ResidencyPriority::automatic ||
                desc.allocationFlags != ResourceAllocationFlag::none)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Shared buffers require device-local memory, automatic residency, and no allocation flags.",
                });
            }
            if (hasAny(desc.usage,
                       BufferUsage::accelerationStructureStorage | BufferUsage::accelerationStructureScratch))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Shared buffers cannot contain acceleration-structure storage or scratch data.",
                });
            }
            return {};
        }

        [[nodiscard]] std::optional<D3D12_CLEAR_VALUE> nativeOptimizedClearValue(const TextureDesc& desc,
                                                                                 const format::Info& info) noexcept
        {
            if (hasAny(desc.usage, TextureUsage::depthStencil))
            {
                D3D12_CLEAR_VALUE value{};
                value.Format = info.view;
                value.DepthStencil = desc.optimizedClearValue.kind == OptimizedClearValueKind::depthStencil
                                 ? D3D12_DEPTH_STENCIL_VALUE{
                                       .Depth = desc.optimizedClearValue.depth,
                                       .Stencil = static_cast<std::uint8_t>(desc.optimizedClearValue.stencil),
                                   }
                                 : D3D12_DEPTH_STENCIL_VALUE{.Depth = 1.0F, .Stencil = 0};
                return value;
            }
            if (hasAny(desc.usage, TextureUsage::renderTarget) && desc.viewFormats.empty())
            {
                D3D12_CLEAR_VALUE value{};
                value.Format = info.view;
                if (desc.optimizedClearValue.kind == OptimizedClearValueKind::color)
                {
                    std::ranges::copy(desc.optimizedClearValue.color, value.Color);
                }
                return value;
            }
            return std::nullopt;
        }

        [[nodiscard]] Result<std::shared_ptr<PlacedResourceLease>> reservePlacement(
            const std::shared_ptr<ResourceHeapRecord>& heap,
            ResourceHeapClass requiredClass,
            std::uint64_t offset,
            const ResourceMemoryRequirements& requirements,
            std::string_view resourceKind)
        {
            if (heap->desc.resourceClass != requiredClass)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    std::string("A placed ") + std::string(resourceKind) +
                        " requires a compatible resource-heap class.",
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
            auto& placements = heap->placements;
            auto iterator = placements.begin();
            while (iterator != placements.end())
            {
                const auto existing = iterator->lock();
                if (existing == nullptr)
                {
                    iterator = placements.erase(iterator);
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
            placements.emplace_back(lease);
            return lease;
        }

        [[nodiscard]] Status synchronizeMappedMemoryRanges(BufferRecord& record,
                                                           std::span<const MappedMemoryRange> ranges,
                                                           MemoryType requiredMemoryType,
                                                           std::string_view operation)
        {
            if (ranges.empty())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    std::string("Mapped-memory ") + std::string(operation) + " requires at least one range.",
                });
            }

            const std::scoped_lock lock(record.mapMutex);
            if (!record.alive)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    std::string("The buffer handle became stale before mapped-memory ") + std::string(operation) + '.',
                });
            }
            if (record.desc.memoryType != requiredMemoryType)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    std::string("Mapped-memory ") + std::string(operation) +
                        (requiredMemoryType == MemoryType::upload ? " requires an upload buffer."
                                                                  : " requires a readback buffer."),
                });
            }
            if (!record.mapped)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    std::string("Mapped-memory ") + std::string(operation) + " requires an active mapping.",
                });
            }

            const auto mappedEnd = record.mappedOffset + record.mappedSize;
            for (const auto& range : ranges)
            {
                if (!validRange(record.desc.size, range.offset, range.size))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        std::string("A mapped-memory ") + std::string(operation) + " range is empty or out of bounds.",
                    });
                }
                const auto resolvedSize = resolveSize(record.desc.size, range.offset, range.size);
                if (range.offset < record.mappedOffset || range.offset + resolvedSize > mappedEnd)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        std::string("A mapped-memory ") + std::string(operation) +
                            " range is not contained in the active mapping.",
                    });
                }
            }

            return {};
        }

        [[nodiscard]] Result<ResourceMemoryRequirements> queryMemoryRequirements(ID3D12Device& device,
                                                                                 const D3D12_RESOURCE_DESC& desc,
                                                                                 bool dedicatedAllocation)
        {
            const auto info = device.GetResourceAllocationInfo(0, 1, &desc);
            if (info.SizeInBytes == 0 || info.SizeInBytes == std::numeric_limits<std::uint64_t>::max() ||
                info.Alignment == 0 || !std::has_single_bit(info.Alignment) || info.SizeInBytes % info.Alignment != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::nativeFailure,
                    0,
                    "D3D12 returned invalid resource allocation requirements.",
                });
            }
            return ResourceMemoryRequirements{
                .size = info.SizeInBytes,
                .alignment = info.Alignment,
                .dedicatedAllocation = dedicatedAllocation,
            };
        }

        [[nodiscard]] std::shared_ptr<ResidencyRecord> createManualResidencyRecord(ID3D12Pageable* pageable)
        {
            auto residency = std::make_shared<ResidencyRecord>();
            residency->pageable = pageable;
            residency->state = ResidencyState::resident;
            return residency;
        }

        void nameAllocatedResource(D3D12MA::Allocation& allocation, ID3D12Resource& resource, std::string_view label)
        {
            const auto wide = wideLabel(label);
            if (wide.empty())
            {
                return;
            }
            allocation.SetName(wide.c_str());
            resource.SetName(wide.c_str());
        }
    } // namespace

    [[nodiscard]] Result<std::shared_ptr<ResourceHeapRecord>> ResourceManager::Impl::resourceHeap(
        ResourceHeapHandle handle) const
    {
        const std::scoped_lock lock(resourceHeapsMutex);
        const auto* record = resourceHeaps.find(handle);
        if (record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The resource-heap handle is invalid or stale.",
            });
        }
        return *record;
    }

    [[nodiscard]] Result<D3D12MA::Pool*> ResourceManager::Impl::residencyPool(ResidencyPoolClass poolClass,
                                                                              ResidencyPriority priority)
    {
        if (!validResidencyPriority(priority))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The resource residency priority is invalid.",
            });
        }
        if (priority == ResidencyPriority::automatic)
        {
            return nullptr;
        }
        const auto nativePriority = nativeResidencyPriority(priority);
        if (residencyDevice == nullptr || !nativePriority.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Explicit resource residency priorities are not supported by this device.",
            });
        }

        const auto poolIndex = static_cast<std::size_t>(poolClass);
        const auto priorityIndex = residencyPriorityIndex(priority);
        const std::scoped_lock lock(residencyPoolsMutex);
        auto& pool = residencyPools[poolIndex][priorityIndex];
        if (pool == nullptr)
        {
            D3D12MA::CPOOL_DESC poolDesc(residencyPoolHeapType(poolClass), residencyPoolHeapFlags(poolClass));
            poolDesc.ResidencyPriority = *nativePriority;
            const HRESULT result = allocator->CreatePool(&poolDesc, pool.ReleaseAndGetAddressOf());
            if (FAILED(result))
            {
                return std::unexpected(nativeError(classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result)),
                                                   result,
                                                   "D3D12MA::Allocator::CreatePool(residency priority)"));
            }
        }
        return pool.Get();
    }

    [[nodiscard]] Result<std::shared_ptr<BufferRecord>> ResourceManager::Impl::buffer(BufferHandle handle) const
    {
        const std::scoped_lock lock(buffersMutex);
        const auto* record = buffers.find(handle);
        if (record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The buffer handle is invalid or stale.",
            });
        }
        return *record;
    }

    [[nodiscard]] Result<std::shared_ptr<TextureRecord>> ResourceManager::Impl::texture(TextureHandle handle) const
    {
        const std::scoped_lock lock(texturesMutex);
        const auto* record = textures.find(handle);
        if (record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The texture handle is invalid or stale.",
            });
        }
        return *record;
    }

    [[nodiscard]] Result<std::shared_ptr<SamplerFeedbackMapRecord>> ResourceManager::Impl::samplerFeedbackMap(
        SamplerFeedbackMapHandle handle) const
    {
        const std::scoped_lock lock(samplerFeedbackMapsMutex);
        const auto* record = samplerFeedbackMaps.find(handle);
        if (record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The sampler-feedback map handle is invalid or stale.",
            });
        }
        return *record;
    }

    [[nodiscard]] Result<std::shared_ptr<AccelerationStructureRecord>> ResourceManager::Impl::accelerationStructure(
        AccelerationStructureHandle handle) const
    {
        const std::scoped_lock lock(accelerationStructuresMutex);
        const auto* record = accelerationStructures.find(handle);
        if (record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The acceleration-structure handle is invalid or stale.",
            });
        }
        return *record;
    }

    namespace
    {

        [[nodiscard]] Result<NativeUploadSlab> createMappedUploadSlab(ID3D12Device* device,
                                                                      const ComPtr<D3D12MA::Allocator>& allocator,
                                                                      std::uint64_t byteCapacity,
                                                                      std::string_view operation)
        {
            NativeUploadSlab slab;
            const D3D12MA::ALLOCATION_DESC allocationDesc{
                .Flags = D3D12MA::ALLOCATION_FLAG_NONE,
                .HeapType = D3D12_HEAP_TYPE_UPLOAD,
                .ExtraHeapFlags = {},
                .CustomPool = {},
                .pPrivateData = {},
            };
            const D3D12_RESOURCE_DESC resourceDesc{
                .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
                .Alignment = 0,
                .Width = byteCapacity,
                .Height = 1,
                .DepthOrArraySize = 1,
                .MipLevels = 1,
                .Format = DXGI_FORMAT_UNKNOWN,
                .SampleDesc = {.Count = 1, .Quality = 0},
                .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                .Flags = D3D12_RESOURCE_FLAG_NONE,
            };
            const HRESULT allocationResult = allocator->CreateResource(&allocationDesc,
                                                                       &resourceDesc,
                                                                       D3D12_RESOURCE_STATE_GENERIC_READ,
                                                                       nullptr,
                                                                       slab.allocation.ReleaseAndGetAddressOf(),
                                                                       IID_PPV_ARGS(&slab.resource));
            if (FAILED(allocationResult))
            {
                const std::string createOp =
                    std::string("D3D12MA::Allocator::CreateResource(") + std::string(operation) + ')';
                return std::unexpected(
                    nativeError(allocationResult == E_OUTOFMEMORY ? ErrorCode::outOfMemory : ErrorCode::nativeFailure,
                                allocationResult,
                                createOp));
            }
            slab.device = device;
            slab.allocator = allocator;
            const D3D12_RANGE noRead{0, 0};
            void* mapped = nullptr;
            const HRESULT mapResult = slab.resource->Map(0, &noRead, &mapped);
            if (FAILED(mapResult))
            {
                const std::string mapOp = std::string("ID3D12Resource::Map(") + std::string(operation) + ')';
                return std::unexpected(nativeError(ErrorCode::nativeFailure, mapResult, mapOp));
            }
            slab.mappedData = static_cast<std::byte*>(mapped);
            slab.gpuVirtualAddress = slab.resource->GetGPUVirtualAddress();
            return slab;
        }

        [[nodiscard]] UploadSlice suballocatedUploadSlice(UploadArena<NativeUploadSlab>::Lease&& lease,
                                                          std::uint64_t size)
        {
            const auto sliceOffset = lease.offset();
            auto& slab = lease.payload();
            ID3D12Resource* const sliceResource = slab.resource.Get();
            std::byte* const sliceMapped = slab.mappedData + sliceOffset;
            const D3D12_GPU_VIRTUAL_ADDRESS sliceGpuAddress = slab.gpuVirtualAddress + sliceOffset;
            auto owner = std::make_shared<UploadSliceLeaseOwner>(std::move(lease));
            return UploadSlice{
                .owner = std::move(owner),
                .resource = sliceResource,
                .mappedData = sliceMapped,
                .gpuVirtualAddress = sliceGpuAddress,
                .offset = sliceOffset,
                .size = size,
            };
        }

        [[nodiscard]] Result<NativeIndirectArgumentSlab> createIndirectArgumentSlab(
            ID3D12Device* device,
            const ComPtr<D3D12MA::Allocator>& allocator,
            std::uint64_t byteCapacity,
            std::string_view operation)
        {
            NativeIndirectArgumentSlab slab;
            const D3D12MA::ALLOCATION_DESC allocationDesc{
                .Flags = D3D12MA::ALLOCATION_FLAG_NONE,
                .HeapType = D3D12_HEAP_TYPE_DEFAULT,
                .ExtraHeapFlags = {},
                .CustomPool = {},
                .pPrivateData = {},
            };
            const D3D12_RESOURCE_DESC resourceDesc{
                .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
                .Alignment = 0,
                .Width = byteCapacity,
                .Height = 1,
                .DepthOrArraySize = 1,
                .MipLevels = 1,
                .Format = DXGI_FORMAT_UNKNOWN,
                .SampleDesc = {.Count = 1, .Quality = 0},
                .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                .Flags = D3D12_RESOURCE_FLAG_NONE,
            };
            const HRESULT allocationResult = allocator->CreateResource(&allocationDesc,
                                                                       &resourceDesc,
                                                                       D3D12_RESOURCE_STATE_COMMON,
                                                                       nullptr,
                                                                       slab.allocation.ReleaseAndGetAddressOf(),
                                                                       IID_PPV_ARGS(&slab.resource));
            if (FAILED(allocationResult))
            {
                const std::string createOp =
                    std::string("D3D12MA::Allocator::CreateResource(") + std::string(operation) + ')';
                return std::unexpected(
                    nativeError(allocationResult == E_OUTOFMEMORY ? ErrorCode::outOfMemory : ErrorCode::nativeFailure,
                                allocationResult,
                                createOp));
            }
            slab.device = device;
            slab.allocator = allocator;
            slab.size = byteCapacity;
            return slab;
        }

        [[nodiscard]] UploadSlice dedicatedUploadSlice(NativeUploadSlab&& slab, std::uint64_t size)
        {
            auto owner = std::make_shared<NativeUploadSlab>(std::move(slab));
            return UploadSlice{
                .owner = owner,
                .resource = owner->resource.Get(),
                .mappedData = owner->mappedData,
                .gpuVirtualAddress = owner->gpuVirtualAddress,
                .offset = 0,
                .size = size,
            };
        }

    } // namespace

    Result<UploadSlice> ResourceManager::Impl::allocateUploadSlice(std::uint64_t size,
                                                                   std::uint64_t alignment,
                                                                   std::string_view operation)
    {
        const auto createSlab = [&](std::uint64_t byteCapacity) {
            return createMappedUploadSlab(device, allocator, byteCapacity, operation);
        };

        try
        {
            auto allocation = uploadArena->allocate(size, alignment, createSlab);
            if (!allocation)
            {
                return std::unexpected(std::move(allocation.error()));
            }
            if (allocation->kind == UploadArenaAllocationKind::suballocated)
            {
                return suballocatedUploadSlice(std::move(allocation->lease), size);
            }

            auto dedicatedSlab = createSlab(size);
            if (!dedicatedSlab)
            {
                return std::unexpected(std::move(dedicatedSlab.error()));
            }
            return dedicatedUploadSlice(std::move(*dedicatedSlab), size);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate upload-arena slice retention state.",
            });
        }
    }

    Result<IndirectArgumentSlice> ResourceManager::Impl::allocateIndirectArgumentSlice(QueueType queueType,
                                                                                       std::uint64_t size,
                                                                                       std::uint64_t alignment,
                                                                                       std::string_view operation)
    {
        const auto arenaIndex = static_cast<std::size_t>(queueType);
        if (arenaIndex >= indirectArgumentArenas.size() || indirectArgumentArenas[arenaIndex] == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Indirect-argument slices require a graphics, compute, or copy queue.",
            });
        }
        const auto createSlab = [&](std::uint64_t byteCapacity) {
            return createIndirectArgumentSlab(device, allocator, byteCapacity, operation);
        };

        try
        {
            auto allocation = indirectArgumentArenas[arenaIndex]->allocate(size, alignment, createSlab);
            if (!allocation)
            {
                return std::unexpected(std::move(allocation.error()));
            }
            if (allocation->kind != UploadArenaAllocationKind::suballocated)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Indirect-argument slices must fit in one indirect-argument arena slab.",
                });
            }
            const auto offset = allocation->lease.offset();
            auto& slab = allocation->lease.payload();
            ID3D12Resource* const resource = slab.resource.Get();
            const auto resourceSize = slab.size;
            auto owner = std::make_shared<IndirectArgumentSliceLeaseOwner>(std::move(allocation->lease));
            return IndirectArgumentSlice{
                .owner = std::move(owner),
                .resource = resource,
                .resourceSize = resourceSize,
                .offset = offset,
            };
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate indirect-argument arena slice retention state.",
            });
        }
    }

    ResourceManager::ResourceManager() : impl_(std::make_unique<Impl>())
    {
    }

    ResourceManager::~ResourceManager()
    {
        if (impl_ == nullptr)
        {
            return;
        }
        if (impl_->uploadArena != nullptr)
        {
            impl_->uploadArena->shutdown();
        }
        for (const auto& arena : impl_->indirectArgumentArenas)
        {
            if (arena != nullptr)
            {
                arena->shutdown();
            }
        }
    }

    namespace
    {

        [[nodiscard]] Status createDepthResolveRootSignature(ID3D12Device& device,
                                                             ComPtr<ID3D12RootSignature>& rootSignature)
        {
            const D3D12_DESCRIPTOR_RANGE range{
                .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                .NumDescriptors = 1,
                .BaseShaderRegister = 0,
                .RegisterSpace = 0,
                .OffsetInDescriptorsFromTableStart = 0,
            };
            const std::array rootParameters{
                D3D12_ROOT_PARAMETER{
                    .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                    .DescriptorTable = {.NumDescriptorRanges = 1, .pDescriptorRanges = &range},
                    .ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL,
                },
                D3D12_ROOT_PARAMETER{
                    .ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    .Constants = {.ShaderRegister = 0, .RegisterSpace = 0, .Num32BitValues = 2},
                    .ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL,
                },
            };
            const D3D12_ROOT_SIGNATURE_DESC rootDesc{
                .NumParameters = static_cast<UINT>(rootParameters.size()),
                .pParameters = rootParameters.data(),
                .NumStaticSamplers = 0,
                .pStaticSamplers = nullptr,
                .Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
                         D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
                         D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
                         D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
                         D3D12_ROOT_SIGNATURE_FLAG_DENY_AMPLIFICATION_SHADER_ROOT_ACCESS |
                         D3D12_ROOT_SIGNATURE_FLAG_DENY_MESH_SHADER_ROOT_ACCESS,
            };
            ComPtr<ID3DBlob> serialized;
            ComPtr<ID3DBlob> diagnostics;
            HRESULT result =
                D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &diagnostics);
            if (FAILED(result))
            {
                return std::unexpected(
                    nativeError(ErrorCode::validationError, result, "D3D12 depth resolve root signature"));
            }
            result = device.CreateRootSignature(
                0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&rootSignature));
            if (FAILED(result))
            {
                return std::unexpected(
                    nativeError(result == E_OUTOFMEMORY ? ErrorCode::outOfMemory : ErrorCode::nativeFailure,
                                result,
                                "ID3D12Device::CreateRootSignature(depth resolve)"));
            }
            return {};
        }

        [[nodiscard]] D3D12_GRAPHICS_PIPELINE_STATE_DESC depthResolvePipelineDesc(
            ID3D12RootSignature* rootSignature) noexcept
        {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{};
            pipeline.pRootSignature = rootSignature;
            pipeline.VS = {g_rachisResolveDepthVertexDxil, sizeof(g_rachisResolveDepthVertexDxil)};
            pipeline.PS = {g_rachisResolveDepthPixelDxil, sizeof(g_rachisResolveDepthPixelDxil)};
            pipeline.BlendState.AlphaToCoverageEnable = FALSE;
            pipeline.BlendState.IndependentBlendEnable = FALSE;
            pipeline.SampleMask = UINT_MAX;
            pipeline.RasterizerState = {
                .FillMode = D3D12_FILL_MODE_SOLID,
                .CullMode = D3D12_CULL_MODE_NONE,
                .FrontCounterClockwise = FALSE,
                .DepthBias = D3D12_DEFAULT_DEPTH_BIAS,
                .DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
                .SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS,
                .DepthClipEnable = TRUE,
                .MultisampleEnable = FALSE,
                .AntialiasedLineEnable = FALSE,
                .ForcedSampleCount = 0,
                .ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
            };
            pipeline.DepthStencilState.DepthEnable = TRUE;
            pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
            pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
            pipeline.DepthStencilState.StencilEnable = FALSE;
            pipeline.DepthStencilState.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
            pipeline.DepthStencilState.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
            pipeline.InputLayout = {nullptr, 0};
            pipeline.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
            pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            pipeline.NumRenderTargets = 0;
            pipeline.DSVFormat = DXGI_FORMAT_D32_FLOAT;
            pipeline.SampleDesc = {1, 0};
            pipeline.NodeMask = 0;
            pipeline.CachedPSO = {};
            pipeline.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
            return pipeline;
        }

        [[nodiscard]] Status createDepthResolvePipeline(ID3D12Device& device,
                                                        ID3D12RootSignature* rootSignature,
                                                        ComPtr<ID3D12PipelineState>& pipeline)
        {
            const auto pipelineDesc = depthResolvePipelineDesc(rootSignature);
            const HRESULT result = device.CreateGraphicsPipelineState(&pipelineDesc, IID_PPV_ARGS(&pipeline));
            if (FAILED(result))
            {
                return std::unexpected(
                    nativeError(result == E_OUTOFMEMORY ? ErrorCode::outOfMemory : ErrorCode::nativeFailure,
                                result,
                                "ID3D12Device::CreateGraphicsPipelineState(depth resolve)"));
            }
            return {};
        }

    } // namespace

    Status ResourceManager::initialize(ID3D12Device& device,
                                       IDXGIAdapter& adapter,
                                       bool enhancedBarriers,
                                       bool variableRateShadingTier2,
                                       bool rayTracing,
                                       bool sparseResources,
                                       bool samplerFeedback)
    {
        static_cast<void>(device.QueryInterface(IID_PPV_ARGS(&impl_->residencyDevice)));
        static_cast<void>(device.QueryInterface(IID_PPV_ARGS(&impl_->residencyControlDevice)));
        static_cast<void>(device.QueryInterface(IID_PPV_ARGS(&impl_->rayTracingDevice)));
        static_cast<void>(device.QueryInterface(IID_PPV_ARGS(&impl_->samplerFeedbackDevice)));
        if (rayTracing && impl_->rayTracingDevice == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The adapter reports inline ray queries but ID3D12Device5 is unavailable.",
            });
        }
        if (samplerFeedback && impl_->samplerFeedbackDevice == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The adapter reports sampler feedback but ID3D12Device8 is unavailable.",
            });
        }
        D3D12MA::ALLOCATOR_DESC allocatorDesc{};
        allocatorDesc.pDevice = &device;
        allocatorDesc.pAdapter = &adapter;
        const HRESULT allocatorResult =
            D3D12MA::CreateAllocator(&allocatorDesc, impl_->allocator.ReleaseAndGetAddressOf());
        if (FAILED(allocatorResult))
        {
            return std::unexpected(
                nativeError(allocatorResult == E_OUTOFMEMORY ? ErrorCode::outOfMemory : ErrorCode::nativeFailure,
                            allocatorResult,
                            "D3D12MA::CreateAllocator"));
        }

        auto rootSignatureCreated = createDepthResolveRootSignature(device, impl_->depthResolveRootSignature);
        if (!rootSignatureCreated)
        {
            return rootSignatureCreated;
        }
        auto pipelineCreated =
            createDepthResolvePipeline(device, impl_->depthResolveRootSignature.Get(), impl_->depthResolvePipeline);
        if (!pipelineCreated)
        {
            return pipelineCreated;
        }
        static_cast<void>(impl_->depthResolveRootSignature->SetName(L"RHI D32 resolve root signature"));
        static_cast<void>(impl_->depthResolvePipeline->SetName(L"RHI D32 resolve pipeline"));
        impl_->device = &device;
        impl_->enhancedBarriers = enhancedBarriers;
        impl_->variableRateShadingTier2 = variableRateShadingTier2;
        impl_->rayTracing = rayTracing;
        impl_->sparseResources = sparseResources;
        impl_->samplerFeedback = samplerFeedback;
        return {};
    }

    std::array<MemoryAllocationStatistics, memorySegmentGroupCount> ResourceManager::memoryAllocationStatistics()
        const noexcept
    {
        std::array<MemoryAllocationStatistics, memorySegmentGroupCount> result{};
        if (impl_->allocator == nullptr)
        {
            return result;
        }

        D3D12MA::Budget local{};
        D3D12MA::Budget nonLocal{};
        impl_->allocator->GetBudget(&local, &nonLocal);
        const std::array<const D3D12MA::Budget*, memorySegmentGroupCount> source{&local, &nonLocal};
        for (std::size_t index = 0; index < source.size(); ++index)
        {
            result[index] = {
                .blockBytes = source[index]->Stats.BlockBytes,
                .allocationBytes = source[index]->Stats.AllocationBytes,
                .blockCount = source[index]->Stats.BlockCount,
                .allocationCount = source[index]->Stats.AllocationCount,
            };
        }
        return result;
    }

    ResourceMemorySnapshot ResourceManager::captureMemorySnapshot() const
    {
        ResourceMemorySnapshot snapshot;
        const auto state = impl_->memoryTracking;
        const std::scoped_lock lock(state->mutex);
        snapshot.entries.reserve(state->entries.size());
        for (auto iterator = state->entries.begin(); iterator != state->entries.end();)
        {
            const auto tracked = iterator->second.lock();
            if (tracked == nullptr)
            {
                iterator = state->entries.erase(iterator);
                continue;
            }
            snapshot.entries.push_back(tracked->entry);
            if (tracked->entry.logicalAlive)
            {
                snapshot.liveAllocationBytes += tracked->entry.allocationBytes;
                ++snapshot.liveAllocationCount;
            }
            else
            {
                snapshot.retainedAllocationBytes += tracked->entry.allocationBytes;
                ++snapshot.retainedAllocationCount;
            }
            ++iterator;
        }
        std::ranges::sort(snapshot.entries, {}, &ResourceMemoryEntry::allocationId);
        return snapshot;
    }

    DescriptorAllocationStatistics ResourceManager::descriptorAllocationStatistics() const noexcept
    {
        return impl_->depthResolveDescriptors.statistics();
    }

    namespace
    {

        [[nodiscard]] constexpr bool supportsResourceHeapAlignment(ResourceHeapClass resourceClass,
                                                                   std::uint64_t alignment) noexcept
        {
            return alignment == defaultResourceHeapAlignment ||
                   (resourceClass == ResourceHeapClass::attachmentTexture &&
                    alignment == multisampleResourceHeapAlignment);
        }

        [[nodiscard]] constexpr bool requiresDeviceLocalMemory(const ResourceHeapDesc& desc) noexcept
        {
            return desc.resourceClass != ResourceHeapClass::buffer ||
                   hasAny(desc.flags, ResourceHeapFlag::allowAliasing) ||
                   hasAny(desc.allocationFlags, ResourceAllocationFlag::manualResidency);
        }

        [[nodiscard]] constexpr bool validSparseBackingHeap(const ResourceHeapDesc& desc) noexcept
        {
            const bool sparseBindableClass = desc.resourceClass == ResourceHeapClass::buffer ||
                                             desc.resourceClass == ResourceHeapClass::nonAttachmentTexture;
            return sparseBindableClass && desc.memoryType == MemoryType::deviceLocal &&
                   !hasAny(desc.flags, ResourceHeapFlag::allowAliasing) &&
                   desc.residencyPriority == ResidencyPriority::automatic &&
                   !hasAny(desc.allocationFlags, ResourceAllocationFlag::manualResidency);
        }

        [[nodiscard]] Status validateResourceHeapDesc(const ResourceHeapDesc& desc, std::uint64_t alignment)
        {
            const bool validSizeAndAlignment = desc.size != 0 &&
                                               supportsResourceHeapAlignment(desc.resourceClass, alignment) &&
                                               desc.size % alignment == 0;
            const bool validEnumerations =
                validMemoryType(desc.memoryType) && validResourceHeapClass(desc.resourceClass) &&
                validResourceHeapFlags(desc.flags) && validResidencyPriority(desc.residencyPriority) &&
                validAllocationFlags(desc.allocationFlags);
            const bool deviceLocal = desc.memoryType == MemoryType::deviceLocal;
            const bool sparseBacking = hasAny(desc.flags, ResourceHeapFlag::sparseBacking);
            if (!validSizeAndAlignment || !validEnumerations || (requiresDeviceLocalMemory(desc) && !deviceLocal) ||
                (sparseBacking && !validSparseBackingHeap(desc)))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A resource heap requires compatible memory, placement, residency, and allocation properties.",
                });
            }
            return {};
        }

        struct ResourceHeapDeviceSupport final
        {
            bool sparseBacking = false;
            bool manualResidency = false;
            bool residencyPriority = false;
        };

        [[nodiscard]] Status requireResourceHeapDeviceSupport(const ResourceHeapDesc& desc,
                                                              const ResourceHeapDeviceSupport& support)
        {
            if (hasAny(desc.flags, ResourceHeapFlag::sparseBacking) && !support.sparseBacking)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Sparse-backing heaps are unavailable on this device.",
                });
            }
            if (hasAny(desc.allocationFlags, ResourceAllocationFlag::manualResidency) && !support.manualResidency)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Manual resource-heap residency is unavailable on this device.",
                });
            }
            if (desc.residencyPriority != ResidencyPriority::automatic && !support.residencyPriority)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Explicit resource-heap residency priorities are unavailable on this device.",
                });
            }
            return {};
        }

        [[nodiscard]] D3D12_HEAP_DESC nativeResourceHeapDesc(const ResourceHeapDesc& desc,
                                                             std::uint64_t alignment) noexcept
        {
            return {
                .SizeInBytes = desc.size,
                .Properties =
                    {
                        .Type = nativeHeapType(desc.memoryType),
                        .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                        .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
                        .CreationNodeMask = 1,
                        .VisibleNodeMask = 1,
                    },
                .Alignment = alignment,
                .Flags = nativeResourceHeapFlags(desc.resourceClass),
            };
        }

        [[nodiscard]] Result<ComPtr<ID3D12Heap>> createNamedResourceHeap(ID3D12Device& device,
                                                                         const D3D12_HEAP_DESC& nativeDesc,
                                                                         std::string_view label)
        {
            ComPtr<ID3D12Heap> heap;
            const HRESULT result = device.CreateHeap(&nativeDesc, IID_PPV_ARGS(&heap));
            if (FAILED(result))
            {
                return std::unexpected(nativeError(classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result)),
                                                   result,
                                                   "ID3D12Device::CreateHeap(resource)"));
            }
            const auto wide = wideLabel(label);
            if (!wide.empty())
            {
                heap->SetName(wide.c_str());
            }
            return heap;
        }

        [[nodiscard]] Status applyResourceHeapResidencyPriority(ID3D12Device1& residencyDevice,
                                                                ID3D12Heap& heap,
                                                                ResidencyPriority residencyPriority)
        {
            const auto nativePriority = nativeResidencyPriority(residencyPriority);
            const auto priority = *nativePriority;
            const std::array<ID3D12Pageable*, 1> pageables{&heap};
            const HRESULT priorityResult = residencyDevice.SetResidencyPriority(1, pageables.data(), &priority);
            if (FAILED(priorityResult))
            {
                return std::unexpected(
                    nativeError(classifyD3D12ResultErrorCode(static_cast<std::int32_t>(priorityResult)),
                                priorityResult,
                                "ID3D12Device1::SetResidencyPriority(resource heap)"));
            }
            return {};
        }

    } // namespace

    Result<ResourceHeapHandle> ResourceManager::createResourceHeap(const ResourceHeapDesc& desc)
    {
        const auto alignment = desc.alignment == 0 ? defaultResourceHeapAlignment : desc.alignment;
        auto valid = validateResourceHeapDesc(desc, alignment);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        const ResourceHeapDeviceSupport deviceSupport{
            .sparseBacking = impl_->sparseResources,
            .manualResidency = impl_->residencyControlDevice != nullptr,
            .residencyPriority = impl_->residencyDevice != nullptr,
        };
        auto supported = requireResourceHeapDeviceSupport(desc, deviceSupport);
        if (!supported)
        {
            return std::unexpected(std::move(supported.error()));
        }

        auto record = std::make_shared<ResourceHeapRecord>();
        record->desc = desc;
        record->desc.alignment = alignment;
        record->label = desc.label;
        record->desc.label = record->label;
        auto allocationIdentity = reserveResourceMemoryIdentity(impl_->memoryTracking);
        if (!allocationIdentity)
        {
            return std::unexpected(std::move(allocationIdentity.error()));
        }
        auto heap = createNamedResourceHeap(*impl_->device, nativeResourceHeapDesc(desc, alignment), desc.label);
        if (!heap)
        {
            return std::unexpected(std::move(heap.error()));
        }
        record->heap = std::move(*heap);
        if (desc.residencyPriority != ResidencyPriority::automatic)
        {
            auto prioritized = applyResourceHeapResidencyPriority(
                *impl_->residencyDevice.Get(), *record->heap.Get(), desc.residencyPriority);
            if (!prioritized)
            {
                return std::unexpected(std::move(prioritized.error()));
            }
        }
        if (hasAny(desc.allocationFlags, ResourceAllocationFlag::manualResidency))
        {
            record->residency = createManualResidencyRecord(record->heap.Get());
        }
        auto trackedMemory = trackResourceMemory(impl_->memoryTracking,
                                                 *allocationIdentity,
                                                 ResourceMemoryKind::resourceHeap,
                                                 desc.memoryType,
                                                 desc.size,
                                                 true,
                                                 false,
                                                 desc.label);
        if (!trackedMemory)
        {
            return std::unexpected(std::move(trackedMemory.error()));
        }
        record->trackedMemory = std::move(*trackedMemory);

        const std::scoped_lock lock(impl_->resourceHeapsMutex);
        return impl_->resourceHeaps.emplace(std::move(record));
    }

    Status ResourceManager::destroyResourceHeap(ResourceHeapHandle heap)
    {
        const std::scoped_lock mappingLock(impl_->sparseMappingsMutex);
        const std::scoped_lock tableLock(impl_->resourceHeapsMutex);
        auto* stored = impl_->resourceHeaps.find(heap);
        if (stored == nullptr)
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
        markResourceMemoryDestroyed(record->trackedMemory);
        static_cast<void>(impl_->resourceHeaps.erase(heap));
        return {};
    }

    Result<ResourceMemoryRequirements> ResourceManager::queryBufferMemoryRequirements(const BufferDesc& desc) const
    {
        auto valid = validateBufferDesc(
            desc, impl_->rayTracing, impl_->residencyDevice != nullptr, impl_->residencyControlDevice != nullptr);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        return queryMemoryRequirements(*impl_->device,
                                       nativeBufferDesc(desc),
                                       hasAny(desc.allocationFlags, ResourceAllocationFlag::manualResidency));
    }

    Result<BufferHandle> ResourceManager::createBuffer(const BufferDesc& desc)
    {
        return createBuffer(desc, false);
    }

    Result<BufferHandle> ResourceManager::createSharedBuffer(const BufferDesc& desc)
    {
        return createBuffer(desc, true);
    }

    namespace
    {

        [[nodiscard]] D3D12MA::ALLOCATION_FLAGS bufferAllocationFlags(const BufferDesc& desc, bool shared) noexcept
        {
            if (hasAny(desc.allocationFlags, ResourceAllocationFlag::manualResidency) || shared)
            {
                return D3D12MA::ALLOCATION_FLAG_COMMITTED;
            }
            if (desc.residencyPriority != ResidencyPriority::automatic)
            {
                return D3D12MA::ALLOCATION_FLAG_STRATEGY_MIN_TIME;
            }
            return D3D12MA::ALLOCATION_FLAG_NONE;
        }

        [[nodiscard]] Status createNativeBufferResource(BufferRecord& record,
                                                        D3D12MA::Allocator& allocator,
                                                        D3D12MA::Pool* residencyPool)
        {
            const auto& desc = record.desc;
            const D3D12MA::ALLOCATION_DESC allocationDesc{
                .Flags = bufferAllocationFlags(desc, record.shared),
                .HeapType = record.shared ? D3D12_HEAP_TYPE_DEFAULT : nativeHeapType(desc.memoryType),
                .ExtraHeapFlags = record.shared ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE,
                .CustomPool = residencyPool,
                .pPrivateData = nullptr,
            };
            const auto resourceDesc = nativeBufferDesc(desc);
            const HRESULT result = allocator.CreateResource(
                &allocationDesc,
                &resourceDesc,
                record.shared ? D3D12_RESOURCE_STATE_COMMON : initialBufferState(desc.memoryType, desc.usage),
                nullptr,
                record.allocation.ReleaseAndGetAddressOf(),
                IID_PPV_ARGS(&record.resource));
            if (FAILED(result))
            {
                return std::unexpected(nativeError(classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result)),
                                                   result,
                                                   "D3D12MA::Allocator::CreateResource(buffer)"));
            }
            record.nativeWidth = resourceDesc.Width;
            return {};
        }

    } // namespace

    [[nodiscard]] Status ResourceManager::Impl::allocateBuffer(BufferRecord& record, D3D12MA::Pool* residencyPool)
    {
        auto allocationIdentity = reserveResourceMemoryIdentity(memoryTracking);
        if (!allocationIdentity)
        {
            return std::unexpected(std::move(allocationIdentity.error()));
        }
        auto created = createNativeBufferResource(record, *allocator.Get(), residencyPool);
        if (!created)
        {
            return created;
        }
        record.device = device;
        record.allocator = allocator;
        record.residencyPool = residencyPool;
        if (hasAny(record.desc.allocationFlags, ResourceAllocationFlag::manualResidency))
        {
            if (record.allocation->GetHeap() != nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::nativeFailure,
                    0,
                    "D3D12MA did not create independently pageable memory for a manual-residency buffer.",
                });
            }
            record.residency = createManualResidencyRecord(record.resource.Get());
        }
        nameAllocatedResource(*record.allocation.Get(), *record.resource.Get(), record.label);
        auto trackedMemory = trackResourceMemory(memoryTracking,
                                                 *allocationIdentity,
                                                 ResourceMemoryKind::buffer,
                                                 record.desc.memoryType,
                                                 *record.allocation.Get(),
                                                 record.shared,
                                                 record.label);
        if (!trackedMemory)
        {
            return std::unexpected(std::move(trackedMemory.error()));
        }
        record.trackedMemory = std::move(*trackedMemory);
        return {};
    }

    Result<BufferHandle> ResourceManager::createBuffer(const BufferDesc& desc, bool shared)
    {
        if (shared)
        {
            auto sharedValid = validateSharedBufferDesc(desc);
            if (!sharedValid)
            {
                return std::unexpected(std::move(sharedValid.error()));
            }
        }
        auto valid = validateBufferDesc(
            desc, impl_->rayTracing, impl_->residencyDevice != nullptr, impl_->residencyControlDevice != nullptr);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        D3D12MA::Pool* residencyPool = nullptr;
        if (!shared)
        {
            auto selectedPool = impl_->residencyPool(bufferPoolClass(desc.memoryType), desc.residencyPriority);
            if (!selectedPool)
            {
                return std::unexpected(std::move(selectedPool.error()));
            }
            residencyPool = *selectedPool;
        }

        auto record = std::make_shared<BufferRecord>();
        record->desc = desc;
        record->label = desc.label;
        record->desc.label = record->label;
        record->shared = shared;
        auto allocated = impl_->allocateBuffer(*record, residencyPool);
        if (!allocated)
        {
            return std::unexpected(std::move(allocated.error()));
        }

        const std::scoped_lock lock(impl_->buffersMutex);
        return impl_->buffers.emplace(std::move(record));
    }

    Result<SharedHandle> ResourceManager::exportSharedBuffer(BufferHandle buffer)
    {
        auto record = impl_->buffer(buffer);
        if (!record)
        {
            return std::unexpected(std::move(record.error()));
        }
        if (!(*record)->shared)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Only buffers created or imported through the shared-buffer API can be exported.",
            });
        }

        HANDLE handle = nullptr;
        const HRESULT result =
            impl_->device->CreateSharedHandle((*record)->resource.Get(), nullptr, GENERIC_ALL, nullptr, &handle);
        if (FAILED(result))
        {
            return std::unexpected(nativeError(classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result)),
                                               result,
                                               "ID3D12Device::CreateSharedHandle(buffer)"));
        }
        return SharedHandle::adopt(SharedHandleView{
            .type = SharedHandleType::windowsNt,
            .value = reinterpret_cast<std::uintptr_t>(handle),
        });
    }

    Result<BufferHandle> ResourceManager::importSharedBuffer(SharedHandleView handle, const BufferDesc& desc)
    {
        if (!handle.isValid() || handle.type != SharedHandleType::windowsNt)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Shared buffer import requires a valid Windows NT handle.",
            });
        }

        ComPtr<ID3D12Resource> resource;
        const HRESULT result =
            impl_->device->OpenSharedHandle(std::bit_cast<HANDLE>(handle.value), IID_PPV_ARGS(&resource));
        if (FAILED(result))
        {
            return std::unexpected(
                nativeError(classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result), ErrorCode::invalidArgument),
                            result,
                            "ID3D12Device::OpenSharedHandle(buffer)"));
        }
        return importBuffer(*resource.Get(), desc, true);
    }

    Result<BufferHandle> ResourceManager::importBuffer(ID3D12Resource& resource, const BufferDesc& desc, bool shared)
    {
        if (shared)
        {
            auto sharedValid = validateSharedBufferDesc(desc);
            if (!sharedValid)
            {
                return std::unexpected(std::move(sharedValid.error()));
            }
        }
        auto valid = validateBufferDesc(
            desc, impl_->rayTracing, impl_->residencyDevice != nullptr, impl_->residencyControlDevice != nullptr);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }

        const auto nativeDesc = resource.GetDesc();
        const auto requiredFlags = bufferFlags(desc.usage);
        if (nativeDesc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
            (nativeDesc.Width != desc.size && nativeDesc.Width != constantBufferViewSize(desc.size)) ||
            nativeDesc.Height != 1 || nativeDesc.DepthOrArraySize != 1 || nativeDesc.MipLevels != 1 ||
            nativeDesc.Format != DXGI_FORMAT_UNKNOWN || nativeDesc.SampleDesc.Count != 1 ||
            nativeDesc.SampleDesc.Quality != 0 || nativeDesc.Layout != D3D12_TEXTURE_LAYOUT_ROW_MAJOR ||
            (nativeDesc.Flags & requiredFlags) != requiredFlags)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The imported buffer description or usage does not match the native D3D12 resource.",
            });
        }

        auto record = std::make_shared<BufferRecord>();
        record->resource = &resource;
        record->desc = desc;
        record->nativeWidth = nativeDesc.Width;
        record->label = desc.label;
        record->desc.label = record->label;
        record->shared = shared;
        const auto label = wideLabel(desc.label);
        if (!label.empty())
        {
            record->resource->SetName(label.c_str());
        }

        const std::scoped_lock lock(impl_->buffersMutex);
        return impl_->buffers.emplace(std::move(record));
    }

    Result<BufferHandle> ResourceManager::createPlacedBuffer(const ResourcePlacement& placement, const BufferDesc& desc)
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
        auto valid = validateBufferDesc(
            desc, impl_->rayTracing, impl_->residencyDevice != nullptr, impl_->residencyControlDevice != nullptr);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        auto heap = impl_->resourceHeap(placement.heap);
        if (!heap)
        {
            return std::unexpected(std::move(heap.error()));
        }
        if ((*heap)->desc.memoryType != desc.memoryType)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A placed buffer's memory type must match its resource heap.",
            });
        }

        const auto nativeDesc = nativeBufferDesc(desc);
        auto requirements = queryMemoryRequirements(*impl_->device, nativeDesc, false);
        if (!requirements)
        {
            return std::unexpected(std::move(requirements.error()));
        }
        auto lease = reservePlacement(*heap, ResourceHeapClass::buffer, placement.offset, *requirements, "buffer");
        if (!lease)
        {
            return std::unexpected(std::move(lease.error()));
        }

        auto record = std::make_shared<BufferRecord>();
        record->placement = std::move(*lease);
        record->residency = (*heap)->residency;
        record->desc = desc;
        record->nativeWidth = nativeDesc.Width;
        record->label = desc.label;
        record->desc.label = record->label;
        const HRESULT result = impl_->device->CreatePlacedResource((*heap)->heap.Get(),
                                                                   placement.offset,
                                                                   &nativeDesc,
                                                                   initialBufferState(desc.memoryType, desc.usage),
                                                                   nullptr,
                                                                   IID_PPV_ARGS(&record->resource));
        if (FAILED(result))
        {
            return std::unexpected(nativeError(classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result)),
                                               result,
                                               "ID3D12Device::CreatePlacedResource(buffer)"));
        }
        const auto label = wideLabel(desc.label);
        if (!label.empty())
        {
            record->resource->SetName(label.c_str());
        }

        const std::scoped_lock lock(impl_->buffersMutex);
        return impl_->buffers.emplace(std::move(record));
    }

    Status ResourceManager::destroyBuffer(BufferHandle buffer)
    {
        const std::scoped_lock mappingLock(impl_->sparseMappingsMutex);
        const std::scoped_lock tableLock(impl_->buffersMutex);
        auto* stored = impl_->buffers.find(buffer);
        if (stored == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The buffer handle is invalid or stale.",
            });
        }

        auto record = *stored;
        const std::scoped_lock mapLock(record->mapMutex);
        if (record->mapped)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A mapped buffer must be unmapped before destruction.",
            });
        }
        record->alive = false;
        markResourceMemoryDestroyed(record->trackedMemory);
        static_cast<void>(impl_->buffers.erase(buffer));
        if (record->sparse)
        {
            impl_->activeSparseResourceCount.fetch_sub(1, std::memory_order_acq_rel);
        }
        return {};
    }

    Result<MappedMemory> ResourceManager::mapBuffer(BufferHandle buffer, std::uint64_t offset, std::uint64_t size)
    {
        auto found = impl_->buffer(buffer);
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        auto& record = **found;
        if (record.desc.memoryType == MemoryType::deviceLocal)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Device-local buffers cannot be mapped directly.",
            });
        }
        if (!validRange(record.desc.size, offset, size))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The requested map range is outside the buffer.",
            });
        }

        const std::scoped_lock lock(record.mapMutex);
        if (!record.alive)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The buffer handle became stale before mapping.",
            });
        }
        if (record.mapped)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The buffer is already mapped.",
            });
        }

        const auto resolvedSize = resolveSize(record.desc.size, offset, size);
        D3D12_RANGE readRange{0, 0};
        if (record.desc.memoryType == MemoryType::readback)
        {
            readRange = {static_cast<SIZE_T>(offset), static_cast<SIZE_T>(offset + resolvedSize)};
        }
        void* data = nullptr;
        const HRESULT result = record.resource->Map(0, &readRange, &data);
        if (FAILED(result))
        {
            return std::unexpected(nativeError(
                classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result)), result, "ID3D12Resource::Map"));
        }
        record.mapped = true;
        record.mappedData = static_cast<std::byte*>(data);
        record.mappedOffset = offset;
        record.mappedSize = resolvedSize;
        return MappedMemory{record.mappedData + offset, resolvedSize};
    }

    Status ResourceManager::unmapBuffer(BufferHandle buffer, std::uint64_t offset, std::uint64_t size)
    {
        auto found = impl_->buffer(buffer);
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        auto& record = **found;
        if (!validRange(record.desc.size, offset, size))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The requested unmap range is outside the buffer.",
            });
        }

        const std::scoped_lock lock(record.mapMutex);
        if (!record.mapped)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The buffer is not mapped.",
            });
        }

        const auto resolvedSize = resolveSize(record.desc.size, offset, size);
        const bool wholeBuffer = offset == 0 && resolvedSize == record.desc.size;
        const auto mappedEnd = record.mappedOffset + record.mappedSize;
        if (!wholeBuffer && (offset < record.mappedOffset || offset + resolvedSize > mappedEnd))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The requested unmap range is not contained in the active mapping.",
            });
        }
        const auto writtenOffset = wholeBuffer ? record.mappedOffset : offset;
        const auto writtenSize = wholeBuffer ? record.mappedSize : resolvedSize;
        D3D12_RANGE writtenRange{0, 0};
        if (record.desc.memoryType == MemoryType::upload)
        {
            writtenRange = {static_cast<SIZE_T>(writtenOffset), static_cast<SIZE_T>(writtenOffset + writtenSize)};
        }
        record.resource->Unmap(0, &writtenRange);
        record.mapped = false;
        record.mappedData = nullptr;
        record.mappedOffset = 0;
        record.mappedSize = 0;
        return {};
    }

    Status ResourceManager::flushMappedMemoryRanges(BufferHandle buffer, std::span<const MappedMemoryRange> ranges)
    {
        auto found = impl_->buffer(buffer);
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        return synchronizeMappedMemoryRanges(**found, ranges, MemoryType::upload, "flush");
    }

    Status ResourceManager::invalidateMappedMemoryRanges(BufferHandle buffer, std::span<const MappedMemoryRange> ranges)
    {
        auto found = impl_->buffer(buffer);
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        return synchronizeMappedMemoryRanges(**found, ranges, MemoryType::readback, "invalidation");
    }

    Status ResourceManager::writeBuffer(BufferHandle buffer, std::uint64_t offset, std::span<const std::byte> data)
    {
        auto found = impl_->buffer(buffer);
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        auto& record = **found;
        if (!validHostRange(record.desc.size, offset, data.size()))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The host buffer write range is empty or outside the buffer.",
            });
        }
        if (record.desc.memoryType != MemoryType::upload)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Host buffer writes require upload memory.",
            });
        }

        const std::scoped_lock lock(record.mapMutex);
        if (!record.alive)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The buffer handle became stale before the host write.",
            });
        }
        if (record.mapped)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A directly mapped buffer cannot be written through the one-shot host API.",
            });
        }

        const D3D12_RANGE noCpuReads{0, 0};
        void* mapped = nullptr;
        const HRESULT result = record.resource->Map(0, &noCpuReads, &mapped);
        if (FAILED(result))
        {
            return std::unexpected(nativeError(classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result)),
                                               result,
                                               "ID3D12Resource::Map(host write)"));
        }
        std::memcpy(static_cast<std::byte*>(mapped) + offset, data.data(), data.size());
        const D3D12_RANGE writtenRange{
            static_cast<SIZE_T>(offset),
            static_cast<SIZE_T>(offset + data.size()),
        };
        record.resource->Unmap(0, &writtenRange);
        return {};
    }

    Status ResourceManager::readBuffer(BufferHandle buffer, std::uint64_t offset, std::span<std::byte> destination)
    {
        auto found = impl_->buffer(buffer);
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        auto& record = **found;
        if (!validHostRange(record.desc.size, offset, destination.size()))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The host buffer read range is empty or outside the buffer.",
            });
        }
        if (record.desc.memoryType != MemoryType::readback)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Host buffer reads require readback memory.",
            });
        }

        const std::scoped_lock lock(record.mapMutex);
        if (!record.alive)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The buffer handle became stale before the host read.",
            });
        }
        if (record.mapped)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A directly mapped buffer cannot be read through the one-shot host API.",
            });
        }

        const D3D12_RANGE readRange{
            static_cast<SIZE_T>(offset),
            static_cast<SIZE_T>(offset + destination.size()),
        };
        void* mapped = nullptr;
        const HRESULT result = record.resource->Map(0, &readRange, &mapped);
        if (FAILED(result))
        {
            return std::unexpected(nativeError(classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result)),
                                               result,
                                               "ID3D12Resource::Map(host read)"));
        }
        std::memcpy(destination.data(), static_cast<std::byte*>(mapped) + offset, destination.size());
        record.resource->Unmap(0, nullptr);
        return {};
    }

    Result<TextureHandle> ResourceManager::createTexture(const TextureDesc& desc)
    {
        return createTexture(desc, false);
    }

    Result<TextureHandle> ResourceManager::createPlacedTexture(const ResourcePlacement& placement,
                                                               const TextureDesc& desc)
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
        const auto info = format::info(desc.format);
        if (!info.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A placed texture requires a supported, non-unknown format.",
            });
        }
        auto valid = validateTextureAllocationDesc(*impl_->device,
                                                   desc,
                                                   *info,
                                                   impl_->variableRateShadingTier2,
                                                   impl_->residencyDevice != nullptr,
                                                   impl_->residencyControlDevice != nullptr,
                                                   false);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        auto heap = impl_->resourceHeap(placement.heap);
        if (!heap)
        {
            return std::unexpected(std::move(heap.error()));
        }
        if ((*heap)->desc.memoryType != MemoryType::deviceLocal)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Placed textures require device-local resource heaps.",
            });
        }

        const auto resourceDesc = nativeTextureDesc(desc, *info);
        auto requirements = queryMemoryRequirements(*impl_->device, resourceDesc, false);
        if (!requirements)
        {
            return std::unexpected(std::move(requirements.error()));
        }
        auto lease =
            reservePlacement(*heap, textureResourceHeapClass(desc.usage), placement.offset, *requirements, "texture");
        if (!lease)
        {
            return std::unexpected(std::move(lease.error()));
        }

        auto record = std::make_shared<TextureRecord>();
        record->placement = std::move(*lease);
        record->residency = (*heap)->residency;
        record->desc = desc;
        record->label = desc.label;
        record->desc.label = record->label;
        record->viewFormats.assign(desc.viewFormats.begin(), desc.viewFormats.end());
        record->desc.viewFormats = record->viewFormats;

        const auto optimizedClearValue = nativeOptimizedClearValue(desc, *info);
        const HRESULT result =
            impl_->device->CreatePlacedResource((*heap)->heap.Get(),
                                                placement.offset,
                                                &resourceDesc,
                                                D3D12_RESOURCE_STATE_COMMON,
                                                optimizedClearValue ? &*optimizedClearValue : nullptr,
                                                IID_PPV_ARGS(&record->resource));
        if (FAILED(result))
        {
            return std::unexpected(nativeError(classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result)),
                                               result,
                                               "ID3D12Device::CreatePlacedResource(texture)"));
        }
        const auto label = wideLabel(desc.label);
        if (!label.empty())
        {
            record->resource->SetName(label.c_str());
        }
        auto tracked = trackLegacyTextureStates(*record, impl_->enhancedBarriers);
        if (!tracked)
        {
            return std::unexpected(std::move(tracked.error()));
        }

        const std::scoped_lock lock(impl_->texturesMutex);
        return impl_->textures.emplace(std::move(record));
    }

    Result<TextureHandle> ResourceManager::createSharedTexture(const TextureDesc& desc)
    {
        return createTexture(desc, true);
    }

    Result<ResourceMemoryRequirements> ResourceManager::queryTextureMemoryRequirements(const TextureDesc& desc) const
    {
        const auto info = format::info(desc.format);
        if (!info.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A texture requires a supported, non-unknown format.",
            });
        }
        auto valid = validateTextureAllocationDesc(*impl_->device,
                                                   desc,
                                                   *info,
                                                   impl_->variableRateShadingTier2,
                                                   impl_->residencyDevice != nullptr,
                                                   impl_->residencyControlDevice != nullptr,
                                                   false);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        return queryMemoryRequirements(*impl_->device,
                                       nativeTextureDesc(desc, *info),
                                       hasAny(desc.allocationFlags, ResourceAllocationFlag::manualResidency));
    }

    namespace
    {

        [[nodiscard]] Status createNativeTextureResource(TextureRecord& record,
                                                         const format::Info& info,
                                                         D3D12MA::Allocator& allocator,
                                                         D3D12MA::Pool* residencyPool)
        {
            const bool manualResidency = hasAny(record.desc.allocationFlags, ResourceAllocationFlag::manualResidency);
            const D3D12MA::ALLOCATION_DESC allocationDesc{
                .Flags = manualResidency || record.shared ? D3D12MA::ALLOCATION_FLAG_COMMITTED
                                                          : D3D12MA::ALLOCATION_FLAG_NONE,
                .HeapType = D3D12_HEAP_TYPE_DEFAULT,
                .ExtraHeapFlags = record.shared ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE,
                .CustomPool = residencyPool,
                .pPrivateData = nullptr,
            };
            const auto resourceDesc = nativeTextureDesc(record.desc, info);
            const auto optimizedClearValue = nativeOptimizedClearValue(record.desc, info);
            const HRESULT result = allocator.CreateResource(&allocationDesc,
                                                            &resourceDesc,
                                                            D3D12_RESOURCE_STATE_COMMON,
                                                            optimizedClearValue ? &*optimizedClearValue : nullptr,
                                                            record.allocation.ReleaseAndGetAddressOf(),
                                                            IID_PPV_ARGS(&record.resource));
            if (FAILED(result))
            {
                return std::unexpected(nativeError(classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result)),
                                                   result,
                                                   "D3D12MA::Allocator::CreateResource(texture)"));
            }
            return {};
        }

    } // namespace

    [[nodiscard]] Status ResourceManager::Impl::allocateTexture(TextureRecord& record,
                                                                const format::Info& info,
                                                                D3D12MA::Pool* residencyPool)
    {
        auto allocationIdentity = reserveResourceMemoryIdentity(memoryTracking);
        if (!allocationIdentity)
        {
            return std::unexpected(std::move(allocationIdentity.error()));
        }
        auto created = createNativeTextureResource(record, info, *allocator.Get(), residencyPool);
        if (!created)
        {
            return created;
        }
        record.device = device;
        record.allocator = allocator;
        record.residencyPool = residencyPool;
        if (hasAny(record.desc.allocationFlags, ResourceAllocationFlag::manualResidency))
        {
            if (record.allocation->GetHeap() != nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::nativeFailure,
                    0,
                    "D3D12MA did not create independently pageable memory for a manual-residency texture.",
                });
            }
            record.residency = createManualResidencyRecord(record.resource.Get());
        }
        nameAllocatedResource(*record.allocation.Get(), *record.resource.Get(), record.label);
        auto trackedMemory = trackResourceMemory(memoryTracking,
                                                 *allocationIdentity,
                                                 ResourceMemoryKind::texture,
                                                 MemoryType::deviceLocal,
                                                 *record.allocation.Get(),
                                                 record.shared,
                                                 record.label);
        if (!trackedMemory)
        {
            return std::unexpected(std::move(trackedMemory.error()));
        }
        record.trackedMemory = std::move(*trackedMemory);
        return {};
    }

    Result<TextureHandle> ResourceManager::createTexture(const TextureDesc& desc, bool shared)
    {
        const auto info = format::info(desc.format);
        if (!info.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A texture requires a supported, non-unknown format.",
            });
        }
        auto valid = validateTextureAllocationDesc(*impl_->device,
                                                   desc,
                                                   *info,
                                                   impl_->variableRateShadingTier2,
                                                   impl_->residencyDevice != nullptr,
                                                   impl_->residencyControlDevice != nullptr,
                                                   shared);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        D3D12MA::Pool* residencyPool = nullptr;
        if (!shared)
        {
            auto selectedPool = impl_->residencyPool(texturePoolClass(desc.usage), desc.residencyPriority);
            if (!selectedPool)
            {
                return std::unexpected(std::move(selectedPool.error()));
            }
            residencyPool = *selectedPool;
        }

        auto record = std::make_shared<TextureRecord>();
        record->desc = desc;
        record->label = desc.label;
        record->desc.label = record->label;
        record->viewFormats.assign(desc.viewFormats.begin(), desc.viewFormats.end());
        record->desc.viewFormats = record->viewFormats;
        record->shared = shared;
        auto allocated = impl_->allocateTexture(*record, *info, residencyPool);
        if (!allocated)
        {
            return std::unexpected(std::move(allocated.error()));
        }
        auto tracked = trackLegacyTextureStates(*record, impl_->enhancedBarriers);
        if (!tracked)
        {
            return std::unexpected(std::move(tracked.error()));
        }

        const std::scoped_lock lock(impl_->texturesMutex);
        return impl_->textures.emplace(std::move(record));
    }

    Result<TextureHandle> ResourceManager::importTexture(ID3D12Resource& resource, const TextureDesc& desc)
    {
        return importTexture(resource, desc, true, false);
    }

    Result<TextureHandle> ResourceManager::importSwapchainTexture(ID3D12Resource& resource,
                                                                  const TextureDesc& desc,
                                                                  std::shared_ptr<SwapchainImageLeaseState> lease)
    {
        if (lease == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A swapchain texture requires a live acquisition lease state.",
            });
        }
        return importTexture(resource, desc, true, false, std::move(lease));
    }

    Result<SharedHandle> ResourceManager::exportSharedTexture(TextureHandle texture)
    {
        auto record = impl_->texture(texture);
        if (!record)
        {
            return std::unexpected(std::move(record.error()));
        }
        if (!(*record)->shared)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Only textures created or imported through the shared-texture API can be exported.",
            });
        }

        HANDLE handle = nullptr;
        const HRESULT result =
            impl_->device->CreateSharedHandle((*record)->resource.Get(), nullptr, GENERIC_ALL, nullptr, &handle);
        if (FAILED(result))
        {
            return std::unexpected(nativeError(classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result)),
                                               result,
                                               "ID3D12Device::CreateSharedHandle(texture)"));
        }
        return SharedHandle::adopt(SharedHandleView{
            .type = SharedHandleType::windowsNt,
            .value = reinterpret_cast<std::uintptr_t>(handle),
        });
    }

    Result<TextureHandle> ResourceManager::importSharedTexture(SharedHandleView handle, const TextureDesc& desc)
    {
        if (!handle.isValid() || handle.type != SharedHandleType::windowsNt)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Shared texture import requires a valid Windows NT handle.",
            });
        }

        ComPtr<ID3D12Resource> resource;
        const HRESULT result =
            impl_->device->OpenSharedHandle(std::bit_cast<HANDLE>(handle.value), IID_PPV_ARGS(&resource));
        if (FAILED(result))
        {
            return std::unexpected(
                nativeError(classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result), ErrorCode::invalidArgument),
                            result,
                            "ID3D12Device::OpenSharedHandle(texture)"));
        }
        return importTexture(*resource.Get(), desc, false, true);
    }

    Result<TextureHandle> ResourceManager::importTexture(ID3D12Resource& resource,
                                                         const TextureDesc& desc,
                                                         bool externallyOwned,
                                                         bool shared,
                                                         std::shared_ptr<SwapchainImageLeaseState> swapchainLease)
    {
        if (desc.residencyPriority != ResidencyPriority::automatic ||
            desc.allocationFlags != ResourceAllocationFlag::none ||
            desc.optimizedClearValue.kind != OptimizedClearValueKind::automatic)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Imported textures cannot override residency policy or optimized clear metadata for externally owned "
                "memory.",
            });
        }
        const auto info = format::info(desc.format);
        if (!info.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "An imported texture requires a supported, non-unknown format.",
            });
        }
        auto valid = validateTextureDesc(*impl_->device, desc, *info, impl_->variableRateShadingTier2);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        const auto nativeDesc = resource.GetDesc();
        const auto nativeFlags = nativeDesc.Flags;
        const bool usageFitsNativeFlags = (!hasAny(desc.usage, TextureUsage::storage) ||
                                           (nativeFlags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0) &&
                                          (!hasAny(desc.usage, TextureUsage::renderTarget) ||
                                           (nativeFlags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) != 0) &&
                                          (!hasAny(desc.usage, TextureUsage::depthStencil) ||
                                           (nativeFlags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0) &&
                                          (!hasAny(desc.usage, TextureUsage::sampled) ||
                                           (nativeFlags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) == 0);

        auto expectedNativeFormat = nativeResourceFormat(desc, *info);
        if (swapchainLease != nullptr && desc.format == Format::bgra8UnormSrgb)
        {
            expectedNativeFormat = DXGI_FORMAT_B8G8R8A8_UNORM;
        }
        else if (swapchainLease != nullptr && desc.format == Format::rgba8UnormSrgb)
        {
            expectedNativeFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
        }
        if (nativeDesc.Dimension != nativeDimension(desc.dimension) || nativeDesc.Width != desc.width ||
            nativeDesc.Height != desc.height || nativeDesc.DepthOrArraySize != desc.depthOrArrayLayers ||
            nativeDesc.MipLevels != desc.mipLevels || nativeDesc.Format != expectedNativeFormat ||
            nativeDesc.SampleDesc.Count != desc.sampleCount || nativeDesc.SampleDesc.Quality != 0 ||
            !usageFitsNativeFlags)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The imported texture description or usage does not match the native D3D12 resource.",
            });
        }

        auto record = std::make_shared<TextureRecord>();
        record->resource = &resource;
        record->desc = desc;
        record->label = desc.label;
        record->desc.label = record->label;
        record->viewFormats.assign(desc.viewFormats.begin(), desc.viewFormats.end());
        record->desc.viewFormats = record->viewFormats;
        record->externallyOwned = externallyOwned;
        record->shared = shared;
        record->swapchainLease = std::move(swapchainLease);
        const auto label = wideLabel(desc.label);
        if (!label.empty())
        {
            record->resource->SetName(label.c_str());
        }
        auto tracked = trackLegacyTextureStates(*record, impl_->enhancedBarriers);
        if (!tracked)
        {
            return std::unexpected(std::move(tracked.error()));
        }

        const std::scoped_lock lock(impl_->texturesMutex);
        return impl_->textures.emplace(std::move(record));
    }

    Status ResourceManager::destroyTexture(TextureHandle texture, bool externallyOwned)
    {
        const std::scoped_lock mappingLock(impl_->sparseMappingsMutex);
        const std::scoped_lock tableLock(impl_->texturesMutex);
        const auto* record = impl_->textures.find(texture);
        if (record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The texture handle is invalid or stale.",
            });
        }
        if ((*record)->externallyOwned != externallyOwned)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                externallyOwned ? "The texture is owned by the application."
                                : "The texture is borrowed from a swapchain.",
            });
        }
        markResourceMemoryDestroyed((*record)->trackedMemory);
        const bool wasSparse = (*record)->sparse;
        static_cast<void>(impl_->textures.erase(texture));
        if (wasSparse)
        {
            impl_->activeSparseResourceCount.fetch_sub(1, std::memory_order_acq_rel);
        }
        return {};
    }

    Result<ResidencyState> ResourceManager::queryResidency(const QueueManager& queues, ResourceHeapHandle heap) const
    {
        auto found = impl_->resourceHeap(heap);
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        return queues.residencyState((*found)->residency);
    }

    Result<ResidencyState> ResourceManager::queryResidency(const QueueManager& queues, BufferHandle buffer) const
    {
        auto found = impl_->buffer(buffer);
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        return queues.residencyState((*found)->residency);
    }

    Result<ResidencyState> ResourceManager::queryResidency(const QueueManager& queues, TextureHandle texture) const
    {
        auto found = impl_->texture(texture);
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        return queues.residencyState((*found)->residency);
    }

    namespace
    {

        [[nodiscard]] constexpr bool isEmptyResidencySet(const ResidencySet& resources) noexcept
        {
            return resources.heaps.empty() && resources.buffers.empty() && resources.textures.empty();
        }

        [[nodiscard]] constexpr bool exceedsPageableObjectLimit(const ResidencySet& resources) noexcept
        {
            constexpr auto maxPageableCount = static_cast<std::size_t>(std::numeric_limits<UINT>::max());
            return resources.heaps.size() > maxPageableCount ||
                   resources.buffers.size() > maxPageableCount - resources.heaps.size() ||
                   resources.textures.size() > maxPageableCount - resources.heaps.size() - resources.buffers.size();
        }

        struct ResidencyTransitionTargets final
        {
            std::vector<std::shared_ptr<ResidencyRecord>> residencies;
            std::vector<ID3D12Pageable*> pageables;
        };

        [[nodiscard]] ResidencyTransitionTargets reserveResidencyTransitionTargets(std::size_t resourceCount)
        {
            ResidencyTransitionTargets targets;
            targets.residencies.reserve(resourceCount);
            targets.pageables.reserve(resourceCount);
            return targets;
        }

        [[nodiscard]] Status appendResidencyTarget(ResidencyTransitionTargets& targets,
                                                   const std::shared_ptr<ResidencyRecord>& residency)
        {
            if (residency == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Explicit residency transitions require manually managed physical allocations.",
                });
            }
            if (std::ranges::any_of(targets.residencies, [&](const auto& existing) { return existing == residency; }))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A residency set cannot contain the same physical allocation more than once.",
                });
            }
            targets.residencies.push_back(residency);
            targets.pageables.push_back(residency->pageable);
            return {};
        }

        template<typename Records, typename Visitor>
        [[nodiscard]] Status forEachResidencyOwner(const Records& records,
                                                   const ResidencySet& resources,
                                                   Visitor&& visit)
        {
            const auto visitEach = [&](const auto& handles, const auto& findRecord) -> Status {
                for (const auto handle : handles)
                {
                    auto found = findRecord(handle);
                    if (!found)
                    {
                        return std::unexpected(std::move(found.error()));
                    }
                    auto visited = visit(*found, (*found)->residency);
                    if (!visited)
                    {
                        return visited;
                    }
                }
                return {};
            };
            auto heapsVisited =
                visitEach(resources.heaps, [&](ResourceHeapHandle handle) { return records.resourceHeap(handle); });
            if (!heapsVisited)
            {
                return heapsVisited;
            }
            auto buffersVisited =
                visitEach(resources.buffers, [&](BufferHandle handle) { return records.buffer(handle); });
            if (!buffersVisited)
            {
                return buffersVisited;
            }
            return visitEach(resources.textures, [&](TextureHandle handle) { return records.texture(handle); });
        }

        [[nodiscard]] Status refreshResidencyStates(const QueueManager& queues,
                                                    std::span<const std::shared_ptr<ResidencyRecord>> residencies)
        {
            for (const auto& residency : residencies)
            {
                auto state = queues.residencyState(residency);
                if (!state)
                {
                    return std::unexpected(std::move(state.error()));
                }
            }
            return {};
        }

        [[nodiscard]] std::vector<std::shared_ptr<ResidencyRecord>> residenciesInLockOrder(
            const std::vector<std::shared_ptr<ResidencyRecord>>& residencies)
        {
            auto ordered = residencies;
            std::ranges::sort(ordered, {}, [](const auto& residency) { return residency.get(); });
            return ordered;
        }

        [[nodiscard]] std::vector<std::unique_lock<std::mutex>> lockResidencies(
            std::span<const std::shared_ptr<ResidencyRecord>> ordered)
        {
            std::vector<std::unique_lock<std::mutex>> locks;
            locks.reserve(ordered.size());
            for (const auto& residency : ordered)
            {
                locks.emplace_back(residency->mutex);
            }
            return locks;
        }

        [[nodiscard]] Status beginMakeResidentTransition(std::span<const std::shared_ptr<ResidencyRecord>> ordered)
        {
            const auto locks = lockResidencies(ordered);
            for (const auto& residency : ordered)
            {
                if (residency->state != ResidencyState::evicted || residency->useCount != 0 ||
                    residency->pageable == nullptr)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "Make-resident requires independently pageable resources in the evicted and unused state.",
                    });
                }
            }
            for (const auto& residency : ordered)
            {
                residency->state = ResidencyState::pending;
                residency->readyAfter = {};
                residency->transitionUnpublishable = false;
            }
            return {};
        }

        [[nodiscard]] bool isPendingWithoutPublication(const ResidencyRecord& residency)
        {
            return residency.state == ResidencyState::pending &&
                   std::ranges::none_of(residency.readyAfter, [](std::uint64_t value) { return value != 0; });
        }

        void revertUnscheduledMakeResident(std::span<const std::shared_ptr<ResidencyRecord>> ordered)
        {
            for (const auto& residency : ordered)
            {
                const std::scoped_lock lock(residency->mutex);
                if (isPendingWithoutPublication(*residency))
                {
                    residency->state = ResidencyState::evicted;
                    residency->transitionUnpublishable = false;
                }
            }
        }

        void markMakeResidentUnpublishable(std::span<const std::shared_ptr<ResidencyRecord>> ordered)
        {
            for (const auto& residency : ordered)
            {
                const std::scoped_lock lock(residency->mutex);
                if (isPendingWithoutPublication(*residency))
                {
                    residency->transitionUnpublishable = true;
                }
            }
        }

        void recordMakeResidentPublication(std::span<const std::shared_ptr<ResidencyRecord>> ordered,
                                           TimelinePoint publication)
        {
            for (const auto& residency : ordered)
            {
                const std::scoped_lock lock(residency->mutex);
                residency->readyAfter[static_cast<std::size_t>(publication.queueType())] = publication.value();
            }
        }

        [[nodiscard]] Result<TimelinePoint> applyMakeResidentSubmission(
            std::span<const std::shared_ptr<ResidencyRecord>> ordered,
            Result<QueueManager::ResidencySubmission>&& scheduled)
        {
            if (!scheduled)
            {
                revertUnscheduledMakeResident(ordered);
                return std::unexpected(std::move(scheduled.error()));
            }
            if (scheduled->failure.has_value())
            {
                markMakeResidentUnpublishable(ordered);
                return std::unexpected(std::move(*scheduled->failure));
            }
            if (!scheduled->publication.isValid())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A successful make-resident submission did not publish a queue timeline point.",
                });
            }
            recordMakeResidentPublication(ordered, scheduled->publication);
            return scheduled->publication;
        }

    } // namespace

    Result<TimelinePoint> ResourceManager::enqueueMakeResident(QueueManager& queues,
                                                               QueueType queueType,
                                                               const ResidencySet& resources,
                                                               ResidencyBudgetPolicy budgetPolicy)
    {
        if (isEmptyResidencySet(resources))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A make-resident request requires at least one resource.",
            });
        }
        if (exceedsPageableObjectLimit(resources))
        {
            return std::unexpected(Error{
                ErrorCode::resourceExhausted,
                0,
                "The make-resident resource count exceeds the D3D12 pageable-object limit.",
            });
        }

        const auto resourceCount = resources.heaps.size() + resources.buffers.size() + resources.textures.size();
        std::vector<std::shared_ptr<void>> owners;
        owners.reserve(resourceCount);
        auto targets = reserveResidencyTransitionTargets(resourceCount);
        auto collected = forEachResidencyOwner(
            *impl_,
            resources,
            [&](std::shared_ptr<void> owner, const std::shared_ptr<ResidencyRecord>& residency) -> Status {
                auto appended = appendResidencyTarget(targets, residency);
                if (appended)
                {
                    owners.push_back(std::move(owner));
                }
                return appended;
            });
        if (!collected)
        {
            return std::unexpected(std::move(collected.error()));
        }
        auto refreshed = refreshResidencyStates(queues, targets.residencies);
        if (!refreshed)
        {
            return std::unexpected(std::move(refreshed.error()));
        }

        const auto ordered = residenciesInLockOrder(targets.residencies);
        auto begun = beginMakeResidentTransition(ordered);
        if (!begun)
        {
            return std::unexpected(std::move(begun.error()));
        }
        auto scheduled = queues.enqueueMakeResident(queueType, targets.pageables, owners, budgetPolicy);
        return applyMakeResidentSubmission(ordered, std::move(scheduled));
    }

    namespace
    {

        [[nodiscard]] Status requireResidentAndUnused(std::span<const std::shared_ptr<ResidencyRecord>> ordered)
        {
            for (const auto& residency : ordered)
            {
                if (residency->state != ResidencyState::resident || residency->useCount != 0 ||
                    residency->pageable == nullptr)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "Eviction requires independently pageable resources in the resident and unused state.",
                    });
                }
            }
            return {};
        }

        void markEvicted(std::span<const std::shared_ptr<ResidencyRecord>> ordered)
        {
            for (const auto& residency : ordered)
            {
                residency->state = ResidencyState::evicted;
                residency->readyAfter = {};
            }
        }

    } // namespace

    Status ResourceManager::evict(const QueueManager& queues, const ResidencySet& resources)
    {
        if (isEmptyResidencySet(resources))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "An eviction request requires at least one resource.",
            });
        }
        if (exceedsPageableObjectLimit(resources))
        {
            return std::unexpected(Error{
                ErrorCode::resourceExhausted,
                0,
                "The eviction resource count exceeds the D3D12 pageable-object limit.",
            });
        }

        const auto resourceCount = resources.heaps.size() + resources.buffers.size() + resources.textures.size();
        auto targets = reserveResidencyTransitionTargets(resourceCount);
        auto collected = forEachResidencyOwner(
            *impl_, resources, [&](const auto&, const std::shared_ptr<ResidencyRecord>& residency) {
                return appendResidencyTarget(targets, residency);
            });
        if (!collected)
        {
            return collected;
        }
        queues.collectGarbage();
        auto refreshed = refreshResidencyStates(queues, targets.residencies);
        if (!refreshed)
        {
            return refreshed;
        }

        const auto ordered = residenciesInLockOrder(targets.residencies);
        const auto locks = lockResidencies(ordered);
        auto evictable = requireResidentAndUnused(ordered);
        if (!evictable)
        {
            return evictable;
        }
        const HRESULT result =
            impl_->device->Evict(static_cast<UINT>(targets.pageables.size()), targets.pageables.data());
        if (FAILED(result))
        {
            return std::unexpected(nativeError(
                classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result)), result, "ID3D12Device::Evict"));
        }
        markEvicted(ordered);
        return {};
    }

    Result<TextureViewSource> ResourceManager::textureViewSource(TextureHandle texture)
    {
        auto found = impl_->texture(texture);
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        const auto info = format::info((*found)->desc.format);
        if (!info)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A live texture contains an unknown stored format.",
            });
        }
        return TextureViewSource{
            .owner = *found,
            .residency = (*found)->residency,
            .swapchainLease = (*found)->swapchainLease,
            .resource = (*found)->resource.Get(),
            .format = info->view,
            .desc = (*found)->desc,
            .sparse = (*found)->sparse,
        };
    }

    Result<SamplerFeedbackBindingSource> ResourceManager::samplerFeedbackBindingSource(
        SamplerFeedbackMapHandle feedbackMap, SamplerFeedbackMapType expectedType)
    {
        auto record = impl_->samplerFeedbackMap(feedbackMap);
        if (!record)
        {
            return std::unexpected(std::move(record.error()));
        }
        if ((*record)->type != expectedType)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The sampler-feedback map type does not match the binding layout entry.",
            });
        }
        return SamplerFeedbackBindingSource{
            .owner = *record,
            .targetedResidency = (*record)->targetedTexture->residency,
            .targetedResource = (*record)->targetedTexture->resource.Get(),
            .feedbackResource = (*record)->resource.Get(),
        };
    }

    Result<BufferBindingSource> ResourceManager::bufferBindingSource(BufferHandle buffer)
    {
        auto found = impl_->buffer(buffer);
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        return BufferBindingSource{
            .owner = *found,
            .residency = (*found)->residency,
            .resource = (*found)->resource.Get(),
            .desc = (*found)->desc,
            .nativeWidth = (*found)->nativeWidth,
            .sparse = (*found)->sparse,
        };
    }

} // namespace rachis::d3d12
