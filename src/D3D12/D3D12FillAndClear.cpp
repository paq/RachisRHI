#include "D3D12/D3D12ResourceInternal.h"

#include "D3D12/D3D12Binding.h"
#include "D3D12/D3D12View.h"
#include "Internal/FillAndClearValidation.h"

#include <bit>

namespace rachis::d3d12
{

    using namespace resource_detail;

    namespace
    {

        inline constexpr std::uint64_t maxTypedBufferClearElements = std::uint64_t{1}
                                                                     << D3D12_REQ_BUFFER_RESOURCE_TEXEL_COUNT_2_TO_EXP;

        [[nodiscard]] Status retainClearDependencies(const QueueManager& queues,
                                                     detail::BackendCommandBuffer& commandBuffer,
                                                     std::span<const QueueManager::ResourceRetention> resources,
                                                     std::span<const std::shared_ptr<SwapchainImageLeaseState>> leases,
                                                     const std::optional<SparseBindingSnapshot>& sparseSnapshot)
        {
            std::vector<QueueManager::ResourceRetention> dependencies(resources.begin(), resources.end());
            std::array<TimelinePoint, 1> sparseDependencies{};
            std::size_t sparseDependencyCount = 0;
            if (sparseSnapshot.has_value())
            {
                dependencies.push_back(QueueManager::ResourceRetention{
                    .owner = sparseSnapshot->owner,
                    .internal = true,
                });
                if (sparseSnapshot->publication.isValid())
                {
                    sparseDependencies[sparseDependencyCount++] = sparseSnapshot->publication;
                }
            }
            return queues.retainCommandDependencies(
                commandBuffer, dependencies, leases, std::span{sparseDependencies}.first(sparseDependencyCount));
        }

        [[nodiscard]] std::uint64_t typedBufferClearElementLimit() noexcept
        {
            const auto forced = test::bufferFillChunkElementLimit().load(std::memory_order_relaxed);
            return forced == 0 ? maxTypedBufferClearElements : (std::min)(forced, maxTypedBufferClearElements);
        }

        void recordFillChunkBarrier(ID3D12GraphicsCommandList& commandList,
                                    ID3D12GraphicsCommandList7* enhanced,
                                    ID3D12Resource& resource) noexcept
        {
            if (enhanced != nullptr)
            {
                const D3D12_BUFFER_BARRIER barrier{
                    .SyncBefore = D3D12_BARRIER_SYNC_CLEAR_UNORDERED_ACCESS_VIEW,
                    .SyncAfter = D3D12_BARRIER_SYNC_CLEAR_UNORDERED_ACCESS_VIEW,
                    .AccessBefore = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    .AccessAfter = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    .pResource = &resource,
                    .Offset = 0,
                    .Size = UINT64_MAX,
                };
                const D3D12_BARRIER_GROUP group{
                    .Type = D3D12_BARRIER_TYPE_BUFFER,
                    .NumBarriers = 1,
                    .pBufferBarriers = &barrier,
                };
                enhanced->Barrier(1, &group);
            }
            else
            {
                const D3D12_RESOURCE_BARRIER barrier{
                    .Type = D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    .UAV = {.pResource = &resource},
                };
                commandList.ResourceBarrier(1, &barrier);
            }
            test::noteBufferFillChunkBarrier();
        }

        void bindClearDescriptorHeaps(ID3D12GraphicsCommandList& commandList, const BindingManager& bindings)
        {
            const auto heaps = bindings.nativeDescriptorHeaps();
            ID3D12DescriptorHeap* const nativeHeaps[]{heaps.resources, heaps.samplers};
            commandList.SetDescriptorHeaps(static_cast<UINT>(std::size(nativeHeaps)), nativeHeaps);
        }

        [[nodiscard]] TextureSubresourceRange storageViewSubresources(const TextureDesc& texture,
                                                                      const TextureViewDesc& view) noexcept
        {
            const bool layered = view.dimension == TextureViewDimension::texture1DArray ||
                                 view.dimension == TextureViewDimension::texture2DArray;
            return TextureSubresourceRange{
                .firstMipLevel = view.firstMipLevel,
                .mipLevelCount = 1,
                .firstArrayLayer = texture.dimension == TextureDimension::texture3D ? 0U : view.firstArrayLayer,
                .arrayLayerCount = layered ? view.arrayLayerCount : 1U,
                .aspect = TextureAspect::color,
            };
        }

    } // namespace

    Status ResourceManager::fillBuffer(QueueManager& queues,
                                       BindingManager& bindings,
                                       detail::BackendCommandBuffer& commandBuffer,
                                       BufferHandle buffer,
                                       std::uint64_t offset,
                                       std::uint64_t size,
                                       std::uint32_t value)
    {
        auto source = bufferBindingSource(buffer);
        if (!source)
        {
            return std::unexpected(std::move(source.error()));
        }
        auto range = detail::resolveBufferFillRange(source->desc, offset, size);
        if (!range)
        {
            return std::unexpected(std::move(range.error()));
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto sparseSnapshot = captureSparseBufferBinding(*source);
        if (!sparseSnapshot)
        {
            return std::unexpected(std::move(sparseSnapshot.error()));
        }

        const std::uint64_t firstElement = offset / detail::bufferFillAlignment;
        const std::uint64_t elementCount = *range / detail::bufferFillAlignment;
        const std::uint64_t chunkElements = typedBufferClearElementLimit();
        ID3D12GraphicsCommandList7* enhanced = nullptr;
        if (impl_->enhancedBarriers && elementCount > chunkElements)
        {
            auto prepared = queues.commandList7(commandBuffer);
            if (!prepared)
            {
                return std::unexpected(std::move(prepared.error()));
            }
            enhanced = *prepared;
        }
        std::vector<ClearResourceDescriptorPair> descriptors;
        std::vector<QueueManager::ResourceRetention> retentions{
            QueueManager::ResourceRetention{
                .owner = source->owner,
                .residency = source->residency,
            },
        };
        for (std::uint64_t cleared = 0; cleared < elementCount; cleared += chunkElements)
        {
            auto pair = bindings.allocateClearResourceDescriptors();
            if (!pair)
            {
                return std::unexpected(std::move(pair.error()));
            }
            D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
            view.Format = DXGI_FORMAT_R32_UINT;
            view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            view.Buffer.FirstElement = firstElement + cleared;
            view.Buffer.NumElements = static_cast<UINT>(std::min(elementCount - cleared, chunkElements));
            impl_->device->CreateUnorderedAccessView(
                source->resource, nullptr, &view, pair->shaderVisible->cpuHandle());
            impl_->device->CreateUnorderedAccessView(source->resource, nullptr, &view, pair->cpuVisible->handle());
            retentions.push_back(QueueManager::ResourceRetention{.owner = pair->shaderVisible, .internal = true});
            retentions.push_back(QueueManager::ResourceRetention{.owner = pair->cpuVisible, .internal = true});
            descriptors.push_back(std::move(*pair));
        }
        auto retained = retainClearDependencies(queues, commandBuffer, retentions, {}, *sparseSnapshot);
        if (!retained)
        {
            return retained;
        }

        bindClearDescriptorHeaps(**commandList, bindings);
        const std::array values{value, value, value, value};
        for (const auto& pair : descriptors)
        {
            if (&pair != &descriptors.front())
            {
                recordFillChunkBarrier(**commandList, enhanced, *source->resource);
            }
            (*commandList)
                ->ClearUnorderedAccessViewUint(pair.shaderVisible->gpuHandle(),
                                               pair.cpuVisible->handle(),
                                               source->resource,
                                               values.data(),
                                               0,
                                               nullptr);
        }
        test::noteResourceCommandRecorded();
        return {};
    }

    Status ResourceManager::clearStorageTexture(QueueManager& queues,
                                                BindingManager& bindings,
                                                detail::BackendCommandBuffer& commandBuffer,
                                                const NativeTextureView& view,
                                                const StorageTextureClearValue& value)
    {
        const auto texture = std::static_pointer_cast<TextureRecord>(view.texture.owner);
        if (texture == nullptr || texture->resource.Get() != view.texture.resource)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The storage-texture clear view does not reference a live texture.",
            });
        }
        auto format = detail::validateStorageTextureClear(texture->desc, view.usage, view.viewDesc.format, value);
        if (!format)
        {
            return std::unexpected(std::move(format.error()));
        }
        auto layoutUse = prepareTextureLayoutUse(queues,
                                                 commandBuffer,
                                                 *texture,
                                                 storageViewSubresources(texture->desc, view.viewDesc),
                                                 TextureLayout::unorderedAccess,
                                                 "clearStorageTexture");
        if (!layoutUse)
        {
            return std::unexpected(std::move(layoutUse.error()));
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto shaderVisible = bindings.allocateResourceDescriptors(1);
        if (!shaderVisible)
        {
            return std::unexpected(std::move(shaderVisible.error()));
        }
        auto sparseSnapshot = captureSparseTextureBinding(view.texture);
        if (!sparseSnapshot)
        {
            return std::unexpected(std::move(sparseSnapshot.error()));
        }
        impl_->device->CopyDescriptorsSimple(
            1, (*shaderVisible)->cpuHandle(), view.descriptor, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        const std::array retentions{
            QueueManager::ResourceRetention{
                .owner = view.texture.owner,
                .residency = view.texture.residency,
            },
            QueueManager::ResourceRetention{.owner = view.owner},
            QueueManager::ResourceRetention{.owner = *shaderVisible, .internal = true},
        };
        const std::array swapchainLeases{view.texture.swapchainLease};
        auto retained = retainClearDependencies(queues, commandBuffer, retentions, swapchainLeases, *sparseSnapshot);
        if (!retained)
        {
            return retained;
        }

        bindClearDescriptorHeaps(**commandList, bindings);
        if (format->type == StorageTextureClearType::float32)
        {
            (*commandList)
                ->ClearUnorderedAccessViewFloat((*shaderVisible)->gpuHandle(),
                                                view.descriptor,
                                                view.texture.resource,
                                                value.float32.data(),
                                                0,
                                                nullptr);
        }
        else
        {
            std::array<UINT, 4> values{};
            for (std::size_t channel = 0; channel < values.size(); ++channel)
            {
                values[channel] = format->type == StorageTextureClearType::uint32
                                      ? value.uint32[channel]
                                      : std::bit_cast<std::uint32_t>(value.sint32[channel]);
            }
            (*commandList)
                ->ClearUnorderedAccessViewUint(
                    (*shaderVisible)->gpuHandle(), view.descriptor, view.texture.resource, values.data(), 0, nullptr);
        }
        commitTextureLayoutUse(*layoutUse);
        test::noteResourceCommandRecorded();
        return {};
    }

} // namespace rachis::d3d12
