#include "D3D12/D3D12ResourceInternal.h"

#include "D3D12/D3D12Binding.h"

namespace rachis::d3d12
{

    using namespace resource_detail;

    namespace
    {

        [[nodiscard]] Status validateSparseBufferDesc(const BufferDesc& desc, bool sparseBuffers)
        {
            constexpr auto allowedUsage = static_cast<std::uint32_t>(BufferUsage::copySource) |
                                          static_cast<std::uint32_t>(BufferUsage::copyDestination) |
                                          static_cast<std::uint32_t>(BufferUsage::constant) |
                                          static_cast<std::uint32_t>(BufferUsage::storage);
            const auto usage = static_cast<std::uint32_t>(desc.usage);
            const auto tileCount = desc.size == 0 ? 0 : ((desc.size - 1) / sparseBufferTileSize) + 1;
            if (desc.size == 0 || usage == 0 || (usage & ~allowedUsage) != 0 ||
                desc.memoryType != MemoryType::deviceLocal || desc.residencyPriority != ResidencyPriority::automatic ||
                desc.allocationFlags != ResourceAllocationFlag::none ||
                tileCount > std::numeric_limits<std::uint32_t>::max())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A sparse buffer requires bounded copy or shader-visible device-local storage with automatic "
                    "residency.",
                });
            }
            if (!sparseBuffers)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Sparse buffers are unavailable on this device.",
                });
            }
            return {};
        }

    } // namespace

    Result<BufferHandle> ResourceManager::createSparseBuffer(const BufferDesc& desc)
    {
        auto valid = validateSparseBufferDesc(desc, impl_->sparseResources);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }

        auto record = std::make_shared<BufferRecord>();
        record->desc = desc;
        record->label = desc.label;
        record->desc.label = record->label;
        record->sparse = true;
        record->sparseMappings = std::make_shared<SparseMappingState>();
        record->sparseState = std::make_shared<SparseMappingController>();
        record->sparseState->current = record->sparseMappings;

        const auto nativeDesc = nativeBufferDesc(desc);
        const HRESULT result = impl_->device->CreateReservedResource(
            &nativeDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&record->resource));
        if (FAILED(result))
        {
            return std::unexpected(nativeError(classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result)),
                                               result,
                                               "ID3D12Device::CreateReservedResource(buffer)"));
        }
        record->nativeWidth = nativeDesc.Width;

        UINT tileCount = 0;
        impl_->device->GetResourceTiling(record->resource.Get(), &tileCount, nullptr, nullptr, nullptr, 0, nullptr);
        const auto expectedTileCount = static_cast<UINT>(((desc.size - 1) / sparseBufferTileSize) + 1);
        if (tileCount == 0 || tileCount != expectedTileCount)
        {
            return std::unexpected(Error{
                ErrorCode::nativeFailure,
                0,
                "D3D12 returned an unexpected tile count for a sparse buffer.",
            });
        }
        record->sparseProperties = {
            .tileSize = sparseBufferTileSize,
            .tileCount = tileCount,
        };
        const auto label = wideLabel(desc.label);
        if (!label.empty())
        {
            record->resource->SetName(label.c_str());
        }

        const std::scoped_lock mappingLock(impl_->sparseMappingsMutex);
        const std::scoped_lock tableLock(impl_->buffersMutex);
        const auto handle = impl_->buffers.emplace(std::move(record));
        if (handle)
        {
            impl_->activeSparseResourceCount.fetch_add(1, std::memory_order_acq_rel);
        }
        return handle;
    }

    Result<SparseBufferProperties> ResourceManager::querySparseBufferProperties(BufferHandle buffer) const
    {
        auto record = impl_->buffer(buffer);
        if (!record)
        {
            return std::unexpected(std::move(record.error()));
        }
        if (!(*record)->sparse)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Sparse-buffer properties require a buffer created through the sparse API.",
            });
        }
        return (*record)->sparseProperties;
    }

    namespace
    {

        [[nodiscard]] bool tileRangesOverlap(std::uint32_t first,
                                             std::uint32_t count,
                                             std::uint32_t otherFirst,
                                             std::uint32_t otherCount) noexcept
        {
            const auto end = static_cast<std::uint64_t>(first) + count;
            const auto otherEnd = static_cast<std::uint64_t>(otherFirst) + otherCount;
            return first < otherEnd && otherFirst < end;
        }

        [[nodiscard]] bool isSparseHeapTileLeaseAncestor(const std::shared_ptr<SparseHeapTileLease>& candidate,
                                                         const SparseHeapTileLease& lease) noexcept
        {
            return candidate.get() == &lease || std::ranges::any_of(lease.ancestors, [&](const auto& ancestor) {
                       return ancestor.lock() == candidate;
                   });
        }

        void inheritSparseHeapTileLeaseAncestry(SparseHeapTileLease& lease,
                                                const std::shared_ptr<SparseHeapTileLease>& ancestor)
        {
            const auto adopt = [&](const std::shared_ptr<SparseHeapTileLease>& candidate) {
                if (tileRangesOverlap(lease.firstTile, lease.tileCount, candidate->firstTile, candidate->tileCount) &&
                    !isSparseHeapTileLeaseAncestor(candidate, lease))
                {
                    lease.ancestors.emplace_back(candidate);
                }
            };
            adopt(ancestor);
            for (const auto& inherited : ancestor->ancestors)
            {
                if (const auto live = inherited.lock())
                {
                    adopt(live);
                }
            }
        }

        [[nodiscard]] Result<std::shared_ptr<SparseHeapTileLease>> reserveSparseHeapTiles(
            const std::shared_ptr<ResourceHeapRecord>& heap,
            std::uint32_t firstTile,
            std::uint32_t tileCount,
            std::span<const TimelinePoint> orderedRetirements,
            std::vector<std::shared_ptr<SparseHeapTileLease>>& chainedRetirements)
        {
            const auto heapTileCount = heap->desc.size / sparseBufferTileSize;
            if (!hasAny(heap->desc.flags, ResourceHeapFlag::sparseBacking) || tileCount == 0 ||
                firstTile > heapTileCount || tileCount > heapTileCount - firstTile)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A sparse mapping requires an in-bounds range in a dedicated sparse-backing heap.",
                });
            }

            auto lease = std::make_shared<SparseHeapTileLease>();
            lease->heap = heap;
            lease->firstTile = firstTile;
            lease->tileCount = tileCount;

            const std::scoped_lock lock(heap->mutex);
            if (!heap->alive)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A sparse mapping requires a live sparse-backing heap.",
                });
            }
            auto& leases = heap->sparseTiles;
            auto iterator = leases.begin();
            while (iterator != leases.end())
            {
                const auto existing = iterator->lock();
                if (existing == nullptr)
                {
                    iterator = leases.erase(iterator);
                    continue;
                }
                const bool orderedRetirement =
                    existing->reusableAfter.isValid() &&
                    std::ranges::find(orderedRetirements, existing->reusableAfter) != orderedRetirements.end();
                if (tileRangesOverlap(firstTile, tileCount, existing->firstTile, existing->tileCount) &&
                    !orderedRetirement)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "Sparse-backing heap tiles cannot overlap while an earlier mapping version remains retained.",
                    });
                }
                if (tileRangesOverlap(firstTile, tileCount, existing->firstTile, existing->tileCount))
                {
                    inheritSparseHeapTileLeaseAncestry(*lease, existing);
                    if (std::ranges::find(chainedRetirements, existing) == chainedRetirements.end())
                    {
                        chainedRetirements.push_back(existing);
                    }
                }
                ++iterator;
            }
            leases.emplace_back(lease);
            return lease;
        }

        [[nodiscard]] Result<std::shared_ptr<SparseHeapTileLease>> sliceSparseHeapTileLease(
            const std::shared_ptr<SparseHeapTileLease>& source, std::uint32_t firstTile, std::uint32_t tileCount)
        {
            if (source == nullptr || source->heap == nullptr || tileCount == 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A sparse mapping region cannot derive a physical lease from invalid storage.",
                });
            }
            const auto sourceEnd = static_cast<std::uint64_t>(source->firstTile) + source->tileCount;
            const auto sliceEnd = static_cast<std::uint64_t>(firstTile) + tileCount;
            if (firstTile < source->firstTile || sliceEnd > sourceEnd)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A sparse mapping region cannot derive a lease outside its retained physical tiles.",
                });
            }

            auto lease = std::make_shared<SparseHeapTileLease>();
            lease->heap = source->heap;
            lease->firstTile = firstTile;
            lease->tileCount = tileCount;
            inheritSparseHeapTileLeaseAncestry(*lease, source);

            const std::scoped_lock lock(lease->heap->mutex);
            auto& leases = lease->heap->sparseTiles;
            auto iterator = leases.begin();
            while (iterator != leases.end())
            {
                const auto existing = iterator->lock();
                if (existing == nullptr)
                {
                    iterator = leases.erase(iterator);
                    continue;
                }
                if (tileRangesOverlap(firstTile, tileCount, existing->firstTile, existing->tileCount) &&
                    !isSparseHeapTileLeaseAncestor(existing, *lease))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "A sliced sparse-backing lease cannot overlap unrelated retained tiles.",
                    });
                }
                ++iterator;
            }
            leases.emplace_back(lease);
            return lease;
        }

        struct CarvedSparseRegions final
        {
            std::vector<SparseMappingRegion> remaining;
            std::vector<SparseMappingRegion> removed;
        };

        struct SparseTileRangeUpdate final
        {
            std::uint32_t virtualFirstTile = 0;
            std::uint32_t tileCount = 0;
            std::shared_ptr<ResourceHeapRecord> heap;
            std::uint32_t heapFirstTile = 0;
        };

        [[nodiscard]] Result<SparseMappingRegion> sliceRegionBeforeOverlap(const SparseMappingRegion& region,
                                                                           std::uint64_t overlapBegin)
        {
            const auto regionBegin = static_cast<std::uint64_t>(region.firstTile);
            const auto leftCount = static_cast<std::uint32_t>(overlapBegin - regionBegin);
            auto leftLease = sliceSparseHeapTileLease(region.lease, region.heapFirstTile, leftCount);
            if (!leftLease)
            {
                return std::unexpected(std::move(leftLease.error()));
            }
            return SparseMappingRegion{
                .firstTile = region.firstTile,
                .tileCount = leftCount,
                .heapFirstTile = region.heapFirstTile,
                .lease = std::move(*leftLease),
            };
        }

        [[nodiscard]] Result<SparseMappingRegion> sliceRegionAfterOverlap(const SparseMappingRegion& region,
                                                                          std::uint64_t overlapEnd)
        {
            const auto regionBegin = static_cast<std::uint64_t>(region.firstTile);
            const auto regionEnd = regionBegin + region.tileCount;
            const auto rightHeapFirst = static_cast<std::uint64_t>(region.heapFirstTile) + overlapEnd - regionBegin;
            if (rightHeapFirst > std::numeric_limits<std::uint32_t>::max())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A sparse mapping remainder produced an unrepresentable physical tile offset.",
                });
            }
            const auto rightCount = static_cast<std::uint32_t>(regionEnd - overlapEnd);
            auto rightLease =
                sliceSparseHeapTileLease(region.lease, static_cast<std::uint32_t>(rightHeapFirst), rightCount);
            if (!rightLease)
            {
                return std::unexpected(std::move(rightLease.error()));
            }
            return SparseMappingRegion{
                .firstTile = static_cast<std::uint32_t>(overlapEnd),
                .tileCount = rightCount,
                .heapFirstTile = static_cast<std::uint32_t>(rightHeapFirst),
                .lease = std::move(*rightLease),
            };
        }

        [[nodiscard]] Status carveUpdatedTilesFromRegion(
            const SparseMappingRegion& region,
            std::uint64_t updateBegin,
            std::uint64_t updateEnd,
            CarvedSparseRegions& carved,
            std::vector<std::shared_ptr<SparseHeapTileLease>>& retiredLeases)
        {
            const auto regionBegin = static_cast<std::uint64_t>(region.firstTile);
            const auto regionEnd = regionBegin + region.tileCount;
            if (regionEnd <= updateBegin || regionBegin >= updateEnd)
            {
                carved.remaining.push_back(region);
                return {};
            }

            const auto overlapBegin = (std::max)(regionBegin, updateBegin);
            const auto overlapEnd = (std::min)(regionEnd, updateEnd);
            const auto heapOverlapBegin = static_cast<std::uint64_t>(region.heapFirstTile) + overlapBegin - regionBegin;
            if (heapOverlapBegin > std::numeric_limits<std::uint32_t>::max())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A sparse mapping region produced an unrepresentable physical tile offset.",
                });
            }
            if (std::ranges::find(retiredLeases, region.lease) == retiredLeases.end())
            {
                retiredLeases.push_back(region.lease);
            }
            carved.removed.push_back({
                .firstTile = static_cast<std::uint32_t>(overlapBegin),
                .tileCount = static_cast<std::uint32_t>(overlapEnd - overlapBegin),
                .heapFirstTile = static_cast<std::uint32_t>(heapOverlapBegin),
                .lease = region.lease,
            });
            if (regionBegin < overlapBegin)
            {
                auto leftRemainder = sliceRegionBeforeOverlap(region, overlapBegin);
                if (!leftRemainder)
                {
                    return std::unexpected(std::move(leftRemainder.error()));
                }
                carved.remaining.push_back(std::move(*leftRemainder));
            }
            if (overlapEnd < regionEnd)
            {
                auto rightRemainder = sliceRegionAfterOverlap(region, overlapEnd);
                if (!rightRemainder)
                {
                    return std::unexpected(std::move(rightRemainder.error()));
                }
                carved.remaining.push_back(std::move(*rightRemainder));
            }
            return {};
        }

        [[nodiscard]] Result<CarvedSparseRegions> carveSparseTileRange(
            std::span<const SparseMappingRegion> regions,
            std::uint32_t firstTile,
            std::uint32_t tileCount,
            std::vector<std::shared_ptr<SparseHeapTileLease>>& retiredLeases)
        {
            const auto updateBegin = static_cast<std::uint64_t>(firstTile);
            const auto updateEnd = updateBegin + tileCount;
            CarvedSparseRegions carved;
            carved.remaining.reserve(regions.size() + 1);
            carved.removed.reserve(regions.size());
            for (const auto& region : regions)
            {
                auto split = carveUpdatedTilesFromRegion(region, updateBegin, updateEnd, carved, retiredLeases);
                if (!split)
                {
                    return std::unexpected(std::move(split.error()));
                }
            }
            return carved;
        }

        [[nodiscard]] Result<std::shared_ptr<SparseHeapTileLease>> leaseUpdatedHeapTiles(
            std::span<const SparseMappingRegion> removed,
            const SparseTileRangeUpdate& update,
            std::span<const TimelinePoint> dependencies,
            std::vector<std::shared_ptr<SparseHeapTileLease>>& retiredLeases)
        {
            const auto reusable = std::ranges::find_if(removed, [&](const auto& region) {
                return region.tileCount == update.tileCount && region.heapFirstTile == update.heapFirstTile &&
                       region.lease != nullptr && region.lease->heap == update.heap;
            });
            if (reusable != removed.end())
            {
                return sliceSparseHeapTileLease(reusable->lease, update.heapFirstTile, update.tileCount);
            }
            return reserveSparseHeapTiles(
                update.heap, update.heapFirstTile, update.tileCount, dependencies, retiredLeases);
        }

        [[nodiscard]] Result<std::vector<SparseMappingRegion>> rebuildSparseRegionsAfterCarve(
            CarvedSparseRegions carved,
            const SparseTileRangeUpdate& update,
            std::span<const TimelinePoint> dependencies,
            std::vector<std::shared_ptr<SparseHeapTileLease>>& retiredLeases)
        {
            if (update.heap != nullptr)
            {
                auto lease = leaseUpdatedHeapTiles(carved.removed, update, dependencies, retiredLeases);
                if (!lease)
                {
                    return std::unexpected(std::move(lease.error()));
                }
                carved.remaining.push_back({
                    .firstTile = update.virtualFirstTile,
                    .tileCount = update.tileCount,
                    .heapFirstTile = update.heapFirstTile,
                    .lease = std::move(*lease),
                });
            }

            std::ranges::sort(carved.remaining, {}, &SparseMappingRegion::firstTile);
            return std::move(carved.remaining);
        }

        [[nodiscard]] Status applySparseTileRangeUpdate(
            std::vector<SparseMappingRegion>& regions,
            const SparseTileRangeUpdate& update,
            std::span<const TimelinePoint> dependencies,
            std::vector<std::shared_ptr<SparseHeapTileLease>>& retiredLeases)
        {
            auto carved = carveSparseTileRange(regions, update.virtualFirstTile, update.tileCount, retiredLeases);
            if (!carved)
            {
                return std::unexpected(std::move(carved.error()));
            }
            auto rebuilt = rebuildSparseRegionsAfterCarve(std::move(*carved), update, dependencies, retiredLeases);
            if (!rebuilt)
            {
                return std::unexpected(std::move(rebuilt.error()));
            }
            regions = std::move(*rebuilt);
            return {};
        }

        struct PreparedSparseBufferMappingBatch final
        {
            struct Replacement final
            {
                BufferHandle handle;
                std::shared_ptr<BufferRecord> before;
                std::shared_ptr<BufferRecord> after;
            };

            std::vector<QueueManager::SparseTileUpdate> nativeMappings;
            std::vector<TimelinePoint> dependencies;
            std::vector<std::shared_ptr<SparseHeapTileLease>> retiredLeases;
            std::vector<Replacement> replacements;
        };

        [[nodiscard]] Status validateSparseBufferTileUpdateRequest(
            const SparseBufferTileMapping& mapping, std::span<const SparseBufferTileMapping> earlierMappings)
        {
            if (mapping.tileCount == 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A sparse-buffer tile update cannot be empty.",
                });
            }
            for (const auto& previous : earlierMappings)
            {
                if (previous.buffer == mapping.buffer &&
                    tileRangesOverlap(
                        previous.bufferTileOffset, previous.tileCount, mapping.bufferTileOffset, mapping.tileCount))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Sparse-buffer mapping updates in one batch cannot overlap virtually.",
                    });
                }
            }
            return {};
        }

        [[nodiscard]] Status validateSparseBufferAcceptsRemapping(const BufferRecord& buffer)
        {
            if (!buffer.sparse)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Sparse-buffer mappings require buffers created through the sparse API.",
                });
            }
            if (buffer.sparseState == nullptr || buffer.sparseState->current != buffer.sparseMappings)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A sparse-buffer handle does not reference its current mapping generation.",
                });
            }
            if (buffer.sparseMappings != nullptr && buffer.sparseMappings->publicationUnavailable)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A sparse-buffer mapping generation has no publishable completion point.",
                });
            }
            if (buffer.sparseMappings != nullptr &&
                buffer.sparseMappings->commandRetentions.load(std::memory_order_acquire) != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A sparse-buffer mapping cannot change while a recorded or submitted command retains its current "
                    "version.",
                });
            }
            return {};
        }

        [[nodiscard]] std::shared_ptr<SparseMappingState> deriveNextSparseMappingState(
            const std::shared_ptr<const SparseMappingState>& current, std::vector<TimelinePoint>& dependencies)
        {
            auto state = std::make_shared<SparseMappingState>();
            if (current != nullptr)
            {
                state->regions = current->regions;
                state->publication = current->publication;
                if (state->publication.isValid() &&
                    std::ranges::find(dependencies, state->publication) == dependencies.end())
                {
                    dependencies.push_back(state->publication);
                }
            }
            return state;
        }

        [[nodiscard]] std::shared_ptr<BufferRecord> makeRemappedSparseBufferRecord(
            const BufferRecord& before, std::shared_ptr<const SparseMappingState> mappings)
        {
            auto after = std::make_shared<BufferRecord>();
            after->resource = before.resource;
            after->desc = before.desc;
            after->nativeWidth = before.nativeWidth;
            after->label = before.label;
            after->desc.label = after->label;
            after->sparseProperties = before.sparseProperties;
            after->sparse = true;
            after->sparseMappings = std::move(mappings);
            after->sparseState = before.sparseState;
            return after;
        }

        template<typename Records>
        [[nodiscard]] Result<std::size_t> findOrBeginSparseBufferReplacement(
            BufferHandle handle,
            const Records& records,
            PreparedSparseBufferMappingBatch& prepared,
            std::vector<std::shared_ptr<SparseMappingState>>& nextStates)
        {
            const auto existing = std::ranges::find_if(
                prepared.replacements, [handle](const auto& replacement) { return replacement.handle == handle; });
            if (existing != prepared.replacements.end())
            {
                return static_cast<std::size_t>(existing - prepared.replacements.begin());
            }

            auto before = records.buffer(handle);
            if (!before)
            {
                return std::unexpected(std::move(before.error()));
            }
            auto remappable = validateSparseBufferAcceptsRemapping(**before);
            if (!remappable)
            {
                return std::unexpected(std::move(remappable.error()));
            }

            auto state = deriveNextSparseMappingState((*before)->sparseMappings, prepared.dependencies);
            auto after = makeRemappedSparseBufferRecord(**before, state);
            prepared.replacements.push_back({
                .handle = handle,
                .before = std::move(*before),
                .after = std::move(after),
            });
            nextStates.push_back(std::move(state));
            return prepared.replacements.size() - 1;
        }

        [[nodiscard]] Status validateSparseBufferTileUpdateTarget(const SparseBufferTileMapping& mapping,
                                                                  const SparseBufferProperties& properties)
        {
            if (mapping.bufferTileOffset >= properties.tileCount ||
                mapping.tileCount > properties.tileCount - mapping.bufferTileOffset)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A sparse-buffer mapping update is outside the virtual tile range.",
                });
            }
            if (!mapping.heap && mapping.heapTileOffset != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A sparse-buffer unmap operation requires a zero heap-tile offset.",
                });
            }
            return {};
        }

        template<typename Records>
        [[nodiscard]] Result<std::shared_ptr<ResourceHeapRecord>> resolveSparseBufferBackingHeap(
            ResourceHeapHandle handle, const Records& records)
        {
            if (!handle)
            {
                return std::shared_ptr<ResourceHeapRecord>{};
            }
            auto heap = records.resourceHeap(handle);
            if (!heap)
            {
                return std::unexpected(std::move(heap.error()));
            }
            if ((*heap)->desc.resourceClass != ResourceHeapClass::buffer)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A sparse buffer requires a dedicated buffer-class sparse-backing heap.",
                });
            }
            return std::move(*heap);
        }

        [[nodiscard]] QueueManager::SparseTileUpdate nativeSparseBufferTileUpdate(
            const SparseBufferTileMapping& mapping,
            const BufferRecord& buffer,
            const std::shared_ptr<ResourceHeapRecord>& heap)
        {
            return {
                .resource = buffer.resource.Get(),
                .heap = heap != nullptr ? heap->heap.Get() : nullptr,
                .coordinate =
                    {
                        .X = mapping.bufferTileOffset,
                        .Y = 0,
                        .Z = 0,
                        .Subresource = 0,
                    },
                .region =
                    {
                        .NumTiles = mapping.tileCount,
                        .UseBox = FALSE,
                        .Width = 0,
                        .Height = 0,
                        .Depth = 0,
                    },
                .tileCount = mapping.tileCount,
                .heapTileOffset = mapping.heapTileOffset,
            };
        }

        template<typename Records>
        [[nodiscard]] Status stageSparseBufferTileMapping(std::span<const SparseBufferTileMapping> mappings,
                                                          std::size_t mappingIndex,
                                                          const Records& records,
                                                          PreparedSparseBufferMappingBatch& prepared,
                                                          std::vector<std::shared_ptr<SparseMappingState>>& nextStates)
        {
            const auto& mapping = mappings[mappingIndex];
            auto acceptable = validateSparseBufferTileUpdateRequest(mapping, mappings.first(mappingIndex));
            if (!acceptable)
            {
                return acceptable;
            }
            auto replacement = findOrBeginSparseBufferReplacement(mapping.buffer, records, prepared, nextStates);
            if (!replacement)
            {
                return std::unexpected(std::move(replacement.error()));
            }
            const auto& before = *prepared.replacements[*replacement].before;
            auto& state = *nextStates[*replacement];
            auto targetable = validateSparseBufferTileUpdateTarget(mapping, before.sparseProperties);
            if (!targetable)
            {
                return targetable;
            }

            auto carved = carveSparseTileRange(
                state.regions, mapping.bufferTileOffset, mapping.tileCount, prepared.retiredLeases);
            if (!carved)
            {
                return std::unexpected(std::move(carved.error()));
            }
            auto heap = resolveSparseBufferBackingHeap(mapping.heap, records);
            if (!heap)
            {
                return std::unexpected(std::move(heap.error()));
            }
            const SparseTileRangeUpdate update{
                .virtualFirstTile = mapping.bufferTileOffset,
                .tileCount = mapping.tileCount,
                .heap = *heap,
                .heapFirstTile = mapping.heapTileOffset,
            };
            auto rebuilt = rebuildSparseRegionsAfterCarve(
                std::move(*carved), update, prepared.dependencies, prepared.retiredLeases);
            if (!rebuilt)
            {
                return std::unexpected(std::move(rebuilt.error()));
            }
            state.regions = std::move(*rebuilt);
            prepared.nativeMappings.push_back(nativeSparseBufferTileUpdate(mapping, before, *heap));
            return {};
        }

        template<typename RecordTable, typename Replacement>
        [[nodiscard]] bool replacedRecordsAreStillCurrent(const RecordTable& table,
                                                          const std::vector<Replacement>& replacements)
        {
            return std::ranges::all_of(replacements, [&](const auto& replacement) {
                const auto* stored = table.find(replacement.handle);
                return stored != nullptr && *stored == replacement.before;
            });
        }

        template<typename RecordTable, typename PreparedBatch>
        void publishSparseMappingBatch(RecordTable& table,
                                       const PreparedBatch& prepared,
                                       std::span<const std::shared_ptr<SparseMappingState>> nextStates,
                                       TimelinePoint publication)
        {
            for (const auto& state : nextStates)
            {
                state->publication = publication;
                state->publicationUnavailable = !publication.isValid();
            }
            for (const auto& lease : prepared.retiredLeases)
            {
                lease->reusableAfter = publication;
            }
            for (const auto& replacement : prepared.replacements)
            {
                auto* stored = table.find(replacement.handle);
                *stored = replacement.after;
                replacement.after->sparseState->current = replacement.after->sparseMappings;
            }
        }

    } // namespace

    Result<TimelinePoint> ResourceManager::enqueueSparseBufferMappings(
        QueueManager& queues, QueueType queueType, std::span<const SparseBufferTileMapping> mappings)
    {
        if (!impl_->sparseResources)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Sparse-buffer mapping is unavailable on this device.",
            });
        }
        if (mappings.empty() || mappings.size() > std::numeric_limits<UINT>::max())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A sparse-buffer mapping batch must contain at least one update within portable count limits.",
            });
        }

        const std::scoped_lock mappingLock(impl_->sparseMappingsMutex);
        auto prepared = std::make_shared<PreparedSparseBufferMappingBatch>();
        prepared->nativeMappings.reserve(mappings.size());
        prepared->replacements.reserve(mappings.size());
        std::vector<std::shared_ptr<SparseMappingState>> nextStates;
        nextStates.reserve(mappings.size());
        for (std::size_t mappingIndex = 0; mappingIndex < mappings.size(); ++mappingIndex)
        {
            auto staged = stageSparseBufferTileMapping(mappings, mappingIndex, *impl_, *prepared, nextStates);
            if (!staged)
            {
                return std::unexpected(std::move(staged.error()));
            }
        }

        const std::scoped_lock tableLock(impl_->buffersMutex);
        if (!replacedRecordsAreStillCurrent(impl_->buffers, prepared->replacements))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A sparse-buffer mapping version was concurrently superseded before native commit.",
            });
        }

        auto submitted =
            queues.enqueueSparseMappings(queueType, prepared->nativeMappings, prepared->dependencies, prepared);
        if (!submitted)
        {
            return std::unexpected(std::move(submitted.error()));
        }
        publishSparseMappingBatch(impl_->buffers, *prepared, nextStates, submitted->publication);
        if (submitted->failure.has_value())
        {
            return std::unexpected(std::move(*submitted->failure));
        }
        return submitted->publication;
    }

    namespace
    {

        [[nodiscard]] Status validateSparseTextureDesc(ID3D12Device& device,
                                                       const TextureDesc& desc,
                                                       const format::Info& info,
                                                       bool sparseTextures,
                                                       bool variableRateShadingTier2)
        {
            auto valid =
                validateTextureAllocationDesc(device, desc, info, variableRateShadingTier2, false, false, false);
            if (!valid)
            {
                return valid;
            }
            constexpr auto allowedUsage = static_cast<std::uint32_t>(TextureUsage::copySource) |
                                          static_cast<std::uint32_t>(TextureUsage::copyDestination) |
                                          static_cast<std::uint32_t>(TextureUsage::sampled) |
                                          static_cast<std::uint32_t>(TextureUsage::storage);
            const auto usage = static_cast<std::uint32_t>(desc.usage);
            if (desc.dimension != TextureDimension::texture2D || desc.depthOrArrayLayers != 1 ||
                desc.sampleCount != 1 || usage == 0 || (usage & ~allowedUsage) != 0 || info.depth || info.stencil ||
                info.compressed || !desc.viewFormats.empty() ||
                desc.residencyPriority != ResidencyPriority::automatic ||
                desc.allocationFlags != ResourceAllocationFlag::none ||
                desc.optimizedClearValue.kind != OptimizedClearValueKind::automatic)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A sparse texture requires single-layer, single-sample 2D color storage with copy, sampled, and/or "
                    "storage usage and automatic residency.",
                });
            }
            if (!sparseTextures)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Sparse textures are unavailable on this device.",
                });
            }
            return {};
        }

        struct SparseTextureTilingMetadata final
        {
            UINT totalTileCount = 0;
            D3D12_PACKED_MIP_INFO packedMips{};
            D3D12_TILE_SHAPE standardTileShape{};
            std::vector<D3D12_SUBRESOURCE_TILING> subresources;
        };

        [[nodiscard]] Result<SparseTextureTilingMetadata> querySparseTextureTiling(ID3D12Device& device,
                                                                                   ID3D12Resource& resource,
                                                                                   std::uint32_t mipLevels)
        {
            SparseTextureTilingMetadata tiling;
            UINT subresourceTilingCount = mipLevels;
            tiling.subresources.resize(mipLevels);
            device.GetResourceTiling(&resource,
                                     &tiling.totalTileCount,
                                     &tiling.packedMips,
                                     &tiling.standardTileShape,
                                     &subresourceTilingCount,
                                     0,
                                     tiling.subresources.data());
            const auto& packedMips = tiling.packedMips;
            const auto mipTotal = static_cast<std::uint32_t>(packedMips.NumStandardMips) +
                                  static_cast<std::uint32_t>(packedMips.NumPackedMips);
            const bool emptyStandardTileShape = tiling.standardTileShape.WidthInTexels == 0 ||
                                                tiling.standardTileShape.HeightInTexels == 0 ||
                                                tiling.standardTileShape.DepthInTexels == 0;
            const bool packedMipTilesOutOfBounds =
                packedMips.NumPackedMips != 0 &&
                (packedMips.NumTilesForPackedMips == 0 ||
                 packedMips.StartTileIndexInOverallResource >= tiling.totalTileCount ||
                 packedMips.NumTilesForPackedMips > tiling.totalTileCount - packedMips.StartTileIndexInOverallResource);
            if (tiling.totalTileCount == 0 || subresourceTilingCount != mipLevels || mipTotal != mipLevels ||
                emptyStandardTileShape || packedMipTilesOutOfBounds)
            {
                return std::unexpected(Error{
                    ErrorCode::nativeFailure,
                    0,
                    "D3D12 returned inconsistent sparse-texture tiling metadata.",
                });
            }
            return tiling;
        }

        [[nodiscard]] Status populateSparseTextureProperties(TextureRecord& record, SparseTextureTilingMetadata tiling)
        {
            const auto& packedMips = tiling.packedMips;
            record.sparseProperties = {
                .tileSize = sparseBufferTileSize,
                .totalTileCount = tiling.totalTileCount,
                .standardMipCount = packedMips.NumStandardMips,
                .packedMipCount = packedMips.NumPackedMips,
                .mipTailTileCount = packedMips.NumTilesForPackedMips,
                .standardTileExtent =
                    {
                        tiling.standardTileShape.WidthInTexels,
                        tiling.standardTileShape.HeightInTexels,
                        tiling.standardTileShape.DepthInTexels,
                    },
            };
            record.sparseProperties.standardMipTileGrids.reserve(packedMips.NumStandardMips);
            for (std::uint32_t mipLevel = 0; mipLevel < packedMips.NumStandardMips; ++mipLevel)
            {
                const auto& mipTiling = tiling.subresources[mipLevel];
                const auto tileCount = static_cast<std::uint64_t>(mipTiling.WidthInTiles) * mipTiling.HeightInTiles *
                                       mipTiling.DepthInTiles;
                if (mipTiling.StartTileIndexInOverallResource == D3D12_PACKED_TILE || tileCount == 0 ||
                    tileCount > tiling.totalTileCount ||
                    mipTiling.StartTileIndexInOverallResource > tiling.totalTileCount - tileCount)
                {
                    return std::unexpected(Error{
                        ErrorCode::nativeFailure,
                        0,
                        "D3D12 returned an invalid standard sparse-texture mip grid.",
                    });
                }
                record.sparseProperties.standardMipTileGrids.push_back({
                    mipTiling.WidthInTiles,
                    mipTiling.HeightInTiles,
                    mipTiling.DepthInTiles,
                });
            }
            record.sparseSubresources = std::move(tiling.subresources);
            record.sparseMipTailFirstTile = packedMips.StartTileIndexInOverallResource;
            return {};
        }

    } // namespace

    Result<TextureHandle> ResourceManager::createSparseTexture(const TextureDesc& desc)
    {
        const auto info = format::info(desc.format);
        if (!info.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A sparse texture requires a supported, non-unknown format.",
            });
        }
        auto valid = validateSparseTextureDesc(
            *impl_->device, desc, *info, impl_->sparseResources, impl_->variableRateShadingTier2);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }

        auto record = std::make_shared<TextureRecord>();
        record->desc = desc;
        record->label = desc.label;
        record->desc.label = record->label;
        record->sparse = true;
        record->sparseMappings = std::make_shared<SparseMappingState>();
        record->sparseState = std::make_shared<SparseMappingController>();
        record->sparseState->current = record->sparseMappings;

        auto nativeDesc = nativeTextureDesc(desc, *info);
        nativeDesc.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
        const HRESULT result = impl_->device->CreateReservedResource(
            &nativeDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&record->resource));
        if (FAILED(result))
        {
            return std::unexpected(nativeError(classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result)),
                                               result,
                                               "ID3D12Device::CreateReservedResource(texture)"));
        }

        auto tiling = querySparseTextureTiling(*impl_->device, *record->resource.Get(), desc.mipLevels);
        if (!tiling)
        {
            return std::unexpected(std::move(tiling.error()));
        }
        auto populated = populateSparseTextureProperties(*record, std::move(*tiling));
        if (!populated)
        {
            return std::unexpected(std::move(populated.error()));
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
        const std::scoped_lock mappingLock(impl_->sparseMappingsMutex);
        const std::scoped_lock tableLock(impl_->texturesMutex);
        const auto handle = impl_->textures.emplace(std::move(record));
        if (handle)
        {
            impl_->activeSparseResourceCount.fetch_add(1, std::memory_order_acq_rel);
        }
        return handle;
    }

    Result<SparseTextureProperties> ResourceManager::querySparseTextureProperties(TextureHandle texture) const
    {
        auto record = impl_->texture(texture);
        if (!record)
        {
            return std::unexpected(std::move(record.error()));
        }
        if (!(*record)->sparse)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Sparse-texture properties require a texture created through the sparse API.",
            });
        }
        return (*record)->sparseProperties;
    }

    namespace
    {

        struct PreparedSparseTextureMappingBatch final
        {
            struct Replacement final
            {
                TextureHandle handle;
                std::shared_ptr<TextureRecord> before;
                std::shared_ptr<TextureRecord> after;
            };

            std::vector<QueueManager::SparseTileUpdate> nativeMappings;
            std::vector<TimelinePoint> dependencies;
            std::vector<std::shared_ptr<SparseHeapTileLease>> retiredLeases;
            std::vector<Replacement> replacements;
        };

        [[nodiscard]] bool sparseTextureTileBoxesOverlap(const SparseTextureTileMapping& previous,
                                                         const SparseTextureTileMapping& mapping) noexcept
        {
            return previous.texture == mapping.texture &&
                   previous.subresource.mipLevel == mapping.subresource.mipLevel &&
                   previous.subresource.arrayLayer == mapping.subresource.arrayLayer &&
                   tileRangesOverlap(previous.textureTileOrigin.x,
                                     previous.textureTileExtent.width,
                                     mapping.textureTileOrigin.x,
                                     mapping.textureTileExtent.width) &&
                   tileRangesOverlap(previous.textureTileOrigin.y,
                                     previous.textureTileExtent.height,
                                     mapping.textureTileOrigin.y,
                                     mapping.textureTileExtent.height) &&
                   tileRangesOverlap(previous.textureTileOrigin.z,
                                     previous.textureTileExtent.depth,
                                     mapping.textureTileOrigin.z,
                                     mapping.textureTileExtent.depth);
        }

        [[nodiscard]] Status validateSparseTextureTileUpdateRequest(
            const SparseTextureTileMapping& mapping, std::span<const SparseTextureTileMapping> earlierMappings)
        {
            if (!mapping.heap && mapping.heapTileOffset != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A sparse-texture standard-tile unmap requires a zero heap-tile offset.",
                });
            }
            for (const auto& previous : earlierMappings)
            {
                if (sparseTextureTileBoxesOverlap(previous, mapping))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Sparse-texture standard-tile updates cannot overlap virtually within one batch.",
                    });
                }
            }
            return {};
        }

        [[nodiscard]] Status validateSparseTextureAcceptsRemapping(const TextureRecord& texture)
        {
            if (!texture.sparse)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Sparse-texture mappings require textures created through the sparse API.",
                });
            }
            if (texture.sparseState == nullptr || texture.sparseState->current != texture.sparseMappings)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A sparse-texture handle does not reference its current mapping generation.",
                });
            }
            if (texture.sparseMappings != nullptr && texture.sparseMappings->publicationUnavailable)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A sparse-texture mapping generation has no publishable completion point.",
                });
            }
            if (texture.sparseMappings != nullptr &&
                texture.sparseMappings->commandRetentions.load(std::memory_order_acquire) != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A sparse-texture mapping cannot change while a recorded or submitted command retains its current "
                    "version.",
                });
            }
            return {};
        }

        [[nodiscard]] std::shared_ptr<TextureRecord> makeRemappedSparseTextureRecord(
            const TextureRecord& before, std::shared_ptr<const SparseMappingState> mappings)
        {
            auto after = std::make_shared<TextureRecord>();
            after->resource = before.resource;
            after->desc = before.desc;
            after->label = before.label;
            after->desc.label = after->label;
            after->sparseProperties = before.sparseProperties;
            after->sparseSubresources = before.sparseSubresources;
            after->sparseMipTailFirstTile = before.sparseMipTailFirstTile;
            after->sparseMappings = std::move(mappings);
            after->sparseState = before.sparseState;
            after->legacyStates = before.legacyStates;
            after->sparse = true;
            return after;
        }

        template<typename Records>
        [[nodiscard]] Result<std::size_t> findOrBeginSparseTextureReplacement(
            TextureHandle handle,
            const Records& records,
            PreparedSparseTextureMappingBatch& prepared,
            std::vector<std::shared_ptr<SparseMappingState>>& nextStates)
        {
            const auto existing = std::ranges::find_if(
                prepared.replacements, [handle](const auto& replacement) { return replacement.handle == handle; });
            if (existing != prepared.replacements.end())
            {
                return static_cast<std::size_t>(existing - prepared.replacements.begin());
            }

            auto before = records.texture(handle);
            if (!before)
            {
                return std::unexpected(std::move(before.error()));
            }
            auto remappable = validateSparseTextureAcceptsRemapping(**before);
            if (!remappable)
            {
                return std::unexpected(std::move(remappable.error()));
            }

            auto state = deriveNextSparseMappingState((*before)->sparseMappings, prepared.dependencies);
            auto after = makeRemappedSparseTextureRecord(**before, state);
            prepared.replacements.push_back({
                .handle = handle,
                .before = std::move(*before),
                .after = std::move(after),
            });
            nextStates.push_back(std::move(state));
            return prepared.replacements.size() - 1;
        }

        [[nodiscard]] constexpr std::uint64_t tileBoxTileCount(const Extent3D& tileExtent) noexcept
        {
            return static_cast<std::uint64_t>(tileExtent.width) * tileExtent.height * tileExtent.depth;
        }

        [[nodiscard]] Status validateSparseTextureTileBox(const SparseTextureTileMapping& mapping,
                                                          const SparseTextureProperties& properties)
        {
            if (mapping.subresource.aspect != TextureAspect::color || mapping.subresource.arrayLayer != 0 ||
                mapping.subresource.mipLevel >= properties.standardMipCount || mapping.textureTileExtent.width == 0 ||
                mapping.textureTileExtent.height == 0 || mapping.textureTileExtent.depth == 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A standard sparse-texture update requires a non-empty color tile box in a standard mip.",
                });
            }
            const auto& grid = properties.standardMipTileGrids[mapping.subresource.mipLevel];
            if (mapping.textureTileOrigin.x >= grid.width ||
                mapping.textureTileExtent.width > grid.width - mapping.textureTileOrigin.x ||
                mapping.textureTileOrigin.y >= grid.height ||
                mapping.textureTileExtent.height > grid.height - mapping.textureTileOrigin.y ||
                mapping.textureTileOrigin.z >= grid.depth ||
                mapping.textureTileExtent.depth > grid.depth - mapping.textureTileOrigin.z ||
                mapping.textureTileExtent.height > std::numeric_limits<UINT16>::max() ||
                mapping.textureTileExtent.depth > std::numeric_limits<UINT16>::max())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A standard sparse-texture tile box is outside the queried mip grid.",
                });
            }
            const auto totalTiles = tileBoxTileCount(mapping.textureTileExtent);
            if (totalTiles == 0 || totalTiles > std::numeric_limits<UINT>::max())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A standard sparse-texture tile box exceeds portable tile-count limits.",
                });
            }
            return {};
        }

        template<typename Records>
        [[nodiscard]] Result<std::shared_ptr<ResourceHeapRecord>> resolveSparseTextureBackingHeap(
            ResourceHeapHandle handle, const Records& records)
        {
            if (!handle)
            {
                return std::shared_ptr<ResourceHeapRecord>{};
            }
            auto heap = records.resourceHeap(handle);
            if (!heap)
            {
                return std::unexpected(std::move(heap.error()));
            }
            if ((*heap)->desc.resourceClass != ResourceHeapClass::nonAttachmentTexture ||
                !hasAny((*heap)->desc.flags, ResourceHeapFlag::sparseBacking))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A sparse texture requires a dedicated non-attachment-texture sparse-backing heap.",
                });
            }
            return std::move(*heap);
        }

        [[nodiscard]] Status remapSparseTextureTileRows(const SparseTextureTileMapping& mapping,
                                                        const D3D12_SUBRESOURCE_TILING& tiling,
                                                        const std::shared_ptr<ResourceHeapRecord>& heap,
                                                        std::vector<SparseMappingRegion>& regions,
                                                        PreparedSparseTextureMappingBatch& prepared)
        {
            std::uint32_t physicalRow = 0;
            for (std::uint32_t z = 0; z < mapping.textureTileExtent.depth; ++z)
            {
                for (std::uint32_t y = 0; y < mapping.textureTileExtent.height; ++y)
                {
                    const auto virtualRow =
                        static_cast<std::uint64_t>(tiling.StartTileIndexInOverallResource) +
                        mapping.textureTileOrigin.x +
                        static_cast<std::uint64_t>(mapping.textureTileOrigin.y + y) * tiling.WidthInTiles +
                        static_cast<std::uint64_t>(mapping.textureTileOrigin.z + z) * tiling.WidthInTiles *
                            tiling.HeightInTiles;
                    const auto physicalOffset = static_cast<std::uint64_t>(mapping.heapTileOffset) + physicalRow;
                    if (virtualRow > std::numeric_limits<std::uint32_t>::max() ||
                        (heap != nullptr && physicalOffset > std::numeric_limits<std::uint32_t>::max()))
                    {
                        return std::unexpected(Error{
                            ErrorCode::invalidArgument,
                            0,
                            "A sparse-texture tile row has an unrepresentable virtual or physical offset.",
                        });
                    }
                    const SparseTileRangeUpdate rowUpdate{
                        .virtualFirstTile = static_cast<std::uint32_t>(virtualRow),
                        .tileCount = mapping.textureTileExtent.width,
                        .heap = heap,
                        .heapFirstTile = heap != nullptr ? static_cast<std::uint32_t>(physicalOffset) : 0,
                    };
                    auto applied =
                        applySparseTileRangeUpdate(regions, rowUpdate, prepared.dependencies, prepared.retiredLeases);
                    if (!applied)
                    {
                        return applied;
                    }
                    physicalRow += mapping.textureTileExtent.width;
                }
            }
            return {};
        }

        [[nodiscard]] QueueManager::SparseTileUpdate nativeSparseTextureTileBoxUpdate(
            const SparseTextureTileMapping& mapping,
            const TextureRecord& texture,
            const std::shared_ptr<ResourceHeapRecord>& heap)
        {
            const auto totalTiles = tileBoxTileCount(mapping.textureTileExtent);
            return {
                .resource = texture.resource.Get(),
                .heap = heap != nullptr ? heap->heap.Get() : nullptr,
                .coordinate =
                    {
                        .X = mapping.textureTileOrigin.x,
                        .Y = mapping.textureTileOrigin.y,
                        .Z = mapping.textureTileOrigin.z,
                        .Subresource = mapping.subresource.mipLevel,
                    },
                .region =
                    {
                        .NumTiles = static_cast<UINT>(totalTiles),
                        .UseBox = TRUE,
                        .Width = mapping.textureTileExtent.width,
                        .Height = static_cast<UINT16>(mapping.textureTileExtent.height),
                        .Depth = static_cast<UINT16>(mapping.textureTileExtent.depth),
                    },
                .tileCount = static_cast<std::uint32_t>(totalTiles),
                .heapTileOffset = mapping.heapTileOffset,
            };
        }

        template<typename Records>
        [[nodiscard]] Status stageSparseTextureTileMapping(std::span<const SparseTextureTileMapping> mappings,
                                                           std::size_t mappingIndex,
                                                           const Records& records,
                                                           PreparedSparseTextureMappingBatch& prepared,
                                                           std::vector<std::shared_ptr<SparseMappingState>>& nextStates)
        {
            const auto& mapping = mappings[mappingIndex];
            auto acceptable = validateSparseTextureTileUpdateRequest(mapping, mappings.first(mappingIndex));
            if (!acceptable)
            {
                return acceptable;
            }
            auto replacement = findOrBeginSparseTextureReplacement(mapping.texture, records, prepared, nextStates);
            if (!replacement)
            {
                return std::unexpected(std::move(replacement.error()));
            }
            const auto& before = *prepared.replacements[*replacement].before;
            auto& state = *nextStates[*replacement];
            auto inGrid = validateSparseTextureTileBox(mapping, before.sparseProperties);
            if (!inGrid)
            {
                return inGrid;
            }
            auto heap = resolveSparseTextureBackingHeap(mapping.heap, records);
            if (!heap)
            {
                return std::unexpected(std::move(heap.error()));
            }

            const auto& tiling = before.sparseSubresources[mapping.subresource.mipLevel];
            auto remapped = remapSparseTextureTileRows(mapping, tiling, *heap, state.regions, prepared);
            if (!remapped)
            {
                return remapped;
            }
            prepared.nativeMappings.push_back(nativeSparseTextureTileBoxUpdate(mapping, before, *heap));
            return {};
        }

        [[nodiscard]] Status validateSparseTextureMipTailUpdateRequest(
            const SparseTextureMipTailMapping& mapping, std::span<const SparseTextureMipTailMapping> earlierMappings)
        {
            if (!mapping.heap && mapping.heapTileOffset != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A sparse-texture mip-tail unmap requires a zero heap-tile offset.",
                });
            }
            if (std::ranges::any_of(earlierMappings, [&](const auto& previous) {
                    return previous.texture == mapping.texture && previous.arrayLayer == mapping.arrayLayer;
                }))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A sparse-texture mapping batch cannot update one packed mip tail more than once.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateSparseTextureMipTailTarget(const SparseTextureMipTailMapping& mapping,
                                                                const SparseTextureProperties& properties)
        {
            if (mapping.arrayLayer != 0 || properties.packedMipCount == 0 || properties.mipTailTileCount == 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A sparse-texture mip-tail update requires the texture's complete queried packed tail.",
                });
            }
            return {};
        }

        [[nodiscard]] QueueManager::SparseTileUpdate nativeSparseTextureMipTailUpdate(
            const SparseTextureMipTailMapping& mapping,
            const TextureRecord& texture,
            const std::shared_ptr<ResourceHeapRecord>& heap)
        {
            return {
                .resource = texture.resource.Get(),
                .heap = heap != nullptr ? heap->heap.Get() : nullptr,
                .coordinate =
                    {
                        .X = 0,
                        .Y = 0,
                        .Z = 0,
                        .Subresource = texture.sparseProperties.standardMipCount,
                    },
                .region =
                    {
                        .NumTiles = texture.sparseProperties.mipTailTileCount,
                        .UseBox = FALSE,
                        .Width = 0,
                        .Height = 0,
                        .Depth = 0,
                    },
                .tileCount = texture.sparseProperties.mipTailTileCount,
                .heapTileOffset = mapping.heapTileOffset,
            };
        }

        template<typename Records>
        [[nodiscard]] Status stageSparseTextureMipTailMapping(
            std::span<const SparseTextureMipTailMapping> mappings,
            std::size_t mappingIndex,
            const Records& records,
            PreparedSparseTextureMappingBatch& prepared,
            std::vector<std::shared_ptr<SparseMappingState>>& nextStates)
        {
            const auto& mapping = mappings[mappingIndex];
            auto acceptable = validateSparseTextureMipTailUpdateRequest(mapping, mappings.first(mappingIndex));
            if (!acceptable)
            {
                return acceptable;
            }
            auto replacement = findOrBeginSparseTextureReplacement(mapping.texture, records, prepared, nextStates);
            if (!replacement)
            {
                return std::unexpected(std::move(replacement.error()));
            }
            const auto& before = *prepared.replacements[*replacement].before;
            auto& state = *nextStates[*replacement];
            auto hasMipTail = validateSparseTextureMipTailTarget(mapping, before.sparseProperties);
            if (!hasMipTail)
            {
                return hasMipTail;
            }
            auto heap = resolveSparseTextureBackingHeap(mapping.heap, records);
            if (!heap)
            {
                return std::unexpected(std::move(heap.error()));
            }

            const SparseTileRangeUpdate mipTailUpdate{
                .virtualFirstTile = before.sparseMipTailFirstTile,
                .tileCount = before.sparseProperties.mipTailTileCount,
                .heap = *heap,
                .heapFirstTile = mapping.heapTileOffset,
            };
            auto applied =
                applySparseTileRangeUpdate(state.regions, mipTailUpdate, prepared.dependencies, prepared.retiredLeases);
            if (!applied)
            {
                return applied;
            }
            prepared.nativeMappings.push_back(nativeSparseTextureMipTailUpdate(mapping, before, *heap));
            return {};
        }

    } // namespace

    Result<TimelinePoint> ResourceManager::enqueueSparseTextureMappings(QueueManager& queues,
                                                                        QueueType queueType,
                                                                        const SparseTextureMappingBatch& mappings)
    {
        const auto mappingCount = mappings.standardTiles.size() + mappings.mipTails.size();
        if (!impl_->sparseResources)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Sparse-texture mapping is unavailable on this device.",
            });
        }
        if (mappingCount == 0 || mappingCount > std::numeric_limits<UINT>::max())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A sparse-texture mapping batch requires at least one update within portable count limits.",
            });
        }

        const std::scoped_lock mappingLock(impl_->sparseMappingsMutex);
        auto prepared = std::make_shared<PreparedSparseTextureMappingBatch>();
        prepared->nativeMappings.reserve(mappingCount);
        prepared->replacements.reserve(mappingCount);
        std::vector<std::shared_ptr<SparseMappingState>> nextStates;
        nextStates.reserve(mappingCount);
        for (std::size_t mappingIndex = 0; mappingIndex < mappings.standardTiles.size(); ++mappingIndex)
        {
            auto staged =
                stageSparseTextureTileMapping(mappings.standardTiles, mappingIndex, *impl_, *prepared, nextStates);
            if (!staged)
            {
                return std::unexpected(std::move(staged.error()));
            }
        }
        for (std::size_t mappingIndex = 0; mappingIndex < mappings.mipTails.size(); ++mappingIndex)
        {
            auto staged =
                stageSparseTextureMipTailMapping(mappings.mipTails, mappingIndex, *impl_, *prepared, nextStates);
            if (!staged)
            {
                return std::unexpected(std::move(staged.error()));
            }
        }

        const std::scoped_lock tableLock(impl_->texturesMutex);
        if (!replacedRecordsAreStillCurrent(impl_->textures, prepared->replacements))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A sparse-texture mapping generation was concurrently superseded before native commit.",
            });
        }

        auto submitted =
            queues.enqueueSparseMappings(queueType, prepared->nativeMappings, prepared->dependencies, prepared);
        if (!submitted)
        {
            return std::unexpected(std::move(submitted.error()));
        }
        publishSparseMappingBatch(impl_->textures, *prepared, nextStates, submitted->publication);
        if (submitted->failure.has_value())
        {
            return std::unexpected(std::move(*submitted->failure));
        }
        return submitted->publication;
    }

    Result<SamplerFeedbackMapHandle> ResourceManager::createMinMipFeedbackMap(const MinMipFeedbackMapDesc& desc)
    {
        return createSamplerFeedbackMap(desc.texture, desc.mipRegion, desc.label, SamplerFeedbackMapType::minMip);
    }

    Result<SamplerFeedbackMapHandle> ResourceManager::createMipRegionUsedFeedbackMap(
        const MipRegionUsedFeedbackMapDesc& desc)
    {
        return createSamplerFeedbackMap(
            desc.texture, desc.mipRegion, desc.label, SamplerFeedbackMapType::mipRegionUsed);
    }

    namespace
    {

        [[nodiscard]] constexpr std::uint32_t logicalFeedbackExtent(std::uint32_t pairedExtent,
                                                                    std::uint32_t mipRegion,
                                                                    std::uint32_t mipLevel) noexcept
        {
            const auto pairedMipExtent = std::max(1U, pairedExtent >> mipLevel);
            return (pairedMipExtent + mipRegion - 1U) / mipRegion;
        }

        [[nodiscard]] Extent3D mipRegionUsedDecodedExtent(const TextureDesc& texture, Extent3D mipRegion) noexcept
        {
            auto minimumDimension = [&](std::uint32_t pairedExtent, std::uint32_t regionExtent) {
                auto result = logicalFeedbackExtent(pairedExtent, regionExtent, 0);
                for (std::uint32_t mip = 1; mip < texture.mipLevels; ++mip)
                {
                    const auto logical = logicalFeedbackExtent(pairedExtent, regionExtent, mip);
                    if (logical > 1)
                    {
                        result = std::max(result, logical << mip);
                    }
                }
                return result;
            };

            Extent3D result{
                .width = minimumDimension(texture.width, mipRegion.width),
                .height = minimumDimension(texture.height, mipRegion.height),
                .depth = 1,
            };
            const auto mipChainDimension = 1U << (texture.mipLevels - 1U);
            if (std::max(result.width, result.height) < mipChainDimension)
            {
                if (result.width >= result.height)
                {
                    result.width = mipChainDimension;
                }
                else
                {
                    result.height = mipChainDimension;
                }
            }
            return result;
        }

        [[nodiscard]] bool isValidFeedbackMipRegion(const TextureDesc& texture, Extent3D mipRegion) noexcept
        {
            return std::has_single_bit(mipRegion.width) && std::has_single_bit(mipRegion.height) &&
                   mipRegion.width >= 4 && mipRegion.height >= 4 && mipRegion.depth == 1 &&
                   mipRegion.width <= texture.width / 2 && mipRegion.height <= texture.height / 2;
        }

        [[nodiscard]] Status validateSamplerFeedbackPairing(const TextureDesc& texture, Extent3D mipRegion)
        {
            const auto info = format::info(texture.format);
            if (!info.has_value() || info->depth || info->stencil || texture.dimension != TextureDimension::texture2D ||
                texture.depthOrArrayLayers != 1 || texture.sampleCount != 1 ||
                !hasAny(texture.usage, TextureUsage::sampled) || !isValidFeedbackMipRegion(texture, mipRegion))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A sampler-feedback map requires a single-layer, single-sample sampled Texture2D color resource "
                    "and a valid power-of-two mip region.",
                });
            }
            return {};
        }

        [[nodiscard]] Extent3D samplerFeedbackDecodedExtent(SamplerFeedbackMapType type,
                                                            const TextureDesc& texture,
                                                            Extent3D mipRegion) noexcept
        {
            if (type == SamplerFeedbackMapType::minMip)
            {
                return {
                    .width = logicalFeedbackExtent(texture.width, mipRegion.width, 0),
                    .height = logicalFeedbackExtent(texture.height, mipRegion.height, 0),
                    .depth = 1,
                };
            }
            return mipRegionUsedDecodedExtent(texture, mipRegion);
        }

        [[nodiscard]] std::shared_ptr<SamplerFeedbackMapRecord> makeSamplerFeedbackMapRecord(
            const std::shared_ptr<TextureRecord>& texture,
            Extent3D mipRegion,
            std::string_view label,
            SamplerFeedbackMapType type)
        {
            const auto& textureDesc = texture->desc;
            auto record = std::make_shared<SamplerFeedbackMapRecord>();
            record->targetedTexture = texture;
            record->type = type;
            record->mipRegion = mipRegion;
            record->pairedExtent = {
                .width = textureDesc.width,
                .height = textureDesc.height,
                .depth = 1,
            };
            record->mipLevels = type == SamplerFeedbackMapType::minMip ? 1 : textureDesc.mipLevels;
            record->decodedExtent = samplerFeedbackDecodedExtent(type, textureDesc, mipRegion);
            record->label = label;
            return record;
        }

        [[nodiscard]] Status createSamplerFeedbackResource(D3D12MA::Allocator& allocator,
                                                           SamplerFeedbackMapRecord& record)
        {
            const auto& textureDesc = record.targetedTexture->desc;
            const D3D12MA::ALLOCATION_DESC allocationDesc{
                .Flags = D3D12MA::ALLOCATION_FLAG_NONE,
                .HeapType = D3D12_HEAP_TYPE_DEFAULT,
                .ExtraHeapFlags = {},
                .CustomPool = {},
                .pPrivateData = {},
            };
            const D3D12_RESOURCE_DESC1 resourceDesc{
                .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                .Alignment = 0,
                .Width = textureDesc.width,
                .Height = textureDesc.height,
                .DepthOrArraySize = 1,
                .MipLevels = static_cast<std::uint16_t>(textureDesc.mipLevels),
                .Format = record.type == SamplerFeedbackMapType::minMip
                              ? DXGI_FORMAT_SAMPLER_FEEDBACK_MIN_MIP_OPAQUE
                              : DXGI_FORMAT_SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE,
                .SampleDesc = {.Count = 1, .Quality = 0},
                .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
                .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                .SamplerFeedbackMipRegion =
                    {
                        .Width = record.mipRegion.width,
                        .Height = record.mipRegion.height,
                        .Depth = 1,
                    },
            };
            const HRESULT result = allocator.CreateResource2(&allocationDesc,
                                                             &resourceDesc,
                                                             D3D12_RESOURCE_STATE_COMMON,
                                                             nullptr,
                                                             record.allocation.ReleaseAndGetAddressOf(),
                                                             IID_PPV_ARGS(&record.resource));
            if (FAILED(result))
            {
                return std::unexpected(nativeError(classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result)),
                                                   result,
                                                   "D3D12MA::Allocator::CreateResource2(sampler-feedback map)"));
            }
            return {};
        }

    } // namespace

    Result<SamplerFeedbackMapHandle> ResourceManager::createSamplerFeedbackMap(TextureHandle textureHandle,
                                                                               Extent3D mipRegion,
                                                                               std::string_view label,
                                                                               SamplerFeedbackMapType type)
    {
        if (!impl_->samplerFeedback || impl_->samplerFeedbackDevice == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Sampler feedback is unavailable on this device.",
            });
        }
        auto texture = impl_->texture(textureHandle);
        if (!texture)
        {
            return std::unexpected(std::move(texture.error()));
        }
        auto valid = validateSamplerFeedbackPairing((*texture)->desc, mipRegion);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }

        auto record = makeSamplerFeedbackMapRecord(*texture, mipRegion, label, type);
        auto allocationIdentity = reserveResourceMemoryIdentity(impl_->memoryTracking);
        if (!allocationIdentity)
        {
            return std::unexpected(std::move(allocationIdentity.error()));
        }
        auto created = createSamplerFeedbackResource(*impl_->allocator.Get(), *record);
        if (!created)
        {
            return std::unexpected(std::move(created.error()));
        }
        record->device = impl_->device;
        record->allocator = impl_->allocator;
        const auto nativeLabel = wideLabel(label);
        if (!nativeLabel.empty())
        {
            record->allocation->SetName(nativeLabel.c_str());
            record->resource->SetName(nativeLabel.c_str());
        }
        auto trackedMemory = trackResourceMemory(impl_->memoryTracking,
                                                 *allocationIdentity,
                                                 ResourceMemoryKind::texture,
                                                 MemoryType::deviceLocal,
                                                 *record->allocation.Get(),
                                                 false,
                                                 label);
        if (!trackedMemory)
        {
            return std::unexpected(std::move(trackedMemory.error()));
        }
        record->trackedMemory = std::move(*trackedMemory);
        const std::scoped_lock lock(impl_->samplerFeedbackMapsMutex);
        return impl_->samplerFeedbackMaps.emplace(std::move(record));
    }

    Result<MinMipFeedbackMapProperties> ResourceManager::queryMinMipFeedbackMapProperties(
        SamplerFeedbackMapHandle feedbackMap) const
    {
        auto record = impl_->samplerFeedbackMap(feedbackMap);
        if (!record)
        {
            return std::unexpected(std::move(record.error()));
        }
        if ((*record)->type != SamplerFeedbackMapType::minMip)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The sampler-feedback map is not a MinMip map.",
            });
        }
        return MinMipFeedbackMapProperties{
            .mipRegion = (*record)->mipRegion,
            .decodedExtent = (*record)->decodedExtent,
        };
    }

    Result<MipRegionUsedFeedbackMapProperties> ResourceManager::queryMipRegionUsedFeedbackMapProperties(
        SamplerFeedbackMapHandle feedbackMap) const
    {
        auto record = impl_->samplerFeedbackMap(feedbackMap);
        if (!record)
        {
            return std::unexpected(std::move(record.error()));
        }
        if ((*record)->type != SamplerFeedbackMapType::mipRegionUsed)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The sampler-feedback map is not a MipRegionUsed map.",
            });
        }
        return MipRegionUsedFeedbackMapProperties{
            .mipRegion = (*record)->mipRegion,
            .decodedExtent = (*record)->decodedExtent,
            .pairedExtent = (*record)->pairedExtent,
            .mipLevels = (*record)->mipLevels,
        };
    }

    Status ResourceManager::destroySamplerFeedbackMap(SamplerFeedbackMapHandle feedbackMap)
    {
        const std::scoped_lock lock(impl_->samplerFeedbackMapsMutex);
        const auto* stored = impl_->samplerFeedbackMaps.find(feedbackMap);
        if (stored == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The sampler-feedback map handle is invalid or stale.",
            });
        }
        markResourceMemoryDestroyed((*stored)->trackedMemory);
        static_cast<void>(impl_->samplerFeedbackMaps.erase(feedbackMap));
        return {};
    }

    Status ResourceManager::resetSamplerFeedbackMap(QueueManager& queues,
                                                    BindingManager& bindings,
                                                    detail::BackendCommandBuffer& commandBuffer,
                                                    SamplerFeedbackMapHandle feedbackMap)
    {
        if (commandBuffer.queueType() == QueueType::copy)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A sampler-feedback reset requires a graphics or compute command buffer.",
            });
        }
        if (commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A sampler-feedback reset cannot be recorded inside a rendering scope.",
            });
        }
        auto record = impl_->samplerFeedbackMap(feedbackMap);
        if (!record)
        {
            return std::unexpected(std::move(record.error()));
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto descriptors = bindings.allocateClearResourceDescriptors();
        if (!descriptors)
        {
            return std::unexpected(std::move(descriptors.error()));
        }
        impl_->samplerFeedbackDevice->CreateSamplerFeedbackUnorderedAccessView(
            (*record)->targetedTexture->resource.Get(),
            (*record)->resource.Get(),
            descriptors->shaderVisible->cpuHandle());
        impl_->samplerFeedbackDevice->CreateSamplerFeedbackUnorderedAccessView(
            (*record)->targetedTexture->resource.Get(), (*record)->resource.Get(), descriptors->cpuVisible->handle());

        auto retention = std::make_shared<std::array<std::shared_ptr<void>, 3>>(
            std::array<std::shared_ptr<void>, 3>{*record, descriptors->shaderVisible, descriptors->cpuVisible});
        auto retained =
            queues.retainInternalResource(commandBuffer, std::move(retention), (*record)->targetedTexture->residency);
        if (!retained)
        {
            return retained;
        }
        commandBuffer.computeState() = {};
        commandBuffer.rayTracingState() = {};
        commandBuffer.graphicsState() = {};
        ID3D12DescriptorHeap* const heaps[]{bindings.nativeDescriptorHeaps().resources};
        (*commandList)->SetDescriptorHeaps(static_cast<UINT>(std::size(heaps)), heaps);
        constexpr std::array<UINT, 4> ignoredClearValue{};
        (*commandList)
            ->ClearUnorderedAccessViewUint(descriptors->shaderVisible->gpuHandle(),
                                           descriptors->cpuVisible->handle(),
                                           (*record)->resource.Get(),
                                           ignoredClearValue.data(),
                                           0,
                                           nullptr);
        return {};
    }

    Status ResourceManager::decodeMinMipFeedback(QueueManager& queues,
                                                 detail::BackendCommandBuffer& commandBuffer,
                                                 const MinMipFeedbackDecode& decode)
    {
        return decodeSamplerFeedback(
            queues, commandBuffer, decode.source, decode.destination, SamplerFeedbackMapType::minMip);
    }

    Status ResourceManager::decodeMipRegionUsedFeedback(QueueManager& queues,
                                                        detail::BackendCommandBuffer& commandBuffer,
                                                        const MipRegionUsedFeedbackDecode& decode)
    {
        return decodeSamplerFeedback(
            queues, commandBuffer, decode.source, decode.destination, SamplerFeedbackMapType::mipRegionUsed);
    }

    namespace
    {

        [[nodiscard]] bool isDecodedSamplerFeedbackTexture(const TextureDesc& texture,
                                                           const SamplerFeedbackMapRecord& map) noexcept
        {
            return texture.dimension == TextureDimension::texture2D && texture.width == map.decodedExtent.width &&
                   texture.height == map.decodedExtent.height && texture.depthOrArrayLayers == 1 &&
                   texture.mipLevels == map.mipLevels && texture.sampleCount == 1 && texture.format == Format::r8Uint;
        }

        [[nodiscard]] Status validateSamplerFeedbackDecodeDestination(const TextureDesc& destination,
                                                                      const SamplerFeedbackMapRecord& source)
        {
            if (!isDecodedSamplerFeedbackTexture(destination, source) ||
                !hasAny(destination.usage, TextureUsage::copyDestination))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A sampler-feedback decode requires an exact R8_UINT copy-destination Texture2D matching the map's "
                    "decoded extent and mip count.",
                });
            }
            return {};
        }

        [[nodiscard]] D3D12_RECT samplerFeedbackLogicalRect(const SamplerFeedbackMapRecord& map, UINT mip) noexcept
        {
            return {
                0,
                0,
                static_cast<LONG>(logicalFeedbackExtent(map.pairedExtent.width, map.mipRegion.width, mip)),
                static_cast<LONG>(logicalFeedbackExtent(map.pairedExtent.height, map.mipRegion.height, mip)),
            };
        }

        void recordSamplerFeedbackDecode(ID3D12GraphicsCommandList1& commandList,
                                         const SamplerFeedbackMapRecord& source,
                                         const TextureRecord& destination)
        {
            if (source.type == SamplerFeedbackMapType::minMip)
            {
                commandList.ResolveSubresourceRegion(destination.resource.Get(),
                                                     0,
                                                     0,
                                                     0,
                                                     source.resource.Get(),
                                                     std::numeric_limits<UINT>::max(),
                                                     nullptr,
                                                     DXGI_FORMAT_R8_UINT,
                                                     D3D12_RESOLVE_MODE_DECODE_SAMPLER_FEEDBACK);
                return;
            }
            for (UINT mip = 0; mip < source.mipLevels; ++mip)
            {
                auto logicalRect = samplerFeedbackLogicalRect(source, mip);
                commandList.ResolveSubresourceRegion(destination.resource.Get(),
                                                     mip,
                                                     0,
                                                     0,
                                                     source.resource.Get(),
                                                     mip,
                                                     &logicalRect,
                                                     DXGI_FORMAT_R8_UINT,
                                                     D3D12_RESOLVE_MODE_DECODE_SAMPLER_FEEDBACK);
            }
        }

        [[nodiscard]] bool samplerFeedbackTextureIsMapped(const TextureRecord& texture,
                                                          const SamplerFeedbackMapRecord& map) noexcept
        {
            for (UINT mip = 0; mip < map.mipLevels; ++mip)
            {
                const auto logicalRect = samplerFeedbackLogicalRect(map, mip);
                const Extent3D extent{
                    .width = static_cast<std::uint32_t>(logicalRect.right),
                    .height = static_cast<std::uint32_t>(logicalRect.bottom),
                    .depth = 1,
                };
                if (!sparseTextureRegionIsMapped(texture, mip, {}, extent))
                {
                    return false;
                }
            }
            return true;
        }

        struct SamplerFeedbackSparseCapture final
        {
            SparseMappingCommandLeases leases;
            SparseMappingDependencies dependencies;
        };

        [[nodiscard]] Result<SamplerFeedbackSparseCapture> captureSamplerFeedbackTexture(
            const std::shared_ptr<TextureRecord>& texture,
            const SamplerFeedbackMapRecord& map,
            std::string_view operation)
        {
            SamplerFeedbackSparseCapture captured;
            auto mapping = captureSparseTextureMapping(texture, captured.leases, captured.dependencies);
            if (!mapping)
            {
                return std::unexpected(std::move(mapping.error()));
            }
            if (!samplerFeedbackTextureIsMapped(*texture, map))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    std::string("A sampler-feedback ") + std::string(operation) +
                        " cannot access an unmapped sparse-texture tile.",
                });
            }
            return captured;
        }

        [[nodiscard]] Status retainSamplerFeedbackResolve(QueueManager& queues,
                                                          detail::BackendCommandBuffer& commandBuffer,
                                                          const std::shared_ptr<SamplerFeedbackMapRecord>& map,
                                                          const std::shared_ptr<TextureRecord>& texture,
                                                          const SamplerFeedbackSparseCapture& sparse)
        {
            std::shared_ptr<void> sparseMapping;
            if (!sparse.leases.empty())
            {
                sparseMapping = sparse.leases[0];
            }
            auto retention = std::make_shared<std::array<std::shared_ptr<void>, 3>>(
                std::array<std::shared_ptr<void>, 3>{map, texture, std::move(sparseMapping)});
            const std::array residencies{map->targetedTexture->residency, texture->residency};
            const std::array resources{QueueManager::ResourceRetention{
                .owner = std::move(retention),
                .residencySet = residencies,
                .internal = true,
            }};
            return queues.retainCommandDependencies(commandBuffer, resources, {}, sparse.dependencies);
        }

    } // namespace

    Status ResourceManager::decodeSamplerFeedback(QueueManager& queues,
                                                  detail::BackendCommandBuffer& commandBuffer,
                                                  SamplerFeedbackMapHandle sourceHandle,
                                                  TextureHandle destinationHandle,
                                                  SamplerFeedbackMapType type)
    {
        if (commandBuffer.queueType() != QueueType::graphics)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A sampler-feedback decode requires a graphics command buffer.",
            });
        }
        if (commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A sampler-feedback decode cannot be recorded inside a rendering scope.",
            });
        }
        auto source = impl_->samplerFeedbackMap(sourceHandle);
        if (!source)
        {
            return std::unexpected(std::move(source.error()));
        }
        if ((*source)->type != type)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The sampler-feedback decode operation does not match the source map type.",
            });
        }
        std::unique_lock<std::mutex> mappingLock(impl_->sparseMappingsMutex, std::defer_lock);
        if (impl_->activeSparseResourceCount.load(std::memory_order_acquire) != 0)
        {
            mappingLock.lock();
        }
        auto destination = impl_->texture(destinationHandle);
        if (!mappingLock.owns_lock() && destination && (*destination)->sparse)
        {
            mappingLock.lock();
            destination = impl_->texture(destinationHandle);
        }
        if (!destination)
        {
            return std::unexpected(std::move(destination.error()));
        }
        auto decodable = validateSamplerFeedbackDecodeDestination((*destination)->desc, **source);
        if (!decodable)
        {
            return decodable;
        }
        auto sparse = captureSamplerFeedbackTexture(*destination, **source, "decode");
        if (!sparse)
        {
            return std::unexpected(std::move(sparse.error()));
        }

        auto commandList = queues.commandList1(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained = retainSamplerFeedbackResolve(queues, commandBuffer, *source, *destination, *sparse);
        if (!retained)
        {
            return retained;
        }
        recordSamplerFeedbackDecode(**commandList, **source, **destination);
        return {};
    }

    Status ResourceManager::encodeMinMipFeedback(QueueManager& queues,
                                                 detail::BackendCommandBuffer& commandBuffer,
                                                 const MinMipFeedbackEncode& encode)
    {
        return encodeSamplerFeedback(
            queues, commandBuffer, encode.source, encode.destination, SamplerFeedbackMapType::minMip);
    }

    Status ResourceManager::encodeMipRegionUsedFeedback(QueueManager& queues,
                                                        detail::BackendCommandBuffer& commandBuffer,
                                                        const MipRegionUsedFeedbackEncode& encode)
    {
        return encodeSamplerFeedback(
            queues, commandBuffer, encode.source, encode.destination, SamplerFeedbackMapType::mipRegionUsed);
    }

    namespace
    {

        [[nodiscard]] Status validateSamplerFeedbackEncodeSource(const TextureDesc& source,
                                                                 const SamplerFeedbackMapRecord& destination)
        {
            if (!isDecodedSamplerFeedbackTexture(source, destination) ||
                !hasAny(source.usage, TextureUsage::copySource))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A sampler-feedback encode requires an exact R8_UINT copy-source Texture2D matching the map's "
                    "decoded extent and mip count.",
                });
            }
            return {};
        }

        void recordSamplerFeedbackEncode(ID3D12GraphicsCommandList1& commandList,
                                         const TextureRecord& source,
                                         const SamplerFeedbackMapRecord& destination)
        {
            if (destination.type == SamplerFeedbackMapType::minMip)
            {
                commandList.ResolveSubresourceRegion(destination.resource.Get(),
                                                     std::numeric_limits<UINT>::max(),
                                                     0,
                                                     0,
                                                     source.resource.Get(),
                                                     0,
                                                     nullptr,
                                                     DXGI_FORMAT_R8_UINT,
                                                     D3D12_RESOLVE_MODE_ENCODE_SAMPLER_FEEDBACK);
                return;
            }
            for (UINT mip = 0; mip < destination.mipLevels; ++mip)
            {
                auto logicalRect = samplerFeedbackLogicalRect(destination, mip);
                commandList.ResolveSubresourceRegion(destination.resource.Get(),
                                                     mip,
                                                     0,
                                                     0,
                                                     source.resource.Get(),
                                                     mip,
                                                     &logicalRect,
                                                     DXGI_FORMAT_R8_UINT,
                                                     D3D12_RESOLVE_MODE_ENCODE_SAMPLER_FEEDBACK);
            }
        }

    } // namespace

    Status ResourceManager::encodeSamplerFeedback(QueueManager& queues,
                                                  detail::BackendCommandBuffer& commandBuffer,
                                                  TextureHandle sourceHandle,
                                                  SamplerFeedbackMapHandle destinationHandle,
                                                  SamplerFeedbackMapType type)
    {
        if (commandBuffer.queueType() != QueueType::graphics)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A sampler-feedback encode requires a graphics command buffer.",
            });
        }
        if (commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A sampler-feedback encode cannot be recorded inside a rendering scope.",
            });
        }
        std::unique_lock<std::mutex> mappingLock(impl_->sparseMappingsMutex, std::defer_lock);
        if (impl_->activeSparseResourceCount.load(std::memory_order_acquire) != 0)
        {
            mappingLock.lock();
        }
        auto source = impl_->texture(sourceHandle);
        if (!mappingLock.owns_lock() && source && (*source)->sparse)
        {
            mappingLock.lock();
            source = impl_->texture(sourceHandle);
        }
        if (!source)
        {
            return std::unexpected(std::move(source.error()));
        }
        auto destination = impl_->samplerFeedbackMap(destinationHandle);
        if (!destination)
        {
            return std::unexpected(std::move(destination.error()));
        }
        if ((*destination)->type != type)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The sampler-feedback encode operation does not match the destination map type.",
            });
        }
        auto encodable = validateSamplerFeedbackEncodeSource((*source)->desc, **destination);
        if (!encodable)
        {
            return encodable;
        }
        auto sparse = captureSamplerFeedbackTexture(*source, **destination, "encode");
        if (!sparse)
        {
            return std::unexpected(std::move(sparse.error()));
        }

        auto commandList = queues.commandList1(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained = retainSamplerFeedbackResolve(queues, commandBuffer, *destination, *source, *sparse);
        if (!retained)
        {
            return retained;
        }
        recordSamplerFeedbackEncode(**commandList, **source, **destination);
        return {};
    }

    Result<std::optional<SparseBindingSnapshot>> ResourceManager::captureSparseBufferBinding(
        const BufferBindingSource& source)
    {
        const auto record = std::static_pointer_cast<BufferRecord>(source.owner);
        if (record == nullptr || record->resource.Get() != source.resource)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A sparse-buffer binding capture requires a valid buffer source.",
            });
        }
        if (!record->sparse)
        {
            return std::optional<SparseBindingSnapshot>{};
        }

        const std::scoped_lock mappingLock(impl_->sparseMappingsMutex);
        if (record->sparseState == nullptr || record->sparseState->current == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A sparse-buffer binding has no current immutable mapping generation.",
            });
        }
        const auto mapping = record->sparseState->current;
        if (mapping->publicationUnavailable)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A sparse-buffer binding's mapping generation has no publishable completion point.",
            });
        }
        auto lease = std::make_shared<SparseMappingCommandLease>(mapping);
        return std::optional<SparseBindingSnapshot>{SparseBindingSnapshot{
            .owner = std::move(lease),
            .publication = mapping->publication,
        }};
    }

    Result<std::optional<SparseBindingSnapshot>> ResourceManager::captureSparseTextureBinding(
        const TextureViewSource& source)
    {
        const auto record = std::static_pointer_cast<TextureRecord>(source.owner);
        if (record == nullptr || record->resource.Get() != source.resource)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A sparse-texture binding capture requires a valid texture source.",
            });
        }
        if (!record->sparse)
        {
            return std::optional<SparseBindingSnapshot>{};
        }

        const std::scoped_lock mappingLock(impl_->sparseMappingsMutex);
        if (record->sparseState == nullptr || record->sparseState->current == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A sparse-texture binding has no current immutable mapping generation.",
            });
        }
        const auto mapping = record->sparseState->current;
        if (mapping->publicationUnavailable)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A sparse-texture binding's mapping generation has no publishable completion point.",
            });
        }
        auto lease = std::make_shared<SparseMappingCommandLease>(mapping);
        return std::optional<SparseBindingSnapshot>{SparseBindingSnapshot{
            .owner = std::move(lease),
            .publication = mapping->publication,
        }};
    }

} // namespace rachis::d3d12
