#pragma once

#include "D3D12/D3D12BarrierTranslation.h"
#include "D3D12/D3D12RuntimePolicies.h"

#include <d3d12.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rachis::d3d12
{

    [[nodiscard]] inline D3D12_RESOURCE_STATES toNativeResourceStates(LegacyResourceState state) noexcept
    {
        D3D12_RESOURCE_STATES result = D3D12_RESOURCE_STATE_COMMON;
        if (hasAny(state, LegacyResourceState::vertexAndConstantBuffer))
        {
            result |= D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        }
        if (hasAny(state, LegacyResourceState::indexBuffer))
        {
            result |= D3D12_RESOURCE_STATE_INDEX_BUFFER;
        }
        if (hasAny(state, LegacyResourceState::renderTarget))
        {
            result |= D3D12_RESOURCE_STATE_RENDER_TARGET;
        }
        if (hasAny(state, LegacyResourceState::unorderedAccess))
        {
            result |= D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        if (hasAny(state, LegacyResourceState::depthWrite))
        {
            result |= D3D12_RESOURCE_STATE_DEPTH_WRITE;
        }
        if (hasAny(state, LegacyResourceState::depthRead))
        {
            result |= D3D12_RESOURCE_STATE_DEPTH_READ;
        }
        if (hasAny(state, LegacyResourceState::nonPixelShaderResource))
        {
            result |= D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }
        if (hasAny(state, LegacyResourceState::pixelShaderResource))
        {
            result |= D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        }
        if (hasAny(state, LegacyResourceState::streamOut))
        {
            result |= D3D12_RESOURCE_STATE_STREAM_OUT;
        }
        if (hasAny(state, LegacyResourceState::indirectArgument))
        {
            result |= D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
        }
        if (hasAny(state, LegacyResourceState::copyDest))
        {
            result |= D3D12_RESOURCE_STATE_COPY_DEST;
        }
        if (hasAny(state, LegacyResourceState::copySource))
        {
            result |= D3D12_RESOURCE_STATE_COPY_SOURCE;
        }
        if (hasAny(state, LegacyResourceState::resolveDest))
        {
            result |= D3D12_RESOURCE_STATE_RESOLVE_DEST;
        }
        if (hasAny(state, LegacyResourceState::resolveSource))
        {
            result |= D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
        }
        if (hasAny(state, LegacyResourceState::raytracingAccelerationStructure))
        {
            result |= D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
        }
        if (hasAny(state, LegacyResourceState::shadingRateSource))
        {
            result |= D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE;
        }
        return result;
    }

    [[nodiscard]] inline D3D12_RESOURCE_STATES legacyQueueResourceStates(QueueType queueType) noexcept
    {
        return toNativeResourceStates(validLegacyStatesForQueue(queueType));
    }

    [[nodiscard]] inline bool legacyStateFitsQueue(D3D12_RESOURCE_STATES state,
                                                   D3D12_RESOURCE_STATES queueStates) noexcept
    {
        return (state & ~queueStates) == 0;
    }

    inline constexpr auto externallyUnknownTextureLayout = static_cast<TextureLayout>(0xFFU);

    class LegacyTextureStateTracker final
    {
    public:
        LegacyTextureStateTracker(ID3D12Resource* resource, std::uint32_t subresourceCount) noexcept
            : resource_(resource), subresourceCount_(subresourceCount)
        {
        }

        LegacyTextureStateTracker(const LegacyTextureStateTracker&) = delete;
        LegacyTextureStateTracker& operator=(const LegacyTextureStateTracker&) = delete;
        LegacyTextureStateTracker(LegacyTextureStateTracker&&) = delete;
        LegacyTextureStateTracker& operator=(LegacyTextureStateTracker&&) = delete;
        ~LegacyTextureStateTracker() = default;

        [[nodiscard]] ID3D12Resource* resource() const noexcept
        {
            return resource_;
        }

        [[nodiscard]] std::uint32_t subresourceCount() const noexcept
        {
            return subresourceCount_;
        }

        [[nodiscard]] std::mutex& mutex() const noexcept
        {
            return mutex_;
        }

        [[nodiscard]] D3D12_RESOURCE_STATES stateLocked(std::uint32_t subresource) const noexcept
        {
            return subresource < states_.size() ? states_[subresource] : D3D12_RESOURCE_STATE_COMMON;
        }

        void reserveLocked()
        {
            if (states_.size() != subresourceCount_)
            {
                states_.resize(subresourceCount_, D3D12_RESOURCE_STATE_COMMON);
            }
        }

        void setStateLocked(std::uint32_t subresource, D3D12_RESOURCE_STATES state) noexcept
        {
            if (subresource < states_.size())
            {
                states_[subresource] = state;
            }
        }

        void setInitialLayout(TextureLayout layout) noexcept
        {
            initialLayout_ = layout;
        }

        void markLayoutUnknown() noexcept
        {
            initialLayout_ = externallyUnknownTextureLayout;
            std::ranges::fill(layouts_, externallyUnknownTextureLayout);
        }

        [[nodiscard]] TextureLayout layoutLocked(std::uint32_t subresource) const noexcept
        {
            return subresource < layouts_.size() ? layouts_[subresource] : initialLayout_;
        }

        void reserveLayoutsLocked()
        {
            if (layouts_.size() != subresourceCount_)
            {
                layouts_.resize(subresourceCount_, initialLayout_);
            }
        }

        void setLayoutLocked(std::uint32_t subresource, TextureLayout layout) noexcept
        {
            if (subresource < layouts_.size())
            {
                layouts_[subresource] = layout;
            }
        }

    private:
        ID3D12Resource* resource_ = nullptr;
        std::uint32_t subresourceCount_ = 0;
        mutable std::mutex mutex_;
        std::vector<D3D12_RESOURCE_STATES> states_;
        std::vector<TextureLayout> layouts_;
        TextureLayout initialLayout_ = TextureLayout::common;
    };

    class LegacyTextureStateIndex final
    {
    public:
        template<typename Values, typename Make>
        [[nodiscard]] std::size_t findOrAppend(Values& values, const LegacyTextureStateTracker* tracker, Make&& make)
        {
            const auto position = std::ranges::lower_bound(entries_, tracker, std::ranges::less{}, &Entry::tracker);
            if (position != entries_.end() && position->tracker == tracker)
            {
                return position->index;
            }
            const auto offset = position - entries_.begin();
            reserveForAppend(entries_, 1U);
            values.push_back(make());
            entries_.insert(entries_.begin() + offset, Entry{.tracker = tracker, .index = values.size() - 1});
            return values.size() - 1;
        }

        void clear() noexcept
        {
            entries_.clear();
        }

        void erase(const LegacyTextureStateTracker* tracker) noexcept
        {
            const auto position = std::ranges::lower_bound(entries_, tracker, std::ranges::less{}, &Entry::tracker);
            if (position != entries_.end() && position->tracker == tracker)
            {
                entries_.erase(position);
            }
        }

        [[nodiscard]] std::optional<std::size_t> find(const LegacyTextureStateTracker* tracker) const noexcept
        {
            const auto position = std::ranges::lower_bound(entries_, tracker, std::ranges::less{}, &Entry::tracker);
            if (position != entries_.end() && position->tracker == tracker)
            {
                return position->index;
            }
            return std::nullopt;
        }

    private:
        struct Entry final
        {
            const LegacyTextureStateTracker* tracker = nullptr;
            std::size_t index = 0;
        };

        std::vector<Entry> entries_;
    };

    enum class LegacySubresourceUse : std::uint8_t
    {
        untouched,
        transitioned,
        discarded,
    };

    struct LegacyTextureStateUse final
    {
        std::shared_ptr<LegacyTextureStateTracker> tracker;
        std::vector<LegacySubresourceUse> uses;
        std::vector<D3D12_RESOURCE_STATES> assumed;
        std::vector<D3D12_RESOURCE_STATES> current;

        [[nodiscard]] D3D12_RESOURCE_STATES stateBeforeLocked(std::uint32_t subresource,
                                                              D3D12_RESOURCE_STATES queueStates) const noexcept
        {
            if (uses[subresource] != LegacySubresourceUse::untouched)
            {
                return current[subresource];
            }
            const auto state = tracker->stateLocked(subresource);
            return legacyStateFitsQueue(state, queueStates) ? state : D3D12_RESOURCE_STATE_COMMON;
        }

        [[nodiscard]] D3D12_RESOURCE_STATES stateBefore(std::uint32_t subresource,
                                                        D3D12_RESOURCE_STATES queueStates) const
        {
            const std::scoped_lock lock(tracker->mutex());
            return stateBeforeLocked(subresource, queueStates);
        }

        void transition(std::uint32_t subresource,
                        D3D12_RESOURCE_STATES before,
                        D3D12_RESOURCE_STATES after,
                        bool discard) noexcept
        {
            if (uses[subresource] == LegacySubresourceUse::untouched)
            {
                uses[subresource] = discard ? LegacySubresourceUse::discarded : LegacySubresourceUse::transitioned;
                assumed[subresource] = before;
            }
            current[subresource] = after;
        }

        std::vector<std::uint8_t> layoutSet;
        std::vector<std::uint8_t> layoutEntryRequired;
        std::vector<TextureLayout> layoutEntry;
        std::vector<TextureLayout> layoutCurrent;

        [[nodiscard]] std::optional<TextureLayout> layoutLeft(std::uint32_t subresource) const noexcept
        {
            return layoutSet[subresource] != 0 ? std::optional{layoutCurrent[subresource]} : std::nullopt;
        }

        void recordLayout(std::uint32_t subresource, TextureLayout before, TextureLayout after) noexcept
        {
            if (layoutSet[subresource] == 0)
            {
                layoutSet[subresource] = 1;
                if (before != TextureLayout::undefined)
                {
                    layoutEntryRequired[subresource] = 1;
                    layoutEntry[subresource] = before;
                }
            }
            layoutCurrent[subresource] = after;
        }
    };

    class LegacyTextureStateRecording final
    {
    public:
        [[nodiscard]] std::size_t use(const std::shared_ptr<LegacyTextureStateTracker>& tracker)
        {
            return indices_.findOrAppend(uses_, tracker.get(), [&] {
                const auto count = tracker->subresourceCount();
                return LegacyTextureStateUse{
                    .tracker = tracker,
                    .uses = std::vector<LegacySubresourceUse>(count, LegacySubresourceUse::untouched),
                    .assumed = std::vector<D3D12_RESOURCE_STATES>(count, D3D12_RESOURCE_STATE_COMMON),
                    .current = std::vector<D3D12_RESOURCE_STATES>(count, D3D12_RESOURCE_STATE_COMMON),
                    .layoutSet = std::vector<std::uint8_t>(count, 0),
                    .layoutEntryRequired = std::vector<std::uint8_t>(count, 0),
                    .layoutEntry = std::vector<TextureLayout>(count, TextureLayout::undefined),
                    .layoutCurrent = std::vector<TextureLayout>(count, TextureLayout::undefined),
                };
            });
        }

        [[nodiscard]] const LegacyTextureStateUse* find(const LegacyTextureStateTracker* tracker) const noexcept
        {
            const auto index = indices_.find(tracker);
            return index.has_value() ? &uses_[*index] : nullptr;
        }

        [[nodiscard]] LegacyTextureStateUse& operator[](std::size_t index) noexcept
        {
            return uses_[index];
        }

        [[nodiscard]] const std::vector<LegacyTextureStateUse>& uses() const noexcept
        {
            return uses_;
        }

        [[nodiscard]] bool empty() const noexcept
        {
            return uses_.empty();
        }

        void truncate(std::size_t size) noexcept
        {
            while (uses_.size() > size)
            {
                indices_.erase(uses_.back().tracker.get());
                uses_.pop_back();
            }
        }

        void clear() noexcept
        {
            uses_.clear();
            indices_.clear();
        }

    private:
        std::vector<LegacyTextureStateUse> uses_;
        LegacyTextureStateIndex indices_;
    };

    class LegacyTextureStateInsertionGuard final
    {
    public:
        explicit LegacyTextureStateInsertionGuard(LegacyTextureStateRecording* recording) noexcept
            : recording_(recording), size_(recording != nullptr ? recording->uses().size() : 0)
        {
        }

        ~LegacyTextureStateInsertionGuard()
        {
            if (recording_ != nullptr)
            {
                recording_->truncate(size_);
            }
        }

        LegacyTextureStateInsertionGuard(const LegacyTextureStateInsertionGuard&) = delete;
        LegacyTextureStateInsertionGuard& operator=(const LegacyTextureStateInsertionGuard&) = delete;
        LegacyTextureStateInsertionGuard(LegacyTextureStateInsertionGuard&&) = delete;
        LegacyTextureStateInsertionGuard& operator=(LegacyTextureStateInsertionGuard&&) = delete;

        void commit() noexcept
        {
            recording_ = nullptr;
        }

    private:
        LegacyTextureStateRecording* recording_ = nullptr;
        std::size_t size_ = 0;
    };

    [[nodiscard]] constexpr std::string_view textureLayoutName(TextureLayout layout) noexcept
    {
        switch (layout)
        {
        case TextureLayout::undefined:
            return "undefined";
        case TextureLayout::common:
            return "common";
        case TextureLayout::present:
            return "present";
        case TextureLayout::shaderResource:
            return "shaderResource";
        case TextureLayout::unorderedAccess:
            return "unorderedAccess";
        case TextureLayout::renderTarget:
            return "renderTarget";
        case TextureLayout::depthStencilWrite:
            return "depthStencilWrite";
        case TextureLayout::depthStencilRead:
            return "depthStencilRead";
        case TextureLayout::copySource:
            return "copySource";
        case TextureLayout::copyDestination:
            return "copyDestination";
        case TextureLayout::resolveSource:
            return "resolveSource";
        case TextureLayout::resolveDestination:
            return "resolveDestination";
        case TextureLayout::shadingRateSource:
            return "shadingRateSource";
        }
        return "unknown";
    }

    [[nodiscard]] inline Error textureLayoutMismatch(TextureLayout layoutBefore,
                                                     TextureLayout current,
                                                     std::uint32_t subresource,
                                                     std::string_view where)
    {
        return Error{
            ErrorCode::invalidState,
            0,
            "A texture barrier's layoutBefore (" + std::string(textureLayoutName(layoutBefore)) +
                ") is not the current layout of subresource " + std::to_string(subresource) + ": " +
                std::string(where) + " " + std::string(textureLayoutName(current)) +
                ". Name the current layout, or undefined to discard the contents.",
        };
    }

    [[nodiscard]] inline Status checkTextureLayoutEntries(
        std::span<const LegacyTextureStateRecording* const> recordings)
    {
        struct Simulated final
        {
            const LegacyTextureStateTracker* tracker = nullptr;
            std::vector<std::uint8_t> touched;
            std::vector<TextureLayout> layouts;
        };
        std::vector<Simulated> simulated;
        LegacyTextureStateIndex index;
        for (const auto* recording : recordings)
        {
            for (const auto& use : recording->uses())
            {
                const auto position = index.findOrAppend(simulated, use.tracker.get(), [&] {
                    const auto count = use.tracker->subresourceCount();
                    return Simulated{
                        .tracker = use.tracker.get(),
                        .touched = std::vector<std::uint8_t>(count, 0),
                        .layouts = std::vector<TextureLayout>(count, TextureLayout::undefined),
                    };
                });
                auto& state = simulated[position];
                const std::scoped_lock lock(use.tracker->mutex());
                for (std::uint32_t subresource = 0; subresource < use.tracker->subresourceCount(); ++subresource)
                {
                    if (use.layoutEntryRequired[subresource] != 0)
                    {
                        const auto current = state.touched[subresource] != 0 ? state.layouts[subresource]
                                                                             : use.tracker->layoutLocked(subresource);
                        if (current != externallyUnknownTextureLayout && current != use.layoutEntry[subresource])
                        {
                            return std::unexpected(textureLayoutMismatch(use.layoutEntry[subresource],
                                                                         current,
                                                                         subresource,
                                                                         "the work submitted before it leaves it in"));
                        }
                    }
                    if (use.layoutSet[subresource] != 0)
                    {
                        state.touched[subresource] = 1;
                        state.layouts[subresource] = use.layoutCurrent[subresource];
                    }
                }
            }
        }
        return {};
    }

    class LegacyTextureStateSubmission final
    {
    public:
        void resolve(const LegacyTextureStateRecording& recording,
                     D3D12_RESOURCE_STATES queueStates,
                     std::vector<D3D12_RESOURCE_BARRIER>& barriers)
        {
            for (const auto& use : recording.uses())
            {
                auto& staged = stage(use.tracker);
                const std::scoped_lock lock(use.tracker->mutex());
                use.tracker->reserveLocked();
                use.tracker->reserveLayoutsLocked();
                const auto first = barriers.size();
                for (std::uint32_t subresource = 0; subresource < use.tracker->subresourceCount(); ++subresource)
                {
                    if (use.layoutSet[subresource] != 0)
                    {
                        staged.layoutTouched[subresource] = 1;
                        staged.layouts[subresource] = use.layoutCurrent[subresource];
                    }
                    if (use.uses[subresource] == LegacySubresourceUse::discarded)
                    {
                        const auto submitted = staged.touched[subresource] != 0 ? staged.states[subresource]
                                                                                : use.tracker->stateLocked(subresource);
                        const auto assumed = use.assumed[subresource];
                        if (submitted != assumed && legacyStateFitsQueue(submitted, queueStates) &&
                            legacyStateFitsQueue(assumed, queueStates))
                        {
                            barriers.push_back(D3D12_RESOURCE_BARRIER{
                                .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                                .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
                                .Transition =
                                    {
                                        .pResource = use.tracker->resource(),
                                        .Subresource = subresource,
                                        .StateBefore = submitted,
                                        .StateAfter = assumed,
                                    },
                            });
                        }
                    }
                    if (use.uses[subresource] != LegacySubresourceUse::untouched)
                    {
                        staged.touched[subresource] = 1;
                        staged.states[subresource] = use.current[subresource];
                    }
                }
                collapseWholeResourceTransition(barriers, first, use.tracker->subresourceCount());
            }
        }

        void commit() const noexcept
        {
            for (const auto& staged : staged_)
            {
                const std::scoped_lock lock(staged.tracker->mutex());
                for (std::uint32_t subresource = 0; subresource < staged.tracker->subresourceCount(); ++subresource)
                {
                    if (staged.touched[subresource] != 0)
                    {
                        staged.tracker->setStateLocked(subresource, staged.states[subresource]);
                    }
                    if (staged.layoutTouched[subresource] != 0)
                    {
                        staged.tracker->setLayoutLocked(subresource, staged.layouts[subresource]);
                    }
                }
            }
        }

    private:
        struct StagedStates final
        {
            std::shared_ptr<LegacyTextureStateTracker> tracker;
            std::vector<std::uint8_t> touched;
            std::vector<D3D12_RESOURCE_STATES> states;
            std::vector<std::uint8_t> layoutTouched;
            std::vector<TextureLayout> layouts;
        };

        [[nodiscard]] StagedStates& stage(const std::shared_ptr<LegacyTextureStateTracker>& tracker)
        {
            const auto index = indices_.findOrAppend(staged_, tracker.get(), [&] {
                const auto count = tracker->subresourceCount();
                return StagedStates{
                    .tracker = tracker,
                    .touched = std::vector<std::uint8_t>(count, 0),
                    .states = std::vector<D3D12_RESOURCE_STATES>(count, D3D12_RESOURCE_STATE_COMMON),
                    .layoutTouched = std::vector<std::uint8_t>(count, 0),
                    .layouts = std::vector<TextureLayout>(count, TextureLayout::undefined),
                };
            });
            return staged_[index];
        }

        static void collapseWholeResourceTransition(std::vector<D3D12_RESOURCE_BARRIER>& barriers,
                                                    std::size_t first,
                                                    std::uint32_t subresourceCount) noexcept
        {
            if (subresourceCount < 2 || barriers.size() - first != subresourceCount)
            {
                return;
            }
            const auto& head = barriers[first].Transition;
            const bool uniform = std::all_of(barriers.begin() + static_cast<std::ptrdiff_t>(first),
                                             barriers.end(),
                                             [&](const D3D12_RESOURCE_BARRIER& barrier) {
                                                 return barrier.Transition.StateBefore == head.StateBefore &&
                                                        barrier.Transition.StateAfter == head.StateAfter;
                                             });
            if (!uniform)
            {
                return;
            }
            barriers[first].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barriers.erase(barriers.begin() + static_cast<std::ptrdiff_t>(first) + 1, barriers.end());
        }

        std::vector<StagedStates> staged_;
        LegacyTextureStateIndex indices_;
    };

} // namespace rachis::d3d12
