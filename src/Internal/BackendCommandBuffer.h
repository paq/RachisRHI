#pragma once

#include "Rachis/Command.h"
#include "Rachis/Pipeline.h"
#include "Rachis/Query.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace rachis::detail
{

    template<std::size_t MaxSlots>
    class FixedBoolMask final
    {
        static_assert(MaxSlots >= 1 && MaxSlots <= 8);

    public:
        class BitReference final
        {
        public:
            constexpr BitReference(std::uint8_t& bits, std::uint8_t mask) noexcept : bits_(&bits), mask_(mask)
            {
            }

            constexpr BitReference& operator=(bool value) noexcept
            {
                if (value)
                {
                    *bits_ = static_cast<std::uint8_t>(*bits_ | mask_);
                }
                else
                {
                    *bits_ = static_cast<std::uint8_t>(*bits_ & static_cast<std::uint8_t>(~mask_));
                }
                return *this;
            }

            constexpr BitReference& operator=(const BitReference& other) noexcept
            {
                return *this = static_cast<bool>(other);
            }

            [[nodiscard]] constexpr operator bool() const noexcept
            {
                return (*bits_ & mask_) != 0U;
            }

        private:
            std::uint8_t* bits_;
            std::uint8_t mask_;
        };

        class ConstIterator final
        {
        public:
            using iterator_concept = std::random_access_iterator_tag;
            using iterator_category = std::input_iterator_tag;
            using value_type = bool;
            using difference_type = std::ptrdiff_t;
            using reference = bool;

            constexpr ConstIterator() noexcept = default;
            constexpr ConstIterator(std::uint8_t bits, std::size_t index) noexcept : bits_(bits), index_(index)
            {
            }

            [[nodiscard]] constexpr bool operator*() const noexcept
            {
                return ((bits_ >> index_) & 1U) != 0U;
            }

            [[nodiscard]] constexpr bool operator[](difference_type offset) const noexcept
            {
                return *(*this + offset);
            }

            constexpr ConstIterator& operator++() noexcept
            {
                ++index_;
                return *this;
            }

            constexpr ConstIterator operator++(int) noexcept
            {
                const auto copy = *this;
                ++(*this);
                return copy;
            }

            constexpr ConstIterator& operator--() noexcept
            {
                --index_;
                return *this;
            }

            constexpr ConstIterator operator--(int) noexcept
            {
                const auto copy = *this;
                --(*this);
                return copy;
            }

            constexpr ConstIterator& operator+=(difference_type n) noexcept
            {
                index_ = static_cast<std::size_t>(static_cast<difference_type>(index_) + n);
                return *this;
            }

            constexpr ConstIterator& operator-=(difference_type n) noexcept
            {
                index_ = static_cast<std::size_t>(static_cast<difference_type>(index_) - n);
                return *this;
            }

            [[nodiscard]] friend constexpr ConstIterator operator+(ConstIterator it, difference_type n) noexcept
            {
                it += n;
                return it;
            }

            [[nodiscard]] friend constexpr ConstIterator operator+(difference_type n, ConstIterator it) noexcept
            {
                it += n;
                return it;
            }

            [[nodiscard]] friend constexpr ConstIterator operator-(ConstIterator it, difference_type n) noexcept
            {
                it -= n;
                return it;
            }

            [[nodiscard]] friend constexpr difference_type operator-(ConstIterator lhs, ConstIterator rhs) noexcept
            {
                return static_cast<difference_type>(lhs.index_) - static_cast<difference_type>(rhs.index_);
            }

            [[nodiscard]] friend constexpr bool operator==(const ConstIterator& lhs, const ConstIterator& rhs) noexcept
            {
                return lhs.index_ == rhs.index_;
            }

            [[nodiscard]] friend constexpr auto operator<=>(const ConstIterator& lhs, const ConstIterator& rhs) noexcept
            {
                return lhs.index_ <=> rhs.index_;
            }

        private:
            std::uint8_t bits_ = 0;
            std::size_t index_ = 0;
        };

        constexpr FixedBoolMask() noexcept = default;

        constexpr FixedBoolMask(std::initializer_list<bool> values) noexcept
        {
            assert(values.size() <= MaxSlots);
            for (const bool value : values)
            {
                if (size_ >= MaxSlots)
                {
                    break;
                }
                push_back(value);
            }
        }

        constexpr void clear() noexcept
        {
            bits_ = 0;
            size_ = 0;
        }

        constexpr void reserve(std::size_t) noexcept
        {
        }

        constexpr void assign(std::size_t count, bool value) noexcept
        {
            assert(count <= MaxSlots);
            const auto clamped = count <= MaxSlots ? count : MaxSlots;
            size_ = static_cast<std::uint8_t>(clamped);
            if (!value || clamped == 0)
            {
                bits_ = 0;
            }
            else
            {
                bits_ = clamped == 8 ? std::uint8_t{0xFFU} : static_cast<std::uint8_t>((1U << clamped) - 1U);
            }
        }

        constexpr void push_back(bool value) noexcept
        {
            assert(size_ < MaxSlots);
            if (size_ < MaxSlots)
            {
                const auto mask = static_cast<std::uint8_t>(1U << size_);
                if (value)
                {
                    bits_ = static_cast<std::uint8_t>(bits_ | mask);
                }
                else
                {
                    bits_ = static_cast<std::uint8_t>(bits_ & static_cast<std::uint8_t>(~mask));
                }
                ++size_;
            }
        }

        [[nodiscard]] constexpr std::size_t size() const noexcept
        {
            return size_;
        }

        [[nodiscard]] constexpr bool empty() const noexcept
        {
            return size_ == 0;
        }

        [[nodiscard]] constexpr bool all() const noexcept
        {
            const auto mask = size_ == 8 ? std::uint8_t{0xFFU} : static_cast<std::uint8_t>((1U << size_) - 1U);
            return (bits_ & mask) == mask;
        }

        [[nodiscard]] constexpr bool operator[](std::size_t index) const noexcept
        {
            assert(index < size_);
            return ((bits_ >> index) & 1U) != 0U;
        }

        [[nodiscard]] constexpr BitReference operator[](std::size_t index) noexcept
        {
            assert(index < size_);
            return BitReference(bits_, static_cast<std::uint8_t>(1U << index));
        }

        [[nodiscard]] constexpr ConstIterator begin() const noexcept
        {
            return ConstIterator(bits_, 0);
        }
        [[nodiscard]] constexpr ConstIterator end() const noexcept
        {
            return ConstIterator(bits_, size_);
        }
        [[nodiscard]] constexpr ConstIterator cbegin() const noexcept
        {
            return begin();
        }
        [[nodiscard]] constexpr ConstIterator cend() const noexcept
        {
            return end();
        }

        [[nodiscard]] friend constexpr bool operator==(const FixedBoolMask& left, const FixedBoolMask& right) noexcept
        {
            if (left.size_ != right.size_)
            {
                return false;
            }
            const auto mask =
                left.size_ == 8 ? std::uint8_t{0xFFU} : static_cast<std::uint8_t>((1U << left.size_) - 1U);
            return (left.bits_ & mask) == (right.bits_ & mask);
        }

    private:
        std::uint8_t bits_ = 0;
        std::uint8_t size_ = 0;
    };

    using BindingSetMask = FixedBoolMask<maxBindingSets>;
    using VertexBufferMask = FixedBoolMask<maxVertexBuffers>;
    using StreamOutputBufferMask = FixedBoolMask<maxStreamOutputBuffers>;

    static_assert(sizeof(BindingSetMask) == 2);
    static_assert(sizeof(VertexBufferMask) == 2);
    static_assert(sizeof(StreamOutputBufferMask) == 2);

    struct ComputeCommandState final
    {
        std::shared_ptr<void> pipelineOwner;
        BindingSetMask boundBindingSets;
    };

    struct RayTracingCommandState final
    {
        std::shared_ptr<void> pipelineOwner;
        BindingSetMask boundBindingSets;
    };

    struct GraphicsCommandState final
    {
        std::shared_ptr<void> renderingOwner;
        std::shared_ptr<void> pipelineOwner;
        BindingSetMask boundBindingSets;
        VertexBufferMask boundVertexBuffers;
        StreamOutputBufferMask boundStreamOutputBuffers;
        bool indexBufferBound = false;
        std::uint32_t viewportCount = 0;
        std::uint32_t scissorCount = 0;
        bool blendConstantsRequired = false;
        bool blendConstantsSet = false;
        bool depthBiasRequired = false;
        bool depthBiasSet = false;
        bool depthBoundsRequired = false;
        bool depthBoundsSet = false;
        bool stencilReferenceRequired = false;
        bool stencilReferenceSet = false;
        std::uint8_t stencilReferenceFacesSet = 0;
        std::uint8_t frontStencilReference = 0;
        std::uint8_t backStencilReference = 0;
    };

    struct QueryWrite final
    {
        QueryPoolHandle queryPool;
        std::uint32_t queryIndex = 0;
        bool resolved = false;
    };

    struct ActiveQuery final
    {
        QueryPoolHandle queryPool;
        std::uint32_t queryIndex = 0;

        std::shared_ptr<void> owner;
        bool begunInRendering = false;
    };

    struct ActivePassTimestampWrites final
    {
        QueryPoolHandle queryPool;
        std::uint32_t endOfPassWriteIndex = timestampWriteUnused;
        std::shared_ptr<void> owner;
    };

    struct ActiveGpuProfileScope final
    {
        QueryPoolHandle queryPool;
        std::uint32_t endQueryIndex = 0;

        std::shared_ptr<void> owner;
    };

    struct ActiveConditionalExecution final
    {
        std::shared_ptr<void> owner;
    };

    struct QueryCommandState final
    {
        std::vector<QueryWrite> timestampWrites;
        std::vector<QueryWrite> occlusionWrites;
        std::vector<QueryWrite> pipelineStatisticsWrites;
        std::vector<QueryWrite> streamOutputStatisticsWrites;
        std::vector<QueryWrite> accelerationStructureCompactedSizeWrites;
        std::vector<QueryWrite> accelerationStructureSerializationInfoWrites;
        std::optional<ActiveQuery> activeOcclusion;
        std::optional<ActiveQuery> activePipelineStatistics;
        std::array<std::optional<ActiveQuery>, maxStreamOutputStreams> activeStreamOutputStatistics;
        std::optional<ActivePassTimestampWrites> activePassTimestampWrites;
        std::optional<ActiveConditionalExecution> activeConditionalExecution;
        std::array<ActiveGpuProfileScope, maxDebugGroupDepth> activeGpuProfileScopes;
        std::uint32_t activeGpuProfileScopeDepth = 0;

        void reset() noexcept
        {
            timestampWrites.clear();
            occlusionWrites.clear();
            pipelineStatisticsWrites.clear();
            streamOutputStatisticsWrites.clear();
            accelerationStructureCompactedSizeWrites.clear();
            accelerationStructureSerializationInfoWrites.clear();
            activeOcclusion.reset();
            activePipelineStatistics.reset();
            for (auto& active : activeStreamOutputStatistics)
            {
                active.reset();
            }
            activePassTimestampWrites.reset();
            activeConditionalExecution.reset();
            for (auto& scope : activeGpuProfileScopes)
            {
                scope = {};
            }
            activeGpuProfileScopeDepth = 0;
        }
    };

    enum class DebugGroupOwner : std::uint8_t
    {
        user,
        gpuProfile,
    };

    struct DebugCommandState final
    {
        std::thread::id groupThread;
        std::uint32_t groupDepth = 0;
        std::array<DebugGroupOwner, maxDebugGroupDepth> groupOwners{};

        void reset() noexcept
        {
            groupThread = {};
            groupDepth = 0;
            groupOwners.fill(DebugGroupOwner::user);
        }
    };

    enum class BackendCommandBufferKind : std::uint8_t
    {
        d3d12,
        vulkan,
        metal,
        custom,
    };

    class RecordingStateLock final
    {
    public:
        RecordingStateLock() noexcept = default;

        explicit RecordingStateLock(std::shared_ptr<std::mutex> mutex) : mutex_(std::move(mutex))
        {
            if (mutex_ != nullptr)
            {
                lock_ = std::unique_lock<std::mutex>(*mutex_);
            }
        }

        ~RecordingStateLock() = default;
        RecordingStateLock(RecordingStateLock&&) noexcept = default;
        RecordingStateLock& operator=(RecordingStateLock&&) = delete;
        RecordingStateLock(const RecordingStateLock&) = delete;
        RecordingStateLock& operator=(const RecordingStateLock&) = delete;

        template<typename Pool>
        [[nodiscard]] static RecordingStateLock lockPool(std::shared_ptr<Pool> pool)
        {
            if (pool == nullptr)
            {
                return {};
            }
            auto* const mutex = &pool->mutex;
            return RecordingStateLock(std::shared_ptr<std::mutex>(std::move(pool), mutex));
        }

        [[nodiscard]] bool ownsLock() const noexcept
        {
            return lock_.owns_lock();
        }

    private:
        std::shared_ptr<std::mutex> mutex_;
        std::unique_lock<std::mutex> lock_;
    };

    class BackendCommandBuffer
    {
    public:
        virtual ~BackendCommandBuffer() = default;

        BackendCommandBuffer(const BackendCommandBuffer&) = delete;
        BackendCommandBuffer& operator=(const BackendCommandBuffer&) = delete;
        BackendCommandBuffer(BackendCommandBuffer&&) = delete;
        BackendCommandBuffer& operator=(BackendCommandBuffer&&) = delete;

        [[nodiscard]] BackendCommandBufferKind kind() const noexcept
        {
            return kind_;
        }
        [[nodiscard]] virtual QueueType queueType() const noexcept = 0;
        [[nodiscard]] virtual std::uint64_t deviceToken() const noexcept = 0;
        [[nodiscard]] virtual RecordingStateLock lockRecordingState() const
        {
            return {};
        }

        [[nodiscard]] ComputeCommandState& computeState() noexcept
        {
            return computeState_;
        }
        [[nodiscard]] const ComputeCommandState& computeState() const noexcept
        {
            return computeState_;
        }
        [[nodiscard]] RayTracingCommandState& rayTracingState() noexcept
        {
            return rayTracingState_;
        }
        [[nodiscard]] const RayTracingCommandState& rayTracingState() const noexcept
        {
            return rayTracingState_;
        }
        [[nodiscard]] GraphicsCommandState& graphicsState() noexcept
        {
            return graphicsState_;
        }
        [[nodiscard]] const GraphicsCommandState& graphicsState() const noexcept
        {
            return graphicsState_;
        }
        [[nodiscard]] QueryCommandState& queryState() noexcept
        {
            return queryState_;
        }
        [[nodiscard]] const QueryCommandState& queryState() const noexcept
        {
            return queryState_;
        }
        [[nodiscard]] DebugCommandState& debugState() noexcept
        {
            return debugState_;
        }
        [[nodiscard]] const DebugCommandState& debugState() const noexcept
        {
            return debugState_;
        }

    protected:
        explicit BackendCommandBuffer(BackendCommandBufferKind kind = BackendCommandBufferKind::custom) noexcept
            : kind_(kind)
        {
        }

        void resetState() noexcept
        {
            computeState_ = {};
            rayTracingState_ = {};
            graphicsState_ = {};
            queryState_.reset();
            debugState_.reset();
        }

    private:
        BackendCommandBufferKind kind_ = BackendCommandBufferKind::custom;
        ComputeCommandState computeState_;
        RayTracingCommandState rayTracingState_;
        GraphicsCommandState graphicsState_;
        QueryCommandState queryState_;
        DebugCommandState debugState_;
    };

} // namespace rachis::detail
