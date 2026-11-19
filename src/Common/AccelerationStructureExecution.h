#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rachis::detail
{

    template<typename T>
    inline void deduplicateSharedPointers(std::vector<std::shared_ptr<T>>& values)
    {
        std::erase(values, nullptr);
        if (values.size() <= 1U)
        {
            return;
        }
        const auto lessByAddress = [](const std::shared_ptr<T>& lhs, const std::shared_ptr<T>& rhs) noexcept {
            return std::less<const T*>{}(lhs.get(), rhs.get());
        };
        bool strictlySorted = true;
        for (std::size_t index = 1; index < values.size(); ++index)
        {
            if (!lessByAddress(values[index - 1U], values[index]))
            {
                strictlySorted = false;
                break;
            }
        }
        if (strictlySorted)
        {
            return;
        }
        std::ranges::sort(values, lessByAddress);
        const auto duplicateTail = std::ranges::unique(values);
        values.erase(duplicateTail.begin(), duplicateTail.end());
    }

    class AccelerationStructureRecordingSequence final
    {
    public:
        constexpr AccelerationStructureRecordingSequence() noexcept = default;

        AccelerationStructureRecordingSequence(const AccelerationStructureRecordingSequence&) = delete;
        AccelerationStructureRecordingSequence& operator=(const AccelerationStructureRecordingSequence&) = delete;
        AccelerationStructureRecordingSequence(AccelerationStructureRecordingSequence&&) = delete;
        AccelerationStructureRecordingSequence& operator=(AccelerationStructureRecordingSequence&&) = delete;

        [[nodiscard]] std::uint64_t next() noexcept
        {
            const std::uint64_t id = next_.fetch_add(1, std::memory_order_relaxed);
            return id != 0 ? id : next_.fetch_add(1, std::memory_order_relaxed);
        }

    private:
        std::atomic<std::uint64_t> next_{1};
    };

    enum class AccelerationStructureWrite : std::uint8_t
    {
        publishVersion,
        updateInPlace,
    };

    enum class AccelerationStructureLeaseConflict : std::uint8_t
    {
        producer,
        consumer,
    };

    class AccelerationStructureExecutionTracker final
    {
    public:
        AccelerationStructureExecutionTracker() noexcept = default;
        ~AccelerationStructureExecutionTracker() = default;

        AccelerationStructureExecutionTracker(const AccelerationStructureExecutionTracker&) = delete;
        AccelerationStructureExecutionTracker& operator=(const AccelerationStructureExecutionTracker&) = delete;
        AccelerationStructureExecutionTracker(AccelerationStructureExecutionTracker&&) = delete;
        AccelerationStructureExecutionTracker& operator=(AccelerationStructureExecutionTracker&&) = delete;

        [[nodiscard]] bool canUpdateInPlace(std::uint64_t recordingId) const noexcept
        {
            if (recordingId == 0)
            {
                return false;
            }
            const std::scoped_lock lock(mutex_);
            return !hasForeignConsumers(recordingId) && (activeProducers_ == 0 || producerRecordingId_ == recordingId);
        }

        [[nodiscard]] std::uint64_t activeConsumerCount() const noexcept
        {
            const std::scoped_lock lock(mutex_);
            std::uint64_t count = 0;
            for (const auto& consumer : consumers_)
            {
                count += consumer.count;
            }
            return count;
        }

        [[nodiscard]] std::uint64_t activeProducerCount() const noexcept
        {
            const std::scoped_lock lock(mutex_);
            return activeProducers_;
        }

        [[nodiscard]] std::uint64_t activeInPlaceProducerCount() const noexcept
        {
            const std::scoped_lock lock(mutex_);
            return activeInPlaceProducers_;
        }

        [[nodiscard]] std::uint64_t activeProducerRecordingId() const noexcept
        {
            const std::scoped_lock lock(mutex_);
            return producerRecordingId_;
        }

        [[nodiscard]] bool hasActiveExecutions() const noexcept
        {
            const std::scoped_lock lock(mutex_);
            return !consumers_.empty() || activeProducers_ != 0;
        }

    private:
        friend class AccelerationStructureExecutionLease;

        struct ConsumerRecording final
        {
            std::uint64_t recordingId = 0;
            std::uint64_t count = 0;
        };

        [[nodiscard]] bool consumableBy(std::uint64_t recordingId) const noexcept
        {
            return activeInPlaceProducers_ == 0 || (recordingId != 0 && producerRecordingId_ == recordingId);
        }

        [[nodiscard]] bool hasForeignConsumers(std::uint64_t recordingId) const noexcept
        {
            return std::ranges::any_of(consumers_, [recordingId](const ConsumerRecording& consumer) noexcept {
                return recordingId == 0 || consumer.recordingId != recordingId;
            });
        }

        [[nodiscard]] bool tryAcquireProducer(std::uint64_t recordingId, AccelerationStructureWrite write) noexcept
        {
            if (recordingId == 0)
            {
                return false;
            }
            const std::scoped_lock lock(mutex_);
            if (hasForeignConsumers(recordingId) || (activeProducers_ != 0 && producerRecordingId_ != recordingId))
            {
                return false;
            }
            producerRecordingId_ = recordingId;
            ++activeProducers_;
            if (write == AccelerationStructureWrite::updateInPlace)
            {
                ++activeInPlaceProducers_;
            }
            return true;
        }

        void releaseProducer(AccelerationStructureWrite write) noexcept
        {
            const std::scoped_lock lock(mutex_);
            if (write == AccelerationStructureWrite::updateInPlace && activeInPlaceProducers_ != 0)
            {
                --activeInPlaceProducers_;
            }
            if (activeProducers_ != 0)
            {
                --activeProducers_;
                if (activeProducers_ == 0)
                {
                    producerRecordingId_ = 0;
                }
            }
        }

        [[nodiscard]] bool tryAcquireConsumer(std::uint64_t recordingId)
        {
            const std::scoped_lock lock(mutex_);
            if (!consumableBy(recordingId))
            {
                return false;
            }
            const auto found = std::ranges::find(consumers_, recordingId, &ConsumerRecording::recordingId);
            if (found != consumers_.end())
            {
                ++found->count;
                return true;
            }
            consumers_.push_back(ConsumerRecording{.recordingId = recordingId, .count = 1});
            return true;
        }

        void releaseConsumer(std::uint64_t recordingId) noexcept
        {
            const std::scoped_lock lock(mutex_);
            const auto found = std::ranges::find(consumers_, recordingId, &ConsumerRecording::recordingId);
            if (found == consumers_.end())
            {
                return;
            }
            if (--found->count == 0)
            {
                *found = consumers_.back();
                consumers_.pop_back();
            }
        }

        mutable std::mutex mutex_;
        std::vector<ConsumerRecording> consumers_;
        std::uint64_t activeProducers_ = 0;
        std::uint64_t activeInPlaceProducers_ = 0;
        std::uint64_t producerRecordingId_ = 0;
    };

    class AccelerationStructureExecutionLease final
    {
    public:
        AccelerationStructureExecutionLease() noexcept = default;

        ~AccelerationStructureExecutionLease()
        {
            reset();
        }

        AccelerationStructureExecutionLease(const AccelerationStructureExecutionLease&) = delete;
        AccelerationStructureExecutionLease& operator=(const AccelerationStructureExecutionLease&) = delete;

        AccelerationStructureExecutionLease(AccelerationStructureExecutionLease&& other) noexcept
            : producer_(std::exchange(other.producer_, nullptr)), producerWrite_(other.producerWrite_),
              consumers_(std::move(other.consumers_)),
              consumerRecordingId_(std::exchange(other.consumerRecordingId_, 0))
        {
            other.consumers_.clear();
        }

        AccelerationStructureExecutionLease& operator=(AccelerationStructureExecutionLease&& other) noexcept
        {
            if (this != &other)
            {
                reset();
                producer_ = std::exchange(other.producer_, nullptr);
                producerWrite_ = other.producerWrite_;
                consumers_ = std::move(other.consumers_);
                other.consumers_.clear();
                consumerRecordingId_ = std::exchange(other.consumerRecordingId_, 0);
            }
            return *this;
        }

        [[nodiscard]] static std::expected<AccelerationStructureExecutionLease, AccelerationStructureLeaseConflict>
        tryAcquireProducer(std::shared_ptr<AccelerationStructureExecutionTracker> producer,
                           std::uint64_t recordingId,
                           std::span<const std::shared_ptr<AccelerationStructureExecutionTracker>> consumers = {},
                           AccelerationStructureWrite write = AccelerationStructureWrite::publishVersion)
        {
            if (producer == nullptr)
            {
                return std::unexpected(AccelerationStructureLeaseConflict::producer);
            }
            auto deduplicated = deduplicateTrackers(consumers);
            if (!producer->tryAcquireProducer(recordingId, write))
            {
                return std::unexpected(AccelerationStructureLeaseConflict::producer);
            }
            bool acquired = false;
            try
            {
                acquired = tryAcquireAllConsumers(deduplicated, recordingId);
            }
            catch (...)
            {
                producer->releaseProducer(write);
                throw;
            }
            if (!acquired)
            {
                producer->releaseProducer(write);
                return std::unexpected(AccelerationStructureLeaseConflict::consumer);
            }
            return AccelerationStructureExecutionLease(
                std::move(producer), write, std::move(deduplicated), recordingId);
        }

        [[nodiscard]] static std::optional<AccelerationStructureExecutionLease> tryAcquireConsumers(
            std::span<const std::shared_ptr<AccelerationStructureExecutionTracker>> consumers,
            std::uint64_t recordingId)
        {
            auto deduplicated = deduplicateTrackers(consumers);
            if (!tryAcquireAllConsumers(deduplicated, recordingId))
            {
                return std::nullopt;
            }
            return AccelerationStructureExecutionLease(
                nullptr, AccelerationStructureWrite::publishVersion, std::move(deduplicated), recordingId);
        }

        [[nodiscard]] static std::optional<AccelerationStructureExecutionLease> tryAcquireConsumer(
            std::shared_ptr<AccelerationStructureExecutionTracker> consumer, std::uint64_t recordingId)
        {
            if (consumer == nullptr)
            {
                return AccelerationStructureExecutionLease{};
            }
            const std::array trackers{std::move(consumer)};
            return tryAcquireConsumers(trackers, recordingId);
        }

        [[nodiscard]] bool active() const noexcept
        {
            return producer_ != nullptr || !consumers_.empty();
        }

        void reset() noexcept
        {
            releaseConsumers(consumers_, consumerRecordingId_);
            consumers_.clear();
            consumerRecordingId_ = 0;
            if (producer_ != nullptr)
            {
                producer_->releaseProducer(producerWrite_);
                producer_.reset();
            }
        }

    private:
        AccelerationStructureExecutionLease(
            std::shared_ptr<AccelerationStructureExecutionTracker> producer,
            AccelerationStructureWrite producerWrite,
            std::vector<std::shared_ptr<AccelerationStructureExecutionTracker>> consumers,
            std::uint64_t consumerRecordingId) noexcept
            : producer_(std::move(producer)), producerWrite_(producerWrite), consumers_(std::move(consumers)),
              consumerRecordingId_(consumerRecordingId)
        {
        }

        [[nodiscard]] static std::vector<std::shared_ptr<AccelerationStructureExecutionTracker>> deduplicateTrackers(
            std::span<const std::shared_ptr<AccelerationStructureExecutionTracker>> trackers)
        {
            std::vector<std::shared_ptr<AccelerationStructureExecutionTracker>> unique(trackers.begin(),
                                                                                       trackers.end());
            deduplicateSharedPointers(unique);
            return unique;
        }

        static void releaseConsumers(std::span<const std::shared_ptr<AccelerationStructureExecutionTracker>> consumers,
                                     std::uint64_t recordingId) noexcept
        {
            for (const auto& consumer : consumers)
            {
                if (consumer != nullptr)
                {
                    consumer->releaseConsumer(recordingId);
                }
            }
        }

        [[nodiscard]] static bool tryAcquireAllConsumers(
            std::span<const std::shared_ptr<AccelerationStructureExecutionTracker>> consumers,
            std::uint64_t recordingId)
        {
            std::size_t acquired = 0;
            try
            {
                for (; acquired < consumers.size(); ++acquired)
                {
                    if (!consumers[acquired]->tryAcquireConsumer(recordingId))
                    {
                        releaseConsumers(consumers.first(acquired), recordingId);
                        return false;
                    }
                }
            }
            catch (...)
            {
                releaseConsumers(consumers.first(acquired), recordingId);
                throw;
            }
            return true;
        }

        std::shared_ptr<AccelerationStructureExecutionTracker> producer_;
        AccelerationStructureWrite producerWrite_ = AccelerationStructureWrite::publishVersion;
        std::vector<std::shared_ptr<AccelerationStructureExecutionTracker>> consumers_;
        std::uint64_t consumerRecordingId_ = 0;
    };

} // namespace rachis::detail
