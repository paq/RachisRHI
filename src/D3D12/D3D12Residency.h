#pragma once

#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Error.h"
#include "Rachis/Resource.h"

#include <d3d12.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace rachis::d3d12
{

    inline constexpr std::size_t residencyQueueCount = 3;

    struct ResidencyRecord final
    {
        mutable std::mutex mutex;
        ID3D12Pageable* pageable = nullptr;
        ResidencyState state = ResidencyState::automatic;
        std::array<std::uint64_t, residencyQueueCount> readyAfter{};
        std::uint64_t useCount = 0;

        bool transitionUnpublishable = false;
    };

    class ResidencyLease final
    {
    public:
        ResidencyLease() noexcept = default;
        ~ResidencyLease();

        ResidencyLease(ResidencyLease&& other) noexcept;
        ResidencyLease& operator=(ResidencyLease&& other) noexcept;

        ResidencyLease(const ResidencyLease&) = delete;
        ResidencyLease& operator=(const ResidencyLease&) = delete;

        [[nodiscard]] ResidencyRecord* record() const noexcept
        {
            return record_.get();
        }

    private:
        explicit ResidencyLease(std::shared_ptr<ResidencyRecord> record) noexcept;
        void release() noexcept;

        std::shared_ptr<ResidencyRecord> record_;

        friend Status appendResidencyLeases(std::vector<ResidencyLease>&,
                                            std::span<const std::shared_ptr<ResidencyRecord>>);
    };

    [[nodiscard]] Status appendResidencyLeases(std::vector<ResidencyLease>& destination,
                                               std::span<const std::shared_ptr<ResidencyRecord>> records);

    struct InPlaceResidencyUse final
    {
        std::shared_ptr<ResidencyRecord> record;
        std::uint32_t count = 0;
    };

    using InPlaceResidencyUses = std::unordered_map<const ResidencyRecord*, InPlaceResidencyUse>;

    class InPlaceDescriptorEpoch final
    {
    public:
        InPlaceDescriptorEpoch() = default;
        ~InPlaceDescriptorEpoch() = default;

        InPlaceDescriptorEpoch(const InPlaceDescriptorEpoch&) = delete;
        InPlaceDescriptorEpoch& operator=(const InPlaceDescriptorEpoch&) = delete;
        InPlaceDescriptorEpoch(InPlaceDescriptorEpoch&&) = delete;
        InPlaceDescriptorEpoch& operator=(InPlaceDescriptorEpoch&&) = delete;

        std::atomic_bool observed = false;
        std::vector<std::shared_ptr<void>> replaced;
    };

    struct ObservedInPlaceDescriptorEpoch final
    {
        std::uint64_t index = 0;
        std::weak_ptr<InPlaceDescriptorEpoch> epoch;
    };

    struct InPlaceDescriptorPublication final
    {
        InPlaceDescriptorPublication() = default;
        InPlaceDescriptorPublication(const InPlaceDescriptorPublication&) = delete;
        InPlaceDescriptorPublication& operator=(const InPlaceDescriptorPublication&) = delete;
        InPlaceDescriptorPublication(InPlaceDescriptorPublication&&) = delete;
        InPlaceDescriptorPublication& operator=(InPlaceDescriptorPublication&&) = delete;

        ~InPlaceDescriptorPublication()
        {
            if (submissions.load(std::memory_order_acquire) != 0)
            {
                test::lockedPublicationDestructionCount().fetch_add(1, std::memory_order_relaxed);
            }
        }

        mutable std::shared_mutex mutex;
        mutable std::atomic_uint32_t submissions = 0;
        InPlaceResidencyUses residencies;
        std::shared_ptr<InPlaceDescriptorEpoch> epoch = std::make_shared<InPlaceDescriptorEpoch>();
        std::vector<ObservedInPlaceDescriptorEpoch> observedEpochs;
        std::uint64_t epochIndex = 0;
    };

    using InPlacePublicationReference = std::shared_ptr<const InPlaceDescriptorPublication>;

} // namespace rachis::d3d12
