#pragma once

#include "Common/InPlaceBindingRetention.h"
#include "Rachis/Error.h"
#include "Rachis/Resource.h"

#include <d3d12.h>

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
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

    using InPlaceResidencyLog = detail::InPlaceBindingRetention<std::shared_ptr<ResidencyRecord>>;

    struct InPlaceDescriptorPublication final
    {
        mutable std::shared_mutex mutex;
        InPlaceResidencyLog residencies;
    };

    struct InPlaceResidencyReference final
    {
        std::shared_ptr<const InPlaceDescriptorPublication> publication;
        std::uint64_t sequence = 0;
    };

} // namespace rachis::d3d12
