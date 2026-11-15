#include "D3D12/D3D12Residency.h"

#include "D3D12/D3D12RuntimePolicies.h"

#include <algorithm>
#include <iterator>
#include <utility>

namespace rachis::d3d12
{

    ResidencyLease::ResidencyLease(std::shared_ptr<ResidencyRecord> record) noexcept : record_(std::move(record))
    {
    }

    ResidencyLease::~ResidencyLease()
    {
        release();
    }

    ResidencyLease::ResidencyLease(ResidencyLease&& other) noexcept : record_(std::move(other.record_))
    {
    }

    ResidencyLease& ResidencyLease::operator=(ResidencyLease&& other) noexcept
    {
        if (this != &other)
        {
            release();
            record_ = std::move(other.record_);
        }
        return *this;
    }

    void ResidencyLease::release() noexcept
    {
        if (record_ == nullptr)
        {
            return;
        }
        auto record = std::move(record_);
        {
            const std::scoped_lock lock(record->mutex);
            if (record->useCount != 0)
            {
                --record->useCount;
            }
        }
    }

    Status appendResidencyLeases(std::vector<ResidencyLease>& destination,
                                 std::span<const std::shared_ptr<ResidencyRecord>> records)
    {
        SmallVector<std::shared_ptr<ResidencyRecord>, 4> unique;
        for (const auto& record : records)
        {
            if (record != nullptr)
            {
                unique.push_back(record);
            }
        }
        if (unique.empty())
        {
            return {};
        }
        const auto address = [](const std::shared_ptr<ResidencyRecord>& record) { return record.get(); };
        std::ranges::sort(unique, {}, address);
        const auto duplicates = std::ranges::unique(unique, {}, address);
        unique.truncate(static_cast<std::size_t>(duplicates.begin() - unique.begin()));

        SmallVector<std::uint8_t, 4> alreadyRetained;
        alreadyRetained.reserve(unique.size());
        for (std::size_t index = 0; index < unique.size(); ++index)
        {
            alreadyRetained.push_back(0);
        }
        for (const auto& lease : destination)
        {
            const auto found = std::ranges::lower_bound(unique, lease.record(), {}, address);
            if (found != unique.end() && found->get() == lease.record())
            {
                alreadyRetained[static_cast<std::size_t>(found - unique.begin())] = 1;
            }
        }

        SmallVector<ResidencyLease, 4> acquired;
        acquired.reserve(unique.size());
        for (std::size_t index = 0; index < unique.size(); ++index)
        {
            if (alreadyRetained[index] != 0)
            {
                continue;
            }
            auto& record = unique[index];
            const std::scoped_lock lock(record->mutex);
            const bool transitionReady =
                record->state != ResidencyState::pending ||
                std::ranges::any_of(record->readyAfter, [](std::uint64_t value) { return value != 0; });
            if (record->state == ResidencyState::evicted || record->transitionUnpublishable || !transitionReady)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    record->transitionUnpublishable
                        ? "A native make-resident transition was accepted without a publishable completion point."
                        : "A manually managed resource must be resident or have a scheduled make-resident transition "
                          "before use.",
                });
            }
            ++record->useCount;
            acquired.emplace_back(ResidencyLease(std::move(record)));
        }
        destination.insert(
            destination.end(), std::make_move_iterator(acquired.begin()), std::make_move_iterator(acquired.end()));
        return {};
    }

} // namespace rachis::d3d12
