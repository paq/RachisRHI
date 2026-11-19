#pragma once

#include "Common/AccelerationStructureExecution.h"

#include <memory>
#include <vector>

namespace rachis::d3d12
{

    using detail::AccelerationStructureExecutionLease;
    using detail::AccelerationStructureExecutionTracker;
    using detail::AccelerationStructureLeaseConflict;
    using detail::AccelerationStructureRecordingSequence;
    using detail::AccelerationStructureWrite;
    using detail::deduplicateSharedPointers;

    struct AccelerationStructureRecord;
    struct AccelerationStructureVersion;

    struct AccelerationStructurePublication final
    {
        std::weak_ptr<AccelerationStructureRecord> record;
        std::weak_ptr<AccelerationStructureVersion> published;
        std::shared_ptr<AccelerationStructureVersion> previous;
    };

    void rollBackAccelerationStructurePublications(
        std::vector<AccelerationStructurePublication>& publications) noexcept;

} // namespace rachis::d3d12
