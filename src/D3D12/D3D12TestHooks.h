#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace rachis
{
    class CommandBuffer;
}

namespace rachis::d3d12::test
{

    struct CommandDependencySnapshot final
    {
        std::size_t resourceOwners = 0;
        std::size_t residencyLeases = 0;
        std::size_t swapchainReferences = 0;
        std::size_t sparseMappingDependencies = 0;
        std::size_t textureLayoutUses = 0;

        [[nodiscard]] friend constexpr bool operator==(const CommandDependencySnapshot&,
                                                       const CommandDependencySnapshot&) noexcept = default;
    };

    [[nodiscard]] std::optional<CommandDependencySnapshot> commandDependencySnapshot(
        const CommandBuffer& commandBuffer) noexcept;

    struct DebugLayerProcessSnapshot final
    {
        bool debugLayerEnabled = false;
        bool gpuValidationEnabled = false;
        std::uint64_t devicesWithoutDebugLayer = 0;
        std::uint64_t devicesWithoutGpuValidation = 0;
    };

    [[nodiscard]] DebugLayerProcessSnapshot debugLayerProcessSnapshot() noexcept;

    [[nodiscard]] bool debugLayerProcessLockAvailable() noexcept;

    using NativeDeviceCreationObserver = void (*)() noexcept;

    [[nodiscard]] inline std::atomic<NativeDeviceCreationObserver>& nativeDeviceCreationObserver() noexcept
    {
        static std::atomic<NativeDeviceCreationObserver> s_value = nullptr;
        return s_value;
    }

    using InPlaceSubmissionObserver = void (*)() noexcept;

    [[nodiscard]] inline std::atomic<InPlaceSubmissionObserver>& inPlaceSubmissionObserver() noexcept
    {
        static std::atomic<InPlaceSubmissionObserver> s_value = nullptr;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_uint64_t& inPlacePublicationContentionCount() noexcept
    {
        static std::atomic_uint64_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_bool& nextSubmissionCloseFailure() noexcept
    {
        static std::atomic_bool s_value = false;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_uint64_t& lockedPublicationDestructionCount() noexcept
    {
        static std::atomic_uint64_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_uint64_t& bindingSetVersionCopyCount() noexcept
    {
        static std::atomic_uint64_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_bool& forceLegacyBarriers() noexcept
    {
        static std::atomic_bool s_value = false;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_bool& forceFullyTypedFormatCastingUnsupported() noexcept
    {
        static std::atomic_bool s_value = false;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_int32_t& nextPresentResult() noexcept
    {
        static std::atomic_int32_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_int32_t& nextResizeBuffersResult() noexcept
    {
        static std::atomic_int32_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_int32_t& nextSwapchainGetBufferResult() noexcept
    {
        static std::atomic_int32_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_uint64_t& nativeSwapchainsReleasedWhileReferenced() noexcept
    {
        static std::atomic_uint64_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_uint64_t& postbuildQueryResetBarrierCount() noexcept
    {
        static std::atomic_uint64_t s_value = 0;
        return s_value;
    }

    inline void notePostbuildQueryResetBarrier() noexcept
    {
        postbuildQueryResetBarrierCount().fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] inline std::atomic_bool& nextSwapchainWaitDeviceLoss() noexcept
    {
        static std::atomic_bool s_value = false;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_bool& nextQueueSignalReservationFailure() noexcept
    {
        static std::atomic_bool s_value = false;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_bool& nextSparseMappingSignalFailure() noexcept
    {
        static std::atomic_bool s_value = false;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_bool& nextLegacyOrderedStateCopyFailure() noexcept
    {
        static std::atomic_bool s_value = false;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_bool& nextPipelineCacheWithoutLibrary1() noexcept
    {
        static std::atomic_bool s_value = false;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_bool& nextPipelineCacheEntryRecordFailure() noexcept
    {
        static std::atomic_bool s_value = false;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_int32_t& swapchainLeaseRetentionFailureCountdown() noexcept
    {
        static std::atomic_int32_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_int32_t& resourceRetentionFailureCountdown() noexcept
    {
        static std::atomic_int32_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_uint64_t& indirectExecutionRecordCount() noexcept
    {
        static std::atomic_uint64_t s_value = 0;
        return s_value;
    }

    inline void noteIndirectExecutionRecorded() noexcept
    {
        indirectExecutionRecordCount().fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] inline std::atomic_uint64_t& resourceCommandRecordCount() noexcept
    {
        static std::atomic_uint64_t s_value = 0;
        return s_value;
    }

    inline void noteResourceCommandRecorded() noexcept
    {
        resourceCommandRecordCount().fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] inline std::atomic_uint64_t& accelerationStructureRecordCount() noexcept
    {
        static std::atomic_uint64_t s_value = 0;
        return s_value;
    }

    inline void noteAccelerationStructureRecorded() noexcept
    {
        accelerationStructureRecordCount().fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] inline std::atomic_uint64_t& bufferFillChunkElementLimit() noexcept
    {
        static std::atomic_uint64_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_uint64_t& bufferFillChunkBarrierCount() noexcept
    {
        static std::atomic_uint64_t s_value = 0;
        return s_value;
    }

    inline void noteBufferFillChunkBarrier() noexcept
    {
        bufferFillChunkBarrierCount().fetch_add(1, std::memory_order_relaxed);
    }

    struct StorageTextureClearSnapshot final
    {
        bool floatPath = false;
        std::array<std::uint32_t, 4> bits{};
    };

    [[nodiscard]] inline StorageTextureClearSnapshot& lastStorageTextureClear() noexcept
    {
        static thread_local StorageTextureClearSnapshot s_value;
        return s_value;
    }

    struct GlobalBarrierSnapshot final
    {
        std::uint32_t syncBefore = 0;
        std::uint32_t syncAfter = 0;
        std::uint32_t accessBefore = 0;
        std::uint32_t accessAfter = 0;
        bool legacyUnorderedAccess = false;
    };

    [[nodiscard]] inline std::optional<GlobalBarrierSnapshot>& lastAccelerationStructureGlobalBarrier() noexcept
    {
        static thread_local std::optional<GlobalBarrierSnapshot> s_value;
        return s_value;
    }

    [[nodiscard]] inline std::optional<GlobalBarrierSnapshot>& lastAccelerationStructureInPlaceUpdateBarrier() noexcept
    {
        static thread_local std::optional<GlobalBarrierSnapshot> s_value;
        return s_value;
    }

    [[nodiscard]] inline bool consumeResourceRetentionFailure() noexcept
    {
        auto remaining = resourceRetentionFailureCountdown().load(std::memory_order_relaxed);
        while (remaining > 0)
        {
            if (resourceRetentionFailureCountdown().compare_exchange_weak(
                    remaining, remaining - 1, std::memory_order_relaxed, std::memory_order_relaxed))
            {
                return remaining == 1;
            }
        }
        return false;
    }

    [[nodiscard]] inline std::atomic_bool& nextDeviceTokenExhaustion() noexcept
    {
        static std::atomic_bool s_value = false;
        return s_value;
    }

    enum class MakeResidentFailurePoint : std::int32_t
    {
        none = 0,
        beforeNativeAcceptance = 1,
        afterNativeAcceptance = 2,
        afterQueueWait = 3,
        afterQueueSignal = 4,
    };

    [[nodiscard]] inline std::atomic_int32_t& nextMakeResidentFailurePoint() noexcept
    {
        static std::atomic_int32_t s_value = static_cast<std::int32_t>(MakeResidentFailurePoint::none);
        return s_value;
    }

    [[nodiscard]] inline bool consumeMakeResidentFailure(MakeResidentFailurePoint point) noexcept
    {
        auto expected = static_cast<std::int32_t>(point);
        return nextMakeResidentFailurePoint().compare_exchange_strong(
            expected, static_cast<std::int32_t>(MakeResidentFailurePoint::none), std::memory_order_relaxed);
    }

    [[nodiscard]] inline std::atomic_int64_t& activeMakeResidentQuarantines() noexcept
    {
        static std::atomic_int64_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_uint64_t& prematureMakeResidentQuarantineReleases() noexcept
    {
        static std::atomic_uint64_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::optional<std::uint64_t>& nextResourceMemoryIdentitySeed() noexcept
    {
        static thread_local std::optional<std::uint64_t> s_value;
        return s_value;
    }

    [[nodiscard]] inline std::uint64_t consumeNextResourceMemoryIdentitySeed() noexcept
    {
        auto& seed = nextResourceMemoryIdentitySeed();
        const auto value = seed.value_or(1);
        seed.reset();
        return value;
    }

} // namespace rachis::d3d12::test
