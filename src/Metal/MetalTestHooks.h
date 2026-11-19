#pragma once

#include "Internal/MetalDiagnostics.h"
#include "Internal/MetalMacosProfile.h"
#include "Rachis/Command.h"
#include "Rachis/Presentation.h"
#include "Rachis/Resource.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

namespace rachis::metal
{

#ifdef RACHIS_TESTING
    void setHostReportOverride(std::optional<MetalHostReport> report) noexcept;
    void setFailNextCommandApiCreate(bool fail) noexcept;
    void setFailNextLogStateCreate(bool fail) noexcept;
    void setFailNextDepthStencilStateCreate(bool fail) noexcept;
    void setFailNextDrawable(bool fail) noexcept;
    void setFailNextResidencySetCreate(bool fail) noexcept;
    void setFailNextPipelineCacheSerialize(bool fail) noexcept;
    void setFailNextDrawIndexTableGrowth(bool fail) noexcept;
    void setFailPushConstantSliceAfter(std::int64_t successes) noexcept;
    void setPipelineCacheTemporaryDirectory(std::string_view directory);

    [[nodiscard]] inline std::recursive_mutex& testHookMutex() noexcept
    {
        static std::recursive_mutex s_mutex;
        return s_mutex;
    }

    class ScopedHostReportOverride final
    {
    public:
        ScopedHostReportOverride() noexcept : lock_(testHookMutex())
        {
            setHostReportOverride(std::nullopt);
        }

        explicit ScopedHostReportOverride(std::optional<MetalHostReport> report) noexcept : lock_(testHookMutex())
        {
            setHostReportOverride(std::move(report));
        }

        ~ScopedHostReportOverride() noexcept
        {
            setHostReportOverride(std::nullopt);
        }

        ScopedHostReportOverride(const ScopedHostReportOverride&) = delete;
        ScopedHostReportOverride& operator=(const ScopedHostReportOverride&) = delete;
        ScopedHostReportOverride(ScopedHostReportOverride&&) = delete;
        ScopedHostReportOverride& operator=(ScopedHostReportOverride&&) = delete;

    private:
        std::unique_lock<std::recursive_mutex> lock_;
    };

    class ScopedFailNextCommandApiCreate final
    {
    public:
        explicit ScopedFailNextCommandApiCreate(bool fail = true) noexcept : lock_(testHookMutex())
        {
            setFailNextCommandApiCreate(fail);
        }

        ~ScopedFailNextCommandApiCreate() noexcept
        {
            setFailNextCommandApiCreate(false);
        }

        ScopedFailNextCommandApiCreate(const ScopedFailNextCommandApiCreate&) = delete;
        ScopedFailNextCommandApiCreate& operator=(const ScopedFailNextCommandApiCreate&) = delete;
        ScopedFailNextCommandApiCreate(ScopedFailNextCommandApiCreate&&) = delete;
        ScopedFailNextCommandApiCreate& operator=(ScopedFailNextCommandApiCreate&&) = delete;

    private:
        std::unique_lock<std::recursive_mutex> lock_;
    };

    class ScopedFailNextLogStateCreate final
    {
    public:
        explicit ScopedFailNextLogStateCreate(bool fail = true) noexcept : lock_(testHookMutex())
        {
            setFailNextLogStateCreate(fail);
        }

        ~ScopedFailNextLogStateCreate() noexcept
        {
            setFailNextLogStateCreate(false);
        }

        ScopedFailNextLogStateCreate(const ScopedFailNextLogStateCreate&) = delete;
        ScopedFailNextLogStateCreate& operator=(const ScopedFailNextLogStateCreate&) = delete;
        ScopedFailNextLogStateCreate(ScopedFailNextLogStateCreate&&) = delete;
        ScopedFailNextLogStateCreate& operator=(ScopedFailNextLogStateCreate&&) = delete;

    private:
        std::unique_lock<std::recursive_mutex> lock_;
    };

    class ScopedFailNextDepthStencilStateCreate final
    {
    public:
        explicit ScopedFailNextDepthStencilStateCreate(bool fail = true) noexcept : lock_(testHookMutex())
        {
            setFailNextDepthStencilStateCreate(fail);
        }

        ~ScopedFailNextDepthStencilStateCreate() noexcept
        {
            setFailNextDepthStencilStateCreate(false);
        }

        ScopedFailNextDepthStencilStateCreate(const ScopedFailNextDepthStencilStateCreate&) = delete;
        ScopedFailNextDepthStencilStateCreate& operator=(const ScopedFailNextDepthStencilStateCreate&) = delete;
        ScopedFailNextDepthStencilStateCreate(ScopedFailNextDepthStencilStateCreate&&) = delete;
        ScopedFailNextDepthStencilStateCreate& operator=(ScopedFailNextDepthStencilStateCreate&&) = delete;

    private:
        std::unique_lock<std::recursive_mutex> lock_;
    };

    class ScopedFailNextPipelineCacheSerialize final
    {
    public:
        explicit ScopedFailNextPipelineCacheSerialize(bool fail = true) noexcept : lock_(testHookMutex())
        {
            setFailNextPipelineCacheSerialize(fail);
        }

        ~ScopedFailNextPipelineCacheSerialize() noexcept
        {
            setFailNextPipelineCacheSerialize(false);
        }

        ScopedFailNextPipelineCacheSerialize(const ScopedFailNextPipelineCacheSerialize&) = delete;
        ScopedFailNextPipelineCacheSerialize& operator=(const ScopedFailNextPipelineCacheSerialize&) = delete;
        ScopedFailNextPipelineCacheSerialize(ScopedFailNextPipelineCacheSerialize&&) = delete;
        ScopedFailNextPipelineCacheSerialize& operator=(ScopedFailNextPipelineCacheSerialize&&) = delete;

    private:
        std::unique_lock<std::recursive_mutex> lock_;
    };

    class ScopedPipelineCacheTemporaryDirectory final
    {
    public:
        explicit ScopedPipelineCacheTemporaryDirectory(std::string_view directory) : lock_(testHookMutex())
        {
            setPipelineCacheTemporaryDirectory(directory);
        }

        ~ScopedPipelineCacheTemporaryDirectory()
        {
            setPipelineCacheTemporaryDirectory({});
        }

        ScopedPipelineCacheTemporaryDirectory(const ScopedPipelineCacheTemporaryDirectory&) = delete;
        ScopedPipelineCacheTemporaryDirectory& operator=(const ScopedPipelineCacheTemporaryDirectory&) = delete;
        ScopedPipelineCacheTemporaryDirectory(ScopedPipelineCacheTemporaryDirectory&&) = delete;
        ScopedPipelineCacheTemporaryDirectory& operator=(ScopedPipelineCacheTemporaryDirectory&&) = delete;

    private:
        std::unique_lock<std::recursive_mutex> lock_;
    };

    class ScopedFailNextDrawable final
    {
    public:
        explicit ScopedFailNextDrawable(bool fail = true) noexcept : lock_(testHookMutex())
        {
            setFailNextDrawable(fail);
        }

        ~ScopedFailNextDrawable() noexcept
        {
            setFailNextDrawable(false);
        }

        ScopedFailNextDrawable(const ScopedFailNextDrawable&) = delete;
        ScopedFailNextDrawable& operator=(const ScopedFailNextDrawable&) = delete;
        ScopedFailNextDrawable(ScopedFailNextDrawable&&) = delete;
        ScopedFailNextDrawable& operator=(ScopedFailNextDrawable&&) = delete;

    private:
        std::unique_lock<std::recursive_mutex> lock_;
    };

    class ScopedFailNextResidencySetCreate final
    {
    public:
        explicit ScopedFailNextResidencySetCreate(bool fail = true) noexcept : lock_(testHookMutex())
        {
            setFailNextResidencySetCreate(fail);
        }

        ~ScopedFailNextResidencySetCreate() noexcept
        {
            setFailNextResidencySetCreate(false);
        }

        ScopedFailNextResidencySetCreate(const ScopedFailNextResidencySetCreate&) = delete;
        ScopedFailNextResidencySetCreate& operator=(const ScopedFailNextResidencySetCreate&) = delete;
        ScopedFailNextResidencySetCreate(ScopedFailNextResidencySetCreate&&) = delete;
        ScopedFailNextResidencySetCreate& operator=(ScopedFailNextResidencySetCreate&&) = delete;

    private:
        std::unique_lock<std::recursive_mutex> lock_;
    };
    [[nodiscard]] Status emitIntentionalValidationFailure();
    [[nodiscard]] Result<MetalCaptureRecord> captureLabeledHeadlessCommand(const MetalCaptureRequest& request);
    [[nodiscard]] MetalValidationConfiguration boundValidationConfiguration();
    [[nodiscard]] Status copyBuffersImmediate(BufferHandle source,
                                              BufferHandle destination,
                                              std::uint64_t sourceOffset,
                                              std::uint64_t destinationOffset,
                                              std::uint64_t size);
    [[nodiscard]] Status copyCapturedBindingBuffer(CommandBuffer& commandBuffer,
                                                   std::uint32_t setIndex,
                                                   BufferHandle destination,
                                                   std::uint64_t size);
    class MetalCommitFeedbackGate final
    {
    public:
        void waitUntilReleased() noexcept
        {
            std::unique_lock lock(mutex_);
            entered_ = true;
            cv_.notify_all();
            cv_.wait(lock, [&] { return released_; });
        }

        [[nodiscard]] bool waitForEntry(std::chrono::milliseconds timeout) noexcept
        {
            std::unique_lock lock(mutex_);
            return cv_.wait_for(lock, timeout, [&] { return entered_; });
        }

        void release() noexcept
        {
            {
                const std::scoped_lock lock(mutex_);
                released_ = true;
            }
            cv_.notify_all();
        }

    private:
        std::mutex mutex_;
        std::condition_variable cv_;
        bool entered_ = false;
        bool released_ = false;
    };

    struct MetalRecordingTelemetry final
    {
        std::uint64_t residencyAddCount = 0;
        std::uint64_t residencyDeduplicatedSkipCount = 0;
        std::uint64_t residencyCommitCount = 0;
        std::uint64_t argumentTableCreateCount = 0;
        std::uint64_t argumentTablePoolReuseCount = 0;
        std::uint64_t argumentTableEncoderReuseCount = 0;
        std::uint64_t pushConstantPageAllocateCount = 0;
        std::uint64_t pushConstantSuballocateCount = 0;
        std::uint64_t renderTimestampVertexStageCount = 0;
        std::uint64_t renderTimestampFragmentStageCount = 0;
    };

    enum class MetalTestErrorDomain : std::uint8_t
    {
        commandQueue4,
        commandBuffer,
        other,
    };

    void resetRecordingTelemetry() noexcept;
    [[nodiscard]] MetalRecordingTelemetry recordingTelemetry() noexcept;
    [[nodiscard]] std::size_t autoreleasedObjectCount(const std::function<void()>& body);
    void resetTeardownQueueDrainCount() noexcept;
    [[nodiscard]] std::uint64_t teardownQueueDrainCount() noexcept;
    [[nodiscard]] DeviceFaultStatus classifyMetalFaultStatusForTesting(MetalTestErrorDomain domain,
                                                                       std::int64_t code) noexcept;

    [[nodiscard]] Status markCommandBufferInFlight(CommandBuffer& commandBuffer, bool inFlight);
    void failNextCommitFeedback();
    void holdNextCommitFeedback(std::shared_ptr<MetalCommitFeedbackGate> gate);
    void holdNextCommandPoolReset(std::shared_ptr<MetalCommitFeedbackGate> gate);
    void holdNextDrawableAcquire(std::shared_ptr<MetalCommitFeedbackGate> gate);
    [[nodiscard]] std::weak_ptr<void> boundBufferOwnerObserver(BufferHandle buffer);
    [[nodiscard]] Result<std::size_t> boundAllocationTelemetryCount();
    [[nodiscard]] Status injectMetalDeviceLoss(DeviceFaultStatus status = DeviceFaultStatus::removed);
    [[nodiscard]] Status notifySurfaceHostClose(SurfaceHandle surface);
    [[nodiscard]] void* boundNativeMetalDeviceHandle() noexcept;
    [[nodiscard]] std::uint32_t boundAppleGpuFamily() noexcept;
    [[nodiscard]] std::uint64_t boundSamplerGpuResourceId(SamplerHandle sampler) noexcept;
    [[nodiscard]] Status loadCookedMetallib(std::span<const std::byte> bytecode, std::string_view entryPoint = {});
    [[nodiscard]] Result<std::array<std::uint32_t, 8>> readLayoutFixtureSentinels(std::span<const std::byte> bytecode);

    struct MetalBarrierStageMask final
    {
        bool vertex = false;
        bool object = false;
        bool mesh = false;
        bool fragment = false;
        bool dispatch = false;

        friend constexpr bool operator==(const MetalBarrierStageMask&, const MetalBarrierStageMask&) noexcept = default;
    };

    [[nodiscard]] MetalBarrierStageMask barrierStagesForTesting(PipelineStage stages) noexcept;
#endif

} // namespace rachis::metal

namespace rachis::metal::test
{

    enum class InjectedPresentFailure : std::uint8_t
    {
        none = 0,
        beforeConsumeValidationError,
        afterConsumeNativeFailure,
        afterConsumeDeviceLost,
    };

    [[nodiscard]] inline std::atomic_int32_t& nextPresentResult() noexcept
    {
        static std::atomic_int32_t s_value = 0;
        return s_value;
    }

    [[nodiscard]] inline std::atomic<InjectedPresentFailure>& nextInjectedPresentFailure() noexcept
    {
        static std::atomic<InjectedPresentFailure> s_value = InjectedPresentFailure::none;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_bool& nextSubmitCloseFailure() noexcept
    {
        static std::atomic_bool s_value = false;
        return s_value;
    }

    [[nodiscard]] inline std::atomic_uint64_t& submittedDrawableWaitCount() noexcept
    {
        static std::atomic_uint64_t s_value = 0;
        return s_value;
    }

} // namespace rachis::metal::test
