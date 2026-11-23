#include "Rachis/Device.h"

#include "Internal/BackendFactory.h"
#include "Internal/DeviceIdentity.h"
#include "Internal/PortableValidation.h"
#include "Internal/VertexInputValidation.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <new>
#include <utility>

namespace rachis
{

    namespace
    {

        [[nodiscard]] Error makeOutOfMemoryError(const char* message = "Host memory allocation failed.") noexcept
        {
            try
            {
                return Error{
                    ErrorCode::outOfMemory,
                    0,
                    message,
                };
            }
            catch (...)
            {
                return Error{
                    ErrorCode::outOfMemory,
                    0,
                    {},
                };
            }
        }

        template<typename Fn>
        auto guardBadAlloc(Fn&& fn) -> decltype(fn())
        {
            try
            {
                return fn();
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(makeOutOfMemoryError());
            }
        }

        [[nodiscard]] Status validateOwnedCommandBuffer(const detail::BackendDevice* backend,
                                                        const detail::BackendCommandBuffer* commandBuffer,
                                                        const char* invalidMessage)
        {
            if (backend == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    invalidMessage,
                });
            }
            if (commandBuffer == nullptr)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend,
                                                                     Error{
                                                                         ErrorCode::invalidState,
                                                                         0,
                                                                         invalidMessage,
                                                                     }));
            }
            if (auto owned = detail::validateCommandBufferOwnership(*commandBuffer, backend->deviceToken()); !owned)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend, std::move(owned.error())));
            }
            return {};
        }

        template<typename Validator, typename... Args>
        [[nodiscard]] Status validateRecordingState(const detail::BackendCommandBuffer& commandBuffer,
                                                    Validator&& validator,
                                                    Args&&... args)
        {
            const auto lock = commandBuffer.lockRecordingState();
            return std::forward<Validator>(validator)(commandBuffer, std::forward<Args>(args)...);
        }

#ifdef RACHIS_TESTING
        std::atomic<detail::MonotonicIdentitySource*> gDeviceTokenSourceOverride{nullptr};
#endif

    } // namespace

    namespace detail
    {

        MonotonicIdentitySource& deviceTokenSource() noexcept
        {
            static MonotonicIdentitySource source;
#ifdef RACHIS_TESTING
            if (auto* overrideSource = gDeviceTokenSourceOverride.load(std::memory_order_acquire);
                overrideSource != nullptr)
            {
                return *overrideSource;
            }
#endif
            return source;
        }

#ifdef RACHIS_TESTING
        void setDeviceTokenSourceOverrideForTesting(MonotonicIdentitySource* overrideSource) noexcept
        {
            gDeviceTokenSourceOverride.store(overrideSource, std::memory_order_release);
        }
#endif

    } // namespace detail

    Device::Device() noexcept = default;

    Device::Device(std::unique_ptr<detail::BackendDevice> backend) noexcept : backend_(std::move(backend))
    {
    }

    Device::~Device() = default;
    Device::Device(Device&& other) noexcept = default;
    Device& Device::operator=(Device&& other) noexcept = default;

    bool Device::isValid() const noexcept
    {
        return backend_ != nullptr;
    }

    Device::operator bool() const noexcept
    {
        return isValid();
    }

    DeviceOperationalState Device::operationalState() const noexcept
    {
        return backend_ != nullptr ? backend_->operationalState() : DeviceOperationalState::invalid;
    }

    std::optional<Error> Device::deviceLossError() const
    {
        return backend_ != nullptr ? backend_->deviceLossError() : std::nullopt;
    }

    std::optional<AdapterInfo> Device::adapterInfo() const
    {
        return backend_ != nullptr ? std::optional<AdapterInfo>{backend_->adapterInfo()} : std::nullopt;
    }

    std::optional<DeviceCapabilities> Device::capabilities() const noexcept
    {
        return backend_ != nullptr ? std::optional<DeviceCapabilities>{backend_->capabilities()} : std::nullopt;
    }

    Result<std::vector<MemoryBudget>> Device::queryMemoryBudgets(std::uint32_t nodeIndex) const
    {
        return guardBadAlloc([&]() -> Result<std::vector<MemoryBudget>> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot query memory budgets from an invalid device.",
                });
            }
            return backend_->queryMemoryBudgets(nodeIndex);
        });
    }

    Result<ResourceMemorySnapshot> Device::captureResourceMemorySnapshot() const
    {
        return guardBadAlloc([&]() -> Result<ResourceMemorySnapshot> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot capture a resource-memory snapshot from an invalid device.",
                });
            }
            return backend_->captureResourceMemorySnapshot();
        });
    }

    Result<DeviceFaultReport> Device::captureDeviceFaultReport() const
    {
        return guardBadAlloc([&]() -> Result<DeviceFaultReport> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot capture a device-fault report from an invalid device.",
                });
            }
            return backend_->captureDeviceFaultReport();
        });
    }

    BackendAllocationSnapshot Device::captureBackendAllocationSnapshot() const noexcept
    {
        return backend_ != nullptr ? backend_->captureBackendAllocationSnapshot() : BackendAllocationSnapshot{};
    }

    FormatProperties Device::queryFormatProperties(Format format) const noexcept
    {
        return backend_ != nullptr ? backend_->queryFormatProperties(format) : FormatProperties{};
    }

    QueueProperties Device::queueProperties(QueueType queueType) const noexcept
    {
        return backend_ != nullptr ? backend_->queueProperties(queueType) : QueueProperties{};
    }

    Result<QueueClockCalibration> Device::calibrateQueueClock(QueueType queueType) const
    {
        return guardBadAlloc([&]() -> Result<QueueClockCalibration> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot calibrate a queue clock from an invalid device.",
                });
            }
            return backend_->calibrateQueueClock(queueType);
        });
    }

    DiagnosticCounts Device::diagnosticCounts() const noexcept
    {
        return backend_ != nullptr ? backend_->diagnosticCounts() : DiagnosticCounts{};
    }

    BindingUpdateStatistics Device::bindingUpdateStatistics() const noexcept
    {
        return backend_ != nullptr ? backend_->bindingUpdateStatistics() : BindingUpdateStatistics{};
    }

    Result<CommandPoolHandle> Device::createCommandPool(const CommandPoolDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<CommandPoolHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a command pool from an invalid device.",
                });
            }
            return backend_->createCommandPool(desc);
        });
    }

    Status Device::destroyCommandPool(CommandPoolHandle commandPool)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot destroy a command pool from an invalid device.",
                });
            }
            return backend_->destroyCommandPool(commandPool);
        });
    }

    Status Device::resetCommandPool(CommandPoolHandle commandPool)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot reset a command pool from an invalid device.",
                });
            }
            return backend_->resetCommandPool(commandPool);
        });
    }

    Result<CommandBuffer> Device::acquireCommandBuffer(CommandPoolHandle commandPool)
    {
        return guardBadAlloc([&]() -> Result<CommandBuffer> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot acquire a command buffer from an invalid device.",
                });
            }

            auto commandBuffer = backend_->acquireCommandBuffer(commandPool);
            if (!commandBuffer)
            {
                return std::unexpected(std::move(commandBuffer.error()));
            }
            return CommandBuffer(std::move(*commandBuffer));
        });
    }

    Status Device::resetCommandBuffer(CommandBuffer& commandBuffer)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(backend_.get(),
                                                        commandBuffer.backend_.get(),
                                                        "Command-buffer reset requires a valid device and recording.");
                !valid)
            {
                return valid;
            }
            if (auto ready =
                    validateRecordingState(*commandBuffer.backend_, detail::validateCommandBufferReadyForReset);
                !ready)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(ready.error())));
            }
            return backend_->resetCommandBuffer(*commandBuffer.backend_);
        });
    }

    Result<TimelinePoint> Device::submit(CommandBuffer&& commandBuffer, std::span<const TimelinePoint> waits)
    {
        return guardBadAlloc([&]() -> Result<TimelinePoint> {
            CommandBuffer submission = std::move(commandBuffer);
            const std::span commands(&submission, 1);
            auto submitted = submit(commands, waits);
            if (submission.isValid())
            {
                commandBuffer = std::move(submission);
            }
            return submitted;
        });
    }

    Result<TimelinePoint> Device::submit(std::span<CommandBuffer> commandBuffers, std::span<const TimelinePoint> waits)
    {
        return guardBadAlloc([&]() -> Result<TimelinePoint> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot submit to an invalid device.",
                });
            }
            if (commandBuffers.empty())
            {
                return std::unexpected(
                    detail::preferDeviceLossError(*backend_,
                                                  Error{
                                                      ErrorCode::invalidArgument,
                                                      0,
                                                      "A submission batch must contain at least one command buffer.",
                                                  }));
            }

            const auto expectedDeviceToken = backend_->deviceToken();
            std::optional<QueueType> queueType;
            for (const auto& commandBuffer : commandBuffers)
            {
                if (commandBuffer.backend_ == nullptr)
                {
                    return std::unexpected(
                        detail::preferDeviceLossError(*backend_,
                                                      Error{
                                                          ErrorCode::invalidArgument,
                                                          0,
                                                          "A submission batch contains an invalid command buffer.",
                                                      }));
                }
                if (auto owned = detail::validateCommandBufferOwnership(*commandBuffer.backend_, expectedDeviceToken);
                    !owned)
                {
                    return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(owned.error())));
                }
                if (!queueType.has_value())
                {
                    queueType = commandBuffer.queueType();
                }
                else if (commandBuffer.queueType() != *queueType)
                {
                    return std::unexpected(detail::preferDeviceLossError(
                        *backend_,
                        Error{
                            ErrorCode::invalidArgument,
                            0,
                            "Every command buffer in a submission batch must target the same queue.",
                        }));
                }
            }

            for (const auto& commandBuffer : commandBuffers)
            {
                if (auto ready =
                        validateRecordingState(*commandBuffer.backend_, detail::validateCommandBufferReadyForSubmit);
                    !ready)
                {
                    return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(ready.error())));
                }
            }

            std::array<std::unique_ptr<detail::BackendCommandBuffer>, 8> inlineCommands{};
            std::vector<std::unique_ptr<detail::BackendCommandBuffer>> heapCommands;
            std::span<std::unique_ptr<detail::BackendCommandBuffer>> nativeSpan;
            if (commandBuffers.size() <= inlineCommands.size())
            {
                nativeSpan = std::span(inlineCommands.data(), commandBuffers.size());
            }
            else
            {
                try
                {
                    heapCommands.resize(commandBuffers.size());
                }
                catch (const std::bad_alloc&)
                {
                    return std::unexpected(makeOutOfMemoryError("Failed to allocate submission command batch."));
                }
                nativeSpan = std::span(heapCommands.data(), commandBuffers.size());
            }

            for (std::size_t index = 0; index < commandBuffers.size(); ++index)
            {
                nativeSpan[index] = std::move(commandBuffers[index].backend_);
            }

            const auto restoreUnconsumed = [&]() noexcept {
                for (std::size_t index = 0; index < commandBuffers.size(); ++index)
                {
                    if (nativeSpan[index] != nullptr)
                    {
                        commandBuffers[index].backend_ = std::move(nativeSpan[index]);
                    }
                }
            };

            try
            {
                auto submitted = backend_->submit(nativeSpan, waits);
                restoreUnconsumed();
                return submitted;
            }
            catch (const std::bad_alloc&)
            {
                restoreUnconsumed();
                return std::unexpected(makeOutOfMemoryError());
            }
        });
    }

    Result<TimelinePointStatus> Device::queryTimelinePoint(TimelinePoint point) const
    {
        return guardBadAlloc([&]() -> Result<TimelinePointStatus> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot query a timeline point on an invalid device.",
                });
            }
            return backend_->queryTimelinePoint(point);
        });
    }

    Result<TimelinePointStatus> Device::wait(TimelinePoint point, std::chrono::nanoseconds timeout)
    {
        return guardBadAlloc([&]() -> Result<TimelinePointStatus> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot wait on an invalid device.",
                });
            }
            return backend_->wait(point, timeout);
        });
    }

    Status Device::wait(TimelinePoint point)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot wait on an invalid device.",
                });
            }
            return backend_->wait(point);
        });
    }

    Result<SharedTimelinePoint> Device::exportTimelinePoint(TimelinePoint point)
    {
        return guardBadAlloc([&]() -> Result<SharedTimelinePoint> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot export a timeline point from an invalid device.",
                });
            }
            return backend_->exportTimelinePoint(point);
        });
    }

    Result<TimelinePoint> Device::enqueueSharedTimelineWait(QueueType queueType, SharedTimelinePointView point)
    {
        return guardBadAlloc([&]() -> Result<TimelinePoint> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot enqueue a shared timeline wait on an invalid device.",
                });
            }
            return backend_->enqueueSharedTimelineWait(queueType, point);
        });
    }

    Status Device::waitIdle(QueueType queueType)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot wait for an invalid device queue to become idle.",
                });
            }
            return backend_->waitIdle(queueType);
        });
    }

    Status Device::waitIdle()
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot wait for an invalid device to become idle.",
                });
            }
            return backend_->waitIdle();
        });
    }

    std::uint64_t Device::completedValue(QueueType queueType) const noexcept
    {
        return backend_ != nullptr ? backend_->completedValue(queueType) : 0;
    }

    void Device::collectGarbage() noexcept
    {
        if (backend_ != nullptr)
        {
            backend_->collectGarbage();
        }
    }

    Status Device::beginDebugGroup(CommandBuffer& commandBuffer, const DebugMarkerDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Beginning a debug group requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            if (backend_->backendName() == "D3D12" && !backend_->capabilities().debugMarkers)
            {
                return backend_->beginDebugGroup(*commandBuffer.backend_, desc);
            }
            if (auto valid = validateRecordingState(*commandBuffer.backend_,
                                                    [&](const detail::BackendCommandBuffer& locked) {
                                                        return detail::validateDebugGroupBegin(locked.debugState(),
                                                                                               desc);
                                                    });
                !valid)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(valid.error())));
            }
            return backend_->beginDebugGroup(*commandBuffer.backend_, desc);
        });
    }

    Status Device::endDebugGroup(CommandBuffer& commandBuffer)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Ending a debug group requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            if (backend_->backendName() == "D3D12" && !backend_->capabilities().debugMarkers)
            {
                return backend_->endDebugGroup(*commandBuffer.backend_);
            }
            if (auto valid = validateRecordingState(*commandBuffer.backend_,
                                                    [](const detail::BackendCommandBuffer& locked) {
                                                        return detail::validateDebugGroupEnd(locked.debugState());
                                                    });
                !valid)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(valid.error())));
            }
            return backend_->endDebugGroup(*commandBuffer.backend_);
        });
    }

    Status Device::insertDebugMarker(CommandBuffer& commandBuffer, const DebugMarkerDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Inserting a debug marker requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            if (backend_->backendName() == "D3D12" && !backend_->capabilities().debugMarkers)
            {
                return backend_->insertDebugMarker(*commandBuffer.backend_, desc);
            }
            if (auto valid = detail::validateDebugMarkerDesc(desc); !valid)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(valid.error())));
            }
            return backend_->insertDebugMarker(*commandBuffer.backend_, desc);
        });
    }

    Result<ResourceHeapHandle> Device::createResourceHeap(const ResourceHeapDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<ResourceHeapHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a resource heap on an invalid device.",
                });
            }
            return backend_->createResourceHeap(desc);
        });
    }

    Status Device::destroyResourceHeap(ResourceHeapHandle heap)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot destroy a resource heap on an invalid device.",
                });
            }
            return backend_->destroyResourceHeap(heap);
        });
    }

    Result<ResidencyState> Device::queryResidency(ResourceHeapHandle heap) const
    {
        return guardBadAlloc([&]() -> Result<ResidencyState> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot query resource-heap residency on an invalid device.",
                });
            }
            return backend_->queryResidency(heap);
        });
    }

    Result<ResourceMemoryRequirements> Device::queryBufferMemoryRequirements(const BufferDesc& desc) const
    {
        return guardBadAlloc([&]() -> Result<ResourceMemoryRequirements> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot query buffer memory requirements on an invalid device.",
                });
            }
            return backend_->queryBufferMemoryRequirements(desc);
        });
    }

    Result<BufferHandle> Device::createBuffer(const BufferDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<BufferHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a buffer on an invalid device.",
                });
            }
            return backend_->createBuffer(desc);
        });
    }

    Result<BufferHandle> Device::createPlacedBuffer(const ResourcePlacement& placement, const BufferDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<BufferHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a placed buffer on an invalid device.",
                });
            }
            return backend_->createPlacedBuffer(placement, desc);
        });
    }

    Result<BufferHandle> Device::createSparseBuffer(const BufferDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<BufferHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a sparse buffer on an invalid device.",
                });
            }
            return backend_->createSparseBuffer(desc);
        });
    }

    Result<SparseBufferProperties> Device::querySparseBufferProperties(BufferHandle buffer) const
    {
        return guardBadAlloc([&]() -> Result<SparseBufferProperties> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot query sparse-buffer properties on an invalid device.",
                });
            }
            return backend_->querySparseBufferProperties(buffer);
        });
    }

    Result<TimelinePoint> Device::enqueueSparseBufferMappings(QueueType queueType,
                                                              std::span<const SparseBufferTileMapping> mappings)
    {
        return guardBadAlloc([&]() -> Result<TimelinePoint> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot enqueue sparse-buffer mappings on an invalid device.",
                });
            }
            return backend_->enqueueSparseBufferMappings(queueType, mappings);
        });
    }

    Result<BufferHandle> Device::createSharedBuffer(const BufferDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<BufferHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a shared buffer on an invalid device.",
                });
            }
            return backend_->createSharedBuffer(desc);
        });
    }

    Result<SharedHandle> Device::exportSharedBuffer(BufferHandle buffer)
    {
        return guardBadAlloc([&]() -> Result<SharedHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot export a shared buffer from an invalid device.",
                });
            }
            return backend_->exportSharedBuffer(buffer);
        });
    }

    Result<BufferHandle> Device::importSharedBuffer(SharedHandleView handle, const BufferDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<BufferHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot import a shared buffer into an invalid device.",
                });
            }
            return backend_->importSharedBuffer(handle, desc);
        });
    }

    Status Device::destroyBuffer(BufferHandle buffer)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot destroy a buffer on an invalid device.",
                });
            }
            return backend_->destroyBuffer(buffer);
        });
    }

    Result<ResidencyState> Device::queryResidency(BufferHandle buffer) const
    {
        return guardBadAlloc([&]() -> Result<ResidencyState> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot query buffer residency on an invalid device.",
                });
            }
            return backend_->queryResidency(buffer);
        });
    }

    Result<MappedMemory> Device::mapBuffer(BufferHandle buffer, std::uint64_t offset, std::uint64_t size)
    {
        return guardBadAlloc([&]() -> Result<MappedMemory> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot map a buffer on an invalid device.",
                });
            }
            return backend_->mapBuffer(buffer, offset, size);
        });
    }

    Status Device::unmapBuffer(BufferHandle buffer, std::uint64_t offset, std::uint64_t size)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot unmap a buffer on an invalid device.",
                });
            }
            return backend_->unmapBuffer(buffer, offset, size);
        });
    }

    Status Device::flushMappedMemoryRanges(BufferHandle buffer, std::span<const MappedMemoryRange> ranges)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot flush mapped memory on an invalid device.",
                });
            }
            return backend_->flushMappedMemoryRanges(buffer, ranges);
        });
    }

    Status Device::invalidateMappedMemoryRanges(BufferHandle buffer, std::span<const MappedMemoryRange> ranges)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot invalidate mapped memory on an invalid device.",
                });
            }
            return backend_->invalidateMappedMemoryRanges(buffer, ranges);
        });
    }

    Status Device::writeBuffer(BufferHandle buffer, std::uint64_t offset, std::span<const std::byte> data)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot write a buffer on an invalid device.",
                });
            }
            return backend_->writeBuffer(buffer, offset, data);
        });
    }

    Status Device::readBuffer(BufferHandle buffer, std::uint64_t offset, std::span<std::byte> destination)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot read a buffer on an invalid device.",
                });
            }
            return backend_->readBuffer(buffer, offset, destination);
        });
    }

    Status Device::copyBuffer(CommandBuffer& commandBuffer, const BufferCopy& copy)
    {
        return guardBadAlloc([&]() -> Status {
            const std::array regions{BufferCopyRegion{
                .sourceOffset = copy.sourceOffset,
                .destinationOffset = copy.destinationOffset,
                .size = copy.size,
            }};
            return copyBuffer(commandBuffer, copy.source, copy.destination, regions);
        });
    }

    Status Device::copyBuffer(CommandBuffer& commandBuffer,
                              BufferHandle source,
                              BufferHandle destination,
                              std::span<const BufferCopyRegion> regions)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(backend_.get(),
                                                        commandBuffer.backend_.get(),
                                                        "Buffer copies require a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            if (auto outside = validateRecordingState(*commandBuffer.backend_,
                                                      detail::requireOutsideRendering,
                                                      "Buffer copies cannot be recorded inside a rendering scope.");
                !outside)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(outside.error())));
            }
            if (auto valid = detail::validateBufferCopyRegions(regions); !valid)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(valid.error())));
            }
            return backend_->copyBufferRegions(*commandBuffer.backend_, source, destination, regions);
        });
    }

    Status Device::globalBarrier(CommandBuffer& commandBuffer, const GlobalBarrier& barrier)
    {
        return guardBadAlloc([&]() -> Status {
            const std::array barriers{barrier};
            return resourceBarriers(commandBuffer, ResourceBarrierBatch{.globalBarriers = barriers});
        });
    }

    Status Device::bufferBarrier(CommandBuffer& commandBuffer, const BufferBarrier& barrier)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(backend_.get(),
                                                        commandBuffer.backend_.get(),
                                                        "Buffer barriers require a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->bufferBarrier(*commandBuffer.backend_, barrier);
        });
    }

    Status Device::samplerFeedbackBarrier(CommandBuffer& commandBuffer, const SamplerFeedbackBarrier& barrier)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(), commandBuffer.backend_.get(), "The device or command buffer is not initialized.");
                !valid)
            {
                return valid;
            }
            return backend_->samplerFeedbackBarrier(*commandBuffer.backend_, barrier);
        });
    }

    Status Device::activateBufferAlias(CommandBuffer& commandBuffer, const BufferAliasActivation& activation)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Buffer alias activation requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->activateBufferAlias(*commandBuffer.backend_, activation);
        });
    }

    Status Device::resourceBarriers(CommandBuffer& commandBuffer, const ResourceBarrierBatch& barriers)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Resource-barrier batching requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->resourceBarriers(*commandBuffer.backend_, barriers);
        });
    }

    Result<AccelerationStructureBuildSizes> Device::queryBottomLevelAccelerationStructureBuildSizes(
        const BottomLevelAccelerationStructureBuildInput& input) const
    {
        return guardBadAlloc([&]() -> Result<AccelerationStructureBuildSizes> {
            if (backend_ == nullptr)
            {
                return std::unexpected(
                    Error{ErrorCode::invalidState, 0, "Cannot query build sizes on an invalid device."});
            }
            return backend_->queryBottomLevelAccelerationStructureBuildSizes(input);
        });
    }

    Result<AccelerationStructureBuildSizes> Device::queryTopLevelAccelerationStructureBuildSizes(
        const TopLevelAccelerationStructureBuildInput& input) const
    {
        return guardBadAlloc([&]() -> Result<AccelerationStructureBuildSizes> {
            if (backend_ == nullptr)
            {
                return std::unexpected(
                    Error{ErrorCode::invalidState, 0, "Cannot query build sizes on an invalid device."});
            }
            return backend_->queryTopLevelAccelerationStructureBuildSizes(input);
        });
    }

    Result<AccelerationStructureHandle> Device::createAccelerationStructure(const AccelerationStructureDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<AccelerationStructureHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(
                    Error{ErrorCode::invalidState, 0, "Cannot create an acceleration structure on an invalid device."});
            }
            return backend_->createAccelerationStructure(desc);
        });
    }

    Status Device::destroyAccelerationStructure(AccelerationStructureHandle accelerationStructure)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState, 0, "Cannot destroy an acceleration structure on an invalid device."});
            }
            return backend_->destroyAccelerationStructure(accelerationStructure);
        });
    }

    Status Device::buildBottomLevelAccelerationStructure(CommandBuffer& commandBuffer,
                                                         const BottomLevelAccelerationStructureBuildDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Bottom-level acceleration-structure builds require a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->buildBottomLevelAccelerationStructure(*commandBuffer.backend_, desc);
        });
    }

    Status Device::buildTopLevelAccelerationStructure(CommandBuffer& commandBuffer,
                                                      const TopLevelAccelerationStructureBuildDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Top-level acceleration-structure builds require a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->buildTopLevelAccelerationStructure(*commandBuffer.backend_, desc);
        });
    }

    Status Device::updateBottomLevelAccelerationStructure(CommandBuffer& commandBuffer,
                                                          const BottomLevelAccelerationStructureUpdateDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Bottom-level acceleration-structure updates require a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->updateBottomLevelAccelerationStructure(*commandBuffer.backend_, desc);
        });
    }

    Status Device::updateTopLevelAccelerationStructure(CommandBuffer& commandBuffer,
                                                       const TopLevelAccelerationStructureUpdateDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Top-level acceleration-structure updates require a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->updateTopLevelAccelerationStructure(*commandBuffer.backend_, desc);
        });
    }

    Status Device::updateBottomLevelAccelerationStructureInPlace(
        CommandBuffer& commandBuffer, const BottomLevelAccelerationStructureInPlaceUpdateDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "In-place bottom-level acceleration-structure updates require a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->updateBottomLevelAccelerationStructureInPlace(*commandBuffer.backend_, desc);
        });
    }

    Status Device::updateTopLevelAccelerationStructureInPlace(
        CommandBuffer& commandBuffer, const TopLevelAccelerationStructureInPlaceUpdateDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "In-place top-level acceleration-structure updates require a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->updateTopLevelAccelerationStructureInPlace(*commandBuffer.backend_, desc);
        });
    }

    Status Device::compactAccelerationStructure(CommandBuffer& commandBuffer,
                                                const AccelerationStructureCompactionDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Acceleration-structure compaction requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->compactAccelerationStructure(*commandBuffer.backend_, desc);
        });
    }

    Status Device::serializeAccelerationStructure(CommandBuffer& commandBuffer,
                                                  const AccelerationStructureSerializationDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Acceleration-structure serialization requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->serializeAccelerationStructure(*commandBuffer.backend_, desc);
        });
    }

    Result<AccelerationStructureArchiveInfo> Device::inspectAccelerationStructureArchive(
        std::span<const std::byte> archive) const
    {
        return guardBadAlloc([&]() -> Result<AccelerationStructureArchiveInfo> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Acceleration-structure archive inspection requires a valid device.",
                });
            }
            return backend_->inspectAccelerationStructureArchive(archive);
        });
    }

    Status Device::deserializeAccelerationStructure(CommandBuffer& commandBuffer,
                                                    const AccelerationStructureDeserializationDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Acceleration-structure deserialization requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->deserializeAccelerationStructure(*commandBuffer.backend_, desc);
        });
    }

    Result<ResourceMemoryRequirements> Device::queryTextureMemoryRequirements(const TextureDesc& desc) const
    {
        return guardBadAlloc([&]() -> Result<ResourceMemoryRequirements> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot query texture memory requirements on an invalid device.",
                });
            }
            return backend_->queryTextureMemoryRequirements(desc);
        });
    }

    Result<TextureHandle> Device::createTexture(const TextureDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<TextureHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a texture on an invalid device.",
                });
            }
            return backend_->createTexture(desc);
        });
    }

    Result<TextureHandle> Device::createPlacedTexture(const ResourcePlacement& placement, const TextureDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<TextureHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a placed texture on an invalid device.",
                });
            }
            return backend_->createPlacedTexture(placement, desc);
        });
    }

    Result<TextureHandle> Device::createSparseTexture(const TextureDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<TextureHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a sparse texture on an invalid device.",
                });
            }
            return backend_->createSparseTexture(desc);
        });
    }

    Result<SparseTextureProperties> Device::querySparseTextureProperties(TextureHandle texture) const
    {
        return guardBadAlloc([&]() -> Result<SparseTextureProperties> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot query sparse-texture properties on an invalid device.",
                });
            }
            return backend_->querySparseTextureProperties(texture);
        });
    }

    Result<TimelinePoint> Device::enqueueSparseTextureMappings(QueueType queueType,
                                                               const SparseTextureMappingBatch& mappings)
    {
        return guardBadAlloc([&]() -> Result<TimelinePoint> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot enqueue sparse-texture mappings on an invalid device.",
                });
            }
            return backend_->enqueueSparseTextureMappings(queueType, mappings);
        });
    }

    Result<SamplerFeedbackMapHandle> Device::createMinMipFeedbackMap(const MinMipFeedbackMapDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<SamplerFeedbackMapHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The device is not initialized.",
                });
            }
            return backend_->createMinMipFeedbackMap(desc);
        });
    }

    Result<MinMipFeedbackMapProperties> Device::queryMinMipFeedbackMapProperties(
        SamplerFeedbackMapHandle feedbackMap) const
    {
        return guardBadAlloc([&]() -> Result<MinMipFeedbackMapProperties> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The device is not initialized.",
                });
            }
            return backend_->queryMinMipFeedbackMapProperties(feedbackMap);
        });
    }

    Result<SamplerFeedbackMapHandle> Device::createMipRegionUsedFeedbackMap(const MipRegionUsedFeedbackMapDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<SamplerFeedbackMapHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The device is not initialized.",
                });
            }
            return backend_->createMipRegionUsedFeedbackMap(desc);
        });
    }

    Result<MipRegionUsedFeedbackMapProperties> Device::queryMipRegionUsedFeedbackMapProperties(
        SamplerFeedbackMapHandle feedbackMap) const
    {
        return guardBadAlloc([&]() -> Result<MipRegionUsedFeedbackMapProperties> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The device is not initialized.",
                });
            }
            return backend_->queryMipRegionUsedFeedbackMapProperties(feedbackMap);
        });
    }

    Status Device::destroySamplerFeedbackMap(SamplerFeedbackMapHandle feedbackMap)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The device is not initialized.",
                });
            }
            return backend_->destroySamplerFeedbackMap(feedbackMap);
        });
    }

    Status Device::resetSamplerFeedbackMap(CommandBuffer& commandBuffer, SamplerFeedbackMapHandle feedbackMap)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(), commandBuffer.backend_.get(), "The device or command buffer is not initialized.");
                !valid)
            {
                return valid;
            }
            return backend_->resetSamplerFeedbackMap(*commandBuffer.backend_, feedbackMap);
        });
    }

    Status Device::decodeMinMipFeedback(CommandBuffer& commandBuffer, const MinMipFeedbackDecode& decode)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(), commandBuffer.backend_.get(), "The device or command buffer is not initialized.");
                !valid)
            {
                return valid;
            }
            return backend_->decodeMinMipFeedback(*commandBuffer.backend_, decode);
        });
    }

    Status Device::decodeMipRegionUsedFeedback(CommandBuffer& commandBuffer, const MipRegionUsedFeedbackDecode& decode)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(), commandBuffer.backend_.get(), "The device or command buffer is not initialized.");
                !valid)
            {
                return valid;
            }
            return backend_->decodeMipRegionUsedFeedback(*commandBuffer.backend_, decode);
        });
    }

    Status Device::encodeMinMipFeedback(CommandBuffer& commandBuffer, const MinMipFeedbackEncode& encode)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(), commandBuffer.backend_.get(), "The device or command buffer is not initialized.");
                !valid)
            {
                return valid;
            }
            return backend_->encodeMinMipFeedback(*commandBuffer.backend_, encode);
        });
    }

    Status Device::encodeMipRegionUsedFeedback(CommandBuffer& commandBuffer, const MipRegionUsedFeedbackEncode& encode)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(), commandBuffer.backend_.get(), "The device or command buffer is not initialized.");
                !valid)
            {
                return valid;
            }
            return backend_->encodeMipRegionUsedFeedback(*commandBuffer.backend_, encode);
        });
    }

    Result<TextureHandle> Device::createSharedTexture(const TextureDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<TextureHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a shared texture on an invalid device.",
                });
            }
            return backend_->createSharedTexture(desc);
        });
    }

    Result<SharedHandle> Device::exportSharedTexture(TextureHandle texture)
    {
        return guardBadAlloc([&]() -> Result<SharedHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot export a shared texture from an invalid device.",
                });
            }
            return backend_->exportSharedTexture(texture);
        });
    }

    Result<TextureHandle> Device::importSharedTexture(SharedHandleView handle, const TextureDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<TextureHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot import a shared texture into an invalid device.",
                });
            }
            return backend_->importSharedTexture(handle, desc);
        });
    }

    Status Device::destroyTexture(TextureHandle texture)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot destroy a texture on an invalid device.",
                });
            }
            return backend_->destroyTexture(texture);
        });
    }

    Result<ResidencyState> Device::queryResidency(TextureHandle texture) const
    {
        return guardBadAlloc([&]() -> Result<ResidencyState> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot query texture residency on an invalid device.",
                });
            }
            return backend_->queryResidency(texture);
        });
    }

    Result<TextureTransferLayout> Device::queryTextureTransferLayout(TextureHandle texture,
                                                                     const BufferTextureCopyRegion& region) const
    {
        return guardBadAlloc([&]() -> Result<TextureTransferLayout> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot query a texture transfer layout on an invalid device.",
                });
            }
            return backend_->queryTextureTransferLayout(texture, region);
        });
    }

    Result<TimelinePoint> Device::enqueueMakeResident(QueueType queueType,
                                                      const ResidencySet& resources,
                                                      ResidencyBudgetPolicy budgetPolicy)
    {
        return guardBadAlloc([&]() -> Result<TimelinePoint> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot enqueue a residency transition on an invalid device.",
                });
            }
            return backend_->enqueueMakeResident(queueType, resources, budgetPolicy);
        });
    }

    Status Device::evict(const ResidencySet& resources)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot evict resources from an invalid device.",
                });
            }
            return backend_->evict(resources);
        });
    }

    Status Device::copyBufferToTexture(CommandBuffer& commandBuffer, const BufferTextureCopy& copy)
    {
        return guardBadAlloc([&]() -> Status {
            const std::array regions{BufferTextureCopyRegion{
                .bufferOffset = copy.bufferOffset,
                .bytesPerRow = copy.bytesPerRow,
                .rowsPerImage = copy.rowsPerImage,
                .textureSubresource = copy.textureSubresource,
                .textureOrigin = copy.textureOrigin,
                .extent = copy.extent,
            }};
            return copyBufferToTexture(commandBuffer, copy.buffer, copy.texture, regions);
        });
    }

    Status Device::copyBufferToTexture(CommandBuffer& commandBuffer,
                                       BufferHandle source,
                                       TextureHandle destination,
                                       std::span<const BufferTextureCopyRegion> regions)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Buffer-to-texture copies require a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            if (auto outside =
                    validateRecordingState(*commandBuffer.backend_,
                                           detail::requireOutsideRendering,
                                           "Buffer-to-texture copies cannot be recorded inside a rendering scope.");
                !outside)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(outside.error())));
            }
            if (auto valid = detail::validateCopyRegionCount(
                    regions.size(),
                    "Buffer-to-texture copies require at least one region and at most uint32_t::max regions.");
                !valid)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(valid.error())));
            }
            return backend_->copyBufferToTextureRegions(*commandBuffer.backend_, source, destination, regions);
        });
    }

    Status Device::copyTextureToBuffer(CommandBuffer& commandBuffer, const BufferTextureCopy& copy)
    {
        return guardBadAlloc([&]() -> Status {
            const std::array regions{BufferTextureCopyRegion{
                .bufferOffset = copy.bufferOffset,
                .bytesPerRow = copy.bytesPerRow,
                .rowsPerImage = copy.rowsPerImage,
                .textureSubresource = copy.textureSubresource,
                .textureOrigin = copy.textureOrigin,
                .extent = copy.extent,
            }};
            return copyTextureToBuffer(commandBuffer, copy.texture, copy.buffer, regions);
        });
    }

    Status Device::copyTextureToBuffer(CommandBuffer& commandBuffer,
                                       TextureHandle source,
                                       BufferHandle destination,
                                       std::span<const BufferTextureCopyRegion> regions)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Texture-to-buffer copies require a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            if (auto outside =
                    validateRecordingState(*commandBuffer.backend_,
                                           detail::requireOutsideRendering,
                                           "Texture-to-buffer copies cannot be recorded inside a rendering scope.");
                !outside)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(outside.error())));
            }
            if (auto valid = detail::validateCopyRegionCount(
                    regions.size(),
                    "Texture-to-buffer copies require at least one region and at most uint32_t::max regions.");
                !valid)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(valid.error())));
            }
            return backend_->copyTextureToBufferRegions(*commandBuffer.backend_, source, destination, regions);
        });
    }

    Status Device::copyTexture(CommandBuffer& commandBuffer, const TextureCopy& copy)
    {
        return guardBadAlloc([&]() -> Status {
            const std::array regions{TextureCopyRegion{
                .sourceSubresource = copy.sourceSubresource,
                .sourceOrigin = copy.sourceOrigin,
                .destinationSubresource = copy.destinationSubresource,
                .destinationOrigin = copy.destinationOrigin,
                .extent = copy.extent,
            }};
            return copyTexture(commandBuffer, copy.source, copy.destination, regions);
        });
    }

    Status Device::copyTexture(CommandBuffer& commandBuffer,
                               TextureHandle source,
                               TextureHandle destination,
                               std::span<const TextureCopyRegion> regions)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(backend_.get(),
                                                        commandBuffer.backend_.get(),
                                                        "Texture copies require a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            if (auto outside = validateRecordingState(*commandBuffer.backend_,
                                                      detail::requireOutsideRendering,
                                                      "Texture copies cannot be recorded inside a rendering scope.");
                !outside)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(outside.error())));
            }
            if (auto valid = detail::validateCopyRegionCount(
                    regions.size(), "Texture copies require at least one region and at most uint32_t::max regions.");
                !valid)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(valid.error())));
            }
            return backend_->copyTextureRegions(*commandBuffer.backend_, source, destination, regions);
        });
    }

    Status Device::resolveTexture(CommandBuffer& commandBuffer, const TextureResolve& resolve)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(backend_.get(),
                                                        commandBuffer.backend_.get(),
                                                        "Texture resolves require a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->resolveTexture(*commandBuffer.backend_, resolve);
        });
    }

    Status Device::generateMipmaps(CommandBuffer& commandBuffer, const TextureMipmapGeneration& generation)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Mipmap generation requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->generateMipmaps(*commandBuffer.backend_, generation);
        });
    }

    Status Device::textureBarrier(CommandBuffer& commandBuffer, const TextureBarrier& barrier)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(backend_.get(),
                                                        commandBuffer.backend_.get(),
                                                        "Texture barriers require a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->textureBarrier(*commandBuffer.backend_, barrier);
        });
    }

    Status Device::activateTextureAlias(CommandBuffer& commandBuffer, const TextureAliasActivation& activation)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Texture alias activation requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->activateTextureAlias(*commandBuffer.backend_, activation);
        });
    }

    Result<TextureViewHandle> Device::createTextureView(const TextureViewDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<TextureViewHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a texture view on an invalid device.",
                });
            }
            return backend_->createTextureView(desc);
        });
    }

    Status Device::destroyTextureView(TextureViewHandle view)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot destroy a texture view on an invalid device.",
                });
            }
            return backend_->destroyTextureView(view);
        });
    }

    Result<SamplerHandle> Device::createSampler(const SamplerDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<SamplerHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a sampler on an invalid device.",
                });
            }
            return backend_->createSampler(desc);
        });
    }

    Status Device::destroySampler(SamplerHandle sampler)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot destroy a sampler on an invalid device.",
                });
            }
            return backend_->destroySampler(sampler);
        });
    }

    Result<BindingLayoutHandle> Device::createBindingLayout(const BindingLayoutDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<BindingLayoutHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a binding layout on an invalid device.",
                });
            }
            if (auto valid = detail::validateBindingLayoutDesc(desc); !valid)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(valid.error())));
            }
            return backend_->createBindingLayout(desc);
        });
    }

    Status Device::destroyBindingLayout(BindingLayoutHandle layout)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot destroy a binding layout on an invalid device.",
                });
            }
            return backend_->destroyBindingLayout(layout);
        });
    }

    Result<BindingPoolHandle> Device::createBindingPool(const BindingPoolDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<BindingPoolHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a binding pool on an invalid device.",
                });
            }
            if (auto valid = detail::validateBindingPoolDesc(desc); !valid)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(valid.error())));
            }
            return backend_->createBindingPool(desc);
        });
    }

    Status Device::destroyBindingPool(BindingPoolHandle pool)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot destroy a binding pool on an invalid device.",
                });
            }
            return backend_->destroyBindingPool(pool);
        });
    }

    Result<BindingSetHandle> Device::createBindingSet(const BindingSetDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<BindingSetHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a binding set on an invalid device.",
                });
            }
            return backend_->createBindingSet(desc);
        });
    }

    Status Device::updateBindingSet(BindingSetHandle set, std::span<const BindingWrite> writes)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot update a binding set on an invalid device.",
                });
            }
            return backend_->updateBindingSet(set, writes);
        });
    }

    Status Device::destroyBindingSet(BindingSetHandle set)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot destroy a binding set on an invalid device.",
                });
            }
            return backend_->destroyBindingSet(set);
        });
    }

    Result<ShaderModuleHandle> Device::createShaderModule(const ShaderModuleDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<ShaderModuleHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a shader module on an invalid device.",
                });
            }
            return backend_->createShaderModule(desc);
        });
    }

    Result<ShaderInterfaceSnapshot> Device::shaderModuleInterface(ShaderModuleHandle shaderModule) const
    {
        return guardBadAlloc([&]() -> Result<ShaderInterfaceSnapshot> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot query a shader-module interface on an invalid device.",
                });
            }
            return backend_->shaderModuleInterface(shaderModule);
        });
    }

    Status Device::destroyShaderModule(ShaderModuleHandle shaderModule)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot destroy a shader module on an invalid device.",
                });
            }
            return backend_->destroyShaderModule(shaderModule);
        });
    }

    Result<ShaderLibraryHandle> Device::createShaderLibrary(const ShaderLibraryDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<ShaderLibraryHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a shader library on an invalid device.",
                });
            }
            return backend_->createShaderLibrary(desc);
        });
    }

    Status Device::destroyShaderLibrary(ShaderLibraryHandle shaderLibrary)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot destroy a shader library on an invalid device.",
                });
            }
            return backend_->destroyShaderLibrary(shaderLibrary);
        });
    }

    Result<PipelineLayoutHandle> Device::createPipelineLayout(const PipelineLayoutDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<PipelineLayoutHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a pipeline layout on an invalid device.",
                });
            }
            return backend_->createPipelineLayout(desc);
        });
    }

    Status Device::destroyPipelineLayout(PipelineLayoutHandle pipelineLayout)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot destroy a pipeline layout on an invalid device.",
                });
            }
            return backend_->destroyPipelineLayout(pipelineLayout);
        });
    }

    Result<PipelineCacheHandle> Device::createPipelineCache(const PipelineCacheDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<PipelineCacheHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a pipeline cache on an invalid device.",
                });
            }
            return backend_->createPipelineCache(desc);
        });
    }

    Status Device::destroyPipelineCache(PipelineCacheHandle pipelineCache)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot destroy a pipeline cache on an invalid device.",
                });
            }
            return backend_->destroyPipelineCache(pipelineCache);
        });
    }

    Result<std::vector<std::byte>> Device::pipelineCacheData(PipelineCacheHandle pipelineCache)
    {
        return guardBadAlloc([&]() -> Result<std::vector<std::byte>> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot serialize a pipeline cache on an invalid device.",
                });
            }
            return backend_->pipelineCacheData(pipelineCache);
        });
    }

    Result<PipelineHandle> Device::createComputePipeline(const ComputePipelineDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<PipelineHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a compute pipeline on an invalid device.",
                });
            }
            return backend_->createComputePipeline(desc);
        });
    }

    Result<PipelineHandle> Device::createGraphicsPipeline(const GraphicsPipelineDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<PipelineHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a graphics pipeline on an invalid device.",
                });
            }
            if (auto valid = detail::validateVertexInputAlignment(desc.vertexBuffers); !valid)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(valid.error())));
            }
            return backend_->createGraphicsPipeline(desc);
        });
    }

    Result<PipelineCompilationHandle> Device::beginComputePipelineCompilation(const ComputePipelineDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<PipelineCompilationHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot begin compute-pipeline compilation on an invalid device.",
                });
            }
            return backend_->beginComputePipelineCompilation(desc);
        });
    }

    Result<PipelineCompilationHandle> Device::beginGraphicsPipelineCompilation(const GraphicsPipelineDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<PipelineCompilationHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot begin graphics-pipeline compilation on an invalid device.",
                });
            }
            if (auto valid = detail::validateVertexInputAlignment(desc.vertexBuffers); !valid)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(valid.error())));
            }
            return backend_->beginGraphicsPipelineCompilation(desc);
        });
    }

    Result<PipelineCompilationState> Device::pipelineCompilationState(PipelineCompilationHandle compilation)
    {
        return guardBadAlloc([&]() -> Result<PipelineCompilationState> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot query pipeline compilation on an invalid device.",
                });
            }
            return backend_->pipelineCompilationState(compilation);
        });
    }

    Result<PipelineHandle> Device::waitPipelineCompilation(PipelineCompilationHandle compilation)
    {
        return guardBadAlloc([&]() -> Result<PipelineHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot wait for pipeline compilation on an invalid device.",
                });
            }
            return backend_->waitPipelineCompilation(compilation);
        });
    }

    Status Device::destroyPipelineCompilation(PipelineCompilationHandle compilation)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot destroy pipeline compilation on an invalid device.",
                });
            }
            return backend_->destroyPipelineCompilation(compilation);
        });
    }

    Result<RayTracingPipelineLibraryHandle> Device::createRayTracingPipelineLibrary(
        const RayTracingPipelineLibraryDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<RayTracingPipelineLibraryHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a ray-tracing pipeline library on an invalid device.",
                });
            }
            return backend_->createRayTracingPipelineLibrary(desc);
        });
    }

    Status Device::destroyRayTracingPipelineLibrary(RayTracingPipelineLibraryHandle library)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot destroy a ray-tracing pipeline library on an invalid device.",
                });
            }
            return backend_->destroyRayTracingPipelineLibrary(library);
        });
    }

    Result<PipelineHandle> Device::createRayTracingPipeline(const RayTracingPipelineDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<PipelineHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a ray-tracing pipeline on an invalid device.",
                });
            }
            return backend_->createRayTracingPipeline(desc);
        });
    }

    Status Device::getRayTracingShaderGroupHandles(PipelineHandle pipeline,
                                                   std::uint32_t firstGroup,
                                                   std::uint32_t groupCount,
                                                   std::span<std::byte> destination)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot query ray-tracing shader-group handles on an invalid device.",
                });
            }
            return backend_->getRayTracingShaderGroupHandles(pipeline, firstGroup, groupCount, destination);
        });
    }

    Result<std::uint32_t> Device::rayTracingShaderGroupRecordSize(PipelineHandle pipeline, std::uint32_t groupIndex)
    {
        return guardBadAlloc([&]() -> Result<std::uint32_t> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot query a ray-tracing shader-group record size on an invalid device.",
                });
            }
            return backend_->rayTracingShaderGroupRecordSize(pipeline, groupIndex);
        });
    }

    Status Device::writeRayTracingShaderGroupRecord(PipelineHandle pipeline,
                                                    std::uint32_t groupIndex,
                                                    std::span<const std::byte> localConstants,
                                                    std::span<std::byte> destination)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot encode a ray-tracing shader-group record on an invalid device.",
                });
            }
            return backend_->writeRayTracingShaderGroupRecord(pipeline, groupIndex, localConstants, destination);
        });
    }

    Status Device::destroyPipeline(PipelineHandle pipeline)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot destroy a pipeline on an invalid device.",
                });
            }
            return backend_->destroyPipeline(pipeline);
        });
    }

    Status Device::bindComputePipeline(CommandBuffer& commandBuffer, PipelineHandle pipeline)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Compute pipeline binding requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->bindComputePipeline(*commandBuffer.backend_, pipeline);
        });
    }

    Status Device::bindComputeBindingSet(CommandBuffer& commandBuffer,
                                         std::uint32_t setIndex,
                                         BindingSetHandle bindingSet,
                                         std::span<const std::uint32_t> dynamicOffsets)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Compute binding-set binding requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->bindComputeBindingSet(*commandBuffer.backend_, setIndex, bindingSet, dynamicOffsets);
        });
    }

    Status Device::bindComputeBindingSets(CommandBuffer& commandBuffer,
                                          std::uint32_t firstSet,
                                          std::span<const BindingSetHandle> bindingSets,
                                          std::span<const std::uint32_t> dynamicOffsets)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Compute binding-set batch binding requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->bindComputeBindingSets(*commandBuffer.backend_, firstSet, bindingSets, dynamicOffsets);
        });
    }

    Status Device::setComputePushConstants(CommandBuffer& commandBuffer,
                                           std::uint32_t offset,
                                           std::span<const std::byte> data)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Compute push constants require a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            if (auto bound = validateRecordingState(*commandBuffer.backend_, detail::requireBoundComputePipeline);
                !bound)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(bound.error())));
            }
            if (auto valid = detail::validateComputePushConstants(offset, data); !valid)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(valid.error())));
            }
            return backend_->setComputePushConstants(*commandBuffer.backend_, offset, data);
        });
    }

    Status Device::dispatch(CommandBuffer& commandBuffer,
                            std::uint32_t groupCountX,
                            std::uint32_t groupCountY,
                            std::uint32_t groupCountZ)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(backend_.get(),
                                                        commandBuffer.backend_.get(),
                                                        "Compute dispatch requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->dispatch(*commandBuffer.backend_, groupCountX, groupCountY, groupCountZ);
        });
    }

    Status Device::dispatchIndirect(CommandBuffer& commandBuffer,
                                    BufferHandle argumentBuffer,
                                    std::uint64_t argumentOffset)
    {
        return guardBadAlloc([&]() -> Status {
            return dispatchIndirect(commandBuffer,
                                    IndirectDispatchDesc{
                                        .argumentBuffer = argumentBuffer,
                                        .argumentOffset = argumentOffset,
                                    });
        });
    }

    Status Device::dispatchIndirect(CommandBuffer& commandBuffer, const IndirectDispatchDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Indirect compute dispatch requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->dispatchIndirect(*commandBuffer.backend_, desc);
        });
    }

    Status Device::bindRayTracingPipeline(CommandBuffer& commandBuffer, PipelineHandle pipeline)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Ray-tracing pipeline binding requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->bindRayTracingPipeline(*commandBuffer.backend_, pipeline);
        });
    }

    Status Device::bindRayTracingBindingSet(CommandBuffer& commandBuffer,
                                            std::uint32_t setIndex,
                                            BindingSetHandle bindingSet,
                                            std::span<const std::uint32_t> dynamicOffsets)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Ray-tracing binding-set binding requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->bindRayTracingBindingSet(*commandBuffer.backend_, setIndex, bindingSet, dynamicOffsets);
        });
    }

    Status Device::bindRayTracingBindingSets(CommandBuffer& commandBuffer,
                                             std::uint32_t firstSet,
                                             std::span<const BindingSetHandle> bindingSets,
                                             std::span<const std::uint32_t> dynamicOffsets)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Ray-tracing binding-set batch binding requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->bindRayTracingBindingSets(*commandBuffer.backend_, firstSet, bindingSets, dynamicOffsets);
        });
    }

    Status Device::setRayTracingPushConstants(CommandBuffer& commandBuffer,
                                              std::uint32_t offset,
                                              std::span<const std::byte> data)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Ray-tracing push constants require a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->setRayTracingPushConstants(*commandBuffer.backend_, offset, data);
        });
    }

    Status Device::dispatchRays(CommandBuffer& commandBuffer, const RayTracingDispatchDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(backend_.get(),
                                                        commandBuffer.backend_.get(),
                                                        "Ray dispatch requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->dispatchRays(*commandBuffer.backend_, desc);
        });
    }

    Status Device::dispatchRaysIndirect(CommandBuffer& commandBuffer, const RayTracingIndirectDispatchDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Indirect ray dispatch requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->dispatchRaysIndirect(*commandBuffer.backend_, desc);
        });
    }

    Status Device::beginRendering(CommandBuffer& commandBuffer, const RenderingDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Beginning rendering requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->beginRendering(*commandBuffer.backend_, desc);
        });
    }

    Status Device::endRendering(CommandBuffer& commandBuffer)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(backend_.get(),
                                                        commandBuffer.backend_.get(),
                                                        "Ending rendering requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            if (auto active = validateRecordingState(*commandBuffer.backend_, detail::validateRenderingEnd); !active)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(active.error())));
            }
            return backend_->endRendering(*commandBuffer.backend_);
        });
    }

    Status Device::clearColorAttachment(CommandBuffer& commandBuffer,
                                        std::uint32_t attachmentIndex,
                                        const ClearColor& clearValue,
                                        std::span<const ScissorRect> rectangles)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Clearing a color attachment requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->clearColorAttachment(*commandBuffer.backend_, attachmentIndex, clearValue, rectangles);
        });
    }

    Status Device::clearDepthStencilAttachment(CommandBuffer& commandBuffer,
                                               DepthStencilClearAspect aspects,
                                               float depth,
                                               std::uint32_t stencil,
                                               std::span<const ScissorRect> rectangles)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Clearing a depth-stencil attachment requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->clearDepthStencilAttachment(*commandBuffer.backend_, aspects, depth, stencil, rectangles);
        });
    }

    Status Device::bindGraphicsPipeline(CommandBuffer& commandBuffer, PipelineHandle pipeline)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Graphics pipeline binding requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->bindGraphicsPipeline(*commandBuffer.backend_, pipeline);
        });
    }

    Status Device::bindGraphicsBindingSet(CommandBuffer& commandBuffer,
                                          std::uint32_t setIndex,
                                          BindingSetHandle bindingSet,
                                          std::span<const std::uint32_t> dynamicOffsets)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Graphics binding-set binding requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->bindGraphicsBindingSet(*commandBuffer.backend_, setIndex, bindingSet, dynamicOffsets);
        });
    }

    Status Device::bindGraphicsBindingSets(CommandBuffer& commandBuffer,
                                           std::uint32_t firstSet,
                                           std::span<const BindingSetHandle> bindingSets,
                                           std::span<const std::uint32_t> dynamicOffsets)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Graphics binding-set batch binding requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->bindGraphicsBindingSets(*commandBuffer.backend_, firstSet, bindingSets, dynamicOffsets);
        });
    }

    Status Device::setGraphicsPushConstants(CommandBuffer& commandBuffer,
                                            ShaderStage visibility,
                                            std::uint32_t offset,
                                            std::span<const std::byte> data)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Graphics push constants require a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            if (auto bound = validateRecordingState(*commandBuffer.backend_, detail::requireBoundGraphicsPipeline);
                !bound)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(bound.error())));
            }
            if (auto valid = detail::validateGraphicsPushConstants(visibility, offset, data); !valid)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(valid.error())));
            }
            return backend_->setGraphicsPushConstants(*commandBuffer.backend_, visibility, offset, data);
        });
    }

    Status Device::setViewport(CommandBuffer& commandBuffer, const Viewport& viewport)
    {
        return guardBadAlloc([&]() -> Status {
            const std::array viewports{viewport};
            return setViewports(commandBuffer, viewports);
        });
    }

    Status Device::setViewports(CommandBuffer& commandBuffer, std::span<const Viewport> viewports)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Setting viewports requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            if (auto active = validateRecordingState(*commandBuffer.backend_,
                                                     detail::requireActiveRendering,
                                                     "Viewports require an active rendering scope.");
                !active)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(active.error())));
            }
            const auto countExceededCode =
                backend_->backendName() == "Metal" ? ErrorCode::invalidArgument : ErrorCode::unsupported;
            if (auto valid = detail::validateViewports(
                    viewports, backend_->capabilities().limits.maxViewports, countExceededCode);
                !valid)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(valid.error())));
            }
            return backend_->setViewports(*commandBuffer.backend_, viewports);
        });
    }

    Status Device::setScissor(CommandBuffer& commandBuffer, const ScissorRect& scissor)
    {
        return guardBadAlloc([&]() -> Status {
            const std::array scissors{scissor};
            return setScissors(commandBuffer, scissors);
        });
    }

    Status Device::setScissors(CommandBuffer& commandBuffer, std::span<const ScissorRect> scissors)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Setting scissor rectangles requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            if (auto active = validateRecordingState(*commandBuffer.backend_,
                                                     detail::requireActiveRendering,
                                                     "Scissor rectangles require an active rendering scope.");
                !active)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(active.error())));
            }
            const auto countExceededCode =
                backend_->backendName() == "Metal" ? ErrorCode::invalidArgument : ErrorCode::unsupported;
            if (auto valid =
                    detail::validateScissors(scissors, backend_->capabilities().limits.maxViewports, countExceededCode);
                !valid)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(valid.error())));
            }
            return backend_->setScissors(*commandBuffer.backend_, scissors);
        });
    }

    Status Device::setBlendConstants(CommandBuffer& commandBuffer, const BlendConstants& constants)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(backend_.get(),
                                                        commandBuffer.backend_.get(),
                                                        "Blend constants require a live device and command buffer.");
                !valid)
            {
                return valid;
            }
            if (auto active = validateRecordingState(*commandBuffer.backend_,
                                                     detail::requireActiveRendering,
                                                     "Blend constants require an active rendering scope.");
                !active)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(active.error())));
            }
            if (auto valid = detail::validateBlendConstants(constants); !valid)
            {
                return std::unexpected(detail::preferDeviceLossError(*backend_, std::move(valid.error())));
            }
            return backend_->setBlendConstants(*commandBuffer.backend_, constants);
        });
    }

    Status Device::setDepthBias(CommandBuffer& commandBuffer, const DepthBias& bias)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Dynamic depth bias requires a live device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->setDepthBias(*commandBuffer.backend_, bias);
        });
    }

    Status Device::setDepthBounds(CommandBuffer& commandBuffer, const DepthBounds& bounds)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Dynamic depth bounds require a live device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->setDepthBounds(*commandBuffer.backend_, bounds);
        });
    }

    Status Device::setStencilReference(CommandBuffer& commandBuffer, std::uint32_t reference)
    {
        return guardBadAlloc(
            [&]() -> Status { return setStencilReference(commandBuffer, StencilFace::frontAndBack, reference); });
    }

    Status Device::setStencilReference(CommandBuffer& commandBuffer, StencilFace face, std::uint32_t reference)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "A stencil reference requires a live device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->setStencilReference(*commandBuffer.backend_, face, reference);
        });
    }

    Status Device::setShadingRate(CommandBuffer& commandBuffer, ShadingRate rate)
    {
        return guardBadAlloc([&]() -> Status { return setShadingRate(commandBuffer, rate, {}); });
    }

    Status Device::setShadingRate(CommandBuffer& commandBuffer, ShadingRate rate, const ShadingRateCombiners& combiners)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(backend_.get(),
                                                        commandBuffer.backend_.get(),
                                                        "A shading rate requires a live device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->setShadingRate(*commandBuffer.backend_, rate, combiners);
        });
    }

    Status Device::setShadingRateImage(CommandBuffer& commandBuffer, TextureHandle image)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "A shading-rate image requires a live device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->setShadingRateImage(*commandBuffer.backend_, image);
        });
    }

    Status Device::bindVertexBuffer(CommandBuffer& commandBuffer,
                                    std::uint32_t slot,
                                    BufferHandle buffer,
                                    std::uint64_t offset)
    {
        return guardBadAlloc([&]() -> Status {
            const VertexBufferBinding binding{.buffer = buffer, .offset = offset};
            return bindVertexBuffers(commandBuffer, slot, std::span{&binding, 1U});
        });
    }

    Status Device::bindVertexBuffers(CommandBuffer& commandBuffer,
                                     std::uint32_t firstSlot,
                                     std::span<const VertexBufferBinding> bindings)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Vertex-buffer binding requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->bindVertexBuffers(*commandBuffer.backend_, firstSlot, bindings);
        });
    }

    Status Device::bindStreamOutputBuffers(CommandBuffer& commandBuffer,
                                           std::uint32_t firstSlot,
                                           std::span<const StreamOutputBufferBinding> bindings)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Stream-output binding requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->bindStreamOutputBuffers(*commandBuffer.backend_, firstSlot, bindings);
        });
    }

    Status Device::bindIndexBuffer(CommandBuffer& commandBuffer,
                                   BufferHandle buffer,
                                   IndexFormat format,
                                   std::uint64_t offset)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Index-buffer binding requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->bindIndexBuffer(*commandBuffer.backend_, buffer, format, offset);
        });
    }

    Status Device::draw(CommandBuffer& commandBuffer,
                        std::uint32_t vertexCount,
                        std::uint32_t instanceCount,
                        std::uint32_t firstVertex,
                        std::uint32_t firstInstance)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(backend_.get(),
                                                        commandBuffer.backend_.get(),
                                                        "Drawing requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->draw(*commandBuffer.backend_, vertexCount, instanceCount, firstVertex, firstInstance);
        });
    }

    Status Device::drawIndexed(CommandBuffer& commandBuffer,
                               std::uint32_t indexCount,
                               std::uint32_t instanceCount,
                               std::uint32_t firstIndex,
                               std::int32_t baseVertex,
                               std::uint32_t firstInstance)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(backend_.get(),
                                                        commandBuffer.backend_.get(),
                                                        "Indexed drawing requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->drawIndexed(
                *commandBuffer.backend_, indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
        });
    }

    Status Device::dispatchMesh(CommandBuffer& commandBuffer,
                                std::uint32_t groupCountX,
                                std::uint32_t groupCountY,
                                std::uint32_t groupCountZ)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(backend_.get(),
                                                        commandBuffer.backend_.get(),
                                                        "Mesh dispatch requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->dispatchMesh(*commandBuffer.backend_, groupCountX, groupCountY, groupCountZ);
        });
    }

    Status Device::dispatchMeshIndirect(CommandBuffer& commandBuffer, const IndirectMeshDispatchDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Indirect mesh dispatch requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->dispatchMeshIndirect(*commandBuffer.backend_, desc);
        });
    }

    Status Device::drawIndirect(CommandBuffer& commandBuffer, const IndirectDrawDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(backend_.get(),
                                                        commandBuffer.backend_.get(),
                                                        "Indirect drawing requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->drawIndirect(*commandBuffer.backend_, desc);
        });
    }

    Status Device::drawIndexedIndirect(CommandBuffer& commandBuffer, const IndirectDrawDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Indexed indirect drawing requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->drawIndexedIndirect(*commandBuffer.backend_, desc);
        });
    }

    Status Device::beginConditionalExecution(CommandBuffer& commandBuffer, const ConditionalExecutionDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Beginning conditional execution requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->beginConditionalExecution(*commandBuffer.backend_, desc);
        });
    }

    Status Device::endConditionalExecution(CommandBuffer& commandBuffer)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Ending conditional execution requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->endConditionalExecution(*commandBuffer.backend_);
        });
    }

    Result<QueryPoolHandle> Device::createTimestampQueryPool(const TimestampQueryPoolDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<QueryPoolHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a timestamp query pool on an invalid device.",
                });
            }
            return backend_->createTimestampQueryPool(desc);
        });
    }

    Status Device::destroyQueryPool(QueryPoolHandle queryPool)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot destroy a query pool on an invalid device.",
                });
            }
            return backend_->destroyQueryPool(queryPool);
        });
    }

    Status Device::resetQueryPool(CommandBuffer& commandBuffer,
                                  QueryPoolHandle queryPool,
                                  std::uint32_t firstQuery,
                                  std::uint32_t queryCount)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(backend_.get(),
                                                        commandBuffer.backend_.get(),
                                                        "Query-pool resets require a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->resetQueryPool(*commandBuffer.backend_, queryPool, firstQuery, queryCount);
        });
    }

    Status Device::writeTimestamp(CommandBuffer& commandBuffer, QueryPoolHandle queryPool, std::uint32_t queryIndex)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(backend_.get(),
                                                        commandBuffer.backend_.get(),
                                                        "Timestamp writes require a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->writeTimestamp(*commandBuffer.backend_, queryPool, queryIndex);
        });
    }

    Status Device::beginGpuProfileScope(CommandBuffer& commandBuffer, const GpuProfileScopeDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Beginning a GPU profile scope requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->beginGpuProfileScope(*commandBuffer.backend_, desc);
        });
    }

    Status Device::endGpuProfileScope(CommandBuffer& commandBuffer)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Ending a GPU profile scope requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->endGpuProfileScope(*commandBuffer.backend_);
        });
    }

    Status Device::resolveTimestamps(CommandBuffer& commandBuffer, const TimestampQueryResolveDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Timestamp resolves require a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->resolveTimestamps(*commandBuffer.backend_, desc);
        });
    }

    Result<QueryPoolHandle> Device::createOcclusionQueryPool(const OcclusionQueryPoolDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<QueryPoolHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create an occlusion query pool on an invalid device.",
                });
            }
            return backend_->createOcclusionQueryPool(desc);
        });
    }

    Status Device::beginOcclusionQuery(CommandBuffer& commandBuffer,
                                       QueryPoolHandle queryPool,
                                       std::uint32_t queryIndex)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Beginning an occlusion query requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->beginOcclusionQuery(*commandBuffer.backend_, queryPool, queryIndex);
        });
    }

    Status Device::endOcclusionQuery(CommandBuffer& commandBuffer)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid =
                    validateOwnedCommandBuffer(backend_.get(),
                                               commandBuffer.backend_.get(),
                                               "Ending an occlusion query requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->endOcclusionQuery(*commandBuffer.backend_);
        });
    }

    Status Device::resolveOcclusionQueries(CommandBuffer& commandBuffer, const OcclusionQueryResolveDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Resolving occlusion queries requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->resolveOcclusionQueries(*commandBuffer.backend_, desc);
        });
    }

    Result<QueryPoolHandle> Device::createPipelineStatisticsQueryPool(const PipelineStatisticsQueryPoolDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<QueryPoolHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a pipeline-statistics query pool on an invalid device.",
                });
            }
            return backend_->createPipelineStatisticsQueryPool(desc);
        });
    }

    Status Device::beginPipelineStatisticsQuery(CommandBuffer& commandBuffer,
                                                QueryPoolHandle queryPool,
                                                std::uint32_t queryIndex)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Beginning a pipeline-statistics query requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->beginPipelineStatisticsQuery(*commandBuffer.backend_, queryPool, queryIndex);
        });
    }

    Status Device::endPipelineStatisticsQuery(CommandBuffer& commandBuffer)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Ending a pipeline-statistics query requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->endPipelineStatisticsQuery(*commandBuffer.backend_);
        });
    }

    Status Device::resolvePipelineStatisticsQueries(CommandBuffer& commandBuffer,
                                                    const PipelineStatisticsQueryResolveDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Resolving pipeline-statistics queries requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->resolvePipelineStatisticsQueries(*commandBuffer.backend_, desc);
        });
    }

    Result<QueryPoolHandle> Device::createStreamOutputStatisticsQueryPool(
        const StreamOutputStatisticsQueryPoolDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<QueryPoolHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a stream-output-statistics query pool on an invalid device.",
                });
            }
            return backend_->createStreamOutputStatisticsQueryPool(desc);
        });
    }

    Status Device::beginStreamOutputStatisticsQuery(CommandBuffer& commandBuffer,
                                                    QueryPoolHandle queryPool,
                                                    std::uint32_t queryIndex)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Beginning a stream-output-statistics query requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->beginStreamOutputStatisticsQuery(*commandBuffer.backend_, queryPool, queryIndex);
        });
    }

    Status Device::endStreamOutputStatisticsQuery(CommandBuffer& commandBuffer, std::uint32_t stream)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Ending a stream-output-statistics query requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->endStreamOutputStatisticsQuery(*commandBuffer.backend_, stream);
        });
    }

    Status Device::resolveStreamOutputStatisticsQueries(CommandBuffer& commandBuffer,
                                                        const StreamOutputStatisticsQueryResolveDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Resolving stream-output-statistics queries requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->resolveStreamOutputStatisticsQueries(*commandBuffer.backend_, desc);
        });
    }

    Result<QueryPoolHandle> Device::createAccelerationStructureCompactedSizeQueryPool(
        const AccelerationStructureCompactedSizeQueryPoolDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<QueryPoolHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create an acceleration-structure compacted-size query pool on an invalid device.",
                });
            }
            return backend_->createAccelerationStructureCompactedSizeQueryPool(desc);
        });
    }

    Status Device::writeAccelerationStructureCompactedSize(CommandBuffer& commandBuffer,
                                                           QueryPoolHandle queryPool,
                                                           std::uint32_t queryIndex,
                                                           AccelerationStructureHandle source)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(backend_.get(),
                                                        commandBuffer.backend_.get(),
                                                        "Writing an acceleration-structure compacted-size query "
                                                        "requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->writeAccelerationStructureCompactedSize(
                *commandBuffer.backend_, queryPool, queryIndex, source);
        });
    }

    Status Device::resolveAccelerationStructureCompactedSizes(
        CommandBuffer& commandBuffer, const AccelerationStructureCompactedSizeQueryResolveDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(
                    backend_.get(),
                    commandBuffer.backend_.get(),
                    "Resolving acceleration-structure compacted sizes requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->resolveAccelerationStructureCompactedSizes(*commandBuffer.backend_, desc);
        });
    }

    Result<QueryPoolHandle> Device::createAccelerationStructureSerializationInfoQueryPool(
        const AccelerationStructureSerializationInfoQueryPoolDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<QueryPoolHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create an acceleration-structure serialization-info query pool on an invalid device.",
                });
            }
            return backend_->createAccelerationStructureSerializationInfoQueryPool(desc);
        });
    }

    Status Device::writeAccelerationStructureSerializationInfo(CommandBuffer& commandBuffer,
                                                               QueryPoolHandle queryPool,
                                                               std::uint32_t queryIndex,
                                                               AccelerationStructureHandle source)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(backend_.get(),
                                                        commandBuffer.backend_.get(),
                                                        "Writing acceleration-structure serialization information "
                                                        "requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->writeAccelerationStructureSerializationInfo(
                *commandBuffer.backend_, queryPool, queryIndex, source);
        });
    }

    Status Device::resolveAccelerationStructureSerializationInfo(
        CommandBuffer& commandBuffer, const AccelerationStructureSerializationInfoQueryResolveDesc& desc)
    {
        return guardBadAlloc([&]() -> Status {
            if (auto valid = validateOwnedCommandBuffer(backend_.get(),
                                                        commandBuffer.backend_.get(),
                                                        "Resolving acceleration-structure serialization information "
                                                        "requires a valid device and command buffer.");
                !valid)
            {
                return valid;
            }
            return backend_->resolveAccelerationStructureSerializationInfo(*commandBuffer.backend_, desc);
        });
    }

    Result<SurfaceHandle> Device::createSurface(const SurfaceDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<SurfaceHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a surface on an invalid device.",
                });
            }
            return backend_->createSurface(desc);
        });
    }

    Result<SurfaceCapabilities> Device::querySurfaceCapabilities(SurfaceHandle surface) const
    {
        return guardBadAlloc([&]() -> Result<SurfaceCapabilities> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot query surface capabilities on an invalid device.",
                });
            }
            return backend_->querySurfaceCapabilities(surface);
        });
    }

    Status Device::destroySurface(SurfaceHandle surface)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot destroy a surface on an invalid device.",
                });
            }
            return backend_->destroySurface(surface);
        });
    }

    Result<SwapchainHandle> Device::createSwapchain(const SwapchainDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<SwapchainHandle> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot create a swapchain on an invalid device.",
                });
            }
            return backend_->createSwapchain(desc);
        });
    }

    Status Device::destroySwapchain(SwapchainHandle swapchain)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot destroy a swapchain on an invalid device.",
                });
            }
            return backend_->destroySwapchain(swapchain);
        });
    }

    Status Device::resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width, std::uint32_t height)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot resize a swapchain on an invalid device.",
                });
            }
            return backend_->resizeSwapchain(swapchain, width, height);
        });
    }

    namespace
    {
        [[nodiscard]] Result<AcquiredSwapchainImage> acquireBackendImage(
            detail::BackendDevice* backend, SwapchainHandle swapchain, std::optional<std::chrono::nanoseconds> timeout)
        {
            return guardBadAlloc([&]() -> Result<AcquiredSwapchainImage> {
                if (backend == nullptr)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "Cannot acquire a swapchain image on an invalid device.",
                    });
                }
                return backend->acquireNextImage(swapchain, timeout);
            });
        }
    } // namespace

    Result<AcquiredSwapchainImage> Device::acquireNextImage(SwapchainHandle swapchain)
    {
        return acquireBackendImage(backend_.get(), swapchain, std::nullopt);
    }

    Result<AcquiredSwapchainImage> Device::acquireNextImage(SwapchainHandle swapchain, std::chrono::nanoseconds timeout)
    {
        return acquireBackendImage(backend_.get(), swapchain, std::max(timeout, std::chrono::nanoseconds::zero()));
    }

    bool Device::hasAcquiredSwapchainImage(SwapchainHandle swapchain) const noexcept
    {
        if (backend_ == nullptr || !swapchain.isValid())
        {
            return false;
        }
        return backend_->hasAcquiredSwapchainImage(swapchain);
    }

    Status Device::cancelSwapchainImage(SwapchainHandle swapchain)
    {
        return guardBadAlloc([&]() -> Status {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot cancel a swapchain image on an invalid device.",
                });
            }
            return backend_->cancelSwapchainImage(swapchain);
        });
    }

    Result<PresentStatus> Device::present(SwapchainHandle swapchain, TimelinePoint renderComplete)
    {
        return guardBadAlloc([&]() -> Result<PresentStatus> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot present a swapchain on an invalid device.",
                });
            }
            return backend_->present(swapchain, renderComplete);
        });
    }

    Result<SwapchainPresentationInfo> Device::querySwapchainPresentation(SwapchainHandle swapchain) const
    {
        return guardBadAlloc([&]() -> Result<SwapchainPresentationInfo> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot query swapchain presentation on an invalid device.",
                });
            }
            return backend_->querySwapchainPresentation(swapchain);
        });
    }

    Result<SwapchainFrameTiming> Device::querySwapchainFrameTiming(SwapchainHandle swapchain) const
    {
        return guardBadAlloc([&]() -> Result<SwapchainFrameTiming> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot query swapchain frame timing on an invalid device.",
                });
            }
            return backend_->querySwapchainFrameTiming(swapchain);
        });
    }

    Result<SwapchainPresentTimings> Device::querySwapchainPresentTimings(SwapchainHandle swapchain,
                                                                         std::uint64_t firstSequence,
                                                                         std::span<PresentTiming> records) const
    {
        return guardBadAlloc([&]() -> Result<SwapchainPresentTimings> {
            if (backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot query swapchain present timings on an invalid device.",
                });
            }
            return backend_->querySwapchainPresentTimings(swapchain, firstSequence, detail::PresentTimingSink{records});
        });
    }

    Result<std::vector<AdapterInfo>> enumerateAdapters()
    {
        return guardBadAlloc(
            [&]() -> Result<std::vector<AdapterInfo>> { return enumerateAdapters(defaultHostBackend()); });
    }

    Result<std::vector<AdapterInfo>> enumerateAdapters(Backend backend)
    {
        return guardBadAlloc(
            [&]() -> Result<std::vector<AdapterInfo>> { return detail::enumerateBackendAdapters(backend); });
    }

    Result<Device> createDevice(const DeviceDesc& desc)
    {
        return guardBadAlloc([&]() -> Result<Device> {
            auto backend = detail::createBackendDevice(desc);
            if (!backend)
            {
                return std::unexpected(std::move(backend.error()));
            }

            return Device(std::move(*backend));
        });
    }

} // namespace rachis
