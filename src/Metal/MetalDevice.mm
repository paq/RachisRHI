#include "Metal/MetalInternal.h"

namespace rachis::metal
{

    [[nodiscard]] rachis::Error metalSelectionError(rachis::ErrorCode code, std::string message)
    {
        return rachis::Error{
            code,
            0,
            std::move(message),
        };
    }

    [[nodiscard]] rachis::Error metalProfileError(const rachis::MetalProfileDecision& decision)
    {
        std::string message{rachis::describeMetalMacosProfileRejection(decision)};
        if (decision.missingItemCount > 0)
        {
            message += " Missing:";
            for (std::uint32_t i = 0; i < decision.missingItemCount; ++i)
            {
                message += ' ';
                message += decision.missingItems[i];
            }
        }
        const auto code = decision.rejection == rachis::MetalProfileRejection::deviceUnavailable
                              ? rachis::ErrorCode::backendUnavailable
                              : rachis::ErrorCode::unsupported;
        return metalSelectionError(code, std::move(message));
    }

    [[nodiscard]] rachis::AdapterInfo adapterFromDevice(id<MTLDevice> device)
    {
        rachis::AdapterInfo info;
        info.id.value = device.registryID;
        const char* const deviceName = device.name.UTF8String;
        info.name = deviceName != nullptr ? deviceName : "Apple GPU";
        info.type = device.hasUnifiedMemory ? rachis::AdapterType::integrated : rachis::AdapterType::discrete;
        info.vendorId = 0x106B;
        info.deviceId = static_cast<std::uint32_t>(device.registryID & 0xffffffffU);
        info.dedicatedVideoMemory = 0;
        info.sharedSystemMemory = device.recommendedMaxWorkingSetSize;
        info.backend = rachis::Backend::metal;
        return info;
    }

    [[nodiscard]] bool formatIsBc(rachis::Format format) noexcept
    {
        switch (format)
        {
        case rachis::Format::bc1RgbaUnorm:
        case rachis::Format::bc1RgbaUnormSrgb:
        case rachis::Format::bc2RgbaUnorm:
        case rachis::Format::bc2RgbaUnormSrgb:
        case rachis::Format::bc3RgbaUnorm:
        case rachis::Format::bc3RgbaUnormSrgb:
        case rachis::Format::bc4RUnorm:
        case rachis::Format::bc4RSnorm:
        case rachis::Format::bc5RgUnorm:
        case rachis::Format::bc5RgSnorm:
        case rachis::Format::bc6hRgbUfloat:
        case rachis::Format::bc6hRgbSfloat:
        case rachis::Format::bc7RgbaUnorm:
        case rachis::Format::bc7RgbaUnormSrgb:
            return true;
        default:
            return false;
        }
    }

    [[nodiscard]] bool formatIsDepth(rachis::Format format) noexcept
    {
        return format == rachis::Format::d32Float || format == rachis::Format::d24UnormS8Uint;
    }

    [[nodiscard]] bool formatIsInteger(rachis::Format format) noexcept
    {
        switch (format)
        {
        case rachis::Format::r8Uint:
        case rachis::Format::rg8Uint:
        case rachis::Format::rgba8Uint:
        case rachis::Format::r8Sint:
        case rachis::Format::rg8Sint:
        case rachis::Format::rgba8Sint:
        case rachis::Format::r16Uint:
        case rachis::Format::rg16Uint:
        case rachis::Format::rgba16Uint:
        case rachis::Format::r16Sint:
        case rachis::Format::rg16Sint:
        case rachis::Format::rgba16Sint:
        case rachis::Format::r32Uint:
        case rachis::Format::rg32Uint:
        case rachis::Format::rgba32Uint:
        case rachis::Format::r32Sint:
        case rachis::Format::rg32Sint:
        case rachis::Format::rgba32Sint:
        case rachis::Format::r64Uint:
            return true;
        default:
            return false;
        }
    }

    [[nodiscard]] bool formatAllowsViewReinterpretation(rachis::Format format) noexcept
    {
        switch (format)
        {
        case rachis::Format::rgba8Unorm:
        case rachis::Format::rgba8UnormSrgb:
        case rachis::Format::bgra8Unorm:
        case rachis::Format::bgra8UnormSrgb:
        case rachis::Format::bc1RgbaUnorm:
        case rachis::Format::bc1RgbaUnormSrgb:
        case rachis::Format::bc2RgbaUnorm:
        case rachis::Format::bc2RgbaUnormSrgb:
        case rachis::Format::bc3RgbaUnorm:
        case rachis::Format::bc3RgbaUnormSrgb:
        case rachis::Format::bc7RgbaUnorm:
        case rachis::Format::bc7RgbaUnormSrgb:
            return true;
        default:
            return false;
        }
    }

    [[nodiscard]] MTLPixelFormat metalPixelFormat(rachis::Format format) noexcept
    {
        switch (format)
        {
        case rachis::Format::r8Unorm:
            return MTLPixelFormatR8Unorm;
        case rachis::Format::rg8Unorm:
            return MTLPixelFormatRG8Unorm;
        case rachis::Format::rgba8Unorm:
            return MTLPixelFormatRGBA8Unorm;
        case rachis::Format::rgba8UnormSrgb:
            return MTLPixelFormatRGBA8Unorm_sRGB;
        case rachis::Format::bgra8Unorm:
            return MTLPixelFormatBGRA8Unorm;
        case rachis::Format::bgra8UnormSrgb:
            return MTLPixelFormatBGRA8Unorm_sRGB;
        case rachis::Format::r16Float:
            return MTLPixelFormatR16Float;
        case rachis::Format::rg16Float:
            return MTLPixelFormatRG16Float;
        case rachis::Format::rgba16Float:
            return MTLPixelFormatRGBA16Float;
        case rachis::Format::r32Float:
            return MTLPixelFormatR32Float;
        case rachis::Format::rg32Float:
            return MTLPixelFormatRG32Float;
        case rachis::Format::rgba32Float:
            return MTLPixelFormatRGBA32Float;
        case rachis::Format::d32Float:
            return MTLPixelFormatDepth32Float;
        case rachis::Format::d24UnormS8Uint:
            return MTLPixelFormatDepth24Unorm_Stencil8;
        case rachis::Format::bc1RgbaUnorm:
            return MTLPixelFormatBC1_RGBA;
        case rachis::Format::bc1RgbaUnormSrgb:
            return MTLPixelFormatBC1_RGBA_sRGB;
        case rachis::Format::bc2RgbaUnorm:
            return MTLPixelFormatBC2_RGBA;
        case rachis::Format::bc2RgbaUnormSrgb:
            return MTLPixelFormatBC2_RGBA_sRGB;
        case rachis::Format::bc3RgbaUnorm:
            return MTLPixelFormatBC3_RGBA;
        case rachis::Format::bc3RgbaUnormSrgb:
            return MTLPixelFormatBC3_RGBA_sRGB;
        case rachis::Format::bc4RUnorm:
            return MTLPixelFormatBC4_RUnorm;
        case rachis::Format::bc4RSnorm:
            return MTLPixelFormatBC4_RSnorm;
        case rachis::Format::bc5RgUnorm:
            return MTLPixelFormatBC5_RGUnorm;
        case rachis::Format::bc5RgSnorm:
            return MTLPixelFormatBC5_RGSnorm;
        case rachis::Format::bc6hRgbUfloat:
            return MTLPixelFormatBC6H_RGBUfloat;
        case rachis::Format::bc6hRgbSfloat:
            return MTLPixelFormatBC6H_RGBFloat;
        case rachis::Format::bc7RgbaUnorm:
            return MTLPixelFormatBC7_RGBAUnorm;
        case rachis::Format::bc7RgbaUnormSrgb:
            return MTLPixelFormatBC7_RGBAUnorm_sRGB;
        case rachis::Format::r32Uint:
            return MTLPixelFormatR32Uint;
        case rachis::Format::r8Uint:
            return MTLPixelFormatR8Uint;
        case rachis::Format::rg8Uint:
            return MTLPixelFormatRG8Uint;
        case rachis::Format::rgba8Uint:
            return MTLPixelFormatRGBA8Uint;
        case rachis::Format::r8Sint:
            return MTLPixelFormatR8Sint;
        case rachis::Format::rg8Sint:
            return MTLPixelFormatRG8Sint;
        case rachis::Format::rgba8Sint:
            return MTLPixelFormatRGBA8Sint;
        case rachis::Format::r16Uint:
            return MTLPixelFormatR16Uint;
        case rachis::Format::rg16Uint:
            return MTLPixelFormatRG16Uint;
        case rachis::Format::rgba16Uint:
            return MTLPixelFormatRGBA16Uint;
        case rachis::Format::r16Sint:
            return MTLPixelFormatR16Sint;
        case rachis::Format::rg16Sint:
            return MTLPixelFormatRG16Sint;
        case rachis::Format::rgba16Sint:
            return MTLPixelFormatRGBA16Sint;
        case rachis::Format::rg32Uint:
            return MTLPixelFormatRG32Uint;
        case rachis::Format::rgba32Uint:
            return MTLPixelFormatRGBA32Uint;
        case rachis::Format::r32Sint:
            return MTLPixelFormatR32Sint;
        case rachis::Format::rg32Sint:
            return MTLPixelFormatRG32Sint;
        case rachis::Format::rgba32Sint:
            return MTLPixelFormatRGBA32Sint;
        case rachis::Format::unknown:
        case rachis::Format::r64Uint:
            return MTLPixelFormatInvalid;
        }
        return MTLPixelFormatInvalid;
    }

    [[nodiscard]] NSString* metalNsString(std::string_view text) noexcept
    {
        if (text.empty())
        {
            return nil;
        }
        return [[NSString alloc] initWithBytes:text.data() length:text.size() encoding:NSUTF8StringEncoding];
    }

    [[nodiscard]] rachis::Error metalNsError(rachis::ErrorCode code,
                                             NSError* error,
                                             std::string_view operation,
                                             std::string_view encoderContext)
    {
        const char* const domainText = error.domain.UTF8String;
        const std::string_view domain = domainText != nullptr ? domainText : "";
        const std::int64_t nativeCode = error != nil ? static_cast<std::int64_t>(error.code) : 0;
        const char* const descriptionText = error.localizedDescription.UTF8String;
        const std::string_view description = descriptionText != nullptr ? descriptionText : "";
        return rachis::metal::makeMetalError(code, nativeCode, domain, operation, description, encoderContext);
    }

    [[nodiscard]] DeviceFaultStatus metalFaultStatusFromError(NSError* error) noexcept
    {
        if (error == nil || error.domain == nil)
        {
            return DeviceFaultStatus::unknown;
        }
        if ([error.domain isEqualToString:MTL4CommandQueueErrorDomain])
        {
            switch (static_cast<NSInteger>(error.code))
            {
            case MTL4CommandQueueErrorTimeout:
                return DeviceFaultStatus::hung;
            case MTL4CommandQueueErrorNotPermitted:
                return DeviceFaultStatus::invalidCall;
            case MTL4CommandQueueErrorOutOfMemory:
                return DeviceFaultStatus::outOfMemory;
            case 4:
                return DeviceFaultStatus::removed;
            case MTL4CommandQueueErrorAccessRevoked:
                return DeviceFaultStatus::reset;
            case MTL4CommandQueueErrorInternal:
                return DeviceFaultStatus::driverInternalError;
            default:
                return DeviceFaultStatus::unknown;
            }
        }
        if ([error.domain isEqualToString:MTLCommandBufferErrorDomain])
        {
            switch (static_cast<NSUInteger>(error.code))
            {
            case MTLCommandBufferErrorInternal:
                return DeviceFaultStatus::driverInternalError;
            case MTLCommandBufferErrorTimeout:
                return DeviceFaultStatus::hung;
            case MTLCommandBufferErrorAccessRevoked:
                return DeviceFaultStatus::reset;
            case MTLCommandBufferErrorOutOfMemory:
                return DeviceFaultStatus::outOfMemory;
            case MTLCommandBufferErrorPageFault:
            case MTLCommandBufferErrorNotPermitted:
            case MTLCommandBufferErrorInvalidResource:
            case MTLCommandBufferErrorMemoryless:
            case MTLCommandBufferErrorStackOverflow:
                return DeviceFaultStatus::invalidCall;
            case 11U:
                return DeviceFaultStatus::removed;
            default:
                return DeviceFaultStatus::unknown;
            }
        }
        return DeviceFaultStatus::unknown;
    }

    void publishMetalDeviceLoss(MetalDeviceNative& native,
                                const Error& error,
                                DeviceFaultStatus status,
                                bool injected,
                                std::string_view queueLabel,
                                std::string_view commandLabel)
    {
        std::shared_ptr<const Error> publishedError;
        std::uint32_t nativeReason = 0;
        bool publish = false;
        {
            const std::scoped_lock lock(native.lossMutex);
            if (native.lossError == nullptr)
            {
                DeviceFaultReport captured{};
                captured.status = status;
                captured.nativeReason = error.nativeCode != 0 ? error.nativeCode : 1;
                captured.dredAvailable = false;
                if (injected)
                {
                    captured.breadcrumbs.push_back(DeviceFaultBreadcrumb{
                        .queueLabel = "RACHIS_TESTING",
                        .commandListLabel = "injected device loss",
                    });
                }
                else if (!queueLabel.empty() || !commandLabel.empty())
                {
                    captured.breadcrumbs.push_back(DeviceFaultBreadcrumb{
                        .queueLabel = std::string(queueLabel),
                        .commandListLabel = std::string(commandLabel),
                    });
                }
                publishedError = std::make_shared<const Error>(error);
                native.lossError = publishedError;
                native.lossReport = std::make_shared<const DeviceFaultReport>(std::move(captured));
                native.lossInjected.store(injected, std::memory_order_release);
                native.operationalState.store(DeviceOperationalState::lost, std::memory_order_release);
                nativeReason = static_cast<std::uint32_t>(native.lossReport->nativeReason);
                publish = true;
            }
            if (!injected)
            {
                native.nativeLost.store(true, std::memory_order_release);
            }
        }
        if (publish && native.sink != nullptr)
        {
            native.sink->publish(DiagnosticSeverity::error, nativeReason, publishedError->message);
        }
    }

    [[nodiscard]] rachis::Error metalExceptionError(NSException* exception, std::string_view operation)
    {
        std::string message{operation};
        message += " raised ";
        const char* const name = exception.name.UTF8String;
        const char* const reason = exception.reason.UTF8String;
        message += name != nullptr ? name : "NSException";
        if (reason != nullptr)
        {
            message += ": ";
            message += reason;
        }
        return metalSelectionError(rachis::ErrorCode::nativeFailure, std::move(message));
    }

    [[nodiscard]] rachis::DiagnosticSeverity metalLogSeverity(MTLLogLevel level) noexcept
    {
        switch (level)
        {
        case MTLLogLevelFault:
            return rachis::DiagnosticSeverity::corruption;
        case MTLLogLevelError:
            return rachis::DiagnosticSeverity::error;
        case MTLLogLevelNotice:
            return rachis::DiagnosticSeverity::warning;
        case MTLLogLevelDebug:
        case MTLLogLevelInfo:
        case MTLLogLevelUndefined:
        default:
            return rachis::DiagnosticSeverity::information;
        }
    }

    [[nodiscard]] std::string sanitizeCaptureToken(std::string_view text)
    {
        std::string token;
        token.reserve(text.size());
        for (const unsigned char ch : text)
        {
            if ((std::isalnum(ch) != 0) || ch == '-' || ch == '_')
            {
                token.push_back(static_cast<char>(ch));
            }
            else if (!token.empty() && token.back() != '-')
            {
                token.push_back('-');
            }
        }
        return token.empty() ? std::string("workload") : token;
    }

#ifdef RACHIS_TESTING
    MetalTestingState& metalTestingState() noexcept
    {
        static MetalTestingState s_state;
        return s_state;
    }

    void bindMetalDiagnosticTestDevice(MetalDevice* device) noexcept
    {
        std::scoped_lock const lock(metalTestingState().mutex);
        metalTestingState().boundDevice = device;
    }

    void unbindMetalDiagnosticTestDevice(MetalDevice* device) noexcept
    {
        std::scoped_lock const lock(metalTestingState().mutex);
        if (metalTestingState().boundDevice == device)
        {
            metalTestingState().boundDevice = nullptr;
        }
    }

    Status consumeInjectedMetalPresentLoss(MetalDevice& device)
    {
        const auto injected = rachis::metal::test::nextPresentResult().exchange(0, std::memory_order_relaxed);
        if (injected == 0)
        {
            return {};
        }
        return device.injectDeviceLoss();
    }

    void recordMetalResidencyAdd() noexcept
    {
        metalTestingState().residencyAddCount.fetch_add(1, std::memory_order_relaxed);
    }

    void recordMetalResidencySkip() noexcept
    {
        metalTestingState().residencyDeduplicatedSkipCount.fetch_add(1, std::memory_order_relaxed);
    }

    void recordMetalResidencyCommit() noexcept
    {
        metalTestingState().residencyCommitCount.fetch_add(1, std::memory_order_relaxed);
    }

    void recordMetalArgumentTableCreate() noexcept
    {
        metalTestingState().argumentTableCreateCount.fetch_add(1, std::memory_order_relaxed);
    }

    void recordMetalArgumentTablePoolReuse() noexcept
    {
        metalTestingState().argumentTablePoolReuseCount.fetch_add(1, std::memory_order_relaxed);
    }

    void recordMetalArgumentTableEncoderReuse() noexcept
    {
        metalTestingState().argumentTableEncoderReuseCount.fetch_add(1, std::memory_order_relaxed);
    }

    void recordMetalPushConstantPageAllocate() noexcept
    {
        metalTestingState().pushConstantPageAllocateCount.fetch_add(1, std::memory_order_relaxed);
    }

    void recordMetalPushConstantSliceSuballocate() noexcept
    {
        metalTestingState().pushConstantSliceSuballocateCount.fetch_add(1, std::memory_order_relaxed);
    }

    void recordMetalTeardownQueueDrain() noexcept
    {
        metalTestingState().teardownQueueDrainCount.fetch_add(1, std::memory_order_relaxed);
    }

    void recordMetalRenderTimestampStage(MTLRenderStages stage) noexcept
    {
        auto& state = metalTestingState();
        auto& count = stage == MTLRenderStageVertex ? state.renderTimestampVertexStageCount
                                                    : state.renderTimestampFragmentStageCount;
        count.fetch_add(1, std::memory_order_relaxed);
    }

    void holdMetalCommandPoolResetForTesting() noexcept
    {
        std::shared_ptr<MetalCommitFeedbackGate> gate;
        {
            const std::scoped_lock lock(metalTestingState().mutex);
            gate = std::move(metalTestingState().nextCommandPoolResetGate);
        }
        if (gate != nullptr)
        {
            gate->waitUntilReleased();
        }
    }
#endif

    MetalDevice::MetalDevice(AdapterInfo adapter,
                             DeviceCapabilities capabilities,
                             std::vector<FormatProperties> formats,
                             QueueProperties graphicsQueue,
                             std::shared_ptr<MetalDeviceNative> native)
        : adapter_(std::move(adapter)), capabilities_(capabilities), formats_(std::move(formats)),
          graphicsQueue_(graphicsQueue), native_(std::move(native)),
          deviceToken_(native_ != nullptr ? native_->deviceToken : 0)
    {
    }

    namespace
    {

        constexpr std::chrono::milliseconds kMetalLostTeardownFeedbackTimeout{2000};

        void awaitLostQueueFeedbackForTeardown(MetalQueueTimeline& timeline,
                                               std::optional<std::chrono::steady_clock::time_point>& deadline)
        {
            if (!deadline.has_value())
            {
                deadline = std::chrono::steady_clock::now() + kMetalLostTeardownFeedbackTimeout;
            }
            const auto remaining =
                std::chrono::duration_cast<std::chrono::milliseconds>(*deadline - std::chrono::steady_clock::now());
            timeline.feedback.wait(timeline.lastSubmitted.load(std::memory_order_acquire),
                                   (std::max)(remaining, std::chrono::milliseconds::zero()));
        }

        void drainQueuesForTeardown(MetalDeviceNative& native) noexcept
        {
            if (native.queueGate == nullptr)
            {
                return;
            }
            const std::scoped_lock gateLock(native.queueGate->mutex);
#ifdef RACHIS_TESTING
            if (!native.nativeLost.load(std::memory_order_acquire))
            {
                recordMetalTeardownQueueDrain();
            }
#endif
            std::optional<std::chrono::steady_clock::time_point> lostDeadline;
            for (auto& timeline : native.timelines)
            {
                try
                {
                    if (!native.nativeLost.load(std::memory_order_acquire))
                    {
                        static_cast<void>(drainMetalQueue(native, timeline));
                    }
                    if (native.nativeLost.load(std::memory_order_acquire))
                    {
                        awaitLostQueueFeedbackForTeardown(timeline, lostDeadline);
                    }
                }
                catch (const std::exception& error)
                {
                    std::fputs("Metal queue teardown failed: ", stderr);
                    std::fputs(error.what(), stderr);
                    std::fputc('\n', stderr);
                }
            }
        }

        [[nodiscard]] std::vector<std::weak_ptr<MetalCommandPoolState>> takeCommandPoolsForTeardown(
            MetalDeviceNative& native, std::mutex& deviceMutex) noexcept
        {
            std::vector<std::weak_ptr<MetalCommandPoolState>> pools;
            {
                std::scoped_lock const lock(deviceMutex);
                pools = std::move(native.allCommandPools);
                native.commandPools.clear();
            }
            return pools;
        }

        void detachCommandEntryLeasesForShutdown(MetalCommandPoolState& pool) noexcept
        {
            for (const auto& entry : pool.entries)
            {
                if (entry != nullptr && entry->lease != nullptr)
                {
                    entry->lease->detachForShutdown();
                }
            }
            pool.available.clear();
            pool.quarantined.clear();
        }

        [[nodiscard]] std::unique_ptr<MetalCommandEntry> takeNextIdleCommandEntry(MetalCommandPoolState& pool,
                                                                                  std::size_t& index) noexcept
        {
            while (index < pool.entries.size())
            {
                auto& entry = pool.entries[index];
                if (entry != nullptr && entry->inFlight)
                {
                    ++index;
                    continue;
                }
                std::unique_ptr<MetalCommandEntry> idleEntry = std::move(entry);
                if (index + 1 < pool.entries.size())
                {
                    pool.entries[index] = std::move(pool.entries.back());
                }
                pool.entries.pop_back();
                return idleEntry;
            }
            return nullptr;
        }

        void releaseEmptyPoolNativeObjects(MetalCommandPoolState& pool) noexcept
        {
            if (!pool.entries.empty())
            {
                return;
            }
            pool.device = nil;
            pool.logState = nil;
        }

        void destroyIdleCommandEntries(const std::weak_ptr<MetalCommandPoolState>& weakPool) noexcept
        {
            auto pool = weakPool.lock();
            if (pool == nullptr)
            {
                return;
            }
            for (std::size_t index = 0;;)
            {
                std::unique_ptr<MetalCommandEntry> idleEntry;
                {
                    const std::scoped_lock poolLock(pool->mutex);
                    if (index == 0)
                    {
                        detachCommandEntryLeasesForShutdown(*pool);
                    }
                    idleEntry = takeNextIdleCommandEntry(*pool, index);
                    if (idleEntry == nullptr)
                    {
                        releaseEmptyPoolNativeObjects(*pool);
                        break;
                    }
                }
                idleEntry.reset();
            }
        }

        void detachDiagnosticSink(MetalDeviceNative& native) noexcept
        {
            if (native.sink == nullptr)
            {
                return;
            }
            native.sink->alive.store(false, std::memory_order_release);
            const std::scoped_lock lock(native.sink->mutex);
            native.sink->callback = nullptr;
            native.sink->userData = nullptr;
        }

    } // namespace

    MetalDevice::~MetalDevice()
    {
#ifdef RACHIS_TESTING
        unbindMetalDiagnosticTestDevice(this);
#endif
        if (native_ != nullptr)
        {
            @autoreleasepool
            {
                drainQueuesForTeardown(*native_);
                for (auto& timeline : native_->timelines)
                {
                    collectMetalQueue(timeline);
                }
                const auto pools = takeCommandPoolsForTeardown(*native_, mutex_);
                for (const auto& weakPool : pools)
                {
                    destroyIdleCommandEntries(weakPool);
                }
                destroyMipmapPipelines();
                detachDiagnosticSink(*native_);
            }
        }
    }

    const AdapterInfo& MetalDevice::adapterInfo() const noexcept
    {
        return adapter_;
    }

    const DeviceCapabilities& MetalDevice::capabilities() const noexcept
    {
        return capabilities_;
    }

    DeviceOperationalState MetalDevice::operationalState() const noexcept
    {
        if (native_ == nullptr)
        {
            return DeviceOperationalState::invalid;
        }
        if (native_->operationalState.load(std::memory_order_acquire) == DeviceOperationalState::lost)
        {
            return DeviceOperationalState::lost;
        }
        std::scoped_lock const lock(mutex_);
        return native_->device != nil ? DeviceOperationalState::active : DeviceOperationalState::invalid;
    }

    std::optional<Error> MetalDevice::deviceLossError() const
    {
        if (native_ == nullptr)
        {
            return std::nullopt;
        }
        const std::scoped_lock lock(native_->lossMutex);
        return native_->lossError != nullptr ? std::optional<Error>{*native_->lossError} : std::nullopt;
    }

    Status MetalDevice::requireActive() const
    {
        if (native_ == nullptr)
        {
            return {};
        }
        if (native_->operationalState.load(std::memory_order_acquire) != DeviceOperationalState::lost)
        {
            return {};
        }
        const std::scoped_lock lock(native_->lossMutex);
        return std::unexpected(native_->lossError != nullptr
                                   ? *native_->lossError
                                   : metalSelectionError(ErrorCode::deviceLost, "The Metal device is lost."));
    }

    void MetalDevice::publishLoss(const Error& error, DeviceFaultStatus status, bool injected) const
    {
        if (native_ == nullptr)
        {
            return;
        }
        publishMetalDeviceLoss(*native_, error, status, injected);
    }

#ifdef RACHIS_TESTING
    void MetalDevice::failNextCommitFeedback() noexcept
    {
        native_->failNextCommitFeedback.store(true, std::memory_order_release);
    }

    void MetalDevice::holdNextCommitFeedback(std::shared_ptr<MetalCommitFeedbackGate> gate) noexcept
    {
        const std::scoped_lock lock(native_->feedbackGateMutex);
        native_->nextCommitFeedbackGate = std::move(gate);
    }

    std::weak_ptr<void> MetalDevice::bufferOwnerObserverForTesting(BufferHandle buffer) const noexcept
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return {};
        }
        const auto* record = native_->buffers.find(buffer);
        return record != nullptr ? std::weak_ptr<void>(record->owner) : std::weak_ptr<void>{};
    }

    std::size_t MetalDevice::allocationTelemetryCountForTesting() const noexcept
    {
        std::scoped_lock const lock(mutex_);
        return native_ != nullptr ? native_->allocations.size() : 0;
    }

    Status MetalDevice::injectDeviceLoss(DeviceFaultStatus status)
    {
        const Error injected{
            ErrorCode::deviceLost,
            1,
            "Injected Metal device loss for RACHIS_TESTING.",
        };
        publishLoss(injected, status, true);
        return requireActive();
    }
#endif

    Result<std::vector<MemoryBudget>> MetalDevice::queryMemoryBudgets(std::uint32_t nodeIndex) const
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::scoped_lock const lock(mutex_);
        if (nodeIndex >= capabilities_.nodeCount || native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "Metal memory budgets require a valid device node."));
        }
        const auto totals = metalAllocationTotals(*native_);
        const auto physicalBytes = totals.bytes;
        const auto physicalCount = totals.count;
        MemoryBudget budget{};
        budget.heapIndex = 0;
        budget.properties = MemoryHeapProperty::deviceLocal;
        budget.features = MemoryBudgetFeature::allocationStatistics | MemoryBudgetFeature::driverReported;
        budget.budgetBytes = native_->device.recommendedMaxWorkingSetSize;
        budget.usageBytes = native_->device.currentAllocatedSize;
        budget.blockBytes = physicalBytes;
        budget.allocationBytes = physicalBytes;
        budget.blockCount = physicalCount;
        budget.allocationCount = physicalCount;
        return std::vector<MemoryBudget>{budget};
    }

    Result<ResourceMemorySnapshot> MetalDevice::captureResourceMemorySnapshot() const
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return ResourceMemorySnapshot{};
        }
        return captureMetalAllocationSnapshot(*native_);
    }

    Result<DeviceFaultReport> MetalDevice::captureDeviceFaultReport() const
    {
        if (native_ != nullptr)
        {
            const std::scoped_lock lock(native_->lossMutex);
            if (native_->lossReport != nullptr)
            {
                return *native_->lossReport;
            }
        }
        DeviceFaultReport report{};
        report.status = DeviceFaultStatus::healthy;
        report.dredAvailable = false;
        return report;
    }

    BackendAllocationSnapshot MetalDevice::captureBackendAllocationSnapshot() const noexcept
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return {};
        }
        const auto live = native_->views.size() + native_->samplers.size();
        return BackendAllocationSnapshot{
            .liveDescriptorCount = live,
            .reservedDescriptorCount = live,
        };
    }

    FormatProperties MetalDevice::queryFormatProperties(Format format) const noexcept
    {
        const auto index = static_cast<std::size_t>(format);
        if (index >= formats_.size())
        {
            return {};
        }
        return formats_[index];
    }

    QueueProperties MetalDevice::queueProperties(QueueType queueType) const noexcept
    {
        static_cast<void>(queueType);
        return graphicsQueue_;
    }

    DiagnosticCounts MetalDevice::diagnosticCounts() const noexcept
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->sink == nullptr)
        {
            return {};
        }
        return native_->sink->counts();
    }

    MetalValidationConfiguration MetalDevice::validationConfiguration() const
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return {};
        }
        return native_->validation;
    }

    BindingUpdateStatistics MetalDevice::bindingUpdateStatistics() const noexcept
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return {};
        }
        return native_->bindingStatistics;
    }

    Status MetalDevice::waitIdle(QueueType queueType)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (!isValidMetalQueueType(queueType))
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument, "The queue type is invalid."));
        }
        if (native_ == nullptr || native_->commandQueue == nil || native_->queueGate == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal waitIdle requires an active device."));
        }
        Status drained = {};
        {
            const std::scoped_lock gateLock(native_->queueGate->mutex);
            drained = drainMetalQueue(*native_, native_->timelines[metalQueueIndex(queueType)]);
        }
        if (!drained)
        {
            return drained;
        }
        if (queueType == QueueType::graphics)
        {
            std::scoped_lock const lock(mutex_);
            if (native_ != nullptr)
            {
                retireMetalPresents(*native_);
            }
        }
        return {};
    }

    Status MetalDevice::waitIdle()
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (native_ == nullptr || native_->commandQueue == nil || native_->queueGate == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal waitIdle requires an active device."));
        }
        {
            const std::scoped_lock gateLock(native_->queueGate->mutex);
            for (auto& timeline : native_->timelines)
            {
                auto drained = drainMetalQueue(*native_, timeline);
                if (!drained)
                {
                    return drained;
                }
            }
        }
        std::scoped_lock const lock(mutex_);
        if (native_ != nullptr)
        {
            retireMetalPresents(*native_);
        }
        return {};
    }

    std::uint64_t MetalDevice::completedValue(QueueType queueType) const noexcept
    {
        if (native_ == nullptr || !isValidMetalQueueType(queueType))
        {
            return 0;
        }
        const auto& timeline = native_->timelines[metalQueueIndex(queueType)];
        return timeline.feedback.completedValue(timeline.event != nil ? timeline.event.signaledValue : 0);
    }

    void MetalDevice::collectGarbage() noexcept
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return;
        }
        for (auto& timeline : native_->timelines)
        {
            collectMetalQueue(timeline);
        }
        retireMetalPresents(*native_);
        pruneMetalAllocations(*native_);
    }

    Status MetalDevice::emitIntentionalValidationFailure()
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "Intentional Metal validation failure requires an active device."));
        }

        NSError* error = nil;
        id<MTLLibrary> const library = [native_->device newLibraryWithSource:@"this is not a Metal shader"
                                                                     options:nil
                                                                       error:&error];
        static_cast<void>(library);
        auto mapped = metalNsError(ErrorCode::validationError, error, "intentional invalid newLibraryWithSource");
        if (native_->sink != nullptr)
        {
            native_->sink->publish(
                DiagnosticSeverity::error, static_cast<std::uint32_t>(mapped.nativeCode), mapped.message);
        }
        return std::unexpected(std::move(mapped));
    }

    namespace
    {

        [[nodiscard]] MetalCaptureRecord makeCaptureRecord(const MetalDeviceNative& native,
                                                           const MetalCaptureRequest& request)
        {
            MetalCaptureRecord record{};
            record.backend = "metal";
            const char* const deviceName = native.device.name.UTF8String;
            record.deviceName = deviceName != nullptr ? deviceName : "";
            record.registryId = native.device.registryID;
            record.commit = request.commit.empty() ? std::string("unspecified") : std::string(request.commit);
            record.workloadLabel = std::string(request.label);
            record.deviceLabel = native.validation.deviceLabel;
            record.queueLabel = native.validation.queueLabel;
            record.commandLabel = std::string(request.label);
            return record;
        }

        [[nodiscard]] std::filesystem::path prepareCaptureOutputPath(const MetalCaptureRequest& request,
                                                                     const MetalCaptureRecord& record)
        {
            std::filesystem::path const outputDir = request.outputDirectory.empty()
                                                        ? std::filesystem::temp_directory_path() / "rhi-metal-capture"
                                                        : std::filesystem::path(std::string(request.outputDirectory));
            std::error_code ioError;
            std::filesystem::create_directories(outputDir, ioError);
            const std::string fileName = "rhi-metal-" + std::to_string(record.registryId) + "-" +
                                         sanitizeCaptureToken(record.commit) + "-" +
                                         sanitizeCaptureToken(record.workloadLabel) + ".gputrace";
            return outputDir / fileName;
        }

        [[nodiscard]] id<MTLCaptureScope> createLabeledCaptureScope(MTLCaptureManager* manager,
                                                                    id<MTL4CommandQueue> commandQueue)
        {
            id<MTLCaptureScope> scope = nil;
            if (manager != nil)
            {
                scope = [manager newCaptureScopeWithMTL4CommandQueue:commandQueue];
                scope.label = @"RHI.Metal.CaptureScope";
            }
            return scope;
        }

        void startGpuTraceCapture(MTLCaptureManager* manager,
                                  id<MTLCaptureScope> scope,
                                  bool gpuTraceSupported,
                                  const MetalDeviceNative& native,
                                  MetalCaptureRecord& record)
        {
            if (!gpuTraceSupported || scope == nil)
            {
                record.toolingUnavailable = true;
                return;
            }
            MTLCaptureDescriptor* const descriptor = [MTLCaptureDescriptor new];
            descriptor.captureObject = scope;
            descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
            descriptor.outputURL = [NSURL fileURLWithPath:metalNsString(record.outputPath)];
            NSError* captureError = nil;
            if (![manager startCaptureWithDescriptor:descriptor error:&captureError])
            {
                record.toolingUnavailable = true;
                if (native.sink != nullptr && captureError != nil)
                {
                    const auto mapped =
                        metalNsError(ErrorCode::nativeFailure, captureError, "startCaptureWithDescriptor");
                    native.sink->publish(
                        DiagnosticSeverity::warning, static_cast<std::uint32_t>(mapped.nativeCode), mapped.message);
                }
            }
        }

        void abortGpuTraceCapture(MTLCaptureManager* manager, id<MTLCaptureScope> scope)
        {
            if (manager != nil && manager.isCapturing)
            {
                [manager stopCapture];
            }
            if (scope != nil)
            {
                [scope endScope];
            }
        }

        void encodeLabeledCaptureCommand(id<MTLLogState> logState,
                                         id<MTL4CommandAllocator> allocator,
                                         id<MTL4CommandBuffer> command,
                                         std::string_view label)
        {
            if (logState != nil)
            {
                MTL4CommandBufferOptions* const options = [MTL4CommandBufferOptions new];
                options.logState = logState;
                [command beginCommandBufferWithAllocator:allocator options:options];
            }
            else
            {
                [command beginCommandBufferWithAllocator:allocator];
            }
            [command pushDebugGroup:metalNsString(label)];
            [command popDebugGroup];
            [command endCommandBuffer];
        }

        [[nodiscard]] Result<TimelinePointStatus> submitCaptureCommandAndWait(
            const std::shared_ptr<MetalDeviceNative>& native,
            id<MTL4CommandAllocator> allocator,
            id<MTL4CommandBuffer> command,
            const MetalCaptureRecord& record)
        {
            MTL4CommitOptions* const commitOptions = [MTL4CommitOptions new];
            const auto backend = native;
            const auto encoderLabel = record.commandLabel;
            const auto queueLabel = record.queueLabel;
            const auto completion = std::make_shared<MetalQueueTimeline>();
            completion->event = [native->device newSharedEvent];
            completion->feedback.begin(1);
            [commitOptions addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
              static_cast<void>(allocator);
              static_cast<void>(command);
              publishMetalSubmissionFailure(*backend, feedback.error, encoderLabel, queueLabel);
              completion->feedback.complete(1);
            }];

            id<MTLSharedEvent> const event = completion->event;
            id<MTL4CommandBuffer> buffers[] = {command};
            if (native->queueGate != nullptr)
            {
                const std::scoped_lock gateLock(native->queueGate->mutex);
                [native->commandQueue commit:buffers count:1 options:commitOptions];
                if (event != nil)
                {
                    [native->commandQueue signalEvent:event value:1];
                }
            }
            else
            {
                [native->commandQueue commit:buffers count:1 options:commitOptions];
                if (event != nil)
                {
                    [native->commandQueue signalEvent:event value:1];
                }
            }
            return waitForMetalValue(*backend, *completion, 1, 5000);
        }

        void finishGpuTraceCapture(MTLCaptureManager* manager, id<MTLCaptureScope> scope)
        {
            if (scope != nil)
            {
                [scope endScope];
            }
            if (manager != nil && manager.isCapturing)
            {
                [manager stopCapture];
            }
        }

    } // namespace

    Result<MetalCaptureRecord> MetalDevice::captureLabeledHeadlessCommand(const MetalCaptureRequest& request)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil || native_->commandQueue == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal capture requires an active device."));
        }
        if (request.label.empty() || request.label.size() > maxDebugLabelBytes || request.label.contains('\0'))
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "Metal capture requires a portable label."));
        }

        @autoreleasepool
        {
            MetalCaptureRecord record = makeCaptureRecord(*native_, request);

            MTLCaptureManager* const manager = [MTLCaptureManager sharedCaptureManager];
            const bool gpuTraceSupported =
                manager != nil && [manager supportsDestination:MTLCaptureDestinationGPUTraceDocument];
            native_->validation.gpuTraceCaptureSupported = gpuTraceSupported;

            const auto outputPath = prepareCaptureOutputPath(request, record);
            record.outputPath = outputPath.string();

            id<MTLCaptureScope> const scope = createLabeledCaptureScope(manager, native_->commandQueue);
            startGpuTraceCapture(manager, scope, gpuTraceSupported, *native_, record);
            if (scope != nil)
            {
                [scope beginScope];
            }

            NSError* allocatorError = nil;
            MTL4CommandAllocatorDescriptor* const allocatorDesc = [MTL4CommandAllocatorDescriptor new];
            allocatorDesc.label = @"RHI.Metal.CaptureAllocator";
            id<MTL4CommandAllocator> const allocator =
                [native_->device newCommandAllocatorWithDescriptor:allocatorDesc error:&allocatorError];
            static_cast<void>(allocatorError);
            id<MTL4CommandBuffer> const command = [native_->device newCommandBuffer];
            if (allocator == nil || command == nil)
            {
                abortGpuTraceCapture(manager, scope);
                return std::unexpected(metalSelectionError(
                    ErrorCode::nativeFailure,
                    "Metal capture could not create a labeled command buffer. Partial Metal objects were released."));
            }
            command.label = metalNsString(request.label);

            encodeLabeledCaptureCommand(native_->logState, allocator, command, request.label);
            auto waited = submitCaptureCommandAndWait(native_, allocator, command, record);
            finishGpuTraceCapture(manager, scope);

            if (!waited)
            {
                return std::unexpected(std::move(waited.error()));
            }
            if (*waited != TimelinePointStatus::complete)
            {
                return std::unexpected(metalSelectionError(ErrorCode::nativeFailure,
                                                           "Metal capture timed out waiting for commit feedback."));
            }

            record.captured = !record.toolingUnavailable && std::filesystem::exists(outputPath);
            if (!record.captured && !record.toolingUnavailable)
            {
                record.toolingUnavailable = !std::filesystem::exists(outputPath);
            }
            return record;
        }
    }

} // namespace rachis::metal

#ifdef RACHIS_TESTING
void rachis::metal::setHostReportOverride(std::optional<rachis::MetalHostReport> report) noexcept
{
    std::scoped_lock const lock(metalTestingState().mutex);
    metalTestingState().hostReportOverride = report;
}

void rachis::metal::setFailCommandApiCreateAttempts(std::uint32_t attempts, std::string_view nativeDescription)
{
    std::scoped_lock const lock(metalTestingState().mutex);
    metalTestingState().failCommandApiCreateAttempts = attempts;
    metalTestingState().failCommandApiNativeDescription =
        attempts == 0 ? std::string{} : std::string{nativeDescription};
}

void rachis::metal::setFailNextLogStateCreate(bool fail) noexcept
{
    std::scoped_lock const lock(metalTestingState().mutex);
    metalTestingState().failNextLogStateCreate = fail;
}

void rachis::metal::setFailNextDepthStencilStateCreate(bool fail) noexcept
{
    std::scoped_lock const lock(metalTestingState().mutex);
    metalTestingState().failNextDepthStencilStateCreate = fail;
}

void rachis::metal::setFailNextDrawable(bool fail) noexcept
{
    std::scoped_lock const lock(metalTestingState().mutex);
    metalTestingState().failNextDrawable = fail;
}

void rachis::metal::setFailNextResidencySetCreate(bool fail) noexcept
{
    std::scoped_lock const lock(metalTestingState().mutex);
    metalTestingState().failNextResidencySetCreate = fail;
}

void rachis::metal::setFailNextPipelineCacheSerialize(bool fail) noexcept
{
    std::scoped_lock const lock(metalTestingState().mutex);
    metalTestingState().failNextPipelineCacheSerialize = fail;
}

void rachis::metal::setPipelineCacheTemporaryDirectory(std::string_view directory)
{
    std::scoped_lock const lock(metalTestingState().mutex);
    metalTestingState().pipelineCacheTemporaryDirectory = std::string(directory);
}

rachis::Status rachis::metal::emitIntentionalValidationFailure()
{
    rachis::metal::MetalDevice* device = nullptr;
    {
        std::scoped_lock const lock(metalTestingState().mutex);
        device = metalTestingState().boundDevice;
    }
    if (device == nullptr)
    {
        return std::unexpected(Error{
            ErrorCode::invalidState,
            0,
            "No live Metal test device is bound for intentional validation failure.",
        });
    }
    return device->emitIntentionalValidationFailure();
}

rachis::Result<rachis::MetalCaptureRecord> rachis::metal::captureLabeledHeadlessCommand(
    const rachis::MetalCaptureRequest& request)
{
    rachis::metal::MetalDevice* device = nullptr;
    {
        std::scoped_lock const lock(metalTestingState().mutex);
        device = metalTestingState().boundDevice;
    }
    if (device == nullptr)
    {
        return std::unexpected(Error{
            ErrorCode::invalidState,
            0,
            "No live Metal test device is bound for capture.",
        });
    }
    return device->captureLabeledHeadlessCommand(request);
}

rachis::MetalValidationConfiguration rachis::metal::boundValidationConfiguration()
{
    rachis::metal::MetalDevice const* device = nullptr;
    {
        std::scoped_lock const lock(metalTestingState().mutex);
        device = metalTestingState().boundDevice;
    }
    if (device == nullptr)
    {
        return {};
    }
    return device->validationConfiguration();
}

rachis::Status rachis::metal::copyBuffersImmediate(rachis::BufferHandle source,
                                                   rachis::BufferHandle destination,
                                                   std::uint64_t sourceOffset,
                                                   std::uint64_t destinationOffset,
                                                   std::uint64_t size)
{
    rachis::metal::MetalDevice* device = nullptr;
    {
        std::scoped_lock const lock(metalTestingState().mutex);
        device = metalTestingState().boundDevice;
    }
    if (device == nullptr)
    {
        return std::unexpected(Error{
            ErrorCode::invalidState,
            0,
            "No live Metal test device is bound for buffer copy.",
        });
    }
    return device->copyBuffersImmediate(source, destination, sourceOffset, destinationOffset, size);
}

rachis::Status rachis::metal::copyCapturedBindingBuffer(rachis::CommandBuffer& commandBuffer,
                                                        std::uint32_t setIndex,
                                                        rachis::BufferHandle destination,
                                                        std::uint64_t size)
{
    rachis::metal::MetalDevice* device = nullptr;
    {
        std::scoped_lock const lock(metalTestingState().mutex);
        device = metalTestingState().boundDevice;
    }
    if (device == nullptr)
    {
        return std::unexpected(Error{
            ErrorCode::invalidState,
            0,
            "No live Metal test device is bound for captured binding copy.",
        });
    }
    auto* backend = rachis::detail::CommandBufferAccess::get(commandBuffer);
    if (backend == nullptr)
    {
        return std::unexpected(Error{
            ErrorCode::invalidArgument,
            0,
            "The command buffer is not a live Metal recording lease.",
        });
    }
    return device->copyCapturedBindingBuffer(*backend, setIndex, destination, size);
}

rachis::Status rachis::metal::loadCookedMetallib(std::span<const std::byte> bytecode, std::string_view entryPoint)
{
    if (bytecode.empty())
    {
        return std::unexpected(Error{
            ErrorCode::invalidArgument,
            0,
            "Cooked metallib bytecode is empty.",
        });
    }

    @autoreleasepool
    {
        id<MTLDevice> const device = MTLCreateSystemDefaultDevice();
        if (device == nil)
        {
            return std::unexpected(Error{
                ErrorCode::deviceCreationFailed,
                0,
                "No system Metal device is available to load a cooked metallib.",
            });
        }

        NSData* const data = [NSData dataWithBytes:bytecode.data() length:bytecode.size()];
        const dispatch_data_t dispatchData =
            dispatch_data_create(data.bytes, data.length, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
              (void)data;
            });
        NSError* error = nil;
        id<MTLLibrary> const library = [device newLibraryWithData:dispatchData error:&error];
        if (library == nil)
        {
            const char* message = error != nil ? error.localizedDescription.UTF8String : nullptr;
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                error != nil ? static_cast<std::int64_t>(error.code) : 0,
                message != nullptr ? message : "Metal rejected the cooked metallib.",
            });
        }
        if (entryPoint.empty())
        {
            return {};
        }

        NSString* const name = [[NSString alloc] initWithBytes:entryPoint.data()
                                                        length:entryPoint.size()
                                                      encoding:NSUTF8StringEncoding];
        if (name == nil || [library newFunctionWithName:name] == nil)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Cooked metallib is missing the requested entry point.",
            });
        }
        return {};
    }
}

namespace
{

    [[nodiscard]] bool hasShaderManifestMagic(std::span<const std::byte> bytecode)
    {
        return bytecode.size() >= rachis::metal::shaderManifestHeaderBytes && bytecode[0] == std::byte{'P'} &&
               bytecode[1] == std::byte{'N'} && bytecode[2] == std::byte{'M'} && bytecode[3] == std::byte{'S'};
    }

    [[nodiscard]] rachis::Error layoutFixtureNativeError(NSError* error, const char* fallback)
    {
        const char* message = error != nil ? error.localizedDescription.UTF8String : nullptr;
        return rachis::Error{
            rachis::ErrorCode::invalidArgument,
            error != nil ? static_cast<std::int64_t>(error.code) : 0,
            message != nullptr ? message : fallback,
        };
    }

    [[nodiscard]] rachis::Result<id<MTLLibrary>> loadLayoutFixtureLibrary(id<MTLDevice> device,
                                                                          std::span<const std::byte> metallib)
    {
        NSData* const data = [NSData dataWithBytes:metallib.data() length:metallib.size()];
        const dispatch_data_t dispatchData =
            dispatch_data_create(data.bytes, data.length, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
              (void)data;
            });
        NSError* error = nil;
        id<MTLLibrary> const library = [device newLibraryWithData:dispatchData error:&error];
        if (library == nil)
        {
            return std::unexpected(layoutFixtureNativeError(error, "Metal rejected the LayoutFixture metallib."));
        }
        return library;
    }

    [[nodiscard]] rachis::Result<id<MTLComputePipelineState>> createLayoutFixturePipeline(
        id<MTLDevice> device, std::span<const std::byte> metallib)
    {
        auto library = loadLayoutFixtureLibrary(device, metallib);
        if (!library)
        {
            return std::unexpected(std::move(library.error()));
        }
        id<MTLFunction> const function = [*library newFunctionWithName:@"computeMain"];
        if (function == nil)
        {
            return std::unexpected(rachis::Error{
                rachis::ErrorCode::invalidArgument,
                0,
                "LayoutFixture metallib is missing computeMain.",
            });
        }

        NSError* error = nil;
        id<MTLComputePipelineState> const pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil)
        {
            return std::unexpected(
                layoutFixtureNativeError(error, "Metal rejected the LayoutFixture compute pipeline."));
        }
        return pipeline;
    }

    [[nodiscard]] std::vector<std::uint8_t> makeLayoutFixtureConstants()
    {
        std::vector<std::uint8_t> constants(rachis::metalConstantBufferSize, 0);
        const float nestedA = 1.0F;
        const float nestedB[3] = {2.0F, 3.0F, 4.0F};
        const float matrixCol0[3] = {5.0F, 6.0F, 7.0F};
        const float arrayValues[2] = {11.0F, 12.0F};
        const std::uint32_t marker = 0xAABBCCDDU;
        std::memcpy(constants.data() + rachis::metalNestedScalarOffset, &nestedA, sizeof(nestedA));
        std::memcpy(constants.data() + rachis::metalNestedFloat3Offset, nestedB, sizeof(nestedB));
        std::memcpy(constants.data() + rachis::metalNonSquareMatrixOffset, matrixCol0, sizeof(matrixCol0));
        std::memcpy(constants.data() + rachis::metalArrayOffset, arrayValues, sizeof(arrayValues));
        std::memcpy(constants.data() + rachis::metalMarkerOffset, &marker, sizeof(marker));
        return constants;
    }

    [[nodiscard]] rachis::Status dispatchLayoutFixture(id<MTLDevice> device,
                                                       id<MTLComputePipelineState> pipeline,
                                                       id<MTLBuffer> constantBuffer,
                                                       id<MTLBuffer> outputBuffer)
    {
        id<MTLCommandQueue> const queue = [device newCommandQueue];
        id<MTLCommandBuffer> const commandBuffer = queue != nil ? [queue commandBuffer] : nil;
        id<MTLComputeCommandEncoder> const encoder = commandBuffer != nil ? [commandBuffer computeCommandEncoder] : nil;
        if (queue == nil || commandBuffer == nil || encoder == nil)
        {
            return std::unexpected(rachis::Error{
                rachis::ErrorCode::invalidState,
                0,
                "LayoutFixture classic compute command objects were missing.",
            });
        }
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:constantBuffer offset:0 atIndex:rachis::metalConstantBufferIndex];
        [encoder setBuffer:outputBuffer offset:0 atIndex:rachis::metalStorageBufferIndex];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.error != nil)
        {
            const char* message = commandBuffer.error.localizedDescription.UTF8String;
            return std::unexpected(rachis::Error{
                rachis::ErrorCode::invalidState,
                static_cast<std::int64_t>(commandBuffer.error.code),
                message != nullptr ? message : "LayoutFixture GPU dispatch failed.",
            });
        }
        return {};
    }

} // namespace

rachis::Result<std::array<std::uint32_t, 8>> rachis::metal::readLayoutFixtureSentinels(
    std::span<const std::byte> bytecode)
{
    auto metallib = bytecode;
    std::optional<MetalShaderPayload> payload;
    if (hasShaderManifestMagic(bytecode))
    {
        auto unpacked = unpackShaderManifest(bytecode);
        if (!unpacked)
        {
            return std::unexpected(std::move(unpacked.error()));
        }
        payload = *unpacked;
        metallib = payload->metallib;
    }
    if (metallib.size() < 4)
    {
        return std::unexpected(Error{
            ErrorCode::invalidArgument,
            0,
            "LayoutFixture metallib bytecode is empty.",
        });
    }

    @autoreleasepool
    {
        id<MTLDevice> const device = MTLCreateSystemDefaultDevice();
        if (device == nil)
        {
            return std::unexpected(Error{
                ErrorCode::deviceCreationFailed,
                0,
                "No system Metal device is available for LayoutFixture readback.",
            });
        }

        auto pipeline = createLayoutFixturePipeline(device, metallib);
        if (!pipeline)
        {
            return std::unexpected(std::move(pipeline.error()));
        }

        std::vector<std::uint8_t> constants = makeLayoutFixtureConstants();
        id<MTLBuffer> const constantBuffer = [device newBufferWithBytes:constants.data()
                                                                 length:constants.size()
                                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> const outputBuffer = [device newBufferWithLength:8 * sizeof(std::uint32_t)
                                                               options:MTLResourceStorageModeShared];
        if (constantBuffer == nil || outputBuffer == nil)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "LayoutFixture readback buffers could not be allocated.",
            });
        }
        std::memset(outputBuffer.contents, 0, 8 * sizeof(std::uint32_t));

        if (auto dispatched = dispatchLayoutFixture(device, *pipeline, constantBuffer, outputBuffer); !dispatched)
        {
            return std::unexpected(std::move(dispatched.error()));
        }

        std::array<std::uint32_t, 8> words{};
        std::memcpy(words.data(), outputBuffer.contents, words.size() * sizeof(std::uint32_t));
        return words;
    }
}

rachis::Status rachis::metal::markCommandBufferInFlight(rachis::CommandBuffer& commandBuffer, bool inFlight)
{
    auto* backend = rachis::detail::CommandBufferAccess::get(commandBuffer);
    auto* native = backend != nullptr && backend->kind() == rachis::detail::BackendCommandBufferKind::metal
                       ? static_cast<rachis::metal::MetalCommandBuffer*>(backend)
                       : nullptr;
    if (native == nullptr || native->entry() == nullptr)
    {
        return std::unexpected(Error{
            ErrorCode::invalidArgument,
            0,
            "The command buffer is not a live Metal recording lease.",
        });
    }
    native->setInFlightForTesting(inFlight);
    return {};
}

namespace
{

    [[nodiscard]] rachis::metal::MetalDevice* boundMetalTestDevice() noexcept
    {
        std::scoped_lock const lock(rachis::metal::metalTestingState().mutex);
        return rachis::metal::metalTestingState().boundDevice;
    }

    [[nodiscard]] rachis::Error unboundMetalTestDeviceError(std::string_view operation)
    {
        return rachis::Error{
            rachis::ErrorCode::invalidState,
            0,
            std::string(operation) + " requires a live Metal test device.",
        };
    }

} // namespace

void rachis::metal::failNextCommitFeedback()
{
    if (auto* device = boundMetalTestDevice())
    {
        device->failNextCommitFeedback();
    }
}

void rachis::metal::holdNextCommitFeedback(std::shared_ptr<rachis::metal::MetalCommitFeedbackGate> gate)
{
    if (auto* device = boundMetalTestDevice())
    {
        device->holdNextCommitFeedback(std::move(gate));
    }
}

void rachis::metal::holdNextCommandPoolReset(std::shared_ptr<rachis::metal::MetalCommitFeedbackGate> gate)
{
    const std::scoped_lock lock(metalTestingState().mutex);
    metalTestingState().nextCommandPoolResetGate = std::move(gate);
}

void rachis::metal::holdNextDrawableAcquire(std::shared_ptr<rachis::metal::MetalCommitFeedbackGate> gate)
{
    const std::scoped_lock lock(metalTestingState().mutex);
    metalTestingState().nextDrawableAcquireGate = std::move(gate);
}

std::weak_ptr<void> rachis::metal::boundBufferOwnerObserver(rachis::BufferHandle buffer)
{
    const auto* device = boundMetalTestDevice();
    return device != nullptr ? device->bufferOwnerObserverForTesting(buffer) : std::weak_ptr<void>{};
}

rachis::Result<std::size_t> rachis::metal::boundAllocationTelemetryCount()
{
    const auto* device = boundMetalTestDevice();
    if (device == nullptr)
    {
        return std::unexpected(unboundMetalTestDeviceError("boundAllocationTelemetryCount"));
    }
    return device->allocationTelemetryCountForTesting();
}

rachis::Status rachis::metal::injectMetalDeviceLoss(rachis::DeviceFaultStatus status)
{
    auto* device = boundMetalTestDevice();
    if (device == nullptr)
    {
        return std::unexpected(unboundMetalTestDeviceError("Injected device loss"));
    }
    return device->injectDeviceLoss(status);
}

rachis::Status rachis::metal::notifySurfaceHostClose(rachis::SurfaceHandle surface)
{
    auto* device = boundMetalTestDevice();
    if (device == nullptr)
    {
        return std::unexpected(unboundMetalTestDeviceError("notifySurfaceHostClose"));
    }
    return device->notifySurfaceHostCloseForTesting(surface);
}

void* rachis::metal::boundNativeMetalDeviceHandle() noexcept
{
    auto* device = boundMetalTestDevice();
    return device != nullptr ? device->nativeMetalDeviceHandle() : nullptr;
}

std::uint64_t rachis::metal::boundSamplerGpuResourceId(rachis::SamplerHandle sampler) noexcept
{
    const auto* device = boundMetalTestDevice();
    return device != nullptr ? device->samplerGpuResourceIdForTesting(sampler) : 0;
}

void rachis::metal::resetRecordingTelemetry() noexcept
{
    auto& state = metalTestingState();
    state.residencyAddCount.store(0, std::memory_order_relaxed);
    state.residencyDeduplicatedSkipCount.store(0, std::memory_order_relaxed);
    state.residencyCommitCount.store(0, std::memory_order_relaxed);
    state.argumentTableCreateCount.store(0, std::memory_order_relaxed);
    state.argumentTablePoolReuseCount.store(0, std::memory_order_relaxed);
    state.argumentTableEncoderReuseCount.store(0, std::memory_order_relaxed);
    state.pushConstantPageAllocateCount.store(0, std::memory_order_relaxed);
    state.pushConstantSliceSuballocateCount.store(0, std::memory_order_relaxed);
    state.renderTimestampVertexStageCount.store(0, std::memory_order_relaxed);
    state.renderTimestampFragmentStageCount.store(0, std::memory_order_relaxed);
}

rachis::metal::MetalRecordingTelemetry rachis::metal::recordingTelemetry() noexcept
{
    const auto& state = metalTestingState();
    return MetalRecordingTelemetry{
        .residencyAddCount = state.residencyAddCount.load(std::memory_order_relaxed),
        .residencyDeduplicatedSkipCount = state.residencyDeduplicatedSkipCount.load(std::memory_order_relaxed),
        .residencyCommitCount = state.residencyCommitCount.load(std::memory_order_relaxed),
        .argumentTableCreateCount = state.argumentTableCreateCount.load(std::memory_order_relaxed),
        .argumentTablePoolReuseCount = state.argumentTablePoolReuseCount.load(std::memory_order_relaxed),
        .argumentTableEncoderReuseCount = state.argumentTableEncoderReuseCount.load(std::memory_order_relaxed),
        .pushConstantPageAllocateCount = state.pushConstantPageAllocateCount.load(std::memory_order_relaxed),
        .pushConstantSuballocateCount = state.pushConstantSliceSuballocateCount.load(std::memory_order_relaxed),
        .renderTimestampVertexStageCount = state.renderTimestampVertexStageCount.load(std::memory_order_relaxed),
        .renderTimestampFragmentStageCount = state.renderTimestampFragmentStageCount.load(std::memory_order_relaxed),
    };
}

std::size_t rachis::metal::autoreleasedObjectCount(const std::function<void()>& body)
{
    constexpr std::uintptr_t poolBlockBytes = 4096;
    constexpr std::uintptr_t poolHeadroomBytes = 1024;
    void* const outer = objc_autoreleasePoolPush();
    auto before = reinterpret_cast<std::uintptr_t>(objc_autoreleasePoolPush());
    while (before % poolBlockBytes > poolBlockBytes - poolHeadroomBytes)
    {
        before = reinterpret_cast<std::uintptr_t>(objc_autoreleasePoolPush());
    }
    body();
    const auto after = reinterpret_cast<std::uintptr_t>(objc_autoreleasePoolPush());
    objc_autoreleasePoolPop(outer);
    if (after <= before || after / poolBlockBytes != before / poolBlockBytes)
    {
        return std::numeric_limits<std::size_t>::max();
    }
    return (after - before) / sizeof(void*) - 1U;
}

void rachis::metal::resetTeardownQueueDrainCount() noexcept
{
    metalTestingState().teardownQueueDrainCount.store(0, std::memory_order_relaxed);
}

std::uint64_t rachis::metal::teardownQueueDrainCount() noexcept
{
    return metalTestingState().teardownQueueDrainCount.load(std::memory_order_relaxed);
}

rachis::DeviceFaultStatus rachis::metal::classifyMetalFaultStatusForTesting(MetalTestErrorDomain domain,
                                                                            std::int64_t code) noexcept
{
    @autoreleasepool
    {
        NSString* nsDomain = nil;
        switch (domain)
        {
        case MetalTestErrorDomain::commandQueue4:
            nsDomain = MTL4CommandQueueErrorDomain;
            break;
        case MetalTestErrorDomain::commandBuffer:
            nsDomain = MTLCommandBufferErrorDomain;
            break;
        case MetalTestErrorDomain::other:
            nsDomain = @"RHI.UnrelatedErrorDomain";
            break;
        }
        NSError* const error = [NSError errorWithDomain:nsDomain code:static_cast<NSInteger>(code) userInfo:nil];
        return metalFaultStatusFromError(error);
    }
}
#endif
