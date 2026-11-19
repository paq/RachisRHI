#include "Metal/MetalInternal.h"

#include "Internal/DeviceIdentity.h"

namespace rachis::metal
{
    namespace
    {

        [[nodiscard]] rachis::SampleCountMask sampleCountsForDevice(id<MTLDevice> device) noexcept
        {
            rachis::SampleCountMask mask = rachis::SampleCountMask::none;
            if ([device supportsTextureSampleCount:1])
            {
                mask |= rachis::SampleCountMask::x1;
            }
            if ([device supportsTextureSampleCount:2])
            {
                mask |= rachis::SampleCountMask::x2;
            }
            if ([device supportsTextureSampleCount:4])
            {
                mask |= rachis::SampleCountMask::x4;
            }
            if ([device supportsTextureSampleCount:8])
            {
                mask |= rachis::SampleCountMask::x8;
            }
            return mask;
        }

        [[nodiscard]] rachis::MetalReadWriteTextureTier readWriteTierForDevice(id<MTLDevice> device) noexcept
        {
            switch (device.readWriteTextureSupport)
            {
            case MTLReadWriteTextureTier1:
                return rachis::MetalReadWriteTextureTier::tier1;
            case MTLReadWriteTextureTier2:
                return rachis::MetalReadWriteTextureTier::tier2;
            case MTLReadWriteTextureTierNone:
                break;
            }
            return rachis::MetalReadWriteTextureTier::none;
        }

        [[nodiscard]] rachis::FormatFeature storageFeaturesForDevice(id<MTLDevice> device,
                                                                     rachis::Format format) noexcept
        {
            if (!rachis::metalReadWriteFormat(format, readWriteTierForDevice(device)))
            {
                return rachis::FormatFeature::none;
            }
            return rachis::FormatFeature::storageRead | rachis::FormatFeature::storageWrite;
        }

    } // namespace

    [[nodiscard]] rachis::FormatProperties probeFormat(id<MTLDevice> device, rachis::Format format) noexcept
    {
        rachis::FormatProperties properties{};
        if (device == nil || metalPixelFormat(format) == MTLPixelFormatInvalid)
        {
            return properties;
        }
        if (format == rachis::Format::d24UnormS8Uint && !device.isDepth24Stencil8PixelFormatSupported)
        {
            return properties;
        }
        if (formatIsBc(format))
        {
            if (!device.supportsBCTextureCompression)
            {
                return properties;
            }
            properties.features = rachis::FormatFeature::texture2D | rachis::FormatFeature::cube |
                                  rachis::FormatFeature::sampled | rachis::FormatFeature::filterable |
                                  rachis::FormatFeature::copySource | rachis::FormatFeature::copyDestination;
            if (formatAllowsViewReinterpretation(format))
            {
                properties.features |= rachis::FormatFeature::viewFormatReinterpretation;
            }
            properties.sampleCounts = rachis::SampleCountMask::x1;
            return properties;
        }

        const auto deviceSamples = sampleCountsForDevice(device);
        const auto nonEmptySamples =
            deviceSamples == rachis::SampleCountMask::none ? rachis::SampleCountMask::x1 : deviceSamples;
        const auto supportsMultisample = [](rachis::SampleCountMask mask) noexcept {
            return rachis::supportsSampleCount(mask, 2) || rachis::supportsSampleCount(mask, 4) ||
                   rachis::supportsSampleCount(mask, 8);
        };

        if (formatIsDepth(format))
        {
            properties.features = rachis::FormatFeature::texture2D | rachis::FormatFeature::sampled |
                                  rachis::FormatFeature::depthStencilAttachment | rachis::FormatFeature::copySource |
                                  rachis::FormatFeature::copyDestination;
            properties.sampleCounts = nonEmptySamples;
            if (supportsMultisample(properties.sampleCounts))
            {
                properties.features |= rachis::FormatFeature::multisampleResolve;
                properties.depthResolveModes = rachis::ResolveModeMask::minimum | rachis::ResolveModeMask::maximum;
            }
            return properties;
        }

        if (formatIsInteger(format))
        {
            const bool int32 = format == rachis::Format::r32Uint || format == rachis::Format::rg32Uint ||
                               format == rachis::Format::rgba32Uint || format == rachis::Format::r32Sint ||
                               format == rachis::Format::rg32Sint || format == rachis::Format::rgba32Sint;
            properties.features = rachis::FormatFeature::texture1D | rachis::FormatFeature::texture2D |
                                  rachis::FormatFeature::texture3D | rachis::FormatFeature::cube |
                                  rachis::FormatFeature::sampled | storageFeaturesForDevice(device, format) |
                                  rachis::FormatFeature::colorAttachment | rachis::FormatFeature::copySource |
                                  rachis::FormatFeature::copyDestination;
            properties.sampleCounts =
                (!int32 || device.supports32BitMSAA) ? nonEmptySamples : rachis::SampleCountMask::x1;
            return properties;
        }

        const bool float32 = format == rachis::Format::r32Float || format == rachis::Format::rg32Float ||
                             format == rachis::Format::rgba32Float;
        properties.features = rachis::FormatFeature::texture1D | rachis::FormatFeature::texture2D |
                              rachis::FormatFeature::texture3D | rachis::FormatFeature::cube |
                              rachis::FormatFeature::sampled | storageFeaturesForDevice(device, format) |
                              rachis::FormatFeature::colorAttachment | rachis::FormatFeature::blendable |
                              rachis::FormatFeature::copySource | rachis::FormatFeature::copyDestination;
        if (!float32 || device.supports32BitFloatFiltering)
        {
            properties.features |= rachis::FormatFeature::filterable;
        }
        if (formatAllowsViewReinterpretation(format))
        {
            properties.features |= rachis::FormatFeature::viewFormatReinterpretation;
        }
        properties.sampleCounts = nonEmptySamples;
        const bool canResolve = (!float32 || device.supports32BitMSAA) &&
                                (format != rachis::Format::rgba32Float || [device supportsFamily:MTLGPUFamilyApple9]);
        if (canResolve && supportsMultisample(properties.sampleCounts))
        {
            properties.features |= rachis::FormatFeature::multisampleResolve;
            properties.colorResolveModes = rachis::ResolveModeMask::average;
        }
        return properties;
    }

    namespace
    {

        [[nodiscard]] std::uint32_t maxTexture2DForDevice(id<MTLDevice> device) noexcept
        {
            if ([device supportsFamily:MTLGPUFamilyMetal4] || [device supportsFamily:MTLGPUFamilyMetal3] ||
                [device supportsFamily:MTLGPUFamilyApple7] || [device supportsFamily:MTLGPUFamilyApple8] ||
                [device supportsFamily:MTLGPUFamilyApple9])
            {
                return 16384;
            }
            return 8192;
        }

        [[nodiscard]] rachis::MetalHostReport probeHost(id<MTLDevice> device)
        {
            rachis::MetalHostReport host{};
            const auto version = NSProcessInfo.processInfo.operatingSystemVersion;
            host.macosMajor = static_cast<std::uint16_t>(version.majorVersion);
            host.macosMinor = static_cast<std::uint16_t>(version.minorVersion);
#ifdef __arm64__
            host.appleSiliconArm64 = true;
            host.intelMac = false;
#else
            host.appleSiliconArm64 = false;
            host.intelMac = true;
#endif
            host.hasDevice = device != nil;
            if (device == nil)
            {
                return host;
            }
            const char* const deviceName = device.name.UTF8String;
            const std::string_view name = deviceName != nullptr ? deviceName : "";
            host.translationLayer = name.contains("Dozen") || name.contains("MoltenVK");
            host.metal3Family = [device supportsFamily:MTLGPUFamilyMetal3];
            host.metal4Family = [device supportsFamily:MTLGPUFamilyMetal4];
            host.classicCommandQueue = [device newCommandQueue] != nil;
            id<MTL4CommandQueue> const queue = [device newMTL4CommandQueue];
            host.metal4CommandQueue = queue != nil;
            host.metal4CommandAllocator = [device newCommandAllocator] != nil;
            host.metal4CommandBuffer = [device newCommandBuffer] != nil;
            const auto rgba = probeFormat(device, rachis::Format::rgba8Unorm);
            const auto bgra = probeFormat(device, rachis::Format::bgra8Unorm);
            const auto depth = probeFormat(device, rachis::Format::d32Float);
            host.formatRgba8Unorm = rachis::hasAny(rgba.features, rachis::FormatFeature::colorAttachment);
            host.formatBgra8Unorm = rachis::hasAny(bgra.features, rachis::FormatFeature::colorAttachment);
            host.formatD32Float = rachis::hasAny(depth.features, rachis::FormatFeature::depthStencilAttachment);
            host.maxBufferLength = device.maxBufferLength;
            host.recommendedMaxWorkingSetSize = device.recommendedMaxWorkingSetSize;
            host.maxTexture2D = maxTexture2DForDevice(device);
            return host;
        }

        [[nodiscard]] rachis::DeviceCapabilities publishMetalCapabilities(id<MTLDevice> device,
                                                                          const rachis::MetalHostReport& host) noexcept
        {
            rachis::DeviceCapabilities capabilities{};
            capabilities.coordinateConventions = {
                .ndcYAxis = rachis::NdcYAxisDirection::up,
                .ndcDepthRange = rachis::NdcDepthRange::zeroToOne,
                .textureOrigin = rachis::TextureOrigin::upperLeft,
            };
            capabilities.nodeCount = 1;
            capabilities.unifiedMemory = device.hasUnifiedMemory;
            capabilities.cacheCoherentUnifiedMemory = device.hasUnifiedMemory;
            capabilities.memoryBudgets = true;
            capabilities.limits.maxTextureDimension1D = host.maxTexture2D;
            capabilities.limits.maxTextureDimension2D = host.maxTexture2D;
            capabilities.limits.maxTextureDimension3D = 2048;
            capabilities.limits.maxTextureArrayLayers = 2048;
            capabilities.limits.maxFramebufferLayers = 2048;
            capabilities.limits.maxTextureMipLevels =
                host.maxTexture2D == 0 ? 0 : static_cast<std::uint32_t>(std::bit_width(host.maxTexture2D));
            const MTLSize threadgroup = device.maxThreadsPerThreadgroup;
            capabilities.limits.maxComputeWorkgroupSize = {
                static_cast<std::uint32_t>(threadgroup.width),
                static_cast<std::uint32_t>(threadgroup.height),
                static_cast<std::uint32_t>(threadgroup.depth),
            };
            capabilities.limits.maxComputeWorkgroupInvocations = static_cast<std::uint32_t>(threadgroup.width);
            capabilities.limits.maxComputeWorkgroupCount = {65535, 65535, 65535};
            capabilities.limits.maxBufferSize = host.maxBufferLength;
            capabilities.limits.maxUniformBufferSize = std::min(host.maxBufferLength, 64ULL * 1024ULL);
            capabilities.limits.minUniformBufferOffsetAlignment = 16;
            capabilities.limits.maxSamplerAnisotropy = 16;
            capabilities.limits.maxSamplerLodBias = 15.999F;
            capabilities.limits.maxColorAttachments = rachis::maxColorAttachments;
            capabilities.limits.maxViewports = 1;
            capabilities.limits.maxVertexBuffers = rachis::maxVertexBuffers;
            capabilities.limits.maxVertexAttributes = rachis::maxVertexAttributes;
            capabilities.limits.maxBindingSets = rachis::maxBindingSets;
            capabilities.limits.maxPushConstantBytes = rachis::maxPushConstantBytes;
            capabilities.limits.maxVariableResourceDescriptors = metalMaxArgumentBufferResources;
            capabilities.limits.maxVariableBufferDescriptors = metalMaxArgumentBufferResources;
            capabilities.limits.maxVariableSamplerDescriptors = metalMaxArgumentBufferSamplers;
            capabilities.vertexInput.supportedFormats = (std::uint64_t{1} << rachis::vertexFormatCount) - 1U;
            capabilities.vertexInput.maxBufferStride = 2048;
            capabilities.vertexInput.maxAttributeOffset = 2048;
            capabilities.vertexInput.maxInstanceStepRate = 2048;
            capabilities.pipelineCaches = true;
            capabilities.mipmapGeneration = true;
            capabilities.layeredRenderingModes = rachis::LayeredRenderingMode::vertexStage;
            capabilities.shaderFloat16 = true;
            capabilities.shaderInt16 = true;
            capabilities.shaderInt64 = true;
            capabilities.subgroupOperations = true;
            capabilities.subgroup = {.minSize = 32, .maxSize = 32};
            capabilities.partiallyBoundBindingArrays = true;
            capabilities.variableDescriptorCountBindingArrays = true;
            capabilities.enhancedBarriers = true;
            capabilities.bufferBarrierRanges = true;
            capabilities.debugMarkers = true;
            capabilities.persistentMapping = true;
            capabilities.textureViewComponentSwizzle = true;
            capabilities.mappedMemory.hostCoherent = true;
            capabilities.mappedMemory.nonCoherentAtomSize = 1;
            capabilities.indirectDispatch = true;
            capabilities.indirectDraws = true;
            capabilities.indirectDrawBaseInstance = true;
            capabilities.multiDrawIndirect = true;
            capabilities.drawIndirectCount = true;
            capabilities.multiDispatchIndirect = true;
            capabilities.dispatchIndirectCount = true;
            capabilities.primitiveRestart = true;
            if ([device supportsFamily:MTLGPUFamilyApple9])
            {
                const auto invocations = static_cast<std::uint32_t>(threadgroup.width);
                capabilities.meshShaders = true;
                capabilities.taskShaders = true;
                capabilities.meshShader = {
                    .maxTaskWorkgroupInvocations = invocations,
                    .maxTaskPayloadBytes = metalMaxMeshPayloadBytes,
                    .maxMeshWorkgroupInvocations = invocations,
                    .maxMeshOutputVertices = metalMaxMeshOutputVertices,
                    .maxMeshOutputPrimitives = metalMaxMeshOutputPrimitives,
                    .maxMeshOutputMemoryBytes = metalMaxMeshOutputMemoryBytes,
                    .maxWorkgroupCount = {65535, 65535, 65535},
                    .maxTotalWorkgroups = metalMaxMeshGridThreadgroups,
                };
            }
            return capabilities;
        }

        [[nodiscard]] id<MTLDevice> selectDevice(const std::optional<rachis::AdapterId>& adapterId)
        {
            NSArray<id<MTLDevice>>* const devices = MTLCopyAllDevices();
            if (adapterId.has_value())
            {
                for (id<MTLDevice> device in devices)
                {
                    if (device != nil && device.registryID == adapterId->value)
                    {
                        return device;
                    }
                }
                return nil;
            }

            for (id<MTLDevice> device in devices)
            {
                if (device != nil)
                {
                    return device;
                }
            }
            return MTLCreateSystemDefaultDevice();
        }

    } // namespace

    Result<std::vector<AdapterInfo>> enumerateAdapters()
    {
        @autoreleasepool
        {
            NSArray<id<MTLDevice>>* const devices = MTLCopyAllDevices();
            std::vector<AdapterInfo> adapters;
            adapters.reserve(devices.count);

            for (id<MTLDevice> device in devices)
            {
                if (device == nil)
                {
                    continue;
                }
                adapters.push_back(adapterFromDevice(device));
            }
            if (adapters.empty())
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::backendUnavailable, "No Metal device is available."));
            }
            return adapters;
        }
    }

    namespace
    {

        [[nodiscard]] MetalHostReport probeSelectedHost(id<MTLDevice> selected)
        {
            MetalHostReport host{};
            if (selected != nil)
            {
                host = probeHost(selected);
            }
            else
            {
#ifdef __arm64__
                host.appleSiliconArm64 = true;
#else
                host.intelMac = true;
#endif
                const auto version = NSProcessInfo.processInfo.operatingSystemVersion;
                host.macosMajor = static_cast<std::uint16_t>(version.majorVersion);
                host.macosMinor = static_cast<std::uint16_t>(version.minorVersion);
            }

#ifdef RACHIS_TESTING
            {
                std::scoped_lock const lock(metalTestingState().mutex);
                if (metalTestingState().hostReportOverride.has_value())
                {
                    host = *metalTestingState().hostReportOverride;
                }
            }
#endif
            return host;
        }

        [[nodiscard]] bool consumeCommandApiFailureInjection()
        {
            bool failCommandApi = false;
#ifdef RACHIS_TESTING
            {
                std::scoped_lock const lock(metalTestingState().mutex);
                failCommandApi = metalTestingState().failNextCommandApiCreate;
                metalTestingState().failNextCommandApiCreate = false;
            }
#endif
            return failCommandApi;
        }

        [[nodiscard]] id<MTL4CommandQueue> createBootstrapCommandQueue(id<MTLDevice> selected)
        {
            const bool failCommandApi = consumeCommandApiFailureInjection();

            id<MTL4CommandQueue> queue = nil;
            id<MTL4CommandAllocator> allocator = nil;
            id<MTL4CommandBuffer> commandBuffer = nil;
            if (!failCommandApi)
            {
                MTL4CommandQueueDescriptor* const queueDesc = [MTL4CommandQueueDescriptor new];
                queueDesc.label = metalNsString(kMetalQueueLabel);
                NSError* queueError = nil;
                queue = [selected newMTL4CommandQueueWithDescriptor:queueDesc error:&queueError];
                NSError* allocatorError = nil;
                MTL4CommandAllocatorDescriptor* const allocatorDesc = [MTL4CommandAllocatorDescriptor new];
                allocatorDesc.label = @"RHI.Metal.BootstrapAllocator";
                allocator = [selected newCommandAllocatorWithDescriptor:allocatorDesc error:&allocatorError];
                static_cast<void>(allocatorError);
                commandBuffer = [selected newCommandBuffer];
                if (commandBuffer != nil)
                {
                    commandBuffer.label = @"RHI.Metal.BootstrapCommand";
                }
                static_cast<void>(queueError);
            }
            if (queue == nil || allocator == nil || commandBuffer == nil)
            {
                queue = nil;
                allocator = nil;
                commandBuffer = nil;
                return nil;
            }
            allocator = nil;
            commandBuffer = nil;
            return queue;
        }

        [[nodiscard]] bool consumeLogStateFailureInjection()
        {
            bool failLogState = false;
#ifdef RACHIS_TESTING
            {
                std::scoped_lock const lock(metalTestingState().mutex);
                failLogState = metalTestingState().failNextLogStateCreate;
                metalTestingState().failNextLogStateCreate = false;
            }
#endif
            return failLogState;
        }

        [[nodiscard]] std::shared_ptr<MetalDiagnosticSink> makeDiagnosticSink(const DeviceDesc& desc)
        {
            auto sink = std::make_shared<MetalDiagnosticSink>();
            sink->callback = desc.diagnosticCallback;
            sink->userData = desc.diagnosticUserData;
            return sink;
        }

        [[nodiscard]] std::string formatMetalLogMessage(NSString* subSystem, NSString* category, NSString* message)
        {
            std::string text;
            const char* const subsystemText = subSystem.UTF8String;
            const char* const categoryText = category.UTF8String;
            const char* const messageText = message.UTF8String;
            if (subsystemText != nullptr && subSystem.length > 0)
            {
                text += subsystemText;
                text += '/';
            }
            if (categoryText != nullptr && category.length > 0)
            {
                text += categoryText;
                text += ": ";
            }
            if (messageText != nullptr)
            {
                text += messageText;
            }
            return text;
        }

        void addDiagnosticLogHandler(id<MTLLogState> logState, const std::shared_ptr<MetalDiagnosticSink>& sink)
        {
            std::weak_ptr<MetalDiagnosticSink> const weakSink = sink;
            [logState addLogHandler:^(NSString* subSystem, NSString* category, MTLLogLevel level, NSString* message) {
              @autoreleasepool
              {
                  const auto locked = weakSink.lock();
                  if (locked == nullptr)
                  {
                      return;
                  }
                  const std::string text = formatMetalLogMessage(subSystem, category, message);
                  locked->publish(metalLogSeverity(level), static_cast<std::uint32_t>(level), text);
              }
            }];
        }

        [[nodiscard]] bool gpuTraceCaptureSupported()
        {
            MTLCaptureManager* const captureManager = [MTLCaptureManager sharedCaptureManager];
            return captureManager != nil && [captureManager supportsDestination:MTLCaptureDestinationGPUTraceDocument];
        }

        [[nodiscard]] std::vector<FormatProperties> probeAllFormats(id<MTLDevice> selected)
        {
            std::vector<FormatProperties> formats(static_cast<std::size_t>(Format::bc6hRgbSfloat) + 1);
            for (std::size_t i = 0; i < formats.size(); ++i)
            {
                formats[i] = probeFormat(selected, static_cast<Format>(i));
            }
            return formats;
        }

        [[nodiscard]] Status createQueueTimelines(id<MTLDevice> selected, MetalDeviceNative& native)
        {
            constexpr std::array timelineQueues{
                QueueType::graphics,
                QueueType::compute,
                QueueType::copy,
            };
            for (std::size_t index = 0; index < timelineQueues.size(); ++index)
            {
                auto& timeline = native.timelines[index];
                timeline.type = timelineQueues[index];
                timeline.event = [selected newSharedEvent];
                if (timeline.event == nil || native.queueGate == nullptr)
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::deviceCreationFailed,
                        "Metal shared-event timelines could not be created. Partial Metal objects were released."));
                }
                timeline.event.label = [NSString stringWithUTF8String:metalQueueTimelineLabel(timelineQueues[index])];
                timeline.nextValue = 1;
                timeline.lastSubmitted.store(0, std::memory_order_relaxed);
            }
            return {};
        }

        [[nodiscard]] Result<std::shared_ptr<MetalDeviceNative>> assembleDeviceNative(
            id<MTLDevice> selected,
            id<MTL4CommandQueue> queue,
            id<MTLLogState> logState,
            std::shared_ptr<MetalDiagnosticSink>& sink)
        {
            auto native =
                std::shared_ptr<MetalDeviceNative>(new MetalDeviceNative(), [](MetalDeviceNative* released) noexcept {
                    @autoreleasepool
                    {
                        delete released;
                    }
                });
            native->device = selected;
            native->commandQueue = queue;
            MTL4CompilerDescriptor* const compilerDesc = [MTL4CompilerDescriptor new];
            compilerDesc.label = @"RHI.Metal.Compiler";
            NSError* compilerError = nil;
            native->compiler = [selected newCompilerWithDescriptor:compilerDesc error:&compilerError];
            if (native->compiler == nil)
            {
                return std::unexpected(
                    metalNsError(ErrorCode::deviceCreationFailed, compilerError, "newCompilerWithDescriptor"));
            }
            native->logState = logState;
            native->sink = std::move(sink);
            const auto token = detail::deviceTokenSource().acquire();
            if (!token.has_value())
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::resourceExhausted, "Metal device-token identities are exhausted."));
            }
            native->deviceToken = *token;
            if (auto timelines = createQueueTimelines(selected, *native); !timelines)
            {
                return std::unexpected(std::move(timelines.error()));
            }
            return native;
        }

        [[nodiscard]] MetalValidationConfiguration makeValidationConfiguration(const DeviceDesc& desc,
                                                                               MetalValidationPlan plan,
                                                                               bool logStateEnabled,
                                                                               bool gpuTraceSupported)
        {
            return MetalValidationConfiguration{
                .requested = desc.validation,
                .requireValidation = desc.requireValidation,
                .plan = plan,
                .logStateEnabled = logStateEnabled,
                .debugLayerEnvironment = metalDebugLayerEnvironmentEnabled(),
                .shaderValidationEnvironment = metalShaderValidationEnvironmentEnabled(),
                .gpuTraceCaptureSupported = gpuTraceSupported,
                .debugMarkers = true,
                .deviceLabel = std::string(kMetalDeviceLabel),
                .queueLabel = std::string(kMetalQueueLabel),
            };
        }

        [[nodiscard]] QueueProperties makeQueueClock(id<MTLDevice> selected)
        {
            QueueProperties queueClock{};
            queueClock.timestampQueries = true;
            queueClock.inlineTimestampWrites = true;
            queueClock.timestampValidBits = 64;
            queueClock.timestampFrequency = [selected queryTimestampFrequency];
            if (queueClock.timestampFrequency == 0)
            {
                queueClock.timestampQueries = false;
                queueClock.inlineTimestampWrites = false;
                queueClock.timestampValidBits = 0;
            }
            return queueClock;
        }

    } // namespace

    Result<std::unique_ptr<detail::BackendDevice>> createDevice(const DeviceDesc& desc)
    {
        @autoreleasepool
        {
            id<MTLDevice> const selected = selectDevice(desc.adapterId);
            if (desc.adapterId.has_value() && selected == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::adapterNotFound,
                                        "The requested Metal adapterId is not in this backend's enumerated list."));
            }

            const MetalHostReport host = probeSelectedHost(selected);
            const auto decision = evaluateMacosMetalProfile(host, MetalCreateIntent{.warpPreference = false});
            if (!decision.accepted)
            {
                return std::unexpected(metalProfileError(decision));
            }
            if (selected == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::backendUnavailable, "No Metal device is available."));
            }

            id<MTL4CommandQueue> queue = createBootstrapCommandQueue(selected);
            if (queue == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::deviceCreationFailed,
                    "Metal 4 command-queue objects could not be created. Partial Metal objects were released."));
            }

            const bool failLogState = consumeLogStateFailureInjection();
            const bool validationRequested = desc.validation != ValidationLevel::disabled || desc.requireValidation;
            auto sink = makeDiagnosticSink(desc);

            id<MTLLogState> logState = nil;
            if (validationRequested && !failLogState)
            {
                MTLLogStateDescriptor* const logDesc = [MTLLogStateDescriptor new];
                logDesc.bufferSize = static_cast<NSInteger>(metalLogStateBufferBytes);
                logDesc.level = desc.validation == ValidationLevel::gpu ? MTLLogLevelDebug : MTLLogLevelNotice;
                NSError* logError = nil;
                logState = [selected newLogStateWithDescriptor:logDesc error:&logError];
                if (logState != nil)
                {
                    addDiagnosticLogHandler(logState, sink);
                }
                else if (desc.requireValidation)
                {
                    queue = nil;
                    return std::unexpected(
                        metalNsError(ErrorCode::validationError, logError, "newLogStateWithDescriptor"));
                }
            }

            const auto plan = planMetalValidation(desc.validation, desc.requireValidation, logState != nil);
            if (plan == MetalValidationPlan::requiredUnavailable)
            {
                queue = nil;
                logState = nil;
                return std::unexpected(
                    metalSelectionError(ErrorCode::validationError,
                                        "MTLLogState could not be created for DeviceDesc::requireValidation. Partial "
                                        "Metal objects were released."));
            }

            const bool gpuTraceSupported = gpuTraceCaptureSupported();
            std::vector<FormatProperties> formats = probeAllFormats(selected);

            auto native = assembleDeviceNative(selected, queue, logState, sink);
            if (!native)
            {
                return std::unexpected(std::move(native.error()));
            }
            (*native)->validation = makeValidationConfiguration(desc, plan, logState != nil, gpuTraceSupported);
            QueueProperties queueClock = makeQueueClock(selected);
            std::unique_ptr<MetalDevice> device =
                std::make_unique<MetalDevice>(adapterFromDevice(selected),
                                              publishMetalCapabilities(selected, host),
                                              std::move(formats),
                                              queueClock,
                                              std::move(*native));
#ifdef RACHIS_TESTING
            bindMetalDiagnosticTestDevice(device.get());
#endif
            return std::unique_ptr<detail::BackendDevice>(std::move(device));
        }
    }

} // namespace rachis::metal
