#include "Rachis/Device.h"

#include "Internal/BackendCommandBuffer.h"
#include "Internal/BackendDevice.h"
#include "Internal/CommandBufferAccess.h"
#include "Internal/DeviceAccess.h"
#include "Internal/FillAndClearValidation.h"
#include "Internal/PortableValidation.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace
{

    using rachis::AcquiredSwapchainImage;
    using rachis::AdapterInfo;
    using rachis::BackendAllocationSnapshot;
    using rachis::BindingFlag;
    using rachis::BindingLayoutDesc;
    using rachis::BindingLayoutEntry;
    using rachis::BindingLayoutHandle;
    using rachis::BindingPoolDesc;
    using rachis::BindingPoolHandle;
    using rachis::BindingPoolSize;
    using rachis::BindingSetDesc;
    using rachis::BindingSetHandle;
    using rachis::BindingType;
    using rachis::BindingUpdateStatistics;
    using rachis::BindingWrite;
    using rachis::BlendConstants;
    using rachis::BufferBarrier;
    using rachis::BufferCopy;
    using rachis::BufferCopyRegion;
    using rachis::BufferDesc;
    using rachis::BufferHandle;
    using rachis::BufferTextureCopyRegion;
    using rachis::ClearColor;
    using rachis::CommandBuffer;
    using rachis::CommandPoolDesc;
    using rachis::CommandPoolHandle;
    using rachis::ComputePipelineDesc;
    using rachis::DebugColor;
    using rachis::DebugMarkerDesc;
    using rachis::DepthBias;
    using rachis::DepthStencilClearAspect;
    using rachis::DeviceCapabilities;
    using rachis::DeviceFaultReport;
    using rachis::DeviceOperationalState;
    using rachis::DiagnosticCounts;
    using rachis::Error;
    using rachis::ErrorCode;
    using rachis::Format;
    using rachis::FormatProperties;
    using rachis::GraphicsPipelineDesc;
    using rachis::IndexFormat;
    using rachis::IndirectDispatchDesc;
    using rachis::IndirectDrawDesc;
    using rachis::MappedMemory;
    using rachis::MappedMemoryRange;
    using rachis::MemoryBudget;
    using rachis::PipelineCacheDesc;
    using rachis::PipelineCacheHandle;
    using rachis::PipelineCompilationHandle;
    using rachis::PipelineCompilationState;
    using rachis::PipelineHandle;
    using rachis::PipelineLayoutDesc;
    using rachis::PipelineLayoutHandle;
    using rachis::PresentStatus;
    using rachis::QueueClockCalibration;
    using rachis::QueueProperties;
    using rachis::QueueType;
    using rachis::RenderingDesc;
    using rachis::ResourceBarrierBatch;
    using rachis::ResourceHeapDesc;
    using rachis::ResourceMemoryRequirements;
    using rachis::ResourceMemorySnapshot;
    using rachis::Result;
    using rachis::SamplerDesc;
    using rachis::SamplerHandle;
    using rachis::ScissorRect;
    using rachis::ShaderInterfaceSnapshot;
    using rachis::ShaderLibraryDesc;
    using rachis::ShaderLibraryHandle;
    using rachis::ShaderModuleDesc;
    using rachis::ShaderModuleHandle;
    using rachis::ShaderStage;
    using rachis::Status;
    using rachis::StencilFace;
    using rachis::SurfaceCapabilities;
    using rachis::SurfaceDesc;
    using rachis::SurfaceHandle;
    using rachis::SwapchainDesc;
    using rachis::SwapchainFrameTiming;
    using rachis::SwapchainHandle;
    using rachis::SwapchainPresentationInfo;
    using rachis::TextureBarrier;
    using rachis::TextureCopyRegion;
    using rachis::TextureDesc;
    using rachis::TextureHandle;
    using rachis::TextureMipmapGeneration;
    using rachis::TextureResolve;
    using rachis::TextureTransferLayout;
    using rachis::TextureViewDesc;
    using rachis::TextureViewHandle;
    using rachis::TimelinePoint;
    using rachis::TimelinePointStatus;
    using rachis::VertexAttribute;
    using rachis::VertexBufferBinding;
    using rachis::VertexBufferLayout;
    using rachis::VertexFormat;
    using rachis::VertexStepMode;
    using rachis::Viewport;
    using rachis::detail::BackendCommandBuffer;
    using rachis::detail::BackendCommandBufferKind;
    using rachis::detail::DebugGroupOwner;

    class TrackingCommandBuffer final : public BackendCommandBuffer
    {
    public:
        explicit TrackingCommandBuffer(std::uint64_t token,
                                       QueueType queueType = QueueType::graphics,
                                       BackendCommandBufferKind kind = BackendCommandBufferKind::custom) noexcept
            : BackendCommandBuffer(kind), queueType_(queueType), token_(token)
        {
        }

        [[nodiscard]] QueueType queueType() const noexcept override
        {
            return queueType_;
        }
        [[nodiscard]] std::uint64_t deviceToken() const noexcept override
        {
            return token_;
        }

    private:
        QueueType queueType_ = QueueType::graphics;
        std::uint64_t token_ = 0;
    };

    class TrackingBackendDevice final : public rachis::detail::BackendDevice
    {
    public:
        std::string name = "Tracking";
        std::uint64_t token = 1001;
        DeviceOperationalState state = DeviceOperationalState::active;
        std::optional<Error> lossError = Error{
            ErrorCode::deviceLost,
            0xDEAD,
            "Simulated GPU fault.",
        };
        DeviceCapabilities caps{};
        AdapterInfo adapter{};
        std::optional<rachis::ShaderWorkgroupSize> moduleWorkgroupSize{};
        std::size_t backendInvocations = 0;

        TrackingBackendDevice()
        {
            caps.debugMarkers = true;
            caps.limits.maxViewports = 4;
            caps.limits.maxPushConstantBytes = rachis::maxPushConstantBytes;
        }

        [[nodiscard]] std::string_view backendName() const noexcept override
        {
            return name;
        }
        [[nodiscard]] std::uint64_t deviceToken() const noexcept override
        {
            return token;
        }
        [[nodiscard]] const AdapterInfo& adapterInfo() const noexcept override
        {
            return adapter;
        }
        [[nodiscard]] const DeviceCapabilities& capabilities() const noexcept override
        {
            return caps;
        }
        [[nodiscard]] DeviceOperationalState operationalState() const noexcept override
        {
            return state;
        }
        [[nodiscard]] std::optional<Error> deviceLossError() const override
        {
            return state == DeviceOperationalState::lost ? lossError : std::nullopt;
        }

        [[nodiscard]] Result<std::vector<MemoryBudget>> queryMemoryBudgets(std::uint32_t) const override
        {
            return std::unexpected(unsupportedOperation("queryMemoryBudgets"));
        }
        [[nodiscard]] Result<ResourceMemorySnapshot> captureResourceMemorySnapshot() const override
        {
            return std::unexpected(unsupportedOperation("captureResourceMemorySnapshot"));
        }
        [[nodiscard]] Result<DeviceFaultReport> captureDeviceFaultReport() const override
        {
            return std::unexpected(unsupportedOperation("captureDeviceFaultReport"));
        }
        [[nodiscard]] BackendAllocationSnapshot captureBackendAllocationSnapshot() const noexcept override
        {
            return {};
        }
        [[nodiscard]] FormatProperties queryFormatProperties(Format) const noexcept override
        {
            return {};
        }
        [[nodiscard]] QueueProperties queueProperties(QueueType) const noexcept override
        {
            return {};
        }
        [[nodiscard]] Result<QueueClockCalibration> calibrateQueueClock(QueueType) const override
        {
            return std::unexpected(unsupportedOperation("calibrateQueueClock"));
        }
        [[nodiscard]] DiagnosticCounts diagnosticCounts() const noexcept override
        {
            return {};
        }
        [[nodiscard]] BindingUpdateStatistics bindingUpdateStatistics() const noexcept override
        {
            return {};
        }

        [[nodiscard]] Result<CommandPoolHandle> createCommandPool(const CommandPoolDesc&) override
        {
            ++backendInvocations;
            return CommandPoolHandle{};
        }
        [[nodiscard]] Status destroyCommandPool(CommandPoolHandle) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status resetCommandPool(CommandPoolHandle) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Result<std::unique_ptr<BackendCommandBuffer>> acquireCommandBuffer(CommandPoolHandle) override
        {
            ++backendInvocations;
            return std::make_unique<TrackingCommandBuffer>(token);
        }
        [[nodiscard]] Status resetCommandBuffer(BackendCommandBuffer& cb) override
        {
            ++backendInvocations;
            cb.debugState().reset();
            return {};
        }
        [[nodiscard]] Result<TimelinePoint> submit(std::span<std::unique_ptr<BackendCommandBuffer>> commandBuffers,
                                                   std::span<const TimelinePoint>) override
        {
            ++backendInvocations;
            for (auto& cb : commandBuffers)
            {
                cb.reset();
            }
            return TimelinePoint{};
        }
        [[nodiscard]] Result<TimelinePointStatus> queryTimelinePoint(TimelinePoint) const override
        {
            return TimelinePointStatus::complete;
        }
        [[nodiscard]] Result<TimelinePointStatus> wait(TimelinePoint, std::chrono::nanoseconds) override
        {
            ++backendInvocations;
            return TimelinePointStatus::complete;
        }
        [[nodiscard]] Status wait(TimelinePoint) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status waitIdle(QueueType) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status waitIdle() override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] std::uint64_t completedValue(QueueType) const noexcept override
        {
            return 0;
        }
        void collectGarbage() noexcept override
        {
        }

        [[nodiscard]] Status beginDebugGroup(BackendCommandBuffer& cb, const DebugMarkerDesc&) override
        {
            ++backendInvocations;
            if (name == "D3D12" && !caps.debugMarkers)
            {
                return std::unexpected(Error{ErrorCode::unsupported, 0, "D3D12 debug markers disabled."});
            }
            auto& debug = cb.debugState();
            if (debug.groupDepth == 0)
            {
                debug.groupThread = std::this_thread::get_id();
            }
            debug.groupOwners[debug.groupDepth] = DebugGroupOwner::user;
            ++debug.groupDepth;
            return {};
        }
        [[nodiscard]] Status endDebugGroup(BackendCommandBuffer& cb) override
        {
            ++backendInvocations;
            if (name == "D3D12" && !caps.debugMarkers)
            {
                return std::unexpected(Error{ErrorCode::unsupported, 0, "D3D12 debug markers disabled."});
            }
            --cb.debugState().groupDepth;
            return {};
        }
        [[nodiscard]] Status insertDebugMarker(BackendCommandBuffer&, const DebugMarkerDesc&) override
        {
            ++backendInvocations;
            if (name == "D3D12" && !caps.debugMarkers)
            {
                return std::unexpected(Error{ErrorCode::unsupported, 0, "D3D12 debug markers disabled."});
            }
            return {};
        }

        [[nodiscard]] Result<ResourceMemoryRequirements> queryBufferMemoryRequirements(const BufferDesc&) const override
        {
            return ResourceMemoryRequirements{};
        }
        [[nodiscard]] Result<BufferHandle> createBuffer(const BufferDesc&) override
        {
            ++backendInvocations;
            return BufferHandle{};
        }
        [[nodiscard]] Status destroyBuffer(BufferHandle) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Result<MappedMemory> mapBuffer(BufferHandle, std::uint64_t, std::uint64_t) override
        {
            ++backendInvocations;
            return MappedMemory{};
        }
        [[nodiscard]] Status unmapBuffer(BufferHandle, std::uint64_t, std::uint64_t) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status flushMappedMemoryRanges(BufferHandle, std::span<const MappedMemoryRange>) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status invalidateMappedMemoryRanges(BufferHandle, std::span<const MappedMemoryRange>) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status writeBuffer(BufferHandle, std::uint64_t, std::span<const std::byte>) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status readBuffer(BufferHandle, std::uint64_t, std::span<std::byte>) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status copyBufferRegions(BackendCommandBuffer&,
                                               BufferHandle,
                                               BufferHandle,
                                               std::span<const BufferCopyRegion>) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status fillBuffer(
            BackendCommandBuffer&, BufferHandle, std::uint64_t, std::uint64_t, std::uint32_t) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status clearStorageTexture(BackendCommandBuffer&,
                                                 TextureViewHandle,
                                                 const rachis::StorageTextureClearValue&) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status bufferBarrier(BackendCommandBuffer&, const BufferBarrier&) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status resourceBarriers(BackendCommandBuffer&, const ResourceBarrierBatch&) override
        {
            ++backendInvocations;
            return {};
        }

        [[nodiscard]] Result<ResourceMemoryRequirements> queryTextureMemoryRequirements(
            const TextureDesc&) const override
        {
            return ResourceMemoryRequirements{};
        }
        [[nodiscard]] Result<TextureHandle> createTexture(const TextureDesc&) override
        {
            ++backendInvocations;
            return TextureHandle{};
        }
        [[nodiscard]] Status destroyTexture(TextureHandle) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Result<TextureTransferLayout> queryTextureTransferLayout(
            TextureHandle, const BufferTextureCopyRegion&) const override
        {
            return TextureTransferLayout{};
        }
        [[nodiscard]] Status copyBufferToTextureRegions(BackendCommandBuffer&,
                                                        BufferHandle,
                                                        TextureHandle,
                                                        std::span<const BufferTextureCopyRegion>) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status copyTextureToBufferRegions(BackendCommandBuffer&,
                                                        TextureHandle,
                                                        BufferHandle,
                                                        std::span<const BufferTextureCopyRegion>) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status copyTextureRegions(BackendCommandBuffer&,
                                                TextureHandle,
                                                TextureHandle,
                                                std::span<const TextureCopyRegion>) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status resolveTexture(BackendCommandBuffer&, const TextureResolve&) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status generateMipmaps(BackendCommandBuffer&, const TextureMipmapGeneration&) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status textureBarrier(BackendCommandBuffer&, const TextureBarrier&) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Result<TextureViewHandle> createTextureView(const TextureViewDesc&) override
        {
            ++backendInvocations;
            return TextureViewHandle{};
        }
        [[nodiscard]] Status destroyTextureView(TextureViewHandle) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Result<SamplerHandle> createSampler(const SamplerDesc&) override
        {
            ++backendInvocations;
            return SamplerHandle{};
        }
        [[nodiscard]] Status destroySampler(SamplerHandle) override
        {
            ++backendInvocations;
            return {};
        }

        [[nodiscard]] Result<BindingLayoutHandle> createBindingLayout(const BindingLayoutDesc& desc) override
        {
            ++backendInvocations;
            for (const auto& entry : desc.entries)
            {
                if (entry.type == BindingType::accelerationStructure && !caps.rayTracing)
                {
                    return std::unexpected(unsupportedOperation("accelerationStructure bindings"));
                }
                if ((entry.type == BindingType::minMipFeedback || entry.type == BindingType::mipRegionUsedFeedback) &&
                    !caps.samplerFeedback)
                {
                    return std::unexpected(unsupportedOperation("sampler-feedback bindings"));
                }
            }
            return BindingLayoutHandle{};
        }
        [[nodiscard]] Status destroyBindingLayout(BindingLayoutHandle) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Result<BindingPoolHandle> createBindingPool(const BindingPoolDesc&) override
        {
            ++backendInvocations;
            return BindingPoolHandle{};
        }
        [[nodiscard]] Status destroyBindingPool(BindingPoolHandle) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Result<BindingSetHandle> createBindingSet(const BindingSetDesc&) override
        {
            ++backendInvocations;
            return BindingSetHandle{};
        }
        [[nodiscard]] Status updateBindingSet(BindingSetHandle, std::span<const BindingWrite>) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status destroyBindingSet(BindingSetHandle) override
        {
            ++backendInvocations;
            return {};
        }

        [[nodiscard]] Result<ShaderLibraryHandle> createShaderLibrary(const ShaderLibraryDesc&) override
        {
            ++backendInvocations;
            return ShaderLibraryHandle{};
        }
        [[nodiscard]] Status destroyShaderLibrary(ShaderLibraryHandle) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Result<ShaderModuleHandle> createShaderModule(const ShaderModuleDesc&) override
        {
            ++backendInvocations;
            return ShaderModuleHandle{};
        }
        [[nodiscard]] Result<ShaderInterfaceSnapshot> shaderModuleInterface(ShaderModuleHandle) const override
        {
            return ShaderInterfaceSnapshot{.workgroupSize = moduleWorkgroupSize};
        }
        [[nodiscard]] Status destroyShaderModule(ShaderModuleHandle) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Result<PipelineLayoutHandle> createPipelineLayout(const PipelineLayoutDesc&) override
        {
            ++backendInvocations;
            return PipelineLayoutHandle{};
        }
        [[nodiscard]] Status destroyPipelineLayout(PipelineLayoutHandle) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Result<PipelineCacheHandle> createPipelineCache(const PipelineCacheDesc&) override
        {
            ++backendInvocations;
            return PipelineCacheHandle{};
        }
        [[nodiscard]] Status destroyPipelineCache(PipelineCacheHandle) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Result<std::vector<std::byte>> pipelineCacheData(PipelineCacheHandle) override
        {
            ++backendInvocations;
            return std::vector<std::byte>{};
        }
        [[nodiscard]] Result<PipelineHandle> createComputePipeline(const ComputePipelineDesc&) override
        {
            ++backendInvocations;
            return PipelineHandle{};
        }
        [[nodiscard]] Result<PipelineHandle> createGraphicsPipeline(const GraphicsPipelineDesc&) override
        {
            ++backendInvocations;
            return PipelineHandle{};
        }
        [[nodiscard]] Result<PipelineCompilationHandle> beginComputePipelineCompilation(
            const ComputePipelineDesc&) override
        {
            ++backendInvocations;
            return PipelineCompilationHandle{};
        }
        [[nodiscard]] Result<PipelineCompilationHandle> beginGraphicsPipelineCompilation(
            const GraphicsPipelineDesc&) override
        {
            ++backendInvocations;
            return PipelineCompilationHandle{};
        }
        [[nodiscard]] Result<PipelineCompilationState> pipelineCompilationState(PipelineCompilationHandle) override
        {
            ++backendInvocations;
            return PipelineCompilationState::succeeded;
        }
        [[nodiscard]] Result<PipelineHandle> waitPipelineCompilation(PipelineCompilationHandle) override
        {
            ++backendInvocations;
            return PipelineHandle{};
        }
        [[nodiscard]] Status destroyPipelineCompilation(PipelineCompilationHandle) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status destroyPipeline(PipelineHandle) override
        {
            ++backendInvocations;
            return {};
        }

        [[nodiscard]] Status bindComputePipeline(BackendCommandBuffer&, PipelineHandle) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status bindComputeBindingSet(BackendCommandBuffer&,
                                                   std::uint32_t,
                                                   BindingSetHandle,
                                                   std::span<const std::uint32_t>) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status bindComputeBindingSets(BackendCommandBuffer&,
                                                    std::uint32_t,
                                                    std::span<const BindingSetHandle>,
                                                    std::span<const std::uint32_t>) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status setComputePushConstants(BackendCommandBuffer&,
                                                     std::uint32_t,
                                                     std::span<const std::byte>) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status dispatch(BackendCommandBuffer&, std::uint32_t, std::uint32_t, std::uint32_t) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status dispatchIndirect(BackendCommandBuffer&, const IndirectDispatchDesc&) override
        {
            ++backendInvocations;
            return {};
        }

        [[nodiscard]] Status beginRendering(BackendCommandBuffer& cb, const RenderingDesc&) override
        {
            ++backendInvocations;
            cb.graphicsState().renderingOwner = std::make_shared<int>(1);
            return {};
        }
        [[nodiscard]] Status endRendering(BackendCommandBuffer& cb) override
        {
            ++backendInvocations;
            cb.graphicsState().renderingOwner.reset();
            return {};
        }
        [[nodiscard]] Status clearColorAttachment(BackendCommandBuffer&,
                                                  std::uint32_t,
                                                  const ClearColor&,
                                                  std::span<const ScissorRect>) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status clearDepthStencilAttachment(
            BackendCommandBuffer&, DepthStencilClearAspect, float, std::uint32_t, std::span<const ScissorRect>) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status bindGraphicsPipeline(BackendCommandBuffer&, PipelineHandle) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status bindGraphicsBindingSet(BackendCommandBuffer&,
                                                    std::uint32_t,
                                                    BindingSetHandle,
                                                    std::span<const std::uint32_t>) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status bindGraphicsBindingSets(BackendCommandBuffer&,
                                                     std::uint32_t,
                                                     std::span<const BindingSetHandle>,
                                                     std::span<const std::uint32_t>) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status setGraphicsPushConstants(BackendCommandBuffer&,
                                                      ShaderStage,
                                                      std::uint32_t,
                                                      std::span<const std::byte>) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status setViewports(BackendCommandBuffer&, std::span<const Viewport>) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status setScissors(BackendCommandBuffer&, std::span<const ScissorRect>) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status setBlendConstants(BackendCommandBuffer&, const BlendConstants&) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status setStencilReference(BackendCommandBuffer&, StencilFace, std::uint32_t) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status bindVertexBuffers(BackendCommandBuffer&,
                                               std::uint32_t,
                                               std::span<const VertexBufferBinding>) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status bindIndexBuffer(BackendCommandBuffer&, BufferHandle, IndexFormat, std::uint64_t) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status draw(
            BackendCommandBuffer&, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status drawIndexed(
            BackendCommandBuffer&, std::uint32_t, std::uint32_t, std::uint32_t, std::int32_t, std::uint32_t) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status drawIndirect(BackendCommandBuffer&, const IndirectDrawDesc&) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status drawIndexedIndirect(BackendCommandBuffer&, const IndirectDrawDesc&) override
        {
            ++backendInvocations;
            return {};
        }

        [[nodiscard]] Result<SurfaceHandle> createSurface(const SurfaceDesc&) override
        {
            ++backendInvocations;
            return SurfaceHandle{};
        }
        [[nodiscard]] Result<SurfaceCapabilities> querySurfaceCapabilities(SurfaceHandle) const override
        {
            return SurfaceCapabilities{};
        }
        [[nodiscard]] Status destroySurface(SurfaceHandle) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Result<SwapchainHandle> createSwapchain(const SwapchainDesc&) override
        {
            ++backendInvocations;
            return SwapchainHandle{};
        }
        [[nodiscard]] Status destroySwapchain(SwapchainHandle) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status resizeSwapchain(SwapchainHandle, std::uint32_t, std::uint32_t) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Result<AcquiredSwapchainImage> acquireNextImage(SwapchainHandle) override
        {
            ++backendInvocations;
            return AcquiredSwapchainImage{};
        }
        [[nodiscard]] bool hasAcquiredSwapchainImage(SwapchainHandle) const noexcept override
        {
            return false;
        }
        [[nodiscard]] Status cancelSwapchainImage(SwapchainHandle) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Result<PresentStatus> present(SwapchainHandle, TimelinePoint) override
        {
            ++backendInvocations;
            return PresentStatus::presented;
        }
        [[nodiscard]] Result<SwapchainPresentationInfo> querySwapchainPresentation(SwapchainHandle) const override
        {
            return SwapchainPresentationInfo{};
        }
        [[nodiscard]] Result<SwapchainFrameTiming> querySwapchainFrameTiming(SwapchainHandle) const override
        {
            return SwapchainFrameTiming{};
        }

        bool instanceRecords = false;
        mutable std::size_t referenceQueries = 0;

        [[nodiscard]] Result<rachis::AccelerationStructureInstanceLayout> queryAccelerationStructureInstanceLayout()
            const override
        {
            if (!instanceRecords)
            {
                return std::unexpected(unsupportedOperation("queryAccelerationStructureInstanceLayout"));
            }
            return rachis::AccelerationStructureInstanceLayout{
                .format = rachis::AccelerationStructureInstanceFormat::packed64,
                .size = 64,
                .alignment = 16,
            };
        }
        [[nodiscard]] Result<std::uint64_t> queryAccelerationStructureReference(
            rachis::AccelerationStructureHandle) const override
        {
            ++referenceQueries;
            return 0x1000;
        }
        [[nodiscard]] Status buildTopLevelAccelerationStructureFromBuffer(
            BackendCommandBuffer&, const rachis::TopLevelAccelerationStructureBufferBuildDesc&) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status updateTopLevelAccelerationStructureFromBuffer(
            BackendCommandBuffer&, const rachis::TopLevelAccelerationStructureBufferUpdateDesc&) override
        {
            ++backendInvocations;
            return {};
        }
        [[nodiscard]] Status updateTopLevelAccelerationStructureInPlaceFromBuffer(
            BackendCommandBuffer&, const rachis::TopLevelAccelerationStructureBufferInPlaceUpdateDesc&) override
        {
            ++backendInvocations;
            return {};
        }
    };

    [[nodiscard]] CommandBuffer makeCommandBuffer(std::uint64_t token, QueueType queueType = QueueType::graphics)
    {
        return rachis::detail::CommandBufferAccess::make(std::make_unique<TrackingCommandBuffer>(token, queueType));
    }

    struct RecordingStateLockProbe final
    {
        std::shared_ptr<std::mutex> mutex = std::make_shared<std::mutex>();
        std::atomic<bool> requested{false};
        std::thread::id requester;
    };

    class LockedStateCommandBuffer final : public BackendCommandBuffer
    {
    public:
        LockedStateCommandBuffer(std::uint64_t token, RecordingStateLockProbe& probe) noexcept
            : token_(token), probe_(&probe)
        {
        }

        [[nodiscard]] QueueType queueType() const noexcept override
        {
            return QueueType::graphics;
        }
        [[nodiscard]] std::uint64_t deviceToken() const noexcept override
        {
            return token_;
        }
        [[nodiscard]] rachis::detail::RecordingStateLock lockRecordingState() const override
        {
            probe_->requester = std::this_thread::get_id();
            probe_->requested.store(true, std::memory_order_release);
            return rachis::detail::RecordingStateLock(probe_->mutex);
        }

    private:
        std::uint64_t token_ = 0;
        RecordingStateLockProbe* probe_ = nullptr;
    };

    template<typename Reject, typename Accept, typename Record>
    void checkStateReadUnderRecordingLock(std::string_view operation, Reject reject, Accept accept, Record record)
    {
        INFO("Operation: " << operation);
        auto backend = std::make_unique<TrackingBackendDevice>();
        auto* raw = backend.get();
        auto device = rachis::detail::DeviceAccess::make(std::move(backend));
        RecordingStateLockProbe probe;
        auto cb =
            rachis::detail::CommandBufferAccess::make(std::make_unique<LockedStateCommandBuffer>(raw->token, probe));
        auto* nativeCb = rachis::detail::CommandBufferAccess::get(cb);
        REQUIRE(nativeCb != nullptr);

        std::optional<Status> recorded;
        std::atomic<bool> returned{false};
        bool requestedBeforeReturn = false;
        bool returnedWhileLocked = true;
        {
            std::jthread worker;
            std::unique_lock held(*probe.mutex);
            reject(*nativeCb);
            worker = std::jthread([&] {
                recorded = record(device, cb);
                returned.store(true, std::memory_order_release);
            });
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (!probe.requested.load(std::memory_order_acquire) && !returned.load(std::memory_order_acquire) &&
                   std::chrono::steady_clock::now() < deadline)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            requestedBeforeReturn = probe.requested.load(std::memory_order_acquire);
            returnedWhileLocked = returned.load(std::memory_order_acquire);
            if (requestedBeforeReturn && !returnedWhileLocked)
            {
                accept(*nativeCb, probe.requester);
            }
        }

        CHECK(requestedBeforeReturn);
        CHECK_FALSE(returnedWhileLocked);
        REQUIRE(recorded.has_value());
        INFO("Error: " << (recorded->has_value() ? std::string{} : recorded->error().message));
        CHECK(recorded->has_value());
        CHECK(raw->backendInvocations == 1);
    }

} // namespace

TEST_CASE("Device frontend rejects foreign and zero-token command buffers before scope checks or backend calls",
          "[size:small][rhi][core]")
{
    auto backend = std::make_unique<TrackingBackendDevice>();
    auto* raw = backend.get();
    auto device = rachis::detail::DeviceAccess::make(std::move(backend));

    auto foreignCb = makeCommandBuffer(raw->token + 1);
    auto* foreignNative = rachis::detail::CommandBufferAccess::get(foreignCb);
    REQUIRE(foreignNative != nullptr);
    foreignNative->graphicsState().renderingOwner = std::make_shared<int>(42);
    foreignNative->queryState().activeOcclusion = rachis::detail::ActiveQuery{};
    foreignNative->debugState().groupDepth = 1;
    foreignNative->debugState().groupThread = std::this_thread::get_id();

    const auto singleSubmit = device.submit(std::move(foreignCb));
    REQUIRE_FALSE(singleSubmit.has_value());
    CHECK(singleSubmit.error().code == ErrorCode::invalidArgument);
    CHECK(foreignCb.isValid());
    CHECK(raw->backendInvocations == 0);

    auto ownedOpenScopeCb = makeCommandBuffer(raw->token);
    rachis::detail::CommandBufferAccess::get(ownedOpenScopeCb)->graphicsState().renderingOwner =
        std::make_shared<int>(7);
    std::array batch{std::move(ownedOpenScopeCb), std::move(foreignCb)};
    const auto batchSubmit = device.submit(batch);
    REQUIRE_FALSE(batchSubmit.has_value());
    CHECK(batchSubmit.error().code == ErrorCode::invalidArgument);
    CHECK(batch[0].isValid());
    CHECK(batch[1].isValid());
    CHECK(raw->backendInvocations == 0);
    foreignCb = std::move(batch[1]);

    CHECK(device.resetCommandBuffer(foreignCb).error().code == ErrorCode::invalidArgument);
    CHECK(device.beginRendering(foreignCb, RenderingDesc{}).error().code == ErrorCode::invalidArgument);
    CHECK(device.endRendering(foreignCb).error().code == ErrorCode::invalidArgument);
    CHECK(device.beginDebugGroup(foreignCb, DebugMarkerDesc{.label = "group"}).error().code ==
          ErrorCode::invalidArgument);
    CHECK(device.endDebugGroup(foreignCb).error().code == ErrorCode::invalidArgument);
    CHECK(device.insertDebugMarker(foreignCb, DebugMarkerDesc{.label = "marker"}).error().code ==
          ErrorCode::invalidArgument);
    CHECK(device.setViewport(foreignCb, Viewport{.width = 8.0F, .height = 8.0F}).error().code ==
          ErrorCode::invalidArgument);
    CHECK(device.setScissor(foreignCb, ScissorRect{.width = 8, .height = 8}).error().code ==
          ErrorCode::invalidArgument);
    CHECK(device.setBlendConstants(foreignCb, BlendConstants{}).error().code == ErrorCode::invalidArgument);
    CHECK(device.setDepthBias(foreignCb, DepthBias{}).error().code == ErrorCode::invalidArgument);
    CHECK(device.copyBuffer(foreignCb, BufferCopy{.size = 16}).error().code == ErrorCode::invalidArgument);
    CHECK(device.draw(foreignCb, 3).error().code == ErrorCode::invalidArgument);
    CHECK(device.dispatch(foreignCb, 1, 1, 1).error().code == ErrorCode::invalidArgument);

    auto zeroTokenCb = makeCommandBuffer(0);
    raw->token = 0;
    CHECK(device.draw(zeroTokenCb, 3).error().code == ErrorCode::invalidArgument);
    CHECK(raw->backendInvocations == 0);
}

TEST_CASE("Device frontend keeps a rendering scope open while a query that began inside it is active",
          "[size:small][rhi][core]")
{
    auto backend = std::make_unique<TrackingBackendDevice>();
    auto* raw = backend.get();
    auto device = rachis::detail::DeviceAccess::make(std::move(backend));
    auto cb = makeCommandBuffer(raw->token);
    auto* native = rachis::detail::CommandBufferAccess::get(cb);
    REQUIRE(native != nullptr);
    native->graphicsState().renderingOwner = std::make_shared<int>(1);
    auto& queries = native->queryState();

    const auto expectScopeKept = [&](const Status& status) {
        REQUIRE_FALSE(status.has_value());
        CHECK(status.error().code == ErrorCode::invalidState);
        CHECK(native->graphicsState().renderingOwner != nullptr);
        CHECK(raw->backendInvocations == 0);
    };

    queries.activeOcclusion =
        rachis::detail::ActiveQuery{.queryPool = {}, .queryIndex = 0, .owner = {}, .begunInRendering = true};
    expectScopeKept(device.endRendering(cb));
    queries.activeOcclusion.reset();

    queries.activePipelineStatistics =
        rachis::detail::ActiveQuery{.queryPool = {}, .queryIndex = 0, .owner = {}, .begunInRendering = true};
    expectScopeKept(device.endRendering(cb));
    queries.activePipelineStatistics.reset();

    queries.activeStreamOutputStatistics[rachis::maxStreamOutputStreams - 1] =
        rachis::detail::ActiveQuery{.queryPool = {}, .queryIndex = 0, .owner = {}, .begunInRendering = true};
    expectScopeKept(device.endRendering(cb));
    queries.activeStreamOutputStatistics[rachis::maxStreamOutputStreams - 1].reset();

    queries.activePipelineStatistics =
        rachis::detail::ActiveQuery{.queryPool = {}, .queryIndex = 0, .owner = {}, .begunInRendering = false};
    queries.activeStreamOutputStatistics[0] =
        rachis::detail::ActiveQuery{.queryPool = {}, .queryIndex = 0, .owner = {}, .begunInRendering = false};
    REQUIRE(device.endRendering(cb).has_value());
    CHECK(raw->backendInvocations == 1);
    CHECK(native->graphicsState().renderingOwner == nullptr);
    CHECK(queries.activePipelineStatistics.has_value());
    CHECK(queries.activeStreamOutputStatistics[0].has_value());
    queries.reset();
}

TEST_CASE("Device frontend allows only draws, dispatches, and state commands inside conditional execution",
          "[size:small][rhi][core][predicate]")
{
    auto backend = std::make_unique<TrackingBackendDevice>();
    auto* raw = backend.get();
    auto device = rachis::detail::DeviceAccess::make(std::move(backend));
    auto cb = makeCommandBuffer(raw->token);
    auto* native = rachis::detail::CommandBufferAccess::get(cb);
    REQUIRE(native != nullptr);
    native->queryState().activeConditionalExecution =
        rachis::detail::ActiveConditionalExecution{.owner = std::make_shared<int>(1)};

    const auto expectRejected = [&](std::string_view operation, const Status& status) {
        INFO("Operation: " << operation);
        REQUIRE_FALSE(status.has_value());
        INFO("Error: " << status.error().message);
        CHECK(status.error().code == ErrorCode::invalidState);
        CHECK(raw->backendInvocations == 0);
    };

    expectRejected("copyBuffer", device.copyBuffer(cb, BufferCopy{.size = 16}));
    expectRejected("copyBufferToTexture", device.copyBufferToTexture(cb, rachis::BufferTextureCopy{}));
    expectRejected("copyTextureToBuffer", device.copyTextureToBuffer(cb, rachis::BufferTextureCopy{}));
    expectRejected("copyTexture", device.copyTexture(cb, rachis::TextureCopy{}));
    expectRejected("resolveTexture", device.resolveTexture(cb, TextureResolve{}));
    expectRejected("generateMipmaps", device.generateMipmaps(cb, TextureMipmapGeneration{}));
    expectRejected("beginRendering", device.beginRendering(cb, RenderingDesc{}));
    expectRejected("dispatchRays", device.dispatchRays(cb, rachis::RayTracingDispatchDesc{}));
    expectRejected("dispatchRaysIndirect", device.dispatchRaysIndirect(cb, rachis::RayTracingIndirectDispatchDesc{}));
    expectRejected(
        "buildBottomLevelAccelerationStructure",
        device.buildBottomLevelAccelerationStructure(cb, rachis::BottomLevelAccelerationStructureBuildDesc{}));
    expectRejected("buildTopLevelAccelerationStructure",
                   device.buildTopLevelAccelerationStructure(cb, rachis::TopLevelAccelerationStructureBuildDesc{}));
    expectRejected(
        "updateBottomLevelAccelerationStructure",
        device.updateBottomLevelAccelerationStructure(cb, rachis::BottomLevelAccelerationStructureUpdateDesc{}));
    expectRejected("updateTopLevelAccelerationStructure",
                   device.updateTopLevelAccelerationStructure(cb, rachis::TopLevelAccelerationStructureUpdateDesc{}));
    expectRejected("buildTopLevelAccelerationStructureFromBuffer",
                   device.buildTopLevelAccelerationStructureFromBuffer(
                       cb, rachis::TopLevelAccelerationStructureBufferBuildDesc{}));
    expectRejected("updateTopLevelAccelerationStructureFromBuffer",
                   device.updateTopLevelAccelerationStructureFromBuffer(
                       cb, rachis::TopLevelAccelerationStructureBufferUpdateDesc{}));
    expectRejected("updateTopLevelAccelerationStructureInPlaceFromBuffer",
                   device.updateTopLevelAccelerationStructureInPlaceFromBuffer(
                       cb, rachis::TopLevelAccelerationStructureBufferInPlaceUpdateDesc{}));
    expectRejected("fillBuffer", device.fillBuffer(cb, {}, 0, 16, 0));
    expectRejected("clearStorageTexture",
                   device.clearStorageTexture(
                       cb, {}, rachis::StorageTextureClearValue{.type = rachis::StorageTextureClearType::uint32}));
    expectRejected("compactAccelerationStructure",
                   device.compactAccelerationStructure(cb, rachis::AccelerationStructureCompactionDesc{}));
    expectRejected("serializeAccelerationStructure",
                   device.serializeAccelerationStructure(cb, rachis::AccelerationStructureSerializationDesc{}));
    expectRejected("deserializeAccelerationStructure",
                   device.deserializeAccelerationStructure(cb, rachis::AccelerationStructureDeserializationDesc{}));
    expectRejected("resetSamplerFeedbackMap", device.resetSamplerFeedbackMap(cb, rachis::SamplerFeedbackMapHandle{}));
    expectRejected("decodeMinMipFeedback", device.decodeMinMipFeedback(cb, rachis::MinMipFeedbackDecode{}));
    expectRejected("resolveTimestamps", device.resolveTimestamps(cb, rachis::TimestampQueryResolveDesc{}));
    expectRejected("resolveOcclusionQueries", device.resolveOcclusionQueries(cb, rachis::OcclusionQueryResolveDesc{}));
    expectRejected("resolvePipelineStatisticsQueries",
                   device.resolvePipelineStatisticsQueries(cb, rachis::PipelineStatisticsQueryResolveDesc{}));

    native->graphicsState().renderingOwner = std::make_shared<int>(2);
    expectRejected("clearColorAttachment", device.clearColorAttachment(cb, 0, ClearColor{}));
    expectRejected("clearDepthStencilAttachment",
                   device.clearDepthStencilAttachment(cb, DepthStencilClearAspect::depth, 1.0F, 0));
    expectRejected("endRendering", device.endRendering(cb));
    CHECK(native->graphicsState().renderingOwner != nullptr);
    CHECK(native->queryState().activeConditionalExecution.has_value());

    REQUIRE(device.draw(cb, 3).has_value());
    REQUIRE(device.drawIndirect(cb, IndirectDrawDesc{}).has_value());
    CHECK(raw->backendInvocations == 2);
    native->graphicsState().renderingOwner.reset();
    REQUIRE(device.bufferBarrier(cb, BufferBarrier{}).has_value());
    REQUIRE(device.dispatch(cb, 1, 1, 1).has_value());
    REQUIRE(device.dispatchIndirect(cb, IndirectDispatchDesc{}).has_value());
    CHECK(raw->backendInvocations == 5);

    native->queryState().activeConditionalExecution.reset();
    REQUIRE(device.copyBuffer(cb, BufferCopy{.size = 16}).has_value());
    REQUIRE(device.beginRendering(cb, RenderingDesc{}).has_value());
    CHECK(raw->backendInvocations == 7);
}

TEST_CASE("DeviceOperationalState::lost takes precedence over frontend invalidArgument, invalidState, and unsupported",
          "[size:small][rhi][core]")
{
    auto backend = std::make_unique<TrackingBackendDevice>();
    auto* raw = backend.get();
    raw->state = DeviceOperationalState::lost;
    auto device = rachis::detail::DeviceAccess::make(std::move(backend));

    auto ownedCb = makeCommandBuffer(raw->token);
    rachis::detail::CommandBufferAccess::get(ownedCb)->graphicsState().renderingOwner = std::make_shared<int>(1);
    auto foreignCb = makeCommandBuffer(raw->token + 99);

    const auto expectLost = [&](const Error& err) {
        CHECK(err.code == ErrorCode::deviceLost);
        CHECK(err.nativeCode == 0xDEAD);
        CHECK(err.message == "Simulated GPU fault.");
    };

    expectLost(device.submit(std::span<CommandBuffer>{}).error());
    expectLost(device.submit(std::move(foreignCb)).error());
    CHECK(foreignCb.isValid());
    expectLost(device.submit(std::move(ownedCb)).error());
    CHECK(ownedCb.isValid());

    expectLost(device.resetCommandBuffer(ownedCb).error());
    expectLost(device.resetCommandBuffer(foreignCb).error());
    expectLost(device.endRendering(foreignCb).error());
    expectLost(device.beginDebugGroup(ownedCb, DebugMarkerDesc{.label = ""}).error());
    expectLost(device.setViewports(ownedCb, {}).error());
    expectLost(device.setScissors(ownedCb, {}).error());
    expectLost(
        device.setBlendConstants(ownedCb, BlendConstants{.red = std::numeric_limits<float>::quiet_NaN()}).error());
    expectLost(device.setComputePushConstants(ownedCb, 0, {}).error());
    expectLost(device.setGraphicsPushConstants(ownedCb, ShaderStage::none, 0, {}).error());
    expectLost(device.copyBuffer(ownedCb, {}, {}, {}).error());
    expectLost(device
                   .createBindingLayout(BindingLayoutDesc{
                       .entries = std::array{BindingLayoutEntry{.count = 0}},
                   })
                   .error());
    expectLost(device.createBindingPool(BindingPoolDesc{}).error());
    expectLost(device.createResourceHeap(ResourceHeapDesc{}).error());

    CHECK(raw->backendInvocations == 0);
}

TEST_CASE("Device frontend validates TLAS instance-buffer ranges and references before backend calls",
          "[size:small][rhi][core]")
{
    auto backend = std::make_unique<TrackingBackendDevice>();
    auto* raw = backend.get();
    auto device = rachis::detail::DeviceAccess::make(std::move(backend));
    auto cb = makeCommandBuffer(raw->token, QueueType::compute);
    const BufferHandle records{.identity = 1};
    const rachis::AccelerationStructureHandle source{.identity = 2};
    const rachis::AccelerationStructureHandle destination{.identity = 3};
    const rachis::TopLevelAccelerationStructureBufferBuildDesc build{
        .destination = destination,
        .instanceBuffer = records,
        .instanceCount = 2,
    };

    const auto unsupportedBuild = device.buildTopLevelAccelerationStructureFromBuffer(cb, build);
    REQUIRE_FALSE(unsupportedBuild.has_value());
    CHECK(unsupportedBuild.error().code == ErrorCode::unsupported);
    const auto unsupportedLayout = device.queryAccelerationStructureInstanceLayout();
    REQUIRE_FALSE(unsupportedLayout.has_value());
    CHECK(unsupportedLayout.error().code == ErrorCode::unsupported);
    CHECK(raw->backendInvocations == 0);

    raw->instanceRecords = true;
    const auto expectRejected = [&](const Status& status) {
        REQUIRE_FALSE(status.has_value());
        CHECK(status.error().code == ErrorCode::invalidArgument);
        CHECK(raw->backendInvocations == 0);
    };
    auto withoutBuffer = build;
    withoutBuffer.instanceBuffer = {};
    expectRejected(device.buildTopLevelAccelerationStructureFromBuffer(cb, withoutBuffer));
    auto withoutInstances = build;
    withoutInstances.instanceCount = 0;
    expectRejected(device.buildTopLevelAccelerationStructureFromBuffer(cb, withoutInstances));
    auto misaligned = build;
    misaligned.instanceOffset = 8;
    expectRejected(device.buildTopLevelAccelerationStructureFromBuffer(cb, misaligned));
    auto overflowing = build;
    overflowing.instanceOffset = std::numeric_limits<std::uint64_t>::max() - 63U;
    expectRejected(device.buildTopLevelAccelerationStructureFromBuffer(cb, overflowing));
    expectRejected(device.updateTopLevelAccelerationStructureFromBuffer(cb,
                                                                        {
                                                                            .source = source,
                                                                            .destination = source,
                                                                            .instanceBuffer = records,
                                                                            .instanceCount = 2,
                                                                        }));
    expectRejected(device.updateTopLevelAccelerationStructureFromBuffer(cb,
                                                                        {
                                                                            .source = source,
                                                                            .destination = destination,
                                                                            .instanceBuffer = records,
                                                                            .instanceOffset = 24,
                                                                            .instanceCount = 2,
                                                                        }));
    expectRejected(device.updateTopLevelAccelerationStructureInPlaceFromBuffer(cb,
                                                                               {
                                                                                   .target = source,
                                                                                   .instanceBuffer = records,
                                                                                   .instanceCount = 0,
                                                                               }));

    auto foreignCb = makeCommandBuffer(raw->token + 1, QueueType::compute);
    expectRejected(device.buildTopLevelAccelerationStructureFromBuffer(foreignCb, build));

    const auto nullReference = device.queryAccelerationStructureReference({});
    REQUIRE_FALSE(nullReference.has_value());
    CHECK(nullReference.error().code == ErrorCode::invalidArgument);
    CHECK(raw->referenceQueries == 0);
    const auto reference = device.queryAccelerationStructureReference(source);
    REQUIRE(reference.has_value());
    CHECK(*reference == 0x1000);
    CHECK(raw->referenceQueries == 1);

    auto aligned = build;
    aligned.instanceOffset = 48;
    REQUIRE(device.buildTopLevelAccelerationStructureFromBuffer(cb, aligned).has_value());
    REQUIRE(device
                .updateTopLevelAccelerationStructureFromBuffer(cb,
                                                               {
                                                                   .source = source,
                                                                   .destination = destination,
                                                                   .instanceBuffer = records,
                                                                   .instanceCount = 2,
                                                               })
                .has_value());
    REQUIRE(device
                .updateTopLevelAccelerationStructureInPlaceFromBuffer(cb,
                                                                      {
                                                                          .target = source,
                                                                          .instanceBuffer = records,
                                                                          .instanceCount = 2,
                                                                      })
                .has_value());
    CHECK(raw->backendInvocations == 3);

    raw->state = DeviceOperationalState::lost;
    const auto lostBuild = device.buildTopLevelAccelerationStructureFromBuffer(cb, misaligned);
    REQUIRE_FALSE(lostBuild.has_value());
    CHECK(lostBuild.error().code == ErrorCode::deviceLost);
    CHECK(raw->backendInvocations == 3);
}

TEST_CASE("Device frontend validates debug markers, debug group stack, thread affinity, and D3D12 disabled markers",
          "[size:small][rhi][core]")
{
    auto backend = std::make_unique<TrackingBackendDevice>();
    auto* raw = backend.get();
    auto device = rachis::detail::DeviceAccess::make(std::move(backend));
    auto cb = makeCommandBuffer(raw->token);

    const std::string oversizedLabel(rachis::maxDebugLabelBytes + 1U, 'a');
    const char embeddedNulChars[] = {'a', '\0', 'b'};
    const std::string_view embeddedNulLabel(embeddedNulChars, sizeof(embeddedNulChars));

    CHECK(device.insertDebugMarker(cb, DebugMarkerDesc{.label = ""}).error().code == ErrorCode::invalidArgument);
    CHECK(device.insertDebugMarker(cb, DebugMarkerDesc{.label = oversizedLabel}).error().code ==
          ErrorCode::invalidArgument);
    CHECK(device.insertDebugMarker(cb, DebugMarkerDesc{.label = embeddedNulLabel}).error().code ==
          ErrorCode::invalidArgument);
    CHECK(device
              .insertDebugMarker(cb,
                                 DebugMarkerDesc{
                                     .label = "nan-color",
                                     .color = DebugColor{.red = std::numeric_limits<float>::quiet_NaN()},
                                 })
              .error()
              .code == ErrorCode::invalidArgument);
    CHECK(device
              .insertDebugMarker(cb,
                                 DebugMarkerDesc{
                                     .label = "out-of-range-color",
                                     .color = DebugColor{.alpha = 1.01F},
                                 })
              .error()
              .code == ErrorCode::invalidArgument);
    CHECK(device.endDebugGroup(cb).error().code == ErrorCode::invalidState);
    CHECK(raw->backendInvocations == 0);

    for (std::uint32_t depth = 0; depth < rachis::maxDebugGroupDepth; ++depth)
    {
        REQUIRE(device.beginDebugGroup(cb, DebugMarkerDesc{.label = "nested"}).has_value());
    }
    const auto invocationsAfterOpen = raw->backendInvocations;
    CHECK(device.beginDebugGroup(cb, DebugMarkerDesc{.label = "overflow"}).error().code == ErrorCode::invalidState);
    CHECK(raw->backendInvocations == invocationsAfterOpen);

    for (std::uint32_t depth = 1; depth < rachis::maxDebugGroupDepth; ++depth)
    {
        REQUIRE(device.endDebugGroup(cb).has_value());
    }
    const auto invocationsBeforeCrossThread = raw->backendInvocations;
    ErrorCode crossThreadBeginCode = ErrorCode::unsupported;
    ErrorCode crossThreadEndCode = ErrorCode::unsupported;
    std::thread worker([&] {
        crossThreadBeginCode = device.beginDebugGroup(cb, DebugMarkerDesc{.label = "worker"}).error().code;
        crossThreadEndCode = device.endDebugGroup(cb).error().code;
    });
    worker.join();
    CHECK(crossThreadBeginCode == ErrorCode::invalidState);
    CHECK(crossThreadEndCode == ErrorCode::invalidState);
    CHECK(raw->backendInvocations == invocationsBeforeCrossThread);

    auto* nativeCb = rachis::detail::CommandBufferAccess::get(cb);
    nativeCb->debugState().groupOwners[0] = DebugGroupOwner::gpuProfile;
    CHECK(device.endDebugGroup(cb).error().code == ErrorCode::invalidState);
    nativeCb->debugState().groupOwners[0] = DebugGroupOwner::user;
    REQUIRE(device.endDebugGroup(cb).has_value());

    raw->name = "D3D12";
    raw->caps.debugMarkers = false;
    CHECK(device.beginDebugGroup(cb, DebugMarkerDesc{.label = ""}).error().code == ErrorCode::unsupported);
    CHECK(device.endDebugGroup(cb).error().code == ErrorCode::unsupported);
    CHECK(device.insertDebugMarker(cb, DebugMarkerDesc{.label = ""}).error().code == ErrorCode::unsupported);
}

TEST_CASE("Device frontend reads command-buffer scope state only while holding the recording-state lock",
          "[size:small][rhi][core]")
{
    const std::array<std::byte, 4> fourBytes{};
    const auto openRendering = [](BackendCommandBuffer& native) {
        native.graphicsState().renderingOwner = std::make_shared<int>(1);
    };
    const auto closeRendering = [](BackendCommandBuffer& native, std::thread::id) {
        native.graphicsState().renderingOwner.reset();
    };
    const auto noRendering = [](BackendCommandBuffer&) {};
    const auto beginRendering = [](BackendCommandBuffer& native, std::thread::id) {
        native.graphicsState().renderingOwner = std::make_shared<int>(1);
    };

    checkStateReadUnderRecordingLock("resetCommandBuffer", openRendering, closeRendering, [](auto& device, auto& cb) {
        return device.resetCommandBuffer(cb);
    });
    checkStateReadUnderRecordingLock(
        "submit",
        [](BackendCommandBuffer& native) { native.debugState().groupDepth = 1; },
        [](BackendCommandBuffer& native, std::thread::id) { native.debugState().reset(); },
        [](auto& device, auto& cb) -> Status {
            auto submitted = device.submit(std::move(cb));
            if (!submitted)
            {
                return std::unexpected(std::move(submitted.error()));
            }
            return {};
        });
    checkStateReadUnderRecordingLock(
        "beginDebugGroup",
        [](BackendCommandBuffer& native) {
            native.debugState().groupDepth = rachis::maxDebugGroupDepth;
            native.debugState().groupThread = std::this_thread::get_id();
        },
        [](BackendCommandBuffer& native, std::thread::id) { native.debugState().reset(); },
        [](auto& device, auto& cb) { return device.beginDebugGroup(cb, DebugMarkerDesc{.label = "locked"}); });
    checkStateReadUnderRecordingLock(
        "endDebugGroup",
        [](BackendCommandBuffer&) {},
        [](BackendCommandBuffer& native, std::thread::id requester) {
            native.debugState().groupDepth = 1;
            native.debugState().groupThread = requester;
            native.debugState().groupOwners[0] = DebugGroupOwner::user;
        },
        [](auto& device, auto& cb) { return device.endDebugGroup(cb); });
    checkStateReadUnderRecordingLock("copyBuffer", openRendering, closeRendering, [](auto& device, auto& cb) {
        return device.copyBuffer(cb, BufferCopy{.size = 16});
    });
    checkStateReadUnderRecordingLock("copyBufferToTexture", openRendering, closeRendering, [](auto& device, auto& cb) {
        const std::array regions{BufferTextureCopyRegion{}};
        return device.copyBufferToTexture(cb, BufferHandle{}, TextureHandle{}, regions);
    });
    checkStateReadUnderRecordingLock("copyTextureToBuffer", openRendering, closeRendering, [](auto& device, auto& cb) {
        const std::array regions{BufferTextureCopyRegion{}};
        return device.copyTextureToBuffer(cb, TextureHandle{}, BufferHandle{}, regions);
    });
    checkStateReadUnderRecordingLock("copyTexture", openRendering, closeRendering, [](auto& device, auto& cb) {
        const std::array regions{TextureCopyRegion{}};
        return device.copyTexture(cb, TextureHandle{}, TextureHandle{}, regions);
    });
    checkStateReadUnderRecordingLock(
        "setComputePushConstants",
        noRendering,
        [](BackendCommandBuffer& native, std::thread::id) {
            native.computeState().pipelineOwner = std::make_shared<int>(2);
        },
        [&](auto& device, auto& cb) { return device.setComputePushConstants(cb, 0, fourBytes); });
    checkStateReadUnderRecordingLock(
        "endRendering",
        [](BackendCommandBuffer& native) {
            native.graphicsState().renderingOwner = std::make_shared<int>(1);
            native.queryState().activeOcclusion = rachis::detail::ActiveQuery{};
        },
        [](BackendCommandBuffer& native, std::thread::id) { native.queryState().activeOcclusion.reset(); },
        [](auto& device, auto& cb) { return device.endRendering(cb); });
    checkStateReadUnderRecordingLock(
        "setGraphicsPushConstants",
        noRendering,
        [](BackendCommandBuffer& native, std::thread::id) {
            native.graphicsState().renderingOwner = std::make_shared<int>(1);
            native.graphicsState().pipelineOwner = std::make_shared<int>(2);
        },
        [&](auto& device, auto& cb) { return device.setGraphicsPushConstants(cb, ShaderStage::vertex, 0, fourBytes); });
    checkStateReadUnderRecordingLock("setViewports", noRendering, beginRendering, [](auto& device, auto& cb) {
        return device.setViewport(cb, Viewport{.width = 8.0F, .height = 8.0F});
    });
    checkStateReadUnderRecordingLock("setScissors", noRendering, beginRendering, [](auto& device, auto& cb) {
        return device.setScissor(cb, ScissorRect{.width = 8, .height = 8});
    });
    checkStateReadUnderRecordingLock("setBlendConstants", noRendering, beginRendering, [](auto& device, auto& cb) {
        return device.setBlendConstants(cb, BlendConstants{});
    });
}

TEST_CASE("Device frontend validates viewports, scissors, blend constants, push constants, and copy regions",
          "[size:small][rhi][core]")
{
    auto backend = std::make_unique<TrackingBackendDevice>();
    auto* raw = backend.get();
    raw->caps.limits.maxViewports = 2;
    auto device = rachis::detail::DeviceAccess::make(std::move(backend));
    auto cb = makeCommandBuffer(raw->token);
    auto* nativeCb = rachis::detail::CommandBufferAccess::get(cb);
    REQUIRE(nativeCb != nullptr);

    const std::array<Viewport, 3> threeDefaultViewports{};
    const std::array<ScissorRect, 3> threeScissors{
        ScissorRect{.width = 4, .height = 4},
        ScissorRect{.width = 4, .height = 4},
        ScissorRect{.width = 4, .height = 4},
    };
    const std::array<std::byte, 4> fourBytes{};
    const std::array<std::byte, 3> threeBytes{};

    CHECK(device.setViewports(cb, {}).error().code == ErrorCode::invalidState);
    CHECK(device.setViewports(cb, threeDefaultViewports).error().code == ErrorCode::invalidState);
    CHECK(device.setScissors(cb, {}).error().code == ErrorCode::invalidState);
    CHECK(device.setScissors(cb, threeScissors).error().code == ErrorCode::invalidState);
    CHECK(device.setBlendConstants(cb, BlendConstants{.alpha = std::numeric_limits<float>::quiet_NaN()}).error().code ==
          ErrorCode::invalidState);
    CHECK(device.setComputePushConstants(cb, 0, {}).error().code == ErrorCode::invalidState);
    CHECK(device.setGraphicsPushConstants(cb, ShaderStage::none, 0, {}).error().code == ErrorCode::invalidState);

    nativeCb->graphicsState().renderingOwner = std::make_shared<int>(1);
    CHECK(device.setGraphicsPushConstants(cb, ShaderStage::none, 0, {}).error().code == ErrorCode::invalidState);

    CHECK(device.copyBuffer(cb, {}, {}, {}).error().code == ErrorCode::invalidState);
    CHECK(device.copyBufferToTexture(cb, {}, {}, {}).error().code == ErrorCode::invalidState);
    CHECK(device.copyTextureToBuffer(cb, {}, {}, {}).error().code == ErrorCode::invalidState);
    CHECK(device.copyTexture(cb, {}, {}, {}).error().code == ErrorCode::invalidState);

    nativeCb->graphicsState().pipelineOwner = std::make_shared<int>(2);
    nativeCb->computeState().pipelineOwner = std::make_shared<int>(3);

    CHECK(device.setViewports(cb, {}).error().code == ErrorCode::invalidArgument);
    CHECK(device.setViewports(cb, threeDefaultViewports).error().code == ErrorCode::unsupported);
    raw->name = "Metal";
    CHECK(device.setViewports(cb, threeDefaultViewports).error().code == ErrorCode::invalidArgument);
    raw->name = "Tracking";
    CHECK(device.setViewport(cb, Viewport{.width = 0.0F, .height = 8.0F}).error().code == ErrorCode::invalidArgument);
    CHECK(device.setViewport(cb, Viewport{.width = 8.0F, .height = -1.0F}).error().code == ErrorCode::invalidArgument);
    CHECK(device.setViewport(cb, Viewport{.x = std::numeric_limits<float>::infinity(), .width = 8.0F, .height = 8.0F})
              .error()
              .code == ErrorCode::invalidArgument);
    CHECK(device.setViewport(cb, Viewport{.width = 8.0F, .height = 8.0F, .minDepth = -0.1F, .maxDepth = 1.0F})
              .error()
              .code == ErrorCode::invalidArgument);
    CHECK(device.setViewport(cb, Viewport{.width = 8.0F, .height = 8.0F, .minDepth = 0.8F, .maxDepth = 0.2F})
              .error()
              .code == ErrorCode::invalidArgument);

    CHECK(device.setScissors(cb, {}).error().code == ErrorCode::invalidArgument);
    CHECK(device.setScissors(cb, threeScissors).error().code == ErrorCode::unsupported);
    raw->name = "Metal";
    CHECK(device.setScissors(cb, threeScissors).error().code == ErrorCode::invalidArgument);
    raw->name = "Tracking";
    CHECK(device.setScissor(cb, ScissorRect{.x = -1, .width = 4, .height = 4}).error().code ==
          ErrorCode::invalidArgument);
    CHECK(device.setScissor(cb, ScissorRect{.width = 0, .height = 4}).error().code == ErrorCode::invalidArgument);
    CHECK(device.setScissor(cb, ScissorRect{.x = std::numeric_limits<std::int32_t>::max(), .width = 1, .height = 4})
              .error()
              .code == ErrorCode::invalidArgument);
    CHECK(device
              .setScissor(cb,
                          ScissorRect{.y = 1,
                                      .width = 4,
                                      .height = static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())})
              .error()
              .code == ErrorCode::invalidArgument);

    CHECK(device.setBlendConstants(cb, BlendConstants{.alpha = std::numeric_limits<float>::quiet_NaN()}).error().code ==
          ErrorCode::invalidArgument);

    CHECK(device.setComputePushConstants(cb, 0, {}).error().code == ErrorCode::invalidArgument);
    CHECK(device.setComputePushConstants(cb, 2, fourBytes).error().code == ErrorCode::invalidArgument);
    CHECK(device.setComputePushConstants(cb, 0, threeBytes).error().code == ErrorCode::invalidArgument);
    CHECK(device.setComputePushConstants(cb, rachis::maxPushConstantBytes, fourBytes).error().code ==
          ErrorCode::invalidArgument);
    CHECK(device.setGraphicsPushConstants(cb, ShaderStage::none, 0, fourBytes).error().code ==
          ErrorCode::invalidArgument);
    CHECK(device.setGraphicsPushConstants(cb, ShaderStage::compute, 0, fourBytes).error().code ==
          ErrorCode::invalidArgument);
    CHECK(device.setGraphicsPushConstants(cb, ShaderStage::vertex | ShaderStage::rayGeneration, 0, fourBytes)
              .error()
              .code == ErrorCode::invalidArgument);

    nativeCb->graphicsState().renderingOwner.reset();
    nativeCb->graphicsState().pipelineOwner.reset();
    nativeCb->computeState().pipelineOwner.reset();

    CHECK(device.copyBuffer(cb, {}, {}, {}).error().code == ErrorCode::invalidArgument);
    CHECK(device.copyBuffer(cb, BufferCopy{.size = 0}).error().code == ErrorCode::invalidArgument);
    CHECK(device
              .copyBuffer(cb,
                          BufferCopy{
                              .sourceOffset = std::numeric_limits<std::uint64_t>::max() - 3U,
                              .size = 4U,
                          })
              .error()
              .code == ErrorCode::invalidArgument);
    CHECK(device
              .copyBuffer(cb,
                          BufferCopy{
                              .destinationOffset = std::numeric_limits<std::uint64_t>::max(),
                              .size = 1U,
                          })
              .error()
              .code == ErrorCode::invalidArgument);
    CHECK(device.copyBufferToTexture(cb, {}, {}, {}).error().code == ErrorCode::invalidArgument);
    CHECK(device.copyTextureToBuffer(cb, {}, {}, {}).error().code == ErrorCode::invalidArgument);
    CHECK(device.copyTexture(cb, {}, {}, {}).error().code == ErrorCode::invalidArgument);

    rachis::detail::CommandBufferAccess::get(cb)->queryState().activeOcclusion = rachis::detail::ActiveQuery{};
    CHECK(device.resetCommandBuffer(cb).error().code == ErrorCode::invalidState);
    rachis::detail::CommandBufferAccess::get(cb)->queryState().reset();

    CHECK(raw->backendInvocations == 0);
}

TEST_CASE("Device frontend restricts buffer fills and storage-texture clears to compute-capable queues outside "
          "rendering and validates ranges and clear values",
          "[size:small][rhi][core][fill][clear]")
{
    using rachis::StorageTextureClearType;
    using rachis::StorageTextureClearValue;
    auto backend = std::make_unique<TrackingBackendDevice>();
    auto* raw = backend.get();
    auto device = rachis::detail::DeviceAccess::make(std::move(backend));
    const StorageTextureClearValue uintValue{.type = StorageTextureClearType::uint32, .uint32 = {1, 2, 3, 4}};

    auto copyCb = makeCommandBuffer(raw->token, QueueType::copy);
    CHECK(device.fillBuffer(copyCb, {}, 0, 16, 0).error().code == ErrorCode::invalidState);
    CHECK(device.clearStorageTexture(copyCb, {}, uintValue).error().code == ErrorCode::invalidState);

    auto cb = makeCommandBuffer(raw->token);
    auto* nativeCb = rachis::detail::CommandBufferAccess::get(cb);
    REQUIRE(nativeCb != nullptr);
    nativeCb->graphicsState().renderingOwner = std::make_shared<int>(1);
    CHECK(device.fillBuffer(cb, {}, 0, 16, 0).error().code == ErrorCode::invalidState);
    CHECK(device.clearStorageTexture(cb, {}, uintValue).error().code == ErrorCode::invalidState);
    nativeCb->graphicsState().renderingOwner.reset();

    CHECK(device.fillBuffer(cb, {}, 2, 16, 0).error().code == ErrorCode::invalidArgument);
    CHECK(device.fillBuffer(cb, {}, 0, 0, 0).error().code == ErrorCode::invalidArgument);
    CHECK(device.fillBuffer(cb, {}, 0, 6, 0).error().code == ErrorCode::invalidArgument);
    CHECK(device.fillBuffer(cb, {}, std::numeric_limits<std::uint64_t>::max() - 3U, 8, 0).error().code ==
          ErrorCode::invalidArgument);

    auto unknownType = uintValue;
    unknownType.type = static_cast<StorageTextureClearType>(3);
    CHECK(device.clearStorageTexture(cb, {}, unknownType).error().code == ErrorCode::invalidArgument);
    CHECK(raw->backendInvocations == 0);

    auto ignoredNaN = uintValue;
    ignoredNaN.float32[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(device.clearStorageTexture(cb, {}, ignoredNaN).has_value());
    const auto notFiniteGreen = StorageTextureClearValue{
        .type = StorageTextureClearType::float32,
        .float32 = {0.0F, std::numeric_limits<float>::infinity(), 0.0F, 0.0F},
    };
    CHECK(device.clearStorageTexture(cb, {}, notFiniteGreen).has_value());
    CHECK(device.fillBuffer(cb, {}, 4, rachis::wholeResource, 0xFFFFFFFFU).has_value());
    CHECK(device.fillBuffer(cb, {}, std::numeric_limits<std::uint64_t>::max() - 7U, 4, 0).has_value());
    auto computeCb = makeCommandBuffer(raw->token, QueueType::compute);
    CHECK(device.fillBuffer(computeCb, {}, 0, 4, 0).has_value());
    CHECK(device.clearStorageTexture(computeCb, {}, uintValue).has_value());
    CHECK(raw->backendInvocations == 6);

    raw->state = DeviceOperationalState::lost;
    CHECK(device.fillBuffer(copyCb, {}, 0, 16, 0).error().code == ErrorCode::deviceLost);
    CHECK(device.clearStorageTexture(cb, {}, unknownType).error().code == ErrorCode::deviceLost);
    CHECK(raw->backendInvocations == 6);
}

TEST_CASE("Shared fill and clear rules resolve buffer ranges, usages, formats, and integer clear ranges",
          "[size:small][rhi][core][fill][clear]")
{
    using rachis::BufferUsage;
    using rachis::StorageTextureClearType;
    using rachis::StorageTextureClearValue;
    using rachis::TextureUsage;
    using rachis::TextureViewUsage;
    const BufferDesc fillable{.size = 64, .usage = BufferUsage::storage | BufferUsage::copyDestination};
    CHECK(rachis::detail::resolveBufferFillRange(fillable, 0, rachis::wholeResource).value() == 64U);
    CHECK(rachis::detail::resolveBufferFillRange(fillable, 60, rachis::wholeResource).value() == 4U);
    CHECK(rachis::detail::resolveBufferFillRange(fillable, 8, 16).value() == 16U);
    CHECK(rachis::detail::resolveBufferFillRange(fillable, 64, rachis::wholeResource).error().code ==
          ErrorCode::invalidArgument);
    CHECK(rachis::detail::resolveBufferFillRange(fillable, 60, 8).error().code == ErrorCode::invalidArgument);
    CHECK(rachis::detail::resolveBufferFillRange(
              BufferDesc{.size = 66, .usage = fillable.usage}, 0, rachis::wholeResource)
              .error()
              .code == ErrorCode::invalidArgument);
    CHECK(rachis::detail::resolveBufferFillRange(BufferDesc{.size = 64, .usage = BufferUsage::storage}, 0, 64)
              .error()
              .code == ErrorCode::invalidArgument);
    CHECK(rachis::detail::resolveBufferFillRange(BufferDesc{.size = 64, .usage = BufferUsage::copyDestination}, 0, 64)
              .error()
              .code == ErrorCode::invalidArgument);

    const TextureDesc storage{
        .format = Format::rgba8Uint,
        .usage = TextureUsage::storage | TextureUsage::copyDestination,
    };
    const StorageTextureClearValue uintValue{.type = StorageTextureClearType::uint32, .uint32 = {255, 0, 7, 1}};
    const auto accepted =
        rachis::detail::validateStorageTextureClear(storage, TextureViewUsage::storage, Format::unknown, uintValue);
    REQUIRE(accepted.has_value());
    CHECK(accepted->type == StorageTextureClearType::uint32);
    CHECK(accepted->channelCount == 4U);
    CHECK(accepted->channelBits == 8U);
    CHECK(rachis::detail::validateStorageTextureClear(storage, TextureViewUsage::sampled, Format::unknown, uintValue)
              .error()
              .code == ErrorCode::invalidArgument);
    auto withoutCopyDestination = storage;
    withoutCopyDestination.usage = TextureUsage::storage;
    CHECK(rachis::detail::validateStorageTextureClear(
              withoutCopyDestination, TextureViewUsage::storage, Format::unknown, uintValue)
              .error()
              .code == ErrorCode::invalidArgument);
    auto multisampled = storage;
    multisampled.sampleCount = 4;
    CHECK(
        rachis::detail::validateStorageTextureClear(multisampled, TextureViewUsage::storage, Format::unknown, uintValue)
            .error()
            .code == ErrorCode::invalidArgument);
    auto tooWide = uintValue;
    tooWide.uint32[1] = 256;
    CHECK(rachis::detail::validateStorageTextureClear(storage, TextureViewUsage::storage, Format::unknown, tooWide)
              .error()
              .code == ErrorCode::invalidArgument);
    const StorageTextureClearValue floatValue{.type = StorageTextureClearType::float32, .float32 = {1.0F}};
    CHECK(rachis::detail::validateStorageTextureClear(storage, TextureViewUsage::storage, Format::unknown, floatValue)
              .error()
              .code == ErrorCode::invalidArgument);

    auto srgb = storage;
    srgb.format = Format::rgba8UnormSrgb;
    CHECK(rachis::detail::validateStorageTextureClear(srgb, TextureViewUsage::storage, Format::rgba8Unorm, floatValue)
              .error()
              .code == ErrorCode::unsupported);
    for (const auto format : {Format::d32Float, Format::bc7RgbaUnorm, Format::r64Uint})
    {
        auto unsupported = storage;
        unsupported.format = format;
        CHECK(rachis::detail::validateStorageTextureClear(
                  unsupported, TextureViewUsage::storage, Format::unknown, uintValue)
                  .error()
                  .code == ErrorCode::unsupported);
    }

    auto signedTexture = storage;
    signedTexture.format = Format::rg16Sint;
    StorageTextureClearValue sintValue{.type = StorageTextureClearType::sint32, .sint32 = {-32768, 32767, 99999, 0}};
    CHECK(rachis::detail::validateStorageTextureClear(
              signedTexture, TextureViewUsage::storage, Format::unknown, sintValue)
              .has_value());
    sintValue.sint32[0] = -32769;
    CHECK(rachis::detail::validateStorageTextureClear(
              signedTexture, TextureViewUsage::storage, Format::unknown, sintValue)
              .error()
              .code == ErrorCode::invalidArgument);
    signedTexture.format = Format::r32Sint;
    CHECK(rachis::detail::validateStorageTextureClear(
              signedTexture, TextureViewUsage::storage, Format::unknown, sintValue)
              .has_value());

    auto floatTexture = storage;
    floatTexture.format = Format::r32Float;
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    constexpr float infinity = std::numeric_limits<float>::infinity();
    const StorageTextureClearValue absentNaN{.type = StorageTextureClearType::float32,
                                             .float32 = {1.0F, nan, infinity, nan}};
    CHECK(
        rachis::detail::validateStorageTextureClear(floatTexture, TextureViewUsage::storage, Format::unknown, absentNaN)
            .has_value());
    const StorageTextureClearValue presentNaN{.type = StorageTextureClearType::float32, .float32 = {nan}};
    CHECK(rachis::detail::validateStorageTextureClear(
              floatTexture, TextureViewUsage::storage, Format::unknown, presentNaN)
              .error()
              .code == ErrorCode::invalidArgument);
    floatTexture.format = Format::rg16Float;
    CHECK(
        rachis::detail::validateStorageTextureClear(floatTexture, TextureViewUsage::storage, Format::unknown, absentNaN)
            .error()
            .code == ErrorCode::invalidArgument);
    auto normalizedTexture = storage;
    normalizedTexture.format = Format::rgba8Unorm;
    const StorageTextureClearValue alphaInfinity{.type = StorageTextureClearType::float32,
                                                 .float32 = {0.0F, 0.0F, 0.0F, -infinity}};
    CHECK(rachis::detail::validateStorageTextureClear(
              normalizedTexture, TextureViewUsage::storage, Format::unknown, alphaInfinity)
              .error()
              .code == ErrorCode::invalidArgument);
}

TEST_CASE("Device frontend validates binding layouts and pools while preserving capability-gated unsupported types",
          "[size:small][rhi][core]")
{
    auto backend = std::make_unique<TrackingBackendDevice>();
    auto* raw = backend.get();
    raw->caps.rayTracing = false;
    raw->caps.samplerFeedback = false;
    auto device = rachis::detail::DeviceAccess::make(std::move(backend));

    const auto checkInvalidLayout = [&](std::span<const BindingLayoutEntry> entries) {
        const auto res = device.createBindingLayout(BindingLayoutDesc{.entries = entries});
        REQUIRE_FALSE(res.has_value());
        CHECK(res.error().code == ErrorCode::invalidArgument);
    };

    checkInvalidLayout(std::array{BindingLayoutEntry{.count = 0}});
    checkInvalidLayout(std::array{BindingLayoutEntry{.visibility = ShaderStage::none}});
    checkInvalidLayout(std::array{BindingLayoutEntry{.visibility = static_cast<ShaderStage>(1U << 15U)}});
    checkInvalidLayout(std::array{BindingLayoutEntry{.flags = static_cast<BindingFlag>(1U << 7U)}});
    checkInvalidLayout(std::array{BindingLayoutEntry{.type = static_cast<BindingType>(99)}});
    checkInvalidLayout(
        std::array{BindingLayoutEntry{.binding = std::numeric_limits<std::uint32_t>::max(), .count = 2}});
    checkInvalidLayout(std::array{BindingLayoutEntry{.count = rachis::unboundedBindingCount}});
    checkInvalidLayout(std::array{BindingLayoutEntry{.count = 4, .flags = BindingFlag::variableDescriptorCount}});
    checkInvalidLayout(std::array{BindingLayoutEntry{
        .type = BindingType::dynamicUniformBuffer,
        .flags = BindingFlag::partiallyBound,
    }});
    checkInvalidLayout(std::array{
        BindingLayoutEntry{.binding = 2},
        BindingLayoutEntry{.binding = 2, .type = BindingType::sampler},
    });
    checkInvalidLayout(std::array{
        BindingLayoutEntry{
            .binding = 1,
            .count = rachis::unboundedBindingCount,
            .flags = BindingFlag::variableDescriptorCount,
        },
        BindingLayoutEntry{.binding = 2},
    });
    checkInvalidLayout(std::array{
        BindingLayoutEntry{
            .binding = 1,
            .count = rachis::unboundedBindingCount,
            .flags = BindingFlag::variableDescriptorCount,
        },
        BindingLayoutEntry{
            .binding = 2,
            .count = rachis::unboundedBindingCount,
            .flags = BindingFlag::variableDescriptorCount,
        },
    });

    CHECK(device.createBindingPool(BindingPoolDesc{}).error().code == ErrorCode::invalidArgument);
    std::array<BindingPoolSize, rachis::bindingTypeCount + 1> oversizedPool{};
    for (std::size_t index = 0; index < oversizedPool.size(); ++index)
    {
        oversizedPool[index] = BindingPoolSize{
            .type = static_cast<BindingType>(index % rachis::bindingTypeCount),
            .descriptorCount = 1,
        };
    }
    CHECK(device.createBindingPool(BindingPoolDesc{.maxBindingSets = 4, .sizes = oversizedPool}).error().code ==
          ErrorCode::invalidArgument);
    const std::array zeroCountPool{BindingPoolSize{.type = BindingType::uniformBuffer, .descriptorCount = 0}};
    CHECK(device.createBindingPool(BindingPoolDesc{.maxBindingSets = 4, .sizes = zeroCountPool}).error().code ==
          ErrorCode::invalidArgument);
    const std::array duplicatePool{
        BindingPoolSize{.type = BindingType::sampler, .descriptorCount = 4},
        BindingPoolSize{.type = BindingType::sampler, .descriptorCount = 8},
    };
    CHECK(device.createBindingPool(BindingPoolDesc{.maxBindingSets = 4, .sizes = duplicatePool}).error().code ==
          ErrorCode::invalidArgument);
    const std::array invalidTypePool{BindingPoolSize{.type = static_cast<BindingType>(99), .descriptorCount = 4}};
    CHECK(device.createBindingPool(BindingPoolDesc{.maxBindingSets = 4, .sizes = invalidTypePool}).error().code ==
          ErrorCode::invalidArgument);

    CHECK(raw->backendInvocations == 0);

    const auto descriptorFreePool = device.createBindingPool(BindingPoolDesc{.maxBindingSets = 4});
    REQUIRE(descriptorFreePool.has_value());
    CHECK(raw->backendInvocations == 1);

    const std::array accelEntries{BindingLayoutEntry{.binding = 0, .type = BindingType::accelerationStructure}};
    const auto accelLayout = device.createBindingLayout(BindingLayoutDesc{.entries = accelEntries});
    REQUIRE_FALSE(accelLayout.has_value());
    CHECK(accelLayout.error().code == ErrorCode::unsupported);

    const std::array feedbackEntries{BindingLayoutEntry{.binding = 0, .type = BindingType::minMipFeedback}};
    const auto feedbackLayout = device.createBindingLayout(BindingLayoutDesc{.entries = feedbackEntries});
    REQUIRE_FALSE(feedbackLayout.has_value());
    CHECK(feedbackLayout.error().code == ErrorCode::unsupported);

    CHECK(raw->backendInvocations == 3);

#if RACHIS_ENABLE_METAL
    if (auto metalDevice = rachis::createDevice(
            rachis::DeviceDesc{.backend = rachis::Backend::metal, .validation = rachis::ValidationLevel::standard});
        metalDevice.has_value())
    {
        const auto emptyLayout = metalDevice->createBindingLayout(BindingLayoutDesc{});
        REQUIRE(emptyLayout.has_value());
        const auto liveEmptyPool = metalDevice->createBindingPool(BindingPoolDesc{.maxBindingSets = 1});
        REQUIRE(liveEmptyPool.has_value());
        const auto emptySet = metalDevice->createBindingSet(rachis::BindingSetDesc{
            .layout = *emptyLayout,
            .pool = *liveEmptyPool,
        });
        REQUIRE(emptySet.has_value());
        REQUIRE(metalDevice->destroyBindingSet(*emptySet).has_value());
        REQUIRE(metalDevice->destroyBindingPool(*liveEmptyPool).has_value());
        REQUIRE(metalDevice->destroyBindingLayout(*emptyLayout).has_value());
    }
#endif
}

TEST_CASE("Device frontend admits update-after-bind entries only for array types without dynamic buffers",
          "[size:small][rhi][core]")
{
    auto backend = std::make_unique<TrackingBackendDevice>();
    auto* raw = backend.get();
    raw->caps.updateAfterBindBindingArrays = true;
    auto device = rachis::detail::DeviceAccess::make(std::move(backend));

    const auto updateAfterBindEntry = [](std::uint32_t binding, BindingType type) {
        return BindingLayoutEntry{
            .binding = binding,
            .type = type,
            .count = 16,
            .flags = BindingFlag::partiallyBound | BindingFlag::updateAfterBind,
        };
    };

    for (const auto type : {BindingType::uniformBuffer,
                            BindingType::sampler,
                            BindingType::dynamicUniformBuffer,
                            BindingType::dynamicReadOnlyStorageBuffer,
                            BindingType::dynamicStorageBuffer,
                            BindingType::accelerationStructure,
                            BindingType::minMipFeedback,
                            BindingType::mipRegionUsedFeedback})
    {
        CAPTURE(static_cast<int>(type));
        const std::array entries{updateAfterBindEntry(0, type)};
        const auto layout = device.createBindingLayout(BindingLayoutDesc{.entries = entries});
        REQUIRE_FALSE(layout.has_value());
        CHECK(layout.error().code == ErrorCode::invalidArgument);
    }

    const std::array withDynamicBuffer{
        BindingLayoutEntry{.binding = 0, .type = BindingType::dynamicUniformBuffer},
        updateAfterBindEntry(1, BindingType::sampledTexture),
    };
    const auto mixed = device.createBindingLayout(BindingLayoutDesc{.entries = withDynamicBuffer});
    REQUIRE_FALSE(mixed.has_value());
    CHECK(mixed.error().code == ErrorCode::invalidArgument);
    CHECK(raw->backendInvocations == 0);

    for (const auto type : {BindingType::sampledTexture,
                            BindingType::storageTexture,
                            BindingType::readOnlyStorageBuffer,
                            BindingType::storageBuffer})
    {
        CAPTURE(static_cast<int>(type));
        const std::array entries{
            BindingLayoutEntry{.binding = 0, .type = BindingType::uniformBuffer},
            updateAfterBindEntry(1, type),
        };
        CHECK(device.createBindingLayout(BindingLayoutDesc{.entries = entries}).has_value());
    }
    const std::array variable{BindingLayoutEntry{
        .binding = 0,
        .type = BindingType::sampledTexture,
        .count = rachis::unboundedBindingCount,
        .flags = BindingFlag::variableDescriptorCount | BindingFlag::updateAfterBind,
    }};
    CHECK(device.createBindingLayout(BindingLayoutDesc{.entries = variable}).has_value());
    CHECK(raw->backendInvocations == 5);

    raw->caps.updateAfterBindBindingArrays = false;
    const std::array supported{updateAfterBindEntry(0, BindingType::sampledTexture)};
    const auto unsupported = device.createBindingLayout(BindingLayoutDesc{.entries = supported});
    REQUIRE_FALSE(unsupported.has_value());
    CHECK(unsupported.error().code == ErrorCode::unsupported);
    const std::array invalidType{updateAfterBindEntry(0, BindingType::sampler)};
    const auto invalidBeforeUnsupported = device.createBindingLayout(BindingLayoutDesc{.entries = invalidType});
    REQUIRE_FALSE(invalidBeforeUnsupported.has_value());
    CHECK(invalidBeforeUnsupported.error().code == ErrorCode::invalidArgument);
    CHECK(raw->backendInvocations == 5);
}

TEST_CASE("Device frontend rejects vertex layouts that break format alignment before backend calls",
          "[size:small][rhi][core]")
{
    auto backend = std::make_unique<TrackingBackendDevice>();
    auto* raw = backend.get();
    auto device = rachis::detail::DeviceAccess::make(std::move(backend));

    const std::array alignedAttributes{VertexAttribute{.format = VertexFormat::float32x2, .shaderLocation = 7}};
    const auto checkRejected = [&](const VertexBufferLayout& layout) {
        const std::array layouts{VertexBufferLayout{.stride = 8, .attributes = alignedAttributes}, layout};
        const GraphicsPipelineDesc desc{.vertexBuffers = layouts};
        const auto pipeline = device.createGraphicsPipeline(desc);
        REQUIRE_FALSE(pipeline.has_value());
        CHECK(pipeline.error().code == ErrorCode::invalidArgument);
        const auto compilation = device.beginGraphicsPipelineCompilation(desc);
        REQUIRE_FALSE(compilation.has_value());
        CHECK(compilation.error().code == ErrorCode::invalidArgument);
    };

    const std::array float2AtTwo{VertexAttribute{.format = VertexFormat::float32x2, .offset = 2}};
    const std::array float2AtZero{VertexAttribute{.format = VertexFormat::float32x2}};
    const std::array unorm8x2AtOne{VertexAttribute{.format = VertexFormat::unorm8x2, .offset = 1}};
    const std::array uint16x2AtTwo{VertexAttribute{.format = VertexFormat::uint16x2, .offset = 2}};
    const std::array mixedPacked{
        VertexAttribute{.format = VertexFormat::unorm8x2, .offset = 0, .shaderLocation = 0},
        VertexAttribute{.format = VertexFormat::unorm8x4, .offset = 2, .shaderLocation = 1},
    };
    checkRejected(VertexBufferLayout{.stride = 12, .attributes = float2AtTwo});
    checkRejected(VertexBufferLayout{.stride = 10, .attributes = float2AtZero});
    checkRejected(VertexBufferLayout{.stride = 4, .attributes = unorm8x2AtOne});
    checkRejected(VertexBufferLayout{.stride = 8, .attributes = uint16x2AtTwo});
    checkRejected(VertexBufferLayout{.stride = 6, .attributes = mixedPacked});
    checkRejected(VertexBufferLayout{
        .stride = 6,
        .stepMode = VertexStepMode::instance,
        .instanceStepRate = 2,
        .attributes = mixedPacked,
    });
    CHECK(raw->backendInvocations == 0);

    const std::array float2AtFour{VertexAttribute{.format = VertexFormat::float32x2, .offset = 4}};
    const std::array unorm8x2Packed{
        VertexAttribute{.format = VertexFormat::unorm8x2, .offset = 0, .shaderLocation = 0},
        VertexAttribute{.format = VertexFormat::unorm8x2, .offset = 2, .shaderLocation = 1},
        VertexAttribute{.format = VertexFormat::unorm8x2, .offset = 4, .shaderLocation = 2},
    };
    const std::array unorm8x2Single{VertexAttribute{.format = VertexFormat::unorm8x2}};
    const std::array acceptedLayouts{
        VertexBufferLayout{.stride = 12, .attributes = float2AtFour},
        VertexBufferLayout{.stride = 6, .attributes = unorm8x2Packed},
        VertexBufferLayout{.stride = 2, .attributes = unorm8x2Single},
    };
    for (const auto& layout : acceptedLayouts)
    {
        const std::array layouts{layout};
        const GraphicsPipelineDesc desc{.vertexBuffers = layouts};
        CHECK(device.createGraphicsPipeline(desc).has_value());
        CHECK(device.beginGraphicsPipelineCompilation(desc).has_value());
    }
    CHECK(raw->backendInvocations == 2 * acceptedLayouts.size());

    raw->state = DeviceOperationalState::lost;
    const std::array lostLayouts{VertexBufferLayout{.stride = 10, .attributes = float2AtZero}};
    const auto lost = device.createGraphicsPipeline(GraphicsPipelineDesc{.vertexBuffers = lostLayouts});
    REQUIRE_FALSE(lost.has_value());
    CHECK(lost.error().code == ErrorCode::deviceLost);
    CHECK(raw->backendInvocations == 2 * acceptedLayouts.size());
}

TEST_CASE("Device frontend gates the draw index on shaderDrawIndex and on stages that can read it",
          "[size:small][rhi][core][draw-index]")
{
    auto backend = std::make_unique<TrackingBackendDevice>();
    auto* raw = backend.get();
    auto device = rachis::detail::DeviceAccess::make(std::move(backend));
    auto cb = makeCommandBuffer(raw->token);

    const auto moduleReadingDrawIndex = [](ShaderStage stage) {
        return ShaderModuleDesc{
            .stage = stage,
            .shaderInterface = rachis::ShaderInterfaceDesc{.drawIndex = true},
        };
    };
    const auto checkCode = [](const auto& result, ErrorCode code) {
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().code == code);
    };

    checkCode(device.createShaderModule(moduleReadingDrawIndex(ShaderStage::vertex)), ErrorCode::unsupported);
    checkCode(device.createShaderModule(moduleReadingDrawIndex(ShaderStage::mesh)), ErrorCode::unsupported);
    checkCode(device.createPipelineLayout(PipelineLayoutDesc{.drawIndex = true}), ErrorCode::unsupported);
    checkCode(device.drawIndirect(cb, IndirectDrawDesc{.drawIndexPrefix = true}), ErrorCode::unsupported);
    checkCode(device.drawIndexedIndirect(cb, IndirectDrawDesc{.drawIndexPrefix = true}), ErrorCode::unsupported);
    checkCode(device.dispatchMeshIndirect(cb, rachis::IndirectMeshDispatchDesc{.drawIndexPrefix = true}),
              ErrorCode::unsupported);
    for (const auto stage : {ShaderStage::pixel,
                             ShaderStage::compute,
                             ShaderStage::geometry,
                             ShaderStage::hull,
                             ShaderStage::domain,
                             ShaderStage::rayGeneration})
    {
        checkCode(device.createShaderModule(moduleReadingDrawIndex(stage)), ErrorCode::invalidArgument);
    }
    CHECK(raw->backendInvocations == 0);

    raw->caps.shaderDrawIndex = true;
    for (const auto stage : {ShaderStage::vertex, ShaderStage::task, ShaderStage::mesh})
    {
        CHECK(device.createShaderModule(moduleReadingDrawIndex(stage)).has_value());
    }
    checkCode(device.createShaderModule(moduleReadingDrawIndex(ShaderStage::pixel)), ErrorCode::invalidArgument);
    CHECK(device.createPipelineLayout(PipelineLayoutDesc{.drawIndex = true}).has_value());
    CHECK(device.drawIndirect(cb, IndirectDrawDesc{.drawIndexPrefix = true}).has_value());
    CHECK(device.drawIndexedIndirect(cb, IndirectDrawDesc{.drawIndexPrefix = true}).has_value());
    CHECK(raw->backendInvocations == 6);

    raw->state = DeviceOperationalState::lost;
    raw->caps.shaderDrawIndex = false;
    checkCode(device.createPipelineLayout(PipelineLayoutDesc{.drawIndex = true}), ErrorCode::deviceLost);
    checkCode(device.createShaderModule(moduleReadingDrawIndex(ShaderStage::pixel)), ErrorCode::deviceLost);
    CHECK(raw->backendInvocations == 6);
}

TEST_CASE("Device frontend checks a required subgroup size against the device and the workgroup before backend calls",
          "[size:small][rhi][core]")
{
    auto backend = std::make_unique<TrackingBackendDevice>();
    auto* raw = backend.get();
    raw->caps.subgroupOperations = true;
    raw->caps.subgroup = {.minSize = 16, .maxSize = 64, .requiredSizeStages = ShaderStage::compute};
    raw->moduleWorkgroupSize = rachis::ShaderWorkgroupSize{64, 2, 1};
    auto device = rachis::detail::DeviceAccess::make(std::move(backend));

    const auto checkRejected = [&](std::uint32_t size, ErrorCode code, std::string_view message) {
        CAPTURE(size);
        const ComputePipelineDesc desc{.requiredSubgroupSize = size};
        const auto pipeline = device.createComputePipeline(desc);
        REQUIRE_FALSE(pipeline.has_value());
        CHECK(pipeline.error().code == code);
        CHECK(pipeline.error().message == message);
        const auto compilation = device.beginComputePipelineCompilation(desc);
        REQUIRE_FALSE(compilation.has_value());
        CHECK(compilation.error().code == code);
    };
    constexpr std::string_view outsideRange =
        "The required subgroup size is outside this device's subgroup size range.";
    checkRejected(
        24, ErrorCode::invalidArgument, "ComputePipelineDesc::requiredSubgroupSize must be zero or a power of two.");
    checkRejected(8, ErrorCode::unsupported, outsideRange);
    checkRejected(128, ErrorCode::unsupported, outsideRange);
    raw->moduleWorkgroupSize = rachis::ShaderWorkgroupSize{48, 2, 1};
    checkRejected(32,
                  ErrorCode::invalidArgument,
                  "A compute pipeline that requires a subgroup size needs a workgroup width that is a multiple of "
                  "that size.");
    CHECK(raw->backendInvocations == 0);

    for (const std::uint32_t size : {0U, 16U})
    {
        CAPTURE(size);
        const ComputePipelineDesc desc{.requiredSubgroupSize = size};
        CHECK(device.createComputePipeline(desc).has_value());
        CHECK(device.beginComputePipelineCompilation(desc).has_value());
    }
    CHECK(raw->backendInvocations == 4);

    raw->caps.subgroup.requiredSizeStages = ShaderStage::none;
    checkRejected(16, ErrorCode::unsupported, "This device cannot require a subgroup size for compute pipelines.");
    CHECK(raw->backendInvocations == 4);
    raw->caps.subgroup.requiredSizeStages = ShaderStage::compute;
    raw->caps.subgroupOperations = false;
    const ComputePipelineDesc withoutSubgroupOperations{.requiredSubgroupSize = 16};
    CHECK(device.createComputePipeline(withoutSubgroupOperations).has_value());
    CHECK(device.beginComputePipelineCompilation(withoutSubgroupOperations).has_value());
    CHECK(raw->backendInvocations == 6);

    raw->caps.subgroup.requiredSizeStages = ShaderStage::none;
    raw->state = DeviceOperationalState::lost;
    const auto lost = device.createComputePipeline(ComputePipelineDesc{.requiredSubgroupSize = 16});
    REQUIRE_FALSE(lost.has_value());
    CHECK(lost.error().code == ErrorCode::deviceLost);
    CHECK(raw->backendInvocations == 6);
}
