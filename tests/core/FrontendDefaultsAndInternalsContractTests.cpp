#include "Rachis/Device.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#ifdef __OBJC__
#error Public Rachis headers must not require Objective-C++.
#endif

#include "Internal/BackendCommandBuffer.h"
#include "Internal/BackendDevice.h"
#include "Internal/CommandBufferAccess.h"
#include "Internal/DeviceAccess.h"
#include "Internal/DeviceIdentity.h"
#include "Internal/TimelinePointAccess.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace
{

    using rachis::AcquiredSwapchainImage;
    using rachis::AdapterInfo;
    using rachis::Backend;
    using rachis::BackendAllocationSnapshot;
    using rachis::BindingLayoutDesc;
    using rachis::BindingLayoutHandle;
    using rachis::BindingPoolDesc;
    using rachis::BindingPoolHandle;
    using rachis::BindingSetDesc;
    using rachis::BindingSetHandle;
    using rachis::BindingUpdateStatistics;
    using rachis::BindingWrite;
    using rachis::BlendConstants;
    using rachis::BufferBarrier;
    using rachis::BufferCopyRegion;
    using rachis::BufferDesc;
    using rachis::BufferHandle;
    using rachis::BufferTextureCopyRegion;
    using rachis::ClearColor;
    using rachis::CommandBuffer;
    using rachis::CommandPoolDesc;
    using rachis::CommandPoolHandle;
    using rachis::ComputePipelineDesc;
    using rachis::DebugMarkerDesc;
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
    using rachis::VertexBufferBinding;
    using rachis::Viewport;
    using rachis::detail::BackendCommandBuffer;
    using rachis::detail::BackendCommandBufferKind;

    class TestCommandBuffer final : public BackendCommandBuffer
    {
    public:
        explicit TestCommandBuffer(QueueType queueType = QueueType::graphics,
                                   std::uint64_t token = 1,
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

        void invokeResetState() noexcept
        {
            resetState();
        }

    private:
        QueueType queueType_ = QueueType::graphics;
        std::uint64_t token_ = 1;
    };

    class StubBackendDevice : public rachis::detail::BackendDevice
    {
    public:
        using SubmitHandler = std::function<Result<TimelinePoint>(std::span<std::unique_ptr<BackendCommandBuffer>>,
                                                                  std::span<const TimelinePoint>)>;

        SubmitHandler onSubmit{};
        std::uint64_t token = 99;
        bool throwOnCreateBuffer = false;
        bool throwOnWaitIdle = false;
        bool throwOnQueryMemoryBudgets = false;

        std::string_view backendName() const noexcept override
        {
            return "Stub";
        }
        std::uint64_t deviceToken() const noexcept override
        {
            return token;
        }
        const AdapterInfo& adapterInfo() const noexcept override
        {
            return adapterInfo_;
        }
        const DeviceCapabilities& capabilities() const noexcept override
        {
            return capabilities_;
        }
        DeviceOperationalState operationalState() const noexcept override
        {
            return DeviceOperationalState::active;
        }
        std::optional<Error> deviceLossError() const override
        {
            return std::nullopt;
        }
        Result<std::vector<MemoryBudget>> queryMemoryBudgets(std::uint32_t) const override
        {
            if (throwOnQueryMemoryBudgets)
            {
                throw std::bad_alloc{};
            }
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<ResourceMemorySnapshot> captureResourceMemorySnapshot() const override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<DeviceFaultReport> captureDeviceFaultReport() const override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        BackendAllocationSnapshot captureBackendAllocationSnapshot() const noexcept override
        {
            return {};
        }
        FormatProperties queryFormatProperties(Format) const noexcept override
        {
            return {};
        }
        QueueProperties queueProperties(QueueType) const noexcept override
        {
            return {};
        }
        Result<QueueClockCalibration> calibrateQueueClock(QueueType) const override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        DiagnosticCounts diagnosticCounts() const noexcept override
        {
            return {};
        }
        BindingUpdateStatistics bindingUpdateStatistics() const noexcept override
        {
            return {};
        }
        Result<CommandPoolHandle> createCommandPool(const CommandPoolDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status destroyCommandPool(CommandPoolHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status resetCommandPool(CommandPoolHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<std::unique_ptr<BackendCommandBuffer>> acquireCommandBuffer(CommandPoolHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status resetCommandBuffer(BackendCommandBuffer&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<TimelinePoint> submit(std::span<std::unique_ptr<BackendCommandBuffer>> commandBuffers,
                                     std::span<const TimelinePoint> waits) override
        {
            if (onSubmit)
            {
                return onSubmit(commandBuffers, waits);
            }
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<TimelinePointStatus> queryTimelinePoint(TimelinePoint) const override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<TimelinePointStatus> wait(TimelinePoint, std::chrono::nanoseconds) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status wait(TimelinePoint) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status waitIdle(QueueType) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status waitIdle() override
        {
            if (throwOnWaitIdle)
            {
                throw std::bad_alloc{};
            }
            return {};
        }
        std::uint64_t completedValue(QueueType) const noexcept override
        {
            return 0;
        }
        void collectGarbage() noexcept override
        {
        }
        Status beginDebugGroup(BackendCommandBuffer&, const DebugMarkerDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status endDebugGroup(BackendCommandBuffer&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status insertDebugMarker(BackendCommandBuffer&, const DebugMarkerDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<ResourceMemoryRequirements> queryBufferMemoryRequirements(const BufferDesc&) const override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<BufferHandle> createBuffer(const BufferDesc&) override
        {
            if (throwOnCreateBuffer)
            {
                throw std::bad_alloc{};
            }
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status destroyBuffer(BufferHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<MappedMemory> mapBuffer(BufferHandle, std::uint64_t, std::uint64_t) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status unmapBuffer(BufferHandle, std::uint64_t, std::uint64_t) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status flushMappedMemoryRanges(BufferHandle, std::span<const MappedMemoryRange>) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status invalidateMappedMemoryRanges(BufferHandle, std::span<const MappedMemoryRange>) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status writeBuffer(BufferHandle, std::uint64_t, std::span<const std::byte>) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status readBuffer(BufferHandle, std::uint64_t, std::span<std::byte>) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status copyBufferRegions(BackendCommandBuffer&,
                                 BufferHandle,
                                 BufferHandle,
                                 std::span<const BufferCopyRegion>) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status bufferBarrier(BackendCommandBuffer&, const BufferBarrier&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status resourceBarriers(BackendCommandBuffer&, const ResourceBarrierBatch&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<ResourceMemoryRequirements> queryTextureMemoryRequirements(const TextureDesc&) const override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<TextureHandle> createTexture(const TextureDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status destroyTexture(TextureHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<TextureTransferLayout> queryTextureTransferLayout(TextureHandle,
                                                                 const BufferTextureCopyRegion&) const override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status copyBufferToTextureRegions(BackendCommandBuffer&,
                                          BufferHandle,
                                          TextureHandle,
                                          std::span<const BufferTextureCopyRegion>) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status copyTextureToBufferRegions(BackendCommandBuffer&,
                                          TextureHandle,
                                          BufferHandle,
                                          std::span<const BufferTextureCopyRegion>) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status copyTextureRegions(BackendCommandBuffer&,
                                  TextureHandle,
                                  TextureHandle,
                                  std::span<const TextureCopyRegion>) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status resolveTexture(BackendCommandBuffer&, const TextureResolve&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status generateMipmaps(BackendCommandBuffer&, const TextureMipmapGeneration&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status textureBarrier(BackendCommandBuffer&, const TextureBarrier&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<TextureViewHandle> createTextureView(const TextureViewDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status destroyTextureView(TextureViewHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<SamplerHandle> createSampler(const SamplerDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status destroySampler(SamplerHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<BindingLayoutHandle> createBindingLayout(const BindingLayoutDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status destroyBindingLayout(BindingLayoutHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<BindingPoolHandle> createBindingPool(const BindingPoolDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status destroyBindingPool(BindingPoolHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<BindingSetHandle> createBindingSet(const BindingSetDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status updateBindingSet(BindingSetHandle, std::span<const BindingWrite>) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status destroyBindingSet(BindingSetHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<ShaderLibraryHandle> createShaderLibrary(const ShaderLibraryDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status destroyShaderLibrary(ShaderLibraryHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<ShaderModuleHandle> createShaderModule(const ShaderModuleDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<ShaderInterfaceSnapshot> shaderModuleInterface(ShaderModuleHandle) const override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status destroyShaderModule(ShaderModuleHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<PipelineLayoutHandle> createPipelineLayout(const PipelineLayoutDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status destroyPipelineLayout(PipelineLayoutHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<PipelineCacheHandle> createPipelineCache(const PipelineCacheDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status destroyPipelineCache(PipelineCacheHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<std::vector<std::byte>> pipelineCacheData(PipelineCacheHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<PipelineHandle> createComputePipeline(const ComputePipelineDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<PipelineHandle> createGraphicsPipeline(const GraphicsPipelineDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<PipelineCompilationHandle> beginComputePipelineCompilation(const ComputePipelineDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<PipelineCompilationHandle> beginGraphicsPipelineCompilation(const GraphicsPipelineDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<PipelineCompilationState> pipelineCompilationState(PipelineCompilationHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<PipelineHandle> waitPipelineCompilation(PipelineCompilationHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status destroyPipelineCompilation(PipelineCompilationHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status destroyPipeline(PipelineHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status bindComputePipeline(BackendCommandBuffer&, PipelineHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status bindComputeBindingSet(BackendCommandBuffer&,
                                     std::uint32_t,
                                     BindingSetHandle,
                                     std::span<const std::uint32_t>) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status bindComputeBindingSets(BackendCommandBuffer&,
                                      std::uint32_t,
                                      std::span<const BindingSetHandle>,
                                      std::span<const std::uint32_t>) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status setComputePushConstants(BackendCommandBuffer&, std::uint32_t, std::span<const std::byte>) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status dispatch(BackendCommandBuffer&, std::uint32_t, std::uint32_t, std::uint32_t) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status dispatchIndirect(BackendCommandBuffer&, const IndirectDispatchDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status beginRendering(BackendCommandBuffer&, const RenderingDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status endRendering(BackendCommandBuffer&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status clearColorAttachment(BackendCommandBuffer&,
                                    std::uint32_t,
                                    const ClearColor&,
                                    std::span<const ScissorRect>) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status clearDepthStencilAttachment(
            BackendCommandBuffer&, DepthStencilClearAspect, float, std::uint32_t, std::span<const ScissorRect>) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status bindGraphicsPipeline(BackendCommandBuffer&, PipelineHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status bindGraphicsBindingSet(BackendCommandBuffer&,
                                      std::uint32_t,
                                      BindingSetHandle,
                                      std::span<const std::uint32_t>) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status bindGraphicsBindingSets(BackendCommandBuffer&,
                                       std::uint32_t,
                                       std::span<const BindingSetHandle>,
                                       std::span<const std::uint32_t>) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status setGraphicsPushConstants(BackendCommandBuffer&,
                                        ShaderStage,
                                        std::uint32_t,
                                        std::span<const std::byte>) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status setViewports(BackendCommandBuffer&, std::span<const Viewport>) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status setScissors(BackendCommandBuffer&, std::span<const ScissorRect>) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status setBlendConstants(BackendCommandBuffer&, const BlendConstants&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status setStencilReference(BackendCommandBuffer&, StencilFace, std::uint32_t) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status bindVertexBuffers(BackendCommandBuffer&, std::uint32_t, std::span<const VertexBufferBinding>) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status bindIndexBuffer(BackendCommandBuffer&, BufferHandle, IndexFormat, std::uint64_t) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status draw(BackendCommandBuffer&, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status drawIndexed(
            BackendCommandBuffer&, std::uint32_t, std::uint32_t, std::uint32_t, std::int32_t, std::uint32_t) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status drawIndirect(BackendCommandBuffer&, const IndirectDrawDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status drawIndexedIndirect(BackendCommandBuffer&, const IndirectDrawDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<SurfaceHandle> createSurface(const SurfaceDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<SurfaceCapabilities> querySurfaceCapabilities(SurfaceHandle) const override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status destroySurface(SurfaceHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<SwapchainHandle> createSwapchain(const SwapchainDesc&) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status destroySwapchain(SwapchainHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Status resizeSwapchain(SwapchainHandle, std::uint32_t, std::uint32_t) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<AcquiredSwapchainImage> acquireNextImage(SwapchainHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        bool hasAcquiredSwapchainImage(SwapchainHandle) const noexcept override
        {
            return false;
        }
        Status cancelSwapchainImage(SwapchainHandle) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<PresentStatus> present(SwapchainHandle, TimelinePoint) override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<SwapchainPresentationInfo> querySwapchainPresentation(SwapchainHandle) const override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }
        Result<SwapchainFrameTiming> querySwapchainFrameTiming(SwapchainHandle) const override
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, "stub"});
        }

    private:
        AdapterInfo adapterInfo_{};
        DeviceCapabilities capabilities_{};
    };

} // namespace

TEST_CASE("default enumerateAdapters and createDevice target defaultHostBackend without DRED rejection",
          "[size:small][rhi][core]")
{
    const auto defaultAdapters = rachis::enumerateAdapters();
    const auto explicitAdapters = rachis::enumerateAdapters(rachis::defaultHostBackend());
    REQUIRE(defaultAdapters.has_value() == explicitAdapters.has_value());
    if (!defaultAdapters)
    {
        CHECK(defaultAdapters.error().code == explicitAdapters.error().code);
    }
    else
    {
        REQUIRE(defaultAdapters->size() == explicitAdapters->size());
        for (std::size_t index = 0; index < defaultAdapters->size(); ++index)
        {
            CHECK((*defaultAdapters)[index].id == (*explicitAdapters)[index].id);
            CHECK((*defaultAdapters)[index].backend == (*explicitAdapters)[index].backend);
            CHECK((*defaultAdapters)[index].backend == rachis::defaultHostBackend());
        }
    }

    auto defaultDevice = rachis::createDevice();
    if (!defaultDevice)
    {
        CHECK_FALSE(defaultDevice.error().message.contains("enableDred"));
    }
    else
    {
        CHECK(defaultDevice->isValid());
        REQUIRE(defaultDevice->adapterInfo().has_value());
        CHECK(defaultDevice->adapterInfo()->backend == rachis::defaultHostBackend());
    }

    auto vulkanDefaultDred = rachis::createDevice(rachis::DeviceDesc{
        .backend = Backend::vulkan,
        .validation = rachis::ValidationLevel::disabled,
    });
    if (!vulkanDefaultDred)
    {
        CHECK_FALSE(vulkanDefaultDred.error().message.contains("enableDred"));
    }

    auto metalDefaultDred = rachis::createDevice(rachis::DeviceDesc{
        .backend = Backend::metal,
        .validation = rachis::ValidationLevel::disabled,
    });
    if (!metalDefaultDred)
    {
        CHECK_FALSE(metalDefaultDred.error().message.contains("enableDred"));
    }
}

TEST_CASE("FixedBoolMask provides compact inline bitmask storage and range semantics", "[size:small][rhi][core]")
{
    using rachis::detail::BindingSetMask;
    using rachis::detail::VertexBufferMask;

    constexpr auto compileTimeCheck = []() constexpr {
        BindingSetMask mask{true, false, true};
        if (mask.size() != 3 || mask.empty() || mask.all())
        {
            return false;
        }
        mask[1] = true;
        return mask.all() && mask[0] && mask[1] && mask[2];
    };
    static_assert(compileTimeCheck());

    BindingSetMask sets{};
    CHECK(sets.empty());
    CHECK(sets.size() == 0);
    CHECK(sets.all());
    CHECK(std::ranges::all_of(sets, [](bool bound) { return bound; }));

    sets.reserve(4);
    sets.push_back(true);
    sets.push_back(false);
    sets.push_back(true);
    sets.push_back(false);
    CHECK(sets.size() == 4);
    CHECK_FALSE(sets.empty());
    CHECK_FALSE(sets.all());
    CHECK_FALSE(std::ranges::all_of(sets, [](bool bound) { return bound; }));
    CHECK(sets == BindingSetMask{true, false, true, false});
    CHECK(sets != BindingSetMask{true, true, true, false});
    CHECK(sets != BindingSetMask{true, false, true});

    sets[1] = true;
    sets[3] = sets[0];
    CHECK(sets[1]);
    CHECK(sets[3]);
    CHECK(sets.all());
    CHECK(std::ranges::all_of(sets, [](bool bound) { return bound; }));
    CHECK(std::ranges::equal(sets, std::array{true, true, true, true}));

    sets.assign(2, false);
    CHECK(sets.size() == 2);
    CHECK_FALSE(sets[0]);
    CHECK_FALSE(sets[1]);
    CHECK_FALSE(sets.all());
    CHECK(sets == BindingSetMask{false, false});

    sets.assign(2, true);
    CHECK(sets.all());
    CHECK(sets == BindingSetMask{true, true});

    sets.clear();
    CHECK(sets.empty());
    CHECK(sets.all());

    VertexBufferMask vbs{};
    vbs.assign(8, false);
    CHECK(vbs.size() == 8);
    CHECK_FALSE(vbs.all());
    for (std::size_t slot = 0; slot < 8; ++slot)
    {
        CHECK_FALSE(vbs.all());
        vbs[slot] = true;
    }
    CHECK(vbs.all());
    CHECK(std::ranges::count(vbs, true) == 8);
    vbs[7] = false;
    CHECK_FALSE(vbs.all());
    CHECK((vbs.end() - vbs.begin()) == 8);
    CHECK(vbs.begin()[6]);
    CHECK_FALSE(vbs.begin()[7]);
}

TEST_CASE("SharedHandleView accepts only known handle types with a usable native value",
          "[size:small][rhi][core][interop]")
{
    using rachis::SharedHandle;
    using rachis::SharedHandleType;
    using rachis::SharedHandleView;
    using rachis::SharedTimelinePoint;
    using rachis::SharedTimelinePointView;

    constexpr auto invalidHandleValue = std::numeric_limits<std::uintptr_t>::max();
    constexpr SharedHandleView usable{.type = SharedHandleType::windowsNt, .value = 4};
    static_assert(usable.isValid());
    CHECK(SharedTimelinePointView{.handle = usable, .value = 1}.isValid());

    const std::array rejected{
        SharedHandleView{},
        SharedHandleView{.type = SharedHandleType::unknown, .value = 4},
        SharedHandleView{.type = SharedHandleType::windowsNt, .value = 0},
        SharedHandleView{.type = SharedHandleType::windowsNt, .value = invalidHandleValue},
        SharedHandleView{.type = static_cast<SharedHandleType>(2), .value = 4},
        SharedHandleView{.type = static_cast<SharedHandleType>(255), .value = 4},
        SharedHandleView{.type = static_cast<SharedHandleType>(2), .value = invalidHandleValue},
    };
    for (const auto view : rejected)
    {
        CAPTURE(static_cast<unsigned int>(view.type), view.value);
        CHECK_FALSE(view.isValid());
        CHECK_FALSE(SharedTimelinePointView{.handle = view, .value = 1}.isValid());

        auto adopted = SharedHandle::adopt(view);
        CHECK_FALSE(adopted.isValid());
        CHECK_FALSE(static_cast<bool>(adopted));
        CHECK(adopted.view().type == view.type);
        CHECK(adopted.view().value == view.value);

        const SharedTimelinePoint point{.handle = SharedHandle::adopt(view), .value = 1};
        CHECK_FALSE(point.isValid());
        CHECK_FALSE(point.view().isValid());

        adopted.reset();
        CHECK(adopted.view().type == SharedHandleType::unknown);
        CHECK(adopted.view().value == 0);
    }

    auto owned = SharedHandle::adopt(usable);
    CHECK(owned.isValid());
    const auto released = owned.release();
    CHECK(released.type == SharedHandleType::windowsNt);
    CHECK(released.value == usable.value);
    CHECK_FALSE(owned.isValid());
}

TEST_CASE("BackendCommandBuffer resetState releases shared owners and clears recorded state", "[size:small][rhi][core]")
{
    TestCommandBuffer cb(QueueType::graphics, 42, BackendCommandBufferKind::metal);
    auto dummyOwner = std::make_shared<int>(7);

    cb.computeState().pipelineOwner = dummyOwner;
    cb.computeState().boundBindingSets = {true, false};
    cb.rayTracingState().pipelineOwner = dummyOwner;
    cb.rayTracingState().boundBindingSets = {false, true};
    cb.graphicsState().renderingOwner = dummyOwner;
    cb.graphicsState().pipelineOwner = dummyOwner;
    cb.graphicsState().boundBindingSets = {true, true};
    cb.graphicsState().boundVertexBuffers.assign(4, true);
    cb.graphicsState().boundStreamOutputBuffers.assign(2, true);
    cb.graphicsState().indexBufferBound = true;
    cb.graphicsState().viewportCount = 2;
    cb.graphicsState().scissorCount = 2;

    auto& query = cb.queryState();

    const rachis::detail::QueryWrite sampleWrite{.queryPool = {}, .queryIndex = 3, .resolved = false};
    query.timestampWrites.push_back(sampleWrite);
    query.occlusionWrites.push_back(sampleWrite);
    query.pipelineStatisticsWrites.push_back(sampleWrite);
    query.streamOutputStatisticsWrites.push_back(sampleWrite);
    query.accelerationStructureCompactedSizeWrites.push_back(sampleWrite);
    query.accelerationStructureSerializationInfoWrites.push_back(sampleWrite);

    query.activeOcclusion = rachis::detail::ActiveQuery{.queryPool = {}, .owner = dummyOwner};
    query.activePipelineStatistics = rachis::detail::ActiveQuery{.queryPool = {}, .owner = dummyOwner};
    query.activeStreamOutputStatistics[0] = rachis::detail::ActiveQuery{.queryPool = {}, .owner = dummyOwner};
    query.activePassTimestampWrites = rachis::detail::ActivePassTimestampWrites{.queryPool = {}, .owner = dummyOwner};
    query.activeConditionalExecution = rachis::detail::ActiveConditionalExecution{.owner = dummyOwner};
    query.activeGpuProfileScopes[0] = rachis::detail::ActiveGpuProfileScope{.queryPool = {}, .owner = dummyOwner};
    query.activeGpuProfileScopes[1] = rachis::detail::ActiveGpuProfileScope{.queryPool = {}, .owner = dummyOwner};
    query.activeGpuProfileScopeDepth = 2;

    cb.debugState().groupThread = std::this_thread::get_id();
    cb.debugState().groupDepth = 3;

    REQUIRE(dummyOwner.use_count() > 1);

    cb.invokeResetState();

    CHECK(dummyOwner.use_count() == 1);
    CHECK(cb.computeState().pipelineOwner == nullptr);
    CHECK(cb.computeState().boundBindingSets.empty());
    CHECK(cb.rayTracingState().pipelineOwner == nullptr);
    CHECK(cb.rayTracingState().boundBindingSets.empty());
    CHECK(cb.graphicsState().renderingOwner == nullptr);
    CHECK(cb.graphicsState().pipelineOwner == nullptr);
    CHECK(cb.graphicsState().boundBindingSets.empty());
    CHECK(cb.graphicsState().boundVertexBuffers.empty());
    CHECK(cb.graphicsState().boundStreamOutputBuffers.empty());
    CHECK_FALSE(cb.graphicsState().indexBufferBound);
    CHECK(cb.graphicsState().viewportCount == 0);
    CHECK(cb.graphicsState().scissorCount == 0);

    CHECK(query.timestampWrites.empty());
    CHECK(query.occlusionWrites.empty());
    CHECK(query.pipelineStatisticsWrites.empty());
    CHECK(query.streamOutputStatisticsWrites.empty());
    CHECK(query.accelerationStructureCompactedSizeWrites.empty());
    CHECK(query.accelerationStructureSerializationInfoWrites.empty());

    CHECK_FALSE(query.activeOcclusion.has_value());
    CHECK_FALSE(query.activePipelineStatistics.has_value());
    for (const auto& active : query.activeStreamOutputStatistics)
    {
        CHECK_FALSE(active.has_value());
    }
    CHECK_FALSE(query.activePassTimestampWrites.has_value());
    CHECK_FALSE(query.activeConditionalExecution.has_value());
    CHECK(query.activeGpuProfileScopeDepth == 0);
    CHECK(query.activeGpuProfileScopes[0].owner == nullptr);
    CHECK(query.activeGpuProfileScopes[1].owner == nullptr);

    CHECK(cb.debugState().groupDepth == 0);
    CHECK(cb.debugState().groupThread == std::thread::id{});

    {
        auto completedScopeOwner = std::make_shared<int>(99);
        const std::weak_ptr<int> weakOwner = completedScopeOwner;
        query.activeGpuProfileScopes[0] =
            rachis::detail::ActiveGpuProfileScope{.queryPool = {}, .owner = completedScopeOwner};
        query.activeGpuProfileScopes[rachis::maxDebugGroupDepth - 1] =
            rachis::detail::ActiveGpuProfileScope{.queryPool = {}, .owner = std::move(completedScopeOwner)};
        query.activeGpuProfileScopeDepth = 0;
        cb.debugState().groupOwners[0] = rachis::detail::DebugGroupOwner::gpuProfile;
        REQUIRE_FALSE(weakOwner.expired());

        cb.invokeResetState();

        CHECK(weakOwner.expired());
        CHECK(query.activeGpuProfileScopes[0].owner == nullptr);
        CHECK(query.activeGpuProfileScopes[rachis::maxDebugGroupDepth - 1].owner == nullptr);
        CHECK(cb.debugState().groupOwners[0] == rachis::detail::DebugGroupOwner::user);
    }

#if RACHIS_ENABLE_METAL
    if (auto device =
            rachis::createDevice({.backend = rachis::Backend::metal, .validation = rachis::ValidationLevel::standard});
        device.has_value())
    {
        auto pool = device->createCommandPool({
            .queueType = rachis::QueueType::compute,
            .maxCommandBuffers = 1,
            .flags = rachis::CommandPoolFlag::allowIndividualReset,
        });
        REQUIRE(pool.has_value());
        auto cmd = device->acquireCommandBuffer(*pool);
        REQUIRE(cmd.has_value());
        if (auto queryPool = device->createTimestampQueryPool({
                .queueType = rachis::QueueType::compute,
                .queryCount = 2,
            });
            queryPool.has_value())
        {
            REQUIRE(device
                        ->beginGpuProfileScope(*cmd,
                                               {
                                                   .queryPool = *queryPool,
                                                   .beginningQueryIndex = 0,
                                                   .endQueryIndex = 1,
                                                   .label = "Completed profile scope reset",
                                               })
                        .has_value());
            auto* backendCmd = rachis::detail::CommandBufferAccess::get(*cmd);
            REQUIRE(backendCmd != nullptr);
            CHECK(backendCmd->queryState().activeGpuProfileScopeDepth == 1);
            REQUIRE(backendCmd->queryState().activeGpuProfileScopes[0].owner != nullptr);
            const std::weak_ptr<void> weakPoolOwner = backendCmd->queryState().activeGpuProfileScopes[0].owner;
            REQUIRE_FALSE(weakPoolOwner.expired());

            REQUIRE(device->endGpuProfileScope(*cmd).has_value());
            CHECK(backendCmd->queryState().activeGpuProfileScopeDepth == 0);
            CHECK(backendCmd->queryState().activeGpuProfileScopes[0].owner == nullptr);

            REQUIRE(device->destroyQueryPool(*queryPool).has_value());
            REQUIRE(device->resetCommandBuffer(*cmd).has_value());
            CHECK(backendCmd->queryState().activeGpuProfileScopes[0].owner == nullptr);
            CHECK(weakPoolOwner.expired());
        }
        REQUIRE(device->destroyCommandPool(*pool).has_value());
    }
#endif
}

TEST_CASE("deviceTokenSource is process-global monotonic and Device::submit handles inline, heap, and bad_alloc paths",
          "[size:small][rhi][core]")
{
    const auto firstToken = rachis::detail::deviceTokenSource().acquire();
    const auto secondToken = rachis::detail::deviceTokenSource().acquire();
    REQUIRE(firstToken.has_value());
    REQUIRE(secondToken.has_value());
    CHECK(*firstToken != 0);
    CHECK(*secondToken > *firstToken);

#ifdef RACHIS_TESTING
    {
        rachis::detail::MonotonicIdentitySource finalIdentity(std::numeric_limits<std::uint64_t>::max());
        struct OverrideGuard final
        {
            ~OverrideGuard() noexcept
            {
                rachis::detail::setDeviceTokenSourceOverrideForTesting(nullptr);
            }
        } guard;
        rachis::detail::setDeviceTokenSourceOverrideForTesting(&finalIdentity);
        const auto lastValid = rachis::detail::deviceTokenSource().acquire();
        REQUIRE(lastValid.has_value());
        CHECK(*lastValid == std::numeric_limits<std::uint64_t>::max());
        CHECK_FALSE(rachis::detail::deviceTokenSource().acquire().has_value());

        const auto hostAdapters = rachis::enumerateAdapters();
        const auto hardware = [](const rachis::AdapterInfo& adapter) {
            return adapter.type != rachis::AdapterType::software;
        };
        if (hostAdapters.has_value() && std::ranges::any_of(*hostAdapters, hardware))
        {
            const auto exhaustedCreate = rachis::createDevice(rachis::DeviceDesc{
                .validation = rachis::ValidationLevel::disabled,
            });
            REQUIRE_FALSE(exhaustedCreate.has_value());
            CHECK(exhaustedCreate.error().code == ErrorCode::resourceExhausted);
        }
    }
    const auto thirdToken = rachis::detail::deviceTokenSource().acquire();
    REQUIRE(thirdToken.has_value());
    CHECK(*thirdToken > *secondToken);
#endif

    auto makeCommands = [](std::size_t count, QueueType queueType = QueueType::graphics) {
        std::vector<CommandBuffer> commands;
        commands.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
        {
            commands.push_back(
                rachis::detail::CommandBufferAccess::make(std::make_unique<TestCommandBuffer>(queueType, 99)));
        }
        return commands;
    };

    auto stub = std::make_unique<StubBackendDevice>();
    auto* stubPtr = stub.get();
    auto device = rachis::detail::DeviceAccess::make(std::move(stub));
    REQUIRE(device.isValid());

    stubPtr->onSubmit = [](std::span<std::unique_ptr<BackendCommandBuffer>> batch,
                           std::span<const TimelinePoint>) -> Result<TimelinePoint> {
        REQUIRE(batch.size() == 8);
        return std::unexpected(Error{ErrorCode::invalidArgument, 0, "rejected before consume"});
    };
    auto inlineBatch = makeCommands(8);
    const auto inlineRejected = device.submit(inlineBatch);
    REQUIRE_FALSE(inlineRejected.has_value());
    CHECK(inlineRejected.error().code == ErrorCode::invalidArgument);
    for (const auto& cb : inlineBatch)
    {
        CHECK(cb.isValid());
    }

    stubPtr->onSubmit = [](std::span<std::unique_ptr<BackendCommandBuffer>> batch,
                           std::span<const TimelinePoint>) -> Result<TimelinePoint> {
        REQUIRE(batch.size() == 8);
        throw std::bad_alloc{};
    };
    const auto inlineOom = device.submit(inlineBatch);
    REQUIRE_FALSE(inlineOom.has_value());
    CHECK(inlineOom.error().code == ErrorCode::outOfMemory);
    for (const auto& cb : inlineBatch)
    {
        CHECK(cb.isValid());
    }

    stubPtr->onSubmit = [](std::span<std::unique_ptr<BackendCommandBuffer>> batch,
                           std::span<const TimelinePoint>) -> Result<TimelinePoint> {
        REQUIRE(batch.size() == 8);
        for (auto& cb : batch)
        {
            REQUIRE(cb != nullptr);
            cb.reset();
        }
        return rachis::detail::TimelinePointAccess::make(QueueType::graphics, 7, 99);
    };
    const auto inlineOk = device.submit(inlineBatch);
    REQUIRE(inlineOk.has_value());
    CHECK(inlineOk->value() == 7);
    for (const auto& cb : inlineBatch)
    {
        CHECK_FALSE(cb.isValid());
    }

    auto heapBatch = makeCommands(12);
    stubPtr->onSubmit = [](std::span<std::unique_ptr<BackendCommandBuffer>> batch,
                           std::span<const TimelinePoint>) -> Result<TimelinePoint> {
        REQUIRE(batch.size() == 12);
        return std::unexpected(Error{ErrorCode::invalidState, 0, "rejected heap batch"});
    };
    const auto heapRejected = device.submit(heapBatch);
    REQUIRE_FALSE(heapRejected.has_value());
    CHECK(heapRejected.error().code == ErrorCode::invalidState);
    for (const auto& cb : heapBatch)
    {
        CHECK(cb.isValid());
    }

    stubPtr->onSubmit = [](std::span<std::unique_ptr<BackendCommandBuffer>> batch,
                           std::span<const TimelinePoint>) -> Result<TimelinePoint> {
        REQUIRE(batch.size() == 12);
        throw std::bad_alloc{};
    };
    const auto heapOom = device.submit(heapBatch);
    REQUIRE_FALSE(heapOom.has_value());
    CHECK(heapOom.error().code == ErrorCode::outOfMemory);
    for (const auto& cb : heapBatch)
    {
        CHECK(cb.isValid());
    }

    rachis::detail::CommandBufferAccess::get(heapBatch[5])->debugState().groupDepth = 1;
    bool backendCalled = false;
    stubPtr->onSubmit = [&](std::span<std::unique_ptr<BackendCommandBuffer>>,
                            std::span<const TimelinePoint>) -> Result<TimelinePoint> {
        backendCalled = true;
        return rachis::detail::TimelinePointAccess::make(QueueType::graphics, 8, 99);
    };
    const auto openScopeRejected = device.submit(heapBatch);
    REQUIRE_FALSE(openScopeRejected.has_value());
    CHECK(openScopeRejected.error().code == ErrorCode::invalidState);
    CHECK_FALSE(backendCalled);
    for (const auto& cb : heapBatch)
    {
        CHECK(cb.isValid());
    }
    rachis::detail::CommandBufferAccess::get(heapBatch[5])->debugState().reset();

    stubPtr->onSubmit = [](std::span<std::unique_ptr<BackendCommandBuffer>> batch,
                           std::span<const TimelinePoint>) -> Result<TimelinePoint> {
        REQUIRE(batch.size() == 12);
        for (auto& cb : batch)
        {
            cb.reset();
        }
        return rachis::detail::TimelinePointAccess::make(QueueType::graphics, 9, 99);
    };
    const auto heapOk = device.submit(heapBatch);
    REQUIRE(heapOk.has_value());
    CHECK(heapOk->value() == 9);
    for (const auto& cb : heapBatch)
    {
        CHECK_FALSE(cb.isValid());
    }

    stubPtr->throwOnCreateBuffer = true;
    const auto bufferOom = device.createBuffer(BufferDesc{.size = 64});
    REQUIRE_FALSE(bufferOom.has_value());
    CHECK(bufferOom.error().code == ErrorCode::outOfMemory);

    stubPtr->throwOnWaitIdle = true;
    const auto waitIdleOom = device.waitIdle();
    REQUIRE_FALSE(waitIdleOom.has_value());
    CHECK(waitIdleOom.error().code == ErrorCode::outOfMemory);

    stubPtr->throwOnQueryMemoryBudgets = true;
    const auto budgetsOom = device.queryMemoryBudgets();
    REQUIRE_FALSE(budgetsOom.has_value());
    CHECK(budgetsOom.error().code == ErrorCode::outOfMemory);
}
