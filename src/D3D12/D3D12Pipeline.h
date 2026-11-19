#pragma once

#include "D3D12/D3D12Binding.h"
#include "D3D12/D3D12Queue.h"
#include "Rachis/Device.h"
#include "Rachis/Pipeline.h"
#include "Rachis/Rendering.h"

#include <d3d12.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace rachis::d3d12
{

    inline constexpr std::uint32_t invalidRootParameter = UINT32_MAX;

    class QueryManager;

    class NativePipelineContinuation final
    {
    public:
        [[nodiscard]] bool hasPipeline() const noexcept;
        void restore(ID3D12GraphicsCommandList& commandList) const noexcept;

    private:
        friend class PipelineManager;

        ID3D12PipelineState* computePipeline_ = nullptr;
        ID3D12GraphicsCommandList4* rayTracingCommandList_ = nullptr;
        ID3D12StateObject* rayTracingPipeline_ = nullptr;
    };

    struct NativeShaderModule final
    {
        std::shared_ptr<void> owner;
        ShaderStage stage = ShaderStage::none;
        D3D12_SHADER_BYTECODE bytecode{};
        std::string_view entryPoint;
    };

    struct NativePipelineLayoutSlot final
    {
        std::shared_ptr<void> bindingLayoutOwner = {};
        std::vector<BindingLayoutEntry> entries = {};
        RootSignatureSlotPlan rootPlan = {};
        std::uint32_t resourceRootParameter = invalidRootParameter;
        std::uint32_t samplerRootParameter = invalidRootParameter;
        std::uint32_t dynamicRootParameter = invalidRootParameter;
        std::uint32_t dynamicBufferCount = 0;
        bool requiresBindingSet = false;
    };

    struct NativePipelineLayout final
    {
        std::shared_ptr<void> owner;
        ID3D12RootSignature* rootSignature = nullptr;
        std::vector<NativePipelineLayoutSlot> slots;
        std::vector<PushConstantRange> pushConstantRanges;
        std::uint32_t pushConstantRootParameter = invalidRootParameter;
        std::uint32_t pushConstantSize = 0;
    };

    class PipelineManager final
    {
    public:
        PipelineManager();
        ~PipelineManager();

        PipelineManager(const PipelineManager&) = delete;
        PipelineManager& operator=(const PipelineManager&) = delete;
        PipelineManager(PipelineManager&&) = delete;
        PipelineManager& operator=(PipelineManager&&) = delete;

        [[nodiscard]] Status initialize(ID3D12Device& device,
                                        std::uint32_t maxComputeWorkgroupInvocations,
                                        const std::array<std::uint32_t, 3>& maxComputeWorkgroupSize,
                                        bool geometryShaders,
                                        std::uint32_t maxPatchControlPoints,
                                        bool meshShaders,
                                        bool taskShaders,
                                        const MeshShaderCapabilities& meshShaderCapabilities,
                                        bool dualSourceBlending,
                                        bool logicOperations,
                                        bool conservativeRasterization,
                                        bool triangleFans,
                                        bool dynamicDepthBias,
                                        bool depthBoundsTest,
                                        bool variableRateShading,
                                        bool additionalShadingRates,
                                        bool variableRateShadingTier2,
                                        std::uint32_t maxClipDistances,
                                        std::uint32_t maxCullDistances,
                                        std::uint32_t maxCombinedClipAndCullDistances,
                                        std::uint32_t maxFramebufferLayers,
                                        std::uint32_t maxMultiviewViewCount,
                                        bool rayTracing,
                                        const RayTracingCapabilities& rayTracingCapabilities,
                                        const SubgroupCapabilities& subgroup);
        [[nodiscard]] Result<ShaderLibraryHandle> createShaderLibrary(const ShaderLibraryDesc& desc);
        [[nodiscard]] Status destroyShaderLibrary(ShaderLibraryHandle shaderLibrary);
        [[nodiscard]] Result<ShaderModuleHandle> createShaderModule(const ShaderModuleDesc& desc);
        [[nodiscard]] Result<ShaderInterfaceSnapshot> shaderModuleInterface(ShaderModuleHandle shaderModule) const;
        [[nodiscard]] Status destroyShaderModule(ShaderModuleHandle shaderModule);
        [[nodiscard]] Result<PipelineLayoutHandle> createPipelineLayout(BindingManager& bindings,
                                                                        const PipelineLayoutDesc& desc);
        [[nodiscard]] Status destroyPipelineLayout(PipelineLayoutHandle pipelineLayout);
        [[nodiscard]] Result<PipelineCacheHandle> createPipelineCache(const PipelineCacheDesc& desc);
        [[nodiscard]] Status destroyPipelineCache(PipelineCacheHandle pipelineCache);
        [[nodiscard]] Result<std::vector<std::byte>> pipelineCacheData(PipelineCacheHandle pipelineCache);
        [[nodiscard]] Result<PipelineHandle> createComputePipeline(const ComputePipelineDesc& desc);
        [[nodiscard]] Result<PipelineHandle> createGraphicsPipeline(const GraphicsPipelineDesc& desc);
        [[nodiscard]] Result<PipelineCompilationHandle> beginComputePipelineCompilation(
            const ComputePipelineDesc& desc);
        [[nodiscard]] Result<PipelineCompilationHandle> beginGraphicsPipelineCompilation(
            const GraphicsPipelineDesc& desc);
        [[nodiscard]] Result<PipelineCompilationState> pipelineCompilationState(PipelineCompilationHandle compilation);
        [[nodiscard]] Result<PipelineHandle> waitPipelineCompilation(PipelineCompilationHandle compilation);
        [[nodiscard]] Status destroyPipelineCompilation(PipelineCompilationHandle compilation);
        [[nodiscard]] Result<RayTracingPipelineLibraryHandle> createRayTracingPipelineLibrary(
            const RayTracingPipelineLibraryDesc& desc);
        [[nodiscard]] Status destroyRayTracingPipelineLibrary(RayTracingPipelineLibraryHandle library);
        [[nodiscard]] Result<PipelineHandle> createRayTracingPipeline(const RayTracingPipelineDesc& desc);
        [[nodiscard]] Status getRayTracingShaderGroupHandles(PipelineHandle pipeline,
                                                             std::uint32_t firstGroup,
                                                             std::uint32_t groupCount,
                                                             std::span<std::byte> destination);
        [[nodiscard]] Result<std::uint32_t> rayTracingShaderGroupRecordSize(PipelineHandle pipeline,
                                                                            std::uint32_t groupIndex);
        [[nodiscard]] Status writeRayTracingShaderGroupRecord(PipelineHandle pipeline,
                                                              std::uint32_t groupIndex,
                                                              std::span<const std::byte> localConstants,
                                                              std::span<std::byte> destination);
        [[nodiscard]] Status destroyPipeline(PipelineHandle pipeline);
        [[nodiscard]] Result<NativePipelineContinuation> prepareNativePipelineContinuation(
            QueueManager& queues, detail::BackendCommandBuffer& commandBuffer) const;
        [[nodiscard]] Status bindComputePipeline(QueueManager& queues,
                                                 BindingManager& bindings,
                                                 detail::BackendCommandBuffer& commandBuffer,
                                                 PipelineHandle pipeline);
        [[nodiscard]] Status bindComputeBindingSet(QueueManager& queues,
                                                   BindingManager& bindings,
                                                   ResourceManager& resources,
                                                   detail::BackendCommandBuffer& commandBuffer,
                                                   std::uint32_t setIndex,
                                                   BindingSetHandle bindingSet,
                                                   std::span<const std::uint32_t> dynamicOffsets);
        [[nodiscard]] Status bindComputeBindingSets(QueueManager& queues,
                                                    BindingManager& bindings,
                                                    ResourceManager& resources,
                                                    detail::BackendCommandBuffer& commandBuffer,
                                                    std::uint32_t firstSet,
                                                    std::span<const BindingSetHandle> bindingSets,
                                                    std::span<const std::uint32_t> dynamicOffsets);
        [[nodiscard]] Status setComputePushConstants(QueueManager& queues,
                                                     detail::BackendCommandBuffer& commandBuffer,
                                                     std::uint32_t offset,
                                                     std::span<const std::byte> data);
        [[nodiscard]] Status dispatch(QueueManager& queues,
                                      detail::BackendCommandBuffer& commandBuffer,
                                      std::uint32_t groupCountX,
                                      std::uint32_t groupCountY,
                                      std::uint32_t groupCountZ);
        [[nodiscard]] Status dispatchIndirect(QueueManager& queues,
                                              ResourceManager& resources,
                                              detail::BackendCommandBuffer& commandBuffer,
                                              const IndirectDispatchDesc& desc);
        [[nodiscard]] Status bindRayTracingPipeline(QueueManager& queues,
                                                    BindingManager& bindings,
                                                    detail::BackendCommandBuffer& commandBuffer,
                                                    PipelineHandle pipeline);
        [[nodiscard]] Status bindRayTracingBindingSet(QueueManager& queues,
                                                      BindingManager& bindings,
                                                      ResourceManager& resources,
                                                      detail::BackendCommandBuffer& commandBuffer,
                                                      std::uint32_t setIndex,
                                                      BindingSetHandle bindingSet,
                                                      std::span<const std::uint32_t> dynamicOffsets);
        [[nodiscard]] Status bindRayTracingBindingSets(QueueManager& queues,
                                                       BindingManager& bindings,
                                                       ResourceManager& resources,
                                                       detail::BackendCommandBuffer& commandBuffer,
                                                       std::uint32_t firstSet,
                                                       std::span<const BindingSetHandle> bindingSets,
                                                       std::span<const std::uint32_t> dynamicOffsets);
        [[nodiscard]] Status setRayTracingPushConstants(QueueManager& queues,
                                                        detail::BackendCommandBuffer& commandBuffer,
                                                        std::uint32_t offset,
                                                        std::span<const std::byte> data);
        [[nodiscard]] Status dispatchRays(QueueManager& queues,
                                          ResourceManager& resources,
                                          detail::BackendCommandBuffer& commandBuffer,
                                          const RayTracingDispatchDesc& desc);
        [[nodiscard]] Status dispatchRaysIndirect(QueueManager& queues,
                                                  ResourceManager& resources,
                                                  detail::BackendCommandBuffer& commandBuffer,
                                                  const RayTracingIndirectDispatchDesc& desc);
        [[nodiscard]] Status beginRendering(QueueManager& queues,
                                            ViewManager& views,
                                            QueryManager& queries,
                                            detail::BackendCommandBuffer& commandBuffer,
                                            const RenderingDesc& desc);
        [[nodiscard]] Status endRendering(QueueManager& queues,
                                          QueryManager& queries,
                                          detail::BackendCommandBuffer& commandBuffer);
        [[nodiscard]] Status clearColorAttachment(QueueManager& queues,
                                                  detail::BackendCommandBuffer& commandBuffer,
                                                  std::uint32_t attachmentIndex,
                                                  const ClearColor& clearValue,
                                                  std::span<const ScissorRect> rectangles);
        [[nodiscard]] Status clearDepthStencilAttachment(QueueManager& queues,
                                                         detail::BackendCommandBuffer& commandBuffer,
                                                         DepthStencilClearAspect aspects,
                                                         float depth,
                                                         std::uint32_t stencil,
                                                         std::span<const ScissorRect> rectangles);
        [[nodiscard]] Status bindGraphicsPipeline(QueueManager& queues,
                                                  BindingManager& bindings,
                                                  detail::BackendCommandBuffer& commandBuffer,
                                                  PipelineHandle pipeline);
        [[nodiscard]] Status bindGraphicsBindingSet(QueueManager& queues,
                                                    BindingManager& bindings,
                                                    ResourceManager& resources,
                                                    detail::BackendCommandBuffer& commandBuffer,
                                                    std::uint32_t setIndex,
                                                    BindingSetHandle bindingSet,
                                                    std::span<const std::uint32_t> dynamicOffsets);
        [[nodiscard]] Status bindGraphicsBindingSets(QueueManager& queues,
                                                     BindingManager& bindings,
                                                     ResourceManager& resources,
                                                     detail::BackendCommandBuffer& commandBuffer,
                                                     std::uint32_t firstSet,
                                                     std::span<const BindingSetHandle> bindingSets,
                                                     std::span<const std::uint32_t> dynamicOffsets);
        [[nodiscard]] Status setGraphicsPushConstants(QueueManager& queues,
                                                      detail::BackendCommandBuffer& commandBuffer,
                                                      ShaderStage visibility,
                                                      std::uint32_t offset,
                                                      std::span<const std::byte> data);
        [[nodiscard]] Status setViewports(QueueManager& queues,
                                          detail::BackendCommandBuffer& commandBuffer,
                                          std::span<const Viewport> viewports);
        [[nodiscard]] Status setScissors(QueueManager& queues,
                                         detail::BackendCommandBuffer& commandBuffer,
                                         std::span<const ScissorRect> scissors);
        [[nodiscard]] Status setBlendConstants(QueueManager& queues,
                                               detail::BackendCommandBuffer& commandBuffer,
                                               const BlendConstants& constants);
        [[nodiscard]] Status setDepthBias(QueueManager& queues,
                                          detail::BackendCommandBuffer& commandBuffer,
                                          const DepthBias& bias);
        [[nodiscard]] Status setDepthBounds(QueueManager& queues,
                                            detail::BackendCommandBuffer& commandBuffer,
                                            const DepthBounds& bounds);
        [[nodiscard]] Status setStencilReference(QueueManager& queues,
                                                 detail::BackendCommandBuffer& commandBuffer,
                                                 StencilFace face,
                                                 std::uint32_t reference);
        [[nodiscard]] Status setShadingRate(QueueManager& queues,
                                            detail::BackendCommandBuffer& commandBuffer,
                                            ShadingRate rate,
                                            const ShadingRateCombiners& combiners);
        [[nodiscard]] Status setShadingRateImage(QueueManager& queues,
                                                 ResourceManager& resources,
                                                 detail::BackendCommandBuffer& commandBuffer,
                                                 TextureHandle image);
        [[nodiscard]] Status bindVertexBuffers(QueueManager& queues,
                                               ResourceManager& resources,
                                               detail::BackendCommandBuffer& commandBuffer,
                                               std::uint32_t firstSlot,
                                               std::span<const VertexBufferBinding> bindings);
        [[nodiscard]] Status bindStreamOutputBuffers(QueueManager& queues,
                                                     ResourceManager& resources,
                                                     detail::BackendCommandBuffer& commandBuffer,
                                                     std::uint32_t firstSlot,
                                                     std::span<const StreamOutputBufferBinding> bindings);
        [[nodiscard]] Status bindIndexBuffer(QueueManager& queues,
                                             ResourceManager& resources,
                                             detail::BackendCommandBuffer& commandBuffer,
                                             BufferHandle buffer,
                                             IndexFormat format,
                                             std::uint64_t offset);
        [[nodiscard]] Status draw(QueueManager& queues,
                                  detail::BackendCommandBuffer& commandBuffer,
                                  std::uint32_t vertexCount,
                                  std::uint32_t instanceCount,
                                  std::uint32_t firstVertex,
                                  std::uint32_t firstInstance);
        [[nodiscard]] Status drawIndexed(QueueManager& queues,
                                         detail::BackendCommandBuffer& commandBuffer,
                                         std::uint32_t indexCount,
                                         std::uint32_t instanceCount,
                                         std::uint32_t firstIndex,
                                         std::int32_t baseVertex,
                                         std::uint32_t firstInstance);
        [[nodiscard]] Status dispatchMesh(QueueManager& queues,
                                          detail::BackendCommandBuffer& commandBuffer,
                                          std::uint32_t groupCountX,
                                          std::uint32_t groupCountY,
                                          std::uint32_t groupCountZ);
        [[nodiscard]] Status dispatchMeshIndirect(QueueManager& queues,
                                                  ResourceManager& resources,
                                                  detail::BackendCommandBuffer& commandBuffer,
                                                  const IndirectMeshDispatchDesc& desc);
        [[nodiscard]] Status drawIndirect(QueueManager& queues,
                                          ResourceManager& resources,
                                          detail::BackendCommandBuffer& commandBuffer,
                                          const IndirectDrawDesc& desc);
        [[nodiscard]] Status drawIndexedIndirect(QueueManager& queues,
                                                 ResourceManager& resources,
                                                 detail::BackendCommandBuffer& commandBuffer,
                                                 const IndirectDrawDesc& desc);
        [[nodiscard]] Result<NativeShaderModule> nativeShaderModule(ShaderModuleHandle shaderModule);
        [[nodiscard]] Result<NativePipelineLayout> nativePipelineLayout(PipelineLayoutHandle pipelineLayout);

    private:
        [[nodiscard]] Status executeIndirect(QueueManager& queues,
                                             ResourceManager& resources,
                                             detail::BackendCommandBuffer& commandBuffer,
                                             const IndirectDrawDesc& desc,
                                             bool indexed);

        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace rachis::d3d12
