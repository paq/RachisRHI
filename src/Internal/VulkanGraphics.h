#pragma once

#include "Internal/VulkanBindings.h"
#include "Internal/VulkanCommands.h"
#include "Internal/VulkanIndirectDispatch.h"
#include "Internal/VulkanPipelines.h"
#include "Internal/VulkanResources.h"
#include "Rachis/Device.h"
#include "Rachis/Query.h"
#include "Rachis/Rendering.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <vector>
#include <volk.h>

namespace rachis::vulkan
{

    struct VulkanRenderingScope final
    {
        std::uint32_t width = 1;
        std::uint32_t height = 1;
        std::uint32_t sampleCount = 1;
        std::uint32_t layerCount = 1;
        std::uint32_t viewMask = 0;
        Format depthStencilFormat = Format::unknown;
        std::vector<Format> colorFormats;
        std::vector<std::shared_ptr<void>> attachments;
        std::vector<VkRenderingAttachmentInfo> colorAttachments;
        std::optional<VkRenderingAttachmentInfo> depthAttachment;
        std::optional<VkRenderingAttachmentInfo> stencilAttachment;
        VkImageView shadingRateView = VK_NULL_HANDLE;
        std::shared_ptr<void> shadingRateKeepAlive;
        bool scissorsEmpty = false;
    };

    class VulkanGraphicsEncoder final
    {
    public:
        void initialize(VkDevice device,
                        const VolkDeviceTable* table,
                        const DeviceCapabilities* capabilities,
                        VulkanResourceAllocator* resources,
                        VulkanPipelineManager* pipelines,
                        VulkanBindingManager* bindings,
                        bool depthBiasClamp = true,
                        VulkanBufferBindingLimits bufferLimits = {},
                        std::uint32_t maxDrawIndirectCount = std::numeric_limits<std::uint32_t>::max(),
                        bool clipScissorsToRenderArea = false) noexcept;

        [[nodiscard]] Status beginRendering(VulkanCommandBuffer& commandBuffer,
                                            const RenderingDesc& desc,
                                            const std::function<void()>& beforeNativeBegin = {});
        [[nodiscard]] Status endRendering(VulkanCommandBuffer& commandBuffer);
        [[nodiscard]] Status clearColorAttachment(VulkanCommandBuffer& commandBuffer,
                                                  std::uint32_t attachmentIndex,
                                                  const ClearColor& clearValue,
                                                  std::span<const ScissorRect> rectangles);
        [[nodiscard]] Status clearDepthStencilAttachment(VulkanCommandBuffer& commandBuffer,
                                                         DepthStencilClearAspect aspects,
                                                         float depth,
                                                         std::uint32_t stencil,
                                                         std::span<const ScissorRect> rectangles);
        [[nodiscard]] Status bindGraphicsPipeline(VulkanCommandBuffer& commandBuffer, PipelineHandle pipeline);
        [[nodiscard]] Status bindGraphicsBindingSets(VulkanCommandBuffer& commandBuffer,
                                                     std::uint32_t firstSet,
                                                     std::span<const BindingSetHandle> bindingSets,
                                                     std::span<const std::uint32_t> dynamicOffsets);
        [[nodiscard]] Status setGraphicsPushConstants(VulkanCommandBuffer& commandBuffer,
                                                      ShaderStage visibility,
                                                      std::uint32_t offset,
                                                      std::span<const std::byte> data);
        [[nodiscard]] Status setViewports(VulkanCommandBuffer& commandBuffer, std::span<const Viewport> viewports);
        [[nodiscard]] Status setScissors(VulkanCommandBuffer& commandBuffer, std::span<const ScissorRect> scissors);
        [[nodiscard]] Status setBlendConstants(VulkanCommandBuffer& commandBuffer, const BlendConstants& constants);
        [[nodiscard]] Status setStencilReference(VulkanCommandBuffer& commandBuffer,
                                                 StencilFace face,
                                                 std::uint32_t reference);
        [[nodiscard]] Status setDepthBias(VulkanCommandBuffer& commandBuffer, const DepthBias& bias);
        [[nodiscard]] Status setDepthBounds(VulkanCommandBuffer& commandBuffer, const DepthBounds& bounds);
        [[nodiscard]] Status bindVertexBuffers(VulkanCommandBuffer& commandBuffer,
                                               std::uint32_t firstSlot,
                                               std::span<const VertexBufferBinding> bindings);
        [[nodiscard]] Status bindIndexBuffer(VulkanCommandBuffer& commandBuffer,
                                             BufferHandle buffer,
                                             IndexFormat format,
                                             std::uint64_t offset);
        [[nodiscard]] Status draw(VulkanCommandBuffer& commandBuffer,
                                  std::uint32_t vertexCount,
                                  std::uint32_t instanceCount,
                                  std::uint32_t firstVertex,
                                  std::uint32_t firstInstance);
        [[nodiscard]] Status drawIndexed(VulkanCommandBuffer& commandBuffer,
                                         std::uint32_t indexCount,
                                         std::uint32_t instanceCount,
                                         std::uint32_t firstIndex,
                                         std::int32_t baseVertex,
                                         std::uint32_t firstInstance);
        [[nodiscard]] Status drawIndirect(VulkanCommandBuffer& commandBuffer, const IndirectDrawDesc& desc);
        [[nodiscard]] Status drawIndexedIndirect(VulkanCommandBuffer& commandBuffer, const IndirectDrawDesc& desc);
        [[nodiscard]] Status setShadingRate(VulkanCommandBuffer& commandBuffer,
                                            ShadingRate rate,
                                            const ShadingRateCombiners& combiners);
        [[nodiscard]] Status setShadingRateImage(VulkanCommandBuffer& commandBuffer, TextureHandle image);
        [[nodiscard]] Status dispatchMesh(VulkanCommandBuffer& commandBuffer,
                                          std::uint32_t groupCountX,
                                          std::uint32_t groupCountY,
                                          std::uint32_t groupCountZ);
        [[nodiscard]] Status dispatchMeshIndirect(VulkanCommandBuffer& commandBuffer,
                                                  const IndirectMeshDispatchDesc& desc);
        [[nodiscard]] Status bindComputePipeline(VulkanCommandBuffer& commandBuffer, PipelineHandle pipeline);
        [[nodiscard]] Status bindComputeBindingSets(VulkanCommandBuffer& commandBuffer,
                                                    std::uint32_t firstSet,
                                                    std::span<const BindingSetHandle> bindingSets,
                                                    std::span<const std::uint32_t> dynamicOffsets);
        [[nodiscard]] Status setComputePushConstants(VulkanCommandBuffer& commandBuffer,
                                                     std::uint32_t offset,
                                                     std::span<const std::byte> data);
        [[nodiscard]] Status dispatch(VulkanCommandBuffer& commandBuffer,
                                      std::uint32_t groupCountX,
                                      std::uint32_t groupCountY,
                                      std::uint32_t groupCountZ);
        [[nodiscard]] Status dispatchIndirect(VulkanCommandBuffer& commandBuffer, const IndirectDispatchDesc& desc);
        [[nodiscard]] Status beginConditionalExecution(VulkanCommandBuffer& commandBuffer,
                                                       const ConditionalExecutionDesc& desc);
        [[nodiscard]] Status endConditionalExecution(VulkanCommandBuffer& commandBuffer);
        void setCountedDispatchPass(VulkanCountedDispatchPass* pass) noexcept;
        [[nodiscard]] Status bindRayTracingPipeline(VulkanCommandBuffer& commandBuffer, PipelineHandle pipeline);
        [[nodiscard]] Status bindRayTracingBindingSets(VulkanCommandBuffer& commandBuffer,
                                                       std::uint32_t firstSet,
                                                       std::span<const BindingSetHandle> bindingSets,
                                                       std::span<const std::uint32_t> dynamicOffsets);
        [[nodiscard]] Status setRayTracingPushConstants(VulkanCommandBuffer& commandBuffer,
                                                        std::uint32_t offset,
                                                        std::span<const std::byte> data);
        [[nodiscard]] Status dispatchRays(VulkanCommandBuffer& commandBuffer, const RayTracingDispatchDesc& desc);
        [[nodiscard]] Status dispatchRaysIndirect(VulkanCommandBuffer& commandBuffer,
                                                  const RayTracingIndirectDispatchDesc& desc);

    private:
        [[nodiscard]] std::vector<std::shared_ptr<void>>* retainSink(VulkanCommandBuffer& commandBuffer) const noexcept;
        [[nodiscard]] Status requireRendering(const VulkanCommandBuffer& commandBuffer) const;
        [[nodiscard]] Status requireBoundPipeline(const VulkanCommandBuffer& commandBuffer) const;
        [[nodiscard]] Status requireMeshDispatchState(const VulkanCommandBuffer& commandBuffer) const;
        [[nodiscard]] bool drawScissoredOut(const VulkanCommandBuffer& commandBuffer) const noexcept;
        void beginNativeRendering(VulkanCommandBuffer& commandBuffer, VulkanRenderingScope& scope) const noexcept;
        [[nodiscard]] Status applyShadingRateImage(VulkanCommandBuffer& commandBuffer,
                                                   VulkanRenderingScope& scope,
                                                   TextureHandle image);
        void suspendConditionalRendering(VulkanCommandBuffer& commandBuffer) const noexcept;
        void resumeConditionalRendering(VulkanCommandBuffer& commandBuffer) const noexcept;
        void restoreComputeBindings(VulkanCommandBuffer& commandBuffer) const noexcept;

        VkDevice device_ = VK_NULL_HANDLE;
        const VolkDeviceTable* table_ = nullptr;
        const DeviceCapabilities* capabilities_ = nullptr;
        VulkanResourceAllocator* resources_ = nullptr;
        VulkanPipelineManager* pipelines_ = nullptr;
        VulkanBindingManager* bindings_ = nullptr;
        VulkanCountedDispatchPass* countedDispatch_ = nullptr;
        bool depthBiasClampSupported_ = true;
        VulkanBufferBindingLimits bufferLimits_{};
        std::uint32_t maxDrawIndirectCount_ = std::numeric_limits<std::uint32_t>::max();
        bool clipScissorsToRenderArea_ = false;
    };

} // namespace rachis::vulkan
