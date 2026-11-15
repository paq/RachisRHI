#pragma once

#include "Internal/VulkanBindings.h"
#include "Internal/VulkanCommands.h"
#include "Internal/VulkanPipelines.h"
#include "Internal/VulkanResources.h"
#include "Internal/VulkanRuntime.h"
#include "Internal/VulkanShaders.h"
#include "Rachis/Device.h"
#include "Rachis/Resource.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <vector>
#include <volk.h>

namespace rachis::vulkan
{

    class VulkanImageUtilities final
    {
    public:
        VulkanImageUtilities() = default;
        ~VulkanImageUtilities();

        VulkanImageUtilities(const VulkanImageUtilities&) = delete;
        VulkanImageUtilities& operator=(const VulkanImageUtilities&) = delete;
        VulkanImageUtilities(VulkanImageUtilities&&) = delete;
        VulkanImageUtilities& operator=(VulkanImageUtilities&&) = delete;

        void initialize(const VulkanPhysicalDeviceRecord* record,
                        const DeviceCapabilities* capabilities,
                        VulkanResourceAllocator* resources,
                        VulkanShaderManager* shaderManager,
                        VulkanBindingManager* bindings,
                        VulkanPipelineManager* pipelineManager) noexcept;
        void shutdown() noexcept;

        [[nodiscard]] Status resolveTexture(VulkanCommandBuffer& commandBuffer, const TextureResolve& resolve);
        [[nodiscard]] Status generateMipmaps(VulkanCommandBuffer& commandBuffer,
                                             const TextureMipmapGeneration& generation);
#ifdef RACHIS_TESTING
        [[nodiscard]] std::size_t mipmapBindingPoolCount() const noexcept
        {
            const std::scoped_lock lock(mipmapMutex_);
            return bindingPools_.size();
        }
#endif

    private:
        struct MipmapPipelineSnapshot final
        {
            BindingLayoutHandle bindingLayout{};
            SamplerHandle sampler{};
            BufferHandle encodeOff{};
            BufferHandle encodeOn{};
            std::array<PipelineHandle, 3> computePipelines{};
        };
        struct MipmapLevelInputs;

        [[nodiscard]] MipmapPipelineSnapshot mipmapPipelineSnapshotLocked() const noexcept;
        [[nodiscard]] Result<MipmapPipelineSnapshot> ensureMipmapPipelines();
        [[nodiscard]] Result<BindingSetHandle> allocateMipmapBindingSet(BindingLayoutHandle layout,
                                                                        std::span<const BindingWrite> writes);
        [[nodiscard]] Status recordMipmapLevel(VulkanCommandBuffer& commandBuffer,
                                               const MipmapPipelineSnapshot& pipelines,
                                               const MipmapLevelInputs& inputs,
                                               std::uint32_t destinationMip);
        void destroyMipmapPipelinesLocked() noexcept;

        const VulkanPhysicalDeviceRecord* record_ = nullptr;
        const DeviceCapabilities* capabilities_ = nullptr;
        VulkanResourceAllocator* resources_ = nullptr;
        VulkanShaderManager* shaderManager_ = nullptr;
        VulkanBindingManager* bindings_ = nullptr;
        VulkanPipelineManager* pipelineManager_ = nullptr;
        mutable std::mutex mipmapMutex_{};
        bool mipmapsReady_ = false;
        BindingLayoutHandle bindingLayout_{};
        PipelineLayoutHandle pipelineLayout_{};
        std::vector<BindingPoolHandle> bindingPools_{};
        std::size_t activeBindingPoolIndex_ = 0;
        SamplerHandle sampler_{};
        BufferHandle encodeOff_{};
        BufferHandle encodeOn_{};
        std::array<ShaderModuleHandle, 3> shaderModules_{};
        std::array<PipelineHandle, 3> computePipelines_{};
    };

} // namespace rachis::vulkan
