#include "Internal/VulkanImageUtilities.h"

#include "Common/VectorReserve.h"
#include "Internal/VulkanDiagnostics.h"
#include "Internal/VulkanImageBarriers.h"
#include "RachisGenerateMipmaps1D.spirv.h"
#include "RachisGenerateMipmaps2D.spirv.h"
#include "RachisGenerateMipmaps3D.spirv.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rachis::vulkan
{
    namespace
    {

        struct BarrierScope final
        {
            PipelineStage stages = PipelineStage::none;
            Access access = Access::none;
        };

        [[nodiscard]] constexpr std::size_t dimensionIndex(TextureDimension dimension) noexcept
        {
            return static_cast<std::size_t>(dimension);
        }

        [[nodiscard]] constexpr std::array<std::uint32_t, 3> threadGroupSize(TextureDimension dimension) noexcept
        {
            switch (dimension)
            {
            case TextureDimension::texture1D:
                return {64, 1, 1};
            case TextureDimension::texture2D:
                return {8, 8, 1};
            case TextureDimension::texture3D:
                return {4, 4, 4};
            }
            return {1, 1, 1};
        }

        [[nodiscard]] constexpr TextureViewDimension mipmapViewDimension(TextureDimension dimension) noexcept
        {
            switch (dimension)
            {
            case TextureDimension::texture1D:
                return TextureViewDimension::texture1DArray;
            case TextureDimension::texture2D:
                return TextureViewDimension::texture2DArray;
            case TextureDimension::texture3D:
                return TextureViewDimension::texture3D;
            }
            return TextureViewDimension::texture2D;
        }

        [[nodiscard]] constexpr std::uint32_t mipDimension(std::uint32_t size, std::uint32_t mipLevel) noexcept
        {
            return std::max(1U, size >> mipLevel);
        }

        [[nodiscard]] constexpr bool layoutCompatible(TextureUsage usage, TextureLayout layout) noexcept
        {
            switch (layout)
            {
            case TextureLayout::common:
                return true;
            case TextureLayout::shaderResource:
                return hasAny(usage, TextureUsage::sampled);
            case TextureLayout::unorderedAccess:
                return hasAny(usage, TextureUsage::storage);
            case TextureLayout::renderTarget:
                return hasAny(usage, TextureUsage::renderTarget);
            case TextureLayout::copySource:
                return hasAny(usage, TextureUsage::copySource);
            case TextureLayout::copyDestination:
                return hasAny(usage, TextureUsage::copyDestination);
            case TextureLayout::resolveSource:
            case TextureLayout::resolveDestination:
                return hasAny(usage, TextureUsage::renderTarget | TextureUsage::depthStencil);
            default:
                return false;
            }
        }

        [[nodiscard]] constexpr PipelineStage shaderReadStages(QueueType queueType) noexcept
        {
            if (queueType == QueueType::compute)
            {
                return PipelineStage::computeShader | PipelineStage::rayTracingShader;
            }
            return PipelineStage::vertexShader | PipelineStage::pixelShader | PipelineStage::computeShader |
                   PipelineStage::geometryShader | PipelineStage::hullShader | PipelineStage::domainShader |
                   PipelineStage::taskShader | PipelineStage::meshShader | PipelineStage::rayTracingShader;
        }

        [[nodiscard]] constexpr std::optional<BarrierScope> barrierScope(TextureLayout layout,
                                                                         QueueType queueType) noexcept
        {
            switch (layout)
            {
            case TextureLayout::common:
                return BarrierScope{PipelineStage::all, Access::common};
            case TextureLayout::shaderResource:
                return BarrierScope{shaderReadStages(queueType), Access::shaderResource};
            case TextureLayout::unorderedAccess:
                return BarrierScope{PipelineStage::computeShader, Access::unorderedAccess};
            case TextureLayout::renderTarget:
                return BarrierScope{PipelineStage::renderTarget, Access::renderTarget};
            case TextureLayout::copySource:
                return BarrierScope{PipelineStage::copy, Access::copySource};
            case TextureLayout::copyDestination:
                return BarrierScope{PipelineStage::copy, Access::copyDestination};
            case TextureLayout::resolveSource:
                return BarrierScope{PipelineStage::resolve, Access::resolveSource};
            case TextureLayout::resolveDestination:
                return BarrierScope{PipelineStage::resolve, Access::resolveDestination};
            default:
                return std::nullopt;
            }
        }

        [[nodiscard]] Result<std::uint32_t> selectedCount(std::uint32_t first,
                                                          std::uint32_t requested,
                                                          std::uint32_t available,
                                                          std::string_view subject)
        {
            if (first >= available)
            {
                return std::unexpected(
                    Error{ErrorCode::invalidArgument, 0, std::string(subject) + " starts out of range."});
            }
            const std::uint32_t count = requested == remainingSubresources ? available - first : requested;
            if (count == 0 || count > available - first)
            {
                return std::unexpected(
                    Error{ErrorCode::invalidArgument, 0, std::string(subject) + " has an invalid count."});
            }
            return count;
        }

        [[nodiscard]] TextureBarrier textureBarrier(TextureHandle texture,
                                                    TextureSubresourceRange range,
                                                    TextureLayout before,
                                                    BarrierScope beforeScope,
                                                    TextureLayout after,
                                                    BarrierScope afterScope)
        {
            return TextureBarrier{
                .texture = texture,
                .stagesBefore = beforeScope.stages,
                .stagesAfter = afterScope.stages,
                .accessBefore = beforeScope.access,
                .accessAfter = afterScope.access,
                .layoutBefore = before,
                .layoutAfter = after,
                .subresources = range,
            };
        }

        [[nodiscard]] constexpr Format linearViewFormat(Format format) noexcept
        {
            switch (format)
            {
            case Format::rgba8UnormSrgb:
                return Format::rgba8Unorm;
            case Format::bgra8UnormSrgb:
                return Format::bgra8Unorm;
            default:
                return Format::unknown;
            }
        }

        [[nodiscard]] constexpr bool isDepthFormat(Format format) noexcept
        {
            return format == Format::d32Float || format == Format::d24UnormS8Uint;
        }

        [[nodiscard]] constexpr VkResolveModeFlagBits nativeResolveMode(ResolveMode mode) noexcept
        {
            return mode == ResolveMode::maximum ? VK_RESOLVE_MODE_MAX_BIT : VK_RESOLVE_MODE_MIN_BIT;
        }

        [[nodiscard]] constexpr ResolveModeMask portableMinMax(VkResolveModeFlags flags) noexcept
        {
            ResolveModeMask mask = ResolveModeMask::none;
            if ((flags & VK_RESOLVE_MODE_MIN_BIT) != 0)
            {
                mask = mask | ResolveModeMask::minimum;
            }
            if ((flags & VK_RESOLVE_MODE_MAX_BIT) != 0)
            {
                mask = mask | ResolveModeMask::maximum;
            }
            return mask;
        }

        [[nodiscard]] std::span<const std::byte> spirvBytes(const unsigned char* data, std::size_t size) noexcept
        {
            return {reinterpret_cast<const std::byte*>(data), size};
        }

        [[nodiscard]] std::vector<std::shared_ptr<void>>* retainSink(VulkanCommandBuffer& commandBuffer) noexcept
        {
            if (commandBuffer.owner() == nullptr ||
                hasAny(commandBuffer.owner()->flags, CommandPoolFlag::unretainedResources))
            {
                return nullptr;
            }
            return &commandBuffer.retained();
        }

        [[nodiscard]] const VolkDeviceTable* deviceTable(const VulkanCommandBuffer& commandBuffer) noexcept
        {
            return commandBuffer.owner() != nullptr ? commandBuffer.owner()->table : nullptr;
        }

        [[nodiscard]] constexpr VkImageAspectFlags nativeAspect(TextureAspect aspect) noexcept
        {
            switch (aspect)
            {
            case TextureAspect::depth:
                return VK_IMAGE_ASPECT_DEPTH_BIT;
            case TextureAspect::stencil:
                return VK_IMAGE_ASPECT_STENCIL_BIT;
            case TextureAspect::color:
                break;
            }
            return VK_IMAGE_ASPECT_COLOR_BIT;
        }

        [[nodiscard]] constexpr validation_detail::CanonicalTransferLayout canonicalResolveLayout(
            VkImageLayout layout) noexcept
        {
            switch (layout)
            {
            case VK_IMAGE_LAYOUT_UNDEFINED:
                return validation_detail::CanonicalTransferLayout::undefined;
            case VK_IMAGE_LAYOUT_GENERAL:
                return validation_detail::CanonicalTransferLayout::general;
            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                return validation_detail::CanonicalTransferLayout::transferSrc;
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                return validation_detail::CanonicalTransferLayout::transferDst;
            default:
                return validation_detail::CanonicalTransferLayout::other;
            }
        }

        void recordImageBarrier(const VolkDeviceTable& table,
                                VkCommandBuffer command,
                                VkImage image,
                                VkImageAspectFlags aspect,
                                std::uint32_t mipLevel,
                                std::uint32_t arrayLayer,
                                VkPipelineStageFlags2 srcStage,
                                VkAccessFlags2 srcAccess,
                                VkPipelineStageFlags2 dstStage,
                                VkAccessFlags2 dstAccess,
                                VkImageLayout oldLayout,
                                VkImageLayout newLayout) noexcept
        {
            VkImageMemoryBarrier2 barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .pNext = {},
                .srcStageMask = srcStage,
                .srcAccessMask = srcAccess,
                .dstStageMask = dstStage,
                .dstAccessMask = dstAccess,
                .oldLayout = oldLayout,
                .newLayout = newLayout,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange =
                    {
                        .aspectMask = aspect,
                        .baseMipLevel = mipLevel,
                        .levelCount = 1,
                        .baseArrayLayer = arrayLayer,
                        .layerCount = 1,
                    },
            };
            ensureImageLayoutTransitionSourceScope(barrier);
            const VkDependencyInfo dependency{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pNext = {},
                .dependencyFlags = {},
                .memoryBarrierCount = {},
                .pMemoryBarriers = {},
                .bufferMemoryBarrierCount = {},
                .pBufferMemoryBarriers = {},
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &barrier,
            };
            table.vkCmdPipelineBarrier2(command, &dependency);
        }

    } // namespace

    VulkanImageUtilities::~VulkanImageUtilities()
    {
        shutdown();
    }

    void VulkanImageUtilities::initialize(const VulkanPhysicalDeviceRecord* record,
                                          const DeviceCapabilities* capabilities,
                                          VulkanResourceAllocator* resources,
                                          VulkanShaderManager* shaderManager,
                                          VulkanBindingManager* bindings,
                                          VulkanPipelineManager* pipelineManager) noexcept
    {
        record_ = record;
        capabilities_ = capabilities;
        resources_ = resources;
        shaderManager_ = shaderManager;
        bindings_ = bindings;
        pipelineManager_ = pipelineManager;
    }

    void VulkanImageUtilities::destroyMipmapPipelinesLocked() noexcept
    {
        if (pipelineManager_ != nullptr)
        {
            for (auto& pipeline : computePipelines_)
            {
                if (pipeline.isValid())
                {
                    static_cast<void>(pipelineManager_->destroyPipeline(pipeline));
                    pipeline = {};
                }
            }
            if (pipelineLayout_.isValid() && bindings_ != nullptr)
            {
                static_cast<void>(bindings_->destroyPipelineLayout(pipelineLayout_));
                pipelineLayout_ = {};
            }
        }
        if (shaderManager_ != nullptr)
        {
            for (auto& shader : shaderModules_)
            {
                if (shader.isValid())
                {
                    static_cast<void>(shaderManager_->destroyShaderModule(shader));
                    shader = {};
                }
            }
        }
        if (bindings_ != nullptr)
        {
            for (auto& pool : bindingPools_)
            {
                if (pool.isValid())
                {
                    static_cast<void>(bindings_->destroyBindingPool(pool));
                }
            }
            bindingPools_.clear();
            activeBindingPoolIndex_ = 0;
            if (bindingLayout_.isValid())
            {
                static_cast<void>(bindings_->destroyBindingLayout(bindingLayout_));
                bindingLayout_ = {};
            }
        }
        if (resources_ != nullptr)
        {
            if (sampler_.isValid())
            {
                static_cast<void>(resources_->destroySampler(sampler_));
                sampler_ = {};
            }
            if (encodeOff_.isValid())
            {
                static_cast<void>(resources_->destroyBuffer(encodeOff_));
                encodeOff_ = {};
            }
            if (encodeOn_.isValid())
            {
                static_cast<void>(resources_->destroyBuffer(encodeOn_));
                encodeOn_ = {};
            }
        }
        mipmapsReady_ = false;
    }

    void VulkanImageUtilities::shutdown() noexcept
    {
        {
            const std::scoped_lock lock(mipmapMutex_);
            destroyMipmapPipelinesLocked();
        }
        record_ = nullptr;
        capabilities_ = nullptr;
        resources_ = nullptr;
        shaderManager_ = nullptr;
        bindings_ = nullptr;
        pipelineManager_ = nullptr;
    }

    namespace
    {

        constexpr std::uint32_t mipmapPoolSetsPerSlab = 64;

        struct MipmapResourceManagers final
        {
            VulkanResourceAllocator& resources;
            VulkanShaderManager& shaderManager;
            VulkanBindingManager& bindings;
            VulkanPipelineManager& pipelineManager;
        };

        struct MipmapPipelineResources final
        {
            BufferHandle encodeOff{};
            BufferHandle encodeOn{};
            SamplerHandle sampler{};
            BindingLayoutHandle bindingLayout{};
            BindingPoolHandle bindingPool{};
            PipelineLayoutHandle pipelineLayout{};
            std::array<ShaderModuleHandle, 3> shaderModules{};
            std::array<PipelineHandle, 3> computePipelines{};
        };

        void destroyMipmapPipelineResources(const MipmapResourceManagers& managers,
                                            const MipmapPipelineResources& created) noexcept
        {
            for (const auto pipeline : created.computePipelines)
            {
                if (pipeline.isValid())
                {
                    static_cast<void>(managers.pipelineManager.destroyPipeline(pipeline));
                }
            }
            for (const auto shader : created.shaderModules)
            {
                if (shader.isValid())
                {
                    static_cast<void>(managers.shaderManager.destroyShaderModule(shader));
                }
            }
            if (created.pipelineLayout.isValid())
            {
                static_cast<void>(managers.bindings.destroyPipelineLayout(created.pipelineLayout));
            }
            if (created.bindingPool.isValid())
            {
                static_cast<void>(managers.bindings.destroyBindingPool(created.bindingPool));
            }
            if (created.bindingLayout.isValid())
            {
                static_cast<void>(managers.bindings.destroyBindingLayout(created.bindingLayout));
            }
            if (created.sampler.isValid())
            {
                static_cast<void>(managers.resources.destroySampler(created.sampler));
            }
            if (created.encodeOn.isValid())
            {
                static_cast<void>(managers.resources.destroyBuffer(created.encodeOn));
            }
            if (created.encodeOff.isValid())
            {
                static_cast<void>(managers.resources.destroyBuffer(created.encodeOff));
            }
        }

        class MipmapPipelineRollback final
        {
        public:
            MipmapPipelineRollback(const MipmapResourceManagers& managers,
                                   const MipmapPipelineResources& created) noexcept
                : managers_(managers), created_(created)
            {
            }

            MipmapPipelineRollback(const MipmapPipelineRollback&) = delete;
            MipmapPipelineRollback& operator=(const MipmapPipelineRollback&) = delete;
            MipmapPipelineRollback(MipmapPipelineRollback&&) = delete;
            MipmapPipelineRollback& operator=(MipmapPipelineRollback&&) = delete;

            ~MipmapPipelineRollback()
            {
                if (!committed_)
                {
                    destroyMipmapPipelineResources(managers_, created_);
                }
            }

            void commit() noexcept
            {
                committed_ = true;
            }

        private:
            const MipmapResourceManagers& managers_;
            const MipmapPipelineResources& created_;
            bool committed_ = false;
        };

        [[nodiscard]] Status createMipmapEncodeConstants(VulkanResourceAllocator& resources,
                                                         MipmapPipelineResources& created)
        {
            constexpr std::uint64_t uniformBytes = 256;
            auto off = resources.createBuffer(BufferDesc{
                .size = uniformBytes,
                .usage = BufferUsage::constant | BufferUsage::copyDestination,
                .memoryType = MemoryType::upload,
                .label = "Vulkan mipmap encode-off constants",
            });
            if (off)
            {
                created.encodeOff = *off;
            }
            auto on = resources.createBuffer(BufferDesc{
                .size = uniformBytes,
                .usage = BufferUsage::constant | BufferUsage::copyDestination,
                .memoryType = MemoryType::upload,
                .label = "Vulkan mipmap encode-on constants",
            });
            if (on)
            {
                created.encodeOn = *on;
            }
            if (!off || !on)
            {
                return std::unexpected(
                    Error{ErrorCode::unsupported, 0, "Mipmap generation constant buffers are unavailable."});
            }

            const std::uint32_t offValue = 0;
            const std::uint32_t onValue = 1;
            if (auto written = resources.writeBuffer(*off, 0, std::as_bytes(std::span(&offValue, 1))); !written)
            {
                return written;
            }
            return resources.writeBuffer(*on, 0, std::as_bytes(std::span(&onValue, 1)));
        }

        [[nodiscard]] Result<SamplerHandle> createMipmapSampler(VulkanResourceAllocator& resources)
        {
            return resources.createSampler(SamplerDesc{
                .minFilter = FilterMode::linear,
                .magFilter = FilterMode::linear,
                .mipFilter = MipmapFilterMode::nearest,
                .label = "Vulkan mipmap linear clamp",
            });
        }

        [[nodiscard]] Result<BindingLayoutHandle> createMipmapBindingLayout(VulkanBindingManager& bindings)
        {
            const std::array layoutEntries{
                BindingLayoutEntry{0, BindingType::uniformBuffer, ShaderStage::compute},
                BindingLayoutEntry{1, BindingType::sampledTexture, ShaderStage::compute},
                BindingLayoutEntry{2, BindingType::storageTexture, ShaderStage::compute},
                BindingLayoutEntry{3, BindingType::sampler, ShaderStage::compute},
            };
            return bindings.createBindingLayout(BindingLayoutDesc{
                .entries = layoutEntries,
                .label = "Vulkan mipmap generation layout",
            });
        }

        [[nodiscard]] Result<BindingPoolHandle> createMipmapBindingPoolSlab(VulkanBindingManager& bindings)
        {
            const std::array poolSizes{
                BindingPoolSize{BindingType::uniformBuffer, mipmapPoolSetsPerSlab},
                BindingPoolSize{BindingType::sampledTexture, mipmapPoolSetsPerSlab},
                BindingPoolSize{BindingType::storageTexture, mipmapPoolSetsPerSlab},
                BindingPoolSize{BindingType::sampler, mipmapPoolSetsPerSlab},
            };
            return bindings.createBindingPool(BindingPoolDesc{
                .maxBindingSets = mipmapPoolSetsPerSlab,
                .sizes = poolSizes,
                .label = "Vulkan mipmap generation pool",
            });
        }

        [[nodiscard]] Result<PipelineLayoutHandle> createMipmapPipelineLayout(VulkanBindingManager& bindings,
                                                                              BindingLayoutHandle bindingLayout)
        {
            const std::array bindingLayouts{bindingLayout};
            return bindings.createPipelineLayout(PipelineLayoutDesc{
                .bindingLayouts = bindingLayouts,
                .label = "Vulkan mipmap generation pipeline layout",
            });
        }

        [[nodiscard]] Status createMipmapComputePipelines(VulkanShaderManager& shaderManager,
                                                          VulkanPipelineManager& pipelineManager,
                                                          MipmapPipelineResources& created)
        {
            static constexpr std::array bindings{
                ShaderBindingRequirement{0, 0, ShaderResourceType::uniformBuffer, 1},
                ShaderBindingRequirement{0, 1, ShaderResourceType::sampledTexture, 1},
                ShaderBindingRequirement{0, 2, ShaderResourceType::storageTexture, 1},
                ShaderBindingRequirement{0, 3, ShaderResourceType::sampler, 1},
            };
            const std::array programs{
                std::pair{spirvBytes(g_rachisGenerateMipmaps1DSpirv, sizeof(g_rachisGenerateMipmaps1DSpirv)),
                          ShaderWorkgroupSize{64, 1, 1}},
                std::pair{spirvBytes(g_rachisGenerateMipmaps2DSpirv, sizeof(g_rachisGenerateMipmaps2DSpirv)),
                          ShaderWorkgroupSize{8, 8, 1}},
                std::pair{spirvBytes(g_rachisGenerateMipmaps3DSpirv, sizeof(g_rachisGenerateMipmaps3DSpirv)),
                          ShaderWorkgroupSize{4, 4, 4}},
            };
            for (std::size_t index = 0; index < programs.size(); ++index)
            {
                auto shader = shaderManager.createShaderModule(ShaderModuleDesc{
                    .stage = ShaderStage::compute,
                    .bytecode = programs[index].first,
                    .entryPoint = "main",
                    .label = "Vulkan mipmap generation",
                    .shaderInterface =
                        ShaderInterfaceDesc{
                            .bindings = bindings,
                            .workgroupSize = programs[index].second,
                        },
                });
                if (!shader)
                {
                    return std::unexpected(std::move(shader.error()));
                }
                created.shaderModules[index] = *shader;
                auto pipeline = pipelineManager.createComputePipeline(ComputePipelineDesc{
                    .layout = created.pipelineLayout,
                    .computeShader = *shader,
                    .label = "Vulkan mipmap generation pipeline",
                });
                if (!pipeline)
                {
                    return std::unexpected(std::move(pipeline.error()));
                }
                created.computePipelines[index] = *pipeline;
            }
            return {};
        }

        [[nodiscard]] Status createMipmapPipelineResources(const MipmapResourceManagers& managers,
                                                           MipmapPipelineResources& created)
        {
            if (auto constants = createMipmapEncodeConstants(managers.resources, created); !constants)
            {
                return constants;
            }
            auto sampler = createMipmapSampler(managers.resources);
            if (!sampler)
            {
                return std::unexpected(std::move(sampler.error()));
            }
            created.sampler = *sampler;

            auto bindingLayout = createMipmapBindingLayout(managers.bindings);
            if (!bindingLayout)
            {
                return std::unexpected(std::move(bindingLayout.error()));
            }
            created.bindingLayout = *bindingLayout;

            auto bindingPool = createMipmapBindingPoolSlab(managers.bindings);
            if (!bindingPool)
            {
                return std::unexpected(std::move(bindingPool.error()));
            }
            created.bindingPool = *bindingPool;

            auto pipelineLayout = createMipmapPipelineLayout(managers.bindings, created.bindingLayout);
            if (!pipelineLayout)
            {
                return std::unexpected(std::move(pipelineLayout.error()));
            }
            created.pipelineLayout = *pipelineLayout;

            return createMipmapComputePipelines(managers.shaderManager, managers.pipelineManager, created);
        }

    } // namespace

    VulkanImageUtilities::MipmapPipelineSnapshot VulkanImageUtilities::mipmapPipelineSnapshotLocked() const noexcept
    {
        return MipmapPipelineSnapshot{
            .bindingLayout = bindingLayout_,
            .sampler = sampler_,
            .encodeOff = encodeOff_,
            .encodeOn = encodeOn_,
            .computePipelines = computePipelines_,
        };
    }

    Result<VulkanImageUtilities::MipmapPipelineSnapshot> VulkanImageUtilities::ensureMipmapPipelines()
    {
        const std::scoped_lock lock(mipmapMutex_);
        if (mipmapsReady_)
        {
            return mipmapPipelineSnapshotLocked();
        }
        if (resources_ == nullptr || shaderManager_ == nullptr || bindings_ == nullptr || pipelineManager_ == nullptr)
        {
            return std::unexpected(
                Error{ErrorCode::unsupported, 0, "Mipmap generation is unavailable on this device."});
        }

        const MipmapResourceManagers managers{
            .resources = *resources_,
            .shaderManager = *shaderManager_,
            .bindings = *bindings_,
            .pipelineManager = *pipelineManager_,
        };
        MipmapPipelineResources created{};
        MipmapPipelineRollback rollback{managers, created};
        if (auto built = createMipmapPipelineResources(managers, created); !built)
        {
            return std::unexpected(std::move(built.error()));
        }

        try
        {
            bindingPools_.push_back(created.bindingPool);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to record the Vulkan mipmap descriptor pool.",
            });
        }
        rollback.commit();
        encodeOff_ = created.encodeOff;
        encodeOn_ = created.encodeOn;
        sampler_ = created.sampler;
        bindingLayout_ = created.bindingLayout;
        activeBindingPoolIndex_ = 0;
        pipelineLayout_ = created.pipelineLayout;
        shaderModules_ = created.shaderModules;
        computePipelines_ = created.computePipelines;
        mipmapsReady_ = true;
        return mipmapPipelineSnapshotLocked();
    }

    Result<BindingSetHandle> VulkanImageUtilities::allocateMipmapBindingSet(BindingLayoutHandle layout,
                                                                            std::span<const BindingWrite> writes)
    {
        const std::scoped_lock lock(mipmapMutex_);
        if (!mipmapsReady_ || bindings_ == nullptr || resources_ == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Mipmap generation descriptor pools are not initialized.",
            });
        }

        const auto poolCount = bindingPools_.size();
        for (std::size_t attempt = 0; attempt < poolCount; ++attempt)
        {
            const auto index = (activeBindingPoolIndex_ + attempt) % poolCount;
            auto set = bindings_->createBindingSet(*resources_,
                                                   BindingSetDesc{
                                                       .layout = layout,
                                                       .writes = writes,
                                                       .label = "Vulkan mipmap generation set",
                                                       .pool = bindingPools_[index],
                                                   });
            if (set)
            {
                activeBindingPoolIndex_ = index;
                return set;
            }
            if (set.error().code != ErrorCode::resourceExhausted)
            {
                return set;
            }
        }

        auto slab = createMipmapBindingPoolSlab(*bindings_);
        if (!slab)
        {
            return std::unexpected(std::move(slab.error()));
        }
        try
        {
            detail::reserveAdditional(bindingPools_, 1U);
        }
        catch (const std::bad_alloc&)
        {
            static_cast<void>(bindings_->destroyBindingPool(*slab));
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to grow Vulkan mipmap descriptor pools.",
            });
        }
        auto set = bindings_->createBindingSet(*resources_,
                                               BindingSetDesc{
                                                   .layout = layout,
                                                   .writes = writes,
                                                   .label = "Vulkan mipmap generation set",
                                                   .pool = *slab,
                                               });
        if (!set)
        {
            static_cast<void>(bindings_->destroyBindingPool(*slab));
            return set;
        }
        bindingPools_.push_back(*slab);
        activeBindingPoolIndex_ = bindingPools_.size() - 1U;
        return set;
    }

    namespace
    {

        using NativeTexture = VulkanResourceAllocator::NativeTexture;

        struct TextureResolveOperation final
        {
            const TextureResolve& request;
            const NativeTexture& source;
            const NativeTexture& destination;
        };

        [[nodiscard]] Status validateResolveTextureCompatibility(const TextureResolveOperation& operation)
        {
            const auto& request = operation.request;
            const auto& sourceDesc = operation.source.desc;
            const auto& destinationDesc = operation.destination.desc;
            if (sourceDesc.format != destinationDesc.format ||
                request.sourceSubresource.aspect != request.destinationSubresource.aspect)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture resolve requires identical formats and matching aspects.",
                });
            }
            if (sourceDesc.sampleCount == 1 || destinationDesc.sampleCount != 1 ||
                sourceDesc.dimension != TextureDimension::texture2D ||
                destinationDesc.dimension != TextureDimension::texture2D)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture resolve requires a multisampled 2D source and a single-sampled 2D destination.",
                });
            }

            const TextureUsage requiredUsage =
                isDepthFormat(sourceDesc.format) ? TextureUsage::depthStencil : TextureUsage::renderTarget;
            if (!hasAny(sourceDesc.usage, requiredUsage) || !hasAny(destinationDesc.usage, requiredUsage))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Texture resolve resources must both have matching attachment usage.",
                });
            }
            return {};
        }

        [[nodiscard]] constexpr bool resolvesDepth(const TextureResolve& request) noexcept
        {
            return request.sourceSubresource.aspect == TextureAspect::depth;
        }

        [[nodiscard]] constexpr bool resolvesStencil(const TextureResolve& request) noexcept
        {
            return request.sourceSubresource.aspect == TextureAspect::stencil;
        }

        [[nodiscard]] Status validateResolveModeForAspect(const TextureResolveOperation& operation)
        {
            const auto format = operation.source.desc.format;
            const auto aspect = operation.request.sourceSubresource.aspect;
            const auto mode = operation.request.mode;
            const bool depthStencilFormat = isDepthFormat(format);
            const bool depthResolve = resolvesDepth(operation.request);
            const bool stencilResolve = resolvesStencil(operation.request);

            const bool invalidDepthStencilResolve =
                depthStencilFormat && ((!depthResolve && !stencilResolve) || mode == ResolveMode::average);
            const bool invalidColorResolve =
                !depthStencilFormat && (aspect != TextureAspect::color || mode != ResolveMode::average);
            if (invalidDepthStencilResolve || invalidColorResolve)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Color resolves use average mode; depth and stencil resolves use minimum or maximum mode.",
                });
            }

            const bool stencilMissingFromFormat = format == Format::d32Float && !depthResolve;
            const bool unresolvableFormat = format != Format::d32Float && format != Format::d24UnormS8Uint;
            if (depthStencilFormat && (stencilMissingFromFormat || unresolvableFormat))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The requested depth/stencil format and aspect do not support multisample resolve.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateResolveSubresources(const TextureResolveOperation& operation)
        {
            const auto& sourceSubresource = operation.request.sourceSubresource;
            const auto& destinationSubresource = operation.request.destinationSubresource;
            const auto& sourceDesc = operation.source.desc;
            const auto& destinationDesc = operation.destination.desc;

            const bool subresourcesInRange = sourceSubresource.mipLevel < sourceDesc.mipLevels &&
                                             destinationSubresource.mipLevel < destinationDesc.mipLevels &&
                                             sourceSubresource.arrayLayer < sourceDesc.depthOrArrayLayers &&
                                             destinationSubresource.arrayLayer < destinationDesc.depthOrArrayLayers;
            const bool extentsMatch = mipDimension(sourceDesc.width, sourceSubresource.mipLevel) ==
                                          mipDimension(destinationDesc.width, destinationSubresource.mipLevel) &&
                                      mipDimension(sourceDesc.height, sourceSubresource.mipLevel) ==
                                          mipDimension(destinationDesc.height, destinationSubresource.mipLevel);
            if (!subresourcesInRange || !extentsMatch)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture resolve requires matching source and destination subresource extents.",
                });
            }

            if (!validation_detail::isTransferUseCompatible(canonicalResolveLayout(operation.source.layout),
                                                            validation_detail::SubresourceTransferUse::resolveSource) ||
                !validation_detail::isTransferUseCompatible(
                    canonicalResolveLayout(operation.destination.layout),
                    validation_detail::SubresourceTransferUse::resolveDestination))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A texture resolve requires resolve-source or general and resolve-destination or general "
                    "subresource layouts.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateTextureResolve(const TextureResolveOperation& operation)
        {
            if (auto compatible = validateResolveTextureCompatibility(operation); !compatible)
            {
                return compatible;
            }
            if (auto modeAllowed = validateResolveModeForAspect(operation); !modeAllowed)
            {
                return modeAllowed;
            }
            return validateResolveSubresources(operation);
        }

        [[nodiscard]] VkExtent2D resolvedMipExtent(const TextureResolveOperation& operation) noexcept
        {
            const auto mipLevel = operation.request.sourceSubresource.mipLevel;
            return {
                mipDimension(operation.source.desc.width, mipLevel),
                mipDimension(operation.source.desc.height, mipLevel),
            };
        }

        void recordResolveTextureUse(VulkanResourceAllocator& resources,
                                     VulkanCommandBuffer& commandBuffer,
                                     const TextureResolveOperation& operation)
        {
            if (auto* retained = retainSink(commandBuffer); retained != nullptr)
            {
                retained->push_back(operation.source.keepAlive);
                retained->push_back(operation.destination.keepAlive);
            }
            resources.recordSubresourceTransferRequirement(commandBuffer.native(),
                                                           operation.request.source,
                                                           operation.request.sourceSubresource,
                                                           operation.source.layout,
                                                           validation_detail::SubresourceTransferUse::resolveSource);
            resources.recordSubresourceTransferRequirement(
                commandBuffer.native(),
                operation.request.destination,
                operation.request.destinationSubresource,
                operation.destination.layout,
                validation_detail::SubresourceTransferUse::resolveDestination);
        }

        [[nodiscard]] VkImageResolve colorResolveRegion(const TextureResolveOperation& operation) noexcept
        {
            const auto extent = resolvedMipExtent(operation);
            return VkImageResolve{
                .srcSubresource =
                    {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel = operation.request.sourceSubresource.mipLevel,
                        .baseArrayLayer = operation.request.sourceSubresource.arrayLayer,
                        .layerCount = 1,
                    },
                .srcOffset = {0, 0, 0},
                .dstSubresource =
                    {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel = operation.request.destinationSubresource.mipLevel,
                        .baseArrayLayer = operation.request.destinationSubresource.arrayLayer,
                        .layerCount = 1,
                    },
                .dstOffset = {0, 0, 0},
                .extent = {extent.width, extent.height, 1},
            };
        }

        [[nodiscard]] Status validateColorResolveSupport(const VulkanResourceAllocator& resources,
                                                         const TextureResolveOperation& operation)
        {
            const auto properties = resources.queryFormatProperties(operation.source.desc.format);
            if (!hasAny(properties.features, FormatFeature::multisampleResolve) ||
                !supportsResolveMode(properties.colorResolveModes, operation.request.mode))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The texture format does not support multisample resolve.",
                });
            }
            return {};
        }

        [[nodiscard]] Status recordColorResolve(VulkanResourceAllocator& resources,
                                                VulkanCommandBuffer& commandBuffer,
                                                const TextureResolveOperation& operation)
        {
            if (auto supported = validateColorResolveSupport(resources, operation); !supported)
            {
                return supported;
            }

            const auto* const table = deviceTable(commandBuffer);
            if (table == nullptr || table->vkCmdResolveImage == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Vulkan texture resolves are unavailable because the device table is incomplete.",
                });
            }
            recordResolveTextureUse(resources, commandBuffer, operation);

            const auto region = colorResolveRegion(operation);
            table->vkCmdResolveImage(commandBuffer.native(),
                                     operation.source.image,
                                     operation.source.layout,
                                     operation.destination.image,
                                     operation.destination.layout,
                                     1,
                                     &region);
            return {};
        }

        struct DepthStencilAttachmentState final
        {
            bool separateAspects = false;
            TextureAspect viewAspect = TextureAspect::depth;
            VkImageAspectFlags barrierAspects = 0;
            VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        };

        [[nodiscard]] DepthStencilAttachmentState depthStencilAttachmentState(
            const VulkanPhysicalDeviceRecord& record, const TextureResolveOperation& operation) noexcept
        {
            const auto aspect = operation.request.sourceSubresource.aspect;
            const bool separateAspects =
                operation.source.desc.format != Format::d24UnormS8Uint || record.separateDepthStencilLayouts;
            if (!separateAspects)
            {
                return DepthStencilAttachmentState{
                    .separateAspects = false,
                    .viewAspect = TextureAspect::depth,
                    .barrierAspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                    .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                };
            }
            return DepthStencilAttachmentState{
                .separateAspects = true,
                .viewAspect = aspect,
                .barrierAspects = nativeAspect(aspect),
                .layout = resolvesDepth(operation.request) ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
                                                           : VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL,
            };
        }

        [[nodiscard]] Status validateDepthStencilResolveSupport(const VulkanPhysicalDeviceRecord& record,
                                                                const TextureResolveOperation& operation,
                                                                const DepthStencilAttachmentState& attachmentState)
        {
            const auto supported = resolvesDepth(operation.request) ? portableMinMax(record.depthResolveModes)
                                                                    : portableMinMax(record.stencilResolveModes);
            const bool singleAspectResolveUnavailable =
                !attachmentState.separateAspects && !record.independentResolveNone && !record.independentResolve;
            if (!supportsResolveMode(supported, operation.request.mode) || singleAspectResolveUnavailable)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The requested depth/stencil resolve mode is unavailable on this adapter.",
                });
            }
            return {};
        }

        struct ResolveAttachmentViews final
        {
            TextureViewHandle source{};
            TextureViewHandle destination{};
        };

        [[nodiscard]] constexpr TextureViewDimension resolveAttachmentViewDimension(const TextureDesc& desc) noexcept
        {
            return desc.depthOrArrayLayers > 1 ? TextureViewDimension::texture2DArray : TextureViewDimension::texture2D;
        }

        [[nodiscard]] Result<ResolveAttachmentViews> createResolveAttachmentViews(
            VulkanResourceAllocator& resources, const TextureResolveOperation& operation, TextureAspect viewAspect)
        {
            auto sourceView = resources.createTextureView(TextureViewDesc{
                .texture = operation.request.source,
                .usage = TextureViewUsage::depthStencil,
                .dimension = resolveAttachmentViewDimension(operation.source.desc),
                .aspect = viewAspect,
                .firstMipLevel = operation.request.sourceSubresource.mipLevel,
                .mipLevelCount = 1,
                .firstArrayLayer = operation.request.sourceSubresource.arrayLayer,
                .arrayLayerCount = 1,
                .label = "Vulkan resolve source",
            });
            auto destinationView = resources.createTextureView(TextureViewDesc{
                .texture = operation.request.destination,
                .usage = TextureViewUsage::depthStencil,
                .dimension = resolveAttachmentViewDimension(operation.destination.desc),
                .aspect = viewAspect,
                .firstMipLevel = operation.request.destinationSubresource.mipLevel,
                .mipLevelCount = 1,
                .firstArrayLayer = operation.request.destinationSubresource.arrayLayer,
                .arrayLayerCount = 1,
                .label = "Vulkan resolve destination",
            });
            if (!sourceView || !destinationView)
            {
                if (sourceView)
                {
                    static_cast<void>(resources.destroyTextureView(*sourceView));
                }
                if (destinationView)
                {
                    static_cast<void>(resources.destroyTextureView(*destinationView));
                }
                return std::unexpected(sourceView ? std::move(destinationView.error()) : std::move(sourceView.error()));
            }
            return ResolveAttachmentViews{.source = *sourceView, .destination = *destinationView};
        }

        constexpr auto resolveAttachmentStages = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                                 VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                                                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        constexpr auto resolveAttachmentAccess = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;

        void transitionResolveTexturesToAttachments(const VolkDeviceTable& table,
                                                    VkCommandBuffer command,
                                                    const TextureResolveOperation& operation,
                                                    const DepthStencilAttachmentState& attachmentState) noexcept
        {
            recordImageBarrier(table,
                               command,
                               operation.source.image,
                               attachmentState.barrierAspects,
                               operation.request.sourceSubresource.mipLevel,
                               operation.request.sourceSubresource.arrayLayer,
                               VK_PIPELINE_STAGE_2_RESOLVE_BIT,
                               VK_ACCESS_2_TRANSFER_READ_BIT,
                               resolveAttachmentStages,
                               resolveAttachmentAccess,
                               operation.source.layout,
                               attachmentState.layout);
            recordImageBarrier(table,
                               command,
                               operation.destination.image,
                               attachmentState.barrierAspects,
                               operation.request.destinationSubresource.mipLevel,
                               operation.request.destinationSubresource.arrayLayer,
                               VK_PIPELINE_STAGE_2_RESOLVE_BIT,
                               VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               resolveAttachmentStages,
                               resolveAttachmentAccess,
                               operation.destination.layout,
                               attachmentState.layout);
        }

        void restoreResolveTextureLayouts(const VolkDeviceTable& table,
                                          VkCommandBuffer command,
                                          const TextureResolveOperation& operation,
                                          const DepthStencilAttachmentState& attachmentState) noexcept
        {
            recordImageBarrier(table,
                               command,
                               operation.source.image,
                               attachmentState.barrierAspects,
                               operation.request.sourceSubresource.mipLevel,
                               operation.request.sourceSubresource.arrayLayer,
                               resolveAttachmentStages,
                               resolveAttachmentAccess,
                               VK_PIPELINE_STAGE_2_RESOLVE_BIT,
                               VK_ACCESS_2_TRANSFER_READ_BIT,
                               attachmentState.layout,
                               operation.source.layout);
            recordImageBarrier(table,
                               command,
                               operation.destination.image,
                               attachmentState.barrierAspects,
                               operation.request.destinationSubresource.mipLevel,
                               operation.request.destinationSubresource.arrayLayer,
                               resolveAttachmentStages,
                               resolveAttachmentAccess,
                               VK_PIPELINE_STAGE_2_RESOLVE_BIT,
                               VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               attachmentState.layout,
                               operation.destination.layout);
        }

        struct DepthStencilResolveAttachments final
        {
            VkRenderingAttachmentInfo depth{};
            VkRenderingAttachmentInfo stencil{};
        };

        [[nodiscard]] DepthStencilResolveAttachments depthStencilResolveAttachments(
            const TextureResolve& request,
            VkImageLayout layout,
            VkImageView sourceView,
            VkImageView destinationView) noexcept
        {
            const bool depthResolve = resolvesDepth(request);
            const bool stencilResolve = resolvesStencil(request);
            const auto mode = nativeResolveMode(request.mode);
            const VkRenderingAttachmentInfo depth{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = {},
                .imageView = sourceView,
                .imageLayout = layout,
                .resolveMode = depthResolve ? mode : VK_RESOLVE_MODE_NONE,
                .resolveImageView = depthResolve ? destinationView : VK_NULL_HANDLE,
                .resolveImageLayout = depthResolve ? layout : VK_IMAGE_LAYOUT_UNDEFINED,
                .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = {},
            };
            VkRenderingAttachmentInfo stencil = depth;
            stencil.resolveMode = stencilResolve ? mode : VK_RESOLVE_MODE_NONE;
            stencil.resolveImageView = stencilResolve ? destinationView : VK_NULL_HANDLE;
            stencil.resolveImageLayout = stencilResolve ? layout : VK_IMAGE_LAYOUT_UNDEFINED;
            return DepthStencilResolveAttachments{.depth = depth, .stencil = stencil};
        }

        void recordResolveRenderingScope(const VolkDeviceTable& table,
                                         VkCommandBuffer command,
                                         const TextureResolveOperation& operation,
                                         const DepthStencilAttachmentState& attachmentState,
                                         const DepthStencilResolveAttachments& attachments) noexcept
        {
            VkRenderingInfo rendering{};
            rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering.renderArea = {{0, 0}, resolvedMipExtent(operation)};
            rendering.layerCount = 1;
            if (attachmentState.separateAspects)
            {
                rendering.pDepthAttachment = resolvesDepth(operation.request) ? &attachments.depth : nullptr;
                rendering.pStencilAttachment = resolvesStencil(operation.request) ? &attachments.stencil : nullptr;
            }
            else
            {
                rendering.pDepthAttachment = &attachments.depth;
                rendering.pStencilAttachment =
                    operation.source.desc.format == Format::d24UnormS8Uint ? &attachments.stencil : nullptr;
            }

            table.vkCmdBeginRendering(command, &rendering);
            table.vkCmdEndRendering(command);
        }

        struct NativeResolveAttachments final
        {
            Result<VulkanResourceAllocator::ResolvedRenderAttachment> source;
            Result<VulkanResourceAllocator::ResolvedRenderAttachment> destination;
        };

        [[nodiscard]] Status recordDepthStencilResolveThroughAttachments(
            VulkanResourceAllocator& resources,
            VulkanCommandBuffer& commandBuffer,
            const TextureResolveOperation& operation,
            const DepthStencilAttachmentState& attachmentState,
            NativeResolveAttachments& nativeAttachments)
        {
            auto& sourceAttachment = nativeAttachments.source;
            auto& destinationAttachment = nativeAttachments.destination;
            if (!sourceAttachment || !destinationAttachment)
            {
                return std::unexpected(sourceAttachment ? std::move(destinationAttachment.error())
                                                        : std::move(sourceAttachment.error()));
            }

            const auto* const table = deviceTable(commandBuffer);
            if (table == nullptr || table->vkCmdBeginRendering == nullptr || table->vkCmdEndRendering == nullptr ||
                table->vkCmdPipelineBarrier2 == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Vulkan depth resolves are unavailable because dynamic rendering is missing.",
                });
            }
            recordResolveTextureUse(resources, commandBuffer, operation);

            const auto renderingAttachments = depthStencilResolveAttachments(
                operation.request, attachmentState.layout, sourceAttachment->view, destinationAttachment->view);
            transitionResolveTexturesToAttachments(*table, commandBuffer.native(), operation, attachmentState);
            recordResolveRenderingScope(
                *table, commandBuffer.native(), operation, attachmentState, renderingAttachments);
            restoreResolveTextureLayouts(*table, commandBuffer.native(), operation, attachmentState);

            auto& retained = commandBuffer.retained();
            retained.push_back(sourceAttachment->keepAlive);
            retained.push_back(destinationAttachment->keepAlive);
            retained.push_back(sourceAttachment->textureKeepAlive);
            retained.push_back(destinationAttachment->textureKeepAlive);
            return {};
        }

        [[nodiscard]] Status recordDepthStencilResolve(VulkanResourceAllocator& resources,
                                                       const VulkanPhysicalDeviceRecord& record,
                                                       VulkanCommandBuffer& commandBuffer,
                                                       const TextureResolveOperation& operation)
        {
            const auto attachmentState = depthStencilAttachmentState(record, operation);
            if (auto supported = validateDepthStencilResolveSupport(record, operation, attachmentState); !supported)
            {
                return supported;
            }

            auto views = createResolveAttachmentViews(resources, operation, attachmentState.viewAspect);
            if (!views)
            {
                return std::unexpected(std::move(views.error()));
            }
            NativeResolveAttachments nativeAttachments{
                .source = resources.resolveRenderAttachment(views->source),
                .destination = resources.resolveRenderAttachment(views->destination),
            };
            auto recorded = recordDepthStencilResolveThroughAttachments(
                resources, commandBuffer, operation, attachmentState, nativeAttachments);
            static_cast<void>(resources.destroyTextureView(views->source));
            static_cast<void>(resources.destroyTextureView(views->destination));
            return recorded;
        }

    } // namespace

    Status VulkanImageUtilities::resolveTexture(VulkanCommandBuffer& commandBuffer, const TextureResolve& resolve)
    {
        if (resources_ == nullptr || record_ == nullptr || commandBuffer.native() == VK_NULL_HANDLE ||
            commandBuffer.queueType() != QueueType::graphics || commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A texture resolve requires a graphics command buffer outside a rendering scope.",
            });
        }

        auto source = resources_->nativeTexture(resolve.source, commandBuffer.native(), resolve.sourceSubresource);
        auto destination =
            resources_->nativeTexture(resolve.destination, commandBuffer.native(), resolve.destinationSubresource);
        if (!source)
        {
            return std::unexpected(std::move(source.error()));
        }
        if (!destination)
        {
            return std::unexpected(std::move(destination.error()));
        }

        const TextureResolveOperation operation{.request = resolve, .source = *source, .destination = *destination};
        if (auto valid = validateTextureResolve(operation); !valid)
        {
            return valid;
        }
        const std::array swapchainLeases{source->swapchainLease, destination->swapchainLease};
        if (auto referenced = resources_->retainSwapchainImages(commandBuffer.native(), swapchainLeases); !referenced)
        {
            return referenced;
        }
        if (!isDepthFormat(source->desc.format))
        {
            return recordColorResolve(*resources_, commandBuffer, operation);
        }
        return recordDepthStencilResolve(*resources_, *record_, commandBuffer, operation);
    }

    namespace
    {

        constexpr BarrierScope mipmapEarlierWork{PipelineStage::all, Access::none};
        constexpr BarrierScope mipmapShaderRead{PipelineStage::computeShader, Access::shaderResource};
        constexpr BarrierScope mipmapShaderWrite{PipelineStage::computeShader, Access::unorderedAccess};

        struct MipmapTarget final
        {
            TextureHandle texture{};
            TextureDimension dimension = TextureDimension::texture2D;
            std::uint32_t width = 1;
            std::uint32_t height = 1;
            std::uint32_t depthOrArrayLayers = 1;
            TextureSubresourceRange range{};
            Format sampledFormat = Format::unknown;
            Format storageFormat = Format::unknown;
            bool encodeSrgb = false;
        };

        struct MipmapGenerationPlan final
        {
            MipmapTarget target{};
            TextureLayout baseLayout = TextureLayout::undefined;
            BarrierScope baseScope{};
            TextureLayout finalLayout = TextureLayout::undefined;
            BarrierScope finalScope{};
        };

        [[nodiscard]] Result<TextureSubresourceRange> selectMipmapRange(const TextureSubresourceRange& requested,
                                                                        const TextureDesc& desc)
        {
            auto mipCount =
                selectedCount(requested.firstMipLevel, requested.mipLevelCount, desc.mipLevels, "The mipmap range");
            const auto availableLayerCount =
                desc.dimension == TextureDimension::texture3D ? 1U : desc.depthOrArrayLayers;
            auto layerCount = selectedCount(
                requested.firstArrayLayer, requested.arrayLayerCount, availableLayerCount, "The mipmap layer range");
            if (!mipCount)
            {
                return std::unexpected(std::move(mipCount.error()));
            }
            if (!layerCount)
            {
                return std::unexpected(std::move(layerCount.error()));
            }
            if (*mipCount < 2)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument, 0, "Mipmap generation requires at least two selected mip levels."});
            }
            return TextureSubresourceRange{
                .firstMipLevel = requested.firstMipLevel,
                .mipLevelCount = *mipCount,
                .firstArrayLayer = requested.firstArrayLayer,
                .arrayLayerCount = *layerCount,
                .aspect = TextureAspect::color,
            };
        }

        [[nodiscard]] Result<MipmapGenerationPlan> planMipmapGeneration(const TextureMipmapGeneration& generation,
                                                                        const NativeTexture& texture,
                                                                        QueueType queueType)
        {
            const auto& desc = texture.desc;
            if (desc.sampleCount != 1 || !hasAny(desc.usage, TextureUsage::sampled) ||
                !hasAny(desc.usage, TextureUsage::storage) || generation.subresources.aspect != TextureAspect::color)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Mipmap generation requires a single-sampled color texture with sampled and storage usage.",
                });
            }

            auto range = selectMipmapRange(generation.subresources, desc);
            if (!range)
            {
                return std::unexpected(std::move(range.error()));
            }
            const auto baseScope = barrierScope(generation.baseLayout, queueType);
            const auto finalScope = barrierScope(generation.finalLayout, queueType);
            if (!baseScope || !finalScope || !layoutCompatible(desc.usage, generation.baseLayout) ||
                !layoutCompatible(desc.usage, generation.finalLayout))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The mipmap seed or final layout is undefined or incompatible with the texture usage.",
                });
            }

            const auto linearFormat = linearViewFormat(desc.format);
            const bool encodeSrgb = linearFormat != Format::unknown;
            if (encodeSrgb && std::ranges::find(texture.viewFormats, linearFormat) == texture.viewFormats.end())
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "sRGB mipmap generation requires the compatible linear format in TextureDesc::viewFormats.",
                });
            }
            return MipmapGenerationPlan{
                .target =
                    {
                        .texture = generation.texture,
                        .dimension = desc.dimension,
                        .width = desc.width,
                        .height = desc.height,
                        .depthOrArrayLayers = desc.depthOrArrayLayers,
                        .range = *range,
                        .sampledFormat = desc.format,
                        .storageFormat = encodeSrgb ? linearFormat : desc.format,
                        .encodeSrgb = encodeSrgb,
                    },
                .baseLayout = generation.baseLayout,
                .baseScope = *baseScope,
                .finalLayout = generation.finalLayout,
                .finalScope = *finalScope,
            };
        }

        [[nodiscard]] constexpr TextureSubresourceRange singleMipRange(const TextureSubresourceRange& range,
                                                                       std::uint32_t mipLevel) noexcept
        {
            return TextureSubresourceRange{
                .firstMipLevel = mipLevel,
                .mipLevelCount = 1,
                .firstArrayLayer = range.firstArrayLayer,
                .arrayLayerCount = range.arrayLayerCount,
                .aspect = range.aspect,
            };
        }

        [[nodiscard]] Status recordMipmapBarrier(VulkanResourceAllocator& resources,
                                                 VulkanCommandBuffer& commandBuffer,
                                                 const TextureBarrier& barrier)
        {
            return resources.textureBarrier(
                commandBuffer.native(), commandBuffer.queueType(), barrier, retainSink(commandBuffer));
        }

        [[nodiscard]] Status transitionMipmapBaseToShaderResource(VulkanResourceAllocator& resources,
                                                                  VulkanCommandBuffer& commandBuffer,
                                                                  const MipmapGenerationPlan& plan)
        {
            const auto& target = plan.target;
            return recordMipmapBarrier(resources,
                                       commandBuffer,
                                       textureBarrier(target.texture,
                                                      singleMipRange(target.range, target.range.firstMipLevel),
                                                      plan.baseLayout,
                                                      plan.baseScope,
                                                      TextureLayout::shaderResource,
                                                      mipmapShaderRead));
        }

        [[nodiscard]] Status transitionMipmapsToFinalLayout(VulkanResourceAllocator& resources,
                                                            VulkanCommandBuffer& commandBuffer,
                                                            const MipmapGenerationPlan& plan)
        {
            return recordMipmapBarrier(resources,
                                       commandBuffer,
                                       textureBarrier(plan.target.texture,
                                                      plan.target.range,
                                                      TextureLayout::shaderResource,
                                                      mipmapShaderRead,
                                                      plan.finalLayout,
                                                      plan.finalScope));
        }

        struct MipmapDispatchPipeline final
        {
            const VolkDeviceTable& table;
            std::shared_ptr<const VulkanNativePipeline> pipeline;
        };

        [[nodiscard]] Result<MipmapDispatchPipeline> acquireMipmapDispatchPipeline(
            const VulkanPipelineManager& pipelineManager,
            PipelineHandle pipeline,
            const VulkanCommandBuffer& commandBuffer)
        {
            auto nativePipeline = pipelineManager.nativePipeline(pipeline);
            if (!nativePipeline || *nativePipeline == nullptr)
            {
                return std::unexpected(nativePipeline
                                           ? Error{ErrorCode::invalidState, 0, "Mipmap compute pipeline is null."}
                                           : std::move(nativePipeline.error()));
            }
            const auto* const table = deviceTable(commandBuffer);
            if (table == nullptr || table->vkCmdBindPipeline == nullptr || table->vkCmdBindDescriptorSets == nullptr ||
                table->vkCmdDispatch == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Mipmap generation is unavailable because the device table is incomplete.",
                });
            }
            return MipmapDispatchPipeline{.table = *table, .pipeline = std::move(*nativePipeline)};
        }

        struct MipmapLevelViews final
        {
            TextureViewHandle sampled{};
            TextureViewHandle storage{};
        };

        [[nodiscard]] Result<MipmapLevelViews> createMipmapLevelViews(VulkanResourceAllocator& resources,
                                                                      const MipmapTarget& target,
                                                                      std::uint32_t destinationMip)
        {
            const auto viewDimension = mipmapViewDimension(target.dimension);
            auto sampled = resources.createTextureView(TextureViewDesc{
                .texture = target.texture,
                .usage = TextureViewUsage::sampled,
                .dimension = viewDimension,
                .format = target.sampledFormat,
                .firstMipLevel = destinationMip - 1U,
                .mipLevelCount = 1,
                .firstArrayLayer = target.range.firstArrayLayer,
                .arrayLayerCount = target.range.arrayLayerCount,
            });
            auto storage = resources.createTextureView(TextureViewDesc{
                .texture = target.texture,
                .usage = TextureViewUsage::storage,
                .dimension = viewDimension,
                .format = target.storageFormat,
                .firstMipLevel = destinationMip,
                .mipLevelCount = 1,
                .firstArrayLayer = target.range.firstArrayLayer,
                .arrayLayerCount = target.range.arrayLayerCount,
            });
            if (!sampled || !storage)
            {
                if (sampled)
                {
                    static_cast<void>(resources.destroyTextureView(*sampled));
                }
                if (storage)
                {
                    static_cast<void>(resources.destroyTextureView(*storage));
                }
                return std::unexpected(sampled ? std::move(storage.error()) : std::move(sampled.error()));
            }
            return MipmapLevelViews{.sampled = *sampled, .storage = *storage};
        }

        [[nodiscard]] std::array<std::uint32_t, 3> mipmapDispatchGroupCounts(const MipmapTarget& target,
                                                                             std::uint32_t mipLevel) noexcept
        {
            const auto group = threadGroupSize(target.dimension);
            const auto width = mipDimension(target.width, mipLevel);
            const auto height = mipDimension(target.height, mipLevel);
            const auto depth = target.dimension == TextureDimension::texture3D
                                   ? mipDimension(target.depthOrArrayLayers, mipLevel)
                                   : target.range.arrayLayerCount;
            return {
                (width + group[0] - 1U) / group[0],
                (height + group[1] - 1U) / group[1],
                (depth + group[2] - 1U) / group[2],
            };
        }

        [[nodiscard]] Status dispatchMipmapLevel(const VulkanBindingManager& bindings,
                                                 VulkanCommandBuffer& commandBuffer,
                                                 const MipmapDispatchPipeline& dispatchPipeline,
                                                 BindingSetHandle set,
                                                 const std::array<std::uint32_t, 3>& groupCounts)
        {
            auto nativeSet = bindings.nativeBindingSet(set);
            if (!nativeSet)
            {
                return std::unexpected(std::move(nativeSet.error()));
            }

            const auto& table = dispatchPipeline.table;
            const auto& pipeline = dispatchPipeline.pipeline;
            table.vkCmdBindPipeline(commandBuffer.native(), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
            table.vkCmdBindDescriptorSets(commandBuffer.native(),
                                          VK_PIPELINE_BIND_POINT_COMPUTE,
                                          pipeline->layout,
                                          0,
                                          1,
                                          &nativeSet->descriptorSet,
                                          0,
                                          nullptr);
            table.vkCmdDispatch(commandBuffer.native(), groupCounts[0], groupCounts[1], groupCounts[2]);

            auto& retained = commandBuffer.retained();
            retained.push_back(std::const_pointer_cast<void>(std::static_pointer_cast<const void>(pipeline)));
            retained.push_back(std::move(nativeSet->owner));
            return {};
        }

    } // namespace

    struct VulkanImageUtilities::MipmapLevelInputs final
    {
        const MipmapTarget& target;
        const MipmapDispatchPipeline& dispatchPipeline;
    };

    Status VulkanImageUtilities::recordMipmapLevel(VulkanCommandBuffer& commandBuffer,
                                                   const MipmapPipelineSnapshot& pipelines,
                                                   const MipmapLevelInputs& inputs,
                                                   std::uint32_t destinationMip)
    {
        const auto& target = inputs.target;
        const auto destinationRange = singleMipRange(target.range, destinationMip);
        if (auto barrier = recordMipmapBarrier(*resources_,
                                               commandBuffer,
                                               textureBarrier(target.texture,
                                                              destinationRange,
                                                              TextureLayout::undefined,
                                                              mipmapEarlierWork,
                                                              TextureLayout::unorderedAccess,
                                                              mipmapShaderWrite));
            !barrier)
        {
            return barrier;
        }

        auto views = createMipmapLevelViews(*resources_, target, destinationMip);
        if (!views)
        {
            return std::unexpected(std::move(views.error()));
        }
        const auto constants = target.encodeSrgb ? pipelines.encodeOn : pipelines.encodeOff;
        const std::array writes{
            BindingWrite{.binding = 0, .resource = BufferBinding{.buffer = constants, .size = 256}},
            BindingWrite{.binding = 1, .resource = views->sampled},
            BindingWrite{.binding = 2, .resource = views->storage},
            BindingWrite{.binding = 3, .resource = pipelines.sampler},
        };
        auto set = allocateMipmapBindingSet(pipelines.bindingLayout, writes);
        if (!set)
        {
            static_cast<void>(resources_->destroyTextureView(views->sampled));
            static_cast<void>(resources_->destroyTextureView(views->storage));
            return std::unexpected(std::move(set.error()));
        }
        auto dispatched = dispatchMipmapLevel(*bindings_,
                                              commandBuffer,
                                              inputs.dispatchPipeline,
                                              *set,
                                              mipmapDispatchGroupCounts(target, destinationMip));
        static_cast<void>(bindings_->destroyBindingSet(*set));
        static_cast<void>(resources_->destroyTextureView(views->sampled));
        static_cast<void>(resources_->destroyTextureView(views->storage));
        if (!dispatched)
        {
            return dispatched;
        }

        return recordMipmapBarrier(*resources_,
                                   commandBuffer,
                                   textureBarrier(target.texture,
                                                  destinationRange,
                                                  TextureLayout::unorderedAccess,
                                                  mipmapShaderWrite,
                                                  TextureLayout::shaderResource,
                                                  mipmapShaderRead));
    }

    Status VulkanImageUtilities::generateMipmaps(VulkanCommandBuffer& commandBuffer,
                                                 const TextureMipmapGeneration& generation)
    {
        auto mipmapPipelines = ensureMipmapPipelines();
        if (!mipmapPipelines)
        {
            return std::unexpected(std::move(mipmapPipelines.error()));
        }
        if ((commandBuffer.queueType() != QueueType::graphics && commandBuffer.queueType() != QueueType::compute) ||
            commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Mipmap generation requires a graphics or compute command buffer outside a rendering scope.",
            });
        }

        auto texture = resources_->nativeTexture(generation.texture, commandBuffer.native());
        if (!texture)
        {
            return std::unexpected(std::move(texture.error()));
        }
        auto plan = planMipmapGeneration(generation, *texture, commandBuffer.queueType());
        if (!plan)
        {
            return std::unexpected(std::move(plan.error()));
        }
        const auto& target = plan->target;

        commandBuffer.computeState() = {};
        commandBuffer.rayTracingState() = {};

        if (auto sampleable = transitionMipmapBaseToShaderResource(*resources_, commandBuffer, *plan); !sampleable)
        {
            return sampleable;
        }
        auto dispatchPipeline = acquireMipmapDispatchPipeline(
            *pipelineManager_, mipmapPipelines->computePipelines[dimensionIndex(target.dimension)], commandBuffer);
        if (!dispatchPipeline)
        {
            return std::unexpected(std::move(dispatchPipeline.error()));
        }

        const MipmapLevelInputs inputs{.target = target, .dispatchPipeline = *dispatchPipeline};
        for (std::uint32_t mipOffset = 1; mipOffset < target.range.mipLevelCount; ++mipOffset)
        {
            const auto destinationMip = target.range.firstMipLevel + mipOffset;
            if (auto level = recordMipmapLevel(commandBuffer, *mipmapPipelines, inputs, destinationMip); !level)
            {
                return level;
            }
        }

        if (auto finished = transitionMipmapsToFinalLayout(*resources_, commandBuffer, *plan); !finished)
        {
            return finished;
        }
        commandBuffer.computeState() = {};
        return {};
    }
} // namespace rachis::vulkan
