#include "Common/VectorReserve.h"
#include "Internal/VulkanDiagnostics.h"
#include "Internal/VulkanImageBarriers.h"
#include "Internal/VulkanResources.h"
#include "Internal/VulkanStageSupport.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

namespace rachis::vulkan
{
    namespace
    {

        struct NativeTextureRange final
        {
            std::uint32_t firstMip = 0;
            std::uint32_t mipCount = 0;
            std::uint32_t firstArray = 0;
            std::uint32_t arrayCount = 0;
            VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        };

        [[nodiscard]] constexpr bool validRange(std::uint64_t resourceSize,
                                                std::uint64_t offset,
                                                std::uint64_t size) noexcept
        {
            return offset < resourceSize && size != 0 && (size == wholeResource || size <= resourceSize - offset);
        }

        [[nodiscard]] constexpr std::uint64_t resolveSize(std::uint64_t resourceSize,
                                                          std::uint64_t offset,
                                                          std::uint64_t size) noexcept
        {
            return size == wholeResource ? resourceSize - offset : size;
        }

        [[nodiscard]] constexpr bool validPipelineStages(PipelineStage stages) noexcept
        {
            constexpr auto allowed =
                static_cast<std::uint32_t>(PipelineStage::all) | static_cast<std::uint32_t>(PipelineStage::draw) |
                static_cast<std::uint32_t>(PipelineStage::vertexInput) |
                static_cast<std::uint32_t>(PipelineStage::vertexShader) |
                static_cast<std::uint32_t>(PipelineStage::pixelShader) |
                static_cast<std::uint32_t>(PipelineStage::computeShader) |
                static_cast<std::uint32_t>(PipelineStage::copy) | static_cast<std::uint32_t>(PipelineStage::resolve) |
                static_cast<std::uint32_t>(PipelineStage::renderTarget) |
                static_cast<std::uint32_t>(PipelineStage::depthStencil) |
                static_cast<std::uint32_t>(PipelineStage::indirect) |
                static_cast<std::uint32_t>(PipelineStage::geometryShader) |
                static_cast<std::uint32_t>(PipelineStage::hullShader) |
                static_cast<std::uint32_t>(PipelineStage::domainShader) |
                static_cast<std::uint32_t>(PipelineStage::taskShader) |
                static_cast<std::uint32_t>(PipelineStage::meshShader) |
                static_cast<std::uint32_t>(PipelineStage::conditionalExecution) |
                static_cast<std::uint32_t>(PipelineStage::accelerationStructureBuild) |
                static_cast<std::uint32_t>(PipelineStage::rayTracingShader) |
                static_cast<std::uint32_t>(PipelineStage::accelerationStructureCopy) |
                static_cast<std::uint32_t>(PipelineStage::streamOutput);
            return (static_cast<std::uint32_t>(stages) & ~allowed) == 0;
        }

        [[nodiscard]] constexpr bool validAccess(Access access) noexcept
        {
            constexpr auto allowed =
                static_cast<std::uint32_t>(Access::common) | static_cast<std::uint32_t>(Access::vertexBuffer) |
                static_cast<std::uint32_t>(Access::constantBuffer) | static_cast<std::uint32_t>(Access::indexBuffer) |
                static_cast<std::uint32_t>(Access::shaderResource) |
                static_cast<std::uint32_t>(Access::unorderedAccess) | static_cast<std::uint32_t>(Access::copySource) |
                static_cast<std::uint32_t>(Access::copyDestination) |
                static_cast<std::uint32_t>(Access::indirectArgument) |
                static_cast<std::uint32_t>(Access::renderTarget) |
                static_cast<std::uint32_t>(Access::depthStencilRead) |
                static_cast<std::uint32_t>(Access::depthStencilWrite) |
                static_cast<std::uint32_t>(Access::resolveSource) |
                static_cast<std::uint32_t>(Access::resolveDestination) | static_cast<std::uint32_t>(Access::predicate) |
                static_cast<std::uint32_t>(Access::shadingRateSource) |
                static_cast<std::uint32_t>(Access::accelerationStructureBuildInput) |
                static_cast<std::uint32_t>(Access::accelerationStructureScratch) |
                static_cast<std::uint32_t>(Access::streamOutputWrite);
            const auto raw = static_cast<std::uint32_t>(access);
            return (raw & ~allowed) == 0 && (!hasAny(access, Access::common) || access == Access::common);
        }

        [[nodiscard]] constexpr bool hasAtMostOneWriteAccess(Access access) noexcept
        {
            constexpr auto writeAccess = Access::unorderedAccess | Access::copyDestination | Access::renderTarget |
                                         Access::depthStencilWrite | Access::resolveDestination |
                                         Access::accelerationStructureScratch | Access::streamOutputWrite;
            const auto writes = static_cast<std::uint32_t>(access & writeAccess);
            return writes == 0 || (writes & (writes - 1U)) == 0;
        }

        [[nodiscard]] constexpr bool accessMatchesStages(PipelineStage stages, Access access) noexcept
        {
            const auto matches = [&](Access requiredAccess, PipelineStage compatibleStages) {
                return !hasAny(access, requiredAccess) || hasAny(stages, PipelineStage::all | compatibleStages);
            };
            return matches(Access::vertexBuffer,
                           PipelineStage::draw | PipelineStage::vertexInput | PipelineStage::vertexShader) &&
                   matches(Access::constantBuffer,
                           PipelineStage::draw | PipelineStage::vertexShader | PipelineStage::pixelShader |
                               PipelineStage::computeShader | PipelineStage::geometryShader |
                               PipelineStage::hullShader | PipelineStage::domainShader | PipelineStage::taskShader |
                               PipelineStage::meshShader | PipelineStage::rayTracingShader) &&
                   matches(Access::indexBuffer, PipelineStage::draw | PipelineStage::vertexInput) &&
                   matches(Access::shaderResource,
                           PipelineStage::draw | PipelineStage::vertexShader | PipelineStage::pixelShader |
                               PipelineStage::computeShader | PipelineStage::geometryShader |
                               PipelineStage::hullShader | PipelineStage::domainShader | PipelineStage::taskShader |
                               PipelineStage::meshShader | PipelineStage::rayTracingShader |
                               PipelineStage::accelerationStructureCopy) &&
                   matches(Access::unorderedAccess,
                           PipelineStage::draw | PipelineStage::vertexShader | PipelineStage::pixelShader |
                               PipelineStage::computeShader | PipelineStage::geometryShader |
                               PipelineStage::hullShader | PipelineStage::domainShader | PipelineStage::taskShader |
                               PipelineStage::meshShader | PipelineStage::rayTracingShader |
                               PipelineStage::accelerationStructureCopy) &&
                   matches(Access::copySource | Access::copyDestination, PipelineStage::copy) &&
                   matches(Access::indirectArgument, PipelineStage::indirect) &&
                   matches(Access::predicate, PipelineStage::conditionalExecution) &&
                   matches(Access::shadingRateSource, PipelineStage::draw | PipelineStage::pixelShader) &&
                   matches(Access::accelerationStructureBuildInput | Access::accelerationStructureScratch,
                           PipelineStage::accelerationStructureBuild) &&
                   matches(Access::streamOutputWrite, PipelineStage::streamOutput) &&
                   matches(Access::renderTarget, PipelineStage::draw | PipelineStage::renderTarget) &&
                   matches(Access::depthStencilRead | Access::depthStencilWrite,
                           PipelineStage::draw | PipelineStage::depthStencil) &&
                   matches(Access::resolveSource | Access::resolveDestination, PipelineStage::resolve);
        }

        [[nodiscard]] constexpr bool supportsBarrierScopeOnQueue(PipelineStage stages,
                                                                 Access access,
                                                                 QueueType queueType) noexcept
        {
            if (stages == PipelineStage::none && access != Access::none)
            {
                return false;
            }
            if (queueType == QueueType::graphics)
            {
                return true;
            }
            const auto rawStages = static_cast<std::uint32_t>(stages);
            const auto rawAccess = static_cast<std::uint32_t>(access);
            const auto stagesFit = [&](PipelineStage allowed) {
                return (rawStages & ~static_cast<std::uint32_t>(allowed)) == 0;
            };
            const auto accessFits = [&](Access allowed) {
                return (rawAccess & ~static_cast<std::uint32_t>(allowed)) == 0;
            };
            if (queueType == QueueType::compute)
            {
                return stagesFit(PipelineStage::all | PipelineStage::computeShader | PipelineStage::copy |
                                 PipelineStage::indirect | PipelineStage::conditionalExecution |
                                 PipelineStage::accelerationStructureBuild | PipelineStage::rayTracingShader |
                                 PipelineStage::accelerationStructureCopy) &&
                       accessFits(Access::common | Access::constantBuffer | Access::shaderResource |
                                  Access::unorderedAccess | Access::copySource | Access::copyDestination |
                                  Access::indirectArgument | Access::predicate |
                                  Access::accelerationStructureBuildInput | Access::accelerationStructureScratch);
            }
            return stagesFit(PipelineStage::all | PipelineStage::copy) &&
                   accessFits(Access::common | Access::copySource | Access::copyDestination);
        }

        [[nodiscard]] constexpr bool validBarrierScope(PipelineStage stages,
                                                       Access access,
                                                       QueueType queueType) noexcept
        {
            return validPipelineStages(stages) && validAccess(access) && accessMatchesStages(stages, access) &&
                   supportsBarrierScopeOnQueue(stages, access, queueType);
        }

        [[nodiscard]] constexpr bool validResourceBarrierScope(PipelineStage stages,
                                                               Access access,
                                                               QueueType queueType) noexcept
        {
            return validBarrierScope(stages, access, queueType) && hasAtMostOneWriteAccess(access);
        }

        [[nodiscard]] constexpr bool hasWriteAccess(Access access) noexcept
        {
            constexpr auto writeAccess = Access::unorderedAccess | Access::copyDestination | Access::renderTarget |
                                         Access::depthStencilWrite | Access::resolveDestination |
                                         Access::accelerationStructureScratch | Access::streamOutputWrite;
            return hasAny(access, writeAccess);
        }

        struct BarrierStageSupport final
        {
            PipelineStage stages = corePipelineStages;
            bool shadingRateAttachments = false;
        };

        [[nodiscard]] constexpr bool scopeFitsEnabledStages(PipelineStage stages,
                                                            Access access,
                                                            const BarrierStageSupport& support) noexcept
        {
            const auto enabled = stages & support.stages;
            if (stages != PipelineStage::none && enabled == PipelineStage::none)
            {
                return false;
            }
            if (hasAny(access, Access::shadingRateSource) && !support.shadingRateAttachments)
            {
                return false;
            }
            return accessMatchesStages(enabled, access);
        }

        [[nodiscard]] Status requireEnabledBarrierStages(PipelineStage stagesBefore,
                                                         Access accessBefore,
                                                         PipelineStage stagesAfter,
                                                         Access accessAfter,
                                                         const BarrierStageSupport& support)
        {
            if (!scopeFitsEnabledStages(stagesBefore, accessBefore, support) ||
                !scopeFitsEnabledStages(stagesAfter, accessAfter, support))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "A barrier scope names only pipeline stages, or an access only in pipeline stages, that this "
                    "Vulkan device has not enabled.",
                });
            }
            return {};
        }

        [[nodiscard]] bool placementsAlias(const std::shared_ptr<PlacedResourceLease>& before,
                                           const std::shared_ptr<PlacedResourceLease>& after) noexcept
        {
            return before != nullptr && after != nullptr && before->heap != nullptr && before->heap == after->heap &&
                   hasAny(before->heap->desc.flags, ResourceHeapFlag::allowAliasing) &&
                   before->offset < after->offset + after->size && after->offset < before->offset + before->size;
        }

        [[nodiscard]] constexpr bool supportsBufferAccess(BufferUsage usage, Access access) noexcept
        {
            if (access == Access::none || access == Access::common)
            {
                return true;
            }
            Access allowed = Access::none;
            if (hasAny(usage, BufferUsage::copySource))
            {
                allowed = allowed | Access::copySource;
            }
            if (hasAny(usage, BufferUsage::copyDestination))
            {
                allowed = allowed | Access::copyDestination;
            }
            if (hasAny(usage, BufferUsage::vertex))
            {
                allowed = allowed | Access::vertexBuffer;
            }
            if (hasAny(usage, BufferUsage::index))
            {
                allowed = allowed | Access::indexBuffer;
            }
            if (hasAny(usage, BufferUsage::constant))
            {
                allowed = allowed | Access::constantBuffer;
            }
            if (hasAny(usage, BufferUsage::storage))
            {
                allowed = allowed | Access::shaderResource | Access::unorderedAccess;
            }
            if (hasAny(usage, BufferUsage::indirect))
            {
                allowed = allowed | Access::indirectArgument;
            }
            if (hasAny(usage, BufferUsage::predicate))
            {
                allowed = allowed | Access::predicate;
            }
            if (hasAny(usage, BufferUsage::accelerationStructureBuildInput))
            {
                allowed = allowed | Access::accelerationStructureBuildInput;
            }
            if (hasAny(usage, BufferUsage::accelerationStructureScratch))
            {
                allowed = allowed | Access::accelerationStructureScratch;
            }
            if (hasAny(usage, BufferUsage::shaderBindingTable))
            {
                allowed = allowed | Access::shaderResource;
            }
            return (static_cast<std::uint32_t>(access) & ~static_cast<std::uint32_t>(allowed)) == 0;
        }

        [[nodiscard]] constexpr bool supportsTextureAccess(TextureUsage usage, Access access) noexcept
        {
            if (access == Access::none || access == Access::common)
            {
                return true;
            }
            Access allowed = Access::none;
            if (hasAny(usage, TextureUsage::sampled))
            {
                allowed = allowed | Access::shaderResource;
            }
            if (hasAny(usage, TextureUsage::storage))
            {
                allowed = allowed | Access::unorderedAccess;
            }
            if (hasAny(usage, TextureUsage::copySource))
            {
                allowed = allowed | Access::copySource | Access::resolveSource;
            }
            if (hasAny(usage, TextureUsage::copyDestination))
            {
                allowed = allowed | Access::copyDestination | Access::resolveDestination;
            }
            if (hasAny(usage, TextureUsage::renderTarget))
            {
                allowed = allowed | Access::renderTarget | Access::resolveSource | Access::resolveDestination;
            }
            if (hasAny(usage, TextureUsage::depthStencil))
            {
                allowed = allowed | Access::depthStencilRead | Access::depthStencilWrite | Access::resolveSource |
                          Access::resolveDestination;
            }
            if (hasAny(usage, TextureUsage::shadingRateSource))
            {
                allowed = allowed | Access::shadingRateSource;
            }
            return (static_cast<std::uint32_t>(access) & ~static_cast<std::uint32_t>(allowed)) == 0;
        }

        [[nodiscard]] constexpr bool accessMatchesTextureLayout(TextureLayout layout, Access access) noexcept
        {
            if (layout == TextureLayout::undefined || layout == TextureLayout::present)
            {
                return access == Access::none;
            }
            if (access == Access::none || access == Access::common)
            {
                return true;
            }
            Access allowed = Access::none;
            switch (layout)
            {
            case TextureLayout::common:
                allowed = Access::shaderResource | Access::copySource | Access::copyDestination;
                break;
            case TextureLayout::shaderResource:
                allowed = Access::shaderResource;
                break;
            case TextureLayout::unorderedAccess:
                allowed = Access::unorderedAccess;
                break;
            case TextureLayout::renderTarget:
                allowed = Access::renderTarget;
                break;
            case TextureLayout::depthStencilWrite:
                allowed = Access::depthStencilRead | Access::depthStencilWrite;
                break;
            case TextureLayout::depthStencilRead:
                allowed = Access::depthStencilRead;
                break;
            case TextureLayout::copySource:
                allowed = Access::copySource;
                break;
            case TextureLayout::copyDestination:
                allowed = Access::copyDestination;
                break;
            case TextureLayout::resolveSource:
                allowed = Access::resolveSource;
                break;
            case TextureLayout::resolveDestination:
                allowed = Access::resolveDestination;
                break;
            case TextureLayout::shadingRateSource:
                allowed = Access::shadingRateSource;
                break;
            case TextureLayout::undefined:
            case TextureLayout::present:
                break;
            }
            return (static_cast<std::uint32_t>(access) & ~static_cast<std::uint32_t>(allowed)) == 0;
        }

        [[nodiscard]] constexpr bool validTextureLayout(TextureLayout layout) noexcept
        {
            switch (layout)
            {
            case TextureLayout::undefined:
            case TextureLayout::common:
            case TextureLayout::present:
            case TextureLayout::shaderResource:
            case TextureLayout::unorderedAccess:
            case TextureLayout::renderTarget:
            case TextureLayout::depthStencilWrite:
            case TextureLayout::depthStencilRead:
            case TextureLayout::copySource:
            case TextureLayout::copyDestination:
            case TextureLayout::resolveSource:
            case TextureLayout::resolveDestination:
            case TextureLayout::shadingRateSource:
                return true;
            }
            return false;
        }

        [[nodiscard]] constexpr bool supportsLayout(TextureUsage usage, TextureLayout layout) noexcept
        {
            switch (layout)
            {
            case TextureLayout::undefined:
            case TextureLayout::common:
                return true;
            case TextureLayout::shaderResource:
                return hasAny(usage, TextureUsage::sampled);
            case TextureLayout::unorderedAccess:
                return hasAny(usage, TextureUsage::storage);
            case TextureLayout::renderTarget:
                return hasAny(usage, TextureUsage::renderTarget);
            case TextureLayout::depthStencilWrite:
            case TextureLayout::depthStencilRead:
                return hasAny(usage, TextureUsage::depthStencil);
            case TextureLayout::copySource:
                return hasAny(usage, TextureUsage::copySource);
            case TextureLayout::copyDestination:
                return hasAny(usage, TextureUsage::copyDestination);
            case TextureLayout::resolveSource:
                return hasAny(usage,
                              TextureUsage::copySource | TextureUsage::renderTarget | TextureUsage::depthStencil);
            case TextureLayout::resolveDestination:
                return hasAny(usage,
                              TextureUsage::copyDestination | TextureUsage::renderTarget | TextureUsage::depthStencil);
            case TextureLayout::shadingRateSource:
                return hasAny(usage, TextureUsage::shadingRateSource);
            case TextureLayout::present:
                return hasAny(usage, TextureUsage::renderTarget);
            }
            return false;
        }

        [[nodiscard]] constexpr bool supportsTextureLayoutOnQueue(TextureLayout layout, QueueType queueType) noexcept
        {
            if (queueType == QueueType::graphics)
            {
                return true;
            }
            if (queueType == QueueType::compute)
            {
                return layout == TextureLayout::undefined || layout == TextureLayout::common ||
                       layout == TextureLayout::shaderResource || layout == TextureLayout::unorderedAccess ||
                       layout == TextureLayout::copySource || layout == TextureLayout::copyDestination;
            }
            return layout == TextureLayout::undefined || layout == TextureLayout::common ||
                   layout == TextureLayout::copySource || layout == TextureLayout::copyDestination;
        }

        [[nodiscard]] constexpr bool isDepthFormat(Format format) noexcept
        {
            return format == Format::d32Float || format == Format::d24UnormS8Uint;
        }

        [[nodiscard]] constexpr bool isStencilFormat(Format format) noexcept
        {
            return format == Format::d24UnormS8Uint;
        }

        [[nodiscard]] constexpr std::optional<VkImageAspectFlags> nativeAspectMask(Format format,
                                                                                   TextureAspect aspect) noexcept
        {
            switch (aspect)
            {
            case TextureAspect::color:
                return isDepthFormat(format) ? std::nullopt : std::optional{VK_IMAGE_ASPECT_COLOR_BIT};
            case TextureAspect::depth:
                return isDepthFormat(format) ? std::optional{VK_IMAGE_ASPECT_DEPTH_BIT} : std::nullopt;
            case TextureAspect::stencil:
                return isStencilFormat(format) ? std::optional{VK_IMAGE_ASPECT_STENCIL_BIT} : std::nullopt;
            }
            return std::nullopt;
        }

        [[nodiscard]] VkPipelineStageFlags2 nativeStages(PipelineStage stages) noexcept
        {
            if (hasAny(stages, PipelineStage::all))
            {
                return VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            }
            VkPipelineStageFlags2 flags = VK_PIPELINE_STAGE_2_NONE;
            if (hasAny(stages, PipelineStage::draw))
            {
                flags |= VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
            }
            if (hasAny(stages, PipelineStage::vertexInput))
            {
                flags |= VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT |
                         VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT;
            }
            if (hasAny(stages, PipelineStage::vertexShader))
            {
                flags |= VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
            }
            if (hasAny(stages, PipelineStage::pixelShader))
            {
                flags |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            }
            if (hasAny(stages, PipelineStage::computeShader))
            {
                flags |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            }
            if (hasAny(stages, PipelineStage::copy))
            {
                flags |= VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT;
            }
            if (hasAny(stages, PipelineStage::resolve))
            {
                flags |= VK_PIPELINE_STAGE_2_RESOLVE_BIT;
            }
            if (hasAny(stages, PipelineStage::renderTarget))
            {
                flags |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            }
            if (hasAny(stages, PipelineStage::depthStencil))
            {
                flags |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            }
            if (hasAny(stages, PipelineStage::indirect))
            {
                flags |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
            }
            if (hasAny(stages, PipelineStage::geometryShader))
            {
                flags |= VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT;
            }
            if (hasAny(stages, PipelineStage::hullShader))
            {
                flags |= VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT;
            }
            if (hasAny(stages, PipelineStage::domainShader))
            {
                flags |= VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT;
            }
            if (hasAny(stages, PipelineStage::taskShader))
            {
                flags |= VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT;
            }
            if (hasAny(stages, PipelineStage::meshShader))
            {
                flags |= VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
            }
            if (hasAny(stages, PipelineStage::conditionalExecution))
            {
                flags |= VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT;
            }
            if (hasAny(stages, PipelineStage::accelerationStructureBuild))
            {
                flags |= VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            }
            if (hasAny(stages, PipelineStage::rayTracingShader))
            {
                flags |= VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            }
            if (hasAny(stages, PipelineStage::accelerationStructureCopy))
            {
                flags |= VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR;
            }
            if (hasAny(stages, PipelineStage::streamOutput))
            {
                flags |= VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT;
            }
            return flags;
        }

        [[nodiscard]] VkPipelineStageFlags2 nativeStagesForAccess(PipelineStage stages, Access access) noexcept
        {
            auto flags = nativeStages(stages);
            if (hasAny(access, Access::vertexBuffer) && hasAny(stages, PipelineStage::vertexShader))
            {
                flags |= VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT;
            }
            if (hasAny(access, Access::shadingRateSource))
            {
                flags |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR;
            }
            if (hasAny(access, Access::unorderedAccess) && hasAny(stages, PipelineStage::computeShader) &&
                !hasAny(stages, PipelineStage::all))
            {
                flags |= VK_PIPELINE_STAGE_2_CLEAR_BIT;
            }
            return flags;
        }

        [[nodiscard]] VkAccessFlags2 nativeAccess(Access access) noexcept
        {
            if (access == Access::none)
            {
                return VK_ACCESS_2_NONE;
            }
            if (access == Access::common)
            {
                return VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
            }
            VkAccessFlags2 flags = VK_ACCESS_2_NONE;
            if (hasAny(access, Access::vertexBuffer))
            {
                flags |= VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
            }
            if (hasAny(access, Access::constantBuffer))
            {
                flags |= VK_ACCESS_2_UNIFORM_READ_BIT;
            }
            if (hasAny(access, Access::indexBuffer))
            {
                flags |= VK_ACCESS_2_INDEX_READ_BIT;
            }
            if (hasAny(access, Access::shaderResource))
            {
                flags |= VK_ACCESS_2_SHADER_READ_BIT;
            }
            if (hasAny(access, Access::unorderedAccess))
            {
                flags |= VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            }
            if (hasAny(access, Access::copySource))
            {
                flags |= VK_ACCESS_2_TRANSFER_READ_BIT;
            }
            if (hasAny(access, Access::copyDestination))
            {
                flags |= VK_ACCESS_2_TRANSFER_WRITE_BIT;
            }
            if (hasAny(access, Access::indirectArgument))
            {
                flags |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
            }
            if (hasAny(access, Access::renderTarget))
            {
                flags |= VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            }
            if (hasAny(access, Access::depthStencilRead))
            {
                flags |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            }
            if (hasAny(access, Access::depthStencilWrite))
            {
                flags |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            }
            if (hasAny(access, Access::resolveSource))
            {
                flags |= VK_ACCESS_2_TRANSFER_READ_BIT;
            }
            if (hasAny(access, Access::resolveDestination))
            {
                flags |= VK_ACCESS_2_TRANSFER_WRITE_BIT;
            }
            if (hasAny(access, Access::predicate))
            {
                flags |= VK_ACCESS_2_CONDITIONAL_RENDERING_READ_BIT_EXT;
            }
            if (hasAny(access, Access::shadingRateSource))
            {
                flags |= VK_ACCESS_2_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR;
            }
            if (hasAny(access, Access::accelerationStructureBuildInput))
            {
                flags |= VK_ACCESS_2_SHADER_READ_BIT;
            }
            if (hasAny(access, Access::accelerationStructureScratch))
            {
                flags |=
                    VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            }
            if (hasAny(access, Access::streamOutputWrite))
            {
                flags |= VK_ACCESS_2_TRANSFORM_FEEDBACK_WRITE_BIT_EXT;
            }
            return flags;
        }

        [[nodiscard]] VkAccessFlags2 nativeAccessForStages(PipelineStage stages, Access access) noexcept
        {
            auto flags = nativeAccess(access);
            if (hasAny(access, Access::unorderedAccess) &&
                hasAny(stages, PipelineStage::all | PipelineStage::computeShader))
            {
                flags |= VK_ACCESS_2_TRANSFER_WRITE_BIT;
            }
            return flags;
        }

        [[nodiscard]] constexpr VkImageLayout nativeLayout(TextureLayout layout) noexcept
        {
            switch (layout)
            {
            case TextureLayout::undefined:
                return VK_IMAGE_LAYOUT_UNDEFINED;
            case TextureLayout::common:
                return shaderImageDescriptorLayout;
            case TextureLayout::present:
                return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            case TextureLayout::shaderResource:
                return shaderImageDescriptorLayout;
            case TextureLayout::unorderedAccess:
                return shaderImageDescriptorLayout;
            case TextureLayout::renderTarget:
                return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            case TextureLayout::depthStencilWrite:
                return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            case TextureLayout::depthStencilRead:
                return VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            case TextureLayout::copySource:
                return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            case TextureLayout::copyDestination:
                return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            case TextureLayout::resolveSource:
                return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            case TextureLayout::resolveDestination:
                return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            case TextureLayout::shadingRateSource:
                return VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR;
            }
            return VK_IMAGE_LAYOUT_UNDEFINED;
        }

        [[nodiscard]] constexpr std::uint32_t textureArraySize(const TextureDesc& desc) noexcept
        {
            return desc.dimension == TextureDimension::texture3D ? 1U : desc.depthOrArrayLayers;
        }

        [[nodiscard]] Result<NativeTextureRange> resolveTextureRange(const TextureDesc& desc,
                                                                     const TextureSubresourceRange& range)
        {
            const auto aspect = nativeAspectMask(desc.format, range.aspect);
            const auto arraySize = textureArraySize(desc);
            const std::uint32_t mipCount =
                range.mipLevelCount == remainingSubresources && range.firstMipLevel < desc.mipLevels
                    ? desc.mipLevels - range.firstMipLevel
                    : range.mipLevelCount;
            const std::uint32_t arrayCount =
                range.arrayLayerCount == remainingSubresources && range.firstArrayLayer < arraySize
                    ? arraySize - range.firstArrayLayer
                    : range.arrayLayerCount;
            if (!aspect.has_value() || range.firstMipLevel >= desc.mipLevels || mipCount == 0 ||
                mipCount > desc.mipLevels - range.firstMipLevel || range.firstArrayLayer >= arraySize ||
                arrayCount == 0 || arrayCount > arraySize - range.firstArrayLayer)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The texture barrier range has an invalid mip, layer, count, or aspect.",
                });
            }
            return NativeTextureRange{range.firstMipLevel, mipCount, range.firstArrayLayer, arrayCount, *aspect};
        }

        [[nodiscard]] Result<NativeTextureRange> resolveWholeTextureRange(const TextureDesc& desc)
        {
            auto range = resolveTextureRange(
                desc, {.aspect = isDepthFormat(desc.format) ? TextureAspect::depth : TextureAspect::color});
            if (range && isStencilFormat(desc.format))
            {
                range->aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
            return range;
        }

        [[nodiscard]] constexpr bool textureRangesOverlap(const NativeTextureRange& left,
                                                          const NativeTextureRange& right) noexcept
        {
            const auto mipOverlaps =
                left.firstMip < right.firstMip + right.mipCount && right.firstMip < left.firstMip + left.mipCount;
            const auto arrayOverlaps = left.firstArray < right.firstArray + right.arrayCount &&
                                       right.firstArray < left.firstArray + left.arrayCount;
            return mipOverlaps && arrayOverlaps && (left.aspect & right.aspect) != 0;
        }

        [[nodiscard]] constexpr bool supportsQueueFamily(const VulkanSharingFamilies& sharing,
                                                         std::uint32_t currentFamily) noexcept
        {
            return currentFamily == VK_QUEUE_FAMILY_IGNORED || sharing.count == 0U || sharing.contains(currentFamily);
        }

        [[nodiscard]] constexpr bool validOwnershipTransfer(OwnershipTransfer ownership) noexcept
        {
            return ownership == OwnershipTransfer::none || ownership == OwnershipTransfer::releaseExternal ||
                   ownership == OwnershipTransfer::acquireExternal;
        }

        struct NativeQueueFamilyTransfer final
        {
            std::uint32_t source = VK_QUEUE_FAMILY_IGNORED;
            std::uint32_t destination = VK_QUEUE_FAMILY_IGNORED;
        };

        [[nodiscard]] constexpr NativeQueueFamilyTransfer nativeQueueFamilyTransfer(
            OwnershipTransfer ownership, const VulkanSharingFamilies& sharing, std::uint32_t currentFamily) noexcept
        {
            const auto localFamily =
                sharing.sharingMode() == VK_SHARING_MODE_CONCURRENT ? VK_QUEUE_FAMILY_IGNORED : currentFamily;
            switch (ownership)
            {
            case OwnershipTransfer::releaseExternal:
                return {localFamily, VK_QUEUE_FAMILY_EXTERNAL};
            case OwnershipTransfer::acquireExternal:
                return {VK_QUEUE_FAMILY_EXTERNAL, localFamily};
            case OwnershipTransfer::none:
                break;
            }
            return {};
        }

        [[nodiscard]] Status validateBufferOwnershipTransfer(const BufferBarrier& barrier, bool shared)
        {
            if (barrier.ownership != OwnershipTransfer::none && !shared)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Buffer ownership transfers require a buffer created or imported through the shared-buffer API.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateTextureOwnershipTransfer(const TextureBarrier& barrier, bool shared)
        {
            if (barrier.ownership == OwnershipTransfer::none)
            {
                return {};
            }
            if (!shared)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Texture ownership transfers require a texture created or imported through the shared-texture "
                    "API.",
                });
            }
            if (barrier.layoutBefore != TextureLayout::common || barrier.layoutAfter != TextureLayout::common)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Texture ownership transfers require the common layout before and after the barrier.",
                });
            }
            return {};
        }

        [[nodiscard]] constexpr std::uint32_t familyFor(const VulkanQueueFamilies& families,
                                                        QueueType queueType) noexcept
        {
            switch (queueType)
            {
            case QueueType::graphics:
                return families.graphics;
            case QueueType::compute:
                return families.compute;
            case QueueType::copy:
                return families.transfer;
            }
            return VK_QUEUE_FAMILY_IGNORED;
        }

        [[nodiscard]] constexpr validation_detail::CanonicalTransferLayout canonicalTransferLayout(
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

        template<typename Visit>
        void forEachLayoutCell(const TextureDesc& desc, const VkImageSubresourceRange& range, Visit&& visit)
        {
            const auto layers = textureArraySize(desc);
            for (const auto aspect :
                 {VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_ASPECT_STENCIL_BIT})
            {
                if ((range.aspectMask & aspect) == 0)
                {
                    continue;
                }
                const auto plane = aspect == VK_IMAGE_ASPECT_STENCIL_BIT ? 1U : 0U;
                const auto portableAspect = aspect == VK_IMAGE_ASPECT_STENCIL_BIT ? TextureAspect::stencil
                                            : aspect == VK_IMAGE_ASPECT_DEPTH_BIT ? TextureAspect::depth
                                                                                  : TextureAspect::color;
                for (std::uint32_t layer = range.baseArrayLayer; layer < range.baseArrayLayer + range.layerCount;
                     ++layer)
                {
                    for (std::uint32_t mip = range.baseMipLevel; mip < range.baseMipLevel + range.levelCount; ++mip)
                    {
                        const auto index = (static_cast<std::size_t>(plane) * layers + layer) * desc.mipLevels + mip;
                        visit(index,
                              TextureSubresource{.mipLevel = mip, .arrayLayer = layer, .aspect = portableAspect});
                    }
                }
            }
        }

        [[nodiscard]] constexpr std::string_view textureLayoutName(TextureLayout layout) noexcept
        {
            switch (layout)
            {
            case TextureLayout::undefined:
                return "undefined";
            case TextureLayout::common:
                return "common";
            case TextureLayout::present:
                return "present";
            case TextureLayout::shaderResource:
                return "shaderResource";
            case TextureLayout::unorderedAccess:
                return "unorderedAccess";
            case TextureLayout::renderTarget:
                return "renderTarget";
            case TextureLayout::depthStencilWrite:
                return "depthStencilWrite";
            case TextureLayout::depthStencilRead:
                return "depthStencilRead";
            case TextureLayout::copySource:
                return "copySource";
            case TextureLayout::copyDestination:
                return "copyDestination";
            case TextureLayout::resolveSource:
                return "resolveSource";
            case TextureLayout::resolveDestination:
                return "resolveDestination";
            case TextureLayout::shadingRateSource:
                return "shadingRateSource";
            }
            return "an unknown layout";
        }

        [[nodiscard]] constexpr std::string_view portableAspectName(TextureAspect aspect) noexcept
        {
            switch (aspect)
            {
            case TextureAspect::depth:
                return "depth";
            case TextureAspect::stencil:
                return "stencil";
            case TextureAspect::color:
                break;
            }
            return "color";
        }

        [[nodiscard]] Error barrierEntryLayoutMismatch(TextureLayout layoutBefore,
                                                       TextureLayout current,
                                                       const TextureSubresource& subresource,
                                                       std::string_view where)
        {
            std::string message = "A texture barrier's layoutBefore (";
            message += textureLayoutName(layoutBefore);
            message += ") is not the current layout of ";
            message += portableAspectName(subresource.aspect);
            message += " mip ";
            message += std::to_string(subresource.mipLevel);
            message += ", array layer ";
            message += std::to_string(subresource.arrayLayer);
            message += ": ";
            message += where;
            message += " ";
            message += textureLayoutName(current);
            message += ". Name the current layout, or undefined to discard the contents.";
            return Error{ErrorCode::invalidState, 0, std::move(message)};
        }

        [[nodiscard]] Status validateResourceBarrierBatch(VkCommandBuffer command, const ResourceBarrierBatch& barriers)
        {
            if (command == VK_NULL_HANDLE)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Resource barriers require a live Vulkan command buffer.",
                });
            }
            const bool batchIsEmpty = barriers.globalBarriers.empty() && barriers.bufferBarriers.empty() &&
                                      barriers.textureBarriers.empty() && barriers.samplerFeedbackBarriers.empty();
            if (batchIsEmpty)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A resource-barrier batch must be non-empty and fit within native barrier-count limits.",
                });
            }
            if (!barriers.samplerFeedbackBarriers.empty())
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Sampler-feedback barriers are not supported on Vulkan.",
                });
            }
            return {};
        }

        [[nodiscard]] Result<std::vector<VkMemoryBarrier2>> translateGlobalBarriers(
            std::span<const GlobalBarrier> barriers, QueueType queueType, const BarrierStageSupport& support)
        {
            std::vector<VkMemoryBarrier2> globals;
            globals.reserve(barriers.size());
            for (const auto& barrier : barriers)
            {
                if (!validBarrierScope(barrier.stagesBefore, barrier.accessBefore, queueType) ||
                    !validBarrierScope(barrier.stagesAfter, barrier.accessAfter, queueType) ||
                    (barrier.accessBefore == Access::common && barrier.accessAfter != Access::common))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A global barrier contains an invalid or queue-incompatible stage/access scope.",
                    });
                }
                if (auto enabled = requireEnabledBarrierStages(
                        barrier.stagesBefore, barrier.accessBefore, barrier.stagesAfter, barrier.accessAfter, support);
                    !enabled)
                {
                    return std::unexpected(std::move(enabled.error()));
                }
                globals.push_back(VkMemoryBarrier2{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                    .pNext = {},
                    .srcStageMask = nativeStagesForAccess(barrier.stagesBefore & support.stages, barrier.accessBefore),
                    .srcAccessMask = nativeAccessForStages(barrier.stagesBefore & support.stages, barrier.accessBefore),
                    .dstStageMask = nativeStagesForAccess(barrier.stagesAfter & support.stages, barrier.accessAfter),
                    .dstAccessMask = nativeAccessForStages(barrier.stagesAfter & support.stages, barrier.accessAfter),
                });
            }
            return globals;
        }

        [[nodiscard]] Status validateBarrierFitsBuffer(const BufferBarrier& barrier,
                                                       const BufferDesc& desc,
                                                       const VulkanSharingFamilies& sharingFamilies,
                                                       std::uint32_t currentFamily)
        {
            if (!supportsBufferAccess(desc.usage, barrier.accessBefore) ||
                !supportsBufferAccess(desc.usage, barrier.accessAfter) ||
                !supportsQueueFamily(sharingFamilies, currentFamily))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A buffer barrier access is incompatible with the buffer usage or command queue.",
                });
            }
            if (!validRange(desc.size, barrier.offset, barrier.size))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A buffer barrier range is empty or out of bounds.",
                });
            }
            return {};
        }

        [[nodiscard]] VkBufferMemoryBarrier2 nativeBufferBarrier(const BufferBarrier& barrier,
                                                                 VkBuffer buffer,
                                                                 std::uint64_t bufferSize,
                                                                 PipelineStage enabledStages,
                                                                 const NativeQueueFamilyTransfer& families) noexcept
        {
            return {
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                .pNext = {},
                .srcStageMask = nativeStagesForAccess(barrier.stagesBefore & enabledStages, barrier.accessBefore),
                .srcAccessMask = nativeAccessForStages(barrier.stagesBefore & enabledStages, barrier.accessBefore),
                .dstStageMask = nativeStagesForAccess(barrier.stagesAfter & enabledStages, barrier.accessAfter),
                .dstAccessMask = nativeAccessForStages(barrier.stagesAfter & enabledStages, barrier.accessAfter),
                .srcQueueFamilyIndex = families.source,
                .dstQueueFamilyIndex = families.destination,
                .buffer = buffer,
                .offset = barrier.offset,
                .size = resolveSize(bufferSize, barrier.offset, barrier.size),
            };
        }

        [[nodiscard]] constexpr bool presentEndpointHasScope(TextureLayout layout,
                                                             PipelineStage stages,
                                                             Access access) noexcept
        {
            return layout == TextureLayout::present && (stages != PipelineStage::none || access != Access::none);
        }

        [[nodiscard]] Status validateTextureBarrierEndpoints(const TextureBarrier& barrier, QueueType queueType)
        {
            const bool scopesValid = validResourceBarrierScope(barrier.stagesBefore, barrier.accessBefore, queueType) &&
                                     validResourceBarrierScope(barrier.stagesAfter, barrier.accessAfter, queueType);
            const bool layoutsValid = validTextureLayout(barrier.layoutBefore) &&
                                      validTextureLayout(barrier.layoutAfter) &&
                                      barrier.layoutAfter != TextureLayout::undefined;
            const bool presentCarriesScope =
                presentEndpointHasScope(barrier.layoutBefore, barrier.stagesBefore, barrier.accessBefore) ||
                presentEndpointHasScope(barrier.layoutAfter, barrier.stagesAfter, barrier.accessAfter);
            if (!scopesValid || !layoutsValid || presentCarriesScope)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture barrier contains an unknown stage, access, or layout value, ends in the undefined "
                    "layout, or uses present outside its no-access endpoint scope.",
                });
            }
            if (!validOwnershipTransfer(barrier.ownership))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture barrier contains an unknown ownership transfer.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateBarrierFitsTexture(const TextureBarrier& barrier,
                                                        TextureUsage usage,
                                                        const VulkanSharingFamilies& sharingFamilies,
                                                        QueueType queueType,
                                                        std::uint32_t currentFamily)
        {
            if (!supportsLayout(usage, barrier.layoutBefore) || !supportsLayout(usage, barrier.layoutAfter) ||
                !supportsTextureLayoutOnQueue(barrier.layoutBefore, queueType) ||
                !supportsTextureLayoutOnQueue(barrier.layoutAfter, queueType) ||
                !supportsQueueFamily(sharingFamilies, currentFamily))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture barrier layout is incompatible with the texture usage or command queue.",
                });
            }
            if (!supportsTextureAccess(usage, barrier.accessBefore) ||
                !supportsTextureAccess(usage, barrier.accessAfter) ||
                !accessMatchesTextureLayout(barrier.layoutBefore, barrier.accessBefore) ||
                !accessMatchesTextureLayout(barrier.layoutAfter, barrier.accessAfter))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture barrier access is incompatible with the texture usage or layout.",
                });
            }
            return {};
        }

        [[nodiscard]] constexpr VkImageSubresourceRange nativeSubresourceRange(const NativeTextureRange& range) noexcept
        {
            return {
                .aspectMask = range.aspect,
                .baseMipLevel = range.firstMip,
                .levelCount = range.mipCount,
                .baseArrayLayer = range.firstArray,
                .layerCount = range.arrayCount,
            };
        }

        [[nodiscard]] VkImageMemoryBarrier2 nativeTextureBarrier(const TextureBarrier& barrier,
                                                                 VkImage image,
                                                                 const NativeTextureRange& range,
                                                                 PipelineStage enabledStages,
                                                                 const NativeQueueFamilyTransfer& families) noexcept
        {
            VkImageMemoryBarrier2 native{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .pNext = {},
                .srcStageMask = nativeStagesForAccess(barrier.stagesBefore & enabledStages, barrier.accessBefore),
                .srcAccessMask = nativeAccessForStages(barrier.stagesBefore & enabledStages, barrier.accessBefore),
                .dstStageMask = nativeStagesForAccess(barrier.stagesAfter & enabledStages, barrier.accessAfter),
                .dstAccessMask = nativeAccessForStages(barrier.stagesAfter & enabledStages, barrier.accessAfter),
                .oldLayout = nativeLayout(barrier.layoutBefore),
                .newLayout = nativeLayout(barrier.layoutAfter),
                .srcQueueFamilyIndex = families.source,
                .dstQueueFamilyIndex = families.destination,
                .image = image,
                .subresourceRange = nativeSubresourceRange(range),
            };
            ensureImageLayoutTransitionSourceScope(native);
            return native;
        }

        template<typename Barrier>
        void waitOnAllCommandsBefore(std::span<Barrier> barriers) noexcept
        {
            for (auto& barrier : barriers)
            {
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            }
        }

        template<typename Record>
        void retainKeepAlives(const std::vector<Record*>& records, std::vector<std::shared_ptr<void>>& retained)
        {
            for (const auto* record : records)
            {
                if (record != nullptr && record->keepAlive != nullptr)
                {
                    retained.push_back(record->keepAlive);
                }
            }
        }

        [[nodiscard]] VkDependencyInfo barrierDependencyInfo(
            std::span<const VkMemoryBarrier2> memoryBarriers,
            std::span<const VkBufferMemoryBarrier2> bufferBarriers,
            std::span<const VkImageMemoryBarrier2> imageBarriers) noexcept
        {
            return {
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pNext = {},
                .dependencyFlags = {},
                .memoryBarrierCount = static_cast<std::uint32_t>(memoryBarriers.size()),
                .pMemoryBarriers = memoryBarriers.empty() ? nullptr : memoryBarriers.data(),
                .bufferMemoryBarrierCount = static_cast<std::uint32_t>(bufferBarriers.size()),
                .pBufferMemoryBarriers = bufferBarriers.empty() ? nullptr : bufferBarriers.data(),
                .imageMemoryBarrierCount = static_cast<std::uint32_t>(imageBarriers.size()),
                .pImageMemoryBarriers = imageBarriers.empty() ? nullptr : imageBarriers.data(),
            };
        }

        [[nodiscard]] Status validateBufferAliasActivation(VkCommandBuffer command,
                                                           QueueType queueType,
                                                           const BufferAliasActivation& activation)
        {
            if (command == VK_NULL_HANDLE)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Buffer alias activation requires a live Vulkan command buffer.",
                });
            }
            if (activation.before == activation.after ||
                !validResourceBarrierScope(activation.stagesBefore, activation.accessBefore, queueType) ||
                !validResourceBarrierScope(activation.stagesAfter, activation.accessAfter, queueType) ||
                !hasWriteAccess(activation.accessAfter))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Buffer alias activation requires distinct aliases plus compatible scopes and a destination write.",
                });
            }
            return {};
        }

        [[nodiscard]] VkMemoryBarrier2 aliasDeactivationBarrier(PipelineStage stagesBefore,
                                                                Access accessBefore,
                                                                PipelineStage stagesAfter,
                                                                Access accessAfter) noexcept
        {
            return {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                .pNext = {},
                .srcStageMask = nativeStagesForAccess(stagesBefore, accessBefore),
                .srcAccessMask = nativeAccessForStages(stagesBefore, accessBefore),
                .dstStageMask = nativeStagesForAccess(stagesAfter, accessAfter),
                .dstAccessMask = nativeAccessForStages(stagesAfter, accessAfter),
            };
        }

        [[nodiscard]] VkBufferMemoryBarrier2 bufferAliasActivationBarrier(const BufferAliasActivation& activation,
                                                                          VkBuffer buffer,
                                                                          std::uint64_t bufferSize,
                                                                          PipelineStage enabledStages) noexcept
        {
            return {
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                .pNext = {},
                .srcStageMask = nativeStagesForAccess(activation.stagesBefore & enabledStages, activation.accessBefore),
                .srcAccessMask = VK_ACCESS_2_NONE,
                .dstStageMask = nativeStagesForAccess(activation.stagesAfter & enabledStages, activation.accessAfter),
                .dstAccessMask = nativeAccessForStages(activation.stagesAfter & enabledStages, activation.accessAfter),
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = buffer,
                .offset = 0,
                .size = bufferSize,
            };
        }

        void retainAliasKeepAlives(const std::shared_ptr<VulkanNativeKeepAlive>& before,
                                   const std::shared_ptr<VulkanNativeKeepAlive>& after,
                                   std::vector<std::shared_ptr<void>>& retained)
        {
            if (before != nullptr)
            {
                retained.push_back(before);
            }
            if (after != nullptr)
            {
                retained.push_back(after);
            }
        }

        [[nodiscard]] Status validateTextureAliasActivation(VkCommandBuffer command,
                                                            QueueType queueType,
                                                            const TextureAliasActivation& activation)
        {
            if (command == VK_NULL_HANDLE)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Texture alias activation requires a live Vulkan command buffer.",
                });
            }
            const bool layoutsDefined = activation.layoutBefore != TextureLayout::undefined &&
                                        activation.layoutAfter != TextureLayout::undefined &&
                                        validTextureLayout(activation.layoutBefore) &&
                                        validTextureLayout(activation.layoutAfter);
            const bool scopesValid =
                validResourceBarrierScope(activation.stagesBefore, activation.accessBefore, queueType) &&
                validResourceBarrierScope(activation.stagesAfter, activation.accessAfter, queueType);
            if (activation.before == activation.after || !layoutsDefined || !scopesValid ||
                !hasWriteAccess(activation.accessAfter))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Texture alias activation requires distinct aliases, defined layouts, compatible scopes, and a "
                    "destination write.",
                });
            }
            return {};
        }

        [[nodiscard]] constexpr bool supportsAliasEndpoint(TextureUsage usage,
                                                           TextureLayout layout,
                                                           Access access,
                                                           QueueType queueType) noexcept
        {
            return supportsLayout(usage, layout) && supportsTextureLayoutOnQueue(layout, queueType) &&
                   supportsTextureAccess(usage, access) && accessMatchesTextureLayout(layout, access);
        }

        [[nodiscard]] VkImageMemoryBarrier2 textureAliasActivationBarrier(const TextureAliasActivation& activation,
                                                                          VkImage image,
                                                                          const NativeTextureRange& range,
                                                                          PipelineStage enabledStages) noexcept
        {
            VkImageMemoryBarrier2 activate{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .pNext = {},
                .srcStageMask = nativeStagesForAccess(activation.stagesBefore & enabledStages, activation.accessBefore),
                .srcAccessMask = VK_ACCESS_2_NONE,
                .dstStageMask = nativeStagesForAccess(activation.stagesAfter & enabledStages, activation.accessAfter),
                .dstAccessMask = nativeAccessForStages(activation.stagesAfter & enabledStages, activation.accessAfter),
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = nativeLayout(activation.layoutAfter),
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange = nativeSubresourceRange(range),
            };
            ensureImageLayoutTransitionSourceScope(activate);
            return activate;
        }
    } // namespace

    Status VulkanResourceAllocator::bufferBarrier(VkCommandBuffer command,
                                                  QueueType queueType,
                                                  const BufferBarrier& barrier,
                                                  std::vector<std::shared_ptr<void>>* retained)
    {
        const std::array barriers{barrier};
        return resourceBarriers(command, queueType, ResourceBarrierBatch{.bufferBarriers = barriers}, retained);
    }

    Status VulkanResourceAllocator::textureBarrier(VkCommandBuffer command,
                                                   QueueType queueType,
                                                   const TextureBarrier& barrier,
                                                   std::vector<std::shared_ptr<void>>* retained)
    {
        const std::array barriers{barrier};
        return resourceBarriers(command, queueType, ResourceBarrierBatch{.textureBarriers = barriers}, retained);
    }

    const VulkanResourceAllocator::CommandLayoutState* VulkanResourceAllocator::commandLayoutState(
        VkCommandBuffer command) const noexcept
    {
        if (command == VK_NULL_HANDLE)
        {
            return nullptr;
        }
        const auto found = commandLayouts_.find(command);
        return found != commandLayouts_.end() ? &found->second : nullptr;
    }

    std::optional<TextureLayout> VulkanResourceAllocator::commandPublicLayout(const CommandLayoutState* state,
                                                                              const LayoutCell& cell) noexcept
    {
        if (state == nullptr)
        {
            return std::nullopt;
        }
        if (const auto write = state->writeIndex.find(cell);
            write != state->writeIndex.end() && state->writes[write->second].submittedPublic != nullptr)
        {
            return state->writes[write->second].publicLayout;
        }
        return std::nullopt;
    }

    void VulkanResourceAllocator::stageLayoutWrite(CommandLayoutState& state, const LayoutWrite& write)
    {
        const LayoutCell cell{write.submitted.get(), write.index};
        if (const auto found = state.writeIndex.find(cell); found != state.writeIndex.end())
        {
            auto& staged = state.writes[found->second];
            staged.layout = write.layout;
            staged.submittedPublic = write.submittedPublic;
            staged.publicLayout = write.publicLayout;
            return;
        }
        state.writeIndex.emplace(cell, state.writes.size());
        state.writes.push_back(write);
    }

    VkImageLayout VulkanResourceAllocator::recordedLayout(VkCommandBuffer command,
                                                          const TextureRecord& record,
                                                          TextureSubresource subresource) const
    {
        if (record.submittedLayouts == nullptr)
        {
            return record.layout;
        }
        const auto layers = record.desc.dimension == TextureDimension::texture3D ? 1U : record.desc.depthOrArrayLayers;
        const auto plane = subresource.aspect == TextureAspect::stencil ? 1U : 0U;
        const auto index = (static_cast<std::size_t>(plane) * layers + subresource.arrayLayer) * record.desc.mipLevels +
                           subresource.mipLevel;
        if (index >= record.submittedLayouts->size())
        {
            return VK_IMAGE_LAYOUT_UNDEFINED;
        }
        if (const auto* state = commandLayoutState(command); state != nullptr)
        {
            if (const auto write = state->writeIndex.find(LayoutCell{record.submittedLayouts.get(), index});
                write != state->writeIndex.end())
            {
                return state->writes[write->second].layout;
            }
        }
        return (*record.submittedLayouts)[index];
    }

    void VulkanResourceAllocator::ensureTrackedLayouts(TextureRecord& record)
    {
        const auto cells = static_cast<std::size_t>(record.desc.mipLevels) * textureArraySize(record.desc) * 2U;
        if (record.submittedLayouts == nullptr)
        {
            record.submittedLayouts = std::make_shared<std::vector<VkImageLayout>>(cells, record.layout);
        }
        if (record.submittedPublicLayouts == nullptr)
        {
            record.submittedPublicLayouts =
                std::make_shared<std::vector<TextureLayout>>(record.submittedLayouts->size(), TextureLayout::undefined);
        }
    }

    void VulkanResourceAllocator::applyCommittedLayout(const LayoutWrite& write) noexcept
    {
        (*write.submitted)[write.index] = write.layout;
        if (write.submittedPublic != nullptr && write.index < write.submittedPublic->size())
        {
            (*write.submittedPublic)[write.index] = write.publicLayout;
        }
    }

    void VulkanResourceAllocator::appendLayoutWrites(TextureRecord& record,
                                                     const VkImageSubresourceRange& range,
                                                     VkImageLayout layout,
                                                     TextureLayout publicLayout,
                                                     std::vector<LayoutWrite>& writes)
    {
        const auto layers = record.desc.dimension == TextureDimension::texture3D ? 1U : record.desc.depthOrArrayLayers;
        ensureTrackedLayouts(record);
        for (const auto aspect : {VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_ASPECT_STENCIL_BIT})
        {
            if ((range.aspectMask & aspect) == 0)
            {
                continue;
            }
            const auto plane = aspect == VK_IMAGE_ASPECT_STENCIL_BIT ? 1U : 0U;
            for (std::uint32_t layer = range.baseArrayLayer; layer < range.baseArrayLayer + range.layerCount; ++layer)
            {
                for (std::uint32_t mip = range.baseMipLevel; mip < range.baseMipLevel + range.levelCount; ++mip)
                {
                    const auto index = (static_cast<std::size_t>(plane) * layers + layer) * record.desc.mipLevels + mip;
                    writes.push_back(
                        {record.submittedLayouts, index, layout, record.submittedPublicLayouts, publicLayout});
                }
            }
        }
    }

    void VulkanResourceAllocator::stageLayout(VkCommandBuffer command,
                                              TextureRecord& record,
                                              const VkImageSubresourceRange& range,
                                              VkImageLayout layout,
                                              TextureLayout publicLayout)
    {
        std::vector<LayoutWrite> writes;
        appendLayoutWrites(record, range, layout, publicLayout, writes);
        auto& state = commandLayouts_[command];
        for (const auto& write : writes)
        {
            stageLayoutWrite(state, write);
        }
    }

    void VulkanResourceAllocator::recordTransferRequirementLocked(VkCommandBuffer command,
                                                                  TextureRecord& record,
                                                                  TextureSubresource subresource,
                                                                  VkImageLayout expectedLayout,
                                                                  validation_detail::SubresourceTransferUse use)
    {
        if (command == VK_NULL_HANDLE)
        {
            return;
        }
        const auto layers = record.desc.dimension == TextureDimension::texture3D ? 1U : record.desc.depthOrArrayLayers;
        ensureTrackedLayouts(record);
        const auto plane = subresource.aspect == TextureAspect::stencil ? 1U : 0U;
        const auto index = (static_cast<std::size_t>(plane) * layers + subresource.arrayLayer) * record.desc.mipLevels +
                           subresource.mipLevel;
        if (index >= record.submittedLayouts->size())
        {
            return;
        }
        const LayoutCell cell{record.submittedLayouts.get(), index};
        auto& state = commandLayouts_[command];
        if (state.writeIndex.contains(cell) || state.transferIndex.contains(cell))
        {
            return;
        }
        const VkImageAspectFlags aspectMask =
            subresource.aspect == TextureAspect::stencil
                ? VK_IMAGE_ASPECT_STENCIL_BIT
                : (subresource.aspect == TextureAspect::depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
        state.transfers.push_back(LayoutRequirement{
            .submitted = record.submittedLayouts,
            .keepAlive = record.keepAlive,
            .image = record.image,
            .aspectMask = aspectMask,
            .mipLevel = subresource.mipLevel,
            .arrayLayer = record.desc.dimension == TextureDimension::texture3D ? 0U : subresource.arrayLayer,
            .index = index,
            .expectedLayout = expectedLayout,
            .use = use,
            .hasSubsequentBarrier = false,
            .firstBarrierOldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });
        try
        {
            state.transferIndex.emplace(cell, state.transfers.size() - 1U);
        }
        catch (...)
        {
            state.transfers.pop_back();
            throw;
        }
    }

    VulkanResourceAllocator::CommandLayoutMark VulkanResourceAllocator::markCommandLayoutsLocked(
        VkCommandBuffer command) const noexcept
    {
        const auto* state = commandLayoutState(command);
        if (state == nullptr)
        {
            return {};
        }
        return CommandLayoutMark{.transfers = state->transfers.size(), .entries = state->entries.size()};
    }

    void VulkanResourceAllocator::rollBackCommandLayoutsLocked(VkCommandBuffer command,
                                                               const CommandLayoutMark& mark) noexcept
    {
        const auto found = commandLayouts_.find(command);
        if (found == commandLayouts_.end())
        {
            return;
        }
        auto& state = found->second;
        for (auto index = mark.transfers; index < state.transfers.size(); ++index)
        {
            const auto& transfer = state.transfers[index];
            state.transferIndex.erase(LayoutCell{transfer.submitted.get(), transfer.index});
        }
        if (mark.transfers < state.transfers.size())
        {
            state.transfers.erase(state.transfers.begin() + static_cast<std::ptrdiff_t>(mark.transfers),
                                  state.transfers.end());
        }
        if (mark.entries < state.entries.size())
        {
            state.entries.erase(state.entries.begin() + static_cast<std::ptrdiff_t>(mark.entries), state.entries.end());
        }
    }

    void VulkanResourceAllocator::recordSubresourceTransferRequirement(VkCommandBuffer command,
                                                                       TextureHandle texture,
                                                                       TextureSubresource subresource,
                                                                       VkImageLayout expectedLayout,
                                                                       validation_detail::SubresourceTransferUse use)
    {
        const std::scoped_lock lock(mutex_);
        auto* record = textures_.find(texture);
        if (record == nullptr || !record->alive || record->image == VK_NULL_HANDLE)
        {
            return;
        }
        recordTransferRequirementLocked(command, *record, subresource, expectedLayout, use);
    }

    void VulkanResourceAllocator::noteBarrierOldLayoutLocked(VkCommandBuffer command,
                                                             TextureRecord& record,
                                                             const VkImageSubresourceRange& range,
                                                             VkImageLayout oldLayout)
    {
        const auto found = commandLayouts_.find(command);
        if (found == commandLayouts_.end() || found->second.transfers.empty() || record.submittedLayouts == nullptr)
        {
            return;
        }
        auto& state = found->second;
        const auto layers = record.desc.dimension == TextureDimension::texture3D ? 1U : record.desc.depthOrArrayLayers;
        for (const auto aspect : {VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_ASPECT_STENCIL_BIT})
        {
            if ((range.aspectMask & aspect) == 0)
            {
                continue;
            }
            const auto plane = aspect == VK_IMAGE_ASPECT_STENCIL_BIT ? 1U : 0U;
            for (std::uint32_t layer = range.baseArrayLayer; layer < range.baseArrayLayer + range.layerCount; ++layer)
            {
                for (std::uint32_t mip = range.baseMipLevel; mip < range.baseMipLevel + range.levelCount; ++mip)
                {
                    const auto index = (static_cast<std::size_t>(plane) * layers + layer) * record.desc.mipLevels + mip;
                    const auto transfer = state.transferIndex.find(LayoutCell{record.submittedLayouts.get(), index});
                    if (transfer == state.transferIndex.end())
                    {
                        continue;
                    }
                    auto& requirement = state.transfers[transfer->second];
                    if (!requirement.hasSubsequentBarrier)
                    {
                        requirement.hasSubsequentBarrier = true;
                        requirement.firstBarrierOldLayout = oldLayout;
                    }
                }
            }
        }
    }

    class VulkanResourceAllocator::SubmittedLayoutSimulation final
    {
    public:
        [[nodiscard]] Result<std::vector<LayoutWrite>> planBridgeTransitions(
            const std::vector<LayoutRequirement>& requirements, CommandLayoutBridgePlan& plan)
        {
            std::vector<LayoutWrite> postBridgeRestores;
            for (const auto& requirement : requirements)
            {
                if (requirement.submitted == nullptr)
                {
                    continue;
                }
                auto decision = evaluateRequirement(requirement, layoutAt(requirement.submitted, requirement.index));
                if (!decision)
                {
                    return std::unexpected(std::move(decision.error()));
                }
                if (decision->requiresPreBridge)
                {
                    plan.preTransitions.push_back(bridgeTransition(requirement, decision->preBridge));
                    setLayout(requirement.submitted,
                              requirement.index,
                              static_cast<VkImageLayout>(decision->preBridge.newLayout));
                }
                if (decision->requiresPostBridge)
                {
                    plan.postTransitions.push_back(bridgeTransition(requirement, decision->postBridge));
                    postBridgeRestores.push_back({
                        requirement.submitted,
                        requirement.index,
                        static_cast<VkImageLayout>(decision->postBridge.newLayout),
                    });
                }
            }
            return postBridgeRestores;
        }

        [[nodiscard]] Status checkEntryRequirements(std::span<const LayoutEntryRequirement> entries)
        {
            for (const auto& entry : entries)
            {
                if (entry.submittedPublic == nullptr || entry.entryLayoutUnknown)
                {
                    continue;
                }
                const auto current = publicLayoutAt(entry.submittedPublic, entry.index);
                if (current != entry.layout)
                {
                    return std::unexpected(barrierEntryLayoutMismatch(
                        entry.layout, current, entry.subresource, "the work submitted before it leaves it in"));
                }
            }
            return {};
        }

        void applyWrites(std::span<const LayoutWrite> writes)
        {
            for (const auto& write : writes)
            {
                if (write.submitted == nullptr)
                {
                    continue;
                }
                setLayout(write.submitted, write.index, write.layout);
                if (write.submittedPublic != nullptr)
                {
                    auto& layouts = publicLayoutsOf(write.submittedPublic);
                    if (write.index < layouts.size())
                    {
                        layouts[write.index] = write.publicLayout;
                    }
                }
            }
        }

    private:
        [[nodiscard]] static Result<validation_detail::SubresourceLayoutBridgeDecision> evaluateRequirement(
            const LayoutRequirement& requirement, VkImageLayout actualLayout)
        {
            return validation_detail::evaluateSubresourceLayoutRequirement(
                static_cast<std::int32_t>(actualLayout),
                canonicalTransferLayout(actualLayout),
                static_cast<std::int32_t>(requirement.expectedLayout),
                canonicalTransferLayout(requirement.expectedLayout),
                requirement.use,
                requirement.hasSubsequentBarrier,
                static_cast<std::int32_t>(requirement.firstBarrierOldLayout),
                static_cast<std::int32_t>(VK_IMAGE_LAYOUT_UNDEFINED));
        }

        [[nodiscard]] static PendingImageBarrierTransition bridgeTransition(
            const LayoutRequirement& requirement, const validation_detail::SubresourceLayoutTransitionStep& bridge)
        {
            return {
                .image = requirement.image,
                .aspectMask = requirement.aspectMask,
                .baseMipLevel = requirement.mipLevel,
                .baseArrayLayer = requirement.arrayLayer,
                .oldLayout = static_cast<VkImageLayout>(bridge.oldLayout),
                .newLayout = static_cast<VkImageLayout>(bridge.newLayout),
                .keepAlive = requirement.keepAlive,
            };
        }

        [[nodiscard]] std::vector<VkImageLayout>& layoutsOf(
            const std::shared_ptr<std::vector<VkImageLayout>>& submitted)
        {
            auto [it, inserted] = layouts_.try_emplace(submitted.get());
            if (inserted)
            {
                it->second = *submitted;
            }
            return it->second;
        }

        [[nodiscard]] VkImageLayout layoutAt(const std::shared_ptr<std::vector<VkImageLayout>>& submitted,
                                             std::size_t index)
        {
            const auto& layouts = layoutsOf(submitted);
            return index < layouts.size() ? layouts[index] : VK_IMAGE_LAYOUT_UNDEFINED;
        }

        void setLayout(const std::shared_ptr<std::vector<VkImageLayout>>& submitted,
                       std::size_t index,
                       VkImageLayout layout)
        {
            auto& layouts = layoutsOf(submitted);
            if (index < layouts.size())
            {
                layouts[index] = layout;
            }
        }

        [[nodiscard]] std::vector<TextureLayout>& publicLayoutsOf(
            const std::shared_ptr<std::vector<TextureLayout>>& submitted)
        {
            auto [it, inserted] = publicLayouts_.try_emplace(submitted.get());
            if (inserted)
            {
                it->second = *submitted;
            }
            return it->second;
        }

        [[nodiscard]] TextureLayout publicLayoutAt(const std::shared_ptr<std::vector<TextureLayout>>& submitted,
                                                   std::size_t index)
        {
            const auto& layouts = publicLayoutsOf(submitted);
            return index < layouts.size() ? layouts[index] : TextureLayout::undefined;
        }

        std::unordered_map<const std::vector<VkImageLayout>*, std::vector<VkImageLayout>> layouts_;
        std::unordered_map<const std::vector<TextureLayout>*, std::vector<TextureLayout>> publicLayouts_;
    };

    Result<std::vector<VulkanResourceAllocator::CommandLayoutBridgePlan>> VulkanResourceAllocator::
        planBatchLayoutTransitions(std::span<const VkCommandBuffer> commands) const
    {
        const std::scoped_lock lock(mutex_);
        std::vector<CommandLayoutBridgePlan> plans(commands.size());
        SubmittedLayoutSimulation simulation;
        for (std::size_t cmdIndex = 0; cmdIndex < commands.size(); ++cmdIndex)
        {
            const auto* state = commandLayoutState(commands[cmdIndex]);
            if (state == nullptr)
            {
                continue;
            }
            auto postBridgeRestores = simulation.planBridgeTransitions(state->transfers, plans[cmdIndex]);
            if (!postBridgeRestores)
            {
                return std::unexpected(std::move(postBridgeRestores.error()));
            }
            if (auto entries = simulation.checkEntryRequirements(state->entries); !entries)
            {
                return std::unexpected(std::move(entries.error()));
            }
            simulation.applyWrites(state->writes);
            simulation.applyWrites(*postBridgeRestores);
        }
        return plans;
    }

    void VulkanResourceAllocator::commitCommandLayouts(VkCommandBuffer command) noexcept
    {
        const std::scoped_lock lock(mutex_);
        if (const auto found = commandLayouts_.find(command); found != commandLayouts_.end())
        {
            for (const auto& write : found->second.writes)
            {
                applyCommittedLayout(write);
            }
            commandLayouts_.erase(found);
        }
        if (const auto images = recordedSwapchainImages_.find(command); images != recordedSwapchainImages_.end())
        {
            for (const auto& lease : images->second)
            {
                lease->markSubmitted();
            }
            recordedSwapchainImages_.erase(images);
        }
    }

    void VulkanResourceAllocator::discardCommandLayouts(VkCommandBuffer command) noexcept
    {
        const std::scoped_lock lock(mutex_);
        commandLayouts_.erase(command);
        releaseSwapchainImagesLocked(command);
    }

    Status VulkanResourceAllocator::retainSwapchainImages(
        VkCommandBuffer command, std::span<const std::shared_ptr<VulkanSwapchainImageLease>> leases)
    {
        if (std::ranges::none_of(leases, [](const auto& lease) noexcept { return lease != nullptr; }))
        {
            return {};
        }
        const std::scoped_lock lock(mutex_);
        return retainSwapchainImagesLocked(command, leases);
    }

    Status VulkanResourceAllocator::retainSwapchainImagesLocked(
        VkCommandBuffer command, std::span<const std::shared_ptr<VulkanSwapchainImageLease>> leases)
    {
        const auto referenced =
            std::ranges::count_if(leases, [](const auto& lease) noexcept { return lease != nullptr; });
        if (referenced == 0 || command == VK_NULL_HANDLE)
        {
            return {};
        }
        std::vector<std::shared_ptr<VulkanSwapchainImageLease>>* retained = nullptr;
        try
        {
            retained = &recordedSwapchainImages_[command];
            detail::reserveAdditional(*retained, static_cast<std::size_t>(referenced));
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to retain a swapchain image command reference.",
            });
        }
        const auto previousCount = retained->size();
        for (const auto& lease : leases)
        {
            if (lease == nullptr || std::ranges::find(*retained, lease) != retained->end())
            {
                continue;
            }
            auto reference = lease->retainRecordedReference();
            if (!reference)
            {
                for (auto index = previousCount; index < retained->size(); ++index)
                {
                    (*retained)[index]->releaseRecordedReference();
                }
                retained->resize(previousCount);
                if (retained->empty())
                {
                    recordedSwapchainImages_.erase(command);
                }
                return reference;
            }
            retained->push_back(lease);
        }
        return {};
    }

    void VulkanResourceAllocator::releaseSwapchainImagesLocked(VkCommandBuffer command) noexcept
    {
        const auto found = recordedSwapchainImages_.find(command);
        if (found == recordedSwapchainImages_.end())
        {
            return;
        }
        for (const auto& lease : found->second)
        {
            lease->releaseRecordedReference();
        }
        recordedSwapchainImages_.erase(found);
    }
    struct VulkanResourceAllocator::ResolvedBufferBarriers final
    {
        std::vector<VkBufferMemoryBarrier2> native;
        std::vector<BufferRecord*> records;
    };

    struct VulkanResourceAllocator::ResolvedTextureBarriers final
    {
        std::vector<VkImageMemoryBarrier2> native;
        std::vector<TextureRecord*> records;
        std::vector<TextureLayout> layoutsBefore;
        std::vector<TextureLayout> layoutsAfter;
    };

    Result<VulkanResourceAllocator::ResolvedBufferBarriers> VulkanResourceAllocator::resolveBufferBarriersLocked(
        std::span<const BufferBarrier> barriers, QueueType queueType)
    {
        const auto currentFamily = familyFor(families_, queueType);
        const BarrierStageSupport support{.stages = barrierStages_,
                                          .shadingRateAttachments = variableRateShadingTier2_};
        ResolvedBufferBarriers resolved;
        resolved.native.reserve(barriers.size());
        for (const auto& barrier : barriers)
        {
            if (!validResourceBarrierScope(barrier.stagesBefore, barrier.accessBefore, queueType) ||
                !validResourceBarrierScope(barrier.stagesAfter, barrier.accessAfter, queueType))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A buffer barrier contains an invalid or queue-incompatible stage/access scope.",
                });
            }
            if (!validOwnershipTransfer(barrier.ownership))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A buffer barrier contains an unknown ownership transfer.",
                });
            }
            if (auto enabled = requireEnabledBarrierStages(
                    barrier.stagesBefore, barrier.accessBefore, barrier.stagesAfter, barrier.accessAfter, support);
                !enabled)
            {
                return std::unexpected(std::move(enabled.error()));
            }
            auto* record = findBuffer(barrier.buffer);
            if (record == nullptr || !record->alive || record->buffer == VK_NULL_HANDLE)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A buffer barrier references an invalid or stale buffer handle.",
                });
            }
            if (auto fits = validateBarrierFitsBuffer(barrier, record->desc, record->sharingFamilies, currentFamily);
                !fits)
            {
                return std::unexpected(std::move(fits.error()));
            }
            if (auto owned = validateBufferOwnershipTransfer(barrier, record->shared); !owned)
            {
                return std::unexpected(std::move(owned.error()));
            }
            if (std::ranges::find(resolved.records, record) != resolved.records.end())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A resource-barrier batch cannot contain multiple transitions for one buffer.",
                });
            }
            resolved.native.push_back(nativeBufferBarrier(
                barrier,
                record->buffer,
                record->desc.size,
                support.stages,
                nativeQueueFamilyTransfer(barrier.ownership, record->sharingFamilies, currentFamily)));
            resolved.records.push_back(record);
        }
        return resolved;
    }

    Result<VulkanResourceAllocator::ResolvedTextureBarriers> VulkanResourceAllocator::resolveTextureBarriersLocked(
        std::span<const TextureBarrier> barriers, QueueType queueType)
    {
        const auto currentFamily = familyFor(families_, queueType);
        const BarrierStageSupport support{.stages = barrierStages_,
                                          .shadingRateAttachments = variableRateShadingTier2_};
        ResolvedTextureBarriers resolved;
        std::vector<NativeTextureRange> ranges;
        resolved.native.reserve(barriers.size());
        for (const auto& barrier : barriers)
        {
            if (auto valid = validateTextureBarrierEndpoints(barrier, queueType); !valid)
            {
                return std::unexpected(std::move(valid.error()));
            }
            if (auto enabled = requireEnabledBarrierStages(
                    barrier.stagesBefore, barrier.accessBefore, barrier.stagesAfter, barrier.accessAfter, support);
                !enabled)
            {
                return std::unexpected(std::move(enabled.error()));
            }
            auto* record = textures_.find(barrier.texture);
            if (record == nullptr || !record->alive || record->image == VK_NULL_HANDLE)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture barrier references an invalid or stale texture handle.",
                });
            }
            auto range = resolveTextureRange(record->desc, barrier.subresources);
            if (!range)
            {
                return std::unexpected(std::move(range.error()));
            }
            if (auto fits = validateBarrierFitsTexture(
                    barrier, record->desc.usage, record->sharingFamilies, queueType, currentFamily);
                !fits)
            {
                return std::unexpected(std::move(fits.error()));
            }
            if (auto owned = validateTextureOwnershipTransfer(barrier, record->shared); !owned)
            {
                return std::unexpected(std::move(owned.error()));
            }
            for (std::size_t index = 0; index < resolved.records.size(); ++index)
            {
                if (resolved.records[index] == record && textureRangesOverlap(ranges[index], *range))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A resource-barrier batch cannot contain overlapping transitions for one texture.",
                    });
                }
            }
            resolved.native.push_back(nativeTextureBarrier(
                barrier,
                record->image,
                *range,
                support.stages,
                nativeQueueFamilyTransfer(barrier.ownership, record->sharingFamilies, currentFamily)));
            resolved.records.push_back(record);
            resolved.layoutsBefore.push_back(barrier.layoutBefore);
            resolved.layoutsAfter.push_back(barrier.layoutAfter);
            ranges.push_back(*range);
        }
        return resolved;
    }

    Status VulkanResourceAllocator::validateBarrierEntryLayoutsLocked(VkCommandBuffer command,
                                                                      const ResolvedTextureBarriers& textures) const
    {
        const auto* state = commandLayoutState(command);
        if (state == nullptr)
        {
            return {};
        }
        for (std::size_t index = 0; index < textures.records.size(); ++index)
        {
            const auto& barrier = textures.native[index];
            const auto& record = *textures.records[index];
            const auto layoutBefore = textures.layoutsBefore[index];
            if (layoutBefore == TextureLayout::undefined || record.submittedLayouts == nullptr)
            {
                continue;
            }
            std::optional<Error> mismatch;
            forEachLayoutCell(
                record.desc, barrier.subresourceRange, [&](std::size_t cell, const TextureSubresource& subresource) {
                    if (mismatch.has_value())
                    {
                        return;
                    }
                    const auto known = commandPublicLayout(state, LayoutCell{record.submittedLayouts.get(), cell});
                    if (known.has_value() && *known != layoutBefore)
                    {
                        mismatch = barrierEntryLayoutMismatch(
                            layoutBefore, *known, subresource, "this command buffer leaves it in");
                    }
                });
            if (mismatch.has_value())
            {
                return std::unexpected(std::move(*mismatch));
            }
        }
        return {};
    }

    void VulkanResourceAllocator::recordBarrierLayoutTransitionsLocked(VkCommandBuffer command,
                                                                       const ResolvedTextureBarriers& textures)
    {
        if (textures.records.empty())
        {
            return;
        }
        auto& state = commandLayouts_[command];
        for (std::size_t index = 0; index < textures.records.size(); ++index)
        {
            const auto& barrier = textures.native[index];
            auto& record = *textures.records[index];
            const auto layoutBefore = textures.layoutsBefore[index];
            const auto layoutAfter = textures.layoutsAfter[index];
            noteBarrierOldLayoutLocked(command, record, barrier.subresourceRange, barrier.oldLayout);
            ensureTrackedLayouts(record);
            forEachLayoutCell(
                record.desc, barrier.subresourceRange, [&](std::size_t cell, const TextureSubresource& subresource) {
                    const LayoutCell key{record.submittedLayouts.get(), cell};
                    if (layoutBefore != TextureLayout::undefined && !commandPublicLayout(&state, key).has_value())
                    {
                        state.entries.push_back(LayoutEntryRequirement{
                            .submittedPublic = record.submittedPublicLayouts,
                            .index = cell,
                            .layout = layoutBefore,
                            .subresource = subresource,
                            .entryLayoutUnknown = record.shared,
                        });
                    }
                    stageLayoutWrite(state,
                                     {
                                         .submitted = record.submittedLayouts,
                                         .index = cell,
                                         .layout = barrier.newLayout,
                                         .submittedPublic = record.submittedPublicLayouts,
                                         .publicLayout = layoutAfter,
                                     });
                });
        }
    }

#ifdef RACHIS_TESTING
    namespace
    {

        std::mutex capturedBarrierMutex;
        std::vector<VulkanCapturedBarrier>* capturedBarriers = nullptr;

        template<typename ResolvedBuffers, typename ResolvedTextures>
        void captureResolvedBarriers(const ResolvedBuffers& buffers,
                                     const ResolvedTextures& textures,
                                     std::uint32_t recordingFamily)
        {
            const std::scoped_lock lock(capturedBarrierMutex);
            if (capturedBarriers == nullptr)
            {
                return;
            }
            for (std::size_t index = 0; index < buffers.native.size(); ++index)
            {
                const auto& native = buffers.native[index];
                capturedBarriers->push_back(VulkanCapturedBarrier{
                    .image = false,
                    .sharingMode = buffers.records[index]->sharingFamilies.sharingMode(),
                    .recordingFamily = recordingFamily,
                    .srcQueueFamilyIndex = native.srcQueueFamilyIndex,
                    .dstQueueFamilyIndex = native.dstQueueFamilyIndex,
                    .srcStageMask = native.srcStageMask,
                    .dstStageMask = native.dstStageMask,
                });
            }
            for (std::size_t index = 0; index < textures.native.size(); ++index)
            {
                const auto& native = textures.native[index];
                capturedBarriers->push_back(VulkanCapturedBarrier{
                    .image = true,
                    .sharingMode = textures.records[index]->sharingFamilies.sharingMode(),
                    .recordingFamily = recordingFamily,
                    .srcQueueFamilyIndex = native.srcQueueFamilyIndex,
                    .dstQueueFamilyIndex = native.dstQueueFamilyIndex,
                    .oldLayout = native.oldLayout,
                    .newLayout = native.newLayout,
                    .srcStageMask = native.srcStageMask,
                    .dstStageMask = native.dstStageMask,
                });
            }
        }

    } // namespace

    void captureVulkanResourceBarriers(std::vector<VulkanCapturedBarrier>* sink) noexcept
    {
        const std::scoped_lock lock(capturedBarrierMutex);
        capturedBarriers = sink;
    }

    void captureAliasBarriers(const VkMemoryBarrier2& deactivate,
                              VkPipelineStageFlags2 activateSrcStages,
                              VkPipelineStageFlags2 activateDstStages,
                              VkAccessFlags2 activateDstAccess)
    {
        const std::scoped_lock lock(capturedBarrierMutex);
        if (capturedBarriers == nullptr)
        {
            return;
        }
        capturedBarriers->push_back(VulkanCapturedBarrier{
            .memory = true,
            .srcStageMask = deactivate.srcStageMask,
            .dstStageMask = deactivate.dstStageMask,
            .srcAccessMask = deactivate.srcAccessMask,
            .dstAccessMask = deactivate.dstAccessMask,
        });
        capturedBarriers->push_back(VulkanCapturedBarrier{
            .srcStageMask = activateSrcStages,
            .dstStageMask = activateDstStages,
            .dstAccessMask = activateDstAccess,
        });
    }
#endif

    Status VulkanResourceAllocator::resourceBarriers(VkCommandBuffer command,
                                                     QueueType queueType,
                                                     const ResourceBarrierBatch& barriers,
                                                     std::vector<std::shared_ptr<void>>* retained)
    {
        if (auto valid = validateResourceBarrierBatch(command, barriers); !valid)
        {
            return valid;
        }

        const std::scoped_lock lock(mutex_);
        if (table_ == nullptr || table_->vkCmdPipelineBarrier2 == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Vulkan resource barriers are unavailable because the device table is incomplete.",
            });
        }

        const BarrierStageSupport support{.stages = barrierStages_,
                                          .shadingRateAttachments = variableRateShadingTier2_};
        auto globals = translateGlobalBarriers(barriers.globalBarriers, queueType, support);
        if (!globals)
        {
            return std::unexpected(std::move(globals.error()));
        }
        auto buffers = resolveBufferBarriersLocked(barriers.bufferBarriers, queueType);
        if (!buffers)
        {
            return std::unexpected(std::move(buffers.error()));
        }
        auto textures = resolveTextureBarriersLocked(barriers.textureBarriers, queueType);
        if (!textures)
        {
            return std::unexpected(std::move(textures.error()));
        }
        if (auto entry = validateBarrierEntryLayoutsLocked(command, *textures); !entry)
        {
            return entry;
        }
        std::vector<std::shared_ptr<VulkanSwapchainImageLease>> swapchainLeases;
        for (const auto* record : textures->records)
        {
            if (record->swapchainLease != nullptr)
            {
                swapchainLeases.push_back(record->swapchainLease);
            }
        }
        if (auto referenced = retainSwapchainImagesLocked(command, swapchainLeases); !referenced)
        {
            return referenced;
        }

        if (retained != nullptr)
        {
            retainKeepAlives(buffers->records, *retained);
            retainKeepAlives(textures->records, *retained);
        }
        if (allCommandsBarrierSources_)
        {
            waitOnAllCommandsBefore(std::span(*globals));
            waitOnAllCommandsBefore(std::span(buffers->native));
            waitOnAllCommandsBefore(std::span(textures->native));
        }
#ifdef RACHIS_TESTING
        captureResolvedBarriers(*buffers, *textures, familyFor(families_, queueType));
#endif
        recordBarrierLayoutTransitionsLocked(command, *textures);
        const auto dependency = barrierDependencyInfo(*globals, buffers->native, textures->native);
        table_->vkCmdPipelineBarrier2(command, &dependency);
        return {};
    }

    Status VulkanResourceAllocator::activateBufferAlias(VkCommandBuffer command,
                                                        QueueType queueType,
                                                        const BufferAliasActivation& activation,
                                                        std::vector<std::shared_ptr<void>>* retained)
    {
        if (auto valid = validateBufferAliasActivation(command, queueType, activation); !valid)
        {
            return valid;
        }
        const BarrierStageSupport support{.stages = barrierStages_,
                                          .shadingRateAttachments = variableRateShadingTier2_};
        if (auto enabled = requireEnabledBarrierStages(activation.stagesBefore,
                                                       activation.accessBefore,
                                                       activation.stagesAfter,
                                                       activation.accessAfter,
                                                       support);
            !enabled)
        {
            return enabled;
        }

        const std::scoped_lock lock(mutex_);
        if (table_ == nullptr || table_->vkCmdPipelineBarrier2 == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Vulkan buffer alias activation is unavailable because the device table is incomplete.",
            });
        }
        auto* before = findBuffer(activation.before);
        auto* after = findBuffer(activation.after);
        if (before == nullptr || !before->alive || before->buffer == VK_NULL_HANDLE || after == nullptr ||
            !after->alive || after->buffer == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Buffer alias activation references an invalid or stale buffer handle.",
            });
        }
        if (!placementsAlias(before->placement, after->placement))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Buffer alias activation requires overlapping placed resources from the same aliasable heap.",
            });
        }
        const auto currentFamily = familyFor(families_, queueType);
        if (!supportsBufferAccess(before->desc.usage, activation.accessBefore) ||
            !supportsBufferAccess(after->desc.usage, activation.accessAfter) ||
            !supportsQueueFamily(after->sharingFamilies, currentFamily))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A buffer alias activation access is incompatible with its buffer usage or command queue.",
            });
        }

        auto deactivate = aliasDeactivationBarrier(activation.stagesBefore & support.stages,
                                                   activation.accessBefore,
                                                   activation.stagesAfter & support.stages,
                                                   activation.accessAfter);
        auto activate = bufferAliasActivationBarrier(activation, after->buffer, after->desc.size, support.stages);
        if (allCommandsBarrierSources_)
        {
            waitOnAllCommandsBefore(std::span(&deactivate, 1));
            waitOnAllCommandsBefore(std::span(&activate, 1));
        }
#ifdef RACHIS_TESTING
        captureAliasBarriers(deactivate, activate.srcStageMask, activate.dstStageMask, activate.dstAccessMask);
#endif
        const auto dependency = barrierDependencyInfo(std::span(&deactivate, 1), std::span(&activate, 1), {});
        table_->vkCmdPipelineBarrier2(command, &dependency);
        if (retained != nullptr)
        {
            retainAliasKeepAlives(before->keepAlive, after->keepAlive, *retained);
        }
        return {};
    }

    Status VulkanResourceAllocator::activateTextureAlias(VkCommandBuffer command,
                                                         QueueType queueType,
                                                         const TextureAliasActivation& activation,
                                                         std::vector<std::shared_ptr<void>>* retained)
    {
        if (auto valid = validateTextureAliasActivation(command, queueType, activation); !valid)
        {
            return valid;
        }
        const BarrierStageSupport support{.stages = barrierStages_,
                                          .shadingRateAttachments = variableRateShadingTier2_};
        if (auto enabled = requireEnabledBarrierStages(activation.stagesBefore,
                                                       activation.accessBefore,
                                                       activation.stagesAfter,
                                                       activation.accessAfter,
                                                       support);
            !enabled)
        {
            return enabled;
        }

        const std::scoped_lock lock(mutex_);
        if (table_ == nullptr || table_->vkCmdPipelineBarrier2 == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Vulkan texture alias activation is unavailable because the device table is incomplete.",
            });
        }
        auto* before = textures_.find(activation.before);
        auto* after = textures_.find(activation.after);
        if (before == nullptr || !before->alive || before->image == VK_NULL_HANDLE || after == nullptr ||
            !after->alive || after->image == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Texture alias activation references an invalid or stale texture handle.",
            });
        }
        if (!placementsAlias(before->placement, after->placement))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Texture alias activation requires overlapping placed resources from the same aliasable heap.",
            });
        }
        const auto currentFamily = familyFor(families_, queueType);
        const bool beforeEndpointSupported =
            supportsAliasEndpoint(before->desc.usage, activation.layoutBefore, activation.accessBefore, queueType);
        const bool afterEndpointSupported =
            supportsAliasEndpoint(after->desc.usage, activation.layoutAfter, activation.accessAfter, queueType) &&
            supportsQueueFamily(after->sharingFamilies, currentFamily);
        if (!beforeEndpointSupported || !afterEndpointSupported)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A texture alias activation scope is incompatible with its texture usage, layout, or command queue.",
            });
        }
        if (hasAny(after->desc.usage, TextureUsage::renderTarget | TextureUsage::depthStencil) &&
            queueType != QueueType::graphics)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Aliased attachment metadata must be initialized on a graphics queue.",
            });
        }

        auto afterRange = resolveWholeTextureRange(after->desc);
        if (!afterRange)
        {
            return std::unexpected(std::move(afterRange.error()));
        }

        auto deactivate = aliasDeactivationBarrier(activation.stagesBefore & support.stages,
                                                   activation.accessBefore,
                                                   activation.stagesAfter & support.stages,
                                                   activation.accessAfter);
        auto activate = textureAliasActivationBarrier(activation, after->image, *afterRange, support.stages);
        if (allCommandsBarrierSources_)
        {
            waitOnAllCommandsBefore(std::span(&deactivate, 1));
            waitOnAllCommandsBefore(std::span(&activate, 1));
        }
#ifdef RACHIS_TESTING
        captureAliasBarriers(deactivate, activate.srcStageMask, activate.dstStageMask, activate.dstAccessMask);
#endif
        const auto dependency = barrierDependencyInfo(std::span(&deactivate, 1), {}, std::span(&activate, 1));
        if (retained != nullptr)
        {
            retainAliasKeepAlives(before->keepAlive, after->keepAlive, *retained);
        }
        noteBarrierOldLayoutLocked(command, *after, activate.subresourceRange, activate.oldLayout);
        stageLayout(command, *after, activate.subresourceRange, activate.newLayout, activation.layoutAfter);
        table_->vkCmdPipelineBarrier2(command, &dependency);
        return {};
    }

#ifdef RACHIS_TESTING
    void VulkanResourceAllocator::markTextureSharedForTesting(TextureHandle texture) noexcept
    {
        const std::scoped_lock lock(mutex_);
        if (auto* record = textures_.find(texture); record != nullptr)
        {
            record->shared = true;
        }
    }
#endif

} // namespace rachis::vulkan
