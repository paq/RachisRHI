#include "Internal/VulkanGraphics.h"

#include "Common/VectorReserve.h"
#include "Internal/PortableValidation.h"
#include "Internal/VulkanDiagnostics.h"
#include "Internal/VulkanStageSupport.h"
#include "Internal/VulkanValidationHelpers.h"
#include "Rachis/Binding.h"

#ifdef RACHIS_TESTING
#include "Vulkan/VulkanTestHooks.h"
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <utility>
#include <vector>

namespace rachis::vulkan
{
    namespace
    {

        [[nodiscard]] constexpr bool isValidLoadOperation(LoadOperation operation) noexcept
        {
            switch (operation)
            {
            case LoadOperation::load:
            case LoadOperation::clear:
            case LoadOperation::discard:
                return true;
            }
            return false;
        }

        [[nodiscard]] constexpr bool isValidStoreOperation(StoreOperation operation) noexcept
        {
            switch (operation)
            {
            case StoreOperation::store:
            case StoreOperation::discard:
                return true;
            }
            return false;
        }

        [[nodiscard]] constexpr bool isContiguousViewMask(std::uint32_t mask) noexcept
        {
            if (mask == 0)
            {
                return true;
            }
            const auto normalized = mask >> std::countr_zero(mask);
            return (normalized & (normalized + 1U)) == 0;
        }

        [[nodiscard]] constexpr std::uint32_t multiviewLayerCount(std::uint32_t mask) noexcept
        {
            return std::bit_width(mask);
        }

        [[nodiscard]] constexpr VkAttachmentLoadOp vulkanLoadOp(LoadOperation operation) noexcept
        {
            switch (operation)
            {
            case LoadOperation::load:
                return VK_ATTACHMENT_LOAD_OP_LOAD;
            case LoadOperation::clear:
                return VK_ATTACHMENT_LOAD_OP_CLEAR;
            case LoadOperation::discard:
                return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            }
            return VK_ATTACHMENT_LOAD_OP_LOAD;
        }

        [[nodiscard]] constexpr VkAttachmentStoreOp vulkanStoreOp(StoreOperation operation) noexcept
        {
            switch (operation)
            {
            case StoreOperation::store:
                return VK_ATTACHMENT_STORE_OP_STORE;
            case StoreOperation::discard:
                return VK_ATTACHMENT_STORE_OP_DONT_CARE;
            }
            return VK_ATTACHMENT_STORE_OP_STORE;
        }

        [[nodiscard]] constexpr VkShaderStageFlags nativePushConstantStages(const DeviceCapabilities* capabilities,
                                                                            ShaderStage stages) noexcept
        {
            const auto enabled = capabilities != nullptr ? enabledShaderStages(*capabilities) : coreShaderStages;
            return nativeShaderStageFlags(stages & enabled);
        }

        [[nodiscard]] constexpr bool isValidShadingRate(ShadingRate rate) noexcept
        {
            switch (rate)
            {
            case ShadingRate::oneByOne:
            case ShadingRate::oneByTwo:
            case ShadingRate::twoByOne:
            case ShadingRate::twoByTwo:
            case ShadingRate::twoByFour:
            case ShadingRate::fourByTwo:
            case ShadingRate::fourByFour:
                return true;
            }
            return false;
        }

        [[nodiscard]] constexpr bool isAdditionalShadingRate(ShadingRate rate) noexcept
        {
            return rate == ShadingRate::twoByFour || rate == ShadingRate::fourByTwo || rate == ShadingRate::fourByFour;
        }

        [[nodiscard]] constexpr bool isValidShadingRateCombiner(ShadingRateCombiner combiner) noexcept
        {
            switch (combiner)
            {
            case ShadingRateCombiner::passthrough:
            case ShadingRateCombiner::override:
            case ShadingRateCombiner::minimum:
            case ShadingRateCombiner::maximum:
            case ShadingRateCombiner::sum:
                return true;
            }
            return false;
        }

        [[nodiscard]] constexpr std::optional<VkExtent2D> vulkanFragmentSize(ShadingRate rate) noexcept
        {
            switch (rate)
            {
            case ShadingRate::oneByOne:
                return VkExtent2D{1, 1};
            case ShadingRate::oneByTwo:
                return VkExtent2D{1, 2};
            case ShadingRate::twoByOne:
                return VkExtent2D{2, 1};
            case ShadingRate::twoByTwo:
                return VkExtent2D{2, 2};
            case ShadingRate::twoByFour:
                return VkExtent2D{2, 4};
            case ShadingRate::fourByTwo:
                return VkExtent2D{4, 2};
            case ShadingRate::fourByFour:
                return VkExtent2D{4, 4};
            default:
                return std::nullopt;
            }
        }

        [[nodiscard]] constexpr std::optional<VkFragmentShadingRateCombinerOpKHR> vulkanShadingRateCombiner(
            ShadingRateCombiner combiner) noexcept
        {
            switch (combiner)
            {
            case ShadingRateCombiner::passthrough:
                return VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR;
            case ShadingRateCombiner::override:
                return VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR;
            case ShadingRateCombiner::minimum:
                return VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MIN_KHR;
            case ShadingRateCombiner::maximum:
                return VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MAX_KHR;
            default:
                return std::nullopt;
            }
        }

        [[nodiscard]] constexpr bool supportsSampleCount(ShadingRate rate, std::uint32_t sampleCount) noexcept
        {
            switch (rate)
            {
            case ShadingRate::oneByOne:
                return true;
            case ShadingRate::oneByTwo:
            case ShadingRate::twoByOne:
            case ShadingRate::twoByTwo:
                return sampleCount <= 4;
            case ShadingRate::twoByFour:
                return sampleCount <= 2;
            case ShadingRate::fourByTwo:
            case ShadingRate::fourByFour:
                return sampleCount == 1;
            default:
                return false;
            }
        }

        [[nodiscard]] constexpr bool hasCompleteViewportState(const detail::GraphicsCommandState& state) noexcept
        {
            return state.viewportCount != 0 && state.viewportCount == state.scissorCount;
        }

        [[nodiscard]] std::optional<VkRect2D> clipScissorToRenderArea(const ScissorRect& scissor,
                                                                      const VulkanRenderingScope& rendering) noexcept
        {
            const auto left = static_cast<std::uint64_t>(scissor.x);
            const auto top = static_cast<std::uint64_t>(scissor.y);
            const auto right = (std::min)(left + scissor.width, static_cast<std::uint64_t>(rendering.width));
            const auto bottom = (std::min)(top + scissor.height, static_cast<std::uint64_t>(rendering.height));
            if (left >= right || top >= bottom)
            {
                return std::nullopt;
            }
            return VkRect2D{
                .offset = {scissor.x, scissor.y},
                .extent = {static_cast<std::uint32_t>(right - left), static_cast<std::uint32_t>(bottom - top)},
            };
        }

        template<std::size_t MaxSlots>
        [[nodiscard]] constexpr bool allBound(const detail::FixedBoolMask<MaxSlots>& values) noexcept
        {
            return values.all();
        }

        [[nodiscard]] Status makeClearRects(const VulkanRenderingScope& rendering,
                                            std::span<const ScissorRect> rectangles,
                                            std::vector<VkClearRect>& native)
        {
            const std::uint32_t layerCount = rendering.viewMask != 0 ? 1 : rendering.layerCount;
            if (rectangles.empty())
            {
                native.push_back(VkClearRect{
                    .rect =
                        {
                            .offset = {0, 0},
                            .extent = {rendering.width, rendering.height},
                        },
                    .baseArrayLayer = 0,
                    .layerCount = layerCount,
                });
                return {};
            }

            native.reserve(rectangles.size());
            for (const auto& rectangle : rectangles)
            {
                if (rectangle.width == 0 || rectangle.height == 0 || rectangle.x < 0 || rectangle.y < 0)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Clear rectangles require a non-empty, non-negative region.",
                    });
                }
                const auto right = static_cast<std::uint64_t>(rectangle.x) + rectangle.width;
                const auto bottom = static_cast<std::uint64_t>(rectangle.y) + rectangle.height;
                if (right > rendering.width || bottom > rendering.height)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Clear rectangles must stay inside the active rendering extent.",
                    });
                }
                native.push_back(VkClearRect{
                    .rect =
                        {
                            .offset = {rectangle.x, rectangle.y},
                            .extent = {rectangle.width, rectangle.height},
                        },
                    .baseArrayLayer = 0,
                    .layerCount = layerCount,
                });
            }
            return {};
        }

        [[nodiscard]] Status validateDynamicBufferOffsets(std::span<const VulkanNativeBindingSet> sets,
                                                          std::span<const std::uint32_t> dynamicOffsets,
                                                          const VulkanBufferBindingLimits& limits)
        {
            std::size_t offsetIndex = 0;
            for (const auto& set : sets)
            {
                if (set.dynamicBuffers.size() != set.dynamicBufferCount)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "The Vulkan binding set has inconsistent dynamic buffer metadata.",
                    });
                }
                for (const auto& dyn : set.dynamicBuffers)
                {
                    if (offsetIndex >= dynamicOffsets.size())
                    {
                        return std::unexpected(Error{
                            ErrorCode::invalidArgument,
                            0,
                            "The dynamic-offset count does not match the bound descriptor sets.",
                        });
                    }
                    const auto dynOffset = static_cast<std::uint64_t>(dynamicOffsets[offsetIndex++]);
                    const auto alignment = dyn.type == BindingType::dynamicUniformBuffer
                                               ? std::max<std::uint64_t>(1, limits.minUniformBufferOffsetAlignment)
                                               : std::max<std::uint64_t>(1, limits.minStorageBufferOffsetAlignment);
                    if (dynOffset % alignment != 0)
                    {
                        return std::unexpected(Error{
                            ErrorCode::invalidArgument,
                            0,
                            "A dynamic buffer offset is not aligned to the device's buffer offset alignment limit.",
                        });
                    }
                    if (dyn.offset > dyn.bufferSize || dyn.range > dyn.bufferSize - dyn.offset ||
                        dynOffset > dyn.bufferSize - dyn.offset - dyn.range)
                    {
                        return std::unexpected(Error{
                            ErrorCode::invalidArgument,
                            0,
                            "A dynamic buffer offset pushes the bound buffer range past the end of the buffer.",
                        });
                    }
                }
            }
            return {};
        }

        struct BindingSetScratch final
        {
            BindingSetScratch() = default;
            BindingSetScratch(const BindingSetScratch&) = delete;
            BindingSetScratch& operator=(const BindingSetScratch&) = delete;
            BindingSetScratch(BindingSetScratch&&) = delete;
            BindingSetScratch& operator=(BindingSetScratch&&) = delete;
            std::array<VulkanNativeBindingSet, maxBindingSets> stackResolved{};
            std::array<VkDescriptorSet, maxBindingSets> stackNative{};
            std::vector<VulkanNativeBindingSet> heapResolved;
            std::vector<VkDescriptorSet> heapNative;
            std::span<VulkanNativeBindingSet> resolvedSets;
            std::span<VkDescriptorSet> nativeSets;
        };

        [[nodiscard]] Status allocateBindingSetScratch(BindingSetScratch& scratch, std::size_t setCount)
        {
            if (setCount <= maxBindingSets)
            {
                scratch.resolvedSets = std::span<VulkanNativeBindingSet>(scratch.stackResolved.data(), setCount);
                scratch.nativeSets = std::span<VkDescriptorSet>(scratch.stackNative.data(), setCount);
                return {};
            }
            try
            {
                scratch.heapResolved.resize(setCount);
                scratch.heapNative.resize(setCount);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Binding-set scratch storage could not be allocated.",
                });
            }
            scratch.resolvedSets = scratch.heapResolved;
            scratch.nativeSets = scratch.heapNative;
            return {};
        }

        [[nodiscard]] Result<std::uint32_t> resolveBindingSetsForPipeline(const VulkanBindingManager& bindings,
                                                                          std::span<const BindingSetHandle> bindingSets,
                                                                          const VulkanNativePipeline& pipeline,
                                                                          std::uint32_t firstSet,
                                                                          bool retainResources,
                                                                          BindingSetScratch& scratch,
                                                                          const char* mismatchMessage)
        {
            std::uint32_t expectedDynamicOffsets = 0;
            for (std::size_t index = 0; index < bindingSets.size(); ++index)
            {
                auto native = bindings.retainBindingSetForCommand(bindingSets[index], retainResources);
                if (!native)
                {
                    return std::unexpected(std::move(native.error()));
                }
                if (native->setLayout != pipeline.setLayouts[firstSet + index] ||
                    native->dynamicBufferCount != pipeline.setDynamicCounts[firstSet + index])
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        mismatchMessage,
                    });
                }
                expectedDynamicOffsets += native->dynamicBufferCount;
                scratch.nativeSets[index] = native->descriptorSet;
                scratch.resolvedSets[index] = std::move(*native);
            }
            return expectedDynamicOffsets;
        }

        [[nodiscard]] Status retainBindingSetUses(VulkanCommandBuffer& commandBuffer,
                                                  std::span<VulkanNativeBindingSet> resolvedSets)
        {
            auto uses = commandBuffer.reserveCommandUses(resolvedSets.size());
            if (!uses)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Binding-set dependencies could not be retained.",
                });
            }
            for (auto& set : resolvedSets)
            {
                (*uses)->add(std::move(set.commandUse));
            }
            return {};
        }

        [[nodiscard]] Status bindDescriptorSetBatch(VulkanBindingManager* bindings,
                                                    const VolkDeviceTable* table,
                                                    VulkanCommandBuffer& commandBuffer,
                                                    std::vector<std::shared_ptr<void>>* retained,
                                                    VkPipelineBindPoint bindPoint,
                                                    const VulkanNativePipeline& pipeline,
                                                    std::uint32_t firstSet,
                                                    std::span<const BindingSetHandle> bindingSets,
                                                    std::span<const std::uint32_t> dynamicOffsets,
                                                    const VulkanBufferBindingLimits& bufferLimits,
                                                    detail::BindingSetMask& boundBindingSets,
                                                    const char* emptyOrExceedsMessage,
                                                    const char* mismatchMessage,
                                                    const char* offsetCountMessage)
        {
            if (bindingSets.empty() || firstSet >= pipeline.setRequired.size() ||
                bindingSets.size() > pipeline.setRequired.size() - firstSet)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    emptyOrExceedsMessage,
                });
            }
            if (bindings == nullptr || table == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The Vulkan graphics encoder is not initialized.",
                });
            }

            BindingSetScratch scratch;
            if (auto allocated = allocateBindingSetScratch(scratch, bindingSets.size()); !allocated)
            {
                return allocated;
            }
            auto expectedDynamicOffsets = resolveBindingSetsForPipeline(
                *bindings, bindingSets, pipeline, firstSet, retained != nullptr, scratch, mismatchMessage);
            if (!expectedDynamicOffsets)
            {
                return std::unexpected(std::move(expectedDynamicOffsets.error()));
            }
            if (dynamicOffsets.size() != *expectedDynamicOffsets)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    offsetCountMessage,
                });
            }
            if (auto validOffsets = validateDynamicBufferOffsets(scratch.resolvedSets, dynamicOffsets, bufferLimits);
                !validOffsets)
            {
                return validOffsets;
            }
            if (auto usesRetained = retainBindingSetUses(commandBuffer, scratch.resolvedSets); !usesRetained)
            {
                return usesRetained;
            }

            table->vkCmdBindDescriptorSets(commandBuffer.native(),
                                           bindPoint,
                                           pipeline.layout,
                                           firstSet,
                                           static_cast<std::uint32_t>(scratch.nativeSets.size()),
                                           scratch.nativeSets.data(),
                                           static_cast<std::uint32_t>(dynamicOffsets.size()),
                                           dynamicOffsets.empty() ? nullptr : dynamicOffsets.data());
            for (std::size_t index = 0; index < bindingSets.size(); ++index)
            {
                boundBindingSets[firstSet + index] = true;
            }
            return {};
        }

        struct ValidatedIndirectArguments final
        {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkBuffer countBuffer = VK_NULL_HANDLE;
            std::shared_ptr<void> keepAlive = {};
            std::shared_ptr<void> countKeepAlive = {};
            std::uint32_t stride = 0;
            bool recordWork = false;
            bool counted = false;
        };

        [[nodiscard]] Result<VulkanResourceAllocator::ResolvedBindingBuffer> resolveIndirectArgumentBuffer(
            const VulkanResourceAllocator& resources, BufferHandle argumentBuffer, std::uint64_t argumentOffset)
        {
            auto arguments = resources.resolveBindingBuffer(argumentBuffer);
            if (!arguments)
            {
                return std::unexpected(std::move(arguments.error()));
            }
            if (!hasAny(arguments->desc.usage, BufferUsage::indirect) || argumentOffset > arguments->desc.size)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The indirect argument buffer requires indirect usage and an in-bounds offset.",
                });
            }
            return arguments;
        }

        [[nodiscard]] Status validateIndirectArgumentSpan(std::uint32_t maxCount,
                                                          std::uint32_t stride,
                                                          std::uint32_t recordSize,
                                                          std::uint64_t availableBytes)
        {
            if (maxCount == 0)
            {
                return {};
            }
            const auto precedingRecords = static_cast<std::uint64_t>(maxCount - 1U);
            if (precedingRecords > (std::numeric_limits<std::uint64_t>::max() - recordSize) / stride)
            {
                return std::unexpected(Error{ErrorCode::invalidArgument, 0, "The indirect argument span overflows."});
            }
            const auto argumentBytes = precedingRecords * stride + recordSize;
            if (argumentBytes > availableBytes)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The indirect argument records exceed the argument buffer.",
                });
            }
            return {};
        }

        [[nodiscard]] Result<VulkanResourceAllocator::ResolvedBindingBuffer> resolveIndirectCountBuffer(
            const VulkanResourceAllocator& resources, BufferHandle countBuffer, std::uint64_t countOffset)
        {
            auto count = resources.resolveBindingBuffer(countBuffer);
            if (!count)
            {
                return std::unexpected(std::move(count.error()));
            }
            if (!hasAny(count->desc.usage, BufferUsage::indirect) || countOffset % indirectArgumentAlignment != 0 ||
                countOffset > count->desc.size || sizeof(std::uint32_t) > count->desc.size - countOffset)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The indirect count buffer requires indirect usage and an aligned in-bounds uint32 count.",
                });
            }
            return count;
        }

        [[nodiscard]] Status validateIndirectCountLimit(std::uint32_t maxCount, std::uint32_t maxIndirectCount)
        {
            if (maxCount > maxIndirectCount)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The indirect draw count exceeds the device's maxDrawIndirectCount limit.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateCountedIndirectSupport(std::uint32_t maxCount,
                                                            bool allowCounted,
                                                            std::uint32_t maxIndirectCount)
        {
            if (!allowCounted)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "GPU-counted indirect execution is not supported on this device.",
                });
            }
            return validateIndirectCountLimit(maxCount, maxIndirectCount);
        }

        [[nodiscard]] Status validateUncountedIndirectSupport(std::uint32_t maxCount,
                                                              bool allowMulti,
                                                              std::uint32_t maxIndirectCount)
        {
            if (maxCount > 1 && !allowMulti)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Multi-draw and multi-dispatch indirect batches are not supported on this device.",
                });
            }
            return validateIndirectCountLimit(maxCount, maxIndirectCount);
        }

        [[nodiscard]] Result<ValidatedIndirectArguments> validateIndirectArguments(
            VulkanResourceAllocator* resources,
            BufferHandle argumentBuffer,
            std::uint64_t argumentOffset,
            std::uint32_t maxCount,
            std::uint32_t stride,
            BufferHandle countBuffer,
            std::uint64_t countOffset,
            std::uint32_t recordSize,
            bool allowCounted = false,
            bool allowMulti = false,
            std::uint32_t maxIndirectCount = std::numeric_limits<std::uint32_t>::max())
        {
            const auto resolvedStride = stride == 0 ? recordSize : stride;
            if (resources == nullptr || !argumentBuffer.isValid() || argumentOffset % indirectArgumentAlignment != 0 ||
                resolvedStride < recordSize || resolvedStride % indirectArgumentAlignment != 0 ||
                (!countBuffer.isValid() && countOffset != 0))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Indirect execution requires an argument buffer, aligned offsets, and a record-sized aligned "
                    "stride.",
                });
            }

            auto arguments = resolveIndirectArgumentBuffer(*resources, argumentBuffer, argumentOffset);
            if (!arguments)
            {
                return std::unexpected(std::move(arguments.error()));
            }
            if (auto validSpan = validateIndirectArgumentSpan(
                    maxCount, resolvedStride, recordSize, arguments->desc.size - argumentOffset);
                !validSpan)
            {
                return std::unexpected(std::move(validSpan.error()));
            }

            if (countBuffer.isValid())
            {
                auto count = resolveIndirectCountBuffer(*resources, countBuffer, countOffset);
                if (!count)
                {
                    return std::unexpected(std::move(count.error()));
                }
                if (auto supported = validateCountedIndirectSupport(maxCount, allowCounted, maxIndirectCount);
                    !supported)
                {
                    return std::unexpected(std::move(supported.error()));
                }
                return ValidatedIndirectArguments{
                    .buffer = arguments->buffer,
                    .countBuffer = count->buffer,
                    .keepAlive = std::move(arguments->keepAlive),
                    .countKeepAlive = std::move(count->keepAlive),
                    .stride = resolvedStride,
                    .recordWork = maxCount != 0,
                    .counted = true,
                };
            }
            if (auto supported = validateUncountedIndirectSupport(maxCount, allowMulti, maxIndirectCount); !supported)
            {
                return std::unexpected(std::move(supported.error()));
            }

            return ValidatedIndirectArguments{
                .buffer = arguments->buffer,
                .keepAlive = std::move(arguments->keepAlive),
                .stride = resolvedStride,
                .recordWork = maxCount != 0,
            };
        }

        [[nodiscard]] Status validateDrawIndexPrefix(const VulkanNativePipeline& pipeline,
                                                     bool drawIndexPrefix,
                                                     std::uint32_t maxCount)
        {
            if (!drawIndexPrefix && maxCount > 1 && pipeline.readsDrawIndex)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The bound pipeline reads the draw index, so an indirect call with more than one record needs "
                    "drawIndexPrefix.",
                });
            }
            return {};
        }

        [[nodiscard]] constexpr std::uint32_t prefixedRecordSize(std::uint32_t argumentSize,
                                                                 bool drawIndexPrefix) noexcept
        {
            return argumentSize + (drawIndexPrefix ? drawIndexPrefixSize : 0U);
        }

        [[nodiscard]] constexpr std::uint64_t nativeArgumentOffset(std::uint64_t argumentOffset,
                                                                   bool drawIndexPrefix) noexcept
        {
            return argumentOffset + (drawIndexPrefix ? drawIndexPrefixSize : 0U);
        }

        [[nodiscard]] bool requestsMultiviewOrTimestampsWithoutAttachments(const RenderingDesc& desc) noexcept
        {
            const bool hasNoAttachments = desc.colorAttachments.empty() && !desc.depthStencilAttachment.has_value();
            return hasNoAttachments && (desc.viewMask != 0 || desc.timestampWrites.has_value());
        }

        [[nodiscard]] Status validateRenderingViewMask(std::uint32_t viewMask, const DeviceCapabilities& capabilities)
        {
            if (!isContiguousViewMask(viewMask))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A multiview rendering mask must contain one contiguous run of view bits.",
                });
            }
            if (viewMask != 0 && (capabilities.maxMultiviewViewCount == 0 ||
                                  multiviewLayerCount(viewMask) > capabilities.maxMultiviewViewCount))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The rendering view mask exceeds the device's multiview support.",
                });
            }
            return {};
        }

        struct RenderingScopeBuilder final
        {
            const VulkanResourceAllocator& resources;
            const DeviceCapabilities& capabilities;
            VulkanRenderingScope& rendering;
            bool extentInitialized = false;
            std::vector<std::shared_ptr<VulkanSwapchainImageLease>> swapchainLeases{};
        };

        [[nodiscard]] bool acceptAttachmentExtent(
            RenderingScopeBuilder& builder,
            const VulkanResourceAllocator::ResolvedRenderAttachment& attachment) noexcept
        {
            auto& rendering = builder.rendering;
            if (attachment.layerCount == 0 ||
                attachment.layerCount > builder.capabilities.limits.maxFramebufferLayers ||
                (rendering.viewMask != 0 && attachment.layerCount < multiviewLayerCount(rendering.viewMask)))
            {
                return false;
            }
            if (!builder.extentInitialized)
            {
                rendering.width = attachment.width;
                rendering.height = attachment.height;
                rendering.sampleCount = attachment.sampleCount;
                rendering.layerCount = attachment.layerCount;
                builder.extentInitialized = true;
                return true;
            }
            return rendering.width == attachment.width && rendering.height == attachment.height &&
                   rendering.sampleCount == attachment.sampleCount && rendering.layerCount == attachment.layerCount;
        }

        [[nodiscard]] Status validateColorAttachmentOperations(const ColorAttachment& attachment)
        {
            const auto& clear = attachment.clearValue;
            if (!isValidLoadOperation(attachment.loadOperation) || !isValidStoreOperation(attachment.storeOperation))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Color attachments require valid load and store operations.",
                });
            }
            if (attachment.loadOperation == LoadOperation::clear &&
                (!std::isfinite(clear.red) || !std::isfinite(clear.green) || !std::isfinite(clear.blue) ||
                 !std::isfinite(clear.alpha)))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Color clear values must be finite.",
                });
            }
            return {};
        }

        [[nodiscard]] VkAttachmentStoreOp vulkanRenderingStoreOp(StoreOperation requested,
                                                                 const DeviceCapabilities& capabilities) noexcept
        {
            return vulkanStoreOp(
                validation_detail::effectiveRenderingStoreOperation(requested, capabilities.variableRateShadingTier2));
        }

        [[nodiscard]] VkClearColorValue vulkanClearColorValue(Format format, const ClearColor& clear) noexcept
        {
            const std::array channels{clear.red, clear.green, clear.blue, clear.alpha};
            const auto encoding = validation_detail::colorClearEncoding(format);
            VkClearColorValue value{};
            for (std::size_t index = 0; index < channels.size(); ++index)
            {
                switch (encoding.numericType)
                {
                case validation_detail::ColorClearNumericType::floatingPoint:
                    value.float32[index] = channels[index];
                    break;
                case validation_detail::ColorClearNumericType::unsignedInteger:
                    value.uint32[index] =
                        validation_detail::unsignedIntegerClearChannel(channels[index], encoding.channelBits);
                    break;
                case validation_detail::ColorClearNumericType::signedInteger:
                    value.int32[index] =
                        validation_detail::signedIntegerClearChannel(channels[index], encoding.channelBits);
                    break;
                }
            }
            return value;
        }

        [[nodiscard]] VkRenderingAttachmentInfo makeColorAttachmentInfo(const ColorAttachment& attachment,
                                                                        VkImageView view,
                                                                        Format format,
                                                                        const DeviceCapabilities& capabilities) noexcept
        {
            VkClearValue clearValue{};
            clearValue.color = vulkanClearColorValue(format, attachment.clearValue);
            return VkRenderingAttachmentInfo{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = {},
                .imageView = view,
                .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .resolveMode = {},
                .resolveImageView = {},
                .resolveImageLayout = {},
                .loadOp = vulkanLoadOp(attachment.loadOperation),
                .storeOp = vulkanRenderingStoreOp(attachment.storeOperation, capabilities),
                .clearValue = clearValue,
            };
        }

        [[nodiscard]] Status addColorAttachment(RenderingScopeBuilder& builder, const ColorAttachment& attachment)
        {
            if (auto validOperations = validateColorAttachmentOperations(attachment); !validOperations)
            {
                return validOperations;
            }
            auto view = builder.resources.resolveRenderAttachment(attachment.view);
            if (!view)
            {
                return std::unexpected(std::move(view.error()));
            }
            if (view->usage != TextureViewUsage::renderTarget || !acceptAttachmentExtent(builder, *view))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Color attachments require render-target views with matching extent, sample count, and layer count "
                    "within the device limit.",
                });
            }
            auto& rendering = builder.rendering;
            if (view->swapchainLease != nullptr)
            {
                builder.swapchainLeases.push_back(view->swapchainLease);
            }
            rendering.attachments.push_back(view->keepAlive);
            rendering.attachments.push_back(view->textureKeepAlive);
            rendering.colorFormats.push_back(view->format);
            rendering.colorAttachments.push_back(
                makeColorAttachmentInfo(attachment, view->view, view->format, builder.capabilities));
            return {};
        }

        [[nodiscard]] Status validateDepthStencilAttachmentOperations(const DepthStencilAttachment& attachment)
        {
            if (!isValidLoadOperation(attachment.depthLoadOperation) ||
                !isValidStoreOperation(attachment.depthStoreOperation) ||
                !isValidLoadOperation(attachment.stencilLoadOperation) ||
                !isValidStoreOperation(attachment.stencilStoreOperation))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The depth-stencil attachment requires valid depth and stencil load and store operations.",
                });
            }
            if (attachment.depthLoadOperation == LoadOperation::clear &&
                (!std::isfinite(attachment.depthClearValue) || attachment.depthClearValue < 0.0F ||
                 attachment.depthClearValue > 1.0F))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Depth clear values must be finite and inside the portable [0, 1] range.",
                });
            }
            if (attachment.stencilClearValue > std::numeric_limits<std::uint8_t>::max())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Stencil clear values must fit in the portable 8-bit stencil range.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateDepthStencilAttachmentFormat(Format format,
                                                                  const DepthStencilAttachment& attachment)
        {
            if (format != Format::d32Float && format != Format::d24UnormS8Uint)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A rendering depth-stencil attachment has an unsupported format.",
                });
            }
            const bool hasStencil = format == Format::d24UnormS8Uint;
            if (!hasStencil &&
                (attachment.stencilLoadOperation != LoadOperation::load ||
                 attachment.stencilStoreOperation != StoreOperation::store || attachment.stencilClearValue != 0))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Stencil attachment operations require the portable d24UnormS8Uint format.",
                });
            }
            return {};
        }

        [[nodiscard]] VkRenderingAttachmentInfo makeDepthAttachmentInfo(const DepthStencilAttachment& attachment,
                                                                        VkImageView view,
                                                                        const DeviceCapabilities& capabilities) noexcept
        {
            VkClearValue clearValue{};
            clearValue.depthStencil.depth = attachment.depthClearValue;
            clearValue.depthStencil.stencil = attachment.stencilClearValue;
            return VkRenderingAttachmentInfo{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = {},
                .imageView = view,
                .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                .resolveMode = {},
                .resolveImageView = {},
                .resolveImageLayout = {},
                .loadOp = vulkanLoadOp(attachment.depthLoadOperation),
                .storeOp = vulkanRenderingStoreOp(attachment.depthStoreOperation, capabilities),
                .clearValue = clearValue,
            };
        }

        [[nodiscard]] Status addDepthStencilAttachment(RenderingScopeBuilder& builder,
                                                       const DepthStencilAttachment& attachment)
        {
            if (auto validOperations = validateDepthStencilAttachmentOperations(attachment); !validOperations)
            {
                return validOperations;
            }
            auto view = builder.resources.resolveRenderAttachment(attachment.view);
            if (!view)
            {
                return std::unexpected(std::move(view.error()));
            }
            if (view->usage != TextureViewUsage::depthStencil || !acceptAttachmentExtent(builder, *view))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The depth attachment requires a depth-stencil view matching the color extent, sample count, and "
                    "layer count within the device limit.",
                });
            }
            if (auto validFormat = validateDepthStencilAttachmentFormat(view->format, attachment); !validFormat)
            {
                return validFormat;
            }
            auto& rendering = builder.rendering;
            if (view->swapchainLease != nullptr)
            {
                builder.swapchainLeases.push_back(view->swapchainLease);
            }
            rendering.attachments.push_back(view->keepAlive);
            rendering.attachments.push_back(view->textureKeepAlive);
            rendering.depthStencilFormat = view->format;
            rendering.depthAttachment = makeDepthAttachmentInfo(attachment, view->view, builder.capabilities);
            if (view->format == Format::d24UnormS8Uint)
            {
                auto stencilInfo = *rendering.depthAttachment;
                stencilInfo.loadOp = vulkanLoadOp(attachment.stencilLoadOperation);
                stencilInfo.storeOp = vulkanRenderingStoreOp(attachment.stencilStoreOperation, builder.capabilities);
                rendering.stencilAttachment = stencilInfo;
            }
            return {};
        }

        [[nodiscard]] Status retainRenderingAttachments(std::vector<std::shared_ptr<void>>* retained,
                                                        const VulkanRenderingScope& rendering,
                                                        std::size_t extraOwnerCapacity)
        {
            if (retained == nullptr)
            {
                return {};
            }
            try
            {
                detail::reserveAdditional(*retained, rendering.attachments.size() + extraOwnerCapacity);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Rendering attachment dependencies could not be retained.",
                });
            }
            retained->insert(retained->end(), rendering.attachments.begin(), rendering.attachments.end());
            return {};
        }

    } // namespace

    void VulkanGraphicsEncoder::initialize(VkDevice device,
                                           const VolkDeviceTable* table,
                                           const DeviceCapabilities* capabilities,
                                           VulkanResourceAllocator* resources,
                                           VulkanPipelineManager* pipelines,
                                           VulkanBindingManager* bindings,
                                           bool depthBiasClamp,
                                           VulkanBufferBindingLimits bufferLimits,
                                           std::uint32_t maxDrawIndirectCount,
                                           bool clipScissorsToRenderArea) noexcept
    {
        device_ = device;
        table_ = table;
        capabilities_ = capabilities;
        resources_ = resources;
        pipelines_ = pipelines;
        bindings_ = bindings;
        depthBiasClampSupported_ = depthBiasClamp;
        bufferLimits_ = VulkanBufferBindingLimits{
            .minUniformBufferOffsetAlignment =
                bufferLimits.minUniformBufferOffsetAlignment != 0 ? bufferLimits.minUniformBufferOffsetAlignment : 256,
            .minStorageBufferOffsetAlignment =
                bufferLimits.minStorageBufferOffsetAlignment != 0 ? bufferLimits.minStorageBufferOffsetAlignment : 16,
            .maxUniformBufferRange =
                bufferLimits.maxUniformBufferRange != 0 ? bufferLimits.maxUniformBufferRange : 65536,
            .maxStorageBufferRange =
                bufferLimits.maxStorageBufferRange != 0 ? bufferLimits.maxStorageBufferRange : (1ULL << 30),
        };
        maxDrawIndirectCount_ = maxDrawIndirectCount;
        clipScissorsToRenderArea_ = clipScissorsToRenderArea;
    }

    std::vector<std::shared_ptr<void>>* VulkanGraphicsEncoder::retainSink(
        VulkanCommandBuffer& commandBuffer) const noexcept
    {
        if (commandBuffer.owner() == nullptr ||
            hasAny(commandBuffer.owner()->flags, CommandPoolFlag::unretainedResources))
        {
            return nullptr;
        }
        return &commandBuffer.retained();
    }

    Status VulkanGraphicsEncoder::requireRendering(const VulkanCommandBuffer& commandBuffer) const
    {
        if (commandBuffer.queueType() != QueueType::graphics || commandBuffer.graphicsState().renderingOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The command requires an active graphics rendering scope.",
            });
        }
        return {};
    }

    bool VulkanGraphicsEncoder::drawScissoredOut(const VulkanCommandBuffer& commandBuffer) const noexcept
    {
        if (!clipScissorsToRenderArea_)
        {
            return false;
        }
        const auto& state = commandBuffer.graphicsState();
        const auto* rendering = static_cast<const VulkanRenderingScope*>(state.renderingOwner.get());
        const auto* pipeline = static_cast<const VulkanNativePipeline*>(state.pipelineOwner.get());
        return rendering != nullptr && pipeline != nullptr && rendering->scissorsEmpty &&
               pipeline->rasterizationEnabled;
    }

    Status VulkanGraphicsEncoder::requireBoundPipeline(const VulkanCommandBuffer& commandBuffer) const
    {
        if (auto rendering = requireRendering(commandBuffer); !rendering)
        {
            return rendering;
        }
        if (commandBuffer.graphicsState().pipelineOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A graphics pipeline must be bound first.",
            });
        }
        return {};
    }

    Status VulkanGraphicsEncoder::beginRendering(VulkanCommandBuffer& commandBuffer,
                                                 const RenderingDesc& desc,
                                                 const std::function<void()>& beforeNativeBegin)
    {
        auto& state = commandBuffer.graphicsState();
        if (table_ == nullptr || resources_ == nullptr || capabilities_ == nullptr ||
            commandBuffer.native() == VK_NULL_HANDLE || commandBuffer.queueType() != QueueType::graphics ||
            state.renderingOwner != nullptr || desc.colorAttachments.size() > maxColorAttachments ||
            requestsMultiviewOrTimestampsWithoutAttachments(desc))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Rendering requires a graphics command buffer, a non-nested scope, and a valid attachment, timestamp, "
                "and multiview configuration.",
            });
        }
        if (auto validViewMask = validateRenderingViewMask(desc.viewMask, *capabilities_); !validViewMask)
        {
            return validViewMask;
        }

        auto rendering = std::make_shared<VulkanRenderingScope>();
        rendering->viewMask = desc.viewMask;
        rendering->colorAttachments.reserve(desc.colorAttachments.size());
        RenderingScopeBuilder builder{
            .resources = *resources_,
            .capabilities = *capabilities_,
            .rendering = *rendering,
        };
        for (const auto& attachment : desc.colorAttachments)
        {
            if (auto added = addColorAttachment(builder, attachment); !added)
            {
                return added;
            }
        }
        if (desc.depthStencilAttachment.has_value())
        {
            if (auto added = addDepthStencilAttachment(builder, *desc.depthStencilAttachment); !added)
            {
                return added;
            }
        }
        if (auto referenced = resources_->retainSwapchainImages(commandBuffer.native(), builder.swapchainLeases);
            !referenced)
        {
            return referenced;
        }
        const std::size_t extraTimestampOwner = desc.timestampWrites.has_value() ? 1U : 0U;
        if (auto attachmentsRetained =
                retainRenderingAttachments(retainSink(commandBuffer), *rendering, extraTimestampOwner);
            !attachmentsRetained)
        {
            return attachmentsRetained;
        }

        if (beforeNativeBegin)
        {
            beforeNativeBegin();
        }
        beginNativeRendering(commandBuffer, *rendering);
        if (capabilities_->variableRateShading)
        {
            const VkExtent2D fragmentSize{1, 1};
            const std::array combiners{
                VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
                VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
            };
            table_->vkCmdSetFragmentShadingRateKHR(commandBuffer.native(), &fragmentSize, combiners.data());
        }

        state = {};
        state.renderingOwner = std::move(rendering);
        commandBuffer.computeState() = {};
        commandBuffer.rayTracingState() = {};
        return {};
    }

    Status VulkanGraphicsEncoder::endRendering(VulkanCommandBuffer& commandBuffer)
    {
        if (auto rendering = requireRendering(commandBuffer); !rendering)
        {
            return rendering;
        }
        if (auto queriesEnded = detail::validateRenderingQueriesEnded(commandBuffer); !queriesEnded)
        {
            return queriesEnded;
        }
        table_->vkCmdEndRendering(commandBuffer.native());
        commandBuffer.graphicsState() = {};
        return {};
    }

    Status VulkanGraphicsEncoder::clearColorAttachment(VulkanCommandBuffer& commandBuffer,
                                                       std::uint32_t attachmentIndex,
                                                       const ClearColor& clearValue,
                                                       std::span<const ScissorRect> rectangles)
    {
        if (auto rendering = requireRendering(commandBuffer); !rendering)
        {
            return rendering;
        }
        auto* const scope = static_cast<VulkanRenderingScope*>(commandBuffer.graphicsState().renderingOwner.get());
        if (attachmentIndex >= scope->colorFormats.size() || !std::isfinite(clearValue.red) ||
            !std::isfinite(clearValue.green) || !std::isfinite(clearValue.blue) || !std::isfinite(clearValue.alpha))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A color clear requires a live color attachment and finite clear values.",
            });
        }
        std::vector<VkClearRect> nativeRects;
        if (auto prepared = makeClearRects(*scope, rectangles, nativeRects); !prepared)
        {
            return prepared;
        }
        VkClearValue value{};
        value.color = vulkanClearColorValue(scope->colorFormats[attachmentIndex], clearValue);
        const auto attachment = VkClearAttachment{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .colorAttachment = attachmentIndex,
            .clearValue = value,
        };
        table_->vkCmdClearAttachments(
            commandBuffer.native(), 1, &attachment, static_cast<std::uint32_t>(nativeRects.size()), nativeRects.data());
        return {};
    }

    Status VulkanGraphicsEncoder::clearDepthStencilAttachment(VulkanCommandBuffer& commandBuffer,
                                                              DepthStencilClearAspect aspects,
                                                              float depth,
                                                              std::uint32_t stencil,
                                                              std::span<const ScissorRect> rectangles)
    {
        if (auto rendering = requireRendering(commandBuffer); !rendering)
        {
            return rendering;
        }
        auto* const scope = static_cast<VulkanRenderingScope*>(commandBuffer.graphicsState().renderingOwner.get());
        const bool clearDepth = hasAny(aspects, DepthStencilClearAspect::depth);
        const bool clearStencil = hasAny(aspects, DepthStencilClearAspect::stencil);
        if (scope->depthStencilFormat == Format::unknown || aspects == DepthStencilClearAspect::none ||
            (clearDepth && (!std::isfinite(depth) || depth < 0.0F || depth > 1.0F)) ||
            stencil > std::numeric_limits<std::uint8_t>::max())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A depth-stencil clear requires a live attachment, selected aspects, and portable clear values.",
            });
        }
        if (clearStencil && scope->depthStencilFormat != Format::d24UnormS8Uint)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Stencil attachment operations require the portable d24UnormS8Uint format.",
            });
        }
        std::vector<VkClearRect> nativeRects;
        if (auto prepared = makeClearRects(*scope, rectangles, nativeRects); !prepared)
        {
            return prepared;
        }
        VkClearValue value{};
        value.depthStencil.depth = depth;
        value.depthStencil.stencil = stencil;
        VkImageAspectFlags aspectMask = 0;
        if (clearDepth)
        {
            aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
        }
        if (clearStencil)
        {
            aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        const auto attachment = VkClearAttachment{
            .aspectMask = aspectMask,
            .colorAttachment = {},
            .clearValue = value,
        };
        table_->vkCmdClearAttachments(
            commandBuffer.native(), 1, &attachment, static_cast<std::uint32_t>(nativeRects.size()), nativeRects.data());
        return {};
    }

    Status VulkanGraphicsEncoder::bindGraphicsPipeline(VulkanCommandBuffer& commandBuffer, PipelineHandle pipeline)
    {
        if (auto rendering = requireRendering(commandBuffer); !rendering)
        {
            return rendering;
        }
        if (pipelines_ == nullptr || table_ == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The Vulkan graphics encoder is not initialized.",
            });
        }
        auto native = pipelines_->nativePipeline(pipeline);
        if (!native || *native == nullptr || (*native)->compute || (*native)->rayTracing ||
            (*native)->pipeline == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The pipeline handle is invalid, stale, or not a graphics pipeline.",
            });
        }
        const auto& bound = *native;
        auto* const scope = static_cast<VulkanRenderingScope*>(commandBuffer.graphicsState().renderingOwner.get());
        if (bound->colorFormats != scope->colorFormats || bound->depthStencilFormat != scope->depthStencilFormat ||
            bound->sampleCount != scope->sampleCount || bound->viewMask != scope->viewMask)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The graphics pipeline attachment formats, sample count, or view mask do not match the active "
                "rendering scope.",
            });
        }

        auto* retained = retainSink(commandBuffer);
        auto& state = commandBuffer.graphicsState();
        try
        {
            if (retained != nullptr)
            {
                detail::reserveAdditional(*retained, 1U);
            }
            state.boundBindingSets.reserve(bound->setRequired.size());
            state.boundVertexBuffers.reserve(bound->vertexStrides.size());
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Graphics pipeline bind state could not be allocated.",
            });
        }

        table_->vkCmdBindPipeline(commandBuffer.native(), VK_PIPELINE_BIND_POINT_GRAPHICS, bound->pipeline);
        if (retained != nullptr)
        {
            retained->push_back(std::const_pointer_cast<void>(std::static_pointer_cast<const void>(bound)));
        }

        state.pipelineOwner = std::const_pointer_cast<VulkanNativePipeline>(bound);
        state.boundBindingSets.clear();
        for (const auto required : bound->setRequired)
        {
            state.boundBindingSets.push_back(required == 0);
        }
        state.boundVertexBuffers.assign(bound->vertexStrides.size(), false);
        state.boundStreamOutputBuffers.clear();
        state.indexBufferBound = false;
        state.blendConstantsRequired = bound->usesBlendConstants;
        state.blendConstantsSet = false;
        state.depthBiasRequired = bound->dynamicDepthBias;
        state.depthBiasSet = false;
        state.depthBoundsRequired = bound->depthBoundsTest;
        state.depthBoundsSet = false;
        state.stencilReferenceRequired = bound->stencilEnabled;
        state.stencilReferenceSet = false;
        state.stencilReferenceFacesSet = 0;
        return {};
    }

    Status VulkanGraphicsEncoder::bindGraphicsBindingSets(VulkanCommandBuffer& commandBuffer,
                                                          std::uint32_t firstSet,
                                                          std::span<const BindingSetHandle> bindingSets,
                                                          std::span<const std::uint32_t> dynamicOffsets)
    {
        if (auto pipelineState = requireBoundPipeline(commandBuffer); !pipelineState)
        {
            return pipelineState;
        }
        auto* const pipeline =
            static_cast<const VulkanNativePipeline*>(commandBuffer.graphicsState().pipelineOwner.get());
        return bindDescriptorSetBatch(bindings_,
                                      table_,
                                      commandBuffer,
                                      retainSink(commandBuffer),
                                      VK_PIPELINE_BIND_POINT_GRAPHICS,
                                      *pipeline,
                                      firstSet,
                                      bindingSets,
                                      dynamicOffsets,
                                      bufferLimits_,
                                      commandBuffer.graphicsState().boundBindingSets,
                                      "The graphics binding-set batch is empty or exceeds the pipeline layout.",
                                      "A graphics binding set does not match the pipeline layout.",
                                      "The dynamic-offset count does not match the bound graphics descriptor sets.");
    }

    Status VulkanGraphicsEncoder::setGraphicsPushConstants(VulkanCommandBuffer& commandBuffer,
                                                           ShaderStage visibility,
                                                           std::uint32_t offset,
                                                           std::span<const std::byte> data)
    {
        if (auto pipelineState = requireBoundPipeline(commandBuffer); !pipelineState)
        {
            return pipelineState;
        }
        auto* const pipeline =
            static_cast<const VulkanNativePipeline*>(commandBuffer.graphicsState().pipelineOwner.get());
        auto subranges = validation_detail::planGraphicsPushConstants(
            pipeline->pushConstantRanges, pipeline->nativePushConstantRanges, visibility, offset, data.size());
        if (!subranges)
        {
            return std::unexpected(std::move(subranges.error()));
        }
        for (const auto& subrange : *subranges)
        {
            table_->vkCmdPushConstants(commandBuffer.native(),
                                       pipeline->layout,
                                       nativePushConstantStages(capabilities_, subrange.stages),
                                       subrange.offset,
                                       subrange.size,
                                       data.data() + (subrange.offset - offset));
        }
        return {};
    }

    Status VulkanGraphicsEncoder::setViewports(VulkanCommandBuffer& commandBuffer, std::span<const Viewport> viewports)
    {
        if (auto rendering = requireRendering(commandBuffer); !rendering)
        {
            return rendering;
        }
        if (viewports.empty())
        {
            return std::unexpected(Error{ErrorCode::invalidArgument, 0, "At least one viewport must be supplied."});
        }
        if (viewports.size() > capabilities_->limits.maxViewports || viewports.size() > maxViewportCount)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The viewport count exceeds the device limit.",
            });
        }

        std::vector<VkViewport> native;
        native.reserve(viewports.size());
        for (const auto& viewport : viewports)
        {
            if (!std::isfinite(viewport.x) || !std::isfinite(viewport.y) || !std::isfinite(viewport.width) ||
                !std::isfinite(viewport.height) || !std::isfinite(viewport.minDepth) ||
                !std::isfinite(viewport.maxDepth) || viewport.width <= 0.0F || viewport.height <= 0.0F ||
                viewport.minDepth < 0.0F || viewport.maxDepth > 1.0F || viewport.minDepth > viewport.maxDepth)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Every viewport requires finite positive dimensions and an ordered depth range within zero and "
                    "one.",
                });
            }
            native.push_back(VkViewport{
                .x = viewport.x,
                .y = viewport.y + viewport.height,
                .width = viewport.width,
                .height = -viewport.height,
                .minDepth = viewport.minDepth,
                .maxDepth = viewport.maxDepth,
            });
        }
        if (capabilities_->multiViewport)
        {
            table_->vkCmdSetViewportWithCount(
                commandBuffer.native(), static_cast<std::uint32_t>(native.size()), native.data());
        }
        else
        {
            table_->vkCmdSetViewport(
                commandBuffer.native(), 0, static_cast<std::uint32_t>(native.size()), native.data());
        }
        commandBuffer.graphicsState().viewportCount = static_cast<std::uint32_t>(native.size());
        return {};
    }

    Status VulkanGraphicsEncoder::setScissors(VulkanCommandBuffer& commandBuffer, std::span<const ScissorRect> scissors)
    {
        if (auto rendering = requireRendering(commandBuffer); !rendering)
        {
            return rendering;
        }
        if (scissors.empty())
        {
            return std::unexpected(
                Error{ErrorCode::invalidArgument, 0, "At least one scissor rectangle must be supplied."});
        }
        if (scissors.size() > capabilities_->limits.maxViewports || scissors.size() > maxViewportCount)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The scissor-rectangle count exceeds the device limit.",
            });
        }

        auto* const scope = clipScissorsToRenderArea_
                                ? static_cast<VulkanRenderingScope*>(commandBuffer.graphicsState().renderingOwner.get())
                                : nullptr;
        std::vector<VkRect2D> native;
        native.reserve(scissors.size());
        bool empty = true;
        for (const auto& scissor : scissors)
        {
            if (scissor.width == 0 || scissor.height == 0 || scissor.x < 0 || scissor.y < 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Every scissor rectangle requires a non-empty, non-negative region.",
                });
            }
            if (scope != nullptr)
            {
                const auto clipped = clipScissorToRenderArea(scissor, *scope);
                empty = empty && !clipped.has_value();
                native.push_back(clipped.value_or(VkRect2D{}));
                continue;
            }
            native.push_back(VkRect2D{
                .offset = {scissor.x, scissor.y},
                .extent = {scissor.width, scissor.height},
            });
        }
        if (capabilities_->multiViewport)
        {
            table_->vkCmdSetScissorWithCount(
                commandBuffer.native(), static_cast<std::uint32_t>(native.size()), native.data());
        }
        else
        {
            table_->vkCmdSetScissor(
                commandBuffer.native(), 0, static_cast<std::uint32_t>(native.size()), native.data());
        }
        commandBuffer.graphicsState().scissorCount = static_cast<std::uint32_t>(native.size());
        if (scope != nullptr)
        {
            scope->scissorsEmpty = empty;
        }
        return {};
    }

    Status VulkanGraphicsEncoder::setBlendConstants(VulkanCommandBuffer& commandBuffer, const BlendConstants& constants)
    {
        if (auto rendering = requireRendering(commandBuffer); !rendering)
        {
            return rendering;
        }
        if (!std::isfinite(constants.red) || !std::isfinite(constants.green) || !std::isfinite(constants.blue) ||
            !std::isfinite(constants.alpha))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Blend constants must be finite.",
            });
        }
        const float values[]{constants.red, constants.green, constants.blue, constants.alpha};
        table_->vkCmdSetBlendConstants(commandBuffer.native(), values);
        commandBuffer.graphicsState().blendConstantsSet = true;
        return {};
    }

    Status VulkanGraphicsEncoder::setStencilReference(VulkanCommandBuffer& commandBuffer,
                                                      StencilFace face,
                                                      std::uint32_t reference)
    {
        constexpr std::uint8_t frontFaceMask = 1U << 0U;
        constexpr std::uint8_t backFaceMask = 1U << 1U;
        constexpr std::uint8_t bothFacesMask = frontFaceMask | backFaceMask;
        auto& state = commandBuffer.graphicsState();
        if (state.renderingOwner == nullptr || reference > std::numeric_limits<std::uint8_t>::max() ||
            (face != StencilFace::front && face != StencilFace::back && face != StencilFace::frontAndBack))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A stencil reference requires an active rendering scope, a valid face selector, and an 8-bit value.",
            });
        }
        if (face != StencilFace::frontAndBack && !capabilities_->independentFrontAndBackStencilReference)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Independent front and back stencil references are not available on this device.",
            });
        }

        const auto nativeReference = static_cast<std::uint8_t>(reference);
        if (face == StencilFace::frontAndBack)
        {
            table_->vkCmdSetStencilReference(commandBuffer.native(), VK_STENCIL_FACE_FRONT_AND_BACK, reference);
            state.frontStencilReference = nativeReference;
            state.backStencilReference = nativeReference;
            state.stencilReferenceFacesSet = bothFacesMask;
        }
        else
        {
            const auto nativeFace = face == StencilFace::front ? VK_STENCIL_FACE_FRONT_BIT : VK_STENCIL_FACE_BACK_BIT;
            table_->vkCmdSetStencilReference(commandBuffer.native(), nativeFace, reference);
            if (face == StencilFace::front)
            {
                state.frontStencilReference = nativeReference;
                state.stencilReferenceFacesSet |= frontFaceMask;
            }
            else
            {
                state.backStencilReference = nativeReference;
                state.stencilReferenceFacesSet |= backFaceMask;
            }
        }
        state.stencilReferenceSet = state.stencilReferenceFacesSet == bothFacesMask;
        return {};
    }

    Status VulkanGraphicsEncoder::setDepthBias(VulkanCommandBuffer& commandBuffer, const DepthBias& bias)
    {
        auto& state = commandBuffer.graphicsState();
        if (state.renderingOwner == nullptr || state.pipelineOwner == nullptr || !state.depthBiasRequired)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Dynamic depth bias requires an active graphics pipeline that opted into the state.",
            });
        }
        if (!std::isfinite(bias.constantFactor) || !std::isfinite(bias.clamp) || !std::isfinite(bias.slopeScale))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Dynamic depth-bias values must be finite.",
            });
        }
        if (auto valid = validation_detail::validateDepthBiasClampSupport(depthBiasClampSupported_, bias.clamp); !valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        table_->vkCmdSetDepthBias(commandBuffer.native(), bias.constantFactor, bias.clamp, bias.slopeScale);
        state.depthBiasSet = true;
        return {};
    }

    Status VulkanGraphicsEncoder::setDepthBounds(VulkanCommandBuffer& commandBuffer, const DepthBounds& bounds)
    {
        auto& state = commandBuffer.graphicsState();
        if (state.renderingOwner == nullptr || state.pipelineOwner == nullptr || !state.depthBoundsRequired)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Dynamic depth bounds require an active graphics pipeline that opted into the test.",
            });
        }
        if (!std::isfinite(bounds.minimum) || !std::isfinite(bounds.maximum) || bounds.minimum < 0.0F ||
            bounds.maximum > 1.0F || bounds.minimum > bounds.maximum)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Depth bounds must be a finite ordered range within [0, 1].",
            });
        }
        table_->vkCmdSetDepthBounds(commandBuffer.native(), bounds.minimum, bounds.maximum);
        state.depthBoundsSet = true;
        return {};
    }

    Status VulkanGraphicsEncoder::bindVertexBuffers(VulkanCommandBuffer& commandBuffer,
                                                    std::uint32_t firstSlot,
                                                    std::span<const VertexBufferBinding> bindings)
    {
        if (auto pipelineState = requireBoundPipeline(commandBuffer); !pipelineState)
        {
            return pipelineState;
        }
        auto* const pipeline =
            static_cast<const VulkanNativePipeline*>(commandBuffer.graphicsState().pipelineOwner.get());
        if (bindings.empty() || firstSlot >= pipeline->vertexStrides.size() ||
            bindings.size() > pipeline->vertexStrides.size() - firstSlot)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The vertex-buffer batch is empty or exceeds the pipeline's declared slots.",
            });
        }
        if (auto aligned = detail::validateVertexBufferOffsetAlignment(pipeline->vertexAlignments, firstSlot, bindings);
            !aligned)
        {
            return aligned;
        }

        auto* retained = retainSink(commandBuffer);
        std::vector<VkBuffer> nativeBuffers;
        std::vector<VkDeviceSize> offsets;
        std::vector<std::shared_ptr<void>> keepAlives;
        try
        {
            nativeBuffers.reserve(bindings.size());
            offsets.reserve(bindings.size());
            if (retained != nullptr)
            {
                keepAlives.reserve(bindings.size());
                detail::reserveAdditional(*retained, bindings.size());
            }
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Vertex-buffer dependencies could not be retained.",
            });
        }
        for (const auto& binding : bindings)
        {
            auto buffer = resources_->resolveBindingBuffer(binding.buffer);
            if (!buffer)
            {
                return std::unexpected(std::move(buffer.error()));
            }
            if (!hasAny(buffer->desc.usage, BufferUsage::vertex) || binding.offset >= buffer->desc.size)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A vertex binding requires a vertex-usage buffer and an in-bounds offset.",
                });
            }
            nativeBuffers.push_back(buffer->buffer);
            offsets.push_back(binding.offset);
            if (retained != nullptr)
            {
                keepAlives.push_back(buffer->keepAlive);
            }
        }
        table_->vkCmdBindVertexBuffers(commandBuffer.native(),
                                       firstSlot,
                                       static_cast<std::uint32_t>(nativeBuffers.size()),
                                       nativeBuffers.data(),
                                       offsets.data());
        if (retained != nullptr)
        {
            for (auto& owner : keepAlives)
            {
                retained->push_back(std::move(owner));
            }
        }
        auto& state = commandBuffer.graphicsState();
        for (std::size_t index = 0; index < bindings.size(); ++index)
        {
            state.boundVertexBuffers[firstSlot + index] = true;
        }
        return {};
    }

    Status VulkanGraphicsEncoder::bindIndexBuffer(VulkanCommandBuffer& commandBuffer,
                                                  BufferHandle buffer,
                                                  IndexFormat format,
                                                  std::uint64_t offset)
    {
        if (auto pipelineState = requireBoundPipeline(commandBuffer); !pipelineState)
        {
            return pipelineState;
        }
        if (format != IndexFormat::uint16 && format != IndexFormat::uint32)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The index format is invalid.",
            });
        }
        auto native = resources_->resolveBindingBuffer(buffer);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        const auto stride = format == IndexFormat::uint16 ? sizeof(std::uint16_t) : sizeof(std::uint32_t);
        if (!hasAny(native->desc.usage, BufferUsage::index) || offset >= native->desc.size ||
            ((native->desc.size - offset) < stride) || (offset % stride) != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "An index binding requires an index-usage buffer and an aligned in-bounds offset.",
            });
        }
        auto* retained = retainSink(commandBuffer);
        if (retained != nullptr)
        {
            try
            {
                detail::reserveAdditional(*retained, 1U);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Index-buffer dependency could not be retained.",
                });
            }
        }
        table_->vkCmdBindIndexBuffer(commandBuffer.native(),
                                     native->buffer,
                                     offset,
                                     format == IndexFormat::uint16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
        if (retained != nullptr)
        {
            retained->push_back(native->keepAlive);
        }
        commandBuffer.graphicsState().indexBufferBound = true;
        return {};
    }

    Status VulkanGraphicsEncoder::draw(VulkanCommandBuffer& commandBuffer,
                                       std::uint32_t vertexCount,
                                       std::uint32_t instanceCount,
                                       std::uint32_t firstVertex,
                                       std::uint32_t firstInstance)
    {
        static_cast<void>(firstVertex);
        static_cast<void>(firstInstance);
        if (auto pipelineState = requireBoundPipeline(commandBuffer); !pipelineState)
        {
            return pipelineState;
        }
        const auto& state = commandBuffer.graphicsState();
        auto* const pipeline = static_cast<const VulkanNativePipeline*>(state.pipelineOwner.get());
        if (pipeline->mesh || (pipeline->rasterizationEnabled && !hasCompleteViewportState(state)) ||
            (state.blendConstantsRequired && !state.blendConstantsSet) ||
            (state.depthBiasRequired && !state.depthBiasSet) || (state.depthBoundsRequired && !state.depthBoundsSet) ||
            (state.stencilReferenceRequired && !state.stencilReferenceSet) || !allBound(state.boundBindingSets) ||
            !allBound(state.boundVertexBuffers) || !allBound(state.boundStreamOutputBuffers) || vertexCount == 0 ||
            instanceCount == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Draw requires complete rendering, pipeline, binding, vertex, viewport, and scissor state.",
            });
        }
        if (drawScissoredOut(commandBuffer))
        {
            return {};
        }
        table_->vkCmdDraw(commandBuffer.native(), vertexCount, instanceCount, firstVertex, firstInstance);
        return {};
    }

    Status VulkanGraphicsEncoder::drawIndexed(VulkanCommandBuffer& commandBuffer,
                                              std::uint32_t indexCount,
                                              std::uint32_t instanceCount,
                                              std::uint32_t firstIndex,
                                              std::int32_t baseVertex,
                                              std::uint32_t firstInstance)
    {
        if (auto pipelineState = requireBoundPipeline(commandBuffer); !pipelineState)
        {
            return pipelineState;
        }
        const auto& state = commandBuffer.graphicsState();
        auto* const pipeline = static_cast<const VulkanNativePipeline*>(state.pipelineOwner.get());
        if (pipeline->mesh || (pipeline->rasterizationEnabled && !hasCompleteViewportState(state)) ||
            (state.blendConstantsRequired && !state.blendConstantsSet) ||
            (state.depthBiasRequired && !state.depthBiasSet) || (state.depthBoundsRequired && !state.depthBoundsSet) ||
            (state.stencilReferenceRequired && !state.stencilReferenceSet) || !state.indexBufferBound ||
            !allBound(state.boundBindingSets) || !allBound(state.boundVertexBuffers) ||
            !allBound(state.boundStreamOutputBuffers) || indexCount == 0 || instanceCount == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Indexed draw requires complete graphics state, a bound index buffer, and non-zero draw counts.",
            });
        }
        if (drawScissoredOut(commandBuffer))
        {
            return {};
        }
        table_->vkCmdDrawIndexed(
            commandBuffer.native(), indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
        return {};
    }

    Status VulkanGraphicsEncoder::drawIndirect(VulkanCommandBuffer& commandBuffer, const IndirectDrawDesc& desc)
    {
        if (auto pipelineState = requireBoundPipeline(commandBuffer); !pipelineState)
        {
            return pipelineState;
        }
        const auto& state = commandBuffer.graphicsState();
        auto* const pipeline = static_cast<const VulkanNativePipeline*>(state.pipelineOwner.get());
        if (pipeline->mesh || (pipeline->rasterizationEnabled && !hasCompleteViewportState(state)) ||
            (state.blendConstantsRequired && !state.blendConstantsSet) ||
            (state.depthBiasRequired && !state.depthBiasSet) || (state.depthBoundsRequired && !state.depthBoundsSet) ||
            (state.stencilReferenceRequired && !state.stencilReferenceSet) || !allBound(state.boundBindingSets) ||
            !allBound(state.boundVertexBuffers) || !allBound(state.boundStreamOutputBuffers))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Indirect draw requires a conventional graphics pipeline and complete rendering state.",
            });
        }
        if (auto prefix = validateDrawIndexPrefix(*pipeline, desc.drawIndexPrefix, desc.maxDrawCount); !prefix)
        {
            return prefix;
        }
        auto arguments =
            validateIndirectArguments(resources_,
                                      desc.argumentBuffer,
                                      desc.argumentOffset,
                                      desc.maxDrawCount,
                                      desc.stride,
                                      desc.countBuffer,
                                      desc.countOffset,
                                      prefixedRecordSize(sizeof(DrawIndirectArguments), desc.drawIndexPrefix),
                                      capabilities_ != nullptr && capabilities_->drawIndirectCount,
                                      capabilities_ != nullptr && capabilities_->multiDrawIndirect,
                                      maxDrawIndirectCount_);
        if (!arguments)
        {
            return std::unexpected(std::move(arguments.error()));
        }
        if (!arguments->recordWork)
        {
            return {};
        }
        if (arguments->counted && table_->vkCmdDrawIndirectCount == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "vkCmdDrawIndirectCount is unavailable on this device.",
            });
        }
        if (drawScissoredOut(commandBuffer))
        {
            return {};
        }
        if (arguments->counted)
        {
            table_->vkCmdDrawIndirectCount(commandBuffer.native(),
                                           arguments->buffer,
                                           nativeArgumentOffset(desc.argumentOffset, desc.drawIndexPrefix),
                                           arguments->countBuffer,
                                           desc.countOffset,
                                           desc.maxDrawCount,
                                           arguments->stride);
        }
        else
        {
            table_->vkCmdDrawIndirect(commandBuffer.native(),
                                      arguments->buffer,
                                      nativeArgumentOffset(desc.argumentOffset, desc.drawIndexPrefix),
                                      desc.maxDrawCount,
                                      arguments->stride);
        }
        if (auto* retained = retainSink(commandBuffer); retained != nullptr)
        {
            retained->push_back(std::move(arguments->keepAlive));
            if (arguments->countKeepAlive)
            {
                retained->push_back(std::move(arguments->countKeepAlive));
            }
        }
        return {};
    }

    Status VulkanGraphicsEncoder::drawIndexedIndirect(VulkanCommandBuffer& commandBuffer, const IndirectDrawDesc& desc)
    {
        if (auto pipelineState = requireBoundPipeline(commandBuffer); !pipelineState)
        {
            return pipelineState;
        }
        const auto& state = commandBuffer.graphicsState();
        auto* const pipeline = static_cast<const VulkanNativePipeline*>(state.pipelineOwner.get());
        if (pipeline->mesh || (pipeline->rasterizationEnabled && !hasCompleteViewportState(state)) ||
            (state.blendConstantsRequired && !state.blendConstantsSet) ||
            (state.depthBiasRequired && !state.depthBiasSet) || (state.depthBoundsRequired && !state.depthBoundsSet) ||
            (state.stencilReferenceRequired && !state.stencilReferenceSet) || !state.indexBufferBound ||
            !allBound(state.boundBindingSets) || !allBound(state.boundVertexBuffers) ||
            !allBound(state.boundStreamOutputBuffers))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Indexed indirect draw requires complete graphics state and a bound index buffer.",
            });
        }
        if (auto prefix = validateDrawIndexPrefix(*pipeline, desc.drawIndexPrefix, desc.maxDrawCount); !prefix)
        {
            return prefix;
        }
        auto arguments =
            validateIndirectArguments(resources_,
                                      desc.argumentBuffer,
                                      desc.argumentOffset,
                                      desc.maxDrawCount,
                                      desc.stride,
                                      desc.countBuffer,
                                      desc.countOffset,
                                      prefixedRecordSize(sizeof(DrawIndexedIndirectArguments), desc.drawIndexPrefix),
                                      capabilities_ != nullptr && capabilities_->drawIndirectCount,
                                      capabilities_ != nullptr && capabilities_->multiDrawIndirect,
                                      maxDrawIndirectCount_);
        if (!arguments)
        {
            return std::unexpected(std::move(arguments.error()));
        }
        if (!arguments->recordWork)
        {
            return {};
        }
        if (arguments->counted && table_->vkCmdDrawIndexedIndirectCount == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "vkCmdDrawIndexedIndirectCount is unavailable on this device.",
            });
        }
        if (drawScissoredOut(commandBuffer))
        {
            return {};
        }
        if (arguments->counted)
        {
            table_->vkCmdDrawIndexedIndirectCount(commandBuffer.native(),
                                                  arguments->buffer,
                                                  nativeArgumentOffset(desc.argumentOffset, desc.drawIndexPrefix),
                                                  arguments->countBuffer,
                                                  desc.countOffset,
                                                  desc.maxDrawCount,
                                                  arguments->stride);
        }
        else
        {
            table_->vkCmdDrawIndexedIndirect(commandBuffer.native(),
                                             arguments->buffer,
                                             nativeArgumentOffset(desc.argumentOffset, desc.drawIndexPrefix),
                                             desc.maxDrawCount,
                                             arguments->stride);
        }
        if (auto* retained = retainSink(commandBuffer); retained != nullptr)
        {
            retained->push_back(std::move(arguments->keepAlive));
            if (arguments->countKeepAlive)
            {
                retained->push_back(std::move(arguments->countKeepAlive));
            }
        }
        return {};
    }

    Status VulkanGraphicsEncoder::bindComputePipeline(VulkanCommandBuffer& commandBuffer, PipelineHandle pipeline)
    {
        if (table_ == nullptr || pipelines_ == nullptr || commandBuffer.native() == VK_NULL_HANDLE ||
            commandBuffer.queueType() == QueueType::copy || commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A compute pipeline requires a graphics or compute command buffer outside a rendering scope.",
            });
        }
        auto native = pipelines_->nativePipeline(pipeline);
        if (!native || *native == nullptr || !(*native)->compute || (*native)->pipeline == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                native && *native != nullptr && !(*native)->compute ? "The pipeline is not a compute pipeline."
                                                                    : "The pipeline handle is invalid or stale.",
            });
        }
        const auto& bound = *native;
        auto* retained = retainSink(commandBuffer);
        auto& state = commandBuffer.computeState();
        try
        {
            if (retained != nullptr)
            {
                detail::reserveAdditional(*retained, 1U);
            }
            state.boundBindingSets.reserve(bound->setRequired.size());
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Compute pipeline bind state could not be allocated.",
            });
        }

        table_->vkCmdBindPipeline(commandBuffer.native(), VK_PIPELINE_BIND_POINT_COMPUTE, bound->pipeline);
        if (retained != nullptr)
        {
            retained->push_back(std::const_pointer_cast<void>(std::static_pointer_cast<const void>(bound)));
        }

        commandBuffer.rayTracingState() = {};
        state.pipelineOwner = std::const_pointer_cast<VulkanNativePipeline>(bound);
        state.boundBindingSets.clear();
        for (const auto required : bound->setRequired)
        {
            state.boundBindingSets.push_back(required == 0);
        }
        return {};
    }

    Status VulkanGraphicsEncoder::bindComputeBindingSets(VulkanCommandBuffer& commandBuffer,
                                                         std::uint32_t firstSet,
                                                         std::span<const BindingSetHandle> bindingSets,
                                                         std::span<const std::uint32_t> dynamicOffsets)
    {
        auto& state = commandBuffer.computeState();
        if (state.pipelineOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A compute pipeline must be bound before binding sets.",
            });
        }
        auto* const pipeline = static_cast<const VulkanNativePipeline*>(state.pipelineOwner.get());
        return bindDescriptorSetBatch(bindings_,
                                      table_,
                                      commandBuffer,
                                      retainSink(commandBuffer),
                                      VK_PIPELINE_BIND_POINT_COMPUTE,
                                      *pipeline,
                                      firstSet,
                                      bindingSets,
                                      dynamicOffsets,
                                      bufferLimits_,
                                      state.boundBindingSets,
                                      "The compute binding-set batch is empty or exceeds the pipeline layout.",
                                      "A compute binding set does not match the pipeline layout.",
                                      "The dynamic-offset count does not match the bound compute descriptor sets.");
    }

    Status VulkanGraphicsEncoder::setComputePushConstants(VulkanCommandBuffer& commandBuffer,
                                                          std::uint32_t offset,
                                                          std::span<const std::byte> data)
    {
        auto& state = commandBuffer.computeState();
        if (state.pipelineOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A compute pipeline must be bound before setting push constants.",
            });
        }
        auto* const pipeline = static_cast<const VulkanNativePipeline*>(state.pipelineOwner.get());
        auto subranges = validation_detail::planComputePushConstants(
            pipeline->pushConstantRanges, pipeline->nativePushConstantRanges, offset, data.size());
        if (!subranges)
        {
            return std::unexpected(std::move(subranges.error()));
        }
        for (const auto& subrange : *subranges)
        {
            table_->vkCmdPushConstants(commandBuffer.native(),
                                       pipeline->layout,
                                       nativePushConstantStages(capabilities_, subrange.stages),
                                       subrange.offset,
                                       subrange.size,
                                       data.data() + (subrange.offset - offset));
        }
        return {};
    }

    Status VulkanGraphicsEncoder::dispatch(VulkanCommandBuffer& commandBuffer,
                                           std::uint32_t groupCountX,
                                           std::uint32_t groupCountY,
                                           std::uint32_t groupCountZ)
    {
        const auto& state = commandBuffer.computeState();
        if (state.pipelineOwner == nullptr || commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Dispatch requires a bound compute pipeline outside a rendering scope.",
            });
        }
        if (!allBound(state.boundBindingSets))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Every required compute binding set must be bound before dispatch.",
            });
        }
        const auto& limits = capabilities_->limits.maxComputeWorkgroupCount;
        if (groupCountX == 0 || groupCountY == 0 || groupCountZ == 0 || groupCountX > limits[0] ||
            groupCountY > limits[1] || groupCountZ > limits[2])
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Dispatch group counts must be between one and the device per-dimension limit.",
            });
        }
        table_->vkCmdDispatch(commandBuffer.native(), groupCountX, groupCountY, groupCountZ);
        return {};
    }

    Status VulkanGraphicsEncoder::dispatchIndirect(VulkanCommandBuffer& commandBuffer, const IndirectDispatchDesc& desc)
    {
        const auto& state = commandBuffer.computeState();
        if (state.pipelineOwner == nullptr || commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Indirect dispatch requires a bound compute pipeline outside a rendering scope.",
            });
        }
        if (!allBound(state.boundBindingSets))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Every required compute binding set must be bound before indirect dispatch.",
            });
        }
        auto arguments = validateIndirectArguments(resources_,
                                                   desc.argumentBuffer,
                                                   desc.argumentOffset,
                                                   desc.maxDispatchCount,
                                                   desc.stride,
                                                   desc.countBuffer,
                                                   desc.countOffset,
                                                   sizeof(DispatchIndirectArguments));
        if (!arguments)
        {
            return std::unexpected(std::move(arguments.error()));
        }
        if (!arguments->recordWork)
        {
            return {};
        }
        table_->vkCmdDispatchIndirect(commandBuffer.native(), arguments->buffer, desc.argumentOffset);
        if (auto* retained = retainSink(commandBuffer); retained != nullptr)
        {
            retained->push_back(std::move(arguments->keepAlive));
        }
        return {};
    }

    Status VulkanGraphicsEncoder::requireMeshDispatchState(const VulkanCommandBuffer& commandBuffer) const
    {
        const auto& state = commandBuffer.graphicsState();
        auto* const pipeline = static_cast<const VulkanNativePipeline*>(state.pipelineOwner.get());
        if (state.renderingOwner == nullptr || pipeline == nullptr || !pipeline->mesh ||
            !hasCompleteViewportState(state) || (state.blendConstantsRequired && !state.blendConstantsSet) ||
            (state.depthBiasRequired && !state.depthBiasSet) || (state.depthBoundsRequired && !state.depthBoundsSet) ||
            (state.stencilReferenceRequired && !state.stencilReferenceSet) || !allBound(state.boundBindingSets))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Mesh dispatch requires complete rendering, mesh-pipeline, binding, viewport, and scissor state.",
            });
        }
        return {};
    }

    void VulkanGraphicsEncoder::beginNativeRendering(VulkanCommandBuffer& commandBuffer,
                                                     VulkanRenderingScope& scope) const noexcept
    {
        VkRenderingFragmentShadingRateAttachmentInfoKHR rateInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR,
            .pNext = {},
            .imageView = scope.shadingRateView,
            .imageLayout = VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR,
            .shadingRateAttachmentTexelSize =
                {
                    capabilities_->shadingRateImageTileSize,
                    capabilities_->shadingRateImageTileSize,
                },
        };
        const auto info = VkRenderingInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = scope.shadingRateView == VK_NULL_HANDLE ? nullptr : &rateInfo,
            .flags = {},
            .renderArea =
                {
                    .offset = {0, 0},
                    .extent = {scope.width, scope.height},
                },
            .layerCount = scope.layerCount,
            .viewMask = scope.viewMask,
            .colorAttachmentCount = static_cast<std::uint32_t>(scope.colorAttachments.size()),
            .pColorAttachments = scope.colorAttachments.empty() ? nullptr : scope.colorAttachments.data(),
            .pDepthAttachment = scope.depthAttachment.has_value() ? &*scope.depthAttachment : nullptr,
            .pStencilAttachment = scope.stencilAttachment.has_value() ? &*scope.stencilAttachment : nullptr,
        };
        table_->vkCmdBeginRendering(commandBuffer.native(), &info);
    }

    namespace
    {

        [[nodiscard]] Result<VulkanResourceAllocator::ResolvedTexture> resolveShadingRateSource(
            const VulkanResourceAllocator& resources, TextureHandle image)
        {
            auto resolved = resources.resolveTexture(image);
            if (!resolved)
            {
                return std::unexpected(std::move(resolved.error()));
            }
            if (!hasAny(resolved->desc.usage, TextureUsage::shadingRateSource) ||
                resolved->desc.format != Format::r8Uint || resolved->desc.dimension != TextureDimension::texture2D ||
                resolved->desc.depthOrArrayLayers != 1 || resolved->desc.mipLevels != 1 ||
                resolved->desc.sampleCount != 1)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A shading-rate image must be a live single-layer, single-mip, single-sample r8Uint 2D texture "
                    "with shading-rate-source usage.",
                });
            }
            return resolved;
        }

        [[nodiscard]] constexpr std::uint32_t shadingRateTexelsCovering(std::uint32_t extent,
                                                                        std::uint32_t tileSize) noexcept
        {
            return extent / tileSize + (extent % tileSize != 0 ? 1U : 0U);
        }

        [[nodiscard]] Status validateShadingRateCoverage(const TextureDesc& image,
                                                         const VulkanRenderingScope& scope,
                                                         std::uint32_t tileSize)
        {
            if (tileSize == 0 || image.width < shadingRateTexelsCovering(scope.width, tileSize) ||
                image.height < shadingRateTexelsCovering(scope.height, tileSize))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A shading-rate image must have one texel for every shading-rate tile of the rendering area.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateRenderingSplitQueryState(const detail::QueryCommandState& queryState)
        {
            const bool activePipelineStats = queryState.activePipelineStatistics.has_value();
            const bool pipelineStatsBegunInPass =
                activePipelineStats && queryState.activePipelineStatistics->begunInRendering;
            return validation_detail::validateShadingRateImageSplitQueryState(
                queryState.activeOcclusion.has_value(), activePipelineStats, pipelineStatsBegunInPass);
        }

        struct ShadingRateViewKeepAlive final
        {
            ShadingRateViewKeepAlive() = default;
            ShadingRateViewKeepAlive(const ShadingRateViewKeepAlive&) = delete;
            ShadingRateViewKeepAlive& operator=(const ShadingRateViewKeepAlive&) = delete;
            ShadingRateViewKeepAlive(ShadingRateViewKeepAlive&&) = delete;
            ShadingRateViewKeepAlive& operator=(ShadingRateViewKeepAlive&&) = delete;
            VkDevice device = VK_NULL_HANDLE;
            const VolkDeviceTable* table = nullptr;
            VkImageView view = VK_NULL_HANDLE;
            std::shared_ptr<void> texture;

            ~ShadingRateViewKeepAlive()
            {
                if (table != nullptr && table->vkDestroyImageView != nullptr && device != VK_NULL_HANDLE &&
                    view != VK_NULL_HANDLE)
                {
                    table->vkDestroyImageView(device, view, nullptr);
                }
#ifdef RACHIS_TESTING
                if (view != VK_NULL_HANDLE)
                {
                    test::liveShadingRateViews().fetch_sub(1, std::memory_order_relaxed);
                }
#endif
            }
        };

        struct ShadingRateAttachmentView final
        {
            VkImageView view = VK_NULL_HANDLE;
            std::shared_ptr<void> keepAlive = {};
        };

        [[nodiscard]] Result<ShadingRateAttachmentView> createShadingRateAttachmentView(
            VkDevice device, const VolkDeviceTable* table, VkImage image, std::shared_ptr<void> textureKeepAlive)
        {
            const VkImageViewCreateInfo viewInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = {},
                .flags = {},
                .image = image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = VK_FORMAT_R8_UINT,
                .components = {},
                .subresourceRange =
                    {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = {},
                        .levelCount = 1,
                        .baseArrayLayer = {},
                        .layerCount = 1,
                    },
            };
            VkImageView view = VK_NULL_HANDLE;
            const auto created = table->vkCreateImageView(device, &viewInfo, nullptr, &view);
            if (created != VK_SUCCESS)
            {
                return std::unexpected(makeVulkanError(ErrorCode::outOfMemory, created, "vkCreateImageView"));
            }
            auto keepAlive = std::make_shared<ShadingRateViewKeepAlive>();
            keepAlive->device = device;
            keepAlive->table = table;
            keepAlive->view = view;
#ifdef RACHIS_TESTING
            test::liveShadingRateViews().fetch_add(1, std::memory_order_relaxed);
#endif
            keepAlive->texture = std::move(textureKeepAlive);
            return ShadingRateAttachmentView{
                .view = view,
                .keepAlive = std::move(keepAlive),
            };
        }

        [[nodiscard]] std::optional<VkMemoryBarrier2> makeRenderingSplitBarrier(const VulkanRenderingScope& scope,
                                                                                bool readsShadingRateImage) noexcept
        {
            VkPipelineStageFlags2 srcStages = VK_PIPELINE_STAGE_2_NONE;
            VkAccessFlags2 srcAccess = VK_ACCESS_2_NONE;
            VkPipelineStageFlags2 dstStages = VK_PIPELINE_STAGE_2_NONE;
            VkAccessFlags2 dstAccess = VK_ACCESS_2_NONE;
            if (!scope.colorAttachments.empty())
            {
                srcStages |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                srcAccess |= VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                dstStages |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                dstAccess |= VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            }
            if (scope.depthAttachment.has_value() || scope.stencilAttachment.has_value())
            {
                srcStages |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
                srcAccess |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                dstStages |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
                dstAccess |=
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            }
            if (readsShadingRateImage)
            {
                dstStages |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR;
                dstAccess |= VK_ACCESS_2_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR;
            }
            if (srcStages == VK_PIPELINE_STAGE_2_NONE && dstStages == VK_PIPELINE_STAGE_2_NONE)
            {
                return std::nullopt;
            }
            if (srcStages == VK_PIPELINE_STAGE_2_NONE)
            {
                srcStages = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            }
            return VkMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                .pNext = {},
                .srcStageMask = srcStages,
                .srcAccessMask = srcAccess,
                .dstStageMask = dstStages,
                .dstAccessMask = dstAccess,
            };
        }

        void recordRenderingSplitBarrier(const VolkDeviceTable& table,
                                         VkCommandBuffer command,
                                         const VulkanRenderingScope& scope,
                                         bool readsShadingRateImage)
        {
            if (table.vkCmdPipelineBarrier2 == nullptr)
            {
                return;
            }
            const auto barrier = makeRenderingSplitBarrier(scope, readsShadingRateImage);
            if (!barrier.has_value())
            {
                return;
            }
            const VkDependencyInfo dependency{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pNext = {},
                .dependencyFlags = {},
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &*barrier,
                .bufferMemoryBarrierCount = {},
                .pBufferMemoryBarriers = {},
                .imageMemoryBarrierCount = {},
                .pImageMemoryBarriers = {},
            };
            table.vkCmdPipelineBarrier2(command, &dependency);
        }

        void preserveAttachmentContentsOnResume(VulkanRenderingScope& scope) noexcept
        {
            for (auto& attachment : scope.colorAttachments)
            {
                attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            }
            if (scope.depthAttachment.has_value())
            {
                scope.depthAttachment->loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            }
            if (scope.stencilAttachment.has_value())
            {
                scope.stencilAttachment->loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            }
        }

    } // namespace

    Status VulkanGraphicsEncoder::applyShadingRateImage(VulkanCommandBuffer& commandBuffer,
                                                        VulkanRenderingScope& scope,
                                                        TextureHandle image)
    {
        std::optional<VulkanResourceAllocator::ResolvedTexture> source;
        if (image)
        {
            auto resolved = resolveShadingRateSource(*resources_, image);
            if (!resolved)
            {
                return std::unexpected(std::move(resolved.error()));
            }
            if (auto covered =
                    validateShadingRateCoverage(resolved->desc, scope, capabilities_->shadingRateImageTileSize);
                !covered)
            {
                return covered;
            }
            source = std::move(*resolved);
        }
        if (auto valid = validateRenderingSplitQueryState(commandBuffer.queryState()); !valid)
        {
            return valid;
        }

        ShadingRateAttachmentView next;
        if (source.has_value())
        {
            auto& retained = commandBuffer.retained();
            try
            {
                detail::reserveAdditional(retained, 1U);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "The shading-rate attachment view could not be retained.",
                });
            }
            auto textureKeepAlive =
                retainSink(commandBuffer) != nullptr ? std::move(source->keepAlive) : std::shared_ptr<void>{};
            auto created = createShadingRateAttachmentView(device_, table_, source->image, std::move(textureKeepAlive));
            if (!created)
            {
                return std::unexpected(std::move(created.error()));
            }
            next = std::move(*created);
            retained.push_back(next.keepAlive);
        }

        table_->vkCmdEndRendering(commandBuffer.native());
        recordRenderingSplitBarrier(*table_, commandBuffer.native(), scope, next.view != VK_NULL_HANDLE);
        preserveAttachmentContentsOnResume(scope);
        scope.shadingRateView = next.view;
        scope.shadingRateKeepAlive = std::move(next.keepAlive);
        beginNativeRendering(commandBuffer, scope);
        return {};
    }

    Status VulkanGraphicsEncoder::setShadingRate(VulkanCommandBuffer& commandBuffer,
                                                 ShadingRate rate,
                                                 const ShadingRateCombiners& combiners)
    {
        if (commandBuffer.graphicsState().renderingOwner == nullptr || !isValidShadingRate(rate) ||
            !isValidShadingRateCombiner(combiners.primitive) || !isValidShadingRateCombiner(combiners.image))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A shading rate requires an active rendering scope and a valid rate.",
            });
        }
        if (capabilities_ == nullptr || !capabilities_->variableRateShading ||
            (isAdditionalShadingRate(rate) && !capabilities_->additionalShadingRates))
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The requested shading rate is unavailable on this adapter.",
            });
        }
        if ((combiners.primitive == ShadingRateCombiner::sum || combiners.image == ShadingRateCombiner::sum))
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Vulkan fragment shading rate has no portable sum combiner.",
            });
        }
        const bool nonDefaultCombiners = combiners.primitive != ShadingRateCombiner::passthrough ||
                                         combiners.image != ShadingRateCombiner::passthrough;
        if (nonDefaultCombiners && !capabilities_->variableRateShadingTier2)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Non-default shading-rate combiners require primitive and attachment fragment shading rates.",
            });
        }
        auto* const scope = static_cast<VulkanRenderingScope*>(commandBuffer.graphicsState().renderingOwner.get());
        if (!supportsSampleCount(rate, scope->sampleCount))
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The requested shading rate is unavailable for the active rendering sample count.",
            });
        }
        const auto fragmentSize = vulkanFragmentSize(rate);
        const auto primitiveCombiner = vulkanShadingRateCombiner(combiners.primitive);
        const auto imageCombiner = vulkanShadingRateCombiner(combiners.image);
        if (!fragmentSize.has_value() || !primitiveCombiner.has_value() || !imageCombiner.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A shading rate requires an active rendering scope and a valid rate.",
            });
        }
        const std::array nativeCombiners{*primitiveCombiner, *imageCombiner};
        table_->vkCmdSetFragmentShadingRateKHR(commandBuffer.native(), &*fragmentSize, nativeCombiners.data());
        return {};
    }

    Status VulkanGraphicsEncoder::setShadingRateImage(VulkanCommandBuffer& commandBuffer, TextureHandle image)
    {
        if (commandBuffer.graphicsState().renderingOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A shading-rate image requires an active rendering scope.",
            });
        }
        if (capabilities_ == nullptr || !capabilities_->variableRateShadingTier2)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Shading-rate images require primitive and attachment fragment shading rates.",
            });
        }
        auto* const scope = static_cast<VulkanRenderingScope*>(commandBuffer.graphicsState().renderingOwner.get());
        return applyShadingRateImage(commandBuffer, *scope, image);
    }

    Status VulkanGraphicsEncoder::dispatchMesh(VulkanCommandBuffer& commandBuffer,
                                               std::uint32_t groupCountX,
                                               std::uint32_t groupCountY,
                                               std::uint32_t groupCountZ)
    {
        if (auto ready = requireMeshDispatchState(commandBuffer); !ready)
        {
            return ready;
        }
        if (capabilities_ == nullptr || !capabilities_->meshShaders)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support mesh shaders.",
            });
        }
        const auto& limits = capabilities_->meshShader;
        const auto totalWorkgroups = static_cast<std::uint64_t>(groupCountX) * groupCountY * groupCountZ;
        if (groupCountX == 0 || groupCountY == 0 || groupCountZ == 0 || groupCountX > limits.maxWorkgroupCount[0] ||
            groupCountY > limits.maxWorkgroupCount[1] || groupCountZ > limits.maxWorkgroupCount[2] ||
            totalWorkgroups > limits.maxTotalWorkgroups)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Mesh dispatch counts must fit every dimension and the total device workgroup limit.",
            });
        }
        if (drawScissoredOut(commandBuffer))
        {
            return {};
        }
        table_->vkCmdDrawMeshTasksEXT(commandBuffer.native(), groupCountX, groupCountY, groupCountZ);
        return {};
    }

    Status VulkanGraphicsEncoder::dispatchMeshIndirect(VulkanCommandBuffer& commandBuffer,
                                                       const IndirectMeshDispatchDesc& desc)
    {
        if (auto ready = requireMeshDispatchState(commandBuffer); !ready)
        {
            return ready;
        }
        if (capabilities_ == nullptr || !capabilities_->meshShaders)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support mesh shaders.",
            });
        }
        const auto* const meshPipeline =
            static_cast<const VulkanNativePipeline*>(commandBuffer.graphicsState().pipelineOwner.get());
        if (auto prefix = validateDrawIndexPrefix(*meshPipeline, desc.drawIndexPrefix, desc.maxDispatchCount); !prefix)
        {
            return prefix;
        }
        auto arguments =
            validateIndirectArguments(resources_,
                                      desc.argumentBuffer,
                                      desc.argumentOffset,
                                      desc.maxDispatchCount,
                                      desc.stride,
                                      desc.countBuffer,
                                      desc.countOffset,
                                      prefixedRecordSize(sizeof(DispatchIndirectArguments), desc.drawIndexPrefix),
                                      capabilities_->drawIndirectCount,
                                      capabilities_->multiDrawIndirect,
                                      maxDrawIndirectCount_);
        if (!arguments)
        {
            return std::unexpected(std::move(arguments.error()));
        }
        if (!arguments->recordWork)
        {
            return {};
        }
        const bool entryPointLoaded = arguments->counted ? table_->vkCmdDrawMeshTasksIndirectCountEXT != nullptr
                                                         : table_->vkCmdDrawMeshTasksIndirectEXT != nullptr;
        if (!entryPointLoaded)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                arguments->counted ? "vkCmdDrawMeshTasksIndirectCountEXT is unavailable on this device."
                                   : "vkCmdDrawMeshTasksIndirectEXT is unavailable on this device.",
            });
        }
        if (drawScissoredOut(commandBuffer))
        {
            return {};
        }
        if (arguments->counted)
        {
            table_->vkCmdDrawMeshTasksIndirectCountEXT(commandBuffer.native(),
                                                       arguments->buffer,
                                                       nativeArgumentOffset(desc.argumentOffset, desc.drawIndexPrefix),
                                                       arguments->countBuffer,
                                                       desc.countOffset,
                                                       desc.maxDispatchCount,
                                                       arguments->stride);
        }
        else
        {
            table_->vkCmdDrawMeshTasksIndirectEXT(commandBuffer.native(),
                                                  arguments->buffer,
                                                  nativeArgumentOffset(desc.argumentOffset, desc.drawIndexPrefix),
                                                  desc.maxDispatchCount,
                                                  arguments->stride);
        }
        if (auto* retained = retainSink(commandBuffer); retained != nullptr)
        {
            retained->push_back(std::move(arguments->keepAlive));
            if (arguments->countKeepAlive)
            {
                retained->push_back(std::move(arguments->countKeepAlive));
            }
        }
        return {};
    }

    Status VulkanGraphicsEncoder::bindRayTracingPipeline(VulkanCommandBuffer& commandBuffer, PipelineHandle pipeline)
    {
        if (table_ == nullptr || pipelines_ == nullptr || commandBuffer.native() == VK_NULL_HANDLE ||
            commandBuffer.queueType() == QueueType::copy || commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A ray-tracing pipeline requires a graphics or compute command buffer outside a rendering scope.",
            });
        }
        auto native = pipelines_->nativePipeline(pipeline);
        if (!native || *native == nullptr || !(*native)->rayTracing || (*native)->pipeline == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                native && *native != nullptr && !(*native)->rayTracing ? "The pipeline is not a ray-tracing pipeline."
                                                                       : "The pipeline handle is invalid or stale.",
            });
        }
        const auto& bound = *native;
        auto* retained = retainSink(commandBuffer);
        auto& state = commandBuffer.rayTracingState();
        try
        {
            if (retained != nullptr)
            {
                detail::reserveAdditional(*retained, 1U);
            }
            state.boundBindingSets.reserve(bound->setRequired.size());
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Ray-tracing pipeline bind state could not be allocated.",
            });
        }

        table_->vkCmdBindPipeline(commandBuffer.native(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, bound->pipeline);
        if (retained != nullptr)
        {
            retained->push_back(std::const_pointer_cast<void>(std::static_pointer_cast<const void>(bound)));
        }

        commandBuffer.computeState() = {};
        state.pipelineOwner = std::const_pointer_cast<VulkanNativePipeline>(bound);
        state.boundBindingSets.clear();
        for (const auto required : bound->setRequired)
        {
            state.boundBindingSets.push_back(required == 0);
        }
        return {};
    }

    Status VulkanGraphicsEncoder::bindRayTracingBindingSets(VulkanCommandBuffer& commandBuffer,
                                                            std::uint32_t firstSet,
                                                            std::span<const BindingSetHandle> bindingSets,
                                                            std::span<const std::uint32_t> dynamicOffsets)
    {
        auto& state = commandBuffer.rayTracingState();
        if (state.pipelineOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A ray-tracing pipeline must be bound before binding sets.",
            });
        }
        auto* const pipeline = static_cast<const VulkanNativePipeline*>(state.pipelineOwner.get());
        return bindDescriptorSetBatch(bindings_,
                                      table_,
                                      commandBuffer,
                                      retainSink(commandBuffer),
                                      VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                      *pipeline,
                                      firstSet,
                                      bindingSets,
                                      dynamicOffsets,
                                      bufferLimits_,
                                      state.boundBindingSets,
                                      "The ray-tracing binding-set batch is empty or exceeds the pipeline layout.",
                                      "A ray-tracing binding set does not match the pipeline layout.",
                                      "The dynamic-offset count does not match the bound ray-tracing descriptor sets.");
    }

    Status VulkanGraphicsEncoder::setRayTracingPushConstants(VulkanCommandBuffer& commandBuffer,
                                                             std::uint32_t offset,
                                                             std::span<const std::byte> data)
    {
        auto& state = commandBuffer.rayTracingState();
        if (state.pipelineOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A ray-tracing pipeline must be bound before setting push constants.",
            });
        }
        auto* const pipeline = static_cast<const VulkanNativePipeline*>(state.pipelineOwner.get());
        auto subranges = validation_detail::planRayTracingPushConstants(
            pipeline->pushConstantRanges, pipeline->nativePushConstantRanges, offset, data.size());
        if (!subranges)
        {
            return std::unexpected(std::move(subranges.error()));
        }
        for (const auto& subrange : *subranges)
        {
            table_->vkCmdPushConstants(commandBuffer.native(),
                                       pipeline->layout,
                                       nativePushConstantStages(capabilities_, subrange.stages),
                                       subrange.offset,
                                       subrange.size,
                                       data.data() + (subrange.offset - offset));
        }
        return {};
    }

    namespace
    {

        [[nodiscard]] Result<VkStridedDeviceAddressRegionKHR> nativeShaderTableRegion(
            VulkanResourceAllocator* resources,
            const RayTracingCapabilities& limits,
            const RayTracingShaderTableRegion& region,
            bool required,
            bool rayGeneration,
            std::vector<std::shared_ptr<void>>& retained)
        {
            if (!region.buffer.isValid())
            {
                if (!required && region.offset == 0 && region.size == 0 && region.stride == 0)
                {
                    return VkStridedDeviceAddressRegionKHR{};
                }
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A shader-table region has an invalid buffer or non-zero metadata for an omitted table.",
                });
            }
            if (resources == nullptr)
            {
                return std::unexpected(
                    Error{ErrorCode::invalidState, 0, "Shader-table resolution requires live resources."});
            }
            auto source = resources->resolveBindingBuffer(region.buffer);
            if (!source)
            {
                return std::unexpected(std::move(source.error()));
            }
            const bool rangeValid =
                region.offset <= source->desc.size && region.size <= source->desc.size - region.offset;
            const bool strideValid = region.stride >= limits.shaderGroupHandleSize &&
                                     region.stride <= limits.maxShaderGroupStride &&
                                     region.stride % limits.shaderGroupHandleAlignment == 0;
            const bool baseAligned = region.offset % limits.shaderGroupBaseAlignment == 0 &&
                                     (source->deviceAddress + region.offset) % limits.shaderGroupBaseAlignment == 0;
            if (!hasAny(source->desc.usage, BufferUsage::shaderBindingTable) || source->deviceAddress == 0 ||
                region.size == 0 || !rangeValid || !baseAligned || !strideValid ||
                (rayGeneration ? region.size != region.stride : region.size % region.stride != 0))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A shader-table region violates its usage, range, base-alignment, record-stride, or table-size "
                    "contract.",
                });
            }
            retained.push_back(source->keepAlive);
            return VkStridedDeviceAddressRegionKHR{
                .deviceAddress = source->deviceAddress + region.offset,
                .stride = region.stride,
                .size = region.size,
            };
        }

        [[nodiscard]] Status validateRayTracingDispatchState(VulkanCommandBuffer& commandBuffer)
        {
            const auto& state = commandBuffer.rayTracingState();
            if (state.pipelineOwner == nullptr || commandBuffer.graphicsState().renderingOwner != nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Ray dispatch requires a bound ray-tracing pipeline outside a rendering scope.",
                });
            }
            if (!allBound(state.boundBindingSets))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Every required ray-tracing binding set must be bound before dispatch.",
                });
            }
            return {};
        }

    } // namespace

    Status VulkanGraphicsEncoder::dispatchRays(VulkanCommandBuffer& commandBuffer, const RayTracingDispatchDesc& desc)
    {
        if (auto state = validateRayTracingDispatchState(commandBuffer); !state)
        {
            return state;
        }
        const auto invocationCount = static_cast<std::uint64_t>(desc.width) * desc.height * desc.depth;
        if (desc.width == 0 || desc.height == 0 || desc.depth == 0 ||
            invocationCount > capabilities_->rayTracingLimits.maxDispatchInvocations)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Ray-dispatch dimensions must be non-zero and fit the device's aggregate invocation limit.",
            });
        }
        std::vector<std::shared_ptr<void>> retained;
        auto rayGeneration = nativeShaderTableRegion(
            resources_, capabilities_->rayTracingLimits, desc.rayGeneration, true, true, retained);
        auto miss =
            nativeShaderTableRegion(resources_, capabilities_->rayTracingLimits, desc.miss, false, false, retained);
        auto hitGroup =
            nativeShaderTableRegion(resources_, capabilities_->rayTracingLimits, desc.hitGroup, false, false, retained);
        auto callable =
            nativeShaderTableRegion(resources_, capabilities_->rayTracingLimits, desc.callable, false, false, retained);
        if (!rayGeneration)
        {
            return std::unexpected(std::move(rayGeneration.error()));
        }
        if (!miss)
        {
            return std::unexpected(std::move(miss.error()));
        }
        if (!hitGroup)
        {
            return std::unexpected(std::move(hitGroup.error()));
        }
        if (!callable)
        {
            return std::unexpected(std::move(callable.error()));
        }
        if (auto* sink = retainSink(commandBuffer); sink != nullptr)
        {
            sink->insert(sink->end(), retained.begin(), retained.end());
        }
        table_->vkCmdTraceRaysKHR(commandBuffer.native(),
                                  &*rayGeneration,
                                  &*miss,
                                  &*hitGroup,
                                  &*callable,
                                  desc.width,
                                  desc.height,
                                  desc.depth);
        return {};
    }

    Status VulkanGraphicsEncoder::dispatchRaysIndirect(VulkanCommandBuffer& commandBuffer,
                                                       const RayTracingIndirectDispatchDesc& desc)
    {
        if (!capabilities_->indirectRayDispatch || table_->vkCmdTraceRaysIndirectKHR == nullptr)
        {
            return std::unexpected(
                Error{ErrorCode::unsupported, 0, "Indirect ray dispatch is unavailable on this device."});
        }
        if (auto state = validateRayTracingDispatchState(commandBuffer); !state)
        {
            return state;
        }
        std::vector<std::shared_ptr<void>> retained;
        auto rayGeneration = nativeShaderTableRegion(
            resources_, capabilities_->rayTracingLimits, desc.rayGeneration, true, true, retained);
        auto miss =
            nativeShaderTableRegion(resources_, capabilities_->rayTracingLimits, desc.miss, false, false, retained);
        auto hitGroup =
            nativeShaderTableRegion(resources_, capabilities_->rayTracingLimits, desc.hitGroup, false, false, retained);
        auto callable =
            nativeShaderTableRegion(resources_, capabilities_->rayTracingLimits, desc.callable, false, false, retained);
        if (!rayGeneration)
        {
            return std::unexpected(std::move(rayGeneration.error()));
        }
        if (!miss)
        {
            return std::unexpected(std::move(miss.error()));
        }
        if (!hitGroup)
        {
            return std::unexpected(std::move(hitGroup.error()));
        }
        if (!callable)
        {
            return std::unexpected(std::move(callable.error()));
        }
        auto arguments = resources_->resolveBindingBuffer(desc.argumentBuffer);
        if (!arguments || !hasAny(arguments->desc.usage, BufferUsage::indirect) || arguments->deviceAddress == 0 ||
            desc.argumentOffset % 4 != 0 || desc.argumentOffset > arguments->desc.size ||
            sizeof(DispatchIndirectArguments) > arguments->desc.size - desc.argumentOffset)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Indirect ray dispatch requires an in-bounds, 4-byte aligned indirect argument record.",
            });
        }
        retained.push_back(arguments->keepAlive);
        if (auto* sink = retainSink(commandBuffer); sink != nullptr)
        {
            sink->insert(sink->end(), retained.begin(), retained.end());
        }
        table_->vkCmdTraceRaysIndirectKHR(commandBuffer.native(),
                                          &*rayGeneration,
                                          &*miss,
                                          &*hitGroup,
                                          &*callable,
                                          arguments->deviceAddress + desc.argumentOffset);
        return {};
    }

} // namespace rachis::vulkan
