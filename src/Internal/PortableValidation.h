#pragma once

#include "Internal/BackendCommandBuffer.h"
#include "Internal/BackendDevice.h"
#include "Rachis/Binding.h"
#include "Rachis/Command.h"
#include "Rachis/Pipeline.h"
#include "Rachis/Rendering.h"
#include "Rachis/Resource.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace rachis::detail
{

    [[nodiscard]] inline Error preferDeviceLossError(const BackendDevice& backend, Error fallbackError)
    {
        if (backend.operationalState() == DeviceOperationalState::lost)
        {
            return backend.deviceLossError().value_or(Error{
                ErrorCode::deviceLost,
                0,
                "The device is lost.",
            });
        }
        return fallbackError;
    }

    [[nodiscard]] inline Status validateCommandBufferOwnership(const BackendCommandBuffer& commandBuffer,
                                                               std::uint64_t expectedDeviceToken)
    {
        if (expectedDeviceToken == 0 || commandBuffer.deviceToken() != expectedDeviceToken)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The command buffer belongs to another device or backend.",
            });
        }
        return {};
    }

    [[nodiscard]] inline Status validateCommandBufferReadyForSubmit(const BackendCommandBuffer& commandBuffer)
    {
        if (commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A command buffer cannot be submitted while a rendering scope is active.",
            });
        }
        if (commandBuffer.queryState().activeOcclusion.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A command buffer cannot be submitted while an occlusion query is active.",
            });
        }
        if (commandBuffer.queryState().activePipelineStatistics.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A command buffer cannot be submitted while a pipeline-statistics query is active.",
            });
        }
        if (std::ranges::any_of(commandBuffer.queryState().activeStreamOutputStatistics,
                                [](const auto& active) { return active.has_value(); }))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Cannot finish a command buffer with an active stream-output-statistics query.",
            });
        }
        if (commandBuffer.queryState().activeGpuProfileScopeDepth != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A command buffer cannot be submitted while a GPU profile scope is active.",
            });
        }
        if (commandBuffer.queryState().activeConditionalExecution.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A command buffer cannot be submitted while conditional execution is active.",
            });
        }
        if (commandBuffer.debugState().groupDepth != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A command buffer cannot be submitted while a debug group is open.",
            });
        }
        return {};
    }

    [[nodiscard]] inline Status validateCommandBufferReadyForReset(const BackendCommandBuffer& commandBuffer)
    {
        if (commandBuffer.graphicsState().renderingOwner != nullptr ||
            commandBuffer.queryState().activeOcclusion.has_value() ||
            commandBuffer.queryState().activePipelineStatistics.has_value() ||
            std::ranges::any_of(commandBuffer.queryState().activeStreamOutputStatistics,
                                [](const auto& active) { return active.has_value(); }) ||
            commandBuffer.queryState().activeGpuProfileScopeDepth != 0 ||
            commandBuffer.queryState().activeConditionalExecution.has_value() ||
            commandBuffer.debugState().groupDepth != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Command recording reset requires every rendering, query, conditional, profile, and debug scope to be "
                "closed.",
            });
        }
        return {};
    }

    [[nodiscard]] inline Status validateRenderingQueriesEnded(const BackendCommandBuffer& commandBuffer)
    {
        const auto& state = commandBuffer.queryState();
        if (state.activeOcclusion.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Rendering cannot end while an occlusion query is active.",
            });
        }
        if (state.activePipelineStatistics.has_value() && state.activePipelineStatistics->begunInRendering)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Rendering cannot end while a pipeline-statistics query that began inside it is active.",
            });
        }
        if (std::ranges::any_of(state.activeStreamOutputStatistics,
                                [](const auto& active) { return active.has_value() && active->begunInRendering; }))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Rendering cannot end while a stream-output-statistics query that began inside it is active.",
            });
        }
        return {};
    }

    [[nodiscard]] inline Status validateQueryEndRenderingScope(const BackendCommandBuffer& commandBuffer,
                                                               const ActiveQuery& active,
                                                               std::string_view message)
    {
        if ((commandBuffer.graphicsState().renderingOwner != nullptr) != active.begunInRendering)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                std::string(message),
            });
        }
        return {};
    }

    [[nodiscard]] inline Status validateDebugMarkerDesc(const DebugMarkerDesc& desc)
    {
        if (desc.label.empty() || desc.label.size() > maxDebugLabelBytes || desc.label.contains('\0'))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Debug marker labels must be non-empty, bounded, and contain no embedded NUL bytes.",
            });
        }
        if (desc.color.has_value())
        {
            const auto& color = *desc.color;
            const std::array components{color.red, color.green, color.blue, color.alpha};
            for (const float component : components)
            {
                if (!std::isfinite(component) || component < 0.0F || component > 1.0F)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Debug marker RGBA color components must be finite values in [0, 1].",
                    });
                }
            }
        }
        return {};
    }

    [[nodiscard]] inline Status validateDebugGroupBegin(const DebugCommandState& state, const DebugMarkerDesc& desc)
    {
        if (auto valid = validateDebugMarkerDesc(desc); !valid)
        {
            return valid;
        }
        if (state.groupDepth >= maxDebugGroupDepth)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The command buffer reached the portable debug-group nesting limit.",
            });
        }
        if (state.groupDepth != 0 && state.groupThread != std::this_thread::get_id())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Nested debug and GPU profile groups must be recorded on the thread that opened the outer group.",
            });
        }
        return {};
    }

    [[nodiscard]] inline Status validateDebugGroupEnd(const DebugCommandState& state,
                                                      DebugGroupOwner expectedOwner = DebugGroupOwner::user)
    {
        if (state.groupDepth == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "No debug group is currently open on the command buffer.",
            });
        }
        if (state.groupDepth > maxDebugGroupDepth || state.groupOwners[state.groupDepth - 1] != expectedOwner)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "endDebugGroup cannot close a GPU-profile-owned debug group.",
            });
        }
        if (state.groupThread != std::this_thread::get_id())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Debug groups must be closed on the thread that opened the outer group.",
            });
        }
        return {};
    }

    [[nodiscard]] inline Status validateViewports(std::span<const Viewport> viewports,
                                                  std::uint32_t maxViewports,
                                                  ErrorCode countExceededCode = ErrorCode::invalidArgument)
    {
        if (viewports.empty())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "At least one viewport must be provided.",
            });
        }
        if (viewports.size() > maxViewports || viewports.size() > maxViewportCount)
        {
            return std::unexpected(Error{
                countExceededCode,
                0,
                "Viewport count exceeds the device multi-viewport limit.",
            });
        }
        for (const auto& viewport : viewports)
        {
            if (!std::isfinite(viewport.x) || !std::isfinite(viewport.y) || !std::isfinite(viewport.width) ||
                !std::isfinite(viewport.height) || !std::isfinite(viewport.minDepth) ||
                !std::isfinite(viewport.maxDepth))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Viewport coordinates, dimensions, and depth bounds must be finite.",
                });
            }
            if (viewport.width <= 0.0F || viewport.height <= 0.0F)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Viewport width and height must be positive.",
                });
            }
            if (viewport.minDepth < 0.0F || viewport.maxDepth > 1.0F || viewport.minDepth > viewport.maxDepth)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Viewport depth bounds must satisfy 0 <= minDepth <= maxDepth <= 1.",
                });
            }
        }
        return {};
    }

    [[nodiscard]] inline Status validateScissors(std::span<const ScissorRect> scissors,
                                                 std::uint32_t maxViewports,
                                                 ErrorCode countExceededCode = ErrorCode::invalidArgument)
    {
        if (scissors.empty())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "At least one scissor rectangle must be provided.",
            });
        }
        if (scissors.size() > maxViewports || scissors.size() > maxViewportCount)
        {
            return std::unexpected(Error{
                countExceededCode,
                0,
                "Scissor count exceeds the device multi-viewport limit.",
            });
        }
        constexpr auto maxCoordinate = static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
        for (const auto& scissor : scissors)
        {
            if (scissor.x < 0 || scissor.y < 0 || scissor.width == 0 || scissor.height == 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Scissor rectangles must have non-negative origin and positive extent.",
                });
            }
            if (static_cast<std::uint64_t>(scissor.x) + static_cast<std::uint64_t>(scissor.width) > maxCoordinate ||
                static_cast<std::uint64_t>(scissor.y) + static_cast<std::uint64_t>(scissor.height) > maxCoordinate)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Scissor rectangle origin plus extent overflows int32_t.",
                });
            }
        }
        return {};
    }

    [[nodiscard]] inline Status validateBlendConstants(const BlendConstants& constants)
    {
        if (!std::isfinite(constants.red) || !std::isfinite(constants.green) || !std::isfinite(constants.blue) ||
            !std::isfinite(constants.alpha))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Blend constants must be finite.",
            });
        }
        return {};
    }

    [[nodiscard]] inline Status requireActiveRendering(const BackendCommandBuffer& commandBuffer,
                                                       std::string_view message)
    {
        if (commandBuffer.graphicsState().renderingOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                std::string(message),
            });
        }
        return {};
    }

    [[nodiscard]] inline Status requireOutsideRendering(const BackendCommandBuffer& commandBuffer,
                                                        std::string_view message)
    {
        if (commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                std::string(message),
            });
        }
        return {};
    }

    [[nodiscard]] inline Status requireComputeCapableOutsideRendering(const BackendCommandBuffer& commandBuffer,
                                                                      std::string_view message)
    {
        const auto queueType = commandBuffer.queueType();
        if ((queueType != QueueType::graphics && queueType != QueueType::compute) ||
            commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                std::string(message),
            });
        }
        return {};
    }

    [[nodiscard]] inline Status requireOutsideConditionalExecution(const BackendCommandBuffer& commandBuffer,
                                                                   std::string_view message)
    {
        if (commandBuffer.queryState().activeConditionalExecution.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                std::string(message),
            });
        }
        return {};
    }

    [[nodiscard]] inline Status validateRenderingEnd(const BackendCommandBuffer& commandBuffer)
    {
        if (commandBuffer.graphicsState().renderingOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "No rendering scope is currently active.",
            });
        }
        return validateRenderingQueriesEnded(commandBuffer);
    }

    [[nodiscard]] inline Status requireBoundComputePipeline(const BackendCommandBuffer& commandBuffer)
    {
        if (commandBuffer.computeState().pipelineOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A compute pipeline must be bound before setting push constants.",
            });
        }
        return {};
    }

    [[nodiscard]] inline Status requireBoundGraphicsPipeline(const BackendCommandBuffer& commandBuffer)
    {
        if (commandBuffer.graphicsState().renderingOwner == nullptr ||
            commandBuffer.graphicsState().pipelineOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Graphics push constants require an active rendering scope and bound graphics pipeline.",
            });
        }
        return {};
    }

    [[nodiscard]] inline Status validateComputePushConstants(std::uint32_t offset, std::span<const std::byte> data)
    {
        if (data.empty() || data.size() > maxPushConstantBytes || (offset % 4U) != 0U || (data.size() % 4U) != 0U ||
            offset > maxPushConstantBytes - static_cast<std::uint32_t>(data.size()))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Push constants must be non-empty, 4-byte aligned, and fit within maxPushConstantBytes.",
            });
        }
        return {};
    }

    [[nodiscard]] inline Status validateGraphicsPushConstants(ShaderStage visibility,
                                                              std::uint32_t offset,
                                                              std::span<const std::byte> data)
    {
        if (visibility == ShaderStage::none || (visibility & ShaderStage::allGraphics) != visibility)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Graphics push constants require one or more graphics shader stages.",
            });
        }
        return validateComputePushConstants(offset, data);
    }

    [[nodiscard]] inline Status validateCopyRegionCount(std::size_t regionCount, std::string_view message)
    {
        if (regionCount == 0 || regionCount > std::numeric_limits<std::uint32_t>::max())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                std::string(message),
            });
        }
        return {};
    }

    [[nodiscard]] inline Status validateBufferCopyRegions(std::span<const BufferCopyRegion> regions)
    {
        if (auto validCount = validateCopyRegionCount(
                regions.size(), "Buffer copies require at least one region and at most uint32_t::max regions.");
            !validCount)
        {
            return validCount;
        }
        for (const auto& region : regions)
        {
            if (region.size == 0 || region.sourceOffset > std::numeric_limits<std::uint64_t>::max() - region.size ||
                region.destinationOffset > std::numeric_limits<std::uint64_t>::max() - region.size)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Buffer copy regions must be non-empty and must not overflow 64-bit offsets.",
                });
            }
        }
        return {};
    }

    [[nodiscard]] inline Status validateBindingLayoutDesc(const BindingLayoutDesc& desc)
    {
        constexpr auto allowedVisibility = static_cast<std::uint16_t>(ShaderStage::all);
        constexpr auto allowedFlags = static_cast<std::uint8_t>(BindingFlag::partiallyBound) |
                                      static_cast<std::uint8_t>(BindingFlag::variableDescriptorCount);

        std::optional<std::uint32_t> variableBinding;
        std::uint32_t maxBinding = 0;

        for (std::size_t index = 0; index < desc.entries.size(); ++index)
        {
            const auto& entry = desc.entries[index];
            if (static_cast<std::size_t>(entry.type) >= bindingTypeCount)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Binding layout entry has an unknown binding type.",
                });
            }

            const auto visibility = static_cast<std::uint16_t>(entry.visibility);
            const auto flags = static_cast<std::uint8_t>(entry.flags);
            const bool variable = hasAny(entry.flags, BindingFlag::variableDescriptorCount);
            const bool unbounded = entry.count == unboundedBindingCount;

            if (entry.count == 0 ||
                (!unbounded && entry.binding > std::numeric_limits<std::uint32_t>::max() - (entry.count - 1U)) ||
                visibility == 0 || (visibility & ~allowedVisibility) != 0 || (flags & ~allowedFlags) != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Binding layout entries require a non-empty register range, valid visibility, and valid flags.",
                });
            }

            if (variable != unbounded)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "An unbounded binding count and the variable-descriptor-count flag must be specified together.",
                });
            }

            const bool isDynamicBuffer = entry.type == BindingType::dynamicUniformBuffer ||
                                         entry.type == BindingType::dynamicReadOnlyStorageBuffer ||
                                         entry.type == BindingType::dynamicStorageBuffer;
            if (isDynamicBuffer && entry.flags != BindingFlag::none)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Dynamic buffer bindings cannot carry partially-bound or variable-descriptor-count flags.",
                });
            }

            for (std::size_t previous = 0; previous < index; ++previous)
            {
                if (desc.entries[previous].binding == entry.binding)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Binding layout binding numbers must be unique.",
                    });
                }
            }

            if (index == 0 || entry.binding > maxBinding)
            {
                maxBinding = entry.binding;
            }

            if (variable)
            {
                if (variableBinding.has_value())
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A binding layout may declare at most one variable-descriptor-count entry.",
                    });
                }
                variableBinding = entry.binding;
            }
        }

        if (variableBinding.has_value() && *variableBinding != maxBinding)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A variable-descriptor-count binding layout entry must use the largest binding number.",
            });
        }

        return {};
    }

    [[nodiscard]] inline Status validateBindingPoolDesc(const BindingPoolDesc& desc)
    {
        if (desc.maxBindingSets == 0 || desc.sizes.size() > bindingTypeCount)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Binding pools require non-zero set capacity and at most one size entry per binding type.",
            });
        }

        std::array<bool, bindingTypeCount> seenTypes{};
        for (const auto& size : desc.sizes)
        {
            const auto typeIndex = static_cast<std::size_t>(size.type);
            if (typeIndex >= bindingTypeCount || size.descriptorCount == 0 || seenTypes[typeIndex])
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Binding pool sizes must use unique valid binding types and non-zero descriptor counts.",
                });
            }
            seenTypes[typeIndex] = true;
        }

        return {};
    }

    [[nodiscard]] constexpr bool canReadDrawIndex(ShaderStage stage) noexcept
    {
        return stage == ShaderStage::vertex || stage == ShaderStage::task || stage == ShaderStage::mesh;
    }

    [[nodiscard]] inline Status validateShaderDrawIndex(const ShaderModuleDesc& desc,
                                                        const DeviceCapabilities& capabilities)
    {
        if (!desc.shaderInterface.drawIndex)
        {
            return {};
        }
        if (!canReadDrawIndex(desc.stage))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Only vertex, task, and mesh shaders can read the draw index.",
            });
        }
        if (!capabilities.shaderDrawIndex)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support shaders that read the draw index.",
            });
        }
        return {};
    }

    [[nodiscard]] inline Status validateDrawIndexSupport(bool requested,
                                                         const DeviceCapabilities& capabilities,
                                                         std::string_view message)
    {
        if (requested && !capabilities.shaderDrawIndex)
        {
            return std::unexpected(Error{ErrorCode::unsupported, 0, std::string(message)});
        }
        return {};
    }

} // namespace rachis::detail
