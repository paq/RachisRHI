#pragma once

#include "Rachis/Command.h"
#include "Rachis/Resource.h"

#include <array>
#include <cstdint>
#include <type_traits>

namespace rachis::d3d12
{

    [[nodiscard]] constexpr bool validPipelineStages(PipelineStage stages) noexcept
    {
        constexpr auto allowed =
            static_cast<std::uint32_t>(PipelineStage::all) | static_cast<std::uint32_t>(PipelineStage::draw) |
            static_cast<std::uint32_t>(PipelineStage::vertexInput) |
            static_cast<std::uint32_t>(PipelineStage::vertexShader) |
            static_cast<std::uint32_t>(PipelineStage::pixelShader) |
            static_cast<std::uint32_t>(PipelineStage::computeShader) | static_cast<std::uint32_t>(PipelineStage::copy) |
            static_cast<std::uint32_t>(PipelineStage::resolve) |
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
            static_cast<std::uint32_t>(Access::shaderResource) | static_cast<std::uint32_t>(Access::unorderedAccess) |
            static_cast<std::uint32_t>(Access::copySource) | static_cast<std::uint32_t>(Access::copyDestination) |
            static_cast<std::uint32_t>(Access::indirectArgument) | static_cast<std::uint32_t>(Access::renderTarget) |
            static_cast<std::uint32_t>(Access::depthStencilRead) |
            static_cast<std::uint32_t>(Access::depthStencilWrite) | static_cast<std::uint32_t>(Access::resolveSource) |
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

    [[nodiscard]] constexpr bool hasWriteAccess(Access access) noexcept
    {
        constexpr auto writeAccess = Access::unorderedAccess | Access::copyDestination | Access::renderTarget |
                                     Access::depthStencilWrite | Access::resolveDestination |
                                     Access::accelerationStructureScratch | Access::streamOutputWrite;
        return hasAny(access, writeAccess);
    }

    [[nodiscard]] constexpr bool combinesWriteWithOtherAccess(Access access) noexcept
    {
        const auto raw = static_cast<std::uint32_t>(access);
        return hasWriteAccess(access) && (raw & (raw - 1U)) != 0;
    }

    [[nodiscard]] constexpr bool accessMatchesStages(PipelineStage stages, Access access) noexcept
    {
        const auto matches = [&](Access requiredAccess, PipelineStage compatibleStages) {
            return !hasAny(access, requiredAccess) || hasAny(stages, PipelineStage::all | compatibleStages);
        };
        return matches(Access::vertexBuffer, PipelineStage::draw | PipelineStage::vertexShader) &&
               matches(Access::constantBuffer,
                       PipelineStage::draw | PipelineStage::vertexShader | PipelineStage::pixelShader |
                           PipelineStage::computeShader | PipelineStage::geometryShader | PipelineStage::hullShader |
                           PipelineStage::domainShader | PipelineStage::taskShader | PipelineStage::meshShader |
                           PipelineStage::rayTracingShader) &&
               matches(Access::indexBuffer, PipelineStage::draw | PipelineStage::vertexInput) &&
               matches(Access::shaderResource,
                       PipelineStage::draw | PipelineStage::vertexShader | PipelineStage::pixelShader |
                           PipelineStage::computeShader | PipelineStage::geometryShader | PipelineStage::hullShader |
                           PipelineStage::domainShader | PipelineStage::taskShader | PipelineStage::meshShader |
                           PipelineStage::rayTracingShader | PipelineStage::accelerationStructureCopy) &&
               matches(Access::unorderedAccess,
                       PipelineStage::draw | PipelineStage::vertexShader | PipelineStage::pixelShader |
                           PipelineStage::computeShader | PipelineStage::geometryShader | PipelineStage::hullShader |
                           PipelineStage::domainShader | PipelineStage::taskShader | PipelineStage::meshShader |
                           PipelineStage::rayTracingShader | PipelineStage::accelerationStructureCopy) &&
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
        const auto accessFits = [&](Access allowed) { return (rawAccess & ~static_cast<std::uint32_t>(allowed)) == 0; };
        if (queueType == QueueType::compute)
        {
            return stagesFit(PipelineStage::all | PipelineStage::computeShader | PipelineStage::copy |
                             PipelineStage::indirect | PipelineStage::conditionalExecution |
                             PipelineStage::accelerationStructureBuild | PipelineStage::rayTracingShader |
                             PipelineStage::accelerationStructureCopy) &&
                   accessFits(Access::common | Access::constantBuffer | Access::shaderResource |
                              Access::unorderedAccess | Access::copySource | Access::copyDestination |
                              Access::indirectArgument | Access::predicate | Access::accelerationStructureBuildInput |
                              Access::accelerationStructureScratch);
        }
        return stagesFit(PipelineStage::all | PipelineStage::copy) &&
               accessFits(Access::common | Access::copySource | Access::copyDestination);
    }

    [[nodiscard]] constexpr bool validBarrierScope(PipelineStage stages, Access access, QueueType queueType) noexcept
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
        if (hasAny(usage, BufferUsage::streamOutput))
        {
            allowed = allowed | Access::streamOutputWrite;
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
            return hasAny(usage, TextureUsage::copySource | TextureUsage::renderTarget | TextureUsage::depthStencil);
        case TextureLayout::resolveDestination:
            return hasAny(usage,
                          TextureUsage::copyDestination | TextureUsage::renderTarget | TextureUsage::depthStencil);
        case TextureLayout::shadingRateSource:
            return hasAny(usage, TextureUsage::shadingRateSource);
        case TextureLayout::present:
            return false;
        }
        return false;
    }

    enum class LegacyResourceState : std::uint32_t
    {
        common = 0,
        present = 0,
        vertexAndConstantBuffer = 1U << 0U,
        indexBuffer = 1U << 1U,
        renderTarget = 1U << 2U,
        unorderedAccess = 1U << 3U,
        depthWrite = 1U << 4U,
        depthRead = 1U << 5U,
        nonPixelShaderResource = 1U << 6U,
        pixelShaderResource = 1U << 7U,
        streamOut = 1U << 8U,
        indirectArgument = 1U << 9U,
        predication = 1U << 9U,
        copyDest = 1U << 10U,
        copySource = 1U << 11U,
        resolveDest = 1U << 12U,
        resolveSource = 1U << 13U,
        raytracingAccelerationStructure = 1U << 22U,
        shadingRateSource = 1U << 24U,
    };

    [[nodiscard]] constexpr LegacyResourceState operator|(LegacyResourceState left, LegacyResourceState right) noexcept
    {
        return static_cast<LegacyResourceState>(static_cast<std::underlying_type_t<LegacyResourceState>>(left) |
                                                static_cast<std::underlying_type_t<LegacyResourceState>>(right));
    }

    constexpr LegacyResourceState& operator|=(LegacyResourceState& left, LegacyResourceState right) noexcept
    {
        left = left | right;
        return left;
    }

    [[nodiscard]] constexpr LegacyResourceState operator&(LegacyResourceState left, LegacyResourceState right) noexcept
    {
        return static_cast<LegacyResourceState>(static_cast<std::underlying_type_t<LegacyResourceState>>(left) &
                                                static_cast<std::underlying_type_t<LegacyResourceState>>(right));
    }

    [[nodiscard]] constexpr LegacyResourceState operator~(LegacyResourceState value) noexcept
    {
        return static_cast<LegacyResourceState>(~static_cast<std::underlying_type_t<LegacyResourceState>>(value));
    }

    [[nodiscard]] constexpr bool hasAny(LegacyResourceState value, LegacyResourceState flags) noexcept
    {
        return (value & flags) != LegacyResourceState::common;
    }

    [[nodiscard]] constexpr LegacyResourceState validLegacyStatesForQueue(QueueType queueType) noexcept
    {
        switch (queueType)
        {
        case QueueType::graphics:
            return LegacyResourceState::vertexAndConstantBuffer | LegacyResourceState::indexBuffer |
                   LegacyResourceState::renderTarget | LegacyResourceState::unorderedAccess |
                   LegacyResourceState::depthWrite | LegacyResourceState::depthRead |
                   LegacyResourceState::nonPixelShaderResource | LegacyResourceState::pixelShaderResource |
                   LegacyResourceState::streamOut | LegacyResourceState::indirectArgument |
                   LegacyResourceState::copyDest | LegacyResourceState::copySource | LegacyResourceState::resolveDest |
                   LegacyResourceState::resolveSource | LegacyResourceState::raytracingAccelerationStructure |
                   LegacyResourceState::shadingRateSource;
        case QueueType::compute:
            return LegacyResourceState::vertexAndConstantBuffer | LegacyResourceState::unorderedAccess |
                   LegacyResourceState::nonPixelShaderResource | LegacyResourceState::indirectArgument |
                   LegacyResourceState::predication | LegacyResourceState::copyDest | LegacyResourceState::copySource |
                   LegacyResourceState::raytracingAccelerationStructure;
        case QueueType::copy:
            return LegacyResourceState::copyDest | LegacyResourceState::copySource;
        }
        return LegacyResourceState::common;
    }

    [[nodiscard]] constexpr LegacyResourceState translateLegacyState(Access access, QueueType queueType) noexcept
    {
        LegacyResourceState result = LegacyResourceState::common;
        if (hasAny(access, Access::vertexBuffer) || hasAny(access, Access::constantBuffer))
        {
            result |= LegacyResourceState::vertexAndConstantBuffer;
        }
        if (hasAny(access, Access::indexBuffer))
        {
            result |= LegacyResourceState::indexBuffer;
        }
        if (hasAny(access, Access::shaderResource))
        {
            result |= LegacyResourceState::nonPixelShaderResource;
            if (queueType == QueueType::graphics)
            {
                result |= LegacyResourceState::pixelShaderResource;
            }
        }
        if (hasAny(access, Access::unorderedAccess))
        {
            result |= LegacyResourceState::unorderedAccess;
        }
        if (hasAny(access, Access::copySource))
        {
            result |= LegacyResourceState::copySource;
        }
        if (hasAny(access, Access::copyDestination))
        {
            result |= LegacyResourceState::copyDest;
        }
        if (hasAny(access, Access::indirectArgument))
        {
            result |= LegacyResourceState::indirectArgument;
        }
        if (hasAny(access, Access::predicate))
        {
            result |= LegacyResourceState::predication;
        }
        if (hasAny(access, Access::renderTarget))
        {
            result |= LegacyResourceState::renderTarget;
        }
        if (hasAny(access, Access::depthStencilRead))
        {
            result |= LegacyResourceState::depthRead;
        }
        if (hasAny(access, Access::depthStencilWrite))
        {
            result |= LegacyResourceState::depthWrite;
        }
        if (hasAny(access, Access::resolveSource))
        {
            result |= LegacyResourceState::resolveSource;
        }
        if (hasAny(access, Access::resolveDestination))
        {
            result |= LegacyResourceState::resolveDest;
        }
        if (hasAny(access, Access::shadingRateSource))
        {
            result |= LegacyResourceState::shadingRateSource;
        }
        if (hasAny(access, Access::accelerationStructureBuildInput))
        {
            result |= LegacyResourceState::nonPixelShaderResource;
        }
        if (hasAny(access, Access::accelerationStructureScratch))
        {
            result |= LegacyResourceState::unorderedAccess;
        }
        if (hasAny(access, Access::streamOutputWrite))
        {
            result |= LegacyResourceState::streamOut;
        }
        return result & validLegacyStatesForQueue(queueType);
    }

    [[nodiscard]] constexpr LegacyResourceState translateLegacyTextureState(TextureLayout layout,
                                                                            Access access,
                                                                            QueueType queueType) noexcept
    {
        if (queueType == QueueType::copy)
        {
            return LegacyResourceState::common;
        }
        LegacyResourceState result = LegacyResourceState::common;
        switch (layout)
        {
        case TextureLayout::present:
            result = access == Access::none ? LegacyResourceState::present : translateLegacyState(access, queueType);
            break;
        case TextureLayout::shaderResource:
            result = translateLegacyState(Access::shaderResource, queueType);
            break;
        case TextureLayout::unorderedAccess:
            result = LegacyResourceState::unorderedAccess;
            break;
        case TextureLayout::renderTarget:
            result = LegacyResourceState::renderTarget;
            break;
        case TextureLayout::depthStencilWrite:
            result = LegacyResourceState::depthWrite;
            break;
        case TextureLayout::depthStencilRead:
            result = LegacyResourceState::depthRead;
            break;
        case TextureLayout::copySource:
            result = LegacyResourceState::copySource;
            break;
        case TextureLayout::copyDestination:
            result = LegacyResourceState::copyDest;
            break;
        case TextureLayout::resolveSource:
            result = LegacyResourceState::resolveSource;
            break;
        case TextureLayout::resolveDestination:
            result = LegacyResourceState::resolveDest;
            break;
        case TextureLayout::shadingRateSource:
            result = LegacyResourceState::shadingRateSource;
            break;
        default:
            result = translateLegacyState(access, queueType);
            break;
        }
        return result & validLegacyStatesForQueue(queueType);
    }

    [[nodiscard]] constexpr LegacyResourceState translateLegacyBufferState(Access access, QueueType queueType) noexcept
    {
        auto result = translateLegacyState(access, queueType);
        if (hasAny(result, LegacyResourceState::indirectArgument))
        {
            result |= LegacyResourceState::copySource;
        }
        return result;
    }

    [[nodiscard]] constexpr bool legacyStateRepresentsTextureAccess(TextureLayout layout,
                                                                    Access access,
                                                                    QueueType queueType) noexcept
    {
        const bool stateFollowsAccess =
            layout == TextureLayout::undefined || layout == TextureLayout::common || layout == TextureLayout::present;
        return queueType == QueueType::copy || !stateFollowsAccess || !combinesWriteWithOtherAccess(access);
    }

    enum class LegacyBufferBarrierKind : std::uint8_t
    {
        none,
        uav,
        transition,
    };

    struct LegacyBufferBarrierPlan final
    {
        LegacyBufferBarrierKind kind = LegacyBufferBarrierKind::none;
        LegacyResourceState stateBefore = LegacyResourceState::common;
        LegacyResourceState stateAfter = LegacyResourceState::common;
    };

    [[nodiscard]] constexpr LegacyBufferBarrierPlan planLegacyBufferBarrier(MemoryType memoryType,
                                                                            Access accessBefore,
                                                                            Access accessAfter,
                                                                            QueueType queueType) noexcept
    {
        if (memoryType != MemoryType::deviceLocal)
        {
            return {};
        }
        const auto before = translateLegacyBufferState(accessBefore, queueType);
        const auto after = translateLegacyBufferState(accessAfter, queueType);
        if (before == after && hasAny(accessAfter, Access::unorderedAccess))
        {
            return LegacyBufferBarrierPlan{
                .kind = LegacyBufferBarrierKind::uav,
                .stateBefore = before,
                .stateAfter = after,
            };
        }
        if (before != after)
        {
            return LegacyBufferBarrierPlan{
                .kind = LegacyBufferBarrierKind::transition,
                .stateBefore = before,
                .stateAfter = after,
            };
        }
        return {};
    }

    enum class EnhancedBarrierSync : std::uint32_t
    {
        none = 0,
        all = 1U << 0U,
        draw = 1U << 1U,
        indexInput = 1U << 2U,
        vertexShading = 1U << 3U,
        pixelShading = 1U << 4U,
        depthStencil = 1U << 5U,
        renderTarget = 1U << 6U,
        computeShading = 1U << 7U,
        raytracing = 1U << 8U,
        copy = 1U << 9U,
        resolve = 1U << 10U,
        executeIndirect = 1U << 11U,
        predication = 1U << 12U,
        buildRaytracingAccelerationStructure = 1U << 14U,
        copyRaytracingAccelerationStructure = 1U << 15U,
        clearUnorderedAccessView = 1U << 16U,
    };

    [[nodiscard]] constexpr EnhancedBarrierSync operator|(EnhancedBarrierSync left, EnhancedBarrierSync right) noexcept
    {
        return static_cast<EnhancedBarrierSync>(static_cast<std::underlying_type_t<EnhancedBarrierSync>>(left) |
                                                static_cast<std::underlying_type_t<EnhancedBarrierSync>>(right));
    }

    constexpr EnhancedBarrierSync& operator|=(EnhancedBarrierSync& left, EnhancedBarrierSync right) noexcept
    {
        left = left | right;
        return left;
    }

    [[nodiscard]] constexpr EnhancedBarrierSync operator&(EnhancedBarrierSync left, EnhancedBarrierSync right) noexcept
    {
        return static_cast<EnhancedBarrierSync>(static_cast<std::underlying_type_t<EnhancedBarrierSync>>(left) &
                                                static_cast<std::underlying_type_t<EnhancedBarrierSync>>(right));
    }

    [[nodiscard]] constexpr bool hasAny(EnhancedBarrierSync value, EnhancedBarrierSync flags) noexcept
    {
        return (value & flags) != EnhancedBarrierSync::none;
    }

    [[nodiscard]] constexpr EnhancedBarrierSync translateEnhancedSync(PipelineStage stages) noexcept
    {
        if (hasAny(stages, PipelineStage::all))
        {
            return EnhancedBarrierSync::all;
        }

        EnhancedBarrierSync result = EnhancedBarrierSync::none;
        if (hasAny(stages, PipelineStage::draw))
        {
            result |= EnhancedBarrierSync::draw;
        }
        if (hasAny(stages, PipelineStage::vertexInput))
        {
            result |= EnhancedBarrierSync::indexInput;
        }
        if (hasAny(stages,
                   PipelineStage::vertexShader | PipelineStage::hullShader | PipelineStage::domainShader |
                       PipelineStage::geometryShader | PipelineStage::taskShader | PipelineStage::meshShader |
                       PipelineStage::streamOutput))
        {
            result |= EnhancedBarrierSync::vertexShading;
        }
        if (hasAny(stages, PipelineStage::pixelShader))
        {
            result |= EnhancedBarrierSync::pixelShading;
        }
        if (hasAny(stages, PipelineStage::computeShader))
        {
            result |= EnhancedBarrierSync::computeShading;
        }
        if (hasAny(stages, PipelineStage::copy))
        {
            result |= EnhancedBarrierSync::copy;
        }
        if (hasAny(stages, PipelineStage::resolve))
        {
            result |= EnhancedBarrierSync::resolve;
        }
        if (hasAny(stages, PipelineStage::renderTarget))
        {
            result |= EnhancedBarrierSync::renderTarget;
        }
        if (hasAny(stages, PipelineStage::depthStencil))
        {
            result |= EnhancedBarrierSync::depthStencil;
        }
        if (hasAny(stages, PipelineStage::indirect))
        {
            result |= EnhancedBarrierSync::executeIndirect;
        }
        if (hasAny(stages, PipelineStage::conditionalExecution))
        {
            result |= EnhancedBarrierSync::predication;
        }
        if (hasAny(stages, PipelineStage::accelerationStructureBuild))
        {
            result |= EnhancedBarrierSync::buildRaytracingAccelerationStructure;
        }
        if (hasAny(stages, PipelineStage::rayTracingShader))
        {
            result |= EnhancedBarrierSync::raytracing;
        }
        if (hasAny(stages, PipelineStage::accelerationStructureCopy))
        {
            result |= EnhancedBarrierSync::copyRaytracingAccelerationStructure;
        }
        return result;
    }

    [[nodiscard]] constexpr EnhancedBarrierSync translateEnhancedSync(PipelineStage stages, Access access) noexcept
    {
        auto result = translateEnhancedSync(stages);
        if (result != EnhancedBarrierSync::all && hasAny(stages, PipelineStage::computeShader) &&
            hasAny(access, Access::unorderedAccess))
        {
            result |= EnhancedBarrierSync::clearUnorderedAccessView;
        }
        return result;
    }

    enum class EnhancedBarrierAccess : std::uint32_t
    {
        common = 0,
        vertexBuffer = 1U << 0U,
        constantBuffer = 1U << 1U,
        indexBuffer = 1U << 2U,
        renderTarget = 1U << 3U,
        unorderedAccess = 1U << 4U,
        depthStencilWrite = 1U << 5U,
        depthStencilRead = 1U << 6U,
        shaderResource = 1U << 7U,
        streamOutput = 1U << 8U,
        indirectArgument = 1U << 9U,
        predication = 1U << 10U,
        copyDest = 1U << 11U,
        copySource = 1U << 12U,
        resolveDest = 1U << 13U,
        resolveSource = 1U << 14U,
        shadingRateSource = 1U << 15U,
        noAccess = 1U << 31U,
    };

    [[nodiscard]] constexpr EnhancedBarrierAccess operator|(EnhancedBarrierAccess left,
                                                            EnhancedBarrierAccess right) noexcept
    {
        return static_cast<EnhancedBarrierAccess>(static_cast<std::underlying_type_t<EnhancedBarrierAccess>>(left) |
                                                  static_cast<std::underlying_type_t<EnhancedBarrierAccess>>(right));
    }

    constexpr EnhancedBarrierAccess& operator|=(EnhancedBarrierAccess& left, EnhancedBarrierAccess right) noexcept
    {
        left = left | right;
        return left;
    }

    [[nodiscard]] constexpr EnhancedBarrierAccess operator&(EnhancedBarrierAccess left,
                                                            EnhancedBarrierAccess right) noexcept
    {
        return static_cast<EnhancedBarrierAccess>(static_cast<std::underlying_type_t<EnhancedBarrierAccess>>(left) &
                                                  static_cast<std::underlying_type_t<EnhancedBarrierAccess>>(right));
    }

    [[nodiscard]] constexpr bool hasAny(EnhancedBarrierAccess value, EnhancedBarrierAccess flags) noexcept
    {
        return (value & flags) != EnhancedBarrierAccess::common;
    }

    [[nodiscard]] constexpr EnhancedBarrierAccess translateEnhancedAccess(Access access) noexcept
    {
        if (access == Access::none)
        {
            return EnhancedBarrierAccess::noAccess;
        }

        EnhancedBarrierAccess result = EnhancedBarrierAccess::common;
        if (hasAny(access, Access::common))
        {
            result |= EnhancedBarrierAccess::common;
        }
        if (hasAny(access, Access::vertexBuffer))
        {
            result |= EnhancedBarrierAccess::vertexBuffer;
        }
        if (hasAny(access, Access::constantBuffer))
        {
            result |= EnhancedBarrierAccess::constantBuffer;
        }
        if (hasAny(access, Access::indexBuffer))
        {
            result |= EnhancedBarrierAccess::indexBuffer;
        }
        if (hasAny(access, Access::shaderResource))
        {
            result |= EnhancedBarrierAccess::shaderResource;
        }
        if (hasAny(access, Access::unorderedAccess))
        {
            result |= EnhancedBarrierAccess::unorderedAccess;
        }
        if (hasAny(access, Access::copySource))
        {
            result |= EnhancedBarrierAccess::copySource;
        }
        if (hasAny(access, Access::copyDestination))
        {
            result |= EnhancedBarrierAccess::copyDest;
        }
        if (hasAny(access, Access::indirectArgument))
        {
            result |= EnhancedBarrierAccess::indirectArgument;
        }
        if (hasAny(access, Access::predicate))
        {
            result |= EnhancedBarrierAccess::predication;
        }
        if (hasAny(access, Access::renderTarget))
        {
            result |= EnhancedBarrierAccess::renderTarget;
        }
        if (hasAny(access, Access::depthStencilRead))
        {
            result |= EnhancedBarrierAccess::depthStencilRead;
        }
        if (hasAny(access, Access::depthStencilWrite))
        {
            result |= EnhancedBarrierAccess::depthStencilWrite;
        }
        if (hasAny(access, Access::resolveSource))
        {
            result |= EnhancedBarrierAccess::resolveSource;
        }
        if (hasAny(access, Access::resolveDestination))
        {
            result |= EnhancedBarrierAccess::resolveDest;
        }
        if (hasAny(access, Access::shadingRateSource))
        {
            result |= EnhancedBarrierAccess::shadingRateSource;
        }
        if (hasAny(access, Access::accelerationStructureBuildInput))
        {
            result |= EnhancedBarrierAccess::shaderResource;
        }
        if (hasAny(access, Access::accelerationStructureScratch))
        {
            result |= EnhancedBarrierAccess::unorderedAccess;
        }
        if (hasAny(access, Access::streamOutputWrite))
        {
            result |= EnhancedBarrierAccess::streamOutput;
        }
        return result;
    }

    enum class EnhancedBarrierLayout : std::uint8_t
    {
        undefined,
        common,
        present,
        shaderResource,
        unorderedAccess,
        renderTarget,
        depthStencilWrite,
        depthStencilRead,
        resolveSource,
        resolveDest,
        shadingRateSource,
    };

    [[nodiscard]] constexpr EnhancedBarrierLayout translateEnhancedLayout(TextureLayout layout,
                                                                          QueueType queueType) noexcept
    {
        if (queueType == QueueType::copy && layout != TextureLayout::undefined)
        {
            return EnhancedBarrierLayout::common;
        }
        switch (layout)
        {
        case TextureLayout::undefined:
            return EnhancedBarrierLayout::undefined;
        case TextureLayout::common:
        case TextureLayout::copySource:
        case TextureLayout::copyDestination:
            return EnhancedBarrierLayout::common;
        case TextureLayout::present:
            return EnhancedBarrierLayout::present;
        case TextureLayout::shaderResource:
            return EnhancedBarrierLayout::shaderResource;
        case TextureLayout::unorderedAccess:
            return EnhancedBarrierLayout::unorderedAccess;
        case TextureLayout::renderTarget:
            return EnhancedBarrierLayout::renderTarget;
        case TextureLayout::depthStencilWrite:
            return EnhancedBarrierLayout::depthStencilWrite;
        case TextureLayout::depthStencilRead:
            return EnhancedBarrierLayout::depthStencilRead;
        case TextureLayout::resolveSource:
            return EnhancedBarrierLayout::resolveSource;
        case TextureLayout::resolveDestination:
            return EnhancedBarrierLayout::resolveDest;
        case TextureLayout::shadingRateSource:
            return EnhancedBarrierLayout::shadingRateSource;
        }
        return EnhancedBarrierLayout::common;
    }

    enum class EnhancedTextureBarrierFlags : std::uint8_t
    {
        none = 0,
        discard = 1,
    };

    enum class EnhancedTextureAliasTarget : std::uint8_t
    {
        before,
        after,
    };

    struct EnhancedTextureAliasStep final
    {
        EnhancedTextureAliasTarget target = EnhancedTextureAliasTarget::before;
        EnhancedBarrierSync syncBefore = EnhancedBarrierSync::none;
        EnhancedBarrierSync syncAfter = EnhancedBarrierSync::none;
        EnhancedBarrierAccess accessBefore = EnhancedBarrierAccess::common;
        EnhancedBarrierAccess accessAfter = EnhancedBarrierAccess::common;
        EnhancedBarrierLayout layoutBefore = EnhancedBarrierLayout::undefined;
        EnhancedBarrierLayout layoutAfter = EnhancedBarrierLayout::undefined;
        EnhancedTextureBarrierFlags flags = EnhancedTextureBarrierFlags::none;
        bool discardAfterBarrier = false;
    };

    struct EnhancedTextureAliasPlan final
    {
        std::array<EnhancedTextureAliasStep, 3> steps{};
        std::uint32_t stepCount = 0;
    };

    [[nodiscard]] constexpr EnhancedTextureAliasPlan planEnhancedTextureAliasActivation(
        const TextureAliasActivation& activation, TextureUsage afterUsage, QueueType queueType) noexcept
    {
        EnhancedTextureAliasPlan plan{};
        plan.steps[0] = EnhancedTextureAliasStep{
            .target = EnhancedTextureAliasTarget::before,
            .syncBefore = translateEnhancedSync(activation.stagesBefore, activation.accessBefore),
            .syncAfter = translateEnhancedSync(activation.stagesAfter),
            .accessBefore = translateEnhancedAccess(activation.accessBefore),
            .accessAfter = EnhancedBarrierAccess::noAccess,
            .layoutBefore = translateEnhancedLayout(activation.layoutBefore, queueType),
            .layoutAfter = EnhancedBarrierLayout::undefined,
            .flags = EnhancedTextureBarrierFlags::none,
            .discardAfterBarrier = false,
        };

        if (hasAny(afterUsage, TextureUsage::renderTarget | TextureUsage::depthStencil))
        {
            const bool depth = hasAny(afterUsage, TextureUsage::depthStencil);
            const auto initializationSync =
                depth ? EnhancedBarrierSync::depthStencil : EnhancedBarrierSync::renderTarget;
            const auto initializationAccess =
                depth ? EnhancedBarrierAccess::depthStencilWrite : EnhancedBarrierAccess::renderTarget;
            const auto initializationLayout =
                depth ? EnhancedBarrierLayout::depthStencilWrite : EnhancedBarrierLayout::renderTarget;
            plan.steps[1] = EnhancedTextureAliasStep{
                .target = EnhancedTextureAliasTarget::after,
                .syncBefore = translateEnhancedSync(activation.stagesBefore),
                .syncAfter = initializationSync,
                .accessBefore = EnhancedBarrierAccess::noAccess,
                .accessAfter = initializationAccess,
                .layoutBefore = EnhancedBarrierLayout::undefined,
                .layoutAfter = initializationLayout,
                .flags = EnhancedTextureBarrierFlags::discard,
                .discardAfterBarrier = true,
            };
            plan.steps[2] = EnhancedTextureAliasStep{
                .target = EnhancedTextureAliasTarget::after,
                .syncBefore = initializationSync,
                .syncAfter = translateEnhancedSync(activation.stagesAfter, activation.accessAfter),
                .accessBefore = initializationAccess,
                .accessAfter = translateEnhancedAccess(activation.accessAfter),
                .layoutBefore = initializationLayout,
                .layoutAfter = translateEnhancedLayout(activation.layoutAfter, queueType),
                .flags = EnhancedTextureBarrierFlags::none,
                .discardAfterBarrier = false,
            };
            plan.stepCount = 3;
            return plan;
        }

        plan.steps[1] = EnhancedTextureAliasStep{
            .target = EnhancedTextureAliasTarget::after,
            .syncBefore = translateEnhancedSync(activation.stagesBefore),
            .syncAfter = translateEnhancedSync(activation.stagesAfter, activation.accessAfter),
            .accessBefore = EnhancedBarrierAccess::noAccess,
            .accessAfter = translateEnhancedAccess(activation.accessAfter),
            .layoutBefore = EnhancedBarrierLayout::undefined,
            .layoutAfter = translateEnhancedLayout(activation.layoutAfter, queueType),
            .flags = EnhancedTextureBarrierFlags::discard,
            .discardAfterBarrier = false,
        };
        plan.stepCount = 2;
        return plan;
    }

} // namespace rachis::d3d12
