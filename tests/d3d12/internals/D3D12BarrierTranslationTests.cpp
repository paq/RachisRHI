#include "D3D12/D3D12BarrierTranslation.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>

#ifdef _WINDOWS_
#error Portable D3D12 barrier translation unit tests must not include Windows.h.
#endif

namespace
{

    using namespace rachis;
    using namespace rachis::d3d12;

} // namespace

TEST_CASE("D3D12 legacy barrier translation keeps compute and copy states within command-list masks",
          "[size:small][rhi][d3d12][barrier]")
{
    SECTION("compute shader-resource scopes never set PIXEL_SHADER_RESOURCE")
    {
        const auto bufferState = translateLegacyState(Access::shaderResource, QueueType::compute);
        REQUIRE(bufferState == LegacyResourceState::nonPixelShaderResource);
        REQUIRE(!hasAny(bufferState, LegacyResourceState::pixelShaderResource));

        const auto textureState =
            translateLegacyTextureState(TextureLayout::shaderResource, Access::shaderResource, QueueType::compute);
        REQUIRE(textureState == LegacyResourceState::nonPixelShaderResource);
        REQUIRE(!hasAny(textureState, LegacyResourceState::pixelShaderResource));

        const auto commonShaderRead =
            translateLegacyTextureState(TextureLayout::common, Access::shaderResource, QueueType::compute);
        REQUIRE(commonShaderRead == LegacyResourceState::nonPixelShaderResource);
        REQUIRE(!hasAny(commonShaderRead, LegacyResourceState::pixelShaderResource));

        REQUIRE(translateLegacyState(Access::constantBuffer, QueueType::compute) ==
                LegacyResourceState::vertexAndConstantBuffer);
        REQUIRE(translateLegacyState(Access::unorderedAccess, QueueType::compute) ==
                LegacyResourceState::unorderedAccess);
    }

    SECTION("graphics shader-resource scopes map to both shader-resource states")
    {
        constexpr auto allShaderResources =
            LegacyResourceState::pixelShaderResource | LegacyResourceState::nonPixelShaderResource;
        REQUIRE(translateLegacyState(Access::shaderResource, QueueType::graphics) == allShaderResources);
        REQUIRE(translateLegacyTextureState(
                    TextureLayout::shaderResource, Access::shaderResource, QueueType::graphics) == allShaderResources);
        REQUIRE(translateLegacyTextureState(TextureLayout::common, Access::shaderResource, QueueType::graphics) ==
                allShaderResources);
        REQUIRE(translateLegacyState(Access::shaderResource | Access::constantBuffer, QueueType::graphics) ==
                (allShaderResources | LegacyResourceState::vertexAndConstantBuffer));
        REQUIRE(translateLegacyState(Access::shaderResource, QueueType::copy) == LegacyResourceState::common);
    }

    SECTION("every valid compute and copy scope translates inside its queue state mask")
    {
        struct ScopeSample final
        {
            PipelineStage stages = PipelineStage::none;
            Access access = Access::none;
        };
        constexpr std::array samples{
            ScopeSample{PipelineStage::none, Access::none},
            ScopeSample{PipelineStage::all, Access::common},
            ScopeSample{PipelineStage::all, Access::constantBuffer},
            ScopeSample{PipelineStage::all, Access::shaderResource},
            ScopeSample{PipelineStage::all, Access::unorderedAccess},
            ScopeSample{PipelineStage::all, Access::copySource},
            ScopeSample{PipelineStage::all, Access::copyDestination},
            ScopeSample{PipelineStage::all, Access::indirectArgument},
            ScopeSample{PipelineStage::all, Access::predicate},
            ScopeSample{PipelineStage::all, Access::accelerationStructureBuildInput},
            ScopeSample{PipelineStage::all, Access::accelerationStructureScratch},
            ScopeSample{PipelineStage::computeShader, Access::constantBuffer | Access::shaderResource},
            ScopeSample{PipelineStage::copy, Access::copySource},
            ScopeSample{PipelineStage::copy, Access::copyDestination},
        };
        constexpr std::array layouts{
            TextureLayout::undefined,
            TextureLayout::common,
            TextureLayout::shaderResource,
            TextureLayout::unorderedAccess,
            TextureLayout::copySource,
            TextureLayout::copyDestination,
        };

        for (const auto queue : {QueueType::compute, QueueType::copy})
        {
            const auto allowedMask = validLegacyStatesForQueue(queue);
            for (const auto& sample : samples)
            {
                if (!validResourceBarrierScope(sample.stages, sample.access, queue))
                {
                    continue;
                }
                const auto state = translateLegacyState(sample.access, queue);
                REQUIRE((state & ~allowedMask) == LegacyResourceState::common);

                for (const auto layout : layouts)
                {
                    if (!supportsTextureLayoutOnQueue(layout, queue) ||
                        !accessMatchesTextureLayout(layout, sample.access))
                    {
                        continue;
                    }
                    const auto texState = translateLegacyTextureState(layout, sample.access, queue);
                    REQUIRE((texState & ~allowedMask) == LegacyResourceState::common);
                }
            }
        }

        REQUIRE_FALSE(
            validResourceBarrierScope(PipelineStage::pixelShader, Access::shaderResource, QueueType::compute));
        REQUIRE_FALSE(validResourceBarrierScope(PipelineStage::draw, Access::shaderResource, QueueType::compute));
        REQUIRE_FALSE(validResourceBarrierScope(PipelineStage::all, Access::vertexBuffer, QueueType::compute));
        REQUIRE_FALSE(validResourceBarrierScope(PipelineStage::computeShader, Access::shaderResource, QueueType::copy));
        REQUIRE_FALSE(validResourceBarrierScope(PipelineStage::all, Access::shaderResource, QueueType::copy));
    }
}

TEST_CASE("D3D12 legacy access checks reject exactly the accesses whose state combines a write with another state",
          "[size:small][rhi][d3d12][barrier]")
{
    constexpr auto writeStates = LegacyResourceState::renderTarget | LegacyResourceState::unorderedAccess |
                                 LegacyResourceState::depthWrite | LegacyResourceState::streamOut |
                                 LegacyResourceState::copyDest | LegacyResourceState::resolveDest;
    const auto invalidCombination = [&](LegacyResourceState state) {
        const auto writes = state & writeStates;
        return writes != LegacyResourceState::common && writes != state;
    };

    REQUIRE(combinesWriteWithOtherAccess(Access::copySource | Access::copyDestination));
    REQUIRE(combinesWriteWithOtherAccess(Access::shaderResource | Access::unorderedAccess));
    REQUIRE(combinesWriteWithOtherAccess(Access::depthStencilRead | Access::depthStencilWrite));
    REQUIRE_FALSE(combinesWriteWithOtherAccess(Access::shaderResource | Access::copySource));
    REQUIRE_FALSE(combinesWriteWithOtherAccess(Access::copyDestination));
    REQUIRE_FALSE(combinesWriteWithOtherAccess(Access::none));
    REQUIRE_FALSE(legacyStateRepresentsTextureAccess(
        TextureLayout::common, Access::copySource | Access::copyDestination, QueueType::graphics));
    REQUIRE(legacyStateRepresentsTextureAccess(
        TextureLayout::common, Access::copySource | Access::copyDestination, QueueType::copy));
    REQUIRE(legacyStateRepresentsTextureAccess(
        TextureLayout::depthStencilWrite, Access::depthStencilRead | Access::depthStencilWrite, QueueType::graphics));

    constexpr std::array accesses{
        Access::vertexBuffer,
        Access::constantBuffer,
        Access::indexBuffer,
        Access::shaderResource,
        Access::unorderedAccess,
        Access::copySource,
        Access::copyDestination,
        Access::indirectArgument,
        Access::renderTarget,
        Access::depthStencilRead,
        Access::depthStencilWrite,
        Access::resolveSource,
        Access::resolveDestination,
        Access::predicate,
        Access::shadingRateSource,
        Access::accelerationStructureBuildInput,
        Access::accelerationStructureScratch,
        Access::streamOutputWrite,
    };
    constexpr std::array layouts{
        TextureLayout::undefined,
        TextureLayout::common,
        TextureLayout::shaderResource,
        TextureLayout::unorderedAccess,
        TextureLayout::renderTarget,
        TextureLayout::depthStencilWrite,
        TextureLayout::depthStencilRead,
        TextureLayout::copySource,
        TextureLayout::copyDestination,
        TextureLayout::resolveSource,
        TextureLayout::resolveDestination,
        TextureLayout::shadingRateSource,
    };
    for (const auto queue : {QueueType::graphics, QueueType::compute, QueueType::copy})
    {
        for (const auto first : accesses)
        {
            for (const auto second : accesses)
            {
                const auto access = first | second;
                if (!validResourceBarrierScope(PipelineStage::all, access, queue))
                {
                    continue;
                }
                CAPTURE(queue, static_cast<std::uint32_t>(access));
                CHECK(combinesWriteWithOtherAccess(access) == invalidCombination(translateLegacyState(access, queue)));
                for (const auto layout : layouts)
                {
                    if (!supportsTextureLayoutOnQueue(layout, queue) || !accessMatchesTextureLayout(layout, access))
                    {
                        continue;
                    }
                    CAPTURE(layout);
                    CHECK(legacyStateRepresentsTextureAccess(layout, access, queue) ==
                          !invalidCombination(translateLegacyTextureState(layout, access, queue)));
                }
            }
        }
    }
}

TEST_CASE("D3D12 texture barrier access checks allow only no access in the undefined layout",
          "[size:small][rhi][d3d12][barrier]")
{
    constexpr std::array accesses{
        Access::common,
        Access::shaderResource,
        Access::unorderedAccess,
        Access::renderTarget,
        Access::depthStencilRead,
        Access::depthStencilWrite,
        Access::copySource,
        Access::copyDestination,
        Access::resolveSource,
        Access::resolveDestination,
        Access::shadingRateSource,
        Access::shaderResource | Access::copySource,
    };

    CHECK(accessMatchesTextureLayout(TextureLayout::undefined, Access::none));
    for (const auto access : accesses)
    {
        CAPTURE(static_cast<std::uint32_t>(access));
        CHECK_FALSE(accessMatchesTextureLayout(TextureLayout::undefined, access));
        CHECK_FALSE(accessMatchesTextureLayout(TextureLayout::present, access));
    }
    CHECK(accessMatchesTextureLayout(TextureLayout::common, Access::copyDestination));
    CHECK(accessMatchesTextureLayout(TextureLayout::renderTarget, Access::renderTarget));
}

TEST_CASE("D3D12 legacy buffer barrier planner elides UPLOAD and READBACK heap transitions",
          "[size:small][rhi][d3d12][barrier]")
{
    for (const auto queue : {QueueType::graphics, QueueType::compute, QueueType::copy})
    {
        const auto uploadPlan = planLegacyBufferBarrier(MemoryType::upload, Access::none, Access::copySource, queue);
        REQUIRE(uploadPlan.kind == LegacyBufferBarrierKind::none);

        const auto readbackPlan =
            planLegacyBufferBarrier(MemoryType::readback, Access::none, Access::copyDestination, queue);
        REQUIRE(readbackPlan.kind == LegacyBufferBarrierKind::none);
    }

    const auto uploadShaderReadPlan =
        planLegacyBufferBarrier(MemoryType::upload, Access::copySource, Access::constantBuffer, QueueType::graphics);
    REQUIRE(uploadShaderReadPlan.kind == LegacyBufferBarrierKind::none);

    const auto deviceLocalTransition = planLegacyBufferBarrier(
        MemoryType::deviceLocal, Access::copyDestination, Access::shaderResource, QueueType::compute);
    REQUIRE(deviceLocalTransition.kind == LegacyBufferBarrierKind::transition);
    REQUIRE(deviceLocalTransition.stateBefore == LegacyResourceState::copyDest);
    REQUIRE(deviceLocalTransition.stateAfter == LegacyResourceState::nonPixelShaderResource);
    REQUIRE(!hasAny(deviceLocalTransition.stateAfter, LegacyResourceState::pixelShaderResource));

    const auto graphicsShaderReadToCopy = planLegacyBufferBarrier(
        MemoryType::deviceLocal, Access::shaderResource, Access::copySource, QueueType::graphics);
    REQUIRE(graphicsShaderReadToCopy.kind == LegacyBufferBarrierKind::transition);
    REQUIRE(graphicsShaderReadToCopy.stateBefore ==
            (LegacyResourceState::pixelShaderResource | LegacyResourceState::nonPixelShaderResource));
    REQUIRE(graphicsShaderReadToCopy.stateAfter == LegacyResourceState::copySource);

    const auto graphicsShaderReadToShaderRead = planLegacyBufferBarrier(
        MemoryType::deviceLocal, Access::shaderResource, Access::shaderResource, QueueType::graphics);
    REQUIRE(graphicsShaderReadToShaderRead.kind == LegacyBufferBarrierKind::none);

    const auto deviceLocalUav = planLegacyBufferBarrier(
        MemoryType::deviceLocal, Access::unorderedAccess, Access::unorderedAccess, QueueType::compute);
    REQUIRE(deviceLocalUav.kind == LegacyBufferBarrierKind::uav);
    REQUIRE(deviceLocalUav.stateBefore == LegacyResourceState::unorderedAccess);
    REQUIRE(deviceLocalUav.stateAfter == LegacyResourceState::unorderedAccess);
}

TEST_CASE("D3D12 legacy buffer barrier planner keeps indirect arguments readable as copy sources",
          "[size:small][rhi][d3d12][barrier]")
{
    for (const auto queue : {QueueType::graphics, QueueType::compute})
    {
        CAPTURE(queue);
        const auto shaderRead =
            queue == QueueType::graphics
                ? (LegacyResourceState::pixelShaderResource | LegacyResourceState::nonPixelShaderResource)
                : LegacyResourceState::nonPixelShaderResource;
        const auto produced =
            planLegacyBufferBarrier(MemoryType::deviceLocal, Access::unorderedAccess, Access::indirectArgument, queue);
        REQUIRE(produced.kind == LegacyBufferBarrierKind::transition);
        REQUIRE(produced.stateBefore == LegacyResourceState::unorderedAccess);
        REQUIRE(produced.stateAfter == (LegacyResourceState::indirectArgument | LegacyResourceState::copySource));

        const auto combined = planLegacyBufferBarrier(
            MemoryType::deviceLocal, Access::unorderedAccess, Access::indirectArgument | Access::shaderResource, queue);
        REQUIRE(combined.kind == LegacyBufferBarrierKind::transition);
        REQUIRE(combined.stateAfter ==
                (LegacyResourceState::indirectArgument | LegacyResourceState::copySource | shaderRead));

        const auto rewritten = planLegacyBufferBarrier(
            MemoryType::deviceLocal, Access::indirectArgument | Access::shaderResource, Access::unorderedAccess, queue);
        REQUIRE(rewritten.kind == LegacyBufferBarrierKind::transition);
        REQUIRE(rewritten.stateBefore == combined.stateAfter);
        REQUIRE(rewritten.stateAfter == LegacyResourceState::unorderedAccess);

        const auto copyRead = planLegacyBufferBarrier(
            MemoryType::deviceLocal, Access::indirectArgument, Access::indirectArgument | Access::copySource, queue);
        REQUIRE(copyRead.kind == LegacyBufferBarrierKind::none);

        REQUIRE(translateLegacyBufferState(Access::shaderResource, queue) == shaderRead);
    }

    REQUIRE(translateLegacyBufferState(Access::indirectArgument, QueueType::copy) == LegacyResourceState::common);
}

TEST_CASE("D3D12 enhanced barriers sync pre-rasterization stages with vertex shading and never with compute",
          "[size:small][rhi][d3d12][barrier]")
{
    constexpr std::array preRasterizationStages{
        PipelineStage::vertexShader,
        PipelineStage::hullShader,
        PipelineStage::domainShader,
        PipelineStage::geometryShader,
        PipelineStage::taskShader,
        PipelineStage::meshShader,
        PipelineStage::streamOutput,
    };
    auto allPreRasterization = PipelineStage::none;
    for (const auto stage : preRasterizationStages)
    {
        CAPTURE(static_cast<std::uint32_t>(stage));
        const auto sync = translateEnhancedSync(stage);
        REQUIRE(sync == EnhancedBarrierSync::vertexShading);
        REQUIRE_FALSE(hasAny(sync, EnhancedBarrierSync::computeShading));
        allPreRasterization = allPreRasterization | stage;
    }
    REQUIRE(translateEnhancedSync(allPreRasterization) == EnhancedBarrierSync::vertexShading);
    REQUIRE(translateEnhancedSync(PipelineStage::geometryShader | PipelineStage::computeShader) ==
            (EnhancedBarrierSync::vertexShading | EnhancedBarrierSync::computeShading));
    REQUIRE(translateEnhancedSync(PipelineStage::meshShader | PipelineStage::pixelShader) ==
            (EnhancedBarrierSync::vertexShading | EnhancedBarrierSync::pixelShading));
}

TEST_CASE("D3D12 enhanced barriers sync compute-stage unordered access with unordered-access-view clears",
          "[size:small][rhi][d3d12][barrier][clear]")
{
    using rachis::Access;
    REQUIRE(translateEnhancedSync(PipelineStage::computeShader, Access::unorderedAccess) ==
            (EnhancedBarrierSync::computeShading | EnhancedBarrierSync::clearUnorderedAccessView));
    REQUIRE(translateEnhancedSync(PipelineStage::computeShader | PipelineStage::pixelShader,
                                  Access::unorderedAccess | Access::shaderResource) ==
            (EnhancedBarrierSync::computeShading | EnhancedBarrierSync::pixelShading |
             EnhancedBarrierSync::clearUnorderedAccessView));
    REQUIRE(translateEnhancedSync(PipelineStage::computeShader, Access::shaderResource) ==
            EnhancedBarrierSync::computeShading);
    REQUIRE(translateEnhancedSync(PipelineStage::computeShader, Access::none) == EnhancedBarrierSync::computeShading);
    REQUIRE(translateEnhancedSync(PipelineStage::computeShader) == EnhancedBarrierSync::computeShading);
    REQUIRE(translateEnhancedSync(PipelineStage::pixelShader, Access::unorderedAccess) ==
            EnhancedBarrierSync::pixelShading);
    REQUIRE(translateEnhancedSync(PipelineStage::copy, Access::copyDestination) == EnhancedBarrierSync::copy);
    REQUIRE(translateEnhancedSync(PipelineStage::all, Access::unorderedAccess) == EnhancedBarrierSync::all);
}

TEST_CASE("D3D12 enhanced barriers map each texture layout to one native layout on every queue that accepts it",
          "[size:small][rhi][d3d12][barrier]")
{
    constexpr std::array layouts{
        TextureLayout::undefined,
        TextureLayout::common,
        TextureLayout::present,
        TextureLayout::shaderResource,
        TextureLayout::unorderedAccess,
        TextureLayout::renderTarget,
        TextureLayout::depthStencilWrite,
        TextureLayout::depthStencilRead,
        TextureLayout::copySource,
        TextureLayout::copyDestination,
        TextureLayout::resolveSource,
        TextureLayout::resolveDestination,
        TextureLayout::shadingRateSource,
    };
    constexpr std::array queues{QueueType::graphics, QueueType::compute, QueueType::copy};

    for (const auto layout : layouts)
    {
        const auto graphicsLayout = translateEnhancedLayout(layout, QueueType::graphics);
        for (const auto queue : queues)
        {
            if (!supportsTextureLayoutOnQueue(layout, queue))
            {
                continue;
            }
            CAPTURE(layout, queue);
            CHECK(translateEnhancedLayout(layout, queue) == graphicsLayout);
            if (queue == QueueType::copy)
            {
                CHECK((graphicsLayout == EnhancedBarrierLayout::undefined ||
                       graphicsLayout == EnhancedBarrierLayout::common));
            }
        }
    }

    CHECK(translateEnhancedLayout(TextureLayout::undefined, QueueType::compute) == EnhancedBarrierLayout::undefined);
    CHECK(translateEnhancedLayout(TextureLayout::common, QueueType::compute) == EnhancedBarrierLayout::common);
    CHECK(translateEnhancedLayout(TextureLayout::shaderResource, QueueType::compute) ==
          EnhancedBarrierLayout::shaderResource);
    CHECK(translateEnhancedLayout(TextureLayout::unorderedAccess, QueueType::compute) ==
          EnhancedBarrierLayout::unorderedAccess);
    CHECK(translateEnhancedLayout(TextureLayout::copySource, QueueType::graphics) == EnhancedBarrierLayout::common);
    CHECK(translateEnhancedLayout(TextureLayout::copyDestination, QueueType::compute) == EnhancedBarrierLayout::common);
    CHECK(translateEnhancedLayout(TextureLayout::present, QueueType::graphics) == EnhancedBarrierLayout::present);
    CHECK(translateEnhancedLayout(TextureLayout::renderTarget, QueueType::graphics) ==
          EnhancedBarrierLayout::renderTarget);
    CHECK(translateEnhancedLayout(TextureLayout::depthStencilWrite, QueueType::graphics) ==
          EnhancedBarrierLayout::depthStencilWrite);
    CHECK(translateEnhancedLayout(TextureLayout::depthStencilRead, QueueType::graphics) ==
          EnhancedBarrierLayout::depthStencilRead);
    CHECK(translateEnhancedLayout(TextureLayout::resolveSource, QueueType::graphics) ==
          EnhancedBarrierLayout::resolveSource);
    CHECK(translateEnhancedLayout(TextureLayout::resolveDestination, QueueType::graphics) ==
          EnhancedBarrierLayout::resolveDest);
    CHECK(translateEnhancedLayout(TextureLayout::shadingRateSource, QueueType::graphics) ==
          EnhancedBarrierLayout::shadingRateSource);
    for (const auto layout : {TextureLayout::present,
                              TextureLayout::renderTarget,
                              TextureLayout::depthStencilWrite,
                              TextureLayout::depthStencilRead,
                              TextureLayout::resolveSource,
                              TextureLayout::resolveDestination,
                              TextureLayout::shadingRateSource})
    {
        CAPTURE(layout);
        CHECK_FALSE(supportsTextureLayoutOnQueue(layout, QueueType::compute));
        CHECK_FALSE(supportsTextureLayoutOnQueue(layout, QueueType::copy));
    }
}

TEST_CASE("D3D12 enhanced texture alias planner sets DISCARD on activating transitions",
          "[size:small][rhi][d3d12][barrier]")
{
    const TextureAliasActivation copyActivation{
        .before = TextureHandle{1},
        .after = TextureHandle{2},
        .stagesBefore = PipelineStage::copy,
        .stagesAfter = PipelineStage::copy,
        .accessBefore = Access::copySource,
        .accessAfter = Access::copyDestination,
        .layoutBefore = TextureLayout::copySource,
        .layoutAfter = TextureLayout::copyDestination,
    };

    SECTION("non-attachment activation sets DISCARD across graphics, compute, and copy queues")
    {
        constexpr auto usage = TextureUsage::copySource | TextureUsage::copyDestination | TextureUsage::sampled;

        const auto graphicsPlan = planEnhancedTextureAliasActivation(copyActivation, usage, QueueType::graphics);
        REQUIRE(graphicsPlan.stepCount == 2);
        REQUIRE(graphicsPlan.steps[0].target == EnhancedTextureAliasTarget::before);
        REQUIRE(graphicsPlan.steps[0].layoutBefore == EnhancedBarrierLayout::common);
        REQUIRE(graphicsPlan.steps[0].layoutAfter == EnhancedBarrierLayout::undefined);
        REQUIRE(graphicsPlan.steps[0].accessAfter == EnhancedBarrierAccess::noAccess);
        REQUIRE(graphicsPlan.steps[0].flags == EnhancedTextureBarrierFlags::none);
        REQUIRE_FALSE(graphicsPlan.steps[0].discardAfterBarrier);

        REQUIRE(graphicsPlan.steps[1].target == EnhancedTextureAliasTarget::after);
        REQUIRE(graphicsPlan.steps[1].layoutBefore == EnhancedBarrierLayout::undefined);
        REQUIRE(graphicsPlan.steps[1].layoutAfter == EnhancedBarrierLayout::common);
        REQUIRE(graphicsPlan.steps[1].accessBefore == EnhancedBarrierAccess::noAccess);
        REQUIRE(graphicsPlan.steps[1].accessAfter == EnhancedBarrierAccess::copyDest);
        REQUIRE(graphicsPlan.steps[1].flags == EnhancedTextureBarrierFlags::discard);
        REQUIRE_FALSE(graphicsPlan.steps[1].discardAfterBarrier);

        const auto computePlan = planEnhancedTextureAliasActivation(copyActivation, usage, QueueType::compute);
        REQUIRE(computePlan.stepCount == 2);
        REQUIRE(computePlan.steps[0].layoutBefore == EnhancedBarrierLayout::common);
        REQUIRE(computePlan.steps[0].flags == EnhancedTextureBarrierFlags::none);
        REQUIRE(computePlan.steps[1].layoutBefore == EnhancedBarrierLayout::undefined);
        REQUIRE(computePlan.steps[1].layoutAfter == EnhancedBarrierLayout::common);
        REQUIRE(computePlan.steps[1].flags == EnhancedTextureBarrierFlags::discard);

        const auto copyQueuePlan = planEnhancedTextureAliasActivation(copyActivation, usage, QueueType::copy);
        REQUIRE(copyQueuePlan.stepCount == 2);
        REQUIRE(copyQueuePlan.steps[0].layoutBefore == EnhancedBarrierLayout::common);
        REQUIRE(copyQueuePlan.steps[0].flags == EnhancedTextureBarrierFlags::none);
        REQUIRE(copyQueuePlan.steps[1].layoutBefore == EnhancedBarrierLayout::undefined);
        REQUIRE(copyQueuePlan.steps[1].layoutAfter == EnhancedBarrierLayout::common);
        REQUIRE(copyQueuePlan.steps[1].flags == EnhancedTextureBarrierFlags::discard);
    }

    SECTION("color and depth-stencil attachment activations set DISCARD on metadata initialization")
    {
        const TextureAliasActivation colorActivation{
            .before = TextureHandle{1},
            .after = TextureHandle{2},
            .stagesBefore = PipelineStage::copy,
            .stagesAfter = PipelineStage::renderTarget,
            .accessBefore = Access::copySource,
            .accessAfter = Access::renderTarget,
            .layoutBefore = TextureLayout::copySource,
            .layoutAfter = TextureLayout::renderTarget,
        };
        const auto colorPlan = planEnhancedTextureAliasActivation(
            colorActivation, TextureUsage::renderTarget | TextureUsage::copySource, QueueType::graphics);
        REQUIRE(colorPlan.stepCount == 3);
        REQUIRE(colorPlan.steps[0].target == EnhancedTextureAliasTarget::before);
        REQUIRE(colorPlan.steps[0].flags == EnhancedTextureBarrierFlags::none);
        REQUIRE_FALSE(colorPlan.steps[0].discardAfterBarrier);

        REQUIRE(colorPlan.steps[1].target == EnhancedTextureAliasTarget::after);
        REQUIRE(colorPlan.steps[1].layoutBefore == EnhancedBarrierLayout::undefined);
        REQUIRE(colorPlan.steps[1].layoutAfter == EnhancedBarrierLayout::renderTarget);
        REQUIRE(colorPlan.steps[1].syncAfter == EnhancedBarrierSync::renderTarget);
        REQUIRE(colorPlan.steps[1].accessAfter == EnhancedBarrierAccess::renderTarget);
        REQUIRE(colorPlan.steps[1].flags == EnhancedTextureBarrierFlags::discard);
        REQUIRE(colorPlan.steps[1].discardAfterBarrier);

        REQUIRE(colorPlan.steps[2].target == EnhancedTextureAliasTarget::after);
        REQUIRE(colorPlan.steps[2].layoutBefore == EnhancedBarrierLayout::renderTarget);
        REQUIRE(colorPlan.steps[2].layoutAfter == EnhancedBarrierLayout::renderTarget);
        REQUIRE(colorPlan.steps[2].flags == EnhancedTextureBarrierFlags::none);
        REQUIRE_FALSE(colorPlan.steps[2].discardAfterBarrier);

        const TextureAliasActivation depthCopyActivation{
            .before = TextureHandle{1},
            .after = TextureHandle{2},
            .stagesBefore = PipelineStage::copy,
            .stagesAfter = PipelineStage::copy,
            .accessBefore = Access::copySource,
            .accessAfter = Access::copyDestination,
            .layoutBefore = TextureLayout::copySource,
            .layoutAfter = TextureLayout::copyDestination,
        };
        const auto depthPlan = planEnhancedTextureAliasActivation(
            depthCopyActivation, TextureUsage::depthStencil | TextureUsage::copyDestination, QueueType::graphics);
        REQUIRE(depthPlan.stepCount == 3);
        REQUIRE(depthPlan.steps[1].layoutBefore == EnhancedBarrierLayout::undefined);
        REQUIRE(depthPlan.steps[1].layoutAfter == EnhancedBarrierLayout::depthStencilWrite);
        REQUIRE(depthPlan.steps[1].syncAfter == EnhancedBarrierSync::depthStencil);
        REQUIRE(depthPlan.steps[1].accessAfter == EnhancedBarrierAccess::depthStencilWrite);
        REQUIRE(depthPlan.steps[1].flags == EnhancedTextureBarrierFlags::discard);
        REQUIRE(depthPlan.steps[1].discardAfterBarrier);
        REQUIRE(depthPlan.steps[2].layoutBefore == EnhancedBarrierLayout::depthStencilWrite);
        REQUIRE(depthPlan.steps[2].layoutAfter == EnhancedBarrierLayout::common);
        REQUIRE(depthPlan.steps[2].flags == EnhancedTextureBarrierFlags::none);
    }
}

TEST_CASE("D3D12 enhanced alias activations order a clear of the old resource before a clear of the new one",
          "[size:small][rhi][d3d12][barrier]")
{
    constexpr auto clearScope = EnhancedBarrierSync::computeShading | EnhancedBarrierSync::clearUnorderedAccessView;

    SECTION("buffer aliases carry the clear scope through both barriers")
    {
        const BufferAliasActivation activation{
            .before = BufferHandle{1},
            .after = BufferHandle{2},
            .stagesBefore = PipelineStage::computeShader,
            .stagesAfter = PipelineStage::computeShader,
            .accessBefore = Access::unorderedAccess,
            .accessAfter = Access::unorderedAccess,
        };
        const auto plan = planEnhancedBufferAliasActivation(activation);
        CHECK(plan.deactivation.syncBefore == clearScope);
        CHECK(plan.deactivation.syncAfter == clearScope);
        CHECK(plan.deactivation.accessBefore == EnhancedBarrierAccess::unorderedAccess);
        CHECK(plan.deactivation.accessAfter == EnhancedBarrierAccess::noAccess);
        CHECK(plan.activation.syncBefore == clearScope);
        CHECK(plan.activation.syncAfter == clearScope);
        CHECK(plan.activation.accessBefore == EnhancedBarrierAccess::noAccess);
        CHECK(plan.activation.accessAfter == EnhancedBarrierAccess::unorderedAccess);
    }

    SECTION("texture aliases carry the clear scope through both barriers")
    {
        const TextureAliasActivation activation{
            .before = TextureHandle{1},
            .after = TextureHandle{2},
            .stagesBefore = PipelineStage::computeShader,
            .stagesAfter = PipelineStage::computeShader,
            .accessBefore = Access::unorderedAccess,
            .accessAfter = Access::unorderedAccess,
            .layoutBefore = TextureLayout::unorderedAccess,
            .layoutAfter = TextureLayout::unorderedAccess,
        };
        const auto plan = planEnhancedTextureAliasActivation(
            activation, TextureUsage::storage | TextureUsage::copyDestination, QueueType::compute);
        REQUIRE(plan.stepCount == 2);
        CHECK(plan.steps[0].syncBefore == clearScope);
        CHECK(plan.steps[0].syncAfter == clearScope);
        CHECK(plan.steps[1].syncBefore == clearScope);
        CHECK(plan.steps[1].syncAfter == clearScope);
    }

    SECTION("an attachment's metadata initialization waits for a clear of the old resource")
    {
        const TextureAliasActivation activation{
            .before = TextureHandle{1},
            .after = TextureHandle{2},
            .stagesBefore = PipelineStage::computeShader,
            .stagesAfter = PipelineStage::renderTarget,
            .accessBefore = Access::unorderedAccess,
            .accessAfter = Access::renderTarget,
            .layoutBefore = TextureLayout::unorderedAccess,
            .layoutAfter = TextureLayout::renderTarget,
        };
        const auto plan =
            planEnhancedTextureAliasActivation(activation, TextureUsage::renderTarget, QueueType::graphics);
        REQUIRE(plan.stepCount == 3);
        CHECK(plan.steps[0].syncBefore == clearScope);
        CHECK(plan.steps[1].syncBefore == clearScope);
        CHECK(plan.steps[1].syncAfter == EnhancedBarrierSync::renderTarget);
    }
}
