#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string>
#include <utility>
#include <vector>

#if RACHIS_ENABLE_METAL
namespace
{

    using rachis::TextureLayout;

    struct LayoutScope final
    {
        rachis::PipelineStage stages = rachis::PipelineStage::none;
        rachis::Access access = rachis::Access::none;
    };

    [[nodiscard]] constexpr LayoutScope layoutScope(TextureLayout layout) noexcept
    {
        switch (layout)
        {
        case TextureLayout::copyDestination:
            return {rachis::PipelineStage::copy, rachis::Access::copyDestination};
        case TextureLayout::copySource:
            return {rachis::PipelineStage::copy, rachis::Access::copySource};
        case TextureLayout::shaderResource:
            return {rachis::PipelineStage::pixelShader, rachis::Access::shaderResource};
        case TextureLayout::unorderedAccess:
            return {rachis::PipelineStage::computeShader, rachis::Access::unorderedAccess};
        default:
            return {};
        }
    }

    [[nodiscard]] rachis::TextureBarrier layoutBarrier(rachis::TextureHandle texture,
                                                       TextureLayout before,
                                                       TextureLayout after,
                                                       rachis::TextureSubresourceRange range = {})
    {
        const auto beforeScope = layoutScope(before);
        const auto afterScope = layoutScope(after);
        return rachis::TextureBarrier{
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

    [[nodiscard]] rachis::Device createMetalDevice(rachis::ValidationLevel validation)
    {
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = validation,
            .requireValidation = false,
            .enableDred = false,
        });
        REQUIRE(created.has_value());
        return std::move(*created);
    }

    [[nodiscard]] rachis::TextureHandle createLayoutTexture(rachis::Device& device, const char* label)
    {
        auto texture = device.createTexture(rachis::TextureDesc{
            .width = 8,
            .height = 8,
            .mipLevels = 2,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                     rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
            .label = label,
        });
        REQUIRE(texture.has_value());
        return *texture;
    }

    [[nodiscard]] rachis::CommandBuffer acquire(rachis::Device& device)
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        return std::move(*command);
    }

    [[nodiscard]] rachis::TimelinePoint submitAll(rachis::Device& device, std::span<rachis::CommandBuffer> commands)
    {
        auto point = device.submit(commands);
        if (!point)
        {
            UNSCOPED_INFO(point.error().message);
        }
        REQUIRE(point.has_value());
        return *point;
    }

    void requireCleanDiagnostics(rachis::Device& device, rachis::TimelinePoint last)
    {
        REQUIRE(device.wait(last).has_value());
        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
    }

} // namespace

TEST_CASE("Metal texture barriers must start from the layout their command buffer left the subresource in",
          "[size:large][rhi][metal][barrier][layout]")
{
    for (const auto validation : {rachis::ValidationLevel::standard, rachis::ValidationLevel::disabled})
    {
        CAPTURE(static_cast<int>(validation));
        auto device = createMetalDevice(validation);
        const auto texture = createLayoutTexture(device, "Metal layout texture");
        const auto other = createLayoutTexture(device, "Metal layout other texture");

        auto command = acquire(device);
        REQUIRE(device
                    .textureBarrier(command,
                                    layoutBarrier(texture, TextureLayout::undefined, TextureLayout::copyDestination))
                    .has_value());
        const auto wrong = device.textureBarrier(
            command, layoutBarrier(texture, TextureLayout::copySource, TextureLayout::shaderResource));
        REQUIRE_FALSE(wrong.has_value());
        CHECK(wrong.error().code == rachis::ErrorCode::invalidState);
        CHECK(wrong.error().message ==
              "A texture barrier's layoutBefore (copySource) is not the current layout of color mip 0, array layer 0: "
              "this command buffer leaves it in copyDestination. Name the current layout, or undefined to discard "
              "the contents.");

        REQUIRE(device
                    .textureBarrier(
                        command, layoutBarrier(texture, TextureLayout::copyDestination, TextureLayout::shaderResource))
                    .has_value());
        const auto repeated = device.textureBarrier(
            command, layoutBarrier(texture, TextureLayout::copyDestination, TextureLayout::shaderResource));
        REQUIRE_FALSE(repeated.has_value());
        CHECK(repeated.error().code == rachis::ErrorCode::invalidState);
        REQUIRE(device
                    .textureBarrier(command,
                                    layoutBarrier(texture, TextureLayout::shaderResource, TextureLayout::copySource))
                    .has_value());

        REQUIRE(device
                    .textureBarrier(command,
                                    layoutBarrier(texture, TextureLayout::undefined, TextureLayout::unorderedAccess))
                    .has_value());

        const auto common = device.textureBarrier(
            command,
            {.texture = texture, .layoutBefore = TextureLayout::common, .layoutAfter = TextureLayout::copySource});
        REQUIRE_FALSE(common.has_value());
        CHECK(common.error().code == rachis::ErrorCode::invalidState);
        REQUIRE(device
                    .textureBarrier(command,
                                    layoutBarrier(texture, TextureLayout::unorderedAccess, TextureLayout::copySource))
                    .has_value());

        REQUIRE(
            device
                .textureBarrier(command, layoutBarrier(other, TextureLayout::undefined, TextureLayout::shaderResource))
                .has_value());
        const std::array batch{
            layoutBarrier(texture, TextureLayout::copySource, TextureLayout::copyDestination),
            layoutBarrier(other, TextureLayout::copySource, TextureLayout::copyDestination),
        };
        const auto batchRejected = device.resourceBarriers(command, {.textureBarriers = batch});
        REQUIRE_FALSE(batchRejected.has_value());
        CHECK(batchRejected.error().code == rachis::ErrorCode::invalidState);
        REQUIRE(device
                    .textureBarrier(command,
                                    layoutBarrier(texture, TextureLayout::copySource, TextureLayout::shaderResource))
                    .has_value());

        REQUIRE(device
                    .textureBarrier(command,
                                    layoutBarrier(texture,
                                                  TextureLayout::shaderResource,
                                                  TextureLayout::copyDestination,
                                                  {.firstMipLevel = 1, .mipLevelCount = 1}))
                    .has_value());
        const auto mixed = device.textureBarrier(
            command, layoutBarrier(texture, TextureLayout::shaderResource, TextureLayout::copySource));
        REQUIRE_FALSE(mixed.has_value());
        CHECK(mixed.error().message.find("mip 1") != std::string::npos);
        REQUIRE(device
                    .textureBarrier(command,
                                    layoutBarrier(texture,
                                                  TextureLayout::shaderResource,
                                                  TextureLayout::copySource,
                                                  {.firstMipLevel = 0, .mipLevelCount = 1}))
                    .has_value());

        std::array submitted{std::move(command)};
        requireCleanDiagnostics(device, submitAll(device, submitted));
    }
}

TEST_CASE("Metal submit checks each command buffer's entry layouts against earlier submissions",
          "[size:large][rhi][metal][barrier][layout]")
{
    for (const auto validation : {rachis::ValidationLevel::standard, rachis::ValidationLevel::disabled})
    {
        CAPTURE(static_cast<int>(validation));
        auto device = createMetalDevice(validation);
        const auto texture = createLayoutTexture(device, "Metal entry layout texture");
        const auto other = createLayoutTexture(device, "Metal entry layout other texture");

        {
            auto fresh = acquire(device);
            REQUIRE(device
                        .textureBarrier(fresh,
                                        layoutBarrier(texture, TextureLayout::common, TextureLayout::copyDestination))
                        .has_value());
            std::array freshBatch{std::move(fresh)};
            const auto rejected = device.submit(freshBatch);
            REQUIRE_FALSE(rejected.has_value());
            CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
        }

        auto init = acquire(device);
        REQUIRE(
            device
                .textureBarrier(init, layoutBarrier(texture, TextureLayout::undefined, TextureLayout::copyDestination))
                .has_value());
        REQUIRE(
            device.textureBarrier(init, layoutBarrier(other, TextureLayout::undefined, TextureLayout::shaderResource))
                .has_value());
        std::array initBatch{std::move(init)};
        auto last = submitAll(device, initBatch);

        auto wrong = acquire(device);
        REQUIRE(
            device
                .textureBarrier(wrong, layoutBarrier(texture, TextureLayout::copySource, TextureLayout::shaderResource))
                .has_value());
        std::array wrongBatch{std::move(wrong)};
        const auto rejected = device.submit(wrongBatch);
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
        CHECK(rejected.error().message ==
              "A texture barrier's layoutBefore (copySource) is not the current layout of color mip 0, array layer 0: "
              "the work submitted before it leaves it in copyDestination. Name the current layout, or undefined to "
              "discard the contents.");

        auto good = acquire(device);
        auto bad = acquire(device);
        REQUIRE(
            device
                .textureBarrier(good, layoutBarrier(texture, TextureLayout::copyDestination, TextureLayout::copySource))
                .has_value());
        REQUIRE(
            device.textureBarrier(bad, layoutBarrier(other, TextureLayout::copyDestination, TextureLayout::copySource))
                .has_value());
        std::array mixedBatch{std::move(good), std::move(bad)};
        REQUIRE_FALSE(device.submit(mixedBatch).has_value());

        auto check = acquire(device);
        REQUIRE(device
                    .textureBarrier(
                        check, layoutBarrier(texture, TextureLayout::copyDestination, TextureLayout::shaderResource))
                    .has_value());
        std::array checkBatch{std::move(check)};
        last = submitAll(device, checkBatch);

        auto first = acquire(device);
        auto second = acquire(device);
        REQUIRE(
            device
                .textureBarrier(first, layoutBarrier(texture, TextureLayout::shaderResource, TextureLayout::copySource))
                .has_value());
        REQUIRE(device
                    .textureBarrier(second,
                                    layoutBarrier(texture, TextureLayout::copySource, TextureLayout::unorderedAccess))
                    .has_value());
        std::array reversed{std::move(second), std::move(first)};
        REQUIRE_FALSE(device.submit(reversed).has_value());
        std::array ordered{std::move(reversed[1]), std::move(reversed[0])};
        last = submitAll(device, ordered);

        auto discard = acquire(device);
        REQUIRE(device
                    .textureBarrier(discard,
                                    layoutBarrier(texture, TextureLayout::undefined, TextureLayout::copyDestination))
                    .has_value());
        std::array discardBatch{std::move(discard)};
        last = submitAll(device, discardBatch);

        requireCleanDiagnostics(device, last);
    }
}

TEST_CASE("Metal mipmap generation requires baseLayout and leaves its range in finalLayout",
          "[size:large][rhi][metal][barrier][layout][mipmap]")
{
    auto device = createMetalDevice(rachis::ValidationLevel::standard);
    auto texture = device.createTexture(rachis::TextureDesc{
        .width = 16,
        .height = 16,
        .mipLevels = 3,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage | rachis::TextureUsage::copySource |
                 rachis::TextureUsage::copyDestination,
        .label = "Metal mipmap layout texture",
    });
    REQUIRE(texture.has_value());
    const rachis::TextureMipmapGeneration generation{
        .texture = *texture,
        .baseLayout = TextureLayout::copyDestination,
        .finalLayout = TextureLayout::shaderResource,
    };

    auto command = acquire(device);
    REQUIRE(
        device.textureBarrier(command, layoutBarrier(*texture, TextureLayout::undefined, TextureLayout::shaderResource))
            .has_value());
    const auto wrongBase = device.generateMipmaps(command, generation);
    REQUIRE_FALSE(wrongBase.has_value());
    CHECK(wrongBase.error().code == rachis::ErrorCode::invalidState);
    CHECK(wrongBase.error().message.starts_with("Mipmap generation's baseLayout (copyDestination)"));
    REQUIRE(device
                .textureBarrier(command,
                                layoutBarrier(*texture,
                                              TextureLayout::shaderResource,
                                              TextureLayout::copyDestination,
                                              {.firstMipLevel = 0, .mipLevelCount = 1}))
                .has_value());
    REQUIRE(device.generateMipmaps(command, generation).has_value());
    REQUIRE(
        device
            .textureBarrier(command, layoutBarrier(*texture, TextureLayout::shaderResource, TextureLayout::copySource))
            .has_value());
    std::array batch{std::move(command)};
    auto last = submitAll(device, batch);

    auto stale = acquire(device);
    REQUIRE(device.generateMipmaps(stale, generation).has_value());
    std::array staleBatch{std::move(stale)};
    const auto rejected = device.submit(staleBatch);
    REQUIRE_FALSE(rejected.has_value());
    CHECK(rejected.error().code == rachis::ErrorCode::invalidState);

    auto next = acquire(device);
    REQUIRE(
        device.textureBarrier(next, layoutBarrier(*texture, TextureLayout::copySource, TextureLayout::copyDestination))
            .has_value());
    REQUIRE(device.generateMipmaps(next, generation).has_value());
    std::array nextBatch{std::move(next)};
    last = submitAll(device, nextBatch);
    requireCleanDiagnostics(device, last);
}

TEST_CASE("Metal texture alias activation leaves the activated alias in layoutAfter",
          "[size:large][rhi][metal][barrier][layout][alias]")
{
    auto device = createMetalDevice(rachis::ValidationLevel::standard);
    const rachis::TextureDesc desc{
        .width = 8,
        .height = 8,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Metal aliased layout texture",
    };
    const auto requirements = device.queryTextureMemoryRequirements(desc);
    REQUIRE(requirements.has_value());
    auto heap = device.createResourceHeap({
        .size = requirements->size,
        .alignment = requirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::nonAttachmentTexture,
        .flags = rachis::ResourceHeapFlag::allowAliasing,
        .label = "Metal aliased layout heap",
    });
    REQUIRE(heap.has_value());
    const auto first = device.createPlacedTexture({.heap = *heap}, desc);
    const auto second = device.createPlacedTexture({.heap = *heap}, desc);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());

    auto command = acquire(device);
    REQUIRE(device
                .activateTextureAlias(command,
                                      {
                                          .before = *first,
                                          .after = *second,
                                          .stagesBefore = rachis::PipelineStage::all,
                                          .stagesAfter = rachis::PipelineStage::copy,
                                          .accessBefore = rachis::Access::common,
                                          .accessAfter = rachis::Access::copyDestination,
                                          .layoutBefore = TextureLayout::common,
                                          .layoutAfter = TextureLayout::copyDestination,
                                      })
                .has_value());
    const auto wrong = device.textureBarrier(
        command, layoutBarrier(*second, TextureLayout::copySource, TextureLayout::copyDestination));
    REQUIRE_FALSE(wrong.has_value());
    CHECK(wrong.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(
        device
            .textureBarrier(command, layoutBarrier(*second, TextureLayout::copyDestination, TextureLayout::copySource))
            .has_value());
    std::array batch{std::move(command)};
    auto last = submitAll(device, batch);

    auto next = acquire(device);
    REQUIRE(
        device.textureBarrier(next, layoutBarrier(*second, TextureLayout::copySource, TextureLayout::copyDestination))
            .has_value());
    std::array nextBatch{std::move(next)};
    last = submitAll(device, nextBatch);
    requireCleanDiagnostics(device, last);
}
#endif
