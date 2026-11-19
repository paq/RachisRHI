#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>

namespace
{

    using rachis::TextureLayout;

    class ForceLegacyBarriers final
    {
    public:
        explicit ForceLegacyBarriers(bool enabled)
            : previous_(rachis::d3d12::test::forceLegacyBarriers().exchange(enabled, std::memory_order_relaxed))
        {
        }

        ~ForceLegacyBarriers()
        {
            rachis::d3d12::test::forceLegacyBarriers().store(previous_, std::memory_order_relaxed);
        }

        ForceLegacyBarriers(const ForceLegacyBarriers&) = delete;
        ForceLegacyBarriers& operator=(const ForceLegacyBarriers&) = delete;
        ForceLegacyBarriers(ForceLegacyBarriers&&) = delete;
        ForceLegacyBarriers& operator=(ForceLegacyBarriers&&) = delete;

    private:
        bool previous_ = false;
    };

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

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
            return {rachis::PipelineStage::pixelShader | rachis::PipelineStage::computeShader,
                    rachis::Access::shaderResource};
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

    [[nodiscard]] rachis::CommandBuffer acquire(rachis::Device& device)
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        return std::move(*command);
    }

    void submitAndWait(rachis::Device& device, rachis::CommandBuffer&& command)
    {
        auto submitted = device.submit(std::move(command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
    }

} // namespace

TEST_CASE("D3D12 texture barriers must name the current layout of every subresource or undefined",
          "[size:medium][rhi][d3d12][warp][barrier][layout]")
{
    const bool legacy = GENERATE(false, true);
    CAPTURE(legacy);
    const ForceLegacyBarriers forced(legacy);
    auto created = rachis::createDevice(rachis::test::d3d12WarpDeviceDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).enhancedBarriers == !legacy);

    const rachis::TextureDesc desc{
        .width = 8,
        .height = 8,
        .mipLevels = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "D3D12 layoutBefore texture",
    };
    auto texture = device.createTexture(desc);
    auto other = device.createTexture(desc);
    REQUIRE(texture.has_value());
    REQUIRE(other.has_value());

    SECTION("a barrier inside one command buffer must start where the command buffer left the subresource")
    {
        auto command = acquire(device);
        REQUIRE(device
                    .textureBarrier(command,
                                    layoutBarrier(*texture, TextureLayout::undefined, TextureLayout::copyDestination))
                    .has_value());
        const auto wrong = device.textureBarrier(
            command, layoutBarrier(*texture, TextureLayout::copySource, TextureLayout::shaderResource));
        REQUIRE_FALSE(wrong.has_value());
        CHECK(wrong.error().code == rachis::ErrorCode::invalidState);
        CHECK(wrong.error().message.find("leaves it in copyDestination") != std::string::npos);

        REQUIRE(device
                    .textureBarrier(
                        command, layoutBarrier(*texture, TextureLayout::copyDestination, TextureLayout::shaderResource))
                    .has_value());
        const auto repeated = device.textureBarrier(
            command, layoutBarrier(*texture, TextureLayout::copyDestination, TextureLayout::shaderResource));
        REQUIRE_FALSE(repeated.has_value());
        CHECK(repeated.error().code == rachis::ErrorCode::invalidState);

        REQUIRE(
            device.textureBarrier(command, layoutBarrier(*texture, TextureLayout::undefined, TextureLayout::copySource))
                .has_value());

        REQUIRE(
            device
                .textureBarrier(command, layoutBarrier(*other, TextureLayout::undefined, TextureLayout::shaderResource))
                .has_value());
        const std::array batch{
            layoutBarrier(*texture, TextureLayout::copySource, TextureLayout::copyDestination),
            layoutBarrier(*other, TextureLayout::copySource, TextureLayout::copyDestination),
        };
        const auto batchRejected = device.resourceBarriers(command, {.textureBarriers = batch});
        REQUIRE_FALSE(batchRejected.has_value());
        CHECK(batchRejected.error().code == rachis::ErrorCode::invalidState);
        REQUIRE(device
                    .textureBarrier(command,
                                    layoutBarrier(*texture, TextureLayout::copySource, TextureLayout::shaderResource))
                    .has_value());

        REQUIRE(device
                    .textureBarrier(command,
                                    layoutBarrier(*texture,
                                                  TextureLayout::shaderResource,
                                                  TextureLayout::copyDestination,
                                                  {.firstMipLevel = 1, .mipLevelCount = 1}))
                    .has_value());
        const auto mixed = device.textureBarrier(
            command, layoutBarrier(*texture, TextureLayout::shaderResource, TextureLayout::copySource));
        REQUIRE_FALSE(mixed.has_value());
        CHECK(mixed.error().message.find("subresource 1") != std::string::npos);
        submitAndWait(device, std::move(command));
    }

    SECTION("a barrier's entry layout is checked against earlier submissions, with no partial effects")
    {
        {
            auto first = acquire(device);
            REQUIRE(device
                        .textureBarrier(first,
                                        layoutBarrier(*texture, TextureLayout::common, TextureLayout::copyDestination))
                        .has_value());
            REQUIRE(device
                        .textureBarrier(first,
                                        layoutBarrier(*other, TextureLayout::undefined, TextureLayout::shaderResource))
                        .has_value());
            submitAndWait(device, std::move(first));
        }

        std::array wrong{acquire(device)};
        REQUIRE(device
                    .textureBarrier(wrong[0],
                                    layoutBarrier(*texture, TextureLayout::copySource, TextureLayout::shaderResource))
                    .has_value());
        const auto rejected = device.submit(wrong);
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
        CHECK(rejected.error().message.find("submitted before it leaves it in copyDestination") != std::string::npos);

        std::array mixedBatch{acquire(device), acquire(device)};
        REQUIRE(device
                    .textureBarrier(mixedBatch[0],
                                    layoutBarrier(*texture, TextureLayout::copyDestination, TextureLayout::copySource))
                    .has_value());
        REQUIRE(device
                    .textureBarrier(mixedBatch[1],
                                    layoutBarrier(*other, TextureLayout::copyDestination, TextureLayout::copySource))
                    .has_value());
        REQUIRE_FALSE(device.submit(mixedBatch).has_value());

        auto check = acquire(device);
        REQUIRE(device
                    .textureBarrier(
                        check, layoutBarrier(*texture, TextureLayout::copyDestination, TextureLayout::shaderResource))
                    .has_value());
        submitAndWait(device, std::move(check));

        std::array reversed{acquire(device), acquire(device)};
        REQUIRE(device
                    .textureBarrier(reversed[0],
                                    layoutBarrier(*texture, TextureLayout::copySource, TextureLayout::unorderedAccess))
                    .has_value());
        REQUIRE(device
                    .textureBarrier(reversed[1],
                                    layoutBarrier(*texture, TextureLayout::shaderResource, TextureLayout::copySource))
                    .has_value());
        REQUIRE_FALSE(device.submit(reversed).has_value());
        std::array ordered{std::move(reversed[1]), std::move(reversed[0])};
        auto orderedPoint = device.submit(ordered);
        REQUIRE(orderedPoint.has_value());
        REQUIRE(device.wait(*orderedPoint).has_value());

        auto discard = acquire(device);
        REQUIRE(device
                    .textureBarrier(discard,
                                    layoutBarrier(*texture, TextureLayout::undefined, TextureLayout::copyDestination))
                    .has_value());
        submitAndWait(device, std::move(discard));
    }

    SECTION("generateMipmaps checks its base layout like a barrier")
    {
        auto command = acquire(device);
        REQUIRE(device
                    .textureBarrier(command,
                                    layoutBarrier(*texture, TextureLayout::undefined, TextureLayout::copyDestination))
                    .has_value());
        const auto wrongBase = device.generateMipmaps(command,
                                                      {.texture = *texture,
                                                       .baseLayout = TextureLayout::shaderResource,
                                                       .finalLayout = TextureLayout::shaderResource});
        REQUIRE_FALSE(wrongBase.has_value());
        CHECK(wrongBase.error().code == rachis::ErrorCode::invalidState);
        REQUIRE(device
                    .generateMipmaps(command,
                                     {.texture = *texture,
                                      .baseLayout = TextureLayout::copyDestination,
                                      .finalLayout = TextureLayout::shaderResource})
                    .has_value());
        REQUIRE(device
                    .textureBarrier(command,
                                    layoutBarrier(*texture, TextureLayout::shaderResource, TextureLayout::copySource))
                    .has_value());
        submitAndWait(device, std::move(command));
    }

    if (!legacy)
    {
        const auto createShared = [&](const char* label) {
            auto shared = device.createSharedTexture({
                .width = 8,
                .height = 8,
                .mipLevels = 2,
                .format = rachis::Format::rgba8Unorm,
                .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                         rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
                .label = label,
            });
            REQUIRE(shared.has_value());
            return *shared;
        };

        SECTION("a shared texture's entry layout is unknown until a submitted barrier establishes it")
        {
            const auto shared = createShared("D3D12 shared layoutBefore texture");
            auto entry = acquire(device);
            REQUIRE(device
                        .textureBarrier(
                            entry, layoutBarrier(shared, TextureLayout::copyDestination, TextureLayout::shaderResource))
                        .has_value());
            submitAndWait(device, std::move(entry));

            std::array inside{acquire(device)};
            REQUIRE(device
                        .textureBarrier(
                            inside[0], layoutBarrier(shared, TextureLayout::copySource, TextureLayout::copyDestination))
                        .has_value());
            const auto wrong = device.textureBarrier(
                inside[0], layoutBarrier(shared, TextureLayout::shaderResource, TextureLayout::copySource));
            REQUIRE_FALSE(wrong.has_value());
            CHECK(wrong.error().code == rachis::ErrorCode::invalidState);
            const auto rejected = device.submit(inside);
            REQUIRE_FALSE(rejected.has_value());
            CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
            CHECK(rejected.error().message.find("leaves it in shaderResource") != std::string::npos);

            auto known = acquire(device);
            REQUIRE(device
                        .textureBarrier(known,
                                        layoutBarrier(shared, TextureLayout::shaderResource, TextureLayout::copySource))
                        .has_value());
            submitAndWait(device, std::move(known));
        }

        SECTION("a shared texture keeps the entry layout its first submitted barrier established, per subresource")
        {
            const auto shared = createShared("D3D12 shared established texture");
            auto entry = acquire(device);
            REQUIRE(device
                        .textureBarrier(entry,
                                        layoutBarrier(shared,
                                                      TextureLayout::undefined,
                                                      TextureLayout::common,
                                                      {.firstMipLevel = 0, .mipLevelCount = 1}))
                        .has_value());
            submitAndWait(device, std::move(entry));

            std::array wrongEntry{acquire(device)};
            REQUIRE(device
                        .textureBarrier(wrongEntry[0],
                                        layoutBarrier(shared,
                                                      TextureLayout::shaderResource,
                                                      TextureLayout::copySource,
                                                      {.firstMipLevel = 0, .mipLevelCount = 1}))
                        .has_value());
            const auto rejected = device.submit(wrongEntry);
            REQUIRE_FALSE(rejected.has_value());
            CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
            CHECK(rejected.error().message.find("leaves it in common") != std::string::npos);

            auto untouched = acquire(device);
            REQUIRE(device
                        .textureBarrier(untouched,
                                        layoutBarrier(shared,
                                                      TextureLayout::shaderResource,
                                                      TextureLayout::copySource,
                                                      {.firstMipLevel = 1, .mipLevelCount = 1}))
                        .has_value());
            submitAndWait(device, std::move(untouched));
        }

        SECTION("a storage clear checks a shared texture's entry layout once a submitted barrier set it")
        {
            const auto shared = createShared("D3D12 shared cleared texture");
            auto view = device.createTextureView({
                .texture = shared,
                .usage = rachis::TextureViewUsage::storage,
                .dimension = rachis::TextureViewDimension::texture2D,
                .mipLevelCount = 1,
            });
            REQUIRE(view.has_value());
            const rachis::StorageTextureClearValue value{
                .type = rachis::StorageTextureClearType::float32,
                .float32 = {0.25F, 0.5F, 0.75F, 1.0F},
            };

            auto established = acquire(device);
            REQUIRE(device
                        .textureBarrier(established,
                                        layoutBarrier(shared,
                                                      TextureLayout::undefined,
                                                      TextureLayout::copySource,
                                                      {.firstMipLevel = 0, .mipLevelCount = 1}))
                        .has_value());
            submitAndWait(device, std::move(established));

            std::array wrong{acquire(device)};
            REQUIRE(device.clearStorageTexture(wrong[0], *view, value).has_value());
            const auto rejected = device.submit(wrong);
            REQUIRE_FALSE(rejected.has_value());
            CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
            CHECK(rejected.error().message.find("leaves it in copySource") != std::string::npos);
            REQUIRE(device.destroyTextureView(*view).has_value());
        }
    }

    const auto counts = device.diagnosticCounts();
    CHECK(counts.corruptions == 0);
}

TEST_CASE("D3D12 a discard recorded after a storage clear starts from the clear's layout",
          "[size:medium][rhi][d3d12][warp][barrier][layout][clear]")
{
    const bool legacy = GENERATE(false, true);
    CAPTURE(legacy);
    const ForceLegacyBarriers forced(legacy);
    auto desc = rachis::test::d3d12WarpDeviceDesc();
    desc.diagnosticCallback = &printDiagnostic;
    auto created = rachis::createDevice(desc);
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).enhancedBarriers == !legacy);

    auto texture = device.createTexture({
        .width = 4,
        .height = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage =
            rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination | rachis::TextureUsage::storage,
        .label = "D3D12 clear-then-discard texture",
    });
    REQUIRE(texture.has_value());
    auto view = device.createTextureView({
        .texture = *texture,
        .usage = rachis::TextureViewUsage::storage,
        .dimension = rachis::TextureViewDimension::texture2D,
    });
    REQUIRE(view.has_value());

    auto clearing = acquire(device);
    REQUIRE(device
                .clearStorageTexture(clearing,
                                     *view,
                                     {
                                         .type = rachis::StorageTextureClearType::float32,
                                         .float32 = {0.25F, 0.5F, 0.75F, 1.0F},
                                     })
                .has_value());
    auto discard = layoutBarrier(*texture, TextureLayout::undefined, TextureLayout::copyDestination);
    discard.stagesBefore = rachis::PipelineStage::all;
    REQUIRE(device.textureBarrier(clearing, discard).has_value());
    REQUIRE(device
                .textureBarrier(clearing,
                                layoutBarrier(*texture, TextureLayout::copyDestination, TextureLayout::copySource))
                .has_value());

    auto entering = acquire(device);
    REQUIRE(
        device.textureBarrier(entering, layoutBarrier(*texture, TextureLayout::common, TextureLayout::unorderedAccess))
            .has_value());
    submitAndWait(device, std::move(entering));
    submitAndWait(device, std::move(clearing));

    auto leaving = acquire(device);
    REQUIRE(
        device
            .textureBarrier(leaving, layoutBarrier(*texture, TextureLayout::copySource, TextureLayout::copyDestination))
            .has_value());
    submitAndWait(device, std::move(leaving));

    REQUIRE(device.destroyTextureView(*view).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    const auto counts = device.diagnosticCounts();
    CHECK(counts.errors == 0);
    CHECK(counts.corruptions == 0);
}
