#include "Rachis/Device.h"

#include "RhiTestSupport.h"
#include "StageLinkageMissing.h"
#include "StageLinkageMoreComponents.h"
#include "StageLinkageNoPosition.h"
#include "StageLinkagePixelFewer.h"
#include "StageLinkagePixelMatching.h"
#include "StageLinkageTypeMismatch.h"
#include "StageLinkageVertex.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string>
#include <string_view>

namespace
{

    struct LinkedPipeline final
    {
        rachis::Result<rachis::PipelineHandle> pipeline;
        std::uint64_t debugErrors = 0;
    };

    [[nodiscard]] LinkedPipeline createLinkedPipeline(rachis::Device& device, const rachis::ShaderModuleDesc& pixelDesc)
    {
        auto vertex = device.createShaderModule(
            rachis_shaders::reflection::StageLinkageVertex::moduleDesc(rachis::Backend::d3d12, "Linkage vertex"));
        REQUIRE(vertex.has_value());
        auto pixel = device.createShaderModule(pixelDesc);
        REQUIRE(pixel.has_value());
        auto layout = device.createPipelineLayout({});
        REQUIRE(layout.has_value());
        const auto errorsBefore = device.diagnosticCounts().errors;
        const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
        auto pipeline = device.createGraphicsPipeline({
            .layout = *layout,
            .vertexShader = *vertex,
            .pixelShader = *pixel,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
            .label = "Stage linkage",
        });
        return LinkedPipeline{std::move(pipeline), device.diagnosticCounts().errors - errorsBefore};
    }

    [[nodiscard]] bool contains(std::string_view text, std::string_view part)
    {
        return text.find(part) != std::string_view::npos;
    }

} // namespace

TEST_CASE("D3D12 graphics pipelines check the vertex outputs a pixel shader reads",
          "[size:medium][rhi][d3d12][warp][pipeline][stage-linkage]")
{
    auto created = rachis::createDevice(rachis::test::d3d12WarpDeviceDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    constexpr auto backend = rachis::Backend::d3d12;

    SECTION("an input the vertex shader writes in full is accepted")
    {
        const auto linked =
            createLinkedPipeline(device, rachis_shaders::reflection::StageLinkagePixelMatching::moduleDesc(backend));
        REQUIRE(linked.pipeline.has_value());
        CHECK(linked.debugErrors == 0);
    }

    SECTION("reading fewer components than written is accepted")
    {
        const auto linked =
            createLinkedPipeline(device, rachis_shaders::reflection::StageLinkagePixelFewer::moduleDesc(backend));
        REQUIRE(linked.pipeline.has_value());
        CHECK(linked.debugErrors == 0);
    }

    SECTION("an input at a location the vertex shader does not write is rejected")
    {
        const auto linked =
            createLinkedPipeline(device, rachis_shaders::reflection::StageLinkageMissing::moduleDesc(backend));
        REQUIRE_FALSE(linked.pipeline.has_value());
        CHECK(linked.pipeline.error().code == rachis::ErrorCode::validationError);
        CHECK(contains(linked.pipeline.error().message, "location 3 component 0"));
        CHECK(contains(linked.pipeline.error().message, "vertex shader does not write"));
    }

    SECTION("an input read with another component type is rejected")
    {
        const auto linked =
            createLinkedPipeline(device, rachis_shaders::reflection::StageLinkageTypeMismatch::moduleDesc(backend));
        REQUIRE_FALSE(linked.pipeline.has_value());
        CHECK(linked.pipeline.error().code == rachis::ErrorCode::validationError);
        CHECK(contains(linked.pipeline.error().message, "as float32, but the vertex shader writes it as uint32"));
    }

    SECTION("an input that reads more components than written is rejected")
    {
        const auto linked =
            createLinkedPipeline(device, rachis_shaders::reflection::StageLinkageMoreComponents::moduleDesc(backend));
        REQUIRE_FALSE(linked.pipeline.has_value());
        CHECK(linked.pipeline.error().code == rachis::ErrorCode::validationError);
        CHECK(contains(linked.pipeline.error().message, "location 2 component 2"));
    }

    SECTION("an input left without the position passes the rule and fails in Direct3D 12's own linker")
    {
        const auto linked =
            createLinkedPipeline(device, rachis_shaders::reflection::StageLinkageNoPosition::moduleDesc(backend));
        REQUIRE_FALSE(linked.pipeline.has_value());
        CHECK(contains(linked.pipeline.error().message, "CreateGraphicsPipelineState"));
    }
}
