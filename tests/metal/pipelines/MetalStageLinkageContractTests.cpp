#include "Internal/MetalShaderManifest.h"
#include "Internal/StageLinkage.h"
#include "Rachis/Device.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#if defined(RACHIS_METALLIB_DIR)
namespace
{

    using rachis::detail::StageComponentType;
    using rachis::detail::StageInterfaceVariable;

    [[nodiscard]] std::vector<std::byte> loadPnms(std::string_view name)
    {
        const auto path =
            std::filesystem::path(RACHIS_METALLIB_DIR) / ("MetalStageLinkage." + std::string(name) + ".pnms");
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        REQUIRE(stream);
        const auto end = stream.tellg();
        REQUIRE(end > 0);
        std::vector<std::byte> bytes(static_cast<std::size_t>(end));
        stream.seekg(0, std::ios::beg);
        stream.read(reinterpret_cast<char*>(bytes.data()), end);
        REQUIRE(stream);
        return bytes;
    }

    [[nodiscard]] rachis::metal::MetalReflectedStageInterface reflectStageInterface(std::span<const std::byte> pnms,
                                                                                    std::string_view entryPoint,
                                                                                    rachis::ShaderStage stage)
    {
        const auto payload = rachis::metal::unpackShaderManifest(pnms);
        REQUIRE(payload);
        const std::string_view json(reinterpret_cast<const char*>(payload->json.data()), payload->json.size());
        auto layout = rachis::metal::parseSlangReflectionJson(json, entryPoint, stage);
        REQUIRE(layout);
        REQUIRE(layout->stageInterface.has_value());
        return *layout->stageInterface;
    }

    enum class Outcome : std::uint8_t
    {
        linked,
        ruleRejects,
        metalRejects,
    };

    struct PixelCase final
    {
        std::string_view artifact;
        std::string_view entryPoint;
        Outcome outcome = Outcome::linked;
        std::string_view message;
    };

    constexpr std::array pixelCases{
        PixelCase{.artifact = "matching.ps", .entryPoint = "matchingPixel", .outcome = Outcome::linked},
        PixelCase{.artifact = "fewer.ps", .entryPoint = "fewerComponentsPixel", .outcome = Outcome::metalRejects},
        PixelCase{.artifact = "renamed.ps", .entryPoint = "renamedPixel", .outcome = Outcome::metalRejects},
        PixelCase{
            .artifact = "more.ps",
            .entryPoint = "moreComponentsPixel",
            .outcome = Outcome::ruleRejects,
            .message = "The pixel shader reads location 0 component 3, which the vertex shader does not write.",
        },
        PixelCase{
            .artifact = "type.ps",
            .entryPoint = "typeMismatchPixel",
            .outcome = Outcome::ruleRejects,
            .message = "The pixel shader reads location 1 component 0 as sint32, but the vertex shader writes it "
                       "as uint32.",
        },
        PixelCase{
            .artifact = "missing.ps",
            .entryPoint = "missingLocationPixel",
            .outcome = Outcome::ruleRejects,
            .message = "The pixel shader reads location 2 component 0, which the vertex shader does not write.",
        },
    };

} // namespace

TEST_CASE("Metal PNMS manifests record user stage inputs and outputs for the inter-stage rule",
          "[size:small][rhi][metal][shader][linkage]")
{
    const auto vertexPnms = loadPnms("vs");
    const auto vertex = reflectStageInterface(vertexPnms, "vertexMain", rachis::ShaderStage::vertex);
    CHECK(vertex.inputs.empty());
    const std::vector<StageInterfaceVariable> expectedOutputs{
        {.location = 0, .firstComponent = 0, .componentCount = 3, .componentType = StageComponentType::float32},
        {.location = 1, .firstComponent = 0, .componentCount = 2, .componentType = StageComponentType::uint32},
    };
    CHECK(vertex.outputs == expectedOutputs);

    for (const auto& pixelCase : pixelCases)
    {
        CAPTURE(pixelCase.entryPoint);
        const auto pixelPnms = loadPnms(pixelCase.artifact);
        const auto pixel = reflectStageInterface(pixelPnms, pixelCase.entryPoint, rachis::ShaderStage::pixel);
        CHECK(pixel.outputs.empty());
        const auto linked =
            rachis::detail::validateStageLinkage(rachis::ShaderStage::vertex, vertex.outputs, pixel.inputs);
        CHECK(linked.has_value() == (pixelCase.outcome != Outcome::ruleRejects));
        if (pixelCase.outcome == Outcome::ruleRejects && !linked)
        {
            CHECK(linked.error().code == rachis::ErrorCode::validationError);
            CHECK(linked.error().message == pixelCase.message);
        }
    }
}

TEST_CASE("Metal graphics pipelines apply the inter-stage rule at every validation level",
          "[size:large][rhi][metal][pipeline][linkage]")
{
    for (const auto validation : {rachis::ValidationLevel::standard, rachis::ValidationLevel::disabled})
    {
        CAPTURE(static_cast<int>(validation));
        auto created = rachis::createDevice({
            .backend = rachis::Backend::metal,
            .validation = validation,
            .requireValidation = false,
            .enableDred = false,
        });
        REQUIRE(created);
        auto device = std::move(*created);

        const auto vertexPnms = loadPnms("vs");
        auto vertex = device.createShaderModule({
            .stage = rachis::ShaderStage::vertex,
            .bytecode = vertexPnms,
            .entryPoint = "vertexMain",
            .label = "Metal stage linkage vertex",
        });
        REQUIRE(vertex);
        auto layout = device.createPipelineLayout({});
        REQUIRE(layout);
        const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};

        for (const auto& pixelCase : pixelCases)
        {
            CAPTURE(pixelCase.entryPoint);
            const auto pixelPnms = loadPnms(pixelCase.artifact);
            auto pixel = device.createShaderModule({
                .stage = rachis::ShaderStage::pixel,
                .bytecode = pixelPnms,
                .entryPoint = pixelCase.entryPoint,
                .label = "Metal stage linkage pixel",
            });
            REQUIRE(pixel);
            auto pipeline = device.createGraphicsPipeline({
                .layout = *layout,
                .vertexShader = *vertex,
                .pixelShader = *pixel,
                .colorTargets = colorTargets,
                .label = "Metal stage linkage pipeline",
            });
            switch (pixelCase.outcome)
            {
            case Outcome::linked:
                if (!pipeline)
                {
                    UNSCOPED_INFO(pipeline.error().message);
                }
                REQUIRE(pipeline);
                CHECK(device.destroyPipeline(*pipeline));
                break;
            case Outcome::ruleRejects:
                REQUIRE_FALSE(pipeline);
                CHECK(pipeline.error().code == rachis::ErrorCode::validationError);
                CHECK(pipeline.error().message == pixelCase.message);
                break;
            case Outcome::metalRejects:
                REQUIRE_FALSE(pipeline);
                UNSCOPED_INFO(pipeline.error().message);
                CHECK(pipeline.error().code == rachis::ErrorCode::validationError);
                CHECK_FALSE(pipeline.error().message.starts_with("The pixel shader reads"));
                break;
            }
            CHECK(device.destroyShaderModule(*pixel));
        }
        CHECK(device.destroyShaderModule(*vertex));
    }
}
#endif
