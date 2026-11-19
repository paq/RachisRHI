#include "Internal/MetalShaderManifest.h"
#include "Metal/MetalTestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>

namespace
{

    constexpr std::uint32_t targetSize = 64;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t readbackSize = static_cast<std::uint64_t>(rowPitch) * targetSize;

    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] std::vector<std::byte> loadBytes(const std::filesystem::path& path)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        const auto end = stream.tellg();
        if (end <= 0)
        {
            return {};
        }
        std::vector<std::byte> bytes(static_cast<std::size_t>(end));
        stream.seekg(0, std::ios::beg);
        stream.read(reinterpret_cast<char*>(bytes.data()), end);
        return stream ? bytes : std::vector<std::byte>{};
    }

    [[nodiscard]] std::vector<std::byte> loadPnms(std::string_view name)
    {
        return loadBytes(std::filesystem::path(RACHIS_METALLIB_DIR) / std::filesystem::path(name));
    }

    [[nodiscard]] std::vector<std::byte> withThreadGroupSize(std::span<const std::byte> cooked,
                                                             const rachis::ShaderWorkgroupSize& size)
    {
        const auto payload = rachis::metal::unpackShaderManifest(cooked);
        REQUIRE(payload.has_value());
        const std::string json(reinterpret_cast<const char*>(payload->json.data()), payload->json.size());
        const std::regex threadGroupSize(R"("threadGroupSize"\s*:\s*\[[^\]]*\])");
        REQUIRE(std::regex_search(json, threadGroupSize));
        const auto replaced = std::regex_replace(json,
                                                 threadGroupSize,
                                                 "\"threadGroupSize\": [" + std::to_string(size.x) + ", " +
                                                     std::to_string(size.y) + ", " + std::to_string(size.z) + "]");
        return rachis::metal::packShaderManifest(std::as_bytes(std::span(replaced)), payload->metallib);
    }

    struct MeshTarget final
    {
        rachis::TextureHandle texture{};
        rachis::TextureViewHandle view{};
        rachis::BufferHandle readback{};
    };

    [[nodiscard]] MeshTarget createMeshTarget(rachis::Device& device)
    {
        auto texture = device.createTexture({
            .width = targetSize,
            .height = targetSize,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
            .label = "Metal mesh contract target",
        });
        REQUIRE(texture.has_value());
        auto view = device.createTextureView({
            .texture = *texture,
            .usage = rachis::TextureViewUsage::renderTarget,
            .label = "Metal mesh contract target view",
        });
        REQUIRE(view.has_value());
        auto readback = device.createBuffer({
            .size = readbackSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Metal mesh contract readback",
        });
        REQUIRE(readback.has_value());
        return {*texture, *view, *readback};
    }

    void beginMeshRendering(rachis::Device& device, rachis::CommandBuffer& command, const MeshTarget& target)
    {
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = target.texture,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
        const std::array attachments{
            rachis::ColorAttachment{
                .view = target.view,
                .loadOperation = rachis::LoadOperation::clear,
                .clearValue = {0.0F, 0.0F, 0.0F, 1.0F},
            },
        };
        REQUIRE(device.beginRendering(command, {.colorAttachments = attachments}).has_value());
    }

    enum class MeshPlacement : std::uint8_t
    {
        left,
        right,
    };

    constexpr float placementScale = 0.3F;
    constexpr float placementOffset = 0.5F;
    constexpr float meshGreen = 0.25F;
    constexpr std::uint32_t sampleRow = 34;
    constexpr std::uint32_t leftColumn = 16;
    constexpr std::uint32_t middleColumn = 32;
    constexpr std::uint32_t rightColumn = 48;

    void placeMesh(rachis::Device& device, rachis::CommandBuffer& command, MeshPlacement placement)
    {
        const float offset = placement == MeshPlacement::left ? -placementOffset : placementOffset;
        const std::array meshControl{placementScale, offset, 0.0F, meshGreen};
        REQUIRE(
            device
                .setGraphicsPushConstants(command, rachis::ShaderStage::mesh, 0, std::as_bytes(std::span(meshControl)))
                .has_value());
    }

    void setMeshDrawState(rachis::Device& device, rachis::CommandBuffer& command)
    {
        REQUIRE(device
                    .setViewport(command,
                                 {
                                     .width = static_cast<float>(targetSize),
                                     .height = static_cast<float>(targetSize),
                                     .maxDepth = 1.0F,
                                 })
                    .has_value());
        REQUIRE(device.setScissor(command, {.width = targetSize, .height = targetSize}).has_value());
        placeMesh(device, command, MeshPlacement::left);
    }

    void finishMeshRendering(rachis::Device& device, rachis::CommandBuffer command, const MeshTarget& target)
    {
        REQUIRE(device.endRendering(command).has_value());
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = target.texture,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        REQUIRE(device
                    .copyTextureToBuffer(command,
                                         {
                                             .buffer = target.readback,
                                             .bytesPerRow = rowPitch,
                                             .rowsPerImage = targetSize,
                                             .texture = target.texture,
                                             .extent = {targetSize, targetSize, 1},
                                         })
                    .has_value());
        auto submitted = device.submit(std::move(command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
    }

    struct MeshCoverage final
    {
        std::uint32_t left = 0;
        std::uint32_t middle = 0;
        std::uint32_t right = 0;
        std::uint32_t corner = 0;
    };

    [[nodiscard]] std::uint32_t meshDrawCount(const std::byte* pixel) noexcept
    {
        const auto green = std::to_integer<std::uint32_t>(pixel[1]);
        const auto red = std::to_integer<std::uint32_t>(pixel[0]);
        const auto blue = std::to_integer<std::uint32_t>(pixel[2]);
        if (red != 0 || blue != 0)
        {
            return 1000;
        }
        return (green + 32U) / 64U;
    }

    [[nodiscard]] MeshCoverage readMeshCoverage(rachis::Device& device, const MeshTarget& target)
    {
        auto mapped = device.mapBuffer(target.readback);
        REQUIRE(mapped.has_value());
        const auto at = [&](std::uint32_t x, std::uint32_t y) {
            return meshDrawCount(mapped->data + static_cast<std::uint64_t>(y) * rowPitch +
                                 static_cast<std::uint64_t>(x) * 4U);
        };
        const MeshCoverage coverage{
            .left = at(leftColumn, sampleRow),
            .middle = at(middleColumn, sampleRow),
            .right = at(rightColumn, sampleRow),
            .corner = at(0, 0),
        };
        REQUIRE(device.unmapBuffer(target.readback).has_value());
        return coverage;
    }

    void checkMeshCoverage(rachis::Device& device, const MeshTarget& target, const MeshCoverage& expected)
    {
        const auto actual = readMeshCoverage(device, target);
        CHECK(actual.left == expected.left);
        CHECK(actual.middle == expected.middle);
        CHECK(actual.right == expected.right);
        CHECK(actual.corner == expected.corner);
    }

    void destroyMeshTarget(rachis::Device& device, const MeshTarget& target)
    {
        REQUIRE(device.destroyTextureView(target.view).has_value());
        REQUIRE(device.destroyTexture(target.texture).has_value());
        REQUIRE(device.destroyBuffer(target.readback).has_value());
    }

} // namespace

TEST_CASE("Metal draw barriers cover every render stage, including object and mesh shaders",
          "[size:small][rhi][metal][mesh][barrier]")
{
    const rachis::metal::MetalBarrierStageMask everyRenderStage{
        .vertex = true,
        .object = true,
        .mesh = true,
        .fragment = true,
    };
    CHECK(rachis::metal::barrierStagesForTesting(rachis::PipelineStage::draw) == everyRenderStage);
    CHECK(rachis::metal::barrierStagesForTesting(rachis::PipelineStage::taskShader) ==
          rachis::metal::MetalBarrierStageMask{.object = true});
    CHECK(rachis::metal::barrierStagesForTesting(rachis::PipelineStage::meshShader) ==
          rachis::metal::MetalBarrierStageMask{.mesh = true});
    CHECK(rachis::metal::barrierStagesForTesting(rachis::PipelineStage::vertexShader) ==
          rachis::metal::MetalBarrierStageMask{.vertex = true});
    CHECK(rachis::metal::barrierStagesForTesting(rachis::PipelineStage::draw | rachis::PipelineStage::computeShader) ==
          rachis::metal::MetalBarrierStageMask{
              .vertex = true,
              .object = true,
              .mesh = true,
              .fragment = true,
              .dispatch = true,
          });
}

TEST_CASE("Metal reports mesh and task shaders on Apple9 GPUs with Metal mesh-grid limits",
          "[size:large][rhi][metal][mesh]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    const auto family = rachis::metal::boundAppleGpuFamily();
    REQUIRE(family >= 7);
    CHECK(capabilities.meshShaders == (family >= 9));
    CHECK(capabilities.taskShaders == (family >= 9));
    if (family < 9)
    {
        SKIP("The Metal GPU is older than the Apple9 family.");
    }
    const auto& limits = capabilities.meshShader;
    CHECK(limits.maxTaskWorkgroupInvocations >= 1);
    CHECK(limits.maxMeshWorkgroupInvocations >= 1);
    CHECK(limits.maxTaskPayloadBytes <= 16384);
    CHECK(limits.maxMeshOutputVertices == 256);
    CHECK(limits.maxMeshOutputPrimitives == 512);
    CHECK(limits.maxTotalWorkgroups == 1048575);
    CHECK(limits.maxWorkgroupCount[0] == 65535);
}

TEST_CASE("Metal mesh pipelines dispatch directly and indirectly with and without a task stage",
          "[size:large][rhi][metal][mesh]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.meshShaders)
    {
        SKIP("The Metal GPU is older than the Apple9 family.");
    }

    const auto taskBytecode = loadPnms("MetalMeshShader.as.pnms");
    const auto meshBytecode = loadPnms("MetalMeshShader.ms.pnms");
    const auto meshNoTaskBytecode = loadPnms("MetalMeshShader.no-task.ms.pnms");
    const auto pixelBytecode = loadPnms("MetalMeshShader.ps.pnms");
    REQUIRE_FALSE(taskBytecode.empty());
    REQUIRE_FALSE(meshBytecode.empty());
    REQUIRE_FALSE(meshNoTaskBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());

    const std::array meshPush{rachis::ShaderPushConstantRequirement{0, 4 * sizeof(float)}};
    const std::array taskBindings{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1}};
    auto taskShader = device.createShaderModule({
        .stage = rachis::ShaderStage::task,
        .bytecode = taskBytecode,
        .entryPoint = "taskMain",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, taskBindings),
    });
    auto meshShader = device.createShaderModule({
        .stage = rachis::ShaderStage::mesh,
        .bytecode = meshBytecode,
        .entryPoint = "meshMain",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, {}, meshPush),
    });
    auto meshNoTaskShader = device.createShaderModule({
        .stage = rachis::ShaderStage::mesh,
        .bytecode = meshNoTaskBytecode,
        .entryPoint = "meshNoTask",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, {}, meshPush),
    });
    auto pixelShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "meshPixel",
        .shaderInterface = {},
    });
    REQUIRE(taskShader.has_value());
    REQUIRE(meshShader.has_value());
    REQUIRE(meshNoTaskShader.has_value());
    REQUIRE(pixelShader.has_value());
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::vertex,
                        .bytecode = meshBytecode,
                        .entryPoint = "meshMain",
                        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, {}, meshPush),
                    })
                    .has_value());

    auto constants = device.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal task-stage constants",
    });
    REQUIRE(constants.has_value());
    auto mappedConstants = device.mapBuffer(*constants);
    REQUIRE(mappedConstants.has_value());
    std::memset(mappedConstants->data, 0, mappedConstants->size);
    const std::array taskColor{0.0F, meshGreen, 0.0F, 1.0F};
    std::memcpy(mappedConstants->data, taskColor.data(), sizeof(taskColor));
    REQUIRE(device.unmapBuffer(*constants).has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::task, 1},
    };
    auto bindingLayout = device.createBindingLayout({.entries = bindingEntries, .label = "Metal task layout"});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingWrites{
        rachis::BindingWrite{.binding = 0, .resource = rachis::BufferBinding{.buffer = *constants, .size = 256}},
    };
    auto bindingSet = rachis::test::createBindingSet(
        device, {.layout = *bindingLayout, .writes = bindingWrites, .label = "Task set"});
    REQUIRE(bindingSet.has_value());

    const std::array pushConstantRanges{rachis::PushConstantRange{rachis::ShaderStage::mesh, 0, 4 * sizeof(float)}};
    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstantRanges,
        .label = "Metal mesh pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());

    const std::array colorTargets{rachis::ColorTargetState{
        .format = rachis::Format::rgba8Unorm,
        .blend =
            {
                .enabled = true,
                .color = {.sourceFactor = rachis::BlendFactor::one, .destinationFactor = rachis::BlendFactor::one},
                .alpha = {.sourceFactor = rachis::BlendFactor::zero, .destinationFactor = rachis::BlendFactor::one},
            },
    }};
    const rachis::GraphicsPipelineDesc noTaskDesc{
        .layout = *pipelineLayout,
        .pixelShader = *pixelShader,
        .meshShader = *meshNoTaskShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .label = "Metal mesh-only pipeline",
    };
    auto invalid = noTaskDesc;
    invalid.meshShader = {};
    CHECK_FALSE(device.createGraphicsPipeline(invalid).has_value());
    invalid = noTaskDesc;
    invalid.vertexShader = *pixelShader;
    CHECK_FALSE(device.createGraphicsPipeline(invalid).has_value());
    invalid = noTaskDesc;
    invalid.topology = rachis::PrimitiveTopology::lineList;
    CHECK_FALSE(device.createGraphicsPipeline(invalid).has_value());
    invalid = noTaskDesc;
    invalid.meshShader = *taskShader;
    CHECK_FALSE(device.createGraphicsPipeline(invalid).has_value());

    const auto missingPixelBytecode = loadPnms("MetalMeshShader.missing.ps.pnms");
    REQUIRE_FALSE(missingPixelBytecode.empty());
    auto missingPixelShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = missingPixelBytecode,
        .entryPoint = "meshPixelMissing",
        .shaderInterface = {},
    });
    REQUIRE(missingPixelShader.has_value());
    invalid = noTaskDesc;
    invalid.pixelShader = *missingPixelShader;
    auto mismatched = device.createGraphicsPipeline(invalid);
    REQUIRE_FALSE(mismatched.has_value());
    CHECK(mismatched.error().code == rachis::ErrorCode::validationError);
    REQUIRE(device.destroyShaderModule(*missingPixelShader).has_value());

    auto noTaskPipeline = device.createGraphicsPipeline(noTaskDesc);
    REQUIRE(noTaskPipeline.has_value());
    auto taskPipeline = device.createGraphicsPipeline({
        .layout = *pipelineLayout,
        .pixelShader = *pixelShader,
        .taskShader = *taskShader,
        .meshShader = *meshShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .label = "Metal task and mesh pipeline",
    });
    REQUIRE(taskPipeline.has_value());

    constexpr std::uint64_t argumentOffset = sizeof(std::uint32_t);
    std::array<std::byte, argumentOffset + sizeof(rachis::DispatchIndirectArguments)> argumentBytes{};
    const rachis::DispatchIndirectArguments dispatchArguments{1, 1, 1};
    std::memcpy(argumentBytes.data() + argumentOffset, &dispatchArguments, sizeof(dispatchArguments));
    auto arguments = device.createBuffer({
        .size = argumentBytes.size(),
        .usage = rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal mesh dispatch arguments",
    });
    auto wrongUsage = device.createBuffer({
        .size = sizeof(dispatchArguments),
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(arguments.has_value());
    REQUIRE(wrongUsage.has_value());
    auto mappedArguments = device.mapBuffer(*arguments);
    REQUIRE(mappedArguments.has_value());
    std::memcpy(mappedArguments->data, argumentBytes.data(), argumentBytes.size());
    REQUIRE(device.unmapBuffer(*arguments).has_value());

    SECTION("Task and mesh stages draw once through direct and once through single-record indirect dispatch")
    {
        const auto target = createMeshTarget(device);
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        CHECK_FALSE(device.dispatchMesh(*command, 1).has_value());
        beginMeshRendering(device, *command, target);
        REQUIRE(device.bindGraphicsPipeline(*command, *taskPipeline).has_value());
        CHECK_FALSE(device.draw(*command, 3).has_value());
        REQUIRE(device.bindGraphicsBindingSet(*command, 0, *bindingSet).has_value());
        setMeshDrawState(device, *command);
        CHECK_FALSE(device.dispatchMesh(*command, 0).has_value());
        CHECK_FALSE(device.dispatchMesh(*command, capabilities.meshShader.maxWorkgroupCount[0] + 1).has_value());
        CHECK_FALSE(device.dispatchMeshIndirect(*command, {.argumentBuffer = *wrongUsage}).has_value());
        CHECK_FALSE(
            device.dispatchMeshIndirect(*command, {.argumentBuffer = *arguments, .argumentOffset = 2}).has_value());
        REQUIRE(device
                    .dispatchMeshIndirect(
                        *command,
                        {.argumentBuffer = *arguments, .argumentOffset = argumentBytes.size(), .maxDispatchCount = 0})
                    .has_value());
        REQUIRE(device.dispatchMesh(*command, 1).has_value());
        placeMesh(device, *command, MeshPlacement::right);
        REQUIRE(device.dispatchMeshIndirect(*command, {.argumentBuffer = *arguments, .argumentOffset = argumentOffset})
                    .has_value());
        finishMeshRendering(device, std::move(*command), target);
        checkMeshCoverage(device, target, MeshCoverage{.left = 1, .middle = 0, .right = 1, .corner = 0});
        destroyMeshTarget(device, target);
    }

    SECTION("A mesh-only pipeline draws without a task stage")
    {
        const auto target = createMeshTarget(device);
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        beginMeshRendering(device, *command, target);
        REQUIRE(device.bindGraphicsPipeline(*command, *noTaskPipeline).has_value());
        REQUIRE(device.bindGraphicsBindingSet(*command, 0, *bindingSet).has_value());
        setMeshDrawState(device, *command);
        placeMesh(device, *command, MeshPlacement::right);
        REQUIRE(device.dispatchMeshIndirect(*command, {.argumentBuffer = *arguments, .argumentOffset = argumentOffset})
                    .has_value());
        finishMeshRendering(device, std::move(*command), target);
        checkMeshCoverage(device, target, MeshCoverage{.left = 0, .middle = 0, .right = 1, .corner = 0});
        destroyMeshTarget(device, target);
    }

    SECTION("Each record of a multi-record mesh dispatch reads its own arguments")
    {
        const auto recordTaskBytecode = loadPnms("MetalMeshRecords.as.pnms");
        const auto recordMeshBytecode = loadPnms("MetalMeshRecords.ms.pnms");
        const auto recordMeshFromTaskBytecode = loadPnms("MetalMeshRecords.task.ms.pnms");
        const auto recordPixelBytecode = loadPnms("MetalMeshRecords.ps.pnms");
        REQUIRE_FALSE(recordTaskBytecode.empty());
        REQUIRE_FALSE(recordMeshBytecode.empty());
        REQUIRE_FALSE(recordMeshFromTaskBytecode.empty());
        REQUIRE_FALSE(recordPixelBytecode.empty());
        const rachis::ShaderInterfaceDesc drawIndexInterface{
            .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1},
            .drawIndex = true,
        };
        auto recordTask = device.createShaderModule({
            .stage = rachis::ShaderStage::task,
            .bytecode = recordTaskBytecode,
            .entryPoint = "recordTask",
            .shaderInterface = drawIndexInterface,
        });
        auto recordMesh = device.createShaderModule({
            .stage = rachis::ShaderStage::mesh,
            .bytecode = recordMeshBytecode,
            .entryPoint = "recordMesh",
            .shaderInterface = drawIndexInterface,
        });
        auto recordMeshFromTask = device.createShaderModule({
            .stage = rachis::ShaderStage::mesh,
            .bytecode = recordMeshFromTaskBytecode,
            .entryPoint = "recordMeshFromTask",
            .shaderInterface = drawIndexInterface,
        });
        auto recordPixel = device.createShaderModule({
            .stage = rachis::ShaderStage::pixel,
            .bytecode = recordPixelBytecode,
            .entryPoint = "recordColor",
        });
        REQUIRE(recordTask.has_value());
        REQUIRE(recordMesh.has_value());
        REQUIRE(recordMeshFromTask.has_value());
        REQUIRE(recordPixel.has_value());
        auto recordLayout = device.createPipelineLayout({.drawIndex = true});
        REQUIRE(recordLayout.has_value());
        auto recordNoTaskPipeline = device.createGraphicsPipeline({
            .layout = *recordLayout,
            .pixelShader = *recordPixel,
            .meshShader = *recordMesh,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
        });
        auto recordTaskPipeline = device.createGraphicsPipeline({
            .layout = *recordLayout,
            .pixelShader = *recordPixel,
            .taskShader = *recordTask,
            .meshShader = *recordMeshFromTask,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
        });
        REQUIRE(recordNoTaskPipeline.has_value());
        REQUIRE(recordTaskPipeline.has_value());

        struct MeshRecord final
        {
            std::uint32_t drawIndex = 0;
            rachis::DispatchIndirectArguments arguments{};
        };
        static_assert(sizeof(MeshRecord) == rachis::drawIndexPrefixSize + sizeof(rachis::DispatchIndirectArguments));
        struct RecordCase final
        {
            std::array<rachis::DispatchIndirectArguments, 3> records{};
            MeshCoverage coverage{};
        };
        const rachis::DispatchIndirectArguments one{1, 1, 1};
        const rachis::DispatchIndirectArguments two{2, 1, 1};
        const rachis::DispatchIndirectArguments none{0, 0, 0};
        const std::array cases{
            RecordCase{.records = {none, one, none}, .coverage = {.middle = 1}},
            RecordCase{.records = {one, none, two}, .coverage = {.left = 1, .right = 2}},
            RecordCase{.records = {none, none, one}, .coverage = {.right = 1}},
            RecordCase{.records = {two, one, none}, .coverage = {.left = 2, .middle = 1}},
        };
        for (const bool taskStage : {false, true})
        {
            for (const auto& recordCase : cases)
            {
                CAPTURE(taskStage);
                CAPTURE(recordCase.coverage.left, recordCase.coverage.middle, recordCase.coverage.right);
                std::array<MeshRecord, 3> records{};
                for (std::uint32_t record = 0; record < records.size(); ++record)
                {
                    records[record] = MeshRecord{.drawIndex = record, .arguments = recordCase.records[record]};
                }
                auto recordBuffer = device.createBuffer({
                    .size = sizeof(records),
                    .usage = rachis::BufferUsage::indirect,
                    .memoryType = rachis::MemoryType::upload,
                });
                REQUIRE(recordBuffer.has_value());
                auto mappedRecords = device.mapBuffer(*recordBuffer);
                REQUIRE(mappedRecords.has_value());
                std::memcpy(mappedRecords->data, records.data(), sizeof(records));
                REQUIRE(device.unmapBuffer(*recordBuffer).has_value());

                const auto target = createMeshTarget(device);
                auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
                REQUIRE(command.has_value());
                beginMeshRendering(device, *command, target);
                REQUIRE(device.bindGraphicsPipeline(*command, taskStage ? *recordTaskPipeline : *recordNoTaskPipeline)
                            .has_value());
                REQUIRE(device
                            .setViewport(*command,
                                         {
                                             .width = static_cast<float>(targetSize),
                                             .height = static_cast<float>(targetSize),
                                             .maxDepth = 1.0F,
                                         })
                            .has_value());
                REQUIRE(device.setScissor(*command, {.width = targetSize, .height = targetSize}).has_value());
                REQUIRE(device
                            .dispatchMeshIndirect(*command,
                                                  {
                                                      .argumentBuffer = *recordBuffer,
                                                      .maxDispatchCount = 3,
                                                      .drawIndexPrefix = true,
                                                  })
                            .has_value());
                finishMeshRendering(device, std::move(*command), target);
                checkMeshCoverage(device, target, recordCase.coverage);
                destroyMeshTarget(device, target);
                REQUIRE(device.destroyBuffer(*recordBuffer).has_value());
            }
        }

        REQUIRE(device.destroyPipeline(*recordTaskPipeline).has_value());
        REQUIRE(device.destroyPipeline(*recordNoTaskPipeline).has_value());
        REQUIRE(device.destroyPipelineLayout(*recordLayout).has_value());
        for (const auto module : {*recordPixel, *recordMeshFromTask, *recordMesh, *recordTask})
        {
            REQUIRE(device.destroyShaderModule(module).has_value());
        }
    }

    SECTION("A mesh pipeline whose shaders read no argument table runs one record per indirect dispatch")
    {
        const auto fixedBytecode = loadPnms("MetalMeshShader.fixed.ms.pnms");
        REQUIRE_FALSE(fixedBytecode.empty());
        auto fixedShader = device.createShaderModule({
            .stage = rachis::ShaderStage::mesh,
            .bytecode = fixedBytecode,
            .entryPoint = "meshFixed",
            .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}),
        });
        REQUIRE(fixedShader.has_value());
        auto emptyLayout = device.createPipelineLayout({});
        REQUIRE(emptyLayout.has_value());
        auto fixedPipeline = device.createGraphicsPipeline({
            .layout = *emptyLayout,
            .pixelShader = *pixelShader,
            .meshShader = *fixedShader,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
            .label = "Metal binding-free mesh pipeline",
        });
        REQUIRE(fixedPipeline.has_value());

        const auto target = createMeshTarget(device);
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        beginMeshRendering(device, *command, target);
        REQUIRE(device.bindGraphicsPipeline(*command, *fixedPipeline).has_value());
        REQUIRE(device
                    .setViewport(*command,
                                 {
                                     .width = static_cast<float>(targetSize),
                                     .height = static_cast<float>(targetSize),
                                     .maxDepth = 1.0F,
                                 })
                    .has_value());
        REQUIRE(device.setScissor(*command, {.width = targetSize, .height = targetSize}).has_value());
        const std::array<rachis::DispatchIndirectArguments, 2> twoRecords{dispatchArguments, dispatchArguments};
        auto twoRecordBuffer = device.createBuffer({
            .size = sizeof(twoRecords),
            .usage = rachis::BufferUsage::indirect,
            .memoryType = rachis::MemoryType::upload,
        });
        REQUIRE(twoRecordBuffer.has_value());
        const auto several =
            device.dispatchMeshIndirect(*command, {.argumentBuffer = *twoRecordBuffer, .maxDispatchCount = 2});
        REQUIRE_FALSE(several.has_value());
        CHECK(several.error().code == rachis::ErrorCode::unsupported);
        REQUIRE(device.dispatchMeshIndirect(*command, {.argumentBuffer = *arguments, .argumentOffset = argumentOffset})
                    .has_value());
        finishMeshRendering(device, std::move(*command), target);
        checkMeshCoverage(device, target, MeshCoverage{.left = 1, .middle = 0, .right = 0, .corner = 0});
        destroyMeshTarget(device, target);

        REQUIRE(device.destroyBuffer(*twoRecordBuffer).has_value());
        REQUIRE(device.destroyPipeline(*fixedPipeline).has_value());
        REQUIRE(device.destroyPipelineLayout(*emptyLayout).has_value());
        REQUIRE(device.destroyShaderModule(*fixedShader).has_value());
    }

    SECTION("Counted mesh dispatch draws only the number of records a GPU fill wrote")
    {
        REQUIRE(capabilities.drawIndirectCount);
        constexpr std::uint32_t recordCount = 3;
        const std::array<rachis::DispatchIndirectArguments, recordCount> records{
            dispatchArguments, dispatchArguments, dispatchArguments};
        auto countedArguments = device.createBuffer({
            .size = sizeof(records),
            .usage = rachis::BufferUsage::indirect,
            .memoryType = rachis::MemoryType::upload,
        });
        REQUIRE(countedArguments.has_value());
        auto mappedRecords = device.mapBuffer(*countedArguments);
        REQUIRE(mappedRecords.has_value());
        std::memcpy(mappedRecords->data, records.data(), sizeof(records));
        REQUIRE(device.unmapBuffer(*countedArguments).has_value());

        for (const std::uint32_t gpuCount : {0U, 2U})
        {
            CAPTURE(gpuCount);
            auto countBuffer = device.createBuffer({
                .size = sizeof(std::uint32_t),
                .usage =
                    rachis::BufferUsage::indirect | rachis::BufferUsage::storage | rachis::BufferUsage::copyDestination,
                .memoryType = rachis::MemoryType::upload,
            });
            REQUIRE(countBuffer.has_value());
            const std::uint32_t staleCount = 1;
            auto mappedCount = device.mapBuffer(*countBuffer);
            REQUIRE(mappedCount.has_value());
            std::memcpy(mappedCount->data, &staleCount, sizeof(staleCount));
            REQUIRE(device.unmapBuffer(*countBuffer).has_value());

            const auto target = createMeshTarget(device);
            auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
            REQUIRE(command.has_value());
            REQUIRE(device.fillBuffer(*command, *countBuffer, 0, rachis::wholeResource, gpuCount).has_value());
            REQUIRE(device
                        .bufferBarrier(*command,
                                       {
                                           .buffer = *countBuffer,
                                           .stagesBefore = rachis::PipelineStage::computeShader,
                                           .stagesAfter = rachis::PipelineStage::indirect,
                                           .accessBefore = rachis::Access::unorderedAccess,
                                           .accessAfter = rachis::Access::indirectArgument,
                                       })
                        .has_value());
            beginMeshRendering(device, *command, target);
            REQUIRE(device.bindGraphicsPipeline(*command, *noTaskPipeline).has_value());
            REQUIRE(device.bindGraphicsBindingSet(*command, 0, *bindingSet).has_value());
            setMeshDrawState(device, *command);
            REQUIRE(device
                        .dispatchMeshIndirect(*command,
                                              {.argumentBuffer = *countedArguments,
                                               .maxDispatchCount = recordCount,
                                               .countBuffer = *countBuffer})
                        .has_value());
            finishMeshRendering(device, std::move(*command), target);
            checkMeshCoverage(device, target, MeshCoverage{.left = gpuCount, .middle = 0, .right = 0, .corner = 0});
            destroyMeshTarget(device, target);
            REQUIRE(device.destroyBuffer(*countBuffer).has_value());
        }
        REQUIRE(device.destroyBuffer(*countedArguments).has_value());
    }

    REQUIRE(device.destroyBuffer(*arguments).has_value());
    REQUIRE(device.destroyBuffer(*wrongUsage).has_value());
    REQUIRE(device.destroyPipeline(*taskPipeline).has_value());
    REQUIRE(device.destroyPipeline(*noTaskPipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyBuffer(*constants).has_value());
    REQUIRE(device.destroyShaderModule(*taskShader).has_value());
    REQUIRE(device.destroyShaderModule(*meshShader).has_value());
    REQUIRE(device.destroyShaderModule(*meshNoTaskShader).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
}

TEST_CASE("Metal task and mesh modules reject workgroups beyond the stage limits",
          "[size:large][rhi][metal][mesh]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.meshShaders)
    {
        SKIP("The Metal GPU is older than the Apple9 family.");
    }
    const auto cooked = loadPnms("MetalMeshShader.fixed.ms.pnms");
    REQUIRE_FALSE(cooked.empty());
    const auto limit = capabilities.meshShader.maxMeshWorkgroupInvocations;
    const auto moduleWith = [&](const rachis::ShaderWorkgroupSize& size) {
        const auto bytecode = withThreadGroupSize(cooked, size);
        return device.createShaderModule({
            .stage = rachis::ShaderStage::mesh,
            .bytecode = bytecode,
            .entryPoint = "meshFixed",
            .shaderInterface = rachis::test::computeShaderInterface(size),
        });
    };

    for (const auto& size : {rachis::ShaderWorkgroupSize{4'194'304, 2'097'152, 2'097'152},
                             rachis::ShaderWorkgroupSize{limit, 2, 1},
                             rachis::ShaderWorkgroupSize{1, 1, limit + 1}})
    {
        CAPTURE(size.x, size.y, size.z);
        const auto rejected = moduleWith(size);
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::unsupported);
    }
    auto accepted = moduleWith({limit, 1, 1});
    REQUIRE(accepted.has_value());
    REQUIRE(device.destroyShaderModule(*accepted).has_value());
}

TEST_CASE("Metal mesh pipeline cache entries depend on the mesh workgroup size", "[size:large][rhi][metal][mesh]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    if (!rachis::test::requireCapabilities(device).meshShaders)
    {
        SKIP("The Metal GPU is older than the Apple9 family.");
    }
    const auto cooked = loadPnms("MetalMeshShader.fixed.ms.pnms");
    const auto pixelBytecode = loadPnms("MetalMeshShader.ps.pnms");
    REQUIRE_FALSE(cooked.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    const auto widerBytecode = withThreadGroupSize(cooked, {2, 1, 1});
    auto narrow = device.createShaderModule({
        .stage = rachis::ShaderStage::mesh,
        .bytecode = cooked,
        .entryPoint = "meshFixed",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}),
    });
    auto wide = device.createShaderModule({
        .stage = rachis::ShaderStage::mesh,
        .bytecode = widerBytecode,
        .entryPoint = "meshFixed",
        .shaderInterface = rachis::test::computeShaderInterface({2, 1, 1}),
    });
    auto pixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "meshPixel",
    });
    REQUIRE(narrow.has_value());
    REQUIRE(wide.has_value());
    REQUIRE(pixel.has_value());
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout.has_value());
    auto cache = device.createPipelineCache({.label = "Metal mesh workgroup cache"});
    REQUIRE(cache.has_value());
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    const auto pipelineWith = [&](rachis::ShaderModuleHandle mesh) {
        return device.createGraphicsPipeline({
            .layout = *layout,
            .pixelShader = *pixel,
            .meshShader = mesh,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
            .cache = *cache,
            .cacheKey = {.high = 0x6d657368U, .low = 1},
        });
    };
    auto first = pipelineWith(*narrow);
    REQUIRE(first.has_value());
    auto second = pipelineWith(*wide);
    CHECK_FALSE(second.has_value());

    REQUIRE(device.destroyPipeline(*first).has_value());
    if (second.has_value())
    {
        REQUIRE(device.destroyPipeline(*second).has_value());
    }
    REQUIRE(device.destroyPipelineCache(*cache).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    for (const auto module : {*pixel, *wide, *narrow})
    {
        REQUIRE(device.destroyShaderModule(module).has_value());
    }
}

TEST_CASE("Metal task shaders launch every mesh workgroup they request", "[size:large][rhi][metal][mesh]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    if (!rachis::test::requireCapabilities(device).meshShaders)
    {
        SKIP("The Metal GPU is older than the Apple9 family.");
    }
    constexpr std::uint32_t meshGroups = 5000;
    const auto taskBytecode = loadPnms("MetalMeshShader.grid.as.pnms");
    const auto meshBytecode = loadPnms("MetalMeshShader.count.ms.pnms");
    const auto pixelBytecode = loadPnms("MetalMeshShader.ps.pnms");
    REQUIRE_FALSE(taskBytecode.empty());
    REQUIRE_FALSE(meshBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    const std::array taskBindings{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1}};
    const std::array meshBindings{rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1}};
    auto task = device.createShaderModule({
        .stage = rachis::ShaderStage::task,
        .bytecode = taskBytecode,
        .entryPoint = "taskGrid",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, taskBindings),
    });
    auto mesh = device.createShaderModule({
        .stage = rachis::ShaderStage::mesh,
        .bytecode = meshBytecode,
        .entryPoint = "meshCount",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, meshBindings),
    });
    auto pixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "meshPixel",
    });
    REQUIRE(task.has_value());
    REQUIRE(mesh.has_value());
    REQUIRE(pixel.has_value());

    auto constants = device.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
    });
    auto counter = device.createBuffer({
        .size = sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::storage,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(constants.has_value());
    REQUIRE(counter.has_value());
    auto mappedConstants = device.mapBuffer(*constants);
    REQUIRE(mappedConstants.has_value());
    std::memset(mappedConstants->data, 0, mappedConstants->size);
    const std::array taskColor{0.0F, 0.0F, 0.0F, static_cast<float>(meshGroups)};
    std::memcpy(mappedConstants->data, taskColor.data(), sizeof(taskColor));
    REQUIRE(device.unmapBuffer(*constants).has_value());
    auto mappedCounter = device.mapBuffer(*counter);
    REQUIRE(mappedCounter.has_value());
    std::memset(mappedCounter->data, 0, sizeof(std::uint32_t));
    REQUIRE(device.unmapBuffer(*counter).has_value());

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::task, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::mesh, 1},
    };
    auto bindingLayout = device.createBindingLayout({.entries = entries});
    REQUIRE(bindingLayout.has_value());
    const std::array writes{
        rachis::BindingWrite{.binding = 0, .resource = rachis::BufferBinding{.buffer = *constants, .size = 256}},
        rachis::BindingWrite{
            .binding = 1,
            .resource = rachis::BufferBinding{.buffer = *counter, .structureByteStride = sizeof(std::uint32_t)},
        },
    };
    auto set = rachis::test::createBindingSet(device, {.layout = *bindingLayout, .writes = writes});
    REQUIRE(set.has_value());
    const std::array layouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts});
    REQUIRE(pipelineLayout.has_value());
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto pipeline = device.createGraphicsPipeline({
        .layout = *pipelineLayout,
        .pixelShader = *pixel,
        .taskShader = *task,
        .meshShader = *mesh,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    REQUIRE(pipeline.has_value());

    const auto target = createMeshTarget(device);
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    beginMeshRendering(device, *command, target);
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindGraphicsBindingSet(*command, 0, *set).has_value());
    REQUIRE(device
                .setViewport(*command,
                             {
                                 .width = static_cast<float>(targetSize),
                                 .height = static_cast<float>(targetSize),
                                 .maxDepth = 1.0F,
                             })
                .has_value());
    REQUIRE(device.setScissor(*command, {.width = targetSize, .height = targetSize}).has_value());
    REQUIRE(device.dispatchMesh(*command, 1).has_value());
    finishMeshRendering(device, std::move(*command), target);
    auto mapped = device.mapBuffer(*counter);
    REQUIRE(mapped.has_value());
    std::uint32_t launched = 0;
    std::memcpy(&launched, mapped->data, sizeof(launched));
    REQUIRE(device.unmapBuffer(*counter).has_value());
    CHECK(launched == meshGroups);
    CHECK(device.diagnosticCounts().errors == 0);

    destroyMeshTarget(device, target);
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingSet(*set).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyBuffer(*counter).has_value());
    REQUIRE(device.destroyBuffer(*constants).has_value());
    REQUIRE(device.destroyShaderModule(*pixel).has_value());
    REQUIRE(device.destroyShaderModule(*mesh).has_value());
    REQUIRE(device.destroyShaderModule(*task).has_value());
}

TEST_CASE("Metal task shaders may declare their own mesh-grid maximum", "[size:large][rhi][metal][mesh]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    if (!rachis::test::requireCapabilities(device).meshShaders)
    {
        SKIP("The Metal GPU is older than the Apple9 family.");
    }
    if (std::system("xcrun -sdk macosx --find metal > /dev/null 2>&1") != 0)
    {
        SKIP("The Metal compiler is not installed.");
    }
    constexpr std::string_view source = R"msl(#include <metal_stdlib>
using namespace metal;
struct GridPayload
{
    uint groups;
};
struct GridVertex
{
    float4 position [[position]];
};
using GridMesh = metal::mesh<GridVertex, void, 3, 1, topology::triangle>;
[[object, max_total_threadgroups_per_mesh_grid(8)]] void gridTask(object_data GridPayload& payload [[payload]],
                                                                 mesh_grid_properties grid)
{
    payload.groups = 4;
    grid.set_threadgroups_per_grid(uint3(4, 1, 1));
}
[[mesh]] void gridMesh(GridMesh output,
                       const object_data GridPayload& payload [[payload]],
                       device atomic_uint* counter [[buffer(0)]])
{
    output.set_primitive_count(0);
    atomic_fetch_add_explicit(counter, payload.groups == 4 ? 1U : 0U, memory_order_relaxed);
}
[[fragment]] float4 gridPixel()
{
    return float4(0.0F, 1.0F, 0.0F, 1.0F);
}
)msl";
    constexpr std::string_view json = R"({
        "parameters": [
            {"name": "counter", "binding": {"kind": "constantBuffer", "index": 0},
             "type": {"kind": "resource", "baseShape": "structuredBuffer", "access": "readWrite"}}
        ],
        "entryPoints": [
            {"name": "gridTask", "stage": "amplification", "threadGroupSize": [1, 1, 1], "bindings": []},
            {"name": "gridMesh", "stage": "mesh", "threadGroupSize": [1, 1, 1],
             "bindings": [{"name": "counter", "binding": {"kind": "constantBuffer", "index": 0, "used": 1}}]},
            {"name": "gridPixel", "stage": "fragment", "bindings": []}
        ],
        "rachisBindings": [{"name": "counter", "set": 0, "binding": 0}]
    })";
    const auto directory =
        std::filesystem::temp_directory_path() / ("rachis-mesh-grid-" + std::to_string(::getpid()));
    std::filesystem::create_directories(directory);
    const auto sourcePath = directory / "grid.metal";
    const auto libraryPath = directory / "grid.metallib";
    {
        std::ofstream stream(sourcePath, std::ios::binary);
        stream << source;
        REQUIRE(stream);
    }
    const auto command = "xcrun -sdk macosx metal -std=metal3.1 -o \"" + libraryPath.string() + "\" \"" +
                         sourcePath.string() + "\"";
    REQUIRE(std::system(command.c_str()) == 0);
    const auto metallib = loadBytes(libraryPath);
    std::filesystem::remove_all(directory);
    REQUIRE_FALSE(metallib.empty());
    const auto bytecode = rachis::metal::packShaderManifest(std::as_bytes(std::span(json)), metallib);

    const std::array counterBinding{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1}};
    auto task = device.createShaderModule({
        .stage = rachis::ShaderStage::task,
        .bytecode = bytecode,
        .entryPoint = "gridTask",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}),
    });
    auto mesh = device.createShaderModule({
        .stage = rachis::ShaderStage::mesh,
        .bytecode = bytecode,
        .entryPoint = "gridMesh",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, counterBinding),
    });
    auto pixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = bytecode,
        .entryPoint = "gridPixel",
    });
    REQUIRE(task.has_value());
    REQUIRE(mesh.has_value());
    REQUIRE(pixel.has_value());

    auto counter = device.createBuffer({
        .size = sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::storage,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(counter.has_value());
    auto mappedCounter = device.mapBuffer(*counter);
    REQUIRE(mappedCounter.has_value());
    std::memset(mappedCounter->data, 0, sizeof(std::uint32_t));
    REQUIRE(device.unmapBuffer(*counter).has_value());
    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::mesh, 1},
    };
    auto bindingLayout = device.createBindingLayout({.entries = entries});
    REQUIRE(bindingLayout.has_value());
    const std::array writes{rachis::BindingWrite{
        .binding = 0,
        .resource = rachis::BufferBinding{.buffer = *counter, .structureByteStride = sizeof(std::uint32_t)},
    }};
    auto set = rachis::test::createBindingSet(device, {.layout = *bindingLayout, .writes = writes});
    REQUIRE(set.has_value());
    const std::array layouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts});
    REQUIRE(pipelineLayout.has_value());
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto pipeline = device.createGraphicsPipeline({
        .layout = *pipelineLayout,
        .pixelShader = *pixel,
        .taskShader = *task,
        .meshShader = *mesh,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    INFO((pipeline.has_value() ? std::string{} : pipeline.error().message));
    REQUIRE(pipeline.has_value());

    const auto target = createMeshTarget(device);
    auto recording = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(recording.has_value());
    beginMeshRendering(device, *recording, target);
    REQUIRE(device.bindGraphicsPipeline(*recording, *pipeline).has_value());
    REQUIRE(device.bindGraphicsBindingSet(*recording, 0, *set).has_value());
    REQUIRE(device
                .setViewport(*recording,
                             {
                                 .width = static_cast<float>(targetSize),
                                 .height = static_cast<float>(targetSize),
                                 .maxDepth = 1.0F,
                             })
                .has_value());
    REQUIRE(device.setScissor(*recording, {.width = targetSize, .height = targetSize}).has_value());
    REQUIRE(device.dispatchMesh(*recording, 1).has_value());
    finishMeshRendering(device, std::move(*recording), target);
    auto mapped = device.mapBuffer(*counter);
    REQUIRE(mapped.has_value());
    std::uint32_t launched = 0;
    std::memcpy(&launched, mapped->data, sizeof(launched));
    REQUIRE(device.unmapBuffer(*counter).has_value());
    CHECK(launched == 4);
    CHECK(device.diagnosticCounts().errors == 0);

    destroyMeshTarget(device, target);
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingSet(*set).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyBuffer(*counter).has_value());
    for (const auto module : {*pixel, *mesh, *task})
    {
        REQUIRE(device.destroyShaderModule(module).has_value());
    }
}
