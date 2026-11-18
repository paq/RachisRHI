#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string_view>
#include <vector>

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
                .clearValue = {0.0F, 0.0F, 1.0F, 1.0F},
            },
        };
        REQUIRE(device.beginRendering(command, {.colorAttachments = attachments}).has_value());
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
        const std::array meshControl{1.0F, 0.0F, 0.0F, 0.0F};
        REQUIRE(
            device
                .setGraphicsPushConstants(command, rachis::ShaderStage::mesh, 0, std::as_bytes(std::span(meshControl)))
                .has_value());
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

    void checkGreenTriangleOnBlue(rachis::Device& device, const MeshTarget& target)
    {
        auto mapped = device.mapBuffer(target.readback);
        REQUIRE(mapped.has_value());
        const auto* corner = mapped->data;
        const auto* center = mapped->data + static_cast<std::uint64_t>(targetSize / 2) * rowPitch +
                             static_cast<std::uint64_t>(targetSize / 2) * 4;
        CHECK(corner[0] == std::byte{0});
        CHECK(corner[1] == std::byte{0});
        CHECK(corner[2] == std::byte{255});
        CHECK(corner[3] == std::byte{255});
        CHECK(center[0] == std::byte{0});
        CHECK(center[1] == std::byte{255});
        CHECK(center[2] == std::byte{0});
        CHECK(center[3] == std::byte{255});
        REQUIRE(device.unmapBuffer(target.readback).has_value());
    }

    void destroyMeshTarget(rachis::Device& device, const MeshTarget& target)
    {
        REQUIRE(device.destroyTextureView(target.view).has_value());
        REQUIRE(device.destroyTexture(target.texture).has_value());
        REQUIRE(device.destroyBuffer(target.readback).has_value());
    }

} // namespace

TEST_CASE("Metal reports mesh and task shaders on Apple9 GPUs with Metal mesh-grid limits",
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
    CHECK(capabilities.taskShaders);
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
    const std::array taskColor{0.0F, 1.0F, 0.0F, 1.0F};
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

    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
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

    SECTION("Task and mesh stages draw through direct and single-record indirect dispatch")
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
        REQUIRE(device.dispatchMeshIndirect(*command, {.argumentBuffer = *arguments, .argumentOffset = argumentOffset})
                    .has_value());
        finishMeshRendering(device, std::move(*command), target);
        checkGreenTriangleOnBlue(device, target);
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
        REQUIRE(device.dispatchMeshIndirect(*command, {.argumentBuffer = *arguments, .argumentOffset = argumentOffset})
                    .has_value());
        finishMeshRendering(device, std::move(*command), target);
        checkGreenTriangleOnBlue(device, target);
        destroyMeshTarget(device, target);
    }

    SECTION("Counted mesh dispatch draws only the GPU-written number of records")
    {
        REQUIRE(capabilities.drawIndirectCount);
        const std::array records{dispatchArguments, dispatchArguments};
        const std::array<std::uint32_t, 2> counts{0, 1};
        auto countedArguments = device.createBuffer({
            .size = sizeof(records),
            .usage = rachis::BufferUsage::indirect,
            .memoryType = rachis::MemoryType::upload,
        });
        auto countBuffer = device.createBuffer({
            .size = sizeof(counts),
            .usage = rachis::BufferUsage::indirect,
            .memoryType = rachis::MemoryType::upload,
        });
        REQUIRE(countedArguments.has_value());
        REQUIRE(countBuffer.has_value());
        auto mappedRecords = device.mapBuffer(*countedArguments);
        REQUIRE(mappedRecords.has_value());
        std::memcpy(mappedRecords->data, records.data(), sizeof(records));
        REQUIRE(device.unmapBuffer(*countedArguments).has_value());
        auto mappedCounts = device.mapBuffer(*countBuffer);
        REQUIRE(mappedCounts.has_value());
        std::memcpy(mappedCounts->data, counts.data(), sizeof(counts));
        REQUIRE(device.unmapBuffer(*countBuffer).has_value());

        for (const std::uint64_t countOffset : {std::uint64_t{0}, std::uint64_t{sizeof(std::uint32_t)}})
        {
            const auto target = createMeshTarget(device);
            auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
            REQUIRE(command.has_value());
            beginMeshRendering(device, *command, target);
            REQUIRE(device.bindGraphicsPipeline(*command, *noTaskPipeline).has_value());
            REQUIRE(device.bindGraphicsBindingSet(*command, 0, *bindingSet).has_value());
            setMeshDrawState(device, *command);
            REQUIRE(device
                        .dispatchMeshIndirect(*command,
                                              {.argumentBuffer = *countedArguments,
                                               .maxDispatchCount = 2,
                                               .countBuffer = *countBuffer,
                                               .countOffset = countOffset})
                        .has_value());
            finishMeshRendering(device, std::move(*command), target);
            auto mapped = device.mapBuffer(target.readback);
            REQUIRE(mapped.has_value());
            const auto* center = mapped->data + static_cast<std::uint64_t>(targetSize / 2) * rowPitch +
                                 static_cast<std::uint64_t>(targetSize / 2) * 4;
            CAPTURE(countOffset);
            CHECK(center[1] == (countOffset == 0 ? std::byte{0} : std::byte{255}));
            CHECK(center[2] == (countOffset == 0 ? std::byte{255} : std::byte{0}));
            REQUIRE(device.unmapBuffer(target.readback).has_value());
            destroyMeshTarget(device, target);
        }
        REQUIRE(device.destroyBuffer(*countBuffer).has_value());
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
