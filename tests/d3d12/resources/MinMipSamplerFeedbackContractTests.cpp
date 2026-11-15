#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

    constexpr std::uint32_t sourceExtent = 64;
    constexpr std::uint32_t sourceMipLevels = 7;
    constexpr rachis::Extent3D feedbackRegion{8, 8, 1};
    constexpr rachis::Extent3D decodedExtent{8, 8, 1};
    constexpr std::uint8_t noMipRequested = 0xff;

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    [[nodiscard]] std::vector<std::byte> loadShader()
    {
        std::ifstream stream(RACHIS_TEST_MIN_MIP_SAMPLER_FEEDBACK_SHADER_PATH, std::ios::binary | std::ios::ate);
        const auto end = stream.tellg();
        if (end <= 0)
        {
            return {};
        }
        std::vector<std::byte> result(static_cast<std::size_t>(end));
        stream.seekg(0, std::ios::beg);
        stream.read(reinterpret_cast<char*>(result.data()), end);
        return stream ? result : std::vector<std::byte>{};
    }

    [[nodiscard]] rachis::TextureDesc sampledTextureDesc()
    {
        return rachis::TextureDesc{
            .width = sourceExtent,
            .height = sourceExtent,
            .mipLevels = sourceMipLevels,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::sampled,
            .label = "MinMip feedback sampled texture",
        };
    }

    [[nodiscard]] rachis::TextureDesc decodedTextureDesc(std::string_view label)
    {
        return rachis::TextureDesc{
            .width = decodedExtent.width,
            .height = decodedExtent.height,
            .format = rachis::Format::r8Uint,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
            .label = label,
        };
    }

    void checkMismatchedPairingRejected(rachis::Device& device,
                                        rachis::SamplerFeedbackMapHandle feedbackMap,
                                        rachis::SamplerHandle sampler,
                                        rachis::BindingLayoutHandle bindingLayout,
                                        rachis::BindingSetHandle bindingSet)
    {
        auto mismatchedTexture = device.createTexture(sampledTextureDesc());
        REQUIRE(mismatchedTexture.has_value());
        auto mismatchedView = device.createTextureView({
            .texture = *mismatchedTexture,
            .usage = rachis::TextureViewUsage::sampled,
            .label = "Mismatched MinMip feedback sampled view",
        });
        REQUIRE(mismatchedView.has_value());

        const std::array mismatchedWrites{
            rachis::BindingWrite{.binding = 0, .resource = *mismatchedView},
            rachis::BindingWrite{.binding = 1, .resource = feedbackMap},
            rachis::BindingWrite{.binding = 2, .resource = sampler},
        };
        auto mismatchedBindingSet =
            rachis::test::createBindingSet(device,
                                           {
                                               .layout = bindingLayout,
                                               .writes = mismatchedWrites,
                                               .label = "Mismatched MinMip feedback binding set",
                                           });
        REQUIRE_FALSE(mismatchedBindingSet.has_value());
        CHECK(mismatchedBindingSet.error().code == rachis::ErrorCode::invalidArgument);

        const std::array mismatchedUpdate{
            rachis::BindingWrite{.binding = 0, .resource = *mismatchedView},
        };
        auto rejectedUpdate = device.updateBindingSet(bindingSet, mismatchedUpdate);
        REQUIRE_FALSE(rejectedUpdate.has_value());
        CHECK(rejectedUpdate.error().code == rachis::ErrorCode::invalidArgument);
        REQUIRE(device.destroyTextureView(*mismatchedView).has_value());
        REQUIRE(device.destroyTexture(*mismatchedTexture).has_value());
    }

    void checkFeedbackProperties(const rachis::MinMipFeedbackMapProperties& properties)
    {
        CHECK(properties.mipRegion.width == feedbackRegion.width);
        CHECK(properties.mipRegion.height == feedbackRegion.height);
        CHECK(properties.mipRegion.depth == feedbackRegion.depth);
        CHECK(properties.decodedExtent.width == decodedExtent.width);
        CHECK(properties.decodedExtent.height == decodedExtent.height);
        CHECK(properties.decodedExtent.depth == decodedExtent.depth);
    }

    void recordClearEncodeDecode(rachis::Device& device,
                                 rachis::CommandBuffer& command,
                                 rachis::SamplerFeedbackMapHandle feedbackMap,
                                 rachis::TextureHandle clearSource,
                                 rachis::TextureHandle decodeDestination)
    {
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = clearSource,
                                        .stagesBefore = rachis::PipelineStage::resolve,
                                        .stagesAfter = rachis::PipelineStage::resolve,
                                        .accessBefore = rachis::Access::resolveDestination,
                                        .accessAfter = rachis::Access::resolveSource,
                                        .layoutBefore = rachis::TextureLayout::resolveDestination,
                                        .layoutAfter = rachis::TextureLayout::resolveSource,
                                    })
                    .has_value());
        REQUIRE(device
                    .samplerFeedbackBarrier(command,
                                            {
                                                .feedbackMap = feedbackMap,
                                                .stagesBefore = rachis::PipelineStage::resolve,
                                                .stagesAfter = rachis::PipelineStage::resolve,
                                                .accessBefore = rachis::Access::resolveSource,
                                                .accessAfter = rachis::Access::resolveDestination,
                                                .layoutBefore = rachis::TextureLayout::resolveSource,
                                                .layoutAfter = rachis::TextureLayout::resolveDestination,
                                            })
                    .has_value());
        REQUIRE(device.encodeMinMipFeedback(command, {.source = clearSource, .destination = feedbackMap}).has_value());
        REQUIRE(device
                    .samplerFeedbackBarrier(command,
                                            {
                                                .feedbackMap = feedbackMap,
                                                .stagesBefore = rachis::PipelineStage::resolve,
                                                .stagesAfter = rachis::PipelineStage::resolve,
                                                .accessBefore = rachis::Access::resolveDestination,
                                                .accessAfter = rachis::Access::resolveSource,
                                                .layoutBefore = rachis::TextureLayout::resolveDestination,
                                                .layoutAfter = rachis::TextureLayout::resolveSource,
                                            })
                    .has_value());
        REQUIRE(device.decodeMinMipFeedback(command, {.source = feedbackMap, .destination = decodeDestination})
                    .has_value());
    }

    void checkNoMipRequested(rachis::Device& device,
                             rachis::BufferHandle buffer,
                             const rachis::TextureTransferLayout& layout)
    {
        auto mapped = device.mapBuffer(buffer);
        REQUIRE(mapped.has_value());
        for (std::uint32_t row = 0; row < layout.rowCount; ++row)
        {
            const auto* rowBegin = reinterpret_cast<const std::uint8_t*>(mapped->data) +
                                   static_cast<std::size_t>(row) * layout.bytesPerRow;
            CHECK(std::all_of(rowBegin,
                              rowBegin + static_cast<std::size_t>(layout.rowDataSize),
                              [](std::uint8_t value) { return value == noMipRequested; }));
        }
        REQUIRE(device.unmapBuffer(buffer).has_value());
    }

    void recordDecodeReadbacks(rachis::Device& device,
                               rachis::CommandBuffer& command,
                               rachis::TextureHandle writtenDecode,
                               rachis::TextureHandle clearedDecode,
                               rachis::TextureHandle encodedDecode,
                               rachis::BufferHandle writtenReadback,
                               rachis::BufferHandle clearedReadback,
                               rachis::BufferHandle encodedReadback,
                               const rachis::TextureTransferLayout& layout)
    {
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = clearedDecode,
                                        .stagesBefore = rachis::PipelineStage::resolve,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::resolveSource,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::resolveSource,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        for (const auto destination : {writtenDecode, encodedDecode})
        {
            REQUIRE(device
                        .textureBarrier(command,
                                        {
                                            .texture = destination,
                                            .stagesBefore = rachis::PipelineStage::resolve,
                                            .stagesAfter = rachis::PipelineStage::copy,
                                            .accessBefore = rachis::Access::resolveDestination,
                                            .accessAfter = rachis::Access::copySource,
                                            .layoutBefore = rachis::TextureLayout::resolveDestination,
                                            .layoutAfter = rachis::TextureLayout::copySource,
                                        })
                        .has_value());
        }

        const std::array copies{
            std::pair{writtenReadback, writtenDecode},
            std::pair{clearedReadback, clearedDecode},
            std::pair{encodedReadback, encodedDecode},
        };
        for (const auto& [readback, decode] : copies)
        {
            REQUIRE(device
                        .copyTextureToBuffer(command,
                                             {
                                                 .buffer = readback,
                                                 .bytesPerRow = layout.bytesPerRow,
                                                 .rowsPerImage = layout.rowsPerImage,
                                                 .texture = decode,
                                                 .extent = decodedExtent,
                                             })
                        .has_value());
        }
    }

} // namespace

TEST_CASE("MinMip sampler-feedback capability gates the complete contract",
          "[size:medium][rhi][sampler-feedback][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    if (!rachis::test::requireCapabilities(device).samplerFeedback)
    {
        auto texture = device.createTexture(sampledTextureDesc());
        REQUIRE(texture.has_value());
        auto map = device.createMinMipFeedbackMap({.texture = *texture, .mipRegion = feedbackRegion});
        REQUIRE_FALSE(map.has_value());
        CHECK(map.error().code == rachis::ErrorCode::unsupported);

        const std::array feedbackEntry{
            rachis::BindingLayoutEntry{1, rachis::BindingType::minMipFeedback, rachis::ShaderStage::compute},
        };
        auto layout = device.createBindingLayout({.entries = feedbackEntry});
        REQUIRE_FALSE(layout.has_value());
        CHECK(layout.error().code == rachis::ErrorCode::unsupported);
        REQUIRE(device.destroyTexture(*texture).has_value());
    }

    {
        auto renderTarget = device.createTexture({
            .width = 4,
            .height = 4,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::renderTarget,
            .label = "Sampler-feedback render-pass guard target",
        });
        REQUIRE(renderTarget.has_value());
        auto renderTargetView = device.createTextureView({
            .texture = *renderTarget,
            .usage = rachis::TextureViewUsage::renderTarget,
        });
        REQUIRE(renderTargetView.has_value());
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *renderTarget,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
        const std::array attachments{
            rachis::ColorAttachment{
                .view = *renderTargetView,
                .loadOperation = rachis::LoadOperation::clear,
                .storeOperation = rachis::StoreOperation::store,
            },
        };
        REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
        const auto resetInsidePass = device.resetSamplerFeedbackMap(*command, {});
        REQUIRE_FALSE(resetInsidePass.has_value());
        CHECK(resetInsidePass.error().code == rachis::ErrorCode::invalidState);
        REQUIRE(device.clearColorAttachment(*command, 0, {0.25F, 0.5F, 0.75F, 1.0F}, {}).has_value());
        REQUIRE(device.endRendering(*command).has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        REQUIRE(device.destroyTextureView(*renderTargetView).has_value());
        REQUIRE(device.destroyTexture(*renderTarget).has_value());
    }

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("MinMip feedback reset write decode and lifetime form one observation contract",
          "[size:medium][rhi][sampler-feedback][compute][hardware]")
{
    auto adapters = rachis::enumerateAdapters();
    if (!adapters)
    {
        INFO(adapters.error().message);
    }
    REQUIRE(adapters.has_value());

    std::optional<rachis::Device> selectedDevice;
    std::string selectedAdapter;
    for (const auto& adapter : *adapters)
    {
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .adapterPreference = adapter.type == rachis::AdapterType::software
                                     ? rachis::AdapterPreference::warp
                                     : rachis::AdapterPreference::highPerformance,
            .adapterId = adapter.id,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = true,
            .diagnosticCallback = &printDiagnostic,
        });
        if (created && rachis::test::requireCapabilities(*created).samplerFeedback)
        {
            selectedAdapter = adapter.name;
            selectedDevice.emplace(std::move(*created));
            break;
        }
    }
    if (!selectedDevice)
    {
        SKIP("No D3D12 adapter reports MinMip sampler-feedback support.");
    }
    INFO("Selected adapter: " << selectedAdapter);
    auto& device = *selectedDevice;
    REQUIRE(rachis::test::requireCapabilities(device).shaderModel >= rachis::ApiVersion{6, 5});

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "MinMip sampler-feedback shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {1, 1, 1},
            std::array{
                rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::sampledTexture, 1},
                rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::minMipFeedback, 1},
                rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::sampler, 1},
            }),
    });
    REQUIRE(shader.has_value());

    auto sampledTexture = device.createTexture(sampledTextureDesc());
    REQUIRE(sampledTexture.has_value());
    auto invalidRegion = device.createMinMipFeedbackMap({.texture = *sampledTexture, .mipRegion = {3, 8, 1}});
    REQUIRE_FALSE(invalidRegion.has_value());
    CHECK(invalidRegion.error().code == rachis::ErrorCode::invalidArgument);
    auto oversizedRegion =
        device.createMinMipFeedbackMap({.texture = *sampledTexture, .mipRegion = {sourceExtent, 8, 1}});
    REQUIRE_FALSE(oversizedRegion.has_value());
    CHECK(oversizedRegion.error().code == rachis::ErrorCode::invalidArgument);
    const std::array unsupportedFeedbackArray{
        rachis::BindingLayoutEntry{
            .binding = 1,
            .type = rachis::BindingType::minMipFeedback,
            .visibility = rachis::ShaderStage::compute,
            .count = 2,
        },
    };
    auto feedbackArrayLayout = device.createBindingLayout({.entries = unsupportedFeedbackArray});
    REQUIRE_FALSE(feedbackArrayLayout.has_value());
    CHECK(feedbackArrayLayout.error().code == rachis::ErrorCode::invalidArgument);

    auto feedbackMap = device.createMinMipFeedbackMap({
        .texture = *sampledTexture,
        .mipRegion = feedbackRegion,
        .label = "MinMip feedback map",
    });
    REQUIRE(feedbackMap.has_value());
    auto properties = device.queryMinMipFeedbackMapProperties(*feedbackMap);
    REQUIRE(properties.has_value());
    checkFeedbackProperties(*properties);

    auto writtenDecode = device.createTexture(decodedTextureDesc("Written MinMip feedback decode"));
    auto clearedDecode = device.createTexture(decodedTextureDesc("Cleared MinMip feedback decode"));
    auto encodedDecode = device.createTexture(decodedTextureDesc("Encoded MinMip feedback decode"));
    auto wrongExtentDecode = device.createTexture({
        .width = decodedExtent.width / 2,
        .height = decodedExtent.height,
        .format = rachis::Format::r8Uint,
        .usage = rachis::TextureUsage::copyDestination,
        .label = "Invalid MinMip feedback decode",
    });
    REQUIRE(writtenDecode.has_value());
    REQUIRE(clearedDecode.has_value());
    REQUIRE(encodedDecode.has_value());
    REQUIRE(wrongExtentDecode.has_value());
    auto transferLayout = device.queryTextureTransferLayout(*writtenDecode, {.extent = decodedExtent});
    REQUIRE(transferLayout.has_value());
    CHECK(transferLayout->rowDataSize == decodedExtent.width);
    CHECK(transferLayout->rowCount == decodedExtent.height);

    auto writtenReadback = device.createBuffer({
        .size = transferLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Written MinMip feedback readback",
    });
    auto clearedReadback = device.createBuffer({
        .size = transferLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Cleared MinMip feedback readback",
    });
    auto encodedReadback = device.createBuffer({
        .size = transferLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Encoded MinMip feedback readback",
    });
    REQUIRE(writtenReadback.has_value());
    REQUIRE(clearedReadback.has_value());
    REQUIRE(encodedReadback.has_value());

    auto sampledView = device.createTextureView({
        .texture = *sampledTexture,
        .usage = rachis::TextureViewUsage::sampled,
        .label = "MinMip feedback sampled view",
    });
    auto sampler = device.createSampler({.label = "MinMip feedback sampler"});
    REQUIRE(sampledView.has_value());
    REQUIRE(sampler.has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{1, rachis::BindingType::minMipFeedback, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{2, rachis::BindingType::sampler, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({
        .entries = bindingEntries,
        .label = "MinMip feedback binding layout",
    });
    REQUIRE(bindingLayout.has_value());

    const std::array writes{
        rachis::BindingWrite{.binding = 0, .resource = *sampledView},
        rachis::BindingWrite{.binding = 1, .resource = *feedbackMap},
        rachis::BindingWrite{.binding = 2, .resource = *sampler},
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .label = "MinMip feedback binding set",
                                                     });
    REQUIRE(bindingSet.has_value());
    checkMismatchedPairingRejected(device, *feedbackMap, *sampler, *bindingLayout, *bindingSet);

    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .label = "MinMip feedback pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "MinMip feedback pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    {
        auto copyCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
        REQUIRE(copyCommand.has_value());
        auto invalidReset = device.resetSamplerFeedbackMap(*copyCommand, *feedbackMap);
        REQUIRE_FALSE(invalidReset.has_value());
        CHECK(invalidReset.error().code == rachis::ErrorCode::invalidArgument);
    }
    {
        auto computeCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(computeCommand.has_value());
        auto invalidDecode =
            device.decodeMinMipFeedback(*computeCommand, {.source = *feedbackMap, .destination = *writtenDecode});
        REQUIRE_FALSE(invalidDecode.has_value());
        CHECK(invalidDecode.error().code == rachis::ErrorCode::invalidArgument);
        auto invalidEncode =
            device.encodeMinMipFeedback(*computeCommand, {.source = *clearedDecode, .destination = *feedbackMap});
        REQUIRE_FALSE(invalidEncode.has_value());
        CHECK(invalidEncode.error().code == rachis::ErrorCode::invalidArgument);
    }
    auto invalidDestination =
        device.decodeMinMipFeedback(*command, {.source = *feedbackMap, .destination = *wrongExtentDecode});
    REQUIRE_FALSE(invalidDestination.has_value());
    CHECK(invalidDestination.error().code == rachis::ErrorCode::invalidArgument);
    auto invalidSource =
        device.encodeMinMipFeedback(*command, {.source = *wrongExtentDecode, .destination = *feedbackMap});
    REQUIRE_FALSE(invalidSource.has_value());
    CHECK(invalidSource.error().code == rachis::ErrorCode::invalidArgument);
    {
        auto renderTarget = device.createTexture({
            .width = 4,
            .height = 4,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::renderTarget,
            .label = "Hardware sampler-feedback render-pass guard target",
        });
        REQUIRE(renderTarget.has_value());
        auto renderTargetView = device.createTextureView({
            .texture = *renderTarget,
            .usage = rachis::TextureViewUsage::renderTarget,
        });
        REQUIRE(renderTargetView.has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *renderTarget,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
        const std::array attachments{
            rachis::ColorAttachment{
                .view = *renderTargetView,
                .loadOperation = rachis::LoadOperation::clear,
                .storeOperation = rachis::StoreOperation::store,
            },
        };
        REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
        const auto resetInsidePass = device.resetSamplerFeedbackMap(*command, *feedbackMap);
        REQUIRE_FALSE(resetInsidePass.has_value());
        CHECK(resetInsidePass.error().code == rachis::ErrorCode::invalidState);
        REQUIRE(device.clearColorAttachment(*command, 0, {0.25F, 0.5F, 0.75F, 1.0F}, {}).has_value());
        REQUIRE(device.endRendering(*command).has_value());
        REQUIRE(device.destroyTextureView(*renderTargetView).has_value());
        REQUIRE(device.destroyTexture(*renderTarget).has_value());
    }

    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *sampledTexture,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::computeShader,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::shaderResource,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::shaderResource,
                                })
                .has_value());
    const std::array duplicateFeedbackBarriers{
        rachis::SamplerFeedbackBarrier{
            .feedbackMap = *feedbackMap,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::computeShader,
            .accessBefore = rachis::Access::common,
            .accessAfter = rachis::Access::unorderedAccess,
            .layoutBefore = rachis::TextureLayout::common,
            .layoutAfter = rachis::TextureLayout::unorderedAccess,
        },
        rachis::SamplerFeedbackBarrier{
            .feedbackMap = *feedbackMap,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::resolve,
            .accessBefore = rachis::Access::common,
            .accessAfter = rachis::Access::resolveSource,
            .layoutBefore = rachis::TextureLayout::common,
            .layoutAfter = rachis::TextureLayout::resolveSource,
        },
    };
    auto duplicateBatch = device.resourceBarriers(*command, {.samplerFeedbackBarriers = duplicateFeedbackBarriers});
    REQUIRE_FALSE(duplicateBatch.has_value());
    CHECK(duplicateBatch.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device
                .samplerFeedbackBarrier(*command,
                                        {
                                            .feedbackMap = *feedbackMap,
                                            .stagesBefore = rachis::PipelineStage::all,
                                            .stagesAfter = rachis::PipelineStage::computeShader,
                                            .accessBefore = rachis::Access::common,
                                            .accessAfter = rachis::Access::unorderedAccess,
                                            .layoutBefore = rachis::TextureLayout::common,
                                            .layoutAfter = rachis::TextureLayout::unorderedAccess,
                                        })
                .has_value());
    for (const auto destination : {*writtenDecode, *clearedDecode, *encodedDecode})
    {
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = destination,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::resolve,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::resolveDestination,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::resolveDestination,
                                    })
                    .has_value());
    }

    REQUIRE(device.resetSamplerFeedbackMap(*command, *feedbackMap).has_value());
    REQUIRE(device
                .samplerFeedbackBarrier(*command,
                                        {
                                            .feedbackMap = *feedbackMap,
                                            .stagesBefore = rachis::PipelineStage::computeShader,
                                            .stagesAfter = rachis::PipelineStage::computeShader,
                                            .accessBefore = rachis::Access::unorderedAccess,
                                            .accessAfter = rachis::Access::unorderedAccess,
                                            .layoutBefore = rachis::TextureLayout::unorderedAccess,
                                            .layoutAfter = rachis::TextureLayout::unorderedAccess,
                                        })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .samplerFeedbackBarrier(*command,
                                        {
                                            .feedbackMap = *feedbackMap,
                                            .stagesBefore = rachis::PipelineStage::computeShader,
                                            .stagesAfter = rachis::PipelineStage::resolve,
                                            .accessBefore = rachis::Access::unorderedAccess,
                                            .accessAfter = rachis::Access::resolveSource,
                                            .layoutBefore = rachis::TextureLayout::unorderedAccess,
                                            .layoutAfter = rachis::TextureLayout::resolveSource,
                                        })
                .has_value());
    REQUIRE(device.decodeMinMipFeedback(*command, {.source = *feedbackMap, .destination = *writtenDecode}).has_value());

    REQUIRE(device
                .samplerFeedbackBarrier(*command,
                                        {
                                            .feedbackMap = *feedbackMap,
                                            .stagesBefore = rachis::PipelineStage::resolve,
                                            .stagesAfter = rachis::PipelineStage::computeShader,
                                            .accessBefore = rachis::Access::resolveSource,
                                            .accessAfter = rachis::Access::unorderedAccess,
                                            .layoutBefore = rachis::TextureLayout::resolveSource,
                                            .layoutAfter = rachis::TextureLayout::unorderedAccess,
                                        })
                .has_value());
    REQUIRE(device.resetSamplerFeedbackMap(*command, *feedbackMap).has_value());
    REQUIRE(device
                .samplerFeedbackBarrier(*command,
                                        {
                                            .feedbackMap = *feedbackMap,
                                            .stagesBefore = rachis::PipelineStage::computeShader,
                                            .stagesAfter = rachis::PipelineStage::resolve,
                                            .accessBefore = rachis::Access::unorderedAccess,
                                            .accessAfter = rachis::Access::resolveSource,
                                            .layoutBefore = rachis::TextureLayout::unorderedAccess,
                                            .layoutAfter = rachis::TextureLayout::resolveSource,
                                        })
                .has_value());
    REQUIRE(device.decodeMinMipFeedback(*command, {.source = *feedbackMap, .destination = *clearedDecode}).has_value());

    recordClearEncodeDecode(device, *command, *feedbackMap, *clearedDecode, *encodedDecode);
    recordDecodeReadbacks(device,
                          *command,
                          *writtenDecode,
                          *clearedDecode,
                          *encodedDecode,
                          *writtenReadback,
                          *clearedReadback,
                          *encodedReadback,
                          *transferLayout);

    const auto staleFeedbackMap = *feedbackMap;
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroySampler(*sampler).has_value());
    REQUIRE(device.destroyTextureView(*sampledView).has_value());
    REQUIRE(device.destroyTexture(*sampledTexture).has_value());
    REQUIRE(device.destroySamplerFeedbackMap(*feedbackMap).has_value());
    CHECK_FALSE(device.queryMinMipFeedbackMapProperties(staleFeedbackMap).has_value());
    CHECK_FALSE(device.destroySamplerFeedbackMap(staleFeedbackMap).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto written = device.mapBuffer(*writtenReadback);
    REQUIRE(written.has_value());
    std::uint32_t requestedCount = 0;
    for (std::uint32_t row = 0; row < transferLayout->rowCount; ++row)
    {
        const auto* rowBegin = reinterpret_cast<const std::uint8_t*>(written->data) +
                               static_cast<std::size_t>(row) * transferLayout->bytesPerRow;
        for (std::size_t column = 0; column < transferLayout->rowDataSize; ++column)
        {
            CAPTURE(row, column, rowBegin[column]);
            CHECK((rowBegin[column] == 0 || rowBegin[column] == noMipRequested));
            requestedCount += rowBegin[column] == 0 ? 1U : 0U;
        }
    }
    CHECK(requestedCount > 0);
    const auto* expectedRow = reinterpret_cast<const std::uint8_t*>(written->data) + transferLayout->bytesPerRow;
    CHECK(expectedRow[1] == 0);
    REQUIRE(device.unmapBuffer(*writtenReadback).has_value());

    checkNoMipRequested(device, *clearedReadback, *transferLayout);
    checkNoMipRequested(device, *encodedReadback, *transferLayout);

    REQUIRE(device.destroyTexture(*writtenDecode).has_value());
    REQUIRE(device.destroyTexture(*clearedDecode).has_value());
    REQUIRE(device.destroyTexture(*encodedDecode).has_value());
    REQUIRE(device.destroyTexture(*wrongExtentDecode).has_value());
    REQUIRE(device.destroyBuffer(*writtenReadback).has_value());
    REQUIRE(device.destroyBuffer(*clearedReadback).has_value());
    REQUIRE(device.destroyBuffer(*encodedReadback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("MinMip feedback decode and encode validate and retain sparse R8_UINT mappings",
          "[size:medium][rhi][sampler-feedback][sparse][hardware]")
{
    auto adapters = rachis::enumerateAdapters();
    if (!adapters)
    {
        INFO(adapters.error().message);
    }
    REQUIRE(adapters.has_value());

    std::optional<rachis::Device> selectedDevice;
    std::string selectedAdapter;
    for (const auto& adapter : *adapters)
    {
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .adapterPreference = adapter.type == rachis::AdapterType::software
                                     ? rachis::AdapterPreference::warp
                                     : rachis::AdapterPreference::highPerformance,
            .adapterId = adapter.id,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = true,
            .diagnosticCallback = &printDiagnostic,
        });
        if (created && rachis::test::requireCapabilities(*created).samplerFeedback &&
            rachis::test::requireCapabilities(*created).sparseTextures)
        {
            selectedAdapter = adapter.name;
            selectedDevice.emplace(std::move(*created));
            break;
        }
    }
    if (!selectedDevice)
    {
        SKIP("No D3D12 adapter reports both MinMip sampler feedback and sparse textures.");
    }
    INFO("Selected adapter: " << selectedAdapter);
    auto& device = *selectedDevice;

    auto pairedTexture = device.createTexture({
        .width = 1024,
        .height = 1024,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
        .label = "Sparse MinMip feedback paired texture",
    });
    REQUIRE(pairedTexture.has_value());
    auto feedbackMap = device.createMinMipFeedbackMap({
        .texture = *pairedTexture,
        .mipRegion = {4, 4, 1},
        .label = "Sparse MinMip feedback map",
    });
    REQUIRE(feedbackMap.has_value());
    auto properties = device.queryMinMipFeedbackMapProperties(*feedbackMap);
    REQUIRE(properties.has_value());
    const auto extent = properties->decodedExtent;

    auto sparse = device.createSparseTexture({
        .width = extent.width,
        .height = extent.height,
        .format = rachis::Format::r8Uint,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Sparse MinMip feedback decode",
    });
    REQUIRE(sparse.has_value());
    auto sparseProperties = device.querySparseTextureProperties(*sparse);
    REQUIRE(sparseProperties.has_value());
    auto heap = device.createResourceHeap({
        .size = sparseProperties->tileSize * sparseProperties->totalTileCount,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::nonAttachmentTexture,
        .flags = rachis::ResourceHeapFlag::sparseBacking,
        .label = "Sparse MinMip feedback decode heap",
    });
    REQUIRE(heap.has_value());
    auto transferLayout = device.queryTextureTransferLayout(*sparse, {.extent = extent});
    REQUIRE(transferLayout.has_value());
    auto readback = device.createBuffer({
        .size = transferLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Sparse MinMip feedback readback",
    });
    REQUIRE(readback.has_value());

    {
        auto unmappedCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(unmappedCommand.has_value());
        auto unmappedDecode =
            device.decodeMinMipFeedback(*unmappedCommand, {.source = *feedbackMap, .destination = *sparse});
        REQUIRE_FALSE(unmappedDecode.has_value());
        CHECK(unmappedDecode.error().code == rachis::ErrorCode::invalidState);
        auto unmappedEncode =
            device.encodeMinMipFeedback(*unmappedCommand, {.source = *sparse, .destination = *feedbackMap});
        REQUIRE_FALSE(unmappedEncode.has_value());
        CHECK(unmappedEncode.error().code == rachis::ErrorCode::invalidState);
    }

    std::vector<rachis::SparseTextureTileMapping> standardMappings;
    std::vector<rachis::SparseTextureTileMapping> standardUnmappings;
    std::uint32_t heapTileOffset = 0;
    for (std::uint32_t mipLevel = 0; mipLevel < sparseProperties->standardMipCount; ++mipLevel)
    {
        const auto grid = sparseProperties->standardMipTileGrids[mipLevel];
        standardMappings.push_back(rachis::SparseTextureTileMapping{
            .texture = *sparse,
            .subresource = {.mipLevel = mipLevel},
            .textureTileExtent = grid,
            .heap = *heap,
            .heapTileOffset = heapTileOffset,
        });
        standardUnmappings.push_back(rachis::SparseTextureTileMapping{
            .texture = *sparse,
            .subresource = {.mipLevel = mipLevel},
            .textureTileExtent = grid,
        });
        heapTileOffset += grid.width * grid.height * grid.depth;
    }
    std::vector<rachis::SparseTextureMipTailMapping> tailMappings;
    std::vector<rachis::SparseTextureMipTailMapping> tailUnmappings;
    if (sparseProperties->packedMipCount != 0)
    {
        tailMappings.push_back(rachis::SparseTextureMipTailMapping{
            .texture = *sparse,
            .heap = *heap,
            .heapTileOffset = heapTileOffset,
        });
        tailUnmappings.push_back(rachis::SparseTextureMipTailMapping{.texture = *sparse});
        heapTileOffset += sparseProperties->mipTailTileCount;
    }
    REQUIRE(heapTileOffset == sparseProperties->totalTileCount);
    const auto mappingPoint = device.enqueueSparseTextureMappings(
        rachis::QueueType::graphics,
        rachis::SparseTextureMappingBatch{.standardTiles = standardMappings, .mipTails = tailMappings});
    REQUIRE(mappingPoint.has_value());
    const rachis::SparseTextureMappingBatch unmapAll{.standardTiles = standardUnmappings, .mipTails = tailUnmappings};

    {
        auto retainingCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(retainingCommand.has_value());
        const auto beforeDecode = rachis::d3d12::test::commandDependencySnapshot(*retainingCommand);
        REQUIRE(beforeDecode.has_value());
        REQUIRE(device.decodeMinMipFeedback(*retainingCommand, {.source = *feedbackMap, .destination = *sparse})
                    .has_value());
        const auto afterDecode = rachis::d3d12::test::commandDependencySnapshot(*retainingCommand);
        REQUIRE(afterDecode.has_value());
        CHECK(afterDecode.value_or(rachis::d3d12::test::CommandDependencySnapshot{}).sparseMappingDependencies ==
              beforeDecode.value_or(rachis::d3d12::test::CommandDependencySnapshot{}).sparseMappingDependencies + 1);
        const auto remappedDuringDecode = device.enqueueSparseTextureMappings(rachis::QueueType::graphics, unmapAll);
        REQUIRE_FALSE(remappedDuringDecode.has_value());
        CHECK(remappedDuringDecode.error().code == rachis::ErrorCode::invalidState);
    }
    {
        auto retainingCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(retainingCommand.has_value());
        const auto beforeEncode = rachis::d3d12::test::commandDependencySnapshot(*retainingCommand);
        REQUIRE(beforeEncode.has_value());
        REQUIRE(device.encodeMinMipFeedback(*retainingCommand, {.source = *sparse, .destination = *feedbackMap})
                    .has_value());
        const auto afterEncode = rachis::d3d12::test::commandDependencySnapshot(*retainingCommand);
        REQUIRE(afterEncode.has_value());
        CHECK(afterEncode.value_or(rachis::d3d12::test::CommandDependencySnapshot{}).sparseMappingDependencies ==
              beforeEncode.value_or(rachis::d3d12::test::CommandDependencySnapshot{}).sparseMappingDependencies + 1);
        const auto remappedDuringEncode = device.enqueueSparseTextureMappings(rachis::QueueType::graphics, unmapAll);
        REQUIRE_FALSE(remappedDuringEncode.has_value());
        CHECK(remappedDuringEncode.error().code == rachis::ErrorCode::invalidState);
    }

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .samplerFeedbackBarrier(*command,
                                        {
                                            .feedbackMap = *feedbackMap,
                                            .stagesBefore = rachis::PipelineStage::all,
                                            .stagesAfter = rachis::PipelineStage::computeShader,
                                            .accessBefore = rachis::Access::common,
                                            .accessAfter = rachis::Access::unorderedAccess,
                                            .layoutBefore = rachis::TextureLayout::common,
                                            .layoutAfter = rachis::TextureLayout::unorderedAccess,
                                        })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *sparse,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::resolve,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::resolveDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::resolveDestination,
                                })
                .has_value());
    REQUIRE(device.resetSamplerFeedbackMap(*command, *feedbackMap).has_value());
    REQUIRE(device
                .samplerFeedbackBarrier(*command,
                                        {
                                            .feedbackMap = *feedbackMap,
                                            .stagesBefore = rachis::PipelineStage::computeShader,
                                            .stagesAfter = rachis::PipelineStage::resolve,
                                            .accessBefore = rachis::Access::unorderedAccess,
                                            .accessAfter = rachis::Access::resolveSource,
                                            .layoutBefore = rachis::TextureLayout::unorderedAccess,
                                            .layoutAfter = rachis::TextureLayout::resolveSource,
                                        })
                .has_value());
    REQUIRE(device.decodeMinMipFeedback(*command, {.source = *feedbackMap, .destination = *sparse}).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *sparse,
                                    .stagesBefore = rachis::PipelineStage::resolve,
                                    .stagesAfter = rachis::PipelineStage::resolve,
                                    .accessBefore = rachis::Access::resolveDestination,
                                    .accessAfter = rachis::Access::resolveSource,
                                    .layoutBefore = rachis::TextureLayout::resolveDestination,
                                    .layoutAfter = rachis::TextureLayout::resolveSource,
                                })
                .has_value());
    REQUIRE(device
                .samplerFeedbackBarrier(*command,
                                        {
                                            .feedbackMap = *feedbackMap,
                                            .stagesBefore = rachis::PipelineStage::resolve,
                                            .stagesAfter = rachis::PipelineStage::resolve,
                                            .accessBefore = rachis::Access::resolveSource,
                                            .accessAfter = rachis::Access::resolveDestination,
                                            .layoutBefore = rachis::TextureLayout::resolveSource,
                                            .layoutAfter = rachis::TextureLayout::resolveDestination,
                                        })
                .has_value());
    REQUIRE(device.encodeMinMipFeedback(*command, {.source = *sparse, .destination = *feedbackMap}).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *sparse,
                                    .stagesBefore = rachis::PipelineStage::resolve,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::resolveSource,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::resolveSource,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device
                .copyTextureToBuffer(*command,
                                     {
                                         .buffer = *readback,
                                         .bytesPerRow = transferLayout->bytesPerRow,
                                         .rowsPerImage = transferLayout->rowsPerImage,
                                         .texture = *sparse,
                                         .extent = extent,
                                     })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();
    checkNoMipRequested(device, *readback, *transferLayout);

    const auto unmapped = device.enqueueSparseTextureMappings(rachis::QueueType::graphics, unmapAll);
    REQUIRE(unmapped.has_value());
    REQUIRE(device.wait(*unmapped).has_value());
    device.collectGarbage();

    REQUIRE(device.destroySamplerFeedbackMap(*feedbackMap).has_value());
    REQUIRE(device.destroyTexture(*pairedTexture).has_value());
    REQUIRE(device.destroyTexture(*sparse).has_value());
    REQUIRE(device.destroyResourceHeap(*heap).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
