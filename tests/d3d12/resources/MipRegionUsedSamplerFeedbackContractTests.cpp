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

    constexpr std::uint32_t sourceWidth = 68;
    constexpr std::uint32_t sourceHeight = 52;
    constexpr std::uint32_t sourceMipLevels = 7;
    constexpr rachis::Extent3D feedbackRegion{8, 8, 1};
    constexpr rachis::Extent3D decodedExtent{64, 8, 1};
    constexpr rachis::Extent3D observedMipDataExtent{3, 2, 1};
    constexpr std::uint32_t observedMipLevel = 2;

    [[nodiscard]] constexpr std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment) noexcept
    {
        return (value + alignment - 1) / alignment * alignment;
    }

    [[nodiscard]] constexpr std::uint32_t logicalExtent(std::uint32_t source,
                                                        std::uint32_t region,
                                                        std::uint32_t mip) noexcept
    {
        const auto mipExtent = std::max(1U, source >> mip);
        return (mipExtent + region - 1U) / region;
    }

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
        std::ifstream stream(RACHIS_TEST_MIP_REGION_USED_SAMPLER_FEEDBACK_SHADER_PATH,
                             std::ios::binary | std::ios::ate);
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
            .width = sourceWidth,
            .height = sourceHeight,
            .mipLevels = sourceMipLevels,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::sampled,
            .label = "MipRegionUsed paired texture",
        };
    }

    [[nodiscard]] rachis::TextureDesc decodedTextureDesc(std::string_view label)
    {
        return rachis::TextureDesc{
            .width = decodedExtent.width,
            .height = decodedExtent.height,
            .mipLevels = sourceMipLevels,
            .format = rachis::Format::r8Uint,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
            .label = label,
        };
    }

    [[nodiscard]] std::optional<rachis::Device> samplerFeedbackDevice(std::string& selectedAdapter)
    {
        auto adapters = rachis::enumerateAdapters();
        if (!adapters)
        {
            return std::nullopt;
        }
        std::ranges::stable_sort(
            *adapters, {}, [](const auto& adapter) { return adapter.type == rachis::AdapterType::software; });
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
                return std::move(*created);
            }
        }
        return std::nullopt;
    }

    void checkDecodedBytes(rachis::Device& device,
                           rachis::BufferHandle buffer,
                           const rachis::TextureTransferLayout& layout,
                           bool expectUsed)
    {
        auto mapped = device.mapBuffer(buffer);
        REQUIRE(mapped.has_value());
        std::uint32_t usedCount = 0;
        bool farQuadrantUsed = false;
        for (std::uint32_t row = 0; row < layout.rowCount; ++row)
        {
            const auto* rowBegin = reinterpret_cast<const std::uint8_t*>(mapped->data) +
                                   static_cast<std::size_t>(row) * layout.bytesPerRow;
            for (std::size_t column = 0; column < layout.rowDataSize; ++column)
            {
                CAPTURE(row, column, rowBegin[column]);
                CHECK((rowBegin[column] == 0 || rowBegin[column] == 0xff));
                usedCount += rowBegin[column] == 0xff ? 1U : 0U;
                farQuadrantUsed = farQuadrantUsed || (rowBegin[column] == 0xff && row >= layout.rowCount / 2U &&
                                                      column >= layout.rowDataSize / 2U);
            }
        }
        CHECK((usedCount > 0) == expectUsed);
        if (expectUsed)
        {
            CHECK(farQuadrantUsed);
        }
        REQUIRE(device.unmapBuffer(buffer).has_value());
    }

    void checkTranscodedBytes(rachis::Device& device,
                              rachis::BufferHandle source,
                              rachis::BufferHandle roundTrip,
                              std::span<const rachis::BufferTextureCopyRegion> regions,
                              std::span<const rachis::TextureTransferLayout> layouts,
                              bool requireExactTransitivity)
    {
        auto sourceMemory = device.mapBuffer(source);
        auto roundTripMemory = device.mapBuffer(roundTrip);
        REQUIRE(sourceMemory.has_value());
        REQUIRE(roundTripMemory.has_value());
        REQUIRE(regions.size() == layouts.size());
        for (std::size_t mip = 0; mip < layouts.size(); ++mip)
        {
            const auto& layout = layouts[mip];
            for (std::uint32_t row = 0; row < layout.rowCount; ++row)
            {
                const auto rowOffset = regions[mip].bufferOffset + static_cast<std::uint64_t>(row) * layout.bytesPerRow;
                const auto* sourceRow = reinterpret_cast<const std::uint8_t*>(sourceMemory->data) + rowOffset;
                const auto* roundTripRow = reinterpret_cast<const std::uint8_t*>(roundTripMemory->data) + rowOffset;
                for (std::uint64_t column = 0; column < layout.rowDataSize; ++column)
                {
                    CAPTURE(mip, row, column, sourceRow[column], roundTripRow[column]);
                    CHECK((sourceRow[column] == 0 || sourceRow[column] == 0xff));
                    CHECK((roundTripRow[column] == 0 || roundTripRow[column] == 0xff));
                    if (requireExactTransitivity || sourceRow[column] == 0xff)
                    {
                        CHECK(roundTripRow[column] == sourceRow[column]);
                    }
                }
            }
        }
        REQUIRE(device.unmapBuffer(roundTrip).has_value());
        REQUIRE(device.unmapBuffer(source).has_value());
    }

    struct FarCellUpload final
    {
        rachis::BufferHandle buffer;
        rachis::BufferTextureCopyRegion region;
    };

    [[nodiscard]] FarCellUpload createFarCellUpload(rachis::Device& device, rachis::TextureHandle texture)
    {
        const rachis::BufferTextureCopyRegion request{
            .textureSubresource = {.mipLevel = observedMipLevel},
            .textureOrigin = {observedMipDataExtent.width - 1U, observedMipDataExtent.height - 1U, 0},
            .extent = {1, 1, 1},
        };
        auto layout = device.queryTextureTransferLayout(texture, request);
        REQUIRE(layout.has_value());
        auto upload = device.createBuffer({
            .size = layout->requiredBufferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "MipRegionUsed far-cell upload",
        });
        REQUIRE(upload.has_value());
        auto mapped = device.mapBuffer(*upload);
        REQUIRE(mapped.has_value());
        std::fill_n(mapped->data, mapped->size, std::byte{0});
        mapped->data[request.bufferOffset] = std::byte{0xff};
        REQUIRE(device.unmapBuffer(*upload).has_value());
        return FarCellUpload{
            .buffer = *upload,
            .region =
                {
                    .bufferOffset = request.bufferOffset,
                    .bytesPerRow = layout->bytesPerRow,
                    .rowsPerImage = layout->rowsPerImage,
                    .textureSubresource = request.textureSubresource,
                    .textureOrigin = request.textureOrigin,
                    .extent = request.extent,
                },
        };
    }

    void recordTranscodeRoundTrip(rachis::Device& device,
                                  rachis::CommandBuffer& command,
                                  rachis::TextureHandle sourceDecode,
                                  const FarCellUpload& farCellUpload,
                                  rachis::SamplerFeedbackMapHandle feedbackMap,
                                  rachis::TextureHandle destinationDecode)
    {
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = sourceDecode,
                                        .stagesBefore = rachis::PipelineStage::resolve,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::resolveDestination,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutBefore = rachis::TextureLayout::resolveDestination,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                    })
                    .has_value());
        REQUIRE(
            device.copyBufferToTexture(command, farCellUpload.buffer, sourceDecode, std::span{&farCellUpload.region, 1})
                .has_value());
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = sourceDecode,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::resolve,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::resolveSource,
                                        .layoutBefore = rachis::TextureLayout::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::resolveSource,
                                    })
                    .has_value());
        REQUIRE(device
                    .samplerFeedbackBarrier(command,
                                            {
                                                .feedbackMap = feedbackMap,
                                                .stagesBefore = rachis::PipelineStage::all,
                                                .stagesAfter = rachis::PipelineStage::resolve,
                                                .accessBefore = rachis::Access::common,
                                                .accessAfter = rachis::Access::resolveDestination,
                                                .layoutBefore = rachis::TextureLayout::common,
                                                .layoutAfter = rachis::TextureLayout::resolveDestination,
                                            })
                    .has_value());
        REQUIRE(device.encodeMipRegionUsedFeedback(command, {.source = sourceDecode, .destination = feedbackMap})
                    .has_value());
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
        REQUIRE(device.decodeMipRegionUsedFeedback(command, {.source = feedbackMap, .destination = destinationDecode})
                    .has_value());
    }

    void recordMipRegionReadbacks(rachis::Device& device,
                                  rachis::CommandBuffer& command,
                                  rachis::TextureHandle writtenDecode,
                                  rachis::TextureHandle clearedDecode,
                                  rachis::TextureHandle roundTripDecode,
                                  rachis::BufferHandle writtenReadback,
                                  rachis::BufferHandle clearedReadback,
                                  rachis::BufferHandle transcodeSourceReadback,
                                  rachis::BufferHandle transcodeRoundTripReadback,
                                  const rachis::TextureTransferLayout& observedLayout,
                                  std::span<const rachis::BufferTextureCopyRegion> transcodeRegions)
    {
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = writtenDecode,
                                        .stagesBefore = rachis::PipelineStage::resolve,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::resolveSource,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::resolveSource,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        for (const auto destination : {clearedDecode, roundTripDecode})
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

        const auto copyObservedMip = [&](rachis::TextureHandle texture, rachis::BufferHandle buffer) {
            return device.copyTextureToBuffer(command,
                                              {
                                                  .buffer = buffer,
                                                  .bytesPerRow = observedLayout.bytesPerRow,
                                                  .rowsPerImage = observedLayout.rowsPerImage,
                                                  .texture = texture,
                                                  .textureSubresource = {.mipLevel = observedMipLevel},
                                                  .extent = observedMipDataExtent,
                                              });
        };
        REQUIRE(copyObservedMip(writtenDecode, writtenReadback).has_value());
        REQUIRE(copyObservedMip(clearedDecode, clearedReadback).has_value());
        REQUIRE(
            device.copyTextureToBuffer(command, writtenDecode, transcodeSourceReadback, transcodeRegions).has_value());
        REQUIRE(device.copyTextureToBuffer(command, roundTripDecode, transcodeRoundTripReadback, transcodeRegions)
                    .has_value());
    }

    void checkInvalidTranscodeRequests(rachis::Device& device,
                                       rachis::CommandBuffer& graphicsCommand,
                                       rachis::SamplerFeedbackMapHandle feedbackMap,
                                       rachis::SamplerFeedbackMapHandle minMipMap,
                                       rachis::SamplerFeedbackMapHandle transcodeDestination,
                                       rachis::TextureHandle wrongShape,
                                       rachis::TextureHandle validShape)
    {
        auto invalidDestination =
            device.decodeMipRegionUsedFeedback(graphicsCommand, {.source = feedbackMap, .destination = wrongShape});
        REQUIRE_FALSE(invalidDestination.has_value());
        CHECK(invalidDestination.error().code == rachis::ErrorCode::invalidArgument);
        CHECK_FALSE(device.decodeMinMipFeedback(graphicsCommand, {.source = feedbackMap, .destination = wrongShape})
                        .has_value());
        CHECK_FALSE(
            device.decodeMipRegionUsedFeedback(graphicsCommand, {.source = minMipMap, .destination = wrongShape})
                .has_value());
        CHECK_FALSE(
            device.encodeMipRegionUsedFeedback(graphicsCommand, {.source = wrongShape, .destination = feedbackMap})
                .has_value());
        CHECK_FALSE(device.encodeMinMipFeedback(graphicsCommand, {.source = validShape, .destination = feedbackMap})
                        .has_value());
        CHECK_FALSE(
            device.encodeMipRegionUsedFeedback(graphicsCommand, {.source = validShape, .destination = minMipMap})
                .has_value());

        auto computeCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(computeCommand.has_value());
        auto invalidEncode = device.encodeMipRegionUsedFeedback(
            *computeCommand, {.source = validShape, .destination = transcodeDestination});
        REQUIRE_FALSE(invalidEncode.has_value());
        CHECK(invalidEncode.error().code == rachis::ErrorCode::invalidArgument);
    }

} // namespace

TEST_CASE("MipRegionUsed feedback preserves per-mip observations and rejects mode confusion",
          "[size:medium][rhi][sampler-feedback][compute][hardware]")
{
    std::string selectedAdapter;
    auto selectedDevice = samplerFeedbackDevice(selectedAdapter);
    if (!selectedDevice)
    {
        SKIP("No D3D12 adapter reports Tier 1.0 sampler-feedback support.");
    }
    INFO("Selected adapter: " << selectedAdapter);
    auto& device = *selectedDevice;

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    const std::array shaderBindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::sampledTexture},
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::mipRegionUsedFeedback},
        rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::sampler},
    };
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "MipRegionUsed feedback shader",
        .shaderInterface =
            rachis::ShaderInterfaceDesc{
                .bindings = shaderBindings,
                .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1},
            },
    });
    REQUIRE(shader.has_value());

    auto sampledTexture = device.createTexture(sampledTextureDesc());
    REQUIRE(sampledTexture.has_value());
    auto invalidRegion = device.createMipRegionUsedFeedbackMap({.texture = *sampledTexture, .mipRegion = {6, 8, 1}});
    REQUIRE_FALSE(invalidRegion.has_value());
    CHECK(invalidRegion.error().code == rachis::ErrorCode::invalidArgument);

    auto feedbackMap = device.createMipRegionUsedFeedbackMap({
        .texture = *sampledTexture,
        .mipRegion = feedbackRegion,
        .label = "MipRegionUsed feedback map",
    });
    auto minMipMap = device.createMinMipFeedbackMap({
        .texture = *sampledTexture,
        .mipRegion = feedbackRegion,
        .label = "Mode-confusion MinMip map",
    });
    auto transcodedFeedbackMap = device.createMipRegionUsedFeedbackMap({
        .texture = *sampledTexture,
        .mipRegion = feedbackRegion,
        .label = "MipRegionUsed transcoded feedback map",
    });
    REQUIRE(feedbackMap.has_value());
    REQUIRE(minMipMap.has_value());
    REQUIRE(transcodedFeedbackMap.has_value());

    auto properties = device.queryMipRegionUsedFeedbackMapProperties(*feedbackMap);
    REQUIRE(properties.has_value());
    CHECK(properties->mipRegion.width == feedbackRegion.width);
    CHECK(properties->mipRegion.height == feedbackRegion.height);
    CHECK(properties->decodedExtent.width == decodedExtent.width);
    CHECK(properties->decodedExtent.height == decodedExtent.height);
    CHECK(properties->pairedExtent.width == sourceWidth);
    CHECK(properties->pairedExtent.height == sourceHeight);
    CHECK(properties->mipLevels == sourceMipLevels);
    CHECK_FALSE(device.queryMinMipFeedbackMapProperties(*feedbackMap).has_value());
    CHECK_FALSE(device.queryMipRegionUsedFeedbackMapProperties(*minMipMap).has_value());

    auto writtenDecode = device.createTexture(decodedTextureDesc("Written MipRegionUsed decode"));
    auto clearedDecode = device.createTexture(decodedTextureDesc("Cleared MipRegionUsed decode"));
    auto roundTripDecode = device.createTexture(decodedTextureDesc("Round-trip MipRegionUsed decode"));
    auto wrongMipCountDecode = device.createTexture({
        .width = decodedExtent.width,
        .height = decodedExtent.height,
        .format = rachis::Format::r8Uint,
        .usage = rachis::TextureUsage::copyDestination,
        .label = "Invalid MipRegionUsed decode",
    });
    REQUIRE(writtenDecode.has_value());
    REQUIRE(clearedDecode.has_value());
    REQUIRE(roundTripDecode.has_value());
    REQUIRE(wrongMipCountDecode.has_value());

    std::vector<rachis::BufferTextureCopyRegion> transcodeRegions;
    std::vector<rachis::TextureTransferLayout> transcodeLayouts;
    std::uint64_t nextBufferOffset = 0;
    for (std::uint32_t mip = 0; mip < sourceMipLevels; ++mip)
    {
        const rachis::BufferTextureCopyRegion request{
            .bufferOffset = nextBufferOffset,
            .textureSubresource = {.mipLevel = mip},
            .extent =
                {
                    logicalExtent(sourceWidth, feedbackRegion.width, mip),
                    logicalExtent(sourceHeight, feedbackRegion.height, mip),
                    1,
                },
        };
        auto layout = device.queryTextureTransferLayout(*writtenDecode, request);
        REQUIRE(layout.has_value());
        transcodeRegions.push_back(rachis::BufferTextureCopyRegion{
            .bufferOffset = request.bufferOffset,
            .bytesPerRow = layout->bytesPerRow,
            .rowsPerImage = layout->rowsPerImage,
            .textureSubresource = request.textureSubresource,
            .extent = request.extent,
        });
        transcodeLayouts.push_back(*layout);
        nextBufferOffset = alignUp(layout->requiredBufferSize, 512);
    }
    REQUIRE_FALSE(transcodeLayouts.empty());
    auto transcodeSourceReadback = device.createBuffer({
        .size = transcodeLayouts.back().requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "MipRegionUsed transcode source readback",
    });
    auto transcodeRoundTripReadback = device.createBuffer({
        .size = transcodeLayouts.back().requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "MipRegionUsed transcode round-trip readback",
    });
    REQUIRE(transcodeSourceReadback.has_value());
    REQUIRE(transcodeRoundTripReadback.has_value());
    const auto farCellUpload = createFarCellUpload(device, *writtenDecode);

    const rachis::BufferTextureCopyRegion observedMipRegion{
        .textureSubresource = {.mipLevel = observedMipLevel},
        .extent = observedMipDataExtent,
    };
    auto transferLayout = device.queryTextureTransferLayout(*writtenDecode, observedMipRegion);
    REQUIRE(transferLayout.has_value());
    CHECK(transferLayout->rowDataSize == observedMipDataExtent.width);
    CHECK(transferLayout->rowCount == observedMipDataExtent.height);
    auto writtenReadback = device.createBuffer({
        .size = transferLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Written MipRegionUsed readback",
    });
    auto clearedReadback = device.createBuffer({
        .size = transferLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Cleared MipRegionUsed readback",
    });
    REQUIRE(writtenReadback.has_value());
    REQUIRE(clearedReadback.has_value());

    auto sampledView = device.createTextureView({
        .texture = *sampledTexture,
        .usage = rachis::TextureViewUsage::sampled,
        .label = "MipRegionUsed paired view",
    });
    auto sampler = device.createSampler({
        .mipFilter = rachis::MipmapFilterMode::nearest,
        .label = "MipRegionUsed sampler",
    });
    REQUIRE(sampledView.has_value());
    REQUIRE(sampler.has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{1, rachis::BindingType::mipRegionUsedFeedback, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{2, rachis::BindingType::sampler, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = bindingEntries, .label = "MipRegionUsed layout"});
    REQUIRE(bindingLayout.has_value());
    const std::array writes{
        rachis::BindingWrite{.binding = 0, .resource = *sampledView},
        rachis::BindingWrite{.binding = 1, .resource = *feedbackMap},
        rachis::BindingWrite{.binding = 2, .resource = *sampler},
    };
    auto bindingSet = rachis::test::createBindingSet(device, {.layout = *bindingLayout, .writes = writes});
    REQUIRE(bindingSet.has_value());

    const std::array minMipEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{1, rachis::BindingType::minMipFeedback, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{2, rachis::BindingType::sampler, rachis::ShaderStage::compute},
    };
    auto minMipBindingLayout = device.createBindingLayout({.entries = minMipEntries});
    REQUIRE(minMipBindingLayout.has_value());
    auto mismatchedBindingSet =
        rachis::test::createBindingSet(device, {.layout = *minMipBindingLayout, .writes = writes});
    REQUIRE_FALSE(mismatchedBindingSet.has_value());
    CHECK(mismatchedBindingSet.error().code == rachis::ErrorCode::invalidArgument);

    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = bindingLayouts});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline.has_value());
    const std::array minMipBindingLayouts{*minMipBindingLayout};
    auto minMipPipelineLayout = device.createPipelineLayout({.bindingLayouts = minMipBindingLayouts});
    REQUIRE(minMipPipelineLayout.has_value());
    auto mismatchedPipeline = device.createComputePipeline({.layout = *minMipPipelineLayout, .computeShader = *shader});
    REQUIRE_FALSE(mismatchedPipeline.has_value());
    CHECK(mismatchedPipeline.error().code == rachis::ErrorCode::validationError);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    checkInvalidTranscodeRequests(
        device, *command, *feedbackMap, *minMipMap, *transcodedFeedbackMap, *wrongMipCountDecode, *writtenDecode);

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
    for (const auto destination : {*writtenDecode, *clearedDecode, *roundTripDecode})
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
    REQUIRE(device.decodeMipRegionUsedFeedback(*command, {.source = *feedbackMap, .destination = *writtenDecode})
                .has_value());
    recordTranscodeRoundTrip(device, *command, *writtenDecode, farCellUpload, *transcodedFeedbackMap, *roundTripDecode);
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
    REQUIRE(device.decodeMipRegionUsedFeedback(*command, {.source = *feedbackMap, .destination = *clearedDecode})
                .has_value());
    recordMipRegionReadbacks(device,
                             *command,
                             *writtenDecode,
                             *clearedDecode,
                             *roundTripDecode,
                             *writtenReadback,
                             *clearedReadback,
                             *transcodeSourceReadback,
                             *transcodeRoundTripReadback,
                             *transferLayout,
                             transcodeRegions);

    const auto staleFeedbackMap = *feedbackMap;
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyPipelineLayout(*minMipPipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*minMipBindingLayout).has_value());
    REQUIRE(device.destroySampler(*sampler).has_value());
    REQUIRE(device.destroyTextureView(*sampledView).has_value());
    REQUIRE(device.destroyTexture(*sampledTexture).has_value());
    REQUIRE(device.destroySamplerFeedbackMap(*feedbackMap).has_value());
    REQUIRE(device.destroySamplerFeedbackMap(*minMipMap).has_value());
    REQUIRE(device.destroySamplerFeedbackMap(*transcodedFeedbackMap).has_value());
    REQUIRE(device.destroyBuffer(farCellUpload.buffer).has_value());
    CHECK_FALSE(device.queryMipRegionUsedFeedbackMapProperties(staleFeedbackMap).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();
    checkDecodedBytes(device, *writtenReadback, *transferLayout, true);
    checkDecodedBytes(device, *clearedReadback, *transferLayout, false);

    checkTranscodedBytes(device,
                         *transcodeSourceReadback,
                         *transcodeRoundTripReadback,
                         transcodeRegions,
                         transcodeLayouts,
                         rachis::test::requireAdapterInfo(device).type != rachis::AdapterType::software);

    REQUIRE(device.destroyTexture(*writtenDecode).has_value());
    REQUIRE(device.destroyTexture(*clearedDecode).has_value());
    REQUIRE(device.destroyTexture(*roundTripDecode).has_value());
    REQUIRE(device.destroyTexture(*wrongMipCountDecode).has_value());
    REQUIRE(device.destroyBuffer(*writtenReadback).has_value());
    REQUIRE(device.destroyBuffer(*clearedReadback).has_value());
    REQUIRE(device.destroyBuffer(*transcodeSourceReadback).has_value());
    REQUIRE(device.destroyBuffer(*transcodeRoundTripReadback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
