#include "D3D12/D3D12TestHooks.h"
#include "HeapAllocationCounter.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string_view>

namespace
{

    class ForceLegacyBarriers final
    {
    public:
        explicit ForceLegacyBarriers(bool enabled = true)
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

    constexpr std::uint64_t bufferSize = 256;
    constexpr std::uint32_t textureWidth = 4;
    constexpr std::uint32_t textureHeight = 4;
    constexpr std::uint32_t textureBytesPerPixel = 4;
    constexpr std::uint32_t textureRowPitch = 256;
    constexpr std::uint64_t textureTransferSize = static_cast<std::uint64_t>(textureRowPitch) * textureHeight;

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    [[nodiscard]] constexpr std::byte bufferValue(std::uint32_t resource, std::uint32_t index) noexcept
    {
        return static_cast<std::byte>((resource * 67U + index * 13U + 5U) & 0xFFU);
    }

    [[nodiscard]] constexpr std::byte textureValue(std::uint32_t x, std::uint32_t y, std::uint32_t channel) noexcept
    {
        return static_cast<std::byte>((x * 29U + y * 43U + channel * 71U + 11U) & 0xFFU);
    }

    void initializeUploads(rachis::Device& device,
                           rachis::BufferHandle bufferUpload,
                           rachis::BufferHandle textureUpload)
    {
        auto buffers = device.mapBuffer(bufferUpload);
        REQUIRE(buffers.has_value());
        for (std::uint32_t resource = 0; resource < 2; ++resource)
        {
            for (std::uint32_t index = 0; index < bufferSize; ++index)
            {
                buffers->data[resource * bufferSize + index] = bufferValue(resource, index);
            }
        }
        REQUIRE(device.unmapBuffer(bufferUpload).has_value());

        auto texture = device.mapBuffer(textureUpload);
        REQUIRE(texture.has_value());
        std::memset(texture->data, 0, static_cast<std::size_t>(textureTransferSize));
        for (std::uint32_t y = 0; y < textureHeight; ++y)
        {
            for (std::uint32_t x = 0; x < textureWidth; ++x)
            {
                for (std::uint32_t channel = 0; channel < textureBytesPerPixel; ++channel)
                {
                    texture->data[y * textureRowPitch + x * textureBytesPerPixel + channel] =
                        textureValue(x, y, channel);
                }
            }
        }
        REQUIRE(device.unmapBuffer(textureUpload).has_value());
    }

    void verifyBuffer(rachis::Device& device, rachis::BufferHandle readback, std::uint32_t resource)
    {
        auto mapped = device.mapBuffer(readback);
        REQUIRE(mapped.has_value());
        std::uint32_t firstMismatch = static_cast<std::uint32_t>(bufferSize);
        for (std::uint32_t index = 0; index < bufferSize; ++index)
        {
            if (mapped->data[index] != bufferValue(resource, index))
            {
                firstMismatch = index;
                break;
            }
        }
        CAPTURE(resource, firstMismatch);
        CHECK(firstMismatch == bufferSize);
        REQUIRE(device.unmapBuffer(readback).has_value());
    }

    void verifyTexture(rachis::Device& device, rachis::BufferHandle readback)
    {
        auto mapped = device.mapBuffer(readback);
        REQUIRE(mapped.has_value());
        std::uint32_t firstMismatch = textureWidth * textureHeight * textureBytesPerPixel;
        for (std::uint32_t y = 0;
             y < textureHeight && firstMismatch == textureWidth * textureHeight * textureBytesPerPixel;
             ++y)
        {
            for (std::uint32_t x = 0; x < textureWidth; ++x)
            {
                for (std::uint32_t channel = 0; channel < textureBytesPerPixel; ++channel)
                {
                    if (mapped->data[y * textureRowPitch + x * textureBytesPerPixel + channel] !=
                        textureValue(x, y, channel))
                    {
                        firstMismatch = (y * textureWidth + x) * textureBytesPerPixel + channel;
                        break;
                    }
                }
            }
        }
        CAPTURE(firstMismatch);
        CHECK(firstMismatch == textureWidth * textureHeight * textureBytesPerPixel);
        REQUIRE(device.unmapBuffer(readback).has_value());
    }

    void verifySolidColor(rachis::Device& device,
                          rachis::BufferHandle readback,
                          const std::array<std::uint8_t, textureBytesPerPixel>& expected)
    {
        auto mapped = device.mapBuffer(readback);
        REQUIRE(mapped.has_value());
        std::uint32_t mismatches = 0;
        for (std::uint32_t y = 0; y < textureHeight; ++y)
        {
            for (std::uint32_t x = 0; x < textureWidth; ++x)
            {
                for (std::uint32_t channel = 0; channel < textureBytesPerPixel; ++channel)
                {
                    if (std::to_integer<std::uint8_t>(
                            mapped->data[y * textureRowPitch + x * textureBytesPerPixel + channel]) !=
                        expected[channel])
                    {
                        ++mismatches;
                    }
                }
            }
        }
        CHECK(mismatches == 0);
        REQUIRE(device.unmapBuffer(readback).has_value());
    }

    struct LegacyDiscardTargets final
    {
        rachis::TextureHandle color = {};
        rachis::TextureHandle depth = {};
        rachis::TextureHandle uploaded = {};
        rachis::TextureViewHandle colorView = {};
        rachis::TextureViewHandle depthView = {};
        rachis::BufferHandle upload = {};
    };

    [[nodiscard]] rachis::TextureBarrier discardColorTarget(const LegacyDiscardTargets& targets)
    {
        return rachis::TextureBarrier{
            .texture = targets.color,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::renderTarget,
            .accessAfter = rachis::Access::renderTarget,
            .layoutBefore = rachis::TextureLayout::undefined,
            .layoutAfter = rachis::TextureLayout::renderTarget,
        };
    }

    [[nodiscard]] rachis::TextureBarrier discardUploadedTexture(const LegacyDiscardTargets& targets)
    {
        return rachis::TextureBarrier{
            .texture = targets.uploaded,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessAfter = rachis::Access::copyDestination,
            .layoutBefore = rachis::TextureLayout::undefined,
            .layoutAfter = rachis::TextureLayout::copyDestination,
        };
    }

    void clearColorTarget(rachis::Device& device,
                          rachis::CommandBuffer& command,
                          const LegacyDiscardTargets& targets,
                          const rachis::ClearColor& color)
    {
        const std::array colorAttachments{
            rachis::ColorAttachment{
                .view = targets.colorView,
                .loadOperation = rachis::LoadOperation::clear,
                .storeOperation = rachis::StoreOperation::store,
                .clearValue = color,
            },
        };
        REQUIRE(device.beginRendering(command, {.colorAttachments = colorAttachments}).has_value());
        REQUIRE(device.endRendering(command).has_value());
    }

    void uploadTexture(rachis::Device& device, rachis::CommandBuffer& command, const LegacyDiscardTargets& targets)
    {
        REQUIRE(device
                    .copyBufferToTexture(command,
                                         {
                                             .buffer = targets.upload,
                                             .bytesPerRow = textureRowPitch,
                                             .rowsPerImage = textureHeight,
                                             .texture = targets.uploaded,
                                             .extent = {textureWidth, textureHeight, 1},
                                         })
                    .has_value());
    }

    void readTexture(rachis::Device& device,
                     rachis::CommandBuffer& command,
                     rachis::TextureHandle texture,
                     rachis::BufferHandle readback)
    {
        REQUIRE(device
                    .copyTextureToBuffer(command,
                                         {
                                             .buffer = readback,
                                             .bytesPerRow = textureRowPitch,
                                             .rowsPerImage = textureHeight,
                                             .texture = texture,
                                             .extent = {textureWidth, textureHeight, 1},
                                         })
                    .has_value());
    }

    [[nodiscard]] rachis::TextureBarrier toCopySource(rachis::TextureHandle texture,
                                                      rachis::PipelineStage stagesBefore,
                                                      rachis::Access accessBefore,
                                                      rachis::TextureLayout layoutBefore)
    {
        return rachis::TextureBarrier{
            .texture = texture,
            .stagesBefore = stagesBefore,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = accessBefore,
            .accessAfter = rachis::Access::copySource,
            .layoutBefore = layoutBefore,
            .layoutAfter = rachis::TextureLayout::copySource,
        };
    }

    void recordLegacyDiscardFrame(rachis::Device& device,
                                  rachis::CommandBuffer& command,
                                  const LegacyDiscardTargets& targets,
                                  const rachis::ClearColor& color)
    {
        const std::array discards{
            discardColorTarget(targets),
            rachis::TextureBarrier{
                .texture = targets.depth,
                .stagesBefore = rachis::PipelineStage::all,
                .stagesAfter = rachis::PipelineStage::depthStencil,
                .accessAfter = rachis::Access::depthStencilWrite,
                .layoutBefore = rachis::TextureLayout::undefined,
                .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                .subresources = {.aspect = rachis::TextureAspect::depth},
            },
            discardUploadedTexture(targets),
        };
        REQUIRE(device.resourceBarriers(command, {.textureBarriers = discards}).has_value());
        uploadTexture(device, command, targets);
        const std::array colorAttachments{
            rachis::ColorAttachment{
                .view = targets.colorView,
                .loadOperation = rachis::LoadOperation::clear,
                .storeOperation = rachis::StoreOperation::store,
                .clearValue = color,
            },
        };
        REQUIRE(device
                    .beginRendering(command,
                                    {
                                        .colorAttachments = colorAttachments,
                                        .depthStencilAttachment =
                                            rachis::DepthStencilAttachment{
                                                .view = targets.depthView,
                                                .depthLoadOperation = rachis::LoadOperation::clear,
                                                .depthStoreOperation = rachis::StoreOperation::store,
                                                .depthClearValue = 0.5F,
                                            },
                                    })
                    .has_value());
        REQUIRE(device.endRendering(command).has_value());
        const std::array reads{
            rachis::TextureBarrier{
                .texture = targets.color,
                .stagesBefore = rachis::PipelineStage::renderTarget,
                .stagesAfter = rachis::PipelineStage::pixelShader,
                .accessBefore = rachis::Access::renderTarget,
                .accessAfter = rachis::Access::shaderResource,
                .layoutBefore = rachis::TextureLayout::renderTarget,
                .layoutAfter = rachis::TextureLayout::shaderResource,
            },
            rachis::TextureBarrier{
                .texture = targets.depth,
                .stagesBefore = rachis::PipelineStage::depthStencil,
                .stagesAfter = rachis::PipelineStage::depthStencil,
                .accessBefore = rachis::Access::depthStencilWrite,
                .accessAfter = rachis::Access::depthStencilRead,
                .layoutBefore = rachis::TextureLayout::depthStencilWrite,
                .layoutAfter = rachis::TextureLayout::depthStencilRead,
                .subresources = {.aspect = rachis::TextureAspect::depth},
            },
            rachis::TextureBarrier{
                .texture = targets.uploaded,
                .stagesBefore = rachis::PipelineStage::copy,
                .stagesAfter = rachis::PipelineStage::pixelShader,
                .accessBefore = rachis::Access::copyDestination,
                .accessAfter = rachis::Access::shaderResource,
                .layoutBefore = rachis::TextureLayout::copyDestination,
                .layoutAfter = rachis::TextureLayout::shaderResource,
            },
        };
        REQUIRE(device.resourceBarriers(command, {.textureBarriers = reads}).has_value());
    }

    constexpr std::uint32_t resourceBarrierBeforeAfterMismatch = 527;
    constexpr std::uint32_t shaderReadChainMipLevels = 3;

    void countBarrierMismatches(const rachis::DiagnosticMessage& message, void* userData) noexcept
    {
        printDiagnostic(message, nullptr);
        if (message.nativeId == resourceBarrierBeforeAfterMismatch)
        {
            static_cast<std::atomic<std::uint32_t>*>(userData)->fetch_add(1, std::memory_order_relaxed);
        }
    }

    struct ShaderReadStageChange final
    {
        rachis::PipelineStage enter = rachis::PipelineStage::none;
        rachis::PipelineStage leave = rachis::PipelineStage::none;
        bool split = false;
    };

    [[nodiscard]] rachis::TextureBarrier textureShaderReadEntry(rachis::TextureHandle texture,
                                                                rachis::PipelineStage stagesAfter,
                                                                const rachis::TextureSubresourceRange& subresources)
    {
        return rachis::TextureBarrier{
            .texture = texture,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = stagesAfter,
            .accessBefore = rachis::Access::copyDestination,
            .accessAfter = rachis::Access::shaderResource,
            .layoutBefore = rachis::TextureLayout::copyDestination,
            .layoutAfter = rachis::TextureLayout::shaderResource,
            .subresources = subresources,
        };
    }

    void uploadTextureBase(rachis::Device& device,
                           rachis::CommandBuffer& command,
                           rachis::TextureHandle texture,
                           rachis::BufferHandle upload)
    {
        REQUIRE(device
                    .textureBarrier(command,
                                    rachis::TextureBarrier{
                                        .texture = texture,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                        .subresources = {.firstMipLevel = 0, .mipLevelCount = 1},
                                    })
                    .has_value());
        REQUIRE(device
                    .copyBufferToTexture(command,
                                         {
                                             .buffer = upload,
                                             .bytesPerRow = textureRowPitch,
                                             .rowsPerImage = textureHeight,
                                             .texture = texture,
                                             .extent = {textureWidth, textureHeight, 1},
                                         })
                    .has_value());
    }

    void splitAfterShaderReadEntry(rachis::Device& device,
                                   rachis::Result<rachis::CommandBuffer>& command,
                                   const ShaderReadStageChange& change)
    {
        if (!change.split)
        {
            return;
        }
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
    }

    void submitAndWait(rachis::Device& device, rachis::CommandBuffer&& command)
    {
        auto submitted = device.submit(std::move(command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        device.collectGarbage();
    }

    void verifyTextureShaderReadStages(rachis::Device& device,
                                       rachis::BufferHandle upload,
                                       const ShaderReadStageChange& change)
    {
        auto texture = device.createTexture({
            .width = textureWidth,
            .height = textureHeight,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::copyDestination |
                     rachis::TextureUsage::copySource,
            .label = "Shader-read stage texture",
        });
        auto readback = device.createBuffer({
            .size = textureTransferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Shader-read stage texture readback",
        });
        REQUIRE(texture.has_value());
        REQUIRE(readback.has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        uploadTextureBase(device, *command, *texture, upload);
        REQUIRE(device.textureBarrier(*command, textureShaderReadEntry(*texture, change.enter, {})).has_value());
        splitAfterShaderReadEntry(device, command, change);
        REQUIRE(device
                    .textureBarrier(*command,
                                    toCopySource(*texture,
                                                 change.leave,
                                                 rachis::Access::shaderResource,
                                                 rachis::TextureLayout::shaderResource))
                    .has_value());
        readTexture(device, *command, *texture, *readback);
        submitAndWait(device, std::move(*command));

        verifyTexture(device, *readback);
        REQUIRE(device.destroyTexture(*texture).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
    }

    void verifyBufferShaderReadStages(rachis::Device& device,
                                      rachis::BufferHandle upload,
                                      const ShaderReadStageChange& change)
    {
        auto buffer = device.createBuffer({
            .size = bufferSize,
            .usage =
                rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination | rachis::BufferUsage::storage,
            .label = "Shader-read stage buffer",
        });
        auto readback = device.createBuffer({
            .size = bufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Shader-read stage buffer readback",
        });
        REQUIRE(buffer.has_value());
        REQUIRE(readback.has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .bufferBarrier(*command,
                                   {
                                       .buffer = *buffer,
                                       .stagesBefore = rachis::PipelineStage::none,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::none,
                                       .accessAfter = rachis::Access::copyDestination,
                                   })
                    .has_value());
        REQUIRE(
            device.copyBuffer(*command, {.source = upload, .destination = *buffer, .size = bufferSize}).has_value());
        REQUIRE(device
                    .bufferBarrier(*command,
                                   {
                                       .buffer = *buffer,
                                       .stagesBefore = rachis::PipelineStage::copy,
                                       .stagesAfter = change.enter,
                                       .accessBefore = rachis::Access::copyDestination,
                                       .accessAfter = rachis::Access::shaderResource,
                                   })
                    .has_value());
        splitAfterShaderReadEntry(device, command, change);
        const std::array leaveBarriers{
            rachis::BufferBarrier{
                .buffer = *buffer,
                .stagesBefore = change.leave,
                .stagesAfter = rachis::PipelineStage::copy,
                .accessBefore = rachis::Access::shaderResource,
                .accessAfter = rachis::Access::copySource,
            },
            rachis::BufferBarrier{
                .buffer = *readback,
                .stagesBefore = rachis::PipelineStage::none,
                .stagesAfter = rachis::PipelineStage::copy,
                .accessBefore = rachis::Access::none,
                .accessAfter = rachis::Access::copyDestination,
            },
        };
        REQUIRE(device.resourceBarriers(*command, {.bufferBarriers = leaveBarriers}).has_value());
        REQUIRE(
            device.copyBuffer(*command, {.source = *buffer, .destination = *readback, .size = bufferSize}).has_value());
        submitAndWait(device, std::move(*command));

        verifyBuffer(device, *readback, 0);
        REQUIRE(device.destroyBuffer(*buffer).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
    }

    void verifyMipmapShaderReadStages(rachis::Device& device,
                                      rachis::BufferHandle upload,
                                      const ShaderReadStageChange& change)
    {
        auto texture = device.createTexture({
            .width = textureWidth,
            .height = textureHeight,
            .mipLevels = shaderReadChainMipLevels,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage |
                     rachis::TextureUsage::copyDestination | rachis::TextureUsage::copySource,
            .label = "Shader-read stage mip chain",
        });
        auto readback = device.createBuffer({
            .size = textureTransferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Shader-read stage mip readback",
        });
        REQUIRE(texture.has_value());
        REQUIRE(readback.has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        uploadTextureBase(device, *command, *texture, upload);
        auto baseLayout = rachis::TextureLayout::copyDestination;
        if (change.enter != rachis::PipelineStage::none)
        {
            REQUIRE(device
                        .textureBarrier(
                            *command,
                            textureShaderReadEntry(*texture, change.enter, {.firstMipLevel = 0, .mipLevelCount = 1}))
                        .has_value());
            baseLayout = rachis::TextureLayout::shaderResource;
        }
        REQUIRE(device
                    .generateMipmaps(*command,
                                     {
                                         .texture = *texture,
                                         .baseLayout = baseLayout,
                                         .finalLayout = rachis::TextureLayout::shaderResource,
                                     })
                    .has_value());
        splitAfterShaderReadEntry(device, command, change);
        REQUIRE(device
                    .textureBarrier(*command,
                                    toCopySource(*texture,
                                                 change.leave,
                                                 rachis::Access::shaderResource,
                                                 rachis::TextureLayout::shaderResource))
                    .has_value());
        readTexture(device, *command, *texture, *readback);
        submitAndWait(device, std::move(*command));

        verifyTexture(device, *readback);
        REQUIRE(device.destroyTexture(*texture).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
    }

    constexpr std::uint32_t barrierLayoutMismatch = 1334;

    void countLayoutMismatches(const rachis::DiagnosticMessage& message, void* userData) noexcept
    {
        printDiagnostic(message, nullptr);
        if (message.nativeId == barrierLayoutMismatch)
        {
            static_cast<std::atomic<std::uint32_t>*>(userData)->fetch_add(1, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] constexpr bool queueAcceptsHandoffLayout(rachis::QueueType queue,
                                                           rachis::TextureLayout layout) noexcept
    {
        if (layout == rachis::TextureLayout::common || layout == rachis::TextureLayout::copySource ||
            layout == rachis::TextureLayout::copyDestination)
        {
            return true;
        }
        return queue != rachis::QueueType::copy;
    }

    [[nodiscard]] constexpr rachis::PipelineStage handoffStages(rachis::TextureLayout layout) noexcept
    {
        if (layout == rachis::TextureLayout::shaderResource || layout == rachis::TextureLayout::unorderedAccess)
        {
            return rachis::PipelineStage::computeShader;
        }
        return rachis::PipelineStage::copy;
    }

    [[nodiscard]] constexpr rachis::Access handoffAccess(rachis::TextureLayout layout) noexcept
    {
        switch (layout)
        {
        case rachis::TextureLayout::shaderResource:
            return rachis::Access::shaderResource;
        case rachis::TextureLayout::unorderedAccess:
            return rachis::Access::unorderedAccess;
        case rachis::TextureLayout::copyDestination:
            return rachis::Access::copyDestination;
        default:
            return rachis::Access::copySource;
        }
    }

    void verifyQueueLayoutHandoff(rachis::Device& device,
                                  rachis::BufferHandle upload,
                                  rachis::QueueType first,
                                  rachis::QueueType second,
                                  rachis::TextureLayout layout)
    {
        auto texture = device.createTexture({
            .width = textureWidth,
            .height = textureHeight,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage |
                     rachis::TextureUsage::copyDestination | rachis::TextureUsage::copySource,
            .label = "Queue handoff texture",
        });
        auto readback = device.createBuffer({
            .size = textureTransferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Queue handoff texture readback",
        });
        REQUIRE(texture.has_value());
        REQUIRE(readback.has_value());

        auto firstCommand = rachis::test::acquireCommandBuffer(device, first);
        REQUIRE(firstCommand.has_value());
        REQUIRE(device
                    .textureBarrier(*firstCommand,
                                    rachis::TextureBarrier{
                                        .texture = *texture,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                    })
                    .has_value());
        REQUIRE(device
                    .copyBufferToTexture(*firstCommand,
                                         {
                                             .buffer = upload,
                                             .bytesPerRow = textureRowPitch,
                                             .rowsPerImage = textureHeight,
                                             .texture = *texture,
                                             .extent = {textureWidth, textureHeight, 1},
                                         })
                    .has_value());
        if (layout != rachis::TextureLayout::copyDestination)
        {
            REQUIRE(device
                        .textureBarrier(*firstCommand,
                                        rachis::TextureBarrier{
                                            .texture = *texture,
                                            .stagesBefore = rachis::PipelineStage::copy,
                                            .stagesAfter = handoffStages(layout),
                                            .accessBefore = rachis::Access::copyDestination,
                                            .accessAfter = handoffAccess(layout),
                                            .layoutBefore = rachis::TextureLayout::copyDestination,
                                            .layoutAfter = layout,
                                        })
                        .has_value());
        }
        auto firstSubmitted = device.submit(std::move(*firstCommand));
        REQUIRE(firstSubmitted.has_value());

        auto secondCommand = rachis::test::acquireCommandBuffer(device, second);
        REQUIRE(secondCommand.has_value());
        REQUIRE(device
                    .textureBarrier(*secondCommand,
                                    toCopySource(*texture, handoffStages(layout), handoffAccess(layout), layout))
                    .has_value());
        readTexture(device, *secondCommand, *texture, *readback);
        const std::array waits{*firstSubmitted};
        auto secondSubmitted = device.submit(std::move(*secondCommand), waits);
        REQUIRE(secondSubmitted.has_value());
        REQUIRE(device.wait(*secondSubmitted).has_value());
        device.collectGarbage();

        verifyTexture(device, *readback);
        REQUIRE(device.destroyTexture(*texture).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
    }

} // namespace

TEST_CASE("resource-barrier batches validate atomically and retain every transitioned resource",
          "[size:medium][rhi][barrier][batch][warp]")
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

    auto bufferUpload = device.createBuffer({
        .size = 2 * bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Barrier batch buffer upload",
    });
    auto textureUpload = device.createBuffer({
        .size = textureTransferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Barrier batch texture upload",
    });
    const auto deviceBufferDesc = rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
    };
    auto firstBuffer = device.createBuffer(deviceBufferDesc);
    auto secondBuffer = device.createBuffer(deviceBufferDesc);
    auto barrierOnlyBuffer = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .label = "Barrier-only lifetime resource",
    });
    auto texture = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Barrier batch texture",
    });
    auto staleBuffer = device.createBuffer(deviceBufferDesc);
    REQUIRE(bufferUpload.has_value());
    REQUIRE(textureUpload.has_value());
    REQUIRE(firstBuffer.has_value());
    REQUIRE(secondBuffer.has_value());
    REQUIRE(barrierOnlyBuffer.has_value());
    REQUIRE(texture.has_value());
    REQUIRE(staleBuffer.has_value());
    initializeUploads(device, *bufferUpload, *textureUpload);
    const auto staleHandle = *staleBuffer;
    REQUIRE(device.destroyBuffer(*staleBuffer).has_value());

    const auto readbackDesc = rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    };
    auto firstReadback = device.createBuffer(readbackDesc);
    auto secondReadback = device.createBuffer(readbackDesc);
    auto textureReadback = device.createBuffer({
        .size = textureTransferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Barrier batch texture readback",
    });
    REQUIRE(firstReadback.has_value());
    REQUIRE(secondReadback.has_value());
    REQUIRE(textureReadback.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command.has_value());
    CHECK_FALSE(device.resourceBarriers(*command, {}).has_value());

    const std::array initialMemoryBarriers{
        rachis::GlobalBarrier{
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::copyDestination,
            .accessAfter = rachis::Access::copyDestination,
        },
    };
    const std::array initialBufferBarriers{
        rachis::BufferBarrier{
            .buffer = *firstBuffer,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::common,
            .accessAfter = rachis::Access::copyDestination,
        },
        rachis::BufferBarrier{
            .buffer = *secondBuffer,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::common,
            .accessAfter = rachis::Access::copyDestination,
        },
        rachis::BufferBarrier{
            .buffer = *barrierOnlyBuffer,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::common,
            .accessAfter = rachis::Access::copyDestination,
        },
    };
    const std::array initialTextureBarriers{
        rachis::TextureBarrier{
            .texture = *texture,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessAfter = rachis::Access::copyDestination,
            .layoutBefore = rachis::TextureLayout::undefined,
            .layoutAfter = rachis::TextureLayout::copyDestination,
        },
    };
    const std::array duplicateBuffers{initialBufferBarriers[0], initialBufferBarriers[0]};
    CHECK_FALSE(device.resourceBarriers(*command, {.bufferBarriers = duplicateBuffers}).has_value());
    const std::array overlappingTextures{initialTextureBarriers[0], initialTextureBarriers[0]};
    CHECK_FALSE(device.resourceBarriers(*command, {.textureBarriers = overlappingTextures}).has_value());
    auto partialBufferRange = initialBufferBarriers[0];
    partialBufferRange.size = bufferSize / 2;
    const std::array partialBufferBatch{partialBufferRange, initialBufferBarriers[1]};
    const auto partialResult = device.resourceBarriers(*command, {.bufferBarriers = partialBufferBatch});
    REQUIRE_FALSE(partialResult.has_value());
    CHECK(partialResult.error().code == rachis::ErrorCode::unsupported);
    auto mismatchedTextureAccess = initialTextureBarriers[0];
    mismatchedTextureAccess.accessAfter = rachis::Access::copySource;
    const std::array mismatchedTextureBatch{mismatchedTextureAccess};
    const auto mismatchedTextureResult = device.resourceBarriers(*command, {.textureBarriers = mismatchedTextureBatch});
    REQUIRE_FALSE(mismatchedTextureResult.has_value());
    CHECK(mismatchedTextureResult.error().code == rachis::ErrorCode::invalidArgument);
    auto unknownStage = initialBufferBarriers[0];
    unknownStage.stagesAfter = static_cast<rachis::PipelineStage>(1U << 31U);
    const std::array unknownStageBatch{unknownStage};
    CHECK_FALSE(device.resourceBarriers(*command, {.bufferBarriers = unknownStageBatch}).has_value());
    auto invalidGlobalCommon = initialMemoryBarriers[0];
    invalidGlobalCommon.stagesBefore = rachis::PipelineStage::all;
    invalidGlobalCommon.accessBefore = rachis::Access::common;
    const std::array invalidGlobalCommonBatch{invalidGlobalCommon};
    CHECK_FALSE(device.resourceBarriers(*command, {.globalBarriers = invalidGlobalCommonBatch}).has_value());
    auto unknownMemoryStage = initialMemoryBarriers[0];
    unknownMemoryStage.stagesAfter = static_cast<rachis::PipelineStage>(1U << 31U);
    const std::array unknownMemoryBatch{unknownMemoryStage};
    CHECK_FALSE(device
                    .resourceBarriers(*command,
                                      {
                                          .globalBarriers = unknownMemoryBatch,
                                          .bufferBarriers = initialBufferBarriers,
                                          .textureBarriers = initialTextureBarriers,
                                      })
                    .has_value());

    REQUIRE(device
                .resourceBarriers(*command,
                                  {
                                      .globalBarriers = initialMemoryBarriers,
                                      .bufferBarriers = initialBufferBarriers,
                                      .textureBarriers = initialTextureBarriers,
                                  })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *bufferUpload,
                                .destination = *firstBuffer,
                                .size = bufferSize,
                            })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *bufferUpload,
                                .sourceOffset = bufferSize,
                                .destination = *secondBuffer,
                                .size = bufferSize,
                            })
                .has_value());
    REQUIRE(device
                .copyBufferToTexture(*command,
                                     {
                                         .buffer = *textureUpload,
                                         .bytesPerRow = textureRowPitch,
                                         .rowsPerImage = textureHeight,
                                         .texture = *texture,
                                         .extent = {textureWidth, textureHeight, 1},
                                     })
                .has_value());

    REQUIRE(device
                .globalBarrier(*command,
                               {
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::copyDestination,
                               })
                .has_value());

    const std::array invalidLateBufferBarriers{
        rachis::BufferBarrier{
            .buffer = *firstBuffer,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::copyDestination,
            .accessAfter = rachis::Access::copySource,
        },
        rachis::BufferBarrier{
            .buffer = staleHandle,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::copyDestination,
            .accessAfter = rachis::Access::copySource,
        },
    };
    CHECK_FALSE(device.resourceBarriers(*command, {.bufferBarriers = invalidLateBufferBarriers}).has_value());

    const std::array finalBufferBarriers{invalidLateBufferBarriers[0],
                                         rachis::BufferBarrier{
                                             .buffer = *secondBuffer,
                                             .stagesBefore = rachis::PipelineStage::copy,
                                             .stagesAfter = rachis::PipelineStage::copy,
                                             .accessBefore = rachis::Access::copyDestination,
                                             .accessAfter = rachis::Access::copySource,
                                         }};
    const std::array finalTextureBarriers{
        rachis::TextureBarrier{
            .texture = *texture,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::copyDestination,
            .accessAfter = rachis::Access::copySource,
            .layoutBefore = rachis::TextureLayout::copyDestination,
            .layoutAfter = rachis::TextureLayout::copySource,
        },
    };
    const std::array finalMemoryBarriers{
        rachis::GlobalBarrier{
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::copyDestination,
            .accessAfter = rachis::Access::copySource,
        },
    };
    REQUIRE(device
                .resourceBarriers(*command,
                                  {
                                      .globalBarriers = finalMemoryBarriers,
                                      .bufferBarriers = finalBufferBarriers,
                                      .textureBarriers = finalTextureBarriers,
                                  })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *firstBuffer,
                                .destination = *firstReadback,
                                .size = bufferSize,
                            })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *secondBuffer,
                                .destination = *secondReadback,
                                .size = bufferSize,
                            })
                .has_value());
    REQUIRE(device
                .copyTextureToBuffer(*command,
                                     {
                                         .buffer = *textureReadback,
                                         .bytesPerRow = textureRowPitch,
                                         .rowsPerImage = textureHeight,
                                         .texture = *texture,
                                         .extent = {textureWidth, textureHeight, 1},
                                     })
                .has_value());

    REQUIRE(device.destroyBuffer(*bufferUpload).has_value());
    REQUIRE(device.destroyBuffer(*textureUpload).has_value());
    REQUIRE(device.destroyBuffer(*firstBuffer).has_value());
    REQUIRE(device.destroyBuffer(*secondBuffer).has_value());
    REQUIRE(device.destroyBuffer(*barrierOnlyBuffer).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    verifyBuffer(device, *firstReadback, 0);
    verifyBuffer(device, *secondReadback, 1);
    verifyTexture(device, *textureReadback);

    REQUIRE(device.destroyBuffer(*firstReadback).has_value());
    REQUIRE(device.destroyBuffer(*secondReadback).has_value());
    REQUIRE(device.destroyBuffer(*textureReadback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("compute shader-resource and host-heap barriers stay validation-clean across barrier modes",
          "[size:medium][rhi][barrier][batch][warp]")
{
    for (const bool legacyBarriers : {false, true})
    {
        CAPTURE(legacyBarriers);
        const ForceLegacyBarriers guard(legacyBarriers);
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
        if (legacyBarriers)
        {
            REQUIRE_FALSE(rachis::test::requireCapabilities(device).enhancedBarriers);
        }

        auto bufferUpload = device.createBuffer({
            .size = 2 * bufferSize,
            .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::constant,
            .memoryType = rachis::MemoryType::upload,
            .label = "Host upload barrier buffer",
        });
        auto textureUpload = device.createBuffer({
            .size = textureTransferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Host texture upload barrier buffer",
        });
        auto storageBuffer = device.createBuffer({
            .size = bufferSize,
            .usage =
                rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination | rachis::BufferUsage::storage,
            .label = "Compute SRV barrier buffer",
        });
        auto sampledTexture = device.createTexture({
            .width = textureWidth,
            .height = textureHeight,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                     rachis::TextureUsage::sampled,
            .label = "Compute SRV barrier texture",
        });
        auto bufferReadback = device.createBuffer({
            .size = bufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Host readback barrier buffer",
        });
        auto textureReadback = device.createBuffer({
            .size = textureTransferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Host texture readback barrier buffer",
        });
        REQUIRE(bufferUpload.has_value());
        REQUIRE(textureUpload.has_value());
        REQUIRE(storageBuffer.has_value());
        REQUIRE(sampledTexture.has_value());
        REQUIRE(bufferReadback.has_value());
        REQUIRE(textureReadback.has_value());
        initializeUploads(device, *bufferUpload, *textureUpload);

        auto copyCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
        REQUIRE(copyCommand.has_value());
        const std::array copyInitBuffers{
            rachis::BufferBarrier{
                .buffer = *bufferUpload,
                .stagesBefore = rachis::PipelineStage::none,
                .stagesAfter = rachis::PipelineStage::copy,
                .accessBefore = rachis::Access::none,
                .accessAfter = rachis::Access::copySource,
            },
            rachis::BufferBarrier{
                .buffer = *storageBuffer,
                .stagesBefore = rachis::PipelineStage::all,
                .stagesAfter = rachis::PipelineStage::copy,
                .accessBefore = rachis::Access::common,
                .accessAfter = rachis::Access::copyDestination,
            },
        };
        const std::array copyInitTextures{
            rachis::TextureBarrier{
                .texture = *sampledTexture,
                .stagesAfter = rachis::PipelineStage::copy,
                .accessAfter = rachis::Access::copyDestination,
                .layoutBefore = rachis::TextureLayout::undefined,
                .layoutAfter = rachis::TextureLayout::copyDestination,
            },
        };
        REQUIRE(device
                    .resourceBarriers(*copyCommand,
                                      {
                                          .bufferBarriers = copyInitBuffers,
                                          .textureBarriers = copyInitTextures,
                                      })
                    .has_value());
        REQUIRE(device
                    .copyBuffer(*copyCommand,
                                {
                                    .source = *bufferUpload,
                                    .destination = *storageBuffer,
                                    .size = bufferSize,
                                })
                    .has_value());
        REQUIRE(device
                    .copyBufferToTexture(*copyCommand,
                                         {
                                             .buffer = *textureUpload,
                                             .bytesPerRow = textureRowPitch,
                                             .rowsPerImage = textureHeight,
                                             .texture = *sampledTexture,
                                             .extent = {textureWidth, textureHeight, 1},
                                         })
                    .has_value());
        const std::array copyReleaseBuffers{
            rachis::BufferBarrier{
                .buffer = *storageBuffer,
                .stagesBefore = rachis::PipelineStage::copy,
                .stagesAfter = rachis::PipelineStage::all,
                .accessBefore = rachis::Access::copyDestination,
                .accessAfter = rachis::Access::common,
            },
        };
        const std::array copyReleaseTextures{
            rachis::TextureBarrier{
                .texture = *sampledTexture,
                .stagesBefore = rachis::PipelineStage::copy,
                .stagesAfter = rachis::PipelineStage::all,
                .accessBefore = rachis::Access::copyDestination,
                .accessAfter = rachis::Access::common,
                .layoutBefore = rachis::TextureLayout::copyDestination,
                .layoutAfter = rachis::TextureLayout::common,
            },
        };
        REQUIRE(device
                    .resourceBarriers(*copyCommand,
                                      {
                                          .bufferBarriers = copyReleaseBuffers,
                                          .textureBarriers = copyReleaseTextures,
                                      })
                    .has_value());
        auto copySubmitted = device.submit(std::move(*copyCommand));
        REQUIRE(copySubmitted.has_value());
        REQUIRE(device.wait(*copySubmitted).has_value());

        auto computeCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(computeCommand.has_value());
        REQUIRE(device
                    .bufferBarrier(*computeCommand,
                                   {
                                       .buffer = *bufferUpload,
                                       .stagesBefore = rachis::PipelineStage::all,
                                       .stagesAfter = rachis::PipelineStage::all,
                                       .accessBefore = rachis::Access::common,
                                       .accessAfter = rachis::Access::constantBuffer,
                                   })
                    .has_value());
        const std::array computeShaderReadBuffers{
            rachis::BufferBarrier{
                .buffer = *storageBuffer,
                .stagesBefore = rachis::PipelineStage::all,
                .stagesAfter = rachis::PipelineStage::all,
                .accessBefore = rachis::Access::common,
                .accessAfter = rachis::Access::shaderResource,
            },
            rachis::BufferBarrier{
                .buffer = *bufferReadback,
                .stagesBefore = rachis::PipelineStage::none,
                .stagesAfter = rachis::PipelineStage::copy,
                .accessBefore = rachis::Access::none,
                .accessAfter = rachis::Access::copyDestination,
            },
        };
        const std::array computeShaderReadTextures{
            rachis::TextureBarrier{
                .texture = *sampledTexture,
                .stagesBefore = rachis::PipelineStage::all,
                .stagesAfter = rachis::PipelineStage::all,
                .accessBefore = rachis::Access::common,
                .accessAfter = rachis::Access::shaderResource,
                .layoutBefore = rachis::TextureLayout::common,
                .layoutAfter = rachis::TextureLayout::shaderResource,
            },
        };
        REQUIRE(device
                    .resourceBarriers(*computeCommand,
                                      {
                                          .bufferBarriers = computeShaderReadBuffers,
                                          .textureBarriers = computeShaderReadTextures,
                                      })
                    .has_value());

        const std::array computeToCopyBuffers{
            rachis::BufferBarrier{
                .buffer = *storageBuffer,
                .stagesBefore = rachis::PipelineStage::all,
                .stagesAfter = rachis::PipelineStage::copy,
                .accessBefore = rachis::Access::shaderResource,
                .accessAfter = rachis::Access::copySource,
            },
            rachis::BufferBarrier{
                .buffer = *textureReadback,
                .stagesBefore = rachis::PipelineStage::none,
                .stagesAfter = rachis::PipelineStage::copy,
                .accessBefore = rachis::Access::none,
                .accessAfter = rachis::Access::copyDestination,
            },
        };
        const std::array computeToCopyTextures{
            rachis::TextureBarrier{
                .texture = *sampledTexture,
                .stagesBefore = rachis::PipelineStage::all,
                .stagesAfter = rachis::PipelineStage::copy,
                .accessBefore = rachis::Access::shaderResource,
                .accessAfter = rachis::Access::copySource,
                .layoutBefore = rachis::TextureLayout::shaderResource,
                .layoutAfter = rachis::TextureLayout::copySource,
            },
        };
        REQUIRE(device
                    .resourceBarriers(*computeCommand,
                                      {
                                          .bufferBarriers = computeToCopyBuffers,
                                          .textureBarriers = computeToCopyTextures,
                                      })
                    .has_value());
        REQUIRE(device
                    .copyBuffer(*computeCommand,
                                {
                                    .source = *storageBuffer,
                                    .destination = *bufferReadback,
                                    .size = bufferSize,
                                })
                    .has_value());
        REQUIRE(device
                    .copyTextureToBuffer(*computeCommand,
                                         {
                                             .buffer = *textureReadback,
                                             .bytesPerRow = textureRowPitch,
                                             .rowsPerImage = textureHeight,
                                             .texture = *sampledTexture,
                                             .extent = {textureWidth, textureHeight, 1},
                                         })
                    .has_value());
        auto computeSubmitted = device.submit(std::move(*computeCommand));
        REQUIRE(computeSubmitted.has_value());
        REQUIRE(device.wait(*computeSubmitted).has_value());
        device.collectGarbage();

        verifyBuffer(device, *bufferReadback, 0);
        verifyTexture(device, *textureReadback);

        REQUIRE(device.destroyBuffer(*bufferUpload).has_value());
        REQUIRE(device.destroyBuffer(*textureUpload).has_value());
        REQUIRE(device.destroyBuffer(*storageBuffer).has_value());
        REQUIRE(device.destroyTexture(*sampledTexture).has_value());
        REQUIRE(device.destroyBuffer(*bufferReadback).has_value());
        REQUIRE(device.destroyBuffer(*textureReadback).has_value());
        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.warnings == 0);
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
    }
}

TEST_CASE("legacy texture barriers from the undefined layout start from each texture's current state",
          "[size:medium][rhi][barrier][batch][warp]")
{
    const ForceLegacyBarriers guard;
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
    REQUIRE_FALSE(rachis::test::requireCapabilities(device).enhancedBarriers);

    auto color = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::sampled | rachis::TextureUsage::copySource,
        .label = "Legacy discard color target",
    });
    auto depth = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::depthStencil,
        .label = "Legacy discard depth target",
    });
    auto uploaded = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage =
            rachis::TextureUsage::sampled | rachis::TextureUsage::copyDestination | rachis::TextureUsage::copySource,
        .label = "Legacy discard uploaded texture",
    });
    auto upload = device.createBuffer({
        .size = textureTransferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Legacy discard texture upload",
    });
    auto colorReadback = device.createBuffer({
        .size = textureTransferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Legacy discard color readback",
    });
    auto uploadedReadback = device.createBuffer({
        .size = textureTransferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Legacy discard uploaded readback",
    });
    REQUIRE(color.has_value());
    REQUIRE(depth.has_value());
    REQUIRE(uploaded.has_value());
    REQUIRE(upload.has_value());
    REQUIRE(colorReadback.has_value());
    REQUIRE(uploadedReadback.has_value());
    auto colorView = device.createTextureView({
        .texture = *color,
        .usage = rachis::TextureViewUsage::renderTarget,
        .label = "Legacy discard color view",
    });
    auto depthView = device.createTextureView({
        .texture = *depth,
        .usage = rachis::TextureViewUsage::depthStencil,
        .aspect = rachis::TextureAspect::depth,
        .label = "Legacy discard depth view",
    });
    REQUIRE(colorView.has_value());
    REQUIRE(depthView.has_value());

    auto mapped = device.mapBuffer(*upload);
    REQUIRE(mapped.has_value());
    std::memset(mapped->data, 0, static_cast<std::size_t>(textureTransferSize));
    for (std::uint32_t y = 0; y < textureHeight; ++y)
    {
        for (std::uint32_t x = 0; x < textureWidth; ++x)
        {
            for (std::uint32_t channel = 0; channel < textureBytesPerPixel; ++channel)
            {
                mapped->data[y * textureRowPitch + x * textureBytesPerPixel + channel] = textureValue(x, y, channel);
            }
        }
    }
    REQUIRE(device.unmapBuffer(*upload).has_value());

    const LegacyDiscardTargets targets{
        .color = *color,
        .depth = *depth,
        .uploaded = *uploaded,
        .colorView = *colorView,
        .depthView = *depthView,
        .upload = *upload,
    };
    const std::array frameColors{
        rachis::ClearColor{1.0F, 0.0F, 0.0F, 1.0F},
        rachis::ClearColor{0.0F, 1.0F, 0.0F, 1.0F},
        rachis::ClearColor{0.0F, 0.0F, 1.0F, 1.0F},
    };
    for (std::size_t frame = 0; frame < frameColors.size(); ++frame)
    {
        CAPTURE(frame);
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        recordLegacyDiscardFrame(device, *command, targets, frameColors[frame]);
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        CHECK(device.diagnosticCounts().errors == 0);
    }

    auto firstColorPass = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    auto secondColorPass = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(firstColorPass.has_value());
    REQUIRE(secondColorPass.has_value());
    const std::array colorDiscard{discardColorTarget(targets)};
    const std::array colorToCopy{toCopySource(*color,
                                              rachis::PipelineStage::renderTarget,
                                              rachis::Access::renderTarget,
                                              rachis::TextureLayout::renderTarget)};
    REQUIRE(device.resourceBarriers(*firstColorPass, {.textureBarriers = colorDiscard}).has_value());
    clearColorTarget(device, *firstColorPass, targets, {1.0F, 1.0F, 0.0F, 1.0F});
    REQUIRE(device.resourceBarriers(*firstColorPass, {.textureBarriers = colorToCopy}).has_value());
    REQUIRE(device.resourceBarriers(*secondColorPass, {.textureBarriers = colorDiscard}).has_value());
    clearColorTarget(device, *secondColorPass, targets, {0.0F, 1.0F, 1.0F, 1.0F});
    REQUIRE(device.resourceBarriers(*secondColorPass, {.textureBarriers = colorToCopy}).has_value());
    readTexture(device, *secondColorPass, *color, *colorReadback);
    auto firstColorSubmitted = device.submit(std::move(*firstColorPass));
    REQUIRE(firstColorSubmitted.has_value());
    auto secondColorSubmitted = device.submit(std::move(*secondColorPass));
    REQUIRE(secondColorSubmitted.has_value());

    auto firstUploadPass = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    auto secondUploadPass = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(firstUploadPass.has_value());
    REQUIRE(secondUploadPass.has_value());
    const std::array uploadDiscard{discardUploadedTexture(targets)};
    const std::array uploadToCopy{toCopySource(*uploaded,
                                               rachis::PipelineStage::copy,
                                               rachis::Access::copyDestination,
                                               rachis::TextureLayout::copyDestination)};
    for (auto* pass : {&*firstUploadPass, &*secondUploadPass})
    {
        REQUIRE(device.resourceBarriers(*pass, {.textureBarriers = uploadDiscard}).has_value());
        uploadTexture(device, *pass, targets);
        REQUIRE(device.resourceBarriers(*pass, {.textureBarriers = uploadToCopy}).has_value());
    }
    readTexture(device, *secondUploadPass, *uploaded, *uploadedReadback);
    std::array uploadPasses{std::move(*firstUploadPass), std::move(*secondUploadPass)};
    auto uploadSubmitted = device.submit(uploadPasses);
    REQUIRE(uploadSubmitted.has_value());
    REQUIRE(device.wait(*firstColorSubmitted).has_value());
    REQUIRE(device.wait(*secondColorSubmitted).has_value());
    REQUIRE(device.wait(*uploadSubmitted).has_value());

    verifySolidColor(device, *colorReadback, {0, 255, 255, 255});
    verifyTexture(device, *uploadedReadback);

    REQUIRE(device.destroyTextureView(*colorView).has_value());
    REQUIRE(device.destroyTextureView(*depthView).has_value());
    REQUIRE(device.destroyTexture(*color).has_value());
    REQUIRE(device.destroyTexture(*depth).has_value());
    REQUIRE(device.destroyTexture(*uploaded).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*colorReadback).has_value());
    REQUIRE(device.destroyBuffer(*uploadedReadback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("legacy barriers reject accesses that combine a write with another access",
          "[size:medium][rhi][barrier][batch][warp]")
{
    const ForceLegacyBarriers guard;
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
    REQUIRE_FALSE(rachis::test::requireCapabilities(device).enhancedBarriers);

    auto bufferUpload = device.createBuffer({
        .size = 2 * bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Mixed-access buffer upload",
    });
    auto textureUpload = device.createBuffer({
        .size = textureTransferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Mixed-access texture upload",
    });
    auto buffer = device.createBuffer({
        .size = 2 * bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination | rachis::BufferUsage::storage,
        .label = "Mixed-access buffer",
    });
    auto texture = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage =
            rachis::TextureUsage::sampled | rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Mixed-access texture",
    });
    auto depth = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::depthStencil,
        .label = "Mixed-access depth target",
    });
    auto bufferReadback = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Mixed-access buffer readback",
    });
    auto textureReadback = device.createBuffer({
        .size = textureTransferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Mixed-access texture readback",
    });
    REQUIRE(bufferUpload.has_value());
    REQUIRE(textureUpload.has_value());
    REQUIRE(buffer.has_value());
    REQUIRE(texture.has_value());
    REQUIRE(depth.has_value());
    REQUIRE(bufferReadback.has_value());
    REQUIRE(textureReadback.has_value());
    initializeUploads(device, *bufferUpload, *textureUpload);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    const std::array uploadBuffers{
        rachis::BufferBarrier{
            .buffer = *buffer,
            .stagesBefore = rachis::PipelineStage::none,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::none,
            .accessAfter = rachis::Access::copyDestination,
        },
    };
    const std::array uploadTextures{
        rachis::TextureBarrier{
            .texture = *texture,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessAfter = rachis::Access::copyDestination,
            .layoutBefore = rachis::TextureLayout::undefined,
            .layoutAfter = rachis::TextureLayout::copyDestination,
        },
    };
    REQUIRE(device.resourceBarriers(*command, {.bufferBarriers = uploadBuffers, .textureBarriers = uploadTextures})
                .has_value());
    REQUIRE(
        device.copyBuffer(*command, {.source = *bufferUpload, .destination = *buffer, .size = bufferSize}).has_value());
    REQUIRE(device
                .copyBufferToTexture(*command,
                                     {
                                         .buffer = *textureUpload,
                                         .bytesPerRow = textureRowPitch,
                                         .rowsPerImage = textureHeight,
                                         .texture = *texture,
                                         .extent = {textureWidth, textureHeight, 1},
                                     })
                .has_value());

    constexpr auto copyReadWrite = rachis::Access::copySource | rachis::Access::copyDestination;
    const std::array rejectedBuffers{
        rachis::BufferBarrier{
            .buffer = *buffer,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::copyDestination,
            .accessAfter = copyReadWrite,
        },
        rachis::BufferBarrier{
            .buffer = *buffer,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::computeShader,
            .accessBefore = rachis::Access::copyDestination,
            .accessAfter = rachis::Access::shaderResource | rachis::Access::unorderedAccess,
        },
        rachis::BufferBarrier{
            .buffer = *buffer,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = copyReadWrite,
            .accessAfter = rachis::Access::copySource,
        },
    };
    for (const auto& barrier : rejectedBuffers)
    {
        const auto rejected = device.bufferBarrier(*command, barrier);
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::unsupported);
    }
    const std::array rejectedTextures{
        rachis::TextureBarrier{
            .texture = *texture,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::copyDestination,
            .accessAfter = copyReadWrite,
            .layoutBefore = rachis::TextureLayout::copyDestination,
            .layoutAfter = rachis::TextureLayout::common,
        },
        rachis::TextureBarrier{
            .texture = *texture,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = copyReadWrite,
            .accessAfter = rachis::Access::copySource,
            .layoutBefore = rachis::TextureLayout::common,
            .layoutAfter = rachis::TextureLayout::copySource,
        },
    };
    for (const auto& barrier : rejectedTextures)
    {
        const auto rejected = device.textureBarrier(*command, barrier);
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::unsupported);
    }
    const std::array batchedBuffers{
        rachis::BufferBarrier{
            .buffer = *buffer,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::copyDestination,
            .accessAfter = rachis::Access::copySource,
        },
    };
    const std::array batchedTextures{rejectedTextures[0]};
    const auto rejectedBatch =
        device.resourceBarriers(*command, {.bufferBarriers = batchedBuffers, .textureBarriers = batchedTextures});
    REQUIRE_FALSE(rejectedBatch.has_value());
    CHECK(rejectedBatch.error().code == rachis::ErrorCode::unsupported);
    const std::array sameBufferRegions{
        rachis::BufferCopyRegion{.sourceOffset = 0, .destinationOffset = bufferSize, .size = bufferSize},
    };
    const auto sameBufferCopy = device.copyBuffer(*command, *buffer, *buffer, sameBufferRegions);
    REQUIRE_FALSE(sameBufferCopy.has_value());
    CHECK(sameBufferCopy.error().code == rachis::ErrorCode::unsupported);

    constexpr auto copyAndShaderRead = rachis::Access::copySource | rachis::Access::shaderResource;
    constexpr auto depthReadWrite = rachis::Access::depthStencilRead | rachis::Access::depthStencilWrite;
    const std::array readBuffers{
        rachis::BufferBarrier{
            .buffer = *buffer,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::copy | rachis::PipelineStage::computeShader,
            .accessBefore = rachis::Access::copyDestination,
            .accessAfter = copyAndShaderRead,
        },
        rachis::BufferBarrier{
            .buffer = *bufferReadback,
            .stagesBefore = rachis::PipelineStage::none,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::none,
            .accessAfter = rachis::Access::copyDestination,
        },
    };
    const std::array readTextures{
        rachis::TextureBarrier{
            .texture = *texture,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::copy | rachis::PipelineStage::pixelShader,
            .accessBefore = rachis::Access::copyDestination,
            .accessAfter = copyAndShaderRead,
            .layoutBefore = rachis::TextureLayout::copyDestination,
            .layoutAfter = rachis::TextureLayout::common,
        },
        rachis::TextureBarrier{
            .texture = *depth,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::depthStencil,
            .accessAfter = depthReadWrite,
            .layoutBefore = rachis::TextureLayout::undefined,
            .layoutAfter = rachis::TextureLayout::depthStencilWrite,
            .subresources = {.aspect = rachis::TextureAspect::depth},
        },
    };
    REQUIRE(device.resourceBarriers(*command, {.bufferBarriers = readBuffers, .textureBarriers = readTextures})
                .has_value());
    REQUIRE(device.copyBuffer(*command, {.source = *buffer, .destination = *bufferReadback, .size = bufferSize})
                .has_value());
    readTexture(device, *command, *texture, *textureReadback);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *depth,
                                    .stagesBefore = rachis::PipelineStage::depthStencil,
                                    .stagesAfter = rachis::PipelineStage::depthStencil,
                                    .accessBefore = depthReadWrite,
                                    .accessAfter = rachis::Access::depthStencilRead,
                                    .layoutBefore = rachis::TextureLayout::depthStencilWrite,
                                    .layoutAfter = rachis::TextureLayout::depthStencilRead,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());
    submitAndWait(device, std::move(*command));

    verifyBuffer(device, *bufferReadback, 0);
    verifyTexture(device, *textureReadback);

    REQUIRE(device.destroyBuffer(*bufferUpload).has_value());
    REQUIRE(device.destroyBuffer(*textureUpload).has_value());
    REQUIRE(device.destroyBuffer(*buffer).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyTexture(*depth).has_value());
    REQUIRE(device.destroyBuffer(*bufferReadback).has_value());
    REQUIRE(device.destroyBuffer(*textureReadback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("enhanced barriers accept accesses that combine a write with another access",
          "[size:medium][rhi][barrier][batch][warp]")
{
    const ForceLegacyBarriers guard(false);
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
    REQUIRE(rachis::test::requireCapabilities(device).enhancedBarriers);

    auto bufferUpload = device.createBuffer({
        .size = 2 * bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Enhanced mixed-access buffer upload",
    });
    auto textureUpload = device.createBuffer({
        .size = textureTransferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Enhanced mixed-access texture upload",
    });
    auto buffer = device.createBuffer({
        .size = 2 * bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .label = "Enhanced mixed-access buffer",
    });
    auto texture = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Enhanced mixed-access texture",
    });
    auto bufferReadback = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Enhanced mixed-access buffer readback",
    });
    auto textureReadback = device.createBuffer({
        .size = textureTransferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Enhanced mixed-access texture readback",
    });
    REQUIRE(bufferUpload.has_value());
    REQUIRE(textureUpload.has_value());
    REQUIRE(buffer.has_value());
    REQUIRE(texture.has_value());
    REQUIRE(bufferReadback.has_value());
    REQUIRE(textureReadback.has_value());
    initializeUploads(device, *bufferUpload, *textureUpload);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    const std::array uploadBuffers{
        rachis::BufferBarrier{
            .buffer = *buffer,
            .stagesBefore = rachis::PipelineStage::none,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::none,
            .accessAfter = rachis::Access::copyDestination,
        },
    };
    const std::array uploadTextures{
        rachis::TextureBarrier{
            .texture = *texture,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessAfter = rachis::Access::copyDestination,
            .layoutBefore = rachis::TextureLayout::undefined,
            .layoutAfter = rachis::TextureLayout::copyDestination,
        },
    };
    REQUIRE(device.resourceBarriers(*command, {.bufferBarriers = uploadBuffers, .textureBarriers = uploadTextures})
                .has_value());
    REQUIRE(
        device.copyBuffer(*command, {.source = *bufferUpload, .destination = *buffer, .size = bufferSize}).has_value());
    REQUIRE(device
                .copyBufferToTexture(*command,
                                     {
                                         .buffer = *textureUpload,
                                         .bytesPerRow = textureRowPitch,
                                         .rowsPerImage = textureHeight,
                                         .texture = *texture,
                                         .extent = {textureWidth, textureHeight, 1},
                                     })
                .has_value());

    constexpr auto copyReadWrite = rachis::Access::copySource | rachis::Access::copyDestination;
    const std::array mixedBuffers{
        rachis::BufferBarrier{
            .buffer = *buffer,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::copyDestination,
            .accessAfter = copyReadWrite,
        },
    };
    const std::array mixedTextures{
        rachis::TextureBarrier{
            .texture = *texture,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::copyDestination,
            .accessAfter = copyReadWrite,
            .layoutBefore = rachis::TextureLayout::copyDestination,
            .layoutAfter = rachis::TextureLayout::common,
        },
    };
    REQUIRE(device.resourceBarriers(*command, {.bufferBarriers = mixedBuffers, .textureBarriers = mixedTextures})
                .has_value());
    const std::array sameBufferRegions{
        rachis::BufferCopyRegion{.sourceOffset = 0, .destinationOffset = bufferSize, .size = bufferSize},
    };
    REQUIRE(device.copyBuffer(*command, *buffer, *buffer, sameBufferRegions).has_value());

    const std::array readBuffers{
        rachis::BufferBarrier{
            .buffer = *buffer,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = copyReadWrite,
            .accessAfter = rachis::Access::copySource,
        },
        rachis::BufferBarrier{
            .buffer = *bufferReadback,
            .stagesBefore = rachis::PipelineStage::none,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::none,
            .accessAfter = rachis::Access::copyDestination,
        },
    };
    const std::array readTextures{
        rachis::TextureBarrier{
            .texture = *texture,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = copyReadWrite,
            .accessAfter = rachis::Access::copySource,
            .layoutBefore = rachis::TextureLayout::common,
            .layoutAfter = rachis::TextureLayout::copySource,
        },
    };
    REQUIRE(device.resourceBarriers(*command, {.bufferBarriers = readBuffers, .textureBarriers = readTextures})
                .has_value());
    const std::array upperHalf{
        rachis::BufferCopyRegion{.sourceOffset = bufferSize, .destinationOffset = 0, .size = bufferSize},
    };
    REQUIRE(device.copyBuffer(*command, *buffer, *bufferReadback, upperHalf).has_value());
    readTexture(device, *command, *texture, *textureReadback);
    submitAndWait(device, std::move(*command));

    verifyBuffer(device, *bufferReadback, 0);
    verifyTexture(device, *textureReadback);

    REQUIRE(device.destroyBuffer(*bufferUpload).has_value());
    REQUIRE(device.destroyBuffer(*textureUpload).has_value());
    REQUIRE(device.destroyBuffer(*buffer).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyBuffer(*bufferReadback).has_value());
    REQUIRE(device.destroyBuffer(*textureReadback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("shader-resource barriers stay validation-clean when the entering and leaving stages differ",
          "[size:medium][rhi][barrier][batch][warp]")
{
    constexpr std::array stageChanges{
        ShaderReadStageChange{.enter = rachis::PipelineStage::pixelShader, .leave = rachis::PipelineStage::all},
        ShaderReadStageChange{.enter = rachis::PipelineStage::all, .leave = rachis::PipelineStage::pixelShader},
        ShaderReadStageChange{
            .enter = rachis::PipelineStage::pixelShader,
            .leave = rachis::PipelineStage::pixelShader | rachis::PipelineStage::computeShader,
        },
        ShaderReadStageChange{.enter = rachis::PipelineStage::computeShader,
                              .leave = rachis::PipelineStage::pixelShader},
        ShaderReadStageChange{
            .enter = rachis::PipelineStage::pixelShader,
            .leave = rachis::PipelineStage::all,
            .split = true,
        },
    };
    constexpr std::array mipmapChanges{
        ShaderReadStageChange{.leave = rachis::PipelineStage::all},
        ShaderReadStageChange{.leave = rachis::PipelineStage::pixelShader},
        ShaderReadStageChange{.leave = rachis::PipelineStage::pixelShader, .split = true},
        ShaderReadStageChange{.enter = rachis::PipelineStage::pixelShader, .leave = rachis::PipelineStage::all},
    };

    for (const bool legacyBarriers : {true, false})
    {
        CAPTURE(legacyBarriers);
        const ForceLegacyBarriers guard(legacyBarriers);
        std::atomic<std::uint32_t> mismatches = 0;
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .adapterPreference = rachis::AdapterPreference::warp,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = true,
            .diagnosticCallback = &countBarrierMismatches,
            .diagnosticUserData = &mismatches,
        });
        if (!created)
        {
            INFO(created.error().message);
        }
        REQUIRE(created.has_value());
        auto device = std::move(*created);
        if (legacyBarriers)
        {
            REQUIRE_FALSE(rachis::test::requireCapabilities(device).enhancedBarriers);
        }

        auto bufferUpload = device.createBuffer({
            .size = 2 * bufferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Shader-read stage buffer upload",
        });
        auto textureUpload = device.createBuffer({
            .size = textureTransferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Shader-read stage texture upload",
        });
        REQUIRE(bufferUpload.has_value());
        REQUIRE(textureUpload.has_value());
        initializeUploads(device, *bufferUpload, *textureUpload);

        for (std::size_t index = 0; index < stageChanges.size(); ++index)
        {
            CAPTURE(index);
            verifyTextureShaderReadStages(device, *textureUpload, stageChanges[index]);
            verifyBufferShaderReadStages(device, *bufferUpload, stageChanges[index]);
            CHECK(mismatches.exchange(0, std::memory_order_relaxed) == 0);
        }
        for (std::size_t index = 0; index < mipmapChanges.size(); ++index)
        {
            CAPTURE(index);
            verifyMipmapShaderReadStages(device, *textureUpload, mipmapChanges[index]);
            CHECK(mismatches.exchange(0, std::memory_order_relaxed) == 0);
        }

        REQUIRE(device.destroyBuffer(*bufferUpload).has_value());
        REQUIRE(device.destroyBuffer(*textureUpload).has_value());
        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.warnings == 0);
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
    }
}

TEST_CASE("enhanced texture layouts hand off between queues without layout mismatches",
          "[size:medium][rhi][barrier][batch][warp]")
{
    const ForceLegacyBarriers guard(false);
    std::atomic<std::uint32_t> mismatches = 0;
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &countLayoutMismatches,
        .diagnosticUserData = &mismatches,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).enhancedBarriers);

    auto bufferUpload = device.createBuffer({
        .size = 2 * bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Queue handoff buffer upload",
    });
    auto textureUpload = device.createBuffer({
        .size = textureTransferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Queue handoff texture upload",
    });
    REQUIRE(bufferUpload.has_value());
    REQUIRE(textureUpload.has_value());
    initializeUploads(device, *bufferUpload, *textureUpload);

    constexpr std::array queues{rachis::QueueType::graphics, rachis::QueueType::compute, rachis::QueueType::copy};
    constexpr std::array layouts{
        rachis::TextureLayout::common,
        rachis::TextureLayout::shaderResource,
        rachis::TextureLayout::unorderedAccess,
        rachis::TextureLayout::copySource,
        rachis::TextureLayout::copyDestination,
    };
    for (const auto first : queues)
    {
        for (const auto second : queues)
        {
            for (const auto layout : layouts)
            {
                if (!queueAcceptsHandoffLayout(first, layout) || !queueAcceptsHandoffLayout(second, layout))
                {
                    continue;
                }
                CAPTURE(first, second, layout);
                verifyQueueLayoutHandoff(device, *textureUpload, first, second, layout);
                CHECK(mismatches.exchange(0, std::memory_order_relaxed) == 0);
            }
        }
    }

    REQUIRE(device.destroyBuffer(*bufferUpload).has_value());
    REQUIRE(device.destroyBuffer(*textureUpload).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("attachment texture barriers reject the undefined layout after in both barrier modes",
          "[size:medium][rhi][barrier][batch][warp]")
{
    for (const bool legacy : {false, true})
    {
        CAPTURE(legacy);
        const ForceLegacyBarriers guard(legacy);
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
        REQUIRE(rachis::test::requireCapabilities(device).enhancedBarriers == !legacy);

        auto color = device.createTexture({
            .width = textureWidth,
            .height = textureHeight,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
            .label = "Undefined-after color target",
        });
        auto depth = device.createTexture({
            .width = textureWidth,
            .height = textureHeight,
            .format = rachis::Format::d32Float,
            .usage = rachis::TextureUsage::depthStencil,
            .label = "Undefined-after depth target",
        });
        auto readback = device.createBuffer({
            .size = textureTransferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Undefined-after color readback",
        });
        REQUIRE(color.has_value());
        REQUIRE(depth.has_value());
        REQUIRE(readback.has_value());
        auto colorView = device.createTextureView({
            .texture = *color,
            .usage = rachis::TextureViewUsage::renderTarget,
        });
        REQUIRE(colorView.has_value());
        const LegacyDiscardTargets targets{.color = *color, .colorView = *colorView};

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        const std::array rejectedBarriers{
            rachis::TextureBarrier{
                .texture = *color,
                .stagesBefore = rachis::PipelineStage::all,
                .stagesAfter = rachis::PipelineStage::renderTarget,
                .accessAfter = rachis::Access::renderTarget,
                .layoutBefore = rachis::TextureLayout::undefined,
                .layoutAfter = rachis::TextureLayout::undefined,
            },
            rachis::TextureBarrier{
                .texture = *color,
                .stagesBefore = rachis::PipelineStage::all,
                .layoutBefore = rachis::TextureLayout::undefined,
                .layoutAfter = rachis::TextureLayout::undefined,
            },
            rachis::TextureBarrier{
                .texture = *depth,
                .stagesBefore = rachis::PipelineStage::all,
                .stagesAfter = rachis::PipelineStage::depthStencil,
                .accessAfter = rachis::Access::depthStencilWrite,
                .layoutBefore = rachis::TextureLayout::undefined,
                .layoutAfter = rachis::TextureLayout::undefined,
                .subresources = {.aspect = rachis::TextureAspect::depth},
            },
        };
        for (const auto& barrier : rejectedBarriers)
        {
            const auto rejected = device.textureBarrier(*command, barrier);
            REQUIRE_FALSE(rejected.has_value());
            CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
        }
        const std::array batchedBarriers{rejectedBarriers[0], rejectedBarriers[2]};
        const auto rejectedBatch = device.resourceBarriers(*command, {.textureBarriers = batchedBarriers});
        REQUIRE_FALSE(rejectedBatch.has_value());
        CHECK(rejectedBatch.error().code == rachis::ErrorCode::invalidArgument);

        REQUIRE(device.textureBarrier(*command, discardColorTarget(targets)).has_value());
        clearColorTarget(device, *command, targets, {1.0F, 0.0F, 1.0F, 1.0F});
        const auto toUndefined = device.textureBarrier(*command,
                                                       {
                                                           .texture = *color,
                                                           .stagesBefore = rachis::PipelineStage::renderTarget,
                                                           .accessBefore = rachis::Access::renderTarget,
                                                           .layoutBefore = rachis::TextureLayout::renderTarget,
                                                           .layoutAfter = rachis::TextureLayout::undefined,
                                                       });
        REQUIRE_FALSE(toUndefined.has_value());
        CHECK(toUndefined.error().code == rachis::ErrorCode::invalidArgument);
        REQUIRE(device
                    .textureBarrier(*command,
                                    toCopySource(*color,
                                                 rachis::PipelineStage::renderTarget,
                                                 rachis::Access::renderTarget,
                                                 rachis::TextureLayout::renderTarget))
                    .has_value());
        readTexture(device, *command, *color, *readback);
        submitAndWait(device, std::move(*command));
        verifySolidColor(device, *readback, {255, 0, 255, 255});

        REQUIRE(device.destroyTextureView(*colorView).has_value());
        REQUIRE(device.destroyTexture(*color).has_value());
        REQUIRE(device.destroyTexture(*depth).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.warnings == 0);
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
    }
}

TEST_CASE("single buffer and texture barriers record without per-barrier scratch allocations",
          "[size:medium][rhi][barrier][batch][warp]")
{
    if (!rachis::test::ThreadHeapAllocationCounter::available())
    {
        SKIP("Heap allocations are counted through the MSVC debug heap.");
    }
    for (const bool legacy : {false, true})
    {
        CAPTURE(legacy);
        const ForceLegacyBarriers guard(legacy);
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
        REQUIRE(rachis::test::requireCapabilities(device).enhancedBarriers == !legacy);

        auto upload = device.createBuffer({
            .size = bufferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Allocation-counted barrier upload",
        });
        auto buffer = device.createBuffer({
            .size = bufferSize,
            .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
            .label = "Allocation-counted barrier buffer",
        });
        auto texture = device.createTexture({
            .width = textureWidth,
            .height = textureHeight,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
            .label = "Allocation-counted barrier texture",
        });
        REQUIRE(upload.has_value());
        REQUIRE(buffer.has_value());
        REQUIRE(texture.has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .bufferBarrier(*command,
                                   {
                                       .buffer = *buffer,
                                       .stagesBefore = rachis::PipelineStage::all,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::common,
                                       .accessAfter = rachis::Access::copyDestination,
                                   })
                    .has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *texture,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                    })
                    .has_value());

        const rachis::BufferBarrier bufferToSource{
            .buffer = *buffer,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::copyDestination,
            .accessAfter = rachis::Access::copySource,
        };
        const rachis::BufferBarrier bufferToDestination{
            .buffer = *buffer,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::copySource,
            .accessAfter = rachis::Access::copyDestination,
        };
        const auto textureToSource = toCopySource(*texture,
                                                  rachis::PipelineStage::copy,
                                                  rachis::Access::copyDestination,
                                                  rachis::TextureLayout::copyDestination);
        const rachis::TextureBarrier textureToDestination{
            .texture = *texture,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::copySource,
            .accessAfter = rachis::Access::copyDestination,
            .layoutBefore = rachis::TextureLayout::copySource,
            .layoutAfter = rachis::TextureLayout::copyDestination,
        };

        constexpr std::size_t iterations = 64;
        std::size_t recorded = 0;
        const auto recordBufferBarriers = [&] {
            for (std::size_t iteration = 0; iteration < iterations; ++iteration)
            {
                recorded += device.bufferBarrier(*command, bufferToSource).has_value() ? 1U : 0U;
                recorded += device.bufferBarrier(*command, bufferToDestination).has_value() ? 1U : 0U;
            }
        };
        const auto recordTextureBarriers = [&] {
            for (std::size_t iteration = 0; iteration < iterations; ++iteration)
            {
                recorded += device.textureBarrier(*command, textureToSource).has_value() ? 1U : 0U;
                recorded += device.textureBarrier(*command, textureToDestination).has_value() ? 1U : 0U;
            }
        };
        recordBufferBarriers();
        recordTextureBarriers();

        rachis::test::ThreadHeapAllocationCounter counter;
        const auto bufferAllocations = counter.count(recordBufferBarriers);
        const auto textureAllocations = counter.count(recordTextureBarriers);
        REQUIRE(recorded == 8 * iterations);

        constexpr std::size_t barriersPerMeasurement = 2 * iterations;
        constexpr std::size_t retentionGrowthAllowance = 8;
        CAPTURE(bufferAllocations, textureAllocations);
        CHECK(bufferAllocations <= barriersPerMeasurement + retentionGrowthAllowance);
        CHECK(textureAllocations <= barriersPerMeasurement + retentionGrowthAllowance);

        REQUIRE(device
                    .copyBuffer(*command,
                                {
                                    .source = *upload,
                                    .destination = *buffer,
                                    .size = bufferSize,
                                })
                    .has_value());
        submitAndWait(device, std::move(*command));
        REQUIRE(device.destroyTexture(*texture).has_value());
        REQUIRE(device.destroyBuffer(*buffer).has_value());
        REQUIRE(device.destroyBuffer(*upload).has_value());
        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.warnings == 0);
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
    }
}

TEST_CASE("large resource-barrier batches report the first repeated buffer or overlapping texture range",
          "[size:medium][rhi][barrier][batch][warp]")
{
    for (const bool legacy : {false, true})
    {
        CAPTURE(legacy);
        const ForceLegacyBarriers guard(legacy);
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
        REQUIRE(rachis::test::requireCapabilities(device).enhancedBarriers == !legacy);

        constexpr std::uint32_t batchSize = 12;
        auto upload = device.createBuffer({
            .size = bufferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Large barrier batch upload",
        });
        REQUIRE(upload.has_value());
        std::array<rachis::BufferHandle, batchSize> buffers{};
        for (auto& buffer : buffers)
        {
            auto createdBuffer = device.createBuffer({
                .size = bufferSize,
                .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
                .label = "Large barrier batch buffer",
            });
            REQUIRE(createdBuffer.has_value());
            buffer = *createdBuffer;
        }
        auto layered = device.createTexture({
            .width = textureWidth,
            .height = textureHeight,
            .depthOrArrayLayers = batchSize,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
            .label = "Large barrier batch layered texture",
        });
        REQUIRE(layered.has_value());

        std::array<rachis::BufferBarrier, batchSize> bufferBarriers{};
        std::array<rachis::TextureBarrier, batchSize> textureBarriers{};
        for (std::uint32_t index = 0; index < batchSize; ++index)
        {
            bufferBarriers[index] = rachis::BufferBarrier{
                .buffer = buffers[index],
                .stagesBefore = rachis::PipelineStage::all,
                .stagesAfter = rachis::PipelineStage::copy,
                .accessBefore = rachis::Access::common,
                .accessAfter = rachis::Access::copyDestination,
            };
            textureBarriers[index] = rachis::TextureBarrier{
                .texture = *layered,
                .stagesAfter = rachis::PipelineStage::copy,
                .accessAfter = rachis::Access::copyDestination,
                .layoutBefore = rachis::TextureLayout::undefined,
                .layoutAfter = rachis::TextureLayout::copyDestination,
                .subresources = {.firstArrayLayer = index, .arrayLayerCount = 1},
            };
        }
        constexpr auto unknownStage = static_cast<rachis::PipelineStage>(1U << 31U);
        constexpr std::string_view repeatedBufferMessage = "multiple transitions for one buffer";
        constexpr std::string_view bufferScopeMessage = "stage/access scope";
        constexpr std::string_view overlappingTextureMessage = "overlapping transitions for one texture";
        constexpr std::string_view textureScopeMessage = "unknown stage, access, or layout value";

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        const auto rejectedBecause = [&](const rachis::ResourceBarrierBatch& batch, std::string_view reason) {
            const auto rejected = device.resourceBarriers(*command, batch);
            REQUIRE_FALSE(rejected.has_value());
            CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
            CAPTURE(rejected.error().message);
            CHECK(rejected.error().message.contains(reason));
        };

        auto repeatedLast = bufferBarriers;
        repeatedLast[batchSize - 1U].buffer = buffers[3];
        rejectedBecause({.bufferBarriers = repeatedLast}, repeatedBufferMessage);
        auto repeatedBeforeInvalid = bufferBarriers;
        repeatedBeforeInvalid[5].buffer = buffers[2];
        repeatedBeforeInvalid[10].stagesAfter = unknownStage;
        rejectedBecause({.bufferBarriers = repeatedBeforeInvalid}, repeatedBufferMessage);
        auto invalidBeforeRepeated = bufferBarriers;
        invalidBeforeRepeated[4].stagesAfter = unknownStage;
        invalidBeforeRepeated[10].buffer = buffers[1];
        rejectedBecause({.bufferBarriers = invalidBeforeRepeated}, bufferScopeMessage);

        auto overlappingLast = textureBarriers;
        overlappingLast[batchSize - 1U].subresources = {.firstArrayLayer = 2, .arrayLayerCount = 3};
        rejectedBecause({.textureBarriers = overlappingLast}, overlappingTextureMessage);
        auto overlappingBeforeInvalid = textureBarriers;
        overlappingBeforeInvalid[4].subresources = {.firstArrayLayer = 0, .arrayLayerCount = 2};
        overlappingBeforeInvalid[9].stagesAfter = unknownStage;
        rejectedBecause({.textureBarriers = overlappingBeforeInvalid}, overlappingTextureMessage);
        auto invalidBeforeOverlapping = textureBarriers;
        invalidBeforeOverlapping[3].stagesAfter = unknownStage;
        invalidBeforeOverlapping[9].subresources = {.firstArrayLayer = 8, .arrayLayerCount = 2};
        rejectedBecause({.textureBarriers = invalidBeforeOverlapping}, textureScopeMessage);

        REQUIRE(
            device.resourceBarriers(*command, {.bufferBarriers = bufferBarriers, .textureBarriers = textureBarriers})
                .has_value());
        for (const auto buffer : buffers)
        {
            REQUIRE(device
                        .copyBuffer(*command,
                                    {
                                        .source = *upload,
                                        .destination = buffer,
                                        .size = bufferSize,
                                    })
                        .has_value());
        }
        submitAndWait(device, std::move(*command));

        REQUIRE(device.destroyTexture(*layered).has_value());
        for (const auto buffer : buffers)
        {
            REQUIRE(device.destroyBuffer(buffer).has_value());
        }
        REQUIRE(device.destroyBuffer(*upload).has_value());
        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.warnings == 0);
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
    }
}
