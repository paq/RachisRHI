#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include "Internal/VulkanResources.h"
#include "Vulkan/VulkanTestHooks.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

namespace
{

    constexpr std::uint64_t bufferSize = 256;
    constexpr std::uint32_t textureWidth = 4;
    constexpr std::uint32_t textureHeight = 4;
    constexpr std::uint32_t textureBytesPerPixel = 4;

    [[nodiscard]] rachis::DeviceDesc vulkanDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] std::byte pattern(std::uint32_t index) noexcept
    {
        return static_cast<std::byte>((index * 17U + 9U) & 0xFFU);
    }

    class ResourceBarrierCapture final
    {
    public:
        ResourceBarrierCapture() noexcept
        {
            rachis::vulkan::captureVulkanResourceBarriers(&barriers_);
        }

        ~ResourceBarrierCapture()
        {
            rachis::vulkan::captureVulkanResourceBarriers(nullptr);
        }

        ResourceBarrierCapture(const ResourceBarrierCapture&) = delete;
        ResourceBarrierCapture& operator=(const ResourceBarrierCapture&) = delete;
        ResourceBarrierCapture(ResourceBarrierCapture&&) = delete;
        ResourceBarrierCapture& operator=(ResourceBarrierCapture&&) = delete;

        [[nodiscard]] std::vector<rachis::vulkan::VulkanCapturedBarrier> take()
        {
            return std::exchange(barriers_, {});
        }

    private:
        std::vector<rachis::vulkan::VulkanCapturedBarrier> barriers_;
    };

    class ForcedAllCommandsBarrierSources final
    {
    public:
        explicit ForcedAllCommandsBarrierSources(bool enabled) noexcept : enabled_(enabled)
        {
            if (enabled_)
            {
                rachis::vulkan::test::forceAllCommandsBarrierSources().store(true, std::memory_order_relaxed);
            }
        }

        ~ForcedAllCommandsBarrierSources()
        {
            if (enabled_)
            {
                rachis::vulkan::test::forceAllCommandsBarrierSources().store(false, std::memory_order_relaxed);
            }
        }

        ForcedAllCommandsBarrierSources(const ForcedAllCommandsBarrierSources&) = delete;
        ForcedAllCommandsBarrierSources& operator=(const ForcedAllCommandsBarrierSources&) = delete;
        ForcedAllCommandsBarrierSources(ForcedAllCommandsBarrierSources&&) = delete;
        ForcedAllCommandsBarrierSources& operator=(ForcedAllCommandsBarrierSources&&) = delete;

    private:
        bool enabled_ = false;
    };

} // namespace

TEST_CASE("Vulkan barriers wait on all earlier commands on portability-subset devices",
          "[size:medium][rhi][vulkan][barrier][sync][portability]")
{
    const bool forced = GENERATE(false, true);
    CAPTURE(forced);
    auto created = [forced] {
        const ForcedAllCommandsBarrierSources force(forced);
        return rachis::createDevice(vulkanDesc());
    }();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
#ifdef __APPLE__
    const bool widened = true;
#else
    const bool widened = forced;
#endif

    auto texture = device.createTexture(rachis::TextureDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::copySource,
        .label = "Vulkan portability barrier texture",
    });
    auto buffer = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Vulkan portability barrier buffer",
    });
    REQUIRE(texture);
    REQUIRE(buffer);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    std::vector<rachis::vulkan::VulkanCapturedBarrier> captured;
    {
        ResourceBarrierCapture capture;
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = *texture,
                                        .stagesAfter = rachis::PipelineStage::pixelShader,
                                        .accessAfter = rachis::Access::shaderResource,
                                        .layoutAfter = rachis::TextureLayout::shaderResource,
                                    })
                    .has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = *texture,
                                        .stagesBefore = rachis::PipelineStage::pixelShader,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::shaderResource,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::shaderResource,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        REQUIRE(device
                    .bufferBarrier(*command,
                                   rachis::BufferBarrier{
                                       .buffer = *buffer,
                                       .stagesBefore = rachis::PipelineStage::computeShader,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::unorderedAccess,
                                       .accessAfter = rachis::Access::copySource,
                                   })
                    .has_value());
        captured = capture.take();
    }
    REQUIRE(captured.size() == 3);
    constexpr VkPipelineStageFlags2 copyStages =
        VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT;
    CHECK(captured[0].srcStageMask == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    CHECK(captured[0].dstStageMask == VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
    CHECK(captured[1].srcStageMask ==
          (widened ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT));
    CHECK(captured[1].dstStageMask == copyStages);
    CHECK(captured[2].srcStageMask == (widened
                                           ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
                                           : (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT)));
    CHECK(captured[2].dstStageMask == copyStages);

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyBuffer(*buffer).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan barrier batches validate atomically and reject illegal scopes",
          "[size:medium][rhi][vulkan][barrier][batch]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).bufferBarrierRanges);

    auto first = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .label = "Vulkan barrier first",
    });
    auto second = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .label = "Vulkan barrier second",
    });
    auto texture = device.createTexture(rachis::TextureDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage =
            rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination | rachis::TextureUsage::sampled,
        .label = "Vulkan barrier texture",
    });
    auto stale = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
    });
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(texture);
    REQUIRE(stale);
    const auto staleHandle = *stale;
    REQUIRE(device.destroyBuffer(*stale).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command);
    CHECK_FALSE(device.resourceBarriers(*command, {}).has_value());

    const std::array initialBuffers{
        rachis::BufferBarrier{
            .buffer = *first,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::common,
            .accessAfter = rachis::Access::copyDestination,
        },
        rachis::BufferBarrier{
            .buffer = *second,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::common,
            .accessAfter = rachis::Access::copyDestination,
        },
    };
    const std::array initialTextures{
        rachis::TextureBarrier{
            .texture = *texture,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessAfter = rachis::Access::copyDestination,
            .layoutBefore = rachis::TextureLayout::undefined,
            .layoutAfter = rachis::TextureLayout::copyDestination,
        },
    };

    const std::array duplicate{initialBuffers[0], initialBuffers[0]};
    CHECK_FALSE(device.resourceBarriers(*command, {.bufferBarriers = duplicate}).has_value());
    const std::array overlapping{initialTextures[0], initialTextures[0]};
    CHECK_FALSE(device.resourceBarriers(*command, {.textureBarriers = overlapping}).has_value());

    auto mismatched = initialTextures[0];
    mismatched.accessAfter = rachis::Access::copySource;
    CHECK_FALSE(device.resourceBarriers(*command, {.textureBarriers = std::array{mismatched}}).has_value());

    auto unknownStage = initialBuffers[0];
    unknownStage.stagesAfter = static_cast<rachis::PipelineStage>(1U << 31U);
    CHECK_FALSE(device.resourceBarriers(*command, {.bufferBarriers = std::array{unknownStage}}).has_value());

    auto invalidCommon = rachis::GlobalBarrier{
        .stagesBefore = rachis::PipelineStage::all,
        .stagesAfter = rachis::PipelineStage::copy,
        .accessBefore = rachis::Access::common,
        .accessAfter = rachis::Access::copyDestination,
    };
    CHECK_FALSE(device.resourceBarriers(*command, {.globalBarriers = std::array{invalidCommon}}).has_value());

    auto unknownMemory = rachis::GlobalBarrier{
        .stagesBefore = rachis::PipelineStage::copy,
        .stagesAfter = static_cast<rachis::PipelineStage>(1U << 31U),
        .accessBefore = rachis::Access::copyDestination,
        .accessAfter = rachis::Access::copyDestination,
    };
    CHECK_FALSE(device
                    .resourceBarriers(*command,
                                      {
                                          .globalBarriers = std::array{unknownMemory},
                                          .bufferBarriers = initialBuffers,
                                          .textureBarriers = initialTextures,
                                      })
                    .has_value());

    const std::array staleBatch{
        initialBuffers[0],
        rachis::BufferBarrier{
            .buffer = staleHandle,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::common,
            .accessAfter = rachis::Access::copyDestination,
        },
    };
    CHECK_FALSE(device.resourceBarriers(*command, {.bufferBarriers = staleBatch}).has_value());

    REQUIRE(device
                .resourceBarriers(*command,
                                  {
                                      .bufferBarriers = initialBuffers,
                                      .textureBarriers = initialTextures,
                                  })
                .has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan buffer barrier ranges are accepted and do not publish on failure",
          "[size:medium][rhi][vulkan][barrier][range]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    CHECK(rachis::test::requireCapabilities(device).bufferBarrierRanges);

    auto buffer = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .label = "Vulkan ranged buffer",
    });
    REQUIRE(buffer);
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command);

    auto emptyRange = device.bufferBarrier(*command,
                                           rachis::BufferBarrier{
                                               .buffer = *buffer,
                                               .stagesAfter = rachis::PipelineStage::copy,
                                               .accessAfter = rachis::Access::copyDestination,
                                               .offset = 16,
                                               .size = 0,
                                           });
    REQUIRE_FALSE(emptyRange);
    CHECK(emptyRange.error().code == rachis::ErrorCode::invalidArgument);

    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *buffer,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessAfter = rachis::Access::copyDestination,
                                   .offset = 16,
                                   .size = 64,
                               })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *buffer,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::copySource,
                                   .offset = 16,
                                   .size = 64,
                               })
                .has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan upload sample depth and cross-queue barriers complete under sync validation",
          "[size:medium][rhi][vulkan][barrier][sync]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto upload = device.createBuffer(rachis::BufferDesc{
        .size = std::uint64_t{textureWidth} * textureHeight * textureBytesPerPixel,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan barrier upload",
    });
    auto sampled = device.createTexture(rachis::TextureDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage =
            rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination | rachis::TextureUsage::sampled,
        .label = "Vulkan sampled color",
    });
    auto depth = device.createTexture(rachis::TextureDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::depthStencil | rachis::TextureUsage::sampled,
        .label = "Vulkan sampled depth",
    });
    auto filled = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan compute fill",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = std::uint64_t{textureWidth} * textureHeight * textureBytesPerPixel,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan texture readback",
    });
    REQUIRE(upload);
    REQUIRE(sampled);
    REQUIRE(depth);
    REQUIRE(filled);
    REQUIRE(readback);

    auto mapped = device.mapBuffer(*upload);
    REQUIRE(mapped);
    for (std::uint32_t index = 0; index < textureWidth * textureHeight * textureBytesPerPixel; ++index)
    {
        mapped->data[index] = pattern(index);
    }
    REQUIRE(device.unmapBuffer(*upload).has_value());
    REQUIRE(rachis::vulkan::copyVulkanBufferToTextureImmediate(*upload, *sampled, 0, 0).has_value());

    auto copy = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(copy);
    REQUIRE(device
                .textureBarrier(*copy,
                                rachis::TextureBarrier{
                                    .texture = *sampled,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::copySource,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::copySource,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    auto copyPoint = device.submit(std::move(*copy));
    REQUIRE(copyPoint);

    auto compute = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(compute);
    REQUIRE(rachis::vulkan::recordVulkanFillBuffer(*compute, *filled, 0xA5A5A5A5U).has_value());
    REQUIRE(device
                .bufferBarrier(*compute,
                               rachis::BufferBarrier{
                                   .buffer = *filled,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    auto computePoint = device.submit(std::move(*compute), std::array{*copyPoint});
    REQUIRE(computePoint);

    auto graphics = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(graphics);
    REQUIRE(device
                .textureBarrier(*graphics,
                                rachis::TextureBarrier{
                                    .texture = *sampled,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::pixelShader,
                                    .accessBefore = rachis::Access::copySource,
                                    .accessAfter = rachis::Access::shaderResource,
                                    .layoutBefore = rachis::TextureLayout::copySource,
                                    .layoutAfter = rachis::TextureLayout::shaderResource,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*graphics,
                                rachis::TextureBarrier{
                                    .texture = *depth,
                                    .stagesAfter = rachis::PipelineStage::depthStencil,
                                    .accessAfter = rachis::Access::depthStencilWrite,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*graphics,
                                rachis::TextureBarrier{
                                    .texture = *depth,
                                    .stagesBefore = rachis::PipelineStage::depthStencil,
                                    .stagesAfter = rachis::PipelineStage::pixelShader,
                                    .accessBefore = rachis::Access::depthStencilWrite,
                                    .accessAfter = rachis::Access::shaderResource,
                                    .layoutBefore = rachis::TextureLayout::depthStencilWrite,
                                    .layoutAfter = rachis::TextureLayout::shaderResource,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());
    auto graphicsPoint = device.submit(std::move(*graphics), std::array{*computePoint});
    REQUIRE(graphicsPoint);
    REQUIRE(device.wait(*graphicsPoint).has_value());

    REQUIRE(rachis::vulkan::copyVulkanTextureToBufferImmediate(*sampled, *readback, 0, 0).has_value());
    auto readMapped = device.mapBuffer(*readback);
    REQUIRE(readMapped);
    std::uint32_t mismatch = textureWidth * textureHeight * textureBytesPerPixel;
    for (std::uint32_t index = 0; index < mismatch; ++index)
    {
        if (readMapped->data[index] != pattern(index))
        {
            mismatch = index;
            break;
        }
    }
    CHECK(mismatch == textureWidth * textureHeight * textureBytesPerPixel);
    REQUIRE(device.unmapBuffer(*readback).has_value());

    std::vector<std::byte> filledBytes(bufferSize);
    REQUIRE(device.readBuffer(*filled, 0, filledBytes).has_value());
    std::uint32_t observed = 0;
    std::memcpy(&observed, filledBytes.data(), sizeof(observed));
    CHECK(observed == 0xA5A5A5A5U);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan vertex-buffer barriers accept the vertex input stage and stay valid in the vertex shader stage",
          "[size:medium][rhi][vulkan][barrier][vertex-input]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto upload = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan vertex upload",
    });
    auto vertices = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::vertex | rachis::BufferUsage::copyDestination,
        .label = "Vulkan device-local vertices",
    });
    REQUIRE(upload);
    REQUIRE(vertices);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device.copyBuffer(*command, {.source = *upload, .destination = *vertices, .size = bufferSize}).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *vertices,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::vertexInput,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::vertexBuffer,
                               })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *vertices,
                                   .stagesBefore = rachis::PipelineStage::vertexInput,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::vertexBuffer,
                                   .accessAfter = rachis::Access::copyDestination,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*command, {.source = *upload, .destination = *vertices, .size = bufferSize}).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *vertices,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::vertexShader,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::vertexBuffer,
                               })
                .has_value());
    auto pixelOnly = device.bufferBarrier(*command,
                                          rachis::BufferBarrier{
                                              .buffer = *vertices,
                                              .stagesBefore = rachis::PipelineStage::vertexShader,
                                              .stagesAfter = rachis::PipelineStage::pixelShader,
                                              .accessBefore = rachis::Access::vertexBuffer,
                                              .accessAfter = rachis::Access::vertexBuffer,
                                          });
    REQUIRE_FALSE(pixelOnly);
    CHECK(pixelOnly.error().code == rachis::ErrorCode::invalidArgument);

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    REQUIRE(device.destroyBuffer(*vertices).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan acceleration-structure buffer barriers are accepted and clean under sync validation",
          "[size:medium][rhi][vulkan][barrier][acceleration-structure]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.rayQuery && !capabilities.rayTracing)
    {
        SKIP("The Vulkan adapter does not advertise acceleration structures.");
    }

    const std::array positions{
        std::array{-1.0F, -1.0F, 0.0F},
        std::array{0.0F, 1.0F, 0.0F},
        std::array{1.0F, -1.0F, 0.0F},
    };
    for (const auto queueType : {rachis::QueueType::graphics, rachis::QueueType::compute})
    {
        auto upload = device.createBuffer(rachis::BufferDesc{
            .size = sizeof(positions),
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Vulkan build-input upload",
        });
        auto vertices = device.createBuffer(rachis::BufferDesc{
            .size = sizeof(positions),
            .usage = rachis::BufferUsage::accelerationStructureBuildInput | rachis::BufferUsage::copyDestination,
            .label = "Vulkan device-local build input",
        });
        REQUIRE(upload);
        REQUIRE(vertices);
        REQUIRE(device.writeBuffer(*upload, 0, std::as_bytes(std::span(positions))).has_value());

        const std::array geometry{
            rachis::AccelerationStructureGeometry{
                .data =
                    rachis::AccelerationStructureTriangleGeometry{
                        .vertexBuffer = *vertices,
                        .vertexCount = 3,
                        .vertexStride = sizeof(positions[0]),
                        .vertexFormat = rachis::AccelerationStructureVertexFormat::float32x3,
                    },
                .flags = rachis::AccelerationStructureGeometryFlag::opaque,
            },
        };
        const rachis::BottomLevelAccelerationStructureBuildInput input{
            .geometries = geometry,
            .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
        };
        auto sizes = device.queryBottomLevelAccelerationStructureBuildSizes(input);
        REQUIRE(sizes);
        auto firstStorage = device.createBuffer(rachis::BufferDesc{
            .size = sizes->accelerationStructureSize,
            .usage = rachis::BufferUsage::accelerationStructureStorage,
            .label = "Vulkan first BLAS storage",
        });
        auto secondStorage = device.createBuffer(rachis::BufferDesc{
            .size = sizes->accelerationStructureSize,
            .usage = rachis::BufferUsage::accelerationStructureStorage,
            .label = "Vulkan second BLAS storage",
        });
        auto scratch = device.createBuffer(rachis::BufferDesc{
            .size = sizes->buildScratchSize,
            .usage = rachis::BufferUsage::accelerationStructureScratch,
            .label = "Vulkan shared BLAS scratch",
        });
        REQUIRE(firstStorage);
        REQUIRE(secondStorage);
        REQUIRE(scratch);
        auto first = device.createAccelerationStructure({
            .type = rachis::AccelerationStructureType::bottomLevel,
            .storage = *firstStorage,
            .label = "Vulkan first BLAS",
        });
        auto second = device.createAccelerationStructure({
            .type = rachis::AccelerationStructureType::bottomLevel,
            .storage = *secondStorage,
            .label = "Vulkan second BLAS",
        });
        REQUIRE(first);
        REQUIRE(second);

        auto command = rachis::test::acquireCommandBuffer(device, queueType);
        REQUIRE(command);
        REQUIRE(device.copyBuffer(*command, {.source = *upload, .destination = *vertices, .size = sizeof(positions)})
                    .has_value());
        const std::array buildInputBarrier{
            rachis::BufferBarrier{
                .buffer = *vertices,
                .stagesBefore = rachis::PipelineStage::copy,
                .stagesAfter = rachis::PipelineStage::accelerationStructureBuild,
                .accessBefore = rachis::Access::copyDestination,
                .accessAfter = rachis::Access::accelerationStructureBuildInput,
            },
        };
        REQUIRE(device.resourceBarriers(*command, {.bufferBarriers = buildInputBarrier}).has_value());
        REQUIRE(device
                    .buildBottomLevelAccelerationStructure(*command,
                                                           {.destination = *first, .input = input, .scratch = *scratch})
                    .has_value());
        REQUIRE(device
                    .bufferBarrier(*command,
                                   rachis::BufferBarrier{
                                       .buffer = *scratch,
                                       .stagesBefore = rachis::PipelineStage::accelerationStructureBuild,
                                       .stagesAfter = rachis::PipelineStage::accelerationStructureBuild,
                                       .accessBefore = rachis::Access::accelerationStructureScratch,
                                       .accessAfter = rachis::Access::accelerationStructureScratch,
                                   })
                    .has_value());
        REQUIRE(device
                    .buildBottomLevelAccelerationStructure(
                        *command, {.destination = *second, .input = input, .scratch = *scratch})
                    .has_value());

        rachis::BufferHandle shaderTable{};
        if (capabilities.rayTracing)
        {
            auto table = device.createBuffer(rachis::BufferDesc{
                .size = bufferSize,
                .usage = rachis::BufferUsage::shaderBindingTable | rachis::BufferUsage::copyDestination,
                .label = "Vulkan copied shader table",
            });
            REQUIRE(table);
            shaderTable = *table;
            REQUIRE(
                device.copyBuffer(*command, {.source = *upload, .destination = shaderTable, .size = sizeof(positions)})
                    .has_value());
            REQUIRE(device
                        .bufferBarrier(*command,
                                       rachis::BufferBarrier{
                                           .buffer = shaderTable,
                                           .stagesBefore = rachis::PipelineStage::copy,
                                           .stagesAfter = rachis::PipelineStage::rayTracingShader,
                                           .accessBefore = rachis::Access::copyDestination,
                                           .accessAfter = rachis::Access::shaderResource,
                                       })
                        .has_value());
        }

        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted).has_value());
        REQUIRE(device.destroyAccelerationStructure(*first).has_value());
        REQUIRE(device.destroyAccelerationStructure(*second).has_value());
        for (const auto buffer : {*upload, *vertices, *firstStorage, *secondStorage, *scratch})
        {
            REQUIRE(device.destroyBuffer(buffer).has_value());
        }
        if (capabilities.rayTracing)
        {
            REQUIRE(device.destroyBuffer(shaderTable).has_value());
        }
    }
    device.collectGarbage();

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan discard barriers order their layout transition after earlier writes under sync validation",
          "[size:medium][rhi][vulkan][barrier][sync]")
{
    const auto stagesBefore =
        GENERATE(rachis::PipelineStage::none, rachis::PipelineStage::copy, rachis::PipelineStage::all);
    CAPTURE(stagesBefore);
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto texture = device.createTexture(rachis::TextureDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Vulkan discarded color",
    });
    REQUIRE(texture);
    const rachis::BufferTextureCopyRegion region{.extent = {.width = textureWidth, .height = textureHeight}};
    auto layout = device.queryTextureTransferLayout(*texture, region);
    REQUIRE(layout);
    const auto imageSize = layout->requiredBufferSize;
    const auto alignment = (std::max)(std::uint64_t{layout->bufferOffsetAlignment}, std::uint64_t{1});
    const auto secondOffset = (imageSize + alignment - 1U) / alignment * alignment;
    auto upload = device.createBuffer(rachis::BufferDesc{
        .size = secondOffset + imageSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan discard upload",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = imageSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan discard readback",
    });
    REQUIRE(upload);
    REQUIRE(readback);

    std::vector<std::byte> pixels(static_cast<std::size_t>(secondOffset + imageSize), std::byte{0x5A});
    for (std::uint64_t index = 0; index < imageSize; ++index)
    {
        pixels[static_cast<std::size_t>(secondOffset + index)] = pattern(static_cast<std::uint32_t>(index));
    }
    REQUIRE(device.writeBuffer(*upload, 0, pixels).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *texture,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *texture, std::array{region}).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *texture,
                                    .stagesBefore = stagesBefore,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    auto second = region;
    second.bufferOffset = secondOffset;
    REQUIRE(device.copyBufferToTexture(*command, *upload, *texture, std::array{second}).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *texture, *readback, std::array{region}).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());

    std::vector<std::byte> observed(static_cast<std::size_t>(imageSize));
    REQUIRE(device.readBuffer(*readback, 0, observed).has_value());
    std::uint64_t mismatch = imageSize;
    for (std::uint64_t index = 0; index < imageSize; ++index)
    {
        if (observed[static_cast<std::size_t>(index)] != pattern(static_cast<std::uint32_t>(index)))
        {
            mismatch = index;
            break;
        }
    }
    CHECK(mismatch == imageSize);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan rejects texture barriers whose layoutBefore is not the texture's current layout",
          "[size:medium][rhi][vulkan][barrier][layout]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    auto texture = device.createTexture(rachis::TextureDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "Vulkan layoutBefore texture",
    });
    REQUIRE(texture);

    const rachis::TextureBarrier toCopyDestination{
        .texture = *texture,
        .stagesAfter = rachis::PipelineStage::copy,
        .accessAfter = rachis::Access::copyDestination,
        .layoutBefore = rachis::TextureLayout::undefined,
        .layoutAfter = rachis::TextureLayout::copyDestination,
    };
    const rachis::TextureBarrier copyDestinationToSampled{
        .texture = *texture,
        .stagesBefore = rachis::PipelineStage::copy,
        .stagesAfter = rachis::PipelineStage::pixelShader,
        .accessBefore = rachis::Access::copyDestination,
        .accessAfter = rachis::Access::shaderResource,
        .layoutBefore = rachis::TextureLayout::copyDestination,
        .layoutAfter = rachis::TextureLayout::shaderResource,
    };

    auto first = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(first);
    REQUIRE(device.textureBarrier(*first, toCopyDestination).has_value());
    REQUIRE(device.textureBarrier(*first, copyDestinationToSampled).has_value());
    const auto repeated = device.textureBarrier(*first, copyDestinationToSampled);
    REQUIRE_FALSE(repeated.has_value());
    CHECK(repeated.error().code == rachis::ErrorCode::invalidState);
    auto firstPoint = device.submit(std::move(*first));
    REQUIRE(firstPoint);

    std::array stale{rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics).value()};
    REQUIRE(device.textureBarrier(stale[0], copyDestinationToSampled).has_value());
    const auto rejected = device.submit(stale);
    REQUIRE_FALSE(rejected.has_value());
    CHECK(rejected.error().code == rachis::ErrorCode::invalidState);

    auto last = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(last);
    REQUIRE(device
                .textureBarrier(*last,
                                rachis::TextureBarrier{
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::pixelShader,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::shaderResource,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::shaderResource,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*last,
                                rachis::TextureBarrier{
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::resolve,
                                    .accessBefore = rachis::Access::copySource,
                                    .accessAfter = rachis::Access::resolveSource,
                                    .layoutBefore = rachis::TextureLayout::copySource,
                                    .layoutAfter = rachis::TextureLayout::resolveSource,
                                })
                .has_value());
    const auto fromCopySource = device.textureBarrier(*last,
                                                      rachis::TextureBarrier{
                                                          .texture = *texture,
                                                          .layoutBefore = rachis::TextureLayout::copySource,
                                                          .layoutAfter = rachis::TextureLayout::copyDestination,
                                                      });
    REQUIRE_FALSE(fromCopySource.has_value());
    CHECK(fromCopySource.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device
                .textureBarrier(*last,
                                rachis::TextureBarrier{
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::resolve,
                                    .stagesAfter = rachis::PipelineStage::computeShader,
                                    .accessBefore = rachis::Access::resolveSource,
                                    .accessAfter = rachis::Access::unorderedAccess,
                                    .layoutBefore = rachis::TextureLayout::resolveSource,
                                    .layoutAfter = rachis::TextureLayout::unorderedAccess,
                                })
                .has_value());
    const auto fromCommon = device.textureBarrier(*last,
                                                  rachis::TextureBarrier{
                                                      .texture = *texture,
                                                      .layoutBefore = rachis::TextureLayout::common,
                                                      .layoutAfter = rachis::TextureLayout::copyDestination,
                                                  });
    REQUIRE_FALSE(fromCommon.has_value());
    CHECK(fromCommon.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.textureBarrier(*last, {.texture = *texture, .layoutAfter = rachis::TextureLayout::copyDestination})
                .has_value());
    auto lastPoint = device.submit(std::move(*last), std::array{*firstPoint});
    REQUIRE(lastPoint);
    REQUIRE(device.wait(*lastPoint).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
