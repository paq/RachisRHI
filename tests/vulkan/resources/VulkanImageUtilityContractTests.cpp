#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace
{

    constexpr std::uint32_t copyWidth = 4;
    constexpr std::uint32_t copyHeight = 4;
    constexpr std::uint32_t bytesPerPixel = 4;
    constexpr std::uint32_t mipLevels = 3;

    [[nodiscard]] rachis::DeviceDesc vulkanDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] std::byte texelByte(std::uint32_t x, std::uint32_t y, std::uint32_t channel) noexcept
    {
        return static_cast<std::byte>((x * 31U + y * 47U + channel * 59U + 11U) & 0xFFU);
    }

    [[nodiscard]] std::array<std::uint8_t, 4> baseTexel(std::uint32_t x, std::uint32_t y) noexcept
    {
        return {
            static_cast<std::uint8_t>(x * 40U),
            static_cast<std::uint8_t>(y * 40U),
            static_cast<std::uint8_t>((x + y) * 20U),
            255,
        };
    }

    [[nodiscard]] std::array<std::uint8_t, 4> mip1Texel(std::uint32_t x, std::uint32_t y) noexcept
    {
        return {
            static_cast<std::uint8_t>(20U + x * 80U),
            static_cast<std::uint8_t>(20U + y * 80U),
            static_cast<std::uint8_t>(20U + (x + y) * 40U),
            255,
        };
    }

    [[nodiscard]] bool nearByte(std::byte actual, std::uint8_t expected) noexcept
    {
        const auto value = std::to_integer<std::uint8_t>(actual);
        const auto lower = expected == 0 ? 0U : static_cast<std::uint32_t>(expected) - 1U;
        const auto upper = (std::min)(static_cast<std::uint32_t>(expected) + 1U, 255U);
        return value >= lower && value <= upper;
    }

    class ScopedEnvironmentVariable final
    {
    public:
        ScopedEnvironmentVariable(const char* name, const char* value) : name_(name)
        {
#ifdef _WIN32
            char* previous = nullptr;
            std::size_t length = 0;
            if (_dupenv_s(&previous, &length, name) == 0 && previous != nullptr)
            {
                previous_ = previous;
                hadPrevious_ = true;
            }
            std::free(previous);
            static_cast<void>(_putenv_s(name, value));
#else
            if (const char* previous = std::getenv(name); previous != nullptr)
            {
                previous_ = previous;
                hadPrevious_ = true;
            }
            static_cast<void>(setenv(name, value, 1));
#endif
        }

        ~ScopedEnvironmentVariable()
        {
#ifdef _WIN32
            static_cast<void>(_putenv_s(name_, hadPrevious_ ? previous_.c_str() : ""));
#else
            if (hadPrevious_)
            {
                static_cast<void>(setenv(name_, previous_.c_str(), 1));
            }
            else
            {
                static_cast<void>(unsetenv(name_));
            }
#endif
        }

        ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
        ScopedEnvironmentVariable& operator=(const ScopedEnvironmentVariable&) = delete;
        ScopedEnvironmentVariable(ScopedEnvironmentVariable&&) = delete;
        ScopedEnvironmentVariable& operator=(ScopedEnvironmentVariable&&) = delete;

    private:
        const char* name_;
        std::string previous_;
        bool hadPrevious_ = false;
    };

#ifdef RACHIS_SPIRV_DIR
    constexpr std::uint32_t renderedSize = 2048;
    constexpr std::uint32_t renderedMipLevels = 12;
    constexpr std::uint32_t renderedLastMip = renderedMipLevels - 1;
    constexpr std::uint32_t renderedIterations = 4;

    [[nodiscard]] std::vector<std::byte> loadSpirv(std::string_view name)
    {
        std::ifstream stream(std::filesystem::path(RACHIS_SPIRV_DIR) / name, std::ios::binary | std::ios::ate);
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

    [[nodiscard]] rachis::PipelineHandle createRenderedMipPipeline(rachis::Device& device,
                                                                   std::string_view pixelSpirv,
                                                                   std::string_view pixelEntry,
                                                                   const rachis::BlendState& blend)
    {
        const auto vertexBytes = loadSpirv("MipmapAfterRendering.vs.spv");
        const auto pixelBytes = loadSpirv(pixelSpirv);
        REQUIRE_FALSE(vertexBytes.empty());
        REQUIRE_FALSE(pixelBytes.empty());
        auto vertex = device.createShaderModule({
            .stage = rachis::ShaderStage::vertex,
            .bytecode = vertexBytes,
            .entryPoint = "fullscreenVertex",
        });
        auto pixel = device.createShaderModule({
            .stage = rachis::ShaderStage::pixel,
            .bytecode = pixelBytes,
            .entryPoint = pixelEntry,
        });
        auto layout = device.createPipelineLayout({});
        REQUIRE(vertex);
        REQUIRE(pixel);
        REQUIRE(layout);
        const std::array targets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm, .blend = blend}};
        auto pipeline = device.createGraphicsPipeline({
            .layout = *layout,
            .vertexShader = *vertex,
            .pixelShader = *pixel,
            .cullMode = rachis::CullMode::none,
            .colorTargets = targets,
            .label = "Vulkan rendered mip pipeline",
        });
        REQUIRE(pipeline);
        return *pipeline;
    }

    [[nodiscard]] rachis::TextureDesc renderedMipDesc() noexcept
    {
        return rachis::TextureDesc{
            .width = renderedSize,
            .height = renderedSize,
            .mipLevels = renderedMipLevels,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::sampled |
                     rachis::TextureUsage::storage | rachis::TextureUsage::copySource,
            .label = "Vulkan rendered mip chain",
        };
    }

    [[nodiscard]] rachis::TextureViewHandle createRenderedMipView(rachis::Device& device,
                                                                  rachis::TextureHandle texture,
                                                                  std::uint32_t mipLevel)
    {
        auto view = device.createTextureView({
            .texture = texture,
            .usage = rachis::TextureViewUsage::renderTarget,
            .format = rachis::Format::rgba8Unorm,
            .firstMipLevel = mipLevel,
            .mipLevelCount = 1,
        });
        REQUIRE(view);
        return *view;
    }

    void recordRenderedMip(rachis::Device& device,
                           rachis::CommandBuffer& command,
                           rachis::TextureViewHandle view,
                           rachis::PipelineHandle pipeline,
                           std::uint32_t size,
                           std::uint32_t instanceCount,
                           rachis::LoadOperation loadOperation)
    {
        const std::array attachments{rachis::ColorAttachment{
            .view = view,
            .loadOperation = loadOperation,
            .clearValue = {.red = 0.0F, .green = 0.0F, .blue = 1.0F, .alpha = 1.0F},
        }};
        REQUIRE(device.beginRendering(command, {.colorAttachments = attachments}).has_value());
        REQUIRE(device.bindGraphicsPipeline(command, pipeline).has_value());
        REQUIRE(device.setViewport(command, {.width = static_cast<float>(size), .height = static_cast<float>(size)})
                    .has_value());
        REQUIRE(device.setScissor(command, {.width = size, .height = size}).has_value());
        REQUIRE(device.draw(command, 3, instanceCount).has_value());
        REQUIRE(device.endRendering(command).has_value());
    }

    void recordRenderedBase(rachis::Device& device,
                            rachis::CommandBuffer& command,
                            rachis::TextureHandle texture,
                            rachis::TextureViewHandle baseView,
                            rachis::PipelineHandle seed)
    {
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = texture,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                        .subresources = {.firstMipLevel = 0, .mipLevelCount = 1},
                                    })
                    .has_value());
        recordRenderedMip(device, command, baseView, seed, renderedSize, 2, rachis::LoadOperation::clear);
    }

    [[nodiscard]] std::array<std::uint32_t, 4> readRenderedLastMip(rachis::Device& device,
                                                                   rachis::CommandBuffer command,
                                                                   rachis::TextureHandle texture)
    {
        const rachis::BufferTextureCopyRegion lastMip{
            .textureSubresource = {.mipLevel = renderedLastMip},
            .extent = {.width = 1, .height = 1},
        };
        auto layout = device.queryTextureTransferLayout(texture, lastMip);
        REQUIRE(layout);
        auto readback = device.createBuffer({
            .size = layout->requiredBufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(readback);
        REQUIRE(device.copyTextureToBuffer(command, texture, *readback, std::array{lastMip}).has_value());
        auto submitted = device.submit(std::move(command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted).has_value());
        auto mapped = device.mapBuffer(*readback);
        REQUIRE(mapped);
        std::array<std::uint32_t, 4> texel{};
        for (std::size_t channel = 0; channel < texel.size(); ++channel)
        {
            texel[channel] = std::to_integer<std::uint32_t>(mapped->data[channel]);
        }
        REQUIRE(device.unmapBuffer(*readback).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
        return texel;
    }

    [[nodiscard]] bool nearTexel(const std::array<std::uint32_t, 4>& actual,
                                 const std::array<std::uint32_t, 4>& expected,
                                 std::uint32_t slack) noexcept
    {
        for (std::size_t channel = 0; channel < actual.size(); ++channel)
        {
            if (actual[channel] + slack < expected[channel] || actual[channel] > expected[channel] + slack)
            {
                return false;
            }
        }
        return true;
    }
#endif

} // namespace

TEST_CASE("Vulkan texture copies move color texels and reject overlapping destinations",
          "[size:medium][rhi][vulkan][copy][texture]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto source = device.createTexture({
        .width = copyWidth,
        .height = copyHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Vulkan image-copy source",
    });
    auto destination = device.createTexture({
        .width = copyWidth,
        .height = copyHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Vulkan image-copy destination",
    });
    REQUIRE(source);
    REQUIRE(destination);

    const rachis::BufferTextureCopyRegion region{.extent = {.width = copyWidth, .height = copyHeight}};
    auto layout = device.queryTextureTransferLayout(*source, region);
    REQUIRE(layout);
    auto upload = device.createBuffer({
        .size = layout->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto readback = device.createBuffer({
        .size = layout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(upload);
    REQUIRE(readback);

    std::vector<std::byte> pixels(static_cast<std::size_t>(layout->requiredBufferSize));
    for (std::uint32_t y = 0; y < copyHeight; ++y)
    {
        for (std::uint32_t x = 0; x < copyWidth; ++x)
        {
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                pixels[static_cast<std::size_t>(y) * layout->bytesPerRow + static_cast<std::size_t>(x) * bytesPerPixel +
                       channel] = texelByte(x, y, channel);
            }
        }
    }
    REQUIRE(device.writeBuffer(*upload, 0, pixels).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *source,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *source, std::array{region}).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *source,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *destination,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());

    const std::span<const rachis::TextureCopyRegion> empty;
    CHECK_FALSE(device.copyTexture(*command, *source, *destination, empty).has_value());
    CHECK_FALSE(device
                    .copyTexture(*command,
                                 {
                                     .source = *source,
                                     .destination = *source,
                                     .extent = {.width = copyWidth, .height = copyHeight},
                                 })
                    .has_value());
    const std::array overlapping{
        rachis::TextureCopyRegion{.extent = {.width = 2, .height = 2}},
        rachis::TextureCopyRegion{.destinationOrigin = {.x = 1, .y = 1}, .extent = {.width = 2, .height = 2}},
    };
    CHECK_FALSE(device.copyTexture(*command, *source, *destination, overlapping).has_value());
    REQUIRE(device
                .copyTexture(*command,
                             {
                                 .source = *source,
                                 .destination = *destination,
                                 .extent = {.width = copyWidth, .height = copyHeight},
                             })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *destination,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *destination, *readback, std::array{region}).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    CHECK(std::memcmp(mapped->data, pixels.data(), pixels.size()) == 0);
    REQUIRE(device.unmapBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan color resolve averages samples and rejects min/max", "[size:medium][rhi][vulkan][resolve][color]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto format = device.queryFormatProperties(rachis::Format::rgba8Unorm);
    REQUIRE(rachis::supportsResolveMode(format.colorResolveModes, rachis::ResolveMode::average));
    REQUIRE_FALSE(rachis::supportsResolveMode(format.colorResolveModes, rachis::ResolveMode::minimum));

    auto source = device.createTexture({
        .width = 2,
        .height = 2,
        .sampleCount = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
        .label = "Vulkan MSAA color source",
    });
    auto destination = device.createTexture({
        .width = 2,
        .height = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource |
                 rachis::TextureUsage::copyDestination,
        .label = "Vulkan resolved color",
    });
    REQUIRE(source);
    REQUIRE(destination);
    auto view = device.createTextureView({
        .texture = *source,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(view);

    const rachis::BufferTextureCopyRegion region{.extent = {.width = 2, .height = 2}};
    auto layout = device.queryTextureTransferLayout(*destination, region);
    REQUIRE(layout);
    auto readback = device.createBuffer({
        .size = layout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(readback);

    auto copyCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(copyCommand);
    CHECK_FALSE(device.resolveTexture(*copyCommand, {.source = *source, .destination = *destination}).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *source,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array attachments{
        rachis::ColorAttachment{
            .view = *view,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.25F, 0.5F, 0.75F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
    CHECK_FALSE(device.resolveTexture(*command, {.source = *source, .destination = *destination}).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *source,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = rachis::PipelineStage::resolve,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = rachis::Access::resolveSource,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::resolveSource,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *destination,
                                    .stagesAfter = rachis::PipelineStage::resolve,
                                    .accessAfter = rachis::Access::resolveDestination,
                                    .layoutAfter = rachis::TextureLayout::resolveDestination,
                                })
                .has_value());
    CHECK_FALSE(device
                    .resolveTexture(*command,
                                    {
                                        .source = *source,
                                        .destination = *destination,
                                        .mode = rachis::ResolveMode::minimum,
                                    })
                    .has_value());
    REQUIRE(device.resolveTexture(*command, {.source = *source, .destination = *destination}).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *destination,
                                    .stagesBefore = rachis::PipelineStage::resolve,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::resolveDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::resolveDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *destination, *readback, std::array{region}).has_value());

    REQUIRE(device.destroyTextureView(*view).has_value());
    REQUIRE(device.destroyTexture(*source).has_value());
    REQUIRE(device.destroyTexture(*destination).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    CHECK(nearByte(mapped->data[0], 64));
    CHECK(nearByte(mapped->data[1], 128));
    CHECK(nearByte(mapped->data[2], 191));
    CHECK(mapped->data[3] == std::byte{255});
    REQUIRE(device.unmapBuffer(*readback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan integer color formats report no resolve and resolveTexture rejects them",
          "[size:medium][rhi][vulkan][resolve][color]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    std::uint32_t attemptedResolves = 0;
    for (const auto format :
         {rachis::Format::rgba8Uint, rachis::Format::r32Uint, rachis::Format::rgba16Sint, rachis::Format::rg32Uint})
    {
        INFO("Format " << static_cast<int>(format));
        const auto properties = device.queryFormatProperties(format);
        CHECK(properties.colorResolveModes == rachis::ResolveModeMask::none);
        CHECK_FALSE(rachis::hasAny(properties.features, rachis::FormatFeature::multisampleResolve));
        if (!rachis::hasAny(properties.features, rachis::FormatFeature::colorAttachment) ||
            !rachis::supportsSampleCount(properties.sampleCounts, 4))
        {
            continue;
        }

        auto source = device.createTexture({
            .width = 2,
            .height = 2,
            .sampleCount = 4,
            .format = format,
            .usage = rachis::TextureUsage::renderTarget,
            .label = "Vulkan integer MSAA source",
        });
        auto destination = device.createTexture({
            .width = 2,
            .height = 2,
            .format = format,
            .usage = rachis::TextureUsage::renderTarget,
            .label = "Vulkan integer resolve destination",
        });
        REQUIRE(source);
        REQUIRE(destination);

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *source,
                                        .stagesAfter = rachis::PipelineStage::resolve,
                                        .accessAfter = rachis::Access::resolveSource,
                                        .layoutAfter = rachis::TextureLayout::resolveSource,
                                    })
                    .has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *destination,
                                        .stagesAfter = rachis::PipelineStage::resolve,
                                        .accessAfter = rachis::Access::resolveDestination,
                                        .layoutAfter = rachis::TextureLayout::resolveDestination,
                                    })
                    .has_value());
        auto resolved = device.resolveTexture(*command, {.source = *source, .destination = *destination});
        REQUIRE_FALSE(resolved);
        CHECK(resolved.error().code == rachis::ErrorCode::unsupported);
        ++attemptedResolves;

        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted).has_value());
        REQUIRE(device.destroyTexture(*source).has_value());
        REQUIRE(device.destroyTexture(*destination).has_value());
    }
    CHECK(attemptedResolves != 0);
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan depth resolve uses min/max without touching an unused packed stencil aspect",
          "[size:medium][rhi][vulkan][resolve][depth]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto depth = device.queryFormatProperties(rachis::Format::d32Float);
    if (!rachis::supportsResolveMode(depth.depthResolveModes, rachis::ResolveMode::minimum))
    {
        return;
    }

    auto source = device.createTexture({
        .width = 2,
        .height = 2,
        .sampleCount = 4,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::depthStencil | rachis::TextureUsage::copySource,
        .label = "Vulkan MSAA depth source",
    });
    auto destination = device.createTexture({
        .width = 2,
        .height = 2,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::depthStencil | rachis::TextureUsage::copySource |
                 rachis::TextureUsage::copyDestination,
        .label = "Vulkan resolved depth",
    });
    REQUIRE(source);
    REQUIRE(destination);
    auto view = device.createTextureView({
        .texture = *source,
        .usage = rachis::TextureViewUsage::depthStencil,
        .aspect = rachis::TextureAspect::depth,
    });
    REQUIRE(view);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *source,
                                    .stagesAfter = rachis::PipelineStage::depthStencil,
                                    .accessAfter = rachis::Access::depthStencilWrite,
                                    .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());
    REQUIRE(device
                .beginRendering(*command,
                                {
                                    .depthStencilAttachment =
                                        rachis::DepthStencilAttachment{
                                            .view = *view,
                                            .depthLoadOperation = rachis::LoadOperation::clear,
                                            .depthClearValue = 0.25F,
                                        },
                                })
                .has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *source,
                                    .stagesBefore = rachis::PipelineStage::depthStencil,
                                    .stagesAfter = rachis::PipelineStage::resolve,
                                    .accessBefore = rachis::Access::depthStencilWrite,
                                    .accessAfter = rachis::Access::resolveSource,
                                    .layoutBefore = rachis::TextureLayout::depthStencilWrite,
                                    .layoutAfter = rachis::TextureLayout::resolveSource,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *destination,
                                    .stagesAfter = rachis::PipelineStage::resolve,
                                    .accessAfter = rachis::Access::resolveDestination,
                                    .layoutAfter = rachis::TextureLayout::resolveDestination,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());
    CHECK_FALSE(device
                    .resolveTexture(*command,
                                    {
                                        .source = *source,
                                        .sourceSubresource = {.aspect = rachis::TextureAspect::depth},
                                        .destination = *destination,
                                        .destinationSubresource = {.aspect = rachis::TextureAspect::depth},
                                        .mode = rachis::ResolveMode::average,
                                    })
                    .has_value());
    REQUIRE(device
                .resolveTexture(*command,
                                {
                                    .source = *source,
                                    .sourceSubresource = {.aspect = rachis::TextureAspect::depth},
                                    .destination = *destination,
                                    .destinationSubresource = {.aspect = rachis::TextureAspect::depth},
                                    .mode = rachis::ResolveMode::minimum,
                                })
                .has_value());

    REQUIRE(device.destroyTextureView(*view).has_value());
    REQUIRE(device.destroyTexture(*source).has_value());
    REQUIRE(device.destroyTexture(*destination).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan mipmap generation filters a 2D color texture and rejects invalid usage",
          "[size:medium][rhi][vulkan][mipmap]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).mipmapGeneration);

    auto texture = device.createTexture({
        .width = copyWidth,
        .height = copyHeight,
        .mipLevels = mipLevels,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "Vulkan generated mips",
    });
    auto sampledOnly = device.createTexture({
        .width = copyWidth,
        .height = copyHeight,
        .mipLevels = mipLevels,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
        .label = "Vulkan sampled-only mips",
    });
    REQUIRE(texture);
    REQUIRE(sampledOnly);

    const rachis::BufferTextureCopyRegion seed{.extent = {.width = copyWidth, .height = copyHeight}};
    auto seedLayout = device.queryTextureTransferLayout(*texture, seed);
    REQUIRE(seedLayout);
    auto upload = device.createBuffer({
        .size = seedLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    const rachis::BufferTextureCopyRegion mip2{
        .textureSubresource = {.mipLevel = 2},
        .extent = {.width = 1, .height = 1},
    };
    auto mipLayout = device.queryTextureTransferLayout(*texture, mip2);
    REQUIRE(mipLayout);
    auto readback = device.createBuffer({
        .size = mipLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(upload);
    REQUIRE(readback);

    std::vector<std::byte> pixels(static_cast<std::size_t>(seedLayout->requiredBufferSize));
    for (std::uint32_t y = 0; y < copyHeight; ++y)
    {
        for (std::uint32_t x = 0; x < copyWidth; ++x)
        {
            const auto texel = baseTexel(x, y);
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                pixels[static_cast<std::size_t>(y) * seedLayout->bytesPerRow +
                       static_cast<std::size_t>(x) * bytesPerPixel + channel] = static_cast<std::byte>(texel[channel]);
            }
        }
    }
    REQUIRE(device.writeBuffer(*upload, 0, pixels).has_value());

    auto copyCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(copyCommand);
    CHECK_FALSE(device
                    .generateMipmaps(*copyCommand,
                                     {
                                         .texture = *texture,
                                         .baseLayout = rachis::TextureLayout::copyDestination,
                                     })
                    .has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *texture, std::array{seed}).has_value());
    CHECK_FALSE(device
                    .generateMipmaps(*command,
                                     {
                                         .texture = *sampledOnly,
                                         .baseLayout = rachis::TextureLayout::shaderResource,
                                     })
                    .has_value());
    REQUIRE(device
                .generateMipmaps(*command,
                                 {
                                     .texture = *texture,
                                     .baseLayout = rachis::TextureLayout::copyDestination,
                                     .finalLayout = rachis::TextureLayout::copySource,
                                 })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *texture, *readback, std::array{mip2}).has_value());

    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyTexture(*sampledOnly).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    CHECK(nearByte(mapped->data[0], 60));
    CHECK(nearByte(mapped->data[1], 60));
    CHECK(nearByte(mapped->data[2], 60));
    CHECK(mapped->data[3] == std::byte{255});
    REQUIRE(device.unmapBuffer(*readback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan mipmap generation filters an odd 3x3 extent to a 1x1 mip", "[size:medium][rhi][vulkan][mipmap][odd]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    constexpr std::uint32_t oddSize = 3;
    auto texture = device.createTexture({
        .width = oddSize,
        .height = oddSize,
        .mipLevels = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "Vulkan odd-extent mips",
    });
    REQUIRE(texture);

    const rachis::BufferTextureCopyRegion seed{.extent = {.width = oddSize, .height = oddSize}};
    auto seedLayout = device.queryTextureTransferLayout(*texture, seed);
    REQUIRE(seedLayout);
    auto upload = device.createBuffer({
        .size = seedLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    const rachis::BufferTextureCopyRegion mip1{
        .textureSubresource = {.mipLevel = 1},
        .extent = {.width = 1, .height = 1},
    };
    auto mipLayout = device.queryTextureTransferLayout(*texture, mip1);
    REQUIRE(mipLayout);
    auto readback = device.createBuffer({
        .size = mipLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(upload);
    REQUIRE(readback);

    std::vector<std::byte> pixels(static_cast<std::size_t>(seedLayout->requiredBufferSize));
    for (std::uint32_t y = 0; y < oddSize; ++y)
    {
        for (std::uint32_t x = 0; x < oddSize; ++x)
        {
            const auto texel = baseTexel(x, y);
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                pixels[static_cast<std::size_t>(y) * seedLayout->bytesPerRow +
                       static_cast<std::size_t>(x) * bytesPerPixel + channel] = static_cast<std::byte>(texel[channel]);
            }
        }
    }
    REQUIRE(device.writeBuffer(*upload, 0, pixels).has_value());

    const auto flags = GENERATE(rachis::CommandPoolFlag::none, rachis::CommandPoolFlag::unretainedResources);
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics, flags);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *texture, std::array{seed}).has_value());
    REQUIRE(device
                .generateMipmaps(*command,
                                 {
                                     .texture = *texture,
                                     .baseLayout = rachis::TextureLayout::copyDestination,
                                     .finalLayout = rachis::TextureLayout::copySource,
                                 })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *texture, *readback, std::array{mip1}).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    CHECK(nearByte(mapped->data[0], 40));
    CHECK(nearByte(mapped->data[1], 40));
    CHECK(nearByte(mapped->data[2], 40));
    CHECK(mapped->data[3] == std::byte{255});
    REQUIRE(device.unmapBuffer(*readback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan mipmap generation synchronizes shader-resource boundaries with graphics shader stages",
          "[size:medium][rhi][vulkan][mipmap][sync]")
{
    auto created = [] {
        const ScopedEnvironmentVariable shaderAccesses{"VK_KHRONOS_VALIDATION_SYNCVAL_SHADER_ACCESSES_HEURISTIC",
                                                       "true"};
        return rachis::createDevice(vulkanDesc());
    }();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).mipmapGeneration);

    auto texture = device.createTexture({
        .width = copyWidth,
        .height = copyHeight,
        .mipLevels = mipLevels,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "Vulkan shader-resource mips",
    });
    REQUIRE(texture);

    const rachis::BufferTextureCopyRegion seed{.extent = {.width = copyWidth, .height = copyHeight}};
    auto seedLayout = device.queryTextureTransferLayout(*texture, seed);
    REQUIRE(seedLayout);
    auto upload = device.createBuffer({
        .size = seedLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    const rachis::BufferTextureCopyRegion mip2{
        .textureSubresource = {.mipLevel = 2},
        .extent = {.width = 1, .height = 1},
    };
    auto mipLayout = device.queryTextureTransferLayout(*texture, mip2);
    REQUIRE(mipLayout);
    auto readback = device.createBuffer({
        .size = mipLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(upload);
    REQUIRE(readback);

    std::vector<std::byte> pixels(static_cast<std::size_t>(seedLayout->requiredBufferSize));
    for (std::uint32_t y = 0; y < copyHeight; ++y)
    {
        for (std::uint32_t x = 0; x < copyWidth; ++x)
        {
            const auto texel = baseTexel(x, y);
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                pixels[static_cast<std::size_t>(y) * seedLayout->bytesPerRow +
                       static_cast<std::size_t>(x) * bytesPerPixel + channel] = static_cast<std::byte>(texel[channel]);
            }
        }
    }
    REQUIRE(device.writeBuffer(*upload, 0, pixels).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *texture, std::array{seed}).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::pixelShader,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::shaderResource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::shaderResource,
                                    .subresources = {.firstMipLevel = 0, .mipLevelCount = 1},
                                })
                .has_value());
    REQUIRE(device
                .generateMipmaps(*command,
                                 {
                                     .texture = *texture,
                                     .baseLayout = rachis::TextureLayout::shaderResource,
                                 })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::pixelShader,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::shaderResource,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::shaderResource,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *texture, *readback, std::array{mip2}).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    CHECK(nearByte(mapped->data[0], 60));
    CHECK(nearByte(mapped->data[1], 60));
    CHECK(nearByte(mapped->data[2], 60));
    CHECK(mapped->data[3] == std::byte{255});
    REQUIRE(device.unmapBuffer(*readback).has_value());

    auto compute = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(compute);
    REQUIRE(device
                .textureBarrier(*compute,
                                {
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::computeShader,
                                    .accessBefore = rachis::Access::copySource,
                                    .accessAfter = rachis::Access::shaderResource,
                                    .layoutBefore = rachis::TextureLayout::copySource,
                                    .layoutAfter = rachis::TextureLayout::shaderResource,
                                })
                .has_value());
    REQUIRE(device
                .generateMipmaps(*compute,
                                 {
                                     .texture = *texture,
                                     .baseLayout = rachis::TextureLayout::shaderResource,
                                 })
                .has_value());
    REQUIRE(device
                .textureBarrier(*compute,
                                {
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::computeShader,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::shaderResource,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::shaderResource,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*compute, *texture, *readback, std::array{mip2}).has_value());
    auto computeSubmitted = device.submit(std::move(*compute), std::array{*submitted});
    REQUIRE(computeSubmitted);
    REQUIRE(device.wait(*computeSubmitted).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    auto remapped = device.mapBuffer(*readback);
    REQUIRE(remapped);
    CHECK(nearByte(remapped->data[0], 60));
    CHECK(nearByte(remapped->data[1], 60));
    CHECK(nearByte(remapped->data[2], 60));
    CHECK(remapped->data[3] == std::byte{255});
    REQUIRE(device.unmapBuffer(*readback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan mipmap generation finishes a deep chain before a copy that waits only on the pixel shader",
          "[size:medium][rhi][vulkan][mipmap][sync][regression]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).mipmapGeneration);

    constexpr std::uint32_t chainSize = 2048;
    constexpr std::uint32_t chainMipLevels = 12;
    constexpr std::uint32_t chainIterations = 6;
    const rachis::BufferTextureCopyRegion seed{.extent = {.width = chainSize, .height = chainSize}};
    const rachis::BufferTextureCopyRegion lastMip{
        .textureSubresource = {.mipLevel = chainMipLevels - 1},
        .extent = {.width = 1, .height = 1},
    };
    for (std::uint32_t iteration = 0; iteration < chainIterations; ++iteration)
    {
        CAPTURE(iteration);
        auto texture = device.createTexture({
            .width = chainSize,
            .height = chainSize,
            .mipLevels = chainMipLevels,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                     rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
            .label = "Vulkan deep shader-resource mips",
        });
        REQUIRE(texture);
        auto seedLayout = device.queryTextureTransferLayout(*texture, seed);
        auto lastLayout = device.queryTextureTransferLayout(*texture, lastMip);
        REQUIRE(seedLayout);
        REQUIRE(lastLayout);
        auto upload = device.createBuffer({
            .size = seedLayout->requiredBufferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
        });
        auto readback = device.createBuffer({
            .size = lastLayout->requiredBufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(upload);
        REQUIRE(readback);

        const std::array<std::uint8_t, 4> color{
            static_cast<std::uint8_t>(30U + iteration * 30U),
            static_cast<std::uint8_t>(200U - iteration * 20U),
            90,
            255,
        };
        std::vector<std::byte> row(seedLayout->bytesPerRow);
        for (std::uint32_t x = 0; x < chainSize; ++x)
        {
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                row[static_cast<std::size_t>(x) * bytesPerPixel + channel] = static_cast<std::byte>(color[channel]);
            }
        }
        std::vector<std::byte> pixels(static_cast<std::size_t>(seedLayout->requiredBufferSize));
        for (std::uint32_t y = 0; y < chainSize; ++y)
        {
            const auto offset = static_cast<std::size_t>(y) * seedLayout->bytesPerRow;
            const auto count = (std::min)(row.size(), pixels.size() - offset);
            std::copy_n(row.begin(), count, pixels.begin() + static_cast<std::ptrdiff_t>(offset));
        }
        REQUIRE(device.writeBuffer(*upload, 0, pixels).has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *texture,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                    })
                    .has_value());
        REQUIRE(device.copyBufferToTexture(*command, *upload, *texture, std::array{seed}).has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *texture,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::pixelShader,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::shaderResource,
                                        .layoutBefore = rachis::TextureLayout::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::shaderResource,
                                        .subresources = {.firstMipLevel = 0, .mipLevelCount = 1},
                                    })
                    .has_value());
        REQUIRE(device
                    .generateMipmaps(*command,
                                     {
                                         .texture = *texture,
                                         .baseLayout = rachis::TextureLayout::shaderResource,
                                     })
                    .has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *texture,
                                        .stagesBefore = rachis::PipelineStage::pixelShader,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::shaderResource,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::shaderResource,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        REQUIRE(device.copyTextureToBuffer(*command, *texture, *readback, std::array{lastMip}).has_value());

        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted).has_value());
        auto mapped = device.mapBuffer(*readback);
        REQUIRE(mapped);
        CHECK(nearByte(mapped->data[0], color[0]));
        CHECK(nearByte(mapped->data[1], color[1]));
        CHECK(nearByte(mapped->data[2], color[2]));
        CHECK(mapped->data[3] == std::byte{255});
        REQUIRE(device.unmapBuffer(*readback).has_value());
        REQUIRE(device.destroyTexture(*texture).has_value());
        REQUIRE(device.destroyBuffer(*upload).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
    }
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

#ifdef RACHIS_SPIRV_DIR
TEST_CASE("Vulkan mipmap generation waits for the render pass that wrote its base level",
          "[size:medium][rhi][vulkan][mipmap][regression]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto seed = createRenderedMipPipeline(device, "MipmapAfterRendering.slow.ps.spv", "slowPixel", {});

    for (std::uint32_t iteration = 0; iteration < renderedIterations; ++iteration)
    {
        CAPTURE(iteration);
        auto texture = device.createTexture(renderedMipDesc());
        REQUIRE(texture);
        const auto baseView = createRenderedMipView(device, *texture, 0);
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        recordRenderedBase(device, *command, *texture, baseView, seed);
        REQUIRE(device
                    .generateMipmaps(*command,
                                     {
                                         .texture = *texture,
                                         .baseLayout = rachis::TextureLayout::renderTarget,
                                         .finalLayout = rachis::TextureLayout::copySource,
                                     })
                    .has_value());
        const auto texel = readRenderedLastMip(device, std::move(*command), *texture);
        CAPTURE(texel[0], texel[1], texel[2], texel[3]);
        CHECK(nearTexel(texel, {255, 128, 64, 255}, 1));
        REQUIRE(device.destroyTextureView(baseView).has_value());
        REQUIRE(device.destroyTexture(*texture).has_value());
    }
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan mipmap generation finishes before a later render pass blends into its last mip",
          "[size:medium][rhi][vulkan][mipmap][regression]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto seed = createRenderedMipPipeline(device, "MipmapAfterRendering.slow.ps.spv", "slowPixel", {});
    constexpr rachis::BlendComponent additive{
        .sourceFactor = rachis::BlendFactor::one,
        .destinationFactor = rachis::BlendFactor::one,
    };
    const auto blend = createRenderedMipPipeline(device,
                                                 "MipmapAfterRendering.blend.ps.spv",
                                                 "blendPixel",
                                                 {.enabled = true, .color = additive, .alpha = additive});

    for (std::uint32_t iteration = 0; iteration < renderedIterations; ++iteration)
    {
        CAPTURE(iteration);
        auto texture = device.createTexture(renderedMipDesc());
        REQUIRE(texture);
        const auto baseView = createRenderedMipView(device, *texture, 0);
        const auto lastView = createRenderedMipView(device, *texture, renderedLastMip);
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        recordRenderedBase(device, *command, *texture, baseView, seed);
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *texture,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::computeShader,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::shaderResource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::shaderResource,
                                        .subresources = {.firstMipLevel = 0, .mipLevelCount = 1},
                                    })
                    .has_value());
        REQUIRE(device
                    .generateMipmaps(*command,
                                     {
                                         .texture = *texture,
                                         .baseLayout = rachis::TextureLayout::shaderResource,
                                         .finalLayout = rachis::TextureLayout::renderTarget,
                                     })
                    .has_value());
        recordRenderedMip(device, *command, lastView, blend, 1, 1, rachis::LoadOperation::load);
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *texture,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                        .subresources = {.firstMipLevel = renderedLastMip, .mipLevelCount = 1},
                                    })
                    .has_value());
        const auto texel = readRenderedLastMip(device, std::move(*command), *texture);
        CAPTURE(texel[0], texel[1], texel[2], texel[3]);
        CHECK(nearTexel(texel, {255, 192, 192, 255}, 2));
        REQUIRE(device.destroyTextureView(lastView).has_value());
        REQUIRE(device.destroyTextureView(baseView).has_value());
        REQUIRE(device.destroyTexture(*texture).has_value());
    }
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
#endif

TEST_CASE("Vulkan mipmap generation discards destination mips that earlier work in the batch wrote",
          "[size:medium][rhi][vulkan][mipmap]")
{
    const bool splitBatch = GENERATE(false, true);
    CAPTURE(splitBatch);
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).mipmapGeneration);

    auto texture = device.createTexture({
        .width = copyWidth,
        .height = copyHeight,
        .mipLevels = mipLevels,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "Vulkan discarded mips",
    });
    REQUIRE(texture);

    std::array<rachis::BufferTextureCopyRegion, mipLevels> regions{};
    std::array<rachis::TextureTransferLayout, mipLevels> layouts{};
    std::uint64_t bufferSize = 0;
    for (std::uint32_t mipLevel = 0; mipLevel < mipLevels; ++mipLevel)
    {
        regions[mipLevel] = rachis::BufferTextureCopyRegion{
            .textureSubresource = {.mipLevel = mipLevel},
            .extent = {.width = (std::max)(copyWidth >> mipLevel, 1U),
                       .height = (std::max)(copyHeight >> mipLevel, 1U)},
        };
        auto layout = device.queryTextureTransferLayout(*texture, regions[mipLevel]);
        REQUIRE(layout);
        const std::uint64_t alignment = (std::max)(layout->bufferOffsetAlignment, 1U);
        bufferSize = (bufferSize + alignment - 1U) / alignment * alignment;
        regions[mipLevel].bufferOffset = bufferSize;
        layouts[mipLevel] = *layout;
        bufferSize += layout->requiredBufferSize;
    }
    auto upload = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto readback = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(upload);
    REQUIRE(readback);

    constexpr std::array<std::uint8_t, 4> sentinel{0x11, 0x22, 0x33, 0x44};
    std::vector<std::byte> pixels(static_cast<std::size_t>(bufferSize));
    for (std::uint32_t y = 0; y < copyHeight; ++y)
    {
        for (std::uint32_t x = 0; x < copyWidth; ++x)
        {
            const auto texel = baseTexel(x, y);
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                pixels[static_cast<std::size_t>(y) * layouts[0].bytesPerRow +
                       static_cast<std::size_t>(x) * bytesPerPixel + channel] = static_cast<std::byte>(texel[channel]);
            }
        }
    }
    for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
    {
        pixels[static_cast<std::size_t>(regions[mipLevels - 1U].bufferOffset) + channel] =
            static_cast<std::byte>(sentinel[channel]);
    }
    REQUIRE(device.writeBuffer(*upload, 0, pixels).has_value());

    std::array<rachis::CommandBuffer, 2> batch{};
    auto first = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(first);
    batch[0] = std::move(*first);
    REQUIRE(device
                .textureBarrier(batch[0],
                                {
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(batch[0], *upload, *texture, std::array{regions[0], regions[mipLevels - 1U]})
                .has_value());
    REQUIRE(device
                .generateMipmaps(batch[0],
                                 {
                                     .texture = *texture,
                                     .baseLayout = rachis::TextureLayout::copyDestination,
                                 })
                .has_value());
    if (splitBatch)
    {
        auto second = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(second);
        batch[1] = std::move(*second);
    }
    auto& tail = splitBatch ? batch[1] : batch[0];
    REQUIRE(device
                .generateMipmaps(tail,
                                 {
                                     .texture = *texture,
                                     .baseLayout = rachis::TextureLayout::shaderResource,
                                     .finalLayout = rachis::TextureLayout::copySource,
                                 })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(tail, *texture, *readback, regions).has_value());

    auto submitted = device.submit(std::span<rachis::CommandBuffer>(batch.data(), splitBatch ? 2U : 1U));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    for (std::uint32_t y = 0; y < copyHeight; ++y)
    {
        for (std::uint32_t x = 0; x < copyWidth; ++x)
        {
            const auto texel = baseTexel(x, y);
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                const auto offset = static_cast<std::size_t>(y) * layouts[0].bytesPerRow +
                                    static_cast<std::size_t>(x) * bytesPerPixel + channel;
                CAPTURE(x, y, channel, texel[channel], std::to_integer<std::uint8_t>(mapped->data[offset]));
                CHECK(mapped->data[offset] == static_cast<std::byte>(texel[channel]));
            }
        }
    }
    for (std::uint32_t y = 0; y < 2; ++y)
    {
        for (std::uint32_t x = 0; x < 2; ++x)
        {
            const auto texel = mip1Texel(x, y);
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                const auto offset = static_cast<std::size_t>(regions[1].bufferOffset) +
                                    static_cast<std::size_t>(y) * layouts[1].bytesPerRow +
                                    static_cast<std::size_t>(x) * bytesPerPixel + channel;
                CAPTURE(x, y, channel, texel[channel], std::to_integer<std::uint8_t>(mapped->data[offset]));
                CHECK(nearByte(mapped->data[offset], texel[channel]));
            }
        }
    }
    constexpr std::array<std::uint8_t, 4> mip2Expected{60, 60, 60, 255};
    for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
    {
        const auto offset = static_cast<std::size_t>(regions[mipLevels - 1U].bufferOffset) + channel;
        CAPTURE(channel, mip2Expected[channel], std::to_integer<std::uint8_t>(mapped->data[offset]));
        CHECK(nearByte(mapped->data[offset], mip2Expected[channel]));
    }
    REQUIRE(device.unmapBuffer(*readback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan packed depth resolve leaves the unused stencil plane untouched",
          "[size:medium][rhi][vulkan][resolve][packed]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto packed = device.queryFormatProperties(rachis::Format::d24UnormS8Uint);
    if (!rachis::supportsResolveMode(packed.depthResolveModes, rachis::ResolveMode::minimum) ||
        !rachis::supportsResolveMode(packed.stencilResolveModes, rachis::ResolveMode::maximum) ||
        !rachis::hasAny(packed.features, rachis::FormatFeature::multisampleResolve))
    {
        return;
    }

    auto source = device.createTexture({
        .width = 2,
        .height = 2,
        .sampleCount = 4,
        .format = rachis::Format::d24UnormS8Uint,
        .usage = rachis::TextureUsage::depthStencil | rachis::TextureUsage::copySource,
        .label = "Vulkan packed MSAA source",
    });
    auto destination = device.createTexture({
        .width = 2,
        .height = 2,
        .format = rachis::Format::d24UnormS8Uint,
        .usage = rachis::TextureUsage::depthStencil | rachis::TextureUsage::copySource |
                 rachis::TextureUsage::copyDestination,
        .label = "Vulkan packed resolve destination",
    });
    REQUIRE(source);
    REQUIRE(destination);
    auto sourceView = device.createTextureView({
        .texture = *source,
        .usage = rachis::TextureViewUsage::depthStencil,
        .aspect = rachis::TextureAspect::depth,
    });
    auto destinationView = device.createTextureView({
        .texture = *destination,
        .usage = rachis::TextureViewUsage::depthStencil,
        .aspect = rachis::TextureAspect::depth,
    });
    REQUIRE(sourceView);
    REQUIRE(destinationView);

    const rachis::BufferTextureCopyRegion stencilRegion{
        .textureSubresource = {.aspect = rachis::TextureAspect::stencil},
        .extent = {.width = 2, .height = 2},
    };
    auto stencilLayout = device.queryTextureTransferLayout(*destination, stencilRegion);
    REQUIRE(stencilLayout);
    auto stencilReadback = device.createBuffer({
        .size = stencilLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(stencilReadback);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    for (const auto aspect : {rachis::TextureAspect::depth, rachis::TextureAspect::stencil})
    {
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *source,
                                        .stagesAfter = rachis::PipelineStage::depthStencil,
                                        .accessAfter = rachis::Access::depthStencilWrite,
                                        .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                        .subresources = {.aspect = aspect},
                                    })
                    .has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *destination,
                                        .stagesAfter = rachis::PipelineStage::depthStencil,
                                        .accessAfter = rachis::Access::depthStencilWrite,
                                        .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                        .subresources = {.aspect = aspect},
                                    })
                    .has_value());
    }
    REQUIRE(device
                .beginRendering(*command,
                                {
                                    .depthStencilAttachment =
                                        rachis::DepthStencilAttachment{
                                            .view = *sourceView,
                                            .depthLoadOperation = rachis::LoadOperation::clear,
                                            .depthClearValue = 0.25F,
                                            .stencilLoadOperation = rachis::LoadOperation::clear,
                                            .stencilClearValue = 37,
                                        },
                                })
                .has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device
                .beginRendering(*command,
                                {
                                    .depthStencilAttachment =
                                        rachis::DepthStencilAttachment{
                                            .view = *destinationView,
                                            .depthLoadOperation = rachis::LoadOperation::clear,
                                            .depthClearValue = 1.0F,
                                            .stencilLoadOperation = rachis::LoadOperation::clear,
                                            .stencilClearValue = 9,
                                        },
                                })
                .has_value());
    REQUIRE(device.endRendering(*command).has_value());

    for (const auto aspect : {rachis::TextureAspect::depth, rachis::TextureAspect::stencil})
    {
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *source,
                                        .stagesBefore = rachis::PipelineStage::depthStencil,
                                        .stagesAfter = rachis::PipelineStage::resolve,
                                        .accessBefore = rachis::Access::depthStencilWrite,
                                        .accessAfter = rachis::Access::resolveSource,
                                        .layoutBefore = rachis::TextureLayout::depthStencilWrite,
                                        .layoutAfter = rachis::TextureLayout::resolveSource,
                                        .subresources = {.aspect = aspect},
                                    })
                    .has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *destination,
                                        .stagesBefore = rachis::PipelineStage::depthStencil,
                                        .stagesAfter = rachis::PipelineStage::resolve,
                                        .accessBefore = rachis::Access::depthStencilWrite,
                                        .accessAfter = rachis::Access::resolveDestination,
                                        .layoutBefore = rachis::TextureLayout::depthStencilWrite,
                                        .layoutAfter = rachis::TextureLayout::resolveDestination,
                                        .subresources = {.aspect = aspect},
                                    })
                    .has_value());
    }

    CHECK_FALSE(device
                    .resolveTexture(*command,
                                    {
                                        .source = *source,
                                        .sourceSubresource = {.aspect = rachis::TextureAspect::depth},
                                        .destination = *destination,
                                        .destinationSubresource = {.aspect = rachis::TextureAspect::stencil},
                                        .mode = rachis::ResolveMode::minimum,
                                    })
                    .has_value());
    REQUIRE(device
                .resolveTexture(*command,
                                {
                                    .source = *source,
                                    .sourceSubresource = {.aspect = rachis::TextureAspect::depth},
                                    .destination = *destination,
                                    .destinationSubresource = {.aspect = rachis::TextureAspect::depth},
                                    .mode = rachis::ResolveMode::minimum,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *destination,
                                    .stagesBefore = rachis::PipelineStage::resolve,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::resolveDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::resolveDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                    .subresources = {.aspect = rachis::TextureAspect::stencil},
                                })
                .has_value());
    REQUIRE(
        device.copyTextureToBuffer(*command, *destination, *stencilReadback, std::array{stencilRegion}).has_value());

    REQUIRE(device.destroyTextureView(*sourceView).has_value());
    REQUIRE(device.destroyTextureView(*destinationView).has_value());
    REQUIRE(device.destroyTexture(*source).has_value());
    REQUIRE(device.destroyTexture(*destination).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto mapped = device.mapBuffer(*stencilReadback);
    REQUIRE(mapped);
    CHECK(mapped->data[0] == std::byte{9});
    CHECK(mapped->data[1] == std::byte{9});
    REQUIRE(device.unmapBuffer(*stencilReadback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan mipmap generation synchronizes lazy pipeline init and scales internal descriptor pools across "
          "concurrent recordings",
          "[size:medium][rhi][vulkan][mipmap][concurrency]")
{
    constexpr std::size_t workerCount = 4;
    constexpr std::size_t texturesPerWorker = 5;
    constexpr std::size_t totalTextures = workerCount * texturesPerWorker;
    constexpr std::uint32_t baseExtent = 16;
    constexpr std::uint32_t chainMipLevels = 5;

    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).mipmapGeneration);

    auto sharedCommandPool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = static_cast<std::uint32_t>(workerCount),
        .label = "Vulkan concurrent mipmap command pool",
        .flags = rachis::CommandPoolFlag::allowIndividualReset,
    });
    REQUIRE(sharedCommandPool);

    const rachis::BufferTextureCopyRegion seedRegion{.extent = {.width = baseExtent, .height = baseExtent}};
    const rachis::BufferTextureCopyRegion tailRegion{
        .textureSubresource = {.mipLevel = chainMipLevels - 1U},
        .extent = {.width = 1, .height = 1},
    };

    std::vector<rachis::TextureHandle> textures(totalTextures);
    std::vector<rachis::BufferHandle> uploads(totalTextures);
    std::vector<rachis::BufferHandle> readbacks(totalTextures);
    std::uint64_t seedBytesPerRow = 0;
    std::uint64_t seedBufferSize = 0;

    for (std::size_t index = 0; index < totalTextures; ++index)
    {
        auto texture = device.createTexture({
            .width = baseExtent,
            .height = baseExtent,
            .mipLevels = chainMipLevels,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                     rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
            .label = "Vulkan concurrent mipmap texture",
        });
        REQUIRE(texture);
        textures[index] = *texture;

        if (index == 0)
        {
            auto seedLayout = device.queryTextureTransferLayout(*texture, seedRegion);
            REQUIRE(seedLayout);
            seedBytesPerRow = seedLayout->bytesPerRow;
            seedBufferSize = seedLayout->requiredBufferSize;
        }
        auto tailLayout = device.queryTextureTransferLayout(*texture, tailRegion);
        REQUIRE(tailLayout);

        auto upload = device.createBuffer({
            .size = seedBufferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
        });
        auto readback = device.createBuffer({
            .size = tailLayout->requiredBufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(upload);
        REQUIRE(readback);
        uploads[index] = *upload;
        readbacks[index] = *readback;
    }

    auto writeSolidSeed = [&](std::size_t index, std::uint8_t red, std::uint8_t green, std::uint8_t blue) {
        std::vector<std::byte> pixels(static_cast<std::size_t>(seedBufferSize));
        for (std::uint32_t y = 0; y < baseExtent; ++y)
        {
            for (std::uint32_t x = 0; x < baseExtent; ++x)
            {
                const auto offset = static_cast<std::size_t>(y) * static_cast<std::size_t>(seedBytesPerRow) +
                                    static_cast<std::size_t>(x) * bytesPerPixel;
                pixels[offset + 0] = static_cast<std::byte>(red);
                pixels[offset + 1] = static_cast<std::byte>(green);
                pixels[offset + 2] = static_cast<std::byte>(blue);
                pixels[offset + 3] = std::byte{255};
            }
        }
        REQUIRE(device.writeBuffer(uploads[index], 0, pixels).has_value());
    };

    for (std::size_t index = 0; index < totalTextures; ++index)
    {
        writeSolidSeed(index,
                       static_cast<std::uint8_t>(20U + index * 9U),
                       static_cast<std::uint8_t>(40U + index * 7U),
                       static_cast<std::uint8_t>(60U + index * 5U));
    }

    auto recordRound = [&](bool firstRound, bool concurrentCollectGarbage) {
        std::array<rachis::CommandBuffer, workerCount> commands{};
        for (std::size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
        {
            auto acquired = device.acquireCommandBuffer(*sharedCommandPool);
            REQUIRE(acquired);
            commands[workerIndex] = std::move(*acquired);
        }

        std::array<bool, workerCount> workerOk{};
        workerOk.fill(true);
        std::atomic<bool> stopCollector{false};
        std::barrier startBarrier(static_cast<std::ptrdiff_t>(workerCount));

        {
            std::jthread collector;
            if (concurrentCollectGarbage)
            {
                collector = std::jthread([&device, &stopCollector] {
                    while (!stopCollector.load(std::memory_order_acquire))
                    {
                        device.collectGarbage();
                        std::this_thread::yield();
                    }
                });
            }

            std::vector<std::jthread> workers;
            workers.reserve(workerCount);
            for (std::size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
            {
                workers.emplace_back([&, workerIndex] {
                    startBarrier.arrive_and_wait();
                    auto& command = commands[workerIndex];
                    const std::size_t beginTexture = workerIndex * texturesPerWorker;
                    for (std::size_t local = 0; local < texturesPerWorker; ++local)
                    {
                        const std::size_t textureIndex = beginTexture + local;
                        const rachis::TextureBarrier toCopyDst{
                            .texture = textures[textureIndex],
                            .stagesBefore = firstRound ? rachis::PipelineStage::none : rachis::PipelineStage::copy,
                            .stagesAfter = rachis::PipelineStage::copy,
                            .accessBefore = firstRound ? rachis::Access::none : rachis::Access::copySource,
                            .accessAfter = rachis::Access::copyDestination,
                            .layoutBefore =
                                firstRound ? rachis::TextureLayout::undefined : rachis::TextureLayout::copySource,
                            .layoutAfter = rachis::TextureLayout::copyDestination,
                        };
                        if (!device.textureBarrier(command, toCopyDst) ||
                            !device.copyBufferToTexture(
                                command, uploads[textureIndex], textures[textureIndex], std::array{seedRegion}) ||
                            !device.generateMipmaps(command,
                                                    {
                                                        .texture = textures[textureIndex],
                                                        .baseLayout = rachis::TextureLayout::copyDestination,
                                                        .finalLayout = rachis::TextureLayout::copySource,
                                                    }) ||
                            !device.copyTextureToBuffer(
                                command, textures[textureIndex], readbacks[textureIndex], std::array{tailRegion}))
                        {
                            workerOk[workerIndex] = false;
                            break;
                        }
                    }
                });
            }
            workers.clear();
            stopCollector.store(true, std::memory_order_release);
        }

        for (const bool ok : workerOk)
        {
            REQUIRE(ok);
        }

        auto submitted = device.submit(commands);
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted).has_value());
    };

    recordRound(true, false);
    device.collectGarbage();

    for (std::size_t index = 0; index < totalTextures; ++index)
    {
        writeSolidSeed(index,
                       static_cast<std::uint8_t>(35U + index * 8U),
                       static_cast<std::uint8_t>(55U + index * 6U),
                       static_cast<std::uint8_t>(75U + index * 4U));
    }
    recordRound(false, true);
    device.collectGarbage();

    for (std::size_t index = 0; index < totalTextures; ++index)
    {
        auto mapped = device.mapBuffer(readbacks[index]);
        REQUIRE(mapped);
        CHECK(nearByte(mapped->data[0], static_cast<std::uint8_t>(35U + index * 8U)));
        CHECK(nearByte(mapped->data[1], static_cast<std::uint8_t>(55U + index * 6U)));
        CHECK(nearByte(mapped->data[2], static_cast<std::uint8_t>(75U + index * 4U)));
        CHECK(mapped->data[3] == std::byte{255});
        REQUIRE(device.unmapBuffer(readbacks[index]).has_value());

        REQUIRE(device.destroyTexture(textures[index]).has_value());
        REQUIRE(device.destroyBuffer(uploads[index]).has_value());
        REQUIRE(device.destroyBuffer(readbacks[index]).has_value());
    }
    REQUIRE(device.destroyCommandPool(*sharedCommandPool).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
