#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include "Vulkan/VulkanTestHooks.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

    constexpr std::uint32_t targetSize = 64;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t imageBytes = static_cast<std::uint64_t>(rowPitch) * targetSize;

    struct DistancePushConstants final
    {
        float clipDistance = 0.0F;
        float cullDistance = 0.0F;
    };

    struct PrimitiveRestartVertex final
    {
        float position[2];
        std::uint8_t color[4];
    };

    [[nodiscard]] rachis::DeviceDesc vulkanDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
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

#ifdef RACHIS_SPIRV_DIR
    [[nodiscard]] std::vector<std::byte> loadSpirv(const char* name)
    {
        return loadBytes(std::filesystem::path(RACHIS_SPIRV_DIR) / name);
    }

    constexpr std::uint32_t pointTargetSize = 8;

    using PixelCoordinate = std::pair<std::uint32_t, std::uint32_t>;

    const std::vector<PixelCoordinate> expectedPointPixels{{1, 3}, {3, 3}, {5, 3}, {7, 3}};

    void countPointSizeMessages(const rachis::DiagnosticMessage& message, void* userData) noexcept
    {
        if (message.description.find("PointSize") != std::string_view::npos)
        {
            static_cast<std::atomic_uint32_t*>(userData)->fetch_add(1, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] rachis::DeviceDesc pointSizeDiagnosticDesc(std::atomic_uint32_t& pointSizeMessages) noexcept
    {
        auto desc = vulkanDesc();
        desc.diagnosticCallback = &countPointSizeMessages;
        desc.diagnosticUserData = &pointSizeMessages;
        return desc;
    }

    class HiddenMaintenance5 final
    {
    public:
        HiddenMaintenance5() noexcept
        {
            rachis::vulkan::test::hideMaintenance5().store(true, std::memory_order_relaxed);
        }

        ~HiddenMaintenance5()
        {
            rachis::vulkan::test::hideMaintenance5().store(false, std::memory_order_relaxed);
        }

        HiddenMaintenance5(const HiddenMaintenance5&) = delete;
        HiddenMaintenance5& operator=(const HiddenMaintenance5&) = delete;
        HiddenMaintenance5(HiddenMaintenance5&&) = delete;
        HiddenMaintenance5& operator=(HiddenMaintenance5&&) = delete;
    };

    class ForcedScissorClipping final
    {
    public:
        explicit ForcedScissorClipping(bool enabled) noexcept : enabled_(enabled)
        {
            if (enabled_)
            {
                rachis::vulkan::test::forceScissorClipping().store(true, std::memory_order_relaxed);
            }
        }

        ~ForcedScissorClipping()
        {
            if (enabled_)
            {
                rachis::vulkan::test::forceScissorClipping().store(false, std::memory_order_relaxed);
            }
        }

        ForcedScissorClipping(const ForcedScissorClipping&) = delete;
        ForcedScissorClipping& operator=(const ForcedScissorClipping&) = delete;
        ForcedScissorClipping(ForcedScissorClipping&&) = delete;
        ForcedScissorClipping& operator=(ForcedScissorClipping&&) = delete;

    private:
        bool enabled_ = false;
    };

    [[nodiscard]] rachis::ShaderModuleHandle createPointShader(rachis::Device& device,
                                                               rachis::ShaderStage stage,
                                                               const char* file,
                                                               std::string_view entryPoint)
    {
        const auto bytecode = loadSpirv(file);
        REQUIRE_FALSE(bytecode.empty());
        auto shader = device.createShaderModule({
            .stage = stage,
            .bytecode = bytecode,
            .entryPoint = entryPoint,
            .shaderInterface = {},
        });
        if (!shader)
        {
            UNSCOPED_INFO(shader.error().message);
        }
        REQUIRE(shader);
        return *shader;
    }

    [[nodiscard]] rachis::Result<rachis::PipelineHandle> createPointPipeline(rachis::Device& device,
                                                                             rachis::PipelineLayoutHandle layout,
                                                                             rachis::ShaderModuleHandle vertexShader,
                                                                             rachis::ShaderModuleHandle pixelShader)
    {
        const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
        return device.createGraphicsPipeline({
            .layout = layout,
            .vertexShader = vertexShader,
            .pixelShader = pixelShader,
            .topology = rachis::PrimitiveTopology::pointList,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
        });
    }

    [[nodiscard]] std::vector<PixelCoordinate> drawFourPoints(rachis::Device& device, rachis::PipelineHandle pipeline)
    {
        auto target = device.createTexture({
            .width = pointTargetSize,
            .height = pointTargetSize,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        });
        auto readback = device.createBuffer({
            .size = static_cast<std::uint64_t>(rowPitch) * pointTargetSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(target);
        REQUIRE(readback);
        auto view = device.createTextureView({.texture = *target, .usage = rachis::TextureViewUsage::renderTarget});
        REQUIRE(view);

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *target,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
        const std::array attachments{
            rachis::ColorAttachment{
                .view = *view,
                .loadOperation = rachis::LoadOperation::clear,
                .clearValue = {},
            },
        };
        REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}));
        REQUIRE(device.bindGraphicsPipeline(*command, pipeline));
        REQUIRE(device.setViewport(*command,
                                   {
                                       .width = static_cast<float>(pointTargetSize),
                                       .height = static_cast<float>(pointTargetSize),
                                       .maxDepth = 1.0F,
                                   }));
        REQUIRE(device.setScissor(*command, {.width = pointTargetSize, .height = pointTargetSize}));
        REQUIRE(device.draw(*command, 4));
        REQUIRE(device.endRendering(*command));
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *target,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        REQUIRE(device.copyTextureToBuffer(*command,
                                           {
                                               .buffer = *readback,
                                               .bytesPerRow = rowPitch,
                                               .rowsPerImage = pointTargetSize,
                                               .texture = *target,
                                               .extent = {pointTargetSize, pointTargetSize, 1},
                                           }));
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted));

        std::vector<PixelCoordinate> lit;
        auto mapped = device.mapBuffer(*readback);
        REQUIRE(mapped);
        for (std::uint32_t y = 0; y < pointTargetSize; ++y)
        {
            for (std::uint32_t x = 0; x < pointTargetSize; ++x)
            {
                const auto* pixel =
                    mapped->data + static_cast<std::size_t>(y) * rowPitch + static_cast<std::size_t>(x) * 4;
                if (std::to_integer<std::uint8_t>(pixel[0]) != 0)
                {
                    lit.emplace_back(x, y);
                }
            }
        }
        REQUIRE(device.unmapBuffer(*readback));
        REQUIRE(device.destroyTextureView(*view));
        REQUIRE(device.destroyTexture(*target));
        REQUIRE(device.destroyBuffer(*readback));
        return lit;
    }
#endif

    [[nodiscard]] bool nearByte(std::byte actual, std::uint8_t expected) noexcept
    {
        const auto value = std::to_integer<std::uint8_t>(actual);
        const auto lower = expected == 0 ? 0U : static_cast<std::uint32_t>(expected) - 1U;
        const auto upper = (std::min)(static_cast<std::uint32_t>(expected) + 1U, 255U);
        return value >= lower && value <= upper;
    }

    [[nodiscard]] std::size_t coloredPixelCount(const rachis::MappedMemory& mapped, std::uint64_t offset) noexcept
    {
        std::size_t count = 0;
        for (std::uint32_t y = 0; y < targetSize; ++y)
        {
            for (std::uint32_t x = 0; x < targetSize; ++x)
            {
                auto* const pixel = mapped.data + static_cast<std::size_t>(offset) +
                                    static_cast<std::size_t>(y) * rowPitch + static_cast<std::size_t>(x) * 4;
                if (std::to_integer<std::uint8_t>(pixel[0]) != 0 || std::to_integer<std::uint8_t>(pixel[1]) != 0 ||
                    std::to_integer<std::uint8_t>(pixel[2]) != 0)
                {
                    ++count;
                }
            }
        }
        return count;
    }

    [[nodiscard]] bool layerHasColor(const rachis::MappedMemory& mapped,
                                     std::uint64_t layerOffset,
                                     std::array<std::byte, 4> expected) noexcept
    {
        for (std::uint32_t y = 0; y < targetSize; ++y)
        {
            for (std::uint32_t x = 0; x < targetSize; ++x)
            {
                const auto offset =
                    layerOffset + static_cast<std::uint64_t>(y) * rowPitch + static_cast<std::uint64_t>(x) * 4;
                const std::span actual{mapped.data + static_cast<std::size_t>(offset), expected.size()};
                if (!std::ranges::equal(actual, expected))
                {
                    return false;
                }
            }
        }
        return true;
    }

#ifdef RACHIS_SPIRV_DIR
    struct TwoViewDraw final
    {
        rachis::PipelineHandle pipeline{};
        rachis::BindingSetHandle bindingSet{};
        rachis::ShaderStage pushConstantStage = rachis::ShaderStage::vertex;
        std::uint32_t vertexCount = 0;
        bool mesh = false;
    };

    [[nodiscard]] std::array<std::size_t, 2> renderTwoViews(rachis::Device& device, const TwoViewDraw& draw)
    {
        constexpr std::uint32_t twoViewMask = 0b11;
        auto target = device.createTexture({
            .width = targetSize,
            .height = targetSize,
            .depthOrArrayLayers = 2,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        });
        auto readback = device.createBuffer({
            .size = imageBytes * 2,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(target);
        REQUIRE(readback);
        auto view = device.createTextureView({
            .texture = *target,
            .usage = rachis::TextureViewUsage::renderTarget,
            .dimension = rachis::TextureViewDimension::texture2DArray,
            .arrayLayerCount = 2,
        });
        REQUIRE(view);

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        REQUIRE(device.textureBarrier(*command,
                                      {
                                          .texture = *target,
                                          .stagesAfter = rachis::PipelineStage::renderTarget,
                                          .accessAfter = rachis::Access::renderTarget,
                                          .layoutAfter = rachis::TextureLayout::renderTarget,
                                      }));
        const std::array attachments{
            rachis::ColorAttachment{
                .view = *view,
                .loadOperation = rachis::LoadOperation::clear,
                .clearValue = {0.0F, 0.0F, 0.0F, 1.0F},
            },
        };
        REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments, .viewMask = twoViewMask}));
        REQUIRE(device.bindGraphicsPipeline(*command, draw.pipeline));
        if (draw.bindingSet.isValid())
        {
            REQUIRE(device.bindGraphicsBindingSet(*command, 0, draw.bindingSet));
        }
        REQUIRE(device.setViewport(*command,
                                   {
                                       .width = static_cast<float>(targetSize),
                                       .height = static_cast<float>(targetSize),
                                       .maxDepth = 1.0F,
                                   }));
        REQUIRE(device.setScissor(*command, {.width = targetSize, .height = targetSize}));
        const std::array control{1.0F, 0.0F, 0.0F, 0.0F};
        REQUIRE(
            device.setGraphicsPushConstants(*command, draw.pushConstantStage, 0, std::as_bytes(std::span(control))));
        if (draw.mesh)
        {
            REQUIRE(device.dispatchMesh(*command, 1));
        }
        else
        {
            REQUIRE(device.draw(*command, draw.vertexCount));
        }
        REQUIRE(device.endRendering(*command));
        REQUIRE(device.textureBarrier(*command,
                                      {
                                          .texture = *target,
                                          .stagesBefore = rachis::PipelineStage::renderTarget,
                                          .stagesAfter = rachis::PipelineStage::copy,
                                          .accessBefore = rachis::Access::renderTarget,
                                          .accessAfter = rachis::Access::copySource,
                                          .layoutBefore = rachis::TextureLayout::renderTarget,
                                          .layoutAfter = rachis::TextureLayout::copySource,
                                      }));
        for (std::uint32_t layer = 0; layer < 2; ++layer)
        {
            REQUIRE(device.copyTextureToBuffer(*command,
                                               {
                                                   .buffer = *readback,
                                                   .bufferOffset = layer * imageBytes,
                                                   .bytesPerRow = rowPitch,
                                                   .rowsPerImage = targetSize,
                                                   .texture = *target,
                                                   .textureSubresource = {.arrayLayer = layer},
                                                   .extent = {targetSize, targetSize, 1},
                                               }));
        }
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted));
        auto mapped = device.mapBuffer(*readback);
        REQUIRE(mapped);
        const std::array counts{coloredPixelCount(*mapped, 0), coloredPixelCount(*mapped, imageBytes)};
        REQUIRE(device.unmapBuffer(*readback));
        REQUIRE(device.destroyTextureView(*view));
        REQUIRE(device.destroyTexture(*target));
        REQUIRE(device.destroyBuffer(*readback));
        return counts;
    }

    void checkTwoViewPipeline(rachis::Device& device, rachis::Result<rachis::PipelineHandle> pipeline, TwoViewDraw draw)
    {
        if (!pipeline)
        {
            CHECK(pipeline.error().code == rachis::ErrorCode::unsupported);
            return;
        }
        draw.pipeline = *pipeline;
        const auto counts = renderTwoViews(device, draw);
        CHECK(counts[0] > 0);
        CHECK(counts[0] == counts[1]);
        REQUIRE(device.destroyPipeline(*pipeline));
    }
#endif

} // namespace

TEST_CASE("Vulkan advertises optional raster families only as complete contracts",
          "[size:medium][rhi][vulkan][raster][capabilities]")
{
#ifdef __APPLE__
    SUCCEED("Complete optional raster advertisement is a Windows/Linux desktop contract.");
    return;
#endif
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = device.capabilities();
    REQUIRE(capabilities.has_value());

    rachis::test::referenceDesktopRequires(capabilities->geometryShaders);
    CHECK(capabilities->fragmentPrimitiveId == capabilities->geometryShaders);
    if (rachis::test::referenceDesktopRequires(capabilities->tessellationShaders))
    {
        REQUIRE(capabilities->tessellation.maxPatchControlPoints >= 3);
    }
    if (rachis::test::referenceDesktopRequires(
            rachis::hasAny(capabilities->layeredRenderingModes, rachis::LayeredRenderingMode::vertexStage)))
    {
        REQUIRE(capabilities->limits.maxFramebufferLayers >= 2);
    }
    if (rachis::test::referenceDesktopRequires(capabilities->multiviewRendering))
    {
        REQUIRE(capabilities->maxMultiviewViewCount >= 2);
    }
    if (rachis::test::referenceDesktopRequires(capabilities->multiViewport))
    {
        REQUIRE(capabilities->limits.maxViewports >= 2);
    }
    rachis::test::referenceDesktopRequires(capabilities->depthBoundsTest);
    rachis::test::referenceDesktopRequires(capabilities->independentFrontAndBackStencilReference);
    rachis::test::referenceDesktopRequires(capabilities->wireframeRasterization);
    rachis::test::referenceDesktopRequires(capabilities->dynamicDepthBias);
    rachis::test::referenceDesktopRequires(capabilities->conservativeRasterization);
    rachis::test::referenceDesktopRequires(capabilities->unclippedDepth);
    rachis::test::referenceDesktopRequires(capabilities->primitiveRestart);
    rachis::test::referenceDesktopRequires(capabilities->triangleFans);
    rachis::test::referenceDesktopRequires(capabilities->samplerMinMaxReduction);
    rachis::test::referenceDesktopRequires(capabilities->fragmentBarycentricCoordinates);
    rachis::test::referenceDesktopRequires(capabilities->dualSourceBlending);
    REQUIRE(capabilities->limits.maxClipDistances >= 1);
    REQUIRE(capabilities->limits.maxCullDistances >= 1);
    REQUIRE(capabilities->limits.maxCombinedClipAndCullDistances >= 2);

    const auto rgba8 = device.queryFormatProperties(rachis::Format::rgba8Unorm);
    const auto r32Uint = device.queryFormatProperties(rachis::Format::r32Uint);
    const auto rgba16Float = device.queryFormatProperties(rachis::Format::rgba16Float);
    CHECK_FALSE(hasAny(rgba8.features, rachis::FormatFeature::logicOperation));
    CHECK_FALSE(hasAny(rgba16Float.features, rachis::FormatFeature::logicOperation));
    CHECK(hasAny(r32Uint.features, rachis::FormatFeature::logicOperation) ==
          rachis::test::referenceDesktopRequires(capabilities->logicOperations));
}

#ifdef RACHIS_SPIRV_DIR
TEST_CASE("Vulkan geometry and tessellation stages produce GPU raster results",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][raster][geometry][tessellation]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    if (rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(device).geometryShaders,
                                            "geometryShaders") ||
        rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(device).tessellationShaders,
                                            "tessellationShaders"))
    {
        return;
    }

    const auto geometryVs = loadSpirv("GeometryShader.vs.spv");
    const auto geometryGs = loadSpirv("GeometryShader.gs.spv");
    const auto geometryPs = loadSpirv("GeometryShader.ps.spv");
    const auto tessVs = loadSpirv("TessellationShader.vs.spv");
    const auto tessHs = loadSpirv("TessellationShader.hs.spv");
    const auto tessDs = loadSpirv("TessellationShader.ds.spv");
    const auto tessPs = loadSpirv("TessellationShader.ps.spv");
    REQUIRE_FALSE(geometryVs.empty());
    REQUIRE_FALSE(geometryGs.empty());
    REQUIRE_FALSE(geometryPs.empty());
    REQUIRE_FALSE(tessVs.empty());
    REQUIRE_FALSE(tessHs.empty());
    REQUIRE_FALSE(tessDs.empty());
    REQUIRE_FALSE(tessPs.empty());

    auto geometryVertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = geometryVs,
        .entryPoint = "pointVertex",
        .shaderInterface = {},
    });
    auto geometryShader = device.createShaderModule({
        .stage = rachis::ShaderStage::geometry,
        .bytecode = geometryGs,
        .entryPoint = "geometryResourceMain",
        .shaderInterface = rachis::test::shaderInterface(
            std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1}},
            std::array{rachis::ShaderPushConstantRequirement{0, 4 * sizeof(float)}}),
    });
    auto geometryPixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = geometryPs,
        .entryPoint = "geometryPixelMain",
        .shaderInterface = {},
    });
    auto tessVertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = tessVs,
        .entryPoint = "tessellationVertex",
        .shaderInterface = {},
    });
    auto tessHull = device.createShaderModule({
        .stage = rachis::ShaderStage::hull,
        .bytecode = tessHs,
        .entryPoint = "tessellationHull",
        .shaderInterface = rachis::test::shaderInterface(
            std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1}}),
    });
    auto tessDomain = device.createShaderModule({
        .stage = rachis::ShaderStage::domain,
        .bytecode = tessDs,
        .entryPoint = "tessellationDomain",
        .shaderInterface =
            rachis::test::shaderInterface({}, std::array{rachis::ShaderPushConstantRequirement{0, 4 * sizeof(float)}}),
    });
    auto tessPixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = tessPs,
        .entryPoint = "tessellationPixel",
        .shaderInterface = {},
    });
    REQUIRE(geometryVertex);
    REQUIRE(geometryShader);
    REQUIRE(geometryPixel);
    REQUIRE(tessVertex);
    REQUIRE(tessHull);
    REQUIRE(tessDomain);
    REQUIRE(tessPixel);

    auto constants = device.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(constants);
    auto mappedConstants = device.mapBuffer(*constants);
    REQUIRE(mappedConstants);
    std::memset(mappedConstants->data, 0, mappedConstants->size);
    const std::array geometryColor{0.0F, 1.0F, 0.0F, 1.0F};
    std::memcpy(mappedConstants->data, geometryColor.data(), sizeof(geometryColor));
    const std::array hullConstants{0.0F, 1.0F, 0.0F, 1.0F, 4.0F, 0.0F, 0.0F, 0.0F};
    std::memcpy(mappedConstants->data, hullConstants.data(), sizeof(hullConstants));
    REQUIRE(device.unmapBuffer(*constants));

    const std::array geometryEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::geometry},
    };
    const std::array hullEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::hull},
    };
    auto geometryBindingLayout = device.createBindingLayout({.entries = geometryEntries});
    auto hullBindingLayout = device.createBindingLayout({.entries = hullEntries});
    REQUIRE(geometryBindingLayout);
    REQUIRE(hullBindingLayout);
    const std::array writes{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*constants, 0, 256}},
    };
    auto geometrySet = rachis::test::createBindingSet(device, {.layout = *geometryBindingLayout, .writes = writes});
    auto hullSet = rachis::test::createBindingSet(device, {.layout = *hullBindingLayout, .writes = writes});
    REQUIRE(geometrySet);
    REQUIRE(hullSet);
    const std::array geometryLayouts{*geometryBindingLayout};
    const std::array hullLayouts{*hullBindingLayout};
    const std::array geometryPush{rachis::PushConstantRange{rachis::ShaderStage::geometry, 0, 16}};
    const std::array domainPush{rachis::PushConstantRange{rachis::ShaderStage::domain, 0, 16}};
    auto geometryLayout =
        device.createPipelineLayout({.bindingLayouts = geometryLayouts, .pushConstantRanges = geometryPush});
    auto tessLayout = device.createPipelineLayout({.bindingLayouts = hullLayouts, .pushConstantRanges = domainPush});
    REQUIRE(geometryLayout);
    REQUIRE(tessLayout);

    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto geometryPipeline = device.createGraphicsPipeline({
        .layout = *geometryLayout,
        .vertexShader = *geometryVertex,
        .pixelShader = *geometryPixel,
        .geometryShader = *geometryShader,
        .topology = rachis::PrimitiveTopology::pointList,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    auto tessPipeline = device.createGraphicsPipeline({
        .layout = *tessLayout,
        .vertexShader = *tessVertex,
        .pixelShader = *tessPixel,
        .hullShader = *tessHull,
        .domainShader = *tessDomain,
        .topology = rachis::PrimitiveTopology::patchList,
        .patchControlPoints = 3,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    auto wrongStage = device.createGraphicsPipeline({
        .layout = *geometryLayout,
        .vertexShader = *geometryVertex,
        .pixelShader = *geometryPixel,
        .geometryShader = *geometryPixel,
        .topology = rachis::PrimitiveTopology::pointList,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    REQUIRE(geometryPipeline);
    REQUIRE(tessPipeline);
    REQUIRE_FALSE(wrongStage);
    CHECK(wrongStage.error().code == rachis::ErrorCode::invalidArgument);

    auto target = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
    });
    auto readback = device.createBuffer({
        .size = imageBytes * 2,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(target);
    REQUIRE(readback);
    auto view = device.createTextureView({.texture = *target, .usage = rachis::TextureViewUsage::renderTarget});
    REQUIRE(view);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *target,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array attachments{
        rachis::ColorAttachment{
            .view = *view,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 0.0F, 1.0F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}));
    REQUIRE(device.bindGraphicsPipeline(*command, *geometryPipeline));
    REQUIRE(device.bindGraphicsBindingSet(*command, 0, *geometrySet));
    REQUIRE(device.setViewport(*command,
                               {
                                   .width = static_cast<float>(targetSize),
                                   .height = static_cast<float>(targetSize),
                                   .maxDepth = 1.0F,
                               }));
    REQUIRE(device.setScissor(*command, {.width = targetSize, .height = targetSize}));
    const std::array geometryControl{1.0F, 0.0F, 0.0F, 0.0F};
    REQUIRE(device.setGraphicsPushConstants(
        *command, rachis::ShaderStage::geometry, 0, std::as_bytes(std::span(geometryControl))));
    REQUIRE(device.draw(*command, 1));
    REQUIRE(device.endRendering(*command));
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *target,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command,
                                       {
                                           .buffer = *readback,
                                           .bytesPerRow = rowPitch,
                                           .rowsPerImage = targetSize,
                                           .texture = *target,
                                           .extent = {targetSize, targetSize, 1},
                                       }));
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *target,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessBefore = rachis::Access::copySource,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutBefore = rachis::TextureLayout::copySource,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}));
    REQUIRE(device.bindGraphicsPipeline(*command, *tessPipeline));
    REQUIRE(device.bindGraphicsBindingSet(*command, 0, *hullSet));
    REQUIRE(device.setViewport(*command,
                               {
                                   .width = static_cast<float>(targetSize),
                                   .height = static_cast<float>(targetSize),
                                   .maxDepth = 1.0F,
                               }));
    REQUIRE(device.setScissor(*command, {.width = targetSize, .height = targetSize}));
    const std::array domainControl{1.0F, 0.0F, 0.0F, 0.0F};
    REQUIRE(device.setGraphicsPushConstants(
        *command, rachis::ShaderStage::domain, 0, std::as_bytes(std::span(domainControl))));
    REQUIRE(device.draw(*command, 3));
    REQUIRE(device.endRendering(*command));
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *target,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command,
                                       {
                                           .buffer = *readback,
                                           .bufferOffset = imageBytes,
                                           .bytesPerRow = rowPitch,
                                           .rowsPerImage = targetSize,
                                           .texture = *target,
                                           .extent = {targetSize, targetSize, 1},
                                       }));
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    CHECK(coloredPixelCount(*mapped, 0) > 0);
    CHECK(coloredPixelCount(*mapped, imageBytes) > 0);
    const auto* geometryCenter = mapped->data + static_cast<std::size_t>(targetSize / 2) * rowPitch +
                                 static_cast<std::size_t>(targetSize / 2) * 4;
    CHECK(nearByte(geometryCenter[1], 255));
    REQUIRE(device.unmapBuffer(*readback));
}

TEST_CASE("Vulkan point lists without a geometry stage draw one-pixel points without a PointSize write",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][raster][points]")
{
    std::atomic_uint32_t pointSizeMessages{0};
    auto created = rachis::createDevice(pointSizeDiagnosticDesc(pointSizeMessages));
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto vertex = createPointShader(device, rachis::ShaderStage::vertex, "PointList.vs.spv", "vertexMain");
    const auto pixel = createPointShader(device, rachis::ShaderStage::pixel, "PointList.ps.spv", "pixelMain");
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout);

    auto pipeline = createPointPipeline(device, *layout, vertex, pixel);
    if (!pipeline)
    {
        UNSCOPED_INFO(pipeline.error().message);
    }
    if (!rachis::test::referenceDesktopRequires(pipeline.has_value()))
    {
        CHECK(pipeline.error().code == rachis::ErrorCode::unsupported);
        SKIP("This Vulkan device does not provide VK_KHR_maintenance5.");
    }
    CHECK(drawFourPoints(device, *pipeline) == expectedPointPixels);
    CHECK(pointSizeMessages.load(std::memory_order_relaxed) == 0);
    CHECK(device.diagnosticCounts().errors == 0);

    REQUIRE(device.destroyPipeline(*pipeline));
    REQUIRE(device.destroyPipelineLayout(*layout));
    REQUIRE(device.destroyShaderModule(pixel));
    REQUIRE(device.destroyShaderModule(vertex));
}

TEST_CASE("Vulkan point lists without maintenance5 require a vertex shader that writes PointSize",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][raster][points]")
{
    const HiddenMaintenance5 hidden;
    std::atomic_uint32_t pointSizeMessages{0};
    auto created = rachis::createDevice(pointSizeDiagnosticDesc(pointSizeMessages));
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto vertex = createPointShader(device, rachis::ShaderStage::vertex, "PointList.vs.spv", "vertexMain");
    const auto sizedVertex =
        createPointShader(device, rachis::ShaderStage::vertex, "PointList.sized.vs.spv", "sizedVertexMain");
    const auto pixel = createPointShader(device, rachis::ShaderStage::pixel, "PointList.ps.spv", "pixelMain");
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout);

    auto unsized = createPointPipeline(device, *layout, vertex, pixel);
    REQUIRE_FALSE(unsized);
    CHECK(unsized.error().code == rachis::ErrorCode::unsupported);
    CHECK(unsized.error().message.find("PointSize") != std::string::npos);

    auto sized = createPointPipeline(device, *layout, sizedVertex, pixel);
    if (!sized)
    {
        UNSCOPED_INFO(sized.error().message);
    }
    REQUIRE(sized);
    CHECK(drawFourPoints(device, *sized) == expectedPointPixels);
    CHECK(pointSizeMessages.load(std::memory_order_relaxed) == 0);
    CHECK(device.diagnosticCounts().errors == 0);

    REQUIRE(device.destroyPipeline(*sized));
    REQUIRE(device.destroyPipelineLayout(*layout));
    REQUIRE(device.destroyShaderModule(pixel));
    REQUIRE(device.destroyShaderModule(sizedVertex));
    REQUIRE(device.destroyShaderModule(vertex));
}

TEST_CASE("Vulkan layered, multiview, clip, and fragment builtin raster contracts hold",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][raster][layered][multiview]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (rachis::test::skipWithoutCapability(
            rachis::hasAny(capabilities.layeredRenderingModes, rachis::LayeredRenderingMode::vertexStage),
            "layeredRenderingModes") ||
        rachis::test::skipWithoutCapability(capabilities.multiviewRendering, "multiviewRendering") ||
        rachis::test::skipWithoutCapability(capabilities.fragmentPrimitiveId, "fragmentPrimitiveId") ||
        rachis::test::skipUnlessReferenceDesktopAdvertises(capabilities.fragmentBarycentricCoordinates,
                                                           "fragmentBarycentricCoordinates"))
    {
        return;
    }
    REQUIRE(capabilities.limits.maxClipDistances >= 1);
    REQUIRE(capabilities.limits.maxCullDistances >= 1);

    const auto layeredVs = loadSpirv("LayeredRendering.vs.spv");
    const auto layeredPs = loadSpirv("LayeredRendering.ps.spv");
    const auto multiviewVs = loadSpirv("MultiviewRendering.vs.spv");
    const auto multiviewPs = loadSpirv("MultiviewRendering.ps.spv");
    const auto distanceVs = loadSpirv("ShaderDistance.vs.spv");
    const auto distancePs = loadSpirv("ShaderDistance.ps.spv");
    const auto primitiveVs = loadSpirv("FragmentShaderBuiltins.primitive.vs.spv");
    const auto primitivePs = loadSpirv("FragmentShaderBuiltins.primitive.ps.spv");
    const auto baryVs = loadSpirv("FragmentShaderBuiltins.barycentric.vs.spv");
    const auto baryPs = loadSpirv("FragmentShaderBuiltins.barycentric.ps.spv");
    REQUIRE_FALSE(layeredVs.empty());
    REQUIRE_FALSE(layeredPs.empty());
    REQUIRE_FALSE(multiviewVs.empty());
    REQUIRE_FALSE(multiviewPs.empty());
    REQUIRE_FALSE(distanceVs.empty());
    REQUIRE_FALSE(distancePs.empty());
    REQUIRE_FALSE(primitiveVs.empty());
    REQUIRE_FALSE(primitivePs.empty());
    REQUIRE_FALSE(baryVs.empty());
    REQUIRE_FALSE(baryPs.empty());

    auto layeredVertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = layeredVs,
        .entryPoint = "vertexMain",
        .shaderInterface = {},
    });
    auto layeredPixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = layeredPs,
        .entryPoint = "pixelMain",
        .shaderInterface = {},
    });
    auto multiviewVertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = multiviewVs,
        .entryPoint = "vsMain",
        .shaderInterface = {},
    });
    auto multiviewPixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = multiviewPs,
        .entryPoint = "psMain",
        .shaderInterface = {},
    });
    auto distanceVertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = distanceVs,
        .entryPoint = "vertexMain",
        .shaderInterface = rachis::test::shaderInterface(
            {}, std::array{rachis::ShaderPushConstantRequirement{0, sizeof(DistancePushConstants)}}),
    });
    auto distancePixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = distancePs,
        .entryPoint = "pixelMain",
        .shaderInterface = {},
    });
    auto primitiveVertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = primitiveVs,
        .entryPoint = "primitiveIdVertex",
        .shaderInterface = {},
    });
    auto primitivePixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = primitivePs,
        .entryPoint = "primitiveIdPixel",
        .shaderInterface = {},
    });
    auto baryVertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = baryVs,
        .entryPoint = "barycentricVertex",
        .shaderInterface = {},
    });
    auto baryPixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = baryPs,
        .entryPoint = "barycentricPixel",
        .shaderInterface = {},
    });
    REQUIRE(layeredVertex);
    REQUIRE(layeredPixel);
    REQUIRE(multiviewVertex);
    REQUIRE(multiviewPixel);
    REQUIRE(distanceVertex);
    REQUIRE(distancePixel);
    REQUIRE(primitiveVertex);
    REQUIRE(primitivePixel);
    REQUIRE(baryVertex);
    REQUIRE(baryPixel);

    auto emptyLayout = device.createPipelineLayout({});
    const std::array distancePush{
        rachis::PushConstantRange{rachis::ShaderStage::vertex, 0, sizeof(DistancePushConstants)}};
    auto distanceLayout = device.createPipelineLayout({.pushConstantRanges = distancePush});
    REQUIRE(emptyLayout);
    REQUIRE(distanceLayout);
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto layeredPipeline = device.createGraphicsPipeline({
        .layout = *emptyLayout,
        .vertexShader = *layeredVertex,
        .pixelShader = *layeredPixel,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    constexpr std::uint32_t twoViewMask = 0b110;
    auto multiviewPipeline = device.createGraphicsPipeline({
        .layout = *emptyLayout,
        .vertexShader = *multiviewVertex,
        .pixelShader = *multiviewPixel,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .viewMask = twoViewMask,
    });
    auto invalidMask = device.createGraphicsPipeline({
        .layout = *emptyLayout,
        .vertexShader = *multiviewVertex,
        .pixelShader = *multiviewPixel,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .viewMask = 0b101,
    });
    auto distancePipeline = device.createGraphicsPipeline({
        .layout = *distanceLayout,
        .vertexShader = *distanceVertex,
        .pixelShader = *distancePixel,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .clipDistanceCount = 1,
        .cullDistanceCount = 1,
    });
    auto primitivePipeline = device.createGraphicsPipeline({
        .layout = *emptyLayout,
        .vertexShader = *primitiveVertex,
        .pixelShader = *primitivePixel,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    auto baryPipeline = device.createGraphicsPipeline({
        .layout = *emptyLayout,
        .vertexShader = *baryVertex,
        .pixelShader = *baryPixel,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    REQUIRE(layeredPipeline);
    REQUIRE(multiviewPipeline);
    REQUIRE_FALSE(invalidMask);
    CHECK(invalidMask.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(distancePipeline);
    REQUIRE(primitivePipeline);
    REQUIRE(baryPipeline);

    auto layeredTarget = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .depthOrArrayLayers = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
    });
    auto multiviewTarget = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .depthOrArrayLayers = 3,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
    });
    auto singleTarget = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
    });
    auto readback = device.createBuffer({
        .size = imageBytes * 4,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(layeredTarget);
    REQUIRE(multiviewTarget);
    REQUIRE(singleTarget);
    REQUIRE(readback);
    auto layeredView = device.createTextureView({
        .texture = *layeredTarget,
        .usage = rachis::TextureViewUsage::renderTarget,
        .dimension = rachis::TextureViewDimension::texture2DArray,
        .arrayLayerCount = 2,
    });
    auto multiviewView = device.createTextureView({
        .texture = *multiviewTarget,
        .usage = rachis::TextureViewUsage::renderTarget,
        .dimension = rachis::TextureViewDimension::texture2DArray,
        .arrayLayerCount = 3,
    });
    auto singleView = device.createTextureView({
        .texture = *singleTarget,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(layeredView);
    REQUIRE(multiviewView);
    REQUIRE(singleView);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    const auto toRender = [](rachis::TextureHandle texture) {
        return rachis::TextureBarrier{
            .texture = texture,
            .stagesAfter = rachis::PipelineStage::renderTarget,
            .accessAfter = rachis::Access::renderTarget,
            .layoutAfter = rachis::TextureLayout::renderTarget,
        };
    };
    const auto toCopy = [](rachis::TextureHandle texture) {
        return rachis::TextureBarrier{
            .texture = texture,
            .stagesBefore = rachis::PipelineStage::renderTarget,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::renderTarget,
            .accessAfter = rachis::Access::copySource,
            .layoutBefore = rachis::TextureLayout::renderTarget,
            .layoutAfter = rachis::TextureLayout::copySource,
        };
    };
    REQUIRE(device.textureBarrier(*command, toRender(*layeredTarget)));
    REQUIRE(device.textureBarrier(*command, toRender(*multiviewTarget)));
    REQUIRE(device.textureBarrier(*command, toRender(*singleTarget)));
    const std::array layeredAttachments{
        rachis::ColorAttachment{
            .view = *layeredView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 0.0F, 0.0F, 0.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = layeredAttachments}));
    REQUIRE(device.bindGraphicsPipeline(*command, *layeredPipeline));
    REQUIRE(device.setViewport(*command,
                               {.width = static_cast<float>(targetSize), .height = static_cast<float>(targetSize)}));
    REQUIRE(device.setScissor(*command, {.width = targetSize, .height = targetSize}));
    REQUIRE(device.draw(*command, 3, 2));
    REQUIRE(device.endRendering(*command));
    REQUIRE(device.textureBarrier(*command, toCopy(*layeredTarget)));
    for (std::uint32_t layer = 0; layer < 2; ++layer)
    {
        REQUIRE(device.copyTextureToBuffer(*command,
                                           {
                                               .buffer = *readback,
                                               .bufferOffset = layer * imageBytes,
                                               .bytesPerRow = rowPitch,
                                               .rowsPerImage = targetSize,
                                               .texture = *layeredTarget,
                                               .textureSubresource = {.arrayLayer = layer},
                                               .extent = {targetSize, targetSize, 1},
                                           }));
    }

    const std::array multiviewAttachments{
        rachis::ColorAttachment{
            .view = *multiviewView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 0.0F, 0.0F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = multiviewAttachments, .viewMask = twoViewMask}));
    REQUIRE(device.bindGraphicsPipeline(*command, *multiviewPipeline));
    REQUIRE(device.setViewport(*command,
                               {.width = static_cast<float>(targetSize), .height = static_cast<float>(targetSize)}));
    REQUIRE(device.setScissor(*command, {.width = targetSize, .height = targetSize}));
    REQUIRE(device.draw(*command, 3));
    REQUIRE(device.endRendering(*command));
    REQUIRE(device.textureBarrier(*command, toCopy(*multiviewTarget)));
    REQUIRE(device.copyTextureToBuffer(*command,
                                       {
                                           .buffer = *readback,
                                           .bufferOffset = 2 * imageBytes,
                                           .bytesPerRow = rowPitch,
                                           .rowsPerImage = targetSize,
                                           .texture = *multiviewTarget,
                                           .textureSubresource = {.arrayLayer = 1},
                                           .extent = {targetSize, targetSize, 1},
                                       }));

    const std::array singleAttachments{
        rachis::ColorAttachment{
            .view = *singleView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 0.0F, 0.0F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = singleAttachments}));
    REQUIRE(device.bindGraphicsPipeline(*command, *distancePipeline));
    REQUIRE(device.setViewport(*command,
                               {.width = static_cast<float>(targetSize), .height = static_cast<float>(targetSize)}));
    REQUIRE(device.setScissor(*command, {.width = targetSize, .height = targetSize}));
    const DistancePushConstants keep{1.0F, 1.0F};
    REQUIRE(
        device.setGraphicsPushConstants(*command, rachis::ShaderStage::vertex, 0, std::as_bytes(std::span(&keep, 1))));
    REQUIRE(device.draw(*command, 3));
    REQUIRE(device.endRendering(*command));
    REQUIRE(device.textureBarrier(*command, toCopy(*singleTarget)));
    REQUIRE(device.copyTextureToBuffer(*command,
                                       {
                                           .buffer = *readback,
                                           .bufferOffset = 3 * imageBytes,
                                           .bytesPerRow = rowPitch,
                                           .rowsPerImage = targetSize,
                                           .texture = *singleTarget,
                                           .extent = {targetSize, targetSize, 1},
                                       }));
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    CHECK(layerHasColor(*mapped, 0, {std::byte{0xFF}, std::byte{0}, std::byte{0}, std::byte{0xFF}}));
    CHECK(layerHasColor(*mapped, imageBytes, {std::byte{0}, std::byte{0xFF}, std::byte{0}, std::byte{0xFF}}));
    CHECK(layerHasColor(*mapped, 2 * imageBytes, {std::byte{0xFF}, std::byte{0}, std::byte{0}, std::byte{0xFF}}));
    CHECK(coloredPixelCount(*mapped, 3 * imageBytes) > 0);
    REQUIRE(device.unmapBuffer(*readback));

    auto builtinTarget = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
    });
    auto builtinReadback = device.createBuffer({
        .size = imageBytes * 2,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(builtinTarget);
    REQUIRE(builtinReadback);
    auto builtinView =
        device.createTextureView({.texture = *builtinTarget, .usage = rachis::TextureViewUsage::renderTarget});
    REQUIRE(builtinView);
    auto builtinCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(builtinCommand);
    REQUIRE(device.textureBarrier(*builtinCommand, toRender(*builtinTarget)));
    const std::array builtinAttachments{
        rachis::ColorAttachment{
            .view = *builtinView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 0.0F, 1.0F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*builtinCommand, {.colorAttachments = builtinAttachments}));
    REQUIRE(device.bindGraphicsPipeline(*builtinCommand, *primitivePipeline));
    REQUIRE(device.setViewport(*builtinCommand,
                               {.width = static_cast<float>(targetSize), .height = static_cast<float>(targetSize)}));
    REQUIRE(device.setScissor(*builtinCommand, {.width = targetSize, .height = targetSize}));
    REQUIRE(device.draw(*builtinCommand, 6));
    REQUIRE(device.endRendering(*builtinCommand));
    REQUIRE(device.textureBarrier(*builtinCommand, toCopy(*builtinTarget)));
    REQUIRE(device.copyTextureToBuffer(*builtinCommand,
                                       {
                                           .buffer = *builtinReadback,
                                           .bytesPerRow = rowPitch,
                                           .rowsPerImage = targetSize,
                                           .texture = *builtinTarget,
                                           .extent = {targetSize, targetSize, 1},
                                       }));
    REQUIRE(device.textureBarrier(*builtinCommand,
                                  {
                                      .texture = *builtinTarget,
                                      .stagesBefore = rachis::PipelineStage::copy,
                                      .stagesAfter = rachis::PipelineStage::renderTarget,
                                      .accessBefore = rachis::Access::copySource,
                                      .accessAfter = rachis::Access::renderTarget,
                                      .layoutBefore = rachis::TextureLayout::copySource,
                                      .layoutAfter = rachis::TextureLayout::renderTarget,
                                  }));
    REQUIRE(device.beginRendering(*builtinCommand, {.colorAttachments = builtinAttachments}));
    REQUIRE(device.bindGraphicsPipeline(*builtinCommand, *baryPipeline));
    REQUIRE(device.setViewport(*builtinCommand,
                               {.width = static_cast<float>(targetSize), .height = static_cast<float>(targetSize)}));
    REQUIRE(device.setScissor(*builtinCommand, {.width = targetSize, .height = targetSize}));
    REQUIRE(device.draw(*builtinCommand, 3));
    REQUIRE(device.endRendering(*builtinCommand));
    REQUIRE(device.textureBarrier(*builtinCommand, toCopy(*builtinTarget)));
    REQUIRE(device.copyTextureToBuffer(*builtinCommand,
                                       {
                                           .buffer = *builtinReadback,
                                           .bufferOffset = imageBytes,
                                           .bytesPerRow = rowPitch,
                                           .rowsPerImage = targetSize,
                                           .texture = *builtinTarget,
                                           .extent = {targetSize, targetSize, 1},
                                       }));
    auto builtinSubmitted = device.submit(std::move(*builtinCommand));
    REQUIRE(builtinSubmitted);
    REQUIRE(device.wait(*builtinSubmitted));
    auto builtinMapped = device.mapBuffer(*builtinReadback);
    REQUIRE(builtinMapped);
    CHECK(coloredPixelCount(*builtinMapped, 0) > 0);
    CHECK(coloredPixelCount(*builtinMapped, imageBytes) > 0);
    REQUIRE(device.unmapBuffer(*builtinReadback));
}

TEST_CASE("Vulkan multiview geometry, tessellation, and mesh pipelines render both views or report unsupported",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][raster][multiview][geometry][tessellation][mesh]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (rachis::test::skipWithoutCapability(capabilities.multiviewRendering, "multiviewRendering") ||
        rachis::test::skipWithoutCapability(capabilities.geometryShaders || capabilities.tessellationShaders ||
                                                capabilities.meshShaders,
                                            "geometry, tessellation, or mesh shaders"))
    {
        return;
    }
    REQUIRE(capabilities.maxMultiviewViewCount >= 2);

    constexpr std::uint32_t twoViewMask = 0b11;
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto constants = device.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(constants);
    auto mappedConstants = device.mapBuffer(*constants);
    REQUIRE(mappedConstants);
    std::memset(mappedConstants->data, 0, mappedConstants->size);
    const std::array stageConstants{0.0F, 1.0F, 0.0F, 1.0F, 4.0F, 0.0F, 0.0F, 0.0F};
    std::memcpy(mappedConstants->data, stageConstants.data(), sizeof(stageConstants));
    REQUIRE(device.unmapBuffer(*constants));
    const std::array writes{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*constants, 0, 256}},
    };

    if (capabilities.geometryShaders)
    {
        const auto vs = loadSpirv("GeometryShader.vs.spv");
        const auto gs = loadSpirv("GeometryShader.gs.spv");
        const auto ps = loadSpirv("GeometryShader.ps.spv");
        REQUIRE_FALSE(vs.empty());
        REQUIRE_FALSE(gs.empty());
        REQUIRE_FALSE(ps.empty());
        auto vertex = device.createShaderModule({
            .stage = rachis::ShaderStage::vertex,
            .bytecode = vs,
            .entryPoint = "pointVertex",
            .shaderInterface = {},
        });
        auto geometry = device.createShaderModule({
            .stage = rachis::ShaderStage::geometry,
            .bytecode = gs,
            .entryPoint = "geometryResourceMain",
            .shaderInterface = rachis::test::shaderInterface(
                std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1}},
                std::array{rachis::ShaderPushConstantRequirement{0, 4 * sizeof(float)}}),
        });
        auto pixel = device.createShaderModule({
            .stage = rachis::ShaderStage::pixel,
            .bytecode = ps,
            .entryPoint = "geometryPixelMain",
            .shaderInterface = {},
        });
        REQUIRE(vertex);
        REQUIRE(geometry);
        REQUIRE(pixel);
        const std::array entries{
            rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::geometry},
        };
        auto bindingLayout = device.createBindingLayout({.entries = entries});
        REQUIRE(bindingLayout);
        auto bindingSet = rachis::test::createBindingSet(device, {.layout = *bindingLayout, .writes = writes});
        REQUIRE(bindingSet);
        const std::array bindingLayouts{*bindingLayout};
        const std::array push{rachis::PushConstantRange{rachis::ShaderStage::geometry, 0, 16}};
        auto layout = device.createPipelineLayout({.bindingLayouts = bindingLayouts, .pushConstantRanges = push});
        REQUIRE(layout);
        checkTwoViewPipeline(device,
                             device.createGraphicsPipeline({
                                 .layout = *layout,
                                 .vertexShader = *vertex,
                                 .pixelShader = *pixel,
                                 .geometryShader = *geometry,
                                 .topology = rachis::PrimitiveTopology::pointList,
                                 .cullMode = rachis::CullMode::none,
                                 .colorTargets = colorTargets,
                                 .viewMask = twoViewMask,
                             }),
                             {
                                 .bindingSet = *bindingSet,
                                 .pushConstantStage = rachis::ShaderStage::geometry,
                                 .vertexCount = 1,
                             });
    }

    if (capabilities.tessellationShaders)
    {
        const auto vs = loadSpirv("TessellationShader.vs.spv");
        const auto hs = loadSpirv("TessellationShader.hs.spv");
        const auto ds = loadSpirv("TessellationShader.ds.spv");
        const auto ps = loadSpirv("TessellationShader.ps.spv");
        REQUIRE_FALSE(vs.empty());
        REQUIRE_FALSE(hs.empty());
        REQUIRE_FALSE(ds.empty());
        REQUIRE_FALSE(ps.empty());
        auto vertex = device.createShaderModule({
            .stage = rachis::ShaderStage::vertex,
            .bytecode = vs,
            .entryPoint = "tessellationVertex",
            .shaderInterface = {},
        });
        auto hull = device.createShaderModule({
            .stage = rachis::ShaderStage::hull,
            .bytecode = hs,
            .entryPoint = "tessellationHull",
            .shaderInterface = rachis::test::shaderInterface(
                std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1}}),
        });
        auto domain = device.createShaderModule({
            .stage = rachis::ShaderStage::domain,
            .bytecode = ds,
            .entryPoint = "tessellationDomain",
            .shaderInterface = rachis::test::shaderInterface(
                {}, std::array{rachis::ShaderPushConstantRequirement{0, 4 * sizeof(float)}}),
        });
        auto pixel = device.createShaderModule({
            .stage = rachis::ShaderStage::pixel,
            .bytecode = ps,
            .entryPoint = "tessellationPixel",
            .shaderInterface = {},
        });
        REQUIRE(vertex);
        REQUIRE(hull);
        REQUIRE(domain);
        REQUIRE(pixel);
        const std::array entries{
            rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::hull},
        };
        auto bindingLayout = device.createBindingLayout({.entries = entries});
        REQUIRE(bindingLayout);
        auto bindingSet = rachis::test::createBindingSet(device, {.layout = *bindingLayout, .writes = writes});
        REQUIRE(bindingSet);
        const std::array bindingLayouts{*bindingLayout};
        const std::array push{rachis::PushConstantRange{rachis::ShaderStage::domain, 0, 16}};
        auto layout = device.createPipelineLayout({.bindingLayouts = bindingLayouts, .pushConstantRanges = push});
        REQUIRE(layout);
        checkTwoViewPipeline(device,
                             device.createGraphicsPipeline({
                                 .layout = *layout,
                                 .vertexShader = *vertex,
                                 .pixelShader = *pixel,
                                 .hullShader = *hull,
                                 .domainShader = *domain,
                                 .topology = rachis::PrimitiveTopology::patchList,
                                 .patchControlPoints = 3,
                                 .cullMode = rachis::CullMode::none,
                                 .colorTargets = colorTargets,
                                 .viewMask = twoViewMask,
                             }),
                             {
                                 .bindingSet = *bindingSet,
                                 .pushConstantStage = rachis::ShaderStage::domain,
                                 .vertexCount = 3,
                             });
    }

    if (capabilities.meshShaders)
    {
        const auto ms = loadSpirv("TaskMeshWorkload.no-task.ms.spv");
        const auto ps = loadSpirv("TaskMeshWorkload.ps.spv");
        REQUIRE_FALSE(ms.empty());
        REQUIRE_FALSE(ps.empty());
        auto mesh = device.createShaderModule({
            .stage = rachis::ShaderStage::mesh,
            .bytecode = ms,
            .entryPoint = "meshNoTask",
            .shaderInterface = rachis::test::computeShaderInterface(
                {1, 1, 1}, {}, std::array{rachis::ShaderPushConstantRequirement{0, 4 * sizeof(float)}}),
        });
        auto pixel = device.createShaderModule({
            .stage = rachis::ShaderStage::pixel,
            .bytecode = ps,
            .entryPoint = "meshPixel",
            .shaderInterface = {},
        });
        REQUIRE(mesh);
        REQUIRE(pixel);
        const std::array push{rachis::PushConstantRange{rachis::ShaderStage::mesh, 0, 16}};
        auto layout = device.createPipelineLayout({.pushConstantRanges = push});
        REQUIRE(layout);
        checkTwoViewPipeline(device,
                             device.createGraphicsPipeline({
                                 .layout = *layout,
                                 .pixelShader = *pixel,
                                 .meshShader = *mesh,
                                 .cullMode = rachis::CullMode::none,
                                 .colorTargets = colorTargets,
                                 .viewMask = twoViewMask,
                             }),
                             {
                                 .pushConstantStage = rachis::ShaderStage::mesh,
                                 .mesh = true,
                             });
    }

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan extended raster state, dual-source, logic ops, and sampler reduction hold",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][raster][extended]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (rachis::test::skipWithoutCapability(capabilities.logicOperations, "logicOperations") ||
        rachis::test::skipWithoutCapability(capabilities.depthBoundsTest, "depthBoundsTest") ||
        rachis::test::skipWithoutCapability(capabilities.conservativeRasterization, "conservativeRasterization"))
    {
        return;
    }
    REQUIRE(capabilities.dynamicDepthBias);
    REQUIRE(capabilities.independentFrontAndBackStencilReference);

    const auto triangleVs = loadSpirv("VulkanGraphicsTriangle.direct.vs.spv");
    const auto trianglePs = loadSpirv("VulkanGraphicsTriangle.ps.spv");
    const auto dualPs = loadSpirv("DualSourceBlend.ps.spv");
    const auto logicPs = loadSpirv("LogicOperation.ps.spv");
    const auto reductionCs = loadSpirv("SamplerReduction.spv");
    const auto fanVs = loadSpirv("TriangleFan.vs.spv");
    const auto fanPs = loadSpirv("TriangleFan.ps.spv");
    const auto restartVs = loadSpirv("PrimitiveRestart.vs.spv");
    const auto restartPs = loadSpirv("PrimitiveRestart.ps.spv");
    const auto viewportGs = loadSpirv("GeometryShader.viewport.gs.spv");
    const auto geometryVs = loadSpirv("GeometryShader.vs.spv");
    const auto geometryPs = loadSpirv("GeometryShader.ps.spv");
    REQUIRE_FALSE(triangleVs.empty());
    REQUIRE_FALSE(trianglePs.empty());
    REQUIRE_FALSE(dualPs.empty());
    REQUIRE_FALSE(logicPs.empty());
    REQUIRE_FALSE(reductionCs.empty());
    REQUIRE_FALSE(fanVs.empty());
    REQUIRE_FALSE(fanPs.empty());
    REQUIRE_FALSE(restartVs.empty());
    REQUIRE_FALSE(restartPs.empty());
    REQUIRE_FALSE(viewportGs.empty());
    REQUIRE_FALSE(geometryVs.empty());
    REQUIRE_FALSE(geometryPs.empty());

    auto vertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = triangleVs,
        .entryPoint = "directVertex",
        .shaderInterface = {},
    });
    auto pixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = trianglePs,
        .entryPoint = "pixelMain",
        .shaderInterface = rachis::test::graphicsPixelShaderInterface(),
    });
    auto dualPixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = dualPs,
        .entryPoint = "main",
        .shaderInterface = {},
    });
    auto logicPixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = logicPs,
        .entryPoint = "pixelMain",
        .shaderInterface = rachis::test::shaderInterface(
            {}, std::array{rachis::ShaderPushConstantRequirement{0, sizeof(std::uint32_t)}}),
    });
    auto reductionShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = reductionCs,
        .entryPoint = "main",
        .shaderInterface = rachis::test::computeShaderInterface(
            {1, 1, 1},
            std::array{
                rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::sampledTexture, 1},
                rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::sampler, 1},
                rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::sampler, 1},
                rachis::ShaderBindingRequirement{0, 3, rachis::ShaderResourceType::storageBuffer, 1},
            }),
    });
    auto fanVertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = fanVs,
        .entryPoint = "vertexMain",
        .shaderInterface = {},
    });
    auto fanPixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = fanPs,
        .entryPoint = "pixelMain",
        .shaderInterface = {},
    });
    auto restartVertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = restartVs,
        .entryPoint = "vertexMain",
        .shaderInterface = rachis::test::shaderInterface({}, {}, std::array{0U, 1U}),
    });
    auto restartPixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = restartPs,
        .entryPoint = "pixelMain",
        .shaderInterface = {},
    });
    auto viewportVertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = geometryVs,
        .entryPoint = "pointVertex",
        .shaderInterface = {},
    });
    auto viewportGeometry = device.createShaderModule({
        .stage = rachis::ShaderStage::geometry,
        .bytecode = viewportGs,
        .entryPoint = "viewportArrayMain",
        .shaderInterface = {},
    });
    auto viewportPixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = geometryPs,
        .entryPoint = "geometryPixelMain",
        .shaderInterface = {},
    });
    REQUIRE(vertex);
    REQUIRE(pixel);
    REQUIRE(dualPixel);
    REQUIRE(logicPixel);
    REQUIRE(reductionShader);
    REQUIRE(fanVertex);
    REQUIRE(fanPixel);
    REQUIRE(restartVertex);
    REQUIRE(restartPixel);
    REQUIRE(viewportVertex);
    REQUIRE(viewportGeometry);
    REQUIRE(viewportPixel);

    auto emptyLayout = device.createPipelineLayout({});
    const std::array trianglePush{rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)}};
    auto triangleLayout = device.createPipelineLayout({.pushConstantRanges = trianglePush});
    const std::array logicPush{rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, sizeof(std::uint32_t)}};
    auto logicLayout = device.createPipelineLayout({.pushConstantRanges = logicPush});
    REQUIRE(emptyLayout);
    REQUIRE(triangleLayout);
    REQUIRE(logicLayout);
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto wireframe = device.createGraphicsPipeline({
        .layout = *triangleLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .cullMode = rachis::CullMode::none,
        .fillMode = rachis::FillMode::wireframe,
        .colorTargets = colorTargets,
    });
    auto conservative = device.createGraphicsPipeline({
        .layout = *triangleLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .conservativeRasterization = rachis::ConservativeRasterizationMode::overestimate,
    });
    auto unclipped = device.createGraphicsPipeline({
        .layout = *triangleLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .cullMode = rachis::CullMode::none,
        .depthClipEnabled = false,
        .colorTargets = colorTargets,
    });
    auto conservativeWire = device.createGraphicsPipeline({
        .layout = *triangleLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .cullMode = rachis::CullMode::none,
        .fillMode = rachis::FillMode::wireframe,
        .colorTargets = colorTargets,
        .conservativeRasterization = rachis::ConservativeRasterizationMode::overestimate,
    });
    auto dualPipeline = device.createGraphicsPipeline({
        .layout = *emptyLayout,
        .vertexShader = *vertex,
        .pixelShader = *dualPixel,
        .cullMode = rachis::CullMode::none,
        .colorTargets = std::array{rachis::ColorTargetState{
            .format = rachis::Format::rgba8Unorm,
            .blend =
                {
                    .enabled = true,
                    .color = {rachis::BlendFactor::source1, rachis::BlendFactor::oneMinusSource1},
                    .alpha = {rachis::BlendFactor::source1Alpha, rachis::BlendFactor::oneMinusSource1Alpha},
                },
        }},
    });
    auto logicPipeline = device.createGraphicsPipeline({
        .layout = *logicLayout,
        .vertexShader = *vertex,
        .pixelShader = *logicPixel,
        .cullMode = rachis::CullMode::none,
        .colorTargets = std::array{rachis::ColorTargetState{.format = rachis::Format::r32Uint}},
        .logicOperation = rachis::LogicOperation::copy,
    });
    auto fanPipeline = device.createGraphicsPipeline({
        .layout = *emptyLayout,
        .vertexShader = *fanVertex,
        .pixelShader = *fanPixel,
        .topology = rachis::PrimitiveTopology::triangleFan,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    const std::array restartAttributes{
        rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .shaderLocation = 0},
        rachis::VertexAttribute{
            .format = rachis::VertexFormat::unorm8x4,
            .offset = 2 * sizeof(float),
            .shaderLocation = 1,
        },
    };
    const std::array restartLayouts{
        rachis::VertexBufferLayout{.stride = sizeof(PrimitiveRestartVertex), .attributes = restartAttributes},
    };
    auto restartPipeline = device.createGraphicsPipeline({
        .layout = *emptyLayout,
        .vertexShader = *restartVertex,
        .pixelShader = *restartPixel,
        .vertexBuffers = restartLayouts,
        .topology = rachis::PrimitiveTopology::triangleStrip,
        .stripIndexFormat = rachis::IndexFormat::uint16,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    auto viewportPipeline = device.createGraphicsPipeline({
        .layout = *emptyLayout,
        .vertexShader = *viewportVertex,
        .pixelShader = *viewportPixel,
        .geometryShader = *viewportGeometry,
        .topology = rachis::PrimitiveTopology::pointList,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    auto depthBoundsPipeline = device.createGraphicsPipeline({
        .layout = *triangleLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .depthStencil =
            {
                .format = rachis::Format::d32Float,
                .depthTestEnabled = true,
                .depthWriteEnabled = true,
                .depthBoundsTestEnabled = true,
            },
    });
    auto dynamicBiasPipeline = device.createGraphicsPipeline({
        .layout = *triangleLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .cullMode = rachis::CullMode::none,
        .dynamicDepthBiasEnabled = true,
        .colorTargets = colorTargets,
    });
    REQUIRE(wireframe);
    REQUIRE(conservative);
    REQUIRE(unclipped);
    REQUIRE_FALSE(conservativeWire);
    CHECK(conservativeWire.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(dualPipeline);
    REQUIRE(logicPipeline);
    REQUIRE(fanPipeline);
    REQUIRE(restartPipeline);
    REQUIRE(viewportPipeline);
    REQUIRE(depthBoundsPipeline);
    REQUIRE(dynamicBiasPipeline);

    auto colorTarget = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
    });
    auto logicTarget = device.createTexture({
        .width = 1,
        .height = 1,
        .format = rachis::Format::r32Uint,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
    });
    auto readback = device.createBuffer({
        .size = imageBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(colorTarget);
    REQUIRE(logicTarget);
    REQUIRE(readback);
    auto colorView =
        device.createTextureView({.texture = *colorTarget, .usage = rachis::TextureViewUsage::renderTarget});
    auto logicView =
        device.createTextureView({.texture = *logicTarget, .usage = rachis::TextureViewUsage::renderTarget});
    REQUIRE(colorView);
    REQUIRE(logicView);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device.textureBarrier(*command,
                                  {
                                      .texture = *colorTarget,
                                      .stagesAfter = rachis::PipelineStage::renderTarget,
                                      .accessAfter = rachis::Access::renderTarget,
                                      .layoutAfter = rachis::TextureLayout::renderTarget,
                                  }));
    const std::array dualAttachments{
        rachis::ColorAttachment{
            .view = *colorView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 1.0F, 0.25F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = dualAttachments}));
    REQUIRE(device.bindGraphicsPipeline(*command, *dualPipeline));
    REQUIRE(device.setViewport(*command,
                               {.width = static_cast<float>(targetSize), .height = static_cast<float>(targetSize)}));
    REQUIRE(device.setScissor(*command, {.width = targetSize, .height = targetSize}));
    REQUIRE(device.draw(*command, 3));
    REQUIRE(device.endRendering(*command));
    REQUIRE(device.textureBarrier(*command,
                                  {
                                      .texture = *colorTarget,
                                      .stagesBefore = rachis::PipelineStage::renderTarget,
                                      .stagesAfter = rachis::PipelineStage::copy,
                                      .accessBefore = rachis::Access::renderTarget,
                                      .accessAfter = rachis::Access::copySource,
                                      .layoutBefore = rachis::TextureLayout::renderTarget,
                                      .layoutAfter = rachis::TextureLayout::copySource,
                                  }));
    REQUIRE(device.copyTextureToBuffer(*command,
                                       {
                                           .buffer = *readback,
                                           .bytesPerRow = rowPitch,
                                           .rowsPerImage = targetSize,
                                           .texture = *colorTarget,
                                           .extent = {targetSize, targetSize, 1},
                                       }));
    REQUIRE(device.textureBarrier(*command,
                                  {
                                      .texture = *logicTarget,
                                      .stagesAfter = rachis::PipelineStage::renderTarget,
                                      .accessAfter = rachis::Access::renderTarget,
                                      .layoutAfter = rachis::TextureLayout::renderTarget,
                                  }));
    const std::array logicAttachments{
        rachis::ColorAttachment{
            .view = *logicView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 0.0F, 0.0F, 0.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = logicAttachments}));
    REQUIRE(device.bindGraphicsPipeline(*command, *logicPipeline));
    REQUIRE(device.setViewport(*command, {.width = 1.0F, .height = 1.0F}));
    REQUIRE(device.setScissor(*command, {.width = 1, .height = 1}));
    const std::uint32_t logicSource = 0xA5A5A5A5U;
    REQUIRE(device.setGraphicsPushConstants(
        *command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(&logicSource, 1))));
    REQUIRE(device.draw(*command, 3));
    REQUIRE(device.endRendering(*command));

    const std::array viewports{
        rachis::Viewport{.width = 32.0F, .height = 64.0F},
        rachis::Viewport{.x = 32.0F, .width = 32.0F, .height = 64.0F},
    };
    const std::array scissors{
        rachis::ScissorRect{.width = 32, .height = 64},
        rachis::ScissorRect{.x = 32, .width = 32, .height = 64},
    };
    REQUIRE(device.textureBarrier(*command,
                                  {
                                      .texture = *colorTarget,
                                      .stagesBefore = rachis::PipelineStage::copy,
                                      .stagesAfter = rachis::PipelineStage::renderTarget,
                                      .accessBefore = rachis::Access::copySource,
                                      .accessAfter = rachis::Access::renderTarget,
                                      .layoutBefore = rachis::TextureLayout::copySource,
                                      .layoutAfter = rachis::TextureLayout::renderTarget,
                                  }));
    REQUIRE(device.beginRendering(*command, {.colorAttachments = dualAttachments}));
    REQUIRE(device.bindGraphicsPipeline(*command, *viewportPipeline));
    REQUIRE(device.setViewports(*command, viewports));
    REQUIRE(device.setScissors(*command, scissors));
    REQUIRE(device.draw(*command, 1));
    REQUIRE(device.endRendering(*command));
    REQUIRE(device.beginRendering(*command, {.colorAttachments = dualAttachments}));
    REQUIRE(device.bindGraphicsPipeline(*command, *dynamicBiasPipeline));
    REQUIRE(device.setViewport(*command,
                               {.width = static_cast<float>(targetSize), .height = static_cast<float>(targetSize)}));
    REQUIRE(device.setScissor(*command, {.width = targetSize, .height = targetSize}));
    const std::array tint{1.0F, 1.0F, 1.0F, 1.0F};
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(tint))));
    REQUIRE(device.setDepthBias(*command, {.constantFactor = 1.0F, .slopeScale = 1.0F}));
    REQUIRE(device.draw(*command, 3));
    REQUIRE(device.endRendering(*command));
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    const auto* center = mapped->data + static_cast<std::uint64_t>(targetSize / 2) * rowPitch +
                         static_cast<std::uint64_t>(targetSize / 2) * 4;
    CHECK(nearByte(center[0], 64));
    CHECK(nearByte(center[1], 128));
    CHECK(nearByte(center[2], 112));
    CHECK(nearByte(center[3], 255));
    REQUIRE(device.unmapBuffer(*readback));

    constexpr std::uint32_t textureRowPitch = 256;
    constexpr std::uint64_t textureUploadSize = static_cast<std::uint64_t>(textureRowPitch) * 2;
    constexpr std::uint64_t outputSize = 2 * sizeof(float);
    auto upload = device.createBuffer({
        .size = textureUploadSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto output = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
    });
    auto reductionReadback = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    auto texture = device.createTexture({
        .width = 2,
        .height = 2,
        .format = rachis::Format::r32Float,
        .usage = rachis::TextureUsage::copyDestination | rachis::TextureUsage::sampled,
    });
    REQUIRE(upload);
    REQUIRE(output);
    REQUIRE(reductionReadback);
    REQUIRE(texture);
    std::array<std::byte, textureUploadSize> uploadBytes{};
    constexpr std::array sourceValues{0.125F, 0.375F, 0.625F, 0.875F};
    std::memcpy(uploadBytes.data(), sourceValues.data(), 2 * sizeof(float));
    std::memcpy(uploadBytes.data() + textureRowPitch, sourceValues.data() + 2, 2 * sizeof(float));
    REQUIRE(device.writeBuffer(*upload, 0, uploadBytes));
    auto sampledView = device.createTextureView({.texture = *texture, .usage = rachis::TextureViewUsage::sampled});
    auto minimumSampler = device.createSampler({
        .minFilter = rachis::FilterMode::linear,
        .magFilter = rachis::FilterMode::linear,
        .reductionMode = rachis::SamplerReductionMode::minimum,
    });
    auto maximumSampler = device.createSampler({
        .minFilter = rachis::FilterMode::linear,
        .magFilter = rachis::FilterMode::linear,
        .reductionMode = rachis::SamplerReductionMode::maximum,
    });
    REQUIRE(sampledView);
    REQUIRE(minimumSampler);
    REQUIRE(maximumSampler);
    CHECK_FALSE(device
                    .createSampler({
                        .compareEnabled = true,
                        .reductionMode = rachis::SamplerReductionMode::minimum,
                    })
                    .has_value());
    const std::array reductionEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{1, rachis::BindingType::sampler, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{2, rachis::BindingType::sampler, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{3, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto reductionLayout = device.createBindingLayout({.entries = reductionEntries});
    REQUIRE(reductionLayout);
    const std::array reductionWrites{
        rachis::BindingWrite{.binding = 0, .resource = *sampledView},
        rachis::BindingWrite{.binding = 1, .resource = *minimumSampler},
        rachis::BindingWrite{.binding = 2, .resource = *maximumSampler},
        rachis::BindingWrite{
            .binding = 3,
            .resource =
                rachis::BufferBinding{.buffer = *output, .size = outputSize, .structureByteStride = sizeof(float)},
        },
    };
    auto reductionSet = rachis::test::createBindingSet(device, {.layout = *reductionLayout, .writes = reductionWrites});
    REQUIRE(reductionSet);
    const std::array reductionLayouts{*reductionLayout};
    auto reductionPipelineLayout = device.createPipelineLayout({.bindingLayouts = reductionLayouts});
    REQUIRE(reductionPipelineLayout);
    auto reductionPipeline = device.createComputePipeline({
        .layout = *reductionPipelineLayout,
        .computeShader = *reductionShader,
    });
    REQUIRE(reductionPipeline);
    auto compute = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(compute);
    REQUIRE(device.textureBarrier(*compute,
                                  {
                                      .texture = *texture,
                                      .stagesAfter = rachis::PipelineStage::copy,
                                      .accessAfter = rachis::Access::copyDestination,
                                      .layoutAfter = rachis::TextureLayout::copyDestination,
                                  }));
    REQUIRE(device.copyBufferToTexture(*compute,
                                       {
                                           .buffer = *upload,
                                           .bytesPerRow = textureRowPitch,
                                           .rowsPerImage = 2,
                                           .texture = *texture,
                                           .extent = {2, 2, 1},
                                       }));
    REQUIRE(device.textureBarrier(*compute,
                                  {
                                      .texture = *texture,
                                      .stagesBefore = rachis::PipelineStage::copy,
                                      .stagesAfter = rachis::PipelineStage::computeShader,
                                      .accessBefore = rachis::Access::copyDestination,
                                      .accessAfter = rachis::Access::shaderResource,
                                      .layoutBefore = rachis::TextureLayout::copyDestination,
                                      .layoutAfter = rachis::TextureLayout::shaderResource,
                                  }));
    REQUIRE(device.bufferBarrier(*compute,
                                 {
                                     .buffer = *output,
                                     .stagesAfter = rachis::PipelineStage::computeShader,
                                     .accessAfter = rachis::Access::unorderedAccess,
                                 }));
    REQUIRE(device.bindComputePipeline(*compute, *reductionPipeline));
    REQUIRE(device.bindComputeBindingSet(*compute, 0, *reductionSet));
    REQUIRE(device.dispatch(*compute, 1));
    REQUIRE(device.bufferBarrier(*compute,
                                 {
                                     .buffer = *output,
                                     .stagesBefore = rachis::PipelineStage::computeShader,
                                     .stagesAfter = rachis::PipelineStage::copy,
                                     .accessBefore = rachis::Access::unorderedAccess,
                                     .accessAfter = rachis::Access::copySource,
                                 }));
    REQUIRE(device.copyBuffer(*compute, {.source = *output, .destination = *reductionReadback, .size = outputSize}));
    auto computeSubmitted = device.submit(std::move(*compute));
    REQUIRE(computeSubmitted);
    REQUIRE(device.wait(*computeSubmitted));
    auto reductionMapped = device.mapBuffer(*reductionReadback);
    REQUIRE(reductionMapped);
    std::array<float, 2> extrema{};
    std::memcpy(extrema.data(), reductionMapped->data, sizeof(extrema));
    REQUIRE(device.unmapBuffer(*reductionReadback));
    CHECK(extrema[0] == Catch::Approx(0.125F).margin(0.01F));
    CHECK(extrema[1] == Catch::Approx(0.875F).margin(0.01F));
}
#endif

#ifdef RACHIS_SPIRV_DIR
TEST_CASE("Vulkan clips scissors to the render area on portability-subset devices",
          "[size:medium][rhi][vulkan][raster][scissor][portability]")
{
    const bool forced = GENERATE(false, true);
    CAPTURE(forced);
    auto created = [forced] {
        const ForcedScissorClipping force(forced);
        return rachis::createDevice(vulkanDesc());
    }();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
#ifdef __APPLE__
    const bool clipping = true;
#else
    const bool clipping = forced;
#endif
    const auto capabilities = rachis::test::requireCapabilities(device);

    struct ScissorCase final
    {
        const char* name = "";
        std::vector<std::vector<rachis::ScissorRect>> calls;
        std::size_t litPixels = 0;
        bool skipped = false;
    };
    std::vector<ScissorCase> cases{
        {"partly outside", {{{32, 32, 64, 64}}}, 1024, false},
        {"oversized", {{{0, 0, 4096, 4096}}}, 4096, false},
        {"wholly outside", {{{64, 64, 16, 16}}}, 0, true},
        {"inside, then wholly outside", {{{0, 0, 16, 16}}, {{64, 64, 16, 16}}}, 0, true},
        {"wholly outside, then inside", {{{64, 64, 16, 16}}, {{8, 8, 16, 16}}}, 256, false},
    };
    if (capabilities.multiViewport && capabilities.limits.maxViewports >= 2)
    {
        cases.push_back({"wholly outside in the drawn viewport only", {{{64, 0, 16, 16}, {0, 0, 16, 16}}}, 0, false});
        cases.push_back({"wholly outside in the other viewport only", {{{0, 0, 16, 16}, {0, 64, 16, 16}}}, 256, false});
        cases.push_back({"wholly outside in both viewports", {{{64, 0, 16, 16}, {0, 64, 16, 16}}}, 0, true});
    }

    const auto vertexBytes = loadSpirv("VulkanBaseShaderFeatures.fullscreen.vs.spv");
    const auto pixelBytes = loadSpirv("VulkanBaseShaderFeatures.solid.ps.spv");
    REQUIRE_FALSE(vertexBytes.empty());
    REQUIRE_FALSE(pixelBytes.empty());
    auto vertexShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytes,
        .entryPoint = "fullscreenVertex",
        .shaderInterface = {},
    });
    auto pixelShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytes,
        .entryPoint = "solidPixel",
        .shaderInterface = {},
    });
    REQUIRE(vertexShader);
    REQUIRE(pixelShader);
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout);
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto pipeline = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    REQUIRE(pipeline);
    auto target = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
    });
    auto readback = device.createBuffer({
        .size = imageBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(target);
    REQUIRE(readback);
    auto view = device.createTextureView({.texture = *target, .usage = rachis::TextureViewUsage::renderTarget});
    REQUIRE(view);
    std::optional<rachis::QueryPoolHandle> statisticsPool;
    std::optional<rachis::BufferHandle> statisticsReadback;
    if (capabilities.pipelineStatisticsQueries)
    {
        auto createdPool =
            device.createPipelineStatisticsQueryPool({.queryCount = static_cast<std::uint32_t>(cases.size())});
        auto createdReadback = device.createBuffer({
            .size = rachis::pipelineStatisticsQueryResultSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(createdPool);
        REQUIRE(createdReadback);
        statisticsPool = *createdPool;
        statisticsReadback = *createdReadback;
    }

    const std::array attachments{
        rachis::ColorAttachment{
            .view = *view,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {},
        },
    };
    for (std::uint32_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex)
    {
        const auto& scissorCase = cases[caseIndex];
        INFO(scissorCase.name);
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        auto toRender = rachis::TextureBarrier{
            .texture = *target,
            .stagesAfter = rachis::PipelineStage::renderTarget,
            .accessAfter = rachis::Access::renderTarget,
            .layoutAfter = rachis::TextureLayout::renderTarget,
        };
        if (caseIndex != 0)
        {
            toRender.stagesBefore = rachis::PipelineStage::copy;
            toRender.accessBefore = rachis::Access::copySource;
            toRender.layoutBefore = rachis::TextureLayout::copySource;
        }
        REQUIRE(device.textureBarrier(*command, toRender).has_value());
        if (statisticsPool.has_value())
        {
            REQUIRE(device.beginPipelineStatisticsQuery(*command, *statisticsPool, caseIndex).has_value());
        }
        REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
        REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
        const std::vector<rachis::Viewport> viewports(
            scissorCase.calls.back().size(),
            rachis::Viewport{.width = static_cast<float>(targetSize), .height = static_cast<float>(targetSize)});
        REQUIRE(device.setViewports(*command, viewports).has_value());
        for (const auto& scissors : scissorCase.calls)
        {
            REQUIRE(device.setScissors(*command, scissors).has_value());
        }
        REQUIRE(device.draw(*command, 3).has_value());
        REQUIRE(device.endRendering(*command).has_value());
        if (statisticsPool.has_value())
        {
            REQUIRE(device.endPipelineStatisticsQuery(*command).has_value());
            REQUIRE(device
                        .resolvePipelineStatisticsQueries(*command,
                                                          {
                                                              .queryPool = *statisticsPool,
                                                              .firstQuery = caseIndex,
                                                              .queryCount = 1,
                                                              .destination = *statisticsReadback,
                                                          })
                        .has_value());
        }
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *target,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        REQUIRE(device
                    .copyTextureToBuffer(*command,
                                         {
                                             .buffer = *readback,
                                             .bytesPerRow = rowPitch,
                                             .rowsPerImage = targetSize,
                                             .texture = *target,
                                             .extent = {targetSize, targetSize, 1},
                                         })
                    .has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted).has_value());

        auto mapped = device.mapBuffer(*readback);
        REQUIRE(mapped);
        const auto lit = coloredPixelCount(*mapped, 0);
        REQUIRE(device.unmapBuffer(*readback).has_value());
        CHECK(lit == scissorCase.litPixels);
        if (statisticsReadback.has_value())
        {
            auto mappedStatistics = device.mapBuffer(*statisticsReadback);
            REQUIRE(mappedStatistics);
            rachis::PipelineStatistics statistics{};
            std::memcpy(&statistics, mappedStatistics->data, sizeof(statistics));
            REQUIRE(device.unmapBuffer(*statisticsReadback).has_value());
            CAPTURE(statistics.vertexShaderInvocations);
            if (clipping && scissorCase.skipped)
            {
                CHECK(statistics.vertexShaderInvocations == 0);
            }
            else
            {
                CHECK(statistics.vertexShaderInvocations >= 3);
            }
        }
    }

    if (statisticsPool.has_value())
    {
        REQUIRE(device.destroyBuffer(*statisticsReadback).has_value());
        REQUIRE(device.destroyQueryPool(*statisticsPool).has_value());
    }
    REQUIRE(device.destroyTextureView(*view).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
#endif
