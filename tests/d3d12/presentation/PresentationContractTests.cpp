#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <Windows.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <span>
#include <vector>

namespace
{

    constexpr std::uint32_t initialWidth = 64;
    constexpr std::uint32_t initialHeight = 64;
    constexpr std::uint32_t resizedWidth = 80;
    constexpr std::uint32_t resizedHeight = 48;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t readbackSize = static_cast<std::uint64_t>(rowPitch) * initialHeight;

    class HiddenWindow final
    {
    public:
        explicit HiddenWindow(DWORD exStyle = 0)
            : window_(CreateWindowExW(exStyle,
                                      L"STATIC",
                                      L"Rachis presentation contract",
                                      WS_OVERLAPPEDWINDOW,
                                      CW_USEDEFAULT,
                                      CW_USEDEFAULT,
                                      static_cast<int>(initialWidth),
                                      static_cast<int>(initialHeight),
                                      nullptr,
                                      nullptr,
                                      GetModuleHandleW(nullptr),
                                      nullptr))
        {
        }

        ~HiddenWindow()
        {
            if (window_ != nullptr)
            {
                static_cast<void>(DestroyWindow(window_));
            }
        }

        HiddenWindow(const HiddenWindow&) = delete;
        HiddenWindow& operator=(const HiddenWindow&) = delete;
        HiddenWindow(HiddenWindow&&) = delete;
        HiddenWindow& operator=(HiddenWindow&&) = delete;

        [[nodiscard]] HWND get() const noexcept
        {
            return window_;
        }

    private:
        HWND window_ = nullptr;
    };

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

    class ForceFullyTypedFormatCastingUnsupported final
    {
    public:
        ForceFullyTypedFormatCastingUnsupported()
            : previous_(rachis::d3d12::test::forceFullyTypedFormatCastingUnsupported().exchange(
                  true, std::memory_order_relaxed))
        {
        }

        ~ForceFullyTypedFormatCastingUnsupported()
        {
            rachis::d3d12::test::forceFullyTypedFormatCastingUnsupported().store(previous_, std::memory_order_relaxed);
        }

        ForceFullyTypedFormatCastingUnsupported(const ForceFullyTypedFormatCastingUnsupported&) = delete;
        ForceFullyTypedFormatCastingUnsupported& operator=(const ForceFullyTypedFormatCastingUnsupported&) = delete;
        ForceFullyTypedFormatCastingUnsupported(ForceFullyTypedFormatCastingUnsupported&&) = delete;
        ForceFullyTypedFormatCastingUnsupported& operator=(ForceFullyTypedFormatCastingUnsupported&&) = delete;

    private:
        bool previous_ = false;
    };

    class FailSwapchainLeaseRetention final
    {
    public:
        explicit FailSwapchainLeaseRetention(std::int32_t countdown)
            : previous_(rachis::d3d12::test::swapchainLeaseRetentionFailureCountdown().exchange(
                  countdown, std::memory_order_relaxed))
        {
        }

        ~FailSwapchainLeaseRetention()
        {
            rachis::d3d12::test::swapchainLeaseRetentionFailureCountdown().store(previous_, std::memory_order_relaxed);
        }

        FailSwapchainLeaseRetention(const FailSwapchainLeaseRetention&) = delete;
        FailSwapchainLeaseRetention& operator=(const FailSwapchainLeaseRetention&) = delete;
        FailSwapchainLeaseRetention(FailSwapchainLeaseRetention&&) = delete;
        FailSwapchainLeaseRetention& operator=(FailSwapchainLeaseRetention&&) = delete;

    private:
        std::int32_t previous_ = 0;
    };

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    [[nodiscard]] std::vector<std::byte> loadShader(const char* path)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
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

    [[nodiscard]] bool nearByte(std::byte actual, std::uint8_t expected) noexcept
    {
        const auto value = std::to_integer<std::uint8_t>(actual);
        return value >= expected - 1U && value <= expected + 1U;
    }

    [[nodiscard]] constexpr std::uint32_t preferredSampleCount(rachis::SampleCountMask mask) noexcept
    {
        for (const auto count : {4U, 2U, 8U})
        {
            if (rachis::supportsSampleCount(mask, count))
            {
                return count;
            }
        }
        return 0;
    }

    [[nodiscard]] rachis::Result<rachis::TimelinePoint> submitPresentableFrame(rachis::Device& device,
                                                                               rachis::TextureHandle image)
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        if (!command)
        {
            return std::unexpected(std::move(command.error()));
        }
        auto leavePresent = device.textureBarrier(*command,
                                                  rachis::TextureBarrier{
                                                      .texture = image,
                                                      .stagesBefore = rachis::PipelineStage::none,
                                                      .stagesAfter = rachis::PipelineStage::copy,
                                                      .accessBefore = rachis::Access::none,
                                                      .accessAfter = rachis::Access::copyDestination,
                                                      .layoutBefore = rachis::TextureLayout::present,
                                                      .layoutAfter = rachis::TextureLayout::copyDestination,
                                                  });
        if (!leavePresent)
        {
            return std::unexpected(std::move(leavePresent.error()));
        }
        auto returnToPresent = device.textureBarrier(*command,
                                                     rachis::TextureBarrier{
                                                         .texture = image,
                                                         .stagesBefore = rachis::PipelineStage::copy,
                                                         .stagesAfter = rachis::PipelineStage::none,
                                                         .accessBefore = rachis::Access::copyDestination,
                                                         .accessAfter = rachis::Access::none,
                                                         .layoutBefore = rachis::TextureLayout::copyDestination,
                                                         .layoutAfter = rachis::TextureLayout::present,
                                                     });
        if (!returnToPresent)
        {
            return std::unexpected(std::move(returnToPresent.error()));
        }
        return device.submit(std::move(*command));
    }

    [[nodiscard]] rachis::SwapchainDesc copyableSwapchainDesc(rachis::SurfaceHandle surface,
                                                              std::uint32_t width,
                                                              std::uint32_t height)
    {
        return rachis::SwapchainDesc{
            .surface = surface,
            .width = width,
            .height = height,
            .imageCount = 2,
            .format = rachis::Format::bgra8Unorm,
            .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copyDestination,
        };
    }

} // namespace

TEST_CASE("surface capabilities define accepted swapchain profiles", "[size:medium][rhi][presentation][warp]")
{
    const HiddenWindow window;
    REQUIRE(window.get() != nullptr);

    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto invalidCapabilities = device.querySurfaceCapabilities({});
    REQUIRE_FALSE(invalidCapabilities.has_value());
    CHECK(invalidCapabilities.error().code == rachis::ErrorCode::invalidArgument);

    auto surface = device.createSurface(rachis::SurfaceDesc{
        .windowHandle = window.get(),
        .label = "Capability negotiation surface",
    });
    REQUIRE(surface.has_value());

    auto capabilities = device.querySurfaceCapabilities(*surface);
    REQUIRE(capabilities.has_value());
    CHECK(capabilities->maxWidth >= initialWidth);
    CHECK(capabilities->maxHeight >= initialHeight);
    CHECK(capabilities->preferredWidth > 0);
    CHECK(capabilities->preferredHeight > 0);
    CHECK(rachis::hasAny(capabilities->usages, rachis::TextureUsage::renderTarget));
    CHECK(rachis::hasAny(capabilities->usages, rachis::TextureUsage::copySource));
    CHECK(rachis::hasAny(capabilities->usages, rachis::TextureUsage::copyDestination));
    CHECK(rachis::hasAny(capabilities->usages, rachis::TextureUsage::sampled));
    CHECK(rachis::supportsPresentMode(capabilities->presentModes, rachis::PresentMode::fifo));
    CHECK(std::ranges::find(capabilities->formats,
                            rachis::SurfaceFormat{
                                .format = rachis::Format::bgra8Unorm,
                                .colorSpace = rachis::SurfaceColorSpace::srgbNonlinear,
                            }) != capabilities->formats.end());
    CHECK(std::ranges::find(capabilities->formats,
                            rachis::SurfaceFormat{
                                .format = rachis::Format::rgba8Unorm,
                                .colorSpace = rachis::SurfaceColorSpace::srgbNonlinear,
                            }) != capabilities->formats.end());

    const rachis::SwapchainDesc swapchainDesc{
        .surface = *surface,
        .width = initialWidth,
        .height = initialHeight,
        .imageCount = 3,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
        .presentMode = rachis::PresentMode::fifo,
        .label = "Capability negotiation swapchain",
    };

    auto invalidExtentDesc = swapchainDesc;
    invalidExtentDesc.width = 0;
    auto invalidExtent = device.createSwapchain(invalidExtentDesc);
    REQUIRE_FALSE(invalidExtent.has_value());
    CHECK(invalidExtent.error().code == rachis::ErrorCode::invalidArgument);

    auto unsupportedColorSpaceDesc = swapchainDesc;
    unsupportedColorSpaceDesc.colorSpace = rachis::SurfaceColorSpace::hdr10St2084;
    auto unsupportedColorSpace = device.createSwapchain(unsupportedColorSpaceDesc);
    REQUIRE_FALSE(unsupportedColorSpace.has_value());
    CHECK(unsupportedColorSpace.error().code == rachis::ErrorCode::unsupported);

    auto unsupportedUsageDesc = swapchainDesc;
    unsupportedUsageDesc.usage = unsupportedUsageDesc.usage | rachis::TextureUsage::storage;
    auto unsupportedUsage = device.createSwapchain(unsupportedUsageDesc);
    REQUIRE_FALSE(unsupportedUsage.has_value());
    CHECK(unsupportedUsage.error().code == rachis::ErrorCode::unsupported);

    auto invalidPresentModeDesc = swapchainDesc;
    invalidPresentModeDesc.presentMode = static_cast<rachis::PresentMode>(0xff);
    auto invalidPresentMode = device.createSwapchain(invalidPresentModeDesc);
    REQUIRE_FALSE(invalidPresentMode.has_value());
    CHECK(invalidPresentMode.error().code == rachis::ErrorCode::invalidArgument);

    auto immediateDesc = swapchainDesc;
    immediateDesc.presentMode = rachis::PresentMode::immediate;
    auto immediate = device.createSwapchain(immediateDesc);
    if (rachis::supportsPresentMode(capabilities->presentModes, rachis::PresentMode::immediate))
    {
        REQUIRE(immediate.has_value());
        auto immediateImage = device.acquireNextImage(*immediate);
        REQUIRE(immediateImage.has_value());
        REQUIRE(device.cancelSwapchainImage(*immediate).has_value());
        REQUIRE(device.destroySwapchain(*immediate).has_value());
    }
    else
    {
        REQUIRE_FALSE(immediate.has_value());
        CHECK(immediate.error().code == rachis::ErrorCode::unsupported);
    }

    auto swapchain = device.createSwapchain(swapchainDesc);
    REQUIRE(swapchain.has_value());
    auto duplicate = device.createSwapchain(swapchainDesc);
    REQUIRE_FALSE(duplicate.has_value());
    CHECK(duplicate.error().code == rachis::ErrorCode::invalidState);
    auto acquired = device.acquireNextImage(*swapchain);
    REQUIRE(acquired.has_value());
    CHECK(acquired->colorSpace == rachis::SurfaceColorSpace::srgbNonlinear);
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());

    auto busySurface = device.destroySurface(*surface);
    REQUIRE_FALSE(busySurface.has_value());
    CHECK(busySurface.error().code == rachis::ErrorCode::invalidState);
    CHECK(device.querySurfaceCapabilities(*surface).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());

    REQUIRE(device.destroySurface(*surface).has_value());
    auto staleCapabilities = device.querySurfaceCapabilities(*surface);
    REQUIRE_FALSE(staleCapabilities.has_value());
    CHECK(staleCapabilities.error().code == rachis::ErrorCode::invalidArgument);
    CHECK_FALSE(device.createSwapchain(swapchainDesc).has_value());
}

TEST_CASE("sRGB swapchains reject sampled usage without fully typed format casting",
          "[size:medium][rhi][presentation][warp]")
{
    const auto format = GENERATE(rachis::Format::bgra8UnormSrgb, rachis::Format::rgba8UnormSrgb);
    CAPTURE(format);

    const HiddenWindow window;
    REQUIRE(window.get() != nullptr);

    const ForceFullyTypedFormatCastingUnsupported noCasting;
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto surface = device.createSurface(rachis::SurfaceDesc{
        .windowHandle = window.get(),
        .label = "sRGB sampled-usage surface",
    });
    REQUIRE(surface.has_value());
    auto capabilities = device.querySurfaceCapabilities(*surface);
    REQUIRE(capabilities.has_value());
    CHECK(rachis::hasAny(capabilities->usages, rachis::TextureUsage::sampled));
    CHECK(std::ranges::find(capabilities->formats,
                            rachis::SurfaceFormat{
                                .format = format,
                                .colorSpace = rachis::SurfaceColorSpace::srgbNonlinear,
                            }) != capabilities->formats.end());

    const rachis::SwapchainDesc renderTargetDesc{
        .surface = *surface,
        .width = initialWidth,
        .height = initialHeight,
        .imageCount = 3,
        .format = format,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource |
                 rachis::TextureUsage::copyDestination,
        .presentMode = rachis::PresentMode::fifo,
        .label = "sRGB sampled-usage swapchain",
    };
    auto sampledDesc = renderTargetDesc;
    sampledDesc.usage = sampledDesc.usage | rachis::TextureUsage::sampled;
    auto sampled = device.createSwapchain(sampledDesc);
    REQUIRE_FALSE(sampled.has_value());
    CHECK(sampled.error().code == rachis::ErrorCode::unsupported);

    auto linearDesc = sampledDesc;
    linearDesc.format =
        format == rachis::Format::bgra8UnormSrgb ? rachis::Format::bgra8Unorm : rachis::Format::rgba8Unorm;
    auto linear = device.createSwapchain(linearDesc);
    REQUIRE(linear.has_value());
    REQUIRE(device.destroySwapchain(*linear).has_value());

    auto swapchain = device.createSwapchain(renderTargetDesc);
    REQUIRE(swapchain.has_value());
    auto acquired = device.acquireNextImage(*swapchain);
    REQUIRE(acquired.has_value());
    CHECK(acquired->format == format);
    CHECK(acquired->renderTargetView.isValid());
    auto sampledView = device.createTextureView({
        .texture = acquired->texture,
        .usage = rachis::TextureViewUsage::sampled,
        .label = "sRGB swapchain sampled view",
    });
    REQUIRE_FALSE(sampledView.has_value());
    CHECK(sampledView.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());

    const auto counts = device.diagnosticCounts();
    CHECK(counts.errors == 0);
    CHECK(counts.corruptions == 0);
}

TEST_CASE("sRGB swapchains accept sampled usage with fully typed format casting",
          "[size:medium][rhi][presentation][warp]")
{
    const auto format = GENERATE(rachis::Format::bgra8UnormSrgb, rachis::Format::rgba8UnormSrgb);
    CAPTURE(format);

    const HiddenWindow window;
    REQUIRE(window.get() != nullptr);

    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto surface = device.createSurface(rachis::SurfaceDesc{
        .windowHandle = window.get(),
        .label = "sRGB sampled swapchain surface",
    });
    REQUIRE(surface.has_value());
    auto swapchain = device.createSwapchain(rachis::SwapchainDesc{
        .surface = *surface,
        .width = initialWidth,
        .height = initialHeight,
        .imageCount = 3,
        .format = format,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::sampled,
        .presentMode = rachis::PresentMode::fifo,
        .label = "sRGB sampled swapchain",
    });
    REQUIRE(swapchain.has_value());
    auto acquired = device.acquireNextImage(*swapchain);
    REQUIRE(acquired.has_value());
    CHECK(acquired->format == format);
    auto sampledView = device.createTextureView({
        .texture = acquired->texture,
        .usage = rachis::TextureViewUsage::sampled,
        .label = "sRGB swapchain sampled view",
    });
    REQUIRE(sampledView.has_value());
    REQUIRE(device.destroyTextureView(*sampledView).has_value());
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());

    const auto counts = device.diagnosticCounts();
    CHECK(counts.errors == 0);
    CHECK(counts.corruptions == 0);
}

TEST_CASE("a window accepts one live swapchain across every surface on it", "[size:medium][rhi][presentation][warp]")
{
    const HiddenWindow window;
    const HiddenWindow otherWindow;
    REQUIRE(window.get() != nullptr);
    REQUIRE(otherWindow.get() != nullptr);

    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto first = device.createSurface(rachis::SurfaceDesc{.windowHandle = window.get()});
    REQUIRE(first.has_value());
    auto second = device.createSurface(rachis::SurfaceDesc{.windowHandle = window.get()});
    REQUIRE(second.has_value());
    auto other = device.createSurface(rachis::SurfaceDesc{.windowHandle = otherWindow.get()});
    REQUIRE(other.has_value());

    const auto desc = [](rachis::SurfaceHandle surface) {
        return rachis::SwapchainDesc{
            .surface = surface,
            .width = initialWidth,
            .height = initialHeight,
            .imageCount = 2,
            .format = rachis::Format::bgra8Unorm,
            .usage = rachis::TextureUsage::renderTarget,
            .presentMode = rachis::PresentMode::fifo,
            .label = "One swapchain per window",
        };
    };

    auto live = device.createSwapchain(desc(*first));
    if (!live)
    {
        INFO(live.error().message);
    }
    REQUIRE(live.has_value());
    auto refused = device.createSwapchain(desc(*second));
    REQUIRE_FALSE(refused.has_value());
    CHECK(refused.error().code == rachis::ErrorCode::invalidState);
    CHECK(refused.error().message == "The native window already has a live swapchain.");
    auto otherSwapchain = device.createSwapchain(desc(*other));
    REQUIRE(otherSwapchain.has_value());

    auto acquired = device.acquireNextImage(*live);
    REQUIRE(acquired.has_value());
    REQUIRE(device.cancelSwapchainImage(*live).has_value());
    REQUIRE(device.resizeSwapchain(*live, resizedWidth, resizedHeight).has_value());
    auto surfaceInUse = device.destroySurface(*first);
    REQUIRE_FALSE(surfaceInUse.has_value());
    CHECK(surfaceInUse.error().code == rachis::ErrorCode::invalidState);
    auto resizedRefused = device.createSwapchain(desc(*second));
    REQUIRE_FALSE(resizedRefused.has_value());
    CHECK(resizedRefused.error().code == rachis::ErrorCode::invalidState);
    auto third = device.createSurface(rachis::SurfaceDesc{.windowHandle = window.get()});
    REQUIRE(third.has_value());
    auto thirdRefused = device.createSwapchain(desc(*third));
    REQUIRE_FALSE(thirdRefused.has_value());
    CHECK(thirdRefused.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.destroySwapchain(*live).has_value());
    REQUIRE(device.destroySurface(*first).has_value());

    auto invalidDesc = desc(*second);
    invalidDesc.usage = invalidDesc.usage | rachis::TextureUsage::storage;
    auto invalid = device.createSwapchain(invalidDesc);
    REQUIRE_FALSE(invalid.has_value());
    CHECK(invalid.error().code == rachis::ErrorCode::unsupported);

    auto replacement = device.createSwapchain(desc(*second));
    if (!replacement)
    {
        INFO(replacement.error().message);
    }
    REQUIRE(replacement.has_value());
    auto replacementImage = device.acquireNextImage(*replacement);
    REQUIRE(replacementImage.has_value());
    REQUIRE(device.cancelSwapchainImage(*replacement).has_value());
    auto thirdStillRefused = device.createSwapchain(desc(*third));
    REQUIRE_FALSE(thirdStillRefused.has_value());
    CHECK(thirdStillRefused.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.destroySwapchain(*replacement).has_value());

    auto recreated = device.createSwapchain(desc(*third));
    REQUIRE(recreated.has_value());
    REQUIRE(device.destroySwapchain(*recreated).has_value());
    REQUIRE(device.destroySwapchain(*otherSwapchain).has_value());
    REQUIRE(device.destroySurface(*second).has_value());
    REQUIRE(device.destroySurface(*third).has_value());
    REQUIRE(device.destroySurface(*other).has_value());
}

TEST_CASE("swapchain images render, present, resize, and preserve ownership", "[size:medium][rhi][presentation][warp]")
{
    const HiddenWindow window;
    REQUIRE(window.get() != nullptr);

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

    CHECK_FALSE(device.createSurface({}).has_value());
    auto surface = device.createSurface(rachis::SurfaceDesc{
        .windowHandle = window.get(),
        .label = "Hidden presentation surface",
    });
    REQUIRE(surface.has_value());

    const rachis::SwapchainDesc swapchainDesc{
        .surface = *surface,
        .width = initialWidth,
        .height = initialHeight,
        .imageCount = 3,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource | rachis::TextureUsage::sampled,
        .presentMode = rachis::PresentMode::fifo,
        .label = "Presentation contract swapchain",
    };
    auto invalidSwapchainDesc = swapchainDesc;
    invalidSwapchainDesc.width = 0;
    CHECK_FALSE(device.createSwapchain(invalidSwapchainDesc).has_value());

    auto swapchain = device.createSwapchain(swapchainDesc);
    if (!swapchain)
    {
        INFO(swapchain.error().message);
    }
    REQUIRE(swapchain.has_value());
    CHECK_FALSE(device.createSwapchain(swapchainDesc).has_value());

    const auto vertexBytecode = loadShader(RACHIS_TEST_GRAPHICS_DIRECT_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_GRAPHICS_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto vertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "directVertex",
        .label = "Presentation vertex shader",
        .shaderInterface = {},
    });
    auto pixelShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "pixelMain",
        .label = "Presentation pixel shader",
        .shaderInterface = rachis::test::graphicsPixelShaderInterface(),
    });
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());

    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
    };
    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel},
    };
    auto bindingLayout = device.createBindingLayout({
        .entries = bindingEntries,
        .label = "Presentation sampled-image layout",
    });
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout(rachis::PipelineLayoutDesc{
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstantRanges,
        .label = "Presentation pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::bgra8Unorm},
    };
    auto pipeline = device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
        .layout = *pipelineLayout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .label = "Presentation pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = readbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Presentation readback",
    });
    REQUIRE(readback.has_value());

    auto acquired = device.acquireNextImage(*swapchain);
    REQUIRE(acquired.has_value());
    CHECK(acquired->width == initialWidth);
    CHECK(acquired->height == initialHeight);
    CHECK(acquired->format == rachis::Format::bgra8Unorm);
    CHECK_FALSE(acquired->contentsDefined);
    CHECK(acquired->generation == 1);
    CHECK_FALSE(device.acquireNextImage(*swapchain).has_value());
    CHECK_FALSE(device.resizeSwapchain(*swapchain, resizedWidth, resizedHeight).has_value());
    CHECK_FALSE(device.destroySwapchain(*swapchain).has_value());
    CHECK_FALSE(device.destroyTexture(acquired->texture).has_value());
    CHECK_FALSE(device.destroyTextureView(acquired->renderTargetView).has_value());
    CHECK_FALSE(device.present(*swapchain, {}).has_value());
    auto sampledView = device.createTextureView({
        .texture = acquired->texture,
        .usage = rachis::TextureViewUsage::sampled,
        .label = "Presentation sampled image view",
    });
    REQUIRE(sampledView.has_value());
    const std::array bindingWrites{
        rachis::BindingWrite{.binding = 0, .resource = *sampledView},
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = bindingWrites,
                                                         .label = "Presentation sampled image set",
                                                     });
    REQUIRE(bindingSet.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    CHECK_FALSE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = acquired->texture,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutBefore = rachis::TextureLayout::present,
                                        .layoutAfter = rachis::TextureLayout::common,
                                    })
                    .has_value());
    CHECK_FALSE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = acquired->texture,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::none,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::common,
                                        .layoutAfter = rachis::TextureLayout::present,
                                    })
                    .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = acquired->texture,
                                    .stagesBefore = rachis::PipelineStage::none,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutBefore = rachis::TextureLayout::present,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    CHECK_FALSE(device.cancelSwapchainImage(*swapchain).has_value());
    auto unrelatedCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(unrelatedCommand.has_value());
    auto unrelatedPoint = device.submit(std::move(*unrelatedCommand));
    REQUIRE(unrelatedPoint.has_value());
    CHECK_FALSE(device.present(*swapchain, *unrelatedPoint).has_value());
    const std::array attachments{
        rachis::ColorAttachment{
            .view = acquired->renderTargetView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.05F, 0.1F, 0.2F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindGraphicsBindingSet(*command, 0, *bindingSet).has_value());
    const auto beforeRejectedBinding = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(beforeRejectedBinding.has_value());
    {
        const FailSwapchainLeaseRetention failSwapchainLeasePublication(1);
        auto failedBinding = device.bindGraphicsBindingSet(*command, 0, *bindingSet);
        REQUIRE_FALSE(failedBinding.has_value());
        CHECK(failedBinding.error().code == rachis::ErrorCode::outOfMemory);
    }
    const auto afterRejectedBinding = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(afterRejectedBinding.has_value());
    CHECK(*afterRejectedBinding == *beforeRejectedBinding);
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyTextureView(*sampledView).has_value());
    REQUIRE(device
                .setViewport(*command,
                             rachis::Viewport{
                                 .width = static_cast<float>(initialWidth),
                                 .height = static_cast<float>(initialHeight),
                             })
                .has_value());
    REQUIRE(device
                .setScissor(*command,
                            rachis::ScissorRect{
                                .width = initialWidth,
                                .height = initialHeight,
                            })
                .has_value());
    const std::array tint{0.125F, 0.5F, 0.75F, 1.0F};
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(tint)))
                .has_value());
    REQUIRE(device.draw(*command, 3).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = acquired->texture,
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
                                     rachis::BufferTextureCopy{
                                         .buffer = *readback,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = initialHeight,
                                         .texture = acquired->texture,
                                         .extent = {initialWidth, initialHeight, 1},
                                     })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = acquired->texture,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::none,
                                    .accessBefore = rachis::Access::copySource,
                                    .accessAfter = rachis::Access::none,
                                    .layoutBefore = rachis::TextureLayout::copySource,
                                    .layoutAfter = rachis::TextureLayout::present,
                                })
                .has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    CHECK_FALSE(device.cancelSwapchainImage(*swapchain).has_value());
    auto wrongPresentCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(wrongPresentCommand.has_value());
    auto wrongPresentPoint = device.submit(std::move(*wrongPresentCommand));
    REQUIRE(wrongPresentPoint.has_value());
    auto wrongPresent = device.present(*swapchain, *wrongPresentPoint);
    REQUIRE_FALSE(wrongPresent.has_value());
    CHECK(wrongPresent.error().code == rachis::ErrorCode::invalidArgument);
    auto presented = device.present(*swapchain, *submitted);
    REQUIRE(presented.has_value());

    REQUIRE(device.resizeSwapchain(*swapchain, resizedWidth, resizedHeight).has_value());
    auto resized = device.acquireNextImage(*swapchain);
    REQUIRE(resized.has_value());
    CHECK(resized->width == resizedWidth);
    CHECK(resized->height == resizedHeight);
    CHECK_FALSE(resized->contentsDefined);
    CHECK(resized->generation == 2);
    CHECK(resized->texture != acquired->texture);
    CHECK(resized->renderTargetView != acquired->renderTargetView);

    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());
    CHECK_FALSE(device.cancelSwapchainImage(*swapchain).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    const auto* corner = mapped->data;
    const auto* center = mapped->data + static_cast<std::uint64_t>(initialHeight / 2) * rowPitch +
                         static_cast<std::uint64_t>(initialWidth / 2) * 4;
    CHECK(nearByte(corner[0], 51));
    CHECK(nearByte(corner[1], 26));
    CHECK(nearByte(corner[2], 13));
    CHECK(nearByte(corner[3], 255));
    CHECK(nearByte(center[0], 191));
    CHECK(nearByte(center[1], 128));
    CHECK(nearByte(center[2], 32));
    CHECK(nearByte(center[3], 255));
    REQUIRE(device.unmapBuffer(*readback).has_value());

    const auto staleSwapchain = *swapchain;
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK_FALSE(device.acquireNextImage(staleSwapchain).has_value());
    CHECK_FALSE(device.destroyTexture(acquired->texture).has_value());
    CHECK_FALSE(device.destroyTextureView(acquired->renderTargetView).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("swapchain cancellation resumes after command reset or abandonment",
          "[size:medium][rhi][presentation][lifetime][warp]")
{
    const HiddenWindow window;
    REQUIRE(window.get() != nullptr);

    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::disabled,
        .enableDred = false,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    auto surface = device.createSurface(rachis::SurfaceDesc{.windowHandle = window.get()});
    REQUIRE(surface.has_value());
    auto swapchain = device.createSwapchain(rachis::SwapchainDesc{
        .surface = *surface,
        .width = initialWidth,
        .height = initialHeight,
        .imageCount = 2,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
    });
    REQUIRE(swapchain.has_value());

    auto resetImage = device.acquireNextImage(*swapchain);
    REQUIRE(resetImage.has_value());
    auto resetPool = device.createCommandPool(rachis::CommandPoolDesc{
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Presentation reset pool",
        .flags = rachis::CommandPoolFlag::allowIndividualReset,
    });
    REQUIRE(resetPool.has_value());
    {
        auto resetCommand = device.acquireCommandBuffer(*resetPool);
        REQUIRE(resetCommand.has_value());
        REQUIRE(device
                    .textureBarrier(*resetCommand,
                                    rachis::TextureBarrier{
                                        .texture = resetImage->texture,
                                        .stagesBefore = rachis::PipelineStage::none,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutBefore = rachis::TextureLayout::present,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
        CHECK_FALSE(device.cancelSwapchainImage(*swapchain).has_value());
        REQUIRE(device.resetCommandBuffer(*resetCommand).has_value());
        REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());
    }
    REQUIRE(device.destroyCommandPool(*resetPool).has_value());

    auto abandonedImage = device.acquireNextImage(*swapchain);
    REQUIRE(abandonedImage.has_value());
    {
        auto abandonedCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(abandonedCommand.has_value());
        REQUIRE(device
                    .textureBarrier(*abandonedCommand,
                                    rachis::TextureBarrier{
                                        .texture = abandonedImage->texture,
                                        .stagesBefore = rachis::PipelineStage::none,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutBefore = rachis::TextureLayout::present,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
        CHECK_FALSE(device.cancelSwapchainImage(*swapchain).has_value());
    }
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
}

TEST_CASE("failed barrier lease publication rolls back every swapchain reference",
          "[size:medium][rhi][presentation][barrier][failure][warp]")
{
    const bool legacyBarriers = GENERATE(false, true);
    const ForceLegacyBarriers barrierMode(legacyBarriers);
    INFO("Barrier implementation: " << (legacyBarriers ? "legacy" : "enhanced"));

    const HiddenWindow firstWindow;
    const HiddenWindow secondWindow;
    REQUIRE(firstWindow.get() != nullptr);
    REQUIRE(secondWindow.get() != nullptr);

    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::disabled,
        .enableDred = false,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    auto firstSurface = device.createSurface(rachis::SurfaceDesc{.windowHandle = firstWindow.get()});
    auto secondSurface = device.createSurface(rachis::SurfaceDesc{.windowHandle = secondWindow.get()});
    REQUIRE(firstSurface.has_value());
    REQUIRE(secondSurface.has_value());
    const auto createSwapchain = [&](rachis::SurfaceHandle surface) {
        return device.createSwapchain(rachis::SwapchainDesc{
            .surface = surface,
            .width = initialWidth,
            .height = initialHeight,
            .imageCount = 2,
            .format = rachis::Format::bgra8Unorm,
            .usage = rachis::TextureUsage::renderTarget,
        });
    };
    auto firstSwapchain = createSwapchain(*firstSurface);
    auto secondSwapchain = createSwapchain(*secondSurface);
    REQUIRE(firstSwapchain.has_value());
    REQUIRE(secondSwapchain.has_value());

    auto firstImage = device.acquireNextImage(*firstSwapchain);
    auto secondImage = device.acquireNextImage(*secondSwapchain);
    REQUIRE(firstImage.has_value());
    REQUIRE(secondImage.has_value());
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    const std::array barriers{
        rachis::TextureBarrier{
            .texture = firstImage->texture,
            .stagesBefore = rachis::PipelineStage::none,
            .stagesAfter = rachis::PipelineStage::renderTarget,
            .accessBefore = rachis::Access::none,
            .accessAfter = rachis::Access::renderTarget,
            .layoutBefore = rachis::TextureLayout::present,
            .layoutAfter = rachis::TextureLayout::renderTarget,
        },
        rachis::TextureBarrier{
            .texture = secondImage->texture,
            .stagesBefore = rachis::PipelineStage::none,
            .stagesAfter = rachis::PipelineStage::renderTarget,
            .accessBefore = rachis::Access::none,
            .accessAfter = rachis::Access::renderTarget,
            .layoutBefore = rachis::TextureLayout::present,
            .layoutAfter = rachis::TextureLayout::renderTarget,
        },
    };
    const auto beforeFailure = rachis::d3d12::test::commandDependencySnapshot(*command);
    const auto beforeRecording = rachis::d3d12::test::resourceCommandRecordCount().load(std::memory_order_relaxed);
    REQUIRE(beforeFailure.has_value());
    {
        const FailSwapchainLeaseRetention failSecondLease(2);
        auto failed = device.resourceBarriers(*command, {.textureBarriers = barriers});
        REQUIRE_FALSE(failed.has_value());
        CHECK(failed.error().code == rachis::ErrorCode::outOfMemory);
    }
    const auto afterFailure = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(afterFailure.has_value());
    CHECK(*afterFailure == *beforeFailure);
    CHECK(rachis::d3d12::test::resourceCommandRecordCount().load(std::memory_order_relaxed) == beforeRecording);
    REQUIRE(device.cancelSwapchainImage(*firstSwapchain).has_value());
    REQUIRE(device.cancelSwapchainImage(*secondSwapchain).has_value());
    REQUIRE(device.destroySwapchain(*firstSwapchain).has_value());
    REQUIRE(device.destroySwapchain(*secondSwapchain).has_value());
    REQUIRE(device.destroySurface(*firstSurface).has_value());
    REQUIRE(device.destroySurface(*secondSurface).has_value());
}

TEST_CASE("failed resolve lease publication restores prior command dependencies",
          "[size:medium][rhi][presentation][resolve][failure][warp]")
{
    const HiddenWindow window;
    REQUIRE(window.get() != nullptr);

    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    const auto formatProperties = device.queryFormatProperties(rachis::Format::bgra8Unorm);
    const auto sampleCount = preferredSampleCount(formatProperties.sampleCounts);
    REQUIRE(rachis::hasAny(formatProperties.features, rachis::FormatFeature::multisampleResolve));
    REQUIRE(sampleCount != 0);

    auto surface = device.createSurface({.windowHandle = window.get()});
    REQUIRE(surface.has_value());
    auto swapchain = device.createSwapchain({
        .surface = *surface,
        .width = initialWidth,
        .height = initialHeight,
        .imageCount = 2,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .presentMode = rachis::PresentMode::fifo,
        .label = "Resolve lease transaction swapchain",
    });
    REQUIRE(swapchain.has_value());
    auto acquired = device.acquireNextImage(*swapchain);
    REQUIRE(acquired.has_value());

    auto source = device.createTexture({
        .width = initialWidth,
        .height = initialHeight,
        .sampleCount = sampleCount,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "Resolve lease transaction source",
    });
    REQUIRE(source.has_value());
    auto sourceView = device.createTextureView({
        .texture = *source,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(sourceView.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *source,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array attachments{
        rachis::ColorAttachment{
            .view = *sourceView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.25F, 0.5F, 0.75F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    const std::array resolveBarriers{
        rachis::TextureBarrier{
            .texture = *source,
            .stagesBefore = rachis::PipelineStage::renderTarget,
            .stagesAfter = rachis::PipelineStage::resolve,
            .accessBefore = rachis::Access::renderTarget,
            .accessAfter = rachis::Access::resolveSource,
            .layoutBefore = rachis::TextureLayout::renderTarget,
            .layoutAfter = rachis::TextureLayout::resolveSource,
        },
        rachis::TextureBarrier{
            .texture = acquired->texture,
            .stagesBefore = acquired->initialStages,
            .stagesAfter = rachis::PipelineStage::resolve,
            .accessBefore = acquired->initialAccess,
            .accessAfter = rachis::Access::resolveDestination,
            .layoutBefore = acquired->initialLayout,
            .layoutAfter = rachis::TextureLayout::resolveDestination,
        },
    };
    REQUIRE(device.resourceBarriers(*command, {.textureBarriers = resolveBarriers}).has_value());

    const auto beforeFailure = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(beforeFailure.has_value());
    {
        const FailSwapchainLeaseRetention failDestinationLease(1);
        auto failed = device.resolveTexture(*command, {.source = *source, .destination = acquired->texture});
        REQUIRE_FALSE(failed.has_value());
        CHECK(failed.error().code == rachis::ErrorCode::outOfMemory);
    }
    const auto afterFailure = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(afterFailure.has_value());
    CHECK(*afterFailure == *beforeFailure);

    REQUIRE(device.resolveTexture(*command, {.source = *source, .destination = acquired->texture}).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = acquired->texture,
                                    .stagesBefore = rachis::PipelineStage::resolve,
                                    .stagesAfter = acquired->requiredFinalStages,
                                    .accessBefore = rachis::Access::resolveDestination,
                                    .accessAfter = acquired->requiredFinalAccess,
                                    .layoutBefore = rachis::TextureLayout::resolveDestination,
                                    .layoutAfter = acquired->requiredFinalLayout,
                                })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.present(*swapchain, *submitted).has_value());
    REQUIRE(device.wait(*submitted).has_value());

    REQUIRE(device.destroyTextureView(*sourceView).has_value());
    REQUIRE(device.destroyTexture(*source).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("failed render-begin dependency publication leaves timestamps and leases unpublished",
          "[size:medium][rhi][presentation][rendering][timestamp][failure][warp]")
{
    const HiddenWindow window;
    REQUIRE(window.get() != nullptr);

    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::disabled,
        .enableDred = false,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    auto surface = device.createSurface(rachis::SurfaceDesc{.windowHandle = window.get()});
    REQUIRE(surface.has_value());
    auto swapchain = device.createSwapchain(rachis::SwapchainDesc{
        .surface = *surface,
        .width = initialWidth,
        .height = initialHeight,
        .imageCount = 2,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
    });
    REQUIRE(swapchain.has_value());
    auto pool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 2,
        .label = "Transactional render-begin timestamps",
    });
    REQUIRE(pool.has_value());
    constexpr std::array priorPayload{
        std::byte{0x52},
        std::byte{0x58},
        std::byte{0x31},
        std::byte{0x32},
    };
    auto priorSource = device.createBuffer(rachis::BufferDesc{
        .size = priorPayload.size(),
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Pre-failure command source",
    });
    auto priorDestination = device.createBuffer(rachis::BufferDesc{
        .size = priorPayload.size(),
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Pre-failure command destination",
    });
    REQUIRE(priorSource.has_value());
    REQUIRE(priorDestination.has_value());
    REQUIRE(device.writeBuffer(*priorSource, 0, priorPayload).has_value());
    auto acquired = device.acquireNextImage(*swapchain);
    REQUIRE(acquired.has_value());
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *priorSource,
                                .destination = *priorDestination,
                                .size = priorPayload.size(),
                            })
                .has_value());
    const std::array attachments{
        rachis::ColorAttachment{
            .view = acquired->renderTargetView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.125F, 0.25F, 0.5F, 1.0F},
        },
    };
    const rachis::RenderingDesc rendering{
        .colorAttachments = attachments,
        .timestampWrites =
            rachis::PassTimestampWrites{
                .queryPool = *pool,
                .beginningOfPassWriteIndex = 0,
                .endOfPassWriteIndex = 1,
            },
    };

    const FailSwapchainLeaseRetention failLease(1);
    auto failed = device.beginRendering(*command, rendering);
    REQUIRE_FALSE(failed.has_value());
    CHECK(failed.error().code == rachis::ErrorCode::outOfMemory);
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());

    acquired = device.acquireNextImage(*swapchain);
    REQUIRE(acquired.has_value());
    const std::array retryAttachments{
        rachis::ColorAttachment{
            .view = acquired->renderTargetView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.125F, 0.25F, 0.5F, 1.0F},
        },
    };
    const rachis::RenderingDesc retryRendering{
        .colorAttachments = retryAttachments,
        .timestampWrites = rendering.timestampWrites,
    };

    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = acquired->texture,
                                    .stagesBefore = rachis::PipelineStage::none,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutBefore = rachis::TextureLayout::present,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    REQUIRE(device.beginRendering(*command, retryRendering).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = acquired->texture,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = rachis::PipelineStage::none,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = rachis::Access::none,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::present,
                                })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.present(*swapchain, *submitted).has_value());
    REQUIRE(device.wait(*submitted).has_value());
    std::array<std::byte, priorPayload.size()> observedPayload{};
    REQUIRE(device.readBuffer(*priorDestination, 0, observedPayload).has_value());
    CHECK(observedPayload == priorPayload);

    REQUIRE(device.destroyQueryPool(*pool).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    REQUIRE(device.destroyBuffer(*priorSource).has_value());
    REQUIRE(device.destroyBuffer(*priorDestination).has_value());
}

TEST_CASE("legacy swapchain copy transitions return to the present endpoint",
          "[size:medium][rhi][presentation][legacy][warp]")
{
    const ForceLegacyBarriers forceLegacy;
    const HiddenWindow window;
    REQUIRE(window.get() != nullptr);

    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    CHECK_FALSE(rachis::test::requireCapabilities(device).enhancedBarriers);

    auto surface = device.createSurface(rachis::SurfaceDesc{.windowHandle = window.get()});
    REQUIRE(surface.has_value());
    auto swapchain = device.createSwapchain(rachis::SwapchainDesc{
        .surface = *surface,
        .width = initialWidth,
        .height = initialHeight,
        .imageCount = 2,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copyDestination,
        .presentMode = rachis::PresentMode::fifo,
    });
    REQUIRE(swapchain.has_value());

    auto upload = device.createBuffer(rachis::BufferDesc{
        .size = readbackSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(upload.has_value());
    auto mapped = device.mapBuffer(*upload);
    REQUIRE(mapped.has_value());
    std::fill_n(mapped->data, mapped->size, std::byte{0x3C});
    REQUIRE(device.unmapBuffer(*upload).has_value());

    auto acquired = device.acquireNextImage(*swapchain);
    REQUIRE(acquired.has_value());
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = acquired->texture,
                                    .stagesBefore = rachis::PipelineStage::none,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::present,
                                    .layoutAfter = rachis::TextureLayout::common,
                                })
                .has_value());
    REQUIRE(device
                .copyBufferToTexture(*command,
                                     rachis::BufferTextureCopy{
                                         .buffer = *upload,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = initialHeight,
                                         .texture = acquired->texture,
                                         .extent = {initialWidth, initialHeight, 1},
                                     })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = acquired->texture,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::none,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::none,
                                    .layoutBefore = rachis::TextureLayout::common,
                                    .layoutAfter = rachis::TextureLayout::present,
                                })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.present(*swapchain, *submitted).has_value());

    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("a present endpoint failure poisons the swapchain until destruction",
          "[size:medium][rhi][presentation][failure][warp]")
{
    const ForceLegacyBarriers forceLegacy;
    const HiddenWindow window;
    REQUIRE(window.get() != nullptr);

    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::disabled,
        .enableDred = false,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    auto surface = device.createSurface(rachis::SurfaceDesc{.windowHandle = window.get()});
    REQUIRE(surface.has_value());
    auto swapchain = device.createSwapchain(rachis::SwapchainDesc{
        .surface = *surface,
        .width = initialWidth,
        .height = initialHeight,
        .imageCount = 2,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copyDestination,
    });
    REQUIRE(swapchain.has_value());

    auto acquired = device.acquireNextImage(*swapchain);
    REQUIRE(acquired.has_value());
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = acquired->texture,
                                    .stagesBefore = rachis::PipelineStage::none,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::present,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    const auto failSignalReservation = GENERATE(false, true);
    if (failSignalReservation)
    {
        rachis::d3d12::test::nextQueueSignalReservationFailure().store(true, std::memory_order_relaxed);
    }
    else
    {
        rachis::d3d12::test::nextPresentResult().store(static_cast<std::int32_t>(DXGI_ERROR_INVALID_CALL),
                                                       std::memory_order_relaxed);
    }
    auto failedPresent = device.present(*swapchain, *submitted);
    REQUIRE_FALSE(failedPresent.has_value());
    CHECK(failedPresent.error().code ==
          (failSignalReservation ? rachis::ErrorCode::outOfMemory : rachis::ErrorCode::nativeFailure));
    CHECK_FALSE(device.acquireNextImage(*swapchain).has_value());
    CHECK_FALSE(device.resizeSwapchain(*swapchain, resizedWidth, resizedHeight).has_value());
    CHECK_FALSE(device.present(*swapchain, *submitted).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
}

TEST_CASE("device loss is a monotonic device-wide terminal state with cached diagnostics",
          "[size:medium][rhi][presentation][device-loss][dred][warp]")
{
    const ForceLegacyBarriers forceLegacy;
    const HiddenWindow window;
    const HiddenWindow abandonedWindow;
    REQUIRE(window.get() != nullptr);
    REQUIRE(abandonedWindow.get() != nullptr);

    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::disabled,
        .enableDred = true,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    CHECK(device.operationalState() == rachis::DeviceOperationalState::active);
    CHECK_FALSE(device.deviceLossError().has_value());

    auto surface = device.createSurface(rachis::SurfaceDesc{.windowHandle = window.get()});
    REQUIRE(surface.has_value());
    auto swapchain = device.createSwapchain(rachis::SwapchainDesc{
        .surface = *surface,
        .width = initialWidth,
        .height = initialHeight,
        .imageCount = 2,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copyDestination,
    });
    REQUIRE(swapchain.has_value());

    auto abandonedSurface = device.createSurface(rachis::SurfaceDesc{.windowHandle = abandonedWindow.get()});
    REQUIRE(abandonedSurface.has_value());
    auto abandonedSwapchain = device.createSwapchain(rachis::SwapchainDesc{
        .surface = *abandonedSurface,
        .width = initialWidth,
        .height = initialHeight,
        .imageCount = 2,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
    });
    REQUIRE(abandonedSwapchain.has_value());
    auto abandonedImage = device.acquireNextImage(*abandonedSwapchain);
    REQUIRE(abandonedImage.has_value());
    auto abandonedView = device.createTextureView({
        .texture = abandonedImage->texture,
        .usage = rachis::TextureViewUsage::renderTarget,
        .label = "Application view of an abandoned swapchain image",
    });
    REQUIRE(abandonedView.has_value());

    auto acquired = device.acquireNextImage(*swapchain);
    REQUIRE(acquired.has_value());
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = acquired->texture,
                                    .stagesBefore = rachis::PipelineStage::none,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::present,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = acquired->texture,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::none,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::none,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::present,
                                })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());

    rachis::d3d12::test::nextPresentResult().store(static_cast<std::int32_t>(DXGI_ERROR_DEVICE_REMOVED),
                                                   std::memory_order_relaxed);
    auto failedPresent = device.present(*swapchain, *submitted);
    REQUIRE_FALSE(failedPresent.has_value());
    CHECK(failedPresent.error().code == rachis::ErrorCode::deviceLost);
    CHECK(failedPresent.error().nativeCode == static_cast<std::int64_t>(DXGI_ERROR_DEVICE_REMOVED));
    CHECK(device.operationalState() == rachis::DeviceOperationalState::lost);

    const auto firstLoss = device.deviceLossError();
    if (!firstLoss.has_value())
    {
        FAIL("device loss must publish its first terminal error");
        return;
    }
    const auto& firstLossError = firstLoss.value();
    CHECK(firstLossError.code == rachis::ErrorCode::deviceLost);
    CHECK(firstLossError.nativeCode == failedPresent.error().nativeCode);
    CHECK(firstLossError.message == failedPresent.error().message);
    const auto checkExactLoss = [&](const rachis::Error& error) {
        CHECK(error.code == firstLossError.code);
        CHECK(error.nativeCode == firstLossError.nativeCode);
        CHECK(error.message == firstLossError.message);
    };

    auto idle = device.waitIdle();
    REQUIRE_FALSE(idle.has_value());
    checkExactLoss(idle.error());
    auto reacquired = device.acquireNextImage(*swapchain);
    REQUIRE_FALSE(reacquired.has_value());
    checkExactLoss(reacquired.error());
    auto resized = device.resizeSwapchain(*swapchain, resizedWidth, resizedHeight);
    REQUIRE_FALSE(resized.has_value());
    checkExactLoss(resized.error());
    auto unpublishedSurface = device.createSurface(rachis::SurfaceDesc{.windowHandle = window.get()});
    REQUIRE_FALSE(unpublishedSurface.has_value());
    checkExactLoss(unpublishedSurface.error());
    auto unpublishedPool = device.createCommandPool(rachis::CommandPoolDesc{
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
    });
    REQUIRE_FALSE(unpublishedPool.has_value());
    checkExactLoss(unpublishedPool.error());

    auto report = device.captureDeviceFaultReport();
    REQUIRE(report.has_value());
    CHECK(report->status == rachis::DeviceFaultStatus::removed);
    CHECK(report->nativeReason == static_cast<std::int64_t>(DXGI_ERROR_DEVICE_REMOVED));
    auto repeatedReport = device.captureDeviceFaultReport();
    REQUIRE(repeatedReport.has_value());
    CHECK(repeatedReport->status == report->status);
    CHECK(repeatedReport->nativeReason == report->nativeReason);
    CHECK(repeatedReport->breadcrumbs.size() == report->breadcrumbs.size());
    CHECK(repeatedReport->allocations.size() == report->allocations.size());

    REQUIRE(device.destroySwapchain(*swapchain).has_value());

    REQUIRE(device.destroySwapchain(*abandonedSwapchain).has_value());
    REQUIRE(device.destroyTextureView(*abandonedView).has_value());
    auto abandonedViewAgain = device.destroyTextureView(*abandonedView);
    REQUIRE_FALSE(abandonedViewAgain.has_value());
    CHECK(abandonedViewAgain.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device.destroySurface(*surface).has_value());
    REQUIRE(device.destroySurface(*abandonedSurface).has_value());
    CHECK(device.operationalState() == rachis::DeviceOperationalState::lost);
}

TEST_CASE("device removal seen in swapchain acquire, resize or destroy is recorded as device loss",
          "[size:medium][rhi][presentation][device-loss][warp]")
{
    enum class LossSite
    {
        acquireWait,
        resizeWait,
        resizeBuffers,
        destroyWait,
    };
    const auto site =
        GENERATE(LossSite::acquireWait, LossSite::resizeWait, LossSite::resizeBuffers, LossSite::destroyWait);
    CAPTURE(static_cast<int>(site));

    const HiddenWindow window;
    REQUIRE(window.get() != nullptr);
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::disabled,
        .enableDred = false,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    auto surface = device.createSurface(rachis::SurfaceDesc{.windowHandle = window.get()});
    REQUIRE(surface.has_value());
    auto swapchain = device.createSwapchain(rachis::SwapchainDesc{
        .surface = *surface,
        .width = initialWidth,
        .height = initialHeight,
        .imageCount = 2,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
    });
    REQUIRE(swapchain.has_value());
    REQUIRE(device.operationalState() == rachis::DeviceOperationalState::active);

    bool destroyed = false;
    switch (site)
    {
    case LossSite::acquireWait: {
        rachis::d3d12::test::nextSwapchainWaitDeviceLoss().store(true, std::memory_order_relaxed);
        auto acquired = device.acquireNextImage(*swapchain);
        REQUIRE_FALSE(acquired.has_value());
        CHECK(acquired.error().code == rachis::ErrorCode::deviceLost);
        break;
    }
    case LossSite::resizeWait: {
        rachis::d3d12::test::nextSwapchainWaitDeviceLoss().store(true, std::memory_order_relaxed);
        auto resized = device.resizeSwapchain(*swapchain, resizedWidth, resizedHeight);
        REQUIRE_FALSE(resized.has_value());
        CHECK(resized.error().code == rachis::ErrorCode::deviceLost);
        break;
    }
    case LossSite::resizeBuffers: {
        rachis::d3d12::test::nextResizeBuffersResult().store(static_cast<std::int32_t>(DXGI_ERROR_DEVICE_RESET),
                                                             std::memory_order_relaxed);
        auto resized = device.resizeSwapchain(*swapchain, resizedWidth, resizedHeight);
        REQUIRE_FALSE(resized.has_value());
        CHECK(resized.error().code == rachis::ErrorCode::deviceLost);
        CHECK(resized.error().nativeCode == static_cast<std::int64_t>(DXGI_ERROR_DEVICE_RESET));
        break;
    }
    case LossSite::destroyWait: {
        rachis::d3d12::test::nextSwapchainWaitDeviceLoss().store(true, std::memory_order_relaxed);
        REQUIRE(device.destroySwapchain(*swapchain).has_value());
        destroyed = true;
        break;
    }
    }
    CHECK_FALSE(rachis::d3d12::test::nextSwapchainWaitDeviceLoss().exchange(false, std::memory_order_relaxed));
    CHECK(rachis::d3d12::test::nextResizeBuffersResult().exchange(0, std::memory_order_relaxed) == 0);

    CHECK(device.operationalState() == rachis::DeviceOperationalState::lost);
    const auto loss = device.deviceLossError();
    REQUIRE(loss.has_value());
    CHECK(loss->code == rachis::ErrorCode::deviceLost);

    auto idle = device.waitIdle();
    REQUIRE_FALSE(idle.has_value());
    CHECK(idle.error().code == rachis::ErrorCode::deviceLost);
    CHECK(idle.error().message == loss->message);
    auto unpublishedSurface = device.createSurface(rachis::SurfaceDesc{.windowHandle = window.get()});
    REQUIRE_FALSE(unpublishedSurface.has_value());
    CHECK(unpublishedSurface.error().code == rachis::ErrorCode::deviceLost);
    if (!destroyed)
    {
        auto reacquired = device.acquireNextImage(*swapchain);
        REQUIRE_FALSE(reacquired.has_value());
        CHECK(reacquired.error().code == rachis::ErrorCode::deviceLost);
        auto resized = device.resizeSwapchain(*swapchain, resizedWidth, resizedHeight);
        REQUIRE_FALSE(resized.has_value());
        CHECK(resized.error().code == rachis::ErrorCode::deviceLost);
        REQUIRE(device.destroySwapchain(*swapchain).has_value());
    }
    else
    {
        CHECK_FALSE(device.destroySwapchain(*swapchain).has_value());
    }
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.operationalState() == rachis::DeviceOperationalState::lost);
}

TEST_CASE("every DXGI device-loss HRESULT from present or ResizeBuffers marks the D3D12 device lost",
          "[size:medium][rhi][presentation][device-loss][warp]")
{
    const auto failure = GENERATE(
        DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_HUNG, DXGI_ERROR_DEVICE_RESET, DXGI_ERROR_DRIVER_INTERNAL_ERROR);
    const bool duringPresent = GENERATE(false, true);
    CAPTURE(static_cast<std::uint32_t>(failure), duringPresent);

    const HiddenWindow window;
    REQUIRE(window.get() != nullptr);
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::disabled,
        .enableDred = false,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    auto surface = device.createSurface(rachis::SurfaceDesc{.windowHandle = window.get()});
    REQUIRE(surface.has_value());
    auto swapchain = device.createSwapchain(copyableSwapchainDesc(*surface, initialWidth, initialHeight));
    REQUIRE(swapchain.has_value());

    rachis::Status failed;
    if (duringPresent)
    {
        auto acquired = device.acquireNextImage(*swapchain);
        REQUIRE(acquired.has_value());
        auto submitted = submitPresentableFrame(device, acquired->texture);
        REQUIRE(submitted.has_value());
        rachis::d3d12::test::nextPresentResult().store(static_cast<std::int32_t>(failure), std::memory_order_relaxed);
        auto presented = device.present(*swapchain, *submitted);
        REQUIRE_FALSE(presented.has_value());
        failed = std::unexpected(std::move(presented.error()));
    }
    else
    {
        rachis::d3d12::test::nextResizeBuffersResult().store(static_cast<std::int32_t>(failure),
                                                             std::memory_order_relaxed);
        failed = device.resizeSwapchain(*swapchain, resizedWidth, resizedHeight);
    }
    CHECK(rachis::d3d12::test::nextPresentResult().exchange(0, std::memory_order_relaxed) == 0);
    CHECK(rachis::d3d12::test::nextResizeBuffersResult().exchange(0, std::memory_order_relaxed) == 0);
    REQUIRE_FALSE(failed.has_value());
    CHECK(failed.error().code == rachis::ErrorCode::deviceLost);
    CHECK(failed.error().nativeCode == static_cast<std::int64_t>(failure));
    CHECK(device.operationalState() == rachis::DeviceOperationalState::lost);
    const auto loss = device.deviceLossError();
    REQUIRE(loss.has_value());
    CHECK(loss->nativeCode == static_cast<std::int64_t>(failure));

    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
}

TEST_CASE("a failed back-buffer rebuild during resize poisons the D3D12 swapchain",
          "[size:medium][rhi][presentation][swapchain][warp]")
{
    const bool resizeBuffersFails = GENERATE(false, true);
    CAPTURE(resizeBuffersFails);

    const HiddenWindow window;
    REQUIRE(window.get() != nullptr);
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::disabled,
        .enableDred = false,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    auto surface = device.createSurface(rachis::SurfaceDesc{.windowHandle = window.get()});
    REQUIRE(surface.has_value());
    auto swapchain = device.createSwapchain(copyableSwapchainDesc(*surface, initialWidth, initialHeight));
    REQUIRE(swapchain.has_value());

    if (resizeBuffersFails)
    {
        rachis::d3d12::test::nextResizeBuffersResult().store(static_cast<std::int32_t>(DXGI_ERROR_INVALID_CALL),
                                                             std::memory_order_relaxed);
    }
    rachis::d3d12::test::nextSwapchainGetBufferResult().store(static_cast<std::int32_t>(E_FAIL),
                                                              std::memory_order_relaxed);
    auto resized = device.resizeSwapchain(*swapchain, resizedWidth, resizedHeight);
    CHECK(rachis::d3d12::test::nextResizeBuffersResult().exchange(0, std::memory_order_relaxed) == 0);
    CHECK(rachis::d3d12::test::nextSwapchainGetBufferResult().exchange(0, std::memory_order_relaxed) == 0);
    REQUIRE_FALSE(resized.has_value());
    CHECK(resized.error().code == rachis::ErrorCode::nativeFailure);
    CHECK(resized.error().nativeCode == static_cast<std::int64_t>(E_FAIL));
    CHECK(device.operationalState() == rachis::DeviceOperationalState::active);

    auto presentation = device.querySwapchainPresentation(*swapchain);
    REQUIRE(presentation.has_value());
    CHECK(presentation->imageCount == 0);
    auto acquired = device.acquireNextImage(*swapchain);
    REQUIRE_FALSE(acquired.has_value());
    CHECK(acquired.error().code == rachis::ErrorCode::invalidState);
    CHECK(acquired.error().message.contains("destroyed and recreated"));
    auto retried = device.resizeSwapchain(*swapchain, resizedWidth, resizedHeight);
    REQUIRE_FALSE(retried.has_value());
    CHECK(retried.error().code == rachis::ErrorCode::invalidState);
    CHECK(retried.error().message.contains("destroyed and recreated"));
    REQUIRE(device.destroySwapchain(*swapchain).has_value());

    auto replacement = device.createSwapchain(copyableSwapchainDesc(*surface, resizedWidth, resizedHeight));
    REQUIRE(replacement.has_value());
    auto image = device.acquireNextImage(*replacement);
    REQUIRE(image.has_value());
    CHECK(image->width == resizedWidth);
    CHECK(image->height == resizedHeight);
    REQUIRE(device.cancelSwapchainImage(*replacement).has_value());
    REQUIRE(device.destroySwapchain(*replacement).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.operationalState() == rachis::DeviceOperationalState::active);
}

TEST_CASE("resizing a D3D12 swapchain fails with invalid state while views of its images are referenced",
          "[size:medium][rhi][presentation][swapchain][warp]")
{
    const bool heldByBindingSet = GENERATE(false, true);
    CAPTURE(heldByBindingSet);

    const HiddenWindow window;
    REQUIRE(window.get() != nullptr);
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    auto surface = device.createSurface(rachis::SurfaceDesc{.windowHandle = window.get()});
    REQUIRE(surface.has_value());
    auto swapchainDesc = copyableSwapchainDesc(*surface, initialWidth, initialHeight);
    swapchainDesc.usage = swapchainDesc.usage | rachis::TextureUsage::sampled;
    auto swapchain = device.createSwapchain(swapchainDesc);
    REQUIRE(swapchain.has_value());
    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel},
    };
    auto bindingLayout = device.createBindingLayout({
        .entries = bindingEntries,
        .label = "Resize-blocking binding layout",
    });
    REQUIRE(bindingLayout.has_value());

    auto viewed = device.acquireNextImage(*swapchain);
    REQUIRE(viewed.has_value());
    auto sampledView = device.createTextureView({
        .texture = viewed->texture,
        .usage = rachis::TextureViewUsage::sampled,
        .label = "Resize-blocking sampled view",
    });
    REQUIRE(sampledView.has_value());
    rachis::BindingSetHandle bindingSet;
    if (heldByBindingSet)
    {
        const std::array bindingWrites{
            rachis::BindingWrite{.binding = 0, .resource = *sampledView},
        };
        auto createdSet = rachis::test::createBindingSet(device,
                                                         {
                                                             .layout = *bindingLayout,
                                                             .writes = bindingWrites,
                                                             .label = "Resize-blocking binding set",
                                                         });
        REQUIRE(createdSet.has_value());
        bindingSet = *createdSet;
        REQUIRE(device.destroyTextureView(*sampledView).has_value());
    }
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());

    auto rejected = device.resizeSwapchain(*swapchain, resizedWidth, resizedHeight);
    REQUIRE_FALSE(rejected.has_value());
    INFO(rejected.error().message);
    CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
    CHECK(rejected.error().message.contains("Destroy views of swapchain images"));
    CHECK(device.operationalState() == rachis::DeviceOperationalState::active);

    auto presentation = device.querySwapchainPresentation(*swapchain);
    REQUIRE(presentation.has_value());
    CHECK(presentation->imageCount == swapchainDesc.imageCount);
    CHECK(presentation->width == initialWidth);
    CHECK(presentation->height == initialHeight);
    auto acquired = device.acquireNextImage(*swapchain);
    REQUIRE(acquired.has_value());
    CHECK(acquired->width == initialWidth);
    CHECK(acquired->height == initialHeight);
    CHECK(acquired->generation == viewed->generation);
    auto submitted = submitPresentableFrame(device, acquired->texture);
    REQUIRE(submitted.has_value());
    REQUIRE(device.present(*swapchain, *submitted).has_value());

    if (heldByBindingSet)
    {
        REQUIRE(device.destroyBindingSet(bindingSet).has_value());
    }
    else
    {
        REQUIRE(device.destroyTextureView(*sampledView).has_value());
    }
    REQUIRE(device.resizeSwapchain(*swapchain, resizedWidth, resizedHeight).has_value());
    auto resized = device.acquireNextImage(*swapchain);
    REQUIRE(resized.has_value());
    CHECK(resized->width == resizedWidth);
    CHECK(resized->height == resizedHeight);
    CHECK(resized->generation == viewed->generation + 1);
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());

    const auto counts = device.diagnosticCounts();
    CHECK(counts.errors == 0);
    CHECK(counts.corruptions == 0);
}

TEST_CASE("destroying a D3D12 device releases live swapchain back buffers before the native swapchain",
          "[size:medium][rhi][presentation][swapchain][warp]")
{
    const HiddenWindow window;
    REQUIRE(window.get() != nullptr);
    const rachis::DeviceDesc deviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::disabled,
        .enableDred = false,
    };
    const auto releasedWhileReferenced =
        rachis::d3d12::test::nativeSwapchainsReleasedWhileReferenced().load(std::memory_order_relaxed);
    {
        auto created = rachis::createDevice(deviceDesc);
        REQUIRE(created.has_value());
        auto device = std::move(*created);
        auto surface = device.createSurface(rachis::SurfaceDesc{.windowHandle = window.get()});
        REQUIRE(surface.has_value());
        auto swapchain = device.createSwapchain(copyableSwapchainDesc(*surface, initialWidth, initialHeight));
        REQUIRE(swapchain.has_value());
        auto acquired = device.acquireNextImage(*swapchain);
        REQUIRE(acquired.has_value());
        auto submitted = submitPresentableFrame(device, acquired->texture);
        REQUIRE(submitted.has_value());
        auto presented = device.present(*swapchain, *submitted);
        REQUIRE(presented.has_value());
        REQUIRE(device.acquireNextImage(*swapchain).has_value());
    }
    CHECK(rachis::d3d12::test::nativeSwapchainsReleasedWhileReferenced().load(std::memory_order_relaxed) ==
          releasedWhileReferenced);

    auto created = rachis::createDevice(deviceDesc);
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    auto surface = device.createSurface(rachis::SurfaceDesc{.windowHandle = window.get()});
    REQUIRE(surface.has_value());
    auto swapchain = device.createSwapchain(copyableSwapchainDesc(*surface, initialWidth, initialHeight));
    REQUIRE(swapchain.has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    CHECK(rachis::d3d12::test::nativeSwapchainsReleasedWhileReferenced().load(std::memory_order_relaxed) ==
          releasedWhileReferenced);
    REQUIRE(device.destroySurface(*surface).has_value());
}

TEST_CASE("DXGI_STATUS_MODE_CHANGED maps to surface-out-of-date and recovers via same-size refresh",
          "[size:medium][rhi][presentation][surface-out-of-date][warp]")
{
    const ForceLegacyBarriers forceLegacy;
    const HiddenWindow window;
    REQUIRE(window.get() != nullptr);

    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::disabled,
        .enableDred = false,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    auto surface = device.createSurface(rachis::SurfaceDesc{.windowHandle = window.get()});
    REQUIRE(surface.has_value());
    auto swapchain = device.createSwapchain(rachis::SwapchainDesc{
        .surface = *surface,
        .width = initialWidth,
        .height = initialHeight,
        .imageCount = 2,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copyDestination,
    });
    REQUIRE(swapchain.has_value());

    auto acquireAndSubmit = [&]() -> rachis::Result<rachis::TimelinePoint> {
        auto acquired = device.acquireNextImage(*swapchain);
        if (!acquired)
        {
            return std::unexpected(std::move(acquired.error()));
        }
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        if (!command)
        {
            return std::unexpected(std::move(command.error()));
        }

        auto leavePresent = device.textureBarrier(*command,
                                                  rachis::TextureBarrier{
                                                      .texture = acquired->texture,
                                                      .stagesBefore = rachis::PipelineStage::none,
                                                      .stagesAfter = rachis::PipelineStage::copy,
                                                      .accessBefore = rachis::Access::none,
                                                      .accessAfter = rachis::Access::copyDestination,
                                                      .layoutBefore = rachis::TextureLayout::present,
                                                      .layoutAfter = rachis::TextureLayout::copyDestination,
                                                  });
        if (!leavePresent)
        {
            return std::unexpected(std::move(leavePresent.error()));
        }
        auto returnToPresent = device.textureBarrier(*command,
                                                     rachis::TextureBarrier{
                                                         .texture = acquired->texture,
                                                         .stagesBefore = rachis::PipelineStage::copy,
                                                         .stagesAfter = rachis::PipelineStage::none,
                                                         .accessBefore = rachis::Access::copyDestination,
                                                         .accessAfter = rachis::Access::none,
                                                         .layoutBefore = rachis::TextureLayout::copyDestination,
                                                         .layoutAfter = rachis::TextureLayout::present,
                                                     });
        if (!returnToPresent)
        {
            return std::unexpected(std::move(returnToPresent.error()));
        }
        return device.submit(std::move(*command));
    };

    auto submitted = acquireAndSubmit();
    REQUIRE(submitted.has_value());
    rachis::d3d12::test::nextPresentResult().store(static_cast<std::int32_t>(DXGI_STATUS_MODE_CHANGED),
                                                   std::memory_order_relaxed);
    auto outOfDate = device.present(*swapchain, *submitted);
    REQUIRE(outOfDate.has_value());
    CHECK(*outOfDate == rachis::PresentStatus::surfaceOutOfDate);

    CHECK_FALSE(device.acquireNextImage(*swapchain).has_value());
    CHECK_FALSE(device.present(*swapchain, *submitted).has_value());

    REQUIRE(device.resizeSwapchain(*swapchain, initialWidth, initialHeight).has_value());
    auto recovered = acquireAndSubmit();
    REQUIRE(recovered.has_value());
    auto presented = device.present(*swapchain, *recovered);
    REQUIRE(presented.has_value());
    CHECK(*presented == rachis::PresentStatus::presented);
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
}

TEST_CASE("D3D12 swapchains support multi-submit and batch-submit passes without std::terminate",
          "[size:medium][rhi][presentation][swapchain][warp]")
{
    const HiddenWindow window;
    REQUIRE(window.get() != nullptr);

    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto surface = device.createSurface(rachis::SurfaceDesc{
        .windowHandle = window.get(),
        .label = "D3D12 multi-submit surface",
    });
    REQUIRE(surface.has_value());
    auto swapchain = device.createSwapchain(rachis::SwapchainDesc{
        .surface = *surface,
        .width = initialWidth,
        .height = initialHeight,
        .imageCount = 3,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
        .presentMode = rachis::PresentMode::fifo,
        .label = "D3D12 multi-submit swapchain",
    });
    REQUIRE(swapchain.has_value());

    auto graphicsPool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 4,
        .label = "Multi-submit graphics pool",
    });
    auto copyPool = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 1,
        .label = "Multi-submit copy pool",
    });
    REQUIRE(graphicsPool.has_value());
    REQUIRE(copyPool.has_value());

    auto readback = device.createBuffer({
        .size = readbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Multi-submit readback",
    });
    REQUIRE(readback.has_value());

    auto frame1 = device.acquireNextImage(*swapchain);
    REQUIRE(frame1.has_value());

    auto copyCommand = device.acquireCommandBuffer(*copyPool);
    REQUIRE(copyCommand.has_value());
    const rachis::BufferTextureCopyRegion region{
        .bytesPerRow = rowPitch,
        .rowsPerImage = initialHeight,
        .extent = {.width = initialWidth, .height = initialHeight, .depth = 1},
    };
    auto rejectedCopy = device.copyTextureToBuffer(*copyCommand, frame1->texture, *readback, std::array{region});
    REQUIRE_FALSE(rejectedCopy.has_value());
    CHECK(rejectedCopy.error().code == rachis::ErrorCode::invalidArgument);
    copyCommand = std::unexpected(rachis::Error{});

    auto pass1 = device.acquireCommandBuffer(*graphicsPool);
    auto pass2 = device.acquireCommandBuffer(*graphicsPool);
    REQUIRE(pass1.has_value());
    REQUIRE(pass2.has_value());

    REQUIRE(device
                .textureBarrier(*pass1,
                                {
                                    .texture = frame1->texture,
                                    .stagesBefore = frame1->initialStages,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessBefore = frame1->initialAccess,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutBefore = frame1->initialLayout,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array clearAttachments{
        rachis::ColorAttachment{
            .view = frame1->renderTargetView,
            .loadOperation = rachis::LoadOperation::clear,
            .storeOperation = rachis::StoreOperation::store,
            .clearValue = {0.25F, 0.5F, 0.75F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*pass1, {.colorAttachments = clearAttachments}).has_value());
    REQUIRE(device.endRendering(*pass1).has_value());

    const std::array loadAttachments{
        rachis::ColorAttachment{
            .view = frame1->renderTargetView,
            .loadOperation = rachis::LoadOperation::load,
            .storeOperation = rachis::StoreOperation::store,
        },
    };
    REQUIRE(device.beginRendering(*pass2, {.colorAttachments = loadAttachments}).has_value());
    REQUIRE(device.endRendering(*pass2).has_value());
    REQUIRE(device
                .textureBarrier(*pass2,
                                {
                                    .texture = frame1->texture,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*pass2, frame1->texture, *readback, std::array{region}).has_value());
    REQUIRE(device
                .textureBarrier(*pass2,
                                {
                                    .texture = frame1->texture,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = frame1->requiredFinalStages,
                                    .accessBefore = rachis::Access::copySource,
                                    .accessAfter = frame1->requiredFinalAccess,
                                    .layoutBefore = rachis::TextureLayout::copySource,
                                    .layoutAfter = frame1->requiredFinalLayout,
                                })
                .has_value());

    auto firstPoint = device.submit(std::move(*pass1));
    REQUIRE(firstPoint.has_value());

    auto prematurePresent = device.present(*swapchain, *firstPoint);
    REQUIRE_FALSE(prematurePresent.has_value());
    CHECK(prematurePresent.error().code == rachis::ErrorCode::invalidState);

    auto secondPoint = device.submit(std::move(*pass2));
    REQUIRE(secondPoint.has_value());

    auto stalePresent = device.present(*swapchain, *firstPoint);
    REQUIRE_FALSE(stalePresent.has_value());
    CHECK(stalePresent.error().code == rachis::ErrorCode::invalidArgument);

    auto presented1 = device.present(*swapchain, *secondPoint);
    REQUIRE(presented1.has_value());
    CHECK(*presented1 == rachis::PresentStatus::presented);

    REQUIRE(device.wait(*secondPoint).has_value());
    std::array<std::byte, 4> pixel{};
    REQUIRE(device.readBuffer(*readback, 0, pixel).has_value());
    CHECK(nearByte(pixel[0], 191));
    CHECK(nearByte(pixel[1], 128));
    CHECK(nearByte(pixel[2], 64));
    CHECK(nearByte(pixel[3], 255));

    auto frame2 = device.acquireNextImage(*swapchain);
    REQUIRE(frame2.has_value());
    auto batch1 = device.acquireCommandBuffer(*graphicsPool);
    auto batch2 = device.acquireCommandBuffer(*graphicsPool);
    REQUIRE(batch1.has_value());
    REQUIRE(batch2.has_value());

    REQUIRE(device
                .textureBarrier(*batch1,
                                {
                                    .texture = frame2->texture,
                                    .stagesBefore = frame2->initialStages,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessBefore = frame2->initialAccess,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutBefore = frame2->initialLayout,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array batchClear{
        rachis::ColorAttachment{
            .view = frame2->renderTargetView,
            .loadOperation = rachis::LoadOperation::clear,
            .storeOperation = rachis::StoreOperation::store,
            .clearValue = {0.1F, 0.2F, 0.3F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*batch1, {.colorAttachments = batchClear}).has_value());
    REQUIRE(device.endRendering(*batch1).has_value());
    REQUIRE(device
                .textureBarrier(*batch2,
                                {
                                    .texture = frame2->texture,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = frame2->requiredFinalStages,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = frame2->requiredFinalAccess,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = frame2->requiredFinalLayout,
                                })
                .has_value());

    std::array batch{std::move(*batch1), std::move(*batch2)};
    auto batchPoint = device.submit(batch);
    REQUIRE(batchPoint.has_value());
    auto presented2 = device.present(*swapchain, *batchPoint);
    REQUIRE(presented2.has_value());
    CHECK(*presented2 == rachis::PresentStatus::presented);

    REQUIRE(device.waitIdle().has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyCommandPool(*graphicsPool).has_value());
    REQUIRE(device.destroyCommandPool(*copyPool).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("DirectComposition surface capabilities omit immediate present mode and enforce contract",
          "[size:medium][rhi][presentation][warp]")
{
    const HiddenWindow window(WS_EX_NOREDIRECTIONBITMAP);
    REQUIRE(window.get() != nullptr);

    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto nullWindow = device.createSurface(rachis::SurfaceDesc{
        .kind = rachis::SurfaceKind::win32Composition,
        .windowHandle = nullptr,
        .label = "Invalid composition surface",
    });
    REQUIRE_FALSE(nullWindow.has_value());
    CHECK(nullWindow.error().code == rachis::ErrorCode::invalidArgument);

    auto surface = device.createSurface(rachis::SurfaceDesc{
        .kind = rachis::SurfaceKind::win32Composition,
        .windowHandle = window.get(),
        .label = "DirectComposition capability surface",
    });
    REQUIRE(surface.has_value());

    auto capabilities = device.querySurfaceCapabilities(*surface);
    REQUIRE(capabilities.has_value());
    CHECK(capabilities->minImageCount == 2);
    CHECK(capabilities->maxImageCount >= 3);
    CHECK(capabilities->maxWidth >= initialWidth);
    CHECK(capabilities->maxHeight >= initialHeight);
    CHECK(rachis::hasAny(capabilities->usages, rachis::TextureUsage::renderTarget));
    CHECK(rachis::hasAny(capabilities->usages, rachis::TextureUsage::copySource));
    CHECK(rachis::hasAny(capabilities->usages, rachis::TextureUsage::copyDestination));
    CHECK(rachis::hasAny(capabilities->usages, rachis::TextureUsage::sampled));
    CHECK(rachis::supportsPresentMode(capabilities->presentModes, rachis::PresentMode::fifo));
    CHECK_FALSE(rachis::supportsPresentMode(capabilities->presentModes, rachis::PresentMode::immediate));
    CHECK(std::ranges::find(capabilities->formats,
                            rachis::SurfaceFormat{
                                .format = rachis::Format::bgra8Unorm,
                                .colorSpace = rachis::SurfaceColorSpace::srgbNonlinear,
                            }) != capabilities->formats.end());
    CHECK(std::ranges::find(capabilities->formats,
                            rachis::SurfaceFormat{
                                .format = rachis::Format::rgba8UnormSrgb,
                                .colorSpace = rachis::SurfaceColorSpace::srgbNonlinear,
                            }) != capabilities->formats.end());

    const rachis::SwapchainDesc swapchainDesc{
        .surface = *surface,
        .width = initialWidth,
        .height = initialHeight,
        .imageCount = 3,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
        .presentMode = rachis::PresentMode::fifo,
        .label = "DirectComposition capability swapchain",
    };

    auto immediateDesc = swapchainDesc;
    immediateDesc.presentMode = rachis::PresentMode::immediate;
    auto immediate = device.createSwapchain(immediateDesc);
    REQUIRE_FALSE(immediate.has_value());
    CHECK(immediate.error().code == rachis::ErrorCode::unsupported);

    const auto leakedBefore = rachis::d3d12::test::nativeSwapchainsReleasedWhileReferenced().load();
    auto swapchain = device.createSwapchain(swapchainDesc);
    if (!swapchain)
    {
        INFO(swapchain.error().message);
    }
    REQUIRE(swapchain.has_value());

    auto busySurface = device.destroySurface(*surface);
    REQUIRE_FALSE(busySurface.has_value());
    CHECK(busySurface.error().code == rachis::ErrorCode::invalidState);

    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    CHECK(rachis::d3d12::test::nativeSwapchainsReleasedWhileReferenced().load() == leakedBefore);
    REQUIRE(device.destroySurface(*surface).has_value());

    const auto counts = device.diagnosticCounts();
    CHECK(counts.errors == 0);
    CHECK(counts.corruptions == 0);
}

TEST_CASE("DirectComposition swapchains render, present, resize, and release cleanly",
          "[size:medium][rhi][presentation][warp]")
{
    const DWORD exStyle = GENERATE(DWORD{0}, DWORD{WS_EX_NOREDIRECTIONBITMAP});
    const auto format = GENERATE(rachis::Format::bgra8Unorm, rachis::Format::bgra8UnormSrgb);
    CAPTURE(exStyle, format);

    const HiddenWindow window(exStyle);
    REQUIRE(window.get() != nullptr);

    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto surface = device.createSurface(rachis::SurfaceDesc{
        .kind = rachis::SurfaceKind::win32Composition,
        .windowHandle = window.get(),
        .label = "DirectComposition render surface",
    });
    REQUIRE(surface.has_value());

    const auto leakedBefore = rachis::d3d12::test::nativeSwapchainsReleasedWhileReferenced().load();
    auto swapchain = device.createSwapchain(rachis::SwapchainDesc{
        .surface = *surface,
        .width = initialWidth,
        .height = initialHeight,
        .imageCount = 3,
        .format = format,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource |
                 rachis::TextureUsage::copyDestination | rachis::TextureUsage::sampled,
        .presentMode = rachis::PresentMode::fifo,
        .label = "DirectComposition swapchain",
    });
    if (!swapchain)
    {
        INFO(swapchain.error().message);
    }
    REQUIRE(swapchain.has_value());

    auto info = device.querySwapchainPresentation(*swapchain);
    REQUIRE(info.has_value());
    CHECK(info->width == initialWidth);
    CHECK(info->height == initialHeight);
    CHECK(info->format == format);
    CHECK(info->presentMode == rachis::PresentMode::fifo);
    CHECK(info->imageCount == 3);

    auto readback = device.createBuffer({
        .size = readbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "DirectComposition readback",
    });
    REQUIRE(readback.has_value());

    auto renderClearAndPresent = [&](std::uint32_t expectedWidth,
                                     std::uint32_t expectedHeight,
                                     std::uint64_t expectedGeneration,
                                     rachis::ClearColor clearColor,
                                     bool verifyReadback) {
        auto acquired = device.acquireNextImage(*swapchain);
        REQUIRE(acquired.has_value());
        CHECK(acquired->width == expectedWidth);
        CHECK(acquired->height == expectedHeight);
        CHECK(acquired->generation == expectedGeneration);
        CHECK(acquired->format == format);

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = acquired->texture,
                                        .stagesBefore = acquired->initialStages,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessBefore = acquired->initialAccess,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutBefore = acquired->initialLayout,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
        const std::array attachments{
            rachis::ColorAttachment{
                .view = acquired->renderTargetView,
                .loadOperation = rachis::LoadOperation::clear,
                .storeOperation = rachis::StoreOperation::store,
                .clearValue = clearColor,
            },
        };
        REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
        REQUIRE(device.endRendering(*command).has_value());
        if (verifyReadback)
        {
            REQUIRE(device
                        .textureBarrier(*command,
                                        {
                                            .texture = acquired->texture,
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
                                             rachis::BufferTextureCopy{
                                                 .buffer = *readback,
                                                 .bytesPerRow = rowPitch,
                                                 .rowsPerImage = expectedHeight,
                                                 .texture = acquired->texture,
                                                 .extent = {1, 1, 1},
                                             })
                        .has_value());
            REQUIRE(device
                        .textureBarrier(*command,
                                        {
                                            .texture = acquired->texture,
                                            .stagesBefore = rachis::PipelineStage::copy,
                                            .stagesAfter = acquired->requiredFinalStages,
                                            .accessBefore = rachis::Access::copySource,
                                            .accessAfter = acquired->requiredFinalAccess,
                                            .layoutBefore = rachis::TextureLayout::copySource,
                                            .layoutAfter = acquired->requiredFinalLayout,
                                        })
                        .has_value());
        }
        else
        {
            REQUIRE(device
                        .textureBarrier(*command,
                                        {
                                            .texture = acquired->texture,
                                            .stagesBefore = rachis::PipelineStage::renderTarget,
                                            .stagesAfter = acquired->requiredFinalStages,
                                            .accessBefore = rachis::Access::renderTarget,
                                            .accessAfter = acquired->requiredFinalAccess,
                                            .layoutBefore = rachis::TextureLayout::renderTarget,
                                            .layoutAfter = acquired->requiredFinalLayout,
                                        })
                        .has_value());
        }
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        auto presented = device.present(*swapchain, *submitted);
        if (!presented)
        {
            INFO(presented.error().message);
        }
        REQUIRE(presented.has_value());
        CHECK((*presented == rachis::PresentStatus::presented || *presented == rachis::PresentStatus::occluded));
        REQUIRE(device.wait(*submitted).has_value());
    };

    renderClearAndPresent(initialWidth, initialHeight, 1, {0.25F, 0.5F, 0.75F, 1.0F}, true);
    {
        std::array<std::byte, 4> pixel{};
        REQUIRE(device.readBuffer(*readback, 0, pixel).has_value());
        if (format == rachis::Format::bgra8Unorm)
        {
            CHECK(nearByte(pixel[0], 191));
            CHECK(nearByte(pixel[1], 128));
            CHECK(nearByte(pixel[2], 64));
            CHECK(nearByte(pixel[3], 255));
        }
        else
        {
            CHECK(std::to_integer<std::uint8_t>(pixel[0]) > 210);
            CHECK(std::to_integer<std::uint8_t>(pixel[1]) > 180);
            CHECK(std::to_integer<std::uint8_t>(pixel[2]) > 130);
            CHECK(nearByte(pixel[3], 255));
        }
    }

    auto timingBeforeResize = device.querySwapchainFrameTiming(*swapchain);
    REQUIRE(timingBeforeResize.has_value());
    CHECK(timingBeforeResize->displayedPresentationSequence <= timingBeforeResize->submittedPresentationSequence);

    REQUIRE(device.resizeSwapchain(*swapchain, resizedWidth, resizedHeight).has_value());
    auto resizedInfo = device.querySwapchainPresentation(*swapchain);
    REQUIRE(resizedInfo.has_value());
    CHECK(resizedInfo->width == resizedWidth);
    CHECK(resizedInfo->height == resizedHeight);

    renderClearAndPresent(resizedWidth, resizedHeight, 2, {1.0F, 0.0F, 0.0F, 1.0F}, true);
    {
        std::array<std::byte, 4> pixel{};
        REQUIRE(device.readBuffer(*readback, 0, pixel).has_value());
        CHECK(pixel[0] == std::byte{0});
        CHECK(pixel[1] == std::byte{0});
        CHECK(pixel[2] == std::byte{255});
        CHECK(pixel[3] == std::byte{255});
    }

    REQUIRE(device.resizeSwapchain(*swapchain, 128, 96).has_value());
    renderClearAndPresent(128, 96, 3, {0.0F, 1.0F, 0.0F, 1.0F}, false);

    auto cancelled = device.acquireNextImage(*swapchain);
    REQUIRE(cancelled.has_value());
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());

    REQUIRE(device.waitIdle().has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    CHECK(rachis::d3d12::test::nativeSwapchainsReleasedWhileReferenced().load() == leakedBefore);
    REQUIRE(device.destroySurface(*surface).has_value());

    const auto counts = device.diagnosticCounts();
    CHECK(counts.errors == 0);
    CHECK(counts.corruptions == 0);
}

TEST_CASE("a window enforces one live swapchain across win32 and win32Composition surfaces",
          "[size:medium][rhi][presentation][warp]")
{
    const HiddenWindow window;
    REQUIRE(window.get() != nullptr);

    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto hwndSurface = device.createSurface(rachis::SurfaceDesc{
        .kind = rachis::SurfaceKind::win32,
        .windowHandle = window.get(),
        .label = "Win32 surface",
    });
    REQUIRE(hwndSurface.has_value());
    auto compSurface = device.createSurface(rachis::SurfaceDesc{
        .kind = rachis::SurfaceKind::win32Composition,
        .windowHandle = window.get(),
        .label = "Win32 composition surface",
    });
    REQUIRE(compSurface.has_value());

    const auto desc = [](rachis::SurfaceHandle surface) {
        return rachis::SwapchainDesc{
            .surface = surface,
            .width = initialWidth,
            .height = initialHeight,
            .imageCount = 2,
            .format = rachis::Format::bgra8Unorm,
            .usage = rachis::TextureUsage::renderTarget,
            .presentMode = rachis::PresentMode::fifo,
        };
    };

    const auto leakedBefore = rachis::d3d12::test::nativeSwapchainsReleasedWhileReferenced().load();

    auto compSwapchain = device.createSwapchain(desc(*compSurface));
    REQUIRE(compSwapchain.has_value());
    auto refusedHwnd = device.createSwapchain(desc(*hwndSurface));
    REQUIRE_FALSE(refusedHwnd.has_value());
    CHECK(refusedHwnd.error().code == rachis::ErrorCode::invalidState);
    CHECK(refusedHwnd.error().message == "The native window already has a live swapchain.");
    REQUIRE(device.destroySwapchain(*compSwapchain).has_value());

    auto hwndSwapchain = device.createSwapchain(desc(*hwndSurface));
    REQUIRE(hwndSwapchain.has_value());
    auto refusedComp = device.createSwapchain(desc(*compSurface));
    REQUIRE_FALSE(refusedComp.has_value());
    CHECK(refusedComp.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.destroySwapchain(*hwndSwapchain).has_value());

    auto recreatedComp1 = device.createSwapchain(desc(*compSurface));
    REQUIRE(recreatedComp1.has_value());
    REQUIRE(device.destroySwapchain(*recreatedComp1).has_value());

    auto recreatedComp2 = device.createSwapchain(desc(*compSurface));
    REQUIRE(recreatedComp2.has_value());
    REQUIRE(device.destroySwapchain(*recreatedComp2).has_value());

    CHECK(rachis::d3d12::test::nativeSwapchainsReleasedWhileReferenced().load() == leakedBefore);
    REQUIRE(device.destroySurface(*compSurface).has_value());
    REQUIRE(device.destroySurface(*hwndSurface).has_value());

    const auto counts = device.diagnosticCounts();
    CHECK(counts.errors == 0);
    CHECK(counts.corruptions == 0);
}

TEST_CASE("DirectComposition swapchain GetBuffer failure detaches target and allows retry",
          "[size:medium][rhi][presentation][warp]")
{
    const HiddenWindow window(WS_EX_NOREDIRECTIONBITMAP);
    REQUIRE(window.get() != nullptr);

    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto surface = device.createSurface(rachis::SurfaceDesc{
        .kind = rachis::SurfaceKind::win32Composition,
        .windowHandle = window.get(),
        .label = "DirectComposition failure recovery surface",
    });
    REQUIRE(surface.has_value());

    const rachis::SwapchainDesc desc{
        .surface = *surface,
        .width = initialWidth,
        .height = initialHeight,
        .imageCount = 2,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .presentMode = rachis::PresentMode::fifo,
    };

    const auto leakedBefore = rachis::d3d12::test::nativeSwapchainsReleasedWhileReferenced().load();
    rachis::d3d12::test::nextSwapchainGetBufferResult().store(static_cast<std::int32_t>(E_FAIL),
                                                              std::memory_order_relaxed);
    auto failed = device.createSwapchain(desc);
    rachis::d3d12::test::nextSwapchainGetBufferResult().store(0, std::memory_order_relaxed);
    REQUIRE_FALSE(failed.has_value());
    CHECK(failed.error().code == rachis::ErrorCode::nativeFailure);

    auto retried = device.createSwapchain(desc);
    if (!retried)
    {
        INFO(retried.error().message);
    }
    REQUIRE(retried.has_value());
    REQUIRE(device.destroySwapchain(*retried).has_value());
    CHECK(rachis::d3d12::test::nativeSwapchainsReleasedWhileReferenced().load() == leakedBefore);
    REQUIRE(device.destroySurface(*surface).has_value());
}

TEST_CASE("DirectComposition swapchains work on hardware D3D12 adapters",
          "[size:medium][requires:HardwareD3D12][rhi][presentation]")
{
    auto adapters = rachis::enumerateAdapters(rachis::Backend::d3d12);
    REQUIRE(adapters.has_value());
    const auto hardware = std::ranges::find_if(
        *adapters, [](const rachis::AdapterInfo& adapter) { return adapter.type != rachis::AdapterType::software; });
    if (hardware == adapters->end())
    {
        SKIP("No hardware D3D12 adapter is available.");
    }

    const HiddenWindow window(WS_EX_NOREDIRECTIONBITMAP);
    REQUIRE(window.get() != nullptr);

    auto created = rachis::createDevice(rachis::DeviceDesc{
        .backend = rachis::Backend::d3d12,
        .adapterId = hardware->id,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto surface = device.createSurface(rachis::SurfaceDesc{
        .kind = rachis::SurfaceKind::win32Composition,
        .windowHandle = window.get(),
        .label = "Hardware DirectComposition surface",
    });
    REQUIRE(surface.has_value());

    const auto leakedBefore = rachis::d3d12::test::nativeSwapchainsReleasedWhileReferenced().load();
    auto swapchain = device.createSwapchain(rachis::SwapchainDesc{
        .surface = *surface,
        .width = initialWidth,
        .height = initialHeight,
        .imageCount = 3,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource |
                 rachis::TextureUsage::copyDestination,
        .presentMode = rachis::PresentMode::fifo,
        .label = "Hardware DirectComposition swapchain",
    });
    if (!swapchain)
    {
        INFO(swapchain.error().message);
    }
    REQUIRE(swapchain.has_value());

    auto frame1 = device.acquireNextImage(*swapchain);
    REQUIRE(frame1.has_value());
    auto point1 = submitPresentableFrame(device, frame1->texture);
    REQUIRE(point1.has_value());
    auto presented1 = device.present(*swapchain, *point1);
    REQUIRE(presented1.has_value());
    REQUIRE(device.wait(*point1).has_value());

    REQUIRE(device.resizeSwapchain(*swapchain, resizedWidth, resizedHeight).has_value());
    auto frame2 = device.acquireNextImage(*swapchain);
    REQUIRE(frame2.has_value());
    CHECK(frame2->width == resizedWidth);
    CHECK(frame2->height == resizedHeight);
    CHECK(frame2->generation == 2);
    auto point2 = submitPresentableFrame(device, frame2->texture);
    REQUIRE(point2.has_value());
    auto presented2 = device.present(*swapchain, *point2);
    REQUIRE(presented2.has_value());
    REQUIRE(device.wait(*point2).has_value());

    auto timing = device.querySwapchainFrameTiming(*swapchain);
    REQUIRE(timing.has_value());
    CHECK(timing->displayedPresentationSequence <= timing->submittedPresentationSequence);

    REQUIRE(device.waitIdle().has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    CHECK(rachis::d3d12::test::nativeSwapchainsReleasedWhileReferenced().load() == leakedBefore);
    REQUIRE(device.destroySurface(*surface).has_value());

    const auto counts = device.diagnosticCounts();
    CHECK(counts.errors == 0);
    CHECK(counts.corruptions == 0);
}
