#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include <catch2/catch_test_macros.hpp>

#include <Windows.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <utility>

namespace
{

    class TestWindow final
    {
    public:
        explicit TestWindow(bool visible)
            : window_(CreateWindowExW(0,
                                      L"STATIC",
                                      L"Rachis Vulkan resize lifecycle",
                                      WS_OVERLAPPEDWINDOW,
                                      CW_USEDEFAULT,
                                      CW_USEDEFAULT,
                                      128,
                                      96,
                                      nullptr,
                                      nullptr,
                                      GetModuleHandleW(nullptr),
                                      nullptr))
        {
            if (visible && window_ != nullptr)
            {
                static_cast<void>(ShowWindow(window_, SW_SHOW));
            }
        }

        ~TestWindow()
        {
            if (window_ != nullptr)
            {
                static_cast<void>(DestroyWindow(window_));
            }
        }

        TestWindow(const TestWindow&) = delete;
        TestWindow& operator=(const TestWindow&) = delete;
        TestWindow(TestWindow&&) = delete;
        TestWindow& operator=(TestWindow&&) = delete;

        [[nodiscard]] HWND get() const noexcept
        {
            return window_;
        }

    private:
        HWND window_ = nullptr;
    };

    void captureErrors(const rachis::DiagnosticMessage& message, void* userData) noexcept
    {
        if (message.severity == rachis::DiagnosticSeverity::error ||
            message.severity == rachis::DiagnosticSeverity::corruption)
        {
            static_cast<std::atomic_uint64_t*>(userData)->fetch_add(1, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] rachis::DeviceDesc vulkanDesc(std::atomic_uint64_t* errors = nullptr) noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
            .diagnosticCallback = errors != nullptr ? &captureErrors : nullptr,
            .diagnosticUserData = errors,
        };
    }

    [[nodiscard]] rachis::Result<rachis::TimelinePoint> drawAcquired(rachis::Device& device,
                                                                     const rachis::AcquiredSwapchainImage& acquired,
                                                                     rachis::TextureViewHandle target = {})
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        if (!command)
        {
            return std::unexpected(std::move(command.error()));
        }
        auto toColor = device.textureBarrier(*command,
                                             {
                                                 .texture = acquired.texture,
                                                 .stagesBefore = acquired.initialStages,
                                                 .stagesAfter = rachis::PipelineStage::renderTarget,
                                                 .accessBefore = acquired.initialAccess,
                                                 .accessAfter = rachis::Access::renderTarget,
                                                 .layoutBefore = acquired.initialLayout,
                                                 .layoutAfter = rachis::TextureLayout::renderTarget,
                                             });
        if (!toColor)
        {
            return std::unexpected(std::move(toColor.error()));
        }
        const std::array attachments{
            rachis::ColorAttachment{
                .view = target.isValid() ? target : acquired.renderTargetView,
                .loadOperation = rachis::LoadOperation::clear,
                .clearValue = {0.2F, 0.1F, 0.4F, 1.0F},
            },
        };
        if (auto began = device.beginRendering(*command, {.colorAttachments = attachments}); !began)
        {
            return std::unexpected(std::move(began.error()));
        }
        if (auto ended = device.endRendering(*command); !ended)
        {
            return std::unexpected(std::move(ended.error()));
        }
        auto toPresent = device.textureBarrier(*command,
                                               {
                                                   .texture = acquired.texture,
                                                   .stagesBefore = rachis::PipelineStage::renderTarget,
                                                   .stagesAfter = acquired.requiredFinalStages,
                                                   .accessBefore = rachis::Access::renderTarget,
                                                   .accessAfter = acquired.requiredFinalAccess,
                                                   .layoutBefore = rachis::TextureLayout::renderTarget,
                                                   .layoutAfter = acquired.requiredFinalLayout,
                                               });
        if (!toPresent)
        {
            return std::unexpected(std::move(toPresent.error()));
        }
        return device.submit(std::move(*command));
    }

} // namespace

TEST_CASE("Vulkan swapchain resize bumps generation and retires borrowed images",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][resize]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireAdapterInfo(device).backend == rachis::Backend::vulkan);

    TestWindow const window(true);
    REQUIRE(window.get() != nullptr);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::win32,
        .windowHandle = window.get(),
        .label = "Vulkan resize surface",
    });
    REQUIRE(surface);
    auto capabilities = device.querySurfaceCapabilities(*surface);
    REQUIRE(capabilities);
    const auto width = capabilities->preferredWidth == 0 ? 96 : capabilities->preferredWidth;
    const auto height = capabilities->preferredHeight == 0 ? 64 : capabilities->preferredHeight;
    const auto format = capabilities->formats.front();
    auto swapchain = device.createSwapchain({
        .surface = *surface,
        .width = width,
        .height = height,
        .imageCount = (std::min)(capabilities->maxImageCount, (std::max)(capabilities->minImageCount, 2U)),
        .format = format.format,
        .colorSpace = format.colorSpace,
        .usage = rachis::TextureUsage::renderTarget,
        .presentMode = rachis::PresentMode::fifo,
    });
    REQUIRE(swapchain);

    auto first = device.acquireNextImage(*swapchain);
    REQUIRE(first);
    const auto firstGeneration = first->generation;
    const auto firstTexture = first->texture;
    auto submitted = drawAcquired(device, *first);
    REQUIRE(submitted);
    auto presented = device.present(*swapchain, *submitted);
    REQUIRE(presented);
    CHECK(*presented == rachis::PresentStatus::presented);

    const auto resizedWidth = (std::min)(width + 16, capabilities->maxWidth == 0 ? width + 16 : capabilities->maxWidth);
    const auto resizedHeight =
        (std::min)(height + 8, capabilities->maxHeight == 0 ? height + 8 : capabilities->maxHeight);
    auto resized = device.resizeSwapchain(*swapchain, resizedWidth, resizedHeight);
    if (!resized)
    {
        INFO(resized.error().message);
        REQUIRE(resized);
    }
    CHECK_FALSE(device.destroyTexture(firstTexture).has_value());

    auto second = device.acquireNextImage(*swapchain);
    REQUIRE(second);
    CHECK(second->width == resizedWidth);
    CHECK(second->height == resizedHeight);
    CHECK(second->generation != firstGeneration);
    auto secondSubmit = drawAcquired(device, *second);
    REQUIRE(secondSubmit);
    auto secondPresent = device.present(*swapchain, *secondSubmit);
    REQUIRE(secondPresent);
    CHECK(*secondPresent == rachis::PresentStatus::presented);

    REQUIRE(device.resizeSwapchain(*swapchain, resizedWidth, resizedHeight).has_value());
    auto refreshed = device.acquireNextImage(*swapchain);
    REQUIRE(refreshed);
    CHECK(refreshed->generation != second->generation);
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Vulkan swapchain resize and destroy reject live application views of swapchain images",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][resize]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireAdapterInfo(device).backend == rachis::Backend::vulkan);

    TestWindow const window(true);
    REQUIRE(window.get() != nullptr);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::win32,
        .windowHandle = window.get(),
        .label = "Vulkan swapchain view lifetime surface",
    });
    REQUIRE(surface);
    auto capabilities = device.querySurfaceCapabilities(*surface);
    REQUIRE(capabilities);
    const auto width = capabilities->preferredWidth == 0 ? 96 : capabilities->preferredWidth;
    const auto height = capabilities->preferredHeight == 0 ? 64 : capabilities->preferredHeight;
    const auto format = capabilities->formats.front();
    auto swapchain = device.createSwapchain({
        .surface = *surface,
        .width = width,
        .height = height,
        .imageCount = (std::min)(capabilities->maxImageCount, (std::max)(capabilities->minImageCount, 2U)),
        .format = format.format,
        .colorSpace = format.colorSpace,
        .usage = rachis::TextureUsage::renderTarget,
        .presentMode = rachis::PresentMode::fifo,
    });
    REQUIRE(swapchain);

    auto first = device.acquireNextImage(*swapchain);
    REQUIRE(first);
    const auto firstTexture = first->texture;
    auto applicationView = device.createTextureView({
        .texture = firstTexture,
        .usage = rachis::TextureViewUsage::renderTarget,
        .label = "Application swapchain image view",
    });
    REQUIRE(applicationView);
    auto submitted = drawAcquired(device, *first, *applicationView);
    REQUIRE(submitted);
    auto presented = device.present(*swapchain, *submitted);
    REQUIRE(presented);
    CHECK(*presented == rachis::PresentStatus::presented);

    const auto resizedWidth = (std::min)(width + 16, capabilities->maxWidth == 0 ? width + 16 : capabilities->maxWidth);
    const auto resizedHeight =
        (std::min)(height + 8, capabilities->maxHeight == 0 ? height + 8 : capabilities->maxHeight);
    auto rejectedResize = device.resizeSwapchain(*swapchain, resizedWidth, resizedHeight);
    REQUIRE_FALSE(rejectedResize.has_value());
    CHECK(rejectedResize.error().code == rachis::ErrorCode::invalidState);
    auto rejectedDestroy = device.destroySwapchain(*swapchain);
    REQUIRE_FALSE(rejectedDestroy.has_value());
    CHECK(rejectedDestroy.error().code == rachis::ErrorCode::invalidState);

    auto second = device.acquireNextImage(*swapchain);
    REQUIRE(second);
    CHECK(second->generation == first->generation);
    auto secondSubmit = drawAcquired(device, *second);
    REQUIRE(secondSubmit);
    auto secondPresent = device.present(*swapchain, *secondSubmit);
    REQUIRE(secondPresent);
    CHECK(*secondPresent == rachis::PresentStatus::presented);

    REQUIRE(device.destroyTextureView(*applicationView).has_value());
    auto resized = device.resizeSwapchain(*swapchain, resizedWidth, resizedHeight);
    if (!resized)
    {
        INFO(resized.error().message);
        REQUIRE(resized);
    }

    auto staleLayout = device.queryTextureTransferLayout(firstTexture,
                                                         {
                                                             .extent = {first->width, first->height, 1},
                                                         });
    REQUIRE_FALSE(staleLayout.has_value());
    CHECK(staleLayout.error().code == rachis::ErrorCode::invalidArgument);
    auto staleDestroy = device.destroyTexture(firstTexture);
    REQUIRE_FALSE(staleDestroy.has_value());
    CHECK(staleDestroy.error().code == rachis::ErrorCode::invalidArgument);
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    auto staleBarrier = device.textureBarrier(*command,
                                              {
                                                  .texture = firstTexture,
                                                  .stagesBefore = rachis::PipelineStage::none,
                                                  .stagesAfter = rachis::PipelineStage::renderTarget,
                                                  .accessBefore = rachis::Access::none,
                                                  .accessAfter = rachis::Access::renderTarget,
                                                  .layoutBefore = rachis::TextureLayout::undefined,
                                                  .layoutAfter = rachis::TextureLayout::renderTarget,
                                              });
    REQUIRE_FALSE(staleBarrier.has_value());
    CHECK(staleBarrier.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device.submit(std::move(*command)).has_value());

    auto third = device.acquireNextImage(*swapchain);
    REQUIRE(third);
    CHECK(third->generation != first->generation);
    CHECK(third->width == resizedWidth);
    CHECK(third->height == resizedHeight);
    auto thirdSubmit = drawAcquired(device, *third);
    REQUIRE(thirdSubmit);
    auto thirdPresent = device.present(*swapchain, *thirdSubmit);
    REQUIRE(thirdPresent);
    CHECK(*thirdPresent == rachis::PresentStatus::presented);

    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Vulkan device teardown retires swapchain images that still have application views",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][resize]")
{
    std::atomic_uint64_t errors{0};
    TestWindow const window(true);
    REQUIRE(window.get() != nullptr);
    {
        auto created = rachis::createDevice(vulkanDesc(&errors));
        if (rachis::test::skipWithoutCreatedDevice(created))
        {
            return;
        }
        auto device = std::move(*created);
        REQUIRE(rachis::test::requireAdapterInfo(device).backend == rachis::Backend::vulkan);

        auto surface = device.createSurface({
            .kind = rachis::SurfaceKind::win32,
            .windowHandle = window.get(),
            .label = "Vulkan teardown view surface",
        });
        REQUIRE(surface);
        auto capabilities = device.querySurfaceCapabilities(*surface);
        REQUIRE(capabilities);
        const auto format = capabilities->formats.front();
        auto swapchain = device.createSwapchain({
            .surface = *surface,
            .width = capabilities->preferredWidth == 0 ? 96 : capabilities->preferredWidth,
            .height = capabilities->preferredHeight == 0 ? 64 : capabilities->preferredHeight,
            .imageCount = (std::min)(capabilities->maxImageCount, (std::max)(capabilities->minImageCount, 2U)),
            .format = format.format,
            .colorSpace = format.colorSpace,
            .usage = rachis::TextureUsage::renderTarget,
            .presentMode = rachis::PresentMode::fifo,
        });
        REQUIRE(swapchain);

        auto acquired = device.acquireNextImage(*swapchain);
        REQUIRE(acquired);
        auto applicationView = device.createTextureView({
            .texture = acquired->texture,
            .usage = rachis::TextureViewUsage::renderTarget,
            .label = "Application swapchain image view at teardown",
        });
        REQUIRE(applicationView);
        auto submitted = drawAcquired(device, *acquired, *applicationView);
        REQUIRE(submitted);
        auto presented = device.present(*swapchain, *submitted);
        REQUIRE(presented);
        CHECK(*presented == rachis::PresentStatus::presented);
        CHECK(device.diagnosticCounts().errors == 0);
    }
    CHECK(errors.load(std::memory_order_relaxed) == 0);
}

TEST_CASE("Vulkan swapchain resize follows the live Win32 extent when the requested size is stale",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][resize]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireAdapterInfo(device).backend == rachis::Backend::vulkan);

    TestWindow const window(true);
    REQUIRE(window.get() != nullptr);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::win32,
        .windowHandle = window.get(),
        .label = "Vulkan stale resize surface",
    });
    REQUIRE(surface);
    auto capabilities = device.querySurfaceCapabilities(*surface);
    REQUIRE(capabilities);
    const auto width = capabilities->preferredWidth == 0 ? 96 : capabilities->preferredWidth;
    const auto height = capabilities->preferredHeight == 0 ? 64 : capabilities->preferredHeight;
    const auto format = capabilities->formats.front();
    auto swapchain = device.createSwapchain({
        .surface = *surface,
        .width = width,
        .height = height,
        .imageCount = (std::min)(capabilities->maxImageCount, (std::max)(capabilities->minImageCount, 2U)),
        .format = format.format,
        .colorSpace = format.colorSpace,
        .usage = rachis::TextureUsage::renderTarget,
        .presentMode = rachis::PresentMode::fifo,
    });
    REQUIRE(swapchain);

    auto first = device.acquireNextImage(*swapchain);
    REQUIRE(first);
    auto submitted = drawAcquired(device, *first);
    REQUIRE(submitted);
    auto presented = device.present(*swapchain, *submitted);
    REQUIRE(presented);
    CHECK(*presented == rachis::PresentStatus::presented);

    RECT outer{0, 0, 160, 96};
    REQUIRE(AdjustWindowRectEx(&outer, WS_OVERLAPPEDWINDOW, FALSE, 0) != 0);
    REQUIRE(SetWindowPos(window.get(),
                         nullptr,
                         0,
                         0,
                         outer.right - outer.left,
                         outer.bottom - outer.top,
                         SWP_NOMOVE | SWP_NOZORDER) != 0);
    MSG message{};
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE) != 0)
    {
        static_cast<void>(TranslateMessage(&message));
        static_cast<void>(DispatchMessageW(&message));
    }

    auto live = device.querySurfaceCapabilities(*surface);
    REQUIRE(live);
    REQUIRE(live->preferredWidth > 0);
    REQUIRE(live->preferredHeight > 0);
    auto resized = device.resizeSwapchain(*swapchain, width, height);
    if (!resized)
    {
        INFO(resized.error().message);
        REQUIRE(resized);
    }
    auto second = device.acquireNextImage(*swapchain);
    REQUIRE(second);
    if (live->minWidth == live->maxWidth && live->minHeight == live->maxHeight)
    {
        CHECK(second->width == live->preferredWidth);
        CHECK(second->height == live->preferredHeight);
    }
    auto secondSubmit = drawAcquired(device, *second);
    REQUIRE(secondSubmit);
    auto secondPresent = device.present(*swapchain, *secondSubmit);
    REQUIRE(secondPresent);
    CHECK(*secondPresent == rachis::PresentStatus::presented);

    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Vulkan swapchain suspends minimize and resumes without resetting device state",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][resize]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    TestWindow const window(true);
    REQUIRE(window.get() != nullptr);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::win32,
        .windowHandle = window.get(),
        .label = "Vulkan minimize surface",
    });
    REQUIRE(surface);
    auto capabilities = device.querySurfaceCapabilities(*surface);
    REQUIRE(capabilities);
    const auto width = capabilities->preferredWidth == 0 ? 96 : capabilities->preferredWidth;
    const auto height = capabilities->preferredHeight == 0 ? 64 : capabilities->preferredHeight;
    const auto format = capabilities->formats.front();
    auto swapchain = device.createSwapchain({
        .surface = *surface,
        .width = width,
        .height = height,
        .imageCount = (std::min)(capabilities->maxImageCount, (std::max)(capabilities->minImageCount, 2U)),
        .format = format.format,
        .colorSpace = format.colorSpace,
        .usage = rachis::TextureUsage::renderTarget,
        .presentMode = rachis::PresentMode::fifo,
    });
    REQUIRE(swapchain);

    auto marker = device.createBuffer({
        .size = 16,
        .usage = rachis::BufferUsage::copyDestination,
        .label = "resize scene marker",
    });
    REQUIRE(marker);

    REQUIRE(device.resizeSwapchain(*swapchain, 0, 0).has_value());
    auto suspended = device.acquireNextImage(*swapchain);
    REQUIRE_FALSE(suspended.has_value());
    CHECK(suspended.error().code == rachis::ErrorCode::invalidState);

    static_cast<void>(ShowWindow(window.get(), SW_MINIMIZE));
    auto minimizedCaps = device.querySurfaceCapabilities(*surface);
    REQUIRE(minimizedCaps);
    CHECK(minimizedCaps->preferredWidth == 0);
    CHECK(minimizedCaps->preferredHeight == 0);

    static_cast<void>(ShowWindow(window.get(), SW_RESTORE));
    REQUIRE(device.resizeSwapchain(*swapchain, width, height).has_value());
    auto restored = device.acquireNextImage(*swapchain);
    REQUIRE(restored);
    auto submitted = drawAcquired(device, *restored);
    REQUIRE(submitted);
    auto presented = device.present(*swapchain, *submitted);
    REQUIRE(presented);
    CHECK(*presented == rachis::PresentStatus::presented);
    REQUIRE(device.destroyBuffer(*marker).has_value());

    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Vulkan presentation teardown stays bounded after close-after-failure",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][resize]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    TestWindow const window(true);
    REQUIRE(window.get() != nullptr);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::win32,
        .windowHandle = window.get(),
        .label = "Vulkan teardown surface",
    });
    REQUIRE(surface);
    auto capabilities = device.querySurfaceCapabilities(*surface);
    REQUIRE(capabilities);
    const auto width = capabilities->preferredWidth == 0 ? 96 : capabilities->preferredWidth;
    const auto height = capabilities->preferredHeight == 0 ? 64 : capabilities->preferredHeight;
    const auto format = capabilities->formats.front();
    auto swapchain = device.createSwapchain({
        .surface = *surface,
        .width = width,
        .height = height,
        .imageCount = (std::min)(capabilities->maxImageCount, (std::max)(capabilities->minImageCount, 2U)),
        .format = format.format,
        .colorSpace = format.colorSpace,
        .usage = rachis::TextureUsage::renderTarget,
        .presentMode = rachis::PresentMode::fifo,
    });
    REQUIRE(swapchain);

    auto acquired = device.acquireNextImage(*swapchain);
    REQUIRE(acquired);
    REQUIRE_FALSE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());

    auto replacementSurface = device.createSurface({
        .kind = rachis::SurfaceKind::win32,
        .windowHandle = window.get(),
        .label = "Vulkan replacement surface",
    });
    REQUIRE(replacementSurface);
    auto replacement = device.createSwapchain({
        .surface = *replacementSurface,
        .width = width,
        .height = height,
        .imageCount = capabilities->minImageCount,
        .format = format.format,
        .colorSpace = format.colorSpace,
        .usage = rachis::TextureUsage::renderTarget,
        .presentMode = rachis::PresentMode::fifo,
    });
    REQUIRE(replacement);
    auto replacementImage = device.acquireNextImage(*replacement);
    REQUIRE(replacementImage);
    auto submitted = drawAcquired(device, *replacementImage);
    REQUIRE(submitted);
    auto presented = device.present(*replacement, *submitted);
    REQUIRE(presented);
    CHECK(*presented == rachis::PresentStatus::presented);
    REQUIRE(device.destroySwapchain(*replacement).has_value());
    REQUIRE(device.destroySurface(*replacementSurface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}
