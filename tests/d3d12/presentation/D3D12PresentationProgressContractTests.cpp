#include "D3D12/D3D12PresentationProgress.h"
#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Device.h"
#include "Rachis/TestSupport/D3D12FrameStatistics.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <Windows.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>

namespace
{

    using rachis::d3d12::PresentationProgressTracker;

    constexpr std::uint32_t windowWidth = 160;
    constexpr std::uint32_t windowHeight = 120;
    constexpr auto displayDeadline = std::chrono::milliseconds(500);
    constexpr int maxFollowUpFrames = 4;

    struct Frontier final
    {
        bool supported = false;
        std::uint64_t generation = 0;
        std::uint64_t submitted = 0;
        std::uint64_t displayed = 0;

        friend constexpr bool operator==(const Frontier&, const Frontier&) noexcept = default;
    };

    [[nodiscard]] Frontier frontier(const PresentationProgressTracker& tracker) noexcept
    {
        return Frontier{
            .supported = tracker.supported(),
            .generation = tracker.generation(),
            .submitted = tracker.submittedSequence(),
            .displayed = tracker.displayedSequence(),
        };
    }

    class TestWindow final
    {
    public:
        explicit TestWindow(bool shown)
            : window_(CreateWindowExW(0,
                                      L"STATIC",
                                      L"Rachis presentation progress",
                                      WS_OVERLAPPEDWINDOW,
                                      CW_USEDEFAULT,
                                      CW_USEDEFAULT,
                                      static_cast<int>(windowWidth),
                                      static_cast<int>(windowHeight),
                                      nullptr,
                                      nullptr,
                                      GetModuleHandleW(nullptr),
                                      nullptr))
        {
            if (window_ != nullptr && shown)
            {
                static_cast<void>(ShowWindow(window_, SW_SHOWNOACTIVATE));
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

    void pumpMessages() noexcept
    {
        MSG message{};
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE) != FALSE)
        {
            static_cast<void>(TranslateMessage(&message));
            static_cast<void>(DispatchMessageW(&message));
        }
    }

    [[nodiscard]] rachis::SwapchainFrameTiming queryTiming(
        const rachis::Device& device,
        rachis::SwapchainHandle swapchain,
        const rachis::test::ScopedD3D12FrameStatistics* statistics = nullptr)
    {
        auto timing = device.querySwapchainFrameTiming(swapchain);
        if (!timing)
        {
            INFO(timing.error().message);
        }
        REQUIRE(timing.has_value());
        CHECK(timing->displayedPresentationSequence <= timing->submittedPresentationSequence);
        if (statistics != nullptr)
        {
            REQUIRE(timing->displayedPresentationSequence >= statistics->observedSequence());
        }
        return *timing;
    }

    [[nodiscard]] rachis::Result<rachis::PresentStatus> clearAndPresent(rachis::Device& device,
                                                                        rachis::SwapchainHandle swapchain,
                                                                        float shade)
    {
        auto image = device.acquireNextImage(swapchain);
        if (!image)
        {
            return std::unexpected(std::move(image.error()));
        }
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        if (!command)
        {
            return std::unexpected(std::move(command.error()));
        }
        auto toTarget = device.textureBarrier(*command,
                                              rachis::TextureBarrier{
                                                  .texture = image->texture,
                                                  .stagesBefore = image->initialStages,
                                                  .stagesAfter = rachis::PipelineStage::renderTarget,
                                                  .accessBefore = image->initialAccess,
                                                  .accessAfter = rachis::Access::renderTarget,
                                                  .layoutBefore = image->initialLayout,
                                                  .layoutAfter = rachis::TextureLayout::renderTarget,
                                              });
        if (!toTarget)
        {
            return std::unexpected(std::move(toTarget.error()));
        }
        const std::array attachments{
            rachis::ColorAttachment{
                .view = image->renderTargetView,
                .loadOperation = rachis::LoadOperation::clear,
                .clearValue = {shade, 0.25F, 0.5F, 1.0F},
            },
        };
        auto begun = device.beginRendering(*command, {.colorAttachments = attachments});
        if (!begun)
        {
            return std::unexpected(std::move(begun.error()));
        }
        auto ended = device.endRendering(*command);
        if (!ended)
        {
            return std::unexpected(std::move(ended.error()));
        }
        auto toPresent = device.textureBarrier(*command,
                                               rachis::TextureBarrier{
                                                   .texture = image->texture,
                                                   .stagesBefore = rachis::PipelineStage::renderTarget,
                                                   .stagesAfter = image->requiredFinalStages,
                                                   .accessBefore = rachis::Access::renderTarget,
                                                   .accessAfter = image->requiredFinalAccess,
                                                   .layoutBefore = rachis::TextureLayout::renderTarget,
                                                   .layoutAfter = image->requiredFinalLayout,
                                               });
        if (!toPresent)
        {
            return std::unexpected(std::move(toPresent.error()));
        }
        auto submitted = device.submit(std::move(*command));
        if (!submitted)
        {
            return std::unexpected(std::move(submitted.error()));
        }
        return device.present(swapchain, *submitted);
    }

    void presentFollowUpFrame(rachis::Device& device, rachis::SwapchainHandle swapchain)
    {
        auto presented = clearAndPresent(device, swapchain, 0.5F);
        if (!presented)
        {
            INFO(presented.error().message);
        }
        REQUIRE(presented.has_value());
        REQUIRE(*presented == rachis::PresentStatus::presented);
        REQUIRE(device.waitIdle().has_value());
    }

    [[nodiscard]] rachis::SwapchainFrameTiming awaitDisplayed(
        rachis::Device& device,
        rachis::SwapchainHandle swapchain,
        std::uint64_t generation,
        std::uint64_t target,
        const rachis::test::ScopedD3D12FrameStatistics& statistics)
    {
        auto timing = queryTiming(device, swapchain, &statistics);
        auto previous = timing.displayedPresentationSequence;
        for (int followUps = 0; timing.displayedPresentationSequence < target && followUps <= maxFollowUpFrames;
             ++followUps)
        {
            if (followUps > 0)
            {
                presentFollowUpFrame(device, swapchain);
            }
            const auto deadline = std::chrono::steady_clock::now() + displayDeadline;
            while (timing.displayedPresentationSequence < target && std::chrono::steady_clock::now() < deadline)
            {
                pumpMessages();
                Sleep(2);
                timing = queryTiming(device, swapchain, &statistics);
                CHECK(timing.presentationGeneration == generation);
                CHECK(timing.submittedPresentationSequence == target + static_cast<std::uint64_t>(followUps));
                CHECK(timing.displayedPresentationSequence >= previous);
                previous = timing.displayedPresentationSequence;
            }
        }
        return timing;
    }

    void checkDisplayedProgress(rachis::Device& device, HWND window)
    {
        rachis::test::ScopedD3D12FrameStatistics statistics;
        auto surface = device.createSurface(rachis::SurfaceDesc{
            .windowHandle = window,
            .label = "Presentation progress surface",
        });
        REQUIRE(surface.has_value());
        auto swapchain = device.createSwapchain(rachis::SwapchainDesc{
            .surface = *surface,
            .width = windowWidth,
            .height = windowHeight,
            .imageCount = 3,
            .format = rachis::Format::bgra8Unorm,
            .usage = rachis::TextureUsage::renderTarget,
            .presentMode = rachis::PresentMode::fifo,
            .label = "Presentation progress swapchain",
        });
        if (!swapchain)
        {
            INFO(swapchain.error().message);
        }
        REQUIRE(swapchain.has_value());

        auto timing = queryTiming(device, *swapchain, &statistics);
        REQUIRE(timing.presentationSupported);
        CHECK(timing.presentationGeneration == 1);
        CHECK(timing.submittedPresentationSequence == 0);
        CHECK(timing.displayedPresentationSequence == 0);

        constexpr std::uint64_t firstFrames = 3;
        std::uint64_t displayed = 0;
        for (std::uint64_t frame = 1; frame <= firstFrames; ++frame)
        {
            auto presented = clearAndPresent(device, *swapchain, static_cast<float>(frame) / 4.0F);
            if (!presented)
            {
                INFO(presented.error().message);
            }
            REQUIRE(presented.has_value());
            REQUIRE(*presented == rachis::PresentStatus::presented);
            timing = queryTiming(device, *swapchain, &statistics);
            CHECK(timing.presentationGeneration == 1);
            CHECK(timing.submittedPresentationSequence == frame);
            CHECK(timing.displayedPresentationSequence >= displayed);
            displayed = timing.displayedPresentationSequence;
        }

        REQUIRE(device.waitIdle().has_value());
        timing = awaitDisplayed(device, *swapchain, 1, firstFrames, statistics);
        CAPTURE(statistics.lastResult(), statistics.observedSequence(), timing.displayedPresentationSequence);
        const bool firstFeedback = statistics.observedSequence() >= firstFrames;
        if (firstFeedback)
        {
            REQUIRE(timing.displayedPresentationSequence >= firstFrames);
        }

        REQUIRE(device.resizeSwapchain(*swapchain, windowWidth + 40, windowHeight + 30).has_value());
        statistics.beginGeneration();
        timing = queryTiming(device, *swapchain, &statistics);
        CHECK(timing.presentationSupported);
        CHECK(timing.presentationGeneration == 2);
        CHECK(timing.submittedPresentationSequence == 0);
        CHECK(timing.displayedPresentationSequence == 0);

        for (int poll = 0; poll < 10; ++poll)
        {
            pumpMessages();
            Sleep(5);
            timing = queryTiming(device, *swapchain, &statistics);
            CHECK(timing.presentationGeneration == 2);
            CHECK(timing.displayedPresentationSequence == 0);
        }

        constexpr std::uint64_t secondFrames = 2;
        for (std::uint64_t frame = 1; frame <= secondFrames; ++frame)
        {
            auto presented = clearAndPresent(device, *swapchain, 1.0F - static_cast<float>(frame) / 4.0F);
            REQUIRE(presented.has_value());
            REQUIRE(*presented == rachis::PresentStatus::presented);
        }
        REQUIRE(device.waitIdle().has_value());
        timing = awaitDisplayed(device, *swapchain, 2, secondFrames, statistics);
        CAPTURE(statistics.lastResult(), statistics.observedSequence(), timing.displayedPresentationSequence);
        const bool secondFeedback = statistics.observedSequence() >= secondFrames;
        if (secondFeedback)
        {
            REQUIRE(timing.displayedPresentationSequence >= secondFrames);
        }

        const auto stale = *swapchain;
        REQUIRE(device.destroySwapchain(*swapchain).has_value());
        auto staleTiming = device.querySwapchainFrameTiming(stale);
        REQUIRE_FALSE(staleTiming.has_value());
        CHECK(staleTiming.error().code == rachis::ErrorCode::invalidArgument);
        REQUIRE(device.destroySurface(*surface).has_value());

        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
        if (!firstFeedback || !secondFeedback)
        {
            SKIP("DXGI did not supply display statistics through the target sequence in both generations; "
                 "submission, native sample mapping, resize and cleanup checks completed.");
        }
    }

} // namespace

TEST_CASE("D3D12 presentation tracker maps DXGI present counts onto generation sequences",
          "[size:small][rhi][presentation][d3d12]")
{
    PresentationProgressTracker tracker;
    CHECK(frontier(tracker) == Frontier{});

    tracker.presented(1);
    tracker.observed(1);
    CHECK(frontier(tracker) == Frontier{});

    SECTION("submitted and displayed frontiers advance separately and never move back")
    {
        tracker.start(1, 0);
        CHECK(frontier(tracker) == Frontier{true, 1, 0, 0});
        tracker.observed(0);
        CHECK(frontier(tracker) == Frontier{true, 1, 0, 0});
        tracker.presented(1);
        tracker.presented(2);
        tracker.presented(3);
        CHECK(frontier(tracker) == Frontier{true, 1, 3, 0});
        tracker.observed(2);
        CHECK(frontier(tracker) == Frontier{true, 1, 3, 2});
        tracker.observed(1);
        CHECK(frontier(tracker) == Frontier{true, 1, 3, 2});

        tracker.observed(4);
        CHECK(frontier(tracker) == Frontier{true, 1, 3, 2});
        tracker.observed(3);
        CHECK(frontier(tracker) == Frontier{true, 1, 3, 3});
    }

    SECTION("a Present that DXGI did not count adds no sequence")
    {
        tracker.start(1, 7);
        tracker.presented(8);
        tracker.presented(8);
        CHECK(frontier(tracker) == Frontier{true, 1, 1, 0});
        tracker.observed(8);
        CHECK(frontier(tracker) == Frontier{true, 1, 1, 1});
        tracker.observed(7);
        CHECK(frontier(tracker) == Frontier{true, 1, 1, 1});
    }

    SECTION("a new generation restarts both frontiers and ignores earlier statistics")
    {
        tracker.start(1, 0);
        for (std::uint32_t count = 1; count <= 8; ++count)
        {
            tracker.presented(count);
        }
        tracker.observed(4);
        CHECK(frontier(tracker) == Frontier{true, 1, 8, 4});
        tracker.beginGeneration(2, 8);
        CHECK(frontier(tracker) == Frontier{true, 2, 0, 0});
        tracker.observed(8);
        tracker.observed(4);
        CHECK(frontier(tracker) == Frontier{true, 2, 0, 0});
        tracker.presented(9);
        tracker.observed(4);
        CHECK(frontier(tracker) == Frontier{true, 2, 1, 0});
        tracker.observed(8);
        CHECK(frontier(tracker) == Frontier{true, 2, 1, 0});
        tracker.observed(9);
        CHECK(frontier(tracker) == Frontier{true, 2, 1, 1});
    }

    SECTION("the native 32-bit counters may wrap")
    {
        constexpr auto top = std::numeric_limits<std::uint32_t>::max();
        tracker.start(5, top - 1);
        tracker.presented(top);
        tracker.presented(0);
        tracker.presented(1);
        CHECK(frontier(tracker) == Frontier{true, 5, 3, 0});
        tracker.observed(top - 1);
        CHECK(frontier(tracker) == Frontier{true, 5, 3, 0});
        tracker.observed(0);
        CHECK(frontier(tracker) == Frontier{true, 5, 3, 2});
        tracker.observed(top);
        CHECK(frontier(tracker) == Frontier{true, 5, 3, 2});
        tracker.observed(1);
        CHECK(frontier(tracker) == Frontier{true, 5, 3, 3});
        tracker.beginGeneration(6, 1);
        tracker.presented(2);
        tracker.observed(0);
        CHECK(frontier(tracker) == Frontier{true, 6, 1, 0});
        tracker.observed(2);
        CHECK(frontier(tracker) == Frontier{true, 6, 1, 1});
    }

    SECTION("a counter that runs backwards disables the feedback for good")
    {
        tracker.start(1, 10);
        tracker.presented(11);
        tracker.observed(11);
        CHECK(frontier(tracker) == Frontier{true, 1, 1, 1});
        tracker.presented(9);
        CHECK(frontier(tracker) == Frontier{});
        tracker.beginGeneration(2, 9);
        tracker.presented(10);
        tracker.observed(10);
        CHECK(frontier(tracker) == Frontier{});
    }

    SECTION("a resize whose counter ran backwards disables the feedback")
    {
        tracker.start(1, 10);
        tracker.presented(11);
        tracker.beginGeneration(2, 10);
        CHECK(frontier(tracker) == Frontier{});
    }
}

TEST_CASE("D3D12 WARP counts only Presents that reached DXGI", "[size:medium][rhi][presentation][d3d12][warp]")
{
    const TestWindow window(false);
    REQUIRE(window.get() != nullptr);
    auto created = rachis::createDevice(rachis::test::d3d12WarpDeviceDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    auto surface = device.createSurface(rachis::SurfaceDesc{.windowHandle = window.get()});
    REQUIRE(surface.has_value());
    auto swapchain = device.createSwapchain(rachis::SwapchainDesc{
        .surface = *surface,
        .width = windowWidth,
        .height = windowHeight,
        .imageCount = 2,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
    });
    REQUIRE(swapchain.has_value());

    rachis::d3d12::test::nextPresentResult().store(static_cast<std::int32_t>(DXGI_STATUS_OCCLUDED),
                                                   std::memory_order_relaxed);
    auto occluded = clearAndPresent(device, *swapchain, 0.5F);
    REQUIRE(occluded.has_value());
    CHECK(*occluded == rachis::PresentStatus::occluded);
    auto timing = queryTiming(device, *swapchain);
    CHECK(timing.presentationSupported);
    CHECK(timing.presentationGeneration == 1);
    CHECK(timing.submittedPresentationSequence == 0);
    CHECK(timing.displayedPresentationSequence == 0);

    auto presented = clearAndPresent(device, *swapchain, 0.25F);
    REQUIRE(presented.has_value());
    CHECK(*presented == rachis::PresentStatus::presented);
    timing = queryTiming(device, *swapchain);
    CHECK(timing.submittedPresentationSequence == 1);

    REQUIRE(device.waitIdle().has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
}

TEST_CASE("D3D12 displayed progress requires a DXGI sample and recovers after missing statistics",
          "[size:medium][rhi][presentation][d3d12][warp]")
{
    using Statistics = rachis::test::ScopedD3D12FrameStatistics;
    Statistics statistics(Statistics::Source::zero);
    const TestWindow window(false);
    REQUIRE(window.get() != nullptr);
    auto created = rachis::createDevice(rachis::test::d3d12WarpDeviceDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    auto surface = device.createSurface(rachis::SurfaceDesc{.windowHandle = window.get()});
    REQUIRE(surface.has_value());
    auto swapchain = device.createSwapchain(rachis::SwapchainDesc{
        .surface = *surface,
        .width = windowWidth,
        .height = windowHeight,
        .imageCount = 3,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
    });
    REQUIRE(swapchain.has_value());
    REQUIRE(queryTiming(device, *swapchain).submittedPresentationSequence == 0);

    for (std::uint64_t frame = 1; frame <= 2; ++frame)
    {
        auto presented = clearAndPresent(device, *swapchain, 0.5F);
        REQUIRE(presented.has_value());
        REQUIRE(*presented == rachis::PresentStatus::presented);
    }
    REQUIRE(device.waitIdle().has_value());
    for (const auto source : {Statistics::Source::zero, Statistics::Source::disjoint})
    {
        statistics.source(source);
        const auto timing = queryTiming(device, *swapchain);
        CHECK(timing.presentationSupported);
        CHECK(timing.presentationGeneration == 1);
        CHECK(timing.submittedPresentationSequence == 2);
        CHECK(timing.displayedPresentationSequence == 0);
    }

    statistics.source(Statistics::Source::currentPresent);
    auto timing = queryTiming(device, *swapchain);
    CHECK(timing.presentationSupported);
    CHECK(timing.submittedPresentationSequence == 2);
    CHECK(timing.displayedPresentationSequence == 2);

    REQUIRE(device.resizeSwapchain(*swapchain, windowWidth + 40, windowHeight + 30).has_value());
    timing = queryTiming(device, *swapchain);
    CHECK(timing.presentationGeneration == 2);
    CHECK(timing.submittedPresentationSequence == 0);
    CHECK(timing.displayedPresentationSequence == 0);
    auto presented = clearAndPresent(device, *swapchain, 0.25F);
    REQUIRE(presented.has_value());
    REQUIRE(*presented == rachis::PresentStatus::presented);
    timing = queryTiming(device, *swapchain);
    CHECK(timing.submittedPresentationSequence == 1);
    CHECK(timing.displayedPresentationSequence == 1);

    rachis::d3d12::test::nextPresentCountResult().store(static_cast<std::int32_t>(E_FAIL));
    REQUIRE(device.resizeSwapchain(*swapchain, windowWidth, windowHeight).has_value());
    timing = queryTiming(device, *swapchain);
    CHECK_FALSE(timing.presentationSupported);
    CHECK(timing.presentationGeneration == 0);
    CHECK(timing.submittedPresentationSequence == 0);
    CHECK(timing.displayedPresentationSequence == 0);
    REQUIRE(device.resizeSwapchain(*swapchain, windowWidth + 40, windowHeight + 30).has_value());
    CHECK_FALSE(queryTiming(device, *swapchain).presentationSupported);

    REQUIRE(device.waitIdle().has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
    CHECK(device.diagnosticCounts().corruptions == 0);
}

TEST_CASE("D3D12 WARP reports displayed presentation sequences apart from GPU completion",
          "[size:medium][requires:InteractiveDesktop][rhi][presentation][d3d12][warp]")
{
    const TestWindow window(true);
    REQUIRE(window.get() != nullptr);
    auto created = rachis::createDevice(rachis::test::d3d12WarpDeviceDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    checkDisplayedProgress(device, window.get());
}

TEST_CASE("D3D12 hardware reports displayed presentation sequences apart from GPU completion",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareD3D12][rhi][presentation][d3d12]")
{
    const TestWindow window(true);
    REQUIRE(window.get() != nullptr);
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .backend = rachis::Backend::d3d12,
        .adapterPreference = rachis::AdapterPreference::highPerformance,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    const auto adapter = rachis::test::requireAdapterInfo(device);
    CAPTURE(adapter.name);
    REQUIRE(adapter.backend == rachis::Backend::d3d12);
    REQUIRE(adapter.type != rachis::AdapterType::software);
    checkDisplayedProgress(device, window.get());
}
