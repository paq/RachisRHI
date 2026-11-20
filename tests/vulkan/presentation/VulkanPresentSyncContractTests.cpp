#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include "Internal/TimelinePointAccess.h"
#include "Internal/VulkanDevice.h"
#include "Vulkan/VulkanTestHooks.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <Windows.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace
{

    class TestWindow final
    {
    public:
        explicit TestWindow(bool visible)
            : window_(CreateWindowExW(0,
                                      L"STATIC",
                                      L"Rachis Vulkan present sync",
                                      WS_OVERLAPPEDWINDOW,
                                      CW_USEDEFAULT,
                                      CW_USEDEFAULT,
                                      96,
                                      64,
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

    [[nodiscard]] rachis::DeviceDesc vulkanDesc(std::atomic_uint64_t* errors) noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
            .enableDeviceFault = true,
            .diagnosticCallback = &captureErrors,
            .diagnosticUserData = errors,
        };
    }

    [[nodiscard]] rachis::TextureBarrier toRenderTarget(const rachis::AcquiredSwapchainImage& acquired) noexcept
    {
        return rachis::TextureBarrier{
            .texture = acquired.texture,
            .stagesBefore = acquired.initialStages,
            .stagesAfter = rachis::PipelineStage::renderTarget,
            .accessBefore = acquired.initialAccess,
            .accessAfter = rachis::Access::renderTarget,
            .layoutBefore = acquired.initialLayout,
            .layoutAfter = rachis::TextureLayout::renderTarget,
        };
    }

    [[nodiscard]] rachis::Status recordFrame(rachis::Device& device,
                                             rachis::CommandBuffer& command,
                                             const rachis::AcquiredSwapchainImage& acquired)
    {
        if (auto toColor = device.textureBarrier(command, toRenderTarget(acquired)); !toColor)
        {
            return toColor;
        }
        const std::array attachments{
            rachis::ColorAttachment{
                .view = acquired.renderTargetView,
                .loadOperation = rachis::LoadOperation::clear,
                .clearValue = {0.1F, 0.2F, 0.7F, 1.0F},
            },
        };
        if (auto began = device.beginRendering(command, {.colorAttachments = attachments}); !began)
        {
            return began;
        }
        if (auto ended = device.endRendering(command); !ended)
        {
            return ended;
        }
        return device.textureBarrier(command,
                                     {
                                         .texture = acquired.texture,
                                         .stagesBefore = rachis::PipelineStage::renderTarget,
                                         .stagesAfter = acquired.requiredFinalStages,
                                         .accessBefore = rachis::Access::renderTarget,
                                         .accessAfter = acquired.requiredFinalAccess,
                                         .layoutBefore = rachis::TextureLayout::renderTarget,
                                         .layoutAfter = acquired.requiredFinalLayout,
                                     });
    }

    struct PresentFixture final
    {
        std::atomic_uint64_t errors{0};
        rachis::TextureUsage usage = rachis::TextureUsage::renderTarget;
        rachis::Device device;
        TestWindow window;
        rachis::SurfaceHandle surface{};
        rachis::SwapchainHandle swapchain{};
        std::uint32_t imageCount = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        rachis::Format format = rachis::Format::bgra8Unorm;
        rachis::SurfaceColorSpace colorSpace = rachis::SurfaceColorSpace::srgbNonlinear;

        explicit PresentFixture(bool visible,
                                std::uint32_t requestedImages,
                                rachis::TextureUsage swapchainUsage = rachis::TextureUsage::renderTarget)
            : usage(swapchainUsage), window(visible)
        {
            auto created = rachis::createDevice(vulkanDesc(&errors));
            if (rachis::test::skipWithoutCreatedDevice(created))
            {
                error = created.error();
                return;
            }
            device = std::move(*created);
            if (window.get() == nullptr)
            {
                error = rachis::Error{rachis::ErrorCode::nativeFailure, 0, "Failed to create the present-sync HWND."};
                return;
            }
            auto createdSurface = device.createSurface({
                .kind = rachis::SurfaceKind::win32,
                .windowHandle = window.get(),
                .label = "Vulkan present sync",
            });
            if (!createdSurface)
            {
                error = createdSurface.error();
                return;
            }
            surface = *createdSurface;
            auto capabilities = device.querySurfaceCapabilities(surface);
            if (!capabilities)
            {
                error = capabilities.error();
                return;
            }
            if ((capabilities->usages & usage) != usage)
            {
                error = rachis::Error{
                    rachis::ErrorCode::unsupported, 0, "The surface does not support the requested usages."};
                return;
            }
            width = capabilities->preferredWidth == 0 ? 96 : capabilities->preferredWidth;
            height = capabilities->preferredHeight == 0 ? 64 : capabilities->preferredHeight;
            const auto formatChoice = capabilities->formats.front();
            format = formatChoice.format;
            colorSpace = formatChoice.colorSpace;
            imageCount =
                (std::min)(capabilities->maxImageCount, (std::max)(capabilities->minImageCount, requestedImages));
            if (requestedImages == capabilities->minImageCount)
            {
                imageCount = capabilities->minImageCount;
            }
            auto createdSwapchain = device.createSwapchain({
                .surface = surface,
                .width = width,
                .height = height,
                .imageCount = imageCount,
                .format = format,
                .colorSpace = colorSpace,
                .usage = usage,
                .presentMode = rachis::PresentMode::fifo,
                .label = "Vulkan present-sync swapchain",
            });
            if (!createdSwapchain)
            {
                error = createdSwapchain.error();
                return;
            }
            swapchain = *createdSwapchain;
            ready = true;
        }

        ~PresentFixture()
        {
            if (swapchain.isValid())
            {
                static_cast<void>(device.cancelSwapchainImage(swapchain));
                static_cast<void>(device.destroySwapchain(swapchain));
            }
            if (surface.isValid())
            {
                static_cast<void>(device.destroySurface(surface));
            }
        }

        PresentFixture(const PresentFixture&) = delete;
        PresentFixture& operator=(const PresentFixture&) = delete;
        PresentFixture(PresentFixture&&) = delete;
        PresentFixture& operator=(PresentFixture&&) = delete;

        [[nodiscard]] rachis::Result<rachis::TimelinePoint> drawAcquired(const rachis::AcquiredSwapchainImage& acquired)
        {
            auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
            if (!command)
            {
                return std::unexpected(std::move(command.error()));
            }
            if (auto recorded = recordFrame(device, *command, acquired); !recorded)
            {
                return std::unexpected(std::move(recorded.error()));
            }
            return device.submit(std::move(*command));
        }

        [[nodiscard]] rachis::Status recreateSwapchain()
        {
            if (swapchain.isValid())
            {
                static_cast<void>(device.cancelSwapchainImage(swapchain));
                if (auto destroyed = device.destroySwapchain(swapchain); !destroyed)
                {
                    return destroyed;
                }
                swapchain = {};
            }
            auto createdSwapchain = device.createSwapchain({
                .surface = surface,
                .width = width,
                .height = height,
                .imageCount = imageCount,
                .format = format,
                .colorSpace = colorSpace,
                .usage = usage,
                .presentMode = rachis::PresentMode::fifo,
                .label = "Vulkan present-sync swapchain",
            });
            if (!createdSwapchain)
            {
                return std::unexpected(std::move(createdSwapchain.error()));
            }
            swapchain = *createdSwapchain;
            return {};
        }

        bool ready = false;
        rachis::Error error{};
    };

    struct ClearAcquireFailureInjections final
    {
        ClearAcquireFailureInjections() = default;
        ~ClearAcquireFailureInjections()
        {
            rachis::vulkan::test::nextGraphicsPresentationSyncResult().store(0, std::memory_order_relaxed);
            rachis::vulkan::test::nextPresentFenceWaitResult().store(0, std::memory_order_relaxed);
        }

        ClearAcquireFailureInjections(const ClearAcquireFailureInjections&) = delete;
        ClearAcquireFailureInjections& operator=(const ClearAcquireFailureInjections&) = delete;
        ClearAcquireFailureInjections(ClearAcquireFailureInjections&&) = delete;
        ClearAcquireFailureInjections& operator=(ClearAcquireFailureInjections&&) = delete;
    };

    [[nodiscard]] std::uint32_t semaphoreCount(rachis::SwapchainHandle swapchain)
    {
        auto snapshot = rachis::vulkan::queryVulkanPresentationSync(swapchain);
        REQUIRE(snapshot);
        CHECK_FALSE(snapshot->acquired);
        return snapshot->unusedAcquireSemaphores + snapshot->assignedAcquireSemaphores;
    }

    [[nodiscard]] std::vector<std::byte> loadSpirv(const char* name)
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

    struct SampledSwapchainPipeline final
    {
        rachis::Device& device;
        rachis::ShaderModuleHandle shader{};
        rachis::BindingLayoutHandle layout{};
        rachis::PipelineLayoutHandle pipelineLayout{};
        rachis::PipelineHandle pipeline{};

        SampledSwapchainPipeline(rachis::Device& owner,
                                 rachis::BindingFlag textureFlags,
                                 std::span<const rachis::BindingLayoutHandle> laterSets = {})
            : device(owner)
        {
            const auto bytecode = loadSpirv("VulkanSampledTexture.spv");
            REQUIRE_FALSE(bytecode.empty());
            const std::array requirements{
                rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::sampledTexture, 1},
                rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::sampler, 1},
                rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::storageBuffer, 1},
            };
            auto createdShader = device.createShaderModule({
                .stage = rachis::ShaderStage::compute,
                .bytecode = bytecode,
                .entryPoint = "main",
                .label = "Vulkan swapchain sampling shader",
                .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, requirements),
            });
            REQUIRE(createdShader.has_value());
            shader = *createdShader;
            const std::array entries{
                rachis::BindingLayoutEntry{
                    0, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, 1, textureFlags},
                rachis::BindingLayoutEntry{1, rachis::BindingType::sampler, rachis::ShaderStage::compute, 1},
                rachis::BindingLayoutEntry{2, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
            };
            auto createdLayout =
                device.createBindingLayout({.entries = entries, .label = "Vulkan swapchain sampling layout"});
            REQUIRE(createdLayout.has_value());
            layout = *createdLayout;
            std::vector<rachis::BindingLayoutHandle> setLayouts{layout};
            setLayouts.insert(setLayouts.end(), laterSets.begin(), laterSets.end());
            auto createdPipelineLayout = device.createPipelineLayout({.bindingLayouts = setLayouts});
            REQUIRE(createdPipelineLayout.has_value());
            pipelineLayout = *createdPipelineLayout;
            auto createdPipeline = device.createComputePipeline({
                .layout = pipelineLayout,
                .computeShader = shader,
                .label = "Vulkan swapchain sampling pipeline",
            });
            REQUIRE(createdPipeline.has_value());
            pipeline = *createdPipeline;
        }

        SampledSwapchainPipeline(const SampledSwapchainPipeline&) = delete;
        SampledSwapchainPipeline& operator=(const SampledSwapchainPipeline&) = delete;
        SampledSwapchainPipeline(SampledSwapchainPipeline&&) = delete;
        SampledSwapchainPipeline& operator=(SampledSwapchainPipeline&&) = delete;

        ~SampledSwapchainPipeline()
        {
            static_cast<void>(device.destroyPipeline(pipeline));
            static_cast<void>(device.destroyPipelineLayout(pipelineLayout));
            static_cast<void>(device.destroyBindingLayout(layout));
            static_cast<void>(device.destroyShaderModule(shader));
        }

        [[nodiscard]] rachis::Status bind(rachis::CommandBuffer& command, rachis::BindingSetHandle set)
        {
            if (auto bound = device.bindComputePipeline(command, pipeline); !bound)
            {
                return bound;
            }
            return device.bindComputeBindingSet(command, 0, set);
        }

        [[nodiscard]] rachis::Status bindAll(rachis::CommandBuffer& command,
                                             std::span<const rachis::BindingSetHandle> sets)
        {
            if (auto bound = device.bindComputePipeline(command, pipeline); !bound)
            {
                return bound;
            }
            return device.bindComputeBindingSets(command, 0, sets);
        }
    };

    struct SampledSwapchainSets final
    {
        rachis::Device& device;
        rachis::BindingPoolHandle pool{};
        rachis::SamplerHandle sampler{};
        rachis::BufferHandle output{};

        SampledSwapchainSets(rachis::Device& owner, std::uint32_t maxSets) : device(owner)
        {
            const std::array poolSizes{
                rachis::BindingPoolSize{rachis::BindingType::sampledTexture, maxSets},
                rachis::BindingPoolSize{rachis::BindingType::sampler, maxSets},
                rachis::BindingPoolSize{rachis::BindingType::storageBuffer, maxSets},
            };
            auto createdPool = device.createBindingPool({.maxBindingSets = maxSets, .sizes = poolSizes});
            REQUIRE(createdPool.has_value());
            pool = *createdPool;
            auto createdSampler = device.createSampler({.label = "Vulkan swapchain sampling sampler"});
            REQUIRE(createdSampler.has_value());
            sampler = *createdSampler;
            auto createdOutput = device.createBuffer({
                .size = 4 * sizeof(float),
                .usage = rachis::BufferUsage::storage,
                .label = "Vulkan swapchain sampling output",
            });
            REQUIRE(createdOutput.has_value());
            output = *createdOutput;
        }

        SampledSwapchainSets(const SampledSwapchainSets&) = delete;
        SampledSwapchainSets& operator=(const SampledSwapchainSets&) = delete;
        SampledSwapchainSets(SampledSwapchainSets&&) = delete;
        SampledSwapchainSets& operator=(SampledSwapchainSets&&) = delete;

        ~SampledSwapchainSets()
        {
            static_cast<void>(device.destroyBuffer(output));
            static_cast<void>(device.destroySampler(sampler));
            static_cast<void>(device.destroyBindingPool(pool));
        }

        [[nodiscard]] rachis::BindingSetHandle create(rachis::BindingLayoutHandle layout,
                                                      rachis::TextureViewHandle view)
        {
            const std::array writes{
                rachis::BindingWrite{0, 0, view},
                rachis::BindingWrite{1, 0, sampler},
                rachis::BindingWrite{2, 0, rachis::BufferBinding{output, 0, 4 * sizeof(float), sizeof(float)}},
            };
            auto set = device.createBindingSet({.layout = layout, .writes = writes, .pool = pool});
            REQUIRE(set.has_value());
            return *set;
        }
    };

    [[nodiscard]] rachis::BindingLayoutHandle createSampledSwapchainLayout(rachis::Device& device)
    {
        const std::array entries{
            rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, 1},
            rachis::BindingLayoutEntry{1, rachis::BindingType::sampler, rachis::ShaderStage::compute, 1},
            rachis::BindingLayoutEntry{2, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
        };
        auto layout = device.createBindingLayout({.entries = entries, .label = "Vulkan later swapchain set layout"});
        REQUIRE(layout.has_value());
        return *layout;
    }

    void presentOne(PresentFixture& fixture)
    {
        auto acquired = fixture.device.acquireNextImage(fixture.swapchain);
        REQUIRE(acquired);
        auto submitted = fixture.drawAcquired(*acquired);
        REQUIRE(submitted);
        auto presented = fixture.device.present(fixture.swapchain, *submitted);
        REQUIRE(presented);
        CHECK(*presented == rachis::PresentStatus::presented);
    }

    void requireAcquireFailure(PresentFixture& fixture,
                               std::atomic_int32_t& injection,
                               const char* operation,
                               std::uint32_t expectedSemaphores)
    {
        injection.store(static_cast<std::int32_t>(VK_ERROR_OUT_OF_HOST_MEMORY), std::memory_order_relaxed);
        auto acquired = fixture.device.acquireNextImage(fixture.swapchain);
        injection.store(0, std::memory_order_relaxed);
        REQUIRE_FALSE(acquired);
        CHECK(acquired.error().code == rachis::ErrorCode::outOfMemory);
        CHECK(acquired.error().nativeCode == static_cast<std::int64_t>(VK_ERROR_OUT_OF_HOST_MEMORY));
        CHECK(acquired.error().message.contains(operation));
        auto snapshot = rachis::vulkan::queryVulkanPresentationSync(fixture.swapchain);
        REQUIRE(snapshot);
        CHECK_FALSE(snapshot->acquired);
        CHECK(snapshot->unusedAcquireSemaphores + snapshot->assignedAcquireSemaphores == expectedSemaphores);
    }

    void requireFenceAcquireFailure(PresentFixture& fixture, std::uint32_t expectedSemaphores)
    {
        auto sync = rachis::vulkan::queryVulkanPresentationSync(fixture.swapchain);
        REQUIRE(sync);
        REQUIRE(sync->strategy == rachis::VulkanPresentCompletionStrategy::swapchainMaintenancePresentFence);
        const int bound = static_cast<int>(sync->imageCount) * 3 + 2;
        for (int attempt = 0; attempt < bound; ++attempt)
        {
            INFO("fence failure attempt " << attempt);
            rachis::vulkan::test::nextPresentFenceWaitResult().store(
                static_cast<std::int32_t>(VK_ERROR_OUT_OF_HOST_MEMORY), std::memory_order_relaxed);
            auto acquired = fixture.device.acquireNextImage(fixture.swapchain);
            if (!acquired && acquired.error().code == rachis::ErrorCode::notReady)
            {
                continue;
            }
            if (acquired)
            {
                rachis::vulkan::test::nextPresentFenceWaitResult().store(0, std::memory_order_relaxed);
                auto submitted = fixture.drawAcquired(*acquired);
                REQUIRE(submitted);
                auto presented = fixture.device.present(fixture.swapchain, *submitted);
                REQUIRE(presented);
                continue;
            }
            rachis::vulkan::test::nextPresentFenceWaitResult().store(0, std::memory_order_relaxed);
            CHECK(acquired.error().code == rachis::ErrorCode::outOfMemory);
            CHECK(acquired.error().message.contains("vkWaitForFences"));
            auto snapshot = rachis::vulkan::queryVulkanPresentationSync(fixture.swapchain);
            REQUIRE(snapshot);
            CHECK_FALSE(snapshot->acquired);
            CHECK(snapshot->unusedAcquireSemaphores + snapshot->assignedAcquireSemaphores == expectedSemaphores);
            return;
        }
        FAIL("present-fence acquire failure was not reached");
    }

} // namespace

TEST_CASE("Vulkan present sync covers min-image-count and injected acquire/present failures",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][present]")
{
    PresentFixture fixture(true, 1);
    REQUIRE(fixture.ready);

    auto capabilities = fixture.device.querySurfaceCapabilities(fixture.surface);
    REQUIRE(capabilities);
    CHECK(fixture.imageCount == capabilities->minImageCount);

    for (int frame = 0; frame < 3; ++frame)
    {
        auto frameImage = fixture.device.acquireNextImage(fixture.swapchain);
        REQUIRE(frameImage);
        auto frameSubmit = fixture.drawAcquired(*frameImage);
        REQUIRE(frameSubmit);
        auto framePresent = fixture.device.present(fixture.swapchain, *frameSubmit);
        REQUIRE(framePresent);
        CHECK(*framePresent == rachis::PresentStatus::presented);
    }

    auto acquired = fixture.device.acquireNextImage(fixture.swapchain);
    REQUIRE(acquired);
    auto secondAcquire = fixture.device.acquireNextImage(fixture.swapchain);
    REQUIRE_FALSE(secondAcquire.has_value());
    CHECK(secondAcquire.error().code == rachis::ErrorCode::invalidState);

    auto badTimeline = fixture.device.present(fixture.swapchain, rachis::TimelinePoint{});
    REQUIRE_FALSE(badTimeline.has_value());
    CHECK(badTimeline.error().code == rachis::ErrorCode::invalidArgument);
    auto stillAcquired = rachis::vulkan::queryVulkanPresentationSync(fixture.swapchain);
    REQUIRE(stillAcquired);
    CHECK(stillAcquired->acquired);

    auto foreign = rachis::detail::TimelinePointAccess::make(rachis::QueueType::graphics, 1, 0xFFFF'FFFFULL);
    auto foreignPresent = fixture.device.present(fixture.swapchain, foreign);
    REQUIRE_FALSE(foreignPresent.has_value());
    CHECK(foreignPresent.error().code == rachis::ErrorCode::invalidArgument);

    REQUIRE(fixture.device.cancelSwapchainImage(fixture.swapchain).has_value());
    auto cancelledPresent = fixture.device.present(fixture.swapchain, rachis::TimelinePoint{});
    REQUIRE_FALSE(cancelledPresent.has_value());
    CHECK(cancelledPresent.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(fixture.recreateSwapchain().has_value());

    auto afterCancel = fixture.device.acquireNextImage(fixture.swapchain);
    REQUIRE(afterCancel);
    auto submitted = fixture.drawAcquired(*afterCancel);
    REQUIRE(submitted);
    auto presented = fixture.device.present(fixture.swapchain, *submitted);
    REQUIRE(presented);
    CHECK(*presented == rachis::PresentStatus::presented);

    auto doublePresent = fixture.device.present(fixture.swapchain, *submitted);
    REQUIRE_FALSE(doublePresent.has_value());
    CHECK(doublePresent.error().code == rachis::ErrorCode::invalidState);

    auto acquiredForDestroy = fixture.device.acquireNextImage(fixture.swapchain);
    REQUIRE(acquiredForDestroy);
    auto destroyWhileAcquired = fixture.device.destroySwapchain(fixture.swapchain);
    REQUIRE_FALSE(destroyWhileAcquired.has_value());
    CHECK(destroyWhileAcquired.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(fixture.device.cancelSwapchainImage(fixture.swapchain).has_value());
    REQUIRE(fixture.device.destroySwapchain(fixture.swapchain).has_value());
    fixture.swapchain = {};
    CHECK(fixture.device.diagnosticCounts().errors == 0);
}

TEST_CASE("injected Vulkan present loss is labeled and terminal",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][present][loss]")
{
    PresentFixture fixture(true, 2);
    REQUIRE(fixture.ready);
    REQUIRE(rachis::test::requireAdapterInfo(fixture.device).backend == rachis::Backend::vulkan);

    auto acquired = fixture.device.acquireNextImage(fixture.swapchain);
    REQUIRE(acquired);
    auto submitted = fixture.drawAcquired(*acquired);
    REQUIRE(submitted);

    rachis::vulkan::test::nextPresentResult().store(static_cast<std::int32_t>(VK_ERROR_DEVICE_LOST),
                                                    std::memory_order_relaxed);
    auto failedPresent = fixture.device.present(fixture.swapchain, *submitted);
    REQUIRE_FALSE(failedPresent.has_value());
    CHECK(failedPresent.error().code == rachis::ErrorCode::deviceLost);
    CHECK(failedPresent.error().nativeCode == static_cast<std::int64_t>(VK_ERROR_DEVICE_LOST));
    CHECK(failedPresent.error().message.contains("injected"));
    CHECK(fixture.device.operationalState() == rachis::DeviceOperationalState::lost);

    const auto firstLoss = fixture.device.deviceLossError();
    REQUIRE(firstLoss.has_value());
    CHECK(firstLoss->message == failedPresent.error().message);
    auto idle = fixture.device.waitIdle();
    REQUIRE_FALSE(idle.has_value());
    CHECK(idle.error().message == firstLoss->message);
    auto report = fixture.device.captureDeviceFaultReport();
    REQUIRE(report);
    REQUIRE_FALSE(report->breadcrumbs.empty());
    CHECK(report->breadcrumbs.front().commandListLabel == "injected device loss");
    fixture.device = {};
    fixture.swapchain = {};
    fixture.surface = {};
    CHECK(fixture.errors.load() == 1);
}

TEST_CASE("Vulkan destroys a swapchain and its surface after injected loss with an acquired or cancelled image",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][present][loss]")
{
    PresentFixture fixture(true, 2);
    REQUIRE(fixture.ready);
    auto presentedImage = fixture.device.acquireNextImage(fixture.swapchain);
    REQUIRE(presentedImage);
    auto rendered = fixture.drawAcquired(*presentedImage);
    REQUIRE(rendered);
    REQUIRE(fixture.device.present(fixture.swapchain, *rendered));
    auto acquired = fixture.device.acquireNextImage(fixture.swapchain);
    REQUIRE(acquired);

    SECTION("injected loss while the image is acquired")
    {
        REQUIRE_FALSE(rachis::vulkan::injectVulkanDeviceLoss());
    }
    SECTION("injected loss, then the image is cancelled")
    {
        REQUIRE_FALSE(rachis::vulkan::injectVulkanDeviceLoss());
        auto cancelled = fixture.device.cancelSwapchainImage(fixture.swapchain);
        INFO((cancelled ? std::string{} : cancelled.error().message));
        REQUIRE(cancelled.has_value());
        CHECK_FALSE(fixture.device.hasAcquiredSwapchainImage(fixture.swapchain));
    }
    REQUIRE(fixture.device.operationalState() == rachis::DeviceOperationalState::lost);

    auto destroyed = fixture.device.destroySwapchain(fixture.swapchain);
    INFO((destroyed ? std::string{} : destroyed.error().message));
    REQUIRE(destroyed.has_value());
    auto destroyedAgain = fixture.device.destroySwapchain(fixture.swapchain);
    REQUIRE_FALSE(destroyedAgain.has_value());
    CHECK(destroyedAgain.error().code == rachis::ErrorCode::invalidArgument);
    fixture.swapchain = {};
    REQUIRE(fixture.device.destroySurface(fixture.surface).has_value());
    fixture.surface = {};
    fixture.device = {};
    CHECK(fixture.errors.load() == 1);
}

TEST_CASE("Vulkan cancelled acquisitions remain reusable without exhausting WSI images",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][present]")
{
    PresentFixture fixture(true, 2);
    REQUIRE(fixture.ready);
    for (int frame = 0; frame < 16; ++frame)
    {
        auto acquired = fixture.device.acquireNextImage(fixture.swapchain);
        REQUIRE(acquired);
        REQUIRE(fixture.device.cancelSwapchainImage(fixture.swapchain));
        CHECK_FALSE(fixture.device.cancelSwapchainImage(fixture.swapchain));
    }
    auto acquired = fixture.device.acquireNextImage(fixture.swapchain);
    REQUIRE(acquired);
    auto submitted = fixture.drawAcquired(*acquired);
    REQUIRE(submitted);
    REQUIRE(fixture.device.present(fixture.swapchain, *submitted));
    REQUIRE(fixture.device.waitIdle());
    CHECK(fixture.errors.load() == 0);
}

TEST_CASE("Vulkan swapchain cancellation resumes after command reset or abandonment",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][present][lifetime]")
{
    PresentFixture fixture(true, 2);
    REQUIRE(fixture.ready);
    auto& device = fixture.device;

    auto resetImage = device.acquireNextImage(fixture.swapchain);
    REQUIRE(resetImage);
    auto resetPool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Vulkan presentation reset pool",
        .flags = rachis::CommandPoolFlag::allowIndividualReset,
    });
    REQUIRE(resetPool);
    {
        auto resetCommand = device.acquireCommandBuffer(*resetPool);
        REQUIRE(resetCommand);
        REQUIRE(device.textureBarrier(*resetCommand, toRenderTarget(*resetImage)).has_value());
        auto rejected = device.cancelSwapchainImage(fixture.swapchain);
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
        CHECK(device.hasAcquiredSwapchainImage(fixture.swapchain));
        REQUIRE(device.resetCommandBuffer(*resetCommand).has_value());
        REQUIRE(device.cancelSwapchainImage(fixture.swapchain).has_value());
    }
    REQUIRE(device.destroyCommandPool(*resetPool).has_value());

    auto abandonedImage = device.acquireNextImage(fixture.swapchain);
    REQUIRE(abandonedImage);
    {
        auto abandonedCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(abandonedCommand);
        REQUIRE(device.textureBarrier(*abandonedCommand, toRenderTarget(*abandonedImage)).has_value());
        REQUIRE_FALSE(device.cancelSwapchainImage(fixture.swapchain).has_value());
    }
    REQUIRE(device.cancelSwapchainImage(fixture.swapchain).has_value());

    auto unretainedImage = device.acquireNextImage(fixture.swapchain);
    REQUIRE(unretainedImage);
    {
        auto unretainedCommand = rachis::test::acquireCommandBuffer(
            device, rachis::QueueType::graphics, rachis::CommandPoolFlag::unretainedResources);
        REQUIRE(unretainedCommand);
        REQUIRE(recordFrame(device, *unretainedCommand, *unretainedImage).has_value());
        REQUIRE_FALSE(device.cancelSwapchainImage(fixture.swapchain).has_value());
    }
    REQUIRE(device.cancelSwapchainImage(fixture.swapchain).has_value());

    auto reused = device.acquireNextImage(fixture.swapchain);
    REQUIRE(reused);
    auto submitted = fixture.drawAcquired(*reused);
    REQUIRE(submitted);
    auto presented = device.present(fixture.swapchain, *submitted);
    REQUIRE(presented);
    CHECK(*presented == rachis::PresentStatus::presented);
    REQUIRE(device.waitIdle().has_value());
    CHECK(fixture.errors.load() == 0);
}

TEST_CASE("Vulkan cancel keeps an image that unsubmitted work records alive across resize",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][present][lifetime]")
{
    PresentFixture fixture(true, 3);
    REQUIRE(fixture.ready);
    auto& device = fixture.device;

    auto acquired = device.acquireNextImage(fixture.swapchain);
    REQUIRE(acquired);
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(recordFrame(device, *command, *acquired).has_value());

    auto cancelled = device.cancelSwapchainImage(fixture.swapchain);
    REQUIRE_FALSE(cancelled.has_value());
    CHECK(cancelled.error().code == rachis::ErrorCode::invalidState);
    CHECK(cancelled.error().message.contains("references the acquired swapchain image"));
    CHECK(device.hasAcquiredSwapchainImage(fixture.swapchain));
    auto resizedWhileAcquired = device.resizeSwapchain(fixture.swapchain, fixture.width, fixture.height);
    REQUIRE_FALSE(resizedWhileAcquired.has_value());
    CHECK(resizedWhileAcquired.error().code == rachis::ErrorCode::invalidState);

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    auto cancelledAfterSubmit = device.cancelSwapchainImage(fixture.swapchain);
    REQUIRE_FALSE(cancelledAfterSubmit.has_value());
    CHECK(cancelledAfterSubmit.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.wait(*submitted).has_value());
    auto presented = device.present(fixture.swapchain, *submitted);
    REQUIRE(presented);
    CHECK(*presented == rachis::PresentStatus::presented);

    REQUIRE(device.resizeSwapchain(fixture.swapchain, fixture.width, fixture.height).has_value());
    auto resized = device.acquireNextImage(fixture.swapchain);
    REQUIRE(resized);
    CHECK(resized->generation != acquired->generation);
    auto resizedSubmit = fixture.drawAcquired(*resized);
    REQUIRE(resizedSubmit);
    auto resizedPresent = device.present(fixture.swapchain, *resizedSubmit);
    REQUIRE(resizedPresent);
    CHECK(*resizedPresent == rachis::PresentStatus::presented);
    REQUIRE(device.waitIdle().has_value());
    CHECK(device.operationalState() == rachis::DeviceOperationalState::active);
    CHECK(device.diagnosticCounts().errors == 0);
    CHECK(fixture.errors.load() == 0);
}

TEST_CASE("Vulkan present and recording reject unsubmitted or unacquired swapchain image work",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][present][lifetime]")
{
    PresentFixture fixture(true, 2);
    REQUIRE(fixture.ready);
    auto& device = fixture.device;

    auto acquired = device.acquireNextImage(fixture.swapchain);
    REQUIRE(acquired);
    auto recorded = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(recorded);
    REQUIRE(recordFrame(device, *recorded, *acquired).has_value());
    auto unrelated = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(unrelated);
    auto unrelatedPoint = device.submit(std::move(*unrelated));
    REQUIRE(unrelatedPoint);

    auto rejected = device.present(fixture.swapchain, *unrelatedPoint);
    REQUIRE_FALSE(rejected.has_value());
    CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
    CHECK(device.hasAcquiredSwapchainImage(fixture.swapchain));

    auto submitted = device.submit(std::move(*recorded));
    REQUIRE(submitted);
    auto presented = device.present(fixture.swapchain, *submitted);
    REQUIRE(presented);
    CHECK(*presented == rachis::PresentStatus::presented);

    auto late = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(late);
    auto lateBarrier = device.textureBarrier(*late, toRenderTarget(*acquired));
    REQUIRE_FALSE(lateBarrier.has_value());
    CHECK(lateBarrier.error().code == rachis::ErrorCode::invalidState);
    const std::array lateAttachments{
        rachis::ColorAttachment{
            .view = acquired->renderTargetView,
            .loadOperation = rachis::LoadOperation::clear,
        },
    };
    auto lateRendering = device.beginRendering(*late, {.colorAttachments = lateAttachments});
    REQUIRE_FALSE(lateRendering.has_value());
    CHECK(lateRendering.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.submit(std::move(*late)).has_value());

    auto next = device.acquireNextImage(fixture.swapchain);
    REQUIRE(next);
    auto nextSubmit = fixture.drawAcquired(*next);
    REQUIRE(nextSubmit);
    auto nextPresent = device.present(fixture.swapchain, *nextSubmit);
    REQUIRE(nextPresent);
    CHECK(*nextPresent == rachis::PresentStatus::presented);
    REQUIRE(device.waitIdle().has_value());
    CHECK(fixture.errors.load() == 0);
}

TEST_CASE("Vulkan binding sets that hold a swapchain image view keep the image recorded until submission",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][present][lifetime]")
{
    PresentFixture fixture(true, 2, rachis::TextureUsage::renderTarget | rachis::TextureUsage::sampled);
    if (!fixture.ready && fixture.error.code == rachis::ErrorCode::unsupported)
    {
        SKIP("The surface does not support sampled swapchain images.");
    }
    REQUIRE(fixture.ready);
    auto& device = fixture.device;
    const auto updateAfterBind = rachis::test::requireCapabilities(device).updateAfterBindBindingArrays;
    SampledSwapchainPipeline regular(device, rachis::BindingFlag::none);
    std::optional<SampledSwapchainPipeline> inPlace;
    if (updateAfterBind)
    {
        inPlace.emplace(device, rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind);
    }

    const std::array poolSizes{
        rachis::BindingPoolSize{rachis::BindingType::sampledTexture, 2},
        rachis::BindingPoolSize{rachis::BindingType::sampler, 2},
        rachis::BindingPoolSize{rachis::BindingType::storageBuffer, 2},
    };
    auto pool = device.createBindingPool({.maxBindingSets = 2, .sizes = poolSizes});
    REQUIRE(pool.has_value());
    auto sampler = device.createSampler({.label = "Vulkan swapchain sampling sampler"});
    REQUIRE(sampler.has_value());
    auto output = device.createBuffer({
        .size = 4 * sizeof(float),
        .usage = rachis::BufferUsage::storage,
        .label = "Vulkan swapchain sampling output",
    });
    REQUIRE(output.has_value());
    auto plain = device.createTexture({
        .width = 2,
        .height = 2,
        .format = rachis::Format::r32Float,
        .usage = rachis::TextureUsage::sampled,
        .label = "Vulkan swapchain sampling replacement",
    });
    REQUIRE(plain.has_value());
    auto plainView = device.createTextureView({.texture = *plain, .usage = rachis::TextureViewUsage::sampled});
    REQUIRE(plainView.has_value());

    auto acquired = device.acquireNextImage(fixture.swapchain);
    REQUIRE(acquired);
    auto swapchainView =
        device.createTextureView({.texture = acquired->texture, .usage = rachis::TextureViewUsage::sampled});
    REQUIRE(swapchainView.has_value());
    const auto setWith = [&](rachis::BindingLayoutHandle layout, rachis::TextureViewHandle view) {
        const std::array writes{
            rachis::BindingWrite{0, 0, view},
            rachis::BindingWrite{1, 0, *sampler},
            rachis::BindingWrite{2, 0, rachis::BufferBinding{*output, 0, 4 * sizeof(float), sizeof(float)}},
        };
        auto set = device.createBindingSet({.layout = layout, .writes = writes, .pool = *pool});
        REQUIRE(set.has_value());
        return *set;
    };

    const auto set = setWith(regular.layout, *swapchainView);
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        REQUIRE(regular.bind(*command, set).has_value());
        auto cancelled = device.cancelSwapchainImage(fixture.swapchain);
        REQUIRE_FALSE(cancelled.has_value());
        CHECK(cancelled.error().code == rachis::ErrorCode::invalidState);
        CHECK(cancelled.error().message.contains("references the acquired swapchain image"));
    }
    REQUIRE(device.cancelSwapchainImage(fixture.swapchain).has_value());
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        auto unacquired = regular.bind(*command, set);
        REQUIRE_FALSE(unacquired.has_value());
        CHECK(unacquired.error().code == rachis::ErrorCode::invalidState);
    }
    REQUIRE(device.destroyBindingSet(set).has_value());

    if (inPlace.has_value())
    {
        auto reacquired = device.acquireNextImage(fixture.swapchain);
        REQUIRE(reacquired);
        auto reacquiredView =
            device.createTextureView({.texture = reacquired->texture, .usage = rachis::TextureViewUsage::sampled});
        REQUIRE(reacquiredView.has_value());
        const auto table = setWith(inPlace->layout, *reacquiredView);
        const std::array swapchainWrite{rachis::BindingWrite{0, 0, *reacquiredView}};
        auto rejected = device.updateBindingSet(table, swapchainWrite);
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::unsupported);
        {
            auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
            REQUIRE(command);
            REQUIRE(inPlace->bind(*command, table).has_value());
            REQUIRE_FALSE(device.cancelSwapchainImage(fixture.swapchain).has_value());
        }
        const std::array replacement{rachis::BindingWrite{0, 0, *plainView}};
        REQUIRE(device.updateBindingSet(table, replacement).has_value());
        REQUIRE(device.cancelSwapchainImage(fixture.swapchain).has_value());
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        REQUIRE(inPlace->bind(*command, table).has_value());
        REQUIRE(device.submit(std::move(*command)).has_value());
        REQUIRE(device.waitIdle().has_value());
        REQUIRE(device.destroyBindingSet(table).has_value());
        REQUIRE(device.destroyTextureView(*reacquiredView).has_value());
    }

    REQUIRE(device.destroyTextureView(*swapchainView).has_value());
    REQUIRE(device.destroyTextureView(*plainView).has_value());
    REQUIRE(device.destroyTexture(*plain).has_value());
    REQUIRE(device.destroyBuffer(*output).has_value());
    REQUIRE(device.destroySampler(*sampler).has_value());
    REQUIRE(device.destroyBindingPool(*pool).has_value());
    CHECK(fixture.errors.load() == 0);
}

TEST_CASE("Vulkan binding batches that fail after retaining a swapchain image leave the image cancellable",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][present][lifetime]")
{
    PresentFixture fixture(true, 3, rachis::TextureUsage::renderTarget | rachis::TextureUsage::sampled);
    if (!fixture.ready && fixture.error.code == rachis::ErrorCode::unsupported)
    {
        SKIP("The surface does not support sampled swapchain images.");
    }
    REQUIRE(fixture.ready);
    auto& device = fixture.device;
    const auto laterLayout = createSampledSwapchainLayout(device);
    std::vector<rachis::TextureViewHandle> views;
    {
        SampledSwapchainSets sets(device, 3);
        const auto sampledView = [&](const rachis::AcquiredSwapchainImage& image) {
            auto view =
                device.createTextureView({.texture = image.texture, .usage = rachis::TextureViewUsage::sampled});
            REQUIRE(view.has_value());
            views.push_back(*view);
            return *view;
        };
        const auto releaseAcquired = [&] {
            if (device.hasAcquiredSwapchainImage(fixture.swapchain))
            {
                REQUIRE(device.cancelSwapchainImage(fixture.swapchain).has_value());
            }
        };

        const std::array laterSampled{laterLayout};
        SampledSwapchainPipeline twoImages(device, rachis::BindingFlag::none, laterSampled);
        auto earlier = device.acquireNextImage(fixture.swapchain);
        REQUIRE(earlier);
        const auto unacquiredSet = sets.create(laterLayout, sampledView(*earlier));
        auto earlierSubmit = fixture.drawAcquired(*earlier);
        REQUIRE(earlierSubmit);
        REQUIRE(device.present(fixture.swapchain, *earlierSubmit).has_value());
        std::optional<rachis::AcquiredSwapchainImage> other;
        for (std::uint32_t attempt = 0; attempt < fixture.imageCount * 3U && !other.has_value(); ++attempt)
        {
            auto acquired = device.acquireNextImage(fixture.swapchain);
            REQUIRE(acquired);
            if (acquired->imageIndex != earlier->imageIndex)
            {
                other = *acquired;
                continue;
            }
            auto submitted = fixture.drawAcquired(*acquired);
            REQUIRE(submitted);
            REQUIRE(device.present(fixture.swapchain, *submitted).has_value());
        }
        REQUIRE(other.has_value());
        const auto acquiredSet = sets.create(twoImages.layout, sampledView(*other));
        {
            auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
            REQUIRE(command);
            const std::array batch{acquiredSet, unacquiredSet};
            auto rejected = twoImages.bindAll(*command, batch);
            REQUIRE_FALSE(rejected.has_value());
            CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
            CHECK(rejected.error().message.contains("only while it is acquired"));
            CHECK(device.cancelSwapchainImage(fixture.swapchain).has_value());
        }
        releaseAcquired();

        if (rachis::test::requireCapabilities(device).rayQuery)
        {
            auto sizes = device.queryTopLevelAccelerationStructureBuildSizes({.instanceCount = 1});
            REQUIRE(sizes.has_value());
            auto storage = device.createBuffer({
                .size = sizes->accelerationStructureSize,
                .usage = rachis::BufferUsage::accelerationStructureStorage,
                .memoryType = rachis::MemoryType::deviceLocal,
                .label = "Vulkan unbuilt TLAS storage",
            });
            REQUIRE(storage.has_value());
            auto unbuilt = device.createAccelerationStructure({
                .type = rachis::AccelerationStructureType::topLevel,
                .storage = *storage,
                .label = "Vulkan unbuilt TLAS",
            });
            REQUIRE(unbuilt.has_value());
            const std::array topLevelEntries{
                rachis::BindingLayoutEntry{
                    0, rachis::BindingType::accelerationStructure, rachis::ShaderStage::compute, 1},
            };
            auto topLevelLayout =
                device.createBindingLayout({.entries = topLevelEntries, .label = "Vulkan unbuilt TLAS layout"});
            REQUIRE(topLevelLayout.has_value());
            const std::array topLevelWrites{rachis::BindingWrite{.binding = 0, .resource = *unbuilt}};
            auto topLevelSet =
                rachis::test::createBindingSet(device, {.layout = *topLevelLayout, .writes = topLevelWrites});
            REQUIRE(topLevelSet.has_value());
            {
                const std::array laterTopLevel{*topLevelLayout};
                SampledSwapchainPipeline withTopLevel(device, rachis::BindingFlag::none, laterTopLevel);
                auto acquired = device.acquireNextImage(fixture.swapchain);
                REQUIRE(acquired);
                const auto imageSet = sets.create(withTopLevel.layout, sampledView(*acquired));
                {
                    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
                    REQUIRE(command);
                    const std::array batch{imageSet, *topLevelSet};
                    auto rejected = withTopLevel.bindAll(*command, batch);
                    REQUIRE_FALSE(rejected.has_value());
                    CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
                    CHECK(rejected.error().message.contains("no built version"));
                    CHECK(device.cancelSwapchainImage(fixture.swapchain).has_value());
                }
                releaseAcquired();
                REQUIRE(device.destroyBindingSet(imageSet).has_value());
            }
            REQUIRE(device.destroyBindingSet(*topLevelSet).has_value());
            REQUIRE(device.destroyBindingLayout(*topLevelLayout).has_value());
            REQUIRE(device.destroyAccelerationStructure(*unbuilt).has_value());
            REQUIRE(device.destroyBuffer(*storage).has_value());
        }
        REQUIRE(device.destroyBindingSet(acquiredSet).has_value());
        REQUIRE(device.destroyBindingSet(unacquiredSet).has_value());
    }
    for (const auto view : views)
    {
        REQUIRE(device.destroyTextureView(view).has_value());
    }
    REQUIRE(device.destroyBindingLayout(laterLayout).has_value());
    REQUIRE(device.waitIdle().has_value());
    CHECK(fixture.errors.load() == 0);
}

TEST_CASE("Vulkan acquire timeout preserves the semaphore pool and permits a later frame",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][present]")
{
    PresentFixture fixture(true, 2);
    REQUIRE(fixture.ready);
    auto before = rachis::vulkan::queryVulkanPresentationSync(fixture.swapchain);
    REQUIRE(before);
    for (int attempt = 0; attempt < 8; ++attempt)
    {
        rachis::vulkan::test::nextAcquireResult().store(VK_TIMEOUT, std::memory_order_relaxed);
        auto timedOut = fixture.device.acquireNextImage(fixture.swapchain);
        REQUIRE_FALSE(timedOut);
        CHECK(timedOut.error().code == rachis::ErrorCode::notReady);
        auto after = rachis::vulkan::queryVulkanPresentationSync(fixture.swapchain);
        REQUIRE(after);
        CHECK_FALSE(after->acquired);
        CHECK(after->unusedAcquireSemaphores == before->unusedAcquireSemaphores);
        CHECK(after->assignedAcquireSemaphores == before->assignedAcquireSemaphores);
    }
    auto acquired = fixture.device.acquireNextImage(fixture.swapchain);
    REQUIRE(acquired);
    auto submitted = fixture.drawAcquired(*acquired);
    REQUIRE(submitted);
    REQUIRE(fixture.device.present(fixture.swapchain, *submitted));
    CHECK(fixture.errors.load() == 0);
}

TEST_CASE("Vulkan WSI rejection consumes presentation synchronization exactly once",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][present]")
{
    PresentFixture fixture(true, 2);
    REQUIRE(fixture.ready);
    const auto failure = GENERATE(VK_ERROR_OUT_OF_DATE_KHR,
                                  VK_ERROR_SURFACE_LOST_KHR,
                                  VK_ERROR_OUT_OF_HOST_MEMORY,
                                  VK_ERROR_OUT_OF_DEVICE_MEMORY);
    auto acquired = fixture.device.acquireNextImage(fixture.swapchain);
    REQUIRE(acquired);
    auto submitted = fixture.drawAcquired(*acquired);
    REQUIRE(submitted);
    rachis::vulkan::test::nextPresentResult().store(failure, std::memory_order_relaxed);
    auto presented = fixture.device.present(fixture.swapchain, *submitted);
    if (failure == VK_ERROR_OUT_OF_DATE_KHR)
    {
        REQUIRE(presented);
        CHECK(*presented == rachis::PresentStatus::surfaceOutOfDate);
    }
    else
    {
        REQUIRE_FALSE(presented);
        CHECK(presented.error().nativeCode == failure);
    }
    if (failure == VK_ERROR_OUT_OF_HOST_MEMORY || failure == VK_ERROR_OUT_OF_DEVICE_MEMORY)
    {
        REQUIRE(fixture.device.present(fixture.swapchain, *submitted));
    }
    REQUIRE(fixture.device.resizeSwapchain(fixture.swapchain, fixture.width, fixture.height));
    auto retryImage = fixture.device.acquireNextImage(fixture.swapchain);
    REQUIRE(retryImage);
    auto retrySubmit = fixture.drawAcquired(*retryImage);
    REQUIRE(retrySubmit);
    REQUIRE(fixture.device.present(fixture.swapchain, *retrySubmit));
    REQUIRE(fixture.device.waitIdle());
    CHECK(fixture.errors.load() == 0);
}

TEST_CASE("Vulkan acquire failure after vkAcquireNextImageKHR leaves no public image",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][present]")
{
    PresentFixture fixture(true, 2);
    INFO(fixture.error.message);
    REQUIRE(fixture.ready);
    [[maybe_unused]] const ClearAcquireFailureInjections clearInjections;

    const auto semaphores = semaphoreCount(fixture.swapchain);
    requireAcquireFailure(
        fixture, rachis::vulkan::test::nextGraphicsPresentationSyncResult(), "vkQueueSubmit2", semaphores);
    presentOne(fixture);

    requireAcquireFailure(fixture,
                          rachis::vulkan::test::nextGraphicsPresentationSyncResult(),
                          "vkQueueSubmit2",
                          semaphoreCount(fixture.swapchain));
    REQUIRE(fixture.device.resizeSwapchain(fixture.swapchain, fixture.width, fixture.height));
    presentOne(fixture);

    auto strategy = rachis::vulkan::queryVulkanPresentationSync(fixture.swapchain);
    REQUIRE(strategy);
    if (strategy->strategy == rachis::VulkanPresentCompletionStrategy::swapchainMaintenancePresentFence)
    {
        requireFenceAcquireFailure(fixture, semaphoreCount(fixture.swapchain));
        presentOne(fixture);
    }

    requireAcquireFailure(fixture,
                          rachis::vulkan::test::nextGraphicsPresentationSyncResult(),
                          "vkQueueSubmit2",
                          semaphoreCount(fixture.swapchain));
    REQUIRE(fixture.device.destroySwapchain(fixture.swapchain));
    fixture.swapchain = {};
    REQUIRE(fixture.device.waitIdle());
    CHECK(fixture.errors.load() == 0);
    CHECK(fixture.device.diagnosticCounts().errors == 0);
}

TEST_CASE("Vulkan resize recovers after vkCreateSwapchainKHR retires the old swapchain and fails",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][present]")
{
    PresentFixture fixture(true, 2);
    INFO(fixture.error.message);
    REQUIRE(fixture.ready);
    const bool destroyAfterFailure = GENERATE(false, true);
    presentOne(fixture);

    rachis::vulkan::test::nextCreateSwapchainResult().store(static_cast<std::int32_t>(VK_ERROR_OUT_OF_DEVICE_MEMORY),
                                                            std::memory_order_relaxed);
    auto failed = fixture.device.resizeSwapchain(fixture.swapchain, fixture.width, fixture.height);
    rachis::vulkan::test::nextCreateSwapchainResult().store(0, std::memory_order_relaxed);
    REQUIRE_FALSE(failed);
    CHECK(failed.error().code == rachis::ErrorCode::outOfMemory);
    CHECK(failed.error().nativeCode == static_cast<std::int64_t>(VK_ERROR_OUT_OF_DEVICE_MEMORY));
    auto refused = fixture.device.acquireNextImage(fixture.swapchain);
    REQUIRE_FALSE(refused);
    CHECK(refused.error().code == rachis::ErrorCode::invalidState);

    if (destroyAfterFailure)
    {
        REQUIRE(fixture.device.destroySwapchain(fixture.swapchain));
        fixture.swapchain = {};
        REQUIRE(fixture.recreateSwapchain());
    }
    else
    {
        auto retried = fixture.device.resizeSwapchain(fixture.swapchain, fixture.width, fixture.height);
        if (!retried)
        {
            INFO(retried.error().message);
            REQUIRE(retried);
        }
    }
    presentOne(fixture);
    REQUIRE(fixture.device.waitIdle());
    CHECK(fixture.errors.load() == 0);
    CHECK(fixture.device.diagnosticCounts().errors == 0);
}
