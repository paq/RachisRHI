#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include "Internal/CommandBufferAccess.h"
#include "Internal/VulkanCommands.h"
#include "Internal/VulkanDevice.h"
#include "Rachis/TestSupport/PresentInjection.h"
#include "Vulkan/VulkanTestHooks.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <Windows.h>
#elif defined(__linux__)
#include "LinuxSurfaceTestHost.h"
#endif

#ifndef RACHIS_TESTING
#error Vulkan device-loss injection must stay test-only.
#endif

namespace
{

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
            .enableDeviceFault = true,
            .diagnosticCallback = errors != nullptr ? &captureErrors : nullptr,
            .diagnosticUserData = errors,
        };
    }

    struct ClearSemaphoreCounterInjection final
    {
        ClearSemaphoreCounterInjection() = default;
        ~ClearSemaphoreCounterInjection()
        {
            rachis::vulkan::test::nextSemaphoreCounterValueResult().store(0, std::memory_order_relaxed);
        }

        ClearSemaphoreCounterInjection(const ClearSemaphoreCounterInjection&) = delete;
        ClearSemaphoreCounterInjection& operator=(const ClearSemaphoreCounterInjection&) = delete;
        ClearSemaphoreCounterInjection(ClearSemaphoreCounterInjection&&) = delete;
        ClearSemaphoreCounterInjection& operator=(ClearSemaphoreCounterInjection&&) = delete;
    };

    void injectSemaphoreCounterResult(VkResult result) noexcept
    {
        rachis::vulkan::test::nextSemaphoreCounterValueResult().store(result, std::memory_order_relaxed);
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

    class LossTestWindow final
    {
    public:
        LossTestWindow()
        {
#if defined(_WIN32)
            window_ = CreateWindowExW(0,
                                      L"STATIC",
                                      L"Rachis Vulkan device loss",
                                      WS_OVERLAPPEDWINDOW,
                                      CW_USEDEFAULT,
                                      CW_USEDEFAULT,
                                      96,
                                      64,
                                      nullptr,
                                      nullptr,
                                      GetModuleHandleW(nullptr),
                                      nullptr);
            if (window_ == nullptr)
            {
                unavailable_ = "CreateWindowExW failed.";
                return;
            }
            static_cast<void>(ShowWindow(window_, SW_SHOW));
#elif defined(__linux__)
            if (!rachis::test::WaylandTestWindow::compositorReachable())
            {
                unavailable_ = "No Wayland compositor accepts connections from this process.";
                return;
            }
            window_ = rachis::test::WaylandTestWindow::open("Rachis Vulkan device loss", unavailable_);
#else
            unavailable_ = "This platform has no window-system host for the Vulkan device-loss tests.";
#endif
        }

        ~LossTestWindow()
        {
#if defined(_WIN32)
            if (window_ != nullptr)
            {
                static_cast<void>(DestroyWindow(window_));
            }
#endif
        }

        LossTestWindow(const LossTestWindow&) = delete;
        LossTestWindow& operator=(const LossTestWindow&) = delete;
        LossTestWindow(LossTestWindow&&) = delete;
        LossTestWindow& operator=(LossTestWindow&&) = delete;

        [[nodiscard]] std::optional<rachis::SurfaceDesc> surfaceDesc() const noexcept
        {
#if defined(_WIN32)
            if (window_ != nullptr)
            {
                return rachis::SurfaceDesc{
                    .kind = rachis::SurfaceKind::win32,
                    .windowHandle = window_,
                    .label = "Vulkan lost-device surface",
                };
            }
#elif defined(__linux__)
            if (window_ != nullptr)
            {
                return rachis::SurfaceDesc{
                    .kind = rachis::SurfaceKind::wayland,
                    .windowHandle = window_->surface(),
                    .displayHandle = window_->display(),
                    .label = "Vulkan lost-device surface",
                };
            }
#endif
            return std::nullopt;
        }

        [[nodiscard]] const std::string& unavailableReason() const noexcept
        {
            return unavailable_;
        }

    private:
        std::string unavailable_;
#if defined(_WIN32)
        HWND window_ = nullptr;
#elif defined(__linux__)
        std::unique_ptr<rachis::test::WaylandTestWindow> window_;
#endif
    };

    [[nodiscard]] rachis::Result<rachis::AcquiredSwapchainImage> acquireWithin(rachis::Device& device,
                                                                               rachis::SwapchainHandle swapchain)
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (true)
        {
            auto acquired = device.acquireNextImage(swapchain);
            if (acquired || acquired.error().code != rachis::ErrorCode::notReady ||
                std::chrono::steady_clock::now() >= deadline)
            {
                return acquired;
            }
        }
    }

} // namespace

TEST_CASE("injected Vulkan device loss is terminal and repeats the original error", "[size:medium][rhi][vulkan][loss]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    CHECK(device.operationalState() == rachis::DeviceOperationalState::active);
    CHECK_FALSE(device.deviceLossError().has_value());

    auto healthy = device.captureDeviceFaultReport();
    REQUIRE(healthy);
    CHECK(healthy->status == rachis::DeviceFaultStatus::healthy);
    CHECK(device.diagnosticCounts().errors == 0);

    auto rejected = device.createBuffer({});
    REQUIRE_FALSE(rejected);
    CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(device.operationalState() == rachis::DeviceOperationalState::active);

    auto injected = rachis::vulkan::injectVulkanDeviceLoss();
    REQUIRE_FALSE(injected);
    CHECK(injected.error().code == rachis::ErrorCode::deviceLost);
    CHECK((injected.error().message.contains("injected") || injected.error().message.contains("Injected")));
    CHECK(device.operationalState() == rachis::DeviceOperationalState::lost);

    const auto first = device.deviceLossError();
    REQUIRE(first.has_value());
    CHECK(first->code == rachis::ErrorCode::deviceLost);
    CHECK(first->message == injected.error().message);

    auto again = rachis::vulkan::injectVulkanDeviceLoss(rachis::DeviceFaultStatus::hung);
    REQUIRE_FALSE(again);
    CHECK(again.error().message == first->message);

    auto createdAfter = device.createBuffer({
        .size = 64,
        .usage = rachis::BufferUsage::copyDestination,
        .label = "lost-device buffer",
    });
    REQUIRE_FALSE(createdAfter);
    CHECK(createdAfter.error().message == first->message);

    auto idle = device.waitIdle();
    REQUIRE_FALSE(idle);
    CHECK(idle.error().message == first->message);

    auto report = device.captureDeviceFaultReport();
    REQUIRE(report);
    CHECK(report->status == rachis::DeviceFaultStatus::removed);
    CHECK(report->nativeReason != 0);
    REQUIRE_FALSE(report->breadcrumbs.empty());
    CHECK(report->breadcrumbs.front().queueLabel == "RACHIS_TESTING");
    CHECK(report->breadcrumbs.front().commandListLabel == "injected device loss");

    auto repeated = device.captureDeviceFaultReport();
    REQUIRE(repeated);
    CHECK(repeated->status == report->status);
    CHECK(repeated->nativeReason == report->nativeReason);
}

TEST_CASE("injected Vulkan device loss rejects host buffer writes, reads and maps before handle checks",
          "[size:medium][rhi][vulkan][loss]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    auto upload = device.createBuffer({
        .size = 16,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "lost-device upload",
    });
    REQUIRE(upload);
    auto readback = device.createBuffer({
        .size = 16,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "lost-device readback",
    });
    REQUIRE(readback);
    auto stale = device.createBuffer({
        .size = 16,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "lost-device stale upload",
    });
    REQUIRE(stale);
    REQUIRE(device.destroyBuffer(*stale));
    std::array<std::byte, 16> bytes{};
    REQUIRE(device.writeBuffer(*upload, 0, bytes));
    REQUIRE(device.readBuffer(*readback, 0, bytes));

    auto injected = rachis::vulkan::injectVulkanDeviceLoss();
    REQUIRE_FALSE(injected);
    const auto first = device.deviceLossError();
    REQUIRE(first.has_value());

    auto written = device.writeBuffer(*upload, 0, bytes);
    REQUIRE_FALSE(written);
    CHECK(written.error().code == rachis::ErrorCode::deviceLost);
    CHECK(written.error().message == first->message);
    auto read = device.readBuffer(*readback, 0, bytes);
    REQUIRE_FALSE(read);
    CHECK(read.error().code == rachis::ErrorCode::deviceLost);
    CHECK(read.error().message == first->message);
    auto mapped = device.mapBuffer(*upload);
    REQUIRE_FALSE(mapped);
    CHECK(mapped.error().code == rachis::ErrorCode::deviceLost);
    CHECK(mapped.error().message == first->message);

    auto staleWrite = device.writeBuffer(*stale, 0, bytes);
    REQUIRE_FALSE(staleWrite);
    CHECK(staleWrite.error().code == rachis::ErrorCode::deviceLost);
    auto staleRead = device.readBuffer(*stale, 0, bytes);
    REQUIRE_FALSE(staleRead);
    CHECK(staleRead.error().code == rachis::ErrorCode::deviceLost);
    auto staleMap = device.mapBuffer(*stale);
    REQUIRE_FALSE(staleMap);
    CHECK(staleMap.error().code == rachis::ErrorCode::deviceLost);
}

TEST_CASE("injected Vulkan loss rejects public waits and safely drains the healthy GPU at teardown",
          "[size:medium][rhi][vulkan][loss][teardown]")
{
    std::atomic_uint64_t errors{0};
    auto created = rachis::createDevice(vulkanDesc(&errors));
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);

    auto injected = rachis::vulkan::injectVulkanDeviceLoss();
    REQUIRE_FALSE(injected);
    auto waited = device.wait(*submitted);
    REQUIRE_FALSE(waited);
    CHECK(waited.error().code == rachis::ErrorCode::deviceLost);
    CHECK(device.operationalState() == rachis::DeviceOperationalState::lost);

    auto afterSubmit = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE_FALSE(afterSubmit);
    CHECK(afterSubmit.error().code == rachis::ErrorCode::deviceLost);
    device = {};

    CHECK(errors.load() == 1);
}

TEST_CASE("injected Vulkan loss rejects acceleration-structure recording without side effects",
          "[size:medium][rhi][vulkan][loss][acceleration-structure]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.rayTracing && !capabilities.rayQuery)
    {
        SKIP("The Vulkan adapter does not advertise acceleration structures.");
    }

    const std::array vertices{
        std::array{-1.0F, -1.0F, 0.0F},
        std::array{0.0F, 1.0F, 0.0F},
        std::array{1.0F, -1.0F, 0.0F},
    };
    auto vertexBuffer = device.createBuffer({
        .size = sizeof(vertices),
        .usage = rachis::BufferUsage::accelerationStructureBuildInput,
        .memoryType = rachis::MemoryType::upload,
        .label = "Lost Vulkan AS vertices",
    });
    REQUIRE(vertexBuffer.has_value());
    REQUIRE(device.writeBuffer(*vertexBuffer, 0, std::as_bytes(std::span(vertices))).has_value());

    const std::array geometry{
        rachis::AccelerationStructureGeometry{
            .data =
                rachis::AccelerationStructureTriangleGeometry{
                    .vertexBuffer = *vertexBuffer,
                    .vertexCount = 3,
                    .vertexStride = sizeof(vertices[0]),
                    .vertexFormat = rachis::AccelerationStructureVertexFormat::float32x3,
                },
            .flags = rachis::AccelerationStructureGeometryFlag::opaque,
        },
    };
    const rachis::BottomLevelAccelerationStructureBuildInput bottomInput{
        .geometries = geometry,
        .flags = rachis::AccelerationStructureBuildFlag::allowUpdate,
    };
    auto bottomSizes = device.queryBottomLevelAccelerationStructureBuildSizes(bottomInput);
    auto topSizes = device.queryTopLevelAccelerationStructureBuildSizes({
        .instanceCount = 1,
        .flags = rachis::AccelerationStructureBuildFlag::allowUpdate,
    });
    REQUIRE(bottomSizes.has_value());
    REQUIRE(topSizes.has_value());

    auto bottomStorage = device.createBuffer({
        .size = bottomSizes->accelerationStructureSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Lost Vulkan BLAS storage",
    });
    auto topStorage = device.createBuffer({
        .size = topSizes->accelerationStructureSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Lost Vulkan TLAS storage",
    });
    const auto scratchSize = (std::max)({
        bottomSizes->buildScratchSize,
        bottomSizes->updateScratchSize,
        topSizes->buildScratchSize,
        topSizes->updateScratchSize,
    });
    auto scratch = device.createBuffer({
        .size = scratchSize,
        .usage = rachis::BufferUsage::accelerationStructureScratch,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Lost Vulkan AS scratch",
    });
    REQUIRE(bottomStorage.has_value());
    REQUIRE(topStorage.has_value());
    REQUIRE(scratch.has_value());
    auto bottomLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::bottomLevel,
        .storage = *bottomStorage,
        .label = "Lost Vulkan BLAS",
    });
    auto topLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::topLevel,
        .storage = *topStorage,
        .label = "Lost Vulkan TLAS",
    });
    REQUIRE(bottomLevel.has_value());
    REQUIRE(topLevel.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    auto* backend = rachis::detail::CommandBufferAccess::get(*command);
    REQUIRE(backend != nullptr);
    auto& native = static_cast<rachis::vulkan::VulkanCommandBuffer&>(*backend);
    const auto retainedBefore = native.retained().size();

    auto injected = rachis::vulkan::injectVulkanDeviceLoss();
    REQUIRE_FALSE(injected);
    const auto loss = device.deviceLossError();
    REQUIRE(loss.has_value());

    const auto expectRejected = [&](const rachis::Status& status) {
        REQUIRE_FALSE(status.has_value());
        CHECK(status.error().code == rachis::ErrorCode::deviceLost);
        CHECK(status.error().message == loss->message);
        CHECK(native.retained().size() == retainedBefore);
    };
    const std::array instances{
        rachis::AccelerationStructureInstance{.bottomLevel = *bottomLevel},
    };
    expectRejected(device.buildBottomLevelAccelerationStructure(*command,
                                                                {
                                                                    .destination = *bottomLevel,
                                                                    .input = bottomInput,
                                                                    .scratch = *scratch,
                                                                }));
    expectRejected(
        device.buildTopLevelAccelerationStructure(*command,
                                                  {
                                                      .destination = *topLevel,
                                                      .instances = instances,
                                                      .flags = rachis::AccelerationStructureBuildFlag::allowUpdate,
                                                      .scratch = *scratch,
                                                  }));
    expectRejected(device.updateBottomLevelAccelerationStructure(*command,
                                                                 {
                                                                     .source = *bottomLevel,
                                                                     .destination = *bottomLevel,
                                                                     .input = bottomInput,
                                                                     .scratch = *scratch,
                                                                 }));
    expectRejected(
        device.updateTopLevelAccelerationStructure(*command,
                                                   {
                                                       .source = *topLevel,
                                                       .destination = *topLevel,
                                                       .instances = instances,
                                                       .flags = rachis::AccelerationStructureBuildFlag::allowUpdate,
                                                       .scratch = *scratch,
                                                   }));
    expectRejected(device.updateBottomLevelAccelerationStructureInPlace(*command,
                                                                        {
                                                                            .target = *bottomLevel,
                                                                            .input = bottomInput,
                                                                            .scratch = *scratch,
                                                                        }));
    expectRejected(device.updateTopLevelAccelerationStructureInPlace(
        *command,
        {
            .target = *topLevel,
            .instances = instances,
            .flags = rachis::AccelerationStructureBuildFlag::allowUpdate,
            .scratch = *scratch,
        }));
    expectRejected(device.compactAccelerationStructure(*command,
                                                       {
                                                           .source = *bottomLevel,
                                                           .destination = *bottomLevel,
                                                           .compactedSize = bottomSizes->accelerationStructureSize,
                                                       }));
}

TEST_CASE("Vulkan timeline queries keep the device active through transient counter failures",
          "[size:medium][rhi][vulkan][loss]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const ClearSemaphoreCounterInjection clear;

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    const auto failure = GENERATE(VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY);
    injectSemaphoreCounterResult(failure);
    auto queried = device.queryTimelinePoint(*submitted);
    REQUIRE_FALSE(queried);
    CHECK(queried.error().code == rachis::ErrorCode::outOfMemory);
    CHECK(queried.error().nativeCode == static_cast<std::int64_t>(failure));
    CHECK(queried.error().message.contains("vkGetSemaphoreCounterValue"));
    CHECK(device.operationalState() == rachis::DeviceOperationalState::active);
    CHECK_FALSE(device.deviceLossError().has_value());

    injectSemaphoreCounterResult(failure);
    auto waited = device.wait(*submitted, std::chrono::nanoseconds::zero());
    REQUIRE_FALSE(waited);
    CHECK(waited.error().code == rachis::ErrorCode::outOfMemory);
    CHECK(device.operationalState() == rachis::DeviceOperationalState::active);

    auto recovered = device.queryTimelinePoint(*submitted);
    REQUIRE(recovered);
    CHECK(*recovered == rachis::TimelinePointStatus::complete);
    auto next = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(next);
    auto resubmitted = device.submit(std::move(*next));
    REQUIRE(resubmitted);
    CHECK(device.wait(*resubmitted).has_value());
}

TEST_CASE("Vulkan timeline queries lose the device when the counter reports device loss",
          "[size:medium][rhi][vulkan][loss]")
{
    std::atomic_uint64_t errors{0};
    auto desc = vulkanDesc(&errors);
    desc.enableDeviceFault = false;
    auto created = rachis::createDevice(desc);
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const ClearSemaphoreCounterInjection clear;

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    injectSemaphoreCounterResult(VK_ERROR_DEVICE_LOST);
    auto queried = device.queryTimelinePoint(*submitted);
    REQUIRE_FALSE(queried);
    CHECK(queried.error().code == rachis::ErrorCode::deviceLost);
    CHECK(queried.error().nativeCode == static_cast<std::int64_t>(VK_ERROR_DEVICE_LOST));
    CHECK(device.operationalState() == rachis::DeviceOperationalState::lost);
    const auto loss = device.deviceLossError();
    REQUIRE(loss.has_value());
    CHECK(loss->message == queried.error().message);
    auto report = device.captureDeviceFaultReport();
    REQUIRE(report);
    CHECK(report->nativeReason == static_cast<std::int64_t>(VK_ERROR_DEVICE_LOST));
    device = {};

    CHECK(errors.load() == 1);
}

TEST_CASE("Vulkan destroy and release calls succeed after device loss and retire their handles",
          "[size:medium][rhi][vulkan][loss]")
{
    std::atomic_uint64_t errors{0};
    auto desc = vulkanDesc(&errors);
    desc.enableDeviceFault = false;
    auto created = rachis::createDevice(desc);
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto bytecode = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanComputeWrite.spv");
    REQUIRE_FALSE(bytecode.empty());

    const rachis::BufferDesc storageDesc{
        .size = 256,
        .usage = rachis::BufferUsage::storage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan lost-device storage buffer",
    };
    auto mapped = device.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan lost-device mapped buffer",
    });
    auto storage = device.createBuffer(storageDesc);
    REQUIRE(mapped);
    REQUIRE(storage);
    REQUIRE(device.mapBuffer(*mapped));
    const auto requirements = device.queryBufferMemoryRequirements(storageDesc);
    REQUIRE(requirements);
    auto heap = device.createResourceHeap({.size = requirements->size * 2, .label = "Vulkan lost-device heap"});
    REQUIRE(heap);
    auto texture = device.createTexture({
        .width = 4,
        .height = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
        .label = "Vulkan lost-device texture",
    });
    REQUIRE(texture);
    auto view = device.createTextureView({.texture = *texture});
    auto sampler = device.createSampler({});
    REQUIRE(view);
    REQUIRE(sampler);

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = entries});
    REQUIRE(bindingLayout);
    const std::array poolSizes{
        rachis::BindingPoolSize{.type = rachis::BindingType::storageBuffer, .descriptorCount = 1}};
    auto bindingPool = device.createBindingPool({.maxBindingSets = 1, .sizes = poolSizes});
    REQUIRE(bindingPool);
    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource = rachis::BufferBinding{.buffer = *storage, .size = 256, .structureByteStride = 4},
        },
    };
    auto bindingSet = device.createBindingSet({.layout = *bindingLayout, .writes = writes, .pool = *bindingPool});
    REQUIRE(bindingSet);
    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
    };
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstantRanges,
    });
    REQUIRE(pipelineLayout);

    const std::array libraryEntries{rachis::ShaderLibraryEntryDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
    }};
    auto library = device.createShaderLibrary({.entries = libraryEntries});
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    REQUIRE(library);
    REQUIRE(shader);
    auto cache = device.createPipelineCache();
    REQUIRE(cache);
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .cache = *cache,
        .cacheKey = {.high = 1, .low = 133},
    });
    auto compilation = device.beginComputePipelineCompilation({
        .layout = *pipelineLayout,
        .computeShader = *shader,
    });
    REQUIRE(pipeline);
    REQUIRE(compilation);

    auto commandPool = device.createCommandPool({.queueType = rachis::QueueType::copy, .maxCommandBuffers = 2});
    auto queryPool = device.createTimestampQueryPool({.queueType = rachis::QueueType::graphics, .queryCount = 2});
    REQUIRE(commandPool);
    REQUIRE(queryPool);

    SECTION("injected loss")
    {
        REQUIRE_FALSE(rachis::vulkan::injectVulkanDeviceLoss());
    }
    SECTION("vkEndCommandBuffer reports VK_ERROR_DEVICE_LOST")
    {
        auto command = device.acquireCommandBuffer(*commandPool);
        REQUIRE(command);
        const rachis::test::ScopedAtomicTestHook<std::int32_t> endFails(
            rachis::vulkan::test::nextEndCommandBufferResult(), static_cast<std::int32_t>(VK_ERROR_DEVICE_LOST));
        auto submitted = device.submit(std::move(*command));
        REQUIRE_FALSE(submitted);
        CHECK(submitted.error().code == rachis::ErrorCode::deviceLost);
    }
    REQUIRE(device.operationalState() == rachis::DeviceOperationalState::lost);
    auto rejected = device.createBuffer({.size = 16, .usage = rachis::BufferUsage::copySource});
    REQUIRE_FALSE(rejected);
    CHECK(rejected.error().code == rachis::ErrorCode::deviceLost);

    const auto released = [](const char* call, const rachis::Status& status) {
        INFO(call << ": " << (status ? std::string{} : status.error().message));
        CHECK(status.has_value());
    };
    released("destroyPipelineCompilation", device.destroyPipelineCompilation(*compilation));
    released("destroyPipeline", device.destroyPipeline(*pipeline));
    released("destroyPipelineCache", device.destroyPipelineCache(*cache));
    released("destroyShaderModule", device.destroyShaderModule(*shader));
    released("destroyShaderLibrary", device.destroyShaderLibrary(*library));
    released("destroyPipelineLayout", device.destroyPipelineLayout(*pipelineLayout));
    released("destroyBindingSet", device.destroyBindingSet(*bindingSet));
    released("destroyBindingPool", device.destroyBindingPool(*bindingPool));
    released("destroyBindingLayout", device.destroyBindingLayout(*bindingLayout));
    released("destroyTextureView", device.destroyTextureView(*view));
    released("destroyTexture", device.destroyTexture(*texture));
    released("destroySampler", device.destroySampler(*sampler));
    released("destroyResourceHeap", device.destroyResourceHeap(*heap));
    released("unmapBuffer", device.unmapBuffer(*mapped));
    released("destroyBuffer(mapped)", device.destroyBuffer(*mapped));
    released("destroyBuffer(storage)", device.destroyBuffer(*storage));
    released("destroyQueryPool", device.destroyQueryPool(*queryPool));
    released("destroyCommandPool", device.destroyCommandPool(*commandPool));

    auto again = device.destroyBuffer(*storage);
    REQUIRE_FALSE(again);
    CHECK(again.error().code == rachis::ErrorCode::invalidArgument);
    auto viewAgain = device.destroyTextureView(*view);
    REQUIRE_FALSE(viewAgain);
    CHECK(viewAgain.error().code == rachis::ErrorCode::invalidArgument);
    auto poolAgain = device.destroyCommandPool(*commandPool);
    REQUIRE_FALSE(poolAgain);
    CHECK(poolAgain.error().code == rachis::ErrorCode::invalidArgument);
    device = {};
    CHECK(errors.load() == 1);
}

TEST_CASE("Vulkan destroySwapchain after device loss retires an acquired image that still has an application view",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][present][loss]")
{
    const LossTestWindow window;
    const auto surfaceDesc = window.surfaceDesc();
    if (!surfaceDesc.has_value())
    {
        SKIP(window.unavailableReason());
    }
    std::atomic_uint64_t errors{0};
    auto created = rachis::createDevice(vulkanDesc(&errors));
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireAdapterInfo(device).backend == rachis::Backend::vulkan);

    auto surface = device.createSurface(*surfaceDesc);
    INFO((surface ? std::string{} : surface.error().message));
    REQUIRE(surface);
    auto capabilities = device.querySurfaceCapabilities(*surface);
    REQUIRE(capabilities);
    REQUIRE_FALSE(capabilities->formats.empty());
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
        .label = "Vulkan lost-device swapchain",
    });
    INFO((swapchain ? std::string{} : swapchain.error().message));
    REQUIRE(swapchain);
    auto acquired = acquireWithin(device, *swapchain);
    INFO((acquired ? std::string{} : acquired.error().message));
    REQUIRE(acquired);
    auto applicationView = device.createTextureView({
        .texture = acquired->texture,
        .usage = rachis::TextureViewUsage::renderTarget,
        .label = "Application view of a lost-device swapchain image",
    });
    REQUIRE(applicationView);

    auto injected = rachis::vulkan::injectVulkanDeviceLoss();
    REQUIRE_FALSE(injected);
    CHECK(injected.error().code == rachis::ErrorCode::deviceLost);
    REQUIRE(device.operationalState() == rachis::DeviceOperationalState::lost);

    auto destroyed = device.destroySwapchain(*swapchain);
    INFO((destroyed ? std::string{} : destroyed.error().message));
    REQUIRE(destroyed.has_value());
    auto destroyedAgain = device.destroySwapchain(*swapchain);
    REQUIRE_FALSE(destroyedAgain.has_value());
    CHECK(destroyedAgain.error().code == rachis::ErrorCode::invalidArgument);

    auto retiredTexture = device.destroyTexture(acquired->texture);
    REQUIRE_FALSE(retiredTexture.has_value());
    CHECK(retiredTexture.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(retiredTexture.error().message.contains("stale"));

    auto viewDestroyed = device.destroyTextureView(*applicationView);
    INFO((viewDestroyed ? std::string{} : viewDestroyed.error().message));
    REQUIRE(viewDestroyed.has_value());
    auto viewDestroyedAgain = device.destroyTextureView(*applicationView);
    REQUIRE_FALSE(viewDestroyedAgain.has_value());
    CHECK(viewDestroyedAgain.error().code == rachis::ErrorCode::invalidArgument);

    REQUIRE(device.destroySurface(*surface).has_value());
    device = {};
    CHECK(errors.load() == 1);
}
