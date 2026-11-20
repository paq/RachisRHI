#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include <catch2/catch_test_macros.hpp>

#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef RACHIS_SPIRV_DIR
namespace
{

    constexpr std::uint32_t inputCount = 100'000;
    constexpr std::uint64_t sliceSize = 256;
    constexpr std::array<std::uint32_t, 4> sliceValues{100, 200, 300, 400};
    constexpr std::array<std::uint32_t, 1> replacementValues{500};
    constexpr std::uint64_t resultRangeSize = 256;
    constexpr std::uint64_t resultBufferSize = 2 * resultRangeSize;

    struct Selection final
    {
        std::uint32_t element = 0;
        std::uint32_t slot = 0;
    };

    [[nodiscard]] std::vector<std::byte> loadShader()
    {
        std::ifstream stream(std::filesystem::path(RACHIS_SPIRV_DIR) / "UpdateAfterBindBinding.spv",
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

    [[nodiscard]] rachis::BufferBinding slice(rachis::BufferHandle values, std::size_t index)
    {
        return rachis::BufferBinding{values, index * sliceSize, sliceSize, sizeof(std::uint32_t)};
    }

    struct ResourceStatisticsDelta final
    {
        std::uint64_t prepared = 0;
        std::uint64_t tablePublishes = 0;
        std::uint64_t published = 0;
    };

    [[nodiscard]] ResourceStatisticsDelta resourceDelta(const rachis::BindingUpdateStatistics& before,
                                                        const rachis::BindingUpdateStatistics& after)
    {
        return ResourceStatisticsDelta{
            .prepared = after.resourceDescriptorsPrepared - before.resourceDescriptorsPrepared,
            .tablePublishes = after.resourceTablePublishCount - before.resourceTablePublishCount,
            .published = after.resourceDescriptorsPublished - before.resourceDescriptorsPublished,
        };
    }

    class LargeArrayFixture final
    {
    public:
        LargeArrayFixture(rachis::Device& device, rachis::BindingFlag arrayFlags) : device_(device)
        {
            const auto bytecode = loadShader();
            REQUIRE_FALSE(bytecode.empty());
            const std::array requirements{
                rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
                rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::readOnlyStorageBuffer, inputCount},
            };
            const std::array pushConstants{rachis::ShaderPushConstantRequirement{0, sizeof(Selection)}};
            auto shader = device_.createShaderModule({
                .stage = rachis::ShaderStage::compute,
                .bytecode = bytecode,
                .entryPoint = "main",
                .label = "Vulkan large binding array shader",
                .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, requirements, pushConstants),
            });
            REQUIRE(shader.has_value());
            shader_ = *shader;

            createBuffers();

            const std::array entries{
                rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
                rachis::BindingLayoutEntry{1,
                                           rachis::BindingType::readOnlyStorageBuffer,
                                           rachis::ShaderStage::compute,
                                           inputCount,
                                           arrayFlags},
            };
            auto layout =
                device_.createBindingLayout({.entries = entries, .label = "Vulkan large binding array layout"});
            const std::string layoutFailure = layout ? std::string{} : layout.error().message;
            CAPTURE(layoutFailure);
            REQUIRE(layout.has_value());
            layout_ = *layout;

            const std::array sizes{
                rachis::BindingPoolSize{rachis::BindingType::storageBuffer, 4},
                rachis::BindingPoolSize{rachis::BindingType::readOnlyStorageBuffer, 3 * inputCount},
            };
            auto pool =
                device_.createBindingPool({.maxBindingSets = 1, .sizes = sizes, .label = "Vulkan large array pool"});
            REQUIRE(pool.has_value());
            pool_ = *pool;

            const std::array writes{
                rachis::BindingWrite{0, 0, rachis::BufferBinding{results_, 0, resultRangeSize, sizeof(std::uint32_t)}},
                rachis::BindingWrite{1, 0, slice(values_, 0)},
                rachis::BindingWrite{1, 1, slice(values_, 1)},
            };
            auto set = device_.createBindingSet({
                .layout = layout_,
                .writes = writes,
                .label = "Vulkan large binding array set",
                .pool = pool_,
            });
            const std::string setFailure = set ? std::string{} : set.error().message;
            CAPTURE(setFailure);
            REQUIRE(set.has_value());
            set_ = *set;

            const std::array layouts{layout_};
            const std::array ranges{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, sizeof(Selection)}};
            auto pipelineLayout =
                device_.createPipelineLayout({.bindingLayouts = layouts, .pushConstantRanges = ranges});
            const std::string pipelineLayoutFailure = pipelineLayout ? std::string{} : pipelineLayout.error().message;
            CAPTURE(pipelineLayoutFailure);
            REQUIRE(pipelineLayout.has_value());
            pipelineLayout_ = *pipelineLayout;
            auto pipeline = device_.createComputePipeline({.layout = pipelineLayout_, .computeShader = shader_});
            const std::string pipelineFailure = pipeline ? std::string{} : pipeline.error().message;
            CAPTURE(pipelineFailure);
            REQUIRE(pipeline.has_value());
            pipeline_ = *pipeline;
        }

        ~LargeArrayFixture()
        {
            static_cast<void>(device_.waitIdle());
            static_cast<void>(device_.destroyPipeline(pipeline_));
            static_cast<void>(device_.destroyPipelineLayout(pipelineLayout_));
            destroySet();
            static_cast<void>(device_.destroyBindingPool(pool_));
            static_cast<void>(device_.destroyBindingLayout(layout_));
            static_cast<void>(device_.destroyShaderModule(shader_));
            destroyValues();
            static_cast<void>(device_.destroyBuffer(results_));
            static_cast<void>(device_.destroyBuffer(readback_));
        }

        LargeArrayFixture(const LargeArrayFixture&) = delete;
        LargeArrayFixture& operator=(const LargeArrayFixture&) = delete;
        LargeArrayFixture(LargeArrayFixture&&) = delete;
        LargeArrayFixture& operator=(LargeArrayFixture&&) = delete;

        [[nodiscard]] rachis::BindingSetHandle set() const noexcept
        {
            return set_;
        }

        [[nodiscard]] rachis::BufferHandle values() const noexcept
        {
            return values_;
        }

        [[nodiscard]] rachis::BufferHandle results() const noexcept
        {
            return results_;
        }

        void destroySet()
        {
            if (set_.isValid())
            {
                REQUIRE(device_.destroyBindingSet(std::exchange(set_, {})).has_value());
            }
        }

        void destroyValues()
        {
            if (values_.isValid())
            {
                REQUIRE(device_.destroyBuffer(std::exchange(values_, {})).has_value());
            }
        }

        [[nodiscard]] rachis::BufferHandle createValueBuffer(std::span<const std::uint32_t> values,
                                                             std::string_view label)
        {
            const auto size = values.size() * sliceSize;
            auto buffer = device_.createBuffer({
                .size = size,
                .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copyDestination,
                .label = label,
            });
            auto staging = device_.createBuffer({
                .size = size,
                .usage = rachis::BufferUsage::copySource,
                .memoryType = rachis::MemoryType::upload,
                .label = "Vulkan large array value staging",
            });
            REQUIRE(buffer.has_value());
            REQUIRE(staging.has_value());
            for (std::size_t index = 0; index < values.size(); ++index)
            {
                REQUIRE(device_.writeBuffer(*staging, index * sliceSize, std::as_bytes(values.subspan(index, 1)))
                            .has_value());
            }
            auto command = rachis::test::acquireCommandBuffer(device_, rachis::QueueType::compute);
            REQUIRE(command.has_value());
            REQUIRE(device_.copyBuffer(*command, {*staging, 0, *buffer, 0, size}).has_value());
            REQUIRE(device_
                        .bufferBarrier(*command,
                                       {
                                           .buffer = *buffer,
                                           .stagesBefore = rachis::PipelineStage::copy,
                                           .stagesAfter = rachis::PipelineStage::computeShader,
                                           .accessBefore = rachis::Access::copyDestination,
                                           .accessAfter = rachis::Access::shaderResource,
                                       })
                        .has_value());
            auto submitted = device_.submit(std::move(*command));
            REQUIRE(submitted.has_value());
            REQUIRE(device_.wait(*submitted).has_value());
            REQUIRE(device_.destroyBuffer(*staging).has_value());
            return *buffer;
        }

        [[nodiscard]] rachis::CommandBuffer record(std::span<const Selection> selections)
        {
            auto command = rachis::test::acquireCommandBuffer(device_, rachis::QueueType::compute);
            REQUIRE(command.has_value());
            recordInto(*command, selections);
            return std::move(*command);
        }

        void recordInto(rachis::CommandBuffer& command, std::span<const Selection> selections)
        {
            REQUIRE(device_
                        .bufferBarrier(command,
                                       {
                                           .buffer = results_,
                                           .stagesBefore = rachis::PipelineStage::all,
                                           .stagesAfter = rachis::PipelineStage::computeShader,
                                           .accessBefore = rachis::Access::common,
                                           .accessAfter = rachis::Access::unorderedAccess,
                                       })
                        .has_value());
            REQUIRE(device_.bindComputePipeline(command, pipeline_).has_value());
            REQUIRE(device_.bindComputeBindingSet(command, 0, set_).has_value());
            for (std::size_t index = 0; index < selections.size(); ++index)
            {
                if (index != 0)
                {
                    REQUIRE(device_
                                .bufferBarrier(command,
                                               {
                                                   .buffer = results_,
                                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                                   .accessBefore = rachis::Access::unorderedAccess,
                                                   .accessAfter = rachis::Access::unorderedAccess,
                                               })
                                .has_value());
                }
                REQUIRE(device_.setComputePushConstants(command, 0, std::as_bytes(std::span(&selections[index], 1)))
                            .has_value());
                REQUIRE(device_.dispatch(command, 1).has_value());
            }
        }

        void recordReadback(rachis::CommandBuffer& command)
        {
            REQUIRE(device_
                        .bufferBarrier(command,
                                       {
                                           .buffer = results_,
                                           .stagesBefore = rachis::PipelineStage::computeShader,
                                           .stagesAfter = rachis::PipelineStage::copy,
                                           .accessBefore = rachis::Access::unorderedAccess,
                                           .accessAfter = rachis::Access::copySource,
                                       })
                        .has_value());
            REQUIRE(device_.copyBuffer(command, {results_, 0, readback_, 0, resultBufferSize}).has_value());
        }

        [[nodiscard]] std::uint32_t result(std::uint64_t offset)
        {
            std::uint32_t value = 0;
            REQUIRE(device_.readBuffer(readback_, offset, std::as_writable_bytes(std::span(&value, 1))).has_value());
            return value;
        }

    private:
        void createBuffers()
        {
            auto results = device_.createBuffer({
                .size = resultBufferSize,
                .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
                .label = "Vulkan large array results",
            });
            auto readback = device_.createBuffer({
                .size = resultBufferSize,
                .usage = rachis::BufferUsage::copyDestination,
                .memoryType = rachis::MemoryType::readback,
                .label = "Vulkan large array readback",
            });
            REQUIRE(results.has_value());
            REQUIRE(readback.has_value());
            results_ = *results;
            readback_ = *readback;
            values_ = createValueBuffer(sliceValues, "Vulkan large array values");
        }

        rachis::Device& device_;
        rachis::ShaderModuleHandle shader_{};
        rachis::BufferHandle values_{};
        rachis::BufferHandle results_{};
        rachis::BufferHandle readback_{};
        rachis::BindingLayoutHandle layout_{};
        rachis::BindingPoolHandle pool_{};
        rachis::BindingSetHandle set_{};
        rachis::PipelineLayoutHandle pipelineLayout_{};
        rachis::PipelineHandle pipeline_{};
    };

    [[nodiscard]] rachis::Status update(rachis::Device& device,
                                        rachis::BindingSetHandle set,
                                        std::uint32_t binding,
                                        std::uint32_t element,
                                        const rachis::BindingResource& resource)
    {
        const std::array writes{rachis::BindingWrite{binding, element, resource}};
        return device.updateBindingSet(set, writes);
    }

    void checkCleanDiagnostics(const rachis::Device& device)
    {
        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.warnings == 0);
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
    }

    [[nodiscard]] bool largeArraySupported(rachis::Device& device, rachis::BindingFlag arrayFlags)
    {
        const auto capabilities = rachis::test::requireCapabilities(device);
        if (capabilities.limits.maxBindingPoolResourceDescriptors < 3 * inputCount + 4)
        {
            return false;
        }
        const std::array entries{
            rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
            rachis::BindingLayoutEntry{
                1, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute, inputCount, arrayFlags},
        };
        auto layout = device.createBindingLayout({.entries = entries});
        if (!layout)
        {
            return false;
        }
        const std::array layouts{*layout};
        const std::array ranges{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, sizeof(Selection)}};
        auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts, .pushConstantRanges = ranges});
        const bool supported = pipelineLayout.has_value();
        if (pipelineLayout)
        {
            REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
        }
        REQUIRE(device.destroyBindingLayout(*layout).has_value());
        return supported;
    }

    [[nodiscard]] bool skipWithoutLargeArray(rachis::Device& device, rachis::BindingFlag arrayFlags)
    {
        return rachis::test::skipUnlessReferenceDesktopAdvertises(
            largeArraySupported(device, arrayFlags),
            "descriptor limits for a 100,000-element storage-buffer array and a pool three times that size");
    }

    [[nodiscard]] bool skipWithoutUpdateAfterBind(rachis::Device& device)
    {
        return rachis::test::skipUnlessReferenceDesktopAdvertises(
            rachis::test::requireCapabilities(device).updateAfterBindBindingArrays, "updateAfterBindBindingArrays");
    }

    class ThreadAllocatedBytes final
    {
    public:
        [[nodiscard]] static constexpr bool available() noexcept
        {
#if defined(_MSC_VER) && defined(_DEBUG)
            return true;
#else
            return false;
#endif
        }

        template<typename Function>
        [[nodiscard]] static std::size_t measure(Function&& function)
        {
#if defined(_MSC_VER) && defined(_DEBUG)
            const auto previous = _CrtSetAllocHook(&countAllocation);
            bytes() = 0;
            counting() = true;
            function();
            counting() = false;
            _CrtSetAllocHook(previous);
            return bytes();
#else
            function();
            return 0;
#endif
        }

    private:
        [[nodiscard]] static bool& counting() noexcept
        {
            static thread_local bool s_counting = false;
            return s_counting;
        }

        [[nodiscard]] static std::size_t& bytes() noexcept
        {
            static thread_local std::size_t s_bytes = 0;
            return s_bytes;
        }

#if defined(_MSC_VER) && defined(_DEBUG)
        static int __cdecl countAllocation(int allocationType,
                                           void* userData,
                                           std::size_t size,
                                           int blockType,
                                           long requestNumber,
                                           const unsigned char* fileName,
                                           int lineNumber)
        {
            static_cast<void>(userData);
            static_cast<void>(requestNumber);
            static_cast<void>(fileName);
            static_cast<void>(lineNumber);
            if (blockType != _CRT_BLOCK && (allocationType == _HOOK_ALLOC || allocationType == _HOOK_REALLOC) &&
                counting())
            {
                bytes() += size;
            }
            return 1;
        }
#endif
    };

    [[nodiscard]] std::uint64_t allocationCount(const rachis::Device& device)
    {
        const auto budgets = device.queryMemoryBudgets();
        REQUIRE(budgets.has_value());
        std::uint64_t count = 0;
        for (const auto& budget : *budgets)
        {
            REQUIRE(rachis::hasAny(budget.features, rachis::MemoryBudgetFeature::allocationStatistics));
            count += budget.allocationCount;
        }
        return count;
    }

} // namespace

TEST_CASE("Vulkan binding set updates republish a large array in full on every update",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][binding][compute]")
{
    auto created = rachis::createDevice(rachis::test::vulkanDeviceDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    if (skipWithoutLargeArray(device, rachis::BindingFlag::partiallyBound))
    {
        return;
    }
    LargeArrayFixture fixture(device, rachis::BindingFlag::partiallyBound);

    const std::array firstSelections{Selection{.element = 0, .slot = 0}};
    auto first = fixture.record(firstSelections);

    const auto beforeInFlight = device.bindingUpdateStatistics();
    REQUIRE(update(device, fixture.set(), 1, 1, slice(fixture.values(), 2)).has_value());
    const auto inFlight = resourceDelta(beforeInFlight, device.bindingUpdateStatistics());
    CHECK(inFlight.prepared == 1);
    CHECK(inFlight.tablePublishes == 1);
    CHECK(inFlight.published == inputCount + 1);
    auto firstSubmitted = device.submit(std::move(first));
    REQUIRE(firstSubmitted.has_value());

    const std::array secondSelections{Selection{.element = 1, .slot = 1}};
    auto second = fixture.record(secondSelections);
    fixture.recordReadback(second);
    auto secondSubmitted = device.submit(std::move(second));
    REQUIRE(secondSubmitted.has_value());
    REQUIRE(device.wait(*secondSubmitted).has_value());
    REQUIRE(device.wait(*firstSubmitted).has_value());
    device.collectGarbage();
    CHECK(fixture.result(0) == sliceValues[0]);
    CHECK(fixture.result(sizeof(std::uint32_t)) == sliceValues[2]);

    const auto beforeIdle = device.bindingUpdateStatistics();
    REQUIRE(update(device, fixture.set(), 1, 1, slice(fixture.values(), 1)).has_value());
    const auto idle = resourceDelta(beforeIdle, device.bindingUpdateStatistics());
    CHECK(idle.prepared == 1);
    CHECK(idle.tablePublishes == 1);
    CHECK(idle.published == inputCount + 1);

    checkCleanDiagnostics(device);
}

#if RACHIS_ENABLE_D3D12
namespace
{

    struct GateTimeline final
    {
        rachis::Device device;
        rachis::SharedTimelinePoint shared;
        std::uint64_t firstValue = 0;
        std::uint64_t signaledValue = 0;

        [[nodiscard]] rachis::SharedTimelinePointView at(std::uint64_t value) const noexcept
        {
            return rachis::SharedTimelinePointView{.handle = shared.handle.view(), .value = value};
        }

        void signal(std::uint64_t value)
        {
            while (signaledValue < value)
            {
                auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
                REQUIRE(command);
                auto submitted = device.submit(std::move(*command));
                REQUIRE(submitted);
                signaledValue = submitted->value();
            }
        }
    };

    [[nodiscard]] std::optional<GateTimeline> createGateTimeline(const rachis::Device& consumer)
    {
        if (!rachis::test::requireCapabilities(consumer).sharedTimelineHandles)
        {
            return std::nullopt;
        }
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::d3d12,
            .adapterPreference = rachis::AdapterPreference::highPerformance,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
        });
        if (!created)
        {
            return std::nullopt;
        }
        const auto consumerAdapter = rachis::test::requireAdapterInfo(consumer);
        const auto producerAdapter = rachis::test::requireAdapterInfo(*created);
        if (producerAdapter.vendorId != consumerAdapter.vendorId ||
            producerAdapter.deviceId != consumerAdapter.deviceId)
        {
            return std::nullopt;
        }
        GateTimeline gate{.device = std::move(*created)};
        auto command = rachis::test::acquireCommandBuffer(gate.device, rachis::QueueType::graphics);
        REQUIRE(command);
        auto first = gate.device.submit(std::move(*command));
        REQUIRE(first);
        REQUIRE(gate.device.wait(*first).has_value());
        auto shared = gate.device.exportTimelinePoint(*first);
        REQUIRE(shared);
        gate.shared = std::move(*shared);
        gate.firstValue = first->value();
        gate.signaledValue = first->value();
        return gate;
    }

    template<typename Call>
    [[nodiscard]] auto callReleasingGate(Call call, GateTimeline& gate, std::uint64_t release)
    {
        auto pending = std::async(std::launch::async, std::move(call));
        const bool returnedPromptly = pending.wait_for(std::chrono::seconds{2}) == std::future_status::ready;
        if (!returnedPromptly)
        {
            gate.signal(release);
        }
        CHECK(returnedPromptly);
        return pending.get();
    }

    [[nodiscard]] bool pending(const rachis::Device& device, rachis::TimelinePoint point)
    {
        const auto status = device.queryTimelinePoint(point);
        REQUIRE(status.has_value());
        return *status == rachis::TimelinePointStatus::pending;
    }

} // namespace

TEST_CASE("Vulkan update-after-bind entries of a large array change in place while a gated submission reads other "
          "elements of the same set",
          "[size:medium][requires:HardwareVulkan][requires:HardwareD3D12][rhi][vulkan][binding][compute]")
{
    auto created = rachis::createDevice(rachis::test::vulkanDeviceDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    constexpr auto arrayFlags = rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind;
    if (skipWithoutUpdateAfterBind(device) || skipWithoutLargeArray(device, arrayFlags))
    {
        return;
    }
    auto gate = createGateTimeline(device);
    if (!gate.has_value())
    {
        SKIP("A Direct3D 12 device on the same adapter must export a timeline that this Vulkan device imports.");
    }
    LargeArrayFixture fixture(device, arrayFlags);
    const auto first = gate->firstValue;

    auto firstGate = device.enqueueSharedTimelineWait(rachis::QueueType::compute, gate->at(first + 1));
    REQUIRE(firstGate.has_value());
    const std::array firstSelections{Selection{.element = 0, .slot = 0}};
    auto firstSubmitted = device.submit(fixture.record(firstSelections));
    REQUIRE(firstSubmitted.has_value());
    CHECK(pending(device, *firstSubmitted));

    const auto beforeInPlace = device.bindingUpdateStatistics();
    REQUIRE(callReleasingGate(
                [&] { return update(device, fixture.set(), 1, 1, slice(fixture.values(), 2)); }, *gate, first + 2)
                .has_value());
    const auto inPlace = resourceDelta(beforeInPlace, device.bindingUpdateStatistics());
    CHECK(inPlace.prepared == 1);
    CHECK(inPlace.tablePublishes == 1);
    CHECK(inPlace.published == 1);
    CHECK(pending(device, *firstSubmitted));

    const std::array secondSelections{Selection{.element = 1, .slot = 1}};
    auto secondSubmitted = device.submit(fixture.record(secondSelections));
    REQUIRE(secondSubmitted.has_value());

    auto secondGate = device.enqueueSharedTimelineWait(rachis::QueueType::compute, gate->at(first + 2));
    REQUIRE(secondGate.has_value());
    const std::array thirdSelections{Selection{.element = 1, .slot = 2}};
    auto thirdSubmitted = device.submit(fixture.record(thirdSelections));
    REQUIRE(thirdSubmitted.has_value());

    const auto beforeVersioned = device.bindingUpdateStatistics();
    REQUIRE(callReleasingGate(
                [&] {
                    return update(device,
                                  fixture.set(),
                                  0,
                                  0,
                                  rachis::BufferBinding{
                                      fixture.results(), resultRangeSize, resultRangeSize, sizeof(std::uint32_t)});
                },
                *gate,
                first + 2)
                .has_value());
    const auto versioned = resourceDelta(beforeVersioned, device.bindingUpdateStatistics());
    CHECK(versioned.prepared == 1);
    CHECK(versioned.tablePublishes == 1);
    CHECK(versioned.published == inputCount + 1);

    const auto beforeNewVersion = device.bindingUpdateStatistics();
    REQUIRE(update(device, fixture.set(), 1, 2, slice(fixture.values(), 3)).has_value());
    const auto newVersion = resourceDelta(beforeNewVersion, device.bindingUpdateStatistics());
    CHECK(newVersion.prepared == 1);
    CHECK(newVersion.tablePublishes == 1);
    CHECK(newVersion.published == 1);
    CHECK(pending(device, *firstSubmitted));
    CHECK(pending(device, *thirdSubmitted));

    const std::array fourthSelections{Selection{.element = 2, .slot = 0}, Selection{.element = 1, .slot = 1}};
    auto fourth = fixture.record(fourthSelections);
    fixture.recordReadback(fourth);
    auto fourthSubmitted = device.submit(std::move(fourth));
    REQUIRE(fourthSubmitted.has_value());

    gate->signal(first + 2);
    for (const auto point : {*firstSubmitted, *secondSubmitted, *thirdSubmitted, *fourthSubmitted})
    {
        REQUIRE(device.wait(point).has_value());
    }

    CHECK(fixture.result(0) == sliceValues[0]);
    CHECK(fixture.result(sizeof(std::uint32_t)) == sliceValues[2]);
    CHECK(fixture.result(2 * sizeof(std::uint32_t)) == sliceValues[2]);
    CHECK(fixture.result(resultRangeSize) == sliceValues[3]);
    CHECK(fixture.result(resultRangeSize + sizeof(std::uint32_t)) == sliceValues[2]);

    checkCleanDiagnostics(device);
    CHECK(gate->device.diagnosticCounts().errors == 0);
}
#endif

TEST_CASE("Vulkan update-after-bind entries of a large array change in place before submission and in a new version "
          "while older versions are in flight",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][binding][compute]")
{
    auto created = rachis::createDevice(rachis::test::vulkanDeviceDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    constexpr auto arrayFlags = rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind;
    if (skipWithoutUpdateAfterBind(device) || skipWithoutLargeArray(device, arrayFlags))
    {
        return;
    }
    LargeArrayFixture fixture(device, arrayFlags);

    const std::array firstSelections{Selection{.element = 0, .slot = 0}};
    auto first = fixture.record(firstSelections);
    const std::array secondSelections{Selection{.element = 1, .slot = 1}};
    auto second = fixture.record(secondSelections);

    const auto beforeInPlace = device.bindingUpdateStatistics();
    REQUIRE(update(device, fixture.set(), 1, 1, slice(fixture.values(), 2)).has_value());
    const auto inPlace = resourceDelta(beforeInPlace, device.bindingUpdateStatistics());
    CHECK(inPlace.prepared == 1);
    CHECK(inPlace.tablePublishes == 1);
    CHECK(inPlace.published == 1);

    const std::array thirdSelections{Selection{.element = 1, .slot = 2}};
    auto third = fixture.record(thirdSelections);
    auto firstSubmitted = device.submit(std::move(first));
    REQUIRE(firstSubmitted.has_value());
    auto secondSubmitted = device.submit(std::move(second));
    REQUIRE(secondSubmitted.has_value());
    auto thirdSubmitted = device.submit(std::move(third));
    REQUIRE(thirdSubmitted.has_value());

    const auto beforeVersioned = device.bindingUpdateStatistics();
    REQUIRE(update(device,
                   fixture.set(),
                   0,
                   0,
                   rachis::BufferBinding{fixture.results(), resultRangeSize, resultRangeSize, sizeof(std::uint32_t)})
                .has_value());
    const auto versioned = resourceDelta(beforeVersioned, device.bindingUpdateStatistics());
    CHECK(versioned.prepared == 1);
    CHECK(versioned.tablePublishes == 1);
    CHECK(versioned.published == inputCount + 1);

    const auto beforeNewVersion = device.bindingUpdateStatistics();
    REQUIRE(update(device, fixture.set(), 1, 2, slice(fixture.values(), 3)).has_value());
    const auto newVersion = resourceDelta(beforeNewVersion, device.bindingUpdateStatistics());
    CHECK(newVersion.prepared == 1);
    CHECK(newVersion.tablePublishes == 1);
    CHECK(newVersion.published == 1);

    const std::array fourthSelections{Selection{.element = 2, .slot = 0}, Selection{.element = 1, .slot = 1}};
    auto fourth = fixture.record(fourthSelections);
    fixture.recordReadback(fourth);
    auto fourthSubmitted = device.submit(std::move(fourth));
    REQUIRE(fourthSubmitted.has_value());
    for (const auto point : {*firstSubmitted, *secondSubmitted, *thirdSubmitted, *fourthSubmitted})
    {
        REQUIRE(device.wait(point).has_value());
    }

    CHECK(fixture.result(0) == sliceValues[0]);
    CHECK(fixture.result(sizeof(std::uint32_t)) == sliceValues[2]);
    CHECK(fixture.result(2 * sizeof(std::uint32_t)) == sliceValues[2]);
    CHECK(fixture.result(resultRangeSize) == sliceValues[3]);
    CHECK(fixture.result(resultRangeSize + sizeof(std::uint32_t)) == sliceValues[2]);

    checkCleanDiagnostics(device);
}

TEST_CASE("Vulkan update-after-bind elements written in place stay alive for work that bound the set before the "
          "update",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][binding][compute]")
{
    auto created = rachis::createDevice(rachis::test::vulkanDeviceDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    constexpr auto arrayFlags = rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind;
    if (skipWithoutUpdateAfterBind(device) || skipWithoutLargeArray(device, arrayFlags))
    {
        return;
    }
    LargeArrayFixture fixture(device, arrayFlags);
    const auto replacement = fixture.createValueBuffer(replacementValues, "Vulkan update-after-bind replacement");

    const std::array selections{Selection{.element = 1, .slot = 1}};
    auto command = fixture.record(selections);
    fixture.recordReadback(command);

    const auto beforeInPlace = device.bindingUpdateStatistics();
    REQUIRE(update(device, fixture.set(), 1, 1, slice(replacement, 0)).has_value());
    CHECK(resourceDelta(beforeInPlace, device.bindingUpdateStatistics()).published == 1);

    auto submitted = device.submit(std::move(command));
    REQUIRE(submitted.has_value());
    const auto live = allocationCount(device);
    fixture.destroySet();
    fixture.destroyValues();
    REQUIRE(device.destroyBuffer(replacement).has_value());
    CHECK(allocationCount(device) == live);

    REQUIRE(device.wait(*submitted).has_value());
    CHECK(allocationCount(device) == live - 2);
    CHECK(fixture.result(sizeof(std::uint32_t)) == replacementValues[0]);

    checkCleanDiagnostics(device);
}

TEST_CASE("Vulkan unretained command pools do not keep update-after-bind elements written in place alive",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][binding][compute]")
{
    auto created = rachis::createDevice(rachis::test::vulkanDeviceDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    constexpr auto arrayFlags = rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind;
    if (skipWithoutUpdateAfterBind(device) || skipWithoutLargeArray(device, arrayFlags))
    {
        return;
    }
    LargeArrayFixture fixture(device, arrayFlags);
    const auto replacement = fixture.createValueBuffer(replacementValues, "Vulkan update-after-bind replacement");
    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::compute,
        .maxCommandBuffers = 1,
        .label = "Vulkan unretained update-after-bind pool",
        .flags = rachis::CommandPoolFlag::unretainedResources,
    });
    REQUIRE(pool.has_value());

    {
        auto command = device.acquireCommandBuffer(*pool);
        REQUIRE(command.has_value());
        const std::array selections{Selection{.element = 1, .slot = 1}};
        fixture.recordInto(*command, selections);
        REQUIRE(update(device, fixture.set(), 1, 1, slice(replacement, 0)).has_value());
        const auto live = allocationCount(device);
        fixture.destroySet();
        REQUIRE(device.destroyBuffer(replacement).has_value());
        CHECK(allocationCount(device) == live - 1);
    }

    REQUIRE(device.destroyCommandPool(*pool).has_value());
    checkCleanDiagnostics(device);
}

TEST_CASE("Vulkan update-after-bind writes cost the same in a four-element array and a large array",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][binding][compute]")
{
    if (!ThreadAllocatedBytes::available())
    {
        SKIP("Counting host allocation sizes needs the MSVC debug heap.");
    }
    auto created = rachis::createDevice(rachis::test::vulkanDeviceDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    constexpr auto arrayFlags = rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind;
    if (skipWithoutUpdateAfterBind(device) || skipWithoutLargeArray(device, arrayFlags))
    {
        return;
    }
    LargeArrayFixture fixture(device, arrayFlags);

    constexpr std::uint32_t smallCount = 4;
    const std::array smallEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{
            1, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute, smallCount, arrayFlags},
    };
    auto smallLayout = device.createBindingLayout({.entries = smallEntries});
    REQUIRE(smallLayout.has_value());
    const std::array smallSizes{
        rachis::BindingPoolSize{rachis::BindingType::storageBuffer, 1},
        rachis::BindingPoolSize{rachis::BindingType::readOnlyStorageBuffer, smallCount},
    };
    auto smallPool = device.createBindingPool({.maxBindingSets = 1, .sizes = smallSizes});
    REQUIRE(smallPool.has_value());
    const std::array smallWrites{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{fixture.results(), 0, resultRangeSize, sizeof(std::uint32_t)}},
        rachis::BindingWrite{1, 0, slice(fixture.values(), 0)},
    };
    auto smallSet = device.createBindingSet({.layout = *smallLayout, .writes = smallWrites, .pool = *smallPool});
    REQUIRE(smallSet.has_value());

    const std::array selections{Selection{.element = 0, .slot = 0}};
    auto pending = fixture.record(selections);
    const auto updateCost = [&](rachis::BindingSetHandle set, std::uint32_t element) {
        rachis::Status updated;
        const auto bytes = ThreadAllocatedBytes::measure(
            [&] { updated = update(device, set, 1, element, slice(fixture.values(), 3)); });
        REQUIRE(updated.has_value());
        return bytes;
    };
    static_cast<void>(updateCost(*smallSet, 1));
    static_cast<void>(updateCost(fixture.set(), 1));
    const auto smallBytes = updateCost(*smallSet, 2);
    const auto largeBytes = updateCost(fixture.set(), 2);
    CAPTURE(smallBytes, largeBytes);
    CHECK(largeBytes <= smallBytes + 4096);

    static_cast<void>(pending);
    REQUIRE(device.destroyBindingSet(*smallSet).has_value());
    REQUIRE(device.destroyBindingPool(*smallPool).has_value());
    REQUIRE(device.destroyBindingLayout(*smallLayout).has_value());
    checkCleanDiagnostics(device);
}

TEST_CASE("Vulkan binding set updates count every variable descriptor they republish",
          "[size:medium][rhi][vulkan][binding]")
{
    auto created = rachis::createDevice(rachis::test::vulkanDeviceDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (rachis::test::skipUnlessReferenceDesktopAdvertises(capabilities.variableDescriptorCountBindingArrays,
                                                           "variableDescriptorCountBindingArrays"))
    {
        return;
    }
    auto values = device.createBuffer({
        .size = 64 * sliceSize,
        .usage = rachis::BufferUsage::storage,
        .label = "Vulkan variable-count statistics values",
    });
    REQUIRE(values.has_value());

    constexpr std::uint32_t variableCount = 64;
    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1,
                                   rachis::BindingType::readOnlyStorageBuffer,
                                   rachis::ShaderStage::compute,
                                   rachis::unboundedBindingCount,
                                   rachis::BindingFlag::partiallyBound | rachis::BindingFlag::variableDescriptorCount},
    };
    auto layout = device.createBindingLayout({.entries = entries});
    REQUIRE(layout.has_value());
    const std::array sizes{
        rachis::BindingPoolSize{rachis::BindingType::storageBuffer, 2},
        rachis::BindingPoolSize{rachis::BindingType::readOnlyStorageBuffer, 2 * variableCount},
    };
    auto pool = device.createBindingPool({.maxBindingSets = 1, .sizes = sizes});
    REQUIRE(pool.has_value());
    std::vector<rachis::BindingWrite> writes{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*values, 0, sliceSize, sizeof(std::uint32_t)}},
    };
    for (std::uint32_t element = 0; element < variableCount; ++element)
    {
        writes.push_back(rachis::BindingWrite{1, element, slice(*values, element)});
    }
    const auto beforeCreate = device.bindingUpdateStatistics();
    auto set = device.createBindingSet({
        .layout = *layout,
        .writes = writes,
        .variableDescriptorCount = variableCount,
        .pool = *pool,
    });
    REQUIRE(set.has_value());
    CHECK(resourceDelta(beforeCreate, device.bindingUpdateStatistics()).published == variableCount + 1);

    const auto beforeReplace = device.bindingUpdateStatistics();
    REQUIRE(update(device, *set, 0, 0, rachis::BufferBinding{*values, sliceSize, sliceSize, sizeof(std::uint32_t)})
                .has_value());
    const auto replaced = resourceDelta(beforeReplace, device.bindingUpdateStatistics());
    CHECK(replaced.prepared == 1);
    CHECK(replaced.tablePublishes == 1);
    CHECK(replaced.published == variableCount + 1);

    REQUIRE(device.destroyBindingSet(*set).has_value());
    REQUIRE(device.destroyBindingPool(*pool).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroyBuffer(*values).has_value());
    checkCleanDiagnostics(device);
}
#endif
