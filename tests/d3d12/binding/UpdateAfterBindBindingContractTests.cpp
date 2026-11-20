#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <Windows.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace
{

    constexpr std::uint32_t inputCount = 100'000;
    constexpr std::uint64_t sliceSize = 256;
    constexpr std::array<std::uint32_t, 4> sliceValues{100, 200, 300, 400};
    constexpr std::array<std::uint32_t, 1> replacementValues{500};
    constexpr std::uint64_t resultRangeSize = 256;
    constexpr std::uint64_t resultBufferSize = 2 * resultRangeSize;
    constexpr std::uint64_t manualResidencyBufferSize = 64ULL * 1024;
    constexpr std::string_view valuesLabel = "Large array values";
    constexpr std::string_view replacementLabel = "Large array update-after-bind replacement";

    struct Selection final
    {
        std::uint32_t element = 0;
        std::uint32_t slot = 0;
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

    [[nodiscard]] rachis::Device createWarpDevice()
    {
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .adapterPreference = rachis::AdapterPreference::warp,
            .validation = rachis::ValidationLevel::standard,
            .enableDred = true,
            .diagnosticCallback = &printDiagnostic,
        });
        if (!created)
        {
            INFO(created.error().message);
        }
        REQUIRE(created.has_value());
        return std::move(*created);
    }

    [[nodiscard]] std::vector<std::byte> loadShader()
    {
        std::ifstream stream(RACHIS_TEST_UPDATE_AFTER_BIND_BINDING_SHADER_PATH, std::ios::binary | std::ios::ate);
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
        LargeArrayFixture(rachis::Device& device, rachis::BindingFlag arrayFlags, bool withSampler = false)
            : device_(device)
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
                .label = "Large binding array shader",
                .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, requirements, pushConstants),
            });
            REQUIRE(shader.has_value());
            shader_ = *shader;

            createBuffers();

            const std::array allEntries{
                rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
                rachis::BindingLayoutEntry{1,
                                           rachis::BindingType::readOnlyStorageBuffer,
                                           rachis::ShaderStage::compute,
                                           inputCount,
                                           arrayFlags},
                rachis::BindingLayoutEntry{2, rachis::BindingType::sampler, rachis::ShaderStage::compute, 1},
            };
            const auto entries = std::span(allEntries).first(withSampler ? 3U : 2U);
            auto layout = device_.createBindingLayout({.entries = entries, .label = "Large binding array layout"});
            const std::string layoutFailure = layout ? std::string{} : layout.error().message;
            CAPTURE(layoutFailure);
            REQUIRE(layout.has_value());
            layout_ = *layout;

            const std::array sizes{
                rachis::BindingPoolSize{rachis::BindingType::storageBuffer, 4},
                rachis::BindingPoolSize{rachis::BindingType::readOnlyStorageBuffer, 3 * inputCount},
                rachis::BindingPoolSize{rachis::BindingType::sampler, 4},
            };
            auto pool = device_.createBindingPool({.maxBindingSets = 1, .sizes = sizes, .label = "Large array pool"});
            REQUIRE(pool.has_value());
            pool_ = *pool;

            if (withSampler)
            {
                for (auto& sampler : samplers_)
                {
                    auto created = device_.createSampler({.label = "Large array sampler"});
                    REQUIRE(created.has_value());
                    sampler = *created;
                }
            }
            const std::array allWrites{
                rachis::BindingWrite{0, 0, rachis::BufferBinding{results_, 0, resultRangeSize, sizeof(std::uint32_t)}},
                rachis::BindingWrite{1, 0, slice(values_, 0)},
                rachis::BindingWrite{1, 1, slice(values_, 1)},
                rachis::BindingWrite{2, 0, samplers_[0]},
            };
            const auto writes = std::span(allWrites).first(withSampler ? 4U : 3U);
            auto set = device_.createBindingSet({
                .layout = layout_,
                .writes = writes,
                .label = "Large binding array set",
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
            for (const auto sampler : samplers_)
            {
                if (sampler.isValid())
                {
                    static_cast<void>(device_.destroySampler(sampler));
                }
            }
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

        [[nodiscard]] rachis::SamplerHandle alternateSampler() const noexcept
        {
            return samplers_[1];
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

        [[nodiscard]] rachis::BufferHandle createValueBuffer(
            std::span<const std::uint32_t> values,
            std::string_view label,
            rachis::ResourceAllocationFlag allocationFlags = rachis::ResourceAllocationFlag::none)
        {
            const auto size = rachis::hasAny(allocationFlags, rachis::ResourceAllocationFlag::manualResidency)
                                  ? manualResidencyBufferSize
                                  : values.size() * sliceSize;
            auto buffer = device_.createBuffer({
                .size = size,
                .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copyDestination,
                .allocationFlags = allocationFlags,
                .label = label,
            });
            auto staging = device_.createBuffer({
                .size = size,
                .usage = rachis::BufferUsage::copySource,
                .memoryType = rachis::MemoryType::upload,
                .label = "Large array value staging",
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

        [[nodiscard]] rachis::CommandBuffer record(std::span<const Selection> selections,
                                                   rachis::QueueType queue = rachis::QueueType::compute)
        {
            auto command = rachis::test::acquireCommandBuffer(device_, queue);
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
            auto mapped = device_.mapBuffer(readback_);
            REQUIRE(mapped.has_value());
            std::uint32_t value = 0;
            std::memcpy(&value, mapped->data + offset, sizeof(value));
            REQUIRE(device_.unmapBuffer(readback_).has_value());
            return value;
        }

    private:
        void createBuffers()
        {
            auto results = device_.createBuffer({
                .size = resultBufferSize,
                .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
                .label = "Large array results",
            });
            auto readback = device_.createBuffer({
                .size = resultBufferSize,
                .usage = rachis::BufferUsage::copyDestination,
                .memoryType = rachis::MemoryType::readback,
                .label = "Large array readback",
            });
            REQUIRE(results.has_value());
            REQUIRE(readback.has_value());
            results_ = *results;
            readback_ = *readback;
            values_ = createValueBuffer(sliceValues, valuesLabel);
        }

        rachis::Device& device_;
        rachis::ShaderModuleHandle shader_{};
        rachis::BufferHandle values_{};
        rachis::BufferHandle results_{};
        rachis::BufferHandle readback_{};
        rachis::BindingLayoutHandle layout_{};
        rachis::BindingPoolHandle pool_{};
        rachis::BindingSetHandle set_{};
        std::array<rachis::SamplerHandle, 2> samplers_{};
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

    [[nodiscard]] std::optional<rachis::ResourceMemoryEntry> memoryEntry(const rachis::Device& device,
                                                                         std::string_view label)
    {
        const auto snapshot = device.captureResourceMemorySnapshot();
        REQUIRE(snapshot.has_value());
        const auto found = std::ranges::find(snapshot->entries, label, &rachis::ResourceMemoryEntry::label);
        if (found == snapshot->entries.end())
        {
            return std::nullopt;
        }
        return *found;
    }

    void checkRetainedAfterDestruction(const rachis::Device& device, std::string_view label)
    {
        CAPTURE(label);
        const auto entry = memoryEntry(device, label);
        REQUIRE(entry.has_value());
        CHECK_FALSE(entry->logicalAlive);
    }

    class OwnedHandle final
    {
    public:
        OwnedHandle() = default;

        ~OwnedHandle()
        {
            if (value_ != nullptr)
            {
                static_cast<void>(CloseHandle(value_));
            }
        }

        OwnedHandle(const OwnedHandle&) = delete;
        OwnedHandle& operator=(const OwnedHandle&) = delete;
        OwnedHandle(OwnedHandle&&) = delete;
        OwnedHandle& operator=(OwnedHandle&&) = delete;

        [[nodiscard]] HANDLE* put() noexcept
        {
            return &value_;
        }

        [[nodiscard]] HANDLE get() const noexcept
        {
            return value_;
        }

    private:
        HANDLE value_ = nullptr;
    };

    class QueueBlocker final
    {
    public:
        explicit QueueBlocker(rachis::Device& device, rachis::QueueType queue = rachis::QueueType::compute)
        {
            REQUIRE(rachis::test::requireCapabilities(device).sharedTimelineHandles);
            Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
            REQUIRE(SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))));
            Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
            REQUIRE(SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))));
            Microsoft::WRL::ComPtr<ID3D12Device> native;
            REQUIRE(SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&native))));
            REQUIRE(SUCCEEDED(native->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence_))));
            REQUIRE(SUCCEEDED(native->CreateSharedHandle(fence_.Get(), nullptr, GENERIC_ALL, nullptr, handle_.put())));
            const rachis::SharedTimelinePointView point{
                .handle =
                    rachis::SharedHandleView{
                        .type = rachis::SharedHandleType::windowsNt,
                        .value = std::bit_cast<std::uintptr_t>(handle_.get()),
                    },
                .value = 1,
            };
            auto waited = device.enqueueSharedTimelineWait(queue, point);
            if (!waited)
            {
                UNSCOPED_INFO(waited.error().message);
            }
            REQUIRE(waited.has_value());
        }

        ~QueueBlocker()
        {
            release();
        }

        QueueBlocker(const QueueBlocker&) = delete;
        QueueBlocker& operator=(const QueueBlocker&) = delete;
        QueueBlocker(QueueBlocker&&) = delete;
        QueueBlocker& operator=(QueueBlocker&&) = delete;

        void release() noexcept
        {
            if (fence_ != nullptr && fence_->GetCompletedValue() < 1)
            {
                static_cast<void>(fence_->Signal(1));
            }
        }

    private:
        Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
        OwnedHandle handle_;
    };

} // namespace

TEST_CASE("binding set updates republish a large array in full while work in flight may read the set",
          "[size:medium][rhi][binding][compute][warp]")
{
    auto device = createWarpDevice();
    LargeArrayFixture fixture(device, rachis::BindingFlag::partiallyBound);
    QueueBlocker blocker(device);

    const std::array firstSelections{Selection{.element = 1, .slot = 0}};
    auto firstSubmitted = device.submit(fixture.record(firstSelections));
    REQUIRE(firstSubmitted.has_value());

    const auto beforeInFlight = device.bindingUpdateStatistics();
    REQUIRE(update(device, fixture.set(), 1, 1, slice(fixture.values(), 2)).has_value());
    const auto inFlight = resourceDelta(beforeInFlight, device.bindingUpdateStatistics());
    CHECK(inFlight.prepared == 1);
    CHECK(inFlight.tablePublishes == 1);
    CHECK(inFlight.published == inputCount + 1);

    const std::array secondSelections{Selection{.element = 1, .slot = 1}};
    auto second = fixture.record(secondSelections);
    fixture.recordReadback(second);
    auto secondSubmitted = device.submit(std::move(second));
    REQUIRE(secondSubmitted.has_value());
    blocker.release();
    REQUIRE(device.wait(*secondSubmitted).has_value());
    REQUIRE(device.wait(*firstSubmitted).has_value());
    device.collectGarbage();
    CHECK(fixture.result(0) == sliceValues[1]);
    CHECK(fixture.result(sizeof(std::uint32_t)) == sliceValues[2]);

    const auto beforeIdle = device.bindingUpdateStatistics();
    REQUIRE(update(device, fixture.set(), 1, 1, slice(fixture.values(), 1)).has_value());
    const auto idle = resourceDelta(beforeIdle, device.bindingUpdateStatistics());
    CHECK(idle.prepared == 1);
    CHECK(idle.tablePublishes == 1);
    CHECK(idle.published == 1);

    checkCleanDiagnostics(device);
}

TEST_CASE("update-after-bind entries of a large array change in place while work that does not read them is in flight",
          "[size:medium][rhi][binding][compute][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);
    LargeArrayFixture fixture(device, rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind);
    QueueBlocker blocker(device);

    const std::array firstSelections{Selection{.element = 0, .slot = 0}};
    auto firstSubmitted = device.submit(fixture.record(firstSelections));
    REQUIRE(firstSubmitted.has_value());
    const std::array secondSelections{Selection{.element = 1, .slot = 1}};
    auto second = fixture.record(secondSelections);

    const auto copiesBefore = rachis::d3d12::test::bindingSetVersionCopyCount().load(std::memory_order_relaxed);
    const auto beforeInPlace = device.bindingUpdateStatistics();
    REQUIRE(update(device, fixture.set(), 1, 1, slice(fixture.values(), 2)).has_value());
    const auto inPlace = resourceDelta(beforeInPlace, device.bindingUpdateStatistics());
    CHECK(inPlace.prepared == 1);
    CHECK(inPlace.tablePublishes == 1);
    CHECK(inPlace.published == 1);
    CHECK(rachis::d3d12::test::bindingSetVersionCopyCount().load(std::memory_order_relaxed) == copiesBefore);

    const std::array thirdSelections{Selection{.element = 1, .slot = 2}, Selection{.element = 2, .slot = 3}};
    auto third = fixture.record(thirdSelections);
    auto secondSubmitted = device.submit(std::move(second));
    REQUIRE(secondSubmitted.has_value());

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
    CHECK(rachis::d3d12::test::bindingSetVersionCopyCount().load(std::memory_order_relaxed) == copiesBefore + 1);

    const auto beforeNewVersion = device.bindingUpdateStatistics();
    REQUIRE(update(device, fixture.set(), 1, 2, slice(fixture.values(), 3)).has_value());
    const auto newVersion = resourceDelta(beforeNewVersion, device.bindingUpdateStatistics());
    CHECK(newVersion.prepared == 1);
    CHECK(newVersion.tablePublishes == 1);
    CHECK(newVersion.published == 1);
    auto thirdSubmitted = device.submit(std::move(third));
    REQUIRE(thirdSubmitted.has_value());

    const std::array fourthSelections{Selection{.element = 2, .slot = 0}, Selection{.element = 1, .slot = 1}};
    auto fourth = fixture.record(fourthSelections);
    fixture.recordReadback(fourth);
    auto fourthSubmitted = device.submit(std::move(fourth));
    REQUIRE(fourthSubmitted.has_value());
    blocker.release();
    for (const auto point : {*firstSubmitted, *secondSubmitted, *thirdSubmitted, *fourthSubmitted})
    {
        REQUIRE(device.wait(point).has_value());
    }

    CHECK(fixture.result(0) == sliceValues[0]);
    CHECK(fixture.result(sizeof(std::uint32_t)) == sliceValues[2]);
    CHECK(fixture.result(2 * sizeof(std::uint32_t)) == sliceValues[2]);
    CHECK(fixture.result(3 * sizeof(std::uint32_t)) == 0);
    CHECK(fixture.result(resultRangeSize) == sliceValues[3]);
    CHECK(fixture.result(resultRangeSize + sizeof(std::uint32_t)) == sliceValues[2]);

    checkCleanDiagnostics(device);
}

TEST_CASE("update-after-bind writes in place share one binding set version across the work that binds it",
          "[size:medium][rhi][binding][compute][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);
    LargeArrayFixture fixture(device, rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind);

    constexpr std::uint32_t editCount = 16;
    const auto copiesBefore = rachis::d3d12::test::bindingSetVersionCopyCount().load(std::memory_order_relaxed);
    const auto before = device.bindingUpdateStatistics();
    std::vector<rachis::CommandBuffer> commands;
    for (std::uint32_t edit = 0; edit < editCount; ++edit)
    {
        const std::array selections{Selection{.element = 1, .slot = 0}};
        commands.push_back(fixture.record(selections));
        REQUIRE(
            update(device, fixture.set(), 1, 2 + edit, slice(fixture.values(), edit % sliceValues.size())).has_value());
    }
    const auto delta = resourceDelta(before, device.bindingUpdateStatistics());
    CHECK(delta.prepared == editCount);
    CHECK(delta.tablePublishes == editCount);
    CHECK(delta.published == editCount);
    CHECK(rachis::d3d12::test::bindingSetVersionCopyCount().load(std::memory_order_relaxed) == copiesBefore);

    auto last = std::move(commands.back());
    commands.pop_back();
    fixture.recordReadback(last);
    commands.clear();
    auto submitted = device.submit(std::move(last));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    CHECK(fixture.result(0) == sliceValues[1]);

    checkCleanDiagnostics(device);
}

TEST_CASE("an update-after-bind write after a sampler update does not reach work recorded before it",
          "[size:medium][rhi][binding][compute][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);
    LargeArrayFixture fixture(device, rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind, true);
    const auto replacement = fixture.createValueBuffer(replacementValues, replacementLabel);
    QueueBlocker blocker(device);

    const std::array selections{Selection{.element = 1, .slot = 0}};
    auto command = fixture.record(selections);
    fixture.recordReadback(command);
    REQUIRE(update(device, fixture.set(), 2, 0, fixture.alternateSampler()).has_value());
    REQUIRE(update(device, fixture.set(), 1, 1, slice(replacement, 0)).has_value());
    auto submitted = device.submit(std::move(command));
    REQUIRE(submitted.has_value());
    fixture.destroySet();
    REQUIRE(device.destroyBuffer(replacement).has_value());

    blocker.release();
    REQUIRE(device.wait(*submitted).has_value());
    CHECK(fixture.result(0) == sliceValues[1]);

    checkCleanDiagnostics(device);
}

TEST_CASE("an update-after-bind element written in place stays alive for work that bound the set before the update",
          "[size:medium][rhi][binding][compute][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);
    LargeArrayFixture fixture(device, rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind);
    const auto replacement = fixture.createValueBuffer(replacementValues, replacementLabel);

    const std::array selections{Selection{.element = 1, .slot = 1}};
    auto command = fixture.record(selections);
    fixture.recordReadback(command);

    const auto beforeInPlace = device.bindingUpdateStatistics();
    REQUIRE(update(device, fixture.set(), 1, 1, slice(replacement, 0)).has_value());
    CHECK(resourceDelta(beforeInPlace, device.bindingUpdateStatistics()).published == 1);

    auto submitted = device.submit(std::move(command));
    REQUIRE(submitted.has_value());
    fixture.destroySet();
    fixture.destroyValues();
    REQUIRE(device.destroyBuffer(replacement).has_value());
    checkRetainedAfterDestruction(device, replacementLabel);
    checkRetainedAfterDestruction(device, valuesLabel);

    REQUIRE(device.wait(*submitted).has_value());
    CHECK_FALSE(memoryEntry(device, replacementLabel).has_value());
    CHECK_FALSE(memoryEntry(device, valuesLabel).has_value());
    CHECK(fixture.result(sizeof(std::uint32_t)) == replacementValues[0]);

    checkCleanDiagnostics(device);
}

TEST_CASE("an update-after-bind element written in place stays alive for submitted work after it is replaced",
          "[size:medium][rhi][binding][compute][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);
    LargeArrayFixture fixture(device, rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind);
    const auto replacement = fixture.createValueBuffer(replacementValues, replacementLabel);
    QueueBlocker blocker(device);

    const std::array selections{Selection{.element = 0, .slot = 0}};
    auto command = fixture.record(selections);
    fixture.recordReadback(command);
    REQUIRE(update(device, fixture.set(), 1, 2, slice(replacement, 0)).has_value());
    auto submitted = device.submit(std::move(command));
    REQUIRE(submitted.has_value());

    REQUIRE(device.destroyBuffer(replacement).has_value());
    REQUIRE(update(device, fixture.set(), 1, 2, slice(fixture.values(), 3)).has_value());
    checkRetainedAfterDestruction(device, replacementLabel);

    blocker.release();
    REQUIRE(device.wait(*submitted).has_value());
    CHECK_FALSE(memoryEntry(device, replacementLabel).has_value());
    CHECK(fixture.result(0) == sliceValues[0]);

    checkCleanDiagnostics(device);
}

TEST_CASE("update-after-bind writes grow the elements kept for work in flight geometrically",
          "[size:medium][rhi][binding][compute][lifetime][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);
    LargeArrayFixture fixture(device, rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind);
    QueueBlocker blocker(device);

    const std::array selections{Selection{.element = 0, .slot = 0}};
    auto command = fixture.record(selections);
    fixture.recordReadback(command);
    auto submitted = device.submit(std::move(command));
    REQUIRE(submitted.has_value());

    constexpr std::uint32_t updates = 1000;
    std::size_t capacity = 0;
    std::uint32_t growths = 0;
    for (std::uint32_t index = 0; index < updates; ++index)
    {
        REQUIRE(update(device, fixture.set(), 1, 2 + index, slice(fixture.values(), index % sliceValues.size()))
                    .has_value());
        const auto storage = rachis::d3d12::test::lastInPlaceReplacedStorage();
        REQUIRE(storage.count == index + 1);
        if (storage.capacity != capacity)
        {
            capacity = storage.capacity;
            ++growths;
        }
    }
    CHECK(growths <= 16);

    blocker.release();
    REQUIRE(device.wait(*submitted).has_value());
    CHECK(fixture.result(0) == sliceValues[0]);
    checkCleanDiagnostics(device);
}

TEST_CASE("update-after-bind elements that no submission saw are released when they are replaced",
          "[size:medium][rhi][binding][compute][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);
    LargeArrayFixture fixture(device, rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind);
    const std::array selections{Selection{.element = 0, .slot = 0}};
    auto command = fixture.record(selections);
    fixture.recordReadback(command);
    constexpr std::array<std::string_view, 5> labels{
        "Streamed element 0", "Streamed element 1", "Streamed element 2", "Streamed element 3", "Streamed element 4"};
    for (const auto label : labels)
    {
        const auto streamed = fixture.createValueBuffer(replacementValues, label);
        REQUIRE(update(device, fixture.set(), 1, 2, slice(streamed, 0)).has_value());
        REQUIRE(device.destroyBuffer(streamed).has_value());
    }
    for (std::size_t index = 0; index + 1 < labels.size(); ++index)
    {
        CAPTURE(labels[index]);
        CHECK_FALSE(memoryEntry(device, labels[index]).has_value());
    }
    checkRetainedAfterDestruction(device, labels.back());

    QueueBlocker blocker(device);
    auto submitted = device.submit(std::move(command));
    REQUIRE(submitted.has_value());
    REQUIRE(update(device, fixture.set(), 1, 2, slice(fixture.values(), 3)).has_value());
    checkRetainedAfterDestruction(device, labels.back());
    blocker.release();
    REQUIRE(device.wait(*submitted).has_value());
    CHECK_FALSE(memoryEntry(device, labels.back()).has_value());
    CHECK(fixture.result(0) == sliceValues[0]);

    checkCleanDiagnostics(device);
}

TEST_CASE("work in flight keeps alive only the update-after-bind elements it could have seen",
          "[size:medium][rhi][binding][compute][lifetime][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);
    LargeArrayFixture fixture(device, rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind);
    constexpr std::string_view seenLabel = "Element seen by stalled work";
    const auto seen = fixture.createValueBuffer(replacementValues, seenLabel);
    REQUIRE(update(device, fixture.set(), 1, 1, slice(seen, 0)).has_value());

    QueueBlocker blocker(device, rachis::QueueType::graphics);
    const std::array stalledSelections{Selection{.element = 0, .slot = 0}};
    auto stalled = device.submit(fixture.record(stalledSelections, rachis::QueueType::graphics));
    REQUIRE(stalled.has_value());

    constexpr std::array<std::string_view, 4> labels{
        "Streamed after stalled work 0",
        "Streamed after stalled work 1",
        "Streamed after stalled work 2",
        "Streamed after stalled work 3",
    };
    for (const auto label : labels)
    {
        const auto streamed = fixture.createValueBuffer(replacementValues, label);
        REQUIRE(update(device, fixture.set(), 1, 1, slice(streamed, 0)).has_value());
        const std::array selections{Selection{.element = 1, .slot = 1}};
        auto reading = device.submit(fixture.record(selections));
        REQUIRE(reading.has_value());
        REQUIRE(device.wait(*reading).has_value());
        device.collectGarbage();
        REQUIRE(device.destroyBuffer(streamed).has_value());
    }
    REQUIRE(device.destroyBuffer(seen).has_value());
    checkRetainedAfterDestruction(device, seenLabel);
    for (std::size_t index = 0; index + 1 < labels.size(); ++index)
    {
        CAPTURE(labels[index]);
        CHECK_FALSE(memoryEntry(device, labels[index]).has_value());
    }
    checkRetainedAfterDestruction(device, labels.back());

    blocker.release();
    REQUIRE(device.wait(*stalled).has_value());
    device.collectGarbage();
    CHECK_FALSE(memoryEntry(device, seenLabel).has_value());
    checkCleanDiagnostics(device);
}

TEST_CASE("work that bound a set needs no residency of an update-after-bind element replaced before submission",
          "[size:medium][rhi][binding][compute][residency][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);
    REQUIRE(rachis::test::requireCapabilities(device).manualResidency);
    LargeArrayFixture fixture(device, rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind);
    const auto manual =
        fixture.createValueBuffer(replacementValues, replacementLabel, rachis::ResourceAllocationFlag::manualResidency);
    const std::array manualBuffers{manual};

    REQUIRE(update(device, fixture.set(), 1, 1, slice(manual, 0)).has_value());
    const std::array selections{Selection{.element = 1, .slot = 1}};
    auto command = fixture.record(selections);
    fixture.recordReadback(command);
    REQUIRE(update(device, fixture.set(), 1, 1, slice(fixture.values(), 2)).has_value());
    REQUIRE(device.evict({.buffers = manualBuffers}).has_value());

    auto submitted = device.submit(std::move(command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    CHECK(fixture.result(sizeof(std::uint32_t)) == sliceValues[2]);

    fixture.destroySet();
    REQUIRE(device.destroyBuffer(manual).has_value());
    checkCleanDiagnostics(device);
}

TEST_CASE("an update-after-bind element written in place must be resident when work that bound the set before the "
          "update is submitted",
          "[size:medium][rhi][binding][compute][residency][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);
    REQUIRE(rachis::test::requireCapabilities(device).manualResidency);
    LargeArrayFixture fixture(device, rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind);
    const auto replacement =
        fixture.createValueBuffer(replacementValues, replacementLabel, rachis::ResourceAllocationFlag::manualResidency);
    const std::array buffers{replacement};
    const rachis::ResidencySet residencySet{.buffers = buffers};

    const std::array selections{Selection{.element = 1, .slot = 1}};
    auto command = fixture.record(selections);
    fixture.recordReadback(command);
    REQUIRE(update(device, fixture.set(), 1, 1, slice(replacement, 0)).has_value());
    REQUIRE(device.evict(residencySet).has_value());

    auto refused = device.submit(std::move(command));
    REQUIRE_FALSE(refused.has_value());
    CHECK(refused.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(command.isValid());

    auto resident = device.enqueueMakeResident(rachis::QueueType::compute, residencySet);
    REQUIRE(resident.has_value());
    const std::array waits{*resident};
    auto submitted = device.submit(std::move(command), waits);
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    CHECK(fixture.result(sizeof(std::uint32_t)) == replacementValues[0]);

    fixture.destroySet();
    REQUIRE(device.destroyBuffer(replacement).has_value());
    checkCleanDiagnostics(device);
}

TEST_CASE("a refused submission keeps no residency of update-after-bind elements written in place",
          "[size:medium][rhi][binding][compute][residency][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);
    REQUIRE(rachis::test::requireCapabilities(device).manualResidency);
    LargeArrayFixture resident(device, rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind);
    LargeArrayFixture evicted(device, rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind);
    const auto residentBuffer = resident.createValueBuffer(
        replacementValues, replacementLabel, rachis::ResourceAllocationFlag::manualResidency);
    const auto evictedBuffer = evicted.createValueBuffer(
        replacementValues, "Evicted update-after-bind element", rachis::ResourceAllocationFlag::manualResidency);
    const std::array residentBuffers{residentBuffer};
    const std::array evictedBuffers{evictedBuffer};

    const std::array selections{Selection{.element = 1, .slot = 1}};
    std::array batch{resident.record(selections), evicted.record(selections)};
    resident.recordReadback(batch[0]);
    REQUIRE(update(device, resident.set(), 1, 1, slice(residentBuffer, 0)).has_value());
    REQUIRE(update(device, evicted.set(), 1, 1, slice(evictedBuffer, 0)).has_value());
    REQUIRE(device.evict({.buffers = evictedBuffers}).has_value());

    auto refused = device.submit(batch);
    REQUIRE_FALSE(refused.has_value());
    CHECK(refused.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(batch[0].isValid());
    REQUIRE(batch[1].isValid());

    REQUIRE(update(device, resident.set(), 1, 1, slice(resident.values(), 2)).has_value());
    REQUIRE(device.evict({.buffers = residentBuffers}).has_value());

    auto submitted = device.submit(std::move(batch[0]));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    CHECK(resident.result(sizeof(std::uint32_t)) == sliceValues[2]);

    resident.destroySet();
    evicted.destroySet();
    REQUIRE(device.destroyBuffer(residentBuffer).has_value());
    REQUIRE(device.destroyBuffer(evictedBuffer).has_value());
    checkCleanDiagnostics(device);
}

TEST_CASE("a submission gathers the residency of each update-after-bind set once however often it is bound",
          "[size:medium][rhi][binding][compute][residency][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);
    REQUIRE(rachis::test::requireCapabilities(device).manualResidency);
    LargeArrayFixture first(device, rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind);
    LargeArrayFixture second(device, rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind);
    const auto firstManual =
        first.createValueBuffer(replacementValues, replacementLabel, rachis::ResourceAllocationFlag::manualResidency);
    const auto secondManual = second.createValueBuffer(
        replacementValues, "Second manual update-after-bind element", rachis::ResourceAllocationFlag::manualResidency);
    REQUIRE(update(device, first.set(), 1, 1, slice(firstManual, 0)).has_value());
    REQUIRE(update(device, second.set(), 1, 1, slice(secondManual, 0)).has_value());

    constexpr std::size_t alternations = 32;
    const std::array selections{Selection{.element = 1, .slot = 0}};
    auto command = first.record(selections);
    for (std::size_t index = 0; index < alternations; ++index)
    {
        second.recordInto(command, selections);
        first.recordInto(command, selections);
    }
    first.recordReadback(command);

    auto& gathered = rachis::d3d12::test::inPlaceResidencyGatherCount();
    const auto before = gathered.load(std::memory_order_relaxed);
    auto submitted = device.submit(std::move(command));
    REQUIRE(submitted.has_value());
    CHECK(gathered.load(std::memory_order_relaxed) - before == 2);
    REQUIRE(device.wait(*submitted).has_value());
    CHECK(first.result(0) == replacementValues[0]);

    first.destroySet();
    second.destroySet();
    REQUIRE(device.destroyBuffer(firstManual).has_value());
    REQUIRE(device.destroyBuffer(secondManual).has_value());
    checkCleanDiagnostics(device);
}

namespace
{

    struct ConcurrentInPlaceUpdate final
    {
        rachis::Device* device = nullptr;
        rachis::BindingSetHandle set = {};
        rachis::BufferHandle buffer = {};
        std::future<rachis::Status> result;
        bool blockedBySubmission = false;
        bool finishedDuringSubmission = false;
    };

    ConcurrentInPlaceUpdate* g_concurrentUpdate = nullptr;

    void startUpdateDuringSubmission() noexcept
    {
        auto& state = *g_concurrentUpdate;
        auto& contention = rachis::d3d12::test::inPlacePublicationContentionCount();
        const auto before = contention.load(std::memory_order_acquire);
        state.result = std::async(std::launch::async,
                                  [&state] { return update(*state.device, state.set, 1, 1, slice(state.buffer, 0)); });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (contention.load(std::memory_order_acquire) == before && std::chrono::steady_clock::now() < deadline)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        state.blockedBySubmission = contention.load(std::memory_order_acquire) != before;
        state.finishedDuringSubmission = state.result.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

} // namespace

TEST_CASE("an update-after-bind write in place waits for a submission that is reading the set's residency",
          "[size:medium][rhi][binding][compute][residency][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);
    REQUIRE(rachis::test::requireCapabilities(device).manualResidency);
    LargeArrayFixture fixture(device, rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind);
    const auto evictedBuffer = fixture.createValueBuffer(
        replacementValues, "Evicted update-after-bind element", rachis::ResourceAllocationFlag::manualResidency);
    const std::array evictedBuffers{evictedBuffer};
    REQUIRE(device.evict({.buffers = evictedBuffers}).has_value());

    const std::array selections{Selection{.element = 0, .slot = 0}};
    auto command = fixture.record(selections);
    fixture.recordReadback(command);

    ConcurrentInPlaceUpdate concurrent{.device = &device, .set = fixture.set(), .buffer = evictedBuffer};
    g_concurrentUpdate = &concurrent;
    rachis::d3d12::test::inPlaceSubmissionObserver().store(&startUpdateDuringSubmission, std::memory_order_release);
    auto submitted = device.submit(std::move(command));
    rachis::d3d12::test::inPlaceSubmissionObserver().store(nullptr, std::memory_order_release);
    g_concurrentUpdate = nullptr;
    REQUIRE(submitted.has_value());
    REQUIRE(concurrent.result.valid());
    CHECK(concurrent.result.get().has_value());
    CHECK(concurrent.blockedBySubmission);
    CHECK_FALSE(concurrent.finishedDuringSubmission);

    REQUIRE(device.wait(*submitted).has_value());
    CHECK(fixture.result(0) == sliceValues[0]);

    fixture.destroySet();
    REQUIRE(device.destroyBuffer(evictedBuffer).has_value());
    checkCleanDiagnostics(device);
}

TEST_CASE("a refused submission unlocks the update-after-bind publications it releases",
          "[size:medium][rhi][binding][compute][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);
    LargeArrayFixture fixture(device, rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind);

    const std::array selections{Selection{.element = 0, .slot = 0}};
    auto command = fixture.record(selections);
    REQUIRE(update(device,
                   fixture.set(),
                   0,
                   0,
                   rachis::BufferBinding{fixture.results(), resultRangeSize, resultRangeSize, sizeof(std::uint32_t)})
                .has_value());

    const auto before = rachis::d3d12::test::lockedPublicationDestructionCount().load(std::memory_order_relaxed);
    rachis::d3d12::test::nextSubmissionCloseFailure().store(true, std::memory_order_relaxed);
    auto refused = device.submit(std::move(command));
    rachis::d3d12::test::nextSubmissionCloseFailure().store(false, std::memory_order_relaxed);
    REQUIRE_FALSE(refused.has_value());
    CHECK(refused.error().code == rachis::ErrorCode::validationError);
    CHECK(rachis::d3d12::test::lockedPublicationDestructionCount().load(std::memory_order_relaxed) == before);
    device.collectGarbage();
}

TEST_CASE("unretained command pools do not keep update-after-bind elements written in place alive",
          "[size:medium][rhi][binding][compute][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);
    LargeArrayFixture fixture(device, rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind);
    const auto replacement = fixture.createValueBuffer(replacementValues, replacementLabel);
    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::compute,
        .maxCommandBuffers = 1,
        .label = "Unretained update-after-bind pool",
        .flags = rachis::CommandPoolFlag::unretainedResources,
    });
    REQUIRE(pool.has_value());

    {
        auto command = device.acquireCommandBuffer(*pool);
        REQUIRE(command.has_value());
        const std::array selections{Selection{.element = 1, .slot = 1}};
        fixture.recordInto(*command, selections);
        REQUIRE(update(device, fixture.set(), 1, 1, slice(replacement, 0)).has_value());
        fixture.destroySet();
        REQUIRE(device.destroyBuffer(replacement).has_value());
        CHECK_FALSE(memoryEntry(device, replacementLabel).has_value());
    }

    REQUIRE(device.destroyCommandPool(*pool).has_value());
    checkCleanDiagnostics(device);
}

TEST_CASE("update-after-bind writes cannot name a sparse buffer on Direct3D 12",
          "[size:medium][rhi][binding][sparse][warp]")
{
    auto device = createWarpDevice();
    const auto capabilities = rachis::test::requireCapabilities(device);
    REQUIRE(capabilities.updateAfterBindBindingArrays);
    if (!capabilities.sparseBuffers)
    {
        SKIP("This adapter reports no sparse buffers.");
    }
    LargeArrayFixture fixture(device, rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind);
    auto sparse = device.createSparseBuffer({
        .size = 64ULL * 1024ULL,
        .usage = rachis::BufferUsage::storage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Update-after-bind sparse buffer",
    });
    REQUIRE(sparse.has_value());

    const auto before = device.bindingUpdateStatistics();
    const auto rejected = update(device, fixture.set(), 1, 1, slice(*sparse, 0));
    REQUIRE_FALSE(rejected.has_value());
    CHECK(rejected.error().code == rachis::ErrorCode::unsupported);

    const std::array mixed{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{fixture.results(), 0, resultRangeSize, sizeof(std::uint32_t)}},
        rachis::BindingWrite{1, 1, slice(*sparse, 0)},
    };
    const auto mixedRejected = device.updateBindingSet(fixture.set(), mixed);
    REQUIRE_FALSE(mixedRejected.has_value());
    CHECK(mixedRejected.error().code == rachis::ErrorCode::unsupported);
    const auto unchanged = resourceDelta(before, device.bindingUpdateStatistics());
    CHECK(unchanged.published == 0);

    REQUIRE(device.destroyBuffer(*sparse).has_value());
    checkCleanDiagnostics(device);
}
