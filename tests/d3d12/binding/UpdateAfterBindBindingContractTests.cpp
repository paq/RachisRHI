#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
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
                .label = "Large binding array shader",
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
            auto layout = device_.createBindingLayout({.entries = entries, .label = "Large binding array layout"});
            const std::string layoutFailure = layout ? std::string{} : layout.error().message;
            CAPTURE(layoutFailure);
            REQUIRE(layout.has_value());
            layout_ = *layout;

            const std::array sizes{
                rachis::BindingPoolSize{rachis::BindingType::storageBuffer, 4},
                rachis::BindingPoolSize{rachis::BindingType::readOnlyStorageBuffer, 3 * inputCount},
            };
            auto pool = device_.createBindingPool({.maxBindingSets = 1, .sizes = sizes, .label = "Large array pool"});
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

} // namespace

TEST_CASE("binding set updates republish a large array in full while work in flight may read the set",
          "[size:medium][rhi][binding][compute][warp]")
{
    auto device = createWarpDevice();
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
    CHECK(idle.published == 1);

    checkCleanDiagnostics(device);
}

TEST_CASE("update-after-bind entries of a large array change in place while work that does not read them is in flight",
          "[size:medium][rhi][binding][compute][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);
    LargeArrayFixture fixture(device, rachis::BindingFlag::partiallyBound | rachis::BindingFlag::updateAfterBind);

    const std::array firstSelections{Selection{.element = 0, .slot = 0}};
    auto first = fixture.record(firstSelections);
    auto firstSubmitted = device.submit(std::move(first));
    REQUIRE(firstSubmitted.has_value());
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
    auto thirdSubmitted = device.submit(std::move(third));
    REQUIRE(thirdSubmitted.has_value());

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
TEST_CASE("update-after-bind writes in place cannot name a sparse buffer on Direct3D 12",
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
    const auto unchanged = resourceDelta(before, device.bindingUpdateStatistics());
    CHECK(unchanged.published == 0);

    REQUIRE(device.destroyBuffer(*sparse).has_value());
    checkCleanDiagnostics(device);
}
