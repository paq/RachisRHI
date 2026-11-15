#include "D3D12/D3D12Residency.h"
#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string_view>
#include <vector>

namespace
{

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    class MakeResidentFailureScope final
    {
    public:
        explicit MakeResidentFailureScope(rachis::d3d12::test::MakeResidentFailurePoint point) noexcept
            : previous_(rachis::d3d12::test::nextMakeResidentFailurePoint().exchange(static_cast<std::int32_t>(point),
                                                                                     std::memory_order_relaxed))
        {
        }

        ~MakeResidentFailureScope()
        {
            rachis::d3d12::test::nextMakeResidentFailurePoint().store(previous_, std::memory_order_relaxed);
        }

        MakeResidentFailureScope(const MakeResidentFailureScope&) = delete;
        MakeResidentFailureScope& operator=(const MakeResidentFailureScope&) = delete;
        MakeResidentFailureScope(MakeResidentFailureScope&&) = delete;
        MakeResidentFailureScope& operator=(MakeResidentFailureScope&&) = delete;

    private:
        std::int32_t previous_ = static_cast<std::int32_t>(rachis::d3d12::test::MakeResidentFailurePoint::none);
    };

    [[nodiscard]] const rachis::ResourceMemoryEntry* findByLabel(const rachis::ResourceMemorySnapshot& snapshot,
                                                                 std::string_view label)
    {
        const auto found = std::ranges::find(snapshot.entries, label, &rachis::ResourceMemoryEntry::label);
        return found != snapshot.entries.end() ? &*found : nullptr;
    }

} // namespace

TEST_CASE("manual residency is independently pageable and timeline synchronized", "[size:medium][rhi][residency][warp]")
{
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
    REQUIRE(rachis::test::requireCapabilities(device).manualResidency);

    auto automatic = device.createBuffer(rachis::BufferDesc{
        .size = 64ULL * 1024,
        .usage = rachis::BufferUsage::copyDestination,
        .label = "Automatic residency buffer",
    });
    REQUIRE(automatic.has_value());
    REQUIRE(device.queryResidency(*automatic) == rachis::ResidencyState::automatic);
    const std::array automaticBuffers{*automatic};
    auto automaticEviction = device.evict(rachis::ResidencySet{.buffers = automaticBuffers});
    REQUIRE_FALSE(automaticEviction.has_value());
    CHECK(automaticEviction.error().code == rachis::ErrorCode::invalidArgument);

    auto invalidUpload = device.createBuffer(rachis::BufferDesc{
        .size = 64ULL * 1024,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .allocationFlags = rachis::ResourceAllocationFlag::manualResidency,
        .label = "Invalid manual-residency upload",
    });
    REQUIRE_FALSE(invalidUpload.has_value());
    CHECK(invalidUpload.error().code == rachis::ErrorCode::invalidArgument);

    auto buffer = device.createBuffer(rachis::BufferDesc{
        .size = 64ULL * 1024,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .allocationFlags = rachis::ResourceAllocationFlag::manualResidency,
        .label = "Manual residency buffer",
    });
    auto texture = device.createTexture(rachis::TextureDesc{
        .width = 16,
        .height = 16,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .allocationFlags = rachis::ResourceAllocationFlag::manualResidency,
        .label = "Manual residency texture",
    });
    REQUIRE(buffer.has_value());
    REQUIRE(texture.has_value());
    REQUIRE(device.queryResidency(*buffer) == rachis::ResidencyState::resident);
    REQUIRE(device.queryResidency(*texture) == rachis::ResidencyState::resident);

    const std::array buffers{*buffer};
    const std::array textures{*texture};
    const rachis::ResidencySet resources{.buffers = buffers, .textures = textures};
    const std::array duplicateBuffers{*buffer, *buffer};
    auto duplicateEviction = device.evict(rachis::ResidencySet{.buffers = duplicateBuffers});
    REQUIRE_FALSE(duplicateEviction.has_value());
    CHECK(duplicateEviction.error().code == rachis::ErrorCode::invalidArgument);

    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .bufferBarrier(*command,
                                   rachis::BufferBarrier{
                                       .buffer = *buffer,
                                       .stagesBefore = rachis::PipelineStage::all,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::common,
                                       .accessAfter = rachis::Access::copyDestination,
                                   })
                    .has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = *texture,
                                        .stagesBefore = rachis::PipelineStage::none,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                    })
                    .has_value());
        auto inUseEviction = device.evict(resources);
        REQUIRE_FALSE(inUseEviction.has_value());
        CHECK(inUseEviction.error().code == rachis::ErrorCode::invalidState);
    }

    REQUIRE(device.evict(resources).has_value());
    REQUIRE(device.queryResidency(*buffer) == rachis::ResidencyState::evicted);
    REQUIRE(device.queryResidency(*texture) == rachis::ResidencyState::evicted);

    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        auto useWhileEvicted = device.bufferBarrier(*command,
                                                    rachis::BufferBarrier{
                                                        .buffer = *buffer,
                                                        .stagesBefore = rachis::PipelineStage::all,
                                                        .stagesAfter = rachis::PipelineStage::copy,
                                                        .accessBefore = rachis::Access::common,
                                                        .accessAfter = rachis::Access::copyDestination,
                                                    });
        REQUIRE_FALSE(useWhileEvicted.has_value());
        CHECK(useWhileEvicted.error().code == rachis::ErrorCode::invalidState);
    }

    auto residencyReady = device.enqueueMakeResident(rachis::QueueType::copy, resources);
    REQUIRE(residencyReady.has_value());
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *buffer,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::copyDestination,
                               })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::none,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    const std::array waits{*residencyReady};
    auto submitted = device.submit(std::move(*command), waits);
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();
    REQUIRE(device.queryResidency(*buffer) == rachis::ResidencyState::resident);
    REQUIRE(device.queryResidency(*texture) == rachis::ResidencyState::resident);

    {
        auto toCommon = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(toCommon.has_value());
        REQUIRE(device
                    .bufferBarrier(*toCommon,
                                   rachis::BufferBarrier{
                                       .buffer = *buffer,
                                       .stagesBefore = rachis::PipelineStage::copy,
                                       .stagesAfter = rachis::PipelineStage::all,
                                       .accessBefore = rachis::Access::copyDestination,
                                       .accessAfter = rachis::Access::common,
                                   })
                    .has_value());
        auto commonSubmitted = device.submit(std::move(*toCommon));
        REQUIRE(commonSubmitted.has_value());
        REQUIRE(device.wait(*commonSubmitted).has_value());
        device.collectGarbage();
    }

    std::array<rachis::BufferHandle, 4> copySinks{};
    for (auto& sink : copySinks)
    {
        auto createdSink = device.createBuffer(rachis::BufferDesc{
            .size = 64,
            .usage = rachis::BufferUsage::copyDestination,
            .label = "Residency concurrent read sink",
        });
        REQUIRE(createdSink.has_value());
        sink = *createdSink;
    }

    REQUIRE(device.evict(resources).has_value());
    auto sameQueueReady = device.enqueueMakeResident(rachis::QueueType::graphics, resources);
    REQUIRE(sameQueueReady.has_value());
    auto sameQueueCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(sameQueueCommand.has_value());
    REQUIRE(device
                .copyBuffer(*sameQueueCommand,
                            rachis::BufferCopy{
                                .source = *buffer,
                                .destination = copySinks[0],
                                .size = 64,
                            })
                .has_value());
    auto sameQueueSubmitted = device.submit(std::move(*sameQueueCommand));
    REQUIRE(sameQueueSubmitted.has_value());

    auto computeAfterSameQueue = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(computeAfterSameQueue.has_value());
    REQUIRE(device
                .copyBuffer(*computeAfterSameQueue,
                            rachis::BufferCopy{
                                .source = *buffer,
                                .destination = copySinks[1],
                                .size = 64,
                            })
                .has_value());
    const std::array originWaits{*sameQueueReady};
    auto computeFirstSubmitted = device.submit(std::move(*computeAfterSameQueue), originWaits);
    REQUIRE(computeFirstSubmitted.has_value());

    auto computeSecondCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(computeSecondCommand.has_value());
    REQUIRE(device
                .copyBuffer(*computeSecondCommand,
                            rachis::BufferCopy{
                                .source = *buffer,
                                .destination = copySinks[2],
                                .size = 64,
                            })
                .has_value());
    auto computeSecondSubmitted = device.submit(std::move(*computeSecondCommand));
    REQUIRE(computeSecondSubmitted.has_value());

    auto copyTransitiveCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(copyTransitiveCommand.has_value());
    REQUIRE(device
                .copyBuffer(*copyTransitiveCommand,
                            rachis::BufferCopy{
                                .source = *buffer,
                                .destination = copySinks[3],
                                .size = 64,
                            })
                .has_value());
    const std::array transitiveWaits{*computeFirstSubmitted};
    auto copyTransitiveSubmitted = device.submit(std::move(*copyTransitiveCommand), transitiveWaits);
    REQUIRE(copyTransitiveSubmitted.has_value());

    REQUIRE(device.wait(*sameQueueSubmitted).has_value());
    REQUIRE(device.wait(*computeSecondSubmitted).has_value());
    REQUIRE(device.wait(*copyTransitiveSubmitted).has_value());
    device.collectGarbage();
    REQUIRE(device.queryResidency(*buffer) == rachis::ResidencyState::resident);

    for (const auto sink : copySinks)
    {
        REQUIRE(device.destroyBuffer(sink).has_value());
    }

    REQUIRE(device.evict(resources).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyBuffer(*buffer).has_value());
    REQUIRE(device.destroyBuffer(*automatic).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("eviction releases command uses that already retired on the GPU", "[size:medium][rhi][residency][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).manualResidency);

    auto buffer = device.createBuffer(rachis::BufferDesc{
        .size = 64ULL * 1024,
        .usage = rachis::BufferUsage::copySource,
        .allocationFlags = rachis::ResourceAllocationFlag::manualResidency,
        .label = "Retired-use eviction buffer",
    });
    REQUIRE(buffer.has_value());
    auto sink = device.createBuffer(rachis::BufferDesc{
        .size = 64,
        .usage = rachis::BufferUsage::copyDestination,
        .label = "Retired-use eviction sink",
    });
    REQUIRE(sink.has_value());

    auto graphicsCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(graphicsCommand.has_value());
    REQUIRE(device
                .copyBuffer(*graphicsCommand,
                            rachis::BufferCopy{
                                .source = *buffer,
                                .destination = *sink,
                                .size = 64,
                            })
                .has_value());
    auto graphicsSubmitted = device.submit(std::move(*graphicsCommand));
    REQUIRE(graphicsSubmitted.has_value());

    auto copyCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(copyCommand.has_value());
    const std::array waits{*graphicsSubmitted};
    auto copySubmitted = device.submit(std::move(*copyCommand), waits);
    REQUIRE(copySubmitted.has_value());
    REQUIRE(device.waitIdle(rachis::QueueType::copy).has_value());

    const std::array buffers{*buffer};
    auto evicted = device.evict(rachis::ResidencySet{.buffers = buffers});
    if (!evicted)
    {
        UNSCOPED_INFO(evicted.error().message);
    }
    REQUIRE(evicted.has_value());
    REQUIRE(device.queryResidency(*buffer) == rachis::ResidencyState::evicted);

    REQUIRE(device.destroyBuffer(*sink).has_value());
    REQUIRE(device.destroyBuffer(*buffer).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("rejected make-resident leaves the evicted allocation unchanged",
          "[size:medium][rhi][residency][memory][failure][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).manualResidency);

    auto buffer = device.createBuffer(rachis::BufferDesc{
        .size = 64ULL * 1024,
        .usage = rachis::BufferUsage::copyDestination,
        .allocationFlags = rachis::ResourceAllocationFlag::manualResidency,
        .label = "Rejected make-resident buffer",
    });
    REQUIRE(buffer.has_value());
    const std::array buffers{*buffer};
    const rachis::ResidencySet resources{.buffers = buffers};
    REQUIRE(device.evict(resources).has_value());
    const auto before = device.captureResourceMemorySnapshot();
    REQUIRE(before.has_value());
    const auto* beforeEntry = findByLabel(*before, "Rejected make-resident buffer");
    REQUIRE(beforeEntry != nullptr);

    rachis::Result<rachis::TimelinePoint> scheduled;
    {
        const MakeResidentFailureScope failure(rachis::d3d12::test::MakeResidentFailurePoint::beforeNativeAcceptance);
        scheduled = device.enqueueMakeResident(rachis::QueueType::copy, resources);
    }
    REQUIRE_FALSE(scheduled.has_value());
    CHECK(scheduled.error().code == rachis::ErrorCode::nativeFailure);
    REQUIRE(device.queryResidency(*buffer) == rachis::ResidencyState::evicted);

    const auto rejected = device.captureResourceMemorySnapshot();
    REQUIRE(rejected.has_value());
    const auto* rejectedEntry = findByLabel(*rejected, "Rejected make-resident buffer");
    REQUIRE(rejectedEntry != nullptr);
    CHECK(rejectedEntry->allocationId == beforeEntry->allocationId);
    CHECK(rejectedEntry->logicalAlive);
    CHECK(rejected->liveAllocationBytes == before->liveAllocationBytes);
    CHECK(rejected->retainedAllocationBytes == before->retainedAllocationBytes);
    CHECK(rejected->liveAllocationCount == before->liveAllocationCount);
    CHECK(rejected->retainedAllocationCount == before->retainedAllocationCount);

    const auto recovered = device.enqueueMakeResident(rachis::QueueType::copy, resources);
    REQUIRE(recovered.has_value());
    REQUIRE(device.wait(*recovered).has_value());
    device.collectGarbage();
    REQUIRE(device.queryResidency(*buffer) == rachis::ResidencyState::resident);
    REQUIRE(device.destroyBuffer(*buffer).has_value());
}

TEST_CASE("accepted make-resident publication failure quarantines physical ownership",
          "[size:medium][rhi][residency][memory][failure][warp]")
{
    const auto initialQuarantineCount =
        rachis::d3d12::test::activeMakeResidentQuarantines().load(std::memory_order_relaxed);
    const auto initialPrematureReleaseCount =
        rachis::d3d12::test::prematureMakeResidentQuarantineReleases().load(std::memory_order_relaxed);

    {
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .adapterPreference = rachis::AdapterPreference::warp,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
        });
        REQUIRE(created.has_value());
        auto device = std::move(*created);
        REQUIRE(rachis::test::requireCapabilities(device).manualResidency);

        struct FailureCase final
        {
            rachis::d3d12::test::MakeResidentFailurePoint point;
            std::string_view label;
        };
        constexpr std::array failures{
            FailureCase{
                rachis::d3d12::test::MakeResidentFailurePoint::afterNativeAcceptance,
                "Make-resident failure after native acceptance",
            },
            FailureCase{
                rachis::d3d12::test::MakeResidentFailurePoint::afterQueueWait,
                "Make-resident failure after queue wait",
            },
            FailureCase{
                rachis::d3d12::test::MakeResidentFailurePoint::afterQueueSignal,
                "Make-resident failure after queue signal",
            },
        };
        std::array<std::uint64_t, failures.size()> allocationIds{};

        for (std::size_t index = 0; index < failures.size(); ++index)
        {
            const auto& failureCase = failures[index];
            auto buffer = device.createBuffer(rachis::BufferDesc{
                .size = 64ULL * 1024,
                .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
                .allocationFlags = rachis::ResourceAllocationFlag::manualResidency,
                .label = failureCase.label,
            });
            REQUIRE(buffer.has_value());
            const auto before = device.captureResourceMemorySnapshot();
            REQUIRE(before.has_value());
            const auto* beforeEntry = findByLabel(*before, failureCase.label);
            REQUIRE(beforeEntry != nullptr);
            allocationIds[index] = beforeEntry->allocationId;

            const std::array buffers{*buffer};
            const rachis::ResidencySet resources{.buffers = buffers};
            REQUIRE(device.evict(resources).has_value());

            rachis::Result<rachis::TimelinePoint> scheduled;
            {
                const MakeResidentFailureScope failure(failureCase.point);
                scheduled = device.enqueueMakeResident(rachis::QueueType::copy, resources);
            }
            REQUIRE_FALSE(scheduled.has_value());
            CHECK(scheduled.error().code == rachis::ErrorCode::nativeFailure);

            const auto state = device.queryResidency(*buffer);
            REQUIRE_FALSE(state.has_value());
            CHECK(state.error().code == rachis::ErrorCode::invalidState);

            if (index == 0)
            {
                auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
                REQUIRE(command.has_value());
                auto useFaulted = device.bufferBarrier(*command,
                                                       rachis::BufferBarrier{
                                                           .buffer = *buffer,
                                                           .stagesBefore = rachis::PipelineStage::all,
                                                           .stagesAfter = rachis::PipelineStage::copy,
                                                           .accessBefore = rachis::Access::common,
                                                           .accessAfter = rachis::Access::copyDestination,
                                                       });
                REQUIRE_FALSE(useFaulted.has_value());
                CHECK(useFaulted.error().code == rachis::ErrorCode::invalidState);
            }

            auto retry = device.enqueueMakeResident(rachis::QueueType::copy, resources);
            REQUIRE_FALSE(retry.has_value());
            CHECK(retry.error().code == rachis::ErrorCode::invalidState);

            REQUIRE(device.destroyBuffer(*buffer).has_value());
            device.collectGarbage();
            const auto retained = device.captureResourceMemorySnapshot();
            REQUIRE(retained.has_value());
            const auto* retainedEntry = findByLabel(*retained, failureCase.label);
            REQUIRE(retainedEntry != nullptr);
            CHECK(retainedEntry->allocationId == allocationIds[index]);
            CHECK_FALSE(retainedEntry->logicalAlive);
        }

        CHECK(rachis::d3d12::test::activeMakeResidentQuarantines().load(std::memory_order_relaxed) ==
              initialQuarantineCount + static_cast<std::int64_t>(failures.size()));

        REQUIRE(device.waitIdle(rachis::QueueType::copy).has_value());

        auto recovery = device.createBuffer(rachis::BufferDesc{
            .size = 64ULL * 1024,
            .usage = rachis::BufferUsage::copyDestination,
            .allocationFlags = rachis::ResourceAllocationFlag::manualResidency,
            .label = "Healthy residency after publication failure",
        });
        REQUIRE(recovery.has_value());
        const std::array recoveryBuffers{*recovery};
        const rachis::ResidencySet recoveryResources{.buffers = recoveryBuffers};
        REQUIRE(device.evict(recoveryResources).has_value());
        const auto recoveryReady = device.enqueueMakeResident(rachis::QueueType::copy, recoveryResources);
        REQUIRE(recoveryReady.has_value());
        REQUIRE(device.wait(*recoveryReady).has_value());
        device.collectGarbage();
        REQUIRE(device.queryResidency(*recovery) == rachis::ResidencyState::resident);
        REQUIRE(device.destroyBuffer(*recovery).has_value());
    }

    CHECK(rachis::d3d12::test::activeMakeResidentQuarantines().load(std::memory_order_relaxed) ==
          initialQuarantineCount);
    CHECK(rachis::d3d12::test::prematureMakeResidentQuarantineReleases().load(std::memory_order_relaxed) ==
          initialPrematureReleaseCount);
}

TEST_CASE("residency lease batches retain each manually resident record once", "[size:small][rhi][residency]")
{
    std::vector<std::shared_ptr<rachis::d3d12::ResidencyRecord>> records;
    for (std::size_t index = 0; index < 20; ++index)
    {
        auto record = std::make_shared<rachis::d3d12::ResidencyRecord>();
        record->state = rachis::ResidencyState::resident;
        records.push_back(std::move(record));
    }
    const auto useCount = [](const std::shared_ptr<rachis::d3d12::ResidencyRecord>& record) {
        const std::scoped_lock lock(record->mutex);
        return record->useCount;
    };

    {
        std::vector<rachis::d3d12::ResidencyLease> leases;
        const std::array<std::shared_ptr<rachis::d3d12::ResidencyRecord>, 6> firstBatch{
            records[3], nullptr, records[0], records[3], records[1], records[2]};
        REQUIRE(rachis::d3d12::appendResidencyLeases(leases, firstBatch).has_value());
        CHECK(leases.size() == 4U);

        std::vector<std::shared_ptr<rachis::d3d12::ResidencyRecord>> secondBatch(records.rbegin(), records.rend());
        secondBatch.push_back(nullptr);
        secondBatch.push_back(records[12]);
        REQUIRE(rachis::d3d12::appendResidencyLeases(leases, secondBatch).has_value());
        CHECK(leases.size() == records.size());
        for (const auto& record : records)
        {
            CHECK(useCount(record) == 1U);
            CHECK(std::ranges::count(leases, record.get(), &rachis::d3d12::ResidencyLease::record) == 1);
        }

        auto evicted = std::make_shared<rachis::d3d12::ResidencyRecord>();
        evicted->state = rachis::ResidencyState::evicted;
        auto pending = std::make_shared<rachis::d3d12::ResidencyRecord>();
        pending->state = rachis::ResidencyState::pending;
        auto rejectedBatch = secondBatch;
        rejectedBatch.push_back(pending);
        rejectedBatch.push_back(evicted);
        const auto rejected = rachis::d3d12::appendResidencyLeases(leases, rejectedBatch);
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
        CHECK(leases.size() == records.size());
        CHECK(useCount(pending) == 0U);
        CHECK(useCount(evicted) == 0U);
    }
    for (const auto& record : records)
    {
        CHECK(useCount(record) == 0U);
    }
}
