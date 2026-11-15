#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <Windows.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace
{

    [[nodiscard]] rachis::Result<rachis::Device> createWarpDevice()
    {
        return rachis::createDevice(rachis::DeviceDesc{
            .adapterPreference = rachis::AdapterPreference::warp,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = true,
        });
    }

    [[nodiscard]] rachis::TimelinePoint submitEmpty(rachis::Device& device)
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        return *submitted;
    }

} // namespace

TEST_CASE("timeline observations are point scoped bounded and device local",
          "[size:medium][rhi][queue][timeline][warp]")
{
    using namespace std::chrono_literals;

    rachis::Device invalidDevice;
    auto invalidQuery = invalidDevice.queryTimelinePoint({});
    auto invalidWait = invalidDevice.wait({}, 0ns);
    REQUIRE_FALSE(invalidQuery.has_value());
    CHECK(invalidQuery.error().code == rachis::ErrorCode::invalidState);
    REQUIRE_FALSE(invalidWait.has_value());
    CHECK(invalidWait.error().code == rachis::ErrorCode::invalidState);

    auto producerCreated = createWarpDevice();
    auto consumerCreated = createWarpDevice();
    REQUIRE(producerCreated.has_value());
    REQUIRE(consumerCreated.has_value());
    auto producer = std::move(*producerCreated);
    auto consumer = std::move(*consumerCreated);

    auto invalidPointQuery = consumer.queryTimelinePoint({});
    REQUIRE_FALSE(invalidPointQuery.has_value());
    CHECK(invalidPointQuery.error().code == rachis::ErrorCode::invalidArgument);

    const auto producerFirst = submitEmpty(producer);
    auto shared = producer.exportTimelinePoint(producerFirst);
    REQUIRE(shared.has_value());

    auto blocked = consumer.enqueueSharedTimelineWait(rachis::QueueType::graphics,
                                                      rachis::SharedTimelinePointView{
                                                          .handle = shared->handle.view(),
                                                          .value = producerFirst.value() + 1,
                                                      });
    REQUIRE(blocked.has_value());

    const auto firstQuery = consumer.queryTimelinePoint(*blocked);
    const auto secondQuery = consumer.queryTimelinePoint(*blocked);
    REQUIRE(firstQuery.has_value());
    REQUIRE(secondQuery.has_value());
    CHECK(*firstQuery == rachis::TimelinePointStatus::pending);
    CHECK(*secondQuery == rachis::TimelinePointStatus::pending);

    auto foreignQuery = producer.queryTimelinePoint(*blocked);
    auto foreignWait = producer.wait(*blocked, 0ns);
    REQUIRE_FALSE(foreignQuery.has_value());
    CHECK(foreignQuery.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE_FALSE(foreignWait.has_value());
    CHECK(foreignWait.error().code == rachis::ErrorCode::invalidArgument);

    auto negativeWait = consumer.wait(*blocked, std::chrono::nanoseconds{-1});
    REQUIRE_FALSE(negativeWait.has_value());
    CHECK(negativeWait.error().code == rachis::ErrorCode::invalidArgument);

    auto zeroWait = consumer.wait(*blocked, 0ns);
    auto timedWait = consumer.wait(*blocked, 1ms);
    REQUIRE(zeroWait.has_value());
    REQUIRE(timedWait.has_value());
    CHECK(*zeroWait == rachis::TimelinePointStatus::pending);
    CHECK(*timedWait == rachis::TimelinePointStatus::pending);

    auto blockedLater = consumer.enqueueSharedTimelineWait(rachis::QueueType::graphics,
                                                           rachis::SharedTimelinePointView{
                                                               .handle = shared->handle.view(),
                                                               .value = producerFirst.value() + 2,
                                                           });
    REQUIRE(blockedLater.has_value());
    auto laterWait = std::async(std::launch::async, [&consumer, point = *blockedLater] {
        return consumer.wait(point, std::chrono::seconds{5});
    });
    CHECK(laterWait.wait_for(25ms) == std::future_status::timeout);

    const auto producerSecond = submitEmpty(producer);
    REQUIRE(producerSecond.value() == producerFirst.value() + 1);

    auto earlierStatus = consumer.queryTimelinePoint(*blocked);
    const auto earlierDeadline = std::chrono::steady_clock::now() + 1s;
    while (earlierStatus.has_value() && *earlierStatus == rachis::TimelinePointStatus::pending &&
           std::chrono::steady_clock::now() < earlierDeadline)
    {
        std::this_thread::yield();
        earlierStatus = consumer.queryTimelinePoint(*blocked);
    }
    CHECK(earlierStatus.has_value());
    if (earlierStatus.has_value())
    {
        CHECK(*earlierStatus == rachis::TimelinePointStatus::complete);
    }
    CHECK(laterWait.wait_for(25ms) == std::future_status::timeout);

    const auto producerThird = submitEmpty(producer);
    REQUIRE(producerThird.value() == producerFirst.value() + 2);
    REQUIRE(laterWait.wait_for(5s) == std::future_status::ready);
    auto completedWait = laterWait.get();
    REQUIRE(completedWait.has_value());
    CHECK(*completedWait == rachis::TimelinePointStatus::complete);
    auto completedQuery = consumer.queryTimelinePoint(*blocked);
    REQUIRE(completedQuery.has_value());
    CHECK(*completedQuery == rachis::TimelinePointStatus::complete);
    auto laterQuery = consumer.queryTimelinePoint(*blockedLater);
    REQUIRE(laterQuery.has_value());
    CHECK(*laterQuery == rachis::TimelinePointStatus::complete);
    REQUIRE(consumer.wait(*blocked).has_value());
    REQUIRE(consumer.wait(*blockedLater).has_value());
    REQUIRE(producer.wait(producerThird).has_value());

    const auto producerDiagnostics = producer.diagnosticCounts();
    const auto consumerDiagnostics = consumer.diagnosticCounts();
    CHECK(producerDiagnostics.errors == 0);
    CHECK(producerDiagnostics.corruptions == 0);
    CHECK(consumerDiagnostics.errors == 0);
    CHECK(consumerDiagnostics.corruptions == 0);
}

TEST_CASE("bounded waits on a blocked timeline point reuse one completion event per value",
          "[size:medium][rhi][queue][timeline][warp]")
{
    using namespace std::chrono_literals;

    auto producerCreated = createWarpDevice();
    auto consumerCreated = createWarpDevice();
    REQUIRE(producerCreated.has_value());
    REQUIRE(consumerCreated.has_value());
    auto producer = std::move(*producerCreated);
    auto consumer = std::move(*consumerCreated);

    const auto producerFirst = submitEmpty(producer);
    auto shared = producer.exportTimelinePoint(producerFirst);
    REQUIRE(shared.has_value());
    auto blocked = consumer.enqueueSharedTimelineWait(rachis::QueueType::graphics,
                                                      rachis::SharedTimelinePointView{
                                                          .handle = shared->handle.view(),
                                                          .value = producerFirst.value() + 1,
                                                      });
    REQUIRE(blocked.has_value());

    auto firstPoll = consumer.wait(*blocked, 1us);
    REQUIRE(firstPoll.has_value());
    REQUIRE(*firstPoll == rachis::TimelinePointStatus::pending);

    DWORD handlesBefore = 0;
    REQUIRE(GetProcessHandleCount(GetCurrentProcess(), &handlesBefore) != FALSE);
    constexpr int pollCount = 2000;
    int pendingPolls = 0;
    for (int poll = 0; poll < pollCount; ++poll)
    {
        auto status = consumer.wait(*blocked, 1us);
        if (status.has_value() && *status == rachis::TimelinePointStatus::pending)
        {
            ++pendingPolls;
        }
    }
    CHECK(pendingPolls == pollCount);
    DWORD handlesAfter = 0;
    REQUIRE(GetProcessHandleCount(GetCurrentProcess(), &handlesAfter) != FALSE);
    CHECK(handlesAfter <= handlesBefore + 32);

    const auto timedStart = std::chrono::steady_clock::now();
    auto timedWait = consumer.wait(*blocked, 20ms);
    const auto timedElapsed = std::chrono::steady_clock::now() - timedStart;
    REQUIRE(timedWait.has_value());
    CHECK(*timedWait == rachis::TimelinePointStatus::pending);
    CHECK(timedElapsed >= 20ms);

    constexpr std::size_t waiterCount = 4;
    std::vector<std::future<rachis::Result<rachis::TimelinePointStatus>>> waiters;
    waiters.reserve(waiterCount);
    for (std::size_t index = 0; index < waiterCount; ++index)
    {
        waiters.push_back(std::async(std::launch::async, [&consumer, point = *blocked] {
            return consumer.wait(point, std::chrono::seconds{5});
        }));
    }
    for (auto& waiter : waiters)
    {
        CHECK(waiter.wait_for(25ms) == std::future_status::timeout);
    }

    const auto producerSecond = submitEmpty(producer);
    REQUIRE(producerSecond.value() == producerFirst.value() + 1);
    for (auto& waiter : waiters)
    {
        REQUIRE(waiter.wait_for(5s) == std::future_status::ready);
        auto completed = waiter.get();
        REQUIRE(completed.has_value());
        CHECK(*completed == rachis::TimelinePointStatus::complete);
    }
    auto completedQuery = consumer.queryTimelinePoint(*blocked);
    REQUIRE(completedQuery.has_value());
    CHECK(*completedQuery == rachis::TimelinePointStatus::complete);
    REQUIRE(consumer.wait(*blocked).has_value());
    REQUIRE(producer.wait(producerSecond).has_value());

    DWORD handlesSettled = 0;
    REQUIRE(GetProcessHandleCount(GetCurrentProcess(), &handlesSettled) != FALSE);
    CHECK(handlesSettled <= handlesBefore + 32);

    const auto producerDiagnostics = producer.diagnosticCounts();
    const auto consumerDiagnostics = consumer.diagnosticCounts();
    CHECK(producerDiagnostics.errors == 0);
    CHECK(producerDiagnostics.corruptions == 0);
    CHECK(consumerDiagnostics.errors == 0);
    CHECK(consumerDiagnostics.corruptions == 0);
}

TEST_CASE("a shared timeline wait on an imported fence at UINT64_MAX keeps the consumer device active",
          "[size:medium][rhi][queue][timeline][device-loss][warp]")
{
    auto consumerCreated = createWarpDevice();
    REQUIRE(consumerCreated.has_value());
    auto consumer = std::move(*consumerCreated);

    ComPtr<IDXGIFactory4> factory;
    REQUIRE(SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))));
    ComPtr<IDXGIAdapter> adapter;
    REQUIRE(SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))));
    ComPtr<ID3D12Device> producer;
    REQUIRE(SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&producer))));
    ComPtr<ID3D12Fence> fence;
    REQUIRE(SUCCEEDED(producer->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence))));
    REQUIRE(SUCCEEDED(fence->Signal(std::numeric_limits<std::uint64_t>::max())));
    REQUIRE(fence->GetCompletedValue() == std::numeric_limits<std::uint64_t>::max());
    HANDLE nativeHandle = nullptr;
    REQUIRE(SUCCEEDED(producer->CreateSharedHandle(fence.Get(), nullptr, GENERIC_ALL, nullptr, &nativeHandle)));
    const auto shared = rachis::SharedHandle::adopt(rachis::SharedHandleView{
        .type = rachis::SharedHandleType::windowsNt,
        .value = reinterpret_cast<std::uintptr_t>(nativeHandle),
    });
    REQUIRE(shared.isValid());

    const auto completedBefore = consumer.completedValue(rachis::QueueType::graphics);
    auto waited = consumer.enqueueSharedTimelineWait(rachis::QueueType::graphics,
                                                     rachis::SharedTimelinePointView{
                                                         .handle = shared.view(),
                                                         .value = 1,
                                                     });
    REQUIRE_FALSE(waited.has_value());
    INFO(waited.error().message);
    CHECK(waited.error().code != rachis::ErrorCode::deviceLost);
    CHECK(waited.error().code == rachis::ErrorCode::invalidState);
    CHECK(consumer.operationalState() == rachis::DeviceOperationalState::active);
    CHECK_FALSE(consumer.deviceLossError().has_value());
    CHECK(consumer.completedValue(rachis::QueueType::graphics) == completedBefore);

    auto buffer = consumer.createBuffer(rachis::BufferDesc{
        .size = 256,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Shared-wait consumer buffer",
    });
    REQUIRE(buffer.has_value());
    const auto submitted = submitEmpty(consumer);
    CHECK(submitted.value() == completedBefore + 1);
    REQUIRE(consumer.wait(submitted).has_value());
    auto completed = consumer.queryTimelinePoint(submitted);
    REQUIRE(completed.has_value());
    CHECK(*completed == rachis::TimelinePointStatus::complete);
    REQUIRE(consumer.destroyBuffer(*buffer).has_value());
    CHECK(consumer.operationalState() == rachis::DeviceOperationalState::active);

    const auto diagnostics = consumer.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
