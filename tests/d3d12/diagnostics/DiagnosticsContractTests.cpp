#include "D3D12/D3D12Diagnostics.h"
#include "D3D12/D3D12NativeHelpers.h"
#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <locale>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace
{

    struct FaultCapture final
    {
        rachis::DiagnosticSeverity severity = rachis::DiagnosticSeverity::information;
        std::string description;
    };

    void captureFault(const rachis::DiagnosticMessage& message, void* userData) noexcept
    {
        auto& capture = *static_cast<FaultCapture*>(userData);
        capture.severity = message.severity;
        try
        {
            capture.description.assign(message.description);
        }
        catch (...)
        {
            capture.description.clear();
        }
    }

    struct DiagnosticLog final
    {
        std::mutex mutex;
        std::vector<rachis::DiagnosticSeverity> severities;
        std::string messages;

        [[nodiscard]] bool contains(rachis::DiagnosticSeverity severity, std::string_view text)
        {
            const std::scoped_lock lock(mutex);
            bool seen = false;
            for (const auto recorded : severities)
            {
                seen = seen || recorded == severity;
            }
            return seen && messages.contains(text);
        }
    };

    void logDiagnostic(const rachis::DiagnosticMessage& message, void* userData) noexcept
    {
        auto& log = *static_cast<DiagnosticLog*>(userData);
        try
        {
            const std::scoped_lock lock(log.mutex);
            log.severities.push_back(message.severity);
            log.messages.append(message.description);
            log.messages.push_back('\n');
        }
        catch (...)
        {
        }
    }

    [[nodiscard]] rachis::DeviceDesc warpDeviceDesc(rachis::ValidationLevel validation,
                                                    bool requireValidation,
                                                    DiagnosticLog* log)
    {
        return rachis::DeviceDesc{
            .adapterPreference = rachis::AdapterPreference::warp,
            .validation = validation,
            .requireValidation = requireValidation,
            .enableDred = true,
            .diagnosticCallback = log != nullptr ? &logDiagnostic : nullptr,
            .diagnosticUserData = log,
        };
    }

    [[nodiscard]] bool d3d12DebugLayerInstalled()
    {
        ComPtr<ID3D12Debug> debug;
        return SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));
    }

    void requireCopyRoundTrip(rachis::Device& device, std::uint8_t seed)
    {
        constexpr std::size_t byteCount = 256;
        auto upload = device.createBuffer(rachis::BufferDesc{
            .size = byteCount,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Validation-mix upload",
        });
        if (!upload)
        {
            UNSCOPED_INFO(upload.error().message);
        }
        REQUIRE(upload.has_value());
        auto readback = device.createBuffer(rachis::BufferDesc{
            .size = byteCount,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Validation-mix readback",
        });
        REQUIRE(readback.has_value());

        std::array<std::byte, byteCount> written{};
        for (std::size_t index = 0; index < written.size(); ++index)
        {
            written[index] = static_cast<std::byte>((index * 13U + seed) & 0xFFU);
        }
        auto write = device.writeBuffer(*upload, 0, written);
        if (!write)
        {
            UNSCOPED_INFO(write.error().message);
        }
        REQUIRE(write.has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .copyBuffer(*command,
                                rachis::BufferCopy{
                                    .source = *upload,
                                    .destination = *readback,
                                    .size = byteCount,
                                })
                    .has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());

        std::array<std::byte, byteCount> read{};
        REQUIRE(device.readBuffer(*readback, 0, read).has_value());
        CHECK(read == written);
        REQUIRE(device.destroyBuffer(*readback).has_value());
        REQUIRE(device.destroyBuffer(*upload).has_value());
        CHECK(device.operationalState() == rachis::DeviceOperationalState::active);
    }

    void removeWarpDevice()
    {
        ComPtr<IDXGIFactory4> factory;
        REQUIRE(SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))));
        ComPtr<IDXGIAdapter> adapter;
        REQUIRE(SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))));
        ComPtr<ID3D12Device5> removable;
        REQUIRE(SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&removable))));
        removable->RemoveDevice();
    }

    void checkPublishedLoss(rachis::Device& device, DiagnosticLog& log)
    {
        CHECK(device.operationalState() == rachis::DeviceOperationalState::lost);
        const auto loss = device.deviceLossError();
        REQUIRE(loss.has_value());
        CHECK(loss->code == rachis::ErrorCode::deviceLost);
        CHECK(loss->nativeCode != 0);
        CHECK(log.contains(rachis::DiagnosticSeverity::error, "D3D12 device fault during"));

        auto report = device.captureDeviceFaultReport();
        REQUIRE(report.has_value());
        CHECK(report->status != rachis::DeviceFaultStatus::healthy);
    }

    [[nodiscard]] rachis::TextureDesc removalTextureDesc()
    {
        return rachis::TextureDesc{
            .width = 16,
            .height = 16,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::copyDestination,
            .label = "Removed-device texture",
        };
    }

    [[nodiscard]] rachis::BufferDesc removalBufferDesc()
    {
        return rachis::BufferDesc{
            .size = 256,
            .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Removed-device buffer",
        };
    }

    struct NativeDeviceCreationLog final
    {
        std::atomic_uint64_t observed = 0;
        std::atomic_uint64_t unserialized = 0;
    };

    [[nodiscard]] NativeDeviceCreationLog& nativeDeviceCreationLog() noexcept
    {
        static NativeDeviceCreationLog s_log;
        return s_log;
    }

    void observeNativeDeviceCreation() noexcept
    {
        auto& log = nativeDeviceCreationLog();
        log.observed.fetch_add(1, std::memory_order_relaxed);
        bool lockAvailable = true;
        try
        {
            std::jthread probe(
                [&lockAvailable] { lockAvailable = rachis::d3d12::test::debugLayerProcessLockAvailable(); });
        }
        catch (...)
        {
        }
        if (lockAvailable)
        {
            log.unserialized.fetch_add(1, std::memory_order_relaxed);
        }
    }

    class NativeDeviceCreationObserverScope final
    {
    public:
        explicit NativeDeviceCreationObserverScope(rachis::d3d12::test::NativeDeviceCreationObserver observer) noexcept
            : previous_(rachis::d3d12::test::nativeDeviceCreationObserver().exchange(observer))
        {
        }

        ~NativeDeviceCreationObserverScope()
        {
            rachis::d3d12::test::nativeDeviceCreationObserver().store(previous_);
        }

        NativeDeviceCreationObserverScope(const NativeDeviceCreationObserverScope&) = delete;
        NativeDeviceCreationObserverScope& operator=(const NativeDeviceCreationObserverScope&) = delete;
        NativeDeviceCreationObserverScope(NativeDeviceCreationObserverScope&&) = delete;
        NativeDeviceCreationObserverScope& operator=(NativeDeviceCreationObserverScope&&) = delete;

    private:
        rachis::d3d12::test::NativeDeviceCreationObserver previous_ = nullptr;
    };

} // namespace

TEST_CASE("device-fault reports are owning and healthy captures are empty", "[size:medium][rhi][diagnostics][warp]")
{
    auto device = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
    });
    REQUIRE(device);

    auto report = device->captureDeviceFaultReport();
    REQUIRE(report);
    CHECK(report->status == rachis::DeviceFaultStatus::healthy);
    CHECK(report->nativeReason == 0);
    CHECK(report->dredAvailable);
    CHECK(report->pageFaultAddress == 0);
    CHECK(report->breadcrumbs.empty());
    CHECK(report->allocations.empty());
    CHECK_FALSE(report->breadcrumbsTruncated);
    CHECK_FALSE(report->allocationsTruncated);

    auto withoutDred = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = false,
    });
    REQUIRE(withoutDred);
    auto withoutDredReport = withoutDred->captureDeviceFaultReport();
    REQUIRE(withoutDredReport);
    CHECK(withoutDredReport->status == rachis::DeviceFaultStatus::healthy);

    const rachis::Device invalid;
    auto invalidReport = invalid.captureDeviceFaultReport();
    REQUIRE_FALSE(invalidReport);
    CHECK(invalidReport.error().code == rachis::ErrorCode::invalidState);
}

TEST_CASE("backend descriptor accounting separates live leases from reusable capacity",
          "[size:medium][rhi][diagnostics][descriptor][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
    });
    REQUIRE(created);
    auto device = std::move(*created);

    const auto baseline = device.captureBackendAllocationSnapshot();
    CHECK(baseline.reservedDescriptorCount >= baseline.liveDescriptorCount);

    auto first = device.createSampler(rachis::SamplerDesc{.label = "Descriptor accounting first sampler"});
    REQUIRE(first);
    const auto firstLive = device.captureBackendAllocationSnapshot();
    CHECK(firstLive.liveDescriptorCount == baseline.liveDescriptorCount + 1);
    CHECK(firstLive.reservedDescriptorCount >= firstLive.liveDescriptorCount);
    CHECK(firstLive.reservedDescriptorCount > baseline.reservedDescriptorCount);

    REQUIRE(device.destroySampler(*first));
    const auto released = device.captureBackendAllocationSnapshot();
    CHECK(released.liveDescriptorCount == baseline.liveDescriptorCount);
    CHECK(released.reservedDescriptorCount == firstLive.reservedDescriptorCount);

    auto reused = device.createSampler(rachis::SamplerDesc{.label = "Descriptor accounting reused sampler"});
    REQUIRE(reused);
    const auto reusedLive = device.captureBackendAllocationSnapshot();
    CHECK(reusedLive.liveDescriptorCount == firstLive.liveDescriptorCount);
    CHECK(reusedLive.reservedDescriptorCount == firstLive.reservedDescriptorCount);
    REQUIRE(device.destroySampler(*reused));

    const rachis::Device invalid;
    const auto invalidSnapshot = invalid.captureBackendAllocationSnapshot();
    CHECK(invalidSnapshot.liveDescriptorCount == 0);
    CHECK(invalidSnapshot.reservedDescriptorCount == 0);
}

TEST_CASE("D3D12 synthetic device removal produces a bounded report", "[size:medium][rhi][diagnostics][dred][warp]")
{
    ComPtr<IDXGIFactory4> factory;
    REQUIRE(SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))));
    ComPtr<IDXGIAdapter> adapter;
    REQUIRE(SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))));
    ComPtr<ID3D12Device> device;
    REQUIRE(SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))));
    ComPtr<ID3D12Device5> device5;
    REQUIRE(SUCCEEDED(device.As(&device5)));

    device5->RemoveDevice();
    auto report = rachis::d3d12::captureDeviceFaultReport(*device.Get());
    REQUIRE(report);
    CHECK(report->status == rachis::DeviceFaultStatus::removed);
    CHECK(report->nativeReason != 0);
    for (const auto& breadcrumb : report->breadcrumbs)
    {
        CHECK(breadcrumb.completedOperationCount <= breadcrumb.totalOperationCount);
        for (const auto& operation : breadcrumb.operations)
        {
            CHECK(operation.index < breadcrumb.totalOperationCount);
            CHECK(operation.completed == (operation.index < breadcrumb.completedOperationCount));
        }
    }
}

TEST_CASE("device-fault report reaches the application diagnostic callback", "[size:small][rhi][diagnostics][dred]")
{
    FaultCapture capture;

    const rachis::DeviceFaultReport report{
        .status = rachis::DeviceFaultStatus::hung,
        .nativeReason = static_cast<std::int64_t>(DXGI_ERROR_DEVICE_HUNG),
        .dredAvailable = true,
        .pageFaultAddress = 0x1234,
        .breadcrumbs = {rachis::DeviceFaultBreadcrumb{
            .queueLabel = "graphics",
            .commandListLabel = "frame",
            .completedOperationCount = 1,
            .totalOperationCount = 2,
            .firstCapturedOperation = 1,
            .operations = {rachis::DeviceFaultOperation{
                .index = 1,
                .nativeOperation = D3D12_AUTO_BREADCRUMB_OP_DISPATCH,
                .completed = false,
                .name = "dispatch",
                .context = "lighting",
            }},
            .operationsTruncated = true,
            .contextsTruncated = true,
        }},
        .allocations = {rachis::DeviceFaultAllocation{
            .status = rachis::DeviceFaultAllocationStatus::recentlyFreed,
            .nativeType = D3D12_DRED_ALLOCATION_TYPE_RESOURCE,
            .typeName = "resource",
            .label = "history buffer",
        }},
    };
    rachis::d3d12::emitDeviceFaultReport(report, "test submission", &captureFault, &capture);

    const std::string& messages = capture.description;
    CHECK(capture.severity == rachis::DiagnosticSeverity::error);
    CHECK(messages.contains("D3D12 device fault during test submission"));
    CHECK(messages.contains("status=hung"));
    CHECK(messages.contains("queue='graphics' list='frame' completed=1/2"));
    CHECK(messages.contains("(operations truncated) (contexts truncated)"));
    CHECK(messages.contains("op[1] pending dispatch context='lighting'"));
    CHECK(messages.contains("state=recentlyFreed type=resource label='history buffer'"));
}

TEST_CASE("D3D12 native errors spell the HRESULT without the global locale's digit grouping",
          "[size:small][rhi][diagnostics]")
{
    struct GroupedDigits final : std::numpunct<char>
    {
    protected:
        [[nodiscard]] char do_thousands_sep() const override
        {
            return ',';
        }

        [[nodiscard]] std::string do_grouping() const override
        {
            return "\3";
        }
    };

    const auto deviceRemoved = static_cast<HRESULT>(DXGI_ERROR_DEVICE_REMOVED);
    const auto previous = std::locale::global(std::locale(std::locale::classic(), new GroupedDigits));
    const auto error = rachis::d3d12::nativeError(
        rachis::ErrorCode::deviceLost, deviceRemoved, "ID3D12Device::CreateCommittedResource");
    const auto label = rachis::d3d12::wideLabel("\xE3\x83\xA9\xE3\x83\x99\xE3\x83\xAB 12345");
    std::locale::global(previous);

    CHECK(error.code == rachis::ErrorCode::deviceLost);
    CHECK(error.nativeCode == static_cast<std::int64_t>(deviceRemoved));
    CHECK(error.message == "ID3D12Device::CreateCommittedResource failed with HRESULT 0x887A0005.");
    CHECK(label == L"\u30E9\u30D9\u30EB 12345");
    CHECK(rachis::d3d12::wideLabel("\xFF\xFE").empty());
    CHECK(rachis::d3d12::wideLabel({}).empty());
    CHECK(rachis::d3d12::mipDimension(13, 2) == 3);
    CHECK(rachis::d3d12::mipDimension(13, 4) == 1);
}

TEST_CASE("a validated D3D12 device never removes live validation-disabled devices",
          "[size:medium][rhi][diagnostics][validation][warp]")
{
    const auto initial = rachis::d3d12::test::debugLayerProcessSnapshot();
    const bool debugLayerInstalled = d3d12DebugLayerInstalled();
    const std::uint64_t ownUnvalidatedDevice = initial.debugLayerEnabled ? 0U : 1U;

    {
        auto unvalidated = rachis::createDevice(warpDeviceDesc(rachis::ValidationLevel::disabled, false, nullptr));
        REQUIRE(unvalidated.has_value());
        const auto afterUnvalidated = rachis::d3d12::test::debugLayerProcessSnapshot();
        CHECK(afterUnvalidated.debugLayerEnabled == initial.debugLayerEnabled);
        CHECK(afterUnvalidated.devicesWithoutDebugLayer == initial.devicesWithoutDebugLayer + ownUnvalidatedDevice);

        DiagnosticLog strictLog;
        auto strict = rachis::createDevice(warpDeviceDesc(rachis::ValidationLevel::standard, true, &strictLog));
        if (initial.debugLayerEnabled)
        {
            REQUIRE(strict.has_value());
        }
        else
        {
            REQUIRE_FALSE(strict.has_value());
            CHECK(strict.error().code == rachis::ErrorCode::validationError);
            if (debugLayerInstalled)
            {
                CHECK(strict.error().message.contains("created without the debug layer"));
            }
        }

        DiagnosticLog lenientLog;
        auto lenient = rachis::createDevice(warpDeviceDesc(rachis::ValidationLevel::standard, false, &lenientLog));
        REQUIRE(lenient.has_value());
        const auto afterLenient = rachis::d3d12::test::debugLayerProcessSnapshot();
        CHECK(afterLenient.debugLayerEnabled == initial.debugLayerEnabled);
        if (!initial.debugLayerEnabled && debugLayerInstalled)
        {
            CHECK(lenient->diagnosticCounts().warnings == 1);
            CHECK(lenientLog.contains(rachis::DiagnosticSeverity::warning, "not enabled for this device"));
            CHECK(afterLenient.devicesWithoutDebugLayer == initial.devicesWithoutDebugLayer + 2);
        }

        requireCopyRoundTrip(*unvalidated, 1);
        requireCopyRoundTrip(*lenient, 2);
        if (strict)
        {
            requireCopyRoundTrip(*strict, 3);
        }
    }

    CHECK(rachis::d3d12::test::debugLayerProcessSnapshot().devicesWithoutDebugLayer ==
          initial.devicesWithoutDebugLayer);

    DiagnosticLog validatedLog;
    auto validated = rachis::createDevice(warpDeviceDesc(rachis::ValidationLevel::standard, false, &validatedLog));
    REQUIRE(validated.has_value());
    CHECK(validated->diagnosticCounts().warnings == 0);
    const auto afterValidated = rachis::d3d12::test::debugLayerProcessSnapshot();
    CHECK(afterValidated.debugLayerEnabled == (initial.debugLayerEnabled || debugLayerInstalled));
    if (!afterValidated.debugLayerEnabled)
    {
        SKIP("The D3D12 debug layer is not installed, so validation-on ordering cannot be exercised.");
    }
    CHECK(afterValidated.devicesWithoutDebugLayer == initial.devicesWithoutDebugLayer);

    auto laterUnvalidated = rachis::createDevice(warpDeviceDesc(rachis::ValidationLevel::disabled, false, nullptr));
    REQUIRE(laterUnvalidated.has_value());
    CHECK(rachis::d3d12::test::debugLayerProcessSnapshot().devicesWithoutDebugLayer ==
          initial.devicesWithoutDebugLayer);

    DiagnosticLog requiredLog;
    auto required = rachis::createDevice(warpDeviceDesc(rachis::ValidationLevel::standard, true, &requiredLog));
    if (!required)
    {
        UNSCOPED_INFO(required.error().message);
    }
    REQUIRE(required.has_value());

    requireCopyRoundTrip(*laterUnvalidated, 4);
    requireCopyRoundTrip(*validated, 5);
    requireCopyRoundTrip(*required, 6);
    CHECK(validated->diagnosticCounts().errors == 0);
    CHECK(validated->diagnosticCounts().corruptions == 0);
    CHECK(required->diagnosticCounts().errors == 0);
    CHECK(required->diagnosticCounts().corruptions == 0);
}

TEST_CASE("a GPU-validated D3D12 device never removes live devices created without GPU-based validation",
          "[size:medium][rhi][diagnostics][validation][warp]")
{
    const auto initial = rachis::d3d12::test::debugLayerProcessSnapshot();
    if (!d3d12DebugLayerInstalled())
    {
        SKIP("The D3D12 debug layer is not installed, so GPU-based validation ordering cannot be exercised.");
    }
    if (initial.gpuValidationEnabled)
    {
        SKIP("GPU-based validation is already on in this process, so its blocking order cannot be exercised.");
    }
    if (!initial.debugLayerEnabled && initial.devicesWithoutDebugLayer > 0)
    {
        SKIP("Devices created without the debug layer are alive, so validation cannot be turned on.");
    }

    {
        DiagnosticLog standardLog;
        auto standard = rachis::createDevice(warpDeviceDesc(rachis::ValidationLevel::standard, false, &standardLog));
        REQUIRE(standard.has_value());
        CHECK(standard->diagnosticCounts().warnings == 0);
        auto unvalidated = rachis::createDevice(warpDeviceDesc(rachis::ValidationLevel::disabled, false, nullptr));
        REQUIRE(unvalidated.has_value());
        const auto beforeGpu = rachis::d3d12::test::debugLayerProcessSnapshot();
        CHECK(beforeGpu.debugLayerEnabled);
        CHECK_FALSE(beforeGpu.gpuValidationEnabled);
        CHECK(beforeGpu.devicesWithoutDebugLayer == initial.devicesWithoutDebugLayer);
        CHECK(beforeGpu.devicesWithoutGpuValidation == initial.devicesWithoutGpuValidation + 2);

        DiagnosticLog strictLog;
        auto strict = rachis::createDevice(warpDeviceDesc(rachis::ValidationLevel::gpu, true, &strictLog));
        REQUIRE_FALSE(strict.has_value());
        CHECK(strict.error().code == rachis::ErrorCode::validationError);
        UNSCOPED_INFO(strict.error().message);
        CHECK(strict.error().message.contains("created without GPU-based validation"));

        DiagnosticLog lenientLog;
        auto lenient = rachis::createDevice(warpDeviceDesc(rachis::ValidationLevel::gpu, false, &lenientLog));
        REQUIRE(lenient.has_value());
        CHECK(lenient->diagnosticCounts().warnings == 1);
        CHECK(lenientLog.contains(rachis::DiagnosticSeverity::warning, "runs with standard validation instead"));
        const auto afterLenient = rachis::d3d12::test::debugLayerProcessSnapshot();
        CHECK_FALSE(afterLenient.gpuValidationEnabled);
        CHECK(afterLenient.devicesWithoutGpuValidation == initial.devicesWithoutGpuValidation + 3);

        requireCopyRoundTrip(*standard, 7);
        requireCopyRoundTrip(*unvalidated, 8);
        requireCopyRoundTrip(*lenient, 9);
        CHECK(standard->diagnosticCounts().errors == 0);
        CHECK(lenient->diagnosticCounts().errors == 0);
    }

    CHECK(rachis::d3d12::test::debugLayerProcessSnapshot().devicesWithoutGpuValidation ==
          initial.devicesWithoutGpuValidation);

    DiagnosticLog gpuLog;
    auto gpuValidated = rachis::createDevice(warpDeviceDesc(rachis::ValidationLevel::gpu, false, &gpuLog));
    REQUIRE(gpuValidated.has_value());
    CHECK(gpuValidated->diagnosticCounts().warnings == 0);
    const auto afterGpu = rachis::d3d12::test::debugLayerProcessSnapshot();
    CHECK(afterGpu.gpuValidationEnabled);
    CHECK(afterGpu.devicesWithoutGpuValidation == initial.devicesWithoutGpuValidation);

    auto laterUnvalidated = rachis::createDevice(warpDeviceDesc(rachis::ValidationLevel::disabled, false, nullptr));
    REQUIRE(laterUnvalidated.has_value());
    CHECK(rachis::d3d12::test::debugLayerProcessSnapshot().devicesWithoutGpuValidation ==
          initial.devicesWithoutGpuValidation);

    DiagnosticLog requiredLog;
    auto required = rachis::createDevice(warpDeviceDesc(rachis::ValidationLevel::gpu, true, &requiredLog));
    if (!required)
    {
        UNSCOPED_INFO(required.error().message);
    }
    REQUIRE(required.has_value());
    CHECK(required->diagnosticCounts().warnings == 0);

    requireCopyRoundTrip(*gpuValidated, 10);
    requireCopyRoundTrip(*laterUnvalidated, 11);
    requireCopyRoundTrip(*required, 12);
    CHECK(gpuValidated->diagnosticCounts().errors == 0);
    CHECK(gpuValidated->diagnosticCounts().corruptions == 0);
    CHECK(required->diagnosticCounts().errors == 0);
    CHECK(required->diagnosticCounts().corruptions == 0);
}

TEST_CASE("D3D12 adapter enumeration and device creation call D3D12CreateDevice under the process lock",
          "[size:medium][rhi][diagnostics][validation][warp]")
{
    auto& log = nativeDeviceCreationLog();
    log.observed.store(0);
    log.unserialized.store(0);
    {
        const NativeDeviceCreationObserverScope observer(&observeNativeDeviceCreation);
        auto adapters = rachis::enumerateAdapters(rachis::Backend::d3d12);
        REQUIRE(adapters.has_value());
        CHECK_FALSE(adapters->empty());
        const auto enumerationCreations = log.observed.load();
        CHECK(enumerationCreations > 0);

        auto created = rachis::createDevice(warpDeviceDesc(rachis::ValidationLevel::disabled, false, nullptr));
        REQUIRE(created.has_value());
        CHECK(log.observed.load() > enumerationCreations);
    }
    CHECK(log.unserialized.load() == 0);
}

TEST_CASE("D3D12 device removal first seen by resource creation is published as device loss",
          "[size:medium][rhi][diagnostics][dred][warp]")
{
    DiagnosticLog log;
    auto created = rachis::createDevice(warpDeviceDesc(rachis::ValidationLevel::disabled, false, &log));
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(device.operationalState() == rachis::DeviceOperationalState::active);

    SECTION("texture creation sees the removal first")
    {
        removeWarpDevice();
        auto texture = device.createTexture(removalTextureDesc());
        REQUIRE_FALSE(texture.has_value());
        INFO(texture.error().message);
        CHECK(texture.error().code == rachis::ErrorCode::deviceLost);
        CHECK(texture.error().nativeCode != 0);
        checkPublishedLoss(device, log);

        auto buffer = device.createBuffer(removalBufferDesc());
        REQUIRE_FALSE(buffer.has_value());
        CHECK(buffer.error().code == rachis::ErrorCode::deviceLost);
    }

    SECTION("buffer creation sees the removal first")
    {
        removeWarpDevice();
        auto buffer = device.createBuffer(removalBufferDesc());
        REQUIRE_FALSE(buffer.has_value());
        INFO(buffer.error().message);
        CHECK(buffer.error().code == rachis::ErrorCode::deviceLost);
        CHECK(buffer.error().nativeCode != 0);
        checkPublishedLoss(device, log);

        auto texture = device.createTexture(removalTextureDesc());
        REQUIRE_FALSE(texture.has_value());
        CHECK(texture.error().code == rachis::ErrorCode::deviceLost);
    }

    SECTION("a new WARP device reports the removal instead of a missing feature level")
    {
        removeWarpDevice();
        auto replacement = rachis::createDevice(warpDeviceDesc(rachis::ValidationLevel::disabled, false, nullptr));
        REQUIRE_FALSE(replacement.has_value());
        INFO(replacement.error().message);
        CHECK(replacement.error().code == rachis::ErrorCode::deviceCreationFailed);
        CHECK(replacement.error().nativeCode == static_cast<std::int64_t>(DXGI_ERROR_DEVICE_REMOVED));
    }
}

TEST_CASE("D3D12 device removal first seen by a shared timeline wait is published as device loss",
          "[size:medium][rhi][diagnostics][dred][warp]")
{
    DiagnosticLog log;
    auto created = rachis::createDevice(warpDeviceDesc(rachis::ValidationLevel::disabled, false, &log));
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command.has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    auto shared = device.exportTimelinePoint(*submitted);
    REQUIRE(shared.has_value());
    REQUIRE(device.operationalState() == rachis::DeviceOperationalState::active);

    removeWarpDevice();
    auto waited = device.enqueueSharedTimelineWait(rachis::QueueType::graphics, shared->view());
    REQUIRE_FALSE(waited.has_value());
    INFO(waited.error().message);
    CHECK(waited.error().code == rachis::ErrorCode::deviceLost);
    CHECK(waited.error().nativeCode != 0);
    checkPublishedLoss(device, log);

    auto buffer = device.createBuffer(removalBufferDesc());
    REQUIRE_FALSE(buffer.has_value());
    CHECK(buffer.error().code == rachis::ErrorCode::deviceLost);
}

TEST_CASE("a published D3D12 device loss counts one error diagnostic", "[size:medium][rhi][diagnostics][dred][warp]")
{
    const bool withCallback = GENERATE(false, true);
    CAPTURE(withCallback);
    DiagnosticLog log;
    auto created =
        rachis::createDevice(warpDeviceDesc(rachis::ValidationLevel::disabled, false, withCallback ? &log : nullptr));
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    const auto before = device.diagnosticCounts();

    removeWarpDevice();
    auto buffer = device.createBuffer(removalBufferDesc());
    REQUIRE_FALSE(buffer.has_value());
    CHECK(buffer.error().code == rachis::ErrorCode::deviceLost);
    auto texture = device.createTexture(removalTextureDesc());
    REQUIRE_FALSE(texture.has_value());
    CHECK(texture.error().code == rachis::ErrorCode::deviceLost);
    CHECK(device.operationalState() == rachis::DeviceOperationalState::lost);

    const auto after = device.diagnosticCounts();
    CHECK(after.errors == before.errors + 1);
    CHECK(after.corruptions == before.corruptions);
    CHECK(after.warnings == before.warnings);
    CHECK(after.information == before.information);
    if (withCallback)
    {
        CHECK(log.contains(rachis::DiagnosticSeverity::error, "D3D12 device fault during"));
        const std::scoped_lock lock(log.mutex);
        CHECK(std::ranges::count(log.severities, rachis::DiagnosticSeverity::error) == 1);
    }
}

TEST_CASE("D3D12 completed timeline values stay monotonic after device removal",
          "[size:medium][rhi][diagnostics][device-loss][warp]")
{
    auto created = rachis::createDevice(warpDeviceDesc(rachis::ValidationLevel::disabled, false, nullptr));
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command.has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    const auto copyCompleted = device.completedValue(rachis::QueueType::copy);
    REQUIRE(copyCompleted >= submitted->value());
    const auto graphicsCompleted = device.completedValue(rachis::QueueType::graphics);

    removeWarpDevice();
    const auto copyAfterRemoval = device.completedValue(rachis::QueueType::copy);
    CHECK(copyAfterRemoval >= copyCompleted);
    CHECK(copyAfterRemoval < UINT64_MAX);
    const auto graphicsAfterRemoval = device.completedValue(rachis::QueueType::graphics);
    CHECK(graphicsAfterRemoval >= graphicsCompleted);
    CHECK(graphicsAfterRemoval < UINT64_MAX);

    auto waited = device.wait(*submitted);
    if (!waited)
    {
        CHECK(waited.error().code == rachis::ErrorCode::deviceLost);
    }
    CHECK(device.completedValue(rachis::QueueType::copy) == copyAfterRemoval);
}
