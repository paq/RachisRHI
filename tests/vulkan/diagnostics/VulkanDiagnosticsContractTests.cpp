#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include "Internal/VulkanDiagnostics.h"
#include "Internal/VulkanRuntime.h"
#include "Internal/VulkanWindowsProfile.h"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

    struct DiagnosticCapture final
    {
        std::atomic<std::uint64_t> warnings{0};
        std::atomic<std::uint64_t> errors{0};
        std::atomic<std::uint64_t> information{0};
    };

    void captureDiagnostic(const rachis::DiagnosticMessage& message, void* userData) noexcept
    {
        auto* capture = static_cast<DiagnosticCapture*>(userData);
        if (capture == nullptr)
        {
            return;
        }
        switch (message.severity)
        {
        case rachis::DiagnosticSeverity::information:
            capture->information.fetch_add(1, std::memory_order_relaxed);
            break;
        case rachis::DiagnosticSeverity::warning:
            capture->warnings.fetch_add(1, std::memory_order_relaxed);
            break;
        case rachis::DiagnosticSeverity::error:
        case rachis::DiagnosticSeverity::corruption:
            capture->errors.fetch_add(1, std::memory_order_relaxed);
            break;
        }
    }

    [[nodiscard]] rachis::DeviceDesc vulkanDesc(rachis::ValidationLevel validation,
                                                bool requireValidation,
                                                rachis::DiagnosticCallback callback = nullptr,
                                                void* userData = nullptr) noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .validation = validation,
            .requireValidation = requireValidation,
            .enableDred = false,
            .diagnosticCallback = callback,
            .diagnosticUserData = userData,
        };
    }

    struct InstanceTeardownTrace final
    {
        std::vector<std::string> calls;
        rachis::vulkan::VulkanDiagnosticSink* sink = nullptr;
    };

    InstanceTeardownTrace* gInstanceTeardown = nullptr;

    VKAPI_ATTR void VKAPI_CALL traceDestroyDebugUtilsMessenger(VkInstance,
                                                               VkDebugUtilsMessengerEXT,
                                                               const VkAllocationCallbacks*)
    {
        gInstanceTeardown->calls.emplace_back("vkDestroyDebugUtilsMessengerEXT");
    }

    VKAPI_ATTR void VKAPI_CALL traceDestroyInstance(VkInstance, const VkAllocationCallbacks*)
    {
        gInstanceTeardown->calls.emplace_back("vkDestroyInstance");
        const VkDebugUtilsMessengerCallbackDataEXT data{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT,
            .pNext = {},
            .flags = {},
            .pMessageIdName = "VUID-vkDestroyInstance-instance-00629",
            .messageIdNumber = 0,
            .pMessage = "VkSurfaceKHR 0x60 has not been destroyed.",
            .queueLabelCount = 0,
            .pQueueLabels = {},
            .cmdBufLabelCount = 0,
            .pCmdBufLabels = {},
            .objectCount = 0,
            .pObjects = {},
        };
        static_cast<void>(rachis::vulkan::vulkanDebugUtilsCallback(VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                                                                   VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
                                                                   &data,
                                                                   gInstanceTeardown->sink));
    }

    void collectDescription(const rachis::DiagnosticMessage& message, void* userData) noexcept
    {
        static_cast<std::vector<std::string>*>(userData)->emplace_back(message.description);
    }

} // namespace

TEST_CASE("Vulkan diagnostic sink ignores callbacks after destruction and stays thread-safe",
          "[size:small][rhi][vulkan][diagnostics]")
{
    rachis::vulkan::VulkanDiagnosticSink sink;
    sink.publish(rachis::DiagnosticSeverity::warning, 1, "before");
    CHECK(sink.counts().warnings == 1);

    sink.alive.store(false, std::memory_order_release);
    sink.publish(rachis::DiagnosticSeverity::error, 2, "after death");
    CHECK(sink.counts().warnings == 1);
    CHECK(sink.counts().errors == 0);

    sink.alive.store(true, std::memory_order_release);
    std::thread first([&] {
        for (int i = 0; i < 64; ++i)
        {
            sink.publish(rachis::DiagnosticSeverity::warning, 3, "threaded warning");
        }
    });
    std::thread second([&] {
        for (int i = 0; i < 64; ++i)
        {
            sink.publish(rachis::DiagnosticSeverity::error, 4, "threaded error");
        }
    });
    first.join();
    second.join();
    CHECK(sink.counts().warnings == 65);
    CHECK(sink.counts().errors == 64);
}

TEST_CASE("Vulkan instance teardown delivers the diagnostics that vkDestroyInstance raises",
          "[size:small][rhi][vulkan][diagnostics]")
{
    std::vector<std::string> messages;
    InstanceTeardownTrace trace;
    rachis::vulkan::VulkanInstanceInfo info{};
    info.instance = reinterpret_cast<VkInstance>(std::uintptr_t{0x50});
    info.messenger = reinterpret_cast<VkDebugUtilsMessengerEXT>(std::uintptr_t{0x51});
    info.table.vkDestroyDebugUtilsMessengerEXT = &traceDestroyDebugUtilsMessenger;
    info.table.vkDestroyInstance = &traceDestroyInstance;
    info.sink = std::make_unique<rachis::vulkan::VulkanDiagnosticSink>();
    info.sink->callback = &collectDescription;
    info.sink->userData = &messages;
    trace.sink = info.sink.get();

    gInstanceTeardown = &trace;
    rachis::vulkan::destroyVulkanInstance(info);
    gInstanceTeardown = nullptr;

    CHECK(info.instance == VK_NULL_HANDLE);
    CHECK(info.messenger == VK_NULL_HANDLE);
    CHECK(info.sink == nullptr);
    CHECK(trace.calls == std::vector<std::string>{"vkDestroyDebugUtilsMessengerEXT", "vkDestroyInstance"});
    REQUIRE(messages.size() == 1);
    CHECK(messages.front() == "VkSurfaceKHR 0x60 has not been destroyed.");
}

TEST_CASE("Vulkan loader warnings about the host driver set are information, not device warnings",
          "[size:small][rhi][vulkan][diagnostics]")
{
    rachis::vulkan::VulkanDiagnosticSink sink;
    const auto deliver = [&sink](VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                 VkDebugUtilsMessageTypeFlagsEXT types,
                                 const char* idName) {
        const VkDebugUtilsMessengerCallbackDataEXT data{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT,
            .pNext = {},
            .flags = {},
            .pMessageIdName = idName,
            .messageIdNumber = 0,
            .pMessage = "message",
            .queueLabelCount = 0,
            .pQueueLabels = {},
            .cmdBufLabelCount = 0,
            .pCmdBufLabels = {},
            .objectCount = 0,
            .pObjects = {},
        };
        CHECK(rachis::vulkan::vulkanDebugUtilsCallback(severity, types, &data, &sink) == VK_FALSE);
    };

    deliver(
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT, VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT, "Loader Message");
    CHECK(sink.counts().information == 1);
    CHECK(sink.counts().warnings == 0);

    deliver(VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT, VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT, "VUID-x");
    deliver(VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT, VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT, nullptr);
    deliver(
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT, VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT, "Loader Message");
    CHECK(sink.counts().information == 1);
    CHECK(sink.counts().warnings == 2);
    CHECK(sink.counts().errors == 1);
}

TEST_CASE("Vulkan validation layer policy is independent of device availability",
          "[size:small][rhi][vulkan][diagnostics]")
{
    CHECK(rachis::planVulkanValidationLayers(false, false, false) == rachis::VulkanValidationLayerPlan::disabled);
    CHECK(rachis::planVulkanValidationLayers(false, false, true) == rachis::VulkanValidationLayerPlan::disabled);
    CHECK(rachis::planVulkanValidationLayers(true, false, true) == rachis::VulkanValidationLayerPlan::enableKhronos);
    CHECK(rachis::planVulkanValidationLayers(true, true, true) == rachis::VulkanValidationLayerPlan::enableKhronos);
    CHECK(rachis::planVulkanValidationLayers(true, true, false) == rachis::VulkanValidationLayerPlan::requiredMissing);
    CHECK(rachis::planVulkanValidationLayers(true, false, false) == rachis::VulkanValidationLayerPlan::optionalMissing);
}

TEST_CASE("required Vulkan validation fails clearly and optional absence is reported",
          "[size:medium][rhi][vulkan][diagnostics]")
{
    DiagnosticCapture requiredCapture;
    auto required =
        rachis::createDevice(vulkanDesc(rachis::ValidationLevel::standard, true, &captureDiagnostic, &requiredCapture));
    if (!required)
    {
        if (required.error().code == rachis::ErrorCode::backendUnavailable)
        {
            CHECK_FALSE(required.error().message.contains("KHRONOS_validation"));
        }
        else if (required.error().code == rachis::ErrorCode::validationError)
        {
            CHECK(required.error().message.contains("KHRONOS_validation"));
            CHECK(required.error().message.contains("D3D12"));
        }
        else if (rachis::test::skipWithoutCreatedDevice(required))
        {
            return;
        }
    }
    else
    {
        CHECK(required->diagnosticCounts().errors == 0);
        CHECK(required->diagnosticCounts().corruptions == 0);
        CHECK(requiredCapture.errors.load() == 0);
    }

    DiagnosticCapture optionalCapture;
    auto optional = rachis::createDevice(
        vulkanDesc(rachis::ValidationLevel::standard, false, &captureDiagnostic, &optionalCapture));
    if (!optional)
    {
        CHECK(optional.error().code != rachis::ErrorCode::validationError);
    }
    if (rachis::test::skipWithoutCreatedDevice(optional))
    {
        return;
    }
    CHECK(optional->diagnosticCounts().errors == 0);
    CHECK(optional->diagnosticCounts().corruptions == 0);
}

TEST_CASE("valid Vulkan bootstrap produces no unexplained diagnostics", "[size:medium][rhi][vulkan][diagnostics]")
{
    DiagnosticCapture capture;
    auto created =
        rachis::createDevice(vulkanDesc(rachis::ValidationLevel::standard, false, &captureDiagnostic, &capture));
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }

    REQUIRE(created->adapterInfo().has_value());
    CHECK(rachis::test::requireAdapterInfo(*created).backend == rachis::Backend::vulkan);
    CHECK(created->diagnosticCounts().errors == 0);
    CHECK(created->diagnosticCounts().corruptions == 0);
    CHECK(capture.errors.load() == 0);

    auto report = created->captureDeviceFaultReport();
    REQUIRE(report);
    CHECK(report->status == rachis::DeviceFaultStatus::healthy);
    CHECK_FALSE(report->dredAvailable);
    CHECK(report->breadcrumbs.empty());
    CHECK(report->allocations.empty());

    auto clock = created->calibrateQueueClock(rachis::QueueType::graphics);
    if (!clock)
    {
        CHECK(clock.error().code == rachis::ErrorCode::unsupported);
        return;
    }
    CHECK(clock->cpuTimestamp > 0);
    CHECK(clock->cpuFrequency > 0);
    CHECK(clock->gpuFrequency == created->queueProperties(rachis::QueueType::graphics).timestampFrequency);
    CHECK(created->diagnosticCounts().errors == 0);
}

TEST_CASE("intentional invalid Vulkan work produces attributed diagnostics", "[size:medium][rhi][vulkan][diagnostics]")
{
    DiagnosticCapture capture;
    auto created =
        rachis::createDevice(vulkanDesc(rachis::ValidationLevel::standard, false, &captureDiagnostic, &capture));
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }

    const auto before = created->diagnosticCounts();
    auto failed = rachis::vulkan::emitIntentionalValidationFailure();
    REQUIRE_FALSE(failed);
    CHECK(failed.error().message.contains("vkCreateBuffer"));
    CHECK(failed.error().message.contains("did not fall back to D3D12"));

    const auto after = created->diagnosticCounts();
    const bool attributed = after.errors > before.errors || after.warnings > before.warnings ||
                            after.information > before.information || capture.errors.load() > 0 ||
                            capture.warnings.load() > 0 || failed.error().nativeCode != 0;
    CHECK(attributed);
}
