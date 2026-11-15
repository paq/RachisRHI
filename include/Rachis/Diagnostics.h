#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace rachis
{

    enum class DiagnosticSeverity : std::uint8_t
    {
        information,
        warning,
        error,
        corruption,
    };

    struct DiagnosticMessage final
    {
        DiagnosticSeverity severity = DiagnosticSeverity::information;
        std::uint32_t nativeId = 0;
        std::string_view description = {};
    };

    using DiagnosticCallback = void (*)(const DiagnosticMessage& message, void* userData) noexcept;

    struct DiagnosticCounts final
    {
        std::uint64_t information = 0;
        std::uint64_t warnings = 0;
        std::uint64_t errors = 0;
        std::uint64_t corruptions = 0;
    };

    struct BackendAllocationSnapshot final
    {
        std::uint64_t liveDescriptorCount = 0;
        std::uint64_t reservedDescriptorCount = 0;
    };

    enum class DeviceOperationalState : std::uint8_t
    {
        invalid,
        active,
        lost,
    };

    enum class DeviceFaultStatus : std::uint8_t
    {
        healthy,
        removed,
        reset,
        hung,
        outOfMemory,
        driverInternalError,
        invalidCall,
        unknown,
    };

    struct DeviceFaultOperation final
    {
        std::uint32_t index = 0;
        std::uint32_t nativeOperation = 0;
        bool completed = false;
        std::string name = {};
        std::string context = {};
    };

    struct DeviceFaultBreadcrumb final
    {
        std::string queueLabel = {};
        std::string commandListLabel = {};
        std::uint32_t completedOperationCount = 0;
        std::uint32_t totalOperationCount = 0;
        std::uint32_t firstCapturedOperation = 0;
        std::vector<DeviceFaultOperation> operations = {};
        bool operationsTruncated = false;
        bool contextsTruncated = false;
    };

    enum class DeviceFaultAllocationStatus : std::uint8_t
    {
        existing,
        recentlyFreed,
    };

    struct DeviceFaultAllocation final
    {
        DeviceFaultAllocationStatus status = DeviceFaultAllocationStatus::existing;
        std::uint32_t nativeType = 0;
        std::string typeName = {};
        std::string label = {};
    };

    struct DeviceFaultReport final
    {
        DeviceFaultStatus status = DeviceFaultStatus::healthy;
        std::int64_t nativeReason = 0;
        bool dredAvailable = false;
        std::uint64_t pageFaultAddress = 0;
        std::vector<DeviceFaultBreadcrumb> breadcrumbs = {};
        std::vector<DeviceFaultAllocation> allocations = {};
        bool breadcrumbsTruncated = false;
        bool allocationsTruncated = false;
    };

} // namespace rachis
