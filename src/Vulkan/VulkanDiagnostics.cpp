#include "Internal/VulkanDiagnostics.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <new>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace rachis::vulkan
{
    namespace
    {

#ifdef RACHIS_TESTING
        std::mutex testDeviceMutex;
        VkDevice testDevice = VK_NULL_HANDLE;
        const VolkDeviceTable* testTable = nullptr;
#endif

        [[nodiscard]] DiagnosticSeverity messageSeverity(VkDebugUtilsMessageSeverityFlagBitsEXT severity) noexcept
        {
            if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0)
            {
                return DiagnosticSeverity::error;
            }
            if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0)
            {
                return DiagnosticSeverity::warning;
            }
            return DiagnosticSeverity::information;
        }

        [[nodiscard]] bool isLoaderMessage(VkDebugUtilsMessageTypeFlagsEXT types,
                                           const VkDebugUtilsMessengerCallbackDataEXT& data) noexcept
        {
            return types == VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT && data.pMessageIdName != nullptr &&
                   std::string_view(data.pMessageIdName) == "Loader Message";
        }

        [[nodiscard]] DiagnosticSeverity callbackSeverity(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                          VkDebugUtilsMessageTypeFlagsEXT types,
                                                          const VkDebugUtilsMessengerCallbackDataEXT& data) noexcept
        {
            const auto mapped = messageSeverity(severity);
            if (mapped == DiagnosticSeverity::warning && isLoaderMessage(types, data))
            {
                return DiagnosticSeverity::information;
            }
            return mapped;
        }

    } // namespace

    void VulkanDiagnosticSink::publish(DiagnosticSeverity severity,
                                       std::uint32_t nativeId,
                                       std::string_view description) noexcept
    {
        if (!alive.load(std::memory_order_acquire))
        {
            return;
        }

        switch (severity)
        {
        case DiagnosticSeverity::information:
            information.fetch_add(1, std::memory_order_relaxed);
            break;
        case DiagnosticSeverity::warning:
            warnings.fetch_add(1, std::memory_order_relaxed);
            break;
        case DiagnosticSeverity::error:
            errors.fetch_add(1, std::memory_order_relaxed);
            break;
        case DiagnosticSeverity::corruption:
            corruptions.fetch_add(1, std::memory_order_relaxed);
            break;
        }

        const DiagnosticMessage message{
            .severity = severity,
            .nativeId = nativeId,
            .description = description,
        };
        const std::scoped_lock lock(mutex);
        if (!alive.load(std::memory_order_acquire) || callback == nullptr)
        {
            return;
        }
        callback(message, userData);
    }

    DiagnosticCounts VulkanDiagnosticSink::counts() const noexcept
    {
        return DiagnosticCounts{
            .information = information.load(std::memory_order_relaxed),
            .warnings = warnings.load(std::memory_order_relaxed),
            .errors = errors.load(std::memory_order_relaxed),
            .corruptions = corruptions.load(std::memory_order_relaxed),
        };
    }

    const char* vulkanResultName(VkResult result) noexcept
    {
        switch (result)
        {
        case VK_SUCCESS:
            return "VK_SUCCESS";
        case VK_NOT_READY:
            return "VK_NOT_READY";
        case VK_TIMEOUT:
            return "VK_TIMEOUT";
        case VK_EVENT_SET:
            return "VK_EVENT_SET";
        case VK_EVENT_RESET:
            return "VK_EVENT_RESET";
        case VK_INCOMPLETE:
            return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY:
            return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
            return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED:
            return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST:
            return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED:
            return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT:
            return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT:
            return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT:
            return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER:
            return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS:
            return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED:
            return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL:
            return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_OUT_OF_POOL_MEMORY:
            return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_UNKNOWN:
            return "VK_ERROR_UNKNOWN";
        case VK_ERROR_VALIDATION_FAILED_EXT:
            return "VK_ERROR_VALIDATION_FAILED_EXT";
        default:
            return "VkResult";
        }
    }

    ErrorCode vulkanResultErrorCode(VkResult result) noexcept
    {
        switch (result)
        {
        case VK_ERROR_OUT_OF_HOST_MEMORY:
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
            return ErrorCode::outOfMemory;
        case VK_ERROR_OUT_OF_POOL_MEMORY:
        case VK_ERROR_FRAGMENTED_POOL:
            return ErrorCode::resourceExhausted;
        case VK_ERROR_DEVICE_LOST:
            return ErrorCode::deviceLost;
        case VK_ERROR_LAYER_NOT_PRESENT:
        case VK_ERROR_VALIDATION_FAILED_EXT:
            return ErrorCode::validationError;
        case VK_ERROR_EXTENSION_NOT_PRESENT:
        case VK_ERROR_FEATURE_NOT_PRESENT:
            return ErrorCode::unsupported;
        default:
            return ErrorCode::nativeFailure;
        }
    }

    Error makeVulkanError(ErrorCode code, VkResult result, std::string_view operation)
    {
        return Error{
            code,
            result,
            std::string(operation) + " failed with " + vulkanResultName(result) +
                ". Selection did not fall back to D3D12.",
        };
    }

    VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugUtilsCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                            VkDebugUtilsMessageTypeFlagsEXT types,
                                                            const VkDebugUtilsMessengerCallbackDataEXT* data,
                                                            void* userData)
    {
        auto* sink = static_cast<VulkanDiagnosticSink*>(userData);
        if (sink == nullptr || data == nullptr)
        {
            return VK_FALSE;
        }
        const std::string_view description = data->pMessage != nullptr ? data->pMessage : "";
        sink->publish(
            callbackSeverity(severity, types, *data), static_cast<std::uint32_t>(data->messageIdNumber), description);
        return VK_FALSE;
    }

    void setVulkanObjectName(const VulkanInstanceTable* instanceTable,
                             VkDevice device,
                             VkObjectType type,
                             std::uint64_t handle,
                             const char* name) noexcept
    {
        if (instanceTable == nullptr || device == VK_NULL_HANDLE || name == nullptr ||
            instanceTable->vkSetDebugUtilsObjectNameEXT == nullptr)
        {
            return;
        }
        const VkDebugUtilsObjectNameInfoEXT info{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .pNext = {},
            .objectType = type,
            .objectHandle = handle,
            .pObjectName = name,
        };
        static_cast<void>(instanceTable->vkSetDebugUtilsObjectNameEXT(device, &info));
    }

    namespace
    {

        constexpr std::uint32_t maxCapturedFaultAddresses = 64;
        constexpr std::uint32_t maxCapturedFaultVendorInfos = 32;

        [[nodiscard]] std::string_view boundedDescription(const char* buffer, std::size_t capacity) noexcept
        {
            if (buffer == nullptr || capacity == 0)
            {
                return {};
            }
            const auto* terminator = static_cast<const char*>(std::memchr(buffer, '\0', capacity));
            const auto length = terminator != nullptr ? static_cast<std::size_t>(terminator - buffer) : capacity;
            return std::string_view(buffer, length);
        }

        [[nodiscard]] std::uint64_t alignFaultAddress(VkDeviceAddress reportedAddress, VkDeviceSize precision) noexcept
        {
            if (precision > 1 && (precision & (precision - 1)) == 0)
            {
                return reportedAddress & ~(precision - 1);
            }
            return reportedAddress;
        }

        [[nodiscard]] std::string_view faultAddressTypeName(VkDeviceFaultAddressTypeEXT type) noexcept
        {
            switch (type)
            {
            case VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_EXT:
                return "none";
            case VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT:
                return "readInvalid";
            case VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT:
                return "writeInvalid";
            case VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT:
                return "executeInvalid";
            case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_EXT:
                return "instructionPointerUnknown";
            case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_INVALID_EXT:
                return "instructionPointerInvalid";
            case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT:
                return "instructionPointerFault";
            default:
                return "unknown";
            }
        }

        [[nodiscard]] bool isDataAccessFaultAddress(VkDeviceFaultAddressTypeEXT type) noexcept
        {
            return type == VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT ||
                   type == VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT ||
                   type == VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT;
        }

        template<typename Entry>
        struct FaultEntryCapture final
        {
            std::vector<Entry> entries;
            std::uint32_t reportedCount = 0;
            bool truncated = false;
        };

        struct CapturedDeviceFault final
        {
            FaultEntryCapture<VkDeviceFaultAddressInfoEXT> addresses;
            FaultEntryCapture<VkDeviceFaultVendorInfoEXT> vendors;
            std::array<char, VK_MAX_DESCRIPTION_SIZE> description = {};
            bool vendorBinaryOmitted = false;
        };

        [[nodiscard]] std::string_view faultDescription(const CapturedDeviceFault& fault) noexcept
        {
            return boundedDescription(fault.description.data(), fault.description.size());
        }

        [[nodiscard]] std::optional<VkDeviceFaultCountsEXT> queryReportedFaultCounts(
            VkDevice device, const VolkDeviceTable& table) noexcept
        {
            VkDeviceFaultCountsEXT counts{
                .sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT,
                .pNext = nullptr,
                .addressInfoCount = 0,
                .vendorInfoCount = 0,
                .vendorBinarySize = 0,
            };
            const VkResult countResult = table.vkGetDeviceFaultInfoEXT(device, &counts, nullptr);
            if (countResult != VK_SUCCESS && countResult != VK_INCOMPLETE)
            {
                return std::nullopt;
            }
            return counts;
        }

        template<typename Entry>
        [[nodiscard]] FaultEntryCapture<Entry> allocateFaultEntries(std::uint32_t reportedCount,
                                                                    std::uint32_t captureLimit) noexcept
        {
            const std::uint32_t requestedCount = std::min(reportedCount, captureLimit);
            FaultEntryCapture<Entry> capture{
                .entries = {},
                .reportedCount = reportedCount,
                .truncated = reportedCount > requestedCount,
            };
            try
            {
                capture.entries.resize(requestedCount);
            }
            catch (const std::bad_alloc&)
            {
                capture.entries.clear();
                capture.truncated = capture.truncated || reportedCount > 0;
            }
            return capture;
        }

        template<typename Entry>
        void discardUnwrittenFaultEntries(FaultEntryCapture<Entry>& capture, std::uint32_t writtenCount) noexcept
        {
            capture.entries.resize(std::min<std::size_t>(writtenCount, capture.entries.size()));
        }

        [[nodiscard]] std::optional<CapturedDeviceFault> queryDeviceFault(VkDevice device,
                                                                          const VolkDeviceTable& table) noexcept
        {
            const auto reportedCounts = queryReportedFaultCounts(device, table);
            if (!reportedCounts)
            {
                return std::nullopt;
            }

            auto addresses = allocateFaultEntries<VkDeviceFaultAddressInfoEXT>(reportedCounts->addressInfoCount,
                                                                               maxCapturedFaultAddresses);
            auto vendors = allocateFaultEntries<VkDeviceFaultVendorInfoEXT>(reportedCounts->vendorInfoCount,
                                                                            maxCapturedFaultVendorInfos);

            VkDeviceFaultCountsEXT capturedCounts = *reportedCounts;
            capturedCounts.addressInfoCount = static_cast<std::uint32_t>(addresses.entries.size());
            capturedCounts.vendorInfoCount = static_cast<std::uint32_t>(vendors.entries.size());
            capturedCounts.vendorBinarySize = 0;

            VkDeviceFaultInfoEXT info{
                .sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT,
                .pNext = nullptr,
                .description = {},
                .pAddressInfos = addresses.entries.empty() ? nullptr : addresses.entries.data(),
                .pVendorInfos = vendors.entries.empty() ? nullptr : vendors.entries.data(),
                .pVendorBinaryData = nullptr,
            };
            const VkResult infoResult = table.vkGetDeviceFaultInfoEXT(device, &capturedCounts, &info);
            if (infoResult != VK_SUCCESS && infoResult != VK_INCOMPLETE)
            {
                return std::nullopt;
            }

            const bool incomplete = infoResult == VK_INCOMPLETE;
            const bool vendorBinaryReported = reportedCounts->vendorBinarySize > 0;
            if (incomplete && !vendorBinaryReported)
            {
                addresses.truncated = true;
                vendors.truncated = true;
            }
            discardUnwrittenFaultEntries(addresses, capturedCounts.addressInfoCount);
            discardUnwrittenFaultEntries(vendors, capturedCounts.vendorInfoCount);

            return CapturedDeviceFault{
                .addresses = std::move(addresses),
                .vendors = std::move(vendors),
                .description = std::to_array(info.description),
                .vendorBinaryOmitted = incomplete && vendorBinaryReported,
            };
        }

        [[nodiscard]] DeviceFaultAllocation makeFaultAddressAllocation(const VkDeviceFaultAddressInfoEXT& entry)
        {
            const auto aligned = alignFaultAddress(entry.reportedAddress, entry.addressPrecision);
            std::ostringstream addressStream;
            addressStream << "address=0x" << std::hex << aligned << " reported=0x" << entry.reportedAddress
                          << " precision=0x" << entry.addressPrecision;
            return DeviceFaultAllocation{
                .status = DeviceFaultAllocationStatus::existing,
                .nativeType = static_cast<std::uint32_t>(entry.addressType),
                .typeName = std::string(faultAddressTypeName(entry.addressType)),
                .label = addressStream.str(),
            };
        }

        [[nodiscard]] std::uint64_t selectPageFaultAddress(
            std::span<const VkDeviceFaultAddressInfoEXT> entries) noexcept
        {
            std::uint64_t firstFaultAddress = 0;
            for (const auto& entry : entries)
            {
                const auto aligned = alignFaultAddress(entry.reportedAddress, entry.addressPrecision);
                if (aligned == 0)
                {
                    continue;
                }
                if (isDataAccessFaultAddress(entry.addressType))
                {
                    return aligned;
                }
                if (firstFaultAddress == 0)
                {
                    firstFaultAddress = aligned;
                }
            }
            return firstFaultAddress;
        }

        void recordFaultAddresses(DeviceFaultReport& report,
                                  const FaultEntryCapture<VkDeviceFaultAddressInfoEXT>& addresses)
        {
            report.allocations.reserve(report.allocations.size() + addresses.entries.size());
            for (const auto& entry : addresses.entries)
            {
                report.allocations.push_back(makeFaultAddressAllocation(entry));
            }
            const auto pageFaultAddress = selectPageFaultAddress(addresses.entries);
            if (pageFaultAddress != 0)
            {
                report.pageFaultAddress = pageFaultAddress;
            }
            report.allocationsTruncated = report.allocationsTruncated || addresses.truncated;
        }

        [[nodiscard]] DeviceFaultOperation makeVendorFaultOperation(const VkDeviceFaultVendorInfoEXT& vendor,
                                                                    std::size_t index)
        {
            const auto vendorDesc = boundedDescription(vendor.description, VK_MAX_DESCRIPTION_SIZE);
            std::ostringstream vendorContext;
            vendorContext << "code=0x" << std::hex << vendor.vendorFaultCode << " data=0x" << vendor.vendorFaultData;
            return DeviceFaultOperation{
                .index = static_cast<std::uint32_t>(index),
                .nativeOperation = static_cast<std::uint32_t>(vendor.vendorFaultCode),
                .completed = false,
                .name = std::string(vendorDesc),
                .context = vendorContext.str(),
            };
        }

        [[nodiscard]] DeviceFaultBreadcrumb makeVendorFaultBreadcrumb(
            const FaultEntryCapture<VkDeviceFaultVendorInfoEXT>& vendors, std::string_view description)
        {
            const std::size_t capturedVendorCount = vendors.entries.size();
            DeviceFaultBreadcrumb breadcrumb{
                .queueLabel = "VK_EXT_device_fault",
                .commandListLabel = std::string(description),
                .completedOperationCount = 0,
                .totalOperationCount =
                    static_cast<std::uint32_t>(std::max<std::size_t>(capturedVendorCount, vendors.reportedCount)),
                .firstCapturedOperation = 0,
                .operations = {},
                .operationsTruncated = vendors.truncated,
                .contextsTruncated = false,
            };
            breadcrumb.operations.reserve(capturedVendorCount);
            for (std::size_t index = 0; index < capturedVendorCount; ++index)
            {
                breadcrumb.operations.push_back(makeVendorFaultOperation(vendors.entries[index], index));
            }
            return breadcrumb;
        }

        void recordVendorFaultBreadcrumb(DeviceFaultReport& report, const CapturedDeviceFault& fault)
        {
            const auto description = faultDescription(fault);
            if (!fault.vendors.entries.empty() || !description.empty() || fault.vendors.truncated)
            {
                report.breadcrumbs.push_back(makeVendorFaultBreadcrumb(fault.vendors, description));
            }
            report.breadcrumbsTruncated = report.breadcrumbsTruncated || fault.vendors.truncated;
        }

        [[nodiscard]] std::string formatFaultSummary(const CapturedDeviceFault& fault)
        {
            const auto description = faultDescription(fault);
            std::ostringstream summary;
            summary << "VK_EXT_device_fault:";
            if (!description.empty())
            {
                summary << " description='" << description << '\'';
            }
            else
            {
                summary << " description=<empty>";
            }
            for (std::size_t index = 0; index < fault.addresses.entries.size(); ++index)
            {
                const auto& entry = fault.addresses.entries[index];
                const auto aligned = alignFaultAddress(entry.reportedAddress, entry.addressPrecision);
                summary << " address[" << std::dec << index << "]={type=" << faultAddressTypeName(entry.addressType)
                        << " aligned=0x" << std::hex << aligned << " reported=0x" << entry.reportedAddress
                        << " precision=0x" << entry.addressPrecision << '}';
            }
            for (std::size_t index = 0; index < fault.vendors.entries.size(); ++index)
            {
                const auto& vendor = fault.vendors.entries[index];
                const auto vendorDesc = boundedDescription(vendor.description, VK_MAX_DESCRIPTION_SIZE);
                summary << " vendor[" << std::dec << index << "]={description='" << vendorDesc << "' code=0x"
                        << std::hex << vendor.vendorFaultCode << " data=0x" << vendor.vendorFaultData << '}';
            }
            if (fault.addresses.truncated || fault.vendors.truncated || fault.vendorBinaryOmitted)
            {
                summary << " [incomplete]";
            }
            return summary.str();
        }

    } // namespace

    bool captureVulkanDeviceFaultInfo(VkDevice device,
                                      const VolkDeviceTable& table,
                                      DeviceFaultReport& report,
                                      std::string& faultSummary) noexcept
    {
        faultSummary.clear();
        if (device == VK_NULL_HANDLE || table.vkGetDeviceFaultInfoEXT == nullptr)
        {
            return false;
        }

        const auto fault = queryDeviceFault(device, table);
        if (!fault)
        {
            return false;
        }

        try
        {
            recordFaultAddresses(report, fault->addresses);
            recordVendorFaultBreadcrumb(report, *fault);
            faultSummary = formatFaultSummary(*fault);
        }
        catch (const std::bad_alloc&)
        {
            report.allocationsTruncated =
                report.allocationsTruncated || fault->addresses.entries.size() > report.allocations.size();
            report.breadcrumbsTruncated = report.breadcrumbsTruncated || !fault->vendors.entries.empty();
            faultSummary.clear();
        }

        return true;
    }

#ifdef RACHIS_TESTING
    void bindVulkanValidationTestDevice(VkDevice device, const VolkDeviceTable* table)
    {
        const std::scoped_lock lock(testDeviceMutex);
        testDevice = device;
        testTable = table;
    }

    void unbindVulkanValidationTestDevice(VkDevice device) noexcept
    {
        const std::scoped_lock lock(testDeviceMutex);
        if (testDevice == device)
        {
            testDevice = VK_NULL_HANDLE;
            testTable = nullptr;
        }
    }

    Status emitIntentionalValidationFailure()
    {
        const std::scoped_lock lock(testDeviceMutex);
        if (testDevice == VK_NULL_HANDLE || testTable == nullptr || testTable->vkCreateBuffer == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "No live Vulkan test device is bound for intentional validation failure.",
            });
        }

        const VkBufferCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = {},
            .flags = {},
            .size = 0,
            .usage = 0,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = {},
            .pQueueFamilyIndices = {},
        };
        VkBuffer buffer = VK_NULL_HANDLE;
        const auto created = testTable->vkCreateBuffer(testDevice, &info, nullptr, &buffer);
        if (buffer != VK_NULL_HANDLE && testTable->vkDestroyBuffer != nullptr)
        {
            testTable->vkDestroyBuffer(testDevice, buffer, nullptr);
        }
        return std::unexpected(
            makeVulkanError(vulkanResultErrorCode(created), created, "intentional invalid vkCreateBuffer"));
    }
#endif

} // namespace rachis::vulkan
