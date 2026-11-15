#include "D3D12/D3D12Diagnostics.h"

#include <dxgi.h>
#include <wrl/client.h>

#include <Windows.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <new>
#include <sstream>
#include <string>
#include <utility>

namespace rachis::d3d12
{
    namespace
    {

        using Microsoft::WRL::ComPtr;

        constexpr std::uint32_t maxBreadcrumbNodes = 32;
        constexpr std::uint32_t operationsBeforeFrontier = 8;
        constexpr std::uint32_t operationsAfterFrontier = 8;
        constexpr std::uint32_t maxAllocationNodesPerList = 32;
        constexpr std::uint32_t maxBreadcrumbContexts = 4'096;

        [[nodiscard]] DeviceFaultStatus deviceFaultStatus(HRESULT reason) noexcept
        {
            switch (reason)
            {
            case S_OK:
                return DeviceFaultStatus::healthy;
            case DXGI_ERROR_DEVICE_REMOVED:
                return DeviceFaultStatus::removed;
            case DXGI_ERROR_DEVICE_RESET:
                return DeviceFaultStatus::reset;
            case DXGI_ERROR_DEVICE_HUNG:
                return DeviceFaultStatus::hung;
            case E_OUTOFMEMORY:
                return DeviceFaultStatus::outOfMemory;
            case DXGI_ERROR_DRIVER_INTERNAL_ERROR:
                return DeviceFaultStatus::driverInternalError;
            case DXGI_ERROR_INVALID_CALL:
                return DeviceFaultStatus::invalidCall;
            default:
                return DeviceFaultStatus::unknown;
            }
        }

        [[nodiscard]] std::string utf8(const wchar_t* value)
        {
            if (value == nullptr)
            {
                return {};
            }
            const int required = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
            if (required <= 1)
            {
                return {};
            }
            std::string result(static_cast<std::size_t>(required), '\0');
            static_cast<void>(WideCharToMultiByte(CP_UTF8, 0, value, -1, result.data(), required, nullptr, nullptr));
            result.pop_back();
            return result;
        }

        [[nodiscard]] std::string objectName(const wchar_t* wide, const char* narrow)
        {
            auto result = utf8(wide);
            if (result.empty() && narrow != nullptr)
            {
                result = narrow;
            }
            return result;
        }

        [[nodiscard]] std::string_view operationName(D3D12_AUTO_BREADCRUMB_OP operation) noexcept
        {
            switch (operation)
            {
            case D3D12_AUTO_BREADCRUMB_OP_SETMARKER:
                return "setMark";
            case D3D12_AUTO_BREADCRUMB_OP_BEGINEVENT:
                return "beginEvent";
            case D3D12_AUTO_BREADCRUMB_OP_ENDEVENT:
                return "endEvent";
            case D3D12_AUTO_BREADCRUMB_OP_DRAWINSTANCED:
                return "drawInstanced";
            case D3D12_AUTO_BREADCRUMB_OP_DRAWINDEXEDINSTANCED:
                return "drawIndexedInstanced";
            case D3D12_AUTO_BREADCRUMB_OP_EXECUTEINDIRECT:
                return "executeIndirect";
            case D3D12_AUTO_BREADCRUMB_OP_DISPATCH:
                return "dispatch";
            case D3D12_AUTO_BREADCRUMB_OP_COPYBUFFERREGION:
                return "copyBufferRegion";
            case D3D12_AUTO_BREADCRUMB_OP_COPYTEXTUREREGION:
                return "copyTextureRegion";
            case D3D12_AUTO_BREADCRUMB_OP_COPYRESOURCE:
                return "copyResource";
            case D3D12_AUTO_BREADCRUMB_OP_COPYTILES:
                return "copyTiles";
            case D3D12_AUTO_BREADCRUMB_OP_RESOLVESUBRESOURCE:
                return "resolveSubresource";
            case D3D12_AUTO_BREADCRUMB_OP_CLEARRENDERTARGETVIEW:
                return "clearRenderTargetView";
            case D3D12_AUTO_BREADCRUMB_OP_CLEARUNORDEREDACCESSVIEW:
                return "clearUnorderedAccessView";
            case D3D12_AUTO_BREADCRUMB_OP_CLEARDEPTHSTENCILVIEW:
                return "clearDepthStencilView";
            case D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER:
                return "resourceBarrier";
            case D3D12_AUTO_BREADCRUMB_OP_EXECUTEBUNDLE:
                return "executeBundle";
            case D3D12_AUTO_BREADCRUMB_OP_PRESENT:
                return "present";
            case D3D12_AUTO_BREADCRUMB_OP_RESOLVEQUERYDATA:
                return "resolveQueryData";
            case D3D12_AUTO_BREADCRUMB_OP_BEGINSUBMISSION:
                return "beginSubmission";
            case D3D12_AUTO_BREADCRUMB_OP_ENDSUBMISSION:
                return "endSubmission";
            case D3D12_AUTO_BREADCRUMB_OP_BUILDRAYTRACINGACCELERATIONSTRUCTURE:
                return "buildRayTracingAccelerationStructure";
            case D3D12_AUTO_BREADCRUMB_OP_EMITRAYTRACINGACCELERATIONSTRUCTUREPOSTBUILDINFO:
                return "emitRayTracingAccelerationStructurePostbuildInfo";
            case D3D12_AUTO_BREADCRUMB_OP_COPYRAYTRACINGACCELERATIONSTRUCTURE:
                return "copyRayTracingAccelerationStructure";
            case D3D12_AUTO_BREADCRUMB_OP_DISPATCHRAYS:
                return "dispatchRays";
            case D3D12_AUTO_BREADCRUMB_OP_SETPIPELINESTATE1:
                return "setPipelineState1";
            case D3D12_AUTO_BREADCRUMB_OP_DISPATCHMESH:
                return "dispatchMesh";
            case D3D12_AUTO_BREADCRUMB_OP_BARRIER:
                return "barrier";
            case D3D12_AUTO_BREADCRUMB_OP_BEGIN_COMMAND_LIST:
                return "beginCommandList";
            default:
                return "unknown";
            }
        }

        [[nodiscard]] std::string_view allocationTypeName(D3D12_DRED_ALLOCATION_TYPE type) noexcept
        {
            switch (type)
            {
            case D3D12_DRED_ALLOCATION_TYPE_COMMAND_QUEUE:
                return "commandQueue";
            case D3D12_DRED_ALLOCATION_TYPE_COMMAND_ALLOCATOR:
                return "commandAllocator";
            case D3D12_DRED_ALLOCATION_TYPE_PIPELINE_STATE:
                return "pipelineState";
            case D3D12_DRED_ALLOCATION_TYPE_COMMAND_LIST:
                return "commandList";
            case D3D12_DRED_ALLOCATION_TYPE_FENCE:
                return "fence";
            case D3D12_DRED_ALLOCATION_TYPE_DESCRIPTOR_HEAP:
                return "descriptorHeap";
            case D3D12_DRED_ALLOCATION_TYPE_HEAP:
                return "heap";
            case D3D12_DRED_ALLOCATION_TYPE_QUERY_HEAP:
                return "queryHeap";
            case D3D12_DRED_ALLOCATION_TYPE_COMMAND_SIGNATURE:
                return "commandSignature";
            case D3D12_DRED_ALLOCATION_TYPE_PIPELINE_LIBRARY:
                return "pipelineLibrary";
            case D3D12_DRED_ALLOCATION_TYPE_RESOURCE:
                return "resource";
            case D3D12_DRED_ALLOCATION_TYPE_STATE_OBJECT:
                return "stateObject";
            case D3D12_DRED_ALLOCATION_TYPE_METACOMMAND:
                return "metaCommand";
            case D3D12_DRED_ALLOCATION_TYPE_INVALID:
                return "invalid";
            default:
                return "unknown";
            }
        }

        [[nodiscard]] std::string breadcrumbContext(const D3D12_AUTO_BREADCRUMB_NODE1& node,
                                                    std::uint32_t operationIndex,
                                                    bool& truncated)
        {
            if (node.pBreadcrumbContexts == nullptr)
            {
                return {};
            }
            const auto contextCount = std::min(node.BreadcrumbContextsCount, maxBreadcrumbContexts);
            truncated = truncated || node.BreadcrumbContextsCount > maxBreadcrumbContexts;
            for (std::uint32_t index = 0; index < contextCount; ++index)
            {
                const auto& context = node.pBreadcrumbContexts[index];
                if (context.BreadcrumbIndex == operationIndex)
                {
                    return utf8(context.pContextString);
                }
            }
            return {};
        }

        void captureBreadcrumbs(const D3D12_AUTO_BREADCRUMB_NODE1* head, DeviceFaultReport& report)
        {
            const auto* node = head;
            std::uint32_t nodeIndex = 0;
            while (node != nullptr && nodeIndex < maxBreadcrumbNodes)
            {
                DeviceFaultBreadcrumb breadcrumb{
                    .queueLabel = objectName(node->pCommandQueueDebugNameW, node->pCommandQueueDebugNameA),
                    .commandListLabel = objectName(node->pCommandListDebugNameW, node->pCommandListDebugNameA),
                    .completedOperationCount =
                        std::min(node->pLastBreadcrumbValue != nullptr ? *node->pLastBreadcrumbValue : 0U,
                                 node->BreadcrumbCount),
                    .totalOperationCount = node->BreadcrumbCount,
                };

                const auto first = breadcrumb.completedOperationCount > operationsBeforeFrontier
                                       ? breadcrumb.completedOperationCount - operationsBeforeFrontier
                                       : 0U;
                const auto remainingOperations = node->BreadcrumbCount - breadcrumb.completedOperationCount;
                const auto end =
                    breadcrumb.completedOperationCount + std::min(remainingOperations, operationsAfterFrontier);
                breadcrumb.firstCapturedOperation = first;
                breadcrumb.operationsTruncated = first != 0 || end != node->BreadcrumbCount;
                if (node->pCommandHistory != nullptr)
                {
                    breadcrumb.operations.reserve(end - first);
                    for (auto operationIndex = first; operationIndex < end; ++operationIndex)
                    {
                        const auto operation = node->pCommandHistory[operationIndex];
                        breadcrumb.operations.push_back(DeviceFaultOperation{
                            .index = operationIndex,
                            .nativeOperation = static_cast<std::uint32_t>(operation),
                            .completed = operationIndex < breadcrumb.completedOperationCount,
                            .name = std::string(operationName(operation)),
                            .context = breadcrumbContext(*node, operationIndex, breadcrumb.contextsTruncated),
                        });
                    }
                }
                else if (node->BreadcrumbCount != 0)
                {
                    breadcrumb.operationsTruncated = true;
                }
                report.breadcrumbs.push_back(std::move(breadcrumb));
                node = node->pNext;
                ++nodeIndex;
            }
            report.breadcrumbsTruncated = node != nullptr;
        }

        void captureAllocations(const D3D12_DRED_ALLOCATION_NODE1* head,
                                DeviceFaultAllocationStatus status,
                                DeviceFaultReport& report)
        {
            const auto* node = head;
            std::uint32_t count = 0;
            while (node != nullptr && count < maxAllocationNodesPerList)
            {
                report.allocations.push_back(DeviceFaultAllocation{
                    .status = status,
                    .nativeType = static_cast<std::uint32_t>(node->AllocationType),
                    .typeName = std::string(allocationTypeName(node->AllocationType)),
                    .label = objectName(node->ObjectNameW, node->ObjectNameA),
                });
                node = node->pNext;
                ++count;
            }
            report.allocationsTruncated = report.allocationsTruncated || node != nullptr;
        }

        [[nodiscard]] std::string_view faultStatusName(DeviceFaultStatus status) noexcept
        {
            constexpr std::array names{
                std::string_view("healthy"),
                std::string_view("removed"),
                std::string_view("reset"),
                std::string_view("hung"),
                std::string_view("outOfMemory"),
                std::string_view("driverInternalError"),
                std::string_view("invalidCall"),
                std::string_view("unknown"),
            };
            const auto index = static_cast<std::size_t>(status);
            return index < names.size() ? names[index] : names.back();
        }

    } // namespace

    Result<DeviceFaultReport> captureDeviceFaultReport(ID3D12Device& device)
    {
        DeviceFaultReport report;
        const HRESULT reason = device.GetDeviceRemovedReason();
        report.status = deviceFaultStatus(reason);
        report.nativeReason = static_cast<std::int64_t>(reason);

        ComPtr<ID3D12DeviceRemovedExtendedData1> dred;
        if (FAILED(device.QueryInterface(IID_PPV_ARGS(&dred))))
        {
            return report;
        }
        report.dredAvailable = true;
        if (report.status == DeviceFaultStatus::healthy)
        {
            return report;
        }

        try
        {
            D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 breadcrumbs{};
            if (SUCCEEDED(dred->GetAutoBreadcrumbsOutput1(&breadcrumbs)))
            {
                captureBreadcrumbs(breadcrumbs.pHeadAutoBreadcrumbNode, report);
            }

            D3D12_DRED_PAGE_FAULT_OUTPUT1 pageFault{};
            if (SUCCEEDED(dred->GetPageFaultAllocationOutput1(&pageFault)))
            {
                report.pageFaultAddress = pageFault.PageFaultVA;
                captureAllocations(
                    pageFault.pHeadExistingAllocationNode, DeviceFaultAllocationStatus::existing, report);
                captureAllocations(
                    pageFault.pHeadRecentFreedAllocationNode, DeviceFaultAllocationStatus::recentlyFreed, report);
            }
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate the owning D3D12 device-fault report.",
            });
        }

        return report;
    }

    void emitDeviceFaultReport(const DeviceFaultReport& report,
                               std::string_view context,
                               DiagnosticCallback callback,
                               void* userData) noexcept
    {
        if (callback == nullptr)
        {
            return;
        }
        try
        {
            std::ostringstream stream;
            stream << "D3D12 device fault during " << context << ": status=" << faultStatusName(report.status)
                   << " reason=0x" << std::hex << static_cast<std::uint32_t>(report.nativeReason)
                   << " DRED=" << std::boolalpha << report.dredAvailable << " pageFault=0x" << std::hex
                   << report.pageFaultAddress << std::dec;
            for (std::size_t breadcrumbIndex = 0; breadcrumbIndex < report.breadcrumbs.size(); ++breadcrumbIndex)
            {
                const auto& breadcrumb = report.breadcrumbs[breadcrumbIndex];
                stream << "\nbreadcrumb[" << breadcrumbIndex << "]: queue='"
                       << (breadcrumb.queueLabel.empty() ? "<unnamed>" : breadcrumb.queueLabel) << "' list='"
                       << (breadcrumb.commandListLabel.empty() ? "<unnamed>" : breadcrumb.commandListLabel)
                       << "' completed=" << breadcrumb.completedOperationCount << '/' << breadcrumb.totalOperationCount
                       << " captured=" << breadcrumb.operations.size();
                if (breadcrumb.operationsTruncated)
                {
                    stream << " (operations truncated)";
                }
                if (breadcrumb.contextsTruncated)
                {
                    stream << " (contexts truncated)";
                }
                for (const auto& operation : breadcrumb.operations)
                {
                    stream << "\n  op[" << operation.index << "] " << (operation.completed ? "completed" : "pending")
                           << ' ' << operation.name;
                    if (!operation.context.empty())
                    {
                        stream << " context='" << operation.context << '\'';
                    }
                }
            }
            for (const auto& allocation : report.allocations)
            {
                stream << "\npage-fault allocation: state="
                       << (allocation.status == DeviceFaultAllocationStatus::existing ? "existing" : "recentlyFreed")
                       << " type=" << allocation.typeName << " label='"
                       << (allocation.label.empty() ? "<unnamed>" : allocation.label) << '\'';
            }
            if (report.breadcrumbsTruncated || report.allocationsTruncated)
            {
                stream << "\ndevice-fault report truncated: breadcrumbs=" << report.breadcrumbsTruncated
                       << " allocations=" << report.allocationsTruncated;
            }
            const auto description = stream.str();
            callback(DiagnosticMessage{DiagnosticSeverity::error, 0, description}, userData);
        }
        catch (...)
        {
            callback(DiagnosticMessage{DiagnosticSeverity::error, 0, "D3D12 device fault report formatting failed."},
                     userData);
        }
    }

} // namespace rachis::d3d12
