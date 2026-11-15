#include "Internal/VulkanFactory.h"

#include "Internal/DeviceIdentity.h"
#include "Internal/VulkanBackendConfiguration.h"
#include "Internal/VulkanDevice.h"
#include "Internal/VulkanDiagnostics.h"
#include "Internal/VulkanLinuxProfile.h"
#include "Internal/VulkanRuntime.h"

#include <algorithm>
#include <new>
#include <utility>

namespace rachis::vulkan
{
    namespace
    {

        [[nodiscard]] Error selectionError(VulkanSelectionFailure failure, std::string message)
        {
            return Error{vulkanSelectionError(failure), 0, std::move(message)};
        }

        void nameCreatedDevice(const VulkanInstanceTable* instanceTable,
                               VkDevice device,
                               const VulkanPhysicalDeviceRecord& record,
                               const VulkanQueueHandles& queues)
        {
            setVulkanObjectName(instanceTable,
                                device,
                                VK_OBJECT_TYPE_DEVICE,
                                reinterpret_cast<std::uint64_t>(device),
                                "RHI.RHI.Vulkan");
            if (queues.graphics != VK_NULL_HANDLE)
            {
                setVulkanObjectName(instanceTable,
                                    device,
                                    VK_OBJECT_TYPE_QUEUE,
                                    reinterpret_cast<std::uint64_t>(queues.graphics),
                                    "RHI.graphics");
            }
            if (queues.compute != VK_NULL_HANDLE)
            {
                setVulkanObjectName(instanceTable,
                                    device,
                                    VK_OBJECT_TYPE_QUEUE,
                                    reinterpret_cast<std::uint64_t>(queues.compute),
                                    "RHI.compute");
            }
            if (queues.copy != VK_NULL_HANDLE)
            {
                setVulkanObjectName(instanceTable,
                                    device,
                                    VK_OBJECT_TYPE_QUEUE,
                                    reinterpret_cast<std::uint64_t>(queues.copy),
                                    "RHI.copy");
            }
            static_cast<void>(record);
        }

    } // namespace

    Result<std::vector<AdapterInfo>> enumerateAdapters()
    {
        auto instance = createVulkanInstance(DeviceDesc{
            .backend = Backend::vulkan,
            .validation = ValidationLevel::disabled,
            .requireValidation = false,
            .enableDred = false,
        });
        if (!instance)
        {
            return std::unexpected(std::move(instance.error()));
        }

        auto records = discoverVulkanPhysicalDevices(instance->instance, instance->table, instance->report);
        if (!records)
        {
            return std::unexpected(std::move(records.error()));
        }

        std::vector<AdapterInfo> adapters;
        adapters.reserve(records->size());
        for (auto& record : *records)
        {
            if (record.report.translationLayer)
            {
                continue;
            }
            adapters.push_back(std::move(record.adapter));
        }
        return adapters;
    }

    Result<std::unique_ptr<detail::BackendDevice>> createDevice(const DeviceDesc& desc)
    {
        if (desc.adapterPreference == AdapterPreference::warp)
        {
            return std::unexpected(selectionError(VulkanSelectionFailure::warpIsD3D12Only,
                                                  "AdapterPreference::warp is D3D12-only and is not a Vulkan ICD."));
        }
        if (desc.enableDred)
        {
            return std::unexpected(selectionError(
                VulkanSelectionFailure::dredIsD3D12Only,
                "DeviceDesc::enableDred is D3D12-only. Use validation layers or optional device-fault."));
        }

        auto instance = createVulkanInstance(desc);
        if (!instance)
        {
            return std::unexpected(std::move(instance.error()));
        }

        auto records = discoverVulkanPhysicalDevices(instance->instance, instance->table, instance->report);
        if (!records)
        {
            return std::unexpected(std::move(records.error()));
        }
        reportVulkanAdapterIdCollisions(*records, *instance->sink);

        if (desc.adapterId.has_value())
        {
            const auto found =
                std::find_if(records->begin(), records->end(), [&](const VulkanPhysicalDeviceRecord& record) {
                    return record.adapter.id == *desc.adapterId && !record.report.translationLayer;
                });
            if (found == records->end())
            {
                return std::unexpected(selectionError(VulkanSelectionFailure::adapterNotFound,
                                                      "The requested adapter is not in the Vulkan enumerate list."));
            }
        }

        const auto* selected = selectVulkanPhysicalDevice(*records, instance->report, desc);
        if (selected == nullptr)
        {
            VulkanProfileDecision rejection{};
            const auto intent = VulkanCreateIntent{
                .policy = VulkanAdapterSelectionPolicy::hardwarePreferred,
                .warpPreference = false,
            };
            bool sawNativeAdapter = false;
            for (const auto& record : *records)
            {
                if (record.report.translationLayer)
                {
                    continue;
                }
                sawNativeAdapter = true;
                rejection = evaluateNativeVulkanProfile(instance->report, record.report, intent);
                break;
            }
            if (!sawNativeAdapter)
            {
                return std::unexpected(selectionError(VulkanSelectionFailure::loaderUnavailable,
                                                      "No native Vulkan adapter is available. Selection did not "
                                                      "fall back to D3D12."));
            }
            if (rejection.rejection == VulkanProfileRejection::none)
            {
                return std::unexpected(selectionError(VulkanSelectionFailure::requiredFeatureMissing,
                                                      "No Vulkan adapter passed the host profile. Selection did not "
                                                      "fall back to D3D12."));
            }
            return std::unexpected(selectionError(VulkanSelectionFailure::requiredFeatureMissing,
                                                  describeNativeVulkanProfileRejection(rejection)));
        }

        if (desc.adapterId.has_value())
        {
            const auto intent = VulkanCreateIntent{
                .policy = VulkanAdapterSelectionPolicy::hardwarePreferred,
                .warpPreference = false,
            };
            const auto decision = evaluateNativeVulkanProfile(instance->report, selected->report, intent);
            if (!decision.accepted)
            {
                return std::unexpected(selectionError(VulkanSelectionFailure::requiredFeatureMissing,
                                                      describeNativeVulkanProfileRejection(decision)));
            }
        }

        VulkanQueueHandles queues{};
        auto device =
            createVulkanLogicalDevice(instance->table, *selected, instance->report, queues, desc.enableDeviceFault);
        if (!device)
        {
            return std::unexpected(std::move(device.error()));
        }

        VolkDeviceTable table{};
        loadVulkanDeviceTable(table, *device, instance->table.vkGetDeviceProcAddr);
        const auto token = detail::deviceTokenSource().acquire();
        if (!token.has_value())
        {
            if (table.vkDestroyDevice != nullptr)
            {
                table.vkDestroyDevice(*device, nullptr);
            }
            return std::unexpected(Error{
                ErrorCode::resourceExhausted,
                0,
                "The Vulkan device identity space is exhausted and cannot be reused safely.",
            });
        }
        nameCreatedDevice(&instance->table, *device, *selected, queues);
        auto capabilities = publishVulkanCapabilities(instance->table, *selected, instance->debugUtils);
        auto backend = std::unique_ptr<VulkanDevice>(
            new (std::nothrow) VulkanDevice(*selected, *device, table, queues, capabilities, *token));
        if (backend == nullptr)
        {
            if (table.vkDestroyDevice != nullptr)
            {
                table.vkDestroyDevice(*device, nullptr);
            }
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate the Vulkan device backend.",
            });
        }
        backend->adoptInstance(std::move(*instance));
        if (auto initialized = backend->initializeResources(); !initialized)
        {
            return std::unexpected(std::move(initialized.error()));
        }
        return std::unique_ptr<detail::BackendDevice>(std::move(backend));
    }

} // namespace rachis::vulkan
