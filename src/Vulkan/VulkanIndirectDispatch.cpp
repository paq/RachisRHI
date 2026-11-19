#include "Internal/VulkanIndirectDispatch.h"

#include "Internal/VulkanDiagnostics.h"
#include "RachisCountedDispatchArguments.spirv.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <new>
#include <vector>

namespace rachis::vulkan
{

    namespace
    {

        struct CountedDispatchParameters final
        {
            VkDeviceAddress arguments = 0;
            VkDeviceAddress count = 0;
            VkDeviceAddress records = 0;
            std::uint32_t stride = 0;
            std::uint32_t maxCount = 0;
            std::uint32_t groupsPerRow = 0;
            std::array<std::uint32_t, 3> padding{};
        };

        static_assert(sizeof(CountedDispatchParameters) == 48);
        static_assert(offsetof(CountedDispatchParameters, records) == 16);
        static_assert(offsetof(CountedDispatchParameters, groupsPerRow) == 32);

        struct ParameterAddress final
        {
            std::uint32_t low = 0;
            std::uint32_t high = 0;
        };

        class TransientBufferHandle final
        {
        public:
            explicit TransientBufferHandle(VulkanResourceAllocator& resources) noexcept : resources_(resources)
            {
            }

            TransientBufferHandle(const TransientBufferHandle&) = delete;
            TransientBufferHandle& operator=(const TransientBufferHandle&) = delete;
            TransientBufferHandle(TransientBufferHandle&&) = delete;
            TransientBufferHandle& operator=(TransientBufferHandle&&) = delete;

            ~TransientBufferHandle()
            {
                if (handle_.isValid())
                {
                    static_cast<void>(resources_.destroyBuffer(handle_));
                }
            }

            void adopt(BufferHandle handle) noexcept
            {
                handle_ = handle;
            }

        private:
            VulkanResourceAllocator& resources_;
            BufferHandle handle_{};
        };

        [[nodiscard]] Result<VulkanResourceAllocator::ResolvedBindingBuffer> createAddressableBuffer(
            VulkanResourceAllocator& resources, const BufferDesc& desc, const char* unaddressable)
        {
            auto created = resources.createBuffer(desc);
            if (!created)
            {
                return std::unexpected(std::move(created.error()));
            }
            TransientBufferHandle handle(resources);
            handle.adopt(*created);
            auto resolved = resources.resolveBindingBuffer(*created);
            if (!resolved)
            {
                return std::unexpected(std::move(resolved.error()));
            }
            if (resolved->deviceAddress == 0 || resolved->deviceAddress % sizeof(std::uint32_t) != 0)
            {
                return std::unexpected(Error{ErrorCode::unsupported, 0, unaddressable});
            }
            return resolved;
        }

    } // namespace

    VulkanCountedDispatchPass::~VulkanCountedDispatchPass()
    {
        shutdown();
    }

    void VulkanCountedDispatchPass::initialize(VkDevice device,
                                               const VolkDeviceTable* table,
                                               VulkanResourceAllocator* resources,
                                               std::uint32_t maxWorkgroupCountX) noexcept
    {
        const std::scoped_lock lock(mutex_);
        device_ = device;
        table_ = table;
        resources_ = resources;
        maxWorkgroupCountX_ = std::max(maxWorkgroupCountX, 1U);
    }

    void VulkanCountedDispatchPass::destroyLocked() noexcept
    {
        for (auto& queue : queues_)
        {
            if (device_ != VK_NULL_HANDLE && table_ != nullptr && queue.pipeline != VK_NULL_HANDLE &&
                table_->vkDestroyPipeline != nullptr)
            {
                table_->vkDestroyPipeline(device_, queue.pipeline, nullptr);
            }
            queue = QueueParameters{};
        }
        if (device_ != VK_NULL_HANDLE && table_ != nullptr)
        {
            if (layout_ != VK_NULL_HANDLE && table_->vkDestroyPipelineLayout != nullptr)
            {
                table_->vkDestroyPipelineLayout(device_, layout_, nullptr);
            }
            if (module_ != VK_NULL_HANDLE && table_->vkDestroyShaderModule != nullptr)
            {
                table_->vkDestroyShaderModule(device_, module_, nullptr);
            }
        }
        layout_ = VK_NULL_HANDLE;
        module_ = VK_NULL_HANDLE;
    }

    void VulkanCountedDispatchPass::shutdown() noexcept
    {
        const std::scoped_lock lock(mutex_);
        destroyLocked();
        device_ = VK_NULL_HANDLE;
        table_ = nullptr;
        resources_ = nullptr;
    }

    std::size_t VulkanCountedDispatchPass::queueSlot(QueueType queueType) noexcept
    {
        return queueType == QueueType::compute ? 1U : 0U;
    }

    Status VulkanCountedDispatchPass::prepareSharedLocked()
    {
        if (layout_ != VK_NULL_HANDLE)
        {
            return {};
        }
        std::vector<std::uint32_t> code;
        try
        {
            code.resize((sizeof(g_rachisCountedDispatchArgumentsSpirv) + sizeof(std::uint32_t) - 1U) /
                        sizeof(std::uint32_t));
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "The GPU-counted indirect dispatch shader could not be staged.",
            });
        }
        std::memcpy(code.data(), g_rachisCountedDispatchArgumentsSpirv, sizeof(g_rachisCountedDispatchArgumentsSpirv));

        const VkShaderModuleCreateInfo moduleInfo{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = {},
            .flags = {},
            .codeSize = sizeof(g_rachisCountedDispatchArgumentsSpirv),
            .pCode = code.data(),
        };
        auto result = table_->vkCreateShaderModule(device_, &moduleInfo, nullptr, &module_);
        if (result != VK_SUCCESS)
        {
            module_ = VK_NULL_HANDLE;
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(result), result, "vkCreateShaderModule"));
        }

        const VkPipelineLayoutCreateInfo layoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = {},
            .flags = {},
            .setLayoutCount = 0,
            .pSetLayouts = nullptr,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = nullptr,
        };
        result = table_->vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &layout_);
        if (result != VK_SUCCESS)
        {
            layout_ = VK_NULL_HANDLE;
            destroyLocked();
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(result), result, "vkCreatePipelineLayout"));
        }
        return {};
    }

    Status VulkanCountedDispatchPass::prepareQueueLocked(QueueParameters& parameters)
    {
        if (parameters.pipeline != VK_NULL_HANDLE)
        {
            return {};
        }
        if (parameters.keepAlive == nullptr)
        {
            auto buffer = createAddressableBuffer(*resources_,
                                                  BufferDesc{
                                                      .size = sizeof(CountedDispatchParameters),
                                                      .usage = BufferUsage::indirect | BufferUsage::copyDestination,
                                                      .memoryType = MemoryType::deviceLocal,
                                                      .label = "Vulkan GPU-counted dispatch parameters",
                                                  },
                                                  "GPU-counted indirect dispatch parameters have no 4-byte aligned "
                                                  "device address.");
            if (!buffer)
            {
                return std::unexpected(std::move(buffer.error()));
            }
            parameters.buffer = buffer->buffer;
            parameters.address = buffer->deviceAddress;
            parameters.keepAlive = std::move(buffer->keepAlive);
        }

        const ParameterAddress address{
            .low = static_cast<std::uint32_t>(parameters.address),
            .high = static_cast<std::uint32_t>(parameters.address >> 32U),
        };
        const std::array entries{
            VkSpecializationMapEntry{
                .constantID = 0, .offset = offsetof(ParameterAddress, low), .size = sizeof(std::uint32_t)},
            VkSpecializationMapEntry{
                .constantID = 1, .offset = offsetof(ParameterAddress, high), .size = sizeof(std::uint32_t)},
        };
        const VkSpecializationInfo specialization{
            .mapEntryCount = static_cast<std::uint32_t>(entries.size()),
            .pMapEntries = entries.data(),
            .dataSize = sizeof(address),
            .pData = &address,
        };
        const VkComputePipelineCreateInfo pipelineInfo{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = {},
            .flags = {},
            .stage =
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = {},
                    .flags = {},
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module = module_,
                    .pName = "main",
                    .pSpecializationInfo = &specialization,
                },
            .layout = layout_,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        };
        VkPipeline pipeline = VK_NULL_HANDLE;
        const auto result =
            table_->vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
        if (result != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(result), result, "vkCreateComputePipelines"));
        }
        parameters.pipeline = pipeline;
        return {};
    }

    Status VulkanCountedDispatchPass::prepare(QueueType queueType)
    {
        const std::scoped_lock lock(mutex_);
        if (queueType != QueueType::graphics && queueType != QueueType::compute)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "GPU-counted indirect dispatch requires a graphics or compute command buffer.",
            });
        }
        auto& parameters = queues_[queueSlot(queueType)];
        if (parameters.pipeline != VK_NULL_HANDLE)
        {
            return {};
        }
        if (device_ == VK_NULL_HANDLE || table_ == nullptr || resources_ == nullptr ||
            table_->vkCreateShaderModule == nullptr || table_->vkCreatePipelineLayout == nullptr ||
            table_->vkCreateComputePipelines == nullptr || table_->vkCmdUpdateBuffer == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "GPU-counted indirect dispatch is unavailable because the Vulkan device table is incomplete.",
            });
        }
        if (auto shared = prepareSharedLocked(); !shared)
        {
            return shared;
        }
        return prepareQueueLocked(parameters);
    }

    Result<VulkanCountedDispatchRecords> VulkanCountedDispatchPass::allocateRecords(QueueType queueType,
                                                                                    std::uint32_t maxCount)
    {
        VulkanResourceAllocator* resources = nullptr;
        std::shared_ptr<void> parametersKeepAlive;
        {
            const std::scoped_lock lock(mutex_);
            resources = resources_;
            parametersKeepAlive = queues_[queueSlot(queueType)].keepAlive;
        }
        if (resources == nullptr || parametersKeepAlive == nullptr || maxCount == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "GPU-counted indirect dispatch records require a prepared pass, live resources, and a non-zero "
                "record count.",
            });
        }
        auto records =
            createAddressableBuffer(*resources,
                                    BufferDesc{
                                        .size = static_cast<std::uint64_t>(maxCount) * countedDispatchRecordSize,
                                        .usage = BufferUsage::indirect,
                                        .memoryType = MemoryType::deviceLocal,
                                        .label = "Vulkan GPU-counted dispatch records",
                                    },
                                    "GPU-counted indirect dispatch records have no 4-byte aligned Vulkan "
                                    "device address.");
        if (!records)
        {
            return std::unexpected(std::move(records.error()));
        }
        return VulkanCountedDispatchRecords{
            .buffer = records->buffer,
            .address = records->deviceAddress,
            .keepAlive = std::move(records->keepAlive),
            .parametersKeepAlive = std::move(parametersKeepAlive),
        };
    }

    void VulkanCountedDispatchPass::record(VkCommandBuffer command,
                                           QueueType queueType,
                                           const VulkanCountedDispatchSource& source,
                                           const VulkanCountedDispatchRecords& records) const noexcept
    {
        const VolkDeviceTable* table = nullptr;
        QueueParameters parameters{};
        std::uint32_t maxWorkgroupCountX = 1;
        bool allCommandsSource = false;
        {
            const std::scoped_lock lock(mutex_);
            table = table_;
            const auto& prepared = queues_[queueSlot(queueType)];
            parameters.buffer = prepared.buffer;
            parameters.pipeline = prepared.pipeline;
            maxWorkgroupCountX = maxWorkgroupCountX_;
            allCommandsSource = resources_ != nullptr && resources_->waitsOnAllCommandsBeforeBarriers();
        }
        if (table == nullptr || parameters.pipeline == VK_NULL_HANDLE || parameters.buffer == VK_NULL_HANDLE ||
            source.maxCount == 0)
        {
            return;
        }

        const std::uint64_t groups = (static_cast<std::uint64_t>(source.maxCount) + countedDispatchWorkgroupSize - 1U) /
                                     countedDispatchWorkgroupSize;
        const auto groupsPerRow = static_cast<std::uint32_t>(std::min<std::uint64_t>(groups, maxWorkgroupCountX));
        const auto rows = static_cast<std::uint32_t>((groups + groupsPerRow - 1U) / groupsPerRow);
        const CountedDispatchParameters values{
            .arguments = source.arguments,
            .count = source.count,
            .records = records.address,
            .stride = source.stride,
            .maxCount = source.maxCount,
            .groupsPerRow = groupsPerRow,
        };

        const std::array sourceBarriers{
            VkMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                .pNext = {},
                .srcStageMask =
                    allCommandsSource ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                .srcAccessMask = VK_ACCESS_2_NONE,
                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            },
            VkMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                .pNext = {},
                .srcStageMask =
                    allCommandsSource ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_NONE,
                .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            },
        };
        const VkDependencyInfo sourceDependency{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = {},
            .dependencyFlags = {},
            .memoryBarrierCount = static_cast<std::uint32_t>(sourceBarriers.size()),
            .pMemoryBarriers = sourceBarriers.data(),
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 0,
            .pImageMemoryBarriers = nullptr,
        };
        table->vkCmdPipelineBarrier2(command, &sourceDependency);
        table->vkCmdUpdateBuffer(command, parameters.buffer, 0, sizeof(values), &values);

        const VkBufferMemoryBarrier2 parametersBarrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .pNext = {},
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = parameters.buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        };
        const VkDependencyInfo parametersDependency{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = {},
            .dependencyFlags = {},
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &parametersBarrier,
            .imageMemoryBarrierCount = 0,
            .pImageMemoryBarriers = nullptr,
        };
        table->vkCmdPipelineBarrier2(command, &parametersDependency);

        table->vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, parameters.pipeline);
        table->vkCmdDispatch(command, groupsPerRow, rows, 1);

        const VkBufferMemoryBarrier2 recordsBarrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .pNext = {},
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
            .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = records.buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        };
        const VkDependencyInfo recordsDependency{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = {},
            .dependencyFlags = {},
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &recordsBarrier,
            .imageMemoryBarrierCount = 0,
            .pImageMemoryBarriers = nullptr,
        };
        table->vkCmdPipelineBarrier2(command, &recordsDependency);
    }

} // namespace rachis::vulkan
