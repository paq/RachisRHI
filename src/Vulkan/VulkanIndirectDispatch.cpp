#include "Internal/VulkanIndirectDispatch.h"

#include "Internal/VulkanDiagnostics.h"
#include "RachisCountedDispatchArguments.spirv.h"

#include <algorithm>
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
            std::uint32_t firstRecord = 0;
            std::uint32_t maxCount = 0;
            std::uint32_t padding = 0;
        };

        static_assert(sizeof(CountedDispatchParameters) == 40);

        class RecordBufferHandle final
        {
        public:
            explicit RecordBufferHandle(VulkanResourceAllocator& resources) noexcept : resources_(resources)
            {
            }

            RecordBufferHandle(const RecordBufferHandle&) = delete;
            RecordBufferHandle& operator=(const RecordBufferHandle&) = delete;
            RecordBufferHandle(RecordBufferHandle&&) = delete;
            RecordBufferHandle& operator=(RecordBufferHandle&&) = delete;

            ~RecordBufferHandle()
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
        if (device_ != VK_NULL_HANDLE && table_ != nullptr)
        {
            if (pipeline_ != VK_NULL_HANDLE && table_->vkDestroyPipeline != nullptr)
            {
                table_->vkDestroyPipeline(device_, pipeline_, nullptr);
            }
            if (layout_ != VK_NULL_HANDLE && table_->vkDestroyPipelineLayout != nullptr)
            {
                table_->vkDestroyPipelineLayout(device_, layout_, nullptr);
            }
            if (module_ != VK_NULL_HANDLE && table_->vkDestroyShaderModule != nullptr)
            {
                table_->vkDestroyShaderModule(device_, module_, nullptr);
            }
        }
        pipeline_ = VK_NULL_HANDLE;
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

    Status VulkanCountedDispatchPass::prepare()
    {
        const std::scoped_lock lock(mutex_);
        if (pipeline_ != VK_NULL_HANDLE)
        {
            return {};
        }
        if (device_ == VK_NULL_HANDLE || table_ == nullptr || table_->vkCreateShaderModule == nullptr ||
            table_->vkCreatePipelineLayout == nullptr || table_->vkCreateComputePipelines == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "GPU-counted indirect dispatch is unavailable because the Vulkan device table is incomplete.",
            });
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

        const VkPushConstantRange range{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(CountedDispatchParameters),
        };
        const VkPipelineLayoutCreateInfo layoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = {},
            .flags = {},
            .setLayoutCount = 0,
            .pSetLayouts = nullptr,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        };
        result = table_->vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &layout_);
        if (result != VK_SUCCESS)
        {
            layout_ = VK_NULL_HANDLE;
            destroyLocked();
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(result), result, "vkCreatePipelineLayout"));
        }

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
                    .pSpecializationInfo = nullptr,
                },
            .layout = layout_,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        };
        result = table_->vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_);
        if (result != VK_SUCCESS)
        {
            pipeline_ = VK_NULL_HANDLE;
            destroyLocked();
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(result), result, "vkCreateComputePipelines"));
        }
        return {};
    }

    Result<VulkanCountedDispatchRecords> VulkanCountedDispatchPass::allocateRecords(std::uint32_t maxCount)
    {
        VulkanResourceAllocator* resources = nullptr;
        {
            const std::scoped_lock lock(mutex_);
            resources = resources_;
        }
        if (resources == nullptr || maxCount == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "GPU-counted indirect dispatch records require live resources and a non-zero record count.",
            });
        }
        auto created = resources->createBuffer(BufferDesc{
            .size = static_cast<std::uint64_t>(maxCount) * countedDispatchRecordSize,
            .usage = BufferUsage::indirect,
            .memoryType = MemoryType::deviceLocal,
            .label = "Vulkan GPU-counted dispatch records",
        });
        if (!created)
        {
            return std::unexpected(std::move(created.error()));
        }
        RecordBufferHandle handle(*resources);
        handle.adopt(*created);
        auto resolved = resources->resolveBindingBuffer(*created);
        if (!resolved)
        {
            return std::unexpected(std::move(resolved.error()));
        }
        if (resolved->deviceAddress == 0)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "GPU-counted indirect dispatch records have no Vulkan device address.",
            });
        }
        return VulkanCountedDispatchRecords{
            .buffer = resolved->buffer,
            .address = resolved->deviceAddress,
            .keepAlive = std::move(resolved->keepAlive),
        };
    }

    void VulkanCountedDispatchPass::record(VkCommandBuffer command,
                                           const VulkanCountedDispatchSource& source,
                                           const VulkanCountedDispatchRecords& records) const noexcept
    {
        const VolkDeviceTable* table = nullptr;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        std::uint32_t maxWorkgroupCountX = 1;
        bool allCommandsSource = false;
        {
            const std::scoped_lock lock(mutex_);
            table = table_;
            pipeline = pipeline_;
            layout = layout_;
            maxWorkgroupCountX = maxWorkgroupCountX_;
            allCommandsSource = resources_ != nullptr && resources_->waitsOnAllCommandsBeforeBarriers();
        }
        if (table == nullptr || pipeline == VK_NULL_HANDLE || source.maxCount == 0)
        {
            return;
        }

        const VkMemoryBarrier2 sourceBarrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .pNext = {},
            .srcStageMask =
                allCommandsSource ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
            .srcAccessMask = VK_ACCESS_2_NONE,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        };
        const VkDependencyInfo sourceDependency{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = {},
            .dependencyFlags = {},
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &sourceBarrier,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 0,
            .pImageMemoryBarriers = nullptr,
        };
        table->vkCmdPipelineBarrier2(command, &sourceDependency);

        table->vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        const std::uint64_t recordsPerDispatch =
            static_cast<std::uint64_t>(maxWorkgroupCountX) * countedDispatchWorkgroupSize;
        for (std::uint64_t first = 0; first < source.maxCount; first += recordsPerDispatch)
        {
            const auto remaining = static_cast<std::uint64_t>(source.maxCount) - first;
            const auto chunk = std::min(remaining, recordsPerDispatch);
            const CountedDispatchParameters parameters{
                .arguments = source.arguments,
                .count = source.count,
                .records = records.address,
                .stride = source.stride,
                .firstRecord = static_cast<std::uint32_t>(first),
                .maxCount = source.maxCount,
                .padding = 0,
            };
            table->vkCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(parameters), &parameters);
            table->vkCmdDispatch(
                command,
                static_cast<std::uint32_t>((chunk + countedDispatchWorkgroupSize - 1U) / countedDispatchWorkgroupSize),
                1,
                1);
        }

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
