#pragma once

#include "Common/HandleTable.h"
#include "Internal/VulkanCommands.h"
#include "Internal/VulkanResources.h"
#include "Rachis/AccelerationStructure.h"
#include "Rachis/Device.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <volk.h>

namespace rachis::vulkan
{

    class VulkanAccelerationManager final
    {
    public:
        struct ResolvedAccelerationStructure final
        {
            VkAccelerationStructureKHR accelerationStructure = VK_NULL_HANDLE;
            VkDeviceAddress deviceAddress = 0;
            AccelerationStructureType type = AccelerationStructureType::bottomLevel;
            std::uint64_t size = 0;
            std::shared_ptr<void> keepAlive;
            std::shared_ptr<void> version;
        };

        VulkanAccelerationManager() = default;
        ~VulkanAccelerationManager();

        VulkanAccelerationManager(const VulkanAccelerationManager&) = delete;
        VulkanAccelerationManager& operator=(const VulkanAccelerationManager&) = delete;
        VulkanAccelerationManager(VulkanAccelerationManager&&) = delete;
        VulkanAccelerationManager& operator=(VulkanAccelerationManager&&) = delete;

        void initialize(VkDevice device,
                        const VolkDeviceTable* table,
                        VulkanResourceAllocator* resources,
                        const DeviceCapabilities* capabilities,
                        std::uint32_t scratchAlignment,
                        const VulkanInstanceTable* instanceTable = nullptr) noexcept;
        void shutdown() noexcept;

        [[nodiscard]] Result<AccelerationStructureBuildSizes> queryBottomLevelBuildSizes(
            const BottomLevelAccelerationStructureBuildInput& input) const;
        [[nodiscard]] Result<AccelerationStructureBuildSizes> queryTopLevelBuildSizes(
            const TopLevelAccelerationStructureBuildInput& input) const;
        [[nodiscard]] Result<AccelerationStructureHandle> createAccelerationStructure(
            const AccelerationStructureDesc& desc);
        [[nodiscard]] Status destroyAccelerationStructure(AccelerationStructureHandle accelerationStructure);
        [[nodiscard]] Status buildBottomLevel(VulkanCommandBuffer& commandBuffer,
                                              const BottomLevelAccelerationStructureBuildDesc& desc);
        [[nodiscard]] Status buildTopLevel(VulkanCommandBuffer& commandBuffer,
                                           const TopLevelAccelerationStructureBuildDesc& desc);
        [[nodiscard]] Status updateBottomLevel(VulkanCommandBuffer& commandBuffer,
                                               const BottomLevelAccelerationStructureUpdateDesc& desc);
        [[nodiscard]] Status updateTopLevel(VulkanCommandBuffer& commandBuffer,
                                            const TopLevelAccelerationStructureUpdateDesc& desc);
        [[nodiscard]] Status updateBottomLevelInPlace(VulkanCommandBuffer& commandBuffer,
                                                      const BottomLevelAccelerationStructureInPlaceUpdateDesc& desc);
        [[nodiscard]] Status updateTopLevelInPlace(VulkanCommandBuffer& commandBuffer,
                                                   const TopLevelAccelerationStructureInPlaceUpdateDesc& desc);
        [[nodiscard]] Status compact(VulkanCommandBuffer& commandBuffer,
                                     const AccelerationStructureCompactionDesc& desc);
        [[nodiscard]] Result<ResolvedAccelerationStructure> resolve(
            AccelerationStructureHandle accelerationStructure) const;
#ifdef RACHIS_TESTING
        friend struct VulkanAccelerationTestAccess;
#endif

    private:
        struct AccelerationStructureStorageLease final
        {
            BufferHandle storage{};
            std::uint64_t offset = 0;
            std::uint64_t size = 0;
        };

        struct AccelerationStructureKeepAlive final
        {
            AccelerationStructureKeepAlive() = default;
            AccelerationStructureKeepAlive(const AccelerationStructureKeepAlive&) = delete;
            AccelerationStructureKeepAlive& operator=(const AccelerationStructureKeepAlive&) = delete;
            AccelerationStructureKeepAlive(AccelerationStructureKeepAlive&&) = delete;
            AccelerationStructureKeepAlive& operator=(AccelerationStructureKeepAlive&&) = delete;
            VkDevice device = VK_NULL_HANDLE;
            const VolkDeviceTable* table = nullptr;
            VkAccelerationStructureKHR accelerationStructure = VK_NULL_HANDLE;
            std::shared_ptr<void> storage;
            mutable std::mutex mutex;

            ~AccelerationStructureKeepAlive();
            void releaseNow() noexcept;
        };

        struct AccelerationStructureRecord final
        {
            AccelerationStructureType type = AccelerationStructureType::bottomLevel;
            BufferHandle storage{};
            std::uint64_t offset = 0;
            std::uint64_t size = 0;
            std::string label;
            VkAccelerationStructureKHR accelerationStructure = VK_NULL_HANDLE;
            VkDeviceAddress deviceAddress = 0;
            AccelerationStructureBuildFlag buildFlags = AccelerationStructureBuildFlag::none;
            std::shared_ptr<AccelerationStructureKeepAlive> keepAlive;
            std::shared_ptr<AccelerationStructureStorageLease> storageLease;
            std::shared_ptr<VulkanAccelerationStructureVersionSlot> version =
                std::make_shared<VulkanAccelerationStructureVersionSlot>();
        };

        [[nodiscard]] bool supported() const noexcept;
        [[nodiscard]] Result<AccelerationStructureBuildSizes> queryBuildSizes(
            VkAccelerationStructureTypeKHR type,
            AccelerationStructureBuildFlag flags,
            std::uint32_t geometryCount,
            const VkAccelerationStructureGeometryKHR* geometries,
            const std::uint32_t* primitiveCounts) const;
        [[nodiscard]] Status recordBuildBarrier(VkCommandBuffer command, QueueType queueType) const;
        [[nodiscard]] Result<std::shared_ptr<AccelerationStructureStorageLease>> leaseStorageRange(BufferHandle storage,
                                                                                                   std::uint64_t offset,
                                                                                                   std::uint64_t size);
        [[nodiscard]] Result<VkAccelerationStructureKHR> createNativeAccelerationStructure(
            const AccelerationStructureDesc& desc, VkBuffer storageBuffer, std::uint64_t size) const;
        [[nodiscard]] std::shared_ptr<AccelerationStructureRecord> makeRecord(
            const AccelerationStructureDesc& desc,
            std::uint64_t size,
            VkAccelerationStructureKHR native,
            std::shared_ptr<void> storageKeepAlive,
            std::shared_ptr<AccelerationStructureStorageLease> storageLease) const;
        [[nodiscard]] Result<AccelerationStructureHandle> registerRecord(
            std::shared_ptr<AccelerationStructureRecord> record);
        [[nodiscard]] Status beginFirstBuild(VulkanCommandBuffer& commandBuffer,
                                             AccelerationStructureHandle destination,
                                             AccelerationStructureBuildFlag flags,
                                             std::vector<std::shared_ptr<void>> buildOwners);

        VkDevice device_ = VK_NULL_HANDLE;
        const VolkDeviceTable* table_ = nullptr;
        const VulkanInstanceTable* instanceTable_ = nullptr;
        VulkanResourceAllocator* resources_ = nullptr;
        const DeviceCapabilities* capabilities_ = nullptr;
        std::uint32_t scratchAlignment_ = 256;
        mutable std::mutex mutex_;
        detail::HandleTable<std::shared_ptr<AccelerationStructureRecord>, AccelerationStructureHandle> structures_;
        std::vector<std::weak_ptr<AccelerationStructureStorageLease>> storageLeases_;
        std::vector<std::weak_ptr<AccelerationStructureKeepAlive>> keepAlives_;
        std::size_t nextKeepAlivePruneThreshold_ = 64;
    };

} // namespace rachis::vulkan
