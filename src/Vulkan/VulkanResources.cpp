#define VMA_IMPLEMENTATION
#ifndef VMA_STATIC_VULKAN_FUNCTIONS
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#endif
#ifndef VMA_DYNAMIC_VULKAN_FUNCTIONS
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#endif

#include "Internal/VulkanResources.h"

#include "Internal/CommandBufferAccess.h"
#include "Internal/TextureFormatRules.h"
#include "Internal/VulkanCommands.h"
#include "Internal/VulkanDiagnostics.h"
#include "Internal/VulkanDispatch.h"
#include "Internal/VulkanRuntime.h"

#ifdef RACHIS_TESTING
#include "Vulkan/VulkanTestHooks.h"
#endif

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#ifdef DeviceCapabilities
#undef DeviceCapabilities
#endif
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

namespace rachis::vulkan
{

    VulkanNativeKeepAlive::~VulkanNativeKeepAlive()
    {
        releaseNow();
#ifdef RACHIS_TESTING
        test::liveNativeKeepAlives().fetch_sub(1, std::memory_order_relaxed);
#endif
    }

    void VulkanExportedMemoryTotals::add(std::uint32_t heapIndex, std::uint64_t size) noexcept
    {
        if (heapIndex >= bytes.size())
        {
            return;
        }
        bytes[heapIndex].fetch_add(size, std::memory_order_relaxed);
        allocations[heapIndex].fetch_add(1, std::memory_order_relaxed);
    }

    void VulkanExportedMemoryTotals::remove(std::uint32_t heapIndex, std::uint64_t size) noexcept
    {
        if (heapIndex >= bytes.size())
        {
            return;
        }
        bytes[heapIndex].fetch_sub(size, std::memory_order_relaxed);
        allocations[heapIndex].fetch_sub(1, std::memory_order_relaxed);
    }

    void VulkanNativeKeepAlive::releaseNow() noexcept
    {
        std::shared_ptr<VulkanNativeKeepAlive> releasedParent;
        std::shared_ptr<PlacedResourceLease> releasedPlacement;
        {
            const std::scoped_lock lock(mutex);
            const auto freeDedicatedMemory = [this]() noexcept {
                if (memory == VK_NULL_HANDLE || device == VK_NULL_HANDLE || table == nullptr ||
                    table->vkFreeMemory == nullptr || allocation != VK_NULL_HANDLE)
                {
                    return;
                }
                table->vkFreeMemory(device, memory, nullptr);
                if (exportedTotals != nullptr)
                {
                    exportedTotals->remove(exportedHeapIndex, exportedBytes);
                }
            };
            switch (kind)
            {
            case Kind::buffer:
                if (allocation != VK_NULL_HANDLE && allocator != VK_NULL_HANDLE && buffer != VK_NULL_HANDLE)
                {
                    vmaDestroyBuffer(allocator, buffer, allocation);
                }
                else if (device != VK_NULL_HANDLE && table != nullptr && table->vkDestroyBuffer != nullptr &&
                         buffer != VK_NULL_HANDLE)
                {
                    table->vkDestroyBuffer(device, buffer, nullptr);
                }
                freeDedicatedMemory();
                break;
            case Kind::image:
                if (allocation != VK_NULL_HANDLE && allocator != VK_NULL_HANDLE && image != VK_NULL_HANDLE)
                {
                    vmaDestroyImage(allocator, image, allocation);
                }
                else if (device != VK_NULL_HANDLE && table != nullptr && table->vkDestroyImage != nullptr &&
                         image != VK_NULL_HANDLE)
                {
                    table->vkDestroyImage(device, image, nullptr);
                }
                freeDedicatedMemory();
                break;
            case Kind::view:
                if (device != VK_NULL_HANDLE && table != nullptr && table->vkDestroyImageView != nullptr &&
                    view != VK_NULL_HANDLE)
                {
                    table->vkDestroyImageView(device, view, nullptr);
                }
                break;
            case Kind::sampler:
                if (device != VK_NULL_HANDLE && table != nullptr && table->vkDestroySampler != nullptr &&
                    sampler != VK_NULL_HANDLE)
                {
                    table->vkDestroySampler(device, sampler, nullptr);
                }
                break;
            }
            allocator = VK_NULL_HANDLE;
            device = VK_NULL_HANDLE;
            table = nullptr;
            buffer = VK_NULL_HANDLE;
            image = VK_NULL_HANDLE;
            view = VK_NULL_HANDLE;
            sampler = VK_NULL_HANDLE;
            allocation = VK_NULL_HANDLE;
            memory = VK_NULL_HANDLE;
            exportedTotals.reset();
            releasedPlacement = std::move(placement);
            releasedParent = std::move(parent);
        }
    }

#ifdef RACHIS_TESTING
    namespace
    {

        thread_local std::array<bool, 3> injectedBadAllocs{};

        void throwInjectedBadAlloc(VulkanBadAllocPoint point)
        {
            if (std::exchange(injectedBadAllocs[static_cast<std::size_t>(point)], false))
            {
                throw std::bad_alloc();
            }
        }

    } // namespace

    void injectVulkanBadAlloc(VulkanBadAllocPoint point, bool armed) noexcept
    {
        injectedBadAllocs[static_cast<std::size_t>(point)] = armed;
    }
#endif

    std::shared_ptr<VulkanNativeKeepAlive> makeVulkanNativeKeepAlive(VulkanNativeKeepAlive::Kind kind,
                                                                     VkDevice device,
                                                                     const VolkDeviceTable* table)
    {
#ifdef RACHIS_TESTING
        throwInjectedBadAlloc(VulkanBadAllocPoint::nativeKeepAlive);
#endif
        auto keepAlive = std::make_shared<VulkanNativeKeepAlive>();
#ifdef RACHIS_TESTING
        test::liveNativeKeepAlives().fetch_add(1, std::memory_order_relaxed);
#endif
        keepAlive->kind = kind;
        keepAlive->device = device;
        keepAlive->table = table;
        return keepAlive;
    }

    ResourceHeapRecord::~ResourceHeapRecord()
    {
        releaseNow();
    }

    void ResourceHeapRecord::releaseNow() noexcept
    {
        const std::scoped_lock lock(mutex);
        alive = false;
        if (mappedBase != nullptr && device != VK_NULL_HANDLE && table != nullptr && table->vkUnmapMemory != nullptr &&
            memory != VK_NULL_HANDLE)
        {
            table->vkUnmapMemory(device, memory);
            mappedBase = nullptr;
        }
        if (device != VK_NULL_HANDLE && table != nullptr && table->vkFreeMemory != nullptr && memory != VK_NULL_HANDLE)
        {
            table->vkFreeMemory(device, memory, nullptr);
        }
        memory = VK_NULL_HANDLE;
        device = VK_NULL_HANDLE;
        table = nullptr;
    }

    namespace
    {

#ifdef _WIN32
        struct SharedExportIdentity final
        {
            HANDLE handle = nullptr;
            bool texture = false;
            BufferDesc buffer{};
            TextureDesc textureDesc{};
            std::vector<Format> textureViewFormats{};
            std::array<std::uint8_t, VK_UUID_SIZE> deviceUuid{};
            std::uint32_t memoryTypeIndex = 0;
            std::uint64_t allocationBytes = 0;
        };

        std::mutex sharedExportMutex;
        std::vector<SharedExportIdentity> sharedExports;
#endif

        void closeExportIdentity(void*& identity) noexcept
        {
#ifdef _WIN32
            if (identity == nullptr)
            {
                return;
            }
            auto* const handle = static_cast<HANDLE>(identity);
            const std::scoped_lock lock(sharedExportMutex);
            std::erase_if(sharedExports, [&](const SharedExportIdentity& entry) { return entry.handle == handle; });
            static_cast<void>(CloseHandle(handle));
            identity = nullptr;
#else
            identity = nullptr;
#endif
        }

#ifdef _WIN32
        using CompareObjectHandlesFn = BOOL(WINAPI*)(HANDLE, HANDLE);

        [[nodiscard]] bool sameKernelObject(HANDLE left, HANDLE right) noexcept
        {
            if (left == nullptr || right == nullptr)
            {
                return false;
            }
            if (left == right)
            {
                return true;
            }
            static const auto s_compare = []() -> CompareObjectHandlesFn {
                auto* module = GetModuleHandleW(L"KernelBase.dll");
                if (module == nullptr)
                {
                    module = LoadLibraryW(L"KernelBase.dll");
                }
                if (module == nullptr)
                {
                    return nullptr;
                }
                return reinterpret_cast<CompareObjectHandlesFn>(GetProcAddress(module, "CompareObjectHandles"));
            }();
            return s_compare != nullptr && s_compare(left, right) != FALSE;
        }

        [[nodiscard]] const SharedExportIdentity* findSharedExport(HANDLE handle) noexcept
        {
            for (const auto& entry : sharedExports)
            {
                if (sameKernelObject(entry.handle, handle))
                {
                    return &entry;
                }
            }
            return nullptr;
        }

        [[nodiscard]] Result<HANDLE> duplicateProcessHandle(HANDLE handle, std::string_view failure)
        {
            HANDLE duplicate = nullptr;
            if (DuplicateHandle(
                    GetCurrentProcess(), handle, GetCurrentProcess(), &duplicate, 0, FALSE, DUPLICATE_SAME_ACCESS) ==
                FALSE)
            {
                return std::unexpected(Error{
                    ErrorCode::nativeFailure,
                    static_cast<std::uint32_t>(GetLastError()),
                    std::string(failure),
                });
            }
            return duplicate;
        }

        [[nodiscard]] Result<SharedHandle> reexportImportIdentity(void* identity, std::string_view failure)
        {
            auto duplicate = duplicateProcessHandle(static_cast<HANDLE>(identity), failure);
            if (!duplicate)
            {
                return std::unexpected(std::move(duplicate.error()));
            }
            return SharedHandle::adopt(SharedHandleView{
                .type = SharedHandleType::windowsNt,
                .value = reinterpret_cast<std::uintptr_t>(*duplicate),
            });
        }

        [[nodiscard]] SharedHandle ownImportIdentity(HANDLE identity) noexcept
        {
            return SharedHandle::adopt(SharedHandleView{
                .type = SharedHandleType::windowsNt,
                .value = reinterpret_cast<std::uintptr_t>(identity),
            });
        }

        template<typename Emplace>
        [[nodiscard]] auto publishImportIdentity(Emplace&& emplace, SharedExportIdentity entry, SharedHandle& identity)
        {
            const std::scoped_lock exports(sharedExportMutex);
#ifdef RACHIS_TESTING
            throwInjectedBadAlloc(VulkanBadAllocPoint::sharedImportRegistry);
#endif
            sharedExports.reserve(sharedExports.size() + 1);
            auto imported = std::forward<Emplace>(emplace)();
            if (imported)
            {
                sharedExports.push_back(std::move(entry));
                static_cast<void>(identity.release());
            }
            return imported;
        }
#endif

        [[nodiscard]] Status validateSharedBufferDesc(const BufferDesc& desc)
        {
            if (desc.memoryType != MemoryType::deviceLocal || desc.residencyPriority != ResidencyPriority::automatic ||
                desc.allocationFlags != ResourceAllocationFlag::none)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Shared buffers require device-local memory, automatic residency, and no allocation flags.",
                });
            }
            if (hasAny(desc.usage,
                       BufferUsage::accelerationStructureStorage | BufferUsage::accelerationStructureScratch))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Shared buffers cannot contain acceleration-structure storage or scratch data.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateSharedTextureDesc(const TextureDesc& desc)
        {
            if (desc.residencyPriority != ResidencyPriority::automatic ||
                desc.allocationFlags != ResourceAllocationFlag::none)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Shared textures require automatic residency and cannot request allocation flags.",
                });
            }
            return {};
        }

        [[maybe_unused]] [[nodiscard]] bool matchingSharedBuffer(const BufferDesc& left,
                                                                 const BufferDesc& right) noexcept
        {
            return left.size == right.size && left.usage == right.usage && left.memoryType == right.memoryType;
        }

        [[maybe_unused]] [[nodiscard]] bool matchingSharedTexture(const TextureDesc& left,
                                                                  const TextureDesc& right) noexcept
        {
            return left.dimension == right.dimension && left.width == right.width && left.height == right.height &&
                   left.depthOrArrayLayers == right.depthOrArrayLayers && left.mipLevels == right.mipLevels &&
                   left.sampleCount == right.sampleCount && left.format == right.format && left.usage == right.usage &&
                   std::ranges::is_permutation(left.viewFormats, right.viewFormats);
        }

        [[maybe_unused]] void retainNative(std::vector<std::shared_ptr<void>>* retained,
                                           const std::shared_ptr<VulkanNativeKeepAlive>& keepAlive)
        {
            if (retained != nullptr && keepAlive != nullptr)
            {
                retained->push_back(keepAlive);
            }
        }

        [[maybe_unused]] [[nodiscard]] VkDeviceMemory exportableMemory(
            const std::shared_ptr<VulkanNativeKeepAlive>& keepAlive, VkDeviceMemory fallback) noexcept
        {
            return keepAlive != nullptr && keepAlive->memory != VK_NULL_HANDLE ? keepAlive->memory : fallback;
        }

    } // namespace

    namespace
    {

        [[nodiscard]] bool validRange(std::uint64_t resourceSize, std::uint64_t offset, std::uint64_t size) noexcept
        {
            return offset < resourceSize && size != 0 && (size == wholeResource || size <= resourceSize - offset);
        }

        [[nodiscard]] std::uint64_t resolveSize(std::uint64_t resourceSize,
                                                std::uint64_t offset,
                                                std::uint64_t size) noexcept
        {
            return size == wholeResource ? resourceSize - offset : size;
        }

        [[nodiscard]] bool validHostRange(std::uint64_t resourceSize, std::uint64_t offset, std::uint64_t size) noexcept
        {
            return offset < resourceSize && size != 0 && size <= resourceSize - offset;
        }

        [[nodiscard]] constexpr bool validMemoryType(MemoryType memoryType) noexcept
        {
            switch (memoryType)
            {
            case MemoryType::deviceLocal:
            case MemoryType::upload:
            case MemoryType::readback:
                return true;
            }
            return false;
        }

        [[nodiscard]] constexpr bool validBufferUsage(BufferUsage usage) noexcept
        {
            constexpr auto allowed =
                static_cast<std::uint32_t>(BufferUsage::copySource) |
                static_cast<std::uint32_t>(BufferUsage::copyDestination) |
                static_cast<std::uint32_t>(BufferUsage::vertex) | static_cast<std::uint32_t>(BufferUsage::index) |
                static_cast<std::uint32_t>(BufferUsage::constant) | static_cast<std::uint32_t>(BufferUsage::storage) |
                static_cast<std::uint32_t>(BufferUsage::indirect) | static_cast<std::uint32_t>(BufferUsage::predicate) |
                static_cast<std::uint32_t>(BufferUsage::accelerationStructureStorage) |
                static_cast<std::uint32_t>(BufferUsage::accelerationStructureBuildInput) |
                static_cast<std::uint32_t>(BufferUsage::accelerationStructureScratch) |
                static_cast<std::uint32_t>(BufferUsage::shaderBindingTable) |
                static_cast<std::uint32_t>(BufferUsage::streamOutput);
            const auto raw = static_cast<std::uint32_t>(usage);
            return raw != 0 && (raw & ~allowed) == 0;
        }

        [[nodiscard]] VkBufferUsageFlags nativeBufferUsage(BufferUsage usage) noexcept
        {
            VkBufferUsageFlags flags = 0;
            if (hasAny(usage, BufferUsage::copySource))
            {
                flags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            }
            if (hasAny(usage, BufferUsage::copyDestination))
            {
                flags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            }
            if (hasAny(usage, BufferUsage::vertex))
            {
                flags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            }
            if (hasAny(usage, BufferUsage::index))
            {
                flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
            }
            if (hasAny(usage, BufferUsage::constant))
            {
                flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            }
            if (hasAny(usage, BufferUsage::storage))
            {
                flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            }
            if (hasAny(usage, BufferUsage::indirect) || hasAny(usage, BufferUsage::predicate))
            {
                flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
            }
            if (hasAny(usage, BufferUsage::accelerationStructureStorage))
            {
                flags |=
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            }
            if (hasAny(usage, BufferUsage::accelerationStructureScratch))
            {
                flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            }
            if (hasAny(usage, BufferUsage::accelerationStructureBuildInput))
            {
                flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            }
            if (hasAny(usage, BufferUsage::shaderBindingTable))
            {
                flags |= VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            }
            return flags;
        }

        [[nodiscard]] std::uint64_t alignDown(std::uint64_t value, std::uint64_t alignment) noexcept
        {
            return alignment <= 1 ? value : value - (value % alignment);
        }

        [[nodiscard]] std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment) noexcept
        {
            if (alignment <= 1)
            {
                return value;
            }
            const auto remainder = value % alignment;
            return remainder == 0 ? value : value + (alignment - remainder);
        }

        inline constexpr std::uint64_t defaultResourceHeapAlignment = 65'536;

        [[nodiscard]] ResourceMemoryRequirements portablePlacementRequirements(const VkMemoryRequirements& native,
                                                                               bool dedicatedAllocation) noexcept
        {
            const auto alignment = std::max(native.alignment, defaultResourceHeapAlignment);
            return ResourceMemoryRequirements{
                .size = alignUp(native.size, alignment),
                .alignment = alignment,
                .dedicatedAllocation = dedicatedAllocation,
            };
        }

        [[nodiscard]] constexpr bool validResidencyPriority(ResidencyPriority priority) noexcept
        {
            switch (priority)
            {
            case ResidencyPriority::automatic:
            case ResidencyPriority::minimum:
            case ResidencyPriority::low:
            case ResidencyPriority::normal:
            case ResidencyPriority::high:
                return true;
            }
            return false;
        }

        [[nodiscard]] constexpr bool validAllocationFlags(ResourceAllocationFlag flags) noexcept
        {
            constexpr auto allowed = static_cast<std::uint32_t>(ResourceAllocationFlag::manualResidency);
            return (static_cast<std::uint32_t>(flags) & ~allowed) == 0;
        }

        [[nodiscard]] constexpr float nativeMemoryPriority(ResidencyPriority priority) noexcept
        {
            switch (priority)
            {
            case ResidencyPriority::minimum:
                return 0.0F;
            case ResidencyPriority::low:
                return 0.25F;
            case ResidencyPriority::high:
                return 1.0F;
            case ResidencyPriority::normal:
            case ResidencyPriority::automatic:
                break;
            }
            return 0.5F;
        }

        [[nodiscard]] constexpr std::size_t residencyPriorityIndex(ResidencyPriority priority) noexcept
        {
            return static_cast<std::size_t>(priority) - 1U;
        }

        [[nodiscard]] constexpr ResidencyPoolClass bufferPoolClass(MemoryType memoryType) noexcept
        {
            switch (memoryType)
            {
            case MemoryType::upload:
                return ResidencyPoolClass::uploadBuffer;
            case MemoryType::readback:
                return ResidencyPoolClass::readbackBuffer;
            case MemoryType::deviceLocal:
                break;
            }
            return ResidencyPoolClass::deviceBuffer;
        }

        [[nodiscard]] constexpr ResidencyPoolClass texturePoolClass(TextureUsage usage) noexcept
        {
            return hasAny(usage, TextureUsage::renderTarget | TextureUsage::depthStencil)
                       ? ResidencyPoolClass::attachmentTexture
                       : ResidencyPoolClass::nonAttachmentTexture;
        }

    } // namespace

    VulkanResourceAllocator::~VulkanResourceAllocator()
    {
        shutdown();
    }

    namespace
    {

        [[nodiscard]] VmaAllocatorCreateFlags allocatorCreateFlags(bool memoryBudget,
                                                                   bool memoryPriority,
                                                                   bool bufferDeviceAddress) noexcept
        {
            VmaAllocatorCreateFlags flags = 0;
            if (memoryBudget)
            {
                flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
            }
            if (memoryPriority)
            {
                flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
            }
            if (bufferDeviceAddress)
            {
                flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
            }
            return flags;
        }

        struct CreatedAllocator final
        {
            VmaAllocator allocator = VK_NULL_HANDLE;
            bool memoryBudgetEnabled = false;
        };

        [[nodiscard]] Result<CreatedAllocator> createAllocatorWithFeatureFallback(VmaAllocatorCreateInfo info,
                                                                                  bool memoryBudget,
                                                                                  bool memoryPriority)
        {
            VmaAllocator allocator = VK_NULL_HANDLE;
            const auto created = vmaCreateAllocator(&info, &allocator);
            if (created == VK_SUCCESS)
            {
                return CreatedAllocator{
                    .allocator = allocator,
                    .memoryBudgetEnabled = memoryBudget,
                };
            }
            if (memoryPriority)
            {
                return std::unexpected(
                    makeVulkanError(vulkanResultErrorCode(created), created, "vmaCreateAllocator(memory priority)"));
            }
            info.flags = 0;
            const auto fallback = vmaCreateAllocator(&info, &allocator);
            if (fallback != VK_SUCCESS)
            {
                return std::unexpected(
                    makeVulkanError(vulkanResultErrorCode(fallback), fallback, "vmaCreateAllocator"));
            }
            return CreatedAllocator{
                .allocator = allocator,
                .memoryBudgetEnabled = false,
            };
        }

        [[nodiscard]] Result<VkCommandPool> createTransferCommandPool(const VolkDeviceTable& table,
                                                                      VkDevice device,
                                                                      std::uint32_t transferFamily)
        {
            VkCommandPool pool = VK_NULL_HANDLE;
            const VkCommandPoolCreateInfo poolInfo{
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = {},
                .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                .queueFamilyIndex = transferFamily,
            };
            const auto poolCreated = table.vkCreateCommandPool(device, &poolInfo, nullptr, &pool);
            if (poolCreated != VK_SUCCESS)
            {
                return std::unexpected(
                    makeVulkanError(vulkanResultErrorCode(poolCreated), poolCreated, "vkCreateCommandPool"));
            }
            return pool;
        }

        struct DeviceUuidAndAllocationLimit final
        {
            std::array<std::uint8_t, VK_UUID_SIZE> deviceUuid{};
            std::uint64_t maxMemoryAllocationSize = std::numeric_limits<std::uint64_t>::max();
        };

        [[nodiscard]] DeviceUuidAndAllocationLimit queryDeviceUuidAndAllocationLimit(
            const VulkanInstanceTable& instanceTable, VkPhysicalDevice physical)
        {
            VkPhysicalDeviceIDProperties id{};
            id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

            VkPhysicalDeviceVulkan11Properties vulkan11{};
            vulkan11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
            vulkan11.pNext = &id;

            VkPhysicalDeviceProperties2 properties2{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                .pNext = &vulkan11,
                .properties = {},
            };
            if (instanceTable.vkGetPhysicalDeviceProperties2 != nullptr)
            {
                instanceTable.vkGetPhysicalDeviceProperties2(physical, &properties2);
            }

            DeviceUuidAndAllocationLimit deviceProperties{};
            std::memcpy(deviceProperties.deviceUuid.data(), id.deviceUUID, VK_UUID_SIZE);
            deviceProperties.maxMemoryAllocationSize = vulkan11.maxMemoryAllocationSize == 0
                                                           ? std::numeric_limits<std::uint64_t>::max()
                                                           : vulkan11.maxMemoryAllocationSize;
            return deviceProperties;
        }

        [[nodiscard]] std::array<bool, 3> queryQueueGraphicsSupport(const VulkanInstanceTable* instanceTable,
                                                                    VkPhysicalDevice physical,
                                                                    const VulkanQueueFamilies& families)
        {
            std::vector<VkQueueFamilyProperties> properties;
            if (instanceTable != nullptr && instanceTable->vkGetPhysicalDeviceQueueFamilyProperties != nullptr &&
                physical != VK_NULL_HANDLE)
            {
                std::uint32_t count = 0;
                instanceTable->vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
                properties.resize(count);
                if (count != 0)
                {
                    instanceTable->vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, properties.data());
                    properties.resize(count);
                }
            }
            const auto supportsGraphics = [&properties, &families](std::uint32_t family) {
                return family == families.graphics ||
                       (family < properties.size() && (properties[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0);
            };
            return {
                supportsGraphics(families.graphics),
                supportsGraphics(families.compute),
                supportsGraphics(families.transfer),
            };
        }

    } // namespace

    Status VulkanResourceAllocator::initialize(VkInstance instance,
                                               const VulkanInstanceTable* instanceTable,
                                               VkPhysicalDevice physical,
                                               VkDevice device,
                                               const VolkDeviceTable* table,
                                               const VkPhysicalDeviceMemoryProperties& memory,
                                               const VulkanQueueFamilies& families,
                                               VkQueue transferQueue,
                                               std::uint64_t nonCoherentAtomSize,
                                               bool samplerMinMaxReduction,
                                               bool textureViewComponentSwizzle,
                                               bool sharedHandles,
                                               bool memoryBudget,
                                               bool memoryPriority,
                                               bool variableRateShadingTier2,
                                               bool bufferDeviceAddress,
                                               bool accelerationStructures,
                                               bool rayTracingPipeline,
                                               std::uint32_t shaderGroupBaseAlignment,
                                               std::uint32_t accelerationStructureScratchAlignment,
                                               ShaderStage enabledShaderStages)
    {
        if (instance == VK_NULL_HANDLE || instanceTable == nullptr || physical == VK_NULL_HANDLE ||
            device == VK_NULL_HANDLE || table == nullptr || transferQueue == VK_NULL_HANDLE ||
            instanceTable->vkGetPhysicalDeviceProperties == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Vulkan resource allocator initialization is missing an instance, dispatch table, device, or transfer "
                "queue.",
            });
        }

        VmaVulkanFunctions functions{};
        populateVulkanAllocatorFunctions(functions, *instanceTable, *table);

        VmaAllocatorCreateInfo info{};
        info.flags = allocatorCreateFlags(memoryBudget, memoryPriority, bufferDeviceAddress);
        info.physicalDevice = physical;
        info.device = device;
        info.instance = instance;
        info.pVulkanFunctions = &functions;
        info.vulkanApiVersion = VK_API_VERSION_1_3;
        auto allocator = createAllocatorWithFeatureFallback(info, memoryBudget, memoryPriority);
        if (!allocator)
        {
            return std::unexpected(std::move(allocator.error()));
        }

        auto pool = createTransferCommandPool(*table, device, families.transfer);
        if (!pool)
        {
            vmaDestroyAllocator(allocator->allocator);
            return std::unexpected(std::move(pool.error()));
        }

        VkPhysicalDeviceProperties properties{};
        instanceTable->vkGetPhysicalDeviceProperties(physical, &properties);
        instanceTable_ = instanceTable;
        physical_ = physical;
        device_ = device;
        table_ = table;
        allocator_ = allocator->allocator;
        memory_ = memory;
        transferPool_ = *pool;
        transferQueue_ = transferQueue;
        families_ = families;
        queueGraphicsSupport_ = queryQueueGraphicsSupport(instanceTable, physical, families);
        formatCache_.reset();
        nonCoherentAtomSize_ = nonCoherentAtomSize == 0 ? 1 : nonCoherentAtomSize;
        limits_ = properties.limits;
        samplerMinMaxReduction_ = samplerMinMaxReduction;
        textureViewComponentSwizzle_ = textureViewComponentSwizzle;
        sharedHandles_ = sharedHandles;
        memoryBudget_ = allocator->memoryBudgetEnabled;
        memoryPriority_ = memoryPriority;
        variableRateShadingTier2_ = variableRateShadingTier2;
        bufferDeviceAddress_ = bufferDeviceAddress;
        accelerationStructures_ = accelerationStructures;
        rayTracingPipeline_ = rayTracingPipeline;
        shaderGroupBaseAlignment_ = rayTracingPipeline ? shaderGroupBaseAlignment : 0;
        accelerationStructureScratchAlignment_ = accelerationStructures ? accelerationStructureScratchAlignment : 0;
        barrierStages_ = enabledPipelineStages(enabledShaderStages, accelerationStructures);
        residencyPools_ = {};
        const auto deviceProperties = queryDeviceUuidAndAllocationLimit(*instanceTable, physical);
        deviceUuid_ = deviceProperties.deviceUuid;
        maxMemoryAllocationSize_ = deviceProperties.maxMemoryAllocationSize;
        return {};
    }

    void VulkanResourceAllocator::waitOnAllCommandsBeforeBarriers(bool enabled) noexcept
    {
        const std::scoped_lock lock(mutex_);
        allCommandsBarrierSources_ = enabled;
    }

#ifdef RACHIS_TESTING
    void VulkanResourceAllocator::initializeDispatchForTesting(VkDevice device,
                                                               const VolkDeviceTable* table,
                                                               const VulkanInstanceTable* instanceTable,
                                                               VkPhysicalDevice physical,
                                                               const VulkanQueueFamilies& families,
                                                               ShaderStage enabledShaderStages,
                                                               bool accelerationStructures) noexcept
    {
        device_ = device;
        table_ = table;
        instanceTable_ = instanceTable;
        physical_ = physical;
        families_ = families;
        queueGraphicsSupport_ = queryQueueGraphicsSupport(instanceTable, physical, families);
        accelerationStructures_ = accelerationStructures;
        barrierStages_ = enabledPipelineStages(enabledShaderStages, accelerationStructures);
        formatCache_.reset();
    }
#endif

    FormatProperties VulkanResourceAllocator::queryFormatProperties(Format format) const noexcept
    {
        return formatCache_.query(instanceTable_, physical_, format);
    }

    bool VulkanResourceAllocator::queueFamilySupportsGraphics(QueueType queueType) const noexcept
    {
        const auto index = static_cast<std::size_t>(queueType);
        return index < queueGraphicsSupport_.size() && queueGraphicsSupport_[index];
    }

    namespace
    {

        void releaseKeepAlivesViewsFirst(const std::vector<std::weak_ptr<VulkanNativeKeepAlive>>& keepAlives) noexcept
        {
            for (const auto& weakKeepAlive : keepAlives)
            {
                if (auto keepAlive = weakKeepAlive.lock();
                    keepAlive != nullptr && keepAlive->kind == VulkanNativeKeepAlive::Kind::view)
                {
                    keepAlive->releaseNow();
                }
            }
            for (const auto& weakKeepAlive : keepAlives)
            {
                if (auto keepAlive = weakKeepAlive.lock(); keepAlive != nullptr)
                {
                    keepAlive->releaseNow();
                }
            }
        }

        void releaseHeapMemory(const std::vector<std::weak_ptr<ResourceHeapRecord>>& heapRecords) noexcept
        {
            for (const auto& weakHeap : heapRecords)
            {
                if (auto heap = weakHeap.lock(); heap != nullptr)
                {
                    heap->releaseNow();
                }
            }
        }

        using ResidencyPoolTable =
            std::array<std::array<std::array<VmaPool, explicitResidencyPriorityCount>, VK_MAX_MEMORY_TYPES>,
                       residencyPoolClassCount>;

        void destroyResidencyPools(VmaAllocator allocator, ResidencyPoolTable& pools) noexcept
        {
            for (auto& byType : pools)
            {
                for (auto& row : byType)
                {
                    for (auto& pool : row)
                    {
                        if (pool == VK_NULL_HANDLE)
                        {
                            continue;
                        }
                        vmaDestroyPool(allocator, pool);
                        pool = VK_NULL_HANDLE;
                    }
                }
            }
        }

    } // namespace

    void VulkanResourceAllocator::discardAllRecordsLocked() noexcept
    {
        views_.forEach([&](TextureViewHandle, TextureViewRecord& record) {
            record.alive = false;
            record.keepAlive.reset();
            record.textureKeepAlive.reset();
        });
        views_.clear();
        samplers_.forEach([&](SamplerHandle, SamplerRecord& record) {
            record.alive = false;
            record.keepAlive.reset();
        });
        samplers_.clear();
        textures_.forEach([&](TextureHandle, TextureRecord& record) {
            closeExportIdentity(record.exportIdentity);
            record.alive = false;
            record.keepAlive.reset();
            record.placement.reset();
        });
        textures_.clear();
        buffers_.forEach([&](BufferHandle, BufferRecord& record) {
            if (record.mapped && record.mappedViaVma && allocator_ != VK_NULL_HANDLE &&
                record.allocation != VK_NULL_HANDLE)
            {
                vmaUnmapMemory(allocator_, record.allocation);
            }
            record.mapped = false;
            record.mappedViaVma = false;
            record.persistentMappedData = nullptr;
            record.mappedBase = nullptr;
            closeExportIdentity(record.exportIdentity);
            record.alive = false;
            record.keepAlive.reset();
            record.placement.reset();
        });
        buffers_.clear();
        heaps_.forEach([&](ResourceHeapHandle, std::shared_ptr<ResourceHeapRecord>& record) {
            if (record != nullptr)
            {
                record->alive = false;
            }
        });
        heaps_.clear();
    }

    void VulkanResourceAllocator::shutdown() noexcept
    {
        const std::scoped_lock lock(mutex_);
        formatCache_.reset();
        commandLayouts_.clear();
        for (const auto& recorded : recordedSwapchainImages_)
        {
            for (const auto& lease : recorded.second)
            {
                lease->releaseRecordedReference();
            }
        }
        recordedSwapchainImages_.clear();
        discardAllRecordsLocked();
        releaseKeepAlivesViewsFirst(nativeKeepAlives_);
        nativeKeepAlives_.clear();
        nextKeepAlivePruneThreshold_ = 64;
        releaseHeapMemory(heapRecords_);
        heapRecords_.clear();
        if (allocator_ != VK_NULL_HANDLE)
        {
            const std::scoped_lock poolLock(residencyPoolsMutex_);
            destroyResidencyPools(allocator_, residencyPools_);
        }
        memoryBudget_ = false;
        memoryPriority_ = false;
        variableRateShadingTier2_ = false;
        bufferDeviceAddress_ = false;
        accelerationStructures_ = false;
        rayTracingPipeline_ = false;
        shaderGroupBaseAlignment_ = 0;
        accelerationStructureScratchAlignment_ = 0;
        barrierStages_ = corePipelineStages;
        allCommandsBarrierSources_ = false;
        maxMemoryAllocationSize_ = std::numeric_limits<std::uint64_t>::max();
        if (device_ != VK_NULL_HANDLE && table_ != nullptr && table_->vkDestroyCommandPool != nullptr &&
            transferPool_ != VK_NULL_HANDLE)
        {
            table_->vkDestroyCommandPool(device_, transferPool_, nullptr);
            transferPool_ = VK_NULL_HANDLE;
        }
        if (allocator_ != VK_NULL_HANDLE)
        {
            vmaDestroyAllocator(allocator_);
            allocator_ = VK_NULL_HANDLE;
        }
        instanceTable_ = nullptr;
        physical_ = VK_NULL_HANDLE;
        device_ = VK_NULL_HANDLE;
        table_ = nullptr;
        transferQueue_ = VK_NULL_HANDLE;
    }

    void VulkanResourceAllocator::trackKeepAliveLocked(const std::shared_ptr<VulkanNativeKeepAlive>& keepAlive)
    {
        if (keepAlive == nullptr)
        {
            return;
        }
        if (nativeKeepAlives_.size() >= nextKeepAlivePruneThreshold_)
        {
            std::erase_if(nativeKeepAlives_, [](const auto& existing) { return existing.expired(); });
            nextKeepAlivePruneThreshold_ = std::max<std::size_t>(64, nativeKeepAlives_.size() * 2);
        }
        nativeKeepAlives_.push_back(keepAlive);
    }

    Result<std::vector<MemoryBudget>> VulkanResourceAllocator::queryMemoryBudgets() const
    {
        if (allocator_ == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Vulkan memory budgets require an initialized allocator.",
            });
        }

        vmaSetCurrentFrameIndex(allocator_, budgetRefreshIndex_.fetch_add(1, std::memory_order_relaxed));
        std::vector<VmaBudget> vmaBudgets(memory_.memoryHeapCount);
        vmaGetHeapBudgets(allocator_, vmaBudgets.data());

        std::vector<MemoryBudget> budgets;
        budgets.reserve(memory_.memoryHeapCount);
        for (std::uint32_t i = 0; i < memory_.memoryHeapCount; ++i)
        {
            const auto& heap = memory_.memoryHeaps[i];
            const auto& budget = vmaBudgets[i];
            const auto exportedBytes = exportedMemory_->bytes[i].load(std::memory_order_relaxed);
            const auto exportedAllocations = exportedMemory_->allocations[i].load(std::memory_order_relaxed);
            budgets.push_back(MemoryBudget{
                .heapIndex = i,
                .properties = (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 ? MemoryHeapProperty::deviceLocal
                                                                                  : MemoryHeapProperty::none,
                .features = MemoryBudgetFeature::allocationStatistics,
                .blockBytes = budget.statistics.blockBytes + exportedBytes,
                .allocationBytes = budget.statistics.allocationBytes + exportedBytes,
                .blockCount = budget.statistics.blockCount + exportedAllocations,
                .allocationCount = budget.statistics.allocationCount + exportedAllocations,
                .budgetBytes = budget.budget != 0 ? budget.budget : heap.size,
                .usageBytes = memoryBudget_ ? budget.usage : budget.usage + exportedBytes,
                .availableForReservationBytes = 0,
                .reservationBytes = 0,
            });
        }
        return budgets;
    }

    Result<ResourceMemorySnapshot> VulkanResourceAllocator::captureResourceMemorySnapshot() const
    {
        const std::scoped_lock lock(mutex_);
        ResourceMemorySnapshot snapshot{};
        const auto addEntry = [&](const ResourceMemoryEntry& entry) {
            snapshot.entries.push_back(entry);
            if (entry.logicalAlive)
            {
                snapshot.liveAllocationBytes += entry.allocationBytes;
                snapshot.liveAllocationCount += 1;
            }
            else
            {
                snapshot.retainedAllocationBytes += entry.allocationBytes;
                snapshot.retainedAllocationCount += 1;
            }
        };
        buffers_.forEach([&](BufferHandle, const BufferRecord& record) {
            if (record.imported)
            {
                return;
            }
            addEntry(ResourceMemoryEntry{
                .allocationId = record.allocationId,
                .kind = ResourceMemoryKind::buffer,
                .memoryType = record.desc.memoryType,
                .allocationBytes = record.allocationBytes,
                .logicalAlive = record.alive,
                .dedicatedAllocation = record.dedicatedAllocation,
                .shared = record.shared,
                .label = record.label,
            });
        });
        textures_.forEach([&](TextureHandle, const TextureRecord& record) {
            if (record.imported)
            {
                return;
            }
            addEntry(ResourceMemoryEntry{
                .allocationId = record.allocationId,
                .kind = ResourceMemoryKind::texture,
                .memoryType = MemoryType::deviceLocal,
                .allocationBytes = record.allocationBytes,
                .logicalAlive = record.alive,
                .dedicatedAllocation = record.dedicatedAllocation,
                .shared = record.shared,
                .label = record.label,
            });
        });
        auto& heaps = heapRecords_;
        auto iterator = heaps.begin();
        while (iterator != heaps.end())
        {
            const auto heap = iterator->lock();
            if (heap == nullptr)
            {
                iterator = heaps.erase(iterator);
                continue;
            }
            addEntry(ResourceMemoryEntry{
                .allocationId = heap->allocationId,
                .kind = ResourceMemoryKind::resourceHeap,
                .memoryType = heap->desc.memoryType,
                .allocationBytes = heap->allocationBytes,
                .logicalAlive = heap->alive,
                .dedicatedAllocation = true,
                .shared = false,
                .label = heap->label,
            });
            ++iterator;
        }
        std::ranges::sort(snapshot.entries, {}, &ResourceMemoryEntry::allocationId);
        return snapshot;
    }

    Result<VkMemoryRequirements2> VulkanResourceAllocator::queryNativeBufferMemoryRequirements(
        const VkBufferCreateInfo& createInfo, VkMemoryDedicatedRequirements& dedicated) const
    {
        if (device_ == VK_NULL_HANDLE || table_ == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Buffer memory requirements require a live Vulkan device.",
            });
        }

        dedicated = VkMemoryDedicatedRequirements{
            .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
            .pNext = {},
            .prefersDedicatedAllocation = {},
            .requiresDedicatedAllocation = {},
        };
        VkMemoryRequirements2 requirements{
            .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
            .pNext = &dedicated,
            .memoryRequirements = {},
        };

        const auto getDeviceReqs = table_->vkGetDeviceBufferMemoryRequirements != nullptr
                                       ? table_->vkGetDeviceBufferMemoryRequirements
                                       : table_->vkGetDeviceBufferMemoryRequirementsKHR;
        if (getDeviceReqs != nullptr)
        {
            const VkDeviceBufferMemoryRequirements deviceInfo{
                .sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS,
                .pNext = {},
                .pCreateInfo = &createInfo,
            };
            getDeviceReqs(device_, &deviceInfo, &requirements);
            return requirements;
        }

        const auto getBufferReqs2 = table_->vkGetBufferMemoryRequirements2 != nullptr
                                        ? table_->vkGetBufferMemoryRequirements2
                                        : table_->vkGetBufferMemoryRequirements2KHR;
        if (table_->vkCreateBuffer == nullptr || table_->vkDestroyBuffer == nullptr ||
            (getBufferReqs2 == nullptr && table_->vkGetBufferMemoryRequirements == nullptr))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Buffer memory requirements require a live Vulkan device.",
            });
        }

        VkBuffer buffer = VK_NULL_HANDLE;
        const auto created = table_->vkCreateBuffer(device_, &createInfo, nullptr, &buffer);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateBuffer"));
        }
        if (getBufferReqs2 != nullptr)
        {
            const VkBufferMemoryRequirementsInfo2 info{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
                .pNext = {},
                .buffer = buffer,
            };
            getBufferReqs2(device_, &info, &requirements);
        }
        else
        {
            table_->vkGetBufferMemoryRequirements(device_, buffer, &requirements.memoryRequirements);
        }
        table_->vkDestroyBuffer(device_, buffer, nullptr);
        return requirements;
    }

    Result<ResourceMemoryRequirements> VulkanResourceAllocator::queryBufferMemoryRequirements(
        const BufferDesc& desc) const
    {
        auto valid = validateBufferDesc(desc);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        VulkanSharingFamilies sharing{};
        auto createInfo = bufferCreateInfo(desc, sharing);
        if (!createInfo)
        {
            return std::unexpected(std::move(createInfo.error()));
        }
        VkMemoryDedicatedRequirements dedicated{};
        auto requirements = queryNativeBufferMemoryRequirements(*createInfo, dedicated);
        if (!requirements)
        {
            return std::unexpected(std::move(requirements.error()));
        }
        return portablePlacementRequirements(requirements->memoryRequirements,
                                             dedicated.requiresDedicatedAllocation == VK_TRUE);
    }

    namespace
    {

        [[nodiscard]] bool isDeviceLocalMemoryType(const VkPhysicalDeviceMemoryProperties& memory,
                                                   std::uint32_t memoryTypeIndex) noexcept
        {
            return memoryTypeIndex < memory.memoryTypeCount &&
                   (memory.memoryTypes[memoryTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        }

        [[nodiscard]] ErrorCode allocationErrorCode(VkResult result) noexcept
        {
            return result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY
                       ? ErrorCode::outOfMemory
                       : vulkanResultErrorCode(result);
        }

        [[nodiscard]] bool requiresDeviceAddress(BufferUsage usage) noexcept
        {
            return hasAny(usage,
                          BufferUsage::accelerationStructureStorage | BufferUsage::accelerationStructureScratch |
                              BufferUsage::accelerationStructureBuildInput | BufferUsage::shaderBindingTable |
                              BufferUsage::indirect);
        }

        [[nodiscard]] VkDeviceAddress queryBufferDeviceAddress(const VolkDeviceTable& table,
                                                               VkDevice device,
                                                               VkBuffer buffer)
        {
            const VkBufferDeviceAddressInfo addressInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                .pNext = {},
                .buffer = buffer,
            };
            return table.vkGetBufferDeviceAddress(device, &addressInfo);
        }

    } // namespace

    Status VulkanResourceAllocator::createNativeBuffer(const VkBufferCreateInfo& createInfo, BufferRecord& record)
    {
        const auto& desc = record.desc;
        std::uint32_t priorityTypeBits = 0;
        std::uint64_t priorityRequiredSize = 0;
        std::uint32_t prioritySelectedMemoryType = ~0U;
        auto allocationCreate = bufferAllocationCreateInfo(createInfo,
                                                           desc.memoryType,
                                                           desc.residencyPriority,
                                                           priorityTypeBits,
                                                           priorityRequiredSize,
                                                           prioritySelectedMemoryType);
        if (!allocationCreate)
        {
            return std::unexpected(std::move(allocationCreate.error()));
        }
        const VkDeviceSize minAlignment = std::max(
            hasAny(desc.usage, BufferUsage::shaderBindingTable) ? shaderGroupBaseAlignment_ : 0,
            hasAny(desc.usage, BufferUsage::accelerationStructureScratch) ? accelerationStructureScratchAlignment_ : 0);
        const auto createBuffer = [&](const VmaAllocationCreateInfo& allocation) {
            if (minAlignment > 1)
            {
                return vmaCreateBufferWithAlignment(allocator_,
                                                    &createInfo,
                                                    &allocation,
                                                    minAlignment,
                                                    &record.buffer,
                                                    &record.allocation,
                                                    &record.allocationInfo);
            }
            return vmaCreateBuffer(
                allocator_, &createInfo, &allocation, &record.buffer, &record.allocation, &record.allocationInfo);
        };
        auto created = createBuffer(*allocationCreate);
        const bool prioritizedUploadExhaustedDeviceLocalMemory =
            created == VK_ERROR_OUT_OF_DEVICE_MEMORY && desc.residencyPriority != ResidencyPriority::automatic &&
            desc.memoryType == MemoryType::upload && isDeviceLocalMemoryType(memory_, prioritySelectedMemoryType);
        if (prioritizedUploadExhaustedDeviceLocalMemory)
        {
            const auto fallbackBits = stripDeviceLocalMemoryTypes(priorityTypeBits);
            if (auto fallbackType = findMemoryType(fallbackBits, desc.memoryType, priorityRequiredSize);
                fallbackType.has_value())
            {
                if (auto fallbackPool =
                        residencyPool(bufferPoolClass(desc.memoryType), *fallbackType, desc.residencyPriority);
                    fallbackPool.has_value())
                {
                    allocationCreate->pool = *fallbackPool;
                    created = createBuffer(*allocationCreate);
                }
            }
        }
        if (created != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(allocationErrorCode(created), created, "vmaCreateBuffer"));
        }
        return {};
    }

    Result<BufferHandle> VulkanResourceAllocator::createBuffer(const BufferDesc& desc)
    {
        auto valid = validateBufferDesc(desc);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        BufferRecord record{};
        auto createInfo = bufferCreateInfo(desc, record.sharingFamilies);
        if (!createInfo)
        {
            return std::unexpected(std::move(createInfo.error()));
        }

        record.desc = desc;
        record.label = std::string(desc.label);
        record.keepAlive = makeVulkanNativeKeepAlive(VulkanNativeKeepAlive::Kind::buffer, VK_NULL_HANDLE, nullptr);
        record.keepAlive->allocator = allocator_;
        if (auto created = createNativeBuffer(*createInfo, record); !created)
        {
            return std::unexpected(std::move(created.error()));
        }
        record.keepAlive->buffer = record.buffer;
        record.keepAlive->allocation = record.allocation;

        record.allocationId = nextAllocationId_.fetch_add(1, std::memory_order_relaxed);
        record.allocationBytes = record.allocationInfo.size;
        VkMemoryPropertyFlags properties = 0;
        vmaGetAllocationMemoryProperties(allocator_, record.allocation, &properties);
        record.hostVisible = (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
        record.hostCoherent = (properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        record.persistentMappedData =
            record.hostVisible ? static_cast<std::byte*>(record.allocationInfo.pMappedData) : nullptr;
        record.memoryHeapIndex = memory_.memoryTypes[record.allocationInfo.memoryType].heapIndex;
        if (!record.label.empty())
        {
            vmaSetAllocationName(allocator_, record.allocation, record.label.c_str());
            setVulkanObjectName(instanceTable_,
                                device_,
                                VK_OBJECT_TYPE_BUFFER,
                                reinterpret_cast<std::uint64_t>(record.buffer),
                                record.label.c_str());
        }

        if (bufferDeviceAddress_ && requiresDeviceAddress(desc.usage) && table_ != nullptr &&
            table_->vkGetBufferDeviceAddress != nullptr)
        {
            record.deviceAddress = queryBufferDeviceAddress(*table_, device_, record.buffer);
        }
        const std::scoped_lock lock(mutex_);
        trackKeepAliveLocked(record.keepAlive);
        return emplaceRecordLocked(buffers_, std::move(record));
    }

    namespace
    {

        [[nodiscard]] VkResult createExternalMemoryBuffer(const VolkDeviceTable& table,
                                                          VkDevice device,
                                                          VkBufferCreateInfo createInfo,
                                                          VkBuffer& buffer)
        {
            const VkExternalMemoryBufferCreateInfo external{
                .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                .pNext = {},
                .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
            };
            createInfo.pNext = &external;
            return table.vkCreateBuffer(device, &createInfo, nullptr, &buffer);
        }

        [[nodiscard]] VkMemoryRequirements queryMemoryRequirementsForBuffer(const VolkDeviceTable& table,
                                                                            VkDevice device,
                                                                            VkBuffer buffer)
        {
            VkMemoryDedicatedRequirements dedicated{
                .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
                .pNext = {},
                .prefersDedicatedAllocation = {},
                .requiresDedicatedAllocation = {},
            };
            VkMemoryRequirements2 requirements{
                .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
                .pNext = &dedicated,
                .memoryRequirements = {},
            };
            const VkBufferMemoryRequirementsInfo2 requirementInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
                .pNext = {},
                .buffer = buffer,
            };
            table.vkGetBufferMemoryRequirements2(device, &requirementInfo, &requirements);
            return requirements.memoryRequirements;
        }

        struct DedicatedMemoryRequest final
        {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkImage image = VK_NULL_HANDLE;
            VkDeviceSize allocationSize = 0;
            std::uint32_t memoryTypeIndex = 0;
            bool deviceAddress = false;
        };

        [[nodiscard]] VkResult allocateDedicatedMemory(const VolkDeviceTable& table,
                                                       VkDevice device,
                                                       const DedicatedMemoryRequest& request,
                                                       const void* externalMemoryInfo,
                                                       VkDeviceMemory& memory)
        {
            const VkMemoryAllocateFlagsInfo allocateFlags{
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                .pNext = externalMemoryInfo,
                .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
                .deviceMask = 0,
            };
            const VkMemoryAllocateInfo allocate{
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .pNext = request.deviceAddress ? static_cast<const void*>(&allocateFlags) : externalMemoryInfo,
                .allocationSize = request.allocationSize,
                .memoryTypeIndex = request.memoryTypeIndex,
            };
            return table.vkAllocateMemory(device, &allocate, nullptr, &memory);
        }

        [[nodiscard]] VkResult allocateExportableMemory(const VolkDeviceTable& table,
                                                        VkDevice device,
                                                        const DedicatedMemoryRequest& request,
                                                        VkDeviceMemory& memory)
        {
            const VkMemoryDedicatedAllocateInfo dedicatedAlloc{
                .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                .pNext = {},
                .image = request.image,
                .buffer = request.buffer,
            };
            const VkExportMemoryAllocateInfo exportAlloc{
                .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
                .pNext = &dedicatedAlloc,
                .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
            };
            return allocateDedicatedMemory(table, device, request, &exportAlloc, memory);
        }

        [[nodiscard]] VmaAllocationInfo dedicatedAllocationInfo(VkDeviceMemory memory,
                                                                std::uint32_t memoryTypeIndex,
                                                                VkDeviceSize size) noexcept
        {
            VmaAllocationInfo info{};
            info.memoryType = memoryTypeIndex;
            info.deviceMemory = memory;
            info.offset = 0;
            info.size = size;
            return info;
        }

        void countExportedMemory(VulkanNativeKeepAlive& keepAlive,
                                 const std::shared_ptr<VulkanExportedMemoryTotals>& totals,
                                 std::uint32_t heapIndex,
                                 std::uint64_t bytes) noexcept
        {
            keepAlive.exportedTotals = totals;
            keepAlive.exportedHeapIndex = heapIndex;
            keepAlive.exportedBytes = bytes;
            totals->add(heapIndex, bytes);
        }

    } // namespace

    Status VulkanResourceAllocator::validateSharedBufferCreation(const BufferDesc& desc) const
    {
        if (!sharedHandles_)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "createSharedBuffer is not implemented on the Vulkan backend yet.",
            });
        }
        if (auto sharedValid = validateSharedBufferDesc(desc); !sharedValid)
        {
            return sharedValid;
        }
        if (auto valid = validateBufferDesc(desc); !valid)
        {
            return valid;
        }
        if (table_ == nullptr || table_->vkCreateBuffer == nullptr ||
            table_->vkGetBufferMemoryRequirements2 == nullptr || table_->vkAllocateMemory == nullptr ||
            table_->vkBindBufferMemory == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Shared buffer creation requires a live Vulkan device table.",
            });
        }
        return {};
    }

    Result<VmaAllocationInfo> VulkanResourceAllocator::backBufferWithExportableMemory(VkBuffer buffer,
                                                                                      MemoryType memoryType,
                                                                                      bool deviceAddress)
    {
        const auto requirements = queryMemoryRequirementsForBuffer(*table_, device_, buffer);
        auto memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, memoryType, requirements.size);
        if (!memoryTypeIndex)
        {
            return std::unexpected(std::move(memoryTypeIndex.error()));
        }

        DedicatedMemoryRequest request{
            .buffer = buffer,
            .image = VK_NULL_HANDLE,
            .allocationSize = requirements.size,
            .memoryTypeIndex = *memoryTypeIndex,
            .deviceAddress = deviceAddress,
        };
        VkDeviceMemory memory = VK_NULL_HANDLE;
        auto allocated = allocateExportableMemory(*table_, device_, request, memory);
        const bool uploadExhaustedDeviceLocalMemory = allocated == VK_ERROR_OUT_OF_DEVICE_MEMORY &&
                                                      memoryType == MemoryType::upload &&
                                                      isDeviceLocalMemoryType(memory_, request.memoryTypeIndex);
        if (uploadExhaustedDeviceLocalMemory)
        {
            const auto fallbackBits = stripDeviceLocalMemoryTypes(requirements.memoryTypeBits);
            if (auto fallbackType = findMemoryType(fallbackBits, memoryType, requirements.size);
                fallbackType.has_value())
            {
                request.memoryTypeIndex = *fallbackType;
                allocated = allocateExportableMemory(*table_, device_, request, memory);
            }
        }
        if (allocated != VK_SUCCESS)
        {
            return std::unexpected(
                makeVulkanError(allocationErrorCode(allocated), allocated, "vkAllocateMemory(shared)"));
        }
        const auto bound = table_->vkBindBufferMemory(device_, buffer, memory, 0);
        if (bound != VK_SUCCESS)
        {
            table_->vkFreeMemory(device_, memory, nullptr);
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(bound), bound, "vkBindBufferMemory(shared)"));
        }
        return dedicatedAllocationInfo(memory, request.memoryTypeIndex, requirements.size);
    }

    Result<BufferHandle> VulkanResourceAllocator::createSharedBuffer(const BufferDesc& desc)
    {
        if (auto creatable = validateSharedBufferCreation(desc); !creatable)
        {
            return std::unexpected(std::move(creatable.error()));
        }
        BufferRecord record{};
        record.desc = desc;
        record.label = std::string(desc.label);
        record.shared = true;
        record.dedicatedAllocation = true;
        auto createInfo = bufferCreateInfo(desc, record.sharingFamilies);
        if (!createInfo)
        {
            return std::unexpected(std::move(createInfo.error()));
        }
        record.keepAlive = makeVulkanNativeKeepAlive(VulkanNativeKeepAlive::Kind::buffer, device_, table_);
        const auto created = createExternalMemoryBuffer(*table_, device_, *createInfo, record.buffer);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(allocationErrorCode(created), created, "vkCreateBuffer(shared)"));
        }
        record.keepAlive->buffer = record.buffer;
        const bool needsDeviceAddress =
            bufferDeviceAddress_ && (createInfo->usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0;
        auto allocation = backBufferWithExportableMemory(record.buffer, desc.memoryType, needsDeviceAddress);
        if (!allocation)
        {
            return std::unexpected(std::move(allocation.error()));
        }
        record.keepAlive->memory = allocation->deviceMemory;

        record.allocationId = nextAllocationId_.fetch_add(1, std::memory_order_relaxed);
        record.allocationInfo = *allocation;
        record.allocationBytes = allocation->size;
        const auto& nativeMemoryType = memory_.memoryTypes[allocation->memoryType];
        record.memoryHeapIndex = nativeMemoryType.heapIndex;
        record.hostVisible = (nativeMemoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
        record.hostCoherent = (nativeMemoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        if (!record.label.empty())
        {
            setVulkanObjectName(instanceTable_,
                                device_,
                                VK_OBJECT_TYPE_BUFFER,
                                reinterpret_cast<std::uint64_t>(record.buffer),
                                record.label.c_str());
        }
        countExportedMemory(*record.keepAlive, exportedMemory_, record.memoryHeapIndex, record.allocationBytes);
        if (needsDeviceAddress && table_->vkGetBufferDeviceAddress != nullptr)
        {
            record.deviceAddress = queryBufferDeviceAddress(*table_, device_, record.buffer);
        }
        const std::scoped_lock lock(mutex_);
        trackKeepAliveLocked(record.keepAlive);
        return emplaceRecordLocked(buffers_, std::move(record));
    }

#ifdef _WIN32
    namespace
    {

        [[nodiscard]] Result<SharedHandle> exportSharedMemory(const VolkDeviceTable& table,
                                                              VkDevice device,
                                                              VkDeviceMemory memory,
                                                              SharedExportIdentity entry,
                                                              void*& identity,
                                                              std::string_view duplicateFailure)
        {
            if (identity == nullptr)
            {
                const std::scoped_lock exports(sharedExportMutex);
#ifdef RACHIS_TESTING
                throwInjectedBadAlloc(VulkanBadAllocPoint::sharedExportRegistry);
#endif
                sharedExports.reserve(sharedExports.size() + 1);
                const VkMemoryGetWin32HandleInfoKHR info{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
                    .pNext = {},
                    .memory = memory,
                    .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
                };
                HANDLE handle = nullptr;
                const auto exported = table.vkGetMemoryWin32HandleKHR(device, &info, &handle);
                if (exported != VK_SUCCESS || handle == nullptr)
                {
                    return std::unexpected(
                        makeVulkanError(vulkanResultErrorCode(exported), exported, "vkGetMemoryWin32HandleKHR"));
                }
                entry.handle = handle;
                sharedExports.push_back(std::move(entry));
                identity = handle;
            }
            HANDLE duplicate = nullptr;
            if (DuplicateHandle(GetCurrentProcess(),
                                static_cast<HANDLE>(identity),
                                GetCurrentProcess(),
                                &duplicate,
                                0,
                                FALSE,
                                DUPLICATE_SAME_ACCESS) == FALSE)
            {
                return std::unexpected(Error{
                    ErrorCode::nativeFailure,
                    static_cast<std::uint32_t>(GetLastError()),
                    std::string(duplicateFailure),
                });
            }
            return SharedHandle::adopt(SharedHandleView{
                .type = SharedHandleType::windowsNt,
                .value = reinterpret_cast<std::uintptr_t>(duplicate),
            });
        }

    } // namespace
#endif

    Result<SharedHandle> VulkanResourceAllocator::exportSharedBuffer(BufferHandle buffer)
    {
#ifndef _WIN32
        static_cast<void>(buffer);
        return std::unexpected(Error{
            ErrorCode::unsupported,
            0,
            "exportSharedBuffer uses Windows NT handles. Linux FD sharing is not implemented.",
        });
#else
        if (!sharedHandles_ || table_ == nullptr || table_->vkGetMemoryWin32HandleKHR == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "exportSharedBuffer is not implemented on the Vulkan backend yet.",
            });
        }
        const std::scoped_lock lock(mutex_);
        auto* record = findBuffer(buffer);
        if (record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "exportSharedBuffer received a stale or foreign buffer handle.",
            });
        }
        if (!record->shared)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Only buffers created or imported through the shared-buffer API can be exported.",
            });
        }
        if (record->imported)
        {
            return reexportImportIdentity(record->exportIdentity,
                                          "DuplicateHandle failed for a re-exported shared buffer.");
        }
        auto* const memory = exportableMemory(record->keepAlive, record->allocationInfo.deviceMemory);
        if (memory == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "exportSharedBuffer could not resolve the dedicated Vulkan device memory.",
            });
        }
        auto identityDesc = record->desc;
        identityDesc.label = {};
        return exportSharedMemory(*table_,
                                  device_,
                                  memory,
                                  SharedExportIdentity{
                                      .buffer = identityDesc,
                                      .deviceUuid = deviceUuid_,
                                      .memoryTypeIndex = record->allocationInfo.memoryType,
                                      .allocationBytes = record->allocationBytes,
                                  },
                                  record->exportIdentity,
                                  "DuplicateHandle failed for a shared buffer export.");
#endif
    }

#ifdef _WIN32
    namespace
    {

        struct ExportedMemoryLayout final
        {
            std::uint32_t memoryTypeIndex = 0;
            std::uint64_t allocationBytes = 0;
        };

        [[nodiscard]] Result<std::optional<ExportedMemoryLayout>> findLocalBufferExportLayout(
            HANDLE handle, const BufferDesc& desc, const std::array<std::uint8_t, VK_UUID_SIZE>& deviceUuid)
        {
            const std::scoped_lock exports(sharedExportMutex);
            const auto* identity = findSharedExport(handle);
            if (identity == nullptr)
            {
                return std::optional<ExportedMemoryLayout>{};
            }
            if (identity->texture || identity->deviceUuid != deviceUuid ||
                !matchingSharedBuffer(identity->buffer, desc))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The imported buffer description or usage does not match the native Vulkan resource.",
                });
            }
            return std::optional<ExportedMemoryLayout>{ExportedMemoryLayout{
                .memoryTypeIndex = identity->memoryTypeIndex,
                .allocationBytes = identity->allocationBytes,
            }};
        }

        [[nodiscard]] VkResult importWin32Memory(const VolkDeviceTable& table,
                                                 VkDevice device,
                                                 const DedicatedMemoryRequest& request,
                                                 HANDLE handle,
                                                 VkDeviceMemory& memory)
        {
            const VkMemoryDedicatedAllocateInfo dedicatedAlloc{
                .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                .pNext = {},
                .image = request.image,
                .buffer = request.buffer,
            };
            const VkImportMemoryWin32HandleInfoKHR importInfo{
                .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR,
                .pNext = &dedicatedAlloc,
                .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
                .handle = handle,
                .name = {},
            };
            return allocateDedicatedMemory(table, device, request, &importInfo, memory);
        }

        [[nodiscard]] ErrorCode importedMemoryErrorCode(VkResult result) noexcept
        {
            return result == VK_ERROR_INVALID_EXTERNAL_HANDLE ? ErrorCode::invalidArgument
                                                              : vulkanResultErrorCode(result);
        }

    } // namespace

    struct VulkanResourceAllocator::SharedMemoryImport final
    {
        HANDLE handle = nullptr;
        std::optional<ExportedMemoryLayout> localExportLayout;
    };

    Status VulkanResourceAllocator::validateSharedBufferImport(SharedHandleView handle, const BufferDesc& desc) const
    {
        if (!sharedHandles_ || table_ == nullptr || table_->vkAllocateMemory == nullptr ||
            table_->vkCreateBuffer == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "importSharedBuffer is not implemented on the Vulkan backend yet.",
            });
        }
        if (!handle.isValid() || handle.type != SharedHandleType::windowsNt)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Shared buffer import requires a valid Windows NT handle.",
            });
        }
        if (auto sharedValid = validateSharedBufferDesc(desc); !sharedValid)
        {
            return sharedValid;
        }
        return validateBufferDesc(desc);
    }

    Result<VmaAllocationInfo> VulkanResourceAllocator::backBufferWithImportedMemory(VkBuffer buffer,
                                                                                    MemoryType memoryType,
                                                                                    const SharedMemoryImport& source,
                                                                                    bool deviceAddress)
    {
        const auto requirements = queryMemoryRequirementsForBuffer(*table_, device_, buffer);
        Result<std::uint32_t> memoryTypeIndex = std::unexpected(Error{
            ErrorCode::unsupported,
            0,
            "Opaque Win32 buffer import could not select a compatible memory type.",
        });
        if (source.localExportLayout.has_value())
        {
            if ((requirements.memoryTypeBits & (1U << source.localExportLayout->memoryTypeIndex)) != 0)
            {
                memoryTypeIndex = source.localExportLayout->memoryTypeIndex;
            }
        }
        else
        {
            memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, memoryType, requirements.size);
        }
        if (!memoryTypeIndex)
        {
            return std::unexpected(std::move(memoryTypeIndex.error()));
        }

        const DedicatedMemoryRequest request{
            .buffer = buffer,
            .image = VK_NULL_HANDLE,
            .allocationSize =
                source.localExportLayout.has_value() ? source.localExportLayout->allocationBytes : requirements.size,
            .memoryTypeIndex = *memoryTypeIndex,
            .deviceAddress = deviceAddress,
        };
        VkDeviceMemory memory = VK_NULL_HANDLE;
        const auto allocated = importWin32Memory(*table_, device_, request, source.handle, memory);
        if (allocated != VK_SUCCESS)
        {
            return std::unexpected(
                makeVulkanError(importedMemoryErrorCode(allocated), allocated, "vkAllocateMemory(shared import)"));
        }
        const auto bound = table_->vkBindBufferMemory(device_, buffer, memory, 0);
        if (bound != VK_SUCCESS)
        {
            table_->vkFreeMemory(device_, memory, nullptr);
            return std::unexpected(
                makeVulkanError(vulkanResultErrorCode(bound), bound, "vkBindBufferMemory(shared import)"));
        }
        return dedicatedAllocationInfo(memory, *memoryTypeIndex, request.allocationSize);
    }
#endif

    Result<BufferHandle> VulkanResourceAllocator::importSharedBuffer(SharedHandleView handle, const BufferDesc& desc)
    {
#ifndef _WIN32
        static_cast<void>(handle);
        static_cast<void>(desc);
        return std::unexpected(Error{
            ErrorCode::unsupported,
            0,
            "importSharedBuffer uses Windows NT handles. Linux FD sharing is not implemented.",
        });
#else
        if (auto importable = validateSharedBufferImport(handle, desc); !importable)
        {
            return std::unexpected(std::move(importable.error()));
        }
        auto* const nativeHandle = std::bit_cast<HANDLE>(handle.value);
        auto localExportLayout = findLocalBufferExportLayout(nativeHandle, desc, deviceUuid_);
        if (!localExportLayout)
        {
            return std::unexpected(std::move(localExportLayout.error()));
        }

        BufferRecord record{};
        record.desc = desc;
        record.label = std::string(desc.label);
        record.shared = true;
        record.imported = true;
        record.dedicatedAllocation = true;
        auto createInfo = bufferCreateInfo(desc, record.sharingFamilies);
        if (!createInfo)
        {
            return std::unexpected(std::move(createInfo.error()));
        }
        record.keepAlive = makeVulkanNativeKeepAlive(VulkanNativeKeepAlive::Kind::buffer, device_, table_);
        const auto created = createExternalMemoryBuffer(*table_, device_, *createInfo, record.buffer);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(
                makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateBuffer(shared import)"));
        }
        record.keepAlive->buffer = record.buffer;
        const bool needsDeviceAddress =
            bufferDeviceAddress_ && (createInfo->usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0;
        const SharedMemoryImport source{
            .handle = nativeHandle,
            .localExportLayout = *localExportLayout,
        };
        auto allocation = backBufferWithImportedMemory(record.buffer, desc.memoryType, source, needsDeviceAddress);
        if (!allocation)
        {
            return std::unexpected(std::move(allocation.error()));
        }
        record.keepAlive->memory = allocation->deviceMemory;
        auto identity = duplicateProcessHandle(nativeHandle, "DuplicateHandle failed for an imported shared buffer.");
        if (!identity)
        {
            return std::unexpected(std::move(identity.error()));
        }
        auto identityOwner = ownImportIdentity(*identity);

        SharedExportIdentity entry{
            .handle = *identity,
            .buffer = desc,
            .deviceUuid = deviceUuid_,
            .memoryTypeIndex = allocation->memoryType,
            .allocationBytes = allocation->size,
        };
        entry.buffer.label = {};
        record.allocationInfo = *allocation;
        record.allocationBytes = allocation->size;
        record.exportIdentity = *identity;
        if (needsDeviceAddress && table_->vkGetBufferDeviceAddress != nullptr)
        {
            record.deviceAddress = queryBufferDeviceAddress(*table_, device_, record.buffer);
        }
        const std::scoped_lock lock(mutex_);
        trackKeepAliveLocked(record.keepAlive);
        return publishImportIdentity(
            [&] { return emplaceRecordLocked(buffers_, std::move(record)); }, std::move(entry), identityOwner);
#endif
    }

    Status VulkanResourceAllocator::destroyBuffer(BufferHandle buffer)
    {
        const std::scoped_lock lock(mutex_);
        auto* record = buffers_.find(buffer);
        if (record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "destroyBuffer received a stale or foreign buffer handle.",
            });
        }
        if (record->mapped)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A mapped buffer cannot be destroyed.",
            });
        }
        closeExportIdentity(record->exportIdentity);
        record->alive = false;
        buffers_.erase(buffer);
        return {};
    }

    namespace
    {

        struct HostMapping final
        {
            std::byte* data = nullptr;
            bool mappedViaVma = false;
        };

        [[nodiscard]] Result<HostMapping> acquireHostMapping(VmaAllocator allocator, const auto& bufferRecord)
        {
            if (bufferRecord.persistentMappedData != nullptr)
            {
                return HostMapping{
                    .data = bufferRecord.persistentMappedData,
                    .mappedViaVma = false,
                };
            }
            if (bufferRecord.allocation != VK_NULL_HANDLE)
            {
                void* mappedData = nullptr;
                const auto mapped = vmaMapMemory(allocator, bufferRecord.allocation, &mappedData);
                if (mapped != VK_SUCCESS)
                {
                    return std::unexpected(makeVulkanError(vulkanResultErrorCode(mapped), mapped, "vmaMapMemory"));
                }
                return HostMapping{
                    .data = static_cast<std::byte*>(mappedData),
                    .mappedViaVma = true,
                };
            }
            const bool placedInMappedHeap = bufferRecord.placement != nullptr &&
                                            bufferRecord.placement->heap != nullptr &&
                                            bufferRecord.placement->heap->mappedBase != nullptr;
            if (!placedInMappedHeap)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The buffer has no host-visible memory mapping.",
                });
            }
            return HostMapping{
                .data = bufferRecord.placement->heap->mappedBase + bufferRecord.placement->offset,
                .mappedViaVma = false,
            };
        }

    } // namespace

    Result<MappedMemory> VulkanResourceAllocator::mapBuffer(BufferHandle buffer,
                                                            std::uint64_t offset,
                                                            std::uint64_t size)
    {
        const std::scoped_lock lock(mutex_);
        auto* record = findBuffer(buffer);
        if (record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "mapBuffer received a stale or foreign buffer handle.",
            });
        }
        if (record->desc.memoryType == MemoryType::deviceLocal || !record->hostVisible)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Device-local buffers cannot be mapped directly.",
            });
        }
        if (!validRange(record->desc.size, offset, size))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The requested map range is outside the buffer.",
            });
        }

        if (!record->alive)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The buffer handle became stale before mapping.",
            });
        }
        if (record->mapped)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The buffer is already mapped.",
            });
        }

        auto mapping = acquireHostMapping(allocator_, *record);
        if (!mapping)
        {
            return std::unexpected(std::move(mapping.error()));
        }
        record->mapped = true;
        record->mappedViaVma = mapping->mappedViaVma;
        record->mappedBase = mapping->data;
        record->mappedOffset = offset;
        record->mappedSize = resolveSize(record->desc.size, offset, size);
        return MappedMemory{record->mappedBase + offset, record->mappedSize};
    }

    Status VulkanResourceAllocator::unmapBuffer(BufferHandle buffer,
                                                std::uint64_t offset,
                                                std::uint64_t size,
                                                bool flushWrites)
    {
        const std::scoped_lock lock(mutex_);
        auto* record = findBuffer(buffer);
        if (record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "unmapBuffer received a stale or foreign buffer handle.",
            });
        }
        if (!validRange(record->desc.size, offset, size))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The requested unmap range is outside the buffer.",
            });
        }

        if (!record->mapped)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The buffer is not mapped.",
            });
        }
        const auto resolvedSize = resolveSize(record->desc.size, offset, size);
        const bool wholeBuffer = offset == 0 && resolvedSize == record->desc.size;
        const auto mappedEnd = record->mappedOffset + record->mappedSize;
        if (!wholeBuffer && (offset < record->mappedOffset || offset + resolvedSize > mappedEnd))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The requested unmap range is not contained in the active mapping.",
            });
        }
        if (flushWrites && record->desc.memoryType == MemoryType::upload)
        {
            auto flushed = synchronizeHostRange(*record,
                                                wholeBuffer ? record->mappedOffset : offset,
                                                wholeBuffer ? record->mappedSize : resolvedSize,
                                                true);
            if (!flushed)
            {
                return std::unexpected(std::move(flushed.error()));
            }
        }
        if (record->mappedViaVma && record->allocation != VK_NULL_HANDLE)
        {
            vmaUnmapMemory(allocator_, record->allocation);
        }
        record->mapped = false;
        record->mappedViaVma = false;
        record->mappedBase = nullptr;
        record->mappedOffset = 0;
        record->mappedSize = 0;
        return {};
    }

    Status VulkanResourceAllocator::flushMappedMemoryRanges(BufferHandle buffer,
                                                            std::span<const MappedMemoryRange> ranges)
    {
        const std::scoped_lock lock(mutex_);
        auto* record = findBuffer(buffer);
        if (record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "flushMappedMemoryRanges received a stale or foreign buffer handle.",
            });
        }
        return synchronizeMappedMemoryRanges(*record, ranges, MemoryType::upload, true);
    }

    Status VulkanResourceAllocator::invalidateMappedMemoryRanges(BufferHandle buffer,
                                                                 std::span<const MappedMemoryRange> ranges)
    {
        const std::scoped_lock lock(mutex_);
        auto* record = findBuffer(buffer);
        if (record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "invalidateMappedMemoryRanges received a stale or foreign buffer handle.",
            });
        }
        return synchronizeMappedMemoryRanges(*record, ranges, MemoryType::readback, false);
    }

    Status VulkanResourceAllocator::writeBuffer(BufferHandle buffer,
                                                std::uint64_t offset,
                                                std::span<const std::byte> data)
    {
        const std::scoped_lock lock(mutex_);
        auto* record = findBuffer(buffer);
        if (record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "writeBuffer received a stale or foreign buffer handle.",
            });
        }
        if (!validHostRange(record->desc.size, offset, data.size()))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The host buffer write range is empty or outside the buffer.",
            });
        }
        if (record->desc.memoryType != MemoryType::upload || !record->hostVisible)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Host buffer writes require upload memory.",
            });
        }

        if (!record->alive)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The buffer handle became stale before the host write.",
            });
        }
        if (record->mapped)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A directly mapped buffer cannot be written through the one-shot host API.",
            });
        }

        auto mapping = acquireHostMapping(allocator_, *record);
        if (!mapping)
        {
            return std::unexpected(std::move(mapping.error()));
        }
        std::memcpy(mapping->data + offset, data.data(), data.size());
        auto flushed = synchronizeHostRange(*record, offset, data.size(), true);
        if (mapping->mappedViaVma)
        {
            vmaUnmapMemory(allocator_, record->allocation);
        }
        if (!flushed)
        {
            return std::unexpected(std::move(flushed.error()));
        }
        return {};
    }

    Status VulkanResourceAllocator::readBuffer(BufferHandle buffer,
                                               std::uint64_t offset,
                                               std::span<std::byte> destination)
    {
        const std::scoped_lock lock(mutex_);
        auto* record = findBuffer(buffer);
        if (record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "readBuffer received a stale or foreign buffer handle.",
            });
        }
        if (!validHostRange(record->desc.size, offset, destination.size()))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The host buffer read range is empty or outside the buffer.",
            });
        }
        if (record->desc.memoryType != MemoryType::readback || !record->hostVisible)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Host buffer reads require readback memory.",
            });
        }

        if (!record->alive)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The buffer handle became stale before the host read.",
            });
        }
        if (record->mapped)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A directly mapped buffer cannot be read through the one-shot host API.",
            });
        }

        auto mapping = acquireHostMapping(allocator_, *record);
        if (!mapping)
        {
            return std::unexpected(std::move(mapping.error()));
        }
        auto invalidated = synchronizeHostRange(*record, offset, destination.size(), false);
        if (!invalidated)
        {
            if (mapping->mappedViaVma)
            {
                vmaUnmapMemory(allocator_, record->allocation);
            }
            return std::unexpected(std::move(invalidated.error()));
        }
        std::memcpy(destination.data(), mapping->data + offset, destination.size());
        if (mapping->mappedViaVma)
        {
            vmaUnmapMemory(allocator_, record->allocation);
        }
        return {};
    }

#ifdef RACHIS_TESTING
    namespace
    {

        [[nodiscard]] Result<VkCommandBuffer> beginImmediateTransfer(const VolkDeviceTable& table,
                                                                     VkDevice device,
                                                                     VkCommandPool pool)
        {
            const VkCommandBufferAllocateInfo allocate{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = {},
                .commandPool = pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1,
            };
            VkCommandBuffer command = VK_NULL_HANDLE;
            const auto allocated = table.vkAllocateCommandBuffers(device, &allocate, &command);
            if (allocated != VK_SUCCESS)
            {
                return std::unexpected(
                    makeVulkanError(vulkanResultErrorCode(allocated), allocated, "vkAllocateCommandBuffers"));
            }

            const VkCommandBufferBeginInfo begin{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext = {},
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                .pInheritanceInfo = {},
            };
            const auto began = table.vkBeginCommandBuffer(command, &begin);
            if (began != VK_SUCCESS)
            {
                table.vkFreeCommandBuffers(device, pool, 1, &command);
                return std::unexpected(makeVulkanError(vulkanResultErrorCode(began), began, "vkBeginCommandBuffer"));
            }
            return command;
        }

        [[nodiscard]] Status submitImmediateTransferAndWait(
            const VolkDeviceTable& table, VkDevice device, VkCommandPool pool, VkQueue queue, VkCommandBuffer command)
        {
            const auto ended = table.vkEndCommandBuffer(command);
            if (ended != VK_SUCCESS)
            {
                table.vkFreeCommandBuffers(device, pool, 1, &command);
                return std::unexpected(makeVulkanError(vulkanResultErrorCode(ended), ended, "vkEndCommandBuffer"));
            }

            const VkSubmitInfo submit{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = {},
                .waitSemaphoreCount = {},
                .pWaitSemaphores = {},
                .pWaitDstStageMask = {},
                .commandBufferCount = 1,
                .pCommandBuffers = &command,
                .signalSemaphoreCount = {},
                .pSignalSemaphores = {},
            };
            const auto submitted = table.vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
            if (submitted != VK_SUCCESS)
            {
                table.vkFreeCommandBuffers(device, pool, 1, &command);
                return std::unexpected(makeVulkanError(vulkanResultErrorCode(submitted), submitted, "vkQueueSubmit"));
            }
            const auto idle = table.vkQueueWaitIdle(queue);
            table.vkFreeCommandBuffers(device, pool, 1, &command);
            if (idle != VK_SUCCESS)
            {
                return std::unexpected(makeVulkanError(vulkanResultErrorCode(idle), idle, "vkQueueWaitIdle"));
            }
            return {};
        }

    } // namespace

    Status VulkanResourceAllocator::copyBufferImmediate(BufferHandle source,
                                                        BufferHandle destination,
                                                        std::uint64_t sourceOffset,
                                                        std::uint64_t destinationOffset,
                                                        std::uint64_t size)
    {
        const std::scoped_lock lock(mutex_);
        auto* sourceRecord = findBuffer(source);
        auto* destinationRecord = findBuffer(destination);
        if (sourceRecord == nullptr || destinationRecord == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Immediate buffer copy received a stale handle.",
            });
        }
        if (size == 0 || sourceOffset + size > sourceRecord->desc.size ||
            destinationOffset + size > destinationRecord->desc.size)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Immediate buffer copy range is empty or out of bounds.",
            });
        }
        if (table_ == nullptr || table_->vkAllocateCommandBuffers == nullptr ||
            table_->vkBeginCommandBuffer == nullptr || table_->vkCmdCopyBuffer == nullptr ||
            table_->vkEndCommandBuffer == nullptr || table_->vkQueueSubmit == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Immediate buffer copy is unavailable because the Vulkan device table is incomplete.",
            });
        }

        auto command = beginImmediateTransfer(*table_, device_, transferPool_);
        if (!command)
        {
            return std::unexpected(std::move(command.error()));
        }
        const VkBufferCopy region{
            .srcOffset = sourceOffset,
            .dstOffset = destinationOffset,
            .size = size,
        };
        table_->vkCmdCopyBuffer(*command, sourceRecord->buffer, destinationRecord->buffer, 1, &region);
        return submitImmediateTransferAndWait(*table_, device_, transferPool_, transferQueue_, *command);
    }

    Status VulkanResourceAllocator::recordFillBuffer(VkCommandBuffer command,
                                                     BufferHandle buffer,
                                                     std::uint32_t value,
                                                     std::vector<std::shared_ptr<void>>* retained)
    {
        if (command == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A fill command requires a live command buffer.",
            });
        }
        const std::scoped_lock lock(mutex_);
        const auto* record = buffers_.find(buffer);
        if (record == nullptr || !record->alive || record->buffer == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The fill destination buffer handle is invalid or stale.",
            });
        }
        if (!hasAny(record->desc.usage, BufferUsage::copyDestination) || record->desc.size < 4 ||
            (record->desc.size % 4U) != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "vkCmdFillBuffer requires a 4-byte-aligned copy-destination buffer.",
            });
        }
        if (table_ == nullptr || table_->vkCmdFillBuffer == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Buffer fill is unavailable because the Vulkan device table is incomplete.",
            });
        }
        retainNative(retained, record->keepAlive);
        table_->vkCmdFillBuffer(command, record->buffer, 0, record->desc.size, value);
        return {};
    }
#endif

    Status VulkanResourceAllocator::validateBufferDesc(const BufferDesc& desc) const
    {
        if (desc.size == 0 || !validBufferUsage(desc.usage) || !validMemoryType(desc.memoryType) ||
            !validResidencyPriority(desc.residencyPriority))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A buffer requires a non-zero size, valid memory and residency values, at least one usage flag, and "
                "valid allocation flags.",
            });
        }
        if (auto residency = validateResidencyRequest(desc.residencyPriority, desc.allocationFlags); !residency)
        {
            return residency;
        }
        if (auto sized = validateAllocationSize(desc.size, desc.memoryType); !sized)
        {
            return sized;
        }
        const auto raw = static_cast<std::uint32_t>(desc.usage);
        const bool accelerationStructureStorage = hasAny(desc.usage, BufferUsage::accelerationStructureStorage);
        const bool accelerationStructureScratch = hasAny(desc.usage, BufferUsage::accelerationStructureScratch);
        const bool accelerationStructureBuildInput = hasAny(desc.usage, BufferUsage::accelerationStructureBuildInput);
        const bool shaderBindingTable = hasAny(desc.usage, BufferUsage::shaderBindingTable);
        if ((accelerationStructureStorage || accelerationStructureScratch || accelerationStructureBuildInput) &&
            !accelerationStructures_)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Acceleration-structure buffers require acceleration-structure support.",
            });
        }
        if (shaderBindingTable && !rayTracingPipeline_)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Shader-binding-table buffers require ray-tracing pipelines.",
            });
        }
        if ((accelerationStructureStorage && desc.usage != BufferUsage::accelerationStructureStorage) ||
            (accelerationStructureScratch && desc.usage != BufferUsage::accelerationStructureScratch) ||
            ((accelerationStructureStorage || accelerationStructureScratch) &&
             desc.memoryType != MemoryType::deviceLocal))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Acceleration-structure storage and scratch buffers are dedicated device-local resources.",
            });
        }
        constexpr auto streamOutput = static_cast<std::uint32_t>(BufferUsage::streamOutput);
        if ((raw & streamOutput) != 0)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Stream-output buffers are not supported on Vulkan.",
            });
        }
        if (desc.memoryType == MemoryType::readback && hasAny(desc.usage, BufferUsage::storage))
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Readback buffers cannot have storage usage on Vulkan; copy shader results into them instead.",
            });
        }
        return {};
    }

    VulkanSharingFamilies VulkanResourceAllocator::bufferSharingFamilies(BufferUsage usage) const noexcept
    {
        VulkanSharingFamilies sharing{};
        sharing.add(families_.graphics);
        if (hasAny(usage,
                   BufferUsage::copySource | BufferUsage::copyDestination | BufferUsage::constant |
                       BufferUsage::storage | BufferUsage::indirect | BufferUsage::predicate |
                       BufferUsage::accelerationStructureStorage | BufferUsage::accelerationStructureBuildInput |
                       BufferUsage::accelerationStructureScratch | BufferUsage::shaderBindingTable))
        {
            sharing.add(families_.compute);
        }
        if (hasAny(usage, BufferUsage::copySource | BufferUsage::copyDestination))
        {
            sharing.add(families_.transfer);
        }
        return sharing;
    }

    VulkanSharingFamilies VulkanResourceAllocator::textureSharingFamilies(TextureUsage usage) const noexcept
    {
        VulkanSharingFamilies sharing{};
        sharing.add(families_.graphics);
        if (hasAny(usage,
                   TextureUsage::copySource | TextureUsage::copyDestination | TextureUsage::sampled |
                       TextureUsage::storage))
        {
            sharing.add(families_.compute);
        }
        if (hasAny(usage, TextureUsage::copySource | TextureUsage::copyDestination))
        {
            sharing.add(families_.transfer);
        }
        return sharing;
    }

    Result<VkBufferCreateInfo> VulkanResourceAllocator::bufferCreateInfo(const BufferDesc& desc,
                                                                         VulkanSharingFamilies& sharing) const
    {
        const auto usage = nativeBufferUsage(desc.usage);
        if (usage == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A buffer requires at least one usage flag.",
            });
        }
        sharing = bufferSharingFamilies(desc.usage);
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = desc.size;
        info.usage = usage;
        info.sharingMode = sharing.sharingMode();
        if (info.sharingMode == VK_SHARING_MODE_CONCURRENT)
        {
            info.queueFamilyIndexCount = sharing.count;
            info.pQueueFamilyIndices = sharing.indices.data();
        }

        if (bufferDeviceAddress_ && hasAny(desc.usage, BufferUsage::indirect))
        {
            info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        }
        return info;
    }

    VmaAllocationCreateInfo VulkanResourceAllocator::allocationCreateInfo(MemoryType memoryType) const noexcept
    {
        VmaAllocationCreateInfo info{};
        info.usage = VMA_MEMORY_USAGE_AUTO;
        info.priority = nativeMemoryPriority(ResidencyPriority::automatic);
        switch (memoryType)
        {
        case MemoryType::deviceLocal:
            info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            break;
        case MemoryType::upload:
            info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            info.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
        case MemoryType::readback:
            info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            info.preferredFlags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
            info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
        }
        return info;
    }

    Status VulkanResourceAllocator::validateResidencyRequest(ResidencyPriority priority,
                                                             ResourceAllocationFlag flags) const
    {
        if (!validResidencyPriority(priority))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The resource residency priority is invalid.",
            });
        }
        if (!validAllocationFlags(flags))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The resource allocation flags are invalid.",
            });
        }
        if (flags != ResourceAllocationFlag::none)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Manual residency is not supported on Vulkan.",
            });
        }
        if (priority != ResidencyPriority::automatic && !memoryPriority_)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Explicit residency priorities are unavailable on this device.",
            });
        }
        return {};
    }

    Status VulkanResourceAllocator::validateAllocationSize(std::uint64_t bytes, MemoryType memoryType) const
    {
        if (bytes == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "An allocation requires a non-zero size.",
            });
        }
        if (bytes > maxMemoryAllocationSize_)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "The allocation exceeds VkPhysicalDeviceVulkan11Properties::maxMemoryAllocationSize.",
            });
        }

        std::uint64_t largestCompatibleHeap = 0;
        for (std::uint32_t index = 0; index < memory_.memoryTypeCount; ++index)
        {
            const auto& type = memory_.memoryTypes[index];
            const auto flags = type.propertyFlags;
            const auto heapSize = memory_.memoryHeaps[type.heapIndex].size;
            const bool deviceLocal = (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
            const bool hostVisible = (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
            const bool matches =
                memoryType == MemoryType::deviceLocal
                    ? deviceLocal
                    : (memoryType == MemoryType::upload || memoryType == MemoryType::readback) && hostVisible;
            if (matches)
            {
                largestCompatibleHeap = std::max(largestCompatibleHeap, heapSize);
            }
        }
        if (largestCompatibleHeap != 0 && bytes > largestCompatibleHeap)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "The allocation exceeds every compatible Vulkan memory heap.",
            });
        }
        return {};
    }

    std::uint32_t VulkanResourceAllocator::stripDeviceLocalMemoryTypes(std::uint32_t typeBits) const noexcept
    {
        for (std::uint32_t index = 0; index < memory_.memoryTypeCount; ++index)
        {
            if ((memory_.memoryTypes[index].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0)
            {
                typeBits &= ~(1U << index);
            }
        }
        return typeBits;
    }

    Result<VmaAllocationCreateInfo> VulkanResourceAllocator::bufferAllocationCreateInfo(
        const VkBufferCreateInfo& createInfo,
        MemoryType memoryType,
        ResidencyPriority priority,
        std::uint32_t& outTypeBits,
        std::uint64_t& outRequiredSize,
        std::uint32_t& outSelectedMemoryType)
    {
        outTypeBits = 0;
        outRequiredSize = 0;
        outSelectedMemoryType = ~0U;
        auto info = allocationCreateInfo(memoryType);
        if (priority == ResidencyPriority::automatic)
        {
            return info;
        }
        if (!validResidencyPriority(priority))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The resource residency priority is invalid.",
            });
        }
        if (!memoryPriority_ || allocator_ == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Explicit residency priorities are unavailable on this device.",
            });
        }

        VkMemoryDedicatedRequirements dedicated{};
        auto requirements = queryNativeBufferMemoryRequirements(createInfo, dedicated);
        if (!requirements)
        {
            return std::unexpected(std::move(requirements.error()));
        }
        outTypeBits = requirements->memoryRequirements.memoryTypeBits;
        outRequiredSize = std::max<std::uint64_t>(createInfo.size, requirements->memoryRequirements.size);
        auto selected = findMemoryType(outTypeBits, memoryType, outRequiredSize);
        if (!selected)
        {
            return std::unexpected(std::move(selected.error()));
        }
        auto pool = residencyPool(bufferPoolClass(memoryType), *selected, priority);
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }
        outSelectedMemoryType = *selected;
        info.pool = *pool;
        info.priority = nativeMemoryPriority(priority);
        return info;
    }

    Result<VmaAllocationCreateInfo> VulkanResourceAllocator::imageAllocationCreateInfo(
        const VkImageCreateInfo& createInfo,
        MemoryType memoryType,
        ResidencyPriority priority,
        ResidencyPoolClass poolClass)
    {
        auto info = allocationCreateInfo(memoryType);
        if (priority == ResidencyPriority::automatic)
        {
            return info;
        }
        if (!validResidencyPriority(priority))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The resource residency priority is invalid.",
            });
        }
        if (!memoryPriority_ || allocator_ == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Explicit residency priorities are unavailable on this device.",
            });
        }

        VkMemoryDedicatedRequirements dedicated{};
        auto requirements = queryNativeImageMemoryRequirements(createInfo, dedicated);
        if (!requirements)
        {
            return std::unexpected(std::move(requirements.error()));
        }
        auto selected = findMemoryType(
            requirements->memoryRequirements.memoryTypeBits, memoryType, requirements->memoryRequirements.size);
        if (!selected)
        {
            return std::unexpected(std::move(selected.error()));
        }
        auto pool = residencyPool(poolClass, *selected, priority);
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }
        info.pool = *pool;
        info.priority = nativeMemoryPriority(priority);
        return info;
    }

    Result<VmaPool> VulkanResourceAllocator::residencyPool(ResidencyPoolClass poolClass,
                                                           std::uint32_t memoryTypeIndex,
                                                           ResidencyPriority priority)
    {
        if (!validResidencyPriority(priority) || priority == ResidencyPriority::automatic)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The resource residency priority is invalid.",
            });
        }
        if (!memoryPriority_ || allocator_ == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Explicit residency priorities are unavailable on this device.",
            });
        }

        const auto poolIndex = static_cast<std::size_t>(poolClass);
        const auto priorityIndex = residencyPriorityIndex(priority);
        if (poolIndex >= residencyPools_.size() || memoryTypeIndex >= memory_.memoryTypeCount ||
            memoryTypeIndex >= VK_MAX_MEMORY_TYPES || priorityIndex >= explicitResidencyPriorityCount)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The residency pool class is invalid.",
            });
        }

        const std::scoped_lock lock(residencyPoolsMutex_);
        auto& pool = residencyPools_[poolIndex][memoryTypeIndex][priorityIndex];
        if (pool != VK_NULL_HANDLE)
        {
            return pool;
        }

        VmaPoolCreateInfo poolInfo{};
        poolInfo.memoryTypeIndex = memoryTypeIndex;
        poolInfo.priority = nativeMemoryPriority(priority);
        const auto created = vmaCreatePool(allocator_, &poolInfo, &pool);
        if (created != VK_SUCCESS)
        {
            pool = VK_NULL_HANDLE;
            return std::unexpected(makeVulkanError(allocationErrorCode(created), created, "vmaCreatePool"));
        }
        return pool;
    }

    Status VulkanResourceAllocator::synchronizeMappedMemoryRanges(BufferRecord& record,
                                                                  std::span<const MappedMemoryRange> ranges,
                                                                  MemoryType requiredMemoryType,
                                                                  bool flush)
    {
        if (ranges.empty())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                std::string("Mapped-memory ") + (flush ? "flush" : "invalidation") + " requires at least one range.",
            });
        }

        if (!record.alive)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The buffer handle became stale before mapped-memory synchronization.",
            });
        }
        if (record.desc.memoryType != requiredMemoryType)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                std::string("Mapped-memory ") +
                    (flush ? "flush requires an upload buffer." : "invalidation requires a readback buffer."),
            });
        }
        if (!record.mapped)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Mapped-memory synchronization requires an active mapping.",
            });
        }

        const auto mappedEnd = record.mappedOffset + record.mappedSize;
        for (const auto& range : ranges)
        {
            if (!validRange(record.desc.size, range.offset, range.size))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A mapped-memory synchronization range is empty or out of bounds.",
                });
            }
            const auto resolvedSize = resolveSize(record.desc.size, range.offset, range.size);
            if (range.offset < record.mappedOffset || range.offset + resolvedSize > mappedEnd)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A mapped-memory synchronization range is not contained in the active mapping.",
                });
            }
        }

        for (const auto& range : ranges)
        {
            auto synchronized = synchronizeHostRange(
                record, range.offset, resolveSize(record.desc.size, range.offset, range.size), flush);
            if (!synchronized)
            {
                return synchronized;
            }
        }
        return {};
    }

    Status VulkanResourceAllocator::synchronizeHostRange(const BufferRecord& record,
                                                         std::uint64_t offset,
                                                         std::uint64_t size,
                                                         bool flush)
    {
        if (record.hostCoherent || size == 0)
        {
            return {};
        }
        if (record.allocation != VK_NULL_HANDLE)
        {
            const auto result = flush ? vmaFlushAllocation(allocator_, record.allocation, offset, size)
                                      : vmaInvalidateAllocation(allocator_, record.allocation, offset, size);
            if (result != VK_SUCCESS)
            {
                return std::unexpected(makeVulkanError(
                    vulkanResultErrorCode(result), result, flush ? "vmaFlushAllocation" : "vmaInvalidateAllocation"));
            }
            return {};
        }
        if (record.placement == nullptr || record.placement->heap == nullptr || table_ == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The buffer has no host-visible memory to synchronize.",
            });
        }

        const auto memorySize = record.placement->heap->allocationBytes;
        const auto begin = alignDown(record.allocationInfo.offset + offset, nonCoherentAtomSize_);
        const auto end = alignUp(record.allocationInfo.offset + offset + size, nonCoherentAtomSize_);
        const VkMappedMemoryRange range{
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .pNext = {},
            .memory = record.allocationInfo.deviceMemory,
            .offset = begin,
            .size = end >= memorySize ? VK_WHOLE_SIZE : end - begin,
        };
        const auto synchronize = flush ? table_->vkFlushMappedMemoryRanges : table_->vkInvalidateMappedMemoryRanges;
        if (synchronize == nullptr)
        {
            return {};
        }
        const auto result = synchronize(device_, 1, &range);
        if (result != VK_SUCCESS)
        {
            return std::unexpected(
                makeVulkanError(vulkanResultErrorCode(result),
                                result,
                                flush ? "vkFlushMappedMemoryRanges" : "vkInvalidateMappedMemoryRanges"));
        }
        return {};
    }

    VulkanResourceAllocator::BufferRecord* VulkanResourceAllocator::findBuffer(BufferHandle buffer) noexcept
    {
        return buffers_.find(buffer);
    }

    const VulkanResourceAllocator::BufferRecord* VulkanResourceAllocator::findBuffer(BufferHandle buffer) const noexcept
    {
        return buffers_.find(buffer);
    }

    namespace
    {

        [[nodiscard]] constexpr bool isDepthFormat(Format format) noexcept
        {
            return format == Format::d32Float || format == Format::d24UnormS8Uint;
        }

        [[nodiscard]] constexpr bool isCompressedFormat(Format format) noexcept
        {
            switch (format)
            {
            case Format::bc1RgbaUnorm:
            case Format::bc1RgbaUnormSrgb:
            case Format::bc2RgbaUnorm:
            case Format::bc2RgbaUnormSrgb:
            case Format::bc3RgbaUnorm:
            case Format::bc3RgbaUnormSrgb:
            case Format::bc4RUnorm:
            case Format::bc4RSnorm:
            case Format::bc5RgUnorm:
            case Format::bc5RgSnorm:
            case Format::bc6hRgbUfloat:
            case Format::bc6hRgbSfloat:
            case Format::bc7RgbaUnorm:
            case Format::bc7RgbaUnormSrgb:
                return true;
            default:
                return false;
            }
        }

        [[nodiscard]] constexpr bool validTextureUsage(TextureUsage usage) noexcept
        {
            constexpr auto allowed = static_cast<std::uint32_t>(TextureUsage::copySource) |
                                     static_cast<std::uint32_t>(TextureUsage::copyDestination) |
                                     static_cast<std::uint32_t>(TextureUsage::sampled) |
                                     static_cast<std::uint32_t>(TextureUsage::storage) |
                                     static_cast<std::uint32_t>(TextureUsage::renderTarget) |
                                     static_cast<std::uint32_t>(TextureUsage::depthStencil) |
                                     static_cast<std::uint32_t>(TextureUsage::shadingRateSource);
            const auto raw = static_cast<std::uint32_t>(usage);
            return raw != 0 && (raw & ~allowed) == 0;
        }

        [[nodiscard]] std::uint32_t fullMipCount(const TextureDesc& desc) noexcept
        {
            auto extent = std::max(desc.width, desc.height);
            if (desc.dimension == TextureDimension::texture3D)
            {
                extent = std::max(extent, desc.depthOrArrayLayers);
            }
            std::uint32_t levels = 1;
            while (extent > 1)
            {
                extent /= 2;
                ++levels;
            }
            return levels;
        }

        [[nodiscard]] VkImageUsageFlags nativeViewUsage(TextureViewUsage usage) noexcept
        {
            switch (usage)
            {
            case TextureViewUsage::storage:
                return VK_IMAGE_USAGE_STORAGE_BIT;
            case TextureViewUsage::renderTarget:
                return VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            case TextureViewUsage::depthStencil:
                return VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            case TextureViewUsage::sampled:
                break;
            }
            return VK_IMAGE_USAGE_SAMPLED_BIT;
        }

        [[nodiscard]] VkImageUsageFlags nativeImageUsage(TextureUsage usage) noexcept
        {
            VkImageUsageFlags flags = 0;
            if (hasAny(usage, TextureUsage::copySource))
            {
                flags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            }
            if (hasAny(usage, TextureUsage::copyDestination))
            {
                flags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            }
            if (hasAny(usage, TextureUsage::sampled))
            {
                flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
            }
            if (hasAny(usage, TextureUsage::storage))
            {
                flags |= VK_IMAGE_USAGE_STORAGE_BIT;
            }
            if (hasAny(usage, TextureUsage::renderTarget))
            {
                flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            }
            if (hasAny(usage, TextureUsage::depthStencil))
            {
                flags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            }
            if (hasAny(usage, TextureUsage::shadingRateSource))
            {
                flags |= VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR;
            }
            return flags;
        }

        [[nodiscard]] VkImageType nativeImageType(TextureDimension dimension) noexcept
        {
            switch (dimension)
            {
            case TextureDimension::texture1D:
                return VK_IMAGE_TYPE_1D;
            case TextureDimension::texture3D:
                return VK_IMAGE_TYPE_3D;
            case TextureDimension::texture2D:
                break;
            }
            return VK_IMAGE_TYPE_2D;
        }

        [[nodiscard]] VkSampleCountFlagBits nativeSampleCount(std::uint32_t sampleCount) noexcept
        {
            switch (sampleCount)
            {
            case 2:
                return VK_SAMPLE_COUNT_2_BIT;
            case 4:
                return VK_SAMPLE_COUNT_4_BIT;
            case 8:
                return VK_SAMPLE_COUNT_8_BIT;
            default:
                return VK_SAMPLE_COUNT_1_BIT;
            }
        }

        [[nodiscard]] VkImageAspectFlags nativeAspect(TextureAspect aspect, Format format) noexcept
        {
            switch (aspect)
            {
            case TextureAspect::depth:
                return VK_IMAGE_ASPECT_DEPTH_BIT;
            case TextureAspect::stencil:
                return VK_IMAGE_ASPECT_STENCIL_BIT;
            case TextureAspect::color:
                break;
            }
            return isDepthFormat(format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        }

        [[nodiscard]] constexpr bool isIdentityLikeComponentMapping(const ComponentMapping& mapping) noexcept
        {
            const auto identityOr = [](ComponentSwizzle value, ComponentSwizzle channel) noexcept {
                return value == ComponentSwizzle::identity || value == channel;
            };
            return identityOr(mapping.red, ComponentSwizzle::red) &&
                   identityOr(mapping.green, ComponentSwizzle::green) &&
                   identityOr(mapping.blue, ComponentSwizzle::blue) &&
                   identityOr(mapping.alpha, ComponentSwizzle::alpha);
        }

        [[nodiscard]] VkComponentSwizzle nativeSwizzle(ComponentSwizzle swizzle, VkComponentSwizzle identity) noexcept
        {
            switch (swizzle)
            {
            case ComponentSwizzle::zero:
                return VK_COMPONENT_SWIZZLE_ZERO;
            case ComponentSwizzle::one:
                return VK_COMPONENT_SWIZZLE_ONE;
            case ComponentSwizzle::red:
                return VK_COMPONENT_SWIZZLE_R;
            case ComponentSwizzle::green:
                return VK_COMPONENT_SWIZZLE_G;
            case ComponentSwizzle::blue:
                return VK_COMPONENT_SWIZZLE_B;
            case ComponentSwizzle::alpha:
                return VK_COMPONENT_SWIZZLE_A;
            case ComponentSwizzle::identity:
                break;
            }
            return identity;
        }

        [[nodiscard]] VkFilter nativeFilter(FilterMode mode) noexcept
        {
            return mode == FilterMode::linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        }

        [[nodiscard]] VkFilter nativeSamplerFilter(const SamplerDesc& desc, FilterMode mode) noexcept
        {
            return desc.maxAnisotropy > 1 ? VK_FILTER_LINEAR : nativeFilter(mode);
        }

        [[nodiscard]] VkSamplerAddressMode nativeAddress(AddressMode mode) noexcept
        {
            switch (mode)
            {
            case AddressMode::repeat:
                return VK_SAMPLER_ADDRESS_MODE_REPEAT;
            case AddressMode::mirrorRepeat:
                return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
            case AddressMode::clampToBorder:
                return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            case AddressMode::clampToEdge:
                break;
            }
            return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        }

        [[nodiscard]] VkCompareOp nativeCompare(CompareOp op) noexcept
        {
            switch (op)
            {
            case CompareOp::never:
                return VK_COMPARE_OP_NEVER;
            case CompareOp::less:
                return VK_COMPARE_OP_LESS;
            case CompareOp::equal:
                return VK_COMPARE_OP_EQUAL;
            case CompareOp::lessEqual:
                return VK_COMPARE_OP_LESS_OR_EQUAL;
            case CompareOp::greater:
                return VK_COMPARE_OP_GREATER;
            case CompareOp::notEqual:
                return VK_COMPARE_OP_NOT_EQUAL;
            case CompareOp::greaterEqual:
                return VK_COMPARE_OP_GREATER_OR_EQUAL;
            case CompareOp::always:
                break;
            }
            return VK_COMPARE_OP_ALWAYS;
        }

        [[nodiscard]] VkBorderColor nativeBorder(BorderColor color) noexcept
        {
            switch (color)
            {
            case BorderColor::opaqueBlack:
                return VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
            case BorderColor::opaqueWhite:
                return VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
            case BorderColor::transparentBlack:
                break;
            }
            return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        }

        [[nodiscard]] float nativeMaxLod(const SamplerDesc& desc) noexcept
        {
            constexpr float unmipmappedMaxLod = 0.25F;
            return desc.mipFilter == MipmapFilterMode::none ? std::min(desc.maxLod, unmipmappedMaxLod) : desc.maxLod;
        }

        [[nodiscard]] constexpr bool validComponentSwizzle(ComponentSwizzle swizzle) noexcept
        {
            switch (swizzle)
            {
            case ComponentSwizzle::identity:
            case ComponentSwizzle::zero:
            case ComponentSwizzle::one:
            case ComponentSwizzle::red:
            case ComponentSwizzle::green:
            case ComponentSwizzle::blue:
            case ComponentSwizzle::alpha:
                return true;
            }
            return false;
        }

        [[nodiscard]] constexpr bool validComponentMapping(const ComponentMapping& mapping) noexcept
        {
            return validComponentSwizzle(mapping.red) && validComponentSwizzle(mapping.green) &&
                   validComponentSwizzle(mapping.blue) && validComponentSwizzle(mapping.alpha);
        }

        [[nodiscard]] constexpr bool dimensionAllowed(TextureDimension textureDimension,
                                                      TextureViewDimension viewDimension) noexcept
        {
            switch (textureDimension)
            {
            case TextureDimension::texture1D:
                return viewDimension == TextureViewDimension::texture1D ||
                       viewDimension == TextureViewDimension::texture1DArray;
            case TextureDimension::texture2D:
                return viewDimension == TextureViewDimension::texture2D ||
                       viewDimension == TextureViewDimension::texture2DArray ||
                       viewDimension == TextureViewDimension::cube || viewDimension == TextureViewDimension::cubeArray;
            case TextureDimension::texture3D:
                return viewDimension == TextureViewDimension::texture3D;
            }
            return false;
        }

        [[maybe_unused]] [[nodiscard]] constexpr std::uint32_t texelByteSize(Format format) noexcept
        {
            switch (format)
            {
            case Format::r8Unorm:
            case Format::r8Uint:
            case Format::r8Sint:
                return 1;
            case Format::rg8Unorm:
            case Format::rg8Uint:
            case Format::rg8Sint:
                return 2;
            case Format::rgba8Unorm:
            case Format::rgba8UnormSrgb:
            case Format::bgra8Unorm:
            case Format::bgra8UnormSrgb:
            case Format::rgba8Uint:
            case Format::rgba8Sint:
            case Format::r32Float:
            case Format::r32Uint:
            case Format::r32Sint:
            case Format::rg16Float:
            case Format::rg16Uint:
            case Format::rg16Sint:
                return 4;
            case Format::rgba16Float:
            case Format::rgba16Uint:
            case Format::rgba16Sint:
            case Format::rg32Float:
            case Format::rg32Uint:
            case Format::rg32Sint:
                return 8;
            case Format::rgba32Float:
            case Format::rgba32Uint:
            case Format::rgba32Sint:
                return 16;
            case Format::r64Uint:
                return 8;
            case Format::r16Float:
            case Format::r16Uint:
            case Format::r16Sint:
                return 2;
            case Format::unknown:
            case Format::d32Float:
            case Format::d24UnormS8Uint:
            case Format::bc1RgbaUnorm:
            case Format::bc1RgbaUnormSrgb:
            case Format::bc2RgbaUnorm:
            case Format::bc2RgbaUnormSrgb:
            case Format::bc3RgbaUnorm:
            case Format::bc3RgbaUnormSrgb:
            case Format::bc4RUnorm:
            case Format::bc4RSnorm:
            case Format::bc5RgUnorm:
            case Format::bc5RgSnorm:
            case Format::bc6hRgbUfloat:
            case Format::bc6hRgbSfloat:
            case Format::bc7RgbaUnorm:
            case Format::bc7RgbaUnormSrgb:
                break;
            }
            return 0;
        }

        [[maybe_unused]] [[nodiscard]] constexpr std::uint32_t mipExtent(std::uint32_t value,
                                                                         std::uint32_t mipLevel) noexcept
        {
            return std::max(1U, value >> mipLevel);
        }

        [[maybe_unused]] [[nodiscard]] VkImageMemoryBarrier2 imageLayoutBarrier(VkImage image,
                                                                                VkImageLayout oldLayout,
                                                                                VkImageLayout newLayout,
                                                                                VkAccessFlags2 srcAccess,
                                                                                VkAccessFlags2 dstAccess,
                                                                                VkImageAspectFlags aspect,
                                                                                std::uint32_t mipLevel,
                                                                                std::uint32_t arrayLayer) noexcept
        {
            return VkImageMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .pNext = {},
                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .srcAccessMask = srcAccess,
                .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                .dstAccessMask = dstAccess,
                .oldLayout = oldLayout,
                .newLayout = newLayout,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange =
                    {
                        .aspectMask = aspect,
                        .baseMipLevel = mipLevel,
                        .levelCount = 1,
                        .baseArrayLayer = arrayLayer,
                        .layerCount = 1,
                    },
            };
        }

        [[nodiscard]] Status validateTextureDescFieldRanges(const TextureDesc& desc)
        {
            if (desc.width == 0 || desc.height == 0 || desc.depthOrArrayLayers == 0 || desc.mipLevels == 0 ||
                !validTextureUsage(desc.usage) || vulkanFormat(desc.format) == VK_FORMAT_UNDEFINED)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture requires a valid dimension, non-zero extents and mip count, plus valid usage flags.",
                });
            }
            if (desc.sampleCount != 1 && desc.sampleCount != 2 && desc.sampleCount != 4 && desc.sampleCount != 8)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture sample count must be 1, 2, 4, or 8.",
                });
            }
            if (!validResidencyPriority(desc.residencyPriority))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The texture residency priority is invalid.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateShadingRateSource(const TextureDesc& desc, bool variableRateShadingTier2)
        {
            if (!hasAny(desc.usage, TextureUsage::shadingRateSource))
            {
                return {};
            }
            if (!variableRateShadingTier2)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Shading-rate images require primitive and attachment fragment shading rates.",
                });
            }
            if (desc.dimension != TextureDimension::texture2D || desc.depthOrArrayLayers != 1 || desc.mipLevels != 1 ||
                desc.sampleCount != 1 || desc.format != Format::r8Uint ||
                hasAny(desc.usage, TextureUsage::renderTarget | TextureUsage::depthStencil))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A shading-rate image must be a single-layer, single-mip, single-sample r8Uint 2D texture without "
                    "attachment usage.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateMultisamplingAndMipCount(const TextureDesc& desc)
        {
            if (desc.sampleCount != 1 &&
                (desc.dimension != TextureDimension::texture2D || desc.mipLevels != 1 ||
                 hasAny(desc.usage, TextureUsage::sampled | TextureUsage::storage) ||
                 (hasAny(desc.usage, TextureUsage::renderTarget) == hasAny(desc.usage, TextureUsage::depthStencil))))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A multisampled texture must be a single-mip 2D attachment with optional copy usage.",
                });
            }
            if (desc.mipLevels > fullMipCount(desc))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The texture mip count exceeds its dimensions.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateFormatUsage(const TextureDesc& desc)
        {
            if (isDepthFormat(desc.format) != hasAny(desc.usage, TextureUsage::depthStencil))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Depth formats and depth-stencil usage must be specified together.",
                });
            }
            if (isDepthFormat(desc.format) && (hasAny(desc.usage, TextureUsage::renderTarget | TextureUsage::storage) ||
                                               !detail::depthTextureSupportsDimension(desc.dimension)))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Depth textures cannot be render-target, storage, or 3D resources.",
                });
            }
            if (isCompressedFormat(desc.format) &&
                hasAny(desc.usage, TextureUsage::renderTarget | TextureUsage::storage | TextureUsage::depthStencil))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Block-compressed textures support copy and sampled usages only.",
                });
            }
            return {};
        }

        [[nodiscard]] std::uint32_t maxTextureExtent(TextureDimension dimension,
                                                     const VkPhysicalDeviceLimits& limits) noexcept
        {
            switch (dimension)
            {
            case TextureDimension::texture1D:
                return limits.maxImageDimension1D;
            case TextureDimension::texture3D:
                return limits.maxImageDimension3D;
            default:
                return limits.maxImageDimension2D;
            }
        }

        [[nodiscard]] Status validateTextureExtents(const TextureDesc& desc, const VkPhysicalDeviceLimits& limits)
        {
            if (desc.dimension == TextureDimension::texture1D && desc.height != 1)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A 1D texture requires height 1.",
                });
            }
            const auto maxExtent = maxTextureExtent(desc.dimension, limits);
            if (desc.width > maxExtent || desc.height > maxExtent ||
                (desc.dimension == TextureDimension::texture3D && desc.depthOrArrayLayers > maxExtent) ||
                (desc.dimension != TextureDimension::texture3D && desc.depthOrArrayLayers > limits.maxImageArrayLayers))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The texture dimensions exceed the Vulkan device limits.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateAlternateViewFormats(const TextureDesc& desc)
        {
            for (std::size_t index = 0; index < desc.viewFormats.size(); ++index)
            {
                const auto viewFormat = desc.viewFormats[index];
                bool duplicate = false;
                for (std::size_t previous = 0; previous < index; ++previous)
                {
                    duplicate = duplicate || desc.viewFormats[previous] == viewFormat;
                }
                if (viewFormat == desc.format || duplicate || !isTextureViewFormatCompatible(desc.format, viewFormat) ||
                    vulkanFormat(viewFormat) == VK_FORMAT_UNDEFINED)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Alternate texture view formats must be unique, non-base members of one compatible family.",
                    });
                }
            }
            return {};
        }

        [[nodiscard]] FormatFeature requiredTextureFormatFeatures(const TextureDesc& desc) noexcept
        {
            FormatFeature features = FormatFeature::none;
            switch (desc.dimension)
            {
            case TextureDimension::texture1D:
                features |= FormatFeature::texture1D;
                break;
            case TextureDimension::texture3D:
                features |= FormatFeature::texture3D;
                break;
            case TextureDimension::texture2D:
                features |= FormatFeature::texture2D;
                break;
            }
            if (hasAny(desc.usage, TextureUsage::sampled))
            {
                features |= FormatFeature::sampled;
            }
            if (hasAny(desc.usage, TextureUsage::storage))
            {
                features |= FormatFeature::storageWrite;
            }
            if (hasAny(desc.usage, TextureUsage::renderTarget))
            {
                features |= FormatFeature::colorAttachment;
            }
            if (hasAny(desc.usage, TextureUsage::depthStencil))
            {
                features |= FormatFeature::depthStencilAttachment;
            }
            if (hasAny(desc.usage, TextureUsage::copySource))
            {
                features |= FormatFeature::copySource;
            }
            if (hasAny(desc.usage, TextureUsage::copyDestination))
            {
                features |= FormatFeature::copyDestination;
            }
            return features;
        }

        [[nodiscard]] Status validateTextureFormatSupport(const TextureDesc& desc,
                                                          const FormatProperties& properties,
                                                          FormatFeature declaredViewFeatures)
        {
            const auto available =
                properties.features | (declaredViewFeatures & (FormatFeature::sampled | FormatFeature::storageWrite));
            if (!hasAll(available, requiredTextureFormatFeatures(desc)) ||
                !supportsSampleCount(properties.sampleCounts, desc.sampleCount))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The Vulkan format does not support the requested texture usage, dimension, or sample count.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateImageFormatLimits(const VulkanInstanceTable& instanceTable,
                                                       VkPhysicalDevice physical,
                                                       const VkImageCreateInfo& createInfo,
                                                       const std::vector<VkFormat>& viewFormats)
        {
            const VkImageFormatListCreateInfo list{
                .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
                .pNext = {},
                .viewFormatCount = static_cast<std::uint32_t>(viewFormats.size()),
                .pViewFormats = viewFormats.data(),
            };
            const VkPhysicalDeviceImageFormatInfo2 info{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
                .pNext = &list,
                .format = createInfo.format,
                .type = createInfo.imageType,
                .tiling = createInfo.tiling,
                .usage = createInfo.usage,
                .flags = createInfo.flags,
            };
            VkImageFormatProperties2 properties{
                .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
                .pNext = {},
                .imageFormatProperties = {},
            };
            const auto queried = instanceTable.vkGetPhysicalDeviceImageFormatProperties2(physical, &info, &properties);
            if (queried == VK_ERROR_FORMAT_NOT_SUPPORTED)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The Vulkan format does not support the requested texture usage, dimension, or sample count.",
                });
            }
            if (queried != VK_SUCCESS)
            {
                return std::unexpected(makeVulkanError(
                    vulkanResultErrorCode(queried), queried, "vkGetPhysicalDeviceImageFormatProperties2"));
            }
            const auto& limits = properties.imageFormatProperties;
            if (createInfo.extent.width > limits.maxExtent.width ||
                createInfo.extent.height > limits.maxExtent.height ||
                createInfo.extent.depth > limits.maxExtent.depth || createInfo.mipLevels > limits.maxMipLevels ||
                createInfo.arrayLayers > limits.maxArrayLayers || (limits.sampleCounts & createInfo.samples) == 0)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The Vulkan format does not support the requested texture extent, mip count, array layer count, "
                    "or sample count with its declared view formats.",
                });
            }
            return {};
        }

    } // namespace

    Status VulkanResourceAllocator::validateTextureDesc(const TextureDesc& desc) const
    {
        if (auto fieldRanges = validateTextureDescFieldRanges(desc); !fieldRanges)
        {
            return fieldRanges;
        }
        if (auto residency = validateResidencyRequest(desc.residencyPriority, desc.allocationFlags); !residency)
        {
            return residency;
        }
        if (auto shadingRate = validateShadingRateSource(desc, variableRateShadingTier2_); !shadingRate)
        {
            return shadingRate;
        }
        if (auto multisamplingAndMips = validateMultisamplingAndMipCount(desc); !multisamplingAndMips)
        {
            return multisamplingAndMips;
        }
        if (auto formatUsage = validateFormatUsage(desc); !formatUsage)
        {
            return formatUsage;
        }
        if (auto extents = validateTextureExtents(desc, limits_); !extents)
        {
            return extents;
        }
        if (auto viewFormats = validateAlternateViewFormats(desc); !viewFormats)
        {
            return viewFormats;
        }
        const bool checkDeclaredViewFormats = !desc.viewFormats.empty() && instanceTable_ != nullptr &&
                                              physical_ != VK_NULL_HANDLE &&
                                              instanceTable_->vkGetPhysicalDeviceImageFormatProperties2 != nullptr;
        FormatFeature declaredViewFeatures = FormatFeature::none;
        if (checkDeclaredViewFormats)
        {
            for (const auto viewFormat : desc.viewFormats)
            {
                declaredViewFeatures |= queryFormatProperties(viewFormat).features;
            }
        }
        if (auto support = validateTextureFormatSupport(desc, queryFormatProperties(desc.format), declaredViewFeatures);
            !support)
        {
            return support;
        }
        if (!checkDeclaredViewFormats)
        {
            return {};
        }
        std::vector<VkFormat> nativeViewFormats;
        VulkanSharingFamilies sharing{};
        auto createInfo = imageCreateInfo(desc, nativeViewFormats, sharing);
        if (!createInfo)
        {
            return std::unexpected(std::move(createInfo.error()));
        }
        return validateImageFormatLimits(*instanceTable_, physical_, *createInfo, nativeViewFormats);
    }

    Result<VkImageCreateInfo> VulkanResourceAllocator::imageCreateInfo(const TextureDesc& desc,
                                                                       std::vector<VkFormat>& viewFormats,
                                                                       VulkanSharingFamilies& sharing) const
    {
        viewFormats.clear();
        viewFormats.push_back(vulkanFormat(desc.format));
        for (const auto format : desc.viewFormats)
        {
            viewFormats.push_back(vulkanFormat(format));
        }
        std::ranges::sort(std::next(viewFormats.begin()), viewFormats.end());
        VkImageCreateFlags flags = 0;
        if (!desc.viewFormats.empty())
        {
            flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
        }
        if (desc.dimension == TextureDimension::texture2D && desc.width == desc.height &&
            desc.depthOrArrayLayers >= 6 && desc.depthOrArrayLayers % 6 == 0 && desc.sampleCount == 1)
        {
            flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        }
        sharing = textureSharingFamilies(desc.usage);
        const auto sharingMode = sharing.sharingMode();
        return VkImageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = {},
            .flags = flags,
            .imageType = nativeImageType(desc.dimension),
            .format = vulkanFormat(desc.format),
            .extent =
                {
                    desc.width,
                    desc.dimension == TextureDimension::texture1D ? 1 : desc.height,
                    desc.dimension == TextureDimension::texture3D ? desc.depthOrArrayLayers : 1,
                },
            .mipLevels = desc.mipLevels,
            .arrayLayers = desc.dimension == TextureDimension::texture3D ? 1 : desc.depthOrArrayLayers,
            .samples = nativeSampleCount(desc.sampleCount),
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = nativeImageUsage(desc.usage),
            .sharingMode = sharingMode,
            .queueFamilyIndexCount = sharingMode == VK_SHARING_MODE_CONCURRENT ? sharing.count : 0U,
            .pQueueFamilyIndices = sharingMode == VK_SHARING_MODE_CONCURRENT ? sharing.indices.data() : nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
    }

    Result<VkMemoryRequirements2> VulkanResourceAllocator::queryNativeImageMemoryRequirements(
        const VkImageCreateInfo& createInfo, VkMemoryDedicatedRequirements& dedicated) const
    {
        if (device_ == VK_NULL_HANDLE || table_ == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Texture memory requirements require a live Vulkan device.",
            });
        }

        dedicated = VkMemoryDedicatedRequirements{
            .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
            .pNext = {},
            .prefersDedicatedAllocation = {},
            .requiresDedicatedAllocation = {},
        };
        VkMemoryRequirements2 requirements{
            .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
            .pNext = &dedicated,
            .memoryRequirements = {},
        };

        const auto getDeviceReqs = table_->vkGetDeviceImageMemoryRequirements != nullptr
                                       ? table_->vkGetDeviceImageMemoryRequirements
                                       : table_->vkGetDeviceImageMemoryRequirementsKHR;
        if (getDeviceReqs != nullptr)
        {
            const VkDeviceImageMemoryRequirements deviceInfo{
                .sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS,
                .pNext = {},
                .pCreateInfo = &createInfo,
                .planeAspect = VK_IMAGE_ASPECT_NONE,
            };
            getDeviceReqs(device_, &deviceInfo, &requirements);
            return requirements;
        }

        const auto getImageReqs2 = table_->vkGetImageMemoryRequirements2 != nullptr
                                       ? table_->vkGetImageMemoryRequirements2
                                       : table_->vkGetImageMemoryRequirements2KHR;
        if (table_->vkCreateImage == nullptr || table_->vkDestroyImage == nullptr ||
            (getImageReqs2 == nullptr && table_->vkGetImageMemoryRequirements == nullptr))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Texture memory requirements require a live Vulkan device.",
            });
        }

        VkImage image = VK_NULL_HANDLE;
        const auto created = table_->vkCreateImage(device_, &createInfo, nullptr, &image);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateImage"));
        }
        if (getImageReqs2 != nullptr)
        {
            const VkImageMemoryRequirementsInfo2 info{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
                .pNext = {},
                .image = image,
            };
            getImageReqs2(device_, &info, &requirements);
        }
        else
        {
            table_->vkGetImageMemoryRequirements(device_, image, &requirements.memoryRequirements);
        }
        table_->vkDestroyImage(device_, image, nullptr);
        return requirements;
    }

    Result<ResourceMemoryRequirements> VulkanResourceAllocator::queryTextureMemoryRequirements(
        const TextureDesc& desc) const
    {
        return queryTextureMemoryRequirements(desc, 0);
    }

    Result<ResourceMemoryRequirements> VulkanResourceAllocator::queryTextureMemoryRequirements(
        const TextureDesc& desc, VkImageCreateFlags additionalFlags) const
    {
        auto valid = validateTextureDesc(desc);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        std::vector<VkFormat> viewFormats;
        VulkanSharingFamilies sharing{};
        auto createInfo = imageCreateInfo(desc, viewFormats, sharing);
        if (!createInfo)
        {
            return std::unexpected(std::move(createInfo.error()));
        }
        createInfo->flags |= additionalFlags;
        VkImageFormatListCreateInfo list{
            .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
            .pNext = {},
            .viewFormatCount = static_cast<std::uint32_t>(viewFormats.size()),
            .pViewFormats = viewFormats.data(),
        };
        if (!desc.viewFormats.empty())
        {
            createInfo->pNext = &list;
        }
        VkMemoryDedicatedRequirements dedicated{};
        auto requirements = queryNativeImageMemoryRequirements(*createInfo, dedicated);
        if (!requirements)
        {
            return std::unexpected(std::move(requirements.error()));
        }
        return portablePlacementRequirements(requirements->memoryRequirements,
                                             dedicated.requiresDedicatedAllocation == VK_TRUE);
    }

    namespace
    {

        [[nodiscard]] std::shared_ptr<std::vector<VkImageLayout>> makeSubmittedLayouts(const TextureDesc& desc,
                                                                                       VkImageLayout layout)
        {
            const auto submittedLayers = desc.dimension == TextureDimension::texture3D ? 1U : desc.depthOrArrayLayers;
            return std::make_shared<std::vector<VkImageLayout>>(
                static_cast<std::size_t>(desc.mipLevels) * submittedLayers * 2U, layout);
        }

    } // namespace

    Result<TextureHandle> VulkanResourceAllocator::createTexture(const TextureDesc& desc)
    {
        auto valid = validateTextureDesc(desc);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        TextureRecord record{};
        record.viewFormats.assign(desc.viewFormats.begin(), desc.viewFormats.end());
        record.desc = desc;
        record.label = std::string(desc.label);
        record.desc.viewFormats = record.viewFormats;
        std::vector<VkFormat> nativeViewFormats;
        auto createInfo = imageCreateInfo(record.desc, nativeViewFormats, record.sharingFamilies);
        if (!createInfo)
        {
            return std::unexpected(std::move(createInfo.error()));
        }
        VkImageFormatListCreateInfo list{
            .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
            .pNext = {},
            .viewFormatCount = static_cast<std::uint32_t>(nativeViewFormats.size()),
            .pViewFormats = nativeViewFormats.data(),
        };
        if (!record.viewFormats.empty())
        {
            createInfo->pNext = &list;
        }
        auto allocationCreate = imageAllocationCreateInfo(
            *createInfo, MemoryType::deviceLocal, desc.residencyPriority, texturePoolClass(desc.usage));
        if (!allocationCreate)
        {
            return std::unexpected(std::move(allocationCreate.error()));
        }
        record.keepAlive = makeVulkanNativeKeepAlive(VulkanNativeKeepAlive::Kind::image, VK_NULL_HANDLE, nullptr);
        record.keepAlive->allocator = allocator_;
        const auto created = vmaCreateImage(
            allocator_, &*createInfo, &*allocationCreate, &record.image, &record.allocation, &record.allocationInfo);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(allocationErrorCode(created), created, "vmaCreateImage"));
        }
        record.keepAlive->image = record.image;
        record.keepAlive->allocation = record.allocation;
        record.allocationId = nextAllocationId_.fetch_add(1, std::memory_order_relaxed);
        record.allocationBytes = record.allocationInfo.size;
        if (!record.label.empty())
        {
            vmaSetAllocationName(allocator_, record.allocation, record.label.c_str());
            setVulkanObjectName(instanceTable_,
                                device_,
                                VK_OBJECT_TYPE_IMAGE,
                                reinterpret_cast<std::uint64_t>(record.image),
                                record.label.c_str());
        }
        record.submittedLayouts = makeSubmittedLayouts(record.desc, record.layout);
        const std::scoped_lock lock(mutex_);
        trackKeepAliveLocked(record.keepAlive);
        return emplaceRecordLocked(textures_, std::move(record));
    }

    namespace
    {

        [[nodiscard]] VkResult createExternalMemoryImage(const VolkDeviceTable& table,
                                                         VkDevice device,
                                                         VkImageCreateInfo createInfo,
                                                         std::span<const VkFormat> declaredViewFormats,
                                                         VkImage& image)
        {
            const VkExternalMemoryImageCreateInfo external{
                .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
                .pNext = {},
                .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
            };
            const VkImageFormatListCreateInfo list{
                .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
                .pNext = &external,
                .viewFormatCount = static_cast<std::uint32_t>(declaredViewFormats.size()),
                .pViewFormats = declaredViewFormats.data(),
            };
            createInfo.pNext =
                declaredViewFormats.empty() ? static_cast<const void*>(&external) : static_cast<const void*>(&list);
            return table.vkCreateImage(device, &createInfo, nullptr, &image);
        }

        [[nodiscard]] VkMemoryRequirements queryMemoryRequirementsForImage(const VolkDeviceTable& table,
                                                                           VkDevice device,
                                                                           VkImage image)
        {
            VkMemoryDedicatedRequirements dedicated{
                .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
                .pNext = {},
                .prefersDedicatedAllocation = {},
                .requiresDedicatedAllocation = {},
            };
            VkMemoryRequirements2 requirements{
                .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
                .pNext = &dedicated,
                .memoryRequirements = {},
            };
            const VkImageMemoryRequirementsInfo2 requirementInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
                .pNext = {},
                .image = image,
            };
            table.vkGetImageMemoryRequirements2(device, &requirementInfo, &requirements);
            return requirements.memoryRequirements;
        }

    } // namespace

    Status VulkanResourceAllocator::validateSharedTextureCreation(const TextureDesc& desc) const
    {
        if (!sharedHandles_)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "createSharedTexture is not implemented on the Vulkan backend yet.",
            });
        }
        if (auto sharedValid = validateSharedTextureDesc(desc); !sharedValid)
        {
            return sharedValid;
        }
        if (auto valid = validateTextureDesc(desc); !valid)
        {
            return valid;
        }
        if (table_ == nullptr || table_->vkCreateImage == nullptr || table_->vkGetImageMemoryRequirements2 == nullptr ||
            table_->vkAllocateMemory == nullptr || table_->vkBindImageMemory == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Shared texture creation requires a live Vulkan device table.",
            });
        }
        return {};
    }

    Result<VmaAllocationInfo> VulkanResourceAllocator::backImageWithExportableMemory(VkImage image)
    {
        const auto requirements = queryMemoryRequirementsForImage(*table_, device_, image);
        auto memoryType = findMemoryType(requirements.memoryTypeBits, MemoryType::deviceLocal, requirements.size);
        if (!memoryType)
        {
            return std::unexpected(std::move(memoryType.error()));
        }

        const DedicatedMemoryRequest request{
            .buffer = VK_NULL_HANDLE,
            .image = image,
            .allocationSize = requirements.size,
            .memoryTypeIndex = *memoryType,
            .deviceAddress = false,
        };
        VkDeviceMemory memory = VK_NULL_HANDLE;
        const auto allocated = allocateExportableMemory(*table_, device_, request, memory);
        if (allocated != VK_SUCCESS)
        {
            return std::unexpected(
                makeVulkanError(allocationErrorCode(allocated), allocated, "vkAllocateMemory(shared image)"));
        }
        const auto bound = table_->vkBindImageMemory(device_, image, memory, 0);
        if (bound != VK_SUCCESS)
        {
            table_->vkFreeMemory(device_, memory, nullptr);
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(bound), bound, "vkBindImageMemory(shared)"));
        }
        return dedicatedAllocationInfo(memory, *memoryType, requirements.size);
    }

    Result<TextureHandle> VulkanResourceAllocator::createSharedTexture(const TextureDesc& desc)
    {
        if (auto creatable = validateSharedTextureCreation(desc); !creatable)
        {
            return std::unexpected(std::move(creatable.error()));
        }
        TextureRecord record{};
        record.viewFormats.assign(desc.viewFormats.begin(), desc.viewFormats.end());
        record.desc = desc;
        record.label = std::string(desc.label);
        record.desc.viewFormats = record.viewFormats;
        record.shared = true;
        record.dedicatedAllocation = true;
        std::vector<VkFormat> nativeViewFormats;
        auto createInfo = imageCreateInfo(record.desc, nativeViewFormats, record.sharingFamilies);
        if (!createInfo)
        {
            return std::unexpected(std::move(createInfo.error()));
        }
        const auto declaredViewFormats =
            record.viewFormats.empty() ? std::span<const VkFormat>{} : std::span<const VkFormat>{nativeViewFormats};
        record.keepAlive = makeVulkanNativeKeepAlive(VulkanNativeKeepAlive::Kind::image, device_, table_);
        const auto created =
            createExternalMemoryImage(*table_, device_, *createInfo, declaredViewFormats, record.image);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(allocationErrorCode(created), created, "vkCreateImage(shared)"));
        }
        record.keepAlive->image = record.image;
        auto allocation = backImageWithExportableMemory(record.image);
        if (!allocation)
        {
            return std::unexpected(std::move(allocation.error()));
        }
        record.keepAlive->memory = allocation->deviceMemory;

        record.allocationId = nextAllocationId_.fetch_add(1, std::memory_order_relaxed);
        record.allocationInfo = *allocation;
        record.allocationBytes = allocation->size;
        if (!record.label.empty())
        {
            setVulkanObjectName(instanceTable_,
                                device_,
                                VK_OBJECT_TYPE_IMAGE,
                                reinterpret_cast<std::uint64_t>(record.image),
                                record.label.c_str());
        }
        countExportedMemory(*record.keepAlive,
                            exportedMemory_,
                            memory_.memoryTypes[allocation->memoryType].heapIndex,
                            record.allocationBytes);
        record.submittedLayouts = makeSubmittedLayouts(record.desc, record.layout);
        const std::scoped_lock lock(mutex_);
        trackKeepAliveLocked(record.keepAlive);
        return emplaceRecordLocked(textures_, std::move(record));
    }

    Result<SharedHandle> VulkanResourceAllocator::exportSharedTexture(TextureHandle texture)
    {
#ifndef _WIN32
        static_cast<void>(texture);
        return std::unexpected(Error{
            ErrorCode::unsupported,
            0,
            "exportSharedTexture uses Windows NT handles. Linux FD sharing is not implemented.",
        });
#else
        if (!sharedHandles_ || table_ == nullptr || table_->vkGetMemoryWin32HandleKHR == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "exportSharedTexture is not implemented on the Vulkan backend yet.",
            });
        }
        const std::scoped_lock lock(mutex_);
        auto* record = textures_.find(texture);
        if (record == nullptr || !record->alive)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "exportSharedTexture received a stale or foreign texture handle.",
            });
        }
        if (!record->shared)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Only textures created or imported through the shared-texture API can be exported.",
            });
        }
        if (record->imported)
        {
            return reexportImportIdentity(record->exportIdentity,
                                          "DuplicateHandle failed for a re-exported shared texture.");
        }
        auto* const memory = exportableMemory(record->keepAlive, record->allocationInfo.deviceMemory);
        if (memory == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "exportSharedTexture could not resolve the dedicated Vulkan device memory.",
            });
        }
        auto identityDesc = record->desc;
        identityDesc.viewFormats = {};
        identityDesc.label = {};
        return exportSharedMemory(*table_,
                                  device_,
                                  memory,
                                  SharedExportIdentity{
                                      .texture = true,
                                      .textureDesc = identityDesc,
                                      .textureViewFormats = record->viewFormats,
                                      .deviceUuid = deviceUuid_,
                                      .memoryTypeIndex = record->allocationInfo.memoryType,
                                      .allocationBytes = record->allocationBytes,
                                  },
                                  record->exportIdentity,
                                  "DuplicateHandle failed for a shared texture export.");
#endif
    }

#ifdef _WIN32
    namespace
    {

        [[nodiscard]] Result<std::optional<ExportedMemoryLayout>> findLocalTextureExportLayout(
            HANDLE handle, const TextureDesc& desc, const std::array<std::uint8_t, VK_UUID_SIZE>& deviceUuid)
        {
            const std::scoped_lock exports(sharedExportMutex);
            const auto* identity = findSharedExport(handle);
            if (identity == nullptr)
            {
                return std::optional<ExportedMemoryLayout>{};
            }
            auto exportedDesc = identity->textureDesc;
            exportedDesc.viewFormats = identity->textureViewFormats;
            if (!identity->texture || identity->deviceUuid != deviceUuid || !matchingSharedTexture(exportedDesc, desc))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The imported texture description or usage does not match the native Vulkan resource.",
                });
            }
            return std::optional<ExportedMemoryLayout>{ExportedMemoryLayout{
                .memoryTypeIndex = identity->memoryTypeIndex,
                .allocationBytes = identity->allocationBytes,
            }};
        }

    } // namespace

    Status VulkanResourceAllocator::validateSharedTextureImport(SharedHandleView handle, const TextureDesc& desc) const
    {
        if (!sharedHandles_ || table_ == nullptr || table_->vkAllocateMemory == nullptr ||
            table_->vkCreateImage == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "importSharedTexture is not implemented on the Vulkan backend yet.",
            });
        }
        if (!handle.isValid() || handle.type != SharedHandleType::windowsNt)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Shared texture import requires a valid Windows NT handle.",
            });
        }
        if (desc.optimizedClearValue.kind != OptimizedClearValueKind::automatic)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Imported textures must use automatic optimized-clear metadata.",
            });
        }
        if (auto sharedValid = validateSharedTextureDesc(desc); !sharedValid)
        {
            return sharedValid;
        }
        return validateTextureDesc(desc);
    }

    Result<VmaAllocationInfo> VulkanResourceAllocator::backImageWithImportedMemory(VkImage image,
                                                                                   const SharedMemoryImport& source)
    {
        const auto requirements = queryMemoryRequirementsForImage(*table_, device_, image);
        Result<std::uint32_t> memoryType = std::unexpected(Error{
            ErrorCode::unsupported,
            0,
            "Opaque Win32 image import could not select a compatible memory type.",
        });
        if (source.localExportLayout.has_value())
        {
            if ((requirements.memoryTypeBits & (1U << source.localExportLayout->memoryTypeIndex)) != 0)
            {
                memoryType = source.localExportLayout->memoryTypeIndex;
            }
        }
        else
        {
            memoryType = findMemoryType(requirements.memoryTypeBits, MemoryType::deviceLocal, requirements.size);
        }
        if (!memoryType)
        {
            return std::unexpected(std::move(memoryType.error()));
        }

        const DedicatedMemoryRequest request{
            .buffer = VK_NULL_HANDLE,
            .image = image,
            .allocationSize =
                source.localExportLayout.has_value() ? source.localExportLayout->allocationBytes : requirements.size,
            .memoryTypeIndex = *memoryType,
            .deviceAddress = false,
        };
        VkDeviceMemory memory = VK_NULL_HANDLE;
        const auto allocated = importWin32Memory(*table_, device_, request, source.handle, memory);
        if (allocated != VK_SUCCESS)
        {
            return std::unexpected(
                makeVulkanError(importedMemoryErrorCode(allocated), allocated, "vkAllocateMemory(shared import)"));
        }
        const auto bound = table_->vkBindImageMemory(device_, image, memory, 0);
        if (bound != VK_SUCCESS)
        {
            table_->vkFreeMemory(device_, memory, nullptr);
            return std::unexpected(
                makeVulkanError(vulkanResultErrorCode(bound), bound, "vkBindImageMemory(shared import)"));
        }
        return dedicatedAllocationInfo(memory, *memoryType, request.allocationSize);
    }
#endif

    Result<TextureHandle> VulkanResourceAllocator::importSharedTexture(SharedHandleView handle, const TextureDesc& desc)
    {
#ifndef _WIN32
        static_cast<void>(handle);
        static_cast<void>(desc);
        return std::unexpected(Error{
            ErrorCode::unsupported,
            0,
            "importSharedTexture uses Windows NT handles. Linux FD sharing is not implemented.",
        });
#else
        if (auto importable = validateSharedTextureImport(handle, desc); !importable)
        {
            return std::unexpected(std::move(importable.error()));
        }
        auto* const nativeHandle = std::bit_cast<HANDLE>(handle.value);
        auto localExportLayout = findLocalTextureExportLayout(nativeHandle, desc, deviceUuid_);
        if (!localExportLayout)
        {
            return std::unexpected(std::move(localExportLayout.error()));
        }

        TextureRecord record{};
        record.viewFormats.assign(desc.viewFormats.begin(), desc.viewFormats.end());
        record.desc = desc;
        record.label = std::string(desc.label);
        record.desc.viewFormats = record.viewFormats;
        std::vector<VkFormat> nativeViewFormats;
        auto createInfo = imageCreateInfo(record.desc, nativeViewFormats, record.sharingFamilies);
        if (!createInfo)
        {
            return std::unexpected(std::move(createInfo.error()));
        }
        const auto declaredViewFormats =
            record.viewFormats.empty() ? std::span<const VkFormat>{} : std::span<const VkFormat>{nativeViewFormats};
        record.keepAlive = makeVulkanNativeKeepAlive(VulkanNativeKeepAlive::Kind::image, device_, table_);
        const auto created =
            createExternalMemoryImage(*table_, device_, *createInfo, declaredViewFormats, record.image);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(
                makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateImage(shared import)"));
        }
        record.keepAlive->image = record.image;
        const SharedMemoryImport source{
            .handle = nativeHandle,
            .localExportLayout = *localExportLayout,
        };
        auto allocation = backImageWithImportedMemory(record.image, source);
        if (!allocation)
        {
            return std::unexpected(std::move(allocation.error()));
        }
        record.keepAlive->memory = allocation->deviceMemory;
        auto identity = duplicateProcessHandle(nativeHandle, "DuplicateHandle failed for an imported shared texture.");
        if (!identity)
        {
            return std::unexpected(std::move(identity.error()));
        }
        auto identityOwner = ownImportIdentity(*identity);

        SharedExportIdentity entry{
            .handle = *identity,
            .texture = true,
            .textureDesc = desc,
            .textureViewFormats = record.viewFormats,
            .deviceUuid = deviceUuid_,
            .memoryTypeIndex = allocation->memoryType,
            .allocationBytes = allocation->size,
        };
        entry.textureDesc.viewFormats = {};
        entry.textureDesc.label = {};
        record.shared = true;
        record.imported = true;
        record.dedicatedAllocation = true;
        record.allocationInfo = *allocation;
        record.allocationBytes = allocation->size;
        record.exportIdentity = *identity;
        record.submittedLayouts = makeSubmittedLayouts(record.desc, record.layout);
        const std::scoped_lock lock(mutex_);
        trackKeepAliveLocked(record.keepAlive);
        return publishImportIdentity(
            [&] { return emplaceRecordLocked(textures_, std::move(record)); }, std::move(entry), identityOwner);
#endif
    }

    void VulkanSwapchainImageLease::beginAcquisition() noexcept
    {
        const std::scoped_lock lock(mutex_);
        acquired_ = true;
        submitted_ = false;
    }

    Status VulkanSwapchainImageLease::cancel()
    {
        const std::scoped_lock lock(mutex_);
        if (!acquired_)
        {
            return std::unexpected(Error{ErrorCode::invalidState, 0, "The swapchain image is not acquired."});
        }
        if (recordedReferenceCount_ != 0 || submitted_)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A recorded or submitted command buffer references the acquired swapchain image.",
            });
        }
        acquired_ = false;
        return {};
    }

    Status VulkanSwapchainImageLease::beginPresentation()
    {
        const std::scoped_lock lock(mutex_);
        if (!acquired_)
        {
            return std::unexpected(Error{ErrorCode::invalidState, 0, "The swapchain image is not acquired."});
        }
        if (recordedReferenceCount_ != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Presentation requires every command buffer that records the acquired swapchain image to be "
                "submitted.",
            });
        }
        acquired_ = false;
        return {};
    }

    void VulkanSwapchainImageLease::abortPresentation() noexcept
    {
        const std::scoped_lock lock(mutex_);
        acquired_ = true;
    }

    Status VulkanSwapchainImageLease::retainRecordedReference()
    {
        const std::scoped_lock lock(mutex_);
        if (!acquired_)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A swapchain image can be recorded only while it is acquired.",
            });
        }
        ++recordedReferenceCount_;
        return {};
    }

    void VulkanSwapchainImageLease::releaseRecordedReference() noexcept
    {
        const std::scoped_lock lock(mutex_);
        if (recordedReferenceCount_ != 0)
        {
            --recordedReferenceCount_;
        }
    }

    void VulkanSwapchainImageLease::markSubmitted() noexcept
    {
        const std::scoped_lock lock(mutex_);
        if (recordedReferenceCount_ != 0)
        {
            --recordedReferenceCount_;
        }
        if (acquired_)
        {
            submitted_ = true;
        }
    }

    Result<TextureHandle> VulkanResourceAllocator::adoptSwapchainImage(VkImage image,
                                                                       const TextureDesc& desc,
                                                                       std::shared_ptr<VulkanSwapchainImageLease> lease)
    {
        if (image == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A swapchain image adopt requires a live VkImage.",
            });
        }
        TextureRecord record{};
        record.image = image;
        record.desc = desc;
        record.label = std::string(desc.label);
        record.sharingFamilies.add(families_.graphics);
        record.swapchainBorrowed = true;
        record.submittedLayouts = makeSubmittedLayouts(record.desc, record.layout);
        record.swapchainLease = std::move(lease);
        const std::scoped_lock lock(mutex_);
        return emplaceRecordLocked(textures_, std::move(record));
    }

    Status VulkanResourceAllocator::forgetSwapchainImage(TextureHandle texture)
    {
        const std::scoped_lock lock(mutex_);
        auto* record = textures_.find(texture);
        if (record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "forgetSwapchainImage received a stale or foreign texture handle.",
            });
        }
        if (!record->swapchainBorrowed)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "forgetSwapchainImage requires a borrowed swapchain image.",
            });
        }
        if (record->liveViews != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A swapchain image with live views cannot be forgotten.",
            });
        }
        record->alive = false;
        record->image = VK_NULL_HANDLE;
        textures_.erase(texture);
        return {};
    }

    void VulkanResourceAllocator::retireSwapchainImage(TextureHandle texture) noexcept
    {
        const std::scoped_lock lock(mutex_);
        auto* record = textures_.find(texture);
        if (record == nullptr || !record->swapchainBorrowed)
        {
            return;
        }
        views_.forEach([texture](TextureViewHandle, TextureViewRecord& view) {
            if (view.texture == texture)
            {
                view.alive = false;
                view.view = VK_NULL_HANDLE;
                view.keepAlive.reset();
                view.textureKeepAlive.reset();
            }
        });
        record->alive = false;
        record->image = VK_NULL_HANDLE;
        textures_.erase(texture);
    }

    bool VulkanResourceAllocator::hasOtherTextureViews(TextureHandle texture, TextureViewHandle view) const noexcept
    {
        const std::scoped_lock lock(mutex_);
        const auto* record = textures_.find(texture);
        if (record == nullptr)
        {
            return false;
        }
        const auto* own = views_.find(view);
        const std::uint32_t ownViews = own != nullptr && own->texture == texture ? 1U : 0U;
        return record->liveViews > ownViews;
    }

    Status VulkanResourceAllocator::destroyTexture(TextureHandle texture)
    {
        const std::scoped_lock lock(mutex_);
        auto* record = textures_.find(texture);
        if (record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "destroyTexture received a stale or foreign texture handle.",
            });
        }
        if (record->swapchainBorrowed)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Borrowed swapchain images cannot be destroyed directly.",
            });
        }
        if (record->liveViews != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A texture with live views cannot be destroyed.",
            });
        }
        closeExportIdentity(record->exportIdentity);
        record->alive = false;
        textures_.erase(texture);
        return {};
    }

    namespace
    {

        [[nodiscard]] Status validateViewFormatDeclared(Format textureFormat,
                                                        const std::vector<Format>& alternateFormats,
                                                        Format viewFormat)
        {
            if (viewFormat == textureFormat)
            {
                return {};
            }
            const auto allowed = std::ranges::find(alternateFormats, viewFormat);
            if (allowed == alternateFormats.end() || !isTextureViewFormatCompatible(textureFormat, viewFormat))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture view format must be the base format or a declared compatible alternate.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateViewComponentSwizzle(const TextureViewDesc& desc, bool componentSwizzleSupported)
        {
            if (isIdentityLikeComponentMapping(desc.components))
            {
                return {};
            }
            if (desc.usage != TextureViewUsage::sampled)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Component swizzle is valid only on sampled texture views.",
                });
            }
            if (!componentSwizzleSupported)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "This device does not support texture view component swizzle.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateViewUsageDeclared(TextureViewUsage viewUsage, TextureUsage textureUsage)
        {
            switch (viewUsage)
            {
            case TextureViewUsage::sampled:
                if (!hasAny(textureUsage, TextureUsage::sampled))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A sampled view requires sampled texture usage.",
                    });
                }
                break;
            case TextureViewUsage::storage:
                if (!hasAny(textureUsage, TextureUsage::storage))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A storage view requires storage texture usage.",
                    });
                }
                break;
            case TextureViewUsage::renderTarget:
                if (!hasAny(textureUsage, TextureUsage::renderTarget))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A render-target view requires render-target texture usage.",
                    });
                }
                break;
            case TextureViewUsage::depthStencil:
                if (!hasAny(textureUsage, TextureUsage::depthStencil))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A depth-stencil view requires depth-stencil texture usage.",
                    });
                }
                break;
            }
            return {};
        }

        [[nodiscard]] Status validateViewDimension(const TextureViewDesc& desc, const TextureDesc& texture)
        {
            if (!dimensionAllowed(texture.dimension, desc.dimension))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The texture view dimension is incompatible with the texture dimension.",
                });
            }
            if ((desc.dimension == TextureViewDimension::cube || desc.dimension == TextureViewDimension::cubeArray) &&
                (desc.usage != TextureViewUsage::sampled || texture.width != texture.height))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Cube views require a sampled square 2D texture.",
                });
            }
            return {};
        }

        [[nodiscard]] FormatFeature requiredViewFormatFeature(TextureViewUsage usage) noexcept
        {
            switch (usage)
            {
            case TextureViewUsage::storage:
                return FormatFeature::storageWrite;
            case TextureViewUsage::renderTarget:
                return FormatFeature::colorAttachment;
            case TextureViewUsage::depthStencil:
                return FormatFeature::depthStencilAttachment;
            case TextureViewUsage::sampled:
                break;
            }
            return FormatFeature::sampled;
        }

        [[nodiscard]] Status validateViewFormatSupport(TextureViewUsage usage,
                                                       Format format,
                                                       const FormatProperties& properties)
        {
            if (!hasAny(properties.features, requiredViewFormatFeature(usage)))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The texture view format does not support the requested usage.",
                });
            }
            if ((usage == TextureViewUsage::storage || usage == TextureViewUsage::renderTarget) &&
                (isDepthFormat(format) || isCompressedFormat(format)))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Storage and render-target views require an uncompressed color format.",
                });
            }
            if (usage == TextureViewUsage::depthStencil && !isDepthFormat(format))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A depth-stencil view requires a depth format.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateViewAspect(TextureAspect aspect, Format textureFormat)
        {
            if ((aspect == TextureAspect::stencil && textureFormat != Format::d24UnormS8Uint) ||
                (aspect == TextureAspect::depth && !isDepthFormat(textureFormat)) ||
                (aspect == TextureAspect::color && isDepthFormat(textureFormat)))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The texture view aspect does not match the texture format.",
                });
            }
            return {};
        }

        [[nodiscard]] Result<std::uint32_t> resolveViewMipCount(const TextureViewDesc& desc, const TextureDesc& texture)
        {
            const bool singleMip = desc.usage != TextureViewUsage::sampled;
            std::uint32_t mipCount = desc.mipLevelCount;
            if (mipCount == remainingSubresources)
            {
                mipCount = singleMip ? 1U : texture.mipLevels - std::min(desc.firstMipLevel, texture.mipLevels);
            }
            if (desc.firstMipLevel >= texture.mipLevels || mipCount == 0 ||
                mipCount > texture.mipLevels - desc.firstMipLevel || (singleMip && mipCount != 1))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The texture view mip range is invalid for its usage.",
                });
            }
            return mipCount;
        }

        [[nodiscard]] Result<std::uint32_t> resolveViewLayerCount(const TextureViewDesc& desc,
                                                                  const TextureDesc& texture)
        {
            const auto textureArraySize =
                texture.dimension == TextureDimension::texture3D ? 1U : texture.depthOrArrayLayers;
            const bool singleLayer = desc.dimension == TextureViewDimension::texture1D ||
                                     desc.dimension == TextureViewDimension::texture2D ||
                                     desc.dimension == TextureViewDimension::texture3D;
            std::uint32_t defaultArrayCount = textureArraySize - std::min(desc.firstArrayLayer, textureArraySize);
            if (singleLayer)
            {
                defaultArrayCount = 1;
            }
            else if (desc.dimension == TextureViewDimension::cube)
            {
                defaultArrayCount = 6;
            }
            const auto layerCount =
                desc.arrayLayerCount == remainingSubresources ? defaultArrayCount : desc.arrayLayerCount;

            const bool outsideTexture = desc.firstArrayLayer >= textureArraySize || layerCount == 0 ||
                                        layerCount > textureArraySize - desc.firstArrayLayer;
            const bool singleLayerMismatch = singleLayer && (textureArraySize != 1 || layerCount != 1);
            const bool cubeMismatch =
                desc.dimension == TextureViewDimension::cube && (desc.firstArrayLayer != 0 || layerCount != 6);
            const bool cubeArrayMismatch = desc.dimension == TextureViewDimension::cubeArray &&
                                           (desc.firstArrayLayer % 6 != 0 || layerCount % 6 != 0);
            const bool volumeLayerOffset =
                texture.dimension == TextureDimension::texture3D && desc.firstArrayLayer != 0;
            if (outsideTexture || singleLayerMismatch || cubeMismatch || cubeArrayMismatch || volumeLayerOffset)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The texture view array or cube-face range is invalid.",
                });
            }
            return layerCount;
        }

    } // namespace

    Result<VulkanResourceAllocator::ResolvedTextureView> VulkanResourceAllocator::resolveTextureView(
        const TextureRecord& texture, const TextureViewDesc& desc) const
    {
        if (!validComponentMapping(desc.components))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The texture view contains an invalid component swizzle.",
            });
        }
        const auto format = desc.format == Format::unknown ? texture.desc.format : desc.format;
        if (auto declared = validateViewFormatDeclared(texture.desc.format, texture.viewFormats, format); !declared)
        {
            return std::unexpected(std::move(declared.error()));
        }
        if (auto swizzle = validateViewComponentSwizzle(desc, textureViewComponentSwizzle_); !swizzle)
        {
            return std::unexpected(std::move(swizzle.error()));
        }
        if (auto usage = validateViewUsageDeclared(desc.usage, texture.desc.usage); !usage)
        {
            return std::unexpected(std::move(usage.error()));
        }
        if (auto dimension = validateViewDimension(desc, texture.desc); !dimension)
        {
            return std::unexpected(std::move(dimension.error()));
        }
        if (auto supported = validateViewFormatSupport(desc.usage, format, queryFormatProperties(format)); !supported)
        {
            return std::unexpected(std::move(supported.error()));
        }
        if (auto aspect = validateViewAspect(desc.aspect, texture.desc.format); !aspect)
        {
            return std::unexpected(std::move(aspect.error()));
        }
        auto mipCount = resolveViewMipCount(desc, texture.desc);
        if (!mipCount)
        {
            return std::unexpected(std::move(mipCount.error()));
        }
        auto layerCount = resolveViewLayerCount(desc, texture.desc);
        if (!layerCount)
        {
            return std::unexpected(std::move(layerCount.error()));
        }
        return ResolvedTextureView{
            .format = format,
            .mipCount = *mipCount,
            .layerCount = *layerCount,
        };
    }

    namespace
    {

        [[nodiscard]] VkImageViewType nativeViewType(TextureViewDimension dimension) noexcept
        {
            switch (dimension)
            {
            case TextureViewDimension::texture1D:
                return VK_IMAGE_VIEW_TYPE_1D;
            case TextureViewDimension::texture1DArray:
                return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
            case TextureViewDimension::texture2DArray:
                return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            case TextureViewDimension::texture3D:
                return VK_IMAGE_VIEW_TYPE_3D;
            case TextureViewDimension::cube:
                return VK_IMAGE_VIEW_TYPE_CUBE;
            case TextureViewDimension::cubeArray:
                return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
            case TextureViewDimension::texture2D:
                break;
            }
            return VK_IMAGE_VIEW_TYPE_2D;
        }

        [[nodiscard]] VkComponentMapping nativeComponentMapping(const ComponentMapping& components) noexcept
        {
            return VkComponentMapping{
                .r = nativeSwizzle(components.red, VK_COMPONENT_SWIZZLE_R),
                .g = nativeSwizzle(components.green, VK_COMPONENT_SWIZZLE_G),
                .b = nativeSwizzle(components.blue, VK_COMPONENT_SWIZZLE_B),
                .a = nativeSwizzle(components.alpha, VK_COMPONENT_SWIZZLE_A),
            };
        }

        [[nodiscard]] VkImageSubresourceRange viewSubresourceRange(const TextureViewDesc& desc,
                                                                   Format textureFormat,
                                                                   std::uint32_t mipCount,
                                                                   std::uint32_t layerCount) noexcept
        {
            const bool depthAndStencil = desc.usage == TextureViewUsage::depthStencil &&
                                         textureFormat == Format::d24UnormS8Uint &&
                                         desc.aspect != TextureAspect::stencil;
            return VkImageSubresourceRange{
                .aspectMask = depthAndStencil ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                                              : nativeAspect(desc.aspect, textureFormat),
                .baseMipLevel = desc.firstMipLevel,
                .levelCount = mipCount,
                .baseArrayLayer = desc.firstArrayLayer,
                .layerCount = desc.dimension == TextureViewDimension::texture3D ? 1 : layerCount,
            };
        }

    } // namespace

    Result<TextureViewHandle> VulkanResourceAllocator::createTextureView(const TextureViewDesc& desc)
    {
        const std::scoped_lock lock(mutex_);
        auto* texture = textures_.find(desc.texture);
        if (texture == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "createTextureView received a stale or foreign texture handle.",
            });
        }
        auto resolved = resolveTextureView(*texture, desc);
        if (!resolved)
        {
            return std::unexpected(std::move(resolved.error()));
        }

        TextureViewRecord record{};
        record.texture = desc.texture;
        record.desc = desc;
        record.resolved = *resolved;
        record.label = std::string(desc.label);
        const VkImageViewUsageCreateInfo usageInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
            .pNext = {},
            .usage = nativeViewUsage(desc.usage),
        };
        const VkImageViewCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = &usageInfo,
            .flags = {},
            .image = texture->image,
            .viewType = nativeViewType(desc.dimension),
            .format = vulkanFormat(resolved->format),
            .components = nativeComponentMapping(desc.components),
            .subresourceRange =
                viewSubresourceRange(desc, texture->desc.format, resolved->mipCount, resolved->layerCount),
        };
        record.keepAlive = makeVulkanNativeKeepAlive(VulkanNativeKeepAlive::Kind::view, device_, table_);
        const auto created = table_->vkCreateImageView(device_, &info, nullptr, &record.view);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateImageView"));
        }
        record.keepAlive->view = record.view;
        if (!record.label.empty())
        {
            setVulkanObjectName(instanceTable_,
                                device_,
                                VK_OBJECT_TYPE_IMAGE_VIEW,
                                reinterpret_cast<std::uint64_t>(record.view),
                                record.label.c_str());
        }
        record.keepAlive->parent = texture->keepAlive;
        record.textureKeepAlive = texture->keepAlive;
        trackKeepAliveLocked(record.keepAlive);
        auto handle = emplaceRecordLocked(views_, std::move(record));
        if (handle)
        {
            texture->liveViews += 1;
        }
        return handle;
    }

    Status VulkanResourceAllocator::destroyTextureView(TextureViewHandle view)
    {
        const std::scoped_lock lock(mutex_);
        auto* record = views_.find(view);
        if (record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "destroyTextureView received a stale or foreign view handle.",
            });
        }
        if (auto* texture = textures_.find(record->texture); texture != nullptr && texture->liveViews > 0)
        {
            texture->liveViews -= 1;
        }
        record->alive = false;
        views_.erase(view);
        return {};
    }

    Status VulkanResourceAllocator::validateSamplerDesc(const SamplerDesc& desc) const
    {
        if (desc.maxAnisotropy == 0 || desc.maxAnisotropy > 16 ||
            static_cast<float>(desc.maxAnisotropy) > limits_.maxSamplerAnisotropy)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Sampler maxAnisotropy must be in [1, min(16, device limit)].",
            });
        }
        if (!std::isfinite(desc.lodBias) || !std::isfinite(desc.minLod) ||
            (!std::isfinite(desc.maxLod) && desc.maxLod != std::numeric_limits<float>::max()) ||
            desc.minLod > desc.maxLod)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Sampler LOD values must be finite and minLod must not exceed maxLod.",
            });
        }
        if (std::fabs(desc.lodBias) > limits_.maxSamplerLodBias)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Sampler lodBias must be in [-maxSamplerLodBias, maxSamplerLodBias] of the device limits.",
            });
        }
        if (desc.mipFilter == MipmapFilterMode::none && desc.minLod != 0.0F)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A sampler without mip filtering cannot request a non-zero minLod.",
            });
        }
        if (desc.reductionMode != SamplerReductionMode::standard && desc.compareEnabled)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Sampler min/max reduction and comparison sampling are mutually exclusive.",
            });
        }
        if (desc.reductionMode != SamplerReductionMode::standard && !samplerMinMaxReduction_)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The device does not support min/max sampler reduction.",
            });
        }
        return {};
    }

    Result<SamplerHandle> VulkanResourceAllocator::createSampler(const SamplerDesc& desc)
    {
        auto valid = validateSamplerDesc(desc);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        SamplerRecord record{};
        record.desc = desc;
        record.label = std::string(desc.label);
        VkSamplerReductionModeCreateInfo reduction{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO,
            .pNext = {},
            .reductionMode = desc.reductionMode == SamplerReductionMode::minimum ? VK_SAMPLER_REDUCTION_MODE_MIN
                                                                                 : VK_SAMPLER_REDUCTION_MODE_MAX,
        };
        VkSamplerCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext = {},
            .flags = {},
            .magFilter = nativeSamplerFilter(desc, desc.magFilter),
            .minFilter = nativeSamplerFilter(desc, desc.minFilter),
            .mipmapMode = desc.mipFilter == MipmapFilterMode::linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                                                     : VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = nativeAddress(desc.addressU),
            .addressModeV = nativeAddress(desc.addressV),
            .addressModeW = nativeAddress(desc.addressW),
            .mipLodBias = desc.lodBias,
            .anisotropyEnable = desc.maxAnisotropy > 1 ? VK_TRUE : VK_FALSE,
            .maxAnisotropy = static_cast<float>(desc.maxAnisotropy),
            .compareEnable = desc.compareEnabled ? VK_TRUE : VK_FALSE,
            .compareOp = nativeCompare(desc.compare),
            .minLod = desc.minLod,
            .maxLod = nativeMaxLod(desc),
            .borderColor = nativeBorder(desc.borderColor),
            .unnormalizedCoordinates = VK_FALSE,
        };
        if (desc.reductionMode != SamplerReductionMode::standard)
        {
            info.pNext = &reduction;
        }
        record.keepAlive = makeVulkanNativeKeepAlive(VulkanNativeKeepAlive::Kind::sampler, device_, table_);
        const auto created = table_->vkCreateSampler(device_, &info, nullptr, &record.sampler);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateSampler"));
        }
        record.keepAlive->sampler = record.sampler;
        if (!record.label.empty())
        {
            setVulkanObjectName(instanceTable_,
                                device_,
                                VK_OBJECT_TYPE_SAMPLER,
                                reinterpret_cast<std::uint64_t>(record.sampler),
                                record.label.c_str());
        }
        const std::scoped_lock lock(mutex_);
        trackKeepAliveLocked(record.keepAlive);
        return emplaceRecordLocked(samplers_, std::move(record));
    }

    Status VulkanResourceAllocator::destroySampler(SamplerHandle sampler)
    {
        const std::scoped_lock lock(mutex_);
        auto* record = samplers_.find(sampler);
        if (record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "destroySampler received a stale or foreign sampler handle.",
            });
        }
        record->alive = false;
        samplers_.erase(sampler);
        return {};
    }

    Result<VulkanResourceAllocator::ResolvedTexture> VulkanResourceAllocator::resolveTexture(
        TextureHandle texture) const
    {
        const std::scoped_lock lock(mutex_);
        const auto* record = textures_.find(texture);
        if (record == nullptr || !record->alive)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The texture handle is invalid or stale.",
            });
        }
        return ResolvedTexture{
            .image = record->image,
            .desc = record->desc,
            .keepAlive = record->keepAlive,
        };
    }

    Result<VulkanResourceAllocator::ResolvedBindingBuffer> VulkanResourceAllocator::resolveBindingBuffer(
        BufferHandle buffer) const
    {
        const std::scoped_lock lock(mutex_);
        const auto* record = buffers_.find(buffer);
        if (record == nullptr || !record->alive)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A binding write references a stale or foreign buffer handle.",
            });
        }
        return ResolvedBindingBuffer{
            .buffer = record->buffer,
            .desc = record->desc,
            .keepAlive = record->keepAlive,
            .deviceAddress = record->deviceAddress,
        };
    }

    Result<VulkanResourceAllocator::ResolvedBindingTextureView> VulkanResourceAllocator::resolveBindingTextureView(
        TextureViewHandle view) const
    {
        const std::scoped_lock lock(mutex_);
        const auto* record = views_.find(view);
        if (record == nullptr || !record->alive)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A binding write references a stale or foreign texture view handle.",
            });
        }
        return ResolvedBindingTextureView{
            .view = record->view,
            .usage = record->desc.usage,
            .keepAlive = record->keepAlive,
        };
    }

    Result<VulkanResourceAllocator::ResolvedBindingSampler> VulkanResourceAllocator::resolveBindingSampler(
        SamplerHandle sampler) const
    {
        const std::scoped_lock lock(mutex_);
        const auto* record = samplers_.find(sampler);
        if (record == nullptr || !record->alive)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A binding write references a stale or foreign sampler handle.",
            });
        }
        return ResolvedBindingSampler{
            .sampler = record->sampler,
            .keepAlive = record->keepAlive,
        };
    }

    Result<VulkanResourceAllocator::ResolvedRenderAttachment> VulkanResourceAllocator::resolveRenderAttachment(
        TextureViewHandle view) const
    {
        const std::scoped_lock lock(mutex_);
        const auto* record = views_.find(view);
        if (record == nullptr || !record->alive)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A rendering attachment references a stale or foreign texture view handle.",
            });
        }
        const auto* texture = textures_.find(record->texture);
        if (texture == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A rendering attachment references a texture view whose source texture is gone.",
            });
        }
        const auto mip = record->desc.firstMipLevel;
        return ResolvedRenderAttachment{
            .view = record->view,
            .format = record->resolved.format,
            .width = std::max(1U, texture->desc.width >> mip),
            .height = std::max(1U, texture->desc.height >> mip),
            .sampleCount = texture->desc.sampleCount,
            .layerCount = record->resolved.layerCount,
            .usage = record->desc.usage,
            .keepAlive = record->keepAlive,
            .textureKeepAlive = record->textureKeepAlive,
            .swapchainLease = texture->swapchainLease,
        };
    }

    BackendAllocationSnapshot VulkanResourceAllocator::captureBackendAllocationSnapshot() const noexcept
    {
        const std::scoped_lock lock(mutex_);
        return BackendAllocationSnapshot{
            .liveDescriptorCount = views_.size() + samplers_.size(),
            .reservedDescriptorCount = views_.size() + samplers_.size(),
        };
    }

#ifdef RACHIS_TESTING
    namespace
    {

        struct ImmediateCopyFootprint final
        {
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            std::uint64_t requiredBytes = 0;
        };

        [[nodiscard]] ImmediateCopyFootprint immediateCopyFootprint(const TextureDesc& texture,
                                                                    std::uint32_t mipLevel) noexcept
        {
            const auto width = mipExtent(texture.width, mipLevel);
            const auto height =
                texture.dimension == TextureDimension::texture1D ? 1U : mipExtent(texture.height, mipLevel);
            return ImmediateCopyFootprint{
                .width = width,
                .height = height,
                .requiredBytes = static_cast<std::uint64_t>(width) * height * texelByteSize(texture.format),
            };
        }

        [[nodiscard]] bool isUncompressedColorSubresource(const TextureDesc& texture,
                                                          std::uint32_t mipLevel,
                                                          std::uint32_t arrayLayer) noexcept
        {
            const auto arrayLayers = texture.dimension == TextureDimension::texture3D ? 1U : texture.depthOrArrayLayers;
            return texelByteSize(texture.format) != 0 && mipLevel < texture.mipLevels && arrayLayer < arrayLayers;
        }

        void recordImageBarrier(const VolkDeviceTable& table,
                                VkCommandBuffer command,
                                const VkImageMemoryBarrier2& barrier)
        {
            const VkDependencyInfo dependency{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pNext = {},
                .dependencyFlags = {},
                .memoryBarrierCount = {},
                .pMemoryBarriers = {},
                .bufferMemoryBarrierCount = {},
                .pBufferMemoryBarriers = {},
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &barrier,
            };
            table.vkCmdPipelineBarrier2(command, &dependency);
        }

        [[nodiscard]] VkBufferImageCopy subresourceCopyRegion(VkImageAspectFlags aspect,
                                                              std::uint32_t mipLevel,
                                                              std::uint32_t arrayLayer,
                                                              const ImmediateCopyFootprint& footprint) noexcept
        {
            return VkBufferImageCopy{
                .bufferOffset = 0,
                .bufferRowLength = {},
                .bufferImageHeight = {},
                .imageSubresource =
                    {
                        .aspectMask = aspect,
                        .mipLevel = mipLevel,
                        .baseArrayLayer = arrayLayer,
                        .layerCount = 1,
                    },
                .imageOffset = {},
                .imageExtent = {footprint.width, footprint.height, 1},
            };
        }

        void recordHostReadBarrier(const VolkDeviceTable& table,
                                   VkCommandBuffer command,
                                   VkBuffer buffer,
                                   std::uint64_t size)
        {
            const VkBufferMemoryBarrier2 hostBarrier{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                .pNext = {},
                .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = buffer,
                .offset = 0,
                .size = size,
            };
            const VkDependencyInfo hostDependency{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pNext = {},
                .dependencyFlags = {},
                .memoryBarrierCount = {},
                .pMemoryBarriers = {},
                .bufferMemoryBarrierCount = 1,
                .pBufferMemoryBarriers = &hostBarrier,
                .imageMemoryBarrierCount = {},
                .pImageMemoryBarriers = {},
            };
            table.vkCmdPipelineBarrier2(command, &hostDependency);
        }

    } // namespace

    Status VulkanResourceAllocator::copyBufferToTextureImmediate(BufferHandle buffer,
                                                                 TextureHandle texture,
                                                                 std::uint32_t mipLevel,
                                                                 std::uint32_t arrayLayer)
    {
        const std::scoped_lock lock(mutex_);
        auto* bufferRecord = findBuffer(buffer);
        auto* textureRecord = textures_.find(texture);
        if (bufferRecord == nullptr || textureRecord == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Immediate buffer-to-texture copy received a stale handle.",
            });
        }
        const auto footprint = immediateCopyFootprint(textureRecord->desc, mipLevel);
        if (footprint.requiredBytes == 0 || footprint.requiredBytes > bufferRecord->desc.size ||
            !isUncompressedColorSubresource(textureRecord->desc, mipLevel, arrayLayer) ||
            !hasAny(textureRecord->desc.usage, TextureUsage::copyDestination) ||
            !hasAny(bufferRecord->desc.usage, BufferUsage::copySource))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Immediate buffer-to-texture copy requires an uncompressed color subresource and matching copy usage.",
            });
        }
        if (table_ == nullptr || table_->vkAllocateCommandBuffers == nullptr ||
            table_->vkBeginCommandBuffer == nullptr || table_->vkCmdPipelineBarrier2 == nullptr ||
            table_->vkCmdCopyBufferToImage == nullptr || table_->vkEndCommandBuffer == nullptr ||
            table_->vkQueueSubmit == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Immediate texture copy is unavailable because the Vulkan device table is incomplete.",
            });
        }

        auto command = beginImmediateTransfer(*table_, device_, transferPool_);
        if (!command)
        {
            return std::unexpected(std::move(command.error()));
        }

        const auto aspect = nativeAspect(TextureAspect::color, textureRecord->desc.format);
        const auto toDestination =
            imageLayoutBarrier(textureRecord->image,
                               recordedLayout(VK_NULL_HANDLE, *textureRecord, {mipLevel, arrayLayer}),
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_ACCESS_2_NONE,
                               VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               aspect,
                               mipLevel,
                               arrayLayer);
        recordImageBarrier(*table_, *command, toDestination);
        const auto region = subresourceCopyRegion(aspect, mipLevel, arrayLayer, footprint);
        table_->vkCmdCopyBufferToImage(
            *command, bufferRecord->buffer, textureRecord->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        const auto toSource = imageLayoutBarrier(textureRecord->image,
                                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                 VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                 VK_ACCESS_2_TRANSFER_READ_BIT,
                                                 aspect,
                                                 mipLevel,
                                                 arrayLayer);
        recordImageBarrier(*table_, *command, toSource);
        if (auto submitted = submitImmediateTransferAndWait(*table_, device_, transferPool_, transferQueue_, *command);
            !submitted)
        {
            return submitted;
        }
        std::vector<LayoutWrite> writes;
        appendLayoutWrites(*textureRecord,
                           toSource.subresourceRange,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           TextureLayout::copySource,
                           writes);
        for (const auto& write : writes)
        {
            applyCommittedLayout(write);
        }
        return {};
    }

    Status VulkanResourceAllocator::copyTextureToBufferImmediate(TextureHandle texture,
                                                                 BufferHandle buffer,
                                                                 std::uint32_t mipLevel,
                                                                 std::uint32_t arrayLayer)
    {
        const std::scoped_lock lock(mutex_);
        auto* bufferRecord = findBuffer(buffer);
        auto* textureRecord = textures_.find(texture);
        if (bufferRecord == nullptr || textureRecord == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Immediate texture-to-buffer copy received a stale handle.",
            });
        }
        const auto footprint = immediateCopyFootprint(textureRecord->desc, mipLevel);
        if (footprint.requiredBytes == 0 || footprint.requiredBytes > bufferRecord->desc.size ||
            !isUncompressedColorSubresource(textureRecord->desc, mipLevel, arrayLayer) ||
            !hasAny(textureRecord->desc.usage, TextureUsage::copySource) ||
            !hasAny(bufferRecord->desc.usage, BufferUsage::copyDestination))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Immediate texture-to-buffer copy requires an uncompressed color subresource and matching copy usage.",
            });
        }
        if (table_ == nullptr || table_->vkAllocateCommandBuffers == nullptr ||
            table_->vkBeginCommandBuffer == nullptr || table_->vkCmdPipelineBarrier2 == nullptr ||
            table_->vkCmdCopyImageToBuffer == nullptr || table_->vkEndCommandBuffer == nullptr ||
            table_->vkQueueSubmit == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Immediate texture copy is unavailable because the Vulkan device table is incomplete.",
            });
        }

        auto command = beginImmediateTransfer(*table_, device_, transferPool_);
        if (!command)
        {
            return std::unexpected(std::move(command.error()));
        }

        const auto aspect = nativeAspect(TextureAspect::color, textureRecord->desc.format);
        const auto previousLayout = recordedLayout(VK_NULL_HANDLE, *textureRecord, {mipLevel, arrayLayer});
        std::vector<LayoutWrite> writes;
        if (previousLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        {
            const auto toSource = imageLayoutBarrier(textureRecord->image,
                                                     previousLayout,
                                                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                     VK_ACCESS_2_NONE,
                                                     VK_ACCESS_2_TRANSFER_READ_BIT,
                                                     aspect,
                                                     mipLevel,
                                                     arrayLayer);
            recordImageBarrier(*table_, *command, toSource);
            appendLayoutWrites(*textureRecord,
                               toSource.subresourceRange,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               TextureLayout::copySource,
                               writes);
        }
        const auto region = subresourceCopyRegion(aspect, mipLevel, arrayLayer, footprint);
        table_->vkCmdCopyImageToBuffer(
            *command, textureRecord->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, bufferRecord->buffer, 1, &region);
        recordHostReadBarrier(*table_, *command, bufferRecord->buffer, footprint.requiredBytes);
        if (auto submitted = submitImmediateTransferAndWait(*table_, device_, transferPool_, transferQueue_, *command);
            !submitted)
        {
            return submitted;
        }
        for (const auto& write : writes)
        {
            applyCommittedLayout(write);
        }
        return {};
    }

    namespace
    {

        VulkanResourceAllocator* boundAllocator = nullptr;

    }

    void bindVulkanResourceAllocator(VulkanResourceAllocator* allocator) noexcept
    {
        boundAllocator = allocator;
    }

    void unbindVulkanResourceAllocator(VulkanResourceAllocator* allocator) noexcept
    {
        if (boundAllocator == allocator)
        {
            boundAllocator = nullptr;
        }
    }

    Status copyVulkanBuffersImmediate(BufferHandle source,
                                      BufferHandle destination,
                                      std::uint64_t sourceOffset,
                                      std::uint64_t destinationOffset,
                                      std::uint64_t size)
    {
        if (boundAllocator == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "No live Vulkan resource allocator is bound for immediate buffer copy.",
            });
        }
        return boundAllocator->copyBufferImmediate(source, destination, sourceOffset, destinationOffset, size);
    }

    Status copyVulkanBufferToTextureImmediate(BufferHandle buffer,
                                              TextureHandle texture,
                                              std::uint32_t mipLevel,
                                              std::uint32_t arrayLayer)
    {
        if (boundAllocator == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "No live Vulkan resource allocator is bound for immediate texture copy.",
            });
        }
        return boundAllocator->copyBufferToTextureImmediate(buffer, texture, mipLevel, arrayLayer);
    }

    Status copyVulkanTextureToBufferImmediate(TextureHandle texture,
                                              BufferHandle buffer,
                                              std::uint32_t mipLevel,
                                              std::uint32_t arrayLayer)
    {
        if (boundAllocator == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "No live Vulkan resource allocator is bound for immediate texture copy.",
            });
        }
        return boundAllocator->copyTextureToBufferImmediate(texture, buffer, mipLevel, arrayLayer);
    }

    Status recordVulkanFillBuffer(CommandBuffer& commandBuffer, BufferHandle buffer, std::uint32_t value)
    {
        if (boundAllocator == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "No live Vulkan resource allocator is bound for recorded buffer fill.",
            });
        }
        auto* backend = detail::CommandBufferAccess::get(commandBuffer);
        auto* native = backend != nullptr && backend->kind() == detail::BackendCommandBufferKind::vulkan
                           ? static_cast<VulkanCommandBuffer*>(backend)
                           : nullptr;
        if (native == nullptr || native->owner() == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A fill command requires a live Vulkan command-buffer lease.",
            });
        }
        const auto owner = native->ownerShared();
        const std::scoped_lock poolLock(owner->mutex);
        if (owner->pool == VK_NULL_HANDLE || native->native() == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A fill command requires a live Vulkan command-buffer lease.",
            });
        }
        std::vector<std::shared_ptr<void>>* retained = nullptr;
        if (!hasAny(owner->flags, CommandPoolFlag::unretainedResources))
        {
            retained = &native->retained();
        }
        return boundAllocator->recordFillBuffer(native->native(), buffer, value, retained);
    }
#endif

} // namespace rachis::vulkan
