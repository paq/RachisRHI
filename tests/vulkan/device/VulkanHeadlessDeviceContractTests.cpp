#include "ForceVulkanRequest.h"
#include "Internal/VulkanWindowsProfile.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <string>

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include <volk.h>

#include "Internal/VulkanRuntime.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string_view>
#include <vector>

namespace
{

    [[nodiscard]] rachis::VulkanInstanceReport completeInstance() noexcept
    {
        return {
            .apiMajor = 1,
            .apiMinor = 3,
            .khrSurface = true,
            .khrWin32Surface = true,
        };
    }

    [[nodiscard]] rachis::VulkanPhysicalDeviceReport completeDevice() noexcept
    {
        return {
            .apiMajor = 1,
            .apiMinor = 3,
            .driverId = "NVIDIA_PROPRIETARY",
            .dynamicRendering = true,
            .synchronization2 = true,
            .timelineSemaphore = true,
            .independentBlend = true,
            .samplerAnisotropy = true,
            .textureCompressionBC = true,
            .shaderSampledImageArrayDynamicIndexing = true,
            .timestampComputeAndGraphics = true,
            .hostQueryReset = true,
            .khrSwapchain = true,
            .formatRgba8Unorm = true,
            .formatRgba8UnormSrgb = true,
            .formatBgra8Unorm = true,
            .formatBgra8UnormSrgb = true,
            .formatD32Float = true,
            .formatD32FloatS8Uint = true,
            .formatRgba16Float = true,
            .formatR32Uint = true,
            .maxImageDimension2D = 8192,
            .maxBoundDescriptorSets = 4,
            .maxPushConstantsSize = 128,
            .maxColorAttachments = 8,
            .maxComputeWorkgroupInvocations = 256,
        };
    }

    struct FakePhysicalDevice final
    {
        std::uintptr_t handle = 0;
        std::array<std::uint8_t, VK_UUID_SIZE> uuid{};
        std::array<std::uint8_t, VK_LUID_SIZE> luid{};
        bool luidValid = false;
        VkDriverId driver = VK_DRIVER_ID_MESA_NVK;
        std::string_view name = "NVIDIA GeForce GTX 1080 (NVK GP104)";
        std::uint32_t apiVersion = VK_API_VERSION_1_3;
    };

    struct FakeChainQuery final
    {
        VkPhysicalDevice physical = VK_NULL_HANDLE;
        std::vector<VkStructureType> chain;
    };

    constexpr std::array<std::uint8_t, VK_UUID_SIZE> nvkUuidBus2d{
        0x34, 0x01, 0xde, 0x10, 0x80, 0x1b, 0x00, 0x00, 0x2d, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00};
    constexpr std::array<std::uint8_t, VK_UUID_SIZE> nvkUuidBus2e{
        0x34, 0x01, 0xde, 0x10, 0x80, 0x1b, 0x00, 0x00, 0x2e, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00};

    std::vector<FakePhysicalDevice> gFakePhysicalDevices;
    std::vector<FakeChainQuery> gFakePropertyQueries;
    std::vector<FakeChainQuery> gFakeFeatureQueries;

    [[nodiscard]] VkPhysicalDevice fakePhysical(std::uintptr_t handle) noexcept
    {
        return reinterpret_cast<VkPhysicalDevice>(handle);
    }

    [[nodiscard]] const FakePhysicalDevice* findFakePhysicalDevice(VkPhysicalDevice physical) noexcept
    {
        const auto found = std::ranges::find_if(gFakePhysicalDevices, [physical](const FakePhysicalDevice& device) {
            return fakePhysical(device.handle) == physical;
        });
        return found == gFakePhysicalDevices.end() ? nullptr : &*found;
    }

    void fillFakeProperties(VkPhysicalDevice physical, VkPhysicalDeviceProperties& properties) noexcept
    {
        properties = {};
        properties.apiVersion = VK_API_VERSION_1_3;
        properties.vendorID = 0x10de;
        properties.deviceID = 0x1b80;
        properties.deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        if (const auto* fake = findFakePhysicalDevice(physical); fake != nullptr)
        {
            properties.apiVersion = fake->apiVersion;
            std::memcpy(properties.deviceName,
                        fake->name.data(),
                        std::min(fake->name.size(), sizeof(properties.deviceName) - 1));
        }
    }

    VKAPI_ATTR VkResult VKAPI_CALL fakeEnumeratePhysicalDevices(VkInstance,
                                                                std::uint32_t* count,
                                                                VkPhysicalDevice* devices)
    {
        const auto available = static_cast<std::uint32_t>(gFakePhysicalDevices.size());
        if (devices == nullptr)
        {
            *count = available;
            return VK_SUCCESS;
        }
        const auto written = std::min(*count, available);
        for (std::uint32_t i = 0; i < written; ++i)
        {
            devices[i] = fakePhysical(gFakePhysicalDevices[i].handle);
        }
        *count = written;
        return written == available ? VK_SUCCESS : VK_INCOMPLETE;
    }

    VKAPI_ATTR void VKAPI_CALL fakeGetPhysicalDeviceProperties(VkPhysicalDevice physical,
                                                               VkPhysicalDeviceProperties* properties)
    {
        fillFakeProperties(physical, *properties);
    }

    void recordFakeChainQuery(std::vector<FakeChainQuery>& queries, VkPhysicalDevice physical, const void* head)
    {
        auto& query = queries.emplace_back(FakeChainQuery{.physical = physical, .chain = {}});
        for (const auto* next = static_cast<const VkBaseInStructure*>(head); next != nullptr; next = next->pNext)
        {
            query.chain.push_back(next->sType);
        }
    }

    [[nodiscard]] std::size_t fakeChainQueryCount(const std::vector<FakeChainQuery>& queries,
                                                  std::uintptr_t handle) noexcept
    {
        return static_cast<std::size_t>(std::ranges::count_if(
            queries, [handle](const FakeChainQuery& query) { return query.physical == fakePhysical(handle); }));
    }

    [[nodiscard]] bool fakeChainQueried(const std::vector<FakeChainQuery>& queries,
                                        std::uintptr_t handle,
                                        VkStructureType type) noexcept
    {
        return std::ranges::any_of(queries, [handle, type](const FakeChainQuery& query) {
            return query.physical == fakePhysical(handle) && std::ranges::find(query.chain, type) != query.chain.end();
        });
    }

    VKAPI_ATTR void VKAPI_CALL fakeGetPhysicalDeviceProperties2(VkPhysicalDevice physical,
                                                                VkPhysicalDeviceProperties2* properties)
    {
        recordFakeChainQuery(gFakePropertyQueries, physical, properties);
        fillFakeProperties(physical, properties->properties);
        const auto* fake = findFakePhysicalDevice(physical);
        if (fake == nullptr)
        {
            return;
        }
        for (auto* next = static_cast<VkBaseOutStructure*>(properties->pNext); next != nullptr; next = next->pNext)
        {
            if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES)
            {
                auto* id = reinterpret_cast<VkPhysicalDeviceIDProperties*>(next);
                std::ranges::copy(fake->uuid, id->deviceUUID);
                std::ranges::copy(fake->luid, id->deviceLUID);
                id->deviceLUIDValid = fake->luidValid ? VK_TRUE : VK_FALSE;
            }
            else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES)
            {
                reinterpret_cast<VkPhysicalDeviceDriverProperties*>(next)->driverID = fake->driver;
            }
        }
    }

    VKAPI_ATTR void VKAPI_CALL fakeGetPhysicalDeviceMemoryProperties(VkPhysicalDevice,
                                                                     VkPhysicalDeviceMemoryProperties* memory)
    {
        *memory = {};
    }

    VKAPI_ATTR void VKAPI_CALL fakeGetPhysicalDeviceFeatures2(VkPhysicalDevice physical,
                                                              VkPhysicalDeviceFeatures2* features)
    {
        recordFakeChainQuery(gFakeFeatureQueries, physical, features);
    }

    VKAPI_ATTR VkResult VKAPI_CALL fakeEnumerateDeviceExtensionProperties(VkPhysicalDevice,
                                                                          const char*,
                                                                          std::uint32_t* count,
                                                                          VkExtensionProperties*)
    {
        *count = 0;
        return VK_SUCCESS;
    }

    [[nodiscard]] rachis::Result<std::vector<rachis::vulkan::VulkanPhysicalDeviceRecord>> discoverFakePhysicalDevices(
        std::vector<FakePhysicalDevice> devices)
    {
        gFakePhysicalDevices = std::move(devices);
        gFakePropertyQueries.clear();
        gFakeFeatureQueries.clear();
        rachis::vulkan::VulkanInstanceTable table{};
        table.vkEnumeratePhysicalDevices = &fakeEnumeratePhysicalDevices;
        table.vkGetPhysicalDeviceProperties = &fakeGetPhysicalDeviceProperties;
        table.vkGetPhysicalDeviceProperties2 = &fakeGetPhysicalDeviceProperties2;
        table.vkGetPhysicalDeviceMemoryProperties = &fakeGetPhysicalDeviceMemoryProperties;
        table.vkGetPhysicalDeviceFeatures2 = &fakeGetPhysicalDeviceFeatures2;
        table.vkEnumerateDeviceExtensionProperties = &fakeEnumerateDeviceExtensionProperties;
        auto records = rachis::vulkan::discoverVulkanPhysicalDevices(
            reinterpret_cast<VkInstance>(std::uintptr_t{0x50}), table, completeInstance());
        gFakePhysicalDevices.clear();
        return records;
    }

    [[nodiscard]] const rachis::vulkan::VulkanPhysicalDeviceRecord* pinFakeAdapter(
        const std::vector<rachis::vulkan::VulkanPhysicalDeviceRecord>& records, rachis::AdapterId id)
    {
        return rachis::vulkan::selectVulkanPhysicalDevice(records,
                                                          completeInstance(),
                                                          rachis::DeviceDesc{
                                                              .backend = rachis::Backend::vulkan,
                                                              .adapterId = id,
                                                          });
    }

    void collectDiagnostic(const rachis::DiagnosticMessage& message, void* userData) noexcept
    {
        static_cast<std::vector<std::string>*>(userData)->emplace_back(message.description);
    }

    std::uint32_t gFakeInstanceVersion = VK_API_VERSION_1_3;
    VkResult gFakeInstanceVersionResult = VK_SUCCESS;

    VKAPI_ATTR VkResult VKAPI_CALL fakeEnumerateInstanceVersion(std::uint32_t* version)
    {
        *version = gFakeInstanceVersion;
        return gFakeInstanceVersionResult;
    }

    [[nodiscard]] rachis::Status checkFakeLoaderVersion(std::uint32_t version, VkResult result = VK_SUCCESS)
    {
        gFakeInstanceVersion = version;
        gFakeInstanceVersionResult = result;
        auto checked = rachis::vulkan::checkVulkanInstanceVersion(&fakeEnumerateInstanceVersion);
        gFakeInstanceVersion = VK_API_VERSION_1_3;
        gFakeInstanceVersionResult = VK_SUCCESS;
        return checked;
    }

    struct SyntheticMemoryType final
    {
        VkMemoryPropertyFlags flags = 0;
        std::uint32_t heap = 0;
    };

    [[nodiscard]] rachis::DeviceCapabilities publishSyntheticMemory(VkPhysicalDeviceType deviceType,
                                                                    std::initializer_list<VkMemoryHeapFlags> heaps,
                                                                    std::initializer_list<SyntheticMemoryType> types)
    {
        rachis::vulkan::VulkanPhysicalDeviceRecord record{};
        record.properties.deviceType = deviceType;
        record.properties.limits.nonCoherentAtomSize = 64;
        for (const auto flags : heaps)
        {
            record.memory.memoryHeaps[record.memory.memoryHeapCount++] = VkMemoryHeap{
                .size = std::uint64_t{256} << 20U,
                .flags = flags,
            };
        }
        for (const auto& memoryType : types)
        {
            record.memory.memoryTypes[record.memory.memoryTypeCount++] = VkMemoryType{
                .propertyFlags = memoryType.flags,
                .heapIndex = memoryType.heap,
            };
        }
        return rachis::vulkan::publishVulkanCapabilities(rachis::vulkan::VulkanInstanceTable{}, record, false);
    }

} // namespace

TEST_CASE("Vulkan instance creation requires a loader that reports Vulkan 1.3", "[size:small][rhi][vulkan][device]")
{
    CHECK(checkFakeLoaderVersion(VK_API_VERSION_1_3).has_value());
    CHECK(checkFakeLoaderVersion(VK_MAKE_API_VERSION(0, 1, 3, 296)).has_value());
    CHECK(checkFakeLoaderVersion(VK_MAKE_API_VERSION(0, 1, 4, 0)).has_value());

    const auto older = checkFakeLoaderVersion(VK_MAKE_API_VERSION(0, 1, 2, 198));
    REQUIRE_FALSE(older.has_value());
    CHECK(older.error().code == rachis::ErrorCode::backendUnavailable);
    CHECK(older.error().message.contains("instance version 1.2"));
    CHECK(older.error().message.contains("did not fall back to D3D12"));

    const auto failed = checkFakeLoaderVersion(VK_API_VERSION_1_3, VK_ERROR_OUT_OF_HOST_MEMORY);
    REQUIRE_FALSE(failed.has_value());
    CHECK(failed.error().code == rachis::ErrorCode::backendUnavailable);
    CHECK(failed.error().nativeCode == VK_ERROR_OUT_OF_HOST_MEMORY);
    CHECK(failed.error().message.contains("vkEnumerateInstanceVersion"));

    const auto firstVersionLoader = rachis::vulkan::checkVulkanInstanceVersion(nullptr);
    REQUIRE_FALSE(firstVersionLoader.has_value());
    CHECK(firstVersionLoader.error().code == rachis::ErrorCode::backendUnavailable);
    CHECK(firstVersionLoader.error().message.contains("instance version 1.0"));
}

TEST_CASE("Vulkan adapter ids tell identical GPUs apart and pin each one", "[size:small][rhi][vulkan][device]")
{
    auto records = discoverFakePhysicalDevices({
        FakePhysicalDevice{
            .handle = 0x60,
            .uuid = nvkUuidBus2e,
            .driver = VK_DRIVER_ID_MESA_DOZEN,
            .name = "Microsoft Direct3D12 (NVIDIA GeForce GTX 1080)",
        },
        FakePhysicalDevice{.handle = 0x61, .uuid = nvkUuidBus2d},
        FakePhysicalDevice{.handle = 0x62, .uuid = nvkUuidBus2e},
        FakePhysicalDevice{.handle = 0x63, .uuid = nvkUuidBus2d},
    });
    REQUIRE(records.has_value());
    REQUIRE(records->size() == 4);
    const auto& translated = (*records)[0];
    const auto& firstCard = (*records)[1];
    const auto& secondCard = (*records)[2];
    const auto& firstCardFromSecondDriver = (*records)[3];
    REQUIRE(translated.report.translationLayer);
    REQUIRE_FALSE(secondCard.report.translationLayer);

    CHECK(firstCard.adapter.id.value != 0);
    CHECK(secondCard.adapter.id.value != 0);
    CHECK(firstCard.adapter.id != secondCard.adapter.id);
    CHECK(firstCardFromSecondDriver.adapter.id == firstCard.adapter.id);

    const auto* pinnedFirst = pinFakeAdapter(*records, firstCard.adapter.id);
    REQUIRE(pinnedFirst != nullptr);
    CHECK(pinnedFirst->physical == firstCard.physical);
    const auto* pinnedSecond = pinFakeAdapter(*records, secondCard.adapter.id);
    REQUIRE(pinnedSecond != nullptr);
    CHECK(pinnedSecond->physical == secondCard.physical);

    auto rediscovered = discoverFakePhysicalDevices({FakePhysicalDevice{.handle = 0x70, .uuid = nvkUuidBus2e}});
    REQUIRE(rediscovered.has_value());
    REQUIRE(rediscovered->size() == 1);
    CHECK(rediscovered->front().adapter.id == secondCard.adapter.id);
}

TEST_CASE("Vulkan adapter ids mix in a valid deviceLUID and are never zero", "[size:small][rhi][vulkan][device]")
{
    constexpr std::array<std::uint8_t, VK_UUID_SIZE> sharedUuid{0x00, 0x00, 0x10, 0x6b, 0x1b, 0x00, 0x02, 0x0a};
    constexpr std::array<std::uint8_t, VK_LUID_SIZE> firstLuid{0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x05, 0xd0};
    constexpr std::array<std::uint8_t, VK_LUID_SIZE> secondLuid{0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x06, 0xd0};
    auto records = discoverFakePhysicalDevices({
        FakePhysicalDevice{.handle = 0x81, .uuid = sharedUuid, .luid = firstLuid, .luidValid = true},
        FakePhysicalDevice{.handle = 0x82, .uuid = sharedUuid, .luid = secondLuid, .luidValid = true},
        FakePhysicalDevice{.handle = 0x83, .uuid = sharedUuid, .luid = firstLuid, .luidValid = true},
        FakePhysicalDevice{.handle = 0x84},
    });
    REQUIRE(records.has_value());
    REQUIRE(records->size() == 4);

    CHECK((*records)[0].adapter.id != (*records)[1].adapter.id);
    CHECK((*records)[0].adapter.id == (*records)[2].adapter.id);
    CHECK((*records)[3].adapter.id.value != 0);

    const auto* pinned = pinFakeAdapter(*records, (*records)[1].adapter.id);
    REQUIRE(pinned != nullptr);
    CHECK(pinned->physical == (*records)[1].physical);
}

TEST_CASE("Vulkan adapter ids shared by different physical devices are reported as warnings",
          "[size:small][rhi][vulkan][device]")
{
    auto records = discoverFakePhysicalDevices({
        FakePhysicalDevice{.handle = 0x91, .uuid = nvkUuidBus2d},
        FakePhysicalDevice{.handle = 0x92, .uuid = nvkUuidBus2e, .name = "NVIDIA GeForce GTX 1080 (second card)"},
        FakePhysicalDevice{.handle = 0x93, .uuid = nvkUuidBus2d},
        FakePhysicalDevice{
            .handle = 0x94,
            .uuid = nvkUuidBus2e,
            .driver = VK_DRIVER_ID_MESA_DOZEN,
            .name = "Microsoft Direct3D12 (NVIDIA GeForce GTX 1080)",
        },
    });
    REQUIRE(records.has_value());
    REQUIRE(records->size() == 4);

    std::vector<std::string> messages;
    rachis::vulkan::VulkanDiagnosticSink sink;
    sink.callback = &collectDiagnostic;
    sink.userData = &messages;

    rachis::vulkan::reportVulkanAdapterIdCollisions(*records, sink);
    CHECK(sink.counts().warnings == 0);
    CHECK(messages.empty());

    (*records)[1].adapter.id = (*records)[0].adapter.id;
    (*records)[3].adapter.id = (*records)[0].adapter.id;
    rachis::vulkan::reportVulkanAdapterIdCollisions(*records, sink);
    CHECK(sink.counts().warnings == 1);
    REQUIRE(messages.size() == 1);
    CHECK(messages.front().contains("NVIDIA GeForce GTX 1080 (second card)"));
    CHECK(messages.front().contains("DeviceDesc::adapterId"));
    CHECK_FALSE(messages.front().contains("Direct3D12"));
}

TEST_CASE("Vulkan discovery stays within the API version each physical device reports",
          "[size:small][rhi][vulkan][device]")
{
    constexpr std::uintptr_t vulkan13 = 0xa0;
    constexpr std::uintptr_t vulkan12 = 0xa1;
    constexpr std::uintptr_t dozen12 = 0xa2;
    constexpr std::uintptr_t vulkan11 = 0xa3;
    constexpr std::uintptr_t vulkan10 = 0xa4;
    constexpr std::array<std::uint8_t, VK_UUID_SIZE> vulkan11Uuid{0x10, 0x02, 0x73, 0x7f, 0x00, 0x00, 0x00, 0x11};
    auto records = discoverFakePhysicalDevices({
        FakePhysicalDevice{.handle = vulkan13, .uuid = nvkUuidBus2d},
        FakePhysicalDevice{
            .handle = vulkan12,
            .uuid = nvkUuidBus2e,
            .driver = VK_DRIVER_ID_NVIDIA_PROPRIETARY,
            .name = "Vulkan 1.2 GPU",
            .apiVersion = VK_API_VERSION_1_2,
        },
        FakePhysicalDevice{
            .handle = dozen12,
            .uuid = nvkUuidBus2e,
            .driver = VK_DRIVER_ID_MESA_DOZEN,
            .name = "Microsoft Direct3D12 (Vulkan 1.2 GPU)",
            .apiVersion = VK_API_VERSION_1_2,
        },
        FakePhysicalDevice{
            .handle = vulkan11,
            .uuid = vulkan11Uuid,
            .name = "Vulkan 1.1 GPU",
            .apiVersion = VK_API_VERSION_1_1,
        },
        FakePhysicalDevice{.handle = vulkan10, .name = "Vulkan 1.0 GPU", .apiVersion = VK_API_VERSION_1_0},
    });
    REQUIRE(records.has_value());
    REQUIRE(records->size() == 5);

    CHECK(fakeChainQueried(gFakePropertyQueries, vulkan13, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES));
    CHECK(fakeChainQueried(gFakeFeatureQueries, vulkan13, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES));

    for (const auto handle : {vulkan12, dozen12, vulkan11, vulkan10})
    {
        CHECK(fakeChainQueryCount(gFakeFeatureQueries, handle) == 0);
        CHECK_FALSE(
            fakeChainQueried(gFakePropertyQueries, handle, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES));
        CHECK_FALSE(
            fakeChainQueried(gFakePropertyQueries, handle, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES));
        CHECK_FALSE(
            fakeChainQueried(gFakePropertyQueries, handle, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES));
    }
    CHECK(fakeChainQueried(gFakePropertyQueries, vulkan12, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES));
    CHECK(fakeChainQueried(gFakePropertyQueries, vulkan12, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES));
    CHECK(fakeChainQueried(gFakePropertyQueries, vulkan11, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES));
    CHECK_FALSE(fakeChainQueried(gFakePropertyQueries, vulkan11, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES));
    CHECK(fakeChainQueryCount(gFakePropertyQueries, vulkan10) == 0);

    const auto& current = (*records)[0];
    const auto& olderCard = (*records)[1];
    const auto& translated = (*records)[2];
    const auto& oldestCard = (*records)[3];
    const auto& firstVersionCard = (*records)[4];
    CHECK(current.report.apiMinor == 3);
    CHECK(olderCard.report.apiMajor == 1);
    CHECK(olderCard.report.apiMinor == 2);
    CHECK(olderCard.driverId == "NVIDIA_PROPRIETARY");
    CHECK(olderCard.adapter.id.value != 0);
    CHECK(olderCard.adapter.id != current.adapter.id);
    CHECK(translated.report.translationLayer);
    CHECK(oldestCard.report.apiMinor == 1);
    CHECK(oldestCard.adapter.id != olderCard.adapter.id);
    CHECK(firstVersionCard.report.apiMinor == 0);
    CHECK(firstVersionCard.adapter.id.value != 0);

    for (const auto* record : {&olderCard, &oldestCard, &firstVersionCard})
    {
        const auto decision = rachis::evaluateWindowsVulkanProfile(completeInstance(), record->report);
        CHECK_FALSE(decision.accepted);
        CHECK(decision.rejection == rachis::VulkanProfileRejection::apiVersionTooLow);
    }
    CHECK(rachis::evaluateWindowsVulkanProfile(completeInstance(), translated.report).rejection ==
          rachis::VulkanProfileRejection::translationLayer);
}

TEST_CASE("Vulkan unified memory follows the device type and not host-visible BAR memory",
          "[size:small][rhi][vulkan][device]")
{
    constexpr VkMemoryHeapFlags deviceHeap = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    constexpr VkMemoryHeapFlags systemHeap = 0;
    constexpr VkMemoryPropertyFlags local = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    constexpr VkMemoryPropertyFlags coherent =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    constexpr VkMemoryPropertyFlags cached = coherent | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    constexpr VkMemoryPropertyFlags lazy = local | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;

    SECTION("a discrete GPU with Resizable BAR is not unified")
    {
        const auto capabilities = publishSyntheticMemory(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
                                                         {deviceHeap, systemHeap},
                                                         {
                                                             {.flags = 0, .heap = 1},
                                                             {.flags = local, .heap = 0},
                                                             {.flags = coherent, .heap = 1},
                                                             {.flags = cached, .heap = 1},
                                                             {.flags = local | coherent, .heap = 0},
                                                             {.flags = local | cached, .heap = 0},
                                                         });
        CHECK_FALSE(capabilities.unifiedMemory);
        CHECK_FALSE(capabilities.cacheCoherentUnifiedMemory);
        CHECK(capabilities.mappedMemory.hostCoherent);
        CHECK(capabilities.mappedMemory.nonCoherentAtomSize == 64);
        CHECK(capabilities.persistentMapping);
    }

    SECTION("a discrete GPU with a 256 MiB BAR heap is not unified")
    {
        const auto capabilities = publishSyntheticMemory(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
                                                         {deviceHeap, systemHeap, deviceHeap},
                                                         {
                                                             {.flags = local, .heap = 0},
                                                             {.flags = coherent, .heap = 1},
                                                             {.flags = cached, .heap = 1},
                                                             {.flags = local | coherent, .heap = 2},
                                                         });
        CHECK_FALSE(capabilities.unifiedMemory);
        CHECK_FALSE(capabilities.cacheCoherentUnifiedMemory);
        CHECK(capabilities.mappedMemory.hostCoherent);
    }

    SECTION("an Apple integrated GPU is unified and cache coherent")
    {
        const auto capabilities = publishSyntheticMemory(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
                                                         {deviceHeap},
                                                         {
                                                             {.flags = local, .heap = 0},
                                                             {.flags = local | cached, .heap = 0},
                                                             {.flags = lazy, .heap = 0},
                                                         });
        CHECK(capabilities.unifiedMemory);
        CHECK(capabilities.cacheCoherentUnifiedMemory);
        CHECK(capabilities.mappedMemory.hostCoherent);
    }

    SECTION("an integrated GPU with only write-combined device-local memory is unified but not cache coherent")
    {
        const auto capabilities = publishSyntheticMemory(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
                                                         {deviceHeap, systemHeap},
                                                         {
                                                             {.flags = local, .heap = 0},
                                                             {.flags = local | coherent, .heap = 0},
                                                             {.flags = coherent, .heap = 1},
                                                             {.flags = cached, .heap = 1},
                                                         });
        CHECK(capabilities.unifiedMemory);
        CHECK_FALSE(capabilities.cacheCoherentUnifiedMemory);
    }

    SECTION("an integrated GPU without host-visible device-local memory is still unified")
    {
        const auto capabilities = publishSyntheticMemory(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
                                                         {deviceHeap, systemHeap},
                                                         {
                                                             {.flags = local, .heap = 0},
                                                             {.flags = cached, .heap = 1},
                                                         });
        CHECK(capabilities.unifiedMemory);
        CHECK_FALSE(capabilities.cacheCoherentUnifiedMemory);
    }

    SECTION("a CPU device is unified and cache coherent")
    {
        const auto capabilities =
            publishSyntheticMemory(VK_PHYSICAL_DEVICE_TYPE_CPU, {deviceHeap}, {{.flags = local | cached, .heap = 0}});
        CHECK(capabilities.unifiedMemory);
        CHECK(capabilities.cacheCoherentUnifiedMemory);
        CHECK(capabilities.mappedMemory.hostCoherent);
    }

    SECTION("virtual and other GPUs are unified only when every heap is device-local")
    {
        const auto passthrough = publishSyntheticMemory(VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,
                                                        {deviceHeap, systemHeap},
                                                        {
                                                            {.flags = local, .heap = 0},
                                                            {.flags = cached, .heap = 1},
                                                            {.flags = local | cached, .heap = 0},
                                                        });
        CHECK_FALSE(passthrough.unifiedMemory);
        CHECK_FALSE(passthrough.cacheCoherentUnifiedMemory);

        const auto shared = publishSyntheticMemory(VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,
                                                   {deviceHeap},
                                                   {
                                                       {.flags = local, .heap = 0},
                                                       {.flags = local | cached, .heap = 0},
                                                   });
        CHECK(shared.unifiedMemory);
        CHECK(shared.cacheCoherentUnifiedMemory);

        const auto writeCombined = publishSyntheticMemory(
            VK_PHYSICAL_DEVICE_TYPE_OTHER, {deviceHeap}, {{.flags = local | coherent, .heap = 0}});
        CHECK(writeCombined.unifiedMemory);
        CHECK_FALSE(writeCombined.cacheCoherentUnifiedMemory);

        const auto deviceOnly =
            publishSyntheticMemory(VK_PHYSICAL_DEVICE_TYPE_OTHER, {deviceHeap}, {{.flags = local, .heap = 0}});
        CHECK_FALSE(deviceOnly.unifiedMemory);
        CHECK_FALSE(deviceOnly.cacheCoherentUnifiedMemory);
    }
}

TEST_CASE("profile rejection diagnostics name every missing item", "[size:small][rhi][vulkan][device]")
{
    auto accepted = rachis::evaluateWindowsVulkanProfile(completeInstance(), completeDevice());
    REQUIRE(accepted.accepted);
    CHECK(rachis::describeVulkanProfileRejection(accepted).empty());

    auto missingFeature = completeDevice();
    missingFeature.dynamicRendering = false;
    const auto feature = rachis::evaluateWindowsVulkanProfile(completeInstance(), missingFeature);
    REQUIRE_FALSE(feature.accepted);
    CHECK(feature.rejection == rachis::VulkanProfileRejection::missingRequiredFeature);
    const auto featureMessage = rachis::describeVulkanProfileRejection(feature);
    CHECK(featureMessage.contains("dynamicRendering"));
    CHECK(featureMessage.contains("did not fall back to D3D12"));

    auto missingExtension = completeDevice();
    missingExtension.khrSwapchain = false;
    const auto extension = rachis::evaluateWindowsVulkanProfile(completeInstance(), missingExtension);
    REQUIRE_FALSE(extension.accepted);
    CHECK(rachis::describeVulkanProfileRejection(extension).contains("VK_KHR_swapchain"));
}

TEST_CASE("headless Vulkan devices create and destroy repeatedly", "[size:medium][rhi][vulkan][device]")
{
    auto adapters = rachis::enumerateAdapters(rachis::Backend::vulkan);
    if (rachis::test::skipWithoutVulkanAdapterList(adapters))
    {
        return;
    }

    rachis::AdapterId id{};
    for (std::uint32_t i = 0; i < 4; ++i)
    {
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .validation = rachis::ValidationLevel::standard,
            .enableDred = false,
        });
        if (rachis::test::skipWithoutCreatedDevice(created))
        {
            return;
        }

        REQUIRE(created->isValid());
        REQUIRE(created->operationalState() == rachis::DeviceOperationalState::active);
        REQUIRE(created->adapterInfo().has_value());
        REQUIRE(created->capabilities().has_value());

        const auto adapter = *created->adapterInfo();
        CHECK(adapter.backend == rachis::Backend::vulkan);
        CHECK_FALSE(adapter.name.contains("Dozen"));
        CHECK(adapter.id.value != 0);
        if (i == 0)
        {
            id = adapter.id;
        }
        else
        {
            CHECK(adapter.id == id);
        }

        const auto capabilities = *created->capabilities();
        CHECK(capabilities.limits.maxTextureDimension2D >= 8192);
        CHECK(capabilities.limits.maxBindingSets >= 4);
        CHECK(capabilities.limits.maxPushConstantBytes >= 128);
        CHECK(capabilities.limits.maxColorAttachments >= 8);
        CHECK(capabilities.limits.maxComputeWorkgroupInvocations >= 256);
        CHECK(capabilities.limits.maxFramebufferLayers >= 1);
        if (capabilities.layeredRenderingModes == rachis::LayeredRenderingMode::none)
        {
            CHECK(capabilities.limits.maxFramebufferLayers == 1);
        }
        CHECK(capabilities.limits.maxViewports >= 1);
        CHECK((capabilities.limits.maxViewports > 1) == capabilities.multiViewport);
        CHECK(capabilities.limits.maxStorageBufferBindingSize >= (std::uint64_t{1} << 27));
        CHECK(capabilities.limits.maxStorageBufferBindingSize <= std::numeric_limits<std::uint32_t>::max());
        CHECK(capabilities.limits.maxUniformBufferSize >= 16384);
        CHECK(capabilities.limits.maxUniformBufferSize <= std::numeric_limits<std::uint32_t>::max());
        for (const auto alignment :
             {capabilities.limits.minUniformBufferOffsetAlignment, capabilities.limits.minStorageBufferOffsetAlignment})
        {
            CHECK(std::has_single_bit(alignment));
            CHECK(alignment <= 256);
        }
        CHECK_FALSE((capabilities.taskShaders && !capabilities.meshShaders));
        CHECK_FALSE((capabilities.variableRateShadingTier2 && !capabilities.variableRateShading));
        CHECK((capabilities.shadingRateImageTileSize != 0) == capabilities.variableRateShadingTier2);
        CHECK_FALSE((capabilities.indirectRayDispatch && !capabilities.rayTracing));
        CHECK((capabilities.rayTracingLimits.shaderGroupHandleSize != 0) == capabilities.rayTracing);
        CHECK(capabilities.sharedTextureHandles == capabilities.sharedBufferHandles);
        CHECK(capabilities.sharedTextureHandles == capabilities.sharedTimelineHandles);

        const auto graphics = created->queueProperties(rachis::QueueType::graphics);
        CHECK(graphics.timestampQueries);
        CHECK(graphics.timestampValidBits > 0);
        CHECK(graphics.timestampFrequency > 0);

        const auto color = created->queryFormatProperties(rachis::Format::rgba8Unorm);
        CHECK(rachis::hasAny(color.features, rachis::FormatFeature::colorAttachment));
        CHECK(rachis::hasAny(color.features, rachis::FormatFeature::sampled));

        const auto depth = created->queryFormatProperties(rachis::Format::d32Float);
        CHECK(rachis::hasAny(depth.features, rachis::FormatFeature::depthStencilAttachment));
        CHECK(rachis::hasAny(depth.features, rachis::FormatFeature::texture2D));
        CHECK_FALSE(rachis::hasAny(depth.features, rachis::FormatFeature::texture3D));
        const auto packedDepthStencil = created->queryFormatProperties(rachis::Format::d24UnormS8Uint);
        CHECK_FALSE(rachis::hasAny(packedDepthStencil.features, rachis::FormatFeature::texture3D));

        auto budgets = created->queryMemoryBudgets();
        REQUIRE(budgets.has_value());
        REQUIRE_FALSE(budgets->empty());
        CHECK_FALSE(created->queryMemoryBudgets(capabilities.nodeCount).has_value());

        auto buffer = created->createBuffer({});
        REQUIRE_FALSE(buffer.has_value());
        CHECK(buffer.error().code == rachis::ErrorCode::invalidArgument);

        REQUIRE(created->waitIdle().has_value());
    }

    auto explicitDevice = rachis::createDevice(rachis::DeviceDesc{
        .backend = rachis::Backend::vulkan,
        .adapterId = id,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = false,
    });
    REQUIRE(explicitDevice.has_value());
    CHECK(rachis::test::requireAdapterInfo(*explicitDevice).id == id);
}

TEST_CASE("Vulkan adapterId pins every hardware adapter that enumerateAdapters lists",
          "[size:medium][rhi][vulkan][device]")
{
    auto adapters = rachis::enumerateAdapters(rachis::Backend::vulkan);
    if (rachis::test::skipWithoutVulkanAdapterList(adapters))
    {
        return;
    }

    std::size_t pinned = 0;
    for (const auto& adapter : *adapters)
    {
        if (adapter.type == rachis::AdapterType::software)
        {
            continue;
        }
        INFO(adapter.name);
        CHECK(adapter.backend == rachis::Backend::vulkan);
        CHECK(adapter.id.value != 0);
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .adapterId = adapter.id,
            .validation = rachis::ValidationLevel::standard,
            .enableDred = false,
        });
        if (!created && created.error().code == rachis::ErrorCode::unsupported)
        {
            UNSCOPED_INFO(adapter.name << ": " << created.error().message);
            continue;
        }
        if (rachis::test::skipWithoutCreatedDevice(created))
        {
            return;
        }
        const auto selected = rachis::test::requireAdapterInfo(*created);
        CHECK(selected.id == adapter.id);
        CHECK(selected.name == adapter.name);
        ++pinned;
    }
    if (pinned == 0)
    {
        SKIP("No hardware Vulkan adapter passed the device profile.");
    }
}

TEST_CASE("Vulkan adapter types follow unified memory on every device", "[size:medium][rhi][vulkan][device]")
{
    auto adapters = rachis::enumerateAdapters(rachis::Backend::vulkan);
    if (rachis::test::skipWithoutVulkanAdapterList(adapters))
    {
        return;
    }

    std::size_t checked = 0;
    for (const auto& adapter : *adapters)
    {
        INFO(adapter.name);
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .adapterId = adapter.id,
            .validation = rachis::ValidationLevel::standard,
            .enableDred = false,
        });
        if (!created && created.error().code == rachis::ErrorCode::unsupported)
        {
            UNSCOPED_INFO(adapter.name << ": " << created.error().message);
            continue;
        }
        if (rachis::test::skipWithoutCreatedDevice(created))
        {
            return;
        }
        const auto info = rachis::test::requireAdapterInfo(*created);
        const auto capabilities = rachis::test::requireCapabilities(*created);
        INFO("unifiedMemory " << capabilities.unifiedMemory << ", cacheCoherentUnifiedMemory "
                              << capabilities.cacheCoherentUnifiedMemory);
        CHECK(info.id == adapter.id);
        CHECK(info.type == adapter.type);
        CHECK((!capabilities.cacheCoherentUnifiedMemory || capabilities.unifiedMemory));
        if (info.type != rachis::AdapterType::unknown)
        {
            const bool expected =
                info.type == rachis::AdapterType::integrated || info.type == rachis::AdapterType::software;
            CHECK(capabilities.unifiedMemory == expected);
        }
        ++checked;
    }
    if (checked == 0)
    {
        SKIP("No Vulkan adapter passed the device profile.");
    }
}

TEST_CASE("live Vulkan device survives concurrent adapter enumeration and second device lifecycle",
          "[size:medium][rhi][vulkan][device][dispatch]")
{
    auto firstCreated = rachis::createDevice(rachis::DeviceDesc{
        .backend = rachis::Backend::vulkan,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = false,
        .applicationName = "Primary Vulkan device",
    });
    if (rachis::test::skipWithoutCreatedDevice(firstCreated))
    {
        return;
    }
    auto deviceA = std::move(*firstCreated);

    CHECK(volkGetLoadedInstance() == VK_NULL_HANDLE);
    CHECK(volkGetLoadedDevice() == VK_NULL_HANDLE);

    auto enumerated = rachis::enumerateAdapters(rachis::Backend::vulkan);
    REQUIRE(enumerated.has_value());
    CHECK_FALSE(enumerated->empty());

    {
        auto secondCreated = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .validation = rachis::ValidationLevel::disabled,
            .requireValidation = false,
            .enableDred = false,
            .applicationName = "Secondary Vulkan device",
        });
        REQUIRE(secondCreated.has_value());
        CHECK(secondCreated->queryFormatProperties(rachis::Format::rgba8Unorm).features != rachis::FormatFeature::none);
        REQUIRE(secondCreated->waitIdle().has_value());
    }

    CHECK(volkGetLoadedInstance() == VK_NULL_HANDLE);
    CHECK(volkGetLoadedDevice() == VK_NULL_HANDLE);

    const auto formatProps = deviceA.queryFormatProperties(rachis::Format::rgba8Unorm);
    CHECK(rachis::hasAny(formatProps.features, rachis::FormatFeature::sampled));
    CHECK(rachis::hasAny(formatProps.features, rachis::FormatFeature::colorAttachment));

    const auto queueProps = deviceA.queueProperties(rachis::QueueType::graphics);
    CHECK(queueProps.timestampQueries);
    CHECK(queueProps.timestampFrequency > 0);

    const auto budgets = deviceA.queryMemoryBudgets();
    REQUIRE(budgets.has_value());
    CHECK_FALSE(budgets->empty());

    constexpr std::uint64_t payloadBytes = 64;
    auto upload = deviceA.createBuffer({
        .size = payloadBytes,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Post-enumeration upload buffer",
    });
    auto readback = deviceA.createBuffer({
        .size = payloadBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Post-enumeration readback buffer",
    });
    auto texture = deviceA.createTexture({
        .width = 4,
        .height = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::copyDestination,
        .label = "Post-enumeration texture",
    });
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(texture.has_value());

    auto view = deviceA.createTextureView({
        .texture = *texture,
        .label = "Post-enumeration view",
    });
    auto sampler = deviceA.createSampler({
        .label = "Post-enumeration sampler",
    });
    REQUIRE(view.has_value());
    REQUIRE(sampler.has_value());

    std::array<std::byte, payloadBytes> expected{};
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        expected[i] = static_cast<std::byte>((i * 19U + 5U) & 0xFFU);
    }
    REQUIRE(deviceA.writeBuffer(*upload, 0, expected).has_value());

    auto pool = deviceA.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Post-enumeration command pool",
    });
    REQUIRE(pool.has_value());
    auto command = deviceA.acquireCommandBuffer(*pool);
    REQUIRE(command.has_value());

    REQUIRE(deviceA.beginDebugGroup(*command, {.label = "Post-enumeration copy group"}).has_value());
    REQUIRE(deviceA.insertDebugMarker(*command, {.label = "Post-enumeration copy marker"}).has_value());
    const std::array copyRegions{
        rachis::BufferCopyRegion{.sourceOffset = 0, .destinationOffset = 0, .size = payloadBytes},
    };
    REQUIRE(deviceA.copyBuffer(*command, *upload, *readback, copyRegions).has_value());
    REQUIRE(deviceA.endDebugGroup(*command).has_value());

    auto submitted = deviceA.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(deviceA.wait(*submitted).has_value());

    std::array<std::byte, payloadBytes> actual{};
    REQUIRE(deviceA.readBuffer(*readback, 0, actual).has_value());
    CHECK(actual == expected);

    REQUIRE(deviceA.destroyCommandPool(*pool).has_value());
    REQUIRE(deviceA.destroySampler(*sampler).has_value());
    REQUIRE(deviceA.destroyTextureView(*view).has_value());
    REQUIRE(deviceA.destroyTexture(*texture).has_value());
    REQUIRE(deviceA.destroyBuffer(*readback).has_value());
    REQUIRE(deviceA.destroyBuffer(*upload).has_value());

    const auto diagnostics = deviceA.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
