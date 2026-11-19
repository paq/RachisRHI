#include "Internal/VulkanRuntime.h"

#include "Internal/TextureFormatRules.h"
#include "Internal/VulkanBackendConfiguration.h"
#include "Internal/VulkanDiagnostics.h"
#include "Internal/VulkanLinuxProfile.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#ifdef __APPLE__
#include <dlfcn.h>
#endif

#ifdef RACHIS_TESTING
#include "Vulkan/VulkanTestHooks.h"
#endif

namespace rachis::vulkan
{
    namespace
    {

        [[nodiscard]] Error selectionError(VulkanSelectionFailure failure, std::string message, std::int64_t native = 0)
        {
            return Error{vulkanSelectionError(failure), native, std::move(message)};
        }

        [[nodiscard]] Error profileError(const VulkanProfileDecision& decision)
        {
            return selectionError(VulkanSelectionFailure::requiredFeatureMissing,
                                  describeNativeVulkanProfileRejection(decision));
        }

        [[nodiscard]] bool hasExtension(const std::vector<VkExtensionProperties>& extensions, const char* name) noexcept
        {
            return std::ranges::any_of(extensions, [&](const VkExtensionProperties& extension) {
                return std::string_view(extension.extensionName) == name;
            });
        }

        [[nodiscard]] bool hasLayer(const std::vector<VkLayerProperties>& layers, const char* name) noexcept
        {
            return std::ranges::any_of(
                layers, [&](const VkLayerProperties& layer) { return std::string_view(layer.layerName) == name; });
        }

        void appendIf(std::vector<const char*>& names, bool wanted, const char* name)
        {
            if (wanted)
            {
                names.push_back(name);
            }
        }

        [[nodiscard]] constexpr bool supportsApiVersion(std::uint32_t apiVersion, std::uint32_t required) noexcept
        {
            return VK_API_VERSION_MAJOR(apiVersion) > VK_API_VERSION_MAJOR(required) ||
                   (VK_API_VERSION_MAJOR(apiVersion) == VK_API_VERSION_MAJOR(required) &&
                    VK_API_VERSION_MINOR(apiVersion) >= VK_API_VERSION_MINOR(required));
        }

        struct VulkanLoaderTable final
        {
            PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
            PFN_vkCreateInstance vkCreateInstance = nullptr;
            PFN_vkEnumerateInstanceVersion vkEnumerateInstanceVersion = nullptr;
            PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties = nullptr;
            PFN_vkEnumerateInstanceLayerProperties vkEnumerateInstanceLayerProperties = nullptr;
        };

        [[nodiscard]] Result<std::vector<VkExtensionProperties>> instanceExtensions(const VulkanLoaderTable& loader)
        {
            std::uint32_t count = 0;
            const auto first = loader.vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
            if (first != VK_SUCCESS)
            {
                return std::unexpected(selectionError(VulkanSelectionFailure::loaderUnavailable,
                                                      "vkEnumerateInstanceExtensionProperties failed.",
                                                      first));
            }
            std::vector<VkExtensionProperties> extensions(count);
            if (count != 0)
            {
                const auto second = loader.vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data());
                if (second != VK_SUCCESS)
                {
                    return std::unexpected(selectionError(VulkanSelectionFailure::loaderUnavailable,
                                                          "vkEnumerateInstanceExtensionProperties failed.",
                                                          second));
                }
            }
            return extensions;
        }

        [[nodiscard]] Result<std::vector<VkLayerProperties>> instanceLayers(const VulkanLoaderTable& loader)
        {
            std::uint32_t count = 0;
            const auto first = loader.vkEnumerateInstanceLayerProperties(&count, nullptr);
            if (first != VK_SUCCESS)
            {
                return std::unexpected(selectionError(
                    VulkanSelectionFailure::loaderUnavailable, "vkEnumerateInstanceLayerProperties failed.", first));
            }
            std::vector<VkLayerProperties> layers(count);
            if (count != 0)
            {
                const auto second = loader.vkEnumerateInstanceLayerProperties(&count, layers.data());
                if (second != VK_SUCCESS)
                {
                    return std::unexpected(selectionError(VulkanSelectionFailure::loaderUnavailable,
                                                          "vkEnumerateInstanceLayerProperties failed.",
                                                          second));
                }
            }
            return layers;
        }

        [[nodiscard]] Result<std::vector<VkExtensionProperties>> deviceExtensions(
            const VulkanInstanceTable& instanceTable, VkPhysicalDevice physical)
        {
            std::uint32_t count = 0;
            const auto first = instanceTable.vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
            if (first != VK_SUCCESS)
            {
                return std::unexpected(selectionError(
                    VulkanSelectionFailure::loaderUnavailable, "vkEnumerateDeviceExtensionProperties failed.", first));
            }
            std::vector<VkExtensionProperties> extensions(count);
            if (count != 0)
            {
                const auto second =
                    instanceTable.vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, extensions.data());
                if (second != VK_SUCCESS)
                {
                    return std::unexpected(selectionError(VulkanSelectionFailure::loaderUnavailable,
                                                          "vkEnumerateDeviceExtensionProperties failed.",
                                                          second));
                }
            }
            return extensions;
        }

        [[nodiscard]] std::string_view driverIdName(VkDriverId id) noexcept
        {
            switch (id)
            {
            case VK_DRIVER_ID_NVIDIA_PROPRIETARY:
                return "NVIDIA_PROPRIETARY";
            case VK_DRIVER_ID_AMD_PROPRIETARY:
                return "AMD_PROPRIETARY";
            case VK_DRIVER_ID_AMD_OPEN_SOURCE:
                return "AMD_OPEN_SOURCE";
            case VK_DRIVER_ID_MESA_RADV:
                return "MESA_RADV";
            case VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS:
                return "INTEL_PROPRIETARY_WINDOWS";
            case VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA:
                return "INTEL_OPEN_SOURCE_MESA";
            case VK_DRIVER_ID_MESA_DOZEN:
                return "MESA_DOZEN";
            case VK_DRIVER_ID_GOOGLE_SWIFTSHADER:
                return "GOOGLE_SWIFTSHADER";
            case VK_DRIVER_ID_MESA_LLVMPIPE:
                return "MESA_LLVMPIPE";
#ifdef VK_DRIVER_ID_MOLTENVK
            case VK_DRIVER_ID_MOLTENVK:
                return "MOLTENVK";
#endif
            default:
                return "UNKNOWN";
            }
        }

        constexpr std::uint64_t fnv1aOffsetBasis = 0xcbf29ce484222325ULL;
        constexpr std::uint64_t fnv1aPrime = 0x00000100000001b3ULL;

        [[nodiscard]] std::uint64_t fnv1a(std::uint64_t hash, std::span<const std::uint8_t> bytes) noexcept
        {
            for (const auto byte : bytes)
            {
                hash = (hash ^ byte) * fnv1aPrime;
            }
            return hash;
        }

        [[nodiscard]] AdapterId adapterIdFromIdentity(const VulkanPhysicalDeviceRecord& record) noexcept
        {
            auto hash = fnv1a(fnv1aOffsetBasis, record.deviceUuid);
            if (record.deviceLuidValid)
            {
                hash = fnv1a(hash, record.deviceLuid);
            }
            return AdapterId{.value = hash != 0 ? hash : fnv1aOffsetBasis};
        }

        [[nodiscard]] bool sameDeviceIdentity(const VulkanPhysicalDeviceRecord& first,
                                              const VulkanPhysicalDeviceRecord& second) noexcept
        {
            return first.deviceUuid == second.deviceUuid && first.deviceLuidValid == second.deviceLuidValid &&
                   (!first.deviceLuidValid || first.deviceLuid == second.deviceLuid);
        }

        [[nodiscard]] const VulkanPhysicalDeviceRecord* firstNativeDeviceWithId(
            const std::vector<VulkanPhysicalDeviceRecord>& devices, AdapterId id) noexcept
        {
            const auto found = std::ranges::find_if(devices, [id](const VulkanPhysicalDeviceRecord& device) {
                return device.adapter.id == id && !device.report.translationLayer;
            });
            return found == devices.end() ? nullptr : &*found;
        }

        [[nodiscard]] AdapterType adapterTypeFrom(VkPhysicalDeviceType type) noexcept
        {
            switch (type)
            {
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                return AdapterType::integrated;
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                return AdapterType::discrete;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                return AdapterType::software;
            default:
                return AdapterType::unknown;
            }
        }

        [[nodiscard]] bool formatHasOptimal(const VulkanInstanceTable& instanceTable,
                                            VkPhysicalDevice physical,
                                            VkFormat format,
                                            VkFormatFeatureFlags required) noexcept
        {
            if (instanceTable.vkGetPhysicalDeviceFormatProperties == nullptr)
            {
                return false;
            }
            VkFormatProperties properties{};
            instanceTable.vkGetPhysicalDeviceFormatProperties(physical, format, &properties);
            return (properties.optimalTilingFeatures & required) == required;
        }

    } // namespace

    VkFormat vulkanFormat(Format format) noexcept
    {
        switch (format)
        {
        case Format::r8Unorm:
            return VK_FORMAT_R8_UNORM;
        case Format::rg8Unorm:
            return VK_FORMAT_R8G8_UNORM;
        case Format::rgba8Unorm:
            return VK_FORMAT_R8G8B8A8_UNORM;
        case Format::rgba8UnormSrgb:
            return VK_FORMAT_R8G8B8A8_SRGB;
        case Format::bgra8Unorm:
            return VK_FORMAT_B8G8R8A8_UNORM;
        case Format::bgra8UnormSrgb:
            return VK_FORMAT_B8G8R8A8_SRGB;
        case Format::r16Float:
            return VK_FORMAT_R16_SFLOAT;
        case Format::rg16Float:
            return VK_FORMAT_R16G16_SFLOAT;
        case Format::rgba16Float:
            return VK_FORMAT_R16G16B16A16_SFLOAT;
        case Format::r32Float:
            return VK_FORMAT_R32_SFLOAT;
        case Format::rg32Float:
            return VK_FORMAT_R32G32_SFLOAT;
        case Format::rgba32Float:
            return VK_FORMAT_R32G32B32A32_SFLOAT;
        case Format::d32Float:
            return VK_FORMAT_D32_SFLOAT;
        case Format::d24UnormS8Uint:
            return VK_FORMAT_D24_UNORM_S8_UINT;
        case Format::bc1RgbaUnorm:
            return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        case Format::bc1RgbaUnormSrgb:
            return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
        case Format::bc3RgbaUnorm:
            return VK_FORMAT_BC3_UNORM_BLOCK;
        case Format::bc3RgbaUnormSrgb:
            return VK_FORMAT_BC3_SRGB_BLOCK;
        case Format::bc5RgUnorm:
            return VK_FORMAT_BC5_UNORM_BLOCK;
        case Format::bc7RgbaUnorm:
            return VK_FORMAT_BC7_UNORM_BLOCK;
        case Format::bc7RgbaUnormSrgb:
            return VK_FORMAT_BC7_SRGB_BLOCK;
        case Format::r32Uint:
            return VK_FORMAT_R32_UINT;
        case Format::r8Uint:
            return VK_FORMAT_R8_UINT;
        case Format::rg8Uint:
            return VK_FORMAT_R8G8_UINT;
        case Format::rgba8Uint:
            return VK_FORMAT_R8G8B8A8_UINT;
        case Format::r8Sint:
            return VK_FORMAT_R8_SINT;
        case Format::rg8Sint:
            return VK_FORMAT_R8G8_SINT;
        case Format::rgba8Sint:
            return VK_FORMAT_R8G8B8A8_SINT;
        case Format::r16Uint:
            return VK_FORMAT_R16_UINT;
        case Format::rg16Uint:
            return VK_FORMAT_R16G16_UINT;
        case Format::rgba16Uint:
            return VK_FORMAT_R16G16B16A16_UINT;
        case Format::r16Sint:
            return VK_FORMAT_R16_SINT;
        case Format::rg16Sint:
            return VK_FORMAT_R16G16_SINT;
        case Format::rgba16Sint:
            return VK_FORMAT_R16G16B16A16_SINT;
        case Format::rg32Uint:
            return VK_FORMAT_R32G32_UINT;
        case Format::rgba32Uint:
            return VK_FORMAT_R32G32B32A32_UINT;
        case Format::r32Sint:
            return VK_FORMAT_R32_SINT;
        case Format::rg32Sint:
            return VK_FORMAT_R32G32_SINT;
        case Format::rgba32Sint:
            return VK_FORMAT_R32G32B32A32_SINT;
        case Format::r64Uint:
            return VK_FORMAT_R64_UINT;
        case Format::bc2RgbaUnorm:
            return VK_FORMAT_BC2_UNORM_BLOCK;
        case Format::bc2RgbaUnormSrgb:
            return VK_FORMAT_BC2_SRGB_BLOCK;
        case Format::bc4RUnorm:
            return VK_FORMAT_BC4_UNORM_BLOCK;
        case Format::bc4RSnorm:
            return VK_FORMAT_BC4_SNORM_BLOCK;
        case Format::bc5RgSnorm:
            return VK_FORMAT_BC5_SNORM_BLOCK;
        case Format::bc6hRgbUfloat:
            return VK_FORMAT_BC6H_UFLOAT_BLOCK;
        case Format::bc6hRgbSfloat:
            return VK_FORMAT_BC6H_SFLOAT_BLOCK;
        case Format::unknown:
            break;
        }
        return VK_FORMAT_UNDEFINED;
    }

    VkFormat vulkanVertexFormat(VertexFormat format) noexcept
    {
        switch (format)
        {
        case VertexFormat::float32:
            return VK_FORMAT_R32_SFLOAT;
        case VertexFormat::float32x2:
            return VK_FORMAT_R32G32_SFLOAT;
        case VertexFormat::float32x3:
            return VK_FORMAT_R32G32B32_SFLOAT;
        case VertexFormat::float32x4:
            return VK_FORMAT_R32G32B32A32_SFLOAT;
        case VertexFormat::uint32:
            return VK_FORMAT_R32_UINT;
        case VertexFormat::uint32x2:
            return VK_FORMAT_R32G32_UINT;
        case VertexFormat::uint32x3:
            return VK_FORMAT_R32G32B32_UINT;
        case VertexFormat::uint32x4:
            return VK_FORMAT_R32G32B32A32_UINT;
        case VertexFormat::sint32:
            return VK_FORMAT_R32_SINT;
        case VertexFormat::sint32x2:
            return VK_FORMAT_R32G32_SINT;
        case VertexFormat::sint32x3:
            return VK_FORMAT_R32G32B32_SINT;
        case VertexFormat::sint32x4:
            return VK_FORMAT_R32G32B32A32_SINT;
        case VertexFormat::uint8x2:
            return VK_FORMAT_R8G8_UINT;
        case VertexFormat::uint8x4:
            return VK_FORMAT_R8G8B8A8_UINT;
        case VertexFormat::sint8x2:
            return VK_FORMAT_R8G8_SINT;
        case VertexFormat::sint8x4:
            return VK_FORMAT_R8G8B8A8_SINT;
        case VertexFormat::unorm8x2:
            return VK_FORMAT_R8G8_UNORM;
        case VertexFormat::unorm8x4:
            return VK_FORMAT_R8G8B8A8_UNORM;
        case VertexFormat::snorm8x2:
            return VK_FORMAT_R8G8_SNORM;
        case VertexFormat::snorm8x4:
            return VK_FORMAT_R8G8B8A8_SNORM;
        case VertexFormat::uint16x2:
            return VK_FORMAT_R16G16_UINT;
        case VertexFormat::uint16x4:
            return VK_FORMAT_R16G16B16A16_UINT;
        case VertexFormat::sint16x2:
            return VK_FORMAT_R16G16_SINT;
        case VertexFormat::sint16x4:
            return VK_FORMAT_R16G16B16A16_SINT;
        case VertexFormat::unorm16x2:
            return VK_FORMAT_R16G16_UNORM;
        case VertexFormat::unorm16x4:
            return VK_FORMAT_R16G16B16A16_UNORM;
        case VertexFormat::snorm16x2:
            return VK_FORMAT_R16G16_SNORM;
        case VertexFormat::snorm16x4:
            return VK_FORMAT_R16G16B16A16_SNORM;
        case VertexFormat::float16x2:
            return VK_FORMAT_R16G16_SFLOAT;
        case VertexFormat::float16x4:
            return VK_FORMAT_R16G16B16A16_SFLOAT;
        }
        return VK_FORMAT_UNDEFINED;
    }

    namespace
    {

        [[nodiscard]] SampleCountMask sampleMask(VkSampleCountFlags counts) noexcept
        {
            SampleCountMask mask = SampleCountMask::none;
            if ((counts & VK_SAMPLE_COUNT_1_BIT) != 0)
            {
                mask |= SampleCountMask::x1;
            }
            if ((counts & VK_SAMPLE_COUNT_2_BIT) != 0)
            {
                mask |= SampleCountMask::x2;
            }
            if ((counts & VK_SAMPLE_COUNT_4_BIT) != 0)
            {
                mask |= SampleCountMask::x4;
            }
            if ((counts & VK_SAMPLE_COUNT_8_BIT) != 0)
            {
                mask |= SampleCountMask::x8;
            }
            return mask;
        }

        [[nodiscard]] bool hasUnitImageTransferGranularity(const VkQueueFamilyProperties& family) noexcept
        {
            const auto& granularity = family.minImageTransferGranularity;
            return granularity.width == 1 && granularity.height == 1 && granularity.depth == 1;
        }

        class QueueFamilyClaims final
        {
        public:
            explicit QueueFamilyClaims(std::span<const VkQueueFamilyProperties> families)
                : families_(families), claimed_(families.size(), 0U)
            {
            }

            [[nodiscard]] bool hasUnclaimedQueue(std::uint32_t family) const noexcept
            {
                return family < families_.size() && claimed_[family] < families_[family].queueCount;
            }

            template<typename Accepts>
            [[nodiscard]] std::uint32_t firstUnclaimed(Accepts accepts) const noexcept
            {
                for (std::uint32_t family = 0; family < families_.size(); ++family)
                {
                    if (hasUnclaimedQueue(family) && accepts(families_[family]))
                    {
                        return family;
                    }
                }
                return VK_QUEUE_FAMILY_IGNORED;
            }

            [[nodiscard]] std::uint32_t claim(std::uint32_t family) noexcept
            {
                return claimed_[family]++;
            }

            [[nodiscard]] bool supports(std::uint32_t family, VkQueueFlags flags) const noexcept
            {
                return family < families_.size() && (families_[family].queueFlags & flags) == flags;
            }

        private:
            std::span<const VkQueueFamilyProperties> families_;
            std::vector<std::uint32_t> claimed_;
        };

        [[nodiscard]] std::uint32_t computeQueueFamily(const QueueFamilyClaims& claims, std::uint32_t graphics) noexcept
        {
            if (claims.supports(graphics, VK_QUEUE_COMPUTE_BIT) && claims.hasUnclaimedQueue(graphics))
            {
                return graphics;
            }
            const auto dedicated = claims.firstUnclaimed([](const VkQueueFamilyProperties& family) {
                return (family.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 &&
                       (family.queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0;
            });
            if (dedicated != VK_QUEUE_FAMILY_IGNORED)
            {
                return dedicated;
            }
            return claims.firstUnclaimed(
                [](const VkQueueFamilyProperties& family) { return (family.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0; });
        }

        [[nodiscard]] std::uint32_t transferQueueFamily(const QueueFamilyClaims& claims,
                                                        std::uint32_t graphics) noexcept
        {
            const auto dedicated = claims.firstUnclaimed([](const VkQueueFamilyProperties& family) {
                return (family.queueFlags & VK_QUEUE_TRANSFER_BIT) != 0 &&
                       (family.queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == 0 &&
                       hasUnitImageTransferGranularity(family);
            });
            if (dedicated != VK_QUEUE_FAMILY_IGNORED)
            {
                return dedicated;
            }
            if (claims.hasUnclaimedQueue(graphics))
            {
                return graphics;
            }
            return claims.firstUnclaimed([](const VkQueueFamilyProperties& family) {
                return (family.queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) != 0;
            });
        }

        struct QueueSlot final
        {
            std::uint32_t family = VK_QUEUE_FAMILY_IGNORED;
            std::uint32_t index = 0;
        };

        [[nodiscard]] QueueSlot claimQueueOrShareGraphics(QueueFamilyClaims& claims,
                                                          std::uint32_t candidate,
                                                          QueueSlot graphics) noexcept
        {
            if (candidate == VK_QUEUE_FAMILY_IGNORED)
            {
                return graphics;
            }
            return QueueSlot{.family = candidate, .index = claims.claim(candidate)};
        }

    } // namespace

    VulkanQueueFamilies selectVulkanQueueFamilies(std::span<const VkQueueFamilyProperties> families)
    {
        QueueFamilyClaims claims(families);
        VulkanQueueFamilies selected{};
        selected.graphics = claims.firstUnclaimed([](const VkQueueFamilyProperties& family) {
            return (family.queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
                   (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);
        });
        if (selected.graphics == VK_QUEUE_FAMILY_IGNORED)
        {
            selected.graphics = claims.firstUnclaimed(
                [](const VkQueueFamilyProperties& family) { return (family.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0; });
        }
        if (selected.graphics == VK_QUEUE_FAMILY_IGNORED)
        {
            return selected;
        }
        selected.graphicsIndex = claims.claim(selected.graphics);
        const QueueSlot graphics{.family = selected.graphics, .index = selected.graphicsIndex};
        const auto compute = claimQueueOrShareGraphics(claims, computeQueueFamily(claims, graphics.family), graphics);
        selected.compute = compute.family;
        selected.computeIndex = compute.index;
        const auto transfer = claimQueueOrShareGraphics(claims, transferQueueFamily(claims, graphics.family), graphics);
        selected.transfer = transfer.family;
        selected.transferIndex = transfer.index;
        return selected;
    }

    namespace
    {

        [[nodiscard]] std::vector<VkQueueFamilyProperties> queueFamilyProperties(
            const VulkanInstanceTable& instanceTable, VkPhysicalDevice physical)
        {
            std::vector<VkQueueFamilyProperties> families;
            if (instanceTable.vkGetPhysicalDeviceQueueFamilyProperties == nullptr)
            {
                return families;
            }
            std::uint32_t count = 0;
            instanceTable.vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
            families.resize(count);
            if (count != 0)
            {
                instanceTable.vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
                families.resize(count);
            }
            return families;
        }

        [[nodiscard]] VulkanQueueFamilyTraits queueFamilyTraits(std::span<const VkQueueFamilyProperties> families,
                                                                std::uint32_t family) noexcept
        {
            if (family >= families.size())
            {
                return {};
            }
            return VulkanQueueFamilyTraits{
                .queueFlags = families[family].queueFlags,
                .timestampValidBits = families[family].timestampValidBits,
            };
        }

    } // namespace

    VulkanInstanceInfo::VulkanInstanceInfo(VulkanInstanceInfo&& other) noexcept
    {
        *this = std::move(other);
    }

    VulkanInstanceInfo& VulkanInstanceInfo::operator=(VulkanInstanceInfo&& other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }
        reset();
        instance = other.instance;
        other.instance = VK_NULL_HANDLE;
        table = other.table;
        other.table = {};
        messenger = other.messenger;
        other.messenger = VK_NULL_HANDLE;
        sink = std::move(other.sink);
        report = other.report;
        validationEnabled = other.validationEnabled;
        validationRequested = other.validationRequested;
        validationMissing = other.validationMissing;
        debugUtils = other.debugUtils;
        return *this;
    }

    VulkanInstanceInfo::~VulkanInstanceInfo()
    {
        reset();
    }

    void VulkanInstanceInfo::reset() noexcept
    {
        if (messenger != VK_NULL_HANDLE && instance != VK_NULL_HANDLE &&
            table.vkDestroyDebugUtilsMessengerEXT != nullptr)
        {
            table.vkDestroyDebugUtilsMessengerEXT(instance, messenger, nullptr);
            messenger = VK_NULL_HANDLE;
        }
        if (instance != VK_NULL_HANDLE)
        {
            if (table.vkDestroyInstance != nullptr)
            {
                table.vkDestroyInstance(instance, nullptr);
            }
            instance = VK_NULL_HANDLE;
        }
        table = {};
        if (sink != nullptr)
        {
            sink->alive.store(false, std::memory_order_release);
        }
        sink.reset();
    }

    void destroyVulkanInstance(VulkanInstanceInfo& info) noexcept
    {
        info.reset();
    }

    namespace
    {

        [[nodiscard]] VkResult initializeVulkanLoaderLocked()
        {
            if (const auto loaded = volkInitialize(); loaded == VK_SUCCESS)
            {
                return loaded;
            }

#ifdef __APPLE__
            const char* sdk = std::getenv("VULKAN_SDK");
            if (sdk == nullptr || sdk[0] == '\0')
            {
                return VK_ERROR_INITIALIZATION_FAILED;
            }

            const std::string loaderPaths[] = {
                std::string(sdk) + "/lib/libvulkan.1.dylib",
                std::string(sdk) + "/macOS/lib/libvulkan.1.dylib",
            };
            static void* loader = nullptr;
            if (loader == nullptr)
            {
                for (const auto& loaderPath : loaderPaths)
                {
                    loader = dlopen(loaderPath.c_str(), RTLD_NOW | RTLD_LOCAL);
                    if (loader != nullptr)
                    {
                        break;
                    }
                }
            }
            if (loader == nullptr)
            {
                return VK_ERROR_INITIALIZATION_FAILED;
            }

            const auto getProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(loader, "vkGetInstanceProcAddr"));
            if (getProc == nullptr)
            {
                return VK_ERROR_INITIALIZATION_FAILED;
            }

            volkInitializeCustom(getProc);
            return VK_SUCCESS;
#else
            return VK_ERROR_INITIALIZATION_FAILED;
#endif
        }

        [[nodiscard]] VkResult initializeVulkanLoader(VulkanLoaderTable& outLoader)
        {
            static std::mutex loaderMutex;
            static std::optional<VulkanLoaderTable> cachedLoader;

            const std::scoped_lock lock(loaderMutex);
            if (cachedLoader.has_value())
            {
                outLoader = *cachedLoader;
                return VK_SUCCESS;
            }

            if (const auto loaded = initializeVulkanLoaderLocked(); loaded != VK_SUCCESS)
            {
                return loaded;
            }

            if (::vkGetInstanceProcAddr == nullptr || ::vkCreateInstance == nullptr ||
                ::vkEnumerateInstanceExtensionProperties == nullptr || ::vkEnumerateInstanceLayerProperties == nullptr)
            {
                return VK_ERROR_INITIALIZATION_FAILED;
            }

            cachedLoader = VulkanLoaderTable{
                .vkGetInstanceProcAddr = ::vkGetInstanceProcAddr,
                .vkCreateInstance = ::vkCreateInstance,
                .vkEnumerateInstanceVersion = ::vkEnumerateInstanceVersion,
                .vkEnumerateInstanceExtensionProperties = ::vkEnumerateInstanceExtensionProperties,
                .vkEnumerateInstanceLayerProperties = ::vkEnumerateInstanceLayerProperties,
            };
            outLoader = *cachedLoader;
            return VK_SUCCESS;
        }

        [[nodiscard]] Result<VulkanLoaderTable> acquireVulkanLoader()
        {
            VulkanLoaderTable loader{};
            if (const auto loaded = initializeVulkanLoader(loader); loaded != VK_SUCCESS)
            {
                return std::unexpected(
#ifdef _WIN32
                    selectionError(VulkanSelectionFailure::loaderUnavailable,
                                   "volk could not load vulkan-1.dll. Selection did not fall back to D3D12.",
                                   loaded));
#elifdef __APPLE__
                    selectionError(
                        VulkanSelectionFailure::loaderUnavailable,
                        "volk could not load libvulkan.1.dylib. Set the LunarG Vulkan SDK environment so its "
                        "loader is discoverable. Selection did not fall back to D3D12.",
                        loaded));
#else
                    selectionError(VulkanSelectionFailure::loaderUnavailable,
                                   "volk could not load libvulkan.so.1. Selection did not fall back to D3D12.",
                                   loaded));
#endif
            }
            return loader;
        }

        [[nodiscard]] bool canUseSurfaceMaintenance1(const VulkanInstanceReport& report) noexcept
        {
            return report.khrGetSurfaceCapabilities2 &&
                   (report.khrSurfaceMaintenance1 || report.extSurfaceMaintenance1);
        }

        [[nodiscard]] VulkanInstanceReport reportInstanceExtensions(
            const std::vector<VkExtensionProperties>& extensions) noexcept
        {
            VulkanInstanceReport report{};
#ifdef _WIN32
            report.apiMajor = vulkanWindowsProfileApiMajor;
            report.apiMinor = vulkanWindowsProfileApiMinor;
            report.khrWin32Surface = hasExtension(extensions, VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elifdef __APPLE__
            report.apiMajor = vulkanMacosProfileApiMajor;
            report.apiMinor = vulkanMacosProfileApiMinor;
            report.khrMetalSurface = hasExtension(extensions, "VK_EXT_metal_surface");
#else
            report.apiMajor = vulkanLinuxProfileApiMajor;
            report.apiMinor = vulkanLinuxProfileApiMinor;
            report.khrWaylandSurface = hasExtension(extensions, "VK_KHR_wayland_surface");
            report.khrXcbSurface = hasExtension(extensions, "VK_KHR_xcb_surface");
#endif
            report.khrSurface = hasExtension(extensions, VK_KHR_SURFACE_EXTENSION_NAME);
            report.khrGetSurfaceCapabilities2 = hasExtension(extensions, "VK_KHR_get_surface_capabilities2");
            report.khrSurfaceMaintenance1 = hasExtension(extensions, "VK_KHR_surface_maintenance1");
            report.extSurfaceMaintenance1 = hasExtension(extensions, "VK_EXT_surface_maintenance1");
            report.khrPortabilityEnumeration = hasExtension(extensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            if (!canUseSurfaceMaintenance1(report))
            {
                report.khrSurfaceMaintenance1 = false;
                report.extSurfaceMaintenance1 = false;
            }
            return report;
        }

        [[nodiscard]] std::vector<const char*> selectInstanceExtensions(const VulkanInstanceInfo& info,
                                                                        bool enablePortabilityEnumeration,
                                                                        bool enableValidationFeatures)
        {
            const auto& report = info.report;
            std::vector<const char*> names;
            appendIf(names, report.khrSurface, VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef _WIN32
            appendIf(names, report.khrWin32Surface, VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif !defined(__APPLE__)
            appendIf(names, report.khrWaylandSurface, "VK_KHR_wayland_surface");
            appendIf(names, report.khrXcbSurface, "VK_KHR_xcb_surface");
#endif
            appendIf(names, enablePortabilityEnumeration, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            if (canUseSurfaceMaintenance1(report))
            {
                names.push_back("VK_KHR_get_surface_capabilities2");
                names.push_back(report.khrSurfaceMaintenance1 ? "VK_KHR_surface_maintenance1"
                                                              : "VK_EXT_surface_maintenance1");
            }
            appendIf(names, info.debugUtils, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            appendIf(names, enableValidationFeatures, VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
            return names;
        }

        [[nodiscard]] Result<VulkanValidationLayerPlan> resolveValidationLayerPlan(
            bool validationRequested, const DeviceDesc& desc, const std::vector<VkLayerProperties>& layers)
        {
            const bool haveValidation = hasLayer(layers, "VK_LAYER_KHRONOS_validation");
            const auto validationPlan = planVulkanValidationLayers(
                validationRequested || desc.requireValidation, desc.requireValidation, haveValidation);
            if (validationPlan == VulkanValidationLayerPlan::requiredMissing)
            {
                return std::unexpected(Error{
                    ErrorCode::validationError,
                    0,
                    "VK_LAYER_KHRONOS_validation is required but was not found. Selection did not fall back to D3D12.",
                });
            }
            return validationPlan;
        }

        [[nodiscard]] std::vector<VkValidationFeatureEnableEXT> selectValidationFeatureEnables(bool validationEnabled,
                                                                                               ValidationLevel level)
        {
            std::vector<VkValidationFeatureEnableEXT> validationEnables;
            if (validationEnabled)
            {
                validationEnables.push_back(VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT);
                if (level == ValidationLevel::gpu)
                {
                    validationEnables.push_back(VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT);
                    validationEnables.push_back(VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT);
                }
            }
            return validationEnables;
        }

        [[nodiscard]] VkDebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo(VulkanDiagnosticSink* sink) noexcept
        {
            return VkDebugUtilsMessengerCreateInfoEXT{
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                .pNext = {},
                .flags = {},
                .messageSeverity =
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
                .pfnUserCallback = &vulkanDebugUtilsCallback,
                .pUserData = sink,
            };
        }

        struct InstanceCreateSelection final
        {
            std::vector<const char*> extensions;
            std::vector<const char*> layers;
            std::vector<VkValidationFeatureEnableEXT> validationEnables;
            bool enumeratePortability = false;
        };

        [[nodiscard]] Status createInstanceAndLoadTable(VulkanInstanceInfo& info,
                                                        const VulkanLoaderTable& loader,
                                                        const DeviceDesc& desc,
                                                        const InstanceCreateSelection& selection)
        {
            VkValidationFeaturesEXT validationFeatures{
                .sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
                .pNext = {},
                .enabledValidationFeatureCount = static_cast<std::uint32_t>(selection.validationEnables.size()),
                .pEnabledValidationFeatures =
                    selection.validationEnables.empty() ? nullptr : selection.validationEnables.data(),
                .disabledValidationFeatureCount = {},
                .pDisabledValidationFeatures = {},
            };
            auto messengerInfo = debugMessengerCreateInfo(info.sink.get());
            void const* next = nullptr;
            if (info.debugUtils)
            {
                messengerInfo.pNext = next;
                next = &messengerInfo;
            }
            if (!selection.validationEnables.empty())
            {
                validationFeatures.pNext = next;
                next = &validationFeatures;
            }

            const auto app = VkApplicationInfo{
                .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                .pNext = {},
                .pApplicationName = desc.applicationName.empty() ? nullptr : desc.applicationName.c_str(),
                .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
                .pEngineName = desc.engineName.empty() ? nullptr : desc.engineName.c_str(),
                .engineVersion = VK_MAKE_VERSION(0, 1, 0),
                .apiVersion = VK_API_VERSION_1_3,
            };
            auto createInfo = VkInstanceCreateInfo{
                .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                .pNext = next,
                .flags = selection.enumeratePortability
                             ? static_cast<VkInstanceCreateFlags>(VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR)
                             : VkInstanceCreateFlags{},
                .pApplicationInfo = &app,
                .enabledLayerCount = static_cast<std::uint32_t>(selection.layers.size()),
                .ppEnabledLayerNames = selection.layers.empty() ? nullptr : selection.layers.data(),
                .enabledExtensionCount = static_cast<std::uint32_t>(selection.extensions.size()),
                .ppEnabledExtensionNames = selection.extensions.empty() ? nullptr : selection.extensions.data(),
            };

            const auto created = loader.vkCreateInstance(&createInfo, nullptr, &info.instance);
            if (created != VK_SUCCESS)
            {
                return std::unexpected(makeVulkanError(ErrorCode::backendUnavailable, created, "vkCreateInstance"));
            }
            loadVulkanInstanceTable(info.table, info.instance, loader.vkGetInstanceProcAddr);
            if (info.table.vkDestroyInstance == nullptr || info.table.vkGetDeviceProcAddr == nullptr ||
                info.table.vkEnumeratePhysicalDevices == nullptr || info.table.vkCreateDevice == nullptr)
            {
                return std::unexpected(makeVulkanError(
                    ErrorCode::backendUnavailable, VK_ERROR_INITIALIZATION_FAILED, "vkGetInstanceProcAddr"));
            }
            return {};
        }

        [[nodiscard]] Status createDebugMessenger(VulkanInstanceInfo& info, bool requireValidation)
        {
            if (!info.debugUtils || info.table.vkCreateDebugUtilsMessengerEXT == nullptr)
            {
                return {};
            }
            const auto messengerInfo = debugMessengerCreateInfo(info.sink.get());
            const auto messengerCreated =
                info.table.vkCreateDebugUtilsMessengerEXT(info.instance, &messengerInfo, nullptr, &info.messenger);
            if (messengerCreated != VK_SUCCESS && requireValidation)
            {
                return std::unexpected(
                    makeVulkanError(ErrorCode::validationError, messengerCreated, "vkCreateDebugUtilsMessengerEXT"));
            }
            return {};
        }

    } // namespace

    Status checkVulkanInstanceVersion(PFN_vkEnumerateInstanceVersion enumerateInstanceVersion)
    {
        std::uint32_t version = VK_API_VERSION_1_0;
        if (enumerateInstanceVersion != nullptr)
        {
            if (const auto queried = enumerateInstanceVersion(&version); queried != VK_SUCCESS)
            {
                return std::unexpected(selectionError(
                    VulkanSelectionFailure::loaderUnavailable, "vkEnumerateInstanceVersion failed.", queried));
            }
        }
        if (supportsApiVersion(version, VK_API_VERSION_1_3))
        {
            return {};
        }
        std::ostringstream message;
        message << "The Vulkan loader reports instance version " << VK_API_VERSION_MAJOR(version) << '.'
                << VK_API_VERSION_MINOR(version)
                << ", below the Vulkan 1.3 that RachisRHI requires. Update the Vulkan loader. Selection did not fall "
                   "back to D3D12.";
        return std::unexpected(selectionError(VulkanSelectionFailure::loaderUnavailable, message.str()));
    }

    Result<VulkanInstanceInfo> createVulkanInstance(const DeviceDesc& desc)
    {
        auto loader = acquireVulkanLoader();
        if (!loader)
        {
            return std::unexpected(std::move(loader.error()));
        }
        if (auto version = checkVulkanInstanceVersion(loader->vkEnumerateInstanceVersion); !version)
        {
            return std::unexpected(std::move(version.error()));
        }
        auto extensions = instanceExtensions(*loader);
        if (!extensions)
        {
            return std::unexpected(std::move(extensions.error()));
        }
        auto layers = instanceLayers(*loader);
        if (!layers)
        {
            return std::unexpected(std::move(layers.error()));
        }

        VulkanInstanceInfo info{};
        info.report = reportInstanceExtensions(*extensions);
        info.debugUtils = hasExtension(*extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        info.validationRequested = desc.validation != ValidationLevel::disabled;

        const auto portabilityPolicy = selectVulkanInstanceCreatePolicy(info.report.khrPortabilityEnumeration);
        if (hostRequiresPortabilityEnumeration() && !portabilityPolicy.enablePortabilityEnumeration)
        {
            return std::unexpected(selectionError(
                VulkanSelectionFailure::loaderUnavailable,
                "VK_KHR_portability_enumeration is required to enumerate MoltenVK on macOS. Set the LunarG Vulkan "
                "SDK environment so its loader and layers are discoverable. Selection did not fall back to D3D12."));
        }
        const bool haveValidationFeatures = hasExtension(*extensions, VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);

        InstanceCreateSelection selection{};
        selection.enumeratePortability = portabilityPolicy.enumeratePortabilityBit;
        selection.extensions = selectInstanceExtensions(
            info, portabilityPolicy.enablePortabilityEnumeration, haveValidationFeatures && info.validationRequested);
        auto validationPlan = resolveValidationLayerPlan(info.validationRequested, desc, *layers);
        if (!validationPlan)
        {
            return std::unexpected(std::move(validationPlan.error()));
        }
        info.validationEnabled = *validationPlan == VulkanValidationLayerPlan::enableKhronos;
        info.validationMissing = *validationPlan == VulkanValidationLayerPlan::optionalMissing;
        appendIf(selection.layers, info.validationEnabled, "VK_LAYER_KHRONOS_validation");
        selection.validationEnables = selectValidationFeatureEnables(info.validationEnabled, desc.validation);

        info.sink = std::make_unique<VulkanDiagnosticSink>();
        info.sink->callback = desc.diagnosticCallback;
        info.sink->userData = desc.diagnosticUserData;

        if (auto created = createInstanceAndLoadTable(info, *loader, desc, selection); !created)
        {
            return std::unexpected(std::move(created.error()));
        }
        if (auto messenger = createDebugMessenger(info, desc.requireValidation); !messenger)
        {
            return std::unexpected(std::move(messenger.error()));
        }

        if (info.validationMissing)
        {
            info.sink->publish(DiagnosticSeverity::warning,
                               0,
                               "VK_LAYER_KHRONOS_validation was requested but is not installed. Optional validation is "
                               "absent; production mode does not require the SDK.");
        }
        return info;
    }

    namespace
    {

        template<typename Head, typename Extension>
        void prependToChain(Head& head, Extension& extension, VkStructureType type) noexcept
        {
            extension.sType = type;
            extension.pNext = head.pNext;
            head.pNext = &extension;
        }

        template<typename Features>
        [[nodiscard]] Features queryExtensionFeatures(const VulkanInstanceTable& instanceTable,
                                                      VkPhysicalDevice physical,
                                                      VkStructureType type) noexcept
        {
            Features features{};
            features.sType = type;

            VkPhysicalDeviceFeatures2 query{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                .pNext = &features,
                .features = {},
            };
            instanceTable.vkGetPhysicalDeviceFeatures2(physical, &query);
            return features;
        }

        template<typename Properties>
        [[nodiscard]] Properties queryExtensionProperties(const VulkanInstanceTable& instanceTable,
                                                          VkPhysicalDevice physical,
                                                          VkStructureType type) noexcept
        {
            Properties properties{};
            properties.sType = type;

            VkPhysicalDeviceProperties2 query{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                .pNext = &properties,
                .properties = {},
            };
            instanceTable.vkGetPhysicalDeviceProperties2(physical, &query);
            return properties;
        }

#ifdef VK_KHR_swapchain_maintenance1
        using SwapchainMaintenance1Features = VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR;
        constexpr VkStructureType swapchainMaintenance1FeaturesType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR;
#elifdef VK_EXT_swapchain_maintenance1
        using SwapchainMaintenance1Features = VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT;
        constexpr VkStructureType swapchainMaintenance1FeaturesType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT;
#endif

        [[nodiscard]] Result<std::vector<VkPhysicalDevice>> enumeratePhysicalDevices(
            VkInstance instance, const VulkanInstanceTable& instanceTable)
        {
            std::uint32_t count = 0;
            const auto first = instanceTable.vkEnumeratePhysicalDevices(instance, &count, nullptr);
            if (first != VK_SUCCESS)
            {
                return std::unexpected(selectionError(
                    VulkanSelectionFailure::loaderUnavailable, "vkEnumeratePhysicalDevices failed.", first));
            }
            std::vector<VkPhysicalDevice> devices(count);
            if (count != 0)
            {
                const auto second = instanceTable.vkEnumeratePhysicalDevices(instance, &count, devices.data());
                if (second != VK_SUCCESS)
                {
                    return std::unexpected(selectionError(
                        VulkanSelectionFailure::loaderUnavailable, "vkEnumeratePhysicalDevices failed.", second));
                }
            }
            return devices;
        }

        struct PhysicalDevicePropertyChain final
        {
            VkPhysicalDeviceDriverProperties driver{};
            VkPhysicalDeviceIDProperties id{};
            VkPhysicalDeviceVulkan11Properties v11{};
            VkPhysicalDeviceVulkan12Properties v12{};
            VkPhysicalDeviceVulkan13Properties v13{};
        };

        void queryPropertyChain(const VulkanInstanceTable& instanceTable,
                                VkPhysicalDevice physical,
                                std::uint32_t apiVersion,
                                PhysicalDevicePropertyChain& chain) noexcept
        {
            if (!supportsApiVersion(apiVersion, VK_API_VERSION_1_1))
            {
                return;
            }
            VkPhysicalDeviceProperties2 query{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                .pNext = {},
                .properties = {},
            };
            if (supportsApiVersion(apiVersion, VK_API_VERSION_1_2))
            {
                prependToChain(query, chain.driver, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES);
            }
            prependToChain(query, chain.id, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES);
            if (supportsApiVersion(apiVersion, VK_API_VERSION_1_3))
            {
                prependToChain(query, chain.v11, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES);
                prependToChain(query, chain.v12, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES);
                prependToChain(query, chain.v13, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES);
            }
            instanceTable.vkGetPhysicalDeviceProperties2(physical, &query);
        }

        struct PhysicalDeviceFeatureChain final
        {
            VkPhysicalDeviceFeatures base{};
            VkPhysicalDeviceVulkan11Features v11{};
            VkPhysicalDeviceVulkan12Features v12{};
            VkPhysicalDeviceVulkan13Features v13{};
#ifdef VK_KHR_fragment_shader_barycentric
            VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR fragmentBarycentric{};
#endif
#ifdef VK_EXT_shader_image_atomic_int64
            VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT imageInt64Atomics{};
#endif
        };

        void queryFeatureChain(const VulkanInstanceTable& instanceTable,
                               VkPhysicalDevice physical,
                               PhysicalDeviceFeatureChain& chain) noexcept
        {
            VkPhysicalDeviceFeatures2 query{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                .pNext = {},
                .features = {},
            };
            prependToChain(query, chain.v13, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
#ifdef VK_KHR_fragment_shader_barycentric
            prependToChain(query,
                           chain.fragmentBarycentric,
                           VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR);
#endif
#ifdef VK_EXT_shader_image_atomic_int64
            prependToChain(query,
                           chain.imageInt64Atomics,
                           VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT);
#endif
            prependToChain(query, chain.v12, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
            prependToChain(query, chain.v11, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
            instanceTable.vkGetPhysicalDeviceFeatures2(physical, &query);
            chain.base = query.features;
        }

        struct PhysicalDeviceProbe final
        {
            const VulkanInstanceTable& instanceTable;
            VkPhysicalDevice physical = VK_NULL_HANDLE;
            const std::vector<VkExtensionProperties>& extensions;
        };

        void pointReportAtOwnedNames(VulkanPhysicalDeviceRecord& record) noexcept
        {
            record.report.deviceName = record.deviceName;
            record.report.driverId = record.driverId;
        }

        void recordDriverIdentity(VulkanPhysicalDeviceRecord& record, const VkPhysicalDeviceDriverProperties& driver)
        {
            record.deviceName = record.properties.deviceName;
            record.driverId = std::string(driverIdName(driver.driverID));
            record.moltenVk = driver.driverID == VK_DRIVER_ID_MOLTENVK;
            record.report.apiMajor = static_cast<std::uint16_t>(VK_API_VERSION_MAJOR(record.properties.apiVersion));
            record.report.apiMinor = static_cast<std::uint16_t>(VK_API_VERSION_MINOR(record.properties.apiVersion));
            record.report.apiPatch = static_cast<std::uint16_t>(VK_API_VERSION_PATCH(record.properties.apiVersion));
            pointReportAtOwnedNames(record);
            record.report.vendorId = record.properties.vendorID;
            record.report.deviceId = record.properties.deviceID;
            record.report.software = record.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
            record.report.translationLayer = driver.driverID == VK_DRIVER_ID_MESA_DOZEN ||
                                             record.deviceName.contains("Dozen") ||
                                             std::string_view(driver.driverName).contains("Dozen");
        }

        void recordUpdateAfterBindLimits(VulkanPhysicalDeviceRecord& record,
                                         const VkPhysicalDeviceVulkan12Properties& properties) noexcept
        {
            record.maxPerStageDescriptorUpdateAfterBindSamplers =
                properties.maxPerStageDescriptorUpdateAfterBindSamplers;
            record.maxPerStageDescriptorUpdateAfterBindUniformBuffers =
                properties.maxPerStageDescriptorUpdateAfterBindUniformBuffers;
            record.maxPerStageDescriptorUpdateAfterBindStorageBuffers =
                properties.maxPerStageDescriptorUpdateAfterBindStorageBuffers;
            record.maxPerStageDescriptorUpdateAfterBindSampledImages =
                properties.maxPerStageDescriptorUpdateAfterBindSampledImages;
            record.maxPerStageDescriptorUpdateAfterBindStorageImages =
                properties.maxPerStageDescriptorUpdateAfterBindStorageImages;
            record.maxPerStageUpdateAfterBindResources = properties.maxPerStageUpdateAfterBindResources;
            record.maxDescriptorSetUpdateAfterBindSamplers = properties.maxDescriptorSetUpdateAfterBindSamplers;
            record.maxDescriptorSetUpdateAfterBindUniformBuffers =
                properties.maxDescriptorSetUpdateAfterBindUniformBuffers;
            record.maxDescriptorSetUpdateAfterBindUniformBuffersDynamic =
                properties.maxDescriptorSetUpdateAfterBindUniformBuffersDynamic;
            record.maxDescriptorSetUpdateAfterBindStorageBuffers =
                properties.maxDescriptorSetUpdateAfterBindStorageBuffers;
            record.maxDescriptorSetUpdateAfterBindStorageBuffersDynamic =
                properties.maxDescriptorSetUpdateAfterBindStorageBuffersDynamic;
            record.maxDescriptorSetUpdateAfterBindSampledImages =
                properties.maxDescriptorSetUpdateAfterBindSampledImages;
            record.maxDescriptorSetUpdateAfterBindStorageImages =
                properties.maxDescriptorSetUpdateAfterBindStorageImages;
        }

        void recordCoreFeatures(VulkanPhysicalDeviceRecord& record, const PhysicalDeviceFeatureChain& features) noexcept
        {
            record.report.dynamicRendering = features.v13.dynamicRendering == VK_TRUE;
            record.report.synchronization2 = features.v13.synchronization2 == VK_TRUE;
            record.maintenance4 = features.v13.maintenance4 == VK_TRUE;
            record.subgroupSizeControl =
                features.v13.subgroupSizeControl == VK_TRUE && features.v13.computeFullSubgroups == VK_TRUE;
            record.report.timelineSemaphore = features.v12.timelineSemaphore == VK_TRUE;
            record.report.independentBlend = features.base.independentBlend == VK_TRUE;
            record.report.samplerAnisotropy = features.base.samplerAnisotropy == VK_TRUE;
            record.report.textureCompressionBC = features.base.textureCompressionBC == VK_TRUE;
            record.report.shaderSampledImageArrayDynamicIndexing =
                features.base.shaderSampledImageArrayDynamicIndexing == VK_TRUE;
            record.runtimeDescriptorArray = features.v12.runtimeDescriptorArray == VK_TRUE;
            record.descriptorBindingPartiallyBound = features.v12.descriptorBindingPartiallyBound == VK_TRUE;
            record.descriptorBindingVariableDescriptorCount =
                features.v12.descriptorBindingVariableDescriptorCount == VK_TRUE;
            record.descriptorBindingUpdateUnusedWhilePending =
                features.v12.descriptorBindingUpdateUnusedWhilePending == VK_TRUE;
            record.descriptorBindingSampledImageUpdateAfterBind =
                features.v12.descriptorBindingSampledImageUpdateAfterBind == VK_TRUE;
            record.descriptorBindingStorageImageUpdateAfterBind =
                features.v12.descriptorBindingStorageImageUpdateAfterBind == VK_TRUE;
            record.descriptorBindingStorageBufferUpdateAfterBind =
                features.v12.descriptorBindingStorageBufferUpdateAfterBind == VK_TRUE;
            record.shaderStorageBufferArrayNonUniformIndexing =
                features.v12.shaderStorageBufferArrayNonUniformIndexing == VK_TRUE;
            record.shaderSampledImageArrayNonUniformIndexing =
                features.v12.shaderSampledImageArrayNonUniformIndexing == VK_TRUE;
            record.shaderStorageImageArrayNonUniformIndexing =
                features.v12.shaderStorageImageArrayNonUniformIndexing == VK_TRUE;
            record.shaderFloat16 = features.v12.shaderFloat16 == VK_TRUE;
            record.shaderInt16 = features.base.shaderInt16 == VK_TRUE;
            record.shaderInt64 = features.base.shaderInt64 == VK_TRUE;
            record.shaderSharedInt64Atomics = features.v12.shaderSharedInt64Atomics == VK_TRUE;
            record.shaderBufferInt64Atomics = features.v12.shaderBufferInt64Atomics == VK_TRUE;
            record.geometryShaders = features.base.geometryShader == VK_TRUE;
            record.tessellationShaders = features.base.tessellationShader == VK_TRUE;
            record.tessellationMaxPatchControlPoints =
                record.tessellationShaders ? record.properties.limits.maxTessellationPatchSize : 0;
            record.shaderOutputLayer = features.v12.shaderOutputLayer == VK_TRUE;
            record.multiview = features.v11.multiview == VK_TRUE;
            record.multiviewGeometryShader = features.v11.multiviewGeometryShader == VK_TRUE;
            record.multiviewTessellationShader = features.v11.multiviewTessellationShader == VK_TRUE;
            record.multiViewport = features.base.multiViewport == VK_TRUE;
            record.depthBounds = features.base.depthBounds == VK_TRUE;
            record.fillModeNonSolid = features.base.fillModeNonSolid == VK_TRUE;
            record.depthClamp = features.base.depthClamp == VK_TRUE;
            record.depthBiasClamp = features.base.depthBiasClamp == VK_TRUE;
            record.separateDepthStencilLayouts = features.v12.separateDepthStencilLayouts == VK_TRUE;
            record.logicOp = features.base.logicOp == VK_TRUE;
            record.shaderClipDistance = features.base.shaderClipDistance == VK_TRUE;
            record.shaderCullDistance = features.base.shaderCullDistance == VK_TRUE;
            record.samplerFilterMinmax = features.v12.samplerFilterMinmax == VK_TRUE;
            record.shaderDrawParameters = features.v11.shaderDrawParameters == VK_TRUE;
            record.occlusionQueryPrecise = features.base.occlusionQueryPrecise == VK_TRUE;
            record.pipelineStatisticsQuery = features.base.pipelineStatisticsQuery == VK_TRUE;
            record.hostQueryReset = features.v12.hostQueryReset == VK_TRUE;
            record.report.hostQueryReset = record.hostQueryReset;
            record.fragmentStoresAndAtomics = features.base.fragmentStoresAndAtomics == VK_TRUE;
            record.multiDrawIndirect = features.base.multiDrawIndirect == VK_TRUE;
            record.drawIndirectCount = features.v12.drawIndirectCount == VK_TRUE;
            record.bufferDeviceAddress = features.v12.bufferDeviceAddress == VK_TRUE;
        }

        void recordOptionalExtensionSupport(VulkanPhysicalDeviceRecord& record,
                                            const PhysicalDeviceProbe& device,
                                            const PhysicalDeviceFeatureChain& features) noexcept
        {
            record.shaderImageInt64Atomics =
#ifdef VK_EXT_shader_image_atomic_int64
                hasExtension(device.extensions, VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME) &&
                features.imageInt64Atomics.shaderImageInt64Atomics == VK_TRUE &&
                formatHasOptimal(device.instanceTable,
                                 device.physical,
                                 VK_FORMAT_R64_UINT,
                                 VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT);
#else
                false;
#endif
            record.conservativeRasterization =
                hasExtension(device.extensions, VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME);
            record.fragmentBarycentric =
#ifdef VK_KHR_fragment_shader_barycentric
                hasExtension(device.extensions, VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME) &&
                features.fragmentBarycentric.fragmentShaderBarycentric == VK_TRUE;
#else
                false;
#endif
            record.calibratedTimestamps =
#ifdef VK_EXT_calibrated_timestamps
                hasExtension(device.extensions, VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
#else
                false;
#endif
            record.memoryBudget = hasExtension(device.extensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
            record.memoryPriority = false;
#ifdef VK_EXT_memory_priority
            if (hasExtension(device.extensions, VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME))
            {
                const auto priority = queryExtensionFeatures<VkPhysicalDeviceMemoryPriorityFeaturesEXT>(
                    device.instanceTable,
                    device.physical,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT);
                record.memoryPriority = priority.memoryPriority == VK_TRUE;
            }
#endif
            record.conditionalRendering = false;
#ifdef VK_EXT_conditional_rendering
            if (hasExtension(device.extensions, VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME))
            {
                const auto conditional = queryExtensionFeatures<VkPhysicalDeviceConditionalRenderingFeaturesEXT>(
                    device.instanceTable,
                    device.physical,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT);
                record.conditionalRendering = conditional.conditionalRendering == VK_TRUE;
            }
#endif
            record.fragmentShaderPixelInterlock = false;
#ifdef VK_EXT_fragment_shader_interlock
            if (record.fragmentStoresAndAtomics &&
                hasExtension(device.extensions, VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME))
            {
                const auto interlock = queryExtensionFeatures<VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT>(
                    device.instanceTable,
                    device.physical,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT);
                record.fragmentShaderPixelInterlock = interlock.fragmentShaderPixelInterlock == VK_TRUE;
            }
#endif
            record.maintenance5 = false;
#ifdef VK_KHR_maintenance5
            if (hasExtension(device.extensions, VK_KHR_MAINTENANCE_5_EXTENSION_NAME))
            {
                const auto maintenance5 = queryExtensionFeatures<VkPhysicalDeviceMaintenance5FeaturesKHR>(
                    device.instanceTable,
                    device.physical,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR);
                record.maintenance5 = maintenance5.maintenance5 == VK_TRUE;
            }
#endif
#ifdef RACHIS_TESTING
            if (test::hideMaintenance5().load(std::memory_order_relaxed))
            {
                record.maintenance5 = false;
            }
#endif
        }

#ifdef VK_EXT_mesh_shader
        [[nodiscard]] MeshShaderCapabilities meshShaderLimits(const VkPhysicalDeviceMeshShaderPropertiesEXT& properties,
                                                              bool taskShaders) noexcept
        {
            const auto meshCountX = properties.maxMeshWorkGroupCount[0];
            const auto meshCountY = properties.maxMeshWorkGroupCount[1];
            const auto meshCountZ = properties.maxMeshWorkGroupCount[2];
            const auto taskCountX = taskShaders ? properties.maxTaskWorkGroupCount[0] : meshCountX;
            const auto taskCountY = taskShaders ? properties.maxTaskWorkGroupCount[1] : meshCountY;
            const auto taskCountZ = taskShaders ? properties.maxTaskWorkGroupCount[2] : meshCountZ;
            return MeshShaderCapabilities{
                .maxTaskWorkgroupInvocations = taskShaders ? properties.maxTaskWorkGroupInvocations : 0,
                .maxTaskPayloadBytes = taskShaders ? properties.maxTaskPayloadSize : 0,
                .maxMeshWorkgroupInvocations = properties.maxMeshWorkGroupInvocations,
                .maxMeshOutputVertices = properties.maxMeshOutputVertices,
                .maxMeshOutputPrimitives = properties.maxMeshOutputPrimitives,
                .maxMeshOutputMemoryBytes = properties.maxMeshOutputMemorySize,
                .maxWorkgroupCount =
                    {
                        std::min(meshCountX, taskCountX),
                        std::min(meshCountY, taskCountY),
                        std::min(meshCountZ, taskCountZ),
                    },
                .maxTotalWorkgroups = properties.maxMeshWorkGroupTotalCount,
            };
        }

        void recordMeshShaderSupport(VulkanPhysicalDeviceRecord& record, const PhysicalDeviceProbe& device) noexcept
        {
            if (!hasExtension(device.extensions, VK_EXT_MESH_SHADER_EXTENSION_NAME))
            {
                return;
            }
            const auto features = queryExtensionFeatures<VkPhysicalDeviceMeshShaderFeaturesEXT>(
                device.instanceTable, device.physical, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT);
            const auto properties = queryExtensionProperties<VkPhysicalDeviceMeshShaderPropertiesEXT>(
                device.instanceTable, device.physical, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT);
            record.meshShaders = features.meshShader == VK_TRUE;
            record.taskShaders = record.meshShaders && features.taskShader == VK_TRUE;
            record.meshShaderQueries = record.meshShaders && features.meshShaderQueries == VK_TRUE;
            record.multiviewMeshShader = record.meshShaders && features.multiviewMeshShader == VK_TRUE;
            if (record.meshShaders)
            {
                record.meshShader = meshShaderLimits(properties, record.taskShaders);
            }
        }
#endif

#ifdef VK_KHR_fragment_shading_rate
        [[nodiscard]] std::vector<VkPhysicalDeviceFragmentShadingRateKHR> querySupportedShadingRates(
            const PhysicalDeviceProbe& device)
        {
            std::uint32_t rateCount = 0;
            const auto counted =
                device.instanceTable.vkGetPhysicalDeviceFragmentShadingRatesKHR(device.physical, &rateCount, nullptr);
            std::vector<VkPhysicalDeviceFragmentShadingRateKHR> rates;
            if (counted == VK_SUCCESS && rateCount != 0)
            {
                rates.resize(rateCount,
                             VkPhysicalDeviceFragmentShadingRateKHR{
                                 .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_KHR,
                                 .pNext = {},
                                 .sampleCounts = {},
                                 .fragmentSize = {},
                             });
                const auto queried = device.instanceTable.vkGetPhysicalDeviceFragmentShadingRatesKHR(
                    device.physical, &rateCount, rates.data());
                if (queried != VK_SUCCESS)
                {
                    rates.clear();
                }
            }
            return rates;
        }

        [[nodiscard]] bool supportsShadingRate(const std::vector<VkPhysicalDeviceFragmentShadingRateKHR>& rates,
                                               std::uint32_t width,
                                               std::uint32_t height) noexcept
        {
            return std::ranges::any_of(rates, [&](const VkPhysicalDeviceFragmentShadingRateKHR& rate) {
                return rate.fragmentSize.width == width && rate.fragmentSize.height == height && rate.sampleCounts != 0;
            });
        }

        [[nodiscard]] std::uint32_t selectShadingRateImageTileSize(
            const VkPhysicalDeviceFragmentShadingRatePropertiesKHR& properties) noexcept
        {
            const auto minWidth = properties.minFragmentShadingRateAttachmentTexelSize.width;
            const auto minHeight = properties.minFragmentShadingRateAttachmentTexelSize.height;
            const auto maxWidth = properties.maxFragmentShadingRateAttachmentTexelSize.width;
            const auto maxHeight = properties.maxFragmentShadingRateAttachmentTexelSize.height;
            const auto minSquare = std::max(minWidth, minHeight);
            const auto maxSquare = std::min(maxWidth, maxHeight);
            for (const auto candidate : {std::uint32_t{16}, std::uint32_t{8}, std::uint32_t{32}})
            {
                if (candidate >= minSquare && candidate <= maxSquare)
                {
                    return candidate;
                }
            }
            if (minSquare > 0 && minSquare <= maxSquare && (minSquare & (minSquare - 1U)) == 0)
            {
                return minSquare;
            }
            return 0;
        }

        void recordShadingRateImageSupport(VulkanPhysicalDeviceRecord& record,
                                           const PhysicalDeviceProbe& device) noexcept
        {
            const auto properties = queryExtensionProperties<VkPhysicalDeviceFragmentShadingRatePropertiesKHR>(
                device.instanceTable,
                device.physical,
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES_KHR);
            const auto tile = selectShadingRateImageTileSize(properties);
            const bool rateImageFormat = formatHasOptimal(device.instanceTable,
                                                          device.physical,
                                                          VK_FORMAT_R8_UINT,
                                                          VK_FORMAT_FEATURE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR);
            if (tile != 0 && rateImageFormat && properties.fragmentShadingRateNonTrivialCombinerOps == VK_TRUE)
            {
                record.variableRateShadingTier2 = true;
                record.shadingRateImageTileSize = tile;
                record.perPrimitiveShadingRateWithViewportIndexing =
                    properties.primitiveFragmentShadingRateWithMultipleViewports == VK_TRUE;
            }
        }

        void recordVariableRateShadingSupport(VulkanPhysicalDeviceRecord& record, const PhysicalDeviceProbe& device)
        {
            if (!hasExtension(device.extensions, VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME))
            {
                return;
            }
            const auto features = queryExtensionFeatures<VkPhysicalDeviceFragmentShadingRateFeaturesKHR>(
                device.instanceTable,
                device.physical,
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR);
            if (features.pipelineFragmentShadingRate != VK_TRUE ||
                device.instanceTable.vkGetPhysicalDeviceFragmentShadingRatesKHR == nullptr)
            {
                return;
            }
            const auto rates = querySupportedShadingRates(device);
            record.variableRateShading = supportsShadingRate(rates, 1, 1) && supportsShadingRate(rates, 2, 2);
            record.additionalShadingRates = record.variableRateShading && supportsShadingRate(rates, 2, 4) &&
                                            supportsShadingRate(rates, 4, 2) && supportsShadingRate(rates, 4, 4);
            if (record.variableRateShading && features.primitiveFragmentShadingRate == VK_TRUE &&
                features.attachmentFragmentShadingRate == VK_TRUE)
            {
                recordShadingRateImageSupport(record, device);
            }
        }
#endif

#ifdef VK_KHR_acceleration_structure
        void recordAccelerationStructureSupport(VulkanPhysicalDeviceRecord& record,
                                                const PhysicalDeviceProbe& device) noexcept
        {
            if (!record.bufferDeviceAddress ||
                !hasExtension(device.extensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME))
            {
                return;
            }
            const auto features = queryExtensionFeatures<VkPhysicalDeviceAccelerationStructureFeaturesKHR>(
                device.instanceTable,
                device.physical,
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR);
            if (features.accelerationStructure != VK_TRUE)
            {
                return;
            }
            const auto properties = queryExtensionProperties<VkPhysicalDeviceAccelerationStructurePropertiesKHR>(
                device.instanceTable,
                device.physical,
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR);
            record.accelerationStructures =
                properties.maxGeometryCount > 0 && properties.maxInstanceCount > 0 && properties.maxPrimitiveCount > 0;
            if (!record.accelerationStructures)
            {
                return;
            }
            record.minAccelerationStructureScratchOffsetAlignment =
                properties.minAccelerationStructureScratchOffsetAlignment == 0
                    ? 256
                    : properties.minAccelerationStructureScratchOffsetAlignment;
            record.maxPerStageDescriptorAccelerationStructures = properties.maxPerStageDescriptorAccelerationStructures;
            record.maxDescriptorSetAccelerationStructures = properties.maxDescriptorSetAccelerationStructures;
            record.maxPerStageDescriptorUpdateAfterBindAccelerationStructures =
                properties.maxPerStageDescriptorUpdateAfterBindAccelerationStructures;
            record.maxDescriptorSetUpdateAfterBindAccelerationStructures =
                properties.maxDescriptorSetUpdateAfterBindAccelerationStructures;
            record.rayTracingLimits.maxGeometriesPerBottomLevel = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(properties.maxGeometryCount, std::numeric_limits<std::uint32_t>::max()));
            record.rayTracingLimits.maxInstancesPerTopLevel = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(properties.maxInstanceCount, std::numeric_limits<std::uint32_t>::max()));
            record.rayTracingLimits.maxPrimitivesPerBottomLevel = properties.maxPrimitiveCount;
        }
#endif

#ifdef VK_KHR_ray_tracing_pipeline
        void recordRayTracingPipelineSupport(VulkanPhysicalDeviceRecord& record,
                                             const PhysicalDeviceProbe& device) noexcept
        {
            if (!record.accelerationStructures ||
                !hasExtension(device.extensions, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME))
            {
                return;
            }
            const auto features = queryExtensionFeatures<VkPhysicalDeviceRayTracingPipelineFeaturesKHR>(
                device.instanceTable,
                device.physical,
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR);
            if (features.rayTracingPipeline != VK_TRUE)
            {
                return;
            }
            const auto properties = queryExtensionProperties<VkPhysicalDeviceRayTracingPipelinePropertiesKHR>(
                device.instanceTable,
                device.physical,
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR);
            record.rayTracingPipeline = properties.shaderGroupHandleSize != 0 &&
                                        properties.shaderGroupHandleAlignment != 0 &&
                                        properties.shaderGroupBaseAlignment != 0 && properties.maxRayRecursionDepth > 0;
            if (!record.rayTracingPipeline)
            {
                return;
            }
            record.indirectRayDispatch = features.rayTracingPipelineTraceRaysIndirect == VK_TRUE;
            record.rayTracingLimits.shaderGroupHandleSize = properties.shaderGroupHandleSize;
            record.rayTracingLimits.shaderGroupHandleAlignment = properties.shaderGroupHandleAlignment;
            record.rayTracingLimits.shaderGroupBaseAlignment = properties.shaderGroupBaseAlignment;
            record.rayTracingLimits.maxShaderGroupStride = properties.maxShaderGroupStride;
            record.rayTracingLimits.maxRecursionDepth = properties.maxRayRecursionDepth;
            record.rayTracingLimits.maxAttributeSize = properties.maxRayHitAttributeSize;
            record.rayTracingLimits.maxDispatchInvocations = properties.maxRayDispatchInvocationCount;
        }
#endif

        void recordRayTracingSupport(VulkanPhysicalDeviceRecord& record, const PhysicalDeviceProbe& device) noexcept
        {
#ifdef VK_KHR_acceleration_structure
            recordAccelerationStructureSupport(record, device);
#endif
#ifdef VK_KHR_deferred_host_operations
            record.deferredHostOperations =
                record.accelerationStructures &&
                hasExtension(device.extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
#endif
#ifdef VK_KHR_ray_tracing_pipeline
            recordRayTracingPipelineSupport(record, device);
#endif
#ifdef VK_KHR_ray_query
            if (record.accelerationStructures && hasExtension(device.extensions, VK_KHR_RAY_QUERY_EXTENSION_NAME))
            {
                const auto features = queryExtensionFeatures<VkPhysicalDeviceRayQueryFeaturesKHR>(
                    device.instanceTable, device.physical, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR);
                record.rayQuery = features.rayQuery == VK_TRUE;
            }
#endif
        }

        void recordSubgroupSupport(VulkanPhysicalDeviceRecord& record,
                                   const PhysicalDevicePropertyChain& properties) noexcept
        {
            const auto subgroupStages = properties.v11.subgroupSupportedStages;
            const auto subgroupOps = properties.v11.subgroupSupportedOperations;
            const bool computeSubgroups = (subgroupStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
            const bool portableSubgroupOps = (subgroupOps & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0 &&
                                             (subgroupOps & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
            const auto minSubgroup =
                properties.v13.minSubgroupSize != 0 ? properties.v13.minSubgroupSize : properties.v11.subgroupSize;
            const auto maxSubgroup =
                properties.v13.maxSubgroupSize != 0 ? properties.v13.maxSubgroupSize : properties.v11.subgroupSize;
            record.subgroupOperations =
                computeSubgroups && portableSubgroupOps && minSubgroup > 0 && maxSubgroup >= minSubgroup;
            if (record.subgroupOperations)
            {
                record.subgroupMinSize = minSubgroup;
                record.subgroupMaxSize = maxSubgroup;
                record.requiredSubgroupSizeStages = properties.v13.requiredSubgroupSizeStages;
            }
        }

#if defined(VK_KHR_external_memory_win32) && defined(VK_KHR_external_semaphore_win32)
        [[nodiscard]] bool canShareBuffersAsWin32Handles(const PhysicalDeviceProbe& device) noexcept
        {
            VkPhysicalDeviceExternalBufferInfo const bufferInfo{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
                .pNext = {},
                .flags = {},
                .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
            };
            VkExternalBufferProperties bufferProps{
                .sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES,
                .pNext = {},
                .externalMemoryProperties = {},
            };
            device.instanceTable.vkGetPhysicalDeviceExternalBufferProperties(
                device.physical, &bufferInfo, &bufferProps);
            const auto bufferFeatures = bufferProps.externalMemoryProperties.externalMemoryFeatures;
            return (bufferFeatures & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0 &&
                   (bufferFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
        }

        [[nodiscard]] bool canShareImagesAsWin32Handles(const PhysicalDeviceProbe& device) noexcept
        {
            VkPhysicalDeviceExternalImageFormatInfo externalImage{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
                .pNext = {},
                .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
            };
            VkPhysicalDeviceImageFormatInfo2 const imageInfo{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
                .pNext = &externalImage,
                .format = VK_FORMAT_R8G8B8A8_UNORM,
                .type = VK_IMAGE_TYPE_2D,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                .flags = {},
            };
            VkExternalImageFormatProperties imageExternal{
                .sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
                .pNext = {},
                .externalMemoryProperties = {},
            };
            VkImageFormatProperties2 imageProps{
                .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
                .pNext = &imageExternal,
                .imageFormatProperties = {},
            };
            if (device.instanceTable.vkGetPhysicalDeviceImageFormatProperties2(
                    device.physical, &imageInfo, &imageProps) != VK_SUCCESS)
            {
                return false;
            }
            const auto imageFeatures = imageExternal.externalMemoryProperties.externalMemoryFeatures;
            return (imageFeatures & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0 &&
                   (imageFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
        }

        [[nodiscard]] bool canShareTimelineSemaphoresAsWin32Handles(const PhysicalDeviceProbe& device) noexcept
        {
            VkSemaphoreTypeCreateInfo semaphoreType{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                .pNext = {},
                .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                .initialValue = {},
            };
            VkPhysicalDeviceExternalSemaphoreInfo const semaphoreInfo{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
                .pNext = &semaphoreType,
                .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT,
            };
            VkExternalSemaphoreProperties semaphoreProps{};
            semaphoreProps.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES;

            device.instanceTable.vkGetPhysicalDeviceExternalSemaphoreProperties(
                device.physical, &semaphoreInfo, &semaphoreProps);
            return (semaphoreProps.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) != 0 &&
                   (semaphoreProps.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT) != 0;
        }
#endif

        [[nodiscard]] bool supportsWin32SharedHandles(const PhysicalDeviceProbe& device) noexcept
        {
#if defined(VK_KHR_external_memory_win32) && defined(VK_KHR_external_semaphore_win32)
            const auto& table = device.instanceTable;
            const bool haveExternalMemoryWin32 =
                hasExtension(device.extensions, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
            const bool haveExternalSemaphoreWin32 =
                hasExtension(device.extensions, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
            const bool canQueryExternalMemory = haveExternalMemoryWin32 &&
                                                table.vkGetPhysicalDeviceExternalBufferProperties != nullptr &&
                                                table.vkGetPhysicalDeviceImageFormatProperties2 != nullptr;
            const bool bufferShare = canQueryExternalMemory && canShareBuffersAsWin32Handles(device);
            const bool imageShare = canQueryExternalMemory && canShareImagesAsWin32Handles(device);
            const bool timelineShare = haveExternalSemaphoreWin32 &&
                                       table.vkGetPhysicalDeviceExternalSemaphoreProperties != nullptr &&
                                       canShareTimelineSemaphoresAsWin32Handles(device);
            return haveExternalMemoryWin32 && haveExternalSemaphoreWin32 && bufferShare && imageShare && timelineShare;
#else
            static_cast<void>(device);
            return false;
#endif
        }

        void reportDeviceExtensions(VulkanPhysicalDeviceReport& report,
                                    const std::vector<VkExtensionProperties>& extensions) noexcept
        {
            report.khrSwapchain = hasExtension(extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
            report.khrPortabilitySubset = hasExtension(extensions, "VK_KHR_portability_subset");
            report.khrSwapchainMaintenance1 = hasExtension(extensions, "VK_KHR_swapchain_maintenance1");
            report.extSwapchainMaintenance1 = hasExtension(extensions, "VK_EXT_swapchain_maintenance1");
            report.khrPresentId = hasExtension(extensions, VK_KHR_PRESENT_ID_EXTENSION_NAME);
            report.khrPresentWait = hasExtension(extensions, VK_KHR_PRESENT_WAIT_EXTENSION_NAME);
            report.extDeviceFault = hasExtension(extensions, VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
        }

        struct ReportedExtensionFeatures final
        {
#ifdef VK_EXT_device_fault
            VkPhysicalDeviceFaultFeaturesEXT deviceFault{};
#endif
#ifdef VK_KHR_present_id
            VkPhysicalDevicePresentIdFeaturesKHR presentId{};
#endif
#ifdef VK_KHR_present_wait
            VkPhysicalDevicePresentWaitFeaturesKHR presentWait{};
#endif
#if defined(VK_KHR_swapchain_maintenance1) || defined(VK_EXT_swapchain_maintenance1)
            SwapchainMaintenance1Features swapchainMaintenance1{};
#endif
#ifdef VK_KHR_portability_subset
            VkPhysicalDevicePortabilitySubsetFeaturesKHR portabilitySubset{};
#endif
        };

        void queryReportedExtensionFeatures(const PhysicalDeviceProbe& device,
                                            const VulkanPhysicalDeviceReport& report,
                                            ReportedExtensionFeatures& features) noexcept
        {
            VkPhysicalDeviceFeatures2 query{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                .pNext = {},
                .features = {},
            };
#ifdef VK_EXT_device_fault
            if (report.extDeviceFault)
            {
                prependToChain(query, features.deviceFault, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT);
            }
#endif
#ifdef VK_KHR_present_id
            if (report.khrPresentId)
            {
                prependToChain(query, features.presentId, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR);
            }
#endif
#ifdef VK_KHR_present_wait
            if (report.khrPresentWait)
            {
                prependToChain(
                    query, features.presentWait, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR);
            }
#endif
#if defined(VK_KHR_swapchain_maintenance1) || defined(VK_EXT_swapchain_maintenance1)
            if (report.khrSwapchainMaintenance1 || report.extSwapchainMaintenance1)
            {
                prependToChain(query, features.swapchainMaintenance1, swapchainMaintenance1FeaturesType);
            }
#endif
#ifdef VK_KHR_portability_subset
            if (report.khrPortabilitySubset)
            {
                prependToChain(query,
                               features.portabilitySubset,
                               VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR);
            }
#endif
            device.instanceTable.vkGetPhysicalDeviceFeatures2(device.physical, &query);
        }

        void recordReportedExtensionFeatures(VulkanPhysicalDeviceRecord& record,
                                             const ReportedExtensionFeatures& features) noexcept
        {
#ifdef VK_EXT_device_fault
            record.deviceFault = features.deviceFault.deviceFault == VK_TRUE;
#endif
#ifdef VK_KHR_present_id
            record.presentId = record.report.khrPresentId && features.presentId.presentId == VK_TRUE;
            record.report.khrPresentId = record.presentId;
#endif
#ifdef VK_KHR_present_wait
            record.presentWait = record.report.khrPresentWait && features.presentWait.presentWait == VK_TRUE;
            record.report.khrPresentWait = record.presentWait;
#endif
#if defined(VK_KHR_swapchain_maintenance1) || defined(VK_EXT_swapchain_maintenance1)
            record.swapchainMaintenance1 = features.swapchainMaintenance1.swapchainMaintenance1 == VK_TRUE;
#endif
#ifdef VK_KHR_portability_subset
            if (record.report.khrPortabilitySubset)
            {
                record.imageViewFormatSwizzle = features.portabilitySubset.imageViewFormatSwizzle == VK_TRUE;
            }
#endif
        }

#ifdef VK_KHR_portability_subset
        void recordPortabilitySubsetLimits(VulkanPhysicalDeviceRecord& record,
                                           const PhysicalDeviceProbe& device) noexcept
        {
            if (!record.report.khrPortabilitySubset)
            {
                return;
            }
            const auto properties = queryExtensionProperties<VkPhysicalDevicePortabilitySubsetPropertiesKHR>(
                device.instanceTable,
                device.physical,
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_PROPERTIES_KHR);
            record.minVertexInputBindingStrideAlignment =
                (std::max)(properties.minVertexInputBindingStrideAlignment, 1U);
        }
#endif

        void reportAttachmentFormats(VulkanPhysicalDeviceReport& report, const PhysicalDeviceProbe& device) noexcept
        {
            const auto isColorTarget = [&](VkFormat format) noexcept {
                return formatHasOptimal(
                    device.instanceTable, device.physical, format, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT);
            };
            const auto isDepthStencilTarget = [&](VkFormat format) noexcept {
                return formatHasOptimal(
                    device.instanceTable, device.physical, format, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
            };
            report.formatRgba8Unorm = isColorTarget(VK_FORMAT_R8G8B8A8_UNORM);
            report.formatRgba8UnormSrgb = isColorTarget(VK_FORMAT_R8G8B8A8_SRGB);
            report.formatBgra8Unorm = isColorTarget(VK_FORMAT_B8G8R8A8_UNORM);
            report.formatBgra8UnormSrgb = isColorTarget(VK_FORMAT_B8G8R8A8_SRGB);
            report.formatD32Float = isDepthStencilTarget(VK_FORMAT_D32_SFLOAT);
            report.formatD24UnormS8Uint = isDepthStencilTarget(VK_FORMAT_D24_UNORM_S8_UINT);
            report.formatD32FloatS8Uint = isDepthStencilTarget(VK_FORMAT_D32_SFLOAT_S8_UINT);
            report.formatRgba16Float = isColorTarget(VK_FORMAT_R16G16B16A16_SFLOAT);
            report.formatR32Uint = isColorTarget(VK_FORMAT_R32_UINT);
        }

        void reportDeviceLimits(VulkanPhysicalDeviceReport& report, const VkPhysicalDeviceLimits& limits) noexcept
        {
            report.timestampComputeAndGraphics = limits.timestampComputeAndGraphics == VK_TRUE;
            report.maxImageDimension2D = limits.maxImageDimension2D;
            report.maxBoundDescriptorSets = limits.maxBoundDescriptorSets;
            report.maxPushConstantsSize = limits.maxPushConstantsSize;
            report.maxColorAttachments = limits.maxColorAttachments;
            report.maxComputeWorkgroupInvocations = limits.maxComputeWorkGroupInvocations;
        }

        [[nodiscard]] AdapterInfo describeAdapter(const VulkanPhysicalDeviceRecord& record)
        {
            std::uint64_t dedicated = 0;
            std::uint64_t shared = 0;
            for (std::uint32_t i = 0; i < record.memory.memoryHeapCount; ++i)
            {
                if ((record.memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0)
                {
                    dedicated += record.memory.memoryHeaps[i].size;
                }
                else
                {
                    shared += record.memory.memoryHeaps[i].size;
                }
            }

            return AdapterInfo{
                .id = adapterIdFromIdentity(record),
                .name = record.deviceName,
                .type = adapterTypeFrom(record.properties.deviceType),
                .vendorId = record.properties.vendorID,
                .deviceId = record.properties.deviceID,
                .dedicatedVideoMemory = dedicated,
                .sharedSystemMemory = shared,
                .backend = Backend::vulkan,
            };
        }

        [[nodiscard]] Result<VulkanPhysicalDeviceRecord> describePhysicalDevice(
            const VulkanInstanceTable& instanceTable, VkPhysicalDevice physical)
        {
            VulkanPhysicalDeviceRecord record{};
            record.physical = physical;
            instanceTable.vkGetPhysicalDeviceProperties(physical, &record.properties);
            instanceTable.vkGetPhysicalDeviceMemoryProperties(physical, &record.memory);

            PhysicalDevicePropertyChain properties{};
            queryPropertyChain(instanceTable, physical, record.properties.apiVersion, properties);
            recordDriverIdentity(record, properties.driver);
            const auto families = queueFamilyProperties(instanceTable, physical);
            recordVulkanQueueFamilies(record, families);
            std::ranges::copy(properties.id.deviceUUID, record.deviceUuid.begin());
            std::ranges::copy(properties.id.deviceLUID, record.deviceLuid.begin());
            record.deviceLuidValid = properties.id.deviceLUIDValid == VK_TRUE;
            if (!supportsApiVersion(record.properties.apiVersion, VK_API_VERSION_1_3))
            {
                record.adapter = describeAdapter(record);
                return record;
            }

            PhysicalDeviceFeatureChain features{};
            queryFeatureChain(instanceTable, physical, features);
            auto extensions = deviceExtensions(instanceTable, physical);
            if (!extensions)
            {
                return std::unexpected(std::move(extensions.error()));
            }
            const PhysicalDeviceProbe device{
                .instanceTable = instanceTable,
                .physical = physical,
                .extensions = *extensions,
            };

            record.depthResolveModes = properties.v12.supportedDepthResolveModes;
            record.stencilResolveModes = properties.v12.supportedStencilResolveModes;
            record.independentResolveNone = properties.v12.independentResolveNone == VK_TRUE;
            record.independentResolve = properties.v12.independentResolve == VK_TRUE;
            recordUpdateAfterBindLimits(record, properties.v12);
            recordCoreFeatures(record, features);
            record.maxMultiviewViewCount = record.multiview ? properties.v11.maxMultiviewViewCount : 0;
            recordOptionalExtensionSupport(record, device, features);
#ifdef VK_EXT_mesh_shader
            recordMeshShaderSupport(record, device);
#endif
#ifdef VK_KHR_fragment_shading_rate
            recordVariableRateShadingSupport(record, device);
#endif
            recordRayTracingSupport(record, device);
            recordSubgroupSupport(record, properties);

            const bool sharedFamily = supportsWin32SharedHandles(device);
            record.sharedBufferHandles = sharedFamily;
            record.sharedTextureHandles = sharedFamily;
            record.sharedTimelineHandles = sharedFamily;

            reportDeviceExtensions(record.report, *extensions);
            ReportedExtensionFeatures extensionFeatures{};
            queryReportedExtensionFeatures(device, record.report, extensionFeatures);
            recordReportedExtensionFeatures(record, extensionFeatures);
#ifdef VK_KHR_portability_subset
            recordPortabilitySubsetLimits(record, device);
#endif
            reportAttachmentFormats(record.report, device);
            reportDeviceLimits(record.report, record.properties.limits);
            record.adapter = describeAdapter(record);
            return record;
        }

    } // namespace

    Result<std::vector<VulkanPhysicalDeviceRecord>> discoverVulkanPhysicalDevices(
        VkInstance instance, const VulkanInstanceTable& instanceTable, const VulkanInstanceReport& instanceReport)
    {
        static_cast<void>(instanceReport);
        if (instance == VK_NULL_HANDLE || instanceTable.vkEnumeratePhysicalDevices == nullptr ||
            instanceTable.vkGetPhysicalDeviceProperties == nullptr ||
            instanceTable.vkGetPhysicalDeviceMemoryProperties == nullptr ||
            instanceTable.vkGetPhysicalDeviceProperties2 == nullptr ||
            instanceTable.vkGetPhysicalDeviceFeatures2 == nullptr ||
            instanceTable.vkEnumerateDeviceExtensionProperties == nullptr)
        {
            return std::unexpected(selectionError(VulkanSelectionFailure::loaderUnavailable,
                                                  "Vulkan instance dispatch table is incomplete."));
        }
        auto devices = enumeratePhysicalDevices(instance, instanceTable);
        if (!devices)
        {
            return std::unexpected(std::move(devices.error()));
        }

        std::vector<VulkanPhysicalDeviceRecord> records;
        records.reserve(devices->size());
        for (auto* const physical : *devices)
        {
            auto record = describePhysicalDevice(instanceTable, physical);
            if (!record)
            {
                return std::unexpected(std::move(record.error()));
            }
            records.push_back(std::move(*record));
            pointReportAtOwnedNames(records.back());
        }
        return records;
    }

    const VulkanPhysicalDeviceRecord* selectVulkanPhysicalDevice(const std::vector<VulkanPhysicalDeviceRecord>& devices,
                                                                 const VulkanInstanceReport& instanceReport,
                                                                 const DeviceDesc& desc)
    {
        if (desc.adapterId.has_value())
        {
            return firstNativeDeviceWithId(devices, *desc.adapterId);
        }

        const VulkanPhysicalDeviceRecord* best = nullptr;
        for (const auto& device : devices)
        {
            if (device.report.translationLayer)
            {
                continue;
            }
            const auto decision = evaluateNativeVulkanProfile(instanceReport, device.report);
            if (!decision.accepted)
            {
                continue;
            }
            if (best == nullptr)
            {
                best = &device;
                continue;
            }
            if (desc.adapterPreference == AdapterPreference::minimumPower)
            {
                if (device.adapter.type == AdapterType::integrated && best->adapter.type != AdapterType::integrated)
                {
                    best = &device;
                }
                continue;
            }
            if (device.adapter.type == AdapterType::discrete && best->adapter.type != AdapterType::discrete)
            {
                best = &device;
                continue;
            }
            if (device.adapter.type == best->adapter.type &&
                device.adapter.dedicatedVideoMemory > best->adapter.dedicatedVideoMemory)
            {
                best = &device;
            }
        }
        return best;
    }

    void reportVulkanAdapterIdCollisions(const std::vector<VulkanPhysicalDeviceRecord>& devices,
                                         VulkanDiagnosticSink& sink)
    {
        for (const auto& device : devices)
        {
            if (device.report.translationLayer)
            {
                continue;
            }
            const auto* selected = firstNativeDeviceWithId(devices, device.adapter.id);
            if (selected == &device || sameDeviceIdentity(*selected, device))
            {
                continue;
            }
            std::ostringstream message;
            message << "Vulkan physical device \"" << device.deviceName << "\" shares AdapterId 0x" << std::hex
                    << device.adapter.id.value << " with \"" << selected->deviceName
                    << "\" although their device UUIDs or LUIDs differ. DeviceDesc::adapterId selects only \""
                    << selected->deviceName << "\".";
            sink.publish(DiagnosticSeverity::warning, 0, message.str());
        }
    }

    namespace
    {

        [[nodiscard]] Status validateLogicalDevicePrerequisites(const VulkanInstanceTable& instanceTable,
                                                                const VulkanPhysicalDeviceRecord& record,
                                                                const VulkanInstanceReport& instanceReport)
        {
            if (instanceTable.vkCreateDevice == nullptr || instanceTable.vkGetDeviceProcAddr == nullptr ||
                instanceTable.vkGetPhysicalDeviceFeatures2 == nullptr)
            {
                return std::unexpected(
                    makeVulkanError(ErrorCode::deviceCreationFailed, VK_ERROR_INITIALIZATION_FAILED, "vkCreateDevice"));
            }
            const auto intent = VulkanCreateIntent{
                .policy = VulkanAdapterSelectionPolicy::hardwarePreferred,
                .warpPreference = false,
            };
            const auto decision = evaluateNativeVulkanProfile(instanceReport, record.report, intent);
            if (!decision.accepted)
            {
                return std::unexpected(profileError(decision));
            }
            if (record.queues.graphics == VK_QUEUE_FAMILY_IGNORED)
            {
                return std::unexpected(
                    selectionError(VulkanSelectionFailure::requiredFeatureMissing,
                                   "Vulkan adapter has no graphics queue family. Selection did not fall back "
                                   "to D3D12."));
            }
            return {};
        }

        constexpr std::array queuePriorities{1.0F, 1.0F, 1.0F};

        [[nodiscard]] std::vector<VkDeviceQueueCreateInfo> uniqueQueueCreateInfos(const VulkanQueueFamilies& families)
        {
            const std::array assignments{
                std::pair{families.graphics, families.graphicsIndex},
                std::pair{families.compute, families.computeIndex},
                std::pair{families.transfer, families.transferIndex},
            };
            std::vector<VkDeviceQueueCreateInfo> queueInfos;
            for (const auto& [family, index] : assignments)
            {
                if (family == VK_QUEUE_FAMILY_IGNORED)
                {
                    continue;
                }
                const auto existing = std::ranges::find(queueInfos, family, &VkDeviceQueueCreateInfo::queueFamilyIndex);
                if (existing != queueInfos.end())
                {
                    existing->queueCount = std::max(existing->queueCount, index + 1U);
                    continue;
                }
                queueInfos.push_back(VkDeviceQueueCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    .pNext = {},
                    .flags = {},
                    .queueFamilyIndex = family,
                    .queueCount = index + 1U,
                    .pQueuePriorities = queuePriorities.data(),
                });
            }
            std::ranges::sort(queueInfos, {}, &VkDeviceQueueCreateInfo::queueFamilyIndex);
            return queueInfos;
        }

        [[nodiscard]] const char* selectSwapchainMaintenanceExtension(
            const VulkanPhysicalDeviceRecord& record, const VulkanInstanceReport& instanceReport) noexcept
        {
            const bool instanceSurfaceMaintenance1 =
                instanceReport.khrSurfaceMaintenance1 || instanceReport.extSurfaceMaintenance1;
            const bool enableKhrMaintenance1 = instanceReport.khrSurfaceMaintenance1 &&
                                               record.report.khrSwapchainMaintenance1 && record.swapchainMaintenance1;
            const bool enableExtMaintenance1 = instanceSurfaceMaintenance1 && record.report.extSwapchainMaintenance1 &&
                                               record.swapchainMaintenance1 && !enableKhrMaintenance1;
#if defined(VK_KHR_swapchain_maintenance1) || defined(VK_EXT_swapchain_maintenance1)
            if (enableKhrMaintenance1)
            {
                return "VK_KHR_swapchain_maintenance1";
            }
            if (enableExtMaintenance1)
            {
                return "VK_EXT_swapchain_maintenance1";
            }
#else
            static_cast<void>(enableKhrMaintenance1);
            static_cast<void>(enableExtMaintenance1);
#endif
            return nullptr;
        }

        [[nodiscard]] bool canEnablePresentWait(const VulkanPhysicalDeviceRecord& record) noexcept
        {
#if defined(VK_KHR_present_id) && defined(VK_KHR_present_wait)
            return record.report.khrSwapchain && record.report.khrPresentId && record.presentId &&
                   record.report.khrPresentWait && record.presentWait;
#else
            static_cast<void>(record);
            return false;
#endif
        }

        struct PresentationAndFaultSelection final
        {
            const char* swapchainMaintenanceExtension = nullptr;
            bool presentWait = false;
            bool deviceFault = false;
        };

        [[nodiscard]] PresentationAndFaultSelection selectPresentationAndFaultSupport(
            const VulkanPhysicalDeviceRecord& record,
            const VulkanInstanceReport& instanceReport,
            bool enableDeviceFault) noexcept
        {
            return PresentationAndFaultSelection{
                .swapchainMaintenanceExtension = selectSwapchainMaintenanceExtension(record, instanceReport),
                .presentWait = canEnablePresentWait(record),
                .deviceFault = enableDeviceFault && record.report.extDeviceFault && record.deviceFault,
            };
        }

        [[nodiscard]] VulkanPresentCompletionStrategy presentCompletionFor(
            const PresentationAndFaultSelection& selection) noexcept
        {
            if (selection.swapchainMaintenanceExtension != nullptr)
            {
                return VulkanPresentCompletionStrategy::swapchainMaintenancePresentFence;
            }
            if (selection.presentWait)
            {
                return VulkanPresentCompletionStrategy::presentWait;
            }
            return VulkanPresentCompletionStrategy::imageLifetimeAssociation;
        }

        [[nodiscard]] std::vector<const char*> selectDeviceExtensions(const VulkanPhysicalDeviceRecord& record,
                                                                      const PresentationAndFaultSelection& selection)
        {
            std::vector<const char*> names;
            appendIf(names, record.report.khrSwapchain, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
            appendIf(names, record.report.khrPortabilitySubset, "VK_KHR_portability_subset");
            if (selection.swapchainMaintenanceExtension != nullptr)
            {
                names.push_back(selection.swapchainMaintenanceExtension);
            }
            appendIf(names, selection.deviceFault, VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
#if defined(VK_KHR_present_id) && defined(VK_KHR_present_wait)
            appendIf(names, selection.presentWait, VK_KHR_PRESENT_ID_EXTENSION_NAME);
            appendIf(names, selection.presentWait, VK_KHR_PRESENT_WAIT_EXTENSION_NAME);
#endif
#ifdef VK_EXT_shader_image_atomic_int64
            appendIf(names, record.shaderImageInt64Atomics, VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME);
#endif
            appendIf(names, record.conservativeRasterization, VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME);
#ifdef VK_EXT_fragment_shader_interlock
            appendIf(names, record.fragmentShaderPixelInterlock, VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME);
#endif
#ifdef VK_KHR_external_memory_win32
            appendIf(names, record.sharedBufferHandles, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#endif
#ifdef VK_KHR_external_semaphore_win32
            appendIf(names, record.sharedTimelineHandles, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#endif
#ifdef VK_EXT_calibrated_timestamps
            appendIf(names, record.calibratedTimestamps, VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
#endif
            appendIf(names, record.memoryBudget, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
#ifdef VK_EXT_memory_priority
            appendIf(names, record.memoryPriority, VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME);
#endif
#ifdef VK_EXT_mesh_shader
            appendIf(names, record.meshShaders, VK_EXT_MESH_SHADER_EXTENSION_NAME);
#endif
#ifdef VK_EXT_conditional_rendering
            appendIf(names, record.conditionalRendering, VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME);
#endif
#ifdef VK_KHR_fragment_shading_rate
            appendIf(names, record.variableRateShading, VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME);
#endif
#ifdef VK_KHR_deferred_host_operations
            appendIf(names, record.deferredHostOperations, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
#endif
#ifdef VK_KHR_acceleration_structure
            appendIf(names, record.accelerationStructures, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
#endif
#ifdef VK_KHR_ray_tracing_pipeline
            appendIf(names, record.rayTracingPipeline, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
#endif
#ifdef VK_KHR_ray_query
            appendIf(names, record.rayQuery, VK_KHR_RAY_QUERY_EXTENSION_NAME);
#endif
#ifdef VK_KHR_fragment_shader_barycentric
            appendIf(names, record.fragmentBarycentric, VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
#endif
#ifdef VK_KHR_maintenance5
            appendIf(names, record.maintenance5, VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
#endif
            return names;
        }

        struct LogicalDeviceFeatureChain final
        {
            VkPhysicalDeviceFeatures2 features2{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                .pNext = {},
                .features = {},
            };
            VkPhysicalDeviceVulkan11Features v11{};
            VkPhysicalDeviceVulkan12Features v12{};
            VkPhysicalDeviceVulkan13Features v13{};
#ifdef VK_KHR_maintenance5
            VkPhysicalDeviceMaintenance5FeaturesKHR maintenance5{};
#endif
#if defined(VK_KHR_swapchain_maintenance1) || defined(VK_EXT_swapchain_maintenance1)
            SwapchainMaintenance1Features swapchainMaintenance1{};
#endif
#if defined(VK_KHR_present_id) && defined(VK_KHR_present_wait)
            VkPhysicalDevicePresentIdFeaturesKHR presentId{};
            VkPhysicalDevicePresentWaitFeaturesKHR presentWait{};
#endif
#ifdef VK_EXT_device_fault
            VkPhysicalDeviceFaultFeaturesEXT deviceFault{};
#endif
#ifdef VK_EXT_shader_image_atomic_int64
            VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT imageInt64Atomics{};
#endif
#ifdef VK_KHR_fragment_shader_barycentric
            VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR fragmentBarycentric{};
#endif
#ifdef VK_EXT_fragment_shader_interlock
            VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT fragmentInterlock{};
#endif
#ifdef VK_EXT_memory_priority
            VkPhysicalDeviceMemoryPriorityFeaturesEXT memoryPriority{};
#endif
#ifdef VK_EXT_mesh_shader
            VkPhysicalDeviceMeshShaderFeaturesEXT meshShader{};
#endif
#ifdef VK_EXT_conditional_rendering
            VkPhysicalDeviceConditionalRenderingFeaturesEXT conditionalRendering{};
#endif
#ifdef VK_KHR_fragment_shading_rate
            VkPhysicalDeviceFragmentShadingRateFeaturesKHR shadingRate{};
#endif
#ifdef VK_KHR_acceleration_structure
            VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructure{};
#endif
#ifdef VK_KHR_ray_tracing_pipeline
            VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipeline{};
#endif
#ifdef VK_KHR_ray_query
            VkPhysicalDeviceRayQueryFeaturesKHR rayQuery{};
#endif
#ifdef VK_KHR_portability_subset
            VkPhysicalDevicePortabilitySubsetFeaturesKHR portabilitySubset{};
#endif
        };

        void chainPresentationAndFaultFeatures(LogicalDeviceFeatureChain& chain,
                                               const PresentationAndFaultSelection& selection) noexcept
        {
#if defined(VK_KHR_swapchain_maintenance1) || defined(VK_EXT_swapchain_maintenance1)
            if (selection.swapchainMaintenanceExtension != nullptr)
            {
                chain.swapchainMaintenance1.swapchainMaintenance1 = VK_TRUE;
                prependToChain(chain.features2, chain.swapchainMaintenance1, swapchainMaintenance1FeaturesType);
            }
#endif
#if defined(VK_KHR_present_id) && defined(VK_KHR_present_wait)
            if (selection.presentWait)
            {
                chain.presentId.presentId = VK_TRUE;
                prependToChain(
                    chain.features2, chain.presentId, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR);
                chain.presentWait.presentWait = VK_TRUE;
                prependToChain(
                    chain.features2, chain.presentWait, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR);
            }
#endif
#ifdef VK_EXT_device_fault
            if (selection.deviceFault)
            {
                chain.deviceFault.deviceFault = VK_TRUE;
                prependToChain(
                    chain.features2, chain.deviceFault, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT);
            }
#endif
        }

        void chainShaderAndMemoryFeatures(LogicalDeviceFeatureChain& chain,
                                          const VulkanPhysicalDeviceRecord& record) noexcept
        {
#ifdef VK_EXT_shader_image_atomic_int64
            if (record.shaderImageInt64Atomics)
            {
                chain.imageInt64Atomics.shaderImageInt64Atomics = VK_TRUE;
                prependToChain(chain.features2,
                               chain.imageInt64Atomics,
                               VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT);
            }
#endif
#ifdef VK_KHR_fragment_shader_barycentric
            if (record.fragmentBarycentric)
            {
                chain.fragmentBarycentric.fragmentShaderBarycentric = VK_TRUE;
                prependToChain(chain.features2,
                               chain.fragmentBarycentric,
                               VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR);
            }
#endif
#ifdef VK_EXT_fragment_shader_interlock
            if (record.fragmentShaderPixelInterlock)
            {
                chain.fragmentInterlock.fragmentShaderPixelInterlock = VK_TRUE;
                prependToChain(chain.features2,
                               chain.fragmentInterlock,
                               VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT);
            }
#endif
#ifdef VK_EXT_memory_priority
            if (record.memoryPriority)
            {
                chain.memoryPriority.memoryPriority = VK_TRUE;
                prependToChain(chain.features2,
                               chain.memoryPriority,
                               VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT);
            }
#endif
#ifdef VK_EXT_mesh_shader
            if (record.meshShaders)
            {
                chain.meshShader.taskShader = record.taskShaders ? VK_TRUE : VK_FALSE;
                chain.meshShader.meshShader = VK_TRUE;
                chain.meshShader.meshShaderQueries = record.meshShaderQueries ? VK_TRUE : VK_FALSE;
                prependToChain(
                    chain.features2, chain.meshShader, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT);
            }
#endif
#ifdef VK_EXT_conditional_rendering
            if (record.conditionalRendering)
            {
                chain.conditionalRendering.conditionalRendering = VK_TRUE;
                prependToChain(chain.features2,
                               chain.conditionalRendering,
                               VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT);
            }
#endif
#ifdef VK_KHR_fragment_shading_rate
            if (record.variableRateShading)
            {
                chain.shadingRate.pipelineFragmentShadingRate = VK_TRUE;
                chain.shadingRate.primitiveFragmentShadingRate = record.variableRateShadingTier2 ? VK_TRUE : VK_FALSE;
                chain.shadingRate.attachmentFragmentShadingRate = record.variableRateShadingTier2 ? VK_TRUE : VK_FALSE;
                prependToChain(chain.features2,
                               chain.shadingRate,
                               VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR);
            }
#endif
        }

        void chainRayTracingFeatures(LogicalDeviceFeatureChain& chain,
                                     const VulkanPhysicalDeviceRecord& record) noexcept
        {
#ifdef VK_KHR_acceleration_structure
            if (record.accelerationStructures)
            {
                chain.accelerationStructure.accelerationStructure = VK_TRUE;
                prependToChain(chain.features2,
                               chain.accelerationStructure,
                               VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR);
            }
#endif
#ifdef VK_KHR_ray_tracing_pipeline
            if (record.rayTracingPipeline)
            {
                chain.rayTracingPipeline.rayTracingPipeline = VK_TRUE;
                chain.rayTracingPipeline.rayTracingPipelineTraceRaysIndirect =
                    record.indirectRayDispatch ? VK_TRUE : VK_FALSE;
                prependToChain(chain.features2,
                               chain.rayTracingPipeline,
                               VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR);
            }
#endif
#ifdef VK_KHR_ray_query
            if (record.rayQuery)
            {
                chain.rayQuery.rayQuery = VK_TRUE;
                prependToChain(
                    chain.features2, chain.rayQuery, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR);
            }
#endif
        }

#ifdef VK_KHR_portability_subset
        void chainPortabilitySubsetFeatures(LogicalDeviceFeatureChain& chain,
                                            const VulkanInstanceTable& instanceTable,
                                            const VulkanPhysicalDeviceRecord& record) noexcept
        {
            if (!record.report.khrPortabilitySubset)
            {
                return;
            }
            chain.portabilitySubset = queryExtensionFeatures<VkPhysicalDevicePortabilitySubsetFeaturesKHR>(
                instanceTable, record.physical, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR);
            prependToChain(chain.features2,
                           chain.portabilitySubset,
                           VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR);
        }
#endif

        struct AvailableCoreFeatures final
        {
            VkPhysicalDeviceFeatures base{};
            VkPhysicalDeviceVulkan11Features v11{};
            VkPhysicalDeviceVulkan12Features v12{};
            VkPhysicalDeviceVulkan13Features v13{};
        };

        void queryAvailableCoreFeatures(const VulkanInstanceTable& instanceTable,
                                        VkPhysicalDevice physical,
                                        AvailableCoreFeatures& available) noexcept
        {
            VkPhysicalDeviceFeatures2 query{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                .pNext = {},
                .features = {},
            };
            prependToChain(query, available.v13, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
            prependToChain(query, available.v12, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
            prependToChain(query, available.v11, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
            instanceTable.vkGetPhysicalDeviceFeatures2(physical, &query);
            available.base = query.features;
        }

        [[nodiscard]] VkPhysicalDeviceFeatures selectBaseFeatures(const VkPhysicalDeviceFeatures& available,
                                                                  const VulkanPhysicalDeviceRecord& record) noexcept
        {
            return VkPhysicalDeviceFeatures{
                .robustBufferAccess = {},
                .fullDrawIndexUint32 = {},
                .imageCubeArray = available.imageCubeArray,
                .independentBlend = VK_TRUE,
                .geometryShader = available.geometryShader,
                .tessellationShader = available.tessellationShader,
                .sampleRateShading = available.sampleRateShading,
                .dualSrcBlend = available.dualSrcBlend,
                .logicOp = available.logicOp,
                .multiDrawIndirect = available.multiDrawIndirect,
                .drawIndirectFirstInstance = available.drawIndirectFirstInstance,
                .depthClamp = available.depthClamp,
                .depthBiasClamp = available.depthBiasClamp,
                .fillModeNonSolid = available.fillModeNonSolid,
                .depthBounds = available.depthBounds,
                .wideLines = {},
                .largePoints = {},
                .alphaToOne = {},
                .multiViewport = available.multiViewport,
                .samplerAnisotropy = VK_TRUE,
                .textureCompressionETC2 = {},
                .textureCompressionASTC_LDR = {},
                .textureCompressionBC = available.textureCompressionBC,
                .occlusionQueryPrecise = available.occlusionQueryPrecise,
                .pipelineStatisticsQuery = available.pipelineStatisticsQuery,
                .vertexPipelineStoresAndAtomics = available.vertexPipelineStoresAndAtomics,
                .fragmentStoresAndAtomics = record.fragmentStoresAndAtomics ? VK_TRUE : VK_FALSE,
                .shaderTessellationAndGeometryPointSize = {},
                .shaderImageGatherExtended = available.shaderImageGatherExtended,
                .shaderStorageImageExtendedFormats = {},
                .shaderStorageImageMultisample = {},
                .shaderStorageImageReadWithoutFormat = {},
                .shaderStorageImageWriteWithoutFormat = {},
                .shaderUniformBufferArrayDynamicIndexing = {},
                .shaderSampledImageArrayDynamicIndexing = VK_TRUE,
                .shaderStorageBufferArrayDynamicIndexing = {},
                .shaderStorageImageArrayDynamicIndexing = {},
                .shaderClipDistance = available.shaderClipDistance,
                .shaderCullDistance = available.shaderCullDistance,
                .shaderFloat64 = available.shaderFloat64,
                .shaderInt64 = available.shaderInt64,
                .shaderInt16 = available.shaderInt16,
                .shaderResourceResidency = {},
                .shaderResourceMinLod = available.shaderResourceMinLod,
                .sparseBinding = {},
                .sparseResidencyBuffer = {},
                .sparseResidencyImage2D = {},
                .sparseResidencyImage3D = {},
                .sparseResidency2Samples = {},
                .sparseResidency4Samples = {},
                .sparseResidency8Samples = {},
                .sparseResidency16Samples = {},
                .sparseResidencyAliased = {},
                .variableMultisampleRate = {},
                .inheritedQueries = {},
            };
        }

        void chainCoreFeatures(LogicalDeviceFeatureChain& chain,
                               const AvailableCoreFeatures& available,
                               const VulkanPhysicalDeviceRecord& record) noexcept
        {
            chain.v13.synchronization2 = VK_TRUE;
            chain.v13.dynamicRendering = VK_TRUE;
            chain.v13.maintenance4 = available.v13.maintenance4 == VK_TRUE ? VK_TRUE : VK_FALSE;
            chain.v13.subgroupSizeControl = record.subgroupSizeControl ? VK_TRUE : VK_FALSE;
            chain.v13.computeFullSubgroups = record.subgroupSizeControl ? VK_TRUE : VK_FALSE;
            prependToChain(chain.features2, chain.v13, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
#ifdef VK_KHR_maintenance5
            if (record.maintenance5)
            {
                chain.maintenance5.maintenance5 = VK_TRUE;
                prependToChain(
                    chain.features2, chain.maintenance5, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR);
            }
#endif

            chain.v12.timelineSemaphore = VK_TRUE;
            chain.v12.separateDepthStencilLayouts = available.v12.separateDepthStencilLayouts;
            chain.v12.runtimeDescriptorArray = available.v12.runtimeDescriptorArray;
            chain.v12.descriptorBindingPartiallyBound = available.v12.descriptorBindingPartiallyBound;
            chain.v12.descriptorBindingVariableDescriptorCount = available.v12.descriptorBindingVariableDescriptorCount;
            chain.v12.descriptorBindingUpdateUnusedWhilePending =
                available.v12.descriptorBindingUpdateUnusedWhilePending;
            chain.v12.descriptorBindingSampledImageUpdateAfterBind =
                available.v12.descriptorBindingSampledImageUpdateAfterBind;
            chain.v12.descriptorBindingStorageImageUpdateAfterBind =
                available.v12.descriptorBindingStorageImageUpdateAfterBind;
            chain.v12.descriptorBindingStorageBufferUpdateAfterBind =
                available.v12.descriptorBindingStorageBufferUpdateAfterBind;
            chain.v12.shaderUniformBufferArrayNonUniformIndexing =
                available.v12.shaderUniformBufferArrayNonUniformIndexing;
            chain.v12.shaderSampledImageArrayNonUniformIndexing =
                available.v12.shaderSampledImageArrayNonUniformIndexing;
            chain.v12.shaderStorageBufferArrayNonUniformIndexing =
                available.v12.shaderStorageBufferArrayNonUniformIndexing;
            chain.v12.shaderStorageImageArrayNonUniformIndexing =
                available.v12.shaderStorageImageArrayNonUniformIndexing;
            chain.v12.descriptorIndexing = available.v12.descriptorIndexing;
            chain.v12.shaderFloat16 = available.v12.shaderFloat16;
            chain.v12.shaderSharedInt64Atomics = available.v12.shaderSharedInt64Atomics;
            chain.v12.shaderBufferInt64Atomics = available.v12.shaderBufferInt64Atomics;
            chain.v12.shaderOutputLayer = available.v12.shaderOutputLayer;
            chain.v12.shaderOutputViewportIndex = available.v12.shaderOutputViewportIndex;
            chain.v12.samplerFilterMinmax = available.v12.samplerFilterMinmax;
            chain.v12.hostQueryReset = VK_TRUE;
            chain.v12.drawIndirectCount = available.v12.drawIndirectCount;
            chain.v12.bufferDeviceAddress = (record.accelerationStructures || record.bufferDeviceAddress) &&
                                                    available.v12.bufferDeviceAddress == VK_TRUE
                                                ? VK_TRUE
                                                : VK_FALSE;
            prependToChain(chain.features2, chain.v12, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);

            chain.v11.multiview = available.v11.multiview;
            chain.v11.shaderDrawParameters = available.v11.shaderDrawParameters;
            chain.v11.storageBuffer16BitAccess = available.v11.storageBuffer16BitAccess;
            chain.v11.uniformAndStorageBuffer16BitAccess = available.v11.uniformAndStorageBuffer16BitAccess;
            chain.v11.storagePushConstant16 = available.v11.storagePushConstant16;
            chain.v11.storageInputOutput16 = available.v11.storageInputOutput16;
            prependToChain(chain.features2, chain.v11, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);

            chain.features2.features = selectBaseFeatures(available.base, record);
        }

        [[nodiscard]] VulkanShaderFeatureSupport enabledShaderFeatures(const LogicalDeviceFeatureChain& chain) noexcept
        {
            const auto& enabled = chain.features2.features;
            return VulkanShaderFeatureSupport{
                .sampleRateShading = enabled.sampleRateShading == VK_TRUE,
                .vertexPipelineStoresAndAtomics = enabled.vertexPipelineStoresAndAtomics == VK_TRUE,
                .shaderImageGatherExtended = enabled.shaderImageGatherExtended == VK_TRUE,
                .shaderResourceMinLod = enabled.shaderResourceMinLod == VK_TRUE,
                .shaderFloat64 = enabled.shaderFloat64 == VK_TRUE,
                .storageBuffer16BitAccess = chain.v11.storageBuffer16BitAccess == VK_TRUE,
                .uniformAndStorageBuffer16BitAccess = chain.v11.uniformAndStorageBuffer16BitAccess == VK_TRUE,
                .storagePushConstant16 = chain.v11.storagePushConstant16 == VK_TRUE,
                .storageInputOutput16 = chain.v11.storageInputOutput16 == VK_TRUE,
                .shaderDrawParameters = chain.v11.shaderDrawParameters == VK_TRUE,
            };
        }

        [[nodiscard]] VulkanMultiviewStageSupport chainMultiviewStageFeatures(
            LogicalDeviceFeatureChain& chain, const VulkanPhysicalDeviceRecord& record) noexcept
        {
            const bool multiview = chain.v11.multiview == VK_TRUE;
            const VulkanMultiviewStageSupport enabled{
                .geometry =
                    multiview && chain.features2.features.geometryShader == VK_TRUE && record.multiviewGeometryShader,
                .tessellation = multiview && chain.features2.features.tessellationShader == VK_TRUE &&
                                record.multiviewTessellationShader,
#ifdef VK_EXT_mesh_shader
                .mesh = multiview && chain.meshShader.meshShader == VK_TRUE && record.multiviewMeshShader,
#else
                .mesh = false,
#endif
            };
            chain.v11.multiviewGeometryShader = enabled.geometry ? VK_TRUE : VK_FALSE;
            chain.v11.multiviewTessellationShader = enabled.tessellation ? VK_TRUE : VK_FALSE;
#ifdef VK_EXT_mesh_shader
            chain.meshShader.multiviewMeshShader = enabled.mesh ? VK_TRUE : VK_FALSE;
#endif
            return enabled;
        }

        [[nodiscard]] bool fetchDeviceQueues(const VulkanInstanceTable& instanceTable,
                                             VkDevice device,
                                             const VulkanQueueFamilies& families,
                                             VulkanQueueHandles& queues) noexcept
        {
            const auto getDeviceQueue = dispatch_detail::resolveVulkanProc<PFN_vkGetDeviceQueue>(
                device, instanceTable.vkGetDeviceProcAddr, "vkGetDeviceQueue");
            if (getDeviceQueue == nullptr)
            {
                return false;
            }
            queues.families = families;
            getDeviceQueue(device, families.graphics, families.graphicsIndex, &queues.graphics);
            getDeviceQueue(device, families.compute, families.computeIndex, &queues.compute);
            getDeviceQueue(device, families.transfer, families.transferIndex, &queues.copy);
            queues.present = queues.graphics;
            return true;
        }

        void destroyDeviceWithoutDispatchTable(const VulkanInstanceTable& instanceTable, VkDevice device) noexcept
        {
            if (const auto destroyDevice = dispatch_detail::resolveVulkanProc<PFN_vkDestroyDevice>(
                    device, instanceTable.vkGetDeviceProcAddr, "vkDestroyDevice");
                destroyDevice != nullptr)
            {
                destroyDevice(device, nullptr);
            }
        }

    } // namespace

    Result<VkDevice> createVulkanLogicalDevice(const VulkanInstanceTable& instanceTable,
                                               const VulkanPhysicalDeviceRecord& record,
                                               const VulkanInstanceReport& instanceReport,
                                               VulkanQueueHandles& queues,
                                               bool enableDeviceFault)
    {
        if (auto valid = validateLogicalDevicePrerequisites(instanceTable, record, instanceReport); !valid)
        {
            return std::unexpected(std::move(valid.error()));
        }

        const auto selection = selectPresentationAndFaultSupport(record, instanceReport, enableDeviceFault);
        queues.presentCompletion = presentCompletionFor(selection);
        queues.deviceFaultEnabled = selection.deviceFault;
        queues.presentWaitEnabled = selection.presentWait;
        queues.retireOnSubmissionFence = record.moltenVk;
        const auto queueInfos = uniqueQueueCreateInfos(record.queues);
        const auto enabledExtensions = selectDeviceExtensions(record, selection);

        LogicalDeviceFeatureChain chain{};
        chainPresentationAndFaultFeatures(chain, selection);
        chainShaderAndMemoryFeatures(chain, record);
        chainRayTracingFeatures(chain, record);
#ifdef VK_KHR_portability_subset
        chainPortabilitySubsetFeatures(chain, instanceTable, record);
#endif
        AvailableCoreFeatures available{};
        queryAvailableCoreFeatures(instanceTable, record.physical, available);
        queues.maintenance4Enabled = available.v13.maintenance4 == VK_TRUE;
        queues.maintenance5Enabled = record.maintenance5;
        if (available.v12.hostQueryReset != VK_TRUE)
        {
            return std::unexpected(
                makeVulkanError(ErrorCode::deviceCreationFailed, VK_ERROR_FEATURE_NOT_PRESENT, "hostQueryReset"));
        }
        chainCoreFeatures(chain, available, record);
        queues.shaderFeaturesEnabled = enabledShaderFeatures(chain);
        queues.multiviewStagesEnabled = chainMultiviewStageFeatures(chain, record);

        const auto createInfo = VkDeviceCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = &chain.features2,
            .flags = {},
            .queueCreateInfoCount = static_cast<std::uint32_t>(queueInfos.size()),
            .pQueueCreateInfos = queueInfos.data(),
            .enabledLayerCount = {},
            .ppEnabledLayerNames = {},
            .enabledExtensionCount = static_cast<std::uint32_t>(enabledExtensions.size()),
            .ppEnabledExtensionNames = enabledExtensions.data(),
            .pEnabledFeatures = {},
        };

        VkDevice device = VK_NULL_HANDLE;
        const auto created = instanceTable.vkCreateDevice(record.physical, &createInfo, nullptr, &device);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(makeVulkanError(ErrorCode::deviceCreationFailed, created, "vkCreateDevice"));
        }
        if (!fetchDeviceQueues(instanceTable, device, record.queues, queues))
        {
            destroyDeviceWithoutDispatchTable(instanceTable, device);
            return std::unexpected(
                makeVulkanError(ErrorCode::deviceCreationFailed, VK_ERROR_INITIALIZATION_FAILED, "vkGetDeviceQueue"));
        }
        return device;
    }

    namespace
    {

        void publishResourceLimits(DeviceCapabilities& capabilities, const VulkanPhysicalDeviceRecord& record) noexcept
        {
            const auto& limits = record.properties.limits;
            capabilities.limits.maxTextureDimension1D = limits.maxImageDimension1D;
            capabilities.limits.maxTextureDimension2D = limits.maxImageDimension2D;
            capabilities.limits.maxTextureDimension3D = limits.maxImageDimension3D;
            capabilities.limits.maxTextureArrayLayers = limits.maxImageArrayLayers;
            capabilities.limits.maxFramebufferLayers = 1;
            if (record.shaderOutputLayer)
            {
                capabilities.limits.maxFramebufferLayers = std::max(1U, limits.maxFramebufferLayers);
            }
            const auto maxDimension =
                std::max({limits.maxImageDimension1D, limits.maxImageDimension2D, limits.maxImageDimension3D});
            capabilities.limits.maxTextureMipLevels =
                maxDimension == 0 ? 0 : static_cast<std::uint32_t>(std::bit_width(maxDimension));
            capabilities.limits.maxComputeWorkgroupInvocations = limits.maxComputeWorkGroupInvocations;
            capabilities.limits.maxComputeWorkgroupSize = {
                limits.maxComputeWorkGroupSize[0],
                limits.maxComputeWorkGroupSize[1],
                limits.maxComputeWorkGroupSize[2],
            };
            capabilities.limits.maxComputeWorkgroupCount = {
                limits.maxComputeWorkGroupCount[0],
                limits.maxComputeWorkGroupCount[1],
                limits.maxComputeWorkGroupCount[2],
            };
            capabilities.limits.maxBufferSize = limits.maxStorageBufferRange;
            capabilities.limits.maxUniformBufferSize = limits.maxUniformBufferRange;
            capabilities.limits.minUniformBufferOffsetAlignment = limits.minUniformBufferOffsetAlignment;
            capabilities.limits.maxSamplerAnisotropy = static_cast<std::uint32_t>(limits.maxSamplerAnisotropy);
            capabilities.limits.maxSamplerLodBias = limits.maxSamplerLodBias;
            capabilities.limits.maxColorAttachments = std::min(limits.maxColorAttachments, maxColorAttachments);
            capabilities.limits.maxViewports = 1;
            if (record.multiViewport && limits.maxViewports > 1)
            {
                capabilities.limits.maxViewports = std::min(limits.maxViewports, maxViewportCount);
            }
            capabilities.limits.maxVertexBuffers = std::min(limits.maxVertexInputBindings, maxVertexBuffers);
            capabilities.limits.maxVertexAttributes = std::min(limits.maxVertexInputAttributes, maxVertexAttributes);
            capabilities.limits.maxBindingSets = std::min(limits.maxBoundDescriptorSets, maxBindingSets);
            capabilities.limits.maxPushConstantBytes = std::min(limits.maxPushConstantsSize, maxPushConstantBytes);
            if (record.shaderClipDistance)
            {
                capabilities.limits.maxClipDistances = limits.maxClipDistances;
            }
            if (record.shaderCullDistance)
            {
                capabilities.limits.maxCullDistances = limits.maxCullDistances;
            }
            if (record.shaderClipDistance || record.shaderCullDistance)
            {
                capabilities.limits.maxCombinedClipAndCullDistances = limits.maxCombinedClipAndCullDistances;
            }
        }

        struct HostVisibleMemoryTraits final
        {
            bool unified = false;
            bool coherentUnified = false;
            bool hostCoherent = false;
        };

        [[nodiscard]] bool unifiedMemoryFrom(VkPhysicalDeviceType type,
                                             bool everyHeapDeviceLocal,
                                             bool deviceLocalHostVisible) noexcept
        {
            switch (type)
            {
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                return true;
            case VK_PHYSICAL_DEVICE_TYPE_OTHER:
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                return everyHeapDeviceLocal && deviceLocalHostVisible;
            default:
                return false;
            }
        }

        [[nodiscard]] HostVisibleMemoryTraits classifyHostVisibleMemory(
            VkPhysicalDeviceType type, const VkPhysicalDeviceMemoryProperties& memory) noexcept
        {
            constexpr VkMemoryPropertyFlags deviceLocalHostVisibleFlags =
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            constexpr VkMemoryPropertyFlags cacheCoherentFlags =
                deviceLocalHostVisibleFlags | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
            HostVisibleMemoryTraits traits{};
            bool anyHostVisible = false;
            bool allHostVisibleCoherent = true;
            bool deviceLocalHostVisible = false;
            bool cacheCoherentDeviceLocal = false;
            for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            {
                const auto flags = memory.memoryTypes[i].propertyFlags;
                if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0)
                {
                    anyHostVisible = true;
                    allHostVisibleCoherent =
                        allHostVisibleCoherent && (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
                }
                deviceLocalHostVisible =
                    deviceLocalHostVisible || (flags & deviceLocalHostVisibleFlags) == deviceLocalHostVisibleFlags;
                cacheCoherentDeviceLocal =
                    cacheCoherentDeviceLocal || (flags & cacheCoherentFlags) == cacheCoherentFlags;
            }
            const bool everyHeapDeviceLocal = std::ranges::all_of(
                std::span{memory.memoryHeaps}.first(memory.memoryHeapCount),
                [](const VkMemoryHeap& heap) { return (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0; });
            traits.unified = unifiedMemoryFrom(type, everyHeapDeviceLocal, deviceLocalHostVisible);
            traits.coherentUnified = traits.unified && cacheCoherentDeviceLocal;
            traits.hostCoherent = anyHostVisible && allHostVisibleCoherent;
            return traits;
        }

        void publishMemoryCapabilities(DeviceCapabilities& capabilities,
                                       const VulkanPhysicalDeviceRecord& record) noexcept
        {
            const auto hostMemory = classifyHostVisibleMemory(record.properties.deviceType, record.memory);
            const auto nonCoherentAtomSize = record.properties.limits.nonCoherentAtomSize;
            capabilities.unifiedMemory = hostMemory.unified;
            capabilities.cacheCoherentUnifiedMemory = hostMemory.coherentUnified;
            capabilities.mappedMemory.hostCoherent = hostMemory.hostCoherent;
            capabilities.mappedMemory.nonCoherentAtomSize = nonCoherentAtomSize;
            capabilities.persistentMapping = hostMemory.hostCoherent || nonCoherentAtomSize > 0;
            capabilities.memoryBudgets = true;
            capabilities.residencyPriorities = record.memoryPriority;
        }

        void publishBindingArrayCapabilities(DeviceCapabilities& capabilities,
                                             const VulkanPhysicalDeviceRecord& record) noexcept
        {
            const bool descriptorIndexingContract =
                record.runtimeDescriptorArray && record.shaderStorageBufferArrayNonUniformIndexing &&
                record.shaderSampledImageArrayNonUniformIndexing && record.shaderStorageImageArrayNonUniformIndexing;
            capabilities.partiallyBoundBindingArrays =
                descriptorIndexingContract && record.descriptorBindingPartiallyBound;
            capabilities.variableDescriptorCountBindingArrays =
                descriptorIndexingContract && record.descriptorBindingVariableDescriptorCount;
            capabilities.updateAfterBindBindingArrays = capabilities.partiallyBoundBindingArrays &&
                                                        record.descriptorBindingUpdateUnusedWhilePending &&
                                                        record.descriptorBindingSampledImageUpdateAfterBind &&
                                                        record.descriptorBindingStorageImageUpdateAfterBind &&
                                                        record.descriptorBindingStorageBufferUpdateAfterBind;
            if (capabilities.variableDescriptorCountBindingArrays)
            {
                const auto& limits = record.properties.limits;
                capabilities.limits.maxVariableResourceDescriptors = std::min({
                    65'536U,
                    variableArrayDescriptorLimit(limits.maxPerStageDescriptorStorageBuffers),
                    variableArrayDescriptorLimit(limits.maxDescriptorSetStorageBuffers),
                    variableArrayDescriptorLimit(limits.maxPerStageDescriptorSampledImages),
                    variableArrayDescriptorLimit(limits.maxDescriptorSetSampledImages),
                    variableArrayDescriptorLimit(limits.maxPerStageDescriptorStorageImages),
                    variableArrayDescriptorLimit(limits.maxDescriptorSetStorageImages),
                    variableArrayDescriptorLimit(limits.maxPerStageResources),
                });
                capabilities.limits.maxVariableBufferDescriptors = std::min({
                    capabilities.limits.maxVariableResourceDescriptors,
                    variableArrayDescriptorLimit(limits.maxPerStageDescriptorUniformBuffers),
                    variableArrayDescriptorLimit(limits.maxDescriptorSetUniformBuffers),
                });
                capabilities.limits.maxVariableSamplerDescriptors = std::min({
                    2'048U,
                    variableArrayDescriptorLimit(limits.maxPerStageDescriptorSamplers),
                    variableArrayDescriptorLimit(limits.maxDescriptorSetSamplers),
                });
            }
        }

        void publishShaderCapabilities(DeviceCapabilities& capabilities,
                                       const VulkanPhysicalDeviceRecord& record) noexcept
        {
            capabilities.shaderFloat16 = record.shaderFloat16;
            capabilities.shaderInt16 = record.shaderInt16;
            capabilities.shaderInt64 = record.shaderInt64;
            capabilities.shaderInt64AtomicGroupShared = record.shaderInt64 && record.shaderSharedInt64Atomics;
            capabilities.shaderInt64AtomicTypedResource = record.shaderInt64 && record.shaderImageInt64Atomics;
            capabilities.shaderInt64AtomicStorageBuffer = record.shaderInt64 && record.shaderBufferInt64Atomics;
            capabilities.subgroupOperations = record.subgroupOperations;
            if (capabilities.subgroupOperations)
            {
                const bool requiredComputeSize = record.subgroupSizeControl &&
                                                 (record.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
                capabilities.subgroup = {
                    .minSize = record.subgroupMinSize,
                    .maxSize = record.subgroupMaxSize,
                    .requiredSizeStages = requiredComputeSize ? ShaderStage::compute : ShaderStage::none,
                };
            }
            capabilities.geometryShaders = record.geometryShaders;
            capabilities.tessellationShaders = record.tessellationShaders;
            if (record.tessellationShaders)
            {
                capabilities.tessellation.maxPatchControlPoints = record.tessellationMaxPatchControlPoints;
            }
            capabilities.fragmentPrimitiveId = record.geometryShaders;
            capabilities.fragmentBarycentricCoordinates = record.fragmentBarycentric;
            capabilities.rasterizerOrderedStorageAccess =
                record.fragmentShaderPixelInterlock && record.fragmentStoresAndAtomics;
            capabilities.meshShaders = record.meshShaders;
            capabilities.taskShaders = record.taskShaders;
            if (record.meshShaders)
            {
                capabilities.meshShader = record.meshShader;
            }
            capabilities.meshPipelineStatisticsQueries = record.meshShaders && record.meshShaderQueries &&
                                                         record.pipelineStatisticsQuery && record.hostQueryReset;
        }

        void publishRayTracingCapabilities(DeviceCapabilities& capabilities,
                                           const VulkanPhysicalDeviceRecord& record) noexcept
        {
            capabilities.rayTracing = record.rayTracingPipeline;
            capabilities.indirectRayDispatch = record.indirectRayDispatch;
            capabilities.rayQuery = record.rayQuery;
            if (record.accelerationStructures || record.rayTracingPipeline)
            {
                capabilities.rayTracingLimits = record.rayTracingLimits;
            }
        }

        void publishRenderStateCapabilities(DeviceCapabilities& capabilities,
                                            const VulkanPhysicalDeviceRecord& record) noexcept
        {
            capabilities.variableRateShading = record.variableRateShading;
            capabilities.additionalShadingRates = record.additionalShadingRates;
            capabilities.variableRateShadingTier2 = record.variableRateShadingTier2;
            capabilities.perPrimitiveShadingRateWithViewportIndexing =
                record.variableRateShadingTier2 && record.perPrimitiveShadingRateWithViewportIndexing;
            capabilities.shadingRateImageTileSize =
                record.variableRateShadingTier2 ? record.shadingRateImageTileSize : 0;
            if (record.shaderOutputLayer)
            {
                capabilities.layeredRenderingModes = LayeredRenderingMode::vertexStage;
            }
            capabilities.multiviewRendering = record.multiview && record.maxMultiviewViewCount > 0;
            capabilities.maxMultiviewViewCount = capabilities.multiviewRendering ? record.maxMultiviewViewCount : 0;
            capabilities.multiViewport = record.multiViewport && capabilities.limits.maxViewports > 1;
            capabilities.depthBoundsTest = record.depthBounds;
            capabilities.independentFrontAndBackStencilReference = true;
            capabilities.wireframeRasterization = record.fillModeNonSolid;
            capabilities.dynamicDepthBias = true;
            capabilities.conservativeRasterization = record.conservativeRasterization;
            capabilities.unclippedDepth = record.depthClamp;
            capabilities.logicOperations = record.logicOp;
            capabilities.primitiveRestart = true;
            capabilities.triangleFans = true;
        }

        void publishVertexInputCapabilities(DeviceCapabilities& capabilities,
                                            const VulkanInstanceTable& instanceTable,
                                            const VulkanPhysicalDeviceRecord& record) noexcept
        {
            const auto& limits = record.properties.limits;
            capabilities.vertexInput.maxBufferStride = limits.maxVertexInputBindingStride;
            capabilities.vertexInput.maxAttributeOffset = limits.maxVertexInputAttributeOffset;
            capabilities.vertexInput.maxInstanceStepRate = 1;
            for (std::uint32_t index = 0; index < vertexFormatCount; ++index)
            {
                const auto format = static_cast<VertexFormat>(index);
                const auto vk = vulkanVertexFormat(format);
                if (vk == VK_FORMAT_UNDEFINED)
                {
                    continue;
                }
                VkFormatProperties native{};
                if (instanceTable.vkGetPhysicalDeviceFormatProperties != nullptr)
                {
                    instanceTable.vkGetPhysicalDeviceFormatProperties(record.physical, vk, &native);
                }
                if ((native.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) != 0)
                {
                    capabilities.vertexInput.supportedFormats |= std::uint64_t{1} << index;
                }
            }
        }

    } // namespace

    DeviceCapabilities publishVulkanCapabilities(const VulkanInstanceTable& instanceTable,
                                                 const VulkanPhysicalDeviceRecord& record,
                                                 bool debugMarkers) noexcept
    {
        DeviceCapabilities capabilities{};
        capabilities.debugMarkers = debugMarkers;
        capabilities.coordinateConventions = {
            .ndcYAxis = NdcYAxisDirection::up,
            .ndcDepthRange = NdcDepthRange::zeroToOne,
            .textureOrigin = TextureOrigin::upperLeft,
        };
        capabilities.nodeCount = 1;
        publishResourceLimits(capabilities, record);
        publishMemoryCapabilities(capabilities, record);
        capabilities.textureViewComponentSwizzle = record.imageViewFormatSwizzle;
        capabilities.bufferBarrierRanges = true;
        capabilities.enhancedBarriers = true;
        capabilities.pipelineCaches = true;
        capabilities.mipmapGeneration = true;
        capabilities.indirectDispatch = true;
        capabilities.multiDispatchIndirect = true;
        capabilities.dispatchIndirectCount = record.bufferDeviceAddress && record.shaderInt64;
        capabilities.indirectDraws = true;
        capabilities.multiDrawIndirect = record.multiDrawIndirect;
        capabilities.drawIndirectCount = record.drawIndirectCount;
        capabilities.shaderDrawIndex = record.shaderDrawParameters;
        capabilities.conditionalExecution = record.conditionalRendering;
        publishBindingArrayCapabilities(capabilities, record);
        publishShaderCapabilities(capabilities, record);

        VkPhysicalDeviceFeatures features{};
        if (instanceTable.vkGetPhysicalDeviceFeatures != nullptr)
        {
            instanceTable.vkGetPhysicalDeviceFeatures(record.physical, &features);
        }
        capabilities.indirectDrawBaseInstance = features.drawIndirectFirstInstance == VK_TRUE;
        capabilities.dualSourceBlending = features.dualSrcBlend == VK_TRUE;

        publishRayTracingCapabilities(capabilities, record);
        capabilities.sharedBufferHandles = record.sharedBufferHandles;
        capabilities.sharedTextureHandles = record.sharedTextureHandles;
        capabilities.sharedTimelineHandles = record.sharedTimelineHandles;
        publishRenderStateCapabilities(capabilities, record);
        capabilities.samplerMinMaxReduction = record.samplerFilterMinmax;
        capabilities.occlusionQueries = record.occlusionQueryPrecise && record.hostQueryReset;
        capabilities.pipelineStatisticsQueries = record.pipelineStatisticsQuery && record.hostQueryReset;
        publishVertexInputCapabilities(capabilities, instanceTable, record);
        return capabilities;
    }

    void recordVulkanQueueFamilies(VulkanPhysicalDeviceRecord& record,
                                   std::span<const VkQueueFamilyProperties> families)
    {
        record.queues = selectVulkanQueueFamilies(families);
        record.queueTraits = VulkanQueueFamilyTraitSet{
            .graphics = queueFamilyTraits(families, record.queues.graphics),
            .compute = queueFamilyTraits(families, record.queues.compute),
            .transfer = queueFamilyTraits(families, record.queues.transfer),
        };
    }

    QueueProperties publishVulkanQueueProperties(const VulkanPhysicalDeviceRecord& record, QueueType queueType) noexcept
    {
        auto traits = record.queueTraits.graphics;
        if (queueType == QueueType::compute)
        {
            traits = record.queueTraits.compute;
        }
        else if (queueType == QueueType::copy)
        {
            traits = record.queueTraits.transfer;
        }
        QueueProperties properties{};
        properties.timestampValidBits = traits.timestampValidBits;
        properties.timestampQueries = traits.timestampValidBits != 0 &&
                                      (traits.queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) != 0 &&
                                      record.properties.limits.timestampComputeAndGraphics == VK_TRUE;
        properties.inlineTimestampWrites = properties.timestampQueries;
        if (record.properties.limits.timestampPeriod > 0.0F)
        {
            properties.timestampFrequency = static_cast<std::uint64_t>(
                1'000'000'000.0 / static_cast<double>(record.properties.limits.timestampPeriod));
        }
        return properties;
    }

    namespace
    {

        [[nodiscard]] constexpr bool isDepthFormat(Format format) noexcept
        {
            return format == Format::d32Float || format == Format::d24UnormS8Uint;
        }

        [[nodiscard]] bool isIntegerColorFormat(Format format) noexcept
        {
            switch (format)
            {
            case Format::r8Uint:
            case Format::rg8Uint:
            case Format::rgba8Uint:
            case Format::r8Sint:
            case Format::rg8Sint:
            case Format::rgba8Sint:
            case Format::r16Uint:
            case Format::rg16Uint:
            case Format::rgba16Uint:
            case Format::r16Sint:
            case Format::rg16Sint:
            case Format::rgba16Sint:
            case Format::r32Uint:
            case Format::rg32Uint:
            case Format::rgba32Uint:
            case Format::r32Sint:
            case Format::rg32Sint:
            case Format::rgba32Sint:
            case Format::r64Uint:
                return true;
            default:
                return false;
            }
        }

        [[nodiscard]] FormatFeature formatFeaturesFrom(VkFormatFeatureFlags native, Format format) noexcept
        {
            FormatFeature features = FormatFeature::none;
            if ((native & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0)
            {
                features |= FormatFeature::sampled | FormatFeature::texture2D;
            }
            if ((native & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0)
            {
                features |= FormatFeature::filterable;
            }
            if ((native & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0)
            {
                features |= FormatFeature::storageRead | FormatFeature::storageWrite;
            }
            if ((native & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) != 0)
            {
                features |= FormatFeature::colorAttachment;
            }
            if ((native & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) != 0)
            {
                features |= FormatFeature::blendable;
            }
            if ((native & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) != 0 && isIntegerColorFormat(format))
            {
                features |= FormatFeature::logicOperation;
            }
            if ((native & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0)
            {
                features |= FormatFeature::depthStencilAttachment;
            }
            if ((native & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT) != 0)
            {
                features |= FormatFeature::copySource;
            }
            if ((native & VK_FORMAT_FEATURE_TRANSFER_DST_BIT) != 0)
            {
                features |= FormatFeature::copyDestination;
            }
            return features;
        }

        void probeImageTypeSupport(FormatProperties& properties,
                                   const VulkanInstanceTable& instanceTable,
                                   VkPhysicalDevice physical,
                                   VkFormat vk) noexcept
        {
            const auto tryImage = [&](VkImageType type, VkImageUsageFlags usage, FormatFeature mark) {
                VkImageFormatProperties image{};
                if (instanceTable.vkGetPhysicalDeviceImageFormatProperties(
                        physical, vk, type, VK_IMAGE_TILING_OPTIMAL, usage, 0, &image) != VK_SUCCESS)
                {
                    return;
                }
                properties.features |= mark;
                properties.sampleCounts |= sampleMask(image.sampleCounts);
            };
            tryImage(VK_IMAGE_TYPE_1D, VK_IMAGE_USAGE_SAMPLED_BIT, FormatFeature::texture1D);
            tryImage(VK_IMAGE_TYPE_1D, VK_IMAGE_USAGE_TRANSFER_DST_BIT, FormatFeature::texture1D);
            tryImage(VK_IMAGE_TYPE_2D, VK_IMAGE_USAGE_SAMPLED_BIT, FormatFeature::texture2D);
            tryImage(VK_IMAGE_TYPE_2D, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, FormatFeature::texture2D);
            tryImage(VK_IMAGE_TYPE_2D, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, FormatFeature::texture2D);
            tryImage(VK_IMAGE_TYPE_2D, VK_IMAGE_USAGE_TRANSFER_DST_BIT, FormatFeature::texture2D);
            tryImage(VK_IMAGE_TYPE_3D, VK_IMAGE_USAGE_SAMPLED_BIT, FormatFeature::texture3D);
            tryImage(VK_IMAGE_TYPE_3D, VK_IMAGE_USAGE_TRANSFER_DST_BIT, FormatFeature::texture3D);
            VkImageFormatProperties cube{};
            if (instanceTable.vkGetPhysicalDeviceImageFormatProperties(physical,
                                                                       vk,
                                                                       VK_IMAGE_TYPE_2D,
                                                                       VK_IMAGE_TILING_OPTIMAL,
                                                                       VK_IMAGE_USAGE_SAMPLED_BIT,
                                                                       VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
                                                                       &cube) == VK_SUCCESS)
            {
                properties.features |= FormatFeature::cube;
            }
        }

        [[nodiscard]] bool hasReinterpretableViewFormat(Format format) noexcept
        {
            for (const auto alternate : {Format::rgba8Unorm,
                                         Format::rgba8UnormSrgb,
                                         Format::bgra8Unorm,
                                         Format::bgra8UnormSrgb,
                                         Format::bc1RgbaUnorm,
                                         Format::bc1RgbaUnormSrgb,
                                         Format::bc2RgbaUnorm,
                                         Format::bc2RgbaUnormSrgb,
                                         Format::bc3RgbaUnorm,
                                         Format::bc3RgbaUnormSrgb,
                                         Format::bc7RgbaUnorm,
                                         Format::bc7RgbaUnormSrgb})
            {
                if (isTextureViewFormatCompatible(format, alternate) && alternate != format)
                {
                    return true;
                }
            }
            return false;
        }

    } // namespace

    FormatProperties queryVulkanFormatProperties(const VulkanInstanceTable& instanceTable,
                                                 VkPhysicalDevice physical,
                                                 Format format) noexcept
    {
        FormatProperties properties{};
        const auto vk = vulkanFormat(format);
        if (vk == VK_FORMAT_UNDEFINED || instanceTable.vkGetPhysicalDeviceFormatProperties == nullptr ||
            instanceTable.vkGetPhysicalDeviceImageFormatProperties == nullptr)
        {
            return properties;
        }

        VkFormatProperties native{};
        instanceTable.vkGetPhysicalDeviceFormatProperties(physical, vk, &native);
        properties.features = formatFeaturesFrom(native.optimalTilingFeatures, format);
        probeImageTypeSupport(properties, instanceTable, physical, vk);
        if (isDepthFormat(format))
        {
            properties.features = detail::depthTextureFormatFeatures(properties.features);
        }
        if (hasReinterpretableViewFormat(format))
        {
            properties.features |= FormatFeature::viewFormatReinterpretation;
        }

        if (hasAny(properties.features, FormatFeature::colorAttachment) && !isIntegerColorFormat(format) &&
            (supportsSampleCount(properties.sampleCounts, 2) || supportsSampleCount(properties.sampleCounts, 4) ||
             supportsSampleCount(properties.sampleCounts, 8)))
        {
            properties.features |= FormatFeature::multisampleResolve;
            properties.colorResolveModes = ResolveModeMask::average;
        }
        return properties;
    }

    void loadVulkanDeviceTable(VolkDeviceTable& table,
                               VkDevice device,
                               PFN_vkGetDeviceProcAddr getDeviceProcAddr) noexcept
    {
        loadVulkanBasicDeviceTable(table, device, getDeviceProcAddr);
        if (device == VK_NULL_HANDLE || getDeviceProcAddr == nullptr)
        {
            return;
        }

#define RACHIS_LOAD_DEVICE_PROC(name)                                                                                  \
    table.name = dispatch_detail::resolveVulkanProc<PFN_##name>(device, getDeviceProcAddr, #name)

#if defined(VK_VERSION_1_0)
        RACHIS_LOAD_DEVICE_PROC(vkAllocateDescriptorSets);
        RACHIS_LOAD_DEVICE_PROC(vkCmdBeginQuery);
        RACHIS_LOAD_DEVICE_PROC(vkCmdBeginRenderPass);
        RACHIS_LOAD_DEVICE_PROC(vkCmdBindDescriptorSets);
        RACHIS_LOAD_DEVICE_PROC(vkCmdBindIndexBuffer);
        RACHIS_LOAD_DEVICE_PROC(vkCmdBindPipeline);
        RACHIS_LOAD_DEVICE_PROC(vkCmdBindVertexBuffers);
        RACHIS_LOAD_DEVICE_PROC(vkCmdBlitImage);
        RACHIS_LOAD_DEVICE_PROC(vkCmdClearAttachments);
        RACHIS_LOAD_DEVICE_PROC(vkCmdClearColorImage);
        RACHIS_LOAD_DEVICE_PROC(vkCmdClearDepthStencilImage);
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyBufferToImage);
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyImage);
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyImageToBuffer);
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyQueryPoolResults);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDispatch);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDispatchIndirect);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDraw);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDrawIndexed);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDrawIndexedIndirect);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDrawIndirect);
        RACHIS_LOAD_DEVICE_PROC(vkCmdEndQuery);
        RACHIS_LOAD_DEVICE_PROC(vkCmdEndRenderPass);
        RACHIS_LOAD_DEVICE_PROC(vkCmdExecuteCommands);
        RACHIS_LOAD_DEVICE_PROC(vkCmdFillBuffer);
        RACHIS_LOAD_DEVICE_PROC(vkCmdNextSubpass);
        RACHIS_LOAD_DEVICE_PROC(vkCmdPipelineBarrier);
        RACHIS_LOAD_DEVICE_PROC(vkCmdPushConstants);
        RACHIS_LOAD_DEVICE_PROC(vkCmdResetEvent);
        RACHIS_LOAD_DEVICE_PROC(vkCmdResetQueryPool);
        RACHIS_LOAD_DEVICE_PROC(vkCmdResolveImage);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetBlendConstants);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDepthBias);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDepthBounds);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetEvent);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetLineWidth);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetScissor);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetStencilCompareMask);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetStencilReference);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetStencilWriteMask);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetViewport);
        RACHIS_LOAD_DEVICE_PROC(vkCmdUpdateBuffer);
        RACHIS_LOAD_DEVICE_PROC(vkCmdWaitEvents);
        RACHIS_LOAD_DEVICE_PROC(vkCmdWriteTimestamp);
        RACHIS_LOAD_DEVICE_PROC(vkCreateBufferView);
        RACHIS_LOAD_DEVICE_PROC(vkCreateComputePipelines);
        RACHIS_LOAD_DEVICE_PROC(vkCreateDescriptorPool);
        RACHIS_LOAD_DEVICE_PROC(vkCreateDescriptorSetLayout);
        RACHIS_LOAD_DEVICE_PROC(vkCreateEvent);
        RACHIS_LOAD_DEVICE_PROC(vkCreateFence);
        RACHIS_LOAD_DEVICE_PROC(vkCreateFramebuffer);
        RACHIS_LOAD_DEVICE_PROC(vkCreateGraphicsPipelines);
        RACHIS_LOAD_DEVICE_PROC(vkCreateImageView);
        RACHIS_LOAD_DEVICE_PROC(vkCreatePipelineCache);
        RACHIS_LOAD_DEVICE_PROC(vkCreatePipelineLayout);
        RACHIS_LOAD_DEVICE_PROC(vkCreateQueryPool);
        RACHIS_LOAD_DEVICE_PROC(vkCreateRenderPass);
        RACHIS_LOAD_DEVICE_PROC(vkCreateSampler);
        RACHIS_LOAD_DEVICE_PROC(vkCreateSemaphore);
        RACHIS_LOAD_DEVICE_PROC(vkCreateShaderModule);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyBufferView);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyDescriptorPool);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyDescriptorSetLayout);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyEvent);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyFence);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyFramebuffer);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyImageView);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyPipeline);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyPipelineCache);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyPipelineLayout);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyQueryPool);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyRenderPass);
        RACHIS_LOAD_DEVICE_PROC(vkDestroySampler);
        RACHIS_LOAD_DEVICE_PROC(vkDestroySemaphore);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyShaderModule);
        RACHIS_LOAD_DEVICE_PROC(vkFreeCommandBuffers);
        RACHIS_LOAD_DEVICE_PROC(vkFreeDescriptorSets);
        RACHIS_LOAD_DEVICE_PROC(vkGetDeviceMemoryCommitment);
        RACHIS_LOAD_DEVICE_PROC(vkGetEventStatus);
        RACHIS_LOAD_DEVICE_PROC(vkGetFenceStatus);
        RACHIS_LOAD_DEVICE_PROC(vkGetImageSparseMemoryRequirements);
        RACHIS_LOAD_DEVICE_PROC(vkGetImageSubresourceLayout);
        RACHIS_LOAD_DEVICE_PROC(vkGetPipelineCacheData);
        RACHIS_LOAD_DEVICE_PROC(vkGetQueryPoolResults);
        RACHIS_LOAD_DEVICE_PROC(vkGetRenderAreaGranularity);
        RACHIS_LOAD_DEVICE_PROC(vkMergePipelineCaches);
        RACHIS_LOAD_DEVICE_PROC(vkQueueBindSparse);
        RACHIS_LOAD_DEVICE_PROC(vkResetCommandBuffer);
        RACHIS_LOAD_DEVICE_PROC(vkResetCommandPool);
        RACHIS_LOAD_DEVICE_PROC(vkResetDescriptorPool);
        RACHIS_LOAD_DEVICE_PROC(vkResetEvent);
        RACHIS_LOAD_DEVICE_PROC(vkResetFences);
        RACHIS_LOAD_DEVICE_PROC(vkSetEvent);
        RACHIS_LOAD_DEVICE_PROC(vkUpdateDescriptorSets);
        RACHIS_LOAD_DEVICE_PROC(vkWaitForFences);
#endif
#if defined(VK_VERSION_1_1)
        RACHIS_LOAD_DEVICE_PROC(vkCmdDispatchBase);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDeviceMask);
        RACHIS_LOAD_DEVICE_PROC(vkCreateDescriptorUpdateTemplate);
        RACHIS_LOAD_DEVICE_PROC(vkCreateSamplerYcbcrConversion);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyDescriptorUpdateTemplate);
        RACHIS_LOAD_DEVICE_PROC(vkDestroySamplerYcbcrConversion);
        RACHIS_LOAD_DEVICE_PROC(vkGetDescriptorSetLayoutSupport);
        RACHIS_LOAD_DEVICE_PROC(vkGetDeviceGroupPeerMemoryFeatures);
        RACHIS_LOAD_DEVICE_PROC(vkGetDeviceQueue2);
        RACHIS_LOAD_DEVICE_PROC(vkGetImageSparseMemoryRequirements2);
        RACHIS_LOAD_DEVICE_PROC(vkTrimCommandPool);
        RACHIS_LOAD_DEVICE_PROC(vkUpdateDescriptorSetWithTemplate);
#endif
#if defined(VK_VERSION_1_2)
        RACHIS_LOAD_DEVICE_PROC(vkCmdBeginRenderPass2);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDrawIndexedIndirectCount);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDrawIndirectCount);
        RACHIS_LOAD_DEVICE_PROC(vkCmdEndRenderPass2);
        RACHIS_LOAD_DEVICE_PROC(vkCmdNextSubpass2);
        RACHIS_LOAD_DEVICE_PROC(vkCreateRenderPass2);
        RACHIS_LOAD_DEVICE_PROC(vkGetBufferDeviceAddress);
        RACHIS_LOAD_DEVICE_PROC(vkGetBufferOpaqueCaptureAddress);
        RACHIS_LOAD_DEVICE_PROC(vkGetDeviceMemoryOpaqueCaptureAddress);
        RACHIS_LOAD_DEVICE_PROC(vkGetSemaphoreCounterValue);
        RACHIS_LOAD_DEVICE_PROC(vkResetQueryPool);
        RACHIS_LOAD_DEVICE_PROC(vkSignalSemaphore);
        RACHIS_LOAD_DEVICE_PROC(vkWaitSemaphores);
#endif
#if defined(VK_VERSION_1_3)
        RACHIS_LOAD_DEVICE_PROC(vkCmdBeginRendering);
        RACHIS_LOAD_DEVICE_PROC(vkCmdBindVertexBuffers2);
        RACHIS_LOAD_DEVICE_PROC(vkCmdBlitImage2);
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyBuffer2);
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyBufferToImage2);
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyImage2);
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyImageToBuffer2);
        RACHIS_LOAD_DEVICE_PROC(vkCmdEndRendering);
        RACHIS_LOAD_DEVICE_PROC(vkCmdPipelineBarrier2);
        RACHIS_LOAD_DEVICE_PROC(vkCmdResetEvent2);
        RACHIS_LOAD_DEVICE_PROC(vkCmdResolveImage2);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetCullMode);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDepthBiasEnable);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDepthBoundsTestEnable);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDepthCompareOp);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDepthTestEnable);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDepthWriteEnable);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetEvent2);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetFrontFace);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetPrimitiveRestartEnable);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetPrimitiveTopology);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetRasterizerDiscardEnable);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetScissorWithCount);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetStencilOp);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetStencilTestEnable);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetViewportWithCount);
        RACHIS_LOAD_DEVICE_PROC(vkCmdWaitEvents2);
        RACHIS_LOAD_DEVICE_PROC(vkCmdWriteTimestamp2);
        RACHIS_LOAD_DEVICE_PROC(vkCreatePrivateDataSlot);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyPrivateDataSlot);
        RACHIS_LOAD_DEVICE_PROC(vkGetDeviceImageSparseMemoryRequirements);
        RACHIS_LOAD_DEVICE_PROC(vkGetPrivateData);
        RACHIS_LOAD_DEVICE_PROC(vkQueueSubmit2);
        RACHIS_LOAD_DEVICE_PROC(vkSetPrivateData);
#endif
#if defined(VK_AMDX_shader_enqueue)
        RACHIS_LOAD_DEVICE_PROC(vkCmdDispatchGraphAMDX);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDispatchGraphIndirectAMDX);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDispatchGraphIndirectCountAMDX);
        RACHIS_LOAD_DEVICE_PROC(vkCmdInitializeGraphScratchMemoryAMDX);
        RACHIS_LOAD_DEVICE_PROC(vkCreateExecutionGraphPipelinesAMDX);
        RACHIS_LOAD_DEVICE_PROC(vkGetExecutionGraphPipelineNodeIndexAMDX);
        RACHIS_LOAD_DEVICE_PROC(vkGetExecutionGraphPipelineScratchSizeAMDX);
#endif
#if defined(VK_AMD_anti_lag)
        RACHIS_LOAD_DEVICE_PROC(vkAntiLagUpdateAMD);
#endif
#if defined(VK_AMD_buffer_marker)
        RACHIS_LOAD_DEVICE_PROC(vkCmdWriteBufferMarkerAMD);
#endif
#if defined(VK_AMD_display_native_hdr)
        RACHIS_LOAD_DEVICE_PROC(vkSetLocalDimmingAMD);
#endif
#if defined(VK_AMD_draw_indirect_count)
        RACHIS_LOAD_DEVICE_PROC(vkCmdDrawIndexedIndirectCountAMD);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDrawIndirectCountAMD);
#endif
#if defined(VK_AMD_shader_info)
        RACHIS_LOAD_DEVICE_PROC(vkGetShaderInfoAMD);
#endif
#if defined(VK_ANDROID_external_memory_android_hardware_buffer)
        RACHIS_LOAD_DEVICE_PROC(vkGetAndroidHardwareBufferPropertiesANDROID);
        RACHIS_LOAD_DEVICE_PROC(vkGetMemoryAndroidHardwareBufferANDROID);
#endif
#if defined(VK_EXT_attachment_feedback_loop_dynamic_state)
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetAttachmentFeedbackLoopEnableEXT);
#endif
#if defined(VK_EXT_buffer_device_address)
        RACHIS_LOAD_DEVICE_PROC(vkGetBufferDeviceAddressEXT);
#endif
#if defined(VK_EXT_calibrated_timestamps)
        RACHIS_LOAD_DEVICE_PROC(vkGetCalibratedTimestampsEXT);
#endif
#if defined(VK_EXT_color_write_enable)
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetColorWriteEnableEXT);
#endif
#if defined(VK_EXT_conditional_rendering)
        RACHIS_LOAD_DEVICE_PROC(vkCmdBeginConditionalRenderingEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdEndConditionalRenderingEXT);
#endif
#if defined(VK_EXT_debug_marker)
        RACHIS_LOAD_DEVICE_PROC(vkCmdDebugMarkerBeginEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDebugMarkerEndEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDebugMarkerInsertEXT);
        RACHIS_LOAD_DEVICE_PROC(vkDebugMarkerSetObjectNameEXT);
        RACHIS_LOAD_DEVICE_PROC(vkDebugMarkerSetObjectTagEXT);
#endif
#if defined(VK_EXT_depth_bias_control)
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDepthBias2EXT);
#endif
#if defined(VK_EXT_descriptor_buffer)
        RACHIS_LOAD_DEVICE_PROC(vkCmdBindDescriptorBufferEmbeddedSamplersEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdBindDescriptorBuffersEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDescriptorBufferOffsetsEXT);
        RACHIS_LOAD_DEVICE_PROC(vkGetBufferOpaqueCaptureDescriptorDataEXT);
        RACHIS_LOAD_DEVICE_PROC(vkGetDescriptorEXT);
        RACHIS_LOAD_DEVICE_PROC(vkGetDescriptorSetLayoutBindingOffsetEXT);
        RACHIS_LOAD_DEVICE_PROC(vkGetDescriptorSetLayoutSizeEXT);
        RACHIS_LOAD_DEVICE_PROC(vkGetImageOpaqueCaptureDescriptorDataEXT);
        RACHIS_LOAD_DEVICE_PROC(vkGetImageViewOpaqueCaptureDescriptorDataEXT);
        RACHIS_LOAD_DEVICE_PROC(vkGetSamplerOpaqueCaptureDescriptorDataEXT);
#endif
#if defined(VK_EXT_descriptor_buffer) && (defined(VK_KHR_acceleration_structure) || defined(VK_NV_ray_tracing))
        RACHIS_LOAD_DEVICE_PROC(vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT);
#endif
#if defined(VK_EXT_device_fault)
        RACHIS_LOAD_DEVICE_PROC(vkGetDeviceFaultInfoEXT);
#endif
#if defined(VK_EXT_device_generated_commands)
        RACHIS_LOAD_DEVICE_PROC(vkCmdExecuteGeneratedCommandsEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdPreprocessGeneratedCommandsEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCreateIndirectCommandsLayoutEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCreateIndirectExecutionSetEXT);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyIndirectCommandsLayoutEXT);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyIndirectExecutionSetEXT);
        RACHIS_LOAD_DEVICE_PROC(vkGetGeneratedCommandsMemoryRequirementsEXT);
        RACHIS_LOAD_DEVICE_PROC(vkUpdateIndirectExecutionSetPipelineEXT);
        RACHIS_LOAD_DEVICE_PROC(vkUpdateIndirectExecutionSetShaderEXT);
#endif
#if defined(VK_EXT_discard_rectangles)
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDiscardRectangleEXT);
#endif
#if defined(VK_EXT_discard_rectangles) && VK_EXT_DISCARD_RECTANGLES_SPEC_VERSION >= 2
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDiscardRectangleEnableEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDiscardRectangleModeEXT);
#endif
#if defined(VK_EXT_display_control)
        RACHIS_LOAD_DEVICE_PROC(vkDisplayPowerControlEXT);
        RACHIS_LOAD_DEVICE_PROC(vkGetSwapchainCounterEXT);
        RACHIS_LOAD_DEVICE_PROC(vkRegisterDeviceEventEXT);
        RACHIS_LOAD_DEVICE_PROC(vkRegisterDisplayEventEXT);
#endif
#if defined(VK_EXT_external_memory_host)
        RACHIS_LOAD_DEVICE_PROC(vkGetMemoryHostPointerPropertiesEXT);
#endif
#if defined(VK_EXT_full_screen_exclusive)
        RACHIS_LOAD_DEVICE_PROC(vkAcquireFullScreenExclusiveModeEXT);
        RACHIS_LOAD_DEVICE_PROC(vkReleaseFullScreenExclusiveModeEXT);
#endif
#if defined(VK_EXT_full_screen_exclusive) && (defined(VK_KHR_device_group) || defined(VK_VERSION_1_1))
        RACHIS_LOAD_DEVICE_PROC(vkGetDeviceGroupSurfacePresentModes2EXT);
#endif
#if defined(VK_EXT_hdr_metadata)
        RACHIS_LOAD_DEVICE_PROC(vkSetHdrMetadataEXT);
#endif
#if defined(VK_EXT_host_image_copy)
        RACHIS_LOAD_DEVICE_PROC(vkCopyImageToImageEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCopyImageToMemoryEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCopyMemoryToImageEXT);
        RACHIS_LOAD_DEVICE_PROC(vkTransitionImageLayoutEXT);
#endif
#if defined(VK_EXT_host_query_reset)
        RACHIS_LOAD_DEVICE_PROC(vkResetQueryPoolEXT);
#endif
#if defined(VK_EXT_image_drm_format_modifier)
        RACHIS_LOAD_DEVICE_PROC(vkGetImageDrmFormatModifierPropertiesEXT);
#endif
#if defined(VK_EXT_line_rasterization)
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetLineStippleEXT);
#endif
#if defined(VK_EXT_mesh_shader)
        RACHIS_LOAD_DEVICE_PROC(vkCmdDrawMeshTasksEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDrawMeshTasksIndirectCountEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDrawMeshTasksIndirectEXT);
#endif
#if defined(VK_EXT_metal_objects)
        RACHIS_LOAD_DEVICE_PROC(vkExportMetalObjectsEXT);
#endif
#if defined(VK_EXT_multi_draw)
        RACHIS_LOAD_DEVICE_PROC(vkCmdDrawMultiEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDrawMultiIndexedEXT);
#endif
#if defined(VK_EXT_opacity_micromap)
        RACHIS_LOAD_DEVICE_PROC(vkBuildMicromapsEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdBuildMicromapsEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyMemoryToMicromapEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyMicromapEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyMicromapToMemoryEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdWriteMicromapsPropertiesEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCopyMemoryToMicromapEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCopyMicromapEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCopyMicromapToMemoryEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCreateMicromapEXT);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyMicromapEXT);
        RACHIS_LOAD_DEVICE_PROC(vkGetDeviceMicromapCompatibilityEXT);
        RACHIS_LOAD_DEVICE_PROC(vkGetMicromapBuildSizesEXT);
        RACHIS_LOAD_DEVICE_PROC(vkWriteMicromapsPropertiesEXT);
#endif
#if defined(VK_EXT_pageable_device_local_memory)
        RACHIS_LOAD_DEVICE_PROC(vkSetDeviceMemoryPriorityEXT);
#endif
#if defined(VK_EXT_pipeline_properties)
        RACHIS_LOAD_DEVICE_PROC(vkGetPipelinePropertiesEXT);
#endif
#if defined(VK_EXT_private_data)
        RACHIS_LOAD_DEVICE_PROC(vkCreatePrivateDataSlotEXT);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyPrivateDataSlotEXT);
        RACHIS_LOAD_DEVICE_PROC(vkGetPrivateDataEXT);
        RACHIS_LOAD_DEVICE_PROC(vkSetPrivateDataEXT);
#endif
#if defined(VK_EXT_sample_locations)
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetSampleLocationsEXT);
#endif
#if defined(VK_EXT_shader_module_identifier)
        RACHIS_LOAD_DEVICE_PROC(vkGetShaderModuleCreateInfoIdentifierEXT);
        RACHIS_LOAD_DEVICE_PROC(vkGetShaderModuleIdentifierEXT);
#endif
#if defined(VK_EXT_shader_object)
        RACHIS_LOAD_DEVICE_PROC(vkCmdBindShadersEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCreateShadersEXT);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyShaderEXT);
        RACHIS_LOAD_DEVICE_PROC(vkGetShaderBinaryDataEXT);
#endif
#if defined(VK_EXT_swapchain_maintenance1)
        RACHIS_LOAD_DEVICE_PROC(vkReleaseSwapchainImagesEXT);
#endif
#if defined(VK_EXT_transform_feedback)
        RACHIS_LOAD_DEVICE_PROC(vkCmdBeginQueryIndexedEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdBeginTransformFeedbackEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdBindTransformFeedbackBuffersEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDrawIndirectByteCountEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdEndQueryIndexedEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdEndTransformFeedbackEXT);
#endif
#if defined(VK_EXT_validation_cache)
        RACHIS_LOAD_DEVICE_PROC(vkCreateValidationCacheEXT);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyValidationCacheEXT);
        RACHIS_LOAD_DEVICE_PROC(vkGetValidationCacheDataEXT);
        RACHIS_LOAD_DEVICE_PROC(vkMergeValidationCachesEXT);
#endif
#if defined(VK_FUCHSIA_buffer_collection)
        RACHIS_LOAD_DEVICE_PROC(vkCreateBufferCollectionFUCHSIA);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyBufferCollectionFUCHSIA);
        RACHIS_LOAD_DEVICE_PROC(vkGetBufferCollectionPropertiesFUCHSIA);
        RACHIS_LOAD_DEVICE_PROC(vkSetBufferCollectionBufferConstraintsFUCHSIA);
        RACHIS_LOAD_DEVICE_PROC(vkSetBufferCollectionImageConstraintsFUCHSIA);
#endif
#if defined(VK_FUCHSIA_external_memory)
        RACHIS_LOAD_DEVICE_PROC(vkGetMemoryZirconHandleFUCHSIA);
        RACHIS_LOAD_DEVICE_PROC(vkGetMemoryZirconHandlePropertiesFUCHSIA);
#endif
#if defined(VK_FUCHSIA_external_semaphore)
        RACHIS_LOAD_DEVICE_PROC(vkGetSemaphoreZirconHandleFUCHSIA);
        RACHIS_LOAD_DEVICE_PROC(vkImportSemaphoreZirconHandleFUCHSIA);
#endif
#if defined(VK_GOOGLE_display_timing)
        RACHIS_LOAD_DEVICE_PROC(vkGetPastPresentationTimingGOOGLE);
        RACHIS_LOAD_DEVICE_PROC(vkGetRefreshCycleDurationGOOGLE);
#endif
#if defined(VK_HUAWEI_cluster_culling_shader)
        RACHIS_LOAD_DEVICE_PROC(vkCmdDrawClusterHUAWEI);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDrawClusterIndirectHUAWEI);
#endif
#if defined(VK_HUAWEI_invocation_mask)
        RACHIS_LOAD_DEVICE_PROC(vkCmdBindInvocationMaskHUAWEI);
#endif
#if defined(VK_HUAWEI_subpass_shading)
        RACHIS_LOAD_DEVICE_PROC(vkCmdSubpassShadingHUAWEI);
        RACHIS_LOAD_DEVICE_PROC(vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI);
#endif
#if defined(VK_INTEL_performance_query)
        RACHIS_LOAD_DEVICE_PROC(vkAcquirePerformanceConfigurationINTEL);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetPerformanceMarkerINTEL);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetPerformanceOverrideINTEL);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetPerformanceStreamMarkerINTEL);
        RACHIS_LOAD_DEVICE_PROC(vkGetPerformanceParameterINTEL);
        RACHIS_LOAD_DEVICE_PROC(vkInitializePerformanceApiINTEL);
        RACHIS_LOAD_DEVICE_PROC(vkQueueSetPerformanceConfigurationINTEL);
        RACHIS_LOAD_DEVICE_PROC(vkReleasePerformanceConfigurationINTEL);
        RACHIS_LOAD_DEVICE_PROC(vkUninitializePerformanceApiINTEL);
#endif
#if defined(VK_KHR_acceleration_structure)
        RACHIS_LOAD_DEVICE_PROC(vkBuildAccelerationStructuresKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdBuildAccelerationStructuresIndirectKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdBuildAccelerationStructuresKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyAccelerationStructureKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyAccelerationStructureToMemoryKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyMemoryToAccelerationStructureKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdWriteAccelerationStructuresPropertiesKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCopyAccelerationStructureKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCopyAccelerationStructureToMemoryKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCopyMemoryToAccelerationStructureKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCreateAccelerationStructureKHR);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyAccelerationStructureKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetAccelerationStructureBuildSizesKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetAccelerationStructureDeviceAddressKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetDeviceAccelerationStructureCompatibilityKHR);
        RACHIS_LOAD_DEVICE_PROC(vkWriteAccelerationStructuresPropertiesKHR);
#endif
#if defined(VK_KHR_buffer_device_address)
        RACHIS_LOAD_DEVICE_PROC(vkGetBufferDeviceAddressKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetBufferOpaqueCaptureAddressKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetDeviceMemoryOpaqueCaptureAddressKHR);
#endif
#if defined(VK_KHR_calibrated_timestamps)
        RACHIS_LOAD_DEVICE_PROC(vkGetCalibratedTimestampsKHR);
#endif
#if defined(VK_KHR_copy_commands2)
        RACHIS_LOAD_DEVICE_PROC(vkCmdBlitImage2KHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyBuffer2KHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyBufferToImage2KHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyImage2KHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyImageToBuffer2KHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdResolveImage2KHR);
#endif
#if defined(VK_KHR_create_renderpass2)
        RACHIS_LOAD_DEVICE_PROC(vkCmdBeginRenderPass2KHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdEndRenderPass2KHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdNextSubpass2KHR);
        RACHIS_LOAD_DEVICE_PROC(vkCreateRenderPass2KHR);
#endif
#if defined(VK_KHR_deferred_host_operations)
        RACHIS_LOAD_DEVICE_PROC(vkCreateDeferredOperationKHR);
        RACHIS_LOAD_DEVICE_PROC(vkDeferredOperationJoinKHR);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyDeferredOperationKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetDeferredOperationMaxConcurrencyKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetDeferredOperationResultKHR);
#endif
#if defined(VK_KHR_descriptor_update_template)
        RACHIS_LOAD_DEVICE_PROC(vkCreateDescriptorUpdateTemplateKHR);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyDescriptorUpdateTemplateKHR);
        RACHIS_LOAD_DEVICE_PROC(vkUpdateDescriptorSetWithTemplateKHR);
#endif
#if defined(VK_KHR_device_group)
        RACHIS_LOAD_DEVICE_PROC(vkCmdDispatchBaseKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDeviceMaskKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetDeviceGroupPeerMemoryFeaturesKHR);
#endif
#if defined(VK_KHR_display_swapchain)
        RACHIS_LOAD_DEVICE_PROC(vkCreateSharedSwapchainsKHR);
#endif
#if defined(VK_KHR_draw_indirect_count)
        RACHIS_LOAD_DEVICE_PROC(vkCmdDrawIndexedIndirectCountKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDrawIndirectCountKHR);
#endif
#if defined(VK_KHR_dynamic_rendering)
        RACHIS_LOAD_DEVICE_PROC(vkCmdBeginRenderingKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdEndRenderingKHR);
#endif
#if defined(VK_KHR_dynamic_rendering_local_read)
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetRenderingAttachmentLocationsKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetRenderingInputAttachmentIndicesKHR);
#endif
#if defined(VK_KHR_external_fence_fd)
        RACHIS_LOAD_DEVICE_PROC(vkGetFenceFdKHR);
        RACHIS_LOAD_DEVICE_PROC(vkImportFenceFdKHR);
#endif
#if defined(VK_KHR_external_fence_win32)
        RACHIS_LOAD_DEVICE_PROC(vkGetFenceWin32HandleKHR);
        RACHIS_LOAD_DEVICE_PROC(vkImportFenceWin32HandleKHR);
#endif
#if defined(VK_KHR_external_memory_fd)
        RACHIS_LOAD_DEVICE_PROC(vkGetMemoryFdKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetMemoryFdPropertiesKHR);
#endif
#if defined(VK_KHR_external_memory_win32)
        RACHIS_LOAD_DEVICE_PROC(vkGetMemoryWin32HandleKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetMemoryWin32HandlePropertiesKHR);
#endif
#if defined(VK_KHR_external_semaphore_fd)
        RACHIS_LOAD_DEVICE_PROC(vkGetSemaphoreFdKHR);
        RACHIS_LOAD_DEVICE_PROC(vkImportSemaphoreFdKHR);
#endif
#if defined(VK_KHR_external_semaphore_win32)
        RACHIS_LOAD_DEVICE_PROC(vkGetSemaphoreWin32HandleKHR);
        RACHIS_LOAD_DEVICE_PROC(vkImportSemaphoreWin32HandleKHR);
#endif
#if defined(VK_KHR_fragment_shading_rate)
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetFragmentShadingRateKHR);
#endif
#if defined(VK_KHR_get_memory_requirements2)
        RACHIS_LOAD_DEVICE_PROC(vkGetImageSparseMemoryRequirements2KHR);
#endif
#if defined(VK_KHR_line_rasterization)
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetLineStippleKHR);
#endif
#if defined(VK_KHR_maintenance1)
        RACHIS_LOAD_DEVICE_PROC(vkTrimCommandPoolKHR);
#endif
#if defined(VK_KHR_maintenance3)
        RACHIS_LOAD_DEVICE_PROC(vkGetDescriptorSetLayoutSupportKHR);
#endif
#if defined(VK_KHR_maintenance4)
        RACHIS_LOAD_DEVICE_PROC(vkGetDeviceImageSparseMemoryRequirementsKHR);
#endif
#if defined(VK_KHR_maintenance5)
        RACHIS_LOAD_DEVICE_PROC(vkCmdBindIndexBuffer2KHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetDeviceImageSubresourceLayoutKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetImageSubresourceLayout2KHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetRenderingAreaGranularityKHR);
#endif
#if defined(VK_KHR_maintenance6)
        RACHIS_LOAD_DEVICE_PROC(vkCmdBindDescriptorSets2KHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdPushConstants2KHR);
#endif
#if defined(VK_KHR_maintenance6) && defined(VK_KHR_push_descriptor)
        RACHIS_LOAD_DEVICE_PROC(vkCmdPushDescriptorSet2KHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdPushDescriptorSetWithTemplate2KHR);
#endif
#if defined(VK_KHR_maintenance6) && defined(VK_EXT_descriptor_buffer)
        RACHIS_LOAD_DEVICE_PROC(vkCmdBindDescriptorBufferEmbeddedSamplers2EXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDescriptorBufferOffsets2EXT);
#endif
#if defined(VK_KHR_map_memory2)
        RACHIS_LOAD_DEVICE_PROC(vkMapMemory2KHR);
        RACHIS_LOAD_DEVICE_PROC(vkUnmapMemory2KHR);
#endif
#if defined(VK_KHR_performance_query)
        RACHIS_LOAD_DEVICE_PROC(vkAcquireProfilingLockKHR);
        RACHIS_LOAD_DEVICE_PROC(vkReleaseProfilingLockKHR);
#endif
#if defined(VK_KHR_pipeline_binary)
        RACHIS_LOAD_DEVICE_PROC(vkCreatePipelineBinariesKHR);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyPipelineBinaryKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetPipelineBinaryDataKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetPipelineKeyKHR);
        RACHIS_LOAD_DEVICE_PROC(vkReleaseCapturedPipelineDataKHR);
#endif
#if defined(VK_KHR_pipeline_executable_properties)
        RACHIS_LOAD_DEVICE_PROC(vkGetPipelineExecutableInternalRepresentationsKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetPipelineExecutablePropertiesKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetPipelineExecutableStatisticsKHR);
#endif
#if defined(VK_KHR_present_wait)
        RACHIS_LOAD_DEVICE_PROC(vkWaitForPresentKHR);
#endif
#if defined(VK_KHR_push_descriptor)
        RACHIS_LOAD_DEVICE_PROC(vkCmdPushDescriptorSetKHR);
#endif
#if defined(VK_KHR_ray_tracing_maintenance1) && defined(VK_KHR_ray_tracing_pipeline)
        RACHIS_LOAD_DEVICE_PROC(vkCmdTraceRaysIndirect2KHR);
#endif
#if defined(VK_KHR_ray_tracing_pipeline)
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetRayTracingPipelineStackSizeKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdTraceRaysIndirectKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdTraceRaysKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCreateRayTracingPipelinesKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetRayTracingCaptureReplayShaderGroupHandlesKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetRayTracingShaderGroupHandlesKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetRayTracingShaderGroupStackSizeKHR);
#endif
#if defined(VK_KHR_sampler_ycbcr_conversion)
        RACHIS_LOAD_DEVICE_PROC(vkCreateSamplerYcbcrConversionKHR);
        RACHIS_LOAD_DEVICE_PROC(vkDestroySamplerYcbcrConversionKHR);
#endif
#if defined(VK_KHR_shared_presentable_image)
        RACHIS_LOAD_DEVICE_PROC(vkGetSwapchainStatusKHR);
#endif
#if defined(VK_KHR_swapchain)
        RACHIS_LOAD_DEVICE_PROC(vkAcquireNextImageKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCreateSwapchainKHR);
        RACHIS_LOAD_DEVICE_PROC(vkDestroySwapchainKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetSwapchainImagesKHR);
        RACHIS_LOAD_DEVICE_PROC(vkQueuePresentKHR);
#endif
#if defined(VK_KHR_synchronization2)
        RACHIS_LOAD_DEVICE_PROC(vkCmdPipelineBarrier2KHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdResetEvent2KHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetEvent2KHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdWaitEvents2KHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdWriteTimestamp2KHR);
        RACHIS_LOAD_DEVICE_PROC(vkQueueSubmit2KHR);
#endif
#if defined(VK_KHR_synchronization2) && defined(VK_AMD_buffer_marker)
        RACHIS_LOAD_DEVICE_PROC(vkCmdWriteBufferMarker2AMD);
#endif
#if defined(VK_KHR_synchronization2) && defined(VK_NV_device_diagnostic_checkpoints)
        RACHIS_LOAD_DEVICE_PROC(vkGetQueueCheckpointData2NV);
#endif
#if defined(VK_KHR_timeline_semaphore)
        RACHIS_LOAD_DEVICE_PROC(vkGetSemaphoreCounterValueKHR);
        RACHIS_LOAD_DEVICE_PROC(vkSignalSemaphoreKHR);
        RACHIS_LOAD_DEVICE_PROC(vkWaitSemaphoresKHR);
#endif
#if defined(VK_KHR_video_decode_queue)
        RACHIS_LOAD_DEVICE_PROC(vkCmdDecodeVideoKHR);
#endif
#if defined(VK_KHR_video_encode_queue)
        RACHIS_LOAD_DEVICE_PROC(vkCmdEncodeVideoKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetEncodedVideoSessionParametersKHR);
#endif
#if defined(VK_KHR_video_queue)
        RACHIS_LOAD_DEVICE_PROC(vkBindVideoSessionMemoryKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdBeginVideoCodingKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdControlVideoCodingKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCmdEndVideoCodingKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCreateVideoSessionKHR);
        RACHIS_LOAD_DEVICE_PROC(vkCreateVideoSessionParametersKHR);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyVideoSessionKHR);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyVideoSessionParametersKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetVideoSessionMemoryRequirementsKHR);
        RACHIS_LOAD_DEVICE_PROC(vkUpdateVideoSessionParametersKHR);
#endif
#if defined(VK_NVX_binary_import)
        RACHIS_LOAD_DEVICE_PROC(vkCmdCuLaunchKernelNVX);
        RACHIS_LOAD_DEVICE_PROC(vkCreateCuFunctionNVX);
        RACHIS_LOAD_DEVICE_PROC(vkCreateCuModuleNVX);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyCuFunctionNVX);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyCuModuleNVX);
#endif
#if defined(VK_NVX_image_view_handle)
        RACHIS_LOAD_DEVICE_PROC(vkGetImageViewAddressNVX);
        RACHIS_LOAD_DEVICE_PROC(vkGetImageViewHandleNVX);
#endif
#if defined(VK_NV_clip_space_w_scaling)
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetViewportWScalingNV);
#endif
#if defined(VK_NV_copy_memory_indirect)
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyMemoryIndirectNV);
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyMemoryToImageIndirectNV);
#endif
#if defined(VK_NV_cuda_kernel_launch)
        RACHIS_LOAD_DEVICE_PROC(vkCmdCudaLaunchKernelNV);
        RACHIS_LOAD_DEVICE_PROC(vkCreateCudaFunctionNV);
        RACHIS_LOAD_DEVICE_PROC(vkCreateCudaModuleNV);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyCudaFunctionNV);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyCudaModuleNV);
        RACHIS_LOAD_DEVICE_PROC(vkGetCudaModuleCacheNV);
#endif
#if defined(VK_NV_device_diagnostic_checkpoints)
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetCheckpointNV);
        RACHIS_LOAD_DEVICE_PROC(vkGetQueueCheckpointDataNV);
#endif
#if defined(VK_NV_device_generated_commands)
        RACHIS_LOAD_DEVICE_PROC(vkCmdBindPipelineShaderGroupNV);
        RACHIS_LOAD_DEVICE_PROC(vkCmdExecuteGeneratedCommandsNV);
        RACHIS_LOAD_DEVICE_PROC(vkCmdPreprocessGeneratedCommandsNV);
        RACHIS_LOAD_DEVICE_PROC(vkCreateIndirectCommandsLayoutNV);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyIndirectCommandsLayoutNV);
        RACHIS_LOAD_DEVICE_PROC(vkGetGeneratedCommandsMemoryRequirementsNV);
#endif
#if defined(VK_NV_device_generated_commands_compute)
        RACHIS_LOAD_DEVICE_PROC(vkCmdUpdatePipelineIndirectBufferNV);
        RACHIS_LOAD_DEVICE_PROC(vkGetPipelineIndirectDeviceAddressNV);
        RACHIS_LOAD_DEVICE_PROC(vkGetPipelineIndirectMemoryRequirementsNV);
#endif
#if defined(VK_NV_external_memory_rdma)
        RACHIS_LOAD_DEVICE_PROC(vkGetMemoryRemoteAddressNV);
#endif
#if defined(VK_NV_external_memory_win32)
        RACHIS_LOAD_DEVICE_PROC(vkGetMemoryWin32HandleNV);
#endif
#if defined(VK_NV_fragment_shading_rate_enums)
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetFragmentShadingRateEnumNV);
#endif
#if defined(VK_NV_low_latency2)
        RACHIS_LOAD_DEVICE_PROC(vkGetLatencyTimingsNV);
        RACHIS_LOAD_DEVICE_PROC(vkLatencySleepNV);
        RACHIS_LOAD_DEVICE_PROC(vkQueueNotifyOutOfBandNV);
        RACHIS_LOAD_DEVICE_PROC(vkSetLatencyMarkerNV);
        RACHIS_LOAD_DEVICE_PROC(vkSetLatencySleepModeNV);
#endif
#if defined(VK_NV_memory_decompression)
        RACHIS_LOAD_DEVICE_PROC(vkCmdDecompressMemoryIndirectCountNV);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDecompressMemoryNV);
#endif
#if defined(VK_NV_mesh_shader)
        RACHIS_LOAD_DEVICE_PROC(vkCmdDrawMeshTasksIndirectCountNV);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDrawMeshTasksIndirectNV);
        RACHIS_LOAD_DEVICE_PROC(vkCmdDrawMeshTasksNV);
#endif
#if defined(VK_NV_optical_flow)
        RACHIS_LOAD_DEVICE_PROC(vkBindOpticalFlowSessionImageNV);
        RACHIS_LOAD_DEVICE_PROC(vkCmdOpticalFlowExecuteNV);
        RACHIS_LOAD_DEVICE_PROC(vkCreateOpticalFlowSessionNV);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyOpticalFlowSessionNV);
#endif
#if defined(VK_NV_ray_tracing)
        RACHIS_LOAD_DEVICE_PROC(vkBindAccelerationStructureMemoryNV);
        RACHIS_LOAD_DEVICE_PROC(vkCmdBuildAccelerationStructureNV);
        RACHIS_LOAD_DEVICE_PROC(vkCmdCopyAccelerationStructureNV);
        RACHIS_LOAD_DEVICE_PROC(vkCmdTraceRaysNV);
        RACHIS_LOAD_DEVICE_PROC(vkCmdWriteAccelerationStructuresPropertiesNV);
        RACHIS_LOAD_DEVICE_PROC(vkCompileDeferredNV);
        RACHIS_LOAD_DEVICE_PROC(vkCreateAccelerationStructureNV);
        RACHIS_LOAD_DEVICE_PROC(vkCreateRayTracingPipelinesNV);
        RACHIS_LOAD_DEVICE_PROC(vkDestroyAccelerationStructureNV);
        RACHIS_LOAD_DEVICE_PROC(vkGetAccelerationStructureHandleNV);
        RACHIS_LOAD_DEVICE_PROC(vkGetAccelerationStructureMemoryRequirementsNV);
        RACHIS_LOAD_DEVICE_PROC(vkGetRayTracingShaderGroupHandlesNV);
#endif
#if defined(VK_NV_scissor_exclusive) && VK_NV_SCISSOR_EXCLUSIVE_SPEC_VERSION >= 2
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetExclusiveScissorEnableNV);
#endif
#if defined(VK_NV_scissor_exclusive)
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetExclusiveScissorNV);
#endif
#if defined(VK_NV_shading_rate_image)
        RACHIS_LOAD_DEVICE_PROC(vkCmdBindShadingRateImageNV);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetCoarseSampleOrderNV);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetViewportShadingRatePaletteNV);
#endif
#if defined(VK_QCOM_tile_properties)
        RACHIS_LOAD_DEVICE_PROC(vkGetDynamicRenderingTilePropertiesQCOM);
        RACHIS_LOAD_DEVICE_PROC(vkGetFramebufferTilePropertiesQCOM);
#endif
#if defined(VK_QNX_external_memory_screen_buffer)
        RACHIS_LOAD_DEVICE_PROC(vkGetScreenBufferPropertiesQNX);
#endif
#if defined(VK_VALVE_descriptor_set_host_mapping)
        RACHIS_LOAD_DEVICE_PROC(vkGetDescriptorSetHostMappingVALVE);
        RACHIS_LOAD_DEVICE_PROC(vkGetDescriptorSetLayoutHostMappingInfoVALVE);
#endif
#if (defined(VK_EXT_depth_clamp_control)) || (defined(VK_EXT_shader_object) && defined(VK_EXT_depth_clamp_control))
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDepthClampRangeEXT);
#endif
#if (defined(VK_EXT_extended_dynamic_state)) || (defined(VK_EXT_shader_object))
        RACHIS_LOAD_DEVICE_PROC(vkCmdBindVertexBuffers2EXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetCullModeEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDepthBoundsTestEnableEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDepthCompareOpEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDepthTestEnableEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDepthWriteEnableEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetFrontFaceEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetPrimitiveTopologyEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetScissorWithCountEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetStencilOpEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetStencilTestEnableEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetViewportWithCountEXT);
#endif
#if (defined(VK_EXT_extended_dynamic_state2)) || (defined(VK_EXT_shader_object))
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDepthBiasEnableEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetLogicOpEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetPatchControlPointsEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetPrimitiveRestartEnableEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetRasterizerDiscardEnableEXT);
#endif
#if (defined(VK_EXT_extended_dynamic_state3)) || (defined(VK_EXT_shader_object))
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetAlphaToCoverageEnableEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetAlphaToOneEnableEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetColorBlendEnableEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetColorBlendEquationEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetColorWriteMaskEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDepthClampEnableEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetLogicOpEnableEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetPolygonModeEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetRasterizationSamplesEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetSampleMaskEXT);
#endif
#if (defined(VK_EXT_extended_dynamic_state3) && (defined(VK_KHR_maintenance2) || defined(VK_VERSION_1_1))) ||          \
    (defined(VK_EXT_shader_object))
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetTessellationDomainOriginEXT);
#endif
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_EXT_transform_feedback)) ||                                 \
    (defined(VK_EXT_shader_object) && defined(VK_EXT_transform_feedback))
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetRasterizationStreamEXT);
#endif
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_EXT_conservative_rasterization)) ||                         \
    (defined(VK_EXT_shader_object) && defined(VK_EXT_conservative_rasterization))
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetConservativeRasterizationModeEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetExtraPrimitiveOverestimationSizeEXT);
#endif
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_EXT_depth_clip_enable)) ||                                  \
    (defined(VK_EXT_shader_object) && defined(VK_EXT_depth_clip_enable))
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDepthClipEnableEXT);
#endif
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_EXT_sample_locations)) ||                                   \
    (defined(VK_EXT_shader_object) && defined(VK_EXT_sample_locations))
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetSampleLocationsEnableEXT);
#endif
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_EXT_blend_operation_advanced)) ||                           \
    (defined(VK_EXT_shader_object) && defined(VK_EXT_blend_operation_advanced))
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetColorBlendAdvancedEXT);
#endif
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_EXT_provoking_vertex)) ||                                   \
    (defined(VK_EXT_shader_object) && defined(VK_EXT_provoking_vertex))
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetProvokingVertexModeEXT);
#endif
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_EXT_line_rasterization)) ||                                 \
    (defined(VK_EXT_shader_object) && defined(VK_EXT_line_rasterization))
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetLineRasterizationModeEXT);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetLineStippleEnableEXT);
#endif
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_EXT_depth_clip_control)) ||                                 \
    (defined(VK_EXT_shader_object) && defined(VK_EXT_depth_clip_control))
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetDepthClipNegativeOneToOneEXT);
#endif
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_clip_space_w_scaling)) ||                                \
    (defined(VK_EXT_shader_object) && defined(VK_NV_clip_space_w_scaling))
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetViewportWScalingEnableNV);
#endif
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_viewport_swizzle)) ||                                    \
    (defined(VK_EXT_shader_object) && defined(VK_NV_viewport_swizzle))
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetViewportSwizzleNV);
#endif
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_fragment_coverage_to_color)) ||                          \
    (defined(VK_EXT_shader_object) && defined(VK_NV_fragment_coverage_to_color))
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetCoverageToColorEnableNV);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetCoverageToColorLocationNV);
#endif
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_framebuffer_mixed_samples)) ||                           \
    (defined(VK_EXT_shader_object) && defined(VK_NV_framebuffer_mixed_samples))
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetCoverageModulationModeNV);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetCoverageModulationTableEnableNV);
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetCoverageModulationTableNV);
#endif
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_shading_rate_image)) ||                                  \
    (defined(VK_EXT_shader_object) && defined(VK_NV_shading_rate_image))
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetShadingRateImageEnableNV);
#endif
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_representative_fragment_test)) ||                        \
    (defined(VK_EXT_shader_object) && defined(VK_NV_representative_fragment_test))
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetRepresentativeFragmentTestEnableNV);
#endif
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_coverage_reduction_mode)) ||                             \
    (defined(VK_EXT_shader_object) && defined(VK_NV_coverage_reduction_mode))
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetCoverageReductionModeNV);
#endif
#if (defined(VK_EXT_host_image_copy)) || (defined(VK_EXT_image_compression_control))
        RACHIS_LOAD_DEVICE_PROC(vkGetImageSubresourceLayout2EXT);
#endif
#if (defined(VK_EXT_shader_object)) || (defined(VK_EXT_vertex_input_dynamic_state))
        RACHIS_LOAD_DEVICE_PROC(vkCmdSetVertexInputEXT);
#endif
#if (defined(VK_KHR_descriptor_update_template) && defined(VK_KHR_push_descriptor)) ||                                 \
    (defined(VK_KHR_push_descriptor) && (defined(VK_VERSION_1_1) || defined(VK_KHR_descriptor_update_template)))
        RACHIS_LOAD_DEVICE_PROC(vkCmdPushDescriptorSetWithTemplateKHR);
#endif
#if (defined(VK_KHR_device_group) && defined(VK_KHR_surface)) || (defined(VK_KHR_swapchain) && defined(VK_VERSION_1_1))
        RACHIS_LOAD_DEVICE_PROC(vkGetDeviceGroupPresentCapabilitiesKHR);
        RACHIS_LOAD_DEVICE_PROC(vkGetDeviceGroupSurfacePresentModesKHR);
#endif
#if (defined(VK_KHR_device_group) && defined(VK_KHR_swapchain)) ||                                                     \
    (defined(VK_KHR_swapchain) && defined(VK_VERSION_1_1))
        RACHIS_LOAD_DEVICE_PROC(vkAcquireNextImage2KHR);
#endif

#undef RACHIS_LOAD_DEVICE_PROC
    }

} // namespace rachis::vulkan
