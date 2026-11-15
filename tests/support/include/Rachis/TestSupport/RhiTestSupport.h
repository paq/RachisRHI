#pragma once

#include "ForceVulkanRequest.h"
#include "Rachis/Device.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace rachis::test
{

    [[nodiscard]] inline ShaderInterfaceDesc shaderInterface(
        std::span<const ShaderBindingRequirement> bindings = {},
        std::span<const ShaderPushConstantRequirement> pushConstants = {},
        std::span<const std::uint32_t> vertexInputLocations = {}) noexcept
    {
        return {
            .bindings = bindings,
            .pushConstants = pushConstants,
            .vertexInputLocations = vertexInputLocations,
        };
    }

    [[nodiscard]] inline ShaderInterfaceDesc computeShaderInterface(
        ShaderWorkgroupSize workgroupSize,
        std::span<const ShaderBindingRequirement> bindings = {},
        std::span<const ShaderPushConstantRequirement> pushConstants = {}) noexcept
    {
        return {
            .bindings = bindings,
            .pushConstants = pushConstants,
            .workgroupSize = workgroupSize,
        };
    }

    [[nodiscard]] inline ShaderInterfaceDesc computeWriteShaderInterface() noexcept
    {
        static constexpr std::array bindings{
            ShaderBindingRequirement{0, 0, ShaderResourceType::storageBuffer, 1},
        };
        static constexpr std::array pushConstants{
            ShaderPushConstantRequirement{0, 2 * sizeof(std::uint32_t)},
        };
        return computeShaderInterface({64, 1, 1}, bindings, pushConstants);
    }

    [[nodiscard]] inline ShaderInterfaceDesc graphicsInputVertexShaderInterface() noexcept
    {
        static constexpr std::array locations{0U};
        return shaderInterface({}, {}, locations);
    }

    [[nodiscard]] inline ShaderInterfaceDesc graphicsPixelShaderInterface() noexcept
    {
        static constexpr std::array pushConstants{
            ShaderPushConstantRequirement{0, 4 * sizeof(float)},
        };
        return shaderInterface({}, pushConstants);
    }

    [[nodiscard]] inline Result<CommandBuffer> acquireCommandBuffer(Device& device,
                                                                    QueueType queueType,
                                                                    CommandPoolFlag flags = CommandPoolFlag::none)
    {
        auto pool = device.createCommandPool({
            .queueType = queueType,
            .maxCommandBuffers = 1,
            .label = "One-shot test command pool",
            .flags = flags,
        });
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }

        auto command = device.acquireCommandBuffer(*pool);
        auto destroyed = device.destroyCommandPool(*pool);
        if (!command)
        {
            return std::unexpected(std::move(command.error()));
        }
        if (!destroyed)
        {
            return std::unexpected(std::move(destroyed.error()));
        }
        return command;
    }

    [[nodiscard]] inline Result<BindingSetHandle> createBindingSet(Device& device, BindingSetDesc desc)
    {
        if (desc.pool.isValid())
        {
            return device.createBindingSet(desc);
        }

        constexpr std::uint32_t resourceCapacityPerType = 4096;
        constexpr std::uint32_t samplerCapacity = 2048;
        constexpr std::uint32_t poolsPerResourceBudget = 64;
        constexpr auto resourceTypeCount = static_cast<std::uint32_t>(bindingTypeCount - 1);
        const auto capabilities = device.capabilities();
        const auto resourceBudget = capabilities.has_value() ? capabilities->limits.maxBindingPoolResourceDescriptors
                                                             : std::numeric_limits<std::uint32_t>::max();
        const auto resourceCapacity =
            std::min(resourceCapacityPerType, resourceBudget / (poolsPerResourceBudget * resourceTypeCount));
        std::array<BindingPoolSize, bindingTypeCount> sizes{};
        for (std::size_t index = 0; index < sizes.size(); ++index)
        {
            const auto type = static_cast<BindingType>(index);
            sizes[index] = BindingPoolSize{
                .type = type,
                .descriptorCount = type == BindingType::sampler ? samplerCapacity : resourceCapacity,
            };
        }
        auto pool = device.createBindingPool({
            .maxBindingSets = 1,
            .sizes = sizes,
            .label = "One-set test binding pool",
        });
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }

        desc.pool = *pool;
        auto set = device.createBindingSet(desc);
        if (!set)
        {
            auto error = std::move(set.error());
            auto destroyed = device.destroyBindingPool(*pool);
            if (!destroyed)
            {
                return std::unexpected(std::move(destroyed.error()));
            }
            return std::unexpected(std::move(error));
        }

        return set;
    }

    [[nodiscard]] inline DeviceDesc vulkanDeviceDesc() noexcept
    {
        return DeviceDesc{
            .backend = Backend::vulkan,
            .validation = ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
            .enableDeviceFault = true,
        };
    }

    [[nodiscard]] inline DeviceDesc d3d12WarpDeviceDesc() noexcept
    {
        return DeviceDesc{
            .backend = Backend::d3d12,
            .adapterPreference = AdapterPreference::warp,
            .validation = ValidationLevel::standard,
            .requireValidation = false,
        };
    }

    enum class DeviceCreateClassification : std::uint8_t
    {
        created,
        backendUnavailable,
        adapterNotFound,
        failed,
    };

    struct DeviceCreateOutcome final
    {
        DeviceCreateClassification classification = DeviceCreateClassification::failed;
        Result<Device> device = std::unexpected(Error{ErrorCode::invalidArgument, 0, "uninitialized device create"});
    };

    [[nodiscard]] inline DeviceCreateClassification classifyDeviceCreateError(ErrorCode code) noexcept
    {
        switch (code)
        {
        case ErrorCode::backendUnavailable:
            return DeviceCreateClassification::backendUnavailable;
        case ErrorCode::adapterNotFound:
            return DeviceCreateClassification::adapterNotFound;
        default:
            return DeviceCreateClassification::failed;
        }
    }

    template<typename T>
    [[nodiscard]] T& requireValue(std::optional<T>& value)
    {
        REQUIRE(value.has_value());
        return *value;
    }

    [[nodiscard]] inline DeviceCapabilities requireCapabilities(const Device& device)
    {
        auto capabilities = device.capabilities();
        REQUIRE(capabilities.has_value());
        return *capabilities;
    }

    [[nodiscard]] inline AdapterInfo requireAdapterInfo(const Device& device)
    {
        auto adapter = device.adapterInfo();
        REQUIRE(adapter.has_value());
        return std::move(*adapter);
    }

    [[nodiscard]] inline bool skipWithoutCapability(bool advertised, const char* name)
    {
        if (!advertised)
        {
            SKIP("This Vulkan device does not advertise " << name << ".");
        }
        return false;
    }

    inline bool referenceDesktopRequires(bool advertised)
    {
#ifdef _WIN32
        REQUIRE(advertised);
        return true;
#else
        return advertised;
#endif
    }

    [[nodiscard]] inline bool skipUnlessReferenceDesktopAdvertises(bool advertised, const char* name)
    {
#ifdef _WIN32
        static_cast<void>(name);
        REQUIRE(advertised);
        return false;
#else
        return skipWithoutCapability(advertised, name);
#endif
    }

    [[nodiscard]] inline bool vulkanHostHasOnlySoftwareAdapters()
    {
        const auto adapters = enumerateAdapters(Backend::vulkan);
        return adapters.has_value() && std::ranges::none_of(*adapters, [](const AdapterInfo& adapter) {
                   return adapter.type != AdapterType::software;
               });
    }

    inline void skipWhenVulkanIsUnavailable(const Error& error)
    {
#ifdef _WIN32
        static_cast<void>(error);
#else
        if (forceVulkanRequested())
        {
            return;
        }
        if (error.code == ErrorCode::backendUnavailable)
        {
#ifdef __APPLE__
            SKIP("The macOS host has no available Vulkan/MoltenVK backend.");
#else
            SKIP("The Linux host has no available Vulkan backend.");
#endif
        }
        if (error.code == ErrorCode::unsupported && vulkanHostHasOnlySoftwareAdapters())
        {
            SKIP("The host offers only software Vulkan implementations, which the backend refuses.");
        }
#endif
    }

    [[nodiscard]] inline bool skipWithoutCreatedDevice(const Result<Device>& created)
    {
        if (created)
        {
            return false;
        }
        INFO(created.error().message);
        skipWhenVulkanIsUnavailable(created.error());
#ifdef _WIN32
        FAIL("A Vulkan device is required for this Windows contract.");
#else
        FAIL("Vulkan device creation failed for a reason other than a missing or software-only Vulkan driver.");
#endif
        return true;
    }

    [[nodiscard]] inline bool skipWithoutVulkanAdapterList(const Result<std::vector<AdapterInfo>>& adapters)
    {
        if (adapters)
        {
            return false;
        }
        INFO(adapters.error().message);
        skipWhenVulkanIsUnavailable(adapters.error());
#ifdef _WIN32
        FAIL("Vulkan adapter enumeration is required for this Windows contract.");
#else
        FAIL("Vulkan adapter enumeration failed for a reason other than a missing Vulkan driver.");
#endif
        return true;
    }

    [[nodiscard]] inline DeviceCreateOutcome tryCreateDevice(const DeviceDesc& desc)
    {
        auto device = createDevice(desc);
        if (device)
        {
            return DeviceCreateOutcome{
                .classification = DeviceCreateClassification::created,
                .device = std::move(device),
            };
        }

        return DeviceCreateOutcome{
            .classification = classifyDeviceCreateError(device.error().code),
            .device = std::move(device),
        };
    }

} // namespace rachis::test
