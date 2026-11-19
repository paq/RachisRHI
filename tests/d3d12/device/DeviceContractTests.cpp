#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

namespace
{

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    [[nodiscard]] rachis::Result<rachis::Device> createAdapterDevice(const rachis::AdapterInfo& adapter,
                                                                     rachis::AdapterPreference preference)
    {
        return rachis::createDevice(rachis::DeviceDesc{
            .adapterPreference = preference,
            .adapterId = adapter.id,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = true,
            .diagnosticCallback = &printDiagnostic,
        });
    }

    void verifyLogicOperationCapabilities(rachis::Device& device)
    {
        CHECK(rachis::test::requireCapabilities(device).logicOperations);
        const auto integerColor = device.queryFormatProperties(rachis::Format::r32Uint);
        CHECK(hasAll(integerColor.features,
                     rachis::FormatFeature::colorAttachment | rachis::FormatFeature::logicOperation));
        CHECK_FALSE(hasAny(integerColor.features, rachis::FormatFeature::blendable));
    }

    [[nodiscard]] bool nativeDescriptorHeapInt64Atomics(rachis::AdapterId adapterId)
    {
        Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
        REQUIRE(SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))));
        const LUID luid{
            .LowPart = static_cast<DWORD>(adapterId.value & 0xFFFF'FFFFU),
            .HighPart = static_cast<LONG>(adapterId.value >> 32U),
        };
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        REQUIRE(SUCCEEDED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter))));
        Microsoft::WRL::ComPtr<ID3D12Device> device;
        REQUIRE(SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))));
        D3D12_FEATURE_DATA_D3D12_OPTIONS11 options{};
        return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS11, &options, sizeof(options))) &&
               options.AtomicInt64OnDescriptorHeapResourceSupported != FALSE;
    }

    void verifyShaderNumericCapabilities(const rachis::DeviceCapabilities& capabilities,
                                         bool descriptorHeapInt64Atomics)
    {
        CHECK(capabilities.shaderFloat16 == capabilities.shaderInt16);
        if (capabilities.shaderFloat16 || capabilities.shaderInt16)
        {
            CHECK(capabilities.shaderModel >= rachis::ApiVersion{6, 2});
        }
        if (capabilities.shaderInt64)
        {
            CHECK(capabilities.shaderModel >= rachis::ApiVersion{6, 0});
        }
        if (capabilities.shaderInt64AtomicGroupShared || capabilities.shaderInt64AtomicTypedResource)
        {
            CHECK(capabilities.shaderInt64);
            CHECK(capabilities.shaderModel >= rachis::ApiVersion{6, 6});
        }
        CHECK(capabilities.shaderInt64AtomicStorageBuffer ==
              (capabilities.shaderInt64 && capabilities.shaderModel >= rachis::ApiVersion{6, 6} &&
               descriptorHeapInt64Atomics));
    }

    void verifyFramebufferLayerAndMultiviewLimits(const rachis::DeviceCapabilities& capabilities)
    {
        const auto& limits = capabilities.limits;
        CHECK(limits.maxFramebufferLayers <= limits.maxTextureArrayLayers);
        CHECK(capabilities.multiviewRendering == (capabilities.maxMultiviewViewCount != 0));
        if (capabilities.multiviewRendering)
        {
            CHECK(capabilities.shaderModel >= rachis::ApiVersion{6, 1});
            CHECK(limits.maxFramebufferLayers >= capabilities.maxMultiviewViewCount);
        }
    }

    void verifyDeviceLimits(const rachis::DeviceCapabilities& capabilities)
    {
        verifyFramebufferLayerAndMultiviewLimits(capabilities);
    }

    void verifySubgroupCapabilities(const rachis::DeviceCapabilities& capabilities)
    {
        CHECK(capabilities.subgroupOperations == (capabilities.subgroup.minSize != 0));
        if (capabilities.subgroupOperations)
        {
            CHECK(capabilities.shaderModel >= rachis::ApiVersion{6, 0});
            CHECK(capabilities.subgroup.minSize <= capabilities.subgroup.maxSize);
        }
        else
        {
            CHECK(capabilities.subgroup.maxSize == 0);
        }
        const bool requiredComputeSize =
            capabilities.subgroupOperations && capabilities.shaderModel >= rachis::ApiVersion{6, 6};
        CHECK(capabilities.subgroup.requiredSizeStages ==
              (requiredComputeSize ? rachis::ShaderStage::compute : rachis::ShaderStage::none));
    }

    void verifyRasterizationCapabilities(const rachis::DeviceCapabilities& capabilities)
    {
        CHECK(capabilities.conservativeRasterization);
        if (capabilities.fragmentBarycentricCoordinates)
        {
            CHECK(capabilities.shaderModel >= rachis::ApiVersion{6, 1});
        }
    }

    void verifyFeatureCapabilities(rachis::Device& device, const rachis::DeviceCapabilities& capabilities)
    {
        verifyShaderNumericCapabilities(capabilities,
                                        nativeDescriptorHeapInt64Atomics(rachis::test::requireAdapterInfo(device).id));
        verifySubgroupCapabilities(capabilities);
        verifyLogicOperationCapabilities(device);
        verifyRasterizationCapabilities(capabilities);
    }

    void verifyMemoryBudgets(rachis::Device& device, const rachis::DeviceCapabilities& capabilities)
    {
        REQUIRE(capabilities.memoryBudgets);
        auto budgets = device.queryMemoryBudgets();
        if (!budgets)
        {
            INFO(budgets.error().message);
        }
        REQUIRE(budgets.has_value());
        REQUIRE(budgets->size() == 2);
        CHECK((*budgets)[0].heapIndex == 0);
        CHECK((*budgets)[0].properties == rachis::MemoryHeapProperty::deviceLocal);
        CHECK(rachis::hasAny((*budgets)[0].features, rachis::MemoryBudgetFeature::reservation));
        CHECK(rachis::hasAny((*budgets)[0].features, rachis::MemoryBudgetFeature::allocationStatistics));
        CHECK((*budgets)[0].blockBytes >= (*budgets)[0].allocationBytes);
        CHECK((*budgets)[0].budgetBytes > 0);
        CHECK((*budgets)[1].heapIndex == 1);
        CHECK((*budgets)[1].properties == rachis::MemoryHeapProperty::none);
        CHECK(rachis::hasAny((*budgets)[1].features, rachis::MemoryBudgetFeature::reservation));
        CHECK(rachis::hasAny((*budgets)[1].features, rachis::MemoryBudgetFeature::allocationStatistics));
        CHECK((*budgets)[1].blockBytes >= (*budgets)[1].allocationBytes);

        auto invalidNode = device.queryMemoryBudgets(capabilities.nodeCount);
        REQUIRE_FALSE(invalidNode.has_value());
        CHECK(invalidNode.error().code == rachis::ErrorCode::invalidArgument);
    }

    void verifyUnknownFormatProperties(rachis::Device& device)
    {
        const auto unknown = device.queryFormatProperties(rachis::Format::unknown);
        CHECK(unknown.features == rachis::FormatFeature::none);
        CHECK(unknown.sampleCounts == rachis::SampleCountMask::none);
        CHECK(unknown.colorResolveModes == rachis::ResolveModeMask::none);
        CHECK(unknown.depthResolveModes == rachis::ResolveModeMask::none);
        CHECK(unknown.stencilResolveModes == rachis::ResolveModeMask::none);
    }

    void verifyColorFormatProperties(rachis::Device& device)
    {
        const auto color = device.queryFormatProperties(rachis::Format::rgba8Unorm);
        constexpr auto requiredColorFeatures =
            rachis::FormatFeature::texture2D | rachis::FormatFeature::sampled | rachis::FormatFeature::filterable |
            rachis::FormatFeature::storageRead | rachis::FormatFeature::storageWrite |
            rachis::FormatFeature::colorAttachment | rachis::FormatFeature::blendable |
            rachis::FormatFeature::copySource | rachis::FormatFeature::copyDestination;
        CHECK(hasAll(color.features, requiredColorFeatures));
        CHECK_FALSE(hasAny(color.features, rachis::FormatFeature::depthStencilAttachment));
        CHECK(supportsSampleCount(color.sampleCounts, 1));
        CHECK(supportsResolveMode(color.colorResolveModes, rachis::ResolveMode::average));
        CHECK_FALSE(supportsResolveMode(color.depthResolveModes, rachis::ResolveMode::minimum));
    }

    void verifyDepthFormatProperties(rachis::Device& device)
    {
        const auto depth = device.queryFormatProperties(rachis::Format::d32Float);
        constexpr auto requiredDepthFeatures =
            rachis::FormatFeature::texture1D | rachis::FormatFeature::texture2D | rachis::FormatFeature::cube |
            rachis::FormatFeature::sampled | rachis::FormatFeature::depthStencilAttachment |
            rachis::FormatFeature::copySource | rachis::FormatFeature::copyDestination;
        CHECK(hasAll(depth.features, requiredDepthFeatures));
        CHECK_FALSE(hasAny(depth.features, rachis::FormatFeature::texture3D));
        CHECK_FALSE(hasAny(depth.features, rachis::FormatFeature::colorAttachment));
        CHECK_FALSE(hasAny(depth.features, rachis::FormatFeature::storageWrite));
        CHECK(supportsSampleCount(depth.sampleCounts, 1));
        CHECK(supportsResolveMode(depth.depthResolveModes, rachis::ResolveMode::minimum));
        CHECK(supportsResolveMode(depth.depthResolveModes, rachis::ResolveMode::maximum));
        CHECK_FALSE(supportsResolveMode(depth.depthResolveModes, rachis::ResolveMode::average));
    }

    void verifyPackedDepthStencilFormatProperties(rachis::Device& device)
    {
        const auto packedDepthStencil = device.queryFormatProperties(rachis::Format::d24UnormS8Uint);
        CHECK(hasAll(packedDepthStencil.features,
                     rachis::FormatFeature::texture2D | rachis::FormatFeature::depthStencilAttachment |
                         rachis::FormatFeature::multisampleResolve));
        CHECK_FALSE(hasAny(packedDepthStencil.features, rachis::FormatFeature::texture3D));
        CHECK(supportsResolveMode(packedDepthStencil.depthResolveModes, rachis::ResolveMode::minimum));
        CHECK(supportsResolveMode(packedDepthStencil.depthResolveModes, rachis::ResolveMode::maximum));
        CHECK_FALSE(supportsResolveMode(packedDepthStencil.depthResolveModes, rachis::ResolveMode::average));
        CHECK(supportsResolveMode(packedDepthStencil.stencilResolveModes, rachis::ResolveMode::minimum));
        CHECK(supportsResolveMode(packedDepthStencil.stencilResolveModes, rachis::ResolveMode::maximum));
        CHECK_FALSE(supportsResolveMode(packedDepthStencil.stencilResolveModes, rachis::ResolveMode::average));
    }

    void verifyCompressedFormatProperties(rachis::Device& device)
    {
        const auto compressed = device.queryFormatProperties(rachis::Format::bc7RgbaUnorm);
        constexpr auto requiredCompressedFeatures =
            rachis::FormatFeature::texture2D | rachis::FormatFeature::sampled | rachis::FormatFeature::filterable |
            rachis::FormatFeature::copySource | rachis::FormatFeature::copyDestination;
        CHECK(hasAll(compressed.features, requiredCompressedFeatures));
        CHECK_FALSE(hasAny(compressed.features,
                           rachis::FormatFeature::storageWrite | rachis::FormatFeature::colorAttachment |
                               rachis::FormatFeature::depthStencilAttachment));
        CHECK(supportsSampleCount(compressed.sampleCounts, 1));
    }

    void verifyFormatProperties(rachis::Device& device)
    {
        verifyUnknownFormatProperties(device);
        verifyColorFormatProperties(device);
        verifyDepthFormatProperties(device);
        verifyPackedDepthStencilFormatProperties(device);
        verifyCompressedFormatProperties(device);
    }

    void verifyReportedFormatsCreateTextures(rachis::Device& device)
    {
        const std::array textureDescs{
            rachis::TextureDesc{
                .width = 4,
                .height = 4,
                .format = rachis::Format::rgba8Unorm,
                .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                         rachis::TextureUsage::sampled | rachis::TextureUsage::storage |
                         rachis::TextureUsage::renderTarget,
                .label = "Capability color",
            },
            rachis::TextureDesc{
                .width = 4,
                .height = 4,
                .format = rachis::Format::d32Float,
                .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::sampled |
                         rachis::TextureUsage::depthStencil,
                .label = "Capability depth",
            },
            rachis::TextureDesc{
                .width = 4,
                .height = 4,
                .format = rachis::Format::bc7RgbaUnorm,
                .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                         rachis::TextureUsage::sampled,
                .label = "Capability BC7",
            },
        };
        for (const auto& desc : textureDescs)
        {
            auto texture = device.createTexture(desc);
            if (!texture)
            {
                INFO(texture.error().message);
            }
            REQUIRE(texture.has_value());
            REQUIRE(device.destroyTexture(*texture).has_value());
        }
    }

    void verifyCapabilityContract(rachis::Device& device)
    {
        const auto capabilities = *device.capabilities();
        verifyDeviceLimits(capabilities);
        verifyFeatureCapabilities(device, capabilities);
        verifyMemoryBudgets(device, capabilities);
        verifyFormatProperties(device);
        verifyReportedFormatsCreateTextures(device);
    }
} // namespace

TEST_CASE("headless WARP device exposes portable capabilities", "[size:medium][rhi][device][warp]")
{
    rachis::Device invalidDevice;
    rachis::CommandBuffer invalidCommand;
    CHECK_FALSE(invalidDevice.adapterInfo().has_value());
    CHECK_FALSE(invalidDevice.capabilities().has_value());
    const auto invalidReset = invalidDevice.resetQueryPool(invalidCommand, {}, 0, 1);
    REQUIRE_FALSE(invalidReset.has_value());
    CHECK(invalidReset.error().code == rachis::ErrorCode::invalidState);

    constexpr rachis::ApiVersion minimumFeatureLevel{12, 0};
    constexpr rachis::ApiVersion minimumShaderModel{6, 0};
    auto adapters = rachis::enumerateAdapters();
    if (!adapters)
    {
        INFO(adapters.error().message);
    }
    REQUIRE(adapters.has_value());

    const auto selected = std::ranges::find_if(
        *adapters, [](const auto& adapter) { return adapter.type == rachis::AdapterType::software; });
    REQUIRE(selected != adapters->end());

    auto result = createAdapterDevice(*selected, rachis::AdapterPreference::warp);
    if (!result)
    {
        INFO(result.error().message);
    }
    REQUIRE(result.has_value());

    auto device = std::move(*result);
    REQUIRE(device.isValid());
    CHECK(rachis::test::requireAdapterInfo(device).id == selected->id);
    CHECK(rachis::test::requireAdapterInfo(device).type == rachis::AdapterType::software);
    CHECK(rachis::test::requireCapabilities(device).featureLevel >= minimumFeatureLevel);
    CHECK(rachis::test::requireCapabilities(device).shaderModel >= minimumShaderModel);
    verifyCapabilityContract(device);

    const auto counts = device.diagnosticCounts();
    CHECK(counts.errors == 0);
    CHECK(counts.corruptions == 0);

    const rachis::Device moved = std::move(device);

    CHECK_FALSE(device.isValid());

    CHECK_FALSE(device.adapterInfo().has_value());

    CHECK_FALSE(device.capabilities().has_value());
    CHECK(moved.isValid());
}

TEST_CASE("headless hardware device satisfies the same contract", "[size:medium][rhi][device][hardware]")
{
    constexpr rachis::ApiVersion minimumFeatureLevel{12, 0};
    auto adapters = rachis::enumerateAdapters();
    if (!adapters)
    {
        INFO(adapters.error().message);
    }
    REQUIRE(adapters.has_value());

    const auto selected = std::ranges::find_if(
        *adapters, [](const auto& adapter) { return adapter.type != rachis::AdapterType::software; });
    if (selected == adapters->end())
    {
        SKIP("No D3D12 hardware adapter supporting feature level 12.0 is available.");
    }

    auto result = createAdapterDevice(*selected, rachis::AdapterPreference::highPerformance);
    if (!result)
    {
        INFO(result.error().message);
    }
    REQUIRE(result.has_value());
    CHECK(rachis::test::requireAdapterInfo(*result).id == selected->id);
    CHECK(rachis::test::requireCapabilities(*result).featureLevel >= minimumFeatureLevel);
    verifyCapabilityContract(*result);
    CHECK(result->diagnosticCounts().errors == 0);
    CHECK(result->diagnosticCounts().corruptions == 0);
}

TEST_CASE("D3D12 reports the WARP adapter as software in enumeration and on its device",
          "[size:medium][rhi][device][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .backend = rachis::Backend::d3d12,
        .adapterPreference = rachis::AdapterPreference::warp,
    });
    if (!created)
    {
        UNSCOPED_INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    const auto warp = rachis::test::requireAdapterInfo(*created);
    INFO("WARP reports unified memory: " << rachis::test::requireCapabilities(*created).unifiedMemory);
    CHECK(warp.type == rachis::AdapterType::software);

    auto adapters = rachis::enumerateAdapters(rachis::Backend::d3d12);
    if (!adapters)
    {
        UNSCOPED_INFO(adapters.error().message);
    }
    REQUIRE(adapters.has_value());
    const auto enumerated =
        std::ranges::find_if(*adapters, [&](const rachis::AdapterInfo& adapter) { return adapter.id == warp.id; });
    REQUIRE(enumerated != adapters->end());
    CHECK(enumerated->type == rachis::AdapterType::software);
    CHECK(std::ranges::count(*adapters, rachis::AdapterType::software, &rachis::AdapterInfo::type) == 1);
}

TEST_CASE("D3D12 opens the enumerated WARP adapter by ID under the hardware preferences",
          "[size:medium][rhi][device][warp]")
{
    auto adapters = rachis::enumerateAdapters(rachis::Backend::d3d12);
    if (!adapters)
    {
        UNSCOPED_INFO(adapters.error().message);
    }
    REQUIRE(adapters.has_value());
    const auto warp = std::ranges::find(*adapters, rachis::AdapterType::software, &rachis::AdapterInfo::type);
    REQUIRE(warp != adapters->end());

    for (const auto preference :
         std::array{rachis::AdapterPreference::highPerformance, rachis::AdapterPreference::minimumPower})
    {
        INFO("adapterPreference " << static_cast<int>(preference));
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::d3d12,
            .adapterPreference = preference,
            .adapterId = warp->id,
        });
        if (!created)
        {
            UNSCOPED_INFO(created.error().message);
        }
        REQUIRE(created.has_value());
        const auto info = rachis::test::requireAdapterInfo(*created);
        CHECK(info.id == warp->id);
        CHECK(info.type == rachis::AdapterType::software);
    }
}

TEST_CASE("D3D12 rejects unknown adapter IDs and hardware adapter IDs under the WARP preference",
          "[size:medium][rhi][device][warp]")
{
    auto adapters = rachis::enumerateAdapters(rachis::Backend::d3d12);
    if (!adapters)
    {
        UNSCOPED_INFO(adapters.error().message);
    }
    REQUIRE(adapters.has_value());

    constexpr rachis::AdapterId unknown{.value = 0xdeadbeefdeadbeefULL};
    REQUIRE(std::ranges::find(*adapters, unknown, &rachis::AdapterInfo::id) == adapters->end());
    for (const auto preference : std::array{rachis::AdapterPreference::highPerformance,
                                            rachis::AdapterPreference::minimumPower,
                                            rachis::AdapterPreference::warp})
    {
        INFO("adapterPreference " << static_cast<int>(preference));
        auto missing = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::d3d12,
            .adapterPreference = preference,
            .adapterId = unknown,
        });
        REQUIRE_FALSE(missing.has_value());
        CHECK(missing.error().code == rachis::ErrorCode::adapterNotFound);
    }

    for (const auto& adapter : *adapters)
    {
        if (adapter.type == rachis::AdapterType::software)
        {
            continue;
        }
        INFO(adapter.name);
        auto rejected = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::d3d12,
            .adapterPreference = rachis::AdapterPreference::warp,
            .adapterId = adapter.id,
        });
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::adapterNotFound);
    }
}

TEST_CASE("D3D12 hardware adapter types follow unified memory in enumeration and on the device",
          "[size:medium][requires:HardwareD3D12][rhi][device][hardware]")
{
    auto adapters = rachis::enumerateAdapters(rachis::Backend::d3d12);
    if (!adapters)
    {
        UNSCOPED_INFO(adapters.error().message);
    }
    REQUIRE(adapters.has_value());

    std::size_t hardwareAdapters = 0;
    for (const auto& adapter : *adapters)
    {
        if (adapter.type == rachis::AdapterType::software)
        {
            continue;
        }
        ++hardwareAdapters;
        INFO(adapter.name);
        CHECK((adapter.type == rachis::AdapterType::integrated || adapter.type == rachis::AdapterType::discrete));

        auto device = createAdapterDevice(adapter, rachis::AdapterPreference::highPerformance);
        if (!device)
        {
            UNSCOPED_INFO(device.error().message);
        }
        REQUIRE(device.has_value());
        const auto info = rachis::test::requireAdapterInfo(*device);
        const auto expected = rachis::test::requireCapabilities(*device).unifiedMemory ? rachis::AdapterType::integrated
                                                                                       : rachis::AdapterType::discrete;
        CHECK(info.id == adapter.id);
        CHECK(info.type == expected);
        CHECK(adapter.type == info.type);
    }
    if (hardwareAdapters == 0)
    {
        SKIP("No D3D12 hardware adapter supporting feature level 12.0 is available.");
    }
}
