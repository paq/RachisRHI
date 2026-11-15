#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

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

} // namespace

TEST_CASE("binding layouts and immutable sets retain complete descriptor state", "[size:medium][rhi][binding][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto constants = device.createBuffer(rachis::BufferDesc{
        .size = 512,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Binding constants",
    });
    auto storage = device.createBuffer(rachis::BufferDesc{
        .size = 1024,
        .usage = rachis::BufferUsage::storage,
        .label = "Binding storage",
    });
    auto texture = device.createTexture(rachis::TextureDesc{
        .width = 16,
        .height = 16,
        .mipLevels = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "Binding texture",
    });
    REQUIRE(constants.has_value());
    REQUIRE(storage.has_value());
    REQUIRE(texture.has_value());

    auto sampledView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *texture,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::texture2D,
        .label = "Binding SRV",
    });
    auto storageView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *texture,
        .usage = rachis::TextureViewUsage::storage,
        .dimension = rachis::TextureViewDimension::texture2D,
        .firstMipLevel = 1,
        .label = "Binding UAV",
    });
    auto sampler = device.createSampler(rachis::SamplerDesc{
        .minFilter = rachis::FilterMode::linear,
        .magFilter = rachis::FilterMode::linear,
        .mipFilter = rachis::MipmapFilterMode::linear,
        .label = "Binding sampler",
    });
    REQUIRE(sampledView.has_value());
    REQUIRE(storageView.has_value());
    REQUIRE(sampler.has_value());

    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::vertex, 1},
        rachis::BindingLayoutEntry{2, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::all, 1},
        rachis::BindingLayoutEntry{4, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{6, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel, 2},
        rachis::BindingLayoutEntry{8, rachis::BindingType::storageTexture, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{10, rachis::BindingType::sampler, rachis::ShaderStage::pixel, 2},
    };
    auto layout = device.createBindingLayout(rachis::BindingLayoutDesc{
        .entries = layoutEntries,
        .label = "Complete binding layout",
    });
    REQUIRE(layout.has_value());

    const std::array writes{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*constants, 0, 512}},
        rachis::BindingWrite{2, 0, rachis::BufferBinding{*storage, 0, 256}},
        rachis::BindingWrite{4, 0, rachis::BufferBinding{*storage, 256, 256, 16}},
        rachis::BindingWrite{6, 0, *sampledView},
        rachis::BindingWrite{6, 1, *sampledView},
        rachis::BindingWrite{8, 0, *storageView},
        rachis::BindingWrite{10, 0, *sampler},
        rachis::BindingWrite{10, 1, *sampler},
    };
    auto set = rachis::test::createBindingSet(device,
                                              rachis::BindingSetDesc{
                                                  .layout = *layout,
                                                  .writes = writes,
                                                  .label = "Complete binding set",
                                              });
    REQUIRE(set.has_value());

    const std::array duplicateBindings{
        rachis::BindingLayoutEntry{1, rachis::BindingType::uniformBuffer},
        rachis::BindingLayoutEntry{1, rachis::BindingType::sampler},
    };
    CHECK_FALSE(device
                    .createBindingLayout(rachis::BindingLayoutDesc{
                        .entries = duplicateBindings,
                    })
                    .has_value());
    const std::array overlappingRegisters{
        rachis::BindingLayoutEntry{1, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel, 4},
        rachis::BindingLayoutEntry{3, rachis::BindingType::readOnlyStorageBuffer},
    };
    CHECK_FALSE(device
                    .createBindingLayout(rachis::BindingLayoutDesc{
                        .entries = overlappingRegisters,
                    })
                    .has_value());
    const std::array zeroCount{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampler, rachis::ShaderStage::pixel, 0},
    };
    CHECK_FALSE(device
                    .createBindingLayout(rachis::BindingLayoutDesc{
                        .entries = zeroCount,
                    })
                    .has_value());

    CHECK_FALSE(rachis::test::createBindingSet(
                    device,
                    rachis::BindingSetDesc{
                        .layout = *layout,
                        .writes = std::span<const rachis::BindingWrite>{writes}.first(writes.size() - 1),
                    })
                    .has_value());
    auto duplicateWrites = writes;
    duplicateWrites.back() = duplicateWrites[duplicateWrites.size() - 2];
    CHECK_FALSE(rachis::test::createBindingSet(device,
                                               rachis::BindingSetDesc{
                                                   .layout = *layout,
                                                   .writes = duplicateWrites,
                                               })
                    .has_value());
    auto wrongTypeWrites = writes;
    wrongTypeWrites[0].resource = *sampler;
    CHECK_FALSE(rachis::test::createBindingSet(device,
                                               rachis::BindingSetDesc{
                                                   .layout = *layout,
                                                   .writes = wrongTypeWrites,
                                               })
                    .has_value());
    auto wrongViewWrites = writes;
    wrongViewWrites[3].resource = *storageView;
    CHECK_FALSE(rachis::test::createBindingSet(device,
                                               rachis::BindingSetDesc{
                                                   .layout = *layout,
                                                   .writes = wrongViewWrites,
                                               })
                    .has_value());
    auto misalignedWrites = writes;
    misalignedWrites[0].resource = rachis::BufferBinding{*constants, 1, 256};
    CHECK_FALSE(rachis::test::createBindingSet(device,
                                               rachis::BindingSetDesc{
                                                   .layout = *layout,
                                                   .writes = misalignedWrites,
                                               })
                    .has_value());
    auto misalignedRawWrites = writes;
    misalignedRawWrites[1].resource = rachis::BufferBinding{*storage, 4, 256};
    CHECK_FALSE(rachis::test::createBindingSet(device,
                                               rachis::BindingSetDesc{
                                                   .layout = *layout,
                                                   .writes = misalignedRawWrites,
                                               })
                    .has_value());

    auto emptyLayout = device.createBindingLayout({});
    REQUIRE(emptyLayout.has_value());
    auto missingPool = device.createBindingSet(rachis::BindingSetDesc{.layout = *emptyLayout});
    REQUIRE_FALSE(missingPool.has_value());
    CHECK(missingPool.error().code == rachis::ErrorCode::invalidArgument);
    auto emptySet = rachis::test::createBindingSet(device,
                                                   rachis::BindingSetDesc{
                                                       .layout = *emptyLayout,
                                                   });
    REQUIRE(emptySet.has_value());
    REQUIRE(device.destroyBindingLayout(*emptyLayout).has_value());
    REQUIRE(device.destroyBindingSet(*emptySet).has_value());

    constexpr std::uint32_t pagedSetCount = 300;
    const std::array pagedPoolSizes{
        rachis::BindingPoolSize{rachis::BindingType::uniformBuffer, pagedSetCount},
        rachis::BindingPoolSize{rachis::BindingType::readOnlyStorageBuffer, pagedSetCount},
        rachis::BindingPoolSize{rachis::BindingType::storageBuffer, pagedSetCount},
        rachis::BindingPoolSize{rachis::BindingType::sampledTexture, 2 * pagedSetCount},
        rachis::BindingPoolSize{rachis::BindingType::storageTexture, pagedSetCount},
        rachis::BindingPoolSize{rachis::BindingType::sampler, 2 * pagedSetCount},
    };
    auto pagedPool = device.createBindingPool(rachis::BindingPoolDesc{
        .maxBindingSets = pagedSetCount,
        .sizes = pagedPoolSizes,
        .label = "Paged binding pool",
    });
    REQUIRE(pagedPool.has_value());
    std::vector<rachis::BindingSetHandle> pagedSets;
    pagedSets.reserve(pagedSetCount);
    for (std::uint32_t index = 0; index < pagedSetCount; ++index)
    {
        auto paged = rachis::test::createBindingSet(device,
                                                    rachis::BindingSetDesc{
                                                        .layout = *layout,
                                                        .writes = writes,
                                                        .pool = *pagedPool,
                                                    });
        REQUIRE(paged.has_value());
        pagedSets.push_back(*paged);
    }
    for (std::size_t index = 0; index < pagedSets.size(); index += 2)
    {
        REQUIRE(device.destroyBindingSet(pagedSets[index]).has_value());
    }
    for (std::uint32_t index = 0; index < pagedSetCount / 2; ++index)
    {
        auto recycled = rachis::test::createBindingSet(device,
                                                       rachis::BindingSetDesc{
                                                           .layout = *layout,
                                                           .writes = writes,
                                                           .pool = *pagedPool,
                                                       });
        REQUIRE(recycled.has_value());
        REQUIRE(device.destroyBindingSet(*recycled).has_value());
    }
    for (std::size_t index = 1; index < pagedSets.size(); index += 2)
    {
        REQUIRE(device.destroyBindingSet(pagedSets[index]).has_value());
    }
    REQUIRE(device.destroyBindingPool(*pagedPool).has_value());

    const auto staleLayout = *layout;
    REQUIRE(device.destroyBuffer(*constants).has_value());
    REQUIRE(device.destroyBuffer(*storage).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyTextureView(*sampledView).has_value());
    REQUIRE(device.destroyTextureView(*storageView).has_value());
    REQUIRE(device.destroySampler(*sampler).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    CHECK_FALSE(rachis::test::createBindingSet(device,
                                               rachis::BindingSetDesc{
                                                   .layout = staleLayout,
                                                   .writes = writes,
                                               })
                    .has_value());

    const auto staleSet = *set;
    REQUIRE(device.destroyBindingSet(*set).has_value());
    CHECK_FALSE(device.destroyBindingSet(staleSet).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
