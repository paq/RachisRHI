#include "Metal/MetalTestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string_view>
#include <vector>

#if RACHIS_ENABLE_METAL

namespace
{

    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] std::vector<std::byte> loadPnmsBytes(std::string_view metallibName)
    {
        const auto path =
            std::filesystem::path(RACHIS_METALLIB_DIR) / std::filesystem::path(metallibName).replace_extension(".pnms");
        INFO("pnms=" << path.string());
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        REQUIRE(stream);
        const auto end = stream.tellg();
        REQUIRE(end > 0);
        std::vector<std::byte> bytes(static_cast<std::size_t>(end));
        stream.seekg(0, std::ios::beg);
        stream.read(reinterpret_cast<char*>(bytes.data()), end);
        REQUIRE(stream);
        return bytes;
    }

    template<typename T>
    [[nodiscard]] std::span<const std::byte> asByteSpan(const T& value) noexcept
    {
        return {reinterpret_cast<const std::byte*>(&value), sizeof(T)};
    }

    [[nodiscard]] std::uint64_t alignUpU64(std::uint64_t value, std::uint64_t alignment) noexcept
    {
        if (alignment <= 1)
        {
            return value;
        }
        return ((value + alignment - 1U) / alignment) * alignment;
    }

    struct CompactUniforms24Host
    {
        std::uint32_t w0 = 0;
        std::uint32_t w1 = 0;
        std::uint32_t w2 = 0;
        std::uint32_t w3 = 0;
        std::uint32_t t0 = 0;
        std::uint32_t t1 = 0;
    };
    static_assert(sizeof(CompactUniforms24Host) == 24);

    struct CompactUniforms16Host
    {
        std::array<std::uint32_t, 4> words{};
    };
    static_assert(sizeof(CompactUniforms16Host) == 16);

    struct RemapVertexHost
    {
        std::array<float, 2> position{};
        std::uint32_t vertexTag = 0;
    };
    static_assert(sizeof(RemapVertexHost) == 12);

    struct UploadedQuadTexture final
    {
        rachis::TextureHandle texture{};
        rachis::TextureViewHandle view{};
        rachis::BufferHandle staging{};
    };

    [[nodiscard]] UploadedQuadTexture create2x2RgbaTexture(rachis::Device& device,
                                                           const std::array<std::array<std::uint8_t, 4>, 4>& texels,
                                                           rachis::PipelineStage consumerStages,
                                                           std::string_view label)
    {
        auto texture = device.createTexture({
            .width = 2,
            .height = 2,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::copyDestination,
            .label = label,
        });
        REQUIRE(texture);
        auto view = device.createTextureView({
            .texture = *texture,
            .usage = rachis::TextureViewUsage::sampled,
            .dimension = rachis::TextureViewDimension::texture2D,
        });
        REQUIRE(view);

        const rachis::BufferTextureCopyRegion queryRegion{
            .extent = {.width = 2, .height = 2, .depth = 1},
        };
        const auto layout = device.queryTextureTransferLayout(*texture, queryRegion);
        REQUIRE(layout);

        auto staging = device.createBuffer({
            .size = layout->requiredBufferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = label,
        });
        REQUIRE(staging);

        std::vector<std::byte> uploadBytes(static_cast<std::size_t>(layout->requiredBufferSize), std::byte{0});
        std::memcpy(uploadBytes.data(), texels[0].data(), 4);
        std::memcpy(uploadBytes.data() + 4, texels[1].data(), 4);
        std::memcpy(uploadBytes.data() + layout->bytesPerRow, texels[2].data(), 4);
        std::memcpy(uploadBytes.data() + layout->bytesPerRow + 4, texels[3].data(), 4);
        REQUIRE(device.writeBuffer(*staging, 0, uploadBytes).has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *texture,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                    })
                    .has_value());
        const rachis::BufferTextureCopyRegion copyRegion{
            .bytesPerRow = layout->bytesPerRow,
            .rowsPerImage = layout->rowsPerImage,
            .extent = {.width = 2, .height = 2, .depth = 1},
        };
        REQUIRE(device.copyBufferToTexture(*command, *staging, *texture, std::array{copyRegion}).has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *texture,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = consumerStages,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::shaderResource,
                                        .layoutBefore = rachis::TextureLayout::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::shaderResource,
                                    })
                    .has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted).has_value());

        return UploadedQuadTexture{
            .texture = *texture,
            .view = *view,
            .staging = *staging,
        };
    }

    [[nodiscard]] bool nearByte(std::byte actual, std::uint8_t expected, std::uint8_t slack = 1) noexcept
    {
        const auto value = std::to_integer<std::uint8_t>(actual);
        const auto lower = expected < slack ? 0U : static_cast<std::uint32_t>(expected) - slack;
        const auto upper = (std::min)(static_cast<std::uint32_t>(expected) + slack, 255U);
        return value >= lower && value <= upper;
    }

    struct MultiWorkloadFixture final
    {
        rachis::BindingLayoutHandle computeSet0Layout{};
        rachis::BindingLayoutHandle computeSet1Layout{};
        rachis::PipelineLayoutHandle computePipeLayout{};
        rachis::PipelineHandle computePipeline{};

        rachis::BindingLayoutHandle graphicsSet0Layout{};
        rachis::BindingLayoutHandle graphicsSet1Layout{};
        rachis::PipelineLayoutHandle graphicsPipeLayout{};
        rachis::PipelineHandle graphicsPipeline{};

        rachis::BufferHandle set0UboBuf{};
        rachis::BufferHandle set1UboBuf{};
        rachis::BufferHandle set0InBuf{};
        rachis::BufferHandle computeOutBuf{};
        rachis::BufferHandle computeReadback{};

        rachis::BufferHandle vboA{};
        rachis::BufferHandle vboB{};
        rachis::TextureHandle renderTarget{};
        rachis::TextureViewHandle renderTargetView{};
        rachis::TextureTransferLayout renderTransfer{};
        rachis::BufferHandle renderReadback{};

        UploadedQuadTexture tex0{};
        UploadedQuadTexture tex1A{};
        UploadedQuadTexture tex1B{};
        rachis::SamplerHandle repeatSampler{};
        rachis::SamplerHandle clampSampler{};

        rachis::BindingPoolHandle bindingPool{};
        rachis::BindingSetHandle computeSet0{};
        rachis::BindingSetHandle computeSet1A{};
        rachis::BindingSetHandle computeSet1B{};
        rachis::BindingSetHandle graphicsSet0{};
        rachis::BindingSetHandle graphicsSet1A{};
        rachis::BindingSetHandle graphicsSet1B{};

        CompactUniforms24Host set0Ubo{};
        std::array<CompactUniforms24Host, 4> set1UboVariants{};
        std::uint32_t set0InWord = 913U;
        std::uint32_t uboStride = 0;
        std::uint32_t ssboStride = 0;
        std::uint32_t packedTex0 = 0;
        std::uint32_t packedTex1A = 0;
        std::uint32_t packedTex1B = 0;
    };

    void createComputeSetAndPipelineLayouts(rachis::Device& device, MultiWorkloadFixture& fx)
    {
        auto s0 = device.createBindingLayout({
            .entries =
                std::array{
                    rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
                    rachis::BindingLayoutEntry{1, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, 1},
                    rachis::BindingLayoutEntry{2, rachis::BindingType::sampler, rachis::ShaderStage::compute, 1},
                    rachis::BindingLayoutEntry{
                        3, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute, 1},
                },
        });
        auto s1 = device.createBindingLayout({
            .entries =
                std::array{
                    rachis::BindingLayoutEntry{
                        0, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 1},
                    rachis::BindingLayoutEntry{1, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, 1},
                    rachis::BindingLayoutEntry{2, rachis::BindingType::sampler, rachis::ShaderStage::compute, 1},
                    rachis::BindingLayoutEntry{
                        3, rachis::BindingType::dynamicStorageBuffer, rachis::ShaderStage::compute, 1},
                },
        });
        REQUIRE(s0);
        REQUIRE(s1);
        fx.computeSet0Layout = *s0;
        fx.computeSet1Layout = *s1;

        auto pl = device.createPipelineLayout({
            .bindingLayouts = std::array{fx.computeSet0Layout, fx.computeSet1Layout},
            .pushConstantRanges = std::array{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}},
        });
        REQUIRE(pl);
        fx.computePipeLayout = *pl;
    }

    void createMultiSetComputePipeline(rachis::Device& device, MultiWorkloadFixture& fx)
    {
        const auto csPnms = loadPnmsBytes("MetalRemapContract.multi.cs.metallib");
        const std::array csReq{
            rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
            rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::sampledTexture, 1},
            rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::sampler, 1},
            rachis::ShaderBindingRequirement{0, 3, rachis::ShaderResourceType::readOnlyStorageBuffer, 1},
            rachis::ShaderBindingRequirement{1, 0, rachis::ShaderResourceType::uniformBuffer, 1},
            rachis::ShaderBindingRequirement{1, 1, rachis::ShaderResourceType::sampledTexture, 1},
            rachis::ShaderBindingRequirement{1, 2, rachis::ShaderResourceType::sampler, 1},
            rachis::ShaderBindingRequirement{1, 3, rachis::ShaderResourceType::storageBuffer, 1},
        };
        const std::array pushReq{rachis::ShaderPushConstantRequirement{0, 16}};
        auto cs = device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = csPnms,
            .entryPoint = "multiSetCompute",
            .shaderInterface = {.bindings = csReq,
                                .pushConstants = pushReq,
                                .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1}},
        });
        REQUIRE(cs);

        createComputeSetAndPipelineLayouts(device, fx);
        auto pipe = device.createComputePipeline({.layout = fx.computePipeLayout, .computeShader = *cs});
        REQUIRE(pipe);
        fx.computePipeline = *pipe;
    }

    void createGraphicsSetAndPipelineLayouts(rachis::Device& device, MultiWorkloadFixture& fx)
    {
        const auto allGfx = rachis::ShaderStage::vertex | rachis::ShaderStage::pixel;
        auto s0 = device.createBindingLayout({
            .entries =
                std::array{
                    rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, allGfx, 1},
                    rachis::BindingLayoutEntry{1, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel, 1},
                    rachis::BindingLayoutEntry{2, rachis::BindingType::sampler, rachis::ShaderStage::pixel, 1},
                },
        });
        auto s1 = device.createBindingLayout({
            .entries =
                std::array{
                    rachis::BindingLayoutEntry{0, rachis::BindingType::dynamicUniformBuffer, allGfx, 1},
                    rachis::BindingLayoutEntry{1, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel, 1},
                    rachis::BindingLayoutEntry{2, rachis::BindingType::sampler, rachis::ShaderStage::pixel, 1},
                },
        });
        REQUIRE(s0);
        REQUIRE(s1);
        fx.graphicsSet0Layout = *s0;
        fx.graphicsSet1Layout = *s1;

        auto pl = device.createPipelineLayout({
            .bindingLayouts = std::array{fx.graphicsSet0Layout, fx.graphicsSet1Layout},
            .pushConstantRanges = std::array{rachis::PushConstantRange{allGfx, 0, 16}},
        });
        REQUIRE(pl);
        fx.graphicsPipeLayout = *pl;
    }

    [[nodiscard]] rachis::PipelineHandle createRemapVertexPipeline(rachis::Device& device,
                                                                   rachis::PipelineLayoutHandle layout,
                                                                   rachis::ShaderModuleHandle vertexShader,
                                                                   rachis::ShaderModuleHandle pixelShader)
    {
        const std::array vertexAttrs{
            rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .offset = 0, .shaderLocation = 0},
            rachis::VertexAttribute{.format = rachis::VertexFormat::uint32, .offset = 8, .shaderLocation = 1},
        };
        const std::array vertexBuffers{
            rachis::VertexBufferLayout{
                .stride = sizeof(RemapVertexHost),
                .stepMode = rachis::VertexStepMode::vertex,
                .attributes = vertexAttrs,
            },
        };
        const std::array colorTargets{
            rachis::ColorTargetState{.format = rachis::Format::rgba32Uint},
        };
        auto pipe = device.createGraphicsPipeline({
            .layout = layout,
            .vertexShader = vertexShader,
            .pixelShader = pixelShader,
            .vertexBuffers = vertexBuffers,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
        });
        REQUIRE(pipe);
        return *pipe;
    }

    void createMultiSetGraphicsPipeline(rachis::Device& device, MultiWorkloadFixture& fx)
    {
        const auto vsPnms = loadPnmsBytes("MetalRemapContract.multi.vs.metallib");
        const auto psPnms = loadPnmsBytes("MetalRemapContract.multi.ps.metallib");
        const std::array vsReq{
            rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
            rachis::ShaderBindingRequirement{1, 0, rachis::ShaderResourceType::uniformBuffer, 1},
        };
        const std::array psReq{
            rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
            rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::sampledTexture, 1},
            rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::sampler, 1},
            rachis::ShaderBindingRequirement{1, 0, rachis::ShaderResourceType::uniformBuffer, 1},
            rachis::ShaderBindingRequirement{1, 1, rachis::ShaderResourceType::sampledTexture, 1},
            rachis::ShaderBindingRequirement{1, 2, rachis::ShaderResourceType::sampler, 1},
        };
        const std::array pushReq{rachis::ShaderPushConstantRequirement{0, 16}};
        const std::array vsLocations{0U, 1U};
        auto vs = device.createShaderModule({
            .stage = rachis::ShaderStage::vertex,
            .bytecode = vsPnms,
            .entryPoint = "multiSetVertex",
            .shaderInterface = {.bindings = vsReq, .pushConstants = pushReq, .vertexInputLocations = vsLocations},
        });
        auto ps = device.createShaderModule({
            .stage = rachis::ShaderStage::pixel,
            .bytecode = psPnms,
            .entryPoint = "multiSetPixel",
            .shaderInterface = {.bindings = psReq, .pushConstants = pushReq},
        });
        REQUIRE(vs);
        REQUIRE(ps);

        createGraphicsSetAndPipelineLayouts(device, fx);
        fx.graphicsPipeline = createRemapVertexPipeline(device, fx.graphicsPipeLayout, *vs, *ps);
    }

    void createUniformAndStorageBuffers(rachis::Device& device, MultiWorkloadFixture& fx, std::uint32_t maxComputeSlots)
    {
        auto set0UboBuf = device.createBuffer({
            .size = fx.uboStride * 2U,
            .usage = rachis::BufferUsage::constant,
            .memoryType = rachis::MemoryType::upload,
        });
        auto set1UboBuf = device.createBuffer({
            .size = static_cast<std::uint64_t>(fx.uboStride) * fx.set1UboVariants.size(),
            .usage = rachis::BufferUsage::constant,
            .memoryType = rachis::MemoryType::upload,
        });
        auto set0InBuf = device.createBuffer({
            .size = 16,
            .usage = rachis::BufferUsage::storage,
            .memoryType = rachis::MemoryType::upload,
        });
        const std::uint64_t computeOutBytes = static_cast<std::uint64_t>(fx.ssboStride) * maxComputeSlots;
        auto computeOutBuf = device.createBuffer({
            .size = computeOutBytes,
            .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        });
        auto computeReadback = device.createBuffer({
            .size = computeOutBytes,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(set0UboBuf);
        REQUIRE(set1UboBuf);
        REQUIRE(set0InBuf);
        REQUIRE(computeOutBuf);
        REQUIRE(computeReadback);
        fx.set0UboBuf = *set0UboBuf;
        fx.set1UboBuf = *set1UboBuf;
        fx.set0InBuf = *set0InBuf;
        fx.computeOutBuf = *computeOutBuf;
        fx.computeReadback = *computeReadback;
    }

    void writeUniformAndInputData(rachis::Device& device, MultiWorkloadFixture& fx)
    {
        fx.set0Ubo = CompactUniforms24Host{
            .w0 = 1100U,
            .w1 = 1200U,
            .w2 = 1300U,
            .w3 = 17U,
            .t0 = 1500U,
            .t1 = 0x5A3CU,
        };
        REQUIRE(device.writeBuffer(fx.set0UboBuf, 0, asByteSpan(fx.set0Ubo)).has_value());
        for (std::size_t v = 0; v < fx.set1UboVariants.size(); ++v)
        {
            const auto idx = static_cast<std::uint32_t>(v);
            fx.set1UboVariants[v] = CompactUniforms24Host{
                .w0 = 2000U + idx * 333U,
                .w1 = 2100U + idx * 111U,
                .w2 = 2200U + idx * 77U,
                .w3 = 29U + idx * 5U,
                .t0 = 0x1020U + idx * 0x0305U,
                .t1 = 3500U + idx * 409U,
            };
            REQUIRE(device
                        .writeBuffer(fx.set1UboBuf,
                                     static_cast<std::uint64_t>(fx.uboStride) * v,
                                     asByteSpan(fx.set1UboVariants[v]))
                        .has_value());
        }
        REQUIRE(device.writeBuffer(fx.set0InBuf, 0, asByteSpan(fx.set0InWord)).has_value());
    }

    void createTaggedTriangleVertexBuffers(rachis::Device& device, MultiWorkloadFixture& fx, std::uint32_t maxDrawCount)
    {
        std::vector<RemapVertexHost> vertsA(static_cast<std::size_t>(maxDrawCount) * 3U);
        std::vector<RemapVertexHost> vertsB(static_cast<std::size_t>(maxDrawCount) * 3U);
        for (std::uint32_t d = 0; d < maxDrawCount; ++d)
        {
            const std::array<std::array<float, 2>, 3> tri{{{-1.0F, -1.0F}, {3.0F, -1.0F}, {-1.0F, 3.0F}}};
            for (std::size_t k = 0; k < 3; ++k)
            {
                vertsA[static_cast<std::size_t>(d) * 3U + k] = {
                    .position = tri[k],
                    .vertexTag = 10000U + d,
                };
                vertsB[static_cast<std::size_t>(d) * 3U + k] = {
                    .position = tri[k],
                    .vertexTag = 50000U + d,
                };
            }
        }
        auto vboA = device.createBuffer({
            .size = vertsA.size() * sizeof(RemapVertexHost),
            .usage = rachis::BufferUsage::vertex,
            .memoryType = rachis::MemoryType::upload,
        });
        auto vboB = device.createBuffer({
            .size = vertsB.size() * sizeof(RemapVertexHost),
            .usage = rachis::BufferUsage::vertex,
            .memoryType = rachis::MemoryType::upload,
        });
        REQUIRE(vboA);
        REQUIRE(vboB);
        fx.vboA = *vboA;
        fx.vboB = *vboB;
        REQUIRE(device
                    .writeBuffer(fx.vboA,
                                 0,
                                 std::span<const std::byte>{reinterpret_cast<const std::byte*>(vertsA.data()),
                                                            vertsA.size() * sizeof(RemapVertexHost)})
                    .has_value());
        REQUIRE(device
                    .writeBuffer(fx.vboB,
                                 0,
                                 std::span<const std::byte>{reinterpret_cast<const std::byte*>(vertsB.data()),
                                                            vertsB.size() * sizeof(RemapVertexHost)})
                    .has_value());
    }

    void createSetTextures(rachis::Device& device, MultiWorkloadFixture& fx)
    {
        const auto consumerStages = rachis::PipelineStage::computeShader | rachis::PipelineStage::pixelShader;
        fx.tex0 = create2x2RgbaTexture(device,
                                       {{{21, 43, 65, 87}, {90, 91, 92, 93}, {94, 95, 96, 97}, {98, 99, 100, 101}}},
                                       consumerStages,
                                       "set0 tex");
        fx.tex1A = create2x2RgbaTexture(
            device, {{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 145, 205}}}, consumerStages, "set1A tex");
        fx.tex1B = create2x2RgbaTexture(
            device, {{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 179, 231}}}, consumerStages, "set1B tex");
        fx.packedTex0 = 21U | (43U << 8U);
        fx.packedTex1A = 145U | (205U << 8U);
        fx.packedTex1B = 179U | (231U << 8U);
    }

    void createRepeatAndClampSamplers(rachis::Device& device, MultiWorkloadFixture& fx)
    {
        auto repeatSampler = device.createSampler({
            .minFilter = rachis::FilterMode::nearest,
            .magFilter = rachis::FilterMode::nearest,
            .addressU = rachis::AddressMode::repeat,
            .addressV = rachis::AddressMode::repeat,
        });
        auto clampSampler = device.createSampler({
            .minFilter = rachis::FilterMode::nearest,
            .magFilter = rachis::FilterMode::nearest,
            .addressU = rachis::AddressMode::clampToEdge,
            .addressV = rachis::AddressMode::clampToEdge,
        });
        REQUIRE(repeatSampler);
        REQUIRE(clampSampler);
        fx.repeatSampler = *repeatSampler;
        fx.clampSampler = *clampSampler;
    }

    void createRenderTargetWithReadback(rachis::Device& device, MultiWorkloadFixture& fx)
    {
        auto rt = device.createTexture({
            .width = 16,
            .height = 16,
            .format = rachis::Format::rgba32Uint,
            .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
        });
        REQUIRE(rt);
        fx.renderTarget = *rt;
        auto rtView = device.createTextureView({
            .texture = fx.renderTarget,
            .usage = rachis::TextureViewUsage::renderTarget,
            .dimension = rachis::TextureViewDimension::texture2D,
        });
        REQUIRE(rtView);
        fx.renderTargetView = *rtView;

        const rachis::BufferTextureCopyRegion rtQuery{.extent = {.width = 16, .height = 16, .depth = 1}};
        auto rtTransfer = device.queryTextureTransferLayout(fx.renderTarget, rtQuery);
        REQUIRE(rtTransfer);
        fx.renderTransfer = *rtTransfer;
        auto renderReadback = device.createBuffer({
            .size = fx.renderTransfer.requiredBufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(renderReadback);
        fx.renderReadback = *renderReadback;
    }

    void createSharedBindingPool(rachis::Device& device, MultiWorkloadFixture& fx)
    {
        auto pool = device.createBindingPool({
            .maxBindingSets = 8,
            .sizes =
                std::array{
                    rachis::BindingPoolSize{rachis::BindingType::uniformBuffer, 4},
                    rachis::BindingPoolSize{rachis::BindingType::dynamicUniformBuffer, 8},
                    rachis::BindingPoolSize{rachis::BindingType::readOnlyStorageBuffer, 4},
                    rachis::BindingPoolSize{rachis::BindingType::dynamicStorageBuffer, 4},
                    rachis::BindingPoolSize{rachis::BindingType::sampledTexture, 12},
                    rachis::BindingPoolSize{rachis::BindingType::sampler, 12},
                },
        });
        REQUIRE(pool);
        fx.bindingPool = *pool;
    }

    [[nodiscard]] rachis::Result<rachis::BindingSetHandle> createComputeSet1(rachis::Device& device,
                                                                             const MultiWorkloadFixture& fx,
                                                                             const UploadedQuadTexture& texture)
    {
        return device.createBindingSet({
            .layout = fx.computeSet1Layout,
            .writes =
                std::array{
                    rachis::BindingWrite{0, 0, rachis::BufferBinding{fx.set1UboBuf, 0, sizeof(CompactUniforms24Host)}},
                    rachis::BindingWrite{1, 0, texture.view},
                    rachis::BindingWrite{2, 0, fx.clampSampler},
                    rachis::BindingWrite{3, 0, rachis::BufferBinding{fx.computeOutBuf, 0, 32}},
                },
            .pool = fx.bindingPool,
        });
    }

    [[nodiscard]] rachis::Result<rachis::BindingSetHandle> createGraphicsSet1(rachis::Device& device,
                                                                              const MultiWorkloadFixture& fx,
                                                                              const UploadedQuadTexture& texture)
    {
        return device.createBindingSet({
            .layout = fx.graphicsSet1Layout,
            .writes =
                std::array{
                    rachis::BindingWrite{0, 0, rachis::BufferBinding{fx.set1UboBuf, 0, sizeof(CompactUniforms24Host)}},
                    rachis::BindingWrite{1, 0, texture.view},
                    rachis::BindingWrite{2, 0, fx.clampSampler},
                },
            .pool = fx.bindingPool,
        });
    }

    void createWorkloadBindingSets(rachis::Device& device, MultiWorkloadFixture& fx)
    {
        auto cs0 = device.createBindingSet({
            .layout = fx.computeSet0Layout,
            .writes =
                std::array{
                    rachis::BindingWrite{0, 0, rachis::BufferBinding{fx.set0UboBuf, 0, sizeof(CompactUniforms24Host)}},
                    rachis::BindingWrite{1, 0, fx.tex0.view},
                    rachis::BindingWrite{2, 0, fx.repeatSampler},
                    rachis::BindingWrite{3, 0, rachis::BufferBinding{fx.set0InBuf, 0, 16}},
                },
            .pool = fx.bindingPool,
        });
        auto cs1A = createComputeSet1(device, fx, fx.tex1A);
        auto cs1B = createComputeSet1(device, fx, fx.tex1B);
        auto gs0 = device.createBindingSet({
            .layout = fx.graphicsSet0Layout,
            .writes =
                std::array{
                    rachis::BindingWrite{0, 0, rachis::BufferBinding{fx.set0UboBuf, 0, sizeof(CompactUniforms24Host)}},
                    rachis::BindingWrite{1, 0, fx.tex0.view},
                    rachis::BindingWrite{2, 0, fx.repeatSampler},
                },
            .pool = fx.bindingPool,
        });
        auto gs1A = createGraphicsSet1(device, fx, fx.tex1A);
        auto gs1B = createGraphicsSet1(device, fx, fx.tex1B);
        REQUIRE(cs0);
        REQUIRE(cs1A);
        REQUIRE(cs1B);
        REQUIRE(gs0);
        REQUIRE(gs1A);
        REQUIRE(gs1B);
        fx.computeSet0 = *cs0;
        fx.computeSet1A = *cs1A;
        fx.computeSet1B = *cs1B;
        fx.graphicsSet0 = *gs0;
        fx.graphicsSet1A = *gs1A;
        fx.graphicsSet1B = *gs1B;
    }

    [[nodiscard]] MultiWorkloadFixture buildMultiWorkloadFixture(rachis::Device& device,
                                                                 std::uint32_t maxComputeSlots,
                                                                 std::uint32_t maxDrawCount)
    {
        MultiWorkloadFixture fx{};
        const auto caps = device.capabilities();
        REQUIRE(caps.has_value());
        fx.uboStride = static_cast<std::uint32_t>(
            alignUpU64(sizeof(CompactUniforms24Host),
                       (std::max<std::uint64_t>)(caps->limits.minUniformBufferOffsetAlignment, 16)));
        fx.ssboStride = static_cast<std::uint32_t>(alignUpU64(8U * sizeof(std::uint32_t), 32));

        createMultiSetComputePipeline(device, fx);
        createMultiSetGraphicsPipeline(device, fx);
        createUniformAndStorageBuffers(device, fx, maxComputeSlots);
        writeUniformAndInputData(device, fx);
        createTaggedTriangleVertexBuffers(device, fx, maxDrawCount);
        createSetTextures(device, fx);
        createRepeatAndClampSamplers(device, fx);
        createRenderTargetWithReadback(device, fx);
        createSharedBindingPool(device, fx);
        createWorkloadBindingSets(device, fx);

        return fx;
    }

} // namespace

TEST_CASE("Metal high-draw recording batches residency commit, deduplicates allocations, and suballocates multi-page "
          "push constants",
          "[size:large][rhi][metal][push][binding][residency]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    constexpr std::uint32_t totalComputeDispatches = 240;
    constexpr std::uint32_t totalGraphicsDraws = 240;
    auto fx = buildMultiWorkloadFixture(device, totalComputeDispatches, totalGraphicsDraws);

    auto cmdPool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .flags = rachis::CommandPoolFlag::allowIndividualReset,
    });
    REQUIRE(cmdPool);

    auto cmd = device.acquireCommandBuffer(*cmdPool);
    REQUIRE(cmd);

    struct ExpectedComputeSlot final
    {
        std::array<std::uint32_t, 4> push{};
        std::size_t uboVariant = 0;
        bool useSetB = false;
    };
    std::vector<ExpectedComputeSlot> expectedCompute;
    expectedCompute.reserve(totalComputeDispatches);

    REQUIRE(device.bindComputePipeline(*cmd, fx.computePipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*cmd, 0, fx.computeSet0).has_value());

    for (std::uint32_t i = 0; i < totalComputeDispatches; ++i)
    {
        const bool unchanged = (i > 0U) && ((i % 10U) == 9U);
        if (unchanged)
        {
            REQUIRE(device
                        .resourceBarriers(*cmd,
                                          {
                                              .globalBarriers =
                                                  std::array{
                                                      rachis::GlobalBarrier{
                                                          .stagesBefore = rachis::PipelineStage::computeShader,
                                                          .stagesAfter = rachis::PipelineStage::computeShader,
                                                          .accessBefore = rachis::Access::unorderedAccess,
                                                          .accessAfter = rachis::Access::unorderedAccess,
                                                      },
                                                  },
                                          })
                        .has_value());
            REQUIRE(device.dispatch(*cmd, 1, 1, 1).has_value());
            continue;
        }

        const auto slotIndex = static_cast<std::uint32_t>(expectedCompute.size());
        const std::size_t uboVariant = slotIndex % fx.set1UboVariants.size();
        const bool useSetB = (slotIndex & 1U) != 0U;
        const std::array<std::uint32_t, 4> push{
            100U + slotIndex * 7U,
            200U + slotIndex * 11U,
            300U + slotIndex * 13U,
            400U + slotIndex * 17U,
        };
        const std::array<std::uint32_t, 2> dynOffsets{
            static_cast<std::uint32_t>(uboVariant) * fx.uboStride,
            slotIndex * fx.ssboStride,
        };
        const auto chosenSet1 = useSetB ? fx.computeSet1B : fx.computeSet1A;
        if ((slotIndex % 32U) == 0U)
        {
            REQUIRE(device
                        .bindComputeBindingSets(
                            *cmd, 0, std::array{fx.computeSet0, chosenSet1}, std::span<const std::uint32_t>{dynOffsets})
                        .has_value());
        }
        else
        {
            REQUIRE(device.bindComputeBindingSet(*cmd, 1, chosenSet1, dynOffsets).has_value());
        }
        REQUIRE(device.setComputePushConstants(*cmd, 0, asByteSpan(push)).has_value());
        REQUIRE(device.dispatch(*cmd, 1, 1, 1).has_value());
        expectedCompute.push_back(ExpectedComputeSlot{
            .push = push,
            .uboVariant = uboVariant,
            .useSetB = useSetB,
        });
    }

    REQUIRE(device
                .resourceBarriers(*cmd,
                                  {
                                      .bufferBarriers =
                                          std::array{
                                              rachis::BufferBarrier{
                                                  .buffer = fx.computeOutBuf,
                                                  .stagesBefore = rachis::PipelineStage::computeShader,
                                                  .stagesAfter = rachis::PipelineStage::copy,
                                                  .accessBefore = rachis::Access::unorderedAccess,
                                                  .accessAfter = rachis::Access::copySource,
                                              },
                                          },
                                      .textureBarriers =
                                          std::array{
                                              rachis::TextureBarrier{
                                                  .texture = fx.renderTarget,
                                                  .stagesAfter = rachis::PipelineStage::renderTarget,
                                                  .accessAfter = rachis::Access::renderTarget,
                                                  .layoutAfter = rachis::TextureLayout::renderTarget,
                                              },
                                          },
                                  })
                .has_value());

    const auto allGfx = rachis::ShaderStage::vertex | rachis::ShaderStage::pixel;
    const std::array colorAttachments{
        rachis::ColorAttachment{
            .view = fx.renderTargetView,
            .loadOperation = rachis::LoadOperation::clear,
        },
    };
    REQUIRE(device.beginRendering(*cmd, {.colorAttachments = colorAttachments}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*cmd, fx.graphicsPipeline).has_value());
    REQUIRE(device.setViewport(*cmd, {.width = 16.0F, .height = 16.0F}).has_value());
    REQUIRE(device.bindGraphicsBindingSet(*cmd, 0, fx.graphicsSet0).has_value());

    struct ExpectedDrawPixel final
    {
        std::array<std::uint32_t, 4> push{};
        std::size_t uboVariant = 0;
        bool useSetB = false;
        bool useVboB = false;
    };
    std::vector<ExpectedDrawPixel> expectedPixels(totalGraphicsDraws);
    std::uint32_t uniqueGraphicsPushCount = 0;
    bool currentVboB = false;
    REQUIRE(device.bindVertexBuffer(*cmd, 0, fx.vboA, 0).has_value());

    for (std::uint32_t d = 0; d < totalGraphicsDraws; ++d)
    {
        const std::int32_t col = static_cast<std::int32_t>(d % 16U);
        const std::int32_t row = static_cast<std::int32_t>(d / 16U);
        REQUIRE(device.setScissor(*cmd, {.x = col, .y = row, .width = 1, .height = 1}).has_value());

        const bool unchanged = (d % 20U) >= 17U;
        if (!unchanged)
        {
            const bool wantVboB = ((d / 16U) & 1U) != 0U;
            if (wantVboB != currentVboB)
            {
                REQUIRE(device.bindVertexBuffer(*cmd, 0, wantVboB ? fx.vboB : fx.vboA, 0).has_value());
                currentVboB = wantVboB;
            }
            const std::size_t uboVariant = uniqueGraphicsPushCount % fx.set1UboVariants.size();
            const bool useSetB = (uniqueGraphicsPushCount % 3U) == 1U;
            const std::array<std::uint32_t, 4> push{
                500U + uniqueGraphicsPushCount * 19U,
                600U + uniqueGraphicsPushCount * 23U,
                700U + uniqueGraphicsPushCount * 29U,
                0xA50000U + uniqueGraphicsPushCount * 31U,
            };
            const std::array<std::uint32_t, 1> dynOffset{
                static_cast<std::uint32_t>(uboVariant) * fx.uboStride,
            };
            const auto chosenSet1 = useSetB ? fx.graphicsSet1B : fx.graphicsSet1A;
            REQUIRE(device.bindGraphicsBindingSet(*cmd, 1, chosenSet1, dynOffset).has_value());
            REQUIRE(device.setGraphicsPushConstants(*cmd, allGfx, 0, asByteSpan(push)).has_value());
            expectedPixels[d] = ExpectedDrawPixel{
                .push = push,
                .uboVariant = uboVariant,
                .useSetB = useSetB,
                .useVboB = currentVboB,
            };
            ++uniqueGraphicsPushCount;
        }
        else
        {
            expectedPixels[d] = expectedPixels[d - 1U];
        }

        REQUIRE(device.draw(*cmd, 3, 1, d * 3U, 0).has_value());
    }
    REQUIRE(device.endRendering(*cmd).has_value());

    REQUIRE(device
                .textureBarrier(*cmd,
                                {
                                    .texture = fx.renderTarget,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    const std::uint64_t usedComputeBytes = static_cast<std::uint64_t>(expectedCompute.size()) * fx.ssboStride;
    REQUIRE(device
                .copyBuffer(*cmd,
                            {
                                .source = fx.computeOutBuf,
                                .destination = fx.computeReadback,
                                .size = usedComputeBytes,
                            })
                .has_value());
    const rachis::BufferTextureCopyRegion rtCopy{
        .bytesPerRow = fx.renderTransfer.bytesPerRow,
        .rowsPerImage = fx.renderTransfer.rowsPerImage,
        .extent = {.width = 16, .height = 16, .depth = 1},
    };
    REQUIRE(device.copyTextureToBuffer(*cmd, fx.renderTarget, fx.renderReadback, std::array{rtCopy}).has_value());

    auto submitted = device.submit(std::move(*cmd));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    std::vector<std::byte> rawCompute(static_cast<std::size_t>(usedComputeBytes));
    REQUIRE(device.readBuffer(fx.computeReadback, 0, rawCompute).has_value());
    for (std::size_t slot = 0; slot < expectedCompute.size(); ++slot)
    {
        std::array<std::uint32_t, 8> actual{};
        std::memcpy(actual.data(), rawCompute.data() + slot * fx.ssboStride, sizeof(actual));
        const auto& exp = expectedCompute[slot];
        const auto& set1Ubo = fx.set1UboVariants[exp.uboVariant];
        const std::uint32_t packed1 = exp.useSetB ? fx.packedTex1B : fx.packedTex1A;
        CAPTURE(slot, exp.uboVariant, exp.useSetB);
        CHECK(actual[0] == (exp.push[0] + fx.set0Ubo.w0));
        CHECK(actual[1] == (exp.push[1] + set1Ubo.w0));
        CHECK(actual[2] == (fx.set0Ubo.t0 + fx.set0InWord));
        CHECK(actual[3] == (set1Ubo.t1 + exp.push[2]));
        CHECK(actual[4] == fx.packedTex0);
        CHECK(actual[5] == packed1);
        CHECK(actual[6] == (fx.set0Ubo.t1 ^ set1Ubo.t0));
        CHECK(actual[7] == (exp.push[3] + fx.set0Ubo.w3 + set1Ubo.w3));
    }

    std::vector<std::byte> rawPixels(static_cast<std::size_t>(fx.renderTransfer.requiredBufferSize));
    REQUIRE(device.readBuffer(fx.renderReadback, 0, rawPixels).has_value());
    for (std::uint32_t d = 0; d < totalGraphicsDraws; ++d)
    {
        const std::size_t col = d % 16U;
        const std::size_t row = d / 16U;
        std::array<std::uint32_t, 4> pixel{};
        std::memcpy(
            pixel.data(), rawPixels.data() + row * fx.renderTransfer.bytesPerRow + col * sizeof(pixel), sizeof(pixel));
        const auto& exp = expectedPixels[d];
        const auto& set1Ubo = fx.set1UboVariants[exp.uboVariant];
        const std::uint32_t vertexTag = (exp.useVboB ? 50000U : 10000U) + d;
        const std::uint32_t packed1 = exp.useSetB ? fx.packedTex1B : fx.packedTex1A;
        CAPTURE(d, col, row, exp.uboVariant, exp.useSetB, exp.useVboB);
        CHECK(pixel[0] == (vertexTag + exp.push[0] + fx.set0Ubo.w0 + set1Ubo.w0));
        CHECK(pixel[1] == (fx.set0Ubo.w0 + fx.set0Ubo.t0 + fx.set0Ubo.t1));
        CHECK(pixel[2] == (set1Ubo.w0 + set1Ubo.t1 + set1Ubo.t0 + exp.push[2]));
        CHECK(pixel[3] == (((fx.packedTex0 << 16U) | packed1) ^ exp.push[3]));
    }

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal command-buffer and command-pool resets recycle argument tables and push-constant pages without churn",
          "[size:large][rhi][metal][push][binding][residency]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    constexpr std::uint32_t computeDispatchesPerIter = 96;
    constexpr std::uint32_t graphicsDrawsPerIter = 96;
    auto fx = buildMultiWorkloadFixture(device, computeDispatchesPerIter, graphicsDrawsPerIter);

    auto cmdPool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .flags = rachis::CommandPoolFlag::allowIndividualReset,
    });
    REQUIRE(cmdPool);

    const auto allGfx = rachis::ShaderStage::vertex | rachis::ShaderStage::pixel;
    auto recordIteration = [&](rachis::CommandBuffer& cmd, std::uint32_t seed) {
        REQUIRE(device.bindComputePipeline(cmd, fx.computePipeline).has_value());
        REQUIRE(device.bindComputeBindingSet(cmd, 0, fx.computeSet0).has_value());
        for (std::uint32_t i = 0; i < computeDispatchesPerIter; ++i)
        {
            const std::size_t uboVariant = (i + seed) % fx.set1UboVariants.size();
            const bool useSetB = ((i + seed) & 1U) != 0U;
            const std::array<std::uint32_t, 4> push{
                seed * 1000U + i * 3U + 1U,
                seed * 2000U + i * 5U + 2U,
                seed * 3000U + i * 7U + 3U,
                seed * 4000U + i * 11U + 4U,
            };
            const std::array<std::uint32_t, 2> dynOffsets{
                static_cast<std::uint32_t>(uboVariant) * fx.uboStride,
                i * fx.ssboStride,
            };
            REQUIRE(device.bindComputeBindingSet(cmd, 1, useSetB ? fx.computeSet1B : fx.computeSet1A, dynOffsets)
                        .has_value());
            REQUIRE(device.setComputePushConstants(cmd, 0, asByteSpan(push)).has_value());
            REQUIRE(device.dispatch(cmd, 1, 1, 1).has_value());
        }

        REQUIRE(device
                    .resourceBarriers(cmd,
                                      {
                                          .bufferBarriers =
                                              std::array{
                                                  rachis::BufferBarrier{
                                                      .buffer = fx.computeOutBuf,
                                                      .stagesBefore = rachis::PipelineStage::computeShader,
                                                      .stagesAfter = rachis::PipelineStage::copy,
                                                      .accessBefore = rachis::Access::unorderedAccess,
                                                      .accessAfter = rachis::Access::copySource,
                                                  },
                                              },
                                          .textureBarriers =
                                              std::array{
                                                  rachis::TextureBarrier{
                                                      .texture = fx.renderTarget,
                                                      .stagesAfter = rachis::PipelineStage::renderTarget,
                                                      .accessAfter = rachis::Access::renderTarget,
                                                      .layoutAfter = rachis::TextureLayout::renderTarget,
                                                  },
                                              },
                                      })
                    .has_value());

        const std::array colorAttachments{
            rachis::ColorAttachment{.view = fx.renderTargetView, .loadOperation = rachis::LoadOperation::clear},
        };
        REQUIRE(device.beginRendering(cmd, {.colorAttachments = colorAttachments}).has_value());
        REQUIRE(device.bindGraphicsPipeline(cmd, fx.graphicsPipeline).has_value());
        REQUIRE(device.setViewport(cmd, {.width = 16.0F, .height = 16.0F}).has_value());
        REQUIRE(device.bindVertexBuffer(cmd, 0, (seed & 1U) != 0U ? fx.vboB : fx.vboA, 0).has_value());
        REQUIRE(device.bindGraphicsBindingSet(cmd, 0, fx.graphicsSet0).has_value());
        for (std::uint32_t d = 0; d < graphicsDrawsPerIter; ++d)
        {
            const std::int32_t col = static_cast<std::int32_t>(d % 16U);
            const std::int32_t row = static_cast<std::int32_t>(d / 16U);
            REQUIRE(device.setScissor(cmd, {.x = col, .y = row, .width = 1, .height = 1}).has_value());
            const std::size_t uboVariant = (d + seed) % fx.set1UboVariants.size();
            const bool useSetB = ((d + seed) % 2U) != 0U;
            const std::array<std::uint32_t, 4> push{
                seed * 5000U + d * 13U + 5U,
                seed * 6000U + d * 17U + 6U,
                seed * 7000U + d * 19U + 7U,
                seed * 8000U + d * 23U + 8U,
            };
            const std::array<std::uint32_t, 1> dynOffset{
                static_cast<std::uint32_t>(uboVariant) * fx.uboStride,
            };
            REQUIRE(device.bindGraphicsBindingSet(cmd, 1, useSetB ? fx.graphicsSet1B : fx.graphicsSet1A, dynOffset)
                        .has_value());
            REQUIRE(device.setGraphicsPushConstants(cmd, allGfx, 0, asByteSpan(push)).has_value());
            REQUIRE(device.draw(cmd, 3, 1, d * 3U, 0).has_value());
        }
        REQUIRE(device.endRendering(cmd).has_value());

        REQUIRE(device
                    .textureBarrier(cmd,
                                    {
                                        .texture = fx.renderTarget,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        REQUIRE(device
                    .copyBuffer(cmd,
                                {
                                    .source = fx.computeOutBuf,
                                    .destination = fx.computeReadback,
                                    .size = static_cast<std::uint64_t>(computeDispatchesPerIter) * fx.ssboStride,
                                })
                    .has_value());
        const rachis::BufferTextureCopyRegion rtCopy{
            .bytesPerRow = fx.renderTransfer.bytesPerRow,
            .rowsPerImage = fx.renderTransfer.rowsPerImage,
            .extent = {.width = 16, .height = 16, .depth = 1},
        };
        REQUIRE(device.copyTextureToBuffer(cmd, fx.renderTarget, fx.renderReadback, std::array{rtCopy}).has_value());
    };

    {
        auto warmCmd = device.acquireCommandBuffer(*cmdPool);
        REQUIRE(warmCmd);
        recordIteration(*warmCmd, 1U);
        auto pt = device.submit(std::move(*warmCmd));
        REQUIRE(pt);
        REQUIRE(device.wait(*pt).has_value());
        device.collectGarbage();
    }

    const auto baselineSnapshot = device.captureResourceMemorySnapshot();
    REQUIRE(baselineSnapshot);
    const auto baselineBudgets = device.queryMemoryBudgets();
    REQUIRE(baselineBudgets);
    REQUIRE_FALSE(baselineBudgets->empty());

    constexpr std::uint32_t replayIterations = 6;
    for (std::uint32_t iter = 0; iter < replayIterations; ++iter)
    {
        if (iter >= 3U)
        {
            REQUIRE(device.resetCommandPool(*cmdPool).has_value());
        }
        auto cmd = device.acquireCommandBuffer(*cmdPool);
        REQUIRE(cmd);

        if (iter < 3U)
        {
            recordIteration(*cmd, 900U + iter);
            REQUIRE(device.resetCommandBuffer(*cmd).has_value());
        }

        const std::uint32_t seed = 10U + iter;
        recordIteration(*cmd, seed);
        auto pt = device.submit(std::move(*cmd));
        REQUIRE(pt);
        REQUIRE(device.wait(*pt).has_value());
        device.collectGarbage();
        device.collectGarbage();

        std::vector<std::byte> rawCompute(static_cast<std::size_t>(computeDispatchesPerIter) * fx.ssboStride);
        REQUIRE(device.readBuffer(fx.computeReadback, 0, rawCompute).has_value());
        for (std::uint32_t i = 0; i < computeDispatchesPerIter; ++i)
        {
            std::array<std::uint32_t, 8> actual{};
            std::memcpy(actual.data(), rawCompute.data() + static_cast<std::size_t>(i) * fx.ssboStride, sizeof(actual));
            const auto& set1Ubo = fx.set1UboVariants[(i + seed) % fx.set1UboVariants.size()];
            const std::uint32_t expected0 = (seed * 1000U + i * 3U + 1U) + fx.set0Ubo.w0;
            const std::uint32_t expected1 = (seed * 2000U + i * 5U + 2U) + set1Ubo.w0;
            CAPTURE(iter, i);
            CHECK(actual[0] == expected0);
            CHECK(actual[1] == expected1);
        }

        std::vector<std::byte> rawPixels(static_cast<std::size_t>(fx.renderTransfer.requiredBufferSize));
        REQUIRE(device.readBuffer(fx.renderReadback, 0, rawPixels).has_value());
        for (std::uint32_t d = 0; d < graphicsDrawsPerIter; ++d)
        {
            const std::size_t col = d % 16U;
            const std::size_t row = d / 16U;
            std::array<std::uint32_t, 4> pixel{};
            std::memcpy(pixel.data(),
                        rawPixels.data() + row * fx.renderTransfer.bytesPerRow + col * sizeof(pixel),
                        sizeof(pixel));
            const auto& set1Ubo = fx.set1UboVariants[(d + seed) % fx.set1UboVariants.size()];
            const std::uint32_t vertexTag = ((seed & 1U) != 0U ? 50000U : 10000U) + d;
            const std::uint32_t push0 = seed * 5000U + d * 13U + 5U;
            CAPTURE(iter, d);
            CHECK(pixel[0] == (vertexTag + push0 + fx.set0Ubo.w0 + set1Ubo.w0));
        }
    }

    const auto afterSnapshot = device.captureResourceMemorySnapshot();
    REQUIRE(afterSnapshot);
    CHECK(afterSnapshot->liveAllocationCount == baselineSnapshot->liveAllocationCount);
    CHECK(afterSnapshot->liveAllocationBytes == baselineSnapshot->liveAllocationBytes);
    CHECK(afterSnapshot->retainedAllocationCount == 0);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal draws and dispatches rewrite one argument table per stage instead of allocating one per call",
          "[size:large][rhi][metal][push][binding][residency]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    constexpr std::uint32_t callsPerStage = 64;
    auto fx = buildMultiWorkloadFixture(device, callsPerStage, callsPerStage);

    auto cmdPool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
    });
    REQUIRE(cmdPool);

    const auto allGfx = rachis::ShaderStage::vertex | rachis::ShaderStage::pixel;
    auto recordAndCheck = [&](std::uint32_t seed) {
        auto cmd = device.acquireCommandBuffer(*cmdPool);
        REQUIRE(cmd);
        REQUIRE(device.bindComputePipeline(*cmd, fx.computePipeline).has_value());
        REQUIRE(device.bindComputeBindingSet(*cmd, 0, fx.computeSet0).has_value());
        for (std::uint32_t i = 0; i < callsPerStage; ++i)
        {
            const std::size_t uboVariant = (i + seed) % fx.set1UboVariants.size();
            const std::array<std::uint32_t, 4> push{
                seed * 1000U + i * 3U + 1U,
                seed * 2000U + i * 5U + 2U,
                seed * 3000U + i * 7U + 3U,
                seed * 4000U + i * 11U + 4U,
            };
            const std::array<std::uint32_t, 2> dynOffsets{
                static_cast<std::uint32_t>(uboVariant) * fx.uboStride,
                i * fx.ssboStride,
            };
            REQUIRE(
                device.bindComputeBindingSet(*cmd, 1, (i & 1U) != 0U ? fx.computeSet1B : fx.computeSet1A, dynOffsets)
                    .has_value());
            REQUIRE(device.setComputePushConstants(*cmd, 0, asByteSpan(push)).has_value());
            REQUIRE(device.dispatch(*cmd, 1, 1, 1).has_value());
        }
        REQUIRE(device
                    .resourceBarriers(*cmd,
                                      {
                                          .bufferBarriers =
                                              std::array{
                                                  rachis::BufferBarrier{
                                                      .buffer = fx.computeOutBuf,
                                                      .stagesBefore = rachis::PipelineStage::computeShader,
                                                      .stagesAfter = rachis::PipelineStage::copy,
                                                      .accessBefore = rachis::Access::unorderedAccess,
                                                      .accessAfter = rachis::Access::copySource,
                                                  },
                                              },
                                          .textureBarriers =
                                              std::array{
                                                  rachis::TextureBarrier{
                                                      .texture = fx.renderTarget,
                                                      .stagesAfter = rachis::PipelineStage::renderTarget,
                                                      .accessAfter = rachis::Access::renderTarget,
                                                      .layoutAfter = rachis::TextureLayout::renderTarget,
                                                  },
                                              },
                                      })
                    .has_value());

        const std::array colorAttachments{
            rachis::ColorAttachment{.view = fx.renderTargetView, .loadOperation = rachis::LoadOperation::clear},
        };
        REQUIRE(device.beginRendering(*cmd, {.colorAttachments = colorAttachments}).has_value());
        REQUIRE(device.bindGraphicsPipeline(*cmd, fx.graphicsPipeline).has_value());
        REQUIRE(device.setViewport(*cmd, {.width = 16.0F, .height = 16.0F}).has_value());
        REQUIRE(device.bindVertexBuffer(*cmd, 0, fx.vboA, 0).has_value());
        REQUIRE(device.bindGraphicsBindingSet(*cmd, 0, fx.graphicsSet0).has_value());
        for (std::uint32_t d = 0; d < callsPerStage; ++d)
        {
            const std::int32_t col = static_cast<std::int32_t>(d % 16U);
            const std::int32_t row = static_cast<std::int32_t>(d / 16U);
            REQUIRE(device.setScissor(*cmd, {.x = col, .y = row, .width = 1, .height = 1}).has_value());
            const std::size_t uboVariant = (d + seed) % fx.set1UboVariants.size();
            const std::array<std::uint32_t, 4> push{
                seed * 5000U + d * 13U + 5U,
                seed * 6000U + d * 17U + 6U,
                seed * 7000U + d * 19U + 7U,
                seed * 8000U + d * 23U + 8U,
            };
            const std::array<std::uint32_t, 1> dynOffset{
                static_cast<std::uint32_t>(uboVariant) * fx.uboStride,
            };
            REQUIRE(
                device.bindGraphicsBindingSet(*cmd, 1, (d & 1U) != 0U ? fx.graphicsSet1B : fx.graphicsSet1A, dynOffset)
                    .has_value());
            REQUIRE(device.setGraphicsPushConstants(*cmd, allGfx, 0, asByteSpan(push)).has_value());
            REQUIRE(device.draw(*cmd, 3, 1, d * 3U, 0).has_value());
        }
        REQUIRE(device.endRendering(*cmd).has_value());
        REQUIRE(device
                    .textureBarrier(*cmd,
                                    {
                                        .texture = fx.renderTarget,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        REQUIRE(device
                    .copyBuffer(*cmd,
                                {
                                    .source = fx.computeOutBuf,
                                    .destination = fx.computeReadback,
                                    .size = static_cast<std::uint64_t>(callsPerStage) * fx.ssboStride,
                                })
                    .has_value());
        const rachis::BufferTextureCopyRegion rtCopy{
            .bytesPerRow = fx.renderTransfer.bytesPerRow,
            .rowsPerImage = fx.renderTransfer.rowsPerImage,
            .extent = {.width = 16, .height = 16, .depth = 1},
        };
        REQUIRE(device.copyTextureToBuffer(*cmd, fx.renderTarget, fx.renderReadback, std::array{rtCopy}).has_value());
        auto pt = device.submit(std::move(*cmd));
        REQUIRE(pt);
        REQUIRE(device.wait(*pt).has_value());

        std::vector<std::byte> rawCompute(static_cast<std::size_t>(callsPerStage) * fx.ssboStride);
        REQUIRE(device.readBuffer(fx.computeReadback, 0, rawCompute).has_value());
        for (std::uint32_t i = 0; i < callsPerStage; ++i)
        {
            std::array<std::uint32_t, 8> actual{};
            std::memcpy(actual.data(), rawCompute.data() + static_cast<std::size_t>(i) * fx.ssboStride, sizeof(actual));
            const auto& set1Ubo = fx.set1UboVariants[(i + seed) % fx.set1UboVariants.size()];
            CAPTURE(seed, i);
            CHECK(actual[0] == (seed * 1000U + i * 3U + 1U) + fx.set0Ubo.w0);
            CHECK(actual[1] == (seed * 2000U + i * 5U + 2U) + set1Ubo.w0);
        }
        std::vector<std::byte> rawPixels(static_cast<std::size_t>(fx.renderTransfer.requiredBufferSize));
        REQUIRE(device.readBuffer(fx.renderReadback, 0, rawPixels).has_value());
        for (std::uint32_t d = 0; d < callsPerStage; ++d)
        {
            const std::size_t col = d % 16U;
            const std::size_t row = d / 16U;
            std::array<std::uint32_t, 4> pixel{};
            std::memcpy(pixel.data(),
                        rawPixels.data() + row * fx.renderTransfer.bytesPerRow + col * sizeof(pixel),
                        sizeof(pixel));
            const auto& set1Ubo = fx.set1UboVariants[(d + seed) % fx.set1UboVariants.size()];
            CAPTURE(seed, d);
            CHECK(pixel[0] == (10000U + d + seed * 5000U + d * 13U + 5U + fx.set0Ubo.w0 + set1Ubo.w0));
        }
    };

    rachis::metal::resetRecordingTelemetry();
    recordAndCheck(1U);
    const auto firstRecording = rachis::metal::recordingTelemetry();
    CAPTURE(firstRecording.argumentTableCreateCount, firstRecording.argumentTablePoolReuseCount);
    CHECK(firstRecording.argumentTableCreateCount <= 3);
    recordAndCheck(2U);
    const auto secondRecording = rachis::metal::recordingTelemetry();
    CHECK(secondRecording.argumentTableCreateCount == firstRecording.argumentTableCreateCount);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal mixed copy, barrier, mipmap, resolve, clear, and indirect encoders share pooled tables and single "
          "residency commit",
          "[size:large][rhi][metal][barrier][residency]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    const auto pushCsPnms = loadPnmsBytes("MetalPushConstant.cs.metallib");
    const std::array pushCsBindings{
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
    };
    const std::array pushCsRange{rachis::ShaderPushConstantRequirement{0, 16}};
    auto pushCs = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = pushCsPnms,
        .entryPoint = "computeMain",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, pushCsBindings, pushCsRange),
    });
    REQUIRE(pushCs);
    auto pushCsBindLayout = device.createBindingLayout({
        .entries =
            std::array{rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute}},
    });
    REQUIRE(pushCsBindLayout);
    auto pushCsPipeLayout = device.createPipelineLayout({
        .bindingLayouts = std::array{*pushCsBindLayout},
        .pushConstantRanges = std::array{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}},
    });
    REQUIRE(pushCsPipeLayout);
    auto pushCsPipeline = device.createComputePipeline({.layout = *pushCsPipeLayout, .computeShader = *pushCs});
    REQUIRE(pushCsPipeline);

    const auto singleVsPnms = loadPnmsBytes("MetalRemapContract.single.vs.metallib");
    const auto singlePsPnms = loadPnmsBytes("MetalRemapContract.single.ps.metallib");
    const std::array vsReq{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1}};
    const std::array psReq{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::sampledTexture, 1},
        rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::sampler, 1},
    };
    const std::array vsLocations{0U, 1U};
    auto singleVs = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = singleVsPnms,
        .entryPoint = "pushAndBindingZeroVertex",
        .shaderInterface = {.bindings = vsReq, .pushConstants = pushCsRange, .vertexInputLocations = vsLocations},
    });
    auto singlePs = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = singlePsPnms,
        .entryPoint = "pushAndBindingZeroPixel",
        .shaderInterface = {.bindings = psReq, .pushConstants = pushCsRange},
    });
    REQUIRE(singleVs);
    REQUIRE(singlePs);
    const auto allGfx = rachis::ShaderStage::vertex | rachis::ShaderStage::pixel;
    auto gfxBindLayout = device.createBindingLayout({
        .entries =
            std::array{
                rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, allGfx, 1},
                rachis::BindingLayoutEntry{1, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel, 1},
                rachis::BindingLayoutEntry{2, rachis::BindingType::sampler, rachis::ShaderStage::pixel, 1},
            },
    });
    REQUIRE(gfxBindLayout);
    auto gfxPipeLayout = device.createPipelineLayout({
        .bindingLayouts = std::array{*gfxBindLayout},
        .pushConstantRanges = std::array{rachis::PushConstantRange{allGfx, 0, 16}},
    });
    REQUIRE(gfxPipeLayout);
    const std::array vertexAttrs{
        rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .offset = 0, .shaderLocation = 0},
        rachis::VertexAttribute{.format = rachis::VertexFormat::uint32, .offset = 8, .shaderLocation = 1},
    };
    const std::array vertexBuffers{
        rachis::VertexBufferLayout{
            .stride = sizeof(RemapVertexHost),
            .stepMode = rachis::VertexStepMode::vertex,
            .attributes = vertexAttrs,
        },
    };
    auto gfxPipeline = device.createGraphicsPipeline({
        .layout = *gfxPipeLayout,
        .vertexShader = *singleVs,
        .pixelShader = *singlePs,
        .vertexBuffers = vertexBuffers,
        .cullMode = rachis::CullMode::none,
        .colorTargets = std::array{rachis::ColorTargetState{.format = rachis::Format::rgba32Uint}},
    });
    REQUIRE(gfxPipeline);

    struct IndirectArgsPayload final
    {
        std::array<std::uint32_t, 4> dispatchGroupCounts{1, 1, 1, 0};
        std::array<std::uint32_t, 4> drawArgs{3, 1, 0, 0};
    };
    auto indirectUpload = device.createBuffer({
        .size = sizeof(IndirectArgsPayload),
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto indirectGpu = device.createBuffer({
        .size = sizeof(IndirectArgsPayload),
        .usage = rachis::BufferUsage::copyDestination | rachis::BufferUsage::indirect,
    });
    REQUIRE(indirectUpload);
    REQUIRE(indirectGpu);
    const IndirectArgsPayload indirectPayload{};
    REQUIRE(device.writeBuffer(*indirectUpload, 0, asByteSpan(indirectPayload)).has_value());

    auto computeStorage = device.createBuffer({
        .size = 16,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
    });
    auto computeReadback = device.createBuffer({
        .size = 16,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(computeStorage);
    REQUIRE(computeReadback);
    auto computeSet = rachis::test::createBindingSet(
        device,
        {
            .layout = *pushCsBindLayout,
            .writes =
                std::array{
                    rachis::BindingWrite{1, 0, rachis::BufferBinding{*computeStorage, 0, sizeof(std::uint32_t), 4}},
                },
        });
    REQUIRE(computeSet);

    auto mipTexture = device.createTexture({
        .width = 4,
        .height = 4,
        .mipLevels = 3,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::copySource,
    });
    REQUIRE(mipTexture);
    auto mipSampledView = device.createTextureView({
        .texture = *mipTexture,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::texture2D,
    });
    REQUIRE(mipSampledView);

    const rachis::BufferTextureCopyRegion mip0Region{.extent = {.width = 4, .height = 4, .depth = 1}};
    const auto mip0Transfer = device.queryTextureTransferLayout(*mipTexture, mip0Region);
    REQUIRE(mip0Transfer);
    const rachis::BufferTextureCopyRegion mip2Region{
        .textureSubresource = {.mipLevel = 2},
        .extent = {.width = 1, .height = 1, .depth = 1},
    };
    const auto mip2Transfer = device.queryTextureTransferLayout(*mipTexture, mip2Region);
    REQUIRE(mip2Transfer);
    auto mipUpload = device.createBuffer({
        .size = mip0Transfer->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto mip2Readback = device.createBuffer({
        .size = mip2Transfer->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(mipUpload);
    REQUIRE(mip2Readback);

    auto msaaSource = device.createTexture({
        .width = 2,
        .height = 2,
        .sampleCount = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
    });
    auto msaaResolved = device.createTexture({
        .width = 2,
        .height = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::copySource,
    });
    REQUIRE(msaaSource);
    REQUIRE(msaaResolved);
    auto msaaView = device.createTextureView({
        .texture = *msaaSource,
        .usage = rachis::TextureViewUsage::renderTarget,
        .dimension = rachis::TextureViewDimension::texture2D,
    });
    REQUIRE(msaaView);
    const rachis::BufferTextureCopyRegion resolveRegion{.extent = {.width = 2, .height = 2, .depth = 1}};
    const auto resolveTransfer = device.queryTextureTransferLayout(*msaaResolved, resolveRegion);
    REQUIRE(resolveTransfer);
    auto resolveReadback = device.createBuffer({
        .size = resolveTransfer->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(resolveReadback);

    auto ubo16Buf = device.createBuffer({
        .size = sizeof(CompactUniforms16Host),
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(ubo16Buf);
    const CompactUniforms16Host ubo16{.words = {10U, 20U, 30U, 40U}};
    REQUIRE(device.writeBuffer(*ubo16Buf, 0, asByteSpan(ubo16)).has_value());

    const std::array<RemapVertexHost, 3> tri{{
        {.position = {-1.0F, -1.0F}, .vertexTag = 333U},
        {.position = {3.0F, -1.0F}, .vertexTag = 333U},
        {.position = {-1.0F, 3.0F}, .vertexTag = 333U},
    }};
    auto vbo = device.createBuffer({
        .size = sizeof(tri),
        .usage = rachis::BufferUsage::vertex,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(vbo);
    REQUIRE(device.writeBuffer(*vbo, 0, asByteSpan(tri)).has_value());

    auto sampler = device.createSampler({
        .minFilter = rachis::FilterMode::nearest,
        .magFilter = rachis::FilterMode::nearest,
    });
    REQUIRE(sampler);
    auto gfxSet = rachis::test::createBindingSet(
        device,
        {
            .layout = *gfxBindLayout,
            .writes =
                std::array{
                    rachis::BindingWrite{0, 0, rachis::BufferBinding{*ubo16Buf, 0, sizeof(CompactUniforms16Host)}},
                    rachis::BindingWrite{1, 0, *mipSampledView},
                    rachis::BindingWrite{2, 0, *sampler},
                },
        });
    REQUIRE(gfxSet);

    auto drawTarget = device.createTexture({
        .width = 2,
        .height = 2,
        .format = rachis::Format::rgba32Uint,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
    });
    REQUIRE(drawTarget);
    auto drawTargetView = device.createTextureView({
        .texture = *drawTarget,
        .usage = rachis::TextureViewUsage::renderTarget,
        .dimension = rachis::TextureViewDimension::texture2D,
    });
    REQUIRE(drawTargetView);
    const rachis::BufferTextureCopyRegion drawRegion{.extent = {.width = 2, .height = 2, .depth = 1}};
    const auto drawTransfer = device.queryTextureTransferLayout(*drawTarget, drawRegion);
    REQUIRE(drawTransfer);
    auto drawReadback = device.createBuffer({
        .size = drawTransfer->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(drawReadback);

    auto cmdPool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .flags = rachis::CommandPoolFlag::allowIndividualReset,
    });
    REQUIRE(cmdPool);

    auto runMixedPass = [&](std::uint8_t mipConstantByte, std::uint32_t markerWord, float clearGreen) {
        std::vector<std::byte> mipBytes(static_cast<std::size_t>(mip0Transfer->requiredBufferSize), std::byte{0});
        for (std::uint32_t y = 0; y < 4; ++y)
        {
            for (std::uint32_t x = 0; x < 4; ++x)
            {
                const std::size_t off = static_cast<std::size_t>(y) * mip0Transfer->bytesPerRow + x * 4U;
                mipBytes[off + 0] = static_cast<std::byte>(mipConstantByte);
                mipBytes[off + 1] = static_cast<std::byte>(mipConstantByte / 2U);
                mipBytes[off + 2] = std::byte{80};
                mipBytes[off + 3] = std::byte{255};
            }
        }
        REQUIRE(device.writeBuffer(*mipUpload, 0, mipBytes).has_value());

        auto cmd = device.acquireCommandBuffer(*cmdPool);
        REQUIRE(cmd);

        REQUIRE(device
                    .resourceBarriers(*cmd,
                                      {
                                          .bufferBarriers =
                                              std::array{
                                                  rachis::BufferBarrier{
                                                      .buffer = *indirectGpu,
                                                      .stagesAfter = rachis::PipelineStage::copy,
                                                      .accessAfter = rachis::Access::copyDestination,
                                                  },
                                              },
                                          .textureBarriers =
                                              std::array{
                                                  rachis::TextureBarrier{
                                                      .texture = *mipTexture,
                                                      .stagesAfter = rachis::PipelineStage::copy,
                                                      .accessAfter = rachis::Access::copyDestination,
                                                      .layoutAfter = rachis::TextureLayout::copyDestination,
                                                      .subresources = {.mipLevelCount = 1},
                                                  },
                                              },
                                      })
                    .has_value());
        REQUIRE(device
                    .copyBuffer(*cmd,
                                {
                                    .source = *indirectUpload,
                                    .destination = *indirectGpu,
                                    .size = sizeof(IndirectArgsPayload),
                                })
                    .has_value());
        const rachis::BufferTextureCopyRegion mip0Copy{
            .bytesPerRow = mip0Transfer->bytesPerRow,
            .rowsPerImage = mip0Transfer->rowsPerImage,
            .extent = {.width = 4, .height = 4, .depth = 1},
        };
        REQUIRE(device.copyBufferToTexture(*cmd, *mipUpload, *mipTexture, std::array{mip0Copy}).has_value());

        REQUIRE(device
                    .resourceBarriers(*cmd,
                                      {
                                          .bufferBarriers =
                                              std::array{
                                                  rachis::BufferBarrier{
                                                      .buffer = *indirectGpu,
                                                      .stagesBefore = rachis::PipelineStage::copy,
                                                      .stagesAfter = rachis::PipelineStage::indirect,
                                                      .accessBefore = rachis::Access::copyDestination,
                                                      .accessAfter = rachis::Access::indirectArgument,
                                                  },
                                              },
                                          .textureBarriers =
                                              std::array{
                                                  rachis::TextureBarrier{
                                                      .texture = *mipTexture,
                                                      .stagesBefore = rachis::PipelineStage::copy,
                                                      .stagesAfter = rachis::PipelineStage::computeShader,
                                                      .accessBefore = rachis::Access::copyDestination,
                                                      .accessAfter = rachis::Access::shaderResource,
                                                      .layoutBefore = rachis::TextureLayout::copyDestination,
                                                      .layoutAfter = rachis::TextureLayout::shaderResource,
                                                      .subresources = {.mipLevelCount = 1},
                                                  },
                                              },
                                      })
                    .has_value());
        REQUIRE(device
                    .generateMipmaps(*cmd,
                                     {
                                         .texture = *mipTexture,
                                         .baseLayout = rachis::TextureLayout::shaderResource,
                                         .finalLayout = rachis::TextureLayout::shaderResource,
                                     })
                    .has_value());

        const std::array<std::uint32_t, 4> csPush{markerWord, 0U, 0U, 0U};
        REQUIRE(device.bindComputePipeline(*cmd, *pushCsPipeline).has_value());
        REQUIRE(device.bindComputeBindingSet(*cmd, 0, *computeSet).has_value());
        REQUIRE(device.setComputePushConstants(*cmd, 0, asByteSpan(csPush)).has_value());
        REQUIRE(device
                    .dispatchIndirect(*cmd,
                                      {
                                          .argumentBuffer = *indirectGpu,
                                          .argumentOffset = 0,
                                          .maxDispatchCount = 1,
                                      })
                    .has_value());

        REQUIRE(device
                    .textureBarrier(*cmd,
                                    {
                                        .texture = *msaaSource,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
        const std::array msaaAttachments{
            rachis::ColorAttachment{
                .view = *msaaView,
                .loadOperation = rachis::LoadOperation::clear,
                .clearValue = {0.0F, 0.0F, 0.0F, 1.0F},
            },
        };
        REQUIRE(device.beginRendering(*cmd, {.colorAttachments = msaaAttachments}).has_value());
        REQUIRE(device
                    .clearColorAttachment(*cmd,
                                          0,
                                          {0.25F, clearGreen, 0.75F, 1.0F},
                                          std::array{rachis::ScissorRect{.x = 0, .y = 0, .width = 2, .height = 2}})
                    .has_value());
        REQUIRE(device.endRendering(*cmd).has_value());

        REQUIRE(device
                    .resourceBarriers(*cmd,
                                      {
                                          .textureBarriers =
                                              std::array{
                                                  rachis::TextureBarrier{
                                                      .texture = *msaaSource,
                                                      .stagesBefore = rachis::PipelineStage::renderTarget,
                                                      .stagesAfter = rachis::PipelineStage::resolve,
                                                      .accessBefore = rachis::Access::renderTarget,
                                                      .accessAfter = rachis::Access::resolveSource,
                                                      .layoutBefore = rachis::TextureLayout::renderTarget,
                                                      .layoutAfter = rachis::TextureLayout::resolveSource,
                                                  },
                                                  rachis::TextureBarrier{
                                                      .texture = *msaaResolved,
                                                      .stagesAfter = rachis::PipelineStage::resolve,
                                                      .accessAfter = rachis::Access::resolveDestination,
                                                      .layoutAfter = rachis::TextureLayout::resolveDestination,
                                                  },
                                              },
                                      })
                    .has_value());
        REQUIRE(device.resolveTexture(*cmd, {.source = *msaaSource, .destination = *msaaResolved}).has_value());

        REQUIRE(device
                    .resourceBarriers(*cmd,
                                      {
                                          .textureBarriers =
                                              std::array{
                                                  rachis::TextureBarrier{
                                                      .texture = *mipTexture,
                                                      .stagesBefore = rachis::PipelineStage::computeShader,
                                                      .stagesAfter = rachis::PipelineStage::pixelShader,
                                                      .accessBefore = rachis::Access::shaderResource,
                                                      .accessAfter = rachis::Access::shaderResource,
                                                      .layoutBefore = rachis::TextureLayout::shaderResource,
                                                      .layoutAfter = rachis::TextureLayout::shaderResource,
                                                  },
                                                  rachis::TextureBarrier{
                                                      .texture = *drawTarget,
                                                      .stagesAfter = rachis::PipelineStage::renderTarget,
                                                      .accessAfter = rachis::Access::renderTarget,
                                                      .layoutAfter = rachis::TextureLayout::renderTarget,
                                                  },
                                              },
                                      })
                    .has_value());
        const std::array drawAttachments{
            rachis::ColorAttachment{.view = *drawTargetView, .loadOperation = rachis::LoadOperation::clear},
        };
        const std::array<std::uint32_t, 4> gfxPush{markerWord + 5U, 20U, 30U, 40U};
        REQUIRE(device.beginRendering(*cmd, {.colorAttachments = drawAttachments}).has_value());
        REQUIRE(device.bindGraphicsPipeline(*cmd, *gfxPipeline).has_value());
        REQUIRE(device.setViewport(*cmd, {.width = 2.0F, .height = 2.0F}).has_value());
        REQUIRE(device.setScissor(*cmd, {.width = 2, .height = 2}).has_value());
        REQUIRE(device.bindVertexBuffer(*cmd, 0, *vbo, 0).has_value());
        REQUIRE(device.bindGraphicsBindingSet(*cmd, 0, *gfxSet).has_value());
        REQUIRE(device.setGraphicsPushConstants(*cmd, allGfx, 0, asByteSpan(gfxPush)).has_value());
        REQUIRE(device
                    .drawIndirect(*cmd,
                                  {
                                      .argumentBuffer = *indirectGpu,
                                      .argumentOffset = 16,
                                      .maxDrawCount = 1,
                                  })
                    .has_value());
        REQUIRE(device.endRendering(*cmd).has_value());

        REQUIRE(device
                    .resourceBarriers(*cmd,
                                      {
                                          .bufferBarriers =
                                              std::array{
                                                  rachis::BufferBarrier{
                                                      .buffer = *computeStorage,
                                                      .stagesBefore = rachis::PipelineStage::computeShader,
                                                      .stagesAfter = rachis::PipelineStage::copy,
                                                      .accessBefore = rachis::Access::unorderedAccess,
                                                      .accessAfter = rachis::Access::copySource,
                                                  },
                                              },
                                          .textureBarriers =
                                              std::array{
                                                  rachis::TextureBarrier{
                                                      .texture = *mipTexture,
                                                      .stagesBefore = rachis::PipelineStage::pixelShader,
                                                      .stagesAfter = rachis::PipelineStage::copy,
                                                      .accessBefore = rachis::Access::shaderResource,
                                                      .accessAfter = rachis::Access::copySource,
                                                      .layoutBefore = rachis::TextureLayout::shaderResource,
                                                      .layoutAfter = rachis::TextureLayout::copySource,
                                                  },
                                                  rachis::TextureBarrier{
                                                      .texture = *msaaResolved,
                                                      .stagesBefore = rachis::PipelineStage::resolve,
                                                      .stagesAfter = rachis::PipelineStage::copy,
                                                      .accessBefore = rachis::Access::resolveDestination,
                                                      .accessAfter = rachis::Access::copySource,
                                                      .layoutBefore = rachis::TextureLayout::resolveDestination,
                                                      .layoutAfter = rachis::TextureLayout::copySource,
                                                  },
                                                  rachis::TextureBarrier{
                                                      .texture = *drawTarget,
                                                      .stagesBefore = rachis::PipelineStage::renderTarget,
                                                      .stagesAfter = rachis::PipelineStage::copy,
                                                      .accessBefore = rachis::Access::renderTarget,
                                                      .accessAfter = rachis::Access::copySource,
                                                      .layoutBefore = rachis::TextureLayout::renderTarget,
                                                      .layoutAfter = rachis::TextureLayout::copySource,
                                                  },
                                              },
                                      })
                    .has_value());
        REQUIRE(device
                    .copyBuffer(*cmd,
                                {
                                    .source = *computeStorage,
                                    .destination = *computeReadback,
                                    .size = sizeof(std::uint32_t),
                                })
                    .has_value());
        const rachis::BufferTextureCopyRegion mip2Copy{
            .bytesPerRow = mip2Transfer->bytesPerRow,
            .rowsPerImage = mip2Transfer->rowsPerImage,
            .textureSubresource = {.mipLevel = 2},
            .extent = {.width = 1, .height = 1, .depth = 1},
        };
        REQUIRE(device.copyTextureToBuffer(*cmd, *mipTexture, *mip2Readback, std::array{mip2Copy}).has_value());
        const rachis::BufferTextureCopyRegion resolveCopy{
            .bytesPerRow = resolveTransfer->bytesPerRow,
            .rowsPerImage = resolveTransfer->rowsPerImage,
            .extent = {.width = 2, .height = 2, .depth = 1},
        };
        REQUIRE(device.copyTextureToBuffer(*cmd, *msaaResolved, *resolveReadback, std::array{resolveCopy}).has_value());
        const rachis::BufferTextureCopyRegion drawCopy{
            .bytesPerRow = drawTransfer->bytesPerRow,
            .rowsPerImage = drawTransfer->rowsPerImage,
            .extent = {.width = 2, .height = 2, .depth = 1},
        };
        REQUIRE(device.copyTextureToBuffer(*cmd, *drawTarget, *drawReadback, std::array{drawCopy}).has_value());

        auto pt = device.submit(std::move(*cmd));
        REQUIRE(pt);
        REQUIRE(device.wait(*pt).has_value());
        device.collectGarbage();

        std::uint32_t computeWord = 0;
        REQUIRE(device
                    .readBuffer(*computeReadback,
                                0,
                                std::span<std::byte>{reinterpret_cast<std::byte*>(&computeWord), sizeof(computeWord)})
                    .has_value());
        CHECK(computeWord == markerWord);

        std::array<std::byte, 4> mip2Texel{};
        REQUIRE(device.readBuffer(*mip2Readback, 0, mip2Texel).has_value());
        CHECK(nearByte(mip2Texel[0], mipConstantByte, 2));
        CHECK(nearByte(mip2Texel[1], static_cast<std::uint8_t>(mipConstantByte / 2U), 2));

        std::array<std::byte, 4> resolvedTexel{};
        REQUIRE(device.readBuffer(*resolveReadback, 0, resolvedTexel).has_value());
        CHECK(nearByte(resolvedTexel[0], 64, 2));
        CHECK(nearByte(resolvedTexel[1], static_cast<std::uint8_t>(clearGreen * 255.0F + 0.5F), 2));
        CHECK(nearByte(resolvedTexel[2], 191, 2));

        std::array<std::uint32_t, 4> drawnPixel{};
        REQUIRE(
            device
                .readBuffer(*drawReadback,
                            0,
                            std::span<std::byte>{reinterpret_cast<std::byte*>(drawnPixel.data()), sizeof(drawnPixel)})
                .has_value());
        CHECK(drawnPixel[0] == (333U + (markerWord + 5U) + ubo16.words[0]));
        CHECK(drawnPixel[2] == (ubo16.words[1] + 30U));
    };

    runMixedPass(160, 0xCAFE1234U, 0.5F);

    REQUIRE(device.resetCommandPool(*cmdPool).has_value());
    runMixedPass(220, 0xBEEF5678U, 0.75F);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

#endif
