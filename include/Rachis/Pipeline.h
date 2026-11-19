#pragma once

#include "Rachis/Binding.h"
#include "Rachis/Error.h"

#include <array>
#include <bit>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace rachis
{

    inline constexpr std::uint32_t maxBindingSets = 4;
    inline constexpr std::uint32_t maxPushConstantBytes = 128;
    inline constexpr std::uint32_t maxColorAttachments = 8;
    inline constexpr std::uint32_t maxVertexBuffers = 8;
    inline constexpr std::uint32_t maxVertexAttributes = 16;
    inline constexpr std::uint32_t maxStreamOutputBuffers = 4;
    inline constexpr std::uint32_t maxStreamOutputStride = 2048;
    inline constexpr std::uint32_t maxSpecializationConstants = 64;
    inline constexpr std::uint32_t pushConstantRegister = 0;
    inline constexpr std::uint32_t pushConstantRegisterSpace = maxBindingSets;
    inline constexpr std::uint32_t drawIndexRegister = 0;
    inline constexpr std::uint32_t drawIndexRegisterSpace = maxBindingSets + 1;

    struct PipelineCacheKey final
    {
        std::uint64_t high = 0;
        std::uint64_t low = 0;

        [[nodiscard]] constexpr bool isValid() const noexcept
        {
            return high != 0 || low != 0;
        }

        friend constexpr auto operator<=>(const PipelineCacheKey&, const PipelineCacheKey&) noexcept = default;
    };

    struct PipelineCacheDesc final
    {
        std::span<const std::byte> initialData = {};
        std::string_view label = {};
    };

    enum class PipelineCachePolicy : std::uint8_t
    {
        populate,

        requireHit,
    };

    enum class PipelineCompilationState : std::uint8_t
    {
        pending,
        succeeded,
        failed,
    };

    enum class SpecializationConstantFormat : std::uint8_t
    {
        boolean,
        float32,
        signedInt32,
        unsignedInt32,
    };

    struct SpecializationConstant final
    {
        std::uint32_t id = 0;
        SpecializationConstantFormat format = SpecializationConstantFormat::unsignedInt32;

        std::uint32_t valueBits = 0;

        [[nodiscard]] static constexpr SpecializationConstant makeBoolean(std::uint32_t id, bool value) noexcept
        {
            return {id, SpecializationConstantFormat::boolean, value ? 1U : 0U};
        }

        [[nodiscard]] static constexpr SpecializationConstant makeFloat32(std::uint32_t id, float value) noexcept
        {
            return {id, SpecializationConstantFormat::float32, std::bit_cast<std::uint32_t>(value)};
        }

        [[nodiscard]] static constexpr SpecializationConstant makeSignedInt32(std::uint32_t id,
                                                                              std::int32_t value) noexcept
        {
            return {id, SpecializationConstantFormat::signedInt32, std::bit_cast<std::uint32_t>(value)};
        }

        [[nodiscard]] static constexpr SpecializationConstant makeUnsignedInt32(std::uint32_t id,
                                                                                std::uint32_t value) noexcept
        {
            return {id, SpecializationConstantFormat::unsignedInt32, value};
        }

        friend constexpr auto operator<=>(const SpecializationConstant&,
                                          const SpecializationConstant&) noexcept = default;
    };

    struct ShaderLibraryEntryDesc final
    {
        ShaderStage stage = ShaderStage::none;
        std::span<const std::byte> bytecode = {};
        std::string_view entryPoint = "main";

        std::span<const SpecializationConstant> specializationConstants = {};
    };

    struct ShaderLibraryDesc final
    {
        std::span<const ShaderLibraryEntryDesc> entries = {};
        std::string_view label = {};
    };

    enum class ShaderResourceType : std::uint8_t
    {
        uniformBuffer,
        readOnlyStorageBuffer,
        storageBuffer,
        sampledTexture,
        storageTexture,
        sampler,
        accelerationStructure,
        minMipFeedback,
        mipRegionUsedFeedback,
    };

    struct ShaderBindingRequirement final
    {
        std::uint32_t set = 0;
        std::uint32_t binding = 0;
        ShaderResourceType type = ShaderResourceType::uniformBuffer;

        std::uint32_t count = 1;
    };

    struct ShaderPushConstantRequirement final
    {
        std::uint32_t offset = 0;
        std::uint32_t size = 0;
    };

    struct ShaderWorkgroupSize final
    {
        std::uint32_t x = 0;
        std::uint32_t y = 0;
        std::uint32_t z = 0;
    };

    struct ShaderRecordConstantRange final
    {
        std::uint32_t shaderRegister = 0;
        std::uint32_t registerSpace = 0;
        std::uint32_t size = 0;

        friend constexpr auto operator<=>(const ShaderRecordConstantRange&,
                                          const ShaderRecordConstantRange&) noexcept = default;
    };

    struct ShaderInterfaceDesc final
    {
        std::span<const ShaderBindingRequirement> bindings = {};
        std::span<const ShaderPushConstantRequirement> pushConstants = {};
        std::span<const std::uint32_t> vertexInputLocations = {};
        std::optional<ShaderWorkgroupSize> workgroupSize = {};

        std::optional<ShaderRecordConstantRange> shaderRecordConstants = {};

        bool drawIndex = false;
    };

    struct ShaderInterfaceSnapshot final
    {
        ShaderStage stage = ShaderStage::none;
        std::vector<ShaderBindingRequirement> bindings = {};
        std::vector<ShaderPushConstantRequirement> pushConstants = {};
        std::vector<std::uint32_t> vertexInputLocations = {};
        std::optional<ShaderWorkgroupSize> workgroupSize = {};
        std::optional<ShaderRecordConstantRange> shaderRecordConstants = {};
        bool drawIndex = false;
    };

    struct ShaderModuleDesc final
    {
        ShaderStage stage = ShaderStage::none;
        std::span<const std::byte> bytecode = {};
        std::string_view entryPoint = "main";
        std::string_view label = {};
        ShaderLibraryHandle library = {};

        ShaderInterfaceDesc shaderInterface = {};

        std::span<const SpecializationConstant> specializationConstants = {};
    };

    struct PushConstantRange final
    {
        ShaderStage visibility = ShaderStage::all;
        std::uint32_t offset = 0;
        std::uint32_t size = 0;
    };

    struct PipelineLayoutDesc final
    {
        std::span<const BindingLayoutHandle> bindingLayouts = {};
        std::span<const PushConstantRange> pushConstantRanges = {};

        bool drawIndex = false;
        std::string_view label = {};
    };

    struct DerivedPipelineLayout final
    {
        std::vector<std::vector<BindingLayoutEntry>> bindingSets = {};
        std::vector<PushConstantRange> pushConstantRanges = {};
        bool drawIndex = false;
    };

    [[nodiscard]] Result<DerivedPipelineLayout> derivePipelineLayout(std::span<const ShaderInterfaceSnapshot> shaders);

    struct ComputePipelineDesc final
    {
        PipelineLayoutHandle layout = {};
        ShaderModuleHandle computeShader = {};
        std::string_view label = {};
        PipelineCacheHandle cache = {};
        PipelineCacheKey cacheKey = {};
        PipelineCachePolicy cachePolicy = PipelineCachePolicy::populate;

        std::uint32_t requiredSubgroupSize = 0;
    };

    enum class RayTracingShaderGroupType : std::uint8_t
    {
        general,
        trianglesHitGroup,
        proceduralHitGroup,
    };

    struct RayTracingShaderGroupDesc final
    {
        RayTracingShaderGroupType type = RayTracingShaderGroupType::general;

        ShaderModuleHandle generalShader = {};

        ShaderModuleHandle closestHitShader = {};
        ShaderModuleHandle anyHitShader = {};
        ShaderModuleHandle intersectionShader = {};

        std::optional<ShaderRecordConstantRange> shaderRecordConstants = {};
    };

    struct RayTracingPipelineLibraryDesc final
    {
        PipelineLayoutHandle layout = {};

        std::span<const RayTracingShaderGroupDesc> groups = {};

        std::uint32_t maxPayloadSize = 0;
        std::uint32_t maxAttributeSize = 0;
        std::string_view label = {};
    };

    struct RayTracingPipelineDesc final
    {
        PipelineLayoutHandle layout = {};

        std::span<const RayTracingPipelineLibraryHandle> libraries = {};

        std::span<const RayTracingShaderGroupDesc> groups = {};
        std::uint32_t maxPayloadSize = 0;
        std::uint32_t maxAttributeSize = 0;
        std::uint32_t maxRecursionDepth = 1;
        std::string_view label = {};
    };

    struct RayTracingShaderTableRegion final
    {
        BufferHandle buffer = {};
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
        std::uint64_t stride = 0;
    };

    struct RayTracingDispatchDesc final
    {
        RayTracingShaderTableRegion rayGeneration = {};
        RayTracingShaderTableRegion miss = {};
        RayTracingShaderTableRegion hitGroup = {};
        RayTracingShaderTableRegion callable = {};
        std::uint32_t width = 1;
        std::uint32_t height = 1;
        std::uint32_t depth = 1;
    };

    struct RayTracingIndirectDispatchDesc final
    {
        RayTracingShaderTableRegion rayGeneration = {};
        RayTracingShaderTableRegion miss = {};
        RayTracingShaderTableRegion hitGroup = {};
        RayTracingShaderTableRegion callable = {};
        BufferHandle argumentBuffer = {};
        std::uint64_t argumentOffset = 0;
    };

    struct RayTracingCapabilities;

    struct RayTracingShaderTableLayoutDesc final
    {
        std::span<const RayTracingShaderGroupDesc> groups = {};

        std::uint32_t rayGenerationGroup = 0;
        std::span<const std::uint32_t> missGroups = {};
        std::span<const std::uint32_t> hitGroups = {};
        std::span<const std::uint32_t> callableGroups = {};
    };

    struct RayTracingShaderTableRecord final
    {
        std::uint32_t group = 0;

        std::uint64_t handleOffset = 0;
        std::uint64_t recordConstantsOffset = 0;
        std::uint32_t recordConstantsSize = 0;
        std::uint32_t recordSize = 0;
    };

    struct RayTracingShaderTableLayout final
    {
        std::uint64_t size = 0;
        std::uint64_t alignment = 0;

        RayTracingShaderTableRegion rayGeneration = {};
        RayTracingShaderTableRegion miss = {};
        RayTracingShaderTableRegion hitGroup = {};
        RayTracingShaderTableRegion callable = {};

        std::vector<RayTracingShaderTableRecord> records = {};

        constexpr void setBuffer(BufferHandle buffer) noexcept
        {
            for (auto* region : {&rayGeneration, &miss, &hitGroup, &callable})
            {
                if (region->size != 0)
                {
                    region->buffer = buffer;
                }
            }
        }
    };

    [[nodiscard]] Result<RayTracingShaderTableLayout> layoutRayTracingShaderTable(
        const RayTracingShaderTableLayoutDesc& desc, const RayTracingCapabilities& limits);

    struct MeshShaderCapabilities final
    {
        std::uint32_t maxTaskWorkgroupInvocations = 0;
        std::uint32_t maxTaskPayloadBytes = 0;
        std::uint32_t maxMeshWorkgroupInvocations = 0;
        std::uint32_t maxMeshOutputVertices = 0;
        std::uint32_t maxMeshOutputPrimitives = 0;
        std::uint32_t maxMeshOutputMemoryBytes = 0;
        std::array<std::uint32_t, 3> maxWorkgroupCount{};
        std::uint32_t maxTotalWorkgroups = 0;
    };

    enum class VertexFormat : std::uint8_t
    {
        float32,
        float32x2,
        float32x3,
        float32x4,
        uint32,
        uint32x2,
        uint32x3,
        uint32x4,
        sint32,
        sint32x2,
        sint32x3,
        sint32x4,
        uint8x2,
        uint8x4,
        sint8x2,
        sint8x4,
        unorm8x2,
        unorm8x4,
        snorm8x2,
        snorm8x4,
        uint16x2,
        uint16x4,
        sint16x2,
        sint16x4,
        unorm16x2,
        unorm16x4,
        snorm16x2,
        snorm16x4,
        float16x2,
        float16x4,
    };

    inline constexpr std::uint32_t vertexFormatCount = 30;

    struct VertexInputCapabilities final
    {
        std::uint64_t supportedFormats = 0;
        std::uint32_t maxBufferStride = 0;
        std::uint32_t maxAttributeOffset = 0;
        std::uint32_t maxInstanceStepRate = 0;

        [[nodiscard]] constexpr bool supports(VertexFormat format) const noexcept
        {
            const auto index = static_cast<std::uint32_t>(format);
            return index < vertexFormatCount && (supportedFormats & (std::uint64_t{1} << index)) != 0;
        }
    };

    enum class VertexStepMode : std::uint8_t
    {
        vertex,
        instance,
    };

    struct VertexAttribute final
    {
        VertexFormat format = VertexFormat::float32;
        std::uint32_t offset = 0;
        std::uint32_t shaderLocation = 0;
    };

    struct VertexBufferLayout final
    {
        std::uint32_t stride = 0;
        VertexStepMode stepMode = VertexStepMode::vertex;

        std::uint32_t instanceStepRate = 1;
        std::span<const VertexAttribute> attributes = {};
    };

    struct VertexBufferBinding final
    {
        BufferHandle buffer = {};
        std::uint64_t offset = 0;
    };

    struct StreamOutputEntry final
    {
        std::uint32_t stream = 0;
        std::uint32_t shaderLocation = 0;
        std::uint32_t startComponent = 0;
        std::uint32_t componentCount = 0;
        std::uint32_t outputSlot = 0;
    };

    struct StreamOutputBufferBinding final
    {
        BufferHandle buffer = {};
        std::uint64_t offset = 0;
        std::uint64_t size = 0;

        BufferHandle filledSizeBuffer = {};
        std::uint64_t filledSizeOffset = 0;
    };

    enum class PrimitiveTopology : std::uint8_t
    {
        pointList,
        lineList,
        lineStrip,
        triangleList,
        triangleStrip,
        patchList,
        triangleFan,
    };

    using StripIndexFormat = std::optional<IndexFormat>;

    enum class FrontFace : std::uint8_t
    {
        clockwise,
        counterClockwise,
    };

    enum class CullMode : std::uint8_t
    {
        none,
        front,
        back,
    };

    enum class FillMode : std::uint8_t
    {
        solid,
        wireframe,
    };

    enum class ConservativeRasterizationMode : std::uint8_t
    {
        disabled,
        overestimate,
    };

    enum class CompareOperation : std::uint8_t
    {
        never,
        less,
        equal,
        lessEqual,
        greater,
        notEqual,
        greaterEqual,
        always,
    };

    enum class ColorWriteMask : std::uint8_t
    {
        none = 0,
        red = 1U << 0U,
        green = 1U << 1U,
        blue = 1U << 2U,
        alpha = 1U << 3U,
        all = (1U << 0U) | (1U << 1U) | (1U << 2U) | (1U << 3U),
    };

    [[nodiscard]] constexpr ColorWriteMask operator|(ColorWriteMask left, ColorWriteMask right) noexcept
    {
        return static_cast<ColorWriteMask>(static_cast<std::uint8_t>(left) | static_cast<std::uint8_t>(right));
    }

    enum class BlendFactor : std::uint8_t
    {
        zero,
        one,
        source,
        oneMinusSource,
        sourceAlpha,
        oneMinusSourceAlpha,
        destination,
        oneMinusDestination,
        destinationAlpha,
        oneMinusDestinationAlpha,
        sourceAlphaSaturated,
        constant,
        oneMinusConstant,
        source1,
        oneMinusSource1,
        source1Alpha,
        oneMinusSource1Alpha,
    };

    enum class BlendOperation : std::uint8_t
    {
        add,
        subtract,
        reverseSubtract,
        minimum,
        maximum,
    };

    enum class LogicOperation : std::uint8_t
    {
        clear,
        andOperation,
        andReverse,
        copy,
        andInverted,
        noOp,
        xorOperation,
        orOperation,
        nor,
        equivalent,
        invert,
        orReverse,
        copyInverted,
        orInverted,
        nand,
        set,
    };

    struct BlendComponent final
    {
        BlendFactor sourceFactor = BlendFactor::one;
        BlendFactor destinationFactor = BlendFactor::zero;
        BlendOperation operation = BlendOperation::add;
    };

    struct BlendState final
    {
        bool enabled = false;
        BlendComponent color = {};
        BlendComponent alpha = {};
    };

    struct ColorTargetState final
    {
        Format format = Format::unknown;
        ColorWriteMask writeMask = ColorWriteMask::all;
        BlendState blend = {};
    };

    enum class StencilOperation : std::uint8_t
    {
        keep,
        zero,
        replace,
        incrementClamp,
        decrementClamp,
        invert,
        incrementWrap,
        decrementWrap,
    };

    enum class StencilFace : std::uint8_t
    {
        front,
        back,
        frontAndBack,
    };

    struct StencilFaceState final
    {
        CompareOperation compare = CompareOperation::always;
        StencilOperation failOperation = StencilOperation::keep;
        StencilOperation depthFailOperation = StencilOperation::keep;
        StencilOperation passOperation = StencilOperation::keep;
    };

    struct DepthStencilState final
    {
        Format format = Format::unknown;
        bool depthTestEnabled = false;
        bool depthWriteEnabled = false;
        CompareOperation depthCompare = CompareOperation::less;

        bool depthBoundsTestEnabled = false;
        bool stencilTestEnabled = false;
        StencilFaceState frontFace = {};
        StencilFaceState backFace = {};
        std::uint8_t stencilReadMask = 0xFF;
        std::uint8_t stencilWriteMask = 0xFF;
    };

    struct GraphicsPipelineDesc final
    {
        PipelineLayoutHandle layout = {};

        ShaderModuleHandle vertexShader = {};
        ShaderModuleHandle pixelShader = {};

        ShaderModuleHandle geometryShader = {};

        ShaderModuleHandle hullShader = {};
        ShaderModuleHandle domainShader = {};
        ShaderModuleHandle taskShader = {};
        ShaderModuleHandle meshShader = {};
        std::span<const VertexBufferLayout> vertexBuffers = {};
        PrimitiveTopology topology = PrimitiveTopology::triangleList;
        std::uint32_t patchControlPoints = 0;
        StripIndexFormat stripIndexFormat = {};
        FrontFace frontFace = FrontFace::counterClockwise;
        CullMode cullMode = CullMode::back;
        FillMode fillMode = FillMode::solid;
        std::int32_t depthBias = 0;
        float depthBiasClamp = 0.0F;
        float depthBiasSlopeScale = 0.0F;

        bool dynamicDepthBiasEnabled = false;
        bool depthClipEnabled = true;
        std::span<const ColorTargetState> colorTargets = {};
        DepthStencilState depthStencil = {};
        std::uint32_t sampleCount = 1;
        std::uint32_t sampleMask = std::numeric_limits<std::uint32_t>::max();
        bool alphaToCoverageEnabled = false;

        std::optional<LogicOperation> logicOperation = {};

        ConservativeRasterizationMode conservativeRasterization = ConservativeRasterizationMode::disabled;

        std::uint32_t clipDistanceCount = 0;
        std::uint32_t cullDistanceCount = 0;

        std::uint32_t viewMask = 0;

        std::span<const StreamOutputEntry> streamOutputEntries = {};
        std::span<const std::uint32_t> streamOutputBufferStrides = {};

        std::optional<std::uint32_t> rasterizedStream = 0;
        std::string_view label = {};
        PipelineCacheHandle cache = {};
        PipelineCacheKey cacheKey = {};
        PipelineCachePolicy cachePolicy = PipelineCachePolicy::populate;
    };

} // namespace rachis
