#include "Metal/MetalInternal.h"

#include <CommonCrypto/CommonDigest.h>
#include <initializer_list>
#include <type_traits>

namespace rachis::metal
{
    namespace
    {

        static_assert(kMetalShaderCodeDigestBytes == CC_SHA256_DIGEST_LENGTH);

        using MetalSpecializationSet = std::vector<SpecializationConstant>;

        [[nodiscard]] Result<MetalSpecializationSet> metalBuildSpecializationSet(
            std::span<const SpecializationConstant> constants)
        {
            if (constants.size() > maxSpecializationConstants)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "A shader specialization set exceeds the portable constant-count limit."));
            }
            MetalSpecializationSet result(constants.begin(), constants.end());
            for (auto& constant : result)
            {
                switch (constant.format)
                {
                case SpecializationConstantFormat::boolean:
                    constant.valueBits = constant.valueBits != 0 ? 1U : 0U;
                    break;
                case SpecializationConstantFormat::float32:
                case SpecializationConstantFormat::signedInt32:
                case SpecializationConstantFormat::unsignedInt32:
                    break;
                default:
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument, "A shader specialization constant has an invalid format."));
                }
            }
            std::ranges::sort(result, {}, &SpecializationConstant::id);
            if (std::ranges::adjacent_find(result, {}, &SpecializationConstant::id) != result.end())
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "Shader specialization constant IDs must be unique within one value set."));
            }
            return result;
        }

        [[nodiscard]] constexpr std::optional<ShaderResourceType> metalShaderResourceType(BindingType type) noexcept
        {
            switch (type)
            {
            case BindingType::uniformBuffer:
            case BindingType::dynamicUniformBuffer:
                return ShaderResourceType::uniformBuffer;
            case BindingType::readOnlyStorageBuffer:
            case BindingType::dynamicReadOnlyStorageBuffer:
                return ShaderResourceType::readOnlyStorageBuffer;
            case BindingType::storageBuffer:
            case BindingType::dynamicStorageBuffer:
                return ShaderResourceType::storageBuffer;
            case BindingType::sampledTexture:
                return ShaderResourceType::sampledTexture;
            case BindingType::storageTexture:
                return ShaderResourceType::storageTexture;
            case BindingType::sampler:
                return ShaderResourceType::sampler;
            case BindingType::accelerationStructure:
                return ShaderResourceType::accelerationStructure;
            case BindingType::minMipFeedback:
                return ShaderResourceType::minMipFeedback;
            case BindingType::mipRegionUsedFeedback:
                return ShaderResourceType::mipRegionUsedFeedback;
            }
            return std::nullopt;
        }

        [[nodiscard]] Status metalValidatePipelineShaderInterface(const ShaderInterfaceSnapshot& shader,
                                                                  const MetalPipelineLayoutRecord& layout)
        {
            for (const auto& binding : shader.bindings)
            {
                if (binding.set >= layout.sets.size())
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::validationError,
                        "A shader-interface binding references a set absent from the pipeline layout."));
                }
                if (layout.sets[binding.set] == nullptr)
                {
                    return std::unexpected(
                        metalSelectionError(ErrorCode::validationError,
                                            "A shader-interface binding references a pipeline-layout set gap."));
                }
                const auto& entries = layout.sets[binding.set]->entries;
                const auto found = std::ranges::find(entries, binding.binding, &BindingLayoutEntry::binding);
                const auto foundType =
                    found == entries.end() ? std::optional<ShaderResourceType>{} : metalShaderResourceType(found->type);
                if (found == entries.end() || !foundType.has_value() || *foundType != binding.type ||
                    found->count != binding.count || !hasAny(found->visibility, shader.stage))
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::validationError,
                        "A pipeline-layout binding does not match the shader interface's type, count, or visibility."));
                }
            }

            std::array<bool, maxPushConstantBytes / sizeof(std::uint32_t)> requiredWords{};
            std::array<bool, maxPushConstantBytes / sizeof(std::uint32_t)> layoutWords{};
            for (const auto& range : shader.pushConstants)
            {
                const auto first = range.offset / sizeof(std::uint32_t);
                const auto count = range.size / sizeof(std::uint32_t);
                std::ranges::fill(std::span(requiredWords).subspan(first, count), true);
            }
            for (const auto& range : layout.pushConstantRanges)
            {
                if (!hasAny(range.visibility, shader.stage))
                {
                    continue;
                }
                const auto first = range.offset / sizeof(std::uint32_t);
                const auto count = range.size / sizeof(std::uint32_t);
                std::ranges::fill(std::span(layoutWords).subspan(first, count), true);
            }
            if (requiredWords != layoutWords)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::validationError,
                    "The pipeline layout's stage-visible push constants do not exactly match the shader interface."));
            }
            return {};
        }

        [[nodiscard]] constexpr MTLVertexFormat metalVertexFormat(VertexFormat format) noexcept
        {
            switch (format)
            {
            case VertexFormat::float32:
                return MTLVertexFormatFloat;
            case VertexFormat::float32x2:
                return MTLVertexFormatFloat2;
            case VertexFormat::float32x3:
                return MTLVertexFormatFloat3;
            case VertexFormat::float32x4:
                return MTLVertexFormatFloat4;
            case VertexFormat::uint32:
                return MTLVertexFormatUInt;
            case VertexFormat::uint32x2:
                return MTLVertexFormatUInt2;
            case VertexFormat::uint32x3:
                return MTLVertexFormatUInt3;
            case VertexFormat::uint32x4:
                return MTLVertexFormatUInt4;
            case VertexFormat::sint32:
                return MTLVertexFormatInt;
            case VertexFormat::sint32x2:
                return MTLVertexFormatInt2;
            case VertexFormat::sint32x3:
                return MTLVertexFormatInt3;
            case VertexFormat::sint32x4:
                return MTLVertexFormatInt4;
            case VertexFormat::uint8x2:
                return MTLVertexFormatUChar2;
            case VertexFormat::uint8x4:
                return MTLVertexFormatUChar4;
            case VertexFormat::sint8x2:
                return MTLVertexFormatChar2;
            case VertexFormat::sint8x4:
                return MTLVertexFormatChar4;
            case VertexFormat::unorm8x2:
                return MTLVertexFormatUChar2Normalized;
            case VertexFormat::unorm8x4:
                return MTLVertexFormatUChar4Normalized;
            case VertexFormat::snorm8x2:
                return MTLVertexFormatChar2Normalized;
            case VertexFormat::snorm8x4:
                return MTLVertexFormatChar4Normalized;
            case VertexFormat::uint16x2:
                return MTLVertexFormatUShort2;
            case VertexFormat::uint16x4:
                return MTLVertexFormatUShort4;
            case VertexFormat::sint16x2:
                return MTLVertexFormatShort2;
            case VertexFormat::sint16x4:
                return MTLVertexFormatShort4;
            case VertexFormat::unorm16x2:
                return MTLVertexFormatUShort2Normalized;
            case VertexFormat::unorm16x4:
                return MTLVertexFormatUShort4Normalized;
            case VertexFormat::snorm16x2:
                return MTLVertexFormatShort2Normalized;
            case VertexFormat::snorm16x4:
                return MTLVertexFormatShort4Normalized;
            case VertexFormat::float16x2:
                return MTLVertexFormatHalf2;
            case VertexFormat::float16x4:
                return MTLVertexFormatHalf4;
            }
            return MTLVertexFormatInvalid;
        }

        [[nodiscard]] constexpr std::uint32_t metalVertexFormatSize(VertexFormat format) noexcept
        {
            switch (format)
            {
            case VertexFormat::float32:
            case VertexFormat::uint32:
            case VertexFormat::sint32:
                return 4;
            case VertexFormat::float32x2:
            case VertexFormat::uint32x2:
            case VertexFormat::sint32x2:
                return 8;
            case VertexFormat::uint16x2:
            case VertexFormat::sint16x2:
            case VertexFormat::unorm16x2:
            case VertexFormat::snorm16x2:
            case VertexFormat::float16x2:
                return 4;
            case VertexFormat::float32x3:
            case VertexFormat::uint32x3:
            case VertexFormat::sint32x3:
                return 12;
            case VertexFormat::float32x4:
            case VertexFormat::uint32x4:
            case VertexFormat::sint32x4:
                return 16;
            case VertexFormat::uint8x2:
            case VertexFormat::sint8x2:
            case VertexFormat::unorm8x2:
            case VertexFormat::snorm8x2:
                return 2;
            case VertexFormat::uint8x4:
            case VertexFormat::sint8x4:
            case VertexFormat::unorm8x4:
            case VertexFormat::snorm8x4:
                return 4;
            case VertexFormat::uint16x4:
            case VertexFormat::sint16x4:
            case VertexFormat::unorm16x4:
            case VertexFormat::snorm16x4:
            case VertexFormat::float16x4:
                return 8;
            }
            return 0;
        }

        [[nodiscard]] constexpr MTLBlendFactor metalBlendFactor(BlendFactor factor) noexcept
        {
            switch (factor)
            {
            case BlendFactor::zero:
                return MTLBlendFactorZero;
            case BlendFactor::one:
                return MTLBlendFactorOne;
            case BlendFactor::source:
                return MTLBlendFactorSourceColor;
            case BlendFactor::oneMinusSource:
                return MTLBlendFactorOneMinusSourceColor;
            case BlendFactor::sourceAlpha:
                return MTLBlendFactorSourceAlpha;
            case BlendFactor::oneMinusSourceAlpha:
                return MTLBlendFactorOneMinusSourceAlpha;
            case BlendFactor::destination:
                return MTLBlendFactorDestinationColor;
            case BlendFactor::oneMinusDestination:
                return MTLBlendFactorOneMinusDestinationColor;
            case BlendFactor::destinationAlpha:
                return MTLBlendFactorDestinationAlpha;
            case BlendFactor::oneMinusDestinationAlpha:
                return MTLBlendFactorOneMinusDestinationAlpha;
            case BlendFactor::sourceAlphaSaturated:
                return MTLBlendFactorSourceAlphaSaturated;
            case BlendFactor::constant:
                return MTLBlendFactorBlendColor;
            case BlendFactor::oneMinusConstant:
                return MTLBlendFactorOneMinusBlendColor;
            case BlendFactor::source1:
                return MTLBlendFactorSource1Color;
            case BlendFactor::oneMinusSource1:
                return MTLBlendFactorOneMinusSource1Color;
            case BlendFactor::source1Alpha:
                return MTLBlendFactorSource1Alpha;
            case BlendFactor::oneMinusSource1Alpha:
                return MTLBlendFactorOneMinusSource1Alpha;
            }
            return MTLBlendFactorZero;
        }

        [[nodiscard]] constexpr bool metalBlendFactorNeedsDualSource(BlendFactor factor) noexcept
        {
            return factor == BlendFactor::source1 || factor == BlendFactor::oneMinusSource1 ||
                   factor == BlendFactor::source1Alpha || factor == BlendFactor::oneMinusSource1Alpha;
        }

        [[nodiscard]] constexpr bool metalBlendFactorUsesConstant(BlendFactor factor) noexcept
        {
            return factor == BlendFactor::constant || factor == BlendFactor::oneMinusConstant;
        }

        [[nodiscard]] constexpr MTLBlendOperation metalBlendOperation(BlendOperation operation) noexcept
        {
            switch (operation)
            {
            case BlendOperation::add:
                return MTLBlendOperationAdd;
            case BlendOperation::subtract:
                return MTLBlendOperationSubtract;
            case BlendOperation::reverseSubtract:
                return MTLBlendOperationReverseSubtract;
            case BlendOperation::minimum:
                return MTLBlendOperationMin;
            case BlendOperation::maximum:
                return MTLBlendOperationMax;
            }
            return MTLBlendOperationAdd;
        }

        [[nodiscard]] constexpr bool metalIsValidBlendFactor(BlendFactor factor) noexcept
        {
            return factor >= BlendFactor::zero && factor <= BlendFactor::oneMinusSource1Alpha;
        }

        [[nodiscard]] constexpr bool metalIsValidBlendOperation(BlendOperation operation) noexcept
        {
            return operation >= BlendOperation::add && operation <= BlendOperation::maximum;
        }

        [[nodiscard]] constexpr bool metalIsValidBlendComponent(const BlendComponent& component) noexcept
        {
            return metalIsValidBlendFactor(component.sourceFactor) &&
                   metalIsValidBlendFactor(component.destinationFactor) &&
                   metalIsValidBlendOperation(component.operation);
        }

        [[nodiscard]] constexpr bool metalHasColorWrite(ColorWriteMask mask, ColorWriteMask bit) noexcept
        {
            return (static_cast<std::uint8_t>(mask) & static_cast<std::uint8_t>(bit)) != 0;
        }

        [[nodiscard]] constexpr MTLColorWriteMask metalColorWriteMask(ColorWriteMask mask) noexcept
        {
            MTLColorWriteMask native = MTLColorWriteMaskNone;
            if (metalHasColorWrite(mask, ColorWriteMask::red))
            {
                native |= MTLColorWriteMaskRed;
            }
            if (metalHasColorWrite(mask, ColorWriteMask::green))
            {
                native |= MTLColorWriteMaskGreen;
            }
            if (metalHasColorWrite(mask, ColorWriteMask::blue))
            {
                native |= MTLColorWriteMaskBlue;
            }
            if (metalHasColorWrite(mask, ColorWriteMask::alpha))
            {
                native |= MTLColorWriteMaskAlpha;
            }
            return native;
        }

        [[nodiscard]] constexpr MTLPrimitiveTopologyClass metalPrimitiveTopologyClass(
            PrimitiveTopology topology) noexcept
        {
            switch (topology)
            {
            case PrimitiveTopology::pointList:
                return MTLPrimitiveTopologyClassPoint;
            case PrimitiveTopology::lineList:
            case PrimitiveTopology::lineStrip:
                return MTLPrimitiveTopologyClassLine;
            case PrimitiveTopology::triangleList:
            case PrimitiveTopology::triangleStrip:
                return MTLPrimitiveTopologyClassTriangle;
            case PrimitiveTopology::patchList:
            case PrimitiveTopology::triangleFan:
                return MTLPrimitiveTopologyClassUnspecified;
            }
            return MTLPrimitiveTopologyClassUnspecified;
        }

        [[nodiscard]] constexpr bool metalIsValidSampleCount(std::uint32_t sampleCount) noexcept
        {
            return sampleCount == 1 || sampleCount == 2 || sampleCount == 4 || sampleCount == 8;
        }

        [[nodiscard]] Status metalValidatePipelineCacheRequest(PipelineCachePolicy policy,
                                                               PipelineCacheHandle cache,
                                                               PipelineCacheKey key)
        {
            if (policy != PipelineCachePolicy::populate && policy != PipelineCachePolicy::requireHit)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "A pipeline cache policy must be populate or requireHit."));
            }
            const bool hasCache = cache.isValid();
            const bool hasKey = key.isValid();
            if (hasCache != hasKey)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "A pipeline cache and cache key must be provided together."));
            }
            if (policy == PipelineCachePolicy::requireHit && !hasCache)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "requireHit cannot compile a missing Metal pipeline-cache entry."));
            }
            return {};
        }

        inline constexpr std::uint64_t metalPipelineCacheBlobMagic = 0x3130544D43494852ULL;
        inline constexpr std::uint32_t metalPipelineCacheBlobVersion = 2;
        inline constexpr std::uint64_t metalPipelineCacheChecksumOffset = 14695981039346656037ULL;
        inline constexpr std::uint64_t metalPipelineCacheChecksumPrime = 1099511628211ULL;

        struct MetalPipelineCacheBlobHeader final
        {
            std::uint64_t magic = metalPipelineCacheBlobMagic;
            std::uint64_t nativeDataSize = 0;
            std::uint64_t deviceRegistryId = 0;
            std::uint32_t version = metalPipelineCacheBlobVersion;
            std::uint32_t entryCount = 0;
            std::uint32_t segmentCount = 0;
            std::uint32_t reserved = 0;
            std::uint64_t checksum = 0;
        };

        struct MetalPipelineCacheBlobEntry final
        {
            PipelineCacheKey key{};
            std::uint32_t kind = 0;
            std::uint32_t segment = 0;
        };

        static_assert(sizeof(MetalPipelineCacheBlobHeader) == 48);
        static_assert(sizeof(MetalPipelineCacheBlobEntry) == 24);

        struct MetalPipelineCacheIndexedEntry final
        {
            MetalPipelineCacheIdentity identity{};
            std::uint32_t segment = 0;
        };

        struct MetalDecodedPipelineCache final
        {
            std::vector<MetalPipelineCacheIndexedEntry> entries;
            std::vector<std::vector<std::byte>> segments;
        };

        [[nodiscard]] std::uint64_t metalPipelineCacheChecksum(const MetalPipelineCacheBlobHeader& header,
                                                               std::span<const std::byte> payload) noexcept
        {
            std::uint64_t hash = metalPipelineCacheChecksumOffset;
            const auto absorbByte = [&](std::uint8_t value) noexcept {
                hash ^= value;
                hash *= metalPipelineCacheChecksumPrime;
            };
            const auto absorb64 = [&](std::uint64_t value) noexcept {
                for (int shift = 0; shift < 64; shift += 8)
                {
                    absorbByte(static_cast<std::uint8_t>(value >> shift));
                }
            };
            absorb64(header.deviceRegistryId);
            absorb64(header.version);
            absorb64(header.entryCount);
            absorb64(header.segmentCount);
            absorb64(header.reserved);
            for (const auto byte : payload)
            {
                absorbByte(static_cast<std::uint8_t>(byte));
            }
            return hash;
        }

        [[nodiscard]] Result<MetalDecodedPipelineCache> metalDecodePipelineCacheBlob(std::span<const std::byte> data,
                                                                                     std::uint64_t deviceRegistryId)
        {
            if (data.empty())
            {
                return MetalDecodedPipelineCache{};
            }
            if (data.size() < sizeof(MetalPipelineCacheBlobHeader))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "The pipeline cache blob is smaller than the RHI Metal header."));
            }
            MetalPipelineCacheBlobHeader header{};
            std::memcpy(&header, data.data(), sizeof(header));
            if (header.magic != metalPipelineCacheBlobMagic || header.version != metalPipelineCacheBlobVersion)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "The pipeline cache blob has an incompatible RHI magic or schema version."));
            }
            if (header.deviceRegistryId != deviceRegistryId)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "The pipeline cache blob does not match this Metal device."));
            }
            const auto remaining = data.size() - sizeof(header);
            if (header.entryCount > remaining / sizeof(MetalPipelineCacheBlobEntry))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "The pipeline cache blob entry table exceeds the supplied data."));
            }
            const auto entriesSize = static_cast<std::size_t>(header.entryCount) * sizeof(MetalPipelineCacheBlobEntry);
            if (header.segmentCount > (remaining - entriesSize) / sizeof(std::uint64_t))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "The pipeline cache blob segment table exceeds the supplied data."));
            }
            const auto segmentTableOffset = sizeof(header) + entriesSize;
            const auto nativeOffset =
                segmentTableOffset + static_cast<std::size_t>(header.segmentCount) * sizeof(std::uint64_t);
            if (header.nativeDataSize != data.size() - nativeOffset)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "The pipeline cache blob has an inconsistent native payload size."));
            }
            const auto payload = data.subspan(sizeof(header));
            if (header.checksum != metalPipelineCacheChecksum(header, payload))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "The pipeline cache blob checksum does not match the payload."));
            }
            MetalDecodedPipelineCache decoded;
            decoded.entries.reserve(header.entryCount);
            std::vector<bool> referencedSegments(header.segmentCount, false);
            for (std::uint32_t index = 0; index < header.entryCount; ++index)
            {
                MetalPipelineCacheBlobEntry entry{};
                std::memcpy(&entry, data.data() + sizeof(header) + index * sizeof(entry), sizeof(entry));
                if (!entry.key.isValid() || entry.segment >= header.segmentCount ||
                    (entry.kind != static_cast<std::uint32_t>(MetalPipelineCacheKind::compute) &&
                     entry.kind != static_cast<std::uint32_t>(MetalPipelineCacheKind::graphics)))
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument, "The pipeline cache blob contains an invalid or duplicate entry."));
                }
                const MetalPipelineCacheIdentity identity{
                    .key = entry.key,
                    .kind = static_cast<MetalPipelineCacheKind>(entry.kind),
                };
                if (std::ranges::find(decoded.entries, identity, &MetalPipelineCacheIndexedEntry::identity) !=
                    decoded.entries.end())
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument, "The pipeline cache blob contains an invalid or duplicate entry."));
                }
                referencedSegments[entry.segment] = true;
                decoded.entries.push_back({.identity = identity, .segment = entry.segment});
            }
            decoded.segments.reserve(header.segmentCount);
            auto segmentOffset = nativeOffset;
            for (std::uint32_t index = 0; index < header.segmentCount; ++index)
            {
                std::uint64_t segmentSize = 0;
                std::memcpy(
                    &segmentSize, data.data() + segmentTableOffset + index * sizeof(segmentSize), sizeof(segmentSize));
                if (segmentSize == 0 || segmentSize > data.size() - segmentOffset || !referencedSegments[index])
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument, "The pipeline cache blob has an invalid archive segment table."));
                }
                const auto first = data.begin() + static_cast<std::ptrdiff_t>(segmentOffset);
                decoded.segments.emplace_back(first, first + static_cast<std::ptrdiff_t>(segmentSize));
                segmentOffset += static_cast<std::size_t>(segmentSize);
            }
            if (header.reserved != 0 || segmentOffset != data.size())
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "The pipeline cache blob has an invalid archive segment table."));
            }
            return decoded;
        }

        [[nodiscard]] std::vector<std::byte> metalEncodePipelineCacheBlob(
            const std::vector<MetalPipelineCacheIndexedEntry>& entries,
            const std::vector<std::shared_ptr<MetalPipelineCacheSegment>>& segments,
            std::uint64_t deviceRegistryId)
        {
            std::size_t nativeSize = 0;
            for (const auto& segment : segments)
            {
                nativeSize += segment->bytes.size();
            }
            MetalPipelineCacheBlobHeader header{
                .magic = metalPipelineCacheBlobMagic,
                .nativeDataSize = nativeSize,
                .deviceRegistryId = deviceRegistryId,
                .version = metalPipelineCacheBlobVersion,
                .entryCount = static_cast<std::uint32_t>(entries.size()),
                .segmentCount = static_cast<std::uint32_t>(segments.size()),
            };
            const auto segmentTableOffset = sizeof(header) + entries.size() * sizeof(MetalPipelineCacheBlobEntry);
            const auto nativeOffset = segmentTableOffset + segments.size() * sizeof(std::uint64_t);
            std::vector<std::byte> blob(nativeOffset + nativeSize);
            for (std::size_t index = 0; index < entries.size(); ++index)
            {
                MetalPipelineCacheBlobEntry entry{
                    .key = entries[index].identity.key,
                    .kind = static_cast<std::uint32_t>(entries[index].identity.kind),
                    .segment = entries[index].segment,
                };
                std::memcpy(blob.data() + sizeof(header) + index * sizeof(entry), &entry, sizeof(entry));
            }
            auto segmentOffset = nativeOffset;
            for (std::size_t index = 0; index < segments.size(); ++index)
            {
                const auto& bytes = segments[index]->bytes;
                const std::uint64_t segmentSize = bytes.size();
                std::memcpy(
                    blob.data() + segmentTableOffset + index * sizeof(segmentSize), &segmentSize, sizeof(segmentSize));
                if (!bytes.empty())
                {
                    std::memcpy(blob.data() + segmentOffset, bytes.data(), bytes.size());
                }
                segmentOffset += bytes.size();
            }
            header.checksum = metalPipelineCacheChecksum(header, std::span{blob}.subspan(sizeof(header)));
            std::memcpy(blob.data(), &header, sizeof(header));
            return blob;
        }

        [[nodiscard]] NSString* metalPipelineCacheTemporaryDirectory()
        {
#ifdef RACHIS_TESTING
            {
                std::scoped_lock const lock(metalTestingState().mutex);
                if (!metalTestingState().pipelineCacheTemporaryDirectory.empty())
                {
                    return metalNsString(metalTestingState().pipelineCacheTemporaryDirectory);
                }
            }
#endif
            return NSTemporaryDirectory();
        }

        [[nodiscard]] bool metalConsumePipelineCacheSerializeFailureInjection() noexcept
        {
            bool fail = false;
#ifdef RACHIS_TESTING
            {
                std::scoped_lock const lock(metalTestingState().mutex);
                fail = metalTestingState().failNextPipelineCacheSerialize;
                metalTestingState().failNextPipelineCacheSerialize = false;
            }
#endif
            return fail;
        }

        [[nodiscard]] NSURL* metalPipelineCacheTemporaryURL()
        {
            NSString* const directory = metalPipelineCacheTemporaryDirectory();
            NSString* const name =
                [NSString stringWithFormat:@"rhi-metal-pipeline-cache-%@.bin", [NSUUID UUID].UUIDString];
            return [NSURL fileURLWithPath:[directory stringByAppendingPathComponent:name] isDirectory:NO];
        }

        class MetalPipelineCacheTemporaryFile final
        {
        public:
            MetalPipelineCacheTemporaryFile() : url_(metalPipelineCacheTemporaryURL())
            {
            }

            ~MetalPipelineCacheTemporaryFile()
            {
                if (url_ != nil)
                {
                    [[NSFileManager defaultManager] removeItemAtURL:url_ error:nil];
                }
            }

            MetalPipelineCacheTemporaryFile(const MetalPipelineCacheTemporaryFile&) = delete;
            MetalPipelineCacheTemporaryFile& operator=(const MetalPipelineCacheTemporaryFile&) = delete;
            MetalPipelineCacheTemporaryFile(MetalPipelineCacheTemporaryFile&&) = delete;
            MetalPipelineCacheTemporaryFile& operator=(MetalPipelineCacheTemporaryFile&&) = delete;

            [[nodiscard]] NSURL* url() const noexcept
            {
                return url_;
            }

            [[nodiscard]] NSURL* keep() noexcept
            {
                NSURL* const kept = url_;
                url_ = nil;
                return kept;
            }

        private:
            __strong NSURL* url_ = nil;
        };

        [[nodiscard]] Result<id<MTL4Archive>> metalLoadPipelineCacheArchive(id<MTLDevice> device,
                                                                            std::span<const std::byte> bytes,
                                                                            NSURL** keptURL)
        {
            if (keptURL != nullptr)
            {
                *keptURL = nil;
            }
            if (device == nil || bytes.empty())
            {
                return nil;
            }
            MetalPipelineCacheTemporaryFile file;
            NSData* const data = [NSData dataWithBytes:bytes.data() length:bytes.size()];
            NSError* writeError = nil;
            if (![data writeToURL:file.url() options:NSDataWritingAtomic error:&writeError])
            {
                return std::unexpected(metalNsError(ErrorCode::nativeFailure, writeError, "writeToURL"));
            }
            NSError* archiveError = nil;
            id<MTL4Archive> const archive = [device newArchiveWithURL:file.url() error:&archiveError];
            if (archive == nil)
            {
                return std::unexpected(metalNsError(ErrorCode::invalidArgument, archiveError, "newArchiveWithURL"));
            }
            if (keptURL != nullptr)
            {
                *keptURL = file.keep();
            }
            return archive;
        }

        [[nodiscard]] bool metalSerializePipelineCacheArchiveToURL(id<MTL4PipelineDataSetSerializer> serializer,
                                                                   NSURL* url,
                                                                   NSError* __autoreleasing* error)
        {
            if (metalConsumePipelineCacheSerializeFailureInjection())
            {
                static_cast<void>([[NSData data] writeToURL:url atomically:NO]);
                return false;
            }
            return [serializer serializeAsArchiveAndFlushToURL:url error:error];
        }

        [[nodiscard]] Result<std::vector<std::byte>> metalSerializePipelineCacheArchive(
            id<MTL4PipelineDataSetSerializer> serializer)
        {
            if (serializer == nil)
            {
                return std::vector<std::byte>{};
            }
            MetalPipelineCacheTemporaryFile file;
            NSError* error = nil;
            if (!metalSerializePipelineCacheArchiveToURL(serializer, file.url(), &error))
            {
                return std::unexpected(
                    metalNsError(ErrorCode::nativeFailure, error, "serializeAsArchiveAndFlushToURL"));
            }
            NSData* const data = [NSData dataWithContentsOfURL:file.url() options:0 error:&error];
            if (data == nil)
            {
                return std::unexpected(metalNsError(ErrorCode::nativeFailure, error, "dataWithContentsOfURL"));
            }
            const auto* bytes = static_cast<const std::byte*>(data.bytes);
            return std::vector<std::byte>(bytes, bytes + data.length);
        }

        [[nodiscard]] Status metalFlushPipelineCacheSerializer(MetalPipelineCacheRecord& record)
        {
            auto native = metalSerializePipelineCacheArchive(record.serializer);
            if (!native)
            {
                return std::unexpected(std::move(native.error()));
            }
            std::shared_ptr<MetalPipelineCacheSegment> segment;
            if (!native->empty())
            {
                segment = std::make_shared<MetalPipelineCacheSegment>();
                segment->bytes = std::move(*native);
                NSURL* keptURL = nil;
                auto archive = metalLoadPipelineCacheArchive(
                    record.compiler != nil ? record.compiler.device : nil, segment->bytes, &keptURL);
                if (archive)
                {
                    segment->archive = *archive;
                    segment->archiveFileURL = keptURL;
                }
                record.segments.push_back(segment);
            }
            for (auto& [identity, live] : record.live)
            {
                static_cast<void>(identity);
                if (live.inSerializer && segment != nullptr)
                {
                    live.segment = segment;
                }
                live.inSerializer = false;
            }
            std::erase_if(record.segments, [&record](const std::shared_ptr<MetalPipelineCacheSegment>& candidate) {
                return std::ranges::none_of(
                    record.live, [&candidate](const auto& item) { return item.second.segment == candidate; });
            });
            record.serializerDirty = false;
            return {};
        }

        [[nodiscard]] std::vector<MetalPipelineCacheIndexedEntry> metalPipelineCacheBlobEntries(
            const MetalPipelineCacheRecord& record)
        {
            std::map<const MetalPipelineCacheSegment*, std::uint32_t> segmentIndices;
            for (std::size_t index = 0; index < record.segments.size(); ++index)
            {
                segmentIndices.emplace(record.segments[index].get(), static_cast<std::uint32_t>(index));
            }
            std::vector<MetalPipelineCacheIndexedEntry> entries;
            entries.reserve(record.live.size());
            for (const auto& [identity, live] : record.live)
            {
                const auto found = segmentIndices.find(live.segment.get());
                if (found != segmentIndices.end())
                {
                    entries.push_back({.identity = identity, .segment = found->second});
                }
            }
            return entries;
        }

        [[nodiscard]] NSArray<id<MTL4Archive>>* metalPipelineCacheLookupArchives(const MetalPipelineCacheRecord& cache)
        {
            NSMutableArray<id<MTL4Archive>>* const archives = [NSMutableArray arrayWithCapacity:cache.segments.size()];
            for (const auto& segment : cache.segments)
            {
                if (segment->archive != nil)
                {
                    [archives addObject:segment->archive];
                }
            }
            return archives;
        }

        [[nodiscard]] MTL4LibraryFunctionDescriptor* metalLibraryFunctionDescriptor(id<MTLLibrary> library,
                                                                                    std::string_view name)
        {
            MTL4LibraryFunctionDescriptor* const descriptor = [MTL4LibraryFunctionDescriptor new];
            descriptor.library = library;
            descriptor.name = [[NSString alloc] initWithBytes:name.data()
                                                       length:name.size()
                                                     encoding:NSUTF8StringEncoding];
            return descriptor;
        }

        [[nodiscard]] Status metalValidateWorkgroupSize(const ShaderWorkgroupSize& workgroup,
                                                        const DeviceCapabilities& capabilities)
        {
            if (workgroup.x == 0 || workgroup.y == 0 || workgroup.z == 0)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "A compute pipeline requires a non-zero workgroup size."));
            }
            const auto& limit = capabilities.limits.maxComputeWorkgroupSize;
            const auto invocations = static_cast<std::uint64_t>(workgroup.x) * workgroup.y * workgroup.z;
            if (workgroup.x > limit[0] || workgroup.y > limit[1] || workgroup.z > limit[2] ||
                invocations > capabilities.limits.maxComputeWorkgroupInvocations)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported, "The compute workgroup exceeds the Metal device threadgroup limits."));
            }
            return {};
        }

        [[nodiscard]] std::vector<std::uint8_t> metalRequiredBindingSets(const MetalPipelineLayoutRecord& layout)
        {
            std::vector<std::uint8_t> required;
            required.reserve(layout.sets.size());
            for (const auto& set : layout.sets)
            {
                required.push_back(set != nullptr && !set->entries.empty() ? std::uint8_t{1} : std::uint8_t{0});
            }
            return required;
        }

        [[nodiscard]] Status metalValidateVertexInterface(const ShaderInterfaceSnapshot& shader,
                                                          std::span<const std::uint32_t> declaredLocations)
        {
            std::vector<std::uint32_t> declared{declaredLocations.begin(), declaredLocations.end()};
            std::ranges::sort(declared);
            auto expected = shader.vertexInputLocations;
            std::ranges::sort(expected);
            if (declared != expected)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::validationError,
                    "The graphics pipeline's vertex locations do not exactly match the shader interface."));
            }
            return {};
        }

        [[nodiscard]] Result<std::vector<std::uint32_t>> metalCollectVertexLocations(
            std::span<const VertexBufferLayout> vertexBuffers, const VertexInputCapabilities& caps)
        {
            if (vertexBuffers.size() > maxVertexBuffers)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Graphics pipelines require valid attachment, rasterization, and multisample state."));
            }
            std::vector<std::uint32_t> locations;
            locations.reserve(maxVertexAttributes);
            std::uint32_t attributeCount = 0;
            for (const auto& buffer : vertexBuffers)
            {
                if (buffer.attributes.size() > maxVertexAttributes - attributeCount)
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument,
                        "Graphics pipelines require valid attachment, rasterization, and multisample state."));
                }
                if (!buffer.attributes.empty() &&
                    (buffer.stride == 0 || buffer.stride > caps.maxBufferStride ||
                     (buffer.stepMode == VertexStepMode::vertex && buffer.instanceStepRate != 1) ||
                     (buffer.stepMode == VertexStepMode::instance &&
                      (buffer.instanceStepRate == 0 || buffer.instanceStepRate > caps.maxInstanceStepRate))))
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument,
                        "Graphics pipelines require valid attachment, rasterization, and multisample state."));
                }
                for (const auto& attribute : buffer.attributes)
                {
                    if (attribute.shaderLocation >= maxVertexAttributes || !caps.supports(attribute.format) ||
                        attribute.offset > caps.maxAttributeOffset || attribute.offset > buffer.stride ||
                        metalVertexFormatSize(attribute.format) > buffer.stride - attribute.offset ||
                        metalVertexFormat(attribute.format) == MTLVertexFormatInvalid)
                    {
                        return std::unexpected(metalSelectionError(
                            ErrorCode::invalidArgument,
                            "Graphics pipelines require valid attachment, rasterization, and multisample state."));
                    }
                    locations.push_back(attribute.shaderLocation);
                    ++attributeCount;
                }
            }
            auto sorted = locations;
            std::ranges::sort(sorted);
            if (std::ranges::adjacent_find(sorted) != sorted.end())
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "Graphics vertex attribute shader locations must be unique."));
            }
            return locations;
        }

        [[nodiscard]] MTLVertexDescriptor* metalMakeVertexDescriptor(std::span<const VertexBufferLayout> vertexBuffers,
                                                                     std::uint32_t bufferBaseIndex)
        {
            if (vertexBuffers.empty())
            {
                return nil;
            }
            bool anyAttribute = false;
            for (const auto& buffer : vertexBuffers)
            {
                anyAttribute = anyAttribute || !buffer.attributes.empty();
            }
            if (!anyAttribute)
            {
                return nil;
            }
            MTLVertexDescriptor* const descriptor = [MTLVertexDescriptor vertexDescriptor];
            for (NSUInteger bufferIndex = 0; bufferIndex < vertexBuffers.size(); ++bufferIndex)
            {
                const auto& buffer = vertexBuffers[bufferIndex];
                if (buffer.attributes.empty())
                {
                    continue;
                }
                auto* const layout = descriptor.layouts[bufferBaseIndex + bufferIndex];
                layout.stride = buffer.stride;
                layout.stepFunction = buffer.stepMode == VertexStepMode::instance ? MTLVertexStepFunctionPerInstance
                                                                                  : MTLVertexStepFunctionPerVertex;
                layout.stepRate = buffer.instanceStepRate;
                for (const auto& attribute : buffer.attributes)
                {
                    auto* const native = descriptor.attributes[attribute.shaderLocation];
                    native.format = metalVertexFormat(attribute.format);
                    native.offset = attribute.offset;
                    native.bufferIndex = bufferBaseIndex + bufferIndex;
                }
            }
            return descriptor;
        }

        [[nodiscard]] constexpr bool metalIsStripTopology(PrimitiveTopology topology) noexcept
        {
            return topology == PrimitiveTopology::lineStrip || topology == PrimitiveTopology::triangleStrip;
        }

        [[nodiscard]] bool metalHasInvalidAttachmentOrRasterState(const GraphicsPipelineDesc& desc,
                                                                  const DeviceCapabilities& capabilities)
        {
            const bool usesStreamOutput = !desc.streamOutputEntries.empty();
            const bool validSampleCount = metalIsValidSampleCount(desc.sampleCount);
            const auto activeSampleMask = validSampleCount ? (1U << desc.sampleCount) - 1U : 0U;
            return desc.colorTargets.size() > maxColorAttachments || desc.vertexBuffers.size() > maxVertexBuffers ||
                   (!usesStreamOutput && desc.colorTargets.empty() && desc.depthStencil.format == Format::unknown) ||
                   !validSampleCount || (desc.fillMode != FillMode::solid && desc.fillMode != FillMode::wireframe) ||
                   (desc.conservativeRasterization != ConservativeRasterizationMode::disabled &&
                    desc.conservativeRasterization != ConservativeRasterizationMode::overestimate) ||
                   desc.clipDistanceCount > capabilities.limits.maxClipDistances ||
                   desc.cullDistanceCount > capabilities.limits.maxCullDistances ||
                   desc.clipDistanceCount + desc.cullDistanceCount >
                       capabilities.limits.maxCombinedClipAndCullDistances ||
                   !std::isfinite(desc.depthBiasClamp) || !std::isfinite(desc.depthBiasSlopeScale) ||
                   (desc.dynamicDepthBiasEnabled &&
                    (desc.depthBias != 0 || desc.depthBiasClamp != 0.0F || desc.depthBiasSlopeScale != 0.0F)) ||
                   (desc.depthStencil.depthBoundsTestEnabled && desc.depthStencil.format == Format::unknown) ||
                   (desc.sampleMask & activeSampleMask) == 0 || (desc.alphaToCoverageEnabled && desc.sampleCount == 1);
        }

        [[nodiscard]] Status metalValidateGraphicsShaderStages(const GraphicsPipelineDesc& desc)
        {
            const bool hasTaskShader = desc.taskShader.isValid();
            const bool hasMeshShader = desc.meshShader.isValid();
            const bool usesMeshPipeline = hasMeshShader;
            if ((hasTaskShader && !hasMeshShader) ||
                (usesMeshPipeline && (desc.vertexShader.isValid() || desc.geometryShader.isValid() ||
                                      desc.hullShader.isValid() || desc.domainShader.isValid() ||
                                      !desc.vertexBuffers.empty() || desc.topology != PrimitiveTopology::triangleList ||
                                      desc.patchControlPoints != 0 || desc.stripIndexFormat.has_value())))
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "Mesh graphics requires a mesh module, an optional task "
                                                           "module, and no conventional input stages or state."));
            }
            if (usesMeshPipeline)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::unsupported, "The device does not support mesh shaders."));
            }
            if (hasTaskShader)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::unsupported, "The device does not support task shaders."));
            }
            if (desc.geometryShader.isValid())
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::unsupported, "The device does not support geometry shaders."));
            }
            const bool hasHullShader = desc.hullShader.isValid();
            const bool hasDomainShader = desc.domainShader.isValid();
            const bool hasTessellationPair = hasHullShader && hasDomainShader;
            const bool validOrdinaryTopology = desc.topology != PrimitiveTopology::patchList && !hasHullShader &&
                                               !hasDomainShader && desc.patchControlPoints == 0;
            const bool validPatchTopology = desc.topology == PrimitiveTopology::patchList && hasTessellationPair &&
                                            desc.patchControlPoints >= 1 && desc.patchControlPoints <= 32;
            if (!validOrdinaryTopology && !validPatchTopology)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "Tessellation requires paired hull and domain shaders, "
                                                           "patch-list topology, and 1-32 control points."));
            }
            if (hasTessellationPair)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::unsupported, "The device does not support tessellation shaders."));
            }
            return {};
        }

        [[nodiscard]] Status metalValidateSupportedRasterFeatures(const GraphicsPipelineDesc& desc)
        {
            if (desc.topology == PrimitiveTopology::triangleFan)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::unsupported, "The device does not support triangle-fan topology."));
            }
            if (desc.fillMode == FillMode::wireframe)
            {
                return std::unexpected(metalSelectionError(ErrorCode::unsupported,
                                                           "The device does not support wireframe rasterization."));
            }
            if (!desc.depthClipEnabled)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::unsupported, "The device does not support unclipped depth."));
            }
            if (desc.dynamicDepthBiasEnabled)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::unsupported, "The device does not support dynamic depth bias."));
            }
            if (desc.depthStencil.depthBoundsTestEnabled)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::unsupported, "The device does not support depth-bounds testing."));
            }
            if (desc.viewMask != 0)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::unsupported, "The device does not support multiview rendering."));
            }
            if (!desc.streamOutputEntries.empty())
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::unsupported, "The device does not support stream output."));
            }
            if (desc.logicOperation.has_value())
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::unsupported, "The device does not support logic operations."));
            }
            if (desc.conservativeRasterization != ConservativeRasterizationMode::disabled)
            {
                return std::unexpected(metalSelectionError(ErrorCode::unsupported,
                                                           "The device does not support conservative rasterization."));
            }
            const auto activeSampleMask = (1U << desc.sampleCount) - 1U;
            if ((desc.sampleMask & activeSampleMask) != activeSampleMask)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported, "The device does not support sample masks that exclude active samples."));
            }
            return {};
        }

        [[nodiscard]] Status metalValidatePrimitiveRestart(const GraphicsPipelineDesc& desc)
        {
            if (desc.stripIndexFormat.has_value() &&
                (!metalIsStripTopology(desc.topology) ||
                 (*desc.stripIndexFormat != IndexFormat::uint16 && *desc.stripIndexFormat != IndexFormat::uint32)))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "Primitive restart requires a strip topology and a valid index format."));
            }
            return {};
        }

        [[nodiscard]] Status metalValidateGraphicsPipelineDesc(const GraphicsPipelineDesc& desc,
                                                               const DeviceCapabilities& capabilities)
        {
            if (metalHasInvalidAttachmentOrRasterState(desc, capabilities))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Graphics pipelines require valid attachment, rasterization, and multisample state."));
            }
            if (auto cache = metalValidatePipelineCacheRequest(desc.cachePolicy, desc.cache, desc.cacheKey); !cache)
            {
                return cache;
            }
            if (auto stages = metalValidateGraphicsShaderStages(desc); !stages)
            {
                return stages;
            }
            if (auto raster = metalValidateSupportedRasterFeatures(desc); !raster)
            {
                return raster;
            }
            if (auto restart = metalValidatePrimitiveRestart(desc); !restart)
            {
                return restart;
            }
            if (metalPrimitiveTopologyClass(desc.topology) == MTLPrimitiveTopologyClassUnspecified)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Graphics pipelines require valid attachment, rasterization, and multisample state."));
            }
            return {};
        }

        [[nodiscard]] constexpr MTLCompareFunction metalCompareFunction(CompareOperation operation) noexcept
        {
            switch (operation)
            {
            case CompareOperation::never:
                return MTLCompareFunctionNever;
            case CompareOperation::less:
                return MTLCompareFunctionLess;
            case CompareOperation::equal:
                return MTLCompareFunctionEqual;
            case CompareOperation::lessEqual:
                return MTLCompareFunctionLessEqual;
            case CompareOperation::greater:
                return MTLCompareFunctionGreater;
            case CompareOperation::notEqual:
                return MTLCompareFunctionNotEqual;
            case CompareOperation::greaterEqual:
                return MTLCompareFunctionGreaterEqual;
            case CompareOperation::always:
                return MTLCompareFunctionAlways;
            }
            return MTLCompareFunctionAlways;
        }

        [[nodiscard]] constexpr MTLStencilOperation metalStencilOperation(StencilOperation operation) noexcept
        {
            switch (operation)
            {
            case StencilOperation::keep:
                return MTLStencilOperationKeep;
            case StencilOperation::zero:
                return MTLStencilOperationZero;
            case StencilOperation::replace:
                return MTLStencilOperationReplace;
            case StencilOperation::incrementClamp:
                return MTLStencilOperationIncrementClamp;
            case StencilOperation::decrementClamp:
                return MTLStencilOperationDecrementClamp;
            case StencilOperation::invert:
                return MTLStencilOperationInvert;
            case StencilOperation::incrementWrap:
                return MTLStencilOperationIncrementWrap;
            case StencilOperation::decrementWrap:
                return MTLStencilOperationDecrementWrap;
            }
            return MTLStencilOperationKeep;
        }

        [[nodiscard]] constexpr bool metalIsValidCompareOperation(CompareOperation operation) noexcept
        {
            return operation >= CompareOperation::never && operation <= CompareOperation::always;
        }

        [[nodiscard]] constexpr bool metalIsValidStencilOperation(StencilOperation operation) noexcept
        {
            return operation >= StencilOperation::keep && operation <= StencilOperation::decrementWrap;
        }

        [[nodiscard]] constexpr bool metalIsValidStencilFace(const StencilFaceState& face) noexcept
        {
            return metalIsValidCompareOperation(face.compare) && metalIsValidStencilOperation(face.failOperation) &&
                   metalIsValidStencilOperation(face.depthFailOperation) &&
                   metalIsValidStencilOperation(face.passOperation);
        }

        [[nodiscard]] MTLStencilDescriptor* metalMakeStencilFace(const StencilFaceState& face,
                                                                 std::uint8_t readMask,
                                                                 std::uint8_t writeMask)
        {
            MTLStencilDescriptor* const descriptor = [MTLStencilDescriptor new];
            descriptor.stencilCompareFunction = metalCompareFunction(face.compare);
            descriptor.stencilFailureOperation = metalStencilOperation(face.failOperation);
            descriptor.depthFailureOperation = metalStencilOperation(face.depthFailOperation);
            descriptor.depthStencilPassOperation = metalStencilOperation(face.passOperation);
            descriptor.readMask = readMask;
            descriptor.writeMask = writeMask;
            return descriptor;
        }

        [[nodiscard]] bool metalConsumeDepthStencilStateFailureInjection()
        {
            bool fail = false;
#ifdef RACHIS_TESTING
            {
                std::scoped_lock const lock(metalTestingState().mutex);
                fail = metalTestingState().failNextDepthStencilStateCreate;
                metalTestingState().failNextDepthStencilStateCreate = false;
            }
#endif
            return fail;
        }

        [[nodiscard]] id<MTLDepthStencilState> metalMakeDepthStencilState(id<MTLDevice> device,
                                                                          const DepthStencilState& state)
        {
            if (metalConsumeDepthStencilStateFailureInjection())
            {
                return nil;
            }
            MTLDepthStencilDescriptor* const descriptor = [MTLDepthStencilDescriptor new];
            descriptor.depthCompareFunction =
                state.depthTestEnabled ? metalCompareFunction(state.depthCompare) : MTLCompareFunctionAlways;
            descriptor.depthWriteEnabled = state.depthTestEnabled && state.depthWriteEnabled;
            if (state.stencilTestEnabled)
            {
                descriptor.frontFaceStencil =
                    metalMakeStencilFace(state.frontFace, state.stencilReadMask, state.stencilWriteMask);
                descriptor.backFaceStencil =
                    metalMakeStencilFace(state.backFace, state.stencilReadMask, state.stencilWriteMask);
            }
            return [device newDepthStencilStateWithDescriptor:descriptor];
        }

        [[nodiscard]] Result<id<MTLDepthStencilState>> metalMakePipelineDepthStencilState(
            id<MTLDevice> device, const DepthStencilState& state)
        {
            if (state.format == Format::unknown)
            {
                return nil;
            }
            id<MTLDepthStencilState> const created = metalMakeDepthStencilState(device, state);
            if (created == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::outOfMemory,
                    "newDepthStencilStateWithDescriptor failed. Partial Metal objects were released."));
            }
            return created;
        }

    } // namespace

    Result<ShaderLibraryHandle> MetalDevice::createShaderLibrary(const ShaderLibraryDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (desc.entries.empty())
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "A shader library requires at least one staged bytecode entry."));
        }
        auto record = std::make_shared<MetalShaderLibraryRecord>();
        record->label = std::string(desc.label);
        record->entries.reserve(desc.entries.size());
        for (const auto& entry : desc.entries)
        {
            const auto stageBits = static_cast<std::uint16_t>(entry.stage);
            const bool singleStage = stageBits != 0 && (stageBits & (stageBits - 1U)) == 0;
            if (!singleStage || entry.bytecode.empty() || entry.entryPoint.empty())
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "Every shader-library entry requires exactly one stage, "
                                                           "non-empty bytecode, and a non-empty name."));
            }
            auto specializationConstants = metalBuildSpecializationSet(entry.specializationConstants);
            if (!specializationConstants)
            {
                return std::unexpected(std::move(specializationConstants.error()));
            }
            if (std::ranges::any_of(record->entries, [&](const MetalShaderLibraryEntryRecord& existing) {
                    return existing.stage == entry.stage && existing.entryPoint == entry.entryPoint &&
                           existing.specializationConstants == *specializationConstants;
                }))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Shader-library entries must be unique by stage, name, and specialization values."));
            }
            auto payload = unpackShaderManifest(entry.bytecode);
            if (!payload)
            {
                return std::unexpected(std::move(payload.error()));
            }
            const auto json =
                std::string_view(reinterpret_cast<const char*>(payload->json.data()), payload->json.size());
            auto reflected = parseSlangReflectionJson(json, entry.entryPoint, entry.stage);
            if (!reflected)
            {
                return std::unexpected(std::move(reflected.error()));
            }
            if (reflected->stage != entry.stage)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "A shader-library entry stage does not match its PNMS reflection."));
            }
            record->entries.push_back({
                .stage = entry.stage,
                .bytecode = std::vector<std::byte>(entry.bytecode.begin(), entry.bytecode.end()),
                .entryPoint = std::string(entry.entryPoint),
                .specializationConstants = std::move(*specializationConstants),
            });
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal createShaderLibrary requires an active device."));
        }
        return native_->shaderLibraries.emplace(std::move(record));
    }

    Status MetalDevice::destroyShaderLibrary(ShaderLibraryHandle shaderLibrary)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || !native_->shaderLibraries.erase(shaderLibrary))
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The shader library handle is invalid or stale."));
        }
        return {};
    }

    namespace
    {

        [[nodiscard]] Status metalValidateShaderInterfaceWorkgroupSize(const ShaderModuleDesc& desc)
        {
            const auto& workgroupSize = desc.shaderInterface.workgroupSize;
            const bool requiresWorkgroup = desc.stage == ShaderStage::compute || desc.stage == ShaderStage::task ||
                                           desc.stage == ShaderStage::mesh;
            if (workgroupSize.has_value() != requiresWorkgroup)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Compute, task, and mesh interfaces require a workgroup size; other shader stages must omit it."));
            }
            if (workgroupSize.has_value() && (workgroupSize->x == 0 || workgroupSize->y == 0 || workgroupSize->z == 0))
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "Shader-interface workgroup dimensions must be non-zero."));
            }
            return {};
        }

        [[nodiscard]] Result<MetalSpecializationSet> metalValidateShaderModuleDesc(const ShaderModuleDesc& desc)
        {
            const auto stageBits = static_cast<std::uint16_t>(desc.stage);
            const bool singleStage = stageBits != 0 && (stageBits & (stageBits - 1U)) == 0;
            const bool hasDirectBytecode = !desc.bytecode.empty();
            const bool hasLibrary = desc.library.isValid();
            if (!singleStage || desc.entryPoint.empty() || hasDirectBytecode == hasLibrary)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "A shader module requires exactly one stage and one "
                                                           "bytecode source: direct PNMS bytes or a live library."));
            }
            if (auto workgroup = metalValidateShaderInterfaceWorkgroupSize(desc); !workgroup)
            {
                return std::unexpected(std::move(workgroup.error()));
            }
            auto specializationConstants = metalBuildSpecializationSet(desc.specializationConstants);
            if (!specializationConstants)
            {
                return std::unexpected(std::move(specializationConstants.error()));
            }
            if (hasDirectBytecode && !specializationConstants->empty())
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Specialization values require a shader library containing precompiled backend variants."));
            }
            return specializationConstants;
        }

        struct MetalLibraryShaderSource final
        {
            std::shared_ptr<MetalShaderLibraryRecord> library;
            std::vector<std::byte> bytecode;
        };

        [[nodiscard]] Result<MetalLibraryShaderSource> metalFindLibraryShaderSource(
            rachis::detail::TimedMutex& deviceMutex,
            const std::shared_ptr<MetalDeviceNative>& native,
            const ShaderModuleDesc& desc,
            const MetalSpecializationSet& specializationConstants)
        {
            std::scoped_lock const lock(deviceMutex);
            if (native == nullptr)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal createShaderModule requires an active device."));
            }
            auto* foundLibrary = native->shaderLibraries.find(desc.library);
            if (foundLibrary == nullptr)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "A shader module references an invalid or stale shader library."));
            }
            const auto foundEntry =
                std::ranges::find_if((*foundLibrary)->entries, [&](const MetalShaderLibraryEntryRecord& entry) {
                    return entry.stage == desc.stage && entry.entryPoint == desc.entryPoint &&
                           entry.specializationConstants == specializationConstants;
                });
            if (foundEntry == (*foundLibrary)->entries.end())
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "The shader library does not contain the requested stage, "
                                                           "entry point, and specialization values."));
            }
            MetalLibraryShaderSource source;
            source.library = *foundLibrary;
            source.bytecode = foundEntry->bytecode;
            return source;
        }

        struct MetalReflectedShaderModule final
        {
            MetalResolvedStageBindings stageBindings{};
            std::optional<MetalReflectedStageInterface> stageInterface{};
        };

        [[nodiscard]] Result<MetalReflectedShaderModule> metalReflectShaderModule(const MetalShaderPayload& payload,
                                                                                  const ShaderModuleDesc& desc)
        {
            const auto json = std::string_view(reinterpret_cast<const char*>(payload.json.data()), payload.json.size());
            auto reflected = parseSlangReflectionJson(json, desc.entryPoint, desc.stage);
            if (!reflected)
            {
                return std::unexpected(std::move(reflected.error()));
            }
            if (reflected->stage != desc.stage)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "The PNMS entry point stage does not match the requested shader stage."));
            }
            auto stageBindings = resolveShaderInterface(*reflected, desc.shaderInterface);
            if (!stageBindings)
            {
                return std::unexpected(std::move(stageBindings.error()));
            }
            return MetalReflectedShaderModule{
                .stageBindings = std::move(*stageBindings),
                .stageInterface = std::move(reflected->stageInterface),
            };
        }

        [[nodiscard]] Result<id<MTLLibrary>> metalNewLibraryFromMetallib(id<MTLDevice> device,
                                                                         std::span<const std::byte> metallib)
        {
            NSData* const data = [NSData dataWithBytes:metallib.data() length:metallib.size()];
            const dispatch_data_t dispatchData =
                dispatch_data_create(data.bytes, data.length, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                  (void)data;
                });
            NSError* error = nil;
            id<MTLLibrary> const library = [device newLibraryWithData:dispatchData error:&error];
            if (library == nil)
            {
                const char* message = error != nil ? error.localizedDescription.UTF8String : nullptr;
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, message != nullptr ? message : "Metal rejected the cooked metallib."));
            }
            return library;
        }

        struct MetalShaderEntryFunction final
        {
            __strong id<MTLFunction> function;
            std::string name;
        };

        [[nodiscard]] Result<MetalShaderEntryFunction> metalFindShaderEntryFunction(id<MTLLibrary> library,
                                                                                    std::string_view entryPoint)
        {
            NSString* const name = [[NSString alloc] initWithBytes:entryPoint.data()
                                                            length:entryPoint.size()
                                                          encoding:NSUTF8StringEncoding];
            id<MTLFunction> function = name != nil ? [library newFunctionWithName:name] : nil;
            std::string functionName(entryPoint);
            if (function == nil && entryPoint == "main")
            {
                function = [library newFunctionWithName:@"main_0"];
                functionName = "main_0";
            }
            if (function == nil)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "Cooked metallib is missing the requested entry point."));
            }
            return MetalShaderEntryFunction{.function = function, .name = std::move(functionName)};
        }

        [[nodiscard]] Status metalValidateDeclaredVertexInputs(id<MTLFunction> function,
                                                               std::span<const std::uint32_t> declaredLocations)
        {
            std::vector<std::uint32_t> active;
            for (MTLVertexAttribute* attribute in function.vertexAttributes)
            {
                if (attribute.active)
                {
                    active.push_back(static_cast<std::uint32_t>(attribute.attributeIndex));
                }
            }
            std::ranges::sort(active);
            std::vector<std::uint32_t> declared{declaredLocations.begin(), declaredLocations.end()};
            std::ranges::sort(declared);
            if (declared != active)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Declared vertex input locations do not match the Metal vertex function's active attributes."));
            }
            return {};
        }

        [[nodiscard]] std::array<std::uint8_t, kMetalShaderCodeDigestBytes> metalShaderCodeDigest(
            std::span<const std::byte> code) noexcept
        {
            CC_SHA256_CTX context{};
            CC_SHA256_Init(&context);
            constexpr std::size_t maxChunk = std::numeric_limits<CC_LONG>::max();
            for (std::size_t offset = 0; offset < code.size(); offset += maxChunk)
            {
                const auto chunk = (std::min)(maxChunk, code.size() - offset);
                CC_SHA256_Update(&context, code.data() + offset, static_cast<CC_LONG>(chunk));
            }
            std::array<std::uint8_t, kMetalShaderCodeDigestBytes> digest{};
            CC_SHA256_Final(digest.data(), &context);
            return digest;
        }

        [[nodiscard]] MetalShaderModuleRecord metalMakeShaderModuleRecord(
            id<MTLLibrary> library,
            const std::array<std::uint8_t, kMetalShaderCodeDigestBytes>& codeDigest,
            MetalShaderEntryFunction&& entry,
            MetalReflectedShaderModule&& reflected,
            std::shared_ptr<MetalShaderLibraryRecord>&& sourceLibrary,
            const ShaderModuleDesc& desc)
        {
            MetalShaderModuleRecord record{};
            record.library = library;
            record.function = entry.function;
            record.codeDigest = codeDigest;
            record.stageBindings = std::move(reflected.stageBindings);
            record.stageInterface = std::move(reflected.stageInterface);
            record.sourceLibrary = std::move(sourceLibrary);
            record.entryPoint = std::move(entry.name);
            record.label = std::string(desc.label);
            record.snapshot.stage = desc.stage;
            record.snapshot.bindings.assign(desc.shaderInterface.bindings.begin(), desc.shaderInterface.bindings.end());
            record.snapshot.pushConstants.assign(desc.shaderInterface.pushConstants.begin(),
                                                 desc.shaderInterface.pushConstants.end());
            record.snapshot.vertexInputLocations.assign(desc.shaderInterface.vertexInputLocations.begin(),
                                                        desc.shaderInterface.vertexInputLocations.end());
            record.snapshot.workgroupSize = desc.shaderInterface.workgroupSize;
            record.snapshot.shaderRecordConstants = desc.shaderInterface.shaderRecordConstants;
            return record;
        }

    } // namespace

    Result<ShaderModuleHandle> MetalDevice::createShaderModule(const ShaderModuleDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        auto specializationConstants = metalValidateShaderModuleDesc(desc);
        if (!specializationConstants)
        {
            return std::unexpected(std::move(specializationConstants.error()));
        }

        std::vector<std::byte> ownedBytecode;
        std::span<const std::byte> bytecode = desc.bytecode;
        std::shared_ptr<MetalShaderLibraryRecord> sourceLibrary;
        if (desc.library.isValid())
        {
            auto source = metalFindLibraryShaderSource(mutex_, native_, desc, *specializationConstants);
            if (!source)
            {
                return std::unexpected(std::move(source.error()));
            }
            sourceLibrary = std::move(source->library);
            ownedBytecode = std::move(source->bytecode);
            bytecode = ownedBytecode;
        }

        auto payload = unpackShaderManifest(bytecode);
        if (!payload)
        {
            return std::unexpected(std::move(payload.error()));
        }
        auto reflected = metalReflectShaderModule(*payload, desc);
        if (!reflected)
        {
            return std::unexpected(std::move(reflected.error()));
        }
        const auto codeDigest = metalShaderCodeDigest(payload->metallib);

        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal createShaderModule requires an active device."));
        }

        @autoreleasepool
        {
            auto library = metalNewLibraryFromMetallib(native_->device, payload->metallib);
            if (!library)
            {
                return std::unexpected(std::move(library.error()));
            }
            auto entry = metalFindShaderEntryFunction(*library, desc.entryPoint);
            if (!entry)
            {
                return std::unexpected(std::move(entry.error()));
            }
            if (auto inputs =
                    metalValidateDeclaredVertexInputs(entry->function, desc.shaderInterface.vertexInputLocations);
                !inputs)
            {
                return std::unexpected(std::move(inputs.error()));
            }
            auto handle = native_->shaderModules.emplace(metalMakeShaderModuleRecord(
                *library, codeDigest, std::move(*entry), std::move(*reflected), std::move(sourceLibrary), desc));
            if (!handle)
            {
                return std::unexpected(std::move(handle.error()));
            }
            return handle;
        }
    }

    Result<ShaderInterfaceSnapshot> MetalDevice::shaderModuleInterface(ShaderModuleHandle shaderModule) const
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal shaderModuleInterface requires an active device."));
        }
        const auto* record = native_->shaderModules.find(shaderModule);
        if (record == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The shader module handle is not a live Metal object."));
        }
        return record->snapshot;
    }

    Status MetalDevice::destroyShaderModule(ShaderModuleHandle shaderModule)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal destroyShaderModule requires an active device."));
        }
        if (native_->shaderModules.find(shaderModule) == nullptr || !native_->shaderModules.erase(shaderModule))
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The shader module handle is not a live Metal object."));
        }
        return {};
    }

    namespace
    {

        [[nodiscard]] Status metalResolveLayoutBindingSets(MetalDeviceNative& native,
                                                           std::span<const BindingLayoutHandle> bindingLayouts,
                                                           MetalPipelineLayoutRecord& record)
        {
            record.sets.reserve(bindingLayouts.size());
            for (const auto handle : bindingLayouts)
            {
                auto* found = native.bindingLayouts.find(handle);
                if (found == nullptr)
                {
                    return std::unexpected(
                        metalSelectionError(ErrorCode::invalidArgument,
                                            "A pipeline layout references an invalid or stale binding layout."));
                }
                record.sets.push_back(*found);
            }
            return {};
        }

        struct MetalStageArgumentUsage final
        {
            std::array<std::uint64_t, 3> slots{};
            bool hasVariableVertexBuffer = false;
        };

        [[nodiscard]] Result<MetalStageArgumentUsage> metalCountStageArguments(
            ShaderStage stage,
            std::span<const PushConstantRange> pushConstantRanges,
            const std::vector<std::shared_ptr<MetalBindingLayoutRecord>>& sets)
        {
            constexpr std::array<std::uint64_t, 3> argumentLimits{
                metalMaxBufferArguments,
                metalMaxTextureArguments,
                metalMaxSamplerArguments,
            };
            MetalStageArgumentUsage usage;
            auto& stageSlots = usage.slots;
            if (std::ranges::any_of(pushConstantRanges,
                                    [&](const PushConstantRange& range) { return hasAny(range.visibility, stage); }))
            {
                stageSlots[0] += 1;
            }
            for (const auto& set : sets)
            {
                if (set == nullptr)
                {
                    continue;
                }
                for (const auto& entry : set->entries)
                {
                    if (!hasAny(entry.visibility, stage))
                    {
                        continue;
                    }
                    const auto kind = metalRangeClassIndex(entry.type);
                    if (entry.count == unboundedBindingCount)
                    {
                        if (stage == ShaderStage::vertex && kind == 0)
                        {
                            usage.hasVariableVertexBuffer = true;
                        }
                        continue;
                    }
                    stageSlots[kind] += entry.count;
                    if (stageSlots[kind] > argumentLimits[kind])
                    {
                        return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                                   "Pipeline layout bindings and push constants exceed "
                                                                   "Metal's per-stage argument-table capacity."));
                    }
                }
            }
            return usage;
        }

        [[nodiscard]] Status metalValidateStageArgumentCapacity(std::span<const PushConstantRange> pushConstantRanges,
                                                                MetalPipelineLayoutRecord& record)
        {
            constexpr std::array<ShaderStage, 3> checkedStages{
                ShaderStage::compute,
                ShaderStage::vertex,
                ShaderStage::pixel,
            };
            bool hasVariableVertexBuffer = false;
            for (const ShaderStage stage : checkedStages)
            {
                auto usage = metalCountStageArguments(stage, pushConstantRanges, record.sets);
                if (!usage)
                {
                    return std::unexpected(std::move(usage.error()));
                }
                hasVariableVertexBuffer = hasVariableVertexBuffer || usage->hasVariableVertexBuffer;
                if (stage == ShaderStage::vertex)
                {
                    record.vertexStageBufferSlots =
                        hasVariableVertexBuffer ? metalMaxBufferArguments : static_cast<std::uint32_t>(usage->slots[0]);
                }
            }
            return {};
        }

        void metalRemapLayoutBindings(MetalPipelineLayoutRecord& record)
        {
            std::array<std::uint32_t, 3> nextClassIndex{};
            for (std::uint32_t setIndex = 0; setIndex < record.sets.size(); ++setIndex)
            {
                const auto& set = record.sets[setIndex];
                if (set == nullptr)
                {
                    continue;
                }
                for (const auto& entry : set->entries)
                {
                    const auto kind = metalRangeClassIndex(entry.type);
                    const auto count = entry.count == unboundedBindingCount
                                           ? (metalArgumentLimit(entry.type) -
                                              std::min(nextClassIndex[kind], metalArgumentLimit(entry.type)))
                                           : entry.count;
                    record.remappedBindings.push_back(MetalLayoutBindingSlotMap{
                        .set = setIndex,
                        .binding = entry.binding,
                        .count = count,
                        .metalBaseIndex = nextClassIndex[kind],
                    });
                    nextClassIndex[kind] += count;
                }
            }
            record.pushConstantMetalIndex = record.reservePushConstantBuffer ? nextClassIndex[0] : 0U;
        }

    } // namespace

    Result<PipelineLayoutHandle> MetalDevice::createPipelineLayout(const PipelineLayoutDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (desc.bindingLayouts.size() > maxBindingSets)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::unsupported, "A pipeline layout exceeds the portable four-set binding limit."));
        }
        auto pushConstantSize = metalValidatePushConstantRanges(desc.pushConstantRanges);
        if (!pushConstantSize)
        {
            return std::unexpected(std::move(pushConstantSize.error()));
        }
        auto record = std::make_shared<MetalPipelineLayoutRecord>();
        record->label = std::string(desc.label);
        record->pushConstantRanges.assign(desc.pushConstantRanges.begin(), desc.pushConstantRanges.end());
        record->pushConstantBytes = *pushConstantSize;
        record->reservePushConstantBuffer = record->pushConstantBytes > 0;
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal createPipelineLayout requires an active device."));
        }
        if (auto sets = metalResolveLayoutBindingSets(*native_, desc.bindingLayouts, *record); !sets)
        {
            return std::unexpected(std::move(sets.error()));
        }
        if (auto capacity = metalValidateStageArgumentCapacity(desc.pushConstantRanges, *record); !capacity)
        {
            return std::unexpected(std::move(capacity.error()));
        }
        metalRemapLayoutBindings(*record);
        return native_->pipelineLayouts.emplace(std::move(record));
    }

    Status MetalDevice::destroyPipelineLayout(PipelineLayoutHandle pipelineLayout)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || !native_->pipelineLayouts.erase(pipelineLayout))
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The pipeline layout handle is invalid or stale."));
        }
        return {};
    }

    Result<PipelineCacheHandle> MetalDevice::createPipelineCache(const PipelineCacheDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::uint64_t registryId = 0;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr || native_->device == nil)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal createPipelineCache requires an active device."));
            }
            registryId = native_->device.registryID;
        }
        auto decoded = metalDecodePipelineCacheBlob(desc.initialData, registryId);
        if (!decoded)
        {
            return std::unexpected(std::move(decoded.error()));
        }
        auto record = std::make_shared<MetalPipelineCacheRecord>();
        record->label = std::string(desc.label);
        record->segments.reserve(decoded->segments.size());
        for (auto& bytes : decoded->segments)
        {
            auto segment = std::make_shared<MetalPipelineCacheSegment>();
            segment->bytes = std::move(bytes);
            record->segments.push_back(std::move(segment));
        }
        for (const auto& entry : decoded->entries)
        {
            record->live[entry.identity].segment = record->segments[entry.segment];
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal createPipelineCache requires an active device."));
        }
        MTL4PipelineDataSetSerializerDescriptor* const serializerDesc = [MTL4PipelineDataSetSerializerDescriptor new];
        serializerDesc.configuration = MTL4PipelineDataSetSerializerConfigurationCaptureBinaries;
        record->serializer = [native_->device newPipelineDataSetSerializerWithDescriptor:serializerDesc];
        if (record->serializer == nil)
        {
            return std::unexpected(metalSelectionError(ErrorCode::nativeFailure,
                                                       "Metal could not create a pipeline data-set serializer."));
        }
        MTL4CompilerDescriptor* const compilerDesc = [MTL4CompilerDescriptor new];
        compilerDesc.label = metalNsString(desc.label.empty() ? "RHI.Metal.PipelineCache" : desc.label);
        compilerDesc.pipelineDataSetSerializer = record->serializer;
        NSError* compilerError = nil;
        record->compiler = [native_->device newCompilerWithDescriptor:compilerDesc error:&compilerError];
        if (record->compiler == nil)
        {
            return std::unexpected(
                metalNsError(ErrorCode::deviceCreationFailed, compilerError, "newCompilerWithDescriptor"));
        }
        for (const auto& segment : record->segments)
        {
            NSURL* keptURL = nil;
            auto archive = metalLoadPipelineCacheArchive(native_->device, segment->bytes, &keptURL);
            if (!archive)
            {
                return std::unexpected(std::move(archive.error()));
            }
            segment->archive = *archive;
            segment->archiveFileURL = keptURL;
        }
        return native_->pipelineCaches.emplace(std::move(record));
    }

    Status MetalDevice::destroyPipelineCache(PipelineCacheHandle pipelineCache)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || !native_->pipelineCaches.erase(pipelineCache))
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The pipeline cache handle is invalid or stale."));
        }
        return {};
    }

    Result<std::vector<std::byte>> MetalDevice::pipelineCacheData(PipelineCacheHandle pipelineCache)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::shared_ptr<MetalPipelineCacheRecord> record;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal pipelineCacheData requires an active device."));
            }
            auto* found = native_->pipelineCaches.find(pipelineCache);
            if (found == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "The pipeline cache handle is invalid or stale."));
            }
            record = *found;
        }
        const std::unique_lock serializeLock(record->serializeMutex);
        const std::scoped_lock cacheLock(record->mutex);
        if (record->serializerDirty && record->serializer != nil)
        {
            if (auto flushed = metalFlushPipelineCacheSerializer(*record); !flushed)
            {
                return std::unexpected(std::move(flushed.error()));
            }
        }
        const std::uint64_t registryId =
            record->compiler != nil && record->compiler.device != nil ? record->compiler.device.registryID : 0;
        return metalEncodePipelineCacheBlob(metalPipelineCacheBlobEntries(*record), record->segments, registryId);
    }

    namespace
    {

        [[nodiscard]] bool metalIsCompiledShaderOfStage(const MetalShaderModuleRecord& shader, ShaderStage stage)
        {
            return shader.snapshot.stage == stage && shader.function != nil && shader.library != nil;
        }

        [[nodiscard]] Result<std::shared_ptr<MetalPipelineCacheRecord>> metalFindOptionalPipelineCache(
            MetalDeviceNative& native, PipelineCacheHandle handle)
        {
            std::shared_ptr<MetalPipelineCacheRecord> cache;
            if (handle.isValid())
            {
                auto* foundCache = native.pipelineCaches.find(handle);
                if (foundCache == nullptr)
                {
                    return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                               "The pipeline cache handle is invalid or stale."));
                }
                cache = *foundCache;
            }
            return cache;
        }

        class MetalPipelineFingerprintWriter final
        {
        public:
            template<typename T>
                requires std::is_trivially_copyable_v<T>
            void write(const T& value)
            {
                const auto* const first = reinterpret_cast<const std::byte*>(&value);
                bytes_.insert(bytes_.end(), first, first + sizeof(T));
            }

            void writeText(std::string_view text)
            {
                write(static_cast<std::uint64_t>(text.size()));
                const auto* const first = reinterpret_cast<const std::byte*>(text.data());
                bytes_.insert(bytes_.end(), first, first + text.size());
            }

            void writeShader(const MetalShaderModuleRecord& shader)
            {
                write(shader.codeDigest);
                writeText(shader.entryPoint);
            }

            [[nodiscard]] std::vector<std::byte> take() noexcept
            {
                return std::move(bytes_);
            }

        private:
            std::vector<std::byte> bytes_;
        };

        [[nodiscard]] bool metalCacheEntryConflicts(const MetalPipelineCacheLive& live,
                                                    const std::vector<std::byte>& fingerprint) noexcept
        {
            return !live.fingerprint.empty() && live.fingerprint != fingerprint;
        }

        [[nodiscard]] Error metalIncompatibleCacheEntryError(MetalPipelineCacheKind kind)
        {
            return metalSelectionError(
                ErrorCode::invalidArgument,
                kind == MetalPipelineCacheKind::compute
                    ? "The Metal pipeline cache key identifies an incompatible compute pipeline."
                    : "The Metal pipeline cache key identifies an incompatible graphics pipeline.");
        }

        [[nodiscard]] Status metalCheckCacheEntryCompatible(MetalPipelineCacheRecord& cache,
                                                            const MetalPipelineCacheIdentity& identity,
                                                            const std::vector<std::byte>& fingerprint)
        {
            std::scoped_lock const cacheLock(cache.mutex);
            const auto found = cache.live.find(identity);
            if (found != cache.live.end() && metalCacheEntryConflicts(found->second, fingerprint))
            {
                return std::unexpected(metalIncompatibleCacheEntryError(identity.kind));
            }
            return {};
        }

        struct MetalPipelineCompilerChoice final
        {
            __strong id<MTL4Compiler> compiler;
            __strong NSArray<id<MTL4Archive>>* lookupArchives;
        };

        [[nodiscard]] Result<MetalPipelineCompilerChoice> metalChoosePipelineCompiler(
            const std::shared_ptr<MetalPipelineCacheRecord>& cache,
            rachis::detail::TimedMutex& deviceMutex,
            const std::shared_ptr<MetalDeviceNative>& native,
            const char* inactiveCompilerMessage)
        {
            id<MTL4Compiler> compiler = nil;
            NSArray<id<MTL4Archive>>* lookupArchives = nil;
            if (cache != nullptr)
            {
                std::scoped_lock const cacheLock(cache->mutex);
                compiler = cache->compiler;
                lookupArchives = metalPipelineCacheLookupArchives(*cache);
            }
            if (compiler == nil)
            {
                std::scoped_lock const lock(deviceMutex);
                if (native == nullptr || native->compiler == nil)
                {
                    return std::unexpected(metalSelectionError(ErrorCode::invalidState, inactiveCompilerMessage));
                }
                compiler = native->compiler;
            }
            return MetalPipelineCompilerChoice{.compiler = compiler, .lookupArchives = lookupArchives};
        }

        [[nodiscard]] MTL4CompilerTaskOptions* metalLookupArchiveTaskOptions(NSArray<id<MTL4Archive>>* lookupArchives)
        {
            MTL4CompilerTaskOptions* options = nil;
            if (lookupArchives.count != 0)
            {
                options = [MTL4CompilerTaskOptions new];
                options.lookupArchives = lookupArchives;
            }
            return options;
        }

        struct MetalComputePipelineInputs final
        {
            std::shared_ptr<MetalPipelineLayoutRecord> layout;
            MetalShaderModuleRecord shader{};
            std::shared_ptr<MetalPipelineCacheRecord> cache;
        };

        [[nodiscard]] Result<MetalComputePipelineInputs> metalAcquireComputePipelineInputs(
            rachis::detail::TimedMutex& deviceMutex,
            const std::shared_ptr<MetalDeviceNative>& native,
            const ComputePipelineDesc& desc)
        {
            std::scoped_lock const lock(deviceMutex);
            if (native == nullptr || native->compiler == nil)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal createComputePipeline requires an active compiler."));
            }
            auto* foundLayout = native->pipelineLayouts.find(desc.layout);
            auto* foundShader = native->shaderModules.find(desc.computeShader);
            if (foundLayout == nullptr || foundShader == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "A compute pipeline references an invalid or stale layout or shader module."));
            }
            if (!metalIsCompiledShaderOfStage(*foundShader, ShaderStage::compute))
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "A compute pipeline requires a compute shader module."));
            }
            MetalComputePipelineInputs inputs;
            inputs.layout = *foundLayout;
            inputs.shader = *foundShader;
            auto cache = metalFindOptionalPipelineCache(*native, desc.cache);
            if (!cache)
            {
                return std::unexpected(std::move(cache.error()));
            }
            inputs.cache = std::move(*cache);
            return inputs;
        }

        [[nodiscard]] Result<ShaderWorkgroupSize> metalValidateComputeShaderInterfaceAndWorkgroup(
            const MetalComputePipelineInputs& inputs, const DeviceCapabilities& capabilities)
        {
            if (auto interfaceValidation = metalValidatePipelineShaderInterface(inputs.shader.snapshot, *inputs.layout);
                !interfaceValidation)
            {
                return std::unexpected(std::move(interfaceValidation.error()));
            }
            if (!inputs.shader.snapshot.workgroupSize.has_value())
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "The compute shader module has no workgroup size."));
            }
            const auto workgroup = *inputs.shader.snapshot.workgroupSize;
            if (auto workgroupValidation = metalValidateWorkgroupSize(workgroup, capabilities); !workgroupValidation)
            {
                return std::unexpected(std::move(workgroupValidation.error()));
            }
            return workgroup;
        }

        [[nodiscard]] MTL4ComputePipelineDescriptor* metalMakeComputePipelineDescriptor(
            const MetalShaderModuleRecord& shader, const ShaderWorkgroupSize& workgroup, std::string_view label)
        {
            MTL4ComputePipelineDescriptor* const pipelineDesc = [MTL4ComputePipelineDescriptor new];
            pipelineDesc.computeFunctionDescriptor = metalLibraryFunctionDescriptor(shader.library, shader.entryPoint);
            pipelineDesc.requiredThreadsPerThreadgroup = MTLSizeMake(workgroup.x, workgroup.y, workgroup.z);
            pipelineDesc.maxTotalThreadsPerThreadgroup =
                static_cast<NSUInteger>(workgroup.x) * workgroup.y * workgroup.z;
            if (!label.empty())
            {
                pipelineDesc.label = metalNsString(label);
            }
            return pipelineDesc;
        }

        [[nodiscard]] std::vector<std::byte> metalComputePipelineFingerprint(const MetalShaderModuleRecord& shader,
                                                                             const ShaderWorkgroupSize& workgroup)
        {
            MetalPipelineFingerprintWriter writer;
            writer.writeShader(shader);
            writer.write(workgroup.x);
            writer.write(workgroup.y);
            writer.write(workgroup.z);
            return writer.take();
        }

        [[nodiscard]] Result<id<MTLComputePipelineState>> metalRequireComputeCacheHit(
            MetalPipelineCacheRecord& cache,
            const MetalPipelineCacheIdentity& identity,
            const std::vector<std::byte>& fingerprint,
            MTL4ComputePipelineDescriptor* pipelineDesc)
        {
            std::scoped_lock const cacheLock(cache.mutex);
            const auto found = cache.live.find(identity);
            if (found == cache.live.end())
            {
                return std::unexpected(metalSelectionError(ErrorCode::pipelineCacheMiss,
                                                           "The Metal pipeline cache has no matching compute entry."));
            }
            if (metalCacheEntryConflicts(found->second, fingerprint))
            {
                return std::unexpected(metalIncompatibleCacheEntryError(identity.kind));
            }
            if (found->second.compute != nil)
            {
                return found->second.compute;
            }
            const auto& segment = found->second.segment;
            if (segment == nullptr || segment->archive == nil)
            {
                return std::unexpected(metalSelectionError(ErrorCode::pipelineCacheMiss,
                                                           "The Metal pipeline cache has no matching compute entry."));
            }
            id<MTLComputePipelineState> pso = nil;
            NSError* archiveError = nil;
            @try
            {
                pso = [segment->archive newComputePipelineStateWithDescriptor:pipelineDesc error:&archiveError];
            }
            @catch (NSException* exception)
            {
                return std::unexpected(metalExceptionError(exception, "newComputePipelineStateWithDescriptor"));
            }
            if (pso == nil)
            {
                return std::unexpected(metalSelectionError(ErrorCode::pipelineCacheMiss,
                                                           "The Metal pipeline cache archive missed a compute entry."));
            }
            found->second.compute = pso;
            found->second.fingerprint = fingerprint;
            return pso;
        }

        [[nodiscard]] ErrorCode metalPipelineCompileErrorCode(NSError* error) noexcept
        {
            if (error == nil || ([error.domain isEqualToString:MTLLibraryErrorDomain] &&
                                 error.code == static_cast<NSInteger>(MTLLibraryErrorInternal)))
            {
                return ErrorCode::nativeFailure;
            }
            return ErrorCode::validationError;
        }

        [[nodiscard]] Result<id<MTLComputePipelineState>> metalCompileComputePipeline(
            MTL4ComputePipelineDescriptor* pipelineDesc,
            const std::shared_ptr<MetalPipelineCacheRecord>& cache,
            const MetalPipelineCacheIdentity& identity,
            const std::vector<std::byte>& fingerprint,
            rachis::detail::TimedMutex& deviceMutex,
            const std::shared_ptr<MetalDeviceNative>& native)
        {
            std::shared_lock<std::shared_mutex> serializeLock;
            if (cache != nullptr)
            {
                serializeLock = std::shared_lock(cache->serializeMutex);
                if (auto compatible = metalCheckCacheEntryCompatible(*cache, identity, fingerprint); !compatible)
                {
                    return std::unexpected(std::move(compatible.error()));
                }
            }
            NSError* error = nil;
            auto choice = metalChoosePipelineCompiler(
                cache, deviceMutex, native, "Metal createComputePipeline requires an active compiler.");
            if (!choice)
            {
                return std::unexpected(std::move(choice.error()));
            }
            MTL4CompilerTaskOptions* options = metalLookupArchiveTaskOptions(choice->lookupArchives);
            id<MTLComputePipelineState> pso = nil;
            @try
            {
                pso = [choice->compiler newComputePipelineStateWithDescriptor:pipelineDesc
                                                          compilerTaskOptions:options
                                                                        error:&error];
            }
            @catch (NSException* exception)
            {
                return std::unexpected(metalExceptionError(exception, "newComputePipelineStateWithDescriptor"));
            }
            if (pso == nil)
            {
                return std::unexpected(
                    metalNsError(metalPipelineCompileErrorCode(error), error, "newComputePipelineStateWithDescriptor"));
            }
            if (cache != nullptr)
            {
                std::scoped_lock const cacheLock(cache->mutex);
                auto& live = cache->live[identity];
                if (metalCacheEntryConflicts(live, fingerprint))
                {
                    return std::unexpected(metalIncompatibleCacheEntryError(identity.kind));
                }
                live.compute = pso;
                live.fingerprint = fingerprint;
                live.inSerializer = true;
                cache->serializerDirty = true;
            }
            return pso;
        }

        [[nodiscard]] std::shared_ptr<MetalPipelineRecord> metalMakeComputePipelineRecord(
            MetalComputePipelineInputs& inputs,
            id<MTLComputePipelineState> pso,
            const ShaderWorkgroupSize& workgroup,
            std::string_view label)
        {
            auto record = std::make_shared<MetalPipelineRecord>();
            record->kind = MetalPipelineKind::compute;
            record->layout = std::move(inputs.layout);
            record->setRequired = metalRequiredBindingSets(*record->layout);
            record->compute = pso;
            record->workgroup = workgroup;
            record->computeStageBindings = inputs.shader.stageBindings;
            record->pushConstantBytes = record->layout->pushConstantBytes;
            record->reservePushConstantBuffer = record->layout->reservePushConstantBuffer;
            record->label = std::string(label);
            return record;
        }

    } // namespace

    Result<PipelineHandle> MetalDevice::createComputePipeline(const ComputePipelineDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (auto cache = metalValidatePipelineCacheRequest(desc.cachePolicy, desc.cache, desc.cacheKey); !cache)
        {
            return std::unexpected(std::move(cache.error()));
        }
        auto inputs = metalAcquireComputePipelineInputs(mutex_, native_, desc);
        if (!inputs)
        {
            return std::unexpected(std::move(inputs.error()));
        }
        auto workgroup = metalValidateComputeShaderInterfaceAndWorkgroup(*inputs, capabilities_);
        if (!workgroup)
        {
            return std::unexpected(std::move(workgroup.error()));
        }

        id<MTLComputePipelineState> pso = nil;
        @autoreleasepool
        {
            MTL4ComputePipelineDescriptor* const pipelineDesc =
                metalMakeComputePipelineDescriptor(inputs->shader, *workgroup, desc.label);
            const MetalPipelineCacheIdentity identity{.key = desc.cacheKey, .kind = MetalPipelineCacheKind::compute};
            const auto& cache = inputs->cache;
            const auto fingerprint = cache != nullptr ? metalComputePipelineFingerprint(inputs->shader, *workgroup)
                                                      : std::vector<std::byte>{};
            if (cache != nullptr && desc.cachePolicy == PipelineCachePolicy::requireHit)
            {
                auto hit = metalRequireComputeCacheHit(*cache, identity, fingerprint, pipelineDesc);
                if (!hit)
                {
                    return std::unexpected(std::move(hit.error()));
                }
                pso = *hit;
            }
            if (pso == nil)
            {
                auto compiled =
                    metalCompileComputePipeline(pipelineDesc, cache, identity, fingerprint, mutex_, native_);
                if (!compiled)
                {
                    return std::unexpected(std::move(compiled.error()));
                }
                pso = *compiled;
            }
            auto record = metalMakeComputePipelineRecord(*inputs, pso, *workgroup, desc.label);
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal createComputePipeline requires an active device."));
            }
            return native_->pipelines.emplace(std::move(record));
        }
    }

    namespace
    {

        struct MetalGraphicsPipelineInputs final
        {
            std::shared_ptr<MetalPipelineLayoutRecord> layout;
            MetalShaderModuleRecord vertex{};
            MetalShaderModuleRecord pixel{};
            std::shared_ptr<MetalPipelineCacheRecord> cache;
            __strong id<MTLDevice> mtlDevice = nil;
            bool hasPixel = false;
        };

        [[nodiscard]] Status metalAcquireGraphicsShaders(MetalDeviceNative& native,
                                                         const GraphicsPipelineDesc& desc,
                                                         MetalGraphicsPipelineInputs& inputs)
        {
            auto* foundLayout = native.pipelineLayouts.find(desc.layout);
            auto* foundVertex = native.shaderModules.find(desc.vertexShader);
            if (foundLayout == nullptr || foundVertex == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "A graphics pipeline references an invalid or stale layout or shader module."));
            }
            if (!metalIsCompiledShaderOfStage(*foundVertex, ShaderStage::vertex))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "A graphics pipeline requires correctly staged conventional or task/mesh shader modules."));
            }
            inputs.layout = *foundLayout;
            inputs.vertex = *foundVertex;
            if (!inputs.hasPixel)
            {
                return {};
            }
            auto* foundPixel = native.shaderModules.find(desc.pixelShader);
            if (foundPixel == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "A graphics pipeline references an invalid or stale layout or shader module."));
            }
            if (!metalIsCompiledShaderOfStage(*foundPixel, ShaderStage::pixel))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "A graphics pipeline requires correctly staged conventional or task/mesh shader modules."));
            }
            inputs.pixel = *foundPixel;
            return {};
        }

        [[nodiscard]] Result<MetalGraphicsPipelineInputs> metalAcquireGraphicsPipelineInputs(
            rachis::detail::TimedMutex& deviceMutex,
            const std::shared_ptr<MetalDeviceNative>& native,
            const GraphicsPipelineDesc& desc)
        {
            MetalGraphicsPipelineInputs inputs;
            inputs.hasPixel = desc.pixelShader.isValid();
            std::scoped_lock const lock(deviceMutex);
            if (native == nullptr || native->compiler == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState, "Metal createGraphicsPipeline requires an active compiler."));
            }
            inputs.mtlDevice = native->device;
            if (auto shaders = metalAcquireGraphicsShaders(*native, desc, inputs); !shaders)
            {
                return std::unexpected(std::move(shaders.error()));
            }
            auto cache = metalFindOptionalPipelineCache(*native, desc.cache);
            if (!cache)
            {
                return std::unexpected(std::move(cache.error()));
            }
            inputs.cache = std::move(*cache);
            return inputs;
        }

        [[nodiscard]] Status metalValidateStageLinkage(const MetalShaderModuleRecord& vertex,
                                                       const MetalShaderModuleRecord& pixel)
        {
            if (!vertex.stageInterface || !pixel.stageInterface)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "A graphics pipeline's vertex and pixel shader manifests must record their stage inputs and "
                    "outputs. Cook the shaders with RachisShaderBinaryTool cook-metallib."));
            }
            return rachis::detail::validateStageLinkage(
                ShaderStage::vertex, vertex.stageInterface->outputs, pixel.stageInterface->inputs);
        }

        [[nodiscard]] Status metalValidateGraphicsShaderInterfaces(const MetalGraphicsPipelineInputs& inputs,
                                                                   const std::vector<std::uint32_t>& locations)
        {
            if (auto vertexInterface = metalValidatePipelineShaderInterface(inputs.vertex.snapshot, *inputs.layout);
                !vertexInterface)
            {
                return std::unexpected(std::move(vertexInterface.error()));
            }
            if (auto vertexLocations = metalValidateVertexInterface(inputs.vertex.snapshot, locations);
                !vertexLocations)
            {
                return std::unexpected(std::move(vertexLocations.error()));
            }
            if (inputs.hasPixel)
            {
                if (auto pixelInterface = metalValidatePipelineShaderInterface(inputs.pixel.snapshot, *inputs.layout);
                    !pixelInterface)
                {
                    return std::unexpected(std::move(pixelInterface.error()));
                }
                if (auto linkage = metalValidateStageLinkage(inputs.vertex, inputs.pixel); !linkage)
                {
                    return std::unexpected(std::move(linkage.error()));
                }
            }
            return {};
        }

        [[nodiscard]] Status metalValidateColorTargetState(const MetalDevice& device, const ColorTargetState& target)
        {
            if (target.format == Format::unknown || metalPixelFormat(target.format) == MTLPixelFormatInvalid)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Graphics pipelines require valid attachment, rasterization, and multisample state."));
            }
            if (!metalIsValidBlendComponent(target.blend.color) || !metalIsValidBlendComponent(target.blend.alpha))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "A graphics color target contains an invalid blend factor or operation."));
            }
            const auto properties = device.queryFormatProperties(target.format);
            if (!hasAny(properties.features, FormatFeature::colorAttachment))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported, "The device does not support the requested color-attachment format."));
            }
            if (target.blend.enabled && !hasAny(properties.features, FormatFeature::blendable))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported,
                    "The device does not support blending for the requested color-attachment format."));
            }
            return {};
        }

        [[nodiscard]] bool metalBlendNeedsDualSource(const BlendState& blend)
        {
            return metalBlendFactorNeedsDualSource(blend.color.sourceFactor) ||
                   metalBlendFactorNeedsDualSource(blend.color.destinationFactor) ||
                   metalBlendFactorNeedsDualSource(blend.alpha.sourceFactor) ||
                   metalBlendFactorNeedsDualSource(blend.alpha.destinationFactor);
        }

        [[nodiscard]] bool metalBlendUsesConstants(const BlendState& blend)
        {
            return blend.enabled && (metalBlendFactorUsesConstant(blend.color.sourceFactor) ||
                                     metalBlendFactorUsesConstant(blend.color.destinationFactor) ||
                                     metalBlendFactorUsesConstant(blend.alpha.sourceFactor) ||
                                     metalBlendFactorUsesConstant(blend.alpha.destinationFactor));
        }

        struct MetalColorTargetSummary final
        {
            std::vector<Format> colorFormats;
            bool usesBlendConstants = false;
        };

        [[nodiscard]] Result<MetalColorTargetSummary> metalValidateColorTargets(
            const MetalDevice& device, std::span<const ColorTargetState> colorTargets)
        {
            bool usesDualSourceBlending = false;
            MetalColorTargetSummary summary;
            summary.colorFormats.reserve(colorTargets.size());
            for (const auto& target : colorTargets)
            {
                if (auto state = metalValidateColorTargetState(device, target); !state)
                {
                    return std::unexpected(std::move(state.error()));
                }
                usesDualSourceBlending = usesDualSourceBlending || metalBlendNeedsDualSource(target.blend);
                summary.usesBlendConstants = summary.usesBlendConstants || metalBlendUsesConstants(target.blend);
                summary.colorFormats.push_back(target.format);
            }
            if (usesDualSourceBlending)
            {
                if (colorTargets.size() != 1)
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument, "Dual-source blending requires exactly one color target."));
                }
                return std::unexpected(
                    metalSelectionError(ErrorCode::unsupported, "The device does not support dual-source blending."));
            }
            return summary;
        }

        [[nodiscard]] Status metalValidateDepthStencilState(const MetalDevice& device,
                                                            const DepthStencilState& depthStencil)
        {
            if (depthStencil.format != Format::unknown)
            {
                if (!formatIsDepth(depthStencil.format))
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument, "A graphics depth-stencil target requires a depth format."));
                }
                const auto properties = device.queryFormatProperties(depthStencil.format);
                if (!hasAny(properties.features, FormatFeature::depthStencilAttachment))
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::unsupported, "The device does not support the requested depth/stencil format."));
                }
            }
            if ((depthStencil.depthWriteEnabled && !depthStencil.depthTestEnabled) ||
                !metalIsValidCompareOperation(depthStencil.depthCompare) ||
                !metalIsValidStencilFace(depthStencil.frontFace) || !metalIsValidStencilFace(depthStencil.backFace))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "The depth-stencil state contains an invalid compare or operation, or enables "
                                        "writes without depth testing."));
            }
            if (depthStencil.stencilTestEnabled && depthStencil.format != Format::d24UnormS8Uint)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::unsupported,
                                        "Stencil testing requires the portable d24UnormS8Uint depth-stencil format."));
            }
            return {};
        }

        [[nodiscard]] Result<std::uint32_t> metalVertexBufferBaseIndex(const MetalPipelineLayoutRecord& layout,
                                                                       const MetalResolvedStageBindings& vertexBindings,
                                                                       std::size_t vertexBufferCount)
        {
            std::uint32_t vertexBufferBaseIndex = layout.vertexStageBufferSlots;
            if (vertexBindings.usesPushConstants)
            {
                vertexBufferBaseIndex = std::max(vertexBufferBaseIndex, vertexBindings.pushConstantMetalIndex + 1U);
            }
            for (const auto& res : vertexBindings.resources)
            {
                if (res.type == ShaderResourceType::uniformBuffer ||
                    res.type == ShaderResourceType::readOnlyStorageBuffer ||
                    res.type == ShaderResourceType::storageBuffer)
                {
                    vertexBufferBaseIndex = std::max(vertexBufferBaseIndex, res.metalIndex + res.count);
                }
            }
            if (vertexBufferBaseIndex + vertexBufferCount > metalMaxBufferArguments)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::unsupported,
                                        "Shader bindings and vertex buffers exceed Metal's buffer argument slots."));
            }
            return vertexBufferBaseIndex;
        }

        struct MetalGraphicsAttachmentLayout final
        {
            MetalColorTargetSummary colorTargets;
            std::uint32_t vertexBufferBaseIndex = 0;
        };

        [[nodiscard]] Result<MetalGraphicsAttachmentLayout> metalResolveGraphicsAttachmentLayout(
            const MetalDevice& device, const GraphicsPipelineDesc& desc, const MetalGraphicsPipelineInputs& inputs)
        {
            auto colorTargets = metalValidateColorTargets(device, desc.colorTargets);
            if (!colorTargets)
            {
                return std::unexpected(std::move(colorTargets.error()));
            }
            if (auto depthStencil = metalValidateDepthStencilState(device, desc.depthStencil); !depthStencil)
            {
                return std::unexpected(std::move(depthStencil.error()));
            }
            auto vertexBufferBaseIndex =
                metalVertexBufferBaseIndex(*inputs.layout, inputs.vertex.stageBindings, desc.vertexBuffers.size());
            if (!vertexBufferBaseIndex)
            {
                return std::unexpected(std::move(vertexBufferBaseIndex.error()));
            }
            return MetalGraphicsAttachmentLayout{
                .colorTargets = std::move(*colorTargets),
                .vertexBufferBaseIndex = *vertexBufferBaseIndex,
            };
        }

        void metalConfigureColorAttachments(MTL4RenderPipelineDescriptor* pipelineDesc,
                                            std::span<const ColorTargetState> colorTargets)
        {
            for (NSUInteger index = 0; index < colorTargets.size(); ++index)
            {
                const auto& target = colorTargets[index];
                auto* const attachment = pipelineDesc.colorAttachments[index];
                attachment.pixelFormat = metalPixelFormat(target.format);
                attachment.blendingState = target.blend.enabled ? MTL4BlendStateEnabled : MTL4BlendStateDisabled;
                attachment.sourceRGBBlendFactor = metalBlendFactor(target.blend.color.sourceFactor);
                attachment.destinationRGBBlendFactor = metalBlendFactor(target.blend.color.destinationFactor);
                attachment.rgbBlendOperation = metalBlendOperation(target.blend.color.operation);
                attachment.sourceAlphaBlendFactor = metalBlendFactor(target.blend.alpha.sourceFactor);
                attachment.destinationAlphaBlendFactor = metalBlendFactor(target.blend.alpha.destinationFactor);
                attachment.alphaBlendOperation = metalBlendOperation(target.blend.alpha.operation);
                attachment.writeMask = metalColorWriteMask(target.writeMask);
            }
        }

        [[nodiscard]] MTL4RenderPipelineDescriptor* metalMakeRenderPipelineDescriptor(
            const GraphicsPipelineDesc& desc,
            const MetalGraphicsPipelineInputs& inputs,
            std::uint32_t vertexBufferBaseIndex)
        {
            MTL4RenderPipelineDescriptor* const pipelineDesc = [MTL4RenderPipelineDescriptor new];
            pipelineDesc.vertexFunctionDescriptor =
                metalLibraryFunctionDescriptor(inputs.vertex.library, inputs.vertex.entryPoint);
            if (inputs.hasPixel)
            {
                pipelineDesc.fragmentFunctionDescriptor =
                    metalLibraryFunctionDescriptor(inputs.pixel.library, inputs.pixel.entryPoint);
            }
            pipelineDesc.vertexDescriptor = metalMakeVertexDescriptor(desc.vertexBuffers, vertexBufferBaseIndex);
            pipelineDesc.rasterSampleCount = desc.sampleCount;
            pipelineDesc.alphaToCoverageState =
                desc.alphaToCoverageEnabled ? MTL4AlphaToCoverageStateEnabled : MTL4AlphaToCoverageStateDisabled;
            pipelineDesc.rasterizationEnabled = desc.rasterizedStream.has_value();
            pipelineDesc.inputPrimitiveTopology = metalPrimitiveTopologyClass(desc.topology);
            if (!desc.label.empty())
            {
                pipelineDesc.label = metalNsString(desc.label);
            }
            metalConfigureColorAttachments(pipelineDesc, desc.colorTargets);
            return pipelineDesc;
        }

        [[nodiscard]] std::vector<std::byte> metalGraphicsPipelineFingerprint(const GraphicsPipelineDesc& desc,
                                                                              const MetalGraphicsPipelineInputs& inputs,
                                                                              std::uint32_t vertexBufferBaseIndex)
        {
            MetalPipelineFingerprintWriter writer;
            writer.writeShader(inputs.vertex);
            writer.write(inputs.hasPixel);
            if (inputs.hasPixel)
            {
                writer.writeShader(inputs.pixel);
            }
            for (std::size_t bufferIndex = 0; bufferIndex < desc.vertexBuffers.size(); ++bufferIndex)
            {
                const auto& buffer = desc.vertexBuffers[bufferIndex];
                if (buffer.attributes.empty())
                {
                    continue;
                }
                writer.write(static_cast<std::uint64_t>(vertexBufferBaseIndex + bufferIndex));
                writer.write(buffer.stride);
                writer.write(buffer.stepMode);
                writer.write(buffer.instanceStepRate);
                writer.write(static_cast<std::uint64_t>(buffer.attributes.size()));
                for (const auto& attribute : buffer.attributes)
                {
                    writer.write(attribute.format);
                    writer.write(attribute.offset);
                    writer.write(attribute.shaderLocation);
                }
            }
            writer.write(desc.sampleCount);
            writer.write(desc.alphaToCoverageEnabled);
            writer.write(desc.rasterizedStream.has_value());
            writer.write(metalPrimitiveTopologyClass(desc.topology));
            writer.write(static_cast<std::uint64_t>(desc.colorTargets.size()));
            for (const auto& target : desc.colorTargets)
            {
                writer.write(target.format);
                writer.write(target.writeMask);
                writer.write(target.blend.enabled);
                for (const auto* component : {&target.blend.color, &target.blend.alpha})
                {
                    writer.write(component->sourceFactor);
                    writer.write(component->destinationFactor);
                    writer.write(component->operation);
                }
            }
            return writer.take();
        }

        [[nodiscard]] Result<id<MTLRenderPipelineState>> metalRequireGraphicsCacheHit(
            MetalPipelineCacheRecord& cache,
            const MetalPipelineCacheIdentity& identity,
            const std::vector<std::byte>& fingerprint,
            MTL4RenderPipelineDescriptor* pipelineDesc)
        {
            std::scoped_lock const cacheLock(cache.mutex);
            const auto found = cache.live.find(identity);
            if (found == cache.live.end())
            {
                return std::unexpected(metalSelectionError(ErrorCode::pipelineCacheMiss,
                                                           "The Metal pipeline cache has no matching graphics entry."));
            }
            if (metalCacheEntryConflicts(found->second, fingerprint))
            {
                return std::unexpected(metalIncompatibleCacheEntryError(identity.kind));
            }
            if (found->second.graphics != nil)
            {
                return found->second.graphics;
            }
            const auto& segment = found->second.segment;
            if (segment == nullptr || segment->archive == nil)
            {
                return std::unexpected(metalSelectionError(ErrorCode::pipelineCacheMiss,
                                                           "The Metal pipeline cache has no matching graphics entry."));
            }
            id<MTLRenderPipelineState> pso = nil;
            NSError* archiveError = nil;
            @try
            {
                pso = [segment->archive newRenderPipelineStateWithDescriptor:pipelineDesc error:&archiveError];
            }
            @catch (NSException* exception)
            {
                return std::unexpected(metalExceptionError(exception, "newRenderPipelineStateWithDescriptor"));
            }
            if (pso == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::pipelineCacheMiss, "The Metal pipeline cache archive missed a graphics entry."));
            }
            found->second.graphics = pso;
            found->second.fingerprint = fingerprint;
            return pso;
        }

        [[nodiscard]] Result<id<MTLRenderPipelineState>> metalCompileGraphicsPipeline(
            MTL4RenderPipelineDescriptor* pipelineDesc,
            const std::shared_ptr<MetalPipelineCacheRecord>& cache,
            const MetalPipelineCacheIdentity& identity,
            const std::vector<std::byte>& fingerprint,
            rachis::detail::TimedMutex& deviceMutex,
            const std::shared_ptr<MetalDeviceNative>& native)
        {
            std::shared_lock<std::shared_mutex> serializeLock;
            if (cache != nullptr)
            {
                serializeLock = std::shared_lock(cache->serializeMutex);
                if (auto compatible = metalCheckCacheEntryCompatible(*cache, identity, fingerprint); !compatible)
                {
                    return std::unexpected(std::move(compatible.error()));
                }
            }
            NSError* error = nil;
            auto choice = metalChoosePipelineCompiler(
                cache, deviceMutex, native, "Metal createGraphicsPipeline requires an active compiler.");
            if (!choice)
            {
                return std::unexpected(std::move(choice.error()));
            }
            MTL4CompilerTaskOptions* options = metalLookupArchiveTaskOptions(choice->lookupArchives);
            id<MTLRenderPipelineState> pso = nil;
            @try
            {
                pso = [choice->compiler newRenderPipelineStateWithDescriptor:pipelineDesc
                                                         compilerTaskOptions:options
                                                                       error:&error];
            }
            @catch (NSException* exception)
            {
                return std::unexpected(metalExceptionError(exception, "newRenderPipelineStateWithDescriptor"));
            }
            if (pso == nil)
            {
                return std::unexpected(
                    metalNsError(metalPipelineCompileErrorCode(error), error, "newRenderPipelineStateWithDescriptor"));
            }
            if (cache != nullptr)
            {
                std::scoped_lock const cacheLock(cache->mutex);
                auto& live = cache->live[identity];
                if (metalCacheEntryConflicts(live, fingerprint))
                {
                    return std::unexpected(metalIncompatibleCacheEntryError(identity.kind));
                }
                live.graphics = pso;
                live.fingerprint = fingerprint;
                live.inSerializer = true;
                cache->serializerDirty = true;
            }
            return pso;
        }

        [[nodiscard]] MetalGraphicsPipelineState metalMakeGraphicsPipelineState(const GraphicsPipelineDesc& desc,
                                                                                MetalColorTargetSummary&& colorTargets,
                                                                                std::uint32_t vertexBufferBaseIndex)
        {
            std::vector<std::uint32_t> vertexBufferAlignments;
            vertexBufferAlignments.reserve(desc.vertexBuffers.size());
            for (const auto& buffer : desc.vertexBuffers)
            {
                vertexBufferAlignments.push_back(detail::vertexAttributesAlignment(buffer.attributes));
            }
            return MetalGraphicsPipelineState{
                .topology = desc.topology,
                .frontFace = desc.frontFace,
                .cullMode = desc.cullMode,
                .fillMode = desc.fillMode,
                .depthBias = desc.depthBias,
                .depthBiasClamp = desc.depthBiasClamp,
                .depthBiasSlopeScale = desc.depthBiasSlopeScale,
                .depthClipEnabled = desc.depthClipEnabled,
                .depthStencil = desc.depthStencil,
                .sampleCount = desc.sampleCount,
                .sampleMask = desc.sampleMask,
                .alphaToCoverageEnabled = desc.alphaToCoverageEnabled,
                .stripIndexFormat = desc.stripIndexFormat,
                .colorFormats = std::move(colorTargets.colorFormats),
                .depthStencilFormat = desc.depthStencil.format,
                .viewMask = desc.viewMask,
                .vertexBufferCount = static_cast<std::uint32_t>(desc.vertexBuffers.size()),
                .vertexBufferBaseIndex = vertexBufferBaseIndex,
                .vertexBufferAlignments = std::move(vertexBufferAlignments),
                .usesBlendConstants = colorTargets.usesBlendConstants,
                .stencilEnabled = desc.depthStencil.stencilTestEnabled,
                .rasterizationEnabled = desc.rasterizedStream.has_value(),
            };
        }

        [[nodiscard]] std::shared_ptr<MetalPipelineRecord> metalMakeGraphicsPipelineRecord(
            MetalGraphicsPipelineInputs& inputs,
            id<MTLRenderPipelineState> pso,
            id<MTLDepthStencilState> depthStencil,
            MetalGraphicsAttachmentLayout&& attachments,
            const GraphicsPipelineDesc& desc)
        {
            auto record = std::make_shared<MetalPipelineRecord>();
            record->kind = MetalPipelineKind::graphics;
            record->layout = std::move(inputs.layout);
            record->setRequired = metalRequiredBindingSets(*record->layout);
            record->graphics = pso;
            record->depthStencil = depthStencil;
            record->vertexStageBindings = inputs.vertex.stageBindings;
            record->hasPixelShader = inputs.hasPixel;
            if (inputs.hasPixel)
            {
                record->pixelStageBindings = inputs.pixel.stageBindings;
            }
            record->pushConstantBytes = record->layout->pushConstantBytes;
            record->reservePushConstantBuffer = record->layout->reservePushConstantBuffer;
            record->graphicsState = metalMakeGraphicsPipelineState(
                desc, std::move(attachments.colorTargets), attachments.vertexBufferBaseIndex);
            record->label = std::string(desc.label);
            return record;
        }

    } // namespace

    Result<PipelineHandle> MetalDevice::createGraphicsPipeline(const GraphicsPipelineDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (auto validated = metalValidateGraphicsPipelineDesc(desc, capabilities_); !validated)
        {
            return std::unexpected(std::move(validated.error()));
        }

        auto locations = metalCollectVertexLocations(desc.vertexBuffers, capabilities_.vertexInput);
        if (!locations)
        {
            return std::unexpected(std::move(locations.error()));
        }
        auto inputs = metalAcquireGraphicsPipelineInputs(mutex_, native_, desc);
        if (!inputs)
        {
            return std::unexpected(std::move(inputs.error()));
        }
        if (auto interfaces = metalValidateGraphicsShaderInterfaces(*inputs, *locations); !interfaces)
        {
            return std::unexpected(std::move(interfaces.error()));
        }
        auto attachments = metalResolveGraphicsAttachmentLayout(*this, desc, *inputs);
        if (!attachments)
        {
            return std::unexpected(std::move(attachments.error()));
        }

        id<MTLRenderPipelineState> pso = nil;
        @autoreleasepool
        {
            auto depthStencil = metalMakePipelineDepthStencilState(inputs->mtlDevice, desc.depthStencil);
            if (!depthStencil)
            {
                return std::unexpected(std::move(depthStencil.error()));
            }
            MTL4RenderPipelineDescriptor* const pipelineDesc =
                metalMakeRenderPipelineDescriptor(desc, *inputs, attachments->vertexBufferBaseIndex);
            const MetalPipelineCacheIdentity identity{.key = desc.cacheKey, .kind = MetalPipelineCacheKind::graphics};
            const auto& cache = inputs->cache;
            const auto fingerprint =
                cache != nullptr ? metalGraphicsPipelineFingerprint(desc, *inputs, attachments->vertexBufferBaseIndex)
                                 : std::vector<std::byte>{};
            if (cache != nullptr && desc.cachePolicy == PipelineCachePolicy::requireHit)
            {
                auto hit = metalRequireGraphicsCacheHit(*cache, identity, fingerprint, pipelineDesc);
                if (!hit)
                {
                    return std::unexpected(std::move(hit.error()));
                }
                pso = *hit;
            }
            if (pso == nil)
            {
                auto compiled =
                    metalCompileGraphicsPipeline(pipelineDesc, cache, identity, fingerprint, mutex_, native_);
                if (!compiled)
                {
                    return std::unexpected(std::move(compiled.error()));
                }
                pso = *compiled;
            }
            auto record = metalMakeGraphicsPipelineRecord(*inputs, pso, *depthStencil, std::move(*attachments), desc);
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal createGraphicsPipeline requires an active device."));
            }
            return native_->pipelines.emplace(std::move(record));
        }
    }

    Result<PipelineCompilationHandle> MetalDevice::beginComputePipelineCompilation(const ComputePipelineDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        auto created = createComputePipeline(desc);
        auto record = std::make_shared<MetalPipelineCompilationRecord>();
        if (created)
        {
            record->state = PipelineCompilationState::succeeded;
            record->pipeline = *created;
        }
        else
        {
            record->state = PipelineCompilationState::failed;
            record->error = created.error();
        }
        if (auto blocked = requireActive(); !blocked)
        {
            if (created)
            {
                std::scoped_lock const lock(mutex_);
                if (native_ != nullptr)
                {
                    native_->pipelines.erase(*created);
                }
            }
            return std::unexpected(std::move(blocked.error()));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "Metal beginComputePipelineCompilation requires an active device."));
        }
        try
        {
            return native_->pipelineCompilations.emplace(std::move(record));
        }
        catch (...)
        {
            if (created)
            {
                native_->pipelines.erase(*created);
            }
            throw;
        }
    }

    Result<PipelineCompilationHandle> MetalDevice::beginGraphicsPipelineCompilation(const GraphicsPipelineDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        auto created = createGraphicsPipeline(desc);
        auto record = std::make_shared<MetalPipelineCompilationRecord>();
        if (created)
        {
            record->state = PipelineCompilationState::succeeded;
            record->pipeline = *created;
        }
        else
        {
            record->state = PipelineCompilationState::failed;
            record->error = created.error();
        }
        if (auto blocked = requireActive(); !blocked)
        {
            if (created)
            {
                std::scoped_lock const lock(mutex_);
                if (native_ != nullptr)
                {
                    native_->pipelines.erase(*created);
                }
            }
            return std::unexpected(std::move(blocked.error()));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "Metal beginGraphicsPipelineCompilation requires an active device."));
        }
        try
        {
            return native_->pipelineCompilations.emplace(std::move(record));
        }
        catch (...)
        {
            if (created)
            {
                native_->pipelines.erase(*created);
            }
            throw;
        }
    }

    Result<PipelineCompilationState> MetalDevice::pipelineCompilationState(PipelineCompilationHandle compilation)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "Metal pipelineCompilationState requires an active device."));
        }
        auto* found = native_->pipelineCompilations.find(compilation);
        if (found == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The pipeline compilation handle is invalid or stale."));
        }
        return (*found)->state;
    }

    Result<PipelineHandle> MetalDevice::waitPipelineCompilation(PipelineCompilationHandle compilation)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "Metal waitPipelineCompilation requires an active device."));
        }
        auto* found = native_->pipelineCompilations.find(compilation);
        if (found == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The pipeline compilation handle is invalid or stale."));
        }
        auto record = *found;
        native_->pipelineCompilations.erase(compilation);
        if (record->state != PipelineCompilationState::succeeded)
        {
            return std::unexpected(record->error);
        }
        return record->pipeline;
    }

    Status MetalDevice::destroyPipelineCompilation(PipelineCompilationHandle compilation)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "Metal destroyPipelineCompilation requires an active device."));
        }
        auto* found = native_->pipelineCompilations.find(compilation);
        if (found == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The pipeline compilation handle is invalid or stale."));
        }
        if ((*found)->state == PipelineCompilationState::succeeded)
        {
            native_->pipelines.erase((*found)->pipeline);
        }
        native_->pipelineCompilations.erase(compilation);
        return {};
    }

    Status MetalDevice::destroyPipeline(PipelineHandle pipeline)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || !native_->pipelines.erase(pipeline))
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The pipeline handle is invalid or stale."));
        }
        return {};
    }

} // namespace rachis::metal
