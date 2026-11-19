#pragma once

#include "Internal/StageLinkage.h"
#include "Rachis/Error.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rachis::d3d12::dxil
{

    enum class SignaturePart : std::uint8_t
    {
        input,
        output,
        primitiveOutput,
    };

    struct SignatureVariable final
    {
        detail::StageInterfaceVariable variable;
        std::uint32_t stream = 0;
    };

    namespace signature_detail
    {

        [[nodiscard]] constexpr std::uint32_t fourCC(char a, char b, char c, char d) noexcept
        {
            return static_cast<std::uint32_t>(static_cast<unsigned char>(a)) |
                   (static_cast<std::uint32_t>(static_cast<unsigned char>(b)) << 8U) |
                   (static_cast<std::uint32_t>(static_cast<unsigned char>(c)) << 16U) |
                   (static_cast<std::uint32_t>(static_cast<unsigned char>(d)) << 24U);
        }

        [[nodiscard]] inline std::uint32_t readU32(std::span<const std::byte> bytes, std::size_t offset) noexcept
        {
            std::uint32_t value = 0;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return value;
        }

        [[nodiscard]] inline std::uint8_t readU8(std::span<const std::byte> bytes, std::size_t offset) noexcept
        {
            return std::to_integer<std::uint8_t>(bytes[offset]);
        }

        [[nodiscard]] constexpr std::optional<detail::StageComponentType> componentType(std::uint32_t type) noexcept
        {
            switch (type)
            {
            case 1:
                return detail::StageComponentType::uint32;
            case 2:
                return detail::StageComponentType::sint32;
            case 3:
                return detail::StageComponentType::float32;
            case 4:
                return detail::StageComponentType::uint16;
            case 5:
                return detail::StageComponentType::sint16;
            case 6:
                return detail::StageComponentType::float16;
            case 7:
                return detail::StageComponentType::uint64;
            case 8:
                return detail::StageComponentType::sint64;
            case 9:
                return detail::StageComponentType::float64;
            default:
                return std::nullopt;
            }
        }

        [[nodiscard]] inline Error malformed(std::string_view what)
        {
            return Error{ErrorCode::invalidArgument, 0, "The shader's DXIL container " + std::string(what) + "."};
        }

        class Cursor final
        {
        public:
            explicit Cursor(std::span<const std::byte> bytes) noexcept : bytes_(bytes)
            {
            }

            [[nodiscard]] std::optional<std::uint32_t> u32() noexcept
            {
                if (bytes_.size() - offset_ < sizeof(std::uint32_t))
                {
                    return std::nullopt;
                }
                const auto value = readU32(bytes_, offset_);
                offset_ += sizeof(std::uint32_t);
                return value;
            }

            [[nodiscard]] std::optional<std::span<const std::byte>> take(std::uint64_t size) noexcept
            {
                if (size > bytes_.size() - offset_)
                {
                    return std::nullopt;
                }
                const auto taken = bytes_.subspan(offset_, static_cast<std::size_t>(size));
                offset_ += static_cast<std::size_t>(size);
                return taken;
            }

        private:
            std::span<const std::byte> bytes_;
            std::size_t offset_ = 0;
        };

        inline constexpr std::size_t runtimeInfo1Size = 36;
        inline constexpr std::size_t inputCountOffset = 28;
        inline constexpr std::size_t signatureElement0Size = 16;
        inline constexpr std::uint8_t arbitrarySemantic = 0;

        [[nodiscard]] inline Result<std::optional<std::span<const std::byte>>> findPart(
            std::span<const std::byte> container, std::uint32_t wanted)
        {
            constexpr std::size_t headerSize = 32;
            if (container.size() < headerSize || readU32(container, 0) != fourCC('D', 'X', 'B', 'C'))
            {
                return std::unexpected(malformed("has no DXBC header"));
            }
            const auto containerSize = readU32(container, 24);
            const auto partCount = readU32(container, 28);
            if (containerSize > container.size() || containerSize < headerSize ||
                partCount > (containerSize - headerSize) / sizeof(std::uint32_t))
            {
                return std::unexpected(malformed("has an invalid size or part count"));
            }
            const auto bytes = container.first(containerSize);
            for (std::uint32_t index = 0; index < partCount; ++index)
            {
                const auto partOffset = readU32(bytes, headerSize + index * sizeof(std::uint32_t));
                if (partOffset > bytes.size() || bytes.size() - partOffset < 8)
                {
                    return std::unexpected(malformed("has a part outside the container"));
                }
                const auto partSize = readU32(bytes, partOffset + 4);
                if (partSize > bytes.size() - partOffset - 8)
                {
                    return std::unexpected(malformed("has a part that runs past the container"));
                }
                if (readU32(bytes, partOffset) == wanted)
                {
                    return std::optional{bytes.subspan(partOffset + 8, partSize)};
                }
            }
            return std::optional<std::span<const std::byte>>{};
        }

        [[nodiscard]] inline std::string_view cStringAt(std::span<const std::byte> table, std::uint32_t offset) noexcept
        {
            if (offset >= table.size())
            {
                return {};
            }
            const auto* begin = reinterpret_cast<const char*>(table.data()) + offset;
            const auto available = table.size() - offset;
            const auto* end = static_cast<const char*>(std::memchr(begin, '\0', available));
            return std::string_view(begin, end != nullptr ? static_cast<std::size_t>(end - begin) : available);
        }

        struct InputReadMask final
        {
            std::string_view name;
            std::uint32_t index = 0;
            std::uint8_t read = 0;
        };

        [[nodiscard]] inline Result<std::vector<InputReadMask>> inputReadMasks(std::span<const std::byte> container)
        {
            auto found = findPart(container, fourCC('I', 'S', 'G', '1'));
            if (!found)
            {
                return std::unexpected(std::move(found.error()));
            }
            std::vector<InputReadMask> masks;
            if (!found->has_value())
            {
                return masks;
            }
            const auto data = **found;
            constexpr std::size_t elementSize = 32;
            if (data.size() < 8)
            {
                return std::unexpected(malformed("has a truncated input signature"));
            }
            const auto count = readU32(data, 0);
            const auto offset = readU32(data, 4);
            if (offset > data.size() || count > (data.size() - offset) / elementSize)
            {
                return std::unexpected(malformed("has input signature elements outside their part"));
            }
            for (std::uint32_t element = 0; element < count; ++element)
            {
                const auto base = offset + element * elementSize;
                masks.push_back(InputReadMask{
                    .name = cStringAt(data, readU32(data, base + 4)),
                    .index = readU32(data, base + 8),
                    .read = readU8(data, base + 25),
                });
            }
            return masks;
        }

        [[nodiscard]] inline bool equalsIgnoringCase(std::string_view left, std::string_view right) noexcept
        {
            if (left.size() != right.size())
            {
                return false;
            }
            for (std::size_t index = 0; index < left.size(); ++index)
            {
                const auto lower = [](char ch) { return ch >= 'A' && ch <= 'Z' ? static_cast<char>(ch + 32) : ch; };
                if (lower(left[index]) != lower(right[index]))
                {
                    return false;
                }
            }
            return true;
        }

        [[nodiscard]] inline bool inputIsRead(std::span<const InputReadMask> masks,
                                              std::string_view name,
                                              std::uint32_t index) noexcept
        {
            for (const auto& mask : masks)
            {
                if (mask.index == index && equalsIgnoringCase(mask.name, name))
                {
                    return (mask.read & 0xFU) != 0;
                }
            }
            return true;
        }

    } // namespace signature_detail

    [[nodiscard]] inline bool isContainer(std::span<const std::byte> bytes) noexcept
    {
        return bytes.size() >= sizeof(std::uint32_t) &&
               signature_detail::readU32(bytes, 0) == signature_detail::fourCC('D', 'X', 'B', 'C');
    }

    namespace signature_detail
    {

        [[nodiscard]] inline Result<std::optional<std::span<const std::byte>>> runtimeInfo(
            std::span<const std::byte> container)
        {
            auto found = findPart(container, fourCC('P', 'S', 'V', '0'));
            if (!found)
            {
                return std::unexpected(std::move(found.error()));
            }
            if (!found->has_value())
            {
                return std::optional<std::span<const std::byte>>{};
            }
            Cursor cursor(**found);
            const auto infoSize = cursor.u32();
            const auto info = infoSize.has_value() ? cursor.take(*infoSize) : std::nullopt;
            if (!info.has_value())
            {
                return std::unexpected(malformed("has a truncated PSV0 runtime info"));
            }
            return std::optional{*info};
        }

    } // namespace signature_detail

    struct WaveLaneCountRange final
    {
        std::uint32_t minimum = 0;
        std::uint32_t maximum = 0;
    };

    [[nodiscard]] inline Result<std::optional<WaveLaneCountRange>> readExpectedWaveLaneCounts(
        std::span<const std::byte> container)
    {
        using namespace signature_detail;
        constexpr std::size_t runtimeInfo0Size = 24;
        constexpr std::size_t minimumWaveLaneCountOffset = 16;
        constexpr std::size_t maximumWaveLaneCountOffset = 20;
        auto info = runtimeInfo(container);
        if (!info)
        {
            return std::unexpected(std::move(info.error()));
        }
        if (!info->has_value() || (*info)->size() < runtimeInfo0Size)
        {
            return std::optional<WaveLaneCountRange>{};
        }
        const WaveLaneCountRange range{
            .minimum = readU32(**info, minimumWaveLaneCountOffset),
            .maximum = readU32(**info, maximumWaveLaneCountOffset),
        };
        if (range.minimum == 0)
        {
            return std::optional<WaveLaneCountRange>{};
        }
        return std::optional{range};
    }

    struct ThreadGroupSize final
    {
        std::uint32_t x = 0;
        std::uint32_t y = 0;
        std::uint32_t z = 0;

        [[nodiscard]] friend constexpr bool operator==(const ThreadGroupSize&,
                                                       const ThreadGroupSize&) noexcept = default;
    };

    [[nodiscard]] inline Result<std::optional<ThreadGroupSize>> readThreadGroupSize(
        std::span<const std::byte> container)
    {
        using namespace signature_detail;
        constexpr std::size_t runtimeInfo2Size = 48;
        constexpr std::size_t numThreadsOffset = 36;
        auto info = runtimeInfo(container);
        if (!info)
        {
            return std::unexpected(std::move(info.error()));
        }
        if (!info->has_value() || (*info)->size() < runtimeInfo2Size)
        {
            return std::optional<ThreadGroupSize>{};
        }
        const ThreadGroupSize size{
            .x = readU32(**info, numThreadsOffset),
            .y = readU32(**info, numThreadsOffset + 4),
            .z = readU32(**info, numThreadsOffset + 8),
        };
        if (size.x == 0 || size.y == 0 || size.z == 0)
        {
            return std::optional<ThreadGroupSize>{};
        }
        return std::optional{size};
    }

    [[nodiscard]] inline Status validateComputeWaveSize(std::span<const std::byte> container,
                                                        std::uint32_t requiredSize,
                                                        const std::optional<ThreadGroupSize>& interfaceWorkgroup,
                                                        WaveLaneCountRange device)
    {
        if (requiredSize == 0 && !isContainer(container))
        {
            return {};
        }
        auto declared = readExpectedWaveLaneCounts(container);
        if (!declared)
        {
            return std::unexpected(std::move(declared.error()));
        }
        if (requiredSize == 0)
        {
            if (declared->has_value() &&
                ((*declared)->maximum < device.minimum || (*declared)->minimum > device.maximum))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The compute shader's [WaveSize] range does not overlap this device's subgroup size range.",
                });
            }
            return {};
        }
        if (!declared->has_value() || (*declared)->minimum != requiredSize || (*declared)->maximum != requiredSize)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "On Direct3D 12, a compute pipeline that requires a subgroup size needs a shader that declares "
                "[WaveSize] with that size.",
            });
        }
        if (requiredSize < device.minimum || requiredSize > device.maximum)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The compute shader's [WaveSize] is outside this device's subgroup size range.",
            });
        }
        auto threads = readThreadGroupSize(container);
        if (!threads)
        {
            return std::unexpected(std::move(threads.error()));
        }
        if (!threads->has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "On Direct3D 12, a compute pipeline that requires a subgroup size needs DXIL whose PSV0 runtime "
                "info records the [numthreads] size.",
            });
        }
        if (interfaceWorkgroup.has_value() && *interfaceWorkgroup != **threads)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The shader interface's workgroup size does not match the [numthreads] size in the DXIL.",
            });
        }
        if ((*threads)->x % requiredSize != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A compute pipeline that requires a subgroup size needs a [numthreads] width that is a multiple of "
                "that size.",
            });
        }
        return {};
    }

    [[nodiscard]] inline Result<std::optional<std::vector<SignatureVariable>>> readStageSignature(
        std::span<const std::byte> container, SignaturePart part)
    {
        using namespace signature_detail;
        auto found = findPart(container, fourCC('P', 'S', 'V', '0'));
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        if (!found->has_value())
        {
            return std::optional<std::vector<SignatureVariable>>{};
        }
        Cursor cursor(**found);
        const auto infoSize = cursor.u32();
        const auto info = infoSize.has_value() ? cursor.take(*infoSize) : std::nullopt;
        if (!info.has_value())
        {
            return std::unexpected(malformed("has a truncated PSV0 runtime info"));
        }
        if (info->size() < runtimeInfo1Size)
        {
            return std::optional<std::vector<SignatureVariable>>{};
        }
        const std::uint32_t inputs = readU8(*info, inputCountOffset);
        const std::uint32_t outputs = readU8(*info, inputCountOffset + 1);
        const std::uint32_t primitives = readU8(*info, inputCountOffset + 2);

        const auto resources = cursor.u32();
        if (!resources.has_value())
        {
            return std::unexpected(malformed("has a truncated PSV0 resource table"));
        }
        if (*resources != 0)
        {
            const auto bindSize = cursor.u32();
            if (!bindSize.has_value() || !cursor.take(std::uint64_t{*bindSize} * *resources).has_value())
            {
                return std::unexpected(malformed("has a truncated PSV0 resource table"));
            }
        }
        const auto stringSize = cursor.u32();
        const auto strings = stringSize.has_value() ? cursor.take(*stringSize) : std::nullopt;
        const auto indexCount = strings.has_value() ? cursor.u32() : std::nullopt;
        const auto indices =
            indexCount.has_value() ? cursor.take(std::uint64_t{*indexCount} * sizeof(std::uint32_t)) : std::nullopt;
        if (!indices.has_value())
        {
            return std::unexpected(malformed("has truncated PSV0 string or semantic index tables"));
        }
        std::vector<SignatureVariable> variables;
        if (inputs + outputs + primitives == 0)
        {
            return std::optional{std::move(variables)};
        }
        const auto elementSize = cursor.u32();
        if (!elementSize.has_value() || *elementSize < signatureElement0Size)
        {
            return std::unexpected(malformed("has an unknown PSV0 signature element size"));
        }
        std::vector<InputReadMask> readMasks;
        if (part == SignaturePart::input)
        {
            auto masks = inputReadMasks(container);
            if (!masks)
            {
                return std::unexpected(std::move(masks.error()));
            }
            readMasks = std::move(*masks);
        }
        const auto skipped = part == SignaturePart::input    ? 0U
                             : part == SignaturePart::output ? inputs
                                                             : inputs + outputs;
        const auto count = part == SignaturePart::input ? inputs : part == SignaturePart::output ? outputs : primitives;
        if (!cursor.take(std::uint64_t{*elementSize} * skipped).has_value())
        {
            return std::unexpected(malformed("has truncated PSV0 signature elements"));
        }
        std::uint32_t location = 0;
        for (std::uint32_t index = 0; index < count; ++index)
        {
            const auto element = cursor.take(*elementSize);
            if (!element.has_value())
            {
                return std::unexpected(malformed("has truncated PSV0 signature elements"));
            }
            const auto rows = readU8(*element, 8);
            const auto columns = static_cast<std::uint32_t>(readU8(*element, 10) & 0xFU);
            const auto kind = readU8(*element, 11);
            const auto type = componentType(readU8(*element, 12));
            const auto stream = static_cast<std::uint32_t>((readU8(*element, 14) >> 4U) & 0x3U);
            if (kind != arbitrarySemantic || !type.has_value() || rows == 0 || columns == 0 ||
                columns > detail::stageComponentsPerLocation)
            {
                continue;
            }
            const auto name = cStringAt(*strings, readU32(*element, 0));
            const auto firstIndex = readU32(*element, 4);
            for (std::uint32_t row = 0; row < rows; ++row)
            {
                const auto semanticIndex = firstIndex + row < *indexCount
                                               ? readU32(*indices, (firstIndex + row) * sizeof(std::uint32_t))
                                               : row;
                if (part == SignaturePart::input && !inputIsRead(readMasks, name, semanticIndex))
                {
                    ++location;
                    continue;
                }
                variables.push_back(SignatureVariable{
                    .variable =
                        detail::StageInterfaceVariable{
                            .location = location++,
                            .firstComponent = 0,
                            .componentCount = columns,
                            .componentType = *type,
                        },
                    .stream = stream,
                });
            }
        }
        return std::optional{std::move(variables)};
    }

    [[nodiscard]] inline Result<std::optional<bool>> bindsConstantBuffer(std::span<const std::byte> container,
                                                                         std::uint32_t shaderRegister,
                                                                         std::uint32_t registerSpace)
    {
        using namespace signature_detail;
        constexpr std::uint32_t constantBufferType = 2;
        constexpr std::size_t bindInfo0Size = 16;
        auto found = findPart(container, fourCC('P', 'S', 'V', '0'));
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        if (!found->has_value())
        {
            return std::optional<bool>{};
        }
        Cursor cursor(**found);
        const auto infoSize = cursor.u32();
        if (!infoSize.has_value() || !cursor.take(*infoSize).has_value())
        {
            return std::unexpected(malformed("has a truncated PSV0 runtime info"));
        }
        const auto resources = cursor.u32();
        if (!resources.has_value())
        {
            return std::unexpected(malformed("has a truncated PSV0 resource table"));
        }
        if (*resources == 0)
        {
            return std::optional{false};
        }
        const auto bindSize = cursor.u32();
        if (!bindSize.has_value() || *bindSize < bindInfo0Size)
        {
            return std::unexpected(malformed("has an unknown PSV0 resource record size"));
        }
        const auto table = cursor.take(std::uint64_t{*bindSize} * *resources);
        if (!table.has_value())
        {
            return std::unexpected(malformed("has a truncated PSV0 resource table"));
        }
        for (std::uint32_t index = 0; index < *resources; ++index)
        {
            const auto record = table->subspan(std::size_t{index} * *bindSize, *bindSize);
            const auto type = readU32(record, 0);
            const auto space = readU32(record, 4);
            const auto lowerBound = readU32(record, 8);
            const auto upperBound = readU32(record, 12);
            if (type == constantBufferType && space == registerSpace && lowerBound <= shaderRegister &&
                shaderRegister <= upperBound)
            {
                return std::optional{true};
            }
        }
        return std::optional{false};
    }

    [[nodiscard]] inline Status validateDrawIndexDeclaration(std::span<const std::byte> container,
                                                             std::uint32_t shaderRegister,
                                                             std::uint32_t registerSpace,
                                                             bool declared)
    {
        if (!declared && !isContainer(container))
        {
            return {};
        }
        auto binds = bindsConstantBuffer(container, shaderRegister, registerSpace);
        if (!binds)
        {
            return std::unexpected(std::move(binds.error()));
        }
        if (!binds->has_value())
        {
            if (!declared)
            {
                return {};
            }
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The shader interface declares the draw index, but the bytecode has no PSV0 resource table that "
                "shows its constant buffer at register b0, space5.",
            });
        }
        if (**binds == declared)
        {
            return {};
        }
        return std::unexpected(Error{
            ErrorCode::invalidArgument,
            0,
            declared ? "The shader interface declares the draw index, but the DXIL binds no constant buffer at "
                       "register b0, space5."
                     : "The DXIL binds the draw-index constant buffer at register b0, space5, but the shader "
                       "interface does not declare the draw index.",
        });
    }

} // namespace rachis::d3d12::dxil
