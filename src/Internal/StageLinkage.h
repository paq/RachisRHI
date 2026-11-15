#pragma once

#include "Rachis/Binding.h"
#include "Rachis/Error.h"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace rachis::detail
{
    enum class StageComponentType : std::uint8_t
    {
        float16,
        float32,
        float64,
        sint16,
        sint32,
        sint64,
        uint16,
        uint32,
        uint64,
    };

    inline constexpr std::uint32_t stageComponentsPerLocation = 4;

    struct StageInterfaceVariable final
    {
        std::uint32_t location = 0;
        std::uint32_t firstComponent = 0;
        std::uint32_t componentCount = stageComponentsPerLocation;
        StageComponentType componentType = StageComponentType::float32;

        friend constexpr bool operator==(const StageInterfaceVariable&,
                                         const StageInterfaceVariable&) noexcept = default;
    };

    [[nodiscard]] constexpr std::string_view stageComponentTypeName(StageComponentType type) noexcept
    {
        switch (type)
        {
        case StageComponentType::float16:
            return "float16";
        case StageComponentType::float32:
            return "float32";
        case StageComponentType::float64:
            return "float64";
        case StageComponentType::sint16:
            return "sint16";
        case StageComponentType::sint32:
            return "sint32";
        case StageComponentType::sint64:
            return "sint64";
        case StageComponentType::uint16:
            return "uint16";
        case StageComponentType::uint32:
            return "uint32";
        case StageComponentType::uint64:
            return "uint64";
        }
        return "unknown";
    }

    [[nodiscard]] constexpr std::string_view stageLinkageProducerName(ShaderStage stage) noexcept
    {
        switch (stage)
        {
        case ShaderStage::vertex:
            return "vertex shader";
        case ShaderStage::domain:
            return "domain shader";
        case ShaderStage::geometry:
            return "geometry shader";
        case ShaderStage::mesh:
            return "mesh shader";
        default:
            return "last pre-rasterization stage";
        }
    }

    [[nodiscard]] constexpr bool isValidStageInterfaceVariable(const StageInterfaceVariable& variable) noexcept
    {
        return variable.componentCount != 0 && variable.firstComponent < stageComponentsPerLocation &&
               variable.componentCount <= stageComponentsPerLocation - variable.firstComponent;
    }

    [[nodiscard]] inline Status validateStageLinkage(ShaderStage producerStage,
                                                     std::span<const StageInterfaceVariable> producerOutputs,
                                                     std::span<const StageInterfaceVariable> pixelInputs)
    {
        const auto producer = std::string(stageLinkageProducerName(producerStage));
        for (const auto& output : producerOutputs)
        {
            if (!isValidStageInterfaceVariable(output))
            {
                return std::unexpected(Error{
                    ErrorCode::validationError,
                    0,
                    "The " + producer + " declares an output at location " + std::to_string(output.location) +
                        " with an invalid component range.",
                });
            }
        }
        for (const auto& input : pixelInputs)
        {
            if (!isValidStageInterfaceVariable(input))
            {
                return std::unexpected(Error{
                    ErrorCode::validationError,
                    0,
                    "The pixel shader declares an input at location " + std::to_string(input.location) +
                        " with an invalid component range.",
                });
            }
            for (std::uint32_t component = input.firstComponent;
                 component < input.firstComponent + input.componentCount;
                 ++component)
            {
                const StageInterfaceVariable* writer = nullptr;
                for (const auto& output : producerOutputs)
                {
                    if (output.location == input.location && output.firstComponent <= component &&
                        component < output.firstComponent + output.componentCount)
                    {
                        writer = &output;
                        break;
                    }
                }
                const auto where =
                    "location " + std::to_string(input.location) + " component " + std::to_string(component);
                if (writer == nullptr)
                {
                    return std::unexpected(Error{
                        ErrorCode::validationError,
                        0,
                        "The pixel shader reads " + where + ", which the " + producer + " does not write.",
                    });
                }
                if (writer->componentType != input.componentType)
                {
                    return std::unexpected(Error{
                        ErrorCode::validationError,
                        0,
                        "The pixel shader reads " + where + " as " +
                            std::string(stageComponentTypeName(input.componentType)) + ", but the " + producer +
                            " writes it as " + std::string(stageComponentTypeName(writer->componentType)) + ".",
                    });
                }
            }
        }
        return {};
    }

} // namespace rachis::detail
