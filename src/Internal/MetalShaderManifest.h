#pragma once

#include "Internal/StageLinkage.h"
#include "Rachis/Error.h"
#include "Rachis/Pipeline.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rachis::metal
{

    inline constexpr std::string_view shaderManifestMagic = "PNMS";
    inline constexpr std::uint32_t shaderManifestVersion = 1;
    inline constexpr std::uint32_t shaderManifestHeaderBytes = 16;
    inline constexpr std::uint32_t reservedPushConstantBufferIndex = 0;

    struct MetalShaderPayload final
    {
        std::span<const std::byte> json;
        std::span<const std::byte> metallib;
    };

    struct MetalReflectedResource final
    {
        std::string name;
        ShaderResourceType type = ShaderResourceType::uniformBuffer;
        std::uint32_t metalIndex = 0;
        std::uint32_t count = 1;
        bool pushConstant = false;
        std::uint32_t uniformBytes = 0;
        std::optional<std::uint32_t> set{};
        std::optional<std::uint32_t> binding{};
        bool argumentBuffer = false;
        std::uint32_t argumentCapacity = 0;
    };

    struct MetalReflectedField final
    {
        std::string path;
        std::uint32_t offset = 0;
        std::uint32_t size = 0;
    };

    struct MetalReflectedStageInterface final
    {
        std::vector<rachis::detail::StageInterfaceVariable> inputs{};
        std::vector<rachis::detail::StageInterfaceVariable> outputs{};
    };

    struct MetalReflectedLayout final
    {
        std::string entryPoint{};
        ShaderStage stage = ShaderStage::none;
        std::optional<ShaderWorkgroupSize> workgroupSize{};
        std::vector<MetalReflectedResource> resources;
        std::vector<MetalReflectedField> uniformFields{};
        std::uint32_t pushConstantBytes = 0;
        std::uint32_t pushConstantMetalIndex = reservedPushConstantBufferIndex;
        std::optional<MetalReflectedStageInterface> stageInterface{};
    };

    struct MetalResolvedResourceBinding final
    {
        std::uint32_t set = 0;
        std::uint32_t binding = 0;
        ShaderResourceType type = ShaderResourceType::uniformBuffer;
        std::uint32_t count = 1;
        std::uint32_t metalIndex = 0;
        bool argumentBuffer = false;
        std::uint32_t argumentCapacity = 0;
    };

    struct MetalResolvedStageBindings final
    {
        std::vector<MetalResolvedResourceBinding> resources{};
        bool usesPushConstants = false;
        std::uint32_t pushConstantBytes = 0;
        std::uint32_t pushConstantMetalIndex = reservedPushConstantBufferIndex;
    };

    [[nodiscard]] std::vector<std::byte> packShaderManifest(std::span<const std::byte> json,
                                                            std::span<const std::byte> metallib);

    [[nodiscard]] Result<MetalShaderPayload> unpackShaderManifest(std::span<const std::byte> bytecode);

    [[nodiscard]] Result<MetalReflectedLayout> parseSlangReflectionJson(std::string_view json,
                                                                        std::string_view entryPoint,
                                                                        ShaderStage stage);

    [[nodiscard]] Result<MetalResolvedStageBindings> resolveShaderInterface(const MetalReflectedLayout& layout,
                                                                            const ShaderInterfaceDesc& declared);

    [[nodiscard]] Status validateShaderInterface(const MetalReflectedLayout& layout,
                                                 const ShaderInterfaceDesc& declared);

} // namespace rachis::metal
