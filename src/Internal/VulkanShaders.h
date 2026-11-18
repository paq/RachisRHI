#pragma once

#include "Common/HandleTable.h"
#include "Internal/StageLinkage.h"
#include "Internal/VulkanDiagnostics.h"
#include "Internal/VulkanRuntime.h"
#include "Rachis/Device.h"
#include "Rachis/Pipeline.h"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <volk.h>

namespace rachis::vulkan
{

    struct VulkanShaderLayoutProbe final
    {
        std::uint32_t pushConstantBytes = 0;
        std::uint32_t readOnlyStorageArrayStride = 0;
        std::uint32_t storageArrayStride = 0;
    };

    class VulkanShaderManager final
    {
    public:
        VulkanShaderManager() = default;
        ~VulkanShaderManager();

        VulkanShaderManager(const VulkanShaderManager&) = delete;
        VulkanShaderManager& operator=(const VulkanShaderManager&) = delete;
        VulkanShaderManager(VulkanShaderManager&&) = delete;
        VulkanShaderManager& operator=(VulkanShaderManager&&) = delete;

        void initialize(VkDevice device,
                        const VolkDeviceTable* table,
                        const DeviceCapabilities& capabilities,
                        const VulkanInstanceTable* instanceTable = nullptr,
                        bool maintenance4 = false,
                        VulkanShaderFeatureSupport features = {},
                        bool retainSpirv = false) noexcept;
        void shutdown() noexcept;

        [[nodiscard]] Result<ShaderLibraryHandle> createShaderLibrary(const ShaderLibraryDesc& desc);
        [[nodiscard]] Status destroyShaderLibrary(ShaderLibraryHandle shaderLibrary);
        [[nodiscard]] Result<ShaderModuleHandle> createShaderModule(const ShaderModuleDesc& desc);
        [[nodiscard]] Result<ShaderInterfaceSnapshot> shaderModuleInterface(ShaderModuleHandle shaderModule) const;
        [[nodiscard]] Status destroyShaderModule(ShaderModuleHandle shaderModule);

        struct NativeShaderModule final
        {
            std::shared_ptr<void> owner;
            VkShaderModule module = VK_NULL_HANDLE;
            std::span<const std::uint32_t> spirv;
            ShaderStage stage = ShaderStage::none;
            std::string entryPoint;
            std::vector<ShaderBindingRequirement> bindings;
            std::vector<ShaderPushConstantRequirement> pushConstants;
            std::vector<std::uint32_t> vertexInputLocations;
            std::vector<SpecializationConstant> specializationConstants;
            std::optional<ShaderRecordConstantRange> shaderRecordConstants;
            bool writesPointSize = false;
            bool drawIndex = false;
            std::vector<detail::StageInterfaceVariable> stageOutputs;
            std::vector<detail::StageInterfaceVariable> stageInputs;
        };

        [[nodiscard]] Result<NativeShaderModule> nativeShaderModule(ShaderModuleHandle shaderModule) const;
#ifdef RACHIS_TESTING
        [[nodiscard]] Result<VulkanShaderLayoutProbe> queryLayout(ShaderModuleHandle shaderModule) const;
#endif

    private:
        struct ShaderLibraryEntryRecord final
        {
            ShaderStage stage = ShaderStage::none;
            std::vector<std::byte> bytecode;
            std::string entryPoint;
            std::vector<SpecializationConstant> specializationConstants;
        };

        struct ShaderLibraryRecord final
        {
            std::vector<ShaderLibraryEntryRecord> entries;
            std::string label;
        };

        struct ShaderInterfaceRecord final
        {
            std::vector<ShaderBindingRequirement> bindings;
            std::vector<ShaderPushConstantRequirement> pushConstants;
            std::vector<std::uint32_t> vertexInputLocations;
            std::optional<ShaderWorkgroupSize> workgroupSize;
            std::optional<ShaderRecordConstantRange> shaderRecordConstants;
            bool drawIndex = false;
        };

        struct ShaderKeepAlive final
        {
            VkDevice device = VK_NULL_HANDLE;
            const VolkDeviceTable* table = nullptr;
            VkShaderModule module = VK_NULL_HANDLE;
            std::vector<std::uint32_t> spirv;

            ShaderKeepAlive() = default;
            ~ShaderKeepAlive();
            ShaderKeepAlive(const ShaderKeepAlive&) = delete;
            ShaderKeepAlive& operator=(const ShaderKeepAlive&) = delete;
            ShaderKeepAlive(ShaderKeepAlive&&) = delete;
            ShaderKeepAlive& operator=(ShaderKeepAlive&&) = delete;
        };

        struct ShaderModuleRecord final
        {
            ShaderStage stage = ShaderStage::none;
            std::vector<std::byte> directBytecode;
            std::shared_ptr<ShaderLibraryRecord> library;
            std::string entryPoint;
            std::string label;
            ShaderInterfaceRecord shaderInterface;
            std::vector<SpecializationConstant> specializationConstants;
            std::vector<detail::StageInterfaceVariable> stageOutputs;
            std::vector<detail::StageInterfaceVariable> stageInputs;
            VulkanShaderLayoutProbe layout{};
            bool writesPointSize = false;
            std::shared_ptr<ShaderKeepAlive> keepAlive;
        };

        struct ShaderBytecodeSource final
        {
            std::shared_ptr<ShaderLibraryRecord> library;
            std::span<const std::byte> bytecode;
        };

        [[nodiscard]] Result<ShaderBytecodeSource> resolveShaderBytecode(
            const ShaderModuleDesc& desc, const std::vector<SpecializationConstant>& specializationConstants);

        VkDevice device_ = VK_NULL_HANDLE;
        const VolkDeviceTable* table_ = nullptr;
        const VulkanInstanceTable* instanceTable_ = nullptr;
        DeviceCapabilities capabilities_{};
        bool maintenance4_ = false;
        VulkanShaderFeatureSupport features_{};
        bool retainSpirv_ = false;
        mutable std::mutex mutex_{};
        detail::HandleTable<std::shared_ptr<ShaderLibraryRecord>, ShaderLibraryHandle> libraries_{};
        detail::HandleTable<ShaderModuleRecord, ShaderModuleHandle> modules_{};
    };

#ifdef RACHIS_TESTING
    [[nodiscard]] Result<VulkanShaderLayoutProbe> queryVulkanShaderLayout(ShaderModuleHandle shaderModule);

    struct VulkanStageInterfaceProbe final
    {
        std::vector<detail::StageInterfaceVariable> outputs;
        std::vector<detail::StageInterfaceVariable> inputs;
    };

    [[nodiscard]] Result<VulkanStageInterfaceProbe> reflectVulkanStageInterface(std::span<const std::byte> bytecode,
                                                                                std::string_view entryPoint,
                                                                                ShaderStage stage);
#endif

} // namespace rachis::vulkan
