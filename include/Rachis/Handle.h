#pragma once

#include <compare>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace rachis
{

    template<typename Tag>
    struct Handle final
    {
        std::uint64_t identity = 0;

        [[nodiscard]] constexpr bool isValid() const noexcept
        {
            return identity != 0;
        }

        [[nodiscard]] constexpr explicit operator bool() const noexcept
        {
            return isValid();
        }

        friend constexpr auto operator<=>(const Handle&, const Handle&) noexcept = default;
    };

    template<typename Tag>
    struct HandleHash final
    {
        [[nodiscard]] constexpr std::size_t operator()(Handle<Tag> handle) const noexcept
        {
            return std::hash<std::uint64_t>{}(handle.identity);
        }
    };

    struct BufferTag;
    struct ResourceHeapTag;
    struct CommandPoolTag;
    struct AccelerationStructureTag;
    struct TextureTag;
    struct TextureViewTag;
    struct SamplerFeedbackMapTag;
    struct SamplerTag;
    struct BindingLayoutTag;
    struct BindingPoolTag;
    struct BindingSetTag;
    struct ShaderLibraryTag;
    struct ShaderModuleTag;
    struct PipelineLayoutTag;
    struct PipelineCacheTag;
    struct PipelineCompilationTag;
    struct RayTracingPipelineLibraryTag;
    struct PipelineTag;
    struct QueryPoolTag;
    struct SurfaceTag;
    struct SwapchainTag;

    using BufferHandle = Handle<BufferTag>;
    using ResourceHeapHandle = Handle<ResourceHeapTag>;
    using CommandPoolHandle = Handle<CommandPoolTag>;
    using AccelerationStructureHandle = Handle<AccelerationStructureTag>;
    using TextureHandle = Handle<TextureTag>;
    using TextureViewHandle = Handle<TextureViewTag>;
    using SamplerFeedbackMapHandle = Handle<SamplerFeedbackMapTag>;
    using SamplerHandle = Handle<SamplerTag>;
    using BindingLayoutHandle = Handle<BindingLayoutTag>;
    using BindingPoolHandle = Handle<BindingPoolTag>;
    using BindingSetHandle = Handle<BindingSetTag>;
    using ShaderLibraryHandle = Handle<ShaderLibraryTag>;
    using ShaderModuleHandle = Handle<ShaderModuleTag>;
    using PipelineLayoutHandle = Handle<PipelineLayoutTag>;
    using PipelineCacheHandle = Handle<PipelineCacheTag>;
    using PipelineCompilationHandle = Handle<PipelineCompilationTag>;
    using RayTracingPipelineLibraryHandle = Handle<RayTracingPipelineLibraryTag>;
    using PipelineHandle = Handle<PipelineTag>;
    using QueryPoolHandle = Handle<QueryPoolTag>;
    using SurfaceHandle = Handle<SurfaceTag>;
    using SwapchainHandle = Handle<SwapchainTag>;

    static_assert(sizeof(BufferHandle) == 8);
    static_assert(std::is_trivially_copyable_v<BufferHandle>);

} // namespace rachis
