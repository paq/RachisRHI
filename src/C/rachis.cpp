#include "Rachis/rachis.h"

#include "C/CAbiDeviceMutex.h"
#include "C/CAbiSlotTable.h"
#include "C/CAbiText.h"
#include "C/CAbiTokenSource.h"
#include "Common/MonotonicIdentitySource.h"
#include "Rachis/Device.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
    enum class TokenKind : std::uint8_t
    {
        buffer,
        texture,

        retiredTexture,
        textureLease,
        textureView,
        retiredView,
        commandPool,
        surface,
        borrowedTexture,
        borrowedView,
        bindingLayout,
        bindingPool,
        bindingSet,
        sampler,
        shaderModule,
        pipelineLayout,
        pipeline,
        timestampPool,
    };

    struct Token final
    {
        TokenKind kind;
        std::uint64_t native = 0;

        rachis_handle parent = 0;

        std::uint64_t dependents = 0;

        std::uint64_t byteSize = 0;
    };

    struct CommandLease final
    {
        mutable rachis::CommandBuffer command;
        mutable std::atomic<bool> inUse{false};

        CommandLease() noexcept = default;
        CommandLease(CommandLease&& other) noexcept : command(std::move(other.command)), inUse(false)
        {
        }
        CommandLease& operator=(CommandLease&& other) noexcept
        {
            if (this != &other)
            {
                command = std::move(other.command);
                inUse.store(false, std::memory_order_relaxed);
            }
            return *this;
        }
        CommandLease(const CommandLease&) = delete;
        CommandLease& operator=(const CommandLease&) = delete;
    };

    struct SwapchainEntry final
    {
        rachis::SwapchainHandle native;
        std::shared_ptr<std::mutex> gate;
        rachis::cabi::detail::CAbiBorrowedImageHandles borrowed;
    };

    rachis::detail::MonotonicIdentitySource nextToken;
#ifdef RACHIS_TESTING
    std::atomic<rachis::detail::MonotonicIdentitySource*> tokenSourceOverride{nullptr};

    struct CommandRecordingHookState final
    {
        rachis::cabi::detail::CommandRecordingHook hook = nullptr;
        void* userData = nullptr;
    };

    struct SwapchainPresentHookState final
    {
        rachis::cabi::detail::SwapchainPresentHook hook = nullptr;
        void* userData = nullptr;
    };

    struct NativeCallHookState final
    {
        rachis::cabi::detail::NativeCallHook hook = nullptr;
        void* userData = nullptr;
    };

    std::mutex testHookMutex;
    CommandRecordingHookState commandRecordingHookState;
    SwapchainPresentHookState swapchainPresentHookState;
    NativeCallHookState nativeCallHookState;

    void invokeCommandRecordingHook(rachis_handle command) noexcept
    {
        CommandRecordingHookState snapshot;
        {
            const std::scoped_lock lock(testHookMutex);
            snapshot = commandRecordingHookState;
        }
        if (snapshot.hook != nullptr)
        {
            snapshot.hook(command, snapshot.userData);
        }
    }

    void invokeSwapchainPresentHook(rachis_handle swapchain) noexcept
    {
        SwapchainPresentHookState snapshot;
        {
            const std::scoped_lock lock(testHookMutex);
            snapshot = swapchainPresentHookState;
        }
        if (snapshot.hook != nullptr)
        {
            snapshot.hook(swapchain, snapshot.userData);
        }
    }

    [[nodiscard]] rachis::Status invokeNativeCallHook(rachis::cabi::detail::NativeCall call)
    {
        NativeCallHookState snapshot;
        {
            const std::scoped_lock lock(testHookMutex);
            snapshot = nativeCallHookState;
        }
        if (snapshot.hook == nullptr)
        {
            return {};
        }
        return snapshot.hook(call, snapshot.userData);
    }
#endif

    template<typename Value>
    [[nodiscard]] constexpr std::uint32_t bitsOf(Value value) noexcept
    {
        return static_cast<std::uint32_t>(value);
    }

    static_assert(bitsOf(RACHIS_ADAPTER_TYPE_UNKNOWN) == bitsOf(rachis::AdapterType::unknown));
    static_assert(bitsOf(RACHIS_ADAPTER_TYPE_INTEGRATED) == bitsOf(rachis::AdapterType::integrated));
    static_assert(bitsOf(RACHIS_ADAPTER_TYPE_DISCRETE) == bitsOf(rachis::AdapterType::discrete));
    static_assert(bitsOf(RACHIS_ADAPTER_TYPE_SOFTWARE) == bitsOf(rachis::AdapterType::software));
    static_assert(bitsOf(RACHIS_STAGE_ALL) == bitsOf(rachis::PipelineStage::all));
    static_assert(bitsOf(RACHIS_STAGE_DRAW) == bitsOf(rachis::PipelineStage::draw));
    static_assert(bitsOf(RACHIS_STAGE_VERTEX_INPUT) == bitsOf(rachis::PipelineStage::vertexInput));
    static_assert(bitsOf(RACHIS_STAGE_VERTEX_SHADER) == bitsOf(rachis::PipelineStage::vertexShader));
    static_assert(bitsOf(RACHIS_STAGE_PIXEL_SHADER) == bitsOf(rachis::PipelineStage::pixelShader));
    static_assert(bitsOf(RACHIS_STAGE_COMPUTE_SHADER) == bitsOf(rachis::PipelineStage::computeShader));
    static_assert(bitsOf(RACHIS_STAGE_COPY) == bitsOf(rachis::PipelineStage::copy));
    static_assert(bitsOf(RACHIS_STAGE_RESOLVE) == bitsOf(rachis::PipelineStage::resolve));
    static_assert(bitsOf(RACHIS_STAGE_RENDER_TARGET) == bitsOf(rachis::PipelineStage::renderTarget));
    static_assert(bitsOf(RACHIS_STAGE_DEPTH_STENCIL) == bitsOf(rachis::PipelineStage::depthStencil));
    static_assert(bitsOf(RACHIS_STAGE_INDIRECT) == bitsOf(rachis::PipelineStage::indirect));
    static_assert(bitsOf(RACHIS_STAGE_TASK_SHADER) == bitsOf(rachis::PipelineStage::taskShader));
    static_assert(bitsOf(RACHIS_STAGE_MESH_SHADER) == bitsOf(rachis::PipelineStage::meshShader));
    static_assert(bitsOf(RACHIS_ACCESS_COMMON) == bitsOf(rachis::Access::common));
    static_assert(bitsOf(RACHIS_ACCESS_VERTEX_BUFFER) == bitsOf(rachis::Access::vertexBuffer));
    static_assert(bitsOf(RACHIS_ACCESS_CONSTANT_BUFFER) == bitsOf(rachis::Access::constantBuffer));
    static_assert(bitsOf(RACHIS_ACCESS_INDEX_BUFFER) == bitsOf(rachis::Access::indexBuffer));
    static_assert(bitsOf(RACHIS_ACCESS_SHADER_RESOURCE) == bitsOf(rachis::Access::shaderResource));
    static_assert(bitsOf(RACHIS_ACCESS_UNORDERED_ACCESS) == bitsOf(rachis::Access::unorderedAccess));
    static_assert(bitsOf(RACHIS_ACCESS_COPY_SOURCE) == bitsOf(rachis::Access::copySource));
    static_assert(bitsOf(RACHIS_ACCESS_COPY_DESTINATION) == bitsOf(rachis::Access::copyDestination));
    static_assert(bitsOf(RACHIS_ACCESS_INDIRECT_ARGUMENT) == bitsOf(rachis::Access::indirectArgument));
    static_assert(bitsOf(RACHIS_ACCESS_RENDER_TARGET) == bitsOf(rachis::Access::renderTarget));
    static_assert(bitsOf(RACHIS_ACCESS_DEPTH_STENCIL_READ) == bitsOf(rachis::Access::depthStencilRead));
    static_assert(bitsOf(RACHIS_ACCESS_DEPTH_STENCIL_WRITE) == bitsOf(rachis::Access::depthStencilWrite));
    static_assert(bitsOf(RACHIS_ACCESS_RESOLVE_SOURCE) == bitsOf(rachis::Access::resolveSource));
    static_assert(bitsOf(RACHIS_ACCESS_RESOLVE_DESTINATION) == bitsOf(rachis::Access::resolveDestination));
    static_assert(bitsOf(RACHIS_LAYOUT_UNDEFINED) == bitsOf(rachis::TextureLayout::undefined));
    static_assert(bitsOf(RACHIS_LAYOUT_COMMON) == bitsOf(rachis::TextureLayout::common));
    static_assert(bitsOf(RACHIS_LAYOUT_PRESENT) == bitsOf(rachis::TextureLayout::present));
    static_assert(bitsOf(RACHIS_LAYOUT_SHADER_RESOURCE) == bitsOf(rachis::TextureLayout::shaderResource));
    static_assert(bitsOf(RACHIS_LAYOUT_UNORDERED_ACCESS) == bitsOf(rachis::TextureLayout::unorderedAccess));
    static_assert(bitsOf(RACHIS_LAYOUT_RENDER_TARGET) == bitsOf(rachis::TextureLayout::renderTarget));
    static_assert(bitsOf(RACHIS_LAYOUT_DEPTH_STENCIL_WRITE) == bitsOf(rachis::TextureLayout::depthStencilWrite));
    static_assert(bitsOf(RACHIS_LAYOUT_DEPTH_STENCIL_READ) == bitsOf(rachis::TextureLayout::depthStencilRead));
    static_assert(bitsOf(RACHIS_LAYOUT_COPY_SOURCE) == bitsOf(rachis::TextureLayout::copySource));
    static_assert(bitsOf(RACHIS_LAYOUT_COPY_DESTINATION) == bitsOf(rachis::TextureLayout::copyDestination));
    static_assert(bitsOf(RACHIS_LAYOUT_RESOLVE_SOURCE) == bitsOf(rachis::TextureLayout::resolveSource));
    static_assert(bitsOf(RACHIS_LAYOUT_RESOLVE_DESTINATION) == bitsOf(rachis::TextureLayout::resolveDestination));
    static_assert(bitsOf(RACHIS_TEXTURE_COPY_SOURCE) == bitsOf(rachis::TextureUsage::copySource));
    static_assert(bitsOf(RACHIS_TEXTURE_COPY_DESTINATION) == bitsOf(rachis::TextureUsage::copyDestination));
    static_assert(bitsOf(RACHIS_TEXTURE_SAMPLED) == bitsOf(rachis::TextureUsage::sampled));
    static_assert(bitsOf(RACHIS_TEXTURE_STORAGE) == bitsOf(rachis::TextureUsage::storage));
    static_assert(bitsOf(RACHIS_TEXTURE_RENDER_TARGET) == bitsOf(rachis::TextureUsage::renderTarget));
    static_assert(bitsOf(RACHIS_TEXTURE_DEPTH_STENCIL) == bitsOf(rachis::TextureUsage::depthStencil));
    static_assert(bitsOf(RACHIS_TEXTURE_3D) == bitsOf(rachis::TextureDimension::texture3D));
    static_assert(bitsOf(RACHIS_BINDING_SAMPLED_TEXTURE) == bitsOf(rachis::BindingType::sampledTexture));
    static_assert(bitsOf(RACHIS_BINDING_STORAGE_TEXTURE) == bitsOf(rachis::BindingType::storageTexture));
    static_assert(bitsOf(RACHIS_BINDING_SAMPLER) == bitsOf(rachis::BindingType::sampler));
    static_assert(bitsOf(RACHIS_BINDING_FLAG_PARTIALLY_BOUND) == bitsOf(rachis::BindingFlag::partiallyBound));
    static_assert(bitsOf(RACHIS_BINDING_FLAG_VARIABLE_DESCRIPTOR_COUNT) ==
                  bitsOf(rachis::BindingFlag::variableDescriptorCount));
    static_assert(bitsOf(RACHIS_BINDING_FLAG_UPDATE_AFTER_BIND) == bitsOf(rachis::BindingFlag::updateAfterBind));
    static_assert(RACHIS_UNBOUNDED_BINDING_COUNT == rachis::unboundedBindingCount);
    static_assert(bitsOf(RACHIS_SHADER_VERTEX) == bitsOf(rachis::ShaderStage::vertex));
    static_assert(bitsOf(RACHIS_SHADER_PIXEL) == bitsOf(rachis::ShaderStage::pixel));
    static_assert(bitsOf(RACHIS_SHADER_COMPUTE) == bitsOf(rachis::ShaderStage::compute));
    static_assert(bitsOf(RACHIS_SHADER_TASK) == bitsOf(rachis::ShaderStage::task));
    static_assert(bitsOf(RACHIS_SHADER_MESH) == bitsOf(rachis::ShaderStage::mesh));
    static_assert(bitsOf(RACHIS_SHADER_ALL) ==
                  bitsOf(rachis::ShaderStage::vertex | rachis::ShaderStage::pixel | rachis::ShaderStage::compute |
                         rachis::ShaderStage::task | rachis::ShaderStage::mesh));
    static_assert(bitsOf(RACHIS_STORAGE_TEXTURE_CLEAR_FLOAT32) == bitsOf(rachis::StorageTextureClearType::float32));
    static_assert(bitsOf(RACHIS_STORAGE_TEXTURE_CLEAR_UINT32) == bitsOf(rachis::StorageTextureClearType::uint32));
    static_assert(bitsOf(RACHIS_STORAGE_TEXTURE_CLEAR_SINT32) == bitsOf(rachis::StorageTextureClearType::sint32));
    static_assert(sizeof(rachis_dispatch_indirect_arguments) == sizeof(rachis::DispatchIndirectArguments));
    static_assert(sizeof(rachis_draw_indirect_arguments) == sizeof(rachis::DrawIndirectArguments));
    static_assert(sizeof(rachis_draw_indexed_indirect_arguments) == sizeof(rachis::DrawIndexedIndirectArguments));
    static_assert(bitsOf(RACHIS_BLEND_ONE_MINUS_SOURCE1_ALPHA) == bitsOf(rachis::BlendFactor::oneMinusSource1Alpha));
    static_assert(bitsOf(RACHIS_COMPARE_LESS) == bitsOf(rachis::CompareOperation::less));
    static_assert(bitsOf(RACHIS_COMPARE_ALWAYS) == bitsOf(rachis::CompareOperation::always));
    static_assert(bitsOf(RACHIS_BLEND_OP_MAXIMUM) == bitsOf(rachis::BlendOperation::maximum));
    static_assert(bitsOf(RACHIS_CULL_BACK) == bitsOf(rachis::CullMode::back));
    static_assert(bitsOf(RACHIS_TOPOLOGY_TRIANGLE_STRIP) == bitsOf(rachis::PrimitiveTopology::triangleStrip));
    static_assert(bitsOf(RACHIS_VIEW_USAGE_DEPTH_STENCIL) == bitsOf(rachis::TextureViewUsage::depthStencil));
    static_assert(bitsOf(RACHIS_VIEW_CUBE_ARRAY) == bitsOf(rachis::TextureViewDimension::cubeArray));
    static_assert(bitsOf(RACHIS_ASPECT_STENCIL) == bitsOf(rachis::TextureAspect::stencil));
    static_assert(bitsOf(RACHIS_LOAD_OP_DISCARD) == bitsOf(rachis::LoadOperation::discard));
    static_assert(bitsOf(RACHIS_STORE_OP_DISCARD) == bitsOf(rachis::StoreOperation::discard));
    static_assert(bitsOf(RACHIS_COLOR_SPACE_HDR10_ST2084) == bitsOf(rachis::SurfaceColorSpace::hdr10St2084));
    static_assert(bitsOf(RACHIS_PRESENT_FIFO) == bitsOf(rachis::PresentMode::fifo));
    static_assert(bitsOf(RACHIS_PRESENT_IMMEDIATE) == bitsOf(rachis::PresentMode::immediate));
    static_assert(bitsOf(RACHIS_PRESENT_MODE_FIFO_BIT) == bitsOf(rachis::PresentModeMask::fifo));
    static_assert(bitsOf(RACHIS_PRESENT_MODE_IMMEDIATE_BIT) == bitsOf(rachis::PresentModeMask::immediate));
    static_assert(bitsOf(RACHIS_PRESENT_MODE_FIFO_BIT) == 1U << bitsOf(RACHIS_PRESENT_FIFO));
    static_assert(bitsOf(RACHIS_PRESENT_MODE_IMMEDIATE_BIT) == 1U << bitsOf(RACHIS_PRESENT_IMMEDIATE));
    static_assert(RACHIS_REMAINING_SUBRESOURCES == rachis::remainingSubresources);
    static_assert(RACHIS_WHOLE_SIZE == rachis::wholeResource);
    static_assert(bitsOf(RACHIS_INDEX_FORMAT_UINT16) == bitsOf(rachis::IndexFormat::uint16));
    static_assert(bitsOf(RACHIS_INDEX_FORMAT_UINT32) == bitsOf(rachis::IndexFormat::uint32));
    static_assert(bitsOf(RACHIS_STRIP_INDEX_UINT16) == bitsOf(rachis::IndexFormat::uint16) + 1U);
    static_assert(bitsOf(RACHIS_STRIP_INDEX_UINT32) == bitsOf(rachis::IndexFormat::uint32) + 1U);
    static_assert(bitsOf(RACHIS_VERTEX_STEP_VERTEX) == bitsOf(rachis::VertexStepMode::vertex));
    static_assert(bitsOf(RACHIS_VERTEX_STEP_INSTANCE) == bitsOf(rachis::VertexStepMode::instance));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_FLOAT32) == bitsOf(rachis::VertexFormat::float32));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_FLOAT32X2) == bitsOf(rachis::VertexFormat::float32x2));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_FLOAT32X3) == bitsOf(rachis::VertexFormat::float32x3));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_FLOAT32X4) == bitsOf(rachis::VertexFormat::float32x4));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_UINT32) == bitsOf(rachis::VertexFormat::uint32));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_UINT32X2) == bitsOf(rachis::VertexFormat::uint32x2));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_UINT32X3) == bitsOf(rachis::VertexFormat::uint32x3));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_UINT32X4) == bitsOf(rachis::VertexFormat::uint32x4));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_SINT32) == bitsOf(rachis::VertexFormat::sint32));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_SINT32X2) == bitsOf(rachis::VertexFormat::sint32x2));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_SINT32X3) == bitsOf(rachis::VertexFormat::sint32x3));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_SINT32X4) == bitsOf(rachis::VertexFormat::sint32x4));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_UINT8X2) == bitsOf(rachis::VertexFormat::uint8x2));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_UINT8X4) == bitsOf(rachis::VertexFormat::uint8x4));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_SINT8X2) == bitsOf(rachis::VertexFormat::sint8x2));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_SINT8X4) == bitsOf(rachis::VertexFormat::sint8x4));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_UNORM8X2) == bitsOf(rachis::VertexFormat::unorm8x2));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_UNORM8X4) == bitsOf(rachis::VertexFormat::unorm8x4));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_SNORM8X2) == bitsOf(rachis::VertexFormat::snorm8x2));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_SNORM8X4) == bitsOf(rachis::VertexFormat::snorm8x4));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_UINT16X2) == bitsOf(rachis::VertexFormat::uint16x2));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_UINT16X4) == bitsOf(rachis::VertexFormat::uint16x4));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_SINT16X2) == bitsOf(rachis::VertexFormat::sint16x2));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_SINT16X4) == bitsOf(rachis::VertexFormat::sint16x4));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_UNORM16X2) == bitsOf(rachis::VertexFormat::unorm16x2));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_UNORM16X4) == bitsOf(rachis::VertexFormat::unorm16x4));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_SNORM16X2) == bitsOf(rachis::VertexFormat::snorm16x2));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_SNORM16X4) == bitsOf(rachis::VertexFormat::snorm16x4));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_FLOAT16X2) == bitsOf(rachis::VertexFormat::float16x2));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_FLOAT16X4) == bitsOf(rachis::VertexFormat::float16x4));
    static_assert(bitsOf(RACHIS_VERTEX_FORMAT_FLOAT16X4) + 1U == rachis::vertexFormatCount);
    static_assert(RACHIS_MAX_VERTEX_BUFFERS == rachis::maxVertexBuffers);
    static_assert(RACHIS_MAX_VERTEX_ATTRIBUTES == rachis::maxVertexAttributes);

    constexpr std::uint32_t knownStages = RACHIS_STAGE_ALL | RACHIS_STAGE_DRAW | RACHIS_STAGE_VERTEX_INPUT |
                                          RACHIS_STAGE_VERTEX_SHADER | RACHIS_STAGE_PIXEL_SHADER |
                                          RACHIS_STAGE_COMPUTE_SHADER | RACHIS_STAGE_COPY | RACHIS_STAGE_RESOLVE |
                                          RACHIS_STAGE_RENDER_TARGET | RACHIS_STAGE_DEPTH_STENCIL |
                                          RACHIS_STAGE_INDIRECT | RACHIS_STAGE_TASK_SHADER | RACHIS_STAGE_MESH_SHADER;
    constexpr std::uint32_t knownAccess =
        RACHIS_ACCESS_COMMON | RACHIS_ACCESS_VERTEX_BUFFER | RACHIS_ACCESS_CONSTANT_BUFFER |
        RACHIS_ACCESS_INDEX_BUFFER | RACHIS_ACCESS_SHADER_RESOURCE | RACHIS_ACCESS_UNORDERED_ACCESS |
        RACHIS_ACCESS_COPY_SOURCE | RACHIS_ACCESS_COPY_DESTINATION | RACHIS_ACCESS_INDIRECT_ARGUMENT |
        RACHIS_ACCESS_RENDER_TARGET | RACHIS_ACCESS_DEPTH_STENCIL_READ | RACHIS_ACCESS_DEPTH_STENCIL_WRITE |
        RACHIS_ACCESS_RESOLVE_SOURCE | RACHIS_ACCESS_RESOLVE_DESTINATION;
    constexpr std::uint32_t knownTextureUsage = RACHIS_TEXTURE_COPY_SOURCE | RACHIS_TEXTURE_COPY_DESTINATION |
                                                RACHIS_TEXTURE_SAMPLED | RACHIS_TEXTURE_STORAGE |
                                                RACHIS_TEXTURE_RENDER_TARGET | RACHIS_TEXTURE_DEPTH_STENCIL;
    constexpr std::uint32_t knownBindingFlags = RACHIS_BINDING_FLAG_PARTIALLY_BOUND |
                                                RACHIS_BINDING_FLAG_VARIABLE_DESCRIPTOR_COUNT |
                                                RACHIS_BINDING_FLAG_UPDATE_AFTER_BIND;

    [[nodiscard]] rachis::detail::MonotonicIdentitySource& activeTokenSource() noexcept
    {
#ifdef RACHIS_TESTING
        if (auto* const overrideSource = tokenSourceOverride.load(std::memory_order_acquire); overrideSource != nullptr)
        {
            return *overrideSource;
        }
#endif
        return nextToken;
    }

    [[nodiscard]] std::string_view optionalString(const char* value) noexcept
    {
        return value == nullptr ? std::string_view{} : std::string_view{value};
    }

    [[nodiscard]] rachis_status statusOf(rachis::ErrorCode code) noexcept
    {
        switch (code)
        {
        case rachis::ErrorCode::invalidArgument:
            return RACHIS_STATUS_INVALID_ARGUMENT;
        case rachis::ErrorCode::unsupported:
            return RACHIS_STATUS_UNSUPPORTED;
        case rachis::ErrorCode::backendUnavailable:
            return RACHIS_STATUS_BACKEND_UNAVAILABLE;
        case rachis::ErrorCode::adapterNotFound:
            return RACHIS_STATUS_ADAPTER_NOT_FOUND;
        case rachis::ErrorCode::deviceCreationFailed:
            return RACHIS_STATUS_DEVICE_CREATION_FAILED;
        case rachis::ErrorCode::resourceExhausted:
            return RACHIS_STATUS_RESOURCE_EXHAUSTED;
        case rachis::ErrorCode::invalidState:
            return RACHIS_STATUS_INVALID_STATE;
        case rachis::ErrorCode::deviceLost:
            return RACHIS_STATUS_DEVICE_LOST;
        case rachis::ErrorCode::outOfMemory:
            return RACHIS_STATUS_OUT_OF_MEMORY;
        case rachis::ErrorCode::nativeFailure:
            return RACHIS_STATUS_NATIVE_FAILURE;
        case rachis::ErrorCode::validationError:
            return RACHIS_STATUS_VALIDATION_ERROR;
        case rachis::ErrorCode::pipelineCacheMiss:
            return RACHIS_STATUS_PIPELINE_CACHE_MISS;
        case rachis::ErrorCode::notReady:
            return RACHIS_STATUS_NOT_READY;
        }
        return RACHIS_STATUS_NATIVE_FAILURE;
    }

    [[nodiscard]] std::uint32_t diagnosticSeverityOf(rachis::DiagnosticSeverity severity) noexcept
    {
        switch (severity)
        {
        case rachis::DiagnosticSeverity::information:
            return RACHIS_DIAGNOSTIC_INFORMATION;
        case rachis::DiagnosticSeverity::warning:
            return RACHIS_DIAGNOSTIC_WARNING;
        case rachis::DiagnosticSeverity::error:
            return RACHIS_DIAGNOSTIC_ERROR;
        case rachis::DiagnosticSeverity::corruption:
            return RACHIS_DIAGNOSTIC_CORRUPTION;
        }
        return RACHIS_DIAGNOSTIC_ERROR;
    }

    [[nodiscard]] std::uint32_t presentStatusOf(rachis::PresentStatus status) noexcept
    {
        switch (status)
        {
        case rachis::PresentStatus::presented:
            return RACHIS_PRESENT_STATUS_PRESENTED;
        case rachis::PresentStatus::occluded:
            return RACHIS_PRESENT_STATUS_OCCLUDED;
        case rachis::PresentStatus::surfaceOutOfDate:
            return RACHIS_PRESENT_STATUS_SURFACE_OUT_OF_DATE;
        }
        return RACHIS_PRESENT_STATUS_SURFACE_OUT_OF_DATE;
    }

    [[nodiscard]] std::optional<rachis::Format> nativeFormat(std::uint32_t format) noexcept
    {
        switch (format)
        {
        case RACHIS_FORMAT_R8_UNORM:
            return rachis::Format::r8Unorm;
        case RACHIS_FORMAT_RG8_UNORM:
            return rachis::Format::rg8Unorm;
        case RACHIS_FORMAT_RGBA8_UNORM:
            return rachis::Format::rgba8Unorm;
        case RACHIS_FORMAT_RGBA8_UNORM_SRGB:
            return rachis::Format::rgba8UnormSrgb;
        case RACHIS_FORMAT_BGRA8_UNORM:
            return rachis::Format::bgra8Unorm;
        case RACHIS_FORMAT_BGRA8_UNORM_SRGB:
            return rachis::Format::bgra8UnormSrgb;
        case RACHIS_FORMAT_R16_FLOAT:
            return rachis::Format::r16Float;
        case RACHIS_FORMAT_RG16_FLOAT:
            return rachis::Format::rg16Float;
        case RACHIS_FORMAT_RGBA16_FLOAT:
            return rachis::Format::rgba16Float;
        case RACHIS_FORMAT_R32_FLOAT:
            return rachis::Format::r32Float;
        case RACHIS_FORMAT_RG32_FLOAT:
            return rachis::Format::rg32Float;
        case RACHIS_FORMAT_RGBA32_FLOAT:
            return rachis::Format::rgba32Float;
        case RACHIS_FORMAT_D32_FLOAT:
            return rachis::Format::d32Float;
        case RACHIS_FORMAT_D24_UNORM_S8_UINT:
            return rachis::Format::d24UnormS8Uint;
        case RACHIS_FORMAT_R32_UINT:
            return rachis::Format::r32Uint;
        default:
            return std::nullopt;
        }
    }

    [[nodiscard]] std::uint32_t formatOf(rachis::Format format) noexcept
    {
        switch (format)
        {
        case rachis::Format::r8Unorm:
            return RACHIS_FORMAT_R8_UNORM;
        case rachis::Format::rg8Unorm:
            return RACHIS_FORMAT_RG8_UNORM;
        case rachis::Format::rgba8Unorm:
            return RACHIS_FORMAT_RGBA8_UNORM;
        case rachis::Format::rgba8UnormSrgb:
            return RACHIS_FORMAT_RGBA8_UNORM_SRGB;
        case rachis::Format::bgra8Unorm:
            return RACHIS_FORMAT_BGRA8_UNORM;
        case rachis::Format::bgra8UnormSrgb:
            return RACHIS_FORMAT_BGRA8_UNORM_SRGB;
        case rachis::Format::r16Float:
            return RACHIS_FORMAT_R16_FLOAT;
        case rachis::Format::rg16Float:
            return RACHIS_FORMAT_RG16_FLOAT;
        case rachis::Format::rgba16Float:
            return RACHIS_FORMAT_RGBA16_FLOAT;
        case rachis::Format::r32Float:
            return RACHIS_FORMAT_R32_FLOAT;
        case rachis::Format::rg32Float:
            return RACHIS_FORMAT_RG32_FLOAT;
        case rachis::Format::rgba32Float:
            return RACHIS_FORMAT_RGBA32_FLOAT;
        case rachis::Format::d32Float:
            return RACHIS_FORMAT_D32_FLOAT;
        case rachis::Format::d24UnormS8Uint:
            return RACHIS_FORMAT_D24_UNORM_S8_UINT;
        case rachis::Format::r32Uint:
            return RACHIS_FORMAT_R32_UINT;
        default:
            return RACHIS_FORMAT_UNKNOWN;
        }
    }

    [[nodiscard]] bool validStages(std::uint32_t stages) noexcept
    {
        return (stages & ~knownStages) == 0;
    }

    [[nodiscard]] bool validAccess(std::uint32_t access) noexcept
    {
        return (access & ~knownAccess) == 0;
    }

    [[nodiscard]] bool validLayout(std::uint32_t layout) noexcept
    {
        return layout <= RACHIS_LAYOUT_RESOLVE_DESTINATION;
    }

    [[nodiscard]] bool validAspect(std::uint32_t aspect) noexcept
    {
        return aspect <= RACHIS_ASPECT_STENCIL;
    }

    [[nodiscard]] rachis::TextureSubresource subresourceOf(const rachis_buffer_texture_region& region) noexcept
    {
        return rachis::TextureSubresource{.mipLevel = region.mip_level,
                                          .arrayLayer = region.array_layer,
                                          .aspect = static_cast<rachis::TextureAspect>(region.aspect)};
    }

    [[nodiscard]] rachis::BufferTextureCopyRegion regionOf(const rachis_buffer_texture_region& region) noexcept
    {
        return rachis::BufferTextureCopyRegion{
            .bufferOffset = region.buffer_offset,
            .bytesPerRow = region.bytes_per_row,
            .rowsPerImage = region.rows_per_image,
            .textureSubresource = subresourceOf(region),
            .textureOrigin = {.x = region.x, .y = region.y, .z = region.z},
            .extent = {.width = region.width, .height = region.height, .depth = region.depth},
        };
    }

    [[nodiscard]] rachis_status fail(rachis_error* error,
                                     rachis_status code,
                                     std::string_view message,
                                     std::int64_t nativeCode = 0) noexcept
    {
        if (error != nullptr)
        {
            error->code = code;
            error->native_code = nativeCode;
            rachis::cabi::detail::copyText(error->message, message);
        }
        return code;
    }

    [[nodiscard]] rachis_status result(rachis_error* error, const rachis::Error& value) noexcept
    {
        return fail(error, statusOf(value.code), value.message, value.nativeCode);
    }

    [[nodiscard]] rachis_status result(rachis_error* error, const rachis::Status& value) noexcept
    {
        if (value)
        {
            return RACHIS_STATUS_OK;
        }
        return result(error, value.error());
    }

    template<typename Action>
    [[nodiscard]] rachis_status boundary(rachis_error* error, Action&& action) noexcept
    {
        if (error != nullptr)
        {
            if (error->struct_size != sizeof(rachis_error))
            {
                return RACHIS_STATUS_ABI_MISMATCH;
            }
            error->code = RACHIS_STATUS_OK;
            error->native_code = 0;
            error->message[0] = '\0';
        }
        try
        {
            return std::forward<Action>(action)();
        }
        catch (const std::bad_alloc&)
        {
            return fail(error, RACHIS_STATUS_OUT_OF_MEMORY, "C ABI allocation failed");
        }
        catch (const std::exception& exception)
        {
            return fail(error, RACHIS_STATUS_NATIVE_FAILURE, exception.what());
        }
        catch (...)
        {
            return fail(error, RACHIS_STATUS_NATIVE_FAILURE, "unknown C++ exception");
        }
    }

    template<typename Desc>
    [[nodiscard]] bool validDesc(const Desc* desc) noexcept
    {
        return desc != nullptr && desc->struct_size == sizeof(Desc) && desc->abi_version == RACHIS_ABI_VERSION;
    }

    [[nodiscard]] bool validHostByteSpan(const void* bytes, std::uint64_t size) noexcept
    {
        if (bytes == nullptr || size == 0 || size == RACHIS_WHOLE_SIZE ||
            size > std::numeric_limits<std::size_t>::max() || size > std::numeric_limits<std::uintptr_t>::max())
        {
            return false;
        }
        const auto address = reinterpret_cast<std::uintptr_t>(bytes);
        return address <= std::numeric_limits<std::uintptr_t>::max() - static_cast<std::uintptr_t>(size);
    }

    [[nodiscard]] bool validHostBufferRange(std::uint64_t bufferSize, std::uint64_t offset, std::uint64_t size) noexcept
    {
        return offset < bufferSize && size != 0 && size <= bufferSize - offset;
    }

    [[nodiscard]] bool validMappedBufferRange(std::uint64_t bufferSize,
                                              std::uint64_t offset,
                                              std::uint64_t size) noexcept
    {
        return size == RACHIS_WHOLE_SIZE ? offset < bufferSize : validHostBufferRange(bufferSize, offset, size);
    }

    template<typename Value>
    [[nodiscard]] bool validArray(const Value* values, std::uint32_t count) noexcept
    {
        return count == 0 || values != nullptr;
    }

    using TokenTable = rachis::cabi::detail::CAbiSlotTable<Token>;

    [[nodiscard]] rachis_handle reserveToken(TokenTable& tokens, TokenKind kind, rachis_handle parent = 0)
    {
        return tokens.emplace(activeTokenSource(), Token{kind, 0, parent});
    }

    [[nodiscard]] const Token* findToken(const TokenTable& tokens, rachis_handle handle, TokenKind kind) noexcept
    {
        const auto* const entry = tokens.find(handle);
        return entry != nullptr && entry->kind == kind && entry->native != 0 ? entry : nullptr;
    }

    [[nodiscard]] Token* findMutableToken(TokenTable& tokens, rachis_handle handle, TokenKind kind) noexcept
    {
        auto* const entry = tokens.find(handle);
        return entry != nullptr && entry->kind == kind && entry->native != 0 ? entry : nullptr;
    }

    [[nodiscard]] const Token* findTexture(const TokenTable& tokens, rachis_handle handle) noexcept
    {
        const auto* const entry = tokens.find(handle);
        if (entry == nullptr || entry->native == 0)
        {
            return nullptr;
        }
        switch (entry->kind)
        {
        case TokenKind::texture:
        case TokenKind::textureLease:
        case TokenKind::borrowedTexture:
            return entry;
        default:
            return nullptr;
        }
    }

    [[nodiscard]] const Token* findView(const TokenTable& tokens, rachis_handle handle) noexcept
    {
        const auto* const entry = tokens.find(handle);
        if (entry == nullptr || entry->native == 0)
        {
            return nullptr;
        }
        return entry->kind == TokenKind::textureView || entry->kind == TokenKind::borrowedView ? entry : nullptr;
    }

    [[nodiscard]] rachis_handle textureOwner(const TokenTable& tokens, rachis_handle handle) noexcept
    {
        const auto* const entry = tokens.find(handle);
        if (entry == nullptr || entry->native == 0)
        {
            return 0;
        }
        if (entry->kind == TokenKind::texture)
        {
            return handle;
        }
        return entry->kind == TokenKind::textureLease ? entry->parent : 0;
    }

    template<typename Create>
    [[nodiscard]] rachis_status createToken(
        rachis_device& device, TokenKind kind, rachis_handle* output, rachis_error* error, Create&& create);
    template<typename Destroy>
    [[nodiscard]] rachis_status destroyToken(
        rachis_device& device, TokenKind kind, rachis_handle handle, rachis_error* error, Destroy&& destroy);

    void onDiagnostic(const rachis::DiagnosticMessage& message, void* userData) noexcept;

} // namespace

struct rachis_device final
{
    rachis_diagnostic_callback callback = nullptr;
    void* callbackUserData = nullptr;
    rachis::Device native;

    rachis::cabi::detail::CAbiDeviceMutex mutex;
    rachis::cabi::detail::CAbiSlotTable<Token> tokens;
    rachis::cabi::detail::CAbiSlotTable<CommandLease> commands;
    rachis::cabi::detail::CAbiSlotTable<rachis::TimelinePoint> timelines;

    rachis::cabi::detail::CAbiSlotTable<SwapchainEntry> swapchains;
};

namespace
{
    template<typename Create>
    rachis_status createToken(
        rachis_device& device, TokenKind kind, rachis_handle* output, rachis_error* error, Create&& create)
    {
        const auto token = reserveToken(device.tokens, kind);
        if (token == 0)
        {
            return fail(error, RACHIS_STATUS_RESOURCE_EXHAUSTED, "C handle space exhausted");
        }
        try
        {
#ifdef RACHIS_TESTING
            if (auto injected = invokeNativeCallHook(rachis::cabi::detail::NativeCall::createObject); !injected)
            {
                device.tokens.erase(token);
                return result(error, injected.error());
            }
#endif
            auto created = std::forward<Create>(create)();
            if (!created)
            {
                device.tokens.erase(token);
                return result(error, created.error());
            }
            device.tokens.find(token)->native = created->identity;
            *output = token;
            return RACHIS_STATUS_OK;
        }
        catch (...)
        {
            device.tokens.erase(token);
            throw;
        }
    }

    template<typename Destroy>
    rachis_status destroyToken(
        rachis_device& device, TokenKind kind, rachis_handle handle, rachis_error* error, Destroy&& destroy)
    {
        const auto* token = findToken(device.tokens, handle, kind);
        if (token == nullptr)
        {
            return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "handle type, owner, or lifetime mismatch");
        }
        const auto destroyed = std::forward<Destroy>(destroy)(token->native);
        if (!destroyed)
        {
            return result(error, destroyed.error());
        }
        device.tokens.erase(handle);
        return RACHIS_STATUS_OK;
    }

    void onDiagnostic(const rachis::DiagnosticMessage& message, void* userData) noexcept
    {
        auto* device = static_cast<rachis_device*>(userData);
        if (device == nullptr || device->callback == nullptr)
        {
            return;
        }
        std::array<char, RACHIS_ERROR_MESSAGE_CAPACITY> buffer{};
        rachis::cabi::detail::copyText(buffer, message.description);
        const char* text = buffer.data();
        std::string whole;
        if (message.description.size() >= buffer.size())
        {
            try
            {
                whole.assign(message.description);
                text = whole.c_str();
            }
            catch (...)
            {
            }
        }
        try
        {
            device->callback(diagnosticSeverityOf(message.severity), message.nativeId, text, device->callbackUserData);
        }

        catch (...)
        {
        }
    }

    [[nodiscard]] rachis_status invalidHandle(rachis_error* error) noexcept
    {
        return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "handle type, owner, or lifetime mismatch");
    }

    [[nodiscard]] const SwapchainEntry* findSwapchain(const rachis_device& device, rachis_handle swapchain) noexcept
    {
        const auto* const entry = device.swapchains.find(swapchain);
        return entry != nullptr && entry->native.isValid() ? entry : nullptr;
    }

    [[nodiscard]] SwapchainEntry* findMutableSwapchain(rachis_device& device, rachis_handle swapchain) noexcept
    {
        auto* const entry = device.swapchains.find(swapchain);
        return entry != nullptr && entry->native.isValid() ? entry : nullptr;
    }

    struct SwapchainGateLock final
    {
        std::shared_ptr<std::mutex> gate;
        std::unique_lock<std::mutex> lock;
    };

    [[nodiscard]] SwapchainGateLock lockSwapchainGate(rachis_device& device, rachis_handle swapchain)
    {
        SwapchainGateLock held;
        {
            std::shared_lock const lock(device.mutex);
            const auto* const entry = findSwapchain(device, swapchain);
            if (entry == nullptr)
            {
                return held;
            }
            held.gate = entry->gate;
        }

        held.lock = std::unique_lock{*held.gate};
        return held;
    }

    [[nodiscard]] std::optional<rachis_status> rejectLostDevice(const rachis_device& device,
                                                                rachis_error* error) noexcept
    {
        if (device.native.operationalState() != rachis::DeviceOperationalState::lost)
        {
            return std::nullopt;
        }
        try
        {
            if (const auto loss = device.native.deviceLossError())
            {
                return fail(error, RACHIS_STATUS_DEVICE_LOST, loss->message, loss->nativeCode);
            }
        }

        catch (...)
        {
        }
        return fail(error, RACHIS_STATUS_DEVICE_LOST, "the device is lost");
    }

    [[nodiscard]] rachis_status releaseTextureDependent(rachis_device& device,
                                                        rachis_handle dependent,
                                                        rachis_error* error)
    {
        const auto owner = device.tokens.find(dependent)->parent;
        auto& ownerToken = *device.tokens.find(owner);
        if (ownerToken.kind == TokenKind::retiredTexture && ownerToken.dependents == 1)
        {
#ifdef RACHIS_TESTING
            if (auto injected = invokeNativeCallHook(rachis::cabi::detail::NativeCall::destroyRetiredTexture);
                !injected)
            {
                return result(error, injected.error());
            }
#endif
            const auto destroyed = device.native.destroyTexture(rachis::TextureHandle{ownerToken.native});
            if (!destroyed)
            {
                return result(error, destroyed.error());
            }
            device.tokens.erase(owner);
        }
        else
        {
            --ownerToken.dependents;
        }
        device.tokens.erase(dependent);
        return RACHIS_STATUS_OK;
    }

    class CommandUseGuard final
    {
    public:
        CommandUseGuard() noexcept = default;
        explicit CommandUseGuard(std::atomic<bool>* flag) noexcept : flag_(flag)
        {
        }

        CommandUseGuard(CommandUseGuard&& other) noexcept : flag_(std::exchange(other.flag_, nullptr))
        {
        }
        CommandUseGuard& operator=(CommandUseGuard&& other) noexcept
        {
            if (this != &other)
            {
                release();
                flag_ = std::exchange(other.flag_, nullptr);
            }
            return *this;
        }
        CommandUseGuard(const CommandUseGuard&) = delete;
        CommandUseGuard& operator=(const CommandUseGuard&) = delete;

        ~CommandUseGuard()
        {
            release();
        }

        void release() noexcept
        {
            if (flag_ != nullptr)
            {
                flag_->store(false, std::memory_order_release);
                flag_ = nullptr;
            }
        }

    private:
        std::atomic<bool>* flag_ = nullptr;
    };

    [[nodiscard]] rachis_status acquireCommandForRecording(const rachis_device& device,
                                                           rachis_handle command,
                                                           CommandUseGuard& guard,
                                                           rachis::CommandBuffer*& outCommand,
                                                           rachis_error* error) noexcept
    {
        const auto* const entry = device.commands.find(command);
        if (entry == nullptr)
        {
            return invalidHandle(error);
        }
        if (entry->inUse.exchange(true, std::memory_order_acquire))
        {
            return fail(error, RACHIS_STATUS_INVALID_STATE, "command buffer is concurrently in use");
        }
        CommandUseGuard localGuard(&entry->inUse);
        if (!entry->command.isValid())
        {
            return invalidHandle(error);
        }
#ifdef RACHIS_TESTING
        invokeCommandRecordingHook(command);
#endif
        outCommand = &entry->command;
        guard = std::move(localGuard);
        return RACHIS_STATUS_OK;
    }

    [[nodiscard]] rachis_status copyBufferTexture(rachis_device* device,
                                                  rachis_handle command,
                                                  const rachis_buffer_texture_copy_desc* desc,
                                                  rachis_error* error,
                                                  bool toTexture)
    {
        if (device == nullptr)
        {
            return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
        }
        if (desc == nullptr)
        {
            return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
        }
        if (!validDesc(desc))
        {
            return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported buffer-texture copy descriptor");
        }
        if (!validArray(desc->regions, desc->region_count))
        {
            return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "buffer-texture copy regions are null");
        }
        std::vector<rachis::BufferTextureCopyRegion> regions;
        regions.reserve(desc->region_count);
        for (std::uint32_t i = 0; i < desc->region_count; ++i)
        {
            if (!validAspect(desc->regions[i].aspect))
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid texture aspect");
            }
            regions.push_back(regionOf(desc->regions[i]));
        }
        std::shared_lock const lock(device->mutex);
        if (const auto lost = rejectLostDevice(*device, error))
        {
            return *lost;
        }
        CommandUseGuard commandUse;
        rachis::CommandBuffer* commandBuffer = nullptr;
        if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
            status != RACHIS_STATUS_OK)
        {
            return status;
        }
        const auto* buffer = findToken(device->tokens, desc->buffer, TokenKind::buffer);
        const auto* texture = findTexture(device->tokens, desc->texture);
        if (buffer == nullptr || texture == nullptr)
        {
            return invalidHandle(error);
        }
        const auto nativeBuffer = rachis::BufferHandle{buffer->native};
        const auto nativeTexture = rachis::TextureHandle{texture->native};
        return result(error,
                      toTexture
                          ? device->native.copyBufferToTexture(*commandBuffer, nativeBuffer, nativeTexture, regions)
                          : device->native.copyTextureToBuffer(*commandBuffer, nativeTexture, nativeBuffer, regions));
    }

    [[nodiscard]] rachis_status synchronizeMappedRanges(rachis_device* device,
                                                        rachis_handle buffer,
                                                        const rachis_mapped_range* ranges,
                                                        std::uint32_t rangeCount,
                                                        rachis_error* error,
                                                        bool flush)
    {
        if (device == nullptr || ranges == nullptr || rangeCount == 0)
        {
            return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null or no mapped ranges were given");
        }
        std::vector<rachis::MappedMemoryRange> nativeRanges;
        nativeRanges.reserve(rangeCount);
        for (std::uint32_t i = 0; i < rangeCount; ++i)
        {
            nativeRanges.push_back({.offset = ranges[i].offset, .size = ranges[i].size});
        }
        std::shared_lock const lock(device->mutex);
        if (const auto lost = rejectLostDevice(*device, error))
        {
            return *lost;
        }
        const auto* token = findToken(device->tokens, buffer, TokenKind::buffer);
        if (token == nullptr)
        {
            return invalidHandle(error);
        }
        for (const auto& range : nativeRanges)
        {
            if (!validMappedBufferRange(token->byteSize, range.offset, range.size))
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "a mapped range is empty or outside the buffer");
            }
        }
        const auto nativeBuffer = rachis::BufferHandle{token->native};
        return result(error,
                      flush ? device->native.flushMappedMemoryRanges(nativeBuffer, nativeRanges)
                            : device->native.invalidateMappedMemoryRanges(nativeBuffer, nativeRanges));
    }

    [[nodiscard]] rachis_status collectBindingWrites(const rachis_device& device,
                                                     std::span<const rachis_buffer_binding> buffers,
                                                     std::span<const rachis_texture_binding> textures,
                                                     std::span<const rachis_sampler_binding> samplers,
                                                     bool zeroHandleEmptiesElement,
                                                     std::vector<rachis::BindingWrite>& writes,
                                                     rachis_error* error)
    {
        writes.reserve(buffers.size() + textures.size() + samplers.size());
        for (const auto& value : buffers)
        {
            if (zeroHandleEmptiesElement && value.buffer == 0)
            {
                writes.push_back({.binding = value.binding, .arrayElement = value.array_element});
                continue;
            }
            const auto* buffer = findToken(device.tokens, value.buffer, TokenKind::buffer);
            if (buffer == nullptr)
            {
                return invalidHandle(error);
            }
            writes.push_back({.binding = value.binding,
                              .arrayElement = value.array_element,
                              .resource = rachis::BufferBinding{.buffer = rachis::BufferHandle{buffer->native},
                                                                .offset = value.offset,
                                                                .size = value.size,
                                                                .structureByteStride = value.structure_byte_stride}});
        }
        for (const auto& value : textures)
        {
            if (zeroHandleEmptiesElement && value.view == 0)
            {
                writes.push_back({.binding = value.binding, .arrayElement = value.array_element});
                continue;
            }
            const auto* view = findView(device.tokens, value.view);
            if (view == nullptr)
            {
                return invalidHandle(error);
            }
            writes.push_back({.binding = value.binding,
                              .arrayElement = value.array_element,
                              .resource = rachis::TextureViewHandle{view->native}});
        }
        for (const auto& value : samplers)
        {
            if (zeroHandleEmptiesElement && value.sampler == 0)
            {
                writes.push_back({.binding = value.binding, .arrayElement = value.array_element});
                continue;
            }
            const auto* sampler = findToken(device.tokens, value.sampler, TokenKind::sampler);
            if (sampler == nullptr)
            {
                return invalidHandle(error);
            }
            writes.push_back({.binding = value.binding,
                              .arrayElement = value.array_element,
                              .resource = rachis::SamplerHandle{sampler->native}});
        }
        return RACHIS_STATUS_OK;
    }

    template<typename Record>
    [[nodiscard]] rachis_status recordCommand(rachis_device& device,
                                              rachis_handle command,
                                              rachis_error* error,
                                              Record&& record)
    {
        std::shared_lock const lock(device.mutex);
        if (const auto lost = rejectLostDevice(device, error))
        {
            return *lost;
        }
        CommandUseGuard commandUse;
        rachis::CommandBuffer* commandBuffer = nullptr;
        if (const auto status = acquireCommandForRecording(device, command, commandUse, commandBuffer, error);
            status != RACHIS_STATUS_OK)
        {
            return status;
        }
        return std::forward<Record>(record)(*commandBuffer);
    }

    struct IndirectBuffers final
    {
        rachis::BufferHandle arguments;
        rachis::BufferHandle count;
    };

    [[nodiscard]] std::optional<IndirectBuffers> findIndirectBuffers(const rachis_device& device,
                                                                     rachis_handle argumentBuffer,
                                                                     rachis_handle countBuffer) noexcept
    {
        const auto* arguments = findToken(device.tokens, argumentBuffer, TokenKind::buffer);
        if (arguments == nullptr)
        {
            return std::nullopt;
        }
        IndirectBuffers buffers{.arguments = rachis::BufferHandle{arguments->native}};
        if (countBuffer != 0)
        {
            const auto* count = findToken(device.tokens, countBuffer, TokenKind::buffer);
            if (count == nullptr)
            {
                return std::nullopt;
            }
            buffers.count = rachis::BufferHandle{count->native};
        }
        return buffers;
    }

    template<typename Desc>
    [[nodiscard]] std::optional<rachis_status> rejectIndirectDesc(const rachis_device* device,
                                                                  const Desc* desc,
                                                                  rachis_error* error) noexcept
    {
        if (device == nullptr)
        {
            return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
        }
        if (desc == nullptr)
        {
            return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
        }
        if (!validDesc(desc))
        {
            return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported indirect command descriptor");
        }
        return std::nullopt;
    }

    [[nodiscard]] rachis_status drawIndirect(rachis_device* device,
                                             rachis_handle command,
                                             const rachis_indirect_draw_desc* desc,
                                             rachis_error* error,
                                             bool indexed)
    {
        if (const auto rejected = rejectIndirectDesc(device, desc, error))
        {
            return *rejected;
        }
        if (desc->draw_index_prefix > 1)
        {
            return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid indirect draw descriptor");
        }
        return recordCommand(*device, command, error, [&](rachis::CommandBuffer& commandBuffer) {
            const auto buffers = findIndirectBuffers(*device, desc->argument_buffer, desc->count_buffer);
            if (!buffers)
            {
                return invalidHandle(error);
            }
            const rachis::IndirectDrawDesc native{
                .argumentBuffer = buffers->arguments,
                .argumentOffset = desc->argument_offset,
                .maxDrawCount = desc->max_draw_count,
                .stride = desc->stride,
                .countBuffer = buffers->count,
                .countOffset = desc->count_offset,
                .drawIndexPrefix = desc->draw_index_prefix != 0,
            };
            return result(error,
                          indexed ? device->native.drawIndexedIndirect(commandBuffer, native)
                                  : device->native.drawIndirect(commandBuffer, native));
        });
    }
} // namespace

extern "C"
{
    std::uint32_t rachis_abi_version(void)
    {
        return RACHIS_ABI_VERSION;
    }

    rachis_status rachis_device_create(const rachis_device_desc* desc, rachis_device** output, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "output device is null");
            }
            *output = nullptr;
            if (desc == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
            }
            if (!validDesc(desc))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported device descriptor layout");
            }
            if (desc->backend > RACHIS_BACKEND_METAL || desc->adapter_preference > RACHIS_ADAPTER_WARP ||
                desc->validation_level > RACHIS_VALIDATION_GPU)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid device enum value");
            }
            auto device = std::make_unique<rachis_device>();
            device->callback = desc->diagnostic_callback;
            device->callbackUserData = desc->diagnostic_user_data;
            rachis::DeviceDesc native{};
            native.backend = static_cast<rachis::Backend>(desc->backend);
            native.adapterPreference = static_cast<rachis::AdapterPreference>(desc->adapter_preference);
            native.validation = static_cast<rachis::ValidationLevel>(desc->validation_level);
            native.requireValidation = desc->require_validation != 0;
            native.enableDred = desc->enable_dred != 0;
            native.enableDeviceFault = desc->enable_device_fault != 0;
            if (desc->adapter_id != 0)
            {
                native.adapterId = rachis::AdapterId{desc->adapter_id};
            }
            native.applicationName = optionalString(desc->application_name);
            native.engineName = optionalString(desc->engine_name);
            if (device->callback != nullptr)
            {
                native.diagnosticCallback = onDiagnostic;
                native.diagnosticUserData = device.get();
            }
            auto created = rachis::createDevice(native);
            if (!created)
            {
                return result(error, created.error());
            }
            device->native = std::move(*created);
            *output = device.release();
            return RACHIS_STATUS_OK;
        });
    }

    void rachis_device_destroy(rachis_device* device)
    {
        try
        {
            delete device;
        }

        catch (...)
        {
        }
    }

    rachis_status rachis_device_diagnostic_counts(rachis_device* device,
                                                  rachis_diagnostic_counts* output,
                                                  rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or diagnostic output is null");
            }
            if (output->struct_size != sizeof(*output))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "invalid diagnostic count output layout");
            }
            std::shared_lock const lock(device->mutex);
            const auto counts = device->native.diagnosticCounts();
            output->information = counts.information;
            output->warnings = counts.warnings;
            output->errors = counts.errors;
            output->corruptions = counts.corruptions;
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_device_operational_state(rachis_device* device, std::uint32_t* output, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output state is null");
            }
            std::shared_lock const lock(device->mutex);
            *output = static_cast<std::uint32_t>(device->native.operationalState());
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_device_backend(rachis_device* device, std::uint32_t* output, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or backend output is null");
            }
            std::shared_lock const lock(device->mutex);
            const auto adapter = device->native.adapterInfo();
            if (!adapter)
            {
                return fail(error, RACHIS_STATUS_INVALID_STATE, "device has no selected adapter");
            }
            *output = static_cast<std::uint32_t>(adapter->backend);
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_device_adapter_info(rachis_device* device, rachis_adapter_info* output, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or adapter output is null");
            }
            if (output->struct_size != sizeof(*output))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "invalid adapter output layout");
            }
            std::shared_lock const lock(device->mutex);
            const auto adapter = device->native.adapterInfo();
            if (!adapter)
            {
                return fail(error, RACHIS_STATUS_INVALID_STATE, "device has no selected adapter");
            }
            output->backend = static_cast<std::uint32_t>(adapter->backend);
            output->id = adapter->id.value;
            output->vendor_id = adapter->vendorId;
            output->device_id = adapter->deviceId;
            output->type = static_cast<std::uint32_t>(adapter->type);
            output->reserved0 = 0;
            output->dedicated_video_memory = adapter->dedicatedVideoMemory;
            output->shared_system_memory = adapter->sharedSystemMemory;
            std::memset(output->name, 0, sizeof(output->name));
            rachis::cabi::detail::copyText(output->name, adapter->name);
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_device_loss_error(rachis_device* device, rachis_error* lossError, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || lossError == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or loss error is null");
            }
            if (lossError->struct_size != sizeof(*lossError))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "invalid loss error layout");
            }
            std::shared_lock const lock(device->mutex);
            const auto loss = device->native.deviceLossError();
            if (loss)
            {
                (void)result(lossError, *loss);
            }
            else
            {
                (void)fail(lossError, RACHIS_STATUS_OK, "");
            }
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_device_query_capabilities(rachis_device* device,
                                                   rachis_device_capabilities* output,
                                                   rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or capabilities output is null");
            }
            if (output->struct_size != sizeof(*output))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "invalid device capabilities output layout");
            }
            std::shared_lock const lock(device->mutex);
            const auto capabilities = device->native.capabilities();
            if (!capabilities)
            {
                return fail(error, RACHIS_STATUS_INVALID_STATE, "device has no capabilities");
            }
            output->unified_memory = capabilities->unifiedMemory ? 1U : 0U;
            output->persistent_mapping = capabilities->persistentMapping ? 1U : 0U;
            output->mapped_memory_host_coherent = capabilities->mappedMemory.hostCoherent ? 1U : 0U;
            output->mapped_memory_non_coherent_atom_size = capabilities->mappedMemory.nonCoherentAtomSize;
            output->vertex_formats = capabilities->vertexInput.supportedFormats;
            output->max_vertex_buffers = capabilities->limits.maxVertexBuffers;
            output->max_vertex_attributes = capabilities->limits.maxVertexAttributes;
            output->max_vertex_buffer_stride = capabilities->vertexInput.maxBufferStride;
            output->max_vertex_attribute_offset = capabilities->vertexInput.maxAttributeOffset;
            output->max_vertex_instance_step_rate = capabilities->vertexInput.maxInstanceStepRate;
            output->reserved0 = 0;
            const auto flag = [](bool value) noexcept { return value ? 1U : 0U; };
            output->indirect_dispatch = flag(capabilities->indirectDispatch);
            output->multi_dispatch_indirect = flag(capabilities->multiDispatchIndirect);
            output->dispatch_indirect_count = flag(capabilities->dispatchIndirectCount);
            output->indirect_draws = flag(capabilities->indirectDraws);
            output->indirect_draw_base_instance = flag(capabilities->indirectDrawBaseInstance);
            output->multi_draw_indirect = flag(capabilities->multiDrawIndirect);
            output->draw_indirect_count = flag(capabilities->drawIndirectCount);
            output->shader_draw_index = flag(capabilities->shaderDrawIndex);
            output->conditional_execution = flag(capabilities->conditionalExecution);
            output->mesh_shaders = flag(capabilities->meshShaders);
            output->task_shaders = flag(capabilities->taskShaders);
            output->partially_bound_binding_arrays = flag(capabilities->partiallyBoundBindingArrays);
            output->variable_descriptor_count_binding_arrays = flag(capabilities->variableDescriptorCountBindingArrays);
            output->update_after_bind_binding_arrays = flag(capabilities->updateAfterBindBindingArrays);
            output->shader_int64 = flag(capabilities->shaderInt64);
            output->shader_int64_atomic_group_shared = flag(capabilities->shaderInt64AtomicGroupShared);
            output->shader_int64_atomic_typed_resource = flag(capabilities->shaderInt64AtomicTypedResource);
            output->shader_int64_atomic_storage_buffer = flag(capabilities->shaderInt64AtomicStorageBuffer);
            output->subgroup_operations = flag(capabilities->subgroupOperations);
            output->subgroup_min_size = capabilities->subgroup.minSize;
            output->subgroup_max_size = capabilities->subgroup.maxSize;
            output->subgroup_required_size_stages =
                bitsOf(capabilities->subgroup.requiredSizeStages) & RACHIS_SHADER_ALL;
            output->max_variable_resource_descriptors = capabilities->limits.maxVariableResourceDescriptors;
            output->max_variable_buffer_descriptors = capabilities->limits.maxVariableBufferDescriptors;
            output->max_variable_sampler_descriptors = capabilities->limits.maxVariableSamplerDescriptors;
            const auto& mesh = capabilities->meshShader;
            output->max_task_workgroup_invocations = mesh.maxTaskWorkgroupInvocations;
            output->max_task_payload_bytes = mesh.maxTaskPayloadBytes;
            output->max_mesh_workgroup_invocations = mesh.maxMeshWorkgroupInvocations;
            output->max_mesh_output_vertices = mesh.maxMeshOutputVertices;
            output->max_mesh_output_primitives = mesh.maxMeshOutputPrimitives;
            output->max_mesh_output_memory_bytes = mesh.maxMeshOutputMemoryBytes;
            for (std::size_t axis = 0; axis < mesh.maxWorkgroupCount.size(); ++axis)
            {
                output->max_mesh_workgroup_count[axis] = mesh.maxWorkgroupCount[axis];
            }
            output->max_mesh_total_workgroups = mesh.maxTotalWorkgroups;
            output->reserved1 = 0;
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_queue_timestamp_frequency(rachis_device* device,
                                                   std::uint32_t queueType,
                                                   std::uint64_t* output,
                                                   rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output frequency is null");
            }
            *output = 0;
            if (queueType > RACHIS_QUEUE_COPY)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid queue type");
            }
            std::shared_lock const lock(device->mutex);
            const auto properties = device->native.queueProperties(static_cast<rachis::QueueType>(queueType));
            if (!properties.timestampQueries || properties.timestampFrequency == 0)
            {
                return fail(error, RACHIS_STATUS_UNSUPPORTED, "the queue does not support timestamp queries");
            }
            *output = properties.timestampFrequency;
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_buffer_create(rachis_device* device,
                                       const rachis_buffer_desc* desc,
                                       rachis_handle* output,
                                       rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (output == nullptr || device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output buffer is null");
            }
            *output = 0;
            if (desc == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
            }
            if (!validDesc(desc))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported buffer descriptor layout");
            }
            constexpr auto knownUsage = RACHIS_BUFFER_COPY_SOURCE | RACHIS_BUFFER_COPY_DESTINATION |
                                        RACHIS_BUFFER_VERTEX | RACHIS_BUFFER_INDEX | RACHIS_BUFFER_CONSTANT |
                                        RACHIS_BUFFER_STORAGE | RACHIS_BUFFER_INDIRECT;
            if (desc->memory_type > RACHIS_MEMORY_READBACK || (desc->usage & ~knownUsage) != 0)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid buffer enum value");
            }
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            const auto status = createToken(*device, TokenKind::buffer, output, error, [&] {
                return device->native.createBuffer(rachis::BufferDesc{
                    .size = desc->size,
                    .usage = static_cast<rachis::BufferUsage>(desc->usage),
                    .memoryType = static_cast<rachis::MemoryType>(desc->memory_type),
                    .label = optionalString(desc->label),
                });
            });
            if (status == RACHIS_STATUS_OK)
            {
                device->tokens.find(*output)->byteSize = desc->size;
            }
            return status;
        });
    }

    rachis_status rachis_buffer_destroy(rachis_device* device, rachis_handle buffer, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::scoped_lock const lock(device->mutex);
            const auto* token = findToken(device->tokens, buffer, TokenKind::buffer);
            if (token == nullptr)
            {
                return invalidHandle(error);
            }
            const auto destroyed = device->native.destroyBuffer(rachis::BufferHandle{token->native});
            if (!destroyed)
            {
                return result(error, destroyed.error());
            }
            device->tokens.erase(buffer);
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_buffer_write(rachis_device* device,
                                      rachis_handle buffer,
                                      std::uint64_t offset,
                                      const void* bytes,
                                      std::uint64_t size,
                                      rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || !validHostByteSpan(bytes, size))
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid buffer write arguments");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            const auto* token = findToken(device->tokens, buffer, TokenKind::buffer);
            if (token == nullptr)
            {
                return invalidHandle(error);
            }
            if (!validHostBufferRange(token->byteSize, offset, size))
            {
                return fail(error,
                            RACHIS_STATUS_INVALID_ARGUMENT,
                            "the host buffer write range is empty or outside the buffer");
            }
            return result(
                error,
                device->native.writeBuffer(rachis::BufferHandle{token->native},
                                           offset,
                                           {static_cast<const std::byte*>(bytes), static_cast<std::size_t>(size)}));
        });
    }

    rachis_status rachis_buffer_read(rachis_device* device,
                                     rachis_handle buffer,
                                     std::uint64_t offset,
                                     void* bytes,
                                     std::uint64_t size,
                                     rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || !validHostByteSpan(bytes, size))
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid buffer read arguments");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            const auto* token = findToken(device->tokens, buffer, TokenKind::buffer);
            if (token == nullptr)
            {
                return invalidHandle(error);
            }
            if (!validHostBufferRange(token->byteSize, offset, size))
            {
                return fail(
                    error, RACHIS_STATUS_INVALID_ARGUMENT, "the host buffer read range is empty or outside the buffer");
            }
            return result(error,
                          device->native.readBuffer(rachis::BufferHandle{token->native},
                                                    offset,
                                                    {static_cast<std::byte*>(bytes), static_cast<std::size_t>(size)}));
        });
    }

    rachis_status rachis_buffer_map(rachis_device* device,
                                    rachis_handle buffer,
                                    std::uint64_t offset,
                                    std::uint64_t size,
                                    rachis_mapped_memory* output,
                                    rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or mapped memory output is null");
            }
            if (output->struct_size != sizeof(*output))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "invalid mapped memory output layout");
            }
            output->data = nullptr;
            output->size = 0;
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            const auto* token = findToken(device->tokens, buffer, TokenKind::buffer);
            if (token == nullptr)
            {
                return invalidHandle(error);
            }
            if (!validMappedBufferRange(token->byteSize, offset, size))
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "the map range is empty or outside the buffer");
            }
            const auto mapped = device->native.mapBuffer(rachis::BufferHandle{token->native}, offset, size);
            if (!mapped)
            {
                return result(error, mapped.error());
            }
            output->data = mapped->data;
            output->size = mapped->size;
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_buffer_unmap(rachis_device* device, rachis_handle buffer, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::shared_lock const lock(device->mutex);
            const auto* token = findToken(device->tokens, buffer, TokenKind::buffer);
            if (token == nullptr)
            {
                return invalidHandle(error);
            }
            return result(error, device->native.unmapBuffer(rachis::BufferHandle{token->native}));
        });
    }

    rachis_status rachis_buffer_flush_mapped_ranges(rachis_device* device,
                                                    rachis_handle buffer,
                                                    const rachis_mapped_range* ranges,
                                                    std::uint32_t rangeCount,
                                                    rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            return synchronizeMappedRanges(device, buffer, ranges, rangeCount, error, true);
        });
    }

    rachis_status rachis_buffer_invalidate_mapped_ranges(rachis_device* device,
                                                         rachis_handle buffer,
                                                         const rachis_mapped_range* ranges,
                                                         std::uint32_t rangeCount,
                                                         rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            return synchronizeMappedRanges(device, buffer, ranges, rangeCount, error, false);
        });
    }

    rachis_status rachis_texture_create(rachis_device* device,
                                        const rachis_texture_desc* desc,
                                        rachis_handle* output,
                                        rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output texture is null");
            }
            *output = 0;
            if (desc == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
            }
            if (!validDesc(desc))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported texture descriptor layout");
            }
            const auto format = nativeFormat(desc->format);
            if (!format || desc->dimension > RACHIS_TEXTURE_3D || (desc->usage & ~knownTextureUsage) != 0)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid texture enum value");
            }
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            return createToken(*device, TokenKind::texture, output, error, [&] {
                rachis::TextureDesc native{};
                native.dimension = static_cast<rachis::TextureDimension>(desc->dimension);
                native.width = desc->width;
                native.height = desc->height;
                native.depthOrArrayLayers = desc->depth_or_array_layers;
                native.mipLevels = desc->mip_levels;
                native.sampleCount = desc->sample_count;
                native.format = *format;
                native.usage = static_cast<rachis::TextureUsage>(desc->usage);
                native.label = optionalString(desc->label);
                return device->native.createTexture(native);
            });
        });
    }

    rachis_status rachis_texture_destroy(rachis_device* device, rachis_handle texture, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::scoped_lock const lock(device->mutex);
            auto* token = findMutableToken(device->tokens, texture, TokenKind::texture);
            if (token == nullptr)
            {
                return invalidHandle(error);
            }
            if (token->dependents != 0)
            {
                token->kind = TokenKind::retiredTexture;
                return RACHIS_STATUS_OK;
            }
            const auto destroyed = device->native.destroyTexture(rachis::TextureHandle{token->native});
            if (!destroyed)
            {
                return result(error, destroyed.error());
            }
            device->tokens.erase(texture);
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_texture_lease_acquire(rachis_device* device,
                                               rachis_handle texture,
                                               rachis_handle* lease,
                                               rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || lease == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output lease is null");
            }
            *lease = 0;
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            const auto owner = textureOwner(device->tokens, texture);
            if (owner == 0)
            {
                return findTexture(device->tokens, texture) != nullptr
                           ? fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "swapchain images cannot be leased")
                           : invalidHandle(error);
            }
            const auto id = reserveToken(device->tokens, TokenKind::textureLease, owner);
            if (id == 0)
            {
                return fail(error, RACHIS_STATUS_RESOURCE_EXHAUSTED, "C handle space exhausted");
            }
            auto& ownerToken = *device->tokens.find(owner);
            device->tokens.find(id)->native = ownerToken.native;
            ++ownerToken.dependents;
            *lease = id;
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_texture_lease_release(rachis_device* device, rachis_handle lease, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::scoped_lock const lock(device->mutex);
            if (findToken(device->tokens, lease, TokenKind::textureLease) == nullptr)
            {
                return invalidHandle(error);
            }
            return releaseTextureDependent(*device, lease, error);
        });
    }

    rachis_status rachis_texture_view_create(rachis_device* device,
                                             const rachis_texture_view_desc* desc,
                                             rachis_handle* output,
                                             rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output view is null");
            }
            *output = 0;
            if (desc == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
            }
            if (!validDesc(desc))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported texture view descriptor layout");
            }
            const auto format = desc->format == RACHIS_FORMAT_UNKNOWN ? std::optional{rachis::Format::unknown}
                                                                      : nativeFormat(desc->format);
            if (!format || desc->usage > RACHIS_VIEW_USAGE_DEPTH_STENCIL || desc->dimension > RACHIS_VIEW_CUBE_ARRAY ||
                !validAspect(desc->aspect))
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid texture view enum value");
            }
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            const auto owner = textureOwner(device->tokens, desc->texture);
            if (owner == 0)
            {
                return findTexture(device->tokens, desc->texture) != nullptr
                           ? fail(error,
                                  RACHIS_STATUS_INVALID_ARGUMENT,
                                  "use the borrowed render-target view of a swapchain image")
                           : invalidHandle(error);
            }
            const auto nativeTexture = rachis::TextureHandle{device->tokens.find(owner)->native};
            const auto status = createToken(*device, TokenKind::textureView, output, error, [&] {
                rachis::TextureViewDesc native{};
                native.texture = nativeTexture;
                native.usage = static_cast<rachis::TextureViewUsage>(desc->usage);
                native.dimension = static_cast<rachis::TextureViewDimension>(desc->dimension);
                native.format = *format;
                native.aspect = static_cast<rachis::TextureAspect>(desc->aspect);
                native.firstMipLevel = desc->first_mip_level;
                native.mipLevelCount = desc->mip_level_count;
                native.firstArrayLayer = desc->first_array_layer;
                native.arrayLayerCount = desc->array_layer_count;
                native.label = optionalString(desc->label);
                return device->native.createTextureView(native);
            });
            if (status == RACHIS_STATUS_OK)
            {
                device->tokens.find(*output)->parent = owner;
                ++device->tokens.find(owner)->dependents;
            }
            return status;
        });
    }

    rachis_status rachis_texture_view_destroy(rachis_device* device, rachis_handle view, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::scoped_lock const lock(device->mutex);
            auto* token = findMutableToken(device->tokens, view, TokenKind::textureView);
            if (token == nullptr)
            {
                token = findMutableToken(device->tokens, view, TokenKind::retiredView);
            }
            if (token == nullptr)
            {
                return invalidHandle(error);
            }
            if (token->kind == TokenKind::textureView)
            {
                const auto destroyed = device->native.destroyTextureView(rachis::TextureViewHandle{token->native});
                if (!destroyed)
                {
                    return result(error, destroyed.error());
                }
                token->kind = TokenKind::retiredView;
            }
            return releaseTextureDependent(*device, view, error);
        });
    }

    rachis_status rachis_texture_query_transfer_layout(rachis_device* device,
                                                       rachis_handle texture,
                                                       const rachis_buffer_texture_region* region,
                                                       rachis_texture_transfer_layout* output,
                                                       rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || region == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device, region, or output layout is null");
            }
            if (output->struct_size != sizeof(*output))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "invalid transfer layout output layout");
            }
            if (!validAspect(region->aspect))
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid texture aspect");
            }
            std::shared_lock const lock(device->mutex);
            const auto* token = findTexture(device->tokens, texture);
            if (token == nullptr)
            {
                return invalidHandle(error);
            }
            const auto layout =
                device->native.queryTextureTransferLayout(rachis::TextureHandle{token->native}, regionOf(*region));
            if (!layout)
            {
                return result(error, layout.error());
            }
            output->bytes_per_row = layout->bytesPerRow;
            output->rows_per_image = layout->rowsPerImage;
            output->row_count = layout->rowCount;
            output->row_data_size = layout->rowDataSize;
            output->bytes_per_image = layout->bytesPerImage;
            output->required_buffer_size = layout->requiredBufferSize;
            output->buffer_offset_alignment = layout->bufferOffsetAlignment;
            output->row_pitch_alignment = layout->rowPitchAlignment;
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_command_pool_create(rachis_device* device,
                                             const rachis_command_pool_desc* desc,
                                             rachis_handle* output,
                                             rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output pool is null");
            }
            *output = 0;
            if (desc == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
            }
            if (!validDesc(desc))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported command pool descriptor layout");
            }
            if (desc->queue_type > RACHIS_QUEUE_COPY)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid queue type");
            }
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            return createToken(*device, TokenKind::commandPool, output, error, [&] {
                return device->native.createCommandPool(rachis::CommandPoolDesc{
                    .queueType = static_cast<rachis::QueueType>(desc->queue_type),
                    .maxCommandBuffers = desc->max_command_buffers,
                    .label = optionalString(desc->label),
                });
            });
        });
    }

    rachis_status rachis_command_pool_destroy(rachis_device* device, rachis_handle pool, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::scoped_lock const lock(device->mutex);
            const auto* token = findToken(device->tokens, pool, TokenKind::commandPool);
            if (token == nullptr)
            {
                return invalidHandle(error);
            }
            const auto destroyed = device->native.destroyCommandPool(rachis::CommandPoolHandle{token->native});
            if (!destroyed)
            {
                return result(error, destroyed.error());
            }
            device->tokens.erase(pool);
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_command_acquire(rachis_device* device,
                                         rachis_handle pool,
                                         rachis_handle* output,
                                         rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output command is null");
            }
            *output = 0;
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            const auto* token = findToken(device->tokens, pool, TokenKind::commandPool);
            if (token == nullptr)
            {
                return invalidHandle(error);
            }
            const auto nativePool = rachis::CommandPoolHandle{token->native};
            const auto id = device->commands.emplace(activeTokenSource());
            if (id == 0)
            {
                return fail(error, RACHIS_STATUS_RESOURCE_EXHAUSTED, "C handle space exhausted");
            }
            try
            {
                auto created = device->native.acquireCommandBuffer(nativePool);
                if (!created)
                {
                    device->commands.erase(id);
                    return result(error, created.error());
                }
                device->commands.find(id)->command = std::move(*created);
                *output = id;
                return RACHIS_STATUS_OK;
            }
            catch (...)
            {
                device->commands.erase(id);
                throw;
            }
        });
    }

    rachis_status rachis_command_copy_buffer(rachis_device* device,
                                             rachis_handle command,
                                             rachis_handle source,
                                             rachis_handle destination,
                                             std::uint64_t sourceOffset,
                                             std::uint64_t destinationOffset,
                                             std::uint64_t size,
                                             rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            CommandUseGuard commandUse;
            rachis::CommandBuffer* commandBuffer = nullptr;
            if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            const auto* sourceEntry = findToken(device->tokens, source, TokenKind::buffer);
            const auto* destinationEntry = findToken(device->tokens, destination, TokenKind::buffer);
            if (sourceEntry == nullptr || destinationEntry == nullptr)
            {
                return invalidHandle(error);
            }
            return result(error,
                          device->native.copyBuffer(
                              *commandBuffer,
                              rachis::BufferCopy{.source = rachis::BufferHandle{sourceEntry->native},
                                                 .sourceOffset = sourceOffset,
                                                 .destination = rachis::BufferHandle{destinationEntry->native},
                                                 .destinationOffset = destinationOffset,
                                                 .size = size}));
        });
    }

    rachis_status rachis_command_copy_buffer_to_texture(rachis_device* device,
                                                        rachis_handle command,
                                                        const rachis_buffer_texture_copy_desc* desc,
                                                        rachis_error* error)
    {
        return boundary(error,
                        [&]() -> rachis_status { return copyBufferTexture(device, command, desc, error, true); });
    }

    rachis_status rachis_command_copy_texture_to_buffer(rachis_device* device,
                                                        rachis_handle command,
                                                        const rachis_buffer_texture_copy_desc* desc,
                                                        rachis_error* error)
    {
        return boundary(error,
                        [&]() -> rachis_status { return copyBufferTexture(device, command, desc, error, false); });
    }

    rachis_status rachis_command_fill_buffer(rachis_device* device,
                                             rachis_handle command,
                                             rachis_handle buffer,
                                             std::uint64_t offset,
                                             std::uint64_t size,
                                             std::uint32_t value,
                                             rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            return recordCommand(*device, command, error, [&](rachis::CommandBuffer& commandBuffer) {
                const auto* token = findToken(device->tokens, buffer, TokenKind::buffer);
                if (token == nullptr)
                {
                    return invalidHandle(error);
                }
                return result(
                    error,
                    device->native.fillBuffer(commandBuffer, rachis::BufferHandle{token->native}, offset, size, value));
            });
        });
    }

    rachis_status rachis_command_clear_storage_texture(rachis_device* device,
                                                       rachis_handle command,
                                                       rachis_handle view,
                                                       const rachis_storage_texture_clear_value* value,
                                                       rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || value == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or clear value is null");
            }
            if (value->type > RACHIS_STORAGE_TEXTURE_CLEAR_SINT32)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid storage texture clear type");
            }
            rachis::StorageTextureClearValue clear{.type = static_cast<rachis::StorageTextureClearType>(value->type)};
            for (std::size_t channel = 0; channel < clear.float32.size(); ++channel)
            {
                clear.float32[channel] = value->float32[channel];
                clear.uint32[channel] = value->uint32[channel];
                clear.sint32[channel] = value->sint32[channel];
            }
            return recordCommand(*device, command, error, [&](rachis::CommandBuffer& commandBuffer) {
                const auto* token = findView(device->tokens, view);
                if (token == nullptr)
                {
                    return invalidHandle(error);
                }
                return result(
                    error,
                    device->native.clearStorageTexture(commandBuffer, rachis::TextureViewHandle{token->native}, clear));
            });
        });
    }

    rachis_status rachis_command_barriers(rachis_device* device,
                                          rachis_handle command,
                                          const rachis_barrier_batch* batch,
                                          rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            if (batch == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
            }
            if (!validDesc(batch))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported barrier batch layout");
            }
            if (!validArray(batch->global_barriers, batch->global_barrier_count) ||
                !validArray(batch->buffer_barriers, batch->buffer_barrier_count) ||
                !validArray(batch->texture_barriers, batch->texture_barrier_count))
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "barrier arrays are null");
            }
            const auto validSync = [](std::uint32_t stagesBefore,
                                      std::uint32_t stagesAfter,
                                      std::uint32_t accessBefore,
                                      std::uint32_t accessAfter) noexcept {
                return validStages(stagesBefore) && validStages(stagesAfter) && validAccess(accessBefore) &&
                       validAccess(accessAfter);
            };
            std::vector<rachis::GlobalBarrier> globals;
            globals.reserve(batch->global_barrier_count);
            for (std::uint32_t i = 0; i < batch->global_barrier_count; ++i)
            {
                const auto& value = batch->global_barriers[i];
                if (!validSync(value.stages_before, value.stages_after, value.access_before, value.access_after))
                {
                    return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid global barrier");
                }
                globals.push_back({.stagesBefore = static_cast<rachis::PipelineStage>(value.stages_before),
                                   .stagesAfter = static_cast<rachis::PipelineStage>(value.stages_after),
                                   .accessBefore = static_cast<rachis::Access>(value.access_before),
                                   .accessAfter = static_cast<rachis::Access>(value.access_after)});
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            CommandUseGuard commandUse;
            rachis::CommandBuffer* commandBuffer = nullptr;
            if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            std::vector<rachis::BufferBarrier> buffers;
            buffers.reserve(batch->buffer_barrier_count);
            for (std::uint32_t i = 0; i < batch->buffer_barrier_count; ++i)
            {
                const auto& value = batch->buffer_barriers[i];
                if (!validSync(value.stages_before, value.stages_after, value.access_before, value.access_after))
                {
                    return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid buffer barrier");
                }
                const auto* buffer = findToken(device->tokens, value.buffer, TokenKind::buffer);
                if (buffer == nullptr)
                {
                    return invalidHandle(error);
                }
                buffers.push_back({.buffer = rachis::BufferHandle{buffer->native},
                                   .stagesBefore = static_cast<rachis::PipelineStage>(value.stages_before),
                                   .stagesAfter = static_cast<rachis::PipelineStage>(value.stages_after),
                                   .accessBefore = static_cast<rachis::Access>(value.access_before),
                                   .accessAfter = static_cast<rachis::Access>(value.access_after),
                                   .offset = value.offset,
                                   .size = value.size});
            }
            std::vector<rachis::TextureBarrier> textures;
            textures.reserve(batch->texture_barrier_count);
            for (std::uint32_t i = 0; i < batch->texture_barrier_count; ++i)
            {
                const auto& value = batch->texture_barriers[i];
                if (!validSync(value.stages_before, value.stages_after, value.access_before, value.access_after) ||
                    !validLayout(value.layout_before) || !validLayout(value.layout_after) || !validAspect(value.aspect))
                {
                    return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid texture barrier");
                }
                const auto* texture = findTexture(device->tokens, value.texture);
                if (texture == nullptr)
                {
                    return invalidHandle(error);
                }
                textures.push_back({
                    .texture = rachis::TextureHandle{texture->native},
                    .stagesBefore = static_cast<rachis::PipelineStage>(value.stages_before),
                    .stagesAfter = static_cast<rachis::PipelineStage>(value.stages_after),
                    .accessBefore = static_cast<rachis::Access>(value.access_before),
                    .accessAfter = static_cast<rachis::Access>(value.access_after),
                    .layoutBefore = static_cast<rachis::TextureLayout>(value.layout_before),
                    .layoutAfter = static_cast<rachis::TextureLayout>(value.layout_after),
                    .subresources = {.firstMipLevel = value.first_mip_level,
                                     .mipLevelCount = value.mip_level_count,
                                     .firstArrayLayer = value.first_array_layer,
                                     .arrayLayerCount = value.array_layer_count,
                                     .aspect = static_cast<rachis::TextureAspect>(value.aspect)},
                });
            }
            return result(error,
                          device->native.resourceBarriers(*commandBuffer,
                                                          rachis::ResourceBarrierBatch{
                                                              .globalBarriers = globals,
                                                              .bufferBarriers = buffers,
                                                              .textureBarriers = textures,
                                                          }));
        });
    }

    rachis_status rachis_command_begin_rendering(rachis_device* device,
                                                 rachis_handle command,
                                                 const rachis_rendering_desc* desc,
                                                 rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            if (desc == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
            }
            if (!validDesc(desc))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported rendering descriptor layout");
            }
            if (!validArray(desc->color_attachments, desc->color_attachment_count))
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "color attachments are null");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            CommandUseGuard commandUse;
            rachis::CommandBuffer* commandBuffer = nullptr;
            if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            std::vector<rachis::ColorAttachment> colors;
            colors.reserve(desc->color_attachment_count);
            for (std::uint32_t i = 0; i < desc->color_attachment_count; ++i)
            {
                const auto& value = desc->color_attachments[i];
                if (value.load_op > RACHIS_LOAD_OP_DISCARD || value.store_op > RACHIS_STORE_OP_DISCARD)
                {
                    return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid color attachment operation");
                }
                const auto* view = findView(device->tokens, value.view);
                if (view == nullptr)
                {
                    return invalidHandle(error);
                }
                colors.push_back({
                    .view = rachis::TextureViewHandle{view->native},
                    .loadOperation = static_cast<rachis::LoadOperation>(value.load_op),
                    .storeOperation = static_cast<rachis::StoreOperation>(value.store_op),
                    .clearValue =
                        {value.clear_color[0], value.clear_color[1], value.clear_color[2], value.clear_color[3]},
                });
            }
            std::optional<rachis::DepthStencilAttachment> depthStencil;
            if (const auto* value = desc->depth_stencil_attachment; value != nullptr)
            {
                if (value->depth_load_op > RACHIS_LOAD_OP_DISCARD || value->depth_store_op > RACHIS_STORE_OP_DISCARD ||
                    value->stencil_load_op > RACHIS_LOAD_OP_DISCARD ||
                    value->stencil_store_op > RACHIS_STORE_OP_DISCARD)
                {
                    return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid depth-stencil attachment operation");
                }
                const auto* view = findView(device->tokens, value->view);
                if (view == nullptr)
                {
                    return invalidHandle(error);
                }
                depthStencil = rachis::DepthStencilAttachment{
                    .view = rachis::TextureViewHandle{view->native},
                    .depthLoadOperation = static_cast<rachis::LoadOperation>(value->depth_load_op),
                    .depthStoreOperation = static_cast<rachis::StoreOperation>(value->depth_store_op),
                    .depthClearValue = value->depth_clear_value,
                    .stencilLoadOperation = static_cast<rachis::LoadOperation>(value->stencil_load_op),
                    .stencilStoreOperation = static_cast<rachis::StoreOperation>(value->stencil_store_op),
                    .stencilClearValue = value->stencil_clear_value,
                };
            }
            return result(error,
                          device->native.beginRendering(*commandBuffer,
                                                        rachis::RenderingDesc{.colorAttachments = colors,
                                                                              .depthStencilAttachment = depthStencil}));
        });
    }

    rachis_status rachis_command_end_rendering(rachis_device* device, rachis_handle command, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            CommandUseGuard commandUse;
            rachis::CommandBuffer* commandBuffer = nullptr;
            if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            return result(error, device->native.endRendering(*commandBuffer));
        });
    }

    rachis_status rachis_command_discard(rachis_device* device, rachis_handle command, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::scoped_lock const lock(device->mutex);
            if (!device->commands.erase(command))
            {
                return invalidHandle(error);
            }
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_command_submit(rachis_device* device,
                                        rachis_handle command,
                                        rachis_handle* timeline,
                                        rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || timeline == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output timeline is null");
            }
            *timeline = 0;
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            auto* const entry = device->commands.find(command);
            if (entry == nullptr || !entry->command.isValid())
            {
                return invalidHandle(error);
            }
            const auto id = device->timelines.emplace(activeTokenSource());
            if (id == 0)
            {
                return fail(error, RACHIS_STATUS_RESOURCE_EXHAUSTED, "C handle space exhausted");
            }
            std::optional<rachis::Result<rachis::TimelinePoint>> submitted;
            try
            {
                submitted.emplace(device->native.submit(std::span<rachis::CommandBuffer>{&entry->command, 1}));
            }
            catch (...)
            {
                device->timelines.erase(id);
                if (!entry->command.isValid())
                {
                    device->commands.erase(command);
                }
                throw;
            }
            if (!*submitted)
            {
                device->timelines.erase(id);
                if (!entry->command.isValid())
                {
                    device->commands.erase(command);
                }
                return result(error, submitted->error());
            }
            *device->timelines.find(id) = **submitted;
            device->commands.erase(command);
            *timeline = id;
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_timeline_wait(rachis_device* device, rachis_handle timeline, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            rachis::TimelinePoint point;
            {
                std::shared_lock const lock(device->mutex);
                const auto* const entry = device->timelines.find(timeline);
                if (entry == nullptr)
                {
                    return invalidHandle(error);
                }
                point = *entry;
            }
            return result(error, device->native.wait(point));
        });
    }

    rachis_status rachis_timeline_release(rachis_device* device, rachis_handle timeline, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::scoped_lock const lock(device->mutex);
            if (!device->timelines.erase(timeline))
            {
                return invalidHandle(error);
            }
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_binding_layout_create(rachis_device* device,
                                               const rachis_binding_layout_desc* desc,
                                               rachis_handle* output,
                                               rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output layout is null");
            }
            *output = 0;
            if (desc == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
            }
            if (!validDesc(desc))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported binding layout descriptor");
            }
            if (!validArray(desc->entries, desc->entry_count))
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "binding layout entries are null");
            }
            std::vector<rachis::BindingLayoutEntry> entries;
            entries.reserve(desc->entry_count);
            for (std::uint32_t i = 0; i < desc->entry_count; ++i)
            {
                const auto& value = desc->entries[i];
                if (value.type > RACHIS_BINDING_SAMPLER || value.count == 0 || value.visibility == 0 ||
                    (value.visibility & ~RACHIS_SHADER_ALL) != 0 || (value.flags & ~knownBindingFlags) != 0)
                {
                    return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid binding layout entry");
                }
                entries.push_back(rachis::BindingLayoutEntry{
                    .binding = value.binding,
                    .type = static_cast<rachis::BindingType>(value.type),
                    .visibility = static_cast<rachis::ShaderStage>(value.visibility),
                    .count = value.count,
                    .flags = static_cast<rachis::BindingFlag>(value.flags),
                });
            }
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            return createToken(*device, TokenKind::bindingLayout, output, error, [&] {
                return device->native.createBindingLayout({.entries = entries, .label = optionalString(desc->label)});
            });
        });
    }

    rachis_status rachis_binding_layout_destroy(rachis_device* device, rachis_handle layout, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::scoped_lock const lock(device->mutex);
            return destroyToken(*device, TokenKind::bindingLayout, layout, error, [&](std::uint64_t native) {
                return device->native.destroyBindingLayout(rachis::BindingLayoutHandle{native});
            });
        });
    }

    rachis_status rachis_binding_pool_create(rachis_device* device,
                                             const rachis_binding_pool_desc* desc,
                                             rachis_handle* output,
                                             rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output pool is null");
            }
            *output = 0;
            if (desc == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
            }
            if (!validDesc(desc))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported binding pool descriptor");
            }
            if (!validArray(desc->sizes, desc->size_count))
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "binding pool sizes are null");
            }
            std::vector<rachis::BindingPoolSize> sizes;
            sizes.reserve(desc->size_count);
            for (std::uint32_t i = 0; i < desc->size_count; ++i)
            {
                const auto& value = desc->sizes[i];
                if (value.type > RACHIS_BINDING_SAMPLER)
                {
                    return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid binding pool type");
                }
                sizes.push_back(
                    {.type = static_cast<rachis::BindingType>(value.type), .descriptorCount = value.descriptor_count});
            }
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            return createToken(*device, TokenKind::bindingPool, output, error, [&] {
                return device->native.createBindingPool(
                    {.maxBindingSets = desc->max_binding_sets, .sizes = sizes, .label = optionalString(desc->label)});
            });
        });
    }

    rachis_status rachis_binding_pool_destroy(rachis_device* device, rachis_handle pool, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::scoped_lock const lock(device->mutex);
            return destroyToken(*device, TokenKind::bindingPool, pool, error, [&](std::uint64_t native) {
                return device->native.destroyBindingPool(rachis::BindingPoolHandle{native});
            });
        });
    }

    rachis_status rachis_binding_set_create(rachis_device* device,
                                            const rachis_binding_set_desc* desc,
                                            rachis_handle* output,
                                            rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output set is null");
            }
            *output = 0;
            if (desc == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
            }
            if (!validDesc(desc))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported binding set descriptor");
            }
            if (!validArray(desc->writes, desc->write_count) ||
                !validArray(desc->texture_writes, desc->texture_write_count) ||
                !validArray(desc->sampler_writes, desc->sampler_write_count))
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "binding writes are null");
            }
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            const auto* layout = findToken(device->tokens, desc->layout, TokenKind::bindingLayout);
            const auto* pool = findToken(device->tokens, desc->pool, TokenKind::bindingPool);
            if (layout == nullptr || pool == nullptr)
            {
                return invalidHandle(error);
            }
            const auto nativeLayout = rachis::BindingLayoutHandle{layout->native};
            const auto nativePool = rachis::BindingPoolHandle{pool->native};
            std::vector<rachis::BindingWrite> writes;
            if (const auto status = collectBindingWrites(*device,
                                                         std::span{desc->writes, desc->write_count},
                                                         std::span{desc->texture_writes, desc->texture_write_count},
                                                         std::span{desc->sampler_writes, desc->sampler_write_count},
                                                         false,
                                                         writes,
                                                         error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            return createToken(*device, TokenKind::bindingSet, output, error, [&] {
                return device->native.createBindingSet({.layout = nativeLayout,
                                                        .writes = writes,
                                                        .variableDescriptorCount = desc->variable_descriptor_count,
                                                        .label = optionalString(desc->label),
                                                        .pool = nativePool});
            });
        });
    }

    rachis_status rachis_binding_set_update(rachis_device* device,
                                            rachis_handle set,
                                            const rachis_binding_set_update_desc* desc,
                                            rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            if (desc == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
            }
            if (!validDesc(desc))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported binding set update descriptor");
            }
            if (!validArray(desc->writes, desc->write_count) ||
                !validArray(desc->texture_writes, desc->texture_write_count) ||
                !validArray(desc->sampler_writes, desc->sampler_write_count))
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "binding writes are null");
            }
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            const auto* token = findToken(device->tokens, set, TokenKind::bindingSet);
            if (token == nullptr)
            {
                return invalidHandle(error);
            }
            std::vector<rachis::BindingWrite> writes;
            if (const auto status = collectBindingWrites(*device,
                                                         std::span{desc->writes, desc->write_count},
                                                         std::span{desc->texture_writes, desc->texture_write_count},
                                                         std::span{desc->sampler_writes, desc->sampler_write_count},
                                                         true,
                                                         writes,
                                                         error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            return result(error, device->native.updateBindingSet(rachis::BindingSetHandle{token->native}, writes));
        });
    }

    rachis_status rachis_binding_set_destroy(rachis_device* device, rachis_handle set, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::scoped_lock const lock(device->mutex);
            return destroyToken(*device, TokenKind::bindingSet, set, error, [&](std::uint64_t native) {
                return device->native.destroyBindingSet(rachis::BindingSetHandle{native});
            });
        });
    }

    rachis_status rachis_sampler_create(rachis_device* device,
                                        const rachis_sampler_desc* desc,
                                        rachis_handle* output,
                                        rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output sampler is null");
            }
            *output = 0;
            if (desc == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
            }
            if (!validDesc(desc))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported sampler descriptor");
            }
            if (desc->min_filter > RACHIS_FILTER_LINEAR || desc->mag_filter > RACHIS_FILTER_LINEAR ||
                desc->mip_filter > RACHIS_MIP_FILTER_LINEAR || desc->address_u > RACHIS_ADDRESS_CLAMP_TO_BORDER ||
                desc->address_v > RACHIS_ADDRESS_CLAMP_TO_BORDER || desc->address_w > RACHIS_ADDRESS_CLAMP_TO_BORDER)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid sampler descriptor");
            }
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            return createToken(*device, TokenKind::sampler, output, error, [&] {
                return device->native.createSampler(
                    {.minFilter = static_cast<rachis::FilterMode>(desc->min_filter),
                     .magFilter = static_cast<rachis::FilterMode>(desc->mag_filter),
                     .mipFilter = static_cast<rachis::MipmapFilterMode>(desc->mip_filter),
                     .addressU = static_cast<rachis::AddressMode>(desc->address_u),
                     .addressV = static_cast<rachis::AddressMode>(desc->address_v),
                     .addressW = static_cast<rachis::AddressMode>(desc->address_w),
                     .label = optionalString(desc->label)});
            });
        });
    }

    rachis_status rachis_sampler_destroy(rachis_device* device, rachis_handle sampler, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::scoped_lock const lock(device->mutex);
            return destroyToken(*device, TokenKind::sampler, sampler, error, [&](std::uint64_t native) {
                return device->native.destroySampler(rachis::SamplerHandle{native});
            });
        });
    }

    rachis_status rachis_shader_module_create(rachis_device* device,
                                              const rachis_shader_module_desc* desc,
                                              rachis_handle* output,
                                              rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output shader is null");
            }
            *output = 0;
            if (desc == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
            }
            if (!validDesc(desc))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported shader descriptor");
            }
            const bool workgroupStage = desc->stage == RACHIS_SHADER_COMPUTE || desc->stage == RACHIS_SHADER_TASK ||
                                        desc->stage == RACHIS_SHADER_MESH;
            if ((desc->stage != RACHIS_SHADER_VERTEX && desc->stage != RACHIS_SHADER_PIXEL && !workgroupStage) ||
                !validHostByteSpan(desc->bytecode, desc->bytecode_size) ||
                !validArray(desc->bindings, desc->binding_count) ||
                !validArray(desc->push_constants, desc->push_constant_count) ||
                !validArray(desc->vertex_input_locations, desc->vertex_input_location_count) ||
                desc->vertex_input_location_count > RACHIS_MAX_VERTEX_ATTRIBUTES || desc->draw_index > 1 ||
                (workgroupStage && (desc->workgroup_x == 0 || desc->workgroup_y == 0 || desc->workgroup_z == 0)) ||
                (!workgroupStage && (desc->workgroup_x != 0 || desc->workgroup_y != 0 || desc->workgroup_z != 0)))
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid shader descriptor");
            }
            std::vector<rachis::ShaderBindingRequirement> bindings;
            bindings.reserve(desc->binding_count);
            for (std::uint32_t i = 0; i < desc->binding_count; ++i)
            {
                const auto& value = desc->bindings[i];
                if (value.type > RACHIS_BINDING_SAMPLER || value.count == 0)
                {
                    return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid shader binding requirement");
                }
                bindings.push_back({.set = value.set,
                                    .binding = value.binding,
                                    .type = static_cast<rachis::ShaderResourceType>(value.type),
                                    .count = value.count});
            }
            std::vector<rachis::ShaderPushConstantRequirement> pushConstants;
            pushConstants.reserve(desc->push_constant_count);
            for (std::uint32_t i = 0; i < desc->push_constant_count; ++i)
            {
                const auto& value = desc->push_constants[i];
                pushConstants.push_back({.offset = value.offset, .size = value.size});
            }
            const auto bytecode = std::span<const std::byte>{static_cast<const std::byte*>(desc->bytecode),
                                                             static_cast<std::size_t>(desc->bytecode_size)};
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            return createToken(*device, TokenKind::shaderModule, output, error, [&] {
                rachis::ShaderModuleDesc native{};
                native.stage = static_cast<rachis::ShaderStage>(desc->stage);
                native.bytecode = bytecode;
                native.entryPoint =
                    desc->entry_point == nullptr ? std::string_view{"main"} : optionalString(desc->entry_point);
                native.label = optionalString(desc->label);
                native.shaderInterface.bindings = bindings;
                native.shaderInterface.pushConstants = pushConstants;
                if (desc->vertex_input_location_count != 0)
                {
                    native.shaderInterface.vertexInputLocations =
                        std::span<const std::uint32_t>{desc->vertex_input_locations, desc->vertex_input_location_count};
                }
                if (workgroupStage)
                {
                    native.shaderInterface.workgroupSize =
                        rachis::ShaderWorkgroupSize{desc->workgroup_x, desc->workgroup_y, desc->workgroup_z};
                }
                native.shaderInterface.drawIndex = desc->draw_index != 0;
                return device->native.createShaderModule(native);
            });
        });
    }

    rachis_status rachis_shader_module_destroy(rachis_device* device, rachis_handle shader, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::scoped_lock const lock(device->mutex);
            return destroyToken(*device, TokenKind::shaderModule, shader, error, [&](std::uint64_t native) {
                return device->native.destroyShaderModule(rachis::ShaderModuleHandle{native});
            });
        });
    }

    rachis_status rachis_pipeline_layout_create(rachis_device* device,
                                                const rachis_pipeline_layout_desc* desc,
                                                rachis_handle* output,
                                                rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output pipeline layout is null");
            }
            *output = 0;
            if (desc == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
            }
            if (!validDesc(desc))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported pipeline layout descriptor");
            }
            if (!validArray(desc->binding_layouts, desc->binding_layout_count) ||
                !validArray(desc->push_constant_ranges, desc->push_constant_range_count) ||
                desc->binding_layout_count > rachis::maxBindingSets || desc->draw_index > 1)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid pipeline binding layouts");
            }
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            std::vector<rachis::BindingLayoutHandle> layouts;
            layouts.reserve(desc->binding_layout_count);
            for (std::uint32_t i = 0; i < desc->binding_layout_count; ++i)
            {
                const auto handle = desc->binding_layouts[i];
                if (handle == 0)
                {
                    layouts.push_back({});
                    continue;
                }
                const auto* layout = findToken(device->tokens, handle, TokenKind::bindingLayout);
                if (layout == nullptr)
                {
                    return invalidHandle(error);
                }
                layouts.push_back(rachis::BindingLayoutHandle{layout->native});
            }
            std::vector<rachis::PushConstantRange> pushRanges;
            pushRanges.reserve(desc->push_constant_range_count);
            for (std::uint32_t i = 0; i < desc->push_constant_range_count; ++i)
            {
                const auto& range = desc->push_constant_ranges[i];
                if (range.visibility == 0 || (range.visibility & ~RACHIS_SHADER_ALL) != 0)
                {
                    return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid push constant visibility");
                }
                pushRanges.push_back({.visibility = static_cast<rachis::ShaderStage>(range.visibility),
                                      .offset = range.offset,
                                      .size = range.size});
            }
            return createToken(*device, TokenKind::pipelineLayout, output, error, [&] {
                rachis::PipelineLayoutDesc native{};
                native.bindingLayouts = layouts;
                native.pushConstantRanges = pushRanges;
                native.drawIndex = desc->draw_index != 0;
                native.label = optionalString(desc->label);
                return device->native.createPipelineLayout(native);
            });
        });
    }

    rachis_status rachis_pipeline_layout_destroy(rachis_device* device, rachis_handle layout, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::scoped_lock const lock(device->mutex);
            return destroyToken(*device, TokenKind::pipelineLayout, layout, error, [&](std::uint64_t native) {
                return device->native.destroyPipelineLayout(rachis::PipelineLayoutHandle{native});
            });
        });
    }

    rachis_status rachis_compute_pipeline_create(rachis_device* device,
                                                 const rachis_compute_pipeline_desc* desc,
                                                 rachis_handle* output,
                                                 rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output pipeline is null");
            }
            *output = 0;
            if (desc == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
            }
            if (!validDesc(desc))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported compute pipeline descriptor");
            }
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            const auto* layout = findToken(device->tokens, desc->layout, TokenKind::pipelineLayout);
            const auto* shader = findToken(device->tokens, desc->shader, TokenKind::shaderModule);
            if (layout == nullptr || shader == nullptr)
            {
                return invalidHandle(error);
            }
            const auto nativeLayout = rachis::PipelineLayoutHandle{layout->native};
            const auto nativeShader = rachis::ShaderModuleHandle{shader->native};
            return createToken(*device, TokenKind::pipeline, output, error, [&] {
                rachis::ComputePipelineDesc native{};
                native.layout = nativeLayout;
                native.computeShader = nativeShader;
                native.label = optionalString(desc->label);
                native.requiredSubgroupSize = desc->required_subgroup_size;
                return device->native.createComputePipeline(native);
            });
        });
    }

    rachis_status rachis_graphics_pipeline_create(rachis_device* device,
                                                  const rachis_graphics_pipeline_desc* desc,
                                                  rachis_handle* output,
                                                  rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output pipeline is null");
            }
            *output = 0;
            if (desc == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
            }
            if (!validDesc(desc))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported graphics pipeline descriptor");
            }
            if (!validArray(desc->color_targets, desc->color_target_count) || desc->color_target_count == 0 ||
                desc->color_target_count > 8 || desc->topology > RACHIS_TOPOLOGY_TRIANGLE_STRIP ||
                desc->cull_mode > RACHIS_CULL_BACK)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid graphics pipeline descriptor");
            }
            std::vector<rachis::ColorTargetState> colorTargets;
            colorTargets.reserve(desc->color_target_count);
            for (std::uint32_t i = 0; i < desc->color_target_count; ++i)
            {
                const auto& target = desc->color_targets[i];
                const auto format = nativeFormat(target.format);
                if (!format || target.write_mask > RACHIS_COLOR_WRITE_ALL || target.blend_enabled > 1 ||
                    target.color_blend.source_factor > RACHIS_BLEND_ONE_MINUS_SOURCE1_ALPHA ||
                    target.color_blend.destination_factor > RACHIS_BLEND_ONE_MINUS_SOURCE1_ALPHA ||
                    target.color_blend.operation > RACHIS_BLEND_OP_MAXIMUM ||
                    target.alpha_blend.source_factor > RACHIS_BLEND_ONE_MINUS_SOURCE1_ALPHA ||
                    target.alpha_blend.destination_factor > RACHIS_BLEND_ONE_MINUS_SOURCE1_ALPHA ||
                    target.alpha_blend.operation > RACHIS_BLEND_OP_MAXIMUM)
                {
                    return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid graphics color target");
                }
                colorTargets.push_back(
                    {.format = *format,
                     .writeMask = static_cast<rachis::ColorWriteMask>(target.write_mask),
                     .blend = {
                         .enabled = target.blend_enabled != 0,
                         .color = {.sourceFactor = static_cast<rachis::BlendFactor>(target.color_blend.source_factor),
                                   .destinationFactor =
                                       static_cast<rachis::BlendFactor>(target.color_blend.destination_factor),
                                   .operation = static_cast<rachis::BlendOperation>(target.color_blend.operation)},
                         .alpha = {.sourceFactor = static_cast<rachis::BlendFactor>(target.alpha_blend.source_factor),
                                   .destinationFactor =
                                       static_cast<rachis::BlendFactor>(target.alpha_blend.destination_factor),
                                   .operation = static_cast<rachis::BlendOperation>(target.alpha_blend.operation)}}});
            }
            rachis::DepthStencilState depthStencil{};
            if (desc->depth_stencil != nullptr)
            {
                const auto& depth = *desc->depth_stencil;
                const auto format = depth.format == RACHIS_FORMAT_UNKNOWN ? std::optional{rachis::Format::unknown}
                                                                          : nativeFormat(depth.format);
                if (!format ||
                    (depth.format != RACHIS_FORMAT_UNKNOWN && depth.format != RACHIS_FORMAT_D32_FLOAT &&
                     depth.format != RACHIS_FORMAT_D24_UNORM_S8_UINT) ||
                    (depth.format == RACHIS_FORMAT_UNKNOWN &&
                     (depth.depth_test_enabled != 0 || depth.depth_write_enabled != 0)) ||
                    depth.depth_test_enabled > 1 || depth.depth_write_enabled > 1 ||
                    depth.depth_compare > RACHIS_COMPARE_ALWAYS)
                {
                    return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid graphics depth state");
                }
                depthStencil.format = *format;
                depthStencil.depthTestEnabled = depth.depth_test_enabled != 0;
                depthStencil.depthWriteEnabled = depth.depth_write_enabled != 0;
                depthStencil.depthCompare = static_cast<rachis::CompareOperation>(depth.depth_compare);
            }
            if (!validArray(desc->vertex_buffers, desc->vertex_buffer_count) ||
                desc->vertex_buffer_count > RACHIS_MAX_VERTEX_BUFFERS ||
                desc->strip_index_format > RACHIS_STRIP_INDEX_UINT32)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid graphics vertex input");
            }
            std::vector<std::vector<rachis::VertexAttribute>> vertexAttributes(desc->vertex_buffer_count);
            std::vector<rachis::VertexBufferLayout> vertexBuffers;
            vertexBuffers.reserve(desc->vertex_buffer_count);
            for (std::uint32_t i = 0; i < desc->vertex_buffer_count; ++i)
            {
                const auto& layout = desc->vertex_buffers[i];
                if (!validArray(layout.attributes, layout.attribute_count) ||
                    layout.attribute_count > RACHIS_MAX_VERTEX_ATTRIBUTES ||
                    layout.step_mode > RACHIS_VERTEX_STEP_INSTANCE)
                {
                    return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid graphics vertex buffer layout");
                }
                auto& attributes = vertexAttributes[i];
                attributes.reserve(layout.attribute_count);
                for (std::uint32_t j = 0; j < layout.attribute_count; ++j)
                {
                    const auto& attribute = layout.attributes[j];
                    if (attribute.format > RACHIS_VERTEX_FORMAT_FLOAT16X4)
                    {
                        return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid graphics vertex attribute format");
                    }
                    attributes.push_back({.format = static_cast<rachis::VertexFormat>(attribute.format),
                                          .offset = attribute.offset,
                                          .shaderLocation = attribute.shader_location});
                }
                vertexBuffers.push_back(
                    {.stride = layout.stride,
                     .stepMode = static_cast<rachis::VertexStepMode>(layout.step_mode),
                     .instanceStepRate = layout.instance_step_rate == 0 ? 1U : layout.instance_step_rate,
                     .attributes = attributes});
            }
            const rachis::StripIndexFormat stripIndexFormat =
                desc->strip_index_format == RACHIS_STRIP_INDEX_NONE
                    ? rachis::StripIndexFormat{}
                    : rachis::StripIndexFormat{static_cast<rachis::IndexFormat>(desc->strip_index_format - 1U)};
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            const auto* layout = findToken(device->tokens, desc->layout, TokenKind::pipelineLayout);
            const auto* pixel = findToken(device->tokens, desc->pixel_shader, TokenKind::shaderModule);
            const auto findStage = [&](rachis_handle handle, rachis::ShaderModuleHandle& native) {
                if (handle == 0)
                {
                    return true;
                }
                const auto* shader = findToken(device->tokens, handle, TokenKind::shaderModule);
                if (shader == nullptr)
                {
                    return false;
                }
                native = rachis::ShaderModuleHandle{shader->native};
                return true;
            };
            rachis::ShaderModuleHandle nativeVertex;
            rachis::ShaderModuleHandle nativeTask;
            rachis::ShaderModuleHandle nativeMesh;
            if (layout == nullptr || pixel == nullptr || (desc->vertex_shader == 0 && desc->mesh_shader == 0) ||
                !findStage(desc->vertex_shader, nativeVertex) || !findStage(desc->task_shader, nativeTask) ||
                !findStage(desc->mesh_shader, nativeMesh))
            {
                return invalidHandle(error);
            }
            const auto nativeLayout = rachis::PipelineLayoutHandle{layout->native};
            const auto nativePixel = rachis::ShaderModuleHandle{pixel->native};
            return createToken(*device, TokenKind::pipeline, output, error, [&] {
                return device->native.createGraphicsPipeline({
                    .layout = nativeLayout,
                    .vertexShader = nativeVertex,
                    .pixelShader = nativePixel,
                    .taskShader = nativeTask,
                    .meshShader = nativeMesh,
                    .vertexBuffers = vertexBuffers,
                    .topology = static_cast<rachis::PrimitiveTopology>(desc->topology),
                    .stripIndexFormat = stripIndexFormat,
                    .cullMode = static_cast<rachis::CullMode>(desc->cull_mode),
                    .colorTargets = colorTargets,
                    .depthStencil = depthStencil,
                    .label = optionalString(desc->label),
                });
            });
        });
    }

    rachis_status rachis_pipeline_destroy(rachis_device* device, rachis_handle pipeline, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::scoped_lock const lock(device->mutex);
            return destroyToken(*device, TokenKind::pipeline, pipeline, error, [&](std::uint64_t native) {
                return device->native.destroyPipeline(rachis::PipelineHandle{native});
            });
        });
    }

    rachis_status rachis_command_bind_compute_pipeline(rachis_device* device,
                                                       rachis_handle command,
                                                       rachis_handle pipeline,
                                                       rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            CommandUseGuard commandUse;
            rachis::CommandBuffer* commandBuffer = nullptr;
            if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            const auto* token = findToken(device->tokens, pipeline, TokenKind::pipeline);
            if (token == nullptr)
            {
                return invalidHandle(error);
            }
            return result(error,
                          device->native.bindComputePipeline(*commandBuffer, rachis::PipelineHandle{token->native}));
        });
    }

    rachis_status rachis_command_bind_graphics_pipeline(rachis_device* device,
                                                        rachis_handle command,
                                                        rachis_handle pipeline,
                                                        rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            CommandUseGuard commandUse;
            rachis::CommandBuffer* commandBuffer = nullptr;
            if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            const auto* token = findToken(device->tokens, pipeline, TokenKind::pipeline);
            if (token == nullptr)
            {
                return invalidHandle(error);
            }
            return result(error,
                          device->native.bindGraphicsPipeline(*commandBuffer, rachis::PipelineHandle{token->native}));
        });
    }

    rachis_status rachis_command_bind_graphics_set(
        rachis_device* device, rachis_handle command, std::uint32_t setIndex, rachis_handle set, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            CommandUseGuard commandUse;
            rachis::CommandBuffer* commandBuffer = nullptr;
            if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            const auto* token = findToken(device->tokens, set, TokenKind::bindingSet);
            if (token == nullptr)
            {
                return invalidHandle(error);
            }
            return result(error,
                          device->native.bindGraphicsBindingSet(
                              *commandBuffer, setIndex, rachis::BindingSetHandle{token->native}));
        });
    }

    rachis_status rachis_command_set_graphics_push_constants(rachis_device* device,
                                                             rachis_handle command,
                                                             std::uint32_t visibility,
                                                             std::uint32_t offset,
                                                             const void* data,
                                                             std::uint32_t size,
                                                             rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || (size != 0 && !validHostByteSpan(data, size)) || visibility == 0 ||
                (visibility & ~RACHIS_SHADER_ALL) != 0)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid push constant arguments");
            }
            if ((visibility & RACHIS_SHADER_COMPUTE) != 0)
            {
                return fail(error,
                            RACHIS_STATUS_INVALID_ARGUMENT,
                            "graphics push constants cannot target the compute stage; use "
                            "rachis_command_set_compute_push_constants");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            CommandUseGuard commandUse;
            rachis::CommandBuffer* commandBuffer = nullptr;
            if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            const auto bytes = size == 0 ? std::span<const std::byte>{}
                                         : std::span<const std::byte>{static_cast<const std::byte*>(data), size};
            return result(error,
                          device->native.setGraphicsPushConstants(
                              *commandBuffer, static_cast<rachis::ShaderStage>(visibility), offset, bytes));
        });
    }

    rachis_status rachis_command_set_viewport(rachis_device* device,
                                              rachis_handle command,
                                              const rachis_viewport* viewport,
                                              rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || viewport == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or viewport is null");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            CommandUseGuard commandUse;
            rachis::CommandBuffer* commandBuffer = nullptr;
            if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            return result(error,
                          device->native.setViewport(*commandBuffer,
                                                     {.x = viewport->x,
                                                      .y = viewport->y,
                                                      .width = viewport->width,
                                                      .height = viewport->height,
                                                      .minDepth = viewport->min_depth,
                                                      .maxDepth = viewport->max_depth}));
        });
    }

    rachis_status rachis_command_set_scissor(rachis_device* device,
                                             rachis_handle command,
                                             const rachis_scissor* scissor,
                                             rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || scissor == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or scissor is null");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            CommandUseGuard commandUse;
            rachis::CommandBuffer* commandBuffer = nullptr;
            if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            return result(error,
                          device->native.setScissor(
                              *commandBuffer,
                              {.x = scissor->x, .y = scissor->y, .width = scissor->width, .height = scissor->height}));
        });
    }

    rachis_status rachis_command_draw(rachis_device* device,
                                      rachis_handle command,
                                      std::uint32_t vertexCount,
                                      std::uint32_t instanceCount,
                                      std::uint32_t firstVertex,
                                      std::uint32_t firstInstance,
                                      rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            CommandUseGuard commandUse;
            rachis::CommandBuffer* commandBuffer = nullptr;
            if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            return result(error,
                          device->native.draw(*commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance));
        });
    }

    rachis_status rachis_command_bind_vertex_buffers(rachis_device* device,
                                                     rachis_handle command,
                                                     std::uint32_t firstSlot,
                                                     const rachis_vertex_buffer_binding* bindings,
                                                     std::uint32_t bindingCount,
                                                     rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || bindings == nullptr || bindingCount == 0 ||
                firstSlot >= RACHIS_MAX_VERTEX_BUFFERS || bindingCount > RACHIS_MAX_VERTEX_BUFFERS - firstSlot)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid vertex buffer bindings");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            CommandUseGuard commandUse;
            rachis::CommandBuffer* commandBuffer = nullptr;
            if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            std::array<rachis::VertexBufferBinding, RACHIS_MAX_VERTEX_BUFFERS> nativeBindings{};
            for (std::uint32_t i = 0; i < bindingCount; ++i)
            {
                const auto* token = findToken(device->tokens, bindings[i].buffer, TokenKind::buffer);
                if (token == nullptr)
                {
                    return invalidHandle(error);
                }
                nativeBindings[i] = {.buffer = rachis::BufferHandle{token->native}, .offset = bindings[i].offset};
            }
            return result(error,
                          device->native.bindVertexBuffers(
                              *commandBuffer, firstSlot, std::span{nativeBindings.data(), bindingCount}));
        });
    }

    rachis_status rachis_command_bind_index_buffer(rachis_device* device,
                                                   rachis_handle command,
                                                   rachis_handle buffer,
                                                   std::uint32_t indexFormat,
                                                   std::uint64_t offset,
                                                   rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || indexFormat > RACHIS_INDEX_FORMAT_UINT32)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null or the index format is invalid");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            CommandUseGuard commandUse;
            rachis::CommandBuffer* commandBuffer = nullptr;
            if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            const auto* token = findToken(device->tokens, buffer, TokenKind::buffer);
            if (token == nullptr)
            {
                return invalidHandle(error);
            }
            return result(error,
                          device->native.bindIndexBuffer(*commandBuffer,
                                                         rachis::BufferHandle{token->native},
                                                         static_cast<rachis::IndexFormat>(indexFormat),
                                                         offset));
        });
    }

    rachis_status rachis_command_draw_indexed(rachis_device* device,
                                              rachis_handle command,
                                              std::uint32_t indexCount,
                                              std::uint32_t instanceCount,
                                              std::uint32_t firstIndex,
                                              std::int32_t baseVertex,
                                              std::uint32_t firstInstance,
                                              rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            CommandUseGuard commandUse;
            rachis::CommandBuffer* commandBuffer = nullptr;
            if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            return result(error,
                          device->native.drawIndexed(
                              *commandBuffer, indexCount, instanceCount, firstIndex, baseVertex, firstInstance));
        });
    }

    rachis_status rachis_command_draw_indirect(rachis_device* device,
                                               rachis_handle command,
                                               const rachis_indirect_draw_desc* desc,
                                               rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status { return drawIndirect(device, command, desc, error, false); });
    }

    rachis_status rachis_command_draw_indexed_indirect(rachis_device* device,
                                                       rachis_handle command,
                                                       const rachis_indirect_draw_desc* desc,
                                                       rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status { return drawIndirect(device, command, desc, error, true); });
    }

    rachis_status rachis_command_dispatch_mesh(rachis_device* device,
                                               rachis_handle command,
                                               std::uint32_t x,
                                               std::uint32_t y,
                                               std::uint32_t z,
                                               rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            return recordCommand(*device, command, error, [&](rachis::CommandBuffer& commandBuffer) {
                return result(error, device->native.dispatchMesh(commandBuffer, x, y, z));
            });
        });
    }

    rachis_status rachis_command_dispatch_mesh_indirect(rachis_device* device,
                                                        rachis_handle command,
                                                        const rachis_indirect_mesh_dispatch_desc* desc,
                                                        rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (const auto rejected = rejectIndirectDesc(device, desc, error))
            {
                return *rejected;
            }
            if (desc->draw_index_prefix > 1)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid indirect mesh dispatch descriptor");
            }
            return recordCommand(*device, command, error, [&](rachis::CommandBuffer& commandBuffer) {
                const auto buffers = findIndirectBuffers(*device, desc->argument_buffer, desc->count_buffer);
                if (!buffers)
                {
                    return invalidHandle(error);
                }
                return result(error,
                              device->native.dispatchMeshIndirect(commandBuffer,
                                                                  {
                                                                      .argumentBuffer = buffers->arguments,
                                                                      .argumentOffset = desc->argument_offset,
                                                                      .maxDispatchCount = desc->max_dispatch_count,
                                                                      .stride = desc->stride,
                                                                      .countBuffer = buffers->count,
                                                                      .countOffset = desc->count_offset,
                                                                      .drawIndexPrefix = desc->draw_index_prefix != 0,
                                                                  }));
            });
        });
    }

    rachis_status rachis_command_bind_compute_set(
        rachis_device* device, rachis_handle command, std::uint32_t setIndex, rachis_handle set, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            CommandUseGuard commandUse;
            rachis::CommandBuffer* commandBuffer = nullptr;
            if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            const auto* token = findToken(device->tokens, set, TokenKind::bindingSet);
            if (token == nullptr)
            {
                return invalidHandle(error);
            }
            return result(error,
                          device->native.bindComputeBindingSet(
                              *commandBuffer, setIndex, rachis::BindingSetHandle{token->native}));
        });
    }

    rachis_status rachis_command_set_compute_push_constants(rachis_device* device,
                                                            rachis_handle command,
                                                            std::uint32_t offset,
                                                            const void* data,
                                                            std::uint32_t size,
                                                            rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || (size != 0 && !validHostByteSpan(data, size)))
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid push constant arguments");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            CommandUseGuard commandUse;
            rachis::CommandBuffer* commandBuffer = nullptr;
            if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            const auto bytes = size == 0 ? std::span<const std::byte>{}
                                         : std::span<const std::byte>{static_cast<const std::byte*>(data), size};
            return result(error, device->native.setComputePushConstants(*commandBuffer, offset, bytes));
        });
    }

    rachis_status rachis_command_dispatch(rachis_device* device,
                                          rachis_handle command,
                                          std::uint32_t x,
                                          std::uint32_t y,
                                          std::uint32_t z,
                                          rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            CommandUseGuard commandUse;
            rachis::CommandBuffer* commandBuffer = nullptr;
            if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            return result(error, device->native.dispatch(*commandBuffer, x, y, z));
        });
    }

    rachis_status rachis_command_dispatch_indirect(rachis_device* device,
                                                   rachis_handle command,
                                                   const rachis_indirect_dispatch_desc* desc,
                                                   rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (const auto rejected = rejectIndirectDesc(device, desc, error))
            {
                return *rejected;
            }
            return recordCommand(*device, command, error, [&](rachis::CommandBuffer& commandBuffer) {
                const auto buffers = findIndirectBuffers(*device, desc->argument_buffer, desc->count_buffer);
                if (!buffers)
                {
                    return invalidHandle(error);
                }
                return result(error,
                              device->native.dispatchIndirect(commandBuffer,
                                                              rachis::IndirectDispatchDesc{
                                                                  .argumentBuffer = buffers->arguments,
                                                                  .argumentOffset = desc->argument_offset,
                                                                  .maxDispatchCount = desc->max_dispatch_count,
                                                                  .stride = desc->stride,
                                                                  .countBuffer = buffers->count,
                                                                  .countOffset = desc->count_offset,
                                                              }));
            });
        });
    }

    rachis_status rachis_timestamp_pool_create(rachis_device* device,
                                               const rachis_timestamp_pool_desc* desc,
                                               rachis_handle* output,
                                               rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output query pool is null");
            }
            *output = 0;
            if (desc == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
            }
            if (!validDesc(desc))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported query pool descriptor");
            }
            if (desc->queue_type > RACHIS_QUEUE_COPY)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid query queue type");
            }
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            return createToken(*device, TokenKind::timestampPool, output, error, [&] {
                return device->native.createTimestampQueryPool(
                    {.queueType = static_cast<rachis::QueueType>(desc->queue_type),
                     .queryCount = desc->query_count,
                     .label = optionalString(desc->label)});
            });
        });
    }

    rachis_status rachis_query_pool_destroy(rachis_device* device, rachis_handle pool, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::scoped_lock const lock(device->mutex);
            return destroyToken(*device, TokenKind::timestampPool, pool, error, [&](std::uint64_t native) {
                return device->native.destroyQueryPool(rachis::QueryPoolHandle{native});
            });
        });
    }

    rachis_status rachis_command_reset_queries(rachis_device* device,
                                               rachis_handle command,
                                               rachis_handle pool,
                                               std::uint32_t firstQuery,
                                               std::uint32_t queryCount,
                                               rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            CommandUseGuard commandUse;
            rachis::CommandBuffer* commandBuffer = nullptr;
            if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            const auto* token = findToken(device->tokens, pool, TokenKind::timestampPool);
            if (token == nullptr)
            {
                return invalidHandle(error);
            }
            return result(error,
                          device->native.resetQueryPool(
                              *commandBuffer, rachis::QueryPoolHandle{token->native}, firstQuery, queryCount));
        });
    }

    rachis_status rachis_command_write_timestamp(
        rachis_device* device, rachis_handle command, rachis_handle pool, std::uint32_t queryIndex, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            CommandUseGuard commandUse;
            rachis::CommandBuffer* commandBuffer = nullptr;
            if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            const auto* token = findToken(device->tokens, pool, TokenKind::timestampPool);
            if (token == nullptr)
            {
                return invalidHandle(error);
            }
            return result(
                error,
                device->native.writeTimestamp(*commandBuffer, rachis::QueryPoolHandle{token->native}, queryIndex));
        });
    }

    rachis_status rachis_command_resolve_timestamps(rachis_device* device,
                                                    rachis_handle command,
                                                    rachis_handle pool,
                                                    std::uint32_t firstQuery,
                                                    std::uint32_t queryCount,
                                                    rachis_handle destination,
                                                    std::uint64_t destinationOffset,
                                                    rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::shared_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            CommandUseGuard commandUse;
            rachis::CommandBuffer* commandBuffer = nullptr;
            if (const auto status = acquireCommandForRecording(*device, command, commandUse, commandBuffer, error);
                status != RACHIS_STATUS_OK)
            {
                return status;
            }
            const auto* query = findToken(device->tokens, pool, TokenKind::timestampPool);
            const auto* buffer = findToken(device->tokens, destination, TokenKind::buffer);
            if (query == nullptr || buffer == nullptr)
            {
                return invalidHandle(error);
            }
            return result(error,
                          device->native.resolveTimestamps(*commandBuffer,
                                                           {.queryPool = rachis::QueryPoolHandle{query->native},
                                                            .firstQuery = firstQuery,
                                                            .queryCount = queryCount,
                                                            .destination = rachis::BufferHandle{buffer->native},
                                                            .destinationOffset = destinationOffset}));
        });
    }

    rachis_status rachis_surface_create(rachis_device* device,
                                        const rachis_surface_desc* desc,
                                        rachis_handle* output,
                                        rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output surface is null");
            }
            *output = 0;
            if (desc == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
            }
            if (!validDesc(desc))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported surface descriptor layout");
            }
            if (desc->kind > RACHIS_SURFACE_CA_METAL_LAYER)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid surface kind");
            }
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            return createToken(*device, TokenKind::surface, output, error, [&] {
                return device->native.createSurface(rachis::SurfaceDesc{
                    .kind = static_cast<rachis::SurfaceKind>(desc->kind),
                    .windowHandle = desc->window_handle,
                    .displayHandle = desc->display_handle,
                    .label = optionalString(desc->label),
                });
            });
        });
    }

    rachis_status rachis_surface_query_capabilities(rachis_device* device,
                                                    rachis_handle surface,
                                                    rachis_surface_capabilities* output,
                                                    rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output capabilities are null");
            }
            if (output->struct_size != sizeof(*output))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "invalid surface capability output layout");
            }
            std::shared_lock const lock(device->mutex);
            const auto* token = findToken(device->tokens, surface, TokenKind::surface);
            if (token == nullptr)
            {
                return invalidHandle(error);
            }
            const auto capabilities = device->native.querySurfaceCapabilities(rachis::SurfaceHandle{token->native});
            if (!capabilities)
            {
                return result(error, capabilities.error());
            }
            output->min_image_count = capabilities->minImageCount;
            output->max_image_count = capabilities->maxImageCount;
            output->min_width = capabilities->minWidth;
            output->min_height = capabilities->minHeight;
            output->max_width = capabilities->maxWidth;
            output->max_height = capabilities->maxHeight;
            output->preferred_width = capabilities->preferredWidth;
            output->preferred_height = capabilities->preferredHeight;
            output->usages = static_cast<std::uint32_t>(capabilities->usages) & knownTextureUsage;
            output->present_modes = 0;
            if (rachis::supportsPresentMode(capabilities->presentModes, rachis::PresentMode::fifo))
            {
                output->present_modes |= RACHIS_PRESENT_MODE_FIFO_BIT;
            }
            if (rachis::supportsPresentMode(capabilities->presentModes, rachis::PresentMode::immediate))
            {
                output->present_modes |= RACHIS_PRESENT_MODE_IMMEDIATE_BIT;
            }
            output->format_count = 0;
            std::memset(output->formats, 0, sizeof(output->formats));
            for (const auto& format : capabilities->formats)
            {
                const auto value = formatOf(format.format);
                if (value == RACHIS_FORMAT_UNKNOWN)
                {
                    continue;
                }
                if (output->format_count == RACHIS_MAX_SURFACE_FORMATS)
                {
                    break;
                }
                output->formats[output->format_count++] =
                    rachis_surface_format{value, static_cast<std::uint32_t>(format.colorSpace)};
            }
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_surface_destroy(rachis_device* device, rachis_handle surface, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            std::scoped_lock const lock(device->mutex);
            const auto* token = findToken(device->tokens, surface, TokenKind::surface);
            if (token == nullptr)
            {
                return invalidHandle(error);
            }
            const auto destroyed = device->native.destroySurface(rachis::SurfaceHandle{token->native});
            if (!destroyed)
            {
                return result(error, destroyed.error());
            }
            device->tokens.erase(surface);
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_swapchain_create(rachis_device* device,
                                          const rachis_swapchain_desc* desc,
                                          rachis_handle* output,
                                          rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output swapchain is null");
            }
            *output = 0;
            if (desc == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "descriptor is null");
            }
            if (!validDesc(desc))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "unsupported swapchain descriptor layout");
            }
            const auto format = nativeFormat(desc->format);
            if (!format || desc->present_mode > RACHIS_PRESENT_IMMEDIATE ||
                desc->color_space > RACHIS_COLOR_SPACE_HDR10_ST2084 || (desc->usage & ~knownTextureUsage) != 0)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "invalid swapchain enum value");
            }
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            const auto* surface = findToken(device->tokens, desc->surface, TokenKind::surface);
            if (surface == nullptr)
            {
                return invalidHandle(error);
            }
            const auto nativeSurface = rachis::SurfaceHandle{surface->native};
            const auto id = device->swapchains.emplace(activeTokenSource(),
                                                       SwapchainEntry{.native = {}, .gate = {}, .borrowed = {}});
            if (id == 0)
            {
                return fail(error, RACHIS_STATUS_RESOURCE_EXHAUSTED, "C handle space exhausted");
            }
            try
            {
                device->swapchains.find(id)->gate = std::make_shared<std::mutex>();
                auto created = device->native.createSwapchain(rachis::SwapchainDesc{
                    .surface = nativeSurface,
                    .width = desc->width,
                    .height = desc->height,
                    .imageCount = desc->image_count,
                    .format = *format,
                    .colorSpace = static_cast<rachis::SurfaceColorSpace>(desc->color_space),
                    .usage = desc->usage == 0 ? rachis::TextureUsage::renderTarget
                                              : static_cast<rachis::TextureUsage>(desc->usage),
                    .presentMode = static_cast<rachis::PresentMode>(desc->present_mode),
                    .label = optionalString(desc->label),
                });
                if (!created)
                {
                    device->swapchains.erase(id);
                    return result(error, created.error());
                }
                device->swapchains.find(id)->native = *created;
                *output = id;
                return RACHIS_STATUS_OK;
            }
            catch (...)
            {
                device->swapchains.erase(id);
                throw;
            }
        });
    }

    rachis_status rachis_swapchain_destroy(rachis_device* device, rachis_handle swapchain, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            const auto gate = lockSwapchainGate(*device, swapchain);
            std::scoped_lock const lock(device->mutex);
            auto* const entry = findMutableSwapchain(*device, swapchain);
            if (entry == nullptr)
            {
                return invalidHandle(error);
            }
            const auto destroyed = device->native.destroySwapchain(entry->native);
            if (!destroyed)
            {
                return result(error, destroyed.error());
            }
            entry->borrowed.erase(device->tokens);
            device->swapchains.erase(swapchain);
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_swapchain_resize(
        rachis_device* device, rachis_handle swapchain, std::uint32_t width, std::uint32_t height, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            const auto gate = lockSwapchainGate(*device, swapchain);
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            auto* const entry = findMutableSwapchain(*device, swapchain);
            if (entry == nullptr)
            {
                return invalidHandle(error);
            }
            const auto resized = device->native.resizeSwapchain(entry->native, width, height);
            if (!resized)
            {
                return result(error, resized.error());
            }
            entry->borrowed.erase(device->tokens);
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_swapchain_acquire(rachis_device* device,
                                           rachis_handle swapchain,
                                           rachis_swapchain_image* output,
                                           rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or output image is null");
            }
            if (output->struct_size != sizeof(*output))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "invalid swapchain image output layout");
            }
            output->texture = 0;
            output->render_target_view = 0;
            const auto gate = lockSwapchainGate(*device, swapchain);
            std::scoped_lock const lock(device->mutex);
            if (const auto lost = rejectLostDevice(*device, error))
            {
                return *lost;
            }
            auto* const swapchainEntry = findMutableSwapchain(*device, swapchain);
            if (swapchainEntry == nullptr)
            {
                return invalidHandle(error);
            }
            const auto texture = reserveToken(device->tokens, TokenKind::borrowedTexture, swapchain);
            if (texture == 0)
            {
                return fail(error, RACHIS_STATUS_RESOURCE_EXHAUSTED, "C handle space exhausted");
            }
            rachis_handle view = 0;
            try
            {
                view = reserveToken(device->tokens, TokenKind::borrowedView, swapchain);
            }
            catch (...)
            {
                device->tokens.erase(texture);
                throw;
            }
            if (view == 0)
            {
                device->tokens.erase(texture);
                return fail(error, RACHIS_STATUS_RESOURCE_EXHAUSTED, "C handle space exhausted");
            }
            std::optional<rachis::Result<rachis::AcquiredSwapchainImage>> acquired;
            try
            {
                acquired.emplace(device->native.acquireNextImage(swapchainEntry->native));
            }
            catch (...)
            {
                device->tokens.erase(texture);
                device->tokens.erase(view);
                throw;
            }
            if (!*acquired)
            {
                device->tokens.erase(texture);
                device->tokens.erase(view);
                return result(error, acquired->error());
            }
            swapchainEntry->borrowed.erase(device->tokens);
            device->tokens.find(texture)->native = (*acquired)->texture.identity;
            device->tokens.find(view)->native = (*acquired)->renderTargetView.identity;
            swapchainEntry->borrowed.publish(texture, view);
            output->texture = texture;
            output->render_target_view = view;
            output->image_index = (*acquired)->imageIndex;
            output->width = (*acquired)->width;
            output->height = (*acquired)->height;
            output->format = formatOf((*acquired)->format);
            output->color_space = static_cast<std::uint32_t>((*acquired)->colorSpace);
            output->initial_layout = static_cast<std::uint32_t>((*acquired)->initialLayout);
            output->initial_stages = static_cast<std::uint32_t>((*acquired)->initialStages);
            output->initial_access = static_cast<std::uint32_t>((*acquired)->initialAccess);
            output->required_final_layout = static_cast<std::uint32_t>((*acquired)->requiredFinalLayout);
            output->required_final_stages = static_cast<std::uint32_t>((*acquired)->requiredFinalStages);
            output->required_final_access = static_cast<std::uint32_t>((*acquired)->requiredFinalAccess);
            output->contents_defined = (*acquired)->contentsDefined ? 1U : 0U;
            output->generation = (*acquired)->generation;
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_swapchain_query_presentation_progress(rachis_device* device,
                                                               rachis_handle swapchain,
                                                               rachis_swapchain_presentation_progress* output,
                                                               rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || output == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or presentation progress is null");
            }
            if (output->struct_size != sizeof(*output))
            {
                return fail(error, RACHIS_STATUS_ABI_MISMATCH, "presentation progress size mismatch");
            }
            *output = {.struct_size = sizeof(*output),
                       .supported = 0,
                       .generation = 0,
                       .submitted_sequence = 0,
                       .displayed_sequence = 0};
            const auto gate = lockSwapchainGate(*device, swapchain);
            rachis::SwapchainHandle native;
            {
                std::shared_lock const lock(device->mutex);
                if (const auto lost = rejectLostDevice(*device, error))
                {
                    return *lost;
                }
                const auto* const entry = findSwapchain(*device, swapchain);
                if (entry == nullptr)
                {
                    return invalidHandle(error);
                }
                native = entry->native;
            }
            const auto timing = device->native.querySwapchainFrameTiming(native);
            if (!timing)
            {
                return result(error, timing.error());
            }
            output->supported = timing->presentationSupported ? 1U : 0U;
            output->generation = timing->presentationGeneration;
            output->submitted_sequence = timing->submittedPresentationSequence;
            output->displayed_sequence = timing->displayedPresentationSequence;
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_swapchain_cancel(rachis_device* device, rachis_handle swapchain, rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device is null");
            }
            const auto gate = lockSwapchainGate(*device, swapchain);
            std::scoped_lock const lock(device->mutex);
            auto* const entry = findMutableSwapchain(*device, swapchain);
            if (entry == nullptr)
            {
                return invalidHandle(error);
            }
            const auto canceled = device->native.cancelSwapchainImage(entry->native);
            if (!canceled)
            {
                return result(error, canceled.error());
            }
            entry->borrowed.erase(device->tokens);
            return RACHIS_STATUS_OK;
        });
    }

    rachis_status rachis_swapchain_present(rachis_device* device,
                                           rachis_handle swapchain,
                                           rachis_handle renderComplete,
                                           std::uint32_t* presentStatus,
                                           rachis_error* error)
    {
        return boundary(error, [&]() -> rachis_status {
            if (device == nullptr || presentStatus == nullptr)
            {
                return fail(error, RACHIS_STATUS_INVALID_ARGUMENT, "device or present status is null");
            }
            const auto gate = lockSwapchainGate(*device, swapchain);
            rachis::SwapchainHandle native;
            rachis::TimelinePoint point;
            std::array<rachis::cabi::detail::CAbiSuspendedHandle, 2> borrowed{};
            {
                std::scoped_lock const lock(device->mutex);
                if (const auto lost = rejectLostDevice(*device, error))
                {
                    return *lost;
                }
                auto* const swapchainEntry = findMutableSwapchain(*device, swapchain);
                const auto* const timelineEntry = device->timelines.find(renderComplete);
                if (swapchainEntry == nullptr || timelineEntry == nullptr)
                {
                    return invalidHandle(error);
                }
                native = swapchainEntry->native;
                point = *timelineEntry;
                borrowed = swapchainEntry->borrowed.suspend(device->tokens);
            }
            const auto retireBorrowed = [&]() noexcept {
                if (auto* const swapchainEntry = findMutableSwapchain(*device, swapchain); swapchainEntry != nullptr)
                {
                    swapchainEntry->borrowed.eraseSuspended(device->tokens, borrowed);
                    return;
                }
                for (const auto& entry : borrowed)
                {
                    if (entry.handle != 0)
                    {
                        device->tokens.erase(entry.handle);
                    }
                }
            };
            const auto reconcileBorrowed = [&]() noexcept {
                if (device->native.hasAcquiredSwapchainImage(native))
                {
                    rachis::cabi::detail::CAbiBorrowedImageHandles::restoreSuspended(device->tokens, borrowed);
                    return;
                }
                retireBorrowed();
            };
            std::optional<rachis::Result<rachis::PresentStatus>> presented;
            try
            {
#ifdef RACHIS_TESTING
                invokeSwapchainPresentHook(swapchain);
#endif
                presented.emplace(device->native.present(native, point));
            }
            catch (...)
            {
                std::scoped_lock const lock(device->mutex);
                reconcileBorrowed();
                throw;
            }
            std::scoped_lock const lock(device->mutex);
            if (!*presented)
            {
                reconcileBorrowed();
                return result(error, presented->error());
            }
            *presentStatus = presentStatusOf(**presented);
            retireBorrowed();
            return RACHIS_STATUS_OK;
        });
    }
}

#ifdef RACHIS_TESTING
namespace rachis::cabi::detail
{
    void setTokenSourceOverrideForTesting(rachis::detail::MonotonicIdentitySource* overrideSource) noexcept
    {
        tokenSourceOverride.store(overrideSource, std::memory_order_release);
    }

    void setCommandRecordingHookForTesting(CommandRecordingHook hook, void* userData) noexcept
    {
        const std::scoped_lock lock(testHookMutex);
        commandRecordingHookState = CommandRecordingHookState{.hook = hook, .userData = userData};
    }

    void setSwapchainPresentHookForTesting(SwapchainPresentHook hook, void* userData) noexcept
    {
        const std::scoped_lock lock(testHookMutex);
        swapchainPresentHookState = SwapchainPresentHookState{.hook = hook, .userData = userData};
    }

    void setNativeCallHookForTesting(NativeCallHook hook, void* userData) noexcept
    {
        const std::scoped_lock lock(testHookMutex);
        nativeCallHookState = NativeCallHookState{.hook = hook, .userData = userData};
    }

    bool deviceLockAdmitsSharedCallersForTesting(rachis_device* device) noexcept
    {
        if (device == nullptr || !device->mutex.try_lock_shared())
        {
            return false;
        }
        device->mutex.unlock_shared();
        return true;
    }

    std::size_t liveTokenCountForTesting(rachis_device* device)
    {
        if (device == nullptr)
        {
            return 0;
        }
        std::shared_lock const lock(device->mutex);
        return device->tokens.size();
    }

    void emitDiagnosticForTesting(rachis_device* device, const rachis::DiagnosticMessage& message) noexcept
    {
        onDiagnostic(message, device);
    }
} // namespace rachis::cabi::detail
#endif
