#include "Internal/FillAndClearValidation.h"
#include "MetalInternal.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace rachis::metal
{

    namespace
    {

        constexpr std::uint32_t kMetalFillThreadgroupSize = 256;
        constexpr std::uint64_t kMetalFillWordsPerDispatch = std::uint64_t{65535} * kMetalFillThreadgroupSize;

        constexpr const char* kMetalFillAndClearSource = R"msl(
#include <metal_stdlib>
using namespace metal;

struct RachisFill
{
    device uint* destination;
    uint words;
    uint value;
};

struct RachisClearValue
{
    float4 f;
    uint4 u;
    int4 s;
};

[[kernel]] void rachisFillBuffer(constant RachisFill& fill [[buffer(0)]], uint index [[thread_position_in_grid]])
{
    if (index < fill.words)
    {
        fill.destination[index] = fill.value;
    }
}

#define RACHIS_CLEAR_KERNELS(SUFFIX, TYPE, VALUE)                                                                     \
    [[kernel]] void rachisClear1D##SUFFIX(texture1d<TYPE, access::write> target [[texture(0)]],                       \
                                          constant RachisClearValue& clear [[buffer(0)]],                              \
                                          uint3 position [[thread_position_in_grid]])                                  \
    {                                                                                                                  \
        if (position.x < target.get_width())                                                                           \
        {                                                                                                              \
            target.write(VALUE, position.x);                                                                           \
        }                                                                                                              \
    }                                                                                                                  \
    [[kernel]] void rachisClear1DArray##SUFFIX(texture1d_array<TYPE, access::write> target [[texture(0)]],            \
                                               constant RachisClearValue& clear [[buffer(0)]],                         \
                                               uint3 position [[thread_position_in_grid]])                             \
    {                                                                                                                  \
        if (position.x < target.get_width() && position.z < target.get_array_size())                                   \
        {                                                                                                              \
            target.write(VALUE, position.x, position.z);                                                               \
        }                                                                                                              \
    }                                                                                                                  \
    [[kernel]] void rachisClear2D##SUFFIX(texture2d<TYPE, access::write> target [[texture(0)]],                       \
                                          constant RachisClearValue& clear [[buffer(0)]],                              \
                                          uint3 position [[thread_position_in_grid]])                                  \
    {                                                                                                                  \
        if (position.x < target.get_width() && position.y < target.get_height())                                       \
        {                                                                                                              \
            target.write(VALUE, position.xy);                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    [[kernel]] void rachisClear2DArray##SUFFIX(texture2d_array<TYPE, access::write> target [[texture(0)]],            \
                                               constant RachisClearValue& clear [[buffer(0)]],                         \
                                               uint3 position [[thread_position_in_grid]])                             \
    {                                                                                                                  \
        if (position.x < target.get_width() && position.y < target.get_height() &&                                     \
            position.z < target.get_array_size())                                                                      \
        {                                                                                                              \
            target.write(VALUE, position.xy, position.z);                                                              \
        }                                                                                                              \
    }                                                                                                                  \
    [[kernel]] void rachisClear3D##SUFFIX(texture3d<TYPE, access::write> target [[texture(0)]],                       \
                                          constant RachisClearValue& clear [[buffer(0)]],                              \
                                          uint3 position [[thread_position_in_grid]])                                  \
    {                                                                                                                  \
        if (position.x < target.get_width() && position.y < target.get_height() &&                                     \
            position.z < target.get_depth())                                                                           \
        {                                                                                                              \
            target.write(VALUE, position);                                                                             \
        }                                                                                                              \
    }

RACHIS_CLEAR_KERNELS(Float, float, clear.f)
RACHIS_CLEAR_KERNELS(Uint, uint, clear.u)
RACHIS_CLEAR_KERNELS(Sint, int, clear.s)
)msl";

        struct MetalFillHeader final
        {
            std::uint64_t destination = 0;
            std::uint32_t words = 0;
            std::uint32_t value = 0;
        };

        struct MetalClearValueBlock final
        {
            std::array<float, 4> f{};
            std::array<std::uint32_t, 4> u{};
            std::array<std::int32_t, 4> s{};
        };

        constexpr std::array<const char*, 5> kMetalClearShapes{"1D", "1DArray", "2D", "2DArray", "3D"};
        constexpr std::array<const char*, 3> kMetalClearTypes{"Float", "Uint", "Sint"};

        [[nodiscard]] Result<id<MTLComputePipelineState>> metalCompileFillFunction(MetalDeviceNative& native,
                                                                                   id<MTLLibrary> library,
                                                                                   NSString* name)
        {
            MTL4LibraryFunctionDescriptor* const function = [MTL4LibraryFunctionDescriptor new];
            function.library = library;
            function.name = name;
            MTL4ComputePipelineDescriptor* const desc = [MTL4ComputePipelineDescriptor new];
            desc.computeFunctionDescriptor = function;
            desc.label = name;
            NSError* error = nil;
            id<MTLComputePipelineState> const pipeline = [native.compiler newComputePipelineStateWithDescriptor:desc
                                                                                            compilerTaskOptions:nil
                                                                                                          error:&error];
            if (pipeline == nil)
            {
                return std::unexpected(metalNsError(ErrorCode::nativeFailure, error, "newComputePipelineState(fill)"));
            }
            return pipeline;
        }

        [[nodiscard]] Status metalCreateFillAndClearPipelines(MetalDeviceNative& native)
        {
            NSError* error = nil;
            id<MTLLibrary> const library =
                [native.device newLibraryWithSource:[NSString stringWithUTF8String:kMetalFillAndClearSource]
                                            options:[MTLCompileOptions new]
                                              error:&error];
            if (library == nil)
            {
                return std::unexpected(
                    metalNsError(ErrorCode::nativeFailure, error, "newLibraryWithSource(fill and clear)"));
            }
            auto fill = metalCompileFillFunction(native, library, @"rachisFillBuffer");
            if (!fill)
            {
                return std::unexpected(std::move(fill.error()));
            }
            std::array<__strong id<MTLComputePipelineState>, kMetalClearShapes.size() * kMetalClearTypes.size()>
                clears{};
            for (std::size_t type = 0; type < kMetalClearTypes.size(); ++type)
            {
                for (std::size_t shape = 0; shape < kMetalClearShapes.size(); ++shape)
                {
                    NSString* const name = [NSString
                        stringWithFormat:@"rachisClear%s%s", kMetalClearShapes[shape], kMetalClearTypes[type]];
                    auto clear = metalCompileFillFunction(native, library, name);
                    if (!clear)
                    {
                        return std::unexpected(std::move(clear.error()));
                    }
                    clears[type * kMetalClearShapes.size() + shape] = *clear;
                }
            }
            native.fillBufferPipeline = *fill;
            for (std::size_t index = 0; index < clears.size(); ++index)
            {
                native.clearTexturePipelines[index] = clears[index];
            }
            return {};
        }

        [[nodiscard]] Status metalEnsureFillAndClearPipelines(MetalDeviceNative& native)
        {
            if (native.fillBufferPipeline != nil)
            {
                return {};
            }
            @autoreleasepool
            {
                return metalCreateFillAndClearPipelines(native);
            }
        }

        [[nodiscard]] std::optional<std::size_t> metalClearShapeIndex(MTLTextureType type) noexcept
        {
            switch (type)
            {
            case MTLTextureType1D:
                return 0;
            case MTLTextureType1DArray:
                return 1;
            case MTLTextureType2D:
                return 2;
            case MTLTextureType2DArray:
                return 3;
            case MTLTextureType3D:
                return 4;
            default:
                return std::nullopt;
            }
        }

        [[nodiscard]] std::size_t metalClearTypeIndex(StorageTextureClearType type) noexcept
        {
            switch (type)
            {
            case StorageTextureClearType::uint32:
                return 1;
            case StorageTextureClearType::sint32:
                return 2;
            case StorageTextureClearType::float32:
                break;
            }
            return 0;
        }

        [[nodiscard]] Status metalEncodeInternalDispatch(id<MTLDevice> device,
                                                         MetalCommandEntry& entry,
                                                         id<MTLComputePipelineState> pipeline,
                                                         std::span<const std::byte> constants,
                                                         id<MTLTexture> texture,
                                                         MTLSize threadgroups,
                                                         MTLSize threadsPerThreadgroup)
        {
            auto slice =
                metalCapturePushConstantSlice(device, entry, constants, static_cast<std::uint32_t>(constants.size()));
            if (!slice)
            {
                return std::unexpected(std::move(slice.error()));
            }
            metalRetainResidency(entry, slice->buffer);
            auto table = acquireMetalArgumentTable(
                device, entry, MetalArgumentTableStage::compute, 1, texture != nil ? 1 : 0, 0);
            if (!table)
            {
                return std::unexpected(std::move(table.error()));
            }
            [*table setAddress:slice->gpuAddress atIndex:0];
            if (texture != nil)
            {
                [*table setTexture:texture.gpuResourceID atIndex:0];
            }
            [entry.computeEncoder setComputePipelineState:pipeline];
            [entry.computeEncoder setArgumentTable:*table];
            [entry.computeEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
            entry.activeComputePipeline = nullptr;
            entry.activeComputeTable = nil;
            entry.computeTableDirty = true;
            return {};
        }

        struct MetalFillTarget final
        {
            std::uint64_t deviceToken = 0;
            __strong id<MTLDevice> device = nil;
            __strong id<MTLBuffer> buffer = nil;
            std::shared_ptr<void> owner;
            std::uint64_t size = 0;
            __strong id<MTLComputePipelineState> pipeline = nil;
        };

        struct MetalClearTarget final
        {
            std::uint64_t deviceToken = 0;
            __strong id<MTLDevice> device = nil;
            __strong id<MTLTexture> view = nil;
            std::shared_ptr<void> owner;
            std::shared_ptr<MetalTextureLayoutTable> layouts;
            TextureViewDesc viewDesc{};
            TextureDesc textureDesc{};
            __strong id<MTLComputePipelineState> pipeline = nil;
        };

    } // namespace

    Status MetalDevice::fillBuffer(detail::BackendCommandBuffer& commandBuffer,
                                   BufferHandle buffer,
                                   std::uint64_t offset,
                                   std::uint64_t size,
                                   std::uint32_t value)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        MetalFillTarget target{};
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr || native_->device == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal fillBuffer requires an active device."));
            }
            auto* record = native_->buffers.find(buffer);
            if (record == nullptr || record->buffer == nil)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "fillBuffer references an invalid or stale buffer."));
            }
            auto resolved = detail::resolveBufferFillRange(record->desc, offset, size);
            if (!resolved)
            {
                return std::unexpected(std::move(resolved.error()));
            }
            if (auto pipelines = metalEnsureFillAndClearPipelines(*native_); !pipelines)
            {
                return pipelines;
            }
            target = MetalFillTarget{
                .deviceToken = native_->deviceToken,
                .device = native_->device,
                .buffer = record->buffer,
                .owner = record->owner,
                .size = *resolved,
                .pipeline = native_->fillBufferPipeline,
            };
        }
        auto* native = asMetalCommand(commandBuffer, target.deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto* entry = native->entry();
        if (entry == nullptr || !entry->begun)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "fillBuffer requires a begun Metal command buffer."));
        }
        if (auto kept = retainMetalApplicationResource(*native, target.owner); !kept)
        {
            return kept;
        }
        if (auto prepared = native->ensureComputeEncoder(); !prepared)
        {
            return prepared;
        }
        metalRetainResidency(*entry, target.buffer);
        const std::uint64_t totalWords = target.size / sizeof(std::uint32_t);
        for (std::uint64_t first = 0; first < totalWords; first += kMetalFillWordsPerDispatch)
        {
            const auto words = static_cast<std::uint32_t>(std::min(kMetalFillWordsPerDispatch, totalWords - first));
            const MetalFillHeader header{
                .destination = target.buffer.gpuAddress + offset + first * sizeof(std::uint32_t),
                .words = words,
                .value = value,
            };
            const auto groups = (words + kMetalFillThreadgroupSize - 1U) / kMetalFillThreadgroupSize;
            if (auto encoded = metalEncodeInternalDispatch(target.device,
                                                           *entry,
                                                           target.pipeline,
                                                           std::as_bytes(std::span(&header, 1)),
                                                           nil,
                                                           MTLSizeMake(groups, 1, 1),
                                                           MTLSizeMake(kMetalFillThreadgroupSize, 1, 1));
                !encoded)
            {
                return encoded;
            }
        }
        return {};
    }

    Status MetalDevice::clearStorageTexture(detail::BackendCommandBuffer& commandBuffer,
                                            TextureViewHandle view,
                                            const StorageTextureClearValue& value)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        MetalClearTarget target{};
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr || native_->device == nil)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal clearStorageTexture requires an active device."));
            }
            auto* viewRecord = native_->views.find(view);
            auto* textureRecord = viewRecord != nullptr ? native_->textures.find(viewRecord->texture) : nullptr;
            if (viewRecord == nullptr || viewRecord->view == nil || textureRecord == nullptr)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "clearStorageTexture references an invalid or stale texture view."));
            }
            auto format = detail::validateStorageTextureClear(
                textureRecord->desc, viewRecord->desc.usage, viewRecord->desc.format, value);
            if (!format)
            {
                return std::unexpected(std::move(format.error()));
            }
            const auto shape = metalClearShapeIndex(viewRecord->view.textureType);
            if (!shape.has_value())
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported, "clearStorageTexture does not support this texture view dimension."));
            }
            if (auto pipelines = metalEnsureFillAndClearPipelines(*native_); !pipelines)
            {
                return pipelines;
            }
            target = MetalClearTarget{
                .deviceToken = native_->deviceToken,
                .device = native_->device,
                .view = viewRecord->view,
                .owner = viewRecord->owner,
                .layouts = metalTextureLayoutTable(*textureRecord),
                .viewDesc = viewRecord->desc,
                .textureDesc = textureRecord->desc,
                .pipeline =
                    native_->clearTexturePipelines[metalClearTypeIndex(value.type) * kMetalClearShapes.size() + *shape],
            };
        }
        auto* native = asMetalCommand(commandBuffer, target.deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto* entry = native->entry();
        if (entry == nullptr || !entry->begun)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "clearStorageTexture requires a begun Metal command buffer."));
        }
        const bool volume = target.view.textureType == MTLTextureType3D;
        const std::array transitions{MetalLayoutTransition{
            .table = target.layouts,
            .firstMip = target.viewDesc.firstMipLevel,
            .mipCount = 1,
            .firstLayer = volume ? 0U : target.viewDesc.firstArrayLayer,
            .layerCount = volume ? 1U : static_cast<std::uint32_t>(std::max<NSUInteger>(target.view.arrayLength, 1)),
            .before = TextureLayout::unorderedAccess,
            .after = TextureLayout::unorderedAccess,
            .operation = "clearStorageTexture",
        }};
        if (auto checked = metalCheckLayoutTransitions(native->layoutState(), transitions); !checked)
        {
            return checked;
        }
        if (auto kept = retainMetalApplicationResource(*native, target.owner); !kept)
        {
            return kept;
        }
        if (auto prepared = native->ensureComputeEncoder(); !prepared)
        {
            return prepared;
        }
        metalStageLayoutTransitions(native->layoutState(), transitions);
        metalRetainResidency(*entry, target.view);
        MetalClearValueBlock block{};
        std::memcpy(block.f.data(), value.float32.data(), sizeof(block.f));
        std::memcpy(block.u.data(), value.uint32.data(), sizeof(block.u));
        std::memcpy(block.s.data(), value.sint32.data(), sizeof(block.s));
        const auto depth = volume ? target.view.depth : std::max<NSUInteger>(target.view.arrayLength, 1);
        const bool oneDimensional =
            target.view.textureType == MTLTextureType1D || target.view.textureType == MTLTextureType1DArray;
        const MTLSize threads = oneDimensional ? MTLSizeMake(64, 1, 1) : MTLSizeMake(8, 8, 1);
        const MTLSize groups = MTLSizeMake((target.view.width + threads.width - 1U) / threads.width,
                                           (std::max<NSUInteger>(target.view.height, 1) + threads.height - 1U) /
                                               threads.height,
                                           depth);
        return metalEncodeInternalDispatch(
            target.device, *entry, target.pipeline, std::as_bytes(std::span(&block, 1)), target.view, groups, threads);
    }

} // namespace rachis::metal
