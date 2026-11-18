#include "MetalInternal.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>

namespace rachis::metal
{

    namespace
    {

        constexpr std::uint32_t kMetalClampThreadgroupSize = 64;

        constexpr const char* kMetalIndirectRecordSource = R"msl(
#include <metal_stdlib>
using namespace metal;

struct RachisIndirectClamp
{
    device const uint* source;
    device const uint* count;
    device uint* destination;
    uint maxRecords;
    uint strideWords;
    uint recordWords;
    uint zeroFirstWord;
    uint zeroWordCount;
};

static void rachisClampRecord(constant RachisIndirectClamp& clamp, uint record)
{
    if (record >= clamp.maxRecords)
    {
        return;
    }
    const uint active = min(*clamp.count, clamp.maxRecords);
    device const uint* source = clamp.source + record * clamp.strideWords;
    device uint* destination = clamp.destination + record * clamp.recordWords;
    for (uint word = 0; word < clamp.recordWords; ++word)
    {
        const bool zero = record >= active && word >= clamp.zeroFirstWord &&
                          word < clamp.zeroFirstWord + clamp.zeroWordCount;
        destination[word] = zero ? 0u : source[word];
    }
}

[[kernel]] void rachisClampIndirectRecordsKernel(constant RachisIndirectClamp& clamp [[buffer(0)]],
                                                 uint record [[thread_position_in_grid]])
{
    rachisClampRecord(clamp, record);
}

[[object]] void rachisClampIndirectRecordsObject(constant RachisIndirectClamp& clamp [[buffer(0)]],
                                                 uint record [[thread_position_in_grid]],
                                                 mesh_grid_properties grid)
{
    rachisClampRecord(clamp, record);
    grid.set_threadgroups_per_grid(uint3(0, 0, 0));
}

struct RachisClampVertex
{
    float4 position [[position]];
};

[[mesh]] void rachisClampIndirectRecordsMesh(metal::mesh<RachisClampVertex, void, 1, 1, topology::point> output)
{
}
)msl";

        struct MetalIndirectClampHeader final
        {
            std::uint64_t source = 0;
            std::uint64_t count = 0;
            std::uint64_t destination = 0;
            std::uint32_t maxRecords = 0;
            std::uint32_t strideWords = 0;
            std::uint32_t recordWords = 0;
            std::uint32_t zeroFirstWord = 0;
            std::uint32_t zeroWordCount = 0;
            std::uint32_t padding = 0;
        };

        static_assert(sizeof(MetalIndirectClampHeader) == 48);

        [[nodiscard]] MTL4LibraryFunctionDescriptor* metalClampFunction(id<MTLLibrary> library, NSString* name)
        {
            MTL4LibraryFunctionDescriptor* const function = [MTL4LibraryFunctionDescriptor new];
            function.library = library;
            function.name = name;
            return function;
        }

        [[nodiscard]] Result<id<MTLLibrary>> metalCompileClampLibrary(id<MTLDevice> device)
        {
            NSError* error = nil;
            MTLCompileOptions* const options = [MTLCompileOptions new];
            id<MTLLibrary> const library =
                [device newLibraryWithSource:[NSString stringWithUTF8String:kMetalIndirectRecordSource]
                                     options:options
                                       error:&error];
            if (library == nil)
            {
                return std::unexpected(
                    metalNsError(ErrorCode::nativeFailure, error, "newLibraryWithSource(indirect records)"));
            }
            return library;
        }

        [[nodiscard]] Status metalCreateClampPipelines(MetalDeviceNative& native)
        {
            auto library = metalCompileClampLibrary(native.device);
            if (!library)
            {
                return std::unexpected(std::move(library.error()));
            }
            NSError* error = nil;
            MTL4ComputePipelineDescriptor* const computeDesc = [MTL4ComputePipelineDescriptor new];
            computeDesc.computeFunctionDescriptor = metalClampFunction(*library, @"rachisClampIndirectRecordsKernel");
            computeDesc.label = @"RHI.Metal.IndirectRecordClamp";
            id<MTLComputePipelineState> const compute =
                [native.compiler newComputePipelineStateWithDescriptor:computeDesc
                                                   compilerTaskOptions:nil
                                                                 error:&error];
            if (compute == nil)
            {
                return std::unexpected(
                    metalNsError(ErrorCode::nativeFailure, error, "newComputePipelineState(indirect records)"));
            }
            MTL4MeshRenderPipelineDescriptor* const renderDesc = [MTL4MeshRenderPipelineDescriptor new];
            renderDesc.objectFunctionDescriptor = metalClampFunction(*library, @"rachisClampIndirectRecordsObject");
            renderDesc.meshFunctionDescriptor = metalClampFunction(*library, @"rachisClampIndirectRecordsMesh");
            renderDesc.maxTotalThreadsPerObjectThreadgroup = kMetalClampThreadgroupSize;
            renderDesc.maxTotalThreadsPerMeshThreadgroup = 1;
            renderDesc.maxTotalThreadgroupsPerMeshGrid = 1;
            renderDesc.rasterizationEnabled = NO;
            renderDesc.label = @"RHI.Metal.IndirectRecordClampObject";
            error = nil;
            id<MTLRenderPipelineState> const render = [native.compiler newRenderPipelineStateWithDescriptor:renderDesc
                                                                                        compilerTaskOptions:nil
                                                                                                      error:&error];
            if (render == nil)
            {
                return std::unexpected(
                    metalNsError(ErrorCode::nativeFailure, error, "newRenderPipelineState(indirect records)"));
            }
            native.indirectClampCompute = compute;
            native.indirectClampRender = render;
            return {};
        }

        [[nodiscard]] std::uint32_t metalClampThreadgroups(std::uint32_t records) noexcept
        {
            return (records + kMetalClampThreadgroupSize - 1U) / kMetalClampThreadgroupSize;
        }

    } // namespace

    Result<MetalIndirectClampPipelines> metalIndirectClampPipelines(std::mutex& deviceMutex,
                                                                    const std::shared_ptr<MetalDeviceNative>& native)
    {
        std::scoped_lock const lock(deviceMutex);
        if (native == nullptr || native->device == nil || native->compiler == nil)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "Indirect calls with a count buffer require an active Metal device."));
        }
        if (native->indirectClampCompute == nil || native->indirectClampRender == nil)
        {
            @autoreleasepool
            {
                if (auto created = metalCreateClampPipelines(*native); !created)
                {
                    return std::unexpected(std::move(created.error()));
                }
            }
        }
        return MetalIndirectClampPipelines{
            .compute = native->indirectClampCompute,
            .render = native->indirectClampRender,
        };
    }

    Result<MetalClampedRecords> metalPrepareClampedRecords(id<MTLDevice> device,
                                                           MetalCommandEntry& entry,
                                                           const MetalIndirectClampRequest& request)
    {
        const auto recordBytes = static_cast<std::uint64_t>(request.recordWords) * sizeof(std::uint32_t);
        const auto length = sizeof(MetalIndirectClampHeader) + recordBytes * request.maxRecords;
        id<MTLBuffer> const scratch = [device newBufferWithLength:static_cast<NSUInteger>(length)
                                                          options:MTLResourceStorageModeShared];
        if (scratch == nil)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::outOfMemory, "Metal could not allocate scratch records for an indirect call."));
        }
        scratch.label = @"RHI.Metal.IndirectRecords";
        try
        {
            entry.indirectScratchBuffers.push_back(scratch);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::outOfMemory, "Failed to retain Metal indirect scratch records."));
        }
        metalRetainResidency(entry, scratch);
        const std::uint64_t recordsAddress = scratch.gpuAddress + sizeof(MetalIndirectClampHeader);
        const MetalIndirectClampHeader header{
            .source = request.sourceAddress,
            .count = request.countAddress,
            .destination = recordsAddress,
            .maxRecords = request.maxRecords,
            .strideWords = request.strideBytes / static_cast<std::uint32_t>(sizeof(std::uint32_t)),
            .recordWords = request.recordWords,
            .zeroFirstWord = request.zeroFirstWord,
            .zeroWordCount = request.zeroWordCount,
        };
        std::memcpy(scratch.contents, &header, sizeof(header));
        return MetalClampedRecords{
            .headerAddress = scratch.gpuAddress,
            .recordsAddress = recordsAddress,
            .recordBytes = static_cast<std::uint32_t>(recordBytes),
        };
    }

    Status metalEncodeComputeRecordClamp(id<MTLDevice> device,
                                         MetalCommandEntry& entry,
                                         id<MTLComputePipelineState> pipeline,
                                         const MetalClampedRecords& records,
                                         std::uint32_t maxRecords)
    {
        auto table = acquireMetalArgumentTable(device, entry, MetalArgumentTableStage::compute, 1, 0, 0);
        if (!table)
        {
            return std::unexpected(std::move(table.error()));
        }
        [*table setAddress:records.headerAddress atIndex:0];
        [entry.computeEncoder setComputePipelineState:pipeline];
        [entry.computeEncoder setArgumentTable:*table];
        [entry.computeEncoder dispatchThreadgroups:MTLSizeMake(metalClampThreadgroups(maxRecords), 1, 1)
                             threadsPerThreadgroup:MTLSizeMake(kMetalClampThreadgroupSize, 1, 1)];
        [entry.computeEncoder barrierAfterEncoderStages:MTLStageDispatch
                                    beforeEncoderStages:MTLStageDispatch
                                      visibilityOptions:MTL4VisibilityOptionDevice];
        entry.activeComputePipeline = nullptr;
        entry.activeComputeTable = nil;
        entry.computeTableDirty = true;
        return {};
    }

    Status metalEncodeRenderRecordClamp(id<MTLDevice> device,
                                        MetalCommandEntry& entry,
                                        id<MTLRenderPipelineState> pipeline,
                                        const MetalClampedRecords& records,
                                        std::uint32_t maxRecords)
    {
        auto table = acquireMetalArgumentTable(device, entry, MetalArgumentTableStage::object, 1, 0, 0);
        if (!table)
        {
            return std::unexpected(std::move(table.error()));
        }
        [*table setAddress:records.headerAddress atIndex:0];
        [entry.renderEncoder setRenderPipelineState:pipeline];
        [entry.renderEncoder setArgumentTable:*table atStages:MTLRenderStageObject];
        [entry.renderEncoder drawMeshThreadgroups:MTLSizeMake(metalClampThreadgroups(maxRecords), 1, 1)
                      threadsPerObjectThreadgroup:MTLSizeMake(kMetalClampThreadgroupSize, 1, 1)
                        threadsPerMeshThreadgroup:MTLSizeMake(1, 1, 1)];
        [entry.renderEncoder barrierAfterEncoderStages:MTLStageObject
                                   beforeEncoderStages:kMetalRenderEncoderStages
                                     visibilityOptions:MTL4VisibilityOptionDevice];
        entry.activeGraphicsPipeline = nullptr;
        entry.vertexTableDirty = true;
        return {};
    }

} // namespace rachis::metal
