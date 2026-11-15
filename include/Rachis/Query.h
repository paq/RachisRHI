#pragma once

#include "Rachis/Command.h"
#include "Rachis/Handle.h"

#include <cstdint>
#include <string_view>

namespace rachis
{

    inline constexpr std::uint32_t maxStreamOutputStreams = 4;

    inline constexpr std::uint64_t timestampQueryResultSize = sizeof(std::uint64_t);
    inline constexpr std::uint64_t timestampQueryResultAlignment = sizeof(std::uint64_t);
    inline constexpr std::uint64_t occlusionQueryResultSize = sizeof(std::uint64_t);
    inline constexpr std::uint64_t occlusionQueryResultAlignment = sizeof(std::uint64_t);
    inline constexpr std::uint64_t accelerationStructureCompactedSizeQueryResultSize = sizeof(std::uint64_t);
    inline constexpr std::uint64_t accelerationStructureCompactedSizeQueryResultAlignment = alignof(std::uint64_t);
    inline constexpr std::uint64_t accelerationStructureSerializationInfoQueryResultAlignment = alignof(std::uint64_t);
    inline constexpr std::uint32_t timestampWriteUnused = UINT32_MAX;

    struct QueueProperties final
    {
        bool timestampQueries = false;
        bool inlineTimestampWrites = false;
        std::uint32_t timestampValidBits = 0;
        std::uint64_t timestampFrequency = 0;
    };

    struct QueueClockCalibration final
    {
        std::uint64_t gpuTimestamp = 0;
        std::uint64_t cpuTimestamp = 0;
        std::uint64_t gpuFrequency = 0;
        std::uint64_t cpuFrequency = 0;
    };

    inline constexpr std::uint64_t predicateValueSize = sizeof(std::uint64_t);
    inline constexpr std::uint64_t predicateValueAlignment = sizeof(std::uint64_t);

    enum class PredicateOperation : std::uint8_t
    {
        equalZero,

        notEqualZero,
    };

    struct ConditionalExecutionDesc final
    {
        BufferHandle buffer = {};
        std::uint64_t offset = 0;
        PredicateOperation operation = PredicateOperation::notEqualZero;
    };

    struct TimestampQueryPoolDesc final
    {
        QueueType queueType = QueueType::graphics;
        std::uint32_t queryCount = 0;
        std::string_view label = {};
    };

    struct TimestampQueryResolveDesc final
    {
        QueryPoolHandle queryPool = {};
        std::uint32_t firstQuery = 0;
        std::uint32_t queryCount = 0;
        BufferHandle destination = {};
        std::uint64_t destinationOffset = 0;
    };

    struct PassTimestampWrites final
    {
        QueryPoolHandle queryPool = {};
        std::uint32_t beginningOfPassWriteIndex = timestampWriteUnused;
        std::uint32_t endOfPassWriteIndex = timestampWriteUnused;
    };

    struct GpuProfileScopeDesc final
    {
        QueryPoolHandle queryPool = {};
        std::uint32_t beginningQueryIndex = 0;
        std::uint32_t endQueryIndex = 0;
        std::string_view label = {};
        std::optional<DebugColor> color = {};
    };

    enum class OcclusionQueryMode : std::uint8_t
    {
        precise,

        binary,
    };

    struct OcclusionQueryPoolDesc final
    {
        OcclusionQueryMode mode = OcclusionQueryMode::precise;
        std::uint32_t queryCount = 0;
        std::string_view label = {};
    };

    struct OcclusionQueryResolveDesc final
    {
        QueryPoolHandle queryPool = {};
        std::uint32_t firstQuery = 0;
        std::uint32_t queryCount = 0;
        BufferHandle destination = {};
        std::uint64_t destinationOffset = 0;
    };

    struct PipelineStatistics final
    {
        std::uint64_t inputAssemblyVertices = 0;
        std::uint64_t inputAssemblyPrimitives = 0;
        std::uint64_t vertexShaderInvocations = 0;
        std::uint64_t geometryShaderInvocations = 0;
        std::uint64_t geometryShaderPrimitives = 0;
        std::uint64_t clippingInvocations = 0;
        std::uint64_t clippingPrimitives = 0;
        std::uint64_t fragmentShaderInvocations = 0;
        std::uint64_t tessellationControlShaderPatches = 0;
        std::uint64_t tessellationEvaluationShaderInvocations = 0;
        std::uint64_t computeShaderInvocations = 0;
    };

    inline constexpr std::uint64_t pipelineStatisticsQueryResultSize = sizeof(PipelineStatistics);
    inline constexpr std::uint64_t pipelineStatisticsQueryResultAlignment = alignof(PipelineStatistics);

    struct MeshPipelineStatistics final
    {
        PipelineStatistics base = {};
        std::uint64_t taskShaderInvocations = 0;
        std::uint64_t meshShaderInvocations = 0;
        std::uint64_t meshShaderPrimitives = 0;
    };

    inline constexpr std::uint64_t meshPipelineStatisticsQueryResultSize = sizeof(MeshPipelineStatistics);
    inline constexpr std::uint64_t meshPipelineStatisticsQueryResultAlignment = alignof(MeshPipelineStatistics);

    struct StreamOutputStatistics final
    {
        std::uint64_t primitivesWritten = 0;
        std::uint64_t primitivesStorageNeeded = 0;
    };

    inline constexpr std::uint64_t streamOutputStatisticsQueryResultSize = sizeof(StreamOutputStatistics);
    inline constexpr std::uint64_t streamOutputStatisticsQueryResultAlignment = alignof(StreamOutputStatistics);

    struct StreamOutputStatisticsQueryPoolDesc final
    {
        std::uint32_t stream = 0;
        std::uint32_t queryCount = 0;
        std::string_view label = {};
    };

    struct StreamOutputStatisticsQueryResolveDesc final
    {
        QueryPoolHandle queryPool = {};
        std::uint32_t firstQuery = 0;
        std::uint32_t queryCount = 0;
        BufferHandle destination = {};
        std::uint64_t destinationOffset = 0;
    };

    enum class PipelineStatisticsQueryMode : std::uint8_t
    {
        base,

        meshShader,
    };

    struct PipelineStatisticsQueryPoolDesc final
    {
        PipelineStatisticsQueryMode mode = PipelineStatisticsQueryMode::base;
        std::uint32_t queryCount = 0;
        std::string_view label = {};
    };

    struct PipelineStatisticsQueryResolveDesc final
    {
        QueryPoolHandle queryPool = {};
        std::uint32_t firstQuery = 0;
        std::uint32_t queryCount = 0;
        BufferHandle destination = {};
        std::uint64_t destinationOffset = 0;
    };

    struct AccelerationStructureCompactedSizeQueryPoolDesc final
    {
        QueueType queueType = QueueType::graphics;
        std::uint32_t queryCount = 0;
        std::string_view label = {};
    };

    struct AccelerationStructureCompactedSizeQueryResolveDesc final
    {
        QueryPoolHandle queryPool = {};
        std::uint32_t firstQuery = 0;
        std::uint32_t queryCount = 0;
        BufferHandle destination = {};
        std::uint64_t destinationOffset = 0;
    };

    struct AccelerationStructureSerializationInfo final
    {
        std::uint64_t serializedSize = 0;
        std::uint64_t bottomLevelPointerCount = 0;
    };

    inline constexpr std::uint64_t accelerationStructureSerializationInfoQueryResultSize =
        sizeof(AccelerationStructureSerializationInfo);

    struct AccelerationStructureSerializationInfoQueryPoolDesc final
    {
        QueueType queueType = QueueType::graphics;
        std::uint32_t queryCount = 0;
        std::string_view label = {};
    };

    struct AccelerationStructureSerializationInfoQueryResolveDesc final
    {
        QueryPoolHandle queryPool = {};
        std::uint32_t firstQuery = 0;
        std::uint32_t queryCount = 0;
        BufferHandle destination = {};
        std::uint64_t destinationOffset = 0;
    };

    static_assert(sizeof(PipelineStatistics) == 11 * sizeof(std::uint64_t));
    static_assert(alignof(PipelineStatistics) == alignof(std::uint64_t));
    static_assert(sizeof(MeshPipelineStatistics) == 14 * sizeof(std::uint64_t));
    static_assert(alignof(MeshPipelineStatistics) == alignof(std::uint64_t));
    static_assert(sizeof(StreamOutputStatistics) == 2 * sizeof(std::uint64_t));
    static_assert(alignof(StreamOutputStatistics) == alignof(std::uint64_t));
    static_assert(sizeof(AccelerationStructureSerializationInfo) == 2 * sizeof(std::uint64_t));
    static_assert(alignof(AccelerationStructureSerializationInfo) == alignof(std::uint64_t));

} // namespace rachis
