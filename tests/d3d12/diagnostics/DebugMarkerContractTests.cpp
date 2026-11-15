#include "D3D12/D3D12DebugMarker.h"
#include "Internal/BackendCommandBuffer.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>
#include <d3d12.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace
{

    [[nodiscard]] rachis::Result<rachis::Device> createWarpDevice()
    {
        return rachis::createDevice(rachis::DeviceDesc{
            .adapterPreference = rachis::AdapterPreference::warp,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = true,
        });
    }

    class RecordingCommandList final : public ID3D12GraphicsCommandList
    {
    public:
        std::vector<std::vector<std::byte>> beginEvents;
        std::size_t endEvents = 0;

        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void** object) override
        {
            *object = nullptr;
            return E_NOINTERFACE;
        }
        ULONG STDMETHODCALLTYPE AddRef() override
        {
            return 1;
        }
        ULONG STDMETHODCALLTYPE Release() override
        {
            return 1;
        }
        HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT*, void*) override
        {
            return E_NOTIMPL;
        }
        HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void*) override
        {
            return E_NOTIMPL;
        }
        HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown*) override
        {
            return E_NOTIMPL;
        }
        HRESULT STDMETHODCALLTYPE SetName(LPCWSTR) override
        {
            return E_NOTIMPL;
        }
        HRESULT STDMETHODCALLTYPE GetDevice(REFIID, void**) override
        {
            return E_NOTIMPL;
        }
        D3D12_COMMAND_LIST_TYPE STDMETHODCALLTYPE GetType() override
        {
            return {};
        }
        HRESULT STDMETHODCALLTYPE Close() override
        {
            return E_NOTIMPL;
        }
        HRESULT STDMETHODCALLTYPE Reset(ID3D12CommandAllocator*, ID3D12PipelineState*) override
        {
            return E_NOTIMPL;
        }
        void STDMETHODCALLTYPE ClearState(ID3D12PipelineState*) override
        {
        }
        void STDMETHODCALLTYPE DrawInstanced(UINT, UINT, UINT, UINT) override
        {
        }
        void STDMETHODCALLTYPE DrawIndexedInstanced(UINT, UINT, UINT, INT, UINT) override
        {
        }
        void STDMETHODCALLTYPE Dispatch(UINT, UINT, UINT) override
        {
        }
        void STDMETHODCALLTYPE CopyBufferRegion(ID3D12Resource*, UINT64, ID3D12Resource*, UINT64, UINT64) override
        {
        }
        void STDMETHODCALLTYPE CopyTextureRegion(const D3D12_TEXTURE_COPY_LOCATION*,
                                                 UINT,
                                                 UINT,
                                                 UINT,
                                                 const D3D12_TEXTURE_COPY_LOCATION*,
                                                 const D3D12_BOX*) override
        {
        }
        void STDMETHODCALLTYPE CopyResource(ID3D12Resource*, ID3D12Resource*) override
        {
        }
        void STDMETHODCALLTYPE CopyTiles(ID3D12Resource*,
                                         const D3D12_TILED_RESOURCE_COORDINATE*,
                                         const D3D12_TILE_REGION_SIZE*,
                                         ID3D12Resource*,
                                         UINT64,
                                         D3D12_TILE_COPY_FLAGS) override
        {
        }
        void STDMETHODCALLTYPE ResolveSubresource(ID3D12Resource*, UINT, ID3D12Resource*, UINT, DXGI_FORMAT) override
        {
        }
        void STDMETHODCALLTYPE IASetPrimitiveTopology(D3D12_PRIMITIVE_TOPOLOGY) override
        {
        }
        void STDMETHODCALLTYPE RSSetViewports(UINT, const D3D12_VIEWPORT*) override
        {
        }
        void STDMETHODCALLTYPE RSSetScissorRects(UINT, const D3D12_RECT*) override
        {
        }
        void STDMETHODCALLTYPE OMSetBlendFactor(const FLOAT[4]) override
        {
        }
        void STDMETHODCALLTYPE OMSetStencilRef(UINT) override
        {
        }
        void STDMETHODCALLTYPE SetPipelineState(ID3D12PipelineState*) override
        {
        }
        void STDMETHODCALLTYPE ResourceBarrier(UINT, const D3D12_RESOURCE_BARRIER*) override
        {
        }
        void STDMETHODCALLTYPE ExecuteBundle(ID3D12GraphicsCommandList*) override
        {
        }
        void STDMETHODCALLTYPE SetDescriptorHeaps(UINT, ID3D12DescriptorHeap* const*) override
        {
        }
        void STDMETHODCALLTYPE SetComputeRootSignature(ID3D12RootSignature*) override
        {
        }
        void STDMETHODCALLTYPE SetGraphicsRootSignature(ID3D12RootSignature*) override
        {
        }
        void STDMETHODCALLTYPE SetComputeRootDescriptorTable(UINT, D3D12_GPU_DESCRIPTOR_HANDLE) override
        {
        }
        void STDMETHODCALLTYPE SetGraphicsRootDescriptorTable(UINT, D3D12_GPU_DESCRIPTOR_HANDLE) override
        {
        }
        void STDMETHODCALLTYPE SetComputeRoot32BitConstant(UINT, UINT, UINT) override
        {
        }
        void STDMETHODCALLTYPE SetGraphicsRoot32BitConstant(UINT, UINT, UINT) override
        {
        }
        void STDMETHODCALLTYPE SetComputeRoot32BitConstants(UINT, UINT, const void*, UINT) override
        {
        }
        void STDMETHODCALLTYPE SetGraphicsRoot32BitConstants(UINT, UINT, const void*, UINT) override
        {
        }
        void STDMETHODCALLTYPE SetComputeRootConstantBufferView(UINT, D3D12_GPU_VIRTUAL_ADDRESS) override
        {
        }
        void STDMETHODCALLTYPE SetGraphicsRootConstantBufferView(UINT, D3D12_GPU_VIRTUAL_ADDRESS) override
        {
        }
        void STDMETHODCALLTYPE SetComputeRootShaderResourceView(UINT, D3D12_GPU_VIRTUAL_ADDRESS) override
        {
        }
        void STDMETHODCALLTYPE SetGraphicsRootShaderResourceView(UINT, D3D12_GPU_VIRTUAL_ADDRESS) override
        {
        }
        void STDMETHODCALLTYPE SetComputeRootUnorderedAccessView(UINT, D3D12_GPU_VIRTUAL_ADDRESS) override
        {
        }
        void STDMETHODCALLTYPE SetGraphicsRootUnorderedAccessView(UINT, D3D12_GPU_VIRTUAL_ADDRESS) override
        {
        }
        void STDMETHODCALLTYPE IASetIndexBuffer(const D3D12_INDEX_BUFFER_VIEW*) override
        {
        }
        void STDMETHODCALLTYPE IASetVertexBuffers(UINT, UINT, const D3D12_VERTEX_BUFFER_VIEW*) override
        {
        }
        void STDMETHODCALLTYPE SOSetTargets(UINT, UINT, const D3D12_STREAM_OUTPUT_BUFFER_VIEW*) override
        {
        }
        void STDMETHODCALLTYPE OMSetRenderTargets(UINT,
                                                  const D3D12_CPU_DESCRIPTOR_HANDLE*,
                                                  BOOL,
                                                  const D3D12_CPU_DESCRIPTOR_HANDLE*) override
        {
        }
        void STDMETHODCALLTYPE ClearDepthStencilView(
            D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_CLEAR_FLAGS, FLOAT, UINT8, UINT, const D3D12_RECT*) override
        {
        }
        void STDMETHODCALLTYPE ClearRenderTargetView(D3D12_CPU_DESCRIPTOR_HANDLE,
                                                     const FLOAT[4],
                                                     UINT,
                                                     const D3D12_RECT*) override
        {
        }
        void STDMETHODCALLTYPE ClearUnorderedAccessViewUint(D3D12_GPU_DESCRIPTOR_HANDLE,
                                                            D3D12_CPU_DESCRIPTOR_HANDLE,
                                                            ID3D12Resource*,
                                                            const UINT[4],
                                                            UINT,
                                                            const D3D12_RECT*) override
        {
        }
        void STDMETHODCALLTYPE ClearUnorderedAccessViewFloat(D3D12_GPU_DESCRIPTOR_HANDLE,
                                                             D3D12_CPU_DESCRIPTOR_HANDLE,
                                                             ID3D12Resource*,
                                                             const FLOAT[4],
                                                             UINT,
                                                             const D3D12_RECT*) override
        {
        }
        void STDMETHODCALLTYPE DiscardResource(ID3D12Resource*, const D3D12_DISCARD_REGION*) override
        {
        }
        void STDMETHODCALLTYPE BeginQuery(ID3D12QueryHeap*, D3D12_QUERY_TYPE, UINT) override
        {
        }
        void STDMETHODCALLTYPE EndQuery(ID3D12QueryHeap*, D3D12_QUERY_TYPE, UINT) override
        {
        }
        void STDMETHODCALLTYPE
        ResolveQueryData(ID3D12QueryHeap*, D3D12_QUERY_TYPE, UINT, UINT, ID3D12Resource*, UINT64) override
        {
        }
        void STDMETHODCALLTYPE SetPredication(ID3D12Resource*, UINT64, D3D12_PREDICATION_OP) override
        {
        }
        void STDMETHODCALLTYPE
        ExecuteIndirect(ID3D12CommandSignature*, UINT, ID3D12Resource*, UINT64, ID3D12Resource*, UINT64) override
        {
        }
        void STDMETHODCALLTYPE SetMarker(UINT, const void*, UINT) override
        {
        }
        void STDMETHODCALLTYPE BeginEvent(UINT, const void* data, UINT size) override
        {
            const auto* bytes = static_cast<const std::byte*>(data);
            beginEvents.emplace_back(bytes, bytes + size);
        }
        void STDMETHODCALLTYPE EndEvent() override
        {
            ++endEvents;
        }
    };

    class MarkerCommandBuffer final : public rachis::detail::BackendCommandBuffer
    {
    public:
        [[nodiscard]] rachis::QueueType queueType() const noexcept override
        {
            return rachis::QueueType::graphics;
        }
        [[nodiscard]] std::uint64_t deviceToken() const noexcept override
        {
            return 1;
        }
    };

    [[nodiscard]] bool encodesWideString(const std::vector<std::byte>& event, std::u16string_view text)
    {
        constexpr std::uint64_t ansiStringFlag = 1ULL << 54U;
        const auto textBytes = text.size() * sizeof(char16_t);
        for (std::size_t offset = sizeof(std::uint64_t); offset + textBytes + sizeof(char16_t) <= event.size();
             offset += sizeof(std::uint64_t))
        {
            if (std::memcmp(event.data() + offset, text.data(), textBytes) != 0)
            {
                continue;
            }
            std::uint64_t header = 0;
            std::memcpy(&header, event.data() + offset - sizeof(header), sizeof(header));
            char16_t terminator = u'x';
            std::memcpy(&terminator, event.data() + offset + textBytes, sizeof(terminator));
            return (header & ansiStringFlag) == 0 && terminator == u'\0';
        }
        return false;
    }

} // namespace

TEST_CASE("debug groups and markers preserve a balanced portable command contract",
          "[size:medium][rhi][debug-marker][warp]")
{
    auto created = createWarpDevice();
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).debugMarkers);

    auto commandPool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Debug marker test command pool",
    });
    REQUIRE(commandPool.has_value());

    auto acquired = device.acquireCommandBuffer(*commandPool);
    REQUIRE(acquired.has_value());

    const rachis::DebugMarkerDesc outer{
        .label = "Frame 17",
        .color = rachis::DebugColor{.red = 0.25F, .green = 0.5F, .blue = 1.0F, .alpha = 0.75F},
    };
    REQUIRE(device.beginDebugGroup(*acquired, outer).has_value());
    REQUIRE(device.beginDebugGroup(*acquired, {.label = "描画パス"}).has_value());
    REQUIRE(device.insertDebugMarker(*acquired, {.label = "Draw 100% complete"}).has_value());
    REQUIRE(device.endDebugGroup(*acquired).has_value());

    rachis::Status crossThreadEnd;
    std::jthread otherThread([&] { crossThreadEnd = device.endDebugGroup(*acquired); });
    otherThread.join();
    REQUIRE_FALSE(crossThreadEnd.has_value());
    CHECK(crossThreadEnd.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.endDebugGroup(*acquired).has_value());

    auto underflow = device.endDebugGroup(*acquired);
    REQUIRE_FALSE(underflow.has_value());
    CHECK(underflow.error().code == rachis::ErrorCode::invalidState);

    auto empty = device.insertDebugMarker(*acquired, {.label = {}});
    REQUIRE_FALSE(empty.has_value());
    CHECK(empty.error().code == rachis::ErrorCode::invalidArgument);

    const std::string embeddedNull("valid\0invalid", 13);
    auto nullLabel = device.insertDebugMarker(*acquired, {.label = embeddedNull});
    REQUIRE_FALSE(nullLabel.has_value());
    CHECK(nullLabel.error().code == rachis::ErrorCode::invalidArgument);

    const std::string longLabel(rachis::maxDebugLabelBytes + 1, 'x');
    auto oversized = device.insertDebugMarker(*acquired, {.label = longLabel});
    REQUIRE_FALSE(oversized.has_value());
    CHECK(oversized.error().code == rachis::ErrorCode::invalidArgument);

    auto invalidColor = device.insertDebugMarker(*acquired,
                                                 {
                                                     .label = "invalid color",
                                                     .color =
                                                         rachis::DebugColor{
                                                             .red = std::numeric_limits<float>::quiet_NaN(),
                                                             .green = 0.0F,
                                                             .blue = 0.0F,
                                                             .alpha = 1.0F,
                                                         },
                                                 });
    REQUIRE_FALSE(invalidColor.has_value());
    CHECK(invalidColor.error().code == rachis::ErrorCode::invalidArgument);

    for (std::uint32_t depth = 0; depth < rachis::maxDebugGroupDepth; ++depth)
    {
        CAPTURE(depth);
        REQUIRE(device.beginDebugGroup(*acquired, {.label = "bounded nesting"}).has_value());
    }
    auto tooDeep = device.beginDebugGroup(*acquired, {.label = "overflow"});
    REQUIRE_FALSE(tooDeep.has_value());
    CHECK(tooDeep.error().code == rachis::ErrorCode::invalidState);
    for (std::uint32_t depth = 0; depth < rachis::maxDebugGroupDepth; ++depth)
    {
        REQUIRE(device.endDebugGroup(*acquired).has_value());
    }

    REQUIRE(device.beginDebugGroup(*acquired, {.label = "must close before submit"}).has_value());
    auto rejectedSubmit = device.submit(std::move(*acquired));
    REQUIRE_FALSE(rejectedSubmit.has_value());
    CHECK(rejectedSubmit.error().code == rachis::ErrorCode::invalidState);
    CHECK(acquired->isValid());
    REQUIRE(device.endDebugGroup(*acquired).has_value());

    auto submitted = device.submit(std::move(*acquired));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto abandoned = device.acquireCommandBuffer(*commandPool);
    REQUIRE(abandoned.has_value());
    REQUIRE(device.beginDebugGroup(*abandoned, {.label = "abandoned but balanced internally"}).has_value());
    *abandoned = rachis::CommandBuffer{};

    auto recycled = device.acquireCommandBuffer(*commandPool);
    REQUIRE(recycled.has_value());
    REQUIRE(device.insertDebugMarker(*recycled, {.label = "recycled command list"}).has_value());
    auto recycledPoint = device.submit(std::move(*recycled));
    REQUIRE(recycledPoint.has_value());
    REQUIRE(device.wait(*recycledPoint).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
    REQUIRE(device.destroyCommandPool(*commandPool).has_value());
}

TEST_CASE("D3D12 debug labels reach PIX events as UTF-16 text", "[size:medium][rhi][debug-marker][warp]")
{
    auto created = createWarpDevice();
    REQUIRE(created.has_value());
    REQUIRE(rachis::test::requireCapabilities(*created).debugMarkers);

    const std::string label = "描画パス Ünïcode 100% ✓";
    constexpr std::u16string_view expected = u"描画パス Ünïcode 100% ✓";
    const rachis::DebugMarkerDesc desc{.label = label};
    const rachis::d3d12::DebugMarkerManager manager;
    RecordingCommandList commandList;
    MarkerCommandBuffer commandBuffer;

    REQUIRE(manager.validateProfileBegin(commandBuffer, desc).has_value());
    manager.commitProfileBegin(commandList, commandBuffer, desc);
    REQUIRE(manager.validateProfileEnd(commandBuffer).has_value());
    manager.commitProfileEnd(commandList, commandBuffer);

    REQUIRE(commandList.beginEvents.size() == 1);
    CHECK(commandList.endEvents == 1);
    CHECK(encodesWideString(commandList.beginEvents.front(), expected));
}
