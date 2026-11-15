#include "D3D12/D3D12DebugMarker.h"

#include "D3D12/D3D12Queue.h"
#include "Internal/BackendCommandBuffer.h"

#include <Windows.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <thread>

#if RACHIS_ENABLE_DEBUG_MARKERS
#include <pix3.h>
#endif

namespace rachis::d3d12
{
    namespace
    {

        alignas(16) constexpr std::array<wchar_t, 16> labelFormat{L'%', L'l', L's', L'\0'};

        [[nodiscard]] bool validColor(const DebugColor& color) noexcept
        {
            const std::array components{color.red, color.green, color.blue, color.alpha};
            return std::ranges::all_of(components, [](float component) {
                return std::isfinite(component) && component >= 0.0F && component <= 1.0F;
            });
        }

        [[nodiscard]] Status validateDesc(const DebugMarkerDesc& desc)
        {
            if (desc.label.empty())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A debug marker label cannot be empty.",
                });
            }
            if (desc.label.size() > maxDebugLabelBytes)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A debug marker label exceeds the portable byte limit.",
                });
            }
            if (desc.label.contains('\0'))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A debug marker label cannot contain an embedded null byte.",
                });
            }
            if (desc.color && !validColor(*desc.color))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Debug marker color components must be finite values in the inclusive range [0, 1].",
                });
            }
            return {};
        }

#if RACHIS_ENABLE_DEBUG_MARKERS
        [[nodiscard]] std::uint64_t pixColor(const std::optional<DebugColor>& color) noexcept
        {
            if (!color)
            {
                return PIX_COLOR_DEFAULT;
            }

            const auto toByte = [](float component) {
                return static_cast<std::uint8_t>(std::lround(component * 255.0F));
            };
            return PIX_COLOR(toByte(color->red), toByte(color->green), toByte(color->blue));
        }

        using PixLabelBuffer = std::array<wchar_t, maxDebugLabelBytes + 16>;

        [[nodiscard]] const wchar_t* copyPixLabel(std::string_view label, PixLabelBuffer& destination) noexcept
        {
            const int converted = MultiByteToWideChar(CP_UTF8,
                                                      0,
                                                      label.data(),
                                                      static_cast<int>(label.size()),
                                                      destination.data(),
                                                      static_cast<int>(maxDebugLabelBytes));
            const auto length = converted > 0 ? static_cast<std::ptrdiff_t>(converted) : 0;
            std::fill_n(destination.begin() + length, 16, L'\0');
            return destination.data();
        }

        void emitBegin(ID3D12GraphicsCommandList& commandList, const DebugMarkerDesc& desc) noexcept
        {
            alignas(16) PixLabelBuffer label;
            PIXBeginEvent(&commandList, pixColor(desc.color), labelFormat.data(), copyPixLabel(desc.label, label));
        }

        void emitMarker(ID3D12GraphicsCommandList& commandList, const DebugMarkerDesc& desc) noexcept
        {
            alignas(16) PixLabelBuffer label;
            PIXSetMarker(&commandList, pixColor(desc.color), labelFormat.data(), copyPixLabel(desc.label, label));
        }

#else
        [[nodiscard]] Status unsupported()
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Debug markers are disabled in this RHI build.",
            });
        }
#endif

        [[nodiscard]] Status validateGroupBegin(const detail::DebugCommandState& state,
                                                const DebugMarkerDesc& desc,
                                                detail::DebugGroupOwner owner,
                                                bool requireNativeMarkers)
        {
#if !RACHIS_ENABLE_DEBUG_MARKERS
            if (requireNativeMarkers)
            {
                return unsupported();
            }
#else
            static_cast<void>(requireNativeMarkers);
#endif
            auto valid = validateDesc(desc);
            if (!valid)
            {
                return valid;
            }
            const auto currentThread = std::this_thread::get_id();
            if (state.groupDepth == maxDebugGroupDepth)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The command buffer reached the portable debug-group nesting limit.",
                });
            }
            if (state.groupDepth != 0 && state.groupThread != currentThread)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Nested debug and GPU profile groups must be recorded on the thread that opened the outer group.",
                });
            }
            static_cast<void>(owner);
            return {};
        }

        [[nodiscard]] Status validateGroupEnd(const detail::DebugCommandState& state,
                                              detail::DebugGroupOwner expectedOwner,
                                              bool requireNativeMarkers)
        {
#if !RACHIS_ENABLE_DEBUG_MARKERS
            if (requireNativeMarkers)
            {
                return unsupported();
            }
#else
            static_cast<void>(requireNativeMarkers);
#endif
            if (state.groupDepth == 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot end a debug or GPU profile group when no group is open.",
                });
            }
            if (state.groupOwners[state.groupDepth - 1] != expectedOwner)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    expectedOwner == detail::DebugGroupOwner::user
                        ? "A user debug-group end cannot close a GPU profile scope."
                        : "A GPU profile-scope end cannot close a nested user debug group.",
                });
            }
            if (state.groupThread != std::this_thread::get_id())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A debug or GPU profile group must end on the thread that opened it.",
                });
            }
            return {};
        }

        void commitGroupBegin(ID3D12GraphicsCommandList& commandList,
                              detail::DebugCommandState& state,
                              const DebugMarkerDesc& desc,
                              detail::DebugGroupOwner owner) noexcept
        {
#if RACHIS_ENABLE_DEBUG_MARKERS
            emitBegin(commandList, desc);
#else
            static_cast<void>(commandList);
            static_cast<void>(desc);
#endif
            if (state.groupDepth == 0)
            {
                state.groupThread = std::this_thread::get_id();
            }
            state.groupOwners[state.groupDepth] = owner;
            ++state.groupDepth;
        }

        void commitGroupEnd(ID3D12GraphicsCommandList& commandList, detail::DebugCommandState& state) noexcept
        {
#if RACHIS_ENABLE_DEBUG_MARKERS
            PIXEndEvent(&commandList);
#else
            static_cast<void>(commandList);
#endif
            --state.groupDepth;
            if (state.groupDepth == 0)
            {
                state.groupThread = {};
            }
        }

    } // namespace

    Status DebugMarkerManager::beginGroup(QueueManager& queues,
                                          detail::BackendCommandBuffer& commandBuffer,
                                          const DebugMarkerDesc& desc) const
    {
#if !RACHIS_ENABLE_DEBUG_MARKERS
        static_cast<void>(queues);
        static_cast<void>(commandBuffer);
        static_cast<void>(desc);
        return unsupported();
#else
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto valid = validateGroupBegin(commandBuffer.debugState(), desc, detail::DebugGroupOwner::user, true);
        if (!valid)
        {
            return valid;
        }
        commitGroupBegin(**commandList, commandBuffer.debugState(), desc, detail::DebugGroupOwner::user);
        return {};
#endif
    }

    Status DebugMarkerManager::endGroup(QueueManager& queues, detail::BackendCommandBuffer& commandBuffer) const
    {
#if !RACHIS_ENABLE_DEBUG_MARKERS
        static_cast<void>(queues);
        static_cast<void>(commandBuffer);
        return unsupported();
#else
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }

        auto valid = validateGroupEnd(commandBuffer.debugState(), detail::DebugGroupOwner::user, true);
        if (!valid)
        {
            return valid;
        }
        commitGroupEnd(**commandList, commandBuffer.debugState());
        return {};
#endif
    }

    Status DebugMarkerManager::insertMarker(QueueManager& queues,
                                            detail::BackendCommandBuffer& commandBuffer,
                                            const DebugMarkerDesc& desc) const
    {
#if !RACHIS_ENABLE_DEBUG_MARKERS
        static_cast<void>(queues);
        static_cast<void>(commandBuffer);
        static_cast<void>(desc);
        return unsupported();
#else
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto valid = validateDesc(desc);
        if (!valid)
        {
            return valid;
        }

        emitMarker(**commandList, desc);
        return {};
#endif
    }

    Status DebugMarkerManager::validateProfileBegin(detail::BackendCommandBuffer& commandBuffer,
                                                    const DebugMarkerDesc& desc) const
    {
        return validateGroupBegin(commandBuffer.debugState(), desc, detail::DebugGroupOwner::gpuProfile, false);
    }

    void DebugMarkerManager::commitProfileBegin(ID3D12GraphicsCommandList& commandList,
                                                detail::BackendCommandBuffer& commandBuffer,
                                                const DebugMarkerDesc& desc) const noexcept
    {
        commitGroupBegin(commandList, commandBuffer.debugState(), desc, detail::DebugGroupOwner::gpuProfile);
    }

    Status DebugMarkerManager::validateProfileEnd(detail::BackendCommandBuffer& commandBuffer) const
    {
        return validateGroupEnd(commandBuffer.debugState(), detail::DebugGroupOwner::gpuProfile, false);
    }

    void DebugMarkerManager::commitProfileEnd(ID3D12GraphicsCommandList& commandList,
                                              detail::BackendCommandBuffer& commandBuffer) const noexcept
    {
        commitGroupEnd(commandList, commandBuffer.debugState());
    }

    void closeAbandonedDebugGroups(ID3D12GraphicsCommandList& commandList, detail::DebugCommandState& state) noexcept
    {
#if RACHIS_ENABLE_DEBUG_MARKERS
        while (state.groupDepth != 0)
        {
            PIXEndEvent(&commandList);
            --state.groupDepth;
        }
#else
        static_cast<void>(commandList);
        state.groupDepth = 0;
#endif
        state.groupThread = {};
    }

} // namespace rachis::d3d12
