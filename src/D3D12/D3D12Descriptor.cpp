#include "D3D12/D3D12Descriptor.h"

#include "D3D12/D3D12NativeHelpers.h"

#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <ranges>
#include <string_view>
#include <utility>
#include <vector>

namespace rachis::d3d12
{
    namespace
    {

        using Microsoft::WRL::ComPtr;

        constexpr std::uint32_t descriptorsPerPage = 256;

    } // namespace

    struct CpuDescriptorPage final
    {
        ComPtr<ID3D12DescriptorHeap> heap;
        D3D12_CPU_DESCRIPTOR_HANDLE start{};
        std::uint32_t increment = 0;
        mutable std::mutex mutex;
        std::uint32_t next = 0;
        std::vector<std::uint32_t> freeIndices;

        [[nodiscard]] bool hasCapacity() const
        {
            const std::scoped_lock lock(mutex);
            return !freeIndices.empty() || next < descriptorsPerPage;
        }

        [[nodiscard]] std::uint32_t allocate()
        {
            const std::scoped_lock lock(mutex);
            if (!freeIndices.empty())
            {
                const auto index = freeIndices.back();
                freeIndices.pop_back();
                return index;
            }
            return next++;
        }

        void release(std::uint32_t index)
        {
            const std::scoped_lock lock(mutex);
            freeIndices.push_back(index);
        }

        [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE handle(std::uint32_t index) const noexcept
        {
            return D3D12_CPU_DESCRIPTOR_HANDLE{
                start.ptr + static_cast<std::size_t>(index) * increment,
            };
        }
    };

    CpuDescriptorLease::CpuDescriptorLease(std::shared_ptr<CpuDescriptorPage> page, std::uint32_t index)
        : page_(std::move(page)), index_(index)
    {
    }

    CpuDescriptorLease::~CpuDescriptorLease()
    {
        if (page_ != nullptr)
        {
            page_->release(index_);
        }
    }

    D3D12_CPU_DESCRIPTOR_HANDLE CpuDescriptorLease::handle() const noexcept
    {
        return page_->handle(index_);
    }

    struct CpuDescriptorPool::Impl final
    {
        explicit Impl(D3D12_DESCRIPTOR_HEAP_TYPE descriptorType) : type(descriptorType)
        {
        }

        D3D12_DESCRIPTOR_HEAP_TYPE type;
        mutable std::mutex mutex;
        std::vector<std::shared_ptr<CpuDescriptorPage>> pages;
        std::size_t cursor = 0;
    };

    CpuDescriptorPool::CpuDescriptorPool(D3D12_DESCRIPTOR_HEAP_TYPE type) : impl_(std::make_unique<Impl>(type))
    {
    }

    CpuDescriptorPool::~CpuDescriptorPool() = default;

    Result<std::shared_ptr<CpuDescriptorLease>> CpuDescriptorPool::allocate(ID3D12Device& device)
    {
        const std::scoped_lock lock(impl_->mutex);
        const auto available = [](const auto& candidate) { return candidate->hasCapacity(); };
        const auto start = impl_->pages.begin() + static_cast<std::ptrdiff_t>(impl_->cursor);
        auto page = std::ranges::find_if(start, impl_->pages.end(), available);
        if (page == impl_->pages.end())
        {
            page = std::ranges::find_if(impl_->pages.begin(), start, available);
            if (page == start)
            {
                page = impl_->pages.end();
            }
        }
        if (page == impl_->pages.end())
        {
            auto created = std::make_shared<CpuDescriptorPage>();
            const D3D12_DESCRIPTOR_HEAP_DESC heapDesc{
                .Type = impl_->type,
                .NumDescriptors = descriptorsPerPage,
                .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                .NodeMask = 0,
            };
            const HRESULT result = device.CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&created->heap));
            if (FAILED(result))
            {
                return std::unexpected(
                    nativeError(result == E_OUTOFMEMORY ? ErrorCode::outOfMemory : ErrorCode::nativeFailure,
                                result,
                                "ID3D12Device::CreateDescriptorHeap"));
            }
            created->start = created->heap->GetCPUDescriptorHandleForHeapStart();
            created->increment = device.GetDescriptorHandleIncrementSize(impl_->type);
            impl_->pages.push_back(std::move(created));
            page = std::prev(impl_->pages.end());
        }

        impl_->cursor = static_cast<std::size_t>(page - impl_->pages.begin());
        const auto index = (*page)->allocate();
        return std::make_shared<CpuDescriptorLease>(*page, index);
    }

    DescriptorAllocationStatistics CpuDescriptorPool::statistics() const noexcept
    {
        const std::scoped_lock poolLock(impl_->mutex);
        DescriptorAllocationStatistics result{};
        for (const auto& page : impl_->pages)
        {
            const std::scoped_lock pageLock(page->mutex);
            result.liveCount +=
                static_cast<std::uint64_t>(page->next) - static_cast<std::uint64_t>(page->freeIndices.size());
            result.reservedCount += descriptorsPerPage;
        }
        return result;
    }

    struct GpuDescriptorLease::State final
    {
        ComPtr<ID3D12DescriptorHeap> heap;
        D3D12_CPU_DESCRIPTOR_HANDLE cpuStart{};
        D3D12_GPU_DESCRIPTOR_HANDLE gpuStart{};
        std::uint32_t increment = 0;
        std::uint32_t capacity = 0;
        std::mutex mutex;
        std::map<std::uint32_t, std::uint32_t> freeRanges;

        void release(std::uint32_t offset, std::uint32_t count)
        {
            const std::scoped_lock lock(mutex);
            auto next = freeRanges.lower_bound(offset);
            if (next != freeRanges.begin())
            {
                auto previous = std::prev(next);
                if (previous->first + previous->second == offset)
                {
                    offset = previous->first;
                    count += previous->second;
                    freeRanges.erase(previous);
                }
            }
            if (next != freeRanges.end() && offset + count == next->first)
            {
                count += next->second;
                freeRanges.erase(next);
            }
            freeRanges.emplace(offset, count);
        }
    };

    GpuDescriptorLease::GpuDescriptorLease(std::shared_ptr<State> state, std::uint32_t offset, std::uint32_t count)
        : state_(std::move(state)), offset_(offset), count_(count)
    {
    }

    GpuDescriptorLease::~GpuDescriptorLease()
    {
        if (state_ != nullptr && count_ != 0)
        {
            state_->release(offset_, count_);
        }
    }

    std::uint32_t GpuDescriptorLease::count() const noexcept
    {
        return count_;
    }

    D3D12_CPU_DESCRIPTOR_HANDLE GpuDescriptorLease::cpuHandle(std::uint32_t index) const noexcept
    {
        return D3D12_CPU_DESCRIPTOR_HANDLE{
            state_->cpuStart.ptr + static_cast<std::size_t>(offset_ + index) * state_->increment,
        };
    }

    D3D12_GPU_DESCRIPTOR_HANDLE GpuDescriptorLease::gpuHandle(std::uint32_t index) const noexcept
    {
        return D3D12_GPU_DESCRIPTOR_HANDLE{
            state_->gpuStart.ptr + static_cast<std::uint64_t>(offset_ + index) * state_->increment,
        };
    }

    GpuDescriptorHeap::GpuDescriptorHeap() = default;

    GpuDescriptorHeap::~GpuDescriptorHeap() = default;

    Status GpuDescriptorHeap::initialize(ID3D12Device& device,
                                         D3D12_DESCRIPTOR_HEAP_TYPE type,
                                         std::uint32_t capacity,
                                         std::uint32_t transientCapacity,
                                         std::uint32_t bindTableCapacity)
    {
        if (capacity == 0 || transientCapacity > std::numeric_limits<std::uint32_t>::max() - capacity ||
            bindTableCapacity > std::numeric_limits<std::uint32_t>::max() - capacity - transientCapacity ||
            (type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV && type != D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A shader-visible descriptor heap requires a non-zero supported capacity.",
            });
        }

        auto state = std::make_shared<GpuDescriptorLease::State>();
        const D3D12_DESCRIPTOR_HEAP_DESC desc{
            .Type = type,
            .NumDescriptors = capacity + transientCapacity + bindTableCapacity,
            .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            .NodeMask = 0,
        };
        const HRESULT result = device.CreateDescriptorHeap(&desc, IID_PPV_ARGS(&state->heap));
        if (FAILED(result))
        {
            return std::unexpected(
                nativeError(result == E_OUTOFMEMORY ? ErrorCode::outOfMemory : ErrorCode::nativeFailure,
                            result,
                            "ID3D12Device::CreateDescriptorHeap(shader-visible)"));
        }
        state->cpuStart = state->heap->GetCPUDescriptorHandleForHeapStart();
        state->gpuStart = state->heap->GetGPUDescriptorHandleForHeapStart();
        state->increment = device.GetDescriptorHandleIncrementSize(type);
        state->capacity = capacity;
        state->freeRanges.emplace(0, capacity);

        const auto makeRange = [&state](std::uint32_t firstDescriptor, std::uint32_t rangeCapacity) {
            std::shared_ptr<GpuDescriptorLease::State> range;
            if (rangeCapacity == 0)
            {
                return range;
            }
            range = std::make_shared<GpuDescriptorLease::State>();
            range->heap = state->heap;
            range->increment = state->increment;
            range->cpuStart = D3D12_CPU_DESCRIPTOR_HANDLE{
                state->cpuStart.ptr + static_cast<std::size_t>(firstDescriptor) * state->increment,
            };
            range->gpuStart = D3D12_GPU_DESCRIPTOR_HANDLE{
                state->gpuStart.ptr + static_cast<std::uint64_t>(firstDescriptor) * state->increment,
            };
            range->capacity = rangeCapacity;
            range->freeRanges.emplace(0, rangeCapacity);
            return range;
        };
        auto transientState = makeRange(capacity, transientCapacity);
        auto bindTableState = makeRange(capacity + transientCapacity, bindTableCapacity);
        state_ = std::move(state);
        transientState_ = std::move(transientState);
        bindTableState_ = std::move(bindTableState);
        return {};
    }

    Result<std::shared_ptr<GpuDescriptorLease>> GpuDescriptorHeap::allocate(std::uint32_t count)
    {
        return allocateFrom(state_, count, "The shader-visible D3D12 descriptor heap is exhausted or fragmented.");
    }

    Result<std::shared_ptr<GpuDescriptorLease>> GpuDescriptorHeap::allocateTransient(std::uint32_t count)
    {
        return allocateFrom(transientState_,
                            count,
                            "The transient range of the shader-visible D3D12 descriptor heap is exhausted or "
                            "fragmented.");
    }

    Result<std::shared_ptr<GpuDescriptorLease>> GpuDescriptorHeap::allocateBindTablePage(std::uint32_t count)
    {
        return allocateFrom(bindTableState_,
                            count,
                            "The dynamic-offset table range of the shader-visible D3D12 descriptor heap is exhausted.");
    }

    Result<std::shared_ptr<GpuDescriptorLease>> GpuDescriptorHeap::allocateFrom(
        const std::shared_ptr<GpuDescriptorLease::State>& state, std::uint32_t count, const char* exhaustedMessage)
    {
        if (state == nullptr || count == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A shader-visible descriptor allocation requires an initialized heap and non-zero size.",
            });
        }

        const std::scoped_lock lock(state->mutex);
        const auto found =
            std::ranges::find_if(state->freeRanges, [count](const auto& range) { return range.second >= count; });
        if (found == state->freeRanges.end())
        {
            return std::unexpected(Error{
                ErrorCode::resourceExhausted,
                0,
                exhaustedMessage,
            });
        }

        const auto offset = found->first;
        const auto remaining = found->second - count;
        state->freeRanges.erase(found);
        if (remaining != 0)
        {
            state->freeRanges.emplace(offset + count, remaining);
        }
        return std::shared_ptr<GpuDescriptorLease>(new GpuDescriptorLease(state, offset, count));
    }

    ID3D12DescriptorHeap* GpuDescriptorHeap::nativeHeap() const noexcept
    {
        return state_ != nullptr ? state_->heap.Get() : nullptr;
    }

    DescriptorAllocationStatistics GpuDescriptorHeap::statistics() const noexcept
    {
        DescriptorAllocationStatistics result{};
        for (auto* state : std::array{state_.get(), transientState_.get(), bindTableState_.get()})
        {
            if (state == nullptr)
            {
                continue;
            }
            const std::scoped_lock lock(state->mutex);
            std::uint64_t freeCount = 0;
            for (const auto& range : state->freeRanges)
            {
                freeCount += range.second;
            }
            result.liveCount += state->capacity - freeCount;
            result.reservedCount += state->capacity;
        }
        return result;
    }

} // namespace rachis::d3d12
