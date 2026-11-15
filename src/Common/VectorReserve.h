#pragma once

#include <algorithm>
#include <cstddef>
#include <new>
#include <vector>

namespace rachis::detail
{

    template<typename T, typename Allocator>
    inline void reserveAdditional(std::vector<T, Allocator>& vec, std::size_t additional)
    {
        if (additional == 0)
        {
            return;
        }
        const std::size_t currentSize = vec.size();
        const std::size_t maxSize = vec.max_size();
        if (additional > maxSize - currentSize)
        {
            throw std::bad_alloc{};
        }
        const std::size_t needed = currentSize + additional;
        const std::size_t currentCapacity = vec.capacity();
        if (needed <= currentCapacity)
        {
            return;
        }
        constexpr std::size_t kMinInitialCapacity = 8U;
        const std::size_t doubled = currentCapacity == 0
                                        ? kMinInitialCapacity
                                        : (currentCapacity <= maxSize / 2U ? currentCapacity * 2U : maxSize);
        vec.reserve((std::max)(needed, doubled));
    }

} // namespace rachis::detail
