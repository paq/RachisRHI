#pragma once

#include <cstddef>

#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif

namespace rachis::test
{

    class ThreadHeapAllocationCounter final
    {
    public:
        [[nodiscard]] static constexpr bool available() noexcept
        {
#if defined(_MSC_VER) && defined(_DEBUG)
            return true;
#else
            return false;
#endif
        }

        ThreadHeapAllocationCounter() noexcept
        {
#if defined(_MSC_VER) && defined(_DEBUG)
            previousHook_ = _CrtSetAllocHook(&countAllocation);
#endif
        }

        ~ThreadHeapAllocationCounter() noexcept
        {
#if defined(_MSC_VER) && defined(_DEBUG)
            _CrtSetAllocHook(previousHook_);
#endif
        }

        ThreadHeapAllocationCounter(const ThreadHeapAllocationCounter&) = delete;
        ThreadHeapAllocationCounter& operator=(const ThreadHeapAllocationCounter&) = delete;
        ThreadHeapAllocationCounter(ThreadHeapAllocationCounter&&) = delete;
        ThreadHeapAllocationCounter& operator=(ThreadHeapAllocationCounter&&) = delete;

        template<typename Function>
        [[nodiscard]] std::size_t count(Function&& function)
        {
            allocations() = 0;
            counting() = true;
            function();
            counting() = false;
            return allocations();
        }

    private:
        [[nodiscard]] static bool& counting() noexcept
        {
            static thread_local bool s_counting = false;
            return s_counting;
        }

        [[nodiscard]] static std::size_t& allocations() noexcept
        {
            static thread_local std::size_t s_allocations = 0;
            return s_allocations;
        }

#if defined(_MSC_VER) && defined(_DEBUG)
        static int __cdecl countAllocation(int allocationType,
                                           void* userData,
                                           std::size_t size,
                                           int blockType,
                                           long requestNumber,
                                           const unsigned char* fileName,
                                           int lineNumber)
        {
            static_cast<void>(userData);
            static_cast<void>(size);
            static_cast<void>(requestNumber);
            static_cast<void>(fileName);
            static_cast<void>(lineNumber);
            if (blockType != _CRT_BLOCK && (allocationType == _HOOK_ALLOC || allocationType == _HOOK_REALLOC) &&
                counting())
            {
                ++allocations();
            }
            return 1;
        }

        _CRT_ALLOC_HOOK previousHook_ = nullptr;
#endif
    };

} // namespace rachis::test
