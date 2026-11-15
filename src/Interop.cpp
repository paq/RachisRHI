#include "Rachis/Interop.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef NOGDI
#define NOGDI
#endif
#include <Windows.h>
#endif

#include <bit>
#include <utility>

namespace rachis
{

    namespace
    {

        void closeSharedHandle(SharedHandleView handle) noexcept
        {
#ifdef _WIN32
            if (handle.type == SharedHandleType::windowsNt && handle.isValid())
            {
                static_cast<void>(CloseHandle(std::bit_cast<HANDLE>(handle.value)));
            }
#else
            static_cast<void>(handle);
#endif
        }

    } // namespace

    SharedHandle::~SharedHandle()
    {
        reset();
    }

    SharedHandle::SharedHandle(SharedHandle&& other) noexcept : view_(std::exchange(other.view_, {}))
    {
    }

    SharedHandle& SharedHandle::operator=(SharedHandle&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            view_ = std::exchange(other.view_, {});
        }
        return *this;
    }

    SharedHandle SharedHandle::adopt(SharedHandleView handle) noexcept
    {
        return SharedHandle(handle);
    }

    SharedHandleView SharedHandle::release() noexcept
    {
        return std::exchange(view_, {});
    }

    void SharedHandle::reset() noexcept
    {
        closeSharedHandle(std::exchange(view_, {}));
    }

} // namespace rachis
