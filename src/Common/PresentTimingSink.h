#pragma once

#include "Rachis/Presentation.h"

#include <cstddef>
#include <memory>
#include <span>

namespace rachis::detail
{

    class PresentTimingSink final
    {
    public:
        explicit PresentTimingSink(std::span<PresentTiming> records) noexcept
            : PresentTimingSink(records.size(),
                                records.data(),
                                [](void* context, std::size_t index, const PresentTiming& record) noexcept {
                                    static_cast<PresentTiming*>(context)[index] = record;
                                })
        {
        }

        template<typename Writer>
        PresentTimingSink(std::size_t capacity, Writer& writer) noexcept
            : PresentTimingSink(capacity,
                                const_cast<void*>(static_cast<const void*>(std::addressof(writer))),
                                [](void* context, std::size_t index, const PresentTiming& record) noexcept {
                                    (*static_cast<Writer*>(context))(index, record);
                                })
        {
        }

        [[nodiscard]] std::size_t capacity() const noexcept
        {
            return capacity_;
        }

        void write(std::size_t index, const PresentTiming& record) const noexcept
        {
            write_(context_, index, record);
        }

    private:
        using WriteFn = void (*)(void*, std::size_t, const PresentTiming&) noexcept;

        PresentTimingSink(std::size_t capacity, void* context, WriteFn write) noexcept
            : capacity_(capacity), context_(context), write_(write)
        {
        }

        std::size_t capacity_ = 0;
        void* context_ = nullptr;
        WriteFn write_ = nullptr;
    };

} // namespace rachis::detail
