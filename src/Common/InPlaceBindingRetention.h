#pragma once

#include "Rachis/Error.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace rachis::detail
{

    template<typename Element>
    class InPlaceBindingRetention final
    {
        static_assert(std::is_nothrow_move_assignable_v<Element>);
        static_assert(std::is_nothrow_default_constructible_v<Element>);

    public:
        struct Write final
        {
            std::uint32_t index = 0;
            Element element = {};
        };

        InPlaceBindingRetention() = default;
        InPlaceBindingRetention(const InPlaceBindingRetention&) = delete;
        InPlaceBindingRetention& operator=(const InPlaceBindingRetention&) = delete;
        InPlaceBindingRetention(InPlaceBindingRetention&&) = delete;
        InPlaceBindingRetention& operator=(InPlaceBindingRetention&&) = delete;
        ~InPlaceBindingRetention() = default;

        [[nodiscard]] Status reserve(std::span<const Write> writes)
        {
            const std::scoped_lock lock(mutex_);
            try
            {
                for (const auto& write : writes)
                {
                    static_cast<void>(nodes_.try_emplace(write.index));
                }
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to reserve update-after-bind binding retention.",
                });
            }
            return {};
        }

        std::uint64_t record(std::span<Write> writes) noexcept
        {
            const std::scoped_lock lock(mutex_);
            if (writes.empty())
            {
                return sequence_;
            }
            const auto next = sequence_ + 1;
            for (auto& write : writes)
            {
                const auto found = nodes_.find(write.index);
                if (found == nodes_.end())
                {
                    continue;
                }
                auto& node = found->second;
                unlink(node);
                node.element = std::move(write.element);
                node.sequence = next;
                link(node);
            }
            sequence_ = next;
            return next;
        }

        template<typename Visitor>
        void visitSince(std::uint64_t sequence, Visitor&& visitor) const
        {
            const std::scoped_lock lock(mutex_);
            for (const Node* node = newest_; node != nullptr && node->sequence > sequence; node = node->older)
            {
                visitor(node->element);
            }
        }

        [[nodiscard]] bool contains(std::uint32_t index) const
        {
            const std::scoped_lock lock(mutex_);
            const auto found = nodes_.find(index);
            return found != nodes_.end() && found->second.sequence != 0;
        }

        [[nodiscard]] std::uint64_t sequence() const
        {
            const std::scoped_lock lock(mutex_);
            return sequence_;
        }

    private:
        struct Node final
        {
            Element element = {};
            std::uint64_t sequence = 0;
            Node* older = nullptr;
            Node* newer = nullptr;
        };

        void unlink(Node& node) noexcept
        {
            if (node.sequence == 0)
            {
                return;
            }
            if (node.older != nullptr)
            {
                node.older->newer = node.newer;
            }
            if (node.newer != nullptr)
            {
                node.newer->older = node.older;
            }
            else
            {
                newest_ = node.older;
            }
            node.older = nullptr;
            node.newer = nullptr;
        }

        void link(Node& node) noexcept
        {
            node.older = newest_;
            node.newer = nullptr;
            if (newest_ != nullptr)
            {
                newest_->newer = &node;
            }
            newest_ = &node;
        }

        mutable std::mutex mutex_;
        std::unordered_map<std::uint32_t, Node> nodes_;
        Node* newest_ = nullptr;
        std::uint64_t sequence_ = 0;
    };

} // namespace rachis::detail
