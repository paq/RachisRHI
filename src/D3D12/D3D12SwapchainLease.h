#pragma once

#include "D3D12/D3D12TestHooks.h"
#include "Internal/TimelinePointAccess.h"
#include "Rachis/Command.h"
#include "Rachis/Error.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <new>
#include <utility>

namespace rachis::d3d12
{

    class SwapchainImageCommandReference;

    class SwapchainImageLeaseState final
    {
    public:
        [[nodiscard]] Status acquire()
        {
            const std::scoped_lock lock(mutex_);
            if (acquired_ || recordedReferenceCount_ != 0 || pendingPublicationCount_ != 0 || submitted_ ||
                submittedPoint_.isValid())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A swapchain image lease cannot be reacquired while it is active or retained by a command buffer.",
                });
            }
            acquired_ = true;
            return {};
        }

        [[nodiscard]] Status cancel()
        {
            const std::scoped_lock lock(mutex_);
            if (!acquired_)
            {
                return std::unexpected(Error{ErrorCode::invalidState, 0, "The swapchain image lease is not acquired."});
            }
            if (recordedReferenceCount_ != 0 || pendingPublicationCount_ != 0 || submitted_)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A recorded or submitted command buffer references the acquired swapchain image.",
                });
            }
            acquired_ = false;
            return {};
        }

        [[nodiscard]] Status validatePresentationPoint(TimelinePoint renderComplete)
        {
            const std::scoped_lock lock(mutex_);
            if (!acquired_)
            {
                return std::unexpected(Error{ErrorCode::invalidState, 0, "The swapchain image lease is not acquired."});
            }
            if (recordedReferenceCount_ != 0 || pendingPublicationCount_ != 0 || !submitted_ ||
                !submittedPoint_.isValid())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Presentation requires the acquired image's completed command publication.",
                });
            }
            if (submittedPoint_ != renderComplete)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Presentation requires the exact timeline point that submitted the acquired image.",
                });
            }
            return {};
        }

        [[nodiscard]] Status beginPresentation(TimelinePoint renderComplete)
        {
            const std::scoped_lock lock(mutex_);
            if (!acquired_ || recordedReferenceCount_ != 0 || pendingPublicationCount_ != 0 || !submitted_ ||
                submittedPoint_ != renderComplete)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Presentation began without the acquired image's exact submitted timeline point.",
                });
            }
            acquired_ = false;
            submitted_ = false;
            submittedPoint_ = {};
            return {};
        }

        [[nodiscard]] Status failPresentation(TimelinePoint renderComplete)
        {
            const std::scoped_lock lock(mutex_);
            if (!acquired_ || recordedReferenceCount_ != 0 || pendingPublicationCount_ != 0 || !submitted_ ||
                submittedPoint_ != renderComplete)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Failed presentation cleanup requires the acquired image's exact submitted timeline point.",
                });
            }
            acquired_ = false;
            submitted_ = false;
            submittedPoint_ = {};
            return {};
        }

        [[nodiscard]] Result<std::shared_ptr<SwapchainImageCommandReference>> retainCommandReference(
            const std::shared_ptr<SwapchainImageLeaseState>& self, QueueType queueType = QueueType::graphics);

    private:
        friend class SwapchainImageCommandReference;

        void releaseRecordedReference() noexcept
        {
            const std::scoped_lock lock(mutex_);
            if (recordedReferenceCount_ != 0)
            {
                --recordedReferenceCount_;
            }
        }

        [[nodiscard]] Status validateSubmission() const
        {
            const std::scoped_lock lock(mutex_);
            if (!acquired_ || recordedReferenceCount_ == 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot submit a swapchain image reference when its lease is not acquired or has no recorded "
                    "reference.",
                });
            }
            if (submittedPoint_.isValid() && submittedPoint_.queueType() != QueueType::graphics)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Swapchain image submissions must target the graphics queue.",
                });
            }
            return {};
        }

        [[nodiscard]] Status markSubmitted()
        {
            const std::scoped_lock lock(mutex_);
            if (!acquired_ || recordedReferenceCount_ == 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot mark a swapchain image reference submitted without an acquired recorded reference.",
                });
            }
            --recordedReferenceCount_;
            ++pendingPublicationCount_;
            submitted_ = true;
            return {};
        }

        [[nodiscard]] Status publishSubmissionPoint(TimelinePoint point)
        {
            const std::scoped_lock lock(mutex_);
            if (!acquired_ || !submitted_ || pendingPublicationCount_ == 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot publish a swapchain submission point without a pending submitted reference.",
                });
            }
            if (!point.isValid() || point.queueType() != QueueType::graphics)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Swapchain image submissions require a valid graphics timeline point.",
                });
            }
            if (submittedPoint_.isValid())
            {
                if (detail::TimelinePointAccess::deviceToken(point) !=
                        detail::TimelinePointAccess::deviceToken(submittedPoint_) ||
                    point.queueType() != submittedPoint_.queueType())
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Swapchain image submissions must originate from the same device and graphics queue.",
                    });
                }
                if (point.value() < submittedPoint_.value())
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "Swapchain image submission timeline points must not regress.",
                    });
                }
            }
            --pendingPublicationCount_;
            submittedPoint_ = point;
            return {};
        }

        mutable std::mutex mutex_;
        std::size_t recordedReferenceCount_ = 0;
        std::size_t pendingPublicationCount_ = 0;
        bool acquired_ = false;
        bool submitted_ = false;
        TimelinePoint submittedPoint_;
    };

    class SwapchainImageCommandReference final
    {
    public:
        explicit SwapchainImageCommandReference(std::shared_ptr<SwapchainImageLeaseState> state)
            : state_(std::move(state))
        {
        }

        ~SwapchainImageCommandReference()
        {
            if (!submitted_ && state_ != nullptr)
            {
                state_->releaseRecordedReference();
            }
        }

        SwapchainImageCommandReference(const SwapchainImageCommandReference&) = delete;
        SwapchainImageCommandReference& operator=(const SwapchainImageCommandReference&) = delete;
        SwapchainImageCommandReference(SwapchainImageCommandReference&&) = delete;
        SwapchainImageCommandReference& operator=(SwapchainImageCommandReference&&) = delete;

        [[nodiscard]] Status validateSubmission(QueueType queueType) const
        {
            if (submitted_)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The swapchain image command reference was already submitted.",
                });
            }
            if (queueType != QueueType::graphics)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Swapchain image command buffers must be submitted on the graphics queue.",
                });
            }
            if (state_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The swapchain image command reference has no lease state.",
                });
            }
            return state_->validateSubmission();
        }

        [[nodiscard]] Status markSubmitted()
        {
            if (submitted_ || state_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The swapchain image command reference cannot be marked submitted.",
                });
            }
            auto marked = state_->markSubmitted();
            if (!marked)
            {
                return marked;
            }
            submitted_ = true;
            return {};
        }

        [[nodiscard]] Status publishSubmissionPoint(TimelinePoint point)
        {
            if (!submitted_ || published_ || state_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The swapchain image command reference is not awaiting timeline publication.",
                });
            }
            auto published = state_->publishSubmissionPoint(point);
            if (!published)
            {
                return published;
            }
            published_ = true;
            return {};
        }

    private:
        std::shared_ptr<SwapchainImageLeaseState> state_;
        bool submitted_ = false;
        bool published_ = false;
    };

    inline Result<std::shared_ptr<SwapchainImageCommandReference>> SwapchainImageLeaseState::retainCommandReference(
        const std::shared_ptr<SwapchainImageLeaseState>& self, QueueType queueType)
    {
        if (queueType != QueueType::graphics)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Swapchain images can only be recorded on graphics command buffers.",
            });
        }
        auto failureCountdown = test::swapchainLeaseRetentionFailureCountdown().load(std::memory_order_relaxed);
        while (failureCountdown > 0)
        {
            if (test::swapchainLeaseRetentionFailureCountdown().compare_exchange_weak(
                    failureCountdown, failureCountdown - 1, std::memory_order_relaxed))
            {
                if (failureCountdown == 1)
                {
                    return std::unexpected(Error{
                        ErrorCode::outOfMemory,
                        0,
                        "Injected swapchain image lease retention failure.",
                    });
                }
                break;
            }
        }
        {
            const std::scoped_lock lock(mutex_);
            if (!acquired_)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A swapchain image can be recorded only while its current lease is acquired.",
                });
            }
            ++recordedReferenceCount_;
        }

        try
        {
            return std::make_shared<SwapchainImageCommandReference>(self);
        }
        catch (const std::bad_alloc&)
        {
            releaseRecordedReference();
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to retain a swapchain image command reference.",
            });
        }
    }

} // namespace rachis::d3d12
