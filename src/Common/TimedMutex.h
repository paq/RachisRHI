#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <type_traits>

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

namespace rachis::detail
{
    using SteadyTimePoint = std::chrono::steady_clock::time_point;

    template<typename Clock, typename Duration>
    [[nodiscard]] SteadyTimePoint toSteadyTimePoint(const std::chrono::time_point<Clock, Duration>& deadline)
    {
        if constexpr (std::is_same_v<Clock, std::chrono::steady_clock>)
        {
            return std::chrono::time_point_cast<std::chrono::steady_clock::duration>(deadline);
        }
        else
        {
            return std::chrono::steady_clock::now() +
                   std::chrono::duration_cast<std::chrono::steady_clock::duration>(deadline - Clock::now());
        }
    }

#if defined(__APPLE__)
#if defined(__OBJC__) && __has_feature(objc_arc)
#define RACHIS_DISPATCH_RETAIN(object) ((__bridge_retained void*)(object))
#define RACHIS_DISPATCH_BORROW(type, raw) ((__bridge type)(raw))
#define RACHIS_DISPATCH_RELEASE(type, raw) static_cast<void>((__bridge_transfer type)(raw))
#else
#define RACHIS_DISPATCH_RETAIN(object) (static_cast<void*>(object))
#define RACHIS_DISPATCH_BORROW(type, raw) (static_cast<type>(raw))
#define RACHIS_DISPATCH_RELEASE(type, raw) dispatch_release(static_cast<type>(raw))
#endif

    class StrictDeadlineWake final
    {
    public:
        StrictDeadlineWake(SteadyTimePoint deadline, std::mutex& mutex, std::condition_variable& changed)
            : mutex_(&mutex), changed_(&changed)
        {
            dispatch_source_t const timer =
                dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER,
                                       0,
                                       DISPATCH_TIMER_STRICT,
                                       dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0));
            dispatch_semaphore_t const cancelled = dispatch_semaphore_create(0);
            if (timer == nullptr || cancelled == nullptr)
            {
#if !(defined(__OBJC__) && __has_feature(objc_arc))
                if (timer != nullptr)
                {
                    dispatch_release(timer);
                }
                if (cancelled != nullptr)
                {
                    dispatch_release(cancelled);
                }
#endif
                return;
            }
            timer_ = RACHIS_DISPATCH_RETAIN(timer);
            cancelled_ = RACHIS_DISPATCH_RETAIN(cancelled);
            const auto left =
                std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - std::chrono::steady_clock::now());
            dispatch_set_context(timer, this);
            dispatch_source_set_event_handler_f(timer, &StrictDeadlineWake::fire);
            dispatch_source_set_cancel_handler_f(timer, &StrictDeadlineWake::finish);
            dispatch_source_set_timer(timer,
                                      dispatch_time(DISPATCH_TIME_NOW, std::max<std::int64_t>(left.count(), 0)),
                                      DISPATCH_TIME_FOREVER,
                                      0);
            dispatch_resume(timer);
        }

        ~StrictDeadlineWake()
        {
            if (timer_ == nullptr)
            {
                return;
            }
            dispatch_source_cancel(RACHIS_DISPATCH_BORROW(dispatch_source_t, timer_));
            static_cast<void>(dispatch_semaphore_wait(RACHIS_DISPATCH_BORROW(dispatch_semaphore_t, cancelled_),
                                                      DISPATCH_TIME_FOREVER));
            RACHIS_DISPATCH_RELEASE(dispatch_source_t, timer_);
            RACHIS_DISPATCH_RELEASE(dispatch_semaphore_t, cancelled_);
        }

        StrictDeadlineWake(const StrictDeadlineWake&) = delete;
        StrictDeadlineWake& operator=(const StrictDeadlineWake&) = delete;
        StrictDeadlineWake(StrictDeadlineWake&&) = delete;
        StrictDeadlineWake& operator=(StrictDeadlineWake&&) = delete;

    private:
        static void fire(void* context) noexcept
        {
            auto& self = *static_cast<StrictDeadlineWake*>(context);
            {
                const std::scoped_lock lock(*self.mutex_);
            }
            self.changed_->notify_all();
        }

        static void finish(void* context) noexcept
        {
            auto& self = *static_cast<StrictDeadlineWake*>(context);
            dispatch_semaphore_signal(RACHIS_DISPATCH_BORROW(dispatch_semaphore_t, self.cancelled_));
        }

        std::mutex* mutex_ = nullptr;
        std::condition_variable* changed_ = nullptr;
        void* timer_ = nullptr;
        void* cancelled_ = nullptr;
    };

#undef RACHIS_DISPATCH_RETAIN
#undef RACHIS_DISPATCH_BORROW
#undef RACHIS_DISPATCH_RELEASE
#endif

    template<typename Done>
    [[nodiscard]] bool strictWaitUntil(std::unique_lock<std::mutex>& lock,
                                       std::condition_variable& changed,
                                       SteadyTimePoint deadline,
                                       Done&& done)
    {
        if (done())
        {
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline)
        {
            return false;
        }
#if defined(__APPLE__)
        std::optional<StrictDeadlineWake> wake;
        wake.emplace(deadline, *lock.mutex(), changed);
#endif
        static_cast<void>(
            changed.wait_until(lock, deadline, [&] { return done() || std::chrono::steady_clock::now() >= deadline; }));
#if defined(__APPLE__)
        lock.unlock();
        wake.reset();
        lock.lock();
#endif
        return done();
    }

    class TimedMutex final
    {
    public:
        TimedMutex() = default;
        ~TimedMutex() = default;

        TimedMutex(const TimedMutex&) = delete;
        TimedMutex& operator=(const TimedMutex&) = delete;
        TimedMutex(TimedMutex&&) = delete;
        TimedMutex& operator=(TimedMutex&&) = delete;

        void lock()
        {
            std::unique_lock lock(mutex_);
            changed_.wait(lock, [&] { return !locked_; });
            locked_ = true;
        }

        [[nodiscard]] bool try_lock()
        {
            const std::scoped_lock lock(mutex_);
            if (locked_)
            {
                return false;
            }
            locked_ = true;
            return true;
        }

        template<typename Clock, typename Duration>
        [[nodiscard]] bool try_lock_until(const std::chrono::time_point<Clock, Duration>& deadline)
        {
            std::unique_lock lock(mutex_);
            if (!strictWaitUntil(lock, changed_, toSteadyTimePoint(deadline), [&] { return !locked_; }))
            {
                return false;
            }
            locked_ = true;
            return true;
        }

        template<typename Rep, typename Period>
        [[nodiscard]] bool try_lock_for(const std::chrono::duration<Rep, Period>& timeout)
        {
            return try_lock_until(std::chrono::steady_clock::now() + timeout);
        }

        void unlock()
        {
            {
                const std::scoped_lock lock(mutex_);
                locked_ = false;
            }
            changed_.notify_all();
        }

    private:
        std::mutex mutex_;
        std::condition_variable changed_;
        bool locked_ = false;
    };
} // namespace rachis::detail
