#include "Common/TimedMutex.h"

#if defined(_WIN32)
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

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

namespace rachis::detail
{
    void fireStrictDeadlineWake(StrictDeadlineWake& self) noexcept
    {
        {
            const std::scoped_lock lock(*self.mutex_);
        }
        self.changed_->notify_all();
    }

    namespace
    {
        void CALLBACK onStrictDeadlineWake(PTP_CALLBACK_INSTANCE, void* context, PTP_WAIT, TP_WAIT_RESULT) noexcept
        {
            fireStrictDeadlineWake(*static_cast<StrictDeadlineWake*>(context));
        }
    } // namespace

    StrictDeadlineWake::StrictDeadlineWake(SteadyTimePoint deadline,
                                           std::mutex& mutex,
                                           std::condition_variable& changed) noexcept
        : mutex_(&mutex), changed_(&changed)
    {
        HANDLE const timer =
            CreateWaitableTimerExW(nullptr,
                                   nullptr,
                                   CREATE_WAITABLE_TIMER_MANUAL_RESET | CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                   TIMER_ALL_ACCESS);
        if (timer == nullptr)
        {
            return;
        }
        PTP_WAIT const wait = CreateThreadpoolWait(&onStrictDeadlineWake, this, nullptr);
        if (wait == nullptr)
        {
            static_cast<void>(CloseHandle(timer));
            return;
        }
        const auto left =
            std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - std::chrono::steady_clock::now());
        const std::int64_t leftNs = std::max<std::int64_t>(left.count(), 1);
        LARGE_INTEGER dueTime{};
        dueTime.QuadPart = -((leftNs / 100) + ((leftNs % 100) != 0 ? 1 : 0));
        SetThreadpoolWait(wait, timer, nullptr);
        if (SetWaitableTimer(timer, &dueTime, 0, nullptr, nullptr, FALSE) == FALSE)
        {
            SetThreadpoolWait(wait, nullptr, nullptr);
            WaitForThreadpoolWaitCallbacks(wait, TRUE);
            CloseThreadpoolWait(wait);
            static_cast<void>(CloseHandle(timer));
            return;
        }
        timer_ = timer;
        wait_ = wait;
    }

    StrictDeadlineWake::~StrictDeadlineWake()
    {
        if (wait_ != nullptr)
        {
            auto* const wait = static_cast<PTP_WAIT>(wait_);
            SetThreadpoolWait(wait, nullptr, nullptr);
            WaitForThreadpoolWaitCallbacks(wait, TRUE);
            CloseThreadpoolWait(wait);
        }
        if (timer_ != nullptr)
        {
            auto* const timer = static_cast<HANDLE>(timer_);
            static_cast<void>(CancelWaitableTimer(timer));
            static_cast<void>(CloseHandle(timer));
        }
    }
} // namespace rachis::detail
#endif
