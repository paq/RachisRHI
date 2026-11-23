#include "C/CAbiDeviceMutex.h"
#include "Common/TimedMutex.h"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <condition_variable>
#include <functional>
#include <latch>
#include <mutex>
#include <thread>

namespace
{
    using Clock = std::chrono::steady_clock;
    using namespace std::chrono_literals;

    struct InjectedTeardownWake final
    {
        static inline std::function<void()> onWaiting;
        static inline std::function<void()> onTeardown;

        InjectedTeardownWake(rachis::detail::SteadyTimePoint, std::mutex&, std::condition_variable&)
        {
            if (onWaiting)
            {
                onWaiting();
            }
        }

        ~InjectedTeardownWake()
        {
            if (onTeardown)
            {
                onTeardown();
            }
        }

        InjectedTeardownWake(const InjectedTeardownWake&) = delete;
        InjectedTeardownWake& operator=(const InjectedTeardownWake&) = delete;
        InjectedTeardownWake(InjectedTeardownWake&&) = delete;
        InjectedTeardownWake& operator=(InjectedTeardownWake&&) = delete;
    };

    struct InjectedTeardownScope final
    {
        InjectedTeardownScope(std::function<void()> waiting, std::function<void()> teardown)
        {
            InjectedTeardownWake::onWaiting = std::move(waiting);
            InjectedTeardownWake::onTeardown = std::move(teardown);
        }

        ~InjectedTeardownScope()
        {
            InjectedTeardownWake::onWaiting = nullptr;
            InjectedTeardownWake::onTeardown = nullptr;
        }

        InjectedTeardownScope(const InjectedTeardownScope&) = delete;
        InjectedTeardownScope& operator=(const InjectedTeardownScope&) = delete;
        InjectedTeardownScope(InjectedTeardownScope&&) = delete;
        InjectedTeardownScope& operator=(InjectedTeardownScope&&) = delete;
    };

    using TestTimedMutex = rachis::detail::BasicTimedMutex<InjectedTeardownWake>;
    using TestCAbiDeviceMutex = rachis::cabi::detail::BasicCAbiDeviceMutex<InjectedTeardownWake>;
} // namespace

TEST_CASE("TimedMutex keeps a lock released before the deadline when another caller tries to take it during "
          "wake teardown",
          "[size:small][rhi][concurrency]")
{
    TestTimedMutex mutex;
    mutex.lock();
    std::latch waiting(1);
    bool takenDuringTeardown = false;
    const InjectedTeardownScope scope([&] { waiting.count_down(); },
                                      [&] {
                                          takenDuringTeardown = mutex.try_lock();
                                          if (takenDuringTeardown)
                                          {
                                              mutex.unlock();
                                          }
                                      });

    const auto deadline = Clock::now() + 10s;
    bool acquired = false;
    Clock::time_point returned;
    std::thread waiter([&] {
        acquired = mutex.try_lock_until(deadline);
        returned = Clock::now();
    });
    waiting.wait();
    mutex.unlock();
    waiter.join();

    CHECK(acquired);
    CHECK_FALSE(takenDuringTeardown);
    CHECK((acquired || returned >= deadline));
    CHECK_FALSE(mutex.try_lock());
    mutex.unlock();
}

TEST_CASE("TimedMutex does not grant a lock released during wake teardown after the deadline has passed",
          "[size:small][rhi][concurrency]")
{
    TestTimedMutex mutex;
    mutex.lock();
    const InjectedTeardownScope scope(nullptr, [&] { mutex.unlock(); });

    const auto deadline = Clock::now() + 20ms;
    bool acquired = true;
    Clock::time_point returned;
    std::thread waiter([&] {
        acquired = mutex.try_lock_until(deadline);
        returned = Clock::now();
    });
    waiter.join();

    CHECK_FALSE(acquired);
    CHECK(returned >= deadline);
    CHECK(mutex.try_lock());
    mutex.unlock();
}

TEST_CASE("CAbiDeviceMutex keeps an admission made before the deadline through wake teardown",
          "[size:small][rhi][concurrency]")
{
    TestCAbiDeviceMutex mutex;
    mutex.lock();
    std::latch waiting(1);
    bool takenDuringTeardown = false;
    const InjectedTeardownScope scope([&] { waiting.count_down(); },
                                      [&] {
                                          takenDuringTeardown = mutex.try_lock();
                                          if (takenDuringTeardown)
                                          {
                                              mutex.unlock();
                                          }
                                      });

    const auto deadline = Clock::now() + 10s;
    bool acquired = false;
    std::thread waiter([&] { acquired = mutex.try_lock_until(deadline); });
    waiting.wait();
    mutex.unlock();
    waiter.join();

    CHECK(acquired);
    CHECK_FALSE(takenDuringTeardown);
    CHECK_FALSE(mutex.try_lock_shared());
    mutex.unlock();
}

TEST_CASE("CAbiDeviceMutex does not admit a timed-out waiter when the lock is released during wake teardown",
          "[size:small][rhi][concurrency]")
{
    TestCAbiDeviceMutex mutex;
    mutex.lock();
    const InjectedTeardownScope scope(nullptr, [&] { mutex.unlock(); });

    const auto deadline = Clock::now() + 20ms;
    bool acquired = true;
    Clock::time_point returned;
    std::thread waiter([&] {
        acquired = mutex.try_lock_until(deadline);
        returned = Clock::now();
    });
    waiter.join();

    CHECK_FALSE(acquired);
    CHECK(returned >= deadline);
    REQUIRE(mutex.try_lock());
    mutex.unlock();
    REQUIRE(mutex.try_lock_shared());
    mutex.unlock_shared();
}

TEST_CASE("CAbiDeviceMutex does not admit a timed-out shared waiter when the lock is released during wake teardown",
          "[size:small][rhi][concurrency]")
{
    TestCAbiDeviceMutex mutex;
    mutex.lock();
    const InjectedTeardownScope scope(nullptr, [&] { mutex.unlock(); });

    const auto deadline = Clock::now() + 20ms;
    bool acquired = true;
    Clock::time_point returned;
    std::thread waiter([&] {
        acquired = mutex.try_lock_shared_until(deadline);
        returned = Clock::now();
    });
    waiter.join();

    CHECK_FALSE(acquired);
    CHECK(returned >= deadline);
    REQUIRE(mutex.try_lock());
    mutex.unlock();
}
