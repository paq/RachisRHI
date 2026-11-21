#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include "MetalSurfaceTestHost.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

namespace
{

    std::atomic<std::uint32_t> g_probeOffMainAccesses{0};

    void countProbeAccess()
    {
        if (!NSThread.isMainThread)
        {
            g_probeOffMainAccesses.fetch_add(1, std::memory_order_relaxed);
        }
    }

} // namespace

@interface RxMetalTestProbeView : NSView
@end

@implementation RxMetalTestProbeView

- (BOOL)isHidden
{
    countProbeAccess();
    return [super isHidden];
}

- (NSWindow*)window
{
    countProbeAccess();
    return [super window];
}

@end

namespace
{

    constexpr std::int64_t mainQueueStartTimeoutNanoseconds = 5 * static_cast<std::int64_t>(NSEC_PER_SEC);

    enum class MainQueueWork : std::uint8_t
    {
        pending,
        running,
        abandoned,
    };

    std::atomic<bool> gAppKitStartedOnMainThread{false};

    void ensureAppKit()
    {
        static dispatch_once_t s_once = 0;
        dispatch_once(&s_once, ^{
          [NSApplication sharedApplication];
          gAppKitStartedOnMainThread.store(NSThread.isMainThread, std::memory_order_release);
        });
    }

    void runOnMain(void (^work)(void))
    {
        if (NSThread.isMainThread)
        {
            ensureAppKit();
            work();
            return;
        }
        const auto state = std::make_shared<std::atomic<MainQueueWork>>(MainQueueWork::pending);
        dispatch_semaphore_t const finished = dispatch_semaphore_create(0);
        dispatch_async(dispatch_get_main_queue(), ^{
          auto expected = MainQueueWork::pending;
          if (!state->compare_exchange_strong(expected, MainQueueWork::running, std::memory_order_acq_rel))
          {
              return;
          }
          ensureAppKit();
          work();
          static_cast<void>(dispatch_semaphore_signal(finished));
        });
        if (dispatch_semaphore_wait(finished, dispatch_time(DISPATCH_TIME_NOW, mainQueueStartTimeoutNanoseconds)) == 0)
        {
            return;
        }
        auto expected = MainQueueWork::pending;
        if (state->compare_exchange_strong(expected, MainQueueWork::abandoned, std::memory_order_acq_rel))
        {
            static_cast<void>(
                std::fputs("MetalSurfaceTestHost: the main thread did not run its dispatch queue within 5 seconds, "
                           "so the AppKit call from a worker thread was skipped.\n",
                           stderr));
            return;
        }
        static_cast<void>(dispatch_semaphore_wait(finished, DISPATCH_TIME_FOREVER));
    }

    [[nodiscard]] CAMetalLayer* makeLayer(std::uint32_t width, std::uint32_t height)
    {
        CAMetalLayer* const layer = [CAMetalLayer layer];
        layer.drawableSize = CGSizeMake(static_cast<CGFloat>(width), static_cast<CGFloat>(height));
        layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        layer.framebufferOnly = YES;
        return layer;
    }

} // namespace

void* rxMetalTestCreateLayer(std::uint32_t width, std::uint32_t height)
{
    __block void* retained = nullptr;
    runOnMain(^{
      @autoreleasepool
      {
          retained = (__bridge_retained void*)makeLayer(width, height);
      }
    });
    return retained;
}

void* rxMetalTestCreateView(void* layer)
{
    __block void* retained = nullptr;
    runOnMain(^{
      @autoreleasepool
      {
          CAMetalLayer* const metalLayer = (__bridge CAMetalLayer*)layer;
          NSView* const view = [[NSView alloc]
              initWithFrame:NSMakeRect(0, 0, metalLayer.drawableSize.width, metalLayer.drawableSize.height)];
          view.wantsLayer = YES;
          view.layer = metalLayer;
          retained = (__bridge_retained void*)view;
      }
    });
    return retained;
}

void* rxMetalTestCreateUnconfiguredLayer(std::uint32_t width, std::uint32_t height)
{
    __block void* retained = nullptr;
    runOnMain(^{
      @autoreleasepool
      {
          CAMetalLayer* const layer = [CAMetalLayer layer];
          layer.drawableSize = CGSizeMake(static_cast<CGFloat>(width), static_cast<CGFloat>(height));
          layer.device = nil;
          layer.pixelFormat = MTLPixelFormatRGBA16Float;
          layer.framebufferOnly = NO;
          retained = (__bridge_retained void*)layer;
      }
    });
    return retained;
}

RxMetalTestLayerState rxMetalTestQueryLayerState(void* layer)
{
    __block RxMetalTestLayerState state{};
    runOnMain(^{
      @autoreleasepool
      {
          CAMetalLayer* const metalLayer = (__bridge CAMetalLayer*)layer;
          state.hasDevice = metalLayer.device != nil;
          state.bgra8Unorm = metalLayer.pixelFormat == MTLPixelFormatBGRA8Unorm;
          state.rgba16Float = metalLayer.pixelFormat == MTLPixelFormatRGBA16Float;
          state.framebufferOnly = metalLayer.framebufferOnly == YES;
          state.displaySyncEnabled = metalLayer.displaySyncEnabled == YES;
          if (metalLayer.name != nil)
          {
              state.name = metalLayer.name.UTF8String;
          }
      }
    });
    return state;
}

void* rxMetalTestCreateProbeView(void* layer)
{
    g_probeOffMainAccesses.store(0, std::memory_order_relaxed);
    __block void* retained = nullptr;
    runOnMain(^{
      @autoreleasepool
      {
          CAMetalLayer* const metalLayer = (__bridge CAMetalLayer*)layer;
          RxMetalTestProbeView* const view = [[RxMetalTestProbeView alloc]
              initWithFrame:NSMakeRect(0, 0, metalLayer.drawableSize.width, metalLayer.drawableSize.height)];
          view.wantsLayer = YES;
          view.layer = metalLayer;
          retained = (__bridge_retained void*)view;
      }
    });
    return retained;
}

std::uint32_t rxMetalTestProbeViewOffMainAccesses()
{
    return g_probeOffMainAccesses.load(std::memory_order_relaxed);
}

void rxMetalTestDrainMainQueue()
{
    if (!NSThread.isMainThread)
    {
        runOnMain(^{
        });
        return;
    }
    ensureAppKit();
    __block bool drained = false;
    dispatch_async(dispatch_get_main_queue(), ^{
      drained = true;
    });
    NSDate* const deadline = [NSDate dateWithTimeIntervalSinceNow:5.0];
    while (!drained && deadline.timeIntervalSinceNow > 0)
    {
        @autoreleasepool
        {
            CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.01, true);
        }
    }
}

void* rxMetalTestCreateLayerOnOtherDevice(void* nativeMetalDevice, std::uint32_t width, std::uint32_t height)
{
    if (nativeMetalDevice == nullptr)
    {
        return nullptr;
    }
    __block void* retained = nullptr;
    runOnMain(^{
      @autoreleasepool
      {
          id<MTLDevice> const bound = (__bridge id<MTLDevice>)nativeMetalDevice;
          for (id<MTLDevice> device in MTLCopyAllDevices())
          {
              if (device == bound)
              {
                  continue;
              }
              CAMetalLayer* const layer = makeLayer(width, height);
              layer.device = device;
              retained = (__bridge_retained void*)layer;
              return;
          }
      }
    });
    return retained;
}

void rxMetalTestSetDrawableSize(void* layer, std::uint32_t width, std::uint32_t height)
{
    runOnMain(^{
      @autoreleasepool
      {
          CAMetalLayer* const metalLayer = (__bridge CAMetalLayer*)layer;
          metalLayer.drawableSize = CGSizeMake(static_cast<CGFloat>(width), static_cast<CGFloat>(height));
      }
    });
}

void rxMetalTestSetContentsScale(void* layer, double scale)
{
    runOnMain(^{
      @autoreleasepool
      {
          CAMetalLayer* const metalLayer = (__bridge CAMetalLayer*)layer;
          metalLayer.contentsScale = static_cast<CGFloat>(scale);
      }
    });
}

void rxMetalTestSetViewHidden(void* view, bool hidden)
{
    runOnMain(^{
      @autoreleasepool
      {
          NSView* const nsView = (__bridge NSView*)view;
          nsView.hidden = hidden ? YES : NO;
      }
    });
}

std::uint32_t rxMetalTestScreenCount()
{
    __block std::uint32_t count = 0;
    runOnMain(^{
      @autoreleasepool
      {
          count = static_cast<std::uint32_t>(NSScreen.screens.count);
      }
    });
    return count;
}

std::uint32_t rxMetalTestDisplayRefreshHz()
{
    __block std::uint32_t hz = 0;
    runOnMain(^{
      @autoreleasepool
      {
          hz = static_cast<std::uint32_t>(NSScreen.mainScreen.maximumFramesPerSecond);
      }
    });
    return hz;
}

void rxMetalTestRelease(void* object)
{
    if (object == nullptr)
    {
        return;
    }
    runOnMain(^{
      CFRelease(object);
    });
}

void* rxMetalTestCreateNonLayerObject()
{
    __block void* retained = nullptr;
    runOnMain(^{
      @autoreleasepool
      {
          retained = (__bridge_retained void*)[[NSObject alloc] init];
      }
    });
    return retained;
}

bool rxMetalTestSupportsGpuTraceDocument()
{
    __block bool supported = false;
    runOnMain(^{
      @autoreleasepool
      {
          MTLCaptureManager* const manager = [MTLCaptureManager sharedCaptureManager];
          supported = manager != nil && [manager supportsDestination:MTLCaptureDestinationGPUTraceDocument];
      }
    });
    return supported;
}

bool rxMetalTestAppKitStartedOnMainThread()
{
    return gAppKitStartedOnMainThread.load(std::memory_order_acquire);
}

void rxMetalTestPumpMainRunLoop(double seconds)
{
    if (!NSThread.isMainThread)
    {
        return;
    }
    static_cast<void>(CFRunLoopRunInMode(kCFRunLoopDefaultMode, seconds, true));
}
