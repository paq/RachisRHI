#pragma once

#include <cstdint>
#include <string>

struct RxMetalTestLayerState final
{
    bool hasDevice = false;
    bool bgra8Unorm = false;
    bool rgba16Float = false;
    bool framebufferOnly = false;
    std::string name;
};

[[nodiscard]] void* rxMetalTestCreateLayer(std::uint32_t width, std::uint32_t height);
[[nodiscard]] void* rxMetalTestCreateView(void* layer);
[[nodiscard]] void* rxMetalTestCreateUnconfiguredLayer(std::uint32_t width, std::uint32_t height);
[[nodiscard]] RxMetalTestLayerState rxMetalTestQueryLayerState(void* layer);
[[nodiscard]] void* rxMetalTestCreateProbeView(void* layer);
[[nodiscard]] std::uint32_t rxMetalTestProbeViewOffMainAccesses();
void rxMetalTestDrainMainQueue();
[[nodiscard]] void* rxMetalTestCreateLayerOnOtherDevice(void* nativeMetalDevice,
                                                        std::uint32_t width,
                                                        std::uint32_t height);
void rxMetalTestSetDrawableSize(void* layer, std::uint32_t width, std::uint32_t height);
void rxMetalTestSetContentsScale(void* layer, double scale);
void rxMetalTestSetViewHidden(void* view, bool hidden);
[[nodiscard]] std::uint32_t rxMetalTestScreenCount();
[[nodiscard]] std::uint32_t rxMetalTestDisplayRefreshHz();
void rxMetalTestRelease(void* object);
[[nodiscard]] void* rxMetalTestCreateNonLayerObject();
[[nodiscard]] bool rxMetalTestSupportsGpuTraceDocument();
[[nodiscard]] bool rxMetalTestAppKitStartedOnMainThread();
void rxMetalTestPumpMainRunLoop(double seconds);
