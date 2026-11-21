#include "Metal/MetalInternal.h"

namespace rachis::metal
{

    namespace
    {

        [[nodiscard]] bool metalReadViewHidden(id view) noexcept
        {
            if (view == nil)
            {
                return false;
            }
            if ([[view valueForKey:@"hidden"] boolValue])
            {
                return true;
            }
            id const window = [view valueForKey:@"window"];
            if (window == nil)
            {
                return false;
            }

            const id visible = [window valueForKey:@"visible"];
            return visible != nil && ![visible boolValue];
        }

        void refreshMetalViewVisibility(const std::shared_ptr<MetalViewVisibility>& visibility) noexcept
        {
            if (visibility == nullptr)
            {
                return;
            }
            if ([NSThread isMainThread])
            {
                @autoreleasepool
                {
                    id const view = visibility->view;
                    visibility->hidden.store(metalReadViewHidden(view), std::memory_order_release);
                }
                return;
            }
            if (visibility->refreshQueued.exchange(true, std::memory_order_acq_rel))
            {
                return;
            }
            const std::shared_ptr<MetalViewVisibility> pending = visibility;
            dispatch_async(dispatch_get_main_queue(), ^{
              pending->refreshQueued.store(false, std::memory_order_release);
              refreshMetalViewVisibility(pending);
            });
        }

    } // namespace

    Result<SurfaceHandle> MetalDevice::createSurface(const SurfaceDesc& desc)
    {
        if (desc.kind != SurfaceKind::caMetalLayer)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::unsupported,
                                    "Metal surfaces require SurfaceKind::caMetalLayer. Selection did not fall back."));
        }
        if (desc.windowHandle == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "A Metal surface requires a host-owned CAMetalLayer handle."));
        }

        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal createSurface requires an active device."));
        }

        @autoreleasepool
        {
            id const layerObject = (__bridge id)desc.windowHandle;
            if (![layerObject isKindOfClass:[CAMetalLayer class]])
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "The Metal surface window handle is not a CAMetalLayer."));
            }
            CAMetalLayer* const layer = (CAMetalLayer*)layerObject;
            if (layer.device != nil && layer.device != native_->device)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "The CAMetalLayer is bound to a different Metal device."));
            }

            std::shared_ptr<MetalViewVisibility> visibility;
            if (desc.displayHandle != nullptr)
            {
                Class const viewClass = NSClassFromString(@"NSView");
                id const viewObject = (__bridge id)desc.displayHandle;
                if (viewClass == nil || ![viewObject isKindOfClass:viewClass])
                {
                    return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                               "The Metal surface display handle is not an NSView."));
                }
                visibility = std::make_shared<MetalViewVisibility>();
                visibility->view = viewObject;
            }

            auto record = std::make_shared<MetalSurfaceRecord>();
            record->layer = layer;
            record->viewVisibility = visibility;
            record->label = std::string(desc.label);
            auto handle = native_->surfaces.emplace(std::move(record));
            if (!handle)
            {
                return std::unexpected(std::move(handle.error()));
            }
            refreshMetalViewVisibility(visibility);
            return handle;
        }
    }

    namespace
    {

        [[nodiscard]] bool metalSurfaceIsHidden(const MetalSurfaceRecord& surface) noexcept
        {
            if (surface.layer == nil)
            {
                return true;
            }
            const auto size = surface.layer.drawableSize;
            if (!(size.width >= 1.0) || !(size.height >= 1.0))
            {
                return true;
            }
            if (surface.viewVisibility == nullptr)
            {
                return false;
            }
            refreshMetalViewVisibility(surface.viewVisibility);
            return surface.viewVisibility->hidden.load(std::memory_order_acquire);
        }

        [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> metalLayerBackingPixels(CAMetalLayer* layer) noexcept
        {
            if (layer == nil)
            {
                return {0, 0};
            }
            const auto size = layer.drawableSize;
            return {
                size.width > 0.0 ? static_cast<std::uint32_t>(size.width) : 0,
                size.height > 0.0 ? static_cast<std::uint32_t>(size.height) : 0,
            };
        }

        void applyMetalPresentMode(CAMetalLayer* layer, PresentMode mode) noexcept
        {
            layer.displaySyncEnabled = mode == PresentMode::immediate ? NO : YES;
        }

        void applyMetalSwapchainLayer(CAMetalLayer* layer, const SwapchainDesc& desc) noexcept
        {
            layer.pixelFormat = metalPixelFormat(desc.format);
            layer.framebufferOnly = hasAny(desc.usage, TextureUsage::copySource) ? NO : YES;
            layer.opaque = YES;
            layer.drawableSize = CGSizeMake(static_cast<CGFloat>(desc.width), static_cast<CGFloat>(desc.height));
            layer.allowsNextDrawableTimeout = YES;
            applyMetalPresentMode(layer, desc.presentMode);
            auto* const srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
            if (srgb != nullptr)
            {
                layer.colorspace = srgb;
                CGColorSpaceRelease(srgb);
            }
        }

    } // namespace

    Result<SurfaceCapabilities> MetalDevice::querySurfaceCapabilities(SurfaceHandle surface) const
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "Metal querySurfaceCapabilities requires an active device."));
        }
        const auto* found = native_->surfaces.find(surface);
        if (found == nullptr || *found == nullptr || (*found)->layer == nil)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The surface handle is not a live Metal CAMetalLayer."));
        }

        @autoreleasepool
        {
            CAMetalLayer* const layer = (*found)->layer;
            const auto maxDim = capabilities_.limits.maxTextureDimension2D;
            const bool hidden = metalSurfaceIsHidden(**found);
            const auto backing = metalLayerBackingPixels(layer);
            bool suspended = false;
            native_->swapchains.forEach([&](SwapchainHandle, std::shared_ptr<MetalSwapchainRecord>& record) {
                if (record != nullptr && record->surface == surface &&
                    (record->desc.width == 0 || record->desc.height == 0))
                {
                    suspended = true;
                }
            });
            SurfaceCapabilities caps{};
            caps.minImageCount = 1;
            caps.maxImageCount = 1;
            caps.minWidth = 1;
            caps.minHeight = 1;
            caps.maxWidth = maxDim;
            caps.maxHeight = maxDim;
            caps.preferredWidth = (hidden || suspended) ? 0 : backing.first;
            caps.preferredHeight = (hidden || suspended) ? 0 : backing.second;
            caps.usages = TextureUsage::renderTarget | TextureUsage::copySource;
            caps.presentModes = PresentModeMask::fifo | PresentModeMask::immediate;
            caps.formats = {
                SurfaceFormat{.format = Format::bgra8Unorm, .colorSpace = SurfaceColorSpace::srgbNonlinear},
                SurfaceFormat{.format = Format::bgra8UnormSrgb, .colorSpace = SurfaceColorSpace::srgbNonlinear},
            };
            return caps;
        }
    }

    Status MetalDevice::destroySurface(SurfaceHandle surface)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal destroySurface requires an active device."));
        }
        auto* found = native_->surfaces.find(surface);
        if (found == nullptr || *found == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The surface handle is not a live Metal CAMetalLayer."));
        }
        if ((*found)->swapchainClaimed)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "A Metal surface with a live swapchain cannot be destroyed."));
        }
        if (!native_->surfaces.erase(surface))
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The surface handle is not a live Metal CAMetalLayer."));
        }
        return {};
    }

    namespace
    {

        [[nodiscard]] std::uint64_t metalNowNs() noexcept
        {
            return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                  std::chrono::steady_clock::now().time_since_epoch())
                                                  .count());
        }

        void dropAcquiredSwapchainLease(MetalDeviceNative& native, MetalSwapchainRecord& record) noexcept
        {
            if (record.view.isValid())
            {
                if (auto* view = native.views.find(record.view); view != nullptr)
                {
                    if (auto* texture = native.textures.find(view->texture);
                        texture != nullptr && texture->liveViews > 0)
                    {
                        texture->liveViews -= 1;
                    }
                    static_cast<void>(native.views.erase(record.view));
                }
                record.view = {};
            }
            if (record.texture.isValid())
            {
                static_cast<void>(native.textures.erase(record.texture));
                record.texture = {};
            }
            record.drawable = nil;
            record.acquired = false;
            record.leaseDroppedByHostClose = false;
            record.currentAcquireId.store(0, std::memory_order_release);
            record.waitedDrawableToken.store(nullptr, std::memory_order_release);
        }

        struct MetalSwapchainOperationLock final
        {
            std::shared_ptr<MetalSwapchainRecord> record;
            std::unique_lock<std::mutex> lock;
        };

        [[nodiscard]] MetalSwapchainOperationLock lockMetalSwapchainOperations(
            std::mutex& deviceMutex, const std::shared_ptr<MetalDeviceNative>& native, SwapchainHandle swapchain)
        {
            MetalSwapchainOperationLock operation;
            {
                std::scoped_lock const lock(deviceMutex);
                if (native != nullptr)
                {
                    if (auto* found = native->swapchains.find(swapchain); found != nullptr)
                    {
                        operation.record = *found;
                    }
                }
            }
            if (operation.record != nullptr)
            {
                operation.lock = std::unique_lock(operation.record->operationMutex);
            }
            return operation;
        }

    } // namespace

#ifdef RACHIS_TESTING
    void* MetalDevice::nativeMetalDeviceHandle() const noexcept
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return nullptr;
        }
        return (__bridge void*)native_->device;
    }

    Status MetalDevice::notifySurfaceHostCloseForTesting(SurfaceHandle surface)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal surface host close requires an active device."));
        }
        auto* found = native_->surfaces.find(surface);
        if (found == nullptr || *found == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The surface handle is not a live Metal CAMetalLayer."));
        }
        @autoreleasepool
        {
            native_->swapchains.forEach([&](SwapchainHandle, std::shared_ptr<MetalSwapchainRecord>& record) {
                if (record != nullptr && record->surface == surface && record->acquired)
                {
                    dropAcquiredSwapchainLease(*native_, *record);
                    record->leaseDroppedByHostClose = true;
                }
            });
            if (!native_->surfaces.erase(surface))
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "The surface handle is not a live Metal CAMetalLayer."));
            }
        }
        return {};
    }
#endif

    namespace
    {

        [[nodiscard]] Result<AcquiredSwapchainImage> publishAcquiredDrawable(MetalDeviceNative& native,
                                                                             MetalSwapchainRecord& record,
                                                                             SwapchainHandle swapchain,
                                                                             id<CAMetalDrawable> drawable)
        {
            id<MTLTexture> const texture = drawable.texture;
            if (texture == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::nativeFailure, "CAMetalDrawable.texture was nil after acquire."));
            }
            const auto width = static_cast<std::uint32_t>(texture.width);
            const auto height = static_cast<std::uint32_t>(texture.height);
            const auto acquireId = record.lastAcquireId + 1;
            MetalTextureRecord textureRecord{};
            textureRecord.owner = std::make_shared<MetalTextureOwner>();
            textureRecord.owner->texture = texture;
            textureRecord.texture = texture;
            textureRecord.desc = TextureDesc{
                .dimension = TextureDimension::texture2D,
                .width = width,
                .height = height,
                .format = record.desc.format,
                .usage = record.desc.usage,
            };
            textureRecord.swapchainImage = true;
            textureRecord.drawable = drawable;
            textureRecord.swapchain = swapchain;
            textureRecord.swapchainAcquireId = acquireId;
            auto textureHandle = native.textures.emplace(std::move(textureRecord));
            if (!textureHandle)
            {
                return std::unexpected(std::move(textureHandle.error()));
            }
            MetalTextureViewRecord viewRecord{};
            viewRecord.owner = std::make_shared<MetalTextureOwner>();
            viewRecord.owner->texture = texture;
            viewRecord.parent = native.textures.find(*textureHandle)->owner;
            viewRecord.view = texture;
            viewRecord.texture = *textureHandle;
            viewRecord.swapchainView = true;
            viewRecord.desc = TextureViewDesc{
                .texture = *textureHandle,
                .usage = TextureViewUsage::renderTarget,
                .dimension = TextureViewDimension::texture2D,
                .format = record.desc.format,
            };
            auto viewHandle = native.views.emplace(std::move(viewRecord));
            if (!viewHandle)
            {
                static_cast<void>(native.textures.erase(*textureHandle));
                return std::unexpected(std::move(viewHandle.error()));
            }
            native.textures.find(*textureHandle)->liveViews = 1;
            record.drawable = drawable;
            record.texture = *textureHandle;
            record.view = *viewHandle;
            record.acquired = true;
            record.lastAcquireId = acquireId;
            record.currentAcquireId.store(acquireId, std::memory_order_release);
            record.waitedDrawableToken.store(nullptr, std::memory_order_release);
            record.lastAcquireNs = metalNowNs();
            return AcquiredSwapchainImage{
                .texture = *textureHandle,
                .renderTargetView = *viewHandle,
                .initialLayout = TextureLayout::present,
                .requiredFinalLayout = TextureLayout::present,
                .contentsDefined = false,
                .imageIndex = 0,
                .width = width,
                .height = height,
                .format = record.desc.format,
                .colorSpace = record.desc.colorSpace,
                .generation = record.generation,
            };
        }

    } // namespace

    Result<SwapchainHandle> MetalDevice::createSwapchain(const SwapchainDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (desc.presentMode != PresentMode::fifo && desc.presentMode != PresentMode::immediate)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "A swapchain present mode must be a valid portable enumerator."));
        }

        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal createSwapchain requires an active device."));
        }
        auto* surfaceFound = native_->surfaces.find(desc.surface);
        if (surfaceFound == nullptr || *surfaceFound == nullptr || (*surfaceFound)->layer == nil)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The surface handle is not a live Metal CAMetalLayer."));
        }
        auto& surface = **surfaceFound;
        if (surface.swapchainClaimed)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "A Metal surface can own only one live swapchain."));
        }
        bool layerClaimed = false;
        native_->surfaces.forEach([&](SurfaceHandle, std::shared_ptr<MetalSurfaceRecord>& other) {
            if (other != nullptr && other.get() != &surface && other->swapchainClaimed && other->layer == surface.layer)
            {
                layerClaimed = true;
            }
        });
        if (layerClaimed)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "The native window already has a live swapchain."));
        }

        const auto drawable = surface.layer.drawableSize;
        const auto maxDim = capabilities_.limits.maxTextureDimension2D;
        if (desc.width == 0 || desc.height == 0 || desc.width > maxDim || desc.height > maxDim ||
            desc.imageCount != 1 || !hasAny(desc.usage, TextureUsage::renderTarget))
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument,
                "A swapchain requires a supported non-empty extent and image count plus render-target usage."));
        }
        static_cast<void>(drawable);
        const bool formatSupported = desc.format == Format::bgra8Unorm || desc.format == Format::bgra8UnormSrgb;
        const auto usageBits = static_cast<std::uint32_t>(desc.usage);
        const auto allowedUsage = static_cast<std::uint32_t>(TextureUsage::renderTarget | TextureUsage::copySource);
        const bool usageSupported = hasAny(desc.usage, TextureUsage::renderTarget) && (usageBits & ~allowedUsage) == 0;
        if (!formatSupported || desc.colorSpace != SurfaceColorSpace::srgbNonlinear || !usageSupported)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::unsupported,
                "The swapchain format, color space, present mode, or usage is not supported by this surface."));
        }

        @autoreleasepool
        {
            CAMetalLayer* const layer = surface.layer;
            if (layer.device != nil && layer.device != native_->device)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "The CAMetalLayer is bound to a different Metal device."));
            }
            auto record = std::make_shared<MetalSwapchainRecord>();
            record->surface = desc.surface;
            record->desc = desc;
            record->label = std::string(desc.label);
            record->desc.label = record->label;
            auto handle = native_->swapchains.emplace(std::move(record));
            if (!handle)
            {
                return std::unexpected(std::move(handle.error()));
            }
            layer.device = native_->device;
            applyMetalSwapchainLayer(layer, desc);
            if (!surface.label.empty())
            {
                layer.name = [[NSString alloc] initWithBytes:surface.label.data()
                                                      length:surface.label.size()
                                                    encoding:NSUTF8StringEncoding];
            }
            surface.swapchainClaimed = true;
            return handle;
        }
    }

    Status MetalDevice::destroySwapchain(SwapchainHandle swapchain)
    {
        const auto operation = lockMetalSwapchainOperations(mutex_, native_, swapchain);
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal destroySwapchain requires an active device."));
        }
        auto* found = native_->swapchains.find(swapchain);
        if (found == nullptr || *found == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The swapchain handle is invalid or stale."));
        }
        auto& record = **found;
        if (record.acquired)
        {
            dropAcquiredSwapchainLease(*native_, record);
        }
        if (record.inFlightPresents != 0 && native_->queueGate != nullptr &&
            !native_->nativeLost.load(std::memory_order_acquire))
        {
            const std::scoped_lock gateLock(native_->queueGate->mutex);
            auto drained = drainMetalQueue(*native_, native_->timelines[metalQueueIndex(QueueType::graphics)]);
            if (drained)
            {
                retireMetalPresents(*native_);
            }
            else if (!native_->nativeLost.load(std::memory_order_acquire))
            {
                return drained;
            }
        }
        if (record.inFlightPresents != 0 && !native_->nativeLost.load(std::memory_order_acquire))
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "A Metal swapchain cannot be destroyed while presents are in flight."));
        }
        if (auto* surface = native_->surfaces.find(record.surface); surface != nullptr && *surface != nullptr)
        {
            (*surface)->swapchainClaimed = false;
        }
        if (!native_->swapchains.erase(swapchain))
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The swapchain handle is invalid or stale."));
        }
        return {};
    }

    Status MetalDevice::resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width, std::uint32_t height)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
#ifdef RACHIS_TESTING
        if (auto injected = consumeInjectedMetalPresentLoss(*this); !injected)
        {
            return std::unexpected(std::move(injected.error()));
        }
#endif
        const auto operation = lockMetalSwapchainOperations(mutex_, native_, swapchain);
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal resizeSwapchain requires an active device."));
        }
        auto* found = native_->swapchains.find(swapchain);
        if (found == nullptr || *found == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The swapchain handle is invalid or stale."));
        }
        auto& record = **found;
        if (record.acquired)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "Cancel the acquired drawable before resizing a Metal swapchain."));
        }
        const auto maxDim = capabilities_.limits.maxTextureDimension2D;
        if ((width != 0 && width > maxDim) || (height != 0 && height > maxDim) || (width == 0) != (height == 0))
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument,
                                    "A swapchain resize requires a supported extent or a zero-by-zero suspend."));
        }
        if (record.generation == std::numeric_limits<std::uint64_t>::max())
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::resourceExhausted, "Swapchain generations are exhausted."));
        }
        auto feedback = std::make_shared<MetalPresentationFeedback>();
        record.desc.width = width;
        record.desc.height = height;
        record.leaseDroppedByHostClose = false;
        ++record.generation;
        record.submittedPresentationSequence = 0;
        record.presentationFeedback = std::move(feedback);
        if (auto* surface = native_->surfaces.find(record.surface);
            surface != nullptr && *surface != nullptr && (*surface)->layer != nil)
        {
            (*surface)->layer.drawableSize = CGSizeMake(static_cast<CGFloat>(width), static_cast<CGFloat>(height));
            applyMetalPresentMode((*surface)->layer, record.desc.presentMode);
        }
        return {};
    }

    namespace
    {

        [[nodiscard]] Result<std::pair<std::shared_ptr<MetalSwapchainRecord>, CAMetalLayer*>>
        validateMetalSwapchainAcquirable(const std::shared_ptr<MetalDeviceNative>& native, SwapchainHandle swapchain)
        {
            if (native == nullptr || native->device == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal acquireNextImage requires an active device."));
            }
            retireMetalPresents(*native);
            auto* found = native->swapchains.find(swapchain);
            if (found == nullptr || *found == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "The swapchain handle is invalid or stale."));
            }
            auto& record = **found;
            if (record.acquired)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "A Metal swapchain already has an acquired drawable."));
            }
            auto* surface = native->surfaces.find(record.surface);
            if (surface == nullptr || *surface == nullptr || (*surface)->layer == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::notReady, "The Metal surface is detached or destroyed and is no longer presentable."));
            }
            if (record.desc.width == 0 || record.desc.height == 0)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState, "A minimized or zero-extent Metal swapchain cannot acquire a drawable."));
            }
            if (metalSurfaceIsHidden(**surface))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::notReady,
                    "The Metal surface is hidden or occluded; acquire did not wait on nextDrawable."));
            }
            const auto live = metalLayerBackingPixels((*surface)->layer);
            if (live.first != record.desc.width || live.second != record.desc.height)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "The swapchain surface is out of date."));
            }
            return std::pair{*found, (*surface)->layer};
        }

#ifdef RACHIS_TESTING
        [[nodiscard]] Status consumeInjectedMetalDrawableFailure()
        {
            bool failAcquire = false;
            {
                std::scoped_lock const testLock(metalTestingState().mutex);
                failAcquire = metalTestingState().failNextDrawable;
                metalTestingState().failNextDrawable = false;
            }
            if (failAcquire)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::notReady, "Injected Metal drawable acquire timeout. Selection did not fall back."));
            }
            return {};
        }

        void holdInjectedMetalDrawableAcquire()
        {
            std::shared_ptr<MetalCommitFeedbackGate> gate;
            {
                std::scoped_lock const testLock(metalTestingState().mutex);
                gate = std::move(metalTestingState().nextDrawableAcquireGate);
            }
            if (gate != nullptr)
            {
                gate->waitUntilReleased();
            }
        }
#endif

        [[nodiscard]] id<CAMetalDrawable> nextMetalDrawable(CAMetalLayer* layer)
        {
#ifdef RACHIS_TESTING
            holdInjectedMetalDrawableAcquire();
#endif
            @autoreleasepool
            {
                return [layer nextDrawable];
            }
        }

        [[nodiscard]] Status validateMetalDrawableWaitOutcome(const std::shared_ptr<MetalDeviceNative>& native,
                                                              const MetalSwapchainRecord& record,
                                                              SwapchainHandle swapchain,
                                                              id<CAMetalDrawable> drawable)
        {
            if (native == nullptr || native->device == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal acquireNextImage requires an active device."));
            }
            const auto* found = native->swapchains.find(swapchain);
            if (found == nullptr || found->get() != &record)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "The swapchain handle is invalid or stale."));
            }
            const auto* surface = native->surfaces.find(record.surface);
            if (surface == nullptr || *surface == nullptr || (*surface)->layer == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::notReady, "The Metal surface is detached or destroyed and is no longer presentable."));
            }
            if (drawable == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::notReady, "CAMetalLayer nextDrawable timed out or returned nil."));
            }
            return {};
        }

    } // namespace

    Result<AcquiredSwapchainImage> MetalDevice::acquireNextImage(SwapchainHandle swapchain)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }

        const auto operation = lockMetalSwapchainOperations(mutex_, native_, swapchain);
        std::unique_lock lock(mutex_);
        auto ready = validateMetalSwapchainAcquirable(native_, swapchain);
        if (!ready)
        {
            return std::unexpected(std::move(ready.error()));
        }
#ifdef RACHIS_TESTING
        if (auto injected = consumeInjectedMetalDrawableFailure(); !injected)
        {
            return std::unexpected(std::move(injected.error()));
        }
#endif
        if (ready->first->inFlightPresents >= 3 && !ready->first->inFlight.empty())
        {
            const auto backend = native_;
            const auto oldestValue = ready->first->inFlight.front().timelineValue;
            lock.unlock();
            auto waited = waitForMetalValue(
                *backend, backend->timelines[metalQueueIndex(QueueType::graphics)], oldestValue, 1000);
            if (!waited)
            {
                return std::unexpected(std::move(waited.error()));
            }
            if (auto blocked = requireActive(); !blocked)
            {
                return std::unexpected(std::move(blocked.error()));
            }
            lock.lock();
            ready = validateMetalSwapchainAcquirable(native_, swapchain);
            if (!ready)
            {
                return std::unexpected(std::move(ready.error()));
            }
        }
        const auto record = ready->first;
        CAMetalLayer* const layer = ready->second;
        if (record->inFlightPresents >= 3)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::notReady, "The CAMetalLayer drawable pool is exhausted until a present retires."));
        }
        lock.unlock();
        id<CAMetalDrawable> const drawable = nextMetalDrawable(layer);
        auto active = requireActive();
        lock.lock();
        if (!active)
        {
            return std::unexpected(std::move(active.error()));
        }
        if (auto current = validateMetalDrawableWaitOutcome(native_, *record, swapchain, drawable); !current)
        {
            return std::unexpected(std::move(current.error()));
        }
        @autoreleasepool
        {
            return publishAcquiredDrawable(*native_, *record, swapchain, drawable);
        }
    }

    bool MetalDevice::hasAcquiredSwapchainImage(SwapchainHandle swapchain) const noexcept
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return false;
        }
        const auto* found = native_->swapchains.find(swapchain);
        return found != nullptr && *found != nullptr && (*found)->acquired && (*found)->drawable != nil;
    }

    Status MetalDevice::cancelSwapchainImage(SwapchainHandle swapchain)
    {
        const auto operation = lockMetalSwapchainOperations(mutex_, native_, swapchain);
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal cancelSwapchainImage requires an active device."));
        }
        auto* found = native_->swapchains.find(swapchain);
        if (found == nullptr || *found == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The swapchain handle is invalid or stale."));
        }
        auto& record = **found;
        if (!record.acquired)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "Cancel requires one currently acquired swapchain image."));
        }
        dropAcquiredSwapchainLease(*native_, record);
        return {};
    }

    namespace
    {

        [[nodiscard]] bool metalRenderCompleteForeignOrUnsubmitted(const MetalDeviceNative& native,
                                                                   TimelinePoint renderComplete)
        {
            return renderComplete.isValid() &&
                   (!isValidMetalQueueType(renderComplete.queueType()) ||
                    detail::TimelinePointAccess::deviceToken(renderComplete) != native.deviceToken ||
                    renderComplete.value() >
                        native.timelines[metalQueueIndex(renderComplete.queueType())].lastSubmitted.load(
                            std::memory_order_acquire));
        }

        [[nodiscard]] Result<MetalSwapchainRecord*> findPresentableMetalSwapchain(MetalDeviceNative& native,
                                                                                  SwapchainHandle swapchain,
                                                                                  TimelinePoint renderComplete)
        {
            if (metalRenderCompleteForeignOrUnsubmitted(native, renderComplete))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "Presentation references a foreign or unsubmitted timeline point."));
            }
            auto* found = native.swapchains.find(swapchain);
            if (found == nullptr || *found == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "The swapchain handle is invalid or stale."));
            }
            auto& record = **found;
            if ((!record.acquired || record.drawable == nil) && !record.leaseDroppedByHostClose)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState, "Presentation requires one currently acquired swapchain image."));
            }
            return &record;
        }

        [[nodiscard]] std::optional<PresentStatus> unpresentableSurfaceStatus(MetalDeviceNative& native,
                                                                              MetalSwapchainRecord& record)
        {
            if (record.leaseDroppedByHostClose)
            {
                record.leaseDroppedByHostClose = false;
                return PresentStatus::surfaceOutOfDate;
            }
            auto* surface = native.surfaces.find(record.surface);
            if (surface == nullptr || *surface == nullptr)
            {
                dropAcquiredSwapchainLease(native, record);
                return PresentStatus::surfaceOutOfDate;
            }
            if (metalSurfaceIsHidden(**surface))
            {
                dropAcquiredSwapchainLease(native, record);
                return PresentStatus::occluded;
            }
            const auto live = metalLayerBackingPixels((*surface)->layer);
            if (live.first != record.desc.width || live.second != record.desc.height)
            {
                dropAcquiredSwapchainLease(native, record);
                return PresentStatus::surfaceOutOfDate;
            }
            return std::nullopt;
        }

        [[nodiscard]] bool metalPresentTimelineExhausted(const MetalSwapchainRecord& record,
                                                         const MetalQueueTimeline& timeline)
        {
            return record.submittedPresentationSequence == std::numeric_limits<std::uint64_t>::max() ||
                   !hasUnreservedMetalTimelineSlot(timeline.nextValue) || timeline.event == nil;
        }

        [[nodiscard]] PresentStatus submitMetalDrawablePresent(MetalDeviceNative& native,
                                                               MetalSwapchainRecord& record,
                                                               MetalQueueTimeline& timeline,
                                                               TimelinePoint renderComplete)
        {
            @autoreleasepool
            {
                id<CAMetalDrawable> const drawable = record.drawable;
                const auto submittedValue = timeline.nextValue;
                record.inFlight.push_back(MetalInFlightPresent{.timelineValue = submittedValue, .drawable = drawable});
                if (renderComplete.isValid())
                {
                    auto& waited = native.timelines[metalQueueIndex(renderComplete.queueType())];
                    [native.commandQueue waitForEvent:waited.event value:renderComplete.value()];
                }
                const void* const drawableToken = (__bridge const void*)drawable;
                const bool alreadyWaited = record.waitedDrawableToken.load(std::memory_order_acquire) == drawableToken;
                dropAcquiredSwapchainLease(native, record);
                if (!alreadyWaited)
                {
                    [native.commandQueue waitForDrawable:drawable];
                }
                const auto presentationSequence = record.submittedPresentationSequence + 1;
                const auto feedback = record.presentationFeedback;
                [drawable addPresentedHandler:^(id<MTLDrawable> presentedDrawable) {
                  @autoreleasepool
                  {
                      feedback->presented(presentationSequence, presentedDrawable.presentedTime);
                  }
                }];
                [native.commandQueue signalDrawable:drawable];
                [drawable present];
                record.submittedPresentationSequence = presentationSequence;
                [native.commandQueue signalEvent:timeline.event value:submittedValue];
                ++timeline.nextValue;
                timeline.lastSubmitted.store(submittedValue, std::memory_order_release);
                record.inFlightPresents = static_cast<std::uint32_t>(record.inFlight.size());
                const auto now = metalNowNs();
                if (record.lastPresentSubmitNs != 0)
                {
                    record.lastPresentIntervalNs = now - record.lastPresentSubmitNs;
                }
                record.lastPresentSubmitNs = now;
                return PresentStatus::presented;
            }
        }

    } // namespace

    Result<PresentStatus> MetalDevice::present(SwapchainHandle swapchain, TimelinePoint renderComplete)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
#ifdef RACHIS_TESTING
        if (auto injected = consumeInjectedMetalPresentLoss(*this); !injected)
        {
            return std::unexpected(std::move(injected.error()));
        }
        const auto injectedPresentFailure = rachis::metal::test::nextInjectedPresentFailure().exchange(
            rachis::metal::test::InjectedPresentFailure::none, std::memory_order_relaxed);
#endif

        const auto operation = lockMetalSwapchainOperations(mutex_, native_, swapchain);
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr || native_->device == nil || native_->commandQueue == nil ||
                native_->queueGate == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal present requires an active device."));
            }
            auto found = findPresentableMetalSwapchain(*native_, swapchain, renderComplete);
            if (!found)
            {
                return std::unexpected(std::move(found.error()));
            }
            auto& record = **found;
#ifdef RACHIS_TESTING
            if (injectedPresentFailure == rachis::metal::test::InjectedPresentFailure::beforeConsumeValidationError)
            {
                return std::unexpected(metalSelectionError(ErrorCode::validationError,
                                                           "Injected pre-consume Metal present validation failure."));
            }
#endif
            if (auto unpresentable = unpresentableSurfaceStatus(*native_, record); unpresentable.has_value())
            {
                return *unpresentable;
            }

            const std::scoped_lock gateLock(native_->queueGate->mutex);
            auto& timeline = native_->timelines[metalQueueIndex(QueueType::graphics)];
            if (metalPresentTimelineExhausted(record, timeline))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::resourceExhausted, "The queue timeline is exhausted and cannot be reused safely."));
            }
#ifdef RACHIS_TESTING
            if (injectedPresentFailure == rachis::metal::test::InjectedPresentFailure::afterConsumeNativeFailure)
            {
                dropAcquiredSwapchainLease(*native_, record);
                return std::unexpected(
                    metalSelectionError(ErrorCode::nativeFailure, "Injected post-consume Metal present failure."));
            }
            if (injectedPresentFailure == rachis::metal::test::InjectedPresentFailure::afterConsumeDeviceLost)
            {
                dropAcquiredSwapchainLease(*native_, record);
            }
            else
#endif
            {
                return submitMetalDrawablePresent(*native_, record, timeline, renderComplete);
            }
        }
#ifdef RACHIS_TESTING
        return std::unexpected(injectDeviceLoss().error());
#endif
    }

    Result<SwapchainPresentationInfo> MetalDevice::querySwapchainPresentation(SwapchainHandle swapchain) const
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "Metal querySwapchainPresentation requires an active device."));
        }
        auto* found = native_->swapchains.find(swapchain);
        if (found == nullptr || *found == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The swapchain handle is invalid or stale."));
        }
        const auto& record = **found;
        return SwapchainPresentationInfo{
            .format = record.desc.format,
            .colorSpace = record.desc.colorSpace,
            .presentMode = record.desc.presentMode,
            .usage = record.desc.usage,
            .imageCount = 1,
            .framesInFlightBound = 3,
            .width = record.desc.width,
            .height = record.desc.height,
            .hdr = false,
        };
    }

    Result<SwapchainFrameTiming> MetalDevice::querySwapchainFrameTiming(SwapchainHandle swapchain) const
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "Metal querySwapchainFrameTiming requires an active device."));
        }
        auto* found = native_->swapchains.find(swapchain);
        if (found == nullptr || *found == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The swapchain handle is invalid or stale."));
        }
        const auto& record = **found;
        const auto* surface = native_->surfaces.find(record.surface);
        const bool hidden = surface != nullptr && *surface != nullptr && metalSurfaceIsHidden(**surface);
        return SwapchainFrameTiming{
            .lastAcquireNs = record.lastAcquireNs,
            .lastPresentSubmitNs = record.lastPresentSubmitNs,
            .lastPresentIntervalNs = record.lastPresentIntervalNs,
            .presentsCompleted = record.presentsCompleted,
            .inFlightPresents = record.inFlightPresents,
            .hidden = hidden || record.desc.width == 0 || record.desc.height == 0,
            .presentationSupported = true,
            .presentationGeneration = record.generation,
            .submittedPresentationSequence = record.submittedPresentationSequence,
            .displayedPresentationSequence = record.presentationFeedback->displayedSequence(),
        };
    }

} // namespace rachis::metal
