------------------------- MODULE VulkanWsiLifecycle -------------------------
(***************************************************************************)
(* Formal safety model for per-image Vulkan WSI ownership, cancelled       *)
(* acquisitions, present-semaphore signal/wait pairing, rejected presents, *)
(* pre-enqueue allocation failures, and resize/destroy synchronization:    *)
(*   - src/Vulkan/VulkanSwapchain.cpp (acquireNextImage,                   *)
(*     cancelSwapchainImage, reacquireCancelledImage, present,             *)
(*     retireBinarySemaphore, resizeSwapchain, destroySwapchain)           *)
(*   - src/Vulkan/VulkanQueues.cpp (presentOnPresentQueue)                 *)
(*                                                                         *)
(* Design-rule constants (TRUE in VulkanWsiLifecycle.cfg, FALSE in         *)
(* counterexample configurations):                                         *)
(*   - SingleWaitOnRejectedPresent: when vkQueuePresentKHR already         *)
(*     enqueued its wait before returning VK_ERROR_OUT_OF_DATE_KHR,        *)
(*     present must not enqueue a second retirement wait on image.present. *)
(*   - CacheCancelledAcquisition: cancelling a public swapchain lease      *)
(*     parks the driver-acquired image in a cached slot for reacquisition  *)
(*     instead of dropping track of its native ownership.                  *)
(*                                                                         *)
(* Trusted boundary: Vulkan WSI driver presentation queue and native       *)
(* VK_EXT_swapchain_maintenance1 fence signaling.                          *)
(***************************************************************************)
EXTENDS Integers

CONSTANTS SingleWaitOnRejectedPresent, CacheCancelledAcquisition

ASSUME {SingleWaitOnRejectedPresent, CacheCancelledAcquisition} \subseteq BOOLEAN

Images == {0, 1}
NoImage == -1
ImageOrNone == Images \cup {NoImage}

VARIABLES
  owned,         \* Set of swapchain images currently owned by the RHI (not WSI).
  lease,         \* Public caller lease (image index or NoImage).
  cached,        \* Cancelled driver-acquired image parked for reacquisition.
  signals,       \* Per-image present-semaphore signal count in current epoch.
  waits,         \* Per-image present-semaphore wait count in current epoch.
  pendingFences, \* Images whose swapchain-maintenance present fence is pending.
  prepared,      \* Whether the bridge submit signaled the present semaphore.
  refresh,       \* Whether VK_ERROR_OUT_OF_DATE_KHR requires resize before acquire.
  closed         \* Whether the swapchain has been destroyed.

vars == <<
  owned,
  lease,
  cached,
  signals,
  waits,
  pendingFences,
  prepared,
  refresh,
  closed
>>

Init ==
  /\ owned = {}
  /\ lease = NoImage
  /\ cached = NoImage
  /\ signals = [image \in Images |-> 0]
  /\ waits = [image \in Images |-> 0]
  /\ pendingFences = {}
  /\ prepared = FALSE
  /\ refresh = FALSE
  /\ closed = FALSE

acquire(image) ==
  /\ ~closed
  /\ ~refresh
  /\ lease = NoImage
  /\ cached = NoImage
  /\ image \in Images
  /\ image \notin owned
  /\ signals[image] = 0
  /\ image \notin pendingFences
  /\ owned' = owned \cup {image}
  /\ lease' = image
  /\ UNCHANGED <<cached, signals, waits, pendingFences, prepared, refresh, closed>>

cancel ==
  /\ ~closed
  /\ lease \in Images
  /\ ~prepared
  /\ lease' = NoImage
  /\ cached' = IF CacheCancelledAcquisition THEN lease ELSE NoImage
  /\ UNCHANGED <<owned, signals, waits, pendingFences, prepared, refresh, closed>>

reacquire ==
  /\ ~closed
  /\ ~refresh
  /\ lease = NoImage
  /\ cached \in Images
  /\ lease' = cached
  /\ cached' = NoImage
  /\ UNCHANGED <<owned, signals, waits, pendingFences, prepared, refresh, closed>>

bridge ==
  /\ ~closed
  /\ lease \in Images
  /\ ~prepared
  /\ signals[lease] = 0
  /\ signals' = [signals EXCEPT ![lease] = 1]
  /\ prepared' = TRUE
  /\ UNCHANGED <<owned, lease, cached, waits, pendingFences, refresh, closed>>

present(rejected) ==
  /\ ~closed
  /\ lease \in Images
  /\ prepared
  /\ owned' = owned \ {lease}
  /\ lease' = NoImage
  /\ waits' =
       [waits EXCEPT ![lease] =
          waits[lease] + (IF ~SingleWaitOnRejectedPresent /\ rejected THEN 2 ELSE 1)]
  /\ pendingFences' = pendingFences \cup {lease}
  /\ prepared' = FALSE
  /\ refresh' = rejected
  /\ UNCHANGED <<cached, signals, closed>>

allocationFailure ==
  /\ ~closed
  /\ lease \in Images
  /\ prepared
  /\ waits' = [waits EXCEPT ![lease] = waits[lease] + 1]
  /\ prepared' = FALSE
  /\ UNCHANGED <<owned, lease, cached, signals, pendingFences, refresh, closed>>

complete(image) ==
  /\ ~closed
  /\ image \in Images
  /\ signals[image] = 1
  /\ waits[image] = 1
  /\ signals' = [signals EXCEPT ![image] = 0]
  /\ waits' = [waits EXCEPT ![image] = 0]
  /\ pendingFences' = pendingFences \ {image}
  /\ UNCHANGED <<owned, lease, cached, prepared, refresh, closed>>

resize ==
  /\ ~closed
  /\ lease = NoImage
  /\ \A image \in Images : signals[image] = 0
  /\ pendingFences = {}
  /\ owned' = {}
  /\ cached' = NoImage
  /\ refresh' = FALSE
  /\ UNCHANGED <<lease, signals, waits, pendingFences, prepared, closed>>

destroy ==
  /\ ~closed
  /\ lease = NoImage
  /\ \A image \in Images : signals[image] = 0
  /\ pendingFences = {}
  /\ owned' = {}
  /\ cached' = NoImage
  /\ closed' = TRUE
  /\ UNCHANGED <<lease, signals, waits, pendingFences, prepared, refresh>>

Next ==
  \/ \E image \in Images : acquire(image)
  \/ cancel
  \/ reacquire
  \/ bridge
  \/ \E rejected \in BOOLEAN : present(rejected)
  \/ allocationFailure
  \/ \E image \in Images : complete(image)
  \/ resize
  \/ destroy

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ owned \in SUBSET Images
  /\ lease \in ImageOrNone
  /\ cached \in ImageOrNone
  /\ signals \in [Images -> 0..1]
  /\ waits \in [Images -> 0..2]
  /\ pendingFences \in SUBSET Images
  /\ prepared \in BOOLEAN
  /\ refresh \in BOOLEAN
  /\ closed \in BOOLEAN

NativeOwnershipAccounted ==
  /\ owned =
       ((IF lease \in Images THEN {lease} ELSE {})
        \cup (IF cached \in Images THEN {cached} ELSE {}))
  /\ ~(lease \in Images /\ cached \in Images)
  /\ (prepared => lease \in Images)

EachSignalHasAtMostOneWait ==
  \A image \in Images : waits[image] <= signals[image]

PendingFenceHasExactWait ==
  \A image \in pendingFences : waits[image] = 1

ClosedSwapchainIsQuiescent ==
  closed =>
    /\ owned = {}
    /\ pendingFences = {}
    /\ \A image \in Images : signals[image] = 0 /\ waits[image] = 0

Safety ==
  /\ NativeOwnershipAccounted
  /\ EachSignalHasAtMostOneWait
  /\ PendingFenceHasExactWait
  /\ ClosedSwapchainIsQuiescent

StepSafety ==
  [][closed => closed']_vars

=============================================================================
