----------------------- MODULE MetalDrawableLifecycle -----------------------
(***************************************************************************)
(* CAMetalDrawable acquisition, cancellation, presentation, GPU            *)
(* retirement, host-surface closure, and swapchain destruction in the      *)
(* Metal 4 backend: `acquireNextImage`, `cancelSwapchainImage`, `present`, *)
(* `destroySwapchain`, and host-close lease dropping in                    *)
(* `src/Metal/MetalSwapchain.mm`; graphics-timeline present retirement     *)
(* (`retireMetalPresents`) in `src/Metal/MetalCommand.mm`; and the         *)
(* `AcquiredSwapchainImage` and `PresentStatus` types in                   *)
(* `include/Rachis/Presentation.h`.                                        *)
(*                                                                         *)
(* Models one public RHI swapchain lease backed by a bounded native        *)
(* `CAMetalLayer` drawable pool (`PoolSize`). At most one unsubmitted      *)
(* lease is published at a time; cancelling an unsubmitted lease releases  *)
(* the underlying `CAMetalDrawable` back to the pool; presenting transfers *)
(* the lease into in-flight GPU presentation occupancy until timeline      *)
(* retirement; and swapchain destruction requires zero remaining occupancy *)
(* and in-flight presents.                                                 *)
(*                                                                         *)
(* Design rules controlled by CONSTANTS (each TRUE in the design):         *)
(*   - CancelReleasesDrawable: `cancelSwapchainImage` releases the native  *)
(*     drawable instead of leaking pool capacity                           *)
(*     (`MetalDrawableLifecycle-cancelleak.cfg` sets FALSE).               *)
(*   - PresentConsumesLease: `present` consumes the public lease so the    *)
(*     same acquisition cannot remain leased after presentation            *)
(*     (`MetalDrawableLifecycle-doublepresent.cfg` sets FALSE).            *)
(*   - WaitInFlightBeforeDestroy: `destroySwapchain` completes only once   *)
(*     in-flight presents have drained to zero                             *)
(*     (`MetalDrawableLifecycle-earlydestroy.cfg` sets FALSE).             *)
(*                                                                         *)
(* Abstracted: CAMetalLayer compositor internals, pixel contents, and      *)
(* non-state-changing `notReady` acquisition timeouts.                     *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS
    PoolSize,                   \* native CAMetalLayer maximumDrawableCount
    CancelReleasesDrawable,     \* TRUE: cancel returns drawable to pool
    PresentConsumesLease,       \* TRUE: present clears publicLease
    WaitInFlightBeforeDestroy   \* TRUE: destroy requires inFlight = 0

VARIABLES
    publicLease,        \* TRUE while caller holds an unsubmitted swapchain lease
    inFlight,           \* count of presented drawables awaiting GPU retirement
    leakedDrawables,    \* count of cancelled drawables leaked from the native pool
    closed,             \* TRUE after host surface closes
    destroyed,          \* TRUE after swapchain is destroyed
    presentedCurrent    \* TRUE after the most recent acquisition was presented

vars == <<
    publicLease, inFlight, leakedDrawables,
    closed, destroyed, presentedCurrent
>>

Occupancy == (IF publicLease THEN 1 ELSE 0) + inFlight + leakedDrawables

Init ==
    /\ publicLease = FALSE
    /\ inFlight = 0
    /\ leakedDrawables = 0
    /\ closed = FALSE
    /\ destroyed = FALSE
    /\ presentedCurrent = FALSE

\* Acquire one drawable from CAMetalLayer when pool capacity is available.
acquire ==
    /\ ~closed
    /\ ~destroyed
    /\ ~publicLease
    /\ Occupancy < PoolSize
    /\ publicLease' = TRUE
    /\ presentedCurrent' = FALSE
    /\ UNCHANGED <<inFlight, leakedDrawables, closed, destroyed>>

\* Cancel an unsubmitted acquisition and return its native drawable to the pool.
cancel ==
    /\ ~destroyed
    /\ publicLease
    /\ ~presentedCurrent
    /\ publicLease' = FALSE
    /\ leakedDrawables' =
        IF CancelReleasesDrawable
        THEN leakedDrawables
        ELSE leakedDrawables + 1
    /\ presentedCurrent' = FALSE
    /\ UNCHANGED <<inFlight, closed, destroyed>>

\* Present consumes the public lease and tracks the drawable in flight.
present ==
    /\ ~closed
    /\ ~destroyed
    /\ publicLease
    /\ ~presentedCurrent
    /\ inFlight < PoolSize
    /\ publicLease' = IF PresentConsumesLease THEN FALSE ELSE publicLease
    /\ inFlight' = inFlight + 1
    /\ presentedCurrent' = TRUE
    /\ UNCHANGED <<leakedDrawables, closed, destroyed>>

\* Graphics timeline retirement releases one completed in-flight present.
complete ==
    /\ ~destroyed
    /\ inFlight > 0
    /\ inFlight' = inFlight - 1
    /\ UNCHANGED <<publicLease, leakedDrawables, closed, destroyed, presentedCurrent>>

\* Host window/surface closure drops any unsubmitted lease and blocks new acquires.
close ==
    /\ ~closed
    /\ ~destroyed
    /\ publicLease' = FALSE
    /\ closed' = TRUE
    /\ presentedCurrent' = FALSE
    /\ UNCHANGED <<inFlight, leakedDrawables, destroyed>>

\* Swapchain destruction requires no active lease and drained in-flight presents.
destroy ==
    /\ ~destroyed
    /\ ~publicLease
    /\ IF WaitInFlightBeforeDestroy THEN inFlight = 0 ELSE TRUE
    /\ destroyed' = TRUE
    /\ UNCHANGED <<publicLease, inFlight, leakedDrawables, closed, presentedCurrent>>

Next ==
    \/ acquire
    \/ cancel
    \/ present
    \/ complete
    \/ close
    \/ destroy

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ PoolSize \in Nat \ {0}
    /\ CancelReleasesDrawable \in BOOLEAN
    /\ PresentConsumesLease \in BOOLEAN
    /\ WaitInFlightBeforeDestroy \in BOOLEAN
    /\ publicLease \in BOOLEAN
    /\ inFlight \in 0..PoolSize
    /\ leakedDrawables \in 0..PoolSize
    /\ closed \in BOOLEAN
    /\ destroyed \in BOOLEAN
    /\ presentedCurrent \in BOOLEAN

CancelReleasesNativeDrawable ==
    /\ leakedDrawables = 0
    /\ ~publicLease => Occupancy = inFlight

PresentedClearsPublicLease ==
    presentedCurrent => ~publicLease

DestroyedAfterDrain ==
    destroyed => ~publicLease /\ inFlight = 0

PoolOccupancyBounded ==
    Occupancy <= PoolSize

ClosedClearsPublicLease ==
    closed => ~publicLease

Safety ==
    /\ CancelReleasesNativeDrawable
    /\ PresentedClearsPublicLease
    /\ DestroyedAfterDrain
    /\ PoolOccupancyBounded
    /\ ClosedClearsPublicLease

=============================================================================
