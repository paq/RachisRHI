----------------------- MODULE DeviceFaultDiagnostics -----------------------
(***************************************************************************)
(* Monotonic device-loss state, single-shot automatic fault report         *)
(* caching/logging, healthy early-return gating, bounded DRED truncation,  *)
(* and owning report immutability after native DRED disposal.              *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Diagnostics.cpp (captureDeviceFaultReport,           *)
(*     emitDeviceFaultReport)                                              *)
(*   - src/D3D12/D3D12Device.cpp (D3D12Device::captureDeviceFaultReport,   *)
(*     D3D12Device::observeDeviceLoss)                                     *)
(*   - include/Rachis/Diagnostics.h (DeviceFaultStatus, DeviceFaultReport) *)
(*                                                                         *)
(* Contract modeled:                                                       *)
(*   - Device removal transitions deviceLost from FALSE to TRUE            *)
(*     monotonically; it never reverts to healthy.                         *)
(*   - On a healthy device (~deviceLost), captureDeviceFaultReport returns *)
(*     early after querying DRED availability (when                        *)
(*     GateDredReadoutOnFault = TRUE, D3D12Diagnostics.cpp),               *)
(*     yielding zero breadcrumbs and zero page-fault allocations.          *)
(*   - ObserveDeviceLoss caches an owning DeviceFaultReport and emits the  *)
(*     diagnostic callback at most once across repeated observations.      *)
(*   - Captured reports are owning values bounded by MaxBreadcrumbs and    *)
(*     MaxAllocations with exact truncation flags, remaining immutable     *)
(*     even if native DRED diagnostics are later discarded.                *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - Host std::bad_alloc during vector allocation inside                 *)
(*     captureDeviceFaultReport (D3D12Diagnostics.cpp).                    *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT GateDredReadoutOnFault

ASSUME GateDredReadoutOnFault \in BOOLEAN

MaxNativeItems == 6
MaxBreadcrumbs == 4
MaxAllocations == 4
Actions == {"none", "record", "remove", "observe", "capture", "discardNative"}

VARIABLES
    deviceLost,                     \* BOOLEAN: hardware/driver device removed
    lossObserved,                   \* BOOLEAN: observeDeviceLoss published
    faultLogCount,                  \* 0..1: automatic fault callback count
    nativeBreadcrumbs,              \* 0..MaxNativeItems
    nativeAllocations,              \* 0..MaxNativeItems
    nativeAvailable,                \* BOOLEAN: ID3D12DeviceRemovedExtendedData1
    cachedReportValid,              \* BOOLEAN: D3D12Device::deviceFaultReport_
    cachedDredAvailable,            \* BOOLEAN
    cachedBreadcrumbs,              \* 0..MaxBreadcrumbs
    cachedAllocations,              \* 0..MaxAllocations
    cachedBreadcrumbsTruncated,     \* BOOLEAN
    cachedAllocationsTruncated,     \* BOOLEAN
    snapshotValid,                  \* BOOLEAN: caller-captured DeviceFaultReport
    snapshotLost,                   \* BOOLEAN
    snapshotDredAvailable,          \* BOOLEAN
    snapshotBreadcrumbs,            \* 0..MaxBreadcrumbs
    snapshotAllocations,            \* 0..MaxAllocations
    snapshotBreadcrumbsTruncated,   \* BOOLEAN
    snapshotAllocationsTruncated,   \* BOOLEAN
    lastAction                      \* Actions

cachedVars == <<cachedReportValid, cachedDredAvailable, cachedBreadcrumbs,
                cachedAllocations, cachedBreadcrumbsTruncated,
                cachedAllocationsTruncated>>
snapshotVars == <<snapshotValid, snapshotLost, snapshotDredAvailable,
                  snapshotBreadcrumbs, snapshotAllocations,
                  snapshotBreadcrumbsTruncated, snapshotAllocationsTruncated>>
vars == <<deviceLost, lossObserved, faultLogCount, nativeBreadcrumbs,
          nativeAllocations, nativeAvailable, cachedReportValid,
          cachedDredAvailable, cachedBreadcrumbs, cachedAllocations,
          cachedBreadcrumbsTruncated, cachedAllocationsTruncated,
          snapshotValid, snapshotLost, snapshotDredAvailable,
          snapshotBreadcrumbs, snapshotAllocations,
          snapshotBreadcrumbsTruncated, snapshotAllocationsTruncated,
          lastAction>>

Bounded(count, limit) ==
    IF count < limit THEN count ELSE limit

ShouldReadNativeDred ==
    /\ nativeAvailable
    /\ (GateDredReadoutOnFault => deviceLost)

TypeOK ==
    /\ deviceLost \in BOOLEAN
    /\ lossObserved \in BOOLEAN
    /\ faultLogCount \in 0..1
    /\ nativeBreadcrumbs \in 0..MaxNativeItems
    /\ nativeAllocations \in 0..MaxNativeItems
    /\ nativeAvailable \in BOOLEAN
    /\ cachedReportValid \in BOOLEAN
    /\ cachedDredAvailable \in BOOLEAN
    /\ cachedBreadcrumbs \in 0..MaxBreadcrumbs
    /\ cachedAllocations \in 0..MaxAllocations
    /\ cachedBreadcrumbsTruncated \in BOOLEAN
    /\ cachedAllocationsTruncated \in BOOLEAN
    /\ snapshotValid \in BOOLEAN
    /\ snapshotLost \in BOOLEAN
    /\ snapshotDredAvailable \in BOOLEAN
    /\ snapshotBreadcrumbs \in 0..MaxBreadcrumbs
    /\ snapshotAllocations \in 0..MaxAllocations
    /\ snapshotBreadcrumbsTruncated \in BOOLEAN
    /\ snapshotAllocationsTruncated \in BOOLEAN
    /\ lastAction \in Actions

Init ==
    /\ deviceLost = FALSE
    /\ lossObserved = FALSE
    /\ faultLogCount = 0
    /\ nativeBreadcrumbs = 0
    /\ nativeAllocations = 0
    /\ nativeAvailable = TRUE
    /\ cachedReportValid = FALSE
    /\ cachedDredAvailable = FALSE
    /\ cachedBreadcrumbs = 0
    /\ cachedAllocations = 0
    /\ cachedBreadcrumbsTruncated = FALSE
    /\ cachedAllocationsTruncated = FALSE
    /\ snapshotValid = FALSE
    /\ snapshotLost = FALSE
    /\ snapshotDredAvailable = FALSE
    /\ snapshotBreadcrumbs = 0
    /\ snapshotAllocations = 0
    /\ snapshotBreadcrumbsTruncated = FALSE
    /\ snapshotAllocationsTruncated = FALSE
    /\ lastAction = "none"

RecordBreadcrumb ==
    /\ ~deviceLost
    /\ nativeAvailable
    /\ nativeBreadcrumbs < MaxNativeItems
    /\ nativeBreadcrumbs' = nativeBreadcrumbs + 1
    /\ lastAction' = "record"
    /\ UNCHANGED <<deviceLost, lossObserved, faultLogCount, nativeAllocations,
                   nativeAvailable, cachedVars, snapshotVars>>

RecordAllocation ==
    /\ ~deviceLost
    /\ nativeAvailable
    /\ nativeAllocations < MaxNativeItems
    /\ nativeAllocations' = nativeAllocations + 1
    /\ lastAction' = "record"
    /\ UNCHANGED <<deviceLost, lossObserved, faultLogCount, nativeBreadcrumbs,
                   nativeAvailable, cachedVars, snapshotVars>>

RemoveDevice ==
    /\ ~deviceLost
    /\ deviceLost' = TRUE
    /\ lastAction' = "remove"
    /\ UNCHANGED <<lossObserved, faultLogCount, nativeBreadcrumbs,
                   nativeAllocations, nativeAvailable, cachedVars, snapshotVars>>

ObserveDeviceLoss ==
    /\ deviceLost
    /\ lastAction' = "observe"
    /\ IF ~lossObserved
       THEN /\ lossObserved' = TRUE
            /\ faultLogCount' = faultLogCount + 1
            /\ cachedReportValid' = TRUE
            /\ cachedDredAvailable' = nativeAvailable
            /\ cachedBreadcrumbs' =
                 IF nativeAvailable THEN Bounded(nativeBreadcrumbs, MaxBreadcrumbs) ELSE 0
            /\ cachedAllocations' =
                 IF nativeAvailable THEN Bounded(nativeAllocations, MaxAllocations) ELSE 0
            /\ cachedBreadcrumbsTruncated' =
                 (nativeAvailable /\ nativeBreadcrumbs > MaxBreadcrumbs)
            /\ cachedAllocationsTruncated' =
                 (nativeAvailable /\ nativeAllocations > MaxAllocations)
            /\ UNCHANGED <<deviceLost, nativeBreadcrumbs, nativeAllocations,
                           nativeAvailable, snapshotVars>>
       ELSE UNCHANGED <<deviceLost, lossObserved, faultLogCount,
                        nativeBreadcrumbs, nativeAllocations, nativeAvailable,
                        cachedVars, snapshotVars>>

Capture ==
    /\ lastAction' = "capture"
    /\ snapshotValid' = TRUE
    /\ IF cachedReportValid
       THEN /\ snapshotLost' = TRUE
            /\ snapshotDredAvailable' = cachedDredAvailable
            /\ snapshotBreadcrumbs' = cachedBreadcrumbs
            /\ snapshotAllocations' = cachedAllocations
            /\ snapshotBreadcrumbsTruncated' = cachedBreadcrumbsTruncated
            /\ snapshotAllocationsTruncated' = cachedAllocationsTruncated
       ELSE /\ snapshotLost' = deviceLost
            /\ snapshotDredAvailable' = nativeAvailable
            /\ snapshotBreadcrumbs' =
                 IF ShouldReadNativeDred
                 THEN Bounded(nativeBreadcrumbs, MaxBreadcrumbs)
                 ELSE 0
            /\ snapshotAllocations' =
                 IF ShouldReadNativeDred
                 THEN Bounded(nativeAllocations, MaxAllocations)
                 ELSE 0
            /\ snapshotBreadcrumbsTruncated' =
                 (ShouldReadNativeDred /\ nativeBreadcrumbs > MaxBreadcrumbs)
            /\ snapshotAllocationsTruncated' =
                 (ShouldReadNativeDred /\ nativeAllocations > MaxAllocations)
    /\ UNCHANGED <<deviceLost, lossObserved, faultLogCount, nativeBreadcrumbs,
                   nativeAllocations, nativeAvailable, cachedVars>>

DiscardNativeDiagnostics ==
    /\ nativeAvailable
    /\ (snapshotValid \/ cachedReportValid)
    /\ nativeBreadcrumbs' = 0
    /\ nativeAllocations' = 0
    /\ nativeAvailable' = FALSE
    /\ lastAction' = "discardNative"
    /\ UNCHANGED <<deviceLost, lossObserved, faultLogCount, cachedVars,
                   snapshotVars>>

Next ==
    \/ RecordBreadcrumb
    \/ RecordAllocation
    \/ RemoveDevice
    \/ ObserveDeviceLoss
    \/ Capture
    \/ DiscardNativeDiagnostics

Spec == Init /\ [][Next]_vars

AutomaticLoggingIsOnceAndTruthful ==
    /\ faultLogCount <= 1
    /\ (lossObserved <=> (faultLogCount = 1 /\ cachedReportValid))
    /\ (lossObserved => deviceLost)

\* Capturing a fault report while the device is healthy returns empty
\* breadcrumbs and page-fault allocation lists (D3D12Diagnostics.cpp).
HealthyCaptureHasEmptyDiagnostics ==
    (snapshotValid /\ ~snapshotLost) =>
        /\ snapshotBreadcrumbs = 0
        /\ snapshotAllocations = 0
        /\ ~snapshotBreadcrumbsTruncated
        /\ ~snapshotAllocationsTruncated

SnapshotsBoundedAndConsistent ==
    /\ (snapshotValid =>
           /\ snapshotBreadcrumbs <= MaxBreadcrumbs
           /\ snapshotAllocations <= MaxAllocations
           /\ (snapshotBreadcrumbsTruncated => snapshotBreadcrumbs = MaxBreadcrumbs)
           /\ (snapshotAllocationsTruncated => snapshotAllocations = MaxAllocations)
           /\ (~snapshotDredAvailable =>
                  snapshotBreadcrumbs = 0 /\ snapshotAllocations = 0))
    /\ (cachedReportValid =>
           /\ deviceLost
           /\ cachedBreadcrumbs <= MaxBreadcrumbs
           /\ cachedAllocations <= MaxAllocations
           /\ (cachedBreadcrumbsTruncated => cachedBreadcrumbs = MaxBreadcrumbs)
           /\ (cachedAllocationsTruncated => cachedAllocations = MaxAllocations))

Safety ==
    /\ AutomaticLoggingIsOnceAndTruthful
    /\ HealthyCaptureHasEmptyDiagnostics
    /\ SnapshotsBoundedAndConsistent

StepSafety ==
    [][/\ (deviceLost => deviceLost')
       /\ (lossObserved => lossObserved')
       /\ (faultLogCount' >= faultLogCount)
       /\ (lastAction' = "discardNative" =>
              /\ cachedVars' = cachedVars
              /\ snapshotVars' = snapshotVars)]_vars
=============================================================================
