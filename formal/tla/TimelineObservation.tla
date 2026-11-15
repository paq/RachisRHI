------------------------ MODULE TimelineObservation -------------------------
(***************************************************************************)
(* Formal safety model for point-scoped timeline queries, bounded and      *)
(* unbounded waits, monotonic completion, device-loss error reporting, and *)
(* lock-free C ABI handle waits:                                           *)
(*   - include/Rachis/Device.h (queryTimelinePoint, wait)                  *)
(*   - src/C/rachis.cpp (rachis_timeline_wait, rachis_timeline_release)    *)
(*   - src/D3D12/D3D12Queue.cpp, src/D3D12/D3D12Device.cpp                 *)
(*   - src/Vulkan/VulkanQueues.cpp, src/Vulkan/VulkanDevice.cpp            *)
(*   - src/Metal/MetalCommand.mm (MetalDevice::queryTimelinePoint, wait)   *)
(*                                                                         *)
(* Models both direct C++ TimelinePoint observation and C ABI              *)
(* rachis_timeline_wait, which copies the TimelinePoint under a shared     *)
(* lock on device->mutex and blocks outside the lock so concurrent         *)
(* rachis_timeline_release or device loss can interleave with an active    *)
(* wait without revoking the copied point or mutating queue state.         *)
(* rachis_timeline_wait itself is unbounded; the model's bounded           *)
(* handle-wait mode over-approximates it.                                  *)
(*                                                                         *)
(* Trusted boundary: OS condition-variable / native fence signaling once   *)
(* GPU work finishes.                                                      *)
(***************************************************************************)
EXTENDS Naturals

Devices == {"local", "foreign"}
Timeouts == {"negative", "zero", "positive"}
WaitModes == {"none", "bounded", "unbounded"}
Operations == {
  "none",
  "issue",
  "gpuComplete",
  "loseDevice",
  "releaseHandle",
  "query",
  "boundedWait",
  "unboundedWait",
  "handleWaitStart",
  "handleWaitFinish"
}
Results == {"none", "pending", "complete", "error"}

VARIABLES
  issued,        \* Whether the native timeline point has been submitted.
  handleLive,    \* Whether the public C timeline handle remains in timelines map.
  completed,     \* Whether GPU execution has reached the timeline point.
  deviceLost,    \* Monotonic device-loss flag.
  activeWait,    \* In-flight lock-free wait mode ("none", "bounded", "unbounded").
  lastOperation, \* Tag of the latest transition.
  lastRequester, \* Device provenance supplied to the latest query/wait.
  lastTimeout,   \* Timeout kind supplied to the latest query/wait.
  lastAccepted,  \* Whether the latest observation/wait call succeeded.
  lastResult     \* Status returned by the latest observation/wait call.

vars == <<
  issued,
  handleLive,
  completed,
  deviceLost,
  activeWait,
  lastOperation,
  lastRequester,
  lastTimeout,
  lastAccepted,
  lastResult
>>

Init ==
  /\ issued = FALSE
  /\ handleLive = FALSE
  /\ completed = FALSE
  /\ deviceLost = FALSE
  /\ activeWait = "none"
  /\ lastOperation = "none"
  /\ lastRequester = "local"
  /\ lastTimeout = "zero"
  /\ lastAccepted = FALSE
  /\ lastResult = "none"

issue ==
  /\ ~issued
  /\ ~deviceLost
  /\ issued' = TRUE
  /\ handleLive' = TRUE
  /\ lastOperation' = "issue"
  /\ lastRequester' = "local"
  /\ lastTimeout' = "zero"
  /\ lastAccepted' = TRUE
  /\ lastResult' = "none"
  /\ UNCHANGED <<completed, deviceLost, activeWait>>

gpuComplete ==
  /\ issued
  /\ ~completed
  /\ ~deviceLost
  /\ completed' = TRUE
  /\ lastOperation' = "gpuComplete"
  /\ lastRequester' = "local"
  /\ lastTimeout' = "zero"
  /\ lastAccepted' = TRUE
  /\ lastResult' = "none"
  /\ UNCHANGED <<issued, handleLive, deviceLost, activeWait>>

loseDevice ==
  /\ ~deviceLost
  /\ deviceLost' = TRUE
  /\ lastOperation' = "loseDevice"
  /\ lastRequester' = "local"
  /\ lastTimeout' = "zero"
  /\ lastAccepted' = TRUE
  /\ lastResult' = "none"
  /\ UNCHANGED <<issued, handleLive, completed, activeWait>>

\* rachis_timeline_release acquires device->mutex and erases the C handle
\* even while an active rachis_timeline_wait is blocked outside device->mutex.
releaseHandle ==
  /\ handleLive
  /\ handleLive' = FALSE
  /\ lastOperation' = "releaseHandle"
  /\ lastRequester' = "local"
  /\ lastTimeout' = "zero"
  /\ lastAccepted' = TRUE
  /\ lastResult' = "none"
  /\ UNCHANGED <<issued, completed, deviceLost, activeWait>>

\* C++ Device::queryTimelinePoint: non-blocking provenance-checked snapshot.
query(requester) ==
  /\ requester \in Devices
  /\ LET accepted == issued /\ requester = "local" /\ ~deviceLost IN
     /\ lastOperation' = "query"
     /\ lastRequester' = requester
     /\ lastTimeout' = "zero"
     /\ lastAccepted' = accepted
     /\ lastResult' =
          IF ~accepted
          THEN "error"
          ELSE IF completed THEN "complete" ELSE "pending"
     /\ UNCHANGED <<issued, handleLive, completed, deviceLost, activeWait>>

\* C++ Device::wait(point, timeout): synchronous bounded wait.
boundedWait(requester, timeout) ==
  /\ requester \in Devices
  /\ timeout \in Timeouts
  /\ LET accepted ==
           issued /\ requester = "local" /\ timeout # "negative" /\ ~deviceLost
     IN /\ lastOperation' = "boundedWait"
        /\ lastRequester' = requester
        /\ lastTimeout' = timeout
        /\ lastAccepted' = accepted
        /\ lastResult' =
             IF ~accepted
             THEN "error"
             ELSE IF completed THEN "complete" ELSE "pending"
        /\ UNCHANGED <<issued, handleLive, completed, deviceLost, activeWait>>

\* C++ Device::wait(point): synchronous unbounded wait (returns only when
\* completed, or errors immediately if invalid/foreign/lost).
unboundedWait(requester) ==
  /\ requester \in Devices
  /\ LET valid == issued /\ requester = "local" /\ ~deviceLost IN
     /\ (~valid \/ completed)
     /\ lastOperation' = "unboundedWait"
     /\ lastRequester' = requester
     /\ lastTimeout' = "positive"
     /\ lastAccepted' = valid
     /\ lastResult' = IF valid THEN "complete" ELSE "error"
     /\ UNCHANGED <<issued, handleLive, completed, deviceLost, activeWait>>

\* Step 1 of rachis_timeline_wait: resolves C handle under device->mutex and
\* copies TimelinePoint before releasing device->mutex to block.
beginHandleWait(requester, mode) ==
  /\ activeWait = "none"
  /\ requester \in Devices
  /\ mode \in {"bounded", "unbounded"}
  /\ LET valid == handleLive /\ requester = "local" /\ ~deviceLost IN
     IF valid
     THEN /\ activeWait' = mode
          /\ lastOperation' = "handleWaitStart"
          /\ lastRequester' = requester
          /\ lastTimeout' = "positive"
          /\ lastAccepted' = TRUE
          /\ lastResult' = "none"
          /\ UNCHANGED <<issued, handleLive, completed, deviceLost>>
     ELSE /\ lastOperation' = "handleWaitStart"
          /\ lastRequester' = requester
          /\ lastTimeout' = "positive"
          /\ lastAccepted' = FALSE
          /\ lastResult' = "error"
          /\ UNCHANGED <<issued, handleLive, completed, deviceLost, activeWait>>

\* Step 2 of rachis_timeline_wait: finishes outside device->mutex using the
\* copied TimelinePoint even if releaseHandle ran concurrently.
finishHandleWait ==
  /\ activeWait # "none"
  /\ (deviceLost \/ completed \/ activeWait = "bounded")
  /\ activeWait' = "none"
  /\ lastOperation' = "handleWaitFinish"
  /\ lastRequester' = "local"
  /\ lastTimeout' = "positive"
  /\ IF deviceLost
     THEN /\ lastAccepted' = FALSE
          /\ lastResult' = "error"
     ELSE IF completed
          THEN /\ lastAccepted' = TRUE
               /\ lastResult' = "complete"
          ELSE /\ lastAccepted' = TRUE
               /\ lastResult' = "pending"
  /\ UNCHANGED <<issued, handleLive, completed, deviceLost>>

Next ==
  \/ issue
  \/ gpuComplete
  \/ loseDevice
  \/ releaseHandle
  \/ \E r \in Devices : query(r)
  \/ \E r \in Devices, t \in Timeouts : boundedWait(r, t)
  \/ \E r \in Devices : unboundedWait(r)
  \/ \E r \in Devices, m \in {"bounded", "unbounded"} : beginHandleWait(r, m)
  \/ finishHandleWait

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ issued \in BOOLEAN
  /\ handleLive \in BOOLEAN
  /\ completed \in BOOLEAN
  /\ deviceLost \in BOOLEAN
  /\ activeWait \in WaitModes
  /\ lastOperation \in Operations
  /\ lastRequester \in Devices
  /\ lastTimeout \in Timeouts
  /\ lastAccepted \in BOOLEAN
  /\ lastResult \in Results

TimelineStateIsWellFormed ==
  /\ (completed => issued)
  /\ (handleLive => issued)
  /\ (activeWait # "none" => issued)

ForeignUnissuedOrLostObservationsReject ==
  lastOperation \in {"query", "boundedWait", "unboundedWait"} =>
    (lastAccepted => lastRequester = "local" /\ issued /\ ~deviceLost)

NegativeTimeoutsReject ==
  (lastOperation = "boundedWait" /\ lastTimeout = "negative") =>
    /\ ~lastAccepted
    /\ lastResult = "error"

ReturnedStatusMatchesCompletion ==
  /\ (lastOperation \in {"query", "boundedWait"} =>
        IF lastAccepted
        THEN lastResult = (IF completed THEN "complete" ELSE "pending")
        ELSE lastResult = "error")
  /\ (lastOperation = "unboundedWait" =>
        IF lastAccepted
        THEN lastResult = "complete" /\ completed
        ELSE lastResult = "error")
  /\ (lastOperation = "handleWaitFinish" =>
        IF deviceLost
        THEN ~lastAccepted /\ lastResult = "error"
        ELSE /\ lastAccepted
             /\ lastResult = (IF completed THEN "complete" ELSE "pending"))

PendingRequiresLiveIncompleteWork ==
  lastResult = "pending" =>
    /\ lastAccepted
    /\ issued
    /\ ~completed
    /\ ~deviceLost

Safety ==
  /\ TimelineStateIsWellFormed
  /\ ForeignUnissuedOrLostObservationsReject
  /\ NegativeTimeoutsReject
  /\ ReturnedStatusMatchesCompletion
  /\ PendingRequiresLiveIncompleteWork

StepSafety ==
  [][/\ (issued => issued')
     /\ (completed => completed')
     /\ (deviceLost => deviceLost')
     /\ (lastOperation' \in {
           "query",
           "boundedWait",
           "unboundedWait",
           "handleWaitStart",
           "handleWaitFinish"
         } =>
           /\ issued' = issued
           /\ handleLive' = handleLive
           /\ completed' = completed
           /\ deviceLost' = deviceLost)]_vars

=============================================================================
