--------------------- MODULE ResourceCommandPublication ---------------------
(***************************************************************************)
(* Transactional publication and rollback of multi-family resource command *)
(* dependencies (application resource ownership, manual-residency leases,  *)
(* internal backend helper ownership + residency, swapchain image lease    *)
(* batches, and sparse mapping leases + timeline dependency batches).      *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Resource.cpp (retainResourceCommandDependencies)     *)
(*   - src/D3D12/D3D12ResourceCopies.cpp and                               *)
(*     src/D3D12/D3D12ResourceBarriers.cpp (callers of                     *)
(*     retainResourceCommandDependencies for copies and barriers)          *)
(*   - src/D3D12/D3D12Queue.cpp (QueueManager::retainCommandDependencies,  *)
(*     retainResource, retainSwapchainImageLeases,                         *)
(*     retainSparseMappingDependencies)                                    *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Up to MaxCommands = 2 commands execute per-stage retention with full  *)
(*   rollback on mid-transaction failure before native work is emitted.    *)
(*   Per-token retention counts prevent 2nd-command set-saturation, and    *)
(*   CommandPoolFlag::unretainedResources skips only application object    *)
(*   ownership while always publishing manual-residency leases.            *)
(***************************************************************************)

EXTENDS Naturals

MaxCommands == 2
ResourceTokens == {"app", "backend", "sparse1", "sparse2"}
ResidencyTokens == {"appRes", "backendRes"}
BatchTokens == {1, 2}
Phases == {"idle", "resources", "swapchain", "timelines", "readyToEmit"}
Outcomes == {"none", "committed", "rolledBack"}

VARIABLES
  retainApplication,      \* Pool retains application object owners (~unretainedResources)
  phase,                  \* Current transactional publication stage
  work,                   \* Emitted native command count on the command list
  resCount,               \* Per-token retained resource counts on PoolEntry
  residencyCount,         \* Per-token residency lease counts on PoolEntry
  leaseCount,             \* Per-slot swapchain image lease batch counts
  timelineCount,          \* Per-slot sparse mapping timeline dependency counts
  savedWork,              \* Snapshot of work at the start of the latest transaction
  savedResCount,          \* Rollback checkpoint for resCount
  savedResidencyCount,    \* Rollback checkpoint for residencyCount
  savedLeaseCount,        \* Rollback checkpoint for leaseCount
  savedTimelineCount,     \* Rollback checkpoint for timelineCount
  usesApp,                \* Whether transaction references an application resource
  usesAppRes,             \* Whether application resource carries manual residency
  usesBackend,            \* Whether transaction uses an internal backend helper
  usesSwapchain,          \* Whether transaction retains a swapchain lease batch
  usesSparseOne,          \* Whether transaction captures a first sparse mapping
  usesSparseTwo,          \* Whether transaction captures a second sparse mapping
  observedWork,           \* Latest externally observed quiescent command count
  observedResCount,       \* Latest externally observed resource counts
  observedResidencyCount, \* Latest externally observed residency counts
  observedLeaseCount,     \* Latest externally observed swapchain lease counts
  observedTimelineCount,  \* Latest externally observed sparse timeline counts
  lastOutcome             \* Outcome of the latest completed/aborted transaction

vars == <<
  retainApplication, phase, work,
  resCount, residencyCount, leaseCount, timelineCount,
  savedWork, savedResCount, savedResidencyCount,
  savedLeaseCount, savedTimelineCount,
  usesApp, usesAppRes, usesBackend, usesSwapchain,
  usesSparseOne, usesSparseTwo,
  observedWork, observedResCount, observedResidencyCount,
  observedLeaseCount, observedTimelineCount,
  lastOutcome
>>

ZeroRes == [t \in ResourceTokens |-> 0]
ZeroResidency == [t \in ResidencyTokens |-> 0]
ZeroBatch == [t \in BatchTokens |-> 0]

ExpectedResDelta(tok, uApp, uBackend, uSp1, uSp2) ==
  CASE tok = "app" -> IF uApp /\ retainApplication THEN 1 ELSE 0
    [] tok = "backend" -> IF uBackend THEN 1 ELSE 0
    [] tok = "sparse1" -> IF uSp1 THEN 1 ELSE 0
    [] tok = "sparse2" -> IF uSp2 THEN 1 ELSE 0

ExpectedResidencyDelta(tok, uAppRes, uBackend) ==
  IF tok = "appRes"
  THEN (IF uAppRes THEN 1 ELSE 0)
  ELSE (IF uBackend THEN 1 ELSE 0)

Init ==
  /\ retainApplication \in BOOLEAN
  /\ phase = "idle"
  /\ work = 0
  /\ resCount = ZeroRes
  /\ residencyCount = ZeroResidency
  /\ leaseCount = ZeroBatch
  /\ timelineCount = ZeroBatch
  /\ savedWork = 0
  /\ savedResCount = ZeroRes
  /\ savedResidencyCount = ZeroResidency
  /\ savedLeaseCount = ZeroBatch
  /\ savedTimelineCount = ZeroBatch
  /\ usesApp = FALSE
  /\ usesAppRes = FALSE
  /\ usesBackend = FALSE
  /\ usesSwapchain = FALSE
  /\ usesSparseOne = FALSE
  /\ usesSparseTwo = FALSE
  /\ observedWork = 0
  /\ observedResCount = ZeroRes
  /\ observedResidencyCount = ZeroResidency
  /\ observedLeaseCount = ZeroBatch
  /\ observedTimelineCount = ZeroBatch
  /\ lastOutcome = "none"

BeginCommand(uApp, uAppRes, uBackend, uSwap, uSp1, uSp2) ==
  /\ phase = "idle"
  /\ work < MaxCommands
  /\ (uApp \/ uBackend \/ uSwap \/ uSp1)
  /\ (uAppRes => uApp)
  /\ (uSp2 => uSp1)
  /\ phase' = "resources"
  /\ savedWork' = work
  /\ savedResCount' = resCount
  /\ savedResidencyCount' = residencyCount
  /\ savedLeaseCount' = leaseCount
  /\ savedTimelineCount' = timelineCount
  /\ usesApp' = uApp
  /\ usesAppRes' = uAppRes
  /\ usesBackend' = uBackend
  /\ usesSwapchain' = uSwap
  /\ usesSparseOne' = uSp1
  /\ usesSparseTwo' = uSp2
  /\ UNCHANGED <<
       retainApplication, work,
       resCount, residencyCount, leaseCount, timelineCount,
       observedWork, observedResCount, observedResidencyCount,
       observedLeaseCount, observedTimelineCount, lastOutcome
     >>

StageResources ==
  /\ phase = "resources"
  /\ phase' = "swapchain"
  /\ resCount' =
       [t \in ResourceTokens |->
          savedResCount[t]
            + ExpectedResDelta(t, usesApp, usesBackend, usesSparseOne, usesSparseTwo)]
  /\ residencyCount' =
       [t \in ResidencyTokens |->
          savedResidencyCount[t] + ExpectedResidencyDelta(t, usesAppRes, usesBackend)]
  /\ UNCHANGED <<
       retainApplication, work, leaseCount, timelineCount,
       savedWork, savedResCount, savedResidencyCount,
       savedLeaseCount, savedTimelineCount,
       usesApp, usesAppRes, usesBackend, usesSwapchain,
       usesSparseOne, usesSparseTwo,
       observedWork, observedResCount, observedResidencyCount,
       observedLeaseCount, observedTimelineCount, lastOutcome
     >>

StageSwapchain ==
  /\ phase = "swapchain"
  /\ phase' = "timelines"
  /\ leaseCount' =
       [t \in BatchTokens |-> savedLeaseCount[t] + (IF usesSwapchain THEN 1 ELSE 0)]
  /\ UNCHANGED <<
       retainApplication, work, resCount, residencyCount, timelineCount,
       savedWork, savedResCount, savedResidencyCount,
       savedLeaseCount, savedTimelineCount,
       usesApp, usesAppRes, usesBackend, usesSwapchain,
       usesSparseOne, usesSparseTwo,
       observedWork, observedResCount, observedResidencyCount,
       observedLeaseCount, observedTimelineCount, lastOutcome
     >>

StageTimelines ==
  /\ phase = "timelines"
  /\ phase' = "readyToEmit"
  /\ timelineCount' =
       [t \in BatchTokens |-> savedTimelineCount[t] + (IF usesSparseOne THEN 1 ELSE 0)]
  /\ UNCHANGED <<
       retainApplication, work, resCount, residencyCount, leaseCount,
       savedWork, savedResCount, savedResidencyCount,
       savedLeaseCount, savedTimelineCount,
       usesApp, usesAppRes, usesBackend, usesSwapchain,
       usesSparseOne, usesSparseTwo,
       observedWork, observedResCount, observedResidencyCount,
       observedLeaseCount, observedTimelineCount, lastOutcome
     >>

\* Any retention stage may fail (OOM or invalid sparse dependency) and roll back
\* all four containers to the pre-command checkpoint before returning to idle.
AbortAndRollback ==
  /\ phase \in {"resources", "swapchain", "timelines"}
  /\ phase' = "idle"
  /\ resCount' = savedResCount
  /\ residencyCount' = savedResidencyCount
  /\ leaseCount' = savedLeaseCount
  /\ timelineCount' = savedTimelineCount
  /\ lastOutcome' = "rolledBack"
  /\ UNCHANGED <<
       retainApplication, work,
       savedWork, savedResCount, savedResidencyCount,
       savedLeaseCount, savedTimelineCount,
       usesApp, usesAppRes, usesBackend, usesSwapchain,
       usesSparseOne, usesSparseTwo,
       observedWork, observedResCount, observedResidencyCount,
       observedLeaseCount, observedTimelineCount
     >>

CommitCommand ==
  /\ phase = "readyToEmit"
  /\ phase' = "idle"
  /\ work' = work + 1
  /\ lastOutcome' = "committed"
  /\ UNCHANGED <<
       retainApplication, resCount, residencyCount, leaseCount, timelineCount,
       savedWork, savedResCount, savedResidencyCount,
       savedLeaseCount, savedTimelineCount,
       usesApp, usesAppRes, usesBackend, usesSwapchain,
       usesSparseOne, usesSparseTwo,
       observedWork, observedResCount, observedResidencyCount,
       observedLeaseCount, observedTimelineCount
     >>

ObserveQuiescent ==
  /\ phase = "idle"
  /\ observedWork' = work
  /\ observedResCount' = resCount
  /\ observedResidencyCount' = residencyCount
  /\ observedLeaseCount' = leaseCount
  /\ observedTimelineCount' = timelineCount
  /\ UNCHANGED <<
       retainApplication, phase, work,
       resCount, residencyCount, leaseCount, timelineCount,
       savedWork, savedResCount, savedResidencyCount,
       savedLeaseCount, savedTimelineCount,
       usesApp, usesAppRes, usesBackend, usesSwapchain,
       usesSparseOne, usesSparseTwo, lastOutcome
     >>

AppChoices == {<<FALSE, FALSE>>, <<TRUE, FALSE>>, <<TRUE, TRUE>>}
SparseChoices == {<<FALSE, FALSE>>, <<TRUE, FALSE>>, <<TRUE, TRUE>>}

Next ==
  \/ \E ac \in AppChoices, uBack, uSwap \in BOOLEAN, sc \in SparseChoices :
       BeginCommand(ac[1], ac[2], uBack, uSwap, sc[1], sc[2])
  \/ StageResources
  \/ StageSwapchain
  \/ StageTimelines
  \/ AbortAndRollback
  \/ CommitCommand
  \/ ObserveQuiescent

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ retainApplication \in BOOLEAN
  /\ phase \in Phases
  /\ work \in 0..MaxCommands
  /\ resCount \in [ResourceTokens -> 0..MaxCommands]
  /\ residencyCount \in [ResidencyTokens -> 0..MaxCommands]
  /\ leaseCount \in [BatchTokens -> 0..MaxCommands]
  /\ timelineCount \in [BatchTokens -> 0..MaxCommands]
  /\ savedWork \in 0..MaxCommands
  /\ savedResCount \in [ResourceTokens -> 0..MaxCommands]
  /\ savedResidencyCount \in [ResidencyTokens -> 0..MaxCommands]
  /\ savedLeaseCount \in [BatchTokens -> 0..MaxCommands]
  /\ savedTimelineCount \in [BatchTokens -> 0..MaxCommands]
  /\ usesApp \in BOOLEAN
  /\ usesAppRes \in BOOLEAN
  /\ usesBackend \in BOOLEAN
  /\ usesSwapchain \in BOOLEAN
  /\ usesSparseOne \in BOOLEAN
  /\ usesSparseTwo \in BOOLEAN
  /\ observedWork \in 0..MaxCommands
  /\ observedResCount \in [ResourceTokens -> 0..MaxCommands]
  /\ observedResidencyCount \in [ResidencyTokens -> 0..MaxCommands]
  /\ observedLeaseCount \in [BatchTokens -> 0..MaxCommands]
  /\ observedTimelineCount \in [BatchTokens -> 0..MaxCommands]
  /\ lastOutcome \in Outcomes

CommandFlagsWellOrdered ==
  /\ (usesAppRes => usesApp)
  /\ (usesSparseTwo => usesSparseOne)

\* unretainedResources skips ONLY application object ownership; manual-residency
\* leases, internal backend helpers, and sparse mapping leases remain tracked.
UnretainedPoolSkipsOnlyApplicationOwner ==
  ~retainApplication =>
    /\ resCount["app"] = 0
    /\ observedResCount["app"] = 0

QuiescentContainersMatchLastTransaction ==
  phase = "idle" =>
    /\ resCount["backend"] = residencyCount["backendRes"]
    /\ leaseCount[1] = leaseCount[2]
    /\ timelineCount[1] = timelineCount[2]
    /\ resCount["sparse1"] = timelineCount[1]
    /\ resCount["sparse2"] <= resCount["sparse1"]
    /\ \A t \in ResourceTokens : resCount[t] <= work
    /\ \A t \in ResidencyTokens : residencyCount[t] <= work
    /\ \A t \in BatchTokens : leaseCount[t] <= work /\ timelineCount[t] <= work
    /\ (lastOutcome = "committed" =>
          /\ work = savedWork + 1
          /\ \A t \in ResourceTokens :
               resCount[t] =
                 savedResCount[t]
                   + ExpectedResDelta(t, usesApp, usesBackend, usesSparseOne, usesSparseTwo)
          /\ \A t \in ResidencyTokens :
               residencyCount[t] =
                 savedResidencyCount[t] + ExpectedResidencyDelta(t, usesAppRes, usesBackend)
          /\ \A t \in BatchTokens :
               /\ leaseCount[t] = savedLeaseCount[t] + (IF usesSwapchain THEN 1 ELSE 0)
               /\ timelineCount[t] = savedTimelineCount[t] + (IF usesSparseOne THEN 1 ELSE 0))
    /\ (lastOutcome = "rolledBack" =>
          /\ work = savedWork
          /\ resCount = savedResCount
          /\ residencyCount = savedResidencyCount
          /\ leaseCount = savedLeaseCount
          /\ timelineCount = savedTimelineCount)

ObservationNeverSeesPartialPublication ==
  /\ observedResCount["backend"] = observedResidencyCount["backendRes"]
  /\ observedLeaseCount[1] = observedLeaseCount[2]
  /\ observedTimelineCount[1] = observedTimelineCount[2]
  /\ observedResCount["sparse1"] = observedTimelineCount[1]
  /\ observedResCount["sparse2"] <= observedResCount["sparse1"]
  /\ \A t \in ResourceTokens : observedResCount[t] <= observedWork
  /\ \A t \in ResidencyTokens : observedResidencyCount[t] <= observedWork
  /\ \A t \in BatchTokens :
       observedLeaseCount[t] <= observedWork /\ observedTimelineCount[t] <= observedWork

Safety ==
  /\ CommandFlagsWellOrdered
  /\ UnretainedPoolSkipsOnlyApplicationOwner
  /\ QuiescentContainersMatchLastTransaction
  /\ ObservationNeverSeesPartialPublication

StepSafety ==
  [][
    /\ retainApplication' = retainApplication
    /\ work' >= work
    /\ (work' > work => phase = "readyToEmit" /\ phase' = "idle" /\ work' = work + 1)
    /\ (phase # "idle" /\ phase' # "idle" =>
          /\ savedWork' = savedWork
          /\ savedResCount' = savedResCount
          /\ savedResidencyCount' = savedResidencyCount
          /\ savedLeaseCount' = savedLeaseCount
          /\ savedTimelineCount' = savedTimelineCount)
  ]_vars

=============================================================================
