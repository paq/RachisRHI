----------------------- MODULE ResourceHeapResidency ------------------------
(***************************************************************************)
(* Manual-residency lifecycle of ResourceHeap allocations and placed       *)
(* resources: per-queue readyAfter fence tracking, cross-queue wait        *)
(* propagation and GPU wait ordering, deferred placed-resource and heap    *)
(* destruction, and promotion/eviction gates.                              *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Queue.cpp (promoteCompleted,                         *)
(*     validateResidencyDependencies, propagateResidencyDependencies,      *)
(*     enqueueMakeResident)                                                *)
(*   - src/D3D12/D3D12RuntimePolicies.h (promoteCompletedResidency,        *)
(*     validateResidencySubmission, propagateResidencySubmission)          *)
(*   - src/D3D12/D3D12Residency.h, src/D3D12/D3D12Residency.cpp            *)
(*     (ResidencyRecord::readyAfter, ResidencyLease)                       *)
(*   - src/D3D12/D3D12Resource.cpp (createResourceHeap, evict,             *)
(*     enqueueMakeResident, destroyResourceHeap, placed buffer/texture     *)
(*     records)                                                            *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   One manual-residency heap, two placed resource slots, two queues, and *)
(*   timeline depth MaxTimeline = 2 cover cross-queue wait propagation     *)
(*   without erasing prior queue readiness and prevent premature promotion *)
(*   before the make-resident queue actually completes on the GPU.         *)
(***************************************************************************)

EXTENDS Naturals

Queues == {0, 1}
Resources == {0, 1}
MaxTimeline == 2
TimelineSlots == 1..MaxTimeline
ResidencyStates == {"evicted", "pending", "resident"}
Phases == {"idle", "recorded", "inFlight"}
Operations == {
  "none", "createResource", "destroyResource", "destroyHeap",
  "enqueueMakeResident", "record", "abandon", "submit",
  "complete", "promote", "collectResource", "collectMakeResident", "evictHeap"
}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  heapLogical,          \* Public ResourceHeapHandle liveness
  heapPhysical,         \* Native ID3D12Heap / ResourceHeapRecord liveness
  residencyState,       \* ResidencyState: "evicted", "pending", or "resident"
  readyAfter,           \* Per-queue fence value after which pending heap is ready
  makeResidentQueue,    \* Queue on which latest EnqueueMakeResident waited/signaled
  makeResidentValue,    \* Fence value signaled by latest EnqueueMakeResident (0 = none)
  makeResidentRetained, \* Whether queue retirement still retains make-resident lease
  resLogical,           \* Public handle liveness per placed resource
  resPhysical,          \* Native placed resource storage liveness per slot
  phase,                \* Command lifecycle per placed resource: idle/recorded/inFlight
  subQueue,             \* Submitted queue per in-flight placed resource
  subValue,             \* Submitted fence value per in-flight placed resource
  submitted,            \* Latest submitted fence value per queue
  completed,            \* Latest GPU-completed fence value per queue
  waitValue,            \* Per-(queue, timelineSlot) cross-queue wait on (1 - q) (0 = none)
  lastOp,               \* Most recent operation kind
  lastOutcome           \* Outcome of most recent operation

coreVars == <<
  heapLogical, heapPhysical, residencyState, readyAfter,
  makeResidentQueue, makeResidentValue, makeResidentRetained,
  resLogical, resPhysical, phase, subQueue, subValue,
  submitted, completed, waitValue
>>

vars == <<
  heapLogical, heapPhysical, residencyState, readyAfter,
  makeResidentQueue, makeResidentValue, makeResidentRetained,
  resLogical, resPhysical, phase, subQueue, subValue,
  submitted, completed, waitValue,
  lastOp, lastOutcome
>>

ZeroQueues == [q \in Queues |-> 0]
OtherQueue(q) == 1 - q

CanPromoteNow ==
  /\ residencyState = "pending"
  /\ \E q \in Queues : readyAfter[q] > 0 /\ completed[q] >= readyAfter[q]

SubmissionOrdered(q, wv) ==
  LET oq == OtherQueue(q) IN
  \/ residencyState = "resident"
  \/ CanPromoteNow
  \/ (residencyState = "pending"
      /\ (readyAfter[q] > 0
          \/ (wv > 0 /\ readyAfter[oq] > 0 /\ wv >= readyAfter[oq])))

CanSubmit(r, q, wv) ==
  LET oq == OtherQueue(q) IN
  /\ phase[r] = "recorded"
  /\ submitted[q] < MaxTimeline
  /\ wv <= submitted[oq]
  /\ residencyState # "evicted"
  /\ SubmissionOrdered(q, wv)

RecomputeHeapPhysical(nextHeapLog, nextMrRet, nextResPhys) ==
  nextHeapLog \/ nextMrRet \/ (\E r \in Resources : nextResPhys[r])

Init ==
  /\ heapLogical = TRUE
  /\ heapPhysical = TRUE
  /\ residencyState = "evicted"
  /\ readyAfter = ZeroQueues
  /\ makeResidentQueue = 0
  /\ makeResidentValue = 0
  /\ makeResidentRetained = FALSE
  /\ resLogical = [r \in Resources |-> FALSE]
  /\ resPhysical = [r \in Resources |-> FALSE]
  /\ phase = [r \in Resources |-> "idle"]
  /\ subQueue = [r \in Resources |-> 0]
  /\ subValue = [r \in Resources |-> 0]
  /\ submitted = ZeroQueues
  /\ completed = ZeroQueues
  /\ waitValue = [q \in Queues |-> [v \in TimelineSlots |-> 0]]
  /\ lastOp = "none"
  /\ lastOutcome = "none"

CreatePlacedResource(r) ==
  /\ heapLogical
  /\ ~resPhysical[r]
  /\ resLogical' = [resLogical EXCEPT ![r] = TRUE]
  /\ resPhysical' = [resPhysical EXCEPT ![r] = TRUE]
  /\ phase' = [phase EXCEPT ![r] = "idle"]
  /\ UNCHANGED <<
       heapLogical, heapPhysical, residencyState, readyAfter,
       makeResidentQueue, makeResidentValue, makeResidentRetained,
       subQueue, subValue, submitted, completed, waitValue
     >>
  /\ lastOp' = "createResource"
  /\ lastOutcome' = "accepted"

\* Placed resources may be destroyed while recorded or in-flight; physical
\* storage and parent heap retention persist until command retirement/abandon.
DestroyPlacedResource(r) ==
  LET nextResLog == [resLogical EXCEPT ![r] = FALSE]
      nextResPhys == [resPhysical EXCEPT ![r] = (phase[r] # "idle")]
  IN
  /\ resLogical[r]
  /\ resLogical' = nextResLog
  /\ resPhysical' = nextResPhys
  /\ heapPhysical' = RecomputeHeapPhysical(heapLogical, makeResidentRetained, nextResPhys)
  /\ UNCHANGED <<
       heapLogical, residencyState, readyAfter,
       makeResidentQueue, makeResidentValue, makeResidentRetained,
       phase, subQueue, subValue, submitted, completed, waitValue
     >>
  /\ lastOp' = "destroyResource"
  /\ lastOutcome' = "accepted"

DestroyHeap ==
  /\ heapLogical
  /\ heapLogical' = FALSE
  /\ heapPhysical' = RecomputeHeapPhysical(FALSE, makeResidentRetained, resPhysical)
  /\ UNCHANGED <<
       residencyState, readyAfter,
       makeResidentQueue, makeResidentValue, makeResidentRetained,
       resLogical, resPhysical, phase, subQueue, subValue,
       submitted, completed, waitValue
     >>
  /\ lastOp' = "destroyHeap"
  /\ lastOutcome' = "accepted"

AttemptEnqueueMakeResident(q) ==
  LET ok ==
        /\ heapLogical
        /\ residencyState = "evicted"
        /\ ~makeResidentRetained
        /\ submitted[q] < MaxTimeline
      nextVal == submitted[q] + 1
  IN
  /\ IF ok
     THEN /\ residencyState' = "pending"
          /\ readyAfter' = [q2 \in Queues |-> IF q2 = q THEN nextVal ELSE 0]
          /\ makeResidentQueue' = q
          /\ makeResidentValue' = nextVal
          /\ makeResidentRetained' = TRUE
          /\ submitted' = [submitted EXCEPT ![q] = nextVal]
          /\ waitValue' = [waitValue EXCEPT ![q][nextVal] = 0]
          /\ UNCHANGED <<
               heapLogical, heapPhysical, resLogical, resPhysical,
               phase, subQueue, subValue, completed
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "enqueueMakeResident"

RecordUse(r) ==
  /\ resLogical[r]
  /\ phase[r] = "idle"
  /\ phase' = [phase EXCEPT ![r] = "recorded"]
  /\ UNCHANGED <<
       heapLogical, heapPhysical, residencyState, readyAfter,
       makeResidentQueue, makeResidentValue, makeResidentRetained,
       resLogical, resPhysical, subQueue, subValue,
       submitted, completed, waitValue
     >>
  /\ lastOp' = "record"
  /\ lastOutcome' = "accepted"

AbandonRecorded(r) ==
  LET nextResPhys == [resPhysical EXCEPT ![r] = resLogical[r]] IN
  /\ phase[r] = "recorded"
  /\ phase' = [phase EXCEPT ![r] = "idle"]
  /\ resPhysical' = nextResPhys
  /\ heapPhysical' = RecomputeHeapPhysical(heapLogical, makeResidentRetained, nextResPhys)
  /\ UNCHANGED <<
       heapLogical, residencyState, readyAfter,
       makeResidentQueue, makeResidentValue, makeResidentRetained,
       resLogical, subQueue, subValue,
       submitted, completed, waitValue
     >>
  /\ lastOp' = "abandon"
  /\ lastOutcome' = "accepted"

AttemptSubmit(r, q, wv) ==
  LET nextVal == submitted[q] + 1
      promotedOnValidate == CanPromoteNow
  IN
  /\ phase[r] = "recorded"
  /\ IF CanSubmit(r, q, wv)
     THEN /\ phase' = [phase EXCEPT ![r] = "inFlight"]
          /\ subQueue' = [subQueue EXCEPT ![r] = q]
          /\ subValue' = [subValue EXCEPT ![r] = nextVal]
          /\ submitted' = [submitted EXCEPT ![q] = nextVal]
          /\ waitValue' = [waitValue EXCEPT ![q][nextVal] = wv]
          /\ residencyState' =
               IF promotedOnValidate THEN "resident" ELSE residencyState
          /\ readyAfter' =
               IF promotedOnValidate
               THEN ZeroQueues
               ELSE IF residencyState = "pending" /\ readyAfter[q] = 0
                    THEN [readyAfter EXCEPT ![q] = nextVal]
                    ELSE readyAfter
          /\ UNCHANGED <<
               heapLogical, heapPhysical,
               makeResidentQueue, makeResidentValue, makeResidentRetained,
               resLogical, resPhysical, completed
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "submit"

\* GPU completion on queue q respects any cross-queue fence wait queued on (1 - q).
Complete(q) ==
  LET nextDone == completed[q] + 1
      oq == OtherQueue(q)
      wv == waitValue[q][nextDone]
  IN
  /\ completed[q] < submitted[q]
  /\ (wv = 0 \/ completed[oq] >= wv)
  /\ completed' = [completed EXCEPT ![q] = nextDone]
  /\ UNCHANGED <<
       heapLogical, heapPhysical, residencyState, readyAfter,
       makeResidentQueue, makeResidentValue, makeResidentRetained,
       resLogical, resPhysical, phase, subQueue, subValue,
       submitted, waitValue
     >>
  /\ lastOp' = "complete"
  /\ lastOutcome' = "accepted"

PromoteHeap ==
  /\ CanPromoteNow
  /\ residencyState' = "resident"
  /\ readyAfter' = ZeroQueues
  /\ UNCHANGED <<
       heapLogical, heapPhysical,
       makeResidentQueue, makeResidentValue, makeResidentRetained,
       resLogical, resPhysical, phase, subQueue, subValue,
       submitted, completed, waitValue
     >>
  /\ lastOp' = "promote"
  /\ lastOutcome' = "accepted"

CollectResource(r) ==
  LET nextResPhys == [resPhysical EXCEPT ![r] = resLogical[r]] IN
  /\ phase[r] = "inFlight"
  /\ completed[subQueue[r]] >= subValue[r]
  /\ phase' = [phase EXCEPT ![r] = "idle"]
  /\ subQueue' = [subQueue EXCEPT ![r] = 0]
  /\ subValue' = [subValue EXCEPT ![r] = 0]
  /\ resPhysical' = nextResPhys
  /\ heapPhysical' = RecomputeHeapPhysical(heapLogical, makeResidentRetained, nextResPhys)
  /\ UNCHANGED <<
       heapLogical, residencyState, readyAfter,
       makeResidentQueue, makeResidentValue, makeResidentRetained,
       resLogical, submitted, completed, waitValue
     >>
  /\ lastOp' = "collectResource"
  /\ lastOutcome' = "accepted"

CollectMakeResident ==
  /\ makeResidentRetained
  /\ completed[makeResidentQueue] >= makeResidentValue
  /\ makeResidentRetained' = FALSE
  /\ heapPhysical' = RecomputeHeapPhysical(heapLogical, FALSE, resPhysical)
  /\ UNCHANGED <<
       heapLogical, residencyState, readyAfter,
       makeResidentQueue, makeResidentValue,
       resLogical, resPhysical, phase, subQueue, subValue,
       submitted, completed, waitValue
     >>
  /\ lastOp' = "collectMakeResident"
  /\ lastOutcome' = "accepted"

AttemptEvictHeap ==
  LET ok ==
        /\ heapLogical
        /\ residencyState = "resident"
        /\ ~makeResidentRetained
        /\ \A r \in Resources : phase[r] # "inFlight"
  IN
  /\ IF ok
     THEN /\ residencyState' = "evicted"
          /\ readyAfter' = ZeroQueues
          /\ UNCHANGED <<
               heapLogical, heapPhysical,
               makeResidentQueue, makeResidentValue, makeResidentRetained,
               resLogical, resPhysical, phase, subQueue, subValue,
               submitted, completed, waitValue
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "evictHeap"

Next ==
  \/ \E r \in Resources : CreatePlacedResource(r)
  \/ \E r \in Resources : DestroyPlacedResource(r)
  \/ DestroyHeap
  \/ \E q \in Queues : AttemptEnqueueMakeResident(q)
  \/ \E r \in Resources : RecordUse(r)
  \/ \E r \in Resources : AbandonRecorded(r)
  \/ \E r \in Resources, q \in Queues, wv \in 0..MaxTimeline :
       AttemptSubmit(r, q, wv)
  \/ \E q \in Queues : Complete(q)
  \/ PromoteHeap
  \/ \E r \in Resources : CollectResource(r)
  \/ CollectMakeResident
  \/ AttemptEvictHeap

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ heapLogical \in BOOLEAN
  /\ heapPhysical \in BOOLEAN
  /\ residencyState \in ResidencyStates
  /\ readyAfter \in [Queues -> 0..MaxTimeline]
  /\ makeResidentQueue \in Queues
  /\ makeResidentValue \in 0..MaxTimeline
  /\ makeResidentRetained \in BOOLEAN
  /\ resLogical \in [Resources -> BOOLEAN]
  /\ resPhysical \in [Resources -> BOOLEAN]
  /\ phase \in [Resources -> Phases]
  /\ subQueue \in [Resources -> Queues]
  /\ subValue \in [Resources -> 0..MaxTimeline]
  /\ submitted \in [Queues -> 0..MaxTimeline]
  /\ completed \in [Queues -> 0..MaxTimeline]
  /\ waitValue \in [Queues -> [TimelineSlots -> 0..MaxTimeline]]
  /\ lastOp \in Operations
  /\ lastOutcome \in Outcomes

PhysicalLifetimesAreExact ==
  /\ \A r \in Resources :
       resPhysical[r] = (resLogical[r] \/ phase[r] # "idle")
  /\ heapPhysical =
       (heapLogical \/ makeResidentRetained \/ (\E r \in Resources : resPhysical[r]))

InFlightPlacedUseNeverReferencesEvictedHeap ==
  \A r \in Resources :
    phase[r] = "inFlight" =>
      /\ heapPhysical
      /\ residencyState # "evicted"
      /\ subValue[r] > 0
      /\ subValue[r] <= submitted[subQueue[r]]

PendingReadyAfterpreservesMakeResidentQueue ==
  residencyState = "pending" =>
    /\ makeResidentValue > 0
    /\ readyAfter[makeResidentQueue] = makeResidentValue
    /\ \A q \in Queues : readyAfter[q] <= submitted[q]

\* Crucial theorem: promotion via ANY queue's readyAfter entry implies the
\* originating make-resident fence has actually completed on the GPU.
PromotedHeapHasCompletedMakeResident ==
  (residencyState = "resident" /\ makeResidentValue > 0) =>
    completed[makeResidentQueue] >= makeResidentValue

TimelinesMonotonicAndBounded ==
  \A q \in Queues : completed[q] <= submitted[q]

Safety ==
  /\ PhysicalLifetimesAreExact
  /\ InFlightPlacedUseNeverReferencesEvictedHeap
  /\ PendingReadyAfterpreservesMakeResidentQueue
  /\ PromotedHeapHasCompletedMakeResident
  /\ TimelinesMonotonicAndBounded

StepSafety ==
  [][
    /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
    /\ \A q \in Queues :
         /\ submitted'[q] >= submitted[q]
         /\ completed'[q] >= completed[q]
    /\ (residencyState = "pending" /\ residencyState' = "pending" =>
          \A q \in Queues :
            /\ readyAfter'[q] >= readyAfter[q]
            /\ (readyAfter[q] > 0 => readyAfter'[q] = readyAfter[q]))
  ]_vars

=============================================================================
