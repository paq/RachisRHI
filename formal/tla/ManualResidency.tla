-------------------------- MODULE ManualResidency ---------------------------
(***************************************************************************)
(* Manual resource residency transitions, eviction gating, per-queue       *)
(* readiness propagation (`readyAfter`), and cross-queue fence ordering in *)
(* `src/D3D12/D3D12Residency.h`, `src/D3D12/D3D12Residency.cpp`,           *)
(* `src/D3D12/D3D12Resource.cpp`, `src/D3D12/D3D12Queue.cpp`, and          *)
(* `src/D3D12/D3D12RuntimePolicies.h` (`validateResidencySubmission` and   *)
(* `propagateResidencySubmission`).                                        *)
(*                                                                         *)
(* Models automatic vs manual pageable resources across two queues.        *)
(* Eviction succeeds only while a manual resource is `"resident"` and      *)
(* `"idle"` (rejected attempts are failure-atomic); `enqueueMakeResident`  *)
(* publishes an initial readiness point on its submission queue; same- and *)
(* cross-queue submissions preserve the earliest readiness point per queue *)
(* (`readyAfter[r][q]`) while enforcing GPU fence waits                    *)
(* (`completed[waitQueue] >= waitValue`) before queue completion advances. *)
(*                                                                         *)
(* Abstracted: OS video-memory paging internals and post-acceptance        *)
(* publication failures (covered by `ManualResidencyPublicationFailure`).  *)
(***************************************************************************)
EXTENDS Integers

Resources == {0, 1}
Queues == {0, 1}
MaxTimeline == 2
TimelineSteps == 1..MaxTimeline
TimelineValues == 0..MaxTimeline
None == -1
WaitQueues == {None} \cup Queues
States == {"automatic", "resident", "evicted", "pending"}
Phases == {"idle", "recording", "inFlight"}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    live,               \* resource handle liveness in [Resources -> BOOLEAN]
    manual,             \* manual-residency flag in [Resources -> BOOLEAN]
    independentOwner,   \* pageable owner id in [Resources -> {None} \cup Resources]
    state,              \* residency state in [Resources -> States]
    phase,              \* command usage phase in [Resources -> Phases]
    originQueue,        \* queue that enqueued makeResident while pending, or None
    originValue,        \* fence value of makeResident on originQueue, or 0
    readyAfter,         \* earliest fence ordering after makeResident per queue
    retireQueue,        \* queue retiring active inFlight work per resource
    retireValue,        \* fence value retiring active inFlight work per resource
    submitted,          \* latest submitted timeline value per queue
    completed,          \* latest completed timeline value per queue
    fenceWaitQueue,     \* cross-queue wait target per (queue, timeline step)
    fenceWaitValue,     \* cross-queue wait value per (queue, timeline step)
    lastOutcome         \* outcome of the latest step in Outcomes

coreVars == <<
    live, manual, independentOwner, state, phase,
    originQueue, originValue, readyAfter, retireQueue, retireValue,
    submitted, completed, fenceWaitQueue, fenceWaitValue
>>

vars == <<
    live, manual, independentOwner, state, phase,
    originQueue, originValue, readyAfter, retireQueue, retireValue,
    submitted, completed, fenceWaitQueue, fenceWaitValue, lastOutcome
>>

ZeroReady == [q \in Queues |-> 0]

ValidWait(waitQueue, waitValue) ==
    IF waitQueue = None
    THEN waitValue = 0
    ELSE /\ waitValue > 0
         /\ waitValue <= submitted[waitQueue]

OrderedForPending(resource, queue, waitQueue, waitValue) ==
    IF readyAfter[resource][queue] > 0
    THEN TRUE
    ELSE IF waitQueue = None
         THEN FALSE
         ELSE /\ readyAfter[resource][waitQueue] > 0
              /\ waitValue >= readyAfter[resource][waitQueue]
              /\ waitValue <= submitted[waitQueue]

Init ==
    /\ live = [r \in Resources |-> FALSE]
    /\ manual = [r \in Resources |-> FALSE]
    /\ independentOwner = [r \in Resources |-> None]
    /\ state = [r \in Resources |-> "automatic"]
    /\ phase = [r \in Resources |-> "idle"]
    /\ originQueue = [r \in Resources |-> None]
    /\ originValue = [r \in Resources |-> 0]
    /\ readyAfter = [r \in Resources |-> ZeroReady]
    /\ retireQueue = [r \in Resources |-> 0]
    /\ retireValue = [r \in Resources |-> 0]
    /\ submitted = [q \in Queues |-> 0]
    /\ completed = [q \in Queues |-> 0]
    /\ fenceWaitQueue = [q \in Queues |-> [t \in TimelineSteps |-> None]]
    /\ fenceWaitValue = [q \in Queues |-> [t \in TimelineSteps |-> 0]]
    /\ lastOutcome = "none"

create(resource, usesManualResidency) ==
    /\ ~live[resource]
    /\ live' = [live EXCEPT ![resource] = TRUE]
    /\ manual' = [manual EXCEPT ![resource] = usesManualResidency]
    /\ independentOwner' =
        [independentOwner EXCEPT ![resource] = IF usesManualResidency THEN resource ELSE None]
    /\ state' =
        [state EXCEPT ![resource] = IF usesManualResidency THEN "resident" ELSE "automatic"]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        phase, originQueue, originValue, readyAfter, retireQueue, retireValue,
        submitted, completed, fenceWaitQueue, fenceWaitValue
       >>

attemptRecord(resource) ==
    LET accepted ==
            /\ live[resource]
            /\ phase[resource] = "idle"
            /\ state[resource] # "evicted"
            /\ (state[resource] = "pending" => \E q \in Queues : readyAfter[resource][q] > 0)
    IN /\ live[resource]
       /\ phase[resource] = "idle"
       /\ IF accepted
          THEN /\ phase' = [phase EXCEPT ![resource] = "recording"]
               /\ UNCHANGED <<
                    live, manual, independentOwner, state, originQueue,
                    originValue, readyAfter, retireQueue, retireValue,
                    submitted, completed, fenceWaitQueue, fenceWaitValue
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

abandon(resource) ==
    /\ phase[resource] = "recording"
    /\ phase' = [phase EXCEPT ![resource] = "idle"]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        live, manual, independentOwner, state, originQueue, originValue,
        readyAfter, retireQueue, retireValue, submitted, completed,
        fenceWaitQueue, fenceWaitValue
       >>

attemptEvict(resource) ==
    LET accepted ==
            /\ live[resource]
            /\ manual[resource]
            /\ state[resource] = "resident"
            /\ phase[resource] = "idle"
    IN /\ live[resource]
       /\ IF accepted
          THEN /\ state' = [state EXCEPT ![resource] = "evicted"]
               /\ originQueue' = [originQueue EXCEPT ![resource] = None]
               /\ originValue' = [originValue EXCEPT ![resource] = 0]
               /\ readyAfter' = [readyAfter EXCEPT ![resource] = ZeroReady]
               /\ UNCHANGED <<
                    live, manual, independentOwner, phase, retireQueue,
                    retireValue, submitted, completed, fenceWaitQueue,
                    fenceWaitValue
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

enqueueMakeResident(resource, queue) ==
    /\ live[resource]
    /\ manual[resource]
    /\ state[resource] = "evicted"
    /\ phase[resource] = "idle"
    /\ submitted[queue] < MaxTimeline
    /\ LET nextValue == submitted[queue] + 1
       IN /\ state' = [state EXCEPT ![resource] = "pending"]
          /\ originQueue' = [originQueue EXCEPT ![resource] = queue]
          /\ originValue' = [originValue EXCEPT ![resource] = nextValue]
          /\ readyAfter' =
               [readyAfter EXCEPT ![resource] =
                   [q \in Queues |-> IF q = queue THEN nextValue ELSE 0]]
          /\ submitted' = [submitted EXCEPT ![queue] = nextValue]
          /\ fenceWaitQueue' = [fenceWaitQueue EXCEPT ![queue][nextValue] = None]
          /\ fenceWaitValue' = [fenceWaitValue EXCEPT ![queue][nextValue] = 0]
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<live, manual, independentOwner, phase, retireQueue, retireValue, completed>>

attemptSubmit(resource, queue, waitQueue, waitValue) ==
    /\ phase[resource] = "recording"
    /\ submitted[queue] < MaxTimeline
    /\ (IF waitQueue = None THEN waitValue = 0 ELSE TRUE)
    /\ LET nextValue == submitted[queue] + 1
           pending == state[resource] = "pending"
           accepted ==
               /\ phase[resource] = "recording"
               /\ state[resource] # "evicted"
               /\ submitted[queue] < MaxTimeline
               /\ ValidWait(waitQueue, waitValue)
               /\ (IF pending THEN OrderedForPending(resource, queue, waitQueue, waitValue) ELSE TRUE)
       IN /\ IF accepted
             THEN /\ phase' = [phase EXCEPT ![resource] = "inFlight"]
                  /\ readyAfter' =
                       IF pending /\ readyAfter[resource][queue] = 0
                       THEN [readyAfter EXCEPT ![resource][queue] = nextValue]
                       ELSE readyAfter
                  /\ retireQueue' = [retireQueue EXCEPT ![resource] = queue]
                  /\ retireValue' = [retireValue EXCEPT ![resource] = nextValue]
                  /\ submitted' = [submitted EXCEPT ![queue] = nextValue]
                  /\ fenceWaitQueue' =
                       [fenceWaitQueue EXCEPT ![queue][nextValue] =
                           IF waitQueue # queue THEN waitQueue ELSE None]
                  /\ fenceWaitValue' =
                       [fenceWaitValue EXCEPT ![queue][nextValue] =
                           IF waitQueue # queue /\ waitQueue # None THEN waitValue ELSE 0]
                  /\ UNCHANGED <<
                       live, manual, independentOwner, state,
                       originQueue, originValue, completed
                     >>
             ELSE UNCHANGED coreVars
          /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

\* GPU timeline step v completes only after any enqueued cross-queue wait completes.
complete(queue) ==
    /\ completed[queue] < submitted[queue]
    /\ LET nextDone == completed[queue] + 1
           wq == fenceWaitQueue[queue][nextDone]
           wv == fenceWaitValue[queue][nextDone]
       IN /\ (IF wq = None THEN TRUE ELSE completed[wq] >= wv)
          /\ completed' = [completed EXCEPT ![queue] = nextDone]
          /\ lastOutcome' = "none"
          /\ UNCHANGED <<
               live, manual, independentOwner, state, phase,
               originQueue, originValue, readyAfter, retireQueue,
               retireValue, submitted, fenceWaitQueue, fenceWaitValue
             >>

promote(resource) ==
    /\ state[resource] = "pending"
    /\ \E q \in Queues :
        /\ readyAfter[resource][q] > 0
        /\ completed[q] >= readyAfter[resource][q]
    /\ state' = [state EXCEPT ![resource] = "resident"]
    /\ originQueue' = [originQueue EXCEPT ![resource] = None]
    /\ originValue' = [originValue EXCEPT ![resource] = 0]
    /\ readyAfter' = [readyAfter EXCEPT ![resource] = ZeroReady]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        live, manual, independentOwner, phase, retireQueue,
        retireValue, submitted, completed, fenceWaitQueue, fenceWaitValue
       >>

collect(resource) ==
    /\ phase[resource] = "inFlight"
    /\ completed[retireQueue[resource]] >= retireValue[resource]
    /\ phase' = [phase EXCEPT ![resource] = "idle"]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        live, manual, independentOwner, state, originQueue, originValue,
        readyAfter, retireQueue, retireValue, submitted, completed,
        fenceWaitQueue, fenceWaitValue
       >>

Next ==
    \/ \E r \in Resources, m \in BOOLEAN : create(r, m)
    \/ \E r \in Resources :
        \/ attemptRecord(r)
        \/ abandon(r)
        \/ attemptEvict(r)
        \/ promote(r)
        \/ collect(r)
    \/ \E r \in Resources, q \in Queues : enqueueMakeResident(r, q)
    \/ \E r \in Resources, q \in Queues, wq \in WaitQueues, wv \in TimelineValues :
        attemptSubmit(r, q, wq, wv)
    \/ \E q \in Queues : complete(q)

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ live \in [Resources -> BOOLEAN]
    /\ manual \in [Resources -> BOOLEAN]
    /\ independentOwner \in [Resources -> {None} \cup Resources]
    /\ state \in [Resources -> States]
    /\ phase \in [Resources -> Phases]
    /\ originQueue \in [Resources -> {None} \cup Queues]
    /\ originValue \in [Resources -> TimelineValues]
    /\ readyAfter \in [Resources -> [Queues -> TimelineValues]]
    /\ retireQueue \in [Resources -> Queues]
    /\ retireValue \in [Resources -> TimelineValues]
    /\ submitted \in [Queues -> TimelineValues]
    /\ completed \in [Queues -> TimelineValues]
    /\ fenceWaitQueue \in [Queues -> [TimelineSteps -> WaitQueues]]
    /\ fenceWaitValue \in [Queues -> [TimelineSteps -> TimelineValues]]
    /\ lastOutcome \in Outcomes
    /\ \A q \in Queues : completed[q] <= submitted[q]

OwnershipMatchesResidencyMode ==
    \A r \in Resources :
        /\ (live[r] /\ ~manual[r]) =>
            /\ state[r] = "automatic"
            /\ independentOwner[r] = None
        /\ (live[r] /\ manual[r]) =>
            /\ state[r] # "automatic"
            /\ independentOwner[r] = r

EvictedResourcesUnused ==
    \A r \in Resources :
        state[r] = "evicted" => phase[r] = "idle"

PendingReadinessConsistent ==
    \A r \in Resources :
        state[r] = "pending" =>
            /\ originQueue[r] \in Queues
            /\ originValue[r] > 0
            /\ readyAfter[r][originQueue[r]] = originValue[r]
            /\ \A q \in Queues : readyAfter[r][q] <= submitted[q]

PropagatedCompletionImpliesOriginComplete ==
    \A r \in Resources, q \in Queues :
        (state[r] = "pending" /\ readyAfter[r][q] > 0 /\ completed[q] >= readyAfter[r][q]) =>
            completed[originQueue[r]] >= originValue[r]

InFlightOrderedAfterMakeResident ==
    \A r \in Resources :
        phase[r] = "inFlight" =>
            /\ retireValue[r] > 0
            /\ retireValue[r] <= submitted[retireQueue[r]]
            /\ (state[r] = "pending" =>
                    /\ readyAfter[r][retireQueue[r]] > 0
                    /\ readyAfter[r][retireQueue[r]] <= retireValue[r]
                    /\ (completed[retireQueue[r]] >= retireValue[r] =>
                            completed[originQueue[r]] >= originValue[r]))

Safety ==
    /\ OwnershipMatchesResidencyMode
    /\ EvictedResourcesUnused
    /\ PendingReadinessConsistent
    /\ PropagatedCompletionImpliesOriginComplete
    /\ InFlightOrderedAfterMakeResident

StepSafety ==
    [][
        /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
        /\ \A q \in Queues :
            /\ submitted[q] <= submitted'[q]
            /\ completed[q] <= completed'[q]
        /\ \A r \in Resources, q \in Queues :
            (state[r] = "pending" /\ state'[r] = "pending" /\ readyAfter[r][q] > 0) =>
                readyAfter'[r][q] = readyAfter[r][q]
    ]_vars

=============================================================================
