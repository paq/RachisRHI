------------------------ MODULE CommandPoolLifecycle ------------------------
(***************************************************************************)
(* Bounded command-pool acquisition, exhaustion/stale rejection,           *)
(* single-queue and cross-pool batch submission, duplicate-command         *)
(* rejection, logical handle destruction, and exact fence retirement.      *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Queue.cpp (CommandPoolState, D3D12CommandBuffer,     *)
(*     QueueManager::acquire, QueueManager::submit,                        *)
(*     QueueManager::destroyCommandPool)                                   *)
(*                                                                         *)
(* Contract modeled:                                                       *)
(*   - Two capacity-2 pools (Pools = {0, 1}, Slots = {0, 1, 2, 3}) over    *)
(*     Queues = {0, 1} with MaxTimeline = 2 per queue.                     *)
(*   - Acquisition succeeds iff the pool handle is logically alive and has *)
(*     an available entry; exhausted or stale-pool acquisitions reject     *)
(*     atomically without mutating pool or slot state.                     *)
(*   - Batch submission rejects duplicate command-buffer leases            *)
(*     (first = second, D3D12Queue.cpp) and mixed-queue batches            *)
(*     (when RequireMatchingBatchQueues = TRUE, D3D12Queue.cpp)            *)
(*     atomically without advancing timelines or consuming recordings.     *)
(*   - Logical pool destruction blocks new acquisitions immediately while  *)
(*     retaining physical pool storage until every recording or in-flight  *)
(*     entry abandons or completes its exact fence value.                  *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - Native D3D12 Close()/Reset() hardware failures that permanently     *)
(*     quarantine individual PoolEntry objects (healthy driver modeled).   *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT RequireMatchingBatchQueues

ASSUME RequireMatchingBatchQueues \in BOOLEAN

Pools == {0, 1}
Slots == {0, 1, 2, 3}
Queues == {0, 1}
MaxTimeline == 2
Phases == {"unavailable", "available", "recording", "inFlight"}
Outcomes == {"none", "accepted", "exhausted", "stale", "invalidBatch"}

VARIABLES
    logicalAlive,       \* [Pools -> BOOLEAN] public CommandPoolHandle alive
    physicalAlive,      \* [Pools -> BOOLEAN] CommandPoolState physically alive
    poolQueue,          \* [Pools -> Queues] queue owned by pool
    phase,              \* [Slots -> Phases] per-entry lifecycle state
    inFlightQueue,      \* [Slots -> Queues] queue executing an in-flight entry
    retireAt,           \* [Slots -> 0..MaxTimeline] fence retirement point
    submitted,          \* [Queues -> 0..MaxTimeline] submitted frontier
    completed,          \* [Queues -> 0..MaxTimeline] completed frontier
    lastOutcome         \* outcome of the most recent step

coreVars == <<logicalAlive, physicalAlive, poolQueue, phase,
              inFlightQueue, retireAt, submitted, completed>>
vars == <<logicalAlive, physicalAlive, poolQueue, phase,
          inFlightQueue, retireAt, submitted, completed, lastOutcome>>

PoolOf(s) == IF s < 2 THEN 0 ELSE 1

SlotsFor(p) == IF p = 0 THEN {0, 1} ELSE {2, 3}

HasOutstanding(p) ==
    \E s \in SlotsFor(p) : phase[s] \in {"recording", "inFlight"}

HasOtherOutstanding(p, excluded) ==
    \E s \in SlotsFor(p) : s # excluded /\ phase[s] \in {"recording", "inFlight"}

TypeOK ==
    /\ logicalAlive \in [Pools -> BOOLEAN]
    /\ physicalAlive \in [Pools -> BOOLEAN]
    /\ poolQueue \in [Pools -> Queues]
    /\ phase \in [Slots -> Phases]
    /\ inFlightQueue \in [Slots -> Queues]
    /\ retireAt \in [Slots -> 0..MaxTimeline]
    /\ submitted \in [Queues -> 0..MaxTimeline]
    /\ completed \in [Queues -> 0..MaxTimeline]
    /\ \A q \in Queues : completed[q] <= submitted[q]
    /\ lastOutcome \in Outcomes

Init ==
    /\ logicalAlive = [p \in Pools |-> FALSE]
    /\ physicalAlive = [p \in Pools |-> FALSE]
    /\ poolQueue = [p \in Pools |-> 0]
    /\ phase = [s \in Slots |-> "unavailable"]
    /\ inFlightQueue = [s \in Slots |-> 0]
    /\ retireAt = [s \in Slots |-> 0]
    /\ submitted = [q \in Queues |-> 0]
    /\ completed = [q \in Queues |-> 0]
    /\ lastOutcome = "none"

CreatePool(p, q) ==
    /\ ~physicalAlive[p]
    /\ logicalAlive' = [logicalAlive EXCEPT ![p] = TRUE]
    /\ physicalAlive' = [physicalAlive EXCEPT ![p] = TRUE]
    /\ poolQueue' = [poolQueue EXCEPT ![p] = q]
    /\ phase' = [s \in Slots |-> IF s \in SlotsFor(p) THEN "available" ELSE phase[s]]
    /\ retireAt' = [s \in Slots |-> IF s \in SlotsFor(p) THEN 0 ELSE retireAt[s]]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<inFlightQueue, submitted, completed>>

AcquireSlot(p, s) ==
    /\ s \in SlotsFor(p)
    /\ logicalAlive[p]
    /\ phase[s] = "available"
    /\ phase' = [phase EXCEPT ![s] = "recording"]
    /\ retireAt' = [retireAt EXCEPT ![s] = 0]
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<logicalAlive, physicalAlive, poolQueue, inFlightQueue,
                   submitted, completed>>

RejectExhaustedAcquire(p) ==
    /\ logicalAlive[p]
    /\ ~(\E s \in SlotsFor(p) : phase[s] = "available")
    /\ lastOutcome' = "exhausted"
    /\ UNCHANGED coreVars

RejectStaleAcquire(p) ==
    /\ ~logicalAlive[p]
    /\ lastOutcome' = "stale"
    /\ UNCHANGED coreVars

Abandon(s) ==
    /\ phase[s] = "recording"
    /\ LET p == PoolOf(s)
           releasePool == ~logicalAlive[p] /\ ~HasOtherOutstanding(p, s)
       IN /\ physicalAlive' = [physicalAlive EXCEPT ![p] = ~releasePool]
          /\ poolQueue' = IF releasePool THEN [poolQueue EXCEPT ![p] = 0] ELSE poolQueue
          /\ phase' =
               IF releasePool
               THEN [c \in Slots |-> IF c \in SlotsFor(p) THEN "unavailable" ELSE phase[c]]
               ELSE [phase EXCEPT ![s] = "available"]
          /\ inFlightQueue' = [inFlightQueue EXCEPT ![s] = 0]
          /\ retireAt' = [retireAt EXCEPT ![s] = 0]
          /\ lastOutcome' = "none"
          /\ UNCHANGED <<logicalAlive, submitted, completed>>

SubmitOne(s) ==
    /\ phase[s] = "recording"
    /\ LET p == PoolOf(s)
           q == poolQueue[p]
           nextVal == submitted[q] + 1
       IN /\ submitted[q] < MaxTimeline
          /\ phase' = [phase EXCEPT ![s] = "inFlight"]
          /\ inFlightQueue' = [inFlightQueue EXCEPT ![s] = q]
          /\ retireAt' = [retireAt EXCEPT ![s] = nextVal]
          /\ submitted' = [submitted EXCEPT ![q] = nextVal]
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<logicalAlive, physicalAlive, poolQueue, completed>>

AttemptSubmitBatch(first, second) ==
    /\ phase[first] = "recording"
    /\ phase[second] = "recording"
    /\ LET p1 == PoolOf(first)
           p2 == PoolOf(second)
           q == poolQueue[p1]
           valid ==
               /\ first # second
               /\ (RequireMatchingBatchQueues => poolQueue[p2] = q)
       IN /\ submitted[q] < MaxTimeline
          /\ IF valid
             THEN LET nextVal == submitted[q] + 1 IN
                  /\ phase' = [[phase EXCEPT ![first] = "inFlight"]
                                             EXCEPT ![second] = "inFlight"]
                  /\ inFlightQueue' = [[inFlightQueue EXCEPT ![first] = q]
                                                      EXCEPT ![second] = q]
                  /\ retireAt' = [[retireAt EXCEPT ![first] = nextVal]
                                            EXCEPT ![second] = nextVal]
                  /\ submitted' = [submitted EXCEPT ![q] = nextVal]
                  /\ lastOutcome' = "accepted"
                  /\ UNCHANGED <<logicalAlive, physicalAlive, poolQueue, completed>>
             ELSE /\ lastOutcome' = "invalidBatch"
                  /\ UNCHANGED coreVars

CompleteQueue(q) ==
    /\ completed[q] < submitted[q]
    /\ completed' = [completed EXCEPT ![q] = completed[q] + 1]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<logicalAlive, physicalAlive, poolQueue, phase,
                   inFlightQueue, retireAt, submitted>>

CollectSlot(s) ==
    /\ phase[s] = "inFlight"
    /\ LET p == PoolOf(s)
           q == inFlightQueue[s]
           releasePool == ~logicalAlive[p] /\ ~HasOtherOutstanding(p, s)
       IN /\ retireAt[s] <= completed[q]
          /\ physicalAlive' = [physicalAlive EXCEPT ![p] = ~releasePool]
          /\ poolQueue' = IF releasePool THEN [poolQueue EXCEPT ![p] = 0] ELSE poolQueue
          /\ phase' =
               IF releasePool
               THEN [c \in Slots |-> IF c \in SlotsFor(p) THEN "unavailable" ELSE phase[c]]
               ELSE [phase EXCEPT ![s] = "available"]
          /\ inFlightQueue' = [inFlightQueue EXCEPT ![s] = 0]
          /\ retireAt' = [retireAt EXCEPT ![s] = 0]
          /\ lastOutcome' = "none"
          /\ UNCHANGED <<logicalAlive, submitted, completed>>

DestroyPool(p) ==
    /\ logicalAlive[p]
    /\ LET releasePool == ~HasOutstanding(p) IN
       /\ logicalAlive' = [logicalAlive EXCEPT ![p] = FALSE]
       /\ physicalAlive' = [physicalAlive EXCEPT ![p] = ~releasePool]
       /\ poolQueue' = IF releasePool THEN [poolQueue EXCEPT ![p] = 0] ELSE poolQueue
       /\ phase' =
            IF releasePool
            THEN [s \in Slots |-> IF s \in SlotsFor(p) THEN "unavailable" ELSE phase[s]]
            ELSE phase
       /\ lastOutcome' = "none"
       /\ UNCHANGED <<inFlightQueue, retireAt, submitted, completed>>

Next ==
    \/ \E p \in Pools, q \in Queues : CreatePool(p, q)
    \/ \E p \in Pools, s \in Slots : AcquireSlot(p, s)
    \/ \E p \in Pools : RejectExhaustedAcquire(p) \/ RejectStaleAcquire(p) \/ DestroyPool(p)
    \/ \E s \in Slots : Abandon(s) \/ SubmitOne(s) \/ CollectSlot(s)
    \/ \E s1, s2 \in Slots : AttemptSubmitBatch(s1, s2)
    \/ \E q \in Queues : CompleteQueue(q)

Spec == Init /\ [][Next]_vars

CapacityIsPhysicalAndExact ==
    \A p \in Pools :
        IF physicalAlive[p]
        THEN \A s \in SlotsFor(p) : phase[s] # "unavailable"
        ELSE \A s \in SlotsFor(p) : phase[s] = "unavailable"

PhysicalLifetimeIsExact ==
    \A p \in Pools :
        physicalAlive[p] <=> (logicalAlive[p] \/ HasOutstanding(p))

\* Every in-flight command buffer executes on its owning pool's exact queue
\* and carries a valid submitted fence point on that queue.
InFlightSlotMatchesPoolQueue ==
    \A s \in Slots :
        phase[s] = "inFlight" =>
            /\ inFlightQueue[s] = poolQueue[PoolOf(s)]
            /\ retireAt[s] > 0
            /\ retireAt[s] <= submitted[inFlightQueue[s]]

Safety ==
    /\ CapacityIsPhysicalAndExact
    /\ PhysicalLifetimeIsExact
    /\ InFlightSlotMatchesPoolQueue

StepSafety ==
    [][/\ (lastOutcome' \in {"exhausted", "stale", "invalidBatch"} => coreVars' = coreVars)
       /\ (\A s \in Slots :
             (phase[s] = "available" /\ phase'[s] = "recording") => logicalAlive[PoolOf(s)])
       /\ (\A q \in Queues : submitted'[q] >= submitted[q] /\ completed'[q] >= completed[q])]_vars
=============================================================================
