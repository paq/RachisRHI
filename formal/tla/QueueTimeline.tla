--------------------------- MODULE QueueTimeline ----------------------------
(***************************************************************************)
(* Bounded command-pool slots, non-wrapping queue fence timelines,         *)
(* cross-queue GPU fence wait ordering, ordered batch submission sharing   *)
(* one retirement point and wait dependency, and queue/device idle drains. *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Queue.cpp (submit, hasUnreservedTimelineSlot,        *)
(*     waitIdle(QueueType), waitIdle(), collectQueue)                      *)
(*   - src/Vulkan/VulkanQueues.cpp (timeline semaphore ordering: submit,   *)
(*     hasUnreservedTimelineSlot, validateBatchWaits,                      *)
(*     waitIdle(QueueType), waitIdle(), collectQueue)                      *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Two command-buffer slots, three queues (graphics/compute/copy), and   *)
(*   a bounded timeline [0..MaxTimeline] cover single and batched          *)
(*   submissions, cross-queue fence wait enforcement before completion,    *)
(*   timeline exhaustion / invalid-wait / mixed-queue batch rejection,     *)
(*   and exact fence-ordered slot recycling.                               *)
(***************************************************************************)

EXTENDS Naturals

Slots == {0, 1}
Queues == {0, 1, 2}
MaxTimeline == 2
Phases == {"available", "recording", "inFlight"}
Operations == {
  "none", "acquire", "abandon", "submit", "submitBatch",
  "complete", "collect", "waitQueueIdle", "waitDeviceIdle"
}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  phase,       \* Lifecycle state per command-pool slot
  ownerQueue,  \* Owning queue per command-pool slot
  retireAt,    \* Fence value at which an in-flight/retired slot completes
  waitQueue,   \* Cross-queue wait target queue per slot
  waitValue,   \* Cross-queue wait target value per slot (0 = no wait)
  submitted,   \* Monotonic submitted frontier per queue
  completed,   \* Monotonic completed frontier per queue
  lastOp,      \* Most recent operation kind
  lastOutcome, \* Outcome of most recent operation
  lastQueue    \* Target queue of most recent queue operation

coreVars == <<
  phase, ownerQueue, retireAt,
  waitQueue, waitValue, submitted, completed
>>

vars == <<
  phase, ownerQueue, retireAt,
  waitQueue, waitValue, submitted, completed,
  lastOp, lastOutcome, lastQueue
>>

WaitArgumentValid(hasWait, depQueue, depValue) ==
  IF ~hasWait
  THEN depValue = 0
  ELSE /\ depQueue \in Queues
       /\ depValue > 0
       /\ depValue <= submitted[depQueue]

CanSubmitSingle(slot, hasWait, depQueue, depValue) ==
  /\ phase[slot] = "recording"
  /\ submitted[ownerQueue[slot]] < MaxTimeline
  /\ WaitArgumentValid(hasWait, depQueue, depValue)

CanSubmitBatch(first, second, hasWait, depQueue, depValue) ==
  /\ first # second
  /\ phase[first] = "recording"
  /\ phase[second] = "recording"
  /\ ownerQueue[first] = ownerQueue[second]
  /\ submitted[ownerQueue[first]] < MaxTimeline
  /\ WaitArgumentValid(hasWait, depQueue, depValue)

\* Next fence increment on `queue` may complete only once every cross-queue
\* fence wait recorded for submissions up to `targetValue` has completed.
WaitsSatisfiedThrough(queue, targetValue) ==
  \A s \in Slots :
    /\ phase[s] = "inFlight"
    /\ ownerQueue[s] = queue
    /\ retireAt[s] <= targetValue
    /\ waitValue[s] > 0
    => completed[waitQueue[s]] >= waitValue[s]

Init ==
  /\ phase = [s \in Slots |-> "available"]
  /\ ownerQueue = [s \in Slots |-> 0]
  /\ retireAt = [s \in Slots |-> 0]
  /\ waitQueue = [s \in Slots |-> 0]
  /\ waitValue = [s \in Slots |-> 0]
  /\ submitted = [q \in Queues |-> 0]
  /\ completed = [q \in Queues |-> 0]
  /\ lastOp = "none"
  /\ lastOutcome = "none"
  /\ lastQueue = 0

Acquire(slot, queue) ==
  /\ phase[slot] = "available"
  /\ phase' = [phase EXCEPT ![slot] = "recording"]
  /\ ownerQueue' = [ownerQueue EXCEPT ![slot] = queue]
  /\ retireAt' = [retireAt EXCEPT ![slot] = 0]
  /\ waitQueue' = [waitQueue EXCEPT ![slot] = 0]
  /\ waitValue' = [waitValue EXCEPT ![slot] = 0]
  /\ UNCHANGED <<submitted, completed>>
  /\ lastOp' = "acquire"
  /\ lastOutcome' = "accepted"
  /\ lastQueue' = queue

Abandon(slot) ==
  /\ phase[slot] = "recording"
  /\ phase' = [phase EXCEPT ![slot] = "available"]
  /\ waitQueue' = [waitQueue EXCEPT ![slot] = 0]
  /\ waitValue' = [waitValue EXCEPT ![slot] = 0]
  /\ UNCHANGED <<ownerQueue, retireAt, submitted, completed>>
  /\ lastOp' = "abandon"
  /\ lastOutcome' = "accepted"
  /\ lastQueue' = ownerQueue[slot]

AttemptSubmitSingle(slot, hasWait, depQueue, depValue) ==
  LET queue == ownerQueue[slot]
      nextValue == submitted[queue] + 1
      wq == IF hasWait THEN depQueue ELSE 0
      wv == IF hasWait THEN depValue ELSE 0
  IN
  /\ phase[slot] = "recording"
  /\ IF CanSubmitSingle(slot, hasWait, depQueue, depValue)
     THEN /\ phase' = [phase EXCEPT ![slot] = "inFlight"]
          /\ retireAt' = [retireAt EXCEPT ![slot] = nextValue]
          /\ waitQueue' = [waitQueue EXCEPT ![slot] = wq]
          /\ waitValue' = [waitValue EXCEPT ![slot] = wv]
          /\ submitted' = [submitted EXCEPT ![queue] = nextValue]
          /\ UNCHANGED <<ownerQueue, completed>>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "submit"
  /\ lastQueue' = queue

AttemptSubmitBatch(first, second, hasWait, depQueue, depValue) ==
  LET queue == ownerQueue[first]
      nextValue == submitted[queue] + 1
      wq == IF hasWait THEN depQueue ELSE 0
      wv == IF hasWait THEN depValue ELSE 0
  IN
  /\ first # second
  /\ phase[first] = "recording"
  /\ phase[second] = "recording"
  /\ IF CanSubmitBatch(first, second, hasWait, depQueue, depValue)
     THEN /\ phase' = [[phase EXCEPT ![first] = "inFlight"] EXCEPT ![second] = "inFlight"]
          /\ retireAt' = [[retireAt EXCEPT ![first] = nextValue] EXCEPT ![second] = nextValue]
          /\ waitQueue' = [[waitQueue EXCEPT ![first] = wq] EXCEPT ![second] = wq]
          /\ waitValue' = [[waitValue EXCEPT ![first] = wv] EXCEPT ![second] = wv]
          /\ submitted' = [submitted EXCEPT ![queue] = nextValue]
          /\ UNCHANGED <<ownerQueue, completed>>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "submitBatch"
  /\ lastQueue' = queue

Complete(queue) ==
  LET nextDone == completed[queue] + 1 IN
  /\ completed[queue] < submitted[queue]
  /\ WaitsSatisfiedThrough(queue, nextDone)
  /\ completed' = [completed EXCEPT ![queue] = nextDone]
  /\ UNCHANGED <<phase, ownerQueue, retireAt, waitQueue, waitValue, submitted>>
  /\ lastOp' = "complete"
  /\ lastOutcome' = "accepted"
  /\ lastQueue' = queue

Collect(slot) ==
  /\ phase[slot] = "inFlight"
  /\ retireAt[slot] <= completed[ownerQueue[slot]]
  /\ phase' = [phase EXCEPT ![slot] = "available"]
  \* Keep retireAt/waitQueue/waitValue until next Acquire as retirement evidence.
  /\ UNCHANGED <<ownerQueue, retireAt, waitQueue, waitValue, submitted, completed>>
  /\ lastOp' = "collect"
  /\ lastOutcome' = "accepted"
  /\ lastQueue' = ownerQueue[slot]

WaitQueueIdle(queue) ==
  /\ WaitsSatisfiedThrough(queue, submitted[queue])
  /\ completed' = [completed EXCEPT ![queue] = submitted[queue]]
  /\ phase' =
       [s \in Slots |->
          IF phase[s] = "inFlight" /\ ownerQueue[s] = queue
          THEN "available"
          ELSE phase[s]]
  /\ UNCHANGED <<ownerQueue, retireAt, waitQueue, waitValue, submitted>>
  /\ lastOp' = "waitQueueIdle"
  /\ lastOutcome' = "accepted"
  /\ lastQueue' = queue

WaitDeviceIdle ==
  /\ completed' = submitted
  /\ phase' =
       [s \in Slots |-> IF phase[s] = "inFlight" THEN "available" ELSE phase[s]]
  /\ UNCHANGED <<ownerQueue, retireAt, waitQueue, waitValue, submitted>>
  /\ lastOp' = "waitDeviceIdle"
  /\ lastOutcome' = "accepted"
  /\ lastQueue' = 0

Next ==
  \/ \E s \in Slots, q \in Queues : Acquire(s, q)
  \/ \E s \in Slots :
       \/ Abandon(s)
       \/ Collect(s)
  \/ \E s \in Slots, q \in Queues, v \in 0..MaxTimeline :
       \/ AttemptSubmitSingle(s, FALSE, 0, 0)
       \/ AttemptSubmitSingle(s, TRUE, q, v)
  \/ \E s1, s2 \in Slots, q \in Queues, v \in 0..MaxTimeline :
       \/ AttemptSubmitBatch(s1, s2, FALSE, 0, 0)
       \/ AttemptSubmitBatch(s1, s2, TRUE, q, v)
  \/ \E q \in Queues :
       \/ Complete(q)
       \/ WaitQueueIdle(q)
  \/ WaitDeviceIdle

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ phase \in [Slots -> Phases]
  /\ ownerQueue \in [Slots -> Queues]
  /\ retireAt \in [Slots -> 0..MaxTimeline]
  /\ waitQueue \in [Slots -> Queues]
  /\ waitValue \in [Slots -> 0..MaxTimeline]
  /\ submitted \in [Queues -> 0..MaxTimeline]
  /\ completed \in [Queues -> 0..MaxTimeline]
  /\ lastOp \in Operations
  /\ lastOutcome \in Outcomes
  /\ lastQueue \in Queues

CompletionNeverPassesSubmitted ==
  \A q \in Queues : completed[q] <= submitted[q]

InFlightCommandsHaveSubmittedFrontier ==
  \A s \in Slots :
    phase[s] = "inFlight" =>
      /\ retireAt[s] > 0
      /\ retireAt[s] <= submitted[ownerQueue[s]]

BatchSiblingsShareWaitDependency ==
  \A s1, s2 \in Slots :
    /\ phase[s1] = "inFlight"
    /\ phase[s2] = "inFlight"
    /\ ownerQueue[s1] = ownerQueue[s2]
    /\ retireAt[s1] = retireAt[s2]
    => /\ waitQueue[s1] = waitQueue[s2]
       /\ waitValue[s1] = waitValue[s2]

WaitsReferenceSubmittedWork ==
  \A s \in Slots :
    phase[s] = "inFlight" /\ waitValue[s] > 0 =>
      waitValue[s] <= submitted[waitQueue[s]]

CompletedWorkSatisfiesCrossQueueWaits ==
  \A s \in Slots :
    /\ phase[s] \in {"inFlight", "available"}
    /\ retireAt[s] > 0
    /\ completed[ownerQueue[s]] >= retireAt[s]
    /\ waitValue[s] > 0
    => completed[waitQueue[s]] >= waitValue[s]

RecyclingIsFenceExact ==
  \A s \in Slots :
    phase[s] = "available" /\ retireAt[s] > 0 =>
      completed[ownerQueue[s]] >= retireAt[s]

Safety ==
  /\ CompletionNeverPassesSubmitted
  /\ InFlightCommandsHaveSubmittedFrontier
  /\ BatchSiblingsShareWaitDependency
  /\ WaitsReferenceSubmittedWork
  /\ CompletedWorkSatisfiesCrossQueueWaits
  /\ RecyclingIsFenceExact

StepSafety ==
  [][
    /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
    /\ (\A q \in Queues :
          /\ submitted'[q] >= submitted[q]
          /\ completed'[q] >= completed[q])
    /\ (lastOp' = "complete" =>
          WaitsSatisfiedThrough(lastQueue', completed[lastQueue'] + 1))
    /\ (lastOp' = "waitQueueIdle" =>
          /\ WaitsSatisfiedThrough(lastQueue', submitted[lastQueue'])
          /\ completed'[lastQueue'] = submitted[lastQueue']
          /\ \A s \in Slots :
               ownerQueue[s] = lastQueue' => phase'[s] # "inFlight")
    /\ (lastOp' = "waitDeviceIdle" =>
          /\ completed' = submitted
          /\ \A s \in Slots : phase'[s] # "inFlight")
  ]_vars

=============================================================================
