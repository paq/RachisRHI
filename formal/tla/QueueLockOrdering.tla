------------------------- MODULE QueueLockOrdering --------------------------
(***************************************************************************)
(* Per-queue submission gate ownership and all-or-nothing multi-queue      *)
(* gate acquisition for device-wide idle synchronization.                  *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Queue.cpp (QueueManager::submit, waitIdle(queue),    *)
(*     waitIdle() std::scoped_lock across graphics/compute/copy gates)     *)
(*   - src/Vulkan/VulkanQueues.cpp (waitIdle() std::scoped_lock across     *)
(*     configured queue gates)                                             *)
(*   - docs/using.md, Threads (submission serializes only the targeted     *)
(*     queue; waitIdle() blocks until every queue is idle)                 *)
(*                                                                         *)
(* Design constants:                                                       *)
(*   - AllOrNothingIdle: TRUE in the design (QueueLockOrdering.cfg),       *)
(*     modeling std::scoped_lock / std::lock try-and-backoff so a probing  *)
(*     idle caller immediately releases held gates on contention before    *)
(*     waiting. Set FALSE in QueueLockOrdering-partialhold.cfg (showing    *)
(*     that even ascending sequential blocking locks wait while holding a  *)
(*     subset) and QueueLockOrdering-deadlock.cfg (showing opposite-order  *)
(*     blocking locks reach an ABBA deadlock).                             *)
(*   - OrderedProbes: TRUE when idle callers probe 0 -> 1 -> 2, FALSE when *)
(*     concurrent idle callers probe opposite gate orders.                 *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Three queue submission gates (0..2), three per-queue submitters, and  *)
(*   two concurrent device-idle callers model single-gate submission and   *)
(*   multi-gate idle acquisition/backoff/release. OS scheduler fairness is *)
(*   abstracted.                                                           *)
(***************************************************************************)

EXTENDS Integers

CONSTANTS AllOrNothingIdle, OrderedProbes

ASSUME AllOrNothingIdle \in BOOLEAN /\ OrderedProbes \in BOOLEAN

QueueCount == 3
Queues == 0..(QueueCount - 1)
Submitters == {0, 1, 2}
IdleActorA == 3
IdleActorB == 4
IdleActors == {IdleActorA, IdleActorB}
Actors == Submitters \cup IdleActors
Phases == {"idle", "waiting", "probing", "holding"}
NoOwner == -1
OwnerDomain == {NoOwner} \cup Actors

VARIABLES
  phase,    \* Per-actor lifecycle phase
  progress, \* Number of queue gates currently held in an idle attempt (0..3)
  owner     \* Current owning actor per queue gate, or NoOwner

vars == <<phase, progress, owner>>

\* Step idx \in 0..(QueueCount - 1) maps to the queue gate probed at that step.
\* When OrderedProbes = TRUE, both idle actors probe 0 -> 1 -> 2.
\* When OrderedProbes = FALSE, IdleActorB probes 2 -> 1 -> 0.
GateAtStep(actor, step) ==
  IF OrderedProbes \/ actor = IdleActorA
  THEN step
  ELSE (QueueCount - 1) - step

FirstGate(actor) ==
  GateAtStep(actor, 0)

NextGate(actor) ==
  GateAtStep(actor, progress[actor])

Init ==
  /\ phase = [a \in Actors |-> "idle"]
  /\ progress = [a \in IdleActors |-> 0]
  /\ owner = [q \in Queues |-> NoOwner]

StartSubmit(submitter) ==
  /\ phase[submitter] = "idle"
  /\ phase' = [phase EXCEPT ![submitter] = "waiting"]
  /\ UNCHANGED <<progress, owner>>

AcquireSubmit(submitter) ==
  /\ phase[submitter] = "waiting"
  /\ owner[submitter] = NoOwner
  /\ phase' = [phase EXCEPT ![submitter] = "holding"]
  /\ owner' = [owner EXCEPT ![submitter] = submitter]
  /\ UNCHANGED progress

FinishSubmit(submitter) ==
  /\ phase[submitter] = "holding"
  /\ owner[submitter] = submitter
  /\ phase' = [phase EXCEPT ![submitter] = "idle"]
  /\ owner' = [owner EXCEPT ![submitter] = NoOwner]
  /\ UNCHANGED progress

StartDeviceIdle(actor) ==
  /\ phase[actor] = "idle"
  /\ progress[actor] = 0
  /\ phase' = [phase EXCEPT ![actor] = "waiting"]
  /\ UNCHANGED <<progress, owner>>

BeginIdleAttempt(actor) ==
  LET gate == FirstGate(actor) IN
  /\ phase[actor] = "waiting"
  /\ progress[actor] = 0
  /\ owner[gate] = NoOwner
  /\ phase' = [phase EXCEPT ![actor] = "probing"]
  /\ progress' = [progress EXCEPT ![actor] = 1]
  /\ owner' = [owner EXCEPT ![gate] = actor]

TryAcquireNextIdleGate(actor) ==
  LET gate == NextGate(actor)
      nextProgress == progress[actor] + 1
  IN
  /\ phase[actor] = "probing"
  /\ progress[actor] > 0
  /\ progress[actor] < QueueCount
  /\ owner[gate] = NoOwner
  /\ phase' =
       [phase EXCEPT ![actor] =
          IF nextProgress = QueueCount THEN "holding" ELSE "probing"]
  /\ progress' = [progress EXCEPT ![actor] = nextProgress]
  /\ owner' = [owner EXCEPT ![gate] = actor]

\* std::scoped_lock / std::lock try-and-backoff: if try_lock on the next gate
\* encounters contention, release every gate held so far before waiting again.
BackoffIdleGates(actor) ==
  /\ AllOrNothingIdle
  /\ phase[actor] = "probing"
  /\ progress[actor] > 0
  /\ progress[actor] < QueueCount
  /\ owner[NextGate(actor)] # NoOwner
  /\ phase' = [phase EXCEPT ![actor] = "waiting"]
  /\ progress' = [progress EXCEPT ![actor] = 0]
  /\ owner' =
       [q \in Queues |-> IF owner[q] = actor THEN NoOwner ELSE owner[q]]

FinishDeviceIdle(actor) ==
  /\ phase[actor] = "holding"
  /\ progress[actor] = QueueCount
  /\ \A q \in Queues : owner[q] = actor
  /\ phase' = [phase EXCEPT ![actor] = "idle"]
  /\ progress' = [progress EXCEPT ![actor] = 0]
  /\ owner' = [q \in Queues |-> NoOwner]

Next ==
  \/ \E s \in Submitters :
       \/ StartSubmit(s)
       \/ AcquireSubmit(s)
       \/ FinishSubmit(s)
  \/ \E a \in IdleActors :
       \/ StartDeviceIdle(a)
       \/ BeginIdleAttempt(a)
       \/ TryAcquireNextIdleGate(a)
       \/ BackoffIdleGates(a)
       \/ FinishDeviceIdle(a)

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ phase \in [Actors -> Phases]
  /\ progress \in [IdleActors -> 0..QueueCount]
  /\ owner \in [Queues -> OwnerDomain]

SubmitterPhasesValid ==
  \A s \in Submitters :
    phase[s] \in {"idle", "waiting", "holding"}

IdleActorPhasesValid ==
  \A a \in IdleActors :
    /\ phase[a] \in {"idle", "waiting", "probing", "holding"}
    /\ (phase[a] \in {"idle", "waiting"} <=> progress[a] = 0)
    /\ (phase[a] = "probing" <=>
          (progress[a] > 0 /\ progress[a] < QueueCount))
    /\ (phase[a] = "holding" <=> progress[a] = QueueCount)

SubmitterOwnershipMatchesPhase ==
  \A s \in Submitters :
    IF phase[s] = "holding"
    THEN /\ owner[s] = s
         /\ \A q \in Queues : q # s => owner[q] # s
    ELSE \A q \in Queues : owner[q] # s

IdleActorOwnershipMatchesProgress ==
  \A a \in IdleActors :
    \A step \in 0..(QueueCount - 1) :
      (owner[GateAtStep(a, step)] = a) <=> (step < progress[a])

\* Caller-visible device-idle execution ("holding") owns every queue gate.
HoldingIdleActorOwnsAllGates ==
  \A a \in IdleActors :
    phase[a] = "holding" => \A q \in Queues : owner[q] = a

\* Any waiting actor (submitter or backed-off idle caller) holds zero gates.
WaitingActorsOwnNoGates ==
  \A a \in Actors :
    phase[a] = "waiting" => \A q \in Queues : owner[q] # a

GateOwnershipConsistent ==
  /\ SubmitterPhasesValid
  /\ IdleActorPhasesValid
  /\ SubmitterOwnershipMatchesPhase
  /\ IdleActorOwnershipMatchesProgress
  /\ HoldingIdleActorOwnsAllGates
  /\ WaitingActorsOwnNoGates

\* Probing idle actor can step immediately without waiting on another thread:
\* either try_lock acquires NextGate, or backoff releases the partial subset.
IdleActorCanStepWithoutWaiting(actor) ==
  \/ owner[NextGate(actor)] = NoOwner
  \/ (AllOrNothingIdle /\ owner[NextGate(actor)] # NoOwner)

\* waitIdle() takes every gate with one std::scoped_lock; the try-and-back-off
\* std::lock of MSVC STL, libstdc++, and libc++ never waits while holding only
\* a subset.
IdleActorNeverWaitsHoldingSubset ==
  \A a \in IdleActors :
    (\E q1, q2 \in Queues : owner[q1] = a /\ owner[q2] # a) =>
      /\ phase[a] = "probing"
      /\ IdleActorCanStepWithoutWaiting(a)

\* Whenever not all actors are idle, at least one active (non-idle) actor can
\* immediately advance without starting an idle actor.
SomeOperationCanAdvance ==
  \/ (\A a \in Actors : phase[a] = "idle")
  \/ (\E s \in Submitters :
        \/ phase[s] = "holding"
        \/ (phase[s] = "waiting" /\ owner[s] = NoOwner))
  \/ (\E a \in IdleActors :
        \/ phase[a] = "holding"
        \/ (phase[a] = "waiting" /\ owner[FirstGate(a)] = NoOwner)
        \/ (phase[a] = "probing" /\ IdleActorCanStepWithoutWaiting(a)))

AllOrNothingPreventsStall ==
  AllOrNothingIdle => SomeOperationCanAdvance

Safety ==
  /\ GateOwnershipConsistent
  /\ IdleActorNeverWaitsHoldingSubset
  /\ AllOrNothingPreventsStall

=============================================================================
