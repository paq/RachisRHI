--------------------- MODULE TimelineHandlePublication ----------------------
(***************************************************************************)
(* Formal safety model for C ABI timeline-handle pre-reservation,          *)
(* rollback on native queue rejection, infallible post-submit publication, *)
(* and bounded slot reuse across submissions:                              *)
(*   - src/C/rachis.cpp (rachis_command_submit, rachis_timeline_release)   *)
(*   - src/C/CAbiSlotTable.h (CAbiSlotTable emplace, find, erase)          *)
(*   - include/Rachis/rachis.h                                             *)
(*                                                                         *)
(* Models a bounded 2-slot C timeline handle table across multiple queue   *)
(* submissions: rachis_command_submit reserves a handle slot before        *)
(* invoking native.submit, rolls back the slot and preserves the recorded  *)
(* command on queue failure, and infallibly publishes the handle on queue  *)
(* acceptance so no accepted GPU submission ever loses its waitable token. *)
(* If native.submit consumed the command before failing, the C command     *)
(* handle is erased as well; the model covers the unconsumed case.         *)
(*                                                                         *)
(* Trusted boundary: CAbiSlotTable::emplace reserving vector capacity      *)
(* before it constructs the slot, and native queue submission internals    *)
(* once native.submit accepts the command.                                 *)
(***************************************************************************)
EXTENDS Naturals

Slots == {1, 2}
SlotOrNone == {0} \cup Slots
Phases == {"idle", "reserved", "accepted"}
Outcomes == {"none", "published", "resourceExhausted", "queueRejected"}
MaxSubmissions == 3

VARIABLES
  commandRecorded,   \* Whether a valid command handle exists in device->commands.
  phase,             \* Stage of an active rachis_command_submit call.
  reservedSlot,      \* Pre-reserved C handle slot (0 when none is held).
  publishedSlots,    \* Live timeline handle slots in device->timelines.
  slotPoint,         \* Native timeline point stored in each published slot.
  submittedFrontier, \* Latest native timeline point accepted by the queue.
  unreleasedPoints,  \* Accepted native timeline points whose C handle is not yet released.
  lastOutcome        \* Outcome of the latest submit transaction.

coreVars == <<
  commandRecorded,
  publishedSlots,
  slotPoint,
  submittedFrontier,
  unreleasedPoints
>>

vars == <<
  commandRecorded,
  phase,
  reservedSlot,
  publishedSlots,
  slotPoint,
  submittedFrontier,
  unreleasedPoints,
  lastOutcome
>>

Init ==
  /\ commandRecorded = FALSE
  /\ phase = "idle"
  /\ reservedSlot = 0
  /\ publishedSlots = {}
  /\ slotPoint = [s \in Slots |-> 0]
  /\ submittedFrontier = 0
  /\ unreleasedPoints = {}
  /\ lastOutcome = "none"

recordCommand ==
  /\ phase = "idle"
  /\ ~commandRecorded
  /\ submittedFrontier < MaxSubmissions
  /\ commandRecorded' = TRUE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       phase,
       reservedSlot,
       publishedSlots,
       slotPoint,
       submittedFrontier,
       unreleasedPoints
     >>

reserveStorage(slot) ==
  /\ phase = "idle"
  /\ commandRecorded
  /\ slot \in Slots \ publishedSlots
  /\ phase' = "reserved"
  /\ reservedSlot' = slot
  /\ lastOutcome' = "none"
  /\ UNCHANGED coreVars

\* Covers both bounded handle-table exhaustion (publishedSlots = Slots) and
\* fallible CAbiSlotTable::emplace allocation failure before native.submit.
rejectStorageReservation ==
  /\ phase = "idle"
  /\ commandRecorded
  /\ lastOutcome # "resourceExhausted"
  /\ lastOutcome' = "resourceExhausted"
  /\ UNCHANGED <<
       commandRecorded,
       phase,
       reservedSlot,
       publishedSlots,
       slotPoint,
       submittedFrontier,
       unreleasedPoints
     >>

acceptQueueWork ==
  /\ phase = "reserved"
  /\ submittedFrontier < MaxSubmissions
  /\ LET nextPoint == submittedFrontier + 1 IN
     /\ phase' = "accepted"
     /\ submittedFrontier' = nextPoint
     /\ unreleasedPoints' = unreleasedPoints \cup {nextPoint}
     /\ lastOutcome' = "none"
     /\ UNCHANGED <<commandRecorded, reservedSlot, publishedSlots, slotPoint>>

\* Native submit failed before consuming the command: erase the pre-reserved
\* timeline slot and keep commandRecorded intact.
rejectQueueWork ==
  /\ phase = "reserved"
  /\ phase' = "idle"
  /\ reservedSlot' = 0
  /\ lastOutcome' = "queueRejected"
  /\ UNCHANGED coreVars

\* Infallible TimelinePoint assignment into the pre-reserved slot; the
\* consumed command handle is erased.
publishTimeline ==
  /\ phase = "accepted"
  /\ commandRecorded' = FALSE
  /\ phase' = "idle"
  /\ reservedSlot' = 0
  /\ publishedSlots' = publishedSlots \cup {reservedSlot}
  /\ slotPoint' = [slotPoint EXCEPT ![reservedSlot] = submittedFrontier]
  /\ lastOutcome' = "published"
  /\ UNCHANGED <<submittedFrontier, unreleasedPoints>>

releaseTimeline(slot) ==
  /\ phase = "idle"
  /\ slot \in publishedSlots
  /\ publishedSlots' = publishedSlots \ {slot}
  /\ slotPoint' = [slotPoint EXCEPT ![slot] = 0]
  /\ unreleasedPoints' = unreleasedPoints \ {slotPoint[slot]}
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<commandRecorded, phase, reservedSlot, submittedFrontier>>

Next ==
  \/ recordCommand
  \/ \E s \in Slots : reserveStorage(s)
  \/ rejectStorageReservation
  \/ acceptQueueWork
  \/ rejectQueueWork
  \/ publishTimeline
  \/ \E s \in Slots : releaseTimeline(s)

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ commandRecorded \in BOOLEAN
  /\ phase \in Phases
  /\ reservedSlot \in SlotOrNone
  /\ publishedSlots \subseteq Slots
  /\ slotPoint \in [Slots -> 0..MaxSubmissions]
  /\ submittedFrontier \in 0..MaxSubmissions
  /\ unreleasedPoints \subseteq 1..MaxSubmissions
  /\ lastOutcome \in Outcomes

ReservationIsPrivateAndValid ==
  /\ (phase = "idle" <=> reservedSlot = 0)
  /\ (reservedSlot # 0 =>
        /\ reservedSlot \in Slots \ publishedSlots
        /\ slotPoint[reservedSlot] = 0
        /\ commandRecorded)

AcceptedWorkAlwaysHasWaitableFuture ==
  \A pt \in unreleasedPoints :
    \/ (phase = "accepted" /\ pt = submittedFrontier /\ reservedSlot # 0)
    \/ \E s \in publishedSlots : slotPoint[s] = pt

PublishedSlotsMatchLivePoints ==
  /\ \A s \in Slots :
       (s \in publishedSlots <=> slotPoint[s] \in unreleasedPoints)
  /\ \A s1, s2 \in publishedSlots :
       slotPoint[s1] = slotPoint[s2] => s1 = s2

RejectedTransactionsLeaveNoDanglingReservation ==
  lastOutcome \in {"resourceExhausted", "queueRejected"} =>
    /\ phase = "idle"
    /\ reservedSlot = 0
    /\ commandRecorded

Safety ==
  /\ ReservationIsPrivateAndValid
  /\ AcceptedWorkAlwaysHasWaitableFuture
  /\ PublishedSlotsMatchLivePoints
  /\ RejectedTransactionsLeaveNoDanglingReservation

StepSafety ==
  [][/\ (lastOutcome' = "resourceExhausted" => coreVars' = coreVars)
     /\ (lastOutcome' = "queueRejected" /\ phase = "reserved" =>
           coreVars' = coreVars)]_vars

=============================================================================
