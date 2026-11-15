-------------------------- MODULE HandleLifecycle ---------------------------
(***************************************************************************)
(* Generational slot allocation, owner-token isolation, stale-handle       *)
(* rejection, and non-wrapping slot retirement in                          *)
(* `src/Common/HandleTable.h` (`include/Rachis/Handle.h`).                 *)
(*                                                                         *)
(* Models two independent handle tables (`"left"` and `"right"`) with one  *)
(* slot each and bounded generation counter `1..MaxGeneration`. Releasing  *)
(* a live slot advances its generation while below `MaxGeneration` and     *)
(* permanently retires the slot at `MaxGeneration` instead of wrapping;    *)
(* allocation attempts on retired capacity reject atomically; and neither  *)
(* foreign-table nor stale handles ever resolve.                           *)
(*                                                                         *)
(* Abstracted: multi-slot free-list indexing up to `(1 << 24) - 1` and     *)
(* payload storage inside `HandleTable`.                                   *)
(***************************************************************************)
EXTENDS Naturals

Tables == {"left", "right"}
Generations == 1..3
MaxGeneration == 3
Outcomes == {"none", "accepted", "rejected"}

HandleType == [owner : Tables, index : {0}, generation : Generations]

VARIABLES
    live,           \* per-table slot liveness in [Tables -> BOOLEAN]
    generation,     \* per-table current generation in [Tables -> Generations]
    retired,        \* per-table permanent retirement flag in [Tables -> BOOLEAN]
    issued,         \* set of all handles ever allocated across both tables
    stale,          \* set of handles previously destroyed across both tables
    lastOutcome     \* outcome of the latest step in Outcomes

coreVars == <<live, generation, retired, issued, stale>>
vars == <<live, generation, retired, issued, stale, lastOutcome>>

CurrentHandle(table) ==
    [owner |-> table, index |-> 0, generation |-> generation[table]]

Resolves(table, handle) ==
    /\ table \in Tables
    /\ handle.owner = table
    /\ handle.index = 0
    /\ live[table]
    /\ generation[table] = handle.generation

Init ==
    /\ live = [t \in Tables |-> FALSE]
    /\ generation = [t \in Tables |-> 1]
    /\ retired = [t \in Tables |-> FALSE]
    /\ issued = {}
    /\ stale = {}
    /\ lastOutcome = "none"

attemptAllocate(table) ==
    LET accepted == ~retired[table]
    IN /\ ~live[table]
       /\ IF accepted
          THEN /\ live' = [live EXCEPT ![table] = TRUE]
               /\ issued' = issued \cup {CurrentHandle(table)}
               /\ UNCHANGED <<generation, retired, stale>>
          ELSE UNCHANGED coreVars
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

destroySlot(table) ==
    /\ live[table]
    /\ live' = [live EXCEPT ![table] = FALSE]
    /\ generation' =
        IF generation[table] < MaxGeneration
        THEN [generation EXCEPT ![table] = @ + 1]
        ELSE generation
    /\ retired' =
        IF generation[table] = MaxGeneration
        THEN [retired EXCEPT ![table] = TRUE]
        ELSE retired
    /\ stale' = stale \cup {CurrentHandle(table)}
    /\ lastOutcome' = "none"
    /\ UNCHANGED issued

Next ==
    \E t \in Tables :
        \/ attemptAllocate(t)
        \/ destroySlot(t)

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ live \in [Tables -> BOOLEAN]
    /\ generation \in [Tables -> Generations]
    /\ retired \in [Tables -> BOOLEAN]
    /\ issued \in SUBSET HandleType
    /\ stale \in SUBSET HandleType
    /\ lastOutcome \in Outcomes

LiveHandlesWereIssued ==
    \A t \in Tables :
        live[t] => CurrentHandle(t) \in issued /\ CurrentHandle(t) \notin stale

ForeignHandlesNeverResolve ==
    \A h \in issued, t \in Tables :
        h.owner # t => ~Resolves(t, h)

StaleHandlesNeverResolve ==
    /\ stale \subseteq issued
    /\ \A h \in stale, t \in Tables : ~Resolves(t, h)

RetiredSlotsNeverResolve ==
    \A t \in Tables :
        retired[t] =>
            /\ ~live[t]
            /\ generation[t] = MaxGeneration
            /\ \A h \in issued : ~Resolves(t, h)

Safety ==
    /\ LiveHandlesWereIssued
    /\ ForeignHandlesNeverResolve
    /\ StaleHandlesNeverResolve
    /\ RetiredSlotsNeverResolve

StepSafety ==
    [][
        /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
        /\ \A t \in Tables :
            /\ generation[t] <= generation'[t]
            /\ (retired[t] => retired'[t])
        /\ issued \subseteq issued'
        /\ stale \subseteq stale'
    ]_vars

=============================================================================
