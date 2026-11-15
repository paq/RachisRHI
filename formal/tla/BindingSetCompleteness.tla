----------------------- MODULE BindingSetCompleteness -----------------------
(***************************************************************************)
(* Complete single-assignment descriptor construction for non-partially-   *)
(* bound binding sets, duplicate/missing write rejection, and resource     *)
(* lifetime retention through construction and publication.                *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Binding.cpp (createBindingSet, applyBindingWrites)   *)
(*                                                                         *)
(* Contract modeled:                                                       *)
(*   - A fixed/non-sparse binding set over Slots = {0, 1} constructs a     *)
(*     private BindingSetRecord before publishing a live handle.           *)
(*   - Writing a stale/destroyed resource handle rejects and aborts        *)
(*     construction without publishing a live set.                         *)
(*   - Writing the same descriptor slot more than once (when design rule   *)
(*     RejectDuplicateWrites = TRUE, matching applyBindingWrites checking  *)
(*     if (written[descriptorOffset]) return invalidArgument) or           *)
(*     attempting to publish with an unwritten slot rejects and aborts     *)
(*     construction, releasing any tentative references to logically       *)
(*     destroyed resources.                                                *)
(*   - Resources written into a candidate or published set remain          *)
(*     physically alive across logical resource destruction until the      *)
(*     candidate aborts or the published set is destroyed.                 *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - Hardware descriptor encoding inside CpuDescriptorPool / GPU heaps.  *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT RejectDuplicateWrites

ASSUME RejectDuplicateWrites \in BOOLEAN

Slots == {0, 1}
Resources == {0, 1}
Phases == {"empty", "building", "published"}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    phase,                  \* "empty", "building", or "published"
    writeCount,             \* [Slots -> 0..2] writes applied to each slot
    owner,                  \* [Slots -> Resources] resource bound to slot
    logicalResourceAlive,   \* [Resources -> BOOLEAN] public resource handle alive
    physicalResourceAlive,  \* [Resources -> BOOLEAN] physical resource alive
    lastOutcome             \* outcome of the last fallible construction step

vars == <<phase, writeCount, owner, logicalResourceAlive,
          physicalResourceAlive, lastOutcome>>

RetainedBySet(r, p, wc, own) ==
    /\ p # "empty"
    /\ \E s \in Slots : wc[s] > 0 /\ own[s] = r

AbortCandidate ==
    /\ phase' = "empty"
    /\ writeCount' = [s \in Slots |-> 0]
    /\ physicalResourceAlive' =
         [r \in Resources |-> physicalResourceAlive[r] /\ logicalResourceAlive[r]]

TypeOK ==
    /\ phase \in Phases
    /\ writeCount \in [Slots -> 0..2]
    /\ owner \in [Slots -> Resources]
    /\ logicalResourceAlive \in [Resources -> BOOLEAN]
    /\ physicalResourceAlive \in [Resources -> BOOLEAN]
    /\ lastOutcome \in Outcomes

Init ==
    /\ phase = "empty"
    /\ writeCount = [s \in Slots |-> 0]
    /\ owner = [s \in Slots |-> 0]
    /\ logicalResourceAlive = [r \in Resources |-> FALSE]
    /\ physicalResourceAlive = [r \in Resources |-> FALSE]
    /\ lastOutcome = "none"

CreateResource(r) ==
    /\ ~logicalResourceAlive[r]
    /\ ~physicalResourceAlive[r]
    /\ logicalResourceAlive' = [logicalResourceAlive EXCEPT ![r] = TRUE]
    /\ physicalResourceAlive' = [physicalResourceAlive EXCEPT ![r] = TRUE]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<phase, writeCount, owner>>

DestroyResource(r) ==
    /\ logicalResourceAlive[r]
    /\ logicalResourceAlive' = [logicalResourceAlive EXCEPT ![r] = FALSE]
    /\ physicalResourceAlive' =
         [physicalResourceAlive EXCEPT ![r] = RetainedBySet(r, phase, writeCount, owner)]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<phase, writeCount, owner>>

StartSet ==
    /\ phase = "empty"
    /\ phase' = "building"
    /\ writeCount' = [s \in Slots |-> 0]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<owner, logicalResourceAlive, physicalResourceAlive>>

AttemptWrite(s, r) ==
    /\ phase = "building"
    /\ writeCount[s] < 2
    /\ IF ~logicalResourceAlive[r] \/ (RejectDuplicateWrites /\ writeCount[s] > 0)
       THEN /\ AbortCandidate
            /\ lastOutcome' = "rejected"
            /\ UNCHANGED <<owner, logicalResourceAlive>>
       ELSE LET prevOwner == owner[s]
                hadPrev == writeCount[s] > 0
                nextWc == [writeCount EXCEPT ![s] = writeCount[s] + 1]
                nextOwn == [owner EXCEPT ![s] = r]
            IN /\ writeCount' = nextWc
               /\ owner' = nextOwn
               /\ physicalResourceAlive' =
                    [res \in Resources |->
                       logicalResourceAlive[res]
                       \/ RetainedBySet(res, "building", nextWc, nextOwn)
                       \/ (hadPrev /\ prevOwner = res /\ RetainedBySet(res, "building", nextWc, nextOwn))]
               /\ lastOutcome' = "accepted"
               /\ UNCHANGED <<phase, logicalResourceAlive>>

AttemptPublish ==
    /\ phase = "building"
    /\ IF \A s \in Slots : writeCount[s] > 0
       THEN /\ phase' = "published"
            /\ lastOutcome' = "accepted"
            /\ UNCHANGED <<writeCount, owner, logicalResourceAlive, physicalResourceAlive>>
       ELSE /\ AbortCandidate
            /\ lastOutcome' = "rejected"
            /\ UNCHANGED <<owner, logicalResourceAlive>>

DestroySet ==
    /\ phase = "published"
    /\ AbortCandidate
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<owner, logicalResourceAlive>>

Next ==
    \/ \E r \in Resources : CreateResource(r) \/ DestroyResource(r)
    \/ StartSet
    \/ \E s \in Slots, r \in Resources : AttemptWrite(s, r)
    \/ AttemptPublish
    \/ DestroySet

Spec == Init /\ [][Next]_vars

\* Every published binding set has every slot written exactly once (no missing
\* descriptors and no duplicate overwrites).
PublishedSlotsWrittenExactlyOnce ==
    phase = "published" => \A s \in Slots : writeCount[s] = 1

AbsentSetHoldsNoWrites ==
    phase = "empty" => \A s \in Slots : writeCount[s] = 0

WrittenSlotsRetainResources ==
    \A s \in Slots :
        (phase # "empty" /\ writeCount[s] > 0) =>
            physicalResourceAlive[owner[s]]

ExactPhysicalResourceOwnership ==
    \A r \in Resources :
        physicalResourceAlive[r] <=>
            (logicalResourceAlive[r] \/ RetainedBySet(r, phase, writeCount, owner))

RejectedConstructionLeavesNoLiveSet ==
    lastOutcome = "rejected" => phase = "empty"

Safety ==
    /\ PublishedSlotsWrittenExactlyOnce
    /\ AbsentSetHoldsNoWrites
    /\ WrittenSlotsRetainResources
    /\ ExactPhysicalResourceOwnership
    /\ RejectedConstructionLeavesNoLiveSet

StepSafety ==
    [][(phase # "published" /\ phase' = "published") =>
          \A s \in Slots : writeCount'[s] = 1]_vars
=============================================================================
