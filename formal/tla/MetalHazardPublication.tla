----------------------- MODULE MetalHazardPublication -----------------------
(***************************************************************************)
(* Command-local hazard/barrier isolation, submission-order publication,  *)
(* and the check of each texture barrier's before-layout in the Metal 4    *)
(* backend (`src/Metal/MetalBarriers.mm`, `src/Metal/MetalCopies.mm`, and  *)
(* `src/Metal/MetalCommand.mm`).                                           *)
(*                                                                         *)
(* Models two independently recorded command buffers operating on two      *)
(* resource cells across all overlapping and disjoint target choices,      *)
(* submission orders, command resets, and abandonments. Encoded barrier    *)
(* effects remain command-local until successful queue submission commits  *)
(* only the touched cells in exact acceptance order.                       *)
(*                                                                         *)
(* A cell's value is the state its last barrier leaves: the hazard         *)
(* obligation and the public TextureLayout, which Metal tracks itself      *)
(* because it has no native layouts. They ride the same staging and        *)
(* publication: a command buffer keeps its layout writes and entry         *)
(* requirements in MetalCommandLayoutState, metalPlanBatchLayouts checks   *)
(* the entries at submit, and metalPublishBatchLayouts publishes the       *)
(* writes after commit. Each command records one barrier on one cell, from *)
(* a before-layout to layout c + 1, and may repeat a barrier to c + 1 on   *)
(* that cell. Undefined as a before-layout discards the contents and       *)
(* matches any layout. actual is the layout the GPU texture is really in.  *)
(* Metal has no shared textures, so unlike VulkanLayoutPublication.tla     *)
(* every cell's entry layout is known and there is no external change.     *)
(*                                                                         *)
(* Design rules controlled by CONSTANTS (each TRUE in the design):         *)
(*   - SelectiveCommit: `submit` publishes only the cell written by that   *)
(*     command buffer rather than overwriting untouched cells with a stale *)
(*     recording-time snapshot                                             *)
(*     (`MetalHazardPublication-overwriteuntouched.cfg` sets FALSE).       *)
(*   - DeferPublishToSubmit: `record` keeps hazard effects staged locally  *)
(*     until `submit` rather than mutating committed queue state eagerly   *)
(*     (`MetalHazardPublication-publishrecording.cfg` sets FALSE).         *)
(*   - CheckBeforeLayouts: a barrier whose before-layout differs from the  *)
(*     layout its command already staged is rejected when recorded         *)
(*     (metalCheckLayoutTransitions), and a submission whose entry         *)
(*     requirement differs from the committed layout is refused without    *)
(*     effects (metalPlanBatchLayouts)                                     *)
(*     (`MetalHazardPublication-trustbefore.cfg` sets FALSE).              *)
(*                                                                         *)
(* Abstracted: Metal driver execution scheduling, cross-queue fences       *)
(* (a caller obligation), and fine-grained MTLStages bitmasks.             *)
(***************************************************************************)
EXTENDS Naturals, Sequences

CONSTANTS
    SelectiveCommit,        \* TRUE: submit commits only target[c]
    DeferPublishToSubmit,   \* TRUE: record does not mutate committed
    CheckBeforeLayouts      \* TRUE: before-layouts are checked at record and submit

Commands == {0, 1}
Cells == {0, 1}
Hazards == 0..2
Undefined == 3
BeforeLayouts == Hazards \cup {Undefined}
Phases == {"empty", "recorded", "abandoned", "submitted"}
InitialCells == [x \in Cells |-> 0]

VARIABLES
    phase,      \* per-command lifecycle state in [Commands -> Phases]
    target,     \* cell targeted by each command in [Commands -> Cells]
    staged,     \* per-command staged hazard map in [Commands -> [Cells -> Hazards]]
    entry,      \* per-command entry requirement: the first barrier's before-layout
    inOrder,    \* per-command: every later barrier started from the staged layout
    committed,  \* queue-committed hazard state per cell in [Cells -> Hazards]
    accepted,   \* sequence of accepted cell updates in submission order
    actual,     \* layout the GPU texture is really in, per cell
    misused     \* a submitted barrier named a wrong before-layout

vars == <<phase, target, staged, entry, inOrder, committed, accepted, actual, misused>>

Init ==
    /\ phase = [c \in Commands |-> "empty"]
    /\ target = [c \in Commands |-> 0]
    /\ staged = [c \in Commands |-> InitialCells]
    /\ entry = [c \in Commands |-> Undefined]
    /\ inOrder = [c \in Commands |-> TRUE]
    /\ committed = InitialCells
    /\ accepted = <<>>
    /\ actual = InitialCells
    /\ misused = FALSE

\* Record a barrier/write on cell s in command c; keep committed untouched when deferred.
\* Recording cannot see the committed layout, so the before-layout becomes the entry requirement.
record(c, s, before) ==
    /\ phase[c] = "empty"
    /\ phase' = [phase EXCEPT ![c] = "recorded"]
    /\ target' = [target EXCEPT ![c] = s]
    /\ staged' = [staged EXCEPT ![c] = [committed EXCEPT ![s] = c + 1]]
    /\ entry' = [entry EXCEPT ![c] = before]
    /\ committed' =
        IF DeferPublishToSubmit
        THEN committed
        ELSE [committed EXCEPT ![s] = c + 1]
    /\ UNCHANGED <<inOrder, accepted, actual, misused>>

\* A further barrier to c + 1 on the command's cell. With before = c + 1 it
\* composes with the first barrier; with the first barrier's before-layout it
\* repeats the same transition.
rebarrier(c, before) ==
    /\ phase[c] = "recorded"
    /\ LET matches == before = Undefined \/ before = staged[c][target[c]]
       IN /\ CheckBeforeLayouts => matches
          /\ inOrder' = [inOrder EXCEPT ![c] = inOrder[c] /\ matches]
    /\ UNCHANGED <<phase, target, staged, entry, committed, accepted, actual, misused>>

\* Resetting a recorded command clears its staged barriers without publishing.
reset(c) ==
    /\ phase[c] = "recorded"
    /\ phase' = [phase EXCEPT ![c] = "empty"]
    /\ staged' = [staged EXCEPT ![c] = InitialCells]
    /\ entry' = [entry EXCEPT ![c] = Undefined]
    /\ inOrder' = [inOrder EXCEPT ![c] = TRUE]
    /\ UNCHANGED <<target, committed, accepted, actual, misused>>

\* Abandoning an unsubmitted command discards staged barriers without publishing.
abandon(c) ==
    /\ phase[c] \in {"empty", "recorded"}
    /\ phase' = [phase EXCEPT ![c] = "abandoned"]
    /\ staged' = [staged EXCEPT ![c] = InitialCells]
    /\ entry' = [entry EXCEPT ![c] = Undefined]
    /\ inOrder' = [inOrder EXCEPT ![c] = TRUE]
    /\ UNCHANGED <<target, committed, accepted, actual, misused>>

EntryMatches(c) ==
    entry[c] = Undefined \/ entry[c] = committed[target[c]]

\* The command's barriers really are wrong: the entry requirement differs
\* from the GPU layout, or a later barrier did not start where the earlier
\* one left the cell.
WrongBefore(c) ==
    \/ ~inOrder[c]
    \/ /\ entry[c] # Undefined
       /\ entry[c] # actual[target[c]]

\* Submitting command c commits only its touched cell in acceptance order.
\* A refused submission is a disabled step: it changes nothing, and the
\* command stays recorded.
submit(c) ==
    /\ phase[c] = "recorded"
    /\ CheckBeforeLayouts => EntryMatches(c)
    /\ phase' = [phase EXCEPT ![c] = "submitted"]
    /\ staged' = [staged EXCEPT ![c] = InitialCells]
    /\ entry' = [entry EXCEPT ![c] = Undefined]
    /\ inOrder' = [inOrder EXCEPT ![c] = TRUE]
    /\ committed' =
        IF SelectiveCommit
        THEN [committed EXCEPT ![target[c]] = staged[c][target[c]]]
        ELSE staged[c]
    /\ accepted' = Append(accepted, [cell |-> target[c], hazard |-> c + 1])
    /\ actual' = [actual EXCEPT ![target[c]] = c + 1]
    /\ misused' = (misused \/ WrongBefore(c))
    /\ UNCHANGED target

Next ==
    \E c \in Commands :
        \/ \E s \in Cells, before \in BeforeLayouts : record(c, s, before)
        \/ \E before \in BeforeLayouts : rebarrier(c, before)
        \/ reset(c)
        \/ abandon(c)
        \/ submit(c)

Spec == Init /\ [][Next]_vars

RECURSIVE Replay(_, _)
Replay(image, writes) ==
    IF writes = <<>>
    THEN image
    ELSE Replay([image EXCEPT ![Head(writes).cell] = Head(writes).hazard], Tail(writes))

ExpectedCommitted == Replay(InitialCells, accepted)

TypeOK ==
    /\ SelectiveCommit \in BOOLEAN
    /\ DeferPublishToSubmit \in BOOLEAN
    /\ CheckBeforeLayouts \in BOOLEAN
    /\ phase \in [Commands -> Phases]
    /\ target \in [Commands -> Cells]
    /\ staged \in [Commands -> [Cells -> Hazards]]
    /\ entry \in [Commands -> BeforeLayouts]
    /\ inOrder \in [Commands -> BOOLEAN]
    /\ committed \in [Cells -> Hazards]
    /\ Len(accepted) \in 0..2
    /\ \A i \in 1..Len(accepted) :
        accepted[i] \in [cell : Cells, hazard : 1..2]
    /\ actual \in [Cells -> Hazards]
    /\ misused \in BOOLEAN

CommittedMatchesAcceptedReplay ==
    committed = ExpectedCommitted

RecordedStagesTargetHazard ==
    \A c \in Commands :
        phase[c] = "recorded" => staged[c][target[c]] = c + 1

RetiredClearsStaged ==
    \A c \in Commands :
        phase[c] \in {"abandoned", "submitted"} =>
            /\ staged[c] = InitialCells
            /\ entry[c] = Undefined
            /\ inOrder[c]

AcceptedEntriesAreSubmitted ==
    \A i \in 1..Len(accepted) :
        phase[accepted[i].hazard - 1] = "submitted"

\* The backend's tracked layout of every cell is the GPU's.
TrackedLayoutsMatchGpu ==
    committed = actual

\* No submitted barrier named a wrong before-layout.
NoMisusedBarrierSubmitted ==
    ~misused

\* A submission is refused only when its barriers really are wrong.
RefusalsAreGenuine ==
    \A c \in Commands :
        (phase[c] = "recorded" /\ CheckBeforeLayouts /\ ~EntryMatches(c)) => WrongBefore(c)

Safety ==
    /\ CommittedMatchesAcceptedReplay
    /\ RecordedStagesTargetHazard
    /\ RetiredClearsStaged
    /\ AcceptedEntriesAreSubmitted
    /\ TrackedLayoutsMatchGpu
    /\ NoMisusedBarrierSubmitted
    /\ RefusalsAreGenuine

=============================================================================
