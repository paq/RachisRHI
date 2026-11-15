---------------------- MODULE VulkanLayoutPublication -----------------------
(***************************************************************************)
(* Formal safety model for per-command image-layout staging, reset,        *)
(* abandonment/quarantine, selective queue-acceptance publication, and the *)
(* check of each texture barrier's before-layout:                          *)
(*   - src/Vulkan/VulkanBarriers.cpp (VulkanResourceAllocator::stageLayout *)
(*     records per-command layout writes; commitCommandLayouts applies     *)
(*     only those writes; discardCommandLayouts drops them;                *)
(*     validateBarrierEntryLayoutsLocked rejects a barrier whose           *)
(*     before-layout differs from the layout the command already set;      *)
(*     recordBarrierLayoutTransitionsLocked records the first barrier's    *)
(*     before-layout as the command's entry requirement;                   *)
(*     planBatchLayoutTransitions compares entry requirements with the     *)
(*     layouts earlier submissions leave)                                  *)
(*   - src/Vulkan/VulkanCopies.cpp (copy validation reads the command's    *)
(*     staged layouts through recordedLayout)                              *)
(*   - src/Vulkan/VulkanCommands.cpp (VulkanCommandBuffer::resetRecording  *)
(*     and abandon discard staged layouts)                                 *)
(*   - src/Vulkan/VulkanQueues.cpp (publishSubmittedBatch commits on queue *)
(*     acceptance; quarantineBatch discards on failed submission)          *)
(*                                                                         *)
(* Each command records one barrier on one cell, from a before-layout to   *)
(* layout c + 1, and may repeat a barrier to c + 1 on that cell. Undefined *)
(* as a before-layout discards the contents and matches any layout. The    *)
(* command's normal form is its entry requirement (the first barrier's     *)
(* before-layout) and its staged exit layout. actual is the layout the GPU *)
(* image is really in; committed is what the backend tracks. Cells in      *)
(* SharedCells belong to a shared texture, whose layout another device or  *)
(* process may change at any time (external).                              *)
(*                                                                         *)
(* Design-rule constants (TRUE in VulkanLayoutPublication.cfg, FALSE in    *)
(* counterexample configurations):                                         *)
(*   - DeferPublishToSubmit: command recording stages layout changes       *)
(*     privately without mutating device-committed layouts before submit.  *)
(*   - SelectiveCommit: queue acceptance updates only subresources written *)
(*     by the submitted command instead of overwriting untouched cells     *)
(*     with stale record-time snapshots.                                   *)
(*   - CheckBeforeLayouts: a barrier whose before-layout differs from the  *)
(*     layout its command already staged is rejected when recorded, and a  *)
(*     submission whose entry requirement differs from the committed       *)
(*     layout is refused without effects.                                  *)
(*   - SkipUnknownEntry: the submit-time check skips cells of shared       *)
(*     textures, whose layout on entry the backend cannot know.            *)
(*                                                                         *)
(* Trusted boundary: before-layouts on shared textures, and cross-queue    *)
(* GPU timeline synchronization.                                           *)
(***************************************************************************)
EXTENDS Naturals, Sequences

CONSTANTS DeferPublishToSubmit, SelectiveCommit, CheckBeforeLayouts, SkipUnknownEntry

ASSUME {DeferPublishToSubmit, SelectiveCommit, CheckBeforeLayouts, SkipUnknownEntry} \subseteq BOOLEAN

Commands == {0, 1}
Cells == {0, 1}
SharedCells == {1}
Layouts == {0, 1, 2}
Undefined == 3
BeforeLayouts == Layouts \cup {Undefined}
Phases == {"empty", "recorded", "discarded", "submitted"}
InitialLayouts == [s \in Cells |-> 0]
WriteRecords == [cell : Cells, layout : {1, 2}]

BoundedSeq(S, maxLen) == UNION {[1..n -> S] : n \in 0..maxLen}

VARIABLES
  phase,     \* Per-command lifecycle state.
  target,    \* Subresource cell targeted by each recorded command.
  staged,    \* Per-command staged layout map captured during recording.
  entry,     \* Per-command entry requirement: the first barrier's before-layout.
  inOrder,   \* Per-command: every later barrier started from the staged layout.
  committed, \* Device-visible committed subresource layouts.
  accepted,  \* Ordered log of accepted submission writes.
  actual,    \* Layout the GPU image is really in.
  misused    \* A submitted barrier on a tracked cell named a wrong before-layout.

vars == <<phase, target, staged, entry, inOrder, committed, accepted, actual, misused>>

Init ==
  /\ phase = [c \in Commands |-> "empty"]
  /\ target = [c \in Commands |-> 0]
  /\ staged = [c \in Commands |-> InitialLayouts]
  /\ entry = [c \in Commands |-> Undefined]
  /\ inOrder = [c \in Commands |-> TRUE]
  /\ committed = InitialLayouts
  /\ accepted = <<>>
  /\ actual = InitialLayouts
  /\ misused = FALSE

record(c, s, before) ==
  /\ phase[c] = "empty"
  /\ phase' = [phase EXCEPT ![c] = "recorded"]
  /\ target' = [target EXCEPT ![c] = s]
  /\ staged' = [staged EXCEPT ![c] = [committed EXCEPT ![s] = c + 1]]
  /\ entry' = [entry EXCEPT ![c] = before]
  /\ IF DeferPublishToSubmit
     THEN UNCHANGED committed
     ELSE committed' = [committed EXCEPT ![s] = c + 1]
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

reset(c) ==
  /\ phase[c] = "recorded"
  /\ phase' = [phase EXCEPT ![c] = "empty"]
  /\ staged' = [staged EXCEPT ![c] = InitialLayouts]
  /\ entry' = [entry EXCEPT ![c] = Undefined]
  /\ inOrder' = [inOrder EXCEPT ![c] = TRUE]
  /\ UNCHANGED <<target, committed, accepted, actual, misused>>

\* Covers both explicit command abandonment and failed/quarantined submission.
discard(c) ==
  /\ phase[c] \in {"empty", "recorded"}
  /\ phase' = [phase EXCEPT ![c] = "discarded"]
  /\ staged' = [staged EXCEPT ![c] = InitialLayouts]
  /\ entry' = [entry EXCEPT ![c] = Undefined]
  /\ inOrder' = [inOrder EXCEPT ![c] = TRUE]
  /\ UNCHANGED <<target, committed, accepted, actual, misused>>

EntryUnknown(c) == SkipUnknownEntry /\ target[c] \in SharedCells

EntryMatches(c) ==
  entry[c] = Undefined \/ EntryUnknown(c) \/ entry[c] = committed[target[c]]

\* The command's barriers really are wrong: the entry requirement differs
\* from the GPU layout, or a later barrier did not start where the earlier
\* one left the cell.
WrongBefore(c) ==
  \/ ~inOrder[c]
  \/ /\ entry[c] # Undefined
     /\ entry[c] # actual[target[c]]

\* A wrong before-layout the backend is responsible for: not one on a shared
\* texture's cell, which is the trusted boundary.
Misuses(c) ==
  \/ ~inOrder[c]
  \/ /\ target[c] \notin SharedCells
     /\ WrongBefore(c)

\* A refused submission is a disabled step: it changes nothing.
submit(c) ==
  /\ phase[c] = "recorded"
  /\ CheckBeforeLayouts => EntryMatches(c)
  /\ phase' = [phase EXCEPT ![c] = "submitted"]
  /\ staged' = [staged EXCEPT ![c] = InitialLayouts]
  /\ entry' = [entry EXCEPT ![c] = Undefined]
  /\ inOrder' = [inOrder EXCEPT ![c] = TRUE]
  /\ committed' =
       IF SelectiveCommit
       THEN [committed EXCEPT ![target[c]] = staged[c][target[c]]]
       ELSE staged[c]
  /\ accepted' = Append(accepted, [cell |-> target[c], layout |-> c + 1])
  /\ actual' = [actual EXCEPT ![target[c]] = c + 1]
  /\ misused' = (misused \/ Misuses(c))
  /\ UNCHANGED target

\* Another device or process transitions a shared texture.
external(s, layout) ==
  /\ s \in SharedCells
  /\ actual' = [actual EXCEPT ![s] = layout]
  /\ UNCHANGED <<phase, target, staged, entry, inOrder, committed, accepted, misused>>

Next ==
  \/ \E c \in Commands :
       \/ \E s \in Cells, before \in BeforeLayouts : record(c, s, before)
       \/ \E before \in BeforeLayouts : rebarrier(c, before)
       \/ reset(c)
       \/ discard(c)
       \/ submit(c)
  \/ \E s \in Cells, layout \in Layouts : external(s, layout)

Spec == Init /\ [][Next]_vars

RECURSIVE Replay(_, _)
Replay(image, writes) ==
  IF writes = <<>>
  THEN image
  ELSE Replay([image EXCEPT ![Head(writes).cell] = Head(writes).layout], Tail(writes))

ExpectedCommitted == Replay(InitialLayouts, accepted)

TypeOK ==
  /\ phase \in [Commands -> Phases]
  /\ target \in [Commands -> Cells]
  /\ staged \in [Commands -> [Cells -> Layouts]]
  /\ entry \in [Commands -> BeforeLayouts]
  /\ inOrder \in [Commands -> BOOLEAN]
  /\ committed \in [Cells -> Layouts]
  /\ accepted \in BoundedSeq(WriteRecords, 2)
  /\ actual \in [Cells -> Layouts]
  /\ misused \in BOOLEAN

CommittedMatchesAcceptedReplay ==
  committed = ExpectedCommitted

RecordedCommandStagesTarget ==
  \A c \in Commands :
    phase[c] = "recorded" => staged[c][target[c]] = c + 1

RetiredCommandClearsStaging ==
  \A c \in Commands :
    phase[c] \in {"empty", "discarded", "submitted"} =>
      /\ staged[c] = InitialLayouts
      /\ entry[c] = Undefined
      /\ inOrder[c]

\* The backend's tracked layout of every cell it can know is the GPU's.
TrackedLayoutsMatchGpu ==
  \A s \in Cells \ SharedCells : committed[s] = actual[s]

\* No submitted barrier on a tracked cell named a wrong before-layout.
NoMisusedBarrierSubmitted ==
  ~misused

\* A submission is refused only when its barriers really are wrong.
RefusalsAreGenuine ==
  \A c \in Commands :
    (phase[c] = "recorded" /\ CheckBeforeLayouts /\ ~EntryMatches(c)) => WrongBefore(c)

Safety ==
  /\ CommittedMatchesAcceptedReplay
  /\ RecordedCommandStagesTarget
  /\ RetiredCommandClearsStaging
  /\ TrackedLayoutsMatchGpu
  /\ NoMisusedBarrierSubmitted
  /\ RefusalsAreGenuine

StepSafety ==
  [][(accepted' = accepted /\ DeferPublishToSubmit) => committed' = committed]_vars

=============================================================================
