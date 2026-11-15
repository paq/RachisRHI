------------------------- MODULE D3D12TextureLayouts -------------------------
(***************************************************************************)
(* Formal safety model for the Direct3D 12 check of each texture barrier's *)
(* before-layout, on both the enhanced and the legacy barrier path:        *)
(*   - src/D3D12/D3D12LegacyTextureStates.h (LegacyTextureStateTracker     *)
(*     holds each subresource's portable layout after the last submitted   *)
(*     command list; LegacyTextureStateUse::recordLayout keeps a command   *)
(*     list's entry requirement and exit layout; checkTextureLayoutEntries *)
(*     compares the entry requirements of a batch, in order; commit        *)
(*     publishes exit layouts after ExecuteCommandLists)                   *)
(*   - src/D3D12/D3D12ResourceBarriers.cpp (prepareTextureBarrierLayouts   *)
(*     rejects a barrier whose before-layout differs from the layout its   *)
(*     command list already set; validateOrderedTextureBarrierLayouts does *)
(*     the same for generateMipmaps' ordered sequence)                     *)
(*   - src/D3D12/D3D12Queue.cpp (submit runs checkSubmissionTextureLayouts *)
(*     before any prologue list is recorded or any list executes)          *)
(*                                                                         *)
(* Each command list records one barrier on one cell, from a before-layout *)
(* to layout c + 1, and may repeat a barrier to c + 1 on that cell. Layout *)
(* 0 is common, in which D3D12 creates a texture. Undefined as a before-   *)
(* layout discards the contents and matches any layout. actual is the      *)
(* layout the GPU resource is really in; committed is what the tracker     *)
(* holds. Cells in SharedCells belong to a shared or imported texture,     *)
(* whose layout another device, process, or the application may change at *)
(* any time (external). A refused submission is an action that changes     *)
(* nothing but records whether the refusal was genuine.                    *)
(*                                                                         *)
(* Design-rule constants (TRUE in D3D12TextureLayouts.cfg, FALSE in        *)
(* counterexample configurations):                                         *)
(*   - CheckBeforeLayouts: a barrier whose before-layout differs from the  *)
(*     layout its command list already set is rejected when recorded, and  *)
(*     a submission whose entry requirement differs from the tracked       *)
(*     layout is refused without effects.                                  *)
(*   - SkipUnknownEntry: the submit-time check skips cells of shared and   *)
(*     imported textures, whose layout on entry the backend cannot know.   *)
(*   - SimulateBatch: command lists submitted together are checked in      *)
(*     order, each against the layouts the ones before it leave, rather    *)
(*     than each against the tracked layouts.                              *)
(*                                                                         *)
(* Trusted boundary: before-layouts on shared and imported textures, and   *)
(* cross-queue GPU timeline synchronization.                               *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS CheckBeforeLayouts, SkipUnknownEntry, SimulateBatch

ASSUME {CheckBeforeLayouts, SkipUnknownEntry, SimulateBatch} \subseteq BOOLEAN

Commands == {0, 1}
Cells == {0, 1}
SharedCells == {1}
Layouts == {0, 1, 2}
Undefined == 3
BeforeLayouts == Layouts \cup {Undefined}
Phases == {"empty", "recorded", "retired"}

VARIABLES
  phase,          \* Per-command lifecycle state.
  target,         \* Cell each recorded command list transitions.
  entry,          \* Per-command entry requirement: the first barrier's before-layout.
  inOrder,        \* Per-command: every later barrier started from the layout it left.
  committed,      \* The tracker's layout of each cell.
  actual,         \* The layout the GPU resource is really in.
  misused,        \* A submitted barrier on a tracked cell named a wrong before-layout.
  falselyRefused  \* A submission was refused although its barriers were right.

vars == <<phase, target, entry, inOrder, committed, actual, misused, falselyRefused>>

Init ==
  /\ phase = [c \in Commands |-> "empty"]
  /\ target = [c \in Commands |-> 0]
  /\ entry = [c \in Commands |-> Undefined]
  /\ inOrder = [c \in Commands |-> TRUE]
  /\ committed = [s \in Cells |-> 0]
  /\ actual = [s \in Cells |-> 0]
  /\ misused = FALSE
  /\ falselyRefused = FALSE

record(c, s, before) ==
  /\ phase[c] = "empty"
  /\ phase' = [phase EXCEPT ![c] = "recorded"]
  /\ target' = [target EXCEPT ![c] = s]
  /\ entry' = [entry EXCEPT ![c] = before]
  /\ UNCHANGED <<inOrder, committed, actual, misused, falselyRefused>>

\* A further barrier to c + 1 on the command list's cell, from `before`.
rebarrier(c, before) ==
  /\ phase[c] = "recorded"
  /\ LET matches == before = Undefined \/ before = c + 1
     IN /\ CheckBeforeLayouts => matches
        /\ inOrder' = [inOrder EXCEPT ![c] = inOrder[c] /\ matches]
  /\ UNCHANGED <<phase, target, entry, committed, actual, misused, falselyRefused>>

\* Reset, abandonment, and quarantine drop the command list's layouts.
retire(c) ==
  /\ phase[c] = "recorded"
  /\ phase' = [phase EXCEPT ![c] = "retired"]
  /\ entry' = [entry EXCEPT ![c] = Undefined]
  /\ inOrder' = [inOrder EXCEPT ![c] = TRUE]
  /\ UNCHANGED <<target, committed, actual, misused, falselyRefused>>

\* The layout a cell is in when command list b starts, in the tracker (`layouts`
\* committed) or on the GPU (`layouts` actual), after the lists in `before` ran.
LayoutAfter(layouts, s, before) ==
  IF \E a \in before : target[a] = s
  THEN (CHOOSE a \in before : target[a] = s) + 1
  ELSE layouts[s]

EntryMatches(c, before) ==
  \/ entry[c] = Undefined
  \/ (SkipUnknownEntry /\ target[c] \in SharedCells)
  \/ entry[c] = LayoutAfter(committed, target[c], before)

WrongBefore(c, before) ==
  \/ ~inOrder[c]
  \/ /\ entry[c] # Undefined
     /\ entry[c] # LayoutAfter(actual, target[c], before)

\* A wrong before-layout the backend is responsible for: not one on a shared
\* or imported texture's cell, which is the trusted boundary.
Misuses(c, before) ==
  \/ ~inOrder[c]
  \/ /\ target[c] \notin SharedCells
     /\ WrongBefore(c, before)

\* The layout of cell s after command list a, then b, ran from `layouts`; a
\* list that does not touch s leaves it as it was.
Exit(layouts, s, a, b) ==
  IF target[b] = s THEN b + 1 ELSE IF target[a] = s THEN a + 1 ELSE layouts[s]

\* ExecuteCommandLists of a then b (a = b for one list), then commit.
Publish(a, b) ==
  /\ phase' = [c \in Commands |-> IF c \in {a, b} THEN "retired" ELSE phase[c]]
  /\ entry' = [c \in Commands |-> IF c \in {a, b} THEN Undefined ELSE entry[c]]
  /\ inOrder' = [c \in Commands |-> IF c \in {a, b} THEN TRUE ELSE inOrder[c]]
  /\ committed' = [s \in Cells |-> Exit(committed, s, a, b)]
  /\ actual' = [s \in Cells |-> Exit(actual, s, a, b)]

submit(c) ==
  /\ phase[c] = "recorded"
  /\ CheckBeforeLayouts => EntryMatches(c, {})
  /\ Publish(c, c)
  /\ misused' = (misused \/ Misuses(c, {}))
  /\ UNCHANGED <<target, falselyRefused>>

refuse(c) ==
  /\ phase[c] = "recorded"
  /\ CheckBeforeLayouts /\ ~EntryMatches(c, {})
  /\ falselyRefused' = (falselyRefused \/ ~WrongBefore(c, {}))
  /\ UNCHANGED <<phase, target, entry, inOrder, committed, actual, misused>>

\* Command lists a then b in one ExecuteCommandLists call. When they share a
\* cell, b starts where a leaves it.
BatchMatches(a, b) ==
  /\ EntryMatches(a, {})
  /\ EntryMatches(b, IF SimulateBatch THEN {a} ELSE {})

submitPair(a, b) ==
  /\ a # b
  /\ phase[a] = "recorded" /\ phase[b] = "recorded"
  /\ CheckBeforeLayouts => BatchMatches(a, b)
  /\ Publish(a, b)
  /\ misused' = (misused \/ Misuses(a, {}) \/ Misuses(b, {a}))
  /\ UNCHANGED <<target, falselyRefused>>

refusePair(a, b) ==
  /\ a # b
  /\ phase[a] = "recorded" /\ phase[b] = "recorded"
  /\ CheckBeforeLayouts /\ ~BatchMatches(a, b)
  /\ falselyRefused' = (falselyRefused \/ ~(WrongBefore(a, {}) \/ WrongBefore(b, {a})))
  /\ UNCHANGED <<phase, target, entry, inOrder, committed, actual, misused>>

\* Another device, process, or the application transitions a shared or
\* imported texture.
external(s, layout) ==
  /\ s \in SharedCells
  /\ actual' = [actual EXCEPT ![s] = layout]
  /\ UNCHANGED <<phase, target, entry, inOrder, committed, misused, falselyRefused>>

Next ==
  \/ \E c \in Commands :
       \/ \E s \in Cells, before \in BeforeLayouts : record(c, s, before)
       \/ \E before \in BeforeLayouts : rebarrier(c, before)
       \/ retire(c)
       \/ submit(c)
       \/ refuse(c)
  \/ \E a, b \in Commands : submitPair(a, b) \/ refusePair(a, b)
  \/ \E s \in Cells, layout \in Layouts : external(s, layout)

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ phase \in [Commands -> Phases]
  /\ target \in [Commands -> Cells]
  /\ entry \in [Commands -> BeforeLayouts]
  /\ inOrder \in [Commands -> BOOLEAN]
  /\ committed \in [Cells -> Layouts]
  /\ actual \in [Cells -> Layouts]
  /\ misused \in BOOLEAN
  /\ falselyRefused \in BOOLEAN

\* The tracker's layout of every cell it can know is the GPU's.
TrackedLayoutsMatchGpu ==
  \A s \in Cells \ SharedCells : committed[s] = actual[s]

\* No submitted barrier on a tracked cell named a wrong before-layout.
NoMisusedBarrierSubmitted ==
  ~misused

\* A submission is refused only when its barriers really are wrong.
NoFalseRefusal ==
  ~falselyRefused

RetiredCommandClearsLayouts ==
  \A c \in Commands : phase[c] \in {"empty", "retired"} => (entry[c] = Undefined /\ inOrder[c])

Safety ==
  /\ TrackedLayoutsMatchGpu
  /\ NoMisusedBarrierSubmitted
  /\ NoFalseRefusal
  /\ RetiredCommandClearsLayouts

=============================================================================
