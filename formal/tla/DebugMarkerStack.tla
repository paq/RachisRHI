-------------------------- MODULE DebugMarkerStack --------------------------
(***************************************************************************)
(* Portable debug-marker and GPU-profile group nesting stack, same-thread  *)
(* ownership immutability, owner-kind matching, balanced submission gate,  *)
(* and failure-atomic rejection.                                           *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12DebugMarker.cpp (validateGroupBegin,                 *)
(*     validateGroupEnd, commitGroupBegin, commitGroupEnd,                 *)
(*     DebugMarkerManager::beginGroup, endGroup, insertMarker)             *)
(*   - src/D3D12/D3D12Queue.cpp (validateNoOpenRecordingScopes via         *)
(*     QueueManager::submit, D3D12CommandBuffer::validateReset)            *)
(*                                                                         *)
(* Contract modeled:                                                       *)
(*   - Group depth is bounded by MaxDepth = 2; opening from depth = 0      *)
(*     records both ownerThread and openingThread for the outer scope.     *)
(*   - Nested BeginGroup and EndGroup require recording state, matching    *)
(*     thread (when EnforceSameThreadNesting = TRUE,                       *)
(*     D3D12DebugMarker.cpp), and matching group-owner                     *)
(*     kind ("user" vs "profile", D3D12DebugMarker.cpp) on End.            *)
(*   - Rejected Begin, End, InsertMarker, and Submit attempts preserve     *)
(*     depth, ownerThread, and stack contents atomically; unbalanced       *)
(*     submissions remain recording and repairable by closing open groups. *)
(*   - Abandon discards open debug groups and clears thread ownership.     *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - Builds with RACHIS_ENABLE_DEBUG_MARKERS == 0, where public user     *)
(*     marker APIs return ErrorCode::unsupported unconditionally           *)
(*     (D3D12DebugMarker.cpp); this spec                                   *)
(*     models an enabled marker build.                                     *)
(***************************************************************************)
EXTENDS Naturals, Sequences

CONSTANT EnforceSameThreadNesting

ASSUME EnforceSameThreadNesting \in BOOLEAN

MaxDepth == 2
Threads == {0, 1}
NoThread == 2
Kinds == {"user", "profile"}
Phases == {"recording", "submitted", "abandoned"}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    cmdPhase,       \* Phases
    depth,          \* 0..MaxDepth
    groupStack,     \* Seq(Kinds) of length depth
    ownerThread,    \* Threads \cup {NoThread}: state.groupThread
    openingThread,  \* Threads \cup {NoThread}: thread that opened depth 0 -> 1
    lastOutcome     \* Outcomes

coreVars == <<cmdPhase, depth, groupStack, ownerThread, openingThread>>
vars == <<cmdPhase, depth, groupStack, ownerThread, openingThread, lastOutcome>>

BeginValid(thread, descValid) ==
    /\ cmdPhase = "recording"
    /\ descValid
    /\ depth < MaxDepth
    /\ (EnforceSameThreadNesting => (depth = 0 \/ ownerThread = thread))

EndValid(thread, kind) ==
    /\ cmdPhase = "recording"
    /\ depth > 0
    /\ groupStack[depth] = kind
    /\ (EnforceSameThreadNesting => ownerThread = thread)

TypeOK ==
    /\ cmdPhase \in Phases
    /\ depth \in 0..MaxDepth
    /\ groupStack \in Seq(Kinds)
    /\ Len(groupStack) = depth
    /\ ownerThread \in (Threads \cup {NoThread})
    /\ openingThread \in (Threads \cup {NoThread})
    /\ lastOutcome \in Outcomes

Init ==
    /\ cmdPhase = "recording"
    /\ depth = 0
    /\ groupStack = <<>>
    /\ ownerThread = NoThread
    /\ openingThread = NoThread
    /\ lastOutcome = "none"

AttemptBegin(thread, kind, descValid) ==
    IF BeginValid(thread, descValid)
    THEN /\ depth' = depth + 1
         /\ groupStack' = Append(groupStack, kind)
         /\ ownerThread' =
              IF depth = 0 \/ ~EnforceSameThreadNesting
              THEN thread
              ELSE ownerThread
         /\ openingThread' = IF depth = 0 THEN thread ELSE openingThread
         /\ lastOutcome' = "accepted"
         /\ UNCHANGED cmdPhase
    ELSE /\ lastOutcome' = "rejected"
         /\ UNCHANGED coreVars

AttemptEnd(thread, kind) ==
    IF EndValid(thread, kind)
    THEN /\ depth' = depth - 1
         /\ groupStack' = SubSeq(groupStack, 1, depth - 1)
         /\ ownerThread' = IF depth = 1 THEN NoThread ELSE ownerThread
         /\ openingThread' = IF depth = 1 THEN NoThread ELSE openingThread
         /\ lastOutcome' = "accepted"
         /\ UNCHANGED cmdPhase
    ELSE /\ lastOutcome' = "rejected"
         /\ UNCHANGED coreVars

AttemptInsertMarker(descValid) ==
    IF cmdPhase = "recording" /\ descValid
    THEN /\ lastOutcome' = "accepted"
         /\ UNCHANGED coreVars
    ELSE /\ lastOutcome' = "rejected"
         /\ UNCHANGED coreVars

AttemptSubmit ==
    IF cmdPhase = "recording" /\ depth = 0
    THEN /\ cmdPhase' = "submitted"
         /\ lastOutcome' = "accepted"
         /\ UNCHANGED <<depth, groupStack, ownerThread, openingThread>>
    ELSE /\ lastOutcome' = "rejected"
         /\ UNCHANGED coreVars

Abandon ==
    /\ cmdPhase = "recording"
    /\ cmdPhase' = "abandoned"
    /\ depth' = 0
    /\ groupStack' = <<>>
    /\ ownerThread' = NoThread
    /\ openingThread' = NoThread
    /\ lastOutcome' = "none"

Next ==
    \/ \E t \in Threads, k \in Kinds, v \in BOOLEAN : AttemptBegin(t, k, v)
    \/ \E t \in Threads, k \in Kinds : AttemptEnd(t, k)
    \/ \E v \in BOOLEAN : AttemptInsertMarker(v)
    \/ AttemptSubmit
    \/ Abandon

Spec == Init /\ [][Next]_vars

OwnershipMatchesDepth ==
    /\ (depth = 0) <=> (ownerThread = NoThread)
    /\ (depth = 0) <=> (openingThread = NoThread)

\* While any debug or profile group remains open, ownerThread stays pinned
\* to the exact thread that opened the outermost group.
ActiveGroupMatchesOpeningThread ==
    depth > 0 => ownerThread = openingThread

TerminalCommandsHaveNoOpenGroups ==
    cmdPhase # "recording" =>
        /\ depth = 0
        /\ groupStack = <<>>
        /\ ownerThread = NoThread
        /\ openingThread = NoThread

Safety ==
    /\ OwnershipMatchesDepth
    /\ ActiveGroupMatchesOpeningThread
    /\ TerminalCommandsHaveNoOpenGroups

StepSafety ==
    [][lastOutcome' = "rejected" => coreVars' = coreVars]_vars
=============================================================================
