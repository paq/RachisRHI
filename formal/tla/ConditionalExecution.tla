------------------------ MODULE ConditionalExecution ------------------------
(***************************************************************************)
(* Conditional execution (predication) scope validation, 8-byte uint64     *)
(* alignment and bounds checking, non-nesting enforcement, buffer          *)
(* retention across recording and GPU flight, and submit balance gate.     *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Query.cpp (QueryManager::beginConditionalExecution,  *)
(*     QueryManager::endConditionalExecution)                              *)
(*   - src/D3D12/D3D12Queue.cpp (QueueManager::retainResource,             *)
(*     QueueManager::submit through resolveSubmissionCommands and          *)
(*     validateNoOpenRecordingScopes)                                      *)
(*   - src/Device.cpp (Device::submit through                              *)
(*     detail::validateCommandBufferReadyForSubmit in                      *)
(*     src/Internal/PortableValidation.h)                                  *)
(*                                                                         *)
(* Contract modeled:                                                       *)
(*   - Conditional execution requires a graphics or compute command buffer *)
(*     (copy queues reject), no already-active scope, a valid operation    *)
(*     (equalZero or notEqualZero), a live buffer with predicate usage,    *)
(*     and (when EnforceAlignmentAndBounds = TRUE, D3D12Query.cpp)         *)
(*     an 8-byte aligned offset whose 8-byte uint64 value fits             *)
(*     inside bufferSize.                                                  *)
(*   - Begin retains the predicate buffer in recording state and enables   *)
(*     native predication; End disables predication while preserving the   *)
(*     command buffer's retained buffer lease until completion/abandon.    *)
(*   - Submission requires any opened conditional scope to be closed first *)
(*     and transfers the recording buffer lease to in-flight retention.    *)
(*   - Rejected Begin, End, ConditionalCommand, and Submit calls are       *)
(*     failure-atomic across all command and buffer state.                 *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - GPU-side asynchronous writes to the predicate uint64 value before   *)
(*     command execution (modeled as nondeterministic predicateZero).      *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT EnforceAlignmentAndBounds

ASSUME EnforceAlignmentAndBounds \in BOOLEAN

PredicateAlignment == 8
PredicateValueBytes == 8
BufferSizes == {8, 16}
Offsets == {0, 4, 8, 16}
Queues == {"graphics", "compute", "copy"}
ValidOps == {"equalZero", "notEqualZero"}
Operations == {"equalZero", "notEqualZero", "invalidOp"}
Phases == {"idle", "recording", "submitted"}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    cmdPhase,                   \* Phases
    queueType,                  \* Queues
    logicalBufferAlive,         \* BOOLEAN
    physicalBufferAlive,        \* BOOLEAN
    bufferSize,                 \* BufferSizes
    hasPredicateUsage,          \* BOOLEAN
    scopeActive,                \* BOOLEAN: portable activeConditionalExecution
    nativePredicateEnabled,     \* BOOLEAN: native command-list predication
    activeOffset,               \* Offsets: bound predicate byte offset
    recordingRetainsBuffer,     \* BOOLEAN: retained in recording command list
    inFlightRetainsBuffer,      \* BOOLEAN: retained in submitted batch
    predicatePasses,            \* BOOLEAN: comparison outcome for active scope
    conditionalCmdRecorded,     \* BOOLEAN: command recorded inside scope
    visibleEffect,              \* BOOLEAN: command produced GPU-visible effect
    lastOutcome                 \* Outcomes

coreVars == <<cmdPhase, queueType, logicalBufferAlive, physicalBufferAlive,
              bufferSize, hasPredicateUsage, scopeActive,
              nativePredicateEnabled, activeOffset, recordingRetainsBuffer,
              inFlightRetainsBuffer, predicatePasses, conditionalCmdRecorded,
              visibleEffect>>
vars == <<cmdPhase, queueType, logicalBufferAlive, physicalBufferAlive,
          bufferSize, hasPredicateUsage, scopeActive,
          nativePredicateEnabled, activeOffset, recordingRetainsBuffer,
          inFlightRetainsBuffer, predicatePasses, conditionalCmdRecorded,
          visibleEffect, lastOutcome>>

RangeAlignedAndInBounds(offset, size) ==
    /\ (offset % PredicateAlignment) = 0
    /\ (offset + PredicateValueBytes) <= size

PredicateEvaluatesTrue(op, isZero) ==
    IF op = "equalZero" THEN isZero ELSE ~isZero

BeginValid(offset, op) ==
    /\ cmdPhase = "recording"
    /\ queueType \in {"graphics", "compute"}
    /\ op \in ValidOps
    /\ ~scopeActive
    /\ logicalBufferAlive
    /\ hasPredicateUsage
    /\ (EnforceAlignmentAndBounds => RangeAlignedAndInBounds(offset, bufferSize))

TypeOK ==
    /\ cmdPhase \in Phases
    /\ queueType \in Queues
    /\ logicalBufferAlive \in BOOLEAN
    /\ physicalBufferAlive \in BOOLEAN
    /\ bufferSize \in BufferSizes
    /\ hasPredicateUsage \in BOOLEAN
    /\ scopeActive \in BOOLEAN
    /\ nativePredicateEnabled \in BOOLEAN
    /\ activeOffset \in Offsets
    /\ recordingRetainsBuffer \in BOOLEAN
    /\ inFlightRetainsBuffer \in BOOLEAN
    /\ predicatePasses \in BOOLEAN
    /\ conditionalCmdRecorded \in BOOLEAN
    /\ visibleEffect \in BOOLEAN
    /\ lastOutcome \in Outcomes

Init ==
    /\ cmdPhase = "idle"
    /\ queueType = "graphics"
    /\ logicalBufferAlive = FALSE
    /\ physicalBufferAlive = FALSE
    /\ bufferSize = 8
    /\ hasPredicateUsage = FALSE
    /\ scopeActive = FALSE
    /\ nativePredicateEnabled = FALSE
    /\ activeOffset = 0
    /\ recordingRetainsBuffer = FALSE
    /\ inFlightRetainsBuffer = FALSE
    /\ predicatePasses = FALSE
    /\ conditionalCmdRecorded = FALSE
    /\ visibleEffect = FALSE
    /\ lastOutcome = "none"

CreateBuffer(size, predUsage) ==
    /\ ~logicalBufferAlive
    /\ ~physicalBufferAlive
    /\ logicalBufferAlive' = TRUE
    /\ physicalBufferAlive' = TRUE
    /\ bufferSize' = size
    /\ hasPredicateUsage' = predUsage
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<cmdPhase, queueType, scopeActive, nativePredicateEnabled,
                   activeOffset, recordingRetainsBuffer, inFlightRetainsBuffer,
                   predicatePasses, conditionalCmdRecorded, visibleEffect>>

DestroyBuffer ==
    /\ logicalBufferAlive
    /\ logicalBufferAlive' = FALSE
    /\ physicalBufferAlive' = (recordingRetainsBuffer \/ inFlightRetainsBuffer)
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<cmdPhase, queueType, bufferSize, hasPredicateUsage,
                   scopeActive, nativePredicateEnabled, activeOffset,
                   recordingRetainsBuffer, inFlightRetainsBuffer,
                   predicatePasses, conditionalCmdRecorded, visibleEffect>>

Acquire(q) ==
    /\ cmdPhase = "idle"
    /\ cmdPhase' = "recording"
    /\ queueType' = q
    /\ scopeActive' = FALSE
    /\ nativePredicateEnabled' = FALSE
    /\ activeOffset' = 0
    /\ recordingRetainsBuffer' = FALSE
    /\ predicatePasses' = FALSE
    /\ conditionalCmdRecorded' = FALSE
    /\ visibleEffect' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<logicalBufferAlive, physicalBufferAlive, bufferSize,
                   hasPredicateUsage, inFlightRetainsBuffer>>

AttemptBegin(offset, op, predicateZero) ==
    IF BeginValid(offset, op)
    THEN /\ scopeActive' = TRUE
         /\ nativePredicateEnabled' = TRUE
         /\ activeOffset' = offset
         /\ recordingRetainsBuffer' = TRUE
         /\ predicatePasses' = PredicateEvaluatesTrue(op, predicateZero)
         /\ lastOutcome' = "accepted"
         /\ UNCHANGED <<cmdPhase, queueType, logicalBufferAlive,
                        physicalBufferAlive, bufferSize, hasPredicateUsage,
                        inFlightRetainsBuffer, conditionalCmdRecorded,
                        visibleEffect>>
    ELSE /\ lastOutcome' = "rejected"
         /\ UNCHANGED coreVars

AttemptEnd ==
    IF cmdPhase = "recording" /\ queueType \in {"graphics", "compute"} /\ scopeActive
    THEN /\ scopeActive' = FALSE
         /\ nativePredicateEnabled' = FALSE
         /\ activeOffset' = 0
         /\ predicatePasses' = FALSE
         /\ lastOutcome' = "accepted"
         /\ UNCHANGED <<cmdPhase, queueType, logicalBufferAlive,
                        physicalBufferAlive, bufferSize, hasPredicateUsage,
                        recordingRetainsBuffer, inFlightRetainsBuffer,
                        conditionalCmdRecorded, visibleEffect>>
    ELSE /\ lastOutcome' = "rejected"
         /\ UNCHANGED coreVars

AttemptConditionalCommand ==
    IF cmdPhase = "recording" /\ scopeActive
    THEN /\ conditionalCmdRecorded' = TRUE
         /\ visibleEffect' = (visibleEffect \/ predicatePasses)
         /\ lastOutcome' = "accepted"
         /\ UNCHANGED <<cmdPhase, queueType, logicalBufferAlive,
                        physicalBufferAlive, bufferSize, hasPredicateUsage,
                        scopeActive, nativePredicateEnabled, activeOffset,
                        recordingRetainsBuffer, inFlightRetainsBuffer,
                        predicatePasses>>
    ELSE /\ lastOutcome' = "rejected"
         /\ UNCHANGED coreVars

AttemptSubmit ==
    IF cmdPhase = "recording" /\ ~scopeActive
    THEN /\ cmdPhase' = "submitted"
         /\ recordingRetainsBuffer' = FALSE
         /\ inFlightRetainsBuffer' = recordingRetainsBuffer
         /\ lastOutcome' = "accepted"
         /\ UNCHANGED <<queueType, logicalBufferAlive, physicalBufferAlive,
                        bufferSize, hasPredicateUsage, scopeActive,
                        nativePredicateEnabled, activeOffset, predicatePasses,
                        conditionalCmdRecorded, visibleEffect>>
    ELSE /\ lastOutcome' = "rejected"
         /\ UNCHANGED coreVars

Complete ==
    /\ cmdPhase = "submitted"
    /\ cmdPhase' = "idle"
    /\ queueType' = "graphics"
    /\ inFlightRetainsBuffer' = FALSE
    /\ physicalBufferAlive' = logicalBufferAlive
    /\ conditionalCmdRecorded' = FALSE
    /\ visibleEffect' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<logicalBufferAlive, bufferSize, hasPredicateUsage,
                   scopeActive, nativePredicateEnabled, activeOffset,
                   recordingRetainsBuffer, predicatePasses>>

Abandon ==
    /\ cmdPhase = "recording"
    /\ cmdPhase' = "idle"
    /\ queueType' = "graphics"
    /\ scopeActive' = FALSE
    /\ nativePredicateEnabled' = FALSE
    /\ activeOffset' = 0
    /\ recordingRetainsBuffer' = FALSE
    /\ predicatePasses' = FALSE
    /\ conditionalCmdRecorded' = FALSE
    /\ visibleEffect' = FALSE
    /\ physicalBufferAlive' = (logicalBufferAlive \/ inFlightRetainsBuffer)
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<logicalBufferAlive, bufferSize, hasPredicateUsage,
                   inFlightRetainsBuffer>>

Next ==
    \/ \E sz \in BufferSizes, pu \in BOOLEAN : CreateBuffer(sz, pu)
    \/ DestroyBuffer
    \/ \E q \in Queues : Acquire(q)
    \/ \E off \in Offsets, op \in Operations, pz \in BOOLEAN :
         AttemptBegin(off, op, pz)
    \/ AttemptEnd
    \/ AttemptConditionalCommand
    \/ AttemptSubmit
    \/ Complete
    \/ Abandon

Spec == Init /\ [][Next]_vars

NativeScopeMatchesPortableScope ==
    scopeActive <=> nativePredicateEnabled

ActiveScopeValidAndRetained ==
    scopeActive =>
        /\ cmdPhase = "recording"
        /\ queueType \in {"graphics", "compute"}
        /\ recordingRetainsBuffer
        /\ hasPredicateUsage
        /\ physicalBufferAlive

\* Every active predication binding references an 8-byte aligned uint64
\* wholly contained inside the retained predicate buffer.
ActivePredicateRangeAlignedAndInBounds ==
    scopeActive => RangeAlignedAndInBounds(activeOffset, bufferSize)

PhysicalBufferLifetimeExact ==
    physicalBufferAlive <=>
        (logicalBufferAlive \/ recordingRetainsBuffer \/ inFlightRetainsBuffer)

SubmittedStateBalanced ==
    cmdPhase = "submitted" =>
        /\ ~scopeActive
        /\ ~nativePredicateEnabled
        /\ ~recordingRetainsBuffer

VisibleEffectsRequireRecordedCommand ==
    visibleEffect => conditionalCmdRecorded

Safety ==
    /\ NativeScopeMatchesPortableScope
    /\ ActiveScopeValidAndRetained
    /\ ActivePredicateRangeAlignedAndInBounds
    /\ PhysicalBufferLifetimeExact
    /\ SubmittedStateBalanced
    /\ VisibleEffectsRequireRecordedCommand

StepSafety ==
    [][/\ (lastOutcome' = "rejected" => coreVars' = coreVars)
       /\ ((cmdPhase = "recording" /\ cmdPhase' = "submitted") =>
              inFlightRetainsBuffer' = recordingRetainsBuffer)]_vars
=============================================================================
