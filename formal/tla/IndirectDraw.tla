---------------------------- MODULE IndirectDraw ----------------------------
(***************************************************************************)
(* Non-indexed and indexed indirect draw validation, GPU count clamping,   *)
(* zero-count no-op semantics, failure atomicity, and buffer retention in  *)
(* `src/D3D12/D3D12CommandRecording.cpp`                                   *)
(* (`PipelineManager::executeIndirect` and shared indirect-argument        *)
(* validation), `src/D3D12/D3D12Pipeline.cpp` (draw command signatures),   *)
(* and `src/D3D12/D3D12Queue.cpp` (`include/Rachis/Device.h`,              *)
(* `include/Rachis/Rendering.h`).                                          *)
(*                                                                         *)
(* Models 16-byte `DrawIndirectArguments` and 20-byte                      *)
(* `DrawIndexedIndirectArguments` spans in resource 0 with optional 4-byte *)
(* GPU counts in resource 1 (or aliased into resource 0). Finite domains   *)
(* (`CountValues = {1, 3}`, `DrawCounts = {0, 1, 2, 4}`) exercise both     *)
(* `gpuCount < maxDrawCount` and `maxDrawCount < gpuCount` clamping        *)
(* branches, packed and padded strides, misaligned and out-of-bounds       *)
(* spans, atomic rejection, and exact physical lifetime retention.         *)
(*                                                                         *)
(* Abstracted: graphics pipeline state (covered by                         *)
(* `GraphicsCommandState.tla`) and partial dependency rollback (covered by *)
(* `IndirectExecutionPublication.tla`).                                    *)
(***************************************************************************)
EXTENDS Naturals

Resources == {0, 1}
Argument == 0
Count == 1
ArgumentSize == 84
CountSize == 8
Offsets == {0, 2, 84}
CountOffsets == {0, 2, 8}
Strides == {0, 12, 20, 32}
ResolvedStrides == {12, 16, 20, 32}
DrawCounts == {0, 2, 4}
CountValues == {1, 3}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    logicalAlive,               \* public handle liveness in [Resources -> BOOLEAN]
    physicalAlive,              \* native buffer liveness in [Resources -> BOOLEAN]
    recordingReferences,        \* resources retained by active recording
    inFlightReferences,         \* resources retained by in-flight submission
    inFlight,                   \* TRUE while command buffer is in flight
    lastOutcome,                \* outcome of latest step in Outcomes
    lastIndexed,                \* whether latest draw attempt was indexed
    lastUsedCount,              \* whether latest draw attempt supplied a count buffer
    lastCountAliasesArgument,   \* whether latest count buffer aliased Argument
    lastStride,                 \* resolved byte stride of latest attempt
    lastGpuCount,               \* GPU-stored draw count observed by latest attempt
    lastMaxDrawCount,           \* declared maxDrawCount of latest attempt
    lastExecutedCount           \* clamped executed draw count of latest attempt

coreVars == <<
    logicalAlive, physicalAlive, recordingReferences,
    inFlightReferences, inFlight
>>

vars == <<
    logicalAlive, physicalAlive, recordingReferences, inFlightReferences,
    inFlight, lastOutcome, lastIndexed, lastUsedCount,
    lastCountAliasesArgument, lastStride, lastGpuCount,
    lastMaxDrawCount, lastExecutedCount
>>

Referenced(resource) ==
    resource \in recordingReferences \/ resource \in inFlightReferences

RecordSize(indexed) ==
    IF indexed THEN 20 ELSE 16

ResolvedStride(indexed, stride) ==
    IF stride = 0 THEN RecordSize(indexed) ELSE stride

ArgumentBytes(indexed, stride, maxDrawCount) ==
    IF maxDrawCount = 0
    THEN 0
    ELSE (maxDrawCount - 1) * ResolvedStride(indexed, stride) + RecordSize(indexed)

RequestValid(
    indexed, argumentOffset, stride, useCount,
    countAliasesArgument, countOffset, maxDrawCount
) ==
    /\ logicalAlive[Argument]
    /\ argumentOffset % 4 = 0
    /\ argumentOffset <= ArgumentSize
    /\ ResolvedStride(indexed, stride) >= RecordSize(indexed)
    /\ ResolvedStride(indexed, stride) % 4 = 0
    /\ (maxDrawCount = 0
        \/ argumentOffset + ArgumentBytes(indexed, stride, maxDrawCount) <= ArgumentSize)
    /\ IF useCount
       THEN /\ IF countAliasesArgument THEN logicalAlive[Argument] ELSE logicalAlive[Count]
            /\ countOffset % 4 = 0
            /\ countOffset + 4 <= (IF countAliasesArgument THEN ArgumentSize ELSE CountSize)
       ELSE countOffset = 0

ExecutedCount(useCount, gpuCount, maxDrawCount) ==
    IF useCount /\ gpuCount < maxDrawCount
    THEN gpuCount
    ELSE maxDrawCount

Init ==
    /\ logicalAlive = [r \in Resources |-> TRUE]
    /\ physicalAlive = [r \in Resources |-> TRUE]
    /\ recordingReferences = {}
    /\ inFlightReferences = {}
    /\ inFlight = FALSE
    /\ lastOutcome = "none"
    /\ lastIndexed = FALSE
    /\ lastUsedCount = FALSE
    /\ lastCountAliasesArgument = FALSE
    /\ lastStride = 16
    /\ lastGpuCount = 1
    /\ lastMaxDrawCount = 0
    /\ lastExecutedCount = 0

attemptDraw(
    indexed, argumentOffset, stride, useCount,
    countAliasesArgument, countOffset, maxDrawCount, gpuCount
) ==
    LET accepted ==
            RequestValid(
                indexed, argumentOffset, stride, useCount,
                countAliasesArgument, countOffset, maxDrawCount
            )
        retained ==
            IF maxDrawCount > 0
            THEN IF useCount /\ ~countAliasesArgument
                 THEN recordingReferences \cup {Argument, Count}
                 ELSE recordingReferences \cup {Argument}
            ELSE recordingReferences
    IN /\ ~inFlight
       /\ (IF countAliasesArgument THEN useCount ELSE TRUE)
       /\ (IF useCount THEN TRUE ELSE countOffset \in {0, 8} /\ gpuCount = 1)
       /\ IF accepted
          THEN /\ recordingReferences' = retained
               /\ lastIndexed' = indexed
               /\ lastUsedCount' = useCount
               /\ lastCountAliasesArgument' = countAliasesArgument
               /\ lastStride' = ResolvedStride(indexed, stride)
               /\ lastGpuCount' = gpuCount
               /\ lastMaxDrawCount' = maxDrawCount
               /\ lastExecutedCount' =
                    ExecutedCount(useCount, gpuCount, maxDrawCount)
               /\ UNCHANGED <<logicalAlive, physicalAlive, inFlightReferences, inFlight>>
          ELSE UNCHANGED <<
                logicalAlive, physicalAlive, recordingReferences,
                inFlightReferences, inFlight, lastIndexed, lastUsedCount,
                lastCountAliasesArgument, lastStride, lastGpuCount,
                lastMaxDrawCount, lastExecutedCount
               >>
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

destroy(resource) ==
    /\ logicalAlive[resource]
    /\ logicalAlive' = [logicalAlive EXCEPT ![resource] = FALSE]
    /\ physicalAlive' = [physicalAlive EXCEPT ![resource] = Referenced(resource)]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        recordingReferences, inFlightReferences, inFlight, lastIndexed,
        lastUsedCount, lastCountAliasesArgument, lastStride, lastGpuCount,
        lastMaxDrawCount, lastExecutedCount
       >>

submit ==
    /\ ~inFlight
    /\ recordingReferences # {}
    /\ recordingReferences' = {}
    /\ inFlightReferences' = recordingReferences
    /\ inFlight' = TRUE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalAlive, physicalAlive, lastIndexed, lastUsedCount,
        lastCountAliasesArgument, lastStride, lastGpuCount,
        lastMaxDrawCount, lastExecutedCount
       >>

complete ==
    /\ inFlight
    /\ physicalAlive' =
        [r \in Resources |-> logicalAlive[r] \/ r \in recordingReferences]
    /\ inFlightReferences' = {}
    /\ inFlight' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalAlive, recordingReferences, lastIndexed, lastUsedCount,
        lastCountAliasesArgument, lastStride, lastGpuCount,
        lastMaxDrawCount, lastExecutedCount
       >>

abandon ==
    /\ ~inFlight
    /\ recordingReferences # {}
    /\ physicalAlive' =
        [r \in Resources |-> logicalAlive[r] \/ r \in inFlightReferences]
    /\ recordingReferences' = {}
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalAlive, inFlightReferences, inFlight, lastIndexed,
        lastUsedCount, lastCountAliasesArgument, lastStride, lastGpuCount,
        lastMaxDrawCount, lastExecutedCount
       >>

Next ==
    \/ /\ ~inFlight
       /\ \/ \E idx \in BOOLEAN, aOff \in Offsets, str \in Strides,
                cOff \in {0, 8}, maxCnt \in DrawCounts :
                  attemptDraw(idx, aOff, str, FALSE, FALSE, cOff, maxCnt, 1)
          \/ \E idx \in BOOLEAN, aOff \in Offsets, str \in Strides,
                alias \in BOOLEAN, cOff \in CountOffsets,
                maxCnt \in DrawCounts, gpuCnt \in CountValues :
                  attemptDraw(idx, aOff, str, TRUE, alias, cOff, maxCnt, gpuCnt)
    \/ \E r \in Resources : destroy(r)
    \/ submit
    \/ complete
    \/ abandon

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ logicalAlive \in [Resources -> BOOLEAN]
    /\ physicalAlive \in [Resources -> BOOLEAN]
    /\ recordingReferences \in SUBSET Resources
    /\ inFlightReferences \in SUBSET Resources
    /\ inFlight \in BOOLEAN
    /\ lastOutcome \in Outcomes
    /\ lastIndexed \in BOOLEAN
    /\ lastUsedCount \in BOOLEAN
    /\ lastCountAliasesArgument \in BOOLEAN
    /\ lastStride \in ResolvedStrides
    /\ lastGpuCount \in CountValues
    /\ lastMaxDrawCount \in DrawCounts
    /\ lastExecutedCount \in 0..4

AcceptedStrideMatchesLayout ==
    lastOutcome = "accepted" =>
        /\ lastStride >= RecordSize(lastIndexed)
        /\ lastStride % 4 = 0

AcceptedBuffersRetained ==
    (lastOutcome = "accepted" /\ lastMaxDrawCount > 0) =>
        /\ Argument \in recordingReferences
        /\ (lastUsedCount /\ ~lastCountAliasesArgument) => Count \in recordingReferences

GpuCountIsClamped ==
    /\ lastExecutedCount <= lastMaxDrawCount
    /\ (lastOutcome = "accepted" /\ lastUsedCount) =>
        /\ lastExecutedCount <= lastGpuCount
        /\ lastExecutedCount =
            IF lastGpuCount < lastMaxDrawCount
            THEN lastGpuCount
            ELSE lastMaxDrawCount

BufferPhysicalLifetimeExact ==
    \A r \in Resources :
        physicalAlive[r] = (logicalAlive[r] \/ Referenced(r))

SubmittedSeparated ==
    inFlight => recordingReferences = {}

Safety ==
    /\ AcceptedStrideMatchesLayout
    /\ AcceptedBuffersRetained
    /\ GpuCountIsClamped
    /\ BufferPhysicalLifetimeExact
    /\ SubmittedSeparated

StepSafety ==
    [][
        /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
        /\ ((lastOutcome' = "accepted" /\ lastMaxDrawCount' = 0) =>
                /\ coreVars' = coreVars
                /\ lastExecutedCount' = 0)
        /\ (lastOutcome' = "accepted" => logicalAlive[Argument])
    ]_vars

=============================================================================
