----------------------- MODULE IndirectDispatchBatch ------------------------
(***************************************************************************)
(* Fixed-count and GPU-counted indirect compute dispatch batches, span and *)
(* stride validation, GPU count clamping, failure atomicity, and buffer    *)
(* lifetime retention in `src/D3D12/D3D12CommandRecording.cpp`             *)
(* (`PipelineManager::dispatchIndirect` and shared indirect-argument       *)
(* validation), `src/D3D12/D3D12Pipeline.cpp` (stride-specific dispatch    *)
(* command signatures), and `src/D3D12/D3D12Queue.cpp`                     *)
(* (`include/Rachis/Device.h`, `include/Rachis/Rendering.h`).              *)
(*                                                                         *)
(* Models packed and padded 12-byte `DispatchIndirectArguments` spans in   *)
(* resource 0 plus an optional 4-byte GPU count in resource 1 (or aliased  *)
(* into resource 0), compute readiness gating, zero-count no-op semantics, *)
(* clamping in both `gpuCount < maxDispatchCount` and                      *)
(* `maxDispatchCount < gpuCount` directions, atomic rejection, and exact   *)
(* physical retention across logical destruction, submission, completion,  *)
(* and command abandonment.                                                *)
(*                                                                         *)
(* Abstracted: shader execution and partial publication failure prefixes   *)
(* (covered by `IndirectExecutionPublication.tla`).                        *)
(***************************************************************************)
EXTENDS Naturals

Resources == {0, 1}
Argument == 0
Count == 1
ArgumentSize == 52
CountSize == 8
RecordSize == 12
Offsets == {0, 2, 28, 52}
CountOffsets == {0, 2, 8}
Strides == {0, 8, 12, 16}
ResolvedStrides == {8, 12, 16}
DispatchCounts == {0, 2, 4}
CountValues == {1, 3}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    logicalAlive,               \* public handle liveness in [Resources -> BOOLEAN]
    physicalAlive,              \* native buffer liveness in [Resources -> BOOLEAN]
    recordingReferences,        \* resources retained by active recording
    inFlightReferences,         \* resources retained by in-flight submission
    inFlight,                   \* TRUE while command buffer is in flight
    computeReady,               \* TRUE when compute pipeline and bindings are ready
    lastOutcome,                \* outcome of latest step in Outcomes
    lastUsedCount,              \* whether latest attempt supplied a count buffer
    lastCountAliasesArgument,   \* whether latest count buffer aliased Argument
    lastStride,                 \* resolved byte stride of latest attempt
    lastGpuCount,               \* GPU-stored count observed by latest attempt
    lastMaxDispatchCount,       \* declared maxDispatchCount of latest attempt
    lastExecutedCount           \* clamped executed dispatch count of latest attempt

coreVars == <<
    logicalAlive, physicalAlive, recordingReferences,
    inFlightReferences, inFlight, computeReady
>>

vars == <<
    logicalAlive, physicalAlive, recordingReferences, inFlightReferences,
    inFlight, computeReady, lastOutcome, lastUsedCount,
    lastCountAliasesArgument, lastStride, lastGpuCount,
    lastMaxDispatchCount, lastExecutedCount
>>

Referenced(resource) ==
    resource \in recordingReferences \/ resource \in inFlightReferences

ResolvedStride(stride) ==
    IF stride = 0 THEN RecordSize ELSE stride

ArgumentBytes(stride, maxDispatchCount) ==
    IF maxDispatchCount = 0
    THEN 0
    ELSE (maxDispatchCount - 1) * ResolvedStride(stride) + RecordSize

RequestValid(
    argumentOffset, stride, useCount,
    countAliasesArgument, countOffset, maxDispatchCount
) ==
    /\ computeReady
    /\ logicalAlive[Argument]
    /\ argumentOffset % 4 = 0
    /\ argumentOffset <= ArgumentSize
    /\ ResolvedStride(stride) >= RecordSize
    /\ ResolvedStride(stride) % 4 = 0
    /\ (maxDispatchCount = 0
        \/ argumentOffset + ArgumentBytes(stride, maxDispatchCount) <= ArgumentSize)
    /\ IF useCount
       THEN /\ IF countAliasesArgument THEN logicalAlive[Argument] ELSE logicalAlive[Count]
            /\ countOffset % 4 = 0
            /\ countOffset + 4 <= (IF countAliasesArgument THEN ArgumentSize ELSE CountSize)
       ELSE countOffset = 0

ExecutedCount(useCount, gpuCount, maxDispatchCount) ==
    IF useCount /\ gpuCount < maxDispatchCount
    THEN gpuCount
    ELSE maxDispatchCount

Init ==
    /\ logicalAlive = [r \in Resources |-> TRUE]
    /\ physicalAlive = [r \in Resources |-> TRUE]
    /\ recordingReferences = {}
    /\ inFlightReferences = {}
    /\ inFlight = FALSE
    /\ computeReady = TRUE
    /\ lastOutcome = "none"
    /\ lastUsedCount = FALSE
    /\ lastCountAliasesArgument = FALSE
    /\ lastStride = RecordSize
    /\ lastGpuCount = 1
    /\ lastMaxDispatchCount = 0
    /\ lastExecutedCount = 0

configureCompute(ready) ==
    /\ ~inFlight
    /\ computeReady # ready
    /\ computeReady' = ready
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalAlive, physicalAlive, recordingReferences, inFlightReferences,
        inFlight, lastUsedCount, lastCountAliasesArgument, lastStride,
        lastGpuCount, lastMaxDispatchCount, lastExecutedCount
       >>

attemptBatch(
    argumentOffset, stride, useCount,
    countAliasesArgument, countOffset, maxDispatchCount, gpuCount
) ==
    LET accepted ==
            RequestValid(
                argumentOffset, stride, useCount,
                countAliasesArgument, countOffset, maxDispatchCount
            )
        retained ==
            IF maxDispatchCount > 0
            THEN IF useCount /\ ~countAliasesArgument
                 THEN recordingReferences \cup {Argument, Count}
                 ELSE recordingReferences \cup {Argument}
            ELSE recordingReferences
    IN /\ ~inFlight
       /\ (IF countAliasesArgument THEN useCount ELSE TRUE)
       /\ (IF useCount THEN TRUE ELSE countOffset \in {0, 8} /\ gpuCount = 1)
       /\ IF accepted
          THEN /\ recordingReferences' = retained
               /\ lastUsedCount' = useCount
               /\ lastCountAliasesArgument' = countAliasesArgument
               /\ lastStride' = ResolvedStride(stride)
               /\ lastGpuCount' = gpuCount
               /\ lastMaxDispatchCount' = maxDispatchCount
               /\ lastExecutedCount' =
                    ExecutedCount(useCount, gpuCount, maxDispatchCount)
               /\ UNCHANGED <<
                    logicalAlive, physicalAlive, inFlightReferences,
                    inFlight, computeReady
                  >>
          ELSE UNCHANGED <<
                logicalAlive, physicalAlive, recordingReferences,
                inFlightReferences, inFlight, computeReady, lastUsedCount,
                lastCountAliasesArgument, lastStride, lastGpuCount,
                lastMaxDispatchCount, lastExecutedCount
               >>
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

destroy(resource) ==
    /\ logicalAlive[resource]
    /\ logicalAlive' = [logicalAlive EXCEPT ![resource] = FALSE]
    /\ physicalAlive' = [physicalAlive EXCEPT ![resource] = Referenced(resource)]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        recordingReferences, inFlightReferences, inFlight, computeReady,
        lastUsedCount, lastCountAliasesArgument, lastStride, lastGpuCount,
        lastMaxDispatchCount, lastExecutedCount
       >>

submit ==
    /\ ~inFlight
    /\ recordingReferences # {}
    /\ recordingReferences' = {}
    /\ inFlightReferences' = recordingReferences
    /\ inFlight' = TRUE
    /\ computeReady' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalAlive, physicalAlive, lastUsedCount, lastCountAliasesArgument,
        lastStride, lastGpuCount, lastMaxDispatchCount, lastExecutedCount
       >>

complete ==
    /\ inFlight
    /\ physicalAlive' =
        [r \in Resources |-> logicalAlive[r] \/ r \in recordingReferences]
    /\ inFlightReferences' = {}
    /\ inFlight' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalAlive, recordingReferences, computeReady, lastUsedCount,
        lastCountAliasesArgument, lastStride, lastGpuCount,
        lastMaxDispatchCount, lastExecutedCount
       >>

abandon ==
    /\ ~inFlight
    /\ recordingReferences # {}
    /\ physicalAlive' =
        [r \in Resources |-> logicalAlive[r] \/ r \in inFlightReferences]
    /\ recordingReferences' = {}
    /\ computeReady' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalAlive, inFlightReferences, inFlight, lastUsedCount,
        lastCountAliasesArgument, lastStride, lastGpuCount,
        lastMaxDispatchCount, lastExecutedCount
       >>

Next ==
    \/ \E ready \in BOOLEAN : configureCompute(ready)
    \/ /\ ~inFlight
       /\ \/ \E aOff \in Offsets, str \in Strides,
                cOff \in {0, 8}, maxCnt \in DispatchCounts :
                  attemptBatch(aOff, str, FALSE, FALSE, cOff, maxCnt, 1)
          \/ \E aOff \in Offsets, str \in Strides, alias \in BOOLEAN,
                cOff \in CountOffsets, maxCnt \in DispatchCounts,
                gpuCnt \in CountValues :
                  attemptBatch(aOff, str, TRUE, alias, cOff, maxCnt, gpuCnt)
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
    /\ computeReady \in BOOLEAN
    /\ lastOutcome \in Outcomes
    /\ lastUsedCount \in BOOLEAN
    /\ lastCountAliasesArgument \in BOOLEAN
    /\ lastStride \in ResolvedStrides
    /\ lastGpuCount \in CountValues
    /\ lastMaxDispatchCount \in DispatchCounts
    /\ lastExecutedCount \in 0..4

AcceptedStrideMatchesLayout ==
    lastOutcome = "accepted" =>
        /\ lastStride >= RecordSize
        /\ lastStride % 4 = 0

AcceptedBuffersRetained ==
    (lastOutcome = "accepted" /\ lastMaxDispatchCount > 0) =>
        /\ Argument \in recordingReferences
        /\ (lastUsedCount /\ ~lastCountAliasesArgument) => Count \in recordingReferences

GpuCountIsClamped ==
    /\ lastExecutedCount <= lastMaxDispatchCount
    /\ (lastOutcome = "accepted" /\ lastUsedCount) =>
        /\ lastExecutedCount <= lastGpuCount
        /\ lastExecutedCount =
            IF lastGpuCount < lastMaxDispatchCount
            THEN lastGpuCount
            ELSE lastMaxDispatchCount

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
        /\ ((lastOutcome' = "accepted" /\ lastMaxDispatchCount' = 0) =>
                /\ coreVars' = coreVars
                /\ lastExecutedCount' = 0)
        /\ (lastOutcome' = "accepted" => computeReady /\ logicalAlive[Argument])
    ]_vars

=============================================================================
