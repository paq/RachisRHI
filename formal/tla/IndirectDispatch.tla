-------------------------- MODULE IndirectDispatch --------------------------
(***************************************************************************)
(* Single-record indirect compute dispatch validation, failure atomicity,  *)
(* and indirect-buffer lifetime retention in `src/Device.cpp` (the         *)
(* single-record `Device::dispatchIndirect` overload),                     *)
(* `src/D3D12/D3D12CommandRecording.cpp`                                   *)
(* (`PipelineManager::dispatchIndirect` and shared indirect-argument       *)
(* validation), `src/D3D12/D3D12Pipeline.cpp` (dispatch command            *)
(* signatures), and `src/D3D12/D3D12Queue.cpp` (`include/Rachis/Device.h`, *)
(* `include/Rachis/Rendering.h`).                                          *)
(*                                                                         *)
(* Models portable 12-byte `DispatchIndirectArguments` validation across   *)
(* aligned, misaligned, exact-end, and out-of-bounds offsets inside a      *)
(* 16-byte buffer, compute-state readiness (pipeline bound, complete       *)
(* bindings, outside rendering), atomic rejection of invalid requests, and *)
(* exact physical retention of the indirect buffer across logical          *)
(* destruction through GPU completion or command abandonment.              *)
(*                                                                         *)
(* Abstracted: GPU-authored workgroup counts, multi-record batches         *)
(* (covered by `IndirectDispatchBatch.tla`), and publication failure       *)
(* prefixes (covered by `IndirectExecutionPublication.tla`).               *)
(***************************************************************************)
EXTENDS Naturals

BufferSize == 16
ArgumentSize == 12
Offsets == {0, 2, 4, 8, 16}
MaxDispatches == 2
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    logicalAlive,       \* TRUE while public BufferHandle is live
    physicalAlive,      \* TRUE while native indirect buffer storage lives
    recordingReference, \* TRUE while active recording retains the buffer
    inFlightReference,  \* TRUE while submitted work retains the buffer
    inFlight,           \* TRUE while command buffer is in flight on the queue
    pipelineBound,      \* TRUE when a compute pipeline is bound
    bindingsComplete,   \* TRUE when required compute binding sets are bound
    rendering,          \* TRUE when inside a graphics render pass
    dispatches,         \* count of accepted dispatches in the active recording
    lastOffset,         \* byte offset supplied to the latest dispatch attempt
    lastOutcome         \* outcome of the latest step in Outcomes

coreVars == <<
    logicalAlive, physicalAlive, recordingReference, inFlightReference,
    inFlight, pipelineBound, bindingsComplete, rendering, dispatches
>>

vars == <<
    logicalAlive, physicalAlive, recordingReference, inFlightReference,
    inFlight, pipelineBound, bindingsComplete, rendering, dispatches,
    lastOffset, lastOutcome
>>

Referenced == recordingReference \/ inFlightReference

ComputeReady == pipelineBound /\ bindingsComplete /\ ~rendering

RequestValid(offset) ==
    /\ ComputeReady
    /\ logicalAlive
    /\ offset % 4 = 0
    /\ offset + ArgumentSize <= BufferSize

Init ==
    /\ logicalAlive = TRUE
    /\ physicalAlive = TRUE
    /\ recordingReference = FALSE
    /\ inFlightReference = FALSE
    /\ inFlight = FALSE
    /\ pipelineBound = FALSE
    /\ bindingsComplete = FALSE
    /\ rendering = FALSE
    /\ dispatches = 0
    /\ lastOffset = 0
    /\ lastOutcome = "none"

configure(pipeline, bindings, insideRendering) ==
    /\ ~inFlight
    /\ pipelineBound' = pipeline
    /\ bindingsComplete' = bindings
    /\ rendering' = insideRendering
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalAlive, physicalAlive, recordingReference,
        inFlightReference, inFlight, dispatches, lastOffset
       >>

attemptDispatch(offset) ==
    LET accepted == RequestValid(offset)
    IN /\ ~inFlight
       /\ dispatches < MaxDispatches
       /\ IF accepted
          THEN /\ recordingReference' = TRUE
               /\ dispatches' = dispatches + 1
               /\ UNCHANGED <<
                    logicalAlive, physicalAlive, inFlightReference,
                    inFlight, pipelineBound, bindingsComplete, rendering
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOffset' = offset
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

destroy ==
    /\ logicalAlive
    /\ logicalAlive' = FALSE
    /\ physicalAlive' = Referenced
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        recordingReference, inFlightReference, inFlight,
        pipelineBound, bindingsComplete, rendering, dispatches, lastOffset
       >>

submit ==
    /\ ~inFlight
    /\ ~rendering
    /\ recordingReference
    /\ recordingReference' = FALSE
    /\ inFlightReference' = TRUE
    /\ inFlight' = TRUE
    /\ pipelineBound' = FALSE
    /\ bindingsComplete' = FALSE
    /\ dispatches' = 0
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<logicalAlive, physicalAlive, rendering, lastOffset>>

complete ==
    /\ inFlight
    /\ physicalAlive' = (logicalAlive \/ recordingReference)
    /\ inFlightReference' = FALSE
    /\ inFlight' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        logicalAlive, recordingReference, pipelineBound,
        bindingsComplete, rendering, dispatches, lastOffset
       >>

abandon ==
    /\ ~inFlight
    /\ recordingReference
    /\ physicalAlive' = (logicalAlive \/ inFlightReference)
    /\ recordingReference' = FALSE
    /\ pipelineBound' = FALSE
    /\ bindingsComplete' = FALSE
    /\ rendering' = FALSE
    /\ dispatches' = 0
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<logicalAlive, inFlightReference, inFlight, lastOffset>>

Next ==
    \/ \E p \in BOOLEAN, b \in BOOLEAN, r \in BOOLEAN : configure(p, b, r)
    \/ \E off \in Offsets : attemptDispatch(off)
    \/ destroy
    \/ submit
    \/ complete
    \/ abandon

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ logicalAlive \in BOOLEAN
    /\ physicalAlive \in BOOLEAN
    /\ recordingReference \in BOOLEAN
    /\ inFlightReference \in BOOLEAN
    /\ inFlight \in BOOLEAN
    /\ pipelineBound \in BOOLEAN
    /\ bindingsComplete \in BOOLEAN
    /\ rendering \in BOOLEAN
    /\ dispatches \in 0..MaxDispatches
    /\ lastOffset \in Offsets
    /\ lastOutcome \in Outcomes

AcceptedOffsetInBounds ==
    lastOutcome = "accepted" =>
        /\ lastOffset % 4 = 0
        /\ lastOffset + ArgumentSize <= BufferSize

PhysicalLifetimeExact ==
    physicalAlive = (logicalAlive \/ Referenced)

DispatchesMatchRecordingState ==
    /\ (dispatches > 0) = recordingReference
    /\ inFlight => ~recordingReference /\ dispatches = 0

Safety ==
    /\ AcceptedOffsetInBounds
    /\ PhysicalLifetimeExact
    /\ DispatchesMatchRecordingState

StepSafety ==
    [][
        /\ (lastOutcome' # "none" =>
                ((lastOutcome' = "accepted") = RequestValid(lastOffset')))
        /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
    ]_vars

=============================================================================
