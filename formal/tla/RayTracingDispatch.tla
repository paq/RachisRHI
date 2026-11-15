------------------------- MODULE RayTracingDispatch -------------------------
(***************************************************************************)
(* Direct ray-tracing dispatch validation (complete command state,         *)
(* non-zero bounded aggregate invocation product, aligned ray-generation   *)
(* and optional miss shader-binding-table ranges) and buffer retention     *)
(* across logical destruction and queue submission.                        *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12RayTracingPipeline.cpp (dispatchRays,                *)
(*     validateRayTracingDispatchState, prepareRayTracingShaderTables)     *)
(*   - src/D3D12/D3D12Resource.cpp (shaderBindingTable buffer usage,       *)
(*     bufferBindingSource)                                                *)
(*   - src/D3D12/D3D12Queue.cpp (command/in-flight resource retention)     *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Two shader-table buffers ("rayGen" with exact size = stride and       *)
(*   optional "miss" with size >= stride), 2D dispatch extents {0,1,2,3}   *)
(*   against MaxInvocations = 4, and aligned/unaligned offsets and strides *)
(*   cover failure-atomic invalid dispatches and multi-buffer retention.   *)
(*   GPU ray traversal and shader execution are trusted to the driver.     *)
(***************************************************************************)

EXTENDS Naturals

Tables == {"rayGen", "miss"}
BufferSize == 128
HandleSize == 32
RecordAlignment == 32
BaseAlignment == 64
MaxStride == 64
MaxInvocations == 4
MaxWork == 2
Offsets == {0, 32, 64}
Sizes == {0, 32, 64}
Strides == {0, 16, 32, 64}
Extents == {0, 2, 3}
Operations == {"none", "configure", "dispatch", "destroy", "submit", "complete", "abandon"}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  pipelineBound,    \* Whether a ray-tracing pipeline is bound
  bindingsComplete, \* Whether all required binding sets are bound
  rendering,        \* Whether an active render pass blocks ray dispatch
  tableLogical,     \* Public BufferHandle liveness per shader-table buffer
  tablePhysical,    \* Native buffer storage liveness per shader-table buffer
  recordingTables,  \* Shader-table buffers retained by open recording
  inFlightTables,   \* Shader-table buffers retained by in-flight submission
  inFlight,         \* Whether the command buffer is currently in flight
  work,             \* Count of accepted dispatches in the open recording
  lastOp,           \* Most recent operation kind
  lastOutcome       \* Outcome of the most recent operation

coreVars == <<
  pipelineBound, bindingsComplete, rendering,
  tableLogical, tablePhysical,
  recordingTables, inFlightTables,
  inFlight, work
>>

vars == <<
  pipelineBound, bindingsComplete, rendering,
  tableLogical, tablePhysical,
  recordingTables, inFlightTables,
  inFlight, work, lastOp, lastOutcome
>>

Retained(table) ==
  table \in recordingTables \/ table \in inFlightTables

CommandReady ==
  /\ pipelineBound
  /\ bindingsComplete
  /\ ~rendering

RayGenTableValid(hasUsage, offset, size, stride) ==
  /\ tableLogical["rayGen"]
  /\ tablePhysical["rayGen"]
  /\ hasUsage
  /\ offset % BaseAlignment = 0
  /\ offset <= BufferSize
  /\ size > 0
  /\ size <= BufferSize - offset
  /\ size = stride
  /\ stride >= HandleSize
  /\ stride <= MaxStride
  /\ stride % RecordAlignment = 0

MissTableValid(includeMiss, size, stride) ==
  IF ~includeMiss
  THEN TRUE
  ELSE /\ tableLogical["miss"]
       /\ tablePhysical["miss"]
       /\ size >= stride
       /\ size <= BufferSize
       /\ stride >= HandleSize
       /\ stride <= MaxStride
       /\ stride % RecordAlignment = 0

DimensionsValid(width, height) ==
  /\ width > 0
  /\ height > 0
  /\ width * height <= MaxInvocations

DispatchValid(hasUsage, offset, rgSize, rgStride,
              includeMiss, missSize, missStride, width, height) ==
  /\ ~inFlight
  /\ CommandReady
  /\ DimensionsValid(width, height)
  /\ RayGenTableValid(hasUsage, offset, rgSize, rgStride)
  /\ MissTableValid(includeMiss, missSize, missStride)

Init ==
  /\ pipelineBound = FALSE
  /\ bindingsComplete = FALSE
  /\ rendering = FALSE
  /\ tableLogical = [t \in Tables |-> TRUE]
  /\ tablePhysical = [t \in Tables |-> TRUE]
  /\ recordingTables = {}
  /\ inFlightTables = {}
  /\ inFlight = FALSE
  /\ work = 0
  /\ lastOp = "none"
  /\ lastOutcome = "none"

ConfigureState(pipeline, bindings, insideRendering) ==
  /\ ~inFlight
  /\ pipelineBound' = pipeline
  /\ bindingsComplete' = bindings
  /\ rendering' = insideRendering
  /\ UNCHANGED <<
       tableLogical, tablePhysical,
       recordingTables, inFlightTables,
       inFlight, work
     >>
  /\ lastOp' = "configure"
  /\ lastOutcome' = "accepted"

AttemptDispatch(hasUsage, offset, rgSize, rgStride,
                includeMiss, missSize, missStride, width, height) ==
  LET usedTables == IF includeMiss THEN {"rayGen", "miss"} ELSE {"rayGen"} IN
  /\ ~inFlight
  /\ work < MaxWork
  /\ IF DispatchValid(hasUsage, offset, rgSize, rgStride,
                      includeMiss, missSize, missStride, width, height)
     THEN /\ recordingTables' = recordingTables \cup usedTables
          /\ work' = work + 1
          /\ UNCHANGED <<
               pipelineBound, bindingsComplete, rendering,
               tableLogical, tablePhysical, inFlightTables, inFlight
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "dispatch"

DestroyTable(table) ==
  /\ tableLogical[table]
  /\ tableLogical' = [tableLogical EXCEPT ![table] = FALSE]
  /\ tablePhysical' = [tablePhysical EXCEPT ![table] = Retained(table)]
  /\ UNCHANGED <<
       pipelineBound, bindingsComplete, rendering,
       recordingTables, inFlightTables,
       inFlight, work
     >>
  /\ lastOp' = "destroy"
  /\ lastOutcome' = "accepted"

Submit ==
  /\ ~inFlight
  /\ ~rendering
  /\ work > 0
  /\ pipelineBound' = FALSE
  /\ bindingsComplete' = FALSE
  /\ rendering' = FALSE
  /\ recordingTables' = {}
  /\ inFlightTables' = recordingTables
  /\ inFlight' = TRUE
  /\ work' = 0
  /\ UNCHANGED <<tableLogical, tablePhysical>>
  /\ lastOp' = "submit"
  /\ lastOutcome' = "accepted"

Complete ==
  /\ inFlight
  /\ tablePhysical' = [t \in Tables |-> tableLogical[t]]
  /\ inFlightTables' = {}
  /\ inFlight' = FALSE
  /\ UNCHANGED <<
       pipelineBound, bindingsComplete, rendering,
       tableLogical, recordingTables, work
     >>
  /\ lastOp' = "complete"
  /\ lastOutcome' = "accepted"

Abandon ==
  /\ ~inFlight
  /\ tablePhysical' = [t \in Tables |-> tableLogical[t]]
  /\ pipelineBound' = FALSE
  /\ bindingsComplete' = FALSE
  /\ rendering' = FALSE
  /\ recordingTables' = {}
  /\ work' = 0
  /\ UNCHANGED <<tableLogical, inFlightTables, inFlight>>
  /\ lastOp' = "abandon"
  /\ lastOutcome' = "accepted"

Next ==
  \/ \E p, b, r \in BOOLEAN : ConfigureState(p, b, r)
  \/ \E u \in BOOLEAN, off \in Offsets, rgs \in Sizes, rgst \in Strides,
        w, h \in Extents :
       AttemptDispatch(u, off, rgs, rgst, FALSE, 0, 0, w, h)
  \/ \E ms \in Sizes, mst \in Strides, w, h \in Extents :
       AttemptDispatch(TRUE, 0, 32, 32, TRUE, ms, mst, w, h)
  \/ \E t \in Tables : DestroyTable(t)
  \/ Submit
  \/ Complete
  \/ Abandon

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ pipelineBound \in BOOLEAN
  /\ bindingsComplete \in BOOLEAN
  /\ rendering \in BOOLEAN
  /\ tableLogical \in [Tables -> BOOLEAN]
  /\ tablePhysical \in [Tables -> BOOLEAN]
  /\ recordingTables \in SUBSET Tables
  /\ inFlightTables \in SUBSET Tables
  /\ inFlight \in BOOLEAN
  /\ work \in 0..MaxWork
  /\ lastOp \in Operations
  /\ lastOutcome \in Outcomes

TablePhysicalLifetimeExact ==
  \A t \in Tables :
    tablePhysical[t] = (tableLogical[t] \/ Retained(t))

WorkRequiresRayGenRetention ==
  work > 0 <=> "rayGen" \in recordingTables

SubmittedReferencesSeparated ==
  inFlight =>
    /\ recordingTables = {}
    /\ work = 0
    /\ "rayGen" \in inFlightTables

Safety ==
  /\ TablePhysicalLifetimeExact
  /\ WorkRequiresRayGenRetention
  /\ SubmittedReferencesSeparated

StepSafety ==
  [][
    /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
    /\ (work' > work =>
          /\ lastOp' = "dispatch"
          /\ lastOutcome' = "accepted"
          /\ CommandReady
          /\ ~inFlight
          /\ tableLogical["rayGen"]
          /\ "rayGen" \in recordingTables'
          /\ work' = work + 1)
    /\ (\A t \in Tables :
          t \in (recordingTables' \ recordingTables) =>
            /\ work' = work + 1
            /\ tableLogical[t]
            /\ tablePhysical'[t])
  ]_vars

=============================================================================
