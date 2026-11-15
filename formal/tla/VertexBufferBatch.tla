-------------------------- MODULE VertexBufferBatch -------------------------
(***************************************************************************)
(* Formal safety model for contiguous multi-slot vertex buffer binding     *)
(* batches, pipeline declared-slot bounds checking, draw validation, and   *)
(* recording/submission buffer retention:                                  *)
(*   - include/Rachis/Device.h (bindVertexBuffers, draw)                   *)
(*   - src/D3D12/D3D12Pipeline.cpp (makeGraphicsPipelineRecord fixes the   *)
(*     declared vertexStrides)                                             *)
(*   - src/D3D12/D3D12CommandRecording.cpp (bindGraphicsPipeline clears    *)
(*     boundVertexBuffers; PipelineManager::bindVertexBuffers validates    *)
(*     the whole batch before IASetVertexBuffers; draw requires every      *)
(*     declared slot)                                                      *)
(*   - src/D3D12/D3D12Resource.cpp (ResourceManager::bufferBindingSource   *)
(*     live-buffer lookup)                                                 *)
(*   - src/D3D12/D3D12Queue.cpp (QueueManager::retainResource retains the  *)
(*     batch owner and residencies or rolls back)                          *)
(*   - src/Vulkan/VulkanPipelines.cpp and src/Vulkan/VulkanGraphics.cpp    *)
(*     implement the same declared-slot, validate-then-bind-and-retain     *)
(*     contract                                                            *)
(*                                                                         *)
(* src/Metal/MetalEncoding.mm applies the same validation and binds the    *)
(* batch together, but retains buffer owners before it checks the          *)
(* rendering scope, bound pipeline, and declared slots, so a rejected      *)
(* batch can leave extra owners retained; it is outside the                *)
(* rejection-preserves-leases claim.                                       *)
(*                                                                         *)
(* Models pipeline declaredSlotCount (0..2) enforced by bindVertexBuffers  *)
(* and draw, eliminates clone-variable tautologies, 1-step before* copies, *)
(* and state-mutating dummy actions, and separates logical buffer          *)
(* destruction from physical reclamation across abandon/submit/complete.   *)
(*                                                                         *)
(* Trusted boundary: input-assembler vertex fetch hardware once every      *)
(* declared slot holds a validated in-bounds vertex buffer.                *)
(***************************************************************************)
EXTENDS Naturals

Slots == {0, 1}
Resources == {0, 1}
DeclaredCounts == {0, 1, 2}
FirstSlotCandidates == {0, 1, 2}
BindingCountCandidates == {0, 1, 2, 3}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  renderingActive,   \* Whether a render pass is currently open.
  pipelineBound,     \* Whether a graphics pipeline is bound in the pass.
  declaredSlotCount, \* Number of vertex buffer slots declared by pipeline (0..2).
  boundSlots,        \* Per-slot bound status within the active pass.
  logicalAlive,      \* Per-buffer public handle liveness.
  physicalAlive,     \* Per-buffer native storage liveness.
  recordingLease,    \* Per-buffer command recording retention flag.
  inFlightLease,     \* Per-buffer GPU submission retention flag.
  recordedWork,      \* Whether the command buffer holds recorded work.
  inFlightWork,      \* Whether the command buffer is in flight on the GPU.
  lastFirstSlot,     \* First slot of latest accepted vertex-buffer batch.
  lastBindingCount,  \* Binding count of latest accepted vertex-buffer batch.
  lastOutcome        \* Outcome of latest batch or draw attempt.

coreVars == <<
  renderingActive,
  pipelineBound,
  declaredSlotCount,
  boundSlots,
  logicalAlive,
  physicalAlive,
  recordingLease,
  inFlightLease,
  recordedWork,
  inFlightWork
>>

vars == <<
  renderingActive,
  pipelineBound,
  declaredSlotCount,
  boundSlots,
  logicalAlive,
  physicalAlive,
  recordingLease,
  inFlightLease,
  recordedWork,
  inFlightWork,
  lastFirstSlot,
  lastBindingCount,
  lastOutcome
>>

DeclaredSlotSet(count) ==
  {s \in Slots : s < count}

BatchValid(firstSlot, bindingCount, firstValid, secondValid) ==
  /\ renderingActive
  /\ pipelineBound
  /\ ~inFlightWork
  /\ bindingCount \in {1, 2}
  /\ firstSlot < declaredSlotCount
  /\ firstSlot + bindingCount <= declaredSlotCount
  /\ firstValid
  /\ (bindingCount = 1 \/ secondValid)
  /\ logicalAlive[0]
  /\ (bindingCount = 1 \/ logicalAlive[1])

Init ==
  /\ renderingActive = FALSE
  /\ pipelineBound = FALSE
  /\ declaredSlotCount = 0
  /\ boundSlots = [s \in Slots |-> FALSE]
  /\ logicalAlive = [r \in Resources |-> TRUE]
  /\ physicalAlive = [r \in Resources |-> TRUE]
  /\ recordingLease = [r \in Resources |-> FALSE]
  /\ inFlightLease = [r \in Resources |-> FALSE]
  /\ recordedWork = FALSE
  /\ inFlightWork = FALSE
  /\ lastFirstSlot = 0
  /\ lastBindingCount = 0
  /\ lastOutcome = "none"

beginRendering ==
  /\ ~renderingActive
  /\ ~inFlightWork
  /\ renderingActive' = TRUE
  /\ pipelineBound' = FALSE
  /\ declaredSlotCount' = 0
  /\ boundSlots' = [s \in Slots |-> FALSE]
  /\ lastFirstSlot' = 0
  /\ lastBindingCount' = 0
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       recordingLease,
       inFlightLease,
       recordedWork,
       inFlightWork
     >>

bindPipeline(slotCount) ==
  /\ renderingActive
  /\ ~inFlightWork
  /\ slotCount \in DeclaredCounts
  /\ pipelineBound' = TRUE
  /\ declaredSlotCount' = slotCount
  /\ boundSlots' = [s \in Slots |-> FALSE]
  /\ lastFirstSlot' = 0
  /\ lastBindingCount' = 0
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       renderingActive,
       logicalAlive,
       physicalAlive,
       recordingLease,
       inFlightLease,
       recordedWork,
       inFlightWork
     >>

attemptBatch(firstSlot, bindingCount, firstValid, secondValid) ==
  /\ firstSlot \in FirstSlotCandidates
  /\ bindingCount \in BindingCountCandidates
  /\ firstValid \in BOOLEAN
  /\ secondValid \in BOOLEAN
  /\ IF BatchValid(firstSlot, bindingCount, firstValid, secondValid)
     THEN /\ boundSlots' =
               [s \in Slots |->
                  boundSlots[s]
                  \/ (s >= firstSlot /\ s < firstSlot + bindingCount)]
          /\ recordingLease' =
               [r \in Resources |->
                  recordingLease[r] \/ r = 0 \/ (bindingCount = 2 /\ r = 1)]
          /\ recordedWork' = TRUE
          /\ lastFirstSlot' = firstSlot
          /\ lastBindingCount' = bindingCount
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               renderingActive,
               pipelineBound,
               declaredSlotCount,
               logicalAlive,
               physicalAlive,
               inFlightLease,
               inFlightWork
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED <<coreVars, lastFirstSlot, lastBindingCount>>

attemptDraw ==
  /\ IF renderingActive
        /\ pipelineBound
        /\ ~inFlightWork
        /\ \A s \in DeclaredSlotSet(declaredSlotCount) : boundSlots[s]
     THEN /\ recordedWork' = TRUE
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               renderingActive,
               pipelineBound,
               declaredSlotCount,
               boundSlots,
               logicalAlive,
               physicalAlive,
               recordingLease,
               inFlightLease,
               inFlightWork,
               lastFirstSlot,
               lastBindingCount
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED <<coreVars, lastFirstSlot, lastBindingCount>>

endRendering ==
  /\ renderingActive
  /\ renderingActive' = FALSE
  /\ pipelineBound' = FALSE
  /\ declaredSlotCount' = 0
  /\ boundSlots' = [s \in Slots |-> FALSE]
  /\ lastFirstSlot' = 0
  /\ lastBindingCount' = 0
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       recordingLease,
       inFlightLease,
       recordedWork,
       inFlightWork
     >>

abandon ==
  /\ ~inFlightWork
  /\ (renderingActive \/ recordedWork)
  /\ renderingActive' = FALSE
  /\ pipelineBound' = FALSE
  /\ declaredSlotCount' = 0
  /\ boundSlots' = [s \in Slots |-> FALSE]
  /\ recordingLease' = [r \in Resources |-> FALSE]
  /\ recordedWork' = FALSE
  /\ lastFirstSlot' = 0
  /\ lastBindingCount' = 0
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<logicalAlive, physicalAlive, inFlightLease, inFlightWork>>

submit ==
  /\ ~renderingActive
  /\ recordedWork
  /\ ~inFlightWork
  /\ recordingLease' = [r \in Resources |-> FALSE]
  /\ inFlightLease' = recordingLease
  /\ recordedWork' = FALSE
  /\ inFlightWork' = TRUE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       renderingActive,
       pipelineBound,
       declaredSlotCount,
       boundSlots,
       logicalAlive,
       physicalAlive,
       lastFirstSlot,
       lastBindingCount
     >>

complete ==
  /\ inFlightWork
  /\ inFlightLease' = [r \in Resources |-> FALSE]
  /\ inFlightWork' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       renderingActive,
       pipelineBound,
       declaredSlotCount,
       boundSlots,
       logicalAlive,
       physicalAlive,
       recordingLease,
       recordedWork,
       lastFirstSlot,
       lastBindingCount
     >>

destroyResource(resource) ==
  /\ logicalAlive[resource]
  /\ logicalAlive' = [logicalAlive EXCEPT ![resource] = FALSE]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       renderingActive,
       pipelineBound,
       declaredSlotCount,
       boundSlots,
       physicalAlive,
       recordingLease,
       inFlightLease,
       recordedWork,
       inFlightWork,
       lastFirstSlot,
       lastBindingCount
     >>

collectResource(resource) ==
  /\ physicalAlive[resource]
  /\ ~logicalAlive[resource]
  /\ ~recordingLease[resource]
  /\ ~inFlightLease[resource]
  /\ physicalAlive' = [physicalAlive EXCEPT ![resource] = FALSE]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       renderingActive,
       pipelineBound,
       declaredSlotCount,
       boundSlots,
       logicalAlive,
       recordingLease,
       inFlightLease,
       recordedWork,
       inFlightWork,
       lastFirstSlot,
       lastBindingCount
     >>

Next ==
  \/ beginRendering
  \/ \E cnt \in DeclaredCounts : bindPipeline(cnt)
  \/ \E fs \in FirstSlotCandidates,
        bc \in BindingCountCandidates,
        v1, v2 \in BOOLEAN :
       attemptBatch(fs, bc, v1, v2)
  \/ attemptDraw
  \/ endRendering
  \/ abandon
  \/ submit
  \/ complete
  \/ \E r \in Resources : destroyResource(r)
  \/ \E r \in Resources : collectResource(r)

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ renderingActive \in BOOLEAN
  /\ pipelineBound \in BOOLEAN
  /\ declaredSlotCount \in DeclaredCounts
  /\ boundSlots \in [Slots -> BOOLEAN]
  /\ logicalAlive \in [Resources -> BOOLEAN]
  /\ physicalAlive \in [Resources -> BOOLEAN]
  /\ recordingLease \in [Resources -> BOOLEAN]
  /\ inFlightLease \in [Resources -> BOOLEAN]
  /\ recordedWork \in BOOLEAN
  /\ inFlightWork \in BOOLEAN
  /\ lastFirstSlot \in Slots
  /\ lastBindingCount \in {0, 1, 2}
  /\ lastOutcome \in Outcomes

BoundSlotsStayWithinPipelineDeclaration ==
  /\ \A s \in Slots : boundSlots[s] =>
       renderingActive /\ pipelineBound /\ s < declaredSlotCount
  /\ (lastBindingCount > 0 =>
        /\ renderingActive
        /\ pipelineBound
        /\ lastFirstSlot + lastBindingCount <= declaredSlotCount
        /\ boundSlots[lastFirstSlot]
        /\ (lastBindingCount = 2 => boundSlots[lastFirstSlot + 1])
        /\ recordingLease[0]
        /\ (lastBindingCount = 2 => recordingLease[1])
        /\ recordedWork)

LeasesAndHandlesRetainPhysicalResources ==
  \A r \in Resources :
    (logicalAlive[r] \/ recordingLease[r] \/ inFlightLease[r]) => physicalAlive[r]

RecordingAndSubmissionAreSeparate ==
  /\ ~(recordedWork /\ inFlightWork)
  /\ \A r \in Resources : ~(recordingLease[r] /\ inFlightLease[r])

Safety ==
  /\ BoundSlotsStayWithinPipelineDeclaration
  /\ LeasesAndHandlesRetainPhysicalResources
  /\ RecordingAndSubmissionAreSeparate

StepSafety ==
  [][lastOutcome' = "rejected" => coreVars' = coreVars]_vars

=============================================================================
