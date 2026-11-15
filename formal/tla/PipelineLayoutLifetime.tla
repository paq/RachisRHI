----------------------- MODULE PipelineLayoutLifetime -----------------------
(***************************************************************************)
(* Transitive multi-owner lifetime from binding layouts to pipeline        *)
(* layouts and downstream compiled pipelines, plus atomic rejection of     *)
(* stale binding or pipeline layout handles.                               *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Pipeline.cpp (createPipelineLayout,                  *)
(*     destroyPipelineLayout, PipelineLayoutRecord::slots                  *)
(*     bindingLayoutOwner retaining BindingLayoutRecord, and               *)
(*     ComputePipelineRecord and GraphicsPipelineRecord retaining          *)
(*     PipelineLayoutRecord)                                               *)
(*   - src/D3D12/D3D12RayTracingPipeline.cpp (RayTracingPipelineRecord     *)
(*     retaining PipelineLayoutRecord)                                     *)
(*   - src/D3D12/D3D12Binding.cpp (nativeBindingLayout rejecting stale     *)
(*     binding-layout handles)                                             *)
(*   - src/Vulkan/VulkanBindings.cpp (createPipelineLayout,                *)
(*     findPipelineSetLayouts, PipelineLayoutRecord::sets retaining        *)
(*     BindingLayoutRecord)                                                *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Two binding-layout slots, one pipeline layout selecting any subset    *)
(*   of slots, and one downstream pipeline cover every creation,           *)
(*   destruction, stale-handle rejection, and transitive retention order.  *)
(*   Root-signature serialization details are trusted to the backend.      *)
(***************************************************************************)

EXTENDS Naturals

Slots == {0, 1}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  logicalBindingAlive,  \* Public BindingLayoutHandle liveness per slot
  physicalBindingAlive, \* Native BindingLayoutRecord liveness per slot
  layoutLogicalAlive,   \* Public PipelineLayoutHandle liveness
  layoutPhysicalAlive,  \* Native PipelineLayoutRecord liveness
  layoutSlots,          \* Binding-layout slots owned by the physical layout
  pipelineAlive,        \* Downstream compiled pipeline retaining the layout
  lastOutcome           \* Outcome of the most recent creation attempt

coreVars == <<
  logicalBindingAlive, physicalBindingAlive,
  layoutLogicalAlive, layoutPhysicalAlive,
  layoutSlots, pipelineAlive
>>

vars == <<
  logicalBindingAlive, physicalBindingAlive,
  layoutLogicalAlive, layoutPhysicalAlive,
  layoutSlots, pipelineAlive, lastOutcome
>>

Init ==
  /\ logicalBindingAlive = [s \in Slots |-> FALSE]
  /\ physicalBindingAlive = [s \in Slots |-> FALSE]
  /\ layoutLogicalAlive = FALSE
  /\ layoutPhysicalAlive = FALSE
  /\ layoutSlots = {}
  /\ pipelineAlive = FALSE
  /\ lastOutcome = "none"

CreateBinding(slot) ==
  /\ ~physicalBindingAlive[slot]
  /\ logicalBindingAlive' = [logicalBindingAlive EXCEPT ![slot] = TRUE]
  /\ physicalBindingAlive' = [physicalBindingAlive EXCEPT ![slot] = TRUE]
  /\ UNCHANGED <<layoutLogicalAlive, layoutPhysicalAlive, layoutSlots, pipelineAlive>>
  /\ lastOutcome' = "accepted"

DestroyBinding(slot) ==
  /\ logicalBindingAlive[slot]
  /\ logicalBindingAlive' = [logicalBindingAlive EXCEPT ![slot] = FALSE]
  /\ physicalBindingAlive' =
       [physicalBindingAlive EXCEPT ![slot] = (layoutPhysicalAlive /\ slot \in layoutSlots)]
  /\ UNCHANGED <<layoutLogicalAlive, layoutPhysicalAlive, layoutSlots, pipelineAlive>>
  /\ lastOutcome' = "accepted"

AttemptCreatePipelineLayout(reqSlots) ==
  /\ ~layoutPhysicalAlive
  /\ IF \A s \in reqSlots : logicalBindingAlive[s]
     THEN /\ layoutLogicalAlive' = TRUE
          /\ layoutPhysicalAlive' = TRUE
          /\ layoutSlots' = reqSlots
          /\ UNCHANGED <<logicalBindingAlive, physicalBindingAlive, pipelineAlive>>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"

DestroyPipelineLayout ==
  /\ layoutLogicalAlive
  /\ layoutLogicalAlive' = FALSE
  /\ layoutPhysicalAlive' = pipelineAlive
  /\ layoutSlots' = IF pipelineAlive THEN layoutSlots ELSE {}
  /\ physicalBindingAlive' =
       [s \in Slots |-> logicalBindingAlive[s] \/ (pipelineAlive /\ s \in layoutSlots)]
  /\ UNCHANGED <<logicalBindingAlive, pipelineAlive>>
  /\ lastOutcome' = "accepted"

AttemptCreatePipeline ==
  /\ ~pipelineAlive
  /\ IF layoutLogicalAlive
     THEN /\ pipelineAlive' = TRUE
          /\ UNCHANGED <<
               logicalBindingAlive, physicalBindingAlive,
               layoutLogicalAlive, layoutPhysicalAlive, layoutSlots
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"

DestroyPipeline ==
  /\ pipelineAlive
  /\ pipelineAlive' = FALSE
  /\ layoutPhysicalAlive' = layoutLogicalAlive
  /\ layoutSlots' = IF layoutLogicalAlive THEN layoutSlots ELSE {}
  /\ physicalBindingAlive' =
       [s \in Slots |-> logicalBindingAlive[s] \/ (layoutLogicalAlive /\ s \in layoutSlots)]
  /\ UNCHANGED <<logicalBindingAlive, layoutLogicalAlive>>
  /\ lastOutcome' = "accepted"

Next ==
  \/ \E s \in Slots :
       \/ CreateBinding(s)
       \/ DestroyBinding(s)
  \/ \E reqSlots \in SUBSET Slots :
       AttemptCreatePipelineLayout(reqSlots)
  \/ DestroyPipelineLayout
  \/ AttemptCreatePipeline
  \/ DestroyPipeline

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ logicalBindingAlive \in [Slots -> BOOLEAN]
  /\ physicalBindingAlive \in [Slots -> BOOLEAN]
  /\ layoutLogicalAlive \in BOOLEAN
  /\ layoutPhysicalAlive \in BOOLEAN
  /\ layoutSlots \in SUBSET Slots
  /\ pipelineAlive \in BOOLEAN
  /\ lastOutcome \in Outcomes

PipelineLayoutLifetimeExact ==
  layoutPhysicalAlive = (layoutLogicalAlive \/ pipelineAlive)

BindingLayoutLifetimeExact ==
  \A s \in Slots :
    physicalBindingAlive[s] =
      (logicalBindingAlive[s] \/ (layoutPhysicalAlive /\ s \in layoutSlots))

DeadPipelineLayoutReleasesSlots ==
  ~layoutPhysicalAlive => layoutSlots = {}

Safety ==
  /\ PipelineLayoutLifetimeExact
  /\ BindingLayoutLifetimeExact
  /\ DeadPipelineLayoutReleasesSlots

StepSafety ==
  [][
    /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
    /\ (~layoutLogicalAlive /\ layoutLogicalAlive' =>
          \A s \in layoutSlots' : logicalBindingAlive[s])
    /\ (~pipelineAlive /\ pipelineAlive' => layoutLogicalAlive)
  ]_vars

=============================================================================
