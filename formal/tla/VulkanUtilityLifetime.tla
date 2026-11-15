----------------------- MODULE VulkanUtilityLifetime ------------------------
(***************************************************************************)
(* Formal safety model for backend-private temporary views, descriptors,   *)
(* and helper objects recorded into retained and unretained command pools: *)
(*   - src/Vulkan/VulkanImageUtilities.cpp (resolveTexture,                *)
(*     generateMipmaps, commandBuffer.retained() vs retainSink)            *)
(*   - src/Vulkan/VulkanCommands.cpp (VulkanCommandBuffer::resetRecording  *)
(*     and abandon release retained())                                     *)
(*   - src/Vulkan/VulkanQueues.cpp (submission moves takeRetained() onto   *)
(*     the pending entry; collectGarbage releases it after completion)     *)
(*                                                                         *)
(* Design-rule constant (TRUE in VulkanUtilityLifetime.cfg, FALSE in       *)
(* VulkanUtilityLifetime-skipprivate.cfg):                                 *)
(*   - RetainPrivateInUnretainedPools: internal utility temporaries push   *)
(*     keep-alive references directly onto commandBuffer.retained() even   *)
(*     when CommandPoolFlag::unretainedResources is enabled, because the   *)
(*     caller never owns or sees those internal handles.                   *)
(*                                                                         *)
(* Trusted boundary: caller-managed lifetime of application resources in   *)
(* unretained command pools and native VkImageView destruction.            *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT RetainPrivateInUnretainedPools

ASSUME RetainPrivateInUnretainedPools \in BOOLEAN

Phases == {"recording", "submitted", "abandoned", "completed"}

VARIABLES
  phase,      \* Command-buffer lifecycle phase.
  unretained, \* Whether CommandPoolFlag::unretainedResources is enabled.
  logical,    \* Whether the utility function's scope-local handle is live.
  physical,   \* Whether the native Vulkan object is still physically alive.
  references, \* Count of command-owned keep-alive references (0..1).
  used        \* Whether the command buffer recorded commands using the object.

vars == <<phase, unretained, logical, physical, references, used>>

Init ==
  \E mode \in BOOLEAN :
    /\ phase = "recording"
    /\ unretained = mode
    /\ logical = FALSE
    /\ physical = FALSE
    /\ references = 0
    /\ used = FALSE

createTemporary ==
  /\ phase = "recording"
  /\ ~physical
  /\ ~used
  /\ logical' = TRUE
  /\ physical' = TRUE
  /\ UNCHANGED <<phase, unretained, references, used>>

recordUse ==
  /\ phase = "recording"
  /\ logical
  /\ ~used
  /\ references' = IF RetainPrivateInUnretainedPools \/ ~unretained THEN 1 ELSE 0
  /\ used' = TRUE
  /\ UNCHANGED <<phase, unretained, logical, physical>>

dropPrivateHandle ==
  /\ logical
  /\ logical' = FALSE
  /\ UNCHANGED <<phase, unretained, physical, references, used>>

collectPhysical ==
  /\ physical
  /\ ~logical
  /\ references = 0
  /\ physical' = FALSE
  /\ UNCHANGED <<phase, unretained, logical, references, used>>

submit ==
  /\ phase = "recording"
  /\ used
  /\ ~logical
  /\ phase' = "submitted"
  /\ UNCHANGED <<unretained, logical, physical, references, used>>

reset ==
  /\ phase = "recording"
  /\ ~logical
  /\ references' = 0
  /\ used' = FALSE
  /\ UNCHANGED <<phase, unretained, logical, physical>>

abandon ==
  /\ phase = "recording"
  /\ ~logical
  /\ phase' = "abandoned"
  /\ references' = 0
  /\ used' = FALSE
  /\ UNCHANGED <<unretained, logical, physical>>

complete ==
  /\ phase = "submitted"
  /\ phase' = "completed"
  /\ references' = 0
  /\ used' = FALSE
  /\ UNCHANGED <<unretained, logical, physical>>

Next ==
  \/ createTemporary
  \/ recordUse
  \/ dropPrivateHandle
  \/ collectPhysical
  \/ submit
  \/ reset
  \/ abandon
  \/ complete

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ phase \in Phases
  /\ unretained \in BOOLEAN
  /\ logical \in BOOLEAN
  /\ physical \in BOOLEAN
  /\ references \in 0..1
  /\ used \in BOOLEAN

OwnersRetainPhysicalObject ==
  (logical \/ references > 0) => physical

PrivateTemporarySurvivesUntilRetirement ==
  (used /\ phase \in {"recording", "submitted"}) => physical

SubmittedCommandRetainsPrivateTemporary ==
  phase = "submitted" => references = 1

RetiredCommandReleasesPrivateTemporary ==
  phase \in {"abandoned", "completed"} => references = 0

Safety ==
  /\ OwnersRetainPhysicalObject
  /\ PrivateTemporarySurvivesUntilRetirement
  /\ SubmittedCommandRetainsPrivateTemporary
  /\ RetiredCommandReleasesPrivateTemporary

StepSafety ==
  [][(physical /\ ~physical') => (~logical' /\ references' = 0)]_vars

=============================================================================
