-------------------------- MODULE TextureCopyBatch --------------------------
(***************************************************************************)
(* Formal safety model for multi-region buffer-to-texture,                 *)
(* texture-to-buffer, and texture-to-texture copy batches:                 *)
(*   - include/Rachis/Device.h, src/Device.cpp (copyBufferToTexture,       *)
(*     copyTextureToBuffer, copyTexture; rendering-scope and region-count  *)
(*     gates)                                                              *)
(*   - src/D3D12/D3D12ResourceCopies.cpp (copyBufferToTextureRegions,      *)
(*     copyTextureToBufferRegions, copyTextureRegions),                    *)
(*     src/D3D12/D3D12Resource.cpp (retainResourceCommandDependencies),    *)
(*     src/D3D12/D3D12Queue.cpp (transactional command retention)          *)
(*   - src/Vulkan/VulkanCopies.cpp (same region validation and retention)  *)
(*   - src/Metal/MetalCopies.mm (same validation, except that overlapping  *)
(*     texture-to-buffer destinations are not rejected)                    *)
(*                                                                         *)
(* Models render-pass exclusion, non-empty region counts, subresource and  *)
(* destination-region overlap rejection, whole-subresource copy rules,     *)
(* atomic batch recording, recording abandonment, and independent physical *)
(* resource reclamation gated on recording and GPU submission leases.      *)
(*                                                                         *)
(* Trusted boundary: arithmetic pitch/footprint overflow checks (such as   *)
(* rowPitch * sliceRows in resolveTransferByteSizes, VulkanCopies.cpp)     *)
(* summarized by regionsValid, and native DMA copy engine execution after  *)
(* submission.                                                             *)
(***************************************************************************)
EXTENDS Naturals

\* 0 = buffer-to-texture, 1 = texture-to-buffer, 2 = texture-to-texture.
Directions == {0, 1, 2}
Resources == {0, 1}
DestinationCells == {0, 1}
RegionCountCandidates == {0, 1, 2, 3}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  renderingActive, \* Whether a render pass is currently open on the command.
  logicalAlive,    \* Per-resource public handle liveness.
  physicalAlive,   \* Per-resource native memory/object liveness.
  destination,     \* Abstract payload recorded into destination cells {0, 1}.
  recordingLease,  \* Per-resource command recording retention flag.
  inFlightLease,   \* Per-resource GPU submission retention flag.
  recordedWork,    \* Whether the command buffer currently holds recorded work.
  inFlightWork,    \* Whether the command buffer is currently in flight on GPU.
  lastRegionCount, \* Region count of the latest copy-batch attempt.
  lastOutcome      \* Outcome of the latest copy-batch attempt.

coreVars == <<
  renderingActive,
  logicalAlive,
  physicalAlive,
  destination,
  recordingLease,
  inFlightLease,
  recordedWork,
  inFlightWork
>>

vars == <<
  renderingActive,
  logicalAlive,
  physicalAlive,
  destination,
  recordingLease,
  inFlightLease,
  recordedWork,
  inFlightWork,
  lastRegionCount,
  lastOutcome
>>

RequestValid(
    direction,
    regionCount,
    regionsValid,
    destinationOverlap,
    sameResource,
    sourceDestinationOverlap,
    wholeCopyRequired,
    wholeRegions) ==
  /\ direction \in Directions
  /\ ~renderingActive
  /\ ~inFlightWork
  /\ regionCount \in {1, 2}
  /\ regionsValid
  /\ ~destinationOverlap
  /\ ~(direction = 2 /\ sameResource /\ sourceDestinationOverlap)
  /\ (~(direction = 2 /\ wholeCopyRequired) \/ wholeRegions)
  /\ \A r \in Resources : logicalAlive[r] /\ ~inFlightLease[r]

Init ==
  /\ renderingActive = FALSE
  /\ logicalAlive = [r \in Resources |-> TRUE]
  /\ physicalAlive = [r \in Resources |-> TRUE]
  /\ destination = [c \in DestinationCells |-> 0]
  /\ recordingLease = [r \in Resources |-> FALSE]
  /\ inFlightLease = [r \in Resources |-> FALSE]
  /\ recordedWork = FALSE
  /\ inFlightWork = FALSE
  /\ lastRegionCount = 0
  /\ lastOutcome = "none"

attemptBatch(
    direction,
    regionCount,
    regionsValid,
    destinationOverlap,
    sameResource,
    sourceDestinationOverlap,
    wholeCopyRequired,
    wholeRegions) ==
  /\ direction \in Directions
  /\ regionCount \in RegionCountCandidates
  /\ LET valid == RequestValid(
           direction,
           regionCount,
           regionsValid,
           destinationOverlap,
           sameResource,
           sourceDestinationOverlap,
           wholeCopyRequired,
           wholeRegions)
         plannedDestination ==
           IF regionCount = 1
           THEN [destination EXCEPT ![0] = 7]
           ELSE [c \in DestinationCells |-> IF c = 0 THEN 7 ELSE 9]
     IN IF valid
        THEN /\ destination' = plannedDestination
             /\ recordingLease' = [r \in Resources |-> TRUE]
             /\ recordedWork' = TRUE
             /\ lastRegionCount' = regionCount
             /\ lastOutcome' = "accepted"
             /\ UNCHANGED <<
                  renderingActive,
                  logicalAlive,
                  physicalAlive,
                  inFlightLease,
                  inFlightWork
                >>
        ELSE /\ lastRegionCount' = regionCount
             /\ lastOutcome' = "rejected"
             /\ UNCHANGED coreVars

enterRendering ==
  /\ ~renderingActive
  /\ ~inFlightWork
  /\ renderingActive' = TRUE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       destination,
       recordingLease,
       inFlightLease,
       recordedWork,
       inFlightWork,
       lastRegionCount
     >>

leaveRendering ==
  /\ renderingActive
  /\ renderingActive' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       destination,
       recordingLease,
       inFlightLease,
       recordedWork,
       inFlightWork,
       lastRegionCount
     >>

destroyResource(resource) ==
  /\ logicalAlive[resource]
  /\ logicalAlive' = [logicalAlive EXCEPT ![resource] = FALSE]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       renderingActive,
       physicalAlive,
       destination,
       recordingLease,
       inFlightLease,
       recordedWork,
       inFlightWork,
       lastRegionCount
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
       logicalAlive,
       destination,
       recordingLease,
       inFlightLease,
       recordedWork,
       inFlightWork,
       lastRegionCount
     >>

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
       logicalAlive,
       physicalAlive,
       destination,
       lastRegionCount
     >>

complete ==
  /\ inFlightWork
  /\ inFlightLease' = [r \in Resources |-> FALSE]
  /\ inFlightWork' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       renderingActive,
       logicalAlive,
       physicalAlive,
       destination,
       recordingLease,
       recordedWork,
       lastRegionCount
     >>

abandon ==
  /\ ~inFlightWork
  /\ (renderingActive \/ recordedWork)
  /\ renderingActive' = FALSE
  /\ destination' = [c \in DestinationCells |-> 0]
  /\ recordingLease' = [r \in Resources |-> FALSE]
  /\ recordedWork' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       logicalAlive,
       physicalAlive,
       inFlightLease,
       inFlightWork,
       lastRegionCount
     >>

Next ==
  \/ \E dir \in Directions,
        cnt \in RegionCountCandidates,
        regOk, dstOv, sameRes, srcDstOv, wholeReq, wholeReg \in BOOLEAN :
       attemptBatch(
         dir, cnt, regOk, dstOv, sameRes, srcDstOv, wholeReq, wholeReg)
  \/ enterRendering
  \/ leaveRendering
  \/ \E r \in Resources : destroyResource(r)
  \/ \E r \in Resources : collectResource(r)
  \/ submit
  \/ complete
  \/ abandon

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ renderingActive \in BOOLEAN
  /\ logicalAlive \in [Resources -> BOOLEAN]
  /\ physicalAlive \in [Resources -> BOOLEAN]
  /\ destination \in [DestinationCells -> {0, 7, 9}]
  /\ recordingLease \in [Resources -> BOOLEAN]
  /\ inFlightLease \in [Resources -> BOOLEAN]
  /\ recordedWork \in BOOLEAN
  /\ inFlightWork \in BOOLEAN
  /\ lastRegionCount \in RegionCountCandidates
  /\ lastOutcome \in Outcomes

AcceptedBatchRecordsEveryRegion ==
  lastOutcome = "accepted" =>
    /\ ~renderingActive
    /\ ~inFlightWork
    /\ lastRegionCount \in {1, 2}
    /\ destination[0] = 7
    /\ (lastRegionCount = 2 => destination[1] = 9)
    /\ (\A r \in Resources : recordingLease[r])
    /\ recordedWork

LeasesAndHandlesRetainPhysicalResources ==
  \A r \in Resources :
    (logicalAlive[r] \/ recordingLease[r] \/ inFlightLease[r]) => physicalAlive[r]

RecordingAndSubmissionAreSeparate ==
  /\ ~(recordedWork /\ inFlightWork)
  /\ \A r \in Resources : ~(recordingLease[r] /\ inFlightLease[r])

Safety ==
  /\ AcceptedBatchRecordsEveryRegion
  /\ LeasesAndHandlesRetainPhysicalResources
  /\ RecordingAndSubmissionAreSeparate

StepSafety ==
  [][/\ (lastOutcome' = "rejected" => coreVars' = coreVars)
     /\ (lastOutcome' = "accepted" /\ lastRegionCount' = 1 =>
           destination'[1] = destination[1])]_vars

=============================================================================
