-------------------------- MODULE BufferCopyBatch ---------------------------
(***************************************************************************)
(* Multi-region buffer-copy batch validation, same-buffer overlap checks,  *)
(* concurrent recording against in-flight buffers, and resource retention. *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12ResourceCopies.cpp (ResourceManager::                *)
(*     copyBufferRegions)                                                  *)
(*   - src/D3D12/D3D12Queue.cpp (command and timeline retention)           *)
(*                                                                         *)
(* Contract modeled:                                                       *)
(*   - Buffers = {0, 1}. When sameBuffer = FALSE, buffer 0 is copied into  *)
(*     buffer 1 (requiring both logical handles alive and leasing both).   *)
(*     When sameBuffer = TRUE, buffer 0 is copied within itself (requiring *)
(*     only buffer 0 alive and leasing only buffer 0, even if buffer 1 was *)
(*     destroyed).                                                         *)
(*   - Buffers already leased by an in-flight submission (inFlightLease)   *)
(*     CAN be copied in a fresh open recording command buffer so long as   *)
(*     their public handles remain logically alive                         *)
(*     (D3D12ResourceCopies.cpp does not forbid in-flight buffers).        *)
(*   - A batch accepts 1 or 2 non-empty in-bounds regions outside an       *)
(*     active rendering scope with valid copy usages, disjoint destination *)
(*     regions, and (when RejectSameBufferOverlap = TRUE and sameBuffer)   *)
(*     disjoint source and destination byte intervals.                     *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - D3D12 CopyBufferRegion DMA execution after command submission.      *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT RejectSameBufferOverlap

ASSUME RejectSameBufferOverlap \in BOOLEAN

Buffers == {0, 1}
Cells == {0, 1}
CellValues == {0, 7, 9}
RegionCounts == {0, 1, 2}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    recordingOpen,          \* a fresh command buffer is open for recording
    renderingActive,        \* active beginRendering..endRendering scope on recording
    logicalAlive,           \* [Buffers -> BOOLEAN] public BufferHandle alive
    physicalAlive,          \* [Buffers -> BOOLEAN] physical D3D12 resource alive
    plannedCells,           \* [Cells -> CellValues] destination values in recording
    recordingLease,         \* [Buffers -> BOOLEAN] retained by open recording
    inFlightLease,          \* [Buffers -> BOOLEAN] retained by submitted command
    recordedWork,           \* open recording has at least one copy batch
    inFlightWork,           \* a submitted command buffer is currently in flight
    sameBufferOverlapSeen,  \* TRUE if an overlapping same-buffer batch was accepted
    lastOutcome             \* outcome of the most recent step

coreVars == <<recordingOpen, renderingActive, logicalAlive, physicalAlive,
              plannedCells, recordingLease, inFlightLease,
              recordedWork, inFlightWork, sameBufferOverlapSeen>>
vars == <<recordingOpen, renderingActive, logicalAlive, physicalAlive,
          plannedCells, recordingLease, inFlightLease,
          recordedWork, inFlightWork, sameBufferOverlapSeen, lastOutcome>>

UsedBuffers(sameBuffer) ==
    IF sameBuffer THEN {0} ELSE {0, 1}

BatchValid(regionCount, rangesAndUsageOk, destOverlap, sameBuffer, crossOverlap) ==
    /\ recordingOpen
    /\ ~renderingActive
    /\ regionCount \in {1, 2}
    /\ rangesAndUsageOk
    /\ (regionCount = 2 => ~destOverlap)
    /\ (RejectSameBufferOverlap => ~(sameBuffer /\ crossOverlap))
    /\ \A b \in UsedBuffers(sameBuffer) : logicalAlive[b]

TypeOK ==
    /\ recordingOpen \in BOOLEAN
    /\ renderingActive \in BOOLEAN
    /\ logicalAlive \in [Buffers -> BOOLEAN]
    /\ physicalAlive \in [Buffers -> BOOLEAN]
    /\ plannedCells \in [Cells -> CellValues]
    /\ recordingLease \in [Buffers -> BOOLEAN]
    /\ inFlightLease \in [Buffers -> BOOLEAN]
    /\ recordedWork \in BOOLEAN
    /\ inFlightWork \in BOOLEAN
    /\ sameBufferOverlapSeen \in BOOLEAN
    /\ lastOutcome \in Outcomes

Init ==
    /\ recordingOpen = TRUE
    /\ renderingActive = FALSE
    /\ logicalAlive = [b \in Buffers |-> TRUE]
    /\ physicalAlive = [b \in Buffers |-> TRUE]
    /\ plannedCells = [c \in Cells |-> 0]
    /\ recordingLease = [b \in Buffers |-> FALSE]
    /\ inFlightLease = [b \in Buffers |-> FALSE]
    /\ recordedWork = FALSE
    /\ inFlightWork = FALSE
    /\ sameBufferOverlapSeen = FALSE
    /\ lastOutcome = "none"

BeginRecording ==
    /\ ~recordingOpen
    /\ recordingOpen' = TRUE
    /\ renderingActive' = FALSE
    /\ plannedCells' = [c \in Cells |-> 0]
    /\ recordingLease' = [b \in Buffers |-> FALSE]
    /\ recordedWork' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<logicalAlive, physicalAlive, inFlightLease,
                   inFlightWork, sameBufferOverlapSeen>>

AttemptBatch(regionCount, rangesAndUsageOk, destOverlap, sameBuffer, crossOverlap) ==
    /\ recordingOpen
    /\ IF BatchValid(regionCount, rangesAndUsageOk, destOverlap, sameBuffer, crossOverlap)
       THEN /\ plannedCells' =
                 IF regionCount = 1
                 THEN [plannedCells EXCEPT ![0] = 7]
                 ELSE [[plannedCells EXCEPT ![0] = 7] EXCEPT ![1] = 9]
            /\ recordingLease' =
                 [b \in Buffers |-> recordingLease[b] \/ b \in UsedBuffers(sameBuffer)]
            /\ recordedWork' = TRUE
            /\ sameBufferOverlapSeen' = (sameBufferOverlapSeen \/ (sameBuffer /\ crossOverlap))
            /\ lastOutcome' = "accepted"
            /\ UNCHANGED <<recordingOpen, renderingActive, logicalAlive,
                           physicalAlive, inFlightLease, inFlightWork>>
       ELSE /\ lastOutcome' = "rejected"
            /\ UNCHANGED coreVars

EnterRendering ==
    /\ recordingOpen
    /\ ~renderingActive
    /\ renderingActive' = TRUE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<recordingOpen, logicalAlive, physicalAlive, plannedCells,
                   recordingLease, inFlightLease, recordedWork, inFlightWork,
                   sameBufferOverlapSeen>>

LeaveRendering ==
    /\ recordingOpen
    /\ renderingActive
    /\ renderingActive' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<recordingOpen, logicalAlive, physicalAlive, plannedCells,
                   recordingLease, inFlightLease, recordedWork, inFlightWork,
                   sameBufferOverlapSeen>>

DestroyBuffer(b) ==
    /\ logicalAlive[b]
    /\ logicalAlive' = [logicalAlive EXCEPT ![b] = FALSE]
    /\ physicalAlive' = [physicalAlive EXCEPT ![b] = recordingLease[b] \/ inFlightLease[b]]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<recordingOpen, renderingActive, plannedCells, recordingLease,
                   inFlightLease, recordedWork, inFlightWork, sameBufferOverlapSeen>>

\* Submitting closes the open recording and moves its leases onto the queue
\* timeline, after which BeginRecording can open a new command buffer while
\* the submitted batch is still inFlightWork.
Submit ==
    /\ recordingOpen
    /\ ~renderingActive
    /\ recordedWork
    /\ ~inFlightWork
    /\ recordingOpen' = FALSE
    /\ recordingLease' = [b \in Buffers |-> FALSE]
    /\ inFlightLease' = recordingLease
    /\ recordedWork' = FALSE
    /\ inFlightWork' = TRUE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<renderingActive, logicalAlive, physicalAlive,
                   plannedCells, sameBufferOverlapSeen>>

Complete ==
    /\ inFlightWork
    /\ inFlightWork' = FALSE
    /\ inFlightLease' = [b \in Buffers |-> FALSE]
    /\ physicalAlive' = [b \in Buffers |-> logicalAlive[b] \/ recordingLease[b]]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<recordingOpen, renderingActive, logicalAlive,
                   plannedCells, recordingLease, recordedWork,
                   sameBufferOverlapSeen>>

Abandon ==
    /\ recordingOpen
    /\ recordingOpen' = FALSE
    /\ renderingActive' = FALSE
    /\ plannedCells' = [c \in Cells |-> 0]
    /\ recordingLease' = [b \in Buffers |-> FALSE]
    /\ recordedWork' = FALSE
    /\ physicalAlive' = [b \in Buffers |-> logicalAlive[b] \/ inFlightLease[b]]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<logicalAlive, inFlightLease, inFlightWork, sameBufferOverlapSeen>>

Next ==
    \/ BeginRecording
    \/ \E rc \in RegionCounts, ok, dOv, same, cOv \in BOOLEAN :
          AttemptBatch(rc, ok, dOv, same, cOv)
    \/ EnterRendering
    \/ LeaveRendering
    \/ \E b \in Buffers : DestroyBuffer(b)
    \/ Submit
    \/ Complete
    \/ Abandon

Spec == Init /\ [][Next]_vars

NoSameBufferReadWriteOverlap ==
    ~sameBufferOverlapSeen

LeasesRetainPhysicalBuffers ==
    /\ \A b \in Buffers : logicalAlive[b] => physicalAlive[b]
    /\ \A b \in Buffers : (recordingLease[b] \/ inFlightLease[b]) => physicalAlive[b]

RecordedWorkHoldsSourceLease ==
    /\ (recordedWork => recordingOpen /\ recordingLease[0])
    /\ (~recordingOpen => ~renderingActive /\ ~recordedWork /\ \A b \in Buffers : ~recordingLease[b])
    /\ (~inFlightWork => \A b \in Buffers : ~inFlightLease[b])

Safety ==
    /\ NoSameBufferReadWriteOverlap
    /\ LeasesRetainPhysicalBuffers
    /\ RecordedWorkHoldsSourceLease

StepSafety ==
    [][/\ (lastOutcome' = "rejected" => coreVars' = coreVars)
       /\ (lastOutcome' = "accepted" =>
             /\ recordingOpen /\ ~renderingActive
             /\ logicalAlive[0]
             /\ plannedCells'[0] = 7
             /\ recordingLease'[0]
             /\ recordedWork')]_vars
=============================================================================
