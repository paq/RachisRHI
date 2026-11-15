------------------------ MODULE ResourceBarrierBatch ------------------------
(***************************************************************************)
(* Atomic global-memory, buffer, texture, and dedicated sampler-feedback   *)
(* barrier batching, legacy vs enhanced barrier capability gating, whole-  *)
(* buffer vs subrange capability gating, and submission retention.         *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - include/Rachis/Device.h (DeviceCapabilities::enhancedBarriers,      *)
(*     DeviceCapabilities::bufferBarrierRanges)                            *)
(*   - src/D3D12/D3D12ResourceBarriers.cpp (resourceBarriers,              *)
(*     validateLegacyGlobalBarriers UAV-only global barrier gate,          *)
(*     whole-buffer range gate, overlap rejection,                         *)
(*     SamplerFeedbackBarrierRetention internal ownership)                 *)
(*   - src/D3D12/D3D12Queue.cpp (command/in-flight retention)              *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Slot 0 is a buffer and Slot 1 is a texture or sampler-feedback map.   *)
(*   Explicit inFlight tracking ensures global-memory-only batches must    *)
(*   still complete on the GPU before command-buffer re-recording.         *)
(***************************************************************************)

EXTENDS Naturals

BufferSlot == 0
TextureLikeSlot == 1
Resources == {BufferSlot, TextureLikeSlot}
States == {0, 1}
GlobalKinds == {"none", "uav", "general"}
BufferRanges == {"none", "whole", "subrange"}
TextureKinds == {"none", "texture", "feedbackMap"}
Operations == {"none", "batch", "destroy", "submit", "complete", "beginRecording"}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  enhancedBarriers,    \* DeviceCapabilities::enhancedBarriers
  bufferBarrierRanges, \* DeviceCapabilities::bufferBarrierRanges
  retainPool,          \* Whether command pool retains application resources
  recording,           \* Whether the command buffer is open for recording
  inFlight,            \* Whether a submitted command buffer is on the GPU
  logicalAlive,        \* Public handle liveness per resource slot
  physicalAlive,       \* Native storage liveness per resource slot
  resourceState,       \* Planned synchronization state per resource slot
  recordingLease,      \* Command-recording ownership per resource slot
  inFlightLease,       \* In-flight GPU ownership per resource slot
  memoryEpoch,         \* Toggles on each accepted global memory barrier
  recordingHasWork,    \* Whether the open recording has at least one batch
  lastOp,              \* Most recent operation kind
  lastOutcome,         \* Outcome of most recent operation
  lastGlobalKind,      \* Global barrier kind of most recent batch attempt
  lastBufferRange,     \* Buffer range kind of most recent batch attempt
  lastTextureKind,     \* Slot-1 barrier kind of most recent batch attempt
  lastDuplicate,       \* Overlap flag of most recent batch attempt
  lastRequestedState   \* Target resource states of most recent batch attempt

coreVars == <<
  enhancedBarriers, bufferBarrierRanges, retainPool,
  recording, inFlight, logicalAlive, physicalAlive,
  resourceState, recordingLease, inFlightLease,
  memoryEpoch, recordingHasWork
>>

vars == <<
  enhancedBarriers, bufferBarrierRanges, retainPool,
  recording, inFlight, logicalAlive, physicalAlive,
  resourceState, recordingLease, inFlightLease,
  memoryEpoch, recordingHasWork,
  lastOp, lastOutcome, lastGlobalKind,
  lastBufferRange, lastTextureKind, lastDuplicate, lastRequestedState
>>

GlobalValid(globalKind) ==
  \/ globalKind = "none"
  \/ globalKind = "uav"
  \/ (globalKind = "general" /\ enhancedBarriers)

BufferEntryValid(bufRange) ==
  \/ bufRange = "none"
  \/ (bufRange = "whole" /\ logicalAlive[BufferSlot])
  \/ (bufRange = "subrange" /\ bufferBarrierRanges /\ logicalAlive[BufferSlot])

TextureEntryValid(texKind) ==
  \/ texKind = "none"
  \/ (texKind \in {"texture", "feedbackMap"} /\ logicalAlive[TextureLikeSlot])

BatchValid(globalKind, bufRange, texKind, duplicateOverlap) ==
  /\ recording
  /\ ~inFlight
  /\ (globalKind # "none" \/ bufRange # "none" \/ texKind # "none")
  /\ ~duplicateOverlap
  /\ GlobalValid(globalKind)
  /\ BufferEntryValid(bufRange)
  /\ TextureEntryValid(texKind)

\* Ordinary buffer/texture barriers retain application ownership iff retainPool;
\* dedicated sampler-feedback map barriers use internal retention in all pools.
SlotRetainedByBatch(slot, bufRange, texKind) ==
  IF slot = BufferSlot
  THEN bufRange # "none" /\ retainPool
  ELSE \/ (texKind = "texture" /\ retainPool)
       \/ texKind = "feedbackMap"

Init ==
  /\ enhancedBarriers \in BOOLEAN
  /\ bufferBarrierRanges \in (IF enhancedBarriers THEN BOOLEAN ELSE {FALSE})
  /\ retainPool \in BOOLEAN
  /\ recording = TRUE
  /\ inFlight = FALSE
  /\ logicalAlive = [s \in Resources |-> TRUE]
  /\ physicalAlive = [s \in Resources |-> TRUE]
  /\ resourceState = [s \in Resources |-> 0]
  /\ recordingLease = [s \in Resources |-> FALSE]
  /\ inFlightLease = [s \in Resources |-> FALSE]
  /\ memoryEpoch = FALSE
  /\ recordingHasWork = FALSE
  /\ lastOp = "none"
  /\ lastOutcome = "none"
  /\ lastGlobalKind = "none"
  /\ lastBufferRange = "none"
  /\ lastTextureKind = "none"
  /\ lastDuplicate = FALSE
  /\ lastRequestedState = [s \in Resources |-> 0]

ResetBatchMeta ==
  /\ lastGlobalKind' = "none"
  /\ lastBufferRange' = "none"
  /\ lastTextureKind' = "none"
  /\ lastDuplicate' = FALSE
  /\ lastRequestedState' = [s \in Resources |-> 0]

AttemptBatch(globalKind, bufRange, texKind, duplicateOverlap, bufState, texState) ==
  LET requested == [s \in Resources |-> IF s = BufferSlot THEN bufState ELSE texState]
      selected == [s \in Resources |->
                     IF s = BufferSlot THEN bufRange # "none" ELSE texKind # "none"]
  IN
  /\ IF BatchValid(globalKind, bufRange, texKind, duplicateOverlap)
     THEN /\ resourceState' =
               [s \in Resources |->
                  IF selected[s] THEN requested[s] ELSE resourceState[s]]
          /\ recordingLease' =
               [s \in Resources |->
                  recordingLease[s] \/ SlotRetainedByBatch(s, bufRange, texKind)]
          /\ memoryEpoch' =
               IF globalKind # "none" THEN ~memoryEpoch ELSE memoryEpoch
          /\ recordingHasWork' = TRUE
          /\ UNCHANGED <<
               enhancedBarriers, bufferBarrierRanges, retainPool,
               recording, inFlight, logicalAlive, physicalAlive, inFlightLease
             >>
          /\ lastOutcome' = "accepted"
          /\ lastGlobalKind' = globalKind
          /\ lastBufferRange' = bufRange
          /\ lastTextureKind' = texKind
          /\ lastDuplicate' = duplicateOverlap
          /\ lastRequestedState' = requested
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
          /\ ResetBatchMeta
  /\ lastOp' = "batch"

\* On unretained pools, the caller preserves unretained application resources
\* while recorded/submitted; retained resources may be destroyed at any time.
DestroyResource(slot) ==
  /\ logicalAlive[slot]
  /\ (retainPool \/ (~recordingHasWork /\ ~inFlight))
  /\ logicalAlive' = [logicalAlive EXCEPT ![slot] = FALSE]
  /\ physicalAlive' =
       [physicalAlive EXCEPT ![slot] = (recordingLease[slot] \/ inFlightLease[slot])]
  /\ UNCHANGED <<
       enhancedBarriers, bufferBarrierRanges, retainPool,
       recording, inFlight, resourceState,
       recordingLease, inFlightLease, memoryEpoch, recordingHasWork
     >>
  /\ lastOp' = "destroy"
  /\ lastOutcome' = "accepted"
  /\ ResetBatchMeta

Submit ==
  /\ recording
  /\ ~inFlight
  /\ recordingHasWork
  /\ recording' = FALSE
  /\ inFlight' = TRUE
  /\ recordingLease' = [s \in Resources |-> FALSE]
  /\ inFlightLease' = recordingLease
  /\ recordingHasWork' = FALSE
  /\ UNCHANGED <<
       enhancedBarriers, bufferBarrierRanges, retainPool,
       logicalAlive, physicalAlive, resourceState, memoryEpoch
     >>
  /\ lastOp' = "submit"
  /\ lastOutcome' = "accepted"
  /\ ResetBatchMeta

Complete ==
  /\ inFlight
  /\ inFlight' = FALSE
  /\ physicalAlive' = [s \in Resources |-> logicalAlive[s]]
  /\ inFlightLease' = [s \in Resources |-> FALSE]
  /\ UNCHANGED <<
       enhancedBarriers, bufferBarrierRanges, retainPool,
       recording, logicalAlive, resourceState,
       recordingLease, memoryEpoch, recordingHasWork
     >>
  /\ lastOp' = "complete"
  /\ lastOutcome' = "accepted"
  /\ ResetBatchMeta

BeginRecording ==
  /\ ~recording
  /\ ~inFlight
  /\ recording' = TRUE
  /\ UNCHANGED <<
       enhancedBarriers, bufferBarrierRanges, retainPool,
       inFlight, logicalAlive, physicalAlive, resourceState,
       recordingLease, inFlightLease, memoryEpoch, recordingHasWork
     >>
  /\ lastOp' = "beginRecording"
  /\ lastOutcome' = "accepted"
  /\ ResetBatchMeta

BufferChoices ==
  {<<"none", 0>>} \cup ({<<"whole", s>> : s \in States}) \cup ({<<"subrange", s>> : s \in States})

TextureChoices ==
  {<<"none", 0>>} \cup ({<<"texture", s>> : s \in States}) \cup ({<<"feedbackMap", s>> : s \in States})

Next ==
  \/ /\ recording
     /\ \E gk \in GlobalKinds, bc \in BufferChoices, tc \in TextureChoices :
          \E dup \in (IF bc[1] = "none" /\ tc[1] = "none" THEN {FALSE} ELSE BOOLEAN) :
            AttemptBatch(gk, bc[1], tc[1], dup, bc[2], tc[2])
  \/ /\ ~recording
     /\ AttemptBatch("uav", "none", "none", FALSE, 0, 0)
  \/ \E s \in Resources : DestroyResource(s)
  \/ Submit
  \/ Complete
  \/ BeginRecording

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ enhancedBarriers \in BOOLEAN
  /\ bufferBarrierRanges \in BOOLEAN
  /\ retainPool \in BOOLEAN
  /\ recording \in BOOLEAN
  /\ inFlight \in BOOLEAN
  /\ logicalAlive \in [Resources -> BOOLEAN]
  /\ physicalAlive \in [Resources -> BOOLEAN]
  /\ resourceState \in [Resources -> States]
  /\ recordingLease \in [Resources -> BOOLEAN]
  /\ inFlightLease \in [Resources -> BOOLEAN]
  /\ memoryEpoch \in BOOLEAN
  /\ recordingHasWork \in BOOLEAN
  /\ lastOp \in Operations
  /\ lastOutcome \in Outcomes
  /\ lastGlobalKind \in GlobalKinds
  /\ lastBufferRange \in BufferRanges
  /\ lastTextureKind \in TextureKinds
  /\ lastDuplicate \in BOOLEAN
  /\ lastRequestedState \in [Resources -> States]

PhysicalLifetimeExact ==
  \A s \in Resources :
    physicalAlive[s] =
      (logicalAlive[s] \/ recordingLease[s] \/ inFlightLease[s])

RecordingAndInFlightAreMutuallyExclusive ==
  /\ ~(recording /\ inFlight)
  /\ (~recording => ~recordingHasWork /\ \A s \in Resources : ~recordingLease[s])
  /\ (~inFlight => \A s \in Resources : ~inFlightLease[s])

UnretainedPoolSkipsBufferLeaseOnly ==
  ~retainPool =>
    /\ ~recordingLease[BufferSlot]
    /\ ~inFlightLease[BufferSlot]

Safety ==
  /\ PhysicalLifetimeExact
  /\ RecordingAndInFlightAreMutuallyExclusive
  /\ UnretainedPoolSkipsBufferLeaseOnly

StepSafety ==
  [][
    /\ UNCHANGED <<enhancedBarriers, bufferBarrierRanges, retainPool>>
    /\ (lastOutcome' = "rejected" => UNCHANGED coreVars)
    /\ (lastOp' = "batch" /\ lastOutcome' = "accepted" =>
          /\ BatchValid(lastGlobalKind', lastBufferRange',
                        lastTextureKind', lastDuplicate')
          /\ recordingHasWork'
          /\ (memoryEpoch' # memoryEpoch <=> lastGlobalKind' # "none")
          /\ (lastBufferRange' # "none" =>
                resourceState'[BufferSlot] = lastRequestedState'[BufferSlot])
          /\ (lastBufferRange' = "none" =>
                /\ resourceState'[BufferSlot] = resourceState[BufferSlot]
                /\ recordingLease'[BufferSlot] = recordingLease[BufferSlot])
          /\ (lastTextureKind' # "none" =>
                resourceState'[TextureLikeSlot] = lastRequestedState'[TextureLikeSlot])
          /\ (lastTextureKind' = "none" =>
                /\ resourceState'[TextureLikeSlot] = resourceState[TextureLikeSlot]
                /\ recordingLease'[TextureLikeSlot] = recordingLease[TextureLikeSlot])
          /\ (lastTextureKind' = "feedbackMap" => recordingLease'[TextureLikeSlot]))
    /\ (lastOp' = "beginRecording" => ~inFlight)
  ]_vars

=============================================================================
