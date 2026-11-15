----------------------- MODULE VariableDescriptorCount ----------------------
(***************************************************************************)
(* Formal safety model for trailing variable-count descriptor allocations, *)
(* actual-count update clamping, zero-count minimum physical allocation,   *)
(* versioned command-buffer snapshots without wraparound aliasing, and     *)
(* descriptor-set lifetime retention:                                      *)
(*   - include/Rachis/Binding.h (BindingFlag::variableDescriptorCount,     *)
(*     unboundedBindingCount, BindingSetDesc::variableDescriptorCount) and *)
(*     include/Rachis/Device.h (createBindingSet, updateBindingSet)        *)
(*   - src/D3D12/D3D12Binding.cpp (findVariableBinding,                    *)
(*     validateVariableDescriptorCount; initializeDescriptorSources        *)
(*     allocates max(count, 1) slots with a null placeholder;              *)
(*     copy-on-write BindingManager::updateBindingSet and                  *)
(*     publishBindingSetVersion)                                           *)
(*   - src/D3D12/D3D12Pipeline.cpp (unbounded root-signature descriptor    *)
(*     ranges)                                                             *)
(*   - src/D3D12/D3D12CommandRecording.cpp (prepareBindingSetBatch and     *)
(*     retainBindingSetBatchDependencies capture the bound version)        *)
(*   - src/D3D12/D3D12Queue.cpp (QueueManager::retainCommandDependencies   *)
(*     holds it until completion)                                          *)
(*                                                                         *)
(* src/Vulkan/VulkanBindings.cpp and src/Metal/MetalBindings.mm enforce    *)
(* the same single-tail, immutable-count, write-bound, and copy-on-write   *)
(* version rules but allocate exactly the logical count, so the            *)
(* one-descriptor zero-count minimum is D3D12-specific.                    *)
(*                                                                         *)
(* Fixes the modulo-2 version wraparound defect in the original Quint      *)
(* model by selecting fresh version identifiers disjoint from active       *)
(* recorded and in-flight snapshots, eliminates clone-variable tautologies *)
(* and 1-step before* copies, and separates logical destruction from       *)
(* physical reclamation across record/submit/complete/abandon lifecycles.  *)
(*                                                                         *)
(* Trusted boundary: GPU shader execution reading only validated slots     *)
(* within recordedCount / inFlightCount.                                   *)
(***************************************************************************)
EXTENDS Naturals

Counts == {0, 1, 2, 3}
Slots == {0, 1, 2, 3}
Versions == {0, 1, 2, 3}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  setLogical,          \* Public binding-set handle liveness.
  setPhysical,         \* Native descriptor allocation liveness.
  actualCount,         \* Variable descriptor count requested at creation.
  physicalCount,       \* Native descriptor slots allocated (>= 1 when live).
  physicalInitialized, \* Whether backing descriptor slots were initialized.
  currentVersion,      \* Version tag of current host-visible bindings.
  currentBound,        \* Per-slot occupancy in currentVersion.
  recording,           \* Whether an open command buffer retains a snapshot.
  recordedCount,       \* Variable count captured at bind time.
  recordedVersion,     \* Version tag captured at bind time.
  recordedBound,       \* Per-slot occupancy captured at bind time.
  inFlight,            \* Whether a submitted command retains a snapshot.
  inFlightCount,       \* Variable count captured on the in-flight submission.
  inFlightVersion,     \* Version tag captured on the in-flight submission.
  inFlightBound,       \* Per-slot occupancy on the in-flight submission.
  dispatchedSlot,      \* Slot checked by the latest accepted recorded dispatch.
  dispatchRecorded,    \* Whether a dispatch is recorded on the open command.
  lastOutcome          \* Outcome of latest update or dispatch attempt.

coreVars == <<
  setLogical,
  setPhysical,
  actualCount,
  physicalCount,
  physicalInitialized,
  currentVersion,
  currentBound,
  recording,
  recordedCount,
  recordedVersion,
  recordedBound,
  inFlight,
  inFlightCount,
  inFlightVersion,
  inFlightBound,
  dispatchedSlot,
  dispatchRecorded
>>

vars == <<
  setLogical,
  setPhysical,
  actualCount,
  physicalCount,
  physicalInitialized,
  currentVersion,
  currentBound,
  recording,
  recordedCount,
  recordedVersion,
  recordedBound,
  inFlight,
  inFlightCount,
  inFlightVersion,
  inFlightBound,
  dispatchedSlot,
  dispatchRecorded,
  lastOutcome
>>

AllocationCount(count) == IF count = 0 THEN 1 ELSE count

ActiveVersions ==
  {currentVersion}
  \cup (IF recording THEN {recordedVersion} ELSE {})
  \cup (IF inFlight THEN {inFlightVersion} ELSE {})

EmptySlots == [s \in Slots |-> FALSE]

Init ==
  /\ setLogical = FALSE
  /\ setPhysical = FALSE
  /\ actualCount = 0
  /\ physicalCount = 0
  /\ physicalInitialized = FALSE
  /\ currentVersion = 0
  /\ currentBound = EmptySlots
  /\ recording = FALSE
  /\ recordedCount = 0
  /\ recordedVersion = 0
  /\ recordedBound = EmptySlots
  /\ inFlight = FALSE
  /\ inFlightCount = 0
  /\ inFlightVersion = 0
  /\ inFlightBound = EmptySlots
  /\ dispatchedSlot = 0
  /\ dispatchRecorded = FALSE
  /\ lastOutcome = "none"

create(count) ==
  /\ ~setLogical
  /\ ~setPhysical
  /\ count \in Counts
  /\ setLogical' = TRUE
  /\ setPhysical' = TRUE
  /\ actualCount' = count
  /\ physicalCount' = AllocationCount(count)
  /\ physicalInitialized' = TRUE
  /\ currentVersion' = 0
  /\ currentBound' = EmptySlots
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       recording,
       recordedCount,
       recordedVersion,
       recordedBound,
       inFlight,
       inFlightCount,
       inFlightVersion,
       inFlightBound,
       dispatchedSlot,
       dispatchRecorded
     >>

bind ==
  /\ setLogical
  /\ ~inFlight
  /\ recording' = TRUE
  /\ recordedCount' = actualCount
  /\ recordedVersion' = currentVersion
  /\ recordedBound' = currentBound
  /\ dispatchedSlot' = 0
  /\ dispatchRecorded' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       setLogical,
       setPhysical,
       actualCount,
       physicalCount,
       physicalInitialized,
       currentVersion,
       currentBound,
       inFlight,
       inFlightCount,
       inFlightVersion,
       inFlightBound
     >>

\* Updates beyond actualCount (including slot 0 when actualCount = 0 despite
\* physicalCount = 1) reject atomically; accepted updates allocate a fresh
\* version tag that cannot alias any retained recorded or in-flight snapshot.
attemptUpdate(slot, makeBound) ==
  /\ slot \in Slots
  /\ makeBound \in BOOLEAN
  /\ IF setLogical /\ slot < actualCount
     THEN \E nextVer \in Versions \ ActiveVersions :
            /\ currentVersion' = nextVer
            /\ currentBound' = [currentBound EXCEPT ![slot] = makeBound]
            /\ lastOutcome' = "accepted"
            /\ UNCHANGED <<
                 setLogical,
                 setPhysical,
                 actualCount,
                 physicalCount,
                 physicalInitialized,
                 recording,
                 recordedCount,
                 recordedVersion,
                 recordedBound,
                 inFlight,
                 inFlightCount,
                 inFlightVersion,
                 inFlightBound,
                 dispatchedSlot,
                 dispatchRecorded
               >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

attemptRecordedDispatch(slot) ==
  /\ slot \in Slots
  /\ IF recording /\ slot < recordedCount /\ recordedBound[slot]
     THEN /\ dispatchedSlot' = slot
          /\ dispatchRecorded' = TRUE
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               setLogical,
               setPhysical,
               actualCount,
               physicalCount,
               physicalInitialized,
               currentVersion,
               currentBound,
               recording,
               recordedCount,
               recordedVersion,
               recordedBound,
               inFlight,
               inFlightCount,
               inFlightVersion,
               inFlightBound
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

abandonRecording ==
  /\ recording
  /\ recording' = FALSE
  /\ recordedCount' = 0
  /\ recordedVersion' = 0
  /\ recordedBound' = EmptySlots
  /\ dispatchedSlot' = 0
  /\ dispatchRecorded' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       setLogical,
       setPhysical,
       actualCount,
       physicalCount,
       physicalInitialized,
       currentVersion,
       currentBound,
       inFlight,
       inFlightCount,
       inFlightVersion,
       inFlightBound
     >>

submit ==
  /\ recording
  /\ ~inFlight
  /\ recording' = FALSE
  /\ recordedCount' = 0
  /\ recordedVersion' = 0
  /\ recordedBound' = EmptySlots
  /\ inFlight' = TRUE
  /\ inFlightCount' = recordedCount
  /\ inFlightVersion' = recordedVersion
  /\ inFlightBound' = recordedBound
  /\ dispatchedSlot' = 0
  /\ dispatchRecorded' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       setLogical,
       setPhysical,
       actualCount,
       physicalCount,
       physicalInitialized,
       currentVersion,
       currentBound
     >>

complete ==
  /\ inFlight
  /\ inFlight' = FALSE
  /\ inFlightCount' = 0
  /\ inFlightVersion' = 0
  /\ inFlightBound' = EmptySlots
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       setLogical,
       setPhysical,
       actualCount,
       physicalCount,
       physicalInitialized,
       currentVersion,
       currentBound,
       recording,
       recordedCount,
       recordedVersion,
       recordedBound,
       dispatchedSlot,
       dispatchRecorded
     >>

destroySet ==
  /\ setLogical
  /\ setLogical' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       setPhysical,
       actualCount,
       physicalCount,
       physicalInitialized,
       currentVersion,
       currentBound,
       recording,
       recordedCount,
       recordedVersion,
       recordedBound,
       inFlight,
       inFlightCount,
       inFlightVersion,
       inFlightBound,
       dispatchedSlot,
       dispatchRecorded
     >>

collectSet ==
  /\ setPhysical
  /\ ~setLogical
  /\ ~recording
  /\ ~inFlight
  /\ setPhysical' = FALSE
  /\ actualCount' = 0
  /\ physicalCount' = 0
  /\ physicalInitialized' = FALSE
  /\ currentVersion' = 0
  /\ currentBound' = EmptySlots
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       setLogical,
       recording,
       recordedCount,
       recordedVersion,
       recordedBound,
       inFlight,
       inFlightCount,
       inFlightVersion,
       inFlightBound,
       dispatchedSlot,
       dispatchRecorded
     >>

Next ==
  \/ \E c \in Counts : create(c)
  \/ bind
  \/ \E s \in Slots, b \in BOOLEAN : attemptUpdate(s, b)
  \/ \E s \in Slots : attemptRecordedDispatch(s)
  \/ abandonRecording
  \/ submit
  \/ complete
  \/ destroySet
  \/ collectSet

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ setLogical \in BOOLEAN
  /\ setPhysical \in BOOLEAN
  /\ actualCount \in Counts
  /\ physicalCount \in {0, 1, 2, 3}
  /\ physicalInitialized \in BOOLEAN
  /\ currentVersion \in Versions
  /\ currentBound \in [Slots -> BOOLEAN]
  /\ recording \in BOOLEAN
  /\ recordedCount \in Counts
  /\ recordedVersion \in Versions
  /\ recordedBound \in [Slots -> BOOLEAN]
  /\ inFlight \in BOOLEAN
  /\ inFlightCount \in Counts
  /\ inFlightVersion \in Versions
  /\ inFlightBound \in [Slots -> BOOLEAN]
  /\ dispatchedSlot \in Slots
  /\ dispatchRecorded \in BOOLEAN
  /\ lastOutcome \in Outcomes

PhysicalSetHasFiniteInitializedAllocation ==
  setPhysical =>
    /\ physicalCount = AllocationCount(actualCount)
    /\ physicalCount >= 1
    /\ physicalInitialized

SlotsBeyondActualCountRemainUnbound ==
  /\ \A s \in Slots : s >= actualCount => ~currentBound[s]
  /\ (recording => \A s \in Slots : s >= recordedCount => ~recordedBound[s])
  /\ (inFlight => \A s \in Slots : s >= inFlightCount => ~inFlightBound[s])

VersionSnapshotsNeverAliasIncompatibleOccupancy ==
  /\ (recording /\ recordedVersion = currentVersion =>
        recordedBound = currentBound /\ recordedCount = actualCount)
  /\ (inFlight /\ inFlightVersion = currentVersion =>
        inFlightBound = currentBound /\ inFlightCount = actualCount)
  /\ (recording /\ inFlight /\ recordedVersion = inFlightVersion =>
        recordedBound = inFlightBound /\ recordedCount = inFlightCount)

RecordedDispatchUsesPopulatedInBoundsSlot ==
  dispatchRecorded =>
    /\ recording
    /\ dispatchedSlot < recordedCount
    /\ recordedBound[dispatchedSlot]

RetainedSnapshotsKeepAllocationPhysical ==
  (setLogical \/ recording \/ inFlight) => setPhysical

Safety ==
  /\ PhysicalSetHasFiniteInitializedAllocation
  /\ SlotsBeyondActualCountRemainUnbound
  /\ VersionSnapshotsNeverAliasIncompatibleOccupancy
  /\ RecordedDispatchUsesPopulatedInBoundsSlot
  /\ RetainedSnapshotsKeepAllocationPhysical

StepSafety ==
  [][/\ (lastOutcome' = "rejected" => coreVars' = coreVars)
     /\ (setPhysical /\ setPhysical' =>
           actualCount' = actualCount /\ physicalCount' = physicalCount)
     /\ (recording /\ recording' /\ recordedVersion' = recordedVersion =>
           recordedBound' = recordedBound /\ recordedCount' = recordedCount)
     /\ (inFlight /\ inFlight' =>
           inFlightVersion' = inFlightVersion
           /\ inFlightBound' = inFlightBound
           /\ inFlightCount' = inFlightCount)]_vars

=============================================================================
