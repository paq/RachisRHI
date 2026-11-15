------------------------ MODULE BindingPoolLifecycle ------------------------
(***************************************************************************)
(* Bounded binding-pool slot and copy-on-write descriptor-version          *)
(* accounting, temporary CoW headroom, logical pool destruction, and       *)
(* exact timeline retirement.                                              *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/Common/BindingPoolAccounting.h                                  *)
(*     (bindingPoolVersionUpdateFor, commitBindingPoolVersionCapacity)     *)
(*   - src/D3D12/D3D12RuntimePolicies.h, src/D3D12/D3D12Binding.cpp        *)
(*     (destroyBindingPool, acquireBindingPoolSet,                         *)
(*     acquireBindingPoolVersion, commitBindingPoolVersionUpdate)          *)
(*   - src/D3D12/D3D12CommandRecording.cpp, src/D3D12/D3D12Queue.cpp       *)
(*   - src/Vulkan/VulkanBindings.cpp (allocateReplacementVersion,          *)
(*     publishReplacementVersion, retainBindingSetForCommand)              *)
(*   - src/Vulkan/VulkanGraphics.cpp (bindDescriptorSetBatch)              *)
(*   - src/Metal/MetalBindings.mm (metalPrepareReplacementVersionLease,    *)
(*     metalCommitReplacementVersionLease,                                 *)
(*     metalRetainBindingSetForCommand)                                    *)
(*   - src/Metal/MetalEncoding.mm (bindComputeBindingSets,                 *)
(*     bindGraphicsBindingSets)                                            *)
(*                                                                         *)
(* Contract modeled:                                                       *)
(*   - A bounded pool has 1 set slot and DescriptorCapacity \in {1, 2}     *)
(*     version slots (Versions = {1, 2}; 0 denotes no version).            *)
(*   - Creating a set requires a logically live pool and free descriptor   *)
(*     capacity. Updating an idle set (no recording or in-flight command   *)
(*     retention) reuses its current version slot in place, so even a      *)
(*     1-set, 1-version pool (DescriptorCapacity = 1) accepts idle         *)
(*     updates; updating while a command retains the set requires          *)
(*     copy-on-write headroom (EnforceBoundCopyOnWrite = TRUE).            *)
(*     Invalid, exhausted, or destroyed-pool attempts reject atomically    *)
(*     without mutating pool or set state.                                 *)
(*   - Logical pool destruction blocks new set creations and updates       *)
(*     immediately while preserving physical pool state until the final    *)
(*     set/command version retires at its exact fence timeline point.      *)
(*   - Design rule RetainPoolForLiveVersions (TRUE in the design) keeps    *)
(*     poolPhysical alive while setSlotHeld holds after DestroyPool.       *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - Per-descriptor-type budget vectors (abstracted to one bounded       *)
(*     version counter per version lease).                                 *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS
    RetainPoolForLiveVersions,
    EnforceBoundCopyOnWrite,
    DescriptorCapacity

ASSUME RetainPoolForLiveVersions \in BOOLEAN
ASSUME EnforceBoundCopyOnWrite \in BOOLEAN
ASSUME DescriptorCapacity \in {1, 2}

Versions == {1, 2}
OptionalVersions == {0, 1, 2}
CommandPhases == {"available", "recording", "inFlight"}
Outcomes == {"none", "accepted", "rejected"}
MaxTimeline == 3

VARIABLES
    poolLogical,        \* public BindingPoolHandle is alive
    poolPhysical,       \* internal pool state is physically alive
    setLogical,         \* public BindingSetHandle is alive
    setSlotHeld,        \* set slot in pool is still leased
    currentVersion,     \* version currently owned by setLogical (0 if none)
    versionAlive,       \* [Versions -> BOOLEAN] active version leases
    descriptorUse,      \* number of live version leases in pool (0..2)
    commandPhase,       \* "available", "recording", or "inFlight"
    commandVersion,     \* version retained by recording/in-flight command (0 if none)
    submitted,          \* latest submitted timeline point (0..MaxTimeline)
    completed,          \* latest completed timeline point (0..MaxTimeline)
    retireAt,           \* timeline point at which in-flight command retires
    lastOutcome         \* outcome of the last fallible pool/set step

coreVars == <<poolLogical, poolPhysical, setLogical, setSlotHeld,
              currentVersion, versionAlive, descriptorUse,
              commandPhase, commandVersion, submitted, completed, retireAt>>
vars == <<poolLogical, poolPhysical, setLogical, setSlotHeld,
          currentVersion, versionAlive, descriptorUse,
          commandPhase, commandVersion, submitted, completed, retireAt,
          lastOutcome>>

CommandActive ==
    commandPhase \in {"recording", "inFlight"}

LiveVersionCount(va) ==
    Cardinality({v \in Versions : va[v]})

TypeOK ==
    /\ poolLogical \in BOOLEAN
    /\ poolPhysical \in BOOLEAN
    /\ setLogical \in BOOLEAN
    /\ setSlotHeld \in BOOLEAN
    /\ currentVersion \in OptionalVersions
    /\ versionAlive \in [Versions -> BOOLEAN]
    /\ descriptorUse \in 0..DescriptorCapacity
    /\ commandPhase \in CommandPhases
    /\ commandVersion \in OptionalVersions
    /\ submitted \in 0..MaxTimeline
    /\ completed \in 0..MaxTimeline
    /\ completed <= submitted
    /\ retireAt \in 0..MaxTimeline
    /\ retireAt <= submitted
    /\ lastOutcome \in Outcomes

Init ==
    /\ poolLogical = FALSE
    /\ poolPhysical = FALSE
    /\ setLogical = FALSE
    /\ setSlotHeld = FALSE
    /\ currentVersion = 0
    /\ versionAlive = [v \in Versions |-> FALSE]
    /\ descriptorUse = 0
    /\ commandPhase = "available"
    /\ commandVersion = 0
    /\ submitted = 0
    /\ completed = 0
    /\ retireAt = 0
    /\ lastOutcome = "none"

CreatePool ==
    /\ ~poolPhysical
    /\ poolLogical' = TRUE
    /\ poolPhysical' = TRUE
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<setLogical, setSlotHeld, currentVersion, versionAlive,
                   descriptorUse, commandPhase, commandVersion, submitted,
                   completed, retireAt>>

DestroyPool ==
    /\ poolLogical
    /\ poolLogical' = FALSE
    /\ poolPhysical' = IF RetainPoolForLiveVersions THEN setSlotHeld ELSE FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<setLogical, setSlotHeld, currentVersion, versionAlive,
                   descriptorUse, commandPhase, commandVersion, submitted,
                   completed, retireAt>>

AttemptCreateSet(version) ==
    IF poolLogical /\ ~setSlotHeld /\ ~setLogical
       /\ ~versionAlive[version] /\ descriptorUse < DescriptorCapacity
    THEN /\ setLogical' = TRUE
         /\ setSlotHeld' = TRUE
         /\ currentVersion' = version
         /\ versionAlive' = [versionAlive EXCEPT ![version] = TRUE]
         /\ descriptorUse' = descriptorUse + 1
         /\ lastOutcome' = "accepted"
         /\ UNCHANGED <<poolLogical, poolPhysical, commandPhase,
                        commandVersion, submitted, completed, retireAt>>
    ELSE /\ lastOutcome' = "rejected"
         /\ UNCHANGED coreVars

BindSet ==
    /\ setLogical
    /\ commandPhase = "available"
    /\ currentVersion \in Versions
    /\ versionAlive[currentVersion]
    /\ commandPhase' = "recording"
    /\ commandVersion' = currentVersion
    /\ retireAt' = 0
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<poolLogical, poolPhysical, setLogical, setSlotHeld,
                   currentVersion, versionAlive, descriptorUse,
                   submitted, completed>>

AttemptUpdateSet(nextVersion) ==
    IF poolLogical /\ setLogical /\ currentVersion \in Versions
       /\ nextVersion # currentVersion /\ ~versionAlive[nextVersion]
       /\ IF CommandActive /\ EnforceBoundCopyOnWrite
          THEN descriptorUse < DescriptorCapacity
          ELSE descriptorUse <= DescriptorCapacity
    THEN LET retainsOld == CommandActive /\ EnforceBoundCopyOnWrite /\ commandVersion = currentVersion
             nextAlive == [[versionAlive EXCEPT ![currentVersion] = retainsOld]
                                         EXCEPT ![nextVersion] = TRUE]
         IN /\ currentVersion' = nextVersion
            /\ versionAlive' = nextAlive
            /\ descriptorUse' = LiveVersionCount(nextAlive)
            /\ lastOutcome' = "accepted"
            /\ UNCHANGED <<poolLogical, poolPhysical, setLogical, setSlotHeld,
                           commandPhase, commandVersion, submitted, completed,
                           retireAt>>
    ELSE /\ lastOutcome' = "rejected"
         /\ UNCHANGED coreVars

DestroySet ==
    /\ setLogical
    /\ currentVersion \in Versions
    /\ LET retainsCurrent == CommandActive /\ commandVersion = currentVersion
           nextAlive == [versionAlive EXCEPT ![currentVersion] = retainsCurrent]
           nextSlot == \E v \in Versions : nextAlive[v]
       IN /\ setLogical' = FALSE
          /\ currentVersion' = 0
          /\ setSlotHeld' = nextSlot
          /\ poolPhysical' = (poolLogical \/ (RetainPoolForLiveVersions /\ nextSlot))
          /\ versionAlive' = nextAlive
          /\ descriptorUse' = LiveVersionCount(nextAlive)
          /\ lastOutcome' = "none"
          /\ UNCHANGED <<poolLogical, commandPhase, commandVersion,
                         submitted, completed, retireAt>>

AbandonCommand ==
    /\ commandPhase = "recording"
    /\ commandVersion \in Versions
    /\ LET keepVer == setLogical /\ commandVersion = currentVersion
           nextAlive == [versionAlive EXCEPT ![commandVersion] = keepVer]
           nextSlot == setLogical \/ (\E v \in Versions : nextAlive[v])
       IN /\ commandPhase' = "available"
          /\ commandVersion' = 0
          /\ retireAt' = 0
          /\ setSlotHeld' = nextSlot
          /\ poolPhysical' = (poolLogical \/ (RetainPoolForLiveVersions /\ nextSlot))
          /\ versionAlive' = nextAlive
          /\ descriptorUse' = LiveVersionCount(nextAlive)
          /\ lastOutcome' = "none"
          /\ UNCHANGED <<poolLogical, setLogical, currentVersion,
                         submitted, completed>>

SubmitCommand ==
    /\ commandPhase = "recording"
    /\ submitted < MaxTimeline
    /\ commandPhase' = "inFlight"
    /\ submitted' = submitted + 1
    /\ retireAt' = submitted + 1
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<poolLogical, poolPhysical, setLogical, setSlotHeld,
                   currentVersion, versionAlive, descriptorUse,
                   commandVersion, completed>>

CompleteTimeline ==
    /\ completed < submitted
    /\ completed' = completed + 1
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<poolLogical, poolPhysical, setLogical, setSlotHeld,
                   currentVersion, versionAlive, descriptorUse,
                   commandPhase, commandVersion, submitted, retireAt>>

CollectCommand ==
    /\ commandPhase = "inFlight"
    /\ commandVersion \in Versions
    /\ retireAt <= completed
    /\ LET keepVer == setLogical /\ commandVersion = currentVersion
           nextAlive == [versionAlive EXCEPT ![commandVersion] = keepVer]
           nextSlot == setLogical \/ (\E v \in Versions : nextAlive[v])
       IN /\ commandPhase' = "available"
          /\ commandVersion' = 0
          /\ retireAt' = 0
          /\ setSlotHeld' = nextSlot
          /\ poolPhysical' = (poolLogical \/ (RetainPoolForLiveVersions /\ nextSlot))
          /\ versionAlive' = nextAlive
          /\ descriptorUse' = LiveVersionCount(nextAlive)
          /\ lastOutcome' = "none"
          /\ UNCHANGED <<poolLogical, setLogical, currentVersion,
                         submitted, completed>>

Next ==
    \/ CreatePool
    \/ DestroyPool
    \/ \E v \in Versions : AttemptCreateSet(v) \/ AttemptUpdateSet(v)
    \/ BindSet
    \/ DestroySet
    \/ AbandonCommand
    \/ SubmitCommand
    \/ CompleteTimeline
    \/ CollectCommand

Spec == Init /\ [][Next]_vars

ExactDescriptorAccounting ==
    descriptorUse = LiveVersionCount(versionAlive)

ExactSetSlotAccounting ==
    setSlotHeld <=> (setLogical \/ (\E v \in Versions : versionAlive[v]))

LiveVersionsRetainPhysicalPool ==
    /\ (poolLogical => poolPhysical)
    /\ (setSlotHeld => poolPhysical)
    /\ (~poolLogical /\ ~setSlotHeld => ~poolPhysical)

LiveSetHasCurrentVersion ==
    setLogical =>
        /\ poolPhysical
        /\ setSlotHeld
        /\ currentVersion \in Versions
        /\ versionAlive[currentVersion]

ActiveCommandRetainsExactVersion ==
    CommandActive =>
        /\ poolPhysical
        /\ setSlotHeld
        /\ commandVersion \in Versions
        /\ versionAlive[commandVersion]

InFlightCommandHasExactRetirement ==
    commandPhase = "inFlight" => retireAt > 0 /\ retireAt <= submitted

Safety ==
    /\ ExactDescriptorAccounting
    /\ ExactSetSlotAccounting
    /\ LiveVersionsRetainPhysicalPool
    /\ LiveSetHasCurrentVersion
    /\ ActiveCommandRetainsExactVersion
    /\ InFlightCommandHasExactRetirement

StepSafety ==
    [][/\ (lastOutcome' = "rejected" => coreVars' = coreVars)
       /\ (~poolLogical =>
             /\ (~setLogical => ~setLogical')
             /\ (setLogical => currentVersion' \in {0, currentVersion}))
       /\ submitted' >= submitted
       /\ completed' >= completed]_vars
=============================================================================
