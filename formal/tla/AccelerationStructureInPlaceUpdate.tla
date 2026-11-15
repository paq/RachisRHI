----------------- MODULE AccelerationStructureInPlaceUpdate -----------------
(***************************************************************************)
(* Exclusive in-place BLAS/TLAS update protocol over shared backing        *)
(* storage, exact-version consumer isolation, and physical retention.      *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12AccelerationStructure.cpp (ResourceManager::         *)
(*     updateBottomLevelAccelerationStructureInPlace and                   *)
(*     updateTopLevelAccelerationStructureInPlace, which enter the shared  *)
(*     update path with inPlace = true; lockAndValidateUpdateTarget)       *)
(*   - src/D3D12/D3D12AccelerationStructureLease.h                         *)
(*     (AccelerationStructureExecutionTracker::canUpdateInPlace,           *)
(*     tryAcquireConsumer, and the consumer/producer execution leases)     *)
(*   - src/D3D12/D3D12Binding.cpp (binding-time consumer lease capture),   *)
(*     src/D3D12/D3D12Query.cpp (postbuild-query consumer leases),         *)
(*     src/D3D12/D3D12Queue.cpp (lease retention until retirement)         *)
(*                                                                         *)
(* Contract modeled:                                                       *)
(*   - An acceleration structure starts at contentVersion = 1 with         *)
(*     updateEnabled \in BOOLEAN (exercising both allowUpdate and          *)
(*     non-updateable structures).                                         *)
(*   - Up to MaxConsumers = 2 external version consumers (bindings, TLAS   *)
(*     dependencies, queries, or traversals) may concurrently retain the   *)
(*     exact contentVersion they captured. Natively these are active       *)
(*     consumer leases on the target's execution tracker: recorded         *)
(*     bindings (a TLAS binding also leases its BLAS dependencies), TLAS   *)
(*     builds/updates reading a BLAS, postbuild queries, and compaction,   *)
(*     serialization, or out-of-place update sources.                      *)
(*   - Exclusivity requires retainedConsumers = 0 AND recordedUpdates = 0  *)
(*     AND inFlightUpdates = 0 so neither an external consumer nor an      *)
(*     already-recorded/in-flight update aliases the shared GPU storage.   *)
(*   - Design rule EnforceExclusivity (TRUE in the design) gates in-place  *)
(*     mutation on full exclusivity across external consumers and active   *)
(*     update commands.                                                    *)
(*   - Design rule ExcludeConsumersDuringUpdate (TRUE in the design) is    *)
(*     the symmetric guard: an external consumer is admitted only while no *)
(*     in-place update is recorded or in flight, so it can never read the  *)
(*     shared storage while another recording mutates it.                  *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - D3D12 PERFORM_UPDATE driver execution and scratch buffer layout.    *)
(*   - canUpdateInPlace also admits further in-place updates recorded into *)
(*     the same command-list recording as the already-active producer,     *)
(*     ordered by the recorded acceleration-structure barrier; the model   *)
(*     bounds this to one recorded or in-flight update.                    *)
(*   - tryAcquireConsumer likewise admits consumers recorded into the      *)
(*     update's own recording after it, ordered by the same barrier; only  *)
(*     consumers from other recordings are modeled.                        *)
(*   - First builds and other publications into unbuilt storage do not     *)
(*     exclude consumers; ordering them before their consumers is the      *)
(*     caller's GPU-ordering obligation.                                   *)
(*   - A built TLAS version holds no lease on its BLAS trackers while no   *)
(*     TLAS build, update, or binding is recorded or in flight, so a later *)
(*     in-place BLAS update is visible to that TLAS's next traversal.      *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS EnforceExclusivity, ExcludeConsumersDuringUpdate

ASSUME EnforceExclusivity \in BOOLEAN
ASSUME ExcludeConsumersDuringUpdate \in BOOLEAN

MaxVersion == 3
MaxConsumers == 2
MaxUpdates == 2
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    logical,            \* public AS handle is alive
    built,              \* AS has a built version through its public handle
    physical,           \* shared GPU backing storage is alive
    updateEnabled,      \* whether the AS was built with allowUpdate
    contentVersion,     \* current content version in shared storage (1..MaxVersion)
    retainedConsumers,  \* external owners holding the captured version (0..MaxConsumers)
    retainedVersion,    \* captured version held by external consumers (0 when none)
    recordedUpdates,    \* in-place updates currently recorded (0..MaxUpdates)
    inFlightUpdates,    \* in-place updates currently executing on GPU (0..MaxUpdates)
    lastOutcome         \* outcome of the most recent update attempt

coreVars == <<logical, built, physical, updateEnabled, contentVersion,
              retainedConsumers, retainedVersion, recordedUpdates, inFlightUpdates>>
vars == <<logical, built, physical, updateEnabled, contentVersion,
          retainedConsumers, retainedVersion, recordedUpdates, inFlightUpdates,
          lastOutcome>>

TypeOK ==
    /\ logical \in BOOLEAN
    /\ built \in BOOLEAN
    /\ physical \in BOOLEAN
    /\ updateEnabled \in BOOLEAN
    /\ contentVersion \in 1..MaxVersion
    /\ retainedConsumers \in 0..MaxConsumers
    /\ retainedVersion \in 0..MaxVersion
    /\ recordedUpdates \in 0..MaxUpdates
    /\ inFlightUpdates \in 0..MaxUpdates
    /\ lastOutcome \in Outcomes

Init ==
    /\ logical = TRUE
    /\ built = TRUE
    /\ physical = TRUE
    /\ updateEnabled \in BOOLEAN
    /\ contentVersion = 1
    /\ retainedConsumers = 0
    /\ retainedVersion = 0
    /\ recordedUpdates = 0
    /\ inFlightUpdates = 0
    /\ lastOutcome = "none"

Exclusive ==
    /\ retainedConsumers = 0
    /\ recordedUpdates = 0
    /\ inFlightUpdates = 0

BaseRequestValid(compatibleLayout, scratchLargeEnough) ==
    /\ logical
    /\ built
    /\ physical
    /\ updateEnabled
    /\ contentVersion < MaxVersion
    /\ recordedUpdates < MaxUpdates
    /\ compatibleLayout
    /\ scratchLargeEnough

AttemptUpdate(compatibleLayout, scratchLargeEnough) ==
    IF BaseRequestValid(compatibleLayout, scratchLargeEnough)
       /\ (EnforceExclusivity => Exclusive)
    THEN /\ contentVersion' = contentVersion + 1
         /\ recordedUpdates' = recordedUpdates + 1
         /\ lastOutcome' = "accepted"
         /\ UNCHANGED <<logical, built, physical, updateEnabled,
                        retainedConsumers, retainedVersion, inFlightUpdates>>
    ELSE /\ lastOutcome' = "rejected"
         /\ UNCHANGED coreVars

RetainConsumer ==
    /\ logical /\ built /\ physical
    /\ retainedConsumers < MaxConsumers
    /\ ExcludeConsumersDuringUpdate => (recordedUpdates = 0 /\ inFlightUpdates = 0)
    /\ (retainedConsumers > 0 => retainedVersion = contentVersion)
    /\ retainedConsumers' = retainedConsumers + 1
    /\ retainedVersion' = IF retainedConsumers = 0 THEN contentVersion ELSE retainedVersion
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<logical, built, physical, updateEnabled, contentVersion,
                   recordedUpdates, inFlightUpdates>>

ReleaseConsumer ==
    /\ retainedConsumers > 0
    /\ retainedConsumers' = retainedConsumers - 1
    /\ retainedVersion' = IF retainedConsumers = 1 THEN 0 ELSE retainedVersion
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<logical, built, physical, updateEnabled, contentVersion,
                   recordedUpdates, inFlightUpdates>>

SubmitUpdate ==
    /\ recordedUpdates > 0
    /\ inFlightUpdates < MaxUpdates
    /\ recordedUpdates' = recordedUpdates - 1
    /\ inFlightUpdates' = inFlightUpdates + 1
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<logical, built, physical, updateEnabled, contentVersion,
                   retainedConsumers, retainedVersion>>

CompleteUpdate ==
    /\ inFlightUpdates > 0
    /\ inFlightUpdates' = inFlightUpdates - 1
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<logical, built, physical, updateEnabled, contentVersion,
                   retainedConsumers, retainedVersion, recordedUpdates>>

Destroy ==
    /\ logical
    /\ logical' = FALSE
    /\ built' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<physical, updateEnabled, contentVersion, retainedConsumers,
                   retainedVersion, recordedUpdates, inFlightUpdates>>

Collect ==
    LET keepPhysical ==
            \/ (logical /\ built)
            \/ retainedConsumers > 0
            \/ recordedUpdates > 0
            \/ inFlightUpdates > 0
    IN /\ physical' = keepPhysical
       /\ lastOutcome' = "none"
       /\ UNCHANGED <<logical, built, updateEnabled, contentVersion,
                      retainedConsumers, retainedVersion, recordedUpdates,
                      inFlightUpdates>>

Next ==
    \/ \E layoutOk, scratchOk \in BOOLEAN : AttemptUpdate(layoutOk, scratchOk)
    \/ RetainConsumer
    \/ ReleaseConsumer
    \/ SubmitUpdate
    \/ CompleteUpdate
    \/ Destroy
    \/ Collect

Spec == Init /\ [][Next]_vars

\* No retained consumer ever observes its captured version mutate in place.
RetainedConsumerSeesStableVersion ==
    retainedConsumers > 0 =>
        /\ physical
        /\ retainedVersion = contentVersion

\* At most one in-place update can be recorded or in flight over the shared storage.
AtMostOneActiveInPlaceUpdate ==
    recordedUpdates + inFlightUpdates <= 1

\* No external consumer reads the shared storage while an in-place update is
\* recorded or in flight, whichever of the two was recorded first.
NoConsumerDuringActiveUpdate ==
    (recordedUpdates + inFlightUpdates > 0) => retainedConsumers = 0

\* Exclusive isolation across external consumers and active update commands.
ExclusiveInPlaceIsolation ==
    /\ RetainedConsumerSeesStableVersion
    /\ AtMostOneActiveInPlaceUpdate
    /\ NoConsumerDuringActiveUpdate

\* Non-updateable acceleration structures never advance past their initial version.
NonUpdateableStaysInitial ==
    ~updateEnabled => contentVersion = 1

\* Active updates keep the shared physical backing allocation alive.
UpdateOwnsPhysicalStorage ==
    (recordedUpdates > 0 \/ inFlightUpdates > 0) => physical

Safety ==
    /\ ExclusiveInPlaceIsolation
    /\ NonUpdateableStaysInitial
    /\ UpdateOwnsPhysicalStorage

StepSafety ==
    [][/\ (contentVersion' > contentVersion) =>
             (logical /\ built /\ physical /\ updateEnabled
              /\ (EnforceExclusivity =>
                    (retainedConsumers = 0 /\ recordedUpdates = 0 /\ inFlightUpdates = 0)))
       /\ (retainedConsumers' > retainedConsumers) =>
             (ExcludeConsumersDuringUpdate => (recordedUpdates = 0 /\ inFlightUpdates = 0))
       /\ contentVersion' >= contentVersion
       /\ updateEnabled' = updateEnabled
       /\ (lastOutcome' = "rejected" /\ coreVars' # coreVars) => FALSE]_vars
=============================================================================
