-------------------- MODULE CommandDependencyPublication --------------------
(***************************************************************************)
(* Transactional 4-family command-dependency retention and rollback across *)
(* resource owners, manual-residency leases, swapchain image leases, and   *)
(* sparse-mapping timeline dependencies.                                   *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Queue.h (QueueManager::ResourceRetention and the     *)
(*     retainCommandDependencies declaration)                              *)
(*   - src/D3D12/D3D12Queue.cpp (PoolEntry dependency vectors;             *)
(*     QueueManager::retainCommandDependencies, retainResource,            *)
(*     retainSwapchainImageLeases, retainSparseMappingDependencies)        *)
(*   - src/D3D12/D3D12Residency.cpp (appendResidencyLeases: all-or-nothing *)
(*     manual-residency lease acquisition inside retainResource)           *)
(*                                                                         *)
(* Contract modeled:                                                       *)
(*   - retainCommandDependencies snapshots all four command-entry vectors  *)
(*     before mutating any of them: retainedResources, residencyLeases,    *)
(*     swapchainImageReferences, and sparseMappingDependencies.            *)
(*   - Existing baseline elements prove rollback truncates back to the     *)
(*     exact prior command state rather than clearing the vectors.         *)
(*   - Stages run sequentially: destination resource + residency, second   *)
(*     resource (pool), optional swapchain image lease, and optional       *)
(*     sparse mapping timeline dependency.                                 *)
(*   - Design rule RollbackAllFourFamilies (TRUE in the design) restores   *)
(*     all four vectors on failure at any stage, including a late sparse   *)
(*     retention failure after resources, residencies, and swapchain       *)
(*     leases were already appended.                                       *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - Native D3D12 command recording after retainCommandDependencies      *)
(*     returns success.                                                    *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT RollbackAllFourFamilies

ASSUME RollbackAllFourFamilies \in BOOLEAN

Destination == 0
Pool == 1
ExistingResource == 2
Resources == {Destination, Pool, ExistingResource}

DestinationResidency == 0
ExistingResidency == 1
Residencies == {DestinationResidency, ExistingResidency}

Lease == 0
ExistingLease == 1
Leases == {Lease, ExistingLease}

SparseDep == 0
ExistingSparse == 1
SparseDeps == {SparseDep, ExistingSparse}

BaselineResources == {ExistingResource}
BaselineResidencies == {ExistingResidency}
BaselineLeases == {ExistingLease}
BaselineSparse == {ExistingSparse}

CompleteResources == BaselineResources \cup {Destination, Pool}
CompleteResidencies == BaselineResidencies \cup {DestinationResidency}
CompleteLeases == BaselineLeases \cup {Lease}
CompleteSparse == BaselineSparse \cup {SparseDep}

Phases == {"idle", "destination", "pool", "lease", "sparse"}
Outcomes == {"none", "pending", "failed", "committedQuery", "committedLease", "committedFull"}

VARIABLES
    phase,                  \* stage inside synchronous retainCommandDependencies
    retainedResources,      \* entry.retainedResources
    retainedResidencies,    \* entry.residencyLeases
    retainedLeases,         \* entry.swapchainImageReferences
    retainedSparse,         \* entry.sparseMappingDependencies
    resourcesBefore,        \* pre-call snapshot of retainedResources
    residenciesBefore,      \* pre-call snapshot of retainedResidencies
    leasesBefore,           \* pre-call snapshot of retainedLeases
    sparseBefore,           \* pre-call snapshot of retainedSparse
    observedResources,      \* caller-visible resources outside call
    observedResidencies,    \* caller-visible residencies outside call
    observedLeases,         \* caller-visible swapchain leases outside call
    observedSparse,         \* caller-visible sparse dependencies outside call
    lastOutcome             \* outcome of the latest stage or transaction

retainedVars == <<retainedResources, retainedResidencies, retainedLeases, retainedSparse>>
beforeVars == <<resourcesBefore, residenciesBefore, leasesBefore, sparseBefore>>
observedVars == <<observedResources, observedResidencies, observedLeases, observedSparse>>
vars == <<phase, retainedResources, retainedResidencies, retainedLeases,
          retainedSparse, resourcesBefore, residenciesBefore, leasesBefore,
          sparseBefore, observedResources, observedResidencies,
          observedLeases, observedSparse, lastOutcome>>

SnapshotBefore ==
    /\ resourcesBefore' = retainedResources
    /\ residenciesBefore' = retainedResidencies
    /\ leasesBefore' = retainedLeases
    /\ sparseBefore' = retainedSparse

RollbackToSnapshot ==
    IF RollbackAllFourFamilies
    THEN /\ retainedResources' = resourcesBefore
         /\ retainedResidencies' = residenciesBefore
         /\ retainedLeases' = leasesBefore
         /\ retainedSparse' = sparseBefore
    ELSE /\ retainedResources' = resourcesBefore
         /\ retainedResidencies' = residenciesBefore
         /\ UNCHANGED <<retainedLeases, retainedSparse>>

TypeOK ==
    /\ phase \in Phases
    /\ lastOutcome \in Outcomes
    /\ retainedResources \subseteq Resources
    /\ retainedResidencies \subseteq Residencies
    /\ retainedLeases \subseteq Leases
    /\ retainedSparse \subseteq SparseDeps
    /\ resourcesBefore \subseteq Resources
    /\ residenciesBefore \subseteq Residencies
    /\ leasesBefore \subseteq Leases
    /\ sparseBefore \subseteq SparseDeps
    /\ observedResources \subseteq Resources
    /\ observedResidencies \subseteq Residencies
    /\ observedLeases \subseteq Leases
    /\ observedSparse \subseteq SparseDeps

Init ==
    /\ phase = "idle"
    /\ retainedResources = BaselineResources
    /\ retainedResidencies = BaselineResidencies
    /\ retainedLeases = BaselineLeases
    /\ retainedSparse = BaselineSparse
    /\ resourcesBefore = BaselineResources
    /\ residenciesBefore = BaselineResidencies
    /\ leasesBefore = BaselineLeases
    /\ sparseBefore = BaselineSparse
    /\ observedResources = BaselineResources
    /\ observedResidencies = BaselineResidencies
    /\ observedLeases = BaselineLeases
    /\ observedSparse = BaselineSparse
    /\ lastOutcome = "none"

RejectDestination ==
    /\ phase = "idle"
    /\ SnapshotBefore
    /\ lastOutcome' = "failed"
    /\ UNCHANGED <<phase, retainedVars, observedVars>>

RetainDestination ==
    /\ phase = "idle"
    /\ phase' = "destination"
    /\ SnapshotBefore
    /\ retainedResources' = retainedResources \cup {Destination}
    /\ retainedResidencies' = retainedResidencies \cup {DestinationResidency}
    /\ lastOutcome' = "pending"
    /\ UNCHANGED <<retainedLeases, retainedSparse, observedVars>>

RejectPool ==
    /\ phase = "destination"
    /\ phase' = "idle"
    /\ RollbackToSnapshot
    /\ lastOutcome' = "failed"
    /\ UNCHANGED <<beforeVars, observedVars>>

RetainPool ==
    /\ phase = "destination"
    /\ phase' = "pool"
    /\ retainedResources' = retainedResources \cup {Pool}
    /\ UNCHANGED <<retainedResidencies, retainedLeases, retainedSparse,
                   beforeVars, observedVars, lastOutcome>>

CommitQuery ==
    /\ phase = "pool"
    /\ phase' = "idle"
    /\ lastOutcome' = "committedQuery"
    /\ UNCHANGED <<retainedVars, beforeVars, observedVars>>

RejectLease ==
    /\ phase = "pool"
    /\ phase' = "idle"
    /\ RollbackToSnapshot
    /\ lastOutcome' = "failed"
    /\ UNCHANGED <<beforeVars, observedVars>>

RetainLease ==
    /\ phase = "pool"
    /\ phase' = "lease"
    /\ retainedLeases' = retainedLeases \cup {Lease}
    /\ UNCHANGED <<retainedResources, retainedResidencies, retainedSparse,
                   beforeVars, observedVars, lastOutcome>>

CommitLease ==
    /\ phase = "lease"
    /\ phase' = "idle"
    /\ lastOutcome' = "committedLease"
    /\ UNCHANGED <<retainedVars, beforeVars, observedVars>>

\* Sparse mapping dependencies are appended after swapchain leases; a failure
\* here must roll back resources, residencies, AND swapchain leases together.
RejectSparse ==
    /\ phase = "lease"
    /\ phase' = "idle"
    /\ RollbackToSnapshot
    /\ lastOutcome' = "failed"
    /\ UNCHANGED <<beforeVars, observedVars>>

RetainSparse ==
    /\ phase = "lease"
    /\ phase' = "sparse"
    /\ retainedSparse' = retainedSparse \cup {SparseDep}
    /\ UNCHANGED <<retainedResources, retainedResidencies, retainedLeases,
                   beforeVars, observedVars, lastOutcome>>

CommitFull ==
    /\ phase = "sparse"
    /\ phase' = "idle"
    /\ lastOutcome' = "committedFull"
    /\ UNCHANGED <<retainedVars, beforeVars, observedVars>>

Observe ==
    /\ phase = "idle"
    /\ observedResources' = retainedResources
    /\ observedResidencies' = retainedResidencies
    /\ observedLeases' = retainedLeases
    /\ observedSparse' = retainedSparse
    /\ UNCHANGED <<phase, retainedVars, beforeVars, lastOutcome>>

Next ==
    \/ RejectDestination
    \/ RetainDestination
    \/ RejectPool
    \/ RetainPool
    \/ CommitQuery
    \/ RejectLease
    \/ RetainLease
    \/ CommitLease
    \/ RejectSparse
    \/ RetainSparse
    \/ CommitFull
    \/ Observe

Spec == Init /\ [][Next]_vars

PartialPrefixIsExact ==
    \/ phase = "idle"
    \/ /\ phase = "destination"
       /\ retainedResources = resourcesBefore \cup {Destination}
       /\ retainedResidencies = residenciesBefore \cup {DestinationResidency}
       /\ retainedLeases = leasesBefore
       /\ retainedSparse = sparseBefore
    \/ /\ phase = "pool"
       /\ retainedResources = resourcesBefore \cup {Destination, Pool}
       /\ retainedResidencies = residenciesBefore \cup {DestinationResidency}
       /\ retainedLeases = leasesBefore
       /\ retainedSparse = sparseBefore
    \/ /\ phase = "lease"
       /\ retainedResources = resourcesBefore \cup {Destination, Pool}
       /\ retainedResidencies = residenciesBefore \cup {DestinationResidency}
       /\ retainedLeases = leasesBefore \cup {Lease}
       /\ retainedSparse = sparseBefore
    \/ /\ phase = "sparse"
       /\ retainedResources = resourcesBefore \cup {Destination, Pool}
       /\ retainedResidencies = residenciesBefore \cup {DestinationResidency}
       /\ retainedLeases = leasesBefore \cup {Lease}
       /\ retainedSparse = sparseBefore \cup {SparseDep}

FailedPublicationRestoresExactPriorState ==
    (phase = "idle" /\ lastOutcome = "failed") =>
        /\ retainedResources = resourcesBefore
        /\ retainedResidencies = residenciesBefore
        /\ retainedLeases = leasesBefore
        /\ retainedSparse = sparseBefore

CommittedPublicationIsComplete ==
    /\ (lastOutcome \in {"committedQuery", "committedLease", "committedFull"} =>
          /\ CompleteResources \subseteq retainedResources
          /\ CompleteResidencies \subseteq retainedResidencies)
    /\ (lastOutcome \in {"committedLease", "committedFull"} =>
          CompleteLeases \subseteq retainedLeases)
    /\ (lastOutcome = "committedFull" =>
          CompleteSparse \subseteq retainedSparse)

ObservationNeverSeesPartialPublication ==
    \/ /\ observedResources = BaselineResources
       /\ observedResidencies = BaselineResidencies
       /\ observedLeases = BaselineLeases
       /\ observedSparse = BaselineSparse
    \/ /\ observedResources = CompleteResources
       /\ observedResidencies = CompleteResidencies
       /\ observedLeases = BaselineLeases
       /\ observedSparse = BaselineSparse
    \/ /\ observedResources = CompleteResources
       /\ observedResidencies = CompleteResidencies
       /\ observedLeases = CompleteLeases
       /\ observedSparse = BaselineSparse
    \/ /\ observedResources = CompleteResources
       /\ observedResidencies = CompleteResidencies
       /\ observedLeases = CompleteLeases
       /\ observedSparse = CompleteSparse

Safety ==
    /\ PartialPrefixIsExact
    /\ FailedPublicationRestoresExactPriorState
    /\ CommittedPublicationIsComplete
    /\ ObservationNeverSeesPartialPublication

StepSafety ==
    [][(phase # "idle" /\ phase' = "idle" /\ lastOutcome' = "failed"
        /\ RollbackAllFourFamilies) =>
          (retainedResources' = resourcesBefore
           /\ retainedResidencies' = residenciesBefore
           /\ retainedLeases' = leasesBefore
           /\ retainedSparse' = sparseBefore)]_vars
=============================================================================
