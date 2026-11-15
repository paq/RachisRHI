------------------------- MODULE MipmapPublication --------------------------
(***************************************************************************)
(* Transactional barrier/descriptor preparation, dependency staging,       *)
(* rollback atomicity, and single-point native mipmap command publication  *)
(* in the D3D12 backend: ordered preparation, `retainMipmapDependencies`,  *)
(* and the no-fail native chain in `MipmapGenerator::generate`             *)
(* (`src/D3D12/D3D12Mipmap.cpp`); native barrier planning                  *)
(* (`prepareNativeTextureBarrierSequence`) in                              *)
(* `src/D3D12/D3D12ResourceBarriers.cpp`; the sparse-binding snapshot      *)
(* (`captureSparseTextureBinding`) in                                      *)
(* `src/D3D12/D3D12SparseAndFeedback.cpp`; and transactional dependency    *)
(* publication with rollback (`retainCommandDependencies`) in              *)
(* `src/D3D12/D3D12Queue.cpp` (`include/Rachis/Device.h`).                 *)
(*                                                                         *)
(* Models preparation steps (`commandList`, `barriers`, `descriptors`,     *)
(* `sparseSnapshot`) followed by dependency retention across optional      *)
(* texture owner, manual residency, and sparse mapping/publication         *)
(* profiles. Failures at any preparation or retention prefix restore prior *)
(* command dependencies, native work count, and bound compute/ray-tracing  *)
(* pipeline state; `commit` publishes the entire mipmap program at once    *)
(* and clears bound compute/ray-tracing state to `"none"`.                 *)
(*                                                                         *)
(* Abstracted: per-mip extent and color-space encoding rules (covered by   *)
(* `MipmapGeneration.tla`).                                                *)
(***************************************************************************)
EXTENDS Naturals

TextureOwner == 0
DescriptorOwner == 1
Residency == 2
SparseMapping == 3
SparsePublication == 4
ExistingDependency == 5
Dependencies == 0..5
PreparationSteps == {"commandList", "barriers", "descriptors", "sparseSnapshot"}
Phases == {"idle", "preparing", "retaining"}
PipelineStates == {"none", "compute", "rayTracing"}
Outcomes == {"none", "committed", "rejected"}
MaxPublications == 2

BaselineCounts ==
    [d \in Dependencies |-> IF d = ExistingDependency THEN 1 ELSE 0]

VARIABLES
    phase,                  \* transaction phase in Phases
    preparedSteps,          \* completed preparation steps
    stagedDependencies,     \* dependencies retained during current transaction
    commandDependencies,    \* reference counts in [Dependencies -> 0..3]
    dependenciesBefore,     \* commandDependencies snapshot before transaction
    nativeWork,             \* recorded native command count in 1..3
    workBefore,             \* nativeWork snapshot before transaction
    publicationCount,       \* committed mipmap publications in 0..MaxPublications
    pipelineState,          \* caller compute/rayTracing state in PipelineStates
    pipelineStateBefore,    \* pipelineState snapshot before transaction
    retainApplicationOwner, \* whether texture owner dependency is required
    hasResidency,           \* whether manual residency dependency is required
    hasSparseMapping,       \* whether sparse mapping owner is required
    hasSparsePublication,   \* whether sparse publication epoch is required
    lastOutcome             \* outcome of latest transaction step in Outcomes

vars == <<
    phase, preparedSteps, stagedDependencies, commandDependencies,
    dependenciesBefore, nativeWork, workBefore, publicationCount,
    pipelineState, pipelineStateBefore, retainApplicationOwner,
    hasResidency, hasSparseMapping, hasSparsePublication, lastOutcome
>>

RequiredDependencies ==
    {DescriptorOwner}
        \cup (IF retainApplicationOwner THEN {TextureOwner} ELSE {})
        \cup (IF hasResidency THEN {Residency} ELSE {})
        \cup (IF hasSparseMapping THEN {SparseMapping} ELSE {})
        \cup (IF hasSparsePublication THEN {SparsePublication} ELSE {})

Incremented(counts, additions) ==
    [d \in Dependencies |-> counts[d] + (IF d \in additions THEN 1 ELSE 0)]

Init ==
    \E ownerRet \in BOOLEAN, res \in BOOLEAN,
       sparseProfile \in {"none", "owner", "published"} :
        /\ phase = "idle"
        /\ preparedSteps = {}
        /\ stagedDependencies = {}
        /\ commandDependencies = BaselineCounts
        /\ dependenciesBefore = BaselineCounts
        /\ nativeWork = 1
        /\ workBefore = 1
        /\ publicationCount = 0
        /\ pipelineState = "none"
        /\ pipelineStateBefore = "none"
        /\ retainApplicationOwner = ownerRet
        /\ hasResidency = res
        /\ hasSparseMapping = (sparseProfile # "none")
        /\ hasSparsePublication = (sparseProfile = "published")
        /\ lastOutcome = "none"

bindPipelineState(targetState) ==
    /\ phase = "idle"
    /\ pipelineState # targetState
    /\ pipelineState' = targetState
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        phase, preparedSteps, stagedDependencies, commandDependencies,
        dependenciesBefore, nativeWork, workBefore, publicationCount,
        pipelineStateBefore, retainApplicationOwner, hasResidency,
        hasSparseMapping, hasSparsePublication
       >>

beginPreparation ==
    /\ phase = "idle"
    /\ publicationCount < MaxPublications
    /\ phase' = "preparing"
    /\ preparedSteps' = {}
    /\ stagedDependencies' = {}
    /\ dependenciesBefore' = commandDependencies
    /\ workBefore' = nativeWork
    /\ pipelineStateBefore' = pipelineState
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        commandDependencies, nativeWork, publicationCount, pipelineState,
        retainApplicationOwner, hasResidency, hasSparseMapping,
        hasSparsePublication
       >>

prepareStep(step) ==
    /\ phase = "preparing"
    /\ step \notin preparedSteps
    /\ preparedSteps' = preparedSteps \cup {step}
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        phase, stagedDependencies, commandDependencies, dependenciesBefore,
        nativeWork, workBefore, publicationCount, pipelineState,
        pipelineStateBefore, retainApplicationOwner, hasResidency,
        hasSparseMapping, hasSparsePublication
       >>

beginRetention ==
    /\ phase = "preparing"
    /\ preparedSteps = PreparationSteps
    /\ phase' = "retaining"
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        preparedSteps, stagedDependencies, commandDependencies,
        dependenciesBefore, nativeWork, workBefore, publicationCount,
        pipelineState, pipelineStateBefore, retainApplicationOwner,
        hasResidency, hasSparseMapping, hasSparsePublication
       >>

retainDependency(dependency) ==
    /\ phase = "retaining"
    /\ dependency \in RequiredDependencies
    /\ dependency \notin stagedDependencies
    /\ stagedDependencies' = stagedDependencies \cup {dependency}
    /\ commandDependencies' =
        [commandDependencies EXCEPT ![dependency] = commandDependencies[dependency] + 1]
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        phase, preparedSteps, dependenciesBefore, nativeWork, workBefore,
        publicationCount, pipelineState, pipelineStateBefore,
        retainApplicationOwner, hasResidency, hasSparseMapping,
        hasSparsePublication
       >>

reject ==
    /\ phase \in {"preparing", "retaining"}
    /\ phase' = "idle"
    /\ preparedSteps' = {}
    /\ stagedDependencies' = {}
    /\ commandDependencies' = dependenciesBefore
    /\ nativeWork' = workBefore
    /\ pipelineState' = pipelineStateBefore
    /\ lastOutcome' = "rejected"
    /\ UNCHANGED <<
        dependenciesBefore, workBefore, publicationCount,
        pipelineStateBefore, retainApplicationOwner, hasResidency,
        hasSparseMapping, hasSparsePublication
       >>

\* Dependency publication is the linearization point; void native barriers and
\* dispatches then record atomically and clear compute/rayTracing state.
commit ==
    /\ phase = "retaining"
    /\ stagedDependencies = RequiredDependencies
    /\ phase' = "idle"
    /\ preparedSteps' = {}
    /\ stagedDependencies' = {}
    /\ nativeWork' = nativeWork + 1
    /\ publicationCount' = publicationCount + 1
    /\ pipelineState' = "none"
    /\ lastOutcome' = "committed"
    /\ UNCHANGED <<
        commandDependencies, dependenciesBefore, workBefore,
        pipelineStateBefore, retainApplicationOwner, hasResidency,
        hasSparseMapping, hasSparsePublication
       >>

resetCommand ==
    /\ phase = "idle"
    /\ publicationCount > 0
    /\ commandDependencies' = BaselineCounts
    /\ dependenciesBefore' = BaselineCounts
    /\ nativeWork' = 1
    /\ workBefore' = 1
    /\ publicationCount' = 0
    /\ pipelineState' = "none"
    /\ pipelineStateBefore' = "none"
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<
        phase, preparedSteps, stagedDependencies, retainApplicationOwner,
        hasResidency, hasSparseMapping, hasSparsePublication
       >>

Next ==
    \/ \E st \in PipelineStates : bindPipelineState(st)
    \/ beginPreparation
    \/ \E step \in PreparationSteps : prepareStep(step)
    \/ beginRetention
    \/ \E dep \in Dependencies : retainDependency(dep)
    \/ reject
    \/ commit
    \/ resetCommand

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ phase \in Phases
    /\ preparedSteps \in SUBSET PreparationSteps
    /\ stagedDependencies \in SUBSET Dependencies
    /\ commandDependencies \in [Dependencies -> 0..3]
    /\ dependenciesBefore \in [Dependencies -> 0..3]
    /\ nativeWork \in 1..3
    /\ workBefore \in 1..3
    /\ publicationCount \in 0..MaxPublications
    /\ pipelineState \in PipelineStates
    /\ pipelineStateBefore \in PipelineStates
    /\ retainApplicationOwner \in BOOLEAN
    /\ hasResidency \in BOOLEAN
    /\ hasSparseMapping \in BOOLEAN
    /\ hasSparsePublication \in BOOLEAN
    /\ lastOutcome \in Outcomes
    /\ (hasSparsePublication => hasSparseMapping)

PreparationDoesNotMutateCommand ==
    phase = "preparing" =>
        /\ commandDependencies = dependenciesBefore
        /\ nativeWork = workBefore
        /\ pipelineState = pipelineStateBefore

PrivatePrefixMatchesSuccessfulRetains ==
    phase = "retaining" =>
        /\ preparedSteps = PreparationSteps
        /\ stagedDependencies \subseteq RequiredDependencies
        /\ commandDependencies = Incremented(dependenciesBefore, stagedDependencies)
        /\ nativeWork = workBefore
        /\ pipelineState = pipelineStateBefore

RejectionRestoresExactPriorCommand ==
    lastOutcome = "rejected" =>
        /\ phase = "idle"
        /\ commandDependencies = dependenciesBefore
        /\ nativeWork = workBefore
        /\ pipelineState = pipelineStateBefore

CommitPublishesWholeProgram ==
    lastOutcome = "committed" =>
        /\ phase = "idle"
        /\ commandDependencies = Incremented(dependenciesBefore, RequiredDependencies)
        /\ nativeWork = workBefore + 1
        /\ pipelineState = "none"

NoPartialNativeRecording ==
    phase # "idle" => nativeWork = workBefore

Safety ==
    /\ PreparationDoesNotMutateCommand
    /\ PrivatePrefixMatchesSuccessfulRetains
    /\ RejectionRestoresExactPriorCommand
    /\ CommitPublishesWholeProgram
    /\ NoPartialNativeRecording

StepSafety ==
    [][
        /\ (lastOutcome' = "rejected" =>
                /\ commandDependencies' = dependenciesBefore
                /\ nativeWork' = workBefore
                /\ pipelineState' = pipelineStateBefore
                /\ publicationCount' = publicationCount)
        /\ (lastOutcome' = "committed" =>
                /\ phase = "retaining"
                /\ stagedDependencies = RequiredDependencies
                /\ nativeWork' = nativeWork + 1
                /\ pipelineState' = "none")
    ]_vars

=============================================================================
