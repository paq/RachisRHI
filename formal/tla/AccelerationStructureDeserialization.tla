---------------- MODULE AccelerationStructureDeserialization ----------------
(***************************************************************************)
(* Acceleration-structure archive compatibility verification, TLAS BLAS    *)
(* pointer relocation, immutable version import, and traversal retention.  *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12AccelerationStructure.cpp (ResourceManager::         *)
(*     inspectAccelerationStructureArchive,                                *)
(*     deserializeAccelerationStructure; free helpers                      *)
(*     writeRelocatedArchive, makeDeserializedVersion)                     *)
(*   - src/D3D12/D3D12Binding.cpp (shader descriptor capture of version)   *)
(*   - src/D3D12/D3D12Queue.cpp (command and timeline retention)           *)
(*                                                                         *)
(* Contract modeled:                                                       *)
(*   - Deserialization accepts only an unbuilt live destination, a         *)
(*     structurally valid driver-compatible archive, matching AS type and  *)
(*     exact deserialized size, matching BLAS relocation pointer count,    *)
(*     and (for TLAS archives) live built BLAS relocation targets.         *)
(*   - Native type checking follows the relocation-table shape: a BLAS     *)
(*     destination rejects relocations and every relocation target must    *)
(*     be a built BLAS for a TLAS destination. The D3D12 archive header    *)
(*     carries no AS type, so typeMatches for an archive without           *)
(*     relocations is a caller obligation.                                 *)
(*   - Rejected import attempts leave destination, relocation, and phase   *)
(*     state completely unchanged (checked by StepSafety).                 *)
(*   - Accepted TLAS imports patch BLAS GPU addresses in the private       *)
(*     upload and capture the relocated BLAS versions inside               *)
(*     version->dependencies so subsequent ray-tracing traversal retains   *)
(*     both the imported TLAS and its relocated BLAS after handle destroy. *)
(*   - Design rule RetainRelocatedBlas (TRUE in the design) keeps          *)
(*     relocated BLAS versions alive while an imported TLAS or traversal   *)
(*     references them.                                                    *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - D3D12 CheckDriverMatchingIdentifier GUID comparison and native      *)
(*     CopyRaytracingAccelerationStructure DESERIALIZE execution.          *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT RetainRelocatedBlas

ASSUME RetainRelocatedBlas \in BOOLEAN

ImportPhases == 0..3
TraversalPhases == 0..3
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    destinationLogical,     \* public handle of destination AS is alive
    destinationPhysical,    \* physical storage of destination AS is retained
    relocationLogical,      \* public handle of relocated BLAS target is alive
    relocationPhysical,     \* physical storage of relocated BLAS is retained
    importPhase,            \* 0:idle, 1:recorded, 2:submitted, 3:completed
    traversalPhase,         \* 0:idle, 1:recorded, 2:submitted, 3:completed
    imported,               \* destination has an imported immutable version
    importedTopLevel,       \* imported version is a TLAS
    dependencyCaptured,     \* imported TLAS version owns relocated BLAS version
    relocationsPatched,     \* archive upload header had BLAS pointers patched
    lastOutcome             \* outcome of the most recent import attempt

coreVars == <<destinationLogical, destinationPhysical, relocationLogical,
              relocationPhysical, importPhase, traversalPhase,
              imported, importedTopLevel, dependencyCaptured, relocationsPatched>>
vars == <<destinationLogical, destinationPhysical, relocationLogical,
          relocationPhysical, importPhase, traversalPhase,
          imported, importedTopLevel, dependencyCaptured, relocationsPatched,
          lastOutcome>>

TypeOK ==
    /\ destinationLogical \in BOOLEAN
    /\ destinationPhysical \in BOOLEAN
    /\ relocationLogical \in BOOLEAN
    /\ relocationPhysical \in BOOLEAN
    /\ importPhase \in ImportPhases
    /\ traversalPhase \in TraversalPhases
    /\ imported \in BOOLEAN
    /\ importedTopLevel \in BOOLEAN
    /\ dependencyCaptured \in BOOLEAN
    /\ relocationsPatched \in BOOLEAN
    /\ lastOutcome \in Outcomes

Init ==
    /\ destinationLogical = TRUE
    /\ destinationPhysical = TRUE
    /\ relocationLogical = TRUE
    /\ relocationPhysical = TRUE
    /\ importPhase = 0
    /\ traversalPhase = 0
    /\ imported = FALSE
    /\ importedTopLevel = FALSE
    /\ dependencyCaptured = FALSE
    /\ relocationsPatched = FALSE
    /\ lastOutcome = "none"

ImportValid(structurallyValid, compatible, exactSize, typeMatches,
            pointerCountMatches, topLevel, relocationBuilt) ==
    /\ destinationLogical
    /\ destinationPhysical
    /\ importPhase = 0
    /\ ~imported
    /\ structurallyValid
    /\ compatible
    /\ exactSize
    /\ typeMatches
    /\ pointerCountMatches
    /\ (topLevel => relocationLogical /\ relocationPhysical /\ relocationBuilt)

AttemptImport(structurallyValid, compatible, exactSize, typeMatches,
              pointerCountMatches, topLevel, relocationBuilt) ==
    IF ImportValid(structurallyValid, compatible, exactSize, typeMatches,
                   pointerCountMatches, topLevel, relocationBuilt)
    THEN /\ importPhase' = 1
         /\ imported' = TRUE
         /\ importedTopLevel' = topLevel
         /\ dependencyCaptured' = topLevel
         /\ relocationsPatched' = topLevel
         /\ lastOutcome' = "accepted"
         /\ UNCHANGED <<destinationLogical, destinationPhysical,
                        relocationLogical, relocationPhysical, traversalPhase>>
    ELSE /\ lastOutcome' = "rejected"
         /\ UNCHANGED coreVars

SubmitImport ==
    /\ importPhase = 1
    /\ importPhase' = 2
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<destinationLogical, destinationPhysical, relocationLogical,
                   relocationPhysical, traversalPhase, imported,
                   importedTopLevel, dependencyCaptured, relocationsPatched>>

CompleteImport ==
    /\ importPhase = 2
    /\ importPhase' = 3
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<destinationLogical, destinationPhysical, relocationLogical,
                   relocationPhysical, traversalPhase, imported,
                   importedTopLevel, dependencyCaptured, relocationsPatched>>

RecordTraversal ==
    /\ destinationLogical /\ destinationPhysical /\ imported
    /\ importPhase = 3
    /\ traversalPhase = 0
    /\ traversalPhase' = 1
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<destinationLogical, destinationPhysical, relocationLogical,
                   relocationPhysical, importPhase, imported,
                   importedTopLevel, dependencyCaptured, relocationsPatched>>

SubmitTraversal ==
    /\ traversalPhase = 1
    /\ traversalPhase' = 2
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<destinationLogical, destinationPhysical, relocationLogical,
                   relocationPhysical, importPhase, imported,
                   importedTopLevel, dependencyCaptured, relocationsPatched>>

CompleteTraversal ==
    /\ traversalPhase = 2
    /\ traversalPhase' = 3
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<destinationLogical, destinationPhysical, relocationLogical,
                   relocationPhysical, importPhase, imported,
                   importedTopLevel, dependencyCaptured, relocationsPatched>>

DestroyDestination ==
    /\ destinationLogical
    /\ destinationLogical' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<destinationPhysical, relocationLogical, relocationPhysical,
                   importPhase, traversalPhase, imported, importedTopLevel,
                   dependencyCaptured, relocationsPatched>>

DestroyRelocation ==
    /\ relocationLogical
    /\ relocationLogical' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<destinationLogical, destinationPhysical, relocationPhysical,
                   importPhase, traversalPhase, imported, importedTopLevel,
                   dependencyCaptured, relocationsPatched>>

Collect ==
    LET keepDestination ==
            \/ destinationLogical
            \/ importPhase \in {1, 2}
            \/ traversalPhase \in {1, 2}
        keepRelocation ==
            \/ relocationLogical
            \/ (RetainRelocatedBlas /\ dependencyCaptured /\ keepDestination)
    IN /\ destinationPhysical' = keepDestination
       /\ relocationPhysical' = keepRelocation
       /\ lastOutcome' = "none"
       /\ UNCHANGED <<destinationLogical, relocationLogical, importPhase,
                      traversalPhase, imported, importedTopLevel,
                      dependencyCaptured, relocationsPatched>>

Next ==
    \/ \E sOk, cOk, szOk, tOk, ptrOk, tl, relOk \in BOOLEAN :
          AttemptImport(sOk, cOk, szOk, tOk, ptrOk, tl, relOk)
    \/ SubmitImport
    \/ CompleteImport
    \/ RecordTraversal
    \/ SubmitTraversal
    \/ CompleteTraversal
    \/ DestroyDestination
    \/ DestroyRelocation
    \/ Collect

Spec == Init /\ [][Next]_vars

ImportWorkRetainsDestination ==
    importPhase \in {1, 2} => destinationPhysical

TraversalRetainsImportedGraph ==
    traversalPhase \in {1, 2} =>
        /\ destinationPhysical
        /\ (importedTopLevel => relocationPhysical)

RelocatedBlasSurvivesForImportedTlas ==
    (dependencyCaptured /\ destinationPhysical) => relocationPhysical

ImportedTopLevelIsRelocated ==
    importedTopLevel => dependencyCaptured /\ relocationsPatched

ImportedMatchesPhase ==
    imported <=> importPhase > 0

Safety ==
    /\ ImportWorkRetainsDestination
    /\ TraversalRetainsImportedGraph
    /\ RelocatedBlasSurvivesForImportedTlas
    /\ ImportedTopLevelIsRelocated
    /\ ImportedMatchesPhase

StepSafety ==
    [][/\ (~imported /\ imported') =>
             (destinationLogical /\ destinationPhysical /\ importPhase = 0
              /\ (importedTopLevel' => relocationLogical /\ relocationPhysical))
       /\ (lastOutcome' = "rejected" /\ coreVars' # coreVars) => FALSE]_vars
=============================================================================
