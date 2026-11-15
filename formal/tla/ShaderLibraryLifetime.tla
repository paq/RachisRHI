----------------------- MODULE ShaderLibraryLifetime ------------------------
(***************************************************************************)
(* Formal safety model for staged shader libraries, exact stage/entry      *)
(* module selection, shared pipeline ownership, and command retention:     *)
(*   - include/Rachis/Pipeline.h (ShaderLibraryDesc, ShaderModuleDesc)     *)
(*   - src/Device.cpp (facade forwarding)                                  *)
(*   - src/D3D12/D3D12Pipeline.cpp (createShaderLibrary,                   *)
(*     destroyShaderLibrary, createShaderModule, destroyShaderModule)      *)
(*   - src/D3D12/D3D12Queue.cpp (command retention)                        *)
(*                                                                         *)
(* Models unique staged entry validation on library creation, two module   *)
(* slots and two pipeline slots so sibling modules/pipelines can share a   *)
(* source across partial destruction, recording abandonment vs submission, *)
(* and independent physical reclamation gated on zero remaining owners.    *)
(*                                                                         *)
(* Trusted boundary: offline shader compilation (DXIL/SPIR-V/PNMS) and     *)
(* native driver PSO destruction once physical references reach zero.      *)
(***************************************************************************)
EXTENDS Naturals

Entries == {"vsMain", "psMain"}
Modules == {0, 1}
Pipelines == {0, 1}
Sources == {"direct", "library", "both", "missing"}
CommandPhases == {"idle", "recorded", "inFlight"}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  libraryLogical,    \* Whether the public shader-library handle is live.
  libraryPhysical,   \* Whether the underlying library bytecode is retained.
  libraryEntries,    \* Set of valid unique (stage, name) entries in library.
  moduleLogical,     \* Per-module public handle liveness.
  modulePhysical,    \* Per-module physical shader bytecode liveness.
  moduleOwnsLibrary, \* Whether each module retains the shader library.
  pipelineLogical,   \* Per-pipeline public handle liveness.
  pipelinePhysical,  \* Per-pipeline physical PSO liveness.
  pipelineModule,    \* Module slot referenced by each pipeline.
  commandPhase,      \* Command lifecycle: idle, recorded, or inFlight.
  commandPipeline,   \* Pipeline slot retained by the command when non-idle.
  lastOutcome        \* Outcome of the latest fallible creation attempt.

coreVars == <<
  libraryLogical,
  libraryPhysical,
  libraryEntries,
  moduleLogical,
  modulePhysical,
  moduleOwnsLibrary,
  pipelineLogical,
  pipelinePhysical,
  pipelineModule,
  commandPhase,
  commandPipeline
>>

vars == <<
  libraryLogical,
  libraryPhysical,
  libraryEntries,
  moduleLogical,
  modulePhysical,
  moduleOwnsLibrary,
  pipelineLogical,
  pipelinePhysical,
  pipelineModule,
  commandPhase,
  commandPipeline,
  lastOutcome
>>

PipelineRetainsModule(m) ==
  \E p \in Pipelines : pipelinePhysical[p] /\ pipelineModule[p] = m

ModuleRetainsLibrary ==
  \E m \in Modules : modulePhysical[m] /\ moduleOwnsLibrary[m]

CommandRetainsPipeline(p) ==
  commandPhase # "idle" /\ commandPipeline = p

Init ==
  /\ libraryLogical = FALSE
  /\ libraryPhysical = FALSE
  /\ libraryEntries = {}
  /\ moduleLogical = [m \in Modules |-> FALSE]
  /\ modulePhysical = [m \in Modules |-> FALSE]
  /\ moduleOwnsLibrary = [m \in Modules |-> FALSE]
  /\ pipelineLogical = [p \in Pipelines |-> FALSE]
  /\ pipelinePhysical = [p \in Pipelines |-> FALSE]
  /\ pipelineModule = [p \in Pipelines |-> 0]
  /\ commandPhase = "idle"
  /\ commandPipeline = 0
  /\ lastOutcome = "none"

\* Library creation validates non-empty bytecode and unique (stage, name) keys.
attemptCreateLibrary(entryA, entryB, bytecodeValid) ==
  /\ ~libraryLogical
  /\ ~libraryPhysical
  /\ LET uniqueNonEmpty == entryA # entryB /\ bytecodeValid
     IN IF uniqueNonEmpty
        THEN /\ libraryLogical' = TRUE
             /\ libraryPhysical' = TRUE
             /\ libraryEntries' = {entryA, entryB}
             /\ lastOutcome' = "accepted"
             /\ UNCHANGED <<
                  moduleLogical,
                  modulePhysical,
                  moduleOwnsLibrary,
                  pipelineLogical,
                  pipelinePhysical,
                  pipelineModule,
                  commandPhase,
                  commandPipeline
                >>
        ELSE /\ lastOutcome' = "rejected"
             /\ UNCHANGED coreVars

attemptCreateSingleEntryLibrary(entry, duplicateKey, bytecodeValid) ==
  /\ ~libraryLogical
  /\ ~libraryPhysical
  /\ LET valid == bytecodeValid /\ ~duplicateKey
     IN IF valid
        THEN /\ libraryLogical' = TRUE
             /\ libraryPhysical' = TRUE
             /\ libraryEntries' = {entry}
             /\ lastOutcome' = "accepted"
             /\ UNCHANGED <<
                  moduleLogical,
                  modulePhysical,
                  moduleOwnsLibrary,
                  pipelineLogical,
                  pipelinePhysical,
                  pipelineModule,
                  commandPhase,
                  commandPipeline
                >>
        ELSE /\ lastOutcome' = "rejected"
             /\ UNCHANGED coreVars

attemptCreateModule(m, source, entry, directValid) ==
  /\ ~moduleLogical[m]
  /\ ~modulePhysical[m]
  /\ LET accepted ==
           \/ (source = "direct" /\ directValid)
           \/ (source = "library" /\ libraryLogical /\ entry \in libraryEntries)
     IN IF accepted
        THEN /\ moduleLogical' = [moduleLogical EXCEPT ![m] = TRUE]
             /\ modulePhysical' = [modulePhysical EXCEPT ![m] = TRUE]
             /\ moduleOwnsLibrary' = [moduleOwnsLibrary EXCEPT ![m] = (source = "library")]
             /\ lastOutcome' = "accepted"
             /\ UNCHANGED <<
                  libraryLogical,
                  libraryPhysical,
                  libraryEntries,
                  pipelineLogical,
                  pipelinePhysical,
                  pipelineModule,
                  commandPhase,
                  commandPipeline
                >>
        ELSE /\ lastOutcome' = "rejected"
             /\ UNCHANGED coreVars

destroyLibrary ==
  /\ libraryLogical
  /\ libraryLogical' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       libraryPhysical,
       libraryEntries,
       moduleLogical,
       modulePhysical,
       moduleOwnsLibrary,
       pipelineLogical,
       pipelinePhysical,
       pipelineModule,
       commandPhase,
       commandPipeline
     >>

collectLibrary ==
  /\ libraryPhysical
  /\ ~libraryLogical
  /\ ~ModuleRetainsLibrary
  /\ libraryPhysical' = FALSE
  /\ libraryEntries' = {}
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       libraryLogical,
       moduleLogical,
       modulePhysical,
       moduleOwnsLibrary,
       pipelineLogical,
       pipelinePhysical,
       pipelineModule,
       commandPhase,
       commandPipeline
     >>

attemptCreatePipeline(p, m) ==
  /\ ~pipelineLogical[p]
  /\ ~pipelinePhysical[p]
  /\ IF moduleLogical[m]
     THEN /\ pipelineLogical' = [pipelineLogical EXCEPT ![p] = TRUE]
          /\ pipelinePhysical' = [pipelinePhysical EXCEPT ![p] = TRUE]
          /\ pipelineModule' = [pipelineModule EXCEPT ![p] = m]
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               libraryLogical,
               libraryPhysical,
               libraryEntries,
               moduleLogical,
               modulePhysical,
               moduleOwnsLibrary,
               commandPhase,
               commandPipeline
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

destroyModule(m) ==
  /\ moduleLogical[m]
  /\ moduleLogical' = [moduleLogical EXCEPT ![m] = FALSE]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       libraryLogical,
       libraryPhysical,
       libraryEntries,
       modulePhysical,
       moduleOwnsLibrary,
       pipelineLogical,
       pipelinePhysical,
       pipelineModule,
       commandPhase,
       commandPipeline
     >>

collectModule(m) ==
  /\ modulePhysical[m]
  /\ ~moduleLogical[m]
  /\ ~PipelineRetainsModule(m)
  /\ modulePhysical' = [modulePhysical EXCEPT ![m] = FALSE]
  /\ moduleOwnsLibrary' = [moduleOwnsLibrary EXCEPT ![m] = FALSE]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       libraryLogical,
       libraryPhysical,
       libraryEntries,
       moduleLogical,
       pipelineLogical,
       pipelinePhysical,
       pipelineModule,
       commandPhase,
       commandPipeline
     >>

recordPipeline(p) ==
  /\ commandPhase = "idle"
  /\ pipelineLogical[p]
  /\ commandPhase' = "recorded"
  /\ commandPipeline' = p
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       libraryLogical,
       libraryPhysical,
       libraryEntries,
       moduleLogical,
       modulePhysical,
       moduleOwnsLibrary,
       pipelineLogical,
       pipelinePhysical,
       pipelineModule
     >>

abandonRecording ==
  /\ commandPhase = "recorded"
  /\ commandPhase' = "idle"
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       libraryLogical,
       libraryPhysical,
       libraryEntries,
       moduleLogical,
       modulePhysical,
       moduleOwnsLibrary,
       pipelineLogical,
       pipelinePhysical,
       pipelineModule,
       commandPipeline
     >>

submit ==
  /\ commandPhase = "recorded"
  /\ commandPhase' = "inFlight"
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       libraryLogical,
       libraryPhysical,
       libraryEntries,
       moduleLogical,
       modulePhysical,
       moduleOwnsLibrary,
       pipelineLogical,
       pipelinePhysical,
       pipelineModule,
       commandPipeline
     >>

complete ==
  /\ commandPhase = "inFlight"
  /\ commandPhase' = "idle"
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       libraryLogical,
       libraryPhysical,
       libraryEntries,
       moduleLogical,
       modulePhysical,
       moduleOwnsLibrary,
       pipelineLogical,
       pipelinePhysical,
       pipelineModule,
       commandPipeline
     >>

destroyPipeline(p) ==
  /\ pipelineLogical[p]
  /\ pipelineLogical' = [pipelineLogical EXCEPT ![p] = FALSE]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       libraryLogical,
       libraryPhysical,
       libraryEntries,
       moduleLogical,
       modulePhysical,
       moduleOwnsLibrary,
       pipelinePhysical,
       pipelineModule,
       commandPhase,
       commandPipeline
     >>

collectPipeline(p) ==
  /\ pipelinePhysical[p]
  /\ ~pipelineLogical[p]
  /\ ~CommandRetainsPipeline(p)
  /\ pipelinePhysical' = [pipelinePhysical EXCEPT ![p] = FALSE]
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       libraryLogical,
       libraryPhysical,
       libraryEntries,
       moduleLogical,
       modulePhysical,
       moduleOwnsLibrary,
       pipelineLogical,
       pipelineModule,
       commandPhase,
       commandPipeline
     >>

Next ==
  \/ \E a, b \in Entries, ok \in BOOLEAN : attemptCreateLibrary(a, b, ok)
  \/ \E e \in Entries, dup, ok \in BOOLEAN : attemptCreateSingleEntryLibrary(e, dup, ok)
  \/ \E m \in Modules, s \in Sources, e \in Entries, ok \in BOOLEAN :
       attemptCreateModule(m, s, e, ok)
  \/ destroyLibrary
  \/ collectLibrary
  \/ \E p \in Pipelines, m \in Modules : attemptCreatePipeline(p, m)
  \/ \E m \in Modules : destroyModule(m)
  \/ \E m \in Modules : collectModule(m)
  \/ \E p \in Pipelines : recordPipeline(p)
  \/ abandonRecording
  \/ submit
  \/ complete
  \/ \E p \in Pipelines : destroyPipeline(p)
  \/ \E p \in Pipelines : collectPipeline(p)

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ libraryLogical \in BOOLEAN
  /\ libraryPhysical \in BOOLEAN
  /\ libraryEntries \in SUBSET Entries
  /\ moduleLogical \in [Modules -> BOOLEAN]
  /\ modulePhysical \in [Modules -> BOOLEAN]
  /\ moduleOwnsLibrary \in [Modules -> BOOLEAN]
  /\ pipelineLogical \in [Pipelines -> BOOLEAN]
  /\ pipelinePhysical \in [Pipelines -> BOOLEAN]
  /\ pipelineModule \in [Pipelines -> Modules]
  /\ commandPhase \in CommandPhases
  /\ commandPipeline \in Pipelines
  /\ lastOutcome \in Outcomes

LogicalObjectsArePhysical ==
  /\ (libraryLogical => libraryPhysical)
  /\ \A m \in Modules : moduleLogical[m] => modulePhysical[m]
  /\ \A p \in Pipelines : pipelineLogical[p] => pipelinePhysical[p]

LibraryHasValidEntries ==
  /\ (libraryPhysical => libraryEntries # {})
  /\ (~libraryPhysical => libraryEntries = {})

ConsumersRetainTheirSources ==
  /\ \A m \in Modules :
       (moduleOwnsLibrary[m] => modulePhysical[m])
       /\ (modulePhysical[m] /\ moduleOwnsLibrary[m] => libraryPhysical)
  /\ \A p \in Pipelines :
       pipelinePhysical[p] => modulePhysical[pipelineModule[p]]
  /\ (commandPhase # "idle" => pipelinePhysical[commandPipeline])

Safety ==
  /\ LogicalObjectsArePhysical
  /\ LibraryHasValidEntries
  /\ ConsumersRetainTheirSources

StepSafety ==
  [][/\ (lastOutcome' = "rejected" => coreVars' = coreVars)
     /\ (~libraryLogical /\ ~libraryPhysical' =>
           \A m \in Modules : ~(~modulePhysical[m] /\ modulePhysical'[m] /\ moduleOwnsLibrary'[m]))]_vars

=============================================================================
