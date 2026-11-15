--------------------- MODULE RayTracingPipelineLibrary ----------------------
(***************************************************************************)
(* Ray-tracing pipeline library compilation, multi-library linking         *)
(* (rejecting stale, duplicate, recursion-invalid, or interface-mismatched *)
(* libraries atomically), deterministic library-before-local shader-group  *)
(* ordering, ray-generation group presence, and transitive retention.      *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12RayTracingPipeline.cpp                               *)
(*     (createRayTracingPipelineLibrary, destroyRayTracingPipelineLibrary, *)
(*     createRayTracingPipeline, addLibrariesToLink,                       *)
(*     linkRayTracingPipeline)                                             *)
(*   - src/D3D12/D3D12Pipeline.cpp (destroyPipeline)                       *)
(*   - src/Device.cpp (portable ray-tracing library/pipeline forwarding)   *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Two distinct library slots, two layouts, two payload/attribute        *)
(*   interfaces, candidate link sequences (including 2-library orders and  *)
(*   duplicates), recursion depths {0, 1, 2}, and command retention of the *)
(*   linked pipeline cover multi-library linking and physical lifetimes.   *)
(***************************************************************************)

EXTENDS Naturals, Sequences

Libraries == {1, 2}
LocalTag == 3
GroupTags == Libraries \cup {LocalTag}
Layouts == {0, 1}
Interfaces == {0, 1}
RecursionDepths == {0, 1, 2}
MaxSupportedRecursion == 1
LinkSequences == {<<>>, <<1>>, <<2>>, <<1, 2>>, <<2, 1>>, <<1, 1>>}
Operations == {
  "none", "createLibrary", "destroyLibrary",
  "link", "retainPipeline", "releasePipeline", "destroyPipeline"
}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
  libLogical,        \* Public RayTracingPipelineLibraryHandle liveness
  libPhysical,       \* Native state-object collection liveness per library
  libLayout,         \* Pipeline layout captured by each library
  libInterface,      \* Payload/attribute interface captured by each library
  libHasRayGen,      \* Whether each library exports a ray-generation group
  pipelineLogical,   \* Public linked PipelineHandle liveness
  pipelinePhysical,  \* Native linked state-object liveness
  pipelineRetained,  \* Command/in-flight retention of the linked pipeline
  pipelineLayout,    \* Layout of the linked pipeline
  pipelineInterface, \* Payload/attribute interface of the linked pipeline
  linkedLibraries,   \* Set of libraries retained by the linked pipeline
  groupOrder,        \* Flattened shader-group provenance sequence
  lastOp,            \* Most recent operation kind
  lastOutcome        \* Outcome of most recent operation

coreVars == <<
  libLogical, libPhysical, libLayout, libInterface, libHasRayGen,
  pipelineLogical, pipelinePhysical, pipelineRetained,
  pipelineLayout, pipelineInterface, linkedLibraries, groupOrder
>>

vars == <<
  libLogical, libPhysical, libLayout, libInterface, libHasRayGen,
  pipelineLogical, pipelinePhysical, pipelineRetained,
  pipelineLayout, pipelineInterface, linkedLibraries, groupOrder,
  lastOp, lastOutcome
>>

SeqElements(seq) ==
  {seq[i] : i \in 1..Len(seq)}

HasNoDuplicates(seq) ==
  \A i, j \in 1..Len(seq) : i # j => seq[i] # seq[j]

LibsCompatible(seq, layout, iface) ==
  \A i \in 1..Len(seq) :
    LET lib == seq[i] IN
    /\ libLogical[lib]
    /\ libPhysical[lib]
    /\ libLayout[lib] = layout
    /\ libInterface[lib] = iface

HasRayGenAcrossProgram(seq, includeLocal, localHasRayGen) ==
  \/ (\E i \in 1..Len(seq) : libHasRayGen[seq[i]])
  \/ (includeLocal /\ localHasRayGen)

CanLink(descValid, seq, includeLocal, localHasRayGen, layout, iface, recursion) ==
  /\ descValid
  /\ ~pipelinePhysical
  /\ recursion > 0
  /\ recursion <= MaxSupportedRecursion
  /\ (Len(seq) > 0 \/ includeLocal)
  /\ HasNoDuplicates(seq)
  /\ LibsCompatible(seq, layout, iface)
  /\ HasRayGenAcrossProgram(seq, includeLocal, localHasRayGen)

FlattenedOrder(seq, includeLocal) ==
  IF includeLocal THEN Append(seq, LocalTag) ELSE seq

ValidOrders ==
  {FlattenedOrder(seq, inc) : seq \in LinkSequences, inc \in BOOLEAN}

Init ==
  /\ libLogical = [lib \in Libraries |-> FALSE]
  /\ libPhysical = [lib \in Libraries |-> FALSE]
  /\ libLayout = [lib \in Libraries |-> 0]
  /\ libInterface = [lib \in Libraries |-> 0]
  /\ libHasRayGen = [lib \in Libraries |-> FALSE]
  /\ pipelineLogical = FALSE
  /\ pipelinePhysical = FALSE
  /\ pipelineRetained = FALSE
  /\ pipelineLayout = 0
  /\ pipelineInterface = 0
  /\ linkedLibraries = {}
  /\ groupOrder = <<>>
  /\ lastOp = "none"
  /\ lastOutcome = "none"

AttemptCreateLibrary(lib, descValid, layout, iface, hasRayGen) ==
  /\ ~libPhysical[lib]
  /\ IF descValid
     THEN /\ libLogical' = [libLogical EXCEPT ![lib] = TRUE]
          /\ libPhysical' = [libPhysical EXCEPT ![lib] = TRUE]
          /\ libLayout' = [libLayout EXCEPT ![lib] = layout]
          /\ libInterface' = [libInterface EXCEPT ![lib] = iface]
          /\ libHasRayGen' = [libHasRayGen EXCEPT ![lib] = hasRayGen]
          /\ UNCHANGED <<
               pipelineLogical, pipelinePhysical, pipelineRetained,
               pipelineLayout, pipelineInterface, linkedLibraries, groupOrder
             >>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "createLibrary"

DestroyLibrary(lib) ==
  /\ libLogical[lib]
  /\ libLogical' = [libLogical EXCEPT ![lib] = FALSE]
  /\ libPhysical' =
       [libPhysical EXCEPT ![lib] = (pipelinePhysical /\ lib \in linkedLibraries)]
  /\ UNCHANGED <<
       libLayout, libInterface, libHasRayGen,
       pipelineLogical, pipelinePhysical, pipelineRetained,
       pipelineLayout, pipelineInterface, linkedLibraries, groupOrder
     >>
  /\ lastOp' = "destroyLibrary"
  /\ lastOutcome' = "accepted"

AttemptLink(descValid, seq, includeLocal, localHasRayGen, layout, iface, recursion) ==
  LET libsUsed == SeqElements(seq) IN
  /\ ~pipelinePhysical
  /\ IF CanLink(descValid, seq, includeLocal, localHasRayGen, layout, iface, recursion)
     THEN /\ pipelineLogical' = TRUE
          /\ pipelinePhysical' = TRUE
          /\ pipelineRetained' = FALSE
          /\ pipelineLayout' = layout
          /\ pipelineInterface' = iface
          /\ linkedLibraries' = libsUsed
          /\ groupOrder' = FlattenedOrder(seq, includeLocal)
          /\ UNCHANGED <<libLogical, libPhysical, libLayout, libInterface, libHasRayGen>>
          /\ lastOutcome' = "accepted"
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
  /\ lastOp' = "link"

RetainPipeline ==
  /\ pipelineLogical
  /\ pipelineRetained' = TRUE
  /\ UNCHANGED <<
       libLogical, libPhysical, libLayout, libInterface, libHasRayGen,
       pipelineLogical, pipelinePhysical,
       pipelineLayout, pipelineInterface, linkedLibraries, groupOrder
     >>
  /\ lastOp' = "retainPipeline"
  /\ lastOutcome' = "accepted"

DestroyPipeline ==
  LET staysAlive == pipelineRetained IN
  /\ pipelineLogical
  /\ pipelineLogical' = FALSE
  /\ pipelinePhysical' = staysAlive
  /\ linkedLibraries' = IF staysAlive THEN linkedLibraries ELSE {}
  /\ groupOrder' = IF staysAlive THEN groupOrder ELSE <<>>
  /\ libPhysical' =
       [lib \in Libraries |->
          libLogical[lib] \/ (staysAlive /\ lib \in linkedLibraries)]
  /\ UNCHANGED <<
       libLogical, libLayout, libInterface, libHasRayGen,
       pipelineRetained, pipelineLayout, pipelineInterface
     >>
  /\ lastOp' = "destroyPipeline"
  /\ lastOutcome' = "accepted"

ReleasePipeline ==
  LET staysAlive == pipelineLogical IN
  /\ pipelineRetained
  /\ pipelineRetained' = FALSE
  /\ pipelinePhysical' = staysAlive
  /\ linkedLibraries' = IF staysAlive THEN linkedLibraries ELSE {}
  /\ groupOrder' = IF staysAlive THEN groupOrder ELSE <<>>
  /\ libPhysical' =
       [lib \in Libraries |->
          libLogical[lib] \/ (staysAlive /\ lib \in linkedLibraries)]
  /\ UNCHANGED <<
       libLogical, libLayout, libInterface, libHasRayGen,
       pipelineLogical, pipelineLayout, pipelineInterface
     >>
  /\ lastOp' = "releasePipeline"
  /\ lastOutcome' = "accepted"

LocalChoices == {<<FALSE, FALSE>>, <<TRUE, FALSE>>, <<TRUE, TRUE>>}

Next ==
  \/ \E lib \in Libraries :
       \/ AttemptCreateLibrary(lib, FALSE, 0, 0, FALSE)
       \/ \E lay \in Layouts, iface \in Interfaces, rg \in BOOLEAN :
            AttemptCreateLibrary(lib, TRUE, lay, iface, rg)
  \/ \E lib \in Libraries : DestroyLibrary(lib)
  \/ AttemptLink(FALSE, <<>>, FALSE, FALSE, 0, 0, 1)
  \/ \E seq \in LinkSequences, lc \in LocalChoices,
        lay \in Layouts, iface \in Interfaces, rec \in RecursionDepths :
       AttemptLink(TRUE, seq, lc[1], lc[2], lay, iface, rec)
  \/ RetainPipeline
  \/ DestroyPipeline
  \/ ReleasePipeline

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ libLogical \in [Libraries -> BOOLEAN]
  /\ libPhysical \in [Libraries -> BOOLEAN]
  /\ libLayout \in [Libraries -> Layouts]
  /\ libInterface \in [Libraries -> Interfaces]
  /\ libHasRayGen \in [Libraries -> BOOLEAN]
  /\ pipelineLogical \in BOOLEAN
  /\ pipelinePhysical \in BOOLEAN
  /\ pipelineRetained \in BOOLEAN
  /\ pipelineLayout \in Layouts
  /\ pipelineInterface \in Interfaces
  /\ linkedLibraries \in SUBSET Libraries
  /\ groupOrder \in ValidOrders
  /\ lastOp \in Operations
  /\ lastOutcome \in Outcomes

PipelineLifetimeExact ==
  pipelinePhysical = (pipelineLogical \/ pipelineRetained)

LibraryLifetimeExact ==
  \A lib \in Libraries :
    libPhysical[lib] =
      (libLogical[lib] \/ (pipelinePhysical /\ lib \in linkedLibraries))

DeadPipelineClearsGroupMetadata ==
  ~pipelinePhysical =>
    /\ linkedLibraries = {}
    /\ groupOrder = <<>>

LinkedLibrariesMatchInterfaceAndHaveNoDuplicates ==
  pipelinePhysical =>
    /\ Len(groupOrder) > 0
    /\ HasNoDuplicates(groupOrder)
    /\ linkedLibraries = (SeqElements(groupOrder) \cap Libraries)
    /\ \A lib \in linkedLibraries :
         /\ libPhysical[lib]
         /\ libLayout[lib] = pipelineLayout
         /\ libInterface[lib] = pipelineInterface
    /\ \A i \in 1..Len(groupOrder) :
         groupOrder[i] = LocalTag => i = Len(groupOrder)

Safety ==
  /\ PipelineLifetimeExact
  /\ LibraryLifetimeExact
  /\ DeadPipelineClearsGroupMetadata
  /\ LinkedLibrariesMatchInterfaceAndHaveNoDuplicates

StepSafety ==
  [][
    /\ (lastOutcome' = "rejected" => coreVars' = coreVars)
    /\ (\A lib \in Libraries :
          libPhysical[lib] /\ libPhysical'[lib] =>
            /\ libLayout'[lib] = libLayout[lib]
            /\ libInterface'[lib] = libInterface[lib]
            /\ libHasRayGen'[lib] = libHasRayGen[lib])
    /\ (~pipelinePhysical /\ pipelinePhysical' =>
          /\ lastOp' = "link"
          /\ lastOutcome' = "accepted"
          /\ \A lib \in linkedLibraries' :
               /\ libLogical[lib]
               /\ libLayout[lib] = pipelineLayout'
               /\ libInterface[lib] = pipelineInterface'
          /\ ((\E lib \in linkedLibraries' : libHasRayGen[lib])
              \/ LocalTag \in SeqElements(groupOrder')))
  ]_vars

=============================================================================
