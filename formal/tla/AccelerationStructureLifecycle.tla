------------------- MODULE AccelerationStructureLifecycle -------------------
(***************************************************************************)
(* DXR capability gating, acceleration-structure handle creation,          *)
(* single-assignment immutable BLAS/TLAS version publication, exact BLAS   *)
(* version retention across handle recreation, and failure atomicity.      *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12Device.cpp (detectRayTracing: any DXR tier with      *)
(*     shader model 6.3 sets DeviceCapabilities::rayTracing)               *)
(*   - src/D3D12/D3D12AccelerationStructure.cpp (ResourceManager::         *)
(*     createAccelerationStructure, destroyAccelerationStructure,          *)
(*     buildBottomLevelAccelerationStructure,                              *)
(*     buildTopLevelAccelerationStructure)                                 *)
(*   - src/D3D12/D3D12Binding.cpp (shader descriptor version capture)      *)
(*   - src/D3D12/D3D12Queue.cpp (command and timeline retention)           *)
(*                                                                         *)
(* Contract modeled:                                                       *)
(*   - Acceleration-structure creation requires native DXR support.        *)
(*   - Each AS handle accepts at most one initial build; rebuilding an     *)
(*     already-built immutable AS or building a TLAS from an unbuilt BLAS  *)
(*     rejects without mutating any lifecycle state.                       *)
(*   - Two distinct BLAS versions (1..2) can be created sequentially by    *)
(*     destroying the first BLAS handle and creating/building a second.    *)
(*     A TLAS built against BLAS version 1 retains exact BLAS version 1    *)
(*     even after BLAS 1 is logically destroyed and BLAS 2 becomes the     *)
(*     current live BLAS handle.                                           *)
(*   - Design rule RetainExactBlasVersion (TRUE in the design) retains the *)
(*     exact captured BLAS version in TLAS dependencies rather than only   *)
(*     the currently live BLAS handle.                                     *)
(*                                                                         *)
(* Trusted boundary / abstracted away:                                     *)
(*   - Geometry vertex/index contents and native BVH traversal math.       *)
(***************************************************************************)
EXTENDS Naturals

CONSTANT RetainExactBlasVersion

ASSUME RetainExactBlasVersion \in BOOLEAN

BlasVersions == 1..2
OptionalBlas == 0..2
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    rayTracingSupported,    \* native DXR support (tier 1.0+, SM 6.3) on the adapter
    nextBlasVersion,        \* next immutable BLAS version ID to mint (1..3)
    blasLogical,            \* current public BLAS handle is alive
    currentBlasVersion,     \* version published on current BLAS handle (0 if unbuilt)
    blasPhysical,           \* per-version physical BLAS storage liveness
    tlasLogical,            \* public TLAS handle is alive
    tlasBuilt,              \* public TLAS handle has an immutable version
    tlasPhysical,           \* physical TLAS storage liveness
    tlasCapturedBlas,       \* exact BLAS version captured by the TLAS (0 if none)
    bindingLogical,         \* shader binding set referencing the TLAS is alive
    recordedBlasBuild,      \* BLAS version with a recorded build command (0 if none)
    recordedTlasBuild,      \* TRUE while a TLAS build command is recorded
    recordedTraversal,      \* TRUE while a ray-tracing dispatch is recorded
    inFlightBlasBuild,      \* BLAS version with an in-flight build command (0 if none)
    inFlightTlasBuild,      \* TRUE while a TLAS build command is in flight
    inFlightTraversal,      \* TRUE while a ray-tracing dispatch is in flight
    lastOutcome             \* outcome of the most recent fallible API call

coreVars == <<rayTracingSupported, nextBlasVersion, blasLogical,
              currentBlasVersion, blasPhysical, tlasLogical, tlasBuilt,
              tlasPhysical, tlasCapturedBlas, bindingLogical,
              recordedBlasBuild, recordedTlasBuild, recordedTraversal,
              inFlightBlasBuild, inFlightTlasBuild, inFlightTraversal>>
vars == <<rayTracingSupported, nextBlasVersion, blasLogical,
          currentBlasVersion, blasPhysical, tlasLogical, tlasBuilt,
          tlasPhysical, tlasCapturedBlas, bindingLogical,
          recordedBlasBuild, recordedTlasBuild, recordedTraversal,
          inFlightBlasBuild, inFlightTlasBuild, inFlightTraversal,
          lastOutcome>>

CommandIdle ==
    /\ recordedBlasBuild = 0
    /\ ~recordedTlasBuild
    /\ ~recordedTraversal

InFlightIdle ==
    /\ inFlightBlasBuild = 0
    /\ ~inFlightTlasBuild
    /\ ~inFlightTraversal

TypeOK ==
    /\ rayTracingSupported \in BOOLEAN
    /\ nextBlasVersion \in 1..3
    /\ blasLogical \in BOOLEAN
    /\ currentBlasVersion \in OptionalBlas
    /\ blasPhysical \in [BlasVersions -> BOOLEAN]
    /\ tlasLogical \in BOOLEAN
    /\ tlasBuilt \in BOOLEAN
    /\ tlasPhysical \in BOOLEAN
    /\ tlasCapturedBlas \in OptionalBlas
    /\ bindingLogical \in BOOLEAN
    /\ recordedBlasBuild \in OptionalBlas
    /\ recordedTlasBuild \in BOOLEAN
    /\ recordedTraversal \in BOOLEAN
    /\ inFlightBlasBuild \in OptionalBlas
    /\ inFlightTlasBuild \in BOOLEAN
    /\ inFlightTraversal \in BOOLEAN
    /\ lastOutcome \in Outcomes

Init ==
    /\ rayTracingSupported \in BOOLEAN
    /\ nextBlasVersion = 1
    /\ blasLogical = FALSE
    /\ currentBlasVersion = 0
    /\ blasPhysical = [v \in BlasVersions |-> FALSE]
    /\ tlasLogical = FALSE
    /\ tlasBuilt = FALSE
    /\ tlasPhysical = FALSE
    /\ tlasCapturedBlas = 0
    /\ bindingLogical = FALSE
    /\ recordedBlasBuild = 0
    /\ recordedTlasBuild = FALSE
    /\ recordedTraversal = FALSE
    /\ inFlightBlasBuild = 0
    /\ inFlightTlasBuild = FALSE
    /\ inFlightTraversal = FALSE
    /\ lastOutcome = "none"

AttemptCreateBlas ==
    /\ ~blasLogical
    /\ nextBlasVersion <= 2
    /\ IF rayTracingSupported
       THEN /\ blasLogical' = TRUE
            /\ currentBlasVersion' = 0
            /\ lastOutcome' = "accepted"
            /\ UNCHANGED <<rayTracingSupported, nextBlasVersion, blasPhysical,
                           tlasLogical, tlasBuilt, tlasPhysical, tlasCapturedBlas,
                           bindingLogical, recordedBlasBuild, recordedTlasBuild,
                           recordedTraversal, inFlightBlasBuild,
                           inFlightTlasBuild, inFlightTraversal>>
       ELSE /\ lastOutcome' = "rejected"
            /\ UNCHANGED coreVars

AttemptBuildBlas ==
    /\ recordedBlasBuild = 0
    /\ nextBlasVersion <= 2
    /\ IF blasLogical /\ currentBlasVersion = 0
       THEN LET v == nextBlasVersion IN
            /\ nextBlasVersion' = v + 1
            /\ currentBlasVersion' = v
            /\ blasPhysical' = [blasPhysical EXCEPT ![v] = TRUE]
            /\ recordedBlasBuild' = v
            /\ lastOutcome' = "accepted"
            /\ UNCHANGED <<rayTracingSupported, blasLogical, tlasLogical,
                           tlasBuilt, tlasPhysical, tlasCapturedBlas,
                           bindingLogical, recordedTlasBuild, recordedTraversal,
                           inFlightBlasBuild, inFlightTlasBuild, inFlightTraversal>>
       ELSE /\ lastOutcome' = "rejected"
            /\ UNCHANGED coreVars

AttemptCreateTlas ==
    /\ ~tlasLogical
    /\ ~tlasPhysical
    /\ IF rayTracingSupported
       THEN /\ tlasLogical' = TRUE
            /\ tlasBuilt' = FALSE
            /\ tlasCapturedBlas' = 0
            /\ lastOutcome' = "accepted"
            /\ UNCHANGED <<rayTracingSupported, nextBlasVersion, blasLogical,
                           currentBlasVersion, blasPhysical, tlasPhysical,
                           bindingLogical, recordedBlasBuild, recordedTlasBuild,
                           recordedTraversal, inFlightBlasBuild,
                           inFlightTlasBuild, inFlightTraversal>>
       ELSE /\ lastOutcome' = "rejected"
            /\ UNCHANGED coreVars

AttemptBuildTlas ==
    /\ ~recordedTlasBuild
    /\ IF tlasLogical /\ ~tlasBuilt /\ blasLogical
          /\ currentBlasVersion # 0 /\ blasPhysical[currentBlasVersion]
       THEN /\ tlasBuilt' = TRUE
            /\ tlasPhysical' = TRUE
            /\ tlasCapturedBlas' = currentBlasVersion
            /\ recordedTlasBuild' = TRUE
            /\ lastOutcome' = "accepted"
            /\ UNCHANGED <<rayTracingSupported, nextBlasVersion, blasLogical,
                           currentBlasVersion, blasPhysical, tlasLogical,
                           bindingLogical, recordedBlasBuild, recordedTraversal,
                           inFlightBlasBuild, inFlightTlasBuild, inFlightTraversal>>
       ELSE /\ lastOutcome' = "rejected"
            /\ UNCHANGED coreVars

CreateBinding ==
    /\ ~bindingLogical
    /\ tlasLogical /\ tlasBuilt /\ tlasPhysical
    /\ bindingLogical' = TRUE
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<rayTracingSupported, nextBlasVersion, blasLogical,
                   currentBlasVersion, blasPhysical, tlasLogical, tlasBuilt,
                   tlasPhysical, tlasCapturedBlas, recordedBlasBuild,
                   recordedTlasBuild, recordedTraversal, inFlightBlasBuild,
                   inFlightTlasBuild, inFlightTraversal>>

RecordTraversal ==
    /\ bindingLogical
    /\ tlasPhysical /\ tlasCapturedBlas # 0 /\ blasPhysical[tlasCapturedBlas]
    /\ ~recordedTraversal
    /\ recordedTraversal' = TRUE
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<rayTracingSupported, nextBlasVersion, blasLogical,
                   currentBlasVersion, blasPhysical, tlasLogical, tlasBuilt,
                   tlasPhysical, tlasCapturedBlas, bindingLogical,
                   recordedBlasBuild, recordedTlasBuild, inFlightBlasBuild,
                   inFlightTlasBuild, inFlightTraversal>>

DestroyBlas ==
    /\ blasLogical
    /\ blasLogical' = FALSE
    /\ currentBlasVersion' = 0
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<rayTracingSupported, nextBlasVersion, blasPhysical,
                   tlasLogical, tlasBuilt, tlasPhysical, tlasCapturedBlas,
                   bindingLogical, recordedBlasBuild, recordedTlasBuild,
                   recordedTraversal, inFlightBlasBuild, inFlightTlasBuild,
                   inFlightTraversal>>

DestroyTlas ==
    /\ tlasLogical
    /\ tlasLogical' = FALSE
    /\ tlasBuilt' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<rayTracingSupported, nextBlasVersion, blasLogical,
                   currentBlasVersion, blasPhysical, tlasPhysical,
                   tlasCapturedBlas, bindingLogical, recordedBlasBuild,
                   recordedTlasBuild, recordedTraversal, inFlightBlasBuild,
                   inFlightTlasBuild, inFlightTraversal>>

DestroyBinding ==
    /\ bindingLogical
    /\ bindingLogical' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<rayTracingSupported, nextBlasVersion, blasLogical,
                   currentBlasVersion, blasPhysical, tlasLogical, tlasBuilt,
                   tlasPhysical, tlasCapturedBlas, recordedBlasBuild,
                   recordedTlasBuild, recordedTraversal, inFlightBlasBuild,
                   inFlightTlasBuild, inFlightTraversal>>

Submit ==
    /\ ~CommandIdle
    /\ InFlightIdle
    /\ inFlightBlasBuild' = recordedBlasBuild
    /\ inFlightTlasBuild' = recordedTlasBuild
    /\ inFlightTraversal' = recordedTraversal
    /\ recordedBlasBuild' = 0
    /\ recordedTlasBuild' = FALSE
    /\ recordedTraversal' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<rayTracingSupported, nextBlasVersion, blasLogical,
                   currentBlasVersion, blasPhysical, tlasLogical, tlasBuilt,
                   tlasPhysical, tlasCapturedBlas, bindingLogical>>

Complete ==
    /\ ~InFlightIdle
    /\ inFlightBlasBuild' = 0
    /\ inFlightTlasBuild' = FALSE
    /\ inFlightTraversal' = FALSE
    /\ lastOutcome' = "none"
    /\ UNCHANGED <<rayTracingSupported, nextBlasVersion, blasLogical,
                   currentBlasVersion, blasPhysical, tlasLogical, tlasBuilt,
                   tlasPhysical, tlasCapturedBlas, bindingLogical,
                   recordedBlasBuild, recordedTlasBuild, recordedTraversal>>

Collect ==
    LET keepTlas ==
            \/ (tlasLogical /\ tlasBuilt)
            \/ bindingLogical
            \/ recordedTlasBuild
            \/ recordedTraversal
            \/ inFlightTlasBuild
            \/ inFlightTraversal
        retainedByTlas(v) ==
            /\ keepTlas
            /\ IF RetainExactBlasVersion
               THEN tlasCapturedBlas = v
               ELSE currentBlasVersion = v
        keepBlas(v) ==
            \/ (blasLogical /\ currentBlasVersion = v)
            \/ recordedBlasBuild = v
            \/ inFlightBlasBuild = v
            \/ retainedByTlas(v)
    IN /\ blasPhysical' = [v \in BlasVersions |-> blasPhysical[v] /\ keepBlas(v)]
       /\ tlasPhysical' = keepTlas
       /\ tlasCapturedBlas' = IF keepTlas THEN tlasCapturedBlas ELSE 0
       /\ lastOutcome' = "none"
       /\ UNCHANGED <<rayTracingSupported, nextBlasVersion, blasLogical,
                      currentBlasVersion, tlasLogical, tlasBuilt,
                      bindingLogical, recordedBlasBuild, recordedTlasBuild,
                      recordedTraversal, inFlightBlasBuild, inFlightTlasBuild,
                      inFlightTraversal>>

Next ==
    \/ AttemptCreateBlas
    \/ AttemptBuildBlas
    \/ AttemptCreateTlas
    \/ AttemptBuildTlas
    \/ CreateBinding
    \/ RecordTraversal
    \/ DestroyBlas
    \/ DestroyTlas
    \/ DestroyBinding
    \/ Submit
    \/ Complete
    \/ Collect

Spec == Init /\ [][Next]_vars

CapabilityGatesCreation ==
    ~rayTracingSupported =>
        /\ ~blasLogical
        /\ ~tlasLogical
        /\ nextBlasVersion = 1

LogicalBuiltStructuresArePhysical ==
    /\ (blasLogical /\ currentBlasVersion # 0) => blasPhysical[currentBlasVersion]
    /\ (tlasLogical /\ tlasBuilt) => tlasPhysical

TlasRetainsExactBlasVersion ==
    tlasPhysical =>
        /\ tlasCapturedBlas \in BlasVersions
        /\ blasPhysical[tlasCapturedBlas]

BindingsRetainCompleteTlasGraphs ==
    bindingLogical =>
        /\ tlasPhysical
        /\ tlasCapturedBlas \in BlasVersions
        /\ blasPhysical[tlasCapturedBlas]

BuildsAndTraversalsRetainGraphs ==
    /\ (recordedBlasBuild # 0 => blasPhysical[recordedBlasBuild])
    /\ (inFlightBlasBuild # 0 => blasPhysical[inFlightBlasBuild])
    /\ (recordedTlasBuild => tlasPhysical /\ tlasCapturedBlas # 0 /\ blasPhysical[tlasCapturedBlas])
    /\ (inFlightTlasBuild => tlasPhysical /\ tlasCapturedBlas # 0 /\ blasPhysical[tlasCapturedBlas])
    /\ (recordedTraversal => tlasPhysical /\ tlasCapturedBlas # 0 /\ blasPhysical[tlasCapturedBlas])
    /\ (inFlightTraversal => tlasPhysical /\ tlasCapturedBlas # 0 /\ blasPhysical[tlasCapturedBlas])

Safety ==
    /\ CapabilityGatesCreation
    /\ LogicalBuiltStructuresArePhysical
    /\ TlasRetainsExactBlasVersion
    /\ BindingsRetainCompleteTlasGraphs
    /\ BuildsAndTraversalsRetainGraphs

StepSafety ==
    [][/\ (lastOutcome' = "rejected" /\ coreVars' # coreVars) => FALSE
       /\ nextBlasVersion' >= nextBlasVersion
       /\ rayTracingSupported' = rayTracingSupported]_vars
=============================================================================
