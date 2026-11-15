--------------------- MODULE SparseTextureAccessStatus ----------------------
(***************************************************************************)
(* Formal safety model for the indivisible Tier 2 sparse-texture and       *)
(* HLSL residency-access-status contract:                                  *)
(*   - src/D3D12/D3D12Device.cpp (sparseTextures capability discovery)     *)
(*   - src/D3D12/D3D12SparseAndFeedback.cpp (createSparseTexture,          *)
(*     enqueueSparseTextureMappings, captureSparseTextureBinding)          *)
(*   - src/D3D12/D3D12Binding.cpp (command-time mapping capture),          *)
(*     src/D3D12/D3D12Queue.cpp (submission retention and retirement)      *)
(*   - tests/d3d12/memory/SparseTextureShaderContractTests.cpp (WARP       *)
(*     CheckAccessFullyMapped observation)                                 *)
(*                                                                         *)
(* Models two virtual tiles where mapped tile 0 stores payload 0 and       *)
(* mapped tile 1 stores payload 1 (so a zero payload never masquerades as  *)
(* a residency status), command-captured mapping generations across        *)
(* record/submit/abandon/complete, rejection of remapping while a command  *)
(* retains the active generation, and immutability of observed status      *)
(* across subsequent remappings.                                           *)
(*                                                                         *)
(* Trusted boundary: D3D12 CheckAccessFullyMapped hardware feedback bit    *)
(* encoding for a given page-table state.                                  *)
(***************************************************************************)
EXTENDS Integers

Accesses == {"tile0", "tile1", "both"}
CommandPhases == {"idle", "recorded", "inFlight"}
Outcomes == {"none", "accepted", "rejected"}
UntouchedValue == -1
MaxVersion == 2

VARIABLES
  nativeTier2,            \* Adapter hardware support for D3D12 Tiled Resources Tier 2.
  reportedSparseTextures, \* Public DeviceCapabilities::sparseTextures flag.
  textureLive,            \* Whether a sparse texture handle is currently live.
  mappingVersion,         \* Current mapping generation on the texture (0..MaxVersion).
  mapped0,                \* Whether virtual tile 0 is currently mapped (payload 0).
  mapped1,                \* Whether virtual tile 1 is currently mapped (payload 1).
  commandPhase,           \* Lifecycle of the sampling command (idle/recorded/inFlight).
  commandAccess,          \* Footprint touched by the recorded shader operation.
  commandVersion,         \* Mapping generation captured at command recording.
  commandMapped0,         \* Tile 0 residency in the command-captured generation.
  commandMapped1,         \* Tile 1 residency in the command-captured generation.
  resultValid,            \* Whether a completed GPU shader observation exists.
  resultAccess,           \* Footprint of the last completed shader observation.
  resultVersion,          \* Generation observed by the last completed shader run.
  resultFullyMapped,      \* Observed CheckAccessFullyMapped status boolean.
  resultValue0,           \* Observed payload for tile 0 (0 if touched, -1 otherwise).
  resultValue1,           \* Observed payload for tile 1 (0 or 1 if touched, -1 otherwise).
  lastOutcome             \* Outcome of the latest fallible creation/mapping attempt.

resultVars == <<
  resultValid,
  resultAccess,
  resultVersion,
  resultFullyMapped,
  resultValue0,
  resultValue1
>>

coreVars == <<
  nativeTier2,
  reportedSparseTextures,
  textureLive,
  mappingVersion,
  mapped0,
  mapped1,
  commandPhase,
  commandAccess,
  commandVersion,
  commandMapped0,
  commandMapped1,
  resultValid,
  resultAccess,
  resultVersion,
  resultFullyMapped,
  resultValue0,
  resultValue1
>>

vars == <<
  nativeTier2,
  reportedSparseTextures,
  textureLive,
  mappingVersion,
  mapped0,
  mapped1,
  commandPhase,
  commandAccess,
  commandVersion,
  commandMapped0,
  commandMapped1,
  resultValid,
  resultAccess,
  resultVersion,
  resultFullyMapped,
  resultValue0,
  resultValue1,
  lastOutcome
>>

Touches0(access) == access = "tile0" \/ access = "both"
Touches1(access) == access = "tile1" \/ access = "both"

Init ==
  \E supported \in BOOLEAN :
    /\ nativeTier2 = supported
    /\ reportedSparseTextures = supported
    /\ textureLive = FALSE
    /\ mappingVersion = 0
    /\ mapped0 = FALSE
    /\ mapped1 = FALSE
    /\ commandPhase = "idle"
    /\ commandAccess = "tile0"
    /\ commandVersion = 0
    /\ commandMapped0 = FALSE
    /\ commandMapped1 = FALSE
    /\ resultValid = FALSE
    /\ resultAccess = "tile0"
    /\ resultVersion = 0
    /\ resultFullyMapped = FALSE
    /\ resultValue0 = UntouchedValue
    /\ resultValue1 = UntouchedValue
    /\ lastOutcome = "none"

attemptCreateTexture ==
  /\ ~textureLive
  /\ mappingVersion = 0
  /\ ~resultValid
  /\ IF reportedSparseTextures
     THEN /\ textureLive' = TRUE
          /\ mappingVersion' = 0
          /\ mapped0' = FALSE
          /\ mapped1' = FALSE
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               nativeTier2,
               reportedSparseTextures,
               commandPhase,
               commandAccess,
               commandVersion,
               commandMapped0,
               commandMapped1,
               resultValid,
               resultAccess,
               resultVersion,
               resultFullyMapped,
               resultValue0,
               resultValue1
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

attemptUpdateMapping(firstMapped, secondMapped) ==
  /\ mappingVersion < MaxVersion
  /\ IF textureLive /\ commandPhase = "idle"
     THEN /\ mappingVersion' = mappingVersion + 1
          /\ mapped0' = firstMapped
          /\ mapped1' = secondMapped
          /\ lastOutcome' = "accepted"
          /\ UNCHANGED <<
               nativeTier2,
               reportedSparseTextures,
               textureLive,
               commandPhase,
               commandAccess,
               commandVersion,
               commandMapped0,
               commandMapped1,
               resultValid,
               resultAccess,
               resultVersion,
               resultFullyMapped,
               resultValue0,
               resultValue1
             >>
     ELSE /\ lastOutcome' = "rejected"
          /\ UNCHANGED coreVars

recordAccess(access) ==
  /\ textureLive
  /\ commandPhase = "idle"
  /\ commandPhase' = "recorded"
  /\ commandAccess' = access
  /\ commandVersion' = mappingVersion
  /\ commandMapped0' = mapped0
  /\ commandMapped1' = mapped1
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       nativeTier2,
       reportedSparseTextures,
       textureLive,
       mappingVersion,
       mapped0,
       mapped1,
       resultValid,
       resultAccess,
       resultVersion,
       resultFullyMapped,
       resultValue0,
       resultValue1
     >>

abandonAccess ==
  /\ commandPhase = "recorded"
  /\ commandPhase' = "idle"
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       nativeTier2,
       reportedSparseTextures,
       textureLive,
       mappingVersion,
       mapped0,
       mapped1,
       commandAccess,
       commandVersion,
       commandMapped0,
       commandMapped1,
       resultValid,
       resultAccess,
       resultVersion,
       resultFullyMapped,
       resultValue0,
       resultValue1
     >>

submitAccess ==
  /\ commandPhase = "recorded"
  /\ commandPhase' = "inFlight"
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       nativeTier2,
       reportedSparseTextures,
       textureLive,
       mappingVersion,
       mapped0,
       mapped1,
       commandAccess,
       commandVersion,
       commandMapped0,
       commandMapped1,
       resultValid,
       resultAccess,
       resultVersion,
       resultFullyMapped,
       resultValue0,
       resultValue1
     >>

\* Mapped tile 0 holds payload 0 (same numeric value as a null tile); mapped
\* tile 1 holds payload 1; CheckAccessFullyMapped reflects residency, not data.
completeAccess ==
  /\ commandPhase = "inFlight"
  /\ commandPhase' = "idle"
  /\ resultValid' = TRUE
  /\ resultAccess' = commandAccess
  /\ resultVersion' = commandVersion
  /\ resultFullyMapped' =
       ((~Touches0(commandAccess) \/ commandMapped0)
        /\ (~Touches1(commandAccess) \/ commandMapped1))
  /\ resultValue0' = IF Touches0(commandAccess) THEN 0 ELSE UntouchedValue
  /\ resultValue1' =
       IF Touches1(commandAccess)
       THEN (IF commandMapped1 THEN 1 ELSE 0)
       ELSE UntouchedValue
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       nativeTier2,
       reportedSparseTextures,
       textureLive,
       mappingVersion,
       mapped0,
       mapped1,
       commandAccess,
       commandVersion,
       commandMapped0,
       commandMapped1
     >>

destroyTexture ==
  /\ textureLive
  /\ commandPhase = "idle"
  /\ textureLive' = FALSE
  /\ lastOutcome' = "none"
  /\ UNCHANGED <<
       nativeTier2,
       reportedSparseTextures,
       mappingVersion,
       mapped0,
       mapped1,
       commandPhase,
       commandAccess,
       commandVersion,
       commandMapped0,
       commandMapped1,
       resultValid,
       resultAccess,
       resultVersion,
       resultFullyMapped,
       resultValue0,
       resultValue1
     >>

Next ==
  \/ attemptCreateTexture
  \/ \E m0, m1 \in BOOLEAN : attemptUpdateMapping(m0, m1)
  \/ \E access \in Accesses : recordAccess(access)
  \/ abandonAccess
  \/ submitAccess
  \/ completeAccess
  \/ destroyTexture

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ nativeTier2 \in BOOLEAN
  /\ reportedSparseTextures \in BOOLEAN
  /\ textureLive \in BOOLEAN
  /\ mappingVersion \in 0..MaxVersion
  /\ mapped0 \in BOOLEAN
  /\ mapped1 \in BOOLEAN
  /\ commandPhase \in CommandPhases
  /\ commandAccess \in Accesses
  /\ commandVersion \in 0..MaxVersion
  /\ commandMapped0 \in BOOLEAN
  /\ commandMapped1 \in BOOLEAN
  /\ resultValid \in BOOLEAN
  /\ resultAccess \in Accesses
  /\ resultVersion \in 0..MaxVersion
  /\ resultFullyMapped \in BOOLEAN
  /\ resultValue0 \in {UntouchedValue, 0}
  /\ resultValue1 \in {UntouchedValue, 0, 1}
  /\ lastOutcome \in Outcomes

SparseProfileIsIndivisible ==
  /\ reportedSparseTextures = nativeTier2
  /\ (textureLive => reportedSparseTextures)

ActiveCommandsLockCapturedGeneration ==
  commandPhase # "idle" =>
    /\ commandVersion = mappingVersion
    /\ commandMapped0 = mapped0
    /\ commandMapped1 = mapped1

ObservedFootprintAndStatusConsistent ==
  resultValid =>
    /\ resultVersion <= mappingVersion
    /\ (Touches0(resultAccess) <=> resultValue0 = 0)
    /\ (Touches1(resultAccess) <=> resultValue1 \in {0, 1})
    /\ (resultValue1 = 1 => Touches1(resultAccess))
    /\ (resultFullyMapped /\ Touches1(resultAccess) => resultValue1 = 1)

Safety ==
  /\ SparseProfileIsIndivisible
  /\ ActiveCommandsLockCapturedGeneration
  /\ ObservedFootprintAndStatusConsistent

StepSafety ==
  [][/\ (lastOutcome' = "rejected" => coreVars' = coreVars)
     /\ (~(commandPhase = "inFlight" /\ commandPhase' = "idle")
         => resultVars' = resultVars)]_vars

=============================================================================
