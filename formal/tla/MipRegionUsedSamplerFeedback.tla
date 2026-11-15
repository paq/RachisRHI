-------------------- MODULE MipRegionUsedSamplerFeedback --------------------
(***************************************************************************)
(* Tier 1.0 MipRegionUsed sampler-feedback capability gating, immutable    *)
(* pairing, per-mip occupancy isolation, mode-confusion rejection, staged  *)
(* recording rollback on abandonment vs queue-submission commit, failure   *)
(* atomicity, and paired resource retention in the D3D12 backend: Tier     *)
(* 1.0, `ID3D12Device8`, and Shader Model 6.5 detection                    *)
(* (`detectMeshShadersAndSamplerFeedback`) in `src/D3D12/D3D12Device.cpp`; *)
(* map lookup and mode-checked binding sources                             *)
(* (`samplerFeedbackBindingSource`) in `src/D3D12/D3D12Resource.cpp`; map  *)
(* creation, reset, mode-checked decode, and destruction                   *)
(* (`createMipRegionUsedFeedbackMap`, `resetSamplerFeedbackMap`,           *)
(* `decodeMipRegionUsedFeedback`, `destroySamplerFeedbackMap`) in          *)
(* `src/D3D12/D3D12SparseAndFeedback.cpp`; layout barriers                 *)
(* (`samplerFeedbackBarrier`) in `src/D3D12/D3D12ResourceBarriers.cpp`;    *)
(* mode-specific binding validation                                        *)
(* (`resolveSamplerFeedbackDescriptorSource`,                              *)
(* `validateSamplerFeedbackPairing`) in `src/D3D12/D3D12Binding.cpp`;      *)
(* shader-interface binding-type validation (`validateShaderInterface`) in *)
(* `src/D3D12/D3D12Pipeline.cpp`; and command retention, abandonment, and  *)
(* submission in `src/D3D12/D3D12Queue.cpp` (`include/Rachis/Device.h`).   *)
(*                                                                         *)
(* Models the conjunction of native Tier 1.0 + ID3D12Device8 + SM 6.5      *)
(* support, a three-mip occupancy vector (`Mips = {0, 1, 2}`), explicit    *)
(* layout barriers, reset initialization, per-mip shader writes that leave *)
(* sibling mips untouched, mode-checked shader binding and full decode,    *)
(* command submission/completion/abandonment, and exact physical lifetime  *)
(* retention across arbitrary map/texture destruction orderings.           *)
(*                                                                         *)
(* Abstracted: sub-mip spatial tile coordinates and MinMip scalar encoding *)
(* (covered by `MinMipSamplerFeedback.tla`).                               *)
(***************************************************************************)
EXTENDS Integers

RequestedModes == {0, 1, 2}
Mips == {0, 1, 2}
RequestedMips == {0, 1, 2, 3}
ValidLayouts == {0, 1, 2}
RequestedLayouts == {0, 1, 2, 3}
Cleared == [m \in Mips |-> FALSE]
Operations == {
    "none", "create", "barrier", "reset", "write", "bind", "decode",
    "destroyMap", "destroyTexture", "submit", "complete", "abandon"
}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    nativeTier1,             \* hardware SamplerFeedback Tier 1.0 support
    nativeDevice8,           \* native ID3D12Device8 interface availability
    nativeShader65,          \* Shader Model 6.5 runtime support
    reportedSupported,       \* capability surfaced by DeviceCapabilities
    mapEverCreated,          \* TRUE once the MipRegionUsed map has been created
    mapLogical,              \* TRUE while public feedback-map handle is live
    mapPhysical,             \* TRUE while native feedback-map resource is live
    textureLogical,          \* TRUE while public sampled-texture handle is live
    texturePhysical,         \* TRUE while native sampled-texture resource is live
    pairedTexture,           \* 0 while mapPhysical retains pairing, -1 otherwise
    mapMode,                 \* 1 (MipRegionUsed) while mapPhysical, -1 otherwise
    layout,                  \* planned map layout in ValidLayouts
    initialized,             \* planned reset-initialized flag in open recording
    contents,                \* planned per-mip occupancy vector in [Mips -> BOOLEAN]
    decodedValid,            \* planned full-decode snapshot flag in open recording
    decodedContents,         \* planned decoded per-mip vector in [Mips -> BOOLEAN]
    committedLayout,         \* committed GPU map layout in ValidLayouts
    committedInitialized,    \* committed GPU reset-initialized flag
    committedContents,       \* committed GPU per-mip occupancy vector
    committedDecodedValid,   \* committed GPU full-decode snapshot flag
    committedDecodedContents,\* committed GPU decoded per-mip vector
    recordingLease,          \* TRUE while open command recording retains the pair
    inFlightLease,           \* TRUE while in-flight submission retains the pair
    lastOperation,           \* latest operation tag in Operations
    lastOutcome,             \* outcome of latest operation in Outcomes
    lastRequestedMode,       \* mode argument of latest mode-checked request
    lastRequestedMip         \* mip argument of latest shader write request

committedVars == <<
    committedLayout, committedInitialized, committedContents,
    committedDecodedValid, committedDecodedContents
>>

coreVars == <<
    nativeTier1, nativeDevice8, nativeShader65, reportedSupported,
    mapEverCreated, mapLogical, mapPhysical, textureLogical,
    texturePhysical, pairedTexture, mapMode, layout, initialized,
    contents, decodedValid, decodedContents,
    committedLayout, committedInitialized, committedContents,
    committedDecodedValid, committedDecodedContents,
    recordingLease, inFlightLease
>>

vars == <<
    nativeTier1, nativeDevice8, nativeShader65, reportedSupported,
    mapEverCreated, mapLogical, mapPhysical, textureLogical,
    texturePhysical, pairedTexture, mapMode, layout, initialized,
    contents, decodedValid, decodedContents,
    committedLayout, committedInitialized, committedContents,
    committedDecodedValid, committedDecodedContents,
    recordingLease, inFlightLease, lastOperation, lastOutcome,
    lastRequestedMode, lastRequestedMip
>>

CompleteNativePrerequisite(tier1, device8, shader65) ==
    tier1 /\ device8 /\ shader65

Retained ==
    recordingLease \/ inFlightLease

Init ==
    \E tier1 \in BOOLEAN, device8 \in BOOLEAN, shader65 \in BOOLEAN :
        /\ nativeTier1 = tier1
        /\ nativeDevice8 = device8
        /\ nativeShader65 = shader65
        /\ reportedSupported = CompleteNativePrerequisite(tier1, device8, shader65)
        /\ mapEverCreated = FALSE
        /\ mapLogical = FALSE
        /\ mapPhysical = FALSE
        /\ textureLogical = TRUE
        /\ texturePhysical = TRUE
        /\ pairedTexture = -1
        /\ mapMode = -1
        /\ layout = 0
        /\ initialized = FALSE
        /\ contents = Cleared
        /\ decodedValid = FALSE
        /\ decodedContents = Cleared
        /\ committedLayout = 0
        /\ committedInitialized = FALSE
        /\ committedContents = Cleared
        /\ committedDecodedValid = FALSE
        /\ committedDecodedContents = Cleared
        /\ recordingLease = FALSE
        /\ inFlightLease = FALSE
        /\ lastOperation = "none"
        /\ lastOutcome = "none"
        /\ lastRequestedMode = 1
        /\ lastRequestedMip = 0

ResetRequestMeta ==
    /\ lastRequestedMode' = 1
    /\ lastRequestedMip' = 0

attemptCreate(qualifier) ==
    LET accepted ==
            /\ qualifier
            /\ reportedSupported
            /\ textureLogical
            /\ ~mapEverCreated
    IN /\ IF accepted
          THEN /\ mapEverCreated' = TRUE
               /\ mapLogical' = TRUE
               /\ mapPhysical' = TRUE
               /\ pairedTexture' = 0
               /\ mapMode' = 1
               /\ layout' = 0
               /\ initialized' = FALSE
               /\ contents' = Cleared
               /\ committedLayout' = 0
               /\ committedInitialized' = FALSE
               /\ committedContents' = Cleared
               /\ UNCHANGED <<
                    nativeTier1, nativeDevice8, nativeShader65,
                    reportedSupported, textureLogical, texturePhysical,
                    decodedValid, decodedContents,
                    committedDecodedValid, committedDecodedContents,
                    recordingLease, inFlightLease
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOperation' = "create"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"
       /\ ResetRequestMeta

attemptBarrier(qualifier, requestedLayout) ==
    LET accepted ==
            /\ qualifier
            /\ mapLogical
            /\ requestedLayout \in ValidLayouts
            /\ ~inFlightLease
    IN /\ IF accepted
          THEN /\ layout' = requestedLayout
               /\ recordingLease' = TRUE
               /\ UNCHANGED <<
                    nativeTier1, nativeDevice8, nativeShader65,
                    reportedSupported, mapEverCreated, mapLogical,
                    mapPhysical, textureLogical, texturePhysical,
                    pairedTexture, mapMode, initialized, contents,
                    decodedValid, decodedContents, committedVars,
                    inFlightLease
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOperation' = "barrier"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"
       /\ ResetRequestMeta

attemptReset(qualifier) ==
    LET accepted == qualifier /\ mapLogical /\ ~inFlightLease
    IN /\ layout = 1
       /\ IF accepted
          THEN /\ initialized' = TRUE
               /\ contents' = Cleared
               /\ recordingLease' = TRUE
               /\ UNCHANGED <<
                    nativeTier1, nativeDevice8, nativeShader65,
                    reportedSupported, mapEverCreated, mapLogical,
                    mapPhysical, textureLogical, texturePhysical,
                    pairedTexture, mapMode, layout, decodedValid,
                    decodedContents, committedVars, inFlightLease
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOperation' = "reset"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"
       /\ ResetRequestMeta

attemptWrite(qualifier, requestedMode, requestedMip) ==
    LET accepted ==
            /\ qualifier
            /\ mapLogical
            /\ textureLogical
            /\ pairedTexture = 0
            /\ requestedMode = 1
            /\ requestedMip \in Mips
            /\ initialized
            /\ ~inFlightLease
    IN /\ layout = 1
       /\ IF accepted
          THEN /\ contents' = [contents EXCEPT ![requestedMip] = TRUE]
               /\ recordingLease' = TRUE
               /\ lastRequestedMode' = requestedMode
               /\ lastRequestedMip' = requestedMip
               /\ UNCHANGED <<
                    nativeTier1, nativeDevice8, nativeShader65,
                    reportedSupported, mapEverCreated, mapLogical,
                    mapPhysical, textureLogical, texturePhysical,
                    pairedTexture, mapMode, layout, initialized,
                    decodedValid, decodedContents, committedVars,
                    inFlightLease
                  >>
          ELSE /\ UNCHANGED coreVars
               /\ ResetRequestMeta
       /\ lastOperation' = "write"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

attemptBind(qualifier, requestedMode) ==
    LET accepted ==
            /\ qualifier
            /\ mapLogical
            /\ textureLogical
            /\ pairedTexture = 0
            /\ requestedMode = 1
            /\ ~inFlightLease
    IN /\ IF accepted
          THEN /\ recordingLease' = TRUE
               /\ lastRequestedMode' = requestedMode
               /\ lastRequestedMip' = 0
               /\ UNCHANGED <<
                    nativeTier1, nativeDevice8, nativeShader65,
                    reportedSupported, mapEverCreated, mapLogical,
                    mapPhysical, textureLogical, texturePhysical,
                    pairedTexture, mapMode, layout, initialized,
                    contents, decodedValid, decodedContents,
                    committedVars, inFlightLease
                  >>
          ELSE /\ UNCHANGED coreVars
               /\ ResetRequestMeta
       /\ lastOperation' = "bind"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

attemptDecode(qualifier, requestedMode) ==
    LET accepted ==
            /\ qualifier
            /\ mapLogical
            /\ requestedMode = 1
            /\ initialized
            /\ ~inFlightLease
    IN /\ layout = 2
       /\ IF accepted
          THEN /\ decodedValid' = TRUE
               /\ decodedContents' = contents
               /\ recordingLease' = TRUE
               /\ lastRequestedMode' = requestedMode
               /\ lastRequestedMip' = 0
               /\ UNCHANGED <<
                    nativeTier1, nativeDevice8, nativeShader65,
                    reportedSupported, mapEverCreated, mapLogical,
                    mapPhysical, textureLogical, texturePhysical,
                    pairedTexture, mapMode, layout, initialized,
                    contents, committedVars, inFlightLease
                  >>
          ELSE /\ UNCHANGED coreVars
               /\ ResetRequestMeta
       /\ lastOperation' = "decode"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

destroyMap ==
    LET nextMapPhysical == Retained
    IN /\ mapLogical
       /\ mapLogical' = FALSE
       /\ mapPhysical' = nextMapPhysical
       /\ texturePhysical' = (textureLogical \/ nextMapPhysical)
       /\ pairedTexture' = IF nextMapPhysical THEN pairedTexture ELSE -1
       /\ mapMode' = IF nextMapPhysical THEN mapMode ELSE -1
       /\ lastOperation' = "destroyMap"
       /\ lastOutcome' = "accepted"
       /\ ResetRequestMeta
       /\ UNCHANGED <<
            nativeTier1, nativeDevice8, nativeShader65,
            reportedSupported, mapEverCreated, textureLogical, layout,
            initialized, contents, decodedValid, decodedContents,
            committedVars, recordingLease, inFlightLease
          >>

destroyTexture ==
    /\ textureLogical
    /\ textureLogical' = FALSE
    /\ texturePhysical' = (mapPhysical \/ Retained)
    /\ lastOperation' = "destroyTexture"
    /\ lastOutcome' = "accepted"
    /\ ResetRequestMeta
    /\ UNCHANGED <<
        nativeTier1, nativeDevice8, nativeShader65, reportedSupported,
        mapEverCreated, mapLogical, mapPhysical, pairedTexture, mapMode,
        layout, initialized, contents, decodedValid, decodedContents,
        committedVars, recordingLease, inFlightLease
       >>

submit ==
    /\ recordingLease
    /\ ~inFlightLease
    /\ recordingLease' = FALSE
    /\ inFlightLease' = TRUE
    /\ committedLayout' = layout
    /\ committedInitialized' = initialized
    /\ committedContents' = contents
    /\ committedDecodedValid' = decodedValid
    /\ committedDecodedContents' = decodedContents
    /\ lastOperation' = "submit"
    /\ lastOutcome' = "accepted"
    /\ ResetRequestMeta
    /\ UNCHANGED <<
        nativeTier1, nativeDevice8, nativeShader65, reportedSupported,
        mapEverCreated, mapLogical, mapPhysical, textureLogical,
        texturePhysical, pairedTexture, mapMode, layout, initialized,
        contents, decodedValid, decodedContents
       >>

complete ==
    LET nextMapPhysical == mapLogical \/ recordingLease
    IN /\ inFlightLease
       /\ mapPhysical' = nextMapPhysical
       /\ texturePhysical' = (textureLogical \/ nextMapPhysical)
       /\ pairedTexture' = IF nextMapPhysical THEN pairedTexture ELSE -1
       /\ mapMode' = IF nextMapPhysical THEN mapMode ELSE -1
       /\ inFlightLease' = FALSE
       /\ lastOperation' = "complete"
       /\ lastOutcome' = "accepted"
       /\ ResetRequestMeta
       /\ UNCHANGED <<
            nativeTier1, nativeDevice8, nativeShader65, reportedSupported,
            mapEverCreated, mapLogical, textureLogical, layout,
            initialized, contents, decodedValid, decodedContents,
            committedVars, recordingLease
          >>

\* Abandoning an open recording discards unsubmitted layout/reset/write/decode
\* mutations and restores the last committed GPU state.
abandon ==
    /\ recordingLease
    /\ ~inFlightLease
    /\ mapPhysical' = mapLogical
    /\ texturePhysical' = (textureLogical \/ mapLogical)
    /\ pairedTexture' = IF mapLogical THEN pairedTexture ELSE -1
    /\ mapMode' = IF mapLogical THEN mapMode ELSE -1
    /\ layout' = committedLayout
    /\ initialized' = committedInitialized
    /\ contents' = committedContents
    /\ decodedValid' = committedDecodedValid
    /\ decodedContents' = committedDecodedContents
    /\ recordingLease' = FALSE
    /\ lastOperation' = "abandon"
    /\ lastOutcome' = "accepted"
    /\ ResetRequestMeta
    /\ UNCHANGED <<
        nativeTier1, nativeDevice8, nativeShader65, reportedSupported,
        mapEverCreated, mapLogical, textureLogical, committedVars,
        inFlightLease
       >>

Next ==
    \/ \E q \in BOOLEAN :
        \/ attemptCreate(q)
        \/ attemptReset(q)
    \/ \E q \in BOOLEAN, lay \in RequestedLayouts : attemptBarrier(q, lay)
    \/ \E q \in BOOLEAN, md \in RequestedModes, mip \in RequestedMips :
        attemptWrite(q, md, mip)
    \/ \E q \in BOOLEAN, md \in RequestedModes :
        \/ attemptBind(q, md)
        \/ attemptDecode(q, md)
    \/ destroyMap
    \/ destroyTexture
    \/ submit
    \/ complete
    \/ abandon

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ nativeTier1 \in BOOLEAN
    /\ nativeDevice8 \in BOOLEAN
    /\ nativeShader65 \in BOOLEAN
    /\ reportedSupported \in BOOLEAN
    /\ mapEverCreated \in BOOLEAN
    /\ mapLogical \in BOOLEAN
    /\ mapPhysical \in BOOLEAN
    /\ textureLogical \in BOOLEAN
    /\ texturePhysical \in BOOLEAN
    /\ pairedTexture \in {-1, 0}
    /\ mapMode \in {-1, 1}
    /\ layout \in ValidLayouts
    /\ initialized \in BOOLEAN
    /\ contents \in [Mips -> BOOLEAN]
    /\ decodedValid \in BOOLEAN
    /\ decodedContents \in [Mips -> BOOLEAN]
    /\ committedLayout \in ValidLayouts
    /\ committedInitialized \in BOOLEAN
    /\ committedContents \in [Mips -> BOOLEAN]
    /\ committedDecodedValid \in BOOLEAN
    /\ committedDecodedContents \in [Mips -> BOOLEAN]
    /\ recordingLease \in BOOLEAN
    /\ inFlightLease \in BOOLEAN
    /\ lastOperation \in Operations
    /\ lastOutcome \in Outcomes
    /\ lastRequestedMode \in RequestedModes
    /\ lastRequestedMip \in RequestedMips

CapabilityMatchesCompleteNativePrerequisite ==
    reportedSupported =
        CompleteNativePrerequisite(nativeTier1, nativeDevice8, nativeShader65)

CreationEstablishesImmutableMipRegionUsedPair ==
    /\ mapPhysical =>
        /\ mapEverCreated
        /\ reportedSupported
        /\ mapMode = 1
        /\ pairedTexture = 0
        /\ texturePhysical
    /\ ~mapPhysical =>
        /\ mapMode = -1
        /\ pairedTexture = -1

MapPhysicalLifetimeExact ==
    mapPhysical = (mapLogical \/ Retained)

TexturePhysicalLifetimeExact ==
    texturePhysical = (textureLogical \/ mapPhysical \/ Retained)

UnrecordedStateMatchesCommitted ==
    /\ (~recordingLease =>
            /\ layout = committedLayout
            /\ initialized = committedInitialized
            /\ contents = committedContents
            /\ decodedValid = committedDecodedValid
            /\ decodedContents = committedDecodedContents)
    /\ (~committedInitialized =>
            /\ committedContents = Cleared
            /\ ~committedDecodedValid)

AcceptedResetClearsEveryMip ==
    (lastOperation = "reset" /\ lastOutcome = "accepted") =>
        /\ initialized
        /\ contents = Cleared

AcceptedBindingAndDecodeRejectModeConfusion ==
    (lastOutcome = "accepted" /\ lastOperation \in {"write", "bind", "decode"}) =>
        lastRequestedMode = 1

AcceptedDecodeSnapshotsEveryMip ==
    (lastOperation = "decode" /\ lastOutcome = "accepted") =>
        /\ decodedValid
        /\ decodedContents = contents

Safety ==
    /\ CapabilityMatchesCompleteNativePrerequisite
    /\ CreationEstablishesImmutableMipRegionUsedPair
    /\ MapPhysicalLifetimeExact
    /\ TexturePhysicalLifetimeExact
    /\ UnrecordedStateMatchesCommitted
    /\ AcceptedResetClearsEveryMip
    /\ AcceptedBindingAndDecodeRejectModeConfusion
    /\ AcceptedDecodeSnapshotsEveryMip

StepSafety ==
    [][
        /\ (lastOutcome' = "rejected" => UNCHANGED coreVars)
        /\ ((lastOperation' # "submit"
             /\ ~(lastOperation' = "create" /\ lastOutcome' = "accepted")) =>
                UNCHANGED committedVars)
        /\ (lastOperation' = "abandon" =>
                /\ layout' = committedLayout
                /\ initialized' = committedInitialized
                /\ contents' = committedContents
                /\ decodedValid' = committedDecodedValid
                /\ decodedContents' = committedDecodedContents)
        /\ ((lastOperation' = "reset" /\ lastOutcome' = "accepted") =>
                /\ layout = 1
                /\ initialized'
                /\ contents' = Cleared)
        /\ ((lastOperation' = "write" /\ lastOutcome' = "accepted") =>
                /\ layout = 1
                /\ initialized
                /\ lastRequestedMip' \in Mips
                /\ contents'[lastRequestedMip']
                /\ \A m \in Mips :
                    m # lastRequestedMip' => contents'[m] = contents[m])
        /\ ((lastOperation' = "decode" /\ lastOutcome' = "accepted") =>
                /\ layout = 2
                /\ initialized
                /\ decodedValid'
                /\ decodedContents' = contents)
    ]_vars

=============================================================================
