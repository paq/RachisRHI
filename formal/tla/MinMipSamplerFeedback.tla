----------------------- MODULE MinMipSamplerFeedback ------------------------
(***************************************************************************)
(* MinMip sampler-feedback capability reporting, map-to-texture pairing,   *)
(* barrier/reset/write/decode state transitions, staged-recording rollback *)
(* on abandonment vs queue-submission commit, failure atomicity, and       *)
(* paired physical lifetime retention in the D3D12 backend: capability     *)
(* detection (`detectMeshShadersAndSamplerFeedback`) in                    *)
(* `src/D3D12/D3D12Device.cpp`; `ID3D12Device8` acquisition, map lookup,   *)
(* and binding sources (`samplerFeedbackBindingSource`) in                 *)
(* `src/D3D12/D3D12Resource.cpp`; map creation, reset, decode, and         *)
(* destruction (`createMinMipFeedbackMap`, `resetSamplerFeedbackMap`,      *)
(* `decodeMinMipFeedback`, `destroySamplerFeedbackMap`) in                 *)
(* `src/D3D12/D3D12SparseAndFeedback.cpp`; whole-map layout barriers       *)
(* (`samplerFeedbackBarrier`) in `src/D3D12/D3D12ResourceBarriers.cpp`;    *)
(* paired shader-write descriptors (`validateSamplerFeedbackPairing`) in   *)
(* `src/D3D12/D3D12Binding.cpp`; and command retention, abandonment, and   *)
(* submission in `src/D3D12/D3D12Queue.cpp` (`include/Rachis/Device.h`).   *)
(*                                                                         *)
(* Models one sampled texture (0) and one MinMip feedback map across       *)
(* capability gating, immutable pairing, whole-map layout barriers         *)
(* (`0` common, `1` unorderedAccess, `2` resolveSource, `3` invalid),      *)
(* reset (`255`), shader write (`0` when initialized), resolve decode,     *)
(* atomic rejection, command submission/completion/abandonment, and exact  *)
(* reference-counted physical lifetime when map or texture handles are     *)
(* destroyed in either order.                                              *)
(*                                                                         *)
(* Abstracted: per-mip region bitmasks (covered by                         *)
(* `MipRegionUsedSamplerFeedback.tla`) and sub-tile GPU sampling math.     *)
(***************************************************************************)
EXTENDS Integers

ValidLayouts == {0, 1, 2}
RequestedLayouts == {0, 1, 2, 3}
Contents == {-1, 0, 255}
Operations == {
    "none", "create", "barrier", "reset", "write", "decode",
    "destroyMap", "destroyTexture", "submit", "complete", "abandon"
}
Outcomes == {"none", "accepted", "rejected"}

VARIABLES
    nativeSupported,         \* hardware MinMip sampler-feedback support
    reportedSupported,       \* capability surfaced by DeviceCapabilities
    mapEverCreated,          \* TRUE once the modeled feedback map has been created
    mapLogical,              \* TRUE while public feedback-map handle is live
    mapPhysical,             \* TRUE while native feedback-map resource is live
    textureLogical,          \* TRUE while public sampled-texture handle is live
    texturePhysical,         \* TRUE while native sampled-texture resource is live
    pairedTexture,           \* 0 while mapPhysical retains pairing, -1 otherwise
    layout,                  \* planned map layout in ValidLayouts
    contents,                \* planned contents: -1 undefined, 0 mip0, 255 none
    decodedValid,            \* planned decode snapshot flag in open recording
    decodedContents,         \* planned decoded contents in Contents
    committedLayout,         \* committed GPU map layout in ValidLayouts
    committedContents,       \* committed GPU feedback contents in Contents
    committedDecodedValid,   \* committed GPU decode snapshot flag
    committedDecodedContents,\* committed GPU decoded contents in Contents
    recordingLease,          \* TRUE while open command recording retains the pair
    inFlightLease,           \* TRUE while in-flight submission retains the pair
    lastOperation,           \* latest operation tag in Operations
    lastOutcome              \* outcome of latest operation in Outcomes

committedVars == <<
    committedLayout, committedContents,
    committedDecodedValid, committedDecodedContents
>>

coreVars == <<
    nativeSupported, reportedSupported, mapEverCreated, mapLogical,
    mapPhysical, textureLogical, texturePhysical, pairedTexture,
    layout, contents, decodedValid, decodedContents,
    committedLayout, committedContents,
    committedDecodedValid, committedDecodedContents,
    recordingLease, inFlightLease
>>

vars == <<
    nativeSupported, reportedSupported, mapEverCreated, mapLogical,
    mapPhysical, textureLogical, texturePhysical, pairedTexture,
    layout, contents, decodedValid, decodedContents,
    committedLayout, committedContents,
    committedDecodedValid, committedDecodedContents,
    recordingLease, inFlightLease, lastOperation, lastOutcome
>>

Retained ==
    recordingLease \/ inFlightLease

Init ==
    \E supported \in BOOLEAN :
        /\ nativeSupported = supported
        /\ reportedSupported = supported
        /\ mapEverCreated = FALSE
        /\ mapLogical = FALSE
        /\ mapPhysical = FALSE
        /\ textureLogical = TRUE
        /\ texturePhysical = TRUE
        /\ pairedTexture = -1
        /\ layout = 0
        /\ contents = -1
        /\ decodedValid = FALSE
        /\ decodedContents = -1
        /\ committedLayout = 0
        /\ committedContents = -1
        /\ committedDecodedValid = FALSE
        /\ committedDecodedContents = -1
        /\ recordingLease = FALSE
        /\ inFlightLease = FALSE
        /\ lastOperation = "none"
        /\ lastOutcome = "none"

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
               /\ layout' = 0
               /\ contents' = -1
               /\ committedLayout' = 0
               /\ committedContents' = -1
               /\ UNCHANGED <<
                    nativeSupported, reportedSupported, textureLogical,
                    texturePhysical, decodedValid, decodedContents,
                    committedDecodedValid, committedDecodedContents,
                    recordingLease, inFlightLease
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOperation' = "create"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

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
                    nativeSupported, reportedSupported, mapEverCreated,
                    mapLogical, mapPhysical, textureLogical, texturePhysical,
                    pairedTexture, contents, decodedValid, decodedContents,
                    committedVars, inFlightLease
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOperation' = "barrier"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

\* Layout is caller-managed in RachisRHI; reset and shader write require UAV layout (1).
attemptReset(qualifier) ==
    LET accepted == qualifier /\ mapLogical /\ ~inFlightLease
    IN /\ layout = 1
       /\ IF accepted
          THEN /\ contents' = 255
               /\ recordingLease' = TRUE
               /\ UNCHANGED <<
                    nativeSupported, reportedSupported, mapEverCreated,
                    mapLogical, mapPhysical, textureLogical, texturePhysical,
                    pairedTexture, layout, decodedValid, decodedContents,
                    committedVars, inFlightLease
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOperation' = "reset"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

attemptWrite(qualifier) ==
    LET accepted ==
            /\ qualifier
            /\ mapLogical
            /\ textureLogical
            /\ pairedTexture = 0
            /\ ~inFlightLease
    IN /\ layout = 1
       /\ IF accepted
          THEN /\ contents' = IF contents = -1 THEN -1 ELSE 0
               /\ recordingLease' = TRUE
               /\ UNCHANGED <<
                    nativeSupported, reportedSupported, mapEverCreated,
                    mapLogical, mapPhysical, textureLogical, texturePhysical,
                    pairedTexture, layout, decodedValid, decodedContents,
                    committedVars, inFlightLease
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOperation' = "write"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

\* Decode consumes resolveSource layout (2).
attemptDecode(qualifier) ==
    LET accepted == qualifier /\ mapLogical /\ ~inFlightLease
    IN /\ layout = 2
       /\ IF accepted
          THEN /\ decodedValid' = TRUE
               /\ decodedContents' = contents
               /\ recordingLease' = TRUE
               /\ UNCHANGED <<
                    nativeSupported, reportedSupported, mapEverCreated,
                    mapLogical, mapPhysical, textureLogical, texturePhysical,
                    pairedTexture, layout, contents,
                    committedVars, inFlightLease
                  >>
          ELSE UNCHANGED coreVars
       /\ lastOperation' = "decode"
       /\ lastOutcome' = IF accepted THEN "accepted" ELSE "rejected"

\* Destroying the map drops its shared_ptr<TextureRecord> once no lease retains the map.
destroyMap ==
    LET nextMapPhysical == Retained
    IN /\ mapLogical
       /\ mapLogical' = FALSE
       /\ mapPhysical' = nextMapPhysical
       /\ texturePhysical' = (textureLogical \/ nextMapPhysical)
       /\ pairedTexture' = IF nextMapPhysical THEN pairedTexture ELSE -1
       /\ lastOperation' = "destroyMap"
       /\ lastOutcome' = "accepted"
       /\ UNCHANGED <<
            nativeSupported, reportedSupported, mapEverCreated,
            textureLogical, layout, contents, decodedValid,
            decodedContents, committedVars, recordingLease, inFlightLease
          >>

destroyTexture ==
    /\ textureLogical
    /\ textureLogical' = FALSE
    /\ texturePhysical' = (mapPhysical \/ Retained)
    /\ lastOperation' = "destroyTexture"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        nativeSupported, reportedSupported, mapEverCreated,
        mapLogical, mapPhysical, pairedTexture, layout, contents,
        decodedValid, decodedContents, committedVars,
        recordingLease, inFlightLease
       >>

submit ==
    /\ recordingLease
    /\ ~inFlightLease
    /\ recordingLease' = FALSE
    /\ inFlightLease' = TRUE
    /\ committedLayout' = layout
    /\ committedContents' = contents
    /\ committedDecodedValid' = decodedValid
    /\ committedDecodedContents' = decodedContents
    /\ lastOperation' = "submit"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        nativeSupported, reportedSupported, mapEverCreated,
        mapLogical, mapPhysical, textureLogical, texturePhysical,
        pairedTexture, layout, contents, decodedValid, decodedContents
       >>

complete ==
    /\ inFlightLease
    /\ mapPhysical' = (mapLogical \/ recordingLease)
    /\ texturePhysical' = (textureLogical \/ mapLogical \/ recordingLease)
    /\ pairedTexture' =
        IF mapLogical \/ recordingLease THEN pairedTexture ELSE -1
    /\ inFlightLease' = FALSE
    /\ lastOperation' = "complete"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        nativeSupported, reportedSupported, mapEverCreated,
        mapLogical, textureLogical, layout, contents,
        decodedValid, decodedContents, committedVars, recordingLease
       >>

\* Abandoning an open recording restores all staged layout/feedback state to
\* the last committed GPU state.
abandon ==
    /\ recordingLease
    /\ ~inFlightLease
    /\ mapPhysical' = mapLogical
    /\ texturePhysical' = (textureLogical \/ mapLogical)
    /\ pairedTexture' = IF mapLogical THEN pairedTexture ELSE -1
    /\ layout' = committedLayout
    /\ contents' = committedContents
    /\ decodedValid' = committedDecodedValid
    /\ decodedContents' = committedDecodedContents
    /\ recordingLease' = FALSE
    /\ lastOperation' = "abandon"
    /\ lastOutcome' = "accepted"
    /\ UNCHANGED <<
        nativeSupported, reportedSupported, mapEverCreated,
        mapLogical, textureLogical, committedVars, inFlightLease
       >>

Next ==
    \/ \E q \in BOOLEAN :
        \/ attemptCreate(q)
        \/ attemptReset(q)
        \/ attemptWrite(q)
        \/ attemptDecode(q)
    \/ \E q \in BOOLEAN, lay \in RequestedLayouts : attemptBarrier(q, lay)
    \/ destroyMap
    \/ destroyTexture
    \/ submit
    \/ complete
    \/ abandon

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ nativeSupported \in BOOLEAN
    /\ reportedSupported \in BOOLEAN
    /\ mapEverCreated \in BOOLEAN
    /\ mapLogical \in BOOLEAN
    /\ mapPhysical \in BOOLEAN
    /\ textureLogical \in BOOLEAN
    /\ texturePhysical \in BOOLEAN
    /\ pairedTexture \in {-1, 0}
    /\ layout \in ValidLayouts
    /\ contents \in Contents
    /\ decodedValid \in BOOLEAN
    /\ decodedContents \in Contents
    /\ committedLayout \in ValidLayouts
    /\ committedContents \in Contents
    /\ committedDecodedValid \in BOOLEAN
    /\ committedDecodedContents \in Contents
    /\ recordingLease \in BOOLEAN
    /\ inFlightLease \in BOOLEAN
    /\ lastOperation \in Operations
    /\ lastOutcome \in Outcomes

CapabilityMatchesNative ==
    reportedSupported = nativeSupported

CreationEstablishesOnePair ==
    /\ mapPhysical =>
        /\ mapEverCreated
        /\ reportedSupported
        /\ pairedTexture = 0
        /\ texturePhysical
    /\ ~mapPhysical => pairedTexture = -1

MapPhysicalLifetimeExact ==
    mapPhysical = (mapLogical \/ Retained)

TexturePhysicalLifetimeExact ==
    texturePhysical = (textureLogical \/ mapPhysical \/ Retained)

UnrecordedStateMatchesCommitted ==
    ~recordingLease =>
        /\ layout = committedLayout
        /\ contents = committedContents
        /\ decodedValid = committedDecodedValid
        /\ decodedContents = committedDecodedContents

AcceptedResetMeansNoRequest ==
    (lastOperation = "reset" /\ lastOutcome = "accepted") => contents = 255

AcceptedDecodeSnapshotsOpaqueContents ==
    (lastOperation = "decode" /\ lastOutcome = "accepted") =>
        /\ decodedValid
        /\ decodedContents = contents

Safety ==
    /\ CapabilityMatchesNative
    /\ CreationEstablishesOnePair
    /\ MapPhysicalLifetimeExact
    /\ TexturePhysicalLifetimeExact
    /\ UnrecordedStateMatchesCommitted
    /\ AcceptedResetMeansNoRequest
    /\ AcceptedDecodeSnapshotsOpaqueContents

StepSafety ==
    [][
        /\ (lastOutcome' = "rejected" => UNCHANGED coreVars)
        /\ ((lastOperation' # "submit"
             /\ ~(lastOperation' = "create" /\ lastOutcome' = "accepted")) =>
                UNCHANGED committedVars)
        /\ (lastOperation' = "abandon" =>
                /\ layout' = committedLayout
                /\ contents' = committedContents
                /\ decodedValid' = committedDecodedValid
                /\ decodedContents' = committedDecodedContents)
        /\ ((lastOperation' = "reset" /\ lastOutcome' = "accepted") =>
                /\ layout = 1
                /\ contents' = 255)
        /\ ((lastOperation' = "write" /\ lastOutcome' = "accepted") =>
                /\ layout = 1
                /\ contents' = (IF contents = -1 THEN -1 ELSE 0))
        /\ ((lastOperation' = "decode" /\ lastOutcome' = "accepted") =>
                /\ layout = 2
                /\ decodedValid'
                /\ decodedContents' = contents)
    ]_vars

=============================================================================
