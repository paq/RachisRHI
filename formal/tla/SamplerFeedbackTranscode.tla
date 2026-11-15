---------------------- MODULE SamplerFeedbackTranscode ----------------------
(***************************************************************************)
(* Sampler-feedback encode and decode validation (graphics command buffer, *)
(* outside active rendering scope, exact map mode, exact R8_UINT 2D shape  *)
(* and copy usage, and resolve-source/resolve-destination layouts),        *)
(* staged-recording rollback on abandonment vs queue-submission commit,    *)
(* internal command retention even on unretained pools, targeted-texture   *)
(* ownership, mutable source contents, and MinMip vs MipRegionUsed         *)
(* round-trip semantics.                                                   *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12SparseAndFeedback.cpp (decodeMinMipFeedback,         *)
(*     decodeMipRegionUsedFeedback, decodeSamplerFeedback,                 *)
(*     encodeMinMipFeedback, encodeMipRegionUsedFeedback,                  *)
(*     encodeSamplerFeedback, mode-specific subresource selection in       *)
(*     recordSamplerFeedbackEncode/recordSamplerFeedbackDecode)            *)
(*   - src/D3D12/D3D12ResourceBarriers.cpp (resolve-source and             *)
(*     resolve-destination layout transitions; encode/decode do not        *)
(*     re-validate layouts themselves)                                     *)
(*   - src/D3D12/D3D12Queue.cpp (retainInternalResource)                   *)
(*   - src/D3D12/D3D12Resource.h (SamplerFeedbackMapType)                  *)
(*   - include/Rachis/Device.h (C++-only encode/decode entry points)       *)
(*     and include/Rachis/Resource.h (MinMipFeedbackEncode,                *)
(*     MipRegionUsedFeedbackEncode)                                        *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Modes {0 = MinMip, 1 = MipRegionUsed} vs invalid 2, layouts {1, 2},   *)
(*   and contents {0, 1, 255} distinguish exact MipRegionUsed round-trips  *)
(*   from conservative MinMip normalization while preserving canonical 255 *)
(*   no-request clears across post-encode source mutations.                *)
(***************************************************************************)

EXTENDS Naturals

MinMipMode == 0
MipRegionUsedMode == 1
Modes == {MinMipMode, MipRegionUsedMode}
RequestedModes == {0, 1, 2}
Contents == {0, 1, 255}
ResolveSource == 1
ResolveDest == 2
Layouts == {ResolveSource, ResolveDest}
Operations == {
  "none", "writeSource", "beginRendering", "endRendering",
  "prepareEncode", "prepareDecode", "encode", "decode",
  "destroySource", "destroyMap", "destroyPaired", "destroyDestination",
  "abandonRecording", "beginRecording", "submit", "complete"
}
Outcomes == {"none", "accepted", "rejected"}

EncodedContents(mode, contents) ==
  IF mode = MipRegionUsedMode \/ contents = 255 THEN contents ELSE 0

VARIABLES
  mapMode,                     \* Configured SamplerFeedbackMapType: 0 or 1
  retainPool,                  \* Pool retains ordinary application resources
  recording,                   \* Whether command buffer is open for recording
  rendering,                   \* Whether an active render pass is open on command buffer
  inFlight,                    \* Whether submitted command buffer is executing on GPU
  recordingHasWork,            \* Whether open recording has recorded any command
  sourceContents,              \* Current CPU/GPU contents of R8_UINT source texture
  opaqueInitialized,           \* Planned feedback-map initialized flag in open recording
  opaqueContents,              \* Planned encoded representation in open recording
  encodedInput,                \* Planned source contents snapshot at latest encode
  decodedValid,                \* Planned destination decoded flag in open recording
  decodedContents,             \* Planned contents written by latest decode
  decodedSourceInput,          \* Planned source provenance of latest decode
  sourceLayout,                \* Planned layout of source texture
  mapLayout,                   \* Planned layout of feedback map
  destinationLayout,           \* Planned layout of destination texture
  committedOpaqueInitialized,  \* Committed GPU feedback-map initialized flag
  committedOpaqueContents,     \* Committed GPU encoded representation
  committedEncodedInput,       \* Committed GPU source snapshot at latest encode
  committedDecodedValid,       \* Committed GPU destination decoded flag
  committedDecodedContents,    \* Committed GPU decoded contents
  committedDecodedSourceInput, \* Committed GPU source provenance of latest decode
  committedSourceLayout,       \* Committed GPU layout of source texture
  committedMapLayout,          \* Committed GPU layout of feedback map
  committedDestinationLayout,  \* Committed GPU layout of destination texture
  sourceLogical,               \* Public handle liveness of source texture
  sourcePhysical,              \* Native storage liveness of source texture
  mapLogical,                  \* Public handle liveness of sampler-feedback map
  mapPhysical,                 \* Native storage liveness of sampler-feedback map
  pairedTextureLogical,        \* Public handle liveness of targeted paired texture
  pairedTexturePhysical,       \* Native storage liveness of targeted paired texture
  destinationLogical,          \* Public handle liveness of destination texture
  destinationPhysical,         \* Native storage liveness of destination texture
  recordingSourceLease,        \* Open command-buffer retention of source texture
  recordingMapLease,           \* Open command-buffer retention of feedback map
  recordingDestinationLease,   \* Open command-buffer retention of destination texture
  inFlightSourceLease,         \* In-flight GPU retention of source texture
  inFlightMapLease,            \* In-flight GPU retention of feedback map
  inFlightDestinationLease,    \* In-flight GPU retention of destination texture
  lastOp,                      \* Most recent operation kind
  lastOutcome,                 \* Outcome of most recent operation
  lastRequestedMode,           \* Mode passed to most recent encode/decode attempt
  lastGraphics,                \* Graphics-queue flag of most recent transcode attempt
  lastExactShape               \* Exact-R8_UINT-shape flag of most recent attempt

committedVars == <<
  committedOpaqueInitialized, committedOpaqueContents, committedEncodedInput,
  committedDecodedValid, committedDecodedContents, committedDecodedSourceInput,
  committedSourceLayout, committedMapLayout, committedDestinationLayout
>>

coreVars == <<
  mapMode, retainPool, recording, rendering, inFlight, recordingHasWork,
  sourceContents, opaqueInitialized, opaqueContents, encodedInput,
  decodedValid, decodedContents, decodedSourceInput,
  sourceLayout, mapLayout, destinationLayout,
  committedOpaqueInitialized, committedOpaqueContents, committedEncodedInput,
  committedDecodedValid, committedDecodedContents, committedDecodedSourceInput,
  committedSourceLayout, committedMapLayout, committedDestinationLayout,
  sourceLogical, sourcePhysical, mapLogical, mapPhysical,
  pairedTextureLogical, pairedTexturePhysical,
  destinationLogical, destinationPhysical,
  recordingSourceLease, recordingMapLease, recordingDestinationLease,
  inFlightSourceLease, inFlightMapLease, inFlightDestinationLease
>>

vars == <<
  mapMode, retainPool, recording, rendering, inFlight, recordingHasWork,
  sourceContents, opaqueInitialized, opaqueContents, encodedInput,
  decodedValid, decodedContents, decodedSourceInput,
  sourceLayout, mapLayout, destinationLayout,
  committedOpaqueInitialized, committedOpaqueContents, committedEncodedInput,
  committedDecodedValid, committedDecodedContents, committedDecodedSourceInput,
  committedSourceLayout, committedMapLayout, committedDestinationLayout,
  sourceLogical, sourcePhysical, mapLogical, mapPhysical,
  pairedTextureLogical, pairedTexturePhysical,
  destinationLogical, destinationPhysical,
  recordingSourceLease, recordingMapLease, recordingDestinationLease,
  inFlightSourceLease, inFlightMapLease, inFlightDestinationLease,
  lastOp, lastOutcome, lastRequestedMode, lastGraphics, lastExactShape
>>

EncodeValid(reqMode, graphics, exactShape) ==
  /\ recording
  /\ ~inFlight
  /\ ~rendering
  /\ graphics
  /\ exactShape
  /\ reqMode = mapMode
  /\ sourceLogical
  /\ mapLogical
  /\ sourceLayout = ResolveSource
  /\ mapLayout = ResolveDest

DecodeValid(reqMode, graphics, exactShape) ==
  /\ recording
  /\ ~inFlight
  /\ ~rendering
  /\ graphics
  /\ exactShape
  /\ reqMode = mapMode
  /\ mapLogical
  /\ destinationLogical
  /\ opaqueInitialized
  /\ mapLayout = ResolveSource
  /\ destinationLayout = ResolveDest

ResetAttemptMeta ==
  /\ lastRequestedMode' = 0
  /\ lastGraphics' = FALSE
  /\ lastExactShape' = FALSE

Init ==
  /\ mapMode \in Modes
  /\ retainPool \in BOOLEAN
  /\ recording = TRUE
  /\ rendering = FALSE
  /\ inFlight = FALSE
  /\ recordingHasWork = FALSE
  /\ sourceContents \in Contents
  /\ opaqueInitialized = FALSE
  /\ opaqueContents = 0
  /\ encodedInput = 0
  /\ decodedValid = FALSE
  /\ decodedContents = 0
  /\ decodedSourceInput = 0
  /\ sourceLayout = ResolveDest
  /\ mapLayout = ResolveDest
  /\ destinationLayout = ResolveSource
  /\ committedOpaqueInitialized = FALSE
  /\ committedOpaqueContents = 0
  /\ committedEncodedInput = 0
  /\ committedDecodedValid = FALSE
  /\ committedDecodedContents = 0
  /\ committedDecodedSourceInput = 0
  /\ committedSourceLayout = ResolveDest
  /\ committedMapLayout = ResolveDest
  /\ committedDestinationLayout = ResolveSource
  /\ sourceLogical = TRUE
  /\ sourcePhysical = TRUE
  /\ mapLogical = TRUE
  /\ mapPhysical = TRUE
  /\ pairedTextureLogical = TRUE
  /\ pairedTexturePhysical = TRUE
  /\ destinationLogical = TRUE
  /\ destinationPhysical = TRUE
  /\ recordingSourceLease = FALSE
  /\ recordingMapLease = FALSE
  /\ recordingDestinationLease = FALSE
  /\ inFlightSourceLease = FALSE
  /\ inFlightMapLease = FALSE
  /\ inFlightDestinationLease = FALSE
  /\ lastOp = "none"
  /\ lastOutcome = "none"
  /\ lastRequestedMode = 0
  /\ lastGraphics = FALSE
  /\ lastExactShape = FALSE

\* Mutate source texture contents when idle so round-trip invariants verify
\* decoded provenance even if sourceContents changes after encode.
WriteSource(newContents) ==
  /\ sourceLogical
  /\ ~recordingHasWork
  /\ ~recordingSourceLease
  /\ ~inFlight
  /\ ~inFlightSourceLease
  /\ sourceContents # newContents
  /\ sourceContents' = newContents
  /\ sourceLayout' = ResolveDest
  /\ committedSourceLayout' = ResolveDest
  /\ lastOp' = "writeSource"
  /\ lastOutcome' = "accepted"
  /\ ResetAttemptMeta
  /\ UNCHANGED <<
       mapMode, retainPool, recording, rendering, inFlight, recordingHasWork,
       opaqueInitialized, opaqueContents, encodedInput,
       decodedValid, decodedContents, decodedSourceInput,
       mapLayout, destinationLayout,
       committedOpaqueInitialized, committedOpaqueContents, committedEncodedInput,
       committedDecodedValid, committedDecodedContents, committedDecodedSourceInput,
       committedMapLayout, committedDestinationLayout,
       sourceLogical, sourcePhysical, mapLogical, mapPhysical,
       pairedTextureLogical, pairedTexturePhysical,
       destinationLogical, destinationPhysical,
       recordingSourceLease, recordingMapLease, recordingDestinationLease,
       inFlightSourceLease, inFlightMapLease, inFlightDestinationLease
     >>

BeginRendering ==
  /\ recording
  /\ ~inFlight
  /\ ~rendering
  /\ rendering' = TRUE
  /\ recordingHasWork' = TRUE
  /\ lastOp' = "beginRendering"
  /\ lastOutcome' = "accepted"
  /\ ResetAttemptMeta
  /\ UNCHANGED <<
       mapMode, retainPool, recording, inFlight,
       sourceContents, opaqueInitialized, opaqueContents, encodedInput,
       decodedValid, decodedContents, decodedSourceInput,
       sourceLayout, mapLayout, destinationLayout, committedVars,
       sourceLogical, sourcePhysical, mapLogical, mapPhysical,
       pairedTextureLogical, pairedTexturePhysical,
       destinationLogical, destinationPhysical,
       recordingSourceLease, recordingMapLease, recordingDestinationLease,
       inFlightSourceLease, inFlightMapLease, inFlightDestinationLease
     >>

EndRendering ==
  /\ recording
  /\ rendering
  /\ rendering' = FALSE
  /\ lastOp' = "endRendering"
  /\ lastOutcome' = "accepted"
  /\ ResetAttemptMeta
  /\ UNCHANGED <<
       mapMode, retainPool, recording, inFlight, recordingHasWork,
       sourceContents, opaqueInitialized, opaqueContents, encodedInput,
       decodedValid, decodedContents, decodedSourceInput,
       sourceLayout, mapLayout, destinationLayout, committedVars,
       sourceLogical, sourcePhysical, mapLogical, mapPhysical,
       pairedTextureLogical, pairedTexturePhysical,
       destinationLogical, destinationPhysical,
       recordingSourceLease, recordingMapLease, recordingDestinationLease,
       inFlightSourceLease, inFlightMapLease, inFlightDestinationLease
     >>

PrepareEncode ==
  /\ recording
  /\ ~inFlight
  /\ ~rendering
  /\ sourceLogical
  /\ mapLogical
  /\ ~(sourceLayout = ResolveSource /\ mapLayout = ResolveDest)
  /\ sourceLayout' = ResolveSource
  /\ mapLayout' = ResolveDest
  /\ recordingSourceLease' = (recordingSourceLease \/ retainPool)
  /\ recordingMapLease' = TRUE
  /\ recordingHasWork' = TRUE
  /\ lastOp' = "prepareEncode"
  /\ lastOutcome' = "accepted"
  /\ ResetAttemptMeta
  /\ UNCHANGED <<
       mapMode, retainPool, recording, rendering, inFlight,
       sourceContents, opaqueInitialized, opaqueContents, encodedInput,
       decodedValid, decodedContents, decodedSourceInput,
       destinationLayout, committedVars,
       sourceLogical, sourcePhysical, mapLogical, mapPhysical,
       pairedTextureLogical, pairedTexturePhysical,
       destinationLogical, destinationPhysical,
       recordingDestinationLease,
       inFlightSourceLease, inFlightMapLease, inFlightDestinationLease
     >>

PrepareDecode ==
  /\ recording
  /\ ~inFlight
  /\ ~rendering
  /\ mapLogical
  /\ destinationLogical
  /\ opaqueInitialized
  /\ ~(mapLayout = ResolveSource /\ destinationLayout = ResolveDest)
  /\ mapLayout' = ResolveSource
  /\ destinationLayout' = ResolveDest
  /\ recordingMapLease' = TRUE
  /\ recordingDestinationLease' = (recordingDestinationLease \/ retainPool)
  /\ recordingHasWork' = TRUE
  /\ lastOp' = "prepareDecode"
  /\ lastOutcome' = "accepted"
  /\ ResetAttemptMeta
  /\ UNCHANGED <<
       mapMode, retainPool, recording, rendering, inFlight,
       sourceContents, opaqueInitialized, opaqueContents, encodedInput,
       decodedValid, decodedContents, decodedSourceInput,
       sourceLayout, committedVars,
       sourceLogical, sourcePhysical, mapLogical, mapPhysical,
       pairedTextureLogical, pairedTexturePhysical,
       destinationLogical, destinationPhysical,
       recordingSourceLease,
       inFlightSourceLease, inFlightMapLease, inFlightDestinationLease
     >>

\* encodeSamplerFeedback calls queues.retainInternalResource, retaining BOTH
\* source and map even when retainPool = FALSE.
AttemptEncode(reqMode, graphics, exactShape) ==
  /\ IF EncodeValid(reqMode, graphics, exactShape)
     THEN /\ opaqueInitialized' = TRUE
          /\ opaqueContents' = EncodedContents(mapMode, sourceContents)
          /\ encodedInput' = sourceContents
          /\ recordingSourceLease' = TRUE
          /\ recordingMapLease' = TRUE
          /\ recordingHasWork' = TRUE
          /\ UNCHANGED <<
               mapMode, retainPool, recording, rendering, inFlight,
               sourceContents, decodedValid, decodedContents, decodedSourceInput,
               sourceLayout, mapLayout, destinationLayout, committedVars,
               sourceLogical, sourcePhysical, mapLogical, mapPhysical,
               pairedTextureLogical, pairedTexturePhysical,
               destinationLogical, destinationPhysical,
               recordingDestinationLease,
               inFlightSourceLease, inFlightMapLease, inFlightDestinationLease
             >>
          /\ lastOutcome' = "accepted"
          /\ lastRequestedMode' = reqMode
          /\ lastGraphics' = graphics
          /\ lastExactShape' = exactShape
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
          /\ ResetAttemptMeta
  /\ lastOp' = "encode"

\* decodeSamplerFeedback calls queues.retainInternalResource, retaining BOTH
\* map and destination even when retainPool = FALSE.
AttemptDecode(reqMode, graphics, exactShape) ==
  /\ IF DecodeValid(reqMode, graphics, exactShape)
     THEN /\ decodedValid' = TRUE
          /\ decodedContents' = opaqueContents
          /\ decodedSourceInput' = encodedInput
          /\ recordingMapLease' = TRUE
          /\ recordingDestinationLease' = TRUE
          /\ recordingHasWork' = TRUE
          /\ UNCHANGED <<
               mapMode, retainPool, recording, rendering, inFlight,
               sourceContents, opaqueInitialized, opaqueContents, encodedInput,
               sourceLayout, mapLayout, destinationLayout, committedVars,
               sourceLogical, sourcePhysical, mapLogical, mapPhysical,
               pairedTextureLogical, pairedTexturePhysical,
               destinationLogical, destinationPhysical,
               recordingSourceLease,
               inFlightSourceLease, inFlightMapLease, inFlightDestinationLease
             >>
          /\ lastOutcome' = "accepted"
          /\ lastRequestedMode' = reqMode
          /\ lastGraphics' = graphics
          /\ lastExactShape' = exactShape
     ELSE /\ UNCHANGED coreVars
          /\ lastOutcome' = "rejected"
          /\ ResetAttemptMeta
  /\ lastOp' = "decode"

DestroySource ==
  /\ sourceLogical
  /\ (retainPool \/ recordingSourceLease \/ inFlightSourceLease \/ (~recordingHasWork /\ ~inFlight))
  /\ sourceLogical' = FALSE
  /\ sourcePhysical' = (recordingSourceLease \/ inFlightSourceLease)
  /\ lastOp' = "destroySource"
  /\ lastOutcome' = "accepted"
  /\ ResetAttemptMeta
  /\ UNCHANGED <<
       mapMode, retainPool, recording, rendering, inFlight, recordingHasWork,
       sourceContents, opaqueInitialized, opaqueContents, encodedInput,
       decodedValid, decodedContents, decodedSourceInput,
       sourceLayout, mapLayout, destinationLayout, committedVars,
       mapLogical, mapPhysical, pairedTextureLogical, pairedTexturePhysical,
       destinationLogical, destinationPhysical,
       recordingSourceLease, recordingMapLease, recordingDestinationLease,
       inFlightSourceLease, inFlightMapLease, inFlightDestinationLease
     >>

DestroyMap ==
  LET nextMapPhys == (recordingMapLease \/ inFlightMapLease) IN
  /\ mapLogical
  /\ mapLogical' = FALSE
  /\ mapPhysical' = nextMapPhys
  /\ pairedTexturePhysical' = (pairedTextureLogical \/ nextMapPhys)
  /\ lastOp' = "destroyMap"
  /\ lastOutcome' = "accepted"
  /\ ResetAttemptMeta
  /\ UNCHANGED <<
       mapMode, retainPool, recording, rendering, inFlight, recordingHasWork,
       sourceContents, opaqueInitialized, opaqueContents, encodedInput,
       decodedValid, decodedContents, decodedSourceInput,
       sourceLayout, mapLayout, destinationLayout, committedVars,
       sourceLogical, sourcePhysical, pairedTextureLogical,
       destinationLogical, destinationPhysical,
       recordingSourceLease, recordingMapLease, recordingDestinationLease,
       inFlightSourceLease, inFlightMapLease, inFlightDestinationLease
     >>

DestroyPairedTexture ==
  /\ pairedTextureLogical
  /\ pairedTextureLogical' = FALSE
  /\ pairedTexturePhysical' = mapPhysical
  /\ lastOp' = "destroyPaired"
  /\ lastOutcome' = "accepted"
  /\ ResetAttemptMeta
  /\ UNCHANGED <<
       mapMode, retainPool, recording, rendering, inFlight, recordingHasWork,
       sourceContents, opaqueInitialized, opaqueContents, encodedInput,
       decodedValid, decodedContents, decodedSourceInput,
       sourceLayout, mapLayout, destinationLayout, committedVars,
       sourceLogical, sourcePhysical, mapLogical, mapPhysical,
       destinationLogical, destinationPhysical,
       recordingSourceLease, recordingMapLease, recordingDestinationLease,
       inFlightSourceLease, inFlightMapLease, inFlightDestinationLease
     >>

DestroyDestination ==
  /\ destinationLogical
  /\ (retainPool \/ recordingDestinationLease \/ inFlightDestinationLease \/ (~recordingHasWork /\ ~inFlight))
  /\ destinationLogical' = FALSE
  /\ destinationPhysical' = (recordingDestinationLease \/ inFlightDestinationLease)
  /\ lastOp' = "destroyDestination"
  /\ lastOutcome' = "accepted"
  /\ ResetAttemptMeta
  /\ UNCHANGED <<
       mapMode, retainPool, recording, rendering, inFlight, recordingHasWork,
       sourceContents, opaqueInitialized, opaqueContents, encodedInput,
       decodedValid, decodedContents, decodedSourceInput,
       sourceLayout, mapLayout, destinationLayout, committedVars,
       sourceLogical, sourcePhysical, mapLogical, mapPhysical,
       pairedTextureLogical, pairedTexturePhysical,
       recordingSourceLease, recordingMapLease, recordingDestinationLease,
       inFlightSourceLease, inFlightMapLease, inFlightDestinationLease
     >>

\* Abandoning an unsubmitted recording discards all staged GPU contents and
\* layout transitions, restoring the last committed GPU state.
AbandonRecording ==
  LET nextMapPhys == (mapLogical \/ inFlightMapLease) IN
  /\ recording
  /\ ~inFlight
  /\ recording' = FALSE
  /\ rendering' = FALSE
  /\ recordingHasWork' = FALSE
  /\ recordingSourceLease' = FALSE
  /\ recordingMapLease' = FALSE
  /\ recordingDestinationLease' = FALSE
  /\ opaqueInitialized' = committedOpaqueInitialized
  /\ opaqueContents' = committedOpaqueContents
  /\ encodedInput' = committedEncodedInput
  /\ decodedValid' = committedDecodedValid
  /\ decodedContents' = committedDecodedContents
  /\ decodedSourceInput' = committedDecodedSourceInput
  /\ sourceLayout' = committedSourceLayout
  /\ mapLayout' = committedMapLayout
  /\ destinationLayout' = committedDestinationLayout
  /\ sourcePhysical' = (sourceLogical \/ inFlightSourceLease)
  /\ mapPhysical' = nextMapPhys
  /\ pairedTexturePhysical' = (pairedTextureLogical \/ nextMapPhys)
  /\ destinationPhysical' = (destinationLogical \/ inFlightDestinationLease)
  /\ lastOp' = "abandonRecording"
  /\ lastOutcome' = "accepted"
  /\ ResetAttemptMeta
  /\ UNCHANGED <<
       mapMode, retainPool, inFlight, sourceContents, committedVars,
       sourceLogical, mapLogical, pairedTextureLogical, destinationLogical,
       inFlightSourceLease, inFlightMapLease, inFlightDestinationLease
     >>

BeginRecording ==
  /\ ~recording
  /\ ~inFlight
  /\ recording' = TRUE
  /\ lastOp' = "beginRecording"
  /\ lastOutcome' = "accepted"
  /\ ResetAttemptMeta
  /\ UNCHANGED <<
       mapMode, retainPool, rendering, inFlight, recordingHasWork,
       sourceContents, opaqueInitialized, opaqueContents, encodedInput,
       decodedValid, decodedContents, decodedSourceInput,
       sourceLayout, mapLayout, destinationLayout, committedVars,
       sourceLogical, sourcePhysical, mapLogical, mapPhysical,
       pairedTextureLogical, pairedTexturePhysical,
       destinationLogical, destinationPhysical,
       recordingSourceLease, recordingMapLease, recordingDestinationLease,
       inFlightSourceLease, inFlightMapLease, inFlightDestinationLease
     >>

Submit ==
  /\ recording
  /\ ~inFlight
  /\ ~rendering
  /\ recordingHasWork
  /\ recording' = FALSE
  /\ inFlight' = TRUE
  /\ recordingHasWork' = FALSE
  /\ recordingSourceLease' = FALSE
  /\ recordingMapLease' = FALSE
  /\ recordingDestinationLease' = FALSE
  /\ inFlightSourceLease' = recordingSourceLease
  /\ inFlightMapLease' = recordingMapLease
  /\ inFlightDestinationLease' = recordingDestinationLease
  /\ committedOpaqueInitialized' = opaqueInitialized
  /\ committedOpaqueContents' = opaqueContents
  /\ committedEncodedInput' = encodedInput
  /\ committedDecodedValid' = decodedValid
  /\ committedDecodedContents' = decodedContents
  /\ committedDecodedSourceInput' = decodedSourceInput
  /\ committedSourceLayout' = sourceLayout
  /\ committedMapLayout' = mapLayout
  /\ committedDestinationLayout' = destinationLayout
  /\ lastOp' = "submit"
  /\ lastOutcome' = "accepted"
  /\ ResetAttemptMeta
  /\ UNCHANGED <<
       mapMode, retainPool, rendering,
       sourceContents, opaqueInitialized, opaqueContents, encodedInput,
       decodedValid, decodedContents, decodedSourceInput,
       sourceLayout, mapLayout, destinationLayout,
       sourceLogical, sourcePhysical, mapLogical, mapPhysical,
       pairedTextureLogical, pairedTexturePhysical,
       destinationLogical, destinationPhysical
     >>

Complete ==
  LET nextMapPhys == mapLogical IN
  /\ inFlight
  /\ inFlight' = FALSE
  /\ inFlightSourceLease' = FALSE
  /\ inFlightMapLease' = FALSE
  /\ inFlightDestinationLease' = FALSE
  /\ sourcePhysical' = sourceLogical
  /\ mapPhysical' = nextMapPhys
  /\ pairedTexturePhysical' = (pairedTextureLogical \/ nextMapPhys)
  /\ destinationPhysical' = destinationLogical
  /\ lastOp' = "complete"
  /\ lastOutcome' = "accepted"
  /\ ResetAttemptMeta
  /\ UNCHANGED <<
       mapMode, retainPool, recording, rendering, recordingHasWork,
       sourceContents, opaqueInitialized, opaqueContents, encodedInput,
       decodedValid, decodedContents, decodedSourceInput,
       sourceLayout, mapLayout, destinationLayout, committedVars,
       sourceLogical, mapLogical, pairedTextureLogical, destinationLogical,
       recordingSourceLease, recordingMapLease, recordingDestinationLease
     >>

Next ==
  \/ \E c \in Contents : WriteSource(c)
  \/ BeginRendering
  \/ EndRendering
  \/ PrepareEncode
  \/ PrepareDecode
  \/ /\ recording
     /\ \E rm \in RequestedModes, g, es \in BOOLEAN : AttemptEncode(rm, g, es)
  \/ /\ ~recording
     /\ AttemptEncode(mapMode, TRUE, TRUE)
  \/ /\ recording
     /\ \E rm \in RequestedModes, g, es \in BOOLEAN : AttemptDecode(rm, g, es)
  \/ /\ ~recording
     /\ AttemptDecode(mapMode, TRUE, TRUE)
  \/ DestroySource
  \/ DestroyMap
  \/ DestroyPairedTexture
  \/ DestroyDestination
  \/ AbandonRecording
  \/ BeginRecording
  \/ Submit
  \/ Complete

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ mapMode \in Modes
  /\ retainPool \in BOOLEAN
  /\ recording \in BOOLEAN
  /\ rendering \in BOOLEAN
  /\ inFlight \in BOOLEAN
  /\ recordingHasWork \in BOOLEAN
  /\ sourceContents \in Contents
  /\ opaqueInitialized \in BOOLEAN
  /\ opaqueContents \in Contents
  /\ encodedInput \in Contents
  /\ decodedValid \in BOOLEAN
  /\ decodedContents \in Contents
  /\ decodedSourceInput \in Contents
  /\ sourceLayout \in Layouts
  /\ mapLayout \in Layouts
  /\ destinationLayout \in Layouts
  /\ committedOpaqueInitialized \in BOOLEAN
  /\ committedOpaqueContents \in Contents
  /\ committedEncodedInput \in Contents
  /\ committedDecodedValid \in BOOLEAN
  /\ committedDecodedContents \in Contents
  /\ committedDecodedSourceInput \in Contents
  /\ committedSourceLayout \in Layouts
  /\ committedMapLayout \in Layouts
  /\ committedDestinationLayout \in Layouts
  /\ sourceLogical \in BOOLEAN
  /\ sourcePhysical \in BOOLEAN
  /\ mapLogical \in BOOLEAN
  /\ mapPhysical \in BOOLEAN
  /\ pairedTextureLogical \in BOOLEAN
  /\ pairedTexturePhysical \in BOOLEAN
  /\ destinationLogical \in BOOLEAN
  /\ destinationPhysical \in BOOLEAN
  /\ recordingSourceLease \in BOOLEAN
  /\ recordingMapLease \in BOOLEAN
  /\ recordingDestinationLease \in BOOLEAN
  /\ inFlightSourceLease \in BOOLEAN
  /\ inFlightMapLease \in BOOLEAN
  /\ inFlightDestinationLease \in BOOLEAN
  /\ lastOp \in Operations
  /\ lastOutcome \in Outcomes
  /\ lastRequestedMode \in RequestedModes
  /\ lastGraphics \in BOOLEAN
  /\ lastExactShape \in BOOLEAN

PhysicalLifetimesExact ==
  /\ sourcePhysical =
       (sourceLogical \/ recordingSourceLease \/ inFlightSourceLease)
  /\ mapPhysical =
       (mapLogical \/ recordingMapLease \/ inFlightMapLease)
  /\ pairedTexturePhysical =
       (pairedTextureLogical \/ mapPhysical)
  /\ destinationPhysical =
       (destinationLogical \/ recordingDestinationLease \/ inFlightDestinationLease)

CommandBufferPhasesConsistent ==
  /\ ~(recording /\ inFlight)
  /\ (rendering => recording /\ ~inFlight)
  /\ (~recording =>
        /\ ~recordingHasWork
        /\ ~recordingSourceLease
        /\ ~recordingMapLease
        /\ ~recordingDestinationLease)
  /\ (~inFlight =>
        /\ ~inFlightSourceLease
        /\ ~inFlightMapLease
        /\ ~inFlightDestinationLease)

IdleRecordingMatchesCommittedState ==
  ~recordingHasWork =>
    /\ opaqueInitialized = committedOpaqueInitialized
    /\ opaqueContents = committedOpaqueContents
    /\ encodedInput = committedEncodedInput
    /\ decodedValid = committedDecodedValid
    /\ decodedContents = committedDecodedContents
    /\ decodedSourceInput = committedDecodedSourceInput
    /\ sourceLayout = committedSourceLayout
    /\ mapLayout = committedMapLayout
    /\ destinationLayout = committedDestinationLayout

EncodedAndDecodedSemanticsExact ==
  /\ (opaqueInitialized =>
        opaqueContents = EncodedContents(mapMode, encodedInput))
  /\ (mapMode = MipRegionUsedMode /\ decodedValid =>
        decodedContents = decodedSourceInput)
  /\ (mapMode = MinMipMode /\ decodedValid =>
        /\ decodedContents = EncodedContents(MinMipMode, decodedSourceInput)
        /\ (decodedSourceInput = 255 => decodedContents = 255))
  /\ (committedOpaqueInitialized =>
        committedOpaqueContents = EncodedContents(mapMode, committedEncodedInput))
  /\ (mapMode = MipRegionUsedMode /\ committedDecodedValid =>
        committedDecodedContents = committedDecodedSourceInput)
  /\ (mapMode = MinMipMode /\ committedDecodedValid =>
        /\ committedDecodedContents =
             EncodedContents(MinMipMode, committedDecodedSourceInput)
        /\ (committedDecodedSourceInput = 255 =>
              committedDecodedContents = 255))

Safety ==
  /\ PhysicalLifetimesExact
  /\ CommandBufferPhasesConsistent
  /\ IdleRecordingMatchesCommittedState
  /\ EncodedAndDecodedSemanticsExact

StepSafety ==
  [][
    /\ UNCHANGED <<mapMode, retainPool>>
    /\ (lastOutcome' = "rejected" => UNCHANGED coreVars)
    /\ (lastOp' \notin {"submit", "writeSource"} => UNCHANGED committedVars)
    /\ (lastOp' = "abandonRecording" =>
          /\ opaqueInitialized' = committedOpaqueInitialized
          /\ opaqueContents' = committedOpaqueContents
          /\ encodedInput' = committedEncodedInput
          /\ decodedValid' = committedDecodedValid
          /\ decodedContents' = committedDecodedContents
          /\ decodedSourceInput' = committedDecodedSourceInput
          /\ sourceLayout' = committedSourceLayout
          /\ mapLayout' = committedMapLayout
          /\ destinationLayout' = committedDestinationLayout)
    /\ (lastOp' = "encode" /\ lastOutcome' = "accepted" =>
          /\ EncodeValid(lastRequestedMode', lastGraphics', lastExactShape')
          /\ opaqueInitialized'
          /\ encodedInput' = sourceContents
          /\ opaqueContents' = EncodedContents(mapMode, sourceContents)
          /\ recordingSourceLease'
          /\ recordingMapLease')
    /\ (lastOp' = "decode" /\ lastOutcome' = "accepted" =>
          /\ DecodeValid(lastRequestedMode', lastGraphics', lastExactShape')
          /\ decodedValid'
          /\ decodedContents' = opaqueContents
          /\ decodedSourceInput' = encodedInput
          /\ recordingMapLease'
          /\ recordingDestinationLease')
  ]_vars

=============================================================================
