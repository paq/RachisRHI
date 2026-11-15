--------------------- MODULE TextureResolvePublication ----------------------
(***************************************************************************)
(* Formal safety model for MSAA texture-resolve dependency staging,        *)
(* rollback on partial retention failure, and portable pipeline-state      *)
(* restoration across color, packed depth/stencil, and shader-depth modes: *)
(*   - include/Rachis/Device.h (Device::resolveTexture)                    *)
(*   - src/D3D12/D3D12ResourceCopies.cpp (resolveTexture,                  *)
(*     recordColorResolve, recordPackedDepthStencilResolve,                *)
(*     prepareDepthResolveDraw, retainDepthResolveDraw)                    *)
(*   - src/D3D12/D3D12Resource.cpp (creates the D32 resolve root           *)
(*     signature and helper pipeline used by the shader-depth mode)        *)
(*   - src/D3D12/D3D12CommandRecording.cpp                                 *)
(*     (prepareNativePipelineContinuation, NativePipelineContinuation::    *)
(*     restore), src/D3D12/D3D12Queue.cpp (retainCommandDependencies)      *)
(*                                                                         *)
(* Models source, destination, private shader-depth descriptors, optional  *)
(* swapchain target lease, and optional manual-residency heap retention;   *)
(* temporarily switches pipelineState to "internalShaderDepth" during the  *)
(* shaderDepth helper and restores savedPipelineState on commit/rollback.  *)
(*                                                                         *)
(* Trusted boundary: no-fail native command-list resolve/draw emission     *)
(* once every required dependency has been retained.                       *)
(***************************************************************************)
EXTENDS Naturals

Source == 0
Destination == 1
SourceDescriptor == 2
DestinationDescriptor == 3
Lease == 4
Residency == 5
ExistingDependency == 6

Dependencies == {
  Source,
  Destination,
  SourceDescriptor,
  DestinationDescriptor,
  Lease,
  Residency,
  ExistingDependency
}

Modes == {"color", "packed", "shaderDepth"}
ModeValues == Modes \cup {"none"}
Phases == {"idle", "prepared", "retaining"}
PortablePipelineStates == {"none", "graphics", "compute", "rayTracing"}
AllPipelineStates == PortablePipelineStates \cup {"internalShaderDepth"}
MaxPublications == 2

BaselineCounts ==
  [d \in Dependencies |-> IF d = ExistingDependency THEN 1 ELSE 0]

Required(mode, hasLease, hasResidency) ==
  {Source, Destination}
  \cup (IF mode = "shaderDepth" THEN {SourceDescriptor, DestinationDescriptor} ELSE {})
  \cup (IF hasLease THEN {Lease} ELSE {})
  \cup (IF hasResidency THEN {Residency} ELSE {})

Incremented(counts, additions) ==
  [d \in Dependencies |-> counts[d] + (IF d \in additions THEN 1 ELSE 0)]

CommittedShape(counts) ==
  /\ counts[ExistingDependency] = 1
  /\ counts[Source] = counts[Destination]
  /\ counts[SourceDescriptor] = counts[DestinationDescriptor]
  /\ counts[SourceDescriptor] <= counts[Source]
  /\ counts[Lease] <= counts[Source]
  /\ counts[Residency] <= counts[Source]
  /\ counts[Source] <= MaxPublications

VARIABLES
  phase,                 \* Current synchronous resolve publication phase.
  pendingMode,           \* Active resolve lowering mode ("none" when idle).
  pendingLease,          \* Whether current resolve targets a swapchain lease.
  pendingResidency,      \* Whether current resolve retains manual residency.
  stagedDependencies,    \* Subset of Required(...) retained so far in batch.
  commandDependencies,   \* Working per-dependency count on command buffer.
  dependenciesBefore,    \* Snapshot of commandDependencies before prepare.
  committedDependencies, \* Last committed per-dependency count.
  recorded,              \* Whether at least one resolve has been recorded.
  recordedBefore,        \* Snapshot of recorded before prepare.
  failed,                \* Whether the latest resolve attempt rolled back.
  publicationCount,      \* Number of committed resolves (0..MaxPublications).
  observedDependencies,  \* Snapshot captured by external caller observation.
  pipelineState,         \* Current native/portable pipeline state on command.
  savedPipelineState     \* Portable pipeline state saved on helper entry.

vars == <<
  phase,
  pendingMode,
  pendingLease,
  pendingResidency,
  stagedDependencies,
  commandDependencies,
  dependenciesBefore,
  committedDependencies,
  recorded,
  recordedBefore,
  failed,
  publicationCount,
  observedDependencies,
  pipelineState,
  savedPipelineState
>>

Init ==
  /\ phase = "idle"
  /\ pendingMode = "none"
  /\ pendingLease = FALSE
  /\ pendingResidency = FALSE
  /\ stagedDependencies = {}
  /\ commandDependencies = BaselineCounts
  /\ dependenciesBefore = BaselineCounts
  /\ committedDependencies = BaselineCounts
  /\ recorded = FALSE
  /\ recordedBefore = FALSE
  /\ failed = FALSE
  /\ publicationCount = 0
  /\ observedDependencies = BaselineCounts
  /\ pipelineState = "none"
  /\ savedPipelineState = "none"

bindPipelineState(state) ==
  /\ phase = "idle"
  /\ state \in PortablePipelineStates
  /\ pipelineState' = state
  /\ savedPipelineState' = state
  /\ UNCHANGED <<
       phase,
       pendingMode,
       pendingLease,
       pendingResidency,
       stagedDependencies,
       commandDependencies,
       dependenciesBefore,
       committedDependencies,
       recorded,
       recordedBefore,
       failed,
       publicationCount,
       observedDependencies
     >>

\* Native interface lookup and private descriptor allocation may fail before
\* any command-owned dependency or pipeline state is mutated.
rejectPreparation(mode, hasLease, hasResidency) ==
  /\ phase = "idle"
  /\ publicationCount < MaxPublications
  /\ mode \in Modes
  /\ hasLease \in BOOLEAN
  /\ hasResidency \in BOOLEAN
  /\ pendingMode' = "none"
  /\ pendingLease' = FALSE
  /\ pendingResidency' = FALSE
  /\ stagedDependencies' = {}
  /\ dependenciesBefore' = commandDependencies
  /\ recordedBefore' = recorded
  /\ failed' = TRUE
  /\ savedPipelineState' = pipelineState
  /\ UNCHANGED <<
       phase,
       commandDependencies,
       committedDependencies,
       recorded,
       publicationCount,
       observedDependencies,
       pipelineState
     >>

\* Preparing shaderDepth temporarily switches native pipelineState to the
\* internal helper pipeline; commit and rejectRetention must both restore it.
prepare(mode, hasLease, hasResidency) ==
  /\ phase = "idle"
  /\ publicationCount < MaxPublications
  /\ mode \in Modes
  /\ hasLease \in BOOLEAN
  /\ hasResidency \in BOOLEAN
  /\ phase' = "prepared"
  /\ pendingMode' = mode
  /\ pendingLease' = hasLease
  /\ pendingResidency' = hasResidency
  /\ stagedDependencies' = {}
  /\ dependenciesBefore' = commandDependencies
  /\ recordedBefore' = recorded
  /\ failed' = FALSE
  /\ savedPipelineState' = pipelineState
  /\ IF mode = "shaderDepth"
     THEN pipelineState' = "internalShaderDepth"
     ELSE UNCHANGED pipelineState
  /\ UNCHANGED <<
       commandDependencies,
       committedDependencies,
       recorded,
       publicationCount,
       observedDependencies
     >>

retainDependency(dependency) ==
  /\ phase \in {"prepared", "retaining"}
  /\ dependency \in Required(pendingMode, pendingLease, pendingResidency)
  /\ ~(dependency \in stagedDependencies)
  /\ phase' = "retaining"
  /\ stagedDependencies' = stagedDependencies \cup {dependency}
  /\ commandDependencies' =
       [commandDependencies EXCEPT ![dependency] = commandDependencies[dependency] + 1]
  /\ UNCHANGED <<
       pendingMode,
       pendingLease,
       pendingResidency,
       dependenciesBefore,
       committedDependencies,
       recorded,
       recordedBefore,
       failed,
       publicationCount,
       observedDependencies,
       pipelineState,
       savedPipelineState
     >>

\* Rejection at any dependency restores both commandDependencies and the
\* caller's portable pipeline state captured before preparation.
rejectRetention ==
  /\ phase \in {"prepared", "retaining"}
  /\ phase' = "idle"
  /\ pendingMode' = "none"
  /\ pendingLease' = FALSE
  /\ pendingResidency' = FALSE
  /\ stagedDependencies' = {}
  /\ commandDependencies' = dependenciesBefore
  /\ recorded' = recordedBefore
  /\ failed' = TRUE
  /\ pipelineState' = savedPipelineState
  /\ UNCHANGED <<
       dependenciesBefore,
       committedDependencies,
       recordedBefore,
       publicationCount,
       observedDependencies,
       savedPipelineState
     >>

commit ==
  /\ phase = "retaining"
  /\ stagedDependencies = Required(pendingMode, pendingLease, pendingResidency)
  /\ phase' = "idle"
  /\ pendingMode' = "none"
  /\ pendingLease' = FALSE
  /\ pendingResidency' = FALSE
  /\ stagedDependencies' = {}
  /\ committedDependencies' = commandDependencies
  /\ recorded' = TRUE
  /\ failed' = FALSE
  /\ publicationCount' = publicationCount + 1
  /\ pipelineState' = savedPipelineState
  /\ UNCHANGED <<
       commandDependencies,
       dependenciesBefore,
       recordedBefore,
       observedDependencies,
       savedPipelineState
     >>

observe ==
  /\ phase = "idle"
  /\ observedDependencies' = commandDependencies
  /\ UNCHANGED <<
       phase,
       pendingMode,
       pendingLease,
       pendingResidency,
       stagedDependencies,
       commandDependencies,
       dependenciesBefore,
       committedDependencies,
       recorded,
       recordedBefore,
       failed,
       publicationCount,
       pipelineState,
       savedPipelineState
     >>

Next ==
  \/ \E s \in PortablePipelineStates : bindPipelineState(s)
  \/ \E m \in Modes, l, r \in BOOLEAN : rejectPreparation(m, l, r)
  \/ \E m \in Modes, l, r \in BOOLEAN : prepare(m, l, r)
  \/ \E d \in Dependencies : retainDependency(d)
  \/ rejectRetention
  \/ commit
  \/ observe

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ phase \in Phases
  /\ pendingMode \in ModeValues
  /\ pendingLease \in BOOLEAN
  /\ pendingResidency \in BOOLEAN
  /\ stagedDependencies \subseteq Dependencies
  /\ commandDependencies \in [Dependencies -> 0..(MaxPublications + 1)]
  /\ dependenciesBefore \in [Dependencies -> 0..MaxPublications]
  /\ committedDependencies \in [Dependencies -> 0..MaxPublications]
  /\ observedDependencies \in [Dependencies -> 0..MaxPublications]
  /\ recorded \in BOOLEAN
  /\ recordedBefore \in BOOLEAN
  /\ failed \in BOOLEAN
  /\ publicationCount \in 0..MaxPublications
  /\ pipelineState \in AllPipelineStates
  /\ savedPipelineState \in PortablePipelineStates

PrivatePrefixMatchesEverySuccessfulRetain ==
  \/ (phase = "idle"
      /\ stagedDependencies = {}
      /\ pendingMode = "none"
      /\ ~pendingLease
      /\ ~pendingResidency)
  \/ (phase \in {"prepared", "retaining"}
      /\ pendingMode \in Modes
      /\ stagedDependencies \subseteq Required(pendingMode, pendingLease, pendingResidency)
      /\ dependenciesBefore = committedDependencies
      /\ commandDependencies = Incremented(dependenciesBefore, stagedDependencies)
      /\ recorded = recordedBefore
      /\ (pendingMode = "shaderDepth" <=> pipelineState = "internalShaderDepth"))

FailedPublicationRestoresExactPriorState ==
  failed =>
    /\ phase = "idle"
    /\ stagedDependencies = {}
    /\ commandDependencies = dependenciesBefore
    /\ commandDependencies = committedDependencies
    /\ recorded = recordedBefore
    /\ pipelineState = savedPipelineState

CommittedPublicationIsExact ==
  phase = "idle" =>
    /\ commandDependencies = committedDependencies
    /\ committedDependencies[Source] = publicationCount
    /\ CommittedShape(committedDependencies)
    /\ (recorded <=> publicationCount > 0)

ObservationNeverSeesPartialPublication ==
  CommittedShape(observedDependencies)

ResolvePreservesPortablePipelineState ==
  phase = "idle" =>
    /\ pipelineState \in PortablePipelineStates
    /\ pipelineState = savedPipelineState

Safety ==
  /\ PrivatePrefixMatchesEverySuccessfulRetain
  /\ FailedPublicationRestoresExactPriorState
  /\ CommittedPublicationIsExact
  /\ ObservationNeverSeesPartialPublication
  /\ ResolvePreservesPortablePipelineState

=============================================================================
