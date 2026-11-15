------------------- MODULE RayTracingIndirectPublication --------------------
(***************************************************************************)
(* Transactional multi-stage dependency publication for D3D12 indirect     *)
(* ray-tracing dispatch across multiple recorded commands: application     *)
(* owners, manual-residency leases, private descriptor upload, and private *)
(* native argument buffer.                                                 *)
(*                                                                         *)
(* Native correspondence:                                                  *)
(*   - src/D3D12/D3D12AccelerationStructure.cpp                            *)
(*     (recordRayTracingIndirectDispatchArguments)                         *)
(*   - src/D3D12/D3D12Queue.cpp (retainCommandDependencies rollback)       *)
(*                                                                         *)
(* Abstraction boundary:                                                   *)
(*   Per-family dependency counts across up to two sequential dispatches   *)
(*   ensure that a failure in the second dispatch cannot hide behind set   *)
(*   saturation from the first dispatch. Only the "idle" phase is caller-  *)
(*   observable; intermediate phases model internal helper steps.          *)
(***************************************************************************)

EXTENDS Naturals

Application == "application"
Residency == "residency"
Upload == "upload"
NativeArguments == "nativeArguments"
Existing == "existing"
CountFamilies == {Application, Residency, Upload, NativeArguments}
Tokens == CountFamilies \cup {Existing}
MaxWork == 2
Phases == {"idle", "application", "upload", "native"}
Outcomes == {"none", "pending", "failed", "committed"}

VARIABLES
  retainsApplication, \* Pool-fixed application-owner retention flag
  usesResidency,      \* Whether the current dispatch touches manual residency
  phase,              \* Internal stage inside retainCommandDependencies
  retainedCounts,     \* Per-family retained dependency counts
  beforeCounts,       \* Snapshot of retainedCounts at transaction entry
  work,               \* Committed indirect dispatch count (0..MaxWork)
  workBefore,         \* Snapshot of work at transaction entry
  committedResCount,  \* Cumulative manual-residency dispatches committed
  observedCounts,     \* Caller-observed per-family counts at "idle"
  observedWork,       \* Caller-observed committed dispatch count
  observedResTarget,  \* Expected residency count at last observation
  observed,           \* Caller-observed token set (includes Existing baseline)
  lastOutcome         \* Status of the current/most recent transaction

vars == <<
  retainsApplication, usesResidency, phase,
  retainedCounts, beforeCounts, work, workBefore,
  committedResCount, observedCounts, observedWork,
  observedResTarget, observed, lastOutcome
>>

ZeroCounts == [f \in CountFamilies |-> 0]

TokensFromCounts(counts) ==
  {Existing} \cup {f \in CountFamilies : counts[f] > 0}

AppDelta == IF retainsApplication THEN 1 ELSE 0
ResDelta == IF usesResidency THEN 1 ELSE 0

Init ==
  /\ retainsApplication \in BOOLEAN
  /\ usesResidency = FALSE
  /\ phase = "idle"
  /\ retainedCounts = ZeroCounts
  /\ beforeCounts = ZeroCounts
  /\ work = 0
  /\ workBefore = 0
  /\ committedResCount = 0
  /\ observedCounts = ZeroCounts
  /\ observedWork = 0
  /\ observedResTarget = 0
  /\ observed = {Existing}
  /\ lastOutcome = "none"

RejectApplication(hasResidency) ==
  /\ phase = "idle"
  /\ work < MaxWork
  /\ usesResidency' = hasResidency
  /\ beforeCounts' = retainedCounts
  /\ workBefore' = work
  /\ lastOutcome' = "failed"
  /\ UNCHANGED <<
       retainsApplication, phase, retainedCounts, work,
       committedResCount, observedCounts, observedWork,
       observedResTarget, observed
     >>

StageApplication(hasResidency) ==
  LET nextApp == retainedCounts[Application] + AppDelta
      nextRes == retainedCounts[Residency] + (IF hasResidency THEN 1 ELSE 0)
  IN
  /\ phase = "idle"
  /\ work < MaxWork
  /\ usesResidency' = hasResidency
  /\ phase' = "application"
  /\ beforeCounts' = retainedCounts
  /\ workBefore' = work
  /\ retainedCounts' =
       [[retainedCounts EXCEPT ![Application] = nextApp]
                        EXCEPT ![Residency] = nextRes]
  /\ lastOutcome' = "pending"
  /\ UNCHANGED <<
       retainsApplication, work, committedResCount,
       observedCounts, observedWork, observedResTarget, observed
     >>

RejectUpload ==
  /\ phase = "application"
  /\ phase' = "idle"
  /\ retainedCounts' = beforeCounts
  /\ work' = workBefore
  /\ lastOutcome' = "failed"
  /\ UNCHANGED <<
       retainsApplication, usesResidency, beforeCounts, workBefore,
       committedResCount, observedCounts, observedWork,
       observedResTarget, observed
     >>

StageUpload ==
  /\ phase = "application"
  /\ phase' = "upload"
  /\ retainedCounts' =
       [retainedCounts EXCEPT ![Upload] = retainedCounts[Upload] + 1]
  /\ UNCHANGED <<
       retainsApplication, usesResidency, beforeCounts, work, workBefore,
       committedResCount, observedCounts, observedWork,
       observedResTarget, observed, lastOutcome
     >>

RejectNativeArguments ==
  /\ phase = "upload"
  /\ phase' = "idle"
  /\ retainedCounts' = beforeCounts
  /\ work' = workBefore
  /\ lastOutcome' = "failed"
  /\ UNCHANGED <<
       retainsApplication, usesResidency, beforeCounts, workBefore,
       committedResCount, observedCounts, observedWork,
       observedResTarget, observed
     >>

StageNativeArguments ==
  /\ phase = "upload"
  /\ phase' = "native"
  /\ retainedCounts' =
       [retainedCounts EXCEPT ![NativeArguments] =
          retainedCounts[NativeArguments] + 1]
  /\ UNCHANGED <<
       retainsApplication, usesResidency, beforeCounts, work, workBefore,
       committedResCount, observedCounts, observedWork,
       observedResTarget, observed, lastOutcome
     >>

Commit ==
  /\ phase = "native"
  /\ phase' = "idle"
  /\ work' = work + 1
  /\ committedResCount' = committedResCount + ResDelta
  /\ lastOutcome' = "committed"
  /\ UNCHANGED <<
       retainsApplication, usesResidency, retainedCounts, beforeCounts,
       workBefore, observedCounts, observedWork, observedResTarget, observed
     >>

Observe ==
  /\ phase = "idle"
  /\ observedCounts' = retainedCounts
  /\ observedWork' = work
  /\ observedResTarget' = committedResCount
  /\ observed' = TokensFromCounts(retainedCounts)
  /\ UNCHANGED <<
       retainsApplication, usesResidency, phase,
       retainedCounts, beforeCounts, work, workBefore,
       committedResCount, lastOutcome
     >>

Next ==
  \/ \E hasRes \in BOOLEAN :
       \/ RejectApplication(hasRes)
       \/ StageApplication(hasRes)
  \/ RejectUpload
  \/ StageUpload
  \/ RejectNativeArguments
  \/ StageNativeArguments
  \/ Commit
  \/ Observe

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ retainsApplication \in BOOLEAN
  /\ usesResidency \in BOOLEAN
  /\ phase \in Phases
  /\ retainedCounts \in [CountFamilies -> 0..MaxWork]
  /\ beforeCounts \in [CountFamilies -> 0..MaxWork]
  /\ work \in 0..MaxWork
  /\ workBefore \in 0..MaxWork
  /\ committedResCount \in 0..MaxWork
  /\ observedCounts \in [CountFamilies -> 0..MaxWork]
  /\ observedWork \in 0..MaxWork
  /\ observedResTarget \in 0..MaxWork
  /\ observed \in SUBSET Tokens
  /\ lastOutcome \in Outcomes

PartialPrefixCountsExact ==
  \/ /\ phase = "idle"
     /\ retainedCounts[Upload] = work
     /\ retainedCounts[NativeArguments] = work
     /\ retainedCounts[Application] = (IF retainsApplication THEN work ELSE 0)
     /\ retainedCounts[Residency] = committedResCount
  \/ /\ phase = "application"
     /\ work = workBefore
     /\ retainedCounts[Application] = beforeCounts[Application] + AppDelta
     /\ retainedCounts[Residency] = beforeCounts[Residency] + ResDelta
     /\ retainedCounts[Upload] = beforeCounts[Upload]
     /\ retainedCounts[NativeArguments] = beforeCounts[NativeArguments]
  \/ /\ phase = "upload"
     /\ work = workBefore
     /\ retainedCounts[Application] = beforeCounts[Application] + AppDelta
     /\ retainedCounts[Residency] = beforeCounts[Residency] + ResDelta
     /\ retainedCounts[Upload] = beforeCounts[Upload] + 1
     /\ retainedCounts[NativeArguments] = beforeCounts[NativeArguments]
  \/ /\ phase = "native"
     /\ work = workBefore
     /\ retainedCounts[Application] = beforeCounts[Application] + AppDelta
     /\ retainedCounts[Residency] = beforeCounts[Residency] + ResDelta
     /\ retainedCounts[Upload] = beforeCounts[Upload] + 1
     /\ retainedCounts[NativeArguments] = beforeCounts[NativeArguments] + 1

FailedPublicationRestoresExactPriorCounts ==
  phase = "idle" /\ lastOutcome = "failed" =>
    /\ retainedCounts = beforeCounts
    /\ work = workBefore

CommittedPublicationAdvancesAllRequiredCounts ==
  phase = "idle" /\ lastOutcome = "committed" =>
    /\ work = workBefore + 1
    /\ retainedCounts[Upload] = beforeCounts[Upload] + 1
    /\ retainedCounts[NativeArguments] = beforeCounts[NativeArguments] + 1
    /\ retainedCounts[Application] = beforeCounts[Application] + AppDelta
    /\ retainedCounts[Residency] = beforeCounts[Residency] + ResDelta

ObservationNeverSeesPartialPublication ==
  /\ Existing \in observed
  /\ observed = TokensFromCounts(observedCounts)
  /\ observedCounts[Upload] = observedWork
  /\ observedCounts[NativeArguments] = observedWork
  /\ observedCounts[Application] =
       (IF retainsApplication THEN observedWork ELSE 0)
  /\ observedCounts[Residency] = observedResTarget
  /\ (observedWork = 0 => observed = {Existing})
  /\ (observedWork > 0 =>
        /\ {Existing, Upload, NativeArguments} \subseteq observed
        /\ (retainsApplication => Application \in observed)
        /\ (~retainsApplication => Application \notin observed))

Safety ==
  /\ PartialPrefixCountsExact
  /\ FailedPublicationRestoresExactPriorCounts
  /\ CommittedPublicationAdvancesAllRequiredCounts
  /\ ObservationNeverSeesPartialPublication

StepSafety ==
  [][
    /\ retainsApplication' = retainsApplication
    /\ work' >= work
    /\ committedResCount' >= committedResCount
  ]_vars

=============================================================================
