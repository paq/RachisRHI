-------------------------- MODULE MetalDeviceLoss ---------------------------
(***************************************************************************)
(* Terminal device-loss publication, submission retirement accounting, and *)
(* destructor drain rules in the Metal 4 backend                           *)
(* (`src/Metal/MetalDevice.mm`, `src/Metal/MetalCommand.mm`, and           *)
(* `src/Metal/MetalSubmissionFeedback.h`).                                 *)
(*                                                                         *)
(* Distinguishes injected synthetic loss (`injectDeviceLoss`, which leaves *)
(* real GPU submissions executing on the native queue) from genuine native *)
(* driver/GPU loss (`driverLosesDevice` / commit failure). The first       *)
(* published loss reason is terminal across later loss observations, each  *)
(* submitted workload retires at most once, and destroying a device whose  *)
(* native GPU queue is still healthy must drain all in-flight work even    *)
(* when synthetic device loss has been reported.                           *)
(*                                                                         *)
(* Design rules controlled by CONSTANTS (each TRUE in the design):         *)
(*   - DrainSyntheticLoss: `~MetalDevice` drains pending GPU work unless   *)
(*     `nativeLost` holds (`MetalDeviceLoss-syntheticdrain.cfg` sets       *)
(*     FALSE, treating any `reported` loss as permission to skip drain).   *)
(*   - RetireOncePerSubmission: each submitted slot retires at most once   *)
(*     (`MetalDeviceLoss-doubleretire.cfg` sets FALSE).                    *)
(*                                                                         *)
(* Abstracted: Metal driver internals, `MTLSharedEvent` vs commit-feedback *)
(* sub-steps (covered by `MetalCommandRetention.tla`), and resource bytes. *)
(***************************************************************************)
EXTENDS Integers

CONSTANTS
    MaxSubmissions,             \* maximum submissions modeled in one behavior
    DrainSyntheticLoss,         \* TRUE: synthetic loss still drains before destroy
    RetireOncePerSubmission     \* TRUE: each submission slot retires at most once

Slots == 1..MaxSubmissions
LossReasons == {"none", "injected", "native"}

VARIABLES
    alive,          \* TRUE until ~MetalDevice completes
    nativeLost,     \* TRUE after an actual Metal driver/GPU submission failure
    reported,       \* TRUE once operationalState becomes lost for callers
    lossReason,     \* first terminal published loss reason in LossReasons
    submittedCount, \* number of submissions accepted onto the native queue
    retiredCount,   \* total queue retirement events observed
    slotRetired     \* per-slot retirement count in [Slots -> 0..2]

vars == <<
    alive, nativeLost, reported, lossReason,
    submittedCount, retiredCount, slotRetired
>>

PendingSubmissions == submittedCount - retiredCount

Init ==
    /\ alive = TRUE
    /\ nativeLost = FALSE
    /\ reported = FALSE
    /\ lossReason = "none"
    /\ submittedCount = 0
    /\ retiredCount = 0
    /\ slotRetired = [s \in Slots |-> 0]

\* Healthy active device accepts work up to MaxSubmissions.
submit ==
    /\ alive
    /\ ~reported
    /\ ~nativeLost
    /\ submittedCount < MaxSubmissions
    /\ submittedCount' = submittedCount + 1
    /\ UNCHANGED <<alive, nativeLost, reported, lossReason, retiredCount, slotRetired>>

\* GPU completion retires a submitted slot (at most once when RetireOncePerSubmission holds).
gpuCompletes(s) ==
    /\ alive
    /\ ~nativeLost
    /\ s <= submittedCount
    /\ IF RetireOncePerSubmission
       THEN slotRetired[s] = 0
       ELSE slotRetired[s] < 2
    /\ slotRetired' = [slotRetired EXCEPT ![s] = @ + 1]
    /\ retiredCount' = retiredCount + 1
    /\ UNCHANGED <<alive, nativeLost, reported, lossReason, submittedCount>>

\* Native Metal driver or hardware faults the underlying queue/device.
driverLosesDevice ==
    /\ alive
    /\ ~nativeLost
    /\ nativeLost' = TRUE
    /\ UNCHANGED <<alive, reported, lossReason, submittedCount, retiredCount, slotRetired>>

\* RACHIS_TESTING synthetic fault injection publishes terminal loss without native GPU fault.
injectLoss ==
    /\ alive
    /\ reported' = TRUE
    /\ lossReason' = IF reported THEN lossReason ELSE "injected"
    /\ UNCHANGED <<alive, nativeLost, submittedCount, retiredCount, slotRetired>>

\* Observing a genuine native fault publishes "native" only if no earlier loss was published.
observeNativeLoss ==
    /\ alive
    /\ nativeLost
    /\ reported' = TRUE
    /\ lossReason' = IF reported THEN lossReason ELSE "native"
    /\ UNCHANGED <<alive, nativeLost, submittedCount, retiredCount, slotRetired>>

\* ~MetalDevice skips queue drain only on genuine native loss.
destroy ==
    /\ alive
    /\ IF DrainSyntheticLoss
       THEN nativeLost \/ PendingSubmissions <= 0
       ELSE reported \/ PendingSubmissions <= 0
    /\ alive' = FALSE
    /\ UNCHANGED <<nativeLost, reported, lossReason, submittedCount, retiredCount, slotRetired>>

Next ==
    \/ submit
    \/ \E s \in Slots : gpuCompletes(s)
    \/ driverLosesDevice
    \/ injectLoss
    \/ observeNativeLoss
    \/ destroy

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ MaxSubmissions \in Nat \ {0}
    /\ DrainSyntheticLoss \in BOOLEAN
    /\ RetireOncePerSubmission \in BOOLEAN
    /\ alive \in BOOLEAN
    /\ nativeLost \in BOOLEAN
    /\ reported \in BOOLEAN
    /\ lossReason \in LossReasons
    /\ submittedCount \in 0..MaxSubmissions
    /\ retiredCount \in 0..(2 * MaxSubmissions)
    /\ slotRetired \in [Slots -> 0..2]

RetirementMatchesSubmissions ==
    /\ retiredCount <= submittedCount
    /\ \A s \in Slots :
        /\ slotRetired[s] <= 1
        /\ s > submittedCount => slotRetired[s] = 0

DrainBeforeSyntheticDestroy ==
    ~alive /\ ~nativeLost =>
        /\ PendingSubmissions = 0
        /\ \A s \in 1..submittedCount : slotRetired[s] >= 1

ReportedReasonConsistent ==
    /\ reported = (lossReason # "none")
    /\ lossReason = "native" => nativeLost

Safety ==
    /\ RetirementMatchesSubmissions
    /\ DrainBeforeSyntheticDestroy
    /\ ReportedReasonConsistent

StepSafety ==
    [][
        /\ (reported => reported' /\ lossReason' = lossReason)
        /\ (nativeLost => nativeLost')
        /\ (~alive => ~alive')
        /\ submittedCount <= submittedCount'
        /\ retiredCount <= retiredCount'
    ]_vars

=============================================================================
