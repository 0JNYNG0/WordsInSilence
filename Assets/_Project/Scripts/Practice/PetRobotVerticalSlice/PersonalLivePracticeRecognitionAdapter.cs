using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using WordsInSilence.Recognition.Common;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.FullHandRuntime;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    /// <summary>Gameplay-only command/result seam over the approved candidate technical harness.</summary>
    public sealed class PersonalLivePracticeRecognitionAdapter : ISegmentedPersonalLivePracticeRecognitionAdapter
    {
        readonly PersonalLiveRecognitionHarness harness;
        readonly IReadOnlyDictionary<string, double> thresholds;

        /// <summary>
        /// F27. Snapshotted ONCE from the owning controller in this constructor. The
        /// controller itself received the mode once, from the harness, at construction, so
        /// no environment variable is read here or anywhere in the capture path.
        /// Off by default, which leaves the legacy path untouched.
        /// </summary>
        readonly bool fullHandGameplayEnabled;

        /// <summary>Whether the owning controller actually holds a full-hand processor.</summary>
        readonly bool fullHandProcessorPresent;

        /// <summary>True when the 46-point full-hand decision is this adapter's authority.</summary>
        public bool FullHandGameplayActive { get { return fullHandGameplayEnabled; } }

        /// <summary>
        /// Frozen Step 10O v1.6 common-2D classifier. Both the Practice route and the
        /// in-world quest route construct this adapter, so attaching the auxiliary
        /// signal here is the single shared seam: one implementation, one artifact
        /// identity, no duplicated model maths in either screen.
        /// </summary>
        readonly ICommonSignClassifier commonClassifier;

        public ICommonSignClassifier CommonClassifier { get { return commonClassifier; } }

        public PersonalLiveCaptureState CaptureState =>
            harness.Controller == null ? PersonalLiveCaptureState.Failed : harness.Controller.State;
        public string CaptureSessionId => harness.Controller == null ? string.Empty : harness.Controller.CaptureSessionId;
        public string ActiveProfileId => harness.Controller == null ? string.Empty : harness.Controller.RuntimeProvenanceId;
        public bool CandidateRouteActive =>
            harness.ProfileMode == PersonalLiveProfileMode.MultiPrototypeCandidate &&
            harness.Controller != null &&
            ((harness.Controller.IsDemoParticipantRuntime && harness.Controller.IsRuntimeReady) ||
             (harness.Controller.IsPersonalModeActive &&
              harness.Controller.ActivePersonalProfileId == PersonalMultiPrototypeRuntimeProfileLoader.ProfileId));

        public PersonalLivePracticeRecognitionAdapter(
            PersonalLiveRecognitionHarness harness,
            string candidateProfileDirectory = null,
            ICommonSignClassifier commonClassifier = null)
        {
            this.harness = harness ?? throw new ArgumentNullException(nameof(harness));
            this.commonClassifier = commonClassifier ?? CommonSignClassifierProvider.Shared;
            if (harness.ProfileMode != PersonalLiveProfileMode.MultiPrototypeCandidate)
                throw new InvalidOperationException("pet-robot practice requires the explicit MultiPrototypeCandidate route");
            if (!harness.EnsureControllerReady())
                throw new InvalidOperationException(BuildHarnessInitializationFailure(harness));
            // F27: the one and only read of the mode for this adapter's lifetime, taken from
            // the owning controller rather than from the environment.
            fullHandGameplayEnabled = harness.Controller.FullHandGameplayEnabled;
            fullHandProcessorPresent = harness.Controller.HasFullHandGameCaptureProcessor;
            if (harness.Controller.IsDemoParticipantRuntime)
            {
                // Demo participant route: the fixed personal profile loader is never invoked.
                IReadOnlyDictionary<string, double> demoThresholds = harness.Controller.DemoRuntimeThresholds;
                if (demoThresholds == null)
                    throw new InvalidOperationException("demo participant runtime thresholds are unavailable");
                thresholds = demoThresholds;
            }
            else
            {
                string profileDirectory = candidateProfileDirectory ?? Path.Combine(
                    Application.streamingAssetsPath,
                    "RecognitionCoreV1",
                    "PersonalProfiles",
                    ApprovedPersonalMultiPrototypeLiveRuntimeFactory.ProfileDirectoryName);
                PersonalMultiPrototypeRuntimeProfile profile =
                    PersonalMultiPrototypeRuntimeProfileLoader.Load(profileDirectory);
                thresholds = profile.Thresholds;
            }
            if (!CandidateRouteActive)
                throw new InvalidOperationException(BuildHarnessInitializationFailure(harness));
        }

        static string BuildHarnessInitializationFailure(PersonalLiveRecognitionHarness harness)
        {
            PersonalLiveRecognitionController controller = harness.Controller;
            if (controller == null)
                return "candidate harness initialization failed: controller unavailable; " +
                    "verify that the Step 9C bridge is assigned and bound";

            PersonalLiveFailure failure = controller.Failure;
            string detail = "state=" + controller.State +
                ", personalModeActive=" + controller.IsPersonalModeActive +
                ", activeProfileId=" +
                (string.IsNullOrWhiteSpace(controller.ActivePersonalProfileId)
                    ? "<empty>" : controller.ActivePersonalProfileId);
            if (failure != null)
                detail += ", failure=" + failure.Code + ": " + failure.Message;
            return "candidate harness initialization failed: " + detail;
        }

        public bool SetExpectedWord(string word)
        {
            if (!PersonalLiveRecognitionController.IsSupportedWord(word)) return false;
            if (word == "GREETING") harness.SetExpectedGreeting();
            else if (word == "LOVE") harness.SetExpectedLove();
            else harness.SetExpectedThankYou();
            return harness.Controller != null && harness.Controller.State == PersonalLiveCaptureState.Ready &&
                   harness.Controller.ExpectedWord == word;
        }

        public bool BeginCapture() => CandidateRouteActive && harness.StartCapture();

        public bool BeginCaptureWithPreRoll(IReadOnlyList<RawMediaPipeObservation> preRoll) =>
            CandidateRouteActive && harness.Controller != null &&
            harness.Controller.BeginCaptureWithPreRoll(preRoll);

        public PracticeRecognitionAttempt FinalizeCapture()
        {
            return FinalizeCaptureCore(() => harness.FinalizeCapture());
        }

        public PracticeRecognitionAttempt FinalizeCaptureWithTrailingTrim(double trailingTrimSeconds)
        {
            return FinalizeCaptureCore(() =>
                harness.Controller != null &&
                harness.Controller.FinalizeCaptureWithTrailingTrim(trailingTrimSeconds));
        }

        public PracticeRecognitionAttempt FailActiveCaptureFromSource(Exception exception)
        {
            PersonalLiveRecognitionController controller = harness.Controller;
            if (controller == null || !controller.FailActiveCaptureFromSource(exception) ||
                controller.State != PersonalLiveCaptureState.Failed)
                return null;

            if (fullHandGameplayEnabled)
            {
                // an explicit technical refusal, never a fallback to the legacy authority
                FullHandGameplayDecision refusal =
                    FullHandGameplayEvaluator.RefuseForSourceFailure(
                        controller.CaptureSessionId, controller.ExpectedWord,
                        exception == null ? null : exception.GetType().Name);
                return SnapshotFullHandAttempt(controller, refusal);
            }

            PracticeRecognitionAttempt attempt = SnapshotCurrentAttempt(controller);
            return attempt != null && attempt.Outcome != null &&
                attempt.Outcome.Kind == GuidedQuestOutcomeKind.TechnicalFailure
                    ? attempt
                    : null;
        }

        PracticeRecognitionAttempt FinalizeCaptureCore(Func<bool> finalize)
        {
            PersonalLiveRecognitionController controller = harness.Controller;
            if (controller == null)
                return null;
            finalize();
            return SnapshotCurrentAttempt(controller);
        }

        PracticeRecognitionAttempt SnapshotCurrentAttempt(PersonalLiveRecognitionController controller)
        {
            // F27: in gameplay mode the typed full-hand decision is the authority, so the
            // legacy fused outcome and the auxiliary common-2D classification below are not
            // consulted and not computed for this attempt.
            if (fullHandGameplayEnabled) return SnapshotFullHandAttempt(controller, null);

            GuidedQuestOutcome outcome = harness.GuidedOutcome;
            PersonalLiveRecognitionSnapshot result = controller.Result;
            string expectedWord = controller.ExpectedWord;
            double threshold;
            if (!thresholds.TryGetValue(expectedWord, out threshold)) threshold = double.NaN;
            RecognitionSequence normalized = result == null ? null : result.NormalizedSequence;

            // The common classification is computed BEFORE the outcome/fusion seam and is
            // passed into it, so the fused evidence state is a real function of the common
            // input rather than a post-hoc attachment.
            CommonSignClassification common = ClassifyCommon(result, expectedWord);
            GuidedQuestOutcome fusedOutcome = GuidedQuestOutcomeEvaluator.EvaluateForController(
                PersonalLiveProfileMode.MultiPrototypeCandidate, controller, common);

            var attempt = new PracticeRecognitionAttempt(
                controller.CaptureSessionId,
                expectedWord,
                result == null ? string.Empty : result.PredictedWord,
                controller.State,
                fusedOutcome ?? outcome,
                result,
                normalized,
                PersonalLiveProfileMode.MultiPrototypeCandidate.ToString(),
                controller.RuntimeProvenanceId,
                threshold,
                controller.Failure,
                controller.PrimaryFailure,
                null,
                common,
                controller.LastFullHandResult);

#if UNITY_EDITOR
            // Editor-only: the classification and the attempt result above are complete
            // before the trace is sealed. Sealing writes nothing and decides nothing; the
            // developer harness asks for an export later, after coordinator finalize.
            WordsInSilence.Recognition.Diagnostics.Common2DDiagnosticTraceBuffer
                .SafeMarkAttemptComplete(expectedWord);
#endif
            return attempt;
        }

        /// <summary>
        /// Frozen common-2D classification over the ACTIVE common-2D capture for this
        /// attempt: raw/post-gap x/y restored to the actual pixel plane with z=0 and
        /// normalized by the approved torso_normalize_2d contract.
        ///
        /// The pseudo-3D <see cref="RecognitionSequence"/> is deliberately NOT used here;
        /// it belongs to the strict and personal routes and has a different basis.
        /// Never throws into the capture path, never writes anything, and never changes
        /// the strict result.
        /// </summary>
        CommonSignClassification ClassifyCommon(
            PersonalLiveRecognitionSnapshot result, string expectedWord)
        {
            if (commonClassifier == null) return null;

            // Out-of-contract targets (the legacy LOVE route) short-circuit BEFORE any
            // common-2D capture, feature build or frozen forward pass is touched. The
            // fused evidence resolves to NotApplicable with no failure reason, and the
            // legacy strict outcome is untouched.
            if (!FrozenCommonSignModel.IsSupportedTarget(expectedWord)) return null;

            CommonTwoDimensionalCapture capture = result == null ? null : result.Common2DCapture;
            if (capture == null)
            {
                string detail = result == null
                    ? "no recognition snapshot was produced for this attempt"
                    : (string.IsNullOrEmpty(result.Common2DFailure)
                        ? "no active common-2D capture was produced for this attempt"
                        : result.Common2DFailure);
                return CommonSignClassification.Failed(
                    commonClassifier.ArtifactVersion,
                    commonClassifier.ArtifactSha256,
                    expectedWord,
                    CommonSignFailureReason.SequenceMissing,
                    detail);
            }
            return commonClassifier.Classify(
                capture.CopyCoordinates(),
                capture.CopyValidMask(),
                capture.FrameCount,
                expectedWord);
        }

        /// <summary>
        /// F27 gameplay attempt. Nothing is recomputed here: the capture, the model, the
        /// geometry and the gates all ran already, and this reads the evidence that the
        /// controller produced for THIS capture identity exactly once. Evidence belonging to
        /// another capture, absent evidence, or an enabled-but-unconfigured pipeline all
        /// produce an explicit typed refusal - never a fallback to the legacy 16-point route.
        ///
        /// The legacy snapshot, failure and primary failure are carried unchanged for
        /// diagnostics. A null legacy result is fine: an enabled full-hand capture finishes
        /// coherently on its own evidence and is never vetoed by a legacy preprocessing or
        /// runtime failure of the same capture.
        /// </summary>
        PracticeRecognitionAttempt SnapshotFullHandAttempt(
            PersonalLiveRecognitionController controller, FullHandGameplayDecision forcedRefusal)
        {
            string captureSessionId = controller.CaptureSessionId;
            FullHandGameCaptureResult evidence = controller.LastFullHandResult;
            // FullHandGameplayCaptureFinalized is the controller's own record that THIS
            // capture finalized on the full-hand route, so a cancelled, unfinalized or
            // invalidly re-finalized capture cannot consume a cached earlier result.
            FullHandGameplayDecision decision = forcedRefusal ??
                FullHandGameplayEvaluator.DecideForCapture(
                    evidence, controller.ExpectedWord, captureSessionId,
                    fullHandProcessorPresent, controller.FullHandGameplayCaptureFinalized);
            return FullHandGameplayAttemptFactory.BuildAttempt(
                decision,
                captureSessionId,
                controller.State,
                controller.Result,
                controller.RuntimeProvenanceId,
                controller.Failure,
                controller.PrimaryFailure,
                evidence);
        }

        public bool ResetCapture() => harness.ResetCapture();
    }
}
