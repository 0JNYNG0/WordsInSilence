using System;
using WordsInSilence.Recognition.Common;

namespace WordsInSilence.Recognition.LiveInput
{
    /// <summary>
    /// Candidate-only learning outcome layered over the immutable strict recognition result.
    /// It never changes acceptance, thresholds, predicted words, or quest-success values.
    /// </summary>
    public enum GuidedQuestOutcomeKind
    {
        StrictSuccess,
        LowConfidenceCorrect,
        WrongWord,
        TechnicalFailure,

        /// <summary>
        /// F27, produced ONLY by the full-hand gameplay route: the model ran and validly
        /// abstained. It is not a technical fault and not evidence that a different word
        /// was performed, and it never grants progress. It is deliberately NOT
        /// LowConfidenceCorrect, which already completes the quest.
        /// </summary>
        NotDistinguishedRetry,

        /// <summary>
        /// F27, produced ONLY by the full-hand gameplay route: the prompted word is outside
        /// the model vocabulary. No learner performance was measured and no progress is
        /// granted; there is no fallback to another route.
        /// </summary>
        UnsupportedWord,
    }

    public sealed class GuidedQuestOutcome
    {
        public GuidedQuestOutcomeKind Kind { get; }
        public string ExpectedWord { get; }
        public string PredictedWord { get; }
        public bool StrictResultAvailable { get; }
        public bool StrictAccepted { get; }
        public bool StrictExpectedWordClassAccept { get; }
        public bool StrictQuestSuccess { get; }
        public bool GuidedProgressAllowed { get; }
        public string Message { get; }

        /// <summary>
        /// Fused frozen common-2D evidence for this attempt. Always non-null so that
        /// removing or ignoring the common input is observable: it degrades to
        /// <see cref="CommonEvidenceState.Unavailable"/> instead of disappearing.
        /// The state never changes <see cref="Kind"/>, the strict booleans or
        /// <see cref="GuidedProgressAllowed"/>.
        /// </summary>
        public CommonEvidence Common { get; }

        internal GuidedQuestOutcome(
            GuidedQuestOutcomeKind kind,
            string expectedWord,
            string predictedWord,
            bool strictResultAvailable,
            bool strictAccepted,
            bool strictExpectedWordClassAccept,
            bool strictQuestSuccess,
            bool guidedProgressAllowed,
            string message,
            CommonEvidence common = null)
        {
            Common = common ?? CommonEvidence.Unavailable;
            Kind = kind;
            ExpectedWord = expectedWord ?? string.Empty;
            PredictedWord = predictedWord ?? string.Empty;
            StrictResultAvailable = strictResultAvailable;
            StrictAccepted = strictAccepted;
            StrictExpectedWordClassAccept = strictExpectedWordClassAccept;
            StrictQuestSuccess = strictQuestSuccess;
            GuidedProgressAllowed = guidedProgressAllowed;
            Message = message ?? string.Empty;
        }
    }

    public static class GuidedQuestOutcomeEvaluator
    {
        public const string StrictSuccessMessage =
            "Strict recognition succeeded.";
        public const string LowConfidenceCorrectMessage =
            "The expected word was predicted with low confidence; guided learning may continue without changing strict acceptance.";
        public const string WrongWordMessage =
            "A different word was predicted; keep the strict result and retry the guided prompt later.";
        public const string TechnicalFailureMessage =
            "The attempt did not produce a completed technical result and must not be scored as learner performance.";

        public static GuidedQuestOutcome EvaluateForController(
            PersonalLiveProfileMode profileMode,
            PersonalLiveRecognitionController controller)
        {
            return EvaluateForController(profileMode, controller, null);
        }

        /// <summary>
        /// Outcome/fusion seam WITH the frozen common-2D evidence supplied as an input.
        /// The strict decision below is computed exactly as before; the common
        /// classification only adds the fused evidence state, so it can neither promote,
        /// demote nor veto anything.
        /// </summary>
        public static GuidedQuestOutcome EvaluateForController(
            PersonalLiveProfileMode profileMode,
            PersonalLiveRecognitionController controller,
            CommonSignClassification commonClassification)
        {
            if (profileMode != PersonalLiveProfileMode.MultiPrototypeCandidate)
                return null;

            PersonalLiveRecognitionSnapshot result = controller == null ? null : controller.Result;
            return EvaluateForRoute(
                profileMode,
                controller == null ? PersonalLiveCaptureState.Failed : controller.State,
                result != null,
                controller == null ? string.Empty : controller.ExpectedWord,
                result == null ? string.Empty : result.PredictedWord,
                result != null && result.Accepted,
                result != null && result.ExpectedWordClassAccept,
                result != null && result.QuestSuccess,
                controller == null || controller.Failure != null,
                controller != null && controller.PrimaryFailure != null,
                commonClassification);
        }

        /// <summary>
        /// Deterministic candidate-only decision seam. The strict booleans are copied into
        /// the outcome for diagnostics and are never promoted or rewritten.
        /// </summary>
        public static GuidedQuestOutcome EvaluateForRoute(
            PersonalLiveProfileMode profileMode,
            PersonalLiveCaptureState captureState,
            bool strictResultAvailable,
            string expectedWord,
            string predictedWord,
            bool strictAccepted,
            bool strictExpectedWordClassAccept,
            bool strictQuestSuccess,
            bool hasCurrentFailure,
            bool hasPrimaryFailure)
        {
            return EvaluateForRoute(
                profileMode, captureState, strictResultAvailable, expectedWord, predictedWord,
                strictAccepted, strictExpectedWordClassAccept, strictQuestSuccess,
                hasCurrentFailure, hasPrimaryFailure, null);
        }

        /// <summary>
        /// Deterministic candidate-only decision seam WITH common evidence.
        ///
        /// The strict branch order below is unchanged and is computed without ever
        /// reading <paramref name="commonClassification"/>. The common input is fused
        /// only into <see cref="GuidedQuestOutcome.Common"/>, which is why:
        ///   Support           cannot promote anything to StrictSuccess,
        ///   NonSupport        cannot alone create WrongWord or TechnicalFailure and
        ///                     cannot veto an existing StrictSuccess,
        ///   TechnicalEvidence cannot alone mark the learner wrong,
        ///   NotApplicable     leaves the legacy LOVE route untouched.
        /// </summary>
        public static GuidedQuestOutcome EvaluateForRoute(
            PersonalLiveProfileMode profileMode,
            PersonalLiveCaptureState captureState,
            bool strictResultAvailable,
            string expectedWord,
            string predictedWord,
            bool strictAccepted,
            bool strictExpectedWordClassAccept,
            bool strictQuestSuccess,
            bool hasCurrentFailure,
            bool hasPrimaryFailure,
            CommonSignClassification commonClassification)
        {
            if (profileMode != PersonalLiveProfileMode.MultiPrototypeCandidate)
                return null;

            CommonEvidence common = CommonEvidence.From(commonClassification, expectedWord);

            if (captureState != PersonalLiveCaptureState.Completed ||
                !strictResultAvailable ||
                hasCurrentFailure ||
                hasPrimaryFailure ||
                !PersonalLiveRecognitionController.IsSupportedWord(expectedWord) ||
                !PersonalLiveRecognitionController.IsSupportedWord(predictedWord))
            {
                return Create(
                    GuidedQuestOutcomeKind.TechnicalFailure,
                    expectedWord,
                    predictedWord,
                    strictResultAvailable,
                    strictAccepted,
                    strictExpectedWordClassAccept,
                    strictQuestSuccess,
                    false,
                    TechnicalFailureMessage,
                    common);
            }

            if (!string.Equals(predictedWord, expectedWord, StringComparison.Ordinal))
            {
                return Create(
                    GuidedQuestOutcomeKind.WrongWord,
                    expectedWord,
                    predictedWord,
                    true,
                    strictAccepted,
                    strictExpectedWordClassAccept,
                    strictQuestSuccess,
                    false,
                    WrongWordMessage,
                    common);
            }

            if (strictQuestSuccess)
            {
                return Create(
                    GuidedQuestOutcomeKind.StrictSuccess,
                    expectedWord,
                    predictedWord,
                    true,
                    strictAccepted,
                    strictExpectedWordClassAccept,
                    true,
                    true,
                    StrictSuccessMessage,
                    common);
            }

            return Create(
                GuidedQuestOutcomeKind.LowConfidenceCorrect,
                expectedWord,
                predictedWord,
                true,
                strictAccepted,
                strictExpectedWordClassAccept,
                false,
                true,
                LowConfidenceCorrectMessage,
                common);
        }

        /// <summary>
        /// F27. The ONLY way the full-hand gameplay route may create an outcome. It performs
        /// no evaluation of its own: every value is supplied by the already finalized
        /// full-hand decision for that one capture. The legacy strict seam above is not
        /// consulted, not re-run and not rewritten, and no common-2D evidence is attached
        /// because none was computed on this route.
        /// </summary>
        public static GuidedQuestOutcome CreateFullHandGameplayOutcome(
            GuidedQuestOutcomeKind kind,
            string expectedWord,
            string predictedWord,
            bool resultAvailable,
            bool accepted,
            bool expectedWordClassAccept,
            bool questSuccess,
            bool progressAllowed,
            string message)
        {
            return new GuidedQuestOutcome(
                kind, expectedWord, predictedWord, resultAvailable, accepted,
                expectedWordClassAccept, questSuccess, progressAllowed, message,
                CommonEvidence.Unavailable);
        }

        static GuidedQuestOutcome Create(
            GuidedQuestOutcomeKind kind,
            string expectedWord,
            string predictedWord,
            bool strictResultAvailable,
            bool strictAccepted,
            bool strictExpectedWordClassAccept,
            bool strictQuestSuccess,
            bool guidedProgressAllowed,
            string message,
            CommonEvidence common)
        {
            return new GuidedQuestOutcome(
                kind,
                expectedWord,
                predictedWord,
                strictResultAvailable,
                strictAccepted,
                strictExpectedWordClassAccept,
                strictQuestSuccess,
                guidedProgressAllowed,
                message,
                common);
        }
    }
}
