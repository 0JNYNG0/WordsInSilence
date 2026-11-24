// F27 - the single place that turns ONE full-hand gameplay decision into ONE practice
// attempt. Both the Practice screen and the in-world NPC quest reach gameplay through the
// same shared adapter, so this factory is the one shape of a full-hand attempt: one
// authority, one set of metrics, one message.
//
// Nothing here scores, re-adapts or re-normalizes anything: the decision it is handed was
// already derived from the evidence the controller produced for this exact capture. The
// legacy 16-point snapshot is carried along UNCHANGED for diagnostics and is never rewritten
// to agree with the full-hand verdict.
using WordsInSilence.Recognition.Common;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.FullHandRuntime;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    public static class FullHandGameplayAttemptFactory
    {
        /// <summary>
        /// The total, label-preserving map. An abstention deliberately does NOT become
        /// LowConfidenceCorrect: that value already grants the NPC item, and an abstention
        /// is not evidence that the learner produced the prompted word.
        /// </summary>
        public static GuidedQuestOutcomeKind MapKind(FullHandGameplayDecisionKind kind)
        {
            switch (kind)
            {
                case FullHandGameplayDecisionKind.Accepted:
                    return GuidedQuestOutcomeKind.StrictSuccess;
                case FullHandGameplayDecisionKind.AcceptedDifferentWord:
                    return GuidedQuestOutcomeKind.WrongWord;
                case FullHandGameplayDecisionKind.NotDistinguished:
                    return GuidedQuestOutcomeKind.NotDistinguishedRetry;
                case FullHandGameplayDecisionKind.UnsupportedWord:
                    return GuidedQuestOutcomeKind.UnsupportedWord;
                default:
                    return GuidedQuestOutcomeKind.TechnicalFailure;
            }
        }

        public static GuidedQuestOutcome BuildOutcome(FullHandGameplayDecision decision)
        {
            GuidedQuestOutcomeKind kind = MapKind(decision.Kind);
            bool accepted = decision.Kind == FullHandGameplayDecisionKind.Accepted ||
                            decision.Kind == FullHandGameplayDecisionKind.AcceptedDifferentWord;
            bool success = decision.GrantsQuestSuccess;
            return GuidedQuestOutcomeEvaluator.CreateFullHandGameplayOutcome(
                kind,
                decision.ExpectedWord,
                decision.Top1,
                decision.Kind != FullHandGameplayDecisionKind.TechnicalRefusal,
                accepted,
                success,
                success,
                success,
                // learner-facing text only: the provenance stays typed on the decision and
                // is written to the technical panel and the log, never into this message
                FullHandGameplayFeedback.Korean(decision));
        }

        /// <summary>
        /// Metrics that describe the SELECTED authority. The legacy distances and prototype
        /// id are deliberately absent: reporting a 16-point snapshot beside a 46-point
        /// verdict would describe a decision nothing made.
        /// </summary>
        public static PracticeRecognitionMetrics BuildMetrics(FullHandGameplayDecision decision)
        {
            return new PracticeRecognitionMetrics
            {
                StrictAccepted = decision.Kind == FullHandGameplayDecisionKind.Accepted ||
                                 decision.Kind == FullHandGameplayDecisionKind.AcceptedDifferentWord,
                StrictExpectedWordClassAccept = decision.GrantsQuestSuccess,
                StrictQuestSuccess = decision.GrantsQuestSuccess,
                GreetingDistance = double.NaN,
                LoveDistance = double.NaN,
                ThankYouDistance = double.NaN,
                BestDistance = double.NaN,
                WinningPrototypeId = string.Empty,
                RawFrameCount = decision.FrameCount,
                NormalizedFrameCount = 0,
                ValidLandmarkCount = 0,
                InterpolatedLandmarkCount = 0,
            };
        }

        /// <summary>
        /// Builds the attempt whose authority is the full-hand decision.
        ///
        /// <paramref name="legacyRecognition"/> may be null: an enabled full-hand capture
        /// finishes coherently on its own evidence and is never vetoed by, and never requires,
        /// a legacy 16-point result. The legacy snapshot and its failures are carried for
        /// diagnostics only. <c>NormalizedSequence</c> is deliberately left null so that no
        /// consumer can mistake a 46-point attempt for a storable 16-point candidate.
        /// </summary>
        public static PracticeRecognitionAttempt BuildAttempt(
            FullHandGameplayDecision decision,
            string captureSessionId,
            PersonalLiveCaptureState captureState,
            PersonalLiveRecognitionSnapshot legacyRecognition,
            string profileId,
            PersonalLiveFailure legacyFailure,
            PersonalLivePrimaryFailureSnapshot legacyPrimaryFailure,
            FullHandGameCaptureResult evidence)
        {
            return new PracticeRecognitionAttempt(
                captureSessionId,
                decision.ExpectedWord,
                decision.Top1,
                captureState,
                BuildOutcome(decision),
                legacyRecognition,
                null,
                PersonalLiveProfileMode.MultiPrototypeCandidate.ToString(),
                profileId,
                double.NaN,
                legacyFailure,
                legacyPrimaryFailure,
                BuildMetrics(decision),
                (CommonSignClassification)null,
                evidence,
                decision);
        }
    }
}
