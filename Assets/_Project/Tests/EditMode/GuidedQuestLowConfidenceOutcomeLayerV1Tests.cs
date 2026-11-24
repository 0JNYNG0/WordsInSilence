using System;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class GuidedQuestLowConfidenceOutcomeLayerV1Tests
    {
        static GuidedQuestOutcome Evaluate(
            PersonalLiveCaptureState state,
            bool resultAvailable,
            string expectedWord,
            string predictedWord,
            bool accepted,
            bool expectedClassAccept,
            bool questSuccess,
            bool currentFailure = false,
            bool primaryFailure = false)
        {
            return GuidedQuestOutcomeEvaluator.EvaluateForRoute(
                PersonalLiveProfileMode.MultiPrototypeCandidate,
                state,
                resultAvailable,
                expectedWord,
                predictedWord,
                accepted,
                expectedClassAccept,
                questSuccess,
                currentFailure,
                primaryFailure);
        }

        [Test]
        public void Step9NFixedGreetingResult_IsStrictSuccess()
        {
            const string sessionId = "8a560d30-bfc7-4a79-b7a4-fc9f84e2efb4";
            const int rawFrames = 149;
            const int normalizedFrames = 149;
            const double greetingDistance = 0.43359546744230326;
            const double loveDistance = 0.68465225560791554;
            const double thankYouDistance = 0.76960002842331465;

            GuidedQuestOutcome outcome = Evaluate(
                PersonalLiveCaptureState.Completed,
                true,
                "GREETING",
                "GREETING",
                true,
                true,
                true);

            Assert.AreEqual(GuidedQuestOutcomeKind.StrictSuccess, outcome.Kind);
            Assert.IsTrue(outcome.GuidedProgressAllowed);
            Assert.IsTrue(outcome.StrictAccepted);
            Assert.IsTrue(outcome.StrictExpectedWordClassAccept);
            Assert.IsTrue(outcome.StrictQuestSuccess);
            Assert.AreEqual("8a560d30-bfc7-4a79-b7a4-fc9f84e2efb4", sessionId);
            Assert.AreEqual(149, rawFrames);
            Assert.AreEqual(rawFrames, normalizedFrames);
            Assert.AreEqual(0.43359546744230326, greetingDistance);
            Assert.AreEqual(0.68465225560791554, loveDistance);
            Assert.AreEqual(0.76960002842331465, thankYouDistance);
        }

        [Test]
        public void Step9QFixedLoveResult_IsStrictSuccess()
        {
            const string sessionId = "024e2b57-eea7-4f5e-b4c4-83a2a05b5383";
            const int rawFrames = 110;
            const int normalizedFrames = 110;

            GuidedQuestOutcome outcome = Evaluate(
                PersonalLiveCaptureState.Completed,
                true,
                "LOVE",
                "LOVE",
                true,
                true,
                true);

            Assert.AreEqual(GuidedQuestOutcomeKind.StrictSuccess, outcome.Kind);
            Assert.IsTrue(outcome.GuidedProgressAllowed);
            Assert.IsTrue(outcome.StrictQuestSuccess);
            Assert.AreEqual("024e2b57-eea7-4f5e-b4c4-83a2a05b5383", sessionId);
            Assert.AreEqual(110, rawFrames);
            Assert.AreEqual(rawFrames, normalizedFrames);
        }

        [Test]
        public void Step9RFixedThankYouResult_IsLowConfidenceCorrectWithoutStrictPromotion()
        {
            const string sessionId = "3aa213e9-1a94-454f-b53c-6ec67462d0ea";
            const int rawFrames = 92;
            const int normalizedFrames = 92;
            const double greetingDistance = 0.7380835730520241;
            const double loveDistance = 0.47699284383599257;
            const double thankYouDistance = 0.44939933821265732;
            const double approvedThreshold = 0.37709977431991587;

            GuidedQuestOutcome outcome = Evaluate(
                PersonalLiveCaptureState.Completed,
                true,
                "THANK_YOU",
                "THANK_YOU",
                false,
                false,
                false);

            Assert.AreEqual(GuidedQuestOutcomeKind.LowConfidenceCorrect, outcome.Kind);
            Assert.IsTrue(outcome.GuidedProgressAllowed);
            Assert.IsFalse(outcome.StrictAccepted);
            Assert.IsFalse(outcome.StrictExpectedWordClassAccept);
            Assert.IsFalse(outcome.StrictQuestSuccess);
            Assert.AreEqual("3aa213e9-1a94-454f-b53c-6ec67462d0ea", sessionId);
            Assert.AreEqual(92, rawFrames);
            Assert.AreEqual(rawFrames, normalizedFrames);
            Assert.AreEqual(0.7380835730520241, greetingDistance);
            Assert.AreEqual(0.47699284383599257, loveDistance);
            Assert.AreEqual(0.44939933821265732, thankYouDistance);
            Assert.AreEqual(0.37709977431991587, approvedThreshold);
            Assert.Greater(thankYouDistance, approvedThreshold);
        }

        [Test]
        public void CompletedPredictedMismatch_IsWrongWordEvenWhenAccepted()
        {
            GuidedQuestOutcome outcome = Evaluate(
                PersonalLiveCaptureState.Completed,
                true,
                "THANK_YOU",
                "LOVE",
                true,
                true,
                false);

            Assert.AreEqual(GuidedQuestOutcomeKind.WrongWord, outcome.Kind);
            Assert.IsFalse(outcome.GuidedProgressAllowed);
            Assert.AreEqual("THANK_YOU", outcome.ExpectedWord);
            Assert.AreEqual("LOVE", outcome.PredictedWord);
            Assert.IsTrue(outcome.StrictAccepted);
            Assert.IsFalse(outcome.StrictQuestSuccess);
        }

        [Test]
        public void BridgeFailure_HasTechnicalFailurePrecedenceOverCompletedStrictResult()
        {
            GuidedQuestOutcome outcome = Evaluate(
                PersonalLiveCaptureState.Completed,
                true,
                "LOVE",
                "LOVE",
                true,
                true,
                true,
                currentFailure: true);

            Assert.AreEqual(GuidedQuestOutcomeKind.TechnicalFailure, outcome.Kind);
            Assert.IsFalse(outcome.GuidedProgressAllowed);
            Assert.IsTrue(outcome.StrictQuestSuccess);
        }

        [Test]
        public void PrimaryFailure_HasTechnicalFailurePrecedenceOverCompletedStrictResult()
        {
            GuidedQuestOutcome outcome = Evaluate(
                PersonalLiveCaptureState.Completed,
                true,
                "GREETING",
                "GREETING",
                true,
                true,
                true,
                primaryFailure: true);

            Assert.AreEqual(GuidedQuestOutcomeKind.TechnicalFailure, outcome.Kind);
            Assert.IsFalse(outcome.GuidedProgressAllowed);
            Assert.IsTrue(outcome.StrictQuestSuccess);
        }

        [Test]
        public void EveryNonCompletedState_IsTechnicalFailure()
        {
            PersonalLiveCaptureState[] states =
            {
                PersonalLiveCaptureState.Idle,
                PersonalLiveCaptureState.Ready,
                PersonalLiveCaptureState.Capturing,
                PersonalLiveCaptureState.Processing,
                PersonalLiveCaptureState.Cancelled,
                PersonalLiveCaptureState.Failed,
            };

            foreach (PersonalLiveCaptureState state in states)
            {
                GuidedQuestOutcome outcome = Evaluate(
                    state,
                    true,
                    "GREETING",
                    "GREETING",
                    true,
                    true,
                    true);
                Assert.AreEqual(GuidedQuestOutcomeKind.TechnicalFailure, outcome.Kind, state.ToString());
                Assert.IsFalse(outcome.GuidedProgressAllowed, state.ToString());
            }
        }

        [Test]
        public void CompletedWithoutStrictResult_IsTechnicalFailure()
        {
            GuidedQuestOutcome outcome = Evaluate(
                PersonalLiveCaptureState.Completed,
                false,
                "LOVE",
                string.Empty,
                false,
                false,
                false);

            Assert.AreEqual(GuidedQuestOutcomeKind.TechnicalFailure, outcome.Kind);
            Assert.IsFalse(outcome.StrictResultAvailable);
            Assert.IsFalse(outcome.GuidedProgressAllowed);
        }

        [Test]
        public void LegacyAndDefaultRoutes_DoNotCreateGuidedOutcome()
        {
            Assert.AreEqual(0, (int)PersonalLiveProfileMode.LegacySingleMedoid);
            GuidedQuestOutcome outcome = GuidedQuestOutcomeEvaluator.EvaluateForRoute(
                PersonalLiveProfileMode.LegacySingleMedoid,
                PersonalLiveCaptureState.Completed,
                true,
                "GREETING",
                "GREETING",
                true,
                true,
                true,
                false,
                false);
            Assert.IsNull(outcome);
        }

        [Test]
        public void HarnessWiring_IsLegacyInactiveAndCandidateOnly()
        {
            GameObject owner = null;
            try
            {
                owner = new GameObject("Step9S-Guided-Outcome-Wiring");
                owner.SetActive(false);
                PersonalLiveRecognitionHarness harness =
                    owner.AddComponent<PersonalLiveRecognitionHarness>();
                Assert.AreEqual(PersonalLiveProfileMode.LegacySingleMedoid, harness.ProfileMode);
                Assert.IsNull(harness.GuidedOutcome);

                FieldInfo mode = typeof(PersonalLiveRecognitionHarness).GetField(
                    "profileMode",
                    BindingFlags.Instance | BindingFlags.NonPublic);
                Assert.NotNull(mode);
                mode.SetValue(harness, PersonalLiveProfileMode.MultiPrototypeCandidate);

                GuidedQuestOutcome candidateOutcome = harness.GuidedOutcome;
                Assert.NotNull(candidateOutcome);
                Assert.AreEqual(GuidedQuestOutcomeKind.TechnicalFailure, candidateOutcome.Kind);
                Assert.IsFalse(candidateOutcome.GuidedProgressAllowed);
            }
            finally
            {
                if (owner != null)
                    UnityEngine.Object.DestroyImmediate(owner);
            }
        }
    }
}
