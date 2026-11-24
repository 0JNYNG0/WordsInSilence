// F27 EditMode tests for the full-hand gameplay connection.
//
// Everything is synthetic and in memory: an explicit F20 in-memory common bank, explicit
// operator-style gates written by the test, and raw capture geometry built by the same
// helpers the existing F20/F21 tests use. There is no webcam, no scene, no private model,
// no persistentDataPath, no Application persistence, no network and no production cutoff.
// The vocabulary used here is GREETING / THANK_YOU, which is what the real model actually
// knows; LOVE is exercised precisely because it is NOT in the vocabulary.
//
// Nothing here asserts recognition accuracy, human performance or any probability of
// correctness. "Accepted" is asserted as a bank/gate decision only.
// NOTHING IN THIS FILE HAS BEEN EXECUTED; the operator runs it.
using System;
using System.Collections.Generic;
using NUnit.Framework;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Practice.PetRobotWorldIntegration;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.FullHandRuntime;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.PhysicalCompatibility;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class FullHandGameplayIntegrationTests
    {
        const int Pose = 33, Hand = 21, Width = 1280, Height = 720;
        const int SourceFrames = 4;
        const string Participant = "participant_f27";
        const string Greeting = "GREETING";
        const string ThankYou = "THANK_YOU";
        const string Love = "LOVE";
        static readonly string[] Vocabulary = { Greeting, ThankYou };

        // ---- synthetic raw capture geometry (the existing F21 helpers) ----------

        sealed class Geo
        {
            public Step9DInferenceFrameGeometry Build()
            {
                return new Step9DInferenceFrameGeometry(
                    Step9DInferenceGeometryProvenance.TestInjected,
                    Width, Height, 30, Width, Height, 0, false, Width, Height, Width, Height,
                    "RGBA32",
                    Step9DPreInferenceTransformState.None, Step9DPreInferenceTransformState.None,
                    0, Step9DPreInferenceTransformState.None,
                    Step9DPreInferenceTransformState.None,
                    Step9DPreInferenceTransformState.None, 0L, 0L, 0, "TopDown");
            }
        }

        static Step9DMergedInferenceGeometry Standard()
        {
            return new Step9DMergedInferenceGeometry(
                Step9DGeometryAcquisitionResult.ForAvailable(new Geo().Build()),
                Step9DGeometryAcquisitionResult.ForAvailable(new Geo().Build()));
        }

        static double[] NaNArray(int count)
        {
            var values = new double[count * 3];
            for (int index = 0; index < values.Length; index++) values[index] = double.NaN;
            return values;
        }

        static void Put(double[] c, bool[] v, int landmark, double x, double y)
        {
            c[landmark * 3] = x; c[landmark * 3 + 1] = y; c[landmark * 3 + 2] = 0.0;
            v[landmark] = true;
        }

        static RawLandmarkObservationSet PoseSet(int frame)
        {
            var c = NaNArray(Pose);
            var v = new bool[Pose];
            double drift = 0.002 * frame;
            Put(c, v, 11, 0.35 + drift, 0.35);
            Put(c, v, 12, 0.65 + drift, 0.35);
            Put(c, v, 23, 0.40 + drift, 0.68);
            Put(c, v, 24, 0.60 + drift, 0.68);
            Put(c, v, 13, 0.30 + drift, 0.45);
            Put(c, v, 14, 0.70 + drift, 0.45);
            Put(c, v, 15, 0.28 + drift, 0.55);
            Put(c, v, 16, 0.72 + drift, 0.55);
            return new RawLandmarkObservationSet(Pose, c, v, new bool[Pose], new bool[Pose]);
        }

        static RawLandmarkObservationSet HandSet(int frame, double baseX, double baseY)
        {
            var c = NaNArray(Hand);
            var v = new bool[Hand];
            for (int joint = 0; joint < Hand; joint++)
                Put(c, v, joint, baseX + 0.004 * joint + 0.001 * frame,
                                 baseY + 0.003 * joint - 0.0005 * frame);
            return new RawLandmarkObservationSet(Hand, c, v, new bool[Hand], new bool[Hand]);
        }

        static List<RawMediaPipeObservation> RawCapture(int frames)
        {
            var list = new List<RawMediaPipeObservation>();
            for (int frame = 0; frame < frames; frame++)
                list.Add(new RawMediaPipeObservation(
                    0.05 * (frame + 1), RawObservationConvention.LiveMediaPipe,
                    PoseSet(frame), HandSet(frame, 0.30, 0.50),
                    HandSet(frame, 0.60, 0.50), Standard()));
            return list;
        }

        // ---- explicit synthetic F20 material (the existing F20/F21 helpers) ----

        static FullHandFeatureSequence Sequence(double seed)
        {
            int points = FullHandCommonSchema.PointCount;
            int dims = FullHandCommonSchema.CoordinateDimensions;
            var coordinates = new double[SourceFrames * points * dims];
            var mask = new bool[SourceFrames * points];
            for (int frame = 0; frame < SourceFrames; frame++)
                for (int point = 0; point < points; point++)
                {
                    int slot = frame * points + point;
                    mask[slot] = true;
                    coordinates[slot * dims] = seed + point * 0.13 + frame * 0.05;
                    coordinates[slot * dims + 1] = seed * 0.5 + point * 0.07 - frame * 0.03;
                    coordinates[slot * dims + 2] = 0.0;
                }
            return FullHandPairGeometry.BuildFeatures(coordinates, mask, SourceFrames);
        }

        static FullHandModel Model(int width)
        {
            var mean = new double[width];
            var scale = new double[width];
            var coefficients = new double[width];
            for (int index = 0; index < width; index++)
            {
                mean[index] = 0.1 * index;
                scale[index] = 1.0 + 0.01 * index;
                coefficients[index] = ((index % 3) - 1) * 0.25;
            }
            return new FullHandModel(mean, scale, coefficients, 0.2);
        }

        static FullHandInMemoryCommonBank Bank()
        {
            return new FullHandInMemoryCommonBank("model_f27_test_v1", Vocabulary,
                new[] { "common_0", "common_1", "common_2", "common_3" },
                new[] { 0, 0, 1, 1 },
                new[] { Sequence(1.0), Sequence(1.4), Sequence(3.0), Sequence(3.4) },
                Model(FullHandPairConstants.FixedFeatureCount),
                Model(FullHandPairConstants.AlignedFeatureCount));
        }

        /// <summary>Arbitrary permissive test literals. No production cutoff appears here.</summary>
        static FullHandGateConfig AcceptAllGates()
        {
            var gates = new FullHandGate[FullHandMethods.All.Length];
            for (int index = 0; index < gates.Length; index++)
                gates[index] = FullHandGate.AtQualityThreshold(
                    "gate_accept", FullHandMethods.All[index], -1.0e9);
            return new FullHandGateConfig("config_accept_f27", gates);
        }

        static FullHandGateConfig RejectAllGates()
        {
            var gates = new FullHandGate[FullHandMethods.All.Length];
            for (int index = 0; index < gates.Length; index++)
                gates[index] = FullHandGate.RejectEverything(
                    "gate_reject", FullHandMethods.All[index]);
            return new FullHandGateConfig("config_reject_f27", gates);
        }

        static FullHandGameCaptureProcessor Processor(FullHandGateConfig gates)
        {
            var pipeline = new FullHandRecognitionPipeline(Bank(), gates, Participant,
                FullHandParticipantReferenceSet.Empty(Participant));
            return new FullHandGameCaptureProcessor(pipeline, () => Participant);
        }

        /// <summary>
        /// The word this synthetic bank actually ranks first for this synthetic capture,
        /// read from one dedicated probe capture. The test asserts the CONTRACT around the
        /// ranked winner; it never asserts that a particular word is the right answer.
        /// </summary>
        static string ProbeTop1(FullHandGameCaptureProcessor processor)
        {
            FullHandGameCaptureResult probe =
                processor.Process("capture_f27_probe", null, RawCapture(6));
            Assert.IsTrue(probe.Succeeded, probe.FailureCode + " " + probe.FailureDetail);
            string top1 = probe.Outcome.Before.Decision(FullHandMethods.Hybrid).Top1;
            Assert.IsFalse(string.IsNullOrEmpty(top1), "the probe capture must rank a class");
            return top1;
        }

        static string Other(string word)
        {
            return string.Equals(word, Greeting, StringComparison.Ordinal) ? ThankYou : Greeting;
        }

        static FullHandGameplayDecision Decide(FullHandGameCaptureResult evidence,
                                               string expectedWord, string captureId)
        {
            return FullHandGameplayEvaluator.Decide(evidence, expectedWord, captureId, true);
        }

        // ---- the gameplay decision ---------------------------------------------

        [Test]
        public void OnlyAnAcceptedMatchingTargetProducesGameplaySuccess()
        {
            FullHandGameCaptureProcessor processor = Processor(AcceptAllGates());
            string target = ProbeTop1(processor);

            const string captureId = "capture_f27_accept";
            FullHandGameCaptureResult evidence =
                processor.Process(captureId, target, RawCapture(6));
            Assert.IsTrue(evidence.Succeeded, evidence.FailureCode + " " + evidence.FailureDetail);

            FullHandGameplayDecision decision = Decide(evidence, target, captureId);
            Assert.AreEqual(FullHandGameplayDecisionKind.Accepted, decision.Kind);
            Assert.IsTrue(decision.GrantsQuestSuccess);

            // the authority is the hybrid method over the COMMON-ONLY bank
            Assert.AreEqual(FullHandMethods.Hybrid, decision.Method);
            Assert.AreEqual(FullHandPipelineBanks.CommonOnly, decision.BankId);
            Assert.AreEqual(FullHandGameplayDecision.SourceId, decision.Source);
            Assert.AreEqual(evidence.Outcome.CommonModelId, decision.ModelId);
            Assert.AreEqual(evidence.Outcome.GateConfigId, decision.GateConfigId);
            Assert.AreEqual(captureId, decision.CaptureSessionId);
            Assert.AreEqual(target, decision.Top1);

            // an accepted capture is a model/gate decision, not certified sign correctness:
            // no probability of any kind is produced
            Assert.AreEqual(GuidedQuestOutcomeKind.StrictSuccess,
                            FullHandGameplayAttemptFactory.MapKind(decision.Kind));
        }

        [Test]
        public void AnAcceptedButDifferentWordNeverGrantsProgress()
        {
            FullHandGameCaptureProcessor processor = Processor(AcceptAllGates());
            string wrongTarget = Other(ProbeTop1(processor));

            const string captureId = "capture_f27_wrongword";
            FullHandGameCaptureResult evidence =
                processor.Process(captureId, wrongTarget, RawCapture(6));
            FullHandGameplayDecision decision = Decide(evidence, wrongTarget, captureId);

            Assert.AreEqual(FullHandGameplayDecisionKind.AcceptedDifferentWord, decision.Kind);
            Assert.IsFalse(decision.GrantsQuestSuccess);
            Assert.AreEqual(GuidedQuestOutcomeKind.WrongWord,
                            FullHandGameplayAttemptFactory.MapKind(decision.Kind));
            Assert.AreNotEqual(wrongTarget, decision.Top1,
                "the declared target must never be treated as the truth");
        }

        [Test]
        public void AnAbstentionIsARetryOutcomeAndNeverTheItemGrantingPath()
        {
            FullHandGameCaptureProcessor processor = Processor(RejectAllGates());
            const string captureId = "capture_f27_abstain";
            FullHandGameCaptureResult evidence =
                processor.Process(captureId, Greeting, RawCapture(6));
            Assert.IsTrue(evidence.Succeeded, "an abstention is evidence, not a bridge failure");

            FullHandGameplayDecision decision = Decide(evidence, Greeting, captureId);
            Assert.AreEqual(FullHandGameplayDecisionKind.NotDistinguished, decision.Kind);
            Assert.IsFalse(decision.GrantsQuestSuccess);

            GuidedQuestOutcomeKind kind = FullHandGameplayAttemptFactory.MapKind(decision.Kind);
            Assert.AreEqual(GuidedQuestOutcomeKind.NotDistinguishedRetry, kind);
            Assert.AreNotEqual(GuidedQuestOutcomeKind.LowConfidenceCorrect, kind,
                "the abstention must not reach the enum value that already grants the NPC item");
            Assert.AreEqual(TrustedHybridOutcomeKind.NotDistinguishedRetry,
                PetRobotSignPracticeSession.MapGuidedToCanonicalWorldOutcome(kind));
            Assert.AreEqual(FullHandGameplayFeedback.NotDistinguished,
                            FullHandGameplayFeedback.Korean(decision));
        }

        [Test]
        public void AnUnsupportedPromptedWordRefusesWithoutAnyFallback()
        {
            FullHandGameCaptureProcessor processor = Processor(AcceptAllGates());
            const string captureId = "capture_f27_unsupported";
            // LOVE is genuinely outside this vocabulary, exactly as it is outside the real one
            FullHandGameCaptureResult evidence =
                processor.Process(captureId, Love, RawCapture(6));
            Assert.IsTrue(evidence.Succeeded);
            Assert.IsFalse(evidence.Outcome.TargetSupported);

            FullHandGameplayDecision decision = Decide(evidence, Love, captureId);
            Assert.AreEqual(FullHandGameplayDecisionKind.UnsupportedWord, decision.Kind);
            Assert.IsFalse(decision.GrantsQuestSuccess);
            Assert.AreEqual(GuidedQuestOutcomeKind.UnsupportedWord,
                            FullHandGameplayAttemptFactory.MapKind(decision.Kind));
            Assert.AreEqual(FullHandGameplayFeedback.Unsupported,
                            FullHandGameplayFeedback.Korean(decision));
        }

        [Test]
        public void AMissingOrRefusedConfigurationIsAnExplicitTechnicalRefusal()
        {
            FullHandGameplayDecision unconfigured =
                FullHandGameplayEvaluator.Decide(null, Greeting, "capture_f27_none", false);
            Assert.AreEqual(FullHandGameplayDecisionKind.TechnicalRefusal, unconfigured.Kind);
            Assert.AreEqual(FullHandGameplayCodes.PipelineNotConfigured, unconfigured.FailureCode);
            Assert.IsFalse(unconfigured.GrantsQuestSuccess);

            FullHandGameplayDecision missing =
                FullHandGameplayEvaluator.Decide(null, Greeting, "capture_f27_missing", true);
            Assert.AreEqual(FullHandGameplayCodes.EvidenceUnavailable, missing.FailureCode);

            var refusing = new FullHandGameCaptureRefusingProcessor(
                FullHandBridgeCodes.ConfigurationGateInvalid, "synthetic invalid configuration");
            const string captureId = "capture_f27_refused";
            FullHandGameplayDecision refused = Decide(
                refusing.Process(captureId, Greeting, RawCapture(4)), Greeting, captureId);
            Assert.AreEqual(FullHandGameplayDecisionKind.TechnicalRefusal, refused.Kind);
            Assert.AreEqual(FullHandBridgeCodes.ConfigurationGateInvalid, refused.FailureCode);
            Assert.AreEqual(string.Empty, refused.Top1, "a refusal invents no winner");
            Assert.IsTrue(double.IsNaN(refused.Score), "a refusal invents no score");
        }

        [Test]
        public void EvidenceFromAnotherCaptureOrAnotherTargetCannotScoreThisAttempt()
        {
            FullHandGameCaptureProcessor processor = Processor(AcceptAllGates());
            string target = ProbeTop1(processor);
            FullHandGameCaptureResult evidence =
                processor.Process("capture_f27_other", target, RawCapture(6));

            FullHandGameplayDecision mismatched =
                Decide(evidence, target, "capture_f27_this_one");
            Assert.AreEqual(FullHandGameplayCodes.CaptureIdentityMismatch, mismatched.FailureCode);
            Assert.IsFalse(mismatched.GrantsQuestSuccess);

            FullHandGameplayDecision otherTarget =
                Decide(evidence, Other(target), "capture_f27_other");
            Assert.AreEqual(FullHandGameplayCodes.TargetMismatch, otherTarget.FailureCode);

            FullHandGameplayDecision noIdentity = Decide(evidence, target, string.Empty);
            Assert.AreEqual(FullHandGameplayCodes.CaptureIdentityMismatch, noIdentity.FailureCode);

            FullHandGameplayDecision empty = Decide(
                processor.Process("capture_f27_empty", target,
                    new List<RawMediaPipeObservation>()), target, "capture_f27_empty");
            Assert.AreEqual(FullHandGameplayDecisionKind.TechnicalRefusal, empty.Kind);
            Assert.AreEqual(FullHandBridgeCodes.EmptyCapture, empty.FailureCode);
        }

        [Test]
        public void ASourceFailureRefusesEvenWhenEvidenceForThatCaptureExists()
        {
            FullHandGameplayDecision refusal = FullHandGameplayEvaluator.RefuseForSourceFailure(
                "capture_f27_source", Greeting, "InvalidOperationException");
            Assert.AreEqual(FullHandGameplayDecisionKind.TechnicalRefusal, refusal.Kind);
            Assert.AreEqual(FullHandGameplayCodes.CaptureSourceFailure, refusal.FailureCode);
            Assert.IsFalse(refusal.GrantsQuestSuccess);
            Assert.AreEqual(GuidedQuestOutcomeKind.TechnicalFailure,
                            FullHandGameplayAttemptFactory.MapKind(refusal.Kind));
        }

        // ---- the real session consumer: CONTRACT ONLY ---------------------------
        //
        // These construct the real PetRobotSignPracticeSession directly, exactly as the
        // quest and practice routes do, and verify what the shared consumer does with a
        // full-hand attempt. They deliberately BYPASS actual in-world NPC entry, which
        // still requires the existing legacy 3/3/3 enrollment gate and the legacy quest
        // hybrid session. They are consumer-contract tests and are NOT an end-to-end NPC
        // or fresh-participant demo proof.

        sealed class FakeAdapter : IPersonalLivePracticeRecognitionAdapter
        {
            public PracticeRecognitionAttempt Next;
            public string Expected = Greeting;

            public PersonalLiveCaptureState CaptureState { get { return PersonalLiveCaptureState.Ready; } }
            public string CaptureSessionId { get { return Next == null ? string.Empty : Next.CaptureSessionId; } }
            public string ActiveProfileId { get { return "f27_fake_profile"; } }
            public bool CandidateRouteActive { get { return true; } }
            public bool SetExpectedWord(string word) { Expected = word; return true; }
            public bool BeginCapture() { return true; }
            public PracticeRecognitionAttempt FinalizeCapture() { return Next; }
            public bool ResetCapture() { return true; }
        }

        sealed class RecordingRepository : IPersonalizationCandidateRepository
        {
            public int SaveCalls;
            public string RootDirectory { get { return string.Empty; } }
            public string LearnerId { get { return Participant; } }
            public CandidateCountSnapshot Counts { get { return CandidateCountSnapshot.Empty; } }
            public CandidateCountSnapshot ReloadAndVerify() { return CandidateCountSnapshot.Empty; }

            public CandidateSaveResult Save(PersonalizationCandidateSaveRequest request)
            {
                SaveCalls++;
                throw new InvalidOperationException(
                    "the full-hand route must never reach the 16-point candidate store");
            }
        }

        /// <summary>
        /// The complete two-member legacy frozen-hybrid seam. It returns null on purpose:
        /// the full-hand quest branch must finish coherently without it and must not call
        /// it at all.
        /// </summary>
        sealed class NullHybridSession : IQuestHybridRecognitionSession
        {
            public int Calls;
            public long Generation { get { return 1L; } }

            public TrustedHybridRecognitionResult Recognize(
                string captureSessionId, RecognitionSequence sequence, string expectedWord)
            {
                Calls++;
                return null;
            }
        }

        static PracticeRecognitionAttempt FullHandAttempt(FullHandGameplayDecision decision,
                                                          string captureId)
        {
            // the legacy 16-point result is deliberately null: an enabled full-hand capture
            // must finish on its own evidence
            return FullHandGameplayAttemptFactory.BuildAttempt(
                decision, captureId, PersonalLiveCaptureState.Failed, null,
                "f27_fake_profile", null, null, null);
        }

        static PracticeRecognitionAttempt AcceptedAttempt(out string target, string captureId)
        {
            FullHandGameCaptureProcessor processor = Processor(AcceptAllGates());
            target = ProbeTop1(processor);
            FullHandGameCaptureResult evidence =
                processor.Process(captureId, target, RawCapture(6));
            return FullHandGameplayAttemptFactory.BuildAttempt(
                Decide(evidence, target, captureId), captureId,
                PersonalLiveCaptureState.Failed, null, "f27_fake_profile", null, null, evidence);
        }

        [Test]
        public void QuestConsumerContract_CompletesOnTheFullHandDecisionWithNoLegacySequence()
        {
            string target;
            PracticeRecognitionAttempt attempt =
                AcceptedAttempt(out target, "capture_f27_quest");
            // the prompted word of this attempt is whatever the synthetic bank ranked first;
            // the assertion below is about the CONTRACT, never about which word is correct
            Assert.AreEqual(target, attempt.ExpectedWord);
            Assert.AreEqual(GuidedQuestOutcomeKind.StrictSuccess, attempt.Outcome.Kind);

            var adapter = new FakeAdapter { Next = attempt };
            var repository = new RecordingRepository();
            var hybrid = new NullHybridSession();
            using (var session = new PetRobotSignPracticeSession(
                adapter, repository, hybrid, new PetRobotGreetingQuestToken()))
            {
                Assert.IsTrue(session.StartExpression());
                Assert.IsTrue(session.FinishExpression());

                Assert.AreEqual(TrustedHybridOutcomeKind.StrictSuccess,
                                session.QuestAuthoritativeOutcome);
                Assert.IsTrue(session.QuestCompleted);
                Assert.AreEqual(0, hybrid.Calls,
                    "the legacy hybrid calculation must not run on the full-hand route");
                Assert.IsNull(session.LegacyHybridDiagnosticOutcome);
                Assert.AreEqual(0, repository.SaveCalls,
                    "a quest attempt stores no personalization candidate");
                Assert.IsNotNull(session.LastFullHandGameplayDecision);
                Assert.AreEqual(FullHandPipelineBanks.CommonOnly,
                                session.LastFullHandGameplayDecision.BankId);
                Assert.IsNull(session.LastAttempt.NormalizedSequence,
                    "a 46-point attempt carries no 16-point normalized sequence");
            }
        }

        [Test]
        public void QuestConsumerContract_AnAbstentionGrantsNoProgressEvenWithNoLegacyResult()
        {
            FullHandGameCaptureProcessor processor = Processor(RejectAllGates());
            const string captureId = "capture_f27_quest_abstain";
            FullHandGameplayDecision decision = Decide(
                processor.Process(captureId, Greeting, RawCapture(6)), Greeting, captureId);

            var adapter = new FakeAdapter { Next = FullHandAttempt(decision, captureId) };
            var repository = new RecordingRepository();
            var hybrid = new NullHybridSession();
            using (var session = new PetRobotSignPracticeSession(
                adapter, repository, hybrid, new PetRobotGreetingQuestToken()))
            {
                Assert.IsTrue(session.StartExpression());
                Assert.IsTrue(session.FinishExpression());

                Assert.AreEqual(TrustedHybridOutcomeKind.NotDistinguishedRetry,
                                session.QuestAuthoritativeOutcome);
                Assert.IsFalse(session.QuestCompleted, "an abstention never completes the quest");
                Assert.IsTrue(session.CanRetry);
                Assert.AreEqual(0, hybrid.Calls);
                Assert.AreEqual(0, repository.SaveCalls);
            }
        }

        [Test]
        public void ThePracticeRouteSharesTheOutcomeAndWritesNoLegacyCandidate()
        {
            string target;
            PracticeRecognitionAttempt attempt =
                AcceptedAttempt(out target, "capture_f27_practice");

            var adapter = new FakeAdapter { Next = attempt, Expected = target };
            var repository = new RecordingRepository();
            using (var session = new PetRobotSignPracticeSession(adapter, repository))
            {
                Assert.IsTrue(session.StartExpression());
                Assert.IsTrue(session.FinishExpression());

                // the SAME decision the quest route would consume, through the same seam
                Assert.AreEqual(GuidedQuestOutcomeKind.StrictSuccess,
                                session.LastLearningOutcome.Kind);
                Assert.AreEqual(0, repository.SaveCalls,
                    "no automatic 16-point candidate may be saved on the full-hand route");
                Assert.AreEqual(CandidateStorageStatus.NotRequested,
                                session.LastStorageResult.Status);
                Assert.AreEqual(FullHandGameplayFeedback.EnrollmentPending,
                                session.LastStorageResult.Message);
                Assert.AreNotEqual(PetRobotPersonalizationEnrollmentState.Personalized,
                                   session.CurrentEnrollmentState,
                    "full-hand enrollment is not implemented and must not be reported complete");
            }
        }

        [Test]
        public void AnUnsupportedWordInThePracticeRouteStopsWithoutFallback()
        {
            FullHandGameCaptureProcessor processor = Processor(AcceptAllGates());
            const string captureId = "capture_f27_practice_love";
            FullHandGameplayDecision decision = Decide(
                processor.Process(captureId, Love, RawCapture(6)), Love, captureId);
            Assert.AreEqual(FullHandGameplayDecisionKind.UnsupportedWord, decision.Kind);

            var adapter = new FakeAdapter { Next = FullHandAttempt(decision, captureId) };
            var repository = new RecordingRepository();
            using (var session = new PetRobotSignPracticeSession(adapter, repository))
            {
                Assert.IsTrue(session.StartExpression());
                Assert.IsTrue(session.FinishExpression());

                Assert.AreEqual(GuidedQuestOutcomeKind.UnsupportedWord,
                                session.LastLearningOutcome.Kind);
                Assert.AreEqual(PetRobotWordLearningStatus.Pending,
                                session.GetWordLearningStatus(session.CurrentWord));
                Assert.IsTrue(session.CanContinueWithNeedsPractice,
                    "the learner can move on truthfully, never as mastered");
                Assert.AreEqual(0, repository.SaveCalls);
            }
        }

        [Test]
        public void TheOffModePathIsUnchanged()
        {
            // an ordinary legacy attempt: FullHandGameplay is null and the existing
            // LowConfidenceCorrect quest behaviour is preserved exactly
            GuidedQuestOutcome outcome = GuidedQuestOutcomeEvaluator.EvaluateForRoute(
                PersonalLiveProfileMode.MultiPrototypeCandidate,
                PersonalLiveCaptureState.Completed, true, Greeting, Greeting,
                true, true, false, false, false);
            Assert.AreEqual(GuidedQuestOutcomeKind.LowConfidenceCorrect, outcome.Kind);

            // A GENUINE 16-point sequence, produced by the approved preprocessing and
            // sequence-input path inside the real controller running its ORDINARY off-mode
            // route over a synthetic raw capture. The unchanged legacy quest handler
            // correctly refuses a null sequence, so this test must supply a real one rather
            // than weaken that guard.
            var legacyFactory = new CountingRuntimeFactory();
            PersonalLiveRecognitionController legacyController =
                LiveController(Processor(AcceptAllGates()), legacyFactory, false);
            Assert.IsTrue(RunLiveCapture(legacyController),
                "the off-mode capture must complete through the legacy route");
            Assert.AreEqual(PersonalLiveCaptureState.Completed, legacyController.State);
            Assert.IsNotNull(legacyController.Result);
            RecognitionSequence legacySequence = legacyController.Result.NormalizedSequence;
            Assert.IsNotNull(legacySequence,
                "the legacy route must produce a real normalized sequence");

            var legacy = new PracticeRecognitionAttempt(
                "capture_f27_legacy", Greeting, Greeting, PersonalLiveCaptureState.Completed,
                outcome, legacyController.Result, legacySequence,
                PersonalLiveProfileMode.MultiPrototypeCandidate.ToString(), "profile",
                0.5d, null, null,
                new PracticeRecognitionMetrics { WinningPrototypeId = string.Empty });
            Assert.IsNull(legacy.FullHandGameplay, "the default path carries no full-hand authority");
            Assert.IsFalse(legacy.FullHandGameplayAuthority);

            var adapter = new FakeAdapter { Next = legacy };
            var repository = new RecordingRepository();
            var hybrid = new NullHybridSession();
            using (var session = new PetRobotSignPracticeSession(
                adapter, repository, hybrid, new PetRobotGreetingQuestToken()))
            {
                Assert.IsTrue(session.StartExpression());
                Assert.IsTrue(session.FinishExpression());
                Assert.AreEqual(TrustedHybridOutcomeKind.LowConfidenceCorrect,
                                session.QuestAuthoritativeOutcome);
                Assert.IsTrue(session.QuestCompleted, "the existing legacy behaviour is unchanged");
            }
        }

        // ---- the REAL learner story flow (what the main UGUI renders) -----------

        /// <summary>Drives the real story flow to the Feedback beat and returns its copy.</summary>
        static MoaStoryPresentation StoryFeedback(MoaStoryFeedbackKind kind,
                                                  bool canContinueWithNeedsPractice,
                                                  PetRobotPersonalizationEnrollmentState enrollment)
        {
            var flow = new MoaStoryGuidedPracticeFlow();
            flow.Reset(Greeting);
            Assert.IsTrue(flow.AdvanceStory());
            Assert.IsTrue(flow.AdvanceStory());
            Assert.IsTrue(flow.AdvanceStory());
            flow.Synchronize(Greeting, PetRobotPracticeState.Ready, MoaStoryFeedbackKind.None,
                false, true, RawMediaPipeNeutralAutoSegmenterState.Inactive, true,
                true, PetRobotPersonalizationEnrollmentState.Pending, 0, 0, 0, false, true);
            Assert.IsTrue(flow.ConfirmAutoAttemptAccepted());
            flow.Synchronize(Greeting, PetRobotPracticeState.Feedback, kind,
                canContinueWithNeedsPractice, true,
                RawMediaPipeNeutralAutoSegmenterState.Inactive, true,
                true, enrollment, 0, 1, 0, false, true);
            Assert.AreEqual(MoaStoryBeat.Feedback, flow.Beat);
            return flow.Current;
        }

        [Test]
        public void TheLearnerScreenShowsDistinctHonestCopyForEachFullHandVerdict()
        {
            MoaStoryPresentation abstention = StoryFeedback(
                MoaStoryFeedbackKind.NotDistinguishedRetry, false,
                PetRobotPersonalizationEnrollmentState.Pending);
            Assert.AreEqual(MoaStoryGuidedPracticeFlow.NotDistinguishedCopy, abstention.BodyText);
            Assert.AreEqual(MoaStoryAction.RetryNotDistinguished, abstention.Action);
            Assert.AreNotEqual(MoaStoryGuidedPracticeFlow.WrongWordCopy, abstention.BodyText);
            Assert.AreNotEqual(MoaStoryGuidedPracticeFlow.TechnicalFailureCopy, abstention.BodyText);

            MoaStoryPresentation unsupported = StoryFeedback(
                MoaStoryFeedbackKind.UnsupportedWord, true,
                PetRobotPersonalizationEnrollmentState.PersonalizationIncomplete);
            Assert.AreEqual(MoaStoryGuidedPracticeFlow.UnsupportedWordCopy, unsupported.BodyText);
            Assert.AreEqual(MoaStoryAction.ContinueWithNeedsPractice, unsupported.Action);

            MoaStoryPresentation success = StoryFeedback(
                MoaStoryFeedbackKind.StrictSuccess, false,
                PetRobotPersonalizationEnrollmentState.PersonalizationIncomplete);
            Assert.AreEqual(MoaStoryAction.ContinueToNextWord, success.Action);
            StringAssert.Contains(MoaStoryGuidedPracticeFlow.FullHandSuccessCopy, success.BodyText);
            StringAssert.Contains(MoaStoryGuidedPracticeFlow.FullHandEnrollmentPendingCopy,
                                  success.BodyText);
            // no saved-sample or completed-personalization claim, and no technical identity
            foreach (string forbidden in new[] { "표본", "개인화 등록", "GREETING", "hybrid",
                                                 "common_only", "wis_full_hand" })
                StringAssert.DoesNotContain(forbidden, success.BodyText);
        }

        // ---- the REAL controller seam -------------------------------------------

        sealed class FakeObservationSource : IPersonalLiveObservationSource
        {
            Action<RawMediaPipeObservation> observations;
            Action<Exception> failures;

            public bool IsBound { get; set; } = true;

            public event Action<RawMediaPipeObservation> ObservationArrived
            {
                add { observations += value; }
                remove { observations -= value; }
            }

            public event Action<Exception> SourceFailed
            {
                add { failures += value; }
                remove { failures -= value; }
            }
        }

        sealed class FakeGlobalRuntime : IPersonalLiveGlobalRuntime { }

        /// <summary>Counts legacy recognitions so "the old route did not run" is observable.</summary>
        sealed class CountingRuntime : IPersonalLiveRecognitionRuntime
        {
            public int RecognizeCalls;
            public bool ThrowOnRecognize;

            public bool IsPersonalActive { get { return true; } }
            public string ActiveProfileId { get { return "f27_fake_profile"; } }

            public PersonalLiveRoutingResult Recognize(string sessionId,
                RecognitionSequence sequence, string expectedWord)
            {
                RecognizeCalls++;
                if (ThrowOnRecognize)
                    throw new InvalidOperationException("injected legacy recognition failure");
                var scores = new Dictionary<string, double>(StringComparer.Ordinal)
                {
                    [Greeting] = 0.12,
                    [Love] = 0.34,
                    [ThankYou] = 0.56,
                };
                var accepts = new Dictionary<string, bool>(StringComparer.Ordinal)
                {
                    [Greeting] = true,
                    [Love] = false,
                    [ThankYou] = false,
                };
                return new PersonalLiveRoutingResult(
                    Greeting, true, true, true, scores, accepts, scores[Greeting],
                    "prototype-GREETING", Array.Empty<string>());
            }
        }

        sealed class CountingRuntimeFactory : IPersonalLiveRuntimeFactory
        {
            public CountingRuntime Runtime = new CountingRuntime();

            public IPersonalLiveGlobalRuntime LoadGlobal(PersonalLiveRuntimePaths paths)
            {
                return new FakeGlobalRuntime();
            }

            public IPersonalLiveRecognitionRuntime CreateAndActivate(
                IPersonalLiveGlobalRuntime globalRuntime, PersonalLiveRuntimePaths paths)
            {
                return Runtime;
            }
        }

        static PersonalLiveRecognitionController LiveController(
            IFullHandGameCaptureProcessor processor, CountingRuntimeFactory factory,
            bool gameplayEnabled)
        {
            var controller = new PersonalLiveRecognitionController(
                new FakeObservationSource(), factory, null, false, false, false, processor,
                gameplayEnabled);
            Assert.IsTrue(controller.Initialize(
                new PersonalLiveRuntimePaths("templates", "thresholds.json", null, true)),
                "the fake runtime must initialize");
            return controller;
        }

        const string LiveCaptureId = "capture_f27_live";

        static bool RunLiveCapture(PersonalLiveRecognitionController controller)
        {
            Assert.IsTrue(controller.SetExpectedWord(Greeting));
            Assert.IsTrue(controller.BeginCaptureWithPreRoll(RawCapture(8), LiveCaptureId));
            return controller.FinalizeCaptureWithTrailingTrim(0.06d);
        }

        [Test]
        public void GameplayModeFinalizesOnFullHandEvidenceAndSkipsTheLegacyRoute()
        {
            var factory = new CountingRuntimeFactory();
            PersonalLiveRecognitionController controller =
                LiveController(Processor(AcceptAllGates()), factory, true);

            Assert.IsTrue(RunLiveCapture(controller), "the capture must finalize coherently");
            Assert.AreEqual(PersonalLiveCaptureState.Completed, controller.State);
            Assert.AreEqual(0, factory.Runtime.RecognizeCalls,
                "the legacy 16-point recognizer must not run in gameplay mode");
            Assert.IsNull(controller.Result, "no fake legacy result is produced");
            Assert.IsNull(controller.Failure);
            Assert.IsTrue(controller.FullHandGameplayCaptureFinalized);

            FullHandGameCaptureResult evidence = controller.LastFullHandResult;
            Assert.IsNotNull(evidence, "the one SafeProcess result is the authority input");
            Assert.IsTrue(evidence.Succeeded, evidence.FailureCode + " " + evidence.FailureDetail);
            Assert.AreEqual(LiveCaptureId, evidence.CaptureSessionId);

            FullHandGameplayDecision decision = FullHandGameplayEvaluator.DecideForCapture(
                evidence, Greeting, LiveCaptureId,
                controller.HasFullHandGameCaptureProcessor,
                controller.FullHandGameplayCaptureFinalized);
            Assert.AreNotEqual(FullHandGameplayDecisionKind.TechnicalRefusal, decision.Kind,
                "a completed full-hand capture yields a real verdict, not a refusal");
        }

        [Test]
        public void OffModeStillRunsTheExistingLegacyRoute()
        {
            var factory = new CountingRuntimeFactory();
            PersonalLiveRecognitionController controller =
                LiveController(Processor(AcceptAllGates()), factory, false);

            Assert.IsTrue(RunLiveCapture(controller));
            Assert.AreEqual(PersonalLiveCaptureState.Completed, controller.State);
            Assert.AreEqual(1, factory.Runtime.RecognizeCalls,
                "the default path is unchanged and still recognizes");
            Assert.IsNotNull(controller.Result);
            Assert.IsFalse(controller.FullHandGameplayCaptureFinalized);
        }

        [Test]
        public void AFailingLegacyRuntimeCannotVetoAnEnabledFullHandCapture()
        {
            var factory = new CountingRuntimeFactory();
            factory.Runtime.ThrowOnRecognize = true;
            PersonalLiveRecognitionController controller =
                LiveController(Processor(AcceptAllGates()), factory, true);

            Assert.IsTrue(RunLiveCapture(controller));
            Assert.AreEqual(PersonalLiveCaptureState.Completed, controller.State);
            Assert.AreEqual(0, factory.Runtime.RecognizeCalls);
            Assert.IsNotNull(controller.LastFullHandResult);
        }

        [Test]
        public void GameplayModeWithoutAProcessorRefusesInsteadOfFallingBack()
        {
            var factory = new CountingRuntimeFactory();
            PersonalLiveRecognitionController controller = LiveController(null, factory, true);

            Assert.IsTrue(RunLiveCapture(controller));
            Assert.IsNull(controller.LastFullHandResult);
            Assert.AreEqual(0, factory.Runtime.RecognizeCalls,
                "an unconfigured gameplay route must not silently use the legacy authority");

            FullHandGameplayDecision decision = FullHandGameplayEvaluator.DecideForCapture(
                controller.LastFullHandResult, Greeting, LiveCaptureId,
                controller.HasFullHandGameCaptureProcessor,
                controller.FullHandGameplayCaptureFinalized);
            Assert.AreEqual(FullHandGameplayDecisionKind.TechnicalRefusal, decision.Kind);
            Assert.AreEqual(FullHandGameplayCodes.PipelineNotConfigured, decision.FailureCode);
        }

        [Test]
        public void CancelledEmptyOrRefinalizedCapturesCarryNoGameplaySuccess()
        {
            var factory = new CountingRuntimeFactory();
            PersonalLiveRecognitionController controller =
                LiveController(Processor(AcceptAllGates()), factory, true);

            // a cancelled capture keeps no evidence and no completion
            Assert.IsTrue(controller.SetExpectedWord(Greeting));
            Assert.IsTrue(controller.BeginCaptureWithPreRoll(RawCapture(6), "capture_f27_cancel"));
            Assert.IsTrue(controller.CancelCapture());
            Assert.IsNull(controller.LastFullHandResult);
            Assert.IsFalse(controller.FullHandGameplayCaptureFinalized);

            // a real success, then a second, invalid finalize: the cached result must not be
            // consumed again as a new success
            Assert.IsTrue(controller.Reset());
            Assert.IsTrue(RunLiveCapture(controller));
            Assert.IsTrue(controller.FullHandGameplayCaptureFinalized);
            FullHandGameCaptureResult cached = controller.LastFullHandResult;
            Assert.IsNotNull(cached);

            Assert.IsFalse(controller.FinalizeCaptureWithTrailingTrim(0.0d),
                "finalizing again is invalid");
            Assert.IsFalse(controller.FullHandGameplayCaptureFinalized);
            FullHandGameplayDecision stale = FullHandGameplayEvaluator.DecideForCapture(
                controller.LastFullHandResult, Greeting, LiveCaptureId,
                controller.HasFullHandGameCaptureProcessor,
                controller.FullHandGameplayCaptureFinalized);
            Assert.AreEqual(FullHandGameplayDecisionKind.TechnicalRefusal, stale.Kind);
            Assert.AreEqual(FullHandGameplayCodes.CaptureNotFinalized, stale.FailureCode);
        }

        [Test]
        public void ASourceFailureInGameplayModeCannotBecomeASuccess()
        {
            var factory = new CountingRuntimeFactory();
            PersonalLiveRecognitionController controller =
                LiveController(Processor(AcceptAllGates()), factory, true);

            Assert.IsTrue(controller.SetExpectedWord(Greeting));
            Assert.IsTrue(controller.BeginCaptureWithPreRoll(RawCapture(6), "capture_f27_src"));
            Assert.IsTrue(controller.FailActiveCaptureFromSource(
                new InvalidOperationException("injected source failure")));
            Assert.AreEqual(PersonalLiveCaptureState.Failed, controller.State);
            Assert.IsFalse(controller.FullHandGameplayCaptureFinalized);

            FullHandGameplayDecision decision = FullHandGameplayEvaluator.DecideForCapture(
                controller.LastFullHandResult, Greeting, "capture_f27_src",
                controller.HasFullHandGameCaptureProcessor,
                controller.FullHandGameplayCaptureFinalized);
            Assert.AreEqual(FullHandGameplayDecisionKind.TechnicalRefusal, decision.Kind);
            Assert.IsFalse(decision.GrantsQuestSuccess);
        }

        // ---- the existing full-hand geometry actually uses the interior joints ---

        /// <summary>
        /// Holds the wrist and all five fingertips of both hands and the whole body block
        /// FIXED, moves only interior finger joints, and shows that the EXISTING F4/F20
        /// feature build and pair features change. This demonstrates that the 46-point route
        /// consumes the full finger articulation; it asserts nothing about score quality and
        /// nothing about human recognition accuracy.
        /// </summary>
        [Test]
        public void MovingOnlyInteriorFingerJointsChangesTheExistingFullHandFeatures()
        {
            int points = FullHandCommonSchema.PointCount;
            int dims = FullHandCommonSchema.CoordinateDimensions;
            var tips = new HashSet<int> { 0, 4, 8, 12, 16, 20 };  // wrist + five fingertips

            double[] baseline; bool[] mask;
            BuildSource(out baseline, out mask);
            var moved = (double[])baseline.Clone();

            int changed = 0;
            for (int frame = 0; frame < SourceFrames; frame++)
                for (int joint = 0; joint < FullHandPairConstants.HandJointCount; joint++)
                {
                    if (tips.Contains(joint)) continue;
                    int point = FullHandPairConstants.LeftSourceBase + joint;
                    int slot = (frame * points + point) * dims;
                    moved[slot] += 0.020;
                    moved[slot + 1] -= 0.015;
                    changed++;
                }
            Assert.Greater(changed, 0, "the test must actually move interior joints");

            // the held points are byte-identical: wrist, fingertips, the four arm points and
            // the whole right hand
            for (int frame = 0; frame < SourceFrames; frame++)
                for (int point = 0; point < points; point++)
                {
                    int joint = point - FullHandPairConstants.LeftSourceBase;
                    bool interiorLeft = joint >= 0 &&
                        joint < FullHandPairConstants.HandJointCount && !tips.Contains(joint);
                    if (interiorLeft) continue;
                    int slot = (frame * points + point) * dims;
                    for (int axis = 0; axis < dims; axis++)
                        Assert.AreEqual(baseline[slot + axis], moved[slot + axis],
                            "point " + point + " axis " + axis + " must be untouched");
                }

            FullHandFeatureSequence a = FullHandPairGeometry.BuildFeatures(baseline, mask, SourceFrames);
            FullHandFeatureSequence b = FullHandPairGeometry.BuildFeatures(moved, mask, SourceFrames);
            Assert.IsTrue(a.Usable, a.Failure);
            Assert.IsTrue(b.Usable, b.Failure);
            Assert.AreNotEqual(FullHandFeatureDigest.Compute(a), FullHandFeatureDigest.Compute(b),
                "interior finger joints must reach the existing 86-point feature sequence");

            FullHandFeatureSequence reference = Sequence(1.0);
            FullHandPairResult pairA = FullHandPairGeometry.AlignedPairFeatures(a, reference);
            FullHandPairResult pairB = FullHandPairGeometry.AlignedPairFeatures(b, reference);
            Assert.IsTrue(pairA.Ok, pairA.Failure);
            Assert.IsTrue(pairB.Ok, pairB.Failure);

            double[] featuresA = pairA.CopyFeatures();
            double[] featuresB = pairB.CopyFeatures();
            Assert.AreEqual(featuresA.Length, featuresB.Length);
            bool anyDifferent = false;
            for (int index = 0; index < featuresA.Length; index++)
                if (featuresA[index] != featuresB[index]) { anyDifferent = true; break; }
            Assert.IsTrue(anyDifferent,
                "the existing pair geometry must respond to interior finger articulation");
        }

        /// <summary>A fully valid 46-point source with two distinguishable hands.</summary>
        static void BuildSource(out double[] coordinates, out bool[] mask)
        {
            int points = FullHandCommonSchema.PointCount;
            int dims = FullHandCommonSchema.CoordinateDimensions;
            coordinates = new double[SourceFrames * points * dims];
            mask = new bool[SourceFrames * points];
            for (int frame = 0; frame < SourceFrames; frame++)
                for (int point = 0; point < points; point++)
                {
                    int slot = frame * points + point;
                    mask[slot] = true;
                    coordinates[slot * dims] = 0.30 + point * 0.011 + frame * 0.004;
                    coordinates[slot * dims + 1] = 0.40 + point * 0.009 - frame * 0.003;
                    coordinates[slot * dims + 2] = 0.0;
                }
        }
    }
}
