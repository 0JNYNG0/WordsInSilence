using System;
using System.Collections.Generic;
using NUnit.Framework;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class PetRobotRetryExhaustionNeedsPracticeCompletionV1Tests
    {
        [Test]
        public void FirstWrongWord_AllowsExactlyRetry1AndNotNeedsPracticeContinue()
        {
            PetRobotSignPracticeSession session = NewSession(out FakeRecognitionAdapter adapter, out FakeRepository repository);
            Finish(session, adapter, Attempt("g-wrong-1", "GREETING", "LOVE", false, false, false));

            Assert.IsTrue(session.CanRetry);
            Assert.IsFalse(session.CanContinueWithNeedsPractice);
            Assert.AreEqual(1, session.CurrentWrongWordAttemptCount);
            Assert.AreEqual(0, repository.SaveCalls);
            Assert.AreEqual(CandidateStorageStatus.NotRequested, session.LastStorageResult.Status);
        }

        [Test]
        public void Retry1WrongWord_BlocksAdditionalRetry()
        {
            PetRobotSignPracticeSession session = TwoWrongAttempts(out _, out FakeRepository repository);

            Assert.IsFalse(session.CanRetry);
            Assert.IsFalse(session.Retry());
            Assert.AreEqual(2, session.CurrentWrongWordAttemptCount);
            Assert.AreEqual(0, repository.SaveCalls);
        }

        [Test]
        public void Retry1WrongWord_ExposesExplicitNeedsPracticeContinue()
        {
            PetRobotSignPracticeSession session = TwoWrongAttempts(out _, out _);

            Assert.IsTrue(session.CanContinueWithNeedsPractice);
            Assert.AreEqual(
                "이번 단어는 연습 필요로 남기고 계속",
                PetRobotPracticeUiPresenter.NeedsPracticeContinueButtonLabel);
        }

        [Test]
        public void Retry1WrongWord_DoesNotAutomaticallyAdvance()
        {
            PetRobotSignPracticeSession session = TwoWrongAttempts(out _, out _);

            Assert.AreEqual(PetRobotPracticeState.Feedback, session.State);
            Assert.AreEqual("GREETING", session.CurrentWord);
            Assert.AreEqual(PetRobotWordLearningStatus.Pending, session.GetWordLearningStatus("GREETING"));
        }

        [Test]
        public void Continue_PreservesWrongWordOutcomeAndStrictFalse()
        {
            PetRobotSignPracticeSession session = TwoWrongAttempts(out _, out _);
            PracticeRecognitionAttempt fixedAttempt = session.GetWordFinalAttempt("GREETING");

            Assert.IsTrue(session.ContinueWithNeedsPractice());

            PracticeRecognitionAttempt preserved = session.GetWordFinalAttempt("GREETING");
            Assert.AreEqual(fixedAttempt.CaptureSessionId, preserved.CaptureSessionId);
            Assert.AreEqual(GuidedQuestOutcomeKind.WrongWord, preserved.Outcome.Kind);
            Assert.IsFalse(preserved.StrictAccepted);
            Assert.IsFalse(preserved.StrictExpectedWordClassAccept);
            Assert.IsFalse(preserved.StrictQuestSuccess);
        }

        [Test]
        public void Continue_DoesNotStoreCandidate()
        {
            PetRobotSignPracticeSession session = TwoWrongAttempts(out _, out FakeRepository repository);

            Assert.IsTrue(session.ContinueWithNeedsPractice());
            Assert.AreEqual(0, repository.SaveCalls);
            Assert.AreEqual(CandidateStorageStatus.NotRequested, session.LastStorageResult.Status);
            Assert.AreEqual(0, session.CandidateCounts.TotalCandidateCount);
        }

        [Test]
        public void Continue_MarksOnlyCurrentWordNeedsPractice()
        {
            PetRobotSignPracticeSession session = TwoWrongAttempts(out _, out _);

            Assert.IsTrue(session.ContinueWithNeedsPractice());
            Assert.AreEqual(PetRobotWordLearningStatus.NeedsPractice, session.GetWordLearningStatus("GREETING"));
            Assert.AreEqual(PetRobotWordLearningStatus.Pending, session.GetWordLearningStatus("LOVE"));
            Assert.AreEqual(PetRobotWordLearningStatus.Pending, session.GetWordLearningStatus("THANK_YOU"));
        }

        [Test]
        public void Continue_NonFinalWordAdvancesToNextWord()
        {
            PetRobotSignPracticeSession session = TwoWrongAttempts(out _, out _);

            Assert.IsTrue(session.ContinueWithNeedsPractice());
            Assert.AreEqual(PetRobotPracticeState.Ready, session.State);
            Assert.AreEqual("LOVE", session.CurrentWord);
            Assert.AreEqual(2, session.CurrentStep);
        }

        [Test]
        public void FinalNeedsPractice_CompletesSession()
        {
            PetRobotSignPracticeSession session = ReachFinalWrongRetryExhausted(out _, out _);

            Assert.IsTrue(session.ContinueWithNeedsPractice());
            Assert.AreEqual(PetRobotPracticeState.Completed, session.State);
            Assert.IsTrue(session.AllWordsComplete);
            Assert.AreEqual(PetRobotWordLearningStatus.NeedsPractice, session.GetWordLearningStatus("THANK_YOU"));
        }

        [Test]
        public void Summary_ReportsMasteredTwoAndNeedsPracticeOne()
        {
            PetRobotSignPracticeSession session = ReachFinalWrongRetryExhausted(out _, out _);
            session.ContinueWithNeedsPractice();

            Assert.AreEqual(2, session.MasteredCount);
            Assert.AreEqual(1, session.NeedsPracticeCount);
            Assert.AreEqual(PetRobotWordLearningStatus.Mastered, session.GetWordLearningStatus("GREETING"));
            Assert.AreEqual(PetRobotWordLearningStatus.Mastered, session.GetWordLearningStatus("LOVE"));
        }

        [Test]
        public void CompletedSession_AllowsExit()
        {
            PetRobotSignPracticeSession session = ReachFinalWrongRetryExhausted(out _, out _);
            session.ContinueWithNeedsPractice();

            Assert.IsTrue(session.ExitPractice());
            Assert.AreEqual(PetRobotPracticeState.Exited, session.State);
        }

        [Test]
        public void ExistingCandidateCount_RemainsTwoAcrossNeedsPracticeContinue()
        {
            var adapter = new FakeRecognitionAdapter();
            var repository = new FakeRepository(1, 1);
            var session = new PetRobotSignPracticeSession(adapter, repository);
            Finish(session, adapter, Attempt("g-existing-wrong-1", "GREETING", "LOVE", false, false, false));
            session.Retry();
            Finish(session, adapter, Attempt("g-existing-wrong-2", "GREETING", "LOVE", false, false, false));

            Assert.IsTrue(session.ContinueWithNeedsPractice());
            Assert.AreEqual(2, session.CandidateCounts.TotalCandidateCount);
            Assert.AreEqual(0, repository.SaveCalls);
        }

        [Test]
        public void TechnicalFailure_IsNotNeedsPracticeAndCannotContinue()
        {
            PetRobotSignPracticeSession session = NewSession(out FakeRecognitionAdapter adapter, out FakeRepository repository);
            Finish(session, adapter, TechnicalAttempt("g-technical", "GREETING"));

            Assert.AreEqual(GuidedQuestOutcomeKind.TechnicalFailure, session.LastLearningOutcome.Kind);
            Assert.AreEqual(PetRobotWordLearningStatus.Pending, session.GetWordLearningStatus("GREETING"));
            Assert.IsFalse(session.CanContinueWithNeedsPractice);
            Assert.AreEqual(0, session.NeedsPracticeCount);
            Assert.AreEqual(0, repository.SaveCalls);
        }

        [Test]
        public void StrictSuccess_RemainsMasteredAndEligible()
        {
            PetRobotSignPracticeSession session = NewSession(out FakeRecognitionAdapter adapter, out FakeRepository repository);
            Finish(session, adapter, Attempt("g-strict", "GREETING", "GREETING", true, true, true));

            Assert.AreEqual(GuidedQuestOutcomeKind.StrictSuccess, session.LastLearningOutcome.Kind);
            Assert.AreEqual(PetRobotWordLearningStatus.Mastered, session.GetWordLearningStatus("GREETING"));
            Assert.AreEqual(PersonalizationCandidateEligibility.Eligible, repository.LastEligibility);
            Assert.AreEqual(1, session.CandidateCounts.TotalCandidateCount);
        }

        [Test]
        public void LowConfidenceCorrect_RemainsMasteredReviewOnlyAndStrictFalse()
        {
            PetRobotSignPracticeSession session = NewSession(out FakeRecognitionAdapter adapter, out FakeRepository repository);
            Finish(session, adapter, Attempt("g-low", "GREETING", "GREETING", false, false, false));

            Assert.AreEqual(GuidedQuestOutcomeKind.LowConfidenceCorrect, session.LastLearningOutcome.Kind);
            Assert.IsFalse(session.LastAttempt.StrictAccepted);
            Assert.IsFalse(session.LastAttempt.StrictExpectedWordClassAccept);
            Assert.IsFalse(session.LastAttempt.StrictQuestSuccess);
            Assert.AreEqual(PetRobotWordLearningStatus.Mastered, session.GetWordLearningStatus("GREETING"));
            Assert.AreEqual(PersonalizationCandidateEligibility.ReviewOnly, repository.LastEligibility);
        }

        static PetRobotSignPracticeSession TwoWrongAttempts(
            out FakeRecognitionAdapter adapter,
            out FakeRepository repository)
        {
            PetRobotSignPracticeSession session = NewSession(out adapter, out repository);
            Finish(session, adapter, Attempt("g-wrong-1", "GREETING", "LOVE", false, false, false));
            Assert.IsTrue(session.Retry());
            Finish(session, adapter, Attempt("g-wrong-2", "GREETING", "LOVE", false, false, false));
            return session;
        }

        static PetRobotSignPracticeSession ReachFinalWrongRetryExhausted(
            out FakeRecognitionAdapter adapter,
            out FakeRepository repository)
        {
            PetRobotSignPracticeSession session = NewSession(out adapter, out repository);
            Finish(session, adapter, Attempt("g-mastered", "GREETING", "GREETING", true, true, true));
            Assert.IsTrue(session.ContinueToNextWord());
            Finish(session, adapter, Attempt("l-mastered", "LOVE", "LOVE", true, true, true));
            Assert.IsTrue(session.ContinueToNextWord());
            Finish(session, adapter, Attempt("t-wrong-1", "THANK_YOU", "LOVE", false, false, false));
            Assert.IsTrue(session.Retry());
            Finish(session, adapter, Attempt("t-wrong-2", "THANK_YOU", "LOVE", false, false, false));
            return session;
        }

        static PetRobotSignPracticeSession NewSession(
            out FakeRecognitionAdapter adapter,
            out FakeRepository repository)
        {
            adapter = new FakeRecognitionAdapter();
            repository = new FakeRepository();
            return new PetRobotSignPracticeSession(adapter, repository,
                () => new DateTime(2026, 8, 11, 12, 0, 0, DateTimeKind.Utc));
        }

        static void Finish(
            PetRobotSignPracticeSession session,
            FakeRecognitionAdapter adapter,
            PracticeRecognitionAttempt attempt)
        {
            adapter.Next = attempt;
            Assert.IsTrue(session.StartExpression());
            Assert.IsTrue(session.FinishExpression());
        }

        static PracticeRecognitionAttempt Attempt(
            string sessionId,
            string expected,
            string predicted,
            bool accepted,
            bool expectedAccept,
            bool questSuccess)
        {
            GuidedQuestOutcome outcome = GuidedQuestOutcomeEvaluator.EvaluateForRoute(
                PersonalLiveProfileMode.MultiPrototypeCandidate,
                PersonalLiveCaptureState.Completed,
                true,
                expected,
                predicted,
                accepted,
                expectedAccept,
                questSuccess,
                false,
                false);
            return new PracticeRecognitionAttempt(
                sessionId,
                expected,
                predicted,
                PersonalLiveCaptureState.Completed,
                outcome,
                null,
                null,
                PersonalLiveProfileMode.MultiPrototypeCandidate.ToString(),
                "step9l_current_user_multi_prototype_v1_0",
                Threshold(expected),
                null,
                null,
                new PracticeRecognitionMetrics
                {
                    StrictAccepted = accepted,
                    StrictExpectedWordClassAccept = expectedAccept,
                    StrictQuestSuccess = questSuccess,
                    GreetingDistance = expected == "GREETING" ? 0.43 : 0.70,
                    LoveDistance = predicted == "LOVE" ? 0.30 : 0.68,
                    ThankYouDistance = expected == "THANK_YOU" ? 0.45 : 0.77,
                    BestDistance = predicted == "LOVE" ? 0.30 : 0.43,
                    WinningPrototypeId = "prototype-" + predicted,
                    RawFrameCount = 5,
                    NormalizedFrameCount = 5,
                    ValidLandmarkCount = 80,
                    InterpolatedLandmarkCount = 0,
                });
        }

        static PracticeRecognitionAttempt TechnicalAttempt(string sessionId, string expected)
        {
            GuidedQuestOutcome outcome = GuidedQuestOutcomeEvaluator.EvaluateForRoute(
                PersonalLiveProfileMode.MultiPrototypeCandidate,
                PersonalLiveCaptureState.Failed,
                false,
                expected,
                string.Empty,
                false,
                false,
                false,
                true,
                true);
            return new PracticeRecognitionAttempt(
                sessionId,
                expected,
                string.Empty,
                PersonalLiveCaptureState.Failed,
                outcome,
                null,
                null,
                PersonalLiveProfileMode.MultiPrototypeCandidate.ToString(),
                "step9l_current_user_multi_prototype_v1_0",
                Threshold(expected),
                null,
                null,
                new PracticeRecognitionMetrics { WinningPrototypeId = string.Empty });
        }

        static double Threshold(string word) =>
            word == "GREETING" ? 0.5113059473720765 :
            word == "LOVE" ? 0.391276023460427 : 0.37709977431991587;

        sealed class FakeRecognitionAdapter : IPersonalLivePracticeRecognitionAdapter
        {
            public PracticeRecognitionAttempt Next;
            PersonalLiveCaptureState state = PersonalLiveCaptureState.Ready;

            public PersonalLiveCaptureState CaptureState => state;
            public string CaptureSessionId => Next == null ? string.Empty : Next.CaptureSessionId;
            public string ActiveProfileId => "step9l_current_user_multi_prototype_v1_0";
            public bool CandidateRouteActive => true;
            public bool SetExpectedWord(string word) =>
                state == PersonalLiveCaptureState.Ready && PersonalLiveRecognitionController.IsSupportedWord(word);
            public bool BeginCapture()
            {
                if (state != PersonalLiveCaptureState.Ready) return false;
                state = PersonalLiveCaptureState.Capturing;
                return true;
            }
            public PracticeRecognitionAttempt FinalizeCapture()
            {
                state = Next == null ? PersonalLiveCaptureState.Failed : Next.CaptureState;
                return Next;
            }
            public bool ResetCapture()
            {
                if (state != PersonalLiveCaptureState.Completed && state != PersonalLiveCaptureState.Failed) return false;
                state = PersonalLiveCaptureState.Ready;
                return true;
            }
        }

        sealed class FakeRepository : IPersonalizationCandidateRepository
        {
            readonly Dictionary<string, int> eligible = WordCounts(0);
            readonly Dictionary<string, int> reviewOnly = WordCounts(0);

            public int SaveCalls;
            public PersonalizationCandidateEligibility LastEligibility;
            public string RootDirectory => "fake";
            public string LearnerId => "local_learner";
            public CandidateCountSnapshot Counts => new CandidateCountSnapshot(eligible, reviewOnly);

            public FakeRepository(int greetingEligible = 0, int loveEligible = 0)
            {
                eligible["GREETING"] = greetingEligible;
                eligible["LOVE"] = loveEligible;
            }

            public CandidateCountSnapshot ReloadAndVerify() => Counts;

            public CandidateSaveResult Save(PersonalizationCandidateSaveRequest request)
            {
                SaveCalls++;
                LastEligibility = request.Eligibility;
                Dictionary<string, int> target = request.Eligibility == PersonalizationCandidateEligibility.Eligible
                    ? eligible
                    : reviewOnly;
                target[request.Attempt.ExpectedWord]++;
                return new CandidateSaveResult(CandidateStorageStatus.Saved, "fake", "saved", Counts);
            }
        }

        static Dictionary<string, int> WordCounts(int value) =>
            new Dictionary<string, int>(StringComparer.Ordinal)
            {
                { "GREETING", value },
                { "LOVE", value },
                { "THANK_YOU", value },
            };
    }
}
