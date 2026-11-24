using System;
using System.Collections.Generic;
using NUnit.Framework;
using WordsInSilence.Practice.PetRobotPortfolioQuest;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Practice.PetRobotWorldIntegration;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class GreetingContextualTransferV1Tests
    {
        // ── spy fakes ──

        sealed class SpyRepository : ITrustedPersonalizationCandidateRepository
        {
            readonly TrustedCandidateInventory inventory;
            public int ReloadCount { get; private set; }
            public int SaveCount { get; private set; }

            public string RootDirectory => string.Empty;
            public string LearnerId => "test-learner";
            public CandidateCountSnapshot Counts => CandidateCountSnapshot.Empty;

            public SpyRepository(TrustedCandidateInventory inventory)
            {
                this.inventory = inventory;
            }

            public TrustedCandidateInventory TrustedInventory => inventory;
            public TrustedCandidateInventory ReloadTrustedAndVerify()
            {
                ReloadCount++;
                return inventory;
            }
            public CandidateCountSnapshot ReloadAndVerify() => CandidateCountSnapshot.Empty;
            public CandidateSaveResult Save(PersonalizationCandidateSaveRequest request)
            {
                SaveCount++;
                return null;
            }
        }

        sealed class SpySessionProvider : IGreetingRecognitionSessionProvider
        {
            readonly bool refreshResult;
            readonly IGreetingRecognitionSession session;
            public int RefreshCount { get; private set; }
            public int AcquireCount { get; private set; }

            public SpySessionProvider(bool refreshResult, IGreetingRecognitionSession session)
            {
                this.refreshResult = refreshResult;
                this.session = session;
            }

            public bool TryRefresh()
            {
                RefreshCount++;
                return refreshResult;
            }

            public IGreetingRecognitionSession AcquireSession()
            {
                AcquireCount++;
                return session;
            }
        }

        sealed class SpyRecognitionSession : IGreetingRecognitionSession
        {
            readonly TrustedHybridOutcomeKind outcomeKind;
            public long Generation { get; }
            public int RecognizeCount { get; private set; }

            public SpyRecognitionSession(long generation, TrustedHybridOutcomeKind outcomeKind)
            {
                Generation = generation;
                this.outcomeKind = outcomeKind;
            }

            public TrustedHybridOutcomeKind RecognizeGreeting(
                string querySessionId, RecognitionSequence query)
            {
                RecognizeCount++;
                return outcomeKind;
            }
        }

        sealed class SpyCaptureAdapter : IPersonalLivePracticeRecognitionAdapter
        {
            readonly PracticeRecognitionAttempt finalizeResult;
            public PersonalLiveCaptureState CaptureState { get; set; }
            public string CaptureSessionId => "spy-session";
            public string ActiveProfileId => "spy-profile";
            public bool CandidateRouteActive => true;
            public int SetExpectedWordCount { get; private set; }
            public int BeginCaptureCount { get; private set; }
            public int FinalizeCaptureCount { get; private set; }
            public int ResetCaptureCount { get; private set; }
            public string LastExpectedWord { get; private set; }

            public SpyCaptureAdapter(PracticeRecognitionAttempt finalizeResult = null)
            {
                this.finalizeResult = finalizeResult;
            }

            public bool SetExpectedWord(string word)
            {
                SetExpectedWordCount++;
                LastExpectedWord = word;
                return true;
            }
            public bool BeginCapture()
            {
                BeginCaptureCount++;
                return true;
            }
            public PracticeRecognitionAttempt FinalizeCapture()
            {
                FinalizeCaptureCount++;
                return finalizeResult;
            }
            public bool ResetCapture()
            {
                ResetCaptureCount++;
                return true;
            }
        }

        // ── helpers ──

        static TrustedCandidateInventory GateReady()
        {
            var counts = new Dictionary<string, int>(StringComparer.Ordinal)
            {
                { "GREETING", 3 }, { "LOVE", 3 }, { "THANK_YOU", 3 }
            };
            return new TrustedCandidateInventory(
                CandidateCountSnapshot.Empty, counts,
                Array.Empty<TrustedPersonalizationCandidate>(), 0);
        }

        static TrustedCandidateInventory GateNotReady()
        {
            var counts = new Dictionary<string, int>(StringComparer.Ordinal)
            {
                { "GREETING", 2 }, { "LOVE", 3 }, { "THANK_YOU", 1 }
            };
            return new TrustedCandidateInventory(
                CandidateCountSnapshot.Empty, counts,
                Array.Empty<TrustedPersonalizationCandidate>(), 0);
        }

        static RecognitionSequence DummySequence()
        {
            const int len = 2;
            const int lm = 16;
            return new RecognitionSequence(
                "test-sample", "GREETING", "normal", "test",
                new string('a', 64), new string('b', 64),
                len, lm,
                new double[len * lm * 3],
                CreateValidMask(len * lm),
                new string('c', 64));
        }

        static bool[] CreateValidMask(int size)
        {
            var mask = new bool[size];
            for (int i = 0; i < size; i++) mask[i] = true;
            return mask;
        }

        static PracticeRecognitionAttempt DummyAttempt(RecognitionSequence seq = null)
        {
            return new PracticeRecognitionAttempt(
                "spy-session", "GREETING", "GREETING",
                PersonalLiveCaptureState.Completed,
                null, null,
                seq ?? DummySequence(),
                "test-route", "test-profile", 0.5,
                null, null);
        }

        static GreetingContextualTransferService CreateService(
            SpyRepository repo, SpySessionProvider provider, SpyCaptureAdapter capture)
        {
            return new GreetingContextualTransferService(repo, provider, capture);
        }

        static GreetingContextualTransferService StartedService(
            TrustedHybridOutcomeKind outcomeKind = TrustedHybridOutcomeKind.StrictSuccess,
            PracticeRecognitionAttempt attempt = null)
        {
            var session = new SpyRecognitionSession(42, outcomeKind);
            var repo = new SpyRepository(GateReady());
            var provider = new SpySessionProvider(true, session);
            var capture = new SpyCaptureAdapter(attempt ?? DummyAttempt());
            var service = CreateService(repo, provider, capture);
            service.TryStart();
            return service;
        }

        // ── 1. gate false blocks greeting transfer ──
        [Test]
        public void GateFalse_BlocksGreetingTransfer()
        {
            var repo = new SpyRepository(GateNotReady());
            var session = new SpyRecognitionSession(1, TrustedHybridOutcomeKind.StrictSuccess);
            var provider = new SpySessionProvider(true, session);
            var capture = new SpyCaptureAdapter();
            var service = CreateService(repo, provider, capture);

            Assert.IsFalse(service.TryStart());
            Assert.AreEqual(GreetingTransferState.GateCheckFailed, service.State);
        }

        // ── 2. gate false: no acquire/capture/save ──
        [Test]
        public void GateFalse_NoAcquireOrCaptureOrSave()
        {
            var repo = new SpyRepository(GateNotReady());
            var session = new SpyRecognitionSession(1, TrustedHybridOutcomeKind.StrictSuccess);
            var provider = new SpySessionProvider(true, session);
            var capture = new SpyCaptureAdapter();
            var service = CreateService(repo, provider, capture);

            service.TryStart();
            Assert.AreEqual(0, service.SessionAcquireCount);
            Assert.AreEqual(0, service.CandidateSaveCount);
            Assert.IsFalse(service.TryBeginCapture());
            Assert.AreEqual(0, provider.AcquireCount);
            Assert.AreEqual(0, capture.BeginCaptureCount);
        }

        // ── 3. gate true: greeting ready ──
        [Test]
        public void GateTrue_GreetingReady()
        {
            var session = new SpyRecognitionSession(42, TrustedHybridOutcomeKind.StrictSuccess);
            var repo = new SpyRepository(GateReady());
            var provider = new SpySessionProvider(true, session);
            var capture = new SpyCaptureAdapter();
            var service = CreateService(repo, provider, capture);

            Assert.IsTrue(service.TryStart());
            Assert.AreEqual(GreetingTransferState.Ready, service.State);
            Assert.AreEqual("GREETING", service.ExpectedWord);
        }

        // ── 4. active view refresh failure ──
        [Test]
        public void ActiveViewRefreshFailure_FailClosed()
        {
            var repo = new SpyRepository(GateReady());
            var provider = new SpySessionProvider(false, null);
            var capture = new SpyCaptureAdapter();
            var service = CreateService(repo, provider, capture);

            Assert.IsFalse(service.TryStart());
            Assert.AreEqual(GreetingTransferState.ActiveViewFailed, service.State);
        }

        // ── 5. session acquire exactly once ──
        [Test]
        public void SessionAcquire_ExactlyOnce()
        {
            var session = new SpyRecognitionSession(10, TrustedHybridOutcomeKind.StrictSuccess);
            var repo = new SpyRepository(GateReady());
            var provider = new SpySessionProvider(true, session);
            var capture = new SpyCaptureAdapter();
            var service = CreateService(repo, provider, capture);

            service.TryStart();
            Assert.AreEqual(1, service.SessionAcquireCount);
            Assert.AreEqual(1, provider.AcquireCount);
        }

        // ── 6. expected word always GREETING ──
        [Test]
        public void ExpectedWord_AlwaysGreeting()
        {
            var service = StartedService();
            Assert.AreEqual("GREETING", service.ExpectedWord);
        }

        // ── 7. frozen generation invariant after start ──
        [Test]
        public void FrozenGeneration_InvariantAfterStart()
        {
            var session = new SpyRecognitionSession(5, TrustedHybridOutcomeKind.StrictSuccess);
            var repo = new SpyRepository(GateReady());
            var provider = new SpySessionProvider(true, session);
            var capture = new SpyCaptureAdapter();
            var service = CreateService(repo, provider, capture);

            service.TryStart();
            Assert.AreEqual(5, service.FrozenGeneration);
        }

        // ── 8. SetExpectedWord called on start ──
        [Test]
        public void Start_SetsExpectedWordOnAdapter()
        {
            var session = new SpyRecognitionSession(1, TrustedHybridOutcomeKind.StrictSuccess);
            var repo = new SpyRepository(GateReady());
            var provider = new SpySessionProvider(true, session);
            var capture = new SpyCaptureAdapter();
            var service = CreateService(repo, provider, capture);

            service.TryStart();
            Assert.AreEqual(1, capture.SetExpectedWordCount);
            Assert.AreEqual("GREETING", capture.LastExpectedWord);
        }

        // ── 9. begin capture calls adapter ──
        [Test]
        public void BeginCapture_CallsAdapterBeginCapture()
        {
            var service = StartedService();
            Assert.IsTrue(service.TryBeginCapture());
            Assert.AreEqual(GreetingTransferState.Capturing, service.State);
        }

        // ── 10. StrictSuccess completes ──
        [Test]
        public void StrictSuccess_Completes()
        {
            var service = StartedService(TrustedHybridOutcomeKind.StrictSuccess, DummyAttempt());
            service.TryBeginCapture();
            Assert.IsTrue(service.TryFinalizeCapture());
            Assert.IsTrue(service.IsCompleted);
            Assert.AreEqual(1, service.CompletionCount);
            StringAssert.Contains("\uC815\uD655\uD788", service.LastFeedbackCopy);
        }

        // ── 11. LowConfidenceCorrect completes with distinct copy ──
        [Test]
        public void LowConfidenceCorrect_CompletesWithDistinctCopy()
        {
            var service = StartedService(TrustedHybridOutcomeKind.LowConfidenceCorrect, DummyAttempt());
            service.TryBeginCapture();
            Assert.IsTrue(service.TryFinalizeCapture());
            Assert.IsTrue(service.IsCompleted);
            StringAssert.Contains("\uB610\uB837\uD558\uAC8C", service.LastFeedbackCopy);
        }

        // ── 12. WrongWord: not completed, goes to AwaitingRetry ──
        [Test]
        public void WrongWord_NotCompleted_AwaitingRetry()
        {
            var service = StartedService(TrustedHybridOutcomeKind.WrongWord, DummyAttempt());
            service.TryBeginCapture();
            service.TryFinalizeCapture();
            Assert.IsFalse(service.IsCompleted);
            Assert.AreEqual(GreetingTransferState.AwaitingRetry, service.State);
            StringAssert.Contains("\uB2E4\uC2DC \uC2DC\uB3C4", service.LastFeedbackCopy);
        }

        // ── 13. TechnicalFailure: not completed, awaiting retry ──
        [Test]
        public void TechnicalFailure_NotCompleted_AwaitingRetry()
        {
            var service = StartedService(TrustedHybridOutcomeKind.TechnicalFailure, DummyAttempt());
            service.TryBeginCapture();
            service.TryFinalizeCapture();
            Assert.IsFalse(service.IsCompleted);
            Assert.AreEqual(GreetingTransferState.AwaitingRetry, service.State);
            StringAssert.Contains("\uCC44\uC810\uD558\uC9C0", service.LastFeedbackCopy);
        }

        // ── 14. null finalize result → TechnicalFailure → AwaitingRetry ──
        [Test]
        public void NullFinalizeResult_TechnicalFailure_AwaitingRetry()
        {
            var session = new SpyRecognitionSession(1, TrustedHybridOutcomeKind.StrictSuccess);
            var repo = new SpyRepository(GateReady());
            var provider = new SpySessionProvider(true, session);
            var capture = new SpyCaptureAdapter(null);
            var service = CreateService(repo, provider, capture);

            service.TryStart();
            service.TryBeginCapture();
            service.TryFinalizeCapture();
            Assert.AreEqual(GreetingTransferState.AwaitingRetry, service.State);
            Assert.AreEqual(GuidedQuestOutcomeKind.TechnicalFailure, service.LastOutcomeKind);
        }

        // ── 15. retry transitions AwaitingRetry → Ready ──
        [Test]
        public void Retry_TransitionsToReady()
        {
            var service = StartedService(TrustedHybridOutcomeKind.WrongWord, DummyAttempt());
            service.TryBeginCapture();
            service.TryFinalizeCapture();
            Assert.AreEqual(GreetingTransferState.AwaitingRetry, service.State);

            Assert.IsTrue(service.TryRetry());
            Assert.AreEqual(GreetingTransferState.Ready, service.State);
        }

        // ── 16. retry from non-AwaitingRetry is no-op ──
        [Test]
        public void Retry_FromReady_IsNoOp()
        {
            var service = StartedService();
            Assert.IsFalse(service.TryRetry());
            Assert.AreEqual(GreetingTransferState.Ready, service.State);
        }

        // ── 17. after retry, no snapshot re-acquire ──
        [Test]
        public void AfterRetry_NoSnapshotReacquire()
        {
            var session = new SpyRecognitionSession(1, TrustedHybridOutcomeKind.WrongWord);
            var repo = new SpyRepository(GateReady());
            var provider = new SpySessionProvider(true, session);
            var capture = new SpyCaptureAdapter(DummyAttempt());
            var service = CreateService(repo, provider, capture);

            service.TryStart();
            Assert.AreEqual(1, provider.AcquireCount);

            service.TryBeginCapture();
            service.TryFinalizeCapture();
            service.TryRetry();
            Assert.AreEqual(1, provider.AcquireCount);
            Assert.AreEqual(1, service.SessionAcquireCount);
        }

        // ── 18. all outcomes: candidate save count zero ──
        [TestCase(TrustedHybridOutcomeKind.StrictSuccess)]
        [TestCase(TrustedHybridOutcomeKind.LowConfidenceCorrect)]
        [TestCase(TrustedHybridOutcomeKind.WrongWord)]
        [TestCase(TrustedHybridOutcomeKind.TechnicalFailure)]
        public void AllOutcomes_CandidateSaveZero(TrustedHybridOutcomeKind kind)
        {
            var session = new SpyRecognitionSession(1, kind);
            var repo = new SpyRepository(GateReady());
            var provider = new SpySessionProvider(true, session);
            var capture = new SpyCaptureAdapter(DummyAttempt());
            var service = CreateService(repo, provider, capture);

            service.TryStart();
            service.TryBeginCapture();
            service.TryFinalizeCapture();
            Assert.AreEqual(0, service.CandidateSaveCount);
            Assert.AreEqual(0, repo.SaveCount);
        }

        // ── 19. completed: re-interaction/retry/capture no-op ──
        [Test]
        public void AfterCompletion_ReinteractionAndRetryAreNoOp()
        {
            var service = StartedService(TrustedHybridOutcomeKind.StrictSuccess, DummyAttempt());
            service.TryBeginCapture();
            service.TryFinalizeCapture();
            Assert.IsTrue(service.IsCompleted);

            Assert.IsFalse(service.TryStart());
            Assert.IsFalse(service.TryBeginCapture());
            Assert.IsFalse(service.TryRetry());
            Assert.AreEqual(1, service.CompletionCount);
        }

        // ── 20. repository reload called exactly once on TryStart ──
        [Test]
        public void RepositoryReload_CalledOnceOnStart()
        {
            var session = new SpyRecognitionSession(1, TrustedHybridOutcomeKind.StrictSuccess);
            var repo = new SpyRepository(GateReady());
            var provider = new SpySessionProvider(true, session);
            var capture = new SpyCaptureAdapter();
            var service = CreateService(repo, provider, capture);

            service.TryStart();
            Assert.AreEqual(1, repo.ReloadCount);
        }

        // ── 21. recognizer called on finalize with attempt sequence ──
        [Test]
        public void Finalize_CallsRecognizerWithAttemptSequence()
        {
            var session = new SpyRecognitionSession(1, TrustedHybridOutcomeKind.StrictSuccess);
            var repo = new SpyRepository(GateReady());
            var provider = new SpySessionProvider(true, session);
            var capture = new SpyCaptureAdapter(DummyAttempt());
            var service = CreateService(repo, provider, capture);

            service.TryStart();
            service.TryBeginCapture();
            service.TryFinalizeCapture();
            Assert.AreEqual(1, session.RecognizeCount);
        }

        // ── 22. reset capture called on retry ──
        [Test]
        public void Retry_CallsResetCapture()
        {
            var session = new SpyRecognitionSession(1, TrustedHybridOutcomeKind.WrongWord);
            var repo = new SpyRepository(GateReady());
            var provider = new SpySessionProvider(true, session);
            var capture = new SpyCaptureAdapter(DummyAttempt());
            var service = CreateService(repo, provider, capture);

            service.TryStart();
            service.TryBeginCapture();
            service.TryFinalizeCapture();
            service.TryRetry();
            Assert.AreEqual(1, capture.ResetCaptureCount);
        }

        // ── 23. no LOVE/THANK_YOU quest path ──
        [Test]
        public void NoLoveOrThankYouQuestPath()
        {
            string source = ReadProject(
                "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/GreetingContextualTransferService.cs");
            Assert.IsFalse(source.Contains("\"LOVE\""));
            Assert.IsFalse(source.Contains("\"THANK_YOU\""));
        }

        // ── 24. player UI: no threshold/path/diagnostic ──
        [Test]
        public void PlayerUi_NoThresholdOrPathOrDiagnostic()
        {
            string presenter = ReadProject(
                "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestUiPresenter.cs");
            Assert.IsFalse(presenter.Contains("threshold"));
            Assert.IsFalse(presenter.Contains("Threshold"));
            Assert.IsFalse(presenter.Contains("candidatePath"));
            Assert.IsFalse(presenter.Contains("diagnostic"));
            string transferSource = ReadProject(
                "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/GreetingContextualTransferService.cs");
            Assert.IsFalse(transferSource.Contains("threshold"));
            Assert.IsFalse(transferSource.Contains("Threshold"));
        }

        // ── 25. world practice-entry key unchanged ──
        [Test]
        public void WorldPracticeEntryKey_SourceUnchanged()
        {
            string worldEntry = ReadProject(
                "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldEntryController.cs");
            Assert.IsTrue(worldEntry.Contains("KeyCode.E"));
            Assert.IsTrue(worldEntry.Contains("KeyCode.Return"));
            Assert.IsTrue(worldEntry.Contains("KeyCode.Space"));
            Assert.IsFalse(worldEntry.Contains("GreetingContextualTransfer"));
        }

        // ── flow integration (from PetRobotPortfolioQuestPresentationV1Tests compatibility) ──

        [Test]
        public void PracticeCompletionSnapshot_FreezePreserved()
        {
            var flow = CompletedFlow();
            Assert.AreEqual(1, flow.CompletionFreezeCount);
            Assert.IsNotNull(flow.Snapshot);
            flow.SetNpcInRange(true);
            flow.TryOpenDialogue(true);
            flow.TryCompleteDialogue(true);
            Assert.AreEqual(PetRobotPortfolioQuestState.AwaitingGreetingAttempt, flow.State);
            Assert.AreEqual(1, flow.CompletionFreezeCount);
        }

        [Test]
        public void NoExplicitNpcInteraction_NoQuestStart()
        {
            var flow = CompletedFlow();
            flow.SetNpcInRange(true);
            Assert.IsFalse(flow.TryOpenDialogue(false));
            Assert.AreEqual(PetRobotPortfolioQuestState.AwaitingNpcReport, flow.State);
        }

        [Test]
        public void NpcRangeFalse_NoQuestStart()
        {
            var flow = CompletedFlow();
            flow.SetNpcInRange(false);
            Assert.IsFalse(flow.TryOpenDialogue(true));
        }

        // ── helpers ──

        static PetRobotPortfolioQuestFlow CompletedFlow()
        {
            var flow = new PetRobotPortfolioQuestFlow();
            flow.ObserveCompleted(new PetRobotPracticeCompletionSnapshot(
                PetRobotWordLearningStatus.Mastered,
                PetRobotWordLearningStatus.Mastered,
                PetRobotWordLearningStatus.Mastered,
                3, 0, 9));
            return flow;
        }

        static string ReadProject(string relativePath) =>
            System.IO.File.ReadAllText(
                System.IO.Path.Combine(
                    System.IO.Directory.GetParent(UnityEngine.Application.dataPath).FullName,
                    relativePath)).Replace("\r\n", "\n");
    }
}
