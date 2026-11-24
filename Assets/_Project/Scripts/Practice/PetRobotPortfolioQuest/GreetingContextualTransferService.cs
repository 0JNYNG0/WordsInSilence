using System;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Practice.PetRobotPortfolioQuest
{
    public enum GreetingTransferState
    {
        NotStarted,
        GateCheckFailed,
        ActiveViewFailed,
        Ready,
        Capturing,
        AwaitingRetry,
        Completed,
    }

    public interface IGreetingRecognitionSessionProvider
    {
        bool TryRefresh();
        IGreetingRecognitionSession AcquireSession();
    }

    public interface IGreetingRecognitionSession
    {
        long Generation { get; }
        TrustedHybridOutcomeKind RecognizeGreeting(
            string querySessionId, RecognitionSequence query);
    }

    public sealed class ProductionGreetingRecognitionSessionProvider
        : IGreetingRecognitionSessionProvider
    {
        readonly TrustedPersonalizationActiveViewService viewService;

        public ProductionGreetingRecognitionSessionProvider(
            TrustedPersonalizationActiveViewService viewService)
        {
            this.viewService = viewService ??
                throw new ArgumentNullException(nameof(viewService));
        }

        public bool TryRefresh()
        {
            TrustedActiveViewBuildResult result = viewService.TryRefresh();
            return result.Succeeded;
        }

        public IGreetingRecognitionSession AcquireSession()
        {
            return new HybridSessionAdapter(viewService.AcquireSession());
        }

        sealed class HybridSessionAdapter : IGreetingRecognitionSession
        {
            readonly TrustedHybridRuntimeSession session;
            public long Generation => session.ActiveViewGeneration;

            internal HybridSessionAdapter(TrustedHybridRuntimeSession session)
            {
                this.session = session ??
                    throw new ArgumentNullException(nameof(session));
            }

            public TrustedHybridOutcomeKind RecognizeGreeting(
                string querySessionId, RecognitionSequence query)
            {
                TrustedHybridRecognitionResult result =
                    session.Recognize(querySessionId, query, "GREETING");
                return result.Kind;
            }
        }
    }

    public sealed class GreetingContextualTransferService
    {
        readonly ITrustedPersonalizationCandidateRepository repository;
        readonly IGreetingRecognitionSessionProvider sessionProvider;
        readonly IPersonalLivePracticeRecognitionAdapter captureAdapter;

        GreetingTransferState state;
        IGreetingRecognitionSession frozenSession;
        int sessionAcquireCount;
        int completionCount;
        int candidateSaveCount;
        GuidedQuestOutcomeKind? lastOutcomeKind;
        string lastFeedbackCopy = string.Empty;

        public GreetingTransferState State => state;
        public string ExpectedWord => "GREETING";
        public long FrozenGeneration => frozenSession == null ? -1 : frozenSession.Generation;
        public int SessionAcquireCount => sessionAcquireCount;
        public int CandidateSaveCount => candidateSaveCount;
        public int CompletionCount => completionCount;
        public bool IsCompleted => state == GreetingTransferState.Completed;
        public GuidedQuestOutcomeKind? LastOutcomeKind => lastOutcomeKind;
        public string LastFeedbackCopy => lastFeedbackCopy;

        public string IntroductionCopy =>
            "\uAD50\uC721 \uB2F4\uB2F9\uC790\uC5D0\uAC8C GREETING\uC73C\uB85C \uC778\uC0AC\uD574 \uBCF4\uC138\uC694.";
        public string GateFailedCopy =>
            "\uC138 \uB2E8\uC5B4 \uAC1C\uC778 \uD45C\uBCF8 \uB4F1\uB85D\uC744 \uBA3C\uC800 \uC644\uB8CC\uD558\uACE0 \uB2E4\uC2DC \uC624\uC138\uC694.";
        public string ActiveViewFailedCopy =>
            "\uAC1C\uC778 \uD45C\uBCF8 \uAC80\uC99D\uC5D0 \uC2E4\uD328\uD588\uC2B5\uB2C8\uB2E4. \uB2E4\uC2DC \uC2DC\uB3C4\uD574 \uC8FC\uC138\uC694.";
        public string ReadyCopy =>
            "\uC900\uBE44 \uC644\uB8CC. \uC778\uC0AC \uD45C\uD604\uC744 \uC2DC\uC791\uD558\uC138\uC694.";
        public string CapturingCopy =>
            "\uD45C\uD604 \uC778\uC2DD \uC911...";
        public string AwaitingRetryCopy =>
            "\uB2E4\uC2DC \uC2DC\uB3C4\uD560 \uC218 \uC788\uC2B5\uB2C8\uB2E4.";
        public string CompletedCopy =>
            "\uCCAB NPC \uC218\uC5B4 Quest\uB97C \uC644\uB8CC\uD588\uC2B5\uB2C8\uB2E4!";

        public GreetingContextualTransferService(
            ITrustedPersonalizationCandidateRepository repository,
            IGreetingRecognitionSessionProvider sessionProvider,
            IPersonalLivePracticeRecognitionAdapter captureAdapter)
        {
            this.repository = repository ??
                throw new ArgumentNullException(nameof(repository));
            this.sessionProvider = sessionProvider ??
                throw new ArgumentNullException(nameof(sessionProvider));
            this.captureAdapter = captureAdapter ??
                throw new ArgumentNullException(nameof(captureAdapter));
        }

        public bool TryStart()
        {
            if (state != GreetingTransferState.NotStarted) return false;

            TrustedCandidateInventory inventory = repository.ReloadTrustedAndVerify();
            if (inventory == null || !inventory.OverallEnrollmentGateReady)
            {
                state = GreetingTransferState.GateCheckFailed;
                lastFeedbackCopy = GateFailedCopy;
                return false;
            }

            if (!sessionProvider.TryRefresh())
            {
                state = GreetingTransferState.ActiveViewFailed;
                lastFeedbackCopy = ActiveViewFailedCopy;
                return false;
            }

            frozenSession = sessionProvider.AcquireSession();
            sessionAcquireCount++;
            captureAdapter.SetExpectedWord("GREETING");
            state = GreetingTransferState.Ready;
            lastFeedbackCopy = ReadyCopy;
            return true;
        }

        public bool TryBeginCapture()
        {
            if (state != GreetingTransferState.Ready) return false;
            if (!captureAdapter.BeginCapture()) return false;
            state = GreetingTransferState.Capturing;
            lastFeedbackCopy = CapturingCopy;
            return true;
        }

        public bool TryFinalizeCapture()
        {
            if (state != GreetingTransferState.Capturing) return false;

            PracticeRecognitionAttempt attempt = captureAdapter.FinalizeCapture();
            if (attempt == null || attempt.NormalizedSequence == null)
            {
                state = GreetingTransferState.AwaitingRetry;
                lastOutcomeKind = GuidedQuestOutcomeKind.TechnicalFailure;
                lastFeedbackCopy =
                    "\uCE74\uBA54\uB77C/\uCD94\uC801 \uBB38\uC81C\uB85C \uCC44\uC810\uD558\uC9C0 \uC54A\uC558\uC2B5\uB2C8\uB2E4. \uB2E4\uC2DC \uC900\uBE44\uD574 \uC8FC\uC138\uC694.";
                return true;
            }

            TrustedHybridOutcomeKind hybridKind = frozenSession.RecognizeGreeting(
                attempt.CaptureSessionId, attempt.NormalizedSequence);

            GuidedQuestOutcomeKind outcomeKind = MapHybridToGuided(hybridKind);
            lastOutcomeKind = outcomeKind;

            switch (outcomeKind)
            {
                case GuidedQuestOutcomeKind.StrictSuccess:
                    state = GreetingTransferState.Completed;
                    completionCount++;
                    lastFeedbackCopy =
                        "\uC778\uC0AC\uB97C \uC815\uD655\uD788 \uC54C\uC544\uBD24\uC2B5\uB2C8\uB2E4!";
                    return true;

                case GuidedQuestOutcomeKind.LowConfidenceCorrect:
                    state = GreetingTransferState.Completed;
                    completionCount++;
                    lastFeedbackCopy =
                        "\uC778\uC0AC\uB97C \uC54C\uC544\uBD24\uC2B5\uB2C8\uB2E4. \uC870\uAE08 \uB354 \uB610\uB837\uD558\uAC8C \uD45C\uD604\uD558\uBA74 \uC88B\uACA0\uC2B5\uB2C8\uB2E4.";
                    return true;

                case GuidedQuestOutcomeKind.WrongWord:
                    state = GreetingTransferState.AwaitingRetry;
                    lastFeedbackCopy =
                        "\uB2E4\uB978 \uD45C\uD604\uC73C\uB85C \uC778\uC2DD\uB418\uC5C8\uC2B5\uB2C8\uB2E4. \uB2E4\uC2DC \uC2DC\uB3C4\uD574 \uC8FC\uC138\uC694.";
                    return true;

                default:
                    state = GreetingTransferState.AwaitingRetry;
                    lastFeedbackCopy =
                        "\uCE74\uBA54\uB77C/\uCD94\uC801 \uBB38\uC81C\uB85C \uCC44\uC810\uD558\uC9C0 \uC54A\uC558\uC2B5\uB2C8\uB2E4. \uB2E4\uC2DC \uC900\uBE44\uD574 \uC8FC\uC138\uC694.";
                    return true;
            }
        }

        public bool TryRetry()
        {
            if (state != GreetingTransferState.AwaitingRetry) return false;
            captureAdapter.ResetCapture();
            state = GreetingTransferState.Ready;
            lastFeedbackCopy = ReadyCopy;
            return true;
        }

        public bool TryCompleteQuest()
        {
            if (state != GreetingTransferState.Completed) return false;
            return true;
        }

        static GuidedQuestOutcomeKind MapHybridToGuided(TrustedHybridOutcomeKind kind)
        {
            switch (kind)
            {
                case TrustedHybridOutcomeKind.StrictSuccess:
                    return GuidedQuestOutcomeKind.StrictSuccess;
                case TrustedHybridOutcomeKind.LowConfidenceCorrect:
                    return GuidedQuestOutcomeKind.LowConfidenceCorrect;
                case TrustedHybridOutcomeKind.WrongWord:
                    return GuidedQuestOutcomeKind.WrongWord;
                default:
                    return GuidedQuestOutcomeKind.TechnicalFailure;
            }
        }
    }
}
