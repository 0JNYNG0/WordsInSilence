using System;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Practice.PetRobotWorldIntegration;

namespace WordsInSilence.Practice.PetRobotPortfolioQuest
{
    public enum PetRobotPortfolioQuestState
    {
        AwaitingPractice,
        AwaitingNpcReport,
        DialogueOpen,
        AwaitingGreetingAttempt,
        DemoCompleted,
    }

    public sealed class PetRobotPortfolioQuestSnapshot
    {
        public PetRobotWordLearningStatus Greeting { get; }
        public PetRobotWordLearningStatus Love { get; }
        public PetRobotWordLearningStatus ThankYou { get; }
        public int MasteredCount { get; }
        public int NeedsPracticeCount { get; }
        public int CandidateSavedTotal { get; }

        PetRobotPortfolioQuestSnapshot(PetRobotPracticeCompletionSnapshot source)
        {
            Greeting = source.Greeting;
            Love = source.Love;
            ThankYou = source.ThankYou;
            MasteredCount = source.MasteredCount;
            NeedsPracticeCount = source.NeedsPracticeCount;
            CandidateSavedTotal = source.CandidateSavedTotal;
        }

        public static PetRobotPortfolioQuestSnapshot FromCompleted(
            PetRobotPracticeCompletionSnapshot source)
        {
            if (source == null || !source.SessionCompleted)
                throw new ArgumentException("a completed practice snapshot is required", nameof(source));
            return new PetRobotPortfolioQuestSnapshot(source);
        }

        public PetRobotWordLearningStatus GetWordStatus(string word)
        {
            if (string.Equals(word, "GREETING", StringComparison.Ordinal)) return Greeting;
            if (string.Equals(word, "LOVE", StringComparison.Ordinal)) return Love;
            if (string.Equals(word, "THANK_YOU", StringComparison.Ordinal)) return ThankYou;
            throw new ArgumentException("unsupported practice word", nameof(word));
        }

        public string ResultHeadline => NeedsPracticeCount == 0
            ? "연습 결과를 확인했습니다. 세 수어를 모두 익혔습니다."
            : "연습은 완료했습니다. " + NeedsPracticeCount + "개 수어는 복습이 필요합니다.";

        public string ResultFollowUp => NeedsPracticeCount == 0
            ? "다음 임무를 시작할 준비가 되었습니다."
            : "단어별 결과를 확인하고 필요한 표현을 다시 연습해 주세요.";
    }

    public sealed class PetRobotPortfolioQuestFlow
    {
        PetRobotPortfolioQuestSnapshot snapshot;

        public PetRobotPortfolioQuestState State { get; private set; } =
            PetRobotPortfolioQuestState.AwaitingPractice;
        public PetRobotPortfolioQuestSnapshot Snapshot => snapshot;
        public bool NpcInRange { get; private set; }
        public int CompletionFreezeCount { get; private set; }
        public int DialogueOpenCount { get; private set; }
        public int DemoCompletionCount { get; private set; }
        public bool CanEnterPractice => State == PetRobotPortfolioQuestState.AwaitingPractice;
        public bool CanReportToNpc => State == PetRobotPortfolioQuestState.AwaitingNpcReport &&
            snapshot != null && NpcInRange;

        public string Objective
        {
            get
            {
                switch (State)
                {
                    case PetRobotPortfolioQuestState.AwaitingPractice:
                        return "동행 애완 로봇과 대화해 수어 연습을 시작하세요 (E / Enter / Space)";
                    case PetRobotPortfolioQuestState.AwaitingNpcReport:
                        return "교육 담당자에게 연습 결과를 보고하세요";
                    case PetRobotPortfolioQuestState.DialogueOpen:
                        return "교육 담당자와 결과를 확인하세요";
                    case PetRobotPortfolioQuestState.AwaitingGreetingAttempt:
                        return "교육 담당자에게 GREETING으로 인사해 보세요";
                    default:
                        return "데모 완료";
                }
            }
        }

        public bool ObserveCompleted(PetRobotPracticeCompletionSnapshot completion)
        {
            if (State != PetRobotPortfolioQuestState.AwaitingPractice || snapshot != null ||
                completion == null || !completion.SessionCompleted) return false;
            snapshot = PetRobotPortfolioQuestSnapshot.FromCompleted(completion);
            CompletionFreezeCount++;
            State = PetRobotPortfolioQuestState.AwaitingNpcReport;
            return true;
        }

        public bool ObserveIncompleteSession() => false;

        public bool ObserveTechnicalFailure() => false;

        public void SetNpcInRange(bool value)
        {
            NpcInRange = value;
        }

        public bool TryOpenDialogue(bool interactionPressed)
        {
            if (!interactionPressed || !CanReportToNpc) return false;
            State = PetRobotPortfolioQuestState.DialogueOpen;
            DialogueOpenCount++;
            return true;
        }

        public bool TryCompleteDialogue(bool explicitConfirmation)
        {
            if (!explicitConfirmation || State != PetRobotPortfolioQuestState.DialogueOpen ||
                snapshot == null) return false;
            State = PetRobotPortfolioQuestState.AwaitingGreetingAttempt;
            return true;
        }

        public bool TryCompleteGreetingQuest()
        {
            if (State != PetRobotPortfolioQuestState.AwaitingGreetingAttempt) return false;
            State = PetRobotPortfolioQuestState.DemoCompleted;
            DemoCompletionCount++;
            return true;
        }

        public bool RollbackToDialogueOpen()
        {
            if (State != PetRobotPortfolioQuestState.AwaitingGreetingAttempt) return false;
            LastQuestResult = null;
            State = PetRobotPortfolioQuestState.DialogueOpen;
            return true;
        }

        public bool RestoreAwaitingGreetingAttempt(PetRobotPracticeCompletionSnapshot priorCompletion)
        {
            if (priorCompletion == null || !priorCompletion.SessionCompleted) return false;
            if (snapshot != null && State != PetRobotPortfolioQuestState.AwaitingPractice) return false;
            if (snapshot == null)
                snapshot = PetRobotPortfolioQuestSnapshot.FromCompleted(priorCompletion);
            State = PetRobotPortfolioQuestState.AwaitingGreetingAttempt;
            return true;
        }

        public PetRobotGreetingQuestResult LastQuestResult { get; private set; }
        public string QuestFeedbackText => LastQuestResult == null ? string.Empty : LastQuestResult.FeedbackText;

        public bool ObserveQuestResult(PetRobotGreetingQuestResult result)
        {
            if (State != PetRobotPortfolioQuestState.AwaitingGreetingAttempt ||
                result == null || LastQuestResult != null) return false;
            LastQuestResult = result;
            return true;
        }
    }
}

