using System;
using System.Collections.Generic;
using WordsInSilence.Practice.FullHandEnrollment;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    public enum MoaStoryBeat
    {
        Welcome,
        LessonOverview,
        HowToUse,
        WordIntroduction,
        Practice,
        Feedback,
        Completed,
    }

    public enum MoaStoryAction
    {
        None,
        AdvanceStory,
        BeginAutoAttempt,
        RetryPreCapture,
        RetryTechnicalFailure,
        /// <summary>F27: retry after a valid full-hand abstention.</summary>
        RetryNotDistinguished,
        ReviewWrongWord,
        RetryEnrollment,
        ContinueToNextWord,
        ContinueWithNeedsPractice,
        /// <summary>F28: end an unfinished record round honestly, collecting nothing.</summary>
        LeaveRoundIncomplete,
        ExitPractice,
    }

    public enum MoaStoryFeedbackKind
    {
        None,
        StrictSuccess,
        LowConfidenceCorrect,
        WrongWord,
        TechnicalFailure,

        /// <summary>F27: the model ran and validly abstained. Not a wrong word.</summary>
        NotDistinguishedRetry,

        /// <summary>F27: the prompted word is outside the model vocabulary.</summary>
        UnsupportedWord,
    }

    public readonly struct MoaStoryPresentation
    {
        public MoaStoryBeat Beat { get; }
        public string BodyText { get; }
        public string ActionLabel { get; }
        public MoaStoryAction Action { get; }
        public bool ActionVisible { get; }
        public bool ActionEnabled { get; }

        public MoaStoryPresentation(
            MoaStoryBeat beat,
            string bodyText,
            string actionLabel,
            MoaStoryAction action,
            bool actionVisible,
            bool actionEnabled)
        {
            Beat = beat;
            BodyText = bodyText ?? string.Empty;
            ActionLabel = actionLabel ?? string.Empty;
            Action = action;
            ActionVisible = actionVisible;
            ActionEnabled = actionEnabled;
        }
    }

    /// <summary>
    /// Pure presentation state for MOA's guided-practice story. It never owns or invokes the
    /// authoritative recognition/session actions; the presentation controller dispatches the
    /// returned action through the existing Step 10A handlers after an explicit user click.
    /// </summary>
    public sealed class MoaStoryGuidedPracticeFlow
    {
        public const string WelcomeCopy =
            "어서 와! 나는 MOA야. 이 연습 공간에서 네 수어를 함께 맞춰볼게.";
        public const string LessonOverviewCopy =
            "오늘은 ‘인사’, ‘사랑’, ‘감사’ 세 표현을 함께 연습해 보자.";
        public const string HowToUseCopy =
            "시범을 본 뒤 직접 해보겠다고 선택하면 돼. 준비 자세에서 손을 올리면 자동으로 시작하고, 다시 편하게 내리면 자동으로 끝낼게.";
        public const string GreetingIntroductionCopy =
            "첫 번째는 ‘인사’야. 손과 팔의 흐름을 보고 천천히 따라 해보자.";
        public const string LoveIntroductionCopy =
            "두 번째는 ‘사랑’이야. 손 모양과 움직임이 이어지는 흐름을 기억해보자.";
        public const string ThankYouIntroductionCopy =
            "마지막은 ‘감사’야. 시범을 본 뒤 네 동작으로 표현해봐.";
        public const string ReadyCopy =
            "편하게 팔을 내리고 준비 자세를 잠깐 유지해줘.";
        public const string CalibrationCopy =
            "먼저 네 기본 자세를 기억할게. 팔을 편하게 아래로 내리고 잠깐 그대로 있어줘.";
        public const string ArmedCopy =
            "좋아, 준비됐어. 이제 수어를 시작해줘.";
        public const string CapturingCopy =
            "좋아. 끝까지 표현한 뒤 다시 편하게 팔을 내려줘.";
        public const string EndCandidateCopy =
            "좋아, 그대로 잠깐만 유지해줘.";
        public const string EvaluatingCopy =
            "확인하고 있어…";
        public const string PreCaptureFailureCopy =
            "카메라 추적이 잠깐 불안정했어. 다시 준비해보자.";
        public const string AttemptUnavailableCopy =
            "아직 카메라 준비가 끝나지 않았어. 준비 상태를 확인하고 다시 눌러줘.";
        public const string StrictSuccessCopy =
            "잘했어! 수어를 정확히 인식했어.";
        public const string LowConfidenceCorrectCopy =
            "뜻은 맞게 전달됐어. 조금 더 또렷하게 하면 더 안정적으로 알아볼 수 있어.";
        public const string WrongWordCopy =
            "다른 표현으로 인식했어. 시범을 한 번 더 보고 다시 해보자.";
        public const string TechnicalFailureCopy =
            "네 수어가 틀렸다는 뜻이 아니야. 카메라나 추적이 잠깐 불안정했어.";
        /// <summary>F27: a valid abstention. Not a wrong word and not a camera fault.</summary>
        public const string NotDistinguishedCopy =
            "동작을 확실히 구별하지 못했어요. 다시 시도해 주세요";
        /// <summary>F27: honest about the words the model actually knows today.</summary>
        public const string UnsupportedWordCopy =
            "이 표현은 아직 인식 모델이 배우지 않았어. 지금은 ‘인사’와 ‘고마워’만 확인할 수 있어.";
        /// <summary>F27: recognition succeeded; it certifies no sign correctness.</summary>
        public const string FullHandSuccessCopy =
            "잘했어! 이번 동작을 지금 배우는 표현으로 알아봤어.";
        /// <summary>F27: brief, because full-hand enrollment is still unfinished.</summary>
        public const string FullHandEnrollmentPendingCopy =
            "지금은 이 결과를 네 개인 학습 데이터로 저장하지는 않아.";
        public const string NeedsPracticeCopy =
            "이번 표현은 연습이 조금 더 필요해. 결과는 그대로 두고 다음 표현으로 이어갈 수 있어.";
        public const string CompletedCopy =
            "좋아, 세 표현을 모두 연습했어. 이제 밖에서 직접 써볼 준비가 됐어.";
        public const string PersonalizationIncompleteCopy =
            "이번 연습은 여기까지 할게. 검증된 개인 표본이 아직 3개보다 적어서 개인화 등록은 미완료야. 다음 표현을 연습하거나 나중에 다시 이어갈 수 있어.";
        public const string TechnicalStopCopy =
            "기술 오류가 연속으로 발생해서 이번 연습은 여기서 멈출게. 카메라와 자세를 확인한 뒤 나중에 다시 이어가자. 개인화 등록은 아직 완료되지 않았어.";
        public const string EnrollmentGateLockedCopy =
            "이번 연습 라운드는 끝났지만 세 표현 모두 검증 표본 3개를 모아야 개인화 등록이 완료돼. 부족한 표현은 나중에 다시 이어서 연습할 수 있어.";

        static readonly string[] legacyWords = { "GREETING", "LOVE", "THANK_YOU" };

        /// <summary>
        /// F28. The words THIS round practises. Off-mode it is exactly the legacy three; in
        /// full-hand mode the presentation controller pins it to the CURRENT model vocabulary.
        /// </summary>
        string[] words = legacyWords;

        string currentWord = legacyWords[0];
        PetRobotPracticeState sessionState = PetRobotPracticeState.Ready;
        MoaStoryFeedbackKind feedbackKind;
        bool onboardingComplete;
        bool wordIntroductionAccepted;
        bool canContinueWithNeedsPractice;
        bool cameraReady;
        bool autoAttemptAvailable;
        bool attemptRequestRejected;
        bool sessionExited;
        bool trustedEnrollment;
        bool overallEnrollmentGateReady;
        /// <summary>F27: the last attempt was decided by the full-hand authority.</summary>
        bool fullHandMode;
        int trustedSampleCount;
        /// <summary>F28: the authoritative record-collection view of the current word.</summary>
        FullHandRecordCollectionView fullHandRecords;
        int scoreableAttemptCount;
        int consecutiveTechnicalFailureCount;
        PetRobotPersonalizationEnrollmentState enrollmentState;
        RawMediaPipeNeutralAutoSegmenterState autoState =
            RawMediaPipeNeutralAutoSegmenterState.Inactive;

        public MoaStoryBeat Beat { get; private set; } = MoaStoryBeat.Welcome;
        public string CurrentWord => currentWord;
        public IReadOnlyList<string> Words => Array.AsReadOnly((string[])words.Clone());

        /// <summary>
        /// F28. Pins the words of THIS round. Every word must have an introduction this build
        /// can actually present: an unknown future model word is REFUSED here instead of being
        /// silently practised, skipped or reported complete.
        /// </summary>
        public void ConfigureVocabulary(IReadOnlyList<string> roundWords)
        {
            if (roundWords == null || roundWords.Count == 0)
                throw new ArgumentException("a non-empty round vocabulary is required",
                                            nameof(roundWords));
            var copy = new string[roundWords.Count];
            for (int index = 0; index < copy.Length; index++)
            {
                copy[index] = roundWords[index];
                if (!HasPresentationFor(copy[index]))
                    throw new ArgumentException(
                        "this build has no introduction for a requested practice word",
                        nameof(roundWords));
            }
            words = copy;
            currentWord = copy[0];
        }

        /// <summary>True only for a word this build can introduce and demonstrate.</summary>
        public static bool HasPresentationFor(string word)
        {
            for (int index = 0; index < legacyWords.Length; index++)
                if (string.Equals(legacyWords[index], word, StringComparison.Ordinal)) return true;
            return false;
        }
        public bool ExpressionStartUnlocked =>
            onboardingComplete && wordIntroductionAccepted &&
            Beat == MoaStoryBeat.Practice &&
            sessionState == PetRobotPracticeState.Ready && cameraReady;
        public MoaStoryPresentation Current => BuildPresentation();

        public void Reset(string firstWord)
        {
            currentWord = SupportedWord(firstWord);
            sessionState = PetRobotPracticeState.Ready;
            feedbackKind = MoaStoryFeedbackKind.None;
            onboardingComplete = false;
            wordIntroductionAccepted = false;
            canContinueWithNeedsPractice = false;
            cameraReady = false;
            autoAttemptAvailable = false;
            attemptRequestRejected = false;
            sessionExited = false;
            trustedEnrollment = false;
            overallEnrollmentGateReady = false;
            fullHandMode = false;
            trustedSampleCount = 0;
            fullHandRecords = default(FullHandRecordCollectionView);
            scoreableAttemptCount = 0;
            consecutiveTechnicalFailureCount = 0;
            enrollmentState = PetRobotPersonalizationEnrollmentState.Pending;
            autoState = RawMediaPipeNeutralAutoSegmenterState.Inactive;
            Beat = MoaStoryBeat.Welcome;
        }

        public void Synchronize(
            string word,
            PetRobotPracticeState authoritativeState,
            MoaStoryFeedbackKind authoritativeFeedback,
            bool authoritativeCanContinueWithNeedsPractice,
            bool authoritativeCameraReady)
        {
            Synchronize(
                word,
                authoritativeState,
                authoritativeFeedback,
                authoritativeCanContinueWithNeedsPractice,
                authoritativeCameraReady,
                RawMediaPipeNeutralAutoSegmenterState.Inactive,
                true);
        }

        public void Synchronize(
            string word,
            PetRobotPracticeState authoritativeState,
            MoaStoryFeedbackKind authoritativeFeedback,
            bool authoritativeCanContinueWithNeedsPractice,
            bool authoritativeCameraReady,
            RawMediaPipeNeutralAutoSegmenterState authoritativeAutoState,
            bool authoritativeAutoAttemptAvailable)
        {
            Synchronize(
                word,
                authoritativeState,
                authoritativeFeedback,
                authoritativeCanContinueWithNeedsPractice,
                authoritativeCameraReady,
                authoritativeAutoState,
                authoritativeAutoAttemptAvailable,
                false,
                PetRobotPersonalizationEnrollmentState.Pending,
                0,
                0,
                0,
                false);
        }

        public void Synchronize(
            string word,
            PetRobotPracticeState authoritativeState,
            MoaStoryFeedbackKind authoritativeFeedback,
            bool authoritativeCanContinueWithNeedsPractice,
            bool authoritativeCameraReady,
            RawMediaPipeNeutralAutoSegmenterState authoritativeAutoState,
            bool authoritativeAutoAttemptAvailable,
            bool authoritativeTrustedEnrollment,
            PetRobotPersonalizationEnrollmentState authoritativeEnrollmentState,
            int authoritativeTrustedSampleCount,
            int authoritativeScoreableAttemptCount,
            int authoritativeConsecutiveTechnicalFailureCount,
            bool authoritativeOverallEnrollmentGateReady,
            bool authoritativeFullHandMode = false,
            FullHandRecordCollectionView authoritativeFullHandRecords =
                default(FullHandRecordCollectionView))
        {
            // F27: true only while the last attempt was decided by the full-hand authority.
            fullHandMode = authoritativeFullHandMode;
            // F28: the REAL record-collection view, never a guess and never a fixed literal.
            fullHandRecords = authoritativeFullHandRecords;
            string resolvedWord = SupportedWord(word);
            bool wordChanged = !string.Equals(currentWord, resolvedWord, StringComparison.Ordinal);
            currentWord = resolvedWord;
            sessionState = authoritativeState;
            feedbackKind = authoritativeFeedback;
            canContinueWithNeedsPractice = authoritativeCanContinueWithNeedsPractice;
            cameraReady = authoritativeCameraReady;
            autoState = authoritativeAutoState;
            autoAttemptAvailable = authoritativeAutoAttemptAvailable;
            sessionExited = authoritativeState == PetRobotPracticeState.Exited;
            trustedEnrollment = authoritativeTrustedEnrollment;
            enrollmentState = authoritativeEnrollmentState;
            trustedSampleCount = Math.Max(0, authoritativeTrustedSampleCount);
            scoreableAttemptCount = Math.Max(0, authoritativeScoreableAttemptCount);
            consecutiveTechnicalFailureCount = Math.Max(0, authoritativeConsecutiveTechnicalFailureCount);
            overallEnrollmentGateReady = authoritativeOverallEnrollmentGateReady;

            if (authoritativeState == PetRobotPracticeState.Completed || sessionExited)
            {
                Beat = MoaStoryBeat.Completed;
                return;
            }

            if (!onboardingComplete)
                return;

            if (wordChanged)
            {
                wordIntroductionAccepted = false;
                attemptRequestRejected = false;
                Beat = MoaStoryBeat.WordIntroduction;
                return;
            }

            if (!wordIntroductionAccepted)
            {
                Beat = MoaStoryBeat.WordIntroduction;
                return;
            }

            switch (authoritativeState)
            {
                case PetRobotPracticeState.Capturing:
                case PetRobotPracticeState.Ready:
                    Beat = MoaStoryBeat.Practice;
                    break;
                case PetRobotPracticeState.Feedback:
                    Beat = MoaStoryBeat.Feedback;
                    break;
            }
        }

        public bool AdvanceStory()
        {
            switch (Beat)
            {
                case MoaStoryBeat.Welcome:
                    Beat = MoaStoryBeat.LessonOverview;
                    return true;
                case MoaStoryBeat.LessonOverview:
                    Beat = MoaStoryBeat.HowToUse;
                    return true;
                case MoaStoryBeat.HowToUse:
                    onboardingComplete = true;
                    wordIntroductionAccepted = false;
                    Beat = MoaStoryBeat.WordIntroduction;
                    return true;
                case MoaStoryBeat.WordIntroduction:
                    return ConfirmAutoAttemptAccepted();
                default:
                    return false;
            }
        }

        public bool ConfirmAutoAttemptAccepted()
        {
            if (Beat != MoaStoryBeat.WordIntroduction ||
                sessionState != PetRobotPracticeState.Ready || !cameraReady ||
                !autoAttemptAvailable)
                return false;
            wordIntroductionAccepted = true;
            attemptRequestRejected = false;
            feedbackKind = MoaStoryFeedbackKind.None;
            Beat = MoaStoryBeat.Practice;
            return true;
        }

        public void ReportAutoAttemptRejected()
        {
            attemptRequestRejected = true;
        }

        public bool ReturnToWordIntroductionForReview()
        {
            if (sessionState != PetRobotPracticeState.Ready || sessionExited) return false;
            wordIntroductionAccepted = false;
            attemptRequestRejected = false;
            feedbackKind = MoaStoryFeedbackKind.None;
            Beat = MoaStoryBeat.WordIntroduction;
            return true;
        }

        MoaStoryPresentation BuildPresentation()
        {
            switch (Beat)
            {
                case MoaStoryBeat.Welcome:
                    return Action(WelcomeCopy, "다음", MoaStoryAction.AdvanceStory, true);
                case MoaStoryBeat.LessonOverview:
                    return Action(LessonOverviewText(), "다음", MoaStoryAction.AdvanceStory, true);
                case MoaStoryBeat.HowToUse:
                    return Action(HowToUseCopy, "첫 표현 보기", MoaStoryAction.AdvanceStory, true);
                case MoaStoryBeat.WordIntroduction:
                {
                    bool canEnterPractice =
                        sessionState == PetRobotPracticeState.Ready && cameraReady &&
                        autoAttemptAvailable;
                    return Action(
                        attemptRequestRejected ? AttemptUnavailableCopy : IntroductionFor(currentWord),
                        "직접 해볼게",
                        MoaStoryAction.BeginAutoAttempt,
                        canEnterPractice);
                }
                case MoaStoryBeat.Practice:
                    return PracticePresentation();
                case MoaStoryBeat.Feedback:
                    return FeedbackPresentation();
                case MoaStoryBeat.Completed:
                    string completedCopy = fullHandRecords.Active
                        ? (fullHandRecords.AllCollected
                            ? FullHandEnrollmentCopy.RoundCollected + " " +
                              FullHandEnrollmentCopy.CollectionNote
                            : FullHandEnrollmentCopy.RoundIncomplete(
                                fullHandRecords.CollectedWordCount, fullHandRecords.WordCount))
                        : trustedEnrollment && !overallEnrollmentGateReady
                        ? EnrollmentGateLockedCopy
                        : CompletedCopy;
                    return sessionExited
                        ? Passive(completedCopy)
                        : Action(completedCopy, "연습 공간 나가기", MoaStoryAction.ExitPractice, true);
                default:
                    return Passive(string.Empty);
            }
        }

        MoaStoryPresentation PracticePresentation()
        {
            if (sessionState == PetRobotPracticeState.Capturing &&
                autoState == RawMediaPipeNeutralAutoSegmenterState.Inactive)
                return Passive(CapturingCopy);

            switch (autoState)
            {
                case RawMediaPipeNeutralAutoSegmenterState.NeedsCalibration:
                case RawMediaPipeNeutralAutoSegmenterState.CalibratingNeutral:
                    return Passive(CalibrationCopy);
                case RawMediaPipeNeutralAutoSegmenterState.WaitingForNeutral:
                    return Passive(ReadyCopy);
                case RawMediaPipeNeutralAutoSegmenterState.Armed:
                    return Passive(ArmedCopy);
                case RawMediaPipeNeutralAutoSegmenterState.Capturing:
                    return Passive(CapturingCopy);
                case RawMediaPipeNeutralAutoSegmenterState.EndCandidate:
                    return Passive(EndCandidateCopy);
                case RawMediaPipeNeutralAutoSegmenterState.Evaluating:
                    return Passive(EvaluatingCopy);
                case RawMediaPipeNeutralAutoSegmenterState.Failed:
                    if (sessionState == PetRobotPracticeState.Ready && feedbackKind == MoaStoryFeedbackKind.None)
                        return Action(
                            PreCaptureFailureCopy,
                            "다시 준비할게",
                            MoaStoryAction.RetryPreCapture,
                            cameraReady && autoAttemptAvailable);
                    return Passive(TechnicalFailureCopy);
                default:
                    return Passive(attemptRequestRejected ? AttemptUnavailableCopy : ReadyCopy);
            }
        }

        MoaStoryPresentation FeedbackPresentation()
        {
            // F29. The spent attempt budget is resolved FIRST. Every other full-hand branch
            // below would offer a retry the session now refuses, so the learner would be stuck
            // on a dead button. This says what was really collected and offers the existing
            // honest end-of-round action; it marks nothing complete.
            if (fullHandRecords.Active && fullHandRecords.AttemptCapReached)
                return Action(
                    fullHandRecords.MovesToNextWord
                        ? FullHandEnrollmentCopy.AttemptCapReachedMovingOn(
                            fullHandRecords.Collected, fullHandRecords.Required,
                            fullHandRecords.Attempts, fullHandRecords.MaxAttempts)
                        : FullHandEnrollmentCopy.AttemptCapReached(
                            fullHandRecords.Collected, fullHandRecords.Required,
                            fullHandRecords.Attempts, fullHandRecords.MaxAttempts),
                    LeaveRoundLabel(),
                    MoaStoryAction.LeaveRoundIncomplete,
                    fullHandRecords.CanLeaveIncomplete);

            // F29. The remaining STOPPED state of a record round, resolved after the cap guard
            // above and before every feedback-kind branch, so a stopped round never offers a
            // retry the session refuses. It claims no success and marks nothing collected.
            if (fullHandRecords.Active && fullHandRecords.CanLeaveIncomplete)
                return Action(TechnicalStopCopyForLeave(),
                              LeaveRoundLabel(),
                              MoaStoryAction.LeaveRoundIncomplete, true);

            // F27. These verdicts come from the full-hand authority and are resolved first,
            // so no enrollment or needs-practice branch below can relabel them.
            if (feedbackKind == MoaStoryFeedbackKind.NotDistinguishedRetry)
                return Action(NotDistinguishedCopy, "다시 해볼게",
                              MoaStoryAction.RetryNotDistinguished, true);
            if (feedbackKind == MoaStoryFeedbackKind.UnsupportedWord)
                return Action(UnsupportedWordCopy,
                              IsLastWord(currentWord) ? "결과 보기" : "다음 표현",
                              MoaStoryAction.ContinueWithNeedsPractice,
                              canContinueWithNeedsPractice);
            if (fullHandMode)
            {
                switch (feedbackKind)
                {
                    case MoaStoryFeedbackKind.StrictSuccess:
                        if (!fullHandRecords.Active)
                            // F27: no record store on this route, and no personalization claim
                            return Action(
                                FullHandSuccessCopy + " " + FullHandEnrollmentPendingCopy,
                                IsLastWord(currentWord) ? "결과 보기" : "다음 표현",
                                MoaStoryAction.ContinueToNextWord, true);
                        // F28: the records for this word are collected; the round may advance
                        if (fullHandRecords.WordCollected)
                            return Action(
                                FullHandEnrollmentCopy.CollectedForWord(
                                    fullHandRecords.Required) + " " +
                                FullHandEnrollmentCopy.CollectionNote,
                                IsLastWord(currentWord) ? "결과 보기" : "다음 표현",
                                MoaStoryAction.ContinueToNextWord, true);
                        // otherwise the SAME word is practised again, and the copy states what
                        // the store actually did with this attempt
                        if (fullHandRecords.SaveStatus == FullHandEnrollmentSaveStatus.Saved)
                            return Action(
                                FullHandEnrollmentCopy.Saved(fullHandRecords.Collected,
                                                             fullHandRecords.Required) + " " +
                                FullHandEnrollmentCopy.CollectionNote,
                                "한 번 더 연습할게", MoaStoryAction.RetryEnrollment, true);
                        if (fullHandRecords.SaveStatus == FullHandEnrollmentSaveStatus.Duplicate)
                            return Action(
                                FullHandEnrollmentCopy.Duplicate(fullHandRecords.Collected,
                                                                 fullHandRecords.Required),
                                "한 번 더 연습할게", MoaStoryAction.RetryEnrollment, true);
                        // not saved: this must never be shown as progress
                        return Action(
                            FullHandEnrollmentCopy.NotSaved(fullHandRecords.Collected,
                                                            fullHandRecords.Required),
                            "한 번 더 연습할게", MoaStoryAction.RetryEnrollment, true);
                    case MoaStoryFeedbackKind.WrongWord:
                        return Action(WrongWordCopy, "시범 다시 보기",
                                      MoaStoryAction.ReviewWrongWord, true);
                    case MoaStoryFeedbackKind.TechnicalFailure:
                        if (fullHandRecords.Active && fullHandRecords.CanLeaveIncomplete)
                            // an honest end of an unfinished round: nothing is collected and
                            // nothing is marked mastered by this
                            return Action(TechnicalStopCopyForLeave(),
                                          LeaveRoundLabel(),
                                          MoaStoryAction.LeaveRoundIncomplete, true);
                        return Action(TechnicalFailureCopy, "다시 준비할게",
                                      MoaStoryAction.RetryTechnicalFailure, true);
                }
            }

            if (trustedEnrollment)
            {
                if (enrollmentState == PetRobotPersonalizationEnrollmentState.PersonalizationIncomplete)
                    return Action(
                        PersonalizationIncompleteCopy,
                        IsLastWord(currentWord) ? "이번 연습 마치기" : "다음 표현",
                        MoaStoryAction.ContinueWithNeedsPractice,
                        true);
                if (enrollmentState == PetRobotPersonalizationEnrollmentState.TechnicalStop)
                    return Action(
                        TechnicalStopCopy,
                        IsLastWord(currentWord) ? "이번 연습 마치기" : "다음 표현",
                        MoaStoryAction.ContinueWithNeedsPractice,
                        true);
                if (enrollmentState == PetRobotPersonalizationEnrollmentState.Pending)
                {
                    string progress = " 검증된 개인 표본 " + trustedSampleCount + "/3 · 연습 시도 " +
                        scoreableAttemptCount + "/10";
                    if (feedbackKind == MoaStoryFeedbackKind.TechnicalFailure)
                        return Action(
                            TechnicalFailureCopy + " 기술 오류 " + consecutiveTechnicalFailureCount + "/3",
                            "다시 준비할게",
                            MoaStoryAction.RetryTechnicalFailure,
                            true);
                    if (feedbackKind == MoaStoryFeedbackKind.WrongWord)
                        return Action(WrongWordCopy + progress, "시범 다시 보기", MoaStoryAction.ReviewWrongWord, true);
                    if (feedbackKind == MoaStoryFeedbackKind.LowConfidenceCorrect)
                        return Action(
                            LowConfidenceCorrectCopy + " 이번 결과는 검증 표본에는 포함하지 않아." + progress,
                            "한 번 더 연습할게",
                            MoaStoryAction.RetryEnrollment,
                            true);
                    if (feedbackKind == MoaStoryFeedbackKind.StrictSuccess)
                        return Action(
                            StrictSuccessCopy + progress + " 표본 3개가 될 때까지 한 번 더 연습해보자.",
                            "한 번 더 연습할게",
                            MoaStoryAction.RetryEnrollment,
                            true);
                }
            }

            if (canContinueWithNeedsPractice)
                return Action(
                    NeedsPracticeCopy,
                    IsLastWord(currentWord) ? "결과 보기" : "다음 표현",
                    MoaStoryAction.ContinueWithNeedsPractice,
                    true);

            switch (feedbackKind)
            {
                case MoaStoryFeedbackKind.StrictSuccess:
                    return Action(
                        StrictSuccessCopy,
                        IsLastWord(currentWord) ? "결과 보기" : "다음 표현",
                        MoaStoryAction.ContinueToNextWord,
                        true);
                case MoaStoryFeedbackKind.LowConfidenceCorrect:
                    return Action(
                        LowConfidenceCorrectCopy,
                        IsLastWord(currentWord) ? "결과 보기" : "다음 표현",
                        MoaStoryAction.ContinueToNextWord,
                        true);
                case MoaStoryFeedbackKind.WrongWord:
                    return Action(
                        WrongWordCopy,
                        "시범 다시 보기",
                        MoaStoryAction.ReviewWrongWord,
                        true);
                case MoaStoryFeedbackKind.TechnicalFailure:
                    return Action(
                        TechnicalFailureCopy,
                        "다시 준비할게",
                        MoaStoryAction.RetryTechnicalFailure,
                        true);
                default:
                    return Passive(ReadyCopy);
            }
        }

        /// <summary>
        /// F32. The label must name what the button actually does: the same action either
        /// moves on to the next expression of this visit or ends the round. It never claims
        /// the practice is finished while another word is still coming.
        /// </summary>
        string LeaveRoundLabel() =>
            fullHandRecords.MovesToNextWord ? "다음 표현으로" : "이번 라운드 마무리";

        string TechnicalStopCopyForLeave() =>
            fullHandRecords.MovesToNextWord
                ? FullHandEnrollmentCopy.TechnicalStopMovingOn
                : FullHandEnrollmentCopy.TechnicalStop;

        MoaStoryPresentation Action(
            string body,
            string label,
            MoaStoryAction action,
            bool enabled)
        {
            return new MoaStoryPresentation(Beat, body, label, action, true, enabled);
        }

        MoaStoryPresentation Passive(string body)
        {
            return new MoaStoryPresentation(
                Beat, body, string.Empty, MoaStoryAction.None, false, false);
        }

        /// <summary>
        /// F28. Names the words THIS round practises. The legacy three-word round returns the
        /// unchanged literal copy.
        /// </summary>
        string LessonOverviewText()
        {
            if (words.Length == legacyWords.Length)
            {
                bool same = true;
                for (int index = 0; index < words.Length && same; index++)
                    if (!string.Equals(words[index], legacyWords[index], StringComparison.Ordinal))
                        same = false;
                if (same) return LessonOverviewCopy;
            }
            var text = new System.Text.StringBuilder("오늘은 ");
            for (int index = 0; index < words.Length; index++)
            {
                if (index > 0) text.Append(", ");
                text.Append('\u2018').Append(KoreanName(words[index])).Append('\u2019');
            }
            return text.Append(" 표현을 함께 연습해 보자.").ToString();
        }

        static string KoreanName(string word)
        {
            switch (word)
            {
                case "LOVE": return "사랑";
                case "THANK_YOU": return "고마워";
                default: return "인사";
            }
        }

        static string IntroductionFor(string word)
        {
            switch (word)
            {
                case "LOVE":
                    return LoveIntroductionCopy;
                case "THANK_YOU":
                    return ThankYouIntroductionCopy;
                default:
                    return GreetingIntroductionCopy;
            }
        }

        string SupportedWord(string word)
        {
            foreach (string supported in words)
                if (string.Equals(word, supported, StringComparison.Ordinal)) return supported;
            throw new ArgumentException("unsupported practice word", nameof(word));
        }

        bool IsLastWord(string word) =>
            words.Length > 0 &&
            string.Equals(words[words.Length - 1], word, StringComparison.Ordinal);
    }
}
