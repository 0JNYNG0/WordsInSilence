using System;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Practice.PetRobotWorldIntegration;

namespace WordsInSilence.Practice.PetRobotPortfolioQuest
{
    public enum InWorldDialogueState
    {
        Idle,
        AwaitingNpcApproach,
        DialogueActive,
        QuestPrompt,
        TrackingPreparation,
        NeutralPreparation,
        ReadyForGreeting,
        Capturing,
        Analyzing,
        RecognitionActive,
        Feedback,
        QuestComplete,
    }

    public enum InWorldDialogueOutcome
    {
        None,
        StrictSuccess,
        LowConfidenceCorrect,
        WrongWord,
        TechnicalFailure,

        /// <summary>F27 full-hand route: a valid abstention. No item, retry offered.</summary>
        NotDistinguished,

        /// <summary>F27 full-hand route: the prompted word is outside the vocabulary.</summary>
        UnsupportedWord,
    }

    public sealed class InWorldDialogueResult
    {
        public string AttemptId { get; }
        public InWorldDialogueOutcome Outcome { get; }
        public string FeedbackText { get; }
        public bool ItemGranted { get; }

        public InWorldDialogueResult(string attemptId, InWorldDialogueOutcome outcome,
            string feedbackText, bool itemGranted)
        {
            AttemptId = attemptId ?? string.Empty;
            Outcome = outcome;
            FeedbackText = feedbackText ?? string.Empty;
            ItemGranted = itemGranted;
        }
    }

    public sealed class InWorldRecognitionReadinessSnapshot
    {
        public bool BridgeAssigned { get; }
        public bool BridgeBound { get; }
        public bool CameraStarted { get; }
        public bool FirstObservationReady { get; }
        public RawMediaPipeNeutralAutoSegmenterState CoordinatorState { get; }
        public string CoordinatorFailure { get; }

        public InWorldRecognitionReadinessSnapshot(
            bool bridgeAssigned, bool bridgeBound,
            bool cameraStarted, bool firstObservationReady,
            RawMediaPipeNeutralAutoSegmenterState coordinatorState,
            string coordinatorFailure)
        {
            BridgeAssigned = bridgeAssigned;
            BridgeBound = bridgeBound;
            CameraStarted = cameraStarted;
            FirstObservationReady = firstObservationReady;
            CoordinatorState = coordinatorState;
            CoordinatorFailure = coordinatorFailure ?? string.Empty;
        }
    }

    public interface IInWorldRecognitionRuntimeOwner : IDisposable
    {
        bool IsActive { get; }
        long FrozenGeneration { get; }
        PetRobotSignPracticeSession Session { get; }
        PetRobotNeutralAutoCaptureCoordinator Coordinator { get; }
        string InitializationFailure { get; }
        bool RequestAttempt();
        bool RetrySession();
        InWorldRecognitionReadinessSnapshot SnapshotReadiness();
    }

    public interface IInWorldDialogueCameraController
    {
        void SaveState();
        void EnterDialogueCamera();
        void RestoreState();
    }

    public interface IInWorldDialoguePlayerLock
    {
        void Lock();
        void Unlock();
    }

    public interface IInWorldNpcPresenter
    {
        void TriggerGreeting();
        void TriggerConfused();
        void TriggerGiveItem();
    }
}
