using System;
using System.IO;
using UnityEngine;
using WordsInSilence.Pose;
using WordsInSilence.Practice.FullHandEnrollment;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Practice.PetRobotWorldIntegration;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Practice.PetRobotPortfolioQuest
{
    public sealed class InWorldGreetingNpcDialogueController : MonoBehaviour,
        WordsInSilence.Practice.PetRobotWorldIntegration.IWorldInteractionClaimant
    {
        [Header("NPC")]
        [SerializeField] Transform targetNpcTransform;
        [SerializeField] Animator npcAnimator;
        [SerializeField] string greetingTrigger = "Greeting";
        [SerializeField] string confusedTrigger = "Confused";
        [SerializeField] string giveItemTrigger = "GiveItem";
        [SerializeField] float interactionRange = 3f;

        [Header("Camera")]
        [SerializeField] Transform dialogueCameraAnchor;
        [SerializeField] float dialogueFov = 40f;

        [Header("Player")]
        [SerializeField] Transform playerTransform;
        [SerializeField] Behaviour[] disableDuringDialogue;

        [Header("Recognition")]
        [SerializeField] string candidateHarnessResource =
            "Diagnostics/PersonalMultiPrototypeLiveRecognitionTechnical";
        [SerializeField] PetRobotCandidateCameraStartupRoute candidateCameraRoute;
        [SerializeField] string learnerProfileId = "local_learner";
        [SerializeField] HandPoseIntegrator providerIntegrator;

        [Header("Timing")]
        [SerializeField] float questPromptSeconds = 2f;
        [SerializeField] float trackingPreparationTimeoutSeconds = 15f;
        [SerializeField] float analysisDisplaySeconds = 0.4f;

        [Header("Presentation")]
        [SerializeField] GameObject webcamPreviewRoot;
        [SerializeField] GameObject moaDialogueRoot;

        InWorldDialogueState state = InWorldDialogueState.Idle;
        bool itemGranted;
        bool playerInsideGreetingTrigger;
        string lastAttemptId;
        InWorldDialogueOutcome lastOutcome = InWorldDialogueOutcome.None;
        string lastFeedbackText = string.Empty;
        TrustedHybridOutcomeKind? lastAppliedOutcomeKind;

        /// <summary>
        /// F27. The exact canonical verdict that was applied, including the two narrow
        /// full-hand values that the coarser InWorldDialogueOutcome bucket cannot express.
        /// </summary>
        public TrustedHybridOutcomeKind? LastAppliedOutcomeKind => lastAppliedOutcomeKind;

        IInWorldRecognitionRuntimeOwner runtimeOwner;
        IInWorldDialogueCameraController cameraController;
        IInWorldDialoguePlayerLock playerLock;
        IInWorldNpcPresenter npcPresenter;
        IQuestGateVerifier gateVerifier;
        PetRobotLearnerRootIdentity identity;
        Func<IInWorldRecognitionRuntimeOwner> runtimeFactory;
        Func<float> timeSource;
        // interaction-key ownership: a frame source that mirrors the existing time source,
        // the frame a REAL close consumed, and whether the retained QuestComplete state has
        // already had its dialogue closed
        Func<int> frameSource;
        int consumedFrame = -1;
        bool questCompleteClosed;

        CandidateCountSnapshot candidateCountsBefore;
        int candidateTotalBefore;

        float promptStartTime;
        bool recognitionInitiated;
        float trackingStartTime;
        bool firstObservationReceived;

        TrustedHybridOutcomeKind? pendingOutcomeKind;
        string pendingOutcomeAttemptId;
        float analyzingStartTime;

        Vector3 savedCameraPosition;
        Quaternion savedCameraRotation;
        float savedCameraFov;
        bool[] savedBehaviourStates;
        bool statesSaved;

        // F59. Presentation only: the player avatar and the MOA companion occlude the NPC
        // close-up, so their renderers are forced off while the dialogue is open and the
        // exact previous value is restored afterwards. This instance owns its snapshot.
        readonly InWorldDialogueActorVisibility actorVisibility =
            new InWorldDialogueActorVisibility();

        public InWorldDialogueState State => state;
        public bool ItemGranted => itemGranted;
        public InWorldDialogueOutcome LastOutcome => lastOutcome;
        public string LastFeedbackText => lastFeedbackText;
        public string LastAttemptId => lastAttemptId;
        public bool NpcInRange => playerInsideGreetingTrigger;
        public bool CanStartDialogue => state == InWorldDialogueState.Idle && NpcInRange && !itemGranted;

        /// <summary>
        /// QuestComplete has been reached AND its dialogue has been closed. The completed
        /// quest and the granted item are deliberately retained; this only records that the
        /// interaction itself is over, so the shared key is released instead of being held
        /// forever.
        /// </summary>
        public bool InteractionFinished =>
            state == InWorldDialogueState.QuestComplete && questCompleteClosed;

        /// <summary>True while an interaction is actually open.</summary>
        public bool DialogueOpen =>
            state != InWorldDialogueState.Idle && !InteractionFinished;

        /// <summary>
        /// A real close consumed this frame's press. Scoped to exactly one frame, set only
        /// by a close that actually closed something, and never used to postpone anything.
        /// </summary>
        public bool ConsumedInteractionThisFrame => consumedFrame == Frame;

        /// <summary>
        /// True while this NPC owns the shared interaction key. Owning the key does not
        /// assert that a dialogue may start: CanStartDialogue and TryStartDialogue still
        /// enforce the enrollment gate and may truthfully refuse.
        /// </summary>
        public bool ClaimsInteractionInput =>
            WordsInSilence.Practice.PetRobotWorldIntegration.WorldInteractionArbiter
                .NpcOwnsInteractionKey(DialogueOpen, NpcInRange, InteractionFinished,
                                       ConsumedInteractionThisFrame);

        public string ClaimReason =>
            ConsumedInteractionThisFrame ? "npc_close_consumed_frame"
            : DialogueOpen ? "npc_dialogue_open:" + state
            : "npc_greeting_trigger";
        public bool CanRetry => state == InWorldDialogueState.Feedback &&
            lastOutcome != InWorldDialogueOutcome.StrictSuccess;
        public HandPoseIntegrator ProviderIntegrator => providerIntegrator;
        public long FrozenGeneration => runtimeOwner != null ? runtimeOwner.FrozenGeneration : 0;
        public int CandidateTotalBefore => candidateTotalBefore;
        public string InitializationFailure => runtimeOwner != null
            ? runtimeOwner.InitializationFailure : string.Empty;
        public float QuestPromptSeconds => questPromptSeconds;
        public float TrackingPreparationTimeoutSeconds => trackingPreparationTimeoutSeconds;
        public float AnalysisDisplaySeconds => analysisDisplaySeconds;
        public bool RecognitionInitiated => recognitionInitiated;
        public bool FirstObservationReceived => firstObservationReceived;
        public float GuidedProgress { get; private set; }

        float Now => timeSource != null ? timeSource() : Time.unscaledTime;
        int Frame => frameSource != null ? frameSource() : Time.frameCount;

        void Awake()
        {
            identity = BuildIdentity();
#if !UNITY_EDITOR
            if (identity == null)
            {
                // blocked: stop before the gate verifier, the runtime owner and the camera
                PetRobotVerticalSlice.PetRobotM3RouteNotice.ShowBlocked();
                enabled = false;
                return;
            }
#endif
            gateVerifier = new ProductionQuestGateVerifier(identity);
        }

        void Update()
        {
            DriveUpdate();
            DriveInput();
        }

        void DriveInput()
        {
            if (Input.GetKeyDown(KeyCode.E) || Input.GetKeyDown(KeyCode.Return) ||
                Input.GetKeyDown(KeyCode.Space))
            {
                switch (state)
                {
                    case InWorldDialogueState.Idle:
                        TryStartDialogue();
                        break;
                    case InWorldDialogueState.QuestComplete:
                        CloseDialogue();
                        break;
                }
            }
            if (Input.GetKeyDown(KeyCode.R) && CanRetry)
                TryRetry();
            if (Input.GetKeyDown(KeyCode.Escape) && state != InWorldDialogueState.Idle)
                CloseDialogue();
        }

        public void DriveUpdate()
        {
            DriveQuestPromptTimer();
            DriveTrackingTimeout();
            DriveGuidedSubState();
            DriveAnalyzingHold();
            DriveSessionOutcome();
#if UNITY_INCLUDE_TESTS
            DrivePendingTestOutcome();
#endif
        }

        void DriveQuestPromptTimer()
        {
            if (state != InWorldDialogueState.QuestPrompt) return;
            if (recognitionInitiated) return;
            float elapsed = Now - promptStartTime;
            if (elapsed < questPromptSeconds) return;
            recognitionInitiated = true;
            if (!TryBeginRecognition())
            {
                // TryBeginRecognition already set Feedback on failure
            }
        }

        void DriveTrackingTimeout()
        {
            if (state != InWorldDialogueState.TrackingPreparation) return;
            if (firstObservationReceived) return;

            if (runtimeOwner != null)
            {
                InWorldRecognitionReadinessSnapshot snap = runtimeOwner.SnapshotReadiness();
                if (snap != null && snap.FirstObservationReady)
                {
                    firstObservationReceived = true;
                    return;
                }
            }

            float elapsed = Now - trackingStartTime;
            if (elapsed < trackingPreparationTimeoutSeconds) return;

            lastOutcome = InWorldDialogueOutcome.TechnicalFailure;
            lastFeedbackText =
                "\uce74\uba54\ub77c\ub098 \ub3d9\uc791 \ucd94\uc801\uc744 \ud655\uc778\ud558\uc9c0 \ubabb\ud588\uc5b4. " +
                "\uc900\ube44 \uc0c1\ud0dc\ub97c \ud655\uc778\ud558\uace0 R\uc744 \ub20c\ub7ec \ub2e4\uc2dc \uc2dc\ub3c4\ud574 \ubd10.";
            DisposeRuntime();
            state = InWorldDialogueState.Feedback;
        }

        void DriveGuidedSubState()
        {
            if (state != InWorldDialogueState.TrackingPreparation &&
                state != InWorldDialogueState.NeutralPreparation &&
                state != InWorldDialogueState.ReadyForGreeting &&
                state != InWorldDialogueState.Capturing &&
                state != InWorldDialogueState.RecognitionActive)
                return;

            if (runtimeOwner == null || runtimeOwner.Coordinator == null) return;

            PetRobotNeutralAutoCaptureCoordinator coord = runtimeOwner.Coordinator;
            RawMediaPipeNeutralAutoSegmenterState segState = coord.State;

            if (segState == RawMediaPipeNeutralAutoSegmenterState.Failed)
            {
                string failure = coord.LastFailure;
                lastOutcome = InWorldDialogueOutcome.TechnicalFailure;
                lastFeedbackText =
                    "\uce74\uba54\ub77c\ub098 \ub3d9\uc791 \ucd94\uc801\uc744 \ud655\uc778\ud558\uc9c0 \ubabb\ud588\uc5b4. " +
                    "\uc900\ube44 \uc0c1\ud0dc\ub97c \ud655\uc778\ud558\uace0 R\uc744 \ub20c\ub7ec \ub2e4\uc2dc \uc2dc\ub3c4\ud574 \ubd10.";
                DisposeRuntime();
                state = InWorldDialogueState.Feedback;
                return;
            }

            InWorldDialogueState next;
            float progress = 0f;
            switch (segState)
            {
                case RawMediaPipeNeutralAutoSegmenterState.Inactive:
                case RawMediaPipeNeutralAutoSegmenterState.NeedsCalibration:
                case RawMediaPipeNeutralAutoSegmenterState.CalibratingNeutral:
                    next = InWorldDialogueState.TrackingPreparation;
                    progress = coord.CalibrationProgress;
                    break;
                case RawMediaPipeNeutralAutoSegmenterState.WaitingForNeutral:
                    next = InWorldDialogueState.NeutralPreparation;
                    progress = coord.NeutralHoldProgress;
                    break;
                case RawMediaPipeNeutralAutoSegmenterState.Armed:
                    next = InWorldDialogueState.ReadyForGreeting;
                    break;
                case RawMediaPipeNeutralAutoSegmenterState.Capturing:
                case RawMediaPipeNeutralAutoSegmenterState.EndCandidate:
                    next = InWorldDialogueState.Capturing;
                    progress = coord.EndHoldProgress;
                    break;
                case RawMediaPipeNeutralAutoSegmenterState.Evaluating:
                    next = InWorldDialogueState.Analyzing;
                    break;
                default:
                    next = InWorldDialogueState.RecognitionActive;
                    break;
            }

            GuidedProgress = progress;
            if (state != next && state != InWorldDialogueState.Feedback &&
                state != InWorldDialogueState.QuestComplete &&
                state != InWorldDialogueState.Analyzing)
                state = next;
        }

        void DriveAnalyzingHold()
        {
            if (state != InWorldDialogueState.Analyzing) return;
            if (!pendingOutcomeKind.HasValue) return;
            float elapsed = Now - analyzingStartTime;
            if (elapsed < analysisDisplaySeconds) return;

            TrustedHybridOutcomeKind kind = pendingOutcomeKind.Value;
            string attemptId = pendingOutcomeAttemptId;
            pendingOutcomeKind = null;
            pendingOutcomeAttemptId = null;

            lastAttemptId = attemptId;
            ApplyOutcome(kind);
        }

        void DriveSessionOutcome()
        {
            if (state != InWorldDialogueState.TrackingPreparation &&
                state != InWorldDialogueState.NeutralPreparation &&
                state != InWorldDialogueState.ReadyForGreeting &&
                state != InWorldDialogueState.Capturing &&
                state != InWorldDialogueState.Analyzing &&
                state != InWorldDialogueState.RecognitionActive)
                return;

            if (pendingOutcomeKind.HasValue) return;

            if (runtimeOwner != null && runtimeOwner.Session != null)
            {
                PetRobotSignPracticeSession session = runtimeOwner.Session;
                if (session.State == PetRobotPracticeState.Feedback &&
                    session.QuestAuthoritativeOutcome.HasValue)
                {
                    QueueSessionOutcome(session);
                    return;
                }
            }
        }

#if UNITY_INCLUDE_TESTS
        TrustedHybridOutcomeKind? pendingTestOutcome;
        string pendingTestAttemptId;

        public void InjectOutcomeForTest(string attemptId, TrustedHybridOutcomeKind kind)
        {
            pendingTestOutcome = kind;
            pendingTestAttemptId = attemptId;
        }

        void DrivePendingTestOutcome()
        {
            if (!pendingTestOutcome.HasValue) return;
            if (pendingOutcomeKind.HasValue) return;

            TrustedHybridOutcomeKind kind = pendingTestOutcome.Value;
            string attemptId = pendingTestAttemptId ?? Guid.NewGuid().ToString("N");
            pendingTestOutcome = null;
            pendingTestAttemptId = null;

            if (!string.IsNullOrEmpty(lastAttemptId) &&
                string.Equals(attemptId, lastAttemptId, StringComparison.Ordinal))
                return;

            pendingOutcomeKind = kind;
            pendingOutcomeAttemptId = attemptId;
            analyzingStartTime = Now;
            if (state != InWorldDialogueState.Feedback &&
                state != InWorldDialogueState.QuestComplete)
                state = InWorldDialogueState.Analyzing;
        }

        void ClearPendingTestOutcome()
        {
            pendingTestOutcome = null;
            pendingTestAttemptId = null;
        }
#endif

        public bool TryStartDialogue()
        {
            if (!CanStartDialogue) return false;

            try
            {
                if (gateVerifier == null || !gateVerifier.VerifyOverallEnrollmentGateReady())
                    return false;
            }
            catch (Exception)
            {
                return false;
            }

            try
            {
                candidateCountsBefore = GetCurrentCandidateCounts();
                candidateTotalBefore = candidateCountsBefore.TotalCandidateCount;
            }
            catch (Exception)
            {
                return false;
            }

            SavePlayerAndCameraState();
            EnterDialogueCamera();
            ShowPresentationHooks();

            lastFeedbackText =
                "\uc544\uae4c \ubc30\uc6e0\ub358 \ub300\ub85c \uc774 \uc0ac\ub78c\uc5d0\uac8c " +
                "\uc218\uc5b4\ub85c '\uc778\uc0ac'\ub97c \uac74\ub124\ubd10!";
            promptStartTime = Now;
            recognitionInitiated = false;
            state = InWorldDialogueState.QuestPrompt;
            TriggerNpcGreeting();
            return true;
        }

        bool TryBeginRecognition()
        {
            try
            {
                runtimeOwner = BuildRecognitionRuntime();
            }
            catch (Exception ex)
            {
                lastFeedbackText =
                    "\uce74\uba54\ub77c\ub098 \ub3d9\uc791 \ucd94\uc801\uc744 \ud655\uc778\ud558\uc9c0 \ubabb\ud588\uc5b4. " +
                    "\uc900\ube44 \uc0c1\ud0dc\ub97c \ud655\uc778\ud558\uace0 R\uc744 \ub20c\ub7ec \ub2e4\uc2dc \uc2dc\ub3c4\ud574 \ubd10.";
                lastOutcome = InWorldDialogueOutcome.TechnicalFailure;
                state = InWorldDialogueState.Feedback;
                return false;
            }

            if (runtimeOwner == null)
            {
                lastFeedbackText =
                    "\uce74\uba54\ub77c\ub098 \ub3d9\uc791 \ucd94\uc801\uc744 \ud655\uc778\ud558\uc9c0 \ubabb\ud588\uc5b4. " +
                    "\uc900\ube44 \uc0c1\ud0dc\ub97c \ud655\uc778\ud558\uace0 R\uc744 \ub20c\ub7ec \ub2e4\uc2dc \uc2dc\ub3c4\ud574 \ubd10.";
                lastOutcome = InWorldDialogueOutcome.TechnicalFailure;
                state = InWorldDialogueState.Feedback;
                return false;
            }

            if (!string.IsNullOrEmpty(runtimeOwner.InitializationFailure))
            {
                lastFeedbackText =
                    "\uce74\uba54\ub77c\ub098 \ub3d9\uc791 \ucd94\uc801\uc744 \ud655\uc778\ud558\uc9c0 \ubabb\ud588\uc5b4. " +
                    "\uc900\ube44 \uc0c1\ud0dc\ub97c \ud655\uc778\ud558\uace0 R\uc744 \ub20c\ub7ec \ub2e4\uc2dc \uc2dc\ub3c4\ud574 \ubd10.";
                lastOutcome = InWorldDialogueOutcome.TechnicalFailure;
                DisposeRuntime();
                state = InWorldDialogueState.Feedback;
                return false;
            }

            bool attemptOk;
            try { attemptOk = runtimeOwner.RequestAttempt(); }
            catch (Exception)
            {
                lastFeedbackText =
                    "\uce74\uba54\ub77c\ub098 \ub3d9\uc791 \ucd94\uc801\uc744 \ud655\uc778\ud558\uc9c0 \ubabb\ud588\uc5b4. " +
                    "\uc900\ube44 \uc0c1\ud0dc\ub97c \ud655\uc778\ud558\uace0 R\uc744 \ub20c\ub7ec \ub2e4\uc2dc \uc2dc\ub3c4\ud574 \ubd10.";
                lastOutcome = InWorldDialogueOutcome.TechnicalFailure;
                DisposeRuntime();
                state = InWorldDialogueState.Feedback;
                return false;
            }
            if (!attemptOk)
            {
                lastFeedbackText =
                    "\uce74\uba54\ub77c\ub098 \ub3d9\uc791 \ucd94\uc801\uc744 \ud655\uc778\ud558\uc9c0 \ubabb\ud588\uc5b4. " +
                    "\uc900\ube44 \uc0c1\ud0dc\ub97c \ud655\uc778\ud558\uace0 R\uc744 \ub20c\ub7ec \ub2e4\uc2dc \uc2dc\ub3c4\ud574 \ubd10.";
                lastOutcome = InWorldDialogueOutcome.TechnicalFailure;
                DisposeRuntime();
                state = InWorldDialogueState.Feedback;
                return false;
            }

            trackingStartTime = Now;
            firstObservationReceived = false;
            lastFeedbackText = string.Empty;
            state = InWorldDialogueState.TrackingPreparation;
            return true;
        }

        void QueueSessionOutcome(PetRobotSignPracticeSession session)
        {
            TrustedHybridOutcomeKind kind = session.QuestAuthoritativeOutcome.Value;
            string attemptId = session.LastAttempt != null
                ? session.LastAttempt.CaptureSessionId : Guid.NewGuid().ToString("N");

            if (!string.IsNullOrEmpty(lastAttemptId) &&
                string.Equals(attemptId, lastAttemptId, StringComparison.Ordinal))
                return;

            pendingOutcomeKind = kind;
            pendingOutcomeAttemptId = attemptId;
            analyzingStartTime = Now;
            if (state != InWorldDialogueState.Feedback &&
                state != InWorldDialogueState.QuestComplete)
                state = InWorldDialogueState.Analyzing;
        }

        void ApplyOutcome(TrustedHybridOutcomeKind kind)
        {
            lastAppliedOutcomeKind = kind;
            switch (kind)
            {
                case TrustedHybridOutcomeKind.StrictSuccess:
                    lastOutcome = InWorldDialogueOutcome.StrictSuccess;
                    lastFeedbackText =
                        "\uc798\ud588\uc5b4! \uc0c1\ub300\uac00 \ub124 \uc778\uc0ac\ub97c \uc774\ud574\ud588\uc5b4.";
                    TriggerNpcGiveItem();
                    itemGranted = true;
                    state = InWorldDialogueState.QuestComplete;
                    break;

                case TrustedHybridOutcomeKind.LowConfidenceCorrect:
                    // Step 10T: the session layer already treats LowConfidenceCorrect as
                    // quest-completing (PetRobotSignPracticeSession sets QuestCompleted
                    // for StrictSuccess and LowConfidenceCorrect alike). This case used
                    // to disagree and send the learner back to a retry prompt, which is
                    // what left the World demo stuck. The outcome LABEL is unchanged:
                    // lastOutcome stays LowConfidenceCorrect here and in every structured
                    // record, and nothing about strict acceptance is redefined.
                    lastOutcome = InWorldDialogueOutcome.LowConfidenceCorrect;
                    lastFeedbackText =
                        "\uc778\uc0ac \ub3d9\uc791\uc744 \uc54c\uc544\ubd24\uc5b4! \uc815\ud655\ub3c4 \uae30\uc900\uae4c\uc9c0\ub294 \ub2ff\uc9c0 \ubabb\ud588\uc9c0\ub9cc, \uc774\ub300\ub85c \uacc4\uc18d \uc9c4\ud589\ud560\uac8c.";
                    TriggerNpcGiveItem();
                    itemGranted = true;
                    state = InWorldDialogueState.QuestComplete;
                    break;

                case TrustedHybridOutcomeKind.WrongWord:
                    lastOutcome = InWorldDialogueOutcome.WrongWord;
                    lastFeedbackText =
                        "\uc0c1\ub300\uac00 \ub2e4\ub978 \ub3d9\uc791\uc73c\ub85c \uc774\ud574\ud588\ub098 \ubd10. " +
                        "R\uc744 \ub20c\ub7ec \uc778\uc0ac\ub97c \ub2e4\uc2dc \ud574\ubcf4\uc790.";
                    TriggerNpcConfused();
                    state = InWorldDialogueState.Feedback;
                    break;

                case TrustedHybridOutcomeKind.TechnicalFailure:
                    lastOutcome = InWorldDialogueOutcome.TechnicalFailure;
                    lastFeedbackText =
                        "\uce74\uba54\ub77c\ub098 \ub3d9\uc791 \ucd94\uc801\uc744 \ud655\uc778\ud558\uc9c0 \ubabb\ud588\uc5b4. " +
                        "\uc900\ube44 \uc0c1\ud0dc\ub97c \ud655\uc778\ud558\uace0 R\uc744 \ub20c\ub7ec \ub2e4\uc2dc \uc2dc\ub3c4\ud574 \ubd10.";
                    state = InWorldDialogueState.Feedback;
                    break;

                case TrustedHybridOutcomeKind.NotDistinguishedRetry:
                    // a valid abstention: no item, no camera claim, and a plain retry
                    lastOutcome = InWorldDialogueOutcome.NotDistinguished;
                    lastFeedbackText =
                        WordsInSilence.Recognition.FullHandRuntime
                            .FullHandGameplayFeedback.NotDistinguished;
                    TriggerNpcConfused();
                    state = InWorldDialogueState.Feedback;
                    break;

                case TrustedHybridOutcomeKind.UnsupportedWord:
                    // honest about the real vocabulary; no fallback to another route
                    lastOutcome = InWorldDialogueOutcome.UnsupportedWord;
                    lastFeedbackText =
                        WordsInSilence.Recognition.FullHandRuntime
                            .FullHandGameplayFeedback.Unsupported;
                    state = InWorldDialogueState.Feedback;
                    break;
            }
        }

        public bool TryRetry()
        {
            if (!CanRetry) return false;

            if (runtimeOwner != null && runtimeOwner.IsActive)
            {
                bool retryOk;
                try { retryOk = runtimeOwner.RetrySession(); }
                catch (Exception)
                {
                    lastOutcome = InWorldDialogueOutcome.TechnicalFailure;
                    lastFeedbackText =
                        "\uce74\uba54\ub77c\ub098 \ub3d9\uc791 \ucd94\uc801\uc744 \ud655\uc778\ud558\uc9c0 \ubabb\ud588\uc5b4. " +
                        "\uc900\ube44 \uc0c1\ud0dc\ub97c \ud655\uc778\ud558\uace0 R\uc744 \ub20c\ub7ec \ub2e4\uc2dc \uc2dc\ub3c4\ud574 \ubd10.";
                    DisposeRuntime();
                    state = InWorldDialogueState.Feedback;
                    return false;
                }
                if (!retryOk)
                {
                    lastOutcome = InWorldDialogueOutcome.TechnicalFailure;
                    lastFeedbackText =
                        "\uce74\uba54\ub77c\ub098 \ub3d9\uc791 \ucd94\uc801\uc744 \ud655\uc778\ud558\uc9c0 \ubabb\ud588\uc5b4. " +
                        "\uc900\ube44 \uc0c1\ud0dc\ub97c \ud655\uc778\ud558\uace0 R\uc744 \ub20c\ub7ec \ub2e4\uc2dc \uc2dc\ub3c4\ud574 \ubd10.";
                    DisposeRuntime();
                    state = InWorldDialogueState.Feedback;
                    return false;
                }

                bool attemptOk;
                try { attemptOk = runtimeOwner.RequestAttempt(); }
                catch (Exception)
                {
                    lastOutcome = InWorldDialogueOutcome.TechnicalFailure;
                    lastFeedbackText =
                        "\uce74\uba54\ub77c\ub098 \ub3d9\uc791 \ucd94\uc801\uc744 \ud655\uc778\ud558\uc9c0 \ubabb\ud588\uc5b4. " +
                        "\uc900\ube44 \uc0c1\ud0dc\ub97c \ud655\uc778\ud558\uace0 R\uc744 \ub20c\ub7ec \ub2e4\uc2dc \uc2dc\ub3c4\ud574 \ubd10.";
                    DisposeRuntime();
                    state = InWorldDialogueState.Feedback;
                    return false;
                }
                if (!attemptOk)
                {
                    lastOutcome = InWorldDialogueOutcome.TechnicalFailure;
                    lastFeedbackText =
                        "\uce74\uba54\ub77c\ub098 \ub3d9\uc791 \ucd94\uc801\uc744 \ud655\uc778\ud558\uc9c0 \ubabb\ud588\uc5b4. " +
                        "\uc900\ube44 \uc0c1\ud0dc\ub97c \ud655\uc778\ud558\uace0 R\uc744 \ub20c\ub7ec \ub2e4\uc2dc \uc2dc\ub3c4\ud574 \ubd10.";
                    DisposeRuntime();
                    state = InWorldDialogueState.Feedback;
                    return false;
                }
            }
            else
            {
                DisposeRuntime();
                try
                {
                    runtimeOwner = BuildRecognitionRuntime();
                }
                catch (Exception)
                {
                    lastOutcome = InWorldDialogueOutcome.TechnicalFailure;
                    lastFeedbackText =
                        "\uce74\uba54\ub77c\ub098 \ub3d9\uc791 \ucd94\uc801\uc744 \ud655\uc778\ud558\uc9c0 \ubabb\ud588\uc5b4. " +
                        "\uc900\ube44 \uc0c1\ud0dc\ub97c \ud655\uc778\ud558\uace0 R\uc744 \ub20c\ub7ec \ub2e4\uc2dc \uc2dc\ub3c4\ud574 \ubd10.";
                    state = InWorldDialogueState.Feedback;
                    return false;
                }
                if (runtimeOwner == null || !string.IsNullOrEmpty(runtimeOwner.InitializationFailure))
                {
                    lastOutcome = InWorldDialogueOutcome.TechnicalFailure;
                    lastFeedbackText =
                        "\uce74\uba54\ub77c\ub098 \ub3d9\uc791 \ucd94\uc801\uc744 \ud655\uc778\ud558\uc9c0 \ubabb\ud588\uc5b4. " +
                        "\uc900\ube44 \uc0c1\ud0dc\ub97c \ud655\uc778\ud558\uace0 R\uc744 \ub20c\ub7ec \ub2e4\uc2dc \uc2dc\ub3c4\ud574 \ubd10.";
                    DisposeRuntime();
                    state = InWorldDialogueState.Feedback;
                    return false;
                }
                bool attemptOk;
                try { attemptOk = runtimeOwner.RequestAttempt(); }
                catch (Exception)
                {
                    lastOutcome = InWorldDialogueOutcome.TechnicalFailure;
                    lastFeedbackText =
                        "\uce74\uba54\ub77c\ub098 \ub3d9\uc791 \ucd94\uc801\uc744 \ud655\uc778\ud558\uc9c0 \ubabb\ud588\uc5b4. " +
                        "\uc900\ube44 \uc0c1\ud0dc\ub97c \ud655\uc778\ud558\uace0 R\uc744 \ub20c\ub7ec \ub2e4\uc2dc \uc2dc\ub3c4\ud574 \ubd10.";
                    DisposeRuntime();
                    state = InWorldDialogueState.Feedback;
                    return false;
                }
                if (!attemptOk)
                {
                    lastOutcome = InWorldDialogueOutcome.TechnicalFailure;
                    lastFeedbackText =
                        "\uce74\uba54\ub77c\ub098 \ub3d9\uc791 \ucd94\uc801\uc744 \ud655\uc778\ud558\uc9c0 \ubabb\ud588\uc5b4. " +
                        "\uc900\ube44 \uc0c1\ud0dc\ub97c \ud655\uc778\ud558\uace0 R\uc744 \ub20c\ub7ec \ub2e4\uc2dc \uc2dc\ub3c4\ud574 \ubd10.";
                    DisposeRuntime();
                    state = InWorldDialogueState.Feedback;
                    return false;
                }
            }

            lastOutcome = InWorldDialogueOutcome.None;
            lastFeedbackText = string.Empty;
            lastAttemptId = null;
            pendingOutcomeKind = null;
            pendingOutcomeAttemptId = null;
            trackingStartTime = Now;
            firstObservationReceived = false;
#if UNITY_INCLUDE_TESTS
            ClearPendingTestOutcome();
#endif
            state = InWorldDialogueState.TrackingPreparation;
            return true;
        }

        public void CloseDialogue()
        {
            // captured BEFORE anything is mutated: only a close that actually closed an
            // open interaction may consume this frame's press, so a repeated close in the
            // retained QuestComplete state renews no suppression
            bool wasOpen = DialogueOpen;
            DisposeRuntime();
            HidePresentationHooks();
            RestorePlayerAndCameraState();
            pendingOutcomeKind = null;
            pendingOutcomeAttemptId = null;
            recognitionInitiated = false;
            firstObservationReceived = false;
            if (state != InWorldDialogueState.QuestComplete)
                state = InWorldDialogueState.Idle;
            lastOutcome = InWorldDialogueOutcome.None;
            lastFeedbackText = string.Empty;
            lastAttemptId = null;
            // QuestComplete is retained on purpose; record that its interaction is over so
            // the shared key is released. Derived from the state above, so it clears itself
            // if a later dialogue ever leaves QuestComplete.
            questCompleteClosed = state == InWorldDialogueState.QuestComplete;
            if (wasOpen) consumedFrame = Frame;
        }

        void OnEnable()
        {
            WordsInSilence.Practice.PetRobotWorldIntegration.WorldInteractionArbiter
                .Register(this);
        }

        void OnDisable()
        {
            WordsInSilence.Practice.PetRobotWorldIntegration.WorldInteractionArbiter
                .Unregister(this);
            // a normal close may never have happened; the camera and player locking
            // lifecycle is intentionally left exactly as it was
            actorVisibility.Restore();
        }

        void OnDestroy()
        {
            WordsInSilence.Practice.PetRobotWorldIntegration.WorldInteractionArbiter
                .Unregister(this);
            DisposeRuntime();
            actorVisibility.Restore();
            if (statesSaved)
                RestorePlayerAndCameraState();
        }

        void DisposeRuntime()
        {
            if (runtimeOwner != null)
            {
                runtimeOwner.Dispose();
                runtimeOwner = null;
            }
        }

        IInWorldRecognitionRuntimeOwner BuildRecognitionRuntime()
        {
            if (runtimeFactory != null) return runtimeFactory();
            return new ProductionInWorldRecognitionRuntimeOwner(
                identity, candidateHarnessResource, candidateCameraRoute, providerIntegrator);
        }

        PetRobotLearnerRootIdentity BuildIdentity()
        {
            if (PetRobotVerticalSlice.DemoParticipantSession.IsActive)
            {
                var demo = PetRobotVerticalSlice.DemoParticipantSession.Current;
                return new PetRobotLearnerRootIdentity(demo.LearnerId, demo.CandidateRoot);
            }

            string explicitRoot = null;
#if UNITY_EDITOR
            string cleanRoot = PetRobotDeveloperCleanCandidateRootOverride.ResolveForBootstrap();
            if (!string.IsNullOrWhiteSpace(cleanRoot))
                explicitRoot = cleanRoot;
#else
            // M3 route: opt-in local working copy. Fail closed; never fall back.
            if (!PetRobotVerticalSlice.PetRobotPlayerCandidateRootOverride
                    .TryResolveForPlayer(learnerProfileId, out explicitRoot))
                return null;
#endif
            return new PetRobotLearnerRootIdentity(learnerProfileId, explicitRoot);
        }

        CandidateCountSnapshot GetCurrentCandidateCounts()
        {
            if (identity == null) return CandidateCountSnapshot.Empty;
            PersonalizationCandidateRepository repo;
            if (identity.HasExplicitRoot)
                repo = new PersonalizationCandidateRepository(identity.LearnerId, identity.ExplicitRoot);
            else
                repo = new PersonalizationCandidateRepository(identity.LearnerId);
            return repo.Counts;
        }

        void SavePlayerAndCameraState()
        {
            if (cameraController != null)
            {
                cameraController.SaveState();
            }
            else
            {
                UnityEngine.Camera main = UnityEngine.Camera.main;
                if (main != null)
                {
                    savedCameraPosition = main.transform.position;
                    savedCameraRotation = main.transform.rotation;
                    savedCameraFov = main.fieldOfView;
                }
            }

            if (playerLock != null)
            {
                playerLock.Lock();
            }
            else if (disableDuringDialogue != null)
            {
                savedBehaviourStates = new bool[disableDuringDialogue.Length];
                for (int i = 0; i < disableDuringDialogue.Length; i++)
                {
                    if (disableDuringDialogue[i] != null)
                    {
                        savedBehaviourStates[i] = disableDuringDialogue[i].enabled;
                        disableDuringDialogue[i].enabled = false;
                    }
                }
            }
            // discovered once, here, at dialogue entry: nothing is searched per frame
            actorVisibility.Hide(playerTransform);
            statesSaved = true;
        }

        void EnterDialogueCamera()
        {
            if (cameraController != null)
            {
                cameraController.EnterDialogueCamera();
                return;
            }
            UnityEngine.Camera main = UnityEngine.Camera.main;
            if (main == null) return;
            if (dialogueCameraAnchor != null)
            {
                main.transform.position = dialogueCameraAnchor.position;
                main.transform.rotation = dialogueCameraAnchor.rotation;
            }
            main.fieldOfView = dialogueFov;
        }

        void RestorePlayerAndCameraState()
        {
            // deliberately before the early return: the avatar must come back even if the
            // camera and player state were never saved for this interaction
            actorVisibility.Restore();
            if (!statesSaved) return;

            if (cameraController != null)
            {
                cameraController.RestoreState();
            }
            else
            {
                UnityEngine.Camera main = UnityEngine.Camera.main;
                if (main != null)
                {
                    main.transform.position = savedCameraPosition;
                    main.transform.rotation = savedCameraRotation;
                    main.fieldOfView = savedCameraFov;
                }
            }

            if (playerLock != null)
            {
                playerLock.Unlock();
            }
            else if (disableDuringDialogue != null && savedBehaviourStates != null)
            {
                for (int i = 0; i < disableDuringDialogue.Length; i++)
                {
                    if (disableDuringDialogue[i] != null && i < savedBehaviourStates.Length)
                        disableDuringDialogue[i].enabled = savedBehaviourStates[i];
                }
            }
            statesSaved = false;
        }

        void ShowPresentationHooks()
        {
            if (webcamPreviewRoot != null) webcamPreviewRoot.SetActive(true);
            if (moaDialogueRoot != null) moaDialogueRoot.SetActive(true);
        }

        void HidePresentationHooks()
        {
            if (webcamPreviewRoot != null) webcamPreviewRoot.SetActive(false);
            if (moaDialogueRoot != null) moaDialogueRoot.SetActive(false);
        }

        void TriggerNpcGreeting()
        {
            if (npcPresenter != null) { npcPresenter.TriggerGreeting(); return; }
            if (npcAnimator != null && !string.IsNullOrEmpty(greetingTrigger))
                npcAnimator.SetTrigger(greetingTrigger);
        }

        void TriggerNpcConfused()
        {
            if (npcPresenter != null) { npcPresenter.TriggerConfused(); return; }
            if (npcAnimator != null && !string.IsNullOrEmpty(confusedTrigger))
                npcAnimator.SetTrigger(confusedTrigger);
        }

        void TriggerNpcGiveItem()
        {
            if (npcPresenter != null) { npcPresenter.TriggerGiveItem(); return; }
            if (npcAnimator != null && !string.IsNullOrEmpty(giveItemTrigger))
                npcAnimator.SetTrigger(giveItemTrigger);
        }

        public void ConfigureForTests(
            Transform npc, Transform player, float range,
            IQuestGateVerifier testGate,
            PetRobotLearnerRootIdentity testIdentity,
            IInWorldRecognitionRuntimeOwner testRuntime,
            IInWorldDialogueCameraController testCamera,
            IInWorldDialoguePlayerLock testPlayerLock,
            IInWorldNpcPresenter testNpcPresenter)
        {
            targetNpcTransform = npc;
            playerTransform = player;
            interactionRange = range;
            gateVerifier = testGate;
            identity = testIdentity;
            runtimeOwner = null;
            cameraController = testCamera;
            playerLock = testPlayerLock;
            npcPresenter = testNpcPresenter;
            state = InWorldDialogueState.Idle;
            itemGranted = false;
            lastAttemptId = null;
            lastOutcome = InWorldDialogueOutcome.None;
            lastFeedbackText = string.Empty;
            statesSaved = false;
            candidateTotalBefore = 0;
            playerInsideGreetingTrigger = false;
            recognitionInitiated = false;
            firstObservationReceived = false;
            pendingOutcomeKind = null;
            pendingOutcomeAttemptId = null;
            runtimeFactory = testRuntime != null
                ? (() => testRuntime)
                : (Func<IInWorldRecognitionRuntimeOwner>)null;
        }

        public void SetRuntimeFactoryForTest(Func<IInWorldRecognitionRuntimeOwner> factory)
        {
            runtimeFactory = factory;
        }

        public void SetPlayerInsideGreetingTrigger(bool inside)
        {
            playerInsideGreetingTrigger = inside;
        }

        public void SetTimeSourceForTest(Func<float> source)
        {
            timeSource = source;
        }

        /// <summary>
        /// Test seam for the one-frame input-consumption rule. Production leaves this unset
        /// and uses the default Unity frame clock.
        /// </summary>
        public void SetFrameSourceForTest(Func<int> source)
        {
            frameSource = source;
        }
    }

    internal sealed class ProductionInWorldRecognitionRuntimeOwner : IInWorldRecognitionRuntimeOwner
    {
        readonly PetRobotSignPracticeSession session;
        readonly PetRobotNeutralAutoCaptureCoordinator coordinator;
        readonly PersonalLivePracticeRecognitionAdapter adapter;
        readonly PersonalLiveRecognitionHarness harness;
        readonly GameObject harnessInstance;
        bool disposed;

        public bool IsActive => !disposed && session != null;
        public long FrozenGeneration => session != null ? session.QuestSessionGeneration : 0;
        public PetRobotSignPracticeSession Session => session;
        public PetRobotNeutralAutoCaptureCoordinator Coordinator => coordinator;
        public string InitializationFailure { get; }

        public ProductionInWorldRecognitionRuntimeOwner(
            PetRobotLearnerRootIdentity identity,
            string candidateHarnessResource,
            PetRobotCandidateCameraStartupRoute cameraRoute,
            HandPoseIntegrator explicitIntegrator)
        {
            if (explicitIntegrator == null)
            {
                InitializationFailure = "providerIntegrator is not assigned in Inspector";
                return;
            }

            try
            {
                GameObject prefab = Resources.Load<GameObject>(candidateHarnessResource);
                if (prefab == null) throw new InvalidOperationException("candidate harness prefab missing");
                harnessInstance = UnityEngine.Object.Instantiate(prefab);
                harnessInstance.name = "InWorldRecognitionRuntime";
                harness = harnessInstance.GetComponent<PersonalLiveRecognitionHarness>();
                if (harness == null) throw new InvalidOperationException("harness component missing");
                if (cameraRoute == null || !cameraRoute.isActiveAndEnabled)
                    throw new InvalidOperationException("camera route not wired");
                harness.SetTechnicalPanelVisible(false);

                if (harness.ObservationBridge == null)
                    throw new InvalidOperationException("harness observation bridge is null");
                harness.ObservationBridge.BindForDiagnostic(explicitIntegrator);
                if (!harness.ObservationBridge.IsBound)
                    throw new InvalidOperationException("bridge did not bind after explicit integrator assignment");

                cameraRoute.BindObservationBridge(harness.ObservationBridge);

                adapter = new PersonalLivePracticeRecognitionAdapter(harness);

                PersonalizationCandidateRepository repository;
                if (identity.HasExplicitRoot)
                    repository = new PersonalizationCandidateRepository(
                        identity.LearnerId, identity.ExplicitRoot);
                else
                    repository = new PersonalizationCandidateRepository(identity.LearnerId);

                var token = new PetRobotGreetingQuestToken();
                if (FullHandEnrollmentRuntime.ModeActive)
                {
                    FullHandEnrollmentBinding fullHandBinding;
                    FullHandParticipantCandidateStore fullHandStore;
                    string storeCode, storeDetail;
                    if (!FullHandEnrollmentRuntime.TryOpenForCurrentParticipant(
                            identity, out fullHandBinding, out fullHandStore,
                            out storeCode, out storeDetail))
                        throw new InvalidOperationException(
                            "the full-hand route is enabled but unavailable: " + storeCode);
                    string questGateFailure;
                    if (!FullHandEnrollmentQuestGate.Verify(fullHandStore, out questGateFailure))
                        throw new InvalidOperationException(
                            "the full-hand practice records are not complete for this participant");
                    session = new PetRobotSignPracticeSession(
                        adapter, repository, token, fullHandBinding, fullHandStore, cameraRoute);
                }
                else
                {
                    TrustedHybridRuntimeSession rawSession = CreateHybridSession(repository);
                    var hybridSession = new TrustedHybridRecognitionSessionAdapter(rawSession);
                    session = new PetRobotSignPracticeSession(
                        adapter, repository, hybridSession, token, cameraRoute);
                }

                var segmentationSettings = PetRobotNeutralAutoSegmentationSettings.Default;
                var profileService = new PetRobotNeutralCalibrationProfileService(
                    identity.LearnerId,
                    PetRobotVerticalSlice.DemoParticipantSession.ProfilesRootOrNull);
                var segmenter = new RawMediaPipeNeutralAutoSegmenter(
                    identity.LearnerId, segmentationSettings);
                var ring = new RawMediaPipeObservationRingBuffer(segmentationSettings.PreRollSeconds);
                var observationSource = new Step9CObservationSourceAdapter(harness.ObservationBridge);
                coordinator = new PetRobotNeutralAutoCaptureCoordinator(
                    session, profileService, segmenter, ring, observationSource, cameraRoute,
                    allowDeferredObservationBinding: true);
            }
            catch (Exception ex)
            {
                InitializationFailure = ex.GetType().Name + ": " + ex.Message;
                Dispose();
            }
        }

        public bool RequestAttempt()
        {
            if (disposed || coordinator == null) return false;
            return coordinator.RequestAttempt();
        }

        public bool RetrySession()
        {
            if (disposed || session == null) return false;
            return session.Retry();
        }

        public InWorldRecognitionReadinessSnapshot SnapshotReadiness()
        {
            if (disposed) return null;
            bool bridgeAssigned = harness != null && harness.ObservationBridge != null;
            bool bridgeBound = bridgeAssigned && harness.ObservationBridge.IsBound;
            bool cameraStarted = false;
            bool firstObs = bridgeAssigned && harness.ObservationBridge.HasReceivedObservation;
            RawMediaPipeNeutralAutoSegmenterState coordState =
                coordinator != null ? coordinator.State : RawMediaPipeNeutralAutoSegmenterState.Inactive;
            string coordFailure = coordinator != null ? coordinator.LastFailure : string.Empty;
            return new InWorldRecognitionReadinessSnapshot(
                bridgeAssigned, bridgeBound, cameraStarted, firstObs, coordState, coordFailure);
        }

        TrustedHybridRuntimeSession CreateHybridSession(
            PersonalizationCandidateRepository repository)
        {
            ITrustedPersonalizationCandidateRepository trusted =
                repository as ITrustedPersonalizationCandidateRepository;
            if (trusted == null)
                throw new InvalidOperationException("trusted repository required");
            TrustedCandidateInventory inventory = trusted.ReloadTrustedAndVerify();
            if (!inventory.OverallEnrollmentGateReady)
                throw new InvalidOperationException("enrollment gate not ready");
            string templateDir = Path.Combine(
                Application.streamingAssetsPath, "RecognitionCoreV1", "Templates");
            string thresholdPath = Path.Combine(
                Application.streamingAssetsPath, "RecognitionCoreV1", "RuntimeGate",
                "frozen_threshold_config.json");
            RuntimeRecognitionService globalService =
                RuntimeRecognitionService.Load(templateDir, thresholdPath);
            if (DemoParticipantSession.IsActive)
            {
                // Participant-only prototypes over the same public global service; the fixed
                // personal profile below is never read in this route.
                return WordsInSilence.Recognition.LiveInput.DemoParticipantLiveRuntimeFactory
                    .CreateQuestSession(globalService, trusted, DemoParticipantSession.Current);
            }
            string fixedProfileDir = Path.Combine(
                Application.streamingAssetsPath, "RecognitionCoreV1", "PersonalProfiles",
                ApprovedPersonalMultiPrototypeLiveRuntimeFactory.ProfileDirectoryName);
            var viewService = new TrustedPersonalizationActiveViewService(
                globalService, fixedProfileDir, trusted);
            TrustedActiveViewBuildResult refreshResult = viewService.TryRefresh();
            if (!refreshResult.Succeeded)
                throw new InvalidOperationException("active view refresh failed: " +
                    refreshResult.FailureType + ": " + refreshResult.FailureMessage);
            return viewService.AcquireSession();
        }

        public void Dispose()
        {
            if (disposed) return;
            disposed = true;
            coordinator?.Dispose();
            session?.Dispose();
            if (harnessInstance != null)
                UnityEngine.Object.Destroy(harnessInstance);
        }
    }
}
