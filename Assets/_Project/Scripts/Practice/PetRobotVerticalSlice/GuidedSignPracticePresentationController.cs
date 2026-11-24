using System.Globalization;
using UnityEngine;
using UnityEngine.UI;
using WordsInSilence.Practice.FullHandEnrollment;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.PhysicalCompatibility;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    /// <summary>
    /// Rounded uGUI presentation adapter over the approved session and neutral auto coordinator.
    /// It dispatches explicit story actions without owning recognition, segmentation, or storage.
    /// </summary>
    public sealed class GuidedSignPracticePresentationController : MonoBehaviour
    {
        public const float ReferenceWidth = 1920f;
        public const float ReferenceHeight = 1080f;
        public static readonly Rect WebcamPanelRect = new Rect(570f, 145f, 780f, 500f);
        public static readonly Rect BottomHudRect = new Rect(150f, 858f, 1620f, 190f);
        public const int RoundedTextureSize = 32;
        public const int RoundedCornerRadius = 8;
        public const float RoundedPixelsPerUnit = 100f;

        [SerializeField] PetRobotPracticeUiPresenter legacyTechnicalUi;
        [SerializeField] SignDemonstrationAvatarPresenter visualStage;
        [SerializeField] WebcamLandmarkOverlayPresenter landmarkOverlay;
        [SerializeField] Font koreanFont;
        [SerializeField] bool diagnosticsVisibleByDefault;

        PetRobotSignPracticeSession session;
        PetRobotPracticePresenter robot;
        PetRobotCandidateCameraStartupRoute cameraRoute;
        PetRobotNeutralAutoCaptureCoordinator neutralAutoCaptureCoordinator;
        readonly MoaStoryGuidedPracticeFlow storyFlow = new MoaStoryGuidedPracticeFlow();
        bool diagnosticsVisible;
        bool questModeActive;
        string lastWord = string.Empty;
        string initializationFailure = string.Empty;
        Texture2D runtimeRoundedTexture;
        Sprite runtimeRoundedSprite;

        GameObject canvasObject;
        Text titleText;
        Text currentWordText;
        Text progressText;
        Text cameraText;
        Text resultText;
        Text candidateText;
        Text completedText;
        Text robotNameText;
        Text robotSpeechText;
        Text diagnosticsText;
        GameObject diagnosticsPanel;
        Button diagnosticsButton;
        Text diagnosticsButtonText;
        Button startButton;
        Button finishButton;
        Button retryButton;
        Button needsPracticeButton;
        Button nextButton;
        Text nextButtonText;
        Button exitButton;
        Button storyActionButton;
        Text storyActionButtonText;
        GameObject demonstrationHintPanel;
        Text demonstrationHintText;

        static readonly Color PanelColor = new Color(0.018f, 0.055f, 0.095f, 0.6f);
        static readonly Color HudColor = new Color(0.012f, 0.032f, 0.065f, 0.74f);
        static readonly Color BorderColor = new Color(0.12f, 0.72f, 0.88f, 0.84f);
        static readonly Color AccentColor = new Color(0.36f, 0.95f, 0.92f, 1f);
        static readonly Color PrimaryButtonColor = new Color(0.08f, 0.5f, 0.55f, 1f);

        public bool DiagnosticsVisible => diagnosticsVisible;
        public bool UsesExistingCameraOwner => true;
        public bool StartsCaptureAutomatically => false;
        public bool StoresRgbFrames => false;
        public bool UsesUgui => true;
        public bool UsesHandsFreeStoryActions => true;
        public bool LegacyManualControlsVisible => false;
        public Sprite RoundedGraphicSource => runtimeRoundedSprite;
        public Texture2D RoundedTextureSource => runtimeRoundedTexture;
        public bool OwnsRuntimeRoundedResources =>
            runtimeRoundedSprite != null && runtimeRoundedTexture != null;

        void Awake()
        {
            diagnosticsVisible = diagnosticsVisibleByDefault;
            BuildUguiHierarchy();
            if (legacyTechnicalUi != null) legacyTechnicalUi.enabled = false;
        }

        void Update()
        {
            RefreshUgui();
        }

        public void Bind(
            PetRobotSignPracticeSession practiceSession,
            PetRobotPracticePresenter presenter,
            PetRobotCandidateCameraStartupRoute route,
            Step9CHandIdentityObservationBridge observationBridge)
        {
            Bind(practiceSession, presenter, route, observationBridge, null);
        }

        public void Bind(
            PetRobotSignPracticeSession practiceSession,
            PetRobotPracticePresenter presenter,
            PetRobotCandidateCameraStartupRoute route,
            Step9CHandIdentityObservationBridge observationBridge,
            PetRobotNeutralAutoCaptureCoordinator autoCaptureCoordinator)
        {
            if (session != null) session.Changed -= OnSessionChanged;
            if (neutralAutoCaptureCoordinator != null)
                neutralAutoCaptureCoordinator.Changed -= OnCoordinatorChanged;
            session = practiceSession;
            robot = presenter;
            cameraRoute = route;
            neutralAutoCaptureCoordinator = autoCaptureCoordinator;
            diagnosticsVisible = diagnosticsVisibleByDefault;
            questModeActive = session != null && session.QuestMode;
            // F28: pin the story to the words THIS session practises
            if (session != null) storyFlow.ConfigureVocabulary(session.Words);
            storyFlow.Reset(session == null ? "GREETING" : session.CurrentWord);
            if (questModeActive)
            {
                storyFlow.AdvanceStory();
                storyFlow.AdvanceStory();
                storyFlow.AdvanceStory();
            }
            if (legacyTechnicalUi != null) legacyTechnicalUi.enabled = false;
            landmarkOverlay?.Bind(route, observationBridge);
            if (session != null)
            {
                session.Changed += OnSessionChanged;
                lastWord = session.CurrentWord;
            }
            if (neutralAutoCaptureCoordinator != null)
                neutralAutoCaptureCoordinator.Changed += OnCoordinatorChanged;
            DemonstrationStage()?.ClearDemonstration();
            RefreshUgui();
        }

        /// <summary>
        /// F30: presentation only. Awake may already have built the hierarchy, so setting the
        /// serialized field alone is not enough; every uGUI Text this component owns is
        /// retargeted as well. Null is ignored, which leaves older variants unchanged.
        /// </summary>
        public void SetPresentationFont(Font font)
        {
            if (font == null) return;
            koreanFont = font;
            if (canvasObject == null) return;
            foreach (Text text in canvasObject.GetComponentsInChildren<Text>(true))
                if (text != null) text.font = font;
        }

        public void ShowInitializationFailure(string message)
        {
            initializationFailure = message ?? "unknown error";
            RefreshUgui();
        }

        void OnDestroy()
        {
            if (session != null) session.Changed -= OnSessionChanged;
            if (neutralAutoCaptureCoordinator != null)
                neutralAutoCaptureCoordinator.Changed -= OnCoordinatorChanged;
            landmarkOverlay?.Unbind();
            ReleaseRoundedGraphicResources();
        }

        void OnSessionChanged()
        {
            if (session == null) return;
            if (session.State != PetRobotPracticeState.Completed &&
                session.State != PetRobotPracticeState.Exited && session.CurrentWord != lastWord)
            {
                lastWord = session.CurrentWord;
            }
            RefreshUgui();
        }

        void OnCoordinatorChanged()
        {
            RefreshUgui();
        }

        void BuildUguiHierarchy()
        {
            if (canvasObject != null) return;
            canvasObject = new GameObject(
                "Step10F_v1_2_GuidedPracticeCanvas",
                typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            canvasObject.transform.SetParent(transform, false);
            Canvas canvas = canvasObject.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 20;
            CanvasScaler scaler = canvasObject.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(ReferenceWidth, ReferenceHeight);
            scaler.screenMatchMode = CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
            scaler.matchWidthOrHeight = 0.5f;

            RectTransform root = canvasObject.GetComponent<RectTransform>();
            RectTransform titleBackplate = CreateRoundedPanel("PracticeTitleBackplate", root,
                new Rect(32f, 20f, 438f, 62f), new Color(0.012f, 0.03f, 0.055f, 0.94f), true);
            titleText = CreateText("PracticeTitle", titleBackplate,
                new Rect(16f, 7f, 406f, 48f), 28,
                TextAnchor.MiddleLeft, Color.white, FontStyle.Bold);
            titleText.text = "MOA와 함께하는 수어 연습";
            // F30: this header carries the record n/3 AND the attempts m/10 progress line, which
            // the old 210 px field truncated. The neighbouring areas grow with it, it stays clear
            // of the diagnostics toggle at x=1660, and the plate is opaque enough for white text.
            RectTransform statusHeader = CreateRoundedPanel("CompactPracticeStatusHeader", root,
                new Rect(470f, 20f, 1170f, 84f), new Color(0.012f, 0.03f, 0.055f, 0.94f), true);
            currentWordText = CreateText("CurrentWordHeader", statusHeader,
                new Rect(18f, 12f, 330f, 60f), 28,
                TextAnchor.MiddleLeft, AccentColor, FontStyle.Bold);
            progressText = CreateText("WordProgress", statusHeader,
                new Rect(360f, 14f, 420f, 56f), 22,
                TextAnchor.MiddleCenter, Color.white, FontStyle.Bold);
            cameraText = CreateText("CameraTrackingReadiness", statusHeader,
                new Rect(790f, 14f, 362f, 56f), 19,
                TextAnchor.MiddleRight, Color.white, FontStyle.Bold);

            demonstrationHintPanel = CreateRoundedPanel(
                "CompactDemonstrationHint", root, new Rect(32f, 120f, 360f, 76f),
                new Color(0.012f, 0.03f, 0.055f, 0.94f), true).gameObject;
            demonstrationHintText = CreateText("DemonstrationHintText", root,
                new Rect(48f, 126f, 328f, 64f), 20, TextAnchor.MiddleCenter, Color.white, FontStyle.Normal);
            demonstrationHintText.text = "3D 수어 시범  ·  ← / → 회전";

            CreateRoundedPanel("WebcamCardAccent", root,
                new Rect(WebcamPanelRect.x, WebcamPanelRect.y - 10f, WebcamPanelRect.width, 4f),
                new Color(AccentColor.r, AccentColor.g, AccentColor.b, 0.78f), false);

            RectTransform webcamPanel = CreateRoundedPanel(
                "RoundedWebcamPanel", root, WebcamPanelRect, PanelColor, true);
            Mask roundedMask = webcamPanel.gameObject.AddComponent<Mask>();
            roundedMask.showMaskGraphic = true;
            landmarkOverlay?.AttachView(webcamPanel);
            RectTransform bubble = CreateRoundedPanel("MoaRobotSpeechBubble", root,
                new Rect(1490f, 118f, 398f, 500f), new Color(0.012f, 0.03f, 0.055f, 0.94f), true);
            robotNameText = CreateText("MoaRobotNameLabel", bubble,
                new Rect(20f, 10f, 358f, 30f), 20, TextAnchor.MiddleLeft,
                AccentColor, FontStyle.Bold);
            robotNameText.text = "MOA";
            robotSpeechText = CreateText("MoaRobotSpeech", bubble,
                new Rect(20f, 46f, 358f, 330f), 24, TextAnchor.UpperLeft,
                Color.white, FontStyle.Bold);
            storyActionButton = CreateButton("MoaStoryActionButton", bubble,
                new Rect(20f, 392f, 358f, 68f), "다음", HandleStoryAction, PrimaryButtonColor);
            storyActionButtonText = storyActionButton.GetComponentInChildren<Text>();

            RectTransform hud = CreateRoundedPanel("RoundedProgressAndControlsHud", root,
                BottomHudRect, HudColor, true);
            resultText = CreateText("OutcomeResult", hud, new Rect(20f, 14f, 660f, 40f), 22,
                TextAnchor.MiddleLeft, Color.white, FontStyle.Bold);
            completedText = CreateText("CompletedSummary", hud, new Rect(690f, 8f, 890f, 52f), 22,
                TextAnchor.MiddleRight, AccentColor, FontStyle.Bold);
            candidateText = CreateText("CandidateStorageResult", hud, new Rect(20f, 56f, 1560f, 36f), 17,
                TextAnchor.MiddleLeft, Color.white, FontStyle.Normal);

            startButton = CreateButton("StartExpressionButton", hud, new Rect(20f, 110f, 190f, 54f),
                "표현 시작", HandleStartExpression, PrimaryButtonColor);
            finishButton = CreateButton("FinishExpressionButton", hud, new Rect(220f, 110f, 190f, 54f),
                "표현 완료", HandleFinishExpression, PrimaryButtonColor);
            retryButton = CreateButton("RetryButton", hud, new Rect(420f, 110f, 170f, 54f),
                "다시 시도", HandleRetry, new Color(0.28f, 0.39f, 0.54f, 1f));
            needsPracticeButton = CreateButton("NeedsPracticeContinueButton", hud,
                new Rect(600f, 110f, 325f, 54f), "연습 필요로 남기고 계속",
                HandleNeedsPracticeContinue, new Color(0.55f, 0.36f, 0.2f, 1f));
            nextButton = CreateButton("NextWordButton", hud, new Rect(935f, 110f, 205f, 54f),
                "다음 단어", HandleNextWord, PrimaryButtonColor);
            nextButtonText = nextButton.GetComponentInChildren<Text>();
            exitButton = CreateButton("ExitPracticeButton", hud, new Rect(1150f, 110f, 450f, 54f),
                "연습 공간 나가기", HandleExitPractice, new Color(0.34f, 0.3f, 0.42f, 1f));

            diagnosticsButton = CreateButton("DiagnosticsToggle", root, new Rect(1660f, 28f, 220f, 50f),
                "진단 정보", ToggleDiagnostics, new Color(0.12f, 0.2f, 0.28f, 0.88f));
            diagnosticsButtonText = diagnosticsButton.GetComponentInChildren<Text>();
            RectTransform diagnostics = CreateRoundedPanel("DiagnosticsPanel", root,
                new Rect(420f, 642f, 1080f, 150f), new Color(0.01f, 0.02f, 0.035f, 0.96f), true);
            diagnosticsPanel = diagnostics.gameObject;
            diagnosticsText = CreateText("DiagnosticsText", diagnostics, new Rect(18f, 8f, 1044f, 134f), 16,
                TextAnchor.UpperLeft, new Color(0.72f, 0.86f, 0.92f, 1f), FontStyle.Normal);
            diagnosticsPanel.SetActive(diagnosticsVisible);
            currentWordText.gameObject.SetActive(false);
            progressText.gameObject.SetActive(false);
            demonstrationHintPanel.SetActive(false);
            demonstrationHintText.gameObject.SetActive(false);
            HideLegacyNormalStoryControls();
        }

        void RefreshUgui()
        {
            if (titleText == null) return;
            diagnosticsPanel?.SetActive(diagnosticsVisible);
            if (diagnosticsButtonText != null)
                diagnosticsButtonText.text = diagnosticsVisible ? "진단 정보 닫기" : "진단 정보";

            if (!string.IsNullOrEmpty(initializationFailure))
            {
                currentWordText.text = "초기화 실패";
                // short line for the learner; the full existing detail goes to the
                // diagnostics panel instead of being clipped in this narrow field
                cameraText.text = "초기화에 실패했습니다. '진단 정보'를 열어 상세를 확인하세요.";
                robotSpeechText.text = "카메라와 연습 경로를 확인해 주세요.";
                if (diagnosticsText != null)
                    diagnosticsText.text = initializationFailure;
                SetAllActionButtonsDisabled();
                return;
            }
            if (session == null)
            {
                currentWordText.text = "연습 세션 준비 중";
                cameraText.text = "카메라 시작 대기";
                robotSpeechText.text = "잠시만 기다려 주세요. 연습을 준비하고 있어요.";
                SetAllActionButtonsDisabled();
                return;
            }

            SynchronizeStoryFlow();
            MoaStoryPresentation story = storyFlow.Current;
            SynchronizeDemonstration(story);
            bool showCurrentWord = story.Beat == MoaStoryBeat.WordIntroduction ||
                story.Beat == MoaStoryBeat.Practice || story.Beat == MoaStoryBeat.Feedback;
            currentWordText.gameObject.SetActive(showCurrentWord);
            progressText.gameObject.SetActive(showCurrentWord);
            demonstrationHintPanel?.SetActive(story.Beat == MoaStoryBeat.WordIntroduction);
            demonstrationHintText?.gameObject.SetActive(story.Beat == MoaStoryBeat.WordIntroduction);
            if (showCurrentWord)
                currentWordText.text = session.CurrentWord + " · " +
                    SignDemonstrationAvatarPresenter.KoreanWord(session.CurrentWord);
            if (questModeActive)
                titleText.text = "NPC GREETING Quest";
            progressText.text = questModeActive ? QuestProgressText() : ProgressText();
            cameraText.text = CameraText();
            resultText.text = OutcomeText();
            candidateText.text = CandidateText();
            completedText.text = AutoPhaseStatusText();
            robotSpeechText.text = story.BodyText;
            storyActionButton.gameObject.SetActive(story.ActionVisible);
            storyActionButton.interactable = story.ActionEnabled;
            storyActionButtonText.text = story.ActionLabel;
            diagnosticsText.text = DiagnosticsText();

            resultText.gameObject.SetActive(false);
            candidateText.gameObject.SetActive(false);
            HideLegacyNormalStoryControls();

            startButton.interactable =
                session.State == PetRobotPracticeState.Ready && session.CameraReady &&
                storyFlow.ExpressionStartUnlocked;
            finishButton.interactable = session.State == PetRobotPracticeState.Capturing;
            retryButton.interactable = session.CanRetry;
            needsPracticeButton.gameObject.SetActive(false);
            needsPracticeButton.interactable = session.CanContinueWithNeedsPractice;
            bool canContinue = session.State == PetRobotPracticeState.Feedback && session.IsCurrentWordComplete;
            nextButton.interactable = canContinue;
            nextButtonText.text = session.CurrentWord == "THANK_YOU" ? "학습 완료" : "다음 단어";
            exitButton.interactable = session.State == PetRobotPracticeState.Completed;
            nextButton.gameObject.SetActive(false);
            exitButton.gameObject.SetActive(false);
            startButton.gameObject.SetActive(false);
            finishButton.gameObject.SetActive(false);
            retryButton.gameObject.SetActive(false);
            needsPracticeButton.gameObject.SetActive(false);
        }

        void HideLegacyNormalStoryControls()
        {
            foreach (Button button in new[]
                     {
                         startButton, finishButton, retryButton, needsPracticeButton, nextButton, exitButton
                     })
                if (button != null) button.gameObject.SetActive(false);
            if (resultText != null) resultText.gameObject.SetActive(false);
            if (candidateText != null) candidateText.gameObject.SetActive(false);
        }

        void HandleStartExpression()
        {
            if (session != null && storyFlow.ExpressionStartUnlocked && session.StartExpression())
                RefreshUgui();
        }

        void HandleFinishExpression()
        {
            if (session == null || !session.FinishExpression()) return;
            robot?.ShowOutcome(session.LastLearningOutcome);
            RefreshUgui();
        }

        void HandleRetry()
        {
            if (session != null && session.Retry())
            {
                robot?.ShowWord(session.CurrentWord);
                RefreshUgui();
            }
        }

        void HandleNeedsPracticeContinue()
        {
            if (session == null) return;
            string resolved = session.CurrentWord;
            if (!session.ContinueWithNeedsPractice()) return;
            if (session.State == PetRobotPracticeState.Completed)
                robot?.ShowCompleted(session.MasteredCount, session.NeedsPracticeCount);
            else
            {
                robot?.ShowNeedsPractice(resolved);
                robot?.ShowWord(session.CurrentWord);
            }
            RefreshUgui();
        }

        void HandleNextWord()
        {
            if (session == null || !session.ContinueToNextWord()) return;
            if (session.State == PetRobotPracticeState.Completed)
                robot?.ShowCompleted(session.MasteredCount, session.NeedsPracticeCount);
            else
                robot?.ShowWord(session.CurrentWord);
            RefreshUgui();
        }

        void HandleExitPractice()
        {
            if (session != null) session.ExitPractice();
        }

        void HandleStoryAction()
        {
            MoaStoryPresentation story = storyFlow.Current;
            if (!story.ActionVisible || !story.ActionEnabled) return;
            switch (story.Action)
            {
                case MoaStoryAction.AdvanceStory:
                    if (storyFlow.AdvanceStory()) RefreshUgui();
                    break;
                case MoaStoryAction.BeginAutoAttempt:
                    HandleBeginAutoAttempt();
                    break;
                case MoaStoryAction.RetryPreCapture:
                    HandlePreCaptureRetry();
                    break;
                case MoaStoryAction.RetryTechnicalFailure:
                    HandleTechnicalFailureRetry();
                    break;
                case MoaStoryAction.RetryNotDistinguished:
                    HandleNotDistinguishedRetry();
                    break;
                case MoaStoryAction.ReviewWrongWord:
                    HandleWrongWordReview();
                    break;
                case MoaStoryAction.RetryEnrollment:
                    HandleEnrollmentRetry();
                    break;
                case MoaStoryAction.ContinueToNextWord:
                    HandleNextWord();
                    break;
                case MoaStoryAction.ContinueWithNeedsPractice:
                    HandleNeedsPracticeContinue();
                    break;
                case MoaStoryAction.LeaveRoundIncomplete:
                    if (session != null && session.LeaveRoundIncomplete()) RefreshUgui();
                    break;
                case MoaStoryAction.ExitPractice:
                    HandleExitPractice();
                    break;
            }
        }

        void HandleBeginAutoAttempt()
        {
            if (session == null || neutralAutoCaptureCoordinator == null ||
                session.State != PetRobotPracticeState.Ready || !session.CameraReady ||
                !RequestCoordinatorAttempt())
            {
                storyFlow.ReportAutoAttemptRejected();
                RefreshUgui();
                return;
            }
            if (!storyFlow.ConfirmAutoAttemptAccepted())
            {
                storyFlow.ReportAutoAttemptRejected();
                RefreshUgui();
                return;
            }
            DemonstrationStage()?.SetDemonstrationPlaybackPaused(true, true);
            RefreshUgui();
        }

        void HandlePreCaptureRetry()
        {
            if (!RequestCoordinatorAttempt()) storyFlow.ReportAutoAttemptRejected();
            RefreshUgui();
        }

        /// <summary>F27: retry after a valid abstention; nothing here claims a fault.</summary>
        void HandleNotDistinguishedRetry()
        {
            if (session == null || !session.Retry()) return;
            if (!RequestCoordinatorAttempt()) storyFlow.ReportAutoAttemptRejected();
            RefreshUgui();
        }

        void HandleTechnicalFailureRetry()
        {
            if (session == null || !session.Retry()) return;
            if (!RequestCoordinatorAttempt()) storyFlow.ReportAutoAttemptRejected();
            RefreshUgui();
        }

        void HandleWrongWordReview()
        {
            if (session == null || !session.Retry()) return;
            if (!storyFlow.ReturnToWordIntroductionForReview()) return;
            SignDemonstrationAvatarPresenter stage = DemonstrationStage();
            stage?.ShowDemonstration(session.CurrentWord);
            stage?.SetDemonstrationPlaybackPaused(false, true);
            RefreshUgui();
        }

        void HandleEnrollmentRetry()
        {
            if (session == null || !session.Retry()) return;
            if (!storyFlow.ReturnToWordIntroductionForReview()) return;
            SignDemonstrationAvatarPresenter stage = DemonstrationStage();
            stage?.ShowDemonstration(session.CurrentWord);
            stage?.SetDemonstrationPlaybackPaused(false, true);
            RefreshUgui();
        }

        bool RequestCoordinatorAttempt()
        {
            return neutralAutoCaptureCoordinator != null &&
                neutralAutoCaptureCoordinator.RequestAttempt();
        }

        void ToggleDiagnostics()
        {
            diagnosticsVisible = !diagnosticsVisible;
            RefreshUgui();
        }

        string ProgressText()
        {
            if (session.FullHandRecordCollectionActive)
                return session.IsCurrentFullHandWordCollected
                    ? FullHandEnrollmentCopy.Collected
                    : FullHandEnrollmentCopy.AttemptProgress(
                          session.CurrentFullHandRecordCount,
                          session.FullHandRequiredRecordCount,
                          session.CurrentFullHandAttemptCount,
                          PetRobotSignPracticeSession.FullHandMaxAttemptsPerWord);
            if (session.TrustedEnrollmentEnabled)
            {
                string value = "검증된 개인 표본 " + session.CurrentTrustedSampleCount + "/3 · 연습 시도 " +
                    session.CurrentScoreableAttemptCount + "/10";
                if (session.CurrentConsecutiveTechnicalFailureCount > 0)
                    value += " · 기술 오류 " + session.CurrentConsecutiveTechnicalFailureCount + "/3";
                return value;
            }
            int step = 0;
            int index = 0;
            foreach (string word in session.Words)
            {
                index++;
                if (word == session.CurrentWord) step = index;
            }
            return "진행 " + step + "/" + session.Words.Count;
        }

        string QuestProgressText()
        {
            if (session.QuestCompleted) return "Quest GREETING \uC131\uACF5";
            if (session.QuestAuthoritativeOutcome.HasValue)
                return "Quest \uACB0\uACFC: " + session.QuestAuthoritativeOutcome.Value;
            return "NPC GREETING Quest";
        }

        string CurrentWordStepText()
        {
            int step = 0;
            int index = 0;
            foreach (string word in session.Words)
            {
                index++;
                if (word == session.CurrentWord) step = index;
            }
            return "현재 단어 " + step + "/" + session.Words.Count;
        }

        string CameraText()
        {
            if (session.CameraReady) return "카메라 준비됨";
            if (cameraRoute == null) return "카메라 준비 중";
            if (cameraRoute.CameraState == CandidateCameraStartupState.Failed)
                return "카메라 시작 실패";
            // frames are arriving; what is missing is a person in view, not the camera
            if (cameraRoute.CameraStarted && cameraRoute.HasReceivedFrame)
                return "카메라 준비됨 · 몸과 손 인식 대기 중";
            if (cameraRoute.CameraStarted)
                return "카메라 시작됨 · 첫 프레임 대기 중";
            return "카메라 준비 중";
        }

        string AutoPhaseStatusText()
        {
            if (session.State == PetRobotPracticeState.Completed)
                return session.FullHandRecordCollectionActive
                    ? (session.AllFullHandRecordsCollected
                        ? FullHandEnrollmentCopy.Collected + " · " +
                          session.CollectedFullHandWordCount + "/" + session.Words.Count +
                          " · 기록 " + session.FullHandRecordTotal
                        : "연습 기록 미완료 · " + session.CollectedFullHandWordCount + "/" +
                          session.Words.Count + " · 기록 " + session.FullHandRecordTotal)
                    : session.TrustedEnrollmentEnabled
                    ? (session.OverallEnrollmentGateReady
                        ? "Personalized · 전체 3/3/3 등록 완료"
                        : "PersonalizationIncomplete · 전체 3/3/3 Gate 잠김")
                    : "Completed · Mastered " + session.MasteredCount + " · NeedsPractice " +
                      session.NeedsPracticeCount;
            if (neutralAutoCaptureCoordinator == null) return CurrentWordStepText();
            switch (neutralAutoCaptureCoordinator.State)
            {
                case RawMediaPipeNeutralAutoSegmenterState.NeedsCalibration:
                case RawMediaPipeNeutralAutoSegmenterState.CalibratingNeutral:
                    return "기본 자세 " + Percent(neutralAutoCaptureCoordinator.CalibrationProgress);
                case RawMediaPipeNeutralAutoSegmenterState.WaitingForNeutral:
                    return "준비 자세 " + Percent(neutralAutoCaptureCoordinator.NeutralHoldProgress);
                case RawMediaPipeNeutralAutoSegmenterState.Armed:
                    return "자동 시작 준비됨";
                case RawMediaPipeNeutralAutoSegmenterState.Capturing:
                    return "수어 표현 중";
                case RawMediaPipeNeutralAutoSegmenterState.EndCandidate:
                    return "자동 종료 " + Percent(neutralAutoCaptureCoordinator.EndHoldProgress);
                case RawMediaPipeNeutralAutoSegmenterState.Evaluating:
                    return "인식 결과 확인 중";
                default:
                    return CurrentWordStepText();
            }
        }

        string OutcomeText()
        {
            GuidedQuestOutcome outcome = session.LastLearningOutcome;
            return outcome == null ? "결과 대기" : "결과 · " + outcome.Kind.ToString();
        }

        string CandidateText()
        {
            CandidateCountSnapshot counts = session.CandidateCounts;
            if (session.FullHandRecordCollectionActive)
            {
                var records = new System.Text.StringBuilder("연습 기록");
                foreach (string word in session.Words)
                    records.Append(' ').Append(session.GetFullHandRecordCount(word)).Append('/')
                           .Append(session.FullHandRequiredRecordCount);
                return records.ToString();
            }
            if (session.TrustedEnrollmentEnabled)
                return "Trusted G " + session.GetTrustedSampleCount("GREETING") + "/3 · L " +
                    session.GetTrustedSampleCount("LOVE") + "/3 · T " +
                    session.GetTrustedSampleCount("THANK_YOU") + "/3 · storage=" +
                    session.LastStorageResult.Status;
            return "Candidate " + session.LastStorageResult.Status + " · Eligible n/3  G " +
                counts.Eligible["GREETING"] + "/3 · L " + counts.Eligible["LOVE"] + "/3 · T " +
                counts.Eligible["THANK_YOU"] + "/3 · 총 " + counts.TotalCandidateCount;
        }

        void SynchronizeStoryFlow()
        {
            MoaStoryFeedbackKind feedback = MoaStoryFeedbackKind.None;
            if (questModeActive && session.QuestAuthoritativeOutcome.HasValue)
            {
                switch (session.QuestAuthoritativeOutcome.Value)
                {
                    case TrustedHybridOutcomeKind.StrictSuccess:
                        feedback = MoaStoryFeedbackKind.StrictSuccess;
                        break;
                    case TrustedHybridOutcomeKind.LowConfidenceCorrect:
                        feedback = MoaStoryFeedbackKind.LowConfidenceCorrect;
                        break;
                    case TrustedHybridOutcomeKind.WrongWord:
                        feedback = MoaStoryFeedbackKind.WrongWord;
                        break;
                    case TrustedHybridOutcomeKind.TechnicalFailure:
                        feedback = MoaStoryFeedbackKind.TechnicalFailure;
                        break;
                    case TrustedHybridOutcomeKind.NotDistinguishedRetry:
                        feedback = MoaStoryFeedbackKind.NotDistinguishedRetry;
                        break;
                    case TrustedHybridOutcomeKind.UnsupportedWord:
                        feedback = MoaStoryFeedbackKind.UnsupportedWord;
                        break;
                }
            }
            else if (!questModeActive)
            {
                GuidedQuestOutcome outcome = session.LastLearningOutcome;
                if (outcome != null)
                {
                    switch (outcome.Kind)
                    {
                        case GuidedQuestOutcomeKind.StrictSuccess:
                            feedback = MoaStoryFeedbackKind.StrictSuccess;
                            break;
                        case GuidedQuestOutcomeKind.LowConfidenceCorrect:
                            feedback = MoaStoryFeedbackKind.LowConfidenceCorrect;
                            break;
                        case GuidedQuestOutcomeKind.WrongWord:
                            feedback = MoaStoryFeedbackKind.WrongWord;
                            break;
                        case GuidedQuestOutcomeKind.TechnicalFailure:
                            feedback = MoaStoryFeedbackKind.TechnicalFailure;
                            break;
                        case GuidedQuestOutcomeKind.NotDistinguishedRetry:
                            feedback = MoaStoryFeedbackKind.NotDistinguishedRetry;
                            break;
                        case GuidedQuestOutcomeKind.UnsupportedWord:
                            feedback = MoaStoryFeedbackKind.UnsupportedWord;
                            break;
                    }
                }
            }
            storyFlow.Synchronize(
                session.CurrentWord,
                session.State,
                feedback,
                session.CanContinueWithNeedsPractice,
                session.CameraReady,
                neutralAutoCaptureCoordinator == null
                    ? RawMediaPipeNeutralAutoSegmenterState.Inactive
                    : neutralAutoCaptureCoordinator.State,
                neutralAutoCaptureCoordinator != null,
                session.TrustedEnrollmentEnabled,
                session.CurrentEnrollmentState,
                session.CurrentTrustedSampleCount,
                session.CurrentScoreableAttemptCount,
                session.CurrentConsecutiveTechnicalFailureCount,
                session.OverallEnrollmentGateReady,
                session.LastFullHandGameplayDecision != null,
                session.FullHandRecordView);
        }

        void SynchronizeDemonstration(MoaStoryPresentation story)
        {
            SignDemonstrationAvatarPresenter stage = DemonstrationStage();
            if (stage == null) return;
            if (story.Beat == MoaStoryBeat.WordIntroduction)
            {
                stage.ShowDemonstration(session.CurrentWord);
                stage.SetDemonstrationPlaybackPaused(false, false);
                return;
            }
            if (story.Beat == MoaStoryBeat.Welcome ||
                story.Beat == MoaStoryBeat.LessonOverview ||
                story.Beat == MoaStoryBeat.HowToUse ||
                story.Beat == MoaStoryBeat.Completed)
                stage.ClearDemonstration();
        }

        SignDemonstrationAvatarPresenter DemonstrationStage()
        {
            if (visualStage != null) return visualStage;
            return robot == null ? null : robot.DemonstrationSlot as SignDemonstrationAvatarPresenter;
        }

        string DiagnosticsText()
        {
            PracticeRecognitionAttempt attempt = session.LastAttempt;
            string value = "Diagnostics · capture=" + session.CaptureState +
                " · sessionFrames=" + session.CameraFrameCount +
                " · sessionObservations=" + session.ProviderObservationCount;
            // the camera owner's own counters, which differ from the session counters above
            value += cameraRoute != null
                ? "\n" + cameraRoute.BuildSafeStatusLine()
                : "\ncamera route not bound to this presenter";
            if (attempt != null)
                value += " · distance G/L/T=" + R(attempt.GreetingDistance) + "/" +
                    R(attempt.LoveDistance) + "/" + R(attempt.ThankYouDistance) +
                    " · threshold=" + R(attempt.ExpectedWordThreshold);
            if (visualStage != null)
            {
                Animator demonstrationAnimator = visualStage.DemonstrationAnimator;
                value += " · demoAnimator=" + visualStage.DemonstrationAnimatorPath +
                    " · clip=" + visualStage.CurrentClipName + "/" +
                    R(visualStage.CurrentClipLength) + "s" +
                    " · graph=" + visualStage.DemonstrationGraphValid +
                    " · time=" + R(visualStage.CurrentPlayableTime) +
                    " · yaw=" + R(visualStage.CurrentDemonstrationYaw);
                if (demonstrationAnimator != null)
                    value += " · demo=" + demonstrationAnimator.enabled + "/" +
                        demonstrationAnimator.cullingMode + "/" + demonstrationAnimator.updateMode;
                value += " · robotAnimator=" + visualStage.RobotAnimatorPath +
                    " · controller=" + visualStage.RobotControllerAssigned +
                    " · normalized=" + R(visualStage.RobotNormalizedTime);
            }
            return value;
        }

        void SetAllActionButtonsDisabled()
        {
            foreach (Button button in new[]
                     {
                         startButton, finishButton, retryButton, needsPracticeButton, nextButton, exitButton,
                         storyActionButton
                     })
                if (button != null) button.interactable = false;
        }

        RectTransform CreateRoundedPanel(
            string name,
            RectTransform parent,
            Rect rect,
            Color color,
            bool border)
        {
            GameObject value = new GameObject(name, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
            RectTransform transformValue = value.GetComponent<RectTransform>();
            PlaceTopLeft(transformValue, parent, rect);
            Image image = value.GetComponent<Image>();
            image.sprite = EnsureRoundedGraphicSource();
            image.type = Image.Type.Sliced;
            image.color = color;
            image.raycastTarget = false;
            if (border)
            {
                Outline outline = value.AddComponent<Outline>();
                outline.effectColor = BorderColor;
                outline.effectDistance = new Vector2(2f, -2f);
                outline.useGraphicAlpha = true;
            }
            return transformValue;
        }

        Text CreateText(
            string name,
            RectTransform parent,
            Rect rect,
            int fontSize,
            TextAnchor alignment,
            Color color,
            FontStyle style)
        {
            GameObject value = new GameObject(name, typeof(RectTransform), typeof(CanvasRenderer), typeof(Text));
            RectTransform transformValue = value.GetComponent<RectTransform>();
            PlaceTopLeft(transformValue, parent, rect);
            Text text = value.GetComponent<Text>();
            text.font = koreanFont != null
                ? koreanFont
                : Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            text.fontSize = fontSize;
            // F30: BMJUA is already a heavy display face; uGUI's simulated bold smears it,
            // so every label uses the font's intrinsic weight regardless of the requested style
            text.fontStyle = FontStyle.Normal;
            text.alignment = alignment;
            text.color = color;
            text.horizontalOverflow = HorizontalWrapMode.Wrap;
            text.verticalOverflow = VerticalWrapMode.Truncate;
            text.raycastTarget = false;
            return text;
        }

        Button CreateButton(
            string name,
            RectTransform parent,
            Rect rect,
            string label,
            UnityEngine.Events.UnityAction action,
            Color normalColor)
        {
            GameObject value = new GameObject(name, typeof(RectTransform), typeof(CanvasRenderer),
                typeof(Image), typeof(Button));
            RectTransform transformValue = value.GetComponent<RectTransform>();
            PlaceTopLeft(transformValue, parent, rect);
            Image image = value.GetComponent<Image>();
            image.sprite = EnsureRoundedGraphicSource();
            image.type = Image.Type.Sliced;
            image.color = Color.white;
            Outline outline = value.AddComponent<Outline>();
            outline.effectColor = new Color(BorderColor.r, BorderColor.g, BorderColor.b, 0.55f);
            outline.effectDistance = new Vector2(1.5f, -1.5f);
            Button button = value.GetComponent<Button>();
            ColorBlock colors = button.colors;
            colors.normalColor = normalColor;
            colors.highlightedColor = Color.Lerp(normalColor, Color.white, 0.18f);
            colors.pressedColor = Color.Lerp(normalColor, Color.black, 0.16f);
            colors.selectedColor = colors.highlightedColor;
            colors.disabledColor = new Color(0.16f, 0.19f, 0.22f, 0.65f);
            colors.colorMultiplier = 1f;
            colors.fadeDuration = 0.08f;
            button.colors = colors;
            button.targetGraphic = image;
            button.onClick.AddListener(action);
            Text text = CreateText(name + "Label", transformValue,
                new Rect(12f, 8f, rect.width - 24f, rect.height - 16f), 22,
                TextAnchor.MiddleCenter, Color.white, FontStyle.Bold);
            text.text = label;
            return button;
        }

        Sprite EnsureRoundedGraphicSource()
        {
            if (runtimeRoundedSprite != null) return runtimeRoundedSprite;

            runtimeRoundedTexture = new Texture2D(
                RoundedTextureSize, RoundedTextureSize, TextureFormat.RGBA32, false, true)
            {
                name = "Step10F_v1_3_RuntimeRoundedTexture",
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                hideFlags = HideFlags.HideAndDontSave
            };
            Color32[] pixels = new Color32[RoundedTextureSize * RoundedTextureSize];
            float half = RoundedTextureSize * 0.5f;
            float inner = half - RoundedCornerRadius;
            for (int y = 0; y < RoundedTextureSize; y++)
            {
                for (int x = 0; x < RoundedTextureSize; x++)
                {
                    float dx = Mathf.Max(Mathf.Abs(x + 0.5f - half) - inner, 0f);
                    float dy = Mathf.Max(Mathf.Abs(y + 0.5f - half) - inner, 0f);
                    float distance = Mathf.Sqrt(dx * dx + dy * dy);
                    byte alpha = (byte)Mathf.RoundToInt(
                        255f * Mathf.Clamp01(RoundedCornerRadius + 0.5f - distance));
                    pixels[y * RoundedTextureSize + x] = new Color32(255, 255, 255, alpha);
                }
            }
            runtimeRoundedTexture.SetPixels32(pixels);
            runtimeRoundedTexture.Apply(false, false);
            Vector4 border = new Vector4(
                RoundedCornerRadius, RoundedCornerRadius,
                RoundedCornerRadius, RoundedCornerRadius);
            runtimeRoundedSprite = Sprite.Create(
                runtimeRoundedTexture,
                new Rect(0f, 0f, RoundedTextureSize, RoundedTextureSize),
                new Vector2(0.5f, 0.5f),
                RoundedPixelsPerUnit,
                0,
                SpriteMeshType.FullRect,
                border);
            runtimeRoundedSprite.name = "Step10F_v1_3_RuntimeRounded9Slice";
            runtimeRoundedSprite.hideFlags = HideFlags.HideAndDontSave;
            return runtimeRoundedSprite;
        }

        void ReleaseRoundedGraphicResources()
        {
            if (runtimeRoundedSprite != null)
            {
                if (Application.isPlaying) Destroy(runtimeRoundedSprite);
                else DestroyImmediate(runtimeRoundedSprite);
            }
            if (runtimeRoundedTexture != null)
            {
                if (Application.isPlaying) Destroy(runtimeRoundedTexture);
                else DestroyImmediate(runtimeRoundedTexture);
            }
            runtimeRoundedSprite = null;
            runtimeRoundedTexture = null;
        }

        static void PlaceTopLeft(RectTransform value, RectTransform parent, Rect rect)
        {
            value.SetParent(parent, false);
            value.anchorMin = value.anchorMax = new Vector2(0f, 1f);
            value.pivot = new Vector2(0f, 1f);
            value.anchoredPosition = new Vector2(rect.x, -rect.y);
            value.sizeDelta = new Vector2(rect.width, rect.height);
        }

        static string R(double value) => value.ToString("R", CultureInfo.InvariantCulture);
        static string Percent(float value) => Mathf.RoundToInt(Mathf.Clamp01(value) * 100f) + "%";
    }
}
