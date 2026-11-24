using System;
using System.Globalization;
using System.IO;
using UnityEngine;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.PhysicalCompatibility;

namespace WordsInSilence.Recognition.LiveInput
{
    /// <summary>
    /// Minimal Step 9J technical verification panel. The approved Step 9C bridge must be
    /// assigned explicitly; this component never searches for, starts, stops, or owns a
    /// camera provider.
    /// </summary>
    public sealed class PersonalLiveRecognitionHarness : MonoBehaviour
    {
        const string LegacyProfileDirectoryName = "step9h_current_user_v1_0";

        [SerializeField] Step9CHandIdentityObservationBridge observationBridge;
        [SerializeField] PersonalLiveProfileMode profileMode = PersonalLiveProfileMode.LegacySingleMedoid;
        [SerializeField] Rect panelRect = new Rect(16f, 16f, 520f, 580f);
        [SerializeField] bool showTechnicalPanel = true;

        Step9CObservationSourceAdapter observationSource;
        PersonalLiveRecognitionController controller;
        int expectedWordIndex;
        Vector2 panelScrollPosition;

        public Step9CHandIdentityObservationBridge ObservationBridge => observationBridge;
        public PersonalLiveRecognitionController Controller => controller;
        public PersonalLiveProfileMode ProfileMode => profileMode;
        public GuidedQuestOutcome GuidedOutcome =>
            GuidedQuestOutcomeEvaluator.EvaluateForController(profileMode, controller);
        public bool TechnicalPanelVisible => showTechnicalPanel;

        public void SetTechnicalPanelVisible(bool visible) => showTechnicalPanel = visible;

        /// <summary>
        /// Rebuilds a controller that was created before the explicitly assigned Step 9C
        /// bridge finished binding. Runtime-instantiated candidate prefabs can observe that
        /// ordering even though every serialized reference is valid.
        /// </summary>
        public bool EnsureControllerReady()
        {
            if (controller != null && controller.State == PersonalLiveCaptureState.Ready &&
                controller.IsRuntimeReady)
                return true;
            if (!isActiveAndEnabled || observationBridge == null || !observationBridge.IsBound)
                return false;

            CreateController();
            return controller != null && controller.State == PersonalLiveCaptureState.Ready &&
                controller.IsRuntimeReady;
        }

        void OnEnable()
        {
            CreateController();
        }

        void OnDisable()
        {
            if (controller != null)
                controller.Disable();
            observationSource?.Dispose();
            observationSource = null;
        }

        void OnDestroy()
        {
            controller?.Dispose();
            controller = null;
            observationSource?.Dispose();
            observationSource = null;
        }

        public void SetExpectedGreeting() => SelectExpectedWord(0);
        public void SetExpectedLove() => SelectExpectedWord(1);
        public void SetExpectedThankYou() => SelectExpectedWord(2);

        public bool StartCapture()
        {
            return controller != null && controller.BeginCapture();
        }

        public bool FinalizeCapture()
        {
            return controller != null && controller.FinalizeCapture();
        }

        public bool CancelCapture()
        {
            return controller != null && controller.CancelCapture();
        }

        public bool ResetCapture()
        {
            return controller != null && controller.Reset();
        }

        void CreateController()
        {
            controller?.Dispose();
            observationSource?.Dispose();
            observationSource = observationBridge == null
                ? null
                : new Step9CObservationSourceAdapter(observationBridge);
            bool demoParticipantActive = DemoParticipantSession.IsActive;
            IPersonalLiveRuntimeFactory runtimeFactory;
            if (demoParticipantActive)
                runtimeFactory = new DemoParticipantLiveRuntimeFactory(DemoParticipantSession.Current);
            else
                runtimeFactory = profileMode == PersonalLiveProfileMode.MultiPrototypeCandidate
                    ? new ApprovedPersonalMultiPrototypeLiveRuntimeFactory()
                    : (IPersonalLiveRuntimeFactory)new ApprovedPersonalLiveRuntimeFactory();
            // F21: off by default. This reads no file and initializes no participant unless
            // the explicit environment opt-in is set, and it never starts, stops or owns a
            // camera and never changes the environment.
            WordsInSilence.Recognition.FullHandRuntime.IFullHandGameCaptureProcessor
                fullHandProcessor = FullHandGameCaptureConfiguration.CreateProcessorOrNull();
            controller = new PersonalLiveRecognitionController(
                observationSource,
                runtimeFactory,
                enableCandidateTransientAmbiguityTolerance:
                    profileMode == PersonalLiveProfileMode.MultiPrototypeCandidate,
                allowDeferredBridgeBinding:
                    profileMode == PersonalLiveProfileMode.MultiPrototypeCandidate,
                allowDemoParticipantRuntime: demoParticipantActive,
                fullHandProcessor: fullHandProcessor,
                // F27: read ONCE here; the controller then owns the mode for its lifetime.
                fullHandGameplayEnabled: WordsInSilence.Recognition.FullHandRuntime
                    .FullHandGameplayConfiguration.IsEnabled());
            controller.Initialize(BuildRuntimePaths(profileMode, demoParticipantActive));
            if (controller.State == PersonalLiveCaptureState.Ready)
                controller.SetExpectedWord(PersonalLiveRecognitionController.SupportedWords[expectedWordIndex]);
        }

        void SelectExpectedWord(int index)
        {
            if (index < 0 || index >= PersonalLiveRecognitionController.SupportedWords.Count)
                return;
            expectedWordIndex = index;
            if (controller != null && controller.State == PersonalLiveCaptureState.Ready)
                controller.SetExpectedWord(PersonalLiveRecognitionController.SupportedWords[index]);
        }

        static PersonalLiveRuntimePaths BuildRuntimePaths(
            PersonalLiveProfileMode mode,
            bool demoParticipantActive)
        {
            string root = Path.Combine(Application.streamingAssetsPath, "RecognitionCoreV1");
            if (demoParticipantActive)
            {
                // Public templates and the frozen threshold only; no fixed personal profile
                // location is named, so no fixed personal loader can ever run in this mode.
                return new PersonalLiveRuntimePaths(
                    Path.Combine(root, "Templates"),
                    Path.Combine(root, "RuntimeGate", "frozen_threshold_config.json"),
                    null,
                    allowNoPersonalProfileDirectory: true);
            }
            string profileDirectoryName = mode == PersonalLiveProfileMode.MultiPrototypeCandidate
                ? ApprovedPersonalMultiPrototypeLiveRuntimeFactory.ProfileDirectoryName
                : LegacyProfileDirectoryName;
            return new PersonalLiveRuntimePaths(
                Path.Combine(root, "Templates"),
                Path.Combine(root, "RuntimeGate", "frozen_threshold_config.json"),
                Path.Combine(root, "PersonalProfiles", profileDirectoryName));
        }

        void OnGUI()
        {
            if (!showTechnicalPanel) return;
            GUILayout.BeginArea(panelRect, GUI.skin.box);
            panelScrollPosition = GUILayout.BeginScrollView(panelScrollPosition);
            GUILayout.Label("Step 9J Personal Live Recognition");
            GUILayout.Label("Configured profile route: " + profileMode);
            GUILayout.Label("Explicit Step 9C bridge: " + (observationBridge == null ? "MISSING" : "ASSIGNED"));
            GUILayout.Label("Step 9C bridge LastError: " +
                (observationBridge == null ? "-" : EmptyAsDash(observationBridge.LastError)));
            GUILayout.Label("ProviderFrameCount: " + BridgeCount(bridge => bridge.ProviderFrameCount));
            GUILayout.Label("ObservationCount: " + BridgeCount(bridge => bridge.ObservationCount));
            GUILayout.Label("DuplicateFrameCount: " + BridgeCount(bridge => bridge.DuplicateFrameCount));
            GUILayout.Label("ResolvedDuplicateFrameCount: " +
                BridgeCount(bridge => bridge.ResolvedDuplicateFrameCount));
            GUILayout.Label("AmbiguousDuplicateFrameCount: " +
                BridgeCount(bridge => bridge.AmbiguousDuplicateFrameCount));

            GUILayout.BeginHorizontal();
            GUILayout.Label("Expected:", GUILayout.Width(76f));
            string[] words = { "GREETING", "LOVE", "THANK_YOU" };
            for (int index = 0; index < words.Length; index++)
            {
                bool selected = expectedWordIndex == index;
                if (GUILayout.Toggle(selected, words[index], "Button") && !selected)
                    SelectExpectedWord(index);
            }
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            bool previousGuiEnabled = GUI.enabled;
            GUI.enabled = previousGuiEnabled && controller != null &&
                controller.State == PersonalLiveCaptureState.Ready;
            if (GUILayout.Button("Start Capture")) StartCapture();
            GUI.enabled = previousGuiEnabled && controller != null &&
                controller.State == PersonalLiveCaptureState.Capturing;
            if (GUILayout.Button("Finalize")) FinalizeCapture();
            GUI.enabled = previousGuiEnabled && controller != null &&
                (controller.State == PersonalLiveCaptureState.Capturing ||
                 controller.State == PersonalLiveCaptureState.Processing);
            if (GUILayout.Button("Cancel")) CancelCapture();
            GUI.enabled = previousGuiEnabled && controller != null &&
                (controller.State == PersonalLiveCaptureState.Completed ||
                 controller.State == PersonalLiveCaptureState.Cancelled ||
                 controller.State == PersonalLiveCaptureState.Failed);
            if (GUILayout.Button("Reset")) ResetCapture();
            GUI.enabled = previousGuiEnabled;
            GUILayout.EndHorizontal();

            if (controller == null)
            {
                GUILayout.Label("Controller: unavailable");
                GUILayout.EndScrollView();
                GUILayout.EndArea();
                return;
            }

            GUILayout.Label("State: " + controller.State);
            GUILayout.Label("Capture/session ID: " + EmptyAsDash(controller.CaptureSessionId));
            GUILayout.Label("Expected word: " + controller.ExpectedWord);
            GUILayout.Label("Buffered raw frames: " + controller.BufferedRawFrameCount);
            GUILayout.Label("Personal mode active: " + controller.IsPersonalModeActive);
            GUILayout.Label("Personal profile ID: " + EmptyAsDash(controller.ActivePersonalProfileId));
            GUILayout.Label("Candidate ambiguity tolerance active: " +
                controller.CandidateTransientAmbiguityToleranceActive);
            GUILayout.Label("Tolerated ambiguity count (current capture): " +
                controller.ToleratedAmbiguousHandIdentityCount);

            PersonalLiveAmbiguityDiagnostic ambiguity =
                controller.LastToleratedAmbiguousHandIdentity;
            if (ambiguity != null)
            {
                GUILayout.Space(6f);
                GUILayout.Label("Last nonfatal ambiguity message: " + ambiguity.Message);
                GUILayout.Label("Last nonfatal ambiguity exception type: " +
                    EmptyAsDash(ambiguity.ExceptionType));
                GUILayout.Label("Last nonfatal ambiguity state: " + ambiguity.StateWhenRecorded);
                GUILayout.Label("Last nonfatal ambiguity capture/session ID: " +
                    EmptyAsDash(ambiguity.CaptureSessionId));
                GUILayout.Label("Last nonfatal ambiguity buffered raw frames: " +
                    ambiguity.BufferedRawFrameCount);
                GUILayout.Label("Last nonfatal ambiguity dropped during capture: " +
                    ambiguity.DroppedDuringCapture);
            }

            PersonalLivePrimaryFailureSnapshot primaryFailure = controller.PrimaryFailure;
            if (primaryFailure != null)
            {
                GUILayout.Space(6f);
                GUILayout.Label("Primary failure code: " + primaryFailure.Code);
                GUILayout.Label("Primary failure message: " + primaryFailure.Message);
                GUILayout.Label("Primary failure exception type: " + EmptyAsDash(primaryFailure.ExceptionType));
                GUILayout.Label("Primary state before failure: " + primaryFailure.StateBeforeFailure);
                GUILayout.Label("Primary capture/session ID: " + EmptyAsDash(primaryFailure.CaptureSessionId));
                GUILayout.Label("Primary buffered raw frames before failure: " +
                    primaryFailure.BufferedRawFrameCount);
            }

            if (controller.Failure != null)
            {
                GUILayout.Space(6f);
                GUILayout.Label("Current failure: " + controller.Failure.Code);
                GUILayout.Label("Current failure message: " + controller.Failure.Message);
            }

            PersonalLiveRecognitionSnapshot result = controller.Result;
            if (result != null)
            {
                GUILayout.Space(8f);
                GUILayout.Label("Predicted word: " + result.PredictedWord);
                GUILayout.Label("Accepted: " + result.Accepted);
                GUILayout.Label("Expected-class accept: " + result.ExpectedWordClassAccept);
                GUILayout.Label("Quest success: " + result.QuestSuccess);
                GUILayout.Label("GREETING distance: " + Distance(result.GreetingDistance));
                GUILayout.Label("LOVE distance: " + Distance(result.LoveDistance));
                GUILayout.Label("THANK_YOU distance: " + Distance(result.ThankYouDistance));
                GUILayout.Label("Best distance: " + Distance(result.BestDistance));
                GUILayout.Label("Winning prototype: " + result.WinningPrototypeId);
                GUILayout.Label("Frames raw/normalized: " + result.RawFrameCount + "/" + result.NormalizedFrameCount);
                GUILayout.Label("Landmarks valid/interpolated: " + result.ValidLandmarkCount + "/" + result.InterpolatedLandmarkCount);
                GUILayout.Label("GREETING narrow-margin warning: " + result.GreetingMarginNarrow);
            }

            GUILayout.Space(8f);
            GuidedQuestOutcome guidedOutcome = GuidedOutcome;
            GUILayout.Label("Guided quest outcome layer active: " +
                (guidedOutcome != null));
            if (guidedOutcome != null)
            {
                GUILayout.Label("Guided outcome (strict result unchanged): " + guidedOutcome.Kind);
                GUILayout.Label("Guided progress allowed: " + guidedOutcome.GuidedProgressAllowed);
                GUILayout.Label("Guided strict result available: " + guidedOutcome.StrictResultAvailable);
                GUILayout.Label("Guided strict Accepted preserved: " + guidedOutcome.StrictAccepted);
                GUILayout.Label("Guided strict Expected-class accept preserved: " +
                    guidedOutcome.StrictExpectedWordClassAccept);
                GUILayout.Label("Guided strict QuestSuccess preserved: " +
                    guidedOutcome.StrictQuestSuccess);
                GUILayout.Label("Guided outcome message: " + guidedOutcome.Message);
            }
            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }

        string BridgeCount(Func<Step9CHandIdentityObservationBridge, int> read) =>
            observationBridge == null ? "-" : read(observationBridge).ToString(CultureInfo.InvariantCulture);

        static string EmptyAsDash(string value) => string.IsNullOrEmpty(value) ? "-" : value;
        static string Distance(double value) => value.ToString("R", CultureInfo.InvariantCulture);
    }
}
