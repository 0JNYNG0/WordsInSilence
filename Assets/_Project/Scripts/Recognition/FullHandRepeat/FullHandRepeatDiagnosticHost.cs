// F7R4 - the diagnostic repeat host. Supersedes the F7R2 host; the F7R1 coordinator is
// unchanged.
//
// STARTUP ORDER. The first real smoke run faulted with CAMERA_ROUTE_NOT_WIRED during
// OnEnable, before the camera route's own Awake/Start had run; the same player then started
// its camera normally and received frames with bridgeBound=false. Dependency work therefore
// no longer happens in OnEnable. It happens at a deterministic later boundary - Start, and
// afterwards the next Update - so the whole scene has completed Awake and OnEnable first.
// The route is NOT waited on for IsReady before binding: IsReady depends on a bound bridge
// observation, so waiting for it before binding would deadlock.
//
// Inert unless WIS_FULL_HAND_REPEAT=1 AND the existing observer opt-in, bundle manifest and
// output root are all valid. It binds the approved technical prefab exactly as the existing
// in-world runtime owner does and then stops: no practice session, repository, quest,
// enrollment, save, PlayerPrefs or neutral recalibration, no camera or provider of its own,
// no model score on screen.
using System;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;
using WordsInSilence.Pose;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.FullHandRuntime;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Recognition.FullHandRepeat
{
    /// <summary>
    /// The one terminal-state rule, kept pure so it can be tested without a scene: a take
    /// that is still open while its controller has already reached a terminal state must be
    /// closed once, through the coordinator's existing safe close.
    /// </summary>
    public static class FullHandRepeatTerminalWatch
    {
        public static bool ShouldCloseNow(bool captureActive, PersonalLiveCaptureState state)
        {
            return captureActive &&
                   (state == PersonalLiveCaptureState.Failed ||
                    state == PersonalLiveCaptureState.Cancelled ||
                    state == PersonalLiveCaptureState.Completed);
        }
    }

    public sealed class FullHandRepeatDiagnosticHost : MonoBehaviour
    {
        public const string RepeatVariable = "WIS_FULL_HAND_REPEAT";
        public const string RepeatValue = "1";

        public const string PhaseOff = "off";
        public const string PhasePendingBoundary = "pending_boundary";
        public const string PhasePendingRoute = "pending_route";
        public const string PhaseReady = "ready";
        public const string PhaseFaulted = "faulted";

        [Header("Explicit wiring")]
        [SerializeField] string candidateHarnessResource =
            "Diagnostics/PersonalMultiPrototypeLiveRecognitionTechnical";
        [SerializeField] PetRobotCandidateCameraStartupRoute cameraRoute;
        [SerializeField] HandPoseIntegrator providerIntegrator;
        [SerializeField] Behaviour[] suppressDuringRepeat;

        [Header("Timing")]
        [SerializeField] float countdownSeconds = 3f;
        [SerializeField] float maxCaptureSeconds = 12f;
        [SerializeField] float routeActivationWaitSeconds = 10f;

        [Header("Display labels only. Nothing branches on a word.")]
        [SerializeField] string[] wordDisplayLabels = { "인사", "사랑", "고맙습니다" };

        static readonly string[] ConditionLabels =
            { "정상 수행", "가만히 있기", "잘못된 손모양", "다른 동작" };
        static readonly string[] PartitionLabels = { "연습 기록", "본 기록" };

        /// <summary>Test seam. Production reads the real environment.</summary>
        public static Func<string, string> EnvironmentForTest;

        GameObject harnessInstance;
        PersonalLiveRecognitionHarness harness;
        FullHandRepeatCoordinator coordinator;
        IFullHandObservationSink resolvedSink;
        readonly List<KeyValuePair<Behaviour, bool>> suppressed =
            new List<KeyValuePair<Behaviour, bool>>();

        bool inputSuppressed;
        bool cursorCaptured;
        bool bridgeBound;
        bool cleanedUp;
        bool boundaryReached;
        bool initializationPending;
        bool configurationValidated;
        float routeWaited;
        string phase = PhaseOff;
        string loggedPhase = string.Empty;
        bool previousCursorVisible;
        CursorLockMode previousCursorLock;

        int wordIndex, conditionIndex, partitionIndex;
        float countdownRemaining = -1f;
        float captureElapsed;
        string setupFault = string.Empty;
        string lastReason = string.Empty;
        string outcomeMessage = string.Empty;
        Vector2 scroll;

        public bool Armed { get; private set; }
        public string SetupFault { get { return setupFault; } }
        public string SetupPhase { get { return phase; } }
        public bool InitializationPendingForTest { get { return initializationPending; } }
        public bool InputSuppressedForTest { get { return inputSuppressed; } }
        public bool CursorCapturedForTest { get { return cursorCaptured; } }
        public bool BridgeBoundForTest { get { return bridgeBound; } }
        public bool HarnessInstantiatedForTest { get { return harnessInstance != null; } }

        /// <summary>Build-time wiring for the in-build scene copy only.</summary>
        public void WireForBuild(PetRobotCandidateCameraStartupRoute route,
                                 HandPoseIntegrator integrator, Behaviour[] suppress)
        {
            cameraRoute = route;
            providerIntegrator = integrator;
            suppressDuringRepeat = suppress;
        }

        public bool HasWiring
        {
            get
            {
                return cameraRoute != null && providerIntegrator != null &&
                       suppressDuringRepeat != null && suppressDuringRepeat.Length > 0;
            }
        }

        static bool RepeatRequested()
        {
            try
            {
                Func<string, string> read =
                    EnvironmentForTest ?? Environment.GetEnvironmentVariable;
                return string.Equals(read(RepeatVariable), RepeatValue, StringComparison.Ordinal);
            }
            catch (Exception) { return false; }
        }

        void Awake()
        {
            // suppression happens in Awake, before the world components' Start can run and
            // begin a practice session in repeat mode
            if (!RepeatRequested()) return;
            SuppressWorldInput();
            CaptureCursor();
        }

        void OnEnable()
        {
            cleanedUp = false;
            Armed = false;
            setupFault = string.Empty;
            routeWaited = 0f;
            if (!RepeatRequested()) { phase = PhaseOff; return; }
            // nothing is loaded, bound or resolved here: the scene has not finished its own
            // Awake and OnEnable yet, which is exactly what the first smoke run tripped over
            SuppressWorldInput();
            CaptureCursor();
            initializationPending = true;
            SetPhase(PhasePendingBoundary);
        }

        void Start()
        {
            boundaryReached = true;                 // the whole scene has now been enabled
            TryInitializeIfPending();
        }

        void OnDisable() { Cleanup(); }

        void OnDestroy() { Cleanup(); }

        /// <summary>
        /// Idempotent, and it releases only what this component acquired. An active take is
        /// closed first so its metadata is preserved exactly once.
        /// </summary>
        void Cleanup()
        {
            if (cleanedUp) return;
            cleanedUp = true;
            try
            {
                if (coordinator != null && coordinator.CaptureActive)
                {
                    string reason;
                    coordinator.TryCancel(out reason);
                }
            }
            catch (Exception) { }
            coordinator = null;
            try
            {
                if (bridgeBound && cameraRoute != null) cameraRoute.UnbindObservationBridge();
            }
            catch (Exception) { }
            bridgeBound = false;
            try { if (harnessInstance != null) DestroyImmediate(harnessInstance); }
            catch (Exception) { }
            harnessInstance = null;
            harness = null;
            resolvedSink = null;
            configurationValidated = false;
            initializationPending = false;
            RestoreWorldInput();
            ReleaseCursor();
            countdownRemaining = -1f;
            captureElapsed = 0f;
            routeWaited = 0f;
            Armed = false;
            if (phase != PhaseFaulted) SetPhase(PhaseOff);
        }

        // ---- the deterministic later boundary ----------------------------------

        void TryInitializeIfPending()
        {
            if (!initializationPending || Armed) return;
            if (!boundaryReached) return;
            if (!RepeatRequested()) { initializationPending = false; return; }

            // a serialized null reference is a distinct, immediate fault: waiting could
            // never fix it. It is checked first, before any file or model work.
            if (cameraRoute == null) { FailSetup("CAMERA_ROUTE_NOT_ASSIGNED"); return; }
            if (providerIntegrator == null) { FailSetup("INTEGRATOR_NOT_ASSIGNED"); return; }

            if (!configurationValidated)
            {
                IFullHandObservationSink sink;
                if (!TryValidateConfiguration(out sink)) return;   // FailSetup already ran
                resolvedSink = sink;
                configurationValidated = true;      // validated once, never per frame
            }

            // assigned but not yet active is a PENDING state, not an unwired one. The route
            // opens its camera asynchronously; IsReady additionally needs a bound bridge, so
            // binding must not wait for it.
            if (!cameraRoute.isActiveAndEnabled)
            {
                SetPhase(PhasePendingRoute);
                routeWaited += Time.unscaledDeltaTime;
                if (routeActivationWaitSeconds > 0f && routeWaited >= routeActivationWaitSeconds)
                    FailSetup("CAMERA_ROUTE_NOT_ACTIVE");
                return;
            }

            if (!TryBindHarness()) return;          // FailSetup already ran
            coordinator = new FullHandRepeatCoordinator(new HarnessPort(harness), resolvedSink,
                                                        line => Debug.Log(line), null);
            initializationPending = false;
            Armed = true;
            SetPhase(PhaseReady);
        }

        void FailSetup(string code)
        {
            setupFault = code;
            initializationPending = false;
            SetPhase(PhaseFaulted);
            Cleanup();                              // hand back only what we acquired
            phase = PhaseFaulted;                   // Cleanup must not hide the fault phase
        }

        /// <summary>One safe status line per phase change: no path, message or data.</summary>
        void SetPhase(string next)
        {
            phase = next;
            if (loggedPhase == next) return;
            loggedPhase = next;
            if (next == PhaseOff) return;           // an OFF tool stays silent
            Debug.Log("[FullHandRepeat] setup=" + next
                      + " route_assigned=" + (cameraRoute != null)
                      + " route_enabled=" + (cameraRoute != null && cameraRoute.enabled)
                      + " route_active_in_hierarchy="
                      + (cameraRoute != null && cameraRoute.gameObject.activeInHierarchy)
                      + " integrator_assigned=" + (providerIntegrator != null)
                      + " bridge_bound=" + bridgeBound
                      + " fault=" + (string.IsNullOrEmpty(setupFault) ? "none" : setupFault));
        }

        bool TryValidateConfiguration(out IFullHandObservationSink sink)
        {
            sink = null;
            Func<string, string> read = EnvironmentForTest ?? Environment.GetEnvironmentVariable;
            try
            {
                if (!string.Equals(read(FullHandAttemptObserver.EnableVariable),
                                   FullHandAttemptObserver.EnableValue, StringComparison.Ordinal))
                { FailSetup("OBSERVER_NOT_ENABLED"); return false; }
                string manifest = read(FullHandAttemptObserver.ManifestVariable);
                if (!FullHandFileObservationSink.IsFullyQualified(manifest))
                { FailSetup("BUNDLE_MANIFEST_NOT_ABSOLUTE"); return false; }
                // the reviewed loader verifies the payload bytes and digest; no copy is made,
                // and this runs once, never per frame
                FullHandRuntimeBundle bundle = FullHandRuntimeBundle.Load(manifest);
                if (bundle == null) { FailSetup("BUNDLE_UNAVAILABLE"); return false; }
                sink = new FullHandFileObservationSink(read(FullHandAttemptObserver.OutputVariable));
                return true;
            }
            catch (FullHandFormatException error) { FailSetup(error.Code); return false; }
            catch (Exception error) { FailSetup("SETUP_" + error.GetType().Name); return false; }
        }

        bool TryBindHarness()
        {
            try
            {
                GameObject prefab = Resources.Load<GameObject>(candidateHarnessResource);
                if (prefab == null) { FailSetup("HARNESS_PREFAB_MISSING"); return false; }
                harnessInstance = Instantiate(prefab);
                harnessInstance.name = "FullHandRepeatRuntime";
                harness = harnessInstance.GetComponent<PersonalLiveRecognitionHarness>();
                if (harness == null) { FailSetup("HARNESS_COMPONENT_MISSING"); return false; }
                harness.SetTechnicalPanelVisible(false);
                if (harness.ObservationBridge == null)
                { FailSetup("HARNESS_BRIDGE_MISSING"); return false; }
                harness.ObservationBridge.BindForDiagnostic(providerIntegrator);
                if (!harness.ObservationBridge.IsBound)
                { FailSetup("BRIDGE_NOT_BOUND"); return false; }
                cameraRoute.BindObservationBridge(harness.ObservationBridge);
                bridgeBound = true;                 // only now is the binding ours to release
                return true;
            }
            catch (Exception error)
            {
                FailSetup("BIND_" + error.GetType().Name);
                return false;
            }
        }

        void SuppressWorldInput()
        {
            if (inputSuppressed || suppressDuringRepeat == null) return;
            foreach (Behaviour behaviour in suppressDuringRepeat)
            {
                if (behaviour == null) continue;
                suppressed.Add(new KeyValuePair<Behaviour, bool>(behaviour, behaviour.enabled));
                behaviour.enabled = false;
            }
            inputSuppressed = suppressed.Count > 0;
        }

        void RestoreWorldInput()
        {
            if (!inputSuppressed) return;
            foreach (KeyValuePair<Behaviour, bool> entry in suppressed)
                if (entry.Key != null) entry.Key.enabled = entry.Value;
            suppressed.Clear();
            inputSuppressed = false;
        }

        void CaptureCursor()
        {
            if (cursorCaptured) return;
            previousCursorVisible = Cursor.visible;
            previousCursorLock = Cursor.lockState;
            Cursor.visible = true;
            Cursor.lockState = CursorLockMode.None;
            cursorCaptured = true;
        }

        void ReleaseCursor()
        {
            if (!cursorCaptured) return;
            Cursor.visible = previousCursorVisible;
            Cursor.lockState = previousCursorLock;
            cursorCaptured = false;
        }

        bool SelectionLocked
        {
            get
            {
                return countdownRemaining >= 0f ||
                       (coordinator != null && coordinator.CaptureActive);
            }
        }

        /// <summary>
        /// Real camera evidence, checked AFTER the bridge is bound and BEFORE the
        /// coordinator's one destructive readiness preparation.
        /// </summary>
        bool CameraEvidenceReady
        {
            get
            {
                return cameraRoute != null && cameraRoute.IsReady &&
                       cameraRoute.HasReceivedFrame && cameraRoute.ProviderObservationReady;
            }
        }

        bool InputReady
        {
            get
            {
                if (coordinator == null || coordinator.CaptureActive) return false;
                if (!CameraEvidenceReady) return false;
                return coordinator.EnsureReadyForNextTake() && coordinator.ReadyToBegin;
            }
        }

        void Update()
        {
            if (initializationPending) { TryInitializeIfPending(); return; }
            if (!Armed || coordinator == null) return;
            if (countdownRemaining >= 0f)
            {
                countdownRemaining -= Time.unscaledDeltaTime;
                if (countdownRemaining <= 0f) { countdownRemaining = -1f; Begin(); }
                return;
            }
            if (!coordinator.CaptureActive) return;

            // a source failure ends the capture without the person pressing anything; close
            // it once through the same safe path, which never overwrites the original failure
            if (FullHandRepeatTerminalWatch.ShouldCloseNow(true, CurrentState))
            {
                Finish(false);
                return;
            }
            captureElapsed += Time.unscaledDeltaTime;
            if (maxCaptureSeconds > 0f && captureElapsed >= maxCaptureSeconds) Finish(true);
        }

        PersonalLiveCaptureState CurrentState
        {
            get
            {
                try
                {
                    return harness == null || harness.Controller == null
                        ? PersonalLiveCaptureState.Idle : harness.Controller.State;
                }
                catch (Exception) { return PersonalLiveCaptureState.Idle; }
            }
        }

        void Begin()
        {
            captureElapsed = 0f;
            outcomeMessage = string.Empty;
            string reason;
            coordinator.TryBegin(SelectedWord(),
                                 FullHandRepeatContract.Conditions[conditionIndex],
                                 FullHandRepeatContract.Partitions[partitionIndex],
                                 out reason);
            lastReason = reason;
        }

        void Finish(bool automatic)
        {
            string reason;
            bool ok = coordinator.TryFinish(automatic, out reason);
            lastReason = reason;
            captureElapsed = 0f;
            outcomeMessage = ok
                ? "기록되었습니다. 다음 시도는 '다시 하기'를 눌러 주세요."
                : "이번 시도는 정상적으로 끝나지 않았습니다. '다시 하기'를 눌러 주세요.";
        }

        string SelectedWord()
        {
            var words = PersonalLiveRecognitionController.SupportedWords;
            int index = wordIndex < 0 || wordIndex >= words.Count ? 0 : wordIndex;
            return words[index];
        }

        string DisplayLabel(int index, IReadOnlyList<string> words)
        {
            if (wordDisplayLabels != null && index < wordDisplayLabels.Length &&
                !string.IsNullOrEmpty(wordDisplayLabels[index]))
                return wordDisplayLabels[index];
            return words[index];
        }

        void OnGUI()
        {
            if (!RepeatRequested()) return;
            if (!Armed)
            {
                if (initializationPending)
                {
                    GUILayout.BeginArea(new Rect(16f, 16f, 520f, 110f), GUI.skin.box);
                    GUILayout.Label("준비 중입니다. 잠시만 기다려 주세요.");
                    GUILayout.EndArea();
                    return;
                }
                if (string.IsNullOrEmpty(setupFault)) return;
                GUILayout.BeginArea(new Rect(16f, 16f, 520f, 140f), GUI.skin.box);
                GUILayout.Label("측정 도구를 시작할 수 없습니다.");
                GUILayout.Label("담당자에게 아래 코드를 알려주세요: " + setupFault);
                GUILayout.EndArea();
                return;
            }

            // sized from the actual window, with a bounded minimum, so the preview and every
            // primary control fit at 1600x900 without scrolling
            float width = Mathf.Clamp(Screen.width - 32f, 480f, 900f);
            float height = Mathf.Clamp(Screen.height - 32f, 440f, 860f);
            float previewHeight = Mathf.Clamp(height * 0.42f, 180f, 360f);
            float previewWidth = previewHeight * 4f / 3f;
            GUILayout.BeginArea(new Rect(16f, 16f, width, height), GUI.skin.box);
            scroll = GUILayout.BeginScrollView(scroll);
            GUILayout.Label("손 동작 반복 측정");

            Texture preview = cameraRoute == null ? null : cameraRoute.PreviewTexture;
            if (preview != null)
                GUILayout.Label(preview, GUILayout.Width(previewWidth),
                                GUILayout.Height(previewHeight));
            else
                GUILayout.Label("카메라 화면이 아직 없습니다.", GUILayout.Height(previewHeight));

            bool locked = SelectionLocked;
            bool previous = GUI.enabled;
            GUI.enabled = previous && !locked;

            var words = PersonalLiveRecognitionController.SupportedWords;
            GUILayout.Label("어떤 단어를 하시겠어요?");
            GUILayout.BeginHorizontal();
            for (int index = 0; index < words.Count; index++)
            {
                bool selected = wordIndex == index;
                if (GUILayout.Toggle(selected, DisplayLabel(index, words), "Button") && !selected)
                    wordIndex = index;
            }
            GUILayout.EndHorizontal();

            GUILayout.Label("이번에는 어떻게 하실 건가요?");
            GUILayout.BeginHorizontal();
            for (int index = 0; index < FullHandRepeatContract.Conditions.Length; index++)
            {
                bool selected = conditionIndex == index;
                string label = index < ConditionLabels.Length ? ConditionLabels[index]
                             : FullHandRepeatContract.Conditions[index];
                if (GUILayout.Toggle(selected, label, "Button") && !selected)
                    conditionIndex = index;
            }
            GUILayout.EndHorizontal();

            GUILayout.Label("기록 종류");
            GUILayout.BeginHorizontal();
            for (int index = 0; index < FullHandRepeatContract.Partitions.Length; index++)
            {
                bool selected = partitionIndex == index;
                string label = index < PartitionLabels.Length ? PartitionLabels[index]
                             : FullHandRepeatContract.Partitions[index];
                if (GUILayout.Toggle(selected, label, "Button") && !selected)
                    partitionIndex = index;
            }
            GUILayout.EndHorizontal();
            GUI.enabled = previous;

            GUILayout.Space(4f);
            bool capturing = coordinator.CaptureActive;
            bool ready = !capturing && InputReady;
            if (countdownRemaining >= 0f)
                GUILayout.Label("준비하세요. " + Seconds(countdownRemaining) + " 후 시작합니다.");
            else if (capturing)
                GUILayout.Label("촬영 중입니다. 동작이 끝나면 '완료'를 누르세요.  "
                                + Seconds(captureElapsed) + " / " + Seconds(maxCaptureSeconds));
            else if (!string.IsNullOrEmpty(outcomeMessage))
                GUILayout.Label(outcomeMessage);
            else if (ready)
                GUILayout.Label("준비되었습니다. '시작'을 누르고 화면에서 손을 떼세요.");
            else
                GUILayout.Label("카메라와 손 인식을 준비하는 중입니다. 잠시 기다려 주세요.");

            if (!string.IsNullOrEmpty(lastReason))
                GUILayout.Label("안내: " + FriendlyReason(lastReason));
            if (coordinator.RecordingFaultCount > 0)
                GUILayout.Label("기록 저장에 실패한 시도가 있습니다. 담당자에게 알려주세요.");
            GUILayout.Label("시도 " + coordinator.AttemptCount
                            + "회, 저장 " + coordinator.RecordedCount + "회");

            GUILayout.Space(4f);
            GUILayout.BeginHorizontal();
            GUI.enabled = previous && !capturing && countdownRemaining < 0f && ready;
            if (GUILayout.Button("시작", GUILayout.Height(44f)))
            {
                lastReason = string.Empty;
                outcomeMessage = string.Empty;
                countdownRemaining = countdownSeconds <= 0f ? 0.01f : countdownSeconds;
            }
            GUI.enabled = previous && capturing;
            if (GUILayout.Button("완료", GUILayout.Height(44f))) Finish(false);
            GUI.enabled = previous && !capturing && countdownRemaining < 0f;
            if (GUILayout.Button("다시 하기", GUILayout.Height(44f)))
            {
                string reason;
                coordinator.TryPrepareNext(out reason);
                lastReason = reason;
                outcomeMessage = string.Empty;
            }
            GUI.enabled = previous;
            if (GUILayout.Button("시험 종료", GUILayout.Height(44f)))
            {
                Cleanup();
                Application.Quit();
            }
            GUILayout.EndHorizontal();
            GUILayout.Label("결과 점수는 화면에 표시하지 않습니다. 나중에 함께 확인합니다.");
            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }

        static string FriendlyReason(string code)
        {
            if (string.IsNullOrEmpty(code)) return string.Empty;
            if (code.StartsWith("START_", StringComparison.Ordinal))
                return "시작하지 못했습니다. '다시 하기'를 눌러 주세요.";
            if (code == "RECORDING_FAULT") return "기록 저장에 실패했습니다.";
            if (code == "CONTROLLER_NOT_READY" || code == "LIVE_INPUT_NOT_BOUND")
                return "카메라 준비가 끝나지 않았습니다.";
            if (code == "already_failed_before_close")
                return "촬영이 중단되었습니다. '다시 하기'를 눌러 주세요.";
            return "'다시 하기'를 눌러 주세요. (" + code + ")";
        }

        static string Seconds(float value)
        {
            return value.ToString("0.0", CultureInfo.InvariantCulture) + "초";
        }

        /// <summary>The production port over the existing harness and its controller.</summary>
        sealed class HarnessPort : IFullHandRepeatHarnessPort
        {
            readonly PersonalLiveRecognitionHarness harness;

            public HarnessPort(PersonalLiveRecognitionHarness harness)
            {
                this.harness = harness;
            }

            PersonalLiveRecognitionController Controller
            {
                get { return harness == null ? null : harness.Controller; }
            }

            public bool LiveInputBound
            {
                get
                {
                    return harness != null && harness.ObservationBridge != null &&
                           harness.ObservationBridge.IsBound;
                }
            }

            public bool EnsureControllerReady()
            {
                return harness != null && harness.EnsureControllerReady();
            }

            public bool TrySelectTarget(string word)
            {
                PersonalLiveRecognitionController controller = Controller;
                return controller != null && controller.SetExpectedWord(word);
            }

            public string EffectiveTarget
            {
                get
                {
                    PersonalLiveRecognitionController controller = Controller;
                    return controller == null ? string.Empty : controller.ExpectedWord;
                }
            }

            public PersonalLiveCaptureState State
            {
                get
                {
                    PersonalLiveRecognitionController controller = Controller;
                    return controller == null ? PersonalLiveCaptureState.Idle : controller.State;
                }
            }

            public string CaptureSessionId
            {
                get
                {
                    PersonalLiveRecognitionController controller = Controller;
                    return controller == null ? string.Empty : controller.CaptureSessionId;
                }
            }

            public int BufferedRawFrameCount
            {
                get
                {
                    PersonalLiveRecognitionController controller = Controller;
                    return controller == null ? -1 : controller.BufferedRawFrameCount;
                }
            }

            public bool StartCapture() { return harness != null && harness.StartCapture(); }
            public bool FinalizeCapture() { return harness != null && harness.FinalizeCapture(); }
            public bool CancelCapture() { return harness != null && harness.CancelCapture(); }
            public bool ResetCapture() { return harness != null && harness.ResetCapture(); }
        }
    }
}
