using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Recognition.PhysicalCompatibility
{
    /// <summary>Temporary Step 9C OnGUI helper; it never owns the camera and never saves the provider scene.</summary>
    public sealed class PhysicalWebcamCoordinateCompatibilityHarness : MonoBehaviour
    {
        [SerializeField] Step9CHandIdentityObservationBridge _bridge;
        [SerializeField] RawImage _existingPreview;
        [SerializeField] string _runId = Step9CPhysicalCompatibilityConstants.RunId;

        readonly List<RawMediaPipeObservation> observed = new List<RawMediaPipeObservation>();
        readonly Step9CCaptureDiagnosticLedger diagnosticLedger = new Step9CCaptureDiagnosticLedger();
        readonly Dictionary<Canvas, bool> hiddenCanvasStates = new Dictionary<Canvas, bool>();
        FixedPhysicalCaptureProtocol protocol;
        RuntimeRecognitionService service;
        ApprovedLivePreprocessingAdapter adapter;
        ManualDiagnosticCaptureController controller;
        PhysicalCaptureEvidenceRecorder recorder;
        PhysicalTakePlan plan;
        PhysicalCaptureEvidence evidence;
        PhysicalEvidenceExportReceipt receipt;
        PhysicalEvidenceResumeReport resumeReport;
        Step9CCaptureDiagnosticSnapshot finalizedDiagnostics;
        RawMediaPipeObservation lastObservation;
        DateTimeOffset captureUtc;
        DateTimeOffset captureLocal;
        bool previewWasObserved;
        bool providerReadinessWasObserved;
        string status = "Not initialized";
        Vector2 scrollPosition;
        bool panelExpanded;

        public string RunId => _runId;
        public bool RuntimeInitialized { get; private set; }
        public bool PhysicalDeviceReadyClaimed => CurrentWebCam() != null && CurrentWebCam().isPlaying && previewWasObserved && providerReadinessWasObserved;
        public string CurrentExpectedWord => protocol?.CurrentExpectedWord ?? string.Empty;
        public int CurrentTakeNumber => protocol?.CurrentTakeNumber ?? 0;
        public string Status => status;
        public PhysicalCaptureEvidence CurrentEvidence => evidence;
        public PhysicalEvidenceExportReceipt LastReceipt => receipt;
        public PhysicalEvidenceResumeReport ResumeReport => resumeReport;
        public ManualDiagnosticCaptureController Controller => controller;

        void OnEnable()
        {
            if (_bridge == null) _bridge = GetComponent<Step9CHandIdentityObservationBridge>();
            if (_bridge == null) _bridge = FindFirstObjectByType<Step9CHandIdentityObservationBridge>();
            if (_bridge != null)
            {
                _bridge.ObservationArrived += OnObservation;
                _bridge.HandIdentityDiagnosticObserved += OnHandIdentityDiagnostic;
                _bridge.BridgeFailed += OnBridgeFailed;
            }
        }

        void Start()
        {
            FindPreview();
            try
            {
                string root = Path.Combine(Application.streamingAssetsPath, "RecognitionCoreV1");
                InitializeRuntime(
                    Path.Combine(root, "Templates"),
                    Path.Combine(root, "RuntimeGate", "frozen_threshold_config.json"));
            }
            catch (Exception exception)
            {
                status = $"Runtime initialization failed: {exception.GetType().Name}: {exception.Message}";
            }
        }

        void Update()
        {
            if (_existingPreview == null) FindPreview();
            WebCamTexture texture = CurrentWebCam();
            if (texture != null && texture.isPlaying && texture.width > 16 && texture.height > 16) previewWasObserved = true;
            if (_bridge != null && _bridge.IsBound && _bridge.HasReceivedObservation) providerReadinessWasObserved = true;
        }

        void OnDisable()
        {
            if (_bridge != null)
            {
                _bridge.ObservationArrived -= OnObservation;
                _bridge.HandIdentityDiagnosticObserved -= OnHandIdentityDiagnostic;
                _bridge.BridgeFailed -= OnBridgeFailed;
            }
            RestoreHiddenCanvases();
        }

        public void InitializeRuntime(string templateDirectory, string thresholdPath)
        {
            if (!string.Equals(_runId, Step9CPhysicalCompatibilityConstants.RunId, StringComparison.Ordinal))
                throw new InvalidOperationException("serialized Step 9C run ID differs from the frozen capture plan");
            service = RuntimeRecognitionService.Load(templateDirectory, thresholdPath);
            if (service.Threshold != Step9CPhysicalCompatibilityConstants.FrozenThreshold)
                throw new InvalidOperationException("frozen threshold changed");
            adapter = new ApprovedLivePreprocessingAdapter();
            controller = new ManualDiagnosticCaptureController(service, adapter);
            recorder = new PhysicalCaptureEvidenceRecorder(service, adapter);
            protocol = new FixedPhysicalCaptureProtocol(_runId);
            resumeReport = PhysicalCaptureEvidenceResume.VerifyReadOnly(EvidenceOutputDirectory());
            foreach (RecoveredPhysicalCapture recovered in resumeReport.RecoveredValidCaptures)
                protocol.RestoreVerifiedExport(recovered.CaptureId, recovered.TakeNumber, recovered.ExpectedWord);
            RuntimeInitialized = true;
            LogResumeState();
        }

        public PhysicalTakePlan PrepareWord()
        {
            EnsureRuntime();
            if (controller.State != ManualDiagnosticCaptureState.Idle) throw new InvalidOperationException("reset the prior capture before preparing a word");
            plan = protocol.PrepareCurrent();
            evidence = null;
            receipt = null;
            observed.Clear();
            finalizedDiagnostics = null;
            diagnosticLedger.Prepare(CurrentBoundary());
            lastObservation = null;
            status = $"Prepared Take {plan.TakeNumber}: {plan.ExpectedWord}; press Start explicitly.";
            Debug.Log($"[Step9C] Prepared current word: Take {plan.TakeNumber} {plan.ExpectedWord}; capture ID={plan.CaptureId}");
            return plan;
        }

        public void StartCapture()
        {
            EnsureRuntime();
            if (plan == null) throw new InvalidOperationException("press Prepare word before Start");
            if (controller.State != ManualDiagnosticCaptureState.Idle) throw new InvalidOperationException("capture state must be Idle before Start");
            observed.Clear();
            finalizedDiagnostics = null;
            lastObservation = null;
            previewWasObserved = IsPreviewReady();
            providerReadinessWasObserved = _bridge != null && _bridge.IsBound && _bridge.HasReceivedObservation;
            _bridge?.ResetObservationState();
            captureUtc = DateTimeOffset.UtcNow;
            captureLocal = DateTimeOffset.Now;
            controller.Begin(plan.CaptureId, RawObservationConvention.LiveMediaPipe);
            diagnosticLedger.Begin(CurrentBoundary());
            status = $"Capturing Take {plan.TakeNumber}: {plan.ExpectedWord}; perform once, then Finalize.";
            Step9CProviderBoundary boundary = CurrentBoundary();
            Debug.Log($"[Step9C] Capture started: Take {plan.TakeNumber} {plan.ExpectedWord}; next lifetime provider frame={boundary.LifetimeProviderFrameCount}; previous provider timestamp={boundary.LastLifetimeProviderTimestampSeconds:R}; perform the sign exactly once, then press Finalize.");
        }

        public PhysicalCaptureEvidence FinalizeCapture()
        {
            EnsureRuntime();
            if (plan == null) throw new InvalidOperationException("no take is prepared");
            DiagnosticRecognitionResult result = null;
            try
            {
                result = controller.FinalizeCapture(plan.CaptureId);
            }
            catch (Exception exception)
            {
                diagnosticLedger.TryRecordFailure(exception);
            }
            finalizedDiagnostics = diagnosticLedger.Finalize(CurrentBoundary());
            evidence = BuildEvidence(result);
            string typedFailure = TypedFailureText();
            string recognition = evidence.Recognition == null
                ? "recognition=NONE"
                : $"predicted={evidence.Recognition.PredictedWord}, accepted={evidence.Recognition.Accepted}";
            status = $"Finalized. Objective validity={ObjectiveValidityText()}; typed failure={typedFailure}; {recognition}. Export evidence before Reset.";
            Debug.Log($"[Step9C] Capture finalized: Take {evidence.TakeNumber} {evidence.ExpectedWord}; objective validity={ObjectiveValidityText()}; typed failure={typedFailure}; {recognition}.");
            return evidence;
        }

        public PhysicalCaptureEvidence CancelCapture()
        {
            EnsureRuntime();
            if (plan == null) throw new InvalidOperationException("no take is prepared");
            controller.Cancel(plan.CaptureId);
            diagnosticLedger.TryRecordFailure(typeof(OperationCanceledException).FullName, "User cancelled the manual capture");
            finalizedDiagnostics = diagnosticLedger.Finalize(CurrentBoundary());
            evidence = BuildEvidence(null);
            status = "Capture cancelled; export typed evidence before Reset.";
            Debug.Log($"[Step9C] Capture cancelled: Take {evidence.TakeNumber} {evidence.ExpectedWord}; typed failure={TypedFailureText()}.");
            return evidence;
        }

        public PhysicalEvidenceExportReceipt ExportEvidence()
        {
            EnsureRuntime();
            if (evidence == null) throw new InvalidOperationException("Finalize or Cancel before Export evidence");
            if (receipt != null) throw new InvalidOperationException("this capture has already been exported");
            string fileName = $"take_{evidence.TakeNumber:D2}_{evidence.ExpectedWord}_{evidence.CaptureId}.json";
            string path = Path.Combine(EvidenceOutputDirectory(), fileName);
            receipt = PhysicalCaptureEvidenceFile.Export(path, evidence);
            protocol.CommitExport(evidence.CaptureId, evidence.PreExportObjectivelyValid, receipt.HashVerified);
            status = evidence.PreExportObjectivelyValid && receipt.HashVerified
                ? $"Evidence verified: {receipt.EvidencePath}. Fixed take consumed regardless of prediction; Reset for next word."
                : $"Evidence verified but capture objectively invalid. Do not repeat automatically; exit Play Mode and report the hand-identity diagnostic. Reasons: {string.Join(",", evidence.ObjectiveInvalidReasons)}. Path: {receipt.EvidencePath}";
            Step9CProviderBoundary exportBoundary = CurrentBoundary();
            Debug.Log($"[Step9C] Evidence exported: path={receipt.EvidencePath}; SHA-256 verified={receipt.HashVerified}; objective validity={ObjectiveValidityText()}; typed failure={TypedFailureText()}; overall complete={protocol.IsComplete}; export after lifetime provider frame={exportBoundary.LastLifetimeProviderFrameIndex}; provider timestamp={exportBoundary.LastLifetimeProviderTimestampSeconds:R}. Completed capture snapshot was not modified.");
            return receipt;
        }

        public void ResetCapture()
        {
            EnsureRuntime();
            if (controller.State == ManualDiagnosticCaptureState.Capturing) throw new InvalidOperationException("Finalize or Cancel and export before Reset");
            if (evidence != null && receipt == null) throw new InvalidOperationException("Export evidence before Reset");
            if (plan != null && evidence == null)
                protocol.CommitExport(plan.CaptureId, objectivelyValid: false, evidenceHashVerified: false);
            controller.Reset();
            _bridge?.ResetObservationState();
            plan = null;
            evidence = null;
            receipt = null;
            observed.Clear();
            finalizedDiagnostics = null;
            diagnosticLedger.Reset();
            lastObservation = null;
            previewWasObserved = false;
            providerReadinessWasObserved = false;
            status = protocol.IsComplete
                ? "All three fixed captures exported. Exit Play Mode and do not save the provider scene."
                : $"Reset complete. Next fixed word: {protocol.CurrentExpectedWord}.";
            Debug.Log(protocol.IsComplete
                ? "[Step9C] Overall completion: ALL THREE FIXED CAPTURES EXPORTED. Exit Play Mode and do not save the provider scene."
                : $"[Step9C] Resume ready: Take {protocol.CurrentTakeNumber} {protocol.CurrentExpectedWord}.");
        }

        PhysicalCaptureEvidence BuildEvidence(DiagnosticRecognitionResult result)
        {
            var context = new PhysicalCaptureContext(
                plan,
                captureUtc == default ? DateTimeOffset.UtcNow : captureUtc,
                captureLocal == default ? DateTimeOffset.Now : captureLocal,
                Application.unityVersion,
                Step9CPhysicalCompatibilityConstants.MediaPipePackageVersion,
                SceneManager.GetActiveScene().name,
                DeviceSnapshot(finalizedDiagnostics),
                finalizedDiagnostics?.Window);
            return recorder.Create(
                context,
                observed.ToArray(),
                controller.State,
                result,
                finalizedDiagnostics?.FailureType,
                finalizedDiagnostics?.FailureMessage,
                finalizedDiagnostics?.Diagnostics);
        }

        PhysicalDeviceSnapshot DeviceSnapshot(Step9CCaptureDiagnosticSnapshot snapshot = null)
        {
            WebCamTexture texture = CurrentWebCam();
            bool opened = texture != null && texture.isPlaying && texture.width > 16 && texture.height > 16 && !string.IsNullOrWhiteSpace(texture.deviceName);
            return new PhysicalDeviceSnapshot(
                texture?.deviceName ?? string.Empty,
                texture?.requestedWidth ?? 0,
                texture?.requestedHeight ?? 0,
                texture?.requestedFPS ?? double.NaN,
                texture?.width ?? 0,
                texture?.height ?? 0,
                opened,
                previewWasObserved,
                _bridge != null && _bridge.IsBound,
                providerReadinessWasObserved,
                snapshot?.Window.BridgeObservationCount ?? _bridge?.ObservationCount ?? 0,
                snapshot?.Window.ProviderFrameCount ?? _bridge?.ProviderFrameCount ?? 0);
        }

        void OnObservation(RawMediaPipeObservation observation)
        {
            lastObservation = observation;
            if (controller == null || controller.State != ManualDiagnosticCaptureState.Capturing) return;
            observed.Add(observation);
            try
            {
                controller.AddObservation(plan.CaptureId, observation);
            }
            catch (Exception exception)
            {
                diagnosticLedger.TryRecordFailure(exception);
                status = $"Observation rejected: {exception.GetType().Name}: {exception.Message}";
            }
        }

        void OnBridgeFailed(Exception exception)
        {
            if (diagnosticLedger.TryRecordFailure(exception))
            {
                status = $"Provider bridge failed during current capture: {exception.GetType().Name}: {exception.Message}";
                Debug.LogWarning($"[Step9C] Capture typed provider failure observed: {exception.GetType().FullName}: {exception.Message}");
            }
            else Debug.LogWarning($"[Step9C] Live preview provider failure outside capture (not evidence): {exception.GetType().FullName}: {exception.Message}");
        }

        void OnHandIdentityDiagnostic(Step9CHandIdentityDiagnostic diagnostic)
        {
            bool includedInCapture = diagnosticLedger.TryRecordDiagnostic(diagnostic);
            Debug.Log($"[Step9C] Hand identity diagnostic: provider frame={diagnostic.ProviderFrameIndex}; " +
                      $"provider timestamp={diagnostic.ProviderTimestampSeconds:R}; hand timestamp={diagnostic.HandFrameTimestampSeconds:R}; " +
                      $"source={diagnostic.SourceCategory0}({diagnostic.SourceScore0:R}),{diagnostic.SourceCategory1}({diagnostic.SourceScore1:R}); " +
                      $"outcome={diagnostic.Outcome}; selected hand0={diagnostic.SelectedHand0Identity}; " +
                      $"direct={diagnostic.DirectAssignmentCost:R}; swapped={diagnostic.SwappedAssignmentCost:R}; " +
                      $"margin={diagnostic.AssignmentMargin:R}; max matched distance={diagnostic.BestAssignmentMaxDistance:R}; reason={diagnostic.Reason}; capture evidence event={includedInCapture}");
        }

        Step9CProviderBoundary CurrentBoundary() => _bridge?.SnapshotBoundary() ?? Step9CProviderBoundary.Empty;

        void FindPreview()
        {
            if (_existingPreview != null && _existingPreview.texture is WebCamTexture) return;
            foreach (RawImage image in FindObjectsByType<RawImage>(FindObjectsSortMode.None))
                if (image.texture is WebCamTexture) { _existingPreview = image; return; }
        }

        WebCamTexture CurrentWebCam() => _existingPreview != null ? _existingPreview.texture as WebCamTexture : null;
        bool IsPreviewReady()
        {
            WebCamTexture texture = CurrentWebCam();
            return texture != null && texture.isPlaying && texture.width > 16 && texture.height > 16;
        }

        static string EvidenceOutputDirectory()
        {
#if UNITY_EDITOR
            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            return Path.Combine(projectRoot, "data", "processed", "sessions", "S001_P001", "step9c_physical_webcam_coordinate_compatibility_v1_0", "physical_captures");
#else
            return Path.Combine(Application.persistentDataPath, "Step9CPhysicalWebcamCoordinateCompatibility", Step9CPhysicalCompatibilityConstants.RunId);
#endif
        }

        void OnGUI()
        {
            if (!panelExpanded)
            {
                Rect toolbar = Step9CPanelLayout.CollapsedToolbar(Screen.width, Screen.height);
                GUI.Box(toolbar, GUIContent.none);
                if (GUI.Button(new Rect(toolbar.x + 6f, toolbar.y + 6f, toolbar.width - 12f, toolbar.height - 12f), "Open Step 9C Capture Panel"))
                    SetPanelExpanded(true);
                return;
            }

            Step9CPanelLayout.CalculateExpanded(Screen.width, Screen.height, out Rect header, out Rect statusPanel, out Rect previewPanel);
            GUI.DrawTexture(new Rect(0f, 0f, Screen.width, Screen.height), Texture2D.blackTexture, ScaleMode.StretchToFill, false);
            var wrappedLabel = new GUIStyle(GUI.skin.label) { wordWrap = true };
            GUILayout.BeginArea(header, GUI.skin.box);
            GUILayout.BeginHorizontal();
            GUILayout.Label("WordsInSilence Step 9C — isolated capture panel", wrappedLabel);
            if (GUILayout.Button("Return to camera controls", GUILayout.Width(210f))) SetPanelExpanded(false);
            GUILayout.EndHorizontal();
            GUILayout.EndArea();

            GUILayout.BeginArea(statusPanel, GUI.skin.box);
            scrollPosition = GUILayout.BeginScrollView(scrollPosition);
            GUILayout.Label($"Run ID: {_runId}", wrappedLabel);
            GUILayout.Label($"CURRENT WORD: {(protocol == null || protocol.IsComplete ? "NONE" : $"Take {protocol.CurrentTakeNumber} {protocol.CurrentExpectedWord}")}", wrappedLabel);
            GUILayout.Label($"RECOVERED VALID CAPTURES: {RecoveredCaptureText()}", wrappedLabel);
            GUILayout.Label($"OVERALL COMPLETION: {(protocol != null && protocol.IsComplete ? "ALL THREE FIXED CAPTURES EXPORTED" : "INCOMPLETE — waiting for the current fixed take")}", wrappedLabel);
            WebCamTexture texture = CurrentWebCam();
            GUILayout.Label($"Camera: {(texture != null && texture.isPlaying ? $"OPEN {texture.deviceName} {texture.width}x{texture.height}" : "NOT OPEN")}", wrappedLabel);
            GUILayout.Label($"LIVE PREVIEW (not evidence): provider={(_bridge != null && _bridge.IsBound ? "BOUND" : "NOT BOUND")}; readiness={(_bridge != null && _bridge.HasReceivedObservation ? "LIVE RESULT" : "WAITING")}; provider frames={(_bridge?.ProviderFrameCount ?? 0)}; bridge observations={(_bridge?.ObservationCount ?? 0)}", wrappedLabel);
            GUILayout.Label($"LIVE PRESENCE (not evidence): pose={(lastObservation != null && AnyValid(lastObservation.Pose))}, left={(lastObservation != null && AnyValid(lastObservation.LeftHand))}, right={(lastObservation != null && AnyValid(lastObservation.RightHand))}", wrappedLabel);
            GUILayout.Label($"Capture state: {(controller == null ? "Unavailable" : controller.State.ToString())}; capture ID: {(plan?.CaptureId ?? "-")}; raw frames: {(evidence?.RawFrameCount ?? observed.Count)}", wrappedLabel);
            GUILayout.Label($"CAPTURE SNAPSHOT: provider frames={CaptureProviderFrameCount()}; bridge observations={CaptureObservationCount()}; duplicate={CaptureDuplicateCount()}; pose-resolved={CaptureResolvedDuplicateCount()}; ambiguous={CaptureAmbiguousDuplicateCount()}; classification={HandIdentityClassificationText()}", wrappedLabel);
            GUILayout.Label($"THIS CAPTURE OBJECTIVE VALIDITY: {ObjectiveValidityText()}", wrappedLabel);
            GUILayout.Label($"TYPED FAILURE: {TypedFailureText()}", wrappedLabel);
            if (evidence?.Recognition != null)
            {
                PhysicalRecognitionEvidence result = evidence.Recognition;
                GUILayout.Label($"Prediction: {result.PredictedWord}; winner: {result.WinningPrototypeId}", wrappedLabel);
                GUILayout.Label($"Distances: GREETING={result.GreetingDistance:R}; LOVE={result.LoveDistance:R}; THANK_YOU={result.ThankYouDistance:R}", wrappedLabel);
                GUILayout.Label($"Best={result.BestDistance:R}; threshold={result.Threshold:R}; accepted={result.Accepted}", wrappedLabel);
            }
            GUILayout.Label($"Evidence path: {(receipt?.EvidencePath ?? "not exported")}", wrappedLabel);
            GUILayout.Label($"STATUS: {status}", wrappedLabel);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Prepare word")) TryAction(() => PrepareWord());
            if (GUILayout.Button("Start")) TryAction(StartCapture);
            if (GUILayout.Button("Finalize")) TryAction(() => FinalizeCapture());
            if (GUILayout.Button("Cancel")) TryAction(() => CancelCapture());
            if (GUILayout.Button("Reset")) TryAction(ResetCapture);
            if (GUILayout.Button("Export evidence")) TryAction(() => ExportEvidence());
            GUILayout.EndHorizontal();
            GUILayout.EndScrollView();
            GUILayout.EndArea();

            GUILayout.BeginArea(previewPanel, GUI.skin.box);
            GUILayout.Label("STEP 9C PREVIEW — existing scene texture, no second webcam", wrappedLabel);
            Step9CHandIdentityDiagnostic latest = LastCaptureDiagnostic();
            if (latest == null) GUILayout.Label("Last capture duplicate: NONE", wrappedLabel);
            else
            {
                GUILayout.Label($"Last capture duplicate frame={latest.ProviderFrameIndex}; t={latest.ProviderTimestampSeconds:R}", wrappedLabel);
                GUILayout.Label($"Source: {latest.SourceCategory0} score={latest.SourceScore0:R}; {latest.SourceCategory1} score={latest.SourceScore1:R}", wrappedLabel);
                GUILayout.Label($"Outcome: {latest.Outcome}; reason={latest.Reason}", wrappedLabel);
            }
            Step9CHandIdentityDiagnostic liveLatest = _bridge?.LastHandIdentityDiagnostic;
            if (liveLatest != null && !ReferenceEquals(liveLatest, latest))
                GUILayout.Label($"Live preview duplicate (not evidence): frame={liveLatest.ProviderFrameIndex}; t={liveLatest.ProviderTimestampSeconds:R}; outcome={liveLatest.Outcome}; reason={liveLatest.Reason}", wrappedLabel);
            if (texture != null)
            {
                Rect previewRect = GUILayoutUtility.GetAspectRect(16f / 10f, GUILayout.ExpandWidth(true), GUILayout.ExpandHeight(true));
                GUI.DrawTexture(previewRect, texture, ScaleMode.ScaleToFit, false);
            }
            else GUILayout.Label("Return to camera controls, start the existing camera, then reopen this panel.", wrappedLabel);
            GUILayout.EndArea();
        }

        void SetPanelExpanded(bool expanded)
        {
            if (panelExpanded == expanded) return;
            panelExpanded = expanded;
            if (expanded)
            {
                hiddenCanvasStates.Clear();
                foreach (Canvas canvas in FindObjectsByType<Canvas>(FindObjectsInactive.Include, FindObjectsSortMode.None))
                {
                    hiddenCanvasStates[canvas] = canvas.enabled;
                    canvas.enabled = false;
                }
            }
            else RestoreHiddenCanvases();
        }

        void RestoreHiddenCanvases()
        {
            foreach (KeyValuePair<Canvas, bool> item in hiddenCanvasStates)
                if (item.Key != null) item.Key.enabled = item.Value;
            hiddenCanvasStates.Clear();
        }

        void TryAction(Action action)
        {
            try { action(); }
            catch (Exception exception)
            {
                status = $"{exception.GetType().Name}: {exception.Message}";
                Debug.LogWarning($"[Step9C] UI action failed: {status}");
            }
        }

        void LogResumeState()
        {
            Debug.Log($"[Step9C] Evidence read-only verification: VERIFIED; JSON={resumeReport.JsonFileCount}; SHA-256={resumeReport.ChecksumFileCount}; no evidence file was written.");
            foreach (RecoveredPhysicalCapture recovered in resumeReport.RecoveredValidCaptures)
                Debug.Log($"[Step9C] Recovered valid Take {recovered.TakeNumber} {recovered.ExpectedWord}: {recovered.CaptureId}");
            status = protocol.IsComplete
                ? "Existing evidence contains all three verified fixed captures."
                : $"Resume ready: Take {protocol.CurrentTakeNumber} {protocol.CurrentExpectedWord}. Confirm preview/provider, then press Prepare word.";
            Debug.Log(protocol.IsComplete
                ? "[Step9C] Overall completion recovered from verified existing evidence."
                : $"[Step9C] Resume ready: Take {protocol.CurrentTakeNumber} {protocol.CurrentExpectedWord}.");
        }

        string RecoveredCaptureText()
        {
            if (resumeReport == null) return "NOT CHECKED";
            if (resumeReport.RecoveredValidCaptures.Count == 0) return "NONE";
            var items = new List<string>(resumeReport.RecoveredValidCaptures.Count);
            foreach (RecoveredPhysicalCapture recovered in resumeReport.RecoveredValidCaptures)
                items.Add($"Take {recovered.TakeNumber} {recovered.ExpectedWord}={recovered.CaptureId}");
            return string.Join(" | ", items);
        }

        string ObjectiveValidityText()
        {
            if (evidence == null) return "NOT FINALIZED";
            return evidence.PreExportObjectivelyValid
                ? "VALID"
                : $"INVALID ({string.Join(",", evidence.ObjectiveInvalidReasons)})";
        }

        string TypedFailureText()
        {
            if (evidence?.Failure != null) return $"{evidence.Failure.ExceptionType}: {evidence.Failure.Message}";
            if (!string.IsNullOrEmpty(diagnosticLedger.FailureType)) return $"{diagnosticLedger.FailureType}: {diagnosticLedger.FailureMessage}";
            return "NONE";
        }

        string HandIdentityClassificationText()
        {
            if (evidence != null)
            {
                if (evidence.AmbiguousDuplicateHandednessFrameCount > 0) return "AMBIGUOUS — INVALID";
                if (evidence.ObjectiveInvalidReasons.Contains("duplicate_handedness_not_transient")) return "STRUCTURAL RATE — INVALID";
                if (evidence.ResolvedDuplicateHandednessFrameCount > 0) return "TRANSIENT, POSE-WRIST RESOLVED";
                return "NO DUPLICATE";
            }
            IReadOnlyList<Step9CHandIdentityDiagnostic> diagnostics = diagnosticLedger.CurrentDiagnostics;
            if (diagnostics.Count == 0) return "NO DUPLICATE OBSERVED";
            IReadOnlyList<string> invalid = Step9CHandIdentityCapturePolicy.InvalidReasons(CaptureProviderFrameCount(), diagnostics);
            if (invalid.Contains("ambiguous_hand_identity_observed")) return "AMBIGUOUS — WILL BE INVALID";
            if (invalid.Contains("duplicate_handedness_not_transient")) return "STRUCTURAL RATE — WILL BE INVALID";
            return "TRANSIENT, POSE-WRIST RESOLVED";
        }

        int CaptureProviderFrameCount() => evidence?.Device.ProviderFrameCount ?? finalizedDiagnostics?.Window.ProviderFrameCount ?? (diagnosticLedger.IsCapturing ? _bridge?.ProviderFrameCount ?? 0 : 0);
        int CaptureObservationCount() => evidence?.Device.BridgeObservationCount ?? finalizedDiagnostics?.Window.BridgeObservationCount ?? (diagnosticLedger.IsCapturing ? observed.Count : 0);
        int CaptureDuplicateCount() => evidence?.DuplicateHandednessFrameCount ?? finalizedDiagnostics?.Diagnostics.Count ?? diagnosticLedger.DiagnosticCount;
        int CaptureResolvedDuplicateCount() => evidence?.ResolvedDuplicateHandednessFrameCount ?? CountOutcome(CurrentCaptureDiagnostics(), Step9CHandIdentityOutcome.ResolvedByPoseWrist);
        int CaptureAmbiguousDuplicateCount() => evidence?.AmbiguousDuplicateHandednessFrameCount ?? CountOutcome(CurrentCaptureDiagnostics(), Step9CHandIdentityOutcome.Ambiguous);
        IReadOnlyList<Step9CHandIdentityDiagnostic> CurrentCaptureDiagnostics() => evidence?.HandIdentityDiagnostics ?? finalizedDiagnostics?.Diagnostics ?? diagnosticLedger.CurrentDiagnostics;
        Step9CHandIdentityDiagnostic LastCaptureDiagnostic()
        {
            IReadOnlyList<Step9CHandIdentityDiagnostic> diagnostics = CurrentCaptureDiagnostics();
            return diagnostics.Count == 0 ? null : diagnostics[diagnostics.Count - 1];
        }
        static int CountOutcome(IReadOnlyList<Step9CHandIdentityDiagnostic> diagnostics, Step9CHandIdentityOutcome outcome)
        {
            int count = 0;
            foreach (Step9CHandIdentityDiagnostic diagnostic in diagnostics) if (diagnostic.Outcome == outcome) count++;
            return count;
        }

        void EnsureRuntime()
        {
            if (!RuntimeInitialized || controller == null || recorder == null || protocol == null)
                throw new InvalidOperationException("runtime service is not initialized");
        }

        static bool AnyValid(RawLandmarkObservationSet set)
        {
            for (int landmark = 0; landmark < set.LandmarkCount; landmark++) if (set.IsValid(landmark)) return true;
            return false;
        }
    }

    public static class Step9CPanelLayout
    {
        const float Margin = 12f;
        const float Gap = 12f;
        const float HeaderHeight = 40f;

        public static Rect CollapsedToolbar(int screenWidth, int screenHeight)
        {
            float width = Mathf.Min(250f, Mathf.Max(120f, screenWidth - 16f));
            float height = Mathf.Min(42f, Mathf.Max(24f, screenHeight - 16f));
            return new Rect(Mathf.Max(8f, screenWidth - width - 8f), 8f, width, height);
        }

        public static void CalculateExpanded(int screenWidth, int screenHeight, out Rect header, out Rect statusPanel, out Rect previewPanel)
        {
            float width = Mathf.Max(1f, screenWidth);
            float height = Mathf.Max(1f, screenHeight);
            header = new Rect(Margin, 8f, Mathf.Max(1f, width - Margin * 2f), HeaderHeight);
            float bodyY = header.yMax + Gap;
            float bodyHeight = Mathf.Max(1f, height - bodyY - Margin);
            float contentWidth = Mathf.Max(1f, width - Margin * 2f);
            float available = Mathf.Max(1f, contentWidth - Gap);
            float leftWidth = Mathf.Clamp(available * 0.58f, Mathf.Min(280f, available), Mathf.Min(500f, available));
            float rightWidth = Mathf.Max(1f, available - leftWidth);
            statusPanel = new Rect(Margin, bodyY, leftWidth, bodyHeight);
            previewPanel = new Rect(statusPanel.xMax + Gap, bodyY, rightWidth, bodyHeight);
        }
    }
}
