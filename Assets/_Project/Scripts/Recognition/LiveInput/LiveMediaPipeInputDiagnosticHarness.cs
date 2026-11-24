using System;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Recognition.LiveInput
{
    /// <summary>Isolated OnGUI diagnostic harness. It does not open or own the existing camera pipeline.</summary>
    public sealed class LiveMediaPipeInputDiagnosticHarness : MonoBehaviour
    {
        [SerializeField] MediaPipeHandPoseObservationBridge _bridge;
        [SerializeField] RawImage _existingPreview;

        string status = "Not initialized";
        string captureId = string.Empty;
        int leftPresenceFrames;
        int rightPresenceFrames;

        public bool RuntimeInitialized { get; private set; }
        public string Status => status;
        public ManualDiagnosticCaptureController Controller { get; private set; }

        void OnEnable()
        {
            if (_bridge == null) _bridge = GetComponent<MediaPipeHandPoseObservationBridge>();
            if (_bridge == null) _bridge = FindFirstObjectByType<MediaPipeHandPoseObservationBridge>();
            if (_bridge != null)
            {
                _bridge.ObservationArrived += OnObservation;
                _bridge.BridgeFailed += OnBridgeFailed;
            }
        }

        void Start()
        {
            if (_existingPreview == null)
            {
                foreach (RawImage image in FindObjectsByType<RawImage>(FindObjectsSortMode.None))
                    if (image.texture is WebCamTexture) { _existingPreview = image; break; }
            }
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

        void OnDisable()
        {
            if (_bridge != null)
            {
                _bridge.ObservationArrived -= OnObservation;
                _bridge.BridgeFailed -= OnBridgeFailed;
            }
        }

        public void InitializeRuntime(string templateDirectory, string thresholdPath)
        {
            RuntimeRecognitionService service = RuntimeRecognitionService.Load(templateDirectory, thresholdPath);
            Controller = new ManualDiagnosticCaptureController(service, new ApprovedLivePreprocessingAdapter());
            RuntimeInitialized = true;
            status = "Runtime ready; explicit Start Capture required";
        }

        public void BeginCapture(string requestedCaptureId = null)
        {
            EnsureRuntime();
            if (Controller.State != ManualDiagnosticCaptureState.Idle) throw new InvalidOperationException("reset the prior diagnostic capture before beginning another");
            captureId = string.IsNullOrWhiteSpace(requestedCaptureId) ? Guid.NewGuid().ToString("D") : requestedCaptureId;
            leftPresenceFrames = 0;
            rightPresenceFrames = 0;
            _bridge?.ResetObservationState();
            Controller.Begin(captureId, RawObservationConvention.LiveMediaPipe);
            status = "Capturing; use Finalize, Cancel or Reset explicitly";
        }

        public DiagnosticRecognitionResult FinalizeCapture()
        {
            EnsureRuntime();
            try
            {
                DiagnosticRecognitionResult result = Controller.FinalizeCapture(captureId);
                status = $"Completed: {result.PredictedWord}, best={result.BestDistance:R}, accepted={result.Accepted}";
                return result;
            }
            catch (Exception exception)
            {
                status = $"Capture failed: {exception.GetType().Name}: {exception.Message}";
                throw;
            }
        }

        public void CancelCapture()
        {
            EnsureRuntime();
            Controller.Cancel(captureId);
            status = "Capture cancelled";
        }

        public void ResetCapture()
        {
            EnsureRuntime();
            Controller.Reset();
            _bridge?.ResetObservationState();
            captureId = string.Empty;
            leftPresenceFrames = 0;
            rightPresenceFrames = 0;
            status = "Runtime ready; capture reset";
        }

        void OnObservation(RawMediaPipeObservation observation)
        {
            if (Controller == null || Controller.State != ManualDiagnosticCaptureState.Capturing) return;
            try
            {
                if (AnyValid(observation.LeftHand)) leftPresenceFrames++;
                if (AnyValid(observation.RightHand)) rightPresenceFrames++;
                Controller.AddObservation(captureId, observation);
            }
            catch (Exception exception)
            {
                status = $"Observation rejected: {exception.GetType().Name}: {exception.Message}";
            }
        }

        void OnBridgeFailed(Exception exception) => status = $"Provider bridge failed: {exception.GetType().Name}: {exception.Message}";

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(12, 12, 560, 470), GUI.skin.box);
            GUILayout.Label("WordsInSilence Step 9B Live MediaPipe Diagnostic");
            GUILayout.Label($"Provider bridge: {(_bridge != null && _bridge.IsBound ? "BOUND" : "NOT BOUND")}");
            GUILayout.Label($"Provider readiness: {(_bridge != null && _bridge.HasReceivedObservation ? "RESULT RECEIVED" : "WAITING/UNVERIFIED")}");
            GUILayout.Label("Source: raw WebCamTexture; preview UV mirror is display-only; handedness categoryName is direct");
            GUILayout.Label($"Capture state: {(Controller == null ? "Unavailable" : Controller.State.ToString())}");
            GUILayout.Label($"Capture ID: {(string.IsNullOrEmpty(captureId) ? "-" : captureId)}");
            GUILayout.Label($"Raw frames: {(Controller == null ? 0 : Controller.RawFrameCount)}; left-present: {leftPresenceFrames}; right-present: {rightPresenceFrames}");
            if (Controller?.Result != null)
            {
                DiagnosticRecognitionResult result = Controller.Result;
                GUILayout.Label($"Normalized frames: {result.NormalizedFrameCount}");
                GUILayout.Label($"Prediction: {result.PredictedWord}; prototype: {result.WinningPrototypeId}");
                GUILayout.Label($"Best: {result.BestDistance:R}; threshold: {result.Threshold:R}; accepted: {result.Accepted}");
            }
            GUILayout.Label($"Status: {status}");
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Start Capture")) TryAction(() => BeginCapture());
            if (GUILayout.Button("Finalize")) TryAction(() => FinalizeCapture());
            if (GUILayout.Button("Cancel")) TryAction(CancelCapture);
            if (GUILayout.Button("Reset")) TryAction(ResetCapture);
            GUILayout.EndHorizontal();
            if (_existingPreview != null && _existingPreview.texture != null)
                GUI.DrawTexture(new Rect(12, 270, 320, 180), _existingPreview.texture, ScaleMode.ScaleToFit, false);
            else
                GUI.Label(new Rect(12, 290, 520, 40), "Camera preview unavailable here; use the existing 02_PoseIntegration RawImage provider scene.");
            GUILayout.EndArea();
        }

        void TryAction(Action action)
        {
            try { action(); }
            catch (Exception exception) { status = $"{exception.GetType().Name}: {exception.Message}"; }
        }

        void EnsureRuntime()
        {
            if (!RuntimeInitialized || Controller == null) throw new InvalidOperationException("runtime service is not initialized");
        }

        static bool AnyValid(RawLandmarkObservationSet set)
        {
            for (int landmark = 0; landmark < set.LandmarkCount; landmark++) if (set.IsValid(landmark)) return true;
            return false;
        }
    }
}
