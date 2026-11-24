using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Text;
using UnityEngine;
using WordsInSilence.Recognition.PhysicalCompatibility;

namespace WordsInSilence.Recognition.Diagnostics
{
    public enum Step9DRawIntermediatePhysicalAcquisitionHarnessState
    {
        Blocked,
        Ready,
        Armed,
        Capturing,
        Failed,
        Completed,
        Disabled,
    }

    public enum Step9DRawIntermediatePhysicalAcquisitionFailureCode
    {
        NONE,
        BRIDGE_NOT_ASSIGNED,
        BRIDGE_NOT_BOUND,
        BRIDGE_FAILURE,
        CAPTURE_INTERRUPTED,
        INVALID_STATE,
        EMPTY_CAPTURE,
        DUPLICATE_CAPTURE_ID,
        INVALID_SHAPE,
        TIMESTAMP_REGRESSION,
        FRAME_IDENTITY_MISMATCH,
        NONFINITE_VALUE,
        OUTPUT_ALREADY_EXISTS,
        SCHEMA_VALIDATION_FAILED,
        CHECKSUM_VERIFICATION_FAILED,
        INFERENCE_GEOMETRY_MISSING,
        INFERENCE_GEOMETRY_CONTRACT_CHANGED,
        UNEXPECTED_FAILURE,
    }

    /// <summary>
    /// Step 9D diagnostic-only, opt-in acquisition harness. It does not own or start a camera,
    /// does not calculate recognition decisions, and performs no capture or disk I/O until the
    /// operator explicitly arms, starts and finalizes the current take.
    /// </summary>
    [DisallowMultipleComponent]
    [AddComponentMenu("WordsInSilence/Diagnostics/Step 9D Raw Intermediate Physical Acquisition Harness")]
    public sealed class Step9DRawIntermediatePhysicalAcquisitionHarness : MonoBehaviour
    {
        const string OutputDirectoryName = "Step9DRawIntermediatePhysicalAcquisition";

        static readonly string[] RequiredWords =
        {
            "GREETING",
            "LOVE",
            "THANK_YOU",
        };

        static readonly ReadOnlyCollection<string> ReadOnlyRequiredWords =
            Array.AsReadOnly(RequiredWords);

        [Header("Explicit diagnostic source")]
        [SerializeField] Step9CHandIdentityObservationBridge _bridge;
        [SerializeField] bool _showDiagnosticPanel = true;

        [Header("Read-only runtime status")]
        [SerializeField] Step9DRawIntermediatePhysicalAcquisitionHarnessState _harnessState =
            Step9DRawIntermediatePhysicalAcquisitionHarnessState.Blocked;
        [SerializeField] string _runId = string.Empty;
        [SerializeField] int _nextTakeIndex;
        [SerializeField] string _currentCaptureId = string.Empty;
        [SerializeField] string _lastCaptureId = string.Empty;
        [SerializeField] int _captureAttemptCount;
        [SerializeField] Step9DRawIntermediatePhysicalAcquisitionFailureCode _lastFailureCode =
            Step9DRawIntermediatePhysicalAcquisitionFailureCode.NONE;
        [SerializeField] string _lastFailureMessage = string.Empty;
        [SerializeField] string _failureRunId = string.Empty;
        [SerializeField] string _failureCaptureId = string.Empty;
        [SerializeField] string _lastExportPath = string.Empty;
        [SerializeField] string _lastSha256 = string.Empty;
        [SerializeField] long _lastFileSize;
        [SerializeField] bool _checksumVerified;

        Step9DRawIntermediateDiagnosticCollector _collector;
        Step9DInferenceImageGeometry _inferenceImageGeometry;
        bool _bridgeSubscribed;
        bool _initialized;
        int _toleratedDropCountThisTake;
        readonly List<Step9DAmbiguityDropRecord> _ambiguityDrops = new List<Step9DAmbiguityDropRecord>();
        Vector2 _guiScrollPosition;

        public Step9DRawIntermediatePhysicalAcquisitionHarnessState HarnessState => _harnessState;
        public IReadOnlyList<string> TakeOrder => ReadOnlyRequiredWords;
        public string RunId
        {
            get
            {
                EnsureInitialized();
                return _runId;
            }
        }
        public int CurrentTakeNumber => _nextTakeIndex < RequiredWords.Length
            ? _nextTakeIndex + 1
            : RequiredWords.Length;
        public string CurrentExpectedWord => _nextTakeIndex < RequiredWords.Length
            ? RequiredWords[_nextTakeIndex]
            : string.Empty;
        public string CurrentCaptureId => _currentCaptureId;
        public string LastCaptureId => _lastCaptureId;
        public int CaptureAttemptCount => _captureAttemptCount;
        public int BufferedObservationCount => _collector?.BufferedObservationCount ?? 0;
        public bool BridgeAssigned => _bridge != null;
        public bool BridgeBound => _bridge != null && _bridge.IsBound;
        public bool BridgeSubscribed => _bridgeSubscribed;
        public Step9DRawIntermediatePhysicalAcquisitionFailureCode LastFailureCode => _lastFailureCode;
        public string LastFailureMessage => _lastFailureMessage;
        public string FailureRunId => _failureRunId;
        public string FailureCaptureId => _failureCaptureId;
        public string LastExportPath => _lastExportPath;
        public string LastSha256 => _lastSha256;
        public long LastFileSize => _lastFileSize;
        public bool ChecksumVerified => _checksumVerified;
        public Step9DInferenceImageGeometry InferenceImageGeometry => _inferenceImageGeometry;
        public int ToleratedDropCountThisTake => _toleratedDropCountThisTake;
        public IReadOnlyList<Step9DAmbiguityDropRecord> AmbiguityDrops => _ambiguityDrops;

        /// <summary>
        /// Sets the inference image geometry to be recorded in all subsequent takes.
        /// Must be called before Arm(). Null clears a previously set geometry.
        /// </summary>
        public bool SetInferenceImageGeometry(Step9DInferenceImageGeometry geometry)
        {
            EnsureInitialized();
            if (_harnessState == Step9DRawIntermediatePhysicalAcquisitionHarnessState.Capturing)
                return RecordFailure(
                    Step9DRawIntermediatePhysicalAcquisitionFailureCode.INVALID_STATE,
                    "inference image geometry cannot change during an active capture");
            _inferenceImageGeometry = geometry?.Clone();
            return true;
        }

        public string RunOutputDirectory => Path.Combine(
            Application.persistentDataPath,
            OutputDirectoryName,
            RunId);
        public string PendingEvidencePath => string.IsNullOrWhiteSpace(_currentCaptureId)
            ? string.Empty
            : BuildEvidencePath(CurrentTakeNumber, CurrentExpectedWord, _currentCaptureId);

        void Awake()
        {
            EnsureInitialized();
        }

        void OnEnable()
        {
            EnsureInitialized();
            if (_harnessState == Step9DRawIntermediatePhysicalAcquisitionHarnessState.Failed ||
                _harnessState == Step9DRawIntermediatePhysicalAcquisitionHarnessState.Completed)
                return;
            _harnessState = AvailabilityState();
        }

        void OnDisable()
        {
            if (_harnessState == Step9DRawIntermediatePhysicalAcquisitionHarnessState.Capturing)
                RecordFailure(
                    Step9DRawIntermediatePhysicalAcquisitionFailureCode.CAPTURE_INTERRUPTED,
                    "component was disabled during an active diagnostic capture");
            DisposeCollectorBindings();
            if (_harnessState != Step9DRawIntermediatePhysicalAcquisitionHarnessState.Failed &&
                _harnessState != Step9DRawIntermediatePhysicalAcquisitionHarnessState.Completed)
                _harnessState = Step9DRawIntermediatePhysicalAcquisitionHarnessState.Disabled;
        }

        void OnDestroy()
        {
            DisposeCollectorBindings();
        }

        /// <summary>Explicitly assigns the existing Step 9C bridge. It never searches the scene.</summary>
        public bool AssignBridgeForDiagnostic(Step9CHandIdentityObservationBridge sourceBridge)
        {
            EnsureInitialized();
            if (_harnessState == Step9DRawIntermediatePhysicalAcquisitionHarnessState.Armed ||
                _harnessState == Step9DRawIntermediatePhysicalAcquisitionHarnessState.Capturing)
            {
                RecordFailure(
                    Step9DRawIntermediatePhysicalAcquisitionFailureCode.INVALID_STATE,
                    "bridge assignment cannot change after the harness is armed");
                return false;
            }

            ReleaseBridgeSubscriptions();
            _bridge = sourceBridge;
            if (_harnessState != Step9DRawIntermediatePhysicalAcquisitionHarnessState.Failed &&
                _harnessState != Step9DRawIntermediatePhysicalAcquisitionHarnessState.Completed)
                _harnessState = AvailabilityState();
            return _bridge != null;
        }

        public bool Arm()
        {
            EnsureInitialized();
            if (IsTerminal()) return false;
            if (!isActiveAndEnabled)
                return RecordFailure(
                    Step9DRawIntermediatePhysicalAcquisitionFailureCode.INVALID_STATE,
                    "the diagnostic harness must be active before it can be armed");
            if (_bridge == null)
                return RecordFailure(
                    Step9DRawIntermediatePhysicalAcquisitionFailureCode.BRIDGE_NOT_ASSIGNED,
                    "Step9CHandIdentityObservationBridge must be explicitly assigned");
            if (!_bridge.IsBound)
                return RecordFailure(
                    Step9DRawIntermediatePhysicalAcquisitionFailureCode.BRIDGE_NOT_BOUND,
                    "the explicitly assigned Step 9C bridge is not bound to its provider");
            if (_harnessState != Step9DRawIntermediatePhysicalAcquisitionHarnessState.Ready)
                return RecordFailure(
                    Step9DRawIntermediatePhysicalAcquisitionFailureCode.INVALID_STATE,
                    "the diagnostic harness is not ready to arm");

            try
            {
                _collector.AttachReadOnlyBridge(_bridge);
                _bridge.BridgeFailed += OnBridgeFailed;
                _bridgeSubscribed = true;
                _harnessState = Step9DRawIntermediatePhysicalAcquisitionHarnessState.Armed;
                return true;
            }
            catch (Exception exception)
            {
                return RecordFailure(
                    Step9DRawIntermediatePhysicalAcquisitionFailureCode.UNEXPECTED_FAILURE,
                    "bridge subscription failed: " + exception.Message);
            }
        }

        public bool StartCurrentTake()
        {
            EnsureInitialized();
            if (IsTerminal()) return false;
            if (_harnessState != Step9DRawIntermediatePhysicalAcquisitionHarnessState.Armed)
                return RecordFailure(
                    Step9DRawIntermediatePhysicalAcquisitionFailureCode.INVALID_STATE,
                    "Arm must succeed before starting the current take");
            if (_bridge == null || !_bridge.IsBound)
                return RecordFailure(
                    _bridge == null
                        ? Step9DRawIntermediatePhysicalAcquisitionFailureCode.BRIDGE_NOT_ASSIGNED
                        : Step9DRawIntermediatePhysicalAcquisitionFailureCode.BRIDGE_NOT_BOUND,
                    "the explicitly assigned Step 9C bridge is unavailable");
            if (_nextTakeIndex >= RequiredWords.Length)
                return RecordFailure(
                    Step9DRawIntermediatePhysicalAcquisitionFailureCode.INVALID_STATE,
                    "all required diagnostic takes are already complete");

            _currentCaptureId = Guid.NewGuid().ToString("N");
            _toleratedDropCountThisTake = 0;
            try
            {
                _collector.Begin(_runId, _currentCaptureId, CurrentExpectedWord, _inferenceImageGeometry,
                    requireGeometry: true);
                _captureAttemptCount++;
                _harnessState = Step9DRawIntermediatePhysicalAcquisitionHarnessState.Capturing;
                return true;
            }
            catch (Step9DRawIntermediateDiagnosticException exception)
            {
                return RecordFailure(MapDiagnosticFailure(exception.Code), exception.Message);
            }
            catch (Exception exception)
            {
                return RecordFailure(
                    Step9DRawIntermediatePhysicalAcquisitionFailureCode.UNEXPECTED_FAILURE,
                    "capture start failed: " + exception.Message);
            }
        }

        public bool FinalizeAndExport()
        {
            EnsureInitialized();
            if (IsTerminal()) return false;
            if (_harnessState != Step9DRawIntermediatePhysicalAcquisitionHarnessState.Capturing)
                return RecordFailure(
                    Step9DRawIntermediatePhysicalAcquisitionFailureCode.INVALID_STATE,
                    "Start Current Take must succeed before Finalize & Export");
            if (_bridge == null || !_bridge.IsBound)
                return RecordFailure(
                    _bridge == null
                        ? Step9DRawIntermediatePhysicalAcquisitionFailureCode.BRIDGE_NOT_ASSIGNED
                        : Step9DRawIntermediatePhysicalAcquisitionFailureCode.BRIDGE_NOT_BOUND,
                    "the explicitly assigned Step 9C bridge became unavailable during capture");

            try
            {
                Step9DRawIntermediateDiagnosticSnapshot snapshot = _collector.FinalizeCapture();
                var perTakeDrops = new List<Step9DAmbiguityDropRecord>();
                for (int i = 0; i < _ambiguityDrops.Count; i++)
                {
                    if (string.Equals(_ambiguityDrops[i].CaptureId, _currentCaptureId, StringComparison.Ordinal))
                        perTakeDrops.Add(_ambiguityDrops[i]);
                }
                snapshot.TransientHandIdentityDrops = perTakeDrops.AsReadOnly();
                string path = BuildEvidencePath(CurrentTakeNumber, CurrentExpectedWord, _currentCaptureId);
                Step9DRawIntermediateEvidenceReceipt receipt =
                    Step9DRawIntermediateEvidenceJson.Export(path, snapshot);
                bool verified = receipt.SchemaValidated &&
                    receipt.ChecksumVerified &&
                    Step9DRawIntermediateEvidenceJson.Verify(receipt.EvidencePath, receipt.ChecksumPath);
                if (!verified)
                    return RecordFailure(
                        Step9DRawIntermediatePhysicalAcquisitionFailureCode.CHECKSUM_VERIFICATION_FAILED,
                        "exported diagnostic evidence failed checksum verification");

                _lastExportPath = receipt.EvidencePath;
                _lastSha256 = receipt.Sha256;
                _lastFileSize = receipt.SizeBytes;
                _checksumVerified = true;
                _lastCaptureId = _currentCaptureId;
                _currentCaptureId = string.Empty;
                _nextTakeIndex++;

                if (_nextTakeIndex == RequiredWords.Length)
                {
                    _harnessState = Step9DRawIntermediatePhysicalAcquisitionHarnessState.Completed;
                    ReleaseBridgeSubscriptions();
                }
                else
                {
                    _harnessState = Step9DRawIntermediatePhysicalAcquisitionHarnessState.Armed;
                }
                return true;
            }
            catch (Step9DRawIntermediateDiagnosticException exception)
            {
                return RecordFailure(MapDiagnosticFailure(exception.Code), exception.Message);
            }
            catch (Exception exception)
            {
                return RecordFailure(
                    Step9DRawIntermediatePhysicalAcquisitionFailureCode.UNEXPECTED_FAILURE,
                    "finalize or export failed: " + exception.Message);
            }
        }

        void OnBridgeFailed(Exception exception)
        {
            if (_harnessState != Step9DRawIntermediatePhysicalAcquisitionHarnessState.Capturing ||
                _lastFailureCode != Step9DRawIntermediatePhysicalAcquisitionFailureCode.NONE)
                return;
            if (exception is Step9DRawIntermediateDiagnosticException diagnostic)
            {
                RecordFailure(MapDiagnosticFailure(diagnostic.Code), diagnostic.Message);
                return;
            }
            if (exception is Step9CAmbiguousHandIdentityException ambiguity)
            {
                if (_toleratedDropCountThisTake < 1)
                {
                    _toleratedDropCountThisTake++;
                    _ambiguityDrops.Add(new Step9DAmbiguityDropRecord(
                        _currentCaptureId,
                        _toleratedDropCountThisTake,
                        ambiguity.Diagnostic));
                    return;
                }
            }
            RecordFailure(
                Step9DRawIntermediatePhysicalAcquisitionFailureCode.BRIDGE_FAILURE,
                exception == null
                    ? "the Step 9C bridge reported an unknown failure"
                    : exception.GetType().Name + ": " + exception.Message);
        }

        void EnsureInitialized()
        {
            if (_initialized) return;
            _collector = new Step9DRawIntermediateDiagnosticCollector();
            _runId = Guid.NewGuid().ToString("N");
            _currentCaptureId = string.Empty;
            _lastCaptureId = string.Empty;
            _lastFailureMessage = string.Empty;
            _failureRunId = string.Empty;
            _failureCaptureId = string.Empty;
            _lastExportPath = string.Empty;
            _lastSha256 = string.Empty;
            _harnessState = AvailabilityState();
            _initialized = true;
        }

        Step9DRawIntermediatePhysicalAcquisitionHarnessState AvailabilityState()
        {
            return _bridge != null && _bridge.IsBound
                ? Step9DRawIntermediatePhysicalAcquisitionHarnessState.Ready
                : Step9DRawIntermediatePhysicalAcquisitionHarnessState.Blocked;
        }

        bool IsTerminal()
        {
            return _harnessState == Step9DRawIntermediatePhysicalAcquisitionHarnessState.Failed ||
                _harnessState == Step9DRawIntermediatePhysicalAcquisitionHarnessState.Completed;
        }

        bool RecordFailure(
            Step9DRawIntermediatePhysicalAcquisitionFailureCode code,
            string message)
        {
            if (_lastFailureCode == Step9DRawIntermediatePhysicalAcquisitionFailureCode.NONE)
            {
                _lastFailureCode = code;
                _lastFailureMessage = string.IsNullOrWhiteSpace(message) ? code.ToString() : message;
                _failureRunId = _runId;
                _failureCaptureId = _currentCaptureId;
            }
            _harnessState = Step9DRawIntermediatePhysicalAcquisitionHarnessState.Failed;
            _collector?.Cancel();
            ReleaseBridgeSubscriptions();
            return false;
        }

        void ReleaseBridgeSubscriptions()
        {
            if (_bridge != null && _bridgeSubscribed)
                _bridge.BridgeFailed -= OnBridgeFailed;
            _collector?.DetachReadOnlyBridge();
            _bridgeSubscribed = false;
        }

        void DisposeCollectorBindings()
        {
            ReleaseBridgeSubscriptions();
            _collector?.Dispose();
        }

        string BuildEvidencePath(int takeNumber, string expectedWord, string captureId)
        {
            string fileName = string.Format(
                "take_{0:D2}_{1}_{2}.json",
                takeNumber,
                expectedWord,
                captureId);
            return Path.Combine(RunOutputDirectory, fileName);
        }

        static Step9DRawIntermediatePhysicalAcquisitionFailureCode MapDiagnosticFailure(
            Step9DRawIntermediateDiagnosticFailureCode code)
        {
            switch (code)
            {
                case Step9DRawIntermediateDiagnosticFailureCode.INVALID_STATE:
                    return Step9DRawIntermediatePhysicalAcquisitionFailureCode.INVALID_STATE;
                case Step9DRawIntermediateDiagnosticFailureCode.EMPTY_CAPTURE:
                    return Step9DRawIntermediatePhysicalAcquisitionFailureCode.EMPTY_CAPTURE;
                case Step9DRawIntermediateDiagnosticFailureCode.DUPLICATE_CAPTURE_ID:
                    return Step9DRawIntermediatePhysicalAcquisitionFailureCode.DUPLICATE_CAPTURE_ID;
                case Step9DRawIntermediateDiagnosticFailureCode.INVALID_SHAPE:
                    return Step9DRawIntermediatePhysicalAcquisitionFailureCode.INVALID_SHAPE;
                case Step9DRawIntermediateDiagnosticFailureCode.TIMESTAMP_REGRESSION:
                    return Step9DRawIntermediatePhysicalAcquisitionFailureCode.TIMESTAMP_REGRESSION;
                case Step9DRawIntermediateDiagnosticFailureCode.FRAME_IDENTITY_MISMATCH:
                    return Step9DRawIntermediatePhysicalAcquisitionFailureCode.FRAME_IDENTITY_MISMATCH;
                case Step9DRawIntermediateDiagnosticFailureCode.NONFINITE_VALUE:
                    return Step9DRawIntermediatePhysicalAcquisitionFailureCode.NONFINITE_VALUE;
                case Step9DRawIntermediateDiagnosticFailureCode.OUTPUT_ALREADY_EXISTS:
                    return Step9DRawIntermediatePhysicalAcquisitionFailureCode.OUTPUT_ALREADY_EXISTS;
                case Step9DRawIntermediateDiagnosticFailureCode.SCHEMA_VALIDATION_FAILED:
                    return Step9DRawIntermediatePhysicalAcquisitionFailureCode.SCHEMA_VALIDATION_FAILED;
                case Step9DRawIntermediateDiagnosticFailureCode.CHECKSUM_VERIFICATION_FAILED:
                    return Step9DRawIntermediatePhysicalAcquisitionFailureCode.CHECKSUM_VERIFICATION_FAILED;
                case Step9DRawIntermediateDiagnosticFailureCode.INFERENCE_GEOMETRY_MISSING:
                    return Step9DRawIntermediatePhysicalAcquisitionFailureCode.INFERENCE_GEOMETRY_MISSING;
                case Step9DRawIntermediateDiagnosticFailureCode.INFERENCE_GEOMETRY_CONTRACT_CHANGED:
                    return Step9DRawIntermediatePhysicalAcquisitionFailureCode.INFERENCE_GEOMETRY_CONTRACT_CHANGED;
                default:
                    return Step9DRawIntermediatePhysicalAcquisitionFailureCode.UNEXPECTED_FAILURE;
            }
        }

        static string TruncateForDisplay(string value, int maxLength)
        {
            if (string.IsNullOrEmpty(value) || value.Length <= maxLength)
                return value ?? string.Empty;
            return value.Substring(0, maxLength) + "...";
        }

        void OnGUI()
        {
            if (!_showDiagnosticPanel) return;
            float areaHeight = Mathf.Min(Screen.height - 24f, 480f);
            GUILayout.BeginArea(new Rect(12f, 12f, 640f, areaHeight), GUI.skin.box);

            GUILayout.Label("Step 9D Raw/Intermediate Physical Acquisition (diagnostic only)");
            GUILayout.Label("Harness state: " + _harnessState + "   Take: " + CurrentTakeNumber + " / " + CurrentExpectedWord);

            GUILayout.BeginHorizontal();
            bool previousEnabled = GUI.enabled;
            GUI.enabled = _harnessState == Step9DRawIntermediatePhysicalAcquisitionHarnessState.Ready;
            if (GUILayout.Button("Arm")) Arm();
            GUI.enabled = _harnessState == Step9DRawIntermediatePhysicalAcquisitionHarnessState.Armed;
            if (GUILayout.Button("Start Current Take")) StartCurrentTake();
            GUI.enabled = _harnessState == Step9DRawIntermediatePhysicalAcquisitionHarnessState.Capturing;
            if (GUILayout.Button("Finalize & Export")) FinalizeAndExport();
            GUI.enabled = previousEnabled;
            GUILayout.EndHorizontal();

            _guiScrollPosition = GUILayout.BeginScrollView(_guiScrollPosition);
            GUILayout.Label("Run ID: " + RunId);
            GUILayout.Label("Capture ID: " + _currentCaptureId);
            GUILayout.Label("Buffered observations: " + BufferedObservationCount);
            GUILayout.Label("Bridge assigned / bound / subscribed: " + BridgeAssigned + " / " + BridgeBound + " / " + BridgeSubscribed);
            GUILayout.Label("Tolerated drops (this take): " + _toleratedDropCountThisTake
                + " / total: " + _ambiguityDrops.Count);
            GUILayout.Label("Last failure: " + _lastFailureCode + " - "
                + TruncateForDisplay(_lastFailureMessage, 120));
            GUILayout.Label("Last export: " + TruncateForDisplay(_lastExportPath, 80));
            GUILayout.Label("Last SHA-256: " + _lastSha256);
            GUILayout.Label("Last file size: " + _lastFileSize);
            GUILayout.Label("Checksum verified: " + _checksumVerified);
            GUILayout.EndScrollView();

            GUILayout.EndArea();
        }
    }

    public sealed class Step9DAmbiguityDropRecord
    {
        public string CaptureId { get; }
        public int DropOrdinal { get; }
        public int ProviderFrameIndex { get; }
        public double ProviderTimestampSeconds { get; }
        public string Category0 { get; }
        public double Score0 { get; }
        public string Category1 { get; }
        public double Score1 { get; }
        public string Reason { get; }

        public Step9DAmbiguityDropRecord(
            string captureId,
            int dropOrdinal,
            Step9CHandIdentityDiagnostic diagnostic)
        {
            CaptureId = captureId ?? string.Empty;
            DropOrdinal = dropOrdinal;
            if (diagnostic != null)
            {
                ProviderFrameIndex = diagnostic.ProviderFrameIndex;
                ProviderTimestampSeconds = diagnostic.ProviderTimestampSeconds;
                Category0 = diagnostic.SourceCategory0 ?? string.Empty;
                Score0 = diagnostic.SourceScore0;
                Category1 = diagnostic.SourceCategory1 ?? string.Empty;
                Score1 = diagnostic.SourceScore1;
                Reason = diagnostic.Reason ?? string.Empty;
            }
            else
            {
                Reason = string.Empty;
                Category0 = string.Empty;
                Category1 = string.Empty;
            }
        }
    }
}
