#if UNITY_EDITOR
using System;
using UnityEngine;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.Diagnostics;

namespace WordsInSilence.Recognition.LiveInput
{
    /// <summary>
    /// Step 10P v1.0c2 developer-only live validation seam: one GREETING take and one
    /// THANK_YOU take through the approved live capture and the frozen common-2D
    /// classifier, with nothing persisted.
    ///
    /// Editor only. The whole file is compiled out of a player build, so no shipped code
    /// path can reach it.
    ///
    /// All orchestration lives in <see cref="CommonSignalTwoTargetValidationCoordinator"/>:
    /// this component owns no target state and never calls the adapter directly, so the
    /// transactional begin and the single-invocation finalize proven offline are exactly
    /// what runs here.
    ///
    /// What it deliberately does NOT do:
    ///  - it never constructs <see cref="PetRobotSignPracticeSession"/>, so the normal
    ///    learning progression, word order, mastery rule and enrollment are untouched;
    ///  - it never constructs a candidate, profile or quest repository, never calls a
    ///    save API, never touches PlayerPrefs and never writes to persistent data, so a
    ///    take cannot store anything - the write owner is simply not instantiated;
    ///  - it never searches the scene for objects: the harness reference is explicit;
    ///  - it never changes a target, threshold, weight, feature, transform or model.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class DeveloperCommonSignalTwoTargetValidationHarness : MonoBehaviour
    {
        [Header("Explicit references - no scene search is performed")]
        [Tooltip("An existing PersonalLiveRecognitionHarness whose Step 9C bridge is bound. "
                 + "Leave empty to instantiate the approved technical prefab at runtime.")]
        [SerializeField] PersonalLiveRecognitionHarness harness;

        [Tooltip("Resources path of the approved candidate technical prefab, used only "
                 + "when no harness reference is assigned.")]
        [SerializeField] string technicalHarnessResource =
            "Diagnostics/PersonalMultiPrototypeLiveRecognitionTechnical";

        [SerializeField] Rect panelRect = new Rect(660f, 24f, 520f, 440f);

        [Header("Step 10P v1.0f2 common-2D diagnostic trace - developer opt-in")]
        [Tooltip("Off by default. When off nothing is traced, allocated, logged or "
                 + "exported and the route behaves exactly as it did before v1.0f2. "
                 + "This component is the only activation surface: ordinary Practice and "
                 + "World construction cannot reach it.")]
        [SerializeField] bool enableCommon2DDiagnosticTrace = false;

        // The export root is NOT configurable. The developer route always writes to the
        // one fixed project-relative validation root the exporter pins, so no serialized
        // string can redirect a diagnostic write into Assets or a persistence root.

        GameObject runtimeHarnessInstance;
        CommonSignalTwoTargetValidationCoordinator coordinator;
        string status = "not initialized";
        string lastTraceMarker = "diagnostic trace disabled";
        Common2DArmResult lastArmResult = Common2DArmResult.RejectedNullOwner;

        /// <summary>The closed result of the last ownership acquisition attempt.</summary>
        public Common2DArmResult LastArmResult { get { return lastArmResult; } }

        /// <summary>The opt-in state. False unless a developer explicitly sets it.</summary>
        public bool Common2DDiagnosticTraceEnabled
        {
            get { return enableCommon2DDiagnosticTrace; }
        }

        public string LastTraceMarker { get { return lastTraceMarker; } }

        public CommonSignalTwoTargetValidationCoordinator Coordinator
        {
            get { return coordinator; }
        }
        public string Status { get { return status; } }

        /// <summary>
        /// The only bridge between the pure coordinator and the approved adapter. It
        /// forwards three calls and composes the diagnostic line; it holds no repository
        /// and decides nothing.
        /// </summary>
        sealed class PracticeAdapterPort : ICommonSignalValidationAdapterPort
        {
            readonly PersonalLivePracticeRecognitionAdapter adapter;

            public PracticeAdapterPort(PersonalLivePracticeRecognitionAdapter adapter)
            {
                this.adapter = adapter;
            }

            public bool SetExpectedWord(string target)
            {
                return adapter.SetExpectedWord(target);
            }

            public bool BeginCapture()
            {
                return adapter.BeginCapture();
            }

            /// <summary>
            /// Step 10P v1.0d1: forwards straight to the adapter operation that already
            /// exists. Nothing is reimplemented and no new state is introduced here.
            /// </summary>
            public bool ResetCapture()
            {
                return adapter.ResetCapture();
            }

            public ValidationAttemptInfo FinalizeCapture()
            {
                PracticeRecognitionAttempt attempt = adapter.FinalizeCapture();
                if (attempt == null) return null;
                return new ValidationAttemptInfo(
                    attempt.CaptureSessionId,
                    attempt.CaptureState.ToString(),
                    CommonSignalAttemptDiagnostics.Build(attempt));
            }
        }

        /// <summary>
        /// Disabling the component cancels the arm it holds and drops any uncommitted
        /// trace, so ordinary Practice can never continue on this component's opt-in.
        /// Re-activation requires Initialize with the explicit opt-in again.
        /// </summary>
        void OnDisable()
        {
            if (Common2DDiagnosticTraceBuffer.ReleaseOwner(this))
                lastTraceMarker = "diagnostic trace released on disable";
        }

        void OnDestroy()
        {
            // Destroy cancels an uncommitted trace and removes the opt-in.
            Common2DDiagnosticTraceBuffer.ReleaseOwner(this);
            coordinator = null;
            if (runtimeHarnessInstance != null)
            {
                Destroy(runtimeHarnessInstance);
                runtimeHarnessInstance = null;
            }
        }

        /// <summary>
        /// Resolves the harness and builds the coordinator. Explicit: nothing happens
        /// until the operator asks for it.
        /// </summary>
        public bool Initialize()
        {
            if (coordinator != null) return true;
            try
            {
                if (harness == null)
                {
                    GameObject prefab = Resources.Load<GameObject>(technicalHarnessResource);
                    if (prefab == null)
                    {
                        status = "approved technical harness prefab not found";
                        return false;
                    }
                    runtimeHarnessInstance = Instantiate(prefab);
                    runtimeHarnessInstance.name = "DeveloperCommon2DValidationRoute";
                    harness = runtimeHarnessInstance.GetComponent<PersonalLiveRecognitionHarness>();
                }
                if (harness == null)
                {
                    status = "no PersonalLiveRecognitionHarness on the resolved object";
                    return false;
                }
                // The adapter loads a read-only runtime profile and the frozen common
                // classifier. It creates no repository of any kind.
                var adapter = new PersonalLivePracticeRecognitionAdapter(harness);
                coordinator = new CommonSignalTwoTargetValidationCoordinator(
                    new PracticeAdapterPort(adapter), line => Debug.Log(line));

                // The single activation point for the v1.0f2 diagnostic trace. When the
                // opt-in is off the buffer is left disarmed and no trace, allocation,
                // marker or export can occur anywhere in the route.
                lastArmResult = enableCommon2DDiagnosticTrace
                    ? Common2DDiagnosticTraceBuffer.TryArm(this)
                    : Common2DArmResult.RejectedNullOwner;
                if (!enableCommon2DDiagnosticTrace)
                    Common2DDiagnosticTraceBuffer.ReleaseOwner(this);
                lastTraceMarker = enableCommon2DDiagnosticTrace
                    ? "diagnostic trace arm: " + lastArmResult
                    : "diagnostic trace disabled";

                status = "ready: " + coordinator.CurrentTarget;
                return true;
            }
            catch (Exception exception)
            {
                coordinator = null;
                status = "initialization failed: " + exception.GetType().Name;
                return false;
            }
        }

        /// <summary>
        /// The reachable developer opt-in action. It works whether or not a coordinator
        /// already exists, so a route that was disabled and re-enabled can be re-armed
        /// only by this explicit action - never implicitly.
        /// </summary>
        public Common2DArmResult EnableCommon2DDiagnosticTrace()
        {
            enableCommon2DDiagnosticTrace = true;
            lastArmResult = Common2DDiagnosticTraceBuffer.TryArm(this);
            lastTraceMarker = "diagnostic trace arm: " + lastArmResult;
            return lastArmResult;
        }

        /// <summary>Turns the developer trace off and releases only this arm.</summary>
        public void DisableCommon2DDiagnosticTrace()
        {
            enableCommon2DDiagnosticTrace = false;
            Common2DDiagnosticTraceBuffer.ReleaseOwner(this);
            lastArmResult = Common2DArmResult.RejectedNullOwner;
            lastTraceMarker = "diagnostic trace disabled";
        }

        /// <summary>Begins the one capture allowed for the current target.</summary>
        public bool BeginCurrentTarget()
        {
            if (coordinator == null && !Initialize()) return false;
            string target = coordinator.CurrentTarget;
            bool began = coordinator.Begin();
            status = began
                ? "capturing " + target
                : (coordinator.State == TwoTargetValidationState.Stopped
                    ? "stopped: " + coordinator.StopReason
                    : "rejected: " + coordinator.LastRejection);
            return began;
        }

        /// <summary>
        /// Finalizes the current target once. The result is reported as-is; nothing is
        /// retried and no threshold, target or model is adjusted afterwards.
        /// </summary>
        public bool FinalizeCurrentTarget()
        {
            if (coordinator == null) return false;
            string target = coordinator.CurrentTarget;
            bool finalized = coordinator.Finalize();
            if (!finalized)
            {
                status = coordinator.State == TwoTargetValidationState.Stopped
                    ? "stopped: " + coordinator.StopReason
                    : "rejected: " + coordinator.LastRejection;
                return false;
            }
            status = coordinator.State == TwoTargetValidationState.Completed
                ? "both takes finalized; stop here and collect the evidence"
                : "finalized " + target + "; next target " + coordinator.CurrentTarget;

            // The coordinator has returned and the accepted terminal state is already
            // committed. Only now is the trace marked terminal-accepted and a disk export
            // requested, and only when the operator opted in. A failure here is reported
            // as a bounded typed marker and cannot change, retry or delay anything above.
            Common2DDiagnosticTraceBuffer.MarkTerminalAccepted(this);
            RequestDiagnosticExport();
            return true;
        }

        /// <summary>
        /// Asks the exporter for one payload for the trace just sealed. It never throws
        /// into the validation route: a failure becomes a typed marker and the already
        /// finalized take, its classification and its outcome are untouched.
        /// </summary>
        void RequestDiagnosticExport()
        {
            if (!enableCommon2DDiagnosticTrace) return;
            try
            {
                Common2DDiagnosticTrace trace = Common2DDiagnosticTraceBuffer.Active;
                if (trace == null || !trace.AttemptComplete || !trace.TerminalAccepted)
                {
                    // A capture whose collection already failed keeps its first closed
                    // reason; it is never replaced by a generic incomplete status.
                    Common2DTraceFailureReason reason =
                        Common2DDiagnosticTraceBuffer.CaptureFailed
                            ? Common2DDiagnosticTraceBuffer.FirstCollectionFailure
                            : (trace != null && !trace.TerminalAccepted
                                ? Common2DTraceFailureReason.TerminalNotAccepted
                                : Common2DTraceFailureReason.IncompletePayload);
                    lastTraceMarker = Common2DDiagnosticLogMarker.BuildFailure(
                        trace, Common2DTraceStatus.Failed, reason);
                }
                else
                {
                    Common2DDiagnosticExportResult export =
                        Common2DDiagnosticTraceExporter.ExportToDeveloperRoot(trace);
                    lastTraceMarker = export.Marker;
                }
                Debug.Log(lastTraceMarker);
                Common2DDiagnosticTraceBuffer.Clear();
            }
            catch (Exception)
            {
                lastTraceMarker = Common2DDiagnosticLogMarker.BuildFailure(
                    null, Common2DTraceStatus.Failed,
                    Common2DTraceFailureReason.WriteFailed);
            }
        }

        void OnGUI()
        {
            GUILayout.BeginArea(panelRect, GUI.skin.box);
            GUILayout.Label("[Step 10P v1.0c2] common-2D two-target validation (developer only)");
            GUILayout.Label("no repository, no session, no candidate/profile/quest write");
            GUILayout.Label("targets: GREETING then THANK_YOU, one take each, never LOVE");
            GUILayout.Label("status: " + status);

            if (coordinator == null)
            {
                if (GUILayout.Button("1. Initialize validation route")) Initialize();
                GUILayout.EndArea();
                return;
            }

            GUILayout.Label("state: " + coordinator.State +
                "   current target: " +
                (string.IsNullOrEmpty(coordinator.CurrentTarget)
                    ? "(complete)" : coordinator.CurrentTarget) +
                "   finalized: " + coordinator.FinalizedCount + "/2" +
                "   records: " + coordinator.RecordedLines);
            GUILayout.Label("adapter calls  set=" + coordinator.AdapterSetExpectedWordCalls +
                "  begin=" + coordinator.AdapterBeginCaptureCalls +
                "  finalize=" + coordinator.AdapterFinalizeCaptureCalls +
                "  reset=" + coordinator.AdapterResetCaptureCalls);
            if (!string.IsNullOrEmpty(coordinator.StopReason))
                GUILayout.Label("STOPPED (no retry, first failure preserved): " +
                                coordinator.StopReason);
            else if (!string.IsNullOrEmpty(coordinator.LastRejection))
                GUILayout.Label("rejected: " + coordinator.LastRejection);

            if (coordinator.CanBegin &&
                GUILayout.Button("2. Begin " + coordinator.CurrentTarget + " (one take)"))
                BeginCurrentTarget();
            if (coordinator.CanFinalize &&
                GUILayout.Button("3. Finalize " + coordinator.CurrentTarget))
                FinalizeCurrentTarget();

            if (coordinator.RecordedLines > 0)
            {
                GUILayout.Space(6f);
                GUILayout.Label("last record (also in Editor.log):");
                GUILayout.TextArea(coordinator.LastRecord, GUILayout.Height(90f));
            }
            GUILayout.EndArea();
        }
    }
}
#endif
