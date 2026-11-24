using System;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;
using WordsInSilence.Calibration;
using WordsInSilence.Camera;
using WordsInSilence.Core;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.PhysicalCompatibility;
using WordsInSilence.Settings;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    public enum CandidateCameraStartupState
    {
        Starting,
        Ready,
        Failed,
        Stopped,
    }

    public interface IPetRobotCandidateCameraReadiness
    {
        event Action Changed;
        CandidateCameraStartupState CameraState { get; }
        bool IsReady { get; }
        bool CameraStarted { get; }
        bool HasReceivedFrame { get; }
        bool ProviderObservationReady { get; }
        int CameraFrameCount { get; }
        int ProviderObservationCount { get; }
        string LastErrorCode { get; }
        string LastErrorMessage { get; }
    }

    /// <summary>
    /// Candidate-only camera lifecycle owner for the pet-robot vertical slice. It owns one
    /// approved UnityWebCamFrameSource, supplies its texture to the existing CameraFeed used
    /// by both Hand/Pose providers, and never delegates camera ownership to the legacy
    /// SignPracticeController or to PersonalLiveRecognitionHarness.
    /// </summary>
    public sealed class PetRobotCandidateCameraStartupRoute : MonoBehaviour,
        IPetRobotCandidateCameraReadiness,
        IPetRobotNeutralCameraContext
    {
        // frames awaited before opening a device, so Unity can finish the deferred
        // Object.Destroy of a previous scene's WebCamTexture
        const int DeferredDestroyFrames = 2;

        [SerializeField] RawImage cameraDisplay;
        [SerializeField] CaptureConfig captureConfig;
        [SerializeField] bool mirrorDisplay = true;

        ICameraFrameSource cameraSource;
        CancellationTokenSource cancellation;
        Step9CHandIdentityObservationBridge observationBridge;
        bool startAttempted;
        bool cleanupStarted;
        bool cameraStarted;
        bool hasReceivedFrame;

        // F54 passive telemetry; null unless WIS_LIVE_PIPELINE_TIMING=1. The camera
        // ownership code below is untouched: nothing here starts, stops or enumerates.
        LivePipelineTiming.Channel timing;
        bool timingInitialized;

        // F56: held only while THIS owner has a starting or playing camera
        bool backgroundLeaseHeld;
        // F58: same lifetime, same hooks; opt-in bounded frame pacing only
        bool framePacingLeaseHeld;
        bool continuityStartupLogged;
        int focusLinesLogged;
        const int FocusLineLimit = 40;
        bool providerObservationReady;
        int cameraFrameCount;
        bool firstFrameLogged;
        int providerObservationCount;
        string lastErrorCode = string.Empty;
        string lastErrorMessage = string.Empty;
        CandidateCameraStartupState state = CandidateCameraStartupState.Starting;

        public event Action Changed;

        public CandidateCameraStartupState CameraState => state;
        public bool IsReady => state == CandidateCameraStartupState.Ready;
        public bool CameraStarted => cameraStarted;
        public bool HasReceivedFrame => hasReceivedFrame;
        public bool ProviderObservationReady => providerObservationReady;
        public int CameraFrameCount => cameraFrameCount;
        public int ProviderObservationCount => providerObservationCount;
        public string LastErrorCode => lastErrorCode;
        public string LastErrorMessage => lastErrorMessage;
        public Texture PreviewTexture => cameraSource == null ? null : cameraSource.ActiveTexture;
        public bool ObservationSubscriptionActive => observationBridge != null;
        public int StartInvocationCount { get; private set; }
        public int StopInvocationCount { get; private set; }
        public int DisposeInvocationCount { get; private set; }

        public bool TrySnapshotCameraFingerprint(out CameraFingerprint fingerprint)
        {
            WebCamTexture texture = cameraSource == null ? null : cameraSource.ActiveTexture;
            if (!cameraStarted || texture == null || texture.width <= 0 || texture.height <= 0 ||
                string.IsNullOrWhiteSpace(cameraSource.ActiveDeviceName))
            {
                fingerprint = null;
                return false;
            }
            int rotation = ((texture.videoRotationAngle % 360) + 360) % 360;
            float fps = texture.requestedFPS > 0f
                ? texture.requestedFPS
                : (captureConfig == null ? 30f : captureConfig.requestedFps);
            fingerprint = new CameraFingerprint
            {
                DeviceName = cameraSource.ActiveDeviceName,
                Width = texture.width,
                Height = texture.height,
                Fps = fps,
                RotationDegrees = rotation,
                PreviewMirrored = mirrorDisplay,
            };
            return true;
        }

        /// <summary>
        /// Safe aggregate for the diagnostics panel and one local snapshot log. Two
        /// short lines: device state, then wiring. Only states, booleans, counts and
        /// texture dimensions - no device name, learner id, path, pixel or payload.
        /// </summary>
        /// <summary>One line for the FIRST owner frame after a start; never per frame.</summary>
        void LogFirstOwnerFrameOnce()
        {
            if (firstFrameLogged) return;
            firstFrameLogged = true;
            Debug.Log("[M3Camera] first-owner-frame " + BuildSafeStatusLine());
        }

        public string BuildSafeStatusLine()
        {
            WebCamTexture texture = cameraSource == null ? null : cameraSource.ActiveTexture;
            bool hasTexture = texture != null;
            string device = "camera state=" + state +
                " started=" + cameraStarted +
                " texture=" + hasTexture +
                (hasTexture ? " size=" + texture.width + "x" + texture.height +
                    " playing=" + texture.isPlaying : " size=- playing=-") +
                " ownerFrames=" + cameraFrameCount +
                " firstFrame=" + hasReceivedFrame;
            string wiring = "wiring providerObs=" + providerObservationCount +
                " obsReady=" + providerObservationReady +
                " bridgeBound=" + (observationBridge != null) +
                " displayBound=" + (cameraDisplay != null) +
                " displayTexture=" + (cameraDisplay != null && cameraDisplay.texture != null) +
                " errorCode=" + (string.IsNullOrEmpty(lastErrorCode) ? "-" : lastErrorCode) +
                " start/stop/dispose=" + StartInvocationCount + "/" + StopInvocationCount +
                "/" + DisposeInvocationCount;
            return device + "\n" + wiring;
        }

        void Awake()
        {
            if (cameraSource == null)
                cameraSource = new UnityWebCamFrameSource();
        }

        async void Start()
        {
            await StartCameraOnceAsync();
        }

        void Update()
        {
            EnsureTiming();
            WebCamTexture texture = cameraSource == null ? null : cameraSource.ActiveTexture;
            bool updatedThisFrame = cameraStarted && texture != null && texture.didUpdateThisFrame;
            if (timing != null)
            {
                timing.MarkEntry(Time.frameCount);
                timing.Increment(updatedThisFrame
                    ? TimingCounter.FrameUpdated : TimingCounter.FrameSkipped);
                // the already-owned device object only; no device enumeration is performed
                if (texture != null && texture.isPlaying)
                    timing.Increment(TimingCounter.CameraPlaying);
                timing.Flush();
            }
            if (!updatedThisFrame) return;
            cameraFrameCount++;
            hasReceivedFrame = true;
            LogFirstOwnerFrameOnce();
            EvaluateReadiness();
        }

        void EnsureTiming()
        {
            if (timingInitialized) return;
            timing = LivePipelineTiming.CreateOrNull("camera_owner");
            timingInitialized = true;
        }

        // F56 focus correlation. These callbacks ONLY log: they never start, stop or reopen a
        // camera, and they carry no name, path, id or landmark.
        void OnApplicationFocus(bool focused) { LogAppTransition("focus", focused); }

        void OnApplicationPause(bool paused) { LogAppTransition("pause", paused); }

        void LogAppTransition(string kind, bool flag)
        {
            EnsureTiming();
            if (timing == null || focusLinesLogged >= FocusLineLimit) return;
            focusLinesLogged++;
            WebCamTexture texture = cameraSource == null ? null : cameraSource.ActiveTexture;
            Debug.Log(LivePipelineTiming.LogPrefix + " camera_owner " + kind + "=" + flag
                + " utc=" + System.DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ",
                    System.Globalization.CultureInfo.InvariantCulture)
                + " state=" + state + " started=" + cameraStarted
                + " playing=" + (texture != null && texture.isPlaying)
                + " ready=" + (state == CandidateCameraStartupState.Ready)
                + " ownerFrames=" + cameraFrameCount
                + " runInBackground=" + Application.runInBackground
                + " leaseHeld=" + backgroundLeaseHeld);
        }

        void OnDisable()
        {
            StopAndDisposeOnce();
        }

        void OnDestroy()
        {
            StopAndDisposeOnce();
        }

        public void BindObservationBridge(Step9CHandIdentityObservationBridge bridge)
        {
            if (ReferenceEquals(observationBridge, bridge))
            {
                EvaluateReadiness();
                return;
            }

            UnbindObservationBridge();
            observationBridge = bridge;
            if (observationBridge != null)
            {
                observationBridge.ObservationArrived += OnObservationArrived;
                providerObservationCount = observationBridge.ObservationCount;
                providerObservationReady = observationBridge.IsBound && observationBridge.HasReceivedObservation;
            }
            EvaluateReadiness();
        }

        public void UnbindObservationBridge()
        {
            if (observationBridge != null)
                observationBridge.ObservationArrived -= OnObservationArrived;
            observationBridge = null;
            providerObservationReady = false;
        }

        /// <summary>
        /// Releases every active camera owner before a scene transition, while the old
        /// scene is still alive. It reuses the existing exact-once cleanup, so calling
        /// it twice or after teardown is harmless. Returns how many owners it released.
        /// </summary>
        public static int ReleaseAllForSceneTransition()
        {
            PetRobotCandidateCameraStartupRoute[] routes = FindObjectsByType<
                PetRobotCandidateCameraStartupRoute>(FindObjectsInactive.Include,
                FindObjectsSortMode.None);
            int released = 0;
            for (int index = 0; index < routes.Length; index++)
            {
                if (routes[index] == null || routes[index].cleanupStarted) continue;
                routes[index].StopAndDisposeOnce();
                released++;
            }
            if (released > 0)
                Debug.Log("[M3Camera] pre-transition release owners=" + released);
            return released;
        }

        async Task StartCameraOnceAsync()
        {
            if (startAttempted || cleanupStarted) return;
            startAttempted = true;
            // taken once per owner, after the cleanup guard, and released in SetFailure or
            // StopAndDisposeOnce; never per frame and never on a camera that is not ours
            bool backgroundBefore = Application.runInBackground;
            int frameRateBefore = CameraFramePacingLease.CurrentValue;
            CameraBackgroundLease.Acquire(ref backgroundLeaseHeld);
            CameraFramePacingLease.Acquire(ref framePacingLeaseHeld);
            if (!continuityStartupLogged)
            {
                continuityStartupLogged = true;
                Debug.Log("[M3Camera] continuity enabled=" + CameraBackgroundLease.Enabled
                    + " leaseHeld=" + backgroundLeaseHeld
                    + " runInBackground_before=" + backgroundBefore
                    + " runInBackground_now=" + Application.runInBackground);
                // F58 verification line, once per owner and never per frame. It reports only
                // settings values; vSyncCount is read, never written.
                Debug.Log("[M3Camera] framePacing enabled=" + CameraFramePacingLease.Enabled
                    + " leaseHeld=" + framePacingLeaseHeld
                    + " targetFrameRate_original=" + frameRateBefore
                    + " targetFrameRate_now=" + CameraFramePacingLease.CurrentValue
                    + " vSyncCount=" + CameraFramePacingLease.CurrentVSyncCount
                    + " (A/B candidate only; no fix is claimed)");
            }
            StartInvocationCount++;
            state = CandidateCameraStartupState.Starting;
            Changed?.Invoke();

            ICameraFrameSource source = cameraSource;
            if (source == null)
            {
                SetFailure("CAMERA_SOURCE_MISSING", "The owned camera source is unavailable.");
                return;
            }

            CancellationTokenSource startupCancellation = new CancellationTokenSource();
            CancellationToken startupToken = startupCancellation.Token;
            cancellation = startupCancellation;
            CameraStartResult result;
            try
            {
                // UnityEngine.Object.Destroy of a previous scene's WebCamTexture is
                // deferred to end of frame. Wait exactly DeferredDestroyFrames frames so
                // that destruction completes before this device open; this is a bounded
                // boundary, not a longer timeout and not a retry.
                for (int frame = 0; frame < DeferredDestroyFrames; frame++)
                {
                    if (cleanupStarted || startupToken.IsCancellationRequested) return;
                    await Awaitable.NextFrameAsync(startupToken);
                }
                result = await source.StartAsync(BuildRequest(), startupToken);
            }
            catch (Exception exception)
            {
                if (cleanupStarted || startupToken.IsCancellationRequested ||
                    !ReferenceEquals(cameraSource, source)) return;
                SetFailure("CAMERA_UNKNOWN_ERROR", exception.GetType().Name + ": " + exception.Message);
                return;
            }

            if (cleanupStarted || startupToken.IsCancellationRequested ||
                !ReferenceEquals(cameraSource, source)) return;
            if (!result.IsSuccess)
            {
                SetFailure(result.ErrorCode, result.ErrorDetail);
                return;
            }

            cameraStarted = true;
            if (cameraDisplay != null)
            {
                cameraDisplay.texture = source.ActiveTexture;
                ApplyDisplayTransform();
            }
            EvaluateReadiness();
            // one post-start snapshot to the private local log; no per-frame logging
            Debug.Log("[M3Camera] post-start " + BuildSafeStatusLine());
            firstFrameLogged = false;
        }

        void StopAndDisposeOnce()
        {
            if (cleanupStarted) return;
            cleanupStarted = true;
            string stopError = null;
            string disposeError = null;
            Debug.Log("[M3Camera] stop-dispose begin state=" + state +
                " started=" + cameraStarted + " ownerFrames=" + cameraFrameCount +
                " firstFrame=" + hasReceivedFrame +
                " start/stop/dispose=" + StartInvocationCount + "/" + StopInvocationCount +
                "/" + DisposeInvocationCount);
            cancellation?.Cancel();
            // released exactly once per owner; a departing owner cannot clear a lease that a
            // newly started owner still holds
            CameraBackgroundLease.Release(ref backgroundLeaseHeld);
            CameraFramePacingLease.Release(ref framePacingLeaseHeld);
            UnbindObservationBridge();

            ICameraFrameSource source = cameraSource;
            cameraSource = null;
            if (source != null)
            {
                StopInvocationCount++;
                try
                {
                    // Unity does not wait for a fire-and-forget async cleanup during
                    // PlayMode or Scene teardown. Stop the native webcam synchronously
                    // on Unity's main thread so the device is released immediately.
                    WebCamTexture texture = source.ActiveTexture;
                    if (texture != null && texture.isPlaying)
                        texture.Stop();
                }
                catch (Exception exception)
                {
                    // Cleanup remains fail-safe; the owned source is still disposed below.
                    // The type is retained so the end line cannot claim a clean stop.
                    stopError = exception.GetType().Name;
                }
                if (source is IDisposable disposable)
                {
                    DisposeInvocationCount++;
                    try
                    {
                        disposable.Dispose();
                    }
                    catch (Exception exception)
                    {
                        // PlayMode teardown must continue even if the source reports
                        // an error after the native WebCamTexture has already stopped.
                        // The type is retained so the end line stays truthful.
                        disposeError = exception.GetType().Name;
                    }
                }
            }

            cancellation?.Dispose();
            cancellation = null;
            if (cameraDisplay != null) cameraDisplay.texture = null;
            cameraStarted = false;
            if (state != CandidateCameraStartupState.Failed)
                state = CandidateCameraStartupState.Stopped;
            Debug.Log("[M3Camera] stop-dispose end state=" + state +
                " stopAttempted=" + (StopInvocationCount > 0) +
                " stopError=" + (stopError ?? "-") +
                " disposeAttempted=" + (DisposeInvocationCount > 0) +
                " disposeError=" + (disposeError ?? "-") +
                " cleanupClean=" + (stopError == null && disposeError == null) +
                " ownerFrames=" + cameraFrameCount +
                " displayTexture=" + (cameraDisplay != null && cameraDisplay.texture != null) +
                " start/stop/dispose=" + StartInvocationCount + "/" + StopInvocationCount +
                "/" + DisposeInvocationCount);
            Changed?.Invoke();
        }

        void OnObservationArrived(RawMediaPipeObservation observation)
        {
            providerObservationCount = observationBridge == null
                ? providerObservationCount + 1
                : observationBridge.ObservationCount;
            providerObservationReady = observationBridge == null || observationBridge.IsBound;
            EvaluateReadiness();
        }

        void EvaluateReadiness()
        {
            if (state == CandidateCameraStartupState.Failed || cleanupStarted) return;
            bool ready = cameraStarted && hasReceivedFrame && providerObservationReady;
            CandidateCameraStartupState next = ready
                ? CandidateCameraStartupState.Ready
                : CandidateCameraStartupState.Starting;
            if (state != next)
            {
                state = next;
                Changed?.Invoke();
            }
        }

        void SetFailure(string code, string message)
        {
            lastErrorCode = string.IsNullOrWhiteSpace(code) ? "CAMERA_UNKNOWN_ERROR" : code;
            lastErrorMessage = message ?? string.Empty;
            // a failed or cancelled start must not leak the lease
            CameraBackgroundLease.Release(ref backgroundLeaseHeld);
            CameraFramePacingLease.Release(ref framePacingLeaseHeld);
            cameraStarted = false;
            state = CandidateCameraStartupState.Failed;
            Changed?.Invoke();
        }

        CameraStartRequest BuildRequest()
        {
            return new CameraStartRequest
            {
                DeviceName = null,
                RequestedWidth = captureConfig == null ? 640 : captureConfig.requestedWidth,
                RequestedHeight = captureConfig == null ? 480 : captureConfig.requestedHeight,
                RequestedFps = captureConfig == null ? 30 : captureConfig.requestedFps,
                MirrorDisplay = mirrorDisplay,
            };
        }

        void ApplyDisplayTransform()
        {
            WebCamTexture texture = cameraSource.ActiveTexture;
            bool vertical = texture != null && texture.videoVerticallyMirrored;
            cameraDisplay.uvRect = new Rect(
                mirrorDisplay ? 1f : 0f,
                vertical ? 1f : 0f,
                mirrorDisplay ? -1f : 1f,
                vertical ? -1f : 1f);
            if (texture != null)
                cameraDisplay.rectTransform.localRotation = Quaternion.Euler(0f, 0f, -texture.videoRotationAngle);
        }
    }
}
