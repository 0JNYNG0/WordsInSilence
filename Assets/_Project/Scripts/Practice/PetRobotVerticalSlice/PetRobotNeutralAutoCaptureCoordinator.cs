using System;
using WordsInSilence.Calibration;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.PhysicalCompatibility;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    public interface IPetRobotNeutralCameraContext
    {
        bool IsReady { get; }
        bool TrySnapshotCameraFingerprint(out CameraFingerprint fingerprint);
    }

    /// <summary>
    /// Orchestrates only calibration/segmentation and the current session's narrow auto seams.
    /// Recognition, scoring, candidate policy, story progression, and webcam ownership remain external.
    /// </summary>
    public sealed class PetRobotNeutralAutoCaptureCoordinator : IDisposable
    {
        readonly PetRobotSignPracticeSession session;
        readonly PetRobotNeutralCalibrationProfileService profileService;
        readonly RawMediaPipeNeutralAutoSegmenter segmenter;
        readonly RawMediaPipeObservationRingBuffer ring;
        readonly IPersonalLiveObservationSource observationSource;
        readonly IPetRobotNeutralCameraContext camera;
        readonly IDisposable ownedObservationSource;
        readonly bool deferredObservationBindingAllowed;

        bool profileResolvedForCamera;
        bool allowProfileUpdate;
        bool attemptPendingForReadiness;
        bool autoCaptureStarted;
        bool autoFinalizeIssued;
        int captureAmbiguityCount;
        bool disposed;

        public event Action Changed;
        public RawMediaPipeNeutralAutoSegmenterState State => segmenter.State;
        public float CalibrationProgress => segmenter.CalibrationProgress;
        public float NeutralHoldProgress => segmenter.NeutralHoldProgress;
        public float EndHoldProgress => segmenter.EndHoldProgress;
        public bool IsProfileLoaded => segmenter.IsProfileLoaded;
        public bool IsAutoAttemptArmed => segmenter.IsAutoAttemptArmed;
        public string LastFailure => segmenter.LastFailure;
        public NeutralObservationReason LastNeutralReason => segmenter.LastNeutralReason;
        public string AttemptId => segmenter.AttemptId;
        public int PreRollObservationCount => ring.Count;
        public PetRobotNeutralAutoSegmentationSettings Settings => segmenter.Settings;

        public PetRobotNeutralAutoCaptureCoordinator(
            PetRobotSignPracticeSession session,
            PetRobotNeutralCalibrationProfileService profileService,
            RawMediaPipeNeutralAutoSegmenter segmenter,
            RawMediaPipeObservationRingBuffer ring,
            IPersonalLiveObservationSource observationSource,
            IPetRobotNeutralCameraContext camera,
            bool allowDeferredObservationBinding = false)
        {
            this.session = session ?? throw new ArgumentNullException(nameof(session));
            this.profileService = profileService ?? throw new ArgumentNullException(nameof(profileService));
            this.segmenter = segmenter ?? throw new ArgumentNullException(nameof(segmenter));
            this.ring = ring ?? throw new ArgumentNullException(nameof(ring));
            this.observationSource = observationSource ?? throw new ArgumentNullException(nameof(observationSource));
            this.camera = camera ?? throw new ArgumentNullException(nameof(camera));
            deferredObservationBindingAllowed = allowDeferredObservationBinding;
            if (!observationSource.IsBound && !deferredObservationBindingAllowed)
                throw new InvalidOperationException("current Step 9C observation source must be bound");
            ownedObservationSource = observationSource as IDisposable;
            observationSource.ObservationArrived += OnObservation;
            observationSource.SourceFailed += OnSourceFailed;
            segmenter.Changed += OnSegmenterChanged;
        }

        public bool RequestAttempt()
        {
            ThrowIfDisposed();
            if (session.State != PetRobotPracticeState.Ready) return false;

            if (!camera.IsReady || !observationSource.IsBound)
            {
                // unchanged: a request made before the camera or the binding is ready
                // discards the stale pre-roll, then waits or refuses as before
                ring.Clear();
                autoCaptureStarted = false;
                autoFinalizeIssued = false;
                captureAmbiguityCount = 0;
                if (!deferredObservationBindingAllowed)
                {
                    attemptPendingForReadiness = false;
                    return false;
                }
                attemptPendingForReadiness = true;
                Changed?.Invoke();
                return true;
            }

            // Nothing is discarded here any more. The ring and the per-attempt flags are
            // reset inside StartRequestedAttempt, AFTER the one preparation call has been
            // accepted, so an attempt the session refuses immediately leaves the pre-roll
            // ring and the segmenter exactly as they were.
            attemptPendingForReadiness = false;
            return StartRequestedAttempt();
        }

        public void Cancel()
        {
            ThrowIfDisposed();
            if (autoCaptureStarted || session.State == PetRobotPracticeState.Capturing)
            {
                segmenter.Fail("auto capture cannot be cancelled through a segmentation-only shortcut");
                return;
            }
            ring.Clear();
            attemptPendingForReadiness = false;
            captureAmbiguityCount = 0;
            segmenter.Cancel();
        }

        public bool RequestRecalibration()
        {
            ThrowIfDisposed();
            CameraFingerprint fingerprint;
            if (!camera.IsReady || !camera.TrySnapshotCameraFingerprint(out fingerprint)) return false;
            ring.Clear();
            attemptPendingForReadiness = false;
            profileResolvedForCamera = true;
            allowProfileUpdate = true;
            autoCaptureStarted = false;
            autoFinalizeIssued = false;
            captureAmbiguityCount = 0;
            segmenter.RequestRecalibration(fingerprint);
            return true;
        }

        public void Dispose()
        {
            if (disposed) return;
            observationSource.ObservationArrived -= OnObservation;
            observationSource.SourceFailed -= OnSourceFailed;
            segmenter.Changed -= OnSegmenterChanged;
            ownedObservationSource?.Dispose();
            ring.Clear();
            attemptPendingForReadiness = false;
            disposed = true;
        }

        void OnObservation(RawMediaPipeObservation observation)
        {
            if (disposed || observation == null) return;
            CameraFingerprint fingerprint;
            if (!camera.IsReady || !camera.TrySnapshotCameraFingerprint(out fingerprint))
            {
                segmenter.Process(observation, false);
                return;
            }

            if (!profileResolvedForCamera)
            {
                NeutralCalibrationProfileLoadResult loaded = profileService.Load(fingerprint);
                profileResolvedForCamera = true;
                allowProfileUpdate = loaded.Status != NeutralCalibrationProfileLoadStatus.Missing;
                if (loaded.Status == NeutralCalibrationProfileLoadStatus.Valid)
                    segmenter.UseSavedProfile(loaded.Profile, fingerprint);
                else
                    segmenter.RequireCalibration(fingerprint, loaded.Message);
            }

            if (attemptPendingForReadiness && !StartRequestedAttempt())
            {
                attemptPendingForReadiness = false;
                segmenter.Fail("current PetRobot session rejected the deferred auto attempt start");
                return;
            }

            if (segmenter.State == RawMediaPipeNeutralAutoSegmenterState.WaitingForNeutral ||
                segmenter.State == RawMediaPipeNeutralAutoSegmenterState.Armed ||
                segmenter.State == RawMediaPipeNeutralAutoSegmenterState.Capturing ||
                segmenter.State == RawMediaPipeNeutralAutoSegmenterState.EndCandidate)
            {
                try
                {
                    ring.Add(observation);
                }
                catch (Exception exception)
                {
                    segmenter.Fail("raw pre-roll ring rejected an observation: " + exception.Message);
                    return;
                }
            }

            RawMediaPipeNeutralAutoSegmenterDecision decision;
            try
            {
                decision = segmenter.Process(observation, true);
            }
            catch (Exception exception)
            {
                segmenter.Fail("neutral auto segmentation rejected an observation: " + exception.Message);
                return;
            }

            if (decision.CalibrationProfile != null)
            {
                NeutralCalibrationProfileSaveResult saved =
                    profileService.Save(decision.CalibrationProfile, allowProfileUpdate);
                if (!saved.IsSaved)
                {
                    segmenter.Fail("neutral profile transaction failed: " + saved.Message);
                    return;
                }
                allowProfileUpdate = false;
                segmenter.ConfirmCalibrationSaved(decision.CalibrationProfile);
                ring.Clear();
            }

            if (decision.BeginCaptureRequested)
            {
                if (autoCaptureStarted || ring.Count == 0 ||
                    !session.StartAutoExpression(ring.Snapshot()))
                {
                    segmenter.Fail("current PetRobot session rejected the auto capture start");
                    return;
                }
                autoCaptureStarted = true;
                autoFinalizeIssued = false;
                captureAmbiguityCount = 0;
            }

            if (decision.FinalizeCaptureRequested)
            {
                if (!autoCaptureStarted || autoFinalizeIssued)
                {
                    segmenter.Fail("auto finalize was requested without one active auto capture");
                    return;
                }
                autoFinalizeIssued = true;
                if (!session.FinishAutoExpression(decision.TrailingTrimSeconds))
                    segmenter.Fail("current PetRobot session rejected the auto finalize request");
            }
        }

        void OnSourceFailed(Exception exception)
        {
            if (disposed) return;
            RawMediaPipeNeutralAutoSegmenterState state = segmenter.State;
            if (state == RawMediaPipeNeutralAutoSegmenterState.Inactive ||
                state == RawMediaPipeNeutralAutoSegmenterState.Evaluating ||
                state == RawMediaPipeNeutralAutoSegmenterState.Failed)
                return;

            if (exception is Step9CAmbiguousHandIdentityException)
            {
                if (state == RawMediaPipeNeutralAutoSegmenterState.NeedsCalibration ||
                    state == RawMediaPipeNeutralAutoSegmenterState.CalibratingNeutral ||
                    state == RawMediaPipeNeutralAutoSegmenterState.WaitingForNeutral ||
                    state == RawMediaPipeNeutralAutoSegmenterState.Armed)
                    return;

                if (state == RawMediaPipeNeutralAutoSegmenterState.Capturing ||
                    state == RawMediaPipeNeutralAutoSegmenterState.EndCandidate)
                {
                    captureAmbiguityCount++;
                    if (captureAmbiguityCount == 1) return;
                }
            }

            string failureMessage = "current Step 9C observation source failed: " +
                (exception == null ? "unknown source failure" : exception.Message);
            segmenter.Fail(failureMessage);
            if (autoCaptureStarted && !autoFinalizeIssued &&
                session.State == PetRobotPracticeState.Capturing)
            {
                autoFinalizeIssued = true;
                if (session.FinishAutoTechnicalFailure(exception))
                    autoCaptureStarted = false;
            }
        }

        void OnSegmenterChanged() => Changed?.Invoke();

        bool StartRequestedAttempt()
        {
            if (session.State != PetRobotPracticeState.Ready ||
                !camera.IsReady || !observationSource.IsBound)
                return false;
            if (!session.PrepareAutoAttempt()) return false;

            // the attempt is really starting only now; the preparation above still runs
            // exactly once per request, for the immediate and the deferred entry alike
            ring.Clear();
            autoCaptureStarted = false;
            autoFinalizeIssued = false;
            captureAmbiguityCount = 0;
            attemptPendingForReadiness = false;
            return segmenter.RequestAttempt();
        }

        void ThrowIfDisposed()
        {
            if (disposed) throw new ObjectDisposedException(nameof(PetRobotNeutralAutoCaptureCoordinator));
        }
    }
}
