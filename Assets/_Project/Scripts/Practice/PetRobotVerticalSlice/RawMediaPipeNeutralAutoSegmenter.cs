using System;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;
using WordsInSilence.Calibration;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    public sealed class PetRobotNeutralAutoSegmentationSettings
    {
        public double NeutralCalibrationHoldSeconds { get; }
        public double ReadyNeutralHoldSeconds { get; }
        public double OnsetDisplacementThreshold { get; }
        public int OnsetConsecutiveFrames { get; }
        public double PreRollSeconds { get; }
        public double CaptureGraceSeconds { get; }
        public double ArmedTimeoutSeconds { get; }
        public double MaxCaptureSeconds { get; }
        public double EndNeutralHoldSeconds { get; }
        public double PostCaptureMarginSeconds { get; }
        public float NeutralArmsDownMaxDegrees { get; }
        public double AutoTrailingTrimSeconds => EndNeutralHoldSeconds - PostCaptureMarginSeconds;

        public static PetRobotNeutralAutoSegmentationSettings Default =>
            new PetRobotNeutralAutoSegmentationSettings(
                2.5d, 1.5d, 0.03d, 3, 0.35d, 0.5d, 6d, 10d, 1d, 0.25d, 45f);

        public PetRobotNeutralAutoSegmentationSettings(
            double neutralCalibrationHoldSeconds,
            double readyNeutralHoldSeconds,
            double onsetDisplacementThreshold,
            int onsetConsecutiveFrames,
            double preRollSeconds,
            double captureGraceSeconds,
            double armedTimeoutSeconds,
            double maxCaptureSeconds,
            double endNeutralHoldSeconds,
            double postCaptureMarginSeconds,
            float neutralArmsDownMaxDegrees)
        {
            RequirePositive(neutralCalibrationHoldSeconds, nameof(neutralCalibrationHoldSeconds));
            RequirePositive(readyNeutralHoldSeconds, nameof(readyNeutralHoldSeconds));
            RequirePositive(onsetDisplacementThreshold, nameof(onsetDisplacementThreshold));
            if (onsetConsecutiveFrames <= 0) throw new ArgumentOutOfRangeException(nameof(onsetConsecutiveFrames));
            RequirePositive(preRollSeconds, nameof(preRollSeconds));
            RequirePositive(captureGraceSeconds, nameof(captureGraceSeconds));
            RequirePositive(armedTimeoutSeconds, nameof(armedTimeoutSeconds));
            RequirePositive(maxCaptureSeconds, nameof(maxCaptureSeconds));
            RequirePositive(endNeutralHoldSeconds, nameof(endNeutralHoldSeconds));
            if (!IsFinite(postCaptureMarginSeconds) || postCaptureMarginSeconds < 0d ||
                postCaptureMarginSeconds >= endNeutralHoldSeconds)
                throw new ArgumentOutOfRangeException(nameof(postCaptureMarginSeconds));
            if (float.IsNaN(neutralArmsDownMaxDegrees) || float.IsInfinity(neutralArmsDownMaxDegrees) ||
                neutralArmsDownMaxDegrees <= 0f || neutralArmsDownMaxDegrees > 90f)
                throw new ArgumentOutOfRangeException(nameof(neutralArmsDownMaxDegrees));

            NeutralCalibrationHoldSeconds = neutralCalibrationHoldSeconds;
            ReadyNeutralHoldSeconds = readyNeutralHoldSeconds;
            OnsetDisplacementThreshold = onsetDisplacementThreshold;
            OnsetConsecutiveFrames = onsetConsecutiveFrames;
            PreRollSeconds = preRollSeconds;
            CaptureGraceSeconds = captureGraceSeconds;
            ArmedTimeoutSeconds = armedTimeoutSeconds;
            MaxCaptureSeconds = maxCaptureSeconds;
            EndNeutralHoldSeconds = endNeutralHoldSeconds;
            PostCaptureMarginSeconds = postCaptureMarginSeconds;
            NeutralArmsDownMaxDegrees = neutralArmsDownMaxDegrees;
        }

        static void RequirePositive(double value, string name)
        {
            if (!IsFinite(value) || value <= 0d) throw new ArgumentOutOfRangeException(name);
        }

        static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
    }

    public enum RawMediaPipeNeutralAutoSegmenterState
    {
        Inactive,
        NeedsCalibration,
        CalibratingNeutral,
        WaitingForNeutral,
        Armed,
        Capturing,
        EndCandidate,
        Evaluating,
        Failed,
    }

    public enum NeutralObservationReason
    {
        Neutral,
        NoPose,
        NonFiniteGeometry,
        ArmsNotDown,
        HandVisible,
        WristNotBottomCompatible,
        AmbiguousWristLoss,
        WristBaselineMismatch,
        BodyScaleMismatch,
        CameraNotReady,
    }

    /// <summary>
    /// F32. How ONE capture actually ended. It is diagnostic only: it changes no threshold,
    /// no gate and no recognition result, and a timeout is never reported as a stable end.
    /// </summary>
    public enum NeutralEndpointResolution
    {
        /// <summary>No endpoint has been resolved for the current attempt.</summary>
        None,
        /// <summary>One second of CONTINUOUSLY OBSERVED still neutral was reached.</summary>
        StableNeutralHold,
        /// <summary>The 10 s capture bound was reached. This is NOT a stable end.</summary>
        MaxCaptureTimeout,
    }

    /// <summary>F32. Why the END hold was last discarded and restarted.</summary>
    public enum NeutralEndpointHoldReset
    {
        None,
        /// <summary>The camera was not ready, so nothing was observed to hold.</summary>
        CameraNotReady,
        /// <summary>More than the endpoint gap bound passed without an observation.</summary>
        ObservationGap,
        /// <summary>A non-finite or non-increasing observation timestamp.</summary>
        NonIncreasingTimestamp,
        /// <summary>The observation was no longer the saved neutral.</summary>
        NeutralLost,
        /// <summary>
        /// Unused since F33. Kept only so historical diagnostic logs keep their enum
        /// identity; the endpoint no longer vetoes a held neutral on wrist motion.
        /// </summary>
        WristMotion,
    }

    public sealed class RawMediaPipeNeutralAutoSegmenterDecision
    {
        public UserCalibrationProfile CalibrationProfile { get; }
        public bool BeginCaptureRequested { get; }
        public bool FinalizeCaptureRequested { get; }
        public bool MaxCaptureReached { get; }
        public bool ArmedTimeoutReached { get; }
        public double TrailingTrimSeconds { get; }

        internal RawMediaPipeNeutralAutoSegmenterDecision(
            UserCalibrationProfile calibrationProfile = null,
            bool beginCaptureRequested = false,
            bool finalizeCaptureRequested = false,
            bool maxCaptureReached = false,
            bool armedTimeoutReached = false,
            double trailingTrimSeconds = 0d)
        {
            CalibrationProfile = calibrationProfile;
            BeginCaptureRequested = beginCaptureRequested;
            FinalizeCaptureRequested = finalizeCaptureRequested;
            MaxCaptureReached = maxCaptureReached;
            ArmedTimeoutReached = armedTimeoutReached;
            TrailingTrimSeconds = trailingTrimSeconds;
        }

        public static RawMediaPipeNeutralAutoSegmenterDecision None { get; } =
            new RawMediaPipeNeutralAutoSegmenterDecision();
    }

    /// <summary>Pure current-observation neutral/readiness/onset/end state machine.</summary>
    public sealed class RawMediaPipeNeutralAutoSegmenter
    {
        const int LeftShoulder = 11;
        const int RightShoulder = 12;
        const int LeftElbow = 13;
        const int RightElbow = 14;
        const int LeftWrist = 15;
        const int RightWrist = 16;
        const double ActivationEvidenceHoldSeconds = 0.20d;
        /// <summary>
        /// F32. The largest gap between two observations that may still be treated as ONE
        /// continuously observed stretch while an END candidate is held. At the ~23 Hz the
        /// production camera actually delivers, consecutive frames are about 0.043 s apart,
        /// so 0.20 s still tolerates roughly four consecutive dropped frames while a whole
        /// unobserved second can never be counted as neutral evidence. It is a continuity
        /// bound only: it does not shorten or lengthen the 1 s end hold, does not change the
        /// 10 s capture bound and does not touch onset or calibration.
        /// </summary>
        public const double EndpointMaxObservationGapSeconds = 0.20d;

        readonly string learnerProfileId;
        readonly PetRobotNeutralAutoSegmentationSettings settings;
        readonly List<NeutralSample> calibrationSamples = new List<NeutralSample>();
        readonly List<float> calibrationMotionSamples = new List<float>();

        UserCalibrationProfile profile;
        CameraFingerprint cameraFingerprint;
        bool attemptRequested;
        bool calibrationRequested;
        double calibrationStart = double.NaN;
        double neutralHoldStart = double.NaN;
        double armedStart = double.NaN;
        double captureStart = double.NaN;
        double endCandidateStart = double.NaN;
        /// <summary>F32. The last observation actually counted into the END hold.</summary>
        double lastEndObservationTimestamp = double.NaN;
        double activationEvidenceStart = double.NaN;
        double lastLeftBottomTimestamp = double.NaN;
        double lastRightBottomTimestamp = double.NaN;
        WristPair previousCalibrationWrists;

        public event Action Changed;
        public RawMediaPipeNeutralAutoSegmenterState State { get; private set; } =
            RawMediaPipeNeutralAutoSegmenterState.Inactive;
        public PetRobotNeutralAutoSegmentationSettings Settings => settings;
        public UserCalibrationProfile Profile => profile;
        public bool IsProfileLoaded => profile != null;
        public bool IsAutoAttemptArmed => State == RawMediaPipeNeutralAutoSegmenterState.Armed;
        public float CalibrationProgress { get; private set; }
        public float NeutralHoldProgress { get; private set; }
        public float EndHoldProgress { get; private set; }
        public string LastFailure { get; private set; } = string.Empty;
        public NeutralObservationReason LastNeutralReason { get; private set; } = NeutralObservationReason.NoPose;
        public string AttemptId { get; private set; } = string.Empty;
        /// <summary>F32. How the last finalize was reached. A timeout says so.</summary>
        public NeutralEndpointResolution LastEndpointResolution { get; private set; } =
            NeutralEndpointResolution.None;
        /// <summary>F32. Why the END hold was last restarted.</summary>
        public NeutralEndpointHoldReset LastEndpointHoldReset { get; private set; } =
            NeutralEndpointHoldReset.None;

        public RawMediaPipeNeutralAutoSegmenter(
            string learnerProfileId,
            PetRobotNeutralAutoSegmentationSettings settings = null)
        {
            if (string.IsNullOrWhiteSpace(learnerProfileId))
                throw new ArgumentException("learner profile ID is required", nameof(learnerProfileId));
            this.learnerProfileId = learnerProfileId;
            this.settings = settings ?? PetRobotNeutralAutoSegmentationSettings.Default;
        }

        public void UseSavedProfile(UserCalibrationProfile savedProfile, CameraFingerprint currentCamera)
        {
            string failure;
            if (!PetRobotNeutralCalibrationProfileService.ValidateProfile(
                    savedProfile, learnerProfileId, out failure))
                throw new ArgumentException(failure, nameof(savedProfile));
            if (!PetRobotNeutralCalibrationProfileService.CameraMatches(savedProfile.Camera, currentCamera))
                throw new ArgumentException("camera fingerprint mismatch", nameof(currentCamera));
            profile = savedProfile;
            cameraFingerprint = CopyCamera(currentCamera);
            calibrationRequested = false;
            LastFailure = string.Empty;
            ResetTemporalState();
            SetState(attemptRequested
                ? RawMediaPipeNeutralAutoSegmenterState.WaitingForNeutral
                : RawMediaPipeNeutralAutoSegmenterState.Inactive);
        }

        public void RequireCalibration(CameraFingerprint currentCamera, string reason)
        {
            profile = null;
            cameraFingerprint = CopyCamera(currentCamera);
            LastFailure = reason ?? string.Empty;
            ResetTemporalState();
            SetState(RawMediaPipeNeutralAutoSegmenterState.NeedsCalibration);
        }

        public void ConfirmCalibrationSaved(UserCalibrationProfile savedProfile)
        {
            string failure;
            if (!PetRobotNeutralCalibrationProfileService.ValidateProfile(
                    savedProfile, learnerProfileId, out failure))
                throw new ArgumentException(failure, nameof(savedProfile));
            profile = savedProfile;
            calibrationRequested = false;
            CalibrationProgress = 1f;
            LastFailure = string.Empty;
            ResetTemporalState(false);
            SetState(attemptRequested
                ? RawMediaPipeNeutralAutoSegmenterState.WaitingForNeutral
                : RawMediaPipeNeutralAutoSegmenterState.Inactive);
        }

        public bool RequestAttempt()
        {
            if (State == RawMediaPipeNeutralAutoSegmenterState.Capturing ||
                State == RawMediaPipeNeutralAutoSegmenterState.EndCandidate)
                return false;
            attemptRequested = true;
            calibrationRequested = profile == null;
            AttemptId = Guid.NewGuid().ToString("D");
            ResetAttemptTimers();
            SetState(profile == null
                ? RawMediaPipeNeutralAutoSegmenterState.NeedsCalibration
                : RawMediaPipeNeutralAutoSegmenterState.WaitingForNeutral);
            return true;
        }

        public void RequestRecalibration(CameraFingerprint currentCamera)
        {
            RequireCalibration(currentCamera, "explicit recalibration requested");
            attemptRequested = true;
            calibrationRequested = true;
            AttemptId = Guid.NewGuid().ToString("D");
        }

        public void Cancel()
        {
            attemptRequested = false;
            calibrationRequested = false;
            AttemptId = string.Empty;
            ResetTemporalState();
            SetState(profile == null
                ? RawMediaPipeNeutralAutoSegmenterState.NeedsCalibration
                : RawMediaPipeNeutralAutoSegmenterState.Inactive);
        }

        public void Fail(string message)
        {
            LastFailure = string.IsNullOrWhiteSpace(message) ? "neutral auto segmentation failed" : message;
            SetState(RawMediaPipeNeutralAutoSegmenterState.Failed);
        }

        public RawMediaPipeNeutralAutoSegmenterDecision Process(
            RawMediaPipeObservation observation,
            bool cameraReady)
        {
            if (observation == null) throw new ArgumentNullException(nameof(observation));
            if (!LiveObservationConventionPolicy.IsAccepted(observation))
                throw new ArgumentException("segmenter requires an accepted live MediaPipe convention: " + LiveObservationConventionPolicy.RejectionMessage, nameof(observation));
            if (!cameraReady)
            {
                LastNeutralReason = NeutralObservationReason.CameraNotReady;
                if (State == RawMediaPipeNeutralAutoSegmenterState.Armed)
                    activationEvidenceStart = double.NaN;
                // F32: an unobserved stretch is not neutral evidence. The END hold is
                // discarded instead of being silently continued across the interruption.
                if (State == RawMediaPipeNeutralAutoSegmenterState.EndCandidate)
                    ResetEndCandidate(NeutralEndpointHoldReset.CameraNotReady,
                                      observation.TimestampSeconds);
                return RawMediaPipeNeutralAutoSegmenterDecision.None;
            }

            switch (State)
            {
                case RawMediaPipeNeutralAutoSegmenterState.NeedsCalibration:
                case RawMediaPipeNeutralAutoSegmenterState.CalibratingNeutral:
                    if (!calibrationRequested)
                        return RawMediaPipeNeutralAutoSegmenterDecision.None;
                    return ProcessCalibration(observation);
                case RawMediaPipeNeutralAutoSegmenterState.WaitingForNeutral:
                    return ProcessWaitingForNeutral(observation);
                case RawMediaPipeNeutralAutoSegmenterState.Armed:
                    return ProcessArmed(observation);
                case RawMediaPipeNeutralAutoSegmenterState.Capturing:
                    return ProcessCapturing(observation);
                case RawMediaPipeNeutralAutoSegmenterState.EndCandidate:
                    return ProcessEndCandidate(observation);
                default:
                    return RawMediaPipeNeutralAutoSegmenterDecision.None;
            }
        }

        RawMediaPipeNeutralAutoSegmenterDecision ProcessCalibration(RawMediaPipeObservation observation)
        {
            NeutralGeometry geometry = EvaluateCalibrationNeutral(observation);
            LastNeutralReason = geometry.Reason;
            if (!geometry.IsNeutral)
            {
                calibrationSamples.Clear();
                calibrationMotionSamples.Clear();
                previousCalibrationWrists = default(WristPair);
                calibrationStart = double.NaN;
                CalibrationProgress = 0f;
                SetState(RawMediaPipeNeutralAutoSegmenterState.NeedsCalibration);
                return RawMediaPipeNeutralAutoSegmenterDecision.None;
            }

            if (double.IsNaN(calibrationStart)) calibrationStart = observation.TimestampSeconds;
            SetState(RawMediaPipeNeutralAutoSegmenterState.CalibratingNeutral);
            calibrationSamples.Add(NeutralSample.From(geometry));
            WristPair wrists = geometry.Wrists;
            if (previousCalibrationWrists.AnyValid && wrists.AnyValid)
            {
                float displacement;
                if (TryMaximumCommonDisplacement(previousCalibrationWrists, wrists, out displacement))
                    calibrationMotionSamples.Add(displacement);
            }
            previousCalibrationWrists = wrists;
            double elapsed = observation.TimestampSeconds - calibrationStart;
            CalibrationProgress = Progress(elapsed, settings.NeutralCalibrationHoldSeconds);
            Changed?.Invoke();
            if (elapsed < settings.NeutralCalibrationHoldSeconds)
                return RawMediaPipeNeutralAutoSegmenterDecision.None;

            UserCalibrationProfile calibrated = BuildCalibrationProfile();
            return new RawMediaPipeNeutralAutoSegmenterDecision(calibrationProfile: calibrated);
        }

        RawMediaPipeNeutralAutoSegmenterDecision ProcessWaitingForNeutral(RawMediaPipeObservation observation)
        {
            NeutralGeometry geometry = EvaluateSavedNeutral(observation);
            LastNeutralReason = geometry.Reason;
            if (geometry.RequiredPoseValid && geometry.ArmsDown &&
                !PetRobotNeutralCalibrationProfileService.IsBodyScaleConsistent(profile, geometry.ShoulderWidth))
            {
                profile = null;
                calibrationRequested = attemptRequested;
                LastNeutralReason = NeutralObservationReason.BodyScaleMismatch;
                LastFailure = "current shoulder width clearly mismatches the saved neutral profile";
                ResetTemporalState();
                SetState(RawMediaPipeNeutralAutoSegmenterState.NeedsCalibration);
                return RawMediaPipeNeutralAutoSegmenterDecision.None;
            }

            if (!geometry.IsNeutral)
            {
                neutralHoldStart = double.NaN;
                NeutralHoldProgress = 0f;
                Changed?.Invoke();
                return RawMediaPipeNeutralAutoSegmenterDecision.None;
            }

            if (double.IsNaN(neutralHoldStart)) neutralHoldStart = observation.TimestampSeconds;
            double elapsed = observation.TimestampSeconds - neutralHoldStart;
            NeutralHoldProgress = Progress(elapsed, settings.ReadyNeutralHoldSeconds);
            Changed?.Invoke();
            if (elapsed < settings.ReadyNeutralHoldSeconds)
                return RawMediaPipeNeutralAutoSegmenterDecision.None;

            armedStart = observation.TimestampSeconds;
            activationEvidenceStart = double.NaN;
            SetState(RawMediaPipeNeutralAutoSegmenterState.Armed);
            return RawMediaPipeNeutralAutoSegmenterDecision.None;
        }

        RawMediaPipeNeutralAutoSegmenterDecision ProcessArmed(RawMediaPipeObservation observation)
        {
            if (observation.TimestampSeconds - armedStart >= settings.ArmedTimeoutSeconds)
            {
                ResetAttemptTimers();
                SetState(RawMediaPipeNeutralAutoSegmenterState.WaitingForNeutral);
                return new RawMediaPipeNeutralAutoSegmenterDecision(armedTimeoutReached: true);
            }

            NeutralGeometry geometry = EvaluatePoseAndArms(observation);
            LastNeutralReason = geometry.Reason;
            if (!geometry.RequiredPoseValid)
            {
                activationEvidenceStart = double.NaN;
                return RawMediaPipeNeutralAutoSegmenterDecision.None;
            }

            if (!PetRobotNeutralCalibrationProfileService.IsBodyScaleConsistent(
                    profile, geometry.ShoulderWidth))
            {
                LastNeutralReason = NeutralObservationReason.BodyScaleMismatch;
                activationEvidenceStart = double.NaN;
                return RawMediaPipeNeutralAutoSegmenterDecision.None;
            }

            bool handEntry = AnyValid(observation.LeftHand) || AnyValid(observation.RightHand);
            bool forearmRaised =
                (geometry.Wrists.LeftValid && !geometry.LeftWristBottomCompatible) ||
                (geometry.Wrists.RightValid && !geometry.RightWristBottomCompatible);
            bool activationEvidence = handEntry || !geometry.ArmsDown || forearmRaised;
            LastNeutralReason = handEntry
                ? NeutralObservationReason.HandVisible
                : !geometry.ArmsDown
                    ? NeutralObservationReason.ArmsNotDown
                    : forearmRaised
                        ? NeutralObservationReason.WristNotBottomCompatible
                        : NeutralObservationReason.Neutral;
            if (!activationEvidence)
            {
                activationEvidenceStart = double.NaN;
                return RawMediaPipeNeutralAutoSegmenterDecision.None;
            }

            if (double.IsNaN(activationEvidenceStart))
                activationEvidenceStart = observation.TimestampSeconds;
            if (observation.TimestampSeconds - activationEvidenceStart < ActivationEvidenceHoldSeconds)
                return RawMediaPipeNeutralAutoSegmenterDecision.None;

            captureStart = observation.TimestampSeconds;
            endCandidateStart = double.NaN;
            EndHoldProgress = 0f;
            SetState(RawMediaPipeNeutralAutoSegmenterState.Capturing);
            return new RawMediaPipeNeutralAutoSegmenterDecision(beginCaptureRequested: true);
        }

        RawMediaPipeNeutralAutoSegmenterDecision ProcessCapturing(RawMediaPipeObservation observation)
        {
            if (observation.TimestampSeconds - captureStart >= settings.MaxCaptureSeconds)
                return FinalizeMaxCapture(observation.TimestampSeconds);
            if (observation.TimestampSeconds - captureStart < settings.CaptureGraceSeconds)
                return RawMediaPipeNeutralAutoSegmenterDecision.None;

            NeutralGeometry geometry = EvaluateSavedNeutral(observation);
            LastNeutralReason = geometry.Reason;
            if (geometry.IsNeutral && IsFiniteSeconds(observation.TimestampSeconds))
            {
                endCandidateStart = observation.TimestampSeconds;
                lastEndObservationTimestamp = observation.TimestampSeconds;
                EndHoldProgress = 0f;
                LastEndpointHoldReset = NeutralEndpointHoldReset.None;
                LogEndpoint("end_candidate_opened", observation.TimestampSeconds, 0d, 0d);
                SetState(RawMediaPipeNeutralAutoSegmenterState.EndCandidate);
            }
            return RawMediaPipeNeutralAutoSegmenterDecision.None;
        }

        RawMediaPipeNeutralAutoSegmenterDecision ProcessEndCandidate(RawMediaPipeObservation observation)
        {
            double timestamp = observation.TimestampSeconds;
            if (IsFiniteSeconds(timestamp) &&
                timestamp - captureStart >= settings.MaxCaptureSeconds)
                return FinalizeMaxCapture(timestamp);

            // F32. A stable END may only be claimed over CONTINUOUSLY OBSERVED evidence: an
            // unobserved gap, a non-finite timestamp and a non-increasing timestamp are not
            // neutral frames and must never be accumulated into the hold. Only the endpoint
            // candidate is reset here; the capture, the attempt and the calibration/onset
            // state are untouched.
            if (!IsFiniteSeconds(timestamp) || !IsFiniteSeconds(lastEndObservationTimestamp) ||
                timestamp <= lastEndObservationTimestamp)
            {
                ResetEndCandidate(NeutralEndpointHoldReset.NonIncreasingTimestamp, timestamp);
                return RawMediaPipeNeutralAutoSegmenterDecision.None;
            }
            if (timestamp - lastEndObservationTimestamp > EndpointMaxObservationGapSeconds)
            {
                ResetEndCandidate(NeutralEndpointHoldReset.ObservationGap, timestamp);
                return RawMediaPipeNeutralAutoSegmenterDecision.None;
            }

            NeutralGeometry geometry = EvaluateSavedNeutral(observation);
            LastNeutralReason = geometry.Reason;
            if (!geometry.IsNeutral)
            {
                ResetEndCandidate(NeutralEndpointHoldReset.NeutralLost, timestamp);
                return RawMediaPipeNeutralAutoSegmenterDecision.None;
            }
            lastEndObservationTimestamp = timestamp;
            double elapsed = timestamp - endCandidateStart;
            EndHoldProgress = Progress(elapsed, settings.EndNeutralHoldSeconds);
            Changed?.Invoke();
            if (elapsed < settings.EndNeutralHoldSeconds)
                return RawMediaPipeNeutralAutoSegmenterDecision.None;

            LastEndpointResolution = NeutralEndpointResolution.StableNeutralHold;
            LastEndpointHoldReset = NeutralEndpointHoldReset.None;
            LogEndpoint("finalize_normal_end", timestamp, elapsed,
                        settings.AutoTrailingTrimSeconds);
            SetState(RawMediaPipeNeutralAutoSegmenterState.Evaluating);
            return new RawMediaPipeNeutralAutoSegmenterDecision(
                finalizeCaptureRequested: true,
                trailingTrimSeconds: settings.AutoTrailingTrimSeconds);
        }

        NeutralGeometry EvaluateCalibrationNeutral(RawMediaPipeObservation observation)
        {
            NeutralGeometry geometry = EvaluatePoseAndArms(observation);
            if (!geometry.RequiredPoseValid || !geometry.ArmsDown) return geometry;

            if ((geometry.Wrists.LeftValid && !geometry.LeftWristBottomCompatible) ||
                (geometry.Wrists.RightValid && !geometry.RightWristBottomCompatible))
                return geometry.WithFailure(NeutralObservationReason.WristNotBottomCompatible);

            if (geometry.HandsVisible &&
                (!geometry.Wrists.LeftValid || !geometry.Wrists.RightValid))
                return geometry.WithFailure(NeutralObservationReason.AmbiguousWristLoss);

            return geometry.AsNeutral();
        }

        NeutralGeometry EvaluateSavedNeutral(RawMediaPipeObservation observation)
        {
            NeutralGeometry geometry = EvaluatePoseAndArms(observation);
            if (!geometry.RequiredPoseValid || !geometry.ArmsDown || profile == null) return geometry;
            bool anyHandVisible = AnyValid(observation.LeftHand) || AnyValid(observation.RightHand);

            if (profile.Neutral.HandsVisible)
            {
                if (!geometry.Wrists.LeftValid || !geometry.Wrists.RightValid)
                    return geometry.WithFailure(NeutralObservationReason.AmbiguousWristLoss);
                float tolerance = (float)settings.OnsetDisplacementThreshold + profile.Neutral.MotionNoiseFloor;
                if (Vector2.Distance(geometry.Wrists.Left, profile.Neutral.LeftWristNeutralBR) > tolerance ||
                    Vector2.Distance(geometry.Wrists.Right, profile.Neutral.RightWristNeutralBR) > tolerance)
                    return geometry.WithFailure(NeutralObservationReason.WristBaselineMismatch);
                return geometry.AsNeutral();
            }

            if (anyHandVisible) return geometry.WithFailure(NeutralObservationReason.HandVisible);
            if (!profile.Neutral.BottomExitAllowed)
                return geometry.WithFailure(NeutralObservationReason.AmbiguousWristLoss);

            bool leftCompatible = ResolveBottomCompatibility(
                geometry.Wrists.LeftValid,
                geometry.LeftWristBottomCompatible,
                observation.TimestampSeconds,
                ref lastLeftBottomTimestamp);
            bool rightCompatible = ResolveBottomCompatibility(
                geometry.Wrists.RightValid,
                geometry.RightWristBottomCompatible,
                observation.TimestampSeconds,
                ref lastRightBottomTimestamp);
            if (!leftCompatible || !rightCompatible)
                return geometry.WithFailure(
                    (geometry.Wrists.LeftValid || geometry.Wrists.RightValid)
                        ? NeutralObservationReason.WristNotBottomCompatible
                        : NeutralObservationReason.AmbiguousWristLoss);
            return geometry.AsNeutral();
        }

        bool ResolveBottomCompatibility(
            bool wristValid,
            bool currentBottomCompatible,
            double timestamp,
            ref double lastBottomTimestamp)
        {
            if (wristValid)
            {
                if (currentBottomCompatible) lastBottomTimestamp = timestamp;
                return currentBottomCompatible;
            }
            if (!double.IsNaN(lastBottomTimestamp))
                return timestamp - lastBottomTimestamp <= settings.ReadyNeutralHoldSeconds;
            // Saved bottom-exit is never used alone: caller already required both hands absent and both arms down.
            return true;
        }

        NeutralGeometry EvaluatePoseAndArms(RawMediaPipeObservation observation)
        {
            RawLandmarkObservationSet pose = observation.Pose;
            if (!Valid(pose, LeftShoulder) || !Valid(pose, RightShoulder) ||
                !Valid(pose, LeftElbow) || !Valid(pose, RightElbow))
                return NeutralGeometry.Failure(NeutralObservationReason.NoPose);

            Vector2 leftShoulder = Point(pose, LeftShoulder);
            Vector2 rightShoulder = Point(pose, RightShoulder);
            Vector2 leftElbow = Point(pose, LeftElbow);
            Vector2 rightElbow = Point(pose, RightElbow);
            if (!Finite(leftShoulder) || !Finite(rightShoulder) || !Finite(leftElbow) || !Finite(rightElbow))
                return NeutralGeometry.Failure(NeutralObservationReason.NonFiniteGeometry);
            float shoulderWidth = Vector2.Distance(leftShoulder, rightShoulder);
            if (!IsFinite(shoulderWidth) || shoulderWidth < 0.01f)
                return NeutralGeometry.Failure(NeutralObservationReason.NonFiniteGeometry);

            float leftVertical = VerticalDownAngle(leftElbow - leftShoulder);
            float rightVertical = VerticalDownAngle(rightElbow - rightShoulder);
            if (!IsFinite(leftVertical) || !IsFinite(rightVertical))
                return NeutralGeometry.Failure(NeutralObservationReason.NonFiniteGeometry);
            bool armsDown = leftVertical <= settings.NeutralArmsDownMaxDegrees &&
                rightVertical <= settings.NeutralArmsDownMaxDegrees;

            WristPair wrists;
            TryGetBodyRelativeWrists(observation, out wrists);
            bool leftBottom = !wrists.LeftValid || BottomCompatible(pose, LeftElbow, LeftWrist);
            bool rightBottom = !wrists.RightValid || BottomCompatible(pose, RightElbow, RightWrist);
            return new NeutralGeometry(
                false,
                armsDown ? NeutralObservationReason.AmbiguousWristLoss : NeutralObservationReason.ArmsNotDown,
                true,
                armsDown,
                shoulderWidth,
                leftVertical,
                rightVertical,
                wrists,
                leftBottom,
                rightBottom,
                AnyValid(observation.LeftHand) || AnyValid(observation.RightHand));
        }

        bool TryGetBodyRelativeWrists(RawMediaPipeObservation observation, out WristPair wrists)
        {
            RawLandmarkObservationSet pose = observation.Pose;
            wrists = default(WristPair);
            if (!Valid(pose, LeftShoulder) || !Valid(pose, RightShoulder)) return false;
            Vector2 leftShoulder = Point(pose, LeftShoulder);
            Vector2 rightShoulder = Point(pose, RightShoulder);
            float scale = Vector2.Distance(leftShoulder, rightShoulder);
            if (!IsFinite(scale) || scale < 0.01f) return false;
            Vector2 origin = (leftShoulder + rightShoulder) * 0.5f;
            wrists.LeftValid = Valid(pose, LeftWrist);
            wrists.RightValid = Valid(pose, RightWrist);
            if (wrists.LeftValid) wrists.Left = (Point(pose, LeftWrist) - origin) / scale;
            if (wrists.RightValid) wrists.Right = (Point(pose, RightWrist) - origin) / scale;
            if ((wrists.LeftValid && !Finite(wrists.Left)) || (wrists.RightValid && !Finite(wrists.Right)))
            {
                wrists = default(WristPair);
                return false;
            }
            return true;
        }

        bool BottomCompatible(RawLandmarkObservationSet pose, int elbowIndex, int wristIndex)
        {
            if (!Valid(pose, elbowIndex) || !Valid(pose, wristIndex)) return false;
            Vector2 elbow = Point(pose, elbowIndex);
            Vector2 wrist = Point(pose, wristIndex);
            Vector2 direction = wrist - elbow;
            return wrist.y <= elbow.y && IsFinite(VerticalDownAngle(direction)) &&
                VerticalDownAngle(direction) <= settings.NeutralArmsDownMaxDegrees;
        }

        UserCalibrationProfile BuildCalibrationProfile()
        {
            if (calibrationSamples.Count == 0 || cameraFingerprint == null)
                throw new InvalidOperationException("calibration samples and camera fingerprint are required");
            var shoulderWidths = new List<float>();
            var leftVertical = new List<float>();
            var rightVertical = new List<float>();
            var leftWrists = new List<Vector2>();
            var rightWrists = new List<Vector2>();
            int handsVisibleCount = 0;
            foreach (NeutralSample sample in calibrationSamples)
            {
                shoulderWidths.Add(sample.ShoulderWidth);
                leftVertical.Add(sample.LeftVertical);
                rightVertical.Add(sample.RightVertical);
                if (sample.LeftWristValid) leftWrists.Add(sample.LeftWrist);
                if (sample.RightWristValid) rightWrists.Add(sample.RightWrist);
                if (sample.HandsVisible) handsVisibleCount++;
            }

            bool handsVisible = handsVisibleCount * 2 > calibrationSamples.Count;
            var result = new UserCalibrationProfile
            {
                SchemaVersion = PetRobotNeutralCalibrationProfileService.SupportedSchemaVersion,
                UserProfileId = learnerProfileId,
                BodyBaseline = new StableBodyBaseline
                {
                    AcceptedSessionCount = 1,
                    Aggregation = "robust_median_mad",
                    LastAcceptedSessionId = AttemptId,
                },
                Camera = CopyCamera(cameraFingerprint),
                BodyScale = new BodyScaleData { ShoulderWidthNorm = Median(shoulderWidths) },
                Neutral = new NeutralPoseData
                {
                    HandsVisible = handsVisible,
                    BottomExitAllowed = !handsVisible,
                    HasArmVerticalBaseline = true,
                    LeftArmVerticalNeutral = Median(leftVertical),
                    RightArmVerticalNeutral = Median(rightVertical),
                    HasWristBaseline = leftWrists.Count > 0 && rightWrists.Count > 0,
                    LeftWristNeutralBR = Median(leftWrists),
                    RightWristNeutralBR = Median(rightWrists),
                    MotionNoiseFloor = calibrationMotionSamples.Count == 0 ? 0f : Median(calibrationMotionSamples),
                },
                SafeFrame = new SafeFrameData(),
                Quality = new CalibrationQuality { Status = "good", ValidFrameRatio = 1f },
            };
            string failure;
            if (!PetRobotNeutralCalibrationProfileService.ValidateProfile(result, learnerProfileId, out failure))
                throw new InvalidOperationException("calibrated neutral profile is invalid: " + failure);
            return result;
        }

        static bool TryMaximumCommonDisplacement(WristPair previous, WristPair current, out float displacement)
        {
            displacement = 0f;
            bool comparable = false;
            if (previous.LeftValid && current.LeftValid)
            {
                displacement = Vector2.Distance(previous.Left, current.Left);
                comparable = true;
            }
            if (previous.RightValid && current.RightValid)
            {
                displacement = comparable
                    ? Mathf.Max(displacement, Vector2.Distance(previous.Right, current.Right))
                    : Vector2.Distance(previous.Right, current.Right);
                comparable = true;
            }
            return comparable && IsFinite(displacement);
        }

        /// <summary>
        /// F32. Discards ONLY the endpoint candidate and returns to Capturing, so the next
        /// continuously observed neutral stretch has to earn the whole 1 s hold again. It
        /// never ends, truncates or fails the capture.
        /// </summary>
        void ResetEndCandidate(NeutralEndpointHoldReset reason, double timestamp)
        {
            if (State != RawMediaPipeNeutralAutoSegmenterState.EndCandidate) return;
            double heldFor = IsFiniteSeconds(endCandidateStart) && IsFiniteSeconds(timestamp)
                ? timestamp - endCandidateStart
                : double.NaN;
            LastEndpointResolution = NeutralEndpointResolution.None;
            LastEndpointHoldReset = reason;
            LogEndpoint("end_hold_reset_" + reason, timestamp, heldFor, 0d);
            endCandidateStart = double.NaN;
            lastEndObservationTimestamp = double.NaN;
            EndHoldProgress = 0f;
            SetState(RawMediaPipeNeutralAutoSegmenterState.Capturing);
        }

        /// <summary>
        /// The 10 s capture bound. This is a TIMEOUT, reported as one: no stable end is
        /// claimed and no trailing trim is applied.
        /// </summary>
        RawMediaPipeNeutralAutoSegmenterDecision FinalizeMaxCapture(double timestamp)
        {
            LastEndpointResolution = NeutralEndpointResolution.MaxCaptureTimeout;
            LogEndpoint("finalize_max_capture_timeout", timestamp,
                        IsFiniteSeconds(endCandidateStart) && IsFiniteSeconds(timestamp)
                            ? timestamp - endCandidateStart : double.NaN,
                        0d);
            endCandidateStart = double.NaN;
            lastEndObservationTimestamp = double.NaN;
            EndHoldProgress = 0f;
            SetState(RawMediaPipeNeutralAutoSegmenterState.Evaluating);
            return new RawMediaPipeNeutralAutoSegmenterDecision(
                finalizeCaptureRequested: true, maxCaptureReached: true, trailingTrimSeconds: 0d);
        }

        static bool IsFiniteSeconds(double value) =>
            !double.IsNaN(value) && !double.IsInfinity(value);

        /// <summary>
        /// F32. Endpoint diagnostics at TRANSITIONS and finalize only: numbers and a phase
        /// name, never a per-frame line, never a raw coordinate and never a participant
        /// identity. The attempt token is the local per-attempt GUID prefix.
        /// </summary>
        void LogEndpoint(string phase, double timestamp, double neutralElapsedSeconds,
                         double trimSeconds)
        {
            Debug.Log(string.Format(
                CultureInfo.InvariantCulture,
                "[NeutralEndpoint] attempt={0} phase={1} capture_start={2:0.###} " +
                "end_candidate_start={3:0.###} at={4:0.###} neutral_elapsed={5:0.###} " +
                "trim={6:0.###} max_gap={7:0.###}",
                AttemptToken(), phase, captureStart, endCandidateStart, timestamp,
                neutralElapsedSeconds, trimSeconds, EndpointMaxObservationGapSeconds));
        }

        string AttemptToken() =>
            string.IsNullOrEmpty(AttemptId) ? "none"
                : AttemptId.Length <= 8 ? AttemptId : AttemptId.Substring(0, 8);

        void ResetTemporalState(bool resetProgress = true)
        {
            calibrationSamples.Clear();
            calibrationMotionSamples.Clear();
            calibrationStart = double.NaN;
            previousCalibrationWrists = default(WristPair);
            lastLeftBottomTimestamp = double.NaN;
            lastRightBottomTimestamp = double.NaN;
            ResetAttemptTimers();
            if (resetProgress) CalibrationProgress = 0f;
        }

        void ResetAttemptTimers()
        {
            neutralHoldStart = double.NaN;
            armedStart = double.NaN;
            captureStart = double.NaN;
            endCandidateStart = double.NaN;
            lastEndObservationTimestamp = double.NaN;
            LastEndpointResolution = NeutralEndpointResolution.None;
            LastEndpointHoldReset = NeutralEndpointHoldReset.None;
            activationEvidenceStart = double.NaN;
            NeutralHoldProgress = 0f;
            EndHoldProgress = 0f;
        }

        void SetState(RawMediaPipeNeutralAutoSegmenterState value)
        {
            if (State == value) return;
            State = value;
            Changed?.Invoke();
        }

        static bool Valid(RawLandmarkObservationSet set, int index) =>
            set != null && index >= 0 && index < set.LandmarkCount && set.IsValid(index);
        static Vector2 Point(RawLandmarkObservationSet set, int index) =>
            new Vector2((float)set.Coordinate(index, 0), (float)set.Coordinate(index, 1));
        static bool AnyValid(RawLandmarkObservationSet set)
        {
            if (set == null) return false;
            for (int index = 0; index < set.LandmarkCount; index++) if (set.IsValid(index)) return true;
            return false;
        }
        static float VerticalDownAngle(Vector2 direction)
        {
            if (!Finite(direction) || direction.sqrMagnitude <= 0f) return float.NaN;
            return Vector2.Angle(Vector2.down, direction);
        }
        static float Progress(double elapsed, double duration) =>
            (float)Math.Max(0d, Math.Min(1d, elapsed / duration));
        static bool Finite(Vector2 value) => IsFinite(value.x) && IsFinite(value.y);
        static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);

        static float Median(List<float> values)
        {
            if (values == null || values.Count == 0) return 0f;
            float[] sorted = values.ToArray();
            Array.Sort(sorted);
            int middle = sorted.Length / 2;
            return sorted.Length % 2 == 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) * 0.5f;
        }

        static Vector2 Median(List<Vector2> values)
        {
            if (values == null || values.Count == 0) return Vector2.zero;
            var x = new List<float>(values.Count);
            var y = new List<float>(values.Count);
            foreach (Vector2 value in values) { x.Add(value.x); y.Add(value.y); }
            return new Vector2(Median(x), Median(y));
        }

        static CameraFingerprint CopyCamera(CameraFingerprint value)
        {
            if (value == null) return null;
            return new CameraFingerprint
            {
                DeviceName = value.DeviceName,
                Width = value.Width,
                Height = value.Height,
                Fps = value.Fps,
                RotationDegrees = value.RotationDegrees,
                PreviewMirrored = value.PreviewMirrored,
            };
        }

        struct WristPair
        {
            public bool LeftValid;
            public bool RightValid;
            public Vector2 Left;
            public Vector2 Right;
            public bool AnyValid => LeftValid || RightValid;
        }

        struct NeutralSample
        {
            public float ShoulderWidth;
            public float LeftVertical;
            public float RightVertical;
            public bool LeftWristValid;
            public bool RightWristValid;
            public Vector2 LeftWrist;
            public Vector2 RightWrist;
            public bool HandsVisible;

            public static NeutralSample From(NeutralGeometry value) => new NeutralSample
            {
                ShoulderWidth = value.ShoulderWidth,
                LeftVertical = value.LeftVertical,
                RightVertical = value.RightVertical,
                LeftWristValid = value.Wrists.LeftValid,
                RightWristValid = value.Wrists.RightValid,
                LeftWrist = value.Wrists.Left,
                RightWrist = value.Wrists.Right,
                HandsVisible = value.HandsVisible,
            };
        }

        struct NeutralGeometry
        {
            public bool IsNeutral;
            public NeutralObservationReason Reason;
            public bool RequiredPoseValid;
            public bool ArmsDown;
            public float ShoulderWidth;
            public float LeftVertical;
            public float RightVertical;
            public WristPair Wrists;
            public bool LeftWristBottomCompatible;
            public bool RightWristBottomCompatible;
            public bool HandsVisible;

            public NeutralGeometry(
                bool isNeutral,
                NeutralObservationReason reason,
                bool requiredPoseValid,
                bool armsDown,
                float shoulderWidth,
                float leftVertical,
                float rightVertical,
                WristPair wrists,
                bool leftWristBottomCompatible,
                bool rightWristBottomCompatible,
                bool handsVisible)
            {
                IsNeutral = isNeutral;
                Reason = reason;
                RequiredPoseValid = requiredPoseValid;
                ArmsDown = armsDown;
                ShoulderWidth = shoulderWidth;
                LeftVertical = leftVertical;
                RightVertical = rightVertical;
                Wrists = wrists;
                LeftWristBottomCompatible = leftWristBottomCompatible;
                RightWristBottomCompatible = rightWristBottomCompatible;
                HandsVisible = handsVisible;
            }

            public static NeutralGeometry Failure(NeutralObservationReason reason) =>
                new NeutralGeometry(false, reason, false, false, 0f, 0f, 0f,
                    default(WristPair), false, false, false);
            public NeutralGeometry AsNeutral()
            {
                IsNeutral = true;
                Reason = NeutralObservationReason.Neutral;
                return this;
            }
            public NeutralGeometry WithFailure(NeutralObservationReason reason)
            {
                IsNeutral = false;
                Reason = reason;
                return this;
            }
        }
    }
}
