using System;
using UnityEngine;
using WordsInSilence.Core;
using WordsInSilence.Pose;
using WordsInSilence.Recognition.Diagnostics;

namespace WordsInSilence.Recognition.LiveInput
{
    /// <summary>Stateful mapper for the project's existing HandPoseIntegrator output.</summary>
    public sealed class MediaPipeHandPoseObservationMapper
    {
        readonly bool[] previousPose = new bool[RawMediaPipeObservation.PoseLandmarkCount];
        readonly bool[] previousLeft = new bool[RawMediaPipeObservation.HandLandmarkCount];
        readonly bool[] previousRight = new bool[RawMediaPipeObservation.HandLandmarkCount];

        public RawMediaPipeObservation Map(HandPoseFrame frame)
        {
            if (frame == null) throw new ArgumentNullException(nameof(frame));
            double timestamp = frame.Timestamp;
            if (double.IsNaN(timestamp) || double.IsInfinity(timestamp) || timestamp < 0.0)
                throw new ArgumentOutOfRangeException(nameof(frame), "provider timestamp is invalid");

            RawLandmarkObservationSet pose = MapPose(frame.PoseFrame, previousPose);
            LandmarkFrame sourceHands = frame.HandFrame;
            HandLandmarks left = null;
            HandLandmarks right = null;
            if (sourceHands?.Hands != null)
            {
                foreach (HandLandmarks hand in sourceHands.Hands)
                {
                    if (hand == null) continue;
                    if (hand.IsLeftHand)
                    {
                        if (left != null) throw new InvalidOperationException("provider emitted duplicate Left handedness categories");
                        left = hand;
                    }
                    else
                    {
                        if (right != null) throw new InvalidOperationException("provider emitted duplicate Right handedness categories");
                        right = hand;
                    }
                }
            }
            RawLandmarkObservationSet leftSet = MapHand(left, previousLeft);
            RawLandmarkObservationSet rightSet = MapHand(right, previousRight);
            // read the mapped geometry the providers recorded; the environment is never
            // re-read here, and a mixed or unknown pair keeps the legacy convention so the
            // existing downstream contract check stays the authority
            RawObservationConvention convention =
                UprightInferencePixelBridge.IsUprightTextureGeometry(frame.MergedGeometry)
                    ? RawObservationConvention.LiveMediaPipeUprightTexture
                    : RawObservationConvention.LiveMediaPipe;
            return new RawMediaPipeObservation(timestamp, convention, pose, leftSet, rightSet, frame.MergedGeometry);
        }

        public void Reset()
        {
            Array.Clear(previousPose, 0, previousPose.Length);
            Array.Clear(previousLeft, 0, previousLeft.Length);
            Array.Clear(previousRight, 0, previousRight.Length);
        }

        static RawLandmarkObservationSet MapPose(UpperBodyPoseFrame frame, bool[] previous)
        {
            int count = RawMediaPipeObservation.PoseLandmarkCount;
            var coordinates = NaNCoordinates(count);
            var valid = new bool[count];
            var reappearance = new bool[count];
            if (frame?.AllPoints != null)
            {
                if (frame.AllPoints.Length != count) throw new InvalidOperationException($"pose provider returned {frame.AllPoints.Length} landmarks; expected {count}");
                for (int landmark = 0; landmark < count; landmark++)
                {
                    Vector3 point = frame.AllPoints[landmark];
                    bool finite = IsFinite(point.x) && IsFinite(point.y) && IsFinite(point.z);
                    valid[landmark] = finite;
                    reappearance[landmark] = finite && !previous[landmark];
                    if (finite) WritePoint(coordinates, landmark, point.x, point.y, point.z);
                }
            }
            Array.Copy(valid, previous, count);
            return new RawLandmarkObservationSet(count, coordinates, valid, reappearance, new bool[count]);
        }

        static RawLandmarkObservationSet MapHand(HandLandmarks hand, bool[] previous)
        {
            int count = RawMediaPipeObservation.HandLandmarkCount;
            var coordinates = NaNCoordinates(count);
            var valid = new bool[count];
            var reappearance = new bool[count];
            if (hand?.Points != null)
            {
                if (hand.Points.Length != count) throw new InvalidOperationException($"hand provider returned {hand.Points.Length} landmarks; expected {count}");
                for (int landmark = 0; landmark < count; landmark++)
                {
                    LandmarkPoint point = hand.Points[landmark];
                    bool finite = point != null && IsFinite(point.X) && IsFinite(point.Y) && IsFinite(point.Z);
                    valid[landmark] = finite;
                    reappearance[landmark] = finite && !previous[landmark];
                    if (finite) WritePoint(coordinates, landmark, point.X, point.Y, point.Z);
                }
            }
            Array.Copy(valid, previous, count);
            return new RawLandmarkObservationSet(count, coordinates, valid, reappearance, new bool[count]);
        }

        static double[] NaNCoordinates(int count)
        {
            var values = new double[count * 3];
            for (int index = 0; index < values.Length; index++) values[index] = double.NaN;
            return values;
        }

        static void WritePoint(double[] values, int landmark, double x, double y, double z)
        {
            int offset = landmark * 3;
            values[offset] = x;
            values[offset + 1] = y;
            values[offset + 2] = z;
        }

        static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
    }

    /// <summary>Thin event bridge; it owns no camera or MediaPipe task resources.</summary>
    public sealed class MediaPipeHandPoseObservationBridge : MonoBehaviour
    {
        [SerializeField] HandPoseIntegrator _integrator;
        readonly MediaPipeHandPoseObservationMapper mapper = new MediaPipeHandPoseObservationMapper();

        public bool IsBound { get; private set; }
        public bool HasReceivedObservation { get; private set; }
        public int ObservationCount { get; private set; }
        public bool LastLeftPresent { get; private set; }
        public bool LastRightPresent { get; private set; }
        public string LastError { get; private set; } = string.Empty;
        public event Action<RawMediaPipeObservation> ObservationArrived;
        public event Action<Exception> BridgeFailed;

        void OnEnable()
        {
            if (_integrator == null) _integrator = FindFirstObjectByType<HandPoseIntegrator>();
            if (_integrator != null)
            {
                _integrator.UnifiedFrameArrived += OnUnifiedFrame;
                IsBound = true;
            }
            else
            {
                IsBound = false;
                LastError = "HandPoseIntegrator was not found; open the existing 02_PoseIntegration provider scene and add this diagnostic prefab.";
            }
        }

        void OnDisable()
        {
            if (_integrator != null) _integrator.UnifiedFrameArrived -= OnUnifiedFrame;
            IsBound = false;
            mapper.Reset();
        }

        public void BindForDiagnostic(HandPoseIntegrator integrator)
        {
            if (isActiveAndEnabled && _integrator != null) _integrator.UnifiedFrameArrived -= OnUnifiedFrame;
            _integrator = integrator;
            if (isActiveAndEnabled && _integrator != null) _integrator.UnifiedFrameArrived += OnUnifiedFrame;
            IsBound = _integrator != null;
            LastError = IsBound ? string.Empty : "HandPoseIntegrator is not assigned";
        }

        public void ResetObservationState()
        {
            mapper.Reset();
            HasReceivedObservation = false;
            ObservationCount = 0;
            LastLeftPresent = false;
            LastRightPresent = false;
            LastError = string.Empty;
        }

        void OnUnifiedFrame(HandPoseFrame frame)
        {
            try
            {
                RawMediaPipeObservation observation = mapper.Map(frame);
                HasReceivedObservation = true;
                ObservationCount++;
                LastLeftPresent = AnyValid(observation.LeftHand);
                LastRightPresent = AnyValid(observation.RightHand);
                LastError = string.Empty;
                ObservationArrived?.Invoke(observation);
            }
            catch (Exception exception)
            {
                LastError = exception.Message;
                BridgeFailed?.Invoke(exception);
            }
        }

        static bool AnyValid(RawLandmarkObservationSet set)
        {
            for (int landmark = 0; landmark < set.LandmarkCount; landmark++) if (set.IsValid(landmark)) return true;
            return false;
        }
    }
}
