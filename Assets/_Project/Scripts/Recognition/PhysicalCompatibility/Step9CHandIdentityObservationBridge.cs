using System;
using UnityEngine;
using WordsInSilence.Pose;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Recognition.PhysicalCompatibility
{
    /// <summary>
    /// Step 9C-only bridge. Approved Step 9B mapping remains unchanged; this component resolves
    /// only objectively separable duplicate provider categories before calling that public mapper.
    /// </summary>
    public sealed class Step9CHandIdentityObservationBridge : MonoBehaviour
    {
        [SerializeField] HandPoseIntegrator _integrator;
        readonly MediaPipeHandPoseObservationMapper mapper = new MediaPipeHandPoseObservationMapper();
        readonly Step9CHandIdentityResolver resolver = new Step9CHandIdentityResolver();

        public bool IsBound { get; private set; }
        public bool HasReceivedObservation { get; private set; }
        public int ProviderFrameCount { get; private set; }
        public int ObservationCount { get; private set; }
        public int DuplicateFrameCount { get; private set; }
        public int ResolvedDuplicateFrameCount { get; private set; }
        public int AmbiguousDuplicateFrameCount { get; private set; }
        public long LifetimeProviderFrameCount { get; private set; }
        public long LastLifetimeProviderFrameIndex { get; private set; } = -1;
        public double FirstProviderTimestampSeconds { get; private set; } = double.NaN;
        public double LastProviderTimestampSeconds { get; private set; } = double.NaN;
        public double LastLifetimeProviderTimestampSeconds { get; private set; } = double.NaN;
        public bool LastLeftPresent { get; private set; }
        public bool LastRightPresent { get; private set; }
        public string LastError { get; private set; } = string.Empty;
        public Step9CHandIdentityDiagnostic LastHandIdentityDiagnostic { get; private set; }
        public event Action<RawMediaPipeObservation> ObservationArrived;
        public event Action<Step9CHandIdentityDiagnostic> HandIdentityDiagnosticObserved;
        public event Action<Exception> BridgeFailed;

        void OnEnable()
        {
            if (_integrator == null) _integrator = FindFirstObjectByType<HandPoseIntegrator>();
            BindCurrent();
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
            BindCurrent();
        }

        public void ResetObservationState()
        {
            mapper.Reset();
            HasReceivedObservation = false;
            ProviderFrameCount = 0;
            ObservationCount = 0;
            DuplicateFrameCount = 0;
            ResolvedDuplicateFrameCount = 0;
            AmbiguousDuplicateFrameCount = 0;
            LastLeftPresent = false;
            LastRightPresent = false;
            LastError = string.Empty;
            LastHandIdentityDiagnostic = null;
            FirstProviderTimestampSeconds = double.NaN;
            LastProviderTimestampSeconds = double.NaN;
        }

        public Step9CProviderBoundary SnapshotBoundary() => new Step9CProviderBoundary(
            ProviderFrameCount,
            ObservationCount,
            LifetimeProviderFrameCount,
            LastLifetimeProviderFrameIndex,
            FirstProviderTimestampSeconds,
            LastProviderTimestampSeconds,
            LastLifetimeProviderTimestampSeconds);

        void BindCurrent()
        {
            if (isActiveAndEnabled && _integrator != null)
            {
                _integrator.UnifiedFrameArrived += OnUnifiedFrame;
                IsBound = true;
                LastError = string.Empty;
            }
            else
            {
                IsBound = false;
                LastError = _integrator == null
                    ? "HandPoseIntegrator was not found; open the existing 02_PoseIntegration provider scene and add this diagnostic prefab."
                    : "Step 9C bridge is not active";
            }
        }

        void OnUnifiedFrame(HandPoseFrame frame)
        {
            int frameIndex = ProviderFrameCount++;
            long lifetimeFrameIndex = LifetimeProviderFrameCount++;
            LastLifetimeProviderFrameIndex = lifetimeFrameIndex;
            double timestamp = frame == null ? double.NaN : frame.Timestamp;
            if (frameIndex == 0) FirstProviderTimestampSeconds = timestamp;
            LastProviderTimestampSeconds = timestamp;
            LastLifetimeProviderTimestampSeconds = timestamp;
            try
            {
                HandPoseFrame resolved = resolver.Resolve(frame, frameIndex, out Step9CHandIdentityDiagnostic diagnostic);
                if (diagnostic != null)
                {
                    RecordDiagnostic(diagnostic);
                    ResolvedDuplicateFrameCount++;
                }
                RawMediaPipeObservation observation = mapper.Map(resolved);
                HasReceivedObservation = true;
                ObservationCount++;
                LastLeftPresent = AnyValid(observation.LeftHand);
                LastRightPresent = AnyValid(observation.RightHand);
                LastError = string.Empty;
                ObservationArrived?.Invoke(observation);
            }
            catch (Step9CAmbiguousHandIdentityException exception)
            {
                RecordDiagnostic(exception.Diagnostic);
                AmbiguousDuplicateFrameCount++;
                LastError = exception.Message;
                BridgeFailed?.Invoke(exception);
            }
            catch (Exception exception)
            {
                LastError = exception.Message;
                BridgeFailed?.Invoke(exception);
            }
        }

        void RecordDiagnostic(Step9CHandIdentityDiagnostic diagnostic)
        {
            LastHandIdentityDiagnostic = diagnostic;
            DuplicateFrameCount++;
            HandIdentityDiagnosticObserved?.Invoke(diagnostic);
        }

        static bool AnyValid(RawLandmarkObservationSet set)
        {
            for (int landmark = 0; landmark < set.LandmarkCount; landmark++) if (set.IsValid(landmark)) return true;
            return false;
        }
    }
}
