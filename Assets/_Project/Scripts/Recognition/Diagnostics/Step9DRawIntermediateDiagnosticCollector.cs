using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.PhysicalCompatibility;

namespace WordsInSilence.Recognition.Diagnostics
{
    public enum Step9DRawIntermediateCollectorState
    {
        Idle,
        Capturing,
        Finalized,
    }

    public sealed class Step9DRawIntermediateDiagnosticSnapshot
    {
        readonly RawMediaPipeObservation[] observations;
        readonly Step9DRawIntermediateFrameIdentity[] frameIdentities;

        public string RunId { get; }
        public string CaptureId { get; }
        public string ExpectedWord { get; }
        public ApprovedPreprocessingResult Result { get; }
        public Step9DRawIntermediatePreprocessingTrace Trace { get; }
        public int ObservationCount => observations.Length;

        /// <summary>
        /// Optional inference image geometry captured at diagnostic start.
        /// Null when geometry was not supplied (backward compatible).
        /// </summary>
        public Step9DInferenceImageGeometry InferenceImageGeometry { get; }

        /// <summary>
        /// Capture-level merged inference geometry from the first observation that carried it.
        /// Null when no observation carried geometry.
        /// </summary>
        public Step9DMergedInferenceGeometry CaptureInferenceGeometry { get; }

        /// <summary>
        /// Capture-wide geometry consistency summary. Null when geometry was not required or not present.
        /// </summary>
        public Step9DCaptureGeometryConsistencySummary GeometryConsistencySummary { get; }

        /// <summary>
        /// Per-take transient hand identity drops tolerated during capture.
        /// Empty when no ambiguity was tolerated. Null for legacy snapshots without provenance.
        /// </summary>
        public IReadOnlyList<Step9DAmbiguityDropRecord> TransientHandIdentityDrops { get; internal set; }

        internal Step9DRawIntermediateDiagnosticSnapshot(
            string runId,
            string captureId,
            string expectedWord,
            RawMediaPipeObservation[] sourceObservations,
            Step9DRawIntermediateFrameIdentity[] identities,
            ApprovedPreprocessingTraceResult preprocessing,
            Step9DInferenceImageGeometry inferenceImageGeometry = null,
            Step9DMergedInferenceGeometry captureInferenceGeometry = null,
            Step9DCaptureGeometryConsistencySummary geometryConsistencySummary = null)
        {
            RunId = RequireText(runId, nameof(runId));
            CaptureId = RequireText(captureId, nameof(captureId));
            ExpectedWord = RequireText(expectedWord, nameof(expectedWord));
            if (sourceObservations == null) throw new ArgumentNullException(nameof(sourceObservations));
            if (identities == null) throw new ArgumentNullException(nameof(identities));
            if (preprocessing == null) throw new ArgumentNullException(nameof(preprocessing));
            observations = CloneObservations(sourceObservations);
            frameIdentities = CloneIdentities(identities);
            Result = preprocessing.Result;
            Trace = preprocessing.Trace.DeepCopy();
            InferenceImageGeometry = inferenceImageGeometry?.Clone();
            CaptureInferenceGeometry = captureInferenceGeometry?.Clone();
            GeometryConsistencySummary = geometryConsistencySummary;
        }

        public RawMediaPipeObservation[] CopyObservations() => CloneObservations(observations);
        public Step9DRawIntermediateFrameIdentity[] CopyFrameIdentities() => CloneIdentities(frameIdentities);

        static string RequireText(string value, string parameter)
        {
            if (string.IsNullOrWhiteSpace(value)) throw new ArgumentException("value is required", parameter);
            return value;
        }

        internal static RawMediaPipeObservation CloneObservation(RawMediaPipeObservation value)
        {
            if (value == null) throw new ArgumentNullException(nameof(value));
            var convention = new RawObservationConvention(
                value.Convention.Id,
                value.Convention.CoordinateSpace,
                value.Convention.AxisAndDepth,
                value.Convention.Handedness,
                value.Convention.Mirror);
            return new RawMediaPipeObservation(
                value.TimestampSeconds,
                convention,
                CloneSet(value.Pose),
                CloneSet(value.LeftHand),
                CloneSet(value.RightHand),
                value.MergedGeometry?.Clone());
        }

        static RawLandmarkObservationSet CloneSet(RawLandmarkObservationSet value)
        {
            var reappearance = new bool[value.LandmarkCount];
            var strong = new bool[value.LandmarkCount];
            for (int landmark = 0; landmark < value.LandmarkCount; landmark++)
            {
                reappearance[landmark] = value.IsReappearance(landmark);
                strong[landmark] = value.IsStrongDepthBoundary(landmark);
            }
            return new RawLandmarkObservationSet(
                value.LandmarkCount,
                value.CopyCoordinates(),
                value.CopyValidMask(),
                reappearance,
                strong);
        }

        static RawMediaPipeObservation[] CloneObservations(RawMediaPipeObservation[] values)
        {
            var copy = new RawMediaPipeObservation[values.Length];
            for (int index = 0; index < values.Length; index++) copy[index] = CloneObservation(values[index]);
            return copy;
        }

        static Step9DRawIntermediateFrameIdentity[] CloneIdentities(Step9DRawIntermediateFrameIdentity[] values)
        {
            var copy = new Step9DRawIntermediateFrameIdentity[values.Length];
            for (int index = 0; index < values.Length; index++)
                copy[index] = values[index]?.Clone() ?? throw new ArgumentException($"identity {index} is null", nameof(values));
            return copy;
        }
    }

    /// <summary>
    /// Opt-in source-only collector. It never writes files and is not attached to a scene or prefab.
    /// A caller must explicitly attach a bridge, begin a capture and invoke finalization.
    /// </summary>
    public sealed class Step9DRawIntermediateDiagnosticCollector : IDisposable
    {
        readonly ApprovedLivePreprocessingAdapter adapter;
        readonly List<RawMediaPipeObservation> observations = new List<RawMediaPipeObservation>();
        readonly List<Step9DRawIntermediateFrameIdentity> identities = new List<Step9DRawIntermediateFrameIdentity>();
        readonly List<Step9DRawIntermediateDiagnosticSnapshot> completed = new List<Step9DRawIntermediateDiagnosticSnapshot>();
        readonly HashSet<string> finalizedCaptureIds = new HashSet<string>(StringComparer.Ordinal);
        Step9CHandIdentityObservationBridge bridge;
        Step9DInferenceImageGeometry inferenceImageGeometry;
        bool geometryRequired;
        Step9DInferenceFrameGeometry frozenHandContract;
        Step9DInferenceFrameGeometry frozenPoseContract;
        int geometryCheckedCount;
        long firstHandTimestampMs;
        long lastHandTimestampMs;
        long firstPoseTimestampMs;
        long lastPoseTimestampMs;
        long maxAbsSkewMs;
        string runId;
        string captureId;
        string expectedWord;

        public Step9DRawIntermediateCollectorState State { get; private set; } = Step9DRawIntermediateCollectorState.Idle;
        public int BufferedObservationCount => observations.Count;
        public int AdaptWithTraceInvocationCount { get; private set; }
        public IReadOnlyList<Step9DRawIntermediateDiagnosticSnapshot> CompletedSnapshots =>
            new ReadOnlyCollection<Step9DRawIntermediateDiagnosticSnapshot>(completed.ToArray());

        public Step9DRawIntermediateDiagnosticCollector()
            : this(new ApprovedLivePreprocessingAdapter())
        {
        }

        public Step9DRawIntermediateDiagnosticCollector(ApprovedLivePreprocessingAdapter adapter)
        {
            this.adapter = adapter ?? throw new ArgumentNullException(nameof(adapter));
        }

        public void AttachReadOnlyBridge(Step9CHandIdentityObservationBridge sourceBridge)
        {
            if (sourceBridge == null) throw new ArgumentNullException(nameof(sourceBridge));
            if (ReferenceEquals(bridge, sourceBridge)) return;
            DetachReadOnlyBridge();
            bridge = sourceBridge;
            bridge.ObservationArrived += OnObservationArrived;
        }

        public void DetachReadOnlyBridge()
        {
            if (bridge != null) bridge.ObservationArrived -= OnObservationArrived;
            bridge = null;
        }

        public void Begin(string newRunId, string newCaptureId, string newExpectedWord)
        {
            Begin(newRunId, newCaptureId, newExpectedWord, null, false);
        }

        public void Begin(string newRunId, string newCaptureId, string newExpectedWord,
            Step9DInferenceImageGeometry geometry)
        {
            Begin(newRunId, newCaptureId, newExpectedWord, geometry, false);
        }

        public void Begin(string newRunId, string newCaptureId, string newExpectedWord,
            Step9DInferenceImageGeometry geometry, bool requireGeometry)
        {
            if (State == Step9DRawIntermediateCollectorState.Capturing)
                throw Failure(Step9DRawIntermediateDiagnosticFailureCode.INVALID_STATE, "a diagnostic capture is already active");
            RequireText(newRunId, nameof(newRunId));
            RequireText(newCaptureId, nameof(newCaptureId));
            RequireText(newExpectedWord, nameof(newExpectedWord));
            if (finalizedCaptureIds.Contains(newCaptureId))
                throw Failure(Step9DRawIntermediateDiagnosticFailureCode.DUPLICATE_CAPTURE_ID, "capture ID was already finalized");

            observations.Clear();
            identities.Clear();
            runId = newRunId;
            captureId = newCaptureId;
            expectedWord = newExpectedWord;
            inferenceImageGeometry = geometry?.Clone();
            geometryRequired = requireGeometry;
            frozenHandContract = null;
            frozenPoseContract = null;
            geometryCheckedCount = 0;
            firstHandTimestampMs = 0;
            lastHandTimestampMs = 0;
            firstPoseTimestampMs = 0;
            lastPoseTimestampMs = 0;
            maxAbsSkewMs = 0;
            State = Step9DRawIntermediateCollectorState.Capturing;
        }

        public void RecordObservation(
            RawMediaPipeObservation observation,
            Step9DRawIntermediateFrameIdentity identity)
        {
            if (State != Step9DRawIntermediateCollectorState.Capturing)
                throw Failure(Step9DRawIntermediateDiagnosticFailureCode.INVALID_STATE, "Begin must be called before recording");
            if (observation == null) throw new ArgumentNullException(nameof(observation));
            if (identity == null) throw new ArgumentNullException(nameof(identity));
            int expectedIndex = observations.Count;
            if (identity.CaptureLocalFrameIndex != expectedIndex || identity.ObservationTimestampSeconds != observation.TimestampSeconds)
                throw Failure(Step9DRawIntermediateDiagnosticFailureCode.FRAME_IDENTITY_MISMATCH, "frame identity does not match observation order/timestamp");
            if (expectedIndex > 0 && observation.TimestampSeconds <= observations[expectedIndex - 1].TimestampSeconds)
                throw Failure(Step9DRawIntermediateDiagnosticFailureCode.TIMESTAMP_REGRESSION, "observation timestamps must be strictly increasing");
            ValidateObservation(observation);
            EnforceGeometryConsistency(observation);
            observations.Add(Step9DRawIntermediateDiagnosticSnapshot.CloneObservation(observation));
            identities.Add(identity.Clone());
        }

        public Step9DRawIntermediateDiagnosticSnapshot FinalizeCapture()
        {
            if (State != Step9DRawIntermediateCollectorState.Capturing)
                throw Failure(Step9DRawIntermediateDiagnosticFailureCode.INVALID_STATE, "no diagnostic capture is active");
            if (observations.Count == 0)
                throw Failure(Step9DRawIntermediateDiagnosticFailureCode.EMPTY_CAPTURE, "cannot finalize an empty diagnostic capture");
            ValidateIdentitySequence();

            RawMediaPipeObservation[] frozenObservations = observations.ToArray();
            Step9DRawIntermediateFrameIdentity[] frozenIdentities = identities.ToArray();
            ApprovedPreprocessingTraceResult preprocessing;
            try
            {
                AdaptWithTraceInvocationCount++;
                preprocessing = adapter.AdaptWithTrace(frozenObservations);
            }
            catch (Step9DRawIntermediateDiagnosticException)
            {
                throw;
            }
            catch (Exception exception)
            {
                throw new Step9DRawIntermediateDiagnosticException(
                    Step9DRawIntermediateDiagnosticFailureCode.SCHEMA_VALIDATION_FAILED,
                    "shared preprocessing core rejected the diagnostic snapshot",
                    exception);
            }

            // Extract capture-level geometry from the first observation with complete, available, compatible geometry
            Step9DMergedInferenceGeometry captureGeometry = null;
            foreach (var obs in frozenObservations)
            {
                var mg = obs.MergedGeometry;
                if (mg != null && mg.BothAvailable && mg.FullContractCompatible)
                {
                    captureGeometry = mg;
                    break;
                }
            }

            // Geometry-required enforcement
            if (geometryRequired && captureGeometry == null)
                throw Failure(Step9DRawIntermediateDiagnosticFailureCode.INFERENCE_GEOMETRY_MISSING,
                    "geometry was required but no observation carried inference geometry");

            // Build consistency summary when geometry was checked
            Step9DCaptureGeometryConsistencySummary consistencySummary = null;
            if (geometryCheckedCount > 0)
            {
                consistencySummary = new Step9DCaptureGeometryConsistencySummary(
                    geometryCheckedCount,
                    firstHandTimestampMs,
                    lastHandTimestampMs,
                    firstPoseTimestampMs,
                    lastPoseTimestampMs,
                    maxAbsSkewMs);
            }

            var snapshot = new Step9DRawIntermediateDiagnosticSnapshot(
                runId, captureId, expectedWord, frozenObservations, frozenIdentities, preprocessing,
                inferenceImageGeometry, captureGeometry, consistencySummary);
            completed.Add(snapshot);
            finalizedCaptureIds.Add(captureId);
            observations.Clear();
            identities.Clear();
            State = Step9DRawIntermediateCollectorState.Finalized;
            return snapshot;
        }

        public void Cancel()
        {
            observations.Clear();
            identities.Clear();
            runId = null;
            captureId = null;
            expectedWord = null;
            inferenceImageGeometry = null;
            geometryRequired = false;
            frozenHandContract = null;
            frozenPoseContract = null;
            geometryCheckedCount = 0;
            if (State == Step9DRawIntermediateCollectorState.Capturing)
                State = completed.Count == 0 ? Step9DRawIntermediateCollectorState.Idle : Step9DRawIntermediateCollectorState.Finalized;
        }

        public void Dispose()
        {
            DetachReadOnlyBridge();
            Cancel();
        }

        void OnObservationArrived(RawMediaPipeObservation observation)
        {
            if (State != Step9DRawIntermediateCollectorState.Capturing) return;
            int localIndex = observations.Count;
            long providerFrameIndex = bridge.ProviderFrameCount - 1L;
            long bridgeObservationIndex = bridge.ObservationCount - 1L;
            long lifetimeProviderFrameIndex = bridge.LastLifetimeProviderFrameIndex;
            RecordObservation(
                observation,
                new Step9DRawIntermediateFrameIdentity(
                    localIndex,
                    observation.TimestampSeconds,
                    providerFrameIndex,
                    bridgeObservationIndex,
                    lifetimeProviderFrameIndex));
        }

        void EnforceGeometryConsistency(RawMediaPipeObservation observation)
        {
            var merged = observation.MergedGeometry;

            if (geometryRequired)
            {
                if (merged == null)
                    throw Failure(Step9DRawIntermediateDiagnosticFailureCode.INFERENCE_GEOMETRY_MISSING,
                        "geometry required but observation has no merged geometry");
                if (merged.HandResult == null || merged.HandResult.Status != Step9DGeometryAcquisitionStatus.Available)
                    throw Failure(Step9DRawIntermediateDiagnosticFailureCode.INFERENCE_GEOMETRY_MISSING,
                        "geometry required but hand acquisition not available");
                if (merged.PoseResult == null || merged.PoseResult.Status != Step9DGeometryAcquisitionStatus.Available)
                    throw Failure(Step9DRawIntermediateDiagnosticFailureCode.INFERENCE_GEOMETRY_MISSING,
                        "geometry required but pose acquisition not available");
                if (!merged.FullContractCompatible)
                    throw Failure(Step9DRawIntermediateDiagnosticFailureCode.INFERENCE_GEOMETRY_MISSING,
                        "geometry required but hand/pose contract incompatible");
            }

            if (merged == null) return;

            var hand = merged.Hand;
            var pose = merged.Pose;
            bool hasHand = hand != null;
            bool hasPose = pose != null;

            if (!hasHand && !hasPose) return;

            geometryCheckedCount++;

            // Track timestamps
            if (hasHand)
            {
                long handTs = hand.MediaPipeTimestampMs;
                if (firstHandTimestampMs == 0 && lastHandTimestampMs == 0)
                    firstHandTimestampMs = handTs;
                lastHandTimestampMs = handTs;
            }
            if (hasPose)
            {
                long poseTs = pose.MediaPipeTimestampMs;
                if (firstPoseTimestampMs == 0 && lastPoseTimestampMs == 0)
                    firstPoseTimestampMs = poseTs;
                lastPoseTimestampMs = poseTs;
            }

            // Track skew
            if (merged.HandPoseTimestampSkewMs.HasValue)
            {
                long skew = merged.HandPoseTimestampSkewMs.Value;
                if (skew > maxAbsSkewMs) maxAbsSkewMs = skew;
            }

            // Freeze first contracts
            if (hasHand && frozenHandContract == null)
                frozenHandContract = hand.Clone();
            if (hasPose && frozenPoseContract == null)
                frozenPoseContract = pose.Clone();

            // Verify against frozen contracts
            if (hasHand && frozenHandContract != null && !frozenHandContract.IsFullContractCompatible(hand))
            {
                string mismatch = frozenHandContract.DescribeFirstMismatch(hand);
                throw Failure(Step9DRawIntermediateDiagnosticFailureCode.INFERENCE_GEOMETRY_CONTRACT_CHANGED,
                    "hand geometry contract changed mid-capture: " + (mismatch ?? "unknown field"));
            }
            if (hasPose && frozenPoseContract != null && !frozenPoseContract.IsFullContractCompatible(pose))
            {
                string mismatch = frozenPoseContract.DescribeFirstMismatch(pose);
                throw Failure(Step9DRawIntermediateDiagnosticFailureCode.INFERENCE_GEOMETRY_CONTRACT_CHANGED,
                    "pose geometry contract changed mid-capture: " + (mismatch ?? "unknown field"));
            }
        }

        void ValidateIdentitySequence()
        {
            if (identities.Count != observations.Count)
                throw Failure(Step9DRawIntermediateDiagnosticFailureCode.FRAME_IDENTITY_MISMATCH, "identity count does not match observation count");
            for (int index = 0; index < identities.Count; index++)
            {
                Step9DRawIntermediateFrameIdentity identity = identities[index];
                if (identity.CaptureLocalFrameIndex != index || identity.ObservationTimestampSeconds != observations[index].TimestampSeconds)
                    throw Failure(Step9DRawIntermediateDiagnosticFailureCode.FRAME_IDENTITY_MISMATCH, "identity order does not match observations");
                if (index == 0 || !identity.HasProviderIdentity || !identities[index - 1].HasProviderIdentity) continue;
                if (identity.ProviderFrameIndex.Value <= identities[index - 1].ProviderFrameIndex.Value ||
                    identity.BridgeObservationIndex.Value <= identities[index - 1].BridgeObservationIndex.Value ||
                    identity.LifetimeProviderFrameIndex.Value <= identities[index - 1].LifetimeProviderFrameIndex.Value)
                    throw Failure(Step9DRawIntermediateDiagnosticFailureCode.FRAME_IDENTITY_MISMATCH, "provider identity values must be strictly increasing");
            }
        }

        static void ValidateObservation(RawMediaPipeObservation observation)
        {
            ValidateSet(observation.Pose, RawMediaPipeObservation.PoseLandmarkCount, "pose");
            ValidateSet(observation.LeftHand, RawMediaPipeObservation.HandLandmarkCount, "left_hand");
            ValidateSet(observation.RightHand, RawMediaPipeObservation.HandLandmarkCount, "right_hand");
        }

        static void ValidateSet(RawLandmarkObservationSet value, int expectedCount, string field)
        {
            if (value.LandmarkCount != expectedCount)
                throw Failure(Step9DRawIntermediateDiagnosticFailureCode.INVALID_SHAPE, $"{field} landmark count is invalid");
            for (int landmark = 0; landmark < value.LandmarkCount; landmark++)
            for (int axis = 0; axis < 3; axis++)
                Step9DRawIntermediateDiagnosticContract.RejectInfinity($"{field}[{landmark},{axis}]", value.Coordinate(landmark, axis));
        }

        static void RequireText(string value, string parameter)
        {
            if (string.IsNullOrWhiteSpace(value)) throw new ArgumentException("value is required", parameter);
        }

        static Step9DRawIntermediateDiagnosticException Failure(
            Step9DRawIntermediateDiagnosticFailureCode code,
            string message) => new Step9DRawIntermediateDiagnosticException(code, message);
    }
}
