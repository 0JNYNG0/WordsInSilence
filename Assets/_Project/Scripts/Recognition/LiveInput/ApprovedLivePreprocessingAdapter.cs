using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Recognition.LiveInput
{
    public sealed class ApprovedPreprocessingResult
    {
        public int RawFrameCount { get; }
        public int NormalizedFrameCount { get; }
        public int ValidLandmarkCount { get; }
        public int InterpolatedLandmarkCount { get; }
        public IReadOnlyList<NormalizedRecognitionFrame> Frames { get; }

        internal ApprovedPreprocessingResult(
            int rawFrameCount,
            int validLandmarkCount,
            int interpolatedLandmarkCount,
            NormalizedRecognitionFrame[] frames)
        {
            RawFrameCount = rawFrameCount;
            NormalizedFrameCount = frames.Length;
            ValidLandmarkCount = validLandmarkCount;
            InterpolatedLandmarkCount = interpolatedLandmarkCount;
            Frames = new ReadOnlyCollection<NormalizedRecognitionFrame>((NormalizedRecognitionFrame[])frames.Clone());
        }
    }

    /// <summary>Narrow C# port of the approved Step 7B raw_framewise/max_gap=1/no_smoothing path.</summary>
    /// <summary>
    /// Immutable active common-2D capture: flattened [T,16,3] coordinates whose channel 2
    /// is structurally 0, the matching [T,16] validity mask, and the actual image geometry
    /// the pixel plane was restored with. This is the ONLY input the frozen common model
    /// may consume; the pseudo-3D <see cref="RecognitionSequence"/> must never be reused
    /// for it.
    /// </summary>
    public sealed class CommonTwoDimensionalCapture
    {
        readonly double[] coordinates;
        readonly bool[] validMask;

        public int FrameCount { get; }
        public int LandmarkCount { get; }
        public int ValidLandmarkCount { get; }
        public int InterpolatedLandmarkCount { get; }
        public int ImageWidth { get; }
        public int ImageHeight { get; }
        public string ConventionId { get; }
        public string DomainContract { get { return "raw_post_gap_pixel_plane_z0_torso_normalize_2d"; } }

        public CommonTwoDimensionalCapture(
            int frameCount, int landmarkCount, double[] coordinates, bool[] validMask,
            int validLandmarkCount, int interpolatedLandmarkCount,
            int imageWidth, int imageHeight, string conventionId)
        {
            FrameCount = frameCount;
            LandmarkCount = landmarkCount;
            ValidLandmarkCount = validLandmarkCount;
            InterpolatedLandmarkCount = interpolatedLandmarkCount;
            ImageWidth = imageWidth;
            ImageHeight = imageHeight;
            ConventionId = conventionId ?? string.Empty;
            this.coordinates = (double[])coordinates.Clone();
            this.validMask = (bool[])validMask.Clone();
        }

        public double[] CopyCoordinates() { return (double[])coordinates.Clone(); }
        public bool[] CopyValidMask() { return (bool[])validMask.Clone(); }
    }

    public sealed class ApprovedLivePreprocessingAdapter
    {
        public const int SelectedMaxGap = 1;
        public const double TimestampDiscontinuitySeconds = 0.1;
        const double MinimumShoulderWidth = 1e-6;

        static readonly int[] PoseOutputLandmarks = { 13, 14, 15, 16 };
        static readonly int[] HandOutputLandmarks = { 0, 4, 8, 12, 16, 20 };
        // F2 candidate only. The six-joint selection above stays the frozen 16 contract.
        static readonly int[] FullHandOutputLandmarks =
            { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };

        sealed class ComponentBuffer
        {
            public readonly int Count;
            public readonly double[,,] Coordinates;
            public readonly bool[,] OriginalValid;
            public readonly bool[,] Interpolated;
            public readonly bool[,] Boundary;
            public readonly bool[,] Strong;
            public readonly bool[,] Reappearance;
            public readonly bool[,] TimestampBoundary;

            public ComponentBuffer(int frames, int count, bool traceEnabled)
            {
                Count = count;
                Coordinates = new double[frames, count, 3];
                OriginalValid = new bool[frames, count];
                Interpolated = new bool[frames, count];
                Boundary = new bool[frames, count];
                Strong = new bool[frames, count];
                Reappearance = traceEnabled ? new bool[frames, count] : null;
                TimestampBoundary = traceEnabled ? new bool[frames, count] : null;
                for (int frame = 0; frame < frames; frame++)
                    for (int landmark = 0; landmark < count; landmark++)
                        for (int axis = 0; axis < 3; axis++) Coordinates[frame, landmark, axis] = double.NaN;
            }
        }

        sealed class TorsoTransform
        {
            public readonly double[] Origin = new double[3];
            public readonly double[] Right = new double[3];
            public readonly double[] Up = new double[3];
            public readonly double[] Forward = new double[3];
            public double Scale;
            public bool Valid;

            public TorsoTransform Clone()
            {
                var clone = new TorsoTransform { Scale = Scale, Valid = Valid };
                Array.Copy(Origin, clone.Origin, 3);
                Array.Copy(Right, clone.Right, 3);
                Array.Copy(Up, clone.Up, 3);
                Array.Copy(Forward, clone.Forward, 3);
                return clone;
            }
        }

        public ApprovedPreprocessingResult Adapt(IReadOnlyList<RawMediaPipeObservation> observations)
        {
            Step9DRawIntermediatePreprocessingTrace ignored;
            return AdaptCore(observations, false, out ignored);
        }

        public ApprovedPreprocessingTraceResult AdaptWithTrace(IReadOnlyList<RawMediaPipeObservation> observations)
        {
            Step9DRawIntermediatePreprocessingTrace trace;
            ApprovedPreprocessingResult result = AdaptCore(observations, true, out trace);
            return new ApprovedPreprocessingTraceResult(result, trace);
        }

        ApprovedPreprocessingResult AdaptCore(
            IReadOnlyList<RawMediaPipeObservation> observations,
            bool traceEnabled,
            out Step9DRawIntermediatePreprocessingTrace trace)
        {
            if (observations == null) throw new ArgumentNullException(nameof(observations));
            if (observations.Count == 0) throw new InvalidOperationException("cannot preprocess a capture with zero observations");

            RawObservationConvention convention = observations[0]?.Convention ?? throw new ArgumentException("observation is null", nameof(observations));
            for (int frame = 0; frame < observations.Count; frame++)
            {
                RawMediaPipeObservation current = observations[frame] ?? throw new ArgumentException($"observation {frame} is null", nameof(observations));
                if (!convention.Equals(current.Convention)) throw new ArgumentException("observation convention changed within the capture", nameof(observations));
                if (frame > 0 && current.TimestampSeconds <= observations[frame - 1].TimestampSeconds)
                    throw new ArgumentException("observation timestamps must be strictly increasing", nameof(observations));
            }

            int frameCount = observations.Count;
            var pose = BuildComponent(observations, frameCount, RawMediaPipeObservation.PoseLandmarkCount, value => value.Pose, traceEnabled);
            var left = BuildComponent(observations, frameCount, RawMediaPipeObservation.HandLandmarkCount, value => value.LeftHand, traceEnabled);
            var right = BuildComponent(observations, frameCount, RawMediaPipeObservation.HandLandmarkCount, value => value.RightHand, traceEnabled);
            BuildBoundaries(observations, pose);
            BuildBoundaries(observations, left);
            BuildBoundaries(observations, right);

            Interpolate(pose);
            Interpolate(left);
            Interpolate(right);
            Step9DTorsoTransformProvenance[] torsoProvenance;
            TorsoTransform[] transforms = BuildTorsoTransforms(pose, traceEnabled, out torsoProvenance);

            int validCount = 0;
            int interpolatedCount = 0;
            var frames = new NormalizedRecognitionFrame[frameCount];
            for (int frame = 0; frame < frameCount; frame++)
            {
                var coordinates = new double[NormalizedRecognitionFrame.CoordinateCount];
                for (int index = 0; index < coordinates.Length; index++) coordinates[index] = double.NaN;
                var mask = new bool[NormalizedRecognitionFrame.LandmarkCount];
                int output = 0;
                output = WriteComponent(frame, pose, PoseOutputLandmarks, transforms[frame], coordinates, mask, output, ref validCount, ref interpolatedCount);
                output = WriteComponent(frame, left, HandOutputLandmarks, transforms[frame], coordinates, mask, output, ref validCount, ref interpolatedCount);
                WriteComponent(frame, right, HandOutputLandmarks, transforms[frame], coordinates, mask, output, ref validCount, ref interpolatedCount);
                frames[frame] = new NormalizedRecognitionFrame(observations[frame].TimestampSeconds, coordinates, mask);
            }
            var result = new ApprovedPreprocessingResult(frameCount, validCount, interpolatedCount, frames);
            trace = traceEnabled
                ? CreateTrace(observations, pose, left, right, transforms, torsoProvenance, frames, validCount, interpolatedCount)
                : null;
            return result;
        }

        /// <summary>
        /// Active common-2D query for the frozen Step 10O v1.6 common model.
        ///
        /// This runs the SAME approved stages as <see cref="Adapt"/> - BuildComponent,
        /// BuildBoundaries, Interpolate - so the raw/post-gap decisions (including the
        /// strong-depth boundary rule and SelectedMaxGap) are byte-identical to the
        /// approved live path. Only the transform INPUT differs: the post-gap
        /// coordinates are restored to the actual pixel plane using the image geometry
        /// carried by the observations, and z is set to 0 BEFORE the torso transform is
        /// built. That reproduces the approved torso_normalize_2d contract exactly -
        /// isotropic shoulder-width scale, the same anchor provenance and the same
        /// mask-aware fallback chain - and forces channel 2 to be exactly 0.
        ///
        /// The pseudo-3D <see cref="Adapt"/> result used by the strict and personal
        /// routes is not affected in any way.
        ///
        /// Fail-closed: returns null and sets <paramref name="failure"/> when the image
        /// geometry is missing, non-positive or inconsistent across the capture segment.
        /// No estimated or constant image size is ever substituted.
        /// </summary>
        public static CommonTwoDimensionalCapture AdaptCommon2D(
            IReadOnlyList<RawMediaPipeObservation> observations,
            out string failure)
        {
            failure = null;
            if (observations == null || observations.Count == 0)
            {
                failure = "common-2D adaptation requires at least one observation";
                return null;
            }

#if UNITY_EDITOR
            // Editor-only observation of the EXACT sequence this adaptation consumes,
            // after the caller's authorized trailing trim and before any normalization.
            // When the developer trace is not armed this call returns immediately and
            // allocates nothing.
            WordsInSilence.Recognition.Diagnostics.Common2DDiagnosticTraceBuffer
                .SafeRecordRawObservations(observations);
#endif

            int width, height;
            if (!TryResolveSegmentImageGeometry(observations, out width, out height, out failure))
                return null;

            int frameCount = observations.Count;
            RawObservationConvention convention = observations[0] == null ? null : observations[0].Convention;
            if (convention == null)
            {
                failure = "observation convention is missing";
                return null;
            }
            for (int frame = 0; frame < frameCount; frame++)
            {
                RawMediaPipeObservation current = observations[frame];
                if (current == null)
                {
                    failure = "observation " + frame + " is null";
                    return null;
                }
                if (!convention.Equals(current.Convention))
                {
                    failure = "observation convention changed within the capture";
                    return null;
                }
            }

            ComponentBuffer pose = BuildComponent(
                observations, frameCount, RawMediaPipeObservation.PoseLandmarkCount,
                value => value.Pose, false);
            ComponentBuffer left = BuildComponent(
                observations, frameCount, RawMediaPipeObservation.HandLandmarkCount,
                value => value.LeftHand, false);
            ComponentBuffer right = BuildComponent(
                observations, frameCount, RawMediaPipeObservation.HandLandmarkCount,
                value => value.RightHand, false);
            BuildBoundaries(observations, pose);
            BuildBoundaries(observations, left);
            BuildBoundaries(observations, right);
            Interpolate(pose);
            Interpolate(left);
            Interpolate(right);

            // Post-gap -> actual pixel plane, z := 0. Applied AFTER interpolation so the
            // approved gap and boundary decisions are untouched.
            bool normalizedImage =
                convention.CoordinateSpace == RawObservationCoordinateSpace.MediaPipeNormalizedImage;
            ToPixelPlaneWithZeroDepth(pose, width, height, normalizedImage);
            ToPixelPlaneWithZeroDepth(left, width, height, normalizedImage);
            ToPixelPlaneWithZeroDepth(right, width, height, normalizedImage);

            Step9DTorsoTransformProvenance[] ignoredProvenance;
            TorsoTransform[] transforms = BuildTorsoTransforms(pose, false, out ignoredProvenance);

            int validCount = 0;
            int interpolatedCount = 0;
            int landmarks = NormalizedRecognitionFrame.LandmarkCount;
            var coordinates = new double[frameCount * landmarks * 3];
            var mask = new bool[frameCount * landmarks];
            for (int frame = 0; frame < frameCount; frame++)
            {
                var frameCoordinates = new double[NormalizedRecognitionFrame.CoordinateCount];
                for (int index = 0; index < frameCoordinates.Length; index++)
                    frameCoordinates[index] = double.NaN;
                var frameMask = new bool[landmarks];
                int output = 0;
                output = WriteComponent(frame, pose, PoseOutputLandmarks, transforms[frame],
                    frameCoordinates, frameMask, output, ref validCount, ref interpolatedCount);
                output = WriteComponent(frame, left, HandOutputLandmarks, transforms[frame],
                    frameCoordinates, frameMask, output, ref validCount, ref interpolatedCount);
                WriteComponent(frame, right, HandOutputLandmarks, transforms[frame],
                    frameCoordinates, frameMask, output, ref validCount, ref interpolatedCount);
                for (int landmark = 0; landmark < landmarks; landmark++)
                {
                    int source = landmark * 3;
                    int target = (frame * landmarks + landmark) * 3;
                    bool valid = frameMask[landmark];
                    mask[frame * landmarks + landmark] = valid;
                    coordinates[target] = valid ? frameCoordinates[source] : 0.0;
                    coordinates[target + 1] = valid ? frameCoordinates[source + 1] : 0.0;
                    // channel 2 is structurally 0 for the common-2D contract
                    coordinates[target + 2] = 0.0;
                }
            }

            if (validCount == 0)
            {
                failure = "common-2D capture contains no valid normalized landmark";
                return null;
            }
            var capture = new CommonTwoDimensionalCapture(
                frameCount, landmarks, coordinates, mask, validCount, interpolatedCount,
                width, height, convention.Id);
#if UNITY_EDITOR
            // The capture object itself is passed, never pre-copied arrays: the observer
            // takes its own defensive copies INSIDE the guarded call, so a disabled or
            // unowned buffer causes no diagnostic allocation on this ordinary path.
            WordsInSilence.Recognition.Diagnostics.Common2DDiagnosticTraceBuffer
                .SafeRecordNormalizedCapture(capture);
#endif
            return capture;
        }

        /// <summary>
        /// F2 full-hand common-2D CANDIDATE. Same approved stages as AdaptCommon2D -
        /// BuildComponent, BuildBoundaries, Interpolate, pixel restoration with z := 0,
        /// the same torso transform - but it emits all 21 joints per hand: 46 points as
        /// pose 13/14/15/16, left hand 0..20, right hand 0..20 (schema
        /// wis_full_hand_common_v1).
        ///
        /// Additional API for later feature work. It feeds no classifier, is on no default
        /// call site, records no trace, and is not completed webcam inference or any
        /// performance claim.
        ///
        /// Two deliberate differences from AdaptCommon2D:
        ///   * a non MediaPipeNormalizedImage convention is REFUSED rather than treated as
        ///     an already-pixel plane
        ///   * a masked point keeps NaN in all three channels instead of being zeroed; a
        ///     valid point's channel 2 is then structurally 0
        /// OriginalValid and Interpolated are reported per point and are NOT the final
        /// mask: a point may be originally valid while its frame has no torso transform,
        /// and the normalized valid mask is then false for it.
        ///
        /// Fail-closed: returns null and sets <paramref name="failure"/> for an empty or
        /// null capture, a null observation, a changed or unsupported convention,
        /// non-increasing or non-finite timestamps, unusable image geometry, or a capture
        /// with no valid normalized point. No estimated size or zeroed observation is ever
        /// substituted.
        /// </summary>
        public static FullHandCommonCapture AdaptFullHandCommon2D(
            IReadOnlyList<RawMediaPipeObservation> observations, out string failure)
        {
            failure = null;
            if (observations == null || observations.Count == 0)
            {
                failure = "full-hand common-2D adaptation requires at least one observation";
                return null;
            }

            int frameCount = observations.Count;
            RawObservationConvention convention =
                observations[0] == null ? null : observations[0].Convention;
            if (convention == null)
            {
                failure = "observation convention is missing";
                return null;
            }
            if (convention.CoordinateSpace !=
                RawObservationCoordinateSpace.MediaPipeNormalizedImage)
            {
                failure = "the full-hand candidate requires the MediaPipeNormalizedImage " +
                    "convention; a metric-3D capture must not be treated as a pixel plane";
                return null;
            }
            for (int frame = 0; frame < frameCount; frame++)
            {
                RawMediaPipeObservation current = observations[frame];
                if (current == null)
                {
                    failure = "observation " + frame + " is null";
                    return null;
                }
                if (!convention.Equals(current.Convention))
                {
                    failure = "observation convention changed within the capture";
                    return null;
                }
                double timestamp = current.TimestampSeconds;
                if (double.IsNaN(timestamp) || double.IsInfinity(timestamp))
                {
                    failure = "observation " + frame + " carries a non-finite timestamp";
                    return null;
                }
                if (frame > 0 && timestamp <= observations[frame - 1].TimestampSeconds)
                {
                    failure = "observation timestamps must be strictly increasing";
                    return null;
                }
            }

            int width, height;
            if (!TryResolveSegmentImageGeometry(observations, out width, out height,
                                                out failure))
                return null;

            ComponentBuffer pose = BuildComponent(
                observations, frameCount, RawMediaPipeObservation.PoseLandmarkCount,
                value => value.Pose, false);
            ComponentBuffer left = BuildComponent(
                observations, frameCount, RawMediaPipeObservation.HandLandmarkCount,
                value => value.LeftHand, false);
            ComponentBuffer right = BuildComponent(
                observations, frameCount, RawMediaPipeObservation.HandLandmarkCount,
                value => value.RightHand, false);
            BuildBoundaries(observations, pose);
            BuildBoundaries(observations, left);
            BuildBoundaries(observations, right);
            Interpolate(pose);
            Interpolate(left);
            Interpolate(right);

            // the convention was already required to be normalized-image above
            ToPixelPlaneWithZeroDepth(pose, width, height, true);
            ToPixelPlaneWithZeroDepth(left, width, height, true);
            ToPixelPlaneWithZeroDepth(right, width, height, true);

            Step9DTorsoTransformProvenance[] ignoredProvenance;
            TorsoTransform[] transforms = BuildTorsoTransforms(pose, false,
                                                               out ignoredProvenance);

            int points = FullHandCommonSchema.PointCount;
            int dims = FullHandCommonSchema.CoordinateDimensions;
            int validCount = 0;
            int interpolatedCount = 0;
            var coordinates = new double[frameCount * points * dims];
            var mask = new bool[frameCount * points];
            var originalValid = new bool[frameCount * points];
            var interpolated = new bool[frameCount * points];
            for (int frame = 0; frame < frameCount; frame++)
            {
                var frameCoordinates = new double[points * dims];
                for (int index = 0; index < frameCoordinates.Length; index++)
                    frameCoordinates[index] = double.NaN;
                var frameMask = new bool[points];
                var frameOriginal = new bool[points];
                var frameInterpolated = new bool[points];

                int output = 0;
                output = WriteComponent(frame, pose, PoseOutputLandmarks, transforms[frame],
                    frameCoordinates, frameMask, output, ref validCount, ref interpolatedCount);
                output = WriteComponent(frame, left, FullHandOutputLandmarks, transforms[frame],
                    frameCoordinates, frameMask, output, ref validCount, ref interpolatedCount);
                WriteComponent(frame, right, FullHandOutputLandmarks, transforms[frame],
                    frameCoordinates, frameMask, output, ref validCount, ref interpolatedCount);

                int provenance = 0;
                provenance = WriteFullHandProvenance(frame, pose, PoseOutputLandmarks,
                    frameOriginal, frameInterpolated, provenance);
                provenance = WriteFullHandProvenance(frame, left, FullHandOutputLandmarks,
                    frameOriginal, frameInterpolated, provenance);
                WriteFullHandProvenance(frame, right, FullHandOutputLandmarks,
                    frameOriginal, frameInterpolated, provenance);

                Array.Copy(frameCoordinates, 0, coordinates, frame * points * dims,
                           frameCoordinates.Length);
                Array.Copy(frameMask, 0, mask, frame * points, points);
                Array.Copy(frameOriginal, 0, originalValid, frame * points, points);
                Array.Copy(frameInterpolated, 0, interpolated, frame * points, points);
            }

            if (validCount == 0)
            {
                failure = "full-hand common-2D capture contains no valid normalized point";
                return null;
            }
            return new FullHandCommonCapture(
                frameCount, coordinates, mask, originalValid, interpolated,
                validCount, interpolatedCount, width, height, convention.Id);
        }

        /// <summary>
        /// Resolves the one pixel plane that may be applied to pose, left-hand AND
        /// right-hand coordinates.
        ///
        /// A single provider's geometry is never representative: the hand and the pose
        /// graph can run on different textures or different pre-inference transforms.
        /// Every observation must therefore carry BOTH provider geometries, they must
        /// agree on the whole pixel-semantic contract for that frame, and the contract
        /// must stay identical for the whole capture segment. Anything else fails closed
        /// with a concrete detail; no default size, no representative fallback and no
        /// transform correction is ever substituted.
        /// </summary>
        static bool TryResolveSegmentImageGeometry(
            IReadOnlyList<RawMediaPipeObservation> observations,
            out int width, out int height, out string failure)
        {
            width = 0;
            height = 0;
            failure = null;
            Step9DInferenceFrameGeometry firstHand = null;
            Step9DInferenceFrameGeometry firstPose = null;

            for (int frame = 0; frame < observations.Count; frame++)
            {
                RawMediaPipeObservation observation = observations[frame];
                Step9DMergedInferenceGeometry merged =
                    observation == null ? null : observation.MergedGeometry;
                if (merged == null)
                {
                    failure = "observation " + frame +
                        " carries no inference image geometry; the common-2D pixel plane " +
                        "cannot be restored and no default size may be substituted";
                    return false;
                }

                // Both providers are required. A one-sided geometry may never stand in
                // for the other provider's pixel plane.
                if (!merged.BothAvailable)
                {
                    failure = "observation " + frame + " does not carry both provider " +
                        "geometries (hand=" + DescribeAcquisition(merged.HandResult) +
                        ", pose=" + DescribeAcquisition(merged.PoseResult) +
                        "); a single provider geometry is not representative";
                    return false;
                }
                Step9DInferenceFrameGeometry hand = merged.Hand;
                Step9DInferenceFrameGeometry pose = merged.Pose;
                if (hand == null || pose == null)
                {
                    failure = "observation " + frame +
                        " reports both providers available but a geometry is null";
                    return false;
                }

                // Same-frame pixel-semantic contract: sizes, format, flips, rotation,
                // crop, resize, letterbox and row order must all agree.
                if (!merged.FullContractCompatible)
                {
                    failure = "observation " + frame + " hand/pose pixel-semantic " +
                        "contract mismatch: " +
                        (merged.FirstMismatchDescription ?? "incompatible");
                    return false;
                }
                string nullableMismatch = DescribeNullableContractMismatch(hand, pose);
                if (nullableMismatch != null)
                {
                    failure = "observation " + frame + " hand/pose pixel-semantic " +
                        "contract mismatch: " + nullableMismatch;
                    return false;
                }

                if (pose.InferenceImageWidth <= 0 || pose.InferenceImageHeight <= 0)
                {
                    failure = "observation " + frame + " reports a non-positive image size " +
                        pose.InferenceImageWidth + "x" + pose.InferenceImageHeight;
                    return false;
                }

                if (frame == 0)
                {
                    firstHand = hand;
                    firstPose = pose;
                    width = pose.InferenceImageWidth;
                    height = pose.InferenceImageHeight;
                    continue;
                }

                // Capture-wide consistency, per provider, over the same full contract.
                string poseDrift = DescribeCaptureDrift(firstPose, pose);
                if (poseDrift != null)
                {
                    failure = "inference image geometry changed within the capture " +
                        "segment at observation " + frame + " (pose): " + poseDrift;
                    return false;
                }
                string handDrift = DescribeCaptureDrift(firstHand, hand);
                if (handDrift != null)
                {
                    failure = "inference image geometry changed within the capture " +
                        "segment at observation " + frame + " (hand): " + handDrift;
                    return false;
                }
            }
            return true;
        }

        static string DescribeAcquisition(Step9DGeometryAcquisitionResult result)
        {
            if (result == null) return "absent";
            return result.Status.ToString();
        }

        /// <summary>
        /// The shared full-contract comparison deliberately treats a nullable transform
        /// state as compatible when only one side knows it. For the common-2D pixel
        /// plane that is not good enough: a known/unknown difference is itself a
        /// mismatch, so it fails closed here.
        /// </summary>
        static string DescribeNullableContractMismatch(
            Step9DInferenceFrameGeometry a, Step9DInferenceFrameGeometry b)
        {
            if (a.ObservedVideoRotationAngleOrNull.HasValue !=
                b.ObservedVideoRotationAngleOrNull.HasValue)
                return "observed_video_rotation_angle known/unknown mismatch: " +
                    Describe(a.ObservedVideoRotationAngleOrNull) + " vs " +
                    Describe(b.ObservedVideoRotationAngleOrNull);
            if (a.ObservedVideoVerticallyMirroredOrNull.HasValue !=
                b.ObservedVideoVerticallyMirroredOrNull.HasValue)
                return "observed_video_vertically_mirrored known/unknown mismatch: " +
                    Describe(a.ObservedVideoVerticallyMirroredOrNull) + " vs " +
                    Describe(b.ObservedVideoVerticallyMirroredOrNull);
            if (a.PreInferenceRotationDegreesOrNull.HasValue !=
                b.PreInferenceRotationDegreesOrNull.HasValue)
                return "pre_inference_rotation_degrees known/unknown mismatch: " +
                    Describe(a.PreInferenceRotationDegreesOrNull) + " vs " +
                    Describe(b.PreInferenceRotationDegreesOrNull);
            return null;
        }

        static string DescribeCaptureDrift(
            Step9DInferenceFrameGeometry first, Step9DInferenceFrameGeometry current)
        {
            if (first == null || current == null) return "geometry missing";
            if (!first.IsFullContractCompatible(current))
                return first.DescribeFirstMismatch(current) ?? "incompatible";
            return DescribeNullableContractMismatch(first, current);
        }

        static string Describe(int? value)
        {
            return value.HasValue ? value.Value.ToString(CultureInfo.InvariantCulture) : "unknown";
        }

        static string Describe(bool? value)
        {
            return value.HasValue ? (value.Value ? "true" : "false") : "unknown";
        }

        static void ToPixelPlaneWithZeroDepth(
            ComponentBuffer component, int width, int height, bool normalizedImage)
        {
            for (int frame = 0; frame < component.Coordinates.GetLength(0); frame++)
            {
                for (int landmark = 0; landmark < component.Count; landmark++)
                {
                    if (normalizedImage)
                    {
                        component.Coordinates[frame, landmark, 0] *= width;
                        component.Coordinates[frame, landmark, 1] *= height;
                    }
                    component.Coordinates[frame, landmark, 2] = 0.0;
                }
            }
        }

        static ComponentBuffer BuildComponent(
            IReadOnlyList<RawMediaPipeObservation> observations,
            int frameCount,
            int landmarkCount,
            Func<RawMediaPipeObservation, RawLandmarkObservationSet> selector,
            bool traceEnabled)
        {
            var buffer = new ComponentBuffer(frameCount, landmarkCount, traceEnabled);
            for (int frame = 0; frame < frameCount; frame++)
            {
                RawLandmarkObservationSet source = selector(observations[frame]);
                for (int landmark = 0; landmark < landmarkCount; landmark++)
                {
                    buffer.OriginalValid[frame, landmark] = source.IsValid(landmark);
                    buffer.Strong[frame, landmark] = source.IsStrongDepthBoundary(landmark);
                    buffer.Boundary[frame, landmark] = source.IsReappearance(landmark) || source.IsStrongDepthBoundary(landmark);
                    if (buffer.Reappearance != null) buffer.Reappearance[frame, landmark] = source.IsReappearance(landmark);
                    for (int axis = 0; axis < 3; axis++) buffer.Coordinates[frame, landmark, axis] = source.Coordinate(landmark, axis);
                }
            }
            return buffer;
        }

        static void BuildBoundaries(IReadOnlyList<RawMediaPipeObservation> observations, ComponentBuffer component)
        {
            for (int landmark = 0; landmark < component.Count; landmark++) component.Boundary[0, landmark] = true;
            if (component.TimestampBoundary != null)
                for (int landmark = 0; landmark < component.Count; landmark++) component.TimestampBoundary[0, landmark] = true;
            for (int frame = 1; frame < observations.Count; frame++)
            {
                double delta = observations[frame].TimestampSeconds - observations[frame - 1].TimestampSeconds;
                if (delta <= 0.0 || delta > TimestampDiscontinuitySeconds)
                {
                    for (int landmark = 0; landmark < component.Count; landmark++) component.Boundary[frame, landmark] = true;
                    if (component.TimestampBoundary != null)
                        for (int landmark = 0; landmark < component.Count; landmark++) component.TimestampBoundary[frame, landmark] = true;
                }
            }
        }

        static void Interpolate(ComponentBuffer component)
        {
            int frames = component.Coordinates.GetLength(0);
            for (int landmark = 0; landmark < component.Count; landmark++)
            {
                int segmentStart = 0;
                while (segmentStart < frames)
                {
                    int segmentEnd = frames - 1;
                    for (int frame = segmentStart + 1; frame < frames; frame++)
                    {
                        if (component.Boundary[frame, landmark]) { segmentEnd = frame - 1; break; }
                    }

                    int cursor = segmentStart;
                    while (cursor <= segmentEnd)
                    {
                        if (component.OriginalValid[cursor, landmark]) { cursor++; continue; }
                        int gapStart = cursor;
                        while (cursor <= segmentEnd && !component.OriginalValid[cursor, landmark]) cursor++;
                        int gapEnd = cursor - 1;
                        int gapLength = gapEnd - gapStart + 1;
                        bool before = gapStart > segmentStart && component.OriginalValid[gapStart - 1, landmark] && IsFinitePoint(component.Coordinates, gapStart - 1, landmark);
                        bool after = cursor <= segmentEnd && component.OriginalValid[cursor, landmark] && IsFinitePoint(component.Coordinates, cursor, landmark);
                        bool strong = false;
                        for (int frame = gapStart; frame <= gapEnd; frame++) strong |= component.Strong[frame, landmark];
                        if (gapLength <= SelectedMaxGap && before && after && !strong)
                        {
                            for (int gap = 0; gap < gapLength; gap++)
                            {
                                double t = (gap + 1.0) / (gapLength + 1.0);
                                for (int axis = 0; axis < 3; axis++)
                                {
                                    double a = component.Coordinates[gapStart - 1, landmark, axis];
                                    double b = component.Coordinates[cursor, landmark, axis];
                                    component.Coordinates[gapStart + gap, landmark, axis] = a * (1.0 - t) + b * t;
                                }
                                component.Interpolated[gapStart + gap, landmark] = true;
                            }
                        }
                    }
                    segmentStart = segmentEnd + 1;
                }
            }
        }

        static TorsoTransform[] BuildTorsoTransforms(
            ComponentBuffer pose,
            bool traceEnabled,
            out Step9DTorsoTransformProvenance[] provenance)
        {
            int frames = pose.Coordinates.GetLength(0);
            var transforms = new TorsoTransform[frames];
            provenance = traceEnabled ? new Step9DTorsoTransformProvenance[frames] : null;
            for (int frame = 0; frame < frames; frame++)
            {
                transforms[frame] = ComputeTorsoTransform(pose, frame);
                if (provenance != null)
                    provenance[frame] = transforms[frame].Valid
                        ? Step9DTorsoTransformProvenance.DIRECT
                        : Step9DTorsoTransformProvenance.UNAVAILABLE;
            }

            TorsoTransform previous = null;
            for (int frame = 0; frame < frames; frame++)
            {
                if (transforms[frame].Valid) previous = transforms[frame];
                else if (previous != null)
                {
                    transforms[frame] = previous.Clone();
                    if (provenance != null) provenance[frame] = Step9DTorsoTransformProvenance.FORWARD_CARRIED;
                }
            }
            TorsoTransform first = null;
            for (int frame = 0; frame < frames; frame++) if (transforms[frame].Valid) { first = transforms[frame]; break; }
            if (first != null)
                for (int frame = 0; frame < frames && !transforms[frame].Valid; frame++)
                {
                    transforms[frame] = first.Clone();
                    if (provenance != null) provenance[frame] = Step9DTorsoTransformProvenance.LEADING_BACKFILLED;
                }
            return transforms;
        }

        static TorsoTransform ComputeTorsoTransform(ComponentBuffer pose, int frame)
        {
            const int leftShoulder = 11, rightShoulder = 12, leftHip = 23, rightHip = 24;
            var result = new TorsoTransform();
            if (!IsOriginalFinite(pose, frame, leftShoulder) || !IsOriginalFinite(pose, frame, rightShoulder)) return result;
            double[] ls = Point(pose.Coordinates, frame, leftShoulder);
            double[] rs = Point(pose.Coordinates, frame, rightShoulder);
            for (int axis = 0; axis < 3; axis++) result.Origin[axis] = (ls[axis] + rs[axis]) / 2.0;
            double[] right = Subtract(rs, ls);
            double shoulderWidth = Norm(right);
            if (shoulderWidth < MinimumShoulderWidth) return result;
            ScaleInPlace(right, 1.0 / shoulderWidth);

            bool hasLeftHip = IsOriginalFinite(pose, frame, leftHip);
            bool hasRightHip = IsOriginalFinite(pose, frame, rightHip);
            double[] upVector;
            if (hasLeftHip && hasRightHip)
            {
                double[] lh = Point(pose.Coordinates, frame, leftHip);
                double[] rh = Point(pose.Coordinates, frame, rightHip);
                upVector = new double[3];
                for (int axis = 0; axis < 3; axis++) upVector[axis] = result.Origin[axis] - (lh[axis] + rh[axis]) / 2.0;
            }
            else if (hasLeftHip) upVector = Subtract(result.Origin, Point(pose.Coordinates, frame, leftHip));
            else if (hasRightHip) upVector = Subtract(result.Origin, Point(pose.Coordinates, frame, rightHip));
            else upVector = new[] { 0.0, -1.0, 0.0 };

            double upNorm = Norm(upVector);
            if (upNorm < MinimumShoulderWidth) { upVector = new[] { 0.0, -1.0, 0.0 }; upNorm = 1.0; }
            ScaleInPlace(upVector, 1.0 / upNorm);
            double[] forward = Cross(right, upVector);
            double forwardNorm = Norm(forward);
            if (forwardNorm < MinimumShoulderWidth) return result;
            ScaleInPlace(forward, 1.0 / forwardNorm);
            double[] correctedUp = Cross(forward, right);
            ScaleInPlace(correctedUp, 1.0 / Norm(correctedUp));

            Array.Copy(right, result.Right, 3);
            Array.Copy(correctedUp, result.Up, 3);
            Array.Copy(forward, result.Forward, 3);
            result.Scale = shoulderWidth;
            result.Valid = true;
            return result;
        }

        static int WriteComponent(
            int frame,
            ComponentBuffer component,
            int[] landmarks,
            TorsoTransform transform,
            double[] outputCoordinates,
            bool[] outputMask,
            int outputIndex,
            ref int validCount,
            ref int interpolatedCount)
        {
            foreach (int landmark in landmarks)
            {
                bool valid = (component.OriginalValid[frame, landmark] || component.Interpolated[frame, landmark]) &&
                    IsFinitePoint(component.Coordinates, frame, landmark) && transform.Valid;
                outputMask[outputIndex] = valid;
                if (valid)
                {
                    double x = component.Coordinates[frame, landmark, 0] - transform.Origin[0];
                    double y = component.Coordinates[frame, landmark, 1] - transform.Origin[1];
                    double z = component.Coordinates[frame, landmark, 2] - transform.Origin[2];
                    int offset = outputIndex * 3;
                    outputCoordinates[offset] = Dot(transform.Right, x, y, z) / transform.Scale;
                    outputCoordinates[offset + 1] = Dot(transform.Up, x, y, z) / transform.Scale;
                    outputCoordinates[offset + 2] = Dot(transform.Forward, x, y, z) / transform.Scale;
                    validCount++;
                    if (component.Interpolated[frame, landmark]) interpolatedCount++;
                }
                outputIndex++;
            }
            return outputIndex;
        }

        /// <summary>
        /// F2 candidate only. Copies the SOURCE original-valid and interpolated flags for
        /// the emitted landmarks. These are provenance, not the final normalized mask.
        /// </summary>
        static int WriteFullHandProvenance(
            int frame,
            ComponentBuffer component,
            int[] landmarks,
            bool[] originalValid,
            bool[] interpolated,
            int outputIndex)
        {
            foreach (int landmark in landmarks)
            {
                originalValid[outputIndex] = component.OriginalValid[frame, landmark];
                interpolated[outputIndex] = component.Interpolated[frame, landmark];
                outputIndex++;
            }
            return outputIndex;
        }

        static Step9DRawIntermediatePreprocessingTrace CreateTrace(
            IReadOnlyList<RawMediaPipeObservation> observations,
            ComponentBuffer pose,
            ComponentBuffer left,
            ComponentBuffer right,
            TorsoTransform[] transforms,
            Step9DTorsoTransformProvenance[] torsoProvenance,
            NormalizedRecognitionFrame[] frames,
            int validLandmarkCount,
            int interpolatedLandmarkCount)
        {
            int frameCount = observations.Count;
            int rawLandmarkCount = Step9DRawIntermediateDiagnosticContract.RawLandmarkCount;
            var identities = new Step9DRawIntermediateFrameIdentity[frameCount];
            var rawCoordinates = new double[frameCount * rawLandmarkCount * 3];
            var rawOriginalValid = new bool[frameCount * rawLandmarkCount];
            var rawReappearance = new bool[frameCount * rawLandmarkCount];
            var rawStrongDepth = new bool[frameCount * rawLandmarkCount];
            var timestampBoundary = new bool[frameCount * rawLandmarkCount];
            var combinedBoundary = new bool[frameCount * rawLandmarkCount];
            var postGapCoordinates = new double[frameCount * rawLandmarkCount * 3];
            var interpolationMask = new bool[frameCount * rawLandmarkCount];
            var torsoSourceValid = new bool[frameCount];
            var torsoFinalValid = new bool[frameCount];
            var torsoOrigin = NewNaNArray(frameCount * 3);
            var torsoRight = NewNaNArray(frameCount * 3);
            var torsoUp = NewNaNArray(frameCount * 3);
            var torsoForward = NewNaNArray(frameCount * 3);
            var torsoScale = NewNaNArray(frameCount);
            var normalizedCoordinates = new double[frameCount * NormalizedRecognitionFrame.CoordinateCount];
            var normalizedMask = new bool[frameCount * NormalizedRecognitionFrame.LandmarkCount];

            for (int frame = 0; frame < frameCount; frame++)
            {
                RawMediaPipeObservation observation = observations[frame];
                identities[frame] = Step9DRawIntermediateFrameIdentity.WithoutProviderIdentity(frame, observation.TimestampSeconds);
                CopyTraceComponent(frame, 0, observation.Pose, pose, rawCoordinates, rawOriginalValid, rawReappearance,
                    rawStrongDepth, timestampBoundary, combinedBoundary, postGapCoordinates, interpolationMask);
                CopyTraceComponent(frame, RawMediaPipeObservation.PoseLandmarkCount, observation.LeftHand, left, rawCoordinates,
                    rawOriginalValid, rawReappearance, rawStrongDepth, timestampBoundary, combinedBoundary,
                    postGapCoordinates, interpolationMask);
                CopyTraceComponent(frame, RawMediaPipeObservation.PoseLandmarkCount + RawMediaPipeObservation.HandLandmarkCount,
                    observation.RightHand, right, rawCoordinates, rawOriginalValid, rawReappearance, rawStrongDepth,
                    timestampBoundary, combinedBoundary, postGapCoordinates, interpolationMask);

                torsoSourceValid[frame] = torsoProvenance[frame] == Step9DTorsoTransformProvenance.DIRECT;
                torsoFinalValid[frame] = transforms[frame].Valid;
                if (transforms[frame].Valid)
                {
                    int vectorOffset = frame * 3;
                    Array.Copy(transforms[frame].Origin, 0, torsoOrigin, vectorOffset, 3);
                    Array.Copy(transforms[frame].Right, 0, torsoRight, vectorOffset, 3);
                    Array.Copy(transforms[frame].Up, 0, torsoUp, vectorOffset, 3);
                    Array.Copy(transforms[frame].Forward, 0, torsoForward, vectorOffset, 3);
                    torsoScale[frame] = transforms[frame].Scale;
                }

                double[] frameCoordinates = frames[frame].CopyCoordinates();
                bool[] frameMask = frames[frame].CopyValidMask();
                Array.Copy(frameCoordinates, 0, normalizedCoordinates, frame * NormalizedRecognitionFrame.CoordinateCount, frameCoordinates.Length);
                Array.Copy(frameMask, 0, normalizedMask, frame * NormalizedRecognitionFrame.LandmarkCount, frameMask.Length);
            }

            return new Step9DRawIntermediatePreprocessingTrace(
                observations[0].Convention,
                identities,
                rawCoordinates,
                rawOriginalValid,
                rawReappearance,
                rawStrongDepth,
                timestampBoundary,
                combinedBoundary,
                postGapCoordinates,
                interpolationMask,
                torsoSourceValid,
                torsoFinalValid,
                torsoProvenance,
                torsoOrigin,
                torsoRight,
                torsoUp,
                torsoForward,
                torsoScale,
                normalizedCoordinates,
                normalizedMask,
                validLandmarkCount,
                interpolatedLandmarkCount,
                SelectedMaxGap,
                TimestampDiscontinuitySeconds);
        }

        static void CopyTraceComponent(
            int frame,
            int rawLandmarkOffset,
            RawLandmarkObservationSet source,
            ComponentBuffer component,
            double[] rawCoordinates,
            bool[] rawOriginalValid,
            bool[] rawReappearance,
            bool[] rawStrongDepth,
            bool[] timestampBoundary,
            bool[] combinedBoundary,
            double[] postGapCoordinates,
            bool[] interpolationMask)
        {
            int rawFrameOffset = frame * Step9DRawIntermediateDiagnosticContract.RawLandmarkCount;
            for (int landmark = 0; landmark < component.Count; landmark++)
            {
                int rawLandmark = rawLandmarkOffset + landmark;
                int maskIndex = rawFrameOffset + rawLandmark;
                int coordinateIndex = maskIndex * 3;
                rawOriginalValid[maskIndex] = component.OriginalValid[frame, landmark];
                rawReappearance[maskIndex] = component.Reappearance[frame, landmark];
                rawStrongDepth[maskIndex] = component.Strong[frame, landmark];
                timestampBoundary[maskIndex] = component.TimestampBoundary[frame, landmark];
                combinedBoundary[maskIndex] = component.Boundary[frame, landmark];
                interpolationMask[maskIndex] = component.Interpolated[frame, landmark];
                for (int axis = 0; axis < 3; axis++)
                {
                    rawCoordinates[coordinateIndex + axis] = source.Coordinate(landmark, axis);
                    postGapCoordinates[coordinateIndex + axis] = component.Coordinates[frame, landmark, axis];
                }
            }
        }

        static double[] NewNaNArray(int length)
        {
            var values = new double[length];
            for (int index = 0; index < values.Length; index++) values[index] = double.NaN;
            return values;
        }

        static bool IsOriginalFinite(ComponentBuffer value, int frame, int landmark) =>
            value.OriginalValid[frame, landmark] && IsFinitePoint(value.Coordinates, frame, landmark);
        static bool IsFinitePoint(double[,,] values, int frame, int landmark) =>
            IsFinite(values[frame, landmark, 0]) && IsFinite(values[frame, landmark, 1]) && IsFinite(values[frame, landmark, 2]);
        static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
        static double[] Point(double[,,] values, int frame, int landmark) =>
            new[] { values[frame, landmark, 0], values[frame, landmark, 1], values[frame, landmark, 2] };
        static double[] Subtract(double[] a, double[] b) => new[] { a[0] - b[0], a[1] - b[1], a[2] - b[2] };
        static double Norm(double[] value) => Math.Sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
        static void ScaleInPlace(double[] value, double scale) { value[0] *= scale; value[1] *= scale; value[2] *= scale; }
        static double[] Cross(double[] a, double[] b) => new[]
        {
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        };
        static double Dot(double[] axis, double x, double y, double z) => axis[0] * x + axis[1] * y + axis[2] * z;
    }
}
