using System;
using WordsInSilence.Recognition.Diagnostics;

namespace WordsInSilence.Recognition.LiveInput
{
    public enum RawObservationCoordinateSpace
    {
        ApprovedFrontCameraMetric3D,
        MediaPipeNormalizedImage,
    }

    /// <summary>Immutable source convention selected once for an entire diagnostic capture.</summary>
    public sealed class RawObservationConvention : IEquatable<RawObservationConvention>
    {
        public static readonly RawObservationConvention ApprovedStep7B = new RawObservationConvention(
            "approved_step7b_front_camera_metric_3d",
            RawObservationCoordinateSpace.ApprovedFrontCameraMetric3D,
            "FRONT_camera_meters; x/y/z preserved",
            "approved Step 7B left_hand/right_hand arrays",
            "not mirrored");

        public static readonly RawObservationConvention LiveMediaPipe = new RawObservationConvention(
            "mediapipe_tasks_webcam_normalized_image_v0_16_3",
            RawObservationCoordinateSpace.MediaPipeNormalizedImage,
            "MediaPipe NormalizedLandmark x-right/y-down/z-source-relative; no axis rewrite",
            "MediaPipe handedness categoryName mapped directly",
            "raw WebCamTexture pixels; RawImage UV mirror is display-only");

        /// <summary>F51 opt-in upright inference; the legacy constant above is unchanged.</summary>
        public static readonly RawObservationConvention LiveMediaPipeUprightTexture = new RawObservationConvention(
            "mediapipe_tasks_webcam_upright_source_texture_v1",
            RawObservationCoordinateSpace.MediaPipeNormalizedImage,
            "upright top-down inference; output x-right/y-up in source texture grid; z unchanged; post-inference y=1-y",
            "MediaPipe handedness categoryName mapped directly from upright inference; never swapped",
            "raw WebCamTexture pixels with pre-inference row reversal; RawImage UV mirror is display-only");

        public string Id { get; }
        public RawObservationCoordinateSpace CoordinateSpace { get; }
        public string AxisAndDepth { get; }
        public string Handedness { get; }
        public string Mirror { get; }

        public RawObservationConvention(
            string id,
            RawObservationCoordinateSpace coordinateSpace,
            string axisAndDepth,
            string handedness,
            string mirror)
        {
            if (string.IsNullOrWhiteSpace(id)) throw new ArgumentException("convention ID is required", nameof(id));
            if (string.IsNullOrWhiteSpace(axisAndDepth)) throw new ArgumentException("axis/depth metadata is required", nameof(axisAndDepth));
            if (string.IsNullOrWhiteSpace(handedness)) throw new ArgumentException("handedness metadata is required", nameof(handedness));
            if (string.IsNullOrWhiteSpace(mirror)) throw new ArgumentException("mirror metadata is required", nameof(mirror));
            Id = id;
            CoordinateSpace = coordinateSpace;
            AxisAndDepth = axisAndDepth;
            Handedness = handedness;
            Mirror = mirror;
        }

        public bool Equals(RawObservationConvention other)
        {
            return other != null &&
                string.Equals(Id, other.Id, StringComparison.Ordinal) &&
                CoordinateSpace == other.CoordinateSpace &&
                string.Equals(AxisAndDepth, other.AxisAndDepth, StringComparison.Ordinal) &&
                string.Equals(Handedness, other.Handedness, StringComparison.Ordinal) &&
                string.Equals(Mirror, other.Mirror, StringComparison.Ordinal);
        }

        public override bool Equals(object obj) => Equals(obj as RawObservationConvention);
        public override int GetHashCode() => Id.GetHashCode();
    }

    /// <summary>Immutable coordinates, validity and discontinuity metadata for one landmark component.</summary>
    public sealed class RawLandmarkObservationSet
    {
        readonly double[] coordinates;
        readonly bool[] valid;
        readonly bool[] reappearance;
        readonly bool[] strongDepthBoundary;

        public int LandmarkCount { get; }

        public RawLandmarkObservationSet(
            int landmarkCount,
            double[] coordinateValues,
            bool[] validMask,
            bool[] reappearanceMask,
            bool[] strongDepthBoundaryMask)
        {
            if (landmarkCount <= 0) throw new ArgumentOutOfRangeException(nameof(landmarkCount));
            if (coordinateValues == null) throw new ArgumentNullException(nameof(coordinateValues));
            if (validMask == null) throw new ArgumentNullException(nameof(validMask));
            if (reappearanceMask == null) throw new ArgumentNullException(nameof(reappearanceMask));
            if (strongDepthBoundaryMask == null) throw new ArgumentNullException(nameof(strongDepthBoundaryMask));
            if (coordinateValues.Length != landmarkCount * 3) throw new ArgumentException("coordinate shape mismatch", nameof(coordinateValues));
            if (validMask.Length != landmarkCount) throw new ArgumentException("valid-mask shape mismatch", nameof(validMask));
            if (reappearanceMask.Length != landmarkCount) throw new ArgumentException("reappearance-mask shape mismatch", nameof(reappearanceMask));
            if (strongDepthBoundaryMask.Length != landmarkCount) throw new ArgumentException("strong-depth-mask shape mismatch", nameof(strongDepthBoundaryMask));

            for (int landmark = 0; landmark < landmarkCount; landmark++)
            {
                int offset = landmark * 3;
                bool allFinite = IsFinite(coordinateValues[offset]) && IsFinite(coordinateValues[offset + 1]) && IsFinite(coordinateValues[offset + 2]);
                bool allNaN = double.IsNaN(coordinateValues[offset]) && double.IsNaN(coordinateValues[offset + 1]) && double.IsNaN(coordinateValues[offset + 2]);
                if (validMask[landmark] && !allFinite)
                    throw new ArgumentException($"valid coordinate is non-finite at landmark {landmark}", nameof(coordinateValues));
                if (!validMask[landmark] && !allNaN)
                    throw new ArgumentException($"absent landmark must contain three NaN values at landmark {landmark}", nameof(coordinateValues));
            }

            LandmarkCount = landmarkCount;
            coordinates = (double[])coordinateValues.Clone();
            valid = (bool[])validMask.Clone();
            reappearance = (bool[])reappearanceMask.Clone();
            strongDepthBoundary = (bool[])strongDepthBoundaryMask.Clone();
        }

        public static RawLandmarkObservationSet Absent(int landmarkCount)
        {
            var values = new double[landmarkCount * 3];
            for (int index = 0; index < values.Length; index++) values[index] = double.NaN;
            return new RawLandmarkObservationSet(
                landmarkCount, values, new bool[landmarkCount], new bool[landmarkCount], new bool[landmarkCount]);
        }

        public double Coordinate(int landmark, int axis)
        {
            if (landmark < 0 || landmark >= LandmarkCount) throw new ArgumentOutOfRangeException(nameof(landmark));
            if (axis < 0 || axis >= 3) throw new ArgumentOutOfRangeException(nameof(axis));
            return coordinates[landmark * 3 + axis];
        }

        public bool IsValid(int landmark) => valid[CheckedLandmark(landmark)];
        public bool IsReappearance(int landmark) => reappearance[CheckedLandmark(landmark)];
        public bool IsStrongDepthBoundary(int landmark) => strongDepthBoundary[CheckedLandmark(landmark)];
        public double[] CopyCoordinates() => (double[])coordinates.Clone();
        public bool[] CopyValidMask() => (bool[])valid.Clone();

        int CheckedLandmark(int landmark)
        {
            if (landmark < 0 || landmark >= LandmarkCount) throw new ArgumentOutOfRangeException(nameof(landmark));
            return landmark;
        }

        static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
    }

    /// <summary>One synchronized provider-neutral pose/left-hand/right-hand observation.</summary>
    public sealed class RawMediaPipeObservation
    {
        public const int PoseLandmarkCount = 33;
        public const int HandLandmarkCount = 21;

        public double TimestampSeconds { get; }
        public RawObservationConvention Convention { get; }
        public RawLandmarkObservationSet Pose { get; }
        public RawLandmarkObservationSet LeftHand { get; }
        public RawLandmarkObservationSet RightHand { get; }

        /// <summary>Merged inference geometry from hand+pose providers. Null when not available.</summary>
        public Step9DMergedInferenceGeometry MergedGeometry { get; }

        public RawMediaPipeObservation(
            double timestampSeconds,
            RawObservationConvention convention,
            RawLandmarkObservationSet pose,
            RawLandmarkObservationSet leftHand,
            RawLandmarkObservationSet rightHand,
            Step9DMergedInferenceGeometry mergedGeometry = null)
        {
            if (!IsFinite(timestampSeconds) || timestampSeconds < 0.0)
                throw new ArgumentOutOfRangeException(nameof(timestampSeconds), "timestamp must be finite and non-negative");
            Convention = convention ?? throw new ArgumentNullException(nameof(convention));
            Pose = RequireCount(pose, PoseLandmarkCount, nameof(pose));
            LeftHand = RequireCount(leftHand, HandLandmarkCount, nameof(leftHand));
            RightHand = RequireCount(rightHand, HandLandmarkCount, nameof(rightHand));
            TimestampSeconds = timestampSeconds;
            MergedGeometry = mergedGeometry;
        }

        static RawLandmarkObservationSet RequireCount(RawLandmarkObservationSet value, int count, string parameter)
        {
            if (value == null) throw new ArgumentNullException(parameter);
            if (value.LandmarkCount != count) throw new ArgumentException($"expected {count} landmarks", parameter);
            return value;
        }

        static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
    }
}
