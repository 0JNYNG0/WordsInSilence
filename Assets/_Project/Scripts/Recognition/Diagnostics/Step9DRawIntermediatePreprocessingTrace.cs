using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Recognition.Diagnostics
{
    public enum Step9DTorsoTransformProvenance
    {
        DIRECT,
        FORWARD_CARRIED,
        LEADING_BACKFILLED,
        UNAVAILABLE,
    }

    public enum Step9DRawIntermediateDiagnosticFailureCode
    {
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
    }

    public sealed class Step9DRawIntermediateDiagnosticException : Exception
    {
        public Step9DRawIntermediateDiagnosticFailureCode Code { get; }

        public Step9DRawIntermediateDiagnosticException(
            Step9DRawIntermediateDiagnosticFailureCode code,
            string message)
            : base(message)
        {
            Code = code;
        }

        public Step9DRawIntermediateDiagnosticException(
            Step9DRawIntermediateDiagnosticFailureCode code,
            string message,
            Exception innerException)
            : base(message, innerException)
        {
            Code = code;
        }
    }

    public static class Step9DRawIntermediateDiagnosticContract
    {
        public const string SchemaVersion = "words_in_silence.step9d.raw_intermediate_diagnostic.v1_0";
        public const int RawLandmarkCount = 75;
        public const int NormalizedLandmarkCount = 16;

        public static void ValidateFlatLength(string fieldName, int actualLength, params int[] shape)
        {
            if (string.IsNullOrWhiteSpace(fieldName)) throw new ArgumentException("field name is required", nameof(fieldName));
            if (shape == null || shape.Length == 0) throw new ArgumentException("shape is required", nameof(shape));
            int expected = 1;
            foreach (int dimension in shape)
            {
                if (dimension < 0) throw new ArgumentOutOfRangeException(nameof(shape));
                checked { expected *= dimension; }
            }
            if (actualLength != expected)
                throw new Step9DRawIntermediateDiagnosticException(
                    Step9DRawIntermediateDiagnosticFailureCode.INVALID_SHAPE,
                    $"{fieldName} has flat length {actualLength}; expected {expected} for shape [{string.Join(",", shape)}]");
        }

        public static void RejectInfinity(string fieldName, double value)
        {
            if (double.IsInfinity(value))
                throw new Step9DRawIntermediateDiagnosticException(
                    Step9DRawIntermediateDiagnosticFailureCode.NONFINITE_VALUE,
                    $"{fieldName} contains Infinity");
        }

        internal static void RejectInfinity(string fieldName, double[] values)
        {
            if (values == null) throw new ArgumentNullException(nameof(values));
            for (int index = 0; index < values.Length; index++) RejectInfinity($"{fieldName}[{index}]", values[index]);
        }
    }

    public sealed class Step9DRawIntermediateFrameIdentity
    {
        public int CaptureLocalFrameIndex { get; }
        public double ObservationTimestampSeconds { get; }
        public long? ProviderFrameIndex { get; }
        public long? BridgeObservationIndex { get; }
        public long? LifetimeProviderFrameIndex { get; }
        public bool HasProviderIdentity => ProviderFrameIndex.HasValue;

        public Step9DRawIntermediateFrameIdentity(
            int captureLocalFrameIndex,
            double observationTimestampSeconds,
            long? providerFrameIndex,
            long? bridgeObservationIndex,
            long? lifetimeProviderFrameIndex)
        {
            if (captureLocalFrameIndex < 0) throw new ArgumentOutOfRangeException(nameof(captureLocalFrameIndex));
            if (double.IsNaN(observationTimestampSeconds) || double.IsInfinity(observationTimestampSeconds))
                throw new Step9DRawIntermediateDiagnosticException(
                    Step9DRawIntermediateDiagnosticFailureCode.NONFINITE_VALUE,
                    "frame identity timestamp must be finite");
            bool any = providerFrameIndex.HasValue || bridgeObservationIndex.HasValue || lifetimeProviderFrameIndex.HasValue;
            bool all = providerFrameIndex.HasValue && bridgeObservationIndex.HasValue && lifetimeProviderFrameIndex.HasValue;
            if (any && !all)
                throw new Step9DRawIntermediateDiagnosticException(
                    Step9DRawIntermediateDiagnosticFailureCode.FRAME_IDENTITY_MISMATCH,
                    "provider, bridge and lifetime identity must be supplied together or all omitted");
            if (all && (providerFrameIndex.Value < 0 || bridgeObservationIndex.Value < 0 || lifetimeProviderFrameIndex.Value < 0))
                throw new Step9DRawIntermediateDiagnosticException(
                    Step9DRawIntermediateDiagnosticFailureCode.FRAME_IDENTITY_MISMATCH,
                    "available frame identity values must be non-negative");

            CaptureLocalFrameIndex = captureLocalFrameIndex;
            ObservationTimestampSeconds = observationTimestampSeconds;
            ProviderFrameIndex = providerFrameIndex;
            BridgeObservationIndex = bridgeObservationIndex;
            LifetimeProviderFrameIndex = lifetimeProviderFrameIndex;
        }

        public static Step9DRawIntermediateFrameIdentity WithoutProviderIdentity(int captureLocalFrameIndex, double timestampSeconds) =>
            new Step9DRawIntermediateFrameIdentity(captureLocalFrameIndex, timestampSeconds, null, null, null);

        internal Step9DRawIntermediateFrameIdentity Clone() => new Step9DRawIntermediateFrameIdentity(
            CaptureLocalFrameIndex,
            ObservationTimestampSeconds,
            ProviderFrameIndex,
            BridgeObservationIndex,
            LifetimeProviderFrameIndex);
    }

    public sealed class Step9DRawIntermediatePreprocessingTrace
    {
        static readonly string[] RawOrdering =
        {
            "pose[0..32]", "left_hand[0..20]", "right_hand[0..20]",
        };

        static readonly string[] NormalizedOrdering =
        {
            "pose[13]", "pose[14]", "pose[15]", "pose[16]",
            "left_hand[0]", "left_hand[4]", "left_hand[8]", "left_hand[12]", "left_hand[16]", "left_hand[20]",
            "right_hand[0]", "right_hand[4]", "right_hand[8]", "right_hand[12]", "right_hand[16]", "right_hand[20]",
        };

        readonly Step9DRawIntermediateFrameIdentity[] frameIdentities;
        readonly double[] rawCoordinates;
        readonly bool[] rawOriginalValidMask;
        readonly bool[] rawReappearanceMask;
        readonly bool[] rawStrongDepthMask;
        readonly bool[] timestampBoundaryMask;
        readonly bool[] combinedBoundaryMask;
        readonly double[] postGapInterpolationCoordinates;
        readonly bool[] interpolationMask;
        readonly bool[] torsoSourceValid;
        readonly bool[] torsoFinalValid;
        readonly Step9DTorsoTransformProvenance[] torsoProvenance;
        readonly double[] torsoOrigin;
        readonly double[] torsoRight;
        readonly double[] torsoUp;
        readonly double[] torsoForward;
        readonly double[] torsoShoulderWidthScale;
        readonly double[] normalizedCoordinates;
        readonly bool[] normalizedMask;

        public string SchemaVersion => Step9DRawIntermediateDiagnosticContract.SchemaVersion;
        public string CoordinateConventionId { get; }
        public RawObservationCoordinateSpace CoordinateSpace { get; }
        public string AxisAndDepth { get; }
        public string Handedness { get; }
        public string Mirror { get; }
        public int SelectedMaxGap { get; }
        public double TimestampDiscontinuitySeconds { get; }
        public int RawFrameCount { get; }
        public int NormalizedFrameCount { get; }
        public int ValidLandmarkCount { get; }
        public int InterpolatedLandmarkCount { get; }
        public int[] RawCoordinatesShape => new[] { RawFrameCount, Step9DRawIntermediateDiagnosticContract.RawLandmarkCount, 3 };
        public int[] RawMaskShape => new[] { RawFrameCount, Step9DRawIntermediateDiagnosticContract.RawLandmarkCount };
        public int[] TorsoVectorShape => new[] { RawFrameCount, 3 };
        public int[] NormalizedCoordinatesShape => new[] { NormalizedFrameCount, Step9DRawIntermediateDiagnosticContract.NormalizedLandmarkCount, 3 };
        public int[] NormalizedMaskShape => new[] { NormalizedFrameCount, Step9DRawIntermediateDiagnosticContract.NormalizedLandmarkCount };

        internal Step9DRawIntermediatePreprocessingTrace(
            RawObservationConvention convention,
            Step9DRawIntermediateFrameIdentity[] identities,
            double[] rawCoordinates,
            bool[] rawOriginalValidMask,
            bool[] rawReappearanceMask,
            bool[] rawStrongDepthMask,
            bool[] timestampBoundaryMask,
            bool[] combinedBoundaryMask,
            double[] postGapInterpolationCoordinates,
            bool[] interpolationMask,
            bool[] torsoSourceValid,
            bool[] torsoFinalValid,
            Step9DTorsoTransformProvenance[] torsoProvenance,
            double[] torsoOrigin,
            double[] torsoRight,
            double[] torsoUp,
            double[] torsoForward,
            double[] torsoShoulderWidthScale,
            double[] normalizedCoordinates,
            bool[] normalizedMask,
            int validLandmarkCount,
            int interpolatedLandmarkCount,
            int selectedMaxGap,
            double timestampDiscontinuitySeconds)
        {
            if (convention == null) throw new ArgumentNullException(nameof(convention));
            if (identities == null) throw new ArgumentNullException(nameof(identities));
            int frames = identities.Length;
            ValidateLengths(
                frames, rawCoordinates, rawOriginalValidMask, rawReappearanceMask, rawStrongDepthMask,
                timestampBoundaryMask, combinedBoundaryMask, postGapInterpolationCoordinates, interpolationMask,
                torsoSourceValid, torsoFinalValid, torsoProvenance, torsoOrigin, torsoRight, torsoUp, torsoForward,
                torsoShoulderWidthScale, normalizedCoordinates, normalizedMask);
            Step9DRawIntermediateDiagnosticContract.RejectInfinity(nameof(rawCoordinates), rawCoordinates);
            Step9DRawIntermediateDiagnosticContract.RejectInfinity(nameof(postGapInterpolationCoordinates), postGapInterpolationCoordinates);
            Step9DRawIntermediateDiagnosticContract.RejectInfinity(nameof(torsoOrigin), torsoOrigin);
            Step9DRawIntermediateDiagnosticContract.RejectInfinity(nameof(torsoRight), torsoRight);
            Step9DRawIntermediateDiagnosticContract.RejectInfinity(nameof(torsoUp), torsoUp);
            Step9DRawIntermediateDiagnosticContract.RejectInfinity(nameof(torsoForward), torsoForward);
            Step9DRawIntermediateDiagnosticContract.RejectInfinity(nameof(torsoShoulderWidthScale), torsoShoulderWidthScale);
            Step9DRawIntermediateDiagnosticContract.RejectInfinity(nameof(normalizedCoordinates), normalizedCoordinates);

            CoordinateConventionId = convention.Id;
            CoordinateSpace = convention.CoordinateSpace;
            AxisAndDepth = convention.AxisAndDepth;
            Handedness = convention.Handedness;
            Mirror = convention.Mirror;
            SelectedMaxGap = selectedMaxGap;
            TimestampDiscontinuitySeconds = timestampDiscontinuitySeconds;
            RawFrameCount = frames;
            NormalizedFrameCount = frames;
            ValidLandmarkCount = validLandmarkCount;
            InterpolatedLandmarkCount = interpolatedLandmarkCount;
            frameIdentities = CloneIdentities(identities);
            this.rawCoordinates = (double[])rawCoordinates.Clone();
            this.rawOriginalValidMask = (bool[])rawOriginalValidMask.Clone();
            this.rawReappearanceMask = (bool[])rawReappearanceMask.Clone();
            this.rawStrongDepthMask = (bool[])rawStrongDepthMask.Clone();
            this.timestampBoundaryMask = (bool[])timestampBoundaryMask.Clone();
            this.combinedBoundaryMask = (bool[])combinedBoundaryMask.Clone();
            this.postGapInterpolationCoordinates = (double[])postGapInterpolationCoordinates.Clone();
            this.interpolationMask = (bool[])interpolationMask.Clone();
            this.torsoSourceValid = (bool[])torsoSourceValid.Clone();
            this.torsoFinalValid = (bool[])torsoFinalValid.Clone();
            this.torsoProvenance = (Step9DTorsoTransformProvenance[])torsoProvenance.Clone();
            this.torsoOrigin = (double[])torsoOrigin.Clone();
            this.torsoRight = (double[])torsoRight.Clone();
            this.torsoUp = (double[])torsoUp.Clone();
            this.torsoForward = (double[])torsoForward.Clone();
            this.torsoShoulderWidthScale = (double[])torsoShoulderWidthScale.Clone();
            this.normalizedCoordinates = (double[])normalizedCoordinates.Clone();
            this.normalizedMask = (bool[])normalizedMask.Clone();
        }

        public IReadOnlyList<string> CopyRawLandmarkOrdering() => new ReadOnlyCollection<string>((string[])RawOrdering.Clone());
        public IReadOnlyList<string> CopyNormalizedLandmarkOrdering() => new ReadOnlyCollection<string>((string[])NormalizedOrdering.Clone());
        public Step9DRawIntermediateFrameIdentity[] CopyFrameIdentities() => CloneIdentities(frameIdentities);
        public double[] CopyRawCoordinates() => (double[])rawCoordinates.Clone();
        public bool[] CopyRawOriginalValidMask() => (bool[])rawOriginalValidMask.Clone();
        public bool[] CopyRawReappearanceMask() => (bool[])rawReappearanceMask.Clone();
        public bool[] CopyRawStrongDepthMask() => (bool[])rawStrongDepthMask.Clone();
        public bool[] CopyTimestampBoundaryMask() => (bool[])timestampBoundaryMask.Clone();
        public bool[] CopyCombinedBoundaryMask() => (bool[])combinedBoundaryMask.Clone();
        public double[] CopyPostGapInterpolationCoordinates() => (double[])postGapInterpolationCoordinates.Clone();
        public bool[] CopyInterpolationMask() => (bool[])interpolationMask.Clone();
        public bool[] CopyTorsoSourceValid() => (bool[])torsoSourceValid.Clone();
        public bool[] CopyTorsoFinalValid() => (bool[])torsoFinalValid.Clone();
        public Step9DTorsoTransformProvenance[] CopyTorsoProvenance() => (Step9DTorsoTransformProvenance[])torsoProvenance.Clone();
        public double[] CopyTorsoOrigin() => (double[])torsoOrigin.Clone();
        public double[] CopyTorsoRight() => (double[])torsoRight.Clone();
        public double[] CopyTorsoUp() => (double[])torsoUp.Clone();
        public double[] CopyTorsoForward() => (double[])torsoForward.Clone();
        public double[] CopyTorsoShoulderWidthScale() => (double[])torsoShoulderWidthScale.Clone();
        public double[] CopyNormalizedCoordinates() => (double[])normalizedCoordinates.Clone();
        public bool[] CopyNormalizedMask() => (bool[])normalizedMask.Clone();

        public Step9DRawIntermediatePreprocessingTrace DeepCopy() => new Step9DRawIntermediatePreprocessingTrace(
            new RawObservationConvention(CoordinateConventionId, CoordinateSpace, AxisAndDepth, Handedness, Mirror),
            CopyFrameIdentities(), CopyRawCoordinates(), CopyRawOriginalValidMask(), CopyRawReappearanceMask(),
            CopyRawStrongDepthMask(), CopyTimestampBoundaryMask(), CopyCombinedBoundaryMask(),
            CopyPostGapInterpolationCoordinates(), CopyInterpolationMask(), CopyTorsoSourceValid(), CopyTorsoFinalValid(),
            CopyTorsoProvenance(), CopyTorsoOrigin(), CopyTorsoRight(), CopyTorsoUp(), CopyTorsoForward(),
            CopyTorsoShoulderWidthScale(), CopyNormalizedCoordinates(), CopyNormalizedMask(), ValidLandmarkCount,
            InterpolatedLandmarkCount, SelectedMaxGap, TimestampDiscontinuitySeconds);

        static void ValidateLengths(
            int frames,
            double[] raw,
            bool[] original,
            bool[] reappearance,
            bool[] strong,
            bool[] timestampBoundary,
            bool[] combinedBoundary,
            double[] postGap,
            bool[] interpolated,
            bool[] torsoSource,
            bool[] torsoFinal,
            Step9DTorsoTransformProvenance[] provenance,
            double[] origin,
            double[] right,
            double[] up,
            double[] forward,
            double[] scale,
            double[] normalized,
            bool[] normalizedMask)
        {
            if (frames <= 0)
                throw new Step9DRawIntermediateDiagnosticException(
                    Step9DRawIntermediateDiagnosticFailureCode.EMPTY_CAPTURE,
                    "trace requires at least one frame");
            if (raw == null || original == null || reappearance == null || strong == null || timestampBoundary == null ||
                combinedBoundary == null || postGap == null || interpolated == null || torsoSource == null || torsoFinal == null ||
                provenance == null || origin == null || right == null || up == null || forward == null || scale == null ||
                normalized == null || normalizedMask == null)
                throw new ArgumentNullException("trace arrays cannot be null");
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength(nameof(raw), raw.Length, frames, 75, 3);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength(nameof(original), original.Length, frames, 75);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength(nameof(reappearance), reappearance.Length, frames, 75);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength(nameof(strong), strong.Length, frames, 75);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength(nameof(timestampBoundary), timestampBoundary.Length, frames, 75);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength(nameof(combinedBoundary), combinedBoundary.Length, frames, 75);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength(nameof(postGap), postGap.Length, frames, 75, 3);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength(nameof(interpolated), interpolated.Length, frames, 75);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength(nameof(torsoSource), torsoSource.Length, frames);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength(nameof(torsoFinal), torsoFinal.Length, frames);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength(nameof(provenance), provenance.Length, frames);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength(nameof(origin), origin.Length, frames, 3);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength(nameof(right), right.Length, frames, 3);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength(nameof(up), up.Length, frames, 3);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength(nameof(forward), forward.Length, frames, 3);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength(nameof(scale), scale.Length, frames);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength(nameof(normalized), normalized.Length, frames, 16, 3);
            Step9DRawIntermediateDiagnosticContract.ValidateFlatLength(nameof(normalizedMask), normalizedMask.Length, frames, 16);
        }

        static Step9DRawIntermediateFrameIdentity[] CloneIdentities(Step9DRawIntermediateFrameIdentity[] values)
        {
            var copy = new Step9DRawIntermediateFrameIdentity[values.Length];
            for (int index = 0; index < values.Length; index++)
                copy[index] = values[index]?.Clone() ?? throw new ArgumentException($"frame identity {index} is null", nameof(values));
            return copy;
        }
    }

    public sealed class ApprovedPreprocessingTraceResult
    {
        public ApprovedPreprocessingResult Result { get; }
        public Step9DRawIntermediatePreprocessingTrace Trace { get; }

        internal ApprovedPreprocessingTraceResult(
            ApprovedPreprocessingResult result,
            Step9DRawIntermediatePreprocessingTrace trace)
        {
            Result = result ?? throw new ArgumentNullException(nameof(result));
            Trace = trace ?? throw new ArgumentNullException(nameof(trace));
        }
    }
}
