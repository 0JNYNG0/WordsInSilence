using System;

namespace WordsInSilence.Recognition.Common
{
    /// <summary>
    /// Builds the frozen 1536-d common-2D feature vector from an already-normalized
    /// [T,16,3] / [T,16] sequence.
    ///
    /// This is a direct port of the approved v1.2a / v1.3b / v1.6 code path:
    ///   - resample to exactly T=32 with the approved mask-aware linear rule
    ///   - feature order is [32,16,2] xy flattened C-order, then [32,16] mask
    ///   - channel 2 (the pseudo-3D forward component) is NEVER read
    /// No new transform, flip, swap, scaling or threshold is introduced.
    /// </summary>
    public static class CommonSignFeatureBuilder
    {
        public const int LandmarkCount = FrozenCommonSignModel.LandmarkCount;
        public const int ChannelCount = FrozenCommonSignModel.ChannelCount;
        public const int ResampleT = FrozenCommonSignModel.ResampleT;
        public const int FeatureCount = FrozenCommonSignModel.FeatureCount;
        public const int XyBlockLength = ResampleT * LandmarkCount * 2;   // 1024

        /// <summary>
        /// Attempts to build the frozen feature vector. Returns false with a reason on
        /// any shape, finiteness or all-missing problem; never produces a partial vector.
        /// </summary>
        public static bool TryBuild(
            double[] coordinates,
            bool[] validMask,
            int frameCount,
            out double[] features,
            out CommonSignFailureReason failure,
            out string detail)
        {
            return TryBuildInternal(
                coordinates, validMask, frameCount, out features, out failure, out detail,
                null, null);
        }

#if UNITY_EDITOR
        /// <summary>
        /// Editor-only observation entry point for the Step 10P v1.0f2 diagnostic trace.
        /// It runs the same <see cref="TryBuildInternal"/> body as
        /// <see cref="TryBuild"/>; the resampled sequence and mask handed back are the
        /// ones this build actually used, copied into caller-owned buffers.
        /// </summary>
        public static bool TryBuildWithTrace(
            double[] coordinates,
            bool[] validMask,
            int frameCount,
            out double[] features,
            out CommonSignFailureReason failure,
            out string detail,
            double[] resampledSink,
            bool[] resampledMaskSink)
        {
            return TryBuildInternal(
                coordinates, validMask, frameCount, out features, out failure, out detail,
                resampledSink, resampledMaskSink);
        }
#endif

        /// <summary>
        /// The single build path. The two optional sinks, when non-null, receive copies
        /// of the resampled sequence and mask this path produced. They are written after
        /// the values exist and never feed back into the feature vector.
        /// </summary>
        static bool TryBuildInternal(
            double[] coordinates,
            bool[] validMask,
            int frameCount,
            out double[] features,
            out CommonSignFailureReason failure,
            out string detail,
            double[] resampledSink,
            bool[] resampledMaskSink)
        {
            features = null;
            failure = CommonSignFailureReason.None;
            detail = string.Empty;

            if (coordinates == null || validMask == null)
            {
                failure = CommonSignFailureReason.SequenceMissing;
                detail = "coordinates or mask is null";
                return false;
            }
            if (frameCount <= 0)
            {
                failure = CommonSignFailureReason.InvalidShape;
                detail = "frame count must be positive, got " + frameCount;
                return false;
            }
            long expectedCoordinates = (long)frameCount * LandmarkCount * ChannelCount;
            long expectedMask = (long)frameCount * LandmarkCount;
            if (coordinates.LongLength != expectedCoordinates)
            {
                failure = CommonSignFailureReason.InvalidShape;
                detail = "coordinate shape mismatch: expected " + expectedCoordinates +
                    ", got " + coordinates.LongLength;
                return false;
            }
            if (validMask.LongLength != expectedMask)
            {
                failure = CommonSignFailureReason.InvalidShape;
                detail = "mask shape mismatch: expected " + expectedMask +
                    ", got " + validMask.LongLength;
                return false;
            }

            bool anyValid = false;
            for (int frame = 0; frame < frameCount; frame++)
            {
                for (int landmark = 0; landmark < LandmarkCount; landmark++)
                {
                    if (!validMask[frame * LandmarkCount + landmark]) continue;
                    anyValid = true;
                    int offset = (frame * LandmarkCount + landmark) * ChannelCount;
                    // Only the two channels the frozen model consumes must be finite.
                    for (int axis = 0; axis < 2; axis++)
                    {
                        double value = coordinates[offset + axis];
                        if (double.IsNaN(value) || double.IsInfinity(value))
                        {
                            failure = CommonSignFailureReason.NonFiniteInput;
                            detail = "non-finite valid coordinate at frame=" + frame +
                                ", landmark=" + landmark + ", axis=" + axis;
                            return false;
                        }
                    }
                }
            }
            if (!anyValid)
            {
                failure = CommonSignFailureReason.AllLandmarksMissing;
                detail = "no valid landmark in any frame";
                return false;
            }

            var resampled = new double[ResampleT * LandmarkCount * 2];
            var resampledMask = new bool[ResampleT * LandmarkCount];
            int denominator = ResampleT - 1;
            if (denominator < 1) denominator = 1;

            for (int t = 0; t < ResampleT; t++)
            {
                double target = t * (double)(frameCount - 1) / denominator;
                int low = (int)Math.Floor(target);
                int high = low + 1;
                if (high > frameCount - 1) high = frameCount - 1;
                double alpha = target - low;

                for (int landmark = 0; landmark < LandmarkCount; landmark++)
                {
                    bool lowValid = validMask[low * LandmarkCount + landmark];
                    bool highValid = validMask[high * LandmarkCount + landmark];
                    int outOffset = (t * LandmarkCount + landmark) * 2;
                    int lowOffset = (low * LandmarkCount + landmark) * ChannelCount;
                    int highOffset = (high * LandmarkCount + landmark) * ChannelCount;

                    if (lowValid && highValid)
                    {
                        for (int axis = 0; axis < 2; axis++)
                            resampled[outOffset + axis] =
                                (1.0 - alpha) * coordinates[lowOffset + axis] +
                                alpha * coordinates[highOffset + axis];
                        resampledMask[t * LandmarkCount + landmark] = true;
                    }
                    else if (lowValid)
                    {
                        for (int axis = 0; axis < 2; axis++)
                            resampled[outOffset + axis] = coordinates[lowOffset + axis];
                        resampledMask[t * LandmarkCount + landmark] = true;
                    }
                    else if (highValid)
                    {
                        for (int axis = 0; axis < 2; axis++)
                            resampled[outOffset + axis] = coordinates[highOffset + axis];
                        resampledMask[t * LandmarkCount + landmark] = true;
                    }
                    else
                    {
                        for (int axis = 0; axis < 2; axis++)
                            resampled[outOffset + axis] = 0.0;
                    }
                }
            }

            var result = new double[FeatureCount];
            Array.Copy(resampled, 0, result, 0, XyBlockLength);
            for (int i = 0; i < ResampleT * LandmarkCount; i++)
                result[XyBlockLength + i] = resampledMask[i] ? 1.0 : 0.0;

            if (resampledSink != null)
                Array.Copy(resampled, 0, resampledSink, 0, resampled.Length);
            if (resampledMaskSink != null)
                Array.Copy(resampledMask, 0, resampledMaskSink, 0, resampledMask.Length);

            features = result;
            return true;
        }
    }
}
