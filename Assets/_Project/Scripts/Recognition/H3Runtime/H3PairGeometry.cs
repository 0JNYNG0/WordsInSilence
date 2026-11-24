// M3_H3_RUNTIME_R1 - faithful C# port of the FROZEN H3 v1 geometry and pair features.
// Prepared in the workspace. NOT installed, NOT referenced by any production script.
//
// This is a numerical deployment port of an algorithm that is already frozen. It is not a
// new classifier, not a fit, not a threshold choice and not a runtime authority. Nothing
// here selects, learns or stores anything.
//
// Ported from, and kept identical to:
//   hybrid_core_v2.check_input_quality / resample_to_fixed / dtw_local_cost / align
//   h3_pair_features.features_from_alignment  (the 36 pinned features)
//
// Layout note: coordinates are the flat frame-major layout the project already uses,
// coords[(frame * 16 + landmark) * 3 + channel], mask[frame * 16 + landmark]. That is
// exactly what ApprovedLivePreprocessingAdapter writes into CommonTwoDimensionalCapture,
// so no repacking is needed at the capture seam.
//
// Faithfulness limits, stated rather than hidden:
//   * NumPy's pairwise summation is reproduced exactly for the two np.mean calls that
//     feed the features, and every DTW accumulation is a plain sequential loop in the
//     Python original, so those match by construction.
//   * np.percentile's 'linear' method, including NumPy's two-branch lerp, is reproduced
//     exactly.
//   * The one operation that cannot be guaranteed bit-identical is the logit dot product,
//     which NumPy may route through BLAS with a different order or FMA. Expect agreement
//     to a few ulp there, not bit equality. The harness measures the actual deviation; no
//     tolerance is asserted as passed here.
using System;
using System.Collections.Generic;

namespace WordsInSilence.Recognition.H3Runtime
{
    /// <summary>Typed failure codes. A failure is never substituted with a value.</summary>
    public static class H3FailureCodes
    {
        public const string Ok = "OK";
        public const string UnconvertibleInput = "UNCONVERTIBLE_INPUT";
        public const string MalformedShape = "MALFORMED_SHAPE";
        public const string MaskShapeMismatch = "MASK_SHAPE_MISMATCH";
        public const string EmptySequence = "EMPTY_SEQUENCE";
        public const string NonfiniteValidCoord = "NONFINITE_VALID_COORD";
        public const string AllLandmarksMissing = "ALL_LANDMARKS_MISSING";
        public const string NonzeroZ = "NONZERO_Z_IN_COMMON2D";
        public const string TooFewValidFrames = "TOO_FEW_VALID_FRAMES";
        public const string CoverageTooLow = "COVERAGE_TOO_LOW";

        public const string EmptyAlignSequence = "EMPTY_SEQUENCE";
        public const string NoValidPath = "NO_VALID_PATH";
        public const string NoJointlyValidLandmark = "NO_JOINTLY_VALID_LANDMARK";
        public const string DegeneratePath = "DEGENERATE_PATH";
        public const string MissingLandmarkOverlap = "MISSING_LANDMARK_OVERLAP";
        public const string NonfinitePairFeature = "NONFINITE_PAIR_FEATURE";
        public const string PairFeatureWidth = "PAIR_FEATURE_WIDTH";
    }

    /// <summary>The frozen common-2D and DTW constants. None of these is tunable.</summary>
    public static class H3Constants
    {
        public const int LandmarkCount = 16;
        public const int CoordDims = 3;
        public const double MissingCost = 1.0;
        public const double DtwBand = 0.10;
        public const double DtwBandEps = 1e-15;
        public const int ResampleFrames = 32;
        public const double ZTolerance = 0.0;
        public const int MinValidFrames = 4;
        public const double MinLandmarkCoverage = 0.10;
        public const int PairFeatureCount = 36;
        public const double ResidualQuantile = 0.90;

        /// <summary>
        /// The 16 selected points are 4 pose points (left/right elbow and wrist) plus 6
        /// per hand (wrist and the five finger tips). That is partial hand geometry, not
        /// full finger-joint articulation and no face at all.
        /// </summary>
        public const string SelectionNote =
            "4 pose (elbows, wrists) + 6 per hand (wrist + 5 finger tips) = 16";
    }

    public sealed class QualityResult
    {
        public bool Ok { get; private set; }
        public string Code { get; private set; }
        public double Coverage { get; private set; }
        public int ValidFrames { get; private set; }

        public QualityResult(bool ok, string code, double coverage, int validFrames)
        {
            Ok = ok; Code = code; Coverage = coverage; ValidFrames = validFrames;
        }
    }

    public sealed class AlignmentResult
    {
        public bool Ok { get; private set; }
        public string Failure { get; private set; }
        public double Distance { get; private set; }
        public int[] PathA { get; private set; }
        public int[] PathB { get; private set; }
        public double JointValidFraction { get; private set; }

        public int PathLength { get { return PathA == null ? 0 : PathA.Length; } }

        AlignmentResult(bool ok, string failure) { Ok = ok; Failure = failure; }

        public AlignmentResult(double distance, int[] pathA, int[] pathB,
                               double jointValidFraction)
        {
            Ok = true; Distance = distance; PathA = pathA; PathB = pathB;
            JointValidFraction = jointValidFraction;
        }

        public static AlignmentResult Failed(string code)
        {
            return new AlignmentResult(false, code);
        }
    }

    public sealed class PairFeatureResult
    {
        public bool Ok { get; private set; }
        public string Failure { get; private set; }
        public double[] Features { get; private set; }
        public int PathLength { get; private set; }
        public int NonDiagonalSteps { get; private set; }

        PairFeatureResult(bool ok, string failure) { Ok = ok; Failure = failure; }

        public PairFeatureResult(double[] features, int pathLength, int nonDiagonalSteps)
        {
            Ok = true; Features = features; PathLength = pathLength;
            NonDiagonalSteps = nonDiagonalSteps;
        }

        public static PairFeatureResult Failed(string code)
        {
            return new PairFeatureResult(false, code);
        }
    }

    public static class H3PairGeometry
    {
        // ---- NumPy-faithful reductions ------------------------------------

        const int PairwiseBlockSize = 128;   // NumPy's PW_BLOCKSIZE

        /// <summary>Reproduces NumPy's pairwise summation, which np.mean uses.</summary>
        public static double PairwiseSum(double[] values, int start, int count)
        {
            if (count < 8)
            {
                double small = 0.0;
                for (int index = 0; index < count; index++) small += values[start + index];
                return small;
            }
            if (count <= PairwiseBlockSize)
            {
                double r0 = values[start], r1 = values[start + 1];
                double r2 = values[start + 2], r3 = values[start + 3];
                double r4 = values[start + 4], r5 = values[start + 5];
                double r6 = values[start + 6], r7 = values[start + 7];
                int i = 8;
                for (; i < count - (count % 8); i += 8)
                {
                    r0 += values[start + i]; r1 += values[start + i + 1];
                    r2 += values[start + i + 2]; r3 += values[start + i + 3];
                    r4 += values[start + i + 4]; r5 += values[start + i + 5];
                    r6 += values[start + i + 6]; r7 += values[start + i + 7];
                }
                double result = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
                for (; i < count; i++) result += values[start + i];
                return result;
            }
            int half = count / 2;
            half -= half % 8;
            return PairwiseSum(values, start, half)
                   + PairwiseSum(values, start + half, count - half);
        }

        public static double Mean(double[] values, int start, int count)
        {
            return PairwiseSum(values, start, count) / count;
        }

        /// <summary>NumPy's two-branch lerp, used by percentile's 'linear' method.</summary>
        public static double Lerp(double a, double b, double t)
        {
            double difference = b - a;
            double result = a + difference * t;
            if (t >= 0.5) result = b - difference * (1.0 - t);
            return result;
        }

        /// <summary>np.percentile(values, 90) with the default linear interpolation.</summary>
        public static double LinearPercentile90(double[] values, int count)
        {
            var sorted = new double[count];
            Array.Copy(values, 0, sorted, 0, count);
            Array.Sort(sorted);
            double virtualIndex = H3Constants.ResidualQuantile * (count - 1);
            int previous = (int)Math.Floor(virtualIndex);
            if (previous < 0) previous = 0;
            int next = previous + 1;
            if (next > count - 1) next = count - 1;
            double gamma = virtualIndex - previous;
            return Lerp(sorted[previous], sorted[next], gamma);
        }

        // ---- quality -------------------------------------------------------

        public static QualityResult CheckInputQuality(double[] coords, bool[] mask,
                                                      int frameCount)
        {
            if (coords == null || mask == null)
                return new QualityResult(false, H3FailureCodes.UnconvertibleInput, 0.0, 0);
            if (frameCount <= 0)
                return new QualityResult(false, H3FailureCodes.EmptySequence, 0.0, 0);
            int landmarks = H3Constants.LandmarkCount;
            if (coords.Length != frameCount * landmarks * H3Constants.CoordDims)
                return new QualityResult(false, H3FailureCodes.MalformedShape, 0.0, 0);
            if (mask.Length != frameCount * landmarks)
                return new QualityResult(false, H3FailureCodes.MaskShapeMismatch, 0.0, 0);

            int validLandmarkSlots = 0;
            int framesWithAnyValid = 0;
            for (int frame = 0; frame < frameCount; frame++)
            {
                bool anyInFrame = false;
                for (int landmark = 0; landmark < landmarks; landmark++)
                {
                    if (!mask[frame * landmarks + landmark]) continue;
                    anyInFrame = true;
                    validLandmarkSlots++;
                    int at = (frame * landmarks + landmark) * H3Constants.CoordDims;
                    for (int channel = 0; channel < H3Constants.CoordDims; channel++)
                    {
                        double value = coords[at + channel];
                        if (double.IsNaN(value) || double.IsInfinity(value))
                            return new QualityResult(false,
                                H3FailureCodes.NonfiniteValidCoord, 0.0, 0);
                    }
                    // the common-2D contract fixes z at exactly zero
                    if (Math.Abs(coords[at + 2]) > H3Constants.ZTolerance)
                        return new QualityResult(false, H3FailureCodes.NonzeroZ, 0.0, 0);
                }
                if (anyInFrame) framesWithAnyValid++;
            }
            if (validLandmarkSlots == 0)
                return new QualityResult(false, H3FailureCodes.AllLandmarksMissing, 0.0, 0);
            double coverage = validLandmarkSlots / (double)(frameCount * landmarks);
            if (framesWithAnyValid < H3Constants.MinValidFrames)
                return new QualityResult(false, H3FailureCodes.TooFewValidFrames,
                                         coverage, framesWithAnyValid);
            if (coverage < H3Constants.MinLandmarkCoverage)
                return new QualityResult(false, H3FailureCodes.CoverageTooLow,
                                         coverage, framesWithAnyValid);
            return new QualityResult(true, H3FailureCodes.Ok, coverage, framesWithAnyValid);
        }

        // ---- resampler -----------------------------------------------------

        /// <summary>
        /// The frozen 32-frame mask-aware resampler. A target frame is valid when EITHER
        /// endpoint is valid; with only one valid endpoint that endpoint's value is copied
        /// rather than interpolated. Invalid targets are written as exactly 0.0.
        /// </summary>
        public static void ResampleToFixed(double[] coords, bool[] mask, int frameCount,
                                           out double[] outCoords, out bool[] outMask)
        {
            int landmarks = H3Constants.LandmarkCount;
            int dims = H3Constants.CoordDims;
            int target = H3Constants.ResampleFrames;
            outCoords = new double[target * landmarks * dims];
            outMask = new bool[target * landmarks];
            int denominator = Math.Max(target - 1, 1);
            for (int t = 0; t < target; t++)
            {
                double tf = t * (double)(frameCount - 1) / denominator;
                int low = (int)Math.Floor(tf);
                int high = Math.Min(low + 1, frameCount - 1);
                double alpha = tf - low;
                for (int landmark = 0; landmark < landmarks; landmark++)
                {
                    bool validLow = mask[low * landmarks + landmark];
                    bool validHigh = mask[high * landmarks + landmark];
                    int destination = (t * landmarks + landmark) * dims;
                    int sourceLow = (low * landmarks + landmark) * dims;
                    int sourceHigh = (high * landmarks + landmark) * dims;
                    if (validLow && validHigh)
                    {
                        for (int channel = 0; channel < dims; channel++)
                            outCoords[destination + channel] =
                                (1 - alpha) * coords[sourceLow + channel]
                                + alpha * coords[sourceHigh + channel];
                        outMask[t * landmarks + landmark] = true;
                    }
                    else if (validLow)
                    {
                        for (int channel = 0; channel < dims; channel++)
                            outCoords[destination + channel] = coords[sourceLow + channel];
                        outMask[t * landmarks + landmark] = true;
                    }
                    else if (validHigh)
                    {
                        for (int channel = 0; channel < dims; channel++)
                            outCoords[destination + channel] = coords[sourceHigh + channel];
                        outMask[t * landmarks + landmark] = true;
                    }
                    // else the slot stays exactly 0.0 and invalid
                }
            }
        }

        // ---- frozen DTW ----------------------------------------------------

        /// <summary>
        /// The frozen local cost: mean over INCLUDED landmarks, where a jointly valid
        /// landmark contributes its Euclidean distance over all three channels and a
        /// landmark valid on exactly one side contributes MISSING_COST.
        /// </summary>
        public static double LocalCost(double[] a, bool[] maskA, int frameA,
                                       double[] b, bool[] maskB, int frameB)
        {
            int landmarks = H3Constants.LandmarkCount;
            int dims = H3Constants.CoordDims;
            double total = 0.0;
            int included = 0;
            for (int landmark = 0; landmark < landmarks; landmark++)
            {
                bool validA = maskA[frameA * landmarks + landmark];
                bool validB = maskB[frameB * landmarks + landmark];
                if (validA && validB)
                {
                    double sum = 0.0;
                    int atA = (frameA * landmarks + landmark) * dims;
                    int atB = (frameB * landmarks + landmark) * dims;
                    for (int channel = 0; channel < dims; channel++)
                    {
                        double delta = a[atA + channel] - b[atB + channel];
                        sum += delta * delta;
                    }
                    total += Math.Sqrt(sum);
                    included++;
                }
                else if (validA || validB)
                {
                    total += H3Constants.MissingCost;
                    included++;
                }
            }
            return included > 0 ? total / included : 1.0;
        }

        /// <summary>Mask-aware banded DTW with the frozen band, steps and tie rules.</summary>
        public static AlignmentResult Align(double[] a, bool[] maskA, int framesA,
                                            double[] b, bool[] maskB, int framesB)
        {
            if (framesA == 0 || framesB == 0)
                return AlignmentResult.Failed(H3FailureCodes.EmptyAlignSequence);
            int landmarks = H3Constants.LandmarkCount;
            int width = framesB + 1;
            var cost = new double[(framesA + 1) * width];
            for (int index = 0; index < cost.Length; index++)
                cost[index] = double.PositiveInfinity;
            cost[0] = 0.0;

            double spanA = Math.Max(framesA - 1, 1);
            double spanB = Math.Max(framesB - 1, 1);
            for (int i = 1; i <= framesA; i++)
            {
                double ri = (i - 1) / spanA;
                for (int j = 1; j <= framesB; j++)
                {
                    double rj = (j - 1) / spanB;
                    if (Math.Abs(ri - rj) > H3Constants.DtwBand + H3Constants.DtwBandEps)
                        continue;
                    double local = LocalCost(a, maskA, i - 1, b, maskB, j - 1);
                    double best = cost[(i - 1) * width + (j - 1)];
                    double vertical = cost[(i - 1) * width + j];
                    if (vertical < best) best = vertical;
                    double horizontal = cost[i * width + (j - 1)];
                    if (horizontal < best) best = horizontal;
                    cost[i * width + j] = local + best;
                }
            }

            double terminal = cost[framesA * width + framesB];
            if (double.IsNaN(terminal) || double.IsInfinity(terminal))
                return AlignmentResult.Failed(H3FailureCodes.NoValidPath);

            var pathA = new List<int>();
            var pathB = new List<int>();
            var jointFractions = new List<double>();
            bool anyJointStep = false;
            int row = framesA, column = framesB;
            while (row > 0 && column > 0)
            {
                pathA.Add(row - 1);
                pathB.Add(column - 1);
                int joint = 0;
                for (int landmark = 0; landmark < landmarks; landmark++)
                    if (maskA[(row - 1) * landmarks + landmark]
                        && maskB[(column - 1) * landmarks + landmark]) joint++;
                if (joint > 0) anyJointStep = true;
                jointFractions.Add(joint / (double)landmarks);

                double diagonal = cost[(row - 1) * width + (column - 1)];
                double vertical = cost[(row - 1) * width + column];
                double horizontal = cost[row * width + (column - 1)];
                double best = diagonal;
                int nextRow = row - 1, nextColumn = column - 1;
                if (vertical < best) { best = vertical; nextRow = row - 1; nextColumn = column; }
                if (horizontal < best) { nextRow = row; nextColumn = column - 1; }
                row = nextRow; column = nextColumn;
            }
            pathA.Reverse();
            pathB.Reverse();
            jointFractions.Reverse();

            // the Python original fails here when no path step had a jointly valid
            // landmark, because its joint residual list would be empty
            if (!anyJointStep)
                return AlignmentResult.Failed(H3FailureCodes.NoJointlyValidLandmark);

            int pathLength = Math.Max(pathA.Count, 1);
            double[] fractions = jointFractions.ToArray();
            return new AlignmentResult(terminal / pathLength, pathA.ToArray(),
                                       pathB.ToArray(),
                                       Mean(fractions, 0, fractions.Length));
        }

        // ---- the 36 pinned pair features -----------------------------------

        public static string[] FeatureNames()
        {
            var names = new List<string>(H3Constants.PairFeatureCount);
            names.Add("dtw_normalized_cost");
            for (int landmark = 0; landmark < H3Constants.LandmarkCount; landmark++)
                names.Add("residual_mean_lm" + landmark.ToString("00"));
            for (int landmark = 0; landmark < H3Constants.LandmarkCount; landmark++)
                names.Add("residual_p90_lm" + landmark.ToString("00"));
            names.Add("path_length_ratio");
            names.Add("non_diagonal_step_fraction");
            names.Add("joint_valid_fraction");
            return names.ToArray();
        }

        public static PairFeatureResult FeaturesFromAlignment(
            AlignmentResult alignment, double[] a, bool[] maskA, int framesA,
            double[] b, bool[] maskB, int framesB)
        {
            if (alignment == null || !alignment.Ok)
                return PairFeatureResult.Failed(
                    alignment == null ? "ALIGNMENT_FAILED"
                                      : (alignment.Failure ?? "ALIGNMENT_FAILED"));
            int steps = alignment.PathLength;
            if (steps < 2) return PairFeatureResult.Failed(H3FailureCodes.DegeneratePath);

            int landmarks = H3Constants.LandmarkCount;
            int dims = H3Constants.CoordDims;
            var residuals = new double[landmarks][];
            var counts = new int[landmarks];
            for (int landmark = 0; landmark < landmarks; landmark++)
                residuals[landmark] = new double[steps];

            for (int step = 0; step < steps; step++)
            {
                int frameA = alignment.PathA[step];
                int frameB = alignment.PathB[step];
                for (int landmark = 0; landmark < landmarks; landmark++)
                {
                    if (!maskA[frameA * landmarks + landmark]) continue;
                    if (!maskB[frameB * landmarks + landmark]) continue;
                    int atA = (frameA * landmarks + landmark) * dims;
                    int atB = (frameB * landmarks + landmark) * dims;
                    // xy only, matching the frozen feature definition
                    double dx = a[atA] - b[atB];
                    double dy = a[atA + 1] - b[atB + 1];
                    residuals[landmark][counts[landmark]++] = Math.Sqrt(dx * dx + dy * dy);
                }
            }

            for (int landmark = 0; landmark < landmarks; landmark++)
                if (counts[landmark] == 0)
                    return PairFeatureResult.Failed(H3FailureCodes.MissingLandmarkOverlap);

            var features = new double[H3Constants.PairFeatureCount];
            features[0] = alignment.Distance;
            for (int landmark = 0; landmark < landmarks; landmark++)
                features[1 + landmark] = Mean(residuals[landmark], 0, counts[landmark]);
            for (int landmark = 0; landmark < landmarks; landmark++)
                features[17 + landmark] =
                    LinearPercentile90(residuals[landmark], counts[landmark]);

            int nonDiagonal = 0;
            for (int step = 1; step < steps; step++)
            {
                bool diagonal = alignment.PathA[step] - alignment.PathA[step - 1] == 1
                                && alignment.PathB[step] - alignment.PathB[step - 1] == 1;
                if (!diagonal) nonDiagonal++;
            }
            features[33] = steps / (double)Math.Max(framesA, framesB);
            features[34] = nonDiagonal / (double)(steps - 1);
            features[35] = alignment.JointValidFraction;

            for (int index = 0; index < features.Length; index++)
                if (double.IsNaN(features[index]) || double.IsInfinity(features[index]))
                    return PairFeatureResult.Failed(H3FailureCodes.NonfinitePairFeature);
            return new PairFeatureResult(features, steps, nonDiagonal);
        }

        /// <summary>Aligns with the frozen DTW, then measures along that path.</summary>
        public static PairFeatureResult PairFeatures(double[] a, bool[] maskA, int framesA,
                                                     double[] b, bool[] maskB, int framesB)
        {
            AlignmentResult alignment = Align(a, maskA, framesA, b, maskB, framesB);
            return FeaturesFromAlignment(alignment, a, maskA, framesA, b, maskB, framesB);
        }
    }
}
