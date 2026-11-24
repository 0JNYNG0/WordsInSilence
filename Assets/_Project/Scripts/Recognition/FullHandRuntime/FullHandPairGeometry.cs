// F4 - standalone C# port of the verified F3R1 full-hand inference math.
// NOT wired to any observer, call site, scene, builder or camera. No promotion.
//
// 86 points = global 46 (F1 body-normalized) + left local 20 + right local 20, five equal
// groups, one shared time axis. Local blocks are wrist-relative and divided by a
// per-capture median wrist->MCP scalar; no rotation, no smoothing, no trim, no learned
// personal profile. 2D only: a valid point's channel 2 is structurally 0.
using System;
using System.Collections.Generic;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Recognition.FullHandRuntime
{
    /// <summary>A refused input or bundle. Never substituted with a usable value.</summary>
    public sealed class FullHandFormatException : Exception
    {
        public string Code { get; private set; }

        public FullHandFormatException(string code, string message)
            : base(code + ": " + message)
        {
            Code = code;
        }
    }

    public static class FullHandFailureCodes
    {
        public const string AllPointsMissing = "ALL_POINTS_MISSING";
        public const string NoLocalHandShape = "NO_LOCAL_HAND_SHAPE";
        public const string NoSharedHandShape = "NO_SHARED_HAND_SHAPE";
        public const string NoValidDtwPath = "NO_VALID_DTW_PATH";
        public const string MalformedShape = "MALFORMED_SHAPE";
        public const string NonFiniteValidCoordinate = "NONFINITE_VALID_COORDINATE";
        public const string NonZeroZ = "NONZERO_Z_IN_MASKED_2D";
        public const string SchemaMismatch = "SCHEMA_MISMATCH";
        public const string NonFiniteLogit = "NONFINITE_LOGIT";
    }

    public static class FullHandStatus
    {
        public const string Ranked = "RANKED";
        public const string TechnicalFailure = "TECHNICAL_FAILURE";
        public const string NoReferenceForClass = "NO_REFERENCE_FOR_CLASS";
        public const string NoUsableReferencePair = "NO_USABLE_REFERENCE_PAIR";
    }

    public static class FullHandPairConstants
    {
        public const string PairSchemaId = "wis_full_hand_pair_v1";
        public const string SourceSchemaId = "wis_full_hand_common_v1";
        public const int SourcePointCount = 46;
        public const int FeaturePointCount = 86;
        public const int CoordDims = 3;
        public const int HandJointCount = 21;
        public const int ResampleFrames = 32;
        public const double MinHandScale = 1e-6;
        public const double DtwBand = 0.10;
        public const double DtwBandEps = 1e-15;
        public const double ResidualQuantile = 0.90;
        public const int GroupCount = 5;
        public const int FixedFeatureCount = 15;
        public const int AlignedFeatureCount = 18;

        public static readonly int[] GroupStart = { 0, 4, 25, 46, 66 };
        public static readonly int[] GroupEnd = { 4, 25, 46, 66, 86 };
        public static readonly string[] GroupNames =
            { "global_pose", "global_left_hand", "global_right_hand",
              "local_left_hand", "local_right_hand" };
        public static readonly int[] LocalGroupIndices = { 3, 4 };
        public static readonly int[] HandMcpJoints = { 5, 9, 13, 17 };
        public const int LeftSourceBase = 4;
        public const int RightSourceBase = 25;
        public const int LeftLocalStart = 46;
        public const int RightLocalStart = 66;

        public static string[] DescriptorNames()
        {
            var names = new List<string>(AlignedFeatureCount);
            for (int group = 0; group < GroupCount; group++)
            {
                names.Add(GroupNames[group] + "_residual_mean");
                names.Add(GroupNames[group] + "_residual_p90");
                names.Add(GroupNames[group] + "_joint_valid_fraction");
            }
            names.Add("dtw_mean_path_cost");
            names.Add("dtw_path_length_ratio");
            names.Add("dtw_non_diagonal_fraction");
            return names.ToArray();
        }
    }

    /// <summary>Immutable 86-point feature sequence at T=32. Masked slots hold NaN.</summary>
    public sealed class FullHandFeatureSequence
    {
        readonly double[] points;      // [32 * 86 * 3]
        readonly bool[] mask;          // [32 * 86]
        readonly bool[] groupAvailable;

        public string Schema { get { return FullHandPairConstants.PairSchemaId; } }
        public int SourceFrames { get; private set; }
        public double LeftHandScale { get; private set; }
        public double RightHandScale { get; private set; }
        public bool Usable { get; private set; }
        public string Failure { get; private set; }

        internal FullHandFeatureSequence(double[] points, bool[] mask, bool[] groupAvailable,
            double leftScale, double rightScale, int sourceFrames, bool usable, string failure)
        {
            this.points = points;
            this.mask = mask;
            this.groupAvailable = groupAvailable;
            LeftHandScale = leftScale;
            RightHandScale = rightScale;
            SourceFrames = sourceFrames;
            Usable = usable;
            Failure = failure;
        }

        public double[] CopyPoints() { return (double[])points.Clone(); }
        public bool[] CopyMask() { return (bool[])mask.Clone(); }
        public bool[] CopyGroupAvailable() { return (bool[])groupAvailable.Clone(); }
        public bool GroupAvailable(int group) { return groupAvailable[group]; }

        public bool IsValid(int frame, int point)
        {
            return mask[frame * FullHandPairConstants.FeaturePointCount + point];
        }

        public double Coordinate(int frame, int point, int axis)
        {
            return points[(frame * FullHandPairConstants.FeaturePointCount + point)
                          * FullHandPairConstants.CoordDims + axis];
        }

        internal double[] PointsReference { get { return points; } }
        internal bool[] MaskReference { get { return mask; } }
    }

    public sealed class FullHandPairResult
    {
        public bool Ok { get; private set; }
        public string Failure { get; private set; }
        public double DtwCost { get; private set; }
        public int PathLength { get; private set; }
        readonly double[] features;

        FullHandPairResult(bool ok, string failure, double[] features, double cost, int length)
        {
            Ok = ok; Failure = failure; this.features = features;
            DtwCost = cost; PathLength = length;
        }

        public static FullHandPairResult Failed(string code)
        {
            return new FullHandPairResult(false, code, null, double.NaN, 0);
        }

        public static FullHandPairResult Succeeded(double[] features, double cost, int length)
        {
            return new FullHandPairResult(true, null, features, cost, length);
        }

        public double[] CopyFeatures() { return features == null ? null : (double[])features.Clone(); }
        internal double[] FeaturesReference { get { return features; } }
    }

    /// <summary>Per-pair distances, joint validity and the grouped cost matrix.</summary>
    public sealed class FullHandPairGeometryData
    {
        public readonly double[] Distances;   // [32 * 32 * 86]
        public readonly bool[] Joint;         // [32 * 32 * 86]
        public readonly double[] Cost;        // [32 * 32], +inf where no group overlaps
        public readonly int[] ContributingGroups;

        internal FullHandPairGeometryData(double[] distances, bool[] joint, double[] cost,
                                          int[] contributing)
        {
            Distances = distances; Joint = joint; Cost = cost; ContributingGroups = contributing;
        }
    }

    public static class FullHandPairGeometry
    {
        // ---- NumPy-faithful reductions, mirroring the verified Python ----------

        const int PairwiseBlockSize = 128;

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

        public static double Lerp(double a, double b, double t)
        {
            double difference = b - a;
            double result = a + difference * t;
            if (t >= 0.5) result = b - difference * (1.0 - t);
            return result;
        }

        /// <summary>np.percentile(values, 90), default linear interpolation.</summary>
        public static double LinearPercentile90(double[] values, int count)
        {
            var sorted = new double[count];
            Array.Copy(values, 0, sorted, 0, count);
            Array.Sort(sorted);
            double virtualIndex = FullHandPairConstants.ResidualQuantile * (count - 1);
            int previous = (int)Math.Floor(virtualIndex);
            if (previous < 0) previous = 0;
            int next = previous + 1;
            if (next > count - 1) next = count - 1;
            return Lerp(sorted[previous], sorted[next], virtualIndex - previous);
        }

        /// <summary>np.median: sorted middle, or the mean of the two middles.</summary>
        public static double Median(List<double> values)
        {
            var sorted = values.ToArray();
            Array.Sort(sorted);
            int count = sorted.Length;
            if (count == 0) return double.NaN;
            int middle = count / 2;
            if (count % 2 == 1) return sorted[middle];
            return (sorted[middle - 1] + sorted[middle]) / 2.0;
        }

        // ---- representation ----------------------------------------------------

        public static FullHandFeatureSequence BuildFeatures(FullHandCommonCapture capture)
        {
            if (capture == null) throw new ArgumentNullException("capture");
            if (!string.Equals(capture.SchemaId, FullHandPairConstants.SourceSchemaId,
                               StringComparison.Ordinal))
                throw new FullHandFormatException(FullHandFailureCodes.SchemaMismatch,
                    "the capture is not " + FullHandPairConstants.SourceSchemaId);
            return BuildFeatures(capture.CopyCoordinates(), capture.CopyValidMask(),
                                 capture.FrameCount);
        }

        /// <summary>Pure array overload. The caller's arrays are read, never retained.</summary>
        public static FullHandFeatureSequence BuildFeatures(double[] coordinates, bool[] mask,
                                                            int frameCount)
        {
            int source = FullHandPairConstants.SourcePointCount;
            int dims = FullHandPairConstants.CoordDims;
            if (coordinates == null || mask == null)
                throw new ArgumentNullException("coordinates");
            if (frameCount <= 0)
                throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                    "an empty sequence carries no observation");
            if (coordinates.LongLength != (long)frameCount * source * dims ||
                mask.LongLength != (long)frameCount * source)
                throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                    "coordinates must be [T, 46, 3] with a matching [T, 46] mask");
            for (int slot = 0; slot < mask.Length; slot++)
            {
                if (!mask[slot]) continue;
                double x = coordinates[slot * dims];
                double y = coordinates[slot * dims + 1];
                double z = coordinates[slot * dims + 2];
                if (double.IsNaN(x) || double.IsInfinity(x) ||
                    double.IsNaN(y) || double.IsInfinity(y))
                    throw new FullHandFormatException(
                        FullHandFailureCodes.NonFiniteValidCoordinate,
                        "a valid point has a non-finite xy");
                if (z != 0.0)
                    throw new FullHandFormatException(FullHandFailureCodes.NonZeroZ,
                        "a valid point has a non-zero z; this contract is masked 2D");
            }

            int points = FullHandPairConstants.FeaturePointCount;
            var built = new double[(long)frameCount * points * dims];
            var builtMask = new bool[(long)frameCount * points];
            for (int index = 0; index < built.Length; index++) built[index] = double.NaN;
            for (int frame = 0; frame < frameCount; frame++)
                for (int point = 0; point < source; point++)
                {
                    int from = frame * source + point;
                    int to = frame * points + point;
                    builtMask[to] = mask[from];
                    if (!mask[from]) continue;
                    for (int axis = 0; axis < dims; axis++)
                        built[to * dims + axis] = coordinates[from * dims + axis];
                }

            double leftScale = HandScale(coordinates, mask, frameCount,
                                         FullHandPairConstants.LeftSourceBase);
            double rightScale = HandScale(coordinates, mask, frameCount,
                                          FullHandPairConstants.RightSourceBase);
            WriteLocalBlock(coordinates, mask, frameCount, FullHandPairConstants.LeftSourceBase,
                            FullHandPairConstants.LeftLocalStart, leftScale, built, builtMask);
            WriteLocalBlock(coordinates, mask, frameCount, FullHandPairConstants.RightSourceBase,
                            FullHandPairConstants.RightLocalStart, rightScale, built, builtMask);

            double[] resampled;
            bool[] resampledMask;
            ResampleToFixed(built, builtMask, frameCount, points, out resampled, out resampledMask);

            var available = new bool[FullHandPairConstants.GroupCount];
            bool anyAvailable = false;
            for (int group = 0; group < FullHandPairConstants.GroupCount; group++)
            {
                for (int frame = 0; frame < FullHandPairConstants.ResampleFrames && !available[group]; frame++)
                    for (int point = FullHandPairConstants.GroupStart[group];
                         point < FullHandPairConstants.GroupEnd[group]; point++)
                        if (resampledMask[frame * points + point]) { available[group] = true; break; }
                anyAvailable |= available[group];
            }
            string failure = null;
            if (!anyAvailable) failure = FullHandFailureCodes.AllPointsMissing;
            else if (!available[3] && !available[4]) failure = FullHandFailureCodes.NoLocalHandShape;
            return new FullHandFeatureSequence(resampled, resampledMask, available,
                                               leftScale, rightScale, frameCount,
                                               failure == null, failure);
        }

        /// <summary>Median of the valid positive wrist->MCP distances over source frames.</summary>
        public static double HandScale(double[] coordinates, bool[] mask, int frameCount, int base_)
        {
            int source = FullHandPairConstants.SourcePointCount;
            int dims = FullHandPairConstants.CoordDims;
            var distances = new List<double>();
            for (int joint = 0; joint < FullHandPairConstants.HandMcpJoints.Length; joint++)
            {
                int mcp = base_ + FullHandPairConstants.HandMcpJoints[joint];
                for (int frame = 0; frame < frameCount; frame++)
                {
                    if (!mask[frame * source + base_] || !mask[frame * source + mcp]) continue;
                    double dx = coordinates[(frame * source + mcp) * dims]
                              - coordinates[(frame * source + base_) * dims];
                    double dy = coordinates[(frame * source + mcp) * dims + 1]
                              - coordinates[(frame * source + base_) * dims + 1];
                    double value = Math.Sqrt(dx * dx + dy * dy);
                    if (double.IsNaN(value) || double.IsInfinity(value)) continue;
                    if (value > FullHandPairConstants.MinHandScale) distances.Add(value);
                }
            }
            return distances.Count == 0 ? double.NaN : Median(distances);
        }

        static void WriteLocalBlock(double[] coordinates, bool[] mask, int frameCount,
                                    int base_, int localStart, double scale,
                                    double[] built, bool[] builtMask)
        {
            if (double.IsNaN(scale) || double.IsInfinity(scale) ||
                scale <= FullHandPairConstants.MinHandScale)
                return;                    // the shape group stays missing; nothing invented
            int source = FullHandPairConstants.SourcePointCount;
            int points = FullHandPairConstants.FeaturePointCount;
            int dims = FullHandPairConstants.CoordDims;
            for (int joint = 1; joint < FullHandPairConstants.HandJointCount; joint++)
            {
                int target = localStart + joint - 1;
                for (int frame = 0; frame < frameCount; frame++)
                {
                    int wrist = frame * source + base_;
                    int from = frame * source + base_ + joint;
                    if (!mask[wrist] || !mask[from]) continue;
                    int to = frame * points + target;
                    for (int axis = 0; axis < dims; axis++)
                        built[to * dims + axis] =
                            (coordinates[from * dims + axis] - coordinates[wrist * dims + axis]) / scale;
                    builtMask[to] = true;
                }
            }
        }

        /// <summary>
        /// The approved mask-aware linear rule: both endpoints valid interpolates, a SINGLE
        /// valid endpoint carries, neither stays missing. Masked slots hold NaN.
        /// </summary>
        public static void ResampleToFixed(double[] points, bool[] mask, int sourceFrames,
                                           int pointCount, out double[] outPoints,
                                           out bool[] outMask)
        {
            int target = FullHandPairConstants.ResampleFrames;
            int dims = FullHandPairConstants.CoordDims;
            outPoints = new double[target * pointCount * dims];
            outMask = new bool[target * pointCount];
            for (int index = 0; index < outPoints.Length; index++) outPoints[index] = double.NaN;
            int denominator = target - 1;
            if (denominator < 1) denominator = 1;
            for (int t = 0; t < target; t++)
            {
                double position = t * (double)(sourceFrames - 1) / denominator;
                int low = (int)Math.Floor(position);
                int high = low + 1;
                if (high > sourceFrames - 1) high = sourceFrames - 1;
                double alpha = position - low;
                for (int point = 0; point < pointCount; point++)
                {
                    bool validLow = mask[low * pointCount + point];
                    bool validHigh = mask[high * pointCount + point];
                    int to = (t * pointCount + point) * dims;
                    int fromLow = (low * pointCount + point) * dims;
                    int fromHigh = (high * pointCount + point) * dims;
                    if (validLow && validHigh)
                        for (int axis = 0; axis < dims; axis++)
                            outPoints[to + axis] = (1.0 - alpha) * points[fromLow + axis]
                                                 + alpha * points[fromHigh + axis];
                    else if (validLow)
                        for (int axis = 0; axis < dims; axis++)
                            outPoints[to + axis] = points[fromLow + axis];
                    else if (validHigh)
                        for (int axis = 0; axis < dims; axis++)
                            outPoints[to + axis] = points[fromHigh + axis];
                    outMask[t * pointCount + point] = validLow || validHigh;
                }
            }
        }

        // ---- pair geometry ------------------------------------------------------

        public static FullHandPairGeometryData ComputeGeometry(FullHandFeatureSequence a,
                                                               FullHandFeatureSequence b)
        {
            int frames = FullHandPairConstants.ResampleFrames;
            int points = FullHandPairConstants.FeaturePointCount;
            int dims = FullHandPairConstants.CoordDims;
            double[] pa = a.PointsReference, pb = b.PointsReference;
            bool[] ma = a.MaskReference, mb = b.MaskReference;

            var distances = new double[frames * frames * points];
            var joint = new bool[frames * frames * points];
            var cost = new double[frames * frames];
            var contributing = new int[frames * frames];
            var groupMeans = new double[FullHandPairConstants.GroupCount];
            var buffer = new double[points];

            for (int i = 0; i < frames; i++)
                for (int j = 0; j < frames; j++)
                {
                    int cell = (i * frames + j) * points;
                    for (int point = 0; point < points; point++)
                    {
                        bool together = ma[i * points + point] && mb[j * points + point];
                        joint[cell + point] = together;
                        if (!together) { distances[cell + point] = double.NaN; continue; }
                        double dx = pa[(i * points + point) * dims] - pb[(j * points + point) * dims];
                        double dy = pa[(i * points + point) * dims + 1] - pb[(j * points + point) * dims + 1];
                        distances[cell + point] = Math.Sqrt(dx * dx + dy * dy);
                    }
                    int available = 0;
                    for (int group = 0; group < FullHandPairConstants.GroupCount; group++)
                    {
                        int low = FullHandPairConstants.GroupStart[group];
                        int width = FullHandPairConstants.GroupEnd[group] - low;
                        int count = 0;
                        for (int offset = 0; offset < width; offset++)
                        {
                            bool together = joint[cell + low + offset];
                            buffer[offset] = together ? distances[cell + low + offset] : 0.0;
                            if (together) count++;
                        }
                        // the sum spans the WHOLE group with zeros, exactly as NumPy does
                        groupMeans[group] = count > 0
                            ? PairwiseSum(buffer, 0, width) / count : double.NaN;
                        if (count > 0) available++;
                    }
                    if (available == 0)
                    {
                        cost[i * frames + j] = double.PositiveInfinity;
                        contributing[i * frames + j] = 0;
                        continue;
                    }
                    double total = 0.0;                 // sequential, like a NumPy axis-0 sum
                    for (int group = 0; group < FullHandPairConstants.GroupCount; group++)
                        total += double.IsNaN(groupMeans[group]) ? 0.0 : groupMeans[group];
                    cost[i * frames + j] = total / available;
                    contributing[i * frames + j] = available;
                }
            return new FullHandPairGeometryData(distances, joint, cost, contributing);
        }

        public static void DiagonalPath(out int[] pathA, out int[] pathB)
        {
            int frames = FullHandPairConstants.ResampleFrames;
            pathA = new int[frames];
            pathB = new int[frames];
            for (int index = 0; index < frames; index++) { pathA[index] = index; pathB[index] = index; }
        }

        /// <summary>Banded DTW, pinned tie order: diagonal, then vertical, then horizontal.</summary>
        public static bool DtwPath(double[] cost, out int[] pathA, out int[] pathB,
                                   out double normalized)
        {
            int frames = FullHandPairConstants.ResampleFrames;
            pathA = null; pathB = null; normalized = double.NaN;
            var accumulated = new double[(frames + 1) * (frames + 1)];
            for (int index = 0; index < accumulated.Length; index++)
                accumulated[index] = double.PositiveInfinity;
            accumulated[0] = 0.0;
            int stride = frames + 1;
            for (int i = 1; i <= frames; i++)
            {
                double ri = (i - 1) / (double)Math.Max(frames - 1, 1);
                for (int j = 1; j <= frames; j++)
                {
                    double rj = (j - 1) / (double)Math.Max(frames - 1, 1);
                    if (Math.Abs(ri - rj) > FullHandPairConstants.DtwBand
                                            + FullHandPairConstants.DtwBandEps) continue;
                    double local = cost[(i - 1) * frames + (j - 1)];
                    if (double.IsNaN(local) || double.IsInfinity(local)) continue;
                    double best = accumulated[(i - 1) * stride + (j - 1)];
                    if (accumulated[(i - 1) * stride + j] < best)
                        best = accumulated[(i - 1) * stride + j];
                    if (accumulated[i * stride + (j - 1)] < best)
                        best = accumulated[i * stride + (j - 1)];
                    accumulated[i * stride + j] = local + best;
                }
            }
            double total = accumulated[frames * stride + frames];
            if (double.IsNaN(total) || double.IsInfinity(total)) return false;

            var reverseA = new List<int>();
            var reverseB = new List<int>();
            int row = frames, column = frames;
            while (row > 0 && column > 0)
            {
                reverseA.Add(row - 1);
                reverseB.Add(column - 1);
                double diagonal = accumulated[(row - 1) * stride + (column - 1)];
                double vertical = accumulated[(row - 1) * stride + column];
                double horizontal = accumulated[row * stride + (column - 1)];
                double best = diagonal;
                int nextRow = row - 1, nextColumn = column - 1;
                if (vertical < best) { best = vertical; nextRow = row - 1; nextColumn = column; }
                if (horizontal < best) { nextRow = row; nextColumn = column - 1; }
                row = nextRow; column = nextColumn;
            }
            reverseA.Reverse();
            reverseB.Reverse();
            pathA = reverseA.ToArray();
            pathB = reverseB.ToArray();
            normalized = total / Math.Max(pathA.Length, 1);
            return true;
        }

        public static int SharedLocalHandPoints(FullHandPairGeometryData geometry,
                                                int[] pathA, int[] pathB)
        {
            int frames = FullHandPairConstants.ResampleFrames;
            int points = FullHandPairConstants.FeaturePointCount;
            int total = 0;
            for (int index = 0; index < FullHandPairConstants.LocalGroupIndices.Length; index++)
            {
                int group = FullHandPairConstants.LocalGroupIndices[index];
                for (int step = 0; step < pathA.Length; step++)
                {
                    int cell = (pathA[step] * frames + pathB[step]) * points;
                    for (int point = FullHandPairConstants.GroupStart[group];
                         point < FullHandPairConstants.GroupEnd[group]; point++)
                        if (geometry.Joint[cell + point]) total++;
                }
            }
            return total;
        }

        /// <summary>15 values: per group residual mean, residual p90, joint valid fraction.</summary>
        public static double[] DescriptorsOnPath(FullHandPairGeometryData geometry,
                                                 int[] pathA, int[] pathB)
        {
            int frames = FullHandPairConstants.ResampleFrames;
            int points = FullHandPairConstants.FeaturePointCount;
            int length = pathA.Length;
            var values = new double[FullHandPairConstants.FixedFeatureCount];
            int at = 0;
            for (int group = 0; group < FullHandPairConstants.GroupCount; group++)
            {
                int low = FullHandPairConstants.GroupStart[group];
                int width = FullHandPairConstants.GroupEnd[group] - low;
                var pool = new double[length * width];
                int pooled = 0;
                for (int step = 0; step < length; step++)
                {
                    int cell = (pathA[step] * frames + pathB[step]) * points;
                    for (int offset = 0; offset < width; offset++)
                        if (geometry.Joint[cell + low + offset])
                            pool[pooled++] = geometry.Distances[cell + low + offset];
                }
                if (pooled > 0)
                {
                    values[at++] = Mean(pool, 0, pooled);
                    values[at++] = LinearPercentile90(pool, pooled);
                }
                else
                {
                    // unavailable, never reported as agreement
                    values[at++] = double.NaN;
                    values[at++] = double.NaN;
                }
                values[at++] = pooled / (double)(length * width);
            }
            return values;
        }

        public static double[] PathStatistics(FullHandPairGeometryData geometry,
                                              int[] pathA, int[] pathB)
        {
            int frames = FullHandPairConstants.ResampleFrames;
            int length = pathA.Length;
            var local = new double[length];
            for (int step = 0; step < length; step++)
                local[step] = geometry.Cost[pathA[step] * frames + pathB[step]];
            int nonDiagonal = 0;
            for (int step = 1; step < length; step++)
                if (!(pathA[step] - pathA[step - 1] == 1 && pathB[step] - pathB[step - 1] == 1))
                    nonDiagonal++;
            int denominator = Math.Max(length - 1, 1);
            return new[] { Mean(local, 0, length),
                           length / (double)FullHandPairConstants.ResampleFrames,
                           nonDiagonal / (double)denominator };
        }

        // ---- the three pair products -------------------------------------------

        public static FullHandPairResult FixedTimeFeatures(FullHandFeatureSequence a,
                                                           FullHandFeatureSequence b)
        {
            FullHandPairGeometryData geometry = ComputeGeometry(a, b);
            int[] pathA, pathB;
            DiagonalPath(out pathA, out pathB);
            if (SharedLocalHandPoints(geometry, pathA, pathB) < 1)
                return FullHandPairResult.Failed(FullHandFailureCodes.NoSharedHandShape);
            return FullHandPairResult.Succeeded(DescriptorsOnPath(geometry, pathA, pathB),
                                                double.NaN, pathA.Length);
        }

        public static FullHandPairResult AlignedPairFeatures(FullHandFeatureSequence a,
                                                             FullHandFeatureSequence b)
        {
            FullHandPairGeometryData geometry = ComputeGeometry(a, b);
            int[] pathA, pathB;
            double normalized;
            if (!DtwPath(geometry.Cost, out pathA, out pathB, out normalized) || pathA.Length < 2)
                return FullHandPairResult.Failed(FullHandFailureCodes.NoValidDtwPath);
            if (SharedLocalHandPoints(geometry, pathA, pathB) < 1)
                return FullHandPairResult.Failed(FullHandFailureCodes.NoSharedHandShape);
            double[] descriptors = DescriptorsOnPath(geometry, pathA, pathB);
            double[] statistics = PathStatistics(geometry, pathA, pathB);
            var features = new double[FullHandPairConstants.AlignedFeatureCount];
            Array.Copy(descriptors, 0, features, 0, descriptors.Length);
            Array.Copy(statistics, 0, features, descriptors.Length, statistics.Length);
            return FullHandPairResult.Succeeded(features, normalized, pathA.Length);
        }

        public static FullHandPairResult DtwDistance(FullHandFeatureSequence a,
                                                     FullHandFeatureSequence b)
        {
            FullHandPairGeometryData geometry = ComputeGeometry(a, b);
            int[] pathA, pathB;
            double normalized;
            if (!DtwPath(geometry.Cost, out pathA, out pathB, out normalized))
                return FullHandPairResult.Failed(FullHandFailureCodes.NoValidDtwPath);
            if (SharedLocalHandPoints(geometry, pathA, pathB) < 1)
                return FullHandPairResult.Failed(FullHandFailureCodes.NoSharedHandShape);
            return FullHandPairResult.Succeeded(null, normalized, pathA.Length);
        }
    }
}
