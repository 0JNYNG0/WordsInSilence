// F4.4 RESEARCH CANDIDATE - staged, opt-in only. Not performance-approved, not promoted.
//
// Exact C# port of three reviewed offline operations:
//   temporal_pattern_features.build_temporal_features  -> BuildTemporal      (32 x 198)
//   phase_learning.phase_pair_features                 -> PhaseVector        (432)
//   temporal_alignment_candidate.constrained_dtw_path  -> ConstrainedPath
// Nothing else is ported: no fit, no predict, no resampling, no smoothing, no thresholds,
// no word branches, no gates and no state. Invalid geometry stays NaN with an explicit mask.
using System;
using System.Collections.Generic;

namespace WordsInSilence.Recognition.FullHandRuntime
{
    public static class FullHandPhasePairFeatures
    {
        public const int Frames = 32;
        public const int Channels = 198;
        public const int PointCount = 86;
        public const int FeatureCount = 432;      // 8 bins * 18 groups * 3 stats
        public const int Bins = 8;
        public const int BinSize = 4;
        public const int Stats = 3;
        public const string TemporalSchemaId = "wis_temporal_pattern_2d_v1";
        public const string PhaseSchemaId = "phase_pair_model_v1";
        public const int MaxConsecutiveNonDiagonal = 2;

        const double Eps = 1e-8;
        const int PointEnd = 2 * PointCount;      // 172
        const int LeftBase = 4;
        const int RightBase = 25;
        const int WristJoint = 0;
        const int MiddleMcpJoint = 9;

        // thumb, index, middle, ring, pinky chains in source joint numbering
        static readonly int[][] Fingers =
        {
            new[] { 1, 2, 3, 4 }, new[] { 5, 6, 7, 8 }, new[] { 9, 10, 11, 12 },
            new[] { 13, 14, 15, 16 }, new[] { 17, 18, 19, 20 }
        };

        /// <summary>Channel group bounds, identical to phase_learning.GROUPS.</summary>
        public static readonly int[] GroupStart =
        {
            0, 8, 50, 92, 132, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196
        };
        public static readonly int[] GroupEnd =
        {
            8, 50, 92, 132, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198
        };
        public const int GroupCount = 18;

        /// <summary>A per-frame [32, 198] block with NaN at every masked-out channel.</summary>
        public sealed class TemporalBlock
        {
            readonly double[] values;
            readonly bool[] mask;

            public double[] Values { get { return values; } }
            public bool[] Mask { get { return mask; } }
            public string SchemaId { get { return TemporalSchemaId; } }

            /// <summary>Takes ownership of nothing: both arrays are cloned and checked.</summary>
            public TemporalBlock(double[] values, bool[] mask)
            {
                if (values == null || mask == null)
                    throw new ArgumentNullException("values");
                if (values.Length != Frames * Channels || mask.Length != Frames * Channels)
                    throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                        "a temporal block must be 32 x 198");
                var v = (double[])values.Clone();
                var m = (bool[])mask.Clone();
                for (int i = 0; i < v.Length; i++)
                {
                    if (!m[i]) { v[i] = double.NaN; continue; }
                    if (double.IsNaN(v[i]) || double.IsInfinity(v[i]))
                        throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                            "a valid temporal channel is not finite; index " + i);
                }
                this.values = v;
                this.mask = m;
            }
        }

        // -----------------------------------------------------------------------------
        // 1) build_temporal_features
        // -----------------------------------------------------------------------------

        /// <summary>
        /// Builds the [32, 198] block: 172 raw xy copies, 20 in-plane bend cosines, 4 in-plane
        /// hand-axis components, 2 relative hand-wrist offsets. 2D descriptors, not 3D angles.
        /// </summary>
        public static TemporalBlock BuildTemporal(FullHandFeatureSequence sequence)
        {
            if (sequence == null) throw new ArgumentNullException("sequence");
            var values = new double[Frames * Channels];
            var mask = new bool[Frames * Channels];
            for (int i = 0; i < values.Length; i++) values[i] = double.NaN;

            for (int t = 0; t < Frames; t++)
            {
                int row = t * Channels;

                // 1) raw xy channels in source order, each point repeated over x and y
                for (int p = 0; p < PointCount; p++)
                {
                    bool ok = sequence.IsValid(t, p);
                    mask[row + 2 * p] = ok;
                    mask[row + 2 * p + 1] = ok;
                    if (!ok) continue;
                    values[row + 2 * p] = sequence.Coordinate(t, p, 0);
                    values[row + 2 * p + 1] = sequence.Coordinate(t, p, 1);
                }

                int col = PointEnd;

                // 2) bend cosines
                for (int side = 0; side < 2; side++)
                {
                    int b0 = side == 0 ? LeftBase : RightBase;
                    for (int f = 0; f < Fingers.Length; f++)
                    {
                        int[] chain = Fingers[f];
                        for (int k = 0; k < 2; k++)
                        {
                            int a = b0 + chain[k], b = b0 + chain[k + 1], c = b0 + chain[k + 2];
                            double ux = X(values, row, b) - X(values, row, a);
                            double uy = Y(values, row, b) - Y(values, row, a);
                            double vx = X(values, row, c) - X(values, row, b);
                            double vy = Y(values, row, c) - Y(values, row, b);
                            double nu = Norm(ux, uy), nv = Norm(vx, vy);
                            bool ok = nu > Eps && nv > Eps &&
                                      mask[row + 2 * a] && mask[row + 2 * b] && mask[row + 2 * c];
                            if (ok)
                            {
                                double cos = (ux * vx + uy * vy) / (nu * nv);
                                if (cos < -1.0) cos = -1.0;
                                else if (cos > 1.0) cos = 1.0;
                                values[row + col] = cos;
                            }
                            mask[row + col] = ok;
                            col++;
                        }
                    }
                }

                // 3) in-plane hand axis: hand wrist -> middle MCP, unit vector
                for (int side = 0; side < 2; side++)
                {
                    int b0 = side == 0 ? LeftBase : RightBase;
                    int w = b0 + WristJoint, mm = b0 + MiddleMcpJoint;
                    double dx = X(values, row, mm) - X(values, row, w);
                    double dy = Y(values, row, mm) - Y(values, row, w);
                    double n = Norm(dx, dy);
                    bool ok = !double.IsNaN(n) && !double.IsInfinity(n) && n > Eps &&
                              mask[row + 2 * w] && mask[row + 2 * mm];
                    if (ok)
                    {
                        values[row + col] = dx / n;
                        values[row + col + 1] = dy / n;
                    }
                    mask[row + col] = ok;
                    mask[row + col + 1] = ok;
                    col += 2;
                }

                // 4) right hand wrist minus left hand wrist
                {
                    int lw = LeftBase + WristJoint, rw = RightBase + WristJoint;
                    bool ok = mask[row + 2 * lw] && mask[row + 2 * rw];
                    if (ok)
                    {
                        values[row + col] = X(values, row, rw) - X(values, row, lw);
                        values[row + col + 1] = Y(values, row, rw) - Y(values, row, lw);
                    }
                    mask[row + col] = ok;
                    mask[row + col + 1] = ok;
                }
            }

            for (int i = 0; i < values.Length; i++) if (!mask[i]) values[i] = double.NaN;
            return new TemporalBlock(values, mask);
        }

        static double X(double[] v, int row, int point) { return v[row + 2 * point]; }
        static double Y(double[] v, int row, int point) { return v[row + 2 * point + 1]; }

        static double Norm(double x, double y)
        {
            double sum = x * x + y * y;
            if (double.IsNaN(sum)) return double.NaN;
            return sum <= 0.0 ? 0.0 : Math.Sqrt(sum);
        }

        // -----------------------------------------------------------------------------
        // 2) phase_pair_features
        // -----------------------------------------------------------------------------

        /// <summary>
        /// The 432 prespecified phase-preserving pair features. Per reference frame values are
        /// averaged FIRST so repeated warping cells never double-count, then aggregated over 8
        /// equal time bins as mean, linear p90 and coverage. A group residual is the RMS over
        /// the two columns of every unit valid in BOTH sequences; a bin with no valid residual
        /// stays NaN.
        /// </summary>
        public static double[] PhaseVector(TemporalBlock q, TemporalBlock r, List<int[]> path)
        {
            if (q == null || r == null) throw new ArgumentNullException("q");
            CheckPath(path);
            bool[][] qv = UnitValidity(q.Mask);
            bool[][] rv = UnitValidity(r.Mask);

            var resSum = new double[Frames * GroupCount];
            var resCnt = new double[Frames * GroupCount];
            var covSum = new double[Frames * GroupCount];
            var cellCnt = new double[Frames];

            for (int step = 0; step < path.Count; step++)
            {
                int i = path[step][0], j = path[step][1];
                cellCnt[j] += 1.0;
                for (int g = 0; g < GroupCount; g++)
                {
                    int start = GroupStart[g];
                    int units = (GroupEnd[g] - start) / 2;
                    bool[] qu = qv[g], ru = rv[g];
                    int nValid = 0;
                    double sq = 0.0;
                    for (int u = 0; u < units; u++)
                    {
                        if (!(qu[i * units + u] && ru[j * units + u])) continue;
                        nValid++;
                        double dx = q.Values[i * Channels + start + 2 * u]
                                  - r.Values[j * Channels + start + 2 * u];
                        double dy = q.Values[i * Channels + start + 2 * u + 1]
                                  - r.Values[j * Channels + start + 2 * u + 1];
                        sq += dx * dx + dy * dy;
                    }
                    covSum[j * GroupCount + g] += (double)nValid / units;
                    if (nValid == 0) continue;
                    double mean = sq / (2.0 * nValid);
                    resSum[j * GroupCount + g] += mean <= 0.0 ? 0.0 : Math.Sqrt(mean);
                    resCnt[j * GroupCount + g] += 1.0;
                }
            }

            var refRes = new double[Frames * GroupCount];
            var refCov = new double[Frames * GroupCount];
            for (int j = 0; j < Frames; j++)
                for (int g = 0; g < GroupCount; g++)
                {
                    int at = j * GroupCount + g;
                    refRes[at] = resCnt[at] > 0.0 ? resSum[at] / resCnt[at] : double.NaN;
                    refCov[at] = covSum[at] / cellCnt[j];      // 0/0 stays NaN, as in NumPy
                }

            var features = new double[FeatureCount];
            var scratch = new double[BinSize];
            for (int b = 0; b < Bins; b++)
                for (int g = 0; g < GroupCount; g++)
                {
                    int kept = 0;
                    for (int k = 0; k < BinSize; k++)
                    {
                        double value = refRes[(b * BinSize + k) * GroupCount + g];
                        if (!double.IsNaN(value) && !double.IsInfinity(value))
                            scratch[kept++] = value;
                    }
                    int at = (b * GroupCount + g) * Stats;
                    if (kept == 0)
                    {
                        features[at] = double.NaN;
                        features[at + 1] = double.NaN;
                    }
                    else
                    {
                        features[at] = FullHandPairGeometry.Mean(scratch, 0, kept);
                        features[at + 1] = FullHandPairGeometry.LinearPercentile90(scratch, kept);
                    }
                    double sum = 0.0;
                    for (int k = 0; k < BinSize; k++)
                        sum += refCov[(b * BinSize + k) * GroupCount + g];
                    features[at + 2] = sum / BinSize;
                }
            return features;
        }

        /// <summary>Per group, [frames, units]: BOTH columns of the 2-column unit are valid.</summary>
        static bool[][] UnitValidity(bool[] mask)
        {
            var out_ = new bool[GroupCount][];
            for (int g = 0; g < GroupCount; g++)
            {
                int start = GroupStart[g];
                int units = (GroupEnd[g] - start) / 2;
                var flags = new bool[Frames * units];
                for (int t = 0; t < Frames; t++)
                    for (int u = 0; u < units; u++)
                        flags[t * units + u] = mask[t * Channels + start + 2 * u]
                                            && mask[t * Channels + start + 2 * u + 1];
                out_[g] = flags;
            }
            return out_;
        }

        static void CheckPath(List<int[]> path)
        {
            if (path == null || path.Count == 0)
                throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                    "the path must be non-empty");
            for (int step = 0; step < path.Count; step++)
            {
                int[] cell = path[step];
                if (cell == null || cell.Length != 2)
                    throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                        "a path cell must hold exactly two indices");
                if (cell[0] < 0 || cell[0] >= Frames || cell[1] < 0 || cell[1] >= Frames)
                    throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                        "a path index is out of range");
                if (step == 0) continue;
                int di = cell[0] - path[step - 1][0], dj = cell[1] - path[step - 1][1];
                if (!((di == 1 && dj == 1) || (di == 1 && dj == 0) || (di == 0 && dj == 1)))
                    throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                        "path steps must be (1,1), (1,0) or (0,1)");
            }
            if (path[0][0] != 0 || path[0][1] != 0 ||
                path[path.Count - 1][0] != Frames - 1 || path[path.Count - 1][1] != Frames - 1)
                throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                    "the path must run from (0, 0) to (31, 31)");
        }

        // -----------------------------------------------------------------------------
        // 3) constrained_dtw_path
        // -----------------------------------------------------------------------------

        /// <summary>
        /// Run-limited monotonic alignment: endpoints preserved, no global band, at most two
        /// consecutive non-diagonal steps (vertical and horizontal share ONE counter), and no
        /// fallback. Returns the exact minimum TOTAL additive cost path over the (i, j, run)
        /// states; <paramref name="meanCost"/> is that selected total divided by the cell count,
        /// a reporting convention only - this is NOT a minimum-mean-cost alignment. Ties resolve
        /// diagonal, then vertical, then horizontal, then the smallest predecessor run; the end
        /// cell resolves to the smallest final run. Returns null when no legal path exists.
        /// </summary>
        public static List<int[]> ConstrainedPath(double[] cost, int n, int m, out double meanCost)
        {
            meanCost = double.PositiveInfinity;
            if (cost == null) throw new ArgumentNullException("cost");
            if (n < 1 || m < 1 || (long)n * m != cost.Length)
                throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                    "the cost grid must be a non-empty n x m block");
            for (int at = 0; at < cost.Length; at++)
            {
                if (double.IsNaN(cost[at]))
                    throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                        "cost holds NaN; an unavailable cell must be +infinity");
                if (cost[at] < 0.0)
                    throw new FullHandFormatException(FullHandFailureCodes.MalformedShape,
                        "cost must be nonnegative or +infinity");
            }

            const int Diag = 0, Vert = 1, Horz = 2;
            int runs = MaxConsecutiveNonDiagonal + 1;
            int states = n * m * runs;
            var best = new double[states];
            var prevStep = new int[states];
            var prevRun = new int[states];
            for (int at = 0; at < states; at++)
            {
                best[at] = double.PositiveInfinity;
                prevStep[at] = -1;
                prevRun[at] = -1;
            }
            if (double.IsInfinity(cost[0])) return null;
            best[0] = cost[0];

            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++)
                {
                    double cell = cost[i * m + j];
                    if (double.IsInfinity(cell)) continue;   // never enterable, never zero-cost
                    for (int r = 0; r < runs; r++)
                    {
                        if (i == 0 && j == 0) continue;      // start state is seeded above
                        double bestBase = double.PositiveInfinity;
                        int bestStep = -1, bestRun = -1;
                        if (r == 0)
                        {
                            if (i > 0 && j > 0)
                                for (int pr = 0; pr < runs; pr++)
                                {
                                    double b = best[((i - 1) * m + (j - 1)) * runs + pr];
                                    if (b < bestBase) { bestBase = b; bestStep = Diag; bestRun = pr; }
                                }
                        }
                        else
                        {
                            int pr = r - 1;
                            if (i > 0)
                            {
                                double b = best[((i - 1) * m + j) * runs + pr];
                                if (b < bestBase) { bestBase = b; bestStep = Vert; bestRun = pr; }
                            }
                            if (j > 0)
                            {
                                double b = best[(i * m + (j - 1)) * runs + pr];
                                // strict < keeps the earlier (vertical) preference on ties
                                if (b < bestBase) { bestBase = b; bestStep = Horz; bestRun = pr; }
                            }
                        }
                        if (bestStep < 0 || double.IsInfinity(bestBase)) continue;
                        int at = (i * m + j) * runs + r;
                        best[at] = bestBase + cell;
                        prevStep[at] = bestStep;
                        prevRun[at] = bestRun;
                    }
                }

            int endI = n - 1, endJ = m - 1, endRun = -1;
            double total = double.PositiveInfinity;
            for (int r = 0; r < runs; r++)          // ascending r, so the smallest run wins ties
            {
                double value = best[(endI * m + endJ) * runs + r];
                if (value < total) { total = value; endRun = r; }
            }
            if (endRun < 0 || double.IsInfinity(total) || double.IsNaN(total)) return null;

            var reversed = new List<int[]>();
            int ci = endI, cj = endJ, cr = endRun;
            while (true)
            {
                reversed.Add(new[] { ci, cj });
                if (ci == 0 && cj == 0) break;
                int at = (ci * m + cj) * runs + cr;
                int step = prevStep[at], pr = prevRun[at];
                if (step == Diag) { ci--; cj--; }
                else if (step == Vert) { ci--; }
                else if (step == Horz) { cj--; }
                else return null;
                cr = pr;
            }
            reversed.Reverse();
            meanCost = total / reversed.Count;
            return reversed;
        }

        // -----------------------------------------------------------------------------
        // pair entry points
        // -----------------------------------------------------------------------------

        /// <summary>Phase features on the fixed diagonal (frame t against frame t).</summary>
        public static FullHandPairResult FixedTimeFeatures(FullHandFeatureSequence a,
                                                           FullHandFeatureSequence b)
        {
            FullHandPairResult unusable = CheckUsable(a, b);
            if (unusable != null) return unusable;
            var path = new List<int[]>(Frames);
            for (int t = 0; t < Frames; t++) path.Add(new[] { t, t });
            double[] features = PhaseVector(BuildTemporal(a), BuildTemporal(b), path);
            return FullHandPairResult.Succeeded(features, double.NaN, path.Count);
        }

        /// <summary>Phase features on the run-limited path over the existing pair cost grid.</summary>
        public static FullHandPairResult AlignedPairFeatures(FullHandFeatureSequence a,
                                                             FullHandFeatureSequence b)
        {
            FullHandPairResult unusable = CheckUsable(a, b);
            if (unusable != null) return unusable;
            FullHandPairGeometryData geometry = FullHandPairGeometry.ComputeGeometry(a, b);
            double[] cost = geometry.Cost;
            if (cost == null || cost.Length != Frames * Frames)
                return FullHandPairResult.Failed(FullHandFailureCodes.MalformedShape);
            for (int at = 0; at < cost.Length; at++)
                if (double.IsNaN(cost[at]) || cost[at] < 0.0)
                    return FullHandPairResult.Failed(FullHandFailureCodes.MalformedShape);
            double meanCost;
            List<int[]> path = ConstrainedPath(cost, Frames, Frames, out meanCost);
            if (path == null || double.IsInfinity(meanCost) || double.IsNaN(meanCost))
                return FullHandPairResult.Failed(FullHandFailureCodes.NoValidDtwPath);
            double[] features = PhaseVector(BuildTemporal(a), BuildTemporal(b), path);
            return FullHandPairResult.Succeeded(features, meanCost, path.Count);
        }

        /// <summary>
        /// Only the sequence-level availability the Python phase code assumes. The old F4
        /// shared-hand-shape check is deliberately NOT applied here: the ported phase feature
        /// definition carries no such extra test.
        /// </summary>
        static FullHandPairResult CheckUsable(FullHandFeatureSequence a, FullHandFeatureSequence b)
        {
            if (a == null || b == null) throw new ArgumentNullException("a");
            if (!a.Usable)
                return FullHandPairResult.Failed(string.IsNullOrEmpty(a.Failure)
                    ? FullHandFailureCodes.AllPointsMissing : a.Failure);
            if (!b.Usable)
                return FullHandPairResult.Failed(string.IsNullOrEmpty(b.Failure)
                    ? FullHandFailureCodes.AllPointsMissing : b.Failure);
            return null;
        }
    }
}
