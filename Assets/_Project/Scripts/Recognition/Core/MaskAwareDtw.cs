using System;
using System.Collections.Generic;

namespace WordsInSilence.Recognition.Core
{
    /// <summary>Frozen Step 8A mask-aware DTW using double precision.</summary>
    public static class MaskAwareDtw
    {
        public const double MissingCost = 1.0;
        const double BandEpsilon = 1e-15;

        public static double LocalCost(RecognitionSequence left, int leftFrame, RecognitionSequence right, int rightFrame)
        {
            if (left == null) throw new ArgumentNullException(nameof(left));
            if (right == null) throw new ArgumentNullException(nameof(right));
            if ((uint)leftFrame >= (uint)left.Length) throw new ArgumentOutOfRangeException(nameof(leftFrame));
            if ((uint)rightFrame >= (uint)right.Length) throw new ArgumentOutOfRangeException(nameof(rightFrame));
            if (left.LandmarkCount != right.LandmarkCount) throw new ArgumentException("landmark counts differ");

            double total = 0.0;
            int included = 0;
            for (int landmark = 0; landmark < left.LandmarkCount; landmark++)
            {
                bool leftValid = left.IsValid(leftFrame, landmark);
                bool rightValid = right.IsValid(rightFrame, landmark);
                if (leftValid && rightValid)
                {
                    double dx = left.Coordinate(leftFrame, landmark, 0) - right.Coordinate(rightFrame, landmark, 0);
                    double dy = left.Coordinate(leftFrame, landmark, 1) - right.Coordinate(rightFrame, landmark, 1);
                    double dz = left.Coordinate(leftFrame, landmark, 2) - right.Coordinate(rightFrame, landmark, 2);
                    total += Math.Sqrt(dx * dx + dy * dy + dz * dz);
                    included++;
                }
                else if (leftValid != rightValid)
                {
                    total += MissingCost;
                    included++;
                }
            }
            return included == 0 ? MissingCost : total / included;
        }

        public static bool IsWithinRelativeBand(int i, int j, int lengthA, int lengthB, double relativeBand)
        {
            if (lengthA <= 1 || lengthB <= 1) return true;
            return Math.Abs((double)i / (lengthA - 1) - (double)j / (lengthB - 1)) <= relativeBand + BandEpsilon;
        }

        public static DtwResult Compute(RecognitionSequence left, RecognitionSequence right, double relativeBand)
        {
            if (left == null) throw new ArgumentNullException(nameof(left));
            if (right == null) throw new ArgumentNullException(nameof(right));
            if (double.IsNaN(relativeBand) || double.IsInfinity(relativeBand) || relativeBand < 0.0 || relativeBand > 1.0)
                throw new ArgumentOutOfRangeException(nameof(relativeBand));
            if (left.LandmarkCount != right.LandmarkCount) throw new ArgumentException("landmark counts differ");

            int lengthA = left.Length;
            int lengthB = right.Length;
            var cumulative = new double[lengthA, lengthB];
            var parent = new byte[lengthA, lengthB]; // 0 unset/start, 1 diagonal, 2 vertical, 3 horizontal
            for (int i = 0; i < lengthA; i++)
            for (int j = 0; j < lengthB; j++)
                cumulative[i, j] = double.PositiveInfinity;

            for (int i = 0; i < lengthA; i++)
            for (int j = 0; j < lengthB; j++)
            {
                if (!IsWithinRelativeBand(i, j, lengthA, lengthB, relativeBand)) continue;
                double local = LocalCost(left, i, right, j);
                if (i == 0 && j == 0)
                {
                    cumulative[i, j] = local;
                    continue;
                }

                double best = double.PositiveInfinity;
                byte direction = 0;
                if (i > 0 && j > 0 && !double.IsPositiveInfinity(cumulative[i - 1, j - 1]))
                {
                    best = cumulative[i - 1, j - 1];
                    direction = 1;
                }
                if (i > 0 && !double.IsPositiveInfinity(cumulative[i - 1, j]) && cumulative[i - 1, j] < best)
                {
                    best = cumulative[i - 1, j];
                    direction = 2;
                }
                if (j > 0 && !double.IsPositiveInfinity(cumulative[i, j - 1]) && cumulative[i, j - 1] < best)
                {
                    best = cumulative[i, j - 1];
                    direction = 3;
                }
                if (direction != 0)
                {
                    cumulative[i, j] = best + local;
                    parent[i, j] = direction;
                }
            }

            double total = cumulative[lengthA - 1, lengthB - 1];
            if (double.IsNaN(total) || double.IsInfinity(total))
                throw new InvalidOperationException("relative band cannot produce an endpoint DTW path");

            int pathI = lengthA - 1;
            int pathJ = lengthB - 1;
            var reverse = new List<DtwPathPoint> { new DtwPathPoint(pathI, pathJ) };
            while (pathI != 0 || pathJ != 0)
            {
                switch (parent[pathI, pathJ])
                {
                    case 1: pathI--; pathJ--; break;
                    case 2: pathI--; break;
                    case 3: pathJ--; break;
                    default: throw new InvalidOperationException("DTW parent path is incomplete");
                }
                reverse.Add(new DtwPathPoint(pathI, pathJ));
            }
            reverse.Reverse();
            return new DtwResult(total / reverse.Count, total, reverse);
        }
    }
}
