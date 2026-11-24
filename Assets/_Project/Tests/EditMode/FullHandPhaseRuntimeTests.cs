// Unit tests for the staged phase runtime candidate. These check ported arithmetic only.
// They make NO claim about recognition accuracy, handshape correctness or false accepts.
// Python-vs-C# parity on private recordings is added separately by the operator.
using System;
using System.Collections.Generic;
using NUnit.Framework;

namespace WordsInSilence.Recognition.FullHandRuntime
{
    [TestFixture]
    public sealed class FullHandPhaseRuntimeTests
    {
        const int Frames = FullHandPhasePairFeatures.Frames;
        const int Channels = FullHandPhasePairFeatures.Channels;
        const int Groups = FullHandPhasePairFeatures.GroupCount;
        const int SourcePoints = 46;

        sealed class Bank : IFullHandScoringBank
        {
            public FullHandModel Ml, Hybrid;
            public FullHandModel MlModel { get { return Ml; } }
            public FullHandModel HybridModel { get { return Hybrid; } }
            public int ReferenceCount { get { return 0; } }
            public string[] CopyVocabulary() { return new string[0]; }
            public int[] ReferencesForClass(int labelIndex) { return new int[0]; }
            public FullHandFeatureSequence ReferenceSequence(int index) { return null; }
        }

        static FullHandFeatureSequence BuildSequence(double shiftX, int droppedFrame,
                                                    int dropFrom, int dropTo)
        {
            var coordinates = new double[Frames * SourcePoints * 3];
            var mask = new bool[Frames * SourcePoints];
            for (int t = 0; t < Frames; t++)
                for (int p = 0; p < SourcePoints; p++)
                {
                    int slot = t * SourcePoints + p;
                    bool drop = t == droppedFrame && p >= dropFrom && p < dropTo;
                    mask[slot] = !drop;
                    coordinates[slot * 3] = drop ? double.NaN : shiftX + 0.01 * p + 0.002 * t;
                    coordinates[slot * 3 + 1] = drop ? double.NaN
                        : 0.013 * p - 0.004 * t + 0.1 * ((p % 3) - 1);
                    coordinates[slot * 3 + 2] = drop ? double.NaN : 0.0;
                }
            return FullHandPairGeometry.BuildFeatures(coordinates, mask, Frames);
        }

        /// <summary>Every channel valid; unit u of group 1 optionally masked out.</summary>
        static FullHandPhasePairFeatures.TemporalBlock Block(double dx, double dy, int droppedUnit)
        {
            var values = new double[Frames * Channels];
            var mask = new bool[Frames * Channels];
            for (int at = 0; at < values.Length; at++)
            {
                values[at] = (at % 2 == 0 ? dx : dy) + 0.001 * (at % Channels);
                mask[at] = true;
            }
            if (droppedUnit >= 0)
                for (int t = 0; t < Frames; t++)
                {
                    int at = t * Channels + 8 + 2 * droppedUnit;
                    mask[at] = false; mask[at + 1] = false;
                    values[at] = double.NaN; values[at + 1] = double.NaN;
                }
            return new FullHandPhasePairFeatures.TemporalBlock(values, mask);
        }

        static List<int[]> Diagonal()
        {
            var path = new List<int[]>();
            for (int t = 0; t < Frames; t++) path.Add(new[] { t, t });
            return path;
        }

        [Test]
        public void TemporalDerivedChannelsAreBoundedUnitAndTranslationInvariant()
        {
            var a = FullHandPhasePairFeatures.BuildTemporal(BuildSequence(0.0, -1, 0, 0));
            var b = FullHandPhasePairFeatures.BuildTemporal(BuildSequence(0.25, -1, 0, 0));
            for (int t = 0; t < Frames; t++)
            {
                int row = t * Channels;
                for (int c = 0; c < Channels; c++)
                    Assert.IsTrue(a.Mask[row + c], "channel " + c + " frame " + t);
                for (int c = 172; c < 192; c++)
                    Assert.LessOrEqual(Math.Abs(a.Values[row + c]), 1.0 + 1e-12, "bend " + c);
                for (int c = 192; c < 196; c += 2)
                {
                    double x = a.Values[row + c], y = a.Values[row + c + 1];
                    Assert.AreEqual(1.0, Math.Sqrt(x * x + y * y), 1e-9, "hand axis " + c);
                }
                for (int c = 172; c < Channels; c++)
                {
                    Assert.AreEqual(a.Mask[row + c], b.Mask[row + c]);
                    Assert.AreEqual(a.Values[row + c], b.Values[row + c], 1e-9, "channel " + c);
                }
            }
        }

        [Test]
        public void ADroppedLeftHandLeavesLeftDerivedChannelsMissing()
        {
            // Source points 4..24 are the left hand joints. The approved resampler carries a
            // single valid bracketing endpoint, so an interior missing frame is still observed
            // from its neighbour; only the LAST source frame has no later frame to carry it.
            const int Last = Frames - 1;
            var block = FullHandPhasePairFeatures.BuildTemporal(BuildSequence(0.0, Last, 4, 25));
            int row = Last * Channels;
            for (int c = 172; c < 182; c++)                 // left bend cosines
            {
                Assert.IsFalse(block.Mask[row + c], "left bend " + c);
                Assert.IsTrue(double.IsNaN(block.Values[row + c]));
            }
            Assert.IsFalse(block.Mask[row + 192]);          // left hand axis x and y
            Assert.IsFalse(block.Mask[row + 193]);
            Assert.IsTrue(block.Mask[192], "frame 0 keeps its left hand axis");
        }

        [Test]
        public void AConstantOffsetGivesTheKnownRootMeanSquareAndFullCoverage()
        {
            double[] features = FullHandPhasePairFeatures.PhaseVector(
                Block(0.3, 0.4, -1), Block(0.0, 0.0, -1), Diagonal());
            Assert.AreEqual(FullHandPhasePairFeatures.FeatureCount, features.Length);
            // The residual is the RMS over the unit's two COMPONENTS, not the Euclidean norm:
            // sqrt((0.09 + 0.16) / 2) = 0.5 / sqrt(2).
            double expected = 0.5 / Math.Sqrt(2.0);
            for (int b = 0; b < FullHandPhasePairFeatures.Bins; b++)
                for (int g = 0; g < Groups; g++)
                {
                    int at = (b * Groups + g) * 3;
                    Assert.AreEqual(expected, features[at], 1e-12, "mean b" + b + " g" + g);
                    Assert.AreEqual(expected, features[at + 1], 1e-12, "p90 b" + b + " g" + g);
                    Assert.AreEqual(1.0, features[at + 2], 1e-12, "coverage b" + b + " g" + g);
                }
        }

        [Test]
        public void AUnitMissingInOneSequenceLowersOnlyThatGroupCoverage()
        {
            double[] features = FullHandPhasePairFeatures.PhaseVector(
                Block(0.3, 0.4, -1), Block(0.0, 0.0, 5), Diagonal());
            int units = (FullHandPhasePairFeatures.GroupEnd[1]
                       - FullHandPhasePairFeatures.GroupStart[1]) / 2;
            for (int b = 0; b < FullHandPhasePairFeatures.Bins; b++)
            {
                int group1 = (b * Groups + 1) * 3;
                Assert.AreEqual((units - 1.0) / units, features[group1 + 2], 1e-12);
                Assert.AreEqual(0.5 / Math.Sqrt(2.0), features[group1], 1e-12,
                                "the residual stays the component RMS over the valid units");
                Assert.AreEqual(1.0, features[(b * Groups) * 3 + 2], 1e-12, "group 0 is intact");
            }
        }

        [Test]
        public void RepeatedReferenceCellsAreAveragedBeforeBinAggregation()
        {
            var reference = Block(0.0, 0.0, -1);
            var shifted = new double[Frames * Channels];
            var mask = new bool[Frames * Channels];
            for (int at = 0; at < shifted.Length; at++)
            {
                mask[at] = true;                            // only query frame 1 differs
                shifted[at] = reference.Values[at]
                    + (at / Channels == 1 ? (at % 2 == 0 ? 0.3 : 0.4) : 0.0);
            }
            var query = new FullHandPhasePairFeatures.TemporalBlock(shifted, mask);
            var path = new List<int[]> { new[] { 0, 0 }, new[] { 1, 0 } };
            for (int k = 2; k <= 31; k++) path.Add(new[] { k, k - 1 });
            path.Add(new[] { 31, 31 });
            double[] features = FullHandPhasePairFeatures.PhaseVector(query, reference, path);
            // A (0.3, 0.4) offset is a component RMS of 0.5 / sqrt(2), so reference frame 0 sees
            // residuals 0 and 0.5/sqrt(2), averaged to 0.25/sqrt(2) BEFORE binning; frames 1..3
            // of bin 0 see 0, so the bin holds {0.25/sqrt(2), 0, 0, 0}.
            double residual = 0.25 / Math.Sqrt(2.0);
            Assert.AreEqual(residual / 4.0, features[0], 1e-12, "bin mean");
            Assert.AreEqual(0.7 * residual, features[1], 1e-12,
                            "linear p90 of {0, 0, 0, 0.25/sqrt(2)}");
        }

        [Test]
        public void TheRunLimitedPathIsExactEndpointPreservingAndHasNoFallback()
        {
            double mean;
            Assert.IsNull(FullHandPhasePairFeatures.ConstrainedPath(new double[8], 1, 8, out mean),
                          "7 straight horizontal steps exceed the run limit");
            Assert.IsTrue(double.IsPositiveInfinity(mean));
            Assert.AreEqual(3, FullHandPhasePairFeatures.ConstrainedPath(
                new double[3], 1, 3, out mean).Count, "two horizontal steps are still legal");

            // 2 x 3 grid; the cheapest total is (0,0) -> (0,1) -> (1,2) at 1 + 0 + 0 = 1.
            var cost = new double[] { 1.0, 0.0, 9.0, 9.0, 9.0, 0.0 };
            List<int[]> path = FullHandPhasePairFeatures.ConstrainedPath(cost, 2, 3, out mean);
            Assert.AreEqual(new[] { 0, 0 }, path[0]);
            Assert.AreEqual(new[] { 0, 1 }, path[1]);
            Assert.AreEqual(new[] { 1, 2 }, path[2]);
            Assert.AreEqual(1.0 / 3.0, mean, 1e-12, "the total is divided by the cell count");

            // all-equal costs: the end state resolves to the smallest run, so the last step
            // is diagonal and the choice is reproducible.
            List<int[]> tied = FullHandPhasePairFeatures.ConstrainedPath(
                new double[6], 2, 3, out mean);
            Assert.AreEqual(new[] { 0, 1 }, tied[1]);
            Assert.AreEqual(0.0, mean, 1e-12);

            var unavailableStart = new double[] { double.PositiveInfinity, 0.0, 0.0, 0.0 };
            Assert.IsNull(FullHandPhasePairFeatures.ConstrainedPath(
                unavailableStart, 2, 2, out mean), "an +inf cell is never entered at zero cost");
        }

        [Test]
        public void APhaseModelImputesItsOwnMeanAndRejectsNonMonotoneCoefficients()
        {
            int width = FullHandPhasePairFeatures.FeatureCount;
            var mean = new double[width];
            var scale = new double[width];
            var coef = new double[width];
            var impute = new double[width];
            for (int index = 0; index < width; index++)
            {
                mean[index] = 1.0;
                scale[index] = 2.0;
                impute[index] = 5.0;                        // deliberately != the scaler mean
                coef[index] = index % 3 == 2 ? 0.25 : -0.25;
            }
            var phase = new FullHandModel(mean, scale, coef, 0.0, impute);
            Assert.IsTrue(phase.IsPhase);
            var features = new double[width];
            for (int index = 0; index < width; index++) features[index] = double.NaN;
            features[0] = 3.0;
            double[] standardized = phase.Standardize(features);
            Assert.AreEqual(1.0, standardized[0], 1e-12, "(3 - 1) / 2");
            Assert.AreEqual(2.0, standardized[1], 1e-12, "(5 - 1) / 2");
            features[0] = double.PositiveInfinity;          // NaN is missing, infinity is not
            Assert.Throws<FullHandFormatException>(delegate { phase.Standardize(features); });
            coef[0] = 1.0;
            Assert.Throws<FullHandFormatException>(
                delegate { new FullHandModel(mean, scale, coef, 0.0, impute); });
            coef[0] = -0.25;
            Assert.Throws<FullHandFormatException>(
                delegate { new FullHandModel(mean, scale, coef, 0.0, null); });
        }

        [Test]
        public void ClassicWidthsKeepZeroImputationAndMixedBankWidthsAreRejected()
        {
            int classic = FullHandPairConstants.FixedFeatureCount;
            var cm = new double[classic];
            var cs = new double[classic];
            var cc = new double[classic];
            for (int index = 0; index < classic; index++) { cs[index] = 2.0; cc[index] = 1.0; }
            var old = new FullHandModel(cm, cs, cc, 0.0);
            Assert.IsFalse(old.IsPhase);
            var raw = new double[classic];
            raw[0] = double.NaN;
            Assert.AreEqual(0.0, old.Standardize(raw)[0], 1e-12, "classic imputation stays 0");

            int width = FullHandPhasePairFeatures.FeatureCount;
            var mean = new double[width];
            var scale = new double[width];
            var coef = new double[width];
            var impute = new double[width];
            for (int index = 0; index < width; index++) { scale[index] = 1.0; }
            var bank = new Bank
            {
                Ml = new FullHandModel(mean, scale, coef, 0.0, impute),
                Hybrid = old
            };
            Assert.Throws<FullHandFormatException>(delegate
            {
                FullHandScorer.Evaluate(bank, BuildSequence(0.0, -1, 0, 0));
            });
        }
    }
}
