// F4 EditMode parity tests: the C# port against goldens produced by the verified Python.
//
// No webcam, no scene, no private learner or model file, no .meta authoring. The fixture is
// the SYNTHETIC export only. Malformed-bundle cases operate on temporary copies. The parity
// report is written ONLY to the path in WIS_FULL_HAND_PARITY_OUTPUT.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Recognition.FullHandRuntime;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class FullHandRuntimeParityTests
    {
        const string FixtureRelative = "Validation/m3_full_hand_20260909/f4_synthetic";
        const string FixtureEnvironment = "WIS_FULL_HAND_FIXTURE";
        const string ReportEnvironment = "WIS_FULL_HAND_PARITY_OUTPUT";
        const double Tolerance = 1e-10;

        static readonly byte[] GoldenMagic = Encoding.ASCII.GetBytes("WISFHG01");
        static readonly string[] MethodOrder = { "dtw_only", "ml_only", "hybrid" };

        [Serializable]
        sealed class GoldenManifest
        {
            public string schema;
            public bool synthetic;
            public string payload_file;
            public long payload_bytes;
            public string payload_sha256;
        }

        [Serializable]
        sealed class CaseReport
        {
            public string name;
            public string kind;
            public int comparisons;
            public string maxAbsoluteError;
            public string status;
            public string detail;
            public string[] values;
        }

        [Serializable]
        sealed class ParityReport
        {
            public string schema = "wis_full_hand_parity_report_v1";
            public string fixturePath;
            public bool synthetic;
            public string bundlePayloadSha256;
            public string goldenPayloadSha256;
            public string tolerance;
            public int caseCount;
            public int totalComparisons;
            public string maxAbsoluteError;
            public int mismatchCount;
            public string[] mismatches;
            public string[] notes;
            public CaseReport[] cases;
        }

        // ---- shared state, computed once -------------------------------------

        static string fixturePath;
        static FullHandRuntimeBundle bundle;
        static GoldenData golden;
        static readonly List<CaseReport> reports = new List<CaseReport>();
        static readonly List<string> mismatches = new List<string>();
        static int totalComparisons;
        static double maxError;
        static string setupSkipReason;

        sealed class GoldenFeatureCase
        {
            public string Name;
            public int SourceFrames;
            public double[] InputCoordinates;
            public bool[] InputMask;
            public bool Usable;
            public string Failure;
            public double LeftScale, RightScale;
            public bool[] GroupAvailable;
            public double[] Points;
            public bool[] Mask;
        }

        sealed class GoldenPairCase
        {
            public string Name, Query, Reference;
            public bool FixedOk; public string FixedFailure; public double[] FixedFeatures;
            public bool AlignedOk; public string AlignedFailure; public double[] AlignedFeatures;
            public double AlignedCost; public int AlignedPathLength;
            public bool DtwOk; public string DtwFailure; public double DtwCost; public int DtwPathLength;
        }

        sealed class GoldenMethod
        {
            public string Method, Status, Reason, Top1;
            public double Score, Margin;
            public int FailedPairs;
            public double[] Raw;
        }

        sealed class GoldenEvaluation
        {
            public string Name, Query;
            public int ClassCount;
            public GoldenMethod[] Methods;
        }

        sealed class GoldenData
        {
            public string Sha256;
            public List<GoldenFeatureCase> Features = new List<GoldenFeatureCase>();
            public List<GoldenPairCase> Pairs = new List<GoldenPairCase>();
            public List<GoldenEvaluation> Evaluations = new List<GoldenEvaluation>();
            public Dictionary<string, FullHandFeatureSequence> Built =
                new Dictionary<string, FullHandFeatureSequence>(StringComparer.Ordinal);
        }

        [OneTimeSetUp]
        public void LoadFixtureAndCompare()
        {
            fixturePath = ResolveFixture();
            if (fixturePath == null)
            {
                setupSkipReason = "the synthetic fixture was not found at " + FixtureRelative
                    + "; export it first or set " + FixtureEnvironment;
                return;
            }
            bundle = FullHandRuntimeBundle.Load(Path.Combine(fixturePath, "runtime_manifest.json"));
            golden = ReadGolden(fixturePath);
            CompareFeatureCases();
            ComparePairCases();
            CompareEvaluations();
        }

        [OneTimeTearDown]
        public void WriteParityReport()
        {
            string target = Environment.GetEnvironmentVariable(ReportEnvironment);
            if (string.IsNullOrEmpty(target) || bundle == null) return;
            var report = new ParityReport
            {
                fixturePath = FixtureRelative,
                synthetic = bundle.Synthetic,
                bundlePayloadSha256 = bundle.PayloadSha256,
                goldenPayloadSha256 = golden == null ? string.Empty : golden.Sha256,
                tolerance = Number(Tolerance),
                caseCount = reports.Count,
                totalComparisons = totalComparisons,
                maxAbsoluteError = Number(maxError),
                mismatchCount = mismatches.Count,
                mismatches = mismatches.ToArray(),
                cases = reports.ToArray(),
                notes = new[]
                {
                    "numbers are invariant round-trip strings, so NaN survives as NaN",
                    "NaN is expected only where a group or class has no evidence",
                    "synthetic fixture only; no user data, no model promotion, NOT_WIRED"
                }
            };
            string directory = Path.GetDirectoryName(Path.GetFullPath(target));
            if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);
            File.WriteAllText(target, JsonUtility.ToJson(report, true), new UTF8Encoding(false));
        }

        static string ResolveFixture()
        {
            string overridePath = Environment.GetEnvironmentVariable(FixtureEnvironment);
            if (!string.IsNullOrEmpty(overridePath) &&
                File.Exists(Path.Combine(overridePath, "runtime_manifest.json")))
                return overridePath;
            string project = Path.GetDirectoryName(Application.dataPath);
            string candidate = Path.Combine(project, FixtureRelative.Replace('/', Path.DirectorySeparatorChar));
            return File.Exists(Path.Combine(candidate, "runtime_manifest.json")) ? candidate : null;
        }

        void RequireFixture()
        {
            if (setupSkipReason != null) Assert.Ignore(setupSkipReason);
        }

        // ---- golden reader ----------------------------------------------------

        static GoldenData ReadGolden(string directory)
        {
            string manifestPath = Path.Combine(directory, "golden_manifest.json");
            var manifest = JsonUtility.FromJson<GoldenManifest>(
                File.ReadAllText(manifestPath, new UTF8Encoding(false)));
            Assert.AreEqual("wis_full_hand_golden_v1", manifest.schema);
            Assert.IsTrue(manifest.synthetic, "only a synthetic golden may be read here");
            byte[] payload = File.ReadAllBytes(Path.Combine(directory, "golden_payload.bin"));
            string digest = Sha256Hex(payload);
            Assert.AreEqual(manifest.payload_sha256, digest, "the golden payload changed");
            Assert.AreEqual(manifest.payload_bytes, payload.LongLength);

            var data = new GoldenData { Sha256 = digest };
            int cursor = 0;
            for (int index = 0; index < GoldenMagic.Length; index++)
                Assert.AreEqual(GoldenMagic[index], payload[cursor++], "golden magic");
            Assert.AreEqual(1, ReadInt(payload, ref cursor), "golden format version");

            int frames = FullHandPairConstants.ResampleFrames;
            int points = FullHandPairConstants.FeaturePointCount;
            int dims = FullHandPairConstants.CoordDims;
            int source = FullHandPairConstants.SourcePointCount;

            int featureCount = ReadInt(payload, ref cursor);
            for (int index = 0; index < featureCount; index++)
            {
                var entry = new GoldenFeatureCase();
                entry.Name = ReadText(payload, ref cursor);
                entry.SourceFrames = ReadInt(payload, ref cursor);
                entry.InputCoordinates = ReadDoubles(payload, ref cursor,
                                                     entry.SourceFrames * source * dims);
                entry.InputMask = ReadMask(payload, ref cursor, entry.SourceFrames * source);
                entry.Usable = payload[cursor++] == 1;
                entry.Failure = ReadText(payload, ref cursor);
                entry.LeftScale = ReadDouble(payload, ref cursor);
                entry.RightScale = ReadDouble(payload, ref cursor);
                entry.GroupAvailable = ReadMask(payload, ref cursor, FullHandPairConstants.GroupCount);
                entry.Points = ReadDoubles(payload, ref cursor, frames * points * dims);
                entry.Mask = ReadMask(payload, ref cursor, frames * points);
                data.Features.Add(entry);
            }

            int pairCount = ReadInt(payload, ref cursor);
            for (int index = 0; index < pairCount; index++)
            {
                var entry = new GoldenPairCase();
                entry.Name = ReadText(payload, ref cursor);
                entry.Query = ReadText(payload, ref cursor);
                entry.Reference = ReadText(payload, ref cursor);
                entry.FixedOk = payload[cursor++] == 1;
                entry.FixedFailure = ReadText(payload, ref cursor);
                entry.FixedFeatures = ReadDoubles(payload, ref cursor,
                                                  FullHandPairConstants.FixedFeatureCount);
                entry.AlignedOk = payload[cursor++] == 1;
                entry.AlignedFailure = ReadText(payload, ref cursor);
                entry.AlignedFeatures = ReadDoubles(payload, ref cursor,
                                                    FullHandPairConstants.AlignedFeatureCount);
                entry.AlignedCost = ReadDouble(payload, ref cursor);
                entry.AlignedPathLength = ReadInt(payload, ref cursor);
                entry.DtwOk = payload[cursor++] == 1;
                entry.DtwFailure = ReadText(payload, ref cursor);
                entry.DtwCost = ReadDouble(payload, ref cursor);
                entry.DtwPathLength = ReadInt(payload, ref cursor);
                data.Pairs.Add(entry);
            }

            int evaluationCount = ReadInt(payload, ref cursor);
            for (int index = 0; index < evaluationCount; index++)
            {
                var entry = new GoldenEvaluation();
                entry.Name = ReadText(payload, ref cursor);
                entry.Query = ReadText(payload, ref cursor);
                entry.ClassCount = ReadInt(payload, ref cursor);
                entry.Methods = new GoldenMethod[MethodOrder.Length];
                for (int method = 0; method < MethodOrder.Length; method++)
                {
                    var value = new GoldenMethod();
                    value.Method = ReadText(payload, ref cursor);
                    value.Status = ReadText(payload, ref cursor);
                    value.Reason = ReadText(payload, ref cursor);
                    value.Top1 = ReadText(payload, ref cursor);
                    value.Score = ReadDouble(payload, ref cursor);
                    value.Margin = ReadDouble(payload, ref cursor);
                    value.FailedPairs = ReadInt(payload, ref cursor);
                    value.Raw = ReadDoubles(payload, ref cursor, entry.ClassCount);
                    entry.Methods[method] = value;
                }
                data.Evaluations.Add(entry);
            }
            Assert.AreEqual(payload.Length, cursor, "the golden payload has trailing bytes");
            return data;
        }

        // ---- comparisons -------------------------------------------------------

        static void CompareFeatureCases()
        {
            foreach (GoldenFeatureCase entry in golden.Features)
            {
                var report = new CaseReport { name = entry.Name, kind = "feature_sequence" };
                FullHandFeatureSequence built = FullHandPairGeometry.BuildFeatures(
                    entry.InputCoordinates, entry.InputMask, entry.SourceFrames);
                golden.Built[entry.Name] = built;
                double error = 0.0;
                int comparisons = 0;
                Expect(entry.Name, "usable", entry.Usable, built.Usable);
                Expect(entry.Name, "failure", entry.Failure, built.Failure ?? string.Empty);
                error = Math.Max(error, Numeric(entry.Name, "left_scale", entry.LeftScale,
                                                built.LeftHandScale, ref comparisons));
                error = Math.Max(error, Numeric(entry.Name, "right_scale", entry.RightScale,
                                                built.RightHandScale, ref comparisons));
                bool[] available = built.CopyGroupAvailable();
                for (int group = 0; group < available.Length; group++)
                    Expect(entry.Name, "group_available_" + group,
                           entry.GroupAvailable[group], available[group]);
                bool[] mask = built.CopyMask();
                double[] pointValues = built.CopyPoints();
                Assert.AreEqual(entry.Mask.Length, mask.Length);
                for (int slot = 0; slot < mask.Length; slot++)
                {
                    Expect(entry.Name, "mask_" + slot, entry.Mask[slot], mask[slot]);
                    for (int axis = 0; axis < FullHandPairConstants.CoordDims; axis++)
                        error = Math.Max(error, Numeric(entry.Name, "point",
                            entry.Points[slot * FullHandPairConstants.CoordDims + axis],
                            pointValues[slot * FullHandPairConstants.CoordDims + axis],
                            ref comparisons));
                }
                report.comparisons = comparisons;
                report.maxAbsoluteError = Number(error);
                report.status = "COMPARED";
                report.values = new[] { "usable=" + built.Usable,
                                        "left_scale=" + Number(built.LeftHandScale),
                                        "right_scale=" + Number(built.RightHandScale),
                                        "valid_points=" + CountTrue(mask) };
                Record(report, error);
            }
        }

        static void ComparePairCases()
        {
            foreach (GoldenPairCase entry in golden.Pairs)
            {
                var report = new CaseReport { name = entry.Name, kind = "pair" };
                FullHandFeatureSequence query = golden.Built[entry.Query];
                FullHandFeatureSequence reference = golden.Built[entry.Reference];
                double error = 0.0;
                int comparisons = 0;
                var values = new List<string>();

                FullHandPairResult fixedTime = FullHandPairGeometry.FixedTimeFeatures(query, reference);
                Expect(entry.Name, "fixed_ok", entry.FixedOk, fixedTime.Ok);
                Expect(entry.Name, "fixed_failure", entry.FixedFailure,
                       fixedTime.Failure ?? string.Empty);
                values.Add("fixed_ok=" + fixedTime.Ok);
                values.Add("fixed_failure=" + (fixedTime.Failure ?? "none"));
                if (entry.FixedOk && fixedTime.Ok)
                {
                    double[] produced = fixedTime.CopyFeatures();
                    for (int index = 0; index < produced.Length; index++)
                    {
                        error = Math.Max(error, Numeric(entry.Name, "fixed_" + index,
                            entry.FixedFeatures[index], produced[index], ref comparisons));
                        values.Add("fixed_" + index + "=" + Number(produced[index]));
                    }
                }

                FullHandPairResult aligned = FullHandPairGeometry.AlignedPairFeatures(query, reference);
                Expect(entry.Name, "aligned_ok", entry.AlignedOk, aligned.Ok);
                Expect(entry.Name, "aligned_failure", entry.AlignedFailure,
                       aligned.Failure ?? string.Empty);
                values.Add("aligned_ok=" + aligned.Ok);
                values.Add("aligned_failure=" + (aligned.Failure ?? "none"));
                values.Add("aligned_path_length=" + aligned.PathLength);
                if (entry.AlignedOk && aligned.Ok)
                {
                    Expect(entry.Name, "aligned_path_length", entry.AlignedPathLength,
                           aligned.PathLength);
                    error = Math.Max(error, Numeric(entry.Name, "aligned_cost",
                        entry.AlignedCost, aligned.DtwCost, ref comparisons));
                    double[] produced = aligned.CopyFeatures();
                    for (int index = 0; index < produced.Length; index++)
                    {
                        error = Math.Max(error, Numeric(entry.Name, "aligned_" + index,
                            entry.AlignedFeatures[index], produced[index], ref comparisons));
                        values.Add("aligned_" + index + "=" + Number(produced[index]));
                    }
                    values.Add("aligned_cost=" + Number(aligned.DtwCost));
                }

                FullHandPairResult distance = FullHandPairGeometry.DtwDistance(query, reference);
                Expect(entry.Name, "dtw_ok", entry.DtwOk, distance.Ok);
                Expect(entry.Name, "dtw_failure", entry.DtwFailure,
                       distance.Failure ?? string.Empty);
                values.Add("dtw_ok=" + distance.Ok);
                values.Add("dtw_failure=" + (distance.Failure ?? "none"));
                values.Add("dtw_path_length=" + distance.PathLength);
                if (entry.DtwOk && distance.Ok)
                {
                    Expect(entry.Name, "dtw_path_length", entry.DtwPathLength, distance.PathLength);
                    error = Math.Max(error, Numeric(entry.Name, "dtw_cost", entry.DtwCost,
                                                    distance.DtwCost, ref comparisons));
                    values.Add("dtw_cost=" + Number(distance.DtwCost));
                }
                report.comparisons = comparisons;
                report.maxAbsoluteError = Number(error);
                report.status = "COMPARED";
                report.values = values.ToArray();
                Record(report, error);
            }
        }

        static void CompareEvaluations()
        {
            string[] vocabulary = bundle.CopyVocabulary();
            foreach (GoldenEvaluation entry in golden.Evaluations)
            {
                var report = new CaseReport { name = entry.Name, kind = "evaluation" };
                FullHandEvaluation produced = FullHandScorer.Evaluate(bundle,
                                                                      golden.Built[entry.Query]);
                double error = 0.0;
                int comparisons = 0;
                var values = new List<string>();
                Assert.AreEqual(entry.ClassCount, vocabulary.Length, entry.Name + " class count");
                for (int index = 0; index < MethodOrder.Length; index++)
                {
                    GoldenMethod expected = entry.Methods[index];
                    FullHandMethodResult actual = produced.Method(MethodOrder[index]);
                    Assert.IsNotNull(actual, entry.Name + " " + MethodOrder[index]);
                    Expect(entry.Name, MethodOrder[index] + "_status", expected.Status, actual.Status);
                    Expect(entry.Name, MethodOrder[index] + "_reason", expected.Reason,
                           actual.Reason ?? string.Empty);
                    Expect(entry.Name, MethodOrder[index] + "_top1", expected.Top1,
                           actual.Top1 ?? string.Empty);
                    Expect(entry.Name, MethodOrder[index] + "_failed_pairs",
                           expected.FailedPairs, actual.FailedPairs);
                    error = Math.Max(error, Numeric(entry.Name, MethodOrder[index] + "_score",
                        expected.Score, actual.Score, ref comparisons));
                    error = Math.Max(error, Numeric(entry.Name, MethodOrder[index] + "_margin",
                        expected.Margin, actual.Margin, ref comparisons));
                    // the oracle exposes per-class scores only on a RANKED outcome; the C#
                    // side keeps them as diagnostics, so they are compared only there
                    if (expected.Status == FullHandStatus.Ranked)
                        for (int label = 0; label < vocabulary.Length; label++)
                            error = Math.Max(error, Numeric(entry.Name,
                                MethodOrder[index] + "_raw_" + label, expected.Raw[label],
                                actual.RawScores[label], ref comparisons));
                    values.Add(MethodOrder[index] + "=" + actual.Status + "/" +
                               (actual.Top1 ?? "none") + "/" + Number(actual.Score) + "/" +
                               Number(actual.Margin));
                    values.Add(MethodOrder[index] + "_reason=" + (actual.Reason ?? "none"));
                    values.Add(MethodOrder[index] + "_failed_pairs=" + actual.FailedPairs);
                    for (int label = 0; label < vocabulary.Length; label++)
                        values.Add(MethodOrder[index] + "_raw_" + vocabulary[label] + "=" +
                                   Number(actual.RawScores[label]));
                }
                report.comparisons = comparisons;
                report.maxAbsoluteError = Number(error);
                report.status = "COMPARED";
                report.values = values.ToArray();
                Record(report, error);
            }
        }

        static void Record(CaseReport report, double error)
        {
            reports.Add(report);
            totalComparisons += report.comparisons;
            if (error > maxError) maxError = error;
        }

        static void Expect(string caseName, string field, object expected, object actual)
        {
            if (!Equals(expected, actual))
                mismatches.Add(caseName + "." + field + ": expected " + expected +
                               ", got " + actual);
        }

        /// <summary>NaN must match NaN exactly; otherwise the absolute error must be tiny.</summary>
        static double Numeric(string caseName, string field, double expected, double actual,
                              ref int comparisons)
        {
            comparisons++;
            bool expectedNaN = double.IsNaN(expected);
            bool actualNaN = double.IsNaN(actual);
            if (expectedNaN || actualNaN)
            {
                if (expectedNaN != actualNaN)
                    mismatches.Add(caseName + "." + field + ": NaN availability differs");
                return 0.0;
            }
            if (double.IsInfinity(expected) || double.IsInfinity(actual))
            {
                if (expected != actual)
                    mismatches.Add(caseName + "." + field + ": infinity differs");
                return 0.0;
            }
            double error = Math.Abs(expected - actual);
            if (error > Tolerance)
                mismatches.Add(caseName + "." + field + ": |" + Number(expected) + " - " +
                               Number(actual) + "| = " + Number(error));
            return error;
        }

        // ---- tests -------------------------------------------------------------

        [Test]
        public void BundleLoadsAsSyntheticWithTheExpectedSchema()
        {
            RequireFixture();
            Assert.IsTrue(bundle.Synthetic, "the parity fixture must be synthetic");
            Assert.GreaterOrEqual(bundle.VocabularyCount, 3);
            Assert.GreaterOrEqual(bundle.ReferenceCount, 2);
            Assert.AreEqual(FullHandPairConstants.FixedFeatureCount, bundle.MlModel.FeatureCount);
            Assert.AreEqual(FullHandPairConstants.AlignedFeatureCount,
                            bundle.HybridModel.FeatureCount);
            string[] vocabulary = bundle.CopyVocabulary();
            for (int index = 1; index < vocabulary.Length; index++)
                Assert.Less(string.CompareOrdinal(vocabulary[index - 1], vocabulary[index]), 0);
            Assert.AreEqual(86, FullHandPairConstants.FeaturePointCount);
            Assert.AreEqual(15, FullHandPairConstants.DescriptorNames().Length - 3);
        }

        [Test]
        public void FeatureSequencesMatchTheGolden()
        {
            RequireFixture();
            Assert.Greater(golden.Features.Count, 0);
            AssertNoMismatch("feature_sequence");
        }

        [Test]
        public void PairDescriptorsCostsAndStatusesMatchTheGolden()
        {
            RequireFixture();
            Assert.Greater(golden.Pairs.Count, 0);
            AssertNoMismatch("pair");
        }

        [Test]
        public void ClassScoresAndRankingsMatchTheGolden()
        {
            RequireFixture();
            Assert.Greater(golden.Evaluations.Count, 0);
            AssertNoMismatch("evaluation");
        }

        [Test]
        public void ShapeAvailabilityAsymmetryAndImmutabilityHold()
        {
            RequireFixture();
            GoldenPairCase disjoint = FindPair("pair_disjoint");
            Assert.IsFalse(disjoint.FixedOk);
            Assert.AreEqual(FullHandFailureCodes.NoSharedHandShape, disjoint.FixedFailure);
            Assert.IsFalse(disjoint.AlignedOk);
            Assert.IsFalse(disjoint.DtwOk);

            GoldenPairCase asymmetric = FindPair("pair_asymmetric");
            Assert.IsFalse(asymmetric.FixedOk, "the fixed diagonal must lack shared shape");
            Assert.IsTrue(asymmetric.AlignedOk, "the aligned path must still share shape");
            FullHandPairResult producedFixed = FullHandPairGeometry.FixedTimeFeatures(
                golden.Built[asymmetric.Query], golden.Built[asymmetric.Reference]);
            FullHandPairResult producedAligned = FullHandPairGeometry.AlignedPairFeatures(
                golden.Built[asymmetric.Query], golden.Built[asymmetric.Reference]);
            Assert.IsFalse(producedFixed.Ok);
            Assert.IsTrue(producedAligned.Ok, "one method's failure must not erase the other");

            // nothing default or private is ever consulted
            Assert.Throws<FullHandFormatException>(
                delegate { FullHandRuntimeBundle.Load(Path.Combine(fixturePath, "absent.json")); });

            FullHandFeatureSequence sequence = golden.Built[asymmetric.Query];
            double[] points = sequence.CopyPoints();
            double first = points.Length > 0 ? points[0] : 0.0;
            points[0] = 12345.0;
            Assert.AreNotEqual(12345.0, sequence.CopyPoints()[0]);
            Assert.AreEqual(first, sequence.CopyPoints()[0]);
            bool[] mask = sequence.CopyMask();
            mask[0] = !mask[0];
            Assert.AreEqual(sequence.IsValid(0, 0), sequence.CopyMask()[0]);
            string[] vocabulary = bundle.CopyVocabulary();
            vocabulary[0] = "MUTATED";
            Assert.AreNotEqual("MUTATED", bundle.CopyVocabulary()[0]);
            double[] coefficients = bundle.HybridModel.CopyCoefficients();
            coefficients[0] += 1.0;
            Assert.AreNotEqual(coefficients[0], bundle.HybridModel.CopyCoefficients()[0]);
            double[] referencePoints = bundle.ReferenceSequence(0).CopyPoints();
            referencePoints[0] = 999.0;
            Assert.AreNotEqual(999.0, bundle.ReferenceSequence(0).CopyPoints()[0]);
        }

        [Test]
        public void MalformedBundlesAreRefused()
        {
            RequireFixture();
            string temporary = Path.Combine(Path.GetTempPath(),
                "wis_full_hand_parity_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(temporary);
            try
            {
                byte[] payload = File.ReadAllBytes(Path.Combine(fixturePath, "runtime_payload.bin"));
                string manifestText = File.ReadAllText(
                    Path.Combine(fixturePath, "runtime_manifest.json"), new UTF8Encoding(false));

                Refuse(temporary, manifestText, Corrupt(payload, 0, 0x00), true, "magic",
                       FullHandBundleCodes.HeaderInvalid);
                // the previous version passed the ORIGINAL bytes with the ORIGINAL digest,
                // which is a valid bundle. Flip one byte and keep the stale digest, so the
                // refusal comes from the digest check and not from a byte-length change.
                Refuse(temporary, manifestText,
                       Corrupt(payload, payload.Length - 1,
                               (byte)(payload[payload.Length - 1] ^ 0xFF)),
                       false, "digest", FullHandBundleCodes.DigestMismatch);
                Refuse(temporary, manifestText, Truncate(payload, 16), true, "truncation",
                       FullHandBundleCodes.Truncated);
                Refuse(temporary, manifestText, Append(payload, 8), true, "trailing bytes",
                       FullHandBundleCodes.TrailingBytes);
                Refuse(temporary, manifestText, Corrupt(payload, 12, 0x7f), true, "dimensions",
                       FullHandBundleCodes.DimensionMismatch);
                Refuse(temporary, manifestText, ZeroFirstScale(payload), true, "zero scale",
                       FullHandBundleCodes.ModelInvalid);
                Refuse(temporary, manifestText, BreakFirstLabelIndex(payload), true,
                       "label index", FullHandBundleCodes.ReferenceInvalid);
            }
            finally
            {
                try { Directory.Delete(temporary, true); } catch (Exception) { }
            }
        }

        // ---- helpers -----------------------------------------------------------

        static void AssertNoMismatch(string kind)
        {
            var relevant = new List<string>();
            foreach (CaseReport report in reports)
                if (report.kind == kind)
                    foreach (string mismatch in mismatches)
                        if (mismatch.StartsWith(report.name + ".", StringComparison.Ordinal)
                            && !relevant.Contains(mismatch))
                            relevant.Add(mismatch);
            Assert.AreEqual(0, relevant.Count,
                            kind + " mismatches: " + string.Join(" | ", relevant.ToArray()));
        }

        static GoldenPairCase FindPair(string name)
        {
            foreach (GoldenPairCase entry in golden.Pairs)
                if (entry.Name == name) return entry;
            Assert.Fail("the fixture is missing the pair case " + name);
            return null;
        }

        static void Refuse(string directory, string manifestText, byte[] payload,
                           bool refreshDigest, string label, string expectedCode)
        {
            var manifest = JsonUtility.FromJson<FullHandBundleManifest>(manifestText);
            manifest.payload_bytes = payload.LongLength;
            if (refreshDigest) manifest.payload_sha256 = Sha256Hex(payload);
            string manifestPath = Path.Combine(directory, label.Replace(' ', '_') + ".json");
            string payloadPath = Path.Combine(directory, label.Replace(' ', '_') + ".bin");
            manifest.payload_file = Path.GetFileName(payloadPath);
            File.WriteAllText(manifestPath, JsonUtility.ToJson(manifest), new UTF8Encoding(false));
            File.WriteAllBytes(payloadPath, payload);
            FullHandFormatException refusal = Assert.Throws<FullHandFormatException>(
                delegate { FullHandRuntimeBundle.Load(manifestPath); },
                "a " + label + " fault must be refused");
            Assert.AreEqual(expectedCode, refusal.Code,
                            "the " + label + " fault must report its own code");
        }

        static byte[] Corrupt(byte[] payload, int offset, byte value)
        {
            var copy = (byte[])payload.Clone();
            copy[offset] = value;
            return copy;
        }

        static byte[] Truncate(byte[] payload, int count)
        {
            var copy = new byte[payload.Length - count];
            Array.Copy(payload, copy, copy.Length);
            return copy;
        }

        static byte[] Append(byte[] payload, int count)
        {
            var copy = new byte[payload.Length + count];
            Array.Copy(payload, copy, payload.Length);
            return copy;
        }

        static int HeaderCursorAfterVocabulary(byte[] payload)
        {
            // this walks the BUNDLE payload; its magic is also eight bytes, but borrowing the
            // golden constant made that agreement accidental rather than stated
            int cursor = 8 + 4 + 5 * 4;                       // magic, version, five dimensions
            int vocabulary = ReadInt(payload, ref cursor);
            for (int index = 0; index < vocabulary; index++) ReadText(payload, ref cursor);
            return cursor;
        }

        static byte[] ZeroFirstScale(byte[] payload)
        {
            int cursor = HeaderCursorAfterVocabulary(payload);
            int width = ReadInt(payload, ref cursor);
            int scaleOffset = cursor + width * 8;              // after the mean block
            var copy = (byte[])payload.Clone();
            Array.Copy(BitConverter.GetBytes(0.0), 0, copy, scaleOffset, 8);
            return copy;
        }

        static byte[] BreakFirstLabelIndex(byte[] payload)
        {
            int cursor = HeaderCursorAfterVocabulary(payload);
            for (int model = 0; model < 2; model++)
            {
                int width = ReadInt(payload, ref cursor);
                cursor += width * 8 * 3 + 8;
            }
            ReadInt(payload, ref cursor);                      // reference count
            ReadText(payload, ref cursor);                     // first reference id
            var copy = (byte[])payload.Clone();
            Array.Copy(BitConverter.GetBytes(9999), 0, copy, cursor, 4);
            return copy;
        }

        static int ReadInt(byte[] payload, ref int cursor)
        {
            int value = payload[cursor] | (payload[cursor + 1] << 8)
                      | (payload[cursor + 2] << 16) | (payload[cursor + 3] << 24);
            cursor += 4;
            return value;
        }

        static double ReadDouble(byte[] payload, ref int cursor)
        {
            double value = BitConverter.ToDouble(payload, cursor);
            cursor += 8;
            return value;
        }

        static double[] ReadDoubles(byte[] payload, ref int cursor, int count)
        {
            var values = new double[count];
            for (int index = 0; index < count; index++) values[index] = ReadDouble(payload, ref cursor);
            return values;
        }

        static bool[] ReadMask(byte[] payload, ref int cursor, int count)
        {
            var values = new bool[count];
            for (int index = 0; index < count; index++) values[index] = payload[cursor++] == 1;
            return values;
        }

        static string ReadText(byte[] payload, ref int cursor)
        {
            int length = ReadInt(payload, ref cursor);
            string value = new UTF8Encoding(false).GetString(payload, cursor, length);
            cursor += length;
            return value;
        }

        static int CountTrue(bool[] values)
        {
            int total = 0;
            for (int index = 0; index < values.Length; index++) if (values[index]) total++;
            return total;
        }

        static string Number(double value)
        {
            return value.ToString("R", CultureInfo.InvariantCulture);
        }

        static string Sha256Hex(byte[] data)
        {
            using (SHA256 sha = SHA256.Create())
            {
                byte[] digest = sha.ComputeHash(data);
                var text = new StringBuilder(digest.Length * 2);
                for (int index = 0; index < digest.Length; index++)
                    text.Append(digest[index].ToString("x2"));
                return text.ToString();
            }
        }
    }
}
