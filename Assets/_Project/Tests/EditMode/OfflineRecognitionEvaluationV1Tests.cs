using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.Evaluation;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class OfflineRecognitionEvaluationV1Tests
    {
        const double Tolerance = 1e-12;
        const double FrozenThreshold = 0.2908679008489196;
        const string Step8AHash = "fa8292c424644bc5f50f7213e7587f5157b71cd651df804b8a542168ffe4bf3c";
        const string Step8BHash = "3401ea74f73902c5ea7b4f34abb06ebfe933ef0a076820f15b6471937a5c7c6d";
        static readonly string[] Labels = { "GREETING", "LOVE", "THANK_YOU" };
        static readonly string[] SealedTestIds =
        {
            "S001_P001_GREETING_NORMAL_R01",
            "S001_P001_GREETING_SLOW_R02",
            "S001_P001_GREETING_SLOW_R05",
            "S001_P001_LOVE_NORMAL_R01",
            "S001_P001_LOVE_SLOW_R02",
            "S001_P001_LOVE_SLOW_R05",
            "S001_P001_THANK_YOU_NORMAL_R01",
            "S001_P001_THANK_YOU_SLOW_R02",
            "S001_P001_THANK_YOU_SLOW_R05",
        };

        string ProjectRoot => Directory.GetParent(Application.dataPath).FullName;
        string TemplateDirectory => Path.Combine(Application.dataPath, "StreamingAssets", "RecognitionCoreV1", "Templates");
        string FixtureDirectory => Path.Combine(Application.dataPath, "_Project", "Tests", "EditMode", "Fixtures", "OfflineRecognitionEvaluationV1");
        string OutputDirectory => Path.Combine(ProjectRoot, "data", "processed", "sessions", "S001_P001", "step8c_offline_recognition_evaluation_v1_0");
        string ThresholdPath => Path.Combine(OutputDirectory, "frozen_threshold_config.json");

        RecognitionTemplateLibrary library;
        Dictionary<string, RecognitionSequence> queries;
        TestFixtureManifest fixture;
        FrozenDistanceGateConfig threshold;

        [Serializable]
        sealed class TestFixtureManifest
        {
            public string version;
            public string run_id;
            public string threshold_config_relative_path;
            public long threshold_file_size_bytes;
            public string threshold_file_sha256;
            public string threshold_canonical_payload_sha256;
            public TestQueryRecord[] test_queries;
        }

        [Serializable]
        sealed class TestQueryRecord
        {
            public string sample_id;
            public string word;
            public string speed;
            public string split;
            public int length;
            public int landmark_count;
            public string relative_path;
            public long size_bytes;
            public string file_sha256;
            public string binary_payload_sha256;
            public string source_npz_sha256;
            public string logical_payload_sha256;
            public string coordinates_sha256;
            public string valid_mask_sha256;
        }

        [SetUp]
        public void SetUp()
        {
            fixture = JsonUtility.FromJson<TestFixtureManifest>(File.ReadAllText(Path.Combine(FixtureDirectory, "test_fixture_manifest.json")));
            threshold = FrozenDistanceGateConfigLoader.LoadFile(ThresholdPath, SealedTestIds);
            library = RecognitionTemplateLibrary.LoadDirectory(TemplateDirectory);
            queries = fixture.test_queries
                .Select(record => RecognitionSequenceBinary.LoadFile(ProjectPath(record.relative_path)))
                .ToDictionary(item => item.SampleId, StringComparer.Ordinal);
        }

        [Test]
        public void FrozenThreshold_HasExactValueProvenanceAndInclusiveBoundary()
        {
            Assert.AreEqual(FrozenThreshold, threshold.Threshold, 0.0);
            Assert.AreEqual("validation_only", threshold.SelectionScope);
            Assert.AreEqual("<=", threshold.ComparisonOperator);
            Assert.AreEqual("(validation_max_true + validation_min_wrong) / 2", threshold.Formula);
            Assert.AreEqual(Step8AHash, threshold.Step8ADeliverySha256);
            Assert.AreEqual(Step8BHash, threshold.Step8BDeliverySha256);
            Assert.IsFalse(threshold.NegativeOrUnknownSamplesAvailable);
            Assert.AreEqual("not_evaluated", threshold.OpenSetRejectionPerformance);
            Assert.AreEqual(9, threshold.ThresholdSelectionQueryIds.Count);
            Assert.AreEqual(0, threshold.ThresholdSelectionQueryIds.Intersect(SealedTestIds, StringComparer.Ordinal).Count());

            RecognitionSequence prototype = library.Templates[0];
            var boundaryResult = new RecognitionResult(
                Array.Empty<TemplateDistance>(),
                new[] { new WordDistance(prototype.Word, FrozenThreshold, prototype) },
                prototype.Word,
                prototype);
            Assert.IsTrue(DistanceGatePolicy.Evaluate(boundaryResult, FrozenThreshold).Accepted);
            var aboveResult = new RecognitionResult(
                Array.Empty<TemplateDistance>(),
                new[] { new WordDistance(prototype.Word, BitIncrement(FrozenThreshold), prototype) },
                prototype.Word,
                prototype);
            Assert.IsFalse(DistanceGatePolicy.Evaluate(aboveResult, FrozenThreshold).Accepted);
        }

        [Test]
        public void FrozenThreshold_FileAndCanonicalPayloadBindToFixtureAndRunId()
        {
            Assert.AreEqual(fixture.run_id, threshold.RunId);
            Assert.AreEqual(fixture.threshold_file_size_bytes, new FileInfo(ThresholdPath).Length);
            Assert.AreEqual(fixture.threshold_file_sha256, Sha256File(ThresholdPath));
            Assert.AreEqual(fixture.threshold_canonical_payload_sha256, threshold.CanonicalPayloadSha256);
            Assert.AreEqual("0918c71c-313f-49d0-a5a9-b525f6630317", threshold.Step8ARunId);
            Assert.AreEqual("f1063bbe-c285-4df6-af66-4b984096f330", threshold.Step8BRunId);
        }

        [Test]
        public void TestBinaries_AllNineRoundTripWithExactPayloadAndMetadataHashes()
        {
            Assert.AreEqual(9, fixture.test_queries.Length);
            foreach (TestQueryRecord record in fixture.test_queries)
            {
                string path = ProjectPath(record.relative_path);
                RecognitionSequence sequence = RecognitionSequenceBinary.LoadFile(path);
                Assert.AreEqual(record.sample_id, sequence.SampleId);
                Assert.AreEqual(record.word, sequence.Word);
                Assert.AreEqual(record.speed, sequence.Speed);
                Assert.AreEqual("test", sequence.Split);
                Assert.AreEqual(record.length, sequence.Length);
                Assert.AreEqual(16, sequence.LandmarkCount);
                Assert.AreEqual(record.source_npz_sha256, sequence.SourceNpzSha256);
                Assert.AreEqual(record.logical_payload_sha256, sequence.LogicalPayloadSha256);
                Assert.AreEqual(record.size_bytes, new FileInfo(path).Length);
                Assert.AreEqual(record.file_sha256, Sha256File(path));
                Assert.AreEqual(record.binary_payload_sha256, sequence.BinaryPayloadSha256);
                Assert.AreEqual(record.coordinates_sha256, CoordinatesSha256(sequence));
                Assert.AreEqual(record.valid_mask_sha256, ValidMaskSha256(sequence));
            }
        }

        [Test]
        public void PythonParity_All81TestDistancesCostsAndPathLengthsMatch()
        {
            var templates = library.Templates.ToDictionary(item => item.SampleId, StringComparer.Ordinal);
            int count = 0;
            foreach (string[] row in ReadCsv(Path.Combine(FixtureDirectory, "test_expected_distances.csv")))
            {
                DtwResult actual = MaskAwareDtw.Compute(queries[row[0]], templates[row[3]], RecognitionCore.RelativeBand);
                Assert.AreEqual(ParseDouble(row[6]), actual.Distance, Tolerance, row[0] + " -> " + row[3] + " distance");
                Assert.AreEqual(ParseDouble(row[7]), actual.CumulativeCost, Tolerance, row[0] + " -> " + row[3] + " cumulative");
                Assert.AreEqual(int.Parse(row[8], CultureInfo.InvariantCulture), actual.PathLength, row[0] + " -> " + row[3] + " path");
                count++;
            }
            Assert.AreEqual(81, count);
        }

        [Test]
        public void PythonParity_AllNinePredictionsWinningPrototypesWordDistancesAndGateDecisionsMatch()
        {
            var core = new RecognitionCore(library);
            int count = 0;
            foreach (string[] row in ReadCsv(Path.Combine(FixtureDirectory, "test_expected_predictions.csv")))
            {
                RecognitionResult actual = core.Recognize(queries[row[0]]);
                DistanceGateDecision decision = DistanceGatePolicy.Evaluate(actual, threshold.Threshold);
                var words = actual.WordDistances.ToDictionary(item => item.Word, StringComparer.Ordinal);
                Assert.AreEqual(row[3], actual.PredictedWord, row[0]);
                Assert.AreEqual(row[5], actual.WinningPrototype.SampleId, row[0]);
                Assert.AreEqual(ParseDouble(row[6]), words["GREETING"].Distance, Tolerance, row[0]);
                Assert.AreEqual(ParseDouble(row[7]), words["LOVE"].Distance, Tolerance, row[0]);
                Assert.AreEqual(ParseDouble(row[8]), words["THANK_YOU"].Distance, Tolerance, row[0]);
                Assert.AreEqual(ParseDouble(row[9]), decision.BestWordDistance, Tolerance, row[0]);
                Assert.AreEqual(bool.Parse(row[13]), decision.Accepted, row[0]);
                Assert.AreEqual(row[3], decision.PredictedWord, row[0]);
                Assert.AreEqual(row[5], decision.WinningPrototype.SampleId, row[0]);
                count++;
            }
            Assert.AreEqual(9, count);
        }

        [Test]
        public void ClosedSetMetrics_AndConfusionMatrixRecomputeExactlyFromCSharpResults()
        {
            var core = new RecognitionCore(library);
            var truth = fixture.test_queries.ToDictionary(record => record.sample_id, record => record.word, StringComparer.Ordinal);
            var matrix = new int[3, 3];
            foreach (string sampleId in SealedTestIds)
            {
                RecognitionResult result = core.Recognize(queries[sampleId]);
                matrix[Array.IndexOf(Labels, truth[sampleId]), Array.IndexOf(Labels, result.PredictedWord)]++;
            }
            int correct = Enumerable.Range(0, 3).Sum(index => matrix[index, index]);
            Assert.AreEqual(1.0, correct / 9.0, 0.0);
            for (int row = 0; row < 3; row++)
            for (int column = 0; column < 3; column++)
                Assert.AreEqual(row == column ? 3 : 0, matrix[row, column]);
            double macroF1 = Enumerable.Range(0, 3).Average(index => ClassF1(matrix, index));
            Assert.AreEqual(1.0, macroF1, 0.0);

            string metrics = File.ReadAllText(Path.Combine(FixtureDirectory, "closed_set_metrics.json"));
            Assert.AreEqual(1.0, JsonNumber(metrics, "accuracy"), 0.0);
            Assert.AreEqual(1.0, JsonNumber(metrics, "macro_f1"), 0.0);
            Assert.IsTrue(Regex.IsMatch(metrics, "\\\"confusion_matrix\\\"\\s*:\\s*\\[\\s*\\[\\s*3\\s*,\\s*0\\s*,\\s*0\\s*\\]\\s*,\\s*\\[\\s*0\\s*,\\s*3\\s*,\\s*0\\s*\\]\\s*,\\s*\\[\\s*0\\s*,\\s*0\\s*,\\s*3\\s*\\]\\s*\\]", RegexOptions.CultureInvariant));
        }

        [Test]
        public void GateMetrics_RecomputeNineAcceptedAndEffectiveSuccessWithNoNegatives()
        {
            var core = new RecognitionCore(library);
            var truth = fixture.test_queries.ToDictionary(record => record.sample_id, record => record.word, StringComparer.Ordinal);
            int accepted = 0;
            int effective = 0;
            foreach (string sampleId in SealedTestIds)
            {
                RecognitionResult result = core.Recognize(queries[sampleId]);
                DistanceGateDecision decision = DistanceGatePolicy.Evaluate(result, threshold.Threshold);
                if (decision.Accepted) accepted++;
                if (decision.Accepted && result.PredictedWord == truth[sampleId]) effective++;
            }
            Assert.AreEqual(9, accepted);
            Assert.AreEqual(9, effective);
            string metrics = File.ReadAllText(Path.Combine(FixtureDirectory, "distance_gate_metrics.json"));
            Assert.AreEqual(0.0, JsonNumber(metrics, "unknown_negative_count"), 0.0);
            Assert.AreEqual(1.0, JsonNumber(metrics, "known_test_acceptance_rate"), 0.0);
            Assert.AreEqual(1.0, JsonNumber(metrics, "correct_and_accepted_rate"), 0.0);
            Assert.IsTrue(metrics.Contains("\"open_set_rejection_performance\": \"not_evaluated\""));
        }

        [Test]
        public void Leakage_TestIdsAreExactAndSplitsNeverCrossTrainValidationOrTest()
        {
            CollectionAssert.AreEqual(SealedTestIds, fixture.test_queries.Select(record => record.sample_id).ToArray());
            Assert.IsTrue(queries.Values.All(item => item.Split == "test"));
            Assert.IsTrue(library.Templates.All(item => item.Split == "train"));
            Assert.AreEqual(0, queries.Keys.Intersect(library.Templates.Select(item => item.SampleId), StringComparer.Ordinal).Count());
            string validationDirectory = Path.Combine(Application.dataPath, "_Project", "Tests", "EditMode", "Fixtures", "RecognitionCoreV1", "ValidationQueries");
            var validationIds = Directory.GetFiles(validationDirectory, "*.wisrq").Select(RecognitionSequenceBinary.LoadFile).Select(item => item.SampleId).ToArray();
            Assert.AreEqual(0, queries.Keys.Intersect(validationIds, StringComparer.Ordinal).Count());
        }

        [Test]
        public void FrozenThreshold_RemainsUnchangedAfterEvaluation()
        {
            Assert.AreEqual(fixture.threshold_file_size_bytes, new FileInfo(ThresholdPath).Length);
            Assert.AreEqual(fixture.threshold_file_sha256, Sha256File(ThresholdPath));
            string audit = File.ReadAllText(Path.Combine(OutputDirectory, "leakage_audit.json"));
            Assert.IsTrue(Regex.IsMatch(audit, "\\\"frozen_threshold_unchanged\\\"\\s*:\\s*true", RegexOptions.CultureInvariant));
            Assert.IsTrue(Regex.IsMatch(audit, "\\\"test_feature_npz_reads\\\"\\s*:\\s*0", RegexOptions.CultureInvariant));
            Assert.IsTrue(Regex.IsMatch(audit, "\\\"configuration_updates_after_test\\\"\\s*:\\s*0", RegexOptions.CultureInvariant));
        }

        [Test]
        public void ThresholdLoader_RejectsMalformedConfigChangedHashAndTestSelectedThreshold()
        {
            byte[] original = File.ReadAllBytes(ThresholdPath);
            Assert.Throws<InvalidDataException>(() => FrozenDistanceGateConfigLoader.Load(Encoding.UTF8.GetBytes("{}\n"), SealedTestIds));
            byte[] changedHash = (byte[])original.Clone();
            changedHash[changedHash.Length - 4] = changedHash[changedHash.Length - 4] == (byte)'0' ? (byte)'1' : (byte)'0';
            Assert.Throws<InvalidDataException>(() => FrozenDistanceGateConfigLoader.Load(changedHash, SealedTestIds));
            Assert.Throws<InvalidDataException>(() => FrozenDistanceGateConfigLoader.ValidateSelectionDoesNotUseTestIds(
                threshold.ThresholdSelectionQueryIds.Concat(new[] { SealedTestIds[0] }), SealedTestIds));
        }

        string ProjectPath(string relativePath) => Path.Combine(ProjectRoot, relativePath.Replace('/', Path.DirectorySeparatorChar));

        static IEnumerable<string[]> ReadCsv(string path)
        {
            string[] lines = File.ReadAllLines(path);
            for (int index = 1; index < lines.Length; index++)
                if (!string.IsNullOrWhiteSpace(lines[index])) yield return lines[index].Split(',');
        }

        static double ParseDouble(string value) => double.Parse(value, NumberStyles.Float, CultureInfo.InvariantCulture);

        static double JsonNumber(string json, string field)
        {
            Match match = Regex.Match(json, "\\\"" + Regex.Escape(field) + "\\\"\\s*:\\s*(-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", RegexOptions.CultureInvariant);
            Assert.IsTrue(match.Success, field + " missing");
            return ParseDouble(match.Groups[1].Value);
        }

        static double ClassF1(int[,] matrix, int index)
        {
            int tp = matrix[index, index];
            int actual = Enumerable.Range(0, 3).Sum(column => matrix[index, column]);
            int predicted = Enumerable.Range(0, 3).Sum(row => matrix[row, index]);
            double precision = predicted == 0 ? 0.0 : (double)tp / predicted;
            double recall = actual == 0 ? 0.0 : (double)tp / actual;
            return precision + recall == 0.0 ? 0.0 : 2.0 * precision * recall / (precision + recall);
        }

        static string Sha256File(string path)
        {
            using (var stream = File.OpenRead(path))
            using (var sha = SHA256.Create()) return RecognitionSequenceBinary.ToLowerHex(sha.ComputeHash(stream));
        }

        static string CoordinatesSha256(RecognitionSequence sequence)
        {
            using (var stream = new MemoryStream())
            using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
            {
                foreach (double value in sequence.CopyCoordinates()) writer.Write(value);
                writer.Flush();
                using (var sha = SHA256.Create()) return RecognitionSequenceBinary.ToLowerHex(sha.ComputeHash(stream.ToArray()));
            }
        }

        static string ValidMaskSha256(RecognitionSequence sequence)
        {
            bool[] mask = sequence.CopyValidMask();
            var bytes = new byte[mask.Length];
            for (int index = 0; index < mask.Length; index++) bytes[index] = mask[index] ? (byte)1 : (byte)0;
            using (var sha = SHA256.Create()) return RecognitionSequenceBinary.ToLowerHex(sha.ComputeHash(bytes));
        }

        static double BitIncrement(double value)
        {
            long bits = BitConverter.DoubleToInt64Bits(value);
            return BitConverter.Int64BitsToDouble(bits + 1);
        }
    }
}
