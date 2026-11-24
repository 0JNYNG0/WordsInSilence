using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Recognition.Core;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class RecognitionCoreV1Tests
    {
        const double ParityTolerance = 1e-12;
        const string ZeroSha = "0000000000000000000000000000000000000000000000000000000000000000";

        string TemplateDirectory => Path.Combine(Application.dataPath, "StreamingAssets", "RecognitionCoreV1", "Templates");
        string FixtureDirectory => Path.Combine(Application.dataPath, "_Project", "Tests", "EditMode", "Fixtures", "RecognitionCoreV1");
        RecognitionTemplateLibrary library;
        Dictionary<string, RecognitionSequence> queries;

        [Serializable]
        sealed class RuntimeTemplateManifest
        {
            public string run_id;
            public RuntimeTemplateRecord[] templates;
        }

        [Serializable]
        sealed class RuntimeTemplateRecord
        {
            public string sample_id;
            public string word;
            public string speed;
            public string split;
            public string relative_path;
            public string file_sha256;
            public string binary_payload_sha256;
            public string source_npz_sha256;
            public string logical_payload_sha256;
            public string coordinates_sha256;
            public string valid_mask_sha256;
            public int length;
        }

        [Serializable]
        sealed class FixtureManifest
        {
            public string run_id;
            public FixtureQueryRecord[] validation_queries;
        }

        [Serializable]
        sealed class FixtureQueryRecord
        {
            public string sample_id;
            public string split;
            public string relative_path;
            public string file_sha256;
            public string logical_payload_sha256;
        }

        [Serializable]
        sealed class SchemaHeader
        {
            public string run_id;
            public int format_version;
        }

        [SetUp]
        public void SetUp()
        {
            library = RecognitionTemplateLibrary.LoadDirectory(TemplateDirectory);
            queries = Directory.GetFiles(Path.Combine(FixtureDirectory, "ValidationQueries"), "*.wisrq")
                .Select(RecognitionSequenceBinary.LoadFile)
                .ToDictionary(item => item.SampleId, StringComparer.Ordinal);
        }

        [Test]
        public void RuntimeTemplateExport_RoundTripMatchesApprovedPayloadMetadataAndHashes()
        {
            string manifestPath = Path.Combine(FixtureDirectory, "runtime_template_manifest.json");
            var manifest = JsonUtility.FromJson<RuntimeTemplateManifest>(File.ReadAllText(manifestPath));
            Assert.AreEqual(9, manifest.templates.Length);
            foreach (RuntimeTemplateRecord record in manifest.templates)
            {
                string path = ProjectPath(record.relative_path);
                RecognitionSequence sequence = RecognitionSequenceBinary.LoadFile(path);
                Assert.AreEqual(record.sample_id, sequence.SampleId);
                Assert.AreEqual(record.word, sequence.Word);
                Assert.AreEqual(record.speed, sequence.Speed);
                Assert.AreEqual(record.split, sequence.Split);
                Assert.AreEqual(record.length, sequence.Length);
                Assert.AreEqual(record.source_npz_sha256, sequence.SourceNpzSha256);
                Assert.AreEqual(record.logical_payload_sha256, sequence.LogicalPayloadSha256);
                Assert.AreEqual(record.file_sha256, Sha256File(path));
                Assert.AreEqual(record.binary_payload_sha256, sequence.BinaryPayloadSha256);
                Assert.AreEqual(record.coordinates_sha256, CoordinatesSha256(sequence));
                Assert.AreEqual(record.valid_mask_sha256, ValidMaskSha256(sequence));
            }
        }

        [Test]
        public void Manifests_UseOneRunIdAndSupportedFormatVersion()
        {
            var runtime = JsonUtility.FromJson<RuntimeTemplateManifest>(File.ReadAllText(Path.Combine(FixtureDirectory, "runtime_template_manifest.json")));
            var fixtures = JsonUtility.FromJson<FixtureManifest>(File.ReadAllText(Path.Combine(FixtureDirectory, "fixture_manifest.json")));
            var schema = JsonUtility.FromJson<SchemaHeader>(File.ReadAllText(Path.Combine(FixtureDirectory, "runtime_template_binary_schema.json")));
            Assert.AreEqual(runtime.run_id, fixtures.run_id);
            Assert.AreEqual(runtime.run_id, schema.run_id);
            Assert.AreEqual(1, schema.format_version);
        }

        [Test]
        public void TemplateLibrary_LoadsNineUniqueWordSpeedGroups()
        {
            Assert.AreEqual(9, library.Templates.Count);
            Assert.AreEqual(9, library.Templates.Select(item => item.Word + "__" + item.Speed).Distinct().Count());
            Assert.IsTrue(library.Templates.All(item => item.Split == "train"));
        }

        [Test]
        public void LocalCost_BothValidUsesThreeDimensionalEuclideanDistance()
        {
            RecognitionSequence left = SingleFrame(new[] { true }, new[] { 0.0, 0.0, 0.0 });
            RecognitionSequence right = SingleFrame(new[] { true }, new[] { 3.0, 4.0, 0.0 });
            Assert.AreEqual(5.0, MaskAwareDtw.LocalCost(left, 0, right, 0), 0.0);
        }

        [Test]
        public void LocalCost_ExactlyOneValidUsesFixedMissingCost()
        {
            RecognitionSequence left = SingleFrame(new[] { true }, new[] { 0.0, 0.0, 0.0 });
            RecognitionSequence right = SingleFrame(new[] { false }, new[] { double.NaN, double.NaN, double.NaN });
            Assert.AreEqual(1.0, MaskAwareDtw.LocalCost(left, 0, right, 0), 0.0);
        }

        [Test]
        public void LocalCost_BothInvalidLandmarkIsExcludedWhenAnotherIsComparable()
        {
            RecognitionSequence left = SingleFrame(new[] { true, false }, new[] { 0.0, 0.0, 0.0, double.NaN, double.NaN, double.NaN });
            RecognitionSequence right = SingleFrame(new[] { true, false }, new[] { 0.0, 0.0, 0.0, double.NaN, double.NaN, double.NaN });
            Assert.AreEqual(0.0, MaskAwareDtw.LocalCost(left, 0, right, 0), 0.0);
        }

        [Test]
        public void LocalCost_NoComparableLandmarksUsesOne()
        {
            RecognitionSequence left = SingleFrame(Array.Empty<bool>(), Array.Empty<double>());
            RecognitionSequence right = SingleFrame(Array.Empty<bool>(), Array.Empty<double>());
            Assert.AreEqual(1.0, MaskAwareDtw.LocalCost(left, 0, right, 0), 0.0);
        }

        [Test]
        public void Dtw_PathHasEndpointsAllowedStepsBandAndPathLengthNormalization()
        {
            RecognitionSequence left = FullyComparableSequence("left", 4, 0.0);
            RecognitionSequence right = FullyComparableSequence("right", 4, 0.2);
            DtwResult result = MaskAwareDtw.Compute(left, right, RecognitionCore.RelativeBand);
            Assert.AreEqual(0, result.Path[0].I);
            Assert.AreEqual(0, result.Path[0].J);
            Assert.AreEqual(left.Length - 1, result.Path[result.Path.Count - 1].I);
            Assert.AreEqual(right.Length - 1, result.Path[result.Path.Count - 1].J);
            for (int index = 1; index < result.Path.Count; index++)
            {
                int di = result.Path[index].I - result.Path[index - 1].I;
                int dj = result.Path[index].J - result.Path[index - 1].J;
                Assert.IsTrue((di == 1 && dj == 1) || (di == 1 && dj == 0) || (di == 0 && dj == 1));
            }
            foreach (DtwPathPoint point in result.Path)
                Assert.IsTrue(MaskAwareDtw.IsWithinRelativeBand(point.I, point.J, left.Length, right.Length, RecognitionCore.RelativeBand));
            Assert.AreEqual(result.CumulativeCost / result.PathLength, result.Distance, 1e-15);
        }

        [Test]
        public void Dtw_PredecessorTieOrderPrefersDiagonal()
        {
            RecognitionSequence left = ConstantFullyComparableSequence("left", 2);
            RecognitionSequence right = ConstantFullyComparableSequence("right", 3);
            DtwResult result = MaskAwareDtw.Compute(left, right, 1.0);
            DtwPathPoint beforeEnd = result.Path[result.Path.Count - 2];
            Assert.AreEqual(0, beforeEnd.I);
            Assert.AreEqual(1, beforeEnd.J);
        }

        [Test]
        public void Dtw_RepresentativePairIsSymmetric()
        {
            RecognitionSequence left = library.Templates[0];
            RecognitionSequence right = library.Templates[4];
            double forward = MaskAwareDtw.Compute(left, right, RecognitionCore.RelativeBand).Distance;
            double reverse = MaskAwareDtw.Compute(right, left, RecognitionCore.RelativeBand).Distance;
            Assert.AreEqual(forward, reverse, ParityTolerance);
        }

        [Test]
        public void Dtw_OrdinaryIdenticalFullyComparableSequenceIsZero()
        {
            RecognitionSequence sequence = FullyComparableSequence("ordinary", 5, 0.35);
            Assert.AreEqual(0.0, MaskAwareDtw.Compute(sequence, sequence, RecognitionCore.RelativeBand).Distance, 1e-15);
        }

        [Test]
        public void Dtw_AllInvalidFramePreservesFrozenNonzeroSelfBehavior()
        {
            RecognitionSequence sequence = SingleFrame(Array.Empty<bool>(), Array.Empty<double>());
            Assert.AreEqual(1.0, MaskAwareDtw.Compute(sequence, sequence, RecognitionCore.RelativeBand).Distance, 0.0);
        }

        [Test]
        public void PythonParity_All81ValidationTemplateDistancesMatch()
        {
            var templates = library.Templates.ToDictionary(item => item.SampleId, StringComparer.Ordinal);
            var mismatches = new List<string>();
            foreach (string[] row in ReadCsv(Path.Combine(FixtureDirectory, "validation_expected_distances.csv")))
            {
                RecognitionSequence query = queries[row[0]];
                RecognitionSequence prototype = templates[row[1]];
                double expected = double.Parse(row[2], CultureInfo.InvariantCulture);
                double actual = MaskAwareDtw.Compute(query, prototype, RecognitionCore.RelativeBand).Distance;
                if (Math.Abs(expected - actual) > ParityTolerance)
                    mismatches.Add($"{row[0]}->{row[1]} expected={expected:R} actual={actual:R}");
            }
            Assert.AreEqual(0, mismatches.Count, string.Join("\n", mismatches));
        }

        [Test]
        public void PythonParity_AllNinePredictionsAndWinningWordDistancesMatch()
        {
            var core = new RecognitionCore(library);
            var mismatches = new List<string>();
            foreach (string[] row in ReadCsv(Path.Combine(FixtureDirectory, "validation_expected_predictions.csv")))
            {
                RecognitionResult actual = core.Recognize(queries[row[0]]);
                if (actual.PredictedWord != row[2]) mismatches.Add($"{row[0]} predicted {actual.PredictedWord}, expected {row[2]}");
                if (actual.WinningPrototype.SampleId != row[3]) mismatches.Add($"{row[0]} prototype {actual.WinningPrototype.SampleId}, expected {row[3]}");
                var wordMap = actual.WordDistances.ToDictionary(item => item.Word, StringComparer.Ordinal);
                CompareDistance(row[0], "GREETING", row[4], wordMap, mismatches);
                CompareDistance(row[0], "LOVE", row[5], wordMap, mismatches);
                CompareDistance(row[0], "THANK_YOU", row[6], wordMap, mismatches);
            }
            Assert.AreEqual(0, mismatches.Count, string.Join("\n", mismatches));
        }

        [Test]
        public void Leakage_RuntimeFixturesAndExpectedRecordsContainOnlyTrainOrValidationPayloads()
        {
            Assert.IsTrue(library.Templates.All(item => item.Split == "train"));
            Assert.IsTrue(queries.Values.All(item => item.Split == "val"));
            var allowed = new HashSet<string>(library.Templates.Select(item => item.SampleId).Concat(queries.Keys), StringComparer.Ordinal);
            foreach (string[] row in ReadCsv(Path.Combine(FixtureDirectory, "validation_expected_distances.csv")))
            {
                Assert.IsTrue(allowed.Contains(row[0]));
                Assert.IsTrue(allowed.Contains(row[1]));
            }
            foreach (string[] row in ReadCsv(Path.Combine(FixtureDirectory, "validation_expected_predictions.csv")))
                Assert.IsTrue(allowed.Contains(row[0]));
            Assert.AreEqual(18, allowed.Count);
        }

        [Test]
        public void BinaryLoader_DetectsCorruptMagic()
        {
            byte[] data = FirstTemplateBytes();
            data[0] ^= 0x01;
            Rehash(data);
            Assert.Throws<InvalidDataException>(() => RecognitionSequenceBinary.Load(data));
        }

        [Test]
        public void BinaryLoader_DetectsUnsupportedVersion()
        {
            byte[] data = FirstTemplateBytes();
            data[8] = 2; data[9] = 0; data[10] = 0; data[11] = 0;
            Rehash(data);
            Assert.Throws<InvalidDataException>(() => RecognitionSequenceBinary.Load(data));
        }

        [Test]
        public void BinaryLoader_DetectsTruncatedData()
        {
            byte[] data = FirstTemplateBytes();
            Array.Resize(ref data, data.Length - 17);
            Assert.Throws<InvalidDataException>(() => RecognitionSequenceBinary.Load(data));
        }

        [Test]
        public void BinaryLoader_DetectsPayloadHashMismatch()
        {
            byte[] data = FirstTemplateBytes();
            data[data.Length - 1] ^= 0x01;
            Assert.Throws<InvalidDataException>(() => RecognitionSequenceBinary.Load(data));
        }

        [Test]
        public void TemplateLibrary_DetectsDuplicateGroup()
        {
            var values = library.Templates.ToList();
            values[values.Count - 1] = values[0];
            Assert.Throws<InvalidDataException>(() => new RecognitionTemplateLibrary(values));
        }

        [Test]
        public void TemplateLibrary_DetectsMissingGroup()
        {
            Assert.Throws<InvalidDataException>(() => new RecognitionTemplateLibrary(library.Templates.Take(8)));
        }

        [Test]
        public void TemplateLibrary_DetectsInjectedTestSplit()
        {
            var values = library.Templates.ToList();
            RecognitionSequence original = values[0];
            values[0] = new RecognitionSequence("SYNTHETIC_TEST_INJECTION", original.Word, original.Speed, "test",
                original.SourceNpzSha256, original.LogicalPayloadSha256, original.Length, original.LandmarkCount,
                original.CopyCoordinates(), original.CopyValidMask());
            Assert.Throws<InvalidDataException>(() => new RecognitionTemplateLibrary(values));
        }

        [Test]
        public void Sequence_RejectsEmptyWrongShapeAndNonfiniteValidCoordinates()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new RecognitionSequence("x", "GREETING", "FAST", "val", ZeroSha, ZeroSha, 0, 16, Array.Empty<double>(), Array.Empty<bool>()));
            Assert.Throws<ArgumentOutOfRangeException>(() => new RecognitionSequence("x", "GREETING", "FAST", "val", ZeroSha, ZeroSha, 1, 15, new double[45], new bool[15]));
            Assert.Throws<ArgumentException>(() => new RecognitionSequence("x", "GREETING", "FAST", "val", ZeroSha, ZeroSha, 1, 16, new double[47], new bool[16]));
            var coordinates = new double[48]; coordinates[0] = double.NaN;
            var mask = new bool[16]; mask[0] = true;
            Assert.Throws<ArgumentException>(() => new RecognitionSequence("x", "GREETING", "FAST", "val", ZeroSha, ZeroSha, 1, 16, coordinates, mask));
        }

        [Test]
        public void Dtw_RejectsBandWithoutEndpointPath()
        {
            RecognitionSequence left = FullyComparableSequence("left", 2, 0.0);
            RecognitionSequence right = FullyComparableSequence("right", 3, 0.0);
            Assert.Throws<InvalidOperationException>(() => MaskAwareDtw.Compute(left, right, 0.0));
        }

        static void CompareDistance(string sampleId, string word, string expectedText,
            Dictionary<string, WordDistance> actual, List<string> mismatches)
        {
            double expected = double.Parse(expectedText, CultureInfo.InvariantCulture);
            if (Math.Abs(expected - actual[word].Distance) > ParityTolerance)
                mismatches.Add($"{sampleId} {word} expected={expected:R} actual={actual[word].Distance:R}");
        }

        string ProjectPath(string relativePath)
        {
            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            return Path.Combine(projectRoot, relativePath.Replace('/', Path.DirectorySeparatorChar));
        }

        byte[] FirstTemplateBytes() => File.ReadAllBytes(Directory.GetFiles(TemplateDirectory, "*.wisrt").OrderBy(path => path, StringComparer.Ordinal).First());

        static void Rehash(byte[] data)
        {
            int payloadLength = data.Length - 32;
            byte[] hash;
            using (var sha = SHA256.Create()) hash = sha.ComputeHash(data, 0, payloadLength);
            Buffer.BlockCopy(hash, 0, data, payloadLength, 32);
        }

        static RecognitionSequence SingleFrame(bool[] specifiedValidity, double[] specifiedCoordinates)
        {
            var coordinates = Enumerable.Repeat(double.NaN, 16 * 3).ToArray();
            var mask = new bool[16];
            for (int landmark = 0; landmark < specifiedValidity.Length; landmark++)
            {
                mask[landmark] = specifiedValidity[landmark];
                for (int axis = 0; axis < 3; axis++) coordinates[landmark * 3 + axis] = specifiedCoordinates[landmark * 3 + axis];
            }
            return new RecognitionSequence("single", "GREETING", "FAST", "val", ZeroSha, ZeroSha, 1, 16, coordinates, mask);
        }

        static RecognitionSequence FullyComparableSequence(string id, int length, double offset)
        {
            var coordinates = new double[length * 16 * 3];
            var mask = Enumerable.Repeat(true, length * 16).ToArray();
            for (int frame = 0; frame < length; frame++)
            for (int landmark = 0; landmark < 16; landmark++)
            {
                int index = (frame * 16 + landmark) * 3;
                coordinates[index] = offset + frame * 0.01 + landmark * 0.001;
                coordinates[index + 1] = offset - frame * 0.02;
                coordinates[index + 2] = landmark * 0.003;
            }
            return new RecognitionSequence(id, "GREETING", "FAST", "val", ZeroSha, ZeroSha, length, 16, coordinates, mask);
        }

        static RecognitionSequence ConstantFullyComparableSequence(string id, int length)
        {
            return new RecognitionSequence(
                id, "GREETING", "FAST", "val", ZeroSha, ZeroSha, length, 16,
                new double[length * 16 * 3], Enumerable.Repeat(true, length * 16).ToArray());
        }

        static IEnumerable<string[]> ReadCsv(string path)
        {
            string[] lines = File.ReadAllLines(path);
            for (int index = 1; index < lines.Length; index++)
            {
                if (!string.IsNullOrWhiteSpace(lines[index])) yield return lines[index].Split(',');
            }
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
            for (int i = 0; i < mask.Length; i++) bytes[i] = mask[i] ? (byte)1 : (byte)0;
            using (var sha = SHA256.Create()) return RecognitionSequenceBinary.ToLowerHex(sha.ComputeHash(bytes));
        }
    }
}
