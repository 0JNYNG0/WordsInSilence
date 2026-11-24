using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.Evaluation;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class RuntimeRecognitionServiceV1Tests
    {
        const double Tolerance = 1e-12;
        const string ThresholdFileSha = "2697bb1d0c301850d2abda41351519c66d0eb8e27841263469175eb3fa7e99e3";
        const string ThresholdCanonicalSha = "8600a9529c82a4d30dea984515bc56f6ade74c12331a3df3fcf59f3ffcf7f7c0";

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
        string RuntimeGateDirectory => Path.Combine(Application.dataPath, "StreamingAssets", "RecognitionCoreV1", "RuntimeGate");
        string RuntimeThresholdPath => Path.Combine(RuntimeGateDirectory, "frozen_threshold_config.json");
        string ApprovedThresholdPath => Path.Combine(ProjectRoot, "data", "processed", "sessions", "S001_P001", "step8c_offline_recognition_evaluation_v1_0", "frozen_threshold_config.json");
        string ReplayFixtureDirectory => Path.Combine(Application.dataPath, "_Project", "Tests", "EditMode", "Fixtures", "RuntimeRecognitionServiceV1");

        RuntimeRecognitionService service;
        TestFixtureManifest fixture;

        [Serializable]
        sealed class RuntimeAssetManifest
        {
            public string version;
            public string run_id;
            public int template_count;
            public ThresholdAsset threshold;
        }

        [Serializable]
        sealed class ThresholdAsset
        {
            public string relative_path;
            public long size_bytes;
            public string file_sha256;
            public string canonical_payload_sha256;
            public string approved_source_relative_path;
        }

        [Serializable]
        sealed class TestFixtureManifest
        {
            public string run_id;
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
            public string relative_path;
            public string source_npz_sha256;
            public string logical_payload_sha256;
            public string coordinates_sha256;
            public string valid_mask_sha256;
        }

        [SetUp]
        public void SetUp()
        {
            service = RuntimeRecognitionService.Load(TemplateDirectory, RuntimeThresholdPath);
            fixture = JsonUtility.FromJson<TestFixtureManifest>(
                File.ReadAllText(Path.Combine(ReplayFixtureDirectory, "replay_fixture_manifest.json")));
        }

        [Test]
        public void RuntimeThresholdCopy_IsByteExactStrictlyLoadedAndManifestBound()
        {
            byte[] approved = File.ReadAllBytes(ApprovedThresholdPath);
            byte[] runtime = File.ReadAllBytes(RuntimeThresholdPath);
            CollectionAssert.AreEqual(approved, runtime);
            Assert.AreEqual(4590, runtime.Length);
            Assert.AreEqual(ThresholdFileSha, Sha256(runtime));
            FrozenDistanceGateConfig config = FrozenDistanceGateConfigLoader.Load(runtime, SealedTestIds);
            Assert.AreEqual(ThresholdCanonicalSha, config.CanonicalPayloadSha256);
            Assert.AreEqual(0.2908679008489196, config.Threshold, 0.0);

            var manifest = JsonUtility.FromJson<RuntimeAssetManifest>(
                File.ReadAllText(Path.Combine(RuntimeGateDirectory, "runtime_recognition_assets.json")));
            Assert.IsFalse(string.IsNullOrWhiteSpace(manifest.run_id));
            Assert.AreEqual(9, manifest.template_count);
            Assert.AreEqual(runtime.Length, manifest.threshold.size_bytes);
            Assert.AreEqual(ThresholdFileSha, manifest.threshold.file_sha256);
            Assert.AreEqual(ThresholdCanonicalSha, manifest.threshold.canonical_payload_sha256);
            Assert.AreEqual(Sha256(runtime), manifest.threshold.file_sha256);
        }

        [Test]
        public void Service_InitializesExactlyNineApprovedImmutableTemplatesAndFrozenGate()
        {
            Assert.AreEqual(9, service.TemplateCount);
            Assert.AreEqual(9, service.TemplateIds.Distinct(StringComparer.Ordinal).Count());
            Assert.AreEqual(0.2908679008489196, service.Threshold, 0.0);
            Assert.AreEqual(ThresholdCanonicalSha, service.ThresholdCanonicalPayloadSha256);
            Assert.IsFalse(typeof(RuntimeRecognitionService).GetProperty(nameof(RuntimeRecognitionService.Threshold)).CanWrite);
            Assert.Throws<NotSupportedException>(() => ((IList<string>)service.TemplateIds)[0] = "changed");
        }

        [Test]
        public void Replay_AllNineQueriesReconstructExactlyAndMatch81DistancesAndNineDecisions()
        {
            Dictionary<string, string[]> predictions = ReadCsv(Path.Combine(ReplayFixtureDirectory, "test_expected_predictions.csv"))
                .ToDictionary(row => row[0], StringComparer.Ordinal);
            Dictionary<string, double> distances = ReadCsv(Path.Combine(ReplayFixtureDirectory, "test_expected_distances.csv"))
                .ToDictionary(row => row[0] + "|" + row[3], row => ParseDouble(row[6]), StringComparer.Ordinal);
            int reconstructedFrames = 0;
            int distanceCount = 0;
            int decisionCount = 0;

            foreach (TestQueryRecord record in fixture.test_queries)
            {
                RecognitionSequence source = RecognitionSequenceBinary.LoadFile(ProjectPath(record.relative_path));
                var session = new RecognitionSequenceInputSession();
                session.Begin(RuntimeSequenceMetadata.FixtureReplay(source));
                for (int frame = 0; frame < source.Length; frame++)
                {
                    var coordinates = new double[NormalizedRecognitionFrame.CoordinateCount];
                    var mask = new bool[NormalizedRecognitionFrame.LandmarkCount];
                    for (int landmark = 0; landmark < source.LandmarkCount; landmark++)
                    {
                        mask[landmark] = source.IsValid(frame, landmark);
                        for (int axis = 0; axis < 3; axis++)
                            coordinates[landmark * 3 + axis] = source.Coordinate(frame, landmark, axis);
                    }
                    session.AddFrame(source.SampleId, new NormalizedRecognitionFrame(frame / 30.0, coordinates, mask));
                }
                Assert.AreEqual(source.Length, session.FrameCount);
                RecognitionSequence replay = session.FinalizeSequence(source.SampleId);
                Assert.AreEqual(RuntimeSequenceSessionState.Finalized, session.State);
                Assert.AreEqual(source.Length, replay.Length);
                Assert.AreEqual(record.sample_id, replay.SampleId);
                Assert.AreEqual(record.word, replay.Word);
                Assert.AreEqual(record.speed, replay.Speed);
                Assert.AreEqual("test", replay.Split);
                Assert.AreEqual(record.source_npz_sha256, replay.SourceNpzSha256);
                Assert.AreEqual(record.logical_payload_sha256, replay.LogicalPayloadSha256);
                Assert.AreEqual(record.coordinates_sha256, CoordinatesSha256(replay));
                Assert.AreEqual(record.valid_mask_sha256, ValidMaskSha256(replay));
                CollectionAssert.AreEqual(source.CopyCoordinates(), replay.CopyCoordinates());
                CollectionAssert.AreEqual(source.CopyValidMask(), replay.CopyValidMask());

                RuntimeRecognitionResult result = service.Recognize(replay.SampleId, replay);
                string[] expected = predictions[record.sample_id];
                Assert.AreEqual(expected[3], result.PredictedWord, record.sample_id);
                Assert.AreEqual(expected[5], result.WinningPrototypeId, record.sample_id);
                Assert.AreEqual(ParseDouble(expected[6]), result.GreetingDistance, Tolerance, record.sample_id);
                Assert.AreEqual(ParseDouble(expected[7]), result.LoveDistance, Tolerance, record.sample_id);
                Assert.AreEqual(ParseDouble(expected[8]), result.ThankYouDistance, Tolerance, record.sample_id);
                Assert.AreEqual(ParseDouble(expected[9]), result.BestDistance, Tolerance, record.sample_id);
                Assert.AreEqual(ParseDouble(expected[12]), result.Threshold, 0.0, record.sample_id);
                Assert.AreEqual(bool.Parse(expected[13]), result.Accepted, record.sample_id);
                Assert.AreEqual(9, result.TemplateDistances.Count);
                foreach (RuntimeTemplateDistance actual in result.TemplateDistances)
                {
                    Assert.AreEqual(distances[record.sample_id + "|" + actual.PrototypeId], actual.Distance, Tolerance,
                        record.sample_id + " -> " + actual.PrototypeId);
                    distanceCount++;
                }
                reconstructedFrames += replay.Length;
                decisionCount++;
            }

            Assert.AreEqual(9, decisionCount);
            Assert.AreEqual(81, distanceCount);
            Assert.AreEqual(fixture.test_queries.Sum(record => record.length), reconstructedFrames);
        }

        [Test]
        public void Session_StateTransitionsFinalizeCancelAndResetAreDeterministic()
        {
            var session = new RecognitionSequenceInputSession();
            Assert.AreEqual(RuntimeSequenceSessionState.Idle, session.State);
            session.Begin(RuntimeSequenceMetadata.RuntimeUnlabeled("runtime-session"));
            Assert.AreEqual(RuntimeSequenceSessionState.Begun, session.State);
            session.AddFrame("runtime-session", ValidFrame(0.0));
            Assert.AreEqual(RuntimeSequenceSessionState.Collecting, session.State);
            RecognitionSequence sequence = session.FinalizeSequence("runtime-session");
            Assert.AreEqual(RuntimeSequenceSessionState.Finalized, session.State);
            Assert.AreEqual(1, sequence.Length);
            session.Reset();
            Assert.AreEqual(RuntimeSequenceSessionState.Idle, session.State);
            Assert.AreEqual(0, session.FrameCount);

            session.Begin(RuntimeSequenceMetadata.RuntimeUnlabeled("cancel-session"));
            session.AddFrame("cancel-session", ValidFrame(0.0));
            session.Cancel("cancel-session");
            Assert.AreEqual(RuntimeSequenceSessionState.Cancelled, session.State);
            session.Reset();
            Assert.AreEqual(RuntimeSequenceSessionState.Idle, session.State);
        }

        [Test]
        public void Session_RejectsProhibitedTransitionsEmptyFinalizeAndMetadataMismatch()
        {
            var session = new RecognitionSequenceInputSession();
            Assert.Throws<InvalidOperationException>(() => session.AddFrame("none", ValidFrame(0.0)));
            Assert.Throws<InvalidOperationException>(() => session.FinalizeSequence("none"));
            Assert.Throws<InvalidOperationException>(() => session.Cancel("none"));
            session.Begin(RuntimeSequenceMetadata.RuntimeUnlabeled("one"));
            Assert.Throws<InvalidOperationException>(() => session.Begin(RuntimeSequenceMetadata.RuntimeUnlabeled("two")));
            Assert.Throws<ArgumentException>(() => session.AddFrame("different", ValidFrame(0.0)));
            Assert.Throws<ArgumentException>(() => session.FinalizeSequence("different"));
            Assert.Throws<InvalidOperationException>(() => session.FinalizeSequence("one"));
            session.AddFrame("one", ValidFrame(0.0));
            session.FinalizeSequence("one");
            Assert.Throws<InvalidOperationException>(() => session.AddFrame("one", ValidFrame(1.0)));
            Assert.Throws<InvalidOperationException>(() => session.FinalizeSequence("one"));
            Assert.Throws<InvalidOperationException>(() => session.Cancel("one"));
            session.Reset();
            session.Begin(RuntimeSequenceMetadata.RuntimeUnlabeled("cancel"));
            session.Cancel("cancel");
            Assert.Throws<InvalidOperationException>(() => session.AddFrame("cancel", ValidFrame(1.0)));
            Assert.Throws<InvalidOperationException>(() => session.FinalizeSequence("cancel"));
        }

        [Test]
        public void FrameAndSession_RejectShapeTimestampAndValidCoordinateErrorsButPreserveInvalidPayload()
        {
            Assert.Throws<ArgumentException>(() => new NormalizedRecognitionFrame(0.0, new double[47], new bool[16]));
            Assert.Throws<ArgumentException>(() => new NormalizedRecognitionFrame(0.0, new double[48], new bool[15]));
            Assert.Throws<ArgumentOutOfRangeException>(() => new NormalizedRecognitionFrame(double.NaN, new double[48], new bool[16]));
            Assert.Throws<ArgumentOutOfRangeException>(() => new NormalizedRecognitionFrame(double.PositiveInfinity, new double[48], new bool[16]));
            var validCoordinates = new double[48];
            var validMask = Enumerable.Repeat(true, 16).ToArray();
            validCoordinates[0] = double.NaN;
            Assert.Throws<ArgumentException>(() => new NormalizedRecognitionFrame(0.0, validCoordinates, validMask));
            validCoordinates[0] = double.PositiveInfinity;
            Assert.Throws<ArgumentException>(() => new NormalizedRecognitionFrame(0.0, validCoordinates, validMask));

            var invalidCoordinates = new double[48];
            var invalidMask = Enumerable.Repeat(true, 16).ToArray();
            invalidMask[0] = false;
            invalidCoordinates[0] = double.NaN;
            invalidCoordinates[1] = double.NaN;
            invalidCoordinates[2] = double.NaN;
            var session = new RecognitionSequenceInputSession();
            session.Begin(RuntimeSequenceMetadata.RuntimeUnlabeled("invalid-preserve"));
            session.AddFrame("invalid-preserve", new NormalizedRecognitionFrame(0.0, invalidCoordinates, invalidMask));
            RecognitionSequence preserved = session.FinalizeSequence("invalid-preserve");
            Assert.IsFalse(preserved.IsValid(0, 0));
            Assert.IsTrue(double.IsNaN(preserved.Coordinate(0, 0, 0)));

            session.Reset();
            session.Begin(RuntimeSequenceMetadata.RuntimeUnlabeled("timestamps"));
            session.AddFrame("timestamps", ValidFrame(1.0));
            Assert.Throws<ArgumentException>(() => session.AddFrame("timestamps", ValidFrame(1.0)));
            Assert.Throws<ArgumentException>(() => session.AddFrame("timestamps", ValidFrame(0.5)));
        }

        [Test]
        public void RuntimeMetadata_IsExplicitlyUnlabeledAndPayloadProvenanceIsDeterministic()
        {
            RecognitionSequence first = BuildRuntimeSequence("runtime-001", new[] { 0.0, 0.1 });
            RecognitionSequence second = BuildRuntimeSequence("runtime-001", new[] { 0.0, 0.1 });
            RecognitionSequence timestampChanged = BuildRuntimeSequence("runtime-001", new[] { 0.0, 0.2 });
            Assert.AreEqual(RuntimeSequenceMetadata.RuntimeWord, first.Word);
            Assert.AreEqual(RuntimeSequenceMetadata.RuntimeSpeed, first.Speed);
            Assert.AreEqual(RuntimeSequenceMetadata.RuntimeSplit, first.Split);
            Assert.AreEqual("runtime-001", first.SampleId);
            Assert.AreEqual(first.SourceNpzSha256, second.SourceNpzSha256);
            Assert.AreEqual(first.LogicalPayloadSha256, second.LogicalPayloadSha256);
            Assert.AreNotEqual(first.SourceNpzSha256, timestampChanged.SourceNpzSha256);
            Assert.AreEqual(first.LogicalPayloadSha256, timestampChanged.LogicalPayloadSha256);
            Assert.IsTrue(RecognitionSequence.IsLowerHexSha256(first.SourceNpzSha256));
            Assert.IsTrue(RecognitionSequence.IsLowerHexSha256(first.LogicalPayloadSha256));
        }

        [Test]
        public void ServiceResult_IsImmutableInclusiveAtBoundaryAndRejectsInvalidQueries()
        {
            RecognitionSequence source = RecognitionSequenceBinary.LoadFile(ProjectPath(fixture.test_queries[0].relative_path));
            RuntimeRecognitionResult result = service.Recognize(source.SampleId, source);
            foreach (PropertyInfo property in typeof(RuntimeRecognitionResult).GetProperties()) Assert.IsFalse(property.CanWrite, property.Name);
            Assert.Throws<NotSupportedException>(() => ((IList<RuntimeTemplateDistance>)result.TemplateDistances)[0] = null);
            Assert.Throws<ArgumentNullException>(() => service.Recognize("query", null));
            Assert.Throws<ArgumentException>(() => service.Recognize("wrong-query-id", source));
            Assert.Throws<ArgumentException>(() => service.Recognize(service.TemplateIds[0], RecognitionSequenceBinary.LoadFile(Directory.GetFiles(TemplateDirectory, "*.wisrt").OrderBy(path => path, StringComparer.Ordinal).First())));

            RecognitionSequence prototype = RecognitionSequenceBinary.LoadFile(Directory.GetFiles(TemplateDirectory, "*.wisrt").OrderBy(path => path, StringComparer.Ordinal).First());
            var boundary = new RecognitionResult(
                Array.Empty<TemplateDistance>(),
                new[] { new WordDistance(prototype.Word, service.Threshold, prototype) },
                prototype.Word,
                prototype);
            Assert.IsTrue(DistanceGatePolicy.Evaluate(boundary, service.Threshold).Accepted);
        }

        [Test]
        public void ThresholdLoaderAndService_RejectMissingMalformedHashChangedAndValueInconsistentAssets()
        {
            string temp = NewTempDirectory();
            try
            {
                string missing = Path.Combine(temp, "missing.json");
                Assert.Throws<FileNotFoundException>(() => RuntimeRecognitionService.Load(TemplateDirectory, missing));
                byte[] original = File.ReadAllBytes(RuntimeThresholdPath);
                string altered = Path.Combine(temp, "altered.json");
                byte[] changed = (byte[])original.Clone();
                changed[changed.Length - 4] = changed[changed.Length - 4] == (byte)'0' ? (byte)'1' : (byte)'0';
                File.WriteAllBytes(altered, changed);
                Assert.Throws<InvalidDataException>(() => RuntimeRecognitionService.Load(TemplateDirectory, altered));
                Assert.Throws<InvalidDataException>(() => FrozenDistanceGateConfigLoader.Load(Encoding.UTF8.GetBytes("{}\n"), SealedTestIds));

                string text = Encoding.UTF8.GetString(original);
                string changedValue = text.Replace("\"threshold\":0.2908679008489196", "\"threshold\":0.2918679008489196");
                Assert.AreNotEqual(text, changedValue);
                byte[] rehashedInconsistent = RehashCanonicalWrapper(changedValue);
                Assert.Throws<InvalidDataException>(() => FrozenDistanceGateConfigLoader.Load(rehashedInconsistent, SealedTestIds));
            }
            finally
            {
                Directory.Delete(temp, true);
            }
        }

        [Test]
        public void Service_RejectsMissingAndDuplicateTemplateGroups()
        {
            string missingDirectory = NewTempDirectory();
            string duplicateDirectory = NewTempDirectory();
            try
            {
                CopyTemplates(missingDirectory);
                string[] missingPaths = Directory.GetFiles(missingDirectory, "*.wisrt");
                File.Delete(missingPaths.OrderBy(path => path, StringComparer.Ordinal).First());
                Assert.Throws<InvalidDataException>(() => RuntimeRecognitionService.Load(missingDirectory, RuntimeThresholdPath));

                CopyTemplates(duplicateDirectory);
                string[] duplicatePaths = Directory.GetFiles(duplicateDirectory, "*.wisrt").OrderBy(path => path, StringComparer.Ordinal).ToArray();
                File.Copy(duplicatePaths[0], duplicatePaths[duplicatePaths.Length - 1], true);
                Assert.Throws<InvalidDataException>(() => RuntimeRecognitionService.Load(duplicateDirectory, RuntimeThresholdPath));
            }
            finally
            {
                Directory.Delete(missingDirectory, true);
                Directory.Delete(duplicateDirectory, true);
            }
        }

        [Test]
        public void TrainValidationAndTestFixturesRemainDisjointByIdAndSplit()
        {
            var trainIds = new HashSet<string>(service.TemplateIds, StringComparer.Ordinal);
            var testIds = new HashSet<string>(fixture.test_queries.Select(record => record.sample_id), StringComparer.Ordinal);
            string validationDirectory = Path.Combine(Application.dataPath, "_Project", "Tests", "EditMode", "Fixtures", "RecognitionCoreV1", "ValidationQueries");
            RecognitionSequence[] validation = Directory.GetFiles(validationDirectory, "*.wisrq")
                .Select(RecognitionSequenceBinary.LoadFile).ToArray();
            var validationIds = new HashSet<string>(validation.Select(item => item.SampleId), StringComparer.Ordinal);
            Assert.AreEqual(9, trainIds.Count);
            Assert.AreEqual(9, validationIds.Count);
            Assert.AreEqual(9, testIds.Count);
            Assert.AreEqual(0, trainIds.Intersect(validationIds).Count());
            Assert.AreEqual(0, trainIds.Intersect(testIds).Count());
            Assert.AreEqual(0, validationIds.Intersect(testIds).Count());
            Assert.IsTrue(validation.All(item => item.Split == "val"));
            Assert.IsTrue(fixture.test_queries.All(item => item.split == "test"));
        }

        RecognitionSequence BuildRuntimeSequence(string sessionId, double[] timestamps)
        {
            var session = new RecognitionSequenceInputSession();
            session.Begin(RuntimeSequenceMetadata.RuntimeUnlabeled(sessionId));
            foreach (double timestamp in timestamps) session.AddFrame(sessionId, ValidFrame(timestamp));
            return session.FinalizeSequence(sessionId);
        }

        static NormalizedRecognitionFrame ValidFrame(double timestamp)
        {
            return new NormalizedRecognitionFrame(timestamp, new double[48], Enumerable.Repeat(true, 16).ToArray());
        }

        string ProjectPath(string relativePath) => Path.Combine(ProjectRoot, relativePath.Replace('/', Path.DirectorySeparatorChar));

        static IEnumerable<string[]> ReadCsv(string path)
        {
            string[] lines = File.ReadAllLines(path);
            for (int index = 1; index < lines.Length; index++)
                if (!string.IsNullOrWhiteSpace(lines[index])) yield return lines[index].Split(',');
        }

        static double ParseDouble(string value) => double.Parse(value, NumberStyles.Float, CultureInfo.InvariantCulture);

        static string CoordinatesSha256(RecognitionSequence sequence)
        {
            using (var stream = new MemoryStream())
            using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
            {
                foreach (double value in sequence.CopyCoordinates()) writer.Write(value);
                writer.Flush();
                return Sha256(stream.ToArray());
            }
        }

        static string ValidMaskSha256(RecognitionSequence sequence)
        {
            bool[] mask = sequence.CopyValidMask();
            var bytes = new byte[mask.Length];
            for (int index = 0; index < mask.Length; index++) bytes[index] = mask[index] ? (byte)1 : (byte)0;
            return Sha256(bytes);
        }

        static string Sha256(byte[] bytes)
        {
            using (var sha = SHA256.Create()) return RecognitionSequenceBinary.ToLowerHex(sha.ComputeHash(bytes));
        }

        static byte[] RehashCanonicalWrapper(string wrapper)
        {
            const string prefix = "{\"canonical_payload\":";
            const string marker = ",\"canonical_payload_sha256\":\"";
            string trimmed = wrapper.TrimEnd('\n');
            int markerIndex = trimmed.LastIndexOf(marker, StringComparison.Ordinal);
            string canonical = trimmed.Substring(prefix.Length, markerIndex - prefix.Length);
            string rebuilt = prefix + canonical + marker + Sha256(Encoding.UTF8.GetBytes(canonical)) + "\"}\n";
            return Encoding.UTF8.GetBytes(rebuilt);
        }

        static string NewTempDirectory()
        {
            string path = Path.Combine(Path.GetTempPath(), "WordsInSilence-Step9A-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(path);
            return path;
        }

        void CopyTemplates(string destination)
        {
            foreach (string source in Directory.GetFiles(TemplateDirectory, "*.wisrt"))
                File.Copy(source, Path.Combine(destination, Path.GetFileName(source)));
        }
    }
}
