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
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class PersonalRuntimeOptInIntegrationV1Tests
    {
        const double Tolerance = 1e-12;
        const string GreetingTemplateSha = "d8d5713334b22d9427b9b589bccf820cdbb4849357bae3dce1c368eaf0049681";
        const string LoveTemplateSha = "013bf5515eb8af33f300400893da64c853668b2fa0442e19203da62b2777ad4b";
        const string ThankYouTemplateSha = "9bb9a7681d55a292a21d8b7f458b4b150c10555bc0ec163da692a780ed5280e5";
        const string ConfigSha = "e84edd1f598c2d94f8412767f3125031e0be330886d3a315ed429798fe49726e";

        string ProjectRoot => Directory.GetParent(Application.dataPath).FullName;
        string ProfileDirectory => Path.Combine(Application.streamingAssetsPath, "RecognitionCoreV1", "PersonalProfiles", "step9h_current_user_v1_0");
        string GlobalTemplateDirectory => Path.Combine(Application.streamingAssetsPath, "RecognitionCoreV1", "Templates");
        string GlobalThresholdPath => Path.Combine(Application.streamingAssetsPath, "RecognitionCoreV1", "RuntimeGate", "frozen_threshold_config.json");
        string FixtureDirectory => Path.Combine(Application.dataPath, "_Project", "Tests", "EditMode", "Fixtures", "PersonalRuntimeOptInIntegrationV1");

        [Serializable]
        sealed class QueryManifest
        {
            public string schema_version;
            public QueryRecord[] queries;
        }

        [Serializable]
        sealed class QueryRecord
        {
            public string query_id;
            public string expected_word;
            public string capture_id;
            public string run_id;
            public string relative_path;
            public int frame_count;
            public long size_bytes;
            public string sha256;
            public string source_payload_sha256;
            public string normalized_payload_sha256;
        }

        [Serializable]
        sealed class GlobalAssetSnapshot
        {
            public string schema_version;
            public GlobalAssetRecord[] assets;
        }

        [Serializable]
        sealed class GlobalAssetRecord
        {
            public string relative_path;
            public long size_bytes;
            public string sha256;
        }

        RuntimeRecognitionService globalService;
        PersonalRuntimeProfile profile;
        QueryManifest queryManifest;
        Dictionary<string, RecognitionSequence> queries;
        Dictionary<string, string[]> expectedScores;
        Dictionary<string, string[]> expectedPredictions;
        Dictionary<string, PersonalRuntimeRecognitionResult> personalResults;

        [OneTimeSetUp]
        public void OneTimeSetUp()
        {
            globalService = RuntimeRecognitionService.Load(GlobalTemplateDirectory, GlobalThresholdPath);
            profile = PersonalRuntimeProfileLoader.Load(ProfileDirectory);
            queryManifest = JsonUtility.FromJson<QueryManifest>(
                File.ReadAllText(Path.Combine(FixtureDirectory, "personal_query_manifest.json")));
            queries = queryManifest.queries.ToDictionary(
                item => item.query_id,
                item => RecognitionSequenceBinary.LoadFile(ProjectPath(item.relative_path)),
                StringComparer.Ordinal);
            expectedScores = ReadCsv(Path.Combine(FixtureDirectory, "personal_expected_scores.csv"))
                .ToDictionary(row => row[1] + "|" + row[3], StringComparer.Ordinal);
            expectedPredictions = ReadCsv(Path.Combine(FixtureDirectory, "personal_expected_predictions.csv"))
                .ToDictionary(row => row[0], StringComparer.Ordinal);

            var session = new PersonalRuntimeOptInSession(globalService);
            session.ActivatePersonalProfile(ProfileDirectory);
            personalResults = new Dictionary<string, PersonalRuntimeRecognitionResult>(StringComparer.Ordinal);
            foreach (QueryRecord record in queryManifest.queries)
            {
                RuntimeOptInRecognitionResult result = session.Recognize(record.query_id, queries[record.query_id], record.expected_word);
                Assert.AreEqual(RuntimeOptInMode.Personal, result.Mode);
                personalResults.Add(record.query_id, result.PersonalResult);
            }
        }

        [Test]
        public void ApprovedStep9HAssets_ThreeTemplatesAndConfigAreByteIdentical()
        {
            AssertAsset(Path.Combine(ProfileDirectory, "personal_template_GREETING.wisrt"), 53564, GreetingTemplateSha);
            AssertAsset(Path.Combine(ProfileDirectory, "personal_template_LOVE.wisrt"), 56262, LoveTemplateSha);
            AssertAsset(Path.Combine(ProfileDirectory, "personal_template_THANK_YOU.wisrt"), 49319, ThankYouTemplateSha);
            AssertAsset(Path.Combine(ProfileDirectory, PersonalRuntimeProfileLoader.ConfigFileName), 4811, ConfigSha);
        }

        [Test]
        public void WisrtEmbeddedChecksumsAndManifestProvenance_MatchApprovedMedoids()
        {
            Assert.AreEqual(3, profile.Templates.Count);
            foreach (string word in profile.Words)
            {
                RecognitionSequence sequence = profile.Templates[word];
                PersonalTemplateProvenance provenance = profile.TemplateProvenance[word];
                Assert.AreEqual(word, sequence.Word);
                Assert.AreEqual(provenance.CaptureId, sequence.SampleId);
                Assert.AreEqual(provenance.SourcePayloadSha256, sequence.SourceNpzSha256);
                Assert.AreEqual(provenance.NormalizedPayloadSha256, sequence.LogicalPayloadSha256);
                Assert.AreEqual(provenance.EmbeddedChecksumSha256, sequence.BinaryPayloadSha256);
                Assert.AreEqual(provenance.CoordinatePayloadSha256, CoordinatesSha256(sequence));
                Assert.AreEqual(provenance.MaskPayloadSha256, ValidMaskSha256(sequence));
            }
        }

        [Test]
        public void ThresholdMap_IsExactlyThreeApprovedFiniteNonNegativeValues()
        {
            CollectionAssert.AreEqual(new[] { "GREETING", "LOVE", "THANK_YOU" }, profile.Thresholds.Keys.ToArray());
            Assert.AreEqual(0.5205639017143886, profile.Thresholds["GREETING"], 0.0);
            Assert.AreEqual(0.39564149928871895, profile.Thresholds["LOVE"], 0.0);
            Assert.AreEqual(0.38124865401356706, profile.Thresholds["THANK_YOU"], 0.0);
            Assert.IsTrue(profile.Thresholds.Values.All(value => !double.IsNaN(value) && !double.IsInfinity(value) && value >= 0.0));
        }

        [Test]
        public void Session_DefaultIsGlobalAndDoesNotRequireOrLoadPersonalProfile()
        {
            var session = new PersonalRuntimeOptInSession(globalService);
            Assert.AreEqual(RuntimeOptInMode.Global, session.Mode);
            Assert.IsFalse(session.IsPersonalActive);
            Assert.AreEqual(string.Empty, session.ActiveProfileId);

            QueryRecord record = queryManifest.queries[0];
            RuntimeRecognitionResult direct = globalService.Recognize(record.query_id, queries[record.query_id]);
            RuntimeOptInRecognitionResult routed = session.Recognize(record.query_id, queries[record.query_id], record.expected_word);
            Assert.AreEqual(RuntimeOptInMode.Global, routed.Mode);
            Assert.IsNull(routed.PersonalResult);
            AssertGlobalEqual(direct, routed.GlobalResult);
        }

        [Test]
        public void ExplicitOptIn_LoadsExactlyThreePersonalMedoids()
        {
            var session = new PersonalRuntimeOptInSession(globalService);
            session.ActivatePersonalProfile(ProfileDirectory);
            Assert.AreEqual(RuntimeOptInMode.Personal, session.Mode);
            Assert.IsTrue(session.IsPersonalActive);
            Assert.AreEqual(PersonalRuntimeProfileLoader.ProfileId, session.ActiveProfileId);
            Assert.AreEqual(3, profile.Templates.Count);
            CollectionAssert.AreEquivalent(
                new[]
                {
                    "87e5b54432ac49a197a596d5ff7c3845",
                    "d958cbc2a1c44e03bb29bf64a6fae195",
                    "2b323671e3f64049a74556c71ff70154",
                },
                profile.Templates.Values.Select(item => item.SampleId).ToArray());
        }

        [Test]
        public void RuntimeScores_AllTwentySevenMatchApprovedPythonWithinTolerance()
        {
            double maxError = 0.0;
            int count = 0;
            foreach (QueryRecord record in queryManifest.queries)
            foreach (string word in profile.Words)
            {
                string[] expected = expectedScores[record.query_id + "|" + word];
                double expectedDistance = ParseDouble(expected[4]);
                double actual = personalResults[record.query_id].Scores[word];
                maxError = Math.Max(maxError, Math.Abs(actual - expectedDistance));
                Assert.AreEqual(expectedDistance, actual, Tolerance, record.query_id + " -> " + word);
                Assert.AreEqual(bool.Parse(expected[10]), personalResults[record.query_id].PerClassAccept[word]);
                count++;
            }
            Assert.AreEqual(27, count);
            Assert.LessOrEqual(maxError, Tolerance);
        }

        [Test]
        public void FullMedoidPrediction_IsNineOfNine()
        {
            int correct = queryManifest.queries.Count(record =>
                personalResults[record.query_id].PredictedWord == record.expected_word);
            Assert.AreEqual(9, correct);
            foreach (QueryRecord record in queryManifest.queries)
                Assert.AreEqual(expectedPredictions[record.query_id][2], personalResults[record.query_id].PredictedWord);
        }

        [Test]
        public void GenuineClassAcceptance_IsNineOfNine()
        {
            Assert.AreEqual(9, queryManifest.queries.Count(record =>
                personalResults[record.query_id].PerClassAccept[record.expected_word]));
        }

        [Test]
        public void NonTargetClassRejection_IsEighteenOfEighteen()
        {
            int rejected = 0;
            foreach (QueryRecord record in queryManifest.queries)
            foreach (string word in profile.Words.Where(word => word != record.expected_word))
                if (!personalResults[record.query_id].PerClassAccept[word]) rejected++;
            Assert.AreEqual(18, rejected);
        }

        [Test]
        public void QuestSuccess_IsNineOfNineWithExpectedWordClassGate()
        {
            Assert.AreEqual(9, personalResults.Values.Count(result =>
                result.Accepted && result.PredictedWord == result.ExpectedWord &&
                result.ExpectedWordClassAccept && result.QuestSuccess));
        }

        [Test]
        public void GreetingNarrowMarginWarning_IsPreserved()
        {
            Assert.IsTrue(profile.GreetingMarginNarrow);
            CollectionAssert.AreEqual(
                new[] { PersonalRuntimeProfileLoader.GreetingMarginWarning }, profile.Warnings);
            Assert.IsTrue(personalResults.Values.All(result =>
                result.Warnings.Contains(PersonalRuntimeProfileLoader.GreetingMarginWarning)));
        }

        [Test]
        public void Loader_RejectsMissingMalformedDuplicateAdditionalAndUnsafeProfilesWithTypedFailures()
        {
            AssertFailure(Copy => File.Delete(Path.Combine(Copy, PersonalRuntimeProfileLoader.ManifestFileName)),
                PersonalProfileFailureCode.ManifestMissing);
            AssertFailure(Copy => File.WriteAllText(Path.Combine(Copy, PersonalRuntimeProfileLoader.ManifestFileName), "{}\n"),
                PersonalProfileFailureCode.ManifestSchemaMismatch);
            AssertFailure(Copy => MutateManifest(Copy, text =>
                new Regex("(\\\"words\\\"\\s*:\\s*\\[\\s*\\\"GREETING\\\"\\s*,\\s*)\\\"LOVE\\\"")
                    .Replace(text, "$1\"GREETING\"", 1)),
                PersonalProfileFailureCode.DuplicateWord);
            AssertFailure(Copy => MutateManifest(Copy, text =>
                new Regex("(\\\"words\\\"\\s*:\\s*\\[\\s*\\\"GREETING\\\"\\s*,\\s*\\\"LOVE\\\"\\s*,\\s*)\\\"THANK_YOU\\\"")
                    .Replace(text, "$1\"WRONG_WORD\"", 1)),
                PersonalProfileFailureCode.WordSetMismatch);
            AssertFailure(Copy => MutateManifest(Copy, text =>
                ReplaceFirst(text, "personal_template_GREETING.wisrt", "../personal_template_GREETING.wisrt")),
                PersonalProfileFailureCode.UnsafeRelativePath);
            AssertFailure(Copy => File.Delete(Path.Combine(Copy, "personal_template_LOVE.wisrt")),
                PersonalProfileFailureCode.AssetMissing);
            AssertFailure(Copy => File.Copy(
                    Path.Combine(Copy, "personal_template_GREETING.wisrt"),
                    Path.Combine(Copy, "unexpected_extra.wisrt")),
                PersonalProfileFailureCode.WordSetMismatch);
        }

        [Test]
        public void Loader_RejectsThresholdKeyAndNegativeThresholdMutations()
        {
            AssertFailure(Copy => MutateConfig(Copy, bytes =>
            {
                string text = Encoding.UTF8.GetString(bytes);
                string changed = ReplaceLast(text,
                    "\"THANK_YOU\": 0.38124865401356706",
                    "\"WRONG_KEY\": 0.38124865401356706");
                return Encoding.UTF8.GetBytes(changed);
            }), PersonalProfileFailureCode.ManifestSchemaMismatch);

            AssertFailure(Copy => MutateConfig(Copy, bytes =>
            {
                string text = Encoding.UTF8.GetString(bytes);
                string changed = ReplaceLast(text,
                    "\"GREETING\": 0.5205639017143886",
                    "\"GREETING\": -0.5205639017143886");
                return Encoding.UTF8.GetBytes(changed);
            }), PersonalProfileFailureCode.ThresholdInvalid);
        }

        [Test]
        public void Loader_RejectsWisrtMagicVersionEndianChecksumHashAndProvenanceMutations()
        {
            AssertFailure(Copy => MutateTemplate(Copy, "personal_template_GREETING.wisrt", bytes =>
            {
                bytes[0] ^= 1;
                RehashEmbedded(bytes);
            }), PersonalProfileFailureCode.WisrtFormatInvalid);
            AssertFailure(Copy => MutateTemplate(Copy, "personal_template_GREETING.wisrt", bytes =>
            {
                bytes[8] ^= 1;
                RehashEmbedded(bytes);
            }), PersonalProfileFailureCode.WisrtFormatInvalid);
            AssertFailure(Copy => MutateTemplate(Copy, "personal_template_GREETING.wisrt", bytes =>
            {
                bytes[12] ^= 1;
                RehashEmbedded(bytes);
            }), PersonalProfileFailureCode.WisrtFormatInvalid);
            AssertFailure(Copy => MutateTemplate(Copy, "personal_template_GREETING.wisrt", bytes =>
                bytes[bytes.Length - 1] ^= 1), PersonalProfileFailureCode.EmbeddedChecksumMismatch);
            AssertFailure(Copy => MutateTemplate(Copy, "personal_template_GREETING.wisrt", bytes =>
            {
                bytes[bytes.Length / 2] ^= 1;
                RehashEmbedded(bytes);
            }), PersonalProfileFailureCode.AssetSha256Mismatch);
            AssertFailure(Copy => MutateManifest(Copy, text => ReplaceFirst(text,
                    "87e5b54432ac49a197a596d5ff7c3845", "87e5b54432ac49a197a596d5ff7c3846")),
                PersonalProfileFailureCode.ProvenanceMismatch);
        }

        [Test]
        public void FailedActivation_IsAtomicAndPreservesExistingSessionMode()
        {
            var defaultSession = new PersonalRuntimeOptInSession(globalService);
            string broken = CopyProfile();
            try
            {
                File.Delete(Path.Combine(broken, PersonalRuntimeProfileLoader.ConfigFileName));
                Assert.Throws<PersonalProfileLoadException>(() => defaultSession.ActivatePersonalProfile(broken));
                Assert.AreEqual(RuntimeOptInMode.Global, defaultSession.Mode);
                Assert.IsFalse(defaultSession.IsPersonalActive);

                var activeSession = new PersonalRuntimeOptInSession(globalService);
                activeSession.ActivatePersonalProfile(ProfileDirectory);
                Assert.Throws<PersonalProfileLoadException>(() => activeSession.ActivatePersonalProfile(broken));
                Assert.AreEqual(RuntimeOptInMode.Personal, activeSession.Mode);
                Assert.AreEqual(PersonalRuntimeProfileLoader.ProfileId, activeSession.ActiveProfileId);
            }
            finally
            {
                Directory.Delete(broken, true);
            }
        }

        [Test]
        public void ExplicitDeactivation_ReturnsToUnchangedGlobalMode()
        {
            var session = new PersonalRuntimeOptInSession(globalService);
            session.ActivatePersonalProfile(ProfileDirectory);
            session.DeactivatePersonalProfile();
            Assert.AreEqual(RuntimeOptInMode.Global, session.Mode);
            Assert.IsFalse(session.IsPersonalActive);
            QueryRecord record = queryManifest.queries[0];
            RuntimeRecognitionResult direct = globalService.Recognize(record.query_id, queries[record.query_id]);
            RuntimeOptInRecognitionResult routed = session.Recognize(record.query_id, queries[record.query_id], record.expected_word);
            AssertGlobalEqual(direct, routed.GlobalResult);
        }

        [Test]
        public void ExistingGlobalRegressionFixture_AllNineResultsRemainUnchanged()
        {
            string directory = Path.Combine(Application.dataPath, "_Project", "Tests", "EditMode", "Fixtures", "RuntimeRecognitionServiceV1");
            Dictionary<string, string[]> expected = ReadCsv(Path.Combine(directory, "test_expected_predictions.csv"))
                .ToDictionary(row => row[0], StringComparer.Ordinal);
            string queryDirectory = Path.Combine(directory, "TestQueries");
            var session = new PersonalRuntimeOptInSession(globalService);
            int count = 0;
            foreach (string path in Directory.GetFiles(queryDirectory, "*.wisrq").OrderBy(item => item, StringComparer.Ordinal))
            {
                RecognitionSequence query = RecognitionSequenceBinary.LoadFile(path);
                string[] row = expected[query.SampleId];
                RuntimeOptInRecognitionResult result = session.Recognize(query.SampleId, query, row[2]);
                Assert.AreEqual(RuntimeOptInMode.Global, result.Mode);
                Assert.AreEqual(row[3], result.PredictedWord, query.SampleId);
                Assert.AreEqual(ParseDouble(row[6]), result.GlobalResult.GreetingDistance, Tolerance, query.SampleId);
                Assert.AreEqual(ParseDouble(row[7]), result.GlobalResult.LoveDistance, Tolerance, query.SampleId);
                Assert.AreEqual(ParseDouble(row[8]), result.GlobalResult.ThankYouDistance, Tolerance, query.SampleId);
                Assert.AreEqual(bool.Parse(row[13]), result.Accepted, query.SampleId);
                count++;
            }
            Assert.AreEqual(9, count);
        }

        [Test]
        public void GlobalRuntimeAssets_MatchFrozenBeforeSnapshot()
        {
            var snapshot = JsonUtility.FromJson<GlobalAssetSnapshot>(
                File.ReadAllText(Path.Combine(FixtureDirectory, "global_runtime_asset_snapshot.json")));
            Assert.AreEqual("words_in_silence.step9i.global_runtime_asset_snapshot.v1_0", snapshot.schema_version);
            Assert.AreEqual(11, snapshot.assets.Length);
            foreach (GlobalAssetRecord record in snapshot.assets)
                AssertAsset(ProjectPath(record.relative_path), record.size_bytes, record.sha256);
        }

        void AssertFailure(Action<string> mutate, PersonalProfileFailureCode expectedCode)
        {
            string copy = CopyProfile();
            try
            {
                mutate(copy);
                PersonalProfileLoadException failure = Assert.Throws<PersonalProfileLoadException>(
                    () => PersonalRuntimeProfileLoader.Load(copy));
                Assert.AreEqual(expectedCode, failure.FailureCode, failure.Message);
            }
            finally
            {
                Directory.Delete(copy, true);
            }
        }

        string CopyProfile()
        {
            string destination = Path.Combine(Path.GetTempPath(), "WordsInSilence-Step9I-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(destination);
            foreach (string source in Directory.GetFiles(ProfileDirectory, "*", SearchOption.TopDirectoryOnly))
                if (!source.EndsWith(".meta", StringComparison.OrdinalIgnoreCase))
                    File.Copy(source, Path.Combine(destination, Path.GetFileName(source)));
            return destination;
        }

        static void MutateManifest(string directory, Func<string, string> mutation)
        {
            string path = Path.Combine(directory, PersonalRuntimeProfileLoader.ManifestFileName);
            string original = File.ReadAllText(path, Encoding.UTF8);
            string changed = mutation(original);
            Assert.AreNotEqual(original, changed);
            File.WriteAllText(path, changed, new UTF8Encoding(false));
        }

        static void MutateConfig(string directory, Func<byte[], byte[]> mutation)
        {
            string path = Path.Combine(directory, PersonalRuntimeProfileLoader.ConfigFileName);
            byte[] changed = mutation(File.ReadAllBytes(path));
            File.WriteAllBytes(path, changed);
            string manifestPath = Path.Combine(directory, PersonalRuntimeProfileLoader.ManifestFileName);
            string manifest = File.ReadAllText(manifestPath, Encoding.UTF8);
            var sizePattern = new Regex("(\\\"kind\\\"\\s*:\\s*\\\"threshold_config\\\"[\\s\\S]*?\\\"size_bytes\\\"\\s*:\\s*)[0-9]+", RegexOptions.CultureInvariant);
            manifest = sizePattern.Replace(manifest, match => match.Groups[1].Value + changed.Length.ToString(CultureInfo.InvariantCulture), 1);
            var shaPattern = new Regex("(\\\"kind\\\"\\s*:\\s*\\\"threshold_config\\\"[\\s\\S]*?\\\"sha256\\\"\\s*:\\s*\\\")[0-9a-f]{64}(\\\")", RegexOptions.CultureInvariant);
            string changedSha = Sha256(changed);
            manifest = shaPattern.Replace(manifest, match => match.Groups[1].Value + changedSha + match.Groups[2].Value, 1);
            File.WriteAllText(manifestPath, manifest, new UTF8Encoding(false));
        }

        static void MutateTemplate(string directory, string fileName, Action<byte[]> mutation)
        {
            string path = Path.Combine(directory, fileName);
            byte[] bytes = File.ReadAllBytes(path);
            mutation(bytes);
            File.WriteAllBytes(path, bytes);
        }

        static void RehashEmbedded(byte[] bytes)
        {
            using (var sha = SHA256.Create())
            {
                byte[] hash = sha.ComputeHash(bytes, 0, bytes.Length - 32);
                Buffer.BlockCopy(hash, 0, bytes, bytes.Length - 32, 32);
            }
        }

        static string ReplaceFirst(string value, string oldValue, string newValue)
        {
            int index = value.IndexOf(oldValue, StringComparison.Ordinal);
            if (index < 0) return value;
            return value.Substring(0, index) + newValue + value.Substring(index + oldValue.Length);
        }

        static string ReplaceLast(string value, string oldValue, string newValue)
        {
            int index = value.LastIndexOf(oldValue, StringComparison.Ordinal);
            if (index < 0) return value;
            return value.Substring(0, index) + newValue + value.Substring(index + oldValue.Length);
        }

        void AssertAsset(string path, long expectedSize, string expectedSha)
        {
            Assert.IsTrue(File.Exists(path), path);
            Assert.AreEqual(expectedSize, new FileInfo(path).Length, path);
            Assert.AreEqual(expectedSha, Sha256File(path), path);
        }

        static void AssertGlobalEqual(RuntimeRecognitionResult expected, RuntimeRecognitionResult actual)
        {
            Assert.AreEqual(expected.PredictedWord, actual.PredictedWord);
            Assert.AreEqual(expected.WinningPrototypeId, actual.WinningPrototypeId);
            Assert.AreEqual(expected.GreetingDistance, actual.GreetingDistance, 0.0);
            Assert.AreEqual(expected.LoveDistance, actual.LoveDistance, 0.0);
            Assert.AreEqual(expected.ThankYouDistance, actual.ThankYouDistance, 0.0);
            Assert.AreEqual(expected.BestDistance, actual.BestDistance, 0.0);
            Assert.AreEqual(expected.Threshold, actual.Threshold, 0.0);
            Assert.AreEqual(expected.Accepted, actual.Accepted);
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

        static string Sha256File(string path)
        {
            using (var stream = File.OpenRead(path))
            using (var sha = SHA256.Create()) return RecognitionSequenceBinary.ToLowerHex(sha.ComputeHash(stream));
        }

        static string Sha256(byte[] bytes)
        {
            using (var sha = SHA256.Create()) return RecognitionSequenceBinary.ToLowerHex(sha.ComputeHash(bytes));
        }
    }
}
