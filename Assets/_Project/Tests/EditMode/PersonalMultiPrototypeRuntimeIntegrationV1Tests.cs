using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class PersonalMultiPrototypeRuntimeIntegrationV1Tests
    {
        const double Tolerance = 1e-12;
        static readonly string[] ExpectedCaptureIds =
        {
                "step9c-take-01-greeting-f7875c16-ad18-425c-8ce5-39e9a6cc13f7",
                "26f436475e2f4e8c832dbce00cbfe883",
                "87e5b54432ac49a197a596d5ff7c3845",
                "step9c-take-02-love-6d68ddd7-2d7e-4afd-bd0a-b1989f8e52f2",
                "cd65ad6b277c48b78fa50785070fdddc",
                "d958cbc2a1c44e03bb29bf64a6fae195",
                "step9c-take-03-thank-you-53870472-c644-489c-a94d-95e9667ded42",
                "2b323671e3f64049a74556c71ff70154",
                "9926cef4a46649bea3139fbace2a0ff3",
        };

        string ProjectRoot => Directory.GetParent(Application.dataPath).FullName;
        string CandidateDirectory => Path.Combine(Application.streamingAssetsPath, "RecognitionCoreV1", "PersonalProfiles", "step9l_current_user_multi_prototype_v1_0");
        string ExistingV1Directory => Path.Combine(Application.streamingAssetsPath, "RecognitionCoreV1", "PersonalProfiles", "step9h_current_user_v1_0");
        string GlobalTemplateDirectory => Path.Combine(Application.streamingAssetsPath, "RecognitionCoreV1", "Templates");
        string GlobalThresholdPath => Path.Combine(Application.streamingAssetsPath, "RecognitionCoreV1", "RuntimeGate", "frozen_threshold_config.json");
        string FixtureDirectory => Path.Combine(Application.dataPath, "_Project", "Tests", "EditMode", "Fixtures", "PersonalMultiPrototypeRuntimeIntegrationV1");

        [Serializable]
        sealed class QueryManifest { public string schema_version; public QueryRecord[] queries; }
        [Serializable]
        sealed class QueryRecord
        {
            public string query_id;
            public string capture_id;
            public string capture_key;
            public string expected_word;
            public string run_id;
            public string relative_path;
            public int frame_count;
            public long size_bytes;
            public string sha256;
            public string source_payload_sha256;
            public string normalized_payload_sha256;
        }

        PersonalMultiPrototypeRuntimeProfile profile;
        QueryManifest manifest;
        Dictionary<string, RecognitionSequence> queries;
        Dictionary<string, string[]> expectedCells;
        RuntimeRecognitionService globalService;

        [OneTimeSetUp]
        public void OneTimeSetUp()
        {
            profile = PersonalMultiPrototypeRuntimeProfileLoader.Load(CandidateDirectory);
            manifest = JsonUtility.FromJson<QueryManifest>(File.ReadAllText(Path.Combine(FixtureDirectory, "query_manifest.json")));
            queries = manifest.queries.ToDictionary(
                item => item.query_id,
                item => RecognitionSequenceBinary.LoadFile(Path.Combine(ProjectRoot, item.relative_path.Replace('/', Path.DirectorySeparatorChar))),
                StringComparer.Ordinal);
            expectedCells = ReadCsv(Path.Combine(FixtureDirectory, "strict_loo_expected_scores.csv"))
                .ToDictionary(row => row[0], StringComparer.Ordinal);
            globalService = RuntimeRecognitionService.Load(GlobalTemplateDirectory, GlobalThresholdPath);
        }

        [Test]
        public void CandidateProfile_HasExactlyNinePrototypesThreePerWordAndFixedIdentity()
        {
            Assert.AreEqual("step9l_current_user_multi_prototype_v1_0", profile.ProfileId);
            Assert.AreEqual(9, profile.Prototypes.Sum(pair => pair.Value.Count));
            foreach (string word in new[] { "GREETING", "LOVE", "THANK_YOU" })
                Assert.AreEqual(3, profile.Prototypes[word].Count, word);
            CollectionAssert.AreEquivalent(ExpectedCaptureIds,
                profile.Prototypes.Values.SelectMany(items => items).Select(item => item.SampleId).ToArray());
            Assert.AreEqual(9, Directory.GetFiles(CandidateDirectory, "*.wisrt", SearchOption.TopDirectoryOnly).Length);
        }

        [Test]
        public void CandidateThresholds_AreExactlyStrictLooMidpointsAndAllGapsArePositive()
        {
            Assert.AreEqual(0.5113059473720765, profile.Thresholds["GREETING"], 0.0);
            Assert.AreEqual(0.391276023460427, profile.Thresholds["LOVE"], 0.0);
            Assert.AreEqual(0.37709977431991587, profile.Thresholds["THANK_YOU"], 0.0);
            Assert.IsTrue(profile.Thresholds.Values.All(value => !double.IsNaN(value) && !double.IsInfinity(value) && value >= 0.0));
        }

        [Test]
        public void StrictLeaveOneOut_ExcludesQueryPrototypeAndReproducesAllTwentySevenScores()
        {
            int cells = 0;
            int correct = 0;
            foreach (QueryRecord record in manifest.queries)
            {
                var scores = new Dictionary<string, double>(StringComparer.Ordinal);
                foreach (string word in profile.Words)
                {
                    RecognitionSequence[] candidates = profile.Prototypes[word]
                        .Where(item => item.SampleId != record.capture_id).ToArray();
                    Assert.AreEqual(word == record.expected_word ? 2 : 3, candidates.Length,
                        record.capture_id + " -> " + word);
                    var ranked = candidates.Select(item => new
                    {
                        Prototype = item,
                        Dtw = MaskAwareDtw.Compute(queries[record.query_id], item, RecognitionCore.RelativeBand),
                    }).OrderBy(item => item.Dtw.Distance).ThenBy(item => item.Prototype.SampleId, StringComparer.Ordinal).ToArray();
                    string[] expected = expectedCells[record.capture_id + "||" + word];
                    Assert.AreEqual(ParseDouble(expected[10]), ranked[0].Dtw.Distance, Tolerance);
                    Assert.AreEqual(expected[8], ranked[0].Prototype.SampleId);
                    Assert.AreNotEqual(record.capture_id, ranked[0].Prototype.SampleId);
                    scores.Add(word, ranked[0].Dtw.Distance);
                    cells++;
                }
                string predicted = profile.Words.OrderBy(word => scores[word]).ThenBy(word => word, StringComparer.Ordinal).First();
                if (predicted == record.expected_word) correct++;
            }
            Assert.AreEqual(27, cells);
            Assert.AreEqual(9, correct);
        }

        [Test]
        public void ExistingV1Route_RemainsExactlyThreeMedoidsAndUsesOriginalProfileId()
        {
            PersonalRuntimeProfile existing = PersonalRuntimeProfileLoader.Load(ExistingV1Directory);
            Assert.AreEqual("step9h_current_user_v1_0", existing.ProfileId);
            Assert.AreEqual(3, existing.Templates.Count);
            CollectionAssert.AreEqual(new[] { "GREETING", "LOVE", "THANK_YOU" }, existing.Templates.Keys.ToArray());
        }

        [Test]
        public void Session_DefaultIsGlobalAndCandidateRequiresExplicitActivation()
        {
            var session = new PersonalRuntimeOptInSession(globalService);
            Assert.AreEqual(RuntimeOptInMode.Global, session.Mode);
            Assert.IsFalse(session.IsPersonalActive);
            Assert.AreEqual(string.Empty, session.ActiveProfileId);
            QueryRecord record = manifest.queries[0];
            RuntimeOptInRecognitionResult result = session.Recognize(record.query_id, queries[record.query_id], record.expected_word);
            Assert.AreEqual(RuntimeOptInMode.Global, result.Mode);
            Assert.IsNull(result.PersonalResult);
        }

        [Test]
        public void ExplicitCandidateActivation_LoadsNineAndRoutesMinimumPerClass()
        {
            var session = new PersonalRuntimeOptInSession(globalService);
            session.ActivatePersonalMultiPrototypeProfile(CandidateDirectory);
            Assert.AreEqual(RuntimeOptInMode.Personal, session.Mode);
            Assert.IsTrue(session.IsPersonalActive);
            Assert.AreEqual("step9l_current_user_multi_prototype_v1_0", session.ActiveProfileId);
            int correct = 0;
            foreach (QueryRecord record in manifest.queries)
            {
                RuntimeOptInRecognitionResult result = session.Recognize(record.query_id, queries[record.query_id], record.expected_word);
                Assert.AreEqual(RuntimeOptInMode.Personal, result.Mode);
                Assert.AreEqual(9, result.PersonalResult.TemplateDistances.Count);
                Assert.AreEqual(result.PersonalResult.TemplateDistances
                    .Where(item => item.Word == result.PredictedWord).Min(item => item.Distance), result.PersonalResult.BestDistance, Tolerance);
                if (result.PredictedWord == record.expected_word && result.QuestSuccess) correct++;
            }
            Assert.AreEqual(9, correct);
        }

        [Test]
        public void FailedCandidateActivation_IsAtomicAndPreservesExistingV1Session()
        {
            var session = new PersonalRuntimeOptInSession(globalService);
            session.ActivatePersonalProfile(ExistingV1Directory);
            string copy = CopyDirectory(CandidateDirectory);
            try
            {
                File.Delete(Path.Combine(copy, PersonalMultiPrototypeRuntimeProfileLoader.ConfigFileName));
                PersonalProfileLoadException failure = Assert.Throws<PersonalProfileLoadException>(() =>
                    session.ActivatePersonalMultiPrototypeProfile(copy));
                Assert.AreEqual(PersonalProfileFailureCode.AssetMissing, failure.FailureCode);
                Assert.AreEqual(RuntimeOptInMode.Personal, session.Mode);
                Assert.AreEqual("step9h_current_user_v1_0", session.ActiveProfileId);
            }
            finally { Directory.Delete(copy, true); }
        }

        [Test]
        public void CandidateDeactivation_ReturnsToUnchangedGlobalMode()
        {
            var session = new PersonalRuntimeOptInSession(globalService);
            session.ActivatePersonalMultiPrototypeProfile(CandidateDirectory);
            session.DeactivatePersonalProfile();
            Assert.AreEqual(RuntimeOptInMode.Global, session.Mode);
            Assert.IsFalse(session.IsPersonalActive);
            Assert.AreEqual(string.Empty, session.ActiveProfileId);
        }

        static double ParseDouble(string value) => double.Parse(value, NumberStyles.Float, CultureInfo.InvariantCulture);

        static List<string[]> ReadCsv(string path)
        {
            string[] lines = File.ReadAllLines(path);
            var rows = new List<string[]>();
            for (int index = 1; index < lines.Length; index++)
                if (!string.IsNullOrWhiteSpace(lines[index])) rows.Add(ParseCsvLine(lines[index]));
            return rows;
        }

        static string[] ParseCsvLine(string line)
        {
            var fields = new List<string>();
            var current = new System.Text.StringBuilder();
            bool quoted = false;
            for (int index = 0; index < line.Length; index++)
            {
                char c = line[index];
                if (c == '"')
                {
                    if (quoted && index + 1 < line.Length && line[index + 1] == '"') { current.Append('"'); index++; }
                    else quoted = !quoted;
                }
                else if (c == ',' && !quoted) { fields.Add(current.ToString()); current.Clear(); }
                else current.Append(c);
            }
            fields.Add(current.ToString());
            return fields.ToArray();
        }

        static string CopyDirectory(string source)
        {
            string destination = Path.Combine(Path.GetTempPath(), "WordsInSilence-Step9L-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(destination);
            foreach (string file in Directory.GetFiles(source))
                File.Copy(file, Path.Combine(destination, Path.GetFileName(file)));
            return destination;
        }
    }
}
