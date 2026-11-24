using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// F18 fresh-participant route. Every fixture is synthetic and temp-only, with one exception:
    /// the cold-start routing smoke tests read the approved common model assets
    /// (StreamingAssets/RecognitionCoreV1/Templates and RuntimeGate/frozen_threshold_config.json)
    /// read-only, and skip themselves when those assets are absent. No personal profile directory,
    /// no persistentDataPath, no camera, no writes outside temp. Temp directories are
    /// containment-checked before deletion.
    /// </summary>
    public sealed class DemoParticipantLiveRuntimeTests
    {
        const double CommonThreshold = 100.0;
        static readonly string[] Words = { "GREETING", "LOVE", "THANK_YOU" };

        readonly List<string> temporaryRoots = new List<string>();

        [SetUp]
        public void SetUp() => DemoParticipantSession.ResetStaticForTests();

        [TearDown]
        public void TearDown()
        {
            DemoParticipantSession.ResetStaticForTests();
            foreach (string root in temporaryRoots) DeleteTemporaryRoot(root);
            temporaryRoots.Clear();
        }

        // ------------------------------------------------------------------ (a) mapping

        [Test]
        public void MapGlobalResult_PreservesGlobalScoresRankingAndAcceptance()
        {
            RuntimeRecognitionResult global = GlobalResult(1.0, 2.0, 3.0, 1.0, 1.5, "GREETING", true);

            PersonalLiveRoutingResult mapped =
                DemoParticipantLiveRuntimeFactory.MapGlobalResult(global, "LOVE");

            Assert.AreEqual(1.0, mapped.Scores["GREETING"]);
            Assert.AreEqual(2.0, mapped.Scores["LOVE"]);
            Assert.AreEqual(3.0, mapped.Scores["THANK_YOU"]);
            Assert.AreEqual("GREETING", mapped.PredictedWord, "the expected word must not re-rank the classes");
            Assert.IsTrue(mapped.Accepted);
            Assert.AreEqual(1.0, mapped.BestDistance);
            Assert.AreEqual("proto-1", mapped.WinningPrototypeId);
            Assert.IsTrue(mapped.PerClassAccept["GREETING"]);
            Assert.IsFalse(mapped.PerClassAccept["LOVE"]);
            Assert.IsFalse(mapped.PerClassAccept["THANK_YOU"]);
            Assert.IsFalse(mapped.ExpectedWordClassAccept);
            Assert.IsFalse(mapped.QuestSuccess);
        }

        [Test]
        public void MapGlobalResult_QuestSuccessOnlyWhenPredictionMatchesAcceptedExpectedClass()
        {
            RuntimeRecognitionResult global = GlobalResult(1.0, 2.0, 3.0, 1.0, 1.5, "GREETING", true);

            Assert.IsTrue(DemoParticipantLiveRuntimeFactory.MapGlobalResult(global, "GREETING").QuestSuccess);
            Assert.IsFalse(DemoParticipantLiveRuntimeFactory.MapGlobalResult(global, "THANK_YOU").QuestSuccess);

            RuntimeRecognitionResult rejected = GlobalResult(2.0, 3.0, 4.0, 2.0, 1.5, "GREETING", false);
            Assert.IsFalse(DemoParticipantLiveRuntimeFactory.MapGlobalResult(rejected, "GREETING").QuestSuccess);
            Assert.IsFalse(DemoParticipantLiveRuntimeFactory.MapGlobalResult(rejected, "GREETING").Accepted);
        }

        [Test]
        public void MapGlobalResult_UsesTheSameGlobalThresholdIncludingEquality()
        {
            RuntimeRecognitionResult global = GlobalResult(1.5, 1.5, 1.6, 1.5, 1.5, "GREETING", true);

            PersonalLiveRoutingResult mapped =
                DemoParticipantLiveRuntimeFactory.MapGlobalResult(global, "LOVE");

            Assert.IsTrue(mapped.PerClassAccept["GREETING"], "distance == threshold accepts, as in the global gate");
            Assert.IsTrue(mapped.PerClassAccept["LOVE"]);
            Assert.IsFalse(mapped.PerClassAccept["THANK_YOU"]);
            Assert.IsTrue(mapped.ExpectedWordClassAccept);
            Assert.IsFalse(mapped.QuestSuccess, "prediction is GREETING, so LOVE cannot succeed");
        }

        // --------------------------------------------------------------- (b) controller

        [Test]
        public void Controller_RejectsInactiveRuntimeWithoutTheExplicitOptIn()
        {
            var runtime = new FakeInactiveRuntime();

            var controller = new PersonalLiveRecognitionController(
                new FakeSource(), new FakeFactory(runtime));

            Assert.IsFalse(controller.Initialize(Paths()));
            Assert.IsFalse(controller.IsRuntimeReady);
            controller.Dispose();
        }

        [Test]
        public void Controller_RejectsAnUnmarkedInactiveRuntimeEvenWithTheOptIn()
        {
            var controller = new PersonalLiveRecognitionController(
                new FakeSource(), new FakeFactory(new FakeInactiveRuntime()),
                allowDemoParticipantRuntime: true);

            Assert.IsFalse(controller.Initialize(Paths()));
            Assert.IsFalse(controller.IsRuntimeReady);
            controller.Dispose();
        }

        [Test]
        public void Controller_AcceptsAnExplicitlyMarkedPublicReferenceRuntimeWithoutClaimingPersonalMode()
        {
            var controller = new PersonalLiveRecognitionController(
                new FakeSource(), new FakeFactory(new FakePublicMarkerRuntime()),
                allowDemoParticipantRuntime: true);

            Assert.IsTrue(controller.Initialize(Paths()));
            Assert.IsTrue(controller.IsRuntimeReady);
            Assert.IsTrue(controller.IsDemoParticipantRuntime);
            Assert.IsFalse(controller.IsPersonalModeActive, "public reference mode is not a personal model");
            Assert.AreEqual(string.Empty, controller.ActivePersonalProfileId);
            Assert.IsFalse(string.IsNullOrEmpty(controller.RuntimeProvenanceId));
            Assert.AreEqual("public_reference_only_v1", controller.RuntimeProvenanceId);
            Assert.IsNotNull(controller.DemoRuntimeThresholds);

            Assert.IsTrue(controller.BeginCapture());
            Assert.IsTrue(controller.CancelCapture());
            Assert.IsTrue(controller.Reset());
            controller.Dispose();
        }

        // ------------------------------------------------------- (c) participant view

        [Test]
        public void CreateParticipantView_UsesOnlyTheNineCurrentParticipantCandidates()
        {
            DemoParticipantIdentity identity = NewIdentity();
            FakeTrustedRepository repository = BuildRepository(identity, loveCandidates: 3);

            TrustedPersonalizationActiveView view =
                DemoParticipantLiveRuntimeFactory.CreateParticipantViewWithCommonThreshold(
                    CommonThreshold, repository, identity);

            int prototypeCount = Words.Sum(word => view.Profile.Prototypes[word].Count);
            Assert.AreEqual(9, prototypeCount, "exactly the participant's own nine candidates");
            foreach (string word in Words)
            {
                Assert.AreEqual(3, view.Profile.Prototypes[word].Count);
                Assert.AreEqual(3, view.Profile.PrototypeProvenance[word].Count);
                Assert.AreEqual(3, view.CandidateCaptureIds[word].Count);
                Assert.AreEqual(CommonThreshold, view.Profile.Thresholds[word],
                    "the common threshold is used unchanged for every class");
            }

            IEnumerable<string> sampleIds = Words.SelectMany(
                word => view.Profile.Prototypes[word].Select(item => item.SampleId));
            Assert.IsTrue(sampleIds.All(id => id.StartsWith("cap_", StringComparison.Ordinal)),
                "no prototype may come from an older personal baseline");
            StringAssert.Contains(identity.LearnerId, view.ProfileId);
        }

        [Test]
        public void CreateParticipantView_RefusesAnotherParticipantsIdentity()
        {
            DemoParticipantIdentity identity = NewIdentity();
            FakeTrustedRepository repository = BuildRepository(identity, loveCandidates: 3);
            DemoParticipantIdentity other = NewIdentity();

            Assert.Throws<InvalidDataException>(() =>
                DemoParticipantLiveRuntimeFactory.CreateParticipantViewWithCommonThreshold(
                    CommonThreshold, repository, other));
        }

        [Test]
        public void CreateParticipantView_RefusesCountsThatDoNotMatchCandidateMembership()
        {
            DemoParticipantIdentity identity = NewIdentity();
            // Counts still claim 3/3/3 while only two LOVE candidates actually exist.
            FakeTrustedRepository repository = BuildRepository(identity, loveCandidates: 2);

            Assert.Throws<InvalidDataException>(() =>
                DemoParticipantLiveRuntimeFactory.CreateParticipantViewWithCommonThreshold(
                    CommonThreshold, repository, identity));
        }

        [Test]
        public void ParticipantRuntime_RefusesSelfReferenceButEvaluatesAFreshQuery()
        {
            DemoParticipantIdentity identity = NewIdentity();
            FakeTrustedRepository repository = BuildRepository(identity, loveCandidates: 3);
            IDemoParticipantLiveRuntime runtime =
                DemoParticipantLiveRuntimeFactory.CreateParticipantRuntimeWithCommonThreshold(
                    CommonThreshold, repository, identity);

            Assert.IsTrue(runtime.IsPersonalActive);
            Assert.AreEqual(runtime.ActiveProfileId, runtime.RuntimeIdentity);

            TrustedPersonalizationCandidate enrolled = repository.TrustedInventory.Candidates[0];

            RecognitionSequence self = Sequence(
                enrolled.CaptureSessionId, 1, enrolled.SourcePayloadSha256, enrolled.LogicalPayloadSha256);
            Assert.Throws<InvalidOperationException>(() =>
                runtime.Recognize(self.SampleId, self, "GREETING"));

            RecognitionSequence alias = Sequence(
                "alias_query", 1, enrolled.SourcePayloadSha256, enrolled.LogicalPayloadSha256);
            Assert.Throws<InvalidOperationException>(() =>
                runtime.Recognize(alias.SampleId, alias, "GREETING"));

            RecognitionSequence fresh = Sequence("fresh_query", 42, Hash("f1"), Hash("f2"));
            PersonalLiveRoutingResult result = runtime.Recognize(fresh.SampleId, fresh, "GREETING");

            // Evaluation must be possible; nothing here asserts that it succeeds.
            Assert.IsNotNull(result);
            Assert.AreEqual(3, result.Scores.Count);
            Assert.AreEqual(3, result.PerClassAccept.Count);
        }

        [Test]
        public void CreateParticipantView_RefusesAnInvalidCommonThreshold()
        {
            DemoParticipantIdentity identity = NewIdentity();
            FakeTrustedRepository repository = BuildRepository(identity, loveCandidates: 3);

            Assert.Throws<InvalidDataException>(() =>
                DemoParticipantLiveRuntimeFactory.CreateParticipantViewWithCommonThreshold(
                    -0.5, repository, identity));
            Assert.Throws<InvalidDataException>(() =>
                DemoParticipantLiveRuntimeFactory.CreateParticipantViewWithCommonThreshold(
                    double.NaN, repository, identity));

            // Zero stays a valid threshold value; nothing here tunes the actual common threshold.
            Assert.DoesNotThrow(() =>
                DemoParticipantLiveRuntimeFactory.CreateParticipantViewWithCommonThreshold(
                    0.0, repository, identity));
        }

        // -------------------------------------------------------- participant staleness

        [Test]
        public void ParticipantRuntime_BecomesNotReadyAndRefusesAfterTheParticipantChanges()
        {
            string root = NewTemporaryRoot();
            DemoParticipantSession.Activate(root);
            DemoParticipantIdentity identity = DemoParticipantSession.Current;
            FakeTrustedRepository repository = BuildRepository(identity, loveCandidates: 3);

            IDemoParticipantLiveRuntime runtime =
                DemoParticipantLiveRuntimeFactory.CreateParticipantRuntimeWithCommonThreshold(
                    CommonThreshold, repository, identity);
            Assert.IsTrue(runtime.IsReady);

            RecognitionSequence fresh = Sequence("fresh_query", 42, Hash("f1"), Hash("f2"));
            Assert.DoesNotThrow(() => runtime.Recognize(fresh.SampleId, fresh, "GREETING"));

            DemoParticipantSession.StartNew();

            Assert.IsFalse(runtime.IsReady, "a runtime for a previous participant is stale");
            Assert.Throws<InvalidOperationException>(() =>
                runtime.Recognize(fresh.SampleId, fresh, "GREETING"));

            DemoParticipantSession.ResetStaticForTests();
            Assert.IsFalse(runtime.IsReady, "an unavailable session is also stale");
        }

        [Test]
        public void PublicReferenceRuntime_BecomesNotReadyAndRefusesAfterTheParticipantChanges()
        {
            RuntimeRecognitionService service = LoadCommonServiceOrIgnore();

            string root = NewTemporaryRoot();
            DemoParticipantSession.Activate(root);
            var runtime = new PublicReferenceOnlyLiveRuntime(
                DemoParticipantSession.Current.LearnerId, service);
            Assert.IsTrue(runtime.IsReady);

            DemoParticipantSession.StartNew();

            Assert.IsFalse(runtime.IsReady, "a public-reference runtime is bound to its participant too");
            RecognitionSequence query = PublicSmokeQuery();
            Assert.Throws<InvalidOperationException>(() =>
                runtime.Recognize(query.SampleId, query, "GREETING"));
        }

        [Test]
        public void Controller_StopsBeingReadyWhenTheMarkedRuntimeGoesStale()
        {
            var marker = new FakePublicMarkerRuntime();
            var controller = new PersonalLiveRecognitionController(
                new FakeSource(), new FakeFactory(marker),
                allowDemoParticipantRuntime: true);

            Assert.IsTrue(controller.Initialize(Paths()));
            Assert.IsTrue(controller.IsRuntimeReady);

            marker.Ready = false;

            Assert.IsFalse(controller.IsRuntimeReady,
                "a marked opted-in runtime answers for its own readiness");
            Assert.IsFalse(controller.BeginCapture(), "a stale runtime must not start a capture");
            controller.Dispose();
        }

        [Test]
        public void Controller_StaleMarkedPersonalRuntimeIsNotReadyEvenThoughItIsPersonallyActive()
        {
            var marker = new FakePersonalMarkerRuntime { Ready = false };
            var controller = new PersonalLiveRecognitionController(
                new FakeSource(), new FakeFactory(marker),
                allowDemoParticipantRuntime: true);

            Assert.IsFalse(controller.Initialize(Paths()),
                "a stale marked runtime must not fall through to the legacy personal branch");
            Assert.IsFalse(controller.IsRuntimeReady);
            controller.Dispose();
        }

        // ------------------------------------------------------------- (d) paths contract

        [Test]
        public void RuntimePaths_RejectAMissingPersonalDirectoryUnlessExplicitlyAllowed()
        {
            Assert.Throws<ArgumentException>(() =>
                new PersonalLiveRuntimePaths("templates", "threshold.json", null));

            var paths = new PersonalLiveRuntimePaths(
                "templates", "threshold.json", null, allowNoPersonalProfileDirectory: true);

            Assert.IsNull(paths.PersonalProfileDirectory);
            Assert.IsTrue(paths.AllowsNoPersonalProfileDirectory);
            Assert.AreEqual("templates", paths.GlobalTemplateDirectory);
            Assert.AreEqual("threshold.json", paths.GlobalThresholdConfigPath);
        }

        // ------------------------------------------- (e) cold-start routing smoke/equality

        /// <summary>
        /// Reference-derived routing smoke over the approved common model only: it shows the
        /// cold-start route forwards the common service's own numbers unchanged. It is NOT a
        /// recognition-accuracy result and says nothing about how a new person will perform.
        /// </summary>
        [Test]
        public void ColdStartRoute_IsPublicOnlyAndRoutesTheCommonServiceResultUnchanged()
        {
            string templateDirectory = CommonTemplateDirectory();
            string thresholdPath = CommonThresholdPath();
            if (!Directory.Exists(templateDirectory) || !File.Exists(thresholdPath))
                Assert.Ignore("approved common model assets are not present in this checkout");

            // Empty temp participant area: no candidates, and no personal profile directory at all.
            DemoParticipantIdentity identity = NewIdentity();
            var factory = new DemoParticipantLiveRuntimeFactory(identity);
            var paths = new PersonalLiveRuntimePaths(
                templateDirectory, thresholdPath, null, allowNoPersonalProfileDirectory: true);

            IPersonalLiveGlobalRuntime global = factory.LoadGlobal(paths);
            var runtime = factory.CreateAndActivate(global, paths) as IDemoParticipantLiveRuntime;

            Assert.IsNotNull(runtime, "the cold-start route must return an explicitly marked runtime");
            Assert.IsTrue(runtime.IsReady);
            Assert.IsFalse(runtime.IsPersonalActive, "a cold start is not a personal model");
            Assert.AreEqual(string.Empty, runtime.ActiveProfileId);
            Assert.AreEqual("public_reference_only_v1", runtime.RuntimeIdentity);
            Assert.IsFalse(string.IsNullOrEmpty(runtime.RuntimeIdentity));

            RuntimeRecognitionService direct =
                RuntimeRecognitionService.Load(templateDirectory, thresholdPath);
            RecognitionSequence query = PublicSmokeQuery();

            RuntimeRecognitionResult expected = direct.Recognize(query.SampleId, query);
            PersonalLiveRoutingResult actual = runtime.Recognize(query.SampleId, query, "GREETING");

            Assert.AreEqual(expected.GreetingDistance, actual.Scores["GREETING"]);
            Assert.AreEqual(expected.LoveDistance, actual.Scores["LOVE"]);
            Assert.AreEqual(expected.ThankYouDistance, actual.Scores["THANK_YOU"]);
            Assert.AreEqual(expected.PredictedWord, actual.PredictedWord);
            Assert.AreEqual(expected.Accepted, actual.Accepted);
            Assert.AreEqual(expected.BestDistance, actual.BestDistance);
            Assert.AreEqual(expected.WinningPrototypeId, actual.WinningPrototypeId);
            foreach (string word in Words)
                Assert.AreEqual(
                    actual.Scores[word] <= expected.Threshold, actual.PerClassAccept[word],
                    "per-class acceptance uses the same common threshold");
        }

        // --------------------------------------------------------------------- fixtures

        static string CommonTemplateDirectory() =>
            Path.Combine(Application.streamingAssetsPath, "RecognitionCoreV1", "Templates");

        static string CommonThresholdPath() =>
            Path.Combine(
                Application.streamingAssetsPath, "RecognitionCoreV1", "RuntimeGate",
                "frozen_threshold_config.json");

        static RuntimeRecognitionService LoadCommonServiceOrIgnore()
        {
            string templateDirectory = CommonTemplateDirectory();
            string thresholdPath = CommonThresholdPath();
            if (!Directory.Exists(templateDirectory) || !File.Exists(thresholdPath))
                Assert.Ignore("approved common model assets are not present in this checkout");
            return RuntimeRecognitionService.Load(templateDirectory, thresholdPath);
        }

        /// <summary>
        /// An approved common template re-labelled as a runtime query: a train split is rejected
        /// as a query, so only the split/metadata changes. The payload is copied unmodified.
        /// </summary>
        static RecognitionSequence PublicSmokeQuery()
        {
            RecognitionSequence source =
                RecognitionTemplateLibrary.LoadDirectory(CommonTemplateDirectory()).Templates[0];
            return new RecognitionSequence(
                "f18_public_smoke",
                "UNLABELED",
                "RUNTIME",
                "runtime",
                source.SourceNpzSha256,
                source.LogicalPayloadSha256,
                source.Length,
                source.LandmarkCount,
                source.CopyCoordinates(),
                source.CopyValidMask());
        }


        static PersonalLiveRuntimePaths Paths() =>
            new PersonalLiveRuntimePaths("templates", "threshold.json", "personal");

        static RuntimeRecognitionResult GlobalResult(
            double greeting, double love, double thankYou,
            double best, double threshold, string predicted, bool accepted) =>
            new RuntimeRecognitionResult(
                "query-1", predicted, "proto-1", greeting, love, thankYou,
                best, threshold, accepted, Array.Empty<RuntimeTemplateDistance>());

        static RecognitionSequence Sequence(
            string sessionId, int variant, string sourceHash, string logicalHash)
        {
            const int frames = 5;
            var coordinates = new double[frames * RecognitionSequence.RequiredLandmarkCount * 3];
            var mask = Enumerable.Repeat(
                true, frames * RecognitionSequence.RequiredLandmarkCount).ToArray();
            for (int index = 0; index < coordinates.Length; index++)
                coordinates[index] = variant * 0.01 + index * 0.001;
            return new RecognitionSequence(
                sessionId, "UNLABELED", "RUNTIME", "runtime",
                sourceHash, logicalHash, frames,
                RecognitionSequence.RequiredLandmarkCount,
                coordinates, mask);
        }

        /// <summary>Distinct synthetic 64-character lowercase hex, one per seed.</summary>
        static string Hash(string seed)
        {
            var text = new System.Text.StringBuilder();
            foreach (char value in seed) text.Append(((int)value).ToString("x2"));
            return (text.ToString() + new string('0', 64)).Substring(0, 64);
        }

        string NewTemporaryRoot()
        {
            string root = Path.Combine(Path.GetTempPath(), "wis_f18_" + Guid.NewGuid().ToString("N"));
            temporaryRoots.Add(root);
            return root;
        }

        DemoParticipantIdentity NewIdentity() =>
            new DemoParticipantStore(NewTemporaryRoot()).GetOrCreate();

        /// <summary>
        /// Nine (or fewer, for the negative case) synthetic candidates with sentinel sequence
        /// files under candidateRoot/&lt;learnerId&gt;/candidates/&lt;word&gt;/&lt;capture&gt;/.
        /// The declared counts always stay 3/3/3.
        /// </summary>
        FakeTrustedRepository BuildRepository(DemoParticipantIdentity identity, int loveCandidates)
        {
            var candidates = new List<TrustedPersonalizationCandidate>();
            foreach (string word in Words)
            {
                int count = word == "LOVE" ? loveCandidates : 3;
                for (int index = 0; index < count; index++)
                {
                    string captureId = "cap_" + word + "_" + index;
                    string directory = Path.Combine(
                        identity.CandidateRoot, identity.LearnerId, "candidates", word, captureId);
                    Directory.CreateDirectory(directory);
                    File.WriteAllText(
                        Path.Combine(directory, "normalized_sequence.wisrq"), "synthetic-sentinel");
                    string source = Hash("src" + word + index);
                    string logical = Hash("log" + word + index);
                    candidates.Add(new TrustedPersonalizationCandidate(
                        captureId, word, directory,
                        Hash("file" + word + index), source, logical,
                        Sequence(captureId, index + 1, source, logical)));
                }
            }

            var counts = Words.ToDictionary(word => word, word => 3, StringComparer.Ordinal);
            return new FakeTrustedRepository(
                identity.LearnerId,
                identity.CandidateRoot,
                new TrustedCandidateInventory(
                    CandidateCountSnapshot.Empty, counts, candidates, 0));
        }

        static void DeleteTemporaryRoot(string root)
        {
            string full = Path.GetFullPath(root);
            string temp = Path.GetFullPath(Path.GetTempPath())
                .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            if (!full.StartsWith(temp + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("refusing to delete outside the temp area: " + full);
            if (Directory.Exists(full)) Directory.Delete(full, true);
        }

        sealed class FakeTrustedRepository : ITrustedPersonalizationCandidateRepository
        {
            public FakeTrustedRepository(string learnerId, string rootDirectory, TrustedCandidateInventory inventory)
            {
                LearnerId = learnerId;
                RootDirectory = rootDirectory;
                TrustedInventory = inventory;
            }

            public string RootDirectory { get; }
            public string LearnerId { get; }
            public CandidateCountSnapshot Counts => CandidateCountSnapshot.Empty;
            public TrustedCandidateInventory TrustedInventory { get; }

            public CandidateCountSnapshot ReloadAndVerify() => Counts;
            public TrustedCandidateInventory ReloadTrustedAndVerify() => TrustedInventory;

            public CandidateSaveResult Save(PersonalizationCandidateSaveRequest request) =>
                throw new NotSupportedException("the F18 tests never write candidates");
        }

        sealed class FakeSource : IPersonalLiveObservationSource
        {
            Action<RawMediaPipeObservation> observations;
            Action<Exception> failures;

            public bool IsBound { get; set; } = true;

            public event Action<RawMediaPipeObservation> ObservationArrived
            {
                add { observations += value; }
                remove { observations -= value; }
            }

            public event Action<Exception> SourceFailed
            {
                add { failures += value; }
                remove { failures -= value; }
            }

            public void Emit(RawMediaPipeObservation observation) => observations?.Invoke(observation);
            public void Fail(Exception exception) => failures?.Invoke(exception);
        }

        sealed class FakeGlobal : IPersonalLiveGlobalRuntime
        {
        }

        sealed class FakeFactory : IPersonalLiveRuntimeFactory
        {
            readonly IPersonalLiveRecognitionRuntime runtime;

            public FakeFactory(IPersonalLiveRecognitionRuntime runtime)
            {
                this.runtime = runtime;
            }

            public IPersonalLiveGlobalRuntime LoadGlobal(PersonalLiveRuntimePaths paths) => new FakeGlobal();

            public IPersonalLiveRecognitionRuntime CreateAndActivate(
                IPersonalLiveGlobalRuntime globalRuntime, PersonalLiveRuntimePaths paths) => runtime;
        }

        sealed class FakeInactiveRuntime : IPersonalLiveRecognitionRuntime
        {
            public bool IsPersonalActive => false;
            public string ActiveProfileId => string.Empty;

            public PersonalLiveRoutingResult Recognize(
                string sessionId, RecognitionSequence sequence, string expectedWord) =>
                throw new NotSupportedException();
        }

        sealed class FakePublicMarkerRuntime : IDemoParticipantLiveRuntime
        {
            public bool Ready { get; set; } = true;

            public string ParticipantId => "demo_test_participant";
            public string RuntimeIdentity => "public_reference_only_v1";
            public bool IsReady => Ready;
            public bool IsPersonalActive => false;
            public string ActiveProfileId => string.Empty;

            public IReadOnlyDictionary<string, double> Thresholds =>
                Words.ToDictionary(word => word, word => 1.0, StringComparer.Ordinal);

            public PersonalLiveRoutingResult Recognize(
                string sessionId, RecognitionSequence sequence, string expectedWord) =>
                throw new NotSupportedException("this fake is only used for lifecycle checks");
        }

        /// <summary>A marked runtime that is personally active but stale, to prove the controller
        /// does not fall back to the legacy personal branch for marked opted-in runtimes.</summary>
        sealed class FakePersonalMarkerRuntime : IDemoParticipantLiveRuntime
        {
            public bool Ready { get; set; } = true;

            public string ParticipantId => "demo_test_participant";
            public string RuntimeIdentity => "demo_participant_only_v1::demo_test_participant";
            public bool IsReady => Ready;
            public bool IsPersonalActive => true;
            public string ActiveProfileId => RuntimeIdentity;

            public IReadOnlyDictionary<string, double> Thresholds =>
                Words.ToDictionary(word => word, word => 1.0, StringComparer.Ordinal);

            public PersonalLiveRoutingResult Recognize(
                string sessionId, RecognitionSequence sequence, string expectedWord) =>
                throw new NotSupportedException("this fake is only used for lifecycle checks");
        }
    }
}
