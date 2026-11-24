using System;
using System.IO;
using NUnit.Framework;
using WordsInSilence.Practice.PetRobotVerticalSlice;

// NOTE(operator): PersonalizationCandidateRepository is referenced unqualified below, matching the
// bootstrap file. If this test assembly does not already resolve it, add its using here only.

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// Storage-identity tests only. No scene, no camera, no capture, no Application.persistentDataPath,
    /// and no claim about how well a new person is recognized (fixed-model isolation is F18).
    /// </summary>
    public sealed class DemoParticipantSessionTests
    {
        string testRoot;

        [SetUp]
        public void SetUp()
        {
            testRoot = Path.Combine(Path.GetTempPath(), "wis_f17_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(testRoot);
            // the handoff channel is process-wide; isolate each test from leftovers
            WordsInSilence.Practice.PetRobotWorldIntegration.PetRobotWorldHandoffChannel.Shared
                .ResetForNewParticipant();
        }

        static WordsInSilence.Practice.PetRobotWorldIntegration.PetRobotPracticeCompletionSnapshot
            CompletedSnapshot() =>
            new WordsInSilence.Practice.PetRobotWorldIntegration.PetRobotPracticeCompletionSnapshot(
                PetRobotWordLearningStatus.Mastered,
                PetRobotWordLearningStatus.Mastered,
                PetRobotWordLearningStatus.Mastered,
                3, 0, 12);

        [TearDown]
        public void TearDown()
        {
            DemoParticipantSession.ResetStaticForTests();
            // the handoff channel is process-wide; leave it clean for the next test
            WordsInSilence.Practice.PetRobotWorldIntegration.PetRobotWorldHandoffChannel.Shared
                .ResetForNewParticipant();
            DeleteInsideTestRoot(testRoot);
        }

        string DemoRoot(string name)
        {
            string path = Path.Combine(testRoot, name);
            Directory.CreateDirectory(path);
            return path;
        }

        /// <summary>Recursive delete is allowed only after the resolved target is proven to be the
        /// per-test temp root itself or a descendant of it.</summary>
        void DeleteInsideTestRoot(string path)
        {
            string root = Path.GetFullPath(testRoot);
            string target = Path.GetFullPath(path);
            bool inside = string.Equals(target, root, StringComparison.OrdinalIgnoreCase) ||
                          target.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase);
            Assert.IsTrue(inside, "refusing to delete outside the test temp root");
            if (Directory.Exists(target)) Directory.Delete(target, true);
        }

        static void WriteSentinel(string path, string content)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, content);
        }

        [Test]
        public void NewParticipant_HasZeroCandidates()
        {
            var store = new DemoParticipantStore(DemoRoot("demo"));
            DemoParticipantIdentity identity = store.GetOrCreate();

            var repository = new PersonalizationCandidateRepository(identity.LearnerId, identity.CandidateRoot);

            Assert.AreEqual(0, repository.Counts.TotalCandidateCount);
            StringAssert.StartsWith("demo_", identity.LearnerId);
        }

        [Test]
        public void Reopen_ReusesSavedIdentity()
        {
            string root = DemoRoot("demo");
            string first = new DemoParticipantStore(root).GetOrCreate().LearnerId;

            string second = new DemoParticipantStore(root).GetOrCreate().LearnerId;

            Assert.AreEqual(first, second);
        }

        [Test]
        public void StartNew_SwitchesIdentity_AndPreservesPreviousAndUnrelatedData()
        {
            string root = DemoRoot("demo");
            var store = new DemoParticipantStore(root);
            DemoParticipantIdentity previous = store.GetOrCreate();

            string oldCandidate = Path.Combine(previous.CandidateRoot, previous.LearnerId, "sentinel.json");
            string oldCalibration = store.ProfilePathFor(previous.LearnerId);
            string training = Path.Combine(testRoot, "training", "model.onnx");
            string aihub = Path.Combine(testRoot, "aihub", "source_meta.json");
            string video = Path.Combine(testRoot, "threeview", "front.mp4");
            WriteSentinel(oldCandidate, "old-candidate");
            WriteSentinel(oldCalibration, "old-calibration");
            WriteSentinel(training, "training");
            WriteSentinel(aihub, "aihub");
            WriteSentinel(video, "video");

            DemoParticipantIdentity fresh = store.StartNew();

            Assert.AreNotEqual(previous.LearnerId, fresh.LearnerId);
            Assert.AreEqual(0, new PersonalizationCandidateRepository(
                fresh.LearnerId, fresh.CandidateRoot).Counts.TotalCandidateCount);
            Assert.IsFalse(File.Exists(store.ProfilePathFor(fresh.LearnerId)),
                "a fresh participant must start with no calibration profile");
            // nothing is erased: previous personal records and unrelated datasets stay on disk
            Assert.AreEqual("old-candidate", File.ReadAllText(oldCandidate));
            Assert.AreEqual("old-calibration", File.ReadAllText(oldCalibration));
            Assert.AreEqual("training", File.ReadAllText(training));
            Assert.AreEqual("aihub", File.ReadAllText(aihub));
            Assert.AreEqual("video", File.ReadAllText(video));
            // and the previous participant is simply no longer the active one
            Assert.AreEqual(fresh.LearnerId, new DemoParticipantStore(root).GetOrCreate().LearnerId);
        }

        [Test]
        public void HandoffChannel_ResetForNewParticipant_ClearsPendingCompletion()
        {
            var channel = WordsInSilence.Practice.PetRobotWorldIntegration.PetRobotWorldHandoffChannel.Shared;
            Assert.IsTrue(channel.Begin("World", "Practice", UnityEngine.Vector3.zero));
            Assert.IsTrue(channel.PublishCompleted(CompletedSnapshot()));
            Assert.IsTrue(channel.HasPendingEntry);
            Assert.IsTrue(channel.HasPublishedCompletion);

            channel.ResetForNewParticipant();

            // a previous participant's completion must not survive the scene reload
            Assert.IsFalse(channel.HasPendingEntry);
            Assert.IsFalse(channel.HasPublishedCompletion);
        }

        [Test]
        public void HandoffChannel_ResetForNewParticipant_ClearsQuestResult()
        {
            var channel = WordsInSilence.Practice.PetRobotWorldIntegration.PetRobotWorldHandoffChannel.Shared;
            var token = new WordsInSilence.Practice.PetRobotWorldIntegration.PetRobotGreetingQuestToken();
            Assert.IsTrue(channel.BeginGreetingQuest("World", "Practice", UnityEngine.Vector3.zero,
                token, CompletedSnapshot(),
                new WordsInSilence.Practice.PetRobotWorldIntegration.PetRobotLearnerRootIdentity(
                    "demo_00000000000000000000000000000000", null)));
            Assert.IsTrue(channel.PublishQuestResult(
                new WordsInSilence.Practice.PetRobotWorldIntegration.PetRobotGreetingQuestResult(
                    token.Id, TrustedHybridOutcomeKind.StrictSuccess, "feedback", null, null)));
            Assert.IsTrue(channel.HasQuestResult);

            channel.ResetForNewParticipant();

            // a previous participant's quest result must not re-award the quest
            Assert.IsFalse(channel.HasQuestResult);
            Assert.IsFalse(channel.HasPendingEntry);
        }

        [Test]
        public void ActiveSession_ResolvesEmptyCandidateRoot_WithoutEnrollment()
        {
            DemoParticipantSession.Activate(DemoRoot("demo"));
            DemoParticipantIdentity identity = DemoParticipantSession.Current;

            string root;
            bool resolved = PetRobotPlayerCandidateRootOverride.TryResolveForPlayer(identity.LearnerId, out root);

            Assert.IsTrue(resolved, "the empty-start route must not be blocked by the startup gate");
            Assert.AreEqual(identity.CandidateRoot, root);
            Assert.AreEqual(identity.ProfilesRoot, DemoParticipantSession.ProfilesRootOrNull);
        }

        [Test]
        public void ActiveSession_LearnerMismatch_FailsClosed()
        {
            DemoParticipantSession.Activate(DemoRoot("demo"));

            string root;
            Assert.Throws<InvalidOperationException>(
                () => PetRobotPlayerCandidateRootOverride.TryResolveForPlayer("local_learner", out root));
        }

        [Test]
        public void WithoutSession_LegacyResolutionIsUnchanged()
        {
            DemoParticipantSession.ResetStaticForTests();

            string root;
            bool resolved = PetRobotPlayerCandidateRootOverride.TryResolveForPlayer("local_learner", out root);

            // no CLI flag in this run: the legacy path stays "not requested, not overridden"
            Assert.IsTrue(resolved);
            Assert.IsNull(root);
            Assert.IsFalse(DemoParticipantSession.IsActive);
            Assert.IsNull(DemoParticipantSession.ProfilesRootOrNull);
        }

        [TestCase("{\"schema\":1,\"learnerId\":\"local_learner\"}")]
        [TestCase("{\"schema\":1.5,\"learnerId\":\"demo_0123456789abcdef0123456789abcdef\"}")]
        [TestCase("{\"schema\":1,\"learnerId\":\"demo_0123456789ABCDEF0123456789abcdef\"}")]
        [TestCase("{\"schema\":1,\"learnerId\":\"demo_0123456789abcdef\"}")]
        [TestCase("{\"learnerId\":\"demo_0123456789abcdef0123456789abcdef\",\"schema\":1}")]
        [TestCase("{\"schema\":1,\"schema\":1,\"learnerId\":\"demo_0123456789abcdef0123456789abcdef\"}")]
        [TestCase("{\"schema\":1,\"learnerId\":\"demo_0123456789abcdef0123456789abcdef\"} trailing")]
        [TestCase("{\"schema\":1,\"learnerId\":\"demo_0123456789abcdef0123456789abcdef\"")]
        public void InvalidManifest_FailsClosed(string manifestText)
        {
            string root = DemoRoot("demo");
            File.WriteAllText(Path.Combine(root, "active.json"), manifestText);

            Assert.Throws<InvalidOperationException>(() => new DemoParticipantStore(root).GetOrCreate());

            Assert.AreEqual(manifestText, File.ReadAllText(Path.Combine(root, "active.json")));
        }

        [TestCase("C:relative_demo_root")]
        [TestCase("\\rooted_current_drive_demo")]
        [TestCase("relative_demo_root")]
        public void NotFullyQualifiedRoot_IsRejected(string root)
        {
            Assert.Catch(() => new DemoParticipantStore(root));
        }

        [Test]
        public void FailedActivation_NeverFallsBackToLegacy()
        {
            Assert.Catch(() => DemoParticipantSession.Activate("C:relative_demo_root"));

            Assert.IsTrue(DemoParticipantSession.IsActive, "a failed activation still blocks legacy use");
            Assert.Throws<InvalidOperationException>(() => { var _ = DemoParticipantSession.Current; });
            Assert.Throws<InvalidOperationException>(
                () => { var _ = DemoParticipantSession.ProfilesRootOrNull; });
        }

        [Test]
        public void LockedManifest_ResetFails_AndKeepsCurrentIdentityAndBytes()
        {
            string root = DemoRoot("demo");
            DemoParticipantSession.Activate(root);
            string originalId = DemoParticipantSession.Current.LearnerId;
            string manifest = Path.Combine(root, "active.json");
            byte[] originalBytes = File.ReadAllBytes(manifest);

            using (new FileStream(manifest, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
            {
                Assert.Throws<InvalidOperationException>(() => DemoParticipantSession.StartNew());
            }

            Assert.AreEqual(originalId, DemoParticipantSession.Current.LearnerId);
            Assert.AreEqual(originalBytes, File.ReadAllBytes(manifest));
            Assert.AreEqual(originalId, new DemoParticipantStore(root).GetOrCreate().LearnerId);
        }
    }
}
