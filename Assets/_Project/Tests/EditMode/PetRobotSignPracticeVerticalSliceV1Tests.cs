using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class PetRobotSignPracticeVerticalSliceV1Tests
    {
        const double GreetingThreshold = 0.5113059473720765;
        const double LoveThreshold = 0.37312048979362906;
        const double ThankYouThreshold = 0.37709977431991587;
        string tempRoot;

        [SetUp]
        public void SetUp()
        {
            tempRoot = Path.Combine(Path.GetTempPath(), "wis_step10a_" + Guid.NewGuid().ToString("N"));
        }

        [TearDown]
        public void TearDown()
        {
            if (Directory.Exists(tempRoot)) Directory.Delete(tempRoot, true);
        }

        [Test]
        public void PracticeOrder_IsGreetingLoveThankYou()
        {
            var session = NewSession(out _, out _);
            CollectionAssert.AreEqual(new[] { "GREETING", "LOVE", "THANK_YOU" }, session.Words);
            Assert.AreEqual("GREETING", session.CurrentWord);
        }

        [Test]
        public void Step9NGreetingFixedResult_IsStrictSuccess()
        {
            GuidedQuestOutcome outcome = Outcome("GREETING", "GREETING", true, true, true);
            Assert.AreEqual(GuidedQuestOutcomeKind.StrictSuccess, outcome.Kind);
            Assert.IsTrue(outcome.GuidedProgressAllowed);
        }

        [Test]
        public void Step9QLoveFixedResult_IsStrictSuccess()
        {
            GuidedQuestOutcome outcome = Outcome("LOVE", "LOVE", true, true, true);
            Assert.AreEqual(GuidedQuestOutcomeKind.StrictSuccess, outcome.Kind);
            Assert.IsTrue(outcome.StrictQuestSuccess);
        }

        [Test]
        public void Step9RThankYouFixedResult_IsLowConfidenceCorrectWithStrictFailurePreserved()
        {
            GuidedQuestOutcome outcome = Outcome("THANK_YOU", "THANK_YOU", false, false, false);
            Assert.AreEqual(GuidedQuestOutcomeKind.LowConfidenceCorrect, outcome.Kind);
            Assert.IsFalse(outcome.StrictAccepted);
            Assert.IsFalse(outcome.StrictExpectedWordClassAccept);
            Assert.IsFalse(outcome.StrictQuestSuccess);
            Assert.IsTrue(outcome.GuidedProgressAllowed);
        }

        [Test]
        public void FourOutcomeBranches_RemainDistinct()
        {
            Assert.AreEqual(GuidedQuestOutcomeKind.StrictSuccess, Outcome("GREETING", "GREETING", true, true, true).Kind);
            Assert.AreEqual(GuidedQuestOutcomeKind.LowConfidenceCorrect, Outcome("THANK_YOU", "THANK_YOU", false, false, false).Kind);
            Assert.AreEqual(GuidedQuestOutcomeKind.WrongWord, Outcome("GREETING", "LOVE", false, false, false).Kind);
            Assert.AreEqual(GuidedQuestOutcomeKind.TechnicalFailure,
                GuidedQuestOutcomeEvaluator.EvaluateForRoute(PersonalLiveProfileMode.MultiPrototypeCandidate,
                    PersonalLiveCaptureState.Failed, false, "GREETING", string.Empty, false, false, false, true, true).Kind);
        }

        [Test]
        public void StrictSuccess_ProgressesAndStoresEligible()
        {
            var session = NewSession(out FakeRecognitionAdapter adapter, out FakeRepository repository);
            adapter.Next = Attempt("s-greeting", "GREETING", "GREETING", true, true, true);
            Assert.IsTrue(session.StartExpression());
            Assert.IsTrue(session.FinishExpression());
            Assert.IsTrue(session.IsCurrentWordComplete);
            Assert.AreEqual(PersonalizationCandidateEligibility.Eligible, repository.LastEligibility);
            Assert.AreEqual(1, repository.SaveCalls);
            Assert.AreEqual(1, session.CandidateCounts.Eligible["GREETING"]);
        }

        [Test]
        public void LowConfidenceCorrect_ProgressesAndStoresReviewOnlyWithoutEligibleCount()
        {
            var session = NewSession(out FakeRecognitionAdapter adapter, out FakeRepository repository);
            adapter.Next = Attempt("s-thanks", "GREETING", "GREETING", false, false, false);
            session.StartExpression();
            session.FinishExpression();
            Assert.IsTrue(session.IsCurrentWordComplete);
            Assert.AreEqual(PersonalizationCandidateEligibility.ReviewOnly, repository.LastEligibility);
            Assert.AreEqual(0, session.CandidateCounts.Eligible["GREETING"]);
            Assert.AreEqual(1, session.CandidateCounts.ReviewOnly["GREETING"]);
        }

        [Test]
        public void WrongWord_DoesNotProgressOrStore()
        {
            var session = NewSession(out FakeRecognitionAdapter adapter, out FakeRepository repository);
            adapter.Next = Attempt("s-wrong", "GREETING", "LOVE", false, false, false);
            session.StartExpression();
            session.FinishExpression();
            Assert.IsFalse(session.IsCurrentWordComplete);
            Assert.AreEqual(0, repository.SaveCalls);
            Assert.AreEqual(CandidateStorageStatus.NotRequested, session.LastStorageResult.Status);
        }

        [Test]
        public void TechnicalFailure_DoesNotProgressStoreOrScoreLearner()
        {
            var session = NewSession(out FakeRecognitionAdapter adapter, out FakeRepository repository);
            adapter.Next = TechnicalAttempt("s-tech", "GREETING");
            session.StartExpression();
            session.FinishExpression();
            Assert.AreEqual(GuidedQuestOutcomeKind.TechnicalFailure, session.LastLearningOutcome.Kind);
            Assert.IsFalse(session.LastLearningOutcome.GuidedProgressAllowed);
            Assert.IsFalse(session.IsCurrentWordComplete);
            Assert.AreEqual(0, repository.SaveCalls);
        }

        [Test]
        public void DuplicateFinalizeAndCallback_ProgressAndSaveOnlyOnce()
        {
            var session = NewSession(out FakeRecognitionAdapter adapter, out FakeRepository repository);
            adapter.Next = Attempt("s-duplicate", "GREETING", "GREETING", true, true, true);
            session.StartExpression();
            Assert.IsTrue(session.FinishExpression());
            Assert.IsFalse(session.FinishExpression());
            Assert.AreEqual(1, adapter.FinalizeCalls);
            Assert.AreEqual(1, repository.SaveCalls);
            Assert.AreEqual(1, session.CandidateCounts.Eligible["GREETING"]);
        }

        [Test]
        public void DuplicateSessionCallbackAcrossWords_IsIgnoredWithoutSecondProgressOrSave()
        {
            var session = NewSession(out FakeRecognitionAdapter adapter, out FakeRepository repository);
            adapter.Next = Attempt("same-session", "GREETING", "GREETING", true, true, true);
            session.StartExpression();
            session.FinishExpression();
            session.ContinueToNextWord();
            adapter.Next = Attempt("same-session", "LOVE", "LOVE", true, true, true);
            session.StartExpression();
            session.FinishExpression();
            Assert.IsFalse(session.IsWordComplete("LOVE"));
            Assert.AreEqual(1, repository.SaveCalls);
            Assert.AreEqual(CandidateStorageStatus.Duplicate, session.LastStorageResult.Status);
        }

        [Test]
        public void StorageFailure_DoesNotChangeLearningSuccess()
        {
            var adapter = new FakeRecognitionAdapter();
            var repository = new ThrowingRepository();
            var session = new PetRobotSignPracticeSession(adapter, repository);
            adapter.Next = Attempt("s-store-fail", "GREETING", "GREETING", true, true, true);
            session.StartExpression();
            session.FinishExpression();
            Assert.IsTrue(session.IsCurrentWordComplete);
            Assert.AreEqual(GuidedQuestOutcomeKind.StrictSuccess, session.LastLearningOutcome.Kind);
            Assert.AreEqual(CandidateStorageStatus.Failed, session.LastStorageResult.Status);
        }

        [Test]
        public void Repository_WisrqRoundTripChecksumManifestAndReloadCountsMatch()
        {
            var repository = new PersonalizationCandidateRepository("learner_a", tempRoot);
            CandidateSaveResult saved = repository.Save(new PersonalizationCandidateSaveRequest(
                Attempt("repo-roundtrip", "GREETING", "GREETING", true, true, true),
                PersonalizationCandidateEligibility.Eligible,
                new DateTime(2026, 8, 10, 12, 0, 0, DateTimeKind.Utc)));
            Assert.AreEqual(CandidateStorageStatus.Saved, saved.Status);
            Assert.AreEqual(1, repository.ReloadAndVerify().Eligible["GREETING"]);
            string[] files = Directory.GetFiles(saved.CandidateDirectory, "*", SearchOption.TopDirectoryOnly);
            CollectionAssert.AreEquivalent(
                new[] { "normalized_sequence.wisrq", "candidate_metadata.json", "checksum_manifest.json" },
                files.Select(Path.GetFileName).ToArray());
            RecognitionSequence loaded = RecognitionSequenceBinary.LoadFile(
                Path.Combine(saved.CandidateDirectory, "normalized_sequence.wisrq"));
            Assert.AreEqual("repo-roundtrip", loaded.SampleId);
            StringAssert.Contains("\"eligible_for_profile\": true",
                File.ReadAllText(Path.Combine(saved.CandidateDirectory, "candidate_metadata.json")));
        }

        [Test]
        public void Repository_ReviewOnlyDoesNotIncrementEligibleCount()
        {
            var repository = new PersonalizationCandidateRepository("learner_b", tempRoot);
            CandidateSaveResult result = repository.Save(new PersonalizationCandidateSaveRequest(
                Attempt("repo-review", "THANK_YOU", "THANK_YOU", false, false, false),
                PersonalizationCandidateEligibility.ReviewOnly, DateTime.UtcNow));
            Assert.AreEqual(CandidateStorageStatus.Saved, result.Status);
            Assert.AreEqual(0, result.Counts.Eligible["THANK_YOU"]);
            Assert.AreEqual(1, result.Counts.ReviewOnly["THANK_YOU"]);
            StringAssert.Contains("\"eligible_for_profile\": false",
                File.ReadAllText(Path.Combine(result.CandidateDirectory, "candidate_metadata.json")));
        }

        // R4: scoped to the fault points a BASIC Save actually reaches, and each one is
        // asserted to have been REACHED before its rollback is asserted. The guided-seed hook
        // BeforeGuidedEnrollmentSeedManifestMove is never reached by Save; it is covered by
        // GuidedLowConfidenceSeedPromotionManifestV1Tests
        //     .AtomicPreMoveFailureLeavesNoTargetAndCleansTemporaryFile,
        // which is unchanged. The repository itself is unchanged.
        [Test]
        public void Repository_BasicSaveInjectedFailuresRollbackFinalAndTemporaryOutputs()
        {
            foreach (CandidateRepositoryFailurePoint point in new[]
            {
                CandidateRepositoryFailurePoint.AfterSequenceWrite,
                CandidateRepositoryFailurePoint.AfterMetadataWrite,
                CandidateRepositoryFailurePoint.AfterFinalDirectoryMove,
                CandidateRepositoryFailurePoint.AfterIndexCommitBeforeComplete,
            })
            {
                string isolated = Path.Combine(tempRoot, point.ToString());
                bool reached = false;
                var repository = new PersonalizationCandidateRepository("learner", isolated,
                    current =>
                    {
                        if (current != point) return;
                        reached = true;
                        throw new IOException("injected " + point);
                    });
                CandidateSaveResult result = repository.Save(new PersonalizationCandidateSaveRequest(
                    Attempt("failure-" + point, "LOVE", "LOVE", true, true, true),
                    PersonalizationCandidateEligibility.Eligible, DateTime.UtcNow));
                Assert.IsTrue(reached,
                    "the injected fault point must actually be reached by Save: " + point);
                Assert.AreEqual(CandidateStorageStatus.Failed, result.Status, point.ToString());
                string learnerRoot = Path.Combine(isolated, "learner");
                Assert.AreEqual(0, Directory.Exists(learnerRoot)
                    ? Directory.GetFiles(learnerRoot, "*", SearchOption.AllDirectories).Length : 0, point.ToString());
            }
        }

        [Test]
        public void Repository_NeverOverwritesCommittedCapture()
        {
            var repository = new PersonalizationCandidateRepository("learner_c", tempRoot);
            var request = new PersonalizationCandidateSaveRequest(
                Attempt("never-overwrite", "LOVE", "LOVE", true, true, true),
                PersonalizationCandidateEligibility.Eligible, DateTime.UtcNow);
            CandidateSaveResult first = repository.Save(request);
            string sequencePath = Path.Combine(first.CandidateDirectory, "normalized_sequence.wisrq");
            string before = Sha256(File.ReadAllBytes(sequencePath));
            CandidateSaveResult second = repository.Save(request);
            Assert.AreEqual(CandidateStorageStatus.Duplicate, second.Status);
            Assert.AreEqual(before, Sha256(File.ReadAllBytes(sequencePath)));
            Assert.AreEqual(1, repository.Counts.Eligible["LOVE"]);
        }

        [Test]
        public void Repository_NeverStoresVideoImageOrAudio()
        {
            var repository = new PersonalizationCandidateRepository("learner_d", tempRoot);
            repository.Save(new PersonalizationCandidateSaveRequest(
                Attempt("no-media", "GREETING", "GREETING", true, true, true),
                PersonalizationCandidateEligibility.Eligible, DateTime.UtcNow));
            string[] prohibited = { ".mp4", ".mov", ".avi", ".webm", ".png", ".jpg", ".jpeg", ".wav", ".mp3", ".ogg" };
            Assert.IsFalse(Directory.GetFiles(tempRoot, "*", SearchOption.AllDirectories)
                .Any(path => prohibited.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase)));
        }

        [Test]
        public void CandidateStorage_DoesNotChangeActiveProfilesPrototypesOrThresholds()
        {
            string profileRoot = Path.Combine(Application.streamingAssetsPath, "RecognitionCoreV1", "PersonalProfiles");
            Dictionary<string, string> before = HashTree(profileRoot);
            var repository = new PersonalizationCandidateRepository("learner_e", tempRoot);
            repository.Save(new PersonalizationCandidateSaveRequest(
                Attempt("asset-preserve", "GREETING", "GREETING", true, true, true),
                PersonalizationCandidateEligibility.Eligible, DateTime.UtcNow));
            CollectionAssert.AreEquivalent(before, HashTree(profileRoot));
        }

        [Test]
        public void LegacyOrDefaultRoute_RemainsInactiveForVerticalSlice()
        {
            var adapter = new FakeRecognitionAdapter { CandidateActive = false };
            Assert.Throws<InvalidOperationException>(() =>
                new PetRobotSignPracticeSession(adapter, new FakeRepository()));
            Assert.AreEqual(0, adapter.BeginCalls);
        }

        [Test]
        public void ControllerGate_RequiresManualRetryBeforeAnotherCapture()
        {
            var session = NewSession(out FakeRecognitionAdapter adapter, out _);
            adapter.Next = Attempt("wrong-manual", "GREETING", "LOVE", false, false, false);
            session.StartExpression();
            session.FinishExpression();
            Assert.IsFalse(session.StartExpression());
            Assert.IsTrue(session.Retry());
            Assert.AreEqual(PetRobotPracticeState.Ready, session.State);
            Assert.AreEqual(1, adapter.ResetCalls);
            Assert.IsTrue(session.StartExpression());
        }

        [Test]
        public void ThreeCompletedWords_FreezeUntilExplicitExit()
        {
            var session = NewSession(out FakeRecognitionAdapter adapter, out FakeRepository repository);
            foreach (string word in new[] { "GREETING", "LOVE", "THANK_YOU" })
            {
                adapter.Next = Attempt("complete-" + word, word, word,
                    word != "THANK_YOU", word != "THANK_YOU", word != "THANK_YOU");
                Assert.IsTrue(session.StartExpression());
                Assert.IsTrue(session.FinishExpression());
                Assert.IsTrue(session.ContinueToNextWord());
            }
            Assert.AreEqual(PetRobotPracticeState.Completed, session.State);
            Assert.IsTrue(session.AllWordsComplete);
            Assert.IsFalse(session.StartExpression());
            Assert.AreEqual(2, repository.Counts.Eligible.Values.Sum());
            Assert.AreEqual(1, repository.Counts.ReviewOnly.Values.Sum());
            Assert.IsTrue(session.ExitPractice());
            Assert.AreEqual(PetRobotPracticeState.Exited, session.State);
        }

        [Test]
        public void ApprovedBinaryWriter_RoundTripsCoordinatesMaskAndEmbeddedChecksum()
        {
            RecognitionSequence source = Sequence("writer-roundtrip", 4);
            RecognitionSequence loaded = RecognitionSequenceBinary.Load(RecognitionSequenceCandidateBinary.Write(source));
            Assert.AreEqual(source.SampleId, loaded.SampleId);
            Assert.AreEqual(source.Length, loaded.Length);
            CollectionAssert.AreEqual(source.CopyCoordinates(), loaded.CopyCoordinates());
            CollectionAssert.AreEqual(source.CopyValidMask(), loaded.CopyValidMask());
            Assert.IsTrue(RecognitionSequence.IsLowerHexSha256(loaded.BinaryPayloadSha256));
        }

        [Test]
        public void TechnicalPanelToggle_DefaultsVisibleAndCanBeHiddenWithoutChangingRoute()
        {
            var gameObject = new GameObject("harness-toggle-test");
            try
            {
                var harness = gameObject.AddComponent<PersonalLiveRecognitionHarness>();
                Assert.IsTrue(harness.TechnicalPanelVisible);
                PersonalLiveProfileMode before = harness.ProfileMode;
                harness.SetTechnicalPanelVisible(false);
                Assert.IsFalse(harness.TechnicalPanelVisible);
                Assert.AreEqual(before, harness.ProfileMode);
            }
            finally { UnityEngine.Object.DestroyImmediate(gameObject); }
        }

        static PetRobotSignPracticeSession NewSession(
            out FakeRecognitionAdapter adapter,
            out FakeRepository repository)
        {
            adapter = new FakeRecognitionAdapter();
            repository = new FakeRepository();
            return new PetRobotSignPracticeSession(adapter, repository,
                () => new DateTime(2026, 8, 10, 12, 0, 0, DateTimeKind.Utc));
        }

        static PracticeRecognitionAttempt Attempt(
            string sessionId,
            string expected,
            string predicted,
            bool accepted,
            bool expectedAccept,
            bool questSuccess)
        {
            GuidedQuestOutcome outcome = Outcome(expected, predicted, accepted, expectedAccept, questSuccess);
            double greeting = expected == "GREETING" ? 0.43359546744230326 : 0.73;
            double love = expected == "LOVE" ? 0.33708012556848395 : 0.68;
            double thanks = expected == "THANK_YOU" ? 0.44939933821265732 : 0.77;
            if (predicted == "LOVE" && expected != "LOVE") love = 0.3;
            double threshold = expected == "GREETING" ? GreetingThreshold : expected == "LOVE" ? LoveThreshold : ThankYouThreshold;
            return new PracticeRecognitionAttempt(
                sessionId, expected, predicted, PersonalLiveCaptureState.Completed, outcome, null,
                Sequence(sessionId, 5), PersonalLiveProfileMode.MultiPrototypeCandidate.ToString(),
                "step9l_current_user_multi_prototype_v1_0", threshold, null, null,
                new PracticeRecognitionMetrics
                {
                    StrictAccepted = accepted,
                    StrictExpectedWordClassAccept = expectedAccept,
                    StrictQuestSuccess = questSuccess,
                    GreetingDistance = greeting,
                    LoveDistance = love,
                    ThankYouDistance = thanks,
                    BestDistance = Math.Min(greeting, Math.Min(love, thanks)),
                    WinningPrototypeId = "prototype-" + predicted,
                    RawFrameCount = 5,
                    NormalizedFrameCount = 5,
                    ValidLandmarkCount = 80,
                    InterpolatedLandmarkCount = 0,
                });
        }

        static PracticeRecognitionAttempt TechnicalAttempt(string sessionId, string expected)
        {
            GuidedQuestOutcome outcome = GuidedQuestOutcomeEvaluator.EvaluateForRoute(
                PersonalLiveProfileMode.MultiPrototypeCandidate, PersonalLiveCaptureState.Failed,
                false, expected, string.Empty, false, false, false, true, true);
            return new PracticeRecognitionAttempt(
                sessionId, expected, string.Empty, PersonalLiveCaptureState.Failed, outcome, null, null,
                PersonalLiveProfileMode.MultiPrototypeCandidate.ToString(),
                "step9l_current_user_multi_prototype_v1_0", GreetingThreshold, null, null,
                new PracticeRecognitionMetrics { WinningPrototypeId = string.Empty });
        }

        static GuidedQuestOutcome Outcome(string expected, string predicted, bool accepted, bool expectedAccept, bool questSuccess)
        {
            return GuidedQuestOutcomeEvaluator.EvaluateForRoute(
                PersonalLiveProfileMode.MultiPrototypeCandidate, PersonalLiveCaptureState.Completed,
                true, expected, predicted, accepted, expectedAccept, questSuccess, false, false);
        }

        static RecognitionSequence Sequence(string sessionId, int frames)
        {
            var coordinates = new double[frames * RecognitionSequence.RequiredLandmarkCount * 3];
            var mask = Enumerable.Repeat(true, frames * RecognitionSequence.RequiredLandmarkCount).ToArray();
            for (int index = 0; index < coordinates.Length; index++) coordinates[index] = index * 0.001;
            return new RecognitionSequence(sessionId, "UNLABELED", "RUNTIME", "runtime",
                new string('a', 64), new string('b', 64), frames,
                RecognitionSequence.RequiredLandmarkCount, coordinates, mask);
        }

        static Dictionary<string, string> HashTree(string root)
        {
            var result = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (string path in Directory.GetFiles(root, "*", SearchOption.AllDirectories))
                result[path.Substring(root.Length).Replace('\\', '/')] = Sha256(File.ReadAllBytes(path));
            return result;
        }

        static string Sha256(byte[] bytes)
        {
            using (var sha = SHA256.Create())
                return string.Concat(sha.ComputeHash(bytes).Select(value => value.ToString("x2")));
        }

        sealed class FakeRecognitionAdapter : IPersonalLivePracticeRecognitionAdapter
        {
            public bool CandidateActive = true;
            public PracticeRecognitionAttempt Next;
            public int BeginCalls;
            public int FinalizeCalls;
            public int ResetCalls;
            string expected = "GREETING";
            PersonalLiveCaptureState state = PersonalLiveCaptureState.Ready;

            public PersonalLiveCaptureState CaptureState => state;
            public string CaptureSessionId => Next == null ? string.Empty : Next.CaptureSessionId;
            public string ActiveProfileId => "step9l_current_user_multi_prototype_v1_0";
            public bool CandidateRouteActive => CandidateActive;
            public bool SetExpectedWord(string word)
            {
                if (state != PersonalLiveCaptureState.Ready || !PersonalLiveRecognitionController.IsSupportedWord(word)) return false;
                expected = word;
                return true;
            }
            public bool BeginCapture()
            {
                if (!CandidateActive || state != PersonalLiveCaptureState.Ready) return false;
                BeginCalls++;
                state = PersonalLiveCaptureState.Capturing;
                return true;
            }
            public PracticeRecognitionAttempt FinalizeCapture()
            {
                FinalizeCalls++;
                state = Next == null ? PersonalLiveCaptureState.Failed : Next.CaptureState;
                return Next;
            }
            public bool ResetCapture()
            {
                if (state != PersonalLiveCaptureState.Completed && state != PersonalLiveCaptureState.Failed) return false;
                ResetCalls++;
                state = PersonalLiveCaptureState.Ready;
                return true;
            }
        }

        sealed class FakeRepository : IPersonalizationCandidateRepository
        {
            readonly Dictionary<string, int> eligible = Words(0);
            readonly Dictionary<string, int> review = Words(0);
            public int SaveCalls;
            public PersonalizationCandidateEligibility LastEligibility;
            public string RootDirectory => "fake";
            public string LearnerId => "fake";
            public CandidateCountSnapshot Counts => new CandidateCountSnapshot(eligible, review);
            public CandidateCountSnapshot ReloadAndVerify() => Counts;
            public CandidateSaveResult Save(PersonalizationCandidateSaveRequest request)
            {
                SaveCalls++;
                LastEligibility = request.Eligibility;
                if (request.Eligibility == PersonalizationCandidateEligibility.Eligible) eligible[request.Attempt.ExpectedWord]++;
                else review[request.Attempt.ExpectedWord]++;
                return new CandidateSaveResult(CandidateStorageStatus.Saved, "fake", "saved", Counts);
            }
        }

        sealed class ThrowingRepository : IPersonalizationCandidateRepository
        {
            public string RootDirectory => "fake";
            public string LearnerId => "fake";
            public CandidateCountSnapshot Counts => CandidateCountSnapshot.Empty;
            public CandidateCountSnapshot ReloadAndVerify() => Counts;
            public CandidateSaveResult Save(PersonalizationCandidateSaveRequest request) =>
                throw new IOException("injected storage failure");
        }

        static Dictionary<string, int> Words(int value) => new Dictionary<string, int>(StringComparer.Ordinal)
        {
            { "GREETING", value }, { "LOVE", value }, { "THANK_YOU", value },
        };
    }
}
